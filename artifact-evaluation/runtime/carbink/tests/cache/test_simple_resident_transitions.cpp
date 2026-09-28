// Ported from nq-resident-six r2/t8; assertions intentionally retained.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "cache/entry.hpp"
#include "cache/region_based_allocator.hpp"
#include "cache/placement/simple_region_heat.hpp"
#include "rdma/config.hpp"
#include "utils/control.hpp"

namespace allocator = FarLib::allocator;
namespace heat = FarLib::simple_region_heat;
using FarLib::cache::EntryStateBits;
using FarLib::cache::FarObjectEntry;
using FarLib::cache::far_obj_t;

namespace {

constexpr std::size_t kRegionCount = 8;
constexpr std::size_t kBin = allocator::RegionBinCount - 8;
constexpr std::size_t kPayloadBytes = 64;

struct ObjectBlock {
    std::unique_ptr<FarObjectEntry> entry;
    allocator::BlockHead *block = nullptr;
};

ObjectBlock add_object(allocator::RegionHead *region, bool resident,
                       std::uint64_t id, std::size_t bytes = kPayloadBytes) {
    ObjectBlock object;
    object.entry = std::make_unique<FarObjectEntry>();
    object.entry->reset<true>(nullptr, 0x1000 + id, bytes, false, resident);
    object.entry->set_remote_backup_reservation(true);
    assert(object.entry->has_remote());
    const far_obj_t token{
        .size = static_cast<std::uint16_t>(bytes),
        .obj_id = static_cast<obj_id_t>(
            reinterpret_cast<std::uintptr_t>(object.entry.get())),
    };
    object.block = region->allocate(token);
    assert(object.block != nullptr);
    object.entry->set_local_addr(object.block + 1);
    return object;
}

void publish(allocator::RegionHead *region) {
    region->state.store(allocator::IN_USE, std::memory_order_relaxed);
    allocator::global_heap.return_back_region(region);
    assert(region->owner_list.load(std::memory_order_acquire) != nullptr);
}

void retire(const ObjectBlock &object) {
    object.block->obj_meta_data.store(far_obj_t::null(),
                                      std::memory_order_release);
    allocator::block_to_region(object.block)->pending_reclaims.fetch_add(
        1, std::memory_order_relaxed);
}

struct TransitionAudit {
    std::size_t calls = 0;
    bool callback_flags_match = true;
    bool backup_preserved = true;
    bool size_match = true;
    allocator::RegionPlacement from = allocator::RegionPlacement::Unclassified;
    allocator::RegionPlacement to = allocator::RegionPlacement::Unclassified;

    void observe(FarObjectEntry &entry, std::size_t bytes,
                 allocator::RegionPlacement source,
                 allocator::RegionPlacement target) {
        ++calls;
        from = source;
        to = target;
        callback_flags_match &=
            entry.is_resident_local() ==
            (source == allocator::RegionPlacement::Resident);
        backup_preserved &= entry.has_remote_backup_reservation();
        size_match &= bytes == kPayloadBytes;
    }
};

void check_supply_band(allocator::RegionHead *region,
                       std::uint8_t expected_dirty) {
    const auto placement = region->load_placement();
    const auto supply = heat::allocation_class_for(
        reinterpret_cast<std::uintptr_t>(region));
    assert((placement == allocator::RegionPlacement::Resident && supply >= 3 &&
            supply < 6) ||
           (placement == allocator::RegionPlacement::Streaming && supply < 3));
    assert(static_cast<std::uint8_t>(supply % 3) == expected_dirty);
}

}  // namespace

int main() {
    setenv("FARLIB_SIMPLE_LOCAL_RESIDENT", "1", 1);
    setenv("FARLIB_SIMPLE_HOTCOLD", "1", 1);
    setenv("FARLIB_SIMPLE_REGION_HEAT", "1", 1);
    setenv("FARLIB_SIMPLE_LOCAL_ROUTING", "1", 1);
    setenv("FARLIB_SIMPLE_REMOTE_HOTCOLD", "1", 1);
    setenv("FARLIB_SIMPLE_SIX_GROUPS", "1", 1);
    setenv("FARLIB_SIMPLE_REGION_BUDGET", "fixed", 1);
    setenv("FARLIB_LIST_ONLY_SIX", "1", 1);
    setenv("FARLIB_LIST_ONLY_SIX_GROUPS", "0", 1);
    setenv("FARLIB_FIXED_SIX_GROUPS", "0", 1);
    setenv("FARLIB_FIXED_SIX_POOLS", "0", 1);

    auto &config = const_cast<FarLib::rdma::Configure &>(FarLib::get_config());
    config.enable_region_resident_placement = true;
    config.region_placement_bind_groups = false;
    config.exclusive_cache = true;
    config.enable_selective_backup = true;
    config.local_resident_budget_bytes = 4 * allocator::RegionSize;

    const std::size_t heap_bytes = kRegionCount * allocator::RegionSize;
    void *memory = std::aligned_alloc(allocator::RegionSize, heap_bytes);
    assert(memory != nullptr);
    allocator::global_heap.register_heap(memory, heap_bytes);
    heat::classes().configure(reinterpret_cast<std::uintptr_t>(memory),
                              heap_bytes, allocator::RegionSize);
    allocator::disable_region_list_fibre_yield_for_current_thread();

    // One public USABLE Resident Region and one public FULL Streaming Region
    // use real FarObjectEntry pointers in their block metadata. The remote
    // address/reservation are synthetic metadata for this CPU-only test.
    // The callback below is an observer, not the cache's real backup-release
    // callback: this verifies flag isolation, not RDMA backup reclamation.
    auto *usable = allocator::global_heap.allocate_region(
        kBin, allocator::RegionPlacement::Resident, 0, 3);
    assert(usable != nullptr &&
           usable->load_placement() == allocator::RegionPlacement::Resident);
    const std::uint8_t usable_dirty = static_cast<std::uint8_t>(
        heat::allocation_class_for(reinterpret_cast<std::uintptr_t>(usable)) % 3);
    ObjectBlock usable_object = add_object(usable, true, 1);
    publish(usable);

    auto *full = allocator::global_heap.allocate_region(
        kBin, allocator::RegionPlacement::Streaming, 0, 1);
    assert(full != nullptr &&
           full->load_placement() == allocator::RegionPlacement::Streaming);
    const std::uint8_t full_dirty = static_cast<std::uint8_t>(
        heat::allocation_class_for(reinterpret_cast<std::uintptr_t>(full)) % 3);
    std::vector<ObjectBlock> full_objects;
    full_objects.reserve((allocator::RegionSize - sizeof(allocator::RegionHead)) /
                         allocator::get_bin_size(kBin));
    std::uint64_t object_id = 2;
    while (full->can_allocate()) {
        full_objects.push_back(add_object(full, false, object_id++));
    }
    assert(!full->can_allocate());
    publish(full);
    assert(full->state.load(std::memory_order_acquire) == allocator::FULL);

    const auto usable_epoch = usable->load_placement_epoch();
    TransitionAudit demote_audit;
    const auto demoted = allocator::global_heap.try_reclassify_hotness_region(
        usable, usable_epoch, allocator::RegionPlacement::Resident,
        allocator::RegionPlacement::Streaming,
        [&](FarObjectEntry &entry, std::size_t bytes,
            allocator::RegionPlacement from,
            allocator::RegionPlacement to) {
            demote_audit.observe(entry, bytes, from, to);
        });
    assert(demoted.changed && demoted.transitionable);
    assert(usable->state.load(std::memory_order_acquire) == allocator::USABLE);
    assert(usable->load_placement() == allocator::RegionPlacement::Streaming);
    assert(!usable_object.entry->is_resident_local());
    assert(usable_object.entry->has_remote_backup_reservation());
    assert(demote_audit.calls == 1 && demote_audit.callback_flags_match &&
           demote_audit.backup_preserved && demote_audit.size_match);
    check_supply_band(usable, usable_dirty);

    const auto full_epoch = full->load_placement_epoch();
    TransitionAudit promote_audit;
    const auto promoted = allocator::global_heap.try_reclassify_hotness_region(
        full, full_epoch, allocator::RegionPlacement::Streaming,
        allocator::RegionPlacement::Resident,
        [&](FarObjectEntry &entry, std::size_t bytes,
            allocator::RegionPlacement from,
            allocator::RegionPlacement to) {
            promote_audit.observe(entry, bytes, from, to);
        });
    assert(promoted.changed && promoted.transitionable);
    assert(full->state.load(std::memory_order_acquire) == allocator::FULL);
    assert(full->load_placement() == allocator::RegionPlacement::Resident);
    assert(promote_audit.calls == full_objects.size());
    assert(promote_audit.callback_flags_match && promote_audit.backup_preserved &&
           promote_audit.size_match);
    for (const auto &object : full_objects) {
        assert(object.entry->is_resident_local());
        assert(object.entry->has_remote_backup_reservation());
    }
    check_supply_band(full, full_dirty);

    // Exchange the same public USABLE/FULL pair back. This exercises the
    // two-region lock ordering and verifies both placement lists are restored
    // with the original state kind and dirty class modulo three.
    const auto hot_epoch = usable->load_placement_epoch();
    const auto cold_epoch = full->load_placement_epoch();
    TransitionAudit exchange_audit;
    const auto exchanged = allocator::global_heap.try_exchange_hotness_regions(
        usable, hot_epoch, full, cold_epoch,
        [&](FarObjectEntry &entry, std::size_t bytes,
            allocator::RegionPlacement from,
            allocator::RegionPlacement to) {
            exchange_audit.observe(entry, bytes, from, to);
        });
    assert(exchanged.changed);
    assert(exchanged.promoted_bytes == allocator::RegionSize);
    assert(exchanged.demoted_bytes == allocator::RegionSize);
    assert(usable->load_placement() == allocator::RegionPlacement::Resident);
    assert(full->load_placement() == allocator::RegionPlacement::Streaming);
    assert(usable->state.load(std::memory_order_acquire) == allocator::USABLE);
    assert(full->state.load(std::memory_order_acquire) == allocator::FULL);
    assert(exchange_audit.calls == full_objects.size() + 1);
    assert(exchange_audit.callback_flags_match && exchange_audit.backup_preserved &&
           exchange_audit.size_match);
    assert(usable_object.entry->is_resident_local());
    for (const auto &object : full_objects)
        assert(!object.entry->is_resident_local());
    check_supply_band(usable, usable_dirty);
    check_supply_band(full, full_dirty);
    const auto counts = FarLib::simple_region_budget::local().snapshot(kBin);
    assert(counts.actual[0] + counts.actual[1] + counts.actual[2] ==
           allocator::global_heap.get_streaming_reserved_regions());
    assert(counts.actual[3] + counts.actual[4] + counts.actual[5] ==
           allocator::global_heap.get_resident_reserved_regions());

    retire(usable_object);
    for (const auto &object : full_objects) retire(object);
    allocator::global_heap.reclaim_all_deallocated_regions();
    allocator::global_heap.verify_no_detached_regions_for_shutdown();
    allocator::global_heap.destroy();
    std::free(memory);
    std::cout << "PASS simple resident public transition/exchange state, "
                 "entry flags, synthetic backup-bit isolation, and supply bands\n";
}
