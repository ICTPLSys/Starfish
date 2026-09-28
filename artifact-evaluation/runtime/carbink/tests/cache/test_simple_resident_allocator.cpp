// Ported from nq-resident-six r2/t8; assertions intentionally retained.
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cache/region_based_allocator.hpp"
#include "cache/placement/simple_region_heat.hpp"
#include "rdma/config.hpp"
#include "utils/control.hpp"

namespace allocator = FarLib::allocator;
namespace heat = FarLib::simple_region_heat;

namespace {

constexpr std::size_t kBin = 7;
constexpr std::size_t kRegionCount = 8;

bool supply_band_matches(const allocator::RegionHead *region) {
    const auto placement = region->load_placement();
    const auto klass = heat::allocation_class_for(
        reinterpret_cast<std::uintptr_t>(region));
    if (placement == allocator::RegionPlacement::Resident)
        return klass >= 3 && klass < 6;
    if (placement == allocator::RegionPlacement::Streaming)
        return klass < 3;
    return false;
}

void retire(allocator::BlockHead *block) {
    block->obj_meta_data.store(FarLib::cache::far_obj_t::null(),
                               std::memory_order_release);
    allocator::block_to_region(block)->pending_reclaims.fetch_add(
        1, std::memory_order_relaxed);
}

}  // namespace

int main() {
    // Keep this executable self-contained when run outside CTest. These are
    // intentionally set before any mode helper is first evaluated.
    setenv("FARLIB_SIMPLE_HOTCOLD", "1", 1);
    setenv("FARLIB_SIMPLE_REGION_HEAT", "1", 1);
    setenv("FARLIB_SIMPLE_LOCAL_RESIDENT", "1", 1);
    setenv("FARLIB_SIMPLE_LOCAL_ROUTING", "1", 1);
    setenv("FARLIB_SIMPLE_REMOTE_HOTCOLD", "1", 1);
    setenv("FARLIB_SIMPLE_SIX_GROUPS", "1", 1);
    setenv("FARLIB_SIMPLE_REGION_BUDGET", "fixed", 1);
    setenv("FARLIB_LIST_ONLY_SIX", "1", 1);
    setenv("FARLIB_LIST_ONLY_SIX_GROUPS", "0", 1);
    setenv("FARLIB_FIXED_SIX_GROUPS", "0", 1);
    setenv("FARLIB_FIXED_SIX_POOLS", "0", 1);

    assert(heat::local_resident_enabled());
    const auto streaming_order = heat::local_fallback_order(0);
    const auto resident_order = heat::local_fallback_order(3);
    assert(streaming_order[0] == 0 && streaming_order[1] == 1 &&
           streaming_order[2] == 2);
    assert(resident_order[0] == 3 && resident_order[1] == 4 &&
           resident_order[2] == 5);
    for (std::uint8_t dirty = 0; dirty < 3; ++dirty) {
        assert(heat::class_for_placement(false, dirty) == dirty);
        assert(heat::class_for_placement(true, dirty) == dirty + 3);
    }

    auto &config = const_cast<FarLib::rdma::Configure &>(FarLib::get_config());
    config.enable_region_resident_placement = true;
    config.region_placement_bind_groups = false;
    config.exclusive_cache = true;
    config.enable_selective_backup = false;
    config.local_resident_budget_bytes = 3 * allocator::RegionSize;

    const std::size_t heap_bytes = kRegionCount * allocator::RegionSize;
    void *memory = std::aligned_alloc(allocator::RegionSize, heap_bytes);
    assert(memory != nullptr);
    allocator::global_heap.register_heap(memory, heap_bytes);
    heat::classes().configure(reinterpret_cast<std::uintptr_t>(memory),
                              heap_bytes, allocator::RegionSize);

    std::vector<allocator::RegionHead *> regions;
    std::vector<allocator::BlockHead *> blocks;
    regions.reserve(4);
    blocks.reserve(4);
    for (std::uint32_t dirty = 0; dirty < 4; ++dirty) {
        const auto *region = allocator::global_heap.allocate_region(
            kBin, allocator::RegionPlacement::Resident, 0,
            static_cast<std::uint32_t>(3 + (dirty % 3)));
        assert(region != nullptr);
        auto *mutable_region = const_cast<allocator::RegionHead *>(region);
        assert(mutable_region->load_placement() ==
               (dirty < 3 ? allocator::RegionPlacement::Resident
                          : allocator::RegionPlacement::Streaming));
        assert(supply_band_matches(mutable_region));
        auto object = FarLib::cache::far_obj_t{
            .size = 8, .obj_id = static_cast<obj_id_t>(0x100 + dirty)};
        auto *block = mutable_region->allocate(object);
        assert(block != nullptr);
        mutable_region->state.store(allocator::IN_USE,
                                    std::memory_order_relaxed);
        regions.push_back(mutable_region);
        blocks.push_back(block);
    }
    assert(allocator::global_heap.get_resident_reserved_regions() == 3);

    // Keep partially-used Regions private until all four claims have been
    // made. Otherwise a public Region would correctly satisfy the next
    // same-band request before a new Region is needed.
    for (auto *region : regions) {
        allocator::global_heap.return_back_region(region);
        assert(region->owner_list.load(std::memory_order_acquire) != nullptr);
    }

    // Free one Resident Region to leave budget headroom while retaining two
    // public same-band donors. The next miss must release its tentative
    // reservation when it consumes a donor instead of leaking the count.
    retire(blocks[2]);
    allocator::global_heap.reclaim_all_deallocated_regions();
    assert(allocator::global_heap.get_resident_reserved_regions() == 2);

    // A resident-class miss must consume the public same-band alternative
    // before fresh supply; it must not reserve a third Resident Region.
    auto *fallback = allocator::global_heap.allocate_region(
        kBin, allocator::RegionPlacement::Resident, 0, 5);
    assert(fallback != nullptr);
    assert(fallback->load_placement() == allocator::RegionPlacement::Resident);
    assert(heat::allocation_class_for(
               reinterpret_cast<std::uintptr_t>(fallback)) >= 3);
    assert(allocator::global_heap.get_resident_reserved_regions() == 2);
    fallback->state.store(allocator::IN_USE, std::memory_order_relaxed);
    allocator::global_heap.return_back_region(fallback);

    // Empty/recycled Regions unregister their supply ledger before entering
    // FREE, so a subsequent claim starts with a clean placement band.
    for (std::size_t i = 0; i < blocks.size(); ++i)
        if (i != 2) retire(blocks[i]);
    allocator::global_heap.reclaim_all_deallocated_regions();
    allocator::global_heap.reclaim_all_deallocated_regions();
    allocator::global_heap.verify_no_detached_regions_for_shutdown();
    allocator::global_heap.destroy();
    std::free(memory);
    std::cout << "PASS simple resident allocator placement-band routing\n";
}
