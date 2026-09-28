#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cache/region_based_allocator.hpp"
#include "cache/alloc/sharded_remote_usage.hpp"
#include "cache/placement/fixed_six_runtime.hpp"
#include "cache/placement/simple_region_budget.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/cpu_cycles.hpp"
#include "utils/stats.hpp"
#include "utils/uthreads.hpp"

namespace FarLib {

namespace allocator {

namespace remote {
class RemoteThreadHeap;
class RemoteRegionList;
inline void flush_all_registered_thread_heaps();

inline std::mutex &remote_thread_heap_registry_mutex() {
    static std::mutex mtx;
    return mtx;
}

inline std::vector<RemoteThreadHeap *> &remote_thread_heap_registry() {
    static std::vector<RemoteThreadHeap *> heaps;
    return heaps;
}

constexpr size_t RegionSize = 512 * 1024;
using FarLib::allocator::BinSize;
using FarLib::allocator::RegionBinCount;

using FarLib::allocator::bin_from_wsize;
using FarLib::allocator::get_bin_size;
using FarLib::allocator::wsize_from_size;

constexpr uint64_t InvalidRemoteAddr = std::numeric_limits<uint64_t>::max();
// definition is the same as RegionHead
// using bitmap instead to manage blocks
// when a region is being operated, it is occupied by only 1 thread
// so we dont need concurrent programming for those operation

class DoubleLinkedListHead {
protected:
    DoubleLinkedListHead *next;
    DoubleLinkedListHead *prev;

public:
    DoubleLinkedListHead() : next(nullptr), prev(nullptr) {}
    void insert_front_unsafe(DoubleLinkedListHead *node) {
        assert(node->next == nullptr && node->prev == nullptr);
        node->next = this;
        node->prev = this->prev;
        this->prev->next = node;
        this->prev = node;
    }

    void insert_back_unsafe(DoubleLinkedListHead *node) {
        assert(node->next == nullptr && node->prev == nullptr);
        node->prev = this;
        node->next = this->next;
        this->next->prev = node;
        this->next = node;
    }

    void remove_self_unsafe() {
        // check if region is "owned" by a thread
        assert(this->prev != nullptr && this->next != nullptr);
        this->prev->next = this->next;
        this->next->prev = this->prev;
        this->prev = nullptr;
        this->next = nullptr;
    }

    void reset() {
        this->next = nullptr;
        this->prev = nullptr;
    }

    friend class RemoteRegionList;
};

class RemoteRegionHead : public DoubleLinkedListHead {
private:
    std::atomic_flag flag;
    uint32_t bin;
    uint32_t endpoint_idx = std::numeric_limits<uint32_t>::max();
    uint32_t last_map_idx;
    // 0 for allocated, 1 for free
    uint64_t *blockmap;
    size_t entry_size;
    uint64_t base_addr;
    uint64_t used_count;
    std::atomic<RemoteRegionList *> list_owner{nullptr};
    RemoteRegionHead *group_prev{nullptr};
    RemoteRegionHead *group_next{nullptr};
    uint32_t indexed_group_id{0};
    static constexpr size_t MapElementBitCount = sizeof(uint64_t) * 8;
    static constexpr uint64_t FreeMap = std::numeric_limits<uint64_t>::max();
    static constexpr uint32_t FreeMap32 = std::numeric_limits<uint32_t>::max();
    static constexpr uint64_t FullMap = 0;

public:
    std::atomic<six_group::Record *> six_record{nullptr};
    // The simple hot/cold mode is deliberately independent of the fixed-six
    // registry.  Keep the class on the physical remote Region so it remains
    // valid while the descriptor moves between a private heap slot and a
    // global usable/full/free list.
    std::atomic<uint8_t> simple_heat_class{
        simple_region_heat::kCold};

    uint8_t simple_class() const {
        return simple_region_heat::normalize_class(
            simple_heat_class.load(std::memory_order_acquire));
    }

    void set_simple_class(uint8_t value) {
        simple_heat_class.store(
            simple_region_heat::normalize_class(value),
            std::memory_order_release);
    }

    void lock() {
        while (flag.test_and_set()) {
        }
    }

    bool try_lock() {
        return !flag.test_and_set(std::memory_order_acquire);
    }

    void unlock() { flag.clear(); }

    void init(uint64_t base_addr, uint32_t bin) {
        this->base_addr = base_addr;
        this->endpoint_idx = static_cast<uint32_t>(
            FarLib::get_config().map_remote_addr(base_addr).first);
        reset<true>(bin);
    }

    template <bool init = false>
    void reset(uint32_t bin) {
        assert(!flag.test());
        this->DoubleLinkedListHead::reset();
        this->entry_size = RegionSize / get_bin_size(bin);
        size_t map_size =
            (this->entry_size + MapElementBitCount - 1) / MapElementBitCount;
        if constexpr (init) {
            this->blockmap = static_cast<uint64_t *>(
                std::realloc(blockmap, map_size * sizeof(uint64_t)));
        } else if (this->bin != bin) {
            this->blockmap = static_cast<uint64_t *>(
                std::realloc(blockmap, map_size * sizeof(uint64_t)));
        }
        this->bin = bin;
        // only first several bit is valid in the last map
        // if entry size is not aligned to 64
        if (entry_size % MapElementBitCount == 0) {
            std::memset(blockmap, FreeMap32, map_size * sizeof(uint64_t));
        } else {
            std::memset(blockmap, FreeMap32, (map_size - 1) * sizeof(uint64_t));
            this->blockmap[map_size - 1] =
                (1UL << (MapElementBitCount -
                         (map_size * MapElementBitCount - entry_size))) -
                1;
        }
        last_map_idx = 0;
        used_count = 0;
        simple_heat_class.store(simple_region_heat::kCold,
                                std::memory_order_relaxed);
    }

    RemoteRegionHead() = default;

    RemoteRegionHead(const RemoteRegionHead &head) = delete;

    RemoteRegionHead(RemoteRegionHead &&head) = delete;

    ~RemoteRegionHead() {
        if (blockmap) {
            std::free(blockmap);
        }
    }

    uint64_t allocate_unsafe() {
        size_t map_size =
            (this->entry_size + MapElementBitCount - 1) / MapElementBitCount;
        for (size_t i = 0; i < map_size; i++) {
            // get the first 1 in the bitmap num
            int bit = __builtin_ffsl(blockmap[last_map_idx]);
            if (bit) {
                // set the bit at "bit" to 0, means this block is allocated
                blockmap[last_map_idx] ^= (1L << (bit - 1));
                used_count++;
                return base_addr +
                       (MapElementBitCount * last_map_idx + (bit - 1)) *
                           get_bin_size(bin);
            } else {
                last_map_idx++;
                if (last_map_idx >= map_size) [[unlikely]] {
                    last_map_idx = 0;
                }
            }
        }
        return InvalidRemoteAddr;
    }

    uint64_t allocate() {
        lock();
        uint64_t addr = allocate_unsafe();
        unlock();
        return addr;
    }

    // Returns true if this call actually freed an allocated slot.
    // Returns false if the slot was already free (double-free) or invalid.
    bool deallocate_unsafe(uint64_t addr) {
        assert(addr >= base_addr && addr < base_addr + RegionSize);
        uint64_t offset = (addr - base_addr) / get_bin_size(bin);
        // Only offsets in [0, entry_size) are valid; the tail bits in the last map
        // element may be 0 to mark them as unavailable.
        assert(offset < entry_size);

        size_t map_idx = offset / MapElementBitCount;
        uint64_t mask = (1ULL << (offset % MapElementBitCount));
        // bitmap: 0=allocated, 1=free
        bool was_allocated = ((blockmap[map_idx] & mask) == 0);
        if (!was_allocated) {
            return false;
        }

        // set 1 for free
        blockmap[map_idx] |= mask;
        used_count--;
        return true;
    }

    bool is_free_unsafe() const { return used_count == 0; }

    bool is_full_unsafe() const { return used_count == entry_size; }

    bool is_full_just_now_unsafe() const {
        return used_count == entry_size - 1;
    }

    uint32_t get_bin() const { return bin; }

    uint32_t get_endpoint_idx() const { return endpoint_idx; }

    uint64_t get_used_count() const { return used_count; }

    // The budget sampler only calls this while holding the Region lock.  It
    // deliberately reports the free capacity of a public usable Region, not
    // the capacity of private or full descriptors.
    uint64_t get_free_bytes_unsafe() const {
        return (entry_size - used_count) * get_bin_size(bin);
    }

    uint32_t six_group_id() const {
        return six_group::group_of(
            six_record.load(std::memory_order_acquire));
    }

    void ensure_six_group(uint32_t preferred = 0);

    void reindex_six_group();

    bool is_in_thread_heap_unsafe() {
        assert(!((next == nullptr) ^ (prev == nullptr)));
        return next == nullptr && prev == nullptr;
    }

    void remove_self() {
        lock();
        remove_self_unsafe();
        unlock();
    }

    friend class RemoteRegionList;
};

class RemoteRegionList {
private:
    struct GroupChain {
        RemoteRegionHead *head{nullptr};
        RemoteRegionHead *tail{nullptr};
    };

    std::atomic_flag flag;
    DoubleLinkedListHead dummy_head;
    DoubleLinkedListHead dummy_tail;
    bool group_index_enabled{false};
    std::unordered_map<uint32_t, GroupChain> group_index;
    // List-only six-way ablation.  The parent is a wrapper and child zero is
    // the sole normal allocation route; all six children remain traversable.
    RemoteRegionList *list_only_six_parent{nullptr};
    RemoteRegionList *list_only_six_children{nullptr};
    // Background budget actuation rotates through each public usable list.
    // The cursor is protected by this list's lock and is advanced after every
    // bounded scan so an ineligible prefix cannot starve later donors.
    DoubleLinkedListHead *budget_scan_cursor{nullptr};

    void lock() { while (flag.test_and_set()); }

    void unlock() { flag.clear(); }

    void insert_group_unsafe(RemoteRegionHead *region) {
        region->list_owner.store(this, std::memory_order_release);
        if (list_only_six_parent != nullptr) {
            region->group_prev = nullptr;
            region->group_next = nullptr;
            region->indexed_group_id = 0;
            return;
        }
        if (!group_index_enabled) return;
        const uint32_t group = region->six_group_id();
        if (group == 0) return;
        auto &chain = group_index[group];
        region->indexed_group_id = group;
        region->group_prev = chain.tail;
        region->group_next = nullptr;
        if (chain.tail != nullptr) chain.tail->group_next = region;
        else chain.head = region;
        chain.tail = region;
    }

    void remove_group_unsafe(RemoteRegionHead *region) {
        if (list_only_six_parent != nullptr) {
            region->group_prev = nullptr;
            region->group_next = nullptr;
            region->indexed_group_id = 0;
            region->list_owner.store(nullptr, std::memory_order_release);
            return;
        }
        if (region->indexed_group_id != 0) {
            auto found = group_index.find(region->indexed_group_id);
            assert(found != group_index.end());
            auto &chain = found->second;
            if (region->group_prev != nullptr)
                region->group_prev->group_next = region->group_next;
            else
                chain.head = region->group_next;
            if (region->group_next != nullptr)
                region->group_next->group_prev = region->group_prev;
            else
                chain.tail = region->group_prev;
        }
        region->group_prev = nullptr;
        region->group_next = nullptr;
        region->indexed_group_id = 0;
        region->list_owner.store(nullptr, std::memory_order_release);
    }

    void detach_unsafe(RemoteRegionHead *region) {
        if (budget_scan_cursor == region) {
            budget_scan_cursor = region->next == &dummy_tail
                                     ? &dummy_head
                                     : region->next;
        }
        remove_group_unsafe(region);
        region->remove_self_unsafe();
    }

public:
    RemoteRegionList() {
        dummy_head.next = &dummy_tail;
        dummy_tail.prev = &dummy_head;
        budget_scan_cursor = &dummy_head;
    }

    void enable_group_index() {
        lock();
        assert(empty());
        group_index.reserve(64);
        group_index_enabled = true;
        unlock();
    }

    void configure_list_only_six_child(RemoteRegionList *parent) {
        assert(parent != nullptr);
        lock();
        assert(empty());
        assert(!group_index_enabled);
        list_only_six_parent = parent;
        unlock();
    }

    void configure_list_only_six_children(RemoteRegionList *children) {
        assert(children != nullptr);
        lock();
        assert(empty());
        assert(list_only_six_parent == nullptr);
        list_only_six_children = children;
        unlock();
    }

    bool is_list_only_six_child() const {
        return list_only_six_parent != nullptr;
    }

    RemoteRegionList *list_only_six_child(size_t slot) const {
        if (list_only_six_children == nullptr ||
            slot >= six_group::list_only_six::kChildCount)
            return nullptr;
        return &list_only_six_children[slot];
    }

    void insert_tail(RemoteRegionHead *node) {
        if (list_only_six_children) {
            const size_t child = simple_region_heat::remote_grouping_enabled()
                                     ? simple_region_heat::normalize_class(
                                           node->simple_class())
                                     : 0;
            list_only_six_children[child].insert_tail(node);
            return;
        }
        lock();
        dummy_tail.insert_front_unsafe(node);
        insert_group_unsafe(node);
        unlock();
    }

    void insert_head(RemoteRegionHead *node) {
        if (list_only_six_children) {
            const size_t child = simple_region_heat::remote_grouping_enabled()
                                     ? simple_region_heat::normalize_class(
                                           node->simple_class())
                                     : 0;
            list_only_six_children[child].insert_head(node);
            return;
        }
        lock();
        dummy_head.insert_back_unsafe(node);
        insert_group_unsafe(node);
        unlock();
    }

    inline bool empty() const {
        if (list_only_six_children) {
            for (size_t i = 0; i < six_group::list_only_six::kChildCount;
                 ++i) {
                if (!list_only_six_children[i].empty()) return false;
            }
            return true;
        }
        assert(!((dummy_head.next == &dummy_tail) ^
                 (dummy_tail.prev == &dummy_head)));
        return dummy_head.next == &dummy_tail;
    }

    RemoteRegionHead *pop_head() {
        if (list_only_six_children) {
            if (simple_region_heat::remote_grouping_enabled()) {
                const auto order = simple_region_heat::fallback_order(
                    simple_region_heat::kCold);
                for (size_t rank = 0;
                     rank < simple_region_heat::class_count(); ++rank)
                    if (auto *region =
                            list_only_six_children[order[rank]].pop_head())
                        return region;
                return nullptr;
            }
            return list_only_six_children[0].pop_head();
        }
        for (;;) {
            lock();
            if (empty()) {
                unlock();
                return nullptr;
            }
            RemoteRegionHead *region =
                static_cast<RemoteRegionHead *>(dummy_head.next);
            // Deallocation holds region->list. Never block on the Region
            // while holding the inverse list->region order.
            if (!region->try_lock()) {
                unlock();
                std::this_thread::yield();
                continue;
            }
            detach_unsafe(region);
            unlock();
            region->unlock();
            return region;
        }
    }

    RemoteRegionHead *pop_tail() {
        if (list_only_six_children) {
            if (simple_region_heat::remote_grouping_enabled()) {
                const auto order = simple_region_heat::fallback_order(
                    simple_region_heat::kCold);
                for (size_t rank = 0;
                     rank < simple_region_heat::class_count(); ++rank)
                    if (auto *region =
                            list_only_six_children[order[rank]].pop_tail())
                        return region;
                return nullptr;
            }
            return list_only_six_children[0].pop_tail();
        }
        for (;;) {
            lock();
            if (empty()) {
                unlock();
                return nullptr;
            }
            RemoteRegionHead *region =
                static_cast<RemoteRegionHead *>(dummy_tail.prev);
            if (!region->try_lock()) {
                unlock();
                std::this_thread::yield();
                continue;
            }
            detach_unsafe(region);
            unlock();
            region->unlock();
            return region;
        }
    }

    RemoteRegionHead *pop_group(uint32_t group) {
        if (list_only_six_children) {
            if (simple_region_heat::remote_grouping_enabled()) {
                return list_only_six_children[
                           simple_region_heat::normalize_class(group)]
                    .pop_head();
            }
            return list_only_six_children[0].pop_group(group);
        }
        for (;;) {
            lock();
            if (!group_index_enabled || group == 0) {
                unlock();
                return nullptr;
            }
            auto found = group_index.find(group);
            RemoteRegionHead *region =
                found == group_index.end() ? nullptr : found->second.head;
            if (region == nullptr) {
                unlock();
                return nullptr;
            }
            if (!region->try_lock()) {
                unlock();
                std::this_thread::yield();
                continue;
            }
            detach_unsafe(region);
            unlock();
            region->unlock();
            return region;
        }
    }

    bool reindex_if_owned(RemoteRegionHead *region) {
        if (list_only_six_children) return false;
        lock();
        if (region->list_owner.load(std::memory_order_acquire) != this) {
            unlock();
            return false;
        }
        remove_group_unsafe(region);
        insert_group_unsafe(region);
        unlock();
        return true;
    }

    // Collect a conservative free-byte snapshot of this public list.  The
    // list lock is acquired before the non-blocking Region lock so a
    // deallocator holding Region -> list can never deadlock the sampler.  A
    // busy Region is skipped; omitting capacity is conservative for the
    // adaptive controller.
    void snapshot_free_bytes(
        simple_region_budget::Supply::value_type &free_bytes) {
        if (list_only_six_children != nullptr) {
            for (size_t slot = 0;
                 slot < six_group::list_only_six::kChildCount; ++slot) {
                list_only_six_children[slot].snapshot_free_bytes(free_bytes);
            }
            return;
        }

        lock();
        auto *node = dummy_head.next;
        while (node != &dummy_tail) {
            auto *next = node->next;
            auto *region = static_cast<RemoteRegionHead *>(node);
            if (region->try_lock()) {
                const size_t slot =
                    simple_region_budget::index(region->simple_class());
                free_bytes[slot] += region->get_free_bytes_unsafe();
                region->unlock();
            }
            node = next;
        }
        unlock();
    }

    // Detach one matching public usable Region for background budget
    // actuation.  The returned Region remains locked: the caller must invoke
    // reassign_public() and republish it before unlocking.  The list lock is
    // held only around the non-blocking Region lock/detach sequence, preserving
    // the normal deallocator order (Region -> list) without deadlock.
    RemoteRegionHead *detach_public_budget_region(
        size_t bin, uint8_t expected_class, uint64_t &remaining,
        size_t &scan_budget, size_t &scanned, size_t &busy) {
        if (list_only_six_children != nullptr || remaining == 0 ||
            scan_budget == 0) {
            return nullptr;
        }

        lock();
        auto *node = budget_scan_cursor;
        if (node == nullptr || node == &dummy_head || node == &dummy_tail) {
            node = dummy_head.next;
        }
        while (node != &dummy_tail && scan_budget != 0) {
            auto *next = node->next;
            ++scanned;
            --scan_budget;
            auto *region = static_cast<RemoteRegionHead *>(node);
            if (!region->try_lock()) {
                ++busy;
                node = next;
                continue;
            }

            // A public usable list must not contain a full Region, but keep
            // this guard explicit so full/private descriptors cannot become
            // actuation donors if a list-state race or future path violates
            // that invariant.
            if (region->get_bin() == bin &&
                simple_region_heat::normalize_class(region->simple_class()) ==
                    simple_region_heat::normalize_class(expected_class) &&
                !region->is_full_unsafe()) {
                --remaining;
                detach_unsafe(region);
                budget_scan_cursor =
                    next == &dummy_tail ? &dummy_head : next;
                unlock();
                // Keep the Region lock held through controller reclassification
                // and usable-child publication.
                return region;
            }
            region->unlock();
            node = next;
        }
        budget_scan_cursor = node == &dummy_tail ? &dummy_head : node;
        unlock();
        return nullptr;
    }

    void remove_list_safe_region_unsafe(RemoteRegionHead *region) {
        if (list_only_six_children) {
            auto *owner = region->list_owner.load(std::memory_order_acquire);
            if (owner != nullptr && owner != this) {
                owner->remove_list_safe_region_unsafe(region);
            }
            return;
        }
        lock();
        detach_unsafe(region);
        unlock();
    }
};

inline void RemoteRegionHead::ensure_six_group(uint32_t preferred) {
    if (!six_group::enabled() || six_group::list_only_six::enabled()) return;
    auto *record = six_record.load(std::memory_order_acquire);
    if (record == nullptr) {
        record = six_group::registry().register_region(
            reinterpret_cast<uintptr_t>(this), true, bin, preferred,
            [](six_group::Record *published, uint32_t) {
                auto *region = reinterpret_cast<RemoteRegionHead *>(
                    published->token.address);
                region->reindex_six_group();
            });
        six_record.store(record, std::memory_order_release);
        return;
    }
    // Callers own this detached descriptor and have reset it to empty.
    assert(used_count == 0);
    six_group::registry().rebind_empty(record, bin, preferred);
}

inline void RemoteRegionHead::reindex_six_group() {
    if (six_group::list_only_six::enabled()) return;
    RemoteRegionList *owner = list_owner.load(std::memory_order_acquire);
    if (owner != nullptr) owner->reindex_if_owned(this);
}

class RemoteGlobalHeap {
private:
    RemoteRegionList usable_region_list[RegionBinCount];
    std::unique_ptr<RemoteRegionList[]> list_only_six_children;
    RemoteRegionList full_region_list;
    RemoteRegionList free_region_list;
    // Fresh regions belonging to the configured standby are parked here while
    // the standby is connected but intentionally allocation-ineligible.  They
    // become candidates immediately after the first endpoint failure.
    RemoteRegionList standby_waiting_region_list[RegionBinCount];
    std::unique_ptr<RemoteRegionHead[]> regions;
    size_t regions_size;
    // One process-wide endpoint eligibility policy is shared by the ordinary
    // flat allocator and the EC stripe manager.  Regions retain their own
    // endpoint metadata, so a dead region can leave the allocation lists while
    // its live bitmap/objects remain valid for later deallocation.
    size_t endpoint_count_ = 0;
    int standby_endpoint_ = -1;
    bool endpoint_liveness_enabled_ = false;
    std::atomic<bool> endpoint_failure_seen_{false};
    std::unique_ptr<std::atomic<uint8_t>[]> endpoint_alive_;
    // Standby observability is dormant unless an ec_batch standby is
    // configured.  Counts cover actual object/shard allocations, not startup
    // registration or connection setup.
    std::atomic<uint64_t> standby_allocation_count_{0};
    std::atomic<uint64_t> standby_flat_allocation_count_{0};
    std::atomic<uint64_t> standby_ec_allocation_count_{0};
    std::atomic<bool> standby_first_allocation_logged_{false};
    std::atomic_size_t used_heap_idx;
    std::atomic<uint64_t> used_bytes{0};
    size_t total_capacity_bytes{0};
    // Rotating starts keep pending donor bins/classes from starving behind a
    // repeatedly busy or ineligible public list prefix.
    size_t budget_scan_bin_cursor{0};
    size_t budget_scan_class_cursor{0};
    std::unique_ptr<std::atomic<uint64_t>[]> server_used_bytes;
    std::unique_ptr<ShardedRemoteUsage> sharded_usage;

#ifdef FARLIB_ALLOC_DEBUG
    std::atomic<uint64_t> dbg_double_free_detected{0};
    std::atomic<uint64_t> dbg_invalid_free_detected{0};
#endif

    // --- EC shard reservation (fault-tolerance path) ------------------------
    // An EC shard is one FarLib::allocator::RegionSize (256 KiB) extent, i.e.
    // two shard units share one remote region (remote::RegionSize == 512 KiB).
    // The stripe allocator claims whole regions exclusively and the regular bin
    // allocator skips them, so a shard can never overlap a bin allocation.
    // Every member below stays empty while ft_method == none (the default).
    static constexpr size_t EcShardUnitSize = FarLib::allocator::RegionSize;
    static constexpr size_t EcShardUnitsPerRegion = RegionSize / EcShardUnitSize;
    std::unique_ptr<std::atomic<uint8_t>[]> ec_region_claimed;  // per region
    std::unique_ptr<std::atomic<size_t>[]> ec_endpoint_cursor;  // per endpoint
    // A regular allocator region remains reserved for flat allocation for its
    // lifetime, even when its object bitmap is temporarily empty.  EC shard
    // reservation consults this marker to avoid claiming a parked standby
    // region (or a region with live flat objects) underneath the flat lists.
    std::unique_ptr<std::atomic<uint8_t>[]> regular_region_claimed;
    std::atomic<size_t> ec_region_count{0};
    std::vector<uint64_t> ec_free_units;  // shard bases released by stripes
    std::mutex ec_pool_mutex;

    // Cached null check: false (single predictable compare) when ft_method=none.
    bool region_reserved_for_ec(size_t region_idx) const {
        return ec_region_claimed != nullptr &&
               ec_region_claimed[region_idx].load(std::memory_order_relaxed) != 0;
    }

    RemoteRegionHead *addr_to_region(uint64_t addr) {
        return &regions[addr / RegionSize];
    }

public:
    RemoteGlobalHeap() {
        if (six_group::list_only_six::enabled()) {
            list_only_six_children = std::make_unique<RemoteRegionList[]>(
                RegionBinCount * six_group::list_only_six::kChildCount);
            for (size_t bin = 0; bin < RegionBinCount; ++bin) {
                auto *children =
                    list_only_six_children.get() +
                    bin * six_group::list_only_six::kChildCount;
                usable_region_list[bin].configure_list_only_six_children(
                    children);
                for (size_t slot = 0;
                     slot < six_group::list_only_six::kChildCount; ++slot) {
                    children[slot].configure_list_only_six_child(
                        &usable_region_list[bin]);
                }
            }
            std::cerr << "allocator.list_only_six enabled=1 fixed_class=" << (simple_region_heat::remote_grouping_enabled() ? -1 : 0)
                      << " group=0 quota=0" << std::endl;
        } else {
            for (auto &list : usable_region_list) list.enable_group_index();
        }
    }

    ~RemoteGlobalHeap() {
        std::cout << "used memory: " << used_heap_idx * RegionSize << std::endl;
        std::cout << "exact used bytes: " << get_used_bytes() << std::endl;
    }

    // Endpoint policy shared with SmallObjectStripeManager.  In non-ec_batch
    // mode the methods intentionally report the legacy all-endpoints policy.
    size_t endpoint_count() const { return endpoint_count_; }
    int standby_endpoint() const { return standby_endpoint_; }
    bool endpoint_liveness_enabled() const {
        return endpoint_liveness_enabled_;
    }

    bool endpoint_is_alive(size_t endpoint_idx) const {
        if (endpoint_idx >= endpoint_count_) return false;
        if (!endpoint_liveness_enabled_ || endpoint_alive_ == nullptr) {
            return true;
        }
        return endpoint_alive_[endpoint_idx].load(std::memory_order_acquire) !=
               0;
    }

    bool standby_active_for_allocation() const {
        return endpoint_liveness_enabled_ && standby_endpoint_ >= 0 &&
               static_cast<size_t>(standby_endpoint_) < endpoint_count_ &&
               endpoint_failure_seen_.load(std::memory_order_acquire) &&
               endpoint_is_alive(static_cast<size_t>(standby_endpoint_));
    }

    bool endpoint_is_eligible_for_allocation(size_t endpoint_idx) const {
        if (endpoint_idx >= endpoint_count_) return false;
        if (!endpoint_liveness_enabled_ || endpoint_alive_ == nullptr) {
            return true;
        }
        if (!endpoint_is_alive(endpoint_idx)) return false;
        if (!endpoint_failure_seen_.load(std::memory_order_acquire) &&
            standby_endpoint_ >= 0 &&
            endpoint_idx == static_cast<size_t>(standby_endpoint_)) {
            return false;
        }
        return true;
    }

    bool endpoint_waiting_for_activation(size_t endpoint_idx) const {
        return endpoint_liveness_enabled_ && standby_endpoint_ >= 0 &&
               static_cast<size_t>(standby_endpoint_) < endpoint_count_ &&
               !endpoint_failure_seen_.load(std::memory_order_acquire) &&
               endpoint_idx == static_cast<size_t>(standby_endpoint_) &&
               endpoint_is_alive(endpoint_idx);
    }

    // Idempotent process-wide failure transition.  Existing region metadata
    // and live objects are retained; allocation callers consult the eligibility
    // predicate before taking a fresh or cached region.
    bool mark_endpoint_dead(size_t endpoint_idx) {
        if (!endpoint_liveness_enabled_ || endpoint_alive_ == nullptr ||
            endpoint_idx >= endpoint_count_) {
            return false;
        }
        uint8_t expected = 1;
        if (!endpoint_alive_[endpoint_idx].compare_exchange_strong(
                expected, 0, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return false;
        }
        const uint64_t prior_allocations = standby_allocation_count();
        const uint64_t prior_flat_allocations = standby_flat_allocation_count();
        const uint64_t prior_ec_allocations = standby_ec_allocation_count();
        const bool first_failure = !endpoint_failure_seen_.exchange(
            true, std::memory_order_acq_rel);
        if (first_failure && standby_endpoint_ >= 0 &&
            endpoint_idx != static_cast<size_t>(standby_endpoint_) &&
            endpoint_is_alive(static_cast<size_t>(standby_endpoint_))) {
            std::ostringstream message;
            message << "INFO: ec_batch standby activated endpoint="
                    << standby_endpoint_ << " failed_endpoint="
                    << endpoint_idx << " prior_allocations="
                    << prior_allocations << " (flat="
                    << prior_flat_allocations << ", ec="
                    << prior_ec_allocations << ")";
            std::cerr << message.str() << std::endl;
        }
        return true;
    }

    size_t live_endpoint_count() const {
        size_t count = 0;
        for (size_t endpoint = 0; endpoint < endpoint_count_; endpoint++) {
            if (endpoint_is_alive(endpoint)) count++;
        }
        return count;
    }

    uint64_t standby_allocation_count() const {
        return standby_allocation_count_.load(std::memory_order_acquire);
    }

    uint64_t standby_flat_allocation_count() const {
        return standby_flat_allocation_count_.load(std::memory_order_acquire);
    }

    uint64_t standby_ec_allocation_count() const {
        return standby_ec_allocation_count_.load(std::memory_order_acquire);
    }

    void record_standby_allocation(size_t endpoint_idx, const char *path) {
        if (!endpoint_liveness_enabled_ || standby_endpoint_ < 0 ||
            endpoint_idx != static_cast<size_t>(standby_endpoint_)) {
            return;
        }
        const uint64_t count = standby_allocation_count_.fetch_add(
                                   1, std::memory_order_acq_rel) +
                               1;
        if (path != nullptr && std::strcmp(path, "flat") == 0) {
            standby_flat_allocation_count_.fetch_add(1,
                                                     std::memory_order_relaxed);
        } else {
            standby_ec_allocation_count_.fetch_add(1,
                                                   std::memory_order_relaxed);
        }
        if (!standby_first_allocation_logged_.exchange(
                true, std::memory_order_acq_rel)) {
            std::ostringstream message;
            message << "INFO: ec_batch standby first allocation endpoint="
                    << endpoint_idx << " path=" << path
                    << " count=" << count;
            std::cerr << message.str() << std::endl;
        }
    }

    void print_used_memory() {
        std::cout << "remote.committed_bytes: " << get_committed_bytes() << std::endl;
        std::cout << "remote.exact_allocated_bytes: " << get_used_bytes() << std::endl;
        if (server_used_bytes) {
            size_t server_count = FarLib::get_config().server_count;
            for (size_t i = 0; i < server_count; i++) {
                std::cout << "remote.server[" << i << "].used_bytes: "
                          << get_server_used_bytes(i) << std::endl;
            }
        }
        std::cout << "remote.capacity_bytes: " << get_capacity_bytes() << std::endl;
#ifdef FARLIB_ALLOC_DEBUG
        std::cout << "remote.dbg_double_free_detected: "
                  << dbg_double_free_detected.load(std::memory_order::relaxed)
                  << std::endl;
        std::cout << "remote.dbg_invalid_free_detected: "
                  << dbg_invalid_free_detected.load(std::memory_order::relaxed)
                  << std::endl;
#endif
    }

    uint64_t get_used_bytes() const {
        return sharded_usage ? sharded_usage->total() : used_bytes.load();
    }
    size_t get_capacity_bytes() const { return total_capacity_bytes; }
    uint64_t get_committed_bytes() const {
        return used_heap_idx.load(std::memory_order::relaxed) * RegionSize;
    }

    void inc_used_bytes(size_t bytes, uint64_t addr) {
        if (sharded_usage) {
            auto [ep, _] = FarLib::get_config().map_remote_addr(addr);
            sharded_usage->add(ep, bytes);
            return;
        }
        used_bytes.fetch_add(bytes);
        if (server_used_bytes) {
            auto [ep, _] = FarLib::get_config().map_remote_addr(addr);
            server_used_bytes[ep].fetch_add(bytes);
        }
    }
    void dec_used_bytes(size_t bytes, uint64_t addr) {
        if (sharded_usage) {
            auto [ep, _] = FarLib::get_config().map_remote_addr(addr);
            sharded_usage->subtract(ep, bytes);
            return;
        }
        used_bytes.fetch_sub(bytes);
        if (server_used_bytes) {
            auto [ep, _] = FarLib::get_config().map_remote_addr(addr);
            server_used_bytes[ep].fetch_sub(bytes);
        }
    }

    uint64_t get_server_used_bytes(size_t endpoint_idx) const {
        if (sharded_usage) return sharded_usage->server(endpoint_idx);
        if (!server_used_bytes) return 0;
        return server_used_bytes[endpoint_idx].load(std::memory_order::relaxed);
    }

    // init function
    // call only once per remote allocator
    void register_remote(size_t size) {
        if (simple_region_budget::enabled()) {
            // Configure before any descriptor can be claimed.  This resets
            // the per-bin ownership ledger while leaving the allocator's
            // existing remote accounting unchanged.
            simple_region_budget::remote().configure(RegionSize);
        }
        regions_size = size / RegionSize;
        total_capacity_bytes = size;
        const auto &config = FarLib::get_config();
        endpoint_count_ = config.server_count > 0
                              ? static_cast<size_t>(config.server_count)
                              : 0;
        endpoint_liveness_enabled_ = config.is_ec_batch_mode() ||
                                     config.ft_method == "ec_batch";
        standby_endpoint_ = endpoint_liveness_enabled_
                                ? config.ft_standby_endpoint
                                : -1;
        endpoint_failure_seen_.store(false, std::memory_order_relaxed);
        standby_allocation_count_.store(0, std::memory_order_relaxed);
        standby_flat_allocation_count_.store(0, std::memory_order_relaxed);
        standby_ec_allocation_count_.store(0, std::memory_order_relaxed);
        standby_first_allocation_logged_.store(false,
                                               std::memory_order_relaxed);
        endpoint_alive_.reset();
        if (endpoint_count_ != 0) {
            endpoint_alive_.reset(
                new std::atomic<uint8_t>[endpoint_count_]);
            for (size_t endpoint = 0; endpoint < endpoint_count_; endpoint++) {
                endpoint_alive_[endpoint].store(1, std::memory_order_relaxed);
            }
        }
        regions.reset(new RemoteRegionHead[regions_size]());
        used_heap_idx.store(0);
        used_bytes.store(0);

        // Initialize per-server counters
        size_t server_count = FarLib::get_config().server_count;
        server_used_bytes.reset(new std::atomic<uint64_t>[server_count]());
        for (size_t i = 0; i < server_count; i++) {
            server_used_bytes[i].store(0);
        }
        const char *shards = std::getenv("FARLIB_REMOTE_USAGE_SHARDS");
        if (shards && shards[0] == '1' && shards[1] == '\0') {
            sharded_usage = std::make_unique<ShardedRemoteUsage>();
            sharded_usage->reset(server_count);
        } else {
            sharded_usage.reset();
        }
        // EC shard pools are only materialized when ft_method != none; with the
        // default ft_method=none the arrays stay null and none of the
        // fault-tolerance code below can execute.
        if (FarLib::get_config().ft_enabled()) {
            ec_region_claimed.reset(new std::atomic<uint8_t>[regions_size]());
            for (size_t i = 0; i < regions_size; i++) {
                ec_region_claimed[i].store(0, std::memory_order_relaxed);
            }
            if (endpoint_liveness_enabled_) {
                regular_region_claimed.reset(
                    new std::atomic<uint8_t>[regions_size]());
                for (size_t i = 0; i < regions_size; i++) {
                    regular_region_claimed[i].store(0,
                                                    std::memory_order_relaxed);
                }
            } else {
                regular_region_claimed.reset();
            }
            ec_endpoint_cursor.reset(new std::atomic<size_t>[server_count]());
            for (size_t i = 0; i < server_count; i++) {
                ec_endpoint_cursor[i].store(0, std::memory_order_relaxed);
            }
            ec_region_count.store(0, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(ec_pool_mutex);
            ec_free_units.clear();
        } else {
            regular_region_claimed.reset();
        }
    }

    RemoteRegionHead *allocate_region(size_t bin,
                                      uint32_t requested_group = 0) {
        // Regions parked on the configured standby become eligible only after
        // the first non-standby endpoint failure.  Keep this check ahead of
        // every routing mode so simple, legacy, and fixed-six allocation all
        // share the same recovery handoff.
        if (standby_active_for_allocation()) {
            RemoteRegionHead *standby_region = nullptr;
            while ((standby_region =
                        standby_waiting_region_list[bin].pop_head()) != nullptr) {
                if (!endpoint_is_eligible_for_allocation(
                        standby_region->get_endpoint_idx())) {
                    continue;
                }
                if (!six_group::list_only_six::enabled() &&
                    six_group::enabled()) {
                    standby_region->ensure_six_group(requested_group);
                }
                return standby_region;
            }
        }

        if (simple_region_heat::remote_grouping_enabled()) {
            const uint8_t requested_class =
                simple_region_heat::normalize_class(requested_group);
            const auto order =
                simple_region_heat::fallback_order(requested_class);
            for (size_t rank = 0; rank < simple_region_heat::class_count();
                 ++rank) {
                const uint8_t candidate_class = order[rank];
                auto *candidate_list =
                    usable_region_list[bin].list_only_six_child(candidate_class);
                ASSERT(candidate_list != nullptr);
                RemoteRegionHead *candidate = nullptr;
                while ((candidate = candidate_list->pop_head()) != nullptr) {
                    if (endpoint_is_eligible_for_allocation(
                            candidate->get_endpoint_idx())) {
                        break;
                    }
                }
                if (rank == 0 && simple_region_budget::enabled()) {
                    // Count only the exact requested global pop. Fallback
                    // classes are intentionally not part of this miss.
                    simple_region_budget::remote().note_request(
                        bin, requested_class, candidate == nullptr);
                }
                if (candidate != nullptr) return candidate;
            }

            auto claim = [&](RemoteRegionHead *region, int old_bin = -1,
                             uint8_t old_class = simple_region_heat::kCold) {
                uint8_t actual_class = requested_class;
                if (simple_region_budget::enabled()) {
                    actual_class = simple_region_budget::remote().claim(
                        bin, requested_class, old_bin, old_class);
                }
                region->set_simple_class(actual_class);
                return region;
            };

            RemoteRegionHead *region = nullptr;
            while ((region = free_region_list.pop_head()) != nullptr) {
                if (!endpoint_is_eligible_for_allocation(
                        region->get_endpoint_idx())) {
                    continue;
                }
                // Capture ownership metadata before reset() overwrites both
                // the bin and the physical heat label.
                const int old_bin = static_cast<int>(region->get_bin());
                const uint8_t old_class = region->simple_class();
                region->reset(bin);
                return claim(region, old_bin, old_class);
            }

            size_t idx = used_heap_idx.load(std::memory_order_relaxed);
        retry_simple:
            if (idx < regions_size) {
                const size_t next = idx + 1;
                if (!used_heap_idx.compare_exchange_weak(
                        idx, next, std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    goto retry_simple;
                }
                bool reserved_for_ec = false;
                if (regular_region_claimed != nullptr) {
                    std::lock_guard<std::mutex> lock(ec_pool_mutex);
                    if (region_reserved_for_ec(idx)) {
                        reserved_for_ec = true;
                    } else {
                        regular_region_claimed[idx].store(
                            1, std::memory_order_release);
                    }
                } else if (region_reserved_for_ec(idx)) {
                    reserved_for_ec = true;
                }
                if (reserved_for_ec) {
                    idx = next;
                    goto retry_simple;
                }
                region = &regions[idx];
                region->init(idx * RegionSize, bin);
                if (!endpoint_is_eligible_for_allocation(
                        region->get_endpoint_idx())) {
                    if (endpoint_waiting_for_activation(
                        region->get_endpoint_idx())) {
                        // Preserve the v25 class/budget ownership while the
                        // physical Region is parked for standby activation.
                        standby_waiting_region_list[bin].insert_tail(
                            claim(region));
                    }
                    idx = next;
                    goto retry_simple;
                }
                return claim(region);
            }

            return nullptr;
        }

        // Fixed-six and legacy routing share the recovery-aware global pop.
        // A dead/parked endpoint is consumed from the list and never reused.
        RemoteRegionHead *region = nullptr;
        while (true) {
            region = requested_group == 0
                         ? usable_region_list[bin].pop_head()
                         : usable_region_list[bin].pop_group(requested_group);
            if (region == nullptr) break;
            if (endpoint_is_eligible_for_allocation(
                    region->get_endpoint_idx())) {
                return region;
            }
        }

        while ((region = free_region_list.pop_head()) != nullptr) {
            if (!endpoint_is_eligible_for_allocation(
                    region->get_endpoint_idx())) {
                continue;
            }
            region->reset(bin);
            if (!six_group::list_only_six::enabled() && six_group::enabled())
                region->ensure_six_group(requested_group);
            return region;
        }

        size_t idx = used_heap_idx.load(std::memory_order_relaxed);
    retry:
        if (idx < regions_size) {
            const size_t next = idx + 1;
            if (!used_heap_idx.compare_exchange_weak(
                    idx, next, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                goto retry;
            }
            bool reserved_for_ec = false;
            if (regular_region_claimed != nullptr) {
                std::lock_guard<std::mutex> lock(ec_pool_mutex);
                if (region_reserved_for_ec(idx)) {
                    reserved_for_ec = true;
                } else {
                    regular_region_claimed[idx].store(
                        1, std::memory_order_release);
                }
            } else if (region_reserved_for_ec(idx)) {
                reserved_for_ec = true;
            }
            if (reserved_for_ec) {
                idx = next;
                goto retry;
            }
            region = &regions[idx];
            region->init(idx * RegionSize, bin);
            if (!endpoint_is_eligible_for_allocation(
                    region->get_endpoint_idx())) {
                if (endpoint_waiting_for_activation(
                        region->get_endpoint_idx())) {
                    standby_waiting_region_list[bin].insert_tail(region);
                }
                idx = next;
                goto retry;
            }
            if (!six_group::list_only_six::enabled() && six_group::enabled())
                region->ensure_six_group(requested_group);
            return region;
        }
        return nullptr;
    }

    // Fault-tolerance path: hand out one whole EC shard, i.e. one
    // EcShardUnitSize (== FarLib::allocator::RegionSize) extent that lives on
    // `endpoint_idx`.  Returns InvalidRemoteAddr when ft_method == none or the
    // capacity is exhausted.  Nothing calls this while ft_method == none.
    uint64_t allocate_whole_region_on_endpoint(size_t endpoint_idx) {
        if (ec_region_claimed == nullptr) return InvalidRemoteAddr;
        const auto &config = FarLib::get_config();
        if (endpoint_idx >= static_cast<size_t>(config.server_count)) {
            return InvalidRemoteAddr;
        }
        if (!endpoint_is_eligible_for_allocation(endpoint_idx)) {
            return InvalidRemoteAddr;
        }
        std::lock_guard<std::mutex> lock(ec_pool_mutex);
        // 1) reuse a shard unit that a stripe released earlier
        for (size_t i = 0; i < ec_free_units.size(); i++) {
            uint64_t base = ec_free_units[i];
            if (config.map_remote_addr(base).first != endpoint_idx) continue;
            ec_free_units[i] = ec_free_units.back();
            ec_free_units.pop_back();
            record_standby_allocation(endpoint_idx, "ec_whole_region");
            return base;
        }
        // 2) claim a fresh remote region and take one of its two shard units
        for (size_t tries = 0; tries < regions_size; tries++) {
            size_t cursor = ec_endpoint_cursor[endpoint_idx].fetch_add(
                1, std::memory_order_relaxed);
            size_t idx = cursor % regions_size;
            uint64_t region_base = idx * RegionSize;
            if (config.map_remote_addr(region_base).first != endpoint_idx) {
                continue;
            }
            if (regular_region_claimed != nullptr &&
                regular_region_claimed[idx].load(std::memory_order_acquire) !=
                    0) {
                continue;
            }
            if (!config.validate_mapping(region_base, RegionSize)) continue;
            uint8_t expected = 0;
            if (!ec_region_claimed[idx].compare_exchange_strong(
                    expected, 1, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                continue;
            }
            ec_region_count.fetch_add(1, std::memory_order_relaxed);
            ec_free_units.push_back(region_base + EcShardUnitSize);
            record_standby_allocation(endpoint_idx, "ec_whole_region");
            return region_base;
        }
        return InvalidRemoteAddr;
    }

    // Fault-tolerance path: return one EC shard to the pool.
    bool release_whole_region(uint64_t base_addr) {
        if (ec_region_claimed == nullptr || base_addr == InvalidRemoteAddr) {
            return false;
        }
        if ((base_addr % EcShardUnitSize) != 0) return false;
        size_t region_idx = static_cast<size_t>(base_addr / RegionSize);
        if (region_idx >= regions_size) return false;
        if (ec_region_claimed[region_idx].load(std::memory_order_relaxed) == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(ec_pool_mutex);
        ec_free_units.push_back(base_addr);
        return true;
    }

    size_t get_ec_shard_region_count() const {
        return ec_region_count.load(std::memory_order_relaxed);
    }

    // Snapshot only public usable Regions.  Full, free-list and private
    // descriptors are intentionally excluded because they are not currently
    // available to the requested-bin global pop path.
    void tick_simple_budget(uint64_t window) {
        if (!simple_region_budget::enabled()) return;
        simple_region_budget::Supply supply{};
        for (size_t bin = 0; bin < RegionBinCount; ++bin) {
            usable_region_list[bin].snapshot_free_bytes(supply[bin]);
        }
        simple_region_budget::remote().tick(supply, window);

        // Fixed/off modes never expose pending transfers.  Keeping the
        // actuation path adaptive-only also guarantees that a fixed run does
        // not scan or move any remote descriptor.
        if (simple_region_budget::mode() !=
            simple_region_budget::Mode::Adaptive) {
            return;
        }

        const auto actuation_start = std::chrono::steady_clock::now();
        auto remaining = simple_region_budget::remote().pending_transfers();
        constexpr size_t kMaxScan = 4096;
        size_t scan_budget = kMaxScan;
        size_t scanned = 0;
        size_t eligible = 0;
        size_t busy = 0;
        size_t applied = 0;
        size_t nonempty_regions = 0;
        uint64_t free_bytes_reclassified = 0;

        const size_t start_bin = budget_scan_bin_cursor % RegionBinCount;
        const size_t class_count = simple_region_heat::class_count();
        const size_t start_class = budget_scan_class_cursor % class_count;
        for (size_t bin_offset = 0;
             bin_offset < RegionBinCount && scan_budget != 0; ++bin_offset) {
            const size_t bin = (start_bin + bin_offset) % RegionBinCount;
            for (size_t class_offset = 0; class_offset < class_count &&
                                           scan_budget != 0;
                 ++class_offset) {
                const size_t donor_index =
                    (start_class + class_offset) % class_count;
                uint64_t &quota = remaining[bin][donor_index];
                if (quota == 0) continue;
                const uint8_t donor_class =
                    simple_region_budget::label(donor_index);
                auto *donor_list =
                    usable_region_list[bin].list_only_six_child(donor_class);
                if (donor_list == nullptr) continue;

                while (quota != 0 && scan_budget != 0) {
                    // The returned Region remains locked through the
                    // controller recheck and usable-child publication.
                    auto *region = donor_list->detach_public_budget_region(
                        bin, donor_class, quota, scan_budget, scanned, busy);
                    if (region == nullptr) break;

                    ++eligible;
                    const uint8_t old_class = region->simple_class();
                    const bool nonempty = region->get_used_count() != 0;
                    const uint64_t free_bytes =
                        region->get_free_bytes_unsafe();
                    const uint8_t new_class =
                        simple_region_budget::remote().reassign_public(
                            bin, old_class);
                    if (simple_region_budget::index(new_class) !=
                        simple_region_budget::index(old_class)) {
                        // Keep the Region lock held while changing its
                        // persistent class and publishing it. Existing
                        // objects and bitmap state stay in place.
                        region->set_simple_class(new_class);
                        ++applied;
                        nonempty_regions += nonempty;
                        free_bytes_reclassified += free_bytes;
                    }
                    // Rejected controller rechecks are republished to the
                    // original child; neither outcome returns a public Region
                    // to the free list.
                    usable_region_list[bin].insert_tail(region);
                    region->unlock();
                }
            }
        }
        budget_scan_bin_cursor = (start_bin + 1) % RegionBinCount;
        budget_scan_class_cursor = (start_class + 1) % class_count;

        const auto pending_after =
            simple_region_budget::remote().pending_transfers();
        uint64_t pending = 0;
        for (const auto &per_bin : pending_after) {
            for (size_t c = 0; c < simple_region_heat::class_count(); ++c)
                pending += per_bin[c];
        }
        const auto elapsed_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                      actuation_start)
                                      .count();
        std::cout << "simple_region_budget.actuation domain=remote window="
                  << window << " scanned=" << scanned
                  << " eligible=" << eligible << " busy=" << busy
                  << " applied=" << applied << " pending=" << pending
                  << " elapsed_ns=" << elapsed_ns
                  << " nonempty_regions=" << nonempty_regions
                  << " free_bytes_reclassified=" << free_bytes_reclassified
                  << " kind=public_usable" << std::endl;
    }

    RemoteRegionHead *allocate_existing_group_region(
        size_t bin, uint32_t requested_group) {
        if (requested_group == 0) return nullptr;
        RemoteRegionHead *region = nullptr;
        while ((region = usable_region_list[bin].pop_group(requested_group)) !=
               nullptr) {
            if (endpoint_is_eligible_for_allocation(
                    region->get_endpoint_idx())) {
                return region;
            }
        }
        return nullptr;
    }

    six_group::Record *record_for(uint64_t addr) {
        if (addr == InvalidRemoteAddr || addr >= regions_size * RegionSize) {
            return nullptr;
        }
        return addr_to_region(addr)->six_record.load(
            std::memory_order_acquire);
    }

    uint8_t simple_class_for(uint64_t addr) const {
        if (addr == InvalidRemoteAddr || regions == nullptr ||
            addr >= regions_size * RegionSize) {
            return simple_region_heat::kCold;
        }
        return regions[addr / RegionSize].simple_class();
    }

    void return_back_region(RemoteRegionHead *region) {
        region->lock();
        return_back_region_unsafe(region);
        region->unlock();
    }

    void return_back_region_unsafe(RemoteRegionHead *region) {
        if (!endpoint_is_eligible_for_allocation(region->get_endpoint_idx())) {
            // The region may still contain live objects.  Keep its bitmap and
            // endpoint metadata in the global region array, but do not put it
            // back on an allocation list where it could be selected again.
            return;
        }
        if (region->is_full_unsafe()) {
            full_region_list.insert_tail(region);
        } else if (region->is_free_unsafe()) {
            free_region_list.insert_tail(region);
        } else {
            usable_region_list[region->get_bin()].insert_tail(region);
        }
    }

    void deallocate(uint64_t addr) {
        auto total_start = get_cycles();
        // Catch sentinel / out-of-range frees early: they would otherwise index
        // outside `regions` and corrupt accounting.
#ifdef FARLIB_ALLOC_DEBUG
        if (addr == InvalidRemoteAddr || addr >= regions_size * RegionSize) {
            dbg_invalid_free_detected.fetch_add(1, std::memory_order::relaxed);
        }
#endif
        ASSERT(addr != InvalidRemoteAddr);
        ASSERT(addr < regions_size * RegionSize);

        RemoteRegionHead *region = addr_to_region(addr);
        auto lock_start = get_cycles();
        region->lock();
        auto lock_acquire = get_cycles();
        auto lock_cycles = (int64_t)(lock_acquire - lock_start);
        int64_t bitmap_cycles = 0;
        int64_t list_cycles = 0;
        deallocate_unsafe(addr, region, &bitmap_cycles, &list_cycles);
        auto lock_hold_cycles = (int64_t)(get_cycles() - lock_acquire);
        region->unlock();
        auto total_cycles = (int64_t)(get_cycles() - total_start);
        profile::count_remote_dealloc(lock_cycles, lock_hold_cycles,
                                      bitmap_cycles, list_cycles, total_cycles);
    }

    void deallocate_batch(std::vector<uint64_t> &addrs) {
        if (addrs.empty()) {
            return;
        }
        for (uint64_t addr : addrs) {
            ASSERT(addr != InvalidRemoteAddr);
            ASSERT(addr < regions_size * RegionSize);
        }
        std::sort(addrs.begin(), addrs.end());
        for (size_t i = 1; i < addrs.size(); ++i) {
            // A duplicate here means two live entries claimed ownership of
            // the same retained backup. Reject it before any allocator state
            // is mutated rather than relying on a debug-only double-free bit.
            ASSERT(addrs[i] != addrs[i - 1]);
        }
        size_t begin = 0;
        while (begin < addrs.size()) {
            RemoteRegionHead *region = addr_to_region(addrs[begin]);
            size_t end = begin + 1;
            const uint64_t region_index = addrs[begin] / RegionSize;
            while (end < addrs.size() &&
                   addrs[end] / RegionSize == region_index) {
                ++end;
            }
            const uint64_t total_start = get_cycles();
            const uint64_t lock_start = get_cycles();
            region->lock();
            const uint64_t lock_acquire = get_cycles();
            int64_t bitmap_cycles = 0;
            int64_t list_cycles = 0;
            for (size_t i = begin; i < end; ++i) {
                int64_t item_bitmap_cycles = 0;
                int64_t item_list_cycles = 0;
                deallocate_unsafe(addrs[i], region,
                                  &item_bitmap_cycles,
                                  &item_list_cycles);
                bitmap_cycles += item_bitmap_cycles;
                list_cycles += item_list_cycles;
            }
            const int64_t lock_hold_cycles =
                static_cast<int64_t>(get_cycles() - lock_acquire);
            region->unlock();
            profile::count_remote_dealloc(
                static_cast<int64_t>(lock_acquire - lock_start),
                lock_hold_cycles, bitmap_cycles, list_cycles,
                static_cast<int64_t>(get_cycles() - total_start));
            for (size_t i = begin + 1; i < end; ++i) {
                // Preserve remote_dealloc_count's per-object denominator;
                // the group-level call above already accounts all cycles.
                profile::count_remote_dealloc(0, 0, 0, 0, 0);
            }
            begin = end;
        }
        addrs.clear();
    }

    void deallocate_unsafe(uint64_t addr, RemoteRegionHead *region,
                           int64_t *bitmap_cycles_out,
                           int64_t *list_cycles_out) {
        auto step1_start = get_cycles();
        bool freed = region->deallocate_unsafe(addr);
        auto step1_end = get_cycles();
        if (bitmap_cycles_out) {
            *bitmap_cycles_out = (int64_t)(step1_end - step1_start);
        }
#ifdef FARLIB_ALLOC_DEBUG
        if (!freed) {
            dbg_double_free_detected.fetch_add(1, std::memory_order::relaxed);
        }
#endif
        if (freed) {
            dec_used_bytes(get_bin_size(region->get_bin()), addr);
        }
        auto list_start = get_cycles();
        if (!region->is_in_thread_heap_unsafe()) {
            if (region->is_free_unsafe()) [[unlikely]] {
                // region in usable list
                usable_region_list[region->get_bin()]
                    .remove_list_safe_region_unsafe(region);
                free_region_list.insert_tail(region);
            } else if (region->is_full_just_now_unsafe()) [[unlikely]] {
                // region in full list
                full_region_list.remove_list_safe_region_unsafe(region);
                usable_region_list[region->get_bin()].insert_tail(region);
            }
        }
        auto list_end = get_cycles();
        if (list_cycles_out) {
            *list_cycles_out = (int64_t)(list_end - list_start);
        }
    }

    void info() {
        // #ifndef NDEBUG
        size_t allocated_size = used_heap_idx * RegionSize;
        std::cout << "DEBUG: RemoteGlobalHeap info: used_heap_idx=" << used_heap_idx << ", regions_size=" << regions_size << ", total_capacity_bytes=" << total_capacity_bytes << std::endl;
        std::cout << "allocated size: " << allocated_size << std::endl;
        std::cout << "allocated size(KB): "
                  << static_cast<double>(allocated_size) / (1L << 10)
                  << std::endl;
        std::cout << "allocated size(MB): "
                  << static_cast<double>(allocated_size) / (1L << 20)
                  << std::endl;
        std::cout << "allocated size(GB): "
                  << static_cast<double>(allocated_size) / (1L << 30)
                  << std::endl;
        // #endif
    }

};

extern RemoteGlobalHeap remote_global_heap;

class RemoteThreadHeap {
private:
    static constexpr size_t SixGroupSlotCount = 6;
    RemoteRegionHead *regions[RegionBinCount]{};
    RemoteRegionHead *six_group_regions[RegionBinCount]
                                         [SixGroupSlotCount]{};
    size_t six_group_replacement[RegionBinCount]{};
    uint64_t six_group_fallback_count_{0};

    static void return_region(RemoteRegionHead *region) {
        if (region == nullptr) return;
        remote_global_heap.return_back_region(region);
    }

    uint64_t allocate_legacy(size_t size, size_t bin) {
        RemoteRegionHead *region = regions[bin];
        if (region) [[likely]] {
            if (!remote_global_heap.endpoint_is_eligible_for_allocation(
                    region->get_endpoint_idx())) {
                // Drop only this thread's cache reference. Existing objects
                // remain represented by the region bitmap for deallocation.
                regions[bin] = nullptr;
            } else {
                region->lock();
                uint64_t addr = region->allocate_unsafe();
                if (addr != InvalidRemoteAddr) {
                    remote_global_heap.inc_used_bytes(get_bin_size(bin), addr);
                    remote_global_heap.record_standby_allocation(
                        region->get_endpoint_idx(), "flat");
                    region->unlock();
                    return addr;
                }
                remote_global_heap.return_back_region_unsafe(region);
                region->unlock();
                regions[bin] = nullptr;
            }
        }
        region = remote_global_heap.allocate_region(bin);
        regions[bin] = region;
        if (region == nullptr) return InvalidRemoteAddr;
        uint64_t addr = region->allocate();
        if (addr != InvalidRemoteAddr) {
            remote_global_heap.inc_used_bytes(get_bin_size(bin), addr);
            remote_global_heap.record_standby_allocation(
                region->get_endpoint_idx(), "flat");
        }
        return addr;
    }

    uint64_t allocate_simple_hotcold(size_t size, size_t bin,
                                     uint8_t requested_class) {
        requested_class =
            simple_region_heat::normalize_class(requested_class);
        auto &slots = six_group_regions[bin];

        auto try_slot = [&](size_t slot) -> uint64_t {
            RemoteRegionHead *region = slots[slot];
            if (region == nullptr || region->simple_class() != slot)
                return InvalidRemoteAddr;
            if (!remote_global_heap.endpoint_is_eligible_for_allocation(
                    region->get_endpoint_idx())) {
                // Keep a failed-endpoint descriptor out of every future
                // allocation list; its bitmap remains valid for deallocation.
                slots[slot] = nullptr;
                return InvalidRemoteAddr;
            }

            region->lock();
            const uint64_t addr = region->allocate_unsafe();
            if (addr != InvalidRemoteAddr) {
                remote_global_heap.inc_used_bytes(get_bin_size(bin), addr);
                remote_global_heap.record_standby_allocation(
                    region->get_endpoint_idx(), "flat");
                region->unlock();
                return addr;
            }
            // The descriptor is still private while its lock is held.  Put it
            // back on the class-selected global list before dropping ownership.
            remote_global_heap.return_back_region_unsafe(region);
            region->unlock();
            slots[slot] = nullptr;
            return InvalidRemoteAddr;
        };

        const auto order = simple_region_heat::fallback_order(requested_class);
        for (size_t rank = 0; rank < simple_region_heat::class_count(); ++rank) {
            if (const uint64_t addr = try_slot(order[rank]);
                addr != InvalidRemoteAddr) {
                if (rank != 0) ++six_group_fallback_count_;
                return addr;
            }
        }

        // allocate_region() reuses an opposite global Region before free/new
        // growth. Keep at most one private Region per class/bin; in particular,
        // do not flush or steal another thread's private cache.
        for (size_t attempt = 0; attempt < 2; ++attempt) {
            RemoteRegionHead *candidate =
                remote_global_heap.allocate_region(bin, requested_class);
            if (candidate == nullptr) {
                return InvalidRemoteAddr;
            }

            const size_t actual_slot = candidate->simple_class();
            // Legacy Hot/Cold labels are semantic class IDs (1 and 4), not
            // dense indices.  Semantic-six mode uses the dense 0..5 IDs.
            const bool valid_slot =
                simple_region_heat::class_count() == 6
                    ? actual_slot < 6
                    : (actual_slot == simple_region_heat::kCold ||
                       actual_slot == simple_region_heat::kHot);
            ASSERT(valid_slot);
            if (slots[actual_slot] == nullptr) {
                slots[actual_slot] = candidate;
                if (const uint64_t addr = try_slot(actual_slot);
                    addr != InvalidRemoteAddr) {
                    return addr;
                }
                continue;
            }

            // This class already has a private Region.  Return the newly
            // detached descriptor and retry the existing slot; this preserves
            // the bounded cache without exposing two owners for one class.
            remote_global_heap.return_back_region(candidate);
            if (const uint64_t addr = try_slot(actual_slot);
                addr != InvalidRemoteAddr) {
                return addr;
            }
        }
        return InvalidRemoteAddr;
    }

    uint64_t allocate_group_from_region(RemoteRegionHead *region, size_t bin,
                                        uint32_t requested_group) {
        if (!remote_global_heap.endpoint_is_eligible_for_allocation(
                region->get_endpoint_idx())) {
            return InvalidRemoteAddr;
        }
        region->lock();
        auto *record = region->six_record.load(std::memory_order_acquire);
        uint64_t addr = InvalidRemoteAddr;
        if (record != nullptr) {
            std::lock_guard<std::mutex> routing_guard(record->routing_mutex);
            if (six_group::group_of(record) == requested_group) {
                addr = region->allocate_unsafe();
            }
        }
        if (addr != InvalidRemoteAddr) {
            remote_global_heap.inc_used_bytes(get_bin_size(bin), addr);
            remote_global_heap.record_standby_allocation(
                region->get_endpoint_idx(), "flat");
            region->unlock();
            return addr;
        }
        remote_global_heap.return_back_region_unsafe(region);
        region->unlock();
        return InvalidRemoteAddr;
    }

    size_t preferred_slot(const std::array<uint32_t, SixGroupSlotCount> &family,
                          uint32_t group) const {
        auto found = std::find(family.begin(), family.end(), group);
        return found == family.end()
                   ? SixGroupSlotCount
                   : static_cast<size_t>(found - family.begin());
    }

    bool install_group_region(size_t bin, RemoteRegionHead *candidate,
                              uint32_t group,
                              const std::array<uint32_t,
                                               SixGroupSlotCount> &family) {
        auto &slots = six_group_regions[bin];
        for (RemoteRegionHead *held : slots) {
            if (held != nullptr && held->six_group_id() == group) {
                return false;
            }
        }
        size_t slot = preferred_slot(family, group);
        if (slot == SixGroupSlotCount || slots[slot] != nullptr) {
            slot = 0;
            while (slot < SixGroupSlotCount && slots[slot] != nullptr) ++slot;
        }
        if (slot == SixGroupSlotCount) {
            slot = six_group_replacement[bin]++ % SixGroupSlotCount;
            RemoteRegionHead *victim = slots[slot];
            slots[slot] = nullptr;
            return_region(victim);
        }
        slots[slot] = candidate;
        return true;
    }

    uint64_t allocate_fixed_six(size_t size, size_t bin,
                                uint32_t requested_group) {
        auto &registry = six_group::registry();
        std::array<uint32_t, SixGroupSlotCount> requested_family{};
        bool requested_family_ready = false;
        auto get_requested_family = [&]() -> const auto & {
            if (!requested_family_ready) {
                requested_family = registry.group_family(requested_group);
                requested_family_ready = true;
            }
            return requested_family;
        };

        for (size_t attempt = 0; attempt < SixGroupSlotCount + 2;
             ++attempt) {
            auto &slots = six_group_regions[bin];

            // Live Record groups are authoritative after an asynchronous
            // publication. Scan six bounded slots rather than trusting the
            // preferred class index.
            for (auto &region : slots) {
                if (region == nullptr) {
                    continue;
                }
                if (!remote_global_heap.endpoint_is_eligible_for_allocation(
                        region->get_endpoint_idx())) {
                    // Existing objects can still be freed through this
                    // descriptor, but no new allocation may use a dead or
                    // pre-failure standby endpoint.
                    region = nullptr;
                    continue;
                }
                if (region->six_group_id() != requested_group) continue;
                RemoteRegionHead *held = region;
                const uint64_t addr = allocate_group_from_region(
                    held, bin, requested_group);
                if (addr != InvalidRemoteAddr) return addr;
                region = nullptr;
            }

            RemoteRegionHead *candidate =
                remote_global_heap.allocate_region(bin, requested_group);
            uint32_t allocation_group = requested_group;
            if (candidate == nullptr) {
                // Soft routing is explicit and limited to this initial
                // owner/bin family. First use already-usable sibling Regions;
                // never retag a nonempty descriptor.
                for (uint32_t sibling : get_requested_family()) {
                    if (sibling == 0 || sibling == requested_group) continue;
                    candidate = remote_global_heap
                                    .allocate_existing_group_region(
                                        bin, sibling);
                    if (candidate != nullptr) {
                        allocation_group = sibling;
                        break;
                    }
                }
            }
            if (candidate == nullptr) {
                // Expose one descriptor owned by this thread before
                // reporting OOM. Cross-thread registry flushing is reserved
                // for quiescent teardown because production heaps are
                // thread-owned and deliberately lock-free.
                RemoteRegionHead *victim = nullptr;
                auto &slots = six_group_regions[bin];
                for (size_t offset = 0; offset < SixGroupSlotCount;
                     ++offset) {
                    const size_t slot =
                        six_group_replacement[bin]++ % SixGroupSlotCount;
                    if (slots[slot] == nullptr) continue;
                    victim = slots[slot];
                    slots[slot] = nullptr;
                    break;
                }
                if (victim != nullptr) {
                    return_region(victim);
                    continue;
                }
            }
            if (candidate == nullptr) return InvalidRemoteAddr;

            const auto allocation_family =
                registry.group_family(allocation_group);
            if (!install_group_region(bin, candidate, allocation_group,
                                      allocation_family)) {
                return_region(candidate);
                continue;
            }
            const uint64_t addr = allocate_group_from_region(
                candidate, bin, allocation_group);
            if (addr != InvalidRemoteAddr) {
                if (allocation_group != requested_group) {
                    ++six_group_fallback_count_;
                }
                return addr;
            }
            for (auto &region : slots) {
                if (region == candidate) {
                    region = nullptr;
                    break;
                }
            }
        }
        return InvalidRemoteAddr;
    }

public:
    RemoteThreadHeap() {
        std::lock_guard<std::mutex> lock(remote_thread_heap_registry_mutex());
        remote_thread_heap_registry().push_back(this);
    }

    ~RemoteThreadHeap() {
        release_all_regions();
        std::lock_guard<std::mutex> lock(remote_thread_heap_registry_mutex());
        auto &registry = remote_thread_heap_registry();
        auto it = std::find(registry.begin(), registry.end(), this);
        if (it != registry.end()) {
            registry.erase(it);
        }
    }

    uint64_t allocate(size_t size, uint32_t behavior_group_id = 0) {
        size_t wsize = wsize_from_size(size);
        size_t bin = bin_from_wsize(wsize);
        assert(get_bin_size(bin) >= size);
        assert(bin == 0 || get_bin_size(bin - 1) < size + sizeof(void *));
        if (simple_region_heat::remote_grouping_enabled()) {
            return allocate_simple_hotcold(
                size, bin,
                simple_region_heat::normalize_class(behavior_group_id));
        }
        if (!six_group::enabled()) {
            return allocate_legacy(size, bin);
        }
        if (behavior_group_id == 0) {
            behavior_group_id = six_group::registry().initial_group(0, bin);
        }
        return allocate_fixed_six(size, bin, behavior_group_id);
    }

    void release_all_regions() {
        for (size_t i = 0; i < RegionBinCount; i++) {
            auto *region = regions[i];
            if (region != nullptr) {
                if (remote_global_heap.endpoint_is_eligible_for_allocation(
                        region->get_endpoint_idx())) {
                    region->lock();
                    remote_global_heap.return_back_region_unsafe(region);
                    region->unlock();
                }
                regions[i] = nullptr;
            }
            for (auto &six_region : six_group_regions[i]) {
                if (six_region == nullptr) continue;
                return_region(six_region);
                six_region = nullptr;
            }
        }
    }

    size_t cached_regions_count() const {
        size_t count = 0;
        for (size_t bin = 0; bin < RegionBinCount; ++bin) {
            count += regions[bin] != nullptr;
            for (auto *region : six_group_regions[bin]) {
                count += region != nullptr;
            }
        }
        return count;
    }

    uint64_t six_group_fallback_count() const {
        return six_group_fallback_count_;
    }

    void deallocate(uint64_t addr) { remote_global_heap.deallocate(addr); }

    void deallocate_batch(std::vector<uint64_t> &addrs) {
        remote_global_heap.deallocate_batch(addrs);
    }
};

inline void flush_all_registered_thread_heaps() {
    std::lock_guard<std::mutex> lock(remote_thread_heap_registry_mutex());
    for (auto *heap : remote_thread_heap_registry()) {
        if (heap) {
            heap->release_all_regions();
        }
    }
}

extern thread_local RemoteThreadHeap remote_thread_heap;

}  // namespace remote
}  // namespace allocator
}  // namespace FarLib
