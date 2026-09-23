#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cache/region_based_allocator.hpp"
#include "cache/alloc/sharded_remote_usage.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/cpu_cycles.hpp"
#include "utils/stats.hpp"
#include "utils/uthreads.hpp"

namespace FarLib {

namespace allocator {

namespace remote {
class RemoteThreadHeap;

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
    uint32_t last_map_idx;
    // 0 for allocated, 1 for free
    uint64_t *blockmap;
    size_t entry_size;
    uint64_t base_addr;
    uint64_t used_count;
    static constexpr size_t MapElementBitCount = sizeof(uint64_t) * 8;
    static constexpr uint64_t FreeMap = std::numeric_limits<uint64_t>::max();
    static constexpr uint32_t FreeMap32 = std::numeric_limits<uint32_t>::max();
    static constexpr uint64_t FullMap = 0;

public:
    void lock() {
        while (flag.test_and_set()) {
        }
    }

    void unlock() { flag.clear(); }

    void init(uint64_t base_addr, uint32_t bin) {
        this->base_addr = base_addr;
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

    uint64_t get_used_count() const { return used_count; }

    bool is_in_thread_heap_unsafe() {
        assert(!((next == nullptr) ^ (prev == nullptr)));
        return next == nullptr && prev == nullptr;
    }

    void remove_self() {
        lock();
        remove_self_unsafe();
        unlock();
    }
};

class RemoteRegionList {
private:
    std::atomic_flag flag;
    DoubleLinkedListHead dummy_head;
    DoubleLinkedListHead dummy_tail;

    void lock() { while (flag.test_and_set()); }

    void unlock() { flag.clear(); }

public:
    RemoteRegionList() {
        dummy_head.next = &dummy_tail;
        dummy_tail.prev = &dummy_head;
    }

    void insert_tail(RemoteRegionHead *node) {
        lock();
        dummy_tail.insert_front_unsafe(node);
        unlock();
    }

    void insert_head(RemoteRegionHead *node) {
        lock();
        dummy_head.insert_back_unsafe(node);
        unlock();
    }

    inline bool empty() const {
        assert(!((dummy_head.next == &dummy_tail) ^
                 (dummy_tail.prev == &dummy_head)));
        return dummy_head.next == &dummy_tail;
    }

    RemoteRegionHead *pop_head() {
        lock();
        if (empty()) {
            unlock();
            return nullptr;
        }
        RemoteRegionHead *region =
            static_cast<RemoteRegionHead *>(dummy_head.next);
        // List membership is protected by this list lock. Taking the region
        // lock here would invert the region->list order used by deallocation.
        region->remove_self_unsafe();
        unlock();
        return region;
    }

    RemoteRegionHead *pop_tail() {
        lock();
        if (empty()) {
            unlock();
            return nullptr;
        }
        RemoteRegionHead *region =
            static_cast<RemoteRegionHead *>(dummy_tail.prev);
        region->remove_self_unsafe();
        unlock();
        return region;
    }

    void remove_list_safe_region_unsafe(RemoteRegionHead *region) {
        lock();
        region->remove_self_unsafe();
        unlock();
    }
};

class RemoteGlobalHeap {
private:
    RemoteRegionList usable_region_list[RegionBinCount];
    RemoteRegionList full_region_list;
    RemoteRegionList free_region_list;
    std::unique_ptr<RemoteRegionHead[]> regions;
    size_t regions_size;
    std::atomic_size_t used_heap_idx;
    std::atomic<uint64_t> used_bytes{0};
    size_t total_capacity_bytes{0};
    std::unique_ptr<std::atomic<uint64_t>[]> server_used_bytes;
    std::unique_ptr<ShardedRemoteUsage> sharded_usage;

#ifdef FARLIB_ALLOC_DEBUG
    std::atomic<uint64_t> dbg_double_free_detected{0};
    std::atomic<uint64_t> dbg_invalid_free_detected{0};
#endif

    RemoteRegionHead *addr_to_region(uint64_t addr) {
        return &regions[addr / RegionSize];
    }

public:
    ~RemoteGlobalHeap() {
        std::cout << "used memory: " << used_heap_idx * RegionSize << std::endl;
        std::cout << "exact used bytes: " << get_used_bytes() << std::endl;
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
        regions_size = size / RegionSize;
        total_capacity_bytes = size;
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
    }

    RemoteRegionHead *allocate_region(size_t bin) {
        RemoteRegionHead *region = usable_region_list[bin].pop_head();
        if (region) {
            return region;
        }
        // register a new region from free list to usable list
        region = free_region_list.pop_head();
        if (region) {
            region->reset(bin);
            return region;
        }
        // lock the global region vector
        size_t idx = used_heap_idx.load();
    retry:
        if (idx < regions_size) {
            size_t new_idx = idx + 1;
            if (!used_heap_idx.compare_exchange_weak(idx, new_idx)) {
                goto retry;
            }
            region = &regions[idx];
            region->init(idx * RegionSize, bin);
            return region;
        }
        return nullptr;
    }

    void return_back_region(RemoteRegionHead *region) {
        region->lock();
        return_back_region_unsafe(region);
        region->unlock();
    }

    void return_back_region_unsafe(RemoteRegionHead *region) {
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
    RemoteRegionHead *regions[RegionBinCount];

public:
    RemoteThreadHeap() {
        std::memset(regions, 0, sizeof(regions));
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

    uint64_t allocate(size_t size) {
        size_t wsize = wsize_from_size(size);
        size_t bin = bin_from_wsize(wsize);
        assert(get_bin_size(bin) >= size);
        assert(bin == 0 || get_bin_size(bin - 1) < size + sizeof(void *));
        RemoteRegionHead *region = regions[bin];
        // region will not empty for now
        // because it has >= 1 object allocated by this thread
        if (region) [[likely]] {
            region->lock();
            uint64_t addr = region->allocate_unsafe();
            if (addr != InvalidRemoteAddr) {
                remote_global_heap.inc_used_bytes(get_bin_size(bin), addr);
                region->unlock();
                return addr;
            } else {
                remote_global_heap.return_back_region_unsafe(region);
                region->unlock();
            }
        }
        region = remote_global_heap.allocate_region(bin);
        regions[bin] = region;
        if (region) [[likely]] {
            uint64_t addr = region->allocate();
            if (addr != InvalidRemoteAddr) {
                remote_global_heap.inc_used_bytes(get_bin_size(bin), addr);
            }
            return addr;
        } else {
            return InvalidRemoteAddr;
        }
    }

    void release_all_regions() {
        for (size_t i = 0; i < RegionBinCount; i++) {
            auto *region = regions[i];
            if (!region) {
                continue;
            }
            region->lock();
            remote_global_heap.return_back_region_unsafe(region);
            region->unlock();
            regions[i] = nullptr;
        }
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
