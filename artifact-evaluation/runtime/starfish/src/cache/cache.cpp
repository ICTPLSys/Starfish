#include "cache/cache.hpp"

#include <sys/cdefs.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#ifdef NO_REMOTE

namespace FarLib {
namespace allocator {
#ifdef USE_BUMP_ALLOCATOR
std::byte *heap = nullptr;
size_t bump_pointer_offset = 0;
#endif
}  // namespace allocator
}  // namespace FarLib

#else

#include "cache/alloc/remote_allocator.hpp"
#include "design2/fixed_six_runtime.hpp"

namespace FarLib {
namespace cache {

std::unique_ptr<Cache> Cache::default_instance;

}  // namespace cache

namespace allocator {

class ThreadHeap;

namespace {
struct ThreadHeapDiagRegistry {
    std::mutex mutex;
    std::unordered_set<ThreadHeap *> live_heaps;
    std::atomic<uint64_t> constructed_total{0};
    std::atomic<uint64_t> destroyed_total{0};
    std::atomic<uint64_t> region_acquires{0};
    std::atomic<uint64_t> region_returns{0};
    std::atomic<uint64_t> release_all_calls{0};
    std::atomic<uint64_t> slot_overwrites{0};
    std::atomic<uint64_t> prevented_slot_overwrites{0};
    std::atomic<uint64_t> migrated_candidate_returns{0};
    std::atomic<uint64_t> region_lock_contentions{0};
    std::atomic<uint64_t> global_region_alloc_calls{0};
    std::atomic<uint64_t> global_region_alloc_cycles{0};
    std::atomic<uint64_t> global_region_return_cycles{0};
    std::atomic<uint64_t> full_return_batch_flushes{0};
    std::atomic<uint64_t> full_return_batched_regions{0};
    std::atomic<uint64_t> full_return_pending_cycles{0};
    std::atomic<uint64_t> full_return_publish_cycles{0};
    std::atomic<uint64_t> full_return_empty_flushes{0};
    std::atomic<uint64_t> full_return_pending_current{0};
    std::atomic<uint64_t> memory_low_calls{0};
    std::atomic<uint64_t> memory_low_free_calls{0};
    std::atomic<uint64_t> memory_low_evacuator_calls{0};
    std::atomic<uint64_t> memory_low_cycles{0};
    std::atomic<uint64_t> slot_placement_mismatches{0};
    std::atomic<uint64_t> slot_exhaustions{0};
    std::atomic<uint64_t> overwritten_region_objects{0};
    std::atomic<uint64_t> six_pressure_reclaim_calls{0};
    std::atomic<uint64_t> six_pressure_reclaimed_regions{0};
    std::atomic<uint64_t> slot_overwrites_by_bin[RegionBinCount]{};
    std::atomic<uint64_t> fibre_heap_creates{0};
    std::atomic<uint64_t> fibre_heap_destroys{0};
    std::atomic<uint64_t> fibre_heap_os_migrations{0};
    std::atomic<uint64_t>
        fallback_slot_allocations[RegionFallbackDirectionCount]{};
    std::atomic<uint64_t>
        fallback_slot_bytes[RegionFallbackDirectionCount]{};
};

ThreadHeapDiagRegistry &thread_heap_diag_registry() {
    static auto *registry = new ThreadHeapDiagRegistry;
    return *registry;
}

bool thread_heap_full_return_batch_enabled() {
    static const bool enabled = [] {
        const char *value =
            std::getenv("FARLIB_THREAD_HEAP_FULL_RETURN_BATCH");
        const bool result =
            value == nullptr || std::strtoull(value, nullptr, 0) != 0;
        std::cerr << "allocator.full_return_batch_enabled=" << result
                  << std::endl;
        return result;
    }();
    return enabled;
}

void record_fallback_slot_allocation(RegionPlacement requested,
                                     RegionPlacement actual, size_t bin) {
    size_t direction;
    if (requested == RegionPlacement::Resident &&
        actual == RegionPlacement::Streaming) {
        direction = 0;
    } else if (requested == RegionPlacement::Streaming &&
               actual == RegionPlacement::Resident) {
        direction = 1;
    } else {
        return;
    }
    auto &diag = thread_heap_diag_registry();
    diag.fallback_slot_allocations[direction].fetch_add(
        1, std::memory_order_relaxed);
    diag.fallback_slot_bytes[direction].fetch_add(
        get_bin_size(bin), std::memory_order_relaxed);
}
}  // namespace

void flush_all_thread_heap_pending_full_returns();
void reclaim_six_group_regions_for_pressure();

class ThreadHeap {
public:
    static constexpr size_t SixGroupSlotCount = 6;

    ThreadHeap();

    ~ThreadHeap();

    static ThreadHeap &current();

    BlockHead *allocate(size_t size, cache::far_obj_t obj,
                        cache::DereferenceScope *scope,
                        RegionPlacement requested_placement,
                        uint32_t requested_group_id,
                        uint32_t requested_behavior_group_id,
                        bool pin_publication) {
        profile::start_allocate();
        BlockHead *block = allocate_block(size + sizeof(BlockHead), obj, scope,
                                          requested_placement,
                                          requested_group_id,
                                          requested_behavior_group_id,
                                          pin_publication);
        if (block != nullptr) {
            assert(block >= global_heap.get_heap());
            assert((char *)block + size + sizeof(BlockHead) <=
                   (char *)global_heap.get_heap() +
                       global_heap.get_heap_size());
        }
        profile::end_allocate();
        return block;
    }

    void bind_fibre(void *fibre) {
        if (fibre == nullptr) {
            ERROR("null fibre for fibre-local ThreadHeap");
        }
        if (owner_fibre == nullptr) {
            owner_fibre = fibre;
            return;
        }
        if (owner_fibre != fibre) {
            ERROR("fibre-local ThreadHeap owner mismatch");
        }
        static const bool migration_diag = [] {
            const char *value = std::getenv("FARLIB_FIBRE_HEAP_DIAG");
            return value != nullptr && std::strtoull(value, nullptr, 0) != 0;
        }();
        if (!migration_diag) return;
        const auto os_thread = std::this_thread::get_id();
        if (last_owner_thread == std::thread::id{}) {
            last_owner_thread = os_thread;
            return;
        }
        if (last_owner_thread != os_thread) {
            thread_heap_diag_registry().fibre_heap_os_migrations.fetch_add(
                1, std::memory_order_relaxed);
            last_owner_thread = os_thread;
        }
    }

    void release_all_regions() {
        auto &diag = thread_heap_diag_registry();
        diag.release_all_calls.fetch_add(1, std::memory_order_relaxed);
        std::vector<RegionHead *> active;
        std::vector<RegionHead *> pending_full;
        detach_all_regions(active, pending_full);
        submit_full_return_vector(pending_full);
        for (RegionHead *region : active) {
            return_region(region);
            diag.region_returns.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void detach_pending_full_returns(std::vector<RegionHead *> &out) {
        const uint64_t start = get_cycles();
        lock_regions();
        const size_t count = pending_full_return_count;
        for (size_t i = 0; i < count; ++i) {
            out.push_back(pending_full_returns[i]);
            pending_full_returns[i] = nullptr;
        }
        pending_full_return_count = 0;
        unlock_regions();
        auto &diag = thread_heap_diag_registry();
        if (count == 0) {
            diag.full_return_empty_flushes.fetch_add(
                1, std::memory_order_relaxed);
            return;
        }
        diag.full_return_pending_current.fetch_sub(
            count, std::memory_order_relaxed);
        const uint64_t elapsed = get_cycles() - start;
        diag.full_return_pending_cycles.fetch_add(
            elapsed, std::memory_order_relaxed);
        diag.global_region_return_cycles.fetch_add(
            elapsed, std::memory_order_relaxed);
    }

    void detach_all_regions_for_shutdown(
        std::vector<RegionHead *> &active,
        std::vector<RegionHead *> &pending_full) {
        detach_all_regions(active, pending_full);
    }

    void detach_six_group_regions_for_pressure(
        std::vector<RegionHead *> &active,
        std::vector<RegionHead *> &pending_full) {
        lock_regions();
        for (size_t placement = 0; placement < RegionPlacementCount;
             ++placement) {
            for (size_t bin = 0; bin < RegionBinCount; ++bin) {
                for (auto &region : six_group_regions[placement][bin]) {
                    if (region != nullptr) active.push_back(region);
                    region = nullptr;
                }
            }
        }
        const size_t count = pending_full_return_count;
        for (size_t i = 0; i < count; ++i) {
            pending_full.push_back(pending_full_returns[i]);
            pending_full_returns[i] = nullptr;
        }
        pending_full_return_count = 0;
        unlock_regions();
        if (count != 0) {
            thread_heap_diag_registry().full_return_pending_current.fetch_sub(
                count, std::memory_order_relaxed);
        }
    }

    static void publish_detached_full_regions(
        std::vector<RegionHead *> &pending_full) {
        submit_full_return_vector(pending_full);
    }

    static void publish_detached_regions(
        std::vector<RegionHead *> &active,
        std::vector<RegionHead *> &pending_full) {
        submit_full_return_vector(pending_full);
        auto &diag = thread_heap_diag_registry();
        for (RegionHead *region : active) {
            return_region(region);
            diag.region_returns.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void collect_region_refs(std::unordered_set<RegionHead *> &refs,
                             size_t &non_null_refs) {
        lock_regions();
        for (size_t placement = 0; placement < RegionPlacementCount;
             ++placement) {
            for (size_t bin = 0; bin < RegionBinCount; ++bin) {
                for (auto *region : six_group_regions[placement][bin]) {
                    if (region == nullptr) continue;
                    ++non_null_refs;
                    refs.insert(region);
                }
            }
        }
        for (size_t placement = 0; placement < RegionPlacementCount;
             ++placement) {
            for (size_t i = 0; i < RegionBinCount; ++i) {
                auto *region = regions[placement][i].load(
                    std::memory_order_acquire);
                if (region == nullptr) {
                    continue;
                }
                ++non_null_refs;
                refs.insert(region);
            }
        }
        for (size_t i = 0; i < pending_full_return_count; ++i) {
            RegionHead *region = pending_full_returns[i];
            if (region == nullptr) continue;
            ++non_null_refs;
            refs.insert(region);
        }
        unlock_regions();
    }

    size_t six_group_cached_count() {
        lock_regions();
        size_t count = 0;
        for (size_t placement = 0; placement < RegionPlacementCount;
             ++placement) {
            for (size_t bin = 0; bin < RegionBinCount; ++bin) {
                for (auto *region : six_group_regions[placement][bin]) {
                    count += region != nullptr;
                }
            }
        }
        unlock_regions();
        return count;
    }

    void collect_six_group_stats(uint64_t &hits, uint64_t &misses,
                                 uint64_t &fallbacks, uint64_t &held) {
        lock_regions();
        hits += six_group_cache_hits;
        misses += six_group_cache_misses;
        fallbacks += six_group_fallback_allocations;
        for (size_t placement = 0; placement < RegionPlacementCount;
             ++placement) {
            for (size_t bin = 0; bin < RegionBinCount; ++bin) {
                for (auto *region : six_group_regions[placement][bin]) {
                    held += region != nullptr;
                }
            }
        }
        unlock_regions();
    }

private:
    void detach_all_regions(std::vector<RegionHead *> &active,
                            std::vector<RegionHead *> &pending_full) {
        active.reserve(RegionPlacementCount * RegionBinCount *
                       (SixGroupSlotCount + 1));
        pending_full.reserve(RegionReturnBatchSize);
        lock_regions();
        for (size_t placement = 0; placement < RegionPlacementCount;
             ++placement) {
            for (size_t bin = 0; bin < RegionBinCount; ++bin) {
                for (auto &region : six_group_regions[placement][bin]) {
                    if (region != nullptr) active.push_back(region);
                    region = nullptr;
                }
            }
        }
        for (size_t placement = 0; placement < RegionPlacementCount;
             ++placement) {
            for (size_t i = 0; i < RegionBinCount; ++i) {
                RegionHead *region = regions[placement][i].exchange(
                    nullptr, std::memory_order_acq_rel);
                if (region != nullptr) active.push_back(region);
            }
        }
        const size_t pending_count = pending_full_return_count;
        for (size_t i = 0; i < pending_count; ++i) {
            pending_full.push_back(pending_full_returns[i]);
            pending_full_returns[i] = nullptr;
        }
        pending_full_return_count = 0;
        unlock_regions();
        if (pending_count != 0) {
            thread_heap_diag_registry().full_return_pending_current.fetch_sub(
                pending_count, std::memory_order_relaxed);
        }
    }

    static void return_region(RegionHead *region) {
        const uint64_t start = get_cycles();
        global_heap.return_back_region(region);
        thread_heap_diag_registry().global_region_return_cycles.fetch_add(
            get_cycles() - start, std::memory_order_relaxed);
    }

    static void submit_full_return_batch(RegionHead **batch, size_t count) {
        if (count == 0) return;
        const uint64_t start = get_cycles();
        global_heap.submit_full_regions_batch(batch, count);
        auto &diag = thread_heap_diag_registry();
        const uint64_t elapsed = get_cycles() - start;
        diag.full_return_batch_flushes.fetch_add(1,
                                                 std::memory_order_relaxed);
        diag.full_return_batched_regions.fetch_add(count,
                                                   std::memory_order_relaxed);
        // Keep the existing log field name for analysis compatibility. With
        // async returns, it measures submission/wakeup or synchronous fallback,
        // not the background consumer's subsequent FULL-list publication.
        diag.full_return_publish_cycles.fetch_add(
            elapsed, std::memory_order_relaxed);
        diag.global_region_return_cycles.fetch_add(
            elapsed, std::memory_order_relaxed);
    }

    static void submit_full_return_vector(
        std::vector<RegionHead *> &pending_full) {
        for (size_t offset = 0; offset < pending_full.size();
             offset += RegionReturnBatchSize) {
            const size_t count = std::min(RegionReturnBatchSize,
                                          pending_full.size() - offset);
            submit_full_return_batch(pending_full.data() + offset, count);
        }
    }

    void return_full_region_batched(RegionHead *region) {
        RegionHead *batch[RegionReturnBatchSize];
        size_t count = 0;
        const uint64_t start = get_cycles();
        auto &diag = thread_heap_diag_registry();
        lock_regions();
        assert(pending_full_return_count < RegionReturnBatchSize);
        pending_full_returns[pending_full_return_count++] = region;
        diag.full_return_pending_current.fetch_add(
            1, std::memory_order_relaxed);
        if (pending_full_return_count == RegionReturnBatchSize) {
            count = pending_full_return_count;
            for (size_t i = 0; i < count; ++i) {
                batch[i] = pending_full_returns[i];
                pending_full_returns[i] = nullptr;
            }
            pending_full_return_count = 0;
            diag.full_return_pending_current.fetch_sub(
                count, std::memory_order_relaxed);
        }
        unlock_regions();
        const uint64_t elapsed = get_cycles() - start;
        diag.full_return_pending_cycles.fetch_add(
            elapsed, std::memory_order_relaxed);
        diag.global_region_return_cycles.fetch_add(
            elapsed, std::memory_order_relaxed);
        submit_full_return_batch(batch, count);
    }

    void lock_regions() {
        if (!region_lock.test_and_set(std::memory_order_acquire)) {
            return;
        }
        thread_heap_diag_registry().region_lock_contentions.fetch_add(
            1, std::memory_order_relaxed);
        while (region_lock.test_and_set(std::memory_order_acquire)) {
            __builtin_ia32_pause();
        }
    }

    void unlock_regions() {
        region_lock.clear(std::memory_order_release);
    }

    BlockHead *allocate_block(size_t block_size, cache::far_obj_t obj,
                              cache::DereferenceScope *scope,
                              RegionPlacement requested_placement,
                              uint32_t requested_group_id,
                              uint32_t requested_behavior_group_id,
                              bool pin_publication) {
        size_t wsize = wsize_from_size(block_size);
        size_t bin = bin_from_wsize(wsize);
        assert(get_bin_size(bin) >= block_size);
        assert(bin == 0 || get_bin_size(bin - 1) < block_size + sizeof(void *));
        BlockHead *block = allocate_block_from_bin(
            bin, obj, scope, requested_placement, requested_group_id,
            requested_behavior_group_id, pin_publication);
        profile::trace_alloc(block, bin);
        return block;
    }

    BlockHead *allocate_block_from_bin(size_t bin, cache::far_obj_t obj,
                                       cache::DereferenceScope *scope,
                                       RegionPlacement requested_placement,
                                       uint32_t requested_group_id,
                                       uint32_t requested_behavior_group_id,
                                       bool pin_publication) {
        if (simple_region_heat::local_routing_enabled()) {
            return allocate_hotcold_block(
                bin, obj, scope, requested_placement, requested_group_id,
                requested_behavior_group_id, pin_publication);
        }
        if (six_group::enabled()) {
            return allocate_six_group_block(
                bin, obj, scope, requested_placement, requested_group_id,
                requested_behavior_group_id, pin_publication);
        }
        if (!global_heap.region_placement_is_enabled()) {
            requested_placement = RegionPlacement::Unclassified;
            requested_group_id = 0;
        }
        for (;;) {
            RegionPlacement slot_placement = requested_placement;
            if (global_heap.region_placement_is_enabled() &&
                (requested_placement == RegionPlacement::Resident ||
                 requested_placement == RegionPlacement::Streaming) &&
                regions[placement_index(requested_placement)][bin].load(
                    std::memory_order_acquire) == nullptr &&
                !global_heap.can_allocate_region_directly_now(
                    bin, requested_placement)) {
                const RegionPlacement fallback_placement =
                    requested_placement == RegionPlacement::Resident
                        ? RegionPlacement::Streaming
                        : RegionPlacement::Resident;
                if (regions[placement_index(fallback_placement)][bin].load(
                        std::memory_order_acquire) != nullptr) {
                    slot_placement = fallback_placement;
                }
            }
            auto &slot =
                regions[placement_index(slot_placement)][bin];
            lock_regions();
            RegionHead *region = slot.load(std::memory_order_relaxed);
            if (region != nullptr) [[likely]] {
                const bool placement_matches =
                    global_heap.region_placement_is_enabled() &&
                            !FarLib::get_config().region_placement_bind_groups
                        ? region->placement_class_matches(slot_placement)
                        : region->placement_matches(slot_placement,
                                                    requested_group_id);
                if (!placement_matches) {
                    thread_heap_diag_registry()
                        .slot_placement_mismatches.fetch_add(
                            1, std::memory_order_relaxed);
                    slot.store(nullptr, std::memory_order_release);
                    unlock_regions();
                    return_region(region);
                    thread_heap_diag_registry().region_returns.fetch_add(
                        1, std::memory_order_relaxed);
                    return current().allocate_block_from_bin(
                        bin, obj, scope, requested_placement,
                        requested_group_id, requested_behavior_group_id,
                        pin_publication);
                }
                if (slot_placement == RegionPlacement::Streaming &&
                    requested_group_id == 0) {
                    global_heap.mark_streaming_region_shared(region);
                }
                BlockHead *block = region->allocate(obj, pin_publication);
                if (block != nullptr) [[likely]] {
                    record_fallback_slot_allocation(
                        requested_placement, slot_placement, bin);
                    unlock_regions();
                    return block;
                }

                slot.store(nullptr, std::memory_order_release);
                unlock_regions();
                thread_heap_diag_registry().slot_exhaustions.fetch_add(
                    1, std::memory_order_relaxed);
                if (thread_heap_full_return_batch_enabled()) {
                    return_full_region_batched(region);
                } else {
                    return_region(region);
                }
                thread_heap_diag_registry().region_returns.fetch_add(
                    1, std::memory_order_relaxed);
                // Publishing to a global list may yield. Resume through the
                // ThreadHeap of the OS thread that now runs this fibre.
                return current().allocate_block_from_bin(
                    bin, obj, scope, requested_placement,
                    requested_group_id, requested_behavior_group_id,
                    pin_publication);
            }
            unlock_regions();

            auto &diag = thread_heap_diag_registry();
            const uint64_t global_alloc_start = get_cycles();
            scope_diag::refill(fibre_self());
            RegionHead *candidate = global_heap.allocate_region(
                bin, requested_placement, requested_group_id);
            diag.global_region_alloc_calls.fetch_add(
                1, std::memory_order_relaxed);
            diag.global_region_alloc_cycles.fetch_add(
                get_cycles() - global_alloc_start,
                std::memory_order_relaxed);

            bool evacuator_waiting = Cache::get_default()->evacuator_waiting();
            const bool memory_low = global_heap.need_evacuate(false);
            if (candidate == nullptr ||
                (evacuator_waiting &&
                 diag.full_return_pending_current.load(
                     std::memory_order_relaxed) != 0)) [[unlikely]] {
                flush_all_thread_heap_pending_full_returns();
                if (candidate == nullptr) {
                    // Bounded progress fallback: do not leave full regions
                    // hidden if the master has not yet scheduled a drain.
                    global_heap.drain_async_full_returns(64);
                }
            }
            if (memory_low || evacuator_waiting) [[unlikely]] {
                const uint64_t memory_low_start = get_cycles();
                diag.memory_low_calls.fetch_add(1, std::memory_order_relaxed);
                diag.memory_low_free_calls.fetch_add(
                    memory_low, std::memory_order_relaxed);
                diag.memory_low_evacuator_calls.fetch_add(
                    evacuator_waiting, std::memory_order_relaxed);
                global_heap.on_memory_low();
                if (scope != nullptr) {
                    Cache::get_default()->update_scope(*scope);
                }
                diag.memory_low_cycles.fetch_add(
                    get_cycles() - memory_low_start,
                    std::memory_order_relaxed);
            }

            // A fibre may resume on another OS thread after on_memory_low().
            // In that case `this` still names the old thread's ThreadHeap, so
            // publishing into its slot would let two threads call the same
            // non-thread-safe RegionHead::allocate(). Return any reserved
            // candidate unused, then retry through the current OS thread.
            if (&current() != this) [[unlikely]] {
                if (candidate == nullptr) {
                    return current().allocate_block_from_bin(
                        bin, obj, scope, requested_placement,
                        requested_group_id, requested_behavior_group_id,
                        pin_publication);
                }
                candidate->state.store(IN_USE, std::memory_order_relaxed);
                diag.migrated_candidate_returns.fetch_add(
                    1, std::memory_order_relaxed);
                diag.region_acquires.fetch_add(1, std::memory_order_relaxed);
                return_region(candidate);
                diag.region_returns.fetch_add(1, std::memory_order_relaxed);
                return current().allocate_block_from_bin(
                    bin, obj, scope, requested_placement,
                    requested_group_id, requested_behavior_group_id,
                    pin_publication);
            }

            if (candidate == nullptr) [[unlikely]] {
                // A fibre that ran while on_memory_low() yielded may have
                // populated either the requested or fallback slot. Retry that
                // winner before reporting OOM.
                lock_regions();
                bool slot_populated =
                    regions[placement_index(requested_placement)][bin].load(
                        std::memory_order_relaxed) != nullptr;
                if (!slot_populated &&
                    global_heap.region_placement_is_enabled() &&
                    (requested_placement == RegionPlacement::Resident ||
                     requested_placement == RegionPlacement::Streaming)) {
                    const RegionPlacement fallback_placement =
                        requested_placement == RegionPlacement::Resident
                            ? RegionPlacement::Streaming
                            : RegionPlacement::Resident;
                    slot_populated =
                        regions[placement_index(fallback_placement)][bin].load(
                            std::memory_order_relaxed) != nullptr;
                }
                unlock_regions();
                if (slot_populated) {
                    continue;
                }
                return nullptr;
            }

            // Match the original allocator state transition: keep the
            // candidate's prior list state while on_memory_low() may yield,
            // then claim it immediately before publication.
            candidate->state.store(IN_USE, std::memory_order_relaxed);
            if (candidate->load_placement() == RegionPlacement::Streaming &&
                requested_group_id == 0) {
                global_heap.mark_streaming_region_shared(candidate);
            }

            auto &candidate_slot =
                regions[placement_index(candidate->load_placement())][bin];
            lock_regions();
            if (candidate_slot.load(std::memory_order_relaxed) != nullptr) {
                unlock_regions();
                diag.prevented_slot_overwrites.fetch_add(
                    1, std::memory_order_relaxed);

                // Another fibre populated this thread's slot while the current
                // fibre waited. The candidate has no published object entry,
                // so return it unused instead of exposing an unbound block to
                // GC.
                diag.region_acquires.fetch_add(1,
                                               std::memory_order_relaxed);
                return_region(candidate);
                diag.region_returns.fetch_add(1,
                                              std::memory_order_relaxed);
                return current().allocate_block_from_bin(
                    bin, obj, scope, requested_placement,
                    requested_group_id, requested_behavior_group_id,
                    pin_publication);
            }

            candidate_slot.store(candidate, std::memory_order_release);
            thread_heap_diag_registry().region_acquires.fetch_add(
                1, std::memory_order_relaxed);
            BlockHead *block = candidate->allocate(obj, pin_publication);
            unlock_regions();
            if (block != nullptr) [[likely]] {
                record_fallback_slot_allocation(
                    requested_placement, candidate->load_placement(), bin);
                return block;
            }
            // A freshly installed region should normally have capacity. Loop
            // through the same atomic detach path if it does not.
        }
    }

private:
    BlockHead *allocate_hotcold_block(
        size_t bin, cache::far_obj_t obj, cache::DereferenceScope *scope,
        RegionPlacement requested_placement, uint32_t requested_group_id,
        uint32_t requested_behavior_group_id, bool pin_publication) {
        // Hot/cold grouping is deliberately the small list-only variant.  It
        // does not participate in placement or in the fixed-six registry;
        // all local capacity is held in the semantic-class slots under the
        // existing six-slot matrix. Legacy mode populates only its 1/4
        // labels; six mode may populate every slot.
        (void)requested_placement;
        (void)requested_group_id;
        constexpr RegionPlacement placement = RegionPlacement::Unclassified;
        const size_t requested_class =
            simple_region_heat::normalize_class(requested_behavior_group_id);
        const auto fallback_order =
            simple_region_heat::fallback_order(requested_class);
        const size_t class_count = simple_region_budget::classes();

        auto retry_current = [&] {
            return current().allocate_hotcold_block(
                bin, obj, scope, requested_placement, requested_group_id,
                requested_behavior_group_id, pin_publication);
        };
        auto return_detached = [&](RegionHead *region, bool full) {
            if (full && thread_heap_full_return_batch_enabled()) {
                return_full_region_batched(region);
            } else {
                return_region(region);
            }
            thread_heap_diag_registry().region_returns.fetch_add(
                1, std::memory_order_relaxed);
        };

        // Reclassification is lazy for locally held Regions.  This helper
        // only touches the two hot/cold slots and must run while region_lock
        // is held; callers publish queued returns only after unlocking.
        auto rebucket_slots = [&](RegionHead **slots,
                                  RegionHead **detached,
                                  bool *detached_full,
                                  size_t &detached_count) {
            auto queue = [&](RegionHead *region, bool full) {
                if (region == nullptr) return;
                for (size_t i = 0; i < detached_count; ++i) {
                    if (detached[i] == region) return;
                }
                assert(detached_count < SixGroupSlotCount);
                detached[detached_count] = region;
                detached_full[detached_count] = full;
                ++detached_count;
            };

            RegionHead *held[SixGroupSlotCount]{};
            for (size_t slot = 0; slot < SixGroupSlotCount; ++slot) {
                held[slot] = slots[slot];
                slots[slot] = nullptr;
            }
            RegionHead *seen[SixGroupSlotCount]{};
            size_t seen_count = 0;
            for (RegionHead *region : held) {
                if (region == nullptr) continue;
                bool duplicate = false;
                for (size_t i = 0; i < seen_count; ++i)
                    duplicate |= seen[i] == region;
                if (duplicate) continue;
                assert(seen_count < SixGroupSlotCount);
                seen[seen_count++] = region;
                if (region->bin != bin ||
                    !region->placement_class_matches(placement)) {
                    thread_heap_diag_registry().slot_placement_mismatches
                        .fetch_add(1, std::memory_order_relaxed);
                    queue(region, !region->can_allocate());
                    continue;
                }
                const size_t actual_class = simple_region_heat::normalize_class(
                    simple_region_heat::allocation_class_for(
                        reinterpret_cast<uintptr_t>(region)));
                assert(actual_class < SixGroupSlotCount);
                if (slots[actual_class] == nullptr) {
                    slots[actual_class] = region;
                } else {
                    // Keep one bounded owner for the class.  A second Region
                    // is returned through the normal state-aware path.
                    queue(region, !region->can_allocate());
                }
            }
        };

        // A global exact-class miss may still be satisfied by this heap's
        // opposite-class Region.  Keep that preference explicit so a global
        // fallback candidate is never allocated and then discarded before a
        // private opposite-class hit is consumed.
        bool prefer_owned_opposite = false;
        bool stop_after_owned_opposite = false;
        for (;;) {
            // Fibre-local ThreadHeaps have one owner. The common case must
            // not take the management lock or rescan both slots per object.
            if (owner_fibre != nullptr &&
                simple_heat_epoch[bin] == simple_region_heat::classes().epoch()) {
                auto &fast_slots = six_group_regions[placement_index(placement)][bin];
                for (size_t i = 0; i < class_count; ++i) {
                    const size_t slot = fallback_order[i];
                    RegionHead *region = fast_slots[slot];
                    if (region == nullptr) continue;
                    BlockHead *fast_block = region->allocate(obj, pin_publication);
                    if (fast_block != nullptr) {
                        if (slot != requested_class) ++six_group_fallback_allocations;
                        return fast_block;
                    }
                    fast_slots[slot] = nullptr;
                    thread_heap_diag_registry().slot_exhaustions.fetch_add(
                        1, std::memory_order_relaxed);
                    return_detached(region, true);
                    if (&current() != this) return retry_current();
                    break;
                }
            }
            RegionHead *detached[SixGroupSlotCount]{};
            bool detached_full[SixGroupSlotCount]{};
            size_t detached_count = 0;
            BlockHead *block = nullptr;

            const bool management_lock = owner_fibre == nullptr;
            if (management_lock) lock_regions();
            auto &slots = six_group_regions[placement_index(placement)][bin];
            rebucket_slots(slots, detached, detached_full, detached_count);
            simple_heat_epoch[bin] = simple_region_heat::classes().epoch();

            // A normal miss asks the global heap for the requested class
            // first.  The owned opposite slot is consulted only after that
            // shared exact/free/new attempt fails (or returns its opposite
            // class), keeping fallback capacity bounded and deterministic.
            size_t local_class = requested_class;
            if (detached_count == 0 && slots[requested_class] == nullptr &&
                prefer_owned_opposite) {
                for (size_t i = 1; i < class_count; ++i) {
                    if (slots[fallback_order[i]] != nullptr) {
                        local_class = fallback_order[i];
                        break;
                    }
                }
            }
            RegionHead *region = detached_count == 0
                                     ? slots[local_class]
                                     : nullptr;
            if (region != nullptr) {
                block = region->allocate(obj, pin_publication);
                if (block == nullptr) {
                    slots[local_class] = nullptr;
                    assert(detached_count < SixGroupSlotCount);
                    detached[detached_count] = region;
                    detached_full[detached_count] = true;
                    ++detached_count;
                    thread_heap_diag_registry().slot_exhaustions.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
            if (management_lock) unlock_regions();

            for (size_t i = 0; i < detached_count; ++i) {
                return_detached(detached[i], detached_full[i]);
            }
            // Returning a Region may publish to a global list and yield.  Do
            // not allocate a block in this iteration before those returns;
            // retry through the current heap after every detached handoff.
            if (detached_count != 0) {
                if (&current() != this) return retry_current();
                continue;
            }
            if (block != nullptr) return block;

            if (prefer_owned_opposite) {
                prefer_owned_opposite = false;
                if (stop_after_owned_opposite) return nullptr;
            }

            auto &diag = thread_heap_diag_registry();
            scope_diag::refill(fibre_self());
            const uint64_t global_alloc_start = get_cycles();
            // GlobalHeap applies the bounded exact-class/free/new/opposite
            // policy.  The requested value is always one of the two class
            // hints, while allocation_class_for(candidate) remains
            // authoritative for the local slot chosen below.
            RegionHead *candidate = global_heap.allocate_region(
                bin, RegionPlacement::Unclassified, 0,
                static_cast<uint32_t>(requested_class));
            diag.global_region_alloc_calls.fetch_add(
                1, std::memory_order_relaxed);
            diag.global_region_alloc_cycles.fetch_add(
                get_cycles() - global_alloc_start,
                std::memory_order_relaxed);
            if (candidate != nullptr) {
                diag.region_acquires.fetch_add(1, std::memory_order_relaxed);
            }

            Cache *cache = Cache::get_default();
            const bool evacuator_waiting =
                cache != nullptr && cache->evacuator_waiting();
            const bool memory_low = global_heap.need_evacuate(false);
            if (candidate == nullptr ||
                (evacuator_waiting &&
                 diag.full_return_pending_current.load(
                     std::memory_order_relaxed) != 0)) {
                flush_all_thread_heap_pending_full_returns();
                if (candidate == nullptr) {
                    // Keep the existing bounded wait path: expose pending
                    // full Regions and let the caller initiate eviction when
                    // no local winner appears.
                    global_heap.drain_async_full_returns(64);
                }
            }
            if (memory_low || evacuator_waiting) [[unlikely]] {
                const uint64_t memory_low_start = get_cycles();
                diag.memory_low_calls.fetch_add(1, std::memory_order_relaxed);
                diag.memory_low_free_calls.fetch_add(
                    memory_low, std::memory_order_relaxed);
                diag.memory_low_evacuator_calls.fetch_add(
                    evacuator_waiting, std::memory_order_relaxed);
                global_heap.on_memory_low();
                if (cache != nullptr && scope != nullptr) {
                    cache->update_scope(*scope);
                }
                diag.memory_low_cycles.fetch_add(
                    get_cycles() - memory_low_start,
                    std::memory_order_relaxed);
            }

            // on_memory_low() can yield and resume this fibre on a different
            // OS thread.  Never publish a candidate into the old heap.
            if (&current() != this) [[unlikely]] {
                if (candidate == nullptr) return retry_current();
                candidate->state.store(IN_USE, std::memory_order_relaxed);
                diag.migrated_candidate_returns.fetch_add(
                    1, std::memory_order_relaxed);
                return_region(candidate);
                diag.region_returns.fetch_add(1, std::memory_order_relaxed);
                return retry_current();
            }

            if (candidate == nullptr) [[unlikely]] {
                // The shared exact/free/new lookup has failed.  Give this
                // heap's opposite-class capacity one bounded chance before
                // returning nullptr to the existing eviction/wait path.
                prefer_owned_opposite = true;
                stop_after_owned_opposite = true;
                continue;
            }

            candidate->state.store(IN_USE, std::memory_order_relaxed);
            const size_t candidate_class = simple_region_heat::normalize_class(
                simple_region_heat::allocation_class_for(
                    reinterpret_cast<uintptr_t>(candidate)));
            auto &candidate_slots =
                six_group_regions[placement_index(placement)][bin];
            const bool candidate_management_lock = owner_fibre == nullptr;
            if (candidate_management_lock) lock_regions();
            if (candidate_slots[candidate_class] != nullptr) {
                if (candidate_management_lock) unlock_regions();
                // Another owner populated the bounded class slot while the
                // global allocation was in flight.  Do not overwrite it or
                // repeatedly refill this heap.
                diag.prevented_slot_overwrites.fetch_add(
                    1, std::memory_order_relaxed);
                return_region(candidate);
                diag.region_returns.fetch_add(1, std::memory_order_relaxed);
                if (&current() != this) return retry_current();
                if (candidate_class != requested_class) {
                    prefer_owned_opposite = true;
                    stop_after_owned_opposite = false;
                }
                continue;
            }

            candidate_slots[candidate_class] = candidate;
            block = candidate->allocate(obj, pin_publication);
            if (block == nullptr) {
                candidate_slots[candidate_class] = nullptr;
                if (candidate_management_lock) unlock_regions();
                thread_heap_diag_registry().slot_exhaustions.fetch_add(
                    1, std::memory_order_relaxed);
                return_detached(candidate, true);
                if (&current() != this) return retry_current();
                continue;
            }
            if (candidate_management_lock) unlock_regions();
            return block;
        }
    }

    BlockHead *allocate_six_group_block(
        size_t bin, cache::far_obj_t obj, cache::DereferenceScope *scope,
        RegionPlacement requested_placement, uint32_t placement_group,
        uint32_t requested_group, bool pin_publication) {
        if (!global_heap.region_placement_is_enabled()) {
            requested_placement = RegionPlacement::Unclassified;
            placement_group = 0;
        }

        auto &registry = six_group::registry();
        if (requested_group == 0) {
            requested_group = registry.initial_group(0, bin);
        }
        std::array<uint32_t, SixGroupSlotCount> requested_family{};
        bool requested_family_ready = false;
        auto get_requested_family = [&]() -> const auto & {
            if (!requested_family_ready) {
                requested_family = registry.group_family(requested_group);
                requested_family_ready = true;
            }
            return requested_family;
        };

        auto retry_current = [&] {
            return current().allocate_six_group_block(
                bin, obj, scope, requested_placement, placement_group,
                requested_group, pin_publication);
        };
        auto placement_matches = [&](RegionHead *region,
                                     RegionPlacement placement) {
            return !FarLib::get_config().region_placement_bind_groups
                       ? region->placement_class_matches(placement)
                       : region->placement_matches(placement,
                                                   placement_group);
        };
        auto placement_order = [&] {
            std::array<RegionPlacement, 2> order{requested_placement,
                                                 requested_placement};
            size_t count = 1;
            if (global_heap.region_placement_is_enabled() &&
                (requested_placement == RegionPlacement::Resident ||
                 requested_placement == RegionPlacement::Streaming) &&
                !global_heap.can_allocate_region_directly_now(
                    bin, requested_placement)) {
                order[count++] =
                    requested_placement == RegionPlacement::Resident
                        ? RegionPlacement::Streaming
                        : RegionPlacement::Resident;
            }
            return std::pair{order, count};
        };
        auto return_detached = [&](RegionHead *region, bool full) {
            if (full && thread_heap_full_return_batch_enabled()) {
                return_full_region_batched(region);
            } else {
                return_region(region);
            }
            thread_heap_diag_registry().region_returns.fetch_add(
                1, std::memory_order_relaxed);
        };

        for (;;) {
            const auto [placements, placement_count] = placement_order();
            RegionHead *detached = nullptr;
            bool detached_full = false;
            for (size_t p = 0; p < placement_count && detached == nullptr;
                 ++p) {
                const RegionPlacement slot_placement = placements[p];
                auto &slots = six_group_regions[placement_index(slot_placement)]
                                                [bin];
                lock_regions();

                // Live group IDs are authoritative. Reclassification can
                // make the preferred class index stale, so scan the bounded
                // six-slot slice and shed only true duplicates.
                for (size_t i = 0; i < SixGroupSlotCount && detached == nullptr;
                     ++i) {
                    RegionHead *region = slots[i];
                    if (region == nullptr) continue;
                    if (region->bin != bin ||
                        !placement_matches(region, slot_placement)) {
                        slots[i] = nullptr;
                        detached = region;
                        thread_heap_diag_registry()
                            .slot_placement_mismatches.fetch_add(
                                1, std::memory_order_relaxed);
                        break;
                    }
                    const uint32_t group = region->six_group_id();
                    for (size_t j = 0; j < i; ++j) {
                        if (slots[j] != nullptr &&
                            slots[j]->six_group_id() == group) {
                            slots[i] = nullptr;
                            detached = region;
                            break;
                        }
                    }
                }

                BlockHead *block = nullptr;
                if (detached == nullptr) {
                    for (auto &region : slots) {
                        if (region == nullptr ||
                            region->six_group_id() != requested_group) {
                            continue;
                        }
                        auto *record = region->six_record.load(
                            std::memory_order_acquire);
                        if (record == nullptr) continue;
                        std::lock_guard<std::mutex> routing_guard(
                            record->routing_mutex);
                        if (six_group::group_of(record) != requested_group) {
                            continue;
                        }
                        block = region->allocate(obj, pin_publication);
                        if (block != nullptr) {
                            ++six_group_cache_hits;
                            record_fallback_slot_allocation(
                                requested_placement, slot_placement, bin);
                            break;
                        }
                        // A failed routing guard can leave capacity after a
                        // concurrent reclassification. Only FULL Regions are
                        // detached as exhaustion.
                        if (!region->can_allocate()) {
                            detached = region;
                            detached_full = true;
                            region = nullptr;
                            thread_heap_diag_registry()
                                .slot_exhaustions.fetch_add(
                                    1, std::memory_order_relaxed);
                            break;
                        }
                    }
                }
                unlock_regions();
                if (block != nullptr) return block;
            }

            if (detached != nullptr) {
                return_detached(detached, detached_full);
                if (&current() != this) return retry_current();
                continue;
            }
            ++six_group_cache_misses;

            auto &diag = thread_heap_diag_registry();
            auto get_region = [&](uint32_t group) {
                scope_diag::refill(fibre_self());
                const uint64_t start = get_cycles();
                RegionHead *region = global_heap.allocate_region(
                    bin, requested_placement, placement_group, group);
                diag.global_region_alloc_calls.fetch_add(
                    1, std::memory_order_relaxed);
                diag.global_region_alloc_cycles.fetch_add(
                    get_cycles() - start, std::memory_order_relaxed);
                if (region != nullptr) {
                    diag.region_acquires.fetch_add(
                        1, std::memory_order_relaxed);
                }
                return region;
            };
            auto get_family_region = [&](uint32_t &allocation_group,
                                         bool allow_family_fallback) {
                allocation_group = requested_group;
                RegionHead *candidate = get_region(allocation_group);
                if (candidate != nullptr || !allow_family_fallback) {
                    return candidate;
                }
                for (uint32_t group : get_requested_family()) {
                    if (group == 0 || group == requested_group) continue;
                    candidate = get_region(group);
                    if (candidate != nullptr) {
                        allocation_group = group;
                        return candidate;
                    }
                }
                return static_cast<RegionHead *>(nullptr);
            };

            uint32_t allocation_group = requested_group;
            RegionHead *candidate = get_family_region(allocation_group, false);
            if (&current() != this) {
                if (candidate != nullptr) {
                    candidate->state.store(IN_USE, std::memory_order_relaxed);
                    return_region(candidate);
                    diag.region_returns.fetch_add(
                        1, std::memory_order_relaxed);
                }
                return retry_current();
            }

            if (candidate == nullptr) {
                // A preferred-group shortage is not necessarily a capacity
                // shortage: an earlier soft fallback may already have left a
                // usable sibling Region in this fibre's bounded cache. Keep
                // exact routing preferred (including the global lookup above),
                // but consume that owned capacity before releasing every slot
                // and flushing all heaps. Otherwise each request for an absent
                // class repeatedly returns and reacquires the same Region.
                const auto [fallback_placements, fallback_count] = placement_order();
                for (size_t p = 0; p < fallback_count; ++p) {
                    const auto actual_placement = fallback_placements[p];
                    BlockHead *block = nullptr;
                    lock_regions();
                    for (auto *held : six_group_regions[
                             placement_index(actual_placement)][bin]) {
                        if (held == nullptr || held->bin != bin ||
                            !placement_matches(held, actual_placement)) continue;
                        auto *record = held->six_record.load(std::memory_order_acquire);
                        if (record == nullptr) continue;
                        std::lock_guard<std::mutex> routing_guard(record->routing_mutex);
                        // A cached, nonempty Region cannot be rebound to a
                        // different owner/bin family. Its stable cached family
                        // avoids taking the global registry lock on this path.
                        const auto &family = record->family_groups;
                        const auto actual_group = six_group::group_of(record);
                        if (std::find(family.begin(), family.end(), requested_group) == family.end() ||
                            std::find(family.begin(), family.end(), actual_group) == family.end()) continue;
                        block = held->allocate(obj, pin_publication);
                        if (block != nullptr) {
                            ++six_group_cache_hits;
                            if (actual_group != requested_group)
                                ++six_group_fallback_allocations;
                            break;
                        }
                    }
                    unlock_regions();
                    if (block != nullptr) {
                        record_fallback_slot_allocation(
                            requested_placement, actual_placement, bin);
                        return block;
                    }
                }
                // First expose capacity owned by this fibre, then flush and
                // drain the aligned async-return path before soft routing.
                release_all_regions();
                flush_all_thread_heap_pending_full_returns();
                global_heap.drain_async_full_returns(64);
                if (&current() != this) return retry_current();
                candidate = get_family_region(allocation_group, true);
            }
            if (candidate == nullptr) {
                // Idle fibres may own every usable Region. Reclaim their
                // bounded six-group slices only on this exhausted slow path.
                reclaim_six_group_regions_for_pressure();
                global_heap.drain_async_full_returns(64);
                if (&current() != this) return retry_current();
                candidate = get_family_region(allocation_group, true);
            }

            Cache *cache = Cache::get_default();
            const bool evacuator_waiting =
                cache != nullptr && cache->evacuator_waiting();
            const bool memory_low = global_heap.need_evacuate(false);
            if (candidate == nullptr ||
                (evacuator_waiting &&
                 diag.full_return_pending_current.load(
                     std::memory_order_relaxed) != 0)) {
                flush_all_thread_heap_pending_full_returns();
                if (candidate == nullptr) {
                    global_heap.drain_async_full_returns(64);
                }
            }
            if (candidate == nullptr || memory_low || evacuator_waiting) {
                const uint64_t memory_low_start = get_cycles();
                diag.memory_low_calls.fetch_add(1,
                                                std::memory_order_relaxed);
                diag.memory_low_free_calls.fetch_add(
                    memory_low, std::memory_order_relaxed);
                diag.memory_low_evacuator_calls.fetch_add(
                    evacuator_waiting, std::memory_order_relaxed);
                global_heap.on_memory_low();
                if (cache != nullptr && scope != nullptr) {
                    cache->update_scope(*scope);
                }
                diag.memory_low_cycles.fetch_add(
                    get_cycles() - memory_low_start,
                    std::memory_order_relaxed);
            }
            if (&current() != this) {
                if (candidate != nullptr) {
                    candidate->state.store(IN_USE, std::memory_order_relaxed);
                    return_region(candidate);
                    diag.region_returns.fetch_add(
                        1, std::memory_order_relaxed);
                }
                return retry_current();
            }
            if (candidate == nullptr) return nullptr;

            candidate->state.store(IN_USE, std::memory_order_relaxed);
            candidate->ensure_six_group(allocation_group);
            if (candidate->six_group_id() != allocation_group) {
                return_region(candidate);
                diag.region_returns.fetch_add(1,
                                              std::memory_order_relaxed);
                continue;
            }
            if (candidate->load_placement() == RegionPlacement::Streaming &&
                placement_group == 0) {
                global_heap.mark_streaming_region_shared(candidate);
            }

            for (;;) {
                const RegionPlacement actual_placement =
                    candidate->load_placement();
                auto &slots =
                    six_group_regions[placement_index(actual_placement)][bin];
                lock_regions();
                bool winner = false;
                for (RegionHead *held : slots) {
                    if (held != nullptr &&
                        held->six_group_id() == allocation_group) {
                        winner = true;
                        break;
                    }
                }
                if (winner) {
                    unlock_regions();
                    return_region(candidate);
                    diag.prevented_slot_overwrites.fetch_add(
                        1, std::memory_order_relaxed);
                    diag.region_returns.fetch_add(
                        1, std::memory_order_relaxed);
                    return retry_current();
                }

                const auto allocation_family =
                    registry.group_family(allocation_group);
                size_t preferred_slot = 0;
                while (preferred_slot < SixGroupSlotCount &&
                       allocation_family[preferred_slot] !=
                           allocation_group) {
                    ++preferred_slot;
                }
                size_t slot = preferred_slot;
                if (slot == SixGroupSlotCount || slots[slot] != nullptr) {
                    slot = 0;
                    while (slot < SixGroupSlotCount &&
                           slots[slot] != nullptr) {
                        ++slot;
                    }
                }
                if (slot == SixGroupSlotCount) {
                    slot = six_group_replacement
                               [placement_index(actual_placement)][bin]++ %
                           SixGroupSlotCount;
                    RegionHead *victim = slots[slot];
                    slots[slot] = nullptr;
                    unlock_regions();
                    return_region(victim);
                    diag.region_returns.fetch_add(
                        1, std::memory_order_relaxed);
                    if (&current() != this) {
                        return_region(candidate);
                        diag.region_returns.fetch_add(
                            1, std::memory_order_relaxed);
                        return retry_current();
                    }
                    continue;
                }

                slots[slot] = candidate;
                BlockHead *block = nullptr;
                auto *record = candidate->six_record.load(
                    std::memory_order_acquire);
                if (record != nullptr) {
                    std::lock_guard<std::mutex> routing_guard(
                        record->routing_mutex);
                    if (six_group::group_of(record) == allocation_group) {
                        block = candidate->allocate(obj, pin_publication);
                    }
                }
                if (block != nullptr &&
                    allocation_group != requested_group) {
                    ++six_group_fallback_allocations;
                }
                unlock_regions();
                if (block != nullptr) {
                    record_fallback_slot_allocation(
                        requested_placement, actual_placement, bin);
                    return block;
                }
                break;
            }
        }
    }

    std::atomic_flag region_lock = ATOMIC_FLAG_INIT;
    void *owner_fibre = nullptr;
    std::thread::id last_owner_thread;
    std::atomic<RegionHead *>
        regions[RegionPlacementCount][RegionBinCount];
    RegionHead *six_group_regions[RegionPlacementCount][RegionBinCount]
                                     [SixGroupSlotCount]{};
    uint64_t simple_heat_epoch[RegionBinCount]{};
    size_t six_group_replacement[RegionPlacementCount][RegionBinCount]{};
    uint64_t six_group_cache_hits = 0;
    uint64_t six_group_cache_misses = 0;
    uint64_t six_group_fallback_allocations = 0;
    RegionHead *pending_full_returns[RegionReturnBatchSize]{};
    size_t pending_full_return_count = 0;
};

GlobalHeap global_heap;

namespace {
bool fibre_thread_heap_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_FIBRE_HEAP");
        const bool result =
            value != nullptr && std::strtoull(value, nullptr, 0) != 0;
        std::cerr << "allocator.fibre_thread_heap_enabled=" << result
                  << std::endl;
        return result;
    }();
    return enabled;
}

void destroy_fibre_thread_heap(void *value) {
    if (value == nullptr) return;
    thread_heap_diag_registry().fibre_heap_destroys.fetch_add(
        1, std::memory_order_relaxed);
    delete static_cast<ThreadHeap *>(value);
}

size_t fibre_thread_heap_key() {
    static const size_t key =
        Fibre::key_create(&destroy_fibre_thread_heap);
    return key;
}

ThreadHeap &get_fibre_thread_heap() {
    Fibre *fibre = fibre_self();
    const size_t key = fibre_thread_heap_key();
    auto *heap = static_cast<ThreadHeap *>(fibre->getspecific(key));
    if (heap == nullptr) {
        heap = new ThreadHeap();
        heap->bind_fibre(fibre);
        fibre->setspecific(key, heap);
        thread_heap_diag_registry().fibre_heap_creates.fetch_add(
            1, std::memory_order_relaxed);
    } else {
        heap->bind_fibre(fibre);
    }
    return *heap;
}

ThreadHeap &get_os_thread_heap() {
    // Keep the fixed-six slot matrix out of ELF static TLS. libfibre creates
    // worker pthreads with PTHREAD_STACK_MIN, and placing the full ThreadHeap
    // in static TLS can make pthread_create fail before the application main
    // fibre runs. A TLS pointer retains one heap and its destructor per OS
    // thread without consuming the worker's small static-TLS allowance.
    static thread_local std::unique_ptr<ThreadHeap> heap;
    if (heap == nullptr) {
        heap = std::make_unique<ThreadHeap>();
    }
    return *heap;
}
}  // namespace

ThreadHeap &ThreadHeap::current() {
    return fibre_thread_heap_enabled() ? get_fibre_thread_heap()
                                       : get_os_thread_heap();
}
ThreadHeap &get_thread_heap() { return ThreadHeap::current(); }

ThreadHeap::ThreadHeap() {
    for (size_t placement = 0; placement < RegionPlacementCount;
         ++placement) {
        for (size_t i = 0; i < RegionBinCount; i++) {
            regions[placement][i].store(nullptr,
                                        std::memory_order_relaxed);
        }
    }
    auto &diag = thread_heap_diag_registry();
    {
        std::lock_guard<std::mutex> guard(diag.mutex);
        diag.live_heaps.insert(this);
    }
    diag.constructed_total.fetch_add(1, std::memory_order_relaxed);
}

ThreadHeap::~ThreadHeap() {
    // if the global heap is dead
    // maybe the heap is freed, so do nothing
    if (!global_heap.dead()) {
        release_all_regions();
    }
    auto &diag = thread_heap_diag_registry();
    {
        std::lock_guard<std::mutex> guard(diag.mutex);
        diag.live_heaps.erase(this);
    }
    diag.destroyed_total.fetch_add(1, std::memory_order_relaxed);
}

void release_current_thread_heap_regions() {
    ThreadHeap::current().release_all_regions();
}

size_t current_thread_six_group_cached_regions() {
    return ThreadHeap::current().six_group_cached_count();
}

void flush_all_thread_heap_pending_full_returns() {
    auto &diag = thread_heap_diag_registry();
    std::vector<RegionHead *> pending_full;
    {
        std::lock_guard<std::mutex> guard(diag.mutex);
        pending_full.reserve(diag.live_heaps.size() * RegionReturnBatchSize);
        for (ThreadHeap *heap : diag.live_heaps) {
            heap->detach_pending_full_returns(pending_full);
        }
    }
    ThreadHeap::publish_detached_full_regions(pending_full);
}

void reclaim_six_group_regions_for_pressure() {
    auto &diag = thread_heap_diag_registry();
    diag.six_pressure_reclaim_calls.fetch_add(
        1, std::memory_order_relaxed);
    std::vector<RegionHead *> active;
    std::vector<RegionHead *> pending_full;
    {
        // Holding the registry lock pins fibre-heap lifetimes. Each owner
        // lock is released before any list publication or possible yield.
        std::lock_guard<std::mutex> guard(diag.mutex);
        active.reserve(diag.live_heaps.size() * RegionPlacementCount *
                       RegionBinCount * ThreadHeap::SixGroupSlotCount);
        pending_full.reserve(diag.live_heaps.size() *
                             RegionReturnBatchSize);
        for (ThreadHeap *heap : diag.live_heaps) {
            heap->detach_six_group_regions_for_pressure(active,
                                                        pending_full);
        }
    }
    diag.six_pressure_reclaimed_regions.fetch_add(
        active.size() + pending_full.size(), std::memory_order_relaxed);
    ThreadHeap::publish_detached_regions(active, pending_full);
}

void release_all_thread_heap_regions_for_shutdown() {
    auto &diag = thread_heap_diag_registry();
    std::vector<RegionHead *> active;
    std::vector<RegionHead *> pending_full;
    {
        std::lock_guard<std::mutex> guard(diag.mutex);
        active.reserve(diag.live_heaps.size() * RegionPlacementCount *
                       RegionBinCount *
                       (ThreadHeap::SixGroupSlotCount + 1));
        pending_full.reserve(diag.live_heaps.size() *
                             RegionReturnBatchSize);
        for (ThreadHeap *heap : diag.live_heaps) {
            heap->detach_all_regions_for_shutdown(active, pending_full);
        }
    }
    ThreadHeap::publish_detached_regions(active, pending_full);
}

// Output-only diagnostic: counters already exist on normal runtime paths.
// No list traversal, lock acquisition, resets, or runtime decisions here.
void print_thread_heap_counters_only(const char *phase, size_t fibres) {
    auto &d = thread_heap_diag_registry();
    std::cerr << "allocator.stage_counters phase=" << phase
              << " fibres=" << fibres << " tsc=" << get_cycles();
#define PRINT_ALLOC_COUNTER(field) \
    std::cerr << " " #field "=" << d.field.load(std::memory_order_relaxed)
    PRINT_ALLOC_COUNTER(region_acquires);
    PRINT_ALLOC_COUNTER(region_returns);
    PRINT_ALLOC_COUNTER(global_region_alloc_calls);
    PRINT_ALLOC_COUNTER(global_region_alloc_cycles);
    PRINT_ALLOC_COUNTER(global_region_return_cycles);
    PRINT_ALLOC_COUNTER(full_return_batch_flushes);
    PRINT_ALLOC_COUNTER(full_return_batched_regions);
    PRINT_ALLOC_COUNTER(full_return_pending_cycles);
    PRINT_ALLOC_COUNTER(full_return_publish_cycles);
    PRINT_ALLOC_COUNTER(full_return_empty_flushes);
    PRINT_ALLOC_COUNTER(full_return_pending_current);
    PRINT_ALLOC_COUNTER(memory_low_calls);
    PRINT_ALLOC_COUNTER(memory_low_free_calls);
    PRINT_ALLOC_COUNTER(memory_low_evacuator_calls);
    PRINT_ALLOC_COUNTER(memory_low_cycles);
    PRINT_ALLOC_COUNTER(region_lock_contentions);
    PRINT_ALLOC_COUNTER(slot_exhaustions);
    PRINT_ALLOC_COUNTER(slot_placement_mismatches);
    PRINT_ALLOC_COUNTER(prevented_slot_overwrites);
    PRINT_ALLOC_COUNTER(migrated_candidate_returns);
    PRINT_ALLOC_COUNTER(fibre_heap_creates);
    PRINT_ALLOC_COUNTER(fibre_heap_destroys);
    PRINT_ALLOC_COUNTER(fibre_heap_os_migrations);
    PRINT_ALLOC_COUNTER(six_pressure_reclaim_calls);
    PRINT_ALLOC_COUNTER(six_pressure_reclaimed_regions);
#undef PRINT_ALLOC_COUNTER
    std::cerr << std::endl;
    global_heap.print_async_full_returns(phase, fibres);
}

void print_thread_heap_diagnostics(const char *phase, size_t iteration,
                                   size_t level) {
    auto &diag = thread_heap_diag_registry();
    std::unordered_set<RegionHead *> owner_refs;
    size_t owner_ref_slots = 0;
    size_t live_heaps = 0;
    uint64_t six_hits = 0;
    uint64_t six_misses = 0;
    uint64_t six_fallbacks = 0;
    uint64_t six_held = 0;
    {
        std::lock_guard<std::mutex> guard(diag.mutex);
        live_heaps = diag.live_heaps.size();
        for (auto *heap : diag.live_heaps) {
            heap->collect_region_refs(owner_refs, owner_ref_slots);
            heap->collect_six_group_stats(
                six_hits, six_misses, six_fallbacks, six_held);
        }
    }

    if (six_group::enabled()) {
        std::cerr << "allocator.six_group_cache phase=" << phase
                  << " iteration=" << iteration
                  << " level=" << level
                  << " hits=" << six_hits
                  << " misses=" << six_misses
                  << " fallbacks=" << six_fallbacks
                  << " held=" << six_held << std::endl;
        std::cerr << "allocator.six_pressure_reclaim calls="
                  << diag.six_pressure_reclaim_calls.load(
                         std::memory_order_relaxed)
                  << " regions="
                  << diag.six_pressure_reclaimed_regions.load(
                         std::memory_order_relaxed)
                  << std::endl;
    }

    size_t in_use_regions = 0;
    size_t full_regions = 0;
    size_t usable_regions = 0;
    size_t free_regions = 0;
    size_t referenced_in_use_regions = 0;
    size_t orphan_in_use_regions = 0;
    size_t referenced_non_in_use_regions = 0;
    uint64_t referenced_in_use_objects = 0;
    uint64_t orphan_in_use_objects = 0;
    uint64_t in_use_by_bin[RegionBinCount]{};

    auto *heap_base = static_cast<char *>(global_heap.get_heap());
    const size_t committed_regions =
        global_heap.get_committed_bytes() / RegionSize;
    for (size_t i = 0; i < committed_regions; ++i) {
        auto *region = reinterpret_cast<RegionHead *>(
            heap_base + i * RegionSize);
        const auto state = region->state.load(std::memory_order_relaxed);
        const bool referenced = owner_refs.contains(region);
        switch (state) {
        case IN_USE:
            ++in_use_regions;
            if (region->bin < RegionBinCount) {
                ++in_use_by_bin[region->bin];
            }
            if (referenced) {
                ++referenced_in_use_regions;
                referenced_in_use_objects += region->used_count;
            } else {
                ++orphan_in_use_regions;
                orphan_in_use_objects += region->used_count;
            }
            break;
        case FULL:
            ++full_regions;
            referenced_non_in_use_regions += referenced;
            break;
        case USABLE:
            ++usable_regions;
            referenced_non_in_use_regions += referenced;
            break;
        case FREE:
            ++free_regions;
            referenced_non_in_use_regions += referenced;
            break;
        }
    }

    std::cerr
        << "allocator.thread_heap_diag"
        << " phase=" << phase
        << " iteration=" << iteration
        << " level=" << level
        << " live_heaps=" << live_heaps
        << " constructed_total="
        << diag.constructed_total.load(std::memory_order_relaxed)
        << " destroyed_total="
        << diag.destroyed_total.load(std::memory_order_relaxed)
        << " fibre_heap_creates="
        << diag.fibre_heap_creates.load(std::memory_order_relaxed)
        << " fibre_heap_destroys="
        << diag.fibre_heap_destroys.load(std::memory_order_relaxed)
        << " fibre_heap_os_migrations="
        << diag.fibre_heap_os_migrations.load(std::memory_order_relaxed)
        << " region_acquires="
        << diag.region_acquires.load(std::memory_order_relaxed)
        << " region_returns="
        << diag.region_returns.load(std::memory_order_relaxed)
        << " release_all_calls="
        << diag.release_all_calls.load(std::memory_order_relaxed)
        << " slot_overwrites="
        << diag.slot_overwrites.load(std::memory_order_relaxed)
        << " prevented_slot_overwrites="
        << diag.prevented_slot_overwrites.load(std::memory_order_relaxed)
        << " migrated_candidate_returns="
        << diag.migrated_candidate_returns.load(std::memory_order_relaxed)
        << " region_lock_contentions="
        << diag.region_lock_contentions.load(std::memory_order_relaxed)
        << " global_region_alloc_calls="
        << diag.global_region_alloc_calls.load(std::memory_order_relaxed)
        << " global_region_alloc_cycles="
        << diag.global_region_alloc_cycles.load(std::memory_order_relaxed)
        << " global_region_return_cycles="
        << diag.global_region_return_cycles.load(std::memory_order_relaxed)
        << " full_return_batch_flushes="
        << diag.full_return_batch_flushes.load(std::memory_order_relaxed)
        << " full_return_batched_regions="
        << diag.full_return_batched_regions.load(std::memory_order_relaxed)
        << " full_return_pending_cycles="
        << diag.full_return_pending_cycles.load(std::memory_order_relaxed)
        << " full_return_publish_cycles="
        << diag.full_return_publish_cycles.load(std::memory_order_relaxed)
        << " full_return_empty_flushes="
        << diag.full_return_empty_flushes.load(std::memory_order_relaxed)
        << " full_return_pending_current="
        << diag.full_return_pending_current.load(std::memory_order_relaxed)
        << " memory_low_calls="
        << diag.memory_low_calls.load(std::memory_order_relaxed)
        << " memory_low_free_calls="
        << diag.memory_low_free_calls.load(std::memory_order_relaxed)
        << " memory_low_evacuator_calls="
        << diag.memory_low_evacuator_calls.load(std::memory_order_relaxed)
        << " memory_low_cycles="
        << diag.memory_low_cycles.load(std::memory_order_relaxed)
        << " slot_placement_mismatches="
        << diag.slot_placement_mismatches.load(std::memory_order_relaxed)
        << " slot_exhaustions="
        << diag.slot_exhaustions.load(std::memory_order_relaxed)
        << " fallback_slot_r_to_s_allocations="
        << diag.fallback_slot_allocations[0].load(std::memory_order_relaxed)
        << " fallback_slot_r_to_s_bytes="
        << diag.fallback_slot_bytes[0].load(std::memory_order_relaxed)
        << " fallback_slot_s_to_r_allocations="
        << diag.fallback_slot_allocations[1].load(std::memory_order_relaxed)
        << " fallback_slot_s_to_r_bytes="
        << diag.fallback_slot_bytes[1].load(std::memory_order_relaxed)
        << " region_list_lock_calls="
        << region_list_lock_diagnostics().list_lock_calls.load(
               std::memory_order_relaxed)
        << " region_list_lock_cycles="
        << region_list_lock_diagnostics().list_lock_cycles.load(
               std::memory_order_relaxed)
        << " region_placement_lock_calls="
        << region_list_lock_diagnostics().placement_lock_calls.load(
               std::memory_order_relaxed)
        << " region_placement_lock_cycles="
        << region_list_lock_diagnostics().placement_lock_cycles.load(
               std::memory_order_relaxed)
        << " overwritten_region_objects="
        << diag.overwritten_region_objects.load(std::memory_order_relaxed)
        << " owner_ref_slots=" << owner_ref_slots
        << " owner_ref_unique=" << owner_refs.size()
        << " committed_regions=" << committed_regions
        << " in_use_regions=" << in_use_regions
        << " referenced_in_use_regions=" << referenced_in_use_regions
        << " orphan_in_use_regions=" << orphan_in_use_regions
        << " referenced_non_in_use_regions="
        << referenced_non_in_use_regions
        << " referenced_in_use_objects=" << referenced_in_use_objects
        << " orphan_in_use_objects=" << orphan_in_use_objects
        << " full_regions=" << full_regions
        << " usable_regions=" << usable_regions
        << " free_regions=" << free_regions;
    for (size_t bin = 0; bin < RegionBinCount; ++bin) {
        if (in_use_by_bin[bin] != 0) {
            std::cerr << " in_use_bin_" << bin << '=' << in_use_by_bin[bin];
        }
        const uint64_t overwrites =
            diag.slot_overwrites_by_bin[bin].load(std::memory_order_relaxed);
        if (overwrites != 0) {
            std::cerr << " slot_overwrites_bin_" << bin << '=' << overwrites;
        }
    }
    std::cerr << std::endl;
}

__attribute_noinline__ BlockHead *thread_local_allocate(
    size_t size, cache::far_obj_t obj, cache::DereferenceScope *scope,
    RegionPlacement requested_placement, uint32_t requested_group_id,
    uint32_t requested_behavior_group_id, bool pin_publication) {
    return get_thread_heap().allocate(size, obj, scope, requested_placement,
                                      requested_group_id,
                                      requested_behavior_group_id,
                                      pin_publication);
}

#ifdef REMOTE_REGION_ALLOCATOR
namespace remote {
RemoteGlobalHeap remote_global_heap;
thread_local RemoteThreadHeap remote_thread_heap;
}  // namespace remote
#endif

}  // namespace allocator

}  // namespace FarLib

#endif
