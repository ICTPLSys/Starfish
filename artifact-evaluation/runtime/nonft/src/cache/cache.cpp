#include "cache/cache.hpp"

#include <sys/cdefs.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
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

namespace FarLib {
namespace cache {

std::unique_ptr<Cache> Cache::default_instance;

BatchedBackupBudget::Handle &current_backup_budget_handle() {
    static const size_t key = Fibre::key_create([](void *p) {
        delete static_cast<BatchedBackupBudget::Handle *>(p);
    });
    Fibre *fibre = fibre_self();
    auto *handle = static_cast<BatchedBackupBudget::Handle *>(fibre->getspecific(key));
    if (!handle) {
        handle = new BatchedBackupBudget::Handle();
        fibre->setspecific(key, handle);
    }
    return *handle;
}

BatchedBackupBudget::Handle &current_backup_thread_handle() {
    static thread_local BatchedBackupBudget::Handle handle;
    return handle;
}
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

class ThreadHeap {
public:
    ThreadHeap();

    ~ThreadHeap();

    static ThreadHeap &current();

    BlockHead *allocate(size_t size, cache::far_obj_t obj,
                        cache::DereferenceScope *scope,
                        RegionPlacement requested_placement,
                        uint32_t requested_group_id) {
        profile::start_allocate();
        BlockHead *block = allocate_block(size + sizeof(BlockHead), obj, scope,
                                          requested_placement,
                                          requested_group_id);
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

private:
    void detach_all_regions(std::vector<RegionHead *> &active,
                            std::vector<RegionHead *> &pending_full) {
        active.reserve(RegionPlacementCount * RegionBinCount);
        pending_full.reserve(RegionReturnBatchSize);
        lock_regions();
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
                              uint32_t requested_group_id) {
        size_t wsize = wsize_from_size(block_size);
        size_t bin = bin_from_wsize(wsize);
        assert(get_bin_size(bin) >= block_size);
        assert(bin == 0 || get_bin_size(bin - 1) < block_size + sizeof(void *));
        BlockHead *block = allocate_block_from_bin(
            bin, obj, scope, requested_placement, requested_group_id);
        profile::trace_alloc(block, bin);
        return block;
    }

    BlockHead *allocate_block_from_bin(size_t bin, cache::far_obj_t obj,
                                       cache::DereferenceScope *scope,
                                       RegionPlacement requested_placement,
                                       uint32_t requested_group_id) {
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
                        requested_group_id);
                }
                if (slot_placement == RegionPlacement::Streaming &&
                    requested_group_id == 0) {
                    global_heap.mark_streaming_region_shared(region);
                }
                BlockHead *block = region->allocate(obj);
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
                    requested_group_id);
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
                        requested_group_id);
                }
                candidate->state.store(IN_USE, std::memory_order_relaxed);
                diag.migrated_candidate_returns.fetch_add(
                    1, std::memory_order_relaxed);
                diag.region_acquires.fetch_add(1, std::memory_order_relaxed);
                return_region(candidate);
                diag.region_returns.fetch_add(1, std::memory_order_relaxed);
                return current().allocate_block_from_bin(
                    bin, obj, scope, requested_placement,
                    requested_group_id);
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
                    requested_group_id);
            }

            candidate_slot.store(candidate, std::memory_order_release);
            thread_heap_diag_registry().region_acquires.fetch_add(
                1, std::memory_order_relaxed);
            BlockHead *block = candidate->allocate(obj);
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
    std::atomic_flag region_lock = ATOMIC_FLAG_INIT;
    void *owner_fibre = nullptr;
    std::thread::id last_owner_thread;
    std::atomic<RegionHead *>
        regions[RegionPlacementCount][RegionBinCount];
    RegionHead *pending_full_returns[RegionReturnBatchSize]{};
    size_t pending_full_return_count = 0;
};

GlobalHeap global_heap;
static thread_local ThreadHeap thread_heap;

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
}  // namespace

ThreadHeap &ThreadHeap::current() {
    return fibre_thread_heap_enabled() ? get_fibre_thread_heap() : thread_heap;
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

void release_all_thread_heap_regions_for_shutdown() {
    auto &diag = thread_heap_diag_registry();
    std::vector<RegionHead *> active;
    std::vector<RegionHead *> pending_full;
    {
        std::lock_guard<std::mutex> guard(diag.mutex);
        active.reserve(diag.live_heaps.size() * RegionPlacementCount *
                       RegionBinCount);
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
    {
        std::lock_guard<std::mutex> guard(diag.mutex);
        live_heaps = diag.live_heaps.size();
        for (auto *heap : diag.live_heaps) {
            heap->collect_region_refs(owner_refs, owner_ref_slots);
        }
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
    RegionPlacement requested_placement, uint32_t requested_group_id) {
    return get_thread_heap().allocate(size, obj, scope, requested_placement,
                                      requested_group_id);
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
