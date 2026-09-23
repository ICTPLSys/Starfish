#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <vector>

#include "cache/entry.hpp"
#include "cache/base/object.hpp"
#include "utils/control.hpp"
#include "utils/alloc_reclaim_rate_diag.hpp"
#include "utils/debug.hpp"
#include "utils/stats.hpp"
#include "utils/bounded_return_queue.hpp"

// Debug (compile-time opt-in; zero cost when disabled)
//
// - FARLIB_ALLOC_DEBUG:
//   Enables debug-only invariants, structured logs (non-interleaving), and tags.
// - FARLIB_ALLOC_DEBUG_FENCE:
//   Adds debug-only fences around publish points to help diagnose ordering issues.
//
// Enable via build system, e.g. CMake:
//   target_compile_definitions(<target> PRIVATE FARLIB_ALLOC_DEBUG FARLIB_ALLOC_DEBUG_FENCE)
#ifdef FARLIB_ALLOC_DEBUG
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#endif
namespace FarLib {

namespace allocator {

#ifdef FARLIB_ALLOC_DEBUG
inline void alloc_debug_printf(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    size_t len = (static_cast<size_t>(n) < sizeof(buf)) ? static_cast<size_t>(n)
                                                        : (sizeof(buf) - 1);
    (void)::write(2, buf, len);
}
#endif

// set to true to prioritize large objects for evacuation
constexpr bool EVACUATE_PRIORITY_LARGE_OBJ = false;
// only valid when EVACUATE_PRIORITY_LARGE_OBJ is true
// objects smaller than this threshold will not be evacuated (will be pinned in the local memory)
constexpr size_t OBJECT_SIZE_THRESHOLD = 256;

constexpr size_t RegionSize = 256 * 1024;

constexpr size_t RegionBinCount = 56;

constexpr size_t BinSize[RegionBinCount] = {
    1,    2,    3,    4,    5,    6,    7,    8,    10,   12,   14,   16,
    20,   24,   28,   32,   40,   48,   56,   64,   80,   96,   112,  128,
    160,  192,  224,  256,  320,  384,  448,  512,  640,  768,  896,  1024,
    1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192,
    10240, 12288, 14336, 16384, 20480, 24576, 28672, 32768
};

inline size_t get_smaller_limit() {
    if (OBJECT_SIZE_THRESHOLD > 0) {
        for (size_t i = 0; i < RegionBinCount; i++) {
            if (BinSize[i] * 8 >= OBJECT_SIZE_THRESHOLD) {
                return i;
            }
        }
    }
    return 0;
}

constexpr size_t MaxBinSize = BinSize[RegionBinCount - 1];
inline constexpr size_t get_bin_size(size_t bin) {
    return BinSize[bin] * sizeof(void *);
}

// aligned to sizeof(void *)
inline constexpr size_t wsize_from_size(size_t size) {
    return (size + sizeof(void *) - 1) / sizeof(void *);
}

// bit scan reverse: return the index of the highest bit
inline uint8_t bsr32(uint32_t x) { return 31 - __builtin_clz(x); }

// get a proper bin from a specific wsize
inline static size_t bin_from_wsize(size_t wsize) {
    if (wsize <= 8) {
        if (wsize <= 1) [[unlikely]] {
            return 0;
        }
        // round to double word sizes
        return (wsize - 1) | 1;
    } else {
        if (wsize > MaxBinSize) [[unlikely]] {
            // size >= MaxBinSize, too large
            ERROR("too large to allocate");
        }
        wsize -= 1;
        uint8_t b = bsr32(static_cast<uint32_t>(wsize));
        // (~16% worst internal fragmentation)
        return ((b << 2) + (uint8_t)((wsize >> (b - 2)) & 0x03)) - 4;
    }
}

// actually the allocator allocates blocks
// a block include its header (metadata) and the user data
struct BlockHead {
    BlockHead *next;
    std::atomic<cache::far_obj_t> obj_meta_data;

    void *get_object_ptr() { return static_cast<void *>(this + 1); }
};

enum RegionState {
    FREE,    // this region has no block allocated
    IN_USE,  // this region belongs to a thread, and is used for allocation
    USABLE,  // this region has free blocks, and is in the global region list
    FULL,    // this region is exhausted
};

enum class RegionPlacement : uint8_t {
    Unclassified = 0,
    Resident = 1,
    Streaming = 2,
    Mixed = 3,
};

enum class EvacuationEligibility : uint8_t {
    All = 0,
    LegacyConcurrent = 1,
};

inline constexpr size_t RegionPlacementCount = 4;
inline constexpr size_t RegionFallbackDirectionCount = 2;

inline constexpr size_t placement_index(RegionPlacement placement) {
    return static_cast<size_t>(placement);
}

struct RegionListNodeBase {
    RegionListNodeBase() = default;
    RegionListNodeBase(RegionListNodeBase *n) : next_region(n) {}

    RegionListNodeBase *next_region;
};

template <typename Fn>
concept BlockInvoker = requires(Fn &&f, BlockHead *block) {
    { f(block) } -> std::same_as<cache::EntryState>;
};

// Only one thread can allocate from the region
// But any threads can deallocate blocks in it
// A region includes a header (metadata) and blocks to be allocate
struct alignas(64) RegionHead : public RegionListNodeBase {
    std::atomic<RegionState> state;
    std::atomic<RegionPlacement> placement;
    std::atomic<uint32_t> placement_group_id;
    std::atomic<uint64_t> placement_epoch;
    std::atomic_bool placement_list_owned;
    std::atomic_flag placement_lock;
    std::atomic_flag requeue_lock;
    std::atomic<uint32_t> pending_reclaims;
    BlockHead *free_list;
    BlockHead *active_list;
    BlockHead *marked_list;
    uint32_t used_count;
    uint32_t unused_offset;
    uint32_t bin;
    // used for multithread mark & evict
    uint32_t sweep_time_stamp;
    std::atomic<uint32_t> last_mark_epoch;
    std::atomic<uint32_t> evict_drained_epoch;
    std::atomic<uint32_t> evict_processed_epoch;
    std::atomic<uint64_t> profile_window_read_references;
    std::atomic<uint64_t> profile_window_write_references;
    std::atomic<uint64_t> profile_ema_read_references;
    std::atomic<uint64_t> profile_ema_write_references;
    std::atomic<uint64_t> profile_total_read_references;
    std::atomic<uint64_t> profile_total_write_references;
    std::atomic<uint64_t> profile_window_resident_fetch_intent_bytes;
    std::atomic<uint64_t> profile_ema_resident_fetch_intent_bytes;
    std::atomic<uint8_t> profile_last_target_placement;
    std::atomic<uint8_t> profile_target_streak;
    std::atomic<uint8_t> profile_pending_target_placement;
    std::atomic<uint64_t> profile_pending_plan_id;
    std::atomic<uint32_t> profile_pending_plan_rank;
#ifdef FARLIB_ALLOC_DEBUG
    // Debug-only: track which RegionList currently owns/enqueues this region.
    // This catches double-enqueue even across different lists.
    std::atomic<void *> dbg_enqueued_list;
    const char *dbg_last_push_tag;
#endif

    void init(size_t bin, uint64_t new_placement_epoch);

    void lock_placement();
    void unlock_placement();
    RegionPlacement load_placement() const;
    uint32_t load_placement_group_id() const;
    uint64_t load_placement_epoch() const;
    bool placement_class_matches(RegionPlacement expected_placement) const;
    bool placement_matches(RegionPlacement expected_placement,
                           uint32_t expected_group_id) const;
    void claim_placement(RegionPlacement new_placement,
                         uint32_t new_group_id,
                         uint64_t new_placement_epoch);
    void reset_placement(uint64_t new_placement_epoch);
    bool try_set_profile_pending_target(
        uint64_t expected_placement_epoch,
        RegionPlacement expected_placement,
        RegionPlacement target_placement, uint64_t plan_id,
        uint32_t plan_rank);
    bool clear_profile_pending_target(uint64_t expected_plan_id);
    bool profile_pending_resident() const;
    bool profile_pending_plan_matches(uint64_t plan_id) const;
    uint32_t load_profile_pending_plan_rank() const;

    // not thread safe
    // return nullptr on exhausted
    BlockHead *allocate(cache::far_obj_t obj);

    void note_mark_epoch(uint32_t epoch);
    void note_evict_drained_epoch(uint32_t epoch);
    void note_evict_processed_epoch();
    bool evict_processed_for_last_mark() const;

    // not thread safe
    bool can_allocate() const;

    size_t free_size() const;

    bool is_empty() const;

    bool has_deallocated_blocks() const;
    size_t reclaim_deallocated_blocks(bool force = false);

    // not thread safe
    // traverse active list
    // return: freed size
    template <BlockInvoker Fn>
    void mark(Fn &&fn);

    // traverse marked list
    template <BlockInvoker Fn>
    void evict(Fn &&fn);
};

// A RegionList snapshot is only a discovery hint. The list may release and
// later reuse the Region after the list lock is dropped, so consumers must
// revalidate both fields while holding the Region placement lock.
struct RegionListSnapshotHandle {
    RegionHead *region = nullptr;
    uint64_t placement_epoch = 0;
    RegionPlacement placement = RegionPlacement::Unclassified;
};

// concurrent double linked list
struct MarkBreakdownStats {
    size_t held_slot = std::numeric_limits<size_t>::max();
    uint32_t sample_state = 1;
    uint32_t pop_countdown = 1;
    uint32_t try_countdown = 1;
    bool segment_enabled = false;
    uint64_t segment_body_cycles = 0;
    uint64_t pop_whole_calls = 0;
    uint64_t pop_whole_cycles = 0;
    uint64_t region_mark_whole_calls = 0;
    uint64_t region_mark_whole_cycles = 0;
    uint64_t post_push_whole_calls = 0;
    uint64_t post_push_whole_cycles = 0;
    uint64_t push_list_lock_calls = 0;
    uint64_t push_list_lock_wait_cycles = 0;
    uint64_t push_list_lock_hold_cycles = 0;
    uint64_t push_placement_lock_cycles = 0;
    uint64_t pop_calls = 0;
    uint64_t nodes_examined = 0;
    uint64_t stamped_skips = 0;
    uint64_t eligible_calls = 0;
    uint64_t eligible_false = 0;
    uint64_t eligible_pending_resident = 0;
    uint64_t eligible_in_use = 0;
    uint64_t eligible_resident_no_reclaim = 0;
    uint64_t eligible_stable = 0;
    uint64_t eligible_inadmissible = 0;
    uint64_t regions_popped = 0;
    uint64_t regions_marked = 0;
    uint64_t resident_placements_skipped = 0;
    uint64_t pop_samples = 0;
    uint64_t list_lock_wait_cycles = 0;
    uint64_t list_lock_wait_max_cycles = 0;
    uint64_t list_lock_hold_cycles = 0;
    uint64_t list_lock_hold_max_cycles = 0;
    uint64_t try_calls = 0;
    uint64_t try_samples = 0;
    uint64_t try_sample_cycles = 0;
    uint64_t try_sample_max_cycles = 0;
    uint64_t try_invalid_retries = 0;
    uint64_t try_cas_retries = 0;
    uint64_t try_demote_retries = 0;
    uint64_t try_yields = 0;
    uint64_t try_retry_cap = 0;
    uint64_t try_results[8] = {};

    uint32_t next_interval() {
        sample_state = sample_state * 1664525u + 1013904223u;
        return 1u + (sample_state % 127u);
    }
    bool sample_pop() {
        if (--pop_countdown != 0) return false;
        pop_countdown = next_interval();
        return true;
    }
    bool sample_try() {
        if (--try_countdown != 0) return false;
        try_countdown = next_interval();
        return true;
    }
};

// Per-marker state for an optional round-local early-stop request. The stop
// is observed only after this worker has completed and published one full
// nonempty batch, so no owned Region is abandoned and every participating
// marker makes progress while round budget/work remains.
struct MarkStopContext {
    std::atomic_bool *requested = nullptr;
    size_t completed_batches = 0;
    size_t completed_regions = 0;
    bool stopped = false;

    bool should_stop() const {
        return requested != nullptr && completed_batches != 0 &&
               requested->load(std::memory_order_acquire);
    }
};

// Per-evict-fibre collector-only diagnostic state. It follows the explicit
// collect call chain and is never recovered from OS-thread-local storage.
struct EvictCollectorDiag {
    size_t held_slot = std::numeric_limits<size_t>::max();
    bool segment_enabled = false;
    uint64_t calls = 0;
    uint64_t nodes_examined = 0;
    uint64_t stamped_skips = 0;
    uint64_t eligible_false = 0;
    uint64_t selected = 0;
    uint64_t list_lock_wait_cycles = 0;
    uint64_t list_lock_hold_cycles = 0;
    uint64_t list_lock_hold_max_cycles = 0;
    uint64_t cursor_more_returns = 0;
    uint64_t cursor_visit_budget_exhaustions = 0;
    uint64_t selected_unmarked_nonempty = 0;
    uint64_t processed_regions = 0;
    uint64_t processing_zero_reclaim = 0;
    uint64_t processing_cycles = 0;
    uint64_t processing_max_cycles = 0;
    uint64_t reclaimed_bytes = 0;
};

struct GcProcessedFilterStats {
    uint64_t accepted_checks = 0;
    uint64_t rejected_checks = 0;
};

// concurrent double linked list
struct RegionList {
    enum PopUnmatchedReason {
        PopUnmatchedEmpty = 0,
        PopUnmatchedHeadMatched = 1,
        PopUnmatchedSuccess = 2,
    };
    std::atomic_flag lock;
    RegionListNodeBase head;
    RegionListNodeBase *tail;
    std::atomic_size_t item_count;

    // Shared, lock-protected progress for one LegacyConcurrent mark epoch.
    // scan_prev is the predecessor of the next original node to inspect;
    // scan_last is the tail captured when the epoch starts.  Appends and
    // prepends are deliberately outside that captured epoch.
    uint32_t mark_scan_epoch;
    bool mark_scan_initialized;
    bool mark_scan_done;
    RegionListNodeBase *mark_scan_prev;
    RegionListNodeBase *mark_scan_last;

    // Independent shared progress for LegacyConcurrent evict selection.
    // The boundary is captured once per epoch, so tail appends are deferred
    // to the next epoch instead of extending the current scan indefinitely.
    uint64_t evict_scan_epoch;
    bool evict_scan_initialized;
    bool evict_scan_done;
    RegionListNodeBase *evict_scan_prev;
    RegionListNodeBase *evict_scan_last;

    // Direct GC has an independent cursor. Collector and GC may run in the
    // same round and must not consume one another's scan position.
    uint64_t gc_scan_epoch;
    bool gc_scan_initialized;
    bool gc_scan_done;
    RegionListNodeBase *gc_scan_prev;
    RegionListNodeBase *gc_scan_last;

    RegionList();

    void push(RegionHead *node);
    void push_dbg(RegionHead *node, const char *tag,
                  MarkBreakdownStats *mark_diag = nullptr);
    void push_batch_dbg(RegionHead **nodes, size_t count, const char *tag);
    RegionHead *pop();
    template <typename EligibleFn>
    RegionHead *pop_first_matching(EligibleFn &&eligible);
    template <typename EligibleFn>
    size_t extract_all_matching(std::vector<RegionHead *> &out,
                                EligibleFn &&eligible);
    void push_front_batch_dbg(RegionHead **nodes, size_t count,
                              const char *tag);
    RegionHead *pop_unmatched(uint32_t time_stamp, int *reason = nullptr);

    template <typename EligibleFn>
    size_t pop_matching_batch(uint32_t time_stamp, RegionHead **out_batch,
                              size_t max_count, int *reason,
                              EligibleFn &&eligible,
                              MarkBreakdownStats *mark_diag = nullptr,
                              EvictCollectorDiag *evict_diag = nullptr);

    template <typename EligibleFn>
    // Calls for different epochs must not overlap on the same list.  The
    // LegacyConcurrent master satisfies this by joining its mark workers
    // before advancing the evacuation timestamp.
    size_t pop_matching_batch_cursor(uint32_t time_stamp,
                                     RegionHead **out_batch,
                                     size_t max_count, int *reason,
                                     EligibleFn &&eligible,
                                     MarkBreakdownStats *mark_diag = nullptr);

    template <typename EligibleFn>
    // Calls for different evict epochs must not overlap on one list. The
    // LegacyConcurrent master joins all evict workers before advancing the
    // shared reclamation timestamp.
    size_t pop_matching_batch_evict_cursor(
        uint32_t time_stamp, RegionHead **out_batch, size_t max_count,
        size_t visit_limit, int *reason, bool *has_unscanned,
        bool *visit_budget_exhausted, EligibleFn &&eligible,
        EvictCollectorDiag *evict_diag = nullptr,
        uint64_t scan_generation = 0, bool gc_cursor = false,
        size_t *visited_out = nullptr);

    template <typename EligibleFn>
    RegionHead *pop_matching(uint32_t time_stamp, int *reason,
                             EligibleFn &&eligible);

    // Batch variant of pop_unmatched():
    // pop up to max_count regions whose sweep timestamp is not equal to time_stamp
    // under a single lock hold, reducing lock contention for parallel workers.
    size_t pop_unmatched_batch(uint32_t time_stamp, RegionHead **out_batch,
                               size_t max_count, int *reason = nullptr,
                               uint32_t max_mark_epoch = 0,
                               MarkBreakdownStats *mark_diag = nullptr,
                               EvictCollectorDiag *evict_diag = nullptr);

    size_t size() const {
        return item_count.load(std::memory_order_acquire);
    }

    bool nonempty() const {
        return size() != 0;
    }

    void append_region_snapshot(
        std::vector<RegionListSnapshotHandle> &regions);

private:
    bool mark_cursor_active_locked() const;
    void finish_mark_cursor_locked();
    void mark_cursor_on_unlink_locked(RegionListNodeBase *prev,
                                      RegionListNodeBase *node);
    void mark_cursor_on_prepend_locked(RegionListNodeBase *new_prefix_last);
    bool evict_cursor_active_locked() const;
    void finish_evict_cursor_locked();
    void evict_cursor_on_unlink_locked(RegionListNodeBase *prev,
                                       RegionListNodeBase *node);
    void evict_cursor_on_prepend_locked(RegionListNodeBase *new_prefix_last);
    bool gc_cursor_active_locked() const;
    void finish_gc_cursor_locked();
    void gc_cursor_on_unlink_locked(RegionListNodeBase *prev,
                                    RegionListNodeBase *node);
    void gc_cursor_on_prepend_locked(RegionListNodeBase *new_prefix_last);
};

inline constexpr size_t RegionReturnBatchSize = 8;

struct EvictTask {
    RegionHead *region = nullptr;
    RegionList *source_list = nullptr;
    uint32_t epoch = 0;
    RegionPlacement placement = RegionPlacement::Unclassified;
    uint32_t placement_group_id = 0;
    uint64_t placement_epoch = 0;
    bool reserved_free_size = false;

    EvictTask() = default;
    EvictTask(RegionHead *region, RegionList *source_list, uint32_t epoch,
              bool reserved_free_size = false)
        : region(region),
          source_list(source_list),
          epoch(epoch),
          placement(region == nullptr ? RegionPlacement::Unclassified
                                      : region->load_placement()),
          placement_group_id(region == nullptr
                                 ? 0
                                 : region->load_placement_group_id()),
          placement_epoch(region == nullptr ? 0
                                            : region->load_placement_epoch()),
          reserved_free_size(reserved_free_size) {}

    bool placement_still_matches() const {
        return region != nullptr &&
               region->load_placement() == placement &&
               (!FarLib::get_config().region_placement_bind_groups ||
                region->load_placement_group_id() == placement_group_id) &&
               region->load_placement_epoch() == placement_epoch;
    }
};

void release_current_thread_heap_regions();
void release_all_thread_heap_regions_for_shutdown();

void print_thread_heap_diagnostics(const char *phase, size_t iteration,
                                   size_t level);

struct GlobalHeapBinSnapshot {
    size_t target_usable_regions = 0;
    size_t free_regions = 0;
    size_t full_regions = 0;
    size_t total_usable_regions = 0;
    size_t committed_regions = 0;
    size_t unallocated_regions = 0;
    bool can_allocate_now = false;
};

struct RegionPlacementGroupSnapshot {
    size_t total_regions = 0;
    size_t resident_regions = 0;
    size_t streaming_regions = 0;
    size_t transitionable_regions = 0;
};

struct RegionHotnessSnapshot {
    RegionHead *region = nullptr;
    uint64_t placement_epoch = 0;
    RegionPlacement placement = RegionPlacement::Unclassified;
    RegionState state = FREE;
    uint32_t bin = 0;
    uint64_t free_bytes = 0;
    uint64_t resident_fetch_intent_bytes = 0;
    uint64_t read_references = 0;
    uint64_t write_references = 0;
    uint64_t weighted_references = 0;
    uint64_t total_weighted_references = 0;
    bool list_owned = false;
};

struct RegionReclassifyResult {
    bool changed = false;
    bool transitionable = false;
    size_t changed_bytes = 0;
};

enum class RegionExchangeFailure : uint8_t {
    None = 0,
    InvalidInput,
    HotInvalidRegionState,
    ColdInvalidRegionState,
    ColdRegionUnavailable,
    HotRegionUnavailable,
    RegionChanged,
    ColdEntryBusy,
    HotEntryBusy,
    Count,
};

struct RegionExchangeResult {
    bool changed = false;
    size_t promoted_bytes = 0;
    size_t demoted_bytes = 0;
    RegionExchangeFailure failure = RegionExchangeFailure::None;
};

using RegionPlacementTransitionFn = std::function<void(
    cache::FarObjectEntry &, size_t, RegionPlacement, RegionPlacement)>;

struct ReclaimSupplySnapshot {
    size_t usable_regions = 0;
    size_t free_regions = 0;
};

class GlobalHeap {
private:
    BoundedReturnQueue<RegionHead, 512> async_full_returns_;
    std::atomic<uint64_t> async_full_return_drain_cycles_{0};
    std::atomic<uint64_t> async_full_return_wakes_{0};
    RegionList usable_region_list[RegionPlacementCount][RegionBinCount];
    RegionList full_region_list[RegionPlacementCount];
    RegionList free_region_list;
    std::atomic_size_t unallocated_offset;
    size_t heap_size;
    int64_t memory_low_water_mark;
    int64_t memory_high_water_mark;
    int64_t memory_release_water_mark;
    void *heap;
    std::atomic_bool is_dead;
    std::atomic_int64_t free_size;
    std::atomic<uint64_t> next_placement_epoch;
    std::atomic_size_t resident_reserved_regions;
    std::atomic_size_t streaming_reserved_regions;
    std::atomic_int64_t resident_counted_free_size;
    std::atomic<uint64_t> full_to_usable_after_mark_count{0};
    std::atomic<uint64_t> full_region_free_size_violation_count{0};
    std::atomic<uint64_t> shutdown_full_origin_free_bytes{0};
    std::atomic<uint64_t>
        placement_fallback_attempts[RegionFallbackDirectionCount]
                                   [RegionBinCount]{};
    std::atomic<uint64_t>
        placement_fallback_successes[RegionFallbackDirectionCount]
                                    [RegionBinCount]{};
    std::atomic<uint64_t>
        placement_fallback_capacity_bytes[RegionFallbackDirectionCount]
                                         [RegionBinCount]{};
    size_t resident_region_budget;
    bool region_placement_enabled;

    uint64_t allocate_placement_epoch();
    void account_region_claim(RegionPlacement placement);
    void account_region_release(RegionPlacement placement);
    void add_resident_counted_free_size(int64_t bytes);
    void sub_resident_counted_free_size(int64_t bytes);
    void reclaim_resident_region(RegionHead *region, RegionList &source_list);

public:
    std::function<void(void)> on_memory_low = [] {};

public:
    GlobalHeap();

    void register_heap(void *buffer, size_t size);

    void destroy();

    bool dead() const;

    void set_on_memory_low(std::function<void(void)> fn);

    bool need_evacuate(bool evacuator_waiting);

    // allocate a region with free blocks
    // this region may not be empty
    // return nullptr if failed to allocate
    // return a locked region if success
    RegionHead *allocate_region(
        size_t bin,
        RegionPlacement requested_placement = RegionPlacement::Unclassified,
        uint32_t requested_group_id = 0);

    RegionPlacement resolve_allocation_placement(
        RegionPlacement requested_placement,
        uint32_t requested_group_id) const;

    void add_free_size(int64_t size);

    void sub_free_size(int64_t size);

    inline void check_free_size_negative_after_alloc(
        const char *where, size_t bin, void *region_ptr, int64_t sub_bytes,
        int64_t region_free_bytes, size_t unalloc_off, size_t heap_sz);

    RegionHead *allocate_region_impl(size_t bin,
                                     RegionPlacement requested_placement,
                                     uint32_t requested_group_id);

    void return_back_region(RegionHead *region);
    void return_back_full_regions_batch(RegionHead **regions, size_t count);
    static bool async_full_returns_enabled();
    void submit_full_regions_batch(RegionHead **regions, size_t count);
    size_t drain_async_full_returns(size_t limit = 64);
    bool has_async_full_returns() const;
    void print_async_full_returns(const char *phase, size_t fibres = 0) const;
    void finish_async_full_returns_for_shutdown();
    void publish_owned_evict_region(RegionHead *region);
    void requeue_evict_task(EvictTask &task);
    bool adopt_collected_evict_task(EvictTask &task);
    void reclaim_all_deallocated_regions();
    void verify_no_detached_regions_for_shutdown() const;
    void mark_streaming_region_shared(RegionHead *region);

    bool region_placement_is_enabled() const;
    size_t get_resident_region_budget() const;
    size_t get_resident_reserved_regions() const;
    size_t get_streaming_reserved_regions() const;
    void get_region_placement_group_snapshots(
        std::vector<RegionPlacementGroupSnapshot> &snapshots) const;
    void collect_region_hotness_snapshots(
        std::vector<RegionHotnessSnapshot> &snapshots,
        size_t write_weight, size_t ema_decay_shift);
    size_t prioritize_profile_pending_regions(uint64_t plan_id);
    RegionReclassifyResult try_reclassify_hotness_region(
        RegionHead *region, uint64_t expected_placement_epoch,
        RegionPlacement from, RegionPlacement to,
        const RegionPlacementTransitionFn &on_entry_transition = {});
    RegionExchangeResult try_exchange_hotness_regions(
        RegionHead *hot_streaming_region,
        uint64_t expected_hot_placement_epoch,
        RegionHead *cold_resident_region,
        uint64_t expected_cold_placement_epoch,
        const RegionPlacementTransitionFn &on_entry_transition = {});
    size_t reclassify_region_placement_group(
        uint32_t placement_group_id, RegionPlacement from,
        RegionPlacement to, size_t max_regions);

    void *get_heap();

    size_t get_heap_size() const;
    int64_t get_free_bytes() const;
    int64_t get_used_bytes() const;
    size_t get_committed_bytes() const;
    ReclaimSupplySnapshot reclaim_supply_snapshot(size_t bin) const;

    void print_used_memory();

    template <BlockInvoker Fn>
    size_t mark_list(Fn &&fn, RegionList &list, uint32_t timestamp,
                     std::vector<EvictTask> *ready_tasks = nullptr,
                     std::atomic_size_t *ready_task_budget = nullptr,
                     bool legacy_cursors = false,
                     EvacuationEligibility eligibility =
                         EvacuationEligibility::All,
                     MarkBreakdownStats *mark_diag = nullptr,
                     std::atomic_size_t *legacy_region_budget = nullptr,
                     MarkStopContext *stop_context = nullptr);

    template <BlockInvoker Fn>
    size_t mark(Fn &&fn, uint32_t timestamp,
                std::vector<EvictTask> *ready_tasks = nullptr,
                std::atomic_size_t *ready_task_budget = nullptr,
                bool legacy_cursors = false,
                EvacuationEligibility eligibility =
                    EvacuationEligibility::All,
                MarkBreakdownStats *mark_diag = nullptr,
                std::atomic_size_t *legacy_region_budget = nullptr,
                MarkStopContext *stop_context = nullptr);

    template <BlockInvoker Fn, typename ReadyTaskSink>
    size_t mark_list_to_sink(Fn &&fn, RegionList &list, uint32_t timestamp,
                             ReadyTaskSink *ready_tasks,
                             std::atomic_size_t *ready_task_budget = nullptr);

    template <BlockInvoker Fn, typename ReadyTaskSink>
    size_t mark_to_sink(Fn &&fn, uint32_t timestamp,
                        ReadyTaskSink *ready_tasks,
                        std::atomic_size_t *ready_task_budget = nullptr);

    // Collect evictable regions from one list into task descriptors.
    // Tasks decouple region selection from region processing so multiple workers
    // can process evictions in parallel without repeatedly contending on list pops.
    bool collect_evict_tasks_from_list(std::vector<EvictTask> &tasks,
                                       RegionList &list, uint32_t timestamp,
                                       size_t max_tasks,
                                       uint32_t max_mark_epoch = 0,
                                       bool legacy_cursors = false,
                                       EvacuationEligibility eligibility =
                                           EvacuationEligibility::All,
                                       EvictCollectorDiag *evict_diag = nullptr,
                                       uint64_t scan_generation = 0,
                                       std::atomic_size_t *visit_budget = nullptr,
                                       bool *list_complete = nullptr);

public:
    // Collect evict tasks across all region lists (full first, then usable bins).
    // Returns true when max_tasks is reached early.
    bool collect_evict_tasks(uint32_t timestamp, std::vector<EvictTask> &tasks,
                             size_t max_tasks = SIZE_MAX,
                             uint32_t max_mark_epoch = 0,
                             bool legacy_cursors = false,
                             EvacuationEligibility eligibility =
                                 EvacuationEligibility::All,
                             EvictCollectorDiag *evict_diag = nullptr,
                             uint64_t scan_generation = 0,
                             std::atomic_size_t *visit_budget = nullptr,
                             std::atomic_bool *pass_complete = nullptr);

    template <BlockInvoker Fn>
    // Process one previously collected eviction task and publish the region back
    // to the appropriate global list based on post-evict occupancy/state.
    void process_evict_task(EvictTask &task, Fn &&fn,
                            EvictCollectorDiag *evict_diag = nullptr);

private:

    template <BlockInvoker Fn>
    size_t evict_list(
        Fn &&fn, RegionList &list, uint32_t timestamp,
        EvacuationEligibility eligibility = EvacuationEligibility::All,
        uint64_t scan_generation = 0,
        std::atomic_size_t *visit_budget = nullptr,
        bool *list_complete = nullptr,
        bool require_evict_processed = false,
        GcProcessedFilterStats *filter_stats = nullptr);

public:
    template <BlockInvoker Fn>
    size_t evict(
        Fn &&fn, uint32_t timestamp,
        EvacuationEligibility eligibility = EvacuationEligibility::All,
        uint64_t scan_generation = 0,
        std::atomic_size_t *visit_budget = nullptr,
        std::atomic_bool *pass_complete = nullptr,
        bool require_evict_processed = false,
        GcProcessedFilterStats *filter_stats = nullptr);

    bool memory_low();

    bool evacuator_continue();

    bool evacuator_should_start();

    bool mutator_release_ready();

    bool can_allocate_region_now(size_t bin) const;

    bool can_allocate_region_directly_now(
        size_t bin, RegionPlacement requested_placement) const;

    GlobalHeapBinSnapshot get_bin_snapshot(size_t bin) const;

    int64_t get_memory_low_water_mark();
    int64_t get_memory_release_water_mark();
    int64_t get_free_size();
    int64_t get_streaming_free_size() const;
};

extern GlobalHeap global_heap;

inline RegionHead *block_to_region(BlockHead *block) {
    auto *heap_start = static_cast<char *>(global_heap.get_heap());
    auto *block_addr = reinterpret_cast<char *>(block);
    auto offset = static_cast<size_t>(block_addr - heap_start);
    auto region_offset = (offset / RegionSize) * RegionSize;
    return reinterpret_cast<RegionHead *>(heap_start + region_offset);
}
BlockHead *thread_local_allocate(size_t size, cache::far_obj_t obj,
                                 cache::DereferenceScope *scope,
                                 RegionPlacement requested_placement =
                                     RegionPlacement::Unclassified,
                                 uint32_t requested_group_id = 0);

constexpr size_t BlockHeadSize = sizeof(BlockHead);

}  // namespace allocator

}  // namespace FarLib

#include "cache/core/alloc/region_list_path.hpp"
#include "cache/core/alloc/global_heap_path.hpp"
#include "cache/core/alloc/async_full_return_path.hpp"
#include "cache/core/alloc/region_head_path.hpp"
