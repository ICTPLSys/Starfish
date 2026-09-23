#pragma once
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "cpu_cycles.hpp"
#include "read_size_profile.hpp"

namespace FarLib {

namespace profile {

constexpr bool Enabled = true;
constexpr bool TraceRDMA = false;
constexpr bool TraceAlloc = false;
constexpr bool TraceMemoryUsage = false;

inline bool cq_timing_enabled() {
    static const bool enabled = [] {
        const char *text = std::getenv("FARLIB_PROFILE_CQ_TIMING");
        if (text == nullptr || *text == '\0' || text[0] == '1') return true;
        if (text[0] == '0' && text[1] == '\0') return false;
        std::abort();
    }();
    return enabled;
}
// Compile-time switch for evacuation fine-grained diagnostics.
// 0: keep only high-level evac stats (default)
// 1: enable detailed per-queue/per-region evac counters
#ifndef FARLIB_ENABLE_EVAC_FINE_PROFILE
#define FARLIB_ENABLE_EVAC_FINE_PROFILE 0
#endif
constexpr bool EnableEvacFineProfile = (FARLIB_ENABLE_EVAC_FINE_PROFILE != 0);
// Compile-time switch for extra evacuation diagnostics beyond phase timing.
// 0: only keep essential evac phase counters in logs (default)
// 1: print verbose evac diagnostics (wait loops, state mix, rounds, etc.)
#ifndef FARLIB_ENABLE_EVAC_VERBOSE_PROFILE
#define FARLIB_ENABLE_EVAC_VERBOSE_PROFILE 0
#endif
constexpr bool EnableEvacVerboseProfile =
    (FARLIB_ENABLE_EVAC_VERBOSE_PROFILE != 0);

inline bool evac_fine_profile_runtime_enabled() {
    static const bool enabled =
        std::getenv("FARLIB_EVAC_FINE_PROFILE") != nullptr ||
        std::getenv("FARLIB_CACHE_PROGRESS_DIAG") != nullptr;
    return enabled;
}

namespace enabled {
constexpr bool Evacuation = Enabled || true;
constexpr bool YieldCount = Enabled || true;
constexpr bool OnMissSchedule = Enabled || false;
};  // namespace enabled

enum class ReferenceHeatClass : uint8_t {
    Cold = 0,
    Warm = 1,
    Hot = 2,
};

enum class ReferenceKind : uint8_t {
    Read = 0,
    Write = 1,
};

struct ProfileData {
    int64_t work_cycles;
    int64_t allocate_cycles;
    int64_t post_fetch_cycles;
    int64_t post_fetch_retry_count;
    int64_t poll_count;
    int64_t poll_cycles;
    int64_t start_yield_cycle;
    int64_t yield_cycles;
    int64_t data_miss_count;
    int64_t yield_count;
    int64_t mark_cycles;
    int64_t evict_cycles;
    int64_t evacuation_count;
    int64_t evacuation_bytes;
    int64_t fork_join_cycles;
    int64_t check_cq_cycles;
    int64_t check_cq_count;
    int64_t mark_count;
    int64_t mark_payload_bytes;
    int64_t not_mark_count;
    int64_t on_miss_cycles;
    int64_t prefetch_cycles;
    int64_t core_switch_count;
    int64_t left_for_use1_cycles;
    int64_t left_for_use2_cycles;
    int64_t left_for_use3_cycles;
    int64_t left_for_use4_cycles;
    int64_t obj_modifed_cnt = 0;
    int64_t obj_unmodifed_cnt = 0;
    // Evacuator master (outer) uthread cycles spent in a GC/STW cycle.
    // This is a single-thread view (does NOT include fork-joined worker uthreads' time).
    int64_t evacuator_master_cycles;
    // Sum of elapsed RDTSCP ticks from failed local allocation until its retry
    // succeeds, across application fibres (including time yielded/descheduled).
    // This is neither CPU execution time nor a global stop-the-world duration.
    // For N application fibres and C wall-interval TSC ticks, S/(N*C) is the
    // mean per-fibre allocation-wait share. N is NOT the OS-worker count.
    // Overlapping waits are counted repeatedly; true all-fibres-blocked time
    // needs entry/exit intervals, not S/N. See PerfResult::print for conversion.
    int64_t stw_mutator_cycles;
    int64_t flip_scope_cycles;
    int64_t flip_scope_yield_cycles;
    int64_t gc_cycles;
    int64_t gc_count;

    // Evacuator breakdown counters
    int64_t evac_mark_phase_cycles;
    int64_t evac_evict_phase_cycles;
    int64_t evac_gc_phase_cycles;
    int64_t evac_flip_scope_cycles;
    int64_t evac_flush_cycles;
    int64_t evac_flush_count;
    int64_t evac_flush_full_count;
    int64_t evac_flush_partial_count;
    int64_t evac_flush_wrs;
    int64_t evac_check_cq_wait_cycles;
    int64_t evac_evict_post_cycles;
    int64_t evac_drain_cycles;
    int64_t evac_try_evict_cycles;
    int64_t evac_post_write_build_wr_cycles;
    int64_t evac_post_write_post_send_cycles;
    int64_t evac_post_write_retry_count;
    int64_t evac_write_complete_count;
    int64_t evac_write_complete_cycles;
    int64_t evac_overlap_mark_cycles;
    int64_t evac_pipeline_idle_cycles;
    int64_t evac_pipeline_rounds;
    int64_t evac_streaming_post_blocked_cycles;
    int64_t evac_streaming_wait_safe_epoch_cycles;
    int64_t evac_streaming_safe_epoch_zero_rounds;
    int64_t evac_streaming_safe_epoch_zero_tasks;
    int64_t evac_try_evict_count;
    int64_t evac_touched_after_mark_count;
    int64_t evac_flip_wait_new_loops;
    int64_t evac_flip_wait_old_loops;
    int64_t evac_flip_wait_new_max;
    int64_t evac_flip_wait_old_max;
    int64_t evac_flip_wait_new_cycles;
    int64_t evac_flip_wait_old_cycles;
    int64_t evac_flip_wait_new_blocked_flips;
    int64_t evac_flip_wait_old_blocked_flips;
    int64_t evac_flip_wait_new_entry_sum;
    int64_t evac_flip_wait_old_entry_sum;
    int64_t evac_flip_wait_new_observed_sum;
    int64_t evac_flip_wait_old_observed_sum;
    int64_t evac_evict_rounds;
    int64_t evac_evict_active_rounds;
    int64_t evac_evict_empty_rounds;
    int64_t evac_evict_active_cycles;
    int64_t evac_evict_empty_cycles;
    int64_t evac_evict_active_wrs;
    int64_t evac_try_evict_pinned;
    int64_t evac_try_evict_free_or_mismatch;
    int64_t evac_try_evict_marked_dirty;
    int64_t evac_try_evict_marked_clean;
    int64_t evac_try_evict_state_local;
    int64_t evac_try_evict_state_fetching;
    int64_t evac_try_evict_state_remote;
    int64_t evac_try_evict_state_busy;
    int64_t evac_try_evict_state_other;
    int64_t evac_evict_pop_empty;
    int64_t evac_evict_pop_head_matched;
    int64_t evac_evict_pop_success;
    int64_t evac_mark_regions_scanned;
    int64_t evac_mark_regions_with_output;
    int64_t evac_mark_regions_without_output;
    int64_t evac_mark_regions_skipped_in_use;
    int64_t evac_mark_regions_skipped_stable;
    int64_t evac_evict_regions_skipped_in_use;
    int64_t evac_evict_regions_to_free;
    int64_t evac_evict_regions_to_usable;
    int64_t evac_evict_regions_to_full;

    // Exclusive-cache attribution / RDMA traffic summary counters
    int64_t excl_remote_alloc_count;
    int64_t excl_remote_alloc_bytes;
    int64_t excl_remote_alloc_cycles;
    int64_t excl_remote_free_on_fetch_count;
    int64_t excl_remote_free_on_fetch_bytes;
    int64_t excl_remote_free_on_fetch_cycles;
    int64_t excl_remote_free_on_fetch_slow_count;
    int64_t excl_remote_free_on_fetch_slow_bytes;
    int64_t excl_remote_free_on_fetch_slow_cycles;
    int64_t excl_interrupted_evict_free_count;
    int64_t excl_interrupted_evict_free_bytes;
    int64_t excl_move_lock_spin_count;
    int64_t excl_move_lock_spin_cycles;

    int64_t remote_backup_admission_attempts;
    int64_t remote_backup_retained_count;
    int64_t remote_backup_retained_bytes;
    int64_t remote_backup_rejected_count;
    int64_t remote_backup_rejected_bytes;
    int64_t remote_backup_reused_count;
    int64_t remote_backup_reused_bytes;
    int64_t remote_backup_invalidated_count;
    int64_t remote_backup_invalidated_bytes;

    int64_t remote_dealloc_count;
    int64_t remote_dealloc_lock_cycles;
    int64_t remote_dealloc_lock_hold_cycles;
    int64_t remote_dealloc_bitmap_cycles;
    int64_t remote_dealloc_list_cycles;
    int64_t remote_dealloc_total_cycles;

    int64_t rdma_read_post_count;
    int64_t rdma_read_post_bytes;
    // Per OS-thread diagnostic, reset at the same boundary as READ totals.
    // libfibre OS workers use PTHREAD_STACK_MIN, which cannot accommodate a
    // 64KiB static TLS object. Allocate once at ProfileData construction.
    std::unique_ptr<read_size_profile::Histogram> rdma_read_size_histogram =
        read_size_profile::enabled()
            ? std::make_unique<read_size_profile::Histogram>() : nullptr;
    int64_t rdma_write_post_count;
    int64_t rdma_write_post_bytes;
    int64_t dirty_evict_bytes;
    int64_t clean_evict_bytes;
    int64_t ref_read_cold_count;
    int64_t ref_read_warm_count;
    int64_t ref_read_hot_count;
    int64_t ref_write_cold_count;
    int64_t ref_write_warm_count;
    int64_t ref_write_hot_count;
    int64_t ref_clean_cold_count;
    int64_t ref_clean_warm_count;
    int64_t ref_clean_hot_count;
    int64_t ref_dirty_cold_count;
    int64_t ref_dirty_warm_count;
    int64_t ref_dirty_hot_count;

    // Online resident-profile and placement attribution.
    int64_t resident_profile_reference_checks;
    int64_t resident_profile_samples;
    int64_t resident_profile_sample_cycles;
    int64_t resident_profile_planner_collection_cycles;
    int64_t resident_profile_planner_candidate_cycles;
    int64_t resident_profile_planner_publish_cycles;
    int64_t resident_profile_promotion_attempts;
    int64_t resident_profile_promotion_successes;
    int64_t resident_profile_promotion_cycles;
    int64_t resident_profile_demotion_attempts;
    int64_t resident_profile_demotion_successes;
    int64_t resident_profile_demotion_cycles;
    int64_t resident_profile_region_placement_attempts;
    int64_t resident_profile_region_placement_cycles;

    ProfileData() { reset(); }

    void reset() {
        work_cycles = 0;
        allocate_cycles = 0;
        post_fetch_cycles = 0;
        post_fetch_retry_count = 0;
        poll_cycles = 0;
        start_yield_cycle = 0;
        yield_cycles = 0;
        data_miss_count = 0;
        yield_count = 0;
        mark_cycles = 0;
        evict_cycles = 0;
        evacuation_count = 0;
        evacuation_bytes = 0;
        fork_join_cycles = 0;
        check_cq_cycles = 0;
        check_cq_count = 0;
        mark_count = 0;
        mark_payload_bytes = 0;
        not_mark_count = 0;
        on_miss_cycles = 0;
        prefetch_cycles = 0;
        core_switch_count = 0;
        left_for_use1_cycles = 0;
        left_for_use2_cycles = 0;
        left_for_use3_cycles = 0;
        left_for_use4_cycles = 0;
        obj_modifed_cnt = 0;
        obj_unmodifed_cnt = 0;
        evacuator_master_cycles = 0;
        stw_mutator_cycles = 0;
        flip_scope_cycles = 0;
        flip_scope_yield_cycles = 0;
        gc_cycles = 0;
        gc_count = 0;

        evac_mark_phase_cycles = 0;
        evac_evict_phase_cycles = 0;
        evac_gc_phase_cycles = 0;
        evac_flip_scope_cycles = 0;
        evac_flush_cycles = 0;
        evac_flush_count = 0;
        evac_flush_full_count = 0;
        evac_flush_partial_count = 0;
        evac_flush_wrs = 0;
        evac_check_cq_wait_cycles = 0;
        evac_evict_post_cycles = 0;
        evac_drain_cycles = 0;
        evac_try_evict_cycles = 0;
        evac_post_write_build_wr_cycles = 0;
        evac_post_write_post_send_cycles = 0;
        evac_post_write_retry_count = 0;
        evac_write_complete_count = 0;
        evac_write_complete_cycles = 0;
        evac_overlap_mark_cycles = 0;
        evac_pipeline_idle_cycles = 0;
        evac_pipeline_rounds = 0;
        evac_streaming_post_blocked_cycles = 0;
        evac_streaming_wait_safe_epoch_cycles = 0;
        evac_streaming_safe_epoch_zero_rounds = 0;
        evac_streaming_safe_epoch_zero_tasks = 0;
        evac_try_evict_count = 0;
        evac_touched_after_mark_count = 0;
        evac_flip_wait_new_loops = 0;
        evac_flip_wait_old_loops = 0;
        evac_flip_wait_new_max = 0;
        evac_flip_wait_old_max = 0;
        evac_flip_wait_new_cycles = 0;
        evac_flip_wait_old_cycles = 0;
        evac_flip_wait_new_blocked_flips = 0;
        evac_flip_wait_old_blocked_flips = 0;
        evac_flip_wait_new_entry_sum = 0;
        evac_flip_wait_old_entry_sum = 0;
        evac_flip_wait_new_observed_sum = 0;
        evac_flip_wait_old_observed_sum = 0;
        evac_evict_rounds = 0;
        evac_evict_active_rounds = 0;
        evac_evict_empty_rounds = 0;
        evac_evict_active_cycles = 0;
        evac_evict_empty_cycles = 0;
        evac_evict_active_wrs = 0;
        evac_try_evict_pinned = 0;
        evac_try_evict_free_or_mismatch = 0;
        evac_try_evict_marked_dirty = 0;
        evac_try_evict_marked_clean = 0;
        evac_try_evict_state_local = 0;
        evac_try_evict_state_fetching = 0;
        evac_try_evict_state_remote = 0;
        evac_try_evict_state_busy = 0;
        evac_try_evict_state_other = 0;
        evac_evict_pop_empty = 0;
        evac_evict_pop_head_matched = 0;
        evac_evict_pop_success = 0;
        evac_mark_regions_scanned = 0;
        evac_mark_regions_with_output = 0;
        evac_mark_regions_without_output = 0;
        evac_mark_regions_skipped_in_use = 0;
        evac_mark_regions_skipped_stable = 0;
        evac_evict_regions_skipped_in_use = 0;
        evac_evict_regions_to_free = 0;
        evac_evict_regions_to_usable = 0;
        evac_evict_regions_to_full = 0;

        excl_remote_alloc_count = 0;
        excl_remote_alloc_bytes = 0;
        excl_remote_alloc_cycles = 0;
        excl_remote_free_on_fetch_count = 0;
        excl_remote_free_on_fetch_bytes = 0;
        excl_remote_free_on_fetch_cycles = 0;
        excl_remote_free_on_fetch_slow_count = 0;
        excl_remote_free_on_fetch_slow_bytes = 0;
        excl_remote_free_on_fetch_slow_cycles = 0;
        excl_interrupted_evict_free_count = 0;
        excl_interrupted_evict_free_bytes = 0;
        excl_move_lock_spin_count = 0;
        excl_move_lock_spin_cycles = 0;

        remote_backup_admission_attempts = 0;
        remote_backup_retained_count = 0;
        remote_backup_retained_bytes = 0;
        remote_backup_rejected_count = 0;
        remote_backup_rejected_bytes = 0;
        remote_backup_reused_count = 0;
        remote_backup_reused_bytes = 0;
        remote_backup_invalidated_count = 0;
        remote_backup_invalidated_bytes = 0;

        remote_dealloc_count = 0;
        remote_dealloc_lock_cycles = 0;
        remote_dealloc_lock_hold_cycles = 0;
        remote_dealloc_bitmap_cycles = 0;
        remote_dealloc_list_cycles = 0;
        remote_dealloc_total_cycles = 0;

        rdma_read_post_count = 0;
        rdma_read_post_bytes = 0;
        if (rdma_read_size_histogram) rdma_read_size_histogram->reset();
        rdma_write_post_count = 0;
        rdma_write_post_bytes = 0;
        dirty_evict_bytes = 0;
        clean_evict_bytes = 0;
        ref_read_cold_count = 0;
        ref_read_warm_count = 0;
        ref_read_hot_count = 0;
        ref_write_cold_count = 0;
        ref_write_warm_count = 0;
        ref_write_hot_count = 0;
        ref_clean_cold_count = 0;
        ref_clean_warm_count = 0;
        ref_clean_hot_count = 0;
        ref_dirty_cold_count = 0;
        ref_dirty_warm_count = 0;
        ref_dirty_hot_count = 0;
        resident_profile_reference_checks = 0;
        resident_profile_samples = 0;
        resident_profile_sample_cycles = 0;
        resident_profile_planner_collection_cycles = 0;
        resident_profile_planner_candidate_cycles = 0;
        resident_profile_planner_publish_cycles = 0;
        resident_profile_promotion_attempts = 0;
        resident_profile_promotion_successes = 0;
        resident_profile_promotion_cycles = 0;
        resident_profile_demotion_attempts = 0;
        resident_profile_demotion_successes = 0;
        resident_profile_demotion_cycles = 0;
        resident_profile_region_placement_attempts = 0;
        resident_profile_region_placement_cycles = 0;
    }
};

constexpr size_t FrequencyHistogramBucketCount = 16;

struct FrequencyHistogramSnapshot {
    uint32_t timestamp = 0;
    uint64_t total_objects = 0;
    uint64_t zero_frequency_objects = 0;
    uint64_t total_weight = 0;
    std::array<uint64_t, FrequencyHistogramBucketCount> buckets{};
    std::array<uint64_t, FrequencyHistogramBucketCount> weighted_buckets{};

    static size_t bucket_index(uint32_t value) {
        if (value <= 1) {
            return 0;
        }
        constexpr size_t MaxIndex = FrequencyHistogramBucketCount - 1;
        size_t index = static_cast<size_t>(31 - __builtin_clz(value));
        return std::min(index, MaxIndex);
    }

    void add(uint32_t value) {
        total_objects++;
        if (value == 0) {
            zero_frequency_objects++;
            return;
        }
        total_weight += value;
        const size_t bucket = bucket_index(value);
        buckets[bucket]++;
        weighted_buckets[bucket] += value;
    }

    void merge_from(const FrequencyHistogramSnapshot &other) {
        total_objects += other.total_objects;
        zero_frequency_objects += other.zero_frequency_objects;
        total_weight += other.total_weight;
        for (size_t i = 0; i < buckets.size(); i++) {
            buckets[i] += other.buckets[i];
            weighted_buckets[i] += other.weighted_buckets[i];
        }
    }
};

struct DualFrequencyHistogramSnapshot {
    uint32_t timestamp = 0;
    uint32_t mark_pass_ordinal = 0;
    size_t ema_mark_interval = 1;
    bool ema_updated = false;
    FrequencyHistogramSnapshot window_histogram;
    FrequencyHistogramSnapshot ema_histogram;

    void merge_from(const DualFrequencyHistogramSnapshot &other) {
        if (mark_pass_ordinal == 0 && other.mark_pass_ordinal != 0) {
            mark_pass_ordinal = other.mark_pass_ordinal;
        }
        if (ema_mark_interval == 1 && other.ema_mark_interval != 1) {
            ema_mark_interval = other.ema_mark_interval;
        }
        ema_updated = ema_updated || other.ema_updated;
        window_histogram.merge_from(other.window_histogram);
        ema_histogram.merge_from(other.ema_histogram);
    }
};

struct FullPopulationFrequencySnapshot {
    uint32_t scan_sequence = 0;
    uint32_t observed_mark_pass_ordinal = 0;
    size_t scan_period_ms = 0;
    uint64_t scan_duration_us = 0;
    FrequencyHistogramSnapshot window_histogram;
    FrequencyHistogramSnapshot ema_histogram;
};

template <typename T, size_t BufferSize>
struct RingBuffer {
    size_t n;
    std::unique_ptr<T[]> buffer;

    void init() { buffer = std::make_unique<T[]>(BufferSize); }

    void add(T value) {
        buffer[n] = value;
        n = (n + 1) % BufferSize;
    }

    template <typename Fn>
    void for_each(Fn &&fn) {
        for (size_t i = n; i < BufferSize; i++) {
            fn(buffer[i]);
        }
        for (size_t i = 0; i < n; i++) {
            fn(buffer[i]);
        }
    }
};

struct AllocTrace {
    enum Operation {
        AllocObject,
        DeallocObject,
        AllocRegion,
        DeallocRegion,
    };
    Operation op;
    uint64_t tsc;
    void *addr;
    size_t bin;
};

struct MemoryUsageTrace {
    int64_t previous_free_size;
    int64_t free_size_modification;
};

struct ThreadLocalProfileData : public ProfileData {
    int64_t work_start_cycle = 0;

    RingBuffer<std::tuple<uint64_t, uint64_t, uint64_t>, 1024> rdma_read_reqs;
    RingBuffer<std::pair<uint64_t, uint64_t>, 1024> rdma_read_wcs;
    RingBuffer<AllocTrace, 4096> alloc_trace;
    RingBuffer<MemoryUsageTrace, 4096> mem_usage_trace;

    ThreadLocalProfileData() {
        if constexpr (TraceRDMA) {
            rdma_read_reqs.init();
            rdma_read_wcs.init();
        }
        if constexpr (TraceAlloc) {
            alloc_trace.init();
        }
        if constexpr (TraceMemoryUsage) {
            mem_usage_trace.init();
        }
        register_thread();
    }

    ~ThreadLocalProfileData() { unregister_thread(); }

    void register_thread();
    void unregister_thread();
};

ThreadLocalProfileData &get_tlpd();
extern ProfileData global_profile_data;
extern bool working;
extern std::atomic_bool work_phase_active;
extern uint64_t global_start_cycles, global_cycles;

void reset_all();
void record_frequency_histogram_pass(
    uint32_t timestamp, const DualFrequencyHistogramSnapshot &local_snapshot);
std::vector<DualFrequencyHistogramSnapshot> collect_frequency_histogram_history();
void record_full_population_frequency_snapshot(
    const FullPopulationFrequencySnapshot &snapshot);
std::vector<FullPopulationFrequencySnapshot>
collect_full_population_frequency_history();
uint32_t current_reference_heat_cold_max_bucket();
uint32_t current_reference_heat_warm_max_bucket();
const char *current_reference_heat_split_mode();
void prepare_frequency_output_scope(const std::string &label = "");
void begin_frequency_output_scope(const std::string &label = "");
void end_frequency_output_scope();
bool should_record_frequency_output();
std::string current_frequency_output_scope_label();
void print_frequency_histogram_history();
void print_full_population_frequency_history();

int64_t collect_work_cycles();
int64_t collect_allocate_cycles();
int64_t collect_post_fetch_cycles();
int64_t collect_poll_cycles();
int64_t collect_yield_cycles();
int64_t collect_data_miss_count();
int64_t collect_yield_count();
int64_t collect_mark_count();
int64_t collect_mark_payload_bytes();
int64_t collect_not_mark_count();
int64_t collect_mark_cycles();
int64_t collect_evict_cycles();
int64_t collect_check_cq_cycles();
int64_t collect_check_cq_count();
int64_t collect_on_miss_cycles();
int64_t collect_prefetch_cycles();
int64_t collect_core_switch_count();
int64_t collect_left_for_use1_cycles();
int64_t collect_left_for_use2_cycles();
int64_t collect_left_for_use3_cycles();
int64_t collect_left_for_use4_cycles();
int64_t collect_obj_modifed_cnt();
int64_t collect_obj_unmodifed_cnt();
int64_t collect_evacuator_master_cycles();
int64_t collect_stw_mutator_cycles();
int64_t collect_flip_scope_cycles();
int64_t collect_flip_scope_yield_cycles();
int64_t collect_gc_cycles();
int64_t collect_gc_count();

int64_t collect_evac_mark_phase_cycles();
int64_t collect_evac_evict_phase_cycles();
int64_t collect_evac_gc_phase_cycles();
int64_t collect_evac_flip_scope_cycles();
int64_t collect_evac_flush_cycles();

int64_t collect_excl_remote_alloc_count();
int64_t collect_excl_remote_alloc_bytes();
int64_t collect_excl_remote_alloc_cycles();
int64_t collect_excl_remote_free_on_fetch_count();
int64_t collect_excl_remote_free_on_fetch_bytes();
int64_t collect_excl_remote_free_on_fetch_cycles();
int64_t collect_excl_remote_free_on_fetch_slow_count();
int64_t collect_excl_remote_free_on_fetch_slow_bytes();
int64_t collect_excl_remote_free_on_fetch_slow_cycles();
int64_t collect_excl_interrupted_evict_free_count();
int64_t collect_excl_interrupted_evict_free_bytes();
int64_t collect_excl_move_lock_spin_count();
int64_t collect_excl_move_lock_spin_cycles();

int64_t collect_remote_backup_admission_attempts();
int64_t collect_remote_backup_retained_count();
int64_t collect_remote_backup_retained_bytes();
int64_t collect_remote_backup_rejected_count();
int64_t collect_remote_backup_rejected_bytes();
int64_t collect_remote_backup_reused_count();
int64_t collect_remote_backup_reused_bytes();
int64_t collect_remote_backup_invalidated_count();
int64_t collect_remote_backup_invalidated_bytes();

int64_t collect_remote_dealloc_count();
int64_t collect_remote_dealloc_lock_cycles();
int64_t collect_remote_dealloc_lock_hold_cycles();
int64_t collect_remote_dealloc_bitmap_cycles();
int64_t collect_remote_dealloc_list_cycles();
int64_t collect_remote_dealloc_total_cycles();

int64_t collect_rdma_read_post_count();
int64_t collect_rdma_read_post_bytes();
int64_t collect_rdma_write_post_count();
int64_t collect_rdma_write_post_bytes();
int64_t collect_dirty_evict_bytes();
int64_t collect_clean_evict_bytes();
int64_t collect_ref_read_cold_count();
int64_t collect_ref_read_warm_count();
int64_t collect_ref_read_hot_count();
int64_t collect_ref_write_cold_count();
int64_t collect_ref_write_warm_count();
int64_t collect_ref_write_hot_count();
int64_t collect_ref_clean_cold_count();
int64_t collect_ref_clean_warm_count();
int64_t collect_ref_clean_hot_count();
int64_t collect_ref_dirty_cold_count();
int64_t collect_ref_dirty_warm_count();
int64_t collect_ref_dirty_hot_count();
int64_t collect_resident_profile_reference_checks();
int64_t collect_resident_profile_samples();
int64_t collect_resident_profile_sample_cycles();
int64_t collect_resident_profile_planner_collection_cycles();
int64_t collect_resident_profile_planner_candidate_cycles();
int64_t collect_resident_profile_planner_publish_cycles();
int64_t collect_resident_profile_promotion_attempts();
int64_t collect_resident_profile_promotion_successes();
int64_t collect_resident_profile_promotion_cycles();
int64_t collect_resident_profile_demotion_attempts();
int64_t collect_resident_profile_demotion_successes();
int64_t collect_resident_profile_demotion_cycles();
int64_t collect_resident_profile_region_placement_attempts();
int64_t collect_resident_profile_region_placement_cycles();
int64_t collect_evac_check_cq_wait_cycles();
int64_t collect_evac_evict_post_cycles();
int64_t collect_evac_drain_cycles();
int64_t collect_evac_try_evict_cycles();
int64_t collect_evac_post_write_build_wr_cycles();
int64_t collect_evac_post_write_post_send_cycles();
int64_t collect_evac_post_write_retry_count();
int64_t collect_evac_write_complete_count();
int64_t collect_evac_write_complete_cycles();
int64_t collect_evac_overlap_mark_cycles();
int64_t collect_evac_pipeline_idle_cycles();
int64_t collect_evac_pipeline_rounds();
int64_t collect_evac_flush_count();
int64_t collect_evac_flush_full_count();
int64_t collect_evac_flush_partial_count();
int64_t collect_evac_flush_wrs();
int64_t collect_evac_streaming_post_blocked_cycles();
int64_t collect_evac_streaming_wait_safe_epoch_cycles();
int64_t collect_evac_streaming_safe_epoch_zero_rounds();
int64_t collect_evac_streaming_safe_epoch_zero_tasks();
int64_t collect_evac_try_evict_count();
int64_t collect_evac_touched_after_mark_count();
int64_t collect_evac_flip_wait_new_loops();
int64_t collect_evac_flip_wait_old_loops();
int64_t collect_evac_flip_wait_new_max();
int64_t collect_evac_flip_wait_old_max();
int64_t collect_evac_flip_wait_new_cycles();
int64_t collect_evac_flip_wait_old_cycles();
int64_t collect_evac_flip_wait_new_blocked_flips();
int64_t collect_evac_flip_wait_old_blocked_flips();
int64_t collect_evac_flip_wait_new_entry_sum();
int64_t collect_evac_flip_wait_old_entry_sum();
int64_t collect_evac_flip_wait_new_observed_sum();
int64_t collect_evac_flip_wait_old_observed_sum();
int64_t collect_evac_evict_rounds();
int64_t collect_evac_evict_active_rounds();
int64_t collect_evac_evict_empty_rounds();
int64_t collect_evac_evict_active_cycles();
int64_t collect_evac_evict_empty_cycles();
int64_t collect_evac_evict_active_wrs();
int64_t collect_evac_try_evict_pinned();
int64_t collect_evac_try_evict_free_or_mismatch();
int64_t collect_evac_try_evict_marked_dirty();
int64_t collect_evac_try_evict_marked_clean();
int64_t collect_evac_try_evict_state_local();
int64_t collect_evac_try_evict_state_fetching();
int64_t collect_evac_try_evict_state_remote();
int64_t collect_evac_try_evict_state_busy();
int64_t collect_evac_try_evict_state_other();
int64_t collect_evac_evict_pop_empty();
int64_t collect_evac_evict_pop_head_matched();
int64_t collect_evac_evict_pop_success();
int64_t collect_evac_mark_regions_scanned();
int64_t collect_evac_mark_regions_with_output();
int64_t collect_evac_mark_regions_without_output();
int64_t collect_evac_mark_regions_skipped_in_use();
int64_t collect_evac_mark_regions_skipped_stable();
int64_t collect_evac_evict_regions_skipped_in_use();
int64_t collect_evac_evict_regions_to_free();
int64_t collect_evac_evict_regions_to_usable();
int64_t collect_evac_evict_regions_to_full();

void evac_thread_work_begin();
void evac_thread_work_end();
int32_t get_evac_active_worker_count();

void print_profile_data();

// Optional cold-path diagnostics for allocation stalls.  The aggregate
// counters are elapsed RDTSCP ticks; they intentionally distinguish average
// fibre-time (wait_ticks) from the union/full-fibre timeline counters.
struct AllocationWaitDiagnosticsSnapshot {
    bool enabled = false;
    bool active = false;
    bool frozen = false;
    uint64_t expected_fibre_count = 0;
    uint64_t event_count = 0;
    uint64_t completed_event_count = 0;
    uint64_t wait_ticks = 0;
    uint64_t max_wait_ticks = 0;
    uint64_t union_ticks = 0;
    uint64_t full_ticks = 0;
    uint64_t max_active = 0;
    uint64_t active_at_end = 0;
    uint64_t unclosed_events = 0;
    uint64_t ignored_events = 0;
    uint64_t over_capacity_events = 0;
    uint64_t invalid_end_events = 0;
    uint64_t timestamp_regressions = 0;
    uint64_t late_end_events = 0;
    // These counters cover only the exclusive+legacy on-demand condition
    // wait hook.  condition_wait_ticks includes condition waiting and the
    // fibre scheduler's return path; it is not pure kernel sleep time.
    // Currently instrumented only in the legacy-exclusive on-demand path.
    // The wait interval ends before reacquiring eviction_mutex; the readiness
    // sample after wake is taken after reacquisition, not at notification time.
    uint64_t condition_wait_count = 0;
    uint64_t condition_wait_ticks = 0;
    uint64_t condition_ready_before_sleep_count = 0;
    uint64_t condition_not_ready_after_wake_count = 0;
};

struct AllocationWaitDiagnosticsEvent {
    uint64_t start_tsc = 0;
    uint64_t generation = 0;
    bool tracked = false;
};

// Enable tracking only when FARLIB_ALLOC_WAIT_DIAG is set to a non-zero
// value.  application_fibre_count is the logical mutator-fibre count (not
// the libfibre OS-worker count).  The end call freezes the snapshot.
void begin_allocation_wait_diagnostics(size_t application_fibre_count);
AllocationWaitDiagnosticsSnapshot end_allocation_wait_diagnostics();
void print_allocation_wait_diagnostics();
// A lock-free gate for cold-path callers.  When false, callers must not
// evaluate diagnostic predicates or take diagnostic timestamps.
bool diagnostics_active();

// Record one exclusive+legacy condition wait.  elapsed_ticks is measured
// with the same TSC clock as allocation-wait diagnostics and includes the
// condition wait plus scheduler return path, not just kernel sleep.
void record_allocation_condition_wait(uint64_t elapsed_ticks,
                                      bool ready_before_sleep,
                                      bool ready_after_wake);

// Internal hooks used only around the cold local-allocation retry path.
AllocationWaitDiagnosticsEvent begin_allocation_wait_event(
    uint64_t start_tsc);
void end_allocation_wait_event(AllocationWaitDiagnosticsEvent event,
                              uint64_t end_tsc);

inline void resume_work(bool suspended) {
    if constexpr (Enabled) {
        if (working && suspended) {
            // assert(get_tlpd().work_start_cycle == 0);
            get_tlpd().work_start_cycle = get_cycles();
        }
    }
}
inline bool suspend_work() {
    if constexpr (Enabled) {
        if (working && get_tlpd().work_start_cycle != 0) {
            get_tlpd().work_cycles +=
                get_cycles() - get_tlpd().work_start_cycle;
            get_tlpd().work_start_cycle = 0;
            return true;
        }
    }
    return false;
}
inline void start_work() {
    assert(!working);
    working = true;
    global_start_cycles = get_cycles();
    work_phase_active.store(true, std::memory_order_release);
}
inline void end_work() {
    assert(working);
    work_phase_active.store(false, std::memory_order_release);
    working = false;
    global_cycles = get_cycles() - global_start_cycles;
}
inline bool is_working() {
    return work_phase_active.load(std::memory_order_acquire);
}
inline void thread_start_work() { resume_work(true); }
inline void thread_end_work() { suspend_work(); }
inline void start_allocate() {
    if constexpr (Enabled) get_tlpd().allocate_cycles -= get_cycles();
}
inline void end_allocate() {
    if constexpr (Enabled) get_tlpd().allocate_cycles += get_cycles();
}
inline void count_obj_modifed() {
    if constexpr (Enabled) get_tlpd().obj_modifed_cnt++;
}
inline void count_obj_unmodifed() {
    if constexpr (Enabled) get_tlpd().obj_unmodifed_cnt++;
}
inline void start_post_fetch() {
    if constexpr (Enabled) {
        get_tlpd().data_miss_count++;
        get_tlpd().post_fetch_cycles -= get_cycles();
    }
}
inline void end_post_fetch() {
    if constexpr (Enabled) get_tlpd().post_fetch_cycles += get_cycles();
}

inline void add_post_retry_count() {
    if constexpr (Enabled) {
        get_tlpd().post_fetch_retry_count++;
    }
}
inline void start_poll() {
    if constexpr (Enabled) get_tlpd().poll_cycles -= get_cycles();
}
inline void end_poll() {
    if constexpr (Enabled) {
        get_tlpd().poll_count++;
        get_tlpd().poll_cycles += get_cycles();
    }
}
inline void start_yield() {
    if constexpr (enabled::YieldCount) {
        get_tlpd().yield_count++;
    }
    if constexpr (Enabled) {
        get_tlpd().start_yield_cycle = get_cycles();
    }
}
inline void end_yield() {
    if constexpr (Enabled) {
        if (get_tlpd().start_yield_cycle != 0) [[likely]] {
            get_tlpd().yield_cycles +=
                get_cycles() - get_tlpd().start_yield_cycle;
            get_tlpd().start_yield_cycle = 0;
        }
    }
}
inline void count_evacuation() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evacuation_count++;
    }
}
inline void count_evacuation_bytes(size_t bytes) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evacuation_bytes += bytes;
    }
}
inline void count_evac_flush(size_t wrs, bool full_flush) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flush_count++;
        get_tlpd().evac_flush_wrs += wrs;
        if (full_flush) {
            get_tlpd().evac_flush_full_count++;
        } else {
            get_tlpd().evac_flush_partial_count++;
        }
    }
}
inline void count_on_miss_cycles(int64_t cycles) {
    if constexpr (enabled::OnMissSchedule) {
        get_tlpd().on_miss_cycles += cycles;
    }
}
inline void start_on_miss() {
    if constexpr (enabled::OnMissSchedule)
        get_tlpd().on_miss_cycles -= get_cycles();
}
inline void end_on_miss() {
    if constexpr (enabled::OnMissSchedule)
        get_tlpd().on_miss_cycles += get_cycles();
}

inline void start_prefetch() {
    if constexpr (enabled::OnMissSchedule)
        get_tlpd().prefetch_cycles -= get_cycles();
}
inline void end_prefetch() {
    if constexpr (enabled::OnMissSchedule)
        get_tlpd().prefetch_cycles += get_cycles();
}

inline void count_core_switch() {
    if constexpr (Enabled) {
        get_tlpd().core_switch_count++;
    }
}

inline int64_t start_mark() {
    if constexpr (enabled::Evacuation) {
        return get_cycles();
    } else {
        return 0;
    }
}
inline void end_mark(int64_t start_cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().mark_cycles += get_cycles() - start_cycles;
    }
}
inline int64_t start_evict() {
    if constexpr (enabled::Evacuation) {
        return get_cycles();
    } else {
        return 0;
    }
}
inline void end_evict(int64_t start_cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evict_cycles += get_cycles() - start_cycles;
    }
}
inline void start_fork_join() {
    if constexpr (Enabled) get_tlpd().fork_join_cycles -= get_cycles();
}
inline void end_fork_join() {
    if constexpr (Enabled) get_tlpd().fork_join_cycles += get_cycles();
}
inline void start_left_for_use1() {
    if constexpr (Enabled) get_tlpd().left_for_use1_cycles -= get_cycles();
}
inline void end_left_for_use1() {
    if constexpr (Enabled) get_tlpd().left_for_use1_cycles += get_cycles();
}
inline void start_left_for_use2() {
    if constexpr (Enabled) get_tlpd().left_for_use2_cycles -= get_cycles();
}
inline void end_left_for_use2() {
    if constexpr (Enabled) get_tlpd().left_for_use2_cycles += get_cycles();
}
inline void start_left_for_use3() {
    if constexpr (Enabled) get_tlpd().left_for_use3_cycles -= get_cycles();
}
inline void end_left_for_use3() {
    if constexpr (Enabled) get_tlpd().left_for_use3_cycles += get_cycles();
}
inline void start_left_for_use4() {
    if constexpr (Enabled) get_tlpd().left_for_use4_cycles -= get_cycles();
}
inline void end_left_for_use4() {
    if constexpr (Enabled) get_tlpd().left_for_use4_cycles += get_cycles();
}

inline void start_check_cq() {
    if constexpr (Enabled) {
        if (cq_timing_enabled()) {
            // Preserve the exact legacy enabled path for the current control.
            get_tlpd().check_cq_cycles -= get_cycles();
            get_tlpd().check_cq_count++;
        } else {
            get_tlpd().check_cq_count++;
        }
    }
}
inline void end_check_cq() {
    if constexpr (Enabled) {
        if (cq_timing_enabled()) {
            get_tlpd().check_cq_cycles += get_cycles();
        }
    }
}
inline void count_mark(size_t payload_bytes) {
    if constexpr (Enabled) {
        get_tlpd().mark_count++;
        get_tlpd().mark_payload_bytes += static_cast<int64_t>(payload_bytes);
    }
}
inline void count_not_mark() {
    if constexpr (Enabled) {
        get_tlpd().not_mark_count++;
    }
}
inline void count_left_for_use1_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().left_for_use1_cycles += cycles;
    }
}
inline void count_left_for_use2_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().left_for_use2_cycles += cycles;
    }
}
inline void count_left_for_use3_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().left_for_use3_cycles += cycles;
    }
}
inline void count_left_for_use4_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().left_for_use4_cycles += cycles;
    }
}

inline void count_stw_mutator_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().stw_mutator_cycles += cycles;
    }
}

inline void count_evac_try_evict() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_count++;
    }
}

inline void count_evac_try_evict_cycles(int64_t cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_cycles += cycles;
    }
}

inline void count_evac_post_write_build_wr_cycles(int64_t cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_post_write_build_wr_cycles += cycles;
    }
}

inline void count_evac_post_write_post_send_cycles(int64_t cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_post_write_post_send_cycles += cycles;
    }
}

inline void count_evac_post_write_retry_count(int64_t retries) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_post_write_retry_count += retries;
    }
}

inline void count_evac_write_complete_cycles(int64_t cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_write_complete_count++;
        get_tlpd().evac_write_complete_cycles += cycles;
    }
}

inline void count_evac_touched_after_mark() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_touched_after_mark_count++;
    }
}

inline void count_evac_try_evict_pinned() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_pinned++;
    }
}

inline void count_evac_try_evict_free_or_mismatch() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_free_or_mismatch++;
    }
}

inline void count_evac_try_evict_marked_dirty() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_marked_dirty++;
    }
}

inline void count_evac_try_evict_marked_clean() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_marked_clean++;
    }
}

inline void count_evac_try_evict_state_local() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_state_local++;
    }
}

inline void count_evac_try_evict_state_fetching() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_state_fetching++;
    }
}

inline void count_evac_try_evict_state_remote() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_state_remote++;
    }
}

inline void count_evac_try_evict_state_busy() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_state_busy++;
    }
}

inline void count_evac_try_evict_state_other() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_try_evict_state_other++;
    }
}

inline void count_evac_evict_pop_empty() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_evict_pop_empty++;
    }
}

inline void count_evac_evict_pop_head_matched() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_evict_pop_head_matched++;
    }
}

inline void count_evac_evict_pop_success() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_evict_pop_success++;
    }
}

inline void count_evac_mark_region_scanned() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_mark_regions_scanned++;
    }
}

inline void count_evac_mark_region_with_output() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_mark_regions_with_output++;
    }
}

inline void count_evac_mark_region_without_output() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_mark_regions_without_output++;
    }
}

inline void count_evac_mark_region_skipped_in_use() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_mark_regions_skipped_in_use++;
    }
}

inline void count_evac_mark_region_skipped_stable() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_mark_regions_skipped_stable++;
    }
}

inline void count_evac_evict_region_skipped_in_use() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_evict_regions_skipped_in_use++;
    }
}

inline void count_evac_evict_region_to_free() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_evict_regions_to_free++;
    }
}

inline void count_evac_evict_region_to_usable() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_evict_regions_to_usable++;
    }
}

inline void count_evac_evict_region_to_full() {
    if constexpr (enabled::Evacuation) {
        if (!EnableEvacFineProfile && !evac_fine_profile_runtime_enabled()) {
            return;
        }
        get_tlpd().evac_evict_regions_to_full++;
    }
}

inline void count_evac_flip_wait_new_loop() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_new_loops++;
    }
}

inline void count_evac_flip_wait_old_loop() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_old_loops++;
    }
}

inline void count_evac_flip_wait_new_max(int64_t v) {
    if constexpr (enabled::Evacuation) {
        if (v > get_tlpd().evac_flip_wait_new_max) {
            get_tlpd().evac_flip_wait_new_max = v;
        }
    }
}

inline void count_evac_flip_wait_old_max(int64_t v) {
    if constexpr (enabled::Evacuation) {
        if (v > get_tlpd().evac_flip_wait_old_max) {
            get_tlpd().evac_flip_wait_old_max = v;
        }
    }
}

inline void count_evac_flip_wait_new_cycles(int64_t cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_new_cycles += cycles;
    }
}

inline void count_evac_flip_wait_old_cycles(int64_t cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_old_cycles += cycles;
    }
}

inline void count_evac_flip_wait_new_blocked_flip(int64_t entry_count) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_new_blocked_flips++;
        get_tlpd().evac_flip_wait_new_entry_sum += entry_count;
    }
}

inline void count_evac_flip_wait_old_blocked_flip(int64_t entry_count) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_old_blocked_flips++;
        get_tlpd().evac_flip_wait_old_entry_sum += entry_count;
    }
}

inline void count_evac_flip_wait_new_observed(int64_t observed_count) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_new_observed_sum += observed_count;
    }
}

inline void count_evac_flip_wait_old_observed(int64_t observed_count) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().evac_flip_wait_old_observed_sum += observed_count;
    }
}

inline void start_stw_mutator() {
    if constexpr (Enabled) {
        get_tlpd().stw_mutator_cycles -= get_cycles();
    }
}

inline void end_stw_mutator(int64_t start_cycles) {
    if constexpr (Enabled) {
        get_tlpd().stw_mutator_cycles += get_cycles() - start_cycles;
    }
}

inline int64_t start_flip_scope() {
    if constexpr (Enabled) {
        return get_cycles();
    }
    return 0;
}

inline void end_flip_scope(int64_t start_cycles) {
    if constexpr (Enabled) {
        get_tlpd().flip_scope_cycles += get_cycles() - start_cycles;
    }
}

inline void count_flip_scope_yield(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().flip_scope_yield_cycles += cycles;
    }
}

inline int64_t start_gc() {
    if constexpr (enabled::Evacuation) {
        return get_cycles();
    }
    return 0;
}

inline void end_gc(int64_t start_cycles) {
    if constexpr (enabled::Evacuation) {
        get_tlpd().gc_cycles += get_cycles() - start_cycles;
    }
}

inline void count_gc() {
    if constexpr (enabled::Evacuation) {
        get_tlpd().gc_count++;
    }
}

inline void count_rdma_read_post(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().rdma_read_post_count++;
        get_tlpd().rdma_read_post_bytes += (int64_t)bytes;
        if (read_size_profile::enabled()) {
            get_tlpd().rdma_read_size_histogram->record_single(bytes);
        }
    }
}

inline void count_rdma_read_posts(size_t count, size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().rdma_read_post_count += static_cast<int64_t>(count);
        get_tlpd().rdma_read_post_bytes += static_cast<int64_t>(bytes);
        if (read_size_profile::enabled()) {
            get_tlpd().rdma_read_size_histogram->record_aggregate(count, bytes);
        }
    }
}

inline void count_rdma_write_post(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().rdma_write_post_count++;
        get_tlpd().rdma_write_post_bytes += (int64_t)bytes;
    }
}

inline void count_remote_backup_admission_attempt() {
    if constexpr (Enabled) {
        get_tlpd().remote_backup_admission_attempts++;
    }
}

inline void count_remote_backup_retained(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().remote_backup_retained_count++;
        get_tlpd().remote_backup_retained_bytes += static_cast<int64_t>(bytes);
    }
}

inline void count_remote_backup_rejected(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().remote_backup_rejected_count++;
        get_tlpd().remote_backup_rejected_bytes += static_cast<int64_t>(bytes);
    }
}

inline void count_remote_backup_reused(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().remote_backup_reused_count++;
        get_tlpd().remote_backup_reused_bytes += static_cast<int64_t>(bytes);
    }
}

inline void count_remote_backup_invalidated(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().remote_backup_invalidated_count++;
        get_tlpd().remote_backup_invalidated_bytes +=
            static_cast<int64_t>(bytes);
    }
}

inline void count_dirty_evict_bytes(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().dirty_evict_bytes += static_cast<int64_t>(bytes);
    }
}

inline void count_clean_evict_bytes(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().clean_evict_bytes += static_cast<int64_t>(bytes);
    }
}

inline void count_reference_access(ReferenceHeatClass heat, ReferenceKind kind,
                                   bool dirty, int64_t weight = 1) {
    if constexpr (!Enabled) {
        return;
    }
    if (weight <= 0) {
        return;
    }
    auto &tlpd = get_tlpd();
    switch (heat) {
    case ReferenceHeatClass::Cold:
        if (dirty) {
            tlpd.ref_dirty_cold_count += weight;
        } else {
            tlpd.ref_clean_cold_count += weight;
        }
        break;
    case ReferenceHeatClass::Warm:
        if (dirty) {
            tlpd.ref_dirty_warm_count += weight;
        } else {
            tlpd.ref_clean_warm_count += weight;
        }
        break;
    case ReferenceHeatClass::Hot:
        if (dirty) {
            tlpd.ref_dirty_hot_count += weight;
        } else {
            tlpd.ref_clean_hot_count += weight;
        }
        break;
    }
    switch (kind) {
    case ReferenceKind::Read:
        switch (heat) {
        case ReferenceHeatClass::Cold:
            tlpd.ref_read_cold_count += weight;
            return;
        case ReferenceHeatClass::Warm:
            tlpd.ref_read_warm_count += weight;
            return;
        case ReferenceHeatClass::Hot:
            tlpd.ref_read_hot_count += weight;
            return;
        }
        return;
    case ReferenceKind::Write:
        switch (heat) {
        case ReferenceHeatClass::Cold:
            tlpd.ref_write_cold_count += weight;
            return;
        case ReferenceHeatClass::Warm:
            tlpd.ref_write_warm_count += weight;
            return;
        case ReferenceHeatClass::Hot:
            tlpd.ref_write_hot_count += weight;
            return;
        }
        return;
    }
}

inline void count_excl_remote_alloc(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().excl_remote_alloc_count++;
        get_tlpd().excl_remote_alloc_bytes += (int64_t)bytes;
    }
}

inline void count_excl_remote_alloc_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().excl_remote_alloc_cycles += cycles;
    }
}

inline void count_excl_remote_free_on_fetch(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().excl_remote_free_on_fetch_count++;
        get_tlpd().excl_remote_free_on_fetch_bytes += (int64_t)bytes;
    }
}

inline void count_excl_remote_free_on_fetch_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().excl_remote_free_on_fetch_cycles += cycles;
    }
}

inline void count_excl_remote_free_on_fetch_slow(int64_t cycles, size_t bytes) {
    if constexpr (Enabled) {
        auto &tlpd = get_tlpd();
        tlpd.excl_remote_free_on_fetch_slow_count++;
        tlpd.excl_remote_free_on_fetch_slow_bytes += (int64_t)bytes;
        tlpd.excl_remote_free_on_fetch_slow_cycles += cycles;
    }
}

inline void count_excl_interrupted_evict_free(size_t bytes) {
    if constexpr (Enabled) {
        get_tlpd().excl_interrupted_evict_free_count++;
        get_tlpd().excl_interrupted_evict_free_bytes += (int64_t)bytes;
    }
}

inline void count_excl_move_lock_spin() {
    if constexpr (Enabled) {
        get_tlpd().excl_move_lock_spin_count++;
    }
}

inline void count_excl_move_lock_spin_cycles(int64_t cycles) {
    if constexpr (Enabled) {
        get_tlpd().excl_move_lock_spin_cycles += cycles;
    }
}

inline void count_remote_dealloc(int64_t lock_cycles, int64_t lock_hold_cycles,
                                 int64_t bitmap_cycles, int64_t list_cycles,
                                 int64_t total_cycles) {
    if constexpr (Enabled) {
        auto &tlpd = get_tlpd();
        tlpd.remote_dealloc_count++;
        tlpd.remote_dealloc_lock_cycles += lock_cycles;
        tlpd.remote_dealloc_lock_hold_cycles += lock_hold_cycles;
        tlpd.remote_dealloc_bitmap_cycles += bitmap_cycles;
        tlpd.remote_dealloc_list_cycles += list_cycles;
        tlpd.remote_dealloc_total_cycles += total_cycles;
    }
}

inline void count_resident_profile_reference_check() {
    get_tlpd().resident_profile_reference_checks++;
}

inline void count_resident_profile_sample(int64_t cycles) {
    get_tlpd().resident_profile_samples++;
    get_tlpd().resident_profile_sample_cycles += cycles;
}

inline void count_resident_profile_planner_collection_cycles(int64_t cycles) {
    get_tlpd().resident_profile_planner_collection_cycles += cycles;
}

inline void count_resident_profile_planner_candidate_cycles(int64_t cycles) {
    get_tlpd().resident_profile_planner_candidate_cycles += cycles;
}

inline void count_resident_profile_planner_publish_cycles(int64_t cycles) {
    get_tlpd().resident_profile_planner_publish_cycles += cycles;
}

inline void count_resident_profile_promotion(int64_t cycles, bool success) {
    get_tlpd().resident_profile_promotion_attempts++;
    get_tlpd().resident_profile_promotion_successes += success;
    get_tlpd().resident_profile_promotion_cycles += cycles;
}

inline void count_resident_profile_demotion(int64_t cycles, bool success) {
    get_tlpd().resident_profile_demotion_attempts++;
    get_tlpd().resident_profile_demotion_successes += success;
    get_tlpd().resident_profile_demotion_cycles += cycles;
}

inline void count_resident_profile_region_placement(int64_t cycles) {
    get_tlpd().resident_profile_region_placement_attempts++;
    get_tlpd().resident_profile_region_placement_cycles += cycles;
}

void print_rdma_trace();

inline void trace_alloc(void *p, size_t bin) {
    if constexpr (TraceAlloc) {
        get_tlpd().alloc_trace.add(
            {AllocTrace::AllocObject, __rdtsc(), p, bin});
    }
}
inline void trace_dealloc(void *p, size_t bin) {
    if constexpr (TraceAlloc) {
        get_tlpd().alloc_trace.add(
            {AllocTrace::DeallocObject, __rdtsc(), p, bin});
    }
}
void print_alloc_trace();

void print_mem_usage_trace();

}  // namespace profile

}  // namespace FarLib
