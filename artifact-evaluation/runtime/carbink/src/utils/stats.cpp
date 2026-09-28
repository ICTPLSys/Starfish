#include "utils/stats.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "async/stream_runner.hpp"
#include "rdma/config.hpp"

#ifdef NO_REMOTE
namespace FarLib {
const rdma::Configure &get_config();
}
#endif

namespace FarLib {
namespace async {
#ifdef PROFILE_STREAM_RUNNER_SCHEDULE
std::atomic_int64_t StreamRunnerProfiler::global_total_cycles;
std::atomic_int64_t StreamRunnerProfiler::global_app_cycles;
std::atomic_int64_t StreamRunnerProfiler::global_sched_cycles;
std::atomic_int64_t StreamRunnerProfiler::global_poll_cq_cycles;
#endif
}  // namespace async
namespace profile {

std::string format_with_commas(int64_t value) {
    std::string s = std::to_string(value);
    int n = s.length();
    int res_len = n + (n - 1) / 3;
    std::string res(res_len, ',');
    for (int i = n - 1, j = res_len - 1, count = 0; i >= 0; i--, j--) {
        res[j] = s[i];
        count++;
        if (count == 3 && j > 0 && s[i - 1] != '-') {
            j--;
            res[j] = ',';
            count = 0;
        }
    }
    return res;
}

std::string format_scientific(int64_t value) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(2) << (double)value;
    return ss.str();
}

std::unordered_map<std::thread::id, ThreadLocalProfileData *> profile_data_map;
thread_local ThreadLocalProfileData tlpd;
// Only exit-time merge and post-work snapshot take this mutex, never READs.
std::mutex read_size_histogram_mutex;
std::mutex frequency_histogram_mutex;
std::vector<DualFrequencyHistogramSnapshot> frequency_histogram_history;
std::mutex full_population_frequency_histogram_mutex;
std::vector<FullPopulationFrequencySnapshot> full_population_frequency_history;
std::mutex frequency_output_scope_mutex;
std::string frequency_output_scope_label;
std::atomic_bool frequency_output_scope_explicit{false};
std::atomic_bool frequency_output_scope_active{true};
std::atomic_uint32_t reference_heat_cold_max_bucket{2};
std::atomic_uint32_t reference_heat_warm_max_bucket{5};
__attribute__((noinline)) ThreadLocalProfileData &get_tlpd() {
    asm volatile("" : : : "memory");
    return tlpd;
}
ProfileData global_profile_data;
bool working = false;
std::atomic_bool work_phase_active{false};
uint64_t global_start_cycles = 0, global_cycles = 0;
std::atomic_int32_t evac_active_worker_count{0};

void evac_thread_work_begin() {
    evac_active_worker_count.fetch_add(1, std::memory_order_relaxed);
}

void evac_thread_work_end() {
    evac_active_worker_count.fetch_sub(1, std::memory_order_relaxed);
}

int32_t get_evac_active_worker_count() {
    return evac_active_worker_count.load(std::memory_order_relaxed);
}

namespace {

struct AllocationWaitDiagnosticsState {
    std::mutex mutex;
    AllocationWaitDiagnosticsSnapshot snapshot;
    uint64_t last_tsc = 0;
    uint64_t active_count = 0;
    uint64_t generation = 0;
    bool accepting = false;
};

AllocationWaitDiagnosticsState allocation_wait_diag_state;
std::atomic_bool allocation_wait_diag_fast_enabled{false};

bool allocation_wait_diag_env_enabled() {
    const char *value = std::getenv("FARLIB_ALLOC_WAIT_DIAG");
    return value != nullptr && std::strtoull(value, nullptr, 0) != 0;
}

// Event boundaries are serialized and timestamped while holding the mutex, so
// the timeline follows lock order even when fibres on different workers
// complete their cold-path updates out of order.  This adds only the tiny
// diagnostic record delay to union/full boundaries; the caller-supplied TSC
// values remain the source for per-event wait_ticks/max_wait_ticks.  A
// backwards TSC value is clamped and reported instead of silently creating
// negative union/full intervals.
void allocation_wait_diag_advance_locked(uint64_t timestamp) {
    auto &state = allocation_wait_diag_state;
    auto &snapshot = state.snapshot;
    if (state.last_tsc == 0) {
        state.last_tsc = timestamp;
        return;
    }
    if (timestamp < state.last_tsc) {
        snapshot.timestamp_regressions++;
        timestamp = state.last_tsc;
    }
    const uint64_t delta = timestamp - state.last_tsc;
    if (state.active_count != 0) {
        snapshot.union_ticks += delta;
    }
    if (snapshot.expected_fibre_count != 0 &&
        state.active_count == snapshot.expected_fibre_count) {
        snapshot.full_ticks += delta;
    }
    state.last_tsc = timestamp;
}

uint64_t effective_reference_bucket_count(
    const FrequencyHistogramSnapshot &snapshot, size_t bucket) {
    if (bucket == 0) {
        return snapshot.zero_frequency_objects + snapshot.buckets[0];
    }
    return snapshot.buckets[bucket];
}

uint64_t effective_reference_bucket_weight(
    const FrequencyHistogramSnapshot &snapshot, size_t bucket) {
    return snapshot.weighted_buckets[bucket];
}

bool reference_heat_split_access_weighted() {
    return FarLib::get_config().reference_heat_split_mode == "access_weighted";
}

void update_reference_heat_thresholds(
    const FrequencyHistogramSnapshot &snapshot) {
    const bool access_weighted = reference_heat_split_access_weighted();
    const uint64_t total_metric =
        access_weighted && snapshot.total_weight > 0 ? snapshot.total_weight
                                                     : snapshot.total_objects;
    if (total_metric == 0) {
        return;
    }
    const uint64_t first_target = (total_metric + 2) / 3;
    const uint64_t second_target = ((total_metric * 2) + 2) / 3;
    uint64_t cumulative = 0;
    uint32_t cold_bucket = 0;
    uint32_t warm_bucket = FrequencyHistogramBucketCount - 1;
    bool found_first = false;
    for (size_t bucket = 0; bucket < FrequencyHistogramBucketCount; bucket++) {
        cumulative += access_weighted && snapshot.total_weight > 0
                          ? effective_reference_bucket_weight(snapshot, bucket)
                          : effective_reference_bucket_count(snapshot, bucket);
        if (!found_first && cumulative >= first_target) {
            cold_bucket = static_cast<uint32_t>(bucket);
            found_first = true;
        }
        if (cumulative >= second_target) {
            warm_bucket = static_cast<uint32_t>(bucket);
            break;
        }
    }
    if (warm_bucket < cold_bucket) {
        warm_bucket = cold_bucket;
    }
    reference_heat_cold_max_bucket.store(cold_bucket, std::memory_order_relaxed);
    reference_heat_warm_max_bucket.store(warm_bucket, std::memory_order_relaxed);
}

uint32_t bucket_lower_bound(uint32_t bucket) {
    if (bucket == 0) {
        return 0;
    }
    return 1u << bucket;
}

uint32_t bucket_upper_bound(uint32_t bucket) {
    if (bucket == 0) {
        return 1;
    }
    if (bucket + 1 >= FrequencyHistogramBucketCount) {
        return UINT32_MAX;
    }
    return (1u << (bucket + 1)) - 1u;
}

}  // namespace

void begin_allocation_wait_diagnostics(size_t application_fibre_count) {
    const bool enabled = allocation_wait_diag_env_enabled();
    allocation_wait_diag_fast_enabled.store(false, std::memory_order_release);

    auto &state = allocation_wait_diag_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    const bool previous_session_was_active = state.accepting;
    const uint64_t previous_active_count = state.active_count;

    state.snapshot = AllocationWaitDiagnosticsSnapshot{};
    state.snapshot.enabled = enabled;
    state.snapshot.expected_fibre_count = application_fibre_count;
    state.snapshot.active = enabled && application_fibre_count != 0;
    state.snapshot.frozen = !state.snapshot.active;
    if (previous_session_was_active) {
        // The old session cannot be recovered after a restart.  Preserve an
        // explicit anomaly marker in the new snapshot and let late event ends
        // below report any tokens from the discarded generation.
        state.snapshot.invalid_end_events++;
        state.snapshot.unclosed_events = previous_active_count;
    }

    state.active_count = 0;
    state.generation++;
    state.accepting = state.snapshot.active;
    state.last_tsc = state.snapshot.active ? get_cycles() : 0;
    if (enabled && application_fibre_count == 0) {
        state.snapshot.invalid_end_events++;
    }
    if (state.accepting) {
        allocation_wait_diag_fast_enabled.store(true,
                                                std::memory_order_release);
    }
}

bool diagnostics_active() {
    return allocation_wait_diag_fast_enabled.load(std::memory_order_acquire);
}

void record_allocation_condition_wait(uint64_t elapsed_ticks,
                                      bool ready_before_sleep,
                                      bool ready_after_wake) {
    if (!allocation_wait_diag_fast_enabled.load(std::memory_order_acquire)) {
        return;
    }

    auto &state = allocation_wait_diag_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.accepting || !state.snapshot.enabled) {
        return;
    }
    state.snapshot.condition_wait_count++;
    state.snapshot.condition_wait_ticks += elapsed_ticks;
    if (ready_before_sleep) {
        state.snapshot.condition_ready_before_sleep_count++;
    }
    if (!ready_after_wake) {
        state.snapshot.condition_not_ready_after_wake_count++;
    }
}

AllocationWaitDiagnosticsSnapshot end_allocation_wait_diagnostics() {
    allocation_wait_diag_fast_enabled.store(false, std::memory_order_release);

    auto &state = allocation_wait_diag_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.accepting) {
        state.snapshot.active = false;
        state.snapshot.frozen = true;
        return state.snapshot;
    }

    allocation_wait_diag_advance_locked(get_cycles());
    state.snapshot.active_at_end = state.active_count;
    state.snapshot.unclosed_events =
        state.snapshot.event_count >= state.snapshot.completed_event_count
            ? state.snapshot.event_count -
                  state.snapshot.completed_event_count
            : 0;
    if (state.active_count != 0 || state.snapshot.unclosed_events != 0) {
        state.snapshot.invalid_end_events++;
    }
    state.accepting = false;
    state.snapshot.active = false;
    state.snapshot.frozen = true;
    return state.snapshot;
}

AllocationWaitDiagnosticsEvent begin_allocation_wait_event(
    uint64_t start_tsc) {
    AllocationWaitDiagnosticsEvent event;
    if (!allocation_wait_diag_fast_enabled.load(std::memory_order_acquire)) {
        return event;
    }

    auto &state = allocation_wait_diag_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.accepting || !state.snapshot.enabled) {
        if (state.snapshot.active) {
            state.snapshot.ignored_events++;
        }
        return event;
    }

    allocation_wait_diag_advance_locked(get_cycles());
    state.active_count++;
    state.snapshot.event_count++;
    if (state.active_count > state.snapshot.max_active) {
        state.snapshot.max_active = state.active_count;
    }
    if (state.active_count > state.snapshot.expected_fibre_count) {
        state.snapshot.over_capacity_events++;
    }
    event.start_tsc = start_tsc;
    event.generation = state.generation;
    event.tracked = true;
    return event;
}

void end_allocation_wait_event(AllocationWaitDiagnosticsEvent event,
                              uint64_t end_tsc) {
    if (!event.tracked) {
        return;
    }

    auto &state = allocation_wait_diag_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.accepting || !state.snapshot.enabled) {
        // The frozen snapshot already reports any token still active at end
        // as unclosed; do not mutate it after the freeze point.
        return;
    }
    if (event.generation != state.generation) {
        state.snapshot.late_end_events++;
        return;
    }
    if (state.active_count == 0) {
        state.snapshot.invalid_end_events++;
        return;
    }

    allocation_wait_diag_advance_locked(get_cycles());
    state.active_count--;
    state.snapshot.completed_event_count++;
    uint64_t duration = 0;
    if (end_tsc >= event.start_tsc) {
        duration = end_tsc - event.start_tsc;
    } else {
        state.snapshot.invalid_end_events++;
        state.snapshot.timestamp_regressions++;
    }
    state.snapshot.wait_ticks += duration;
    if (duration > state.snapshot.max_wait_ticks) {
        state.snapshot.max_wait_ticks = duration;
    }
}

void print_allocation_wait_diagnostics() {
    AllocationWaitDiagnosticsSnapshot snapshot;
    {
        auto &state = allocation_wait_diag_state;
        std::lock_guard<std::mutex> lock(state.mutex);
        snapshot = state.snapshot;
    }
    const bool sane = !snapshot.enabled ||
                      (snapshot.frozen && snapshot.active_at_end == 0 &&
                       snapshot.unclosed_events == 0 &&
                       snapshot.over_capacity_events == 0 &&
                       snapshot.invalid_end_events == 0 &&
                       snapshot.ignored_events == 0 &&
                       snapshot.timestamp_regressions == 0 &&
                       snapshot.late_end_events == 0 &&
                       snapshot.condition_wait_ticks <= snapshot.wait_ticks);
    std::cout << "allocation_wait_diag"
              << " enabled=" << (snapshot.enabled ? 1 : 0)
              << " active=" << (snapshot.active ? 1 : 0)
              << " frozen=" << (snapshot.frozen ? 1 : 0)
              << " status="
              << (!snapshot.enabled ? "disabled" : sane ? "ok" : "anomaly")
              << " expected_fibres=" << snapshot.expected_fibre_count
              << " event_count=" << snapshot.event_count
              << " completed_event_count=" << snapshot.completed_event_count
              << " wait_ticks=" << snapshot.wait_ticks
              << " max_wait_ticks=" << snapshot.max_wait_ticks
              << " union_ticks=" << snapshot.union_ticks
              << " full_ticks=" << snapshot.full_ticks
              << " max_active=" << snapshot.max_active
              << " active_at_end=" << snapshot.active_at_end
              << " unclosed_events=" << snapshot.unclosed_events
              << " ignored_events=" << snapshot.ignored_events
              << " over_capacity_events="
              << snapshot.over_capacity_events
              << " invalid_end_events=" << snapshot.invalid_end_events
              << " timestamp_regressions="
              << snapshot.timestamp_regressions
              << " late_end_events=" << snapshot.late_end_events
              << " condition_wait_count=" << snapshot.condition_wait_count
              << " condition_wait_ticks=" << snapshot.condition_wait_ticks
              << " condition_ready_before_sleep_count="
              << snapshot.condition_ready_before_sleep_count
              << " condition_not_ready_after_wake_count="
              << snapshot.condition_not_ready_after_wake_count
              << std::endl;
}

void ThreadLocalProfileData::register_thread() {
    static std::mutex mtx;
    mtx.lock();
    profile_data_map[std::this_thread::get_id()] = this;
    mtx.unlock();
}

void ThreadLocalProfileData::unregister_thread() {
    global_profile_data.work_cycles += work_cycles;
    global_profile_data.allocate_cycles += allocate_cycles;
    global_profile_data.post_fetch_cycles += post_fetch_cycles;
    global_profile_data.poll_cycles += poll_cycles;
    global_profile_data.core_switch_count += core_switch_count;
    global_profile_data.yield_cycles += yield_cycles;
    global_profile_data.data_miss_count += data_miss_count;
    global_profile_data.yield_count += yield_count;
    global_profile_data.mark_cycles += mark_cycles;
    global_profile_data.evict_cycles += evict_cycles;
    global_profile_data.evacuation_count += evacuation_count;
    global_profile_data.evacuation_bytes += evacuation_bytes;
    global_profile_data.fork_join_cycles += fork_join_cycles;
    global_profile_data.check_cq_cycles += check_cq_cycles;
    global_profile_data.check_cq_count += check_cq_count;
    global_profile_data.mark_count += mark_count;
    global_profile_data.mark_payload_bytes += mark_payload_bytes;
    global_profile_data.not_mark_count += not_mark_count;
    global_profile_data.on_miss_cycles += on_miss_cycles;
    global_profile_data.left_for_use1_cycles += left_for_use1_cycles;
    global_profile_data.left_for_use2_cycles += left_for_use2_cycles;
    global_profile_data.left_for_use3_cycles += left_for_use3_cycles;
    global_profile_data.left_for_use4_cycles += left_for_use4_cycles;
    global_profile_data.obj_modifed_cnt += obj_modifed_cnt;
    global_profile_data.obj_unmodifed_cnt += obj_unmodifed_cnt;
    global_profile_data.evacuator_master_cycles += evacuator_master_cycles;
    global_profile_data.stw_mutator_cycles += stw_mutator_cycles;
    global_profile_data.flip_scope_cycles += flip_scope_cycles;
    global_profile_data.flip_scope_yield_cycles += flip_scope_yield_cycles;
    global_profile_data.gc_cycles += gc_cycles;
    global_profile_data.gc_count += gc_count;

    global_profile_data.evac_mark_phase_cycles += evac_mark_phase_cycles;
    global_profile_data.evac_evict_phase_cycles += evac_evict_phase_cycles;
    global_profile_data.evac_gc_phase_cycles += evac_gc_phase_cycles;
    global_profile_data.evac_flip_scope_cycles += evac_flip_scope_cycles;
    global_profile_data.evac_flush_cycles += evac_flush_cycles;
    global_profile_data.evac_check_cq_wait_cycles += evac_check_cq_wait_cycles;
    global_profile_data.evac_evict_post_cycles += evac_evict_post_cycles;
    global_profile_data.evac_drain_cycles += evac_drain_cycles;
    global_profile_data.evac_try_evict_cycles += evac_try_evict_cycles;
    global_profile_data.evac_post_write_build_wr_cycles +=
        evac_post_write_build_wr_cycles;
    global_profile_data.evac_post_write_post_send_cycles +=
        evac_post_write_post_send_cycles;
    global_profile_data.evac_post_write_retry_count +=
        evac_post_write_retry_count;
    global_profile_data.evac_write_complete_count += evac_write_complete_count;
    global_profile_data.evac_write_complete_cycles +=
        evac_write_complete_cycles;
    global_profile_data.evac_overlap_mark_cycles += evac_overlap_mark_cycles;
    global_profile_data.evac_pipeline_idle_cycles += evac_pipeline_idle_cycles;
    global_profile_data.evac_pipeline_rounds += evac_pipeline_rounds;
    global_profile_data.evac_streaming_post_blocked_cycles += evac_streaming_post_blocked_cycles;
    global_profile_data.evac_streaming_wait_safe_epoch_cycles +=
        evac_streaming_wait_safe_epoch_cycles;
    global_profile_data.evac_streaming_safe_epoch_zero_rounds +=
        evac_streaming_safe_epoch_zero_rounds;
    global_profile_data.evac_streaming_safe_epoch_zero_tasks +=
        evac_streaming_safe_epoch_zero_tasks;
    global_profile_data.evac_try_evict_count += evac_try_evict_count;
    global_profile_data.evac_touched_after_mark_count +=
        evac_touched_after_mark_count;
    global_profile_data.evac_flip_wait_new_loops += evac_flip_wait_new_loops;
    global_profile_data.evac_flip_wait_old_loops += evac_flip_wait_old_loops;
    if (evac_flip_wait_new_max > global_profile_data.evac_flip_wait_new_max) {
        global_profile_data.evac_flip_wait_new_max = evac_flip_wait_new_max;
    }
    if (evac_flip_wait_old_max > global_profile_data.evac_flip_wait_old_max) {
        global_profile_data.evac_flip_wait_old_max = evac_flip_wait_old_max;
    }
    global_profile_data.evac_flip_wait_new_cycles += evac_flip_wait_new_cycles;
    global_profile_data.evac_flip_wait_old_cycles += evac_flip_wait_old_cycles;
    global_profile_data.evac_flip_wait_new_blocked_flips +=
        evac_flip_wait_new_blocked_flips;
    global_profile_data.evac_flip_wait_old_blocked_flips +=
        evac_flip_wait_old_blocked_flips;
    global_profile_data.evac_flip_wait_new_entry_sum += evac_flip_wait_new_entry_sum;
    global_profile_data.evac_flip_wait_old_entry_sum += evac_flip_wait_old_entry_sum;
    global_profile_data.evac_flip_wait_new_observed_sum +=
        evac_flip_wait_new_observed_sum;
    global_profile_data.evac_flip_wait_old_observed_sum +=
        evac_flip_wait_old_observed_sum;
    global_profile_data.evac_evict_rounds += evac_evict_rounds;
    global_profile_data.evac_evict_active_rounds += evac_evict_active_rounds;
    global_profile_data.evac_evict_empty_rounds += evac_evict_empty_rounds;
    global_profile_data.evac_evict_active_cycles += evac_evict_active_cycles;
    global_profile_data.evac_evict_empty_cycles += evac_evict_empty_cycles;
    global_profile_data.evac_evict_active_wrs += evac_evict_active_wrs;
    global_profile_data.evac_try_evict_pinned += evac_try_evict_pinned;
    global_profile_data.evac_try_evict_free_or_mismatch +=
        evac_try_evict_free_or_mismatch;
    global_profile_data.evac_try_evict_marked_dirty +=
        evac_try_evict_marked_dirty;
    global_profile_data.evac_try_evict_marked_clean +=
        evac_try_evict_marked_clean;
    global_profile_data.evac_try_evict_state_local +=
        evac_try_evict_state_local;
    global_profile_data.evac_try_evict_state_fetching +=
        evac_try_evict_state_fetching;
    global_profile_data.evac_try_evict_state_remote +=
        evac_try_evict_state_remote;
    global_profile_data.evac_try_evict_state_busy += evac_try_evict_state_busy;
    global_profile_data.evac_try_evict_state_other +=
        evac_try_evict_state_other;
    global_profile_data.evac_evict_pop_empty += evac_evict_pop_empty;
    global_profile_data.evac_evict_pop_head_matched +=
        evac_evict_pop_head_matched;
    global_profile_data.evac_evict_pop_success += evac_evict_pop_success;
    global_profile_data.evac_mark_regions_scanned += evac_mark_regions_scanned;
    global_profile_data.evac_mark_regions_with_output += evac_mark_regions_with_output;
    global_profile_data.evac_mark_regions_without_output +=
        evac_mark_regions_without_output;
    global_profile_data.evac_mark_regions_skipped_in_use +=
        evac_mark_regions_skipped_in_use;
    global_profile_data.evac_mark_regions_skipped_stable +=
        evac_mark_regions_skipped_stable;
    global_profile_data.evac_evict_regions_skipped_in_use +=
        evac_evict_regions_skipped_in_use;
    global_profile_data.evac_evict_regions_to_free += evac_evict_regions_to_free;
    global_profile_data.evac_evict_regions_to_usable += evac_evict_regions_to_usable;
    global_profile_data.evac_evict_regions_to_full += evac_evict_regions_to_full;

    global_profile_data.excl_remote_alloc_count += excl_remote_alloc_count;
    global_profile_data.excl_remote_alloc_bytes += excl_remote_alloc_bytes;
    global_profile_data.excl_remote_alloc_cycles += excl_remote_alloc_cycles;
    global_profile_data.excl_remote_free_on_fetch_count +=
        excl_remote_free_on_fetch_count;
    global_profile_data.excl_remote_free_on_fetch_bytes +=
        excl_remote_free_on_fetch_bytes;
    global_profile_data.excl_remote_free_on_fetch_cycles +=
        excl_remote_free_on_fetch_cycles;
    global_profile_data.excl_remote_free_on_fetch_slow_count +=
        excl_remote_free_on_fetch_slow_count;
    global_profile_data.excl_remote_free_on_fetch_slow_bytes +=
        excl_remote_free_on_fetch_slow_bytes;
    global_profile_data.excl_remote_free_on_fetch_slow_cycles +=
        excl_remote_free_on_fetch_slow_cycles;
    global_profile_data.excl_interrupted_evict_free_count +=
        excl_interrupted_evict_free_count;
    global_profile_data.excl_interrupted_evict_free_bytes +=
        excl_interrupted_evict_free_bytes;
    global_profile_data.excl_move_lock_spin_count += excl_move_lock_spin_count;
    global_profile_data.excl_move_lock_spin_cycles += excl_move_lock_spin_cycles;

    global_profile_data.remote_backup_admission_attempts +=
        remote_backup_admission_attempts;
    global_profile_data.remote_backup_retained_count +=
        remote_backup_retained_count;
    global_profile_data.remote_backup_retained_bytes +=
        remote_backup_retained_bytes;
    global_profile_data.remote_backup_rejected_count +=
        remote_backup_rejected_count;
    global_profile_data.remote_backup_rejected_bytes +=
        remote_backup_rejected_bytes;
    global_profile_data.remote_backup_reused_count += remote_backup_reused_count;
    global_profile_data.remote_backup_reused_bytes += remote_backup_reused_bytes;
    global_profile_data.remote_backup_invalidated_count +=
        remote_backup_invalidated_count;
    global_profile_data.remote_backup_invalidated_bytes +=
        remote_backup_invalidated_bytes;

    global_profile_data.remote_dealloc_count += remote_dealloc_count;
    global_profile_data.remote_dealloc_lock_cycles += remote_dealloc_lock_cycles;
    global_profile_data.remote_dealloc_lock_hold_cycles +=
        remote_dealloc_lock_hold_cycles;
    global_profile_data.remote_dealloc_bitmap_cycles +=
        remote_dealloc_bitmap_cycles;
    global_profile_data.remote_dealloc_list_cycles += remote_dealloc_list_cycles;
    global_profile_data.remote_dealloc_total_cycles += remote_dealloc_total_cycles;

    global_profile_data.rdma_read_post_count += rdma_read_post_count;
    global_profile_data.rdma_read_post_bytes += rdma_read_post_bytes;
    if (read_size_profile::enabled()) {
        std::lock_guard<std::mutex> lock(read_size_histogram_mutex);
        global_profile_data.rdma_read_size_histogram->merge(
            *rdma_read_size_histogram);
    }
    global_profile_data.rdma_write_post_count += rdma_write_post_count;
    global_profile_data.rdma_write_post_bytes += rdma_write_post_bytes;
    global_profile_data.dirty_evict_bytes += dirty_evict_bytes;
    global_profile_data.clean_evict_bytes += clean_evict_bytes;
    global_profile_data.ref_read_cold_count += ref_read_cold_count;
    global_profile_data.ref_read_warm_count += ref_read_warm_count;
    global_profile_data.ref_read_hot_count += ref_read_hot_count;
    global_profile_data.ref_write_cold_count += ref_write_cold_count;
    global_profile_data.ref_write_warm_count += ref_write_warm_count;
    global_profile_data.ref_write_hot_count += ref_write_hot_count;
    global_profile_data.ref_clean_cold_count += ref_clean_cold_count;
    global_profile_data.ref_clean_warm_count += ref_clean_warm_count;
    global_profile_data.ref_clean_hot_count += ref_clean_hot_count;
    global_profile_data.ref_dirty_cold_count += ref_dirty_cold_count;
    global_profile_data.ref_dirty_warm_count += ref_dirty_warm_count;
    global_profile_data.ref_dirty_hot_count += ref_dirty_hot_count;
    global_profile_data.resident_profile_reference_checks +=
        resident_profile_reference_checks;
    global_profile_data.resident_profile_samples += resident_profile_samples;
    global_profile_data.resident_profile_sample_cycles +=
        resident_profile_sample_cycles;
    global_profile_data.resident_profile_planner_collection_cycles +=
        resident_profile_planner_collection_cycles;
    global_profile_data.resident_profile_planner_candidate_cycles +=
        resident_profile_planner_candidate_cycles;
    global_profile_data.resident_profile_planner_publish_cycles +=
        resident_profile_planner_publish_cycles;
    global_profile_data.resident_profile_promotion_attempts +=
        resident_profile_promotion_attempts;
    global_profile_data.resident_profile_promotion_successes +=
        resident_profile_promotion_successes;
    global_profile_data.resident_profile_promotion_cycles +=
        resident_profile_promotion_cycles;
    global_profile_data.resident_profile_demotion_attempts +=
        resident_profile_demotion_attempts;
    global_profile_data.resident_profile_demotion_successes +=
        resident_profile_demotion_successes;
    global_profile_data.resident_profile_demotion_cycles +=
        resident_profile_demotion_cycles;
    global_profile_data.resident_profile_region_placement_attempts +=
        resident_profile_region_placement_attempts;
    global_profile_data.resident_profile_region_placement_cycles +=
        resident_profile_region_placement_cycles;

    profile_data_map.erase(std::this_thread::get_id());
}

// Requires the existing profile-registry quiescence contract. NQ invokes this
// after its query fork_join and progress-thread join, before runtime teardown.
// Use heap storage: an 8192-counter snapshot must not consume a Fibre's 64KiB stack.
void print_rdma_read_size_histogram() {
    if (!read_size_profile::enabled()) return;
    if (is_working()) {
        std::cout << "read_size_histogram status=invalid_live_snapshot" << std::endl;
        return;
    }
    auto snapshot = std::make_unique<read_size_profile::Histogram>();
    {
        std::lock_guard<std::mutex> lock(read_size_histogram_mutex);
        snapshot->merge(*global_profile_data.rdma_read_size_histogram);
        for (const auto &it : profile_data_map) {
            snapshot->merge(*it.second->rdma_read_size_histogram);
        }
    }
    uint64_t count = 0, bytes = 0;
    for (size_t i = 0; i < read_size_profile::BucketCount; ++i) {
        const uint64_t n = snapshot->counts[i];
        if (n == 0) continue;
        const uint64_t payload = i * read_size_profile::Quantum;
        count += n;
        bytes += n * payload;
        std::cout << "read_size_bucket payload_bytes=" << payload
                  << " count=" << n << " read_bytes=" << n * payload
                  << '\n';
    }
    std::cout << "read_size_histogram status=post_work enabled=1"
              << " scope=since_profile_reset quantum_bytes="
              << read_size_profile::Quantum
              << " represented_count=" << count
              << " represented_bytes=" << bytes
              << " unbucketed_count=" << snapshot->unbucketed_count
              << " unbucketed_bytes=" << snapshot->unbucketed_bytes
              << " rdma_read_post_count=" << collect_rdma_read_post_count()
              << " rdma_read_post_bytes=" << collect_rdma_read_post_bytes()
              << std::endl;
}

void reset_all() {
    global_profile_data.reset();
    evac_active_worker_count.store(0, std::memory_order_relaxed);
    for (auto &it : profile_data_map) {
        it.second->reset();
    }
    {
        std::lock_guard<std::mutex> lock(frequency_histogram_mutex);
        frequency_histogram_history.clear();
    }
    {
        std::lock_guard<std::mutex> lock(full_population_frequency_histogram_mutex);
        full_population_frequency_history.clear();
    }
    {
        std::lock_guard<std::mutex> lock(frequency_output_scope_mutex);
        frequency_output_scope_label.clear();
    }
    frequency_output_scope_explicit.store(false, std::memory_order_relaxed);
    frequency_output_scope_active.store(true, std::memory_order_relaxed);
    reference_heat_cold_max_bucket.store(2, std::memory_order_relaxed);
    reference_heat_warm_max_bucket.store(5, std::memory_order_relaxed);
    global_cycles = 0;
}

void prepare_frequency_output_scope(const std::string &label) {
    frequency_output_scope_explicit.store(true, std::memory_order_relaxed);
    frequency_output_scope_active.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(frequency_histogram_mutex);
        frequency_histogram_history.clear();
    }
    {
        std::lock_guard<std::mutex> lock(full_population_frequency_histogram_mutex);
        full_population_frequency_history.clear();
    }
    {
        std::lock_guard<std::mutex> lock(frequency_output_scope_mutex);
        frequency_output_scope_label = label;
    }
}

void begin_frequency_output_scope(const std::string &label) {
    if (!frequency_output_scope_explicit.load(std::memory_order_relaxed)) {
        prepare_frequency_output_scope(label);
    } else if (!label.empty()) {
        std::lock_guard<std::mutex> lock(frequency_output_scope_mutex);
        frequency_output_scope_label = label;
    }
    frequency_output_scope_active.store(true, std::memory_order_relaxed);
}

void end_frequency_output_scope() {
    frequency_output_scope_active.store(false, std::memory_order_relaxed);
}

bool should_record_frequency_output() {
    return !frequency_output_scope_explicit.load(std::memory_order_relaxed) ||
           frequency_output_scope_active.load(std::memory_order_relaxed);
}

std::string current_frequency_output_scope_label() {
    if (!frequency_output_scope_explicit.load(std::memory_order_relaxed)) {
        return {};
    }
    std::lock_guard<std::mutex> lock(frequency_output_scope_mutex);
    return frequency_output_scope_label;
}

void record_frequency_histogram_pass(
    uint32_t timestamp, const DualFrequencyHistogramSnapshot &local_snapshot) {
    if (!should_record_frequency_output()) {
        return;
    }
    std::lock_guard<std::mutex> lock(frequency_histogram_mutex);
    if (frequency_histogram_history.empty() ||
        frequency_histogram_history.back().timestamp != timestamp) {
        frequency_histogram_history.push_back(DualFrequencyHistogramSnapshot{});
        frequency_histogram_history.back().timestamp = timestamp;
    }
    frequency_histogram_history.back().merge_from(local_snapshot);
    update_reference_heat_thresholds(frequency_histogram_history.back().ema_histogram);
}

std::vector<DualFrequencyHistogramSnapshot> collect_frequency_histogram_history() {
    std::lock_guard<std::mutex> lock(frequency_histogram_mutex);
    return frequency_histogram_history;
}

void record_full_population_frequency_snapshot(
    const FullPopulationFrequencySnapshot &snapshot) {
    if (!should_record_frequency_output()) {
        return;
    }
    std::lock_guard<std::mutex> lock(full_population_frequency_histogram_mutex);
    full_population_frequency_history.push_back(snapshot);
    update_reference_heat_thresholds(snapshot.ema_histogram);
}

std::vector<FullPopulationFrequencySnapshot>
collect_full_population_frequency_history() {
    std::lock_guard<std::mutex> lock(full_population_frequency_histogram_mutex);
    return full_population_frequency_history;
}

uint32_t current_reference_heat_cold_max_bucket() {
    return reference_heat_cold_max_bucket.load(std::memory_order_relaxed);
}

uint32_t current_reference_heat_warm_max_bucket() {
    return reference_heat_warm_max_bucket.load(std::memory_order_relaxed);
}

const char *current_reference_heat_split_mode() {
    return reference_heat_split_access_weighted() ? "access_weighted"
                                                  : "object_count";
}

namespace {

void print_frequency_histogram_snapshot(const char *label,
                                        const FrequencyHistogramSnapshot &snapshot) {
    std::cout << label << " total=" << format_with_commas(snapshot.total_objects)
              << " zero=" << format_with_commas(snapshot.zero_frequency_objects)
              << std::endl;
    std::cout << label << " bin 0 [0]: "
              << format_with_commas(snapshot.zero_frequency_objects)
              << " objects" << std::endl;
    for (size_t i = 0; i < snapshot.buckets.size(); i++) {
        const uint32_t lower = (i == 0) ? 1u : (1u << i);
        const uint32_t upper =
            (i + 1 == snapshot.buckets.size()) ? UINT32_MAX : ((1u << (i + 1)) - 1u);
        std::cout << label << " bin " << (i + 1) << " [" << lower << ","
                  << upper << "]: "
                  << format_with_commas(snapshot.buckets[i]) << " objects"
                  << std::endl;
    }
}

}  // namespace

void print_frequency_histogram_history() {
    auto history = collect_frequency_histogram_history();
    if (history.empty()) {
        return;
    }
    const auto label = current_frequency_output_scope_label();
    const std::string title =
        label.empty() ? "resident frequency histogram"
                      : ("resident frequency histogram [" + label + "]");
    std::cout << std::setw(32) << std::internal << title << std::endl;
    std::cout << std::right;
    std::cout << std::string(32, '-') << std::endl;
    for (const auto &snapshot : history) {
        std::cout << "mark pass " << snapshot.timestamp
                  << " ordinal=" << snapshot.mark_pass_ordinal
                  << " ema interval=" << snapshot.ema_mark_interval
                  << " ema update="
                  << (snapshot.ema_updated ? "yes" : "no") << std::endl;
        print_frequency_histogram_snapshot("ema frequency",
                                           snapshot.ema_histogram);
        print_frequency_histogram_snapshot("window frequency",
                                           snapshot.window_histogram);
    }
}

void print_full_population_frequency_history() {
    auto history = collect_full_population_frequency_history();
    if (history.empty()) {
        return;
    }
    const auto label = current_frequency_output_scope_label();
    const std::string title =
        label.empty() ? "full population frequency histogram"
                      : ("full population frequency histogram [" + label + "]");
    std::cout << std::setw(32) << std::internal << title << std::endl;
    std::cout << std::right;
    std::cout << std::string(32, '-') << std::endl;
    for (const auto &snapshot : history) {
        std::cout << "async full-pop snapshot " << snapshot.scan_sequence
                  << " period_ms=" << snapshot.scan_period_ms
                  << " scan_us=" << snapshot.scan_duration_us
                  << " observed ordinal="
                  << snapshot.observed_mark_pass_ordinal
                  << std::endl;
        print_frequency_histogram_snapshot("full-pop ema frequency",
                                           snapshot.ema_histogram);
        print_frequency_histogram_snapshot("full-pop window frequency",
                                           snapshot.window_histogram);
    }
}

#define DEFINE_COLLECT(FIELD)                    \
    int64_t collect_##FIELD() {                  \
        int64_t sum = global_profile_data.FIELD; \
        for (auto &it : profile_data_map) {      \
            sum += it.second->FIELD;             \
        }                                        \
        return sum;                              \
    }

#define DEFINE_COLLECT_EACH(FIELD)                    \
    int64_t collect_##FIELD() {                  \
        std::cout << std::setw(16) << #FIELD << std::setw(16) << global_profile_data.FIELD << std::endl; \
        for (auto &it : profile_data_map) {      \
            std::cout << std::setw(16) << #FIELD << std::setw(16) << "Thread ID " << it.first << ": " << it.second->FIELD << std::endl; \
        }                                        \
    }


DEFINE_COLLECT(work_cycles)
DEFINE_COLLECT(allocate_cycles);
DEFINE_COLLECT(post_fetch_cycles);
DEFINE_COLLECT(poll_cycles);
DEFINE_COLLECT(yield_cycles);
DEFINE_COLLECT(data_miss_count);
DEFINE_COLLECT(yield_count);
DEFINE_COLLECT(mark_cycles);
DEFINE_COLLECT(evict_cycles);
DEFINE_COLLECT(evacuation_count);
DEFINE_COLLECT(evacuation_bytes);
DEFINE_COLLECT(fork_join_cycles);
DEFINE_COLLECT(check_cq_cycles);
DEFINE_COLLECT(check_cq_count);
DEFINE_COLLECT(mark_count);
DEFINE_COLLECT(mark_payload_bytes);
DEFINE_COLLECT(not_mark_count);
DEFINE_COLLECT(on_miss_cycles);
DEFINE_COLLECT(prefetch_cycles);
DEFINE_COLLECT(core_switch_count);
DEFINE_COLLECT(left_for_use1_cycles);
DEFINE_COLLECT(left_for_use2_cycles);
DEFINE_COLLECT(left_for_use3_cycles);
DEFINE_COLLECT(left_for_use4_cycles);
DEFINE_COLLECT(obj_modifed_cnt);
DEFINE_COLLECT(obj_unmodifed_cnt);
DEFINE_COLLECT(evacuator_master_cycles);
DEFINE_COLLECT(stw_mutator_cycles);
DEFINE_COLLECT(flip_scope_cycles);
DEFINE_COLLECT(flip_scope_yield_cycles);
DEFINE_COLLECT(gc_cycles);
DEFINE_COLLECT(gc_count);

DEFINE_COLLECT(evac_mark_phase_cycles);
DEFINE_COLLECT(evac_evict_phase_cycles);
DEFINE_COLLECT(evac_gc_phase_cycles);
DEFINE_COLLECT(evac_flip_scope_cycles);
DEFINE_COLLECT(evac_flush_cycles);
DEFINE_COLLECT(evac_flush_count);
DEFINE_COLLECT(evac_flush_full_count);
DEFINE_COLLECT(evac_flush_partial_count);
DEFINE_COLLECT(evac_flush_wrs);
DEFINE_COLLECT(evac_check_cq_wait_cycles);
DEFINE_COLLECT(evac_evict_post_cycles);
DEFINE_COLLECT(evac_drain_cycles);
DEFINE_COLLECT(evac_try_evict_cycles);
DEFINE_COLLECT(evac_post_write_build_wr_cycles);
DEFINE_COLLECT(evac_post_write_post_send_cycles);
DEFINE_COLLECT(evac_post_write_retry_count);
DEFINE_COLLECT(evac_write_complete_count);
DEFINE_COLLECT(evac_write_complete_cycles);
DEFINE_COLLECT(evac_overlap_mark_cycles);
DEFINE_COLLECT(evac_pipeline_idle_cycles);
DEFINE_COLLECT(evac_pipeline_rounds);
DEFINE_COLLECT(evac_streaming_post_blocked_cycles);
DEFINE_COLLECT(evac_streaming_wait_safe_epoch_cycles);
DEFINE_COLLECT(evac_streaming_safe_epoch_zero_rounds);
DEFINE_COLLECT(evac_streaming_safe_epoch_zero_tasks);
DEFINE_COLLECT(evac_try_evict_count);
DEFINE_COLLECT(evac_touched_after_mark_count);
DEFINE_COLLECT(evac_flip_wait_new_loops);
DEFINE_COLLECT(evac_flip_wait_old_loops);
DEFINE_COLLECT(evac_flip_wait_new_max);
DEFINE_COLLECT(evac_flip_wait_old_max);
DEFINE_COLLECT(evac_flip_wait_new_cycles);
DEFINE_COLLECT(evac_flip_wait_old_cycles);
DEFINE_COLLECT(evac_flip_wait_new_blocked_flips);
DEFINE_COLLECT(evac_flip_wait_old_blocked_flips);
DEFINE_COLLECT(evac_flip_wait_new_entry_sum);
DEFINE_COLLECT(evac_flip_wait_old_entry_sum);
DEFINE_COLLECT(evac_flip_wait_new_observed_sum);
DEFINE_COLLECT(evac_flip_wait_old_observed_sum);
DEFINE_COLLECT(evac_evict_rounds);
DEFINE_COLLECT(evac_evict_active_rounds);
DEFINE_COLLECT(evac_evict_empty_rounds);
DEFINE_COLLECT(evac_evict_active_cycles);
DEFINE_COLLECT(evac_evict_empty_cycles);
DEFINE_COLLECT(evac_evict_active_wrs);
DEFINE_COLLECT(evac_try_evict_pinned);
DEFINE_COLLECT(evac_try_evict_free_or_mismatch);
DEFINE_COLLECT(evac_try_evict_marked_dirty);
DEFINE_COLLECT(evac_try_evict_marked_clean);
DEFINE_COLLECT(evac_try_evict_state_local);
DEFINE_COLLECT(evac_try_evict_state_fetching);
DEFINE_COLLECT(evac_try_evict_state_remote);
DEFINE_COLLECT(evac_try_evict_state_busy);
DEFINE_COLLECT(evac_try_evict_state_other);
DEFINE_COLLECT(evac_evict_pop_empty);
DEFINE_COLLECT(evac_evict_pop_head_matched);
DEFINE_COLLECT(evac_evict_pop_success);
DEFINE_COLLECT(evac_mark_regions_scanned);
DEFINE_COLLECT(evac_mark_regions_with_output);
DEFINE_COLLECT(evac_mark_regions_without_output);
DEFINE_COLLECT(evac_mark_regions_skipped_in_use);
DEFINE_COLLECT(evac_mark_regions_skipped_stable);
DEFINE_COLLECT(evac_evict_regions_skipped_in_use);
DEFINE_COLLECT(evac_evict_regions_to_free);
DEFINE_COLLECT(evac_evict_regions_to_usable);
DEFINE_COLLECT(evac_evict_regions_to_full);

DEFINE_COLLECT(excl_remote_alloc_count);
DEFINE_COLLECT(excl_remote_alloc_bytes);
DEFINE_COLLECT(excl_remote_alloc_cycles);
DEFINE_COLLECT(excl_remote_free_on_fetch_count);
DEFINE_COLLECT(excl_remote_free_on_fetch_bytes);
DEFINE_COLLECT(excl_remote_free_on_fetch_cycles);
DEFINE_COLLECT(excl_remote_free_on_fetch_slow_count);
DEFINE_COLLECT(excl_remote_free_on_fetch_slow_bytes);
DEFINE_COLLECT(excl_remote_free_on_fetch_slow_cycles);
DEFINE_COLLECT(excl_interrupted_evict_free_count);
DEFINE_COLLECT(excl_interrupted_evict_free_bytes);
DEFINE_COLLECT(excl_move_lock_spin_count);
DEFINE_COLLECT(excl_move_lock_spin_cycles);

DEFINE_COLLECT(remote_backup_admission_attempts);
DEFINE_COLLECT(remote_backup_retained_count);
DEFINE_COLLECT(remote_backup_retained_bytes);
DEFINE_COLLECT(remote_backup_rejected_count);
DEFINE_COLLECT(remote_backup_rejected_bytes);
DEFINE_COLLECT(remote_backup_reused_count);
DEFINE_COLLECT(remote_backup_reused_bytes);
DEFINE_COLLECT(remote_backup_invalidated_count);
DEFINE_COLLECT(remote_backup_invalidated_bytes);

DEFINE_COLLECT(remote_dealloc_count);
DEFINE_COLLECT(remote_dealloc_lock_cycles);
DEFINE_COLLECT(remote_dealloc_lock_hold_cycles);
DEFINE_COLLECT(remote_dealloc_bitmap_cycles);
DEFINE_COLLECT(remote_dealloc_list_cycles);
DEFINE_COLLECT(remote_dealloc_total_cycles);

DEFINE_COLLECT(rdma_read_post_count);
DEFINE_COLLECT(rdma_read_post_bytes);
DEFINE_COLLECT(rdma_write_post_count);
DEFINE_COLLECT(rdma_write_post_bytes);
DEFINE_COLLECT(dirty_evict_bytes);
DEFINE_COLLECT(clean_evict_bytes);
DEFINE_COLLECT(ref_read_cold_count);
DEFINE_COLLECT(ref_read_warm_count);
DEFINE_COLLECT(ref_read_hot_count);
DEFINE_COLLECT(ref_write_cold_count);
DEFINE_COLLECT(ref_write_warm_count);
DEFINE_COLLECT(ref_write_hot_count);
DEFINE_COLLECT(ref_clean_cold_count);
DEFINE_COLLECT(ref_clean_warm_count);
DEFINE_COLLECT(ref_clean_hot_count);
DEFINE_COLLECT(ref_dirty_cold_count);
DEFINE_COLLECT(ref_dirty_warm_count);
DEFINE_COLLECT(ref_dirty_hot_count);
DEFINE_COLLECT(resident_profile_reference_checks);
DEFINE_COLLECT(resident_profile_samples);
DEFINE_COLLECT(resident_profile_sample_cycles);
DEFINE_COLLECT(resident_profile_planner_collection_cycles);
DEFINE_COLLECT(resident_profile_planner_candidate_cycles);
DEFINE_COLLECT(resident_profile_planner_publish_cycles);
DEFINE_COLLECT(resident_profile_promotion_attempts);
DEFINE_COLLECT(resident_profile_promotion_successes);
DEFINE_COLLECT(resident_profile_promotion_cycles);
DEFINE_COLLECT(resident_profile_demotion_attempts);
DEFINE_COLLECT(resident_profile_demotion_successes);
DEFINE_COLLECT(resident_profile_demotion_cycles);
DEFINE_COLLECT(resident_profile_region_placement_attempts);
DEFINE_COLLECT(resident_profile_region_placement_cycles);

void print_profile_data() {
    constexpr bool PrintEnabled =
        Enabled || enabled::Evacuation || enabled::YieldCount;
    if constexpr (!(PrintEnabled)) return;
#define PRINT_IF(NAME, VALUE, ENABLED)                               \
    if constexpr (ENABLED) {                                         \
        auto val = (VALUE);                                          \
        std::cout << std::setw(20) << NAME << std::setw(20)          \
                  << format_with_commas(val) << " ("                 \
                  << format_scientific(val) << ")" << std::endl;     \
    }
    auto print_ratio_line = [](const char *label, int64_t lhs, int64_t rhs,
                               const char *lhs_name, const char *rhs_name) {
        const int64_t total = lhs + rhs;
        const double rhs_share =
            total == 0 ? 0.0 : (100.0 * static_cast<double>(rhs) / total);
        std::cout << label << " " << lhs_name << "=" << format_with_commas(lhs)
                  << " " << rhs_name << "=" << format_with_commas(rhs)
                  << " " << rhs_name << "_share=" << std::fixed
                  << std::setprecision(2) << rhs_share << "%" << std::endl;
        std::cout.unsetf(std::ios::floatfield);
    };
    auto print_heat_split = []() {
        const uint32_t cold_bucket = current_reference_heat_cold_max_bucket();
        const uint32_t warm_bucket = current_reference_heat_warm_max_bucket();
        std::cout << "ref heat split mode=" << current_reference_heat_split_mode()
                  << " cold=[0," << bucket_upper_bound(cold_bucket)
                  << "] warm=[" << bucket_lower_bound(cold_bucket + 1) << ","
                  << bucket_upper_bound(warm_bucket) << "] hot=["
                  << bucket_lower_bound(warm_bucket + 1) << ",inf]" << std::endl;
    };
    PRINT_IF("wall time cycles", global_cycles, PrintEnabled);
    std::cout << std::setw(32) << std::internal << "breakdown" << std::endl;
    std::cout << std::right;
    std::cout << std::string(32, '-') << std::endl;
    PRINT_IF("work cycles", collect_work_cycles(), Enabled);
    PRINT_IF("alloc cycles", collect_allocate_cycles(), Enabled);
    PRINT_IF("post cycles", collect_post_fetch_cycles(), Enabled);
    PRINT_IF("poll cycles", collect_poll_cycles(), Enabled);
    PRINT_IF("core switch count", collect_core_switch_count(), Enabled);
    PRINT_IF("yield cycles", collect_yield_cycles(), Enabled);
    PRINT_IF("miss count", collect_data_miss_count(), Enabled);
    PRINT_IF("yield count", collect_yield_count(), enabled::YieldCount);
    PRINT_IF("mark cycles", collect_mark_cycles(), enabled::Evacuation);
    PRINT_IF("evict cycles", collect_evict_cycles(), enabled::Evacuation);
    PRINT_IF("evacuate count", collect_evacuation_count(), enabled::Evacuation);
    PRINT_IF("evacuate bytes", collect_evacuation_bytes(), enabled::Evacuation);
    PRINT_IF("dirty evict bytes", collect_dirty_evict_bytes(), Enabled);
    PRINT_IF("clean evict bytes", collect_clean_evict_bytes(), Enabled);
    PRINT_IF("ref read cold", collect_ref_read_cold_count(), Enabled);
    PRINT_IF("ref read warm", collect_ref_read_warm_count(), Enabled);
    PRINT_IF("ref read hot", collect_ref_read_hot_count(), Enabled);
    PRINT_IF("ref write cold", collect_ref_write_cold_count(), Enabled);
    PRINT_IF("ref write warm", collect_ref_write_warm_count(), Enabled);
    PRINT_IF("ref write hot", collect_ref_write_hot_count(), Enabled);
    PRINT_IF("ref clean cold", collect_ref_clean_cold_count(), Enabled);
    PRINT_IF("ref clean warm", collect_ref_clean_warm_count(), Enabled);
    PRINT_IF("ref clean hot", collect_ref_clean_hot_count(), Enabled);
    PRINT_IF("ref dirty cold", collect_ref_dirty_cold_count(), Enabled);
    PRINT_IF("ref dirty warm", collect_ref_dirty_warm_count(), Enabled);
    PRINT_IF("ref dirty hot", collect_ref_dirty_hot_count(), Enabled);
    PRINT_IF("resident prof checks",
             collect_resident_profile_reference_checks(), Enabled);
    PRINT_IF("resident prof samples", collect_resident_profile_samples(),
             Enabled);
    PRINT_IF("resident prof sample cycles",
             collect_resident_profile_sample_cycles(), Enabled);
    PRINT_IF("resident planner collect cycles",
             collect_resident_profile_planner_collection_cycles(), Enabled);
    PRINT_IF("resident planner candidate cycles",
             collect_resident_profile_planner_candidate_cycles(), Enabled);
    PRINT_IF("resident planner publish cycles",
             collect_resident_profile_planner_publish_cycles(), Enabled);
    PRINT_IF("resident promotion attempts",
             collect_resident_profile_promotion_attempts(), Enabled);
    PRINT_IF("resident promotion successes",
             collect_resident_profile_promotion_successes(), Enabled);
    PRINT_IF("resident promotion cycles",
             collect_resident_profile_promotion_cycles(), Enabled);
    PRINT_IF("resident demotion attempts",
             collect_resident_profile_demotion_attempts(), Enabled);
    PRINT_IF("resident demotion successes",
             collect_resident_profile_demotion_successes(), Enabled);
    PRINT_IF("resident demotion cycles",
             collect_resident_profile_demotion_cycles(), Enabled);
    PRINT_IF("resident region place attempts",
             collect_resident_profile_region_placement_attempts(), Enabled);
    PRINT_IF("resident region place cycles",
             collect_resident_profile_region_placement_cycles(), Enabled);
    if constexpr (Enabled) {
        print_heat_split();
    }
    PRINT_IF("evac master cycles", collect_evacuator_master_cycles(),
             enabled::Evacuation);
    PRINT_IF("stw mutator cycles", collect_stw_mutator_cycles(), Enabled);
    PRINT_IF("flip scope cycles", collect_flip_scope_cycles(), Enabled);
    PRINT_IF("flip scope yield cycles", collect_flip_scope_yield_cycles(), Enabled);
    PRINT_IF("gc cycles", collect_gc_cycles(), enabled::Evacuation);
    PRINT_IF("gc count", collect_gc_count(), enabled::Evacuation);

    PRINT_IF("evac mark ph cycles", collect_evac_mark_phase_cycles(), enabled::Evacuation);
    PRINT_IF("evac evict ph cycles", collect_evac_evict_phase_cycles(), enabled::Evacuation);
    PRINT_IF("evac gc ph cycles", collect_evac_gc_phase_cycles(), enabled::Evacuation);
    PRINT_IF("evac flip ph cycles", collect_evac_flip_scope_cycles(), enabled::Evacuation);
    PRINT_IF("evac flush cycles", collect_evac_flush_cycles(), enabled::Evacuation);
    PRINT_IF("evac flush count", collect_evac_flush_count(), enabled::Evacuation);
    PRINT_IF("evac flush full", collect_evac_flush_full_count(), enabled::Evacuation);
    PRINT_IF("evac flush partial", collect_evac_flush_partial_count(),
             enabled::Evacuation);
    PRINT_IF("evac flush wrs", collect_evac_flush_wrs(), enabled::Evacuation);
    PRINT_IF("evac streaming post blocked cycles",
             collect_evac_streaming_post_blocked_cycles(), enabled::Evacuation);
    PRINT_IF("evac flip wait old loops", collect_evac_flip_wait_old_loops(),
             enabled::Evacuation);
    PRINT_IF("evac flip wait old max", collect_evac_flip_wait_old_max(),
             enabled::Evacuation);
    PRINT_IF("evac flip wait old cy", collect_evac_flip_wait_old_cycles(),
             enabled::Evacuation);
    PRINT_IF("evac flip blk old", collect_evac_flip_wait_old_blocked_flips(),
             enabled::Evacuation);
    PRINT_IF("evac flip old entry", collect_evac_flip_wait_old_entry_sum(),
             enabled::Evacuation);
    PRINT_IF("evac flip old obs", collect_evac_flip_wait_old_observed_sum(),
             enabled::Evacuation);
    PRINT_IF("evac evict rounds", collect_evac_evict_rounds(),
             enabled::Evacuation);
    PRINT_IF("evac evict active rounds", collect_evac_evict_active_rounds(),
             enabled::Evacuation);
    PRINT_IF("evac evict empty rounds", collect_evac_evict_empty_rounds(),
             enabled::Evacuation);
    PRINT_IF("evac evict active cycles", collect_evac_evict_active_cycles(),
             enabled::Evacuation);
    PRINT_IF("evac evict empty cycles", collect_evac_evict_empty_cycles(),
             enabled::Evacuation);
    PRINT_IF("evac evict active wrs", collect_evac_evict_active_wrs(),
             enabled::Evacuation);
    if constexpr (Enabled) {
        print_ratio_line("evict dirty/clean", collect_clean_evict_bytes(),
                         collect_dirty_evict_bytes(), "clean", "dirty");
        print_ratio_line("ref mix cold", collect_ref_read_cold_count(),
                         collect_ref_write_cold_count(), "read", "write");
        print_ratio_line("ref mix warm", collect_ref_read_warm_count(),
                         collect_ref_write_warm_count(), "read", "write");
        print_ratio_line("ref mix hot", collect_ref_read_hot_count(),
                         collect_ref_write_hot_count(), "read", "write");
        print_ratio_line("ref dirtiness cold", collect_ref_clean_cold_count(),
                         collect_ref_dirty_cold_count(), "clean", "dirty");
        print_ratio_line("ref dirtiness warm", collect_ref_clean_warm_count(),
                         collect_ref_dirty_warm_count(), "clean", "dirty");
        print_ratio_line("ref dirtiness hot", collect_ref_clean_hot_count(),
                         collect_ref_dirty_hot_count(), "clean", "dirty");
    }
#if FARLIB_ENABLE_EVAC_VERBOSE_PROFILE
    PRINT_IF("evac cq wait cycles", collect_evac_check_cq_wait_cycles(), enabled::Evacuation);
    PRINT_IF("evac evict post cycles", collect_evac_evict_post_cycles(), enabled::Evacuation);
    PRINT_IF("evac drain cycles", collect_evac_drain_cycles(), enabled::Evacuation);
    PRINT_IF("evac overlap mark cycles", collect_evac_overlap_mark_cycles(), enabled::Evacuation);
    PRINT_IF("evac pipeline idle cycles", collect_evac_pipeline_idle_cycles(), enabled::Evacuation);
    PRINT_IF("evac pipeline rounds", collect_evac_pipeline_rounds(), enabled::Evacuation);
    PRINT_IF("evac streaming post blocked cycles", collect_evac_streaming_post_blocked_cycles(), enabled::Evacuation);
    PRINT_IF("evac streaming wait safe epoch cycles",
             collect_evac_streaming_wait_safe_epoch_cycles(),
             enabled::Evacuation);
    PRINT_IF("evac streaming safe epoch zero rounds",
             collect_evac_streaming_safe_epoch_zero_rounds(),
             enabled::Evacuation);
    PRINT_IF("evac streaming safe epoch zero tasks",
             collect_evac_streaming_safe_epoch_zero_tasks(),
             enabled::Evacuation);
    PRINT_IF("evac try evict cnt", collect_evac_try_evict_count(), enabled::Evacuation);
    PRINT_IF("evac touched-after-mark cnt",
             collect_evac_touched_after_mark_count(), enabled::Evacuation);
    PRINT_IF("evac flip wait new loops", collect_evac_flip_wait_new_loops(),
             enabled::Evacuation);
    PRINT_IF("evac flip wait new max", collect_evac_flip_wait_new_max(),
             enabled::Evacuation);
    PRINT_IF("evac flip wait new cy", collect_evac_flip_wait_new_cycles(),
             enabled::Evacuation);
    PRINT_IF("evac flip blk new", collect_evac_flip_wait_new_blocked_flips(),
             enabled::Evacuation);
    PRINT_IF("evac flip new entry", collect_evac_flip_wait_new_entry_sum(),
             enabled::Evacuation);
    PRINT_IF("evac flip new obs", collect_evac_flip_wait_new_observed_sum(),
             enabled::Evacuation);
    PRINT_IF("evac skip pinned", collect_evac_try_evict_pinned(),
             enabled::Evacuation);
    PRINT_IF("evac skip free/mismatch", collect_evac_try_evict_free_or_mismatch(),
             enabled::Evacuation);
    PRINT_IF("evac marked dirty", collect_evac_try_evict_marked_dirty(),
             enabled::Evacuation);
    PRINT_IF("evac marked clean", collect_evac_try_evict_marked_clean(),
             enabled::Evacuation);
    PRINT_IF("evac state local", collect_evac_try_evict_state_local(),
             enabled::Evacuation);
    PRINT_IF("evac state fetching", collect_evac_try_evict_state_fetching(),
             enabled::Evacuation);
    PRINT_IF("evac state remote", collect_evac_try_evict_state_remote(),
             enabled::Evacuation);
    PRINT_IF("evac state busy", collect_evac_try_evict_state_busy(),
             enabled::Evacuation);
    PRINT_IF("evac state other", collect_evac_try_evict_state_other(),
             enabled::Evacuation);
    PRINT_IF("evac mark regions", collect_evac_mark_regions_scanned(),
             enabled::Evacuation);
    PRINT_IF("evac mark reg output", collect_evac_mark_regions_with_output(),
             enabled::Evacuation);
    PRINT_IF("evac mark reg empty", collect_evac_mark_regions_without_output(),
             enabled::Evacuation);
    PRINT_IF("evac mark skip inuse", collect_evac_mark_regions_skipped_in_use(),
             enabled::Evacuation);
    PRINT_IF("evac mark skip stable", collect_evac_mark_regions_skipped_stable(),
             enabled::Evacuation);
    PRINT_IF("evac evict skip inuse", collect_evac_evict_regions_skipped_in_use(),
             enabled::Evacuation);
    PRINT_IF("evac evict to free", collect_evac_evict_regions_to_free(),
             enabled::Evacuation);
    PRINT_IF("evac evict to usable", collect_evac_evict_regions_to_usable(),
             enabled::Evacuation);
    PRINT_IF("evac evict to full", collect_evac_evict_regions_to_full(),
             enabled::Evacuation);
#if FARLIB_ENABLE_EVAC_FINE_PROFILE
    PRINT_IF("evac pop empty", collect_evac_evict_pop_empty(), enabled::Evacuation);
    PRINT_IF("evac pop head matched", collect_evac_evict_pop_head_matched(),
             enabled::Evacuation);
    PRINT_IF("evac pop success", collect_evac_evict_pop_success(), enabled::Evacuation);
#endif
    if constexpr (enabled::Evacuation) {
        int64_t wr_active_min = INT64_MAX, wr_active_max = 0;
        int64_t round_active_min = INT64_MAX, round_active_max = 0;
        int64_t wr_active_threads = 0, round_active_threads = 0;
        for (auto &it : profile_data_map) {
            auto *t = it.second;
            if (t->evac_evict_active_wrs > 0) {
                wr_active_threads++;
                if (t->evac_evict_active_wrs < wr_active_min) {
                    wr_active_min = t->evac_evict_active_wrs;
                }
                if (t->evac_evict_active_wrs > wr_active_max) {
                    wr_active_max = t->evac_evict_active_wrs;
                }
            }
            if (t->evac_evict_active_rounds > 0) {
                round_active_threads++;
                if (t->evac_evict_active_rounds < round_active_min) {
                    round_active_min = t->evac_evict_active_rounds;
                }
                if (t->evac_evict_active_rounds > round_active_max) {
                    round_active_max = t->evac_evict_active_rounds;
                }
            }
        }
        if (wr_active_threads == 0) {
            wr_active_min = 0;
        }
        if (round_active_threads == 0) {
            round_active_min = 0;
        }
        PRINT_IF("evac wr-active threads", wr_active_threads, enabled::Evacuation);
        PRINT_IF("evac wr-active min", wr_active_min, enabled::Evacuation);
        PRINT_IF("evac wr-active max", wr_active_max, enabled::Evacuation);
        PRINT_IF("evac round-active threads", round_active_threads,
                 enabled::Evacuation);
        PRINT_IF("evac round-active min", round_active_min, enabled::Evacuation);
        PRINT_IF("evac round-active max", round_active_max, enabled::Evacuation);
    }
#endif

    PRINT_IF("forkjoin cycles", collect_fork_join_cycles(), Enabled);
    PRINT_IF("check cq cycles", collect_check_cq_cycles(), Enabled);
    PRINT_IF("check cq count", collect_check_cq_count(), Enabled);
    PRINT_IF("mark count", collect_mark_count(), Enabled);
    PRINT_IF("mark payload bytes", collect_mark_payload_bytes(), Enabled);
    PRINT_IF("not mark count", collect_not_mark_count(), Enabled);
    PRINT_IF("on miss cycles", collect_on_miss_cycles(),
             enabled::OnMissSchedule);
    PRINT_IF("prefetch cycles", collect_prefetch_cycles(),
             enabled::OnMissSchedule);
    PRINT_IF("left for use1 cycles", collect_left_for_use1_cycles(), Enabled);
    PRINT_IF("left for use2 cycles", collect_left_for_use2_cycles(), Enabled);
    PRINT_IF("left for use3 cycles", collect_left_for_use3_cycles(), Enabled);
    PRINT_IF("left for use4 cycles", collect_left_for_use4_cycles(), Enabled);
    PRINT_IF("obj modifed count", collect_obj_modifed_cnt(), Enabled);
    PRINT_IF("obj unmodifed count", collect_obj_unmodifed_cnt(), Enabled);

    PRINT_IF("rdma read posts", collect_rdma_read_post_count(), Enabled);
    PRINT_IF("rdma read bytes", collect_rdma_read_post_bytes(), Enabled);
    print_rdma_read_size_histogram();
    PRINT_IF("rdma write posts", collect_rdma_write_post_count(), Enabled);
    PRINT_IF("rdma write bytes", collect_rdma_write_post_bytes(), Enabled);

    PRINT_IF("excl ralloc cnt", collect_excl_remote_alloc_count(), Enabled);
    PRINT_IF("excl ralloc bytes", collect_excl_remote_alloc_bytes(), Enabled);
    PRINT_IF("excl ralloc cycles", collect_excl_remote_alloc_cycles(), Enabled);
    PRINT_IF("excl rfree/fetch cnt", collect_excl_remote_free_on_fetch_count(), Enabled);
    PRINT_IF("excl rfree/fetch bytes", collect_excl_remote_free_on_fetch_bytes(), Enabled);
    PRINT_IF("excl rfree/fetch cycles", collect_excl_remote_free_on_fetch_cycles(), Enabled);
    PRINT_IF("excl rfree/fetch slow cnt", collect_excl_remote_free_on_fetch_slow_count(), Enabled);
    PRINT_IF("excl rfree/fetch slow bytes", collect_excl_remote_free_on_fetch_slow_bytes(), Enabled);
    PRINT_IF("excl rfree/fetch slow cycles", collect_excl_remote_free_on_fetch_slow_cycles(), Enabled);
    PRINT_IF("excl rfree/int cnt", collect_excl_interrupted_evict_free_count(), Enabled);
    PRINT_IF("excl rfree/int bytes", collect_excl_interrupted_evict_free_bytes(), Enabled);
    PRINT_IF("excl movelock spins", collect_excl_move_lock_spin_count(), Enabled);
    PRINT_IF("excl movelock cycles", collect_excl_move_lock_spin_cycles(), Enabled);
    PRINT_IF("backup admit attempts", collect_remote_backup_admission_attempts(),
             Enabled);
    PRINT_IF("backup retained cnt", collect_remote_backup_retained_count(),
             Enabled);
    PRINT_IF("backup retained bytes", collect_remote_backup_retained_bytes(),
             Enabled);
    PRINT_IF("backup rejected cnt", collect_remote_backup_rejected_count(),
             Enabled);
    PRINT_IF("backup rejected bytes", collect_remote_backup_rejected_bytes(),
             Enabled);
    PRINT_IF("backup reused cnt", collect_remote_backup_reused_count(), Enabled);
    PRINT_IF("backup reused bytes", collect_remote_backup_reused_bytes(),
             Enabled);
    PRINT_IF("backup invalidated cnt", collect_remote_backup_invalidated_count(),
             Enabled);
    PRINT_IF("backup invalidated bytes",
             collect_remote_backup_invalidated_bytes(), Enabled);
    PRINT_IF("remote dealloc cnt", collect_remote_dealloc_count(), Enabled);
    PRINT_IF("remote dealloc lockcy", collect_remote_dealloc_lock_cycles(), Enabled);
    PRINT_IF("remote dealloc lockhold", collect_remote_dealloc_lock_hold_cycles(), Enabled);
    PRINT_IF("remote dealloc bmcy", collect_remote_dealloc_bitmap_cycles(), Enabled);
    PRINT_IF("remote dealloc listcy", collect_remote_dealloc_list_cycles(), Enabled);
    PRINT_IF("remote dealloc totcy", collect_remote_dealloc_total_cycles(), Enabled);
    print_full_population_frequency_history();
#undef PRINT_IF
}

void print_rdma_trace() {
    if constexpr (TraceRDMA) {
        static std::atomic_flag printing = false;
        if (printing.test_and_set()) return;
        for (auto &it : profile_data_map) {
            auto tid = it.first;
            auto data = it.second;
            std::cout << "rdma read requests for thread " << tid << std::endl;
            data->rdma_read_reqs.for_each(
                [](std::tuple<uint64_t, uint64_t, uint64_t> p) {
                    std::cout << std::setw(32) << std::get<0>(p)
                              << std::setw(32) << std::get<1>(p)
                              << std::setw(32) << std::get<2>(p) << std::endl;
                });
            std::cout << std::endl;
            std::cout << "rdma read completions for thread " << tid
                      << std::endl;
            data->rdma_read_wcs.for_each([](std::pair<uint64_t, uint64_t> p) {
                std::cout << std::setw(24) << p.first << std::setw(24)
                          << p.second << std::endl;
            });
            std::cout << std::endl;
        }
        printing.clear();
    }
}

void print_alloc_trace() {
    if constexpr (TraceAlloc) {
        static std::atomic_flag printing = false;
        if (printing.test_and_set()) return;
        for (auto &it : profile_data_map) {
            auto tid = it.first;
            auto data = it.second;
            std::cout << "allocation trace for thread " << tid << std::endl;
            data->alloc_trace.for_each([](const AllocTrace &p) {
                std::cout << std::setw(24) << p.op << std::setw(24) << p.tsc
                          << std::setw(24) << p.addr << std::setw(24) << p.bin
                          << std::endl;
            });
            std::cout << std::endl;
        }
        printing.clear();
    }
}

void print_mem_usage_trace() {
    if constexpr (TraceMemoryUsage) {
        static std::atomic_flag printing = false;
        if (printing.test_and_set()) return;
        for (auto &it : profile_data_map) {
            auto tid = it.first;
            auto data = it.second;
            std::cout << "memory usage trace for thread " << tid << std::endl;
            data->mem_usage_trace.for_each([](const MemoryUsageTrace &p) {
                std::cout << std::setw(24) << p.previous_free_size
                          << std::setw(24) << p.free_size_modification
                          << std::endl;
            });
            std::cout << std::endl;
        }
        printing.clear();
    }
}

}  // namespace profile

namespace async {
#ifdef PROFILE_STREAM_RUNNER_SCHEDULE
std::atomic_int64_t StreamRunnerProfiler::global_total_cycles = 0;
std::atomic_int64_t StreamRunnerProfiler::global_app_cycles = 0;
std::atomic_int64_t StreamRunnerProfiler::global_sched_cycles = 0;
std::atomic_int64_t StreamRunnerProfiler::global_poll_cq_cycles = 0;
StreamRunnerProfiler::Warn StreamRunnerProfiler::warn;
#endif
}  // namespace async

}  // namespace FarLib
