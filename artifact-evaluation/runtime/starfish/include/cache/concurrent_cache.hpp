#pragma once
#include "design1/backup_usage_shards.hpp"
#include "design1/batched_backup_budget.hpp"
#include <x86intrin.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <boost/lockfree/queue.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cache/entry.hpp"
#include "design2/simple_dirty_observer.hpp"
#include "cache/base/handler.hpp"
#include "cache/base/object.hpp"
#include "cache/base/scope.hpp"
#include "design1/placement_policy.hpp"
#include "design1/runtime_types.hpp"
#include "cache/core/profile/profile_types.hpp"
#include "rdma/client.hpp"
#include "rdma/sponge_rpc.hpp"
#include "region_based_allocator.hpp"
#include "cache/alloc/remote_allocator.hpp"
#include "cache/alloc/ec_batch_write.hpp"
#include "recovery/ec_read_recovery.hpp"
#include "recovery/ec_read_context.hpp"
#include "recovery/ec_recovery_scratch.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/fork_join.hpp"
#include "utils/signal.hpp"
#include "utils/stats.hpp"
#include "utils/uthreads.hpp"
#include "cache/core/common/sharded_scope_counters.hpp"
#include "utils/wait_trace.hpp"
#include "utils/inclusive_reclaim_diag.hpp"
#include "utils/scope_diag.hpp"
#include "utils/read_supply_timeline.hpp"
#include "utils/request_interval_diag.hpp"

// Compile the complete client and its static libraries with the same value.
// This experiment changes cache-object layout only, not backup admission.
#ifndef FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION
#define FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION 0
#endif
#if FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION != 0 && \
    FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION != 1
#error "FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION must be 0 or 1"
#endif

// Always log critical errors to FD 2 for atomicity/visibility
inline void exclusive_error_log(const std::string &msg) {
    auto res = write(2, msg.c_str(), msg.size());
    (void)res;
}
#define EXCLUSIVE_ERROR(msg) exclusive_error_log(msg)

namespace FarLib {
namespace cache {
inline constexpr int kSpinYieldThreshold = 10000;

// ft_method=ec_batch: bound of the ref_cnt wait in deallocate() (see the
// bounded escape there).  A group's six CQEs are reaped within microseconds of
// the post and every spin of that wait also polls this client's CQs (every 8th
// spin sweeps all CQs), so 1e7 spins is far beyond any real completion latency
// and the healthy path never reaches the bound.  A stuck teardown measured in
// the field sat at ~3e7 full CQ sweeps with a harvest of 0, well past this.
inline constexpr uint64_t kEcBatchDeallocStuckSpinLimit = 10000000ull;

inline bool env_flag_or_default(const char *name, bool default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    return std::strtol(value, nullptr, 0) != 0;
}

inline bool optimized_full_prime_mark_enabled() {
    static const bool enabled = env_flag_or_default(
        "FARLIB_OPT_FULL_PRIME_MARK",
        ::FarLib::get_config().optimized_evacuator);
    return enabled;
}

inline bool optimized_serialize_mark_evict_enabled() {
    static const bool enabled = env_flag_or_default(
        "FARLIB_OPT_SERIALIZE_MARK_EVICT",
        ::FarLib::get_config().optimized_evacuator);
    return enabled;
}

inline bool optimized_ready_queue_enabled() {
    static const bool enabled = env_flag_or_default(
        "FARLIB_OPT_READY_QUEUE",
        ::FarLib::get_config().optimized_evacuator);
    return enabled;
}

inline bool optimized_legacy_exclusive_pipeline_enabled() {
    static const bool enabled = env_flag_or_default(
        "FARLIB_OPT_LEGACY_EXCLUSIVE_PIPELINE", false);
    return enabled;
}

inline bool optimized_ready_full_workers_enabled() {
    static const bool enabled = env_flag_or_default(
        "FARLIB_OPT_READY_FULL_WORKERS",
        ::FarLib::get_config().optimized_evacuator);
    return enabled;
}

inline size_t optimized_ready_mark_task_limit() {
    const char *value = std::getenv("FARLIB_OPT_READY_MARK_TASK_LIMIT");
    return value ? std::strtoull(value, nullptr, 0) : 0;
}

inline bool resident_profile_stop_after_first_commit() {
    return env_flag_or_default(
        "FARLIB_RESIDENT_PROFILE_STOP_AFTER_FIRST_COMMIT",
        ::FarLib::get_config().resident_profile_stop_after_first_commit);
}

inline bool resident_profile_phase_triggered_enabled() {
    return env_flag_or_default(
        "FARLIB_RESIDENT_PROFILE_PHASE_TRIGGERED", false);
}

inline bool region_fetch_diagnostics_enabled() {
    return env_flag_or_default("FARLIB_REGION_FETCH_DIAGNOSTICS", true);
}

inline bool profiled_backup_nonblocking_on_full_enabled() {
    return env_flag_or_default(
        "FARLIB_PROFILED_BACKUP_NONBLOCKING_ON_FULL", true);
}

inline bool backup_budget_diagnostics_enabled() {
    return env_flag_or_default("FARLIB_BACKUP_BUDGET_DIAGNOSTICS", false);
}

inline bool alloc_scope_checkpoint_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_ALLOC_SCOPE_CHECKPOINT");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

class ConcurrentArrayCache {
    friend class DereferenceScope;
    friend struct detail::FullPopulationFrequencyTracker;

public:
    enum MutatorState { OutOfScope, InScopeV0, InScopeV1, MutatorStateCount };

    struct alignas(64) {
        std::atomic<MutatorState> global_state = InScopeV0;
        std::atomic_size_t count[MutatorStateCount];
    } mutator_states;

    // Experimental toggle, fixed for this Cache's lifetime. Legacy OFF path
    // remains available for a same-binary performance control.
    const bool scope_counter_shards_enabled =
        env_flag_or_default("FARLIB_SCOPE_COUNTER_SHARDS", false);
    ShardedScopeCounters scope_counters;

    // Epoch-based release mechanism for streaming pipeline
    std::atomic<uint32_t> produce_epoch{0};
    std::atomic<uint32_t> safe_epoch{0};

    template <typename T>
    friend class UniqueFarPtr;

    ::FarLib::allocator::PendingLocalAllocation allocate_local(
        size_t size, far_obj_t obj, DereferenceScope &scope,
        ::FarLib::allocator::RegionPlacement requested_placement =
            ::FarLib::allocator::RegionPlacement::Unclassified,
        uint32_t requested_group_id = 0,
        uint32_t requested_behavior_group_id = 0) {
        scope_diag::Guard sd_alloc_guard(fibre_self(), scope_diag::ALLOC);
        auto *wc_sample = wc_object_diag::find_object(obj.obj_id);
        const uint64_t wc_alloc_begin = wc_sample ? wc_object_diag::stamp() : 0;
        if (alloc_scope_checkpoint_enabled()) {
            update_scope(scope);
        }
        const size_t alloc_block_size =
            size + sizeof(::FarLib::allocator::BlockHead);
        const size_t alloc_wsize =
            ::FarLib::allocator::wsize_from_size(alloc_block_size);
        const size_t alloc_bin =
            ::FarLib::allocator::bin_from_wsize(alloc_wsize);
        ::FarLib::allocator::BlockHead *block =
            ::FarLib::allocator::thread_local_allocate(
                size, obj, &scope, requested_placement,
                requested_group_id, requested_behavior_group_id,
                ::FarLib::allocator::six_group::enabled() ||
                    ::FarLib::simple_region_budget::six_enabled());
        if (block == nullptr) [[unlikely]] {
            // can not allocate, evict
#ifdef ASSERT_ALL_LOCAL
            ERROR("should not evict in allocate local");
#endif
            if (inclusive_reclaim_diag::enabled()) {
                reclaim_diag_last_alloc_bin.store(alloc_bin,
                                                  std::memory_order_relaxed);
            }
            log_cache_progress("alloc_wait_begin", alloc_bin, size);
            if (wait_trace::nq_supply_only.load(std::memory_order_relaxed)) {
                log_cache_progress("alloc_wait_placement", alloc_bin,
                                   static_cast<uint64_t>(requested_placement),
                                   requested_group_id);
            }
            // Time the whole allocation retry interval, including scheduler
            // waits. The summed counter is fibre elapsed-time, not CPU usage
            // or the wall-time union of simultaneous allocation stalls.
            auto stw_start_cycles = get_cycles();
            auto allocation_wait_diag_event =
                profile::begin_allocation_wait_event(stw_start_cycles);
            bool suspended = profile::suspend_work();
            size_t retry_count = 0;
            scope.begin_eviction();
            do {
                on_demand_invoke_eviction(alloc_bin);
                block = ::FarLib::allocator::thread_local_allocate(
                    size, obj, nullptr, requested_placement,
                    requested_group_id, requested_behavior_group_id,
                    ::FarLib::allocator::six_group::enabled() ||
                        ::FarLib::simple_region_budget::six_enabled());
                retry_count++;
                log_cache_progress(block == nullptr ? "alloc_retry_fail"
                                                    : "alloc_retry_succ",
                                   alloc_bin, retry_count);
                if (retry_count > 1024) {
                    ERROR("can not allocate!");
                }
                // nullptr when local memory buffer is full
            } while (block == nullptr);
            scope.end_eviction();
            profile::resume_work(suspended);
            auto stw_end_cycles = get_cycles();
            profile::end_allocation_wait_event(allocation_wait_diag_event,
                                               stw_end_cycles);
            profile::count_stw_mutator_cycles(stw_end_cycles -
                                               stw_start_cycles);
            auto *allocated_region =
                ::FarLib::allocator::block_to_region(block);
            if (inclusive_reclaim_diag::enabled() &&
                inclusive_reclaim_diag::sample_region(allocated_region)) {
                wait_trace::emit(
                    "reclaim_alloc_take", alloc_bin, retry_count,
                    allocated_region->placement_epoch.load(
                        std::memory_order_relaxed),
                    reinterpret_cast<uint64_t>(allocated_region));
            }
            log_cache_progress("alloc_wait_end", alloc_bin, retry_count);
            if (wait_trace::nq_supply_only.load(std::memory_order_relaxed)) {
                log_cache_progress("alloc_take_placement", alloc_bin,
                                   static_cast<uint64_t>(allocated_region->load_placement()),
                                   reinterpret_cast<uint64_t>(allocated_region));
            }
        }
        assert(obj == block->obj_meta_data);
        wc_object_diag::allocation_done(wc_sample, wc_alloc_begin);
        alloc_reclaim_rate_diag::record_allocation(
            ::FarLib::allocator::get_bin_size(alloc_bin));
        return {block, ::FarLib::allocator::six_group::enabled() ||
                           ::FarLib::simple_region_budget::six_enabled()};
    }

private:
    using FrequencyProfileMarkResult = detail::FrequencyProfileMarkResult;
    using FrequencyMarkPassContext = detail::FrequencyMarkPassContext;
    using FullPopulationFrequencyTracker = detail::FullPopulationFrequencyTracker;

    using BackupBehaviorGroupProfile =
        design1::BackupBehaviorGroupProfile;
    using ResidentPlacementGroupProfile =
        design1::ResidentPlacementGroupProfile;
    using ActiveResidentPlacementGroup =
        design1::ActiveResidentPlacementGroup;
    static constexpr uint64_t kResidentMigrationGateClosed = uint64_t{1}
                                                               << 63;
    static constexpr uint64_t kResidentMigrationInflightMask =
        kResidentMigrationGateClosed - 1;
    struct ResidentProfileMigrationInflightGuard {
        std::atomic<uint64_t> &gate;
        bool entered{false};
        explicit ResidentProfileMigrationInflightGuard(
            std::atomic<uint64_t> &gate)
            : gate(gate) {
            uint64_t state = gate.load(std::memory_order_acquire);
            while ((state & kResidentMigrationGateClosed) == 0) {
                ASSERT((state & kResidentMigrationInflightMask) !=
                       kResidentMigrationInflightMask);
                if (gate.compare_exchange_weak(
                        state, state + 1, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    entered = true;
                    break;
                }
            }
        }
        ~ResidentProfileMigrationInflightGuard() {
            if (entered) {
                const uint64_t previous =
                    gate.fetch_sub(1, std::memory_order_acq_rel);
                ASSERT((previous & kResidentMigrationInflightMask) != 0);
            }
        }
        explicit operator bool() const { return entered; }
    };

    static constexpr size_t kBackupBehaviorGroupCount =
        ::FarLib::allocator::RegionBinCount;
    static constexpr size_t kMaxLogicalObjectCount = 4096;
    // 256K groups cover 64 GiB at the default 256 KiB group size.
    static constexpr size_t kMaxResidentPlacementGroupCount = 256 * 1024;

    RemoteAllocator remote_allocator;
    std::array<BackupBehaviorGroupProfile, kBackupBehaviorGroupCount>
        backup_behavior_groups{};
    std::array<BackupBehaviorGroupProfile, kMaxLogicalObjectCount>
        logical_object_profiles{};
    std::unique_ptr<ResidentPlacementGroupProfile[]>
        resident_placement_groups;
    std::unordered_map<uint64_t, ActiveResidentPlacementGroup>
        active_resident_placement_groups;
    std::mutex resident_placement_group_mutex;
    std::atomic<uint32_t> next_resident_placement_group_id{1};
    std::atomic<uint32_t> next_logical_owner_id{1};
    alignas(FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION
                ? 64 : alignof(std::atomic<uint64_t>))
        std::atomic<uint64_t> retained_backup_bytes{0};
    alignas(FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION
                ? 64 : alignof(std::atomic<uint64_t>))
        std::atomic<uint64_t> peak_retained_backup_bytes{0};
    alignas(FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION
                ? 64 : alignof(uint64_t))
        uint64_t retained_backup_budget_bytes{0};
    // Keep the next mutable field off the read-mostly budget's cache line.
    alignas(FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION
                ? 64 : alignof(std::atomic<uint64_t>))
        std::atomic<uint64_t> resident_local_bytes{0};
    std::atomic<uint64_t> peak_resident_local_bytes{0};
    uint64_t resident_local_budget_bytes{0};
    bool segmented_backup_mode{false};
    bool profiled_backup_mode{false};
    bool object_profiled_backup_mode{false};
    bool backup_covers_transient_window{false};
    bool logical_object_profile_enabled_flag{false};
    size_t logical_object_profile_sample_shift{0};
    size_t backup_profile_window_events{4096};
    size_t backup_profile_min_score_pct{50};
    bool resident_profile_planner_enabled_flag{false};
    // Latched once when the cache is constructed.  Completion handling and
    // endpoint recovery may run concurrently with teardown; per-entry
    // Resident migration must not consult a mutable/corrupted global config
    // copy to decide whether EC owns the placement lifecycle.
    bool ec_batch_mode_{false};
    design1::PlacementPolicy region_fetch_placement_policy{
        design1::PlacementPolicy::Hotness};
    bool resident_profile_require_work_phase_flag{false};
    bool resident_profile_apply_plan_flag{true};
    bool resident_profile_manual_trigger_flag{false};
    size_t resident_profile_warmup_ms{30000};
    size_t resident_profile_sample_shift{6};
    size_t resident_profile_group_bytes{256 * 1024};
    size_t resident_profile_interval_ms{1000};
    size_t resident_profile_initial_sample_windows{10};
    size_t resident_profile_min_benefit_refs_per_replacement_object{4};
    size_t resident_profile_write_weight{2};
    size_t resident_profile_ema_decay_shift{1};
    size_t resident_profile_migration_budget_pct{5};
    size_t resident_profile_migration_budget_objects{0};
    std::atomic_bool resident_profile_active{false};
    std::atomic_bool resident_profile_plan_ready{false};
    // Learned object/group metadata is sufficient to route future fetches
    // even when the physical Region ranking requires no R<->S exchange.
    std::atomic_bool resident_region_fetch_plan_ready{false};
    uint64_t resident_region_fetch_next_plan_check{1};
    size_t resident_region_fetch_window_multiplier{1};
    uint8_t resident_region_fetch_stable_plan_streak{0};
    std::atomic_bool resident_region_fetch_reclassification_enabled{true};
    uint64_t resident_region_fetch_last_target_deficit_bytes{0};
    uint64_t resident_region_fetch_last_refill_successes{0};
    uint8_t resident_region_fetch_low_efficiency_streak{0};
    uint64_t resident_region_fetch_disabled_deficit_bytes{0};
    uint8_t resident_region_fetch_retry_cooldown_windows{0};
    inline static thread_local bool resident_profile_prefetch_active = false;
    std::thread resident_profile_thread;
    std::mutex resident_profile_mutex;
    std::condition_variable resident_profile_cond;
    std::atomic<uint64_t> resident_profile_plan_check_count{0};
    std::atomic<uint64_t> resident_profile_plan_count{0};
    std::atomic<uint64_t> resident_profile_published_plan_count{0};
    std::atomic<uint64_t> resident_profile_convergence_gate_count{0};
    std::atomic<uint64_t>
        resident_profile_convergence_gate_pending_bytes_total{0};
    std::atomic<uint64_t>
        resident_profile_convergence_gate_pending_bytes_last{0};
    std::atomic<uint64_t> resident_profile_migration_gate{
        kResidentMigrationGateClosed};
    static constexpr uint64_t kResidentCandidateStablePct = 1;
    static constexpr uint8_t kResidentDirectionStableComparisons = 2;
    static constexpr uint8_t kResidentPromotionStallChecks = 2;
    std::vector<uint64_t> resident_profile_committed_targets;
    std::vector<uint64_t> resident_profile_previous_candidates;
    std::vector<uint64_t> resident_profile_previous_current;
    std::vector<int8_t> resident_profile_previous_candidate_directions;
    std::vector<uint8_t> resident_profile_candidate_direction_streaks;
    std::vector<uint8_t> resident_profile_promotion_stall_checks;
    std::vector<uint8_t> resident_profile_group_planner_initialized;
    std::vector<uint8_t> resident_profile_candidate_seen;
    std::atomic<uint64_t> resident_profile_candidate_check_count{0};
    std::atomic<uint64_t> resident_profile_candidate_stable_group_checks{0};
    std::atomic<uint64_t> resident_profile_candidate_rejected_count{0};
    std::atomic<uint64_t> resident_profile_benefit_rejected_count{0};
    std::atomic<uint64_t> resident_profile_candidate_replacement_bytes_last{0};
    std::atomic<uint64_t> resident_profile_candidate_replacement_bytes_total{0};
    std::atomic<uint64_t> resident_profile_candidate_active_delta_bytes_last{0};
    std::atomic<uint64_t> resident_profile_candidate_target_overlap_bytes_last{0};
    std::atomic<uint64_t> resident_profile_stalled_promotion_cancelled_bytes{0};
    std::atomic<uint64_t> resident_profile_stalled_promotion_cancelled_groups{0};
    std::atomic<uint64_t> resident_profile_prevented_reversal_count{0};
    std::atomic<uint64_t> resident_profile_publish_while_unsafe_count{0};
    std::atomic<uint64_t> resident_profile_collection_us_total{0};
    std::atomic<uint64_t> resident_profile_collection_us_max{0};
    std::atomic<uint64_t> resident_profile_candidate_build_us_total{0};
    std::atomic<uint64_t> resident_profile_candidate_build_us_max{0};
    std::atomic<uint64_t> resident_profile_publish_us_total{0};
    std::atomic<uint64_t> resident_profile_publish_us_max{0};
    std::atomic<uint64_t> resident_profile_active_plan_id{0};
    std::atomic<uint64_t> resident_profile_active_plan_publish_check{0};
    std::atomic<uint64_t> resident_profile_active_plan_publish_time_us{0};
    std::atomic<uint64_t> resident_profile_active_plan_scheduled_promotion_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_scheduled_demotion_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_promotion_started_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_demotion_started_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_promotion_committed_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_demotion_committed_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_promotion_rolled_back_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_demotion_rolled_back_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_promotion_cancelled_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_demotion_cancelled_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_promotion_carried_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_demotion_carried_bytes{0};
    std::atomic<uint64_t> resident_profile_active_plan_first_start_check{0};
    std::atomic<uint64_t> resident_profile_active_plan_first_start_time_us{0};
    std::atomic<uint64_t> resident_profile_active_plan_complete_check{0};
    std::atomic<uint64_t> resident_profile_active_plan_complete_time_us{0};
    std::atomic<uint64_t> resident_profile_promotion_started_bytes_total{0};
    std::atomic<uint64_t> resident_profile_demotion_started_bytes_total{0};
    std::atomic<uint64_t> resident_profile_promotion_committed_bytes_total{0};
    std::atomic<uint64_t> resident_profile_demotion_committed_bytes_total{0};
    std::atomic<uint64_t> resident_profile_promotion_rolled_back_bytes_total{0};
    std::atomic<uint64_t> resident_profile_demotion_rolled_back_bytes_total{0};
    std::atomic<uint64_t> resident_profile_promotion_cancelled_bytes_total{0};
    std::atomic<uint64_t> resident_profile_demotion_cancelled_bytes_total{0};
    std::atomic<uint64_t> resident_profile_promotion_carried_bytes_total{0};
    std::atomic<uint64_t> resident_profile_demotion_carried_bytes_total{0};
    std::atomic<uint64_t> resident_profile_promoted_bytes{0};
    std::atomic<uint64_t> resident_profile_demoted_bytes{0};
    std::atomic<uint64_t> resident_profile_released_backup_bytes{0};
    std::atomic<uint64_t> resident_transition_backup_deferred_count{0};
    std::atomic<uint64_t> resident_region_pending_mark_bypasses{0};
    std::atomic<uint64_t> resident_region_pending_unmarks{0};
    std::atomic<uint64_t> resident_region_pending_evicting_seen{0};
    std::atomic<uint64_t> resident_region_last_distribution_plan_id{0};
    std::atomic<uint64_t> segmented_backup_wait_count{0};
    std::atomic<uint64_t> segmented_backup_wait_cycles{0};
    std::atomic<uint64_t> profiled_backup_budget_full_reject_count{0};
    std::atomic<uint64_t> profiled_backup_budget_full_reject_bytes{0};
    std::atomic<uint64_t> quiesce_recovered_stale_local_ref_count{0};
    bool all_nonresident_backup_mode{false};
    std::atomic<uint64_t> nonresident_interrupted_keep_count{0};
    std::atomic<uint64_t> nonresident_interrupted_keep_bytes{0};
    std::atomic<uint64_t> region_fetch_requested_resident_count{0};
    std::atomic<uint64_t> region_fetch_requested_streaming_count{0};
    std::atomic<uint64_t> region_fetch_actual_resident_count{0};
    std::atomic<uint64_t> region_fetch_actual_streaming_count{0};
    std::atomic<uint64_t>
        region_fetch_resident_to_streaming_fallback_count{0};
    std::atomic<uint64_t>
        region_fetch_resident_to_streaming_fallback_bytes{0};
    std::atomic<uint64_t>
        region_fetch_window_resident_fallback_count
            [::FarLib::allocator::RegionBinCount]{};
    std::atomic<uint64_t>
        region_fetch_window_resident_fallback_bytes
            [::FarLib::allocator::RegionBinCount]{};
    std::atomic<uint64_t> region_fetch_reserve_refill_checks{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_attempts{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_successes{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_capacity_bytes{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_exchange_bytes{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_promoted_intent_bytes{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_cycles{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_no_streaming_candidate{0};
    std::atomic<uint64_t> region_fetch_reserve_refill_no_resident_cold{0};
    std::atomic_bool working;
    std::atomic_bool mutator_can_not_allocate;
    std::atomic<uint64_t> eviction_request_seq{0};
    std::atomic<uint32_t> hybrid_profile_mark_pass_counter{0};
    uint64_t inclusive_evict_scan_generation{1};
    uint64_t inclusive_gc_scan_generation{1};
    static constexpr size_t kAnyAllocBin =
        ::FarLib::allocator::RegionBinCount;
    uthread::Mutex eviction_mutex;
    uthread::Condition eviction_cond;
    uthread::Condition mutator_cond;
    std::atomic_uint32_t mutator_waiters{0};
    // libfibre Cluster destruction is incomplete; the opt-in background
    // cluster intentionally has process lifetime after all fibres are joined.
    uthread::Cluster *background_cluster_ = nullptr;
    std::unique_ptr<UThread> master_evacuation_thread;
    size_t evacuate_thread_cnt;
    std::atomic_flag flag;
    std::atomic_bool stw_active{false};
    std::atomic_bool read_supply_timeline_stop{false};
    std::thread read_supply_timeline_thread;
    std::atomic<size_t> reclaim_diag_last_alloc_bin{
        inclusive_reclaim_diag::kNoAllocBin};
    FullPopulationFrequencyTracker full_population_frequency_tracker;

    static std::unique_ptr<ConcurrentArrayCache> default_instance;

    uint64_t resident_profile_migrations_inflight() const {
        return resident_profile_migration_gate.load(
                   std::memory_order_acquire) &
               kResidentMigrationInflightMask;
    }

    void resident_profile_close_migration_gate() {
        resident_profile_migration_gate.fetch_or(
            kResidentMigrationGateClosed, std::memory_order_acq_rel);
        resident_profile_plan_ready.store(false, std::memory_order_release);
        while (resident_profile_migrations_inflight() != 0) {
            std::this_thread::yield();
        }
    }

    void resident_profile_open_migration_gate() {
        ASSERT(resident_profile_migrations_inflight() == 0);
        resident_profile_plan_ready.store(true, std::memory_order_release);
        resident_profile_migration_gate.fetch_and(
            kResidentMigrationInflightMask, std::memory_order_release);
    }

    bool alloc_bin_ready(size_t alloc_bin) {
        return alloc_bin >= kAnyAllocBin ||
               ::FarLib::allocator::global_heap.can_allocate_region_now(alloc_bin);
    }

    bool mutator_waiters_need_progress() {
        return mutator_waiters.load(std::memory_order_acquire) != 0;
    }

    bool mutators_can_resume() {
        return true;
    }

    bool use_legacy_baseline_ondemand() const {
        return !::FarLib::get_config().optimized_evacuator;
    }

    void log_cache_progress(const char *name, size_t bin = kAnyAllocBin,
                            uint64_t a = 0, uint64_t b = 0) {
        wait_trace::emit(name, bin, a, b, (uint64_t)uthread::get_tls());
    }

    void start_read_supply_timeline();
    void stop_read_supply_timeline();

    static bool full_population_frequency_stats_enabled();
    static size_t full_population_frequency_scan_period_ms();

    static size_t local_allocation_footprint(size_t size) {
        const size_t block_size =
            size + sizeof(::FarLib::allocator::BlockHead);
        const size_t wsize =
            ::FarLib::allocator::wsize_from_size(block_size);
        const size_t bin = ::FarLib::allocator::bin_from_wsize(wsize);
        return ::FarLib::allocator::get_bin_size(bin);
    }
    static uint32_t resident_allocation_bin(size_t size) {
        const size_t block_size =
            size + sizeof(::FarLib::allocator::BlockHead);
        const size_t wsize = ::FarLib::allocator::wsize_from_size(block_size);
        return static_cast<uint32_t>(::FarLib::allocator::bin_from_wsize(wsize));
    }
    bool automatic_resident_group_planner_enabled() const {
        return resident_profile_planner_enabled_flag &&
               !resident_profile_manual_trigger_flag;
    }
    bool resident_profile_work_phase_active() const {
        return !resident_profile_require_work_phase_flag ||
               profile::is_working();
    }
    bool region_resident_placement_enabled() const {
        return ::FarLib::allocator::global_heap.region_placement_is_enabled();
    }
    bool region_hotness_placement_enabled() const {
        return region_resident_placement_enabled() &&
               ::FarLib::get_config().enable_region_hotness_placement &&
               !::FarLib::get_config().region_placement_bind_groups;
    }
    bool region_fetch_hotness_placement_enabled() const {
        return region_hotness_placement_enabled() &&
               ::FarLib::get_config()
                   .enable_region_fetch_hotness_placement;
    }
    bool region_group_binding_enabled() const {
        return region_resident_placement_enabled() &&
               ::FarLib::get_config().region_placement_bind_groups;
    }
    ::FarLib::allocator::RegionPlacement record_region_fetch_request(
        ::FarLib::allocator::RegionPlacement placement);
    void record_region_fetch_actual(
        ::FarLib::allocator::RegionPlacement requested_placement,
        ::FarLib::allocator::RegionPlacement actual_placement,
        size_t size, ::FarLib::allocator::RegionHead *actual_region);
    ::FarLib::allocator::RegionPlacement region_fetch_placement(
        const FarObjectEntry &entry);
    struct RemoteFetchPlacement {
        ::FarLib::allocator::RegionPlacement requested_placement;
        uint32_t allocation_group_id;
        bool backup_policy_admitted;
        bool backup_reserved;
    };
    RemoteFetchPlacement prepare_remote_fetch_placement(
        FarObjectEntry &entry, size_t size, DereferenceScope &scope,
        bool allow_backup);
    ::FarLib::allocator::RegionPlacement finalize_remote_fetch_placement(
        FarObjectEntry &entry, size_t size, void *local_ptr,
        RemoteFetchPlacement &placement);
    uint32_t register_resident_placement_group(uint32_t logical_owner_id,
                                               size_t size,
                                               uint64_t footprint_bytes);
    void release_resident_placement_allocation(
        const FarObjectEntry &entry, uint64_t footprint_bytes);

    bool selective_backup_enabled() const {
        const auto &config = ::FarLib::get_config();
        return config.exclusive_cache && config.enable_selective_backup &&
               retained_backup_budget_bytes != 0;
    }

    bool local_placement_allows_remote_backup(
        const FarObjectEntry &entry) const {
        return !region_resident_placement_enabled() ||
               !entry.is_resident_local();
    }

    void handle_region_placement_transition_backup(
        FarObjectEntry &entry, size_t bytes,
        ::FarLib::allocator::RegionPlacement from,
        ::FarLib::allocator::RegionPlacement to,
        std::vector<uint64_t> *deferred_remote_frees = nullptr);

    bool segmented_backup_enabled() const {
        return selective_backup_enabled() && segmented_backup_mode;
    }

    bool profiled_backup_enabled() const {
        return segmented_backup_enabled() &&
               (profiled_backup_mode || object_profiled_backup_mode);
    }

    bool logical_object_profile_enabled() const {
        return logical_object_profile_enabled_flag;
    }

    bool size_profiled_backup_enabled() const {
        return segmented_backup_enabled() && profiled_backup_mode;
    }

    static size_t backup_behavior_group_for_size(size_t size) {
        const size_t block_size =
            size + sizeof(::FarLib::allocator::BlockHead);
        const size_t wsize =
            ::FarLib::allocator::wsize_from_size(block_size);
        const size_t group = ::FarLib::allocator::bin_from_wsize(wsize);
        ASSERT(group < kBackupBehaviorGroupCount);
        return group;
    }

    static uint32_t percent(uint64_t numerator, uint64_t denominator) {
        if (denominator == 0) {
            return 0;
        }
        return static_cast<uint32_t>(
            std::min<uint64_t>(100, numerator * 100 / denominator));
    }

    void publish_backup_profile(BackupBehaviorGroupProfile &group) {
        const uint64_t observed_evictions =
            group.window_clean_evictions.load(std::memory_order_relaxed) +
            group.window_dirty_evictions.load(std::memory_order_relaxed);
        if (observed_evictions < backup_profile_window_events ||
            group.publishing.test_and_set(std::memory_order_acquire)) {
            return;
        }

        const uint64_t ready_evictions =
            group.window_clean_evictions.load(std::memory_order_relaxed) +
            group.window_dirty_evictions.load(std::memory_order_relaxed);
        if (ready_evictions < backup_profile_window_events) {
            group.publishing.clear(std::memory_order_release);
            return;
        }

        const uint64_t clean =
            group.window_clean_evictions.exchange(0, std::memory_order_acq_rel);
        const uint64_t dirty =
            group.window_dirty_evictions.exchange(0, std::memory_order_acq_rel);
        group.window_fetches.exchange(0, std::memory_order_acq_rel);
        const uint64_t evictions = clean + dirty;
        const uint32_t clean_pct = percent(clean, evictions);
        // Fetches and evictions are different pipeline phases. Computing their
        // ratio from the same short window makes a streaming workload oscillate
        // between 0% and 100% refetch probability depending on phase alignment.
        // Use lifetime totals for refetch propensity and the latest window for
        // clean probability, which is the signal that must adapt to mutations.
        const uint64_t total_fetches =
            group.total_fetches.load(std::memory_order_relaxed);
        const uint64_t total_evictions =
            group.total_clean_evictions.load(std::memory_order_relaxed) +
            group.total_dirty_evictions.load(std::memory_order_relaxed);
        const uint32_t refetch_pct =
            percent(std::min(total_fetches, total_evictions), total_evictions);
        const uint32_t score_pct = clean_pct * refetch_pct / 100;

        group.published_clean_pct.store(clean_pct, std::memory_order_release);
        group.published_refetch_pct.store(refetch_pct,
                                          std::memory_order_release);
        group.published_score_pct.store(score_pct,
                                        std::memory_order_release);
        group.published_samples.store(evictions, std::memory_order_release);
        group.publishing.clear(std::memory_order_release);
    }

    void record_backup_group_fetch(size_t size) {
        if (!size_profiled_backup_enabled()) {
            return;
        }
        auto &group =
            backup_behavior_groups[backup_behavior_group_for_size(size)];
        group.window_fetches.fetch_add(1, std::memory_order_relaxed);
        group.total_fetches.fetch_add(1, std::memory_order_relaxed);
    }

    void record_backup_group_eviction(size_t size, bool dirty) {
        if (!size_profiled_backup_enabled()) {
            return;
        }
        const size_t group_idx = backup_behavior_group_for_size(size);
        auto &group = backup_behavior_groups[group_idx];
        if (dirty) {
            group.window_dirty_evictions.fetch_add(1,
                                                    std::memory_order_relaxed);
            group.total_dirty_evictions.fetch_add(1,
                                                   std::memory_order_relaxed);
        } else {
            group.window_clean_evictions.fetch_add(1,
                                                    std::memory_order_relaxed);
            group.total_clean_evictions.fetch_add(1,
                                                   std::memory_order_relaxed);
        }
        publish_backup_profile(group);
    }

    bool sample_logical_object_event(uint64_t &sequence,
                                     uint64_t &weight) const {
        const size_t shift = logical_object_profile_sample_shift;
        weight = uint64_t{1} << shift;
        const uint64_t mask = weight - 1;
        return (sequence++ & mask) == 0;
    }

    void record_resident_group_fetch(
        const FarObjectEntry &entry, size_t size,
        ::FarLib::allocator::RegionPlacement actual_placement) {
        if (!automatic_resident_group_planner_enabled() ||
            !resident_profile_work_phase_active()) {
            return;
        }
        const uint32_t group_id = entry.resident_group_id();
        if (group_id == 0) {
            return;
        }
        ASSERT(group_id < next_resident_placement_group_id.load(
                                  std::memory_order_acquire));
        static thread_local uint64_t sample_sequence = 0;
        const uint64_t weight = uint64_t{1} << resident_profile_sample_shift;
        if ((sample_sequence++ & (weight - 1)) != 0) {
            return;
        }
        auto &group = resident_placement_groups[group_id];
        const uint64_t weighted_bytes = size * weight;
        group.window_fetches.fetch_add(weight, std::memory_order_relaxed);
        group.window_fetch_bytes.fetch_add(weighted_bytes,
                                            std::memory_order_relaxed);
        group.total_fetches.fetch_add(weight, std::memory_order_relaxed);
        group.total_fetch_bytes.fetch_add(weighted_bytes,
                                           std::memory_order_relaxed);
        (void)actual_placement;
    }

    void record_resident_group_eviction(const FarObjectEntry &entry,
                                        size_t size, bool dirty) {
        if (!automatic_resident_group_planner_enabled() ||
            !resident_profile_work_phase_active()) {
            return;
        }
        const uint32_t group_id = entry.resident_group_id();
        if (group_id == 0) {
            return;
        }
        ASSERT(group_id < next_resident_placement_group_id.load(
                                  std::memory_order_acquire));
        static thread_local uint64_t sample_sequence = 0;
        const uint64_t weight = uint64_t{1} << resident_profile_sample_shift;
        if ((sample_sequence++ & (weight - 1)) != 0) {
            return;
        }
        auto &group = resident_placement_groups[group_id];
        const uint64_t weighted_bytes = size * weight;
        auto &window_events = dirty ? group.window_dirty_evictions
                                    : group.window_clean_evictions;
        auto &window_bytes = dirty ? group.window_dirty_evict_bytes
                                   : group.window_clean_evict_bytes;
        auto &total_events = dirty ? group.total_dirty_evictions
                                   : group.total_clean_evictions;
        auto &total_bytes = dirty ? group.total_dirty_evict_bytes
                                  : group.total_clean_evict_bytes;
        window_events.fetch_add(weight, std::memory_order_relaxed);
        window_bytes.fetch_add(weighted_bytes, std::memory_order_relaxed);
        total_events.fetch_add(weight, std::memory_order_relaxed);
        total_bytes.fetch_add(weighted_bytes, std::memory_order_relaxed);
    }

    void record_logical_object_fetch(
        const FarObjectEntry &entry, size_t size,
        ::FarLib::allocator::RegionPlacement actual_placement) {
        record_resident_group_fetch(entry, size, actual_placement);
        const uint32_t owner_id = entry.logical_owner_id();
        if (!logical_object_profile_enabled() || owner_id == 0) {
            return;
        }
        static thread_local uint64_t sample_sequence = 0;
        uint64_t weight;
        if (!sample_logical_object_event(sample_sequence, weight)) {
            return;
        }
        ASSERT(owner_id < logical_object_profiles.size());
        auto &profile = logical_object_profiles[owner_id];
        profile.window_fetches.fetch_add(weight, std::memory_order_relaxed);
        profile.total_fetches.fetch_add(weight, std::memory_order_relaxed);
    }

    void record_logical_object_eviction(const FarObjectEntry &entry,
                                        size_t size, bool dirty) {
        record_resident_group_eviction(entry, size, dirty);
        const uint32_t owner_id = entry.logical_owner_id();
        if (!logical_object_profile_enabled() || owner_id == 0) {
            return;
        }
        static thread_local uint64_t sample_sequence = 0;
        uint64_t weight;
        if (!sample_logical_object_event(sample_sequence, weight)) {
            return;
        }
        ASSERT(owner_id < logical_object_profiles.size());
        auto &profile = logical_object_profiles[owner_id];
        if (dirty) {
            profile.window_dirty_evictions.fetch_add(
                weight, std::memory_order_relaxed);
            profile.total_dirty_evictions.fetch_add(
                weight, std::memory_order_relaxed);
        } else {
            profile.window_clean_evictions.fetch_add(
                weight, std::memory_order_relaxed);
            profile.total_clean_evictions.fetch_add(
                weight, std::memory_order_relaxed);
        }
        publish_backup_profile(profile);
    }

    bool backup_profile_admits(const BackupBehaviorGroupProfile &profile) const {
        return profile.published_samples.load(std::memory_order_acquire) != 0 &&
               profile.published_score_pct.load(std::memory_order_acquire) >=
                   backup_profile_min_score_pct;
    }

    bool profiled_backup_should_admit(const FarObjectEntry &entry,
                                      size_t size) const {
        // Placement, not object identity or learned admission, decides whether
        // a clean fetched object retains its remote backing in this mode.
        if (all_nonresident_backup_mode) return true;
        if (!profiled_backup_enabled()) {
            return true;
        }
        // Selection cannot improve write traffic when the backup budget can
        // cover every nonresident byte that may coexist in local memory.
        // Keep profiling for observability, but retain the complete rolling
        // transient window until there is actual capacity competition.
        if (object_profiled_backup_mode &&
            backup_covers_transient_window) {
            return true;
        }
        if (object_profiled_backup_mode) {
            const uint32_t owner_id = entry.logical_owner_id();
            if (owner_id == 0) {
                return true;
            }
            ASSERT(owner_id < logical_object_profiles.size());
            const auto &profile = logical_object_profiles[owner_id];
            const uint64_t allocated_objects =
                profile.allocated_objects.load(std::memory_order_relaxed);
            const uint64_t profiled_decisions =
                profile.admitted_fetches.load(std::memory_order_relaxed) +
                profile.rejected_fetches.load(std::memory_order_relaxed);
            // Initialization writes can publish a misleading zero score before
            // the first read pass. Explore roughly one object population before
            // using the learned score so cold-start clean evictions keep backups.
            if (allocated_objects == 0 ||
                profiled_decisions < allocated_objects) {
                return true;
            }
            return backup_profile_admits(profile);
        }
        const auto &group =
            backup_behavior_groups[backup_behavior_group_for_size(size)];
        return backup_profile_admits(group);
    }

    void record_profiled_backup_decision(const FarObjectEntry &entry,
                                         size_t size, bool admitted) {
        if (!profiled_backup_enabled()) {
            if (!logical_object_profile_enabled()) {
                return;
            }
        }
        if (size_profiled_backup_enabled()) {
            auto &group =
                backup_behavior_groups[backup_behavior_group_for_size(size)];
            if (admitted) {
                group.admitted_fetches.fetch_add(1,
                                                  std::memory_order_relaxed);
            } else {
                group.rejected_fetches.fetch_add(1,
                                                  std::memory_order_relaxed);
            }
        }
        const uint32_t owner_id = entry.logical_owner_id();
        if (logical_object_profile_enabled() && owner_id != 0) {
            static thread_local uint64_t sample_sequence = 0;
            uint64_t weight;
            if (sample_logical_object_event(sample_sequence, weight)) {
                ASSERT(owner_id < logical_object_profiles.size());
                auto &profile = logical_object_profiles[owner_id];
                if (admitted) {
                    profile.admitted_fetches.fetch_add(
                        weight, std::memory_order_relaxed);
                } else {
                    profile.rejected_fetches.fetch_add(
                        weight, std::memory_order_relaxed);
                }
            }
        }
        if (!admitted && profiled_backup_enabled()) {
            profile::count_remote_backup_admission_attempt();
            profile::count_remote_backup_rejected(size);
        }
    }

    bool try_acquire_remote_backup_budget(size_t bytes, bool fibre_context = false) {
        if (!selective_backup_enabled() ||
            bytes > retained_backup_budget_bytes) {
            return false;
        }
        if (batched_backup_budget) {
            auto &handle = fibre_context ? current_backup_budget_handle()
                                         : current_backup_thread_handle();
            return batched_backup_budget->acquire(handle, bytes);
        }
        if (bypass_backup_usage) {
            bypass_backup_usage->adjust(static_cast<int64_t>(bytes));
            return true;
        }
        uint64_t current = retained_backup_bytes.load(std::memory_order_relaxed);
        while (current <= retained_backup_budget_bytes - bytes) {
            if (retained_backup_bytes.compare_exchange_weak(
                    current, current + bytes, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                uint64_t peak =
                    peak_retained_backup_bytes.load(std::memory_order_relaxed);
                while (peak < current + bytes &&
                       !peak_retained_backup_bytes.compare_exchange_weak(
                           peak, current + bytes, std::memory_order_relaxed,
                           std::memory_order_relaxed)) {
                }
                return true;
            }
        }
        return false;
    }

    bool try_reserve_remote_backup(size_t bytes, bool fibre_context = false) {
        profile::count_remote_backup_admission_attempt();
        if (try_acquire_remote_backup_budget(bytes, fibre_context)) {
            profile::count_remote_backup_retained(bytes);
            return true;
        }
        profile::count_remote_backup_rejected(bytes);
        return false;
    }

    bool reserve_segmented_remote_backup(size_t bytes,
                                         DereferenceScope &scope) {
        ASSERT(segmented_backup_enabled());
        profile::count_remote_backup_admission_attempt();
        if (bytes > retained_backup_budget_bytes) {
            profile::count_remote_backup_rejected(bytes);
            return false;
        }
        if (try_acquire_remote_backup_budget(bytes, true)) {
            profile::count_remote_backup_retained(bytes);
            return true;
        }

        // Profiled admission is advisory.  A selected fetch must not block
        // merely because the rolling backup window is currently full; doing
        // so can leave every runnable mutator waiting for budget that only a
        // later phase would release.
        if (profiled_backup_enabled() &&
            profiled_backup_nonblocking_on_full_enabled()) {
            const uint64_t reject_ordinal =
                profiled_backup_budget_full_reject_count.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            profiled_backup_budget_full_reject_bytes.fetch_add(
                bytes, std::memory_order_relaxed);
            if (backup_budget_diagnostics_enabled() && reject_ordinal <= 32) {
                std::cerr << "profiled_backup.budget_full action=reject"
                          << " ordinal=" << reject_ordinal
                          << " bytes=" << bytes
                          << " retained="
                          << backup_usage_snapshot()
                          << " budget=" << retained_backup_budget_bytes
                          << std::endl;
            }
            profile::count_remote_backup_rejected(bytes);
            return false;
        }

        const uint64_t wait_ordinal =
            segmented_backup_wait_count.fetch_add(
                1, std::memory_order_relaxed) +
            1;
        if (backup_budget_diagnostics_enabled() && wait_ordinal <= 32) {
            std::cerr << "profiled_backup.budget_full action=wait"
                      << " ordinal=" << wait_ordinal
                      << " bytes=" << bytes
                      << " retained="
                      << backup_usage_snapshot()
                      << " budget=" << retained_backup_budget_bytes
                      << std::endl;
        }
        const uint64_t wait_start = get_cycles();
        bool suspended = profile::suspend_work();
        scope.begin_eviction();
        uthread::lock(&eviction_mutex);
        eviction_request_seq.fetch_add(1, std::memory_order_acq_rel);
        mutator_waiters.fetch_add(1, std::memory_order_relaxed);
        uthread::notify_all_locked(&eviction_cond);
        uthread::unlock(&eviction_mutex);
        bool acquired = false;
        while (working.load(std::memory_order_acquire) && !acquired) {
            acquired = try_acquire_remote_backup_budget(bytes, true);
            if (acquired) {
                break;
            }
            uthread::yield();
        }
        mutator_waiters.fetch_sub(1, std::memory_order_relaxed);
        scope.end_eviction();
        profile::resume_work(suspended);
        segmented_backup_wait_cycles.fetch_add(get_cycles() - wait_start,
                                                std::memory_order_relaxed);
        if (!acquired) {
            profile::count_remote_backup_rejected(bytes);
            return false;
        }
        profile::count_remote_backup_retained(bytes);
        return true;
    }

    bool try_reserve_resident_local(size_t bytes) {
        if (resident_local_budget_bytes == 0 ||
            bytes > resident_local_budget_bytes) {
            return false;
        }
        uint64_t current = resident_local_bytes.load(std::memory_order_relaxed);
        while (current <= resident_local_budget_bytes - bytes) {
            if (resident_local_bytes.compare_exchange_weak(
                    current, current + bytes, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                uint64_t peak =
                    peak_resident_local_bytes.load(std::memory_order_relaxed);
                while (peak < current + bytes &&
                       !peak_resident_local_bytes.compare_exchange_weak(
                           peak, current + bytes, std::memory_order_relaxed,
                           std::memory_order_relaxed)) {
                }
                return true;
            }
        }
        return false;
    }

    void release_resident_local(size_t bytes) {
        uint64_t previous =
            resident_local_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
        ASSERT(previous >= bytes);
    }

    bool resident_profile_planner_enabled() const {
        return resident_profile_planner_enabled_flag;
    }

public:
    // Count only completed synchronous demand accesses, using the same local
    // pointer about to be returned to the application. Never inspect an
    // entry's moving remote/local address at fetch submission or CQ polling.
    void record_simple_local_access(const void *local_ptr) const {
        if (!::FarLib::simple_region_heat::enabled() ||
            resident_profile_prefetch_active) return;
        auto &heat = ::FarLib::simple_region_heat::monitor();
        if (heat.select()) heat.add(reinterpret_cast<uintptr_t>(local_ptr));
    }

private:
    void record_resident_profile_reference(
        const FarObjectEntry &entry, profile::ReferenceKind kind) {
        if (::FarLib::simple_dirty_observer::objects_enabled() &&
            !resident_profile_prefetch_active) {
            const auto slot = entry.object_trace_slot();
            if (slot) ::FarLib::simple_dirty_observer::monitor().access(
                slot, kind == profile::ReferenceKind::Write);
        }
        if (::FarLib::allocator::six_group::enabled() &&
            !resident_profile_prefetch_active) {
            trace_object_event(entry,
                kind == profile::ReferenceKind::Write
                    ? ::FarLib::object_group_trace::Kind::Write
                    : ::FarLib::object_group_trace::Kind::Read);
            ::FarLib::allocator::six_group::note_demand(entry.six_record(),
                kind == profile::ReferenceKind::Write);
        }
        if (!resident_profile_active.load(std::memory_order_relaxed) ||
            !resident_profile_planner_enabled_flag ||
            !resident_profile_work_phase_active() ||
            resident_profile_prefetch_active) {
            return;
        }
        profile::count_resident_profile_reference_check();
        if (!ec_batch_mode_ &&
            !region_resident_placement_enabled() &&
            automatic_resident_group_planner_enabled() &&
            resident_profile_plan_ready.load(std::memory_order_acquire) &&
            !entry.is_resident_local()) {
            const auto state = entry.load_state(std::memory_order_relaxed);
            try_promote_profiled_resident(
                const_cast<FarObjectEntry &>(entry), state.size);
        }
        const uint64_t weight = uint64_t{1} << resident_profile_sample_shift;
        const uint64_t mask = weight - 1;
        static thread_local uint64_t sample_sequence = 0;
        if ((sample_sequence++ & mask) != 0) {
            return;
        }
        struct SampleCycleGuard {
            uint64_t start = get_cycles();
            ~SampleCycleGuard() {
                profile::count_resident_profile_sample(get_cycles() - start);
            }
        } sample_cycle_guard;
        if (region_hotness_placement_enabled()) {
            const auto state = entry.load_state(std::memory_order_acquire);
            void *local_ptr = entry.local_addr();
            const auto heap_begin = reinterpret_cast<uintptr_t>(
                ::FarLib::allocator::global_heap.get_heap());
            const auto heap_end = heap_begin +
                ::FarLib::allocator::global_heap.get_heap_size();
            const auto local_addr = reinterpret_cast<uintptr_t>(local_ptr);
            if (!state.invalid && state.state != FREE &&
                state.state != REMOTE && state.state != BUSY &&
                local_ptr != nullptr &&
                local_addr >=
                    heap_begin + sizeof(::FarLib::allocator::BlockHead) &&
                local_addr < heap_end) {
                auto *block =
                    static_cast<::FarLib::allocator::BlockHead *>(local_ptr) -
                    1;
                const auto obj = block->obj_meta_data.load(
                    std::memory_order_acquire);
                if (!obj.is_null() && obj.get_entry_ptr() == &entry &&
                    entry.local_addr() == local_ptr) {
                    auto *region =
                        ::FarLib::allocator::block_to_region(block);
                    auto &counter = kind == profile::ReferenceKind::Write
                                        ? region->profile_window_write_references
                                        : region->profile_window_read_references;
                    counter.fetch_add(weight, std::memory_order_relaxed);
                    auto &total_counter =
                        kind == profile::ReferenceKind::Write
                            ? region->profile_total_write_references
                            : region->profile_total_read_references;
                    total_counter.fetch_add(weight,
                                            std::memory_order_relaxed);
                }
            }
        }
        const uint32_t owner_id = entry.logical_owner_id();
        if (owner_id != 0) {
            ASSERT(owner_id < logical_object_profiles.size());
            auto &owner_profile = logical_object_profiles[owner_id];
            owner_profile.resident_window_references.fetch_add(
                weight, std::memory_order_relaxed);
            owner_profile.resident_total_references.fetch_add(
                weight, std::memory_order_relaxed);
        }
        const uint32_t group_id = entry.resident_group_id();
        if (group_id == 0) {
            return;
        }
        ASSERT(group_id < next_resident_placement_group_id.load(
                                  std::memory_order_acquire));
        auto &group_profile = resident_placement_groups[group_id];
        auto &window = kind == profile::ReferenceKind::Write
                           ? group_profile.window_write_references
                           : group_profile.window_read_references;
        auto &total = kind == profile::ReferenceKind::Write
                          ? group_profile.total_write_references
                          : group_profile.total_read_references;
        window.fetch_add(weight, std::memory_order_relaxed);
        total.fetch_add(weight, std::memory_order_relaxed);
    }

    void publish_resident_profile_plan() {
        struct Candidate {
            uint32_t owner_id;
            uint64_t references;
            uint64_t allocated_bytes;
            uint64_t current_bytes;
            uint64_t target_bytes;
        };

        resident_profile_active.store(false, std::memory_order_release);
        std::vector<Candidate> candidates;
        const uint32_t owner_limit =
            next_logical_owner_id.load(std::memory_order_acquire);
        for (uint32_t owner_id = 1; owner_id < owner_limit; ++owner_id) {
            auto &profile = logical_object_profiles[owner_id];
            const uint64_t allocated = profile.allocated_footprint_bytes.load(
                std::memory_order_relaxed);
            if (allocated == 0) {
                continue;
            }
            candidates.push_back(
                {.owner_id = owner_id,
                 .references = profile.resident_window_references.exchange(
                     0, std::memory_order_acq_rel),
                 .allocated_bytes = allocated,
                 .current_bytes = profile.resident_current_bytes.load(
                     std::memory_order_relaxed),
                 .target_bytes = 0});
        }
        const uint64_t migration_hysteresis_pct =
            ::FarLib::get_config()
                .resident_profile_migration_hysteresis_pct;

        const auto density_greater = [](const Candidate &left,
                                        const Candidate &right) {
            const __uint128_t left_score =
                static_cast<__uint128_t>(left.references) *
                right.allocated_bytes;
            const __uint128_t right_score =
                static_cast<__uint128_t>(right.references) *
                left.allocated_bytes;
            return left_score > right_score;
        };
        std::stable_sort(
            candidates.begin(), candidates.end(),
            [&density_greater](const Candidate &left,
                               const Candidate &right) {
                if (density_greater(left, right)) {
                    return true;
                }
                if (density_greater(right, left)) {
                    return false;
                }
                if (left.current_bytes != right.current_bytes) {
                    return left.current_bytes > right.current_bytes;
                }
                return left.owner_id < right.owner_id;
            });

        uint64_t assigned = 0;
        for (auto &candidate : candidates) {
            ASSERT(candidate.current_bytes <= candidate.allocated_bytes);
            candidate.target_bytes = candidate.current_bytes;
            assigned += candidate.target_bytes;
        }

        // First reconcile an under/over-filled resident budget without moving
        // capacity between owners. Normal runs enter with an exactly full
        // budget, but this also handles a plan published during allocation.
        if (assigned < resident_local_budget_bytes) {
            uint64_t unassigned = resident_local_budget_bytes - assigned;
            for (auto &candidate : candidates) {
                const uint64_t capacity =
                    candidate.allocated_bytes - candidate.target_bytes;
                const uint64_t add = std::min(capacity, unassigned);
                candidate.target_bytes += add;
                unassigned -= add;
                if (unassigned == 0) {
                    break;
                }
            }
        } else if (assigned > resident_local_budget_bytes) {
            uint64_t excess = assigned - resident_local_budget_bytes;
            for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
                const uint64_t remove =
                    std::min(it->target_bytes, excess);
                it->target_bytes -= remove;
                excess -= remove;
                if (excess == 0) {
                    break;
                }
            }
        }

        const auto worth_migrating =
            [migration_hysteresis_pct](const Candidate &receiver,
                                       const Candidate &donor) {
                const __uint128_t receiver_score =
                    static_cast<__uint128_t>(receiver.references) *
                    donor.allocated_bytes * 100;
                const __uint128_t donor_score =
                    static_cast<__uint128_t>(donor.references) *
                    receiver.allocated_bytes *
                    (100 + migration_hysteresis_pct);
                return receiver_score > donor_score;
            };

        // Start from the current placement. Only exchange resident bytes when
        // an unresident segment beats the coldest resident segment by the
        // configured margin. A partially resident owner therefore does not
        // receive a residency bonus for its unresident remainder.
        for (auto &receiver : candidates) {
            while (receiver.target_bytes < receiver.allocated_bytes) {
                Candidate *donor = nullptr;
                for (auto it = candidates.rbegin(); it != candidates.rend();
                     ++it) {
                    if (it->owner_id != receiver.owner_id &&
                        it->target_bytes != 0) {
                        donor = &*it;
                        break;
                    }
                }
                if (donor == nullptr ||
                    !worth_migrating(receiver, *donor)) {
                    break;
                }
                const uint64_t transfer = std::min(
                    receiver.allocated_bytes - receiver.target_bytes,
                    donor->target_bytes);
                receiver.target_bytes += transfer;
                donor->target_bytes -= transfer;
            }
        }

        assigned = 0;
        for (const auto &candidate : candidates) {
            logical_object_profiles[candidate.owner_id]
                .resident_target_bytes.store(candidate.target_bytes,
                                             std::memory_order_release);
            assigned += candidate.target_bytes;
        }
        const uint64_t remaining =
            assigned < resident_local_budget_bytes
                ? resident_local_budget_bytes - assigned
                : 0;

        const uint64_t plan_id =
            resident_profile_plan_count.fetch_add(1,
                                                  std::memory_order_relaxed) +
            1;
        std::cerr << "resident.plan id=" << plan_id
                  << " owners=" << candidates.size()
                  << " budget=" << resident_local_budget_bytes
                  << " migration_hysteresis_pct="
                  << migration_hysteresis_pct
                  << " unassigned=" << remaining
                  << " apply=" << resident_profile_apply_plan_flag
                  << std::endl;
        for (const auto &candidate : candidates) {
            const uint64_t target =
                logical_object_profiles[candidate.owner_id]
                    .resident_target_bytes.load(std::memory_order_relaxed);
            std::cerr << "resident.plan owner=" << candidate.owner_id
                      << " references=" << candidate.references
                      << " allocated=" << candidate.allocated_bytes
                      << " current=" << candidate.current_bytes
                      << " target=" << target << std::endl;
        }

        if (!resident_profile_apply_plan_flag) {
            return;
        }
        resident_profile_open_migration_gate();
        eviction_request_seq.fetch_add(1, std::memory_order_acq_rel);
        uthread::notify_all(&eviction_cond, &eviction_mutex);
    }

    static uint64_t resident_profile_ema_update(uint64_t previous,
                                                uint64_t observed,
                                                size_t decay_shift) {
        const uint64_t carry = decay_shift >= 64
                                   ? 0
                                   : (previous >> decay_shift);
        return observed > std::numeric_limits<uint64_t>::max() - carry
                   ? std::numeric_limits<uint64_t>::max()
                   : observed + carry;
    }

    static uint64_t resident_profile_now_us() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    static void resident_profile_update_max(std::atomic<uint64_t> &maximum,
                                            uint64_t value) {
        uint64_t observed = maximum.load(std::memory_order_relaxed);
        while (observed < value &&
               !maximum.compare_exchange_weak(
                   observed, value, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }

    void resident_profile_maybe_complete_active_plan(uint64_t plan_id) {
        if (plan_id == 0 ||
            resident_profile_active_plan_id.load(std::memory_order_acquire) !=
                plan_id) {
            return;
        }
        const uint64_t scheduled_promotion =
            resident_profile_active_plan_scheduled_promotion_bytes.load(
                std::memory_order_relaxed);
        const uint64_t scheduled_demotion =
            resident_profile_active_plan_scheduled_demotion_bytes.load(
                std::memory_order_relaxed);
        const uint64_t completed_promotion =
            resident_profile_active_plan_promotion_committed_bytes.load(
                std::memory_order_relaxed) +
            resident_profile_active_plan_promotion_cancelled_bytes.load(
                std::memory_order_relaxed) +
            resident_profile_active_plan_promotion_carried_bytes.load(
                std::memory_order_relaxed);
        const uint64_t completed_demotion =
            resident_profile_active_plan_demotion_committed_bytes.load(
                std::memory_order_relaxed) +
            resident_profile_active_plan_demotion_cancelled_bytes.load(
                std::memory_order_relaxed) +
            resident_profile_active_plan_demotion_carried_bytes.load(
                std::memory_order_relaxed);
        if (completed_promotion < scheduled_promotion ||
            completed_demotion < scheduled_demotion) {
            return;
        }
        uint64_t expected = 0;
        const uint64_t now = resident_profile_now_us();
        if (resident_profile_active_plan_complete_time_us
                .compare_exchange_strong(expected, now,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
            resident_profile_active_plan_complete_check.store(
                resident_profile_candidate_check_count.load(
                    std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
    }

    void resident_profile_record_migration_started(uint64_t plan_id,
                                                    uint64_t bytes,
                                                    bool promote) {
        if (plan_id == 0 ||
            resident_profile_active_plan_id.load(std::memory_order_acquire) !=
                plan_id) {
            return;
        }
        auto &active = promote
                           ? resident_profile_active_plan_promotion_started_bytes
                           : resident_profile_active_plan_demotion_started_bytes;
        auto &total = promote ? resident_profile_promotion_started_bytes_total
                              : resident_profile_demotion_started_bytes_total;
        active.fetch_add(bytes, std::memory_order_relaxed);
        total.fetch_add(bytes, std::memory_order_relaxed);
        uint64_t expected = 0;
        const uint64_t now = resident_profile_now_us();
        if (resident_profile_active_plan_first_start_time_us
                .compare_exchange_strong(expected, now,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
            resident_profile_active_plan_first_start_check.store(
                resident_profile_candidate_check_count.load(
                    std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
    }

    void resident_profile_record_migration_rolled_back(uint64_t plan_id,
                                                       uint64_t bytes,
                                                       bool promote) {
        if (plan_id == 0 ||
            resident_profile_active_plan_id.load(std::memory_order_acquire) !=
                plan_id) {
            return;
        }
        auto &active =
            promote
                ? resident_profile_active_plan_promotion_rolled_back_bytes
                : resident_profile_active_plan_demotion_rolled_back_bytes;
        auto &total =
            promote ? resident_profile_promotion_rolled_back_bytes_total
                    : resident_profile_demotion_rolled_back_bytes_total;
        active.fetch_add(bytes, std::memory_order_relaxed);
        total.fetch_add(bytes, std::memory_order_relaxed);
    }

    void resident_profile_record_migration_committed(uint64_t plan_id,
                                                      uint64_t bytes,
                                                      bool promote) {
        if (plan_id == 0 ||
            resident_profile_active_plan_id.load(std::memory_order_acquire) !=
                plan_id) {
            return;
        }
        auto &active =
            promote ? resident_profile_active_plan_promotion_committed_bytes
                    : resident_profile_active_plan_demotion_committed_bytes;
        const uint64_t scheduled =
            promote
                ? resident_profile_active_plan_scheduled_promotion_bytes.load(
                      std::memory_order_relaxed)
                : resident_profile_active_plan_scheduled_demotion_bytes.load(
                      std::memory_order_relaxed);
        auto &total = promote ? resident_profile_promotion_committed_bytes_total
                              : resident_profile_demotion_committed_bytes_total;
        total.fetch_add(bytes, std::memory_order_relaxed);
        uint64_t observed = active.load(std::memory_order_relaxed);
        while (observed < scheduled) {
            const uint64_t accounted =
                std::min(bytes, scheduled - observed);
            if (active.compare_exchange_weak(
                    observed, observed + accounted,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        }
        resident_profile_maybe_complete_active_plan(plan_id);
    }

    void resident_profile_finalize_active_plan(
        uint64_t check_id, uint64_t cancel_promotion,
        uint64_t cancel_demotion, uint64_t carry_promotion,
        uint64_t carry_demotion) {
        const uint64_t plan_id = resident_profile_active_plan_id.load(
            std::memory_order_acquire);
        if (plan_id == 0) {
            return;
        }
        const uint64_t scheduled_promotion =
            resident_profile_active_plan_scheduled_promotion_bytes.load();
        const uint64_t scheduled_demotion =
            resident_profile_active_plan_scheduled_demotion_bytes.load();
        const uint64_t committed_promotion =
            resident_profile_active_plan_promotion_committed_bytes.load();
        const uint64_t committed_demotion =
            resident_profile_active_plan_demotion_committed_bytes.load();
        if (region_resident_placement_enabled()) {
            std::cerr << "resident.region_plan_lifecycle"
                      << " plan_id=" << plan_id
                      << " publish_check="
                      << resident_profile_active_plan_publish_check.load()
                      << " scheduled_promotion_bytes="
                      << scheduled_promotion
                      << " scheduled_demotion_bytes="
                      << scheduled_demotion
                      << " committed_promotion_bytes="
                      << committed_promotion
                      << " committed_demotion_bytes="
                      << committed_demotion
                      << " replaced_at_check=" << check_id << std::endl;
            return;
        }
        const auto fit_remaining = [](uint64_t scheduled,
                                      uint64_t committed,
                                      uint64_t &cancel,
                                      uint64_t &carry) {
            ASSERT(committed <= scheduled);
            uint64_t remaining = scheduled - committed;
            cancel = std::min(cancel, remaining);
            remaining -= cancel;
            carry = std::min(carry, remaining);
        };
        fit_remaining(scheduled_promotion, committed_promotion,
                      cancel_promotion, carry_promotion);
        fit_remaining(scheduled_demotion, committed_demotion,
                      cancel_demotion, carry_demotion);
        resident_profile_active_plan_promotion_cancelled_bytes.fetch_add(
            cancel_promotion, std::memory_order_relaxed);
        resident_profile_active_plan_demotion_cancelled_bytes.fetch_add(
            cancel_demotion, std::memory_order_relaxed);
        resident_profile_promotion_cancelled_bytes_total.fetch_add(
            cancel_promotion, std::memory_order_relaxed);
        resident_profile_demotion_cancelled_bytes_total.fetch_add(
            cancel_demotion, std::memory_order_relaxed);
        resident_profile_active_plan_promotion_carried_bytes.fetch_add(
            carry_promotion, std::memory_order_relaxed);
        resident_profile_active_plan_demotion_carried_bytes.fetch_add(
            carry_demotion, std::memory_order_relaxed);
        resident_profile_promotion_carried_bytes_total.fetch_add(
            carry_promotion, std::memory_order_relaxed);
        resident_profile_demotion_carried_bytes_total.fetch_add(
            carry_demotion, std::memory_order_relaxed);
        resident_profile_maybe_complete_active_plan(plan_id);
        std::cerr
            << "resident.group_plan_lifecycle plan_id=" << plan_id
            << " publish_check="
            << resident_profile_active_plan_publish_check.load()
            << " publish_time_us="
            << resident_profile_active_plan_publish_time_us.load()
            << " scheduled_promotion_bytes=" << scheduled_promotion
            << " scheduled_demotion_bytes=" << scheduled_demotion
            << " promotion_started_bytes="
            << resident_profile_active_plan_promotion_started_bytes.load()
            << " demotion_started_bytes="
            << resident_profile_active_plan_demotion_started_bytes.load()
            << " promotion_committed_bytes=" << committed_promotion
            << " demotion_committed_bytes=" << committed_demotion
            << " promotion_rolled_back_bytes="
            << resident_profile_active_plan_promotion_rolled_back_bytes.load()
            << " demotion_rolled_back_bytes="
            << resident_profile_active_plan_demotion_rolled_back_bytes.load()
            << " promotion_cancelled_bytes="
            << resident_profile_active_plan_promotion_cancelled_bytes.load()
            << " demotion_cancelled_bytes="
            << resident_profile_active_plan_demotion_cancelled_bytes.load()
            << " promotion_carried_bytes="
            << resident_profile_active_plan_promotion_carried_bytes.load()
            << " demotion_carried_bytes="
            << resident_profile_active_plan_demotion_carried_bytes.load()
            << " first_start_check="
            << resident_profile_active_plan_first_start_check.load()
            << " first_start_time_us="
            << resident_profile_active_plan_first_start_time_us.load()
            << " complete_check="
            << resident_profile_active_plan_complete_check.load()
            << " complete_time_us="
            << resident_profile_active_plan_complete_time_us.load()
            << " replaced_at_check=" << check_id << std::endl;
    }

    void resident_profile_begin_active_plan(uint64_t plan_id,
                                            uint64_t check_id,
                                            uint64_t promotion_bytes,
                                            uint64_t demotion_bytes) {
        resident_profile_active_plan_publish_check.store(
            check_id, std::memory_order_relaxed);
        resident_profile_active_plan_publish_time_us.store(
            resident_profile_now_us(), std::memory_order_relaxed);
        resident_profile_active_plan_scheduled_promotion_bytes.store(
            promotion_bytes, std::memory_order_relaxed);
        resident_profile_active_plan_scheduled_demotion_bytes.store(
            demotion_bytes, std::memory_order_relaxed);
        resident_profile_active_plan_promotion_started_bytes.store(0);
        resident_profile_active_plan_demotion_started_bytes.store(0);
        resident_profile_active_plan_promotion_committed_bytes.store(0);
        resident_profile_active_plan_demotion_committed_bytes.store(0);
        resident_profile_active_plan_promotion_rolled_back_bytes.store(0);
        resident_profile_active_plan_demotion_rolled_back_bytes.store(0);
        resident_profile_active_plan_promotion_cancelled_bytes.store(0);
        resident_profile_active_plan_demotion_cancelled_bytes.store(0);
        resident_profile_active_plan_promotion_carried_bytes.store(0);
        resident_profile_active_plan_demotion_carried_bytes.store(0);
        resident_profile_active_plan_first_start_check.store(0);
        resident_profile_active_plan_first_start_time_us.store(0);
        resident_profile_active_plan_complete_check.store(0);
        resident_profile_active_plan_complete_time_us.store(0);
        resident_profile_active_plan_id.store(plan_id,
                                              std::memory_order_release);
        resident_profile_maybe_complete_active_plan(plan_id);
    }

    void apply_region_resident_targets(uint32_t group_limit) {
        if (!region_group_binding_enabled() || group_limit <= 1) {
            return;
        }
        std::vector<::FarLib::allocator::RegionPlacementGroupSnapshot>
            snapshots(group_limit);
        ::FarLib::allocator::global_heap
            .get_region_placement_group_snapshots(snapshots);
        const uint64_t plan_id = resident_profile_active_plan_id.load(
            std::memory_order_acquire);

        for (uint32_t group_id = 1; group_id < group_limit; ++group_id) {
            const size_t target_regions = static_cast<size_t>(
                resident_placement_groups[group_id]
                    .resident_target_bytes.load(std::memory_order_acquire) /
                ::FarLib::allocator::RegionSize);
            const size_t current_regions = snapshots[group_id].resident_regions;
            if (current_regions <= target_regions) {
                continue;
            }
            const size_t changed =
                ::FarLib::allocator::global_heap
                    .reclassify_region_placement_group(
                        group_id,
                        ::FarLib::allocator::RegionPlacement::Resident,
                        ::FarLib::allocator::RegionPlacement::Streaming,
                        current_regions - target_regions);
            if (changed != 0) {
                const uint64_t bytes =
                    changed * ::FarLib::allocator::RegionSize;
                resident_profile_demoted_bytes.fetch_add(
                    bytes, std::memory_order_relaxed);
            }
        }

        ::FarLib::allocator::global_heap
            .get_region_placement_group_snapshots(snapshots);
        for (uint32_t group_id = 1; group_id < group_limit; ++group_id) {
            const size_t target_regions = static_cast<size_t>(
                resident_placement_groups[group_id]
                    .resident_target_bytes.load(std::memory_order_acquire) /
                ::FarLib::allocator::RegionSize);
            const size_t current_regions = snapshots[group_id].resident_regions;
            if (current_regions >= target_regions) {
                continue;
            }
            const size_t changed =
                ::FarLib::allocator::global_heap
                    .reclassify_region_placement_group(
                        group_id,
                        ::FarLib::allocator::RegionPlacement::Streaming,
                        ::FarLib::allocator::RegionPlacement::Resident,
                        target_regions - current_regions);
            if (changed != 0) {
                const uint64_t bytes =
                    changed * ::FarLib::allocator::RegionSize;
                resident_profile_promoted_bytes.fetch_add(
                    bytes, std::memory_order_relaxed);
            }
        }

        ::FarLib::allocator::global_heap
            .get_region_placement_group_snapshots(snapshots);
        for (uint32_t group_id = 1; group_id < group_limit; ++group_id) {
            resident_placement_groups[group_id]
                .resident_current_bytes.store(
                    snapshots[group_id].resident_regions *
                        ::FarLib::allocator::RegionSize,
                    std::memory_order_release);
        }
        if (plan_id != 0) {
            uint64_t remaining_promotion = 0;
            uint64_t remaining_demotion = 0;
            for (uint32_t group_id = 1; group_id < group_limit; ++group_id) {
                const uint64_t target =
                    resident_placement_groups[group_id]
                        .resident_target_bytes.load(
                            std::memory_order_acquire);
                const uint64_t current =
                    snapshots[group_id].resident_regions *
                    ::FarLib::allocator::RegionSize;
                if (target > current) {
                    remaining_promotion += target - current;
                } else {
                    remaining_demotion += current - target;
                }
            }
            const uint64_t scheduled_promotion =
                resident_profile_active_plan_scheduled_promotion_bytes.load(
                    std::memory_order_acquire);
            const uint64_t scheduled_demotion =
                resident_profile_active_plan_scheduled_demotion_bytes.load(
                    std::memory_order_acquire);
            resident_profile_active_plan_promotion_committed_bytes.store(
                scheduled_promotion > remaining_promotion
                    ? scheduled_promotion - remaining_promotion
                    : 0,
                std::memory_order_release);
            resident_profile_active_plan_demotion_committed_bytes.store(
                scheduled_demotion > remaining_demotion
                    ? scheduled_demotion - remaining_demotion
                    : 0,
                std::memory_order_release);
            resident_profile_maybe_complete_active_plan(plan_id);
            const uint64_t last_logged =
                resident_region_last_distribution_plan_id.exchange(
                    plan_id, std::memory_order_acq_rel);
            if (last_logged != plan_id) {
                const uint32_t midpoint = group_limit / 2;
                uint64_t lower_current = 0;
                uint64_t upper_current = 0;
                uint64_t lower_target = 0;
                uint64_t upper_target = 0;
                for (uint32_t group_id = 1; group_id < group_limit;
                     ++group_id) {
                    const uint64_t current =
                        snapshots[group_id].resident_regions *
                        ::FarLib::allocator::RegionSize;
                    const uint64_t target =
                        resident_placement_groups[group_id]
                            .resident_target_bytes.load(
                                std::memory_order_relaxed);
                    if (group_id < midpoint) {
                        lower_current += current;
                        lower_target += target;
                    } else {
                        upper_current += current;
                        upper_target += target;
                    }
                }
                std::cerr << "resident.region_distribution"
                          << " plan_id=" << plan_id
                          << " lower_current_bytes=" << lower_current
                          << " upper_current_bytes=" << upper_current
                          << " lower_target_bytes=" << lower_target
                          << " upper_target_bytes=" << upper_target
                          << std::endl;
            }
        }
        resident_local_bytes.store(
            ::FarLib::allocator::global_heap
                    .get_resident_reserved_regions() *
                ::FarLib::allocator::RegionSize,
            std::memory_order_release);
    }

    void publish_region_fetch_hotness_predictions(uint64_t check_id);

    bool refill_resident_fetch_capacity(
        const std::vector<::FarLib::allocator::RegionHotnessSnapshot>
            &snapshots,
        uint64_t check_id);

    void publish_resident_region_hotness_plan(
        bool complete_phase = false);

    void publish_resident_group_plan(bool complete_phase = false) {
        if (region_hotness_placement_enabled()) {
            publish_resident_region_hotness_plan(complete_phase);
            return;
        }
        struct Candidate {
            uint32_t group_id;
            uint64_t weighted_references;
            uint64_t allocated_bytes;
            uint64_t current_bytes;
            uint64_t committed_target_bytes;
            uint64_t candidate_target_bytes;
            uint64_t footprint_quantum;
            uint64_t promotion_opportunities;
        };

        const auto check_start = std::chrono::steady_clock::now();
        const uint64_t collection_start_cycles = get_cycles();
        const bool had_plan =
            resident_profile_plan_ready.load(std::memory_order_acquire);
        const uint64_t check_time_us = resident_profile_now_us();
        const uint64_t check_id =
            resident_profile_candidate_check_count.fetch_add(
                1, std::memory_order_relaxed) +
            1;
        resident_profile_plan_check_count.fetch_add(
            1, std::memory_order_relaxed);
        const uint32_t group_limit =
            next_resident_placement_group_id.load(std::memory_order_acquire);

        const bool grouped_region_placement =
            region_group_binding_enabled();
        if (grouped_region_placement && had_plan) {
            apply_region_resident_targets(group_limit);
        }
        std::vector<::FarLib::allocator::RegionPlacementGroupSnapshot>
            region_group_snapshots;
        if (grouped_region_placement) {
            region_group_snapshots.resize(group_limit);
            ::FarLib::allocator::global_heap
                .get_region_placement_group_snapshots(
                    region_group_snapshots);
        }

        resident_profile_committed_targets.resize(group_limit, 0);
        resident_profile_previous_candidates.resize(group_limit, 0);
        resident_profile_previous_current.resize(group_limit, 0);
        resident_profile_previous_candidate_directions.resize(group_limit, 2);
        resident_profile_candidate_direction_streaks.resize(group_limit, 0);
        resident_profile_promotion_stall_checks.resize(group_limit, 0);
        resident_profile_group_planner_initialized.resize(group_limit, 0);
        resident_profile_candidate_seen.resize(group_limit, 0);

        std::vector<Candidate> candidates;
        candidates.reserve(group_limit > 0 ? group_limit - 1 : 0);
        std::vector<uint32_t> inactive_group_ids;
        bool dynamic_lifetime_unsupported = false;
        for (uint32_t group_id = 1; group_id < group_limit; ++group_id) {
            auto &profile = resident_placement_groups[group_id];
            const uint64_t window_read =
                profile.window_read_references.exchange(
                    0, std::memory_order_acq_rel);
            const uint64_t window_write =
                profile.window_write_references.exchange(
                    0, std::memory_order_acq_rel);
            const uint64_t opportunities =
                profile.window_promotion_opportunities.exchange(
                    0, std::memory_order_acq_rel);
            const uint64_t ema_read = resident_profile_ema_update(
                profile.ema_read_references.load(std::memory_order_relaxed),
                window_read, resident_profile_ema_decay_shift);
            const uint64_t ema_write = resident_profile_ema_update(
                profile.ema_write_references.load(std::memory_order_relaxed),
                window_write, resident_profile_ema_decay_shift);
            profile.ema_read_references.store(ema_read,
                                               std::memory_order_relaxed);
            profile.ema_write_references.store(ema_write,
                                                std::memory_order_relaxed);
            const __uint128_t weighted =
                static_cast<__uint128_t>(ema_write) *
                    resident_profile_write_weight +
                ema_read;
            const uint64_t weighted_references =
                weighted > std::numeric_limits<uint64_t>::max()
                    ? std::numeric_limits<uint64_t>::max()
                    : static_cast<uint64_t>(weighted);
            const uint64_t live_footprint =
                profile.live_footprint_bytes.load(
                    std::memory_order_relaxed);
            uint64_t allocated = grouped_region_placement
                                           ? ((live_footprint +
                                               ::FarLib::allocator::RegionSize -
                                               1) /
                                              ::FarLib::allocator::RegionSize) *
                                                 ::FarLib::allocator::RegionSize
                                           : live_footprint;
            const uint64_t current = grouped_region_placement
                                         ? region_group_snapshots[group_id]
                                                   .resident_regions *
                                               ::FarLib::allocator::RegionSize
                                         : profile.resident_current_bytes.load(
                                               std::memory_order_relaxed);
            if (grouped_region_placement && current > allocated) {
                allocated = current;
            }
            if (grouped_region_placement) {
                profile.resident_current_bytes.store(
                    current, std::memory_order_relaxed);
            }
            if (!resident_profile_group_planner_initialized[group_id]) {
                // Cold-start T is the actual placement R, never the zero-filled
                // public target array.
                resident_profile_committed_targets[group_id] =
                    had_plan ? profile.resident_target_bytes.load(
                                   std::memory_order_acquire)
                             : current;
                resident_profile_previous_candidates[group_id] = current;
                resident_profile_previous_current[group_id] = current;
                resident_profile_previous_candidate_directions[group_id] = 2;
                resident_profile_group_planner_initialized[group_id] = 1;
            }
            if (allocated == 0) {
                dynamic_lifetime_unsupported |=
                    resident_profile_committed_targets[group_id] != 0;
                inactive_group_ids.push_back(group_id);
                continue;
            }
            candidates.push_back(
                {.group_id = group_id,
                 .weighted_references = weighted_references,
                 .allocated_bytes = allocated,
                 .current_bytes = current,
                 .committed_target_bytes =
                     resident_profile_committed_targets[group_id],
                 .candidate_target_bytes = 0,
                 .footprint_quantum = grouped_region_placement
                                          ? ::FarLib::allocator::RegionSize
                                          : ::FarLib::allocator::get_bin_size(
                                                profile.allocation_bin),
                 .promotion_opportunities = opportunities});
        }
        const uint64_t collection_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - check_start)
                .count());
        resident_profile_collection_us_total.fetch_add(
            collection_us, std::memory_order_relaxed);
        resident_profile_update_max(resident_profile_collection_us_max,
                                    collection_us);
        profile::count_resident_profile_planner_collection_cycles(
            get_cycles() - collection_start_cycles);

        const auto density_greater = [](const Candidate &left,
                                        const Candidate &right) {
            const __uint128_t left_score =
                static_cast<__uint128_t>(left.weighted_references) *
                right.allocated_bytes;
            const __uint128_t right_score =
                static_cast<__uint128_t>(right.weighted_references) *
                left.allocated_bytes;
            return left_score > right_score;
        };
        const uint64_t planner_resident_budget_bytes =
            region_resident_placement_enabled()
                ? ::FarLib::allocator::global_heap
                          .get_resident_region_budget() *
                      ::FarLib::allocator::RegionSize
                : resident_local_budget_bytes;
        const bool skip_exhausted_donor_scan =
            ::FarLib::get_config().exclusive_cache &&
            env_flag_or_default("FARLIB_PLANNER_BUDGET_EARLY_EXIT", false);
        const auto compute_candidate_targets =
            [this, &density_greater,
             planner_resident_budget_bytes,
             skip_exhausted_donor_scan](std::vector<Candidate> &work) {
                std::stable_sort(
                    work.begin(), work.end(),
                    [&density_greater](const Candidate &left,
                                       const Candidate &right) {
                        if (density_greater(left, right)) {
                            return true;
                        }
                        if (density_greater(right, left)) {
                            return false;
                        }
                        if (left.current_bytes != right.current_bytes) {
                            return left.current_bytes > right.current_bytes;
                        }
                        return left.group_id < right.group_id;
                    });

                uint64_t assigned = 0;
                for (auto &candidate : work) {
                    ASSERT(candidate.current_bytes <=
                           candidate.allocated_bytes);
                    candidate.candidate_target_bytes =
                        candidate.current_bytes;
                    assigned += candidate.candidate_target_bytes;
                }
                if (assigned < planner_resident_budget_bytes) {
                    uint64_t unassigned =
                        planner_resident_budget_bytes - assigned;
                    for (auto &candidate : work) {
                        const uint64_t capacity =
                            candidate.allocated_bytes -
                            candidate.candidate_target_bytes;
                        const uint64_t limit =
                            std::min(capacity, unassigned);
                        const uint64_t add =
                            limit - limit % candidate.footprint_quantum;
                        candidate.candidate_target_bytes += add;
                        unassigned -= add;
                        if (unassigned == 0) {
                            break;
                        }
                    }
                } else if (assigned > planner_resident_budget_bytes) {
                    uint64_t excess =
                        assigned - planner_resident_budget_bytes;
                    for (auto it = work.rbegin();
                         it != work.rend() && excess != 0; ++it) {
                        const uint64_t rounded_excess =
                            ((excess + it->footprint_quantum - 1) /
                             it->footprint_quantum) *
                            it->footprint_quantum;
                        const uint64_t remove = std::min(
                            it->candidate_target_bytes, rounded_excess);
                        it->candidate_target_bytes -= remove;
                        excess = remove >= excess ? 0 : excess - remove;
                    }
                }

                const uint64_t migration_hysteresis_pct =
                    ::FarLib::get_config()
                        .resident_profile_migration_hysteresis_pct;
                const auto worth_migrating =
                    [migration_hysteresis_pct](const Candidate &receiver,
                                               const Candidate &donor) {
                        const __uint128_t receiver_score =
                            static_cast<__uint128_t>(
                                receiver.weighted_references) *
                            donor.allocated_bytes * 100;
                        const __uint128_t donor_score =
                            static_cast<__uint128_t>(
                                donor.weighted_references) *
                            receiver.allocated_bytes *
                            (100 + migration_hysteresis_pct);
                        return receiver_score > donor_score;
                    };
                uint64_t migration_remaining =
                    planner_resident_budget_bytes *
                    resident_profile_migration_budget_pct / 100;
                uint64_t migration_objects_remaining =
                    resident_profile_migration_budget_objects;
                const bool limit_migration_objects =
                    migration_objects_remaining != 0;
                for (auto &receiver : work) {
                    while (migration_remaining != 0 &&
                           (!limit_migration_objects ||
                            migration_objects_remaining != 0) &&
                           receiver.candidate_target_bytes <
                               receiver.allocated_bytes) {
                        // A positive transfer is a multiple of both object
                        // quanta and consumes at least two object operations.
                        if (skip_exhausted_donor_scan &&
                            (migration_remaining < receiver.footprint_quantum ||
                             (limit_migration_objects &&
                              migration_objects_remaining < 2))) {
                            break;
                        }
                        Candidate *donor = nullptr;
                        uint64_t transfer = 0;
                        for (auto it = work.rbegin(); it != work.rend(); ++it) {
                            if (it->group_id == receiver.group_id ||
                                it->candidate_target_bytes == 0) {
                                continue;
                            }
                            if (!worth_migrating(receiver, *it)) {
                                break;
                            }
                            const uint64_t quantum = std::lcm(
                                receiver.footprint_quantum,
                                it->footprint_quantum);
                            const uint64_t limit = std::min(
                                {receiver.allocated_bytes -
                                     receiver.candidate_target_bytes,
                                 it->candidate_target_bytes,
                                 migration_remaining});
                            uint64_t transfer_units = limit / quantum;
                            if (limit_migration_objects) {
                                const uint64_t object_ops_per_unit =
                                    quantum / receiver.footprint_quantum +
                                    quantum / it->footprint_quantum;
                                transfer_units = std::min(
                                    transfer_units,
                                    migration_objects_remaining /
                                        object_ops_per_unit);
                            }
                            transfer = transfer_units * quantum;
                            if (transfer != 0) {
                                donor = &*it;
                                break;
                            }
                        }
                        if (donor == nullptr) {
                            break;
                        }
                        receiver.candidate_target_bytes += transfer;
                        donor->candidate_target_bytes -= transfer;
                        migration_remaining -= transfer;
                        if (limit_migration_objects) {
                            const uint64_t object_ops =
                                transfer / receiver.footprint_quantum +
                                transfer / donor->footprint_quantum;
                            ASSERT(object_ops <= migration_objects_remaining);
                            migration_objects_remaining -= object_ops;
                        }
                    }
                    if (migration_remaining == 0) {
                        break;
                    }
                }
            };

        const auto direction = [](uint64_t target,
                                  uint64_t baseline) -> int8_t {
            return target > baseline ? 1 : (target < baseline ? -1 : 0);
        };
        const auto candidate_build_start = std::chrono::steady_clock::now();
        const uint64_t candidate_build_start_cycles = get_cycles();
        compute_candidate_targets(candidates);

        std::vector<uint64_t> optimistic_targets(group_limit, 0);
        std::vector<uint64_t> optimistic_current(group_limit, 0);
        std::vector<int8_t> candidate_directions(group_limit, 0);
        std::vector<uint8_t> next_direction_streaks(group_limit, 0);
        std::vector<uint8_t> next_stall_checks(
            resident_profile_promotion_stall_checks);
        std::vector<uint8_t> stable_groups(group_limit, 0);
        uint64_t replacement_in = 0;
        uint64_t replacement_out = 0;
        uint64_t active_delta_in = 0;
        uint64_t active_delta_out = 0;
        uint64_t target_overlap_bytes = 0;
        uint64_t stable_group_count = 0;
        uint64_t estimated_promotion_objects = 0;
        uint64_t estimated_demotion_objects = 0;
        long double expected_local_reference_score_delta = 0;
        bool all_candidates_seen = true;
        for (const auto &candidate : candidates) {
            const uint32_t group_id = candidate.group_id;
            const uint64_t C = candidate.candidate_target_bytes;
            const uint64_t T = candidate.committed_target_bytes;
            const uint64_t R = candidate.current_bytes;
            optimistic_targets[group_id] = C;
            optimistic_current[group_id] = R;
            const int8_t desired_direction = direction(C, T);
            candidate_directions[group_id] = desired_direction;
            if (complete_phase && desired_direction != 0) {
                next_direction_streaks[group_id] =
                    kResidentDirectionStableComparisons;
            } else if (!resident_profile_candidate_seen[group_id]) {
                all_candidates_seen = false;
                next_direction_streaks[group_id] = 0;
            } else if (
                resident_profile_previous_candidate_directions[group_id] ==
                desired_direction) {
                next_direction_streaks[group_id] =
                    std::min<uint8_t>(
                        std::numeric_limits<uint8_t>::max(),
                        resident_profile_candidate_direction_streaks[group_id] +
                            1);
            } else {
                next_direction_streaks[group_id] = 0;
            }
            if (next_direction_streaks[group_id] >=
                kResidentDirectionStableComparisons) {
                stable_groups[group_id] = 1;
                ++stable_group_count;
            }

            if (resident_profile_candidate_seen[group_id]) {
                const uint64_t previous_C =
                    resident_profile_previous_candidates[group_id];
                if (C > previous_C) {
                    replacement_in += C - previous_C;
                } else {
                    replacement_out += previous_C - C;
                }
            }
            if (C > T) {
                active_delta_in += C - T;
            } else {
                active_delta_out += T - C;
            }
            target_overlap_bytes += std::min(C, T);
            if (candidate.allocated_bytes != 0) {
                expected_local_reference_score_delta +=
                    static_cast<long double>(candidate.weighted_references) *
                    (static_cast<long double>(C) -
                     static_cast<long double>(T)) /
                    static_cast<long double>(candidate.allocated_bytes);
            }
            if (C > R) {
                estimated_promotion_objects +=
                    (C - R) / candidate.footprint_quantum;
            } else {
                estimated_demotion_objects +=
                    (R - C) / candidate.footprint_quantum;
            }

            if (T > R) {
                const bool progressed =
                    R > resident_profile_previous_current[group_id];
                if (progressed || candidate.promotion_opportunities != 0) {
                    next_stall_checks[group_id] = 0;
                } else {
                    next_stall_checks[group_id] = std::min<uint8_t>(
                        std::numeric_limits<uint8_t>::max(),
                        resident_profile_promotion_stall_checks[group_id] + 1);
                }
            } else {
                next_stall_checks[group_id] = 0;
            }
        }
        uint64_t candidate_hash = 1469598103934665603ULL;
        for (uint32_t group_id = 1; group_id < group_limit; ++group_id) {
            candidate_hash ^= group_id;
            candidate_hash *= 1099511628211ULL;
            candidate_hash ^= optimistic_targets[group_id];
            candidate_hash *= 1099511628211ULL;
        }
        const uint64_t replacement_bytes =
            std::max(replacement_in, replacement_out);
        const uint64_t active_delta_bytes =
            std::max(active_delta_in, active_delta_out);
        const uint64_t replacement_protection_bytes =
            resident_local_budget_bytes * kResidentCandidateStablePct / 100;
        const bool whole_candidate_within_protection =
            all_candidates_seen &&
            replacement_bytes <= replacement_protection_bytes;
        const uint64_t candidate_build_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - candidate_build_start)
                .count());
        resident_profile_candidate_build_us_total.fetch_add(
            candidate_build_us, std::memory_order_relaxed);
        resident_profile_update_max(
            resident_profile_candidate_build_us_max, candidate_build_us);
        profile::count_resident_profile_planner_candidate_cycles(
            get_cycles() - candidate_build_start_cycles);
        resident_profile_candidate_stable_group_checks.fetch_add(
            stable_group_count, std::memory_order_relaxed);
        resident_profile_candidate_replacement_bytes_last.store(
            replacement_bytes, std::memory_order_relaxed);
        resident_profile_candidate_replacement_bytes_total.fetch_add(
            replacement_bytes, std::memory_order_relaxed);
        resident_profile_candidate_active_delta_bytes_last.store(
            active_delta_bytes, std::memory_order_relaxed);
        resident_profile_candidate_target_overlap_bytes_last.store(
            target_overlap_bytes, std::memory_order_relaxed);

        const auto save_rejected_snapshot = [&] {
            for (const auto &candidate : candidates) {
                const uint32_t group_id = candidate.group_id;
                resident_profile_previous_candidates[group_id] =
                    optimistic_targets[group_id];
                resident_profile_previous_candidate_directions[group_id] =
                    candidate_directions[group_id];
                resident_profile_candidate_direction_streaks[group_id] =
                    next_direction_streaks[group_id];
                resident_profile_promotion_stall_checks[group_id] =
                    next_stall_checks[group_id];
                resident_profile_previous_current[group_id] =
                    optimistic_current[group_id];
                resident_profile_candidate_seen[group_id] = 1;
            }
        };
        const auto log_candidate = [&](const char *decision,
                                       uint64_t publish_us,
                                       uint64_t authoritative_drift_bytes,
                                       uint64_t participating_groups) {
            std::cerr
                << "resident.group_candidate check_id=" << check_id
                << " check_time_us=" << check_time_us
                << " candidate_hash=" << candidate_hash
                << " groups=" << candidates.size()
                << " stable_groups=" << stable_group_count
                << " participating_groups=" << participating_groups
                << " replacement_bytes=" << replacement_bytes
                << " replacement_in_bytes=" << replacement_in
                << " replacement_out_bytes=" << replacement_out
                << " replacement_protection_bytes="
                << replacement_protection_bytes
                << " whole_candidate_within_protection="
                << whole_candidate_within_protection
                << " active_target_delta_bytes=" << active_delta_bytes
                << " target_overlap_bytes=" << target_overlap_bytes
                << " expected_local_reference_score_delta="
                << static_cast<double>(
                       expected_local_reference_score_delta)
                << " estimated_promotion_objects="
                << estimated_promotion_objects
                << " estimated_demotion_objects="
                << estimated_demotion_objects
                << " authoritative_drift_bytes="
                << authoritative_drift_bytes
                << " profile_collection_us=" << collection_us
                << " candidate_build_us=" << candidate_build_us
                << " budget_early_exit=" << skip_exhausted_donor_scan
                << " publish_us=" << publish_us
                << " dynamic_lifetime_unsupported="
                << dynamic_lifetime_unsupported
                << " decision=" << decision
                << " apply=" << resident_profile_apply_plan_flag
                << std::endl;
        };

        if (dynamic_lifetime_unsupported) {
            resident_profile_candidate_rejected_count.fetch_add(
                1, std::memory_order_relaxed);
            save_rejected_snapshot();
            log_candidate("dynamic_lifetime_unsupported", 0, 0, 0);
            return;
        }

        if (!resident_profile_apply_plan_flag) {
            const uint64_t plan_id =
                resident_profile_plan_count.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            for (const auto &candidate : candidates) {
                resident_placement_groups[candidate.group_id]
                    .resident_target_bytes.store(
                        candidate.candidate_target_bytes,
                        std::memory_order_release);
            }
            for (uint32_t group_id : inactive_group_ids) {
                resident_placement_groups[group_id]
                    .resident_target_bytes.store(
                        0, std::memory_order_release);
            }
            save_rejected_snapshot();
            log_candidate("profile_only", 0, 0, stable_group_count);
            std::cerr << "resident.group_plan id=" << plan_id
                      << " check_id=" << check_id
                      << " profile_only=1 apply=0" << std::endl;
            return;
        }

        if (!complete_phase &&
            check_id <= resident_profile_initial_sample_windows) {
            resident_profile_candidate_rejected_count.fetch_add(
                1, std::memory_order_relaxed);
            save_rejected_snapshot();
            log_candidate("initial_profile", 0, 0, stable_group_count);
            return;
        }

        uint64_t optimistic_participating_groups = 0;
        for (const auto &candidate : candidates) {
            if (stable_groups[candidate.group_id] &&
                candidate_directions[candidate.group_id] != 0) {
                ++optimistic_participating_groups;
            }
        }
        if (optimistic_participating_groups == 0) {
            resident_profile_candidate_rejected_count.fetch_add(
                1, std::memory_order_relaxed);
            save_rejected_snapshot();
            log_candidate("direction_unstable", 0, 0, 0);
            return;
        }

        const auto publish_start = std::chrono::steady_clock::now();
        struct PublishCycleGuard {
            uint64_t start = get_cycles();
            ~PublishCycleGuard() {
                profile::count_resident_profile_planner_publish_cycles(
                    get_cycles() - start);
            }
        } publish_cycle_guard;
        resident_profile_close_migration_gate();
        if (resident_profile_migrations_inflight() != 0) {
            resident_profile_publish_while_unsafe_count.fetch_add(
                1, std::memory_order_relaxed);
        }

        if (region_group_binding_enabled()) {
            ::FarLib::allocator::global_heap
                .get_region_placement_group_snapshots(
                    region_group_snapshots);
        }
        for (auto &candidate : candidates) {
            auto &profile = resident_placement_groups[candidate.group_id];
            if (region_group_binding_enabled()) {
                const uint64_t live_footprint =
                    profile.live_footprint_bytes.load(
                        std::memory_order_acquire);
                candidate.allocated_bytes =
                    ((live_footprint + ::FarLib::allocator::RegionSize - 1) /
                     ::FarLib::allocator::RegionSize) *
                    ::FarLib::allocator::RegionSize;
                candidate.current_bytes =
                    region_group_snapshots[candidate.group_id]
                        .resident_regions *
                    ::FarLib::allocator::RegionSize;
                candidate.allocated_bytes = std::max(
                    candidate.allocated_bytes, candidate.current_bytes);
            } else {
                candidate.allocated_bytes =
                    profile.live_footprint_bytes.load(
                        std::memory_order_acquire);
                candidate.current_bytes =
                    profile.resident_current_bytes.load(
                        std::memory_order_acquire);
            }
            candidate.promotion_opportunities +=
                profile.window_promotion_opportunities.exchange(
                    0, std::memory_order_acq_rel);
            candidate.committed_target_bytes =
                resident_profile_committed_targets[candidate.group_id];
        }
        compute_candidate_targets(candidates);

        uint64_t authoritative_drift_in = 0;
        uint64_t authoritative_drift_out = 0;
        std::vector<int8_t> authoritative_directions(group_limit, 0);
        for (const auto &candidate : candidates) {
            const uint32_t group_id = candidate.group_id;
            const uint64_t authoritative_C =
                candidate.candidate_target_bytes;
            const uint64_t optimistic_C = optimistic_targets[group_id];
            if (authoritative_C > optimistic_C) {
                authoritative_drift_in += authoritative_C - optimistic_C;
            } else {
                authoritative_drift_out += optimistic_C - authoritative_C;
            }
            authoritative_directions[group_id] = direction(
                authoritative_C, candidate.committed_target_bytes);
        }
        const uint64_t authoritative_drift_bytes =
            std::max(authoritative_drift_in, authoritative_drift_out);
        if (authoritative_drift_bytes > replacement_protection_bytes) {
            if (had_plan) {
                resident_profile_open_migration_gate();
            }
            resident_profile_candidate_rejected_count.fetch_add(
                1, std::memory_order_relaxed);
            save_rejected_snapshot();
            const uint64_t publish_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - publish_start)
                    .count());
            resident_profile_publish_us_total.fetch_add(
                publish_us, std::memory_order_relaxed);
            resident_profile_update_max(resident_profile_publish_us_max,
                                        publish_us);
            log_candidate("authoritative_drift", publish_us,
                          authoritative_drift_bytes, 0);
            return;
        }

        std::vector<uint64_t> proposed_targets(
            resident_profile_committed_targets);
        std::vector<uint64_t> publish_targets(
            resident_profile_committed_targets);
        uint64_t participating_groups = 0;
        uint64_t prevented_reversals = 0;
        uint64_t stalled_receiver_groups = 0;
        uint64_t stalled_receiver_pending_bytes = 0;
        uint64_t stalled_neutral_requested_bytes = 0;
        std::vector<uint8_t> forced_neutral_donors(group_limit, 0);
        for (const auto &candidate : candidates) {
            const uint32_t group_id = candidate.group_id;
            if (!stable_groups[group_id] ||
                authoritative_directions[group_id] !=
                    candidate_directions[group_id]) {
                continue;
            }
            const uint64_t T = candidate.committed_target_bytes;
            const uint64_t R = candidate.current_bytes;
            uint64_t proposed = candidate.candidate_target_bytes;
            if (T > R) {
                const bool progressed =
                    R > resident_profile_previous_current[group_id] ||
                    candidate.promotion_opportunities != 0;
                const bool stalled =
                    next_stall_checks[group_id] >=
                        kResidentPromotionStallChecks &&
                    candidate.promotion_opportunities == 0;
                if (stalled) {
                    ++stalled_receiver_groups;
                    stalled_receiver_pending_bytes += T - R;
                }
                if (stalled && !progressed) {
                    // No eligible access has advanced this target for two
                    // complete profile checks. Release the entire unrealized
                    // reservation even if its decayed EMA is still high.
                    proposed = R;
                    stalled_neutral_requested_bytes += T - R;
                    forced_neutral_donors[group_id] = 1;
                    ++prevented_reversals;
                } else {
                    if (proposed < T) {
                        proposed = T;
                        ++prevented_reversals;
                    }
                }
            } else if (T < R && proposed > R) {
                proposed = R;
                ++prevented_reversals;
            }
            proposed_targets[group_id] = proposed;
            if (proposed != T) {
                ++participating_groups;
            }
        }

        for (auto &receiver : candidates) {
            const uint32_t receiver_id = receiver.group_id;
            if (proposed_targets[receiver_id] <=
                publish_targets[receiver_id]) {
                continue;
            }
            for (uint8_t forced_pass = 0;
                 forced_pass < 2 &&
                 publish_targets[receiver_id] <
                     proposed_targets[receiver_id];
                 ++forced_pass) {
                for (auto donor_it = candidates.rbegin();
                     donor_it != candidates.rend() &&
                     publish_targets[receiver_id] <
                         proposed_targets[receiver_id];
                     ++donor_it) {
                    const uint32_t donor_id = donor_it->group_id;
                    const bool forced = forced_neutral_donors[donor_id] != 0;
                    if ((forced_pass == 0) != forced ||
                        donor_id == receiver_id ||
                        publish_targets[donor_id] <=
                            proposed_targets[donor_id]) {
                        continue;
                    }
                    const uint64_t quantum = std::lcm(
                        receiver.footprint_quantum,
                        donor_it->footprint_quantum);
                    const uint64_t limit = std::min(
                        proposed_targets[receiver_id] -
                            publish_targets[receiver_id],
                        publish_targets[donor_id] -
                            proposed_targets[donor_id]);
                    const uint64_t transfer = limit - limit % quantum;
                    if (transfer == 0) {
                        continue;
                    }
                    publish_targets[receiver_id] += transfer;
                    publish_targets[donor_id] -= transfer;
                }
            }
        }

        uint64_t target_change_in = 0;
        uint64_t target_change_out = 0;
        uint64_t target_change_objects = 0;
        uint64_t scheduled_promotion_bytes = 0;
        uint64_t scheduled_demotion_bytes = 0;
        uint64_t scheduled_promotion_objects = 0;
        uint64_t scheduled_demotion_objects = 0;
        long double published_expected_local_reference_score_delta = 0;
        uint64_t cancelled_stalled_promotion_bytes = 0;
        uint64_t cancelled_stalled_promotion_groups = 0;
        uint64_t cancelled_promotion_bytes = 0;
        uint64_t cancelled_demotion_bytes = 0;
        uint64_t carried_promotion_bytes = 0;
        uint64_t carried_demotion_bytes = 0;
        uint64_t assigned = 0;
        bool direction_safe = true;
        for (const auto &candidate : candidates) {
            const uint32_t group_id = candidate.group_id;
            const uint64_t T = candidate.committed_target_bytes;
            const uint64_t R = candidate.current_bytes;
            const uint64_t P = publish_targets[group_id];
            ASSERT(P % candidate.footprint_quantum == 0);
            assigned += P;
            if (P > T) {
                target_change_in += P - T;
                target_change_objects +=
                    (P - T) / candidate.footprint_quantum;
            } else {
                target_change_out += T - P;
                target_change_objects +=
                    (T - P) / candidate.footprint_quantum;
            }
            if (candidate.allocated_bytes != 0) {
                published_expected_local_reference_score_delta +=
                    static_cast<long double>(candidate.weighted_references) *
                    (static_cast<long double>(P) -
                     static_cast<long double>(T)) /
                    static_cast<long double>(candidate.allocated_bytes);
            }
            if (P > R) {
                scheduled_promotion_bytes += P - R;
                scheduled_promotion_objects +=
                    (P - R) / candidate.footprint_quantum;
            } else {
                scheduled_demotion_bytes += R - P;
                scheduled_demotion_objects +=
                    (R - P) / candidate.footprint_quantum;
            }
            if ((T > R && P < R) || (T < R && P > R)) {
                direction_safe = false;
            }
            if (T > R) {
                if (P < T) {
                    cancelled_promotion_bytes += T - P;
                    carried_promotion_bytes += P - R;
                } else {
                    carried_promotion_bytes += T - R;
                }
            } else if (T < R) {
                if (P > T) {
                    cancelled_demotion_bytes += P - T;
                    carried_demotion_bytes += R - P;
                } else {
                    carried_demotion_bytes += R - T;
                }
            }
            if (T > R &&
                next_stall_checks[group_id] >=
                    kResidentPromotionStallChecks &&
                candidate.promotion_opportunities == 0 && P < T) {
                const uint64_t cancelled =
                    std::min(T - R, T - P);
                if (cancelled != 0) {
                    cancelled_stalled_promotion_bytes += cancelled;
                    ++cancelled_stalled_promotion_groups;
                }
            }
        }
        const uint64_t published_replacement_bytes =
            std::max(target_change_in, target_change_out);
        if (!direction_safe || target_change_in != target_change_out) {
            resident_profile_publish_while_unsafe_count.fetch_add(
                1, std::memory_order_relaxed);
        }
        ASSERT(direction_safe);
        ASSERT(target_change_in == target_change_out);
        if (published_replacement_bytes == 0) {
            if (had_plan) {
                resident_profile_open_migration_gate();
            }
            resident_profile_candidate_rejected_count.fetch_add(
                1, std::memory_order_relaxed);
            save_rejected_snapshot();
            const uint64_t publish_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - publish_start)
                    .count());
            resident_profile_publish_us_total.fetch_add(
                publish_us, std::memory_order_relaxed);
            resident_profile_update_max(resident_profile_publish_us_max,
                                        publish_us);
            log_candidate("no_paired_change", publish_us,
                          authoritative_drift_bytes, participating_groups);
            return;
        }

        const long double required_reference_score =
            static_cast<long double>(
                resident_profile_min_benefit_refs_per_replacement_object) *
            static_cast<long double>(target_change_objects);
        if (cancelled_stalled_promotion_bytes == 0 &&
            published_expected_local_reference_score_delta <
                required_reference_score) {
            if (had_plan) {
                resident_profile_open_migration_gate();
            }
            resident_profile_candidate_rejected_count.fetch_add(
                1, std::memory_order_relaxed);
            resident_profile_benefit_rejected_count.fetch_add(
                1, std::memory_order_relaxed);
            save_rejected_snapshot();
            const uint64_t publish_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - publish_start)
                    .count());
            resident_profile_publish_us_total.fetch_add(
                publish_us, std::memory_order_relaxed);
            resident_profile_update_max(resident_profile_publish_us_max,
                                        publish_us);
            log_candidate("benefit_below_threshold", publish_us,
                          authoritative_drift_bytes,
                          participating_groups);
            std::cerr
                << "resident.group_plan_rejected check_id=" << check_id
                << " reason=benefit_below_threshold"
                << " replacement_bytes=" << published_replacement_bytes
                << " replacement_objects=" << target_change_objects
                << " expected_local_reference_score_delta="
                << static_cast<double>(
                       published_expected_local_reference_score_delta)
                << " min_benefit_refs_per_replacement_object="
                << resident_profile_min_benefit_refs_per_replacement_object
                << " cancelled_stalled_promotion_bytes="
                << cancelled_stalled_promotion_bytes
                << " publish_us=" << publish_us << std::endl;
            return;
        }

        resident_profile_finalize_active_plan(
            check_id, cancelled_promotion_bytes,
            cancelled_demotion_bytes, carried_promotion_bytes,
            carried_demotion_bytes);
        const uint64_t plan_id =
            resident_profile_plan_count.fetch_add(
                1, std::memory_order_relaxed) +
            1;
        for (const auto &candidate : candidates) {
            const uint32_t group_id = candidate.group_id;
            resident_profile_committed_targets[group_id] =
                publish_targets[group_id];
            resident_placement_groups[group_id]
                .resident_target_bytes.store(
                    publish_targets[group_id],
                    std::memory_order_release);
        }
        for (uint32_t group_id : inactive_group_ids) {
            resident_profile_committed_targets[group_id] = 0;
            resident_placement_groups[group_id]
                .resident_target_bytes.store(0,
                                             std::memory_order_release);
        }
        resident_profile_stalled_promotion_cancelled_bytes.fetch_add(
            cancelled_stalled_promotion_bytes,
            std::memory_order_relaxed);
        resident_profile_stalled_promotion_cancelled_groups.fetch_add(
            cancelled_stalled_promotion_groups,
            std::memory_order_relaxed);
        resident_profile_prevented_reversal_count.fetch_add(
            prevented_reversals, std::memory_order_relaxed);
        resident_profile_begin_active_plan(
            plan_id, check_id, scheduled_promotion_bytes,
            scheduled_demotion_bytes);
        if (region_group_binding_enabled()) {
            apply_region_resident_targets(group_limit);
        }
        const uint64_t published_id =
            resident_profile_published_plan_count.fetch_add(
                1, std::memory_order_relaxed) +
            1;
        resident_profile_open_migration_gate();

        for (const auto &candidate : candidates) {
            const uint32_t group_id = candidate.group_id;
            resident_profile_previous_candidates[group_id] =
                candidate.candidate_target_bytes;
            resident_profile_previous_candidate_directions[group_id] =
                direction(candidate.candidate_target_bytes,
                          publish_targets[group_id]);
            resident_profile_candidate_direction_streaks[group_id] = 0;
            resident_profile_promotion_stall_checks[group_id] = 0;
            resident_profile_previous_current[group_id] =
                candidate.current_bytes;
            resident_profile_candidate_seen[group_id] = 1;
        }
        const uint64_t publish_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - publish_start)
                .count());
        resident_profile_publish_us_total.fetch_add(
            publish_us, std::memory_order_relaxed);
        resident_profile_update_max(resident_profile_publish_us_max,
                                    publish_us);
        log_candidate("published", publish_us,
                      authoritative_drift_bytes, participating_groups);
        std::cerr
            << "resident.group_plan_published published_id=" << published_id
            << " plan_id=" << plan_id
            << " check_id=" << check_id
            << " publish_time_us="
            << resident_profile_active_plan_publish_time_us.load()
            << " groups=" << candidates.size()
            << " assigned=" << assigned
            << " replacement_bytes=" << published_replacement_bytes
            << " replacement_objects=" << target_change_objects
            << " expected_local_reference_score_delta="
            << static_cast<double>(
                   published_expected_local_reference_score_delta)
            << " scheduled_promotion_bytes="
            << scheduled_promotion_bytes
            << " scheduled_promotion_objects="
            << scheduled_promotion_objects
            << " scheduled_demotion_bytes="
            << scheduled_demotion_bytes
            << " scheduled_demotion_objects="
            << scheduled_demotion_objects
            << " cancelled_stalled_promotion_bytes="
            << cancelled_stalled_promotion_bytes
            << " cancelled_stalled_promotion_groups="
            << cancelled_stalled_promotion_groups
            << " prevented_reversals=" << prevented_reversals
            << " stalled_receiver_groups=" << stalled_receiver_groups
            << " stalled_receiver_pending_bytes="
            << stalled_receiver_pending_bytes
            << " stalled_neutral_requested_bytes="
            << stalled_neutral_requested_bytes
            << " publish_while_unsafe="
            << resident_profile_publish_while_unsafe_count.load()
            << " plan_us=" << publish_us << std::endl;
        if (scheduled_demotion_bytes != 0) {
            eviction_request_seq.fetch_add(1,
                                           std::memory_order_acq_rel);
            uthread::notify_all(&eviction_cond,
                                &eviction_mutex);
        }
    }

    void run_resident_profile_planner() {
        ::FarLib::allocator::
            disable_region_list_fibre_yield_for_current_thread();
        while (working.load(std::memory_order_acquire)) {
            while (working.load(std::memory_order_acquire) &&
                   !resident_profile_work_phase_active()) {
                std::unique_lock<std::mutex> lock(resident_profile_mutex);
                resident_profile_cond.wait_for(
                    lock, std::chrono::milliseconds(10), [this] {
                        return !working.load(std::memory_order_acquire);
                    });
            }
            if (!working.load(std::memory_order_acquire)) {
                return;
            }

            const auto warmup_deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(resident_profile_warmup_ms);
            while (working.load(std::memory_order_acquire) &&
                   resident_profile_work_phase_active() &&
                   std::chrono::steady_clock::now() < warmup_deadline) {
                const auto remaining = warmup_deadline -
                    std::chrono::steady_clock::now();
                std::unique_lock<std::mutex> lock(resident_profile_mutex);
                resident_profile_cond.wait_for(
                    lock,
                    std::min<std::chrono::steady_clock::duration>(
                        remaining, std::chrono::milliseconds(10)),
                    [this] {
                        return !working.load(std::memory_order_acquire);
                    });
            }
            if (!resident_profile_work_phase_active()) {
                continue;
            }

            while (working.load(std::memory_order_acquire) &&
                   resident_profile_work_phase_active()) {
                publish_resident_group_plan();
                if (resident_profile_stop_after_first_commit() &&
                    resident_profile_promoted_bytes.load(
                        std::memory_order_acquire) != 0) {
                    resident_profile_active.store(false,
                                                  std::memory_order_release);
                    std::cerr
                        << "resident.profile.planner_stopped"
                        << " reason=first_committed_region_exchange"
                        << " published_plans="
                        << resident_profile_published_plan_count.load(
                               std::memory_order_relaxed)
                        << " promoted_bytes="
                        << resident_profile_promoted_bytes.load(
                               std::memory_order_relaxed)
                        << std::endl;
                    return;
                }
                std::unique_lock<std::mutex> lock(resident_profile_mutex);
                resident_profile_cond.wait_for(
                    lock,
                    std::chrono::milliseconds(resident_profile_interval_ms),
                    [this] {
                        return !working.load(std::memory_order_acquire) ||
                               !resident_profile_work_phase_active();
                    });
            }
        }
    }

    void stop_resident_profile_planner() {
        resident_profile_active.store(false, std::memory_order_release);
        resident_profile_cond.notify_all();
        if (resident_profile_thread.joinable()) {
            resident_profile_thread.join();
        }
    }

    std::atomic<uint64_t> *resident_profile_current_counter(
        const FarObjectEntry &entry) {
        // Whole-region placement is the source of truth in Region-hotness
        // mode. A physical R<->S exchange changes many entries together and
        // deliberately does not maintain the old object/group current-byte
        // accounting used by object-level migration.
        if (region_hotness_placement_enabled()) {
            return nullptr;
        }
        if (automatic_resident_group_planner_enabled()) {
            const uint32_t group_id = entry.resident_group_id();
            if (group_id == 0) {
                return nullptr;
            }
            ASSERT(group_id < next_resident_placement_group_id.load(
                                      std::memory_order_acquire));
            return &resident_placement_groups[group_id]
                        .resident_current_bytes;
        }
        const uint32_t owner_id = entry.logical_owner_id();
        if (owner_id == 0) {
            return nullptr;
        }
        ASSERT(owner_id < logical_object_profiles.size());
        return &logical_object_profiles[owner_id].resident_current_bytes;
    }

    std::atomic<uint64_t> *resident_profile_target_counter(
        const FarObjectEntry &entry) {
        if (automatic_resident_group_planner_enabled()) {
            const uint32_t group_id = entry.resident_group_id();
            if (group_id == 0) {
                return nullptr;
            }
            ASSERT(group_id < next_resident_placement_group_id.load(
                                      std::memory_order_acquire));
            return &resident_placement_groups[group_id]
                        .resident_target_bytes;
        }
        const uint32_t owner_id = entry.logical_owner_id();
        if (owner_id == 0) {
            return nullptr;
        }
        ASSERT(owner_id < logical_object_profiles.size());
        return &logical_object_profiles[owner_id].resident_target_bytes;
    }

    void release_profiled_resident_bytes(const FarObjectEntry &entry,
                                         size_t bytes) {
        auto *current_counter = resident_profile_current_counter(entry);
        if (current_counter == nullptr) {
            if (!region_hotness_placement_enabled()) {
                return;
            }
            const uint32_t group_id = entry.resident_group_id();
            if (group_id == 0) {
                const uint32_t owner_id = entry.logical_owner_id();
                if (owner_id != 0) {
                    ASSERT(owner_id < logical_object_profiles.size());
                    const uint64_t owner_previous =
                        logical_object_profiles[owner_id]
                            .resident_current_bytes.fetch_sub(
                                bytes, std::memory_order_acq_rel);
                    ASSERT(owner_previous >= bytes);
                }
                return;
            }
            ASSERT(group_id < next_resident_placement_group_id.load(
                                  std::memory_order_acquire));
            auto &group_counter =
                resident_placement_groups[group_id].resident_current_bytes;
            const uint64_t group_previous =
                group_counter.fetch_sub(bytes, std::memory_order_acq_rel);
            ASSERT(group_previous >= bytes);
            const uint32_t owner_id = entry.logical_owner_id();
            if (owner_id != 0) {
                ASSERT(owner_id < logical_object_profiles.size());
                const uint64_t owner_previous =
                    logical_object_profiles[owner_id]
                        .resident_current_bytes.fetch_sub(
                            bytes, std::memory_order_acq_rel);
                ASSERT(owner_previous >= bytes);
            }
            return;
        }
        const uint64_t previous = current_counter->fetch_sub(
            bytes, std::memory_order_acq_rel);
        ASSERT(previous >= bytes);
        if (automatic_resident_group_planner_enabled()) {
            const uint32_t owner_id = entry.logical_owner_id();
            if (owner_id != 0) {
                ASSERT(owner_id < logical_object_profiles.size());
                const uint64_t owner_previous =
                    logical_object_profiles[owner_id]
                        .resident_current_bytes.fetch_sub(
                            bytes, std::memory_order_acq_rel);
                ASSERT(owner_previous >= bytes);
            }
        }
    }

    void add_profiled_resident_bytes(const FarObjectEntry &entry,
                                     size_t bytes) {
        auto *current_counter = resident_profile_current_counter(entry);
        if (current_counter == nullptr) {
            if (!region_hotness_placement_enabled()) {
                return;
            }
            const uint32_t group_id = entry.resident_group_id();
            if (group_id == 0) {
                const uint32_t owner_id = entry.logical_owner_id();
                if (owner_id != 0) {
                    ASSERT(owner_id < logical_object_profiles.size());
                    logical_object_profiles[owner_id]
                        .resident_current_bytes.fetch_add(
                            bytes, std::memory_order_relaxed);
                }
                return;
            }
            ASSERT(group_id < next_resident_placement_group_id.load(
                                  std::memory_order_acquire));
            resident_placement_groups[group_id]
                .resident_current_bytes.fetch_add(
                    bytes, std::memory_order_relaxed);
            const uint32_t owner_id = entry.logical_owner_id();
            if (owner_id != 0) {
                ASSERT(owner_id < logical_object_profiles.size());
                logical_object_profiles[owner_id]
                    .resident_current_bytes.fetch_add(
                        bytes, std::memory_order_relaxed);
            }
            return;
        }
        current_counter->fetch_add(bytes, std::memory_order_relaxed);
        if (automatic_resident_group_planner_enabled()) {
            const uint32_t owner_id = entry.logical_owner_id();
            if (owner_id != 0) {
                ASSERT(owner_id < logical_object_profiles.size());
                logical_object_profiles[owner_id]
                    .resident_current_bytes.fetch_add(
                        bytes, std::memory_order_relaxed);
            }
        }
    }

    void update_owner_resident_bytes_after_group_migration(
        const FarObjectEntry &entry, size_t bytes, bool promote) {
        if (!automatic_resident_group_planner_enabled()) {
            return;
        }
        const uint32_t owner_id = entry.logical_owner_id();
        if (owner_id == 0) {
            return;
        }
        ASSERT(owner_id < logical_object_profiles.size());
        auto &counter =
            logical_object_profiles[owner_id].resident_current_bytes;
        if (promote) {
            counter.fetch_add(bytes, std::memory_order_relaxed);
        } else {
            const uint64_t previous =
                counter.fetch_sub(bytes, std::memory_order_acq_rel);
            ASSERT(previous >= bytes);
        }
    }

    bool try_demote_profiled_resident(FarObjectEntry &entry, size_t size) {
        // EC batch objects are recovered and rewritten as whole groups.  The
        // Resident planner mutates per-entry placement flags, which is not a
        // valid operation while an EC group may be in-flight or being
        // reconstructed.  Keep this hard guard at the operation boundary so
        // callers such as the eviction marker cannot bypass the reference
        // recording guard above.
        if (!resident_profile_planner_enabled_flag || ec_batch_mode_) {
            return false;
        }
        if (!resident_profile_plan_ready.load(std::memory_order_acquire) ||
            !entry.is_resident_local()) {
            return false;
        }
        bool demotion_success = false;
        struct DemotionCycleGuard {
            uint64_t start = get_cycles();
            bool &success;
            ~DemotionCycleGuard() {
                profile::count_resident_profile_demotion(
                    get_cycles() - start, success);
            }
        } demotion_cycle_guard{get_cycles(), demotion_success};
        auto *current_counter = resident_profile_current_counter(entry);
        auto *target_counter = resident_profile_target_counter(entry);
        if (current_counter == nullptr || target_counter == nullptr) {
            return false;
        }
        const uint64_t bytes = local_allocation_footprint(size);
        if (current_counter->load(std::memory_order_relaxed) <=
            target_counter->load(std::memory_order_acquire)) {
            return false;
        }
        ResidentProfileMigrationInflightGuard inflight_guard(
            resident_profile_migration_gate);
        if (!inflight_guard ||
            !resident_profile_plan_ready.load(std::memory_order_acquire)) {
            return false;
        }
        const uint64_t plan_id = resident_profile_active_plan_id.load(
            std::memory_order_acquire);
        const uint64_t target =
            target_counter->load(std::memory_order_acquire);
        uint64_t current =
            current_counter->load(std::memory_order_relaxed);
        while (current > target) {
            ASSERT(current >= bytes);
            if (current_counter->compare_exchange_weak(
                    current, current - bytes, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                break;
            }
        }
        if (current <= target) {
            return false;
        }
        resident_profile_record_migration_started(plan_id, bytes, false);

        if (!::FarLib::get_config().exclusive_cache) {
            const auto state = entry.load_state();
            if (state.state != LOCAL || !entry.is_resident_local()) {
                current_counter->fetch_add(bytes, std::memory_order_relaxed);
                resident_profile_record_migration_rolled_back(
                    plan_id, bytes, false);
                return false;
            }
            entry.set_resident_local(false);
            update_owner_resident_bytes_after_group_migration(
                entry, bytes, false);
            release_resident_local(bytes);
            resident_profile_demoted_bytes.fetch_add(
                bytes, std::memory_order_relaxed);
            resident_profile_record_migration_committed(
                plan_id, bytes, false);
            demotion_success = true;
            return true;
        }

        auto old_state = entry.load_state();
        if (old_state.state != LOCAL || old_state.invalid ||
            !entry.is_resident_local()) {
            current_counter->fetch_add(bytes, std::memory_order_relaxed);
            resident_profile_record_migration_rolled_back(
                plan_id, bytes, false);
            return false;
        }
        auto lock_state = old_state;
        lock_state.invalid = 1;
        if (!entry.cas_state_weak(old_state, lock_state)) {
            current_counter->fetch_add(bytes, std::memory_order_relaxed);
            resident_profile_record_migration_rolled_back(
                plan_id, bytes, false);
            return false;
        }
        entry.set_resident_local(false);
        auto final_state = lock_state;
        final_state.invalid = 0;
        auto expected = lock_state;
        ASSERT(entry.cas_state_strong(expected, final_state));
        update_owner_resident_bytes_after_group_migration(entry, bytes,
                                                          false);
        release_resident_local(bytes);
        resident_profile_demoted_bytes.fetch_add(bytes,
                                                  std::memory_order_relaxed);
        resident_profile_record_migration_committed(plan_id, bytes, false);
        demotion_success = true;
        return true;
    }

    bool try_promote_profiled_resident(FarObjectEntry &entry, size_t size) {
        // See try_demote_profiled_resident(): EC batch recovery owns the
        // object's placement lifecycle.  This guard covers the direct call
        // from the eviction marker as well as profiled reference sampling.
        if (!resident_profile_planner_enabled_flag || ec_batch_mode_) {
            return false;
        }
        if (!resident_profile_plan_ready.load(std::memory_order_acquire) ||
            entry.is_resident_local()) {
            return false;
        }
        bool promotion_success = false;
        struct PromotionCycleGuard {
            uint64_t start = get_cycles();
            bool &success;
            ~PromotionCycleGuard() {
                profile::count_resident_profile_promotion(
                    get_cycles() - start, success);
            }
        } promotion_cycle_guard{get_cycles(), promotion_success};
        auto *current_counter = resident_profile_current_counter(entry);
        auto *target_counter = resident_profile_target_counter(entry);
        if (current_counter == nullptr || target_counter == nullptr) {
            return false;
        }
        const uint64_t bytes = local_allocation_footprint(size);
        const uint64_t preliminary_target =
            target_counter->load(std::memory_order_acquire);
        if (preliminary_target < bytes ||
            current_counter->load(std::memory_order_relaxed) >
                preliminary_target - bytes) {
            return false;
        }
        ResidentProfileMigrationInflightGuard inflight_guard(
            resident_profile_migration_gate);
        if (!inflight_guard ||
            !resident_profile_plan_ready.load(std::memory_order_acquire)) {
            return false;
        }
        const uint64_t plan_id = resident_profile_active_plan_id.load(
            std::memory_order_acquire);
        const uint64_t target =
            target_counter->load(std::memory_order_acquire);
        if (target < bytes) {
            return false;
        }
        uint64_t current =
            current_counter->load(std::memory_order_relaxed);
        if (current > target - bytes) {
            return false;
        }
        const uint32_t opportunity_group_id = entry.resident_group_id();
        if (opportunity_group_id != 0) {
            ASSERT(opportunity_group_id <
                   next_resident_placement_group_id.load(
                       std::memory_order_acquire));
            resident_placement_groups[opportunity_group_id]
                .window_promotion_opportunities.fetch_add(
                    1, std::memory_order_relaxed);
        }
        while (current <= target - bytes) {
            if (current_counter->compare_exchange_weak(
                    current, current + bytes, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                break;
            }
        }
        if (current > target - bytes) {
            return false;
        }
        resident_profile_record_migration_started(plan_id, bytes, true);
        if (!try_reserve_resident_local(bytes)) {
            current_counter->fetch_sub(bytes, std::memory_order_relaxed);
            resident_profile_record_migration_rolled_back(
                plan_id, bytes, true);
            return false;
        }

        if (!::FarLib::get_config().exclusive_cache) {
            const auto state = entry.load_state();
            if (state.state != LOCAL || entry.is_resident_local()) {
                current_counter->fetch_sub(bytes, std::memory_order_relaxed);
                release_resident_local(bytes);
                resident_profile_record_migration_rolled_back(
                    plan_id, bytes, true);
                return false;
            }
            ASSERT(entry.has_remote());
            entry.set_resident_local(true);
            update_owner_resident_bytes_after_group_migration(
                entry, bytes, true);
            ASSERT(entry.has_remote());
            resident_profile_promoted_bytes.fetch_add(
                bytes, std::memory_order_relaxed);
            resident_profile_record_migration_committed(
                plan_id, bytes, true);
            promotion_success = true;
            return true;
        }

        auto old_state = entry.load_state();
        if (old_state.state != LOCAL || old_state.invalid ||
            entry.is_resident_local()) {
            current_counter->fetch_sub(bytes, std::memory_order_relaxed);
            release_resident_local(bytes);
            resident_profile_record_migration_rolled_back(
                plan_id, bytes, true);
            return false;
        }
        auto lock_state = old_state;
        lock_state.invalid = 1;
        if (!entry.cas_state_weak(old_state, lock_state)) {
            current_counter->fetch_sub(bytes, std::memory_order_relaxed);
            release_resident_local(bytes);
            resident_profile_record_migration_rolled_back(
                plan_id, bytes, true);
            return false;
        }

        if (entry.has_remote()) {
            const uint64_t remote_addr = entry.remote_addr();
            const bool had_backup = entry.has_remote_backup_reservation();
            entry.set_remote_backup_reservation(false);
            entry.set_remote_invalid();
            if (had_backup) {
                release_remote_backup_budget(size);
                resident_profile_released_backup_bytes.fetch_add(
                    size, std::memory_order_relaxed);
            }
            remote_allocator.deallocate(remote_addr);
        }
        entry.set_resident_local(true);
        auto final_state = lock_state;
        final_state.invalid = 0;
        auto expected = lock_state;
        if (!entry.cas_state_strong(expected, final_state)) {
            std::cerr << "resident.promote unlock_race"
                      << " lock_state=" << static_cast<uint32_t>(lock_state.state)
                      << " lock_invalid=" << static_cast<uint32_t>(lock_state.invalid)
                      << " lock_dirty=" << static_cast<uint32_t>(lock_state.dirty)
                      << " lock_hotness=" << static_cast<uint32_t>(lock_state.hotness)
                      << " lock_ref_cnt=" << static_cast<uint32_t>(lock_state.ref_cnt)
                      << " observed_state=" << static_cast<uint32_t>(expected.state)
                      << " observed_invalid=" << static_cast<uint32_t>(expected.invalid)
                      << " observed_dirty=" << static_cast<uint32_t>(expected.dirty)
                      << " observed_hotness=" << static_cast<uint32_t>(expected.hotness)
                      << " observed_ref_cnt=" << static_cast<uint32_t>(expected.ref_cnt)
                      << std::endl;
            ASSERT(false);
        }
        update_owner_resident_bytes_after_group_migration(entry, bytes, true);
        resident_profile_promoted_bytes.fetch_add(bytes,
                                                   std::memory_order_relaxed);
        resident_profile_record_migration_committed(plan_id, bytes, true);
        promotion_success = true;
        return true;
    }

    void release_remote_backup_budget(size_t bytes) {
        if (batched_backup_budget) {
            batched_backup_budget->release(current_backup_thread_handle(), bytes);
            return;
        }
        if (bypass_backup_usage) {
            bypass_backup_usage->adjust(-static_cast<int64_t>(bytes));
            return;
        }
        uint64_t previous =
            retained_backup_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
        ASSERT(previous >= bytes);
    }

    bool invalidate_retained_backup_for_write(FarObjectEntry &entry,
                                               size_t bytes) {
        if (!selective_backup_enabled() ||
            !entry.has_remote_backup_reservation()) {
            return false;
        }
    retry:
        auto old_state = entry.load_state();
        if (old_state.invalid) [[unlikely]] {
            profile::count_excl_move_lock_spin();
            uthread::yield();
            goto retry;
        }
        if (old_state.dirty || !entry.has_remote() ||
            old_state.state == FETCHING || old_state.state == EVICTING ||
            old_state.state == REMOTE || old_state.state == FREE ||
            old_state.state == BUSY) {
            return false;
        }

        auto lock_state = old_state;
        lock_state.invalid = 1;
        if (!entry.cas_state_weak(old_state, lock_state)) [[unlikely]] {
            goto retry;
        }

        uint64_t old_remote = entry.remote_addr();
        ASSERT(old_remote != FarObjectEntry::RemoteAddrInvalid48);
        entry.set_remote_invalid();
        entry.set_remote_backup_reservation(false);

        auto final_state = lock_state;
        final_state.invalid = 0;
        final_state.dirty = 1;
        if (final_state.state == MARKED) {
            final_state.state = LOCAL;
        }
        auto expected = lock_state;
        ASSERT(entry.cas_state_strong(expected, final_state));

        release_remote_backup_budget(bytes);
        remote_allocator.deallocate(old_remote);
        profile::count_remote_backup_invalidated(bytes);
        return true;
    }

public:

    uint64_t allocate_remote(size_t size, uint32_t behavior_group = 0) {
        return remote_allocator.allocate(size, behavior_group);
    }

    static bool six_dirty_routing_enabled();

    static uint32_t initial_simple_behavior_group(bool hot = false);

    static uint32_t current_behavior_group(const FarObjectEntry& entry);

    static void trace_object_event_locked(const FarObjectEntry& entry,
                                         ::FarLib::object_group_trace::Kind kind);

    static void trace_object_event(const FarObjectEntry& entry,
                                  ::FarLib::object_group_trace::Kind kind);

    static void trace_object_allocation(FarObjectEntry& entry, size_t size);

    static void six_bind_local(FarObjectEntry& entry, void* local, size_t size);

    static void six_commit_fetch(FarObjectEntry& entry, void* local, size_t size);

    static void six_commit_evict(FarObjectEntry& entry, size_t size, bool dirty);

    static void six_free(FarObjectEntry& entry, size_t size);

    static constexpr size_t EvictBatchSize = 64;
    bool all_nonresident_backup_enabled() const {
        return all_nonresident_backup_mode;
    }
    int64_t backup_usage_snapshot() const {
        if (batched_backup_budget) return batched_backup_budget->snapshot().actual_bytes;
        return bypass_backup_usage ? bypass_backup_usage->snapshot()
            : static_cast<int64_t>(retained_backup_bytes.load());
    }
    uint64_t backup_peak_snapshot() const {
        if (batched_backup_budget) return batched_backup_budget->snapshot().peak_issued_bytes;
        return bypass_backup_usage ? static_cast<uint64_t>(bypass_backup_usage->sampled_peak())
            : peak_retained_backup_bytes.load();
    }
    void print_nonresident_backup_snapshot(const char *phase) const {
        if (!all_nonresident_backup_mode) return;
        const auto credits = batched_backup_budget ? batched_backup_budget->snapshot()
                                                  : BatchedBackupBudget::Snapshot{};
        const int64_t observed = batched_backup_budget ? credits.actual_bytes : backup_usage_snapshot();
        std::cout << "nonresident_backup_snapshot phase=" << phase
                  << " retained_bytes=" << observed
                  << " peak_retained_bytes=" << (batched_backup_budget ? credits.peak_issued_bytes : backup_peak_snapshot())
                  << " counter_bypass=" << static_cast<bool>(bypass_backup_usage)
                  << " batched_credits=" << static_cast<bool>(batched_backup_budget)
                  << " budget_enforced=" << !bypass_backup_usage
                  << " peak_kind=" << (batched_backup_budget ? "issued_upper_bound" : (bypass_backup_usage ? "sampled_only" : "per_update"))
                  << " issued_bytes=" << credits.issued_bytes
                  << " cached_credit_bytes=" << credits.cached_bytes
                  << " credit_acquire_count=" << credits.acquire_count
                  << " credit_release_count=" << credits.release_count
                  << " credit_grant_count=" << credits.grant_count
                  << " credit_refund_count=" << credits.refund_count
                  << " credit_grant_cas_attempts=" << credits.grant_cas_attempts
                  << " credit_peak_cas_attempts=" << credits.peak_cas_attempts
                  << " credit_pressure_scans=" << credits.pressure_scans
                  << " credit_handles=" << credits.handles
                  << " shared_counter_bytes=" << retained_backup_bytes.load()
                  << " shared_peak_bytes=" << peak_retained_backup_bytes.load()
                  << " observer_writers=" << (bypass_backup_usage ? bypass_backup_usage->writers() : 0)
                  << " resident_region_bytes="
                  << ::FarLib::allocator::global_heap.get_resident_reserved_regions() * ::FarLib::allocator::RegionSize
                  << " streaming_region_bytes="
                  << ::FarLib::allocator::global_heap.get_streaming_reserved_regions() * ::FarLib::allocator::RegionSize
                  << " budget_full_rejects=" << profiled_backup_budget_full_reject_count.load()
                  << " interrupted_keep_count=" << nonresident_interrupted_keep_count.load()
                  << " interrupted_keep_bytes=" << nonresident_interrupted_keep_bytes.load()
                  << std::endl;
    }
    
    // Endpoint-aware EvictBuffer: partition by endpoint
    struct EvictBuffer {
        struct EvictRequest {
            void *local_addr;
            uint64_t remote_addr : 48;
            uint64_t size : 16;
        };
        EvictRequest requests[EvictBatchSize];
        size_t count = 0;
    };

    // EC (ft_method=sponge) commit RPC staging: one batch per endpoint, so a
    // commit batch is flushed together with the RDMA-write batch of the same
    // endpoint.  Only allocated when ft_method != none.
    struct SpongeEvictBuffer {
        rdma::SpongeCommitBatchBuffer batch;
        size_t committed_bytes = 0;

        void reset() {
            batch.reset();
            committed_bytes = 0;
        }

        bool empty() const { return batch.header.record_count == 0; }
    };

    struct EvictThreadWorkGuard {
        EvictThreadWorkGuard() { profile::evac_thread_work_begin(); }
        ~EvictThreadWorkGuard() { profile::evac_thread_work_end(); }
    };
    
    // Per-endpoint EvictBuffer array
    struct EvictBufferSet {
        std::unique_ptr<EvictBuffer[]> buffers;
        std::unique_ptr<SpongeEvictBuffer[]> sponge_buffers;
        size_t server_count;
        
        EvictBufferSet() : buffers(nullptr), server_count(0) {}
        
        void init(size_t count) {
            server_count = count;
            buffers.reset(new EvictBuffer[server_count]);
            if (::FarLib::get_config().ft_enabled()) {
                sponge_buffers.reset(new SpongeEvictBuffer[server_count]);
            }
        }
        
        EvictBuffer& get_buffer(uint64_t remote_addr) {
            if (server_count <= 1) {
                return buffers[0];
            }
            auto& config = ::FarLib::get_config();
            auto [endpoint, _] = config.map_remote_addr(remote_addr);
            return buffers[endpoint];
        }
        
        void flush_all(size_t client_idx, ConcurrentArrayCache* cache) {
            for (size_t i = 0; i < server_count; i++) {
                if (buffers[i].count > 0) {
                    cache->post_write_requests(buffers[i], client_idx, i);
                }
                if (sponge_buffers && !sponge_buffers[i].empty()) {
                    cache->post_sponge_commit_requests(sponge_buffers[i],
                                                       client_idx, i);
                }
            }
            if (::FarLib::get_config().is_ec_batch_mode()) {
                // EC (ft_method=ec_batch): the end of an eviction batch is the
                // flush point of the group path - seal the partly filled group
                // and post every sealed group, so no object of this batch is
                // left behind in EVICTING.
                cache->flush_ec_batch_groups(client_idx);
            }
        }

        size_t pending_count() const {
            size_t total = 0;
            for (size_t i = 0; i < server_count; i++) {
                total += buffers[i].count;
                if (sponge_buffers) {
                    total += sponge_buffers[i].batch.header.record_count;
                }
            }
            return total;
        }
    };

    // Ordinary READ wr_id carries a per-block generation token; writes keep
    // the historical local-address contract.
    template <bool IsReadRequest>
    void post_rdma_request(far_obj_t obj, size_t client_idx,
                           bool sync_batch_eligible = false) {
        auto &entry = get_entry_of(obj);
        uint64_t remote_addr = entry.remote_addr();
        assert(remote_addr != FarObjectEntry::RemoteAddrInvalid48);
        void *local_addr = entry.local_addr();
        auto block = static_cast<::FarLib::allocator::BlockHead *>(local_addr) - 1;
        assert(block->obj_meta_data == obj);
        uint64_t wr_id = reinterpret_cast<uint64_t>(local_addr);
        auto *client = rdma::get_client(client_idx);
        size_t qp_idx = client->get_qp_idx();
        const auto& config = ::FarLib::get_config();
        size_t endpoint_idx = 0;
        size_t offset = remote_addr;
        if (config.server_count > 1) {
            auto mapped = config.map_remote_addr(remote_addr);
            endpoint_idx = mapped.first;
            offset = mapped.second;
        }
        // EC read-side recovery (ft_method=ec_batch): the endpoint this read
        // would go to is known to be gone, so the read can never complete
        // normally - serve it from the surviving segments of the object's slot
        // group instead.  The entry stays FETCHING and the caller's wait
        // (check_fetch / fetch_wait_until_local) keeps its semantics; when the
        // degraded path cannot help, the legacy post below is taken unchanged.
        // Gated and inert: with ft_method=none no endpoint is ever marked dead.
        if constexpr (IsReadRequest) {
            if (config.server_count > 1 &&
                ec_recovery_endpoint_is_dead(endpoint_idx)) {
                const auto recovery =
                    post_ec_degraded_read(obj, client_idx, local_addr);
                if (recovery !=
                    ec_read_recovery::EcReadPostResult::kUnavailable) {
                    ec_recovery_routed_reads_.fetch_add(
                        1, std::memory_order_relaxed);
                    return;
                }
                // Recovery closes the per-block ordinary-READ gate before it
                // tries to acquire its token.  Do not fall back to posting a
                // new READ to the dead endpoint when the degraded round is
                // temporarily unavailable; the fetch wait will retry it.
                if (::FarLib::allocator::normal_read_recovery_active(block)) {
                    return;
                }
            }

            // Pin that allocator block before it can be queued or posted: an
            // endpoint failure may make the completion arrive much later,
            // after the fetch has otherwise become reclaimable.  Overlapping
            // requests in one FETCHING epoch share the same generation; the
            // helper serializes the first-pin transition so no legitimate
            // completion is rejected as stale.
            const uint16_t generation =
                ::FarLib::allocator::acquire_normal_read_pin(block);
            // A recovery round may have closed the gate after the endpoint
            // liveness check above.  Leave the entry FETCHING so its wait
            // path can retry recovery; never post a normal READ in this case.
            if (generation == 0) {
                return;
            }
            wr_id = ::FarLib::allocator::encode_normal_read_wr_id(
                local_addr, generation);
            wc_object_diag::prepost(obj.obj_id, wr_id);
        }
        const bool use_read_batch = IsReadRequest && sync_batch_eligible &&
                                    client->sync_read_batching_enabled();
        request_interval_diag::Token request_interval_token{};
        if constexpr (IsReadRequest) {
            if (!use_read_batch) {
                client->record_immediate_read_generated(obj.size);
                auto *diag_slot = scope_diag::current(fibre_self());
                if (diag_slot != nullptr) {
                    request_interval_token =
                        request_interval_diag::request_prepost(
                            scope_diag::index(diag_slot),
                            reinterpret_cast<uint64_t>(fibre_self()),
                            reinterpret_cast<uint64_t>(&entry), wr_id,
                            mutator_waiters.load(std::memory_order_relaxed));
                }
            }
        }
        
        // Validate mapping invariant I2: endpoint_offset + size <= server_buffer_size
        if (config.server_count > 1 && !config.validate_mapping(remote_addr, obj.size)) {
            ERROR(("Invalid mapping: remote_addr=" + std::to_string(remote_addr) + 
                  " size=" + std::to_string(obj.size)).c_str());
        }
        
    retry:
        profile::add_post_retry_count();
        bool posted;
        if constexpr (IsReadRequest) {
            assert(entry.load_state().state == FETCHING);
            if (use_read_batch) {
                const auto queued = client->enqueue_sync_read(
                    offset, local_addr, obj.size, wr_id, obj.obj_id, 0,
                    endpoint_idx);
                if (queued.status ==
                    rdma::read_batch::QueueStatus::Busy) {
                    client->record_backpressure_yield();
                    uthread::yield();
                    goto retry;
                }
                if (queued.status ==
                    rdma::read_batch::QueueStatus::Full) {
                    const size_t completed =
                        this->check_cq_idx_with_client_idx_endpoint(
                        qp_idx, client_idx, endpoint_idx);
                    if (completed == 0) {
                        client->record_backpressure_yield();
                        uthread::yield();
                    }
                    goto retry;
                }
                if (!queued.enqueued) std::abort();
                return;
            }
            request_interval_diag::set_post_context(request_interval_token);
            posted = client->post_read(
                offset, local_addr, obj.size, wr_id, obj.obj_id, 0,
                endpoint_idx);
            request_interval_diag::clear_post_context();
        } else {
            // Use original post_write for now
            posted = rdma::get_client(client_idx)->post_write(
                offset, local_addr, obj.size, wr_id, obj.obj_id, 0, endpoint_idx);
        }
        if (!posted) {
            this->check_cq_idx_with_client_idx_endpoint(qp_idx, client_idx, endpoint_idx);
            goto retry;
        }

        if constexpr (IsReadRequest) {
            // B1 observer: count only after ibv_post_send accepted the READ.
            read_supply_timeline::record_read_accepted(client_idx, qp_idx,
                                                        wr_id);
        } else {
            profile::count_rdma_write_post(obj.size);
        }

    }

    void post_read_request_with_idx(far_obj_t obj, size_t client_idx,
                                    bool sync_batch_eligible = false) {
        profile::start_post_fetch();
        post_rdma_request<true>(obj, client_idx, sync_batch_eligible);
        profile::end_post_fetch();
    }

    void post_write_requests(EvictBuffer &buffer, size_t client_idx, size_t endpoint_idx = 0) {
        if (buffer.count == 0) return;
        assert(buffer.count <= EvictBatchSize);
        const size_t posted_count = buffer.count;
        auto client = rdma::get_client(client_idx);
        size_t qp_idx = client->get_qp_idx();
        const auto& config = ::FarLib::get_config();

        ibv_sge write_sge[EvictBatchSize];
        ibv_send_wr write_wr[EvictBatchSize];
        size_t total_bytes = 0;
        const bool fine_profile = profile::evac_fine_profile_runtime_enabled();
        const uint64_t build_start = fine_profile ? get_cycles() : 0;
        for (size_t i = 0; i < buffer.count; i++) {
            auto &req = buffer.requests[i];
            assert(req.remote_addr != FarObjectEntry::RemoteAddrInvalid48);
            total_bytes += req.size;
            
            size_t offset = req.remote_addr;
            if (config.server_count > 1) {
                offset = config.map_remote_addr(req.remote_addr).second;
            }
            client->build_send_wr(write_wr[i], write_sge[i], offset,
                                  req.local_addr, req.size,
                                  reinterpret_cast<uint64_t>(req.local_addr),
                                  true, IBV_WR_RDMA_WRITE, endpoint_idx);
            if (i > 0) {
                write_wr[i - 1].next = &(write_wr[i]);
            }
        }
        if (fine_profile) {
            profile::count_evac_post_write_build_wr_cycles(
                (int64_t)(get_cycles() - build_start));
        }
        ibv_send_wr *bad_wr = write_wr;

        const uint64_t post_start = fine_profile ? get_cycles() : 0;
        bool posted_all = client->post_writes(bad_wr, &bad_wr, qp_idx, endpoint_idx);

        int64_t retry_count = 0;
        while (!posted_all) [[unlikely]] {
            retry_count++;
            check_cq_idx_with_client_idx(qp_idx, client_idx);
            posted_all = client->post_writes(bad_wr, &bad_wr, qp_idx, endpoint_idx);
        }
        if (fine_profile) {
            profile::count_evac_post_write_post_send_cycles(
                (int64_t)(get_cycles() - post_start));
            profile::count_evac_post_write_retry_count(retry_count);
        }

        // All WRs in this buffer have now been accepted, including any suffix
        // retried after a partial post.
        read_supply_timeline::record_write_posts(client_idx, qp_idx,
                                                 posted_count);

        profile::count_evacuation_bytes(total_bytes);
        profile::count_evac_flush(posted_count, posted_count == EvictBatchSize);
        buffer.count = 0;
    }

    void add_write_request(EvictBuffer &buffer, void *local_addr,
                           uint64_t remote_addr, uint64_t size, size_t endpoint_idx) {
        size_t client_idx = rdma::thread_info.thread_id;
        if (buffer.count == EvictBatchSize) [[unlikely]] {
            post_write_requests(buffer, client_idx, endpoint_idx);
        }
        profile::count_rdma_write_post(size);
        assert(buffer.count < EvictBatchSize);
        buffer.requests[buffer.count] = {local_addr, remote_addr, (uint64_t)size};
        buffer.count++;
    }
    
    // Endpoint-aware version using EvictBufferSet
    void add_write_request(EvictBufferSet &buffer_set, void *local_addr,
                           uint64_t remote_addr, uint64_t size) {
        auto& config = ::FarLib::get_config();
        size_t endpoint_idx = 0;
        if (config.server_count > 1) {
            endpoint_idx = config.map_remote_addr(remote_addr).first;
        }
        auto& buffer = buffer_set.buffers[endpoint_idx];
        add_write_request(buffer, local_addr, remote_addr, size, endpoint_idx);
    }

    // ---------------------------------------------------------------------
    // EC (ft_method=sponge) commit path.  Reachable only when a small object
    // is evicted while ft_method != none (see try_evict()); with the default
    // ft_method=none EvictBufferSet::sponge_buffers stays null and neither
    // function below is ever called.
    // ---------------------------------------------------------------------
    void post_sponge_commit_requests(SpongeEvictBuffer &buffer,
                                     size_t client_idx,
                                     size_t endpoint_idx) {
        if (buffer.empty()) return;
        const size_t posted_count = buffer.batch.header.record_count;
        const size_t posted_bytes = buffer.committed_bytes;
        auto *client = rdma::get_client(client_idx);
        size_t qp_idx = client->get_qp_idx();
        const bool fine_profile = profile::evac_fine_profile_runtime_enabled();
        const uint64_t post_start = fine_profile ? get_cycles() : 0;
        int64_t retry_count = 0;
        while (!client->post_sponge_commit_batch(endpoint_idx, qp_idx,
                                                 buffer.batch)) [[unlikely]] {
            retry_count++;
            check_cq_idx_with_client_idx_endpoint(qp_idx, client_idx,
                                                  endpoint_idx);
            if ((retry_count & 63) == 0) {
                uthread::yield();
            }
        }
        if (fine_profile) {
            profile::count_evac_post_write_post_send_cycles(
                (int64_t)(get_cycles() - post_start));
            profile::count_evac_post_write_retry_count(retry_count);
        }
        profile::count_evacuation_bytes(posted_bytes);
        profile::count_evac_flush(posted_count,
                                  buffer.batch.header.payload_bytes ==
                                      rdma::kSpongeBatchPayloadBytes);
        buffer.reset();
    }

    // Appends one commit record to the endpoint's sponge batch, flushing the
    // batch first when the record does not fit.
    void add_sponge_commit_request(EvictBufferSet &buffer_set,
                                   size_t endpoint_idx,
                                   const rdma::SpongeCommitRecord &record) {
        ASSERT(buffer_set.sponge_buffers != nullptr);
        auto &buffer = buffer_set.sponge_buffers[endpoint_idx];
        const bool fine_profile = profile::evac_fine_profile_runtime_enabled();
        uint64_t build_start = fine_profile ? get_cycles() : 0;
        if (!buffer.batch.append(record)) {
            if (fine_profile) {
                profile::count_evac_post_write_build_wr_cycles(
                    (int64_t)(get_cycles() - build_start));
            }
            post_sponge_commit_requests(buffer, rdma::thread_info.thread_id,
                                        endpoint_idx);
            build_start = fine_profile ? get_cycles() : 0;
            if (!buffer.batch.append(record)) {
                ERROR("sponge commit record does not fit in empty batch");
            }
        }
        if (fine_profile) {
            profile::count_evac_post_write_build_wr_cycles(
                (int64_t)(get_cycles() - build_start));
        }
        buffer.committed_bytes += record.slot_size;
        profile::count_rdma_write_post(record.obj_size);
    }

    EntryState try_mark(
        ::FarLib::allocator::BlockHead *block,
        profile::DualFrequencyHistogramSnapshot *frequency_histogram = nullptr,
        bool update_ema_this_pass = true);
    EntryState try_evict(::FarLib::allocator::BlockHead *block,
                         EvictBufferSet &buffer_set);

    size_t get_mark_worker_count() const;
    size_t get_evict_worker_count(size_t mark_workers) const;
    FrequencyMarkPassContext begin_frequency_mark_pass();

    struct StreamingMarkMasterArgs {
        ConcurrentArrayCache *cache;
        uint32_t timestamp;
        size_t worker_count;
        FrequencyMarkPassContext frequency_mark_pass;
        bool flip_after_mark;
        std::vector<::FarLib::allocator::EvictTask> *ready_tasks_out;
        std::atomic_size_t *ready_task_budget;
        std::atomic_size_t *legacy_region_budget = nullptr;
        bool legacy_region_budget_per_worker = false;
        std::atomic_bool *evict_stop_after_mark = nullptr;
        std::atomic_bool *mark_stop_after_evict = nullptr;
    };

    struct StreamingEvictMasterArgs {
        struct ResumeScanRound {
            uint64_t evict_generation;
            uint64_t gc_generation;
            std::atomic_size_t *evict_visit_budget;
            std::atomic_size_t *gc_visit_budget;
            std::atomic_bool *evict_pass_complete;
            std::atomic_bool *gc_pass_complete;
        };
        ConcurrentArrayCache *cache;
        uint32_t safe_epoch_to_wait;
        uint32_t evict_timestamp;
        uint32_t gc_timestamp;
        size_t worker_count;
        std::atomic_int64_t *posted_wrs_total;
        const std::vector<::FarLib::allocator::EvictTask> *ready_tasks_in;
        std::vector<::FarLib::allocator::EvictTask> *deferred_tasks_out;
        ResumeScanRound *resume_scan = nullptr;
        std::atomic_bool *evict_stop_after_mark = nullptr;
        std::atomic_size_t *remaining_evict_post_workers = nullptr;
        std::atomic_bool *mark_stop_after_evict = nullptr;
    };

    static void run_streaming_mark_master(StreamingMarkMasterArgs *args);
    static void run_streaming_evict_master(StreamingEvictMasterArgs *args);

    void wait_for_eviction_trigger(uint64_t handled_eviction_request_seq);
    void notify_mutators_ready();
    void evacuate_work();
    template <typename ReadyTaskSink>
    void mark_phase_to_sink(
        uint32_t timestamp,
        const FrequencyMarkPassContext &frequency_mark_pass,
        ReadyTaskSink *ready_tasks = nullptr,
        std::atomic_size_t *ready_task_budget = nullptr);
    void mark_phase(uint32_t timestamp,
                    const FrequencyMarkPassContext &frequency_mark_pass,
                    std::vector<::FarLib::allocator::EvictTask> *ready_tasks =
                        nullptr,
                    std::atomic_size_t *ready_task_budget = nullptr,
                    size_t diag_worker_slot =
                        inclusive_reclaim_diag::kNoWorker,
                    std::atomic_size_t *legacy_region_budget = nullptr,
                    std::atomic_bool *mark_stop_after_evict = nullptr);
    void evict_post_phase(uint32_t timestamp);
    void evict_post_worker_logic(uint32_t timestamp,
                                 std::atomic_int64_t &posted_wrs_total,
                                 bool ignore_safe_epoch = false,
                                 size_t diag_worker_slot =
                                     inclusive_reclaim_diag::kNoWorker,
                                 ::FarLib::allocator::EvacuationEligibility
                                     eligibility = ::FarLib::allocator::
                                         EvacuationEligibility::All,
                                 StreamingEvictMasterArgs::ResumeScanRound
                                     *resume_scan = nullptr,
                                 std::atomic_bool *evict_stop_after_mark =
                                     nullptr);
    void evict_ready_worker_logic(
        const std::vector<::FarLib::allocator::EvictTask> &ready_tasks,
        std::atomic_size_t &next_task_idx, uint32_t current_safe_epoch,
        std::vector<::FarLib::allocator::EvictTask> &deferred_tasks,
        std::atomic_int64_t &posted_wrs_total);
    void evict_drain_phase();
    void drain_eviction_completions_for_shutdown();
    void evacuate_phase(uint32_t timestamp);
    void gc_phase(
        uint32_t timestamp,
        ::FarLib::allocator::EvacuationEligibility eligibility =
            ::FarLib::allocator::EvacuationEligibility::All,
        StreamingEvictMasterArgs::ResumeScanRound *resume_scan = nullptr);
    void flip_scope_state(uint32_t epoch_to_release = 0);
    void on_demand_invoke_eviction(size_t alloc_bin = kAnyAllocBin);

    void deallocate_local(void *ptr, FarObjectEntry &entry) {
        auto block = static_cast<::FarLib::allocator::BlockHead *>(ptr) - 1;
        // assert(block->get_object_ptr() == entry.local_addr());
        // assert(block->obj_meta_data.load().get_entry_ptr() == &entry);
        block->obj_meta_data = far_obj_t::null();
        ::FarLib::allocator::block_to_region(block)
            ->pending_reclaims.fetch_add(1, std::memory_order_release);
    }

    template <bool IncreaseRefCount>
    bool post_fetch(far_obj_t obj, DereferenceScope &scope,
                    bool sync_batch_eligible = false);
    template <bool Mut>
    static bool fetch_lite_fast_path(EntryStateBits state);
    template <bool Mut, bool Profile = true>
    bool post_fetch_lite(far_obj_t obj, DereferenceScope &scope,
                         bool sync_batch_eligible = false);
    template <bool Mut, bool Profile = true>
    bool post_fetch_lite_slow_path(FarObjectEntry &entry, far_obj_t obj,
                                   DereferenceScope &scope,
                                   bool sync_batch_eligible = false);

    template <bool DeallocateEntry, bool OldAccessor = false>
    void deallocate(FarObjectEntry &entry, size_t size) {
        int invalid_spin = 0;
    retry_load:
        auto old_state = entry.load_state();
        if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
            auto spin_start = get_cycles();
            profile::count_excl_move_lock_spin();
            this->check_cq_idx_with_client_idx(0, entry.get_client_idx());
            uthread::yield();
            profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
            if (++invalid_spin > kSpinYieldThreshold) {
                full_checker();
                invalid_spin = 0;
            }
            goto retry_load;
        }
        if (old_state.ref_cnt != 0) {
            const bool quiesced_local_without_rdma =
                !::FarLib::get_config().is_ec_batch_mode() &&
                !working.load(std::memory_order_acquire) &&
                !old_state.invalid &&
                (old_state.state == LOCAL || old_state.state == MARKED ||
                 old_state.state == PINNED) &&
                entry.remote_addr() == FarObjectEntry::RemoteAddrInvalid48;
            if (quiesced_local_without_rdma) {
                auto recovered_state = old_state;
                recovered_state.ref_cnt = 0;
                if (entry.cas_state_weak(old_state, recovered_state)) {
                    quiesce_recovered_stale_local_ref_count.fetch_add(
                        1, std::memory_order_relaxed);
                }
                goto retry_load;
            }
            int spin = 0;
            int ec_spin = 0;
            uint64_t ec_stuck_spin = 0;
            const auto ec_dealloc_started = ::FarLib::get_config().is_ec_batch_mode()
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            if (::FarLib::get_config().is_ec_batch_mode()) [[unlikely]] {
                // A partially filled group owns a real write reference too.
                // Submit it before waiting; application destruction may run
                // before the evacuator's next ordinary batch flush.
                this->flush_ec_batch_groups(rdma::thread_info.thread_id);
                // ft_method=ec_batch diagnostics: reaching this wait with the
                // object already EVICTING means the group that owns its write
                // reference has not delivered its completions yet.  One damped
                // line per object; the wait itself is untouched.
                if (entry.load_state().state == EVICTING) {
                    this->ec_batch_report_stuck_spin(
                        entry, 0,
                        ec_batch_last_poll_harvest_.load(
                            std::memory_order_relaxed),
                        "evicting_enter", true);
                }
            }
            while (entry.load_state().ref_cnt != 0) {
                spin++;
                this->check_cq_idx_with_client_idx(0, entry.get_client_idx());
                if (::FarLib::get_config().is_ec_batch_mode()) [[unlikely]] {
                    // ft_method=ec_batch: an object's write reference belongs to
                    // its group, and the six CQEs of a group are reaped by
                    // whichever thread polls the (client, QP, endpoint) CQ they
                    // land in.  Help those CQs along instead of watching ref_cnt
                    // only: a group whose CQEs nobody polls any more would never
                    // release this object.  While the cache is working the real
                    // completion is still waited for; only a quiesced cache lets
                    // the teardown path settle groups that cannot complete.
                    ec_spin++;
                    if ((ec_spin & 7) == 0) {
                        const size_t poll_harvest =
                            this->ec_batch_poll_all_cqs_once();
                        ec_batch_last_poll_harvest_.store(
                            poll_harvest, std::memory_order_relaxed);
                    }
                    // Diagnostics only: report this spin once it is clearly
                    // stuck (first line at 1e5 spins, then every 1e6) so a hang
                    // shows what it waits for.  No wait/settle change.
                    ec_stuck_spin++;
                    if (ec_stuck_spin == 100000ull ||
                        (ec_stuck_spin > 100000ull &&
                         (ec_stuck_spin % 1000000ull) == 0)) {
                        this->ec_batch_report_stuck_spin(
                            entry, ec_stuck_spin,
                            ec_batch_last_poll_harvest_.load(
                                std::memory_order_relaxed),
                            "ref_cnt_spin", false);
                    }
                    if ((ec_stuck_spin & 1023ull) == 0 &&
                        std::chrono::steady_clock::now() - ec_dealloc_started >=
                            std::chrono::seconds(30)) {
                        this->ec_batch_report_stuck_spin(
                            entry, ec_stuck_spin,
                            ec_batch_last_poll_harvest_.load(
                                std::memory_order_relaxed),
                            "deallocate_timeout", true);
                        ERROR("ec_batch: live reference did not drain; refusing forced free");
                    }
                }
                if (spin > kSpinYieldThreshold) {
                    full_checker();
                    uthread::yield();
                    spin = 0;
                }
            }
            goto retry_load;
        }
    retry:
        auto new_state = old_state;
        if constexpr (OldAccessor) {
            new_state.ref_cnt--;
            if (new_state.ref_cnt != 0) {
                // someone still hold obj's accessor
                // cas and return
                if (entry.cas_state_weak(old_state, new_state)) {
                    return;
                }
                goto retry_load;
            }
        }
        switch (old_state.state) {
        case FREE:
            ERROR("should not deallocate FREE object");
        case FETCHING:
        case EVICTING: {
            int spin = 0;
            while (true) {
                auto cur_state = entry.load_state();
                if (cur_state.state != FETCHING && cur_state.state != EVICTING) break;
                spin++;
                this->check_cq_idx_with_client_idx(0, entry.get_client_idx());
                if (spin > kSpinYieldThreshold) {
                    full_checker();
                    uthread::yield();
                    spin = 0;
                }
            }
        }
            // TODO: can we deallocate concurrenctly?
        case BUSY:
            goto retry_load;
        case PINNED:
        case MARKED:
        case LOCAL: {
            // check if package about this object
            // are on the fly
            if (new_state.ref_cnt != 0) {
                int spin = 0;
                while (entry.load_state().ref_cnt != 0) {
                    spin++;
                    this->check_cq_idx_with_client_idx(0, entry.get_client_idx());
                    if (spin > kSpinYieldThreshold) {
                        full_checker();
                        uthread::yield();
                        spin = 0;
                    }
                }
                goto retry_load;
            }
            new_state.state = FREE;
            if (!entry.cas_state_weak(old_state, new_state)) [[unlikely]] {
                goto retry;
            }
            assert(new_state.ref_cnt == 0);
            six_free(entry, size);
            unregister_live_entry(&entry);
            deallocate_local(entry.local_addr(), entry);
            {
                uint64_t remote_addr = entry.remote_addr();
                bool had_retained_backup =
                    entry.has_remote_backup_reservation();
                if (!::FarLib::get_config().exclusive_cache ||
                    remote_addr != FarObjectEntry::RemoteAddrInvalid48) {
                    remote_allocator.deallocate(remote_addr);
                }
                if (had_retained_backup) {
                    release_remote_backup_budget(size);
                }
                if (entry.is_resident_local() &&
                    !region_group_binding_enabled()) {
                    const size_t resident_bytes =
                        local_allocation_footprint(size);
                    release_profiled_resident_bytes(entry, resident_bytes);
                    if (!region_resident_placement_enabled()) {
                        release_resident_local(resident_bytes);
                    }
                }
                release_resident_placement_allocation(
                    entry, local_allocation_footprint(size));
                entry.store_placement_flags(0);
            }
            if constexpr (DeallocateEntry) {
                deallocate_entry(&entry);
            }
            break;
        }
        case REMOTE:
            ASSERT(!entry.is_resident_local());
            ASSERT(!entry.has_remote_backup_reservation());
            new_state.state = FREE;
            if (!entry.cas_state_weak(old_state, new_state)) [[unlikely]] {
                goto retry;
            }
            six_free(entry, size);
            unregister_live_entry(&entry);
            {
                uint64_t remote_addr = entry.remote_addr();
                if (remote_addr == FarObjectEntry::RemoteAddrInvalid48) {
                    std::string err = "DEALLOC_REMOTE_FAIL: remote_addr is invalid for REMOTE state\n";
                    EXCLUSIVE_ERROR(err);
                    std::abort();
                }
                remote_allocator.deallocate(remote_addr);
            }
            release_resident_placement_allocation(
                entry, local_allocation_footprint(size));
            if constexpr (DeallocateEntry) {
                deallocate_entry(&entry);
            }
            break;
        };
    }

    void print_design1_diagnostics() const;

public:
    ConcurrentArrayCache(void *local_buf, size_t local_buf_size,
                         size_t remote_buf_size, size_t evict_batch_size)
        : remote_allocator(remote_buf_size),
          working(true),
          mutator_can_not_allocate(false) {
        // EC (ft_method=ec_batch): carve the resident staging range out of the
        // registered client buffer before the heap is registered, so the
        // allocator can never hand those bytes out as object memory.
        local_buf_size = init_ec_batch_staging(local_buf, local_buf_size);
        // EC recovery scratch is registered lazily in separate grow-on-demand
        // chunks. It does not impose a fixed depth or carve out application MR.
        local_buf_size = init_ec_read_recovery(local_buf, local_buf_size);
        const auto member_offset = [this](const void *member) {
            return reinterpret_cast<uintptr_t>(member) -
                   reinterpret_cast<uintptr_t>(this);
        };
        const auto retained_offset = member_offset(&retained_backup_bytes);
        const auto peak_offset = member_offset(&peak_retained_backup_bytes);
        const auto budget_offset = member_offset(&retained_backup_budget_bytes);
        const auto resident_offset = member_offset(&resident_local_bytes);
        if constexpr (FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION) {
            if (reinterpret_cast<uintptr_t>(this) % 64 != 0 ||
                retained_offset % 64 != 0 || peak_offset % 64 != 0 ||
                budget_offset % 64 != 0 || resident_offset % 64 != 0 ||
                peak_offset - retained_offset < 64 ||
                budget_offset - peak_offset < 64 ||
                resident_offset - budget_offset < 64) {
                ERROR("isolated backup accounting layout check failed");
            }
        }
        std::cerr << "backup_accounting_layout isolated="
                  << FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION
                  << " cache_size=" << sizeof(*this)
                  << " cache_alignment=" << alignof(ConcurrentArrayCache)
                  << " retained_offset=" << retained_offset
                  << " peak_offset=" << peak_offset
                  << " budget_offset=" << budget_offset
                  << " resident_offset=" << resident_offset << '\n';
        const auto &config = ::FarLib::get_config();
        if (::FarLib::allocator::six_group::enabled() && !config.exclusive_cache)
            ERROR("fixed-six aligned runtime currently requires exclusive_cache");
        backup_profile_window_events = std::max<size_t>(
            1, config.remote_backup_profile_window_events);
        backup_profile_min_score_pct = std::min<size_t>(
            100, config.remote_backup_profile_min_score_pct);
        logical_object_profile_enabled_flag =
            config.enable_logical_object_profile;
        logical_object_profile_sample_shift = std::min<size_t>(
            20, config.logical_object_profile_sample_shift);
        // EC batch owns object/group placement while a group is staged,
        // reconstructed, and rewritten.  The Resident planner mutates
        // per-entry placement state asynchronously and has a second direct
        // eviction call path, so it must be disabled for the entire EC cache
        // lifetime rather than relying on a transient config predicate at
        // each call site.
        const bool ec_batch_config = config.ft_method == "ec_batch";
        ec_batch_mode_ = ec_batch_config;
        resident_profile_planner_enabled_flag =
            config.enable_resident_profile_planner;
        if (config.enable_region_hotness_placement && !ec_batch_config) {
            ASSERT(config.enable_region_resident_placement);
            ASSERT(!config.region_placement_bind_groups);
            ASSERT(config.enable_resident_profile_planner);
        }
        if (config.enable_region_fetch_hotness_placement) {
            ASSERT(config.enable_region_hotness_placement);
            const auto parsed_policy = design1::parse_placement_policy(
                config.region_fetch_placement_policy);
            if (!parsed_policy.has_value()) {
                ERROR("region_fetch_placement_policy must be hotness or design1_gain");
            }
            region_fetch_placement_policy = *parsed_policy;
        }
        resident_profile_require_work_phase_flag = env_flag_or_default(
            "FARLIB_RESIDENT_PROFILE_REQUIRE_WORK_PHASE",
            config.resident_profile_require_work_phase);
        resident_profile_apply_plan_flag = config.resident_profile_apply_plan;
        resident_profile_manual_trigger_flag =
            config.resident_profile_manual_trigger;
        resident_profile_warmup_ms = config.resident_profile_warmup_ms;
        resident_profile_sample_shift =
            std::min<size_t>(20, config.resident_profile_sample_shift);
        resident_profile_group_bytes =
            std::max<size_t>(1, config.resident_profile_group_bytes);
        resident_profile_interval_ms =
            std::max<size_t>(1, config.resident_profile_interval_ms);
        resident_profile_initial_sample_windows =
            config.resident_profile_initial_sample_windows;
        resident_profile_min_benefit_refs_per_replacement_object =
            config.resident_profile_min_benefit_refs_per_replacement_object;
        resident_profile_write_weight = config.resident_profile_write_weight;
        resident_profile_ema_decay_shift =
            std::min<size_t>(63, config.resident_profile_ema_decay_shift);
        resident_profile_migration_budget_pct = std::min<size_t>(
            100, config.resident_profile_migration_budget_pct);
        resident_profile_migration_budget_objects =
            config.resident_profile_migration_budget_objects;
        resident_local_budget_bytes = std::min<uint64_t>(
            config.local_resident_budget_bytes, local_buf_size);
        if (config.exclusive_cache && config.enable_selective_backup) {
            uint64_t configured_budget = config.remote_backup_budget_bytes;
            if (configured_budget == 0) {
                configured_budget =
                    remote_buf_size * config.remote_backup_budget_pct / 100;
            }
            retained_backup_budget_bytes =
                std::min<uint64_t>(configured_budget, remote_buf_size);
            if (config.remote_backup_mode != "greedy" &&
                config.remote_backup_mode != "segmented" &&
                config.remote_backup_mode != "profiled" &&
                config.remote_backup_mode != "object_profiled") {
                ERROR("remote_backup_mode must be greedy, segmented, profiled, or object_profiled");
            }
            segmented_backup_mode = config.remote_backup_mode != "greedy";
            profiled_backup_mode = config.remote_backup_mode == "profiled";
            object_profiled_backup_mode =
                config.remote_backup_mode == "object_profiled";
            logical_object_profile_enabled_flag =
                logical_object_profile_enabled_flag || object_profiled_backup_mode;
            if (segmented_backup_mode) {
                ASSERT(resident_local_budget_bytes != 0);
            }
        }
        const uint64_t max_transient_local_bytes =
            local_buf_size - resident_local_budget_bytes;
        backup_covers_transient_window =
            segmented_backup_mode &&
            retained_backup_budget_bytes >= max_transient_local_bytes;
        all_nonresident_backup_mode = env_flag_or_default("FARLIB_ALL_NONRESIDENT_BACKUP", false);
        if (all_nonresident_backup_mode) {
            if (!config.enable_region_resident_placement ||
                !selective_backup_enabled() || !segmented_backup_enabled() ||
                !backup_covers_transient_window) {
                ERROR("all-nonresident backup requires exclusive region placement and full transient coverage");
            }
            std::cout << "nonresident_backup_policy enabled=1 whitelist=none resident_backup=0"
                      << " local_bytes=" << local_buf_size
                      << " resident_budget_bytes=" << resident_local_budget_bytes
                      << " transient_capacity_bytes=" << max_transient_local_bytes
                      << " backup_budget_bytes=" << retained_backup_budget_bytes << std::endl;
        }
        std::cout << "remote backup covers transient window: "
                  << backup_covers_transient_window << std::endl;
        if (env_flag_or_default("FARLIB_BACKUP_COUNTER_BYPASS", false)) {
            ASSERT(all_nonresident_backup_mode);
            bypass_backup_usage = std::make_unique<BackupUsageShards>();
            std::cout << "backup_counter_bypass enabled=1 capacity_check=disabled"
                      << " observer=per_os_thread_signed_shards peak=sampled_only"
                       << " reservation_flags=unchanged" << std::endl;
        }
        if (env_flag_or_default("FARLIB_BACKUP_CREDITS", false)) {
            ASSERT(all_nonresident_backup_mode && !bypass_backup_usage);
            batched_backup_budget = std::make_unique<BatchedBackupBudget>(
                retained_backup_budget_bytes, ::FarLib::allocator::RegionSize);
            std::cout << "backup_credits enabled=1 capacity_check=strict"
                      << " acquire=fibre_fetch_or_native_completion release=os_thread_local"
                      << " quantum_bytes=" << ::FarLib::allocator::RegionSize
                      << " peak=issued_upper_bound reservation_flags=unchanged" << std::endl;
        }
        if (resident_profile_planner_enabled_flag) {
            ASSERT(resident_local_budget_bytes != 0);
            logical_object_profile_enabled_flag = true;
            if (!resident_profile_manual_trigger_flag) {
                resident_placement_groups =
                    std::make_unique<ResidentPlacementGroupProfile[]>(
                        kMaxResidentPlacementGroupCount);
            }
            resident_profile_active.store(
                !resident_profile_manual_trigger_flag,
                std::memory_order_release);
        }
        ::FarLib::allocator::global_heap.register_heap(local_buf, local_buf_size);
        if (::FarLib::simple_dirty_observer::enabled()) {
            // The two-byte trace handle has one owner. Do not combine this
            // observer with the legacy fixed-six object's registry/tracer.
            if (!::FarLib::simple_region_budget::six_enabled() ||
                !::FarLib::simple_region_heat::enabled() ||
                ::FarLib::allocator::six_group::enabled() ||
                !config.exclusive_cache || config.enable_selective_backup)
                throw std::invalid_argument("dirty observer requires simple-six exclusive, backup OFF");
#ifdef REMOTE_REGION_ALLOCATOR
            ::FarLib::simple_dirty_observer::monitor().configure(
                reinterpret_cast<uintptr_t>(local_buf), local_buf_size,
                ::FarLib::allocator::RegionSize, remote_buf_size,
                ::FarLib::allocator::remote::RegionSize);
            ::FarLib::simple_dirty_observer::monitor().set_class_reader(
                [base=reinterpret_cast<uintptr_t>(local_buf)](bool remote,size_t region) {
                    return remote
                        ? ::FarLib::allocator::remote::remote_global_heap.simple_class_for(
                              region * ::FarLib::allocator::remote::RegionSize)
                        : ::FarLib::simple_region_heat::allocation_class_for(
                              base + region * ::FarLib::allocator::RegionSize);
                });
#else
            throw std::invalid_argument("dirty observer requires Region remote allocator");
#endif
        }
        if (::FarLib::simple_region_heat::enabled()) {
            ::FarLib::simple_region_heat::monitor().configure(
                reinterpret_cast<uintptr_t>(local_buf), local_buf_size,
                ::FarLib::allocator::RegionSize);
            if(::FarLib::simple_region_heat::grouping_enabled() ||
               ::FarLib::allocator::six_group::list_only_six::grouped())
                ::FarLib::simple_region_heat::monitor().set_reclassify([] {
                    return ::FarLib::allocator::global_heap.reclassify_simple_heat();
                });
            if (::FarLib::simple_region_budget::enabled())
                ::FarLib::simple_region_heat::monitor().set_budget_tick([](uint64_t window) {
                    ::FarLib::allocator::global_heap.tick_simple_budget(window);
#ifdef REMOTE_REGION_ALLOCATOR
                    ::FarLib::allocator::remote::remote_global_heap.tick_simple_budget(window);
#endif
                });
        }
        if (::FarLib::allocator::six_group::enabled()) {
            const bool work_only = env_flag_or_default("FARLIB_SIX_REQUIRE_WORK_PHASE", true);
            ::FarLib::allocator::six_group::registry().start(
                local_buf_size, work_only ? +[] { return profile::is_working(); } : nullptr,
                &::FarLib::allocator::disable_region_list_fibre_yield_for_current_thread);
        }
        void (*evict_fn)(ConcurrentArrayCache *) =
            [](ConcurrentArrayCache *cache) { cache->evacuate_work(); };
        ASSERT(::FarLib::get_config().evacuate_thread_cnt > 0);
        evacuate_thread_cnt = ::FarLib::get_config().evacuate_thread_cnt;
        if (uthread::separate_background_cluster_enabled()) {
            const size_t background_workers = evacuate_thread_cnt;
            const char *cpu_base_text =
                std::getenv("FARLIB_BACKGROUND_CPU_BASE");
            if (cpu_base_text == nullptr || cpu_base_text[0] == '\0') {
                ERROR("FARLIB_BACKGROUND_CPU_BASE is required");
            }
            char *cpu_base_end = nullptr;
            const unsigned long cpu_base =
                std::strtoul(cpu_base_text, &cpu_base_end, 10);
            if (cpu_base_end == cpu_base_text || *cpu_base_end != '\0' ||
                cpu_base >= CPU_SETSIZE ||
                background_workers > CPU_SETSIZE - cpu_base) {
                ERROR("invalid FARLIB_BACKGROUND_CPU_BASE");
            }
            std::vector<pthread_t> app_tids(
                ::FarLib::get_config().max_thread_cnt);
            const size_t app_actual = uthread::get_current_worker_sys_ids(
                app_tids.data(), app_tids.size());
            ASSERT(app_actual == app_tids.size());
            cpu_set_t app_cpu_union;
            CPU_ZERO(&app_cpu_union);
            for (pthread_t tid : app_tids) {
                cpu_set_t worker_set;
                CPU_ZERO(&worker_set);
                if (pthread_getaffinity_np(tid, sizeof(worker_set),
                                           &worker_set) != 0) {
                    ERROR("failed to read app cluster affinity");
                }
                CPU_OR(&app_cpu_union, &app_cpu_union, &worker_set);
            }
            for (size_t i = 0; i < background_workers; ++i) {
                if (CPU_ISSET(static_cast<int>(cpu_base + i),
                              &app_cpu_union)) {
                    ERROR("background CPU overlaps app cluster affinity");
                }
            }
            background_cluster_ =
                &uthread::create_cluster(background_workers);
            std::vector<pthread_t> background_tids(background_workers);
            const size_t actual = uthread::get_worker_sys_ids(
                *background_cluster_, background_tids.data(),
                background_tids.size());
            ASSERT(actual == background_workers);
            for (size_t i = 0; i < background_workers; ++i) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(static_cast<int>(cpu_base + i), &set);
                if (pthread_setaffinity_np(background_tids[i], sizeof(set),
                                           &set) != 0) {
                    ERROR("failed to pin background cluster worker");
                }
                cpu_set_t observed;
                CPU_ZERO(&observed);
                if (pthread_getaffinity_np(background_tids[i],
                                           sizeof(observed), &observed) != 0 ||
                    CPU_COUNT(&observed) != 1 ||
                    !CPU_ISSET(static_cast<int>(cpu_base + i), &observed)) {
                    ERROR("background cluster affinity verification failed");
                }
                rdma::register_thread_id(
                    background_tids[i],
                    ::FarLib::get_config().max_thread_cnt + i);
            }
            master_evacuation_thread = uthread::create_on<true>(
                *background_cluster_, evict_fn, this,
                std::string("evacuation master"));
        } else {
            master_evacuation_thread = uthread::create<true>(
                evict_fn, this, std::string("evacuation master"));
        }
        ::FarLib::allocator::global_heap.set_on_memory_low(
            [this] {
                uthread::notify_all(&eviction_cond, &eviction_mutex);
            });
        if (full_population_frequency_stats_enabled()) {
            full_population_frequency_tracker.start(*this);
        }
        if (resident_profile_planner_enabled_flag &&
            !resident_profile_manual_trigger_flag &&
            !resident_profile_phase_triggered_enabled()) {
            resident_profile_thread =
                std::thread([this] { run_resident_profile_planner(); });
        }
        start_read_supply_timeline();
    }

    ~ConcurrentArrayCache() {
        ::FarLib::simple_region_heat::end_work();
        ec_batch_diag_report("cache_dtor_begin");
        ec_read_recovery_diag_report("cache_dtor_begin");
        stop_read_supply_timeline();
        quiesce_background_evacuation();
        if (::FarLib::allocator::six_group::enabled()) {
            ::FarLib::allocator::six_group::registry().stop();
            ::FarLib::allocator::six_group::registry().dump();
        }
        ::FarLib::object_group_trace::dump();
        ::FarLib::allocator::release_all_thread_heap_regions_for_shutdown();
        ::FarLib::allocator::global_heap.finish_async_full_returns_for_shutdown();
        ::FarLib::allocator::global_heap.reclaim_all_deallocated_regions();
        ::FarLib::allocator::global_heap
            .verify_no_detached_regions_for_shutdown();
        ::FarLib::allocator::global_heap.print_used_memory();
        print_design1_diagnostics();
        ::FarLib::allocator::global_heap.destroy();
        ec_batch_diag_step("cache_dtor_end");
    }

    void quiesce_background_evacuation() {
        ec_batch_diag_step("quiesce_background_evacuation_begin");
        // ft_method=ec_batch: one bounded drain of the write round *before* the
        // workers stop, so every group that is still observable gets its real
        // completions.  What is left after that is settled further down.
        drain_ec_batch_in_flight_groups_for_shutdown();
        bool was_working = working.exchange(false, std::memory_order_acq_rel);
        if (!was_working) {
            stop_resident_profile_planner();
            full_population_frequency_tracker.stop();
            settle_ec_batch_in_flight_groups(
                "quiesce_settle_never_completing_groups");
            ec_batch_diag_step("before_drain_eviction_completions_for_shutdown");
            drain_eviction_completions_for_shutdown();
            return;
        }
        stop_resident_profile_planner();
        mutator_can_not_allocate.store(false, std::memory_order_release);
        uthread::notify_all(&eviction_cond, &eviction_mutex);
        uthread::notify_all(&mutator_cond, &eviction_mutex);
        if (master_evacuation_thread) {
            uthread::join(std::move(master_evacuation_thread));
        }
        // Every normal eviction worker drains its CQ before returning, but a
        // stop can race the boundary between the final post and its CQ poll.
        // Once the master and its fork/join workers have exited, no producer
        // remains. Resolve EVICTING entries before Region reclaim so no late
        // completion can touch recycled memory.
        // Settle ec_batch groups that will never complete before
        // drain_eviction_completions_for_shutdown() scans the heap for EVICTING
        // entries: it is exactly their objects' write reference that keeps them
        // EVICTING, and its timeout would otherwise fire on them.
        settle_ec_batch_in_flight_groups(
            "quiesce_settle_never_completing_groups");
        drain_eviction_completions_for_shutdown();
        full_population_frequency_tracker.stop();
    }

    void begin_resident_profile_window() {
        if (!resident_profile_planner_enabled_flag ||
            !resident_profile_manual_trigger_flag) {
            return;
        }
        const uint32_t owner_limit =
            next_logical_owner_id.load(std::memory_order_acquire);
        for (uint32_t owner_id = 1; owner_id < owner_limit; ++owner_id) {
            auto &profile = logical_object_profiles[owner_id];
            profile.resident_window_references.store(
                0, std::memory_order_relaxed);
            profile.resident_total_references.store(
                0, std::memory_order_relaxed);
        }
        resident_profile_active.store(true, std::memory_order_release);
        std::cerr << "resident.profile.begin mode=manual owners="
                  << (owner_limit - 1) << std::endl;
    }

    void publish_resident_profile_plan_now() {
        if (!resident_profile_planner_enabled_flag ||
            !resident_profile_manual_trigger_flag ||
            !resident_profile_active.load(std::memory_order_acquire)) {
            return;
        }
        publish_resident_profile_plan();
    }

    static FarObjectEntry &get_entry_of(far_obj_t obj) {
        return *obj.get_entry_ptr();
    }

    static ConcurrentArrayCache *init_default(void *local_buf,
                                              size_t local_buf_size,
                                              size_t remote_buf_size,
                                              size_t evict_batch_size) {
        default_instance.reset(new ConcurrentArrayCache(
            local_buf, local_buf_size, remote_buf_size, evict_batch_size));
        return default_instance.get();
    }

    static ConcurrentArrayCache *get_default() {
        return default_instance.get();
    }

    static void quiesce_default() {
        if (default_instance != nullptr) {
            default_instance->ec_batch_diag_step("cache_quiesce_default_begin");
        }
        if (default_instance) {
            default_instance->quiesce_background_evacuation();
        }
    }


    static void destroy_default() {
        // ec_batch teardown trace (no effect unless ft_method=ec_batch): this is
        // the last cache-side step before the rdma client singleton is reset
        // and INFO("destroying rdma client") is printed.
        if (default_instance != nullptr) {
            default_instance->ec_batch_diag_step("cache_destroy_default");
        }
        default_instance.reset();
    }

    void register_live_entry(FarObjectEntry *entry);
    void unregister_live_entry(FarObjectEntry *entry);
    void move_live_entry(FarObjectEntry *from, FarObjectEntry *to);

    struct LogicalObjectProfileSnapshot {
        uint64_t allocated_objects;
        uint64_t allocated_bytes;
        uint64_t allocated_footprint_bytes;
        uint64_t fetches;
        uint64_t clean_evictions;
        uint64_t dirty_evictions;
        uint64_t resident_references;
        uint64_t resident_current_bytes;
        uint64_t resident_target_bytes;
        uint32_t score_pct;
    };

    uint32_t register_logical_object() {
        if (!logical_object_profile_enabled()) {
            return 0;
        }
        const uint32_t id =
            next_logical_owner_id.fetch_add(1, std::memory_order_relaxed);
        ASSERT(id < logical_object_profiles.size());
        return id;
    }

    void record_logical_object_allocation(uint32_t owner_id,
                                          uint64_t objects,
                                          uint64_t bytes) {
        if (owner_id == 0) {
            return;
        }
        ASSERT(owner_id < logical_object_profiles.size());
        auto &profile = logical_object_profiles[owner_id];
        profile.allocated_objects.fetch_add(objects, std::memory_order_relaxed);
        profile.allocated_bytes.fetch_add(bytes, std::memory_order_relaxed);
        if (objects != 0) {
            ASSERT(bytes % objects == 0);
            const uint64_t footprint =
                objects * local_allocation_footprint(bytes / objects);
            profile.allocated_footprint_bytes.fetch_add(
                footprint, std::memory_order_relaxed);
        }
    }

    LogicalObjectProfileSnapshot logical_object_profile_snapshot(
        uint32_t owner_id) const {
        ASSERT(owner_id > 0 && owner_id < logical_object_profiles.size());
        const auto &profile = logical_object_profiles[owner_id];
        return {.allocated_objects = profile.allocated_objects.load(),
                .allocated_bytes = profile.allocated_bytes.load(),
                .allocated_footprint_bytes =
                    profile.allocated_footprint_bytes.load(),
                .fetches = profile.total_fetches.load(),
                .clean_evictions = profile.total_clean_evictions.load(),
                .dirty_evictions = profile.total_dirty_evictions.load(),
                .resident_references =
                    profile.resident_total_references.load(),
                .resident_current_bytes =
                    profile.resident_current_bytes.load(),
                .resident_target_bytes =
                    profile.resident_target_bytes.load(),
                .score_pct = profile.published_score_pct.load()};
    }

    struct ResidentPlacementGroupSnapshot {
        uint32_t logical_owner_id;
        uint32_t allocation_bin;
        uint64_t live_objects;
        uint64_t live_footprint_bytes;
        uint64_t window_read_references;
        uint64_t window_write_references;
        uint64_t ema_read_references;
        uint64_t ema_write_references;
        uint64_t read_references;
        uint64_t write_references;
        uint64_t resident_current_bytes;
        uint64_t resident_target_bytes;
    };

    uint32_t resident_placement_group_count() const {
        return next_resident_placement_group_id.load(
                   std::memory_order_acquire) -
               1;
    }

    ResidentPlacementGroupSnapshot resident_placement_group_snapshot(
        uint32_t group_id) const {
        ASSERT(group_id > 0 &&
               group_id < next_resident_placement_group_id.load(
                              std::memory_order_acquire));
        const auto &profile = resident_placement_groups[group_id];
        return {
            .logical_owner_id = profile.logical_owner_id,
            .allocation_bin = profile.allocation_bin,
            .live_objects = profile.live_objects.load(),
            .live_footprint_bytes = profile.live_footprint_bytes.load(),
            .window_read_references =
                profile.window_read_references.load(),
            .window_write_references =
                profile.window_write_references.load(),
            .ema_read_references = profile.ema_read_references.load(),
            .ema_write_references = profile.ema_write_references.load(),
            .read_references = profile.total_read_references.load(),
            .write_references = profile.total_write_references.load(),
            .resident_current_bytes =
                profile.resident_current_bytes.load(),
            .resident_target_bytes = profile.resident_target_bytes.load(),
        };
    }

    struct ResidentProfilePlannerSnapshot {
        uint64_t check_count;
        uint64_t plan_count;
        uint64_t published_plan_count;
        uint64_t candidate_rejected_count;
        uint64_t stalled_promotion_cancelled_bytes;
        uint64_t stalled_promotion_cancelled_groups;
        uint64_t prevented_reversal_count;
        uint64_t publish_while_unsafe_count;
        uint64_t convergence_gate_count;
        uint64_t convergence_gate_pending_bytes_total;
        uint64_t convergence_gate_pending_bytes_last;
        uint64_t migrations_inflight;
        bool plan_ready;
    };

    ResidentProfilePlannerSnapshot resident_profile_planner_snapshot() const {
        return {
            .check_count = resident_profile_plan_check_count.load(),
            .plan_count = resident_profile_plan_count.load(),
            .published_plan_count =
                resident_profile_published_plan_count.load(),
            .candidate_rejected_count =
                resident_profile_candidate_rejected_count.load(),
            .stalled_promotion_cancelled_bytes =
                resident_profile_stalled_promotion_cancelled_bytes.load(),
            .stalled_promotion_cancelled_groups =
                resident_profile_stalled_promotion_cancelled_groups.load(),
            .prevented_reversal_count =
                resident_profile_prevented_reversal_count.load(),
            .publish_while_unsafe_count =
                resident_profile_publish_while_unsafe_count.load(),
            .convergence_gate_count =
                resident_profile_convergence_gate_count.load(),
            .convergence_gate_pending_bytes_total =
                resident_profile_convergence_gate_pending_bytes_total.load(),
            .convergence_gate_pending_bytes_last =
                resident_profile_convergence_gate_pending_bytes_last.load(),
            .migrations_inflight = resident_profile_migrations_inflight(),
            .plan_ready =
                resident_profile_plan_ready.load(std::memory_order_acquire),
        };
    }

    void publish_resident_group_plan_now(bool complete_phase = false) {
        ASSERT(automatic_resident_group_planner_enabled());
        // Automatic periodic planning and application-triggered phase planning
        // are mutually exclusive. Allowing both callers concurrently can
        // publish two large, conflicting Region exchanges from the same
        // window. Fail fast instead of silently creating a second planner.
        ASSERT(resident_profile_phase_triggered_enabled());
        // A complete-phase publication consumes the window immediately after
        // profile::end_work(), when sampling has already stopped. Non-complete
        // manual publications still require an active work phase.
        if (!complete_phase && !resident_profile_work_phase_active()) {
            return;
        }
        publish_resident_group_plan(complete_phase);
    }

    bool freeze_resident_group_plan_now(bool allow_partial = false) {
        ASSERT(automatic_resident_group_planner_enabled());
        resident_profile_close_migration_gate();
        const uint64_t plan_id = resident_profile_active_plan_id.load(
            std::memory_order_acquire);
        const uint64_t scheduled_promotion =
            resident_profile_active_plan_scheduled_promotion_bytes.load(
                std::memory_order_acquire);
        const uint64_t scheduled_demotion =
            resident_profile_active_plan_scheduled_demotion_bytes.load(
                std::memory_order_acquire);
        const uint64_t committed_promotion =
            resident_profile_active_plan_promotion_committed_bytes.load(
                std::memory_order_acquire);
        const uint64_t committed_demotion =
            resident_profile_active_plan_demotion_committed_bytes.load(
                std::memory_order_acquire);
        const bool converged = plan_id != 0 &&
            committed_promotion == scheduled_promotion &&
            committed_demotion == scheduled_demotion;
        const bool partially_converged = plan_id != 0 && allow_partial &&
            committed_demotion == scheduled_demotion;
        const bool frozen = converged || partially_converged;
        if (!frozen) {
            resident_profile_open_migration_gate();
        }
        return frozen;
    }

    void add_resident_group_profile_sample_for_test(
        uint32_t group_id, uint64_t read_references,
        uint64_t write_references) {
        ASSERT(automatic_resident_group_planner_enabled());
        ASSERT(group_id > 0 &&
               group_id < next_resident_placement_group_id.load(
                              std::memory_order_acquire));
        auto &profile = resident_placement_groups[group_id];
        profile.window_read_references.fetch_add(
            read_references, std::memory_order_relaxed);
        profile.window_write_references.fetch_add(
            write_references, std::memory_order_relaxed);
        profile.total_read_references.fetch_add(
            read_references, std::memory_order_relaxed);
        profile.total_write_references.fetch_add(
            write_references, std::memory_order_relaxed);
    }

    template <bool Lite = false>
    std::pair<far_obj_t, void *> allocate(FarObjectEntry *entry, size_t size,
                                          bool dirty, DereferenceScope &scope,
                                          uint32_t logical_owner_id = 0) {
        far_obj_t obj = {.size = size,
                         .obj_id = reinterpret_cast<uint64_t>(entry)};
        const size_t resident_footprint = local_allocation_footprint(size);
        const uint32_t resident_group_id = register_resident_placement_group(
            logical_owner_id, size, resident_footprint);
        bool resident = false;
        ::FarLib::allocator::RegionPlacement requested_placement =
            ::FarLib::allocator::RegionPlacement::Unclassified;
        if (region_resident_placement_enabled()) {
            if (region_hotness_placement_enabled()) {
                // Cold start fills R in allocation order while leaving an
                // optional one-time headroom for later metadata-guided
                // fetches. During measured work, new groups default to S;
                // physical Region exchange remains the planner's migration
                // mechanism for already-local data.
                if (!profile::is_working()) {
                    const size_t hard_budget =
                        ::FarLib::allocator::global_heap
                            .get_resident_region_budget();
                    const size_t headroom =
                        region_fetch_hotness_placement_enabled()
                            ? hard_budget *
                                  std::min<size_t>(
                                      100,
                                      ::FarLib::get_config()
                                          .region_fetch_resident_headroom_pct) /
                                  100
                            : 0;
                    const size_t initial_budget =
                        hard_budget > headroom ? hard_budget - headroom : 0;
                    requested_placement =
                        ::FarLib::allocator::global_heap
                                    .get_resident_reserved_regions() <
                                initial_budget
                            ? ::FarLib::allocator::RegionPlacement::Resident
                            : ::FarLib::allocator::RegionPlacement::Streaming;
                } else {
                    requested_placement =
                        ::FarLib::allocator::RegionPlacement::Streaming;
                }
            } else if (resident_group_id == 0) {
                requested_placement =
                    ::FarLib::allocator::RegionPlacement::Resident;
            } else {
                const auto &group =
                    resident_placement_groups[resident_group_id];
                const uint64_t target = group.resident_target_bytes.load(
                    std::memory_order_acquire);
                const uint64_t current = group.resident_current_bytes.load(
                    std::memory_order_relaxed);
                requested_placement =
                    !resident_profile_plan_ready.load(
                        std::memory_order_acquire) ||
                            current < target
                        ? ::FarLib::allocator::RegionPlacement::Resident
                        : ::FarLib::allocator::RegionPlacement::Streaming;
            }
        } else {
            resident = try_reserve_resident_local(resident_footprint);
        }
        const uint32_t allocation_group_id =
            region_group_binding_enabled() ? resident_group_id : 0;
        uint32_t behavior_group = 0;
        if (::FarLib::simple_region_budget::six_enabled()) {
            behavior_group = initial_simple_behavior_group();
        } else if (::FarLib::allocator::six_group::enabled()) {
            const auto bin = ::FarLib::allocator::bin_from_wsize(
                ::FarLib::allocator::wsize_from_size(size + sizeof(::FarLib::allocator::BlockHead)));
            behavior_group = ::FarLib::allocator::six_group::registry().initial_group(logical_owner_id, bin);
        }
        auto allocation = allocate_local(size, obj, scope,
                                         requested_placement,
                                         allocation_group_id, behavior_group);
        void* local_ptr = allocation.get();
        if (::FarLib::simple_region_heat::remote_grouping_enabled()) {
            // The first remote copy is created before the object has an
            // entry-level heat hint.  Inherit only the physical local
            // Region's hot/cold dimension: allocator fallback must not
            // replace the new object's Low dirtiness prior with Medium.
            behavior_group = initial_simple_behavior_group(
                ::FarLib::simple_region_heat::is_hot(
                    ::FarLib::simple_region_heat::class_for(
                        reinterpret_cast<uintptr_t>(local_ptr))));
        }
        if (region_resident_placement_enabled()) {
            auto *block =
                static_cast<::FarLib::allocator::BlockHead *>(local_ptr) - 1;
            resident = ::FarLib::allocator::block_to_region(block)
                           ->load_placement() ==
                ::FarLib::allocator::RegionPlacement::Resident;
        }
        uint64_t remote_ptr;
        if (::FarLib::get_config().exclusive_cache) {
            remote_ptr = FarObjectEntry::RemoteAddrInvalid48;
        } else {
            remote_ptr = allocate_remote(size, behavior_group);
        }
        entry->reset<Lite>(local_ptr, remote_ptr, size, dirty, resident,
                           logical_owner_id, resident_group_id);
        if (::FarLib::simple_region_heat::grouping_enabled()) {
            const auto remote_class =
                ::FarLib::simple_region_heat::remote_grouping_enabled() &&
                        remote_ptr != FarObjectEntry::RemoteAddrInvalid48
                    ? ::FarLib::allocator::remote::remote_global_heap
                          .simple_class_for(remote_ptr)
                    : behavior_group;
            entry->set_simple_heat_hot(
                ::FarLib::simple_region_heat::is_hot(
                    ::FarLib::simple_region_heat::fetch_class_hint(
                        behavior_group, remote_class)));
        }
        six_bind_local(*entry, local_ptr, size);
        trace_object_allocation(*entry, size);
        if (::FarLib::simple_dirty_observer::objects_enabled())
            entry->set_object_trace_slot(
                ::FarLib::simple_dirty_observer::monitor().register_object(
                    size, reinterpret_cast<uintptr_t>(local_ptr),
                    entry->simple_dirty_score_q(), entry->simple_dirty_score_known(),
                    entry->simple_heat_hot()));
        if (!region_group_binding_enabled() &&
            resident_group_id != 0 && resident) {
            resident_placement_groups[resident_group_id]
                .resident_current_bytes.fetch_add(
                    resident_footprint, std::memory_order_relaxed);
        }
        if (!region_group_binding_enabled() && resident &&
            logical_owner_id != 0) {
            ASSERT(logical_owner_id < logical_object_profiles.size());
            logical_object_profiles[logical_owner_id]
                .resident_current_bytes.fetch_add(
                    resident_footprint, std::memory_order_relaxed);
        }
        register_live_entry(entry);
        return {obj, local_ptr};
    }

    template <bool Lite = false>
    std::pair<far_obj_t, void *> allocate(size_t size, bool dirty,
                                          DereferenceScope &scope) {
        FarObjectEntry *entry = allocate_entry(scope);
        return allocate<Lite>(entry, size, dirty, scope);
    }

    FarObjectEntry *allocate_entry(DereferenceScope &scope) {
        return new FarObjectEntry;
    }

    void deallocate_entry(FarObjectEntry *entry) { delete entry; }

    template <bool OldAccessor = false>
    void deallocate(far_obj_t obj) {
        deallocate<true, OldAccessor>(get_entry_of(obj), obj.size);
    }

    void deallocate_unique(FarObjectEntry &entry, size_t size) {
        deallocate<false>(entry, size);
    }

    bool at_local(far_obj_t obj) {
        return obj.is_null() || get_entry_of(obj).is_local();
    }

    std::pair<bool, void *> async_fetch(far_obj_t obj,
                                        DereferenceScope &scope) {
        bool at_local = post_fetch<true>(obj, scope);
        return {at_local, get_entry_of(obj).local_addr()};
    }

    bool prefetch(far_obj_t obj, DereferenceScope &scope) {
        profile::start_prefetch();
        struct PrefetchProfileGuard {
            bool previous = resident_profile_prefetch_active;
            PrefetchProfileGuard() {
                resident_profile_prefetch_active = true;
            }
            ~PrefetchProfileGuard() {
                resident_profile_prefetch_active = previous;
            }
        } prefetch_profile_guard;
        bool at_local = post_fetch<false>(obj, scope);
        profile::end_prefetch();
        return at_local;
    }

    void *sync_fetch(far_obj_t obj, DereferenceScope &scope) {
        ON_MISS_BEGIN
        ON_MISS_END
        return fetch_with_miss_handler(obj, __on_miss__, scope);
    }

    bool check_fetch(FarObjectEntry *entry, fetch_ddl_t &ddl);


    void *fetch_with_miss_handler(far_obj_t obj, const DataMissHandler &handler,
                                  DereferenceScope &scope);
    void fetch_wait_until_local(FarObjectEntry *entry, far_obj_t obj,
                                size_t qp_idx, size_t client_idx,
                                size_t endpoint_idx);
public:
    static constexpr const char *kProfilingMethodDisabled = "disabled";
    static constexpr const char *kProfilingMethodFineGrained = "fine_grained";
    static constexpr const char *kProfilingMethodCoarseGrained =
        "coarse_grained";
    static const std::string &profiling_method();
    static bool hybrid_profiling_enabled();
    static bool hybrid_profile_object_access_enabled();
    static bool hybrid_profile_dereference_access_enabled();
    static uint32_t local_fast_path_sample_weight();
    static bool should_sample_local_fast_path();
    static profile::ReferenceHeatClass
    published_reference_heat_class(const FarObjectEntry &entry);
    void record_reference_access(const FarObjectEntry &entry,
                                 profile::ReferenceKind kind,
                                 int64_t weight = 1) const;
    void record_local_fast_path_reference(const FarObjectEntry &entry,
                                          profile::ReferenceKind kind) const;
    void record_non_fast_path_reference(const FarObjectEntry &entry,
                                        profile::ReferenceKind kind) const;
    void record_local_fast_path_access(const FarObjectEntry &entry,
                                       profile::ReferenceKind kind) const;
    void record_non_fast_path_access(const FarObjectEntry &entry,
                                     profile::ReferenceKind kind) const;
    FrequencyProfileMarkResult update_frequency_profile_on_mark(
        const FarObjectEntry &entry, bool update_ema_this_pass) const;

    // do not pin object, to reduce CAS
    template <bool Mut>
    void *fetch_lite(far_obj_t obj, const DataMissHandler &handler,
                     DereferenceScope &scope);

    template <bool Mut>
    void *fetch_lite_no_profile(far_obj_t obj, const DataMissHandler &handler,
                                DereferenceScope &scope);

    static constexpr int STAT_THRESHOLD = 256;

    // main path for local memory miss & fetch from remote.
    template <bool Mut>
    void *fetch_lite_slow_path(far_obj_t obj, const DataMissHandler &handler,
                               DereferenceScope &scope);

    template <bool Mut>
    std::pair<bool, void *> async_fetch_lite(far_obj_t obj,
                                             DereferenceScope &scope);

    template <bool Mut>
    std::pair<bool, void *> async_fetch_lite_no_profile(
        far_obj_t obj, DereferenceScope &scope);

    void pin(far_obj_t obj);
    void unpin(far_obj_t obj);
    void mark_dirty(far_obj_t obj);
    void release_cache(FarObjectEntry *entry, bool dirty);
    void release_cache(far_obj_t obj, bool dirty);
    void enter_scope();
    void exit_scope();
    bool evacuator_waiting();
    void update_scope(DereferenceScope &scope);

public:
    // EC (ft_method=sponge) completion path.  complete_evict_writeback() is
    // the shared EVICTING -> REMOTE release used by both the RDMA-write
    // completion and the sponge commit ACK; the sponge entry points are only
    // reached when ft_method != none (see handle_work_complete()).
    void complete_evict_writeback(void *local_ptr);
    void handle_sponge_commit_ack(uint64_t wr_id, uint16_t status);
    bool handle_sponge_ack_complete(const ibv_wc &wc);
    void check_sponge_ack_cq_ft();
    void handle_rdma_write_complete(const ibv_wc &wc);
    // `release_read_pin` is false only for the synthetic completion emitted
    // after an EC degraded read rebuilds the object.  The original ordinary
    // READ (and its error WC) owns the allocator pin in that case.
    void handle_rdma_read_complete(const ibv_wc &wc,
                                   bool release_read_pin = true);
    void handle_work_complete(const ibv_wc &wc);

    // --- EC (ft_method=ec_batch) group write path -------------------------
    // All of these are inert while ft_method != ec_batch: the staging range is
    // only carved out by the constructor in that mode, post_ec_batch_group()
    // refuses to post anything otherwise, and no ec_batch wr_id can exist.
    size_t init_ec_batch_staging(void *local_buf, size_t local_buf_size);
    bool post_ec_batch_group(const ec_batch::EcGroupSendRecord &record,
                             size_t client_idx, size_t qp_idx);
    size_t post_ec_batch_pending(ec_batch::EcGroupBuilder &builder,
                                 size_t client_idx, size_t qp_idx);
    // Segment completion of one group write; called from
    // handle_rdma_write_complete() when the wr_id carries the ec_batch tag.
    void handle_ec_batch_write_complete(uint64_t wr_id, bool success = true);
    // Group staging used by try_evict(): ec_batch_staging_ready() reports
    // whether a small object may be grouped (ft_method=ec_batch plus a usable
    // staging pool) and lazily creates the builder; stage_ec_batch_object()
    // copies the object into a data slot of the open group and publishes that
    // segment's address as the object's remote address; the call returns false
    // when the object has to take the flat path instead.  Both are serialized
    // by ec_batch_stage_mutex_ against flush_ec_batch_groups().
    bool ec_batch_staging_ready();
    bool stage_ec_batch_object(void *local_addr, size_t size,
                               FarObjectEntry *entry);
    // Flush point of the group path: seals the partly filled group (holes zero
    // filled) and posts every sealed group.  Called from
    // EvictBufferSet::flush_all().
    void flush_ec_batch_groups(size_t client_idx);

    // --- EC (ft_method=ec_batch) degraded READ path (read-side recovery) ----
    // All of these are inert while ft_method != ec_batch: the scratch range is
    // only carved out in that mode, every entry point tests
    // is_ec_batch_mode() first, and no tagged read wr_id can exist otherwise.
    //
    //   init_ec_read_recovery()   sizes the endpoint liveness bitmap from
    //                             config.server_count and carves the resident
    //                             scratch pool out of the client MR;
    //   note_ec_recovery_error_wc()  remembers the endpoint of a failed CQE
    //                             (called from handle_work_complete());
    //   ec_recovery_is_read_wr_id()  the wr_id classifier of the read round,
    //                             checked before handle_rdma_read_complete();
    //   ec_recovery_endpoint_is_dead() / ec_recovery_endpoint_count();
    //   post_ec_degraded_read()/post_ec_degraded_read_entry()  post the reads of
    //                             the five surviving segments of a group;
    //   handle_ec_read_segment_complete()  completion of one tagged read;
    //   ec_recovery_assist_wait()  the in-flight re-post hook of the fetch
    //                             wait/check loops.
    size_t init_ec_read_recovery(void *local_buf, size_t local_buf_size);
    void note_ec_recovery_error_wc(const ibv_wc &wc);
    bool ec_recovery_is_read_wr_id(uint64_t wr_id) const;
    bool ec_recovery_endpoint_is_dead(size_t endpoint_idx) const;
    // Reuse paths may only keep a clean remote copy when the entry's own
    // current endpoint is still alive.  The check is deliberately gated on
    // ec_batch and maps this address (rather than inspecting other segments
    // in its EC group), so a dead parity/data peer does not invalidate a
    // healthy object backup.
    bool reuse_ec_recovery_endpoint_is_dead(uint64_t remote_addr) const {
        const auto &config = ::FarLib::get_config();
        if (!config.is_ec_batch_mode() || config.server_count <= 1 ||
            remote_addr == FarObjectEntry::RemoteAddrInvalid48) {
            return false;
        }
        return ec_recovery_endpoint_is_dead(
            config.map_remote_addr(remote_addr).first);
    }
    size_t ec_recovery_endpoint_count() const;
    int ec_recovery_endpoint_for_qp_num(uint32_t qp_num) const;
    ec_read_recovery::EcReadPostResult post_ec_degraded_read(
        far_obj_t obj, size_t client_idx, void *local_addr);
    ec_read_recovery::EcReadPostResult post_ec_degraded_read_entry(
        FarObjectEntry *entry, size_t client_idx,
        void *expected_local_addr = nullptr);
    // Reverse map of one entry's remote address to its slot group + read plan.
    bool ec_recovery_group_view(FarObjectEntry *entry, uint32_t byte_count,
                                EcRecoveryGroupView *view_out);
    bool ec_recovery_drain_normal_reads_once(
        ::FarLib::allocator::BlockHead *block);
    void handle_ec_read_segment_complete(uint64_t wr_id);
    bool ec_recovery_assist_wait(FarObjectEntry *entry, size_t qp_idx,
                                 size_t client_idx);
    void ec_read_recovery_diag_report(const char *where);

    // Observability of the degraded read path / of the write-side group
    // allocation gate.  Throttled, flushed lines printed from the events
    // themselves (see ec_read_recovery.ipp); nothing reads their counters.
    static bool ec_recovery_throttled(std::atomic<uint64_t> &counter,
                                      uint64_t first, uint64_t every);
    size_t ec_recovery_dead_endpoint_count() const;
    void ec_recovery_report_counters();
    void ec_recovery_report_degraded_read(FarObjectEntry *entry,
                                          const EcRecoveryGroupView &view);
    void ec_recovery_report_group_alloc_blocked(const char *reason);
    // Runtime self-check of a rebuilt shard: re-encode the four data shards and
    // compare the two parity shards read back off the disk; counts and prints.
    void ec_recovery_verify_rebuild(const ec_read_recovery::EcReadContext &token);
    void finish_ec_read_context(uint64_t token_id,
                               const ec_read_recovery::EcReadContextEvent &event);

    // Shutdown side of the group write path (bodies in
    // cache/core/rdma/ec_batch_path.ipp, all inert unless ft_method=ec_batch):
    //   ec_batch_in_flight_group_count() counts posted plus not yet posted
    //   groups; ec_batch_poll_all_cqs_once() polls every CQ of every client and
    //   QP (full_checker() only walks QP index 0);
    //   drain_ec_batch_in_flight_groups_for_shutdown() is the bounded drain at
    //   the start of quiesce, settle_ec_batch_in_flight_groups() writes off what
    //   can no longer complete (quiesced cache only).
    size_t ec_batch_in_flight_group_count();
    size_t ec_batch_poll_all_cqs_once();
    void drain_ec_batch_in_flight_groups_for_shutdown();
    size_t settle_ec_batch_in_flight_groups(const char *reason);
    //   ec_batch_report_stuck_spin() is the rate-limited status line of the
    //   ref_cnt spin inside deallocate(): prints the entry state, a reverse
    //   lookup of the entry's remote address in the stripe manager (is it a
    //   group data slot; group id / slot index / group live count / state) and
    //   the EC in-flight counters.  Diagnostics only - it reads and prints, it
    //   never waits, settles or releases anything.
    void ec_batch_report_stuck_spin(FarObjectEntry &entry, uint64_t spin,
                                    uint64_t harvested, const char *where,
                                    bool damp_enter_report = false);
    // ec_batch diagnostics (observability only; bodies in
    // cache/core/rdma/ec_batch_path.ipp).
    //   ec_batch_diag_note_status() classifies the EcBatchStatus of one
    //   add_object() call - this is also how a refused allocate_slot_group() is
    //   classified, because that builder call returns bool only: the reason
    //   codes reachable here are kManagerRejected (allocate_slot_group()
    //   refused), kStagingExhausted (refused before the allocator was called)
    //   and kObjectTooLarge (allocated, but the group's slot_size was too
    //   small and the group was handed back).
    //   ec_batch_diag_step()/ec_batch_diag_report() print the teardown trace and
    //   the counter set; both print only while ft_method=ec_batch.
    void ec_batch_diag_note_status(ec_batch::EcBatchStatus status);
    uint64_t ec_batch_diag_pending_groups();
    void ec_batch_diag_step(const char *step);
    void ec_batch_diag_report(const char *where);
    ec_batch::EcStagingPool &ec_batch_staging_pool() {
        return ec_staging_pool_;
    }
    const ec_batch::EcBatchTokenTable &ec_batch_tokens() const {
        return ec_batch_tokens_;
    }
    const ec_read_recovery::EcReadContextPool &ec_read_tokens() const {
        return ec_read_tokens_;
    }
    const ec_read_recovery::EcRecoveryScratchPool &ec_read_scratch_pool() const {
        return ec_read_scratch_pool_;
    }
    size_t ec_batch_staging_reserved_bytes() const {
        return ec_staging_reserved_bytes_;
    }
    size_t ec_batch_staging_slot_size() const { return ec_staging_slot_size_; }
    uint64_t ec_batch_groups_posted() const {
        return ec_batch_groups_posted_.load(std::memory_order_relaxed);
    }
    uint64_t ec_batch_groups_completed() const {
        return ec_batch_groups_completed_.load(std::memory_order_relaxed);
    }

public:
    bool memory_low() {
        return ::FarLib::allocator::global_heap.memory_low();
    }

    // Diagnostics-only lifetime fence: the benchmark calls this before its
    // object metadata vectors are cleared so the observer cannot dereference
    // a stale pending-entry pointer during teardown.
    static void stop_default_read_supply_timeline_for_diagnostics() {
        if (default_instance) {
            default_instance->stop_read_supply_timeline();
        }
    }

public:
    size_t check_cq();
    size_t check_cq_idx(size_t qp_idx);
    size_t check_cq_idx_with_client_idx(size_t qp_idx, size_t thread_id);
    size_t check_cq_idx_with_client_idx_endpoint(size_t qp_idx,
                                                 size_t thread_id,
                                                 size_t endpoint_idx);
    size_t poll_cq_idx_with_client_idx_endpoint_no_flush(
        size_t qp_idx, size_t thread_id, size_t endpoint_idx);
    void full_checker();
    void check_memory_low(auto &&scope) {
        if (memory_low()) {
            scope.begin_eviction();
            on_demand_invoke_eviction(kAnyAllocBin);
            scope.end_eviction();
        }
    }

    void invoke_eviction() { on_demand_invoke_eviction(kAnyAllocBin); }

    template <typename T, bool Mut>
    friend class LiteAccessor;
private:
    // Append new diagnostic storage so existing counter/metadata offsets stay unchanged.
    std::unique_ptr<BackupUsageShards> bypass_backup_usage;
    std::unique_ptr<BatchedBackupBudget> batched_backup_budget;

    // --- EC (ft_method=ec_batch) staging + completion tokens ---------------
    // Resident staging pool (bound to the reserved tail of the registered
    // client MR) and the bounded group-token table.  Appended here so the
    // offsets of the existing counters/metadata stay unchanged; both are inert
    // while ft_method != ec_batch.
    ec_batch::EcStagingPool ec_staging_pool_;
    ec_batch::EcBatchTokenTable ec_batch_tokens_;
    // The group builder (lazily created on the first grouped eviction; inert
    // and null while ft_method != ec_batch) and the mutex that serializes
    // staging against sealing/posting.
    std::unique_ptr<ec_batch::EcGroupBuilder> ec_group_builder_;
    // Staging can poll completions and yield. A kernel-thread mutex here
    // would block a fibre worker while the lock-owning fibre is suspended.
    uthread::Mutex ec_batch_stage_mutex_;
    size_t ec_staging_reserved_bytes_ = 0;
    size_t ec_staging_slot_size_ = 0;
    std::atomic<uint64_t> ec_batch_groups_posted_{0};
    std::atomic<uint64_t> ec_batch_groups_completed_{0};
    std::atomic<uint64_t> ec_batch_post_rejects_{0};
    std::atomic<uint64_t> ec_batch_staging_release_failures_{0};
    // Last value returned by ec_batch_poll_all_cqs_once() from the ref_cnt
    // spin of deallocate(); reported by ec_batch_report_stuck_spin().
    // Read-only diagnostics, unused outside ft_method=ec_batch.
    std::atomic<uint64_t> ec_batch_last_poll_harvest_{0};
    // --- ec_batch diagnostics (observability only) --------------------------
    // Relaxed counters placed on the existing ec_batch path, plus the two
    // counters of the try_evict() candidate gate.  No decision, no return value
    // and no teardown ordering reads any of them; every print of these values
    // (ec_batch_diag_report / ec_batch_diag_step) is gated on
    // ft_method=ec_batch, so ft_method=none keeps its output byte for byte.
    std::atomic<uint64_t> ec_candidate_checked_{0};
    std::atomic<uint64_t> ec_candidate_true_{0};
    std::atomic<uint64_t> ec_diag_stage_called_{0};
    std::atomic<uint64_t> ec_diag_stage_ok_{0};
    std::atomic<uint64_t> ec_diag_stage_failed_{0};
    std::atomic<uint64_t> ec_diag_stage_fail_not_ready_{0};
    std::atomic<uint64_t> ec_diag_stage_fail_no_client_{0};
    std::atomic<uint64_t> ec_diag_stage_fail_bad_segment_addr_{0};
    std::atomic<uint64_t> ec_diag_stage_fail_rounds_exhausted_{0};
    std::atomic<uint64_t> ec_diag_add_obj_calls_{0};
    std::atomic<uint64_t> ec_diag_add_obj_ok_{0};
    std::atomic<uint64_t> ec_diag_status_group_full_{0};
    std::atomic<uint64_t> ec_diag_status_staging_exhausted_{0};
    std::atomic<uint64_t> ec_diag_status_pending_queue_full_{0};
    std::atomic<uint64_t> ec_diag_status_manager_rejected_{0};
    std::atomic<uint64_t> ec_diag_status_encode_rejected_{0};
    std::atomic<uint64_t> ec_diag_status_seal_rejected_{0};
    std::atomic<uint64_t> ec_diag_status_object_too_large_{0};
    std::atomic<uint64_t> ec_diag_status_invalid_argument_{0};
    std::atomic<uint64_t> ec_diag_flush_calls_{0};
    std::atomic<uint64_t> ec_diag_flush_sealed_ok_{0};
    std::atomic<uint64_t> ec_diag_post_fail_no_mode_or_staging_{0};
    std::atomic<uint64_t> ec_diag_post_fail_not_postable_{0};
    std::atomic<uint64_t> ec_diag_post_fail_no_client_{0};
    std::atomic<uint64_t> ec_diag_post_fail_no_token_{0};
    std::atomic<uint64_t> ec_diag_post_pending_failed_{0};
    std::atomic<uint64_t> ec_diag_post_write_retries_{0};
    std::atomic<uint64_t> ec_diag_complete_cqes_{0};
    std::atomic<uint64_t> ec_diag_complete_not_last_{0};
    std::atomic<uint64_t> ec_diag_complete_last_{0};

    // --- EC (ft_method=ec_batch) read-side recovery ------------------------
    // Endpoint liveness bitmap sized by config.server_count (not by the 6
    // segments of one group) and the bounded degraded-read bookkeeping.
    // Appended here so the offsets of the existing counters/metadata stay
    // unchanged; everything below is inert while ft_method != ec_batch.
    //
    // A dead endpoint is never revived: its RC QP ended in error, the tree has
    // no reset/re-handshake path, and posting to it again would only produce
    // more flushed completions.
    std::unique_ptr<std::atomic<bool>[]> ec_endpoint_dead_;
    size_t ec_endpoint_dead_count_ = 0;
    // Grow-on-demand recovery buffers (separately registered MRs) and tokens.
    ec_read_recovery::EcRecoveryScratchPool ec_read_scratch_pool_;
    ec_read_recovery::EcReadContextPool ec_read_tokens_{
        static_cast<size_t>(::FarLib::get_config().max_thread_cnt) +
        static_cast<size_t>(::FarLib::get_config().evacuate_thread_cnt) + 2};
    // Diagnostics only: relaxed counters of the degraded read path.  No
    // decision reads them and every print is gated on ft_method=ec_batch.
    std::atomic<uint64_t> ec_recovery_dead_endpoints_{0};
    std::atomic<uint64_t> ec_recovery_error_wcs_{0};
    std::atomic<uint64_t> ec_recovery_routed_reads_{0};
    std::atomic<uint64_t> ec_recovery_posts_{0};
    std::atomic<uint64_t> ec_recovery_post_duplicates_{0};
    std::atomic<uint64_t> ec_recovery_post_unavailable_{0};
    std::atomic<uint64_t> ec_recovery_scratch_backpressure_{0};
    std::atomic<uint64_t> ec_recovery_completions_{0};
    std::atomic<uint64_t> ec_recovery_rebuild_failures_{0};
    std::atomic<uint64_t> ec_recovery_abandons_{0};
    std::atomic<uint64_t> ec_recovery_scratch_release_failures_{0};
    std::atomic<uint64_t> ec_recovery_wait_assists_{0};
    // --- Observability of the degraded read / group allocation gate --------
    // The counters above used to be visible only from the cache_dtor_begin
    // report, which a hung teardown never reaches; the new lines are printed
    // from the events themselves.  Pure counters and prints: no decision, no
    // return value and no wait reads any of them, and every print is gated on
    // ft_method=ec_batch, so ft_method=none keeps its output byte for byte.
    std::atomic<uint64_t> ec_recovery_degraded_read_events_{0};
    std::atomic<uint64_t> ec_recovery_rebuilds_{0};
    // Runtime self-check of the rebuilt shards (see
    // ec_recovery_verify_rebuild()): verify_pass + verify_fail count the
    // re-encode-and-compare outcomes, verify_events only throttles the line.
    std::atomic<uint64_t> ec_recovery_verify_pass_{0};
    std::atomic<uint64_t> ec_recovery_verify_fail_{0};
    std::atomic<uint64_t> ec_recovery_verify_events_{0};
    std::atomic<uint64_t> ec_recovery_group_alloc_blocked_{0};
    std::atomic<uint64_t> ec_recovery_last_summary_ms_{0};
};

}  // namespace cache

}  // namespace FarLib

#include "design1/diagnostics.ipp"
#include "design1/fetch_placement.ipp"
#include "design1/placement_planner.ipp"
#include "design2/cache_hooks.ipp"
#include "cache/core/evict/evict_path.ipp"
#include "cache/core/evict/evacuate_path.ipp"
#include "cache/core/profile/profile_path.ipp"
#include "cache/core/fetch/fetch_path.ipp"
#include "cache/core/rdma/rdma_completion.ipp"
#include "cache/core/rdma/ec_batch_path.ipp"
#include "recovery/ec_read_recovery.ipp"
#include "cache/core/common/common_path.ipp"
#include "cache/core/diagnostics/read_supply_timeline.ipp"
