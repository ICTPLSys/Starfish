#pragma once

namespace FarLib::cache {

inline void ConcurrentArrayCache::print_design1_diagnostics() const {
    if (batched_backup_budget) {
        batched_backup_budget->reclaim_idle();
        const auto c = batched_backup_budget->snapshot();
        ASSERT(c.actual_bytes == 0 && c.issued_bytes == 0 && c.cached_bytes == 0);
        ASSERT(retained_backup_bytes.load() == 0 && peak_retained_backup_bytes.load() == 0);
        std::cout << "remote backup peak kind: issued_upper_bound" << std::endl;
        print_nonresident_backup_snapshot("after_cleanup");
    }
    std::cout << "remote backup budget bytes: " << retained_backup_budget_bytes
              << std::endl;
    std::cout << "remote backup current bytes: "
              << backup_usage_snapshot()
              << std::endl;
    std::cout << "remote backup peak bytes: "
              << backup_peak_snapshot()
              << std::endl;
    if (bypass_backup_usage) {
        std::cout << "remote backup peak kind: sampled_only" << std::endl;
        ASSERT(retained_backup_bytes.load() == 0);
        ASSERT(peak_retained_backup_bytes.load() == 0);
        ASSERT(backup_usage_snapshot() == 0);
    }
    std::cout << "remote backup mode: "
              << (object_profiled_backup_mode
                      ? "object_profiled"
                      : (profiled_backup_mode
                             ? "profiled"
                             : (segmented_backup_mode ? "segmented"
                                                      : "greedy")))
              << std::endl;
    std::cout << "resident local budget bytes: " << resident_local_budget_bytes
              << std::endl;
    std::cout << "resident local current bytes: "
              << resident_local_bytes.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident local peak bytes: "
              << peak_resident_local_bytes.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner enabled: "
              << resident_profile_planner_enabled_flag << std::endl;
    std::cout << "resident physical-region hotness placement: "
              << region_hotness_placement_enabled() << std::endl;
    std::cout << "resident object-metadata fetch placement: "
              << region_fetch_hotness_placement_enabled() << std::endl;
    std::cout << "resident object-metadata fetch policy: "
              << design1::to_string(region_fetch_placement_policy) << std::endl;
    std::cout << "resident fetch headroom pct: "
              << ::FarLib::get_config().region_fetch_resident_headroom_pct
              << std::endl;
    std::cout << "resident fetch max window multiplier: "
              << ::FarLib::get_config().region_fetch_plan_max_window_multiplier
              << std::endl;
    std::cout
        << "resident fetch min reclassification efficiency pct: "
        << ::FarLib::get_config().region_fetch_min_reclassify_efficiency_pct
        << std::endl;
    std::cout << "resident planner requires work phase: "
              << resident_profile_require_work_phase_flag << std::endl;
    std::cout << "resident planner bounded freeze after first commit: "
              << resident_profile_stop_after_first_commit() << std::endl;
    std::cout << "resident planner apply plan: "
              << resident_profile_apply_plan_flag << std::endl;
    std::cout << "resident planner manual trigger: "
              << resident_profile_manual_trigger_flag << std::endl;
    std::cout << "resident planner plan check count: "
              << resident_profile_plan_check_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner plan count: "
              << resident_profile_plan_count.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner published plan count: "
              << resident_profile_published_plan_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner convergence gate count: "
              << resident_profile_convergence_gate_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner convergence gate pending bytes total: "
              << resident_profile_convergence_gate_pending_bytes_total.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner convergence gate pending bytes last: "
              << resident_profile_convergence_gate_pending_bytes_last.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner migrations inflight: "
              << resident_profile_migrations_inflight() << std::endl;
    std::cout << "resident planner candidate checks: "
              << resident_profile_candidate_check_count.load() << std::endl;
    std::cout << "resident planner initial sample windows: "
              << resident_profile_initial_sample_windows << std::endl;
    std::cout
        << "resident planner minimum benefit refs per replacement object: "
        << resident_profile_min_benefit_refs_per_replacement_object
        << std::endl;
    std::cout << "resident planner benefit rejected checks: "
              << resident_profile_benefit_rejected_count.load() << std::endl;
    std::cout << "resident planner candidate stable group checks: "
              << resident_profile_candidate_stable_group_checks.load()
              << std::endl;
    std::cout << "resident planner candidate rejected checks: "
              << resident_profile_candidate_rejected_count.load() << std::endl;
    std::cout << "resident planner candidate replacement bytes last: "
              << resident_profile_candidate_replacement_bytes_last.load()
              << std::endl;
    std::cout << "resident planner candidate replacement bytes total: "
              << resident_profile_candidate_replacement_bytes_total.load()
              << std::endl;
    std::cout << "resident planner candidate active target delta bytes last: "
              << resident_profile_candidate_active_delta_bytes_last.load()
              << std::endl;
    std::cout << "resident planner candidate target overlap bytes last: "
              << resident_profile_candidate_target_overlap_bytes_last.load()
              << std::endl;
    std::cout << "resident planner stalled promotion cancelled bytes: "
              << resident_profile_stalled_promotion_cancelled_bytes.load()
              << std::endl;
    std::cout << "resident planner stalled promotion cancelled groups: "
              << resident_profile_stalled_promotion_cancelled_groups.load()
              << std::endl;
    std::cout << "resident planner prevented reversals: "
              << resident_profile_prevented_reversal_count.load() << std::endl;
    std::cout << "resident planner publish while unsafe: "
              << resident_profile_publish_while_unsafe_count.load()
              << std::endl;
    std::cout << "resident planner profile collection us total: "
              << resident_profile_collection_us_total.load() << std::endl;
    std::cout << "resident planner profile collection us max: "
              << resident_profile_collection_us_max.load() << std::endl;
    std::cout << "resident planner candidate build us total: "
              << resident_profile_candidate_build_us_total.load() << std::endl;
    std::cout << "resident planner candidate build us max: "
              << resident_profile_candidate_build_us_max.load() << std::endl;
    std::cout << "resident planner publish us total: "
              << resident_profile_publish_us_total.load() << std::endl;
    std::cout << "resident planner publish us max: "
              << resident_profile_publish_us_max.load() << std::endl;
    std::cout << "resident planner active plan id: "
              << resident_profile_active_plan_id.load() << std::endl;
    std::cout << "resident planner promotion started bytes total: "
              << resident_profile_promotion_started_bytes_total.load()
              << std::endl;
    std::cout << "resident planner demotion started bytes total: "
              << resident_profile_demotion_started_bytes_total.load()
              << std::endl;
    std::cout << "resident planner promotion committed bytes total: "
              << resident_profile_promotion_committed_bytes_total.load()
              << std::endl;
    std::cout << "resident planner demotion committed bytes total: "
              << resident_profile_demotion_committed_bytes_total.load()
              << std::endl;
    std::cout << "resident planner promotion rolled back bytes total: "
              << resident_profile_promotion_rolled_back_bytes_total.load()
              << std::endl;
    std::cout << "resident planner demotion rolled back bytes total: "
              << resident_profile_demotion_rolled_back_bytes_total.load()
              << std::endl;
    std::cout << "resident planner promotion cancelled bytes total: "
              << resident_profile_promotion_cancelled_bytes_total.load()
              << std::endl;
    std::cout << "resident planner demotion cancelled bytes total: "
              << resident_profile_demotion_cancelled_bytes_total.load()
              << std::endl;
    std::cout << "resident planner promotion carried bytes total: "
              << resident_profile_promotion_carried_bytes_total.load()
              << std::endl;
    std::cout << "resident planner demotion carried bytes total: "
              << resident_profile_demotion_carried_bytes_total.load()
              << std::endl;
    std::cout << "resident planner promoted bytes: "
              << resident_profile_promoted_bytes.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner demoted bytes: "
              << resident_profile_demoted_bytes.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident planner released backup bytes: "
              << resident_profile_released_backup_bytes.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "resident transition backup deferred count: "
              << resident_transition_backup_deferred_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "segmented backup wait count: "
              << segmented_backup_wait_count.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "segmented backup wait cycles: "
              << segmented_backup_wait_cycles.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "profiled backup budget-full reject count: "
              << profiled_backup_budget_full_reject_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "profiled backup budget-full reject bytes: "
              << profiled_backup_budget_full_reject_bytes.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "quiesce recovered stale local refs: "
              << quiesce_recovered_stale_local_ref_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch requested resident: "
              << region_fetch_requested_resident_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch requested streaming: "
              << region_fetch_requested_streaming_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch actual resident: "
              << region_fetch_actual_resident_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch actual streaming: "
              << region_fetch_actual_streaming_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch resident-to-streaming fallback count: "
              << region_fetch_resident_to_streaming_fallback_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch resident-to-streaming fallback bytes: "
              << region_fetch_resident_to_streaming_fallback_bytes.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill checks: "
              << region_fetch_reserve_refill_checks.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill attempts: "
              << region_fetch_reserve_refill_attempts.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill successes: "
              << region_fetch_reserve_refill_successes.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill capacity bytes: "
              << region_fetch_reserve_refill_capacity_bytes.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill exchange bytes: "
              << region_fetch_reserve_refill_exchange_bytes.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill promoted intent bytes: "
              << region_fetch_reserve_refill_promoted_intent_bytes.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill cycles: "
              << region_fetch_reserve_refill_cycles.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill no streaming candidate: "
              << region_fetch_reserve_refill_no_streaming_candidate.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch reserve refill no resident cold: "
              << region_fetch_reserve_refill_no_resident_cold.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region fetch metadata plan ready: "
              << resident_region_fetch_plan_ready.load(
                     std::memory_order_relaxed)
              << std::endl;
    if (size_profiled_backup_enabled()) {
        std::cout << "backup profile window events: "
                  << backup_profile_window_events << std::endl;
        std::cout << "backup profile min score pct: "
                  << backup_profile_min_score_pct << std::endl;
        for (size_t i = 0; i < backup_behavior_groups.size(); ++i) {
            const auto &group = backup_behavior_groups[i];
            const uint64_t fetches =
                group.total_fetches.load(std::memory_order_relaxed);
            const uint64_t clean =
                group.total_clean_evictions.load(std::memory_order_relaxed);
            const uint64_t dirty =
                group.total_dirty_evictions.load(std::memory_order_relaxed);
            if (fetches == 0 && clean == 0 && dirty == 0) {
                continue;
            }
            std::cout << "backup profile group " << i << ": fetches=" << fetches
                      << " clean_evictions=" << clean
                      << " dirty_evictions=" << dirty
                      << " clean_pct=" << group.published_clean_pct.load()
                      << " refetch_pct=" << group.published_refetch_pct.load()
                      << " score_pct=" << group.published_score_pct.load()
                      << " admitted=" << group.admitted_fetches.load()
                      << " rejected=" << group.rejected_fetches.load()
                      << std::endl;
        }
    }
    if (logical_object_profile_enabled()) {
        std::cout << "logical object profile window events: "
                  << backup_profile_window_events << std::endl;
        std::cout << "logical object profile sample shift: "
                  << logical_object_profile_sample_shift << std::endl;
        for (uint32_t id = 1;
             id < next_logical_owner_id.load(std::memory_order_relaxed); ++id) {
            const auto &profile = logical_object_profiles[id];
            std::cout << "logical object " << id << ": allocated_objects="
                      << profile.allocated_objects.load()
                      << " allocated_bytes=" << profile.allocated_bytes.load()
                      << " allocated_footprint_bytes="
                      << profile.allocated_footprint_bytes.load()
                      << " fetches=" << profile.total_fetches.load()
                      << " clean_evictions="
                      << profile.total_clean_evictions.load()
                      << " dirty_evictions="
                      << profile.total_dirty_evictions.load()
                      << " clean_pct=" << profile.published_clean_pct.load()
                      << " refetch_pct=" << profile.published_refetch_pct.load()
                      << " score_pct=" << profile.published_score_pct.load()
                      << " admitted=" << profile.admitted_fetches.load()
                      << " rejected=" << profile.rejected_fetches.load()
                      << " resident_references="
                      << profile.resident_total_references.load()
                      << " resident_current_bytes="
                      << profile.resident_current_bytes.load()
                      << " resident_target_bytes="
                      << profile.resident_target_bytes.load() << std::endl;
        }
    }
}

}  // namespace FarLib::cache
