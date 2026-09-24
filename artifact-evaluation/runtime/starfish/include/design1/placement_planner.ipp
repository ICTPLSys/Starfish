#pragma once

namespace FarLib::cache {

inline void ConcurrentArrayCache::publish_region_fetch_hotness_predictions(
    uint64_t check_id) {
    if (!region_fetch_hotness_placement_enabled()) {
        return;
    }
    if (resident_region_fetch_plan_ready.load(std::memory_order_acquire) &&
        check_id < resident_region_fetch_next_plan_check) {
        return;
    }
    struct FetchCandidate {
        uint32_t group_id;
        uint64_t weighted_references;
        uint64_t live_bytes;
        uint64_t resident_gain_byte_work;
        uint64_t streaming_gain_byte_work;
        long double resident_advantage_per_live_byte;
        bool currently_resident_preferred;
        bool desired_resident{false};
        bool stable_desired{false};
    };
    const bool use_design1_gain =
        region_fetch_placement_policy == design1::PlacementPolicy::Design1Gain;

    const uint64_t collection_start = get_cycles();
    const uint32_t group_limit =
        next_resident_placement_group_id.load(std::memory_order_acquire);
    std::vector<FetchCandidate> candidates;
    candidates.reserve(group_limit > 0 ? group_limit - 1 : 0);
    for (uint32_t group_id = 1; group_id < group_limit; ++group_id) {
        auto &group = resident_placement_groups[group_id];
        const uint64_t window_read =
            group.window_read_references.exchange(0, std::memory_order_acq_rel);
        const uint64_t window_write = group.window_write_references.exchange(
            0, std::memory_order_acq_rel);
        const uint64_t ema_read = resident_profile_ema_update(
            group.ema_read_references.load(std::memory_order_relaxed),
            window_read, resident_profile_ema_decay_shift);
        const uint64_t ema_write = resident_profile_ema_update(
            group.ema_write_references.load(std::memory_order_relaxed),
            window_write, resident_profile_ema_decay_shift);
        group.ema_read_references.store(ema_read, std::memory_order_relaxed);
        group.ema_write_references.store(ema_write, std::memory_order_relaxed);
        const uint64_t window_fetch_bytes =
            group.window_fetch_bytes.exchange(0, std::memory_order_acq_rel);
        const uint64_t window_clean_evict_bytes =
            group.window_clean_evict_bytes.exchange(0,
                                                    std::memory_order_acq_rel);
        const uint64_t window_dirty_evict_bytes =
            group.window_dirty_evict_bytes.exchange(0,
                                                    std::memory_order_acq_rel);
        // Event counts are lifetime observability; the byte windows drive
        // the gain calculation and are the only values consumed here.
        group.window_fetches.exchange(0, std::memory_order_acq_rel);
        group.window_clean_evictions.exchange(0, std::memory_order_acq_rel);
        group.window_dirty_evictions.exchange(0, std::memory_order_acq_rel);
        const uint64_t ema_fetch_bytes = resident_profile_ema_update(
            group.ema_fetch_bytes.load(std::memory_order_relaxed),
            window_fetch_bytes, resident_profile_ema_decay_shift);
        const uint64_t ema_clean_evict_bytes = resident_profile_ema_update(
            group.ema_clean_evict_bytes.load(std::memory_order_relaxed),
            window_clean_evict_bytes, resident_profile_ema_decay_shift);
        const uint64_t ema_dirty_evict_bytes = resident_profile_ema_update(
            group.ema_dirty_evict_bytes.load(std::memory_order_relaxed),
            window_dirty_evict_bytes, resident_profile_ema_decay_shift);
        group.ema_fetch_bytes.store(ema_fetch_bytes, std::memory_order_relaxed);
        group.ema_clean_evict_bytes.store(ema_clean_evict_bytes,
                                          std::memory_order_relaxed);
        group.ema_dirty_evict_bytes.store(ema_dirty_evict_bytes,
                                          std::memory_order_relaxed);
        const uint64_t live_bytes =
            group.live_footprint_bytes.load(std::memory_order_relaxed);
        const uint64_t live_objects =
            group.live_objects.load(std::memory_order_relaxed);
        if (live_bytes == 0) {
            group.fetch_preferred_resident.store(false,
                                                 std::memory_order_release);
            group.resident_target_bytes.store(0, std::memory_order_release);
            group.fetch_candidate_streak.store(0, std::memory_order_relaxed);
            continue;
        }
        const uint64_t total_fetch_bytes =
            group.total_fetch_bytes.load(std::memory_order_relaxed);
        const uint64_t total_clean_evict_bytes =
            group.total_clean_evict_bytes.load(std::memory_order_relaxed);
        const uint64_t total_dirty_evict_bytes =
            group.total_dirty_evict_bytes.load(std::memory_order_relaxed);
        const auto score = design1::score_group(
            {.ema_read_references = ema_read,
             .ema_write_references = ema_write,
             .write_weight = resident_profile_write_weight,
             .live_bytes = live_bytes,
             .live_objects = live_objects,
             .total_fetch_bytes = total_fetch_bytes,
             .total_clean_evict_bytes = total_clean_evict_bytes,
             .total_dirty_evict_bytes = total_dirty_evict_bytes,
             .ema_clean_evict_bytes = ema_clean_evict_bytes,
             .ema_dirty_evict_bytes = ema_dirty_evict_bytes});
        candidates.push_back(
            {.group_id = group_id,
             .weighted_references = score.weighted_references,
             .live_bytes = live_bytes,
             .resident_gain_byte_work = score.resident_gain_byte_work,
             .streaming_gain_byte_work = score.streaming_gain_byte_work,
             .resident_advantage_per_live_byte =
                 score.resident_advantage_per_live_byte,
             .currently_resident_preferred =
                 group.fetch_preferred_resident.load(
                     std::memory_order_relaxed)});
    }
    profile::count_resident_profile_planner_collection_cycles(get_cycles() -
                                                              collection_start);

    const uint64_t candidate_start = get_cycles();
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [this](const FetchCandidate &left, const FetchCandidate &right) {
            return design1::ranks_before(
                region_fetch_placement_policy,
                {.group_id = left.group_id,
                 .live_bytes = left.live_bytes,
                 .currently_resident_preferred =
                     left.currently_resident_preferred,
                 .score = {.weighted_references = left.weighted_references,
                           .resident_gain_byte_work =
                               left.resident_gain_byte_work,
                           .streaming_gain_byte_work =
                               left.streaming_gain_byte_work,
                           .resident_advantage_per_live_byte =
                               left.resident_advantage_per_live_byte}},
                {.group_id = right.group_id,
                 .live_bytes = right.live_bytes,
                 .currently_resident_preferred =
                     right.currently_resident_preferred,
                 .score = {
                     .weighted_references = right.weighted_references,
                     .resident_gain_byte_work = right.resident_gain_byte_work,
                     .streaming_gain_byte_work = right.streaming_gain_byte_work,
                     .resident_advantage_per_live_byte =
                         right.resident_advantage_per_live_byte}});
        });

    const uint64_t resident_budget_bytes =
        ::FarLib::allocator::global_heap.get_resident_region_budget() *
        ::FarLib::allocator::RegionSize;
    uint64_t remaining_budget = resident_budget_bytes;
    uint64_t desired_bytes = 0;
    uint64_t published_bytes = 0;
    uint64_t desired_groups = 0;
    uint64_t published_groups = 0;
    uint64_t changed_groups = 0;
    uint64_t current_resident_bytes = 0;
    uint64_t target_resident_deficit_bytes = 0;
    uint64_t non_target_resident_bytes = 0;
    uint64_t desired_group_hash = 1469598103934665603ULL;
    const bool warmup_complete =
        check_id > resident_profile_initial_sample_windows;
    const bool had_fetch_plan_before_publish =
        resident_region_fetch_plan_ready.load(std::memory_order_acquire);
    std::array<uint32_t, 4> top_group_ids{};
    std::array<uint64_t, 4> top_resident_gains{};
    std::array<uint64_t, 4> top_streaming_gains{};
    for (size_t i = 0; i < std::min(top_group_ids.size(), candidates.size());
         ++i) {
        top_group_ids[i] = candidates[i].group_id;
        top_resident_gains[i] = candidates[i].resident_gain_byte_work;
        top_streaming_gains[i] = candidates[i].streaming_gain_byte_work;
    }
    for (auto &candidate : candidates) {
        auto &group = resident_placement_groups[candidate.group_id];
        const bool desired = candidate.live_bytes <= remaining_budget;
        candidate.desired_resident = desired;
        if (desired) {
            remaining_budget -= candidate.live_bytes;
            desired_bytes += candidate.live_bytes;
            ++desired_groups;
            desired_group_hash ^= candidate.group_id;
            desired_group_hash *= 1099511628211ULL;
        }

        const bool previous_candidate =
            group.fetch_previous_candidate_resident.exchange(
                desired, std::memory_order_acq_rel);
        uint8_t streak = 1;
        if (previous_candidate == desired) {
            const uint8_t old_streak =
                group.fetch_candidate_streak.load(std::memory_order_relaxed);
            streak = old_streak == std::numeric_limits<uint8_t>::max()
                         ? old_streak
                         : static_cast<uint8_t>(old_streak + 1);
        }
        group.fetch_candidate_streak.store(streak, std::memory_order_relaxed);
        candidate.stable_desired =
            streak >= kResidentDirectionStableComparisons;
        current_resident_bytes +=
            group.resident_current_bytes.load(std::memory_order_relaxed);
    }
    if (warmup_complete) {
        const bool had_fetch_plan = had_fetch_plan_before_publish;
        uint64_t working_published_bytes = 0;
        if (!had_fetch_plan) {
            // First publication installs one complete top-set atomically.
            for (const auto &candidate : candidates) {
                resident_placement_groups[candidate.group_id]
                    .fetch_preferred_resident.store(candidate.desired_resident,
                                                    std::memory_order_relaxed);
                if (candidate.desired_resident) {
                    working_published_bytes += candidate.live_bytes;
                }
            }
        } else {
            for (const auto &candidate : candidates) {
                if (resident_placement_groups[candidate.group_id]
                        .fetch_preferred_resident.load(
                            std::memory_order_relaxed)) {
                    working_published_bytes += candidate.live_bytes;
                }
            }
            // Live group sizes may grow after publication. Preserve the
            // hard R budget by dropping the lowest-ranked groups first.
            for (auto it = candidates.rbegin();
                 it != candidates.rend() &&
                 working_published_bytes > resident_budget_bytes;
                 ++it) {
                auto &group = resident_placement_groups[it->group_id];
                if (!group.fetch_preferred_resident.load(
                        std::memory_order_relaxed)) {
                    continue;
                }
                group.fetch_preferred_resident.store(false,
                                                     std::memory_order_relaxed);
                working_published_bytes -= it->live_bytes;
            }

            std::vector<const FetchCandidate *> challengers;
            std::vector<const FetchCandidate *> incumbents;
            for (const auto &candidate : candidates) {
                const bool preferred =
                    resident_placement_groups[candidate.group_id]
                        .fetch_preferred_resident.load(
                            std::memory_order_relaxed);
                if (candidate.stable_desired && candidate.desired_resident &&
                    !preferred) {
                    challengers.push_back(&candidate);
                } else if (candidate.stable_desired &&
                           !candidate.desired_resident && preferred) {
                    incumbents.push_back(&candidate);
                }
            }
            std::reverse(incumbents.begin(), incumbents.end());
            const uint64_t migration_budget_bytes = std::max<uint64_t>(
                1, resident_budget_bytes *
                       resident_profile_migration_budget_pct / 100);
            uint64_t migrated_bytes = 0;
            const long double hysteresis =
                static_cast<long double>(
                    ::FarLib::get_config()
                        .resident_profile_migration_hysteresis_pct) /
                100.0L;
            const size_t pairs =
                std::min(challengers.size(), incumbents.size());
            for (size_t i = 0; i < pairs; ++i) {
                const auto &challenger = *challengers[i];
                const auto &incumbent = *incumbents[i];
                const long double challenger_score =
                    use_design1_gain
                        ? challenger.resident_advantage_per_live_byte
                        : static_cast<long double>(
                              challenger.weighted_references) /
                              challenger.live_bytes;
                const long double incumbent_score =
                    use_design1_gain
                        ? incumbent.resident_advantage_per_live_byte
                        : static_cast<long double>(
                              incumbent.weighted_references) /
                              incumbent.live_bytes;
                const long double required_score =
                    incumbent_score +
                    std::max(std::abs(incumbent_score), 1.0e-12L) * hysteresis;
                if (challenger_score <= required_score ||
                    migrated_bytes + challenger.live_bytes >
                        migration_budget_bytes) {
                    break;
                }
                auto &old_group = resident_placement_groups[incumbent.group_id];
                auto &new_group =
                    resident_placement_groups[challenger.group_id];
                old_group.fetch_preferred_resident.store(
                    false, std::memory_order_relaxed);
                working_published_bytes -= incumbent.live_bytes;
                if (challenger.live_bytes <=
                    resident_budget_bytes - working_published_bytes) {
                    new_group.fetch_preferred_resident.store(
                        true, std::memory_order_relaxed);
                    working_published_bytes += challenger.live_bytes;
                    migrated_bytes += challenger.live_bytes;
                }
            }
            // Fill genuine unused budget without displacing an incumbent.
            for (const auto &candidate : candidates) {
                if (!candidate.stable_desired || !candidate.desired_resident) {
                    continue;
                }
                auto &group = resident_placement_groups[candidate.group_id];
                if (group.fetch_preferred_resident.load(
                        std::memory_order_relaxed) ||
                    candidate.live_bytes >
                        resident_budget_bytes - working_published_bytes) {
                    continue;
                }
                group.fetch_preferred_resident.store(true,
                                                     std::memory_order_relaxed);
                working_published_bytes += candidate.live_bytes;
            }
        }
        ASSERT(working_published_bytes <= resident_budget_bytes);
        for (const auto &candidate : candidates) {
            auto &group = resident_placement_groups[candidate.group_id];
            const bool preferred =
                group.fetch_preferred_resident.load(std::memory_order_relaxed);
            const bool old = candidate.currently_resident_preferred;
            changed_groups += old != preferred;
            group.resident_target_bytes.store(preferred ? candidate.live_bytes
                                                        : 0,
                                              std::memory_order_release);
            const uint64_t current =
                group.resident_current_bytes.load(std::memory_order_relaxed);
            if (preferred) {
                published_bytes += candidate.live_bytes;
                ++published_groups;
                if (current < candidate.live_bytes) {
                    target_resident_deficit_bytes +=
                        candidate.live_bytes - current;
                }
            } else {
                non_target_resident_bytes += current;
            }
        }
    }
    if (warmup_complete && published_groups != 0) {
        resident_region_fetch_plan_ready.store(true, std::memory_order_release);
    }
    uint64_t observed_reclassification_bytes = 0;
    uint64_t observed_deficit_reduction_bytes = 0;
    const uint64_t material_target_change_threshold = std::max<uint64_t>(
        2, published_groups *
               std::max<size_t>(1, resident_profile_migration_budget_pct) /
               100);
    const uint64_t material_deficit_growth_bytes = std::max<uint64_t>(
        8 * ::FarLib::allocator::RegionSize,
        resident_budget_bytes *
            std::max<size_t>(1, resident_profile_migration_budget_pct) / 100);
    const char *reclassification_transition = "none";
    if (warmup_complete && published_groups != 0) {
        const uint64_t refill_successes =
            region_fetch_reserve_refill_successes.load(
                std::memory_order_relaxed);
        if (had_fetch_plan_before_publish) {
            observed_reclassification_bytes =
                (refill_successes -
                 resident_region_fetch_last_refill_successes) *
                ::FarLib::allocator::RegionSize;
            observed_deficit_reduction_bytes =
                resident_region_fetch_last_target_deficit_bytes >
                        target_resident_deficit_bytes
                    ? resident_region_fetch_last_target_deficit_bytes -
                          target_resident_deficit_bytes
                    : 0;
            if (changed_groups >= material_target_change_threshold) {
                resident_region_fetch_reclassification_enabled.store(
                    true, std::memory_order_release);
                resident_region_fetch_low_efficiency_streak = 0;
                resident_region_fetch_retry_cooldown_windows = 0;
                resident_region_fetch_disabled_deficit_bytes = 0;
                reclassification_transition = "target_change";
            } else if (!resident_region_fetch_reclassification_enabled.load(
                           std::memory_order_acquire)) {
                if (resident_region_fetch_retry_cooldown_windows != 0) {
                    --resident_region_fetch_retry_cooldown_windows;
                }
                const bool deficit_grew_materially =
                    target_resident_deficit_bytes >
                        resident_region_fetch_disabled_deficit_bytes &&
                    target_resident_deficit_bytes -
                            resident_region_fetch_disabled_deficit_bytes >=
                        material_deficit_growth_bytes;
                if (resident_region_fetch_retry_cooldown_windows == 0 &&
                    deficit_grew_materially) {
                    resident_region_fetch_reclassification_enabled.store(
                        true, std::memory_order_release);
                    resident_region_fetch_low_efficiency_streak = 0;
                    resident_region_fetch_disabled_deficit_bytes = 0;
                    reclassification_transition = "deficit_growth";
                }
            } else if (observed_reclassification_bytes != 0) {
                const uint64_t min_efficiency_pct = std::min<size_t>(
                    100, ::FarLib::get_config()
                             .region_fetch_min_reclassify_efficiency_pct);
                const bool efficient =
                    static_cast<__uint128_t>(observed_deficit_reduction_bytes) *
                        100 >=
                    static_cast<__uint128_t>(observed_reclassification_bytes) *
                        min_efficiency_pct;
                if (efficient) {
                    resident_region_fetch_low_efficiency_streak = 0;
                } else {
                    resident_region_fetch_low_efficiency_streak =
                        resident_region_fetch_low_efficiency_streak ==
                                std::numeric_limits<uint8_t>::max()
                            ? resident_region_fetch_low_efficiency_streak
                            : static_cast<uint8_t>(
                                  resident_region_fetch_low_efficiency_streak +
                                  1);
                    if (resident_region_fetch_low_efficiency_streak >= 2) {
                        resident_region_fetch_reclassification_enabled.store(
                            false, std::memory_order_release);
                        resident_region_fetch_disabled_deficit_bytes =
                            target_resident_deficit_bytes;
                        resident_region_fetch_retry_cooldown_windows =
                            static_cast<uint8_t>(std::min<size_t>(
                                std::numeric_limits<uint8_t>::max(),
                                std::max<size_t>(
                                    2, resident_region_fetch_window_multiplier *
                                           2)));
                        reclassification_transition = "low_efficiency";
                    }
                }
            }
        }
        resident_region_fetch_last_target_deficit_bytes =
            target_resident_deficit_bytes;
        resident_region_fetch_last_refill_successes = refill_successes;
    }
    const size_t max_window_multiplier = std::max<size_t>(
        1, ::FarLib::get_config().region_fetch_plan_max_window_multiplier);
    if (warmup_complete && changed_groups >= material_target_change_threshold) {
        resident_region_fetch_window_multiplier = std::min(
            max_window_multiplier, resident_region_fetch_window_multiplier * 2);
        resident_region_fetch_stable_plan_streak = 0;
    } else if (warmup_complete && changed_groups == 0) {
        resident_region_fetch_stable_plan_streak =
            resident_region_fetch_stable_plan_streak ==
                    std::numeric_limits<uint8_t>::max()
                ? resident_region_fetch_stable_plan_streak
                : static_cast<uint8_t>(
                      resident_region_fetch_stable_plan_streak + 1);
        if (resident_region_fetch_stable_plan_streak >= 2 &&
            resident_region_fetch_window_multiplier > 1) {
            resident_region_fetch_window_multiplier = std::max<size_t>(
                1, resident_region_fetch_window_multiplier / 2);
            resident_region_fetch_stable_plan_streak = 0;
        }
    } else if (warmup_complete) {
        resident_region_fetch_stable_plan_streak = 0;
    }
    resident_region_fetch_next_plan_check =
        check_id + resident_region_fetch_window_multiplier;
    profile::count_resident_profile_planner_candidate_cycles(get_cycles() -
                                                             candidate_start);
    std::cerr
        << "resident.region_fetch_plan"
        << " check_id=" << check_id
        << " policy=" << design1::to_string(region_fetch_placement_policy)
        << " groups=" << candidates.size()
        << " desired_groups=" << desired_groups
        << " desired_bytes=" << desired_bytes
        << " published_groups=" << published_groups
        << " published_bytes=" << published_bytes
        << " current_resident_bytes=" << current_resident_bytes
        << " target_resident_deficit_bytes=" << target_resident_deficit_bytes
        << " non_target_resident_bytes=" << non_target_resident_bytes
        << " observed_reclassification_bytes="
        << observed_reclassification_bytes
        << " observed_deficit_reduction_bytes="
        << observed_deficit_reduction_bytes
        << " intent_reclassification_enabled="
        << resident_region_fetch_reclassification_enabled.load(
               std::memory_order_relaxed)
        << " low_efficiency_streak="
        << static_cast<uint32_t>(resident_region_fetch_low_efficiency_streak)
        << " intent_reclassification_transition=" << reclassification_transition
        << " retry_cooldown_windows="
        << static_cast<uint32_t>(resident_region_fetch_retry_cooldown_windows)
        << " disabled_deficit_bytes="
        << resident_region_fetch_disabled_deficit_bytes
        << " material_deficit_growth_bytes=" << material_deficit_growth_bytes
        << " changed_groups=" << changed_groups
        << " window_multiplier=" << resident_region_fetch_window_multiplier
        << " next_plan_check=" << resident_region_fetch_next_plan_check
        << " ready="
        << resident_region_fetch_plan_ready.load(std::memory_order_relaxed)
        << " desired_group_hash=" << desired_group_hash
        << " top_groups=" << top_group_ids[0] << ',' << top_group_ids[1] << ','
        << top_group_ids[2] << ',' << top_group_ids[3]
        << " top_r_gain=" << top_resident_gains[0] << ','
        << top_resident_gains[1] << ',' << top_resident_gains[2] << ','
        << top_resident_gains[3] << " top_s_gain=" << top_streaming_gains[0]
        << ',' << top_streaming_gains[1] << ',' << top_streaming_gains[2] << ','
        << top_streaming_gains[3] << std::endl;
}

inline bool ConcurrentArrayCache::refill_resident_fetch_capacity(
    const std::vector<::FarLib::allocator::RegionHotnessSnapshot> &snapshots,
    uint64_t check_id) {
    using ::FarLib::allocator::FULL;
    using ::FarLib::allocator::RegionExchangeFailure;
    using ::FarLib::allocator::RegionHotnessSnapshot;
    using ::FarLib::allocator::RegionPlacement;
    using ::FarLib::allocator::RegionSize;
    using ::FarLib::allocator::USABLE;

    const size_t headroom_pct = std::min<size_t>(
        100, ::FarLib::get_config().region_fetch_resident_headroom_pct);
    if (!region_fetch_hotness_placement_enabled() || headroom_pct == 0 ||
        !resident_region_fetch_reclassification_enabled.load(
            std::memory_order_acquire) ||
        !resident_region_fetch_plan_ready.load(std::memory_order_acquire)) {
        return false;
    }

    std::array<uint64_t, ::FarLib::allocator::RegionBinCount> fallback_counts{};
    std::array<uint64_t, ::FarLib::allocator::RegionBinCount> fallback_bytes{};
    uint64_t total_fallback_count = 0;
    uint64_t total_fallback_bytes = 0;
    for (size_t bin = 0; bin < ::FarLib::allocator::RegionBinCount; ++bin) {
        fallback_counts[bin] =
            region_fetch_window_resident_fallback_count[bin].exchange(
                0, std::memory_order_acq_rel);
        fallback_bytes[bin] =
            region_fetch_window_resident_fallback_bytes[bin].exchange(
                0, std::memory_order_acq_rel);
        total_fallback_count += fallback_counts[bin];
        total_fallback_bytes += fallback_bytes[bin];
    }
    if (total_fallback_count == 0) {
        return false;
    }

    const uint64_t start = get_cycles();
    region_fetch_reserve_refill_checks.fetch_add(1, std::memory_order_relaxed);
    const size_t resident_budget_regions =
        ::FarLib::allocator::global_heap.get_resident_region_budget();
    const size_t target_refill_regions =
        std::max<size_t>(1, resident_budget_regions * headroom_pct / 100);
    const size_t max_attempts = target_refill_regions;
    std::vector<const RegionHotnessSnapshot *> streaming_candidates;
    std::vector<const RegionHotnessSnapshot *> resident_cold;
    for (const auto &snapshot : snapshots) {
        if (!snapshot.list_owned) {
            continue;
        }
        if (snapshot.placement == RegionPlacement::Streaming &&
            (snapshot.state == USABLE || snapshot.state == FULL) &&
            snapshot.resident_fetch_intent_bytes != 0) {
            streaming_candidates.push_back(&snapshot);
        }
    }
    auto intent_better = [&fallback_bytes](const RegionHotnessSnapshot *left,
                                           const RegionHotnessSnapshot *right) {
        if (left->resident_fetch_intent_bytes !=
            right->resident_fetch_intent_bytes) {
            return left->resident_fetch_intent_bytes >
                   right->resident_fetch_intent_bytes;
        }
        if (fallback_bytes[left->bin] != fallback_bytes[right->bin]) {
            return fallback_bytes[left->bin] > fallback_bytes[right->bin];
        }
        if (left->weighted_references != right->weighted_references) {
            return left->weighted_references > right->weighted_references;
        }
        if (left->free_bytes != right->free_bytes) {
            return left->free_bytes > right->free_bytes;
        }
        return left->region < right->region;
    };
    const size_t streaming_keep =
        std::min(max_attempts, streaming_candidates.size());
    if (streaming_keep != 0) {
        std::partial_sort(streaming_candidates.begin(),
                          streaming_candidates.begin() + streaming_keep,
                          streaming_candidates.end(), intent_better);
        streaming_candidates.resize(streaming_keep);
    }
    for (const auto &snapshot : snapshots) {
        if (snapshot.list_owned &&
            snapshot.placement == RegionPlacement::Resident &&
            (snapshot.state == FULL || snapshot.state == USABLE)) {
            resident_cold.push_back(&snapshot);
        }
    }
    auto colder_resident = [](const RegionHotnessSnapshot *left,
                              const RegionHotnessSnapshot *right) {
        if (left->state != right->state) {
            return left->state == FULL;
        }
        if (left->total_weighted_references !=
            right->total_weighted_references) {
            return left->total_weighted_references <
                   right->total_weighted_references;
        }
        if (left->free_bytes != right->free_bytes) {
            return left->free_bytes < right->free_bytes;
        }
        return left->region < right->region;
    };
    const size_t resident_keep = std::min(max_attempts, resident_cold.size());
    if (resident_keep != 0) {
        std::partial_sort(resident_cold.begin(),
                          resident_cold.begin() + resident_keep,
                          resident_cold.end(), colder_resident);
        resident_cold.resize(resident_keep);
    }

    if (streaming_candidates.empty()) {
        region_fetch_reserve_refill_no_streaming_candidate.fetch_add(
            1, std::memory_order_relaxed);
    }
    if (resident_cold.empty()) {
        region_fetch_reserve_refill_no_resident_cold.fetch_add(
            1, std::memory_order_relaxed);
    }

    const size_t max_refill_regions =
        std::min({target_refill_regions, streaming_candidates.size(),
                  resident_cold.size()});
    uint64_t attempts = 0;
    uint64_t successes = 0;
    uint64_t new_resident_free_bytes = 0;
    uint64_t exchange_bytes = 0;
    uint64_t promoted_intent_bytes = 0;
    std::array<uint64_t, static_cast<size_t>(RegionExchangeFailure::Count)>
        failures{};
    std::vector<uint64_t> deferred_remote_frees;
    auto transition = [this, &deferred_remote_frees](
                          FarObjectEntry &entry, size_t bytes,
                          RegionPlacement from, RegionPlacement to) {
        handle_region_placement_transition_backup(entry, bytes, from, to,
                                                  &deferred_remote_frees);
    };
    const size_t pair_count = std::min(max_refill_regions, max_attempts);
    const uint64_t reserve_plan_id = (uint64_t{1} << 63) | check_id;
    std::vector<uint8_t> pair_pending(pair_count, 0);
    uint64_t pending_pairs = 0;
    for (size_t i = 0; i < pair_count; ++i) {
        const auto *hot = streaming_candidates[i];
        const auto *cold = resident_cold[i];
        const uint32_t rank = static_cast<uint32_t>(i + 1);
        const bool hot_pending = hot->region->try_set_profile_pending_target(
            hot->placement_epoch, RegionPlacement::Streaming,
            RegionPlacement::Resident, reserve_plan_id, rank);
        const bool cold_pending = cold->region->try_set_profile_pending_target(
            cold->placement_epoch, RegionPlacement::Resident,
            RegionPlacement::Streaming, reserve_plan_id, rank);
        if (hot_pending && cold_pending) {
            pair_pending[i] = 1;
            ++pending_pairs;
            continue;
        }
        if (hot_pending) {
            hot->region->clear_profile_pending_target(reserve_plan_id);
        }
        if (cold_pending) {
            cold->region->clear_profile_pending_target(reserve_plan_id);
        }
    }
    const uint64_t prioritized_regions =
        ::FarLib::allocator::global_heap.prioritize_profile_pending_regions(
            reserve_plan_id);
    for (size_t i = 0; i < pair_count; ++i) {
        if (!pair_pending[i]) {
            continue;
        }
        const auto *hot = streaming_candidates[i];
        const auto *cold = resident_cold[i];
        ++attempts;
        const auto result =
            ::FarLib::allocator::global_heap.try_exchange_hotness_regions(
                hot->region, hot->placement_epoch, cold->region,
                cold->placement_epoch, transition);
        if (result.changed) {
            ++successes;
            new_resident_free_bytes += hot->free_bytes;
            exchange_bytes += 2 * RegionSize;
            promoted_intent_bytes += hot->resident_fetch_intent_bytes;
            continue;
        }
        ++failures[static_cast<size_t>(result.failure)];
        hot->region->clear_profile_pending_target(reserve_plan_id);
        cold->region->clear_profile_pending_target(reserve_plan_id);
    }
    remote_allocator.deallocate_batch(deferred_remote_frees);
    region_fetch_reserve_refill_attempts.fetch_add(attempts,
                                                   std::memory_order_relaxed);
    region_fetch_reserve_refill_successes.fetch_add(successes,
                                                    std::memory_order_relaxed);
    region_fetch_reserve_refill_capacity_bytes.fetch_add(
        new_resident_free_bytes, std::memory_order_relaxed);
    region_fetch_reserve_refill_exchange_bytes.fetch_add(
        exchange_bytes, std::memory_order_relaxed);
    region_fetch_reserve_refill_promoted_intent_bytes.fetch_add(
        promoted_intent_bytes, std::memory_order_relaxed);
    const uint64_t elapsed = get_cycles() - start;
    region_fetch_reserve_refill_cycles.fetch_add(elapsed,
                                                 std::memory_order_relaxed);
    std::cerr
        << "resident.fetch_reserve_refill"
        << " check_id=" << check_id
        << " fallback_count=" << total_fallback_count
        << " fallback_bytes=" << total_fallback_bytes
        << " target_regions=" << target_refill_regions
        << " streaming_candidates=" << streaming_candidates.size()
        << " resident_cold_candidates=" << resident_cold.size()
        << " pending_pairs=" << pending_pairs
        << " prioritized_regions=" << prioritized_regions
        << " attempts=" << attempts << " successes="
        << successes
        // Keep the historical log key for experiment parser compatibility.
        << " capacity_bytes=" << new_resident_free_bytes
        << " exchange_bytes=" << exchange_bytes
        << " promoted_intent_bytes=" << promoted_intent_bytes
        << " fail_hot_unavailable="
        << failures[static_cast<size_t>(
               RegionExchangeFailure::HotRegionUnavailable)]
        << " fail_cold_unavailable="
        << failures[static_cast<size_t>(
               RegionExchangeFailure::ColdRegionUnavailable)]
        << " fail_region_changed="
        << failures[static_cast<size_t>(RegionExchangeFailure::RegionChanged)]
        << " fail_hot_busy="
        << failures[static_cast<size_t>(RegionExchangeFailure::HotEntryBusy)]
        << " fail_cold_busy="
        << failures[static_cast<size_t>(RegionExchangeFailure::ColdEntryBusy)]
        << " fail_invalid_input="
        << failures[static_cast<size_t>(RegionExchangeFailure::InvalidInput)]
        << " fail_hot_invalid_state="
        << failures[static_cast<size_t>(
               RegionExchangeFailure::HotInvalidRegionState)]
        << " fail_cold_invalid_state="
        << failures[static_cast<size_t>(
               RegionExchangeFailure::ColdInvalidRegionState)]
        << " cycles=" << elapsed << std::endl;
    return successes != 0;
}

inline void ConcurrentArrayCache::publish_resident_region_hotness_plan(
    bool complete_phase) {
    using ::FarLib::allocator::RegionHotnessSnapshot;
    using ::FarLib::allocator::RegionPlacement;
    constexpr uint64_t RegionBytes = ::FarLib::allocator::RegionSize;

    const uint64_t check_id = resident_profile_candidate_check_count.fetch_add(
                                  1, std::memory_order_relaxed) +
                              1;
    resident_profile_plan_check_count.fetch_add(1, std::memory_order_relaxed);

    const uint64_t collection_start = get_cycles();
    std::vector<RegionHotnessSnapshot> snapshots;
    ::FarLib::allocator::global_heap.collect_region_hotness_snapshots(
        snapshots, resident_profile_write_weight,
        resident_profile_ema_decay_shift);
    profile::count_resident_profile_planner_collection_cycles(get_cycles() -
                                                              collection_start);

    publish_region_fetch_hotness_predictions(check_id);

    const uint64_t candidate_start = get_cycles();
    std::stable_sort(
        snapshots.begin(), snapshots.end(),
        [](const RegionHotnessSnapshot &left,
           const RegionHotnessSnapshot &right) {
            if (left.weighted_references != right.weighted_references) {
                return left.weighted_references > right.weighted_references;
            }
            if (left.placement != right.placement) {
                // Preserve the current assignment on a score tie.
                return left.placement == RegionPlacement::Resident;
            }
            return left.region < right.region;
        });

    // A fallback-driven generation exchange is itself the physical plan
    // for this window. Do not immediately re-scan and run the independent
    // Region-hotness exchange over stale snapshots in the same window.
    if (refill_resident_fetch_capacity(snapshots, check_id)) {
        profile::count_resident_profile_planner_candidate_cycles(
            get_cycles() - candidate_start);
        return;
    }

    const size_t resident_current_regions = static_cast<size_t>(std::count_if(
        snapshots.begin(), snapshots.end(),
        [](const RegionHotnessSnapshot &snapshot) {
            return snapshot.placement == RegionPlacement::Resident;
        }));
    const size_t resident_reserved_regions =
        ::FarLib::allocator::global_heap.get_resident_reserved_regions();
    const size_t fixed_resident_regions =
        resident_reserved_regions > resident_current_regions
            ? resident_reserved_regions - resident_current_regions
            : 0;
    const size_t resident_budget_regions =
        ::FarLib::allocator::global_heap.get_resident_region_budget();
    const size_t fetch_headroom_regions =
        region_fetch_hotness_placement_enabled()
            ? resident_budget_regions *
                  std::min<size_t>(100,
                                   ::FarLib::get_config()
                                       .region_fetch_resident_headroom_pct) /
                  100
            : 0;
    const size_t soft_resident_budget_regions =
        resident_budget_regions > fetch_headroom_regions
            ? resident_budget_regions - fetch_headroom_regions
            : 0;
    const size_t resident_total_current_regions =
        std::min(resident_budget_regions,
                 resident_current_regions + fixed_resident_regions);
    const size_t planner_resident_budget_regions =
        region_fetch_hotness_placement_enabled()
            ? std::max(soft_resident_budget_regions,
                       resident_total_current_regions)
            : resident_budget_regions;
    const size_t resident_target_regions =
        std::min(snapshots.size(),
                 planner_resident_budget_regions > fixed_resident_regions
                     ? planner_resident_budget_regions - fixed_resident_regions
                     : 0);
    std::vector<const RegionHotnessSnapshot *> promotion_candidates;
    std::vector<const RegionHotnessSnapshot *> demotion_candidates;
    size_t promotion_mismatches = 0;
    size_t demotion_mismatches = 0;
    size_t stable_promotion_mismatches = 0;
    size_t stable_demotion_mismatches = 0;
    size_t transitionable_promotion_mismatches = 0;
    size_t transitionable_demotion_mismatches = 0;
    size_t promotion_candidate_usable = 0;
    size_t promotion_candidate_full = 0;
    size_t demotion_candidate_usable = 0;
    size_t demotion_candidate_full = 0;
    for (size_t rank = 0; rank < snapshots.size(); ++rank) {
        auto &snapshot = snapshots[rank];
        const RegionPlacement desired = rank < resident_target_regions
                                            ? RegionPlacement::Resident
                                            : RegionPlacement::Streaming;
        if (desired == snapshot.placement) {
            snapshot.region->profile_last_target_placement.store(
                static_cast<uint8_t>(desired), std::memory_order_relaxed);
            snapshot.region->profile_target_streak.store(
                0, std::memory_order_relaxed);
            continue;
        }
        if (desired == RegionPlacement::Resident) {
            ++promotion_mismatches;
        } else {
            ++demotion_mismatches;
        }
        const uint8_t desired_value = static_cast<uint8_t>(desired);
        const uint8_t previous =
            snapshot.region->profile_last_target_placement.exchange(
                desired_value, std::memory_order_acq_rel);
        uint8_t streak = 1;
        if (complete_phase) {
            // A complete application phase is already a stable sampling
            // window. Phase-triggered mode performs exactly one planner
            // check, so requiring a second check would make every first
            // plan an unconditional no-op.
            streak = kResidentDirectionStableComparisons;
        } else if (previous == desired_value) {
            const uint8_t old_streak =
                snapshot.region->profile_target_streak.load(
                    std::memory_order_relaxed);
            streak = old_streak == std::numeric_limits<uint8_t>::max()
                         ? old_streak
                         : static_cast<uint8_t>(old_streak + 1);
        }
        snapshot.region->profile_target_streak.store(streak,
                                                     std::memory_order_relaxed);
        const bool stable = streak >= kResidentDirectionStableComparisons;
        const bool transitionable_state =
            snapshot.list_owned &&
            (snapshot.state == ::FarLib::allocator::USABLE ||
             snapshot.state == ::FarLib::allocator::FULL);
        if (stable) {
            if (desired == RegionPlacement::Resident) {
                ++stable_promotion_mismatches;
            } else {
                ++stable_demotion_mismatches;
            }
        }
        if (stable && transitionable_state) {
            if (desired == RegionPlacement::Resident) {
                ++transitionable_promotion_mismatches;
            } else {
                ++transitionable_demotion_mismatches;
            }
        }
        if (!stable || !transitionable_state) {
            continue;
        }
        if (desired == RegionPlacement::Resident) {
            promotion_candidates.push_back(&snapshot);
            if (snapshot.state == ::FarLib::allocator::USABLE) {
                ++promotion_candidate_usable;
            } else {
                ++promotion_candidate_full;
            }
        } else {
            demotion_candidates.push_back(&snapshot);
            if (snapshot.state == ::FarLib::allocator::USABLE) {
                ++demotion_candidate_usable;
            } else {
                ++demotion_candidate_full;
            }
        }
    }
    // The rank scan encounters the warmest demotion candidate first;
    // exchange the coldest Resident regions first.
    std::reverse(demotion_candidates.begin(), demotion_candidates.end());
    profile::count_resident_profile_planner_candidate_cycles(get_cycles() -
                                                             candidate_start);

    if (!complete_phase &&
        check_id <= resident_profile_initial_sample_windows) {
        std::cerr << "resident.region_hotness_plan"
                  << " check_id=" << check_id << " phase=warmup"
                  << " regions=" << snapshots.size()
                  << " resident_current=" << resident_current_regions
                  << " resident_target=" << resident_target_regions
                  << " resident_hard_budget=" << resident_budget_regions
                  << " fetch_headroom=" << fetch_headroom_regions
                  << " promotion_mismatches=" << promotion_mismatches
                  << " demotion_mismatches=" << demotion_mismatches
                  << " stable_promotion_mismatches="
                  << stable_promotion_mismatches
                  << " stable_demotion_mismatches="
                  << stable_demotion_mismatches
                  << " transitionable_promotion_mismatches="
                  << transitionable_promotion_mismatches
                  << " transitionable_demotion_mismatches="
                  << transitionable_demotion_mismatches << std::endl;
        return;
    }
    if (!resident_profile_plan_ready.load(std::memory_order_acquire)) {
        resident_profile_open_migration_gate();
    }

    const size_t max_exchanges =
        std::max<size_t>(1, resident_target_regions *
                                resident_profile_migration_budget_pct / 100);
    size_t swap_count = std::min({promotion_candidates.size(),
                                  demotion_candidates.size(), max_exchanges});
    while (swap_count != 0) {
        const auto *hot = promotion_candidates[swap_count - 1];
        const auto *cold = demotion_candidates[swap_count - 1];
        const __uint128_t hot_score =
            static_cast<__uint128_t>(hot->weighted_references) * 100;
        const __uint128_t cold_score =
            static_cast<__uint128_t>(cold->weighted_references) *
            (100 +
             ::FarLib::get_config().resident_profile_migration_hysteresis_pct);
        if (hot_score > cold_score) {
            break;
        }
        --swap_count;
    }

    const size_t resident_deficit =
        resident_current_regions < resident_target_regions
            ? resident_target_regions - resident_current_regions
            : 0;
    const size_t resident_excess =
        resident_current_regions > resident_target_regions
            ? resident_current_regions - resident_target_regions
            : 0;
    const size_t extra_promotions =
        std::min(resident_deficit, promotion_candidates.size() - swap_count);
    const size_t extra_demotions =
        std::min(resident_excess, demotion_candidates.size() - swap_count);
    const uint64_t scheduled_promotion_bytes =
        (swap_count + extra_promotions) * RegionBytes;
    const uint64_t scheduled_demotion_bytes =
        (swap_count + extra_demotions) * RegionBytes;
    // A check with no transitionable Region is not a placement plan.  Do
    // not mint a plan id or wake the migration path for this no-op.
    if (scheduled_promotion_bytes == 0 && scheduled_demotion_bytes == 0) {
        std::cerr << "resident.region_hotness_plan"
                  << " check_id=" << check_id << " phase=noop"
                  << " regions=" << snapshots.size()
                  << " resident_before=" << resident_current_regions
                  << " resident_target=" << resident_target_regions
                  << " resident_hard_budget=" << resident_budget_regions
                  << " fetch_headroom=" << fetch_headroom_regions
                  << " promotion_mismatches=" << promotion_mismatches
                  << " demotion_mismatches=" << demotion_mismatches
                  << " stable_promotion_mismatches="
                  << stable_promotion_mismatches
                  << " stable_demotion_mismatches="
                  << stable_demotion_mismatches
                  << " transitionable_promotion_mismatches="
                  << transitionable_promotion_mismatches
                  << " transitionable_demotion_mismatches="
                  << transitionable_demotion_mismatches << std::endl;
        return;
    }
    const uint64_t plan_id = resident_profile_published_plan_count.fetch_add(
                                 1, std::memory_order_relaxed) +
                             1;
    resident_profile_plan_count.fetch_add(1, std::memory_order_relaxed);
    resident_profile_begin_active_plan(
        plan_id, check_id, scheduled_promotion_bytes, scheduled_demotion_bytes);

    uint64_t committed_promotion_bytes = 0;
    uint64_t committed_demotion_bytes = 0;
    uint64_t rolled_back_demotion_bytes = 0;
    uint64_t failed_promotion_regions = 0;
    uint64_t failed_demotion_regions = 0;
    std::array<size_t, static_cast<size_t>(
                           ::FarLib::allocator::RegionExchangeFailure::Count)>
        exchange_failure_counts{};
    const uint64_t publish_start = get_cycles();

    auto account_committed_promotion = [&] {
        committed_promotion_bytes += RegionBytes;
        resident_profile_promoted_bytes.fetch_add(RegionBytes,
                                                  std::memory_order_relaxed);
        resident_profile_promotion_committed_bytes_total.fetch_add(
            RegionBytes, std::memory_order_relaxed);
    };
    auto account_committed_demotion = [&] {
        committed_demotion_bytes += RegionBytes;
        resident_profile_demoted_bytes.fetch_add(RegionBytes,
                                                 std::memory_order_relaxed);
        resident_profile_demotion_committed_bytes_total.fetch_add(
            RegionBytes, std::memory_order_relaxed);
    };
    std::vector<uint64_t> deferred_remote_frees;
    auto transition_backup = [this, &deferred_remote_frees](
                                 FarObjectEntry &entry, size_t bytes,
                                 RegionPlacement from, RegionPlacement to) {
        handle_region_placement_transition_backup(entry, bytes, from, to,
                                                  &deferred_remote_frees);
    };

    std::vector<uint8_t> swap_committed(swap_count, 0);
    std::vector<uint8_t> hot_pending_target_set(swap_count, 0);
    std::vector<uint8_t> cold_pending_target_set(swap_count, 0);
    uint64_t pending_targets_set = 0;
    uint64_t pending_target_set_failures = 0;
    uint64_t pending_targets_cleared = 0;
    uint64_t exchange_attempts = 0;
    uint64_t exchange_retry_passes = 0;
    uint64_t pending_regions_prioritized = 0;
    for (size_t i = 0; i < swap_count; ++i) {
        const auto *hot = promotion_candidates[i];
        const auto *cold = demotion_candidates[i];
        const uint32_t plan_rank = static_cast<uint32_t>(i + 1);
        const bool pending_set = hot->region->try_set_profile_pending_target(
            hot->placement_epoch, RegionPlacement::Streaming,
            RegionPlacement::Resident, plan_id, plan_rank);
        hot_pending_target_set[i] = pending_set;
        pending_targets_set += pending_set;
        pending_target_set_failures += !pending_set;
        const bool cold_pending_set =
            cold->region->try_set_profile_pending_target(
                cold->placement_epoch, RegionPlacement::Resident,
                RegionPlacement::Streaming, plan_id, plan_rank);
        cold_pending_target_set[i] = cold_pending_set;
        pending_targets_set += cold_pending_set;
        pending_target_set_failures += !cold_pending_set;
        if (pending_set != cold_pending_set) {
            if (pending_set) {
                hot->region->clear_profile_pending_target(plan_id);
                hot_pending_target_set[i] = 0;
                ++pending_targets_cleared;
            }
            if (cold_pending_set) {
                cold->region->clear_profile_pending_target(plan_id);
                cold_pending_target_set[i] = 0;
                ++pending_targets_cleared;
            }
        }
        resident_profile_demotion_started_bytes_total.fetch_add(
            RegionBytes, std::memory_order_relaxed);
        resident_profile_active_plan_demotion_started_bytes.fetch_add(
            RegionBytes, std::memory_order_relaxed);
        resident_profile_promotion_started_bytes_total.fetch_add(
            RegionBytes, std::memory_order_relaxed);
        resident_profile_active_plan_promotion_started_bytes.fetch_add(
            RegionBytes, std::memory_order_relaxed);
    }

    const auto retry_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(complete_phase ? 1000 : 0);
    while (true) {
        ++exchange_retry_passes;
        pending_regions_prioritized +=
            ::FarLib::allocator::global_heap.prioritize_profile_pending_regions(
                plan_id);
        size_t remaining_swaps = 0;
        size_t committed_this_pass = 0;
        for (size_t i = 0; i < swap_count; ++i) {
            if (swap_committed[i]) {
                continue;
            }
            ++remaining_swaps;
            const auto *cold = demotion_candidates[i];
            const auto *hot = promotion_candidates[i];
            if (!hot_pending_target_set[i] || !cold_pending_target_set[i]) {
                const uint32_t plan_rank = static_cast<uint32_t>(i + 1);
                const bool pending_set =
                    hot->region->try_set_profile_pending_target(
                        hot->placement_epoch, RegionPlacement::Streaming,
                        RegionPlacement::Resident, plan_id, plan_rank);
                hot_pending_target_set[i] = pending_set;
                pending_targets_set += pending_set;
                pending_target_set_failures += !pending_set;
                const bool cold_pending_set =
                    cold->region->try_set_profile_pending_target(
                        cold->placement_epoch, RegionPlacement::Resident,
                        RegionPlacement::Streaming, plan_id, plan_rank);
                cold_pending_target_set[i] = cold_pending_set;
                pending_targets_set += cold_pending_set;
                pending_target_set_failures += !cold_pending_set;
                if (!pending_set || !cold_pending_set) {
                    if (pending_set) {
                        hot->region->clear_profile_pending_target(plan_id);
                        hot_pending_target_set[i] = 0;
                        ++pending_targets_cleared;
                    }
                    if (cold_pending_set) {
                        cold->region->clear_profile_pending_target(plan_id);
                        cold_pending_target_set[i] = 0;
                        ++pending_targets_cleared;
                    }
                    continue;
                }
            }
            ++exchange_attempts;
            const auto exchanged =
                ::FarLib::allocator::global_heap.try_exchange_hotness_regions(
                    hot->region, hot->placement_epoch, cold->region,
                    cold->placement_epoch, transition_backup);
            if (exchanged.changed) {
                swap_committed[i] = 1;
                ++committed_this_pass;
                --remaining_swaps;
                account_committed_demotion();
                account_committed_promotion();
            } else {
                ++exchange_failure_counts[static_cast<size_t>(
                    exchanged.failure)];
            }
        }
        if (remaining_swaps == 0 || !complete_phase ||
            std::chrono::steady_clock::now() >= retry_deadline) {
            break;
        }
        if (committed_this_pass == 0) {
            uthread::yield();
        }
    }

    for (size_t i = 0; i < swap_count; ++i) {
        if (swap_committed[i]) {
            continue;
        }
        ++failed_promotion_regions;
        ++failed_demotion_regions;
        if (hot_pending_target_set[i]) {
            pending_targets_cleared +=
                promotion_candidates[i]->region->clear_profile_pending_target(
                    plan_id);
        }
        if (cold_pending_target_set[i]) {
            pending_targets_cleared +=
                demotion_candidates[i]->region->clear_profile_pending_target(
                    plan_id);
        }
    }

    for (size_t i = swap_count; i < swap_count + extra_promotions; ++i) {
        const auto *hot = promotion_candidates[i];
        resident_profile_promotion_started_bytes_total.fetch_add(
            RegionBytes, std::memory_order_relaxed);
        resident_profile_active_plan_promotion_started_bytes.fetch_add(
            RegionBytes, std::memory_order_relaxed);
        const auto promoted =
            ::FarLib::allocator::global_heap.try_reclassify_hotness_region(
                hot->region, hot->placement_epoch, RegionPlacement::Streaming,
                RegionPlacement::Resident, transition_backup);
        if (promoted.changed) {
            account_committed_promotion();
        } else {
            ++failed_promotion_regions;
        }
    }
    for (size_t i = swap_count; i < swap_count + extra_demotions; ++i) {
        const auto *cold = demotion_candidates[i];
        resident_profile_demotion_started_bytes_total.fetch_add(
            RegionBytes, std::memory_order_relaxed);
        resident_profile_active_plan_demotion_started_bytes.fetch_add(
            RegionBytes, std::memory_order_relaxed);
        const auto demoted =
            ::FarLib::allocator::global_heap.try_reclassify_hotness_region(
                cold->region, cold->placement_epoch, RegionPlacement::Resident,
                RegionPlacement::Streaming, transition_backup);
        if (demoted.changed) {
            account_committed_demotion();
        } else {
            ++failed_demotion_regions;
        }
    }
    remote_allocator.deallocate_batch(deferred_remote_frees);

    uint64_t pending_targets_remaining = 0;
    for (size_t i = 0; i < swap_count; ++i) {
        pending_targets_remaining +=
            promotion_candidates[i]->region->profile_pending_plan_matches(
                plan_id);
        pending_targets_remaining +=
            demotion_candidates[i]->region->profile_pending_plan_matches(
                plan_id);
    }

    resident_profile_active_plan_promotion_committed_bytes.store(
        committed_promotion_bytes, std::memory_order_release);
    resident_profile_active_plan_demotion_committed_bytes.store(
        committed_demotion_bytes, std::memory_order_release);
    resident_profile_active_plan_demotion_rolled_back_bytes.store(
        rolled_back_demotion_bytes, std::memory_order_release);
    resident_profile_active_plan_promotion_cancelled_bytes.store(
        scheduled_promotion_bytes > committed_promotion_bytes
            ? scheduled_promotion_bytes - committed_promotion_bytes
            : 0,
        std::memory_order_release);
    resident_profile_active_plan_demotion_cancelled_bytes.store(
        scheduled_demotion_bytes >
                committed_demotion_bytes + rolled_back_demotion_bytes
            ? scheduled_demotion_bytes - committed_demotion_bytes -
                  rolled_back_demotion_bytes
            : 0,
        std::memory_order_release);
    resident_profile_maybe_complete_active_plan(plan_id);
    resident_local_bytes.store(
        ::FarLib::allocator::global_heap.get_resident_reserved_regions() *
            RegionBytes,
        std::memory_order_release);
    profile::count_resident_profile_planner_publish_cycles(get_cycles() -
                                                           publish_start);
    std::cerr
        << "resident.region_hotness_plan"
        << " check_id=" << check_id << " plan_id=" << plan_id
        << " regions=" << snapshots.size()
        << " resident_before=" << resident_current_regions
        << " resident_target=" << resident_target_regions
        << " promotion_mismatches=" << promotion_mismatches
        << " demotion_mismatches=" << demotion_mismatches
        << " stable_promotion_mismatches=" << stable_promotion_mismatches
        << " stable_demotion_mismatches=" << stable_demotion_mismatches
        << " transitionable_promotion_mismatches="
        << transitionable_promotion_mismatches
        << " transitionable_demotion_mismatches="
        << transitionable_demotion_mismatches
        << " promotion_candidates=" << promotion_candidates.size()
        << " demotion_candidates=" << demotion_candidates.size()
        << " scheduled_promotion_bytes=" << scheduled_promotion_bytes
        << " scheduled_demotion_bytes=" << scheduled_demotion_bytes
        << " committed_promotion_bytes=" << committed_promotion_bytes
        << " committed_demotion_bytes=" << committed_demotion_bytes
        << " rolled_back_demotion_bytes=" << rolled_back_demotion_bytes
        << " failed_promotion_regions=" << failed_promotion_regions
        << " failed_demotion_regions=" << failed_demotion_regions
        << " pending_targets_set=" << pending_targets_set
        << " pending_target_set_failures=" << pending_target_set_failures
        << " pending_targets_cleared=" << pending_targets_cleared
        << " pending_targets_remaining=" << pending_targets_remaining
        << " exchange_attempts=" << exchange_attempts
        << " exchange_retry_passes=" << exchange_retry_passes
        << " pending_regions_prioritized=" << pending_regions_prioritized
        << " pending_mark_bypasses="
        << resident_region_pending_mark_bypasses.load(std::memory_order_relaxed)
        << " pending_unmarks="
        << resident_region_pending_unmarks.load(std::memory_order_relaxed)
        << " pending_evicting_seen="
        << resident_region_pending_evicting_seen.load(std::memory_order_relaxed)
        << " exchange_fail_invalid_input="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::InvalidInput)]
        << " exchange_fail_hot_invalid_state="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::
                   HotInvalidRegionState)]
        << " exchange_fail_cold_invalid_state="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::
                   ColdInvalidRegionState)]
        << " exchange_fail_cold_unavailable="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::
                   ColdRegionUnavailable)]
        << " exchange_fail_hot_unavailable="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::
                   HotRegionUnavailable)]
        << " exchange_fail_region_changed="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::RegionChanged)]
        << " exchange_fail_cold_entry_busy="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::ColdEntryBusy)]
        << " exchange_fail_hot_entry_busy="
        << exchange_failure_counts[static_cast<size_t>(
               ::FarLib::allocator::RegionExchangeFailure::HotEntryBusy)]
        << " promotion_candidate_usable=" << promotion_candidate_usable
        << " promotion_candidate_full=" << promotion_candidate_full
        << " demotion_candidate_usable=" << demotion_candidate_usable
        << " demotion_candidate_full=" << demotion_candidate_full << std::endl;
}

}  // namespace FarLib::cache
