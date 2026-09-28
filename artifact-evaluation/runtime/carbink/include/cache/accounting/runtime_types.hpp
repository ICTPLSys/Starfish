#pragma once

#include <atomic>
#include <cstdint>

namespace FarLib::cache::design1 {

// Per-group counters are updated by mutators and consumed by the planner.
// Window counters are reset at publication; lifetime totals are monotonic.
struct BackupBehaviorGroupProfile {
    std::atomic<uint64_t> allocated_objects{0};
    std::atomic<uint64_t> allocated_bytes{0};
    std::atomic<uint64_t> allocated_footprint_bytes{0};
    std::atomic<uint64_t> window_fetches{0};
    std::atomic<uint64_t> window_clean_evictions{0};
    std::atomic<uint64_t> window_dirty_evictions{0};
    std::atomic<uint64_t> total_fetches{0};
    std::atomic<uint64_t> total_clean_evictions{0};
    std::atomic<uint64_t> total_dirty_evictions{0};
    std::atomic<uint64_t> admitted_fetches{0};
    std::atomic<uint64_t> rejected_fetches{0};
    std::atomic<uint64_t> resident_window_references{0};
    std::atomic<uint64_t> resident_total_references{0};
    std::atomic<uint64_t> resident_current_bytes{0};
    std::atomic<uint64_t> resident_target_bytes{0};
    std::atomic<uint32_t> published_score_pct{0};
    std::atomic<uint32_t> published_clean_pct{0};
    std::atomic<uint32_t> published_refetch_pct{0};
    std::atomic<uint64_t> published_samples{0};
    std::atomic_flag publishing = ATOMIC_FLAG_INIT;
};

struct ResidentPlacementGroupProfile {
    std::atomic<uint64_t> live_objects{0};
    std::atomic<uint64_t> live_footprint_bytes{0};
    std::atomic<uint64_t> window_read_references{0};
    std::atomic<uint64_t> window_write_references{0};
    std::atomic<uint64_t> window_promotion_opportunities{0};
    std::atomic<uint64_t> total_read_references{0};
    std::atomic<uint64_t> total_write_references{0};
    std::atomic<uint64_t> ema_read_references{0};
    std::atomic<uint64_t> ema_write_references{0};
    std::atomic<uint64_t> window_fetches{0};
    std::atomic<uint64_t> window_fetch_bytes{0};
    std::atomic<uint64_t> total_fetches{0};
    std::atomic<uint64_t> total_fetch_bytes{0};
    std::atomic<uint64_t> window_clean_evictions{0};
    std::atomic<uint64_t> window_clean_evict_bytes{0};
    std::atomic<uint64_t> total_clean_evictions{0};
    std::atomic<uint64_t> total_clean_evict_bytes{0};
    std::atomic<uint64_t> window_dirty_evictions{0};
    std::atomic<uint64_t> window_dirty_evict_bytes{0};
    std::atomic<uint64_t> total_dirty_evictions{0};
    std::atomic<uint64_t> total_dirty_evict_bytes{0};
    std::atomic<uint64_t> ema_fetch_bytes{0};
    std::atomic<uint64_t> ema_clean_evict_bytes{0};
    std::atomic<uint64_t> ema_dirty_evict_bytes{0};
    std::atomic<uint64_t> resident_current_bytes{0};
    std::atomic<uint64_t> resident_target_bytes{0};
    std::atomic<bool> fetch_preferred_resident{false};
    std::atomic<bool> fetch_previous_candidate_resident{false};
    std::atomic<uint8_t> fetch_candidate_streak{0};
    uint32_t logical_owner_id{0};
    uint32_t allocation_bin{0};
};

// The active cohort is protected by resident_placement_group_mutex.
struct ActiveResidentPlacementGroup {
    uint32_t group_id{0};
    uint64_t filled_bytes{0};
};

}  // namespace FarLib::cache::design1
