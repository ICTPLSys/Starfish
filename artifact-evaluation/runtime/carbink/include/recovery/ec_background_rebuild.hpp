#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>

namespace FarLib::cache::ec_background {
inline constexpr uint32_t kWorkers = 4;
inline constexpr uint32_t kPipelineDepth = 8;
inline constexpr uint32_t kOpsPerJob = 5;
inline constexpr uint32_t kDualOpsPerJob = 6;
inline constexpr uint32_t kSlots = kWorkers * kPipelineDepth * kDualOpsPerJob;
inline constexpr uint64_t kBurstBytes = 4ull * 1024 * 1024;

// Disjoint from normal pointer/READ/WRITE, RMW, EC batch and EC READ tags.
inline constexpr uint64_t kTag = 3ull << 60;
inline constexpr uint64_t kTagMask = 15ull << 60;
inline constexpr uint64_t kSlotMask = 0xffull;
inline constexpr uint64_t kSequenceShift = 8;
inline constexpr uint64_t kSequenceMask = (1ull << 52) - 1;
inline bool is_wr_id(uint64_t id) { return (id & kTagMask) == kTag; }

inline uint32_t slot_from_wr_id(uint64_t id) {
    if (!is_wr_id(id)) return kSlots;
    const uint64_t slot = id & kSlotMask;
    return slot < kSlots ? static_cast<uint32_t>(slot) : kSlots;
}

// One completion slot owns one generation stream. Completion release is the
// last callback access; a new generation is never armed until the prior WR has
// terminated.
struct Completion {
    std::atomic<uint64_t> expected{0};
    std::atomic<int> status{0};
    uint64_t sequence = 0;
    uint64_t arm(uint32_t slot = 0) {
        if (slot >= kSlots) return 0;
        if (sequence == kSequenceMask) return 0;
        status.store(0, std::memory_order_relaxed);
        const uint64_t id =
            kTag | (++sequence << kSequenceShift) | slot;
        expected.store(id, std::memory_order_release);
        return id;
    }
    bool matches(uint64_t id) const {
        return id == expected.load(std::memory_order_acquire);
    }
    void complete(bool success) {
        status.store(success ? 1 : -1, std::memory_order_release);
    }
};

// Shared global ticket clock for one-WR bursts. READ and WRITE use separate
// Pacer instances. Call once per intended post, never for an unaccepted retry.
struct Pacer {
    std::atomic<uint64_t> next_ns{0};
    bool try_reserve(uint64_t now, uint64_t bytes, uint64_t mbps,
                     uint64_t burst_bytes, uint64_t *ready_ns) {
        if (ready_ns == nullptr || bytes == 0 || mbps == 0 ||
            burst_bytes < bytes) {
            if (ready_ns != nullptr) *ready_ns = 0;
            return false;
        }
        const uint64_t duration = (bytes * 1000 + mbps - 1) / mbps;
        const uint64_t burst_slots = burst_bytes / bytes;
        if (burst_slots == 0) {
            *ready_ns = 0;
            return false;
        }
        const uint64_t allowance = (burst_slots - 1) * duration;
        uint64_t observed = next_ns.load(std::memory_order_acquire);
        for (;;) {
            const uint64_t start = std::max(now, observed);
            if (start > now + allowance) {
                *ready_ns = start - allowance;
                return false;  // blocked callers do not charge the clock
            }
            const uint64_t desired = start + duration;
            if (next_ns.compare_exchange_weak(
                    observed, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                *ready_ns = start;
                return true;
            }
        }
    }

    uint64_t reserve(uint64_t now, uint64_t bytes, uint64_t mbps) {
        if (mbps == 0) return 0;
        const uint64_t duration = (bytes * 1000 + mbps - 1) / mbps;
        uint64_t observed = next_ns.load(std::memory_order_acquire);
        for (;;) {
            const uint64_t start = std::max(now, observed);
            const uint64_t desired = start + duration;
            if (next_ns.compare_exchange_weak(
                    observed, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return start;
            }
        }
    }
};

struct SharedState {
    std::array<Completion, kSlots> completions;
    Pacer reads;
    Pacer writes;
    std::atomic<uint32_t> worker_ids{0};
    std::atomic<uint32_t> workers_done{0};
    std::atomic<bool> claimed{false};
    std::atomic<bool> ready{false};
    std::atomic<bool> failed{false};
    std::atomic<bool> fully_rebuilt{false};
    uint64_t start_ns = 0;
    uint64_t scan_limit = 0;
    std::atomic<uint64_t> cursor{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> empty_reactivated{0};
    std::atomic<uint64_t> verified{0};
    std::atomic<uint64_t> read_bytes{0};
    std::atomic<uint64_t> write_bytes{0};
    std::atomic<uint64_t> selected_bytes{0};
    std::atomic<uint64_t> skipped_bytes{0};
    std::atomic<uint64_t> completed_ranges{0};
    std::atomic<uint64_t> live_groups{0};
    std::atomic<uint64_t> live_objects{0};
    std::atomic<uint64_t> busy_visits{0};
    std::atomic<uint64_t> in_flight{0};
    std::atomic<uint64_t> max_in_flight{0};
    std::atomic<uint64_t> prefetched{0};
    std::atomic<uint64_t> write_completed{0};
    std::atomic<uint64_t> decode_ns{0};
    std::atomic<uint64_t> post_attempts{0};
    std::atomic<uint64_t> pace_blocked{0};
    std::atomic<uint64_t> sq_full{0};
    std::atomic<uint64_t> cq_polls{0};
    std::array<std::atomic<uint64_t>, 64> endpoint_shards{};
    std::array<std::atomic<uint64_t>, 64> endpoint_read_bytes{};
    std::array<std::atomic<uint64_t>, 64> endpoint_write_bytes{};
    std::array<std::atomic<uint64_t>, 64> endpoint_live_groups{};
    std::array<std::atomic<uint64_t>, 64> endpoint_live_objects{};
    std::array<std::atomic<uint64_t>, 64> endpoint_selected_bytes{};
    std::array<std::atomic<uint64_t>, 64> endpoint_skipped_bytes{};
    std::array<std::atomic<uint64_t>, 64> endpoint_completed_ranges{};
    std::array<std::atomic<uint64_t>, 64> endpoint_empty_reactivated{};
    std::atomic<uint64_t> spare_shards{0};
    std::atomic<uint64_t> regular_shards{0};
};
}  // namespace FarLib::cache::ec_background
