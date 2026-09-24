#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

// Exploratory observer only. Runtime decisions never read this state.
namespace inclusive_reclaim_diag {

inline constexpr size_t kMaxLogicalWorkers = 64;
inline constexpr size_t kNoWorker = std::numeric_limits<size_t>::max();
inline constexpr size_t kNoAllocBin = std::numeric_limits<size_t>::max();

enum HeldPhase : uint64_t { HeldNone = 0, HeldMark = 1, HeldEvict = 2 };

struct alignas(64) HeldSlot {
    std::atomic<uint64_t> phase{HeldNone};
    std::atomic<uint64_t> regions{0};
    // Sum of region free_size observed when the current batch was acquired.
    // It is decremented by that same captured value as each region is returned.
    std::atomic<uint64_t> selected_free_bytes{0};
};

struct HeldSnapshot {
    uint64_t mark_regions = 0;
    uint64_t mark_selected_free_bytes = 0;
    uint64_t evict_regions = 0;
    uint64_t evict_selected_free_bytes = 0;
    uint64_t slot_errors = 0;
    uint64_t snapshot_unstable = 0;
};

inline std::array<HeldSlot, kMaxLogicalWorkers> held_slots{};
inline std::atomic<uint64_t> slot_errors{0};

inline bool enabled() {
    static const bool value = [] {
        const char *text = std::getenv("FARLIB_INCLUSIVE_RECLAIM_DIAG");
        return text != nullptr && std::strcmp(text, "1") == 0;
    }();
    return value;
}

inline void begin_hold(size_t slot, HeldPhase phase, uint64_t regions,
                       uint64_t selected_free_bytes) {
    if (!enabled()) return;
    if (slot >= held_slots.size()) {
        slot_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    auto &held = held_slots[slot];
    if (held.regions.load(std::memory_order_relaxed) != 0) {
        slot_errors.fetch_add(1, std::memory_order_relaxed);
    }
    held.phase.store(phase, std::memory_order_relaxed);
    held.selected_free_bytes.store(selected_free_bytes,
                                   std::memory_order_relaxed);
    held.regions.store(regions, std::memory_order_release);
}

inline void release_one(size_t slot, uint64_t selected_free_bytes) {
    if (!enabled() || slot >= held_slots.size()) return;
    auto &held = held_slots[slot];
    const uint64_t prior_regions =
        held.regions.fetch_sub(1, std::memory_order_acq_rel);
    const uint64_t prior_bytes = held.selected_free_bytes.fetch_sub(
        selected_free_bytes, std::memory_order_acq_rel);
    if (prior_regions == 0 || prior_bytes < selected_free_bytes) {
        slot_errors.fetch_add(1, std::memory_order_relaxed);
        held.regions.store(0, std::memory_order_release);
        held.selected_free_bytes.store(0, std::memory_order_release);
    } else if (prior_regions == 1) {
        held.phase.store(HeldNone, std::memory_order_release);
    }
}

inline void clear_hold(size_t slot) {
    if (!enabled() || slot >= held_slots.size()) return;
    auto &held = held_slots[slot];
    if (held.regions.load(std::memory_order_acquire) != 0 ||
        held.selected_free_bytes.load(std::memory_order_acquire) != 0) {
        slot_errors.fetch_add(1, std::memory_order_relaxed);
    }
    held.regions.store(0, std::memory_order_release);
    held.selected_free_bytes.store(0, std::memory_order_release);
    held.phase.store(HeldNone, std::memory_order_release);
}

class HeldRegionGuard {
public:
    HeldRegionGuard(bool active, size_t slot, uint64_t selected_free_bytes)
        : active_(active), slot_(slot), bytes_(selected_free_bytes) {}
    HeldRegionGuard(const HeldRegionGuard &) = delete;
    HeldRegionGuard &operator=(const HeldRegionGuard &) = delete;
    ~HeldRegionGuard() {
        if (active_) release_one(slot_, bytes_);
    }

private:
    bool active_;
    size_t slot_;
    uint64_t bytes_;
};

inline HeldSnapshot snapshot() {
    HeldSnapshot out;
    if (!enabled()) return out;
    for (const auto &held : held_slots) {
        const uint64_t phase_before =
            held.phase.load(std::memory_order_acquire);
        const uint64_t regions = held.regions.load(std::memory_order_acquire);
        if (regions == 0) continue;
        const uint64_t bytes =
            held.selected_free_bytes.load(std::memory_order_relaxed);
        const uint64_t regions_after =
            held.regions.load(std::memory_order_acquire);
        const uint64_t phase_after =
            held.phase.load(std::memory_order_acquire);
        if (regions_after != regions || phase_after != phase_before) {
            ++out.snapshot_unstable;
            continue;
        }
        if (phase_before == HeldMark) {
            out.mark_regions += regions;
            out.mark_selected_free_bytes += bytes;
        } else if (phase_before == HeldEvict) {
            out.evict_regions += regions;
            out.evict_selected_free_bytes += bytes;
        } else {
            ++out.snapshot_unstable;
        }
    }
    out.slot_errors += slot_errors.load(std::memory_order_relaxed);
    return out;
}

inline bool sample_region(const void *region) {
    uint64_t value = reinterpret_cast<uintptr_t>(region);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return (value & 63U) == 0;
}

}  // namespace inclusive_reclaim_diag
