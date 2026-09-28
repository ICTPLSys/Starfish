#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace FarLib::profile {

// A phase bit is published only at the mark/evict window boundaries.  RDMA
// hooks sample the mask and update their thread-local shard; they never
// contend on a global byte counter.
enum class EvacPhase : uint8_t {
    Mark = 0,
    Evict = 1,
};

enum class EvacRdmaTraffic : uint8_t {
    Ordinary = 0,
    EcBatch = 1,
};

enum class EvacRdmaDirection : uint8_t {
    Read = 0,
    Write = 1,
};

enum class EvacPhaseCoverage : uint8_t {
    MarkOnly = 0,
    EvictOnly = 1,
    Overlap = 2,
    Count = 3,
};

inline constexpr uint8_t kEvacPhaseMarkBit = uint8_t{1} << 0;
inline constexpr uint8_t kEvacPhaseEvictBit = uint8_t{1} << 1;
inline constexpr uint8_t kEvacPhaseMask =
    kEvacPhaseMarkBit | kEvacPhaseEvictBit;

inline constexpr uint8_t evac_phase_bit(EvacPhase phase) noexcept {
    return phase == EvacPhase::Mark ? kEvacPhaseMarkBit
                                    : kEvacPhaseEvictBit;
}

inline constexpr EvacPhaseCoverage evac_phase_coverage_from_mask(
    uint8_t mask) noexcept {
    const uint8_t active = mask & kEvacPhaseMask;
    if (active == kEvacPhaseMarkBit) return EvacPhaseCoverage::MarkOnly;
    if (active == kEvacPhaseEvictBit) return EvacPhaseCoverage::EvictOnly;
    if (active == kEvacPhaseMask) return EvacPhaseCoverage::Overlap;
    return EvacPhaseCoverage::Count;
}

struct EvacPhaseScope {
    EvacPhase phase = EvacPhase::Mark;
    uint8_t bit = 0;
    uint64_t start_cycles = 0;  // boundary timestamp for diagnostics only
    bool active = false;
};

// This is embedded in ProfileData, so every OS-thread profile shard owns one
// instance.  `coverage_cycles` partitions the collected Work wall interval at
// phase-boundary changes; `phase_cycles` are derived from those slices so a
// scope that crosses reset/Work boundaries cannot add initialization time.
// Payload arrays partition successful post payloads by coverage, traffic kind,
// and direction.
struct EvacPhaseMetricsShard {
    std::array<uint64_t, 2> phase_cycles{};
    std::array<uint64_t, 3> coverage_cycles{};
    uint64_t payload_bytes[3][2][2]{};
    uint64_t payload_ops[3][2][2]{};

    static uint64_t load_relaxed(const uint64_t &value) noexcept {
        return std::atomic_ref<uint64_t>(const_cast<uint64_t &>(value)).load(
            std::memory_order_relaxed);
    }

    static void store_relaxed(uint64_t &value, uint64_t next) noexcept {
        std::atomic_ref<uint64_t>(value).store(next, std::memory_order_relaxed);
    }

    static void add_relaxed(uint64_t &value, uint64_t delta) noexcept {
        std::atomic_ref<uint64_t>(value).fetch_add(
            delta, std::memory_order_relaxed);
    }

    void reset() noexcept {
        for (size_t phase = 0; phase < 2; ++phase) {
            store_relaxed(phase_cycles[phase], 0);
        }
        for (size_t coverage = 0; coverage < 3; ++coverage) {
            store_relaxed(coverage_cycles[coverage], 0);
        }
        for (size_t coverage = 0; coverage < 3; ++coverage) {
            for (size_t traffic = 0; traffic < 2; ++traffic) {
                for (size_t direction = 0; direction < 2; ++direction) {
                    store_relaxed(payload_bytes[coverage][traffic][direction],
                                  0);
                    store_relaxed(payload_ops[coverage][traffic][direction], 0);
                }
            }
        }
    }

    void derive_phase_cycles() noexcept {
        store_relaxed(phase_cycles[0],
                      load_relaxed(coverage_cycles[0]) +
                          load_relaxed(coverage_cycles[2]));
        store_relaxed(phase_cycles[1],
                      load_relaxed(coverage_cycles[1]) +
                          load_relaxed(coverage_cycles[2]));
    }

    void merge_from(const EvacPhaseMetricsShard &other) noexcept {
        for (size_t phase = 0; phase < 2; ++phase) {
            add_relaxed(phase_cycles[phase], load_relaxed(other.phase_cycles[phase]));
        }
        for (size_t coverage = 0; coverage < 3; ++coverage) {
            add_relaxed(coverage_cycles[coverage],
                        load_relaxed(other.coverage_cycles[coverage]));
            for (size_t traffic = 0; traffic < 2; ++traffic) {
                for (size_t direction = 0; direction < 2; ++direction) {
                    add_relaxed(payload_bytes[coverage][traffic][direction],
                                load_relaxed(
                                    other.payload_bytes[coverage][traffic][direction]));
                    add_relaxed(payload_ops[coverage][traffic][direction],
                                load_relaxed(
                                    other.payload_ops[coverage][traffic][direction]));
                }
            }
        }
    }
};

// A relaxed read of this mask is the only shared operation on the per-RDMA
// hook.  The enter/leave functions update it only at phase boundaries.
extern std::atomic<uint8_t> evac_phase_active_mask;
extern std::atomic_bool evac_phase_metrics_collecting;

bool evac_phase_metrics_enabled() noexcept;
EvacPhaseScope evac_phase_enter(EvacPhase phase) noexcept;
void evac_phase_leave(EvacPhaseScope &scope) noexcept;
void reset_evac_phase_metrics_state() noexcept;
void begin_evac_phase_metrics_window() noexcept;
void end_evac_phase_metrics_window() noexcept;

}  // namespace FarLib::profile
