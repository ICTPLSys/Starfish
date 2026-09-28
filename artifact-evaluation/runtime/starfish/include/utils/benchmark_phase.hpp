#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// Small, allocation-free stage state shared by benchmark drivers and the
// optional EC-RMW timing extension. This deliberately has no request-level
// observer, locks, syscall probes, or static record ring.
namespace FarLib::benchmark_phase {

inline std::atomic<uint32_t> stage_ordinal{0};
inline std::atomic<uint32_t> formal_fibres{0};

inline void begin_stage(std::size_t fibres) noexcept {
    stage_ordinal.fetch_add(1, std::memory_order_relaxed);
    formal_fibres.store(static_cast<uint32_t>(fibres),
                        std::memory_order_relaxed);
}

inline void end_stage() noexcept {}

}  // namespace FarLib::benchmark_phase
