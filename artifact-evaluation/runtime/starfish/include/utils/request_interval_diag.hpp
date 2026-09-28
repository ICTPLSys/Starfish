#pragma once

#include <cstddef>
#include <cstdint>

#include "utils/benchmark_phase.hpp"

// Compatibility surface for the two shared benchmark drivers. The former
// request-level observer (sampling ring, locks, and syscall/TSC probes) is
// retired; stage state is kept in benchmark_phase for the optional EC-RMW
// timing extension.
namespace FarLib::request_interval_diag {

inline void begin_stage(std::size_t fibres) noexcept {
    ::FarLib::benchmark_phase::begin_stage(fibres);
}

inline void end_stage() noexcept { ::FarLib::benchmark_phase::end_stage(); }

// Retain the shared-app call shape; request-level consumer tracing is gone.
inline void consumer_end(std::size_t, uint64_t) noexcept {}
inline void dump() noexcept {}

}  // namespace FarLib::request_interval_diag
