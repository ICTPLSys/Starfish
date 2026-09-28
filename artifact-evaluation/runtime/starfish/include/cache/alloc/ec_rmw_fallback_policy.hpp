#pragma once
#include <cstdint>

namespace FarLib::cache::ec_rmw_runtime {
inline constexpr bool allow_background_bank_yield(
    bool background_rebuild, bool endpoint_dead, bool bank_ready) noexcept {
    return background_rebuild && endpoint_dead && !bank_ready;
}
// Once background repair freezes affected old stripes, lost writable capacity
// is a pool-wide liveness problem. Ordinary sources can fill every bounded
// producer bank, so requiring a failed source can prevent any bank retiring.
// Ordinary-source admission waits for full background reconstruction, so it
// cannot preempt capacity admission for still-unrepaired sources.
// Admission still requires a real failure; fresh stripes consume the existing
// one-credit-per-failed-stripe pool. The caller must be unreserved/unposted.
inline constexpr bool allow_capacity_replacement(
    bool recovery_enabled, bool source_eligible, bool background_rebuilt) noexcept {
    return recovery_enabled && (source_eligible || background_rebuilt);
}
// A failure fallback is safe only before writes and after every participant
// is terminal (including unposted/cancelled reads). The caller retains source
// ownership until the immutable writer has accepted it.
inline constexpr bool allow_read_failure_fallback(
    bool enabled, bool one_sided, uint8_t read_error,
    uint8_t read_terminal, bool writes_decided) noexcept {
    return enabled && one_sided && read_error != 0 &&
           read_terminal == uint8_t{7} && !writes_decided;
}
} // namespace FarLib::cache::ec_rmw_runtime
