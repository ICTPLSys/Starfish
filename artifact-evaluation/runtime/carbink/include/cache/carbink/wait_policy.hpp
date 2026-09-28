#pragma once
namespace FarLib::cache::carbink {
inline bool poll_global_wait_cqs(bool carbink_mode, bool stable_page,
                                 bool any_endpoint_dead) noexcept {
    // Healthy full-span Carbink READs have a stable owner and exact CQ.
    // Keep the legacy global progress fallback for failures and other paths.
    return !carbink_mode || !stable_page || any_endpoint_dead;
}
inline bool yield_periodically_in_wait(bool carbink_mode, bool stable_page,
                                       bool any_endpoint_dead) noexcept {
    // Healthy whole-span reads keep polling their exact CQ without the
    // inherited every-1024-round scheduler yield. Recovery and generic paths
    // retain their established cooperative-progress behavior.
    return !carbink_mode || !stable_page || any_endpoint_dead;
}
} // namespace FarLib::cache::carbink
