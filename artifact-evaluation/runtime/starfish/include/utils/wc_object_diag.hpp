#pragma once

#include <chrono>
#include <cstdint>

// Compatibility surface for shared applications that still call the retired
// WC object/window hooks.  The runtime no longer records object diagnostics;
// these operations are deliberately allocation-free and stateless.
namespace FarLib::wc_object_diag {

struct Sample {};

inline void begin_capture() noexcept {}
inline void end_capture() noexcept {}

inline Sample *begin(uint64_t, uint64_t, uint64_t) noexcept { return nullptr; }
inline void access_done(Sample *) noexcept {}
inline void finish(Sample *) noexcept {}

inline void window(const char *, uint64_t,
                   std::chrono::steady_clock::time_point,
                   std::chrono::steady_clock::time_point) noexcept {}
inline void dump() noexcept {}

}  // namespace FarLib::wc_object_diag
