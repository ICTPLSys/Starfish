#pragma once
#include <atomic>
#include <cstdio>
#include <cstdlib>
namespace FarLib::ec_benchmark_phase {
// Runtime-only experimental policy; not a Configure field.
inline bool enabled() {
    static const bool value = [] {
        const char *v = std::getenv("FARLIB_EC_BENCHMARK_PHASED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return value;
}
inline std::atomic<bool> work_started{false};
inline void begin_runtime() {
    if (enabled()) work_started.store(false, std::memory_order_release);
}
inline bool steady() {
    return enabled() && work_started.load(std::memory_order_acquire);
}
inline void begin_work() {
    if (enabled() && !work_started.exchange(true, std::memory_order_acq_rel))
        std::fprintf(stderr, "INFO: ec_benchmark_phase transition=steady init=full_stripe work=one_sided_rmw sticky=1\n");
}
}
