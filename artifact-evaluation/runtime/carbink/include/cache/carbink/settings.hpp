#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include "utils/cpu_cycles.hpp"

namespace FarLib::cache::carbink {
inline size_t positive_setting(const char *name, size_t fallback) {
    const char *s = std::getenv(name);
    if (s == nullptr || !*s) return fallback;
    char *end = nullptr;
    const auto n = std::strtoull(s, &end, 10);
    return end != s && !*end && n ? static_cast<size_t>(n) : fallback;
}
inline size_t inflight_update_limit() {
    return positive_setting("FARLIB_COMPACTION_INFLIGHT_UPDATES", 128);
}
inline size_t drain_task_budget(size_t workers) {
    // Original override only applies to a multi-worker configuration.
    return workers <= 1 ? std::numeric_limits<size_t>::max()
        : positive_setting("FARLIB_COMPACTION_DRAIN_TASK_BUDGET",
                           std::numeric_limits<size_t>::max());
}
inline uint64_t max_updates_per_sec() {
    static const uint64_t rate = positive_setting(
        "FARLIB_COMPACTION_MAX_UPDATES_PER_SEC", 0);
    return rate;
}
inline void rate_limit_one_update() {
    const uint64_t rate = max_updates_per_sec();
    if (!rate) return;
    const uint64_t interval = std::max<uint64_t>(1, 1000000000ull / rate);
    static std::atomic<uint64_t> next_allowed_ns{0};
    auto now = get_time_ns();
    auto old = next_allowed_ns.load(std::memory_order_relaxed);
    for (;;) {
        const auto reservation = std::max<uint64_t>(old, now);
        if (next_allowed_ns.compare_exchange_weak(
                old, reservation + interval, std::memory_order_relaxed)) {
            while (now < reservation) {
#if defined(__x86_64__) || defined(__i386__)
                asm volatile("pause" ::: "memory");
#endif
                now = get_time_ns();
            }
            return;
        }
    }
}
} // namespace FarLib::cache::carbink
