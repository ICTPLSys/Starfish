#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cache/cache.hpp"
#include "utils/cpu_cycles.hpp"
#include "utils/uthreads.hpp"

namespace FarLib {

namespace detail {

inline bool spinlock_diag_enabled() {
    static const bool enabled = std::getenv("FARLIB_SPINLOCK_DIAG") != nullptr;
    return enabled;
}

inline uint64_t spinlock_diag_first_wait_loops() {
    const char *value = std::getenv("FARLIB_SPINLOCK_DIAG_FIRST_LOOPS");
    return value ? std::strtoull(value, nullptr, 0) : (1ull << 20);
}

inline uint64_t spinlock_diag_wait_period_loops() {
    const char *value = std::getenv("FARLIB_SPINLOCK_DIAG_PERIOD_LOOPS");
    return value ? std::strtoull(value, nullptr, 0) : (1ull << 22);
}

inline uint64_t spinlock_diag_slow_cycles() {
    const char *value = std::getenv("FARLIB_SPINLOCK_DIAG_SLOW_CYCLES");
    return value ? std::strtoull(value, nullptr, 0) : 1000000000ull;
}

inline const char *&spinlock_diag_context() {
    static thread_local const char *context = nullptr;
    return context;
}

inline uintptr_t &spinlock_diag_arg0() {
    static thread_local uintptr_t arg = 0;
    return arg;
}

inline uint64_t &spinlock_diag_arg1() {
    static thread_local uint64_t arg = 0;
    return arg;
}

inline void spinlock_diag_log(const char *event, const void *lock,
                              uint64_t loops, uint64_t cycles) {
    const char *context = spinlock_diag_context();
    std::fprintf(stderr,
                 "spinlock_diag event=%s lock=%p context=%s loops=%lu "
                 "cycles=%lu arg0=%lu arg1=%lu\n",
                 event, lock, context ? context : "-",
                 static_cast<unsigned long>(loops),
                 static_cast<unsigned long>(cycles),
                 static_cast<unsigned long>(spinlock_diag_arg0()),
                 static_cast<unsigned long>(spinlock_diag_arg1()));
}

}  // namespace detail

class SpinlockDiagScope {
public:
    explicit SpinlockDiagScope(const char *context, uintptr_t arg0 = 0,
                               uint64_t arg1 = 0)
        : prev_(detail::spinlock_diag_context()),
          prev_arg0_(detail::spinlock_diag_arg0()),
          prev_arg1_(detail::spinlock_diag_arg1()) {
        detail::spinlock_diag_context() = context;
        detail::spinlock_diag_arg0() = arg0;
        detail::spinlock_diag_arg1() = arg1;
    }

    ~SpinlockDiagScope() {
        detail::spinlock_diag_context() = prev_;
        detail::spinlock_diag_arg0() = prev_arg0_;
        detail::spinlock_diag_arg1() = prev_arg1_;
    }

    SpinlockDiagScope(const SpinlockDiagScope &) = delete;
    SpinlockDiagScope &operator=(const SpinlockDiagScope &) = delete;

private:
    const char *prev_;
    uintptr_t prev_arg0_;
    uint64_t prev_arg1_;
};

class Spinlock {
public:
    Spinlock() = default;

    Spinlock(const Spinlock& obj) = delete;
    Spinlock& operator=(const Spinlock& obj) = delete;
    Spinlock(const Spinlock&& obj) = delete;
    Spinlock& operator=(const Spinlock&& obj) = delete;

    void lock() {
        const bool diag = detail::spinlock_diag_enabled();
        uint64_t start_cycles = diag ? get_cycles() : 0;
        uint64_t loops = 0;
        uint64_t next_log = diag ? detail::spinlock_diag_first_wait_loops() : 0;
        const uint64_t period =
            diag ? detail::spinlock_diag_wait_period_loops() : 0;
        while (m_flag.test_and_set(std::memory_order_acquire)) {
            ++loops;
            if (diag && next_log != 0 && loops >= next_log) {
                detail::spinlock_diag_log("wait", this, loops,
                                          get_cycles() - start_cycles);
                next_log += period ? period : next_log;
            }
            uthread::yield();
        }
        if (diag && loops != 0) {
            uint64_t cycles = get_cycles() - start_cycles;
            if (cycles >= detail::spinlock_diag_slow_cycles() ||
                loops >= detail::spinlock_diag_first_wait_loops()) {
                detail::spinlock_diag_log("acquired", this, loops, cycles);
            }
        }
    }

    void unlock() {
        // printf("%p unlock %p\n", fibre_self(), this);
        m_flag.clear(std::memory_order_release);
    }

    void lock(DereferenceScope& scope) {
        // printf("%p scope lock %p\n", fibre_self(), this);
        const bool diag = detail::spinlock_diag_enabled();
        uint64_t start_cycles = diag ? get_cycles() : 0;
        uint64_t loops = 0;
        uint64_t next_log = diag ? detail::spinlock_diag_first_wait_loops() : 0;
        const uint64_t period =
            diag ? detail::spinlock_diag_wait_period_loops() : 0;
        while (m_flag.test_and_set(std::memory_order_acquire)) {
            ++loops;
            if (diag && next_log != 0 && loops >= next_log) {
                detail::spinlock_diag_log("wait_scope", this, loops,
                                          get_cycles() - start_cycles);
                next_log += period ? period : next_log;
            }
            cache::check_memory_low(scope);
            uthread::yield();
        }
        if (diag && loops != 0) {
            uint64_t cycles = get_cycles() - start_cycles;
            if (cycles >= detail::spinlock_diag_slow_cycles() ||
                loops >= detail::spinlock_diag_first_wait_loops()) {
                detail::spinlock_diag_log("acquired_scope", this, loops,
                                          cycles);
            }
        }
        // printf("%p scope lock end %p\n", fibre_self(), this);
    }

private:
    std::atomic_flag m_flag = ATOMIC_FLAG_INIT;
};

}  // namespace FarLib
