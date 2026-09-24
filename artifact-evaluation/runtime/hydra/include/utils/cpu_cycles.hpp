#pragma once

#include <x86intrin.h>

#include <chrono>
#include <cstdint>

inline uint64_t get_cycles() {
    unsigned int _;
    return __rdtscp(&_);
}

inline uint64_t get_time_ns() {
    std::chrono::nanoseconds ns =
        std::chrono::high_resolution_clock::now().time_since_epoch();
    return ns.count();
}

inline void delay_cycles(int32_t cycles) {
    uint64_t start = __rdtsc();
    while (__rdtsc() < start + cycles);
}