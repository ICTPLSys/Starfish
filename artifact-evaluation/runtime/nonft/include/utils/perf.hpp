#pragma once
#ifndef FARLIB_HAS_PAPI
#define FARLIB_HAS_PAPI 0
#endif

#if FARLIB_HAS_PAPI
extern "C" {
#include <papi.h>
}
#endif

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "utils/cpu_cycles.hpp"
#include "utils/debug.hpp"
#include "utils/uthreads.hpp"

namespace FarLib {
// this value may modified when kernel changed.
static constexpr size_t TID_OFFS = 0x2d0;
static constexpr bool RESULT_PER_TH = false;
struct pthread_fake {
    char padding[TID_OFFS];
    pid_t tid;
};
struct PerfResult {
    double runtime_ms;
    // Elapsed RDTSCP/TSC ticks for the measured wall interval, not PAPI core
    // cycles and not the sum over workers. Both perf_profile branches use TSC.
    size_t total_cycles;
    size_t instructions;
    size_t l2_cache_miss;
    size_t l3_cache_miss;

    void print(size_t stw_fibre_count = 0,
               int64_t stw_mutator_cycles = 0) const {
        std::cout << std::internal << std::setw(32) << "perf result"
                  << std::endl;
        std::cout << std::right << std::string(32, '-') << std::endl;
        std::cout << std::setw(16) << "runtime: " << std::setw(16) << std::fixed
                  << runtime_ms << " ms" << std::endl;
        std::cout << std::setw(16) << "cycles: " << std::setw(16)
                  << total_cycles << std::endl;
        std::cout << std::setw(16) << "Insttructions: " << std::setw(16)
                  << instructions << std::endl;
        std::cout << std::setw(16) << "L2 miss: " << std::setw(16)
                  << l2_cache_miss << std::endl;
        std::cout << std::setw(16) << "L3 miss: " << std::setw(16)
                  << l3_cache_miss << std::endl;
        // S = summed allocation-wait TSC ticks; C = wall-interval TSC ticks;
        // N = application fibres participating in this same measurement.
        // Mean wait_ms = (S/N) * runtime_ms/C; wait_share = S/(N*C).
        // Do not substitute OS threads for N or mix a TSC numerator with
        // PAPI_TOT_CYC. The mean is NOT time when all fibres are stopped.
        // B/(1-wait_share) is only a hypothetical no-wait throughput model:
        // it assumes unchanged active processing rate, miss mix, and no new
        // bottleneck. Other fibres/RDMA may already overlap these waits.
        if (stw_fibre_count != 0 && stw_mutator_cycles >= 0 &&
            total_cycles != 0 && runtime_ms > 0.0) {
            const double cycles_per_fibre =
                static_cast<double>(stw_mutator_cycles) /
                static_cast<double>(stw_fibre_count);
            const double mean_fibre_wait_ms =
                cycles_per_fibre * runtime_ms /
                static_cast<double>(total_cycles);
            const double mean_fibre_wait_share_pct =
                100.0 * mean_fibre_wait_ms / runtime_ms;
            const auto old_flags = std::cout.flags();
            const auto old_precision = std::cout.precision();
            std::cout << "allocation_wait_summary"
                      << " fibre_count=" << stw_fibre_count
                      << " sum_tsc_cycles=" << stw_mutator_cycles
                      << " wall_tsc_cycles=" << total_cycles
                      << " cycles_per_fibre=" << std::fixed
                      << std::setprecision(2) << cycles_per_fibre
                      << " mean_fibre_wait_ms=" << mean_fibre_wait_ms
                      << " mean_fibre_wait_share_pct=" << mean_fibre_wait_share_pct
                      << std::endl;
            std::cout.flags(old_flags);
            std::cout.precision(old_precision);
        }
    }
};

template <typename Fn>
PerfResult perf_profile(Fn&& fn) {
#if FARLIB_HAS_PAPI
    constexpr size_t PAPIEventCount = 3;
    int papi_events[PAPIEventCount] = {PAPI_TOT_INS, PAPI_L2_TCM, PAPI_L3_TCM};
    std::vector<int> events;
    long long papi_values[PAPIEventCount] = {0};
    size_t worker_num = uthread::get_worker_count();
    for (int i = 0; i < worker_num; i++) {
        auto pthread_id = uthread::es->get_tids()[i];
        auto id = ((struct pthread_fake*)pthread_id)->tid;
        int papi_event_set = PAPI_NULL;
        ASSERT(PAPI_create_eventset(&papi_event_set) == PAPI_OK);
        ASSERT(PAPI_add_events(papi_event_set, papi_events, PAPIEventCount) ==
               PAPI_OK);
        ASSERT(PAPI_attach(papi_event_set, id) == PAPI_OK);
        events.push_back(papi_event_set);
    }
    for (auto e : events) {
        ASSERT(PAPI_start(e) == PAPI_OK);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto start_cycles = get_cycles();

    fn();

    auto end_cycles = get_cycles();
    auto end = std::chrono::high_resolution_clock::now();

    for (auto e : events) {
        long long th_values[PAPIEventCount] = {0};
        ASSERT(PAPI_stop(e, th_values) == PAPI_OK);
        ASSERT(PAPI_detach(e) == PAPI_OK);
        ASSERT(PAPI_cleanup_eventset(e) == PAPI_OK);
        ASSERT(PAPI_destroy_eventset(&e) == PAPI_OK);
        for (int i = 0; i < PAPIEventCount; i++) {
            papi_values[i] += th_values[i];
        }
        if constexpr (RESULT_PER_TH) {
            PerfResult result;
            result.instructions = th_values[0];
            result.l2_cache_miss = th_values[1];
            result.l3_cache_miss = th_values[2];
            std::cout << "-------------------" << std::endl;
            result.print();
            std::cout << "-------------------" << std::endl;
        }
    }

    std::chrono::duration<double, std::milli> duration = end - start;

    PerfResult result;
    result.runtime_ms = duration.count();
    result.total_cycles = end_cycles - start_cycles;
    result.instructions = papi_values[0];
    result.l2_cache_miss = papi_values[1];
    result.l3_cache_miss = papi_values[2];
    return result;
#else
    auto start = std::chrono::high_resolution_clock::now();
    auto start_cycles = get_cycles();
    fn();
    auto end_cycles = get_cycles();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    PerfResult result;
    result.runtime_ms = duration.count();
    result.total_cycles = end_cycles - start_cycles;
    result.instructions = 0;
    result.l2_cache_miss = 0;
    result.l3_cache_miss = 0;
    return result;
#endif
}

inline void perf_init() {
#if FARLIB_HAS_PAPI
    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT) {
        std::cerr << "PAPI library init failed, with code " << ret
                  << std::endl;
    }
    ASSERT(ret == PAPI_VER_CURRENT);
#endif
}

}  // namespace FarLib
