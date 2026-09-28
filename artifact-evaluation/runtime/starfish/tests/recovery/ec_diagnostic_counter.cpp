#include "utils/sharded_diagnostic_counter.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

using Counter = FarLib::profile::ShardedDiagnosticCounter;

bool check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    static_assert(Counter::kShardCount == 64);
    static_assert(alignof(Counter) >= 64);
    static_assert(sizeof(Counter) % 64 == 0);

    std::vector<Counter> isolated(2);
    const auto first = reinterpret_cast<std::uintptr_t>(&isolated[0]);
    const auto second = reinterpret_cast<std::uintptr_t>(&isolated[1]);
    if (!check(first % 64 == 0 && second % 64 == 0,
               "counter objects are cache-line aligned") ||
        !check(second - first >= 64,
               "counter objects do not share a cache line")) {
        return 1;
    }

    constexpr size_t kThreads = 96;
    constexpr size_t kIterations = 4000;
    Counter counter(11);
    Counter weighted(13);
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> stop_reader{false};
    std::atomic<uint64_t> read_samples{0};
    std::thread reader([&] {
        while (!stop_reader.load(std::memory_order_acquire)) {
            (void)counter.load(std::memory_order_relaxed);
            (void)weighted.load(std::memory_order_relaxed);
            read_samples.fetch_add(1, std::memory_order_relaxed);
        }
    });
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (size_t thread = 0; thread < kThreads; ++thread) {
        workers.emplace_back([&, thread] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (size_t i = 0; i < kIterations; ++i) {
                counter.fetch_add(1, std::memory_order_relaxed);
                weighted.fetch_add(static_cast<uint64_t>(thread + 1),
                                   std::memory_order_relaxed);
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kThreads) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) worker.join();
    stop_reader.store(true, std::memory_order_release);
    reader.join();

    const uint64_t expected_counter =
        11 + static_cast<uint64_t>(kThreads) * kIterations;
    const uint64_t expected_weighted =
        13 + static_cast<uint64_t>(kIterations) *
                 (static_cast<uint64_t>(kThreads) * (kThreads + 1) / 2);
    if (!check(counter.load(std::memory_order_relaxed) == expected_counter,
               "post-join counter sum is exact") ||
        !check(weighted.load(std::memory_order_relaxed) == expected_weighted,
               "independent post-join counter sum is exact") ||
        !check(read_samples.load(std::memory_order_relaxed) != 0,
               "concurrent loads completed")) {
        return 1;
    }
    std::printf("ec_diagnostic_counter PASS threads=%zu iterations=%zu total=%llu\n",
                kThreads, kIterations,
                static_cast<unsigned long long>(expected_counter));
    return 0;
}
