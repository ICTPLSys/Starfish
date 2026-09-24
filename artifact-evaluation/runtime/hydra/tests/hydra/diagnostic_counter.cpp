#ifdef NDEBUG
#undef NDEBUG
#endif

#include "hydra/diagnostic_counter.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <latch>
#include <thread>
#include <vector>

namespace {

using FarLib::hydra::ShardedDiagnosticCounter;

void test_initial_and_independent_counters() {
    ShardedDiagnosticCounter zero;
    ShardedDiagnosticCounter initial(41);
    assert(zero.load() == 0);
    assert(initial.load() == 41);

    zero.fetch_add(3);
    initial.fetch_add(1);
    assert(zero.load() == 3);
    assert(initial.load() == 42);

    ShardedDiagnosticCounter other(9);
    other.fetch_add(7);
    assert(initial.load() == 42);
    assert(other.load() == 16);
}

void test_multithreaded_accumulation() {
    constexpr size_t kWriters = 8;
    constexpr size_t kAddsPerWriter = 25'000;
    ShardedDiagnosticCounter counter(17);
    std::latch ready(kWriters + 1);
    std::latch start(1);
    std::vector<std::thread> writers;
    writers.reserve(kWriters);
    for (size_t i = 0; i < kWriters; ++i) {
        writers.emplace_back([&] {
            ready.count_down();
            start.wait();
            for (size_t n = 0; n < kAddsPerWriter; ++n) {
                counter.fetch_add(1);
            }
        });
    }
    ready.count_down();
    ready.wait();
    start.count_down();
    for (auto &writer : writers) writer.join();
    assert(counter.load() ==
           17 + static_cast<uint64_t>(kWriters * kAddsPerWriter));
}

void test_concurrent_read_write() {
    constexpr size_t kWriters = 4;
    constexpr size_t kReaders = 4;
    constexpr size_t kAddsPerWriter = 20'000;
    constexpr size_t kLoadsPerReader = 30'000;
    ShardedDiagnosticCounter counter;
    std::latch ready(kWriters + kReaders + 1);
    std::latch start(1);
    std::atomic<uint64_t> observed_loads{0};
    std::vector<std::thread> workers;
    workers.reserve(kWriters + kReaders);

    for (size_t i = 0; i < kWriters; ++i) {
        workers.emplace_back([&] {
            ready.count_down();
            start.wait();
            for (size_t n = 0; n < kAddsPerWriter; ++n) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (size_t i = 0; i < kReaders; ++i) {
        workers.emplace_back([&] {
            ready.count_down();
            start.wait();
            uint64_t local = 0;
            for (size_t n = 0; n < kLoadsPerReader; ++n) {
                local ^= counter.load(std::memory_order_relaxed);
            }
            observed_loads.fetch_add(local, std::memory_order_relaxed);
        });
    }

    ready.count_down();
    ready.wait();
    start.count_down();
    for (auto &worker : workers) worker.join();

    // The concurrent reads are intentionally not asserted as a snapshot.  The
    // post-join load is exact and checks every writer update.
    (void)observed_loads.load(std::memory_order_relaxed);
    assert(counter.load(std::memory_order_acquire) ==
           static_cast<uint64_t>(kWriters * kAddsPerWriter));
}

}  // namespace

int main() {
    test_initial_and_independent_counters();
    test_multithreaded_accumulation();
    test_concurrent_read_write();
    return 0;
}
