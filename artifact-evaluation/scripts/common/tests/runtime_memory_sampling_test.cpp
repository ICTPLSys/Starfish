#include "benchmark_memory.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>

using namespace FarLib::benchmark_memory;
constexpr uint64_t second = 1'000'000'000ULL;

static void configure(uint64_t start) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    assert(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
    int cpu = 0;
    while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &allowed)) ++cpu;
    assert(cpu < CPU_SETSIZE);
    setenv("FARLIB_REMOTE_MEMORY_SAMPLES", "1", 1);
    setenv("FARLIB_REMOTE_MEMORY_OBSERVER_CPU", std::to_string(cpu).c_str(), 1);
    setenv("FARLIB_BENCHMARK_START_MONOTONIC_NS", std::to_string(start).c_str(), 1);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string mode(argv[1]);
    if (mode == "schedule") {
        DeadlineSchedule schedule(100);
        assert(schedule.take_due(100 + 9 * second, 100).empty());
        for (uint64_t target : kScheduledSeconds) {
            auto due = schedule.take_due(100 + target * second, 100);
            assert(due.size() == 1);
            assert(due[0].scheduled_elapsed_s == target);
            assert(!due[0].missed_notready);
        }
        assert(schedule.done());
        DeadlineSchedule late(0);
        auto due = late.take_due(35 * second, 25 * second);
        assert(due.size() == 3);
        assert(due[0].missed_notready && due[1].missed_notready);
        assert(!due[2].missed_notready && due[2].scheduled_elapsed_s == 30);
        assert(late.next_deadline_ns() == 40 * second);
        std::cout << "schedule passed\n";
        return 0;
    }
    if (mode == "disabled") {
        unsetenv("FARLIB_REMOTE_MEMORY_SAMPLES");
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture",
                        [] { std::abort(); return std::vector<uint64_t>{}; });
        sampler.start();
        sampler.stop();
        assert(!sampler.enabled());
        return 0;
    }
    if (mode == "future") {
        configure(monotonic_ns() + second);
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture",
                        [] { return std::vector<uint64_t>{100, 200}; });
        try {
            sampler.start();
            assert(false);
        } catch (const std::runtime_error &) {
            std::cout << "future rejected\n";
        }
        return 0;
    }
    std::atomic_int calls{0};
    auto snapshot = [&] {
        ++calls;
        return std::vector<uint64_t>{100, 200};
    };
    if (mode == "early") {
        configure(monotonic_ns());
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", snapshot);
        sampler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        const auto before = monotonic_ns();
        sampler.stop();
        sampler.stop();
        assert(monotonic_ns() - before < second);
        assert(calls == 0);
        return 0;
    }
    if (mode == "one") {
        // Virtual origin only in this unit fixture: the production interval
        // remains fixed at ten seconds and no benchmark is launched.
        configure(monotonic_ns() - 9 * second - 900'000'000ULL);
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", snapshot);
        sampler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(220));
        sampler.stop();
        assert(calls == 1);
        return 0;
    }
    if (mode == "notready") {
        configure(monotonic_ns() - 35 * second);
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", snapshot);
        sampler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        sampler.stop();
        assert(calls == 0);
        return 0;
    }
    assert(false);
}
