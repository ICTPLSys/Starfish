#include "benchmark_memory.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace FarLib::benchmark_memory;
constexpr uint64_t second = 1'000'000'000ULL;

static void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "fixture failure: " << message << "\n";
        std::exit(2);
    }
}

static int first_allowed_cpu() {
    cpu_set_t allowed;
    require(sched_getaffinity(0, sizeof(allowed), &allowed) == 0,
            "sched_getaffinity");
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &allowed)) return cpu;
    require(false, "no allowed CPU");
    return 0;
}

static void configure(const char *profile = "normal",
                      const char *origin = "profile_start_work") {
    setenv("FARLIB_REMOTE_MEMORY_SAMPLES", "1", 1);
    setenv("FARLIB_REMOTE_MEMORY_OBSERVER_CPU",
           std::to_string(first_allowed_cpu()).c_str(), 1);
    setenv("FARLIB_REMOTE_MEMORY_SAMPLING_PROFILE", profile, 1);
    setenv("FARLIB_REMOTE_MEMORY_WORK_ORIGIN", origin, 1);
}

static void wait_for(const std::atomic<bool> &flag, const char *message) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (!flag.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    require(flag.load(std::memory_order_acquire), message);
}

static void wait_for_count(const std::atomic_int &value, int target,
                           const char *message) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (value.load(std::memory_order_acquire) < target &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    require(value.load(std::memory_order_acquire) >= target, message);
}

static std::vector<uint64_t> values() { return {100, 200}; }

int main(int argc, char **argv) {
    require(argc == 2, "one fixture mode");
    const std::string mode(argv[1]);

    if (mode == "schedule") {
        DeadlineSchedule schedule(100);
        require(schedule.take_due(100 + 9 * second, 100).empty(),
                "early schedule");
        for (uint64_t target : kScheduledSeconds) {
            auto due = schedule.take_due(100 + target * second, 100);
            require(due.size() == 1 && due[0].scheduled_elapsed_s == target,
                    "normal schedule point");
            require(!due[0].missed_notready, "normal ready point");
        }
        DeadlineSchedule bfs(100, {3});
        require(bfs.take_due(100 + 3 * second, 100).size() == 1,
                "BFS schedule point");
        DeadlineSchedule late(0);
        auto due = late.take_due(35 * second, 25 * second);
        require(due.size() == 3 && due[0].missed_notready &&
                    due[1].missed_notready && !due[2].missed_notready,
                "late schedule readiness");
        std::cout << "schedule passed\n";
        return 0;
    }

    if (mode == "disabled" || mode == "init_delay" ||
        mode == "destroy_before_work") {
        unsetenv("FARLIB_REMOTE_MEMORY_SAMPLES");
        setenv("FARLIB_BENCHMARK_START_MONOTONIC_NS", "garbage", 1);
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture",
                        [] { std::abort(); return std::vector<uint64_t>{}; });
        sampler.arm();
        sampler.shutdown();
        require(!sampler.enabled(), "disabled sampler");
        return 0;
    }

    if (mode == "invalid_profile") {
        configure("not-a-profile");
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", values);
        try {
            sampler.arm();
            require(false, "invalid profile rejected");
        } catch (const std::runtime_error &) {
            std::cout << "invalid profile rejected\n";
        }
        return 0;
    }

    if (mode == "old_env_ignored") {
        configure();
        setenv("FARLIB_BENCHMARK_START_MONOTONIC_NS", "not-a-number", 1);
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", values);
        sampler.arm();
        const uint64_t start = monotonic_ns() - 10 * second;
        sampler.begin_work_for_test(start, start);
        sampler.tick_for_test(monotonic_ns());
        sampler.end_work_for_test(start + 11 * second);
        sampler.shutdown();
        return 0;
    }

    if (mode == "backlog") {
        configure();
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", values);
        sampler.arm();
        const uint64_t start = monotonic_ns();
        sampler.begin_work_for_test(start, start);
        sampler.tick_for_test(start + 35 * second);
        sampler.end_work_for_test(start + 36 * second);
        sampler.shutdown();
        return 0;
    }

    if (mode == "missing_end") {
        configure();
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", values);
        sampler.arm();
        const uint64_t start = monotonic_ns();
        sampler.begin_work_for_test(start, start);
        sampler.shutdown();
        return 0;
    }

    if (mode == "origin_mismatch") {
        configure("normal", "kvs_request_start");
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture", values);
        sampler.arm();
        const uint64_t start = monotonic_ns();
        sampler.begin_work_at(start, "profile_start_work");
        sampler.end_work_at(start + 20 * second, "profile_start_work");
        sampler.shutdown();
        return 0;
    }

    if (mode == "normal" || mode == "bfs" || mode == "short" ||
        mode == "repeat" || mode == "kvs") {
        const char *profile = mode == "bfs" ? "bfs_work_3s" : "normal";
        const char *origin = mode == "kvs" ? "kvs_request_start"
                                            : "profile_start_work";
        configure(profile, origin);
        std::atomic_int calls{0};
        std::atomic_int completed{0};
        auto snapshot = [&] {
            ++calls;
            auto result = values();
            ++completed;
            return result;
        };
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture",
                        snapshot);
        sampler.arm();
        if (mode == "kvs") {
            explicit_work_origin_context = &sampler;
            explicit_work_begin_observer = &Sampler::begin_work_at_callback;
            explicit_work_end_observer = &Sampler::end_work_at_callback;
        }
        auto one_window = [&](uint64_t start, uint64_t elapsed,
                              const char *window_origin) {
            if (window_origin != nullptr)
                begin_work_at(start, window_origin);
            else
                sampler.begin_work_for_test(start, start);
            if (elapsed != 0) sampler.tick_for_test(start + elapsed * second);
            const uint64_t end_elapsed = elapsed == 2 ? elapsed : elapsed + 1;
            if (window_origin != nullptr)
                end_work_at(start + end_elapsed * second, window_origin);
            else
                sampler.end_work_for_test(start + end_elapsed * second);
        };
        if (mode == "kvs") sampler.begin_work();  // must be ignored in KVS mode
        const uint64_t start = monotonic_ns();
        if (mode == "short") {
            one_window(start, 2, nullptr);
        } else if (mode == "bfs") {
            one_window(start, 3, nullptr);
        } else if (mode == "kvs") {
            one_window(start, 3, "kvs_request_start");
        } else if (mode == "repeat") {
            one_window(start, 10, nullptr);
            wait_for_count(completed, 1, "first repeated sample");
            const uint64_t second_start = monotonic_ns();
            one_window(second_start, 10, nullptr);
        } else {
            sampler.begin_work_for_test(start, start);
            for (uint64_t offset : kScheduledSeconds)
                sampler.tick_for_test(start + offset * second);
            sampler.end_work_for_test(start + 51 * second);
        }
        explicit_work_begin_observer = nullptr;
        explicit_work_end_observer = nullptr;
        explicit_work_origin_context = nullptr;
        sampler.shutdown();
        return 0;
    }

    if (mode == "nonblocking") {
        configure();
        std::atomic<bool> entered{false};
        std::atomic<bool> release{false};
        auto snapshot = [&] {
            entered.store(true, std::memory_order_release);
            while (!release.load(std::memory_order_acquire))
                std::this_thread::yield();
            return values();
        };
        Sampler sampler(2, "allocator_occupied_bytes", "unit_fixture",
                        snapshot);
        sampler.arm();
        const uint64_t start = monotonic_ns() - 10 * second;
        sampler.begin_work_for_test(start, start);
        sampler.tick_for_test(monotonic_ns());
        wait_for(entered, "blocking snapshot entered");
        const auto before = std::chrono::steady_clock::now();
        sampler.end_work_for_test(monotonic_ns());
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - before)
                                 .count();
        require(elapsed < 100, "end hook blocked on snapshot");
        release.store(true, std::memory_order_release);
        sampler.shutdown();
        return 0;
    }

    if (mode == "prefix") {
        std::fputs("assistant answer without newline", stdout);
        std::fflush(stdout);
        emit_stdout_record(
            "runtime_remote_memory schema_version=3 sample=0 work_window_id=1 "
            "origin=profile_start_work scheduled_elapsed_s=10 "
            "observed_elapsed_s=10.25 occupied_bytes=300 "
            "metric=allocator_occupied_bytes source=unit_fixture "
            "start_monotonic_ns=1 window=work_start_10_20_30_40_50s "
            "sampling=work_start_10_20_30_40_50s snapshot_start_monotonic_ns=2 "
            "snapshot_ns=3 endpoint_count=2 endpoint_bytes=0:100,1:200 "
            "consistency=rolling_allocator_snapshot");
        return 0;
    }

    if (mode == "concurrent") {
        constexpr size_t kThreads = 8;
        constexpr size_t kRecordsPerThread = 32;
        std::vector<std::thread> writers;
        for (size_t tid = 0; tid < kThreads; ++tid) {
            writers.emplace_back([tid] {
                for (size_t seq = 0; seq < kRecordsPerThread; ++seq) {
                    std::fprintf(stdout, "partial_prompt thread=%zu seq=%zu", tid,
                                 seq);
                    std::fwrite("**", 1, 2, stdout);
                    emit_stdout_record(
                        "runtime_remote_memory_test thread=" +
                        std::to_string(tid) + " seq=" + std::to_string(seq) +
                        " occupied_bytes=300");
                }
            });
        }
        for (auto &writer : writers) writer.join();
        return 0;
    }

    require(false, "unknown fixture mode");
    return 2;
}
