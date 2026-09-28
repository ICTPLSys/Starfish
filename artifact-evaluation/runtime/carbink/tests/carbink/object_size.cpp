// Carbink object-size microbenchmark.
//
// This preserves the original 512-byte workload shape, full warm-up, Zipf/random
// selection, fibre worker structure, HDR sampling, pacing, and phase timing.
// It is intentionally not a CTest or an automatic run: pass a real two-host
// configuration and keep the original capacity/environment settings.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/accessor.hpp"
#include "cache/cache.hpp"
#include "cache/alloc/memory_usage.hpp"
#include "utils/control.hpp"
#include "utils/cpu_cycles.hpp"
#include "utils/fork_join.hpp"
#include "utils/perf.hpp"
#include "utils/stats.hpp"
#include "utils/zipfian.hpp"

#include <hdr/hdr_histogram.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define PRESSURE_TEST

namespace {

using namespace FarLib;
using namespace FarLib::cache;

constexpr size_t kObjectSize = 512;
constexpr size_t kSpanSize = 8192;
constexpr size_t kDefaultBenchmarkSecs = 20;

struct Object {
    char data[kObjectSize];
};

double zipf_skew = 0.0;
uint64_t op_duration_ns = 0;
size_t benchmark_secs = kDefaultBenchmarkSecs;
size_t total_memory_size = 0;
size_t hot_vec_size = 0;
size_t total_object_num = 0;

std::vector<UniqueFarPtr<Object>> hot_object_ptr_list;
std::vector<ZipfianGenerator<false>> zipfian_generators;
hdr_histogram *hist_latency_total = nullptr;

std::atomic_bool is_finished{false};
std::atomic_size_t total_request_num_finished{0};
std::atomic_size_t warmup_hot_done{0};
std::atomic_size_t warmup_threads_done{0};

bool verify_mode() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_OBJECT_SIZE_VERIFY");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

size_t read_env_size_t(const char *name, size_t fallback) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return fallback;
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0' || parsed == 0) return fallback;
    return static_cast<size_t>(parsed);
}

size_t benchmark_seconds_override() {
    return read_env_size_t("FARLIB_OBJECT_SIZE_BENCHMARK_SECS",
                           kDefaultBenchmarkSecs);
}

size_t phase_count() {
    const size_t phases = read_env_size_t("FARLIB_OBJECT_SIZE_PHASES", 3);
    return phases > 3 ? 3 : phases;
}

size_t warmup_timeout_seconds() {
    return read_env_size_t("FARLIB_OBJECT_SIZE_WARMUP_TIMEOUT_S", 120);
}

size_t total_memory_size_override() {
    return read_env_size_t("FARLIB_OBJECT_SIZE_TOTAL_MEMORY_BYTES",
                           16ull * 1024ull * 1024ull * 1024ull);
}

uint8_t verify_byte(size_t object_index, size_t byte_index) {
    uint64_t value = (static_cast<uint64_t>(object_index) + 1) *
                     UINT64_C(0x9e3779b97f4a7c15);
    value ^= (static_cast<uint64_t>(byte_index) + 1) *
             UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    return static_cast<uint8_t>(value & 0xffu);
}

void initialize_object(Object *object, size_t object_index) {
    assert(object != nullptr);
    if (!verify_mode()) {
        std::memset(object->data, 0, kObjectSize);
        return;
    }
    for (size_t byte = 0; byte < kObjectSize; ++byte)
        object->data[byte] = static_cast<char>(
            verify_byte(object_index, byte));
}

void verify_object(const Object *object, size_t object_index) {
    assert(object != nullptr);
    for (size_t byte = 0; byte < kObjectSize; ++byte) {
        const uint8_t expected = verify_byte(object_index, byte);
        if (static_cast<uint8_t>(object->data[byte]) != expected) {
            std::cerr << "OBJECT_SIZE_VERIFY_MISMATCH object=" << object_index
                      << " byte=" << byte << " expected="
                      << static_cast<unsigned>(expected) << " actual="
                      << static_cast<unsigned>(
                             static_cast<uint8_t>(object->data[byte]))
                      << std::endl;
            std::abort();
        }
    }
}

void warm_up(size_t thread_count) {
    warmup_hot_done.store(0, std::memory_order_relaxed);
    warmup_threads_done.store(0, std::memory_order_relaxed);
    std::atomic_bool warmup_finished{false};
    const size_t timeout = warmup_timeout_seconds();

    std::thread watchdog([&] {
        const auto begin = std::chrono::steady_clock::now();
        while (!warmup_finished.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (warmup_finished.load(std::memory_order_acquire)) break;
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - begin).count();
            std::cout << "warm up progress: hot_done="
                      << warmup_hot_done.load(std::memory_order_relaxed) << "/"
                      << hot_vec_size << " threads_done="
                      << warmup_threads_done.load(std::memory_order_relaxed)
                      << "/" << thread_count << " elapsed_s=" << elapsed
                      << std::endl;
            if (elapsed >= static_cast<long long>(timeout)) {
                std::cerr << "warm_up exceeded " << timeout << "s"
                          << std::endl;
                std::abort();
            }
        }
    });

    uthread::fork_join(thread_count, [&](size_t tid) {
        RootDereferenceScope scope;
        std::vector<LiteAccessor<Object, true>> accessors;
        accessors.reserve(std::max<size_t>(1, kSpanSize / kObjectSize));
        const size_t begin = hot_vec_size / thread_count * tid;
        const size_t end = (tid + 1 == thread_count)
                               ? hot_vec_size
                               : hot_vec_size / thread_count * (tid + 1);
        for (size_t index = begin; index < end; ++index) {
            auto accessor =
                hot_object_ptr_list[index].allocate_lite_uninitialized<true>(
                    scope);
            initialize_object(accessor.as_ptr(), index);
            accessors.push_back(std::move(accessor));
            if (accessors.size() == accessors.capacity()) accessors.clear();
            warmup_hot_done.fetch_add(1, std::memory_order_relaxed);
        }
        accessors.clear();
        warmup_threads_done.fetch_add(1, std::memory_order_relaxed);
    });

    warmup_finished.store(true, std::memory_order_release);
    watchdog.join();
}

size_t run_phase(size_t thread_count, size_t phase) {
    total_request_num_finished.store(0, std::memory_order_relaxed);
    is_finished.store(false, std::memory_order_release);
    const auto reads_before = profile::collect_rdma_read_post_count();
    const auto read_bytes_before = profile::collect_rdma_read_post_bytes();
    const auto writes_before = profile::collect_rdma_write_post_count();
    const auto write_bytes_before = profile::collect_rdma_write_post_bytes();

    auto benchmark_fn = [&](size_t tid) {
        std::default_random_engine random_engine(std::random_device{}());
        RootDereferenceScope scope;
        const uint64_t start_time = get_time_ns();
        uint64_t next_op_time = start_time;
        size_t operations = 0;
        if (tid == 0) {
            std::cout << "start benchmarking phase=" << phase
                      << " seconds=" << benchmark_secs << std::endl;
            std::cout << "Current Time (Cycles): " << get_cycles() << std::endl;
        }
        auto zipfian_generator = zipfian_generators[tid];

        while (!is_finished.load(std::memory_order_acquire)) {
            if (get_time_ns() - start_time >= benchmark_secs * 1000000000ull) {
                is_finished.store(true, std::memory_order_release);
                break;
            }
            const auto start_tsc = get_cycles();
            const size_t index = static_cast<size_t>(
                zipfian_generator(random_engine));

#ifdef PRESSURE_TEST
            ON_MISS_BEGIN
                uthread::yield();
            ON_MISS_END
            auto object = hot_object_ptr_list[index].access(__on_miss__, scope);
#else
            auto object = hot_object_ptr_list[index].access(scope);
#endif
            volatile char last_byte = object->data[kObjectSize - 1];
            (void)last_byte;
            if (verify_mode() &&
                static_cast<uint8_t>(last_byte) !=
                    verify_byte(index, kObjectSize - 1)) {
                std::cerr << "OBJECT_SIZE_TIMED_LAST_BYTE_MISMATCH object="
                          << index << " expected="
                          << static_cast<unsigned>(verify_byte(index,kObjectSize-1))
                          << " actual=" << static_cast<unsigned>(
                                 static_cast<uint8_t>(last_byte))
                          << " prefix=";
                for (size_t byte=0; byte<16; ++byte)
                    std::cerr << static_cast<unsigned>(
                        static_cast<uint8_t>(object->data[byte])) << ',';
                std::cerr << std::endl;
                verify_object(object.as_ptr(),index);
                std::abort();
            }

            const auto end_tsc = get_cycles();
            if ((operations & 1023u) == 0)
                hdr_record_value_atomic(hist_latency_total,
                                         end_tsc - start_tsc);
            next_op_time += op_duration_ns;
            while (get_time_ns() < next_op_time) {
            }
            ++operations;
        }
        total_request_num_finished.fetch_add(operations,
                                             std::memory_order_relaxed);
    };

    const auto begin = std::chrono::steady_clock::now();
    uthread::fork_join(thread_count, benchmark_fn, "benchmark_fn");
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();

    const auto reads =
        profile::collect_rdma_read_post_count() - reads_before;
    const auto read_bytes =
        profile::collect_rdma_read_post_bytes() - read_bytes_before;
    const auto writes =
        profile::collect_rdma_write_post_count() - writes_before;
    const auto write_bytes =
        profile::collect_rdma_write_post_bytes() - write_bytes_before;
    const size_t operations =
        total_request_num_finished.load(std::memory_order_relaxed);

    std::cout << std::setprecision(9) << "OBJECT_SIZE_PHASE phase=" << phase
              << " workers=" << thread_count
              << " object_size=" << kObjectSize
              << " total_memory_bytes=" << total_memory_size
              << " seconds=" << seconds << " operations=" << operations
              << " ops_per_sec=" << (seconds > 0 ? operations / seconds : 0)
              << " rdma_read_posts=" << reads
              << " rdma_read_bytes=" << read_bytes
              << " rdma_write_posts=" << writes
              << " rdma_write_bytes=" << write_bytes
              << " verify=" << (verify_mode() ? 1 : 0) << std::endl;
    return operations;
}

void verify_all_objects() {
    if (!verify_mode()) return;
    std::cerr << "OBJECT_SIZE_VERIFY_BEGIN objects=" << hot_vec_size
              << std::endl;
    const size_t thread_count =
        std::max<size_t>(1, get_config().max_thread_cnt);
    uthread::fork_join(thread_count, [&](size_t tid) {
        RootDereferenceScope scope;
        const size_t begin = hot_vec_size / thread_count * tid;
        const size_t end = (tid + 1 == thread_count)
                               ? hot_vec_size
                               : hot_vec_size / thread_count * (tid + 1);
        for (size_t index = begin; index < end; ++index) {
            auto object = hot_object_ptr_list[index].access(scope);
            verify_object(object.as_ptr(), index);
        }
    }, "object_size_verify");
    std::cerr << "OBJECT_SIZE_VERIFY_PASS objects=" << hot_vec_size
              << " bytes_per_object=" << kObjectSize << std::endl;
}

void print_cdf(const std::string &path) {
    FILE *file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        std::perror("fopen hdr output");
        std::abort();
    }
    hdr_percentiles_print(hist_latency_total, file, 100, 1, CLASSIC);
    std::fclose(file);
}

void print_hdr_percentiles() {
    if (hist_latency_total == nullptr) return;
    std::cout << "latency_mean_cycles=" << hdr_mean(hist_latency_total)
              << " p50_cycles=" << hdr_value_at_percentile(hist_latency_total, 50)
              << " p90_cycles=" << hdr_value_at_percentile(hist_latency_total, 90)
              << " p95_cycles=" << hdr_value_at_percentile(hist_latency_total, 95)
              << " p99_cycles=" << hdr_value_at_percentile(hist_latency_total, 99)
              << " p99.9_cycles="
              << hdr_value_at_percentile(hist_latency_total, 99.9)
              << std::endl;
}
void do_work_with_profile(size_t thread_count, size_t phase) {
    profile::reset_all();
    hdr_reset(hist_latency_total);
    const PerfResult result = perf_profile([&] {
        profile::start_work();
        profile::thread_start_work();
        const size_t operations = run_phase(thread_count, phase);
        std::cout << "OBJECT_SIZE_PHASE_OPERATIONS phase=" << phase
                  << " operations=" << operations << std::endl;
        profile::thread_end_work();
        profile::end_work();
    });
    result.print();
    profile::print_profile_data();
    print_hdr_percentiles();
}

}  // namespace

int main(int argc, const char *argv[]) {
    if (argc != 5 && argc != 6) {
        std::cerr << "usage: " << argv[0]
                  << " <config_path> <hdr_file> <zipf_skew> <op_duration_ns>"
                  << std::endl;
        return 2;
    }

    std::string config_path = argv[1];
    std::string hdr_file;
    if (argc == 6) {
        hdr_file = argv[3];
        zipf_skew = std::stod(argv[4]);
        op_duration_ns = std::stoull(argv[5]);
    } else {
        hdr_file = argv[2];
        zipf_skew = std::stod(argv[3]);
        op_duration_ns = std::stoull(argv[4]);
    }
    benchmark_secs = benchmark_seconds_override();

    rdma::Configure config;
    config.from_file(config_path.c_str());
    runtime_init(config);
    perf_init();
    hdr_init(1, 10'000'000, 3, &hist_latency_total);

    total_memory_size = total_memory_size_override();
    hot_vec_size = total_memory_size / kObjectSize;
    total_object_num = hot_vec_size;
    if (hot_vec_size == 0 || hot_vec_size > static_cast<size_t>(INT32_MAX))
        std::abort();

    std::cout << "hot_vec_size: " << hot_vec_size
              << " total_memory_size: " << total_memory_size
              << " object_size: " << kObjectSize
              << " benchmark_secs: " << benchmark_secs
              << " phases: " << phase_count()
              << " verify: " << (verify_mode() ? 1 : 0) << std::endl;
    hot_object_ptr_list.resize(hot_vec_size);

    const size_t max_workers = config.max_thread_cnt * 4;
    zipfian_generators.clear();
    zipfian_generators.reserve(max_workers);
    for (size_t tid = 0; tid < max_workers; ++tid) {
        zipfian_generators.emplace_back(
            static_cast<int>(total_object_num), zipf_skew);
    }

    allocator::print_combined_memory_usage();
    warm_up(config.max_thread_cnt);
    std::cout << "warm up done" << std::endl;

    for (size_t phase = 1; phase <= phase_count(); ++phase) {
        do_work_with_profile(config.max_thread_cnt * phase, phase);
    }

    std::cout << "=== memory snapshot before final verification/quiescence ==="
              << std::endl;
    allocator::print_combined_memory_usage();
    verify_all_objects();

    // Keep the requested teardown boundary: verify while objects are still
    // live, then quiesce workers, clear handles, and destroy the runtime.
    runtime_quiesce_cache();
    hot_object_ptr_list.clear();
    zipfian_generators.clear();
    print_cdf(hdr_file);
    hdr_close(hist_latency_total);
    runtime_destroy();
    std::cout << "OBJECT_SIZE_PASS cleanup complete" << std::endl;
    return 0;
}
