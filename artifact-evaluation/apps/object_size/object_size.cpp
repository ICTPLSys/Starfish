// test throughtput under different object size

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdarg>
#include <sys/resource.h>
#include <hdr/hdr_histogram.h>

#include "async/scoped_inline_task.hpp"
#include "cache/cache.hpp"
#include "cache/alloc/memory_usage.hpp"
#include "utils/control.hpp"
#include "utils/zipfian.hpp"
#include "utils/perf.hpp"
#include "utils/stats.hpp"
#include "utils/wait_trace.hpp"
#include "utils/scope_diag.hpp"
#include "utils/request_interval_diag.hpp"

#define PRESSURE_TEST
namespace FarLib::allocator {
void print_thread_heap_counters_only(const char *phase, size_t fibres);
}
// #define MUT_TEST

uint64_t op_duration_ns = 0;
double zipf_skew = 0;
#ifndef OBJECT_SIZE_BYTES
#define OBJECT_SIZE_BYTES 4000
#endif
const size_t object_size = OBJECT_SIZE_BYTES;
const size_t span_size = 8192;
double local_memory_ratio = 0.2;
size_t total_memory_size = 0;
size_t hot_vec_size = 0;
size_t cold_vec_size = 0;
size_t total_object_num = 0;
size_t request_object_num = 0;
size_t total_request_num = 0;
size_t benchmark_secs = 10;

struct Object {
    char data[object_size];
};

struct ObjectSpan {
    Object objects[span_size / object_size];
};

using namespace FarLib;

std::vector<UniqueFarPtr<Object>> hot_object_ptr_list;
std::vector<UniqueFarPtr<ObjectSpan>> cold_object_span_list;

hdr_histogram *hist_latency_total = nullptr;

void warm_up(size_t thread_cnt) {
    uthread::fork_join(thread_cnt, [&](size_t tid) {
        RootDereferenceScope scope;
        std::pair<size_t, size_t> hot_vec_range;
        std::pair<size_t, size_t> cold_vec_range;
        if (tid + 1 == thread_cnt) {
            hot_vec_range = std::make_pair(hot_vec_size / thread_cnt * tid, hot_vec_size);
            cold_vec_range = std::make_pair(cold_vec_size / thread_cnt * tid, cold_vec_size);
        } else {
            hot_vec_range = std::make_pair(hot_vec_size / thread_cnt * tid, hot_vec_size / thread_cnt * (tid + 1));
            cold_vec_range = std::make_pair(cold_vec_size / thread_cnt * tid, cold_vec_size / thread_cnt * (tid + 1));
        }
        for (size_t i = hot_vec_range.first; i < hot_vec_range.second; i++) {
            auto data_ptr = hot_object_ptr_list[i].allocate_lite_uninitialized<true>(scope);
            memset(data_ptr->data, 0, object_size);
        }
        for (size_t i = cold_vec_range.first; i < cold_vec_range.second; i++) {
            auto data_ptr = cold_object_span_list[i].allocate_lite_uninitialized<true>(scope);
            memset(&(data_ptr->objects[0].data), 0, object_size * (span_size / object_size));
        }
    });

}

std::atomic_bool is_finished = false;
std::atomic_size_t total_request_num_finished = 0;

const int step = 1797;

int sequential_generator(int previous_index, size_t thread_cnt) {
    return (previous_index + step) % (request_object_num / thread_cnt);
}

std::vector<ZipfianGenerator<false>> zipfian_generators;
std::mutex zipfian_generators_mutex;

size_t run(size_t thread_cnt) {
    is_finished.store(false, std::memory_order_relaxed);
    total_request_num_finished.store(0, std::memory_order_relaxed);
    std::vector<std::random_device::result_type> rng_seeds(thread_cnt);
    auto benchmark_fn = [&](size_t tid) {
        const auto rng_seed = std::random_device{}();
        rng_seeds[tid] = rng_seed;
        std::default_random_engine random_engine(rng_seed);
        auto *sd_slot=scope_diag::register_fibre(tid, fibre_self());
        RootDereferenceScope scope;
        auto start_time = get_time_ns();
        auto next_op_time = start_time;
        size_t i = 0;
        size_t start_idx = 3;
        if (tid == 0) {
            std::cout << "start benchmarking..." << std::endl;
            std::cout << "Current Time (Cycles): " << get_cycles() << std::endl;
        }
        auto zipfian_generator = zipfian_generators[tid];
        while (!is_finished.load()) {
            if (get_time_ns() - start_time > benchmark_secs * 1000 * (size_t)(1000 * 1000)) {
                is_finished.store(true);
                break;
            }
            auto start_tsc = get_cycles();
            scope_diag::begin_op(sd_slot);

            auto idx = zipfian_generator(random_engine);
            assert(idx < hot_vec_size);

            scope_diag::Guard sd_access_guard(fibre_self(), scope_diag::ACCESS);

            #ifdef PRESSURE_TEST
            ON_MISS_BEGIN
                uthread::yield();
            ON_MISS_END

            auto obj = hot_object_ptr_list[idx].access(__on_miss__, scope);
            #ifdef MUT_TEST
            auto mut_obj = obj.as_mut();
            #endif
            volatile char temp = obj->data[object_size - 1];
            (void)temp;
            #else

            auto obj = hot_object_ptr_list[idx].access(scope);
            #ifdef MUT_TEST
            auto mut_obj = obj.as_mut();
            #endif
            volatile char temp = obj->data[object_size - 1];
            (void)temp;
            #endif

            auto end_tsc = get_cycles();
            if (i % 1024 == 0) {
                    hdr_record_value_atomic(hist_latency_total, end_tsc - start_tsc);
            }

            next_op_time += op_duration_ns;
            while (get_time_ns() < next_op_time) {};
            scope_diag::end_op(sd_slot);
            request_interval_diag::consumer_end(
                tid, reinterpret_cast<uint64_t>(fibre_self()));
            // start_idx = idx_local;
            i++;
        }

        total_request_num_finished.fetch_add(i);
    };
    uthread::fork_join(thread_cnt, benchmark_fn, "benchmark_fn");
    std::cout << "rng_seed_result access_distribution=uniform_id_v1"
              << " engine=std::default_random_engine"
              << " distribution=std::uniform_int_distribution<int>"
              << " seed_source=std::random_device once_per_fibre=1"
              << " fibre_count=" << thread_cnt << " seeds=";
    for (size_t tid = 0; tid < thread_cnt; ++tid) {
        if (tid != 0) {
            std::cout << ',';
        }
        std::cout << rng_seeds[tid];
    }
    std::cout << std::endl;
    return total_request_num_finished.load();
}

void hdr_print_percentiles(hdr_histogram *hist) {
    double mean = hdr_mean(hist);
    uint64_t p50 = hdr_value_at_percentile(hist, 50);
    uint64_t p90 = hdr_value_at_percentile(hist, 90);
    uint64_t p95 = hdr_value_at_percentile(hist, 95);
    uint64_t p99 = hdr_value_at_percentile(hist, 99);
    uint64_t p999 = hdr_value_at_percentile(hist, 99.9);
    std::cout << "mean: " << mean << " p50: " << p50 << " p90: " << p90 << " p95: " << p95 << " p99: " << p99 << " p999: " << p999 << std::endl;
}

void print_cdf(const std::string &hdr_file) {
    std::string file_latency =
        hdr_file;
    FILE *fq = fopen(file_latency.c_str(), "w");
    hdr_percentiles_print(hist_latency_total, fq, 100, 1, format_type::CLASSIC);
    fclose(fq);
}

void online_profiler() {
    // every 100ms, print the profile data
    uint64_t start_time = get_time_ns();
    uint64_t last_miss_count = 0;
    while (true) {
        if (get_time_ns() - start_time > benchmark_secs * 1000 * (size_t)(1000 * 1000)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // uint64_t miss_count = profile::collect_data_miss_count();
        // std::cout << "online profiler: " << miss_count - last_miss_count << " miss count: " << miss_count << std::endl;
        // last_miss_count = miss_count;
        // profile::print_profile_miss_count();

        std::cout << "remote used bytes: " << allocator::global_heap.get_free_size() << std::endl;
    }
}

bool profile_memory_enabled() {
    const char *env = std::getenv("OBJECT_SIZE_PROFILE_MEMORY");
    return env != nullptr && std::strtoull(env, nullptr, 0) != 0;
}

bool object_size_only_48_enabled() {
    const char *env = std::getenv("OBJECT_SIZE_ONLY_48");
    return env != nullptr && std::strtoull(env, nullptr, 0) != 0;
}

double rusage_seconds(const timeval &value) {
    return static_cast<double>(value.tv_sec) +
           static_cast<double>(value.tv_usec) / 1'000'000.0;
}

#define DO_WORK_WITH_PROFILE(thread_cnt)                                            \
{                                                                                   \
    allocator::print_thread_heap_counters_only("begin", (thread_cnt)); \
    request_interval_diag::begin_stage((thread_cnt)); \
    profile::reset_all(); \
    rdma::reset_read_batch_diagnostics(); \
    hdr_reset(hist_latency_total); \
    wait_trace::start(); \
    struct rusage __object_size_rusage_start {}; \
    struct rusage __object_size_rusage_stop {}; \
    if (getrusage(RUSAGE_SELF, &__object_size_rusage_start) != 0) std::abort(); \
    auto __object_size_start = std::chrono::steady_clock::now(); \
    auto __object_size_perf = perf_profile([&] { \
        profile::begin_allocation_wait_diagnostics((thread_cnt)); \
        profile::start_work(); \
        profile::thread_start_work(); \
        size_t total_request_num_finished = run(thread_cnt); \
        std::cout << "total request num finished: " << total_request_num_finished << std::endl; \
        profile::thread_end_work(); \
        profile::end_work(); \
        profile::end_allocation_wait_diagnostics(); \
    }); \
    request_interval_diag::end_stage(); \
    auto __object_size_stop = std::chrono::steady_clock::now(); \
    allocator::print_thread_heap_counters_only("end", (thread_cnt)); \
    if (getrusage(RUSAGE_SELF, &__object_size_rusage_stop) != 0) std::abort(); \
    rdma::print_read_batch_diagnostics(); \
    wait_trace::stop_and_dump(); \
    __object_size_perf.print((thread_cnt), profile::collect_stw_mutator_cycles()); \
    double __object_size_elapsed_s = std::chrono::duration<double>( \
        __object_size_stop - __object_size_start).count(); \
    uint64_t __object_size_ops = total_request_num_finished.load(); \
    uint64_t __object_size_miss = profile::collect_data_miss_count(); \
    std::cout << "object_size_result" \
              << " object_size=" << object_size \
              << " thread_cnt=" << (thread_cnt) \
              << " read_batch_attribution=" << rdma::read_batch_attribution_mode() \
              << " cq_timing_enabled=" << profile::cq_timing_enabled() \
              << " elapsed_s=" << __object_size_elapsed_s \
              << " cpu_user_s=" << (rusage_seconds(__object_size_rusage_stop.ru_utime) - rusage_seconds(__object_size_rusage_start.ru_utime)) \
              << " cpu_system_s=" << (rusage_seconds(__object_size_rusage_stop.ru_stime) - rusage_seconds(__object_size_rusage_start.ru_stime)) \
              << " ops=" << __object_size_ops \
              << " ops_s=" << (__object_size_elapsed_s == 0.0 ? 0.0 : static_cast<double>(__object_size_ops) / __object_size_elapsed_s) \
              << " miss_count=" << __object_size_miss \
              << " miss_s=" << (__object_size_elapsed_s == 0.0 ? 0.0 : static_cast<double>(__object_size_miss) / __object_size_elapsed_s) \
              << " rdma_read_count=" << profile::collect_rdma_read_post_count() \
              << " rdma_read_bytes=" << profile::collect_rdma_read_post_bytes() \
              << std::endl; \
    profile::print_allocation_wait_diagnostics(); \
    profile::print_profile_data(); \
    hdr_print_percentiles(hist_latency_total); \
}

int main(int argc, const char *argv[]) {
    if (argc != 6) {
        std::cerr << "usage: " << argv[0] << " <config_path> <local_memory_ratio> <hdr_file> <zipf_skew> <op_duration_ns>" << std::endl;
        return -EINVAL;
    }

    std::string config_path = argv[1];
    local_memory_ratio = std::stod(argv[2]);
    std::string hdr_file = argv[3];
    zipf_skew = std::stod(argv[4]);
    op_duration_ns = std::stoull(argv[5]);

    FarLib::rdma::Configure config;
    config.from_file(config_path.c_str());
    runtime_init(config);
    scope_diag::inspect_entry=[](uintptr_t address)->uint64_t {
        auto *entry=reinterpret_cast<::FarLib::cache::FarObjectEntry *>(address);
        const auto state=entry->load_state(std::memory_order_relaxed);
        return static_cast<uint64_t>(state.state) |
               (static_cast<uint64_t>(state.invalid)<<32);
    };
    perf_init();
    hdr_init(1, 10'000'000, 3, &hist_latency_total);

    total_memory_size = (size_t)(16 * 1024 * 1024) * 1024;
    hot_vec_size = total_memory_size * local_memory_ratio / object_size;
    cold_vec_size = total_memory_size * (1 - local_memory_ratio) / span_size;
    total_object_num = cold_vec_size * (span_size / object_size) + hot_vec_size;
    request_object_num = hot_vec_size;
    // total_request_num = 128 * 1024 * 1024;
    // control the total request time to 50 seconds

    std::cout << "hot_vec_size: " << hot_vec_size << " cold_vec_size: "
              << cold_vec_size << " request_object_num: " << request_object_num
              << " total_object_num: " << total_object_num << std::endl;

    hot_object_ptr_list.resize(hot_vec_size);
    cold_object_span_list.resize(cold_vec_size);

    auto init_zipfian_generators = [&](size_t tid) {
        auto local_zipfian_generators = ZipfianGenerator<false>(request_object_num, zipf_skew);
        {
            std::lock_guard<std::mutex> lock(zipfian_generators_mutex);
            zipfian_generators.push_back(local_zipfian_generators);
        } 
    };
    uthread::fork_join(config.max_thread_cnt * 4, init_zipfian_generators);

    warm_up(config.max_thread_cnt);
    std::cout << "warm up done" << std::endl;

    auto profile_mem_usage = [&]() {
        allocator::print_combined_memory_usage();
    };

    std::thread profile_mem_usage_thread;
    if (profile_memory_enabled()) {
        profile_mem_usage_thread = std::thread([&]() {
            uint64_t start_time = get_time_ns();
            while (get_time_ns() - start_time <
                   benchmark_secs * 3 * 1000 * (size_t)(1000 * 1000)) {
                profile_mem_usage();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    size_t formal_fibres = config.max_thread_cnt * 2;
    if (const char *text = std::getenv("OBJECT_SIZE_FORMAL_FIBRES")) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(text, &end, 10);
        if (end == text || *end != '\0' ||
            (parsed != 24 && parsed != 48)) {
            std::cerr << "OBJECT_SIZE_FORMAL_FIBRES must be 24 or 48"
                      << std::endl;
            return -EINVAL;
        }
        formal_fibres = static_cast<size_t>(parsed);
    }
    std::cout << "object_size_formal_config formal_fibres="
              << formal_fibres
              << " preformal_fibres=" << config.max_thread_cnt
              << " postformal_fibres=" << config.max_thread_cnt * 3
              << std::endl;

    if (object_size_only_48_enabled()) {
        DO_WORK_WITH_PROFILE(formal_fibres);
    } else {
        DO_WORK_WITH_PROFILE(config.max_thread_cnt * 1);
        DO_WORK_WITH_PROFILE(formal_fibres);
        DO_WORK_WITH_PROFILE(config.max_thread_cnt * 3);
    }
   
    if (profile_mem_usage_thread.joinable()) {
        profile_mem_usage_thread.join();
    }
    hdr_print_percentiles(hist_latency_total);

    print_cdf(hdr_file);
    hdr_close(hist_latency_total);
    FarLib::cache::ConcurrentArrayCache::
        stop_default_read_supply_timeline_for_diagnostics();
    request_interval_diag::dump();
    hot_object_ptr_list.clear();
    cold_object_span_list.clear();
    runtime_destroy();
}
