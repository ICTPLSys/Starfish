// This microbenchmark is to test the performance of n-hop graph traversal.
// COmpare Page-I/O and Object-I/O.

#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#include <set>
#include <unordered_map>
#include <fstream>

#include "latency_mode.hpp"
#include "cache/cache.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/parallel.hpp"
#include "utils/perf.hpp"
#include "utils/stats.hpp"
#include "utils/timer.hpp"
#include "utils/uthreads.hpp"

#include "graph_utils.hpp"
using namespace FarLib;

const int HOP_NUM = 3;
// const int RANDOM_NUM = 20;
const int DEFAULT_RANDOM_NUM = 50;
const int PIN_THRESHOLD = cache::ConcurrentArrayCache::STAT_THRESHOLD;
// const int PIN_THRESHOLD = 208;
// data structure:
// adjacency list
constexpr size_t RealSize[] = {
    1 * 8,    2 * 8,    3 * 8,    4 * 8,    5 * 8,    6 * 8,    7 * 8,    8 * 8,    10 * 8,   12 * 8,   14 * 8,   16 * 8,
    20 * 8,   24 * 8,   28 * 8,   32 * 8,   40 * 8,   48 * 8,   56 * 8,   64 * 8,   80 * 8,   96 * 8,   112 * 8,  128 * 8,
    160 * 8,  192 * 8,  224 * 8,  256 * 8,  320 * 8,  384 * 8,  448 * 8,  512 * 8,  640 * 8,  768 * 8,  896 * 8,  1024 * 8,
    1280 * 8, 1536 * 8, 1792 * 8, 2048 * 8, 2560 * 8, 3072 * 8, 3584 * 8, 4096 * 8, 5120 * 8, 6144 * 8, 7168 * 8, 8192 * 8,
    10240 * 8, 12288 * 8, 14336 * 8, 16384 * 8, 20480 * 8, 24576 * 8, 28672 * 8, 32768 * 8
};

inline size_t align_to_real_size(size_t size) {
    for (size_t i = 0; i < sizeof(RealSize) / sizeof(RealSize[0]); i++) {
        if (size <= RealSize[i]) return RealSize[i];
    }
    return RealSize[sizeof(RealSize) / sizeof(RealSize[0]) - 1];
}


uint64_t read_tsc() {
    return __rdtsc();
}

size_t get_random_num() {
    const char *env = std::getenv("NHOP_RANDOM_NUM");
    if (env == nullptr || env[0] == '\0') {
        return DEFAULT_RANDOM_NUM;
    }
    char *end = nullptr;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || value == 0) {
        return DEFAULT_RANDOM_NUM;
    }
    return static_cast<size_t>(value);
}

bool get_print_size_distribution() {
    const char *env = std::getenv("NHOP_PRINT_SIZE_DISTRIBUTION");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::strtoull(env, nullptr, 0) != 0;
}

bool get_work_progress_enabled() {
    const char *env = std::getenv("NHOP_WORK_PROGRESS");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return std::strtoull(env, nullptr, 0) != 0;
}

size_t get_uthread_factor() {
    const char *env = std::getenv("NHOP_UTHREAD_FACTOR");
    if (env == nullptr || env[0] == '\0') {
        return 2;
    }
    char *end = nullptr;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || value == 0) {
        return 2;
    }
    return static_cast<size_t>(value);
}

struct PhaseStats {
    int64_t miss_count;
    int64_t rdma_read_count;
    int64_t rdma_read_bytes;
    int64_t rdma_write_count;
    int64_t rdma_write_bytes;
    int64_t stw_mutator_cycles;
    int64_t gc_count;
    int64_t gc_cycles;
};

PhaseStats collect_phase_stats() {
    return {
        profile::collect_data_miss_count(),
        profile::collect_rdma_read_post_count(),
        profile::collect_rdma_read_post_bytes(),
        profile::collect_rdma_write_post_count(),
        profile::collect_rdma_write_post_bytes(),
        profile::collect_stw_mutator_cycles(),
        profile::collect_gc_count(),
        profile::collect_gc_cycles(),
    };
}

PhaseStats operator-(const PhaseStats &after, const PhaseStats &before) {
    return {
        after.miss_count - before.miss_count,
        after.rdma_read_count - before.rdma_read_count,
        after.rdma_read_bytes - before.rdma_read_bytes,
        after.rdma_write_count - before.rdma_write_count,
        after.rdma_write_bytes - before.rdma_write_bytes,
        after.stw_mutator_cycles - before.stw_mutator_cycles,
        after.gc_count - before.gc_count,
        after.gc_cycles - before.gc_cycles,
    };
}

void print_phase_stats(const char *phase, double elapsed_s, size_t ops,
                       const PhaseStats &delta) {
    const double miss_s =
        elapsed_s == 0.0 ? 0.0 : static_cast<double>(delta.miss_count) / elapsed_s;
    const double rdma_read_s =
        elapsed_s == 0.0 ? 0.0 : static_cast<double>(delta.rdma_read_count) / elapsed_s;
    const double rdma_write_s =
        elapsed_s == 0.0 ? 0.0 : static_cast<double>(delta.rdma_write_count) / elapsed_s;
    const double rdma_read_gib_s =
        elapsed_s == 0.0
            ? 0.0
            : static_cast<double>(delta.rdma_read_bytes) /
                  static_cast<double>(1ull << 30) / elapsed_s;
    const double rdma_write_gib_s =
        elapsed_s == 0.0
            ? 0.0
            : static_cast<double>(delta.rdma_write_bytes) /
                  static_cast<double>(1ull << 30) / elapsed_s;
    const double ops_s =
        elapsed_s == 0.0 ? 0.0 : static_cast<double>(ops) / elapsed_s;
    std::cout << "nhop_phase_stats"
              << " phase=" << phase
              << " elapsed_s=" << elapsed_s
              << " ops=" << ops
              << " ops_s=" << ops_s
              << " remote_used_bytes="
              << allocator::remote::remote_global_heap.get_used_bytes()
              << " miss_count=" << delta.miss_count
              << " miss_s=" << miss_s
              << " rdma_read_count=" << delta.rdma_read_count
              << " rdma_read_s=" << rdma_read_s
              << " rdma_read_bytes=" << delta.rdma_read_bytes
              << " rdma_read_gib_s=" << rdma_read_gib_s
              << " rdma_write_count=" << delta.rdma_write_count
              << " rdma_write_s=" << rdma_write_s
              << " rdma_write_bytes=" << delta.rdma_write_bytes
              << " rdma_write_gib_s=" << rdma_write_gib_s
              << " stw_mutator_cycles=" << delta.stw_mutator_cycles
              << " gc_count=" << delta.gc_count
              << " gc_cycles=" << delta.gc_cycles
              << std::endl;
}

uint64_t get_time_us() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000;
}

struct AdjacencyList {
    // A AdjacencyList contains a list of uint64_t value, each value is another vertex id.
    size_t size;
    uint64_t ngbrs[];
};

// #define DEBUGGING

std::vector<UniqueFarPtr<AdjacencyList>> adjacency_lists;

bool readGraph(const std::string& filename,
               std::vector<std::vector<uint64_t>>& adjList,
               uint64_t& numVertices,
               const GraphLoadAffinity &load_affinity,
               std::vector<nq_latency::OracleQuery>* latency_oracle = nullptr) {

    uint64_t start_time = read_tsc();
    auto progress_start = std::chrono::steady_clock::now();
    if (!readGraph_parallel(filename, adjList, numVertices, load_affinity)) {
        return false;
    }
    if (latency_oracle) {
        if (numVertices != 65608366 || adjList.size() != 65608367)
            throw std::invalid_argument("NQ latency requires the full Friendster graph");
        *latency_oracle = nq_latency::make_oracle(adjList, numVertices);
    }
    uint64_t end_time = read_tsc();
    std::cout << "readGraph_parallel time: " << end_time - start_time << " cycles" << std::endl;

    adjacency_lists.clear();
    adjacency_lists.resize(numVertices + 1);

    RootDereferenceScope scope;
    size_t total_size = 0;
    size_t real_size = 0;
    auto print_build_progress = [&](const char *phase, size_t vertices_done) {
        if (!graph_progress_enabled()) {
            return;
        }
        std::cout << "nhop_graph_build_progress"
                  << " phase=" << phase
                  << " elapsed_s=" << graph_elapsed_s(progress_start)
                  << " vertices_done=" << vertices_done
                  << " total_vertices=" << adjList.size()
                  << " total_size=" << total_size
                  << " real_size=" << real_size
                  << " remote_used_bytes="
                  << allocator::remote::remote_global_heap.get_used_bytes()
                  << " rss_bytes=" << graph_current_rss_bytes() << std::endl;
    };
    const size_t progress_step = std::max<size_t>(1, adjList.size() / 100);
    print_build_progress("start", 0);
    for (size_t i = 0; i < adjList.size(); i++) {
        #ifdef DEBUGGING
        if (i == 120) {
            for (size_t j = 0; j < adjList[i].size(); j++) {
                std::cout << "adjList[" << i << "][" << j << "]: " << adjList[i][j] << ", ";
            }
            std::cout << std::endl;
        }
        #endif

        uint64_t tmp_size = (adjList[i].size() + 1) * sizeof(uint64_t);
        real_size += align_to_real_size(tmp_size);
        total_size += tmp_size;
        LiteAccessor<AdjacencyList, true> acc; 
        if (tmp_size <= PIN_THRESHOLD) {
            acc = adjacency_lists[i].allocate_lite_with_size(tmp_size, scope);
        } else {
            acc = adjacency_lists[i].allocate_lite_with_size(tmp_size, scope);
        }
        auto accessor = acc.as_mut();
        accessor->size = (adjList[i].size());
        for (size_t j = 0; j < adjList[i].size(); j++) {
            accessor->ngbrs[j] = (uint64_t)adjList[i][j];
            // if (accessor->ngbrs[j] > 4000000) {
            //     std::cerr << "ngbrs[j] > 4000000, ngbrs[j] = " << accessor->ngbrs[j] << ", size = " << accessor->size << std::endl;
            //     ASSERT(false);
            // }
        }
        if ((i + 1) % progress_step == 0 || i + 1 == adjList.size()) {
            print_build_progress("build_adjacency_lists", i + 1);
        }
    }

    print_build_progress("done", adjList.size());
    std::cout << "real_size: " << real_size << std::endl;
    adjList.clear();
    return true;
}

uint64_t benchmark_lite(uint64_t start, RootDereferenceScope &scope) {
    // only get 2 hop neighbors of start
    std::vector<uint64_t> direct_neighbors;

    ON_MISS_BEGIN
        uthread::yield();
    ON_MISS_END
    auto accessor = adjacency_lists[start].access(__on_miss__, scope);
    direct_neighbors.reserve(accessor->size);
    for (size_t i = 0; i < accessor->size; i++) {
        // if (accessor->ngbrs[i] == start) continue;
        direct_neighbors.push_back(accessor->ngbrs[i]);
    }
    uint64_t second_hop_count = 0;
    for (size_t i = 0; i < direct_neighbors.size(); i++) {
        auto accessor_ngbr = adjacency_lists[direct_neighbors[i]].access(__on_miss__, scope);
        second_hop_count += accessor_ngbr->size;
    }

    #ifdef DEBUGGING

    std::sort(direct_neighbors.begin(), direct_neighbors.end());
    for (size_t i = 0; i < direct_neighbors.size(); i++) {
        std::cout << "direct_neighbors[" << i << "]: " << direct_neighbors[i];
        if (i < direct_neighbors.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl << std::endl;
    // std::cout << "direct_neighbors.size(): " << direct_neighbors.size() << std::endl;
    // for (size_t i = 0; i < ngbrs_of_ngbrs.size(); i++) {
    //     std::cout << "ngbrs_of_ngbrs[" << i << "]: " << ngbrs_of_ngbrs[i];
    //     if (i < ngbrs_of_ngbrs.size() - 1) std::cout << ", ";
    // }
    // std::cout << std::endl << std::endl;
    #endif
    return direct_neighbors.size() + second_hop_count;
}

void stat_graph_size_distribution() {
    RootDereferenceScope scope;
    // std::vector<size_t> size_distribution(100);
    std::map<size_t, size_t> size_distribution;
    for (size_t i = 0; i < adjacency_lists.size(); i++) {
        auto accessor = adjacency_lists[i].access(scope);
        size_distribution[align_to_real_size((accessor->size + 1) * sizeof(uint64_t))]++;
    }

    size_t accumulated_size = 0;
    for (auto& [size, count] : size_distribution) {
        accumulated_size += size * count;
        std::cout << "size: " << size << " bytes, count: " << count << ", accumulated_size: " << accumulated_size << std::endl;
    }
    std::cout << "accumulated_size: " << accumulated_size << std::endl;
}

std::atomic<uint64_t> benchmark_lite_results = 0;

bool resident_state_check_enabled() {
    const char *value = std::getenv("NHOP_RESIDENT_STATE_CHECK");
    return value != nullptr && std::string(value) == "1";
}

void print_resident_state(const char *phase, bool quiesced = false) {
    if (!resident_state_check_enabled()) return;
#ifdef FARLIB_NQ_NONFT_COMPAT
    throw std::runtime_error("NHOP_RESIDENT_STATE_CHECK requires the Starfish runtime");
#else
    const auto s = Cache::get_default()->simple_resident_state_snapshot();
    const bool budgets_ok = s.backup_used_bytes <= s.backup_budget_bytes &&
                           s.backup_peak_bytes <= s.backup_budget_bytes &&
                           s.resident_regions <= s.resident_budget_regions;
    const bool mapping_ok = !s.local_resident_mapping ||
        (s.local_hot_regions == s.resident_regions &&
         s.local_cold_regions == s.streaming_regions);
    std::cout << "nhop_resident_state phase=" << phase
              << " quiesced=" << quiesced
              << " mapping=" << s.local_resident_mapping
              << " resident_regions=" << s.resident_regions
              << " resident_budget_regions=" << s.resident_budget_regions
              << " resident_bytes=" << s.resident_bytes
              << " resident_budget_bytes=" << s.resident_budget_bytes
              << " streaming_regions=" << s.streaming_regions
              << " local_hot_regions=" << s.local_hot_regions
              << " local_cold_regions=" << s.local_cold_regions
              << " backup_used_bytes=" << s.backup_used_bytes
              << " backup_peak_bytes=" << s.backup_peak_bytes
              << " backup_budget_bytes=" << s.backup_budget_bytes
              << " budgets_ok=" << budgets_ok
              << " mapping_ok=" << mapping_ok << std::endl;
    if (quiesced && (!budgets_ok || !mapping_ok))
        throw std::runtime_error("NQ Resident/group state invariant failed");
#endif
}

int main(int argc, char *argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--describe-latency-mode") {
        std::fputs("nq_latency_capability schema=1 arrival=poisson_per_fibre "
                   "include_queue_wait=1 hist_sample_period=1 "
                   "queue_deadline=drop_before_execution\n", stdout);
        return 0;
    }
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config_path> <graph_path>" << std::endl;
        return -EINVAL;
    }
    std::optional<nq_latency::Options> latency_options;
    try {
        latency_options = nq_latency::options_from_env();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "\nnq_latency_error reason=configuration detail=%s\n",
                     error.what());
        return 2;
    }
#ifdef FARLIB_NQ_NONFT_COMPAT
    if (resident_state_check_enabled()) {
        std::cerr << "NHOP_RESIDENT_STATE_CHECK requires the Starfish runtime" << std::endl;
        return 2;
    }
#endif

    const GraphLoadAffinity graph_load_affinity;
    std::string config_path = argv[1];
    std::string graph_path = argv[2];

    FarLib::rdma::Configure config;
    config.from_file(config_path.c_str());
    runtime_init(config);
    perf_init();

    profile::reset_all();
    std::vector<std::vector<uint64_t>> adjList;
    std::vector<nq_latency::OracleQuery> latency_oracle;
    uint64_t numVertices;
    auto phase_before = collect_phase_stats();
    auto phase_start = std::chrono::steady_clock::now();
    if (latency_options) {
        bool loaded = false;
        try {
            loaded = readGraph(graph_path, adjList, numVertices, graph_load_affinity,
                               &latency_oracle);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "\nnq_latency_error reason=graph_load detail=%s\n",
                         error.what());
        }
        if (!loaded) {
            Cache::quiesce_default();
            adjacency_lists.clear();
            runtime_destroy();
            return 2;
        }
    } else {
        readGraph(graph_path, adjList, numVertices, graph_load_affinity);
    }
    auto phase_stop = std::chrono::steady_clock::now();
    print_phase_stats(
        "load_build",
        std::chrono::duration<double>(phase_stop - phase_start).count(),
        numVertices, collect_phase_stats() - phase_before);

    std::cout << "Load file done" << std::endl;
    std::cout << "numVertices: " << numVertices << std::endl;
    std::cout << "adjacency_lists.size(): " << adjacency_lists.size() << std::endl;

    // check_data_integrity();
    if (get_print_size_distribution()) {
        phase_before = collect_phase_stats();
        phase_start = std::chrono::steady_clock::now();
        stat_graph_size_distribution();
        phase_stop = std::chrono::steady_clock::now();
        print_phase_stats(
            "size_distribution",
            std::chrono::duration<double>(phase_stop - phase_start).count(),
            adjacency_lists.size(), collect_phase_stats() - phase_before);
    } else {
        std::cout << "size distribution skipped" << std::endl;
    }
    std::cout << "Check data integrity done" << std::endl;
    print_resident_state("work_begin");

    if (latency_options) {
        const auto query = [](uint64_t vertex) {
            RootDereferenceScope scope;
            return benchmark_lite(vertex, scope);
        };
        const auto parallel = [](size_t count, auto& worker) {
            uthread::fork_join(count, worker);
        };
        bool measured = false, verified = false;
        profile::reset_all();
        perf_profile([&] {
            profile::start_work();
            profile::thread_start_work();
            try {
                if (config.max_thread_cnt * get_uthread_factor() != nq_latency::Fibres)
                    throw std::invalid_argument("NQ latency requires 48 query fibres");
                measured = nq_latency::run(*latency_options, numVertices,
                    adjacency_lists.size(), uthread::get_worker_count(), query,
                    parallel, [] { uthread::yield(); });
            } catch (const std::exception& error) {
                std::fprintf(stderr, "\nnq_latency_error reason=measurement detail=%s\n",
                             error.what());
            }
            profile::thread_end_work();
            profile::end_work();
        }).print();
        try {
            verified = nq_latency::post_verify(latency_oracle, query);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "\nnq_latency_error reason=post_verify detail=%s\n",
                         error.what());
        }
        profile::print_profile_data();
        allocator::remote::remote_global_heap.print_used_memory();
        Cache::quiesce_default();
        print_resident_state("work_end", true);
        adjacency_lists.clear();
        print_resident_state("after_cleanup", true);
        runtime_destroy();
        return measured && verified ? 0 : 2;
    }

    #ifdef DEBUGGING
    RootDereferenceScope scope;
    benchmark_lite(120, scope);
    #endif

    profile::reset_all();
    const size_t worker_count = config.max_thread_cnt * get_uthread_factor();
    const size_t random_num = get_random_num();
    const size_t queries_per_worker = numVertices / random_num;
    const size_t query_count = worker_count * queries_per_worker;
    std::cout << "nhop_work_config"
              << " worker_count=" << worker_count
              << " max_thread_cnt=" << config.max_thread_cnt
              << " queries_per_worker=" << queries_per_worker
              << " query_count=" << query_count << std::endl;
    phase_before = collect_phase_stats();
    auto work_start = std::chrono::steady_clock::now();
    std::atomic<uint64_t> completed_queries{0};
    std::atomic_bool work_done{false};
    std::thread progress_thread;
    if (get_work_progress_enabled()) {
        progress_thread = std::thread([&] {
            while (!work_done.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                auto now = std::chrono::steady_clock::now();
                double elapsed_s =
                    std::chrono::duration<double>(now - work_start).count();
                print_phase_stats("work_progress", elapsed_s,
                                  completed_queries.load(std::memory_order_relaxed),
                                  collect_phase_stats() - phase_before);
            }
        });
    }
    perf_profile([&] {
        profile::start_work();
        profile::thread_start_work();

        auto benchmark_fn = [&](size_t tid) {
            RootDereferenceScope scope;
            std::mt19937_64 gen(0x9e3779b97f4a7c15ULL + tid);
            std::uniform_int_distribution<uint64_t> dist(0, numVertices - 1);
            uint64_t local_result = 0;
            for (size_t i = 0; i < queries_per_worker; i++) {
                local_result += benchmark_lite(dist(gen), scope);
                if (((i + 1) & 0xff) == 0) {
                    completed_queries.fetch_add(0x100,
                                                std::memory_order_relaxed);
                }
            }
            completed_queries.fetch_add(queries_per_worker & 0xff,
                                        std::memory_order_relaxed);
            
            benchmark_lite_results.fetch_add(local_result, std::memory_order_relaxed);
        };
        uthread::fork_join(worker_count, benchmark_fn);
        profile::thread_end_work();
        profile::end_work();
    }).print();
    auto work_stop = std::chrono::steady_clock::now();
    work_done.store(true, std::memory_order_release);
    if (progress_thread.joinable()) {
        progress_thread.join();
    }
    double elapsed_s =
        std::chrono::duration<double>(work_stop - work_start).count();
    print_phase_stats("work", elapsed_s, query_count,
                      collect_phase_stats() - phase_before);
    if (resident_state_check_enabled()) {
        // Validation is outside the Work timer and after its counter receipt.
        Cache::quiesce_default();
        print_resident_state("work_end", true);
    }
    profile::print_profile_data();
    allocator::remote::remote_global_heap.print_used_memory();

    double queries_s =
        elapsed_s == 0.0 ? 0.0 : static_cast<double>(query_count) / elapsed_s;
    std::cout << "benchmark_lite_results: " << benchmark_lite_results.load()
              << std::endl;
    std::cout << "nhop_graph_result"
              << " vertices=" << numVertices
              << " adjacency_lists=" << adjacency_lists.size()
              << " worker_count=" << worker_count
              << " random_num=" << random_num
              << " queries=" << query_count
              << " elapsed_s=" << elapsed_s
              << " queries_s=" << queries_s
              << " result=" << benchmark_lite_results.load() << std::endl;
    adjacency_lists.clear();
    print_resident_state("after_cleanup", true);
    runtime_destroy();

    return 0;
}
