#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cache/cache.hpp"
#include "data_structure/chunked_graph.hpp"
#include "graph_utils.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/parallel.hpp"
#include "utils/perf.hpp"
#include "utils/stats.hpp"
#include "utils/timer.hpp"
#include "utils/uthreads.hpp"

using namespace FarLib;

using Graph = ChunkedGraph<false>;
using vertex_t = Graph::vertex_t;
using parent_t = int32_t;
using depth_t = int32_t;
using ParentArray = std::vector<std::atomic<parent_t>>;
using DepthArray = std::vector<std::atomic<depth_t>>;
constexpr size_t kMaxWorkerCount = 256;
constexpr depth_t kUnvisitedDepth = -1;

struct PhaseStats {
    int64_t miss_count;
    int64_t rdma_read_count;
    int64_t rdma_read_bytes;
    int64_t rdma_write_count;
    int64_t rdma_write_bytes;
    int64_t stw_mutator_cycles;
    int64_t gc_count;
    int64_t gc_cycles;
    int64_t evict_rounds;
    int64_t evict_active_rounds;
    int64_t evict_empty_rounds;
    int64_t evict_active_wrs;
    int64_t try_evict_count;
    int64_t try_evict_pinned;
    int64_t try_evict_free_or_mismatch;
    int64_t mark_regions;
    int64_t mark_regions_with_output;
    int64_t mark_regions_without_output;
    int64_t mark_regions_skipped_in_use;
    int64_t mark_regions_skipped_stable;
    int64_t evict_regions_skipped_in_use;
    int64_t evict_regions_to_free;
    int64_t evict_regions_to_usable;
    int64_t evict_regions_to_full;
    int64_t evict_pop_empty;
    int64_t evict_pop_head_matched;
    int64_t evict_pop_success;
    int64_t mark_count;
    int64_t not_mark_count;
};

struct CycleBreakdown {
    int64_t work;
    int64_t allocate;
    int64_t post_fetch;
    int64_t yield;
    int64_t on_miss;
    int64_t flip_scope;
    int64_t flip_scope_yield;
};

CycleBreakdown collect_cycle_breakdown() {
    return {
        profile::collect_work_cycles(),
        profile::collect_allocate_cycles(),
        profile::collect_post_fetch_cycles(),
        profile::collect_yield_cycles(),
        profile::collect_on_miss_cycles(),
        profile::collect_flip_scope_cycles(),
        profile::collect_flip_scope_yield_cycles(),
    };
}

CycleBreakdown operator-(const CycleBreakdown &after,
                         const CycleBreakdown &before) {
    return {
        after.work - before.work,
        after.allocate - before.allocate,
        after.post_fetch - before.post_fetch,
        after.yield - before.yield,
        after.on_miss - before.on_miss,
        after.flip_scope - before.flip_scope,
        after.flip_scope_yield - before.flip_scope_yield,
    };
}

void print_cycle_breakdown(const CycleBreakdown &delta,
                           const PerfResult &perf_result,
                           size_t fibre_count) {
    auto estimated_wall_ms = [&](int64_t aggregate_cycles) {
        if (aggregate_cycles <= 0 || fibre_count == 0 ||
            perf_result.total_cycles == 0) {
            return 0.0;
        }
        return static_cast<double>(aggregate_cycles) /
               static_cast<double>(fibre_count) * perf_result.runtime_ms /
               static_cast<double>(perf_result.total_cycles);
    };
    std::cout << "bfs_cycle_breakdown"
              << " fibre_count=" << fibre_count
              << " work_cycles=" << delta.work
              << " work_wall_ms=" << estimated_wall_ms(delta.work)
              << " alloc_cycles=" << delta.allocate
              << " alloc_wall_ms=" << estimated_wall_ms(delta.allocate)
              << " post_fetch_cycles=" << delta.post_fetch
              << " post_fetch_wall_ms="
              << estimated_wall_ms(delta.post_fetch)
              << " yield_cycles=" << delta.yield
              << " yield_wall_ms=" << estimated_wall_ms(delta.yield)
              << " on_miss_cycles=" << delta.on_miss
              << " on_miss_wall_ms=" << estimated_wall_ms(delta.on_miss)
              << " flip_scope_cycles=" << delta.flip_scope
              << " flip_scope_wall_ms="
              << estimated_wall_ms(delta.flip_scope)
              << " flip_scope_yield_cycles=" << delta.flip_scope_yield
              << " flip_scope_yield_wall_ms="
              << estimated_wall_ms(delta.flip_scope_yield)
              << std::endl;
}

struct BfsResult {
    size_t levels = 0;
    size_t visited = 0;
    uint64_t parent_checksum = 0;
    uint64_t mode_switches = 0;
    uint64_t top_down_steps = 0;
    uint64_t bottom_up_steps = 0;
    size_t max_depth = 0;
    bool closed = false;
};

struct BfsVerifyStats {
    size_t visited = 0;
    size_t unreachable = 0;
    size_t parent_errors = 0;
    size_t edge_depth_errors = 0;
    uint64_t depth_checksum = 0;
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
        profile::collect_evac_evict_rounds(),
        profile::collect_evac_evict_active_rounds(),
        profile::collect_evac_evict_empty_rounds(),
        profile::collect_evac_evict_active_wrs(),
        profile::collect_evac_try_evict_count(),
        profile::collect_evac_try_evict_pinned(),
        profile::collect_evac_try_evict_free_or_mismatch(),
        profile::collect_evac_mark_regions_scanned(),
        profile::collect_evac_mark_regions_with_output(),
        profile::collect_evac_mark_regions_without_output(),
        profile::collect_evac_mark_regions_skipped_in_use(),
        profile::collect_evac_mark_regions_skipped_stable(),
        profile::collect_evac_evict_regions_skipped_in_use(),
        profile::collect_evac_evict_regions_to_free(),
        profile::collect_evac_evict_regions_to_usable(),
        profile::collect_evac_evict_regions_to_full(),
        profile::collect_evac_evict_pop_empty(),
        profile::collect_evac_evict_pop_head_matched(),
        profile::collect_evac_evict_pop_success(),
        profile::collect_mark_count(),
        profile::collect_not_mark_count(),
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
        after.evict_rounds - before.evict_rounds,
        after.evict_active_rounds - before.evict_active_rounds,
        after.evict_empty_rounds - before.evict_empty_rounds,
        after.evict_active_wrs - before.evict_active_wrs,
        after.try_evict_count - before.try_evict_count,
        after.try_evict_pinned - before.try_evict_pinned,
        after.try_evict_free_or_mismatch -
            before.try_evict_free_or_mismatch,
        after.mark_regions - before.mark_regions,
        after.mark_regions_with_output - before.mark_regions_with_output,
        after.mark_regions_without_output - before.mark_regions_without_output,
        after.mark_regions_skipped_in_use - before.mark_regions_skipped_in_use,
        after.mark_regions_skipped_stable - before.mark_regions_skipped_stable,
        after.evict_regions_skipped_in_use -
            before.evict_regions_skipped_in_use,
        after.evict_regions_to_free - before.evict_regions_to_free,
        after.evict_regions_to_usable - before.evict_regions_to_usable,
        after.evict_regions_to_full - before.evict_regions_to_full,
        after.evict_pop_empty - before.evict_pop_empty,
        after.evict_pop_head_matched - before.evict_pop_head_matched,
        after.evict_pop_success - before.evict_pop_success,
        after.mark_count - before.mark_count,
        after.not_mark_count - before.not_mark_count,
    };
}

void print_phase_stats(const char *phase, double elapsed_s, size_t ops,
                       const PhaseStats &delta,
                       size_t iteration = std::numeric_limits<size_t>::max()) {
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
    std::cout << "gapbs_bfs_phase_stats"
              << " phase=" << phase;
    if (iteration != std::numeric_limits<size_t>::max()) {
        std::cout << " iteration=" << iteration;
    }
    std::cout << " elapsed_s=" << elapsed_s
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
              << " local_free_bytes="
              << allocator::global_heap.get_free_bytes()
              << " local_committed_bytes="
              << allocator::global_heap.get_committed_bytes()
              << " evict_rounds=" << delta.evict_rounds
              << " evict_active_rounds=" << delta.evict_active_rounds
              << " evict_empty_rounds=" << delta.evict_empty_rounds
              << " evict_active_wrs=" << delta.evict_active_wrs
              << " try_evict_count=" << delta.try_evict_count
              << " try_evict_pinned=" << delta.try_evict_pinned
              << " try_evict_free_or_mismatch="
              << delta.try_evict_free_or_mismatch
              << " mark_regions=" << delta.mark_regions
              << " mark_regions_with_output="
              << delta.mark_regions_with_output
              << " mark_regions_without_output="
              << delta.mark_regions_without_output
              << " mark_regions_skipped_in_use="
              << delta.mark_regions_skipped_in_use
              << " mark_regions_skipped_stable="
              << delta.mark_regions_skipped_stable
              << " evict_regions_skipped_in_use="
              << delta.evict_regions_skipped_in_use
              << " evict_regions_to_free="
              << delta.evict_regions_to_free
              << " evict_regions_to_usable="
              << delta.evict_regions_to_usable
              << " evict_regions_to_full="
              << delta.evict_regions_to_full
              << " evict_pop_empty=" << delta.evict_pop_empty
              << " evict_pop_head_matched="
              << delta.evict_pop_head_matched
              << " evict_pop_success=" << delta.evict_pop_success
              << " mark_count=" << delta.mark_count
              << " not_mark_count=" << delta.not_mark_count
              << std::endl;
}

size_t env_size(const char *name, size_t default_value) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return default_value;
    }
    char *end = nullptr;
    unsigned long long value = std::strtoull(env, &end, 0);
    if (end == env) {
        return default_value;
    }
    return static_cast<size_t>(value);
}

bool env_bool(const char *name, bool default_value) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return default_value;
    }
    return std::strtoull(env, nullptr, 0) != 0;
}

bool thread_heap_diag_enabled = false;
size_t thread_heap_diag_iteration = std::numeric_limits<size_t>::max();

std::string env_string(const char *name, const std::string &default_value = "") {
    const char *env = std::getenv(name);
    if (env == nullptr) {
        return default_value;
    }
    return env;
}

void read_graph(const std::string &filename, Graph &graph,
                uint64_t &num_vertices) {
    std::vector<std::vector<uint64_t>> adj_list;
    uint64_t start_tsc = __rdtsc();
    ASSERT(readGraph_parallel(filename, adj_list, num_vertices));
    uint64_t end_tsc = __rdtsc();
    std::cout << "readGraph_parallel time: " << end_tsc - start_tsc
              << " cycles" << std::endl;

    graph.build_from_adjacency_lists(adj_list, false);
    adj_list.clear();
    std::cout << "gapbs_bfs_far_objects"
              << " count=" << graph.far_object_count()
              << " payload_bytes=" << graph.far_object_payload_bytes()
              << " footprint_bytes=" << graph.far_object_footprint_bytes()
              << std::endl;
}

void build_grid3d_graph(Graph &graph, uint64_t &num_vertices) {
    const size_t x_dim = env_size("GAPBS_GRID_X", 32);
    const size_t y_dim = env_size("GAPBS_GRID_Y", x_dim);
    const size_t z_dim = env_size("GAPBS_GRID_Z", x_dim);
    ASSERT(x_dim > 0);
    ASSERT(y_dim > 0);
    ASSERT(z_dim > 0);
    ASSERT(x_dim <= std::numeric_limits<vertex_t>::max());
    ASSERT(y_dim <= std::numeric_limits<vertex_t>::max());
    ASSERT(z_dim <= std::numeric_limits<vertex_t>::max());
    ASSERT(x_dim <= std::numeric_limits<size_t>::max() / y_dim);
    const size_t xy_dim = x_dim * y_dim;
    ASSERT(z_dim <= std::numeric_limits<size_t>::max() / xy_dim);
    const size_t vertex_count = xy_dim * z_dim;
    ASSERT(vertex_count <=
           static_cast<size_t>(std::numeric_limits<vertex_t>::max()) + 1);

    auto to_xyz = [=](vertex_t v, size_t &x, size_t &y, size_t &z) {
        size_t value = v;
        x = value % x_dim;
        value /= x_dim;
        y = value % y_dim;
        z = value / y_dim;
    };
    auto to_vertex = [=](size_t x, size_t y, size_t z) -> vertex_t {
        return static_cast<vertex_t>((z * y_dim + y) * x_dim + x);
    };
    auto degree_fn = [=](vertex_t v) -> size_t {
        size_t x = 0;
        size_t y = 0;
        size_t z = 0;
        to_xyz(v, x, y, z);
        return (x > 0) + (x + 1 < x_dim) + (y > 0) + (y + 1 < y_dim) +
               (z > 0) + (z + 1 < z_dim);
    };
    auto fill_fn = [=](vertex_t v, Graph::EdgeList *edge_list) {
        size_t x = 0;
        size_t y = 0;
        size_t z = 0;
        to_xyz(v, x, y, z);
        Graph::degree_t idx = 0;
        auto add = [&](size_t nx, size_t ny, size_t nz) {
            ASSERT(idx < edge_list->degree);
            edge_list->neighbors[idx++].ngh = to_vertex(nx, ny, nz);
        };
        if (x > 0) add(x - 1, y, z);
        if (x + 1 < x_dim) add(x + 1, y, z);
        if (y > 0) add(x, y - 1, z);
        if (y + 1 < y_dim) add(x, y + 1, z);
        if (z > 0) add(x, y, z - 1);
        if (z + 1 < z_dim) add(x, y, z + 1);
        ASSERT(idx == edge_list->degree);
    };

    std::cout << "gapbs_graph_generator"
              << " type=grid3d"
              << " x=" << x_dim
              << " y=" << y_dim
              << " z=" << z_dim
              << " vertices=" << vertex_count
              << " expected_max_depth_from_corner="
              << ((x_dim - 1) + (y_dim - 1) + (z_dim - 1)) << std::endl;
    graph.build_from_generated_adjacency(vertex_count, degree_fn, fill_fn);
    num_vertices = vertex_count;
}

void load_or_generate_graph(const std::string &filename, Graph &graph,
                            uint64_t &num_vertices) {
    std::string generator = env_string("GAPBS_GRAPH_GENERATOR");
    if (generator.empty() || generator == "file") {
        read_graph(filename, graph, num_vertices);
        return;
    }
    if (generator == "grid3d") {
        build_grid3d_graph(graph, num_vertices);
        return;
    }
    std::cerr << "Unsupported GAPBS_GRAPH_GENERATOR=" << generator
              << std::endl;
    std::exit(EINVAL);
}

class BfsBitmap {
private:
    std::vector<std::atomic<uint64_t>> words;

public:
    explicit BfsBitmap(size_t bits) : words((bits + 63) / 64) { reset(); }

    void reset() {
        for (auto &word : words) {
            word.store(0, std::memory_order_relaxed);
        }
    }

    void set(size_t idx) {
        const uint64_t mask = 1ull << (idx & 63);
        words[idx >> 6].fetch_or(mask, std::memory_order_relaxed);
    }

    void copy_from(const BfsBitmap &other) {
        ASSERT(words.size() == other.words.size());
        for (size_t i = 0; i < words.size(); ++i) {
            words[i].store(other.words[i].load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
        }
    }

    bool get(size_t idx) const {
        const uint64_t mask = 1ull << (idx & 63);
        return (words[idx >> 6].load(std::memory_order_relaxed) & mask) != 0;
    }
};

class SlidingQueue {
private:
    std::vector<vertex_t> values;
    size_t shared_out_start = 0;
    size_t shared_out_end = 0;

public:
    explicit SlidingQueue(size_t capacity) { values.reserve(capacity); }

    void reset(vertex_t source) {
        values.clear();
        values.push_back(source);
        shared_out_start = 0;
        shared_out_end = 1;
    }

    size_t begin() const { return shared_out_start; }
    size_t end() const { return shared_out_end; }
    size_t size() const { return shared_out_end - shared_out_start; }
    bool empty() const { return size() == 0; }
    vertex_t operator[](size_t idx) const { return values[idx]; }

    void append_buffers(const std::vector<std::vector<vertex_t>> &buffers) {
        shared_out_start = shared_out_end;
        size_t total = 0;
        for (const auto &buffer : buffers) {
            total += buffer.size();
        }
        values.reserve(values.size() + total);
        for (const auto &buffer : buffers) {
            values.insert(values.end(), buffer.begin(), buffer.end());
        }
        shared_out_end = values.size();
    }
};

template <bool Optimize>
BfsVerifyStats verify_bfs_tree(
    Graph &graph, vertex_t source,
    const std::vector<std::atomic<parent_t>> &parent,
    const std::vector<std::atomic<depth_t>> &depth, size_t worker_count,
    bool require_closed_search) {
    const size_t vertex_count = graph.vertex_count();
    ASSERT(parent.size() == vertex_count);
    ASSERT(depth.size() == vertex_count);

    BfsVerifyStats stats;
    std::vector<BfsVerifyStats> local(worker_count);
    uthread::fork_join(worker_count, [&](size_t tid) {
        const size_t stride = (vertex_count + worker_count - 1) / worker_count;
        const size_t start = stride * tid;
        const size_t stop = std::min(vertex_count, start + stride);
        BfsVerifyStats current;
        for (size_t v = start; v < stop; ++v) {
            parent_t p = parent[v].load(std::memory_order_relaxed);
            depth_t d = depth[v].load(std::memory_order_relaxed);
            if (d == kUnvisitedDepth) {
                current.unreachable++;
                if (p != -1) {
                    current.parent_errors++;
                }
                continue;
            }
            current.visited++;
            current.depth_checksum +=
                (static_cast<uint64_t>(v) + 1) *
                (static_cast<uint64_t>(d) + 1);
            if (v == source) {
                if (p != static_cast<parent_t>(source) || d != 0) {
                    current.parent_errors++;
                }
                continue;
            }
            if (p < 0 || static_cast<size_t>(p) >= vertex_count) {
                current.parent_errors++;
                continue;
            }
            depth_t parent_depth =
                depth[static_cast<size_t>(p)].load(std::memory_order_relaxed);
            if (parent_depth == kUnvisitedDepth || parent_depth + 1 != d) {
                current.parent_errors++;
            }
        }
        local[tid] = current;
    });

    for (const auto &item : local) {
        stats.visited += item.visited;
        stats.unreachable += item.unreachable;
        stats.parent_errors += item.parent_errors;
        stats.depth_checksum += item.depth_checksum;
    }

    std::atomic_size_t parent_edge_errors{0};
    std::atomic_size_t edge_depth_errors{0};
    auto verify_edges = [&](const Graph::EdgeList *edge_list) {
        vertex_t src = edge_list->vertex_id;
        ASSERT(src < vertex_count);
        parent_t src_parent = parent[src].load(std::memory_order_relaxed);
        depth_t src_depth = depth[src].load(std::memory_order_relaxed);
        bool needs_parent_edge =
            src_depth != kUnvisitedDepth && src != source && src_parent >= 0 &&
            static_cast<size_t>(src_parent) < vertex_count;
        bool found_parent_edge = false;

        for (Graph::degree_t i = 0; i < edge_list->degree; ++i) {
            vertex_t dst = edge_list->neighbors[i].ngh;
            ASSERT(dst < vertex_count);
            depth_t dst_depth = depth[dst].load(std::memory_order_relaxed);
            if (needs_parent_edge && dst == static_cast<vertex_t>(src_parent)) {
                found_parent_edge = true;
            }

            const bool src_visited = src_depth != kUnvisitedDepth;
            const bool dst_visited = dst_depth != kUnvisitedDepth;
            if (src_visited != dst_visited) {
                if (require_closed_search) {
                    edge_depth_errors.fetch_add(1, std::memory_order_relaxed);
                }
                continue;
            }
            if (!src_visited) {
                continue;
            }
            depth_t diff = src_depth > dst_depth ? src_depth - dst_depth
                                                 : dst_depth - src_depth;
            if (diff > 1) {
                edge_depth_errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (needs_parent_edge && !found_parent_edge) {
            parent_edge_errors.fetch_add(1, std::memory_order_relaxed);
        }
    };

    if (worker_count <= 24) {
        graph.template for_each_out_list<Optimize, 24>(verify_edges);
    } else if (worker_count <= 48) {
        graph.template for_each_out_list<Optimize, 48>(verify_edges);
    } else if (worker_count <= 64) {
        graph.template for_each_out_list<Optimize, 64>(verify_edges);
    } else if (worker_count <= 128) {
        graph.template for_each_out_list<Optimize, 128>(verify_edges);
    } else {
        graph.template for_each_out_list<Optimize, 256>(verify_edges);
    }

    stats.parent_errors +=
        parent_edge_errors.load(std::memory_order_relaxed);
    stats.edge_depth_errors =
        edge_depth_errors.load(std::memory_order_relaxed);
    return stats;
}

template <bool Optimize, bool Verify = false>
BfsResult gapbs_bfs_chunked(Graph &graph, vertex_t source, size_t max_levels,
                            size_t alpha, size_t beta,
                            size_t worker_count,
                            ParentArray *parent_output = nullptr,
                            DepthArray *depth_output = nullptr) {
    const size_t vertex_count = graph.vertex_count();
    ASSERT(vertex_count <=
           static_cast<size_t>(std::numeric_limits<parent_t>::max()));
    ParentArray parent(vertex_count);
    DepthArray depth(Verify ? vertex_count : 0);
    uthread::parallel_for<1024>(worker_count, vertex_count, [&](size_t i) {
        parent[i].store(-1, std::memory_order_relaxed);
        if constexpr (Verify) {
            depth[i].store(kUnvisitedDepth, std::memory_order_relaxed);
        }
    });
    parent[source].store(source, std::memory_order_relaxed);
    if constexpr (Verify) {
        depth[source].store(0, std::memory_order_relaxed);
    }

    SlidingQueue queue(vertex_count);
    queue.reset(source);
    BfsBitmap frontier(vertex_count);
    BfsBitmap next(vertex_count);
    frontier.set(source);

    size_t scout_count = graph.out_degree(source);
    size_t edges_to_check = graph.edge_count();
    size_t awake_count = 1;
    bool use_bottom_up = false;
    BfsResult result;

    while (awake_count > 0 && result.levels < max_levels) {
        if (!use_bottom_up && scout_count > edges_to_check / alpha) {
            use_bottom_up = true;
            result.mode_switches++;
        }

        if (!use_bottom_up) {
            const size_t previous_scout_count = scout_count;
            const size_t begin = queue.begin();
            const size_t end = queue.end();
            const size_t frontier_size = end - begin;
            const size_t n_threads = std::max<size_t>(1, worker_count);
            std::vector<std::vector<vertex_t>> buffers(n_threads);
            std::vector<size_t> local_scout(n_threads, 0);

            uthread::fork_join(n_threads, [&](size_t tid) {
                {
                    RootDereferenceScope scope;
                    auto &buffer = buffers[tid];
                    size_t scout = 0;
                    const size_t stride =
                        (frontier_size + n_threads - 1) / n_threads;
                    const size_t start = begin + stride * tid;
                    const size_t stop = std::min(end, start + stride);
                    buffer.reserve(stop > start ? stop - start : 0);
                    for (size_t idx = start; idx < stop; ++idx) {
                        vertex_t src = queue[idx];
                        ON_MISS_BEGIN
                        ON_MISS_END
                        auto edge_list =
                            graph.out_list(src, __on_miss__, scope);
                        for (Graph::degree_t i = 0;
                             i < edge_list->degree; ++i) {
                            vertex_t dst = edge_list->neighbors[i].ngh;
                            parent_t expected = -1;
                            if (parent[dst].compare_exchange_strong(
                                    expected, static_cast<parent_t>(src),
                                    std::memory_order_relaxed)) {
                                if constexpr (Verify) {
                                    depth[dst].store(
                                        static_cast<depth_t>(result.levels + 1),
                                        std::memory_order_relaxed);
                                }
                                buffer.push_back(dst);
                                scout += graph.out_degree(dst);
                            }
                        }
                    }
                    local_scout[tid] = scout;
                }
            });

            queue.append_buffers(buffers);
            frontier.reset();
            for (size_t idx = queue.begin(); idx < queue.end(); ++idx) {
                frontier.set(queue[idx]);
            }
            scout_count =
                std::accumulate(local_scout.begin(), local_scout.end(), size_t{0});
            awake_count = queue.size();
            edges_to_check = edges_to_check > previous_scout_count
                                 ? edges_to_check - previous_scout_count
                                 : 0;
            result.top_down_steps++;
        } else {
            next.reset();
            std::vector<size_t> awake(worker_count, 0);
            std::vector<size_t> scout(worker_count, 0);
            auto scan_unvisited = [&](const Graph::EdgeList *edge_list) {
                    vertex_t src = edge_list->vertex_id;
                    if (parent[src].load(std::memory_order_relaxed) != -1) {
                        return;
                    }
                    for (Graph::degree_t i = 0; i < edge_list->degree; ++i) {
                        vertex_t dst = edge_list->neighbors[i].ngh;
                        if (frontier.get(dst)) {
                            parent[src].store(static_cast<parent_t>(dst),
                                              std::memory_order_relaxed);
                            if constexpr (Verify) {
                                depth[src].store(
                                    static_cast<depth_t>(result.levels + 1),
                                    std::memory_order_relaxed);
                            }
                            next.set(src);
                            break;
                        }
                    }
            };
            if (worker_count <= 24) {
                graph.template for_each_out_list<Optimize, 24>(scan_unvisited);
            } else if (worker_count <= 48) {
                graph.template for_each_out_list<Optimize, 48>(scan_unvisited);
            } else if (worker_count <= 64) {
                graph.template for_each_out_list<Optimize, 64>(scan_unvisited);
            } else if (worker_count <= 128) {
                graph.template for_each_out_list<Optimize, 128>(scan_unvisited);
            } else {
                graph.template for_each_out_list<Optimize, 256>(scan_unvisited);
            }

            uthread::fork_join(worker_count, [&](size_t tid) {
                const size_t stride = (vertex_count + worker_count - 1) / worker_count;
                const size_t start = stride * tid;
                const size_t stop = std::min(vertex_count, start + stride);
                size_t local_awake = 0;
                size_t local_scout = 0;
                for (size_t v = start; v < stop; ++v) {
                    if (next.get(v)) {
                        local_awake++;
                        local_scout += graph.out_degree(static_cast<vertex_t>(v));
                    }
                }
                awake[tid] = local_awake;
                scout[tid] = local_scout;
            });

            awake_count =
                std::accumulate(awake.begin(), awake.end(), size_t{0});
            scout_count =
                std::accumulate(scout.begin(), scout.end(), size_t{0});
            frontier.copy_from(next);
            next.reset();
            if (awake_count < vertex_count / beta) {
                use_bottom_up = false;
                queue.reset(source);
                std::vector<std::vector<vertex_t>> buffers(worker_count);
                uthread::fork_join(worker_count, [&](size_t tid) {
                    const size_t stride =
                        (vertex_count + worker_count - 1) / worker_count;
                    const size_t start = stride * tid;
                    const size_t stop = std::min(vertex_count, start + stride);
                    for (size_t v = start; v < stop; ++v) {
                        if (frontier.get(v)) {
                            buffers[tid].push_back(static_cast<vertex_t>(v));
                        }
                    }
                });
                queue.append_buffers(buffers);
                scout_count = 1;
                result.mode_switches++;
            }
            result.bottom_up_steps++;
        }

        result.levels++;
        if (thread_heap_diag_enabled) {
            allocator::print_thread_heap_diagnostics(
                "bfs_level", thread_heap_diag_iteration, result.levels);
        }
    }

    std::vector<size_t> visited(worker_count, 0);
    std::vector<uint64_t> checksums(worker_count, 0);
    std::vector<size_t> max_depths(worker_count, 0);
    uthread::fork_join(worker_count, [&](size_t tid) {
        const size_t stride = (vertex_count + worker_count - 1) / worker_count;
        const size_t start = stride * tid;
        const size_t stop = std::min(vertex_count, start + stride);
        size_t local_visited = 0;
        uint64_t local_checksum = 0;
        size_t local_max_depth = 0;
        for (size_t v = start; v < stop; ++v) {
            parent_t p = parent[v].load(std::memory_order_relaxed);
            if (p >= 0) {
                local_visited++;
                local_checksum +=
                    (static_cast<uint64_t>(v) + 1) *
                    (static_cast<uint64_t>(p) + 1);
                if constexpr (Verify) {
                    depth_t d = depth[v].load(std::memory_order_relaxed);
                    ASSERT(d >= 0);
                    local_max_depth =
                        std::max(local_max_depth, static_cast<size_t>(d));
                }
            }
        }
        visited[tid] = local_visited;
        checksums[tid] = local_checksum;
        max_depths[tid] = local_max_depth;
    });
    result.visited = std::accumulate(visited.begin(), visited.end(), size_t{0});
    result.parent_checksum =
        std::accumulate(checksums.begin(), checksums.end(), uint64_t{0});
    if constexpr (Verify) {
        result.max_depth =
            *std::max_element(max_depths.begin(), max_depths.end());
    } else {
        result.max_depth =
            result.levels == 0 ? 0 : (awake_count == 0 ? result.levels - 1
                                                       : result.levels);
    }
    result.closed = awake_count == 0;
    if constexpr (Verify) {
        if (parent_output != nullptr && depth_output != nullptr) {
            *parent_output = std::move(parent);
            *depth_output = std::move(depth);
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config_path> <graph_path>"
                  << std::endl;
        return -EINVAL;
    }

    std::string config_path = argv[1];
    std::string graph_path = argv[2];

    FarLib::rdma::Configure config;
    config.from_file(config_path.c_str());
    runtime_init(config);
    perf_init();
    profile::reset_all();
    thread_heap_diag_enabled =
        env_bool("GAPBS_THREAD_HEAP_DIAG", false);

    Graph graph;
    uint64_t num_vertices = 0;
    auto phase_before = collect_phase_stats();
    auto phase_start = std::chrono::steady_clock::now();
    load_or_generate_graph(graph_path, graph, num_vertices);
    auto phase_stop = std::chrono::steady_clock::now();
    print_phase_stats(
        "load_build",
        std::chrono::duration<double>(phase_stop - phase_start).count(),
        num_vertices, collect_phase_stats() - phase_before);
    if (thread_heap_diag_enabled) {
        allocator::print_thread_heap_diagnostics(
            "load_build", std::numeric_limits<size_t>::max(),
            std::numeric_limits<size_t>::max());
    }

    const size_t requested_worker_count = env_size(
        "GAPBS_BFS_WORKERS",
        config.max_thread_cnt * env_size("GAPBS_BFS_UTHREAD_FACTOR", 2));
    const size_t worker_count =
        std::max<size_t>(1, std::min(kMaxWorkerCount, requested_worker_count));
    const size_t source =
        env_size("GAPBS_BFS_SOURCE", std::min<size_t>(1, graph.vertex_count() - 1));
    const size_t max_levels =
        env_size("GAPBS_BFS_MAX_LEVELS", std::numeric_limits<size_t>::max());
    const size_t alpha = env_size("GAPBS_BFS_ALPHA", 15);
    const size_t beta = env_size("GAPBS_BFS_BETA", 18);
    const bool optimize = env_bool("GAPBS_BFS_OPTIMIZE", true);
    const bool verify = env_bool("GAPBS_BFS_VERIFY", false);
    const size_t repetitions = env_size("GAPBS_BFS_REPETITIONS", 1);
    const bool phase_replan = env_bool("GAPBS_BFS_PHASE_REPLAN", false);
    const size_t freeze_after_iteration = env_size(
        "GAPBS_BFS_FREEZE_AFTER_ITERATION",
        std::numeric_limits<size_t>::max());

    ASSERT(source < graph.vertex_count());
    ASSERT(alpha > 0);
    ASSERT(beta > 0);
    ASSERT(repetitions > 0);

    std::cout << "gapbs_bfs_config"
              << " vertices=" << graph.vertex_count()
              << " edges=" << graph.edge_count()
              << " source=" << source
              << " worker_count=" << worker_count
              << " alpha=" << alpha
              << " beta=" << beta
              << " max_levels=" << max_levels
              << " optimize=" << optimize
              << " verify=" << verify
              << " repetitions=" << repetitions
              << " phase_replan=" << phase_replan
              << " freeze_after_iteration=" << freeze_after_iteration
              << std::endl;

    ParentArray verification_parent;
    DepthArray verification_depth;
    BfsResult verification_result;
    bool placement_frozen = false;
    for (size_t iteration = 0; iteration < repetitions; ++iteration) {
        thread_heap_diag_iteration = iteration;
        std::cout << "gapbs_bfs_iteration_begin"
                  << " iteration=" << iteration
                  << " source=" << source << std::endl;
        phase_before = collect_phase_stats();
        const CycleBreakdown cycle_before = collect_cycle_breakdown();
        auto work_start = std::chrono::steady_clock::now();
        BfsResult result;
        PerfResult perf_result = perf_profile([&] {
            profile::start_work();
            profile::thread_start_work();
            if (optimize) {
                if (verify) {
                    ParentArray *parent_output =
                        iteration + 1 == repetitions ? &verification_parent
                                                     : nullptr;
                    DepthArray *depth_output =
                        iteration + 1 == repetitions ? &verification_depth
                                                     : nullptr;
                    result = gapbs_bfs_chunked<true, true>(
                        graph, static_cast<vertex_t>(source), max_levels,
                        alpha, beta, worker_count, parent_output, depth_output);
                } else {
                    result = gapbs_bfs_chunked<true, false>(
                        graph, static_cast<vertex_t>(source), max_levels,
                        alpha, beta, worker_count);
                }
            } else {
                if (verify) {
                    ParentArray *parent_output =
                        iteration + 1 == repetitions ? &verification_parent
                                                     : nullptr;
                    DepthArray *depth_output =
                        iteration + 1 == repetitions ? &verification_depth
                                                     : nullptr;
                    result = gapbs_bfs_chunked<false, true>(
                        graph, static_cast<vertex_t>(source), max_levels,
                        alpha, beta, worker_count, parent_output, depth_output);
                } else {
                    result = gapbs_bfs_chunked<false, false>(
                        graph, static_cast<vertex_t>(source), max_levels,
                        alpha, beta, worker_count);
                }
            }
            profile::thread_end_work();
            profile::end_work();
        });
        const int64_t measured_stw_mutator_cycles =
            profile::collect_stw_mutator_cycles() -
            phase_before.stw_mutator_cycles;
        perf_result.print(worker_count, measured_stw_mutator_cycles);
        print_cycle_breakdown(
            collect_cycle_breakdown() - cycle_before, perf_result,
            worker_count);
        if (phase_replan && !placement_frozen &&
            iteration + 1 < repetitions) {
            if (iteration == freeze_after_iteration) {
                const bool frozen =
                    Cache::get_default()->freeze_resident_group_plan_now();
                ASSERT(frozen);
                placement_frozen = true;
            } else {
                Cache::get_default()->publish_resident_group_plan_now(true);
            }
        }
        auto work_stop = std::chrono::steady_clock::now();
        const double elapsed_s =
            std::chrono::duration<double>(work_stop - work_start).count();
        print_phase_stats("work", elapsed_s, graph.edge_count(),
                          collect_phase_stats() - phase_before, iteration);
        allocator::remote::remote_global_heap.print_used_memory();
        std::cout << "gapbs_bfs_result"
                  << " iteration=" << iteration
                  << " vertices=" << graph.vertex_count()
                  << " edges=" << graph.edge_count()
                  << " source=" << source
                  << " elapsed_s=" << elapsed_s
                  << " levels=" << result.levels
                  << " visited=" << result.visited
                  << " max_depth=" << result.max_depth
                  << " parent_checksum=" << result.parent_checksum
                  << " mode_switches=" << result.mode_switches
                  << " top_down_steps=" << result.top_down_steps
                  << " bottom_up_steps=" << result.bottom_up_steps
                  << std::endl;
        if (thread_heap_diag_enabled) {
            allocator::print_thread_heap_diagnostics(
                "bfs_iteration_end", iteration, result.levels);
        }
        if (verify && iteration + 1 == repetitions) {
            verification_result = result;
        }
    }

    if (verify) {
        phase_before = collect_phase_stats();
        auto verify_start = std::chrono::steady_clock::now();
        BfsVerifyStats verify_stats;
        if (optimize) {
            verify_stats = verify_bfs_tree<true>(
                graph, static_cast<vertex_t>(source), verification_parent,
                verification_depth, worker_count, verification_result.closed);
        } else {
            verify_stats = verify_bfs_tree<false>(
                graph, static_cast<vertex_t>(source), verification_parent,
                verification_depth, worker_count, verification_result.closed);
        }
        const bool ok =
            verify_stats.visited == verification_result.visited &&
            verify_stats.parent_errors == 0 &&
            verify_stats.edge_depth_errors == 0;
        auto verify_stop = std::chrono::steady_clock::now();
        const double verify_elapsed_s =
            std::chrono::duration<double>(verify_stop - verify_start).count();
        std::cout << "gapbs_bfs_verify"
                  << " status=" << (ok ? "pass" : "fail")
                  << " scope=after_measured_repetitions"
                  << " closed=" << verification_result.closed
                  << " visited=" << verify_stats.visited
                  << " unreachable=" << verify_stats.unreachable
                  << " max_depth=" << verification_result.max_depth
                  << " depth_checksum=" << verify_stats.depth_checksum
                  << " parent_errors=" << verify_stats.parent_errors
                  << " edge_depth_errors=" << verify_stats.edge_depth_errors
                  << std::endl;
        print_phase_stats("verification", verify_elapsed_s,
                          graph.edge_count(),
                          collect_phase_stats() - phase_before);
        ASSERT(ok);
    }

    runtime_quiesce_cache();
    graph.release_without_deallocate();
    runtime_destroy();
    return 0;
}
