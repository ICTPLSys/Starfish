#include <sched.h>

#include <atomic>
#include <boost/sort/parallel_stable_sort/parallel_stable_sort.hpp>
#include <boost/sort/sort.hpp>
#include <boost/timer/progress_display.hpp>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

#include "cache/cache.hpp"
#include "cache/alloc/region_remote_allocator.hpp"
#include "utils/parallel.hpp"

namespace FarLib {

template <bool TimeStamp = false,
          size_t ChunkSize /* in byte */ = 4096 - allocator::BlockHeadSize>
class ChunkedGraph {
public:
    using vertex_t = uint32_t;
    using count_t = uint32_t;
    using degree_t = count_t;

    using timestamp_t = uint32_t;

    struct edge_with_ts {
        vertex_t ngh;
        timestamp_t create_ts;
        timestamp_t destroy_ts;
    };

    struct edge_without_ts {
        vertex_t ngh;
    };

    using edge_t = std::conditional_t<TimeStamp, edge_with_ts, edge_without_ts>;

    // list for edges of a vertex
    struct EdgeList {
        vertex_t vertex_id;
        degree_t degree;
        edge_t neighbors[0];

        count_t size_in_bytes() const {
            return sizeof(EdgeList) + degree * sizeof(edge_t);
        }
    };

    struct EdgeListChunkHeader {
        count_t count;          // count of EdgeList
        count_t size_in_bytes;  // size in bytes

        EdgeListChunkHeader()
            : count(0), size_in_bytes(sizeof(EdgeListChunkHeader)) {}

        EdgeList *get_list(size_t offset) {
            return reinterpret_cast<EdgeList *>(
                (reinterpret_cast<char *>(this) + offset));
        }

        const EdgeList *get_list(size_t offset) const {
            return reinterpret_cast<const EdgeList *>(
                (reinterpret_cast<const char *>(this) + offset));
        }
    };

    // referencing an EdgeList in EdgeLists
    struct EdgeListRef {
        uint32_t chunk_idx;
        uint32_t offset;  // in byte
        degree_t degree;
    };

    using ChunkPtr = UniqueFarPtr<void>;

    // edge lists for all vertices
    struct EdgeLists {
        std::vector<ChunkPtr> chunks;
        size_t allocated_payload_bytes = 0;
        size_t allocated_footprint_bytes = 0;

        void clear() {
            chunks.clear();
            allocated_payload_bytes = 0;
            allocated_footprint_bytes = 0;
        }

        void release_without_deallocate() {
            for (auto &chunk : chunks) {
                chunk.get_entry().set_free();
            }
            chunks.clear();
            allocated_payload_bytes = 0;
            allocated_footprint_bytes = 0;
        }

        void swap(EdgeLists &other) {
            std::swap(chunks, other.chunks);
            std::swap(allocated_payload_bytes, other.allocated_payload_bytes);
            std::swap(allocated_footprint_bytes,
                      other.allocated_footprint_bytes);
        }

        size_t chunk_count() const { return chunks.size(); }

        void add_chunk(ChunkPtr &&chunk) {
            const size_t payload_bytes = chunk.size();
            const size_t block_bytes =
                payload_bytes + allocator::BlockHeadSize;
            const size_t bin = allocator::bin_from_wsize(
                allocator::wsize_from_size(block_bytes));
            allocated_payload_bytes += payload_bytes;
            allocated_footprint_bytes += allocator::get_bin_size(bin);
            chunks.push_back(std::move(chunk));
        }

        LiteAccessor<EdgeList> get(const EdgeListRef &ref,
                                   DereferenceScope &scope) const {
            const ChunkPtr &ptr = chunks[ref.chunk_idx];
            LiteAccessor<EdgeListChunkHeader> chunk(ptr, scope);
            return LiteAccessor<EdgeList>(chunk, chunk->get_list(ref.offset));
        }

        LiteAccessor<EdgeList> get(const EdgeListRef &ref, __DMH__,
                                   DereferenceScope &scope) const {
            const ChunkPtr &ptr = chunks[ref.chunk_idx];
            LiteAccessor<EdgeListChunkHeader> chunk(ptr, __on_miss__, scope);
            return LiteAccessor<EdgeList>(chunk, chunk->get_list(ref.offset));
        }

        void prefetch(const EdgeListRef &ref, DereferenceScope &scope) const {
            const ChunkPtr &ptr = chunks[ref.chunk_idx];
            Cache::get_default()->prefetch(ptr.obj(), scope);
        }

        // should not in dereference scope
        template <bool Optimize = true, size_t MaxThreadCount = 1,
                  size_t Granularity = 64, std::invocable<const EdgeList *> Fn>
        void for_each(Fn &&fn) const {
            const size_t prefetch_distance = []() {
                const char *env = std::getenv("GAPBS_CHUNK_PREFETCH_DISTANCE");
                if (env == nullptr || env[0] == '\0') {
                    return size_t{128};
                }
                return std::max<size_t>(1, std::strtoull(env, nullptr, 0));
            }();
            std::atomic_size_t next_chunk{0};
            auto scan_chunk = [&](size_t j, size_t &prefetch_idx,
                                  DereferenceScope &scope) {
                    ON_MISS_BEGIN
                        __define_oms__(scope);
                        if constexpr (Optimize) {
                            auto cache = FarLib::Cache::get_default();
                            size_t &i = prefetch_idx;
                            for (i = std::max(i, j + 1);
                                 i < chunks.size() &&
                                 i < j + prefetch_distance;
                                 i++) {
                                cache->prefetch(chunks[i].obj(), oms);
                                if (i % 8 == 0 &&
                                    cache::check_fetch(__entry__, __ddl__))
                                    break;
                            }
                            while (!cache::check_fetch(__entry__, __ddl__)) {
                                uthread::yield();
                            }
                        } else {
                            do {
                                uthread::yield();
                            } while (!cache::check_fetch(__entry__, __ddl__));
                        }
                    ON_MISS_END
                    LiteAccessor<EdgeListChunkHeader> chunk(chunks[j],
                                                            __on_miss__, scope);
                    size_t offset = sizeof(EdgeListChunkHeader);
                    for (size_t i = 0; i < chunk->count; i++) {
                        const EdgeList *list = chunk->get_list(offset);
                        fn(list);
                        offset += list->size_in_bytes();
                        assert(offset <= chunk->size_in_bytes);
                    }
                    if constexpr (Optimize) {
                        uthread::yield();
                    }
                };
            auto worker = [&](size_t) {
                size_t prefetch_idx = 0;
                while (true) {
                    size_t start =
                        next_chunk.fetch_add(Granularity,
                                             std::memory_order_relaxed);
                    if (start >= chunks.size()) {
                        break;
                    }
                    size_t stop = std::min(start + Granularity, chunks.size());
                    {
                        RootDereferenceScope scope;
                        for (size_t j = start; j < stop; ++j) {
                            scan_chunk(j, prefetch_idx, scope);
                        }
                    }
                    uthread::yield();
                }
            };
            size_t thread_count = std::min(
                MaxThreadCount,
                std::max<size_t>(1, (chunks.size() + Granularity - 1) /
                                        Granularity));
            if (thread_count == 1) {
                worker(0);
            } else {
                uthread::fork_join(thread_count, worker, "chunked_graph_scan");
            }
        }

        template <bool Optimize = true, size_t MaxThreadCount = 1,
                  size_t Granularity = 64,
                  std::invocable<const EdgeList *, DereferenceScope &> Fn>
        void for_each_with_scope(Fn &&fn) const {
            uthread::parallel_for_with_scope<Granularity, Optimize>(
                MaxThreadCount, chunks.size(),
                [&](size_t j, size_t &prefetch_idx, DereferenceScope &scope) {
                    ON_MISS_BEGIN
                        __define_oms__(scope);
                        if constexpr (Optimize) {
                            auto cache = FarLib::Cache::get_default();
                            size_t &i = prefetch_idx;
                            for (i = std::max(i, j + 1);
                                 i < chunks.size() && i < j + 1024; i++) {
                                cache->prefetch(chunks[i].obj(), oms);
                                if (i % 8 == 0 &&
                                    cache::check_fetch(__entry__, __ddl__))
                                    break;
                            }
                        } else {
                            do {
                                uthread::yield();
                            } while (!cache::check_fetch(__entry__, __ddl__));
                        }
                    ON_MISS_END
                    LiteAccessor<EdgeListChunkHeader> chunk(chunks[j],
                                                            __on_miss__, scope);
                    size_t offset = sizeof(EdgeListChunkHeader);
                    for (size_t i = 0; i < chunk->count; i++) {
                        const EdgeList *list = chunk->get_list(offset);
                        fn(list, scope);
                        offset += list->size_in_bytes();
                        assert(offset <= chunk->size_in_bytes);
                    }
                },
                [](size_t) -> size_t { return 0; });
        }
    };

    class EdgeListsBuilder {
    private:
        EdgeLists &output;
        ChunkPtr current_chunk;
        LiteAccessor<EdgeListChunkHeader, true> accessor;

    public:
        EdgeListsBuilder(EdgeLists &output) : output(output) {}

        void pin() const { accessor.pin(); }
        void unpin() const { accessor.unpin(); }

        void init(DereferenceScope &scope) {
            accessor = current_chunk.allocate_lite<true>(ChunkSize, scope)
                           .template as<EdgeListChunkHeader>();
            accessor->count = 0;
            accessor->size_in_bytes = sizeof(EdgeListChunkHeader);
        }

        template <std::invocable<EdgeList *> Fn>
        EdgeListRef add(vertex_t vertex_id, degree_t degree, Fn &&initializer,
                        DereferenceScope &scope) {
            EdgeList list_header{.vertex_id = vertex_id, .degree = degree};
            count_t list_size = list_header.size_in_bytes();
            if (accessor->size_in_bytes + list_size > current_chunk.size()) {
                // can not allocate in a chunk
                accessor = {};
                output.add_chunk(std::move(current_chunk));
                size_t chunk_size = std::max(
                    ChunkSize, list_size + sizeof(EdgeListChunkHeader));
                accessor = current_chunk.allocate_lite<true>(chunk_size, scope)
                               .template as<EdgeListChunkHeader>();
                accessor->count = 0;
                accessor->size_in_bytes = sizeof(EdgeListChunkHeader);
            }
            uint32_t offset = accessor->size_in_bytes;
            accessor->count++;
            accessor->size_in_bytes += list_size;
            EdgeList *edge_list = accessor->get_list(offset);
            *edge_list = list_header;
            initializer(edge_list);
            return EdgeListRef{
                .chunk_idx = static_cast<uint32_t>(output.chunk_count()),
                .offset = offset,
                .degree = degree,
            };
        }

        EdgeListRef add(vertex_t vertex_id, degree_t degree,
                        DereferenceScope &scope) {
            return add(vertex_id, degree, [](auto) {}, scope);
        }

        void close() {
            if (accessor->count != 0) {
                accessor = {};
                output.add_chunk(std::move(current_chunk));
            }
        }
    };

private:
    std::vector<EdgeListRef> out_edges;
    std::vector<EdgeListRef> in_edges;
    EdgeLists out_edge_lists;
    EdgeLists in_edge_lists;
    size_t edge_count_;

public:
    uint64_t sparse_edge_map_cycles;
    uint64_t dense_edge_map_cycles;
    uint64_t vertex_map_cycles;

public:
    size_t vertex_count() const { return out_edges.size(); }
    size_t edge_count() const { return edge_count_; }
    size_t far_object_count() const {
        return out_edge_lists.chunk_count() + in_edge_lists.chunk_count();
    }
    size_t far_object_payload_bytes() const {
        return out_edge_lists.allocated_payload_bytes +
               in_edge_lists.allocated_payload_bytes;
    }
    size_t far_object_footprint_bytes() const {
        return out_edge_lists.allocated_footprint_bytes +
               in_edge_lists.allocated_footprint_bytes;
    }

    void clear() {
        out_edges.clear();
        in_edges.clear();
        out_edge_lists.clear();
        in_edge_lists.clear();
        edge_count_ = 0;
        sparse_edge_map_cycles = 0;
        dense_edge_map_cycles = 0;
        vertex_map_cycles = 0;
    }

    void release_without_deallocate() {
        out_edges.clear();
        in_edges.clear();
        out_edge_lists.release_without_deallocate();
        in_edge_lists.release_without_deallocate();
        edge_count_ = 0;
        sparse_edge_map_cycles = 0;
        dense_edge_map_cycles = 0;
        vertex_map_cycles = 0;
    }

    void build_from_adjacency_lists(
        const std::vector<std::vector<uint64_t>> &adj_lists,
        bool build_in_edges = false) {
        clear();

        auto progress_enabled = []() {
            const char *env = std::getenv("NHOP_GRAPH_PROGRESS");
            if (env == nullptr || env[0] == '\0') {
                return true;
            }
            return std::strtoull(env, nullptr, 0) != 0;
        };
        auto rss_bytes = []() -> uint64_t {
            std::ifstream statm("/proc/self/statm");
            uint64_t total_pages = 0;
            uint64_t resident_pages = 0;
            if (!(statm >> total_pages >> resident_pages)) {
                return 0;
            }
            return resident_pages *
                   static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
        };
        auto start = std::chrono::steady_clock::now();
        auto elapsed_s = [&]() {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                .count();
        };
        auto print_progress = [&](const char *phase, size_t vertices_done) {
            if (!progress_enabled()) {
                return;
            }
            std::cout << "chunked_graph_build_progress"
                      << " phase=" << phase
                      << " elapsed_s=" << elapsed_s()
                      << " vertices_done=" << vertices_done
                      << " total_vertices=" << adj_lists.size()
                      << " chunks=" << out_edge_lists.chunk_count()
                      << " edges=" << edge_count_
                      << " remote_used_bytes="
                      << allocator::remote::remote_global_heap.get_used_bytes()
                      << " rss_bytes=" << rss_bytes() << std::endl;
        };
        const size_t progress_step =
            std::max<size_t>(1, adj_lists.size() / 100);
        print_progress("start", 0);

        out_edges.reserve(adj_lists.size());
        struct Scope : public RootDereferenceScope {
            EdgeListsBuilder builder;
            Scope(EdgeLists &output) : builder(output) {}
            void pin() const override { builder.pin(); }
            void unpin() const override { builder.unpin(); }
        } scope(out_edge_lists);
        scope.builder.init(scope);
        for (size_t v = 0; v < adj_lists.size(); ++v) {
            const auto &neighbors = adj_lists[v];
            ASSERT(v <= std::numeric_limits<vertex_t>::max());
            ASSERT(neighbors.size() <= std::numeric_limits<degree_t>::max());
            degree_t degree = static_cast<degree_t>(neighbors.size());
            edge_count_ += degree;
            EdgeListRef ref = scope.builder.add(
                static_cast<vertex_t>(v), degree,
                [&](EdgeList *edge_list) {
                    for (size_t i = 0; i < neighbors.size(); ++i) {
                        ASSERT(neighbors[i] <=
                               std::numeric_limits<vertex_t>::max());
                        edge_list->neighbors[i].ngh =
                            static_cast<vertex_t>(neighbors[i]);
                        if constexpr (TimeStamp) {
                            edge_list->neighbors[i].create_ts = 0;
                            edge_list->neighbors[i].destroy_ts =
                                std::numeric_limits<timestamp_t>::max();
                        }
                    }
                },
                scope);
            out_edges.push_back(ref);
            if ((v + 1) % progress_step == 0 || v + 1 == adj_lists.size()) {
                print_progress("build_out_edges", v + 1);
            }
        }
        scope.builder.close();
        print_progress("out_edges_done", adj_lists.size());
        if (build_in_edges) {
            generate_in_edges();
            print_progress("in_edges_done", adj_lists.size());
        }
    }

    template <typename DegreeFn, typename FillFn>
    void build_from_generated_adjacency(size_t vertex_count,
                                        DegreeFn &&degree_fn,
                                        FillFn &&fill_fn) {
        clear();

        ASSERT(vertex_count <=
               static_cast<size_t>(std::numeric_limits<vertex_t>::max()) + 1);
        out_edges.reserve(vertex_count);

        struct Scope : public RootDereferenceScope {
            EdgeListsBuilder builder;
            Scope(EdgeLists &output) : builder(output) {}
            void pin() const override { builder.pin(); }
            void unpin() const override { builder.unpin(); }
        } scope(out_edge_lists);
        scope.builder.init(scope);
        for (size_t v = 0; v < vertex_count; ++v) {
            ASSERT(v <= std::numeric_limits<vertex_t>::max());
            size_t degree_size = degree_fn(static_cast<vertex_t>(v));
            ASSERT(degree_size <= std::numeric_limits<degree_t>::max());
            degree_t degree = static_cast<degree_t>(degree_size);
            edge_count_ += degree;
            EdgeListRef ref = scope.builder.add(
                static_cast<vertex_t>(v), degree,
                [&](EdgeList *edge_list) {
                    fill_fn(static_cast<vertex_t>(v), edge_list);
                },
                scope);
            out_edges.push_back(ref);
        }
        scope.builder.close();
    }

    LiteAccessor<EdgeList> out_list(vertex_t vertex_id,
                                    DereferenceScope &scope) const {
        return out_edge_lists.get(out_edges[vertex_id], scope);
    }

    LiteAccessor<EdgeList> out_list(vertex_t vertex_id, __DMH__,
                                    DereferenceScope &scope) const {
        return out_edge_lists.get(out_edges[vertex_id], __on_miss__, scope);
    }

    void load(std::string file_name) {
        out_edges.clear();
        std::ifstream ifs(file_name);
        std::string header;
        size_t v_count;
        ASSERT(ifs >> header);
        ASSERT(header == "AdjacencyGraph");
        ASSERT(ifs >> v_count >> edge_count_);
        ASSERT(v_count <= std::numeric_limits<vertex_t>::max());

        std::vector<degree_t> degrees;
        degrees.reserve(v_count);

        size_t last_offset;
        ASSERT(ifs >> last_offset);
        ASSERT(last_offset == 0);
        for (size_t i = 1; i < v_count; i++) {
            size_t offset;
            ASSERT(ifs >> offset);
            degrees.push_back(offset - last_offset);
            last_offset = offset;
        }
        degrees.push_back(edge_count() - last_offset);
        ASSERT(degrees.size() == v_count);

        out_edges.reserve(v_count);
        {
            boost::timer::progress_display show_progress(v_count);
            struct Scope : public RootDereferenceScope {
                EdgeListsBuilder builder;
                Scope(EdgeLists &output) : builder(output) {}
                void pin() const override { builder.pin(); }
                void unpin() const override { builder.unpin(); }
            } scope(out_edge_lists);
            scope.builder.init(scope);
            for (size_t i = 0; i < v_count; i++) {
                degree_t degree = degrees[i];
                EdgeListRef ref = scope.builder.add(
                    i, degree,
                    [&](EdgeList *edge_list) {
                        for (size_t i = 0; i < degree; i++) {
                            vertex_t ngh;
                            ASSERT(ifs >> ngh);
                            edge_list->neighbors[i].ngh = ngh;
                            if constexpr (TimeStamp) {
                                edge_list->neighbors[i].create_ts = 0;
                                edge_list->neighbors[i].destroy_ts =
                                    std::numeric_limits<timestamp_t>::max();
                            }
                        }
                    },
                    scope);
                out_edges.push_back(ref);
                ++show_progress;
            }
            scope.builder.close();
        }
        generate_in_edges();
    }

    size_t out_degree(vertex_t vertex_id) {
        return out_edges[vertex_id].degree;
    }
    size_t in_degree(vertex_t vertex_id) { return in_edges[vertex_id].degree; }

    // apply fn on each out edge list of vertex in vertices
    // should not be in a dereference scope
    template <bool Optimize, size_t MaxThreadCount = 1,
              size_t Granularity = 1024, std::invocable<size_t, EdgeList *> Fn>
    void for_each_out_list(const std::vector<vertex_t> &vertices, Fn &&fn) {
        uint64_t start_tsc = get_cycles();
        uthread::parallel_for_with_scope<Granularity, Optimize>(
            MaxThreadCount, vertices.size(),
            [&](size_t i, size_t &prefetch_idx, DereferenceScope &scope) {
                vertex_t vertex = vertices[i];
                LiteAccessor<EdgeList> edge_list;
                ON_MISS_BEGIN
                    __define_oms__(scope);
                    if constexpr (Optimize) {
                        size_t &j = prefetch_idx;
                        for (j = std::max(j, i + 1);
                             j < vertices.size() && j < i + 1024; j++) {
                            out_edge_lists.prefetch(out_edges[vertices[j]],
                                                    oms);
                            if (j % 8 == 0 &&
                                FarLib::cache::check_fetch(__entry__, __ddl__))
                                break;
                        }
                    } else {
                        do {
                            uthread::yield();
                        } while (!cache::check_fetch(__entry__, __ddl__));
                    }
                ON_MISS_END
                edge_list =
                    out_edge_lists.get(out_edges[vertex], __on_miss__, scope);
                fn(i, edge_list.as_ptr());
            },
            [](size_t) -> size_t { return 0; });
        uint64_t end_tsc = get_cycles();
        sparse_edge_map_cycles += end_tsc - start_tsc;
    }

    template <bool Optimize, size_t MaxThreadCount = 1, size_t Granularity = 64,
              std::invocable<const EdgeList *> Fn>
    void for_each_in_list(Fn &&fn) {
        uint64_t start_tsc = get_cycles();
        in_edge_lists
            .template for_each<Optimize, MaxThreadCount, Granularity, Fn>(
                std::forward<Fn>(fn));
        uint64_t end_tsc = get_cycles();
        dense_edge_map_cycles += end_tsc - start_tsc;
    }

    template <bool Optimize, size_t MaxThreadCount = 1, size_t Granularity = 64,
              std::invocable<const EdgeList *> Fn>
    void for_each_out_list(Fn &&fn) {
        uint64_t start_tsc = get_cycles();
        out_edge_lists
            .template for_each<Optimize, MaxThreadCount, Granularity, Fn>(
                std::forward<Fn>(fn));
        uint64_t end_tsc = get_cycles();
        dense_edge_map_cycles += end_tsc - start_tsc;
    }

    void transpose() {
        in_edges.swap(out_edges);
        in_edge_lists.swap(out_edge_lists);
    }

private:
    void generate_in_edges() {
        in_edges.clear();

        // TODO: maybe we should use remote vector
        std::vector<std::pair<vertex_t, vertex_t>> in_pairs;
        out_edge_lists.for_each_with_scope(
            [&](const EdgeList *list, DereferenceScope &scope) {
                vertex_t src = list->vertex_id;
                for (size_t j = 0; j < list->degree; j++) {
                    vertex_t dst = list->neighbors[j].ngh;
                    in_pairs.push_back({dst, src});
                }
            });

        std::thread([&] {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            int num_cpus = sysconf(_SC_NPROCESSORS_CONF);
            for (int i = 0; i < num_cpus; ++i) {
                CPU_SET(i, &mask);
            }
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
            boost::sort::parallel_stable_sort(in_pairs.begin(), in_pairs.end());
        }).join();

        in_edges.reserve(vertex_count());
        auto in_pairs_it = in_pairs.begin();

        struct Scope : public RootDereferenceScope {
            EdgeListsBuilder builder;
            Scope(EdgeLists &output) : builder(output) {}
            void pin() const override { builder.pin(); }
            void unpin() const override { builder.unpin(); }
        } scope(in_edge_lists);
        scope.builder.init(scope);
        for (size_t i = 0; i < vertex_count(); i++) {
            vertex_t degree = 0;
            for (auto it = in_pairs_it; it != in_pairs.end() && it->first == i;
                 it++) {
                degree++;
            }
            auto ref = scope.builder.add(
                i, degree,
                [&](EdgeList *edge_list) {
                    for (degree_t j = 0; j < degree; j++) {
                        auto &edge = edge_list->neighbors[j];
                        edge.ngh = in_pairs_it->second;
                        if constexpr (TimeStamp) {
                            edge.create_ts = 0;
                            edge.destroy_ts =
                                std::numeric_limits<timestamp_t>::max();
                        }
                        ++in_pairs_it;
                    }
                },
                scope);
            in_edges.push_back(ref);
        }
        scope.builder.close();
    }
};

}  // namespace FarLib
