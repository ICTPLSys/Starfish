#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <fstream>

struct AdjacencyList_inner {
    // A AdjacencyList contains a list of uint64_t value, each value is another vertex id.
    size_t size;
    uint64_t ngbrs[];
};

struct Page {
    uint64_t page_idx = -1; // size of the page
    char list_data[];
};

struct AdjacencyList_outer {
    // for objects smaller than OBJ_SPAN_UPPER_BOUND bytes, which page this adjacency list resides in
    // for objects larger than OBJ_SPAN_UPPER_BOUND bytes, page_id is the index of the page in the adjacency_pages vector
    uint64_t page_id; 
    uint64_t offset_in_page; // for objects smaller than OBJ_SPAN_UPPER_BOUND bytes, offset in page
    uint64_t size; // for objects smaller than OBJ_SPAN_UPPER_BOUND bytes, size of the adjacency list
};

static constexpr uint64_t size_classes[] = {
    8, 16, 32, 64, 80, 96, 112, 128, 144, 176, 208, 256, 304, 384, 448,
    512, 576, 704, 896, 1024, 1152, 1408, 1664, 2048, 2304, 2688, 3328, 4096
};

static constexpr uint64_t size_classes_full[] = {
    8, 16, 32, 64, 80, 96, 112, 128, 144, 176, 208, 256, 304, 384, 448,
    512, 576, 704, 896, 1024, 1152, 1408, 1664, 2048, 2304, 2688, 3328,
    4096, 4480, 5504, 6528, 8192, 9344, 11392, 13696, 16384, 18688, 21760,
    26112, 29056, 32768, 37376, 43648, 52352, 65536, 87296, 104832, 131072,
    174720
};

// apply size class only for objects smaller than OBJ_SPAN_UPPER_BOUND bytes
constexpr uint64_t size_classes_size_full = sizeof(size_classes_full) / sizeof(size_classes_full[0]);
constexpr uint64_t size_classes_size = sizeof(size_classes) / sizeof(size_classes[0]);
// each size class has a vector of pages

inline uint64_t get_size_class_id_full(uint64_t size) {
    int l = 0, r = size_classes_size_full - 1;
    while (l < r) {
        int mid = (l + r) >> 1;
        if (size <= size_classes_full[mid]) r = mid;
        else l = mid + 1;
    }
    return l;
    // for (size_t i = 0; i < size_classes_size_full; i++) {
    //     if (size <= size_classes_full[i]) return i;
    // }
    // return size_classes_size_full - 1;
}


inline uint64_t align_to_size_class(uint64_t size) {
    for (size_t i = 0; i < size_classes_size; i++) {
        if (size <= size_classes[i]) return size_classes[i];
    }
    return size_classes[size_classes_size - 1];
}

inline bool graph_progress_enabled() {
    const char *env = std::getenv("NHOP_GRAPH_PROGRESS");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::strtoull(env, nullptr, 0) != 0;
}

inline uint64_t graph_current_rss_bytes() {
    std::ifstream statm("/proc/self/statm");
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) {
        return 0;
    }
    return resident_pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
}

inline double graph_elapsed_s(
    const std::chrono::steady_clock::time_point &start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
}

inline void print_graph_progress(
    const char *phase, const std::chrono::steady_clock::time_point &start,
    const std::string &extra = "") {
    if (!graph_progress_enabled()) {
        return;
    }
    std::cout << "graph_load_progress"
              << " phase=" << phase
              << " elapsed_s=" << graph_elapsed_s(start)
              << " rss_bytes=" << graph_current_rss_bytes();
    if (!extra.empty()) {
        std::cout << " " << extra;
    }
    std::cout << std::endl;
}

bool readGraph_parallel(const std::string& baseFilename, std::vector<std::vector<uint64_t>>& adjList, uint64_t& numVertices) {
    const int numFiles = 32;
    auto progress_start = std::chrono::steady_clock::now();
    std::vector<std::set<uint64_t>> vertexSets(numFiles);
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> edgeLists(numFiles);
    std::atomic<int> read_files_done{0};
    std::atomic<int> fill_files_done{0};
    std::mutex progress_mutex;

    print_graph_progress("start", progress_start,
                         "files=32 base=" + baseFilename);

    auto readFile = [&](int idx) {
        std::ifstream fin(baseFilename + "." + std::to_string(idx));
        std::string line;
        while (getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            uint64_t u, v;
            std::istringstream ss(line);
            if (!(ss >> u >> v)) continue;
            vertexSets[idx].insert(u);
            vertexSets[idx].insert(v);
            edgeLists[idx].emplace_back(u, v);
        }
        int completed = read_files_done.fetch_add(1) + 1;
        if (graph_progress_enabled()) {
            std::lock_guard<std::mutex> guard(progress_mutex);
            print_graph_progress(
                "read_shard_done", progress_start,
                "shard=" + std::to_string(idx) +
                    " completed=" + std::to_string(completed) +
                    " shard_edges=" + std::to_string(edgeLists[idx].size()) +
                    " shard_vertices=" + std::to_string(vertexSets[idx].size()));
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numFiles; i++) threads.emplace_back(readFile, i);
    for (auto& t : threads) t.join();

    uint64_t total_input_edges = 0;
    for (int i = 0; i < numFiles; i++) {
        total_input_edges += edgeLists[i].size();
    }
    print_graph_progress("read_shards_joined", progress_start,
                         "input_edges=" + std::to_string(total_input_edges));

    std::set<uint64_t> allVertices;
    for (int i = 0; i < numFiles; i++) {
        allVertices.insert(vertexSets[i].begin(), vertexSets[i].end());
        if ((i + 1) % 4 == 0 || i + 1 == numFiles) {
            print_graph_progress(
                "merge_vertices", progress_start,
                "merged_shards=" + std::to_string(i + 1) +
                    " vertices_so_far=" + std::to_string(allVertices.size()));
        }
    }
    numVertices = allVertices.size();

    std::unordered_map<uint64_t, uint64_t> vertexMap;
    uint64_t idx = 0;
    for (auto vert : allVertices) vertexMap[vert] = idx++;
    print_graph_progress("vertex_map_done", progress_start,
                         "vertices=" + std::to_string(numVertices));

    adjList.clear();
    adjList.resize(numVertices + 1);
    print_graph_progress("adjacency_resize_done", progress_start,
                         "adjacency_lists=" + std::to_string(adjList.size()));

    // for (int i = 0; i < numFiles; i++) {
    //     std::cout << "Reading file " << i << " of " << numFiles << " started" << std::endl;
    //     for (auto& e : edgeLists[i]) {
    //         uint64_t u = vertexMap[e.first];
    //         uint64_t v = vertexMap[e.second];
    //         adjList[u].push_back(v);
    //         adjList[v].push_back(u);
    //     }
    // }


    std::vector<std::thread> threads_new;
    std::cout << "Creating locks" << std::endl;
    std::vector<std::atomic_flag> locks(adjList.size());
    for (auto& l : locks) l.clear();
    print_graph_progress("locks_ready", progress_start,
                         "locks=" + std::to_string(locks.size()));

    auto worker = [&](int i) {
        for (auto& e : edgeLists[i]) {
            uint64_t u = vertexMap.at(e.first);
            uint64_t v = vertexMap.at(e.second);
            while (locks[u].test_and_set(std::memory_order_acquire));
            adjList[u].push_back(v);
            locks[u].clear(std::memory_order_release);
            while (locks[v].test_and_set(std::memory_order_acquire));
            adjList[v].push_back(u);
            locks[v].clear(std::memory_order_release);
        }
        int completed = fill_files_done.fetch_add(1) + 1;
        if (graph_progress_enabled()) {
            std::lock_guard<std::mutex> guard(progress_mutex);
            print_graph_progress(
                "fill_shard_done", progress_start,
                "shard=" + std::to_string(i) +
                    " completed=" + std::to_string(completed) +
                    " shard_edges=" + std::to_string(edgeLists[i].size()));
        }
    };

    for (int i = 0; i < numFiles; i++)
        threads_new.emplace_back(worker, i);
    for (auto& t : threads_new)
        t.join();

    print_graph_progress("fill_adjacency_done", progress_start,
                         "undirected_edges=" +
                             std::to_string(total_input_edges * 2));
    std::cout << "Read graph done" << std::endl;
    return true;
}

bool readGraph_serial(const std::string& filename, std::vector<std::vector<uint64_t>>& adjList, uint64_t& numVertices) {      
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    // uint64_t numEdges = 1806067135;
    uint64_t numEdges = 34681189;

    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        // stringstream ss(line);
        // if (!(ss >> rows >> cols >> numEdges)) {
        //     cerr << "Failed to read matrix size line" << endl;
        //     return false;
        // }
        break;
    }

    // 第一遍扫描，收集所有顶点编号
    std::set<uint64_t> vertexSet;
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    edges.reserve(numEdges);

    uint64_t u, v;
    while (fin >> u >> v) {
        vertexSet.insert(u);
        vertexSet.insert(v);
        edges.emplace_back(u, v);
    }
    fin.close();

    numVertices = vertexSet.size();
    // assert(numVertices == 65608366);
    assert(numVertices == 3997962);
    adjList.clear();
    adjList.resize(numVertices + 1);

    std::unordered_map<uint64_t, uint64_t> vertexMap;
    uint64_t idx = 0;
    for (auto vert : vertexSet) {
        vertexMap[vert] = idx++;
    }

    // 用映射后的编号构建邻接表
    for (auto& e : edges) {
        adjList[vertexMap[e.first]].push_back(vertexMap[e.second]);
        adjList[vertexMap[e.second]].push_back(vertexMap[e.first]);
    }

    return true;
}
