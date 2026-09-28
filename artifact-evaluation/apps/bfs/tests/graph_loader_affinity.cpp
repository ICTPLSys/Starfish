// Offline no-RDMA regression for the GraphLoadAffinity API.
// The fixture exercises loader-thread affinity and graph-content preservation
// without starting the runtime or requiring Friendster/RDMA data.

#include "graph_load_affinity.hpp"
#include "graph_utils.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using CpuSet = std::set<int>;

CpuSet current_affinity() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    assert(sched_getaffinity(0, sizeof(mask), &mask) == 0);
    CpuSet result;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) result.insert(cpu);
    }
    return result;
}

void set_affinity(const CpuSet &cpus) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu : cpus) CPU_SET(cpu, &mask);
    assert(!cpus.empty());
    assert(sched_setaffinity(0, sizeof(mask), &mask) == 0);
}

struct Fixture {
    fs::path base;
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    std::vector<std::vector<uint64_t>> expected;
    uint64_t vertices = 0;
};

Fixture make_fixture() {
    const size_t shards = 32;
    const size_t edges_per_shard = [] {
        const char *value = std::getenv("BFS_LOADER_AUDIT_EDGES_PER_SHARD");
        return value == nullptr ? size_t{4096}
                                : std::max<size_t>(128, std::strtoull(value, nullptr, 0));
    }();
    const uint64_t stride = static_cast<uint64_t>(edges_per_shard + 2);
    std::array<char, 128> pattern{};
    std::snprintf(pattern.data(), pattern.size(), "%s/starfish-loader-affinity-XXXXXX",
                  fs::temp_directory_path().string().c_str());
    char *directory = ::mkdtemp(pattern.data());
    assert(directory != nullptr);
    const fs::path root(directory);
    const fs::path base = root / "graph";

    Fixture fixture{base};
    for (size_t shard = 0; shard < shards; ++shard) {
        std::ofstream output(base.string() + "." + std::to_string(shard));
        assert(output.good());
        const uint64_t first = static_cast<uint64_t>(shard) * stride;
        for (size_t i = 0; i < edges_per_shard; ++i) {
            const uint64_t u = first + i;
            const uint64_t v = u + 1;
            output << u << ' ' << v << '\n';
            fixture.edges.emplace_back(u, v);
        }
        // Add one deterministic cross-shard edge so the global vertex map and
        // merge path are checked, not merely 32 independent components.
        if (shard + 1 < shards) {
            const uint64_t u = first + edges_per_shard;
            const uint64_t v = static_cast<uint64_t>(shard + 1) * stride;
            output << u << ' ' << v << '\n';
            fixture.edges.emplace_back(u, v);
        }
    }

    std::map<uint64_t, std::vector<uint64_t>> by_id;
    for (const auto &[u, v] : fixture.edges) {
        by_id[u].push_back(v);
        by_id[v].push_back(u);
    }
    fixture.vertices = by_id.size();
    fixture.expected.resize(fixture.vertices + 1);
    std::map<uint64_t, size_t> compact_id;
    size_t compact = 0;
    for (const auto &[id, neighbors] : by_id) {
        (void)neighbors;
        compact_id.emplace(id, compact++);
    }
    for (const auto &[id, neighbors] : by_id) {
        auto &row = fixture.expected.at(compact_id.at(id));
        for (uint64_t neighbor : neighbors)
            row.push_back(compact_id.at(neighbor));
        std::sort(row.begin(), row.end());
    }
    return fixture;
}

void assert_adjacency(const Fixture &fixture,
                      const std::vector<std::vector<uint64_t>> &actual,
                      uint64_t vertices) {
    assert(vertices == fixture.vertices);
    assert(actual.size() == fixture.expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        auto row = actual[i];
        std::sort(row.begin(), row.end());
        assert(row == fixture.expected[i]);
    }
}

void assert_affinity_diagnostics(const std::string &output,
                                 size_t expected_cpus) {
    size_t seen = 0;
    for (const std::string &line : [&] {
             std::vector<std::string> lines;
             std::istringstream stream(output);
             for (std::string item; std::getline(stream, item);)
                 lines.push_back(std::move(item));
             return lines;
         }()) {
        std::map<std::string, std::string> fields;
        std::istringstream stream(line);
        for (std::string token; stream >> token;) {
            const size_t equals = token.find('=');
            if (equals != std::string::npos)
                fields[token.substr(0, equals)] = token.substr(equals + 1);
        }
        const auto phase_it = fields.find("phase");
        if (phase_it == fields.end() ||
            (phase_it->second != "read" && phase_it->second != "fill"))
            continue;
        const auto cpu_it = fields.find("cpu_count");
        assert(fields["threads"] == "32");
        assert(fields["applied"] == "32");
        assert(cpu_it != fields.end());
        assert(std::stoull(cpu_it->second) == expected_cpus);
        assert(fields["error"] == "0");
        ++seen;
    }
    // The production API contract is one summary for the read and one for
    // the fill phase.  This also proves that diagnostics covered both pools,
    // rather than merely observing the parent thread.
    assert(seen == 2);
}

int main() {
    const CpuSet original_mask = current_affinity();
    assert(!original_mask.empty());
    std::vector<int> available(original_mask.begin(), original_mask.end());
    std::vector<CpuSet> cases;
    cases.emplace_back(CpuSet{available.front()});
    if (available.size() >= 2)
        cases.emplace_back(CpuSet{available[0], available[1]});
    if (available.size() >= 4)
        cases.emplace_back(CpuSet{available[0], available[1], available[2], available[3]});

    const Fixture fixture = make_fixture();
    for (const CpuSet &launch_mask : cases) {
        set_affinity(launch_mask);
        // Capture before deliberately reducing only the parent to one CPU.
        const GraphLoadAffinity load_affinity;
        const int parent_cpu = *launch_mask.begin();
        set_affinity({parent_cpu});
        const CpuSet parent_pin = current_affinity();
        assert(parent_pin == CpuSet{parent_cpu});

        std::vector<std::vector<uint64_t>> adjacency;
        uint64_t vertices = 0;
        std::ostringstream captured;
        std::streambuf *old = std::cout.rdbuf(captured.rdbuf());
        const bool ok = readGraph_parallel(fixture.base.string(), adjacency,
                                           vertices, load_affinity);
        std::cout.rdbuf(old);
        assert(ok);
        assert_adjacency(fixture, adjacency, vertices);
        assert(current_affinity() == parent_pin);
        assert_affinity_diagnostics(captured.str(), launch_mask.size());
        set_affinity(launch_mask);
        assert(current_affinity() == launch_mask);
    }

    // Restore the caller's launch mask before the process exits.
    set_affinity(original_mask);
    assert(current_affinity() == original_mask);
    fs::remove_all(fixture.base.parent_path());
    std::cout << "BFS_GRAPH_LOADER_AFFINITY_PASS cases=" << cases.size() << "\n";
    return 0;
}
