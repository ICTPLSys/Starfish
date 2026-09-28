#include "latency_mode.hpp"
#include <cassert>
#include <iostream>

int main() {
    const std::vector<std::vector<uint64_t>> graph{{1, 2}, {0}, {0, 3}, {2}};
    const std::array<uint64_t, 4> expected{5, 3, 5, 3};
    for (size_t i = 0; i < graph.size(); ++i)
        assert(nq_latency::host_query(graph, i) == expected[i]);
    const auto oracle = nq_latency::make_oracle(graph, 3);
    const auto replay = nq_latency::make_oracle(graph, 3);
    assert(oracle.size() == 4096);
    for (size_t i = 0; i < oracle.size(); ++i) {
        assert(oracle[i].vertex < 3);
        assert(oracle[i].expected == expected[oracle[i].vertex]);
        assert(oracle[i].vertex == replay[i].vertex &&
               oracle[i].expected == replay[i].expected);
    }
    auto correct_query = [&](uint64_t vertex) { return expected[vertex]; };
    auto incorrect_query = [&](uint64_t vertex) { return expected[vertex] + 1; };
    assert(nq_latency::post_verify(oracle, correct_query));
    assert(!nq_latency::post_verify(oracle, incorrect_query));
    auto invalid = graph;
    invalid[0].push_back(99);
    bool rejected = false;
    try { (void)nq_latency::host_query(invalid, 0); }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);
    std::mt19937_64 continuous(nq_latency::VertexSeed + 7);
    std::mt19937_64 split(nq_latency::VertexSeed + 7);
    std::uniform_int_distribution<uint64_t> dist(0, 65608365);
    std::vector<uint64_t> continuous_vertices, split_vertices;
    for (size_t i = 0; i < 200; ++i) continuous_vertices.push_back(dist(continuous));
    for (size_t phase = 0; phase < 2; ++phase)
        for (size_t i = 0; i < 100; ++i) split_vertices.push_back(dist(split));
    assert(continuous_vertices == split_vertices);
    std::cout << "nq_latency_cpu_test status=pass oracle_samples=4096"
                 " correct_oracle=pass wrong_oracle=rejected invalid_neighbor=rejected"
                 " uniform_stream_continuity=pass\n";
}
