#pragma once

#include "../kvs/latency_mode.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <vector>

namespace nq_latency {

constexpr size_t Fibres = 48;
constexpr uint64_t ArrivalSeed = 20260917;
constexpr uint64_t VertexSeed = 0x9e3779b97f4a7c15ULL;

inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void log(const std::string& line) {
    const std::string record = "\n" + line + "\n";
    std::fputs(record.c_str(), stderr);
}

struct Options {
    uint64_t offered_load_ops, warmup_ns, measurement_ns, drain_timeout_ns;
    uint64_t max_queue_delay_ns = 0;
    std::filesystem::path output_dir;
};

inline std::optional<Options> options_from_env() {
    const char* rate = std::getenv("NHOP_OFFERED_LOAD_OPS");
    if (!rate) return std::nullopt;
    const auto duration = [](const char* name) {
        return kv_latency::milliseconds_to_ns(
            kv_latency::parse_unsigned(std::getenv(name), name));
    };
    Options result{kv_latency::parse_unsigned(rate, "NHOP_OFFERED_LOAD_OPS"),
        duration("NHOP_LATENCY_WARMUP_MS"), duration("NHOP_LATENCY_MEASURE_MS"),
        duration("NHOP_DRAIN_TIMEOUT_MS"), 0, {}};
    if (const char* value = std::getenv("NHOP_MAX_QUEUE_DELAY_US"))
        result.max_queue_delay_ns = kv_latency::microseconds_to_ns(
            kv_latency::parse_unsigned(value, "NHOP_MAX_QUEUE_DELAY_US", true));
    const char* output = std::getenv("NHOP_LATENCY_OUTPUT_DIR");
    if (!output || !*output)
        throw std::invalid_argument("missing NHOP_LATENCY_OUTPUT_DIR");
    result.output_dir = output;
    const uint64_t range = kv_latency::checked_add(
        std::max(result.warmup_ns, result.measurement_ns), result.drain_timeout_ns);
    if (range > static_cast<uint64_t>(INT64_MAX))
        throw std::overflow_error("NQ latency histogram range overflow");
    kv_latency::checked_add(now_ns(), range);
    return result;
}

struct OracleQuery { uint64_t vertex, expected; };

// Independent host-vector calculation, before the source graph is discarded.
template <typename Graph>
uint64_t host_query(const Graph& graph, uint64_t vertex) {
    if (vertex >= graph.size()) throw std::out_of_range("NQ oracle vertex");
    uint64_t answer = graph[vertex].size();
    for (uint64_t neighbor : graph[vertex]) {
        if (neighbor >= graph.size()) throw std::out_of_range("NQ oracle neighbor");
        answer = kv_latency::checked_add(answer, graph[neighbor].size());
    }
    return answer;
}

template <typename Graph>
std::vector<OracleQuery> make_oracle(const Graph& graph, uint64_t vertices) {
    if (!vertices || vertices >= graph.size())
        throw std::invalid_argument("NQ oracle graph shape");
    std::mt19937_64 random(ArrivalSeed);
    std::uniform_int_distribution<uint64_t> vertex(0, vertices - 1);
    std::vector<OracleQuery> oracle;
    oracle.reserve(4096);
    for (size_t i = 0; i < 4096; ++i) {
        const uint64_t v = vertex(random);
        oracle.push_back({v, host_query(graph, v)});
    }
    return oracle;
}

template <typename Query>
bool post_verify(const std::vector<OracleQuery>& oracle, Query& query) {
    uint64_t failures = 0;
    for (const auto& sample : oracle)
        if (query(sample.vertex) != sample.expected) ++failures;
    std::ostringstream line;
    line << "nq_latency_post_verify checked=" << oracle.size()
         << " failures=" << failures;
    log(line.str());
    return oracle.size() == 4096 && failures == 0;
}

// Query owns its far-memory scope; neither arrival generation nor waiting does.
template <typename Query, typename Parallel, typename Yield>
bool run(const Options& options, uint64_t vertices, size_t adjacency_count,
         size_t os_workers, Query query, Parallel parallel, Yield yield) {
    if (vertices != 65608366 || adjacency_count != 65608367 || os_workers != 24)
        throw std::invalid_argument("NQ latency requires full Friendster and 24 OS workers");
    std::error_code error;
    std::filesystem::create_directories(options.output_dir, error);
    if (error) {
        log("nq_latency_error reason=histogram_directory");
        return false;
    }
    {
        std::ostringstream line;
        line << "nq_latency_config offered_load_ops=" << options.offered_load_ops
             << " fibres=" << Fibres << " os_workers=" << os_workers
             << " warmup_ns=" << options.warmup_ns
             << " measurement_ns=" << options.measurement_ns
             << " drain_timeout_ns=" << options.drain_timeout_ns
             << " max_queue_delay_ns=" << options.max_queue_delay_ns
             << " deadline_action=" << (options.max_queue_delay_ns
                 ? "drop_before_execution" : "disabled")
             << " arrival=poisson_per_fibre queue_model=fifo_independent_lanes"
             << " include_queue_wait=1 hist_sample_period=1"
             << " vertices=" << vertices << " adjacency_lists=" << adjacency_count
             << " query=benchmark_lite vertex_distribution=uniform arrival_seed=" << ArrivalSeed;
        log(line.str());
    }
    struct VertexStream {
        std::mt19937_64 random;
        std::uniform_int_distribution<uint64_t> vertex;
    };
    std::vector<VertexStream> streams;
    streams.reserve(Fibres);
    for (size_t lane = 0; lane < Fibres; ++lane)
        streams.push_back({std::mt19937_64(VertexSeed + lane),
                          std::uniform_int_distribution<uint64_t>(0, vertices - 1)});

    const auto phase = [&](const char* name, uint64_t phase_id, uint64_t window_ns) {
        struct Lane {
            kv_latency::LaneCounts counts;
            uint64_t fingerprint = 0, result_checksum = 0;
            std::unique_ptr<kv_latency::HistogramSet> histogram;
        };
        const uint64_t range = kv_latency::checked_add(window_ns, options.drain_timeout_ns);
        std::vector<Lane> lanes(Fibres);
        for (auto& lane : lanes)
            lane.histogram = std::make_unique<kv_latency::HistogramSet>(range);
        std::atomic_size_t ready{0};
        std::atomic_bool go{false};
        std::atomic<uint64_t> epoch{0};
        auto worker = [&](size_t tid) {
            auto& lane = lanes[tid];
            auto& stream = streams[tid];
            lane.fingerprint = 1469598103934665603ULL ^ tid;
            if (ready.fetch_add(1, std::memory_order_acq_rel) + 1 == Fibres) {
                const uint64_t start = now_ns();
                kv_latency::checked_add(start, range);
                epoch.store(start, std::memory_order_relaxed);
                std::ostringstream line;
                line << "nq_latency_phase name=" << name
                     << " event=start monotonic_ns=" << start;
                log(line.str());
                go.store(true, std::memory_order_release);
            }
            while (!go.load(std::memory_order_acquire)) yield();
            kv_latency::LaneSchedule schedule(
                epoch.load(std::memory_order_relaxed), window_ns,
                options.drain_timeout_ns, options.offered_load_ops, Fibres,
                kv_latency::lane_seed(ArrivalSeed, phase_id, tid),
                options.max_queue_delay_ns);
            uint64_t due;
            while (schedule.next(due)) {
                // Advance the original uniform vertex stream for every offered
                // query, even when it is later dropped or drain-skipped.
                const uint64_t vertex = stream.vertex(stream.random);
                lane.fingerprint ^= vertex;
                lane.fingerprint *= 1099511628211ULL;
                uint64_t begin = now_ns();
                while (begin < due) {
                    yield();
                    begin = now_ns();
                }
                if (schedule.start_decision(begin) !=
                    kv_latency::LaneSchedule::StartDecision::Started) {
                    if ((schedule.counts.scheduled & 63) == 0) yield();
                    continue;
                }
                const uint64_t value = query(vertex);
                const uint64_t complete = now_ns();
                schedule.complete(complete);
                lane.result_checksum += value;
                lane.histogram->record(due, begin, complete);
                if ((schedule.counts.scheduled & 63) == 0) yield();
            }
            while (now_ns() < schedule.window_end()) yield();
            lane.counts = schedule.counts;
        };
        parallel(Fibres, worker);
        const uint64_t end = now_ns();
        const uint64_t start = epoch.load(std::memory_order_relaxed);
        {
            std::ostringstream line;
            line << "nq_latency_phase name=" << name
                 << " event=end monotonic_ns=" << end;
            log(line.str());
        }
        kv_latency::LaneCounts total;
        uint64_t fingerprint = 0, checksum = 0, merge_dropped = 0;
        kv_latency::HistogramSet histogram(range);
        for (const auto& lane : lanes) {
            total.scheduled = kv_latency::checked_add(total.scheduled, lane.counts.scheduled);
            total.started = kv_latency::checked_add(total.started, lane.counts.started);
            total.completed = kv_latency::checked_add(total.completed, lane.counts.completed);
            total.completed_in_window = kv_latency::checked_add(
                total.completed_in_window, lane.counts.completed_in_window);
            total.deadline_dropped = kv_latency::checked_add(
                total.deadline_dropped, lane.counts.deadline_dropped);
            total.completed_after_deadline = kv_latency::checked_add(
                total.completed_after_deadline, lane.counts.completed_after_deadline);
            total.skipped = kv_latency::checked_add(total.skipped, lane.counts.skipped);
            total.drain_timeout |= lane.counts.drain_timeout;
            fingerprint ^= lane.fingerprint;
            checksum += lane.result_checksum;
            merge_dropped = kv_latency::checked_add(
                merge_dropped, histogram.merge(*lane.histogram));
        }
        if (total.started != total.completed ||
            total.scheduled != total.started + total.deadline_dropped + total.skipped)
            throw std::logic_error("NQ latency query accounting does not close");
        {
            std::ostringstream line;
            line << "nq_latency_receipt name=" << name
                 << " scheduled=" << total.scheduled << " started=" << total.started
                 << " completed=" << total.completed
                 << " completed_in_window=" << total.completed_in_window
                 << " deadline_dropped=" << total.deadline_dropped
                 << " completed_after_deadline=" << total.completed_after_deadline
                 << " skipped=" << total.skipped
                 << " request_fingerprint=" << fingerprint
                 << " result_checksum=" << checksum;
            log(line.str());
        }
        uint64_t write_errors = 0;
        const char* labels[] = {"total", "service", "dispatch"};
        for (size_t i = 0; i < 3; ++i) {
            const auto path = options.output_dir /
                (std::string(name) + "." + labels[i] + ".hgrm");
            FILE* file = std::fopen(path.c_str(), "wx");
            if (!file) {
                ++write_errors;
                continue;
            }
            if (hdr_percentiles_print(histogram.get(
                    static_cast<kv_latency::HistogramSet::Kind>(i)),
                    file, 100, 1, format_type::CLASSIC) != 0)
                ++write_errors;
            if (std::fclose(file) != 0) ++write_errors;
        }
        const bool passed = write_errors == 0 &&
            kv_latency::complete_measurement(total, histogram, merge_dropped);
        {
            std::ostringstream line;
            line << "nq_latency_result name=" << name
                 << " status=" << (passed ? "passed" : "invalid")
                 << " arrival_window_ns=" << window_ns
                 << " drain_elapsed_ns=" << end - kv_latency::checked_add(start, window_ns)
                 << " drain_timeout=" << total.drain_timeout
                 << " hist_count=" << histogram.count(kv_latency::HistogramSet::Total)
                 << " hist_service_count=" << histogram.count(kv_latency::HistogramSet::Service)
                 << " hist_dispatch_count=" << histogram.count(kv_latency::HistogramSet::Dispatch)
                 << " hist_dropped=" << histogram.dropped()
                 << " merge_dropped=" << merge_dropped
                 << " hist_write_errors=" << write_errors;
            if (passed) {
                line << " p99_ns=" << histogram.p99(kv_latency::HistogramSet::Total)
                     << " p99_service_ns=" << histogram.p99(kv_latency::HistogramSet::Service)
                     << " p99_dispatch_ns=" << histogram.p99(kv_latency::HistogramSet::Dispatch);
                const char* suffix[] = {"_ns", "_service_ns", "_dispatch_ns"};
                const char* names[] = {"p50", "p90", "p95", "p999", "p9999"};
                const double quantiles[] = {50, 90, 95, 99.9, 99.99};
                for (size_t kind = 0; kind < 3; ++kind) {
                    auto* raw = histogram.get(
                        static_cast<kv_latency::HistogramSet::Kind>(kind));
                    for (size_t q = 0; q < 5; ++q)
                        line << " " << names[q] << suffix[kind] << "="
                             << hdr_value_at_percentile(raw, quantiles[q]);
                    line << " max" << suffix[kind] << "=" << hdr_max(raw);
                }
            }
            log(line.str());
        }
        return passed;
    };
    const bool warmup_ok = phase("warmup", 0, options.warmup_ns);
    return warmup_ok && phase("measurement", 1, options.measurement_ns);
}

}  // namespace nq_latency
