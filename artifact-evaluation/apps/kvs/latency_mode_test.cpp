#include "latency_mode.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

namespace {
struct Simulation {
    kv_latency::LaneCounts counts;
    kv_latency::HistogramSet hist;
    uint64_t arrival_fingerprint = 1469598103934665603ULL;
    uint64_t request_fingerprint = 1469598103934665603ULL;
    std::array<uint64_t, 2> generated{}, completed{}, dropped{}, skipped{};
    explicit Simulation(uint64_t range) : hist(range) {}
};

Simulation simulate(uint64_t window, uint64_t drain, uint64_t service,
                    uint64_t max_queue_delay_ns = 0) {
    Simulation result(window + drain);
    kv_latency::LaneSchedule schedule(0, window, drain, 100000, 1, 17,
                                     max_queue_delay_ns);
    std::mt19937_64 key_random(20), op_random(21);
    uint64_t now = 0, due;
    while (schedule.next(due)) {
        result.arrival_fingerprint ^= due;
        result.arrival_fingerprint *= 1099511628211ULL;
        const uint64_t key = key_random();
        const size_t op = op_random() % 20 == 0 ? 1 : 0;
        ++result.generated[op];
        result.request_fingerprint ^= key;
        result.request_fingerprint *= 1099511628211ULL;
        result.request_fingerprint ^= op;
        result.request_fingerprint *= 1099511628211ULL;
        now = std::max(now, due);
        const auto admission = schedule.start_decision(now);
        if (admission != kv_latency::LaneSchedule::StartDecision::Started) {
            if (admission == kv_latency::LaneSchedule::StartDecision::DeadlineDropped)
                ++result.dropped[op];
            else
                ++result.skipped[op];
            continue;
        }
        const uint64_t begin = now;
        now += service;
        schedule.complete(now);
        ++result.completed[op];
        result.hist.record(due, begin, now);
    }
    result.counts = schedule.counts;
    for (size_t op = 0; op < 2; ++op)
        assert(result.generated[op] ==
               result.completed[op] + result.dropped[op] + result.skipped[op]);
    return result;
}

template <typename Fn>
void rejects(Fn fn) {
    bool rejected = false;
    try { fn(); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
}

int main() {
    using namespace kv_latency;
    for (const char* invalid : {"", "-1", " 1", "1x", "18446744073709551616", "0"})
        rejects([&] { (void)parse_unsigned(invalid, "test"); });
    assert(parse_unsigned("18446744073709551615", "test") == UINT64_MAX);
    rejects([] { (void)milliseconds_to_ns(UINT64_MAX); });
    rejects([] { (void)microseconds_to_ns(UINT64_MAX); });
    assert(microseconds_to_ns(500) == 500000);
    assert(parse_unsigned("0", "queue deadline", true) == 0);
    rejects([] { (void)checked_add(UINT64_MAX, 1); });
    rejects([] { PoissonArrivals invalid(0, 1, 0); });
    rejects([] { PoissonArrivals invalid(1, 0, 0); });
    rejects([] { LaneSchedule invalid(UINT64_MAX, 1, 1, 1, 1, 0); });
    rejects([] { LaneSchedule invalid(0, UINT64_MAX, 1, 1, 1, 0); });
    rejects([] { HistogramSet invalid(UINT64_MAX); });

    const auto fast = simulate(10'000'000, 1'000'000, 1);
    auto queued = simulate(10'000'000, 200'000'000, 100'000);
    auto timed_out = simulate(10'000'000, 1'000'000, 100'000);
    assert(fast.counts.scheduled == queued.counts.scheduled);
    assert(queued.counts.scheduled == timed_out.counts.scheduled);
    assert(fast.arrival_fingerprint == queued.arrival_fingerprint);
    assert(queued.arrival_fingerprint == timed_out.arrival_fingerprint);
    assert(timed_out.counts.skipped > 0);
    assert(timed_out.counts.started == timed_out.counts.completed);
    assert(timed_out.counts.scheduled ==
           timed_out.counts.completed + timed_out.counts.skipped);
    assert(complete_measurement(fast.counts, fast.hist, 0));
    assert(complete_measurement(queued.counts, queued.hist, 0));
    assert(!complete_measurement(timed_out.counts, timed_out.hist, 0));
    assert(queued.hist.p99(HistogramSet::Total) >
           100 * queued.hist.p99(HistogramSet::Service));
    assert(queued.hist.p99(HistogramSet::Dispatch) > 0);

    using Decision = LaneSchedule::StartDecision;
    constexpr uint64_t queue_deadline = 500000;
    auto deadline = simulate(10'000'000, 200'000'000, 100'000, queue_deadline);
    auto explicit_zero = simulate(10'000'000, 200'000'000, 100'000, 0);
    assert(deadline.counts.deadline_dropped > 0 && deadline.counts.skipped == 0);
    assert(deadline.counts.started == deadline.counts.completed);
    assert(deadline.counts.scheduled ==
           deadline.counts.completed + deadline.counts.deadline_dropped);
    assert(deadline.arrival_fingerprint == queued.arrival_fingerprint);
    assert(deadline.request_fingerprint == queued.request_fingerprint);
    assert(deadline.generated == queued.generated);
    assert(complete_measurement(deadline.counts, deadline.hist, 0));
    assert(deadline.counts.completed_after_deadline > 0);
    assert(deadline.hist.p99(HistogramSet::Total) > queue_deadline);
    assert(explicit_zero.counts.deadline_dropped == 0);
    assert(explicit_zero.counts.completed_after_deadline == 0);
    assert(explicit_zero.request_fingerprint == queued.request_fingerprint);
    assert(explicit_zero.hist.p99(HistogramSet::Total) ==
           queued.hist.p99(HistogramSet::Total));

    uint64_t boundary_due;
    LaneSchedule equal(0, 10000000, 1000000, 1000000000, 1, 19, queue_deadline);
    assert(equal.next(boundary_due));
    assert(equal.start_decision(boundary_due + queue_deadline) == Decision::Started);
    equal.complete(boundary_due + queue_deadline);
    assert(equal.counts.completed_after_deadline == 0);
    LaneSchedule over(0, 10000000, 1000000, 1000000000, 1, 19, queue_deadline);
    assert(over.next(boundary_due));
    assert(over.start_decision(boundary_due + queue_deadline + 1) ==
           Decision::DeadlineDropped);
    assert(over.counts.started == 0 && over.counts.deadline_dropped == 1);
    LaneSchedule inflight(0, 10000000, 1000000, 1000000000, 1, 19, queue_deadline);
    assert(inflight.next(boundary_due));
    const uint64_t admitted = boundary_due + queue_deadline - 1;
    assert(inflight.start_decision(admitted) == Decision::Started);
    inflight.complete(boundary_due + 2000000);
    HistogramSet slow_hist(11000000);
    slow_hist.record(boundary_due, admitted, boundary_due + 2000000);
    assert(inflight.counts.completed == 1 &&
           inflight.counts.completed_after_deadline == 1 &&
           inflight.counts.deadline_dropped == 0);
    assert(slow_hist.count(HistogramSet::Total) == 1 &&
           slow_hist.p99(HistogramSet::Total) >= 2000000);
    LaneSchedule drain_first(0, 1000000, 1000000, 1000000000, 1, 19, queue_deadline);
    assert(drain_first.next(boundary_due));
    assert(drain_first.start_decision(drain_first.deadline()) == Decision::DrainSkipped);
    assert(drain_first.counts.deadline_dropped == 0 && drain_first.counts.drain_timeout);
    LaneSchedule all_dropped(0, 10000000, 1000000, 100000, 1, 17, queue_deadline);
    while (all_dropped.next(boundary_due))
        assert(all_dropped.start_decision(boundary_due + queue_deadline + 1) ==
               Decision::DeadlineDropped);
    HistogramSet empty_hist(11000000);
    assert(all_dropped.counts.scheduled == all_dropped.counts.deadline_dropped);
    assert(!complete_measurement(all_dropped.counts, empty_hist, 0));

    PoissonArrivals replay(100000, 1, 17);
    uint64_t count = 0, offset = 0;
    while (replay.next(10'000'000, offset)) {
        assert(offset < 10'000'000);
        ++count;
    }
    assert(count == timed_out.counts.scheduled);
    HistogramSet merged(210'000'000);
    assert(merged.merge(queued.hist) == 0);
    assert(merged.count(HistogramSet::Total) == queued.counts.completed);
    assert(merged.count(HistogramSet::Service) == queued.counts.completed);
    assert(merged.count(HistogramSet::Dispatch) == queued.counts.completed);

    HistogramSet zero_dispatch(1000);
    zero_dispatch.record(10, 10, 20);
    assert(zero_dispatch.count(HistogramSet::Dispatch) == 1);
    assert(zero_dispatch.p99(HistogramSet::Dispatch) == 0);
    HistogramSet too_small(1000);
    too_small.record(0, 1000000, 2000000);
    assert(too_small.dropped() == 3);
    assert(!complete_measurement(queued.counts, too_small, 0));
    HistogramSet full_count(1000);
    full_count.get(HistogramSet::Total)->total_count = INT64_MAX;
    rejects([&] { (void)full_count.merge(zero_dispatch); });

    LaneSchedule late(0, 1000, 100, 1000000000, 1, 19);
    uint64_t due;
    assert(late.next(due));
    assert(late.start(1099));
    late.complete(1101);
    assert(late.counts.drain_timeout && late.counts.completed == 1);
    while (late.next(due)) assert(!late.start(1101));
    assert(late.counts.scheduled == late.counts.completed + late.counts.skipped);

    uint64_t aggregate = 0;
    for (size_t lane = 0; lane < 48; ++lane) {
        PoissonArrivals stream(4800000, 48, lane_seed(20260917, 0, lane));
        while (stream.next(10000000, offset)) ++aggregate;
    }
    assert(aggregate > 47000 && aggregate < 49000);
    assert(lane_seed(20260917, 0, 0) != lane_seed(20260917, 1, 0));
    assert(lane_seed(20260917, 0, 0) != lane_seed(20260917, 0, 1));

    std::cout << "kv_latency_cpu_test status=pass scheduled=" << count
              << " skipped_on_timeout=" << timed_out.counts.skipped
              << " aggregate_48_lane=" << aggregate
              << " queued_p99_ns=" << queued.hist.p99(HistogramSet::Total)
              << " service_p99_ns=" << queued.hist.p99(HistogramSet::Service)
              << " deadline_dropped=" << deadline.counts.deadline_dropped
              << " completed_after_deadline=" << deadline.counts.completed_after_deadline
              << std::endl;
}
