// Deterministic CPU regression for the migration-safe evict breakdown helper.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "utils/evict_breakdown.hpp"

#if FARLIB_ENABLE_EVICT_BREAKDOWN

namespace {

using FarLib::profile::evict_breakdown::Stage;
using FarLib::profile::evict_breakdown::Worker;
using FarLib::profile::evict_breakdown::clipped;
using FarLib::profile::evict_breakdown::work_begin;
using FarLib::profile::evict_breakdown::work_end;

struct Checks {
    int failures = 0;

    void expect(bool condition, const char *label) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << label << '\n';
        }
    }

    void equal(uint64_t got, uint64_t want, const char *label) {
        expect(got == want, label);
        if (got != want) {
            std::cerr << "  got=" << got << " want=" << want << '\n';
        }
    }
};

void window(uint64_t begin, uint64_t end) {
    work_end.store(end, std::memory_order_release);
    work_begin.store(begin, std::memory_order_release);
}

uint64_t ticks(const Worker &worker, Stage stage) {
    return worker.ticks[static_cast<size_t>(stage)];
}

void test_nested_exclusive(Checks &checks) {
    window(100, 200);
    Worker worker;
    worker.start(100);
    const auto outer = worker.transition(Stage::GroupAllocate, 110);
    const auto inner = worker.transition(Stage::Encode, 120);
    worker.transition(inner, 130);
    worker.transition(outer, 150);
    worker.finish(180);

    checks.expect(outer == Stage::Other, "outer transition returns prior stage");
    checks.expect(inner == Stage::GroupAllocate,
                  "nested transition returns parent stage");
    checks.equal(ticks(worker, Stage::Other), 40, "nested other ticks");
    checks.equal(ticks(worker, Stage::GroupAllocate), 30,
                 "nested group allocation ticks");
    checks.equal(ticks(worker, Stage::Encode), 10, "nested encode ticks");
    checks.equal(worker.total(), 80, "nested categories are exclusive");
}

void test_window_clipping(Checks &checks) {
    window(100, 0);
    Worker init_to_work;
    init_to_work.start(50);
    init_to_work.transition(Stage::Encode, 120);
    init_to_work.finish(160);
    checks.equal(ticks(init_to_work, Stage::Other), 20,
                 "initialization prefix is clipped");
    checks.equal(ticks(init_to_work, Stage::Encode), 40,
                 "open work window retains work suffix");

    window(100, 150);
    Worker end_mid_stage;
    end_mid_stage.start(120);
    end_mid_stage.transition(Stage::Encode, 130);
    end_mid_stage.finish(200);
    checks.equal(ticks(end_mid_stage, Stage::Other), 10,
                 "work end clips prior stage");
    checks.equal(ticks(end_mid_stage, Stage::Encode), 20,
                 "work end clips active stage");
    checks.equal(end_mid_stage.total(), 30, "mid-stage work end total");

    window(0, 0);
    Worker outside_work;
    outside_work.start(10);
    outside_work.transition(Stage::PostSend, 20);
    outside_work.finish(30);
    checks.equal(outside_work.total(), 0, "no work window contributes zero");
}

void test_sequential_handoff(Checks &checks) {
    window(100, 200);
    Worker worker;
    worker.start(100);
    std::thread first([&] { worker.transition(Stage::Encode, 125); });
    first.join();
    std::thread second([&] { worker.transition(Stage::PostSend, 150); });
    second.join();
    worker.finish(175);
    checks.equal(ticks(worker, Stage::Other), 25,
                 "handoff preserves prefix category");
    checks.equal(ticks(worker, Stage::Encode), 25,
                 "handoff preserves encode category");
    checks.equal(ticks(worker, Stage::PostSend), 25,
                 "handoff preserves post category");
    checks.equal(worker.total(), 75,
                 "logical worker context is independent of OS thread");
}

void test_overlap_and_latest(Checks &checks) {
    window(100, 320);
    Worker workers[3];
    workers[0].start(100);
    workers[0].transition(Stage::ObjectScan, 180);
    workers[0].finish(260);  // longest duration, but not latest end
    workers[1].start(220);
    workers[1].transition(Stage::PostSend, 260);
    workers[1].finish(300);  // latest end
    workers[2].start(150);
    workers[2].transition(Stage::CqProcess, 190);
    workers[2].finish(240);

    const uint64_t cumulative = workers[0].total() + workers[1].total() +
                                 workers[2].total();
    checks.expect(cumulative > clipped(100, 320),
                  "overlapping worker fibre time exceeds wall time");
    size_t latest = 0;
    size_t longest = 0;
    for (size_t i = 1; i < 3; ++i) {
        if (workers[i].end > workers[latest].end) latest = i;
        if (workers[i].total() > workers[longest].total()) longest = i;
    }
    checks.equal(latest, 1, "latest worker is selected by end timestamp");
    checks.equal(longest, 0, "longest worker is a distinct worker");

    const uint64_t prefix = workers[latest].begin - 100;
    const uint64_t selected = workers[latest].total();
    const uint64_t tail = 320 - workers[latest].end;
    checks.equal(prefix + selected + tail, clipped(100, 320),
                 "prefix plus latest worker plus tail closes phase");
}

}  // namespace

int main() {
    setenv("FARLIB_EVAC_BREAKDOWN", "1", 1);
    FarLib::profile::evict_breakdown::prepare();
    Checks checks;
    test_nested_exclusive(checks);
    test_window_clipping(checks);
    test_sequential_handoff(checks);
    test_overlap_and_latest(checks);
    if (checks.failures != 0) return 1;
    std::cout << "evict breakdown regression PASS\n";
    return 0;
}

#else

int main() {
    std::cerr << "SKIP: FARLIB_ENABLE_EVICT_BREAKDOWN=0\n";
    return 77;
}

#endif
