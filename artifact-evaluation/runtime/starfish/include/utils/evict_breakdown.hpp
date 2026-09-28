#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "utils/cpu_cycles.hpp"

#ifndef FARLIB_ENABLE_EVICT_BREAKDOWN
#define FARLIB_ENABLE_EVICT_BREAKDOWN 0
#endif

namespace FarLib::profile::evict_breakdown {

// Diagnostic extension of the runtime's RDTSCP Work/evacuation profiling.
// A Worker belongs to a logical eviction invocation, NOT an OS thread. Nested
// scopes partition elapsed fibre time, including waits inside each scope.
enum class Stage : uint8_t {
    Other, CollectAdopt, ObjectScan, StageControl, GroupAllocate, CopyPad,
    Encode, SealQueue, SendPrepare, PostSend, CqProcess, BackpressureYield,
    GcPublish,
    // RMW stages are deliberately separate from the generic EC/update stages
    // above.  Call sites may nest the fine-grained scopes under RmwControl;
    // Worker::transition() keeps the resulting tick sums exclusive.
    RmwControl, RmwPrepare, RmwCollect, RmwSubmit, RmwEncode, RmwCommit,
    Count
};
inline constexpr size_t StageCount = static_cast<size_t>(Stage::Count);
inline constexpr const char *Names[StageCount] = {
    "other", "collect_adopt", "object_scan", "stage_control", "group_allocate",
    "copy_pad", "encode", "seal_queue", "send_prepare", "post_send",
    "cq_process", "backpressure_yield", "gc_publish", "rmw_control",
    "rmw_prepare", "rmw_collect", "rmw_submit", "rmw_encode", "rmw_commit"
};
inline constexpr size_t MaxWorkers = 16;
inline constexpr size_t MaxRounds = 8192;
inline std::atomic<uint64_t> work_begin{0}, work_end{0};

inline bool enabled() {
    if constexpr (!FARLIB_ENABLE_EVICT_BREAKDOWN) return false;
    static const bool on = [] {
        const char *v = std::getenv("FARLIB_EVAC_BREAKDOWN");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

inline uint64_t clipped(uint64_t begin, uint64_t end) {
    const auto wb = work_begin.load(std::memory_order_acquire);
    if (wb == 0 || end <= begin || end <= wb) return 0;
    const auto we = work_end.load(std::memory_order_acquire);
    begin = std::max(begin, wb);
    if (we != 0) end = std::min(end, we);
    return end > begin ? end - begin : 0;
}

struct Worker {
    uint64_t begin = 0, end = 0, last = 0;
    Stage stage = Stage::Other;
    std::array<uint64_t, StageCount> ticks{};

    void start(uint64_t now) { begin = last = now; }
    Stage transition(Stage next, uint64_t now) {
        const Stage previous = stage;
        ticks[static_cast<size_t>(stage)] += clipped(last, now);
        last = now;
        stage = next;
        return previous;
    }
    void finish(uint64_t now) { transition(Stage::Other, now); end = now; }
    uint64_t total() const {
        uint64_t n = 0;
        for (auto v : ticks) n += v;
        return n;
    }
};

struct Scope {
    Worker *worker = nullptr;
    Stage previous = Stage::Other;
    Scope(Worker *p, Stage next) {
        if constexpr (FARLIB_ENABLE_EVICT_BREAKDOWN) {
            worker = p;
            if (worker) previous = worker->transition(next, get_cycles());
        }
    }
    ~Scope() {
        if constexpr (FARLIB_ENABLE_EVICT_BREAKDOWN) {
            if (worker) worker->transition(previous, get_cycles());
        }
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
};

struct WorkerRun {
    Worker *worker;
    explicit WorkerRun(Worker *p) : worker(p) {
        if (worker) worker->start(get_cycles());
    }
    ~WorkerRun() { if (worker) worker->finish(get_cycles()); }
};

struct Round {
    uint64_t begin = 0, end = 0;
    size_t count = 0;
    std::array<Worker, MaxWorkers> workers{};
};
// Written only by the sequential eviction master, read after quiescence.
// Allocate once outside Work; there is no logging/allocation/global counter
// update on a per-object profiling transition.
inline std::unique_ptr<Round[]> history;
inline size_t rounds = 0, dropped = 0;
inline bool printed = false;

inline void prepare() {
    if (enabled() && !history) history = std::make_unique<Round[]>(MaxRounds);
}
inline void begin_window(uint64_t now) {
    if (!enabled()) return;
    work_end.store(0, std::memory_order_relaxed);
    work_begin.store(now, std::memory_order_release);
}
inline void end_window(uint64_t now) {
    if (enabled()) work_end.store(now, std::memory_order_release);
}
inline Round *open_round(uint64_t now, size_t count) {
    if (!enabled()) return nullptr;
    if (!history || rounds == MaxRounds || count > MaxWorkers || count == 0) {
        ++dropped;
        return nullptr;
    }
    Round *r = &history[rounds++];
    r->begin = now;
    r->count = count;
    return r;
}

// Last-finishing-worker path plus launch/join delay partitions one round's
// wall interval. It is distinct from the sum over overlapping worker times.
inline void report() {
    if (!enabled() || printed) return;
    printed = true;
    size_t unclosed = 0;
    for (size_t i = 0; i < rounds; ++i) {
        if (!history[i].end) ++unclosed;
        for (size_t w = 0; w < history[i].count; ++w)
            if (!history[i].workers[w].begin || !history[i].workers[w].end) ++unclosed;
    }
    const auto window_begin = work_begin.load(std::memory_order_acquire);
    const auto window_end = work_end.load(std::memory_order_acquire);
    std::cout << "evict_breakdown_window begin_tsc=" << window_begin
              << " end_tsc=" << window_end
              << " work_ticks=" << clipped(window_begin, window_end)
              << " rounds=" << rounds
              << " dropped=" << dropped << " unclosed=" << unclosed << " categories=";
    for (size_t i = 0; i < StageCount; ++i)
        std::cout << (i ? "," : "") << Names[i];
    std::cout << '\n';
    for (size_t i = 0; i < rounds; ++i) {
        const auto &r = history[i];
        if (clipped(r.begin, r.end) == 0) continue;
        size_t latest = 0;
        uint64_t earliest_begin = r.workers[0].begin;
        for (size_t w = 1; w < r.count; ++w)
        {
            if (r.workers[w].end > r.workers[latest].end) latest = w;
            earliest_begin = std::min(earliest_begin, r.workers[w].begin);
        }
        const auto &last = r.workers[latest];
        const bool boundary = r.begin < work_begin.load() || r.end > work_end.load();
        std::cout << "evict_breakdown_round id=" << i << " begin_tsc=" << r.begin
                  << " end_tsc=" << r.end << " ticks=" << clipped(r.begin, r.end)
                  << " boundary=" << boundary << " workers=" << r.count << " latest=" << latest
                  << " launch_ticks=" << clipped(r.begin, earliest_begin)
                  << " pre_latest_ticks=" << clipped(earliest_begin, last.begin)
                  << " join_ticks=" << clipped(last.end, r.end) << '\n';
        for (size_t w = 0; w < r.count; ++w) {
            const auto &p = r.workers[w];
            std::cout << "evict_breakdown_worker round=" << i << " worker=" << w
                      << " begin_tsc=" << p.begin << " end_tsc=" << p.end
                      << " ticks=" << clipped(p.begin, p.end)
                      << " category_sum=" << p.total();
            for (size_t k = 0; k < StageCount; ++k)
                std::cout << ' ' << Names[k] << '=' << p.ticks[k];
            std::cout << '\n';
        }
    }
    std::cout.flush();
}
}  // namespace FarLib::profile::evict_breakdown
