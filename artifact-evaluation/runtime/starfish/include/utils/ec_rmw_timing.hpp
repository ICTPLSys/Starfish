#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "utils/stats.hpp"
#include "utils/ec_rmw_prepare_trace.hpp"
#include "utils/benchmark_phase.hpp"

#ifndef FARLIB_ENABLE_EC_RMW_TIMING
#define FARLIB_ENABLE_EC_RMW_TIMING 0
#endif

// RMW extension of the native runtime TSC profiler. One private record per
// existing in-flight slot; no WR callback hooks, shared updates or hot malloc.
namespace FarLib::profile::ec_rmw_timing {
// Object timing is intentionally sampled to keep the private per-owner
// history bounded.  Keep this contract in the producer as well as in parser
// output; sampled sums must never be presented as exhaustive timing.
inline constexpr uint64_t SampleEvery = 64;
inline constexpr bool CompileEnabled = FARLIB_ENABLE_EC_RMW_TIMING != 0;
inline bool enabled() {
    if constexpr (!CompileEnabled) {
        return false;
    } else {
        static const bool on = [] {
            const char *v = std::getenv("FARLIB_EC_RMW_TIMING");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        return on;
    }
}
enum Metric { Prepare, Read, Encode, Write, Commit, BusyWait, ReadyWait,
              BatchPrepare, BatchRead, BatchEncode, BatchWrite, BatchCommit,
              BatchLife, ReadSubmit, WriteSubmit, FallbackPrepare,
              ReadVerbs, WriteVerbs, ReadTransport, WriteTransport,
              PrepEnqueue, PrepSummary, PrepLockWait, PrepSelect, PrepIndex,
              PrepOther, PrepPublish, PrepRetire, PrepSameGroup, PrepOtherGroup,
              EncodeDiff, EncodeUpdate, EncodeOther, Count };
inline const char *names[Count] = {
    "prepare_object", "read_object", "encode_object", "write_object",
    "commit_object", "busy_wait_object", "read_ready_to_encode_object",
    "prepare_batch", "read_span_batch", "encode_batch", "write_span_batch",
    "commit_batch", "lifetime_batch", "read_submit_batch", "write_submit_batch",
    "fallback_prepare_object",
    "read_verbs_batch", "write_verbs_batch", "read_transport_batch", "write_transport_batch",
    "prepare_enqueue_batch", "prepare_summary_batch", "prepare_lock_wait_batch",
    "prepare_select_validate_batch", "prepare_index_batch", "prepare_other_batch",
    "prepare_publish_batch", "prepare_retire_old_batch",
    "prepare_same_group_total_batch", "prepare_other_group_total_batch",
    "encode_diff_batch", "encode_isal_update_batch", "encode_other_batch"
};
// Eight bins per power of two. Quantiles are reported as explicit bounds,
// never presented as exact percentiles. Exact sums retain the exact mean.
struct Distribution {
    uint64_t count = 0, sum = 0, maximum = 0;
    std::array<uint64_t, 512> bins{};
    static size_t bin(uint64_t x) {
        if (x < 8) return size_t(x);
        const unsigned e = 63u - unsigned(std::countl_zero(x));
        return size_t(e * 8 + ((x >> (e - 3)) - 8));
    }
    static uint64_t lower(size_t b) {
        if (b < 8) return b;
        return uint64_t(8 + b % 8) << (b / 8 - 3);
    }
    static uint64_t upper(size_t b) {
        if (b < 8) return b;
        const auto step = uint64_t{1} << (b / 8 - 3);
        return lower(b) + step - 1;
    }
    void add(uint64_t x) {
        ++count; sum += x; maximum = std::max(maximum, x); ++bins[bin(x)];
    }
    size_t quantile(unsigned pct) const {
        const uint64_t rank = (count * pct + 99) / 100;
        uint64_t n = 0;
        for (size_t b = 0; b < bins.size(); ++b) {
            n += bins[b]; if (n >= rank) return b;
        }
        return 0;
    }
};
struct Object {
    uint64_t prepare = 0, read_begin = 0, read_end = 0, encode_begin = 0;
    uint64_t encode = 0, write_begin = 0, write_end = 0, commit = 0;
    uint64_t busy_begin = 0, busy_end = 0;
    uint64_t prepare_enqueue = 0, prepare_same_group = 0, prepare_other_group = 0;
    uint64_t prepare_publish = 0, prepare_retire = 0, other_group_calls = 0;
    ec_rmw_prepare::Trace prepare_trace{};
    uint64_t encode_diff = 0, encode_update = 0;
    bool committed = false, fallback = false;
};
struct Batch {
    bool selected = false;
    bool bulk_allocator = false;
    uint32_t phase = 0, fibres = 0;
    size_t objects = 0;
    uint64_t begin = 0, read_submit = 0, write_submit = 0, busy_attempts = 0;
    uint64_t read_verbs = 0, write_verbs = 0, read_transport = 0, write_transport = 0;
    uint64_t read_post_calls = 0, read_post_attempted_wrs = 0, read_post_accepted_wrs = 0;
    uint64_t read_post_partial_calls = 0, read_post_max_chain = 0;
    uint64_t write_post_calls = 0, write_post_attempted_wrs = 0, write_post_accepted_wrs = 0;
    uint64_t write_post_partial_calls = 0, write_post_max_chain = 0;
    std::array<Object, 64> object{};
};
struct Window {
    bool bulk_allocator = false;
    uint32_t fibres = 0;
    uint64_t batches = 0, objects = 0, committed = 0, fallback = 0;
    uint64_t busy_attempts = 0, boundary_dropped = 0;
    uint64_t summary_calls = 0, summary_misses = 0, other_group_calls = 0;
    uint64_t prepare_accounting_errors = 0, encode_accounting_errors = 0;
    uint64_t read_post_calls = 0, read_post_attempted_wrs = 0, read_post_accepted_wrs = 0;
    uint64_t read_post_partial_calls = 0, read_post_max_chain = 0;
    uint64_t write_post_calls = 0, write_post_attempted_wrs = 0, write_post_accepted_wrs = 0;
    uint64_t write_post_partial_calls = 0, write_post_max_chain = 0;
    std::array<Distribution, Count> metrics{};
};
struct Owner {
    uint64_t sequence = 0;
    std::array<Batch, 4> batch{};
    std::array<Window, 4> window{};
    void begin(size_t b, size_t objects) {
        auto &p = batch[b];
        p.selected = false;
        if (!is_working() || ((++sequence & (SampleEvery - 1)) != 0)) return;
        const auto phase = benchmark_phase::stage_ordinal.load(std::memory_order_acquire);
        if (phase == 0 || phase >= window.size()) return;
        p = Batch{};
        p.selected = true;
        p.phase = phase;
        p.fibres = benchmark_phase::formal_fibres.load(std::memory_order_relaxed);
        p.objects = objects;
        p.begin = get_cycles();
    }
    void finish(size_t b) {
        auto &p = batch[b];
        if (!p.selected) return;
        p.selected = false;
        auto &w = window[p.phase];
        w.fibres = p.fibres;
        if (!is_working() || p.phase != benchmark_phase::stage_ordinal.load(std::memory_order_acquire)) {
            ++w.boundary_dropped; return;
        }
        uint64_t prepare = 0, encode = 0, commit = 0;
        uint64_t rb = UINT64_MAX, re = 0, wb = UINT64_MAX, we = 0;
        size_t completed = 0;
        std::array<uint64_t, Count> parts{};
        ++w.batches; w.objects += p.objects; w.busy_attempts += p.busy_attempts;
        w.bulk_allocator |= p.bulk_allocator;
        w.read_post_calls += p.read_post_calls;
        w.read_post_attempted_wrs += p.read_post_attempted_wrs;
        w.read_post_accepted_wrs += p.read_post_accepted_wrs;
        w.read_post_partial_calls += p.read_post_partial_calls;
        w.read_post_max_chain = std::max(w.read_post_max_chain, p.read_post_max_chain);
        w.write_post_calls += p.write_post_calls;
        w.write_post_attempted_wrs += p.write_post_attempted_wrs;
        w.write_post_accepted_wrs += p.write_post_accepted_wrs;
        w.write_post_partial_calls += p.write_post_partial_calls;
        w.write_post_max_chain = std::max(w.write_post_max_chain, p.write_post_max_chain);
        for (size_t i = 0; i < p.objects; ++i) {
            const auto &o = p.object[i];
            prepare += o.prepare;
            const auto &q = o.prepare_trace;
            w.summary_calls += q.summary_calls;
            w.summary_misses += q.summary_misses;
            w.other_group_calls += o.other_group_calls;
            const uint64_t known_prepare = o.prepare_enqueue + q.summary +
                q.lock_wait + q.stripe_held + o.prepare_publish + o.prepare_retire;
            if (q.stripe_held < q.index || o.prepare < known_prepare) {
                ++w.prepare_accounting_errors;
            } else {
                parts[PrepEnqueue] += o.prepare_enqueue;
                parts[PrepSummary] += q.summary;
                parts[PrepLockWait] += q.lock_wait;
                parts[PrepSelect] += q.stripe_held - q.index;
                parts[PrepIndex] += q.index;
                parts[PrepOther] += o.prepare - known_prepare;
                parts[PrepPublish] += o.prepare_publish;
                parts[PrepRetire] += o.prepare_retire;
            }
            parts[PrepSameGroup] += o.prepare_same_group;
            parts[PrepOtherGroup] += o.prepare_other_group;
            if (o.fallback) { ++w.fallback; w.metrics[FallbackPrepare].add(o.prepare); }
            if (!o.committed) continue;
            ++completed; ++w.committed;
            if (o.encode < o.encode_diff + o.encode_update) {
                ++w.encode_accounting_errors;
            } else {
                parts[EncodeDiff] += o.encode_diff;
                parts[EncodeUpdate] += o.encode_update;
                parts[EncodeOther] += o.encode - o.encode_diff - o.encode_update;
            }
            w.metrics[Prepare].add(o.prepare);
            w.metrics[Read].add(o.read_end - o.read_begin);
            w.metrics[Encode].add(o.encode);
            w.metrics[Write].add(o.write_end - o.write_begin);
            w.metrics[Commit].add(o.commit);
            w.metrics[ReadyWait].add(o.encode_begin - o.read_end);
            if (o.busy_begin) w.metrics[BusyWait].add(o.busy_end - o.busy_begin);
            encode += o.encode; commit += o.commit;
            rb = std::min(rb, o.read_begin); re = std::max(re, o.read_end);
            wb = std::min(wb, o.write_begin); we = std::max(we, o.write_end);
        }
        if (completed) {
            w.metrics[BatchPrepare].add(prepare);
            w.metrics[BatchRead].add(re - rb);
            w.metrics[BatchEncode].add(encode);
            w.metrics[BatchWrite].add(we - wb);
            w.metrics[BatchCommit].add(commit);
            w.metrics[BatchLife].add(get_cycles() - p.begin);
            w.metrics[ReadSubmit].add(p.read_submit);
            w.metrics[WriteSubmit].add(p.write_submit);
            w.metrics[ReadVerbs].add(p.read_verbs);
            w.metrics[WriteVerbs].add(p.write_verbs);
            w.metrics[ReadTransport].add(p.read_transport);
            w.metrics[WriteTransport].add(p.write_transport);
            for (size_t m = PrepEnqueue; m < Count; ++m)
                w.metrics[m].add(parts[m]);
        }
    }
};
struct History {
    size_t count;
    std::unique_ptr<Owner[]> owners;
    explicit History(size_t n) : count(n), owners(new Owner[n]) {
        // object_size warms up beyond local capacity, before start_work().
        // Refuse to silently allocate diagnostic storage inside timed Work.
        if (is_working()) std::abort();
    }
    ~History() {
        for (size_t o = 0; o < count; ++o) for (size_t s = 1; s < 4; ++s) {
            const auto &w = owners[o].window[s];
            if (!w.batches && !w.boundary_dropped) continue;
            std::printf("ec_rmw_timing_window owner=%zu phase=%zu fibres=%u sample_every=%llu batches=%llu objects=%llu committed=%llu fallback=%llu busy_attempts=%llu boundary_dropped=%llu units=tsc_cycles completion=worker_observed_all_three\n", o, s, w.fibres,
                (unsigned long long)SampleEvery,
                (unsigned long long)w.batches, (unsigned long long)w.objects,
                (unsigned long long)w.committed, (unsigned long long)w.fallback,
                (unsigned long long)w.busy_attempts, (unsigned long long)w.boundary_dropped);
            std::printf("ec_rmw_prepare_window owner=%zu phase=%zu fibres=%u summary_calls=%llu summary_misses=%llu other_group_calls=%llu prepare_accounting_errors=%llu encode_accounting_errors=%llu\n", o, s, w.fibres,
                (unsigned long long)w.summary_calls, (unsigned long long)w.summary_misses,
                (unsigned long long)w.other_group_calls,
                (unsigned long long)w.prepare_accounting_errors,
                (unsigned long long)w.encode_accounting_errors);
            std::printf("ec_rmw_verbs_window owner=%zu phase=%zu fibres=%u read_post_calls=%llu read_post_attempted_wrs=%llu read_post_accepted_wrs=%llu read_post_partial_calls=%llu read_post_max_chain=%llu write_post_calls=%llu write_post_attempted_wrs=%llu write_post_accepted_wrs=%llu write_post_partial_calls=%llu write_post_max_chain=%llu prepare_attribution=%s commit_attribution=%s\n",
                o, s, w.fibres,
                (unsigned long long)w.read_post_calls,
                (unsigned long long)w.read_post_attempted_wrs,
                (unsigned long long)w.read_post_accepted_wrs,
                (unsigned long long)w.read_post_partial_calls,
                (unsigned long long)w.read_post_max_chain,
                (unsigned long long)w.write_post_calls,
                (unsigned long long)w.write_post_attempted_wrs,
                (unsigned long long)w.write_post_accepted_wrs,
                (unsigned long long)w.write_post_partial_calls,
                (unsigned long long)w.write_post_max_chain,
                w.bulk_allocator ? "worker_batch_amortized" : "object_interval",
                w.bulk_allocator ? "worker_batch_amortized" : "object_interval");
            for (size_t m = 0; m < Count; ++m) {
                const auto &d = w.metrics[m];
                if (!d.count) continue;
                std::printf("ec_rmw_timing owner=%zu phase=%zu fibres=%u metric=%s count=%llu sum=%llu max=%llu hist=", o, s, w.fibres, names[m],
                    (unsigned long long)d.count, (unsigned long long)d.sum, (unsigned long long)d.maximum);
                for (size_t b = 0; b < d.bins.size(); ++b)
                    if (d.bins[b]) std::printf("%zu:%llu,", b, (unsigned long long)d.bins[b]);
                std::putchar('\n');
            }
        }
    }
};
inline Owner *owner(size_t index, size_t count) {
    if constexpr (!CompileEnabled) {
        (void)index;
        (void)count;
        return nullptr;
    } else {
        if (!enabled()) return nullptr;
        static History history(count);
        return &history.owners[index];
    }
}
} // namespace FarLib::profile::ec_rmw_timing
