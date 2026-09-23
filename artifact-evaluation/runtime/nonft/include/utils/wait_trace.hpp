#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <x86intrin.h>

// Exploratory diagnostics only. No runtime decisions depend on this trace.
namespace wait_trace {
struct Record {
  uint64_t tsc, fiber, bin, a, b;
  const char *event;
};
inline constexpr uint64_t capacity = 1U << 22;
inline std::unique_ptr<Record[]> records;
inline std::atomic<bool> enabled{false};
inline std::atomic<bool> round_gap_only{false};
inline std::atomic<bool> nq_supply_only{false};
inline std::atomic<uint64_t> cursor{0}, writers{0};
inline uint64_t begin_tsc = 0, end_tsc = 0;
inline const char *output = nullptr;
inline uint32_t stage_ordinal = 0;
inline char staged_output[4096]{};

inline bool is_nq_supply_event(const char *event) {
  // Work-phase aggregate records, allocation cold paths, and 1/64 region samples.
  // Exclude per-access Scope events and per-object allocation batches.
  if (std::strncmp(event, "reclaim_", 8) == 0) return true;
  static constexpr const char *events[] = {
      "alloc_wait_begin", "alloc_wait_end", "alloc_retry_fail", "alloc_retry_succ",
      "alloc_wait_placement", "alloc_take_placement",
      "wait_sleep", "wait_return", "wait_relocked",
      "ondemand_enter", "ondemand_notify", "notify_mutators_ready", "notify_locked",
      "evac_round_begin_legacy_opt", "evac_round_end_legacy_opt",
      "streaming_mark_begin", "mark_dispatch", "mark_worker_begin", "mark_worker_end",
      "mark_scan_end", "mark_evict_stop", "mark_scan_stop",
      "flip_scope_begin", "flip_scope_end", "streaming_mark_end",
      "streaming_evict_begin", "streaming_evict_end", "evict_post_worker_end",
      "evict_post_stop", "evict_mark_stop", "gc_begin", "gc_end",
      "gc_filter_accept", "gc_filter_reject", "gc_objects_checks", "gc_objects_free",
      "gc_objects_marked", "gc_objects_evicting",
      "tail_join_mark_begin", "tail_join_mark_end", "tail_join_evict_begin", "tail_join_evict_end",
      "resume_evict_scan", "resume_gc_scan", "tail_evict_end_free",
      "evac_wait_trigger_enter", "evac_wait_trigger_sleep", "evac_wait_trigger_exit"
  };
  for (const char *allowed : events)
    if (std::strcmp(event, allowed) == 0) return true;
  return false;
}

inline bool is_round_gap_event(const char *event) {
    // Observation only: expose existing per-worker aggregate mark records.
    static const bool marker_work_trace = [] {
        const char *v = std::getenv("FARLIB_MARK_WORK_TRACE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (marker_work_trace) {
        static constexpr const char *detail[] = {
            "reclaim_mark_scan", "reclaim_mark_regions", "reclaim_mark_lock",
            "reclaim_mark_lock_samples", "reclaim_mark_cycles",
            "reclaim_mark_body", "reclaim_mark_eligible"
        };
        for (const char *name : detail)
            if (std::strcmp(event, name) == 0) return true;
    }
  static constexpr const char *events[] = {
      "evac_round_begin_legacy_opt",
      "streaming_mark_begin",
      "mark_dispatch",
      "mark_worker_begin",
      "mark_worker_end",
      "mark_scan_end",
      "mark_evict_stop",
      "mark_scan_stop",
      "flip_scope_begin",
      "flip_scope_end",
      "streaming_mark_end",
      "streaming_evict_begin",
      "gc_begin",
      "gc_filter_accept",
      "gc_filter_reject",
      "gc_objects_checks",
      "gc_objects_free",
      "gc_objects_marked",
      "gc_objects_evicting",
      "gc_end",
      "streaming_evict_end",
      "evict_post_worker_end",
      "evict_post_stop",
      "evict_mark_stop",
      "tail_join_mark_begin",
      "tail_join_mark_end",
      "tail_join_evict_begin",
      "tail_join_evict_end",
      "evac_round_end_legacy_opt",
      "evac_wait_trigger_enter",
      "evac_wait_trigger_sleep",
      "evac_wait_trigger_exit",
      "scope_flip_begin",
      "scope_old_count",
      "scope_refill",
      "scope_enter",
      "scope_update_before",
      "scope_update_before_state",
      "scope_update_after",
      "scope_update_after_state",
      "scope_exit",
      "scope_flip_end",
  };
  for (const char *allowed : events) {
    if (std::strcmp(event, allowed) == 0) return true;
  }
  return false;
}

inline void emit(const char *event, uint64_t bin = ~uint64_t(0),
                 uint64_t a = 0, uint64_t b = 0, uint64_t fiber = 0) {
  if (nq_supply_only.load(std::memory_order_relaxed) &&
      !is_nq_supply_event(event)) return;
  if (round_gap_only.load(std::memory_order_relaxed) &&
      !is_round_gap_event(event)) {
    return;
  }
  if (!enabled.load(std::memory_order_relaxed)) return;
  writers.fetch_add(1, std::memory_order_acq_rel);
  if (enabled.load(std::memory_order_acquire)) {
    const auto ts = __rdtsc();
    const auto i = cursor.fetch_add(1, std::memory_order_relaxed);
    if (i < capacity) records[i] = Record{ts, fiber, bin, a, b, event};
  }
  writers.fetch_sub(1, std::memory_order_release);
}
// Per-OS-thread counters; no global atomic operation for every object.
// At most 127 allocations per used thread/bin remain at measurement end.
inline void allocated(uint64_t bin, uint64_t footprint, uint64_t payload) {
  if (nq_supply_only.load(std::memory_order_relaxed)) return;
  if (round_gap_only.load(std::memory_order_relaxed)) return;
  if (!enabled.load(std::memory_order_relaxed)) return;
  struct Batch { uint64_t count=0, footprint=0, payload=0, begin_tsc=0; };
  static thread_local Batch batches[128];
  if (bin >= 128) return;
  auto &x=batches[bin];
  if (x.count == 0) x.begin_tsc=__rdtsc();
  ++x.count; x.footprint+=footprint; x.payload+=payload;
  if (x.count == 128) {
    // For this event only, the fiber column carries the batch's first TSC.
    emit("object_alloc_batch", bin, x.footprint, x.payload, x.begin_tsc);
    x=Batch{};
  }
}
inline void start() {
    // Equal allocation/pre-touch in trace-on and trace-off control runs.
    records.reset(new Record[capacity]());
    ++stage_ordinal;
    const char *nq_supply = std::getenv("FARLIB_NQ_SUPPLY_TRACE");
    nq_supply_only.store(nq_supply != nullptr && std::strcmp(nq_supply, "1") == 0,
                         std::memory_order_relaxed);
    const char *round_gap = std::getenv("FARLIB_ROUND_GAP_TRACE");
    const bool phase_only = round_gap != nullptr && round_gap[0] == '1' &&
                            round_gap[1] == '\0';
    round_gap_only.store(phase_only, std::memory_order_relaxed);
    const char *configured_output = std::getenv("OBJECT_WAIT_TRACE");
    output = nullptr;
    if (!configured_output || !*configured_output) return;
    const char *reclaim_diag =
        std::getenv("FARLIB_INCLUSIVE_RECLAIM_DIAG");
    if (phase_only ||
        (reclaim_diag != nullptr && reclaim_diag[0] == '1' &&
         reclaim_diag[1] == '\0')) {
        const int written = std::snprintf(
            staged_output, sizeof(staged_output), "%s.stage%u.tsv",
            configured_output, stage_ordinal);
        if (written <= 0 || static_cast<size_t>(written) >=
                                sizeof(staged_output)) {
            return;
        }
        output = staged_output;
    } else {
        output = configured_output;
    }
  cursor.store(0);
  begin_tsc = __rdtsc();
  enabled.store(true, std::memory_order_release);
}
inline void stop_and_dump() {
  if (!output || !*output) return;
  enabled.store(false, std::memory_order_release);
  end_tsc = __rdtsc();
  while (writers.load(std::memory_order_acquire)) _mm_pause();
  auto *f = std::fopen(output, "w");
  if (!f) { std::perror("OBJECT_WAIT_TRACE"); return; }
  const uint64_t n = cursor.load();
  std::fprintf(f, "# stage_ordinal=%u begin_tsc=%llu end_tsc=%llu events=%llu dropped=%llu\n",
      stage_ordinal,
      (unsigned long long)begin_tsc, (unsigned long long)end_tsc,
      (unsigned long long)n, (unsigned long long)(n > capacity ? n-capacity : 0));
  std::fprintf(f, "tsc\tfiber\tevent\tbin\ta\tb\n");
  for (uint64_t i=0; i<n && i<capacity; ++i) {
    const auto &r=records[i];
    std::fprintf(f, "%llu\t%llu\t%s\t%llu\t%llu\t%llu\n",
      (unsigned long long)r.tsc, (unsigned long long)r.fiber, r.event,
      (unsigned long long)r.bin, (unsigned long long)r.a, (unsigned long long)r.b);
  }
  std::fclose(f);
}
}
