#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
inline std::atomic<uint64_t> cursor{0}, writers{0};
inline uint64_t begin_tsc = 0, end_tsc = 0;
inline const char *output = nullptr;
inline uint32_t stage_ordinal = 0;

inline void emit(const char *event, uint64_t bin = ~uint64_t(0),
                 uint64_t a = 0, uint64_t b = 0, uint64_t fiber = 0) {
  if (!enabled.load(std::memory_order_relaxed)) return;
  writers.fetch_add(1, std::memory_order_acq_rel);
  if (enabled.load(std::memory_order_acquire)) {
    const auto ts = __rdtsc();
    const auto i = cursor.fetch_add(1, std::memory_order_relaxed);
    if (i < capacity) records[i] = Record{ts, fiber, bin, a, b, event};
  }
  writers.fetch_sub(1, std::memory_order_release);
}
inline void start() {
    ++stage_ordinal;
    const char *configured_output = std::getenv("OBJECT_WAIT_TRACE");
    output = nullptr;
    if (!configured_output || !*configured_output) return;
    // Allocate the optional trace buffer only when an output was requested.
    records.reset(new Record[capacity]());
    output = configured_output;
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
