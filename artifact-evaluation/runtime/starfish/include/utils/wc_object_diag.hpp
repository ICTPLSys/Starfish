#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
#include <x86intrin.h>

namespace FarLib::wc_object_diag {
constexpr size_t kCapacity = 16384;
constexpr size_t kLookupSize = 16384;
constexpr uint64_t kSampleMask = 4095;
inline bool configured() {
    static const bool value = [] {
        const char *p = std::getenv("FARLIB_WC_PROFILE_PATH");
        return p && *p;
    }();
    return value;
}
inline bool object_enabled() {
    static const bool value = [] {
        const char *p = std::getenv("FARLIB_WC_OBJECT_PROFILE");
        return p && p[0] == '1' && p[1] == '\0';
    }();
    return value;
}
inline uint64_t stamp() {
    unsigned aux;
    _mm_lfence();
    const uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}
inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct alignas(64) Sample {
    uint64_t index = 0, batch = 0, object = 0, wr_id = 0;
    uint32_t tid_start = 0, tid_end = 0;
    std::atomic<uint64_t> access_begin{0}, alloc_begin{0}, alloc_end{0};
    std::atomic<uint64_t> alloc_ticks{0}, alloc_calls{0};
    std::atomic<uint64_t> prepost{0}, submit_begin{0}, submit_return{0};
    std::atomic<uint64_t> cq_reaped{0}, access_end{0}, copy_end{0};
    std::atomic<uint64_t> post_attempts{0};
    bool armed = false;
};
inline std::array<Sample, kCapacity> samples{};
inline std::array<std::atomic<Sample *>, kLookupSize> object_slots{}, wr_slots{};
inline std::atomic<uint64_t> selected{0}, object_collisions{0}, wr_collisions{0}, overflow{0};
inline std::atomic<bool> active{false};
inline uint64_t calibration_begin_ns = 0, calibration_end_ns = 0;
inline uint64_t calibration_begin_tsc = 0, calibration_end_tsc = 0;
inline uint64_t mono_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void begin_capture() {
    if (object_enabled()) {
        calibration_begin_ns = mono_ns();
        calibration_begin_tsc = stamp();
    }
    active.store(object_enabled(), std::memory_order_release);
}
inline void end_capture() {
    active.store(false, std::memory_order_release);
    if (object_enabled()) {
        calibration_end_tsc = stamp();
        calibration_end_ns = mono_ns();
    }
}
inline size_t slot(uint64_t key) { return mix(key) & (kLookupSize - 1); }
inline Sample *find_object(uint64_t object) {
    if (!object_enabled() || !active.load(std::memory_order_relaxed)) return nullptr;
    Sample *p = object_slots[slot(object)].load(std::memory_order_acquire);
    return p && p->object == object ? p : nullptr;
}
inline Sample *begin(uint64_t index, uint64_t batch, uint64_t object) {
    if (!object_enabled() || (mix(index ^ 0x92731ca5ULL) & kSampleMask) != 0) return nullptr;
    const uint64_t id = selected.fetch_add(1, std::memory_order_relaxed);
    if (id >= kCapacity) { overflow.fetch_add(1); return nullptr; }
    Sample *p = &samples[id];
    p->index = index; p->batch = batch; p->object = object;
    p->tid_start = static_cast<uint32_t>(::syscall(SYS_gettid));
    Sample *empty = nullptr;
    if (!object_slots[slot(object)].compare_exchange_strong(empty, p, std::memory_order_release)) {
        object_collisions.fetch_add(1); return nullptr;
    }
    p->armed = true;
    p->access_begin.store(stamp(), std::memory_order_relaxed);
    return p;
}
inline void allocation_done(Sample *p, uint64_t begin_tsc) {
    if (!p) return;
    const uint64_t now = stamp();
    uint64_t zero = 0;
    (void)p->alloc_begin.compare_exchange_strong(zero, begin_tsc);
    p->alloc_end.store(now);
    p->alloc_ticks.fetch_add(now - begin_tsc);
    p->alloc_calls.fetch_add(1);
}
inline void prepost(uint64_t object, uint64_t wr_id) {
    Sample *p = find_object(object);
    if (!p) return;
    p->wr_id = wr_id;
    p->prepost.store(stamp());
    Sample *empty = nullptr;
    if (!wr_slots[slot(wr_id)].compare_exchange_strong(empty, p, std::memory_order_release) && empty != p)
        wr_collisions.fetch_add(1);
}
inline void cq_reaped(uint64_t wr_id) {
    if (!object_enabled() || !active.load(std::memory_order_relaxed)) return;
    Sample *p = wr_slots[slot(wr_id)].load(std::memory_order_acquire);
    if (p && p->wr_id == wr_id) p->cq_reaped.store(stamp(), std::memory_order_release);
}
inline void access_done(Sample *p) {
    if (p) p->access_end.store(stamp());
}
inline void finish(Sample *p) {
    if (!p) return;
    p->copy_end.store(stamp());
    p->tid_end = static_cast<uint32_t>(::syscall(SYS_gettid));
    Sample *expected = p;
    (void)object_slots[slot(p->object)].compare_exchange_strong(expected, nullptr);
    if (p->wr_id) {
        expected = p;
        (void)wr_slots[slot(p->wr_id)].compare_exchange_strong(expected, nullptr);
    }
}
struct Window { const char *name; uint64_t ordinal, begin_ns, end_ns; };
inline std::vector<Window> windows;
inline void window(const char *name, uint64_t ordinal,
                   std::chrono::steady_clock::time_point begin,
                   std::chrono::steady_clock::time_point end) {
    if (!configured()) return;
    windows.push_back({name, ordinal,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch()).count()),
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count())});
}
inline void dump() {
    if (!configured()) return;
    const std::string prefix(std::getenv("FARLIB_WC_PROFILE_PATH"));
    FILE *f = std::fopen((prefix + ".windows.csv").c_str(), "w");
    if (!f) std::abort();
    std::fprintf(f, "name,ordinal,begin_ns,end_ns\n");
    for (const auto &w : windows)
        std::fprintf(f, "%s,%lu,%lu,%lu\n", w.name, w.ordinal, w.begin_ns, w.end_ns);
    std::fclose(f);
    if (!object_enabled()) return;
    f = std::fopen((prefix + ".objects.csv").c_str(), "w");
    if (!f) std::abort();
    std::fprintf(f, "index,batch,object,wr_id,tid_start,tid_end,access_begin,alloc_begin,alloc_end,alloc_ticks,alloc_calls,prepost,submit_begin,submit_return,cq_reaped,access_end,copy_end,post_attempts\n");
    const uint64_t n = std::min<uint64_t>(selected.load(), kCapacity);
    for (size_t i = 0; i < n; ++i) {
        const auto &s = samples[i];
        if (!s.armed) continue;
        std::fprintf(f, "%lu,%lu,%lu,%lu,%u,%u,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n",
            s.index, s.batch, s.object, s.wr_id, s.tid_start, s.tid_end,
            s.access_begin.load(), s.alloc_begin.load(), s.alloc_end.load(), s.alloc_ticks.load(),
            s.alloc_calls.load(), s.prepost.load(), s.submit_begin.load(), s.submit_return.load(),
            s.cq_reaped.load(), s.access_end.load(), s.copy_end.load(), s.post_attempts.load());
    }
    std::fclose(f);
    const double hz = double(calibration_end_tsc - calibration_begin_tsc) /
        double(calibration_end_ns - calibration_begin_ns) * 1e9;
    std::fprintf(stderr, "wc_object_profile selected=%lu object_collisions=%lu wr_collisions=%lu overflow=%lu sample_period=4096 calibration_hz=%.6f calibration_begin_ns=%lu calibration_end_ns=%lu calibration_begin_tsc=%lu calibration_end_tsc=%lu\n",
        selected.load(), object_collisions.load(), wr_collisions.load(), overflow.load(), hz,
        calibration_begin_ns, calibration_end_ns, calibration_begin_tsc, calibration_end_tsc);
}
} // namespace FarLib::wc_object_diag
