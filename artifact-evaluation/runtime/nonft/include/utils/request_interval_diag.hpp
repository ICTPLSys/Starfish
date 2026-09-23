#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

namespace FarLib::request_interval_diag {

constexpr size_t kMaxFibres = 128;
constexpr size_t kBucketCount = 8192;
constexpr size_t kRecordCapacity = 65536;
constexpr uint64_t kSamplePeriod = 4096;

struct Stamp {
    uint64_t tsc = 0;
    uint32_t aux = 0;
};

inline Stamp ordered_stamp() {
    unsigned aux = 0;
    _mm_lfence();
    const uint64_t tsc = __rdtscp(&aux);
    _mm_lfence();
    return {tsc, aux};
}

struct ThreadPoint {
    int32_t os_tid = -1;
    int32_t cpu = -1;
    uint64_t thread_cpu_ns = 0;
};

inline ThreadPoint thread_point() {
    timespec ts{};
    ThreadPoint point;
    point.os_tid = static_cast<int32_t>(::syscall(SYS_gettid));
    point.cpu = ::sched_getcpu();
    if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        point.thread_cpu_ns =
            static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
            static_cast<uint64_t>(ts.tv_nsec);
    }
    return point;
}

struct Sample {
    uint64_t sample_ordinal = 0;
    uint32_t stage_ordinal = 0;
    uint32_t formal_fibres = 0;
    uint32_t fibre_slot = 0;
    uint64_t fid = 0;
    uint64_t generation = 0;
    uint64_t current_entry = 0;
    uint64_t current_wr_id = 0;
    uint64_t next_entry = 0;
    uint64_t next_wr_id = 0;
    uint32_t current_prepost_waiters = 0;
    uint32_t next_prepost_waiters = 0;
    uint32_t current_post_attempts = 0;
    uint32_t next_post_attempts = 0;
    uint64_t cq_handler_fid = 0;
    uint32_t cq_handler_priority = 0;
    char cq_handler_name[40]{};
    int32_t cq_handler_os_tid = -1;
    int32_t cq_handler_cpu = -1;
    uint64_t cq_handler_thread_cpu_ns = 0;
    int32_t local_os_tid = -1;
    int32_t local_cpu = -1;
    uint64_t local_thread_cpu_ns = 0;
    int32_t requester_resume_os_tid = -1;
    int32_t requester_resume_cpu = -1;
    uint64_t requester_resume_thread_cpu_ns = 0;
    Stamp current_prepost;
    Stamp current_first_post_begin;
    Stamp current_accepted_post_begin;
    Stamp current_post_return;
    Stamp cq_reaped;
    Stamp local_cas_before;
    Stamp local_cas_after;
    Stamp callback_done;
    Stamp requester_resume;
    Stamp consumer_end;
    Stamp next_read_prepost;
    Stamp next_first_post_begin;
    Stamp next_accepted_post_begin;
    Stamp next_post_return;
    Stamp current_last_attempt_begin;
    Stamp next_last_attempt_begin;
    bool callback_complete = false;
    bool next_post_complete = false;
};

struct alignas(64) FibreSlot {
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> miss_ordinal{0};
    std::atomic<uint64_t> active_generation{0};
    std::atomic<bool> dropped{false};
    Sample sample;
};

struct alignas(64) Bucket {
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> key{0};
    std::atomic<bool> drop_requested{false};
    uint64_t generation = 0;
    uint32_t fibre_slot = 0;
};

struct Token {
    bool active;
    uint32_t fibre_slot;
    uint64_t generation;
    uint64_t wr_id;
};

inline std::array<FibreSlot, kMaxFibres> fibre_slots{};
inline std::array<Bucket, kBucketCount> buckets{};
inline std::array<Sample, kRecordCapacity> records{};
inline std::array<std::atomic<bool>, kRecordCapacity> records_ready{};
inline std::atomic<uint64_t> next_generation{1};
inline std::atomic<uint64_t> next_sample_ordinal{1};
inline std::atomic<uint64_t> record_count{0};
inline std::atomic<uint32_t> stage_ordinal{0};
inline std::atomic<uint32_t> formal_fibres{0};
inline std::atomic<bool> capture_active{false};
inline std::atomic<uint64_t> selected_count{0};
inline std::atomic<uint64_t> published_count{0};
inline std::atomic<uint64_t> completed_count{0};
inline std::atomic<uint64_t> bucket_collision_count{0};
inline std::atomic<uint64_t> bucket_lock_fail_count{0};
inline std::atomic<uint64_t> slot_lock_fail_count{0};
inline std::atomic<uint64_t> tag_mismatch_count{0};
inline std::atomic<uint64_t> dropped_count{0};
inline std::atomic<uint64_t> ring_overflow_count{0};

inline bool enabled() {
    static const bool value = [] {
        const char *text = std::getenv("FARLIB_REQUEST_INTERVAL_DIAG");
        return text != nullptr && text[0] == '1' && text[1] == '\0';
    }();
    return value;
}

inline size_t bucket_index(uint64_t wr_id) {
    uint64_t x = wr_id;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<size_t>(x) & (kBucketCount - 1);
}

inline bool try_lock(std::atomic_flag &lock) {
    return !lock.test_and_set(std::memory_order_acquire);
}

inline void unlock(std::atomic_flag &lock) {
    lock.clear(std::memory_order_release);
}

inline void begin_stage(size_t fibres) {
    const uint32_t ordinal =
        stage_ordinal.fetch_add(1, std::memory_order_relaxed) + 1;
    formal_fibres.store(static_cast<uint32_t>(fibres),
                        std::memory_order_relaxed);
    const bool capture = enabled() && ordinal == 2 && fibres == 48;
    capture_active.store(capture, std::memory_order_release);
    if (capture) {
        for (auto &slot : fibre_slots) {
            slot.miss_ordinal.store(0, std::memory_order_relaxed);
        }
    }
}

inline void end_stage() {
    capture_active.store(false, std::memory_order_release);
}

inline void reset_slot_locked(FibreSlot &slot) {
    slot.active_generation.store(0, std::memory_order_release);
    slot.dropped.store(false, std::memory_order_relaxed);
    slot.sample = {};
}

inline void finalize_locked(FibreSlot &slot) {
    auto &s = slot.sample;
    if (!s.callback_complete || !s.next_post_complete) return;
    const bool complete =
        s.current_prepost.tsc && s.current_first_post_begin.tsc &&
        s.current_accepted_post_begin.tsc && s.current_post_return.tsc &&
        s.cq_reaped.tsc && s.local_cas_before.tsc && s.local_cas_after.tsc &&
        s.callback_done.tsc && s.requester_resume.tsc && s.consumer_end.tsc &&
        s.next_read_prepost.tsc && s.next_first_post_begin.tsc &&
        s.next_accepted_post_begin.tsc && s.next_post_return.tsc;
    if (!complete || slot.dropped.load(std::memory_order_relaxed)) {
        dropped_count.fetch_add(1, std::memory_order_relaxed);
    } else {
        const uint64_t idx =
            record_count.fetch_add(1, std::memory_order_relaxed);
        if (idx < kRecordCapacity) {
            records[idx] = s;
            records_ready[idx].store(true, std::memory_order_release);
            completed_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            ring_overflow_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    reset_slot_locked(slot);
}

inline Token request_prepost(size_t fibre_slot, uint64_t fid,
                             uint64_t entry, uint64_t wr_id,
                             uint32_t waiters) {
    if (!capture_active.load(std::memory_order_acquire) ||
        fibre_slot >= kMaxFibres) {
        return {};
    }
    auto &slot = fibre_slots[fibre_slot];
    const uint64_t active =
        slot.active_generation.load(std::memory_order_acquire);
    if (active != 0) {
        if (slot.dropped.load(std::memory_order_relaxed)) return {};
        const Stamp stamp = ordered_stamp();
        if (!try_lock(slot.lock)) {
            slot.dropped.store(true, std::memory_order_relaxed);
            slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
        Token token{};
        if (slot.active_generation.load(std::memory_order_relaxed) == active &&
            slot.sample.fid == fid &&
            slot.sample.next_read_prepost.tsc == 0 &&
            !slot.dropped.load(std::memory_order_relaxed)) {
            slot.sample.next_entry = entry;
            slot.sample.next_wr_id = wr_id;
            slot.sample.next_prepost_waiters = waiters;
            slot.sample.next_read_prepost = stamp;
            token = {true, static_cast<uint32_t>(fibre_slot), active, wr_id};
        }
        unlock(slot.lock);
        return token;
    }

    const uint64_t ordinal =
        slot.miss_ordinal.fetch_add(1, std::memory_order_relaxed) + 1;
    if (ordinal % kSamplePeriod != 0) return {};
    const Stamp prepost = ordered_stamp();
    const uint64_t generation =
        next_generation.fetch_add(1, std::memory_order_relaxed);
    const uint64_t sample_ordinal =
        next_sample_ordinal.fetch_add(1, std::memory_order_relaxed);
    selected_count.fetch_add(1, std::memory_order_relaxed);

    auto &bucket = buckets[bucket_index(wr_id)];
    if (!try_lock(bucket.lock)) {
        bucket.drop_requested.store(true, std::memory_order_relaxed);
        bucket_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return {};
    }
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        unlock(bucket.lock);
        return {};
    }
    if (slot.active_generation.load(std::memory_order_relaxed) != 0 ||
        bucket.key.load(std::memory_order_relaxed) != 0) {
        bucket_collision_count.fetch_add(1, std::memory_order_relaxed);
        unlock(slot.lock);
        unlock(bucket.lock);
        return {};
    }
    slot.sample = {};
    slot.sample.sample_ordinal = sample_ordinal;
    slot.sample.stage_ordinal =
        stage_ordinal.load(std::memory_order_relaxed);
    slot.sample.formal_fibres =
        formal_fibres.load(std::memory_order_relaxed);
    slot.sample.fibre_slot = static_cast<uint32_t>(fibre_slot);
    slot.sample.fid = fid;
    slot.sample.generation = generation;
    slot.sample.current_entry = entry;
    slot.sample.current_wr_id = wr_id;
    slot.sample.current_prepost_waiters = waiters;
    slot.sample.current_prepost = prepost;
    slot.dropped.store(false, std::memory_order_relaxed);
    slot.active_generation.store(generation, std::memory_order_release);
    bucket.generation = generation;
    bucket.fibre_slot = static_cast<uint32_t>(fibre_slot);
    bucket.drop_requested.store(false, std::memory_order_relaxed);
    bucket.key.store(wr_id, std::memory_order_release);
    published_count.fetch_add(1, std::memory_order_relaxed);
    unlock(slot.lock);
    unlock(bucket.lock);
    return {true, static_cast<uint32_t>(fibre_slot), generation, wr_id};
}

inline thread_local Token post_context{};
inline thread_local Token completion_context{};

inline void set_post_context(Token token) { post_context = token; }
inline void clear_post_context() { post_context = {}; }
inline void set_completion_context(Token token) { completion_context = token; }
inline void clear_completion_context() { completion_context = {}; }

inline void post_attempt_begin() {
    const Token token = post_context;
    if (!token.active || token.fibre_slot >= kMaxFibres) return;
    const Stamp stamp = ordered_stamp();
    auto &slot = fibre_slots[token.fibre_slot];
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (slot.active_generation.load(std::memory_order_relaxed) !=
            token.generation ||
        slot.dropped.load(std::memory_order_relaxed)) {
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
        unlock(slot.lock);
        return;
    }
    if (slot.sample.current_wr_id == token.wr_id &&
        slot.sample.current_post_return.tsc == 0) {
        ++slot.sample.current_post_attempts;
        if (!slot.sample.current_first_post_begin.tsc)
            slot.sample.current_first_post_begin = stamp;
        slot.sample.current_last_attempt_begin = stamp;
    } else if (slot.sample.next_wr_id == token.wr_id &&
               slot.sample.next_post_return.tsc == 0) {
        ++slot.sample.next_post_attempts;
        if (!slot.sample.next_first_post_begin.tsc)
            slot.sample.next_first_post_begin = stamp;
        slot.sample.next_last_attempt_begin = stamp;
    } else {
        slot.dropped.store(true, std::memory_order_relaxed);
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
    }
    unlock(slot.lock);
}

inline void post_attempt_result(bool success) {
    const Token token = post_context;
    if (!token.active || token.fibre_slot >= kMaxFibres || !success) return;
    const Stamp stamp = ordered_stamp();
    auto &slot = fibre_slots[token.fibre_slot];
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (slot.active_generation.load(std::memory_order_relaxed) !=
            token.generation ||
        slot.dropped.load(std::memory_order_relaxed)) {
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
        unlock(slot.lock);
        return;
    }
    if (slot.sample.current_wr_id == token.wr_id &&
        slot.sample.current_post_return.tsc == 0) {
        slot.sample.current_accepted_post_begin =
            slot.sample.current_last_attempt_begin;
        slot.sample.current_post_return = stamp;
    } else if (slot.sample.next_wr_id == token.wr_id &&
               slot.sample.next_post_return.tsc == 0) {
        slot.sample.next_accepted_post_begin =
            slot.sample.next_last_attempt_begin;
        slot.sample.next_post_return = stamp;
        slot.sample.next_post_complete = true;
        finalize_locked(slot);
    } else {
        slot.dropped.store(true, std::memory_order_relaxed);
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
    }
    unlock(slot.lock);
}

inline bool tracked(uint64_t wr_id) {
    if (!enabled()) return false;
    auto &bucket = buckets[bucket_index(wr_id)];
    return bucket.key.load(std::memory_order_acquire) == wr_id;
}

inline Token cq_reaped(uint64_t wr_id, uint64_t handler_fid,
                       uint32_t handler_priority,
                       const char *handler_name) {
    if (!enabled()) return {};
    auto &bucket = buckets[bucket_index(wr_id)];
    if (bucket.key.load(std::memory_order_acquire) != wr_id) return {};
    if (!try_lock(bucket.lock)) {
        bucket.drop_requested.store(true, std::memory_order_relaxed);
        bucket_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return {};
    }
    if (bucket.key.load(std::memory_order_relaxed) != wr_id ||
        bucket.fibre_slot >= kMaxFibres) {
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
        unlock(bucket.lock);
        return {};
    }
    auto &slot = fibre_slots[bucket.fibre_slot];
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        unlock(bucket.lock);
        return {};
    }
    Token token{};
    if (slot.active_generation.load(std::memory_order_relaxed) ==
            bucket.generation &&
        slot.sample.current_wr_id == wr_id &&
        slot.sample.cq_reaped.tsc == 0) {
        if (bucket.drop_requested.exchange(false,
                                            std::memory_order_relaxed))
            slot.dropped.store(true, std::memory_order_relaxed);
        slot.sample.cq_reaped = ordered_stamp();
        slot.sample.cq_handler_fid = handler_fid;
        slot.sample.cq_handler_priority = handler_priority;
        std::snprintf(slot.sample.cq_handler_name,
                      sizeof(slot.sample.cq_handler_name), "%s",
                      handler_name == nullptr ? "" : handler_name);
        const auto point = thread_point();
        slot.sample.cq_handler_os_tid = point.os_tid;
        slot.sample.cq_handler_cpu = point.cpu;
        slot.sample.cq_handler_thread_cpu_ns = point.thread_cpu_ns;
        token = {true, bucket.fibre_slot, bucket.generation, wr_id};
    } else {
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
    }
    unlock(slot.lock);
    unlock(bucket.lock);
    return token;
}

inline void local_published(uint64_t wr_id, Stamp before, Stamp after) {
    const Token token = completion_context;
    if (!token.active || token.fibre_slot >= kMaxFibres ||
        token.wr_id != wr_id) return;
    auto &slot = fibre_slots[token.fibre_slot];
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (slot.active_generation.load(std::memory_order_relaxed) ==
            token.generation &&
        slot.sample.current_wr_id == wr_id &&
        slot.sample.local_cas_before.tsc == 0) {
        const auto point = thread_point();
        slot.sample.local_cas_before = before;
        slot.sample.local_cas_after = after;
        slot.sample.local_os_tid = point.os_tid;
        slot.sample.local_cpu = point.cpu;
        slot.sample.local_thread_cpu_ns = point.thread_cpu_ns;
    } else {
        slot.dropped.store(true, std::memory_order_relaxed);
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
    }
    unlock(slot.lock);
}

inline void callback_done(Token token) {
    if (!token.active || token.fibre_slot >= kMaxFibres) return;
    const Stamp stamp = ordered_stamp();
    auto &bucket = buckets[bucket_index(token.wr_id)];
    if (!try_lock(bucket.lock)) {
        bucket.drop_requested.store(true, std::memory_order_relaxed);
        bucket_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (bucket.key.load(std::memory_order_relaxed) != token.wr_id ||
        bucket.generation != token.generation ||
        bucket.fibre_slot != token.fibre_slot) {
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
        unlock(bucket.lock);
        return;
    }
    auto &slot = fibre_slots[token.fibre_slot];
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        unlock(bucket.lock);
        return;
    }
    if (slot.active_generation.load(std::memory_order_relaxed) ==
            token.generation &&
        slot.sample.current_wr_id == token.wr_id) {
        if (bucket.drop_requested.exchange(false,
                                            std::memory_order_relaxed))
            slot.dropped.store(true, std::memory_order_relaxed);
        slot.sample.callback_done = stamp;
        slot.sample.callback_complete = true;
        bucket.key.store(0, std::memory_order_release);
        finalize_locked(slot);
    } else {
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
    }
    unlock(slot.lock);
    unlock(bucket.lock);
}

inline void requester_resumed(size_t fibre_slot, uint64_t fid,
                              uint64_t entry) {
    if (!enabled() || fibre_slot >= kMaxFibres) return;
    auto &slot = fibre_slots[fibre_slot];
    const uint64_t generation =
        slot.active_generation.load(std::memory_order_acquire);
    if (!generation || slot.dropped.load(std::memory_order_relaxed)) return;
    const Stamp stamp = ordered_stamp();
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (slot.active_generation.load(std::memory_order_relaxed) == generation &&
        slot.sample.fid == fid && slot.sample.current_entry == entry &&
        slot.sample.requester_resume.tsc == 0) {
        const auto point = thread_point();
        slot.sample.requester_resume = stamp;
        slot.sample.requester_resume_os_tid = point.os_tid;
        slot.sample.requester_resume_cpu = point.cpu;
        slot.sample.requester_resume_thread_cpu_ns = point.thread_cpu_ns;
    } else {
        slot.dropped.store(true, std::memory_order_relaxed);
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
    }
    unlock(slot.lock);
}

inline void consumer_end(size_t fibre_slot, uint64_t fid) {
    if (!enabled() || fibre_slot >= kMaxFibres) return;
    auto &slot = fibre_slots[fibre_slot];
    const uint64_t generation =
        slot.active_generation.load(std::memory_order_acquire);
    if (!generation || slot.dropped.load(std::memory_order_relaxed)) return;
    const Stamp stamp = ordered_stamp();
    if (!try_lock(slot.lock)) {
        slot.dropped.store(true, std::memory_order_relaxed);
        slot_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (slot.active_generation.load(std::memory_order_relaxed) == generation &&
        slot.sample.fid == fid && slot.sample.requester_resume.tsc != 0 &&
        slot.sample.consumer_end.tsc == 0) {
        slot.sample.consumer_end = stamp;
    } else if (slot.sample.consumer_end.tsc == 0) {
        slot.dropped.store(true, std::memory_order_relaxed);
        tag_mismatch_count.fetch_add(1, std::memory_order_relaxed);
    }
    unlock(slot.lock);
}

inline void dump() {
    if (!enabled()) return;
    const char *path = std::getenv("FARLIB_REQUEST_INTERVAL_PATH");
    if (!path || !path[0]) return;
    FILE *file = std::fopen(path, "w");
    if (!file) {
        std::perror("FARLIB_REQUEST_INTERVAL_PATH");
        return;
    }
    std::fprintf(file,
        "sample_ordinal,stage_ordinal,formal_fibres,fibre_slot,fid,"
        "current_entry,current_wr_id,next_entry,next_wr_id,"
        "current_prepost_waiters,next_prepost_waiters,current_post_attempts,"
        "next_post_attempts,cq_handler_fid,cq_handler_priority,"
        "cq_handler_name,cq_handler_os_tid,cq_handler_cpu,"
        "cq_handler_thread_cpu_ns,local_os_tid,local_cpu,"
        "local_thread_cpu_ns,requester_resume_os_tid,"
        "requester_resume_cpu,requester_resume_thread_cpu_ns");
    const char *event_names[] = {
        "current_prepost", "current_first_post_begin",
        "current_accepted_post_begin", "current_post_return", "cq_reaped",
        "local_cas_before", "local_cas_after", "callback_done",
        "requester_resume", "consumer_end", "next_read_prepost",
        "next_first_post_begin", "next_accepted_post_begin",
        "next_post_return"};
    for (const char *name : event_names)
        std::fprintf(file, ",%s_tsc,%s_aux", name, name);
    std::fputc('\n', file);
    const uint64_t count = std::min<uint64_t>(
        record_count.load(std::memory_order_acquire), kRecordCapacity);
    uint64_t not_ready = 0;
    for (uint64_t i = 0; i < count; ++i) {
        if (!records_ready[i].load(std::memory_order_acquire)) {
            ++not_ready;
            continue;
        }
        const auto &s = records[i];
        std::fprintf(file,
            "%llu,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%u,%u,%u,%u,%llu,"
            "%u,%s,%d,%d,%llu,%d,%d,%llu,%d,%d,%llu",
            static_cast<unsigned long long>(s.sample_ordinal),
            s.stage_ordinal, s.formal_fibres, s.fibre_slot,
            static_cast<unsigned long long>(s.fid),
            static_cast<unsigned long long>(s.current_entry),
            static_cast<unsigned long long>(s.current_wr_id),
            static_cast<unsigned long long>(s.next_entry),
            static_cast<unsigned long long>(s.next_wr_id),
            s.current_prepost_waiters, s.next_prepost_waiters,
            s.current_post_attempts, s.next_post_attempts,
            static_cast<unsigned long long>(s.cq_handler_fid),
            s.cq_handler_priority, s.cq_handler_name,
            s.cq_handler_os_tid, s.cq_handler_cpu,
            static_cast<unsigned long long>(s.cq_handler_thread_cpu_ns),
            s.local_os_tid, s.local_cpu,
            static_cast<unsigned long long>(s.local_thread_cpu_ns),
            s.requester_resume_os_tid, s.requester_resume_cpu,
            static_cast<unsigned long long>(
                s.requester_resume_thread_cpu_ns));
        const Stamp stamps[] = {
            s.current_prepost, s.current_first_post_begin,
            s.current_accepted_post_begin, s.current_post_return,
            s.cq_reaped, s.local_cas_before, s.local_cas_after,
            s.callback_done, s.requester_resume, s.consumer_end,
            s.next_read_prepost, s.next_first_post_begin,
            s.next_accepted_post_begin, s.next_post_return};
        for (const auto stamp : stamps)
            std::fprintf(file, ",%llu,%u",
                         static_cast<unsigned long long>(stamp.tsc),
                         stamp.aux);
        std::fputc('\n', file);
    }
    std::fclose(file);
    uint64_t incomplete = 0;
    uint64_t active_buckets = 0;
    for (auto &slot : fibre_slots)
        if (slot.active_generation.load(std::memory_order_acquire))
            ++incomplete;
    for (auto &bucket : buckets)
        if (bucket.key.load(std::memory_order_acquire))
            ++active_buckets;
    std::fprintf(stderr,
        "request_interval_diag path=%s period=%llu selected=%llu "
        "published=%llu completed=%llu bucket_collisions=%llu "
        "bucket_lock_fail=%llu slot_lock_fail=%llu tag_mismatch=%llu "
        "dropped=%llu incomplete=%llu records_not_ready=%llu "
        "ring_overflow=%llu active_buckets=%llu\n",
        path, static_cast<unsigned long long>(kSamplePeriod),
        static_cast<unsigned long long>(selected_count.load()),
        static_cast<unsigned long long>(published_count.load()),
        static_cast<unsigned long long>(completed_count.load()),
        static_cast<unsigned long long>(bucket_collision_count.load()),
        static_cast<unsigned long long>(bucket_lock_fail_count.load()),
        static_cast<unsigned long long>(slot_lock_fail_count.load()),
        static_cast<unsigned long long>(tag_mismatch_count.load()),
        static_cast<unsigned long long>(dropped_count.load()),
        static_cast<unsigned long long>(incomplete),
        static_cast<unsigned long long>(not_ready),
        static_cast<unsigned long long>(ring_overflow_count.load()),
        static_cast<unsigned long long>(active_buckets));
}

}  // namespace FarLib::request_interval_diag
