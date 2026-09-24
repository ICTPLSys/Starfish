#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <x86intrin.h>
#include "utils/wc_object_diag.hpp"

namespace FarLib::read_supply_timeline {

constexpr size_t kMaxClients = 128;
constexpr size_t kMaxQpsPerClient = 4;
constexpr uint64_t kReadCountMask = 0xffffffffULL;
constexpr uint64_t kAcceptedIncrement = 1ULL << 32;
constexpr uint64_t kPollPublishBatch = 64;
constexpr uint64_t kLifecycleSamplePeriod = 4096;
constexpr size_t kLifecycleTableSize = 8192;
constexpr size_t kLifecycleRecordCapacity = 65536;

// A single load obtains an internally coherent accepted/reaped pair.  This is
// intentionally per Client/QP so the hot updates do not share a global line.
// Both halves are guarded at report time; this experiment stays far below 2^32.
struct alignas(64) Slot {
    std::atomic<uint64_t> accepted_reaped{0};
    std::atomic<uint64_t> write_posts{0};
    std::atomic<uint64_t> cq_empty_polls{0};
    std::atomic<uint64_t> cq_nonempty_polls{0};
    std::atomic<uint64_t> cq_read_wcs{0};
    std::atomic<uint64_t> cq_write_wcs{0};
    // One sparse lifecycle sample may be active per Client/QP.
    std::atomic<uint32_t> lifecycle_table_index{0};
};

inline std::array<Slot, kMaxClients * kMaxQpsPerClient> slots{};

enum LifecycleState : uint32_t {
    LifecycleEmpty = 0,
    LifecycleInitializing = 1,
    LifecycleArmed = 2,
    LifecycleReaped = 3,
    LifecycleLocal = 4,
    LifecycleAborted = 5,
};

struct alignas(64) LifecycleEntry {
    std::atomic<uint32_t> state{LifecycleEmpty};
    uint32_t client_idx = 0;
    uint32_t qp_idx = 0;
    uint64_t generation = 0;
    uint64_t wr_id = 0;
    uint64_t accepted_tsc = 0;
    uint64_t reaped_tsc = 0;
    uint64_t local_tsc = 0;
    std::atomic<uint64_t> first_poll_tsc{0};
    std::atomic<uint64_t> last_empty_poll_tsc{0};
    std::atomic<uint32_t> empty_polls_before_reap{0};
};

struct LifecycleRecord {
    uint64_t generation = 0;
    uint64_t wr_id = 0;
    uint64_t accepted_tsc = 0;
    uint64_t reaped_tsc = 0;
    uint64_t local_tsc = 0;
    uint64_t done_tsc = 0;
    uint64_t first_poll_tsc = 0;
    uint64_t last_empty_poll_tsc = 0;
    uint32_t client_idx = 0;
    uint32_t qp_idx = 0;
    uint32_t empty_polls_before_reap = 0;
};

inline std::array<LifecycleEntry, kLifecycleTableSize> lifecycle_table{};
inline std::array<LifecycleRecord, kLifecycleRecordCapacity> lifecycle_records{};
inline std::atomic<uint64_t> lifecycle_next_generation{1};
inline std::atomic<uint64_t> lifecycle_record_count{0};
inline std::atomic<uint64_t> lifecycle_selected{0};
inline std::atomic<uint64_t> lifecycle_claimed{0};
inline std::atomic<uint64_t> lifecycle_collisions{0};
inline std::atomic<uint64_t> lifecycle_early_reap{0};
inline std::atomic<uint64_t> lifecycle_reaped_matches{0};
inline std::atomic<uint64_t> lifecycle_local_matches{0};
inline std::atomic<uint64_t> lifecycle_completed{0};
inline std::atomic<uint64_t> lifecycle_missing_reap{0};
inline std::atomic<uint64_t> lifecycle_missing_local{0};
inline std::atomic<uint64_t> lifecycle_out_of_order{0};
inline std::atomic<uint64_t> lifecycle_ring_overflow{0};

inline size_t lifecycle_index(uint64_t wr_id) {
    uint64_t x = wr_id;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<size_t>(x) & (kLifecycleTableSize - 1);
}

inline bool enabled() {
    static const bool value = [] {
        const char *path = std::getenv("FARLIB_READ_SUPPLY_TIMELINE_PATH");
        return path != nullptr && path[0] != '\0';
    }();
    return value;
}

inline bool nowait_gap_enabled() {
    static const bool value = [] {
        const char *text = std::getenv("FARLIB_NOWAIT_GAP_DIAG");
        return text != nullptr && text[0] == '1' && text[1] == '\0';
    }();
    return value;
}

inline size_t slot_index(size_t client_idx, size_t qp_idx) {
    if (client_idx >= kMaxClients || qp_idx >= kMaxQpsPerClient) {
        std::abort();
    }
    return client_idx * kMaxQpsPerClient + qp_idx;
}

inline void record_read_accepted(size_t client_idx, size_t qp_idx,
                                 uint64_t wr_id, uint32_t count = 1) {
    if (!enabled()) return;
    const uint64_t old =
        slots[slot_index(client_idx, qp_idx)].accepted_reaped.fetch_add(
        static_cast<uint64_t>(count) << 32, std::memory_order_relaxed);
    if (!nowait_gap_enabled()) return;
    // The fixed experiment is READ-B1, so one accepted ordinal selects one WR.
    if (count != 1) return;
    const uint32_t ordinal = static_cast<uint32_t>(old >> 32) + 1;
    if (ordinal % kLifecycleSamplePeriod != 0) return;
    lifecycle_selected.fetch_add(1, std::memory_order_relaxed);

    auto &entry = lifecycle_table[lifecycle_index(wr_id)];
    uint32_t expected = LifecycleEmpty;
    if (!entry.state.compare_exchange_strong(
            expected, LifecycleInitializing, std::memory_order_acquire,
            std::memory_order_relaxed)) {
        lifecycle_collisions.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    entry.client_idx = static_cast<uint32_t>(client_idx);
    entry.qp_idx = static_cast<uint32_t>(qp_idx);
    entry.generation = lifecycle_next_generation.fetch_add(
        1, std::memory_order_relaxed);
    entry.wr_id = wr_id;
    entry.accepted_tsc = __rdtsc();
    entry.reaped_tsc = 0;
    entry.local_tsc = 0;
    entry.first_poll_tsc.store(0, std::memory_order_relaxed);
    entry.last_empty_poll_tsc.store(0, std::memory_order_relaxed);
    entry.empty_polls_before_reap.store(0, std::memory_order_relaxed);
    expected = LifecycleInitializing;
    if (entry.state.compare_exchange_strong(
            expected, LifecycleArmed, std::memory_order_release,
            std::memory_order_acquire)) {
        auto &slot = slots[slot_index(client_idx, qp_idx)];
        uint32_t no_sample = 0;
        const uint32_t table_index =
            static_cast<uint32_t>(lifecycle_index(wr_id) + 1);
        if (!slot.lifecycle_table_index.compare_exchange_strong(
                no_sample, table_index, std::memory_order_release,
                std::memory_order_relaxed)) {
            lifecycle_collisions.fetch_add(1, std::memory_order_relaxed);
            entry.state.store(LifecycleEmpty, std::memory_order_release);
            return;
        }
        lifecycle_claimed.fetch_add(1, std::memory_order_relaxed);
        if (entry.state.load(std::memory_order_acquire) != LifecycleArmed) {
            uint32_t encoded = table_index;
            (void)slot.lifecycle_table_index.compare_exchange_strong(
                encoded, 0, std::memory_order_release,
                std::memory_order_relaxed);
        }
        return;
    }
    if (expected != LifecycleAborted) {
        lifecycle_out_of_order.fetch_add(1, std::memory_order_relaxed);
    }
    entry.state.store(LifecycleEmpty, std::memory_order_release);
}

inline void record_lifecycle_poll(size_t client_idx, size_t qp_idx,
                                  size_t completion_count) {
    if (!nowait_gap_enabled()) return;
    auto &slot = slots[slot_index(client_idx, qp_idx)];
    const uint32_t encoded =
        slot.lifecycle_table_index.load(std::memory_order_acquire);
    if (encoded == 0 || encoded > kLifecycleTableSize) return;
    auto &entry = lifecycle_table[encoded - 1];
    if (entry.state.load(std::memory_order_acquire) != LifecycleArmed ||
        entry.client_idx != client_idx || entry.qp_idx != qp_idx) {
        return;
    }
    const bool need_first =
        entry.first_poll_tsc.load(std::memory_order_relaxed) == 0;
    if (!need_first && completion_count != 0) return;
    const uint64_t now = __rdtsc();
    if (need_first) {
        uint64_t empty = 0;
        (void)entry.first_poll_tsc.compare_exchange_strong(
            empty, now, std::memory_order_relaxed,
            std::memory_order_relaxed);
    }
    if (completion_count == 0) {
        entry.empty_polls_before_reap.fetch_add(1,
                                                std::memory_order_relaxed);
        entry.last_empty_poll_tsc.store(now, std::memory_order_relaxed);
    }
}

inline void record_lifecycle_reaped(uint64_t wr_id) {
    if (!nowait_gap_enabled()) return;
    auto &entry = lifecycle_table[lifecycle_index(wr_id)];
    uint32_t state = entry.state.load(std::memory_order_acquire);
    if (state == LifecycleInitializing) {
        uint32_t expected = LifecycleInitializing;
        if (entry.state.compare_exchange_strong(
                expected, LifecycleAborted, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            lifecycle_early_reap.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    if (state != LifecycleArmed || entry.wr_id != wr_id) return;
    entry.reaped_tsc = __rdtsc();
    uint32_t expected = LifecycleArmed;
    if (entry.state.compare_exchange_strong(
            expected, LifecycleReaped, std::memory_order_release,
            std::memory_order_relaxed)) {
        lifecycle_reaped_matches.fetch_add(1, std::memory_order_relaxed);
    } else {
        lifecycle_out_of_order.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void record_lifecycle_local(uint64_t wr_id) {
    if (!nowait_gap_enabled()) return;
    auto &entry = lifecycle_table[lifecycle_index(wr_id)];
    const uint32_t state = entry.state.load(std::memory_order_acquire);
    if (state != LifecycleReaped || entry.wr_id != wr_id) return;
    entry.local_tsc = __rdtsc();
    uint32_t expected = LifecycleReaped;
    if (entry.state.compare_exchange_strong(
            expected, LifecycleLocal, std::memory_order_release,
            std::memory_order_relaxed)) {
        lifecycle_local_matches.fetch_add(1, std::memory_order_relaxed);
    } else {
        lifecycle_out_of_order.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void clear_lifecycle_slot(const LifecycleEntry &entry, uint64_t wr_id) {
    auto &slot = slots[slot_index(entry.client_idx, entry.qp_idx)];
    uint32_t encoded = static_cast<uint32_t>(lifecycle_index(wr_id) + 1);
    (void)slot.lifecycle_table_index.compare_exchange_strong(
        encoded, 0, std::memory_order_release, std::memory_order_relaxed);
}

inline void record_lifecycle_done(uint64_t wr_id) {
    if (!nowait_gap_enabled()) return;
    auto &entry = lifecycle_table[lifecycle_index(wr_id)];
    const uint32_t state = entry.state.load(std::memory_order_acquire);
    if (entry.wr_id != wr_id) return;
    if (state != LifecycleLocal) {
        if (state == LifecycleArmed) {
            lifecycle_missing_reap.fetch_add(1, std::memory_order_relaxed);
        } else if (state == LifecycleReaped) {
            lifecycle_missing_local.fetch_add(1, std::memory_order_relaxed);
        } else {
            return;
        }
        clear_lifecycle_slot(entry, wr_id);
        entry.state.store(LifecycleEmpty, std::memory_order_release);
        return;
    }

    const uint64_t done_tsc = __rdtsc();
    if (!(entry.accepted_tsc <= entry.reaped_tsc &&
          entry.reaped_tsc <= entry.local_tsc &&
          entry.local_tsc <= done_tsc)) {
        lifecycle_out_of_order.fetch_add(1, std::memory_order_relaxed);
        clear_lifecycle_slot(entry, wr_id);
        entry.state.store(LifecycleEmpty, std::memory_order_release);
        return;
    }
    const uint64_t record_idx = lifecycle_record_count.fetch_add(
        1, std::memory_order_relaxed);
    if (record_idx < kLifecycleRecordCapacity) {
        auto &record = lifecycle_records[record_idx];
        record.generation = entry.generation;
        record.wr_id = entry.wr_id;
        record.accepted_tsc = entry.accepted_tsc;
        record.reaped_tsc = entry.reaped_tsc;
        record.local_tsc = entry.local_tsc;
        record.done_tsc = done_tsc;
        record.first_poll_tsc =
            entry.first_poll_tsc.load(std::memory_order_relaxed);
        record.last_empty_poll_tsc =
            entry.last_empty_poll_tsc.load(std::memory_order_relaxed);
        record.client_idx = entry.client_idx;
        record.qp_idx = entry.qp_idx;
        record.empty_polls_before_reap =
            entry.empty_polls_before_reap.load(std::memory_order_relaxed);
        lifecycle_completed.fetch_add(1, std::memory_order_relaxed);
    } else {
        lifecycle_ring_overflow.fetch_add(1, std::memory_order_relaxed);
    }
    clear_lifecycle_slot(entry, wr_id);
    entry.state.store(LifecycleEmpty, std::memory_order_release);
}

struct PendingPolls {
    size_t owner = slots.size();
    uint64_t polls = 0;
    uint64_t empty = 0;
    uint64_t nonempty = 0;
    uint64_t read_wcs = 0;
    uint64_t write_wcs = 0;
};

inline thread_local PendingPolls pending_polls;

inline void publish_pending_polls(PendingPolls &pending) {
    if (pending.owner >= slots.size()) return;
    auto &slot = slots[pending.owner];
    if (pending.empty != 0) {
        slot.cq_empty_polls.fetch_add(pending.empty,
                                      std::memory_order_relaxed);
    }
    if (pending.nonempty != 0) {
        slot.cq_nonempty_polls.fetch_add(pending.nonempty,
                                         std::memory_order_relaxed);
    }
    if (pending.read_wcs != 0) {
        slot.cq_read_wcs.fetch_add(pending.read_wcs,
                                   std::memory_order_relaxed);
    }
    if (pending.write_wcs != 0) {
        slot.cq_write_wcs.fetch_add(pending.write_wcs,
                                    std::memory_order_relaxed);
    }
    pending.polls = 0;
    pending.empty = 0;
    pending.nonempty = 0;
    pending.read_wcs = 0;
    pending.write_wcs = 0;
}

// Called immediately after ibv_poll_cq and before completion callbacks.  READ
// WCs update the coherent accepted/reaped word immediately; high-rate poll
// shape counters publish only once per 64 polls to limit observer overhead.
inline void record_cq_result(size_t client_idx, size_t qp_idx,
                             const ibv_wc *wc, size_t count) {
    if (!enabled()) return;
    record_lifecycle_poll(client_idx, qp_idx, count);
    const size_t owner = slot_index(client_idx, qp_idx);
    uint32_t read_wcs = 0;
    uint32_t write_wcs = 0;
    for (size_t i = 0; i < count; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) continue;
        if (wc[i].opcode == IBV_WC_RDMA_READ) {
            ++read_wcs;
            ::FarLib::wc_object_diag::cq_reaped(wc[i].wr_id);
            record_lifecycle_reaped(wc[i].wr_id);
        }
        if (wc[i].opcode == IBV_WC_RDMA_WRITE) ++write_wcs;
    }
    if (read_wcs != 0) {
        slots[owner].accepted_reaped.fetch_add(read_wcs,
                                               std::memory_order_relaxed);
    }

    auto &pending = pending_polls;
    if (pending.owner != owner) {
        publish_pending_polls(pending);
        pending.owner = owner;
    }
    if (count == 0) {
        ++pending.empty;
    } else {
        ++pending.nonempty;
    }
    pending.read_wcs += read_wcs;
    pending.write_wcs += write_wcs;
    if (++pending.polls == kPollPublishBatch) publish_pending_polls(pending);
}

inline void record_write_posts(size_t client_idx, size_t qp_idx,
                               uint64_t count) {
    if (!enabled() || count == 0) return;
    slots[slot_index(client_idx, qp_idx)].write_posts.fetch_add(
        count, std::memory_order_relaxed);
}

struct SlotSnapshot {
    uint32_t accepted = 0;
    uint32_t reaped = 0;
    int64_t pending = 0;
    uint64_t write_posts = 0;
    uint64_t cq_empty_polls = 0;
    uint64_t cq_nonempty_polls = 0;
    uint64_t cq_read_wcs = 0;
    uint64_t cq_write_wcs = 0;
};

inline SlotSnapshot snapshot_slot(size_t client_idx, size_t qp_idx) {
    const auto &slot = slots[slot_index(client_idx, qp_idx)];
    SlotSnapshot result;
    const uint64_t packed =
        slot.accepted_reaped.load(std::memory_order_relaxed);
    result.accepted = static_cast<uint32_t>(packed >> 32);
    result.reaped = static_cast<uint32_t>(packed & kReadCountMask);
    result.pending = static_cast<int64_t>(result.accepted) -
                     static_cast<int64_t>(result.reaped);
    result.write_posts = slot.write_posts.load(std::memory_order_relaxed);
    result.cq_empty_polls =
        slot.cq_empty_polls.load(std::memory_order_relaxed);
    result.cq_nonempty_polls =
        slot.cq_nonempty_polls.load(std::memory_order_relaxed);
    result.cq_read_wcs = slot.cq_read_wcs.load(std::memory_order_relaxed);
    result.cq_write_wcs = slot.cq_write_wcs.load(std::memory_order_relaxed);
    return result;
}

}  // namespace FarLib::read_supply_timeline
