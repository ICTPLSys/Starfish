#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace FarLib::alloc_reclaim_rate_diag {

inline bool enabled() {
    static const bool value = [] {
        const char *text = std::getenv("FARLIB_ALLOC_RECLAIM_RATE_DIAG");
        return text != nullptr && std::strcmp(text, "1") == 0;
    }();
    return value;
}

enum class ReclaimKind : uint8_t { Mark, Evict, GC };

struct alignas(64) Shard {
    std::atomic<uint64_t> alloc_slots{0};
    std::atomic<uint64_t> alloc_slot_bytes{0};
    std::atomic<uint64_t> mark_slots{0};
    std::atomic<uint64_t> mark_slot_bytes{0};
    std::atomic<uint64_t> evict_slots{0};
    std::atomic<uint64_t> evict_slot_bytes{0};
    std::atomic<uint64_t> gc_slots{0};
    std::atomic<uint64_t> gc_slot_bytes{0};
    std::atomic<uint64_t> published_slots{0};
    std::atomic<uint64_t> published_slot_bytes{0};
};

inline constexpr size_t kMaxShards = 128;
inline std::array<Shard, kMaxShards> shards;
inline std::atomic_size_t registered_shards{0};
inline std::atomic<uint64_t> dropped_records{0};
inline thread_local Shard *local_shard = nullptr;

inline Shard *get_local_shard() {
    if (!enabled()) return nullptr;
    if (local_shard != nullptr) return local_shard;
    const size_t index =
        registered_shards.fetch_add(1, std::memory_order_relaxed);
    if (index >= kMaxShards) {
        dropped_records.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    local_shard = &shards[index];
    return local_shard;
}

inline void record_allocation(size_t slot_bytes) {
    Shard *shard = get_local_shard();
    if (shard == nullptr) return;
    shard->alloc_slots.fetch_add(1, std::memory_order_relaxed);
    shard->alloc_slot_bytes.fetch_add(slot_bytes, std::memory_order_relaxed);
}

inline void record_reclaim(ReclaimKind kind, size_t slots,
                           size_t slot_bytes, bool published) {
    if (slots == 0) return;
    Shard *shard = get_local_shard();
    if (shard == nullptr) return;
    switch (kind) {
    case ReclaimKind::Mark:
        shard->mark_slots.fetch_add(slots, std::memory_order_relaxed);
        shard->mark_slot_bytes.fetch_add(slot_bytes,
                                         std::memory_order_relaxed);
        break;
    case ReclaimKind::Evict:
        shard->evict_slots.fetch_add(slots, std::memory_order_relaxed);
        shard->evict_slot_bytes.fetch_add(slot_bytes,
                                          std::memory_order_relaxed);
        break;
    case ReclaimKind::GC:
        shard->gc_slots.fetch_add(slots, std::memory_order_relaxed);
        shard->gc_slot_bytes.fetch_add(slot_bytes, std::memory_order_relaxed);
        break;
    }
    if (published) {
        shard->published_slots.fetch_add(slots, std::memory_order_relaxed);
        shard->published_slot_bytes.fetch_add(slot_bytes,
                                              std::memory_order_relaxed);
    }
}

struct Snapshot {
    uint64_t shard_count = 0;
    uint64_t dropped = 0;
    uint64_t alloc_slots = 0;
    uint64_t alloc_slot_bytes = 0;
    uint64_t mark_slots = 0;
    uint64_t mark_slot_bytes = 0;
    uint64_t evict_slots = 0;
    uint64_t evict_slot_bytes = 0;
    uint64_t gc_slots = 0;
    uint64_t gc_slot_bytes = 0;
    uint64_t published_slots = 0;
    uint64_t published_slot_bytes = 0;
};

inline Snapshot snapshot() {
    Snapshot result;
    result.shard_count = std::min(registered_shards.load(
                                      std::memory_order_acquire),
                                  kMaxShards);
    result.dropped = dropped_records.load(std::memory_order_relaxed);
    for (size_t i = 0; i < result.shard_count; ++i) {
        const Shard &shard = shards[i];
        result.alloc_slots +=
            shard.alloc_slots.load(std::memory_order_relaxed);
        result.alloc_slot_bytes +=
            shard.alloc_slot_bytes.load(std::memory_order_relaxed);
        result.mark_slots +=
            shard.mark_slots.load(std::memory_order_relaxed);
        result.mark_slot_bytes +=
            shard.mark_slot_bytes.load(std::memory_order_relaxed);
        result.evict_slots +=
            shard.evict_slots.load(std::memory_order_relaxed);
        result.evict_slot_bytes +=
            shard.evict_slot_bytes.load(std::memory_order_relaxed);
        result.gc_slots += shard.gc_slots.load(std::memory_order_relaxed);
        result.gc_slot_bytes +=
            shard.gc_slot_bytes.load(std::memory_order_relaxed);
        result.published_slots +=
            shard.published_slots.load(std::memory_order_relaxed);
        result.published_slot_bytes +=
            shard.published_slot_bytes.load(std::memory_order_relaxed);
    }
    return result;
}

}  // namespace FarLib::alloc_reclaim_rate_diag
