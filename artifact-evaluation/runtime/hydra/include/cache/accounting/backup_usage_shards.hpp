#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace FarLib::cache {
// Diagnostic only: no admission decisions and no shared per-object total.
// Releases may occur on another OS thread, so individual nets can be negative.
class BackupUsageShards {
    struct alignas(128) Shard { std::atomic<int64_t> net{0}; };
    static_assert(sizeof(Shard) == 128);
    std::array<Shard, 256> shards{};
    alignas(128) mutable std::atomic<int64_t> observed_peak{0};
    inline static std::atomic<size_t> next_thread_slot{0};
    static size_t thread_slot() {
        static thread_local const size_t slot =
            next_thread_slot.fetch_add(1, std::memory_order_relaxed);
        if (slot >= 256) std::abort();
        return slot;
    }
public:
    void adjust(int64_t delta) {
        shards[thread_slot()].net.fetch_add(delta, std::memory_order_relaxed);
    }
    // Concurrent snapshots are approximate; exact after all writers quiesce.
    // Preserve signed values, including any inconsistent negative snapshot.
    int64_t snapshot() const {
        int64_t value = 0;
        for (const auto &s : shards) value += s.net.load(std::memory_order_relaxed);
        auto peak = observed_peak.load(std::memory_order_relaxed);
        while (peak < value && !observed_peak.compare_exchange_weak(
                   peak, value, std::memory_order_relaxed)) {}
        return value;
    }
    int64_t sampled_peak() const { return observed_peak.load(std::memory_order_relaxed); }
    size_t writers() const { return next_thread_slot.load(std::memory_order_relaxed); }
};
}
