#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace FarLib::hydra {

// A counter for telemetry only.  It deliberately does not return a previous
// value and must not be used for sequence numbers, ownership, or correctness
// decisions.  All instances share one process-local registry of logical shard
// IDs: an OS thread obtains an ID once in TLS and reuses that shard for every
// counter it updates.  IDs are intentionally assigned modulo the shard count;
// collisions after 64 OS threads affect contention only, never the sum.
class ShardedDiagnosticCounter final {
public:
    static constexpr size_t kShardCount = 64;

    explicit ShardedDiagnosticCounter(uint64_t initial = 0) noexcept {
        shards_[0].value.store(initial, std::memory_order_relaxed);
    }

    ShardedDiagnosticCounter(const ShardedDiagnosticCounter &) = delete;
    ShardedDiagnosticCounter &operator=(const ShardedDiagnosticCounter &) =
        delete;

    // Add to the current OS thread's shard.  The returned old value is
    // intentionally discarded: this operation is not a synchronization or
    // sequence primitive.
    void fetch_add(uint64_t value,
                   std::memory_order order = std::memory_order_relaxed) noexcept {
        shards_[thread_shard()].value.fetch_add(value, order);
    }

    // Concurrent loads are not a linearizable global snapshot: another thread
    // may update a shard between two shard loads.  After writers are joined or
    // otherwise quiesced, the sum is exact.
    uint64_t load(
        std::memory_order order = std::memory_order_relaxed) const noexcept {
        uint64_t total = 0;
        for (const Shard &shard : shards_) {
            total += shard.value.load(order);
        }
        return total;
    }

private:
    struct alignas(64) Shard {
        alignas(64) std::atomic<uint64_t> value{0};
    };

    static_assert(alignof(Shard) >= 64,
                  "diagnostic counter shards must be cache-line aligned");
    static_assert(sizeof(Shard) % 64 == 0,
                  "diagnostic counter shards must not share cache lines");

    static size_t thread_shard() noexcept {
        static thread_local uint8_t shard_id = kUnassignedShard;
        if (shard_id == kUnassignedShard) {
            shard_id = static_cast<uint8_t>(
                registry_next_.fetch_add(1, std::memory_order_relaxed) &
                (kShardCount - 1));
        }
        return shard_id;
    }

    static constexpr uint8_t kUnassignedShard = 0xff;
    inline static std::atomic<uint64_t> registry_next_{0};

    std::array<Shard, kShardCount> shards_{};
};

}  // namespace FarLib::hydra
