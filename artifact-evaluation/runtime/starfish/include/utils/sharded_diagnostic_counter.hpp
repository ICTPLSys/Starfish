#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace FarLib::profile {

// A telemetry-only counter whose writers are spread over cache-line-isolated
// shards selected once per OS thread.  This deliberately does not return the
// old value from fetch_add, so it cannot accidentally become a sequence,
// ownership, or correctness primitive.  load() is compatible with the
// std::atomic<uint64_t>::load API; while writers are active it is a relaxed
// diagnostic snapshot, and after writers quiesce its sum is exact.
class ShardedDiagnosticCounter final {
public:
    static constexpr size_t kShardCount = 64;

    explicit ShardedDiagnosticCounter(uint64_t initial = 0) noexcept {
        shards_[0].value.store(initial, std::memory_order_relaxed);
    }

    ShardedDiagnosticCounter(const ShardedDiagnosticCounter &) = delete;
    ShardedDiagnosticCounter &operator=(const ShardedDiagnosticCounter &) =
        delete;

    void fetch_add(
        uint64_t value,
        std::memory_order order = std::memory_order_relaxed) noexcept {
        shards_[thread_shard()].value.fetch_add(value, order);
    }

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

}  // namespace FarLib::profile
