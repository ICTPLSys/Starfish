#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace FarLib::cache {
// Prototype: one cache-line-isolated counter pair per native thread. A scope
// token retains the entry shard across fibre migration. No libfibre ABI change.
class ShardedScopeCounters {
public:
    static constexpr size_t MaxShards = 256;
    struct alignas(64) Shard {
        std::atomic<size_t> count[2]{};
    };
    static_assert(sizeof(Shard) == 64);

    static int state(int token) { return token & 3; } // V0=1, V1=2
    static size_t shard(int token) { return static_cast<unsigned>(token) >> 2; }
    static size_t registered_shards() { return next_shard.load(std::memory_order_seq_cst); }

    int enter() { return enter_on(thread_shard()); }
    int enter_on(size_t id) {
        if (id >= MaxShards) std::abort();
        for (;;) {
            const uint64_t e = epoch.load(std::memory_order_seq_cst);
            auto& count = shards[id].count[e & 1];
            count.fetch_add(1, std::memory_order_seq_cst);
            // Admission must be visible to the flipper, or observe its flip.
            // A full generation (not just its parity) prevents an ABA accept.
            if (epoch.load(std::memory_order_seq_cst) == e)
                return static_cast<int>((id << 2) | ((e & 1) + 1));
            count.fetch_sub(1, std::memory_order_release);
        }
    }
    void leave(int token) {
        const size_t id = shard(token);
        const int s = state(token);
        if (id >= MaxShards || s < 1 || s > 2) std::abort();
        const auto before = shards[id].count[s - 1].fetch_sub(1, std::memory_order_release);
        if (before == 0) std::abort();
    }
    int current_state() const {
        return static_cast<int>((epoch.load(std::memory_order_seq_cst) & 1) + 1);
    }
    // Caller serializes flips and drains the previous flip before the next.
    int flip() {
        return static_cast<int>((epoch.fetch_add(1, std::memory_order_seq_cst) & 1) + 1);
    }
    size_t old_count(int s) const {
        if (s < 1 || s > 2) std::abort();
        size_t sum = 0;
        // Fixed inventory: registrations racing the scan cannot be omitted.
        // 16KiB total, cold path only; optimize scan length separately later.
        for (const auto& slot : shards)
            sum += slot.count[s - 1].load(std::memory_order_seq_cst);
        return sum;
    }
private:
    static size_t thread_shard() {
        static thread_local const size_t id = next_shard.fetch_add(1, std::memory_order_seq_cst);
        if (id >= MaxShards) std::abort();
        return id;
    }
    inline static std::atomic<size_t> next_shard{0};
    alignas(64) std::atomic<uint64_t> epoch{0};
    std::array<Shard, MaxShards> shards{};
};
}
