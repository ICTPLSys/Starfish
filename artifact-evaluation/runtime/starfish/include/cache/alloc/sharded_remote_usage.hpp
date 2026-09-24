#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace FarLib::allocator::remote {

// Accounting only: this class neither owns nor changes any allocation bitmap.
// A shard survives its writer thread until reset/destruction of the whole heap.
class ShardedRemoteUsage {
    struct alignas(64) Cell { uint64_t value = 0; };
    struct alignas(64) Shard {
        std::atomic_flag busy = ATOMIC_FLAG_INIT;
        // Unsigned modular deltas intentionally allow a thread to free another
        // thread's allocation. Only the aggregate is a nonnegative byte count.
        std::vector<Cell> delta;
        explicit Shard(size_t endpoints) : delta(endpoints) {}
        void lock() {
            while (busy.test_and_set(std::memory_order_acquire)) {
                while (busy.test(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#endif
                }
            }
        }
        void unlock() { busy.clear(std::memory_order_release); }
    };
    struct CachedShard {
        const ShardedRemoteUsage *owner;
        uint64_t generation;
        Shard *shard;
    };
    inline static std::atomic<uint64_t> next_generation{1};
    inline static thread_local CachedShard cached{nullptr, 0, nullptr};
    mutable std::mutex registry_lock;
    std::vector<std::unique_ptr<Shard>> shards;
    size_t endpoints = 0;
    uint64_t generation = 0;

    Shard &local_shard() {
        if (cached.owner == this && cached.generation == generation)
            return *cached.shard;
        // First touch only. Writers never hold a shard lock while registering.
        std::lock_guard<std::mutex> guard(registry_lock);
        auto p = std::make_unique<Shard>(endpoints);
        Shard *raw = p.get();
        shards.push_back(std::move(p));
        cached.owner = this;
        cached.generation = generation;
        cached.shard = raw;
        return *raw;
    }

public:
    // Same quiescent-only contract as RemoteGlobalHeap::register_remote:
    // no concurrent updates/snapshots/reset/destruction during this operation.
    void reset(size_t count) {
        assert(count != 0);
        std::lock_guard<std::mutex> guard(registry_lock);
        shards.clear();
        endpoints = count;
        generation = next_generation.fetch_add(1, std::memory_order_relaxed);
    }
    void add(size_t endpoint, uint64_t bytes) {
        assert(endpoint < endpoints);
        auto &s = local_shard();
        s.lock();
        s.delta[endpoint].value += bytes;
        s.unlock();
    }
    void subtract(size_t endpoint, uint64_t bytes) {
        assert(endpoint < endpoints);
        auto &s = local_shard();
        s.lock();
        s.delta[endpoint].value -= bytes;
        s.unlock();
    }
    std::vector<uint64_t> snapshot() const {
        // Freeze registration, then acquire every shard in registration order.
        // At the last lock acquisition all deltas coexist at one real instant.
        // Writers take only one shard, never registry_lock while holding it.
        std::lock_guard<std::mutex> guard(registry_lock);
        std::vector<uint64_t> result(endpoints, 0); // allocate before shard locks
        for (auto &s : shards) s->lock();
        for (auto &s : shards)
            for (size_t ep = 0; ep < endpoints; ++ep)
                result[ep] += s->delta[ep].value;
        for (auto it = shards.rbegin(); it != shards.rend(); ++it)
            (*it)->unlock();
        return result;
    }
    uint64_t total() const {
        const auto values = snapshot();
        uint64_t n = 0;
        for (auto v : values) n += v;
        return n;
    }
    uint64_t server(size_t ep) const {
        assert(ep < endpoints);
        return snapshot()[ep];
    }
};
} // namespace FarLib::allocator::remote
