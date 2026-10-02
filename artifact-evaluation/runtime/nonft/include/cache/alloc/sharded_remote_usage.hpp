#pragma once
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace FarLib::allocator::remote {

// Accounting only: this class neither owns nor changes any allocation bitmap.
// A shard survives its writer thread until reset/destruction of the whole heap.
class ShardedRemoteUsage {
    struct alignas(64) Cell { uint64_t value = 0; };
    struct alignas(64) Shard {
        std::atomic_flag busy = ATOMIC_FLAG_INIT;
        std::atomic<uint64_t> version{0};
        // Unsigned modular deltas intentionally allow a thread to free another
        // thread's allocation. Only the aggregate is a nonnegative byte count.
        std::vector<Cell> delta;
        std::unique_ptr<std::atomic<uint64_t>[]> observer_delta;
        explicit Shard(size_t endpoints) : delta(endpoints),
                                            observer_delta(new std::atomic<uint64_t>[endpoints]) {
            for (size_t i = 0; i < endpoints; ++i)
                observer_delta[i].store(0, std::memory_order_relaxed);
        }
        void lock() {
            while (busy.test_and_set(std::memory_order_acquire)) {
                while (busy.test(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#endif
                }
            }
            version.fetch_add(1, std::memory_order_acq_rel);
            // Pair through an observed relaxed payload with the observer's
            // acquire fence: a new payload cannot validate an old even version.
            std::atomic_thread_fence(std::memory_order_release);
        }
        void unlock() {
            version.fetch_add(1, std::memory_order_release);
            busy.clear(std::memory_order_release);
        }
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
    using PublishedShards = std::vector<Shard *>;
    std::shared_ptr<const PublishedShards> published_shards;
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
        auto next = std::make_shared<PublishedShards>();
        if (const auto current = std::atomic_load_explicit(
                &published_shards, std::memory_order_acquire))
            *next = *current;
        next->push_back(raw);
        std::atomic_store_explicit(
            &published_shards,
            std::shared_ptr<const PublishedShards>(std::move(next)),
            std::memory_order_release);
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
        std::atomic_store_explicit(
            &published_shards, std::shared_ptr<const PublishedShards>(),
            std::memory_order_release);
        shards.clear();
        endpoints = count;
        generation = next_generation.fetch_add(1, std::memory_order_relaxed);
        std::atomic_store_explicit(
            &published_shards,
            std::make_shared<const PublishedShards>(),
            std::memory_order_release);
    }
    void add(size_t endpoint, uint64_t bytes) {
        assert(endpoint < endpoints);
        auto &s = local_shard();
        s.lock();
        s.delta[endpoint].value += bytes;
        s.observer_delta[endpoint].fetch_add(bytes, std::memory_order_relaxed);
        s.unlock();
    }
    void subtract(size_t endpoint, uint64_t bytes) {
        assert(endpoint < endpoints);
        auto &s = local_shard();
        s.lock();
        s.delta[endpoint].value -= bytes;
        s.observer_delta[endpoint].fetch_sub(bytes, std::memory_order_relaxed);
        s.unlock();
    }
    // Observer-only snapshot: no registry or shard lock. All shards are read
    // with a bounded global double-collect; a writer racing every attempt is
    // reported explicitly instead of returning a torn or fabricated total.
    std::vector<uint64_t> snapshot_observer() const {
        // Cross-thread frees use modular deltas. All shard values must coexist
        // at one instant: independent per-shard snapshots can wrap the sum.
        constexpr unsigned kMaxAttempts = 4096;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(5);
        std::vector<uint64_t> result(endpoints, 0);
        std::vector<uint64_t> versions;
        std::vector<uint64_t> values;
        for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
            const auto published = std::atomic_load_explicit(
                &published_shards, std::memory_order_acquire);
            if (!published) return result; // reset is quiescent-only
            const size_t count = published->size();
            // Reuse buffers on retries; only a newly published shard can
            // require additional storage. Allocate before starting collection.
            versions.resize(count);
            values.resize(count * endpoints);
            bool valid = true;
            for (size_t i = 0; i < count; ++i) {
                const Shard *shard = (*published)[i];
                versions[i] = shard->version.load(std::memory_order_acquire);
                if (versions[i] & 1) {
                    valid = false;
                    break;
                }
                for (size_t ep = 0; ep < endpoints; ++ep)
                    values[i * endpoints + ep] =
                        shard->observer_delta[ep].load(std::memory_order_relaxed);
            }
            // Keep all payload reads before the second version collection.
            // Together with each initial acquire and the writer's release,
            // this is the atomic-payload seqlock read-side fence.
            std::atomic_thread_fence(std::memory_order_acquire);
            if (valid) {
                for (size_t i = 0; i < count; ++i) {
                    if ((*published)[i]->version.load(std::memory_order_relaxed) !=
                        versions[i]) {
                        valid = false;
                        break;
                    }
                }
            }
            // Registration precedes a new shard's first update. A changed
            // publication must retry too, or its positive delta may be absent
            // while a captured shard already contains the matching free.
            if (valid && published != std::atomic_load_explicit(
                    &published_shards, std::memory_order_acquire))
                valid = false;
            if (valid) {
                for (size_t i = 0; i < count; ++i)
                    for (size_t ep = 0; ep < endpoints; ++ep)
                        result[ep] += values[i * endpoints + ep];
                return result;
            }
            if (std::chrono::steady_clock::now() >= deadline) break;
            std::this_thread::yield(); // observer only; never lock a writer
        }
        throw std::runtime_error(
            "sharded remote observer snapshot retry exhausted");
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
