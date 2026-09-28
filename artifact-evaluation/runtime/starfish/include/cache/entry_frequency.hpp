#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>

namespace FarLib::allocator::six_group {
struct Record;
}

namespace FarLib::cache::detail {

// The recovery Entry is 48B and therefore cannot retain the legacy fixed-six
// atomic pointer inline.  This sidecar is deliberately lazy: ordinary entries
// and the semantic-six path never create a record.  Only the old fixed-six
// registry calls six_binding(), which creates one stable atomic per live Entry.
// References returned by binding() remain valid across unordered_map rehashes;
// lifecycle code erases a key only after the owning Entry is quiescent.
struct EntrySixBindingStore {
    using Record = ::FarLib::allocator::six_group::Record;
    static constexpr size_t ShardCount = 64;

private:
    struct Shard {
        std::mutex mutex;
        std::unordered_map<const void *, std::atomic<Record *>> records;
    };

    struct Store {
        std::array<Shard, ShardCount> shards;
    };

    static std::atomic<Store *> &store_slot() noexcept {
        static std::atomic<Store *> slot{nullptr};
        return slot;
    }

    static Store *ensure_store() {
        Store *store = store_slot().load(std::memory_order_acquire);
        if (store != nullptr) return store;
        Store *candidate = new Store;
        Store *expected = nullptr;
        if (store_slot().compare_exchange_strong(
                expected, candidate, std::memory_order_release,
                std::memory_order_acquire)) {
            return candidate;
        }
        delete candidate;
        return expected;
    }

    static size_t shard_index(const void *key) noexcept {
        uint64_t hash = reinterpret_cast<uintptr_t>(key);
        hash ^= hash >> 33;
        hash *= UINT64_C(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        return static_cast<size_t>(hash & (ShardCount - 1));
    }

public:
    // Called only by the old fixed-six registry.  The map node owns no Record;
    // it stores the same non-owning pointer that the former inline atomic held.
    static std::atomic<Record *> &binding(const void *key) {
        Store *store = ensure_store();
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto [it, inserted] = shard.records.try_emplace(key, nullptr);
        (void)inserted;
        return it->second;
    }

    static Record *load(const void *key) noexcept {
        Store *store = store_slot().load(std::memory_order_acquire);
        if (store == nullptr || key == nullptr) return nullptr;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        return it == shard.records.end()
                   ? nullptr
                   : it->second.load(std::memory_order_acquire);
    }

    static Record *exchange(const void *key, Record *desired) noexcept {
        if (key == nullptr) return nullptr;
        if (desired != nullptr) {
            return binding(key).exchange(desired, std::memory_order_acq_rel);
        }
        Store *store = store_slot().load(std::memory_order_acquire);
        if (store == nullptr) return nullptr;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        return it == shard.records.end()
                   ? nullptr
                   : it->second.exchange(nullptr, std::memory_order_acq_rel);
    }

    static void store(const void *key, Record *value) {
        if (key == nullptr) return;
        if (value == nullptr) {
            (void)exchange(key, nullptr);
            return;
        }
        binding(key).store(value, std::memory_order_release);
    }

    static void erase(const void *key) noexcept {
        Store *store = store_slot().load(std::memory_order_acquire);
        if (store == nullptr || key == nullptr) return;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        if (it == shard.records.end()) return;
        it->second.store(nullptr, std::memory_order_release);
        shard.records.erase(it);
    }

    static size_t live_records() noexcept {
        Store *store = store_slot().load(std::memory_order_acquire);
        if (store == nullptr) return 0;
        size_t total = 0;
        for (Shard &shard : store->shards) {
            std::lock_guard<std::mutex> lock(shard.mutex);
            total += shard.records.size();
        }
        return total;
    }
};

// Optional per-entry frequency sidecar.  FarObjectEntry keeps the identity and
// lifecycle; this store is enabled once during cache initialization when the
// selected profiling policy needs the counters.  The backing allocation is
// intentionally process-lifetime so cache/static destruction order cannot make
// a late diagnostic access dereference a destroyed store.
struct EntryFrequencyStore {
    static constexpr size_t ShardCount = 64;
    static constexpr uint32_t CounterMax =
        std::numeric_limits<uint32_t>::max();

private:
    struct Record {
        uint32_t window = 0;
        uint32_t ema = 0;
        uint32_t published_window = 0;
        uint32_t published_ema = 0;
    };

    struct Shard {
        std::mutex mutex;
        std::unordered_map<const void *, Record> records;
    };

    struct Store {
        std::array<Shard, ShardCount> shards;
    };

    static std::atomic<bool> &enabled_flag() noexcept {
        static std::atomic<bool> enabled{false};
        return enabled;
    }

    static std::atomic<uint64_t> &allocation_failure_count() noexcept {
        static std::atomic<uint64_t> failures{0};
        return failures;
    }

    static std::atomic<Store *> &store_slot() noexcept {
        // The atomic itself is a function-local static; Store is allocated once
        // and deliberately never deleted.  This avoids a static-destruction
        // ordering dependency with ConcurrentArrayCache diagnostics.
        static std::atomic<Store *> slot{nullptr};
        return slot;
    }

    static Store *ensure_store() {
        Store *store = store_slot().load(std::memory_order_acquire);
        if (store != nullptr) return store;

        Store *candidate = new Store;
        Store *expected = nullptr;
        if (store_slot().compare_exchange_strong(
                expected, candidate, std::memory_order_release,
                std::memory_order_acquire)) {
            return candidate;
        }
        delete candidate;
        return expected;
    }

    static Store *load_store_if_enabled() noexcept {
        if (!enabled_flag().load(std::memory_order_acquire)) return nullptr;
        return store_slot().load(std::memory_order_acquire);
    }

    static size_t shard_index(const void *key) noexcept {
        // Entry addresses are aligned and frequently contiguous. Mix high
        // address bits too; masking identity hashes would use few shards.
        uint64_t hash = reinterpret_cast<uintptr_t>(key);
        hash ^= hash >> 33;
        hash *= UINT64_C(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        return hash & (ShardCount - 1);
    }

    static void clear_store(Store &store) {
        for (Shard &shard : store.shards) {
            std::lock_guard<std::mutex> lock(shard.mutex);
            shard.records.clear();
        }
    }

public:
    // Configuration is expected to be quiescent: call this during cache
    // initialization/teardown, not while entry lifecycle operations run.
    // Disabling first makes late statistics no-op; the existing sidecar is then
    // cleared without freeing the process-lifetime Store object.
    static void set_enabled(bool enabled) {
        if (!enabled) {
            enabled_flag().store(false, std::memory_order_release);
            Store *store = store_slot().load(std::memory_order_acquire);
            if (store != nullptr) clear_store(*store);
            return;
        }
        (void)ensure_store();
        enabled_flag().store(true, std::memory_order_release);
    }

    static bool enabled() noexcept {
        return enabled_flag().load(std::memory_order_acquire);
    }

    static uint64_t allocation_failures() noexcept {
        return allocation_failure_count().load(std::memory_order_relaxed);
    }

    // Entry lifecycle owns record creation.  Statistics arriving after erase
    // must never recreate a record for a reused address.
    static void register_entry(const void *key) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        try {
            shard.records.try_emplace(key);
        } catch (const std::bad_alloc &) {
            // Diagnostics must not unwind an already allocated cache object.
            // Missing records remain no-op, and the loss is reported explicitly.
            allocation_failure_count().fetch_add(1, std::memory_order_relaxed);
        }
    }

    static uint32_t load_window(const void *key) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return 0;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        return it == shard.records.end() ? 0 : it->second.window;
    }

    static uint32_t load_ema(const void *key) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return 0;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        return it == shard.records.end() ? 0 : it->second.ema;
    }

    static uint32_t load_published_window(const void *key) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return 0;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        return it == shard.records.end() ? 0 : it->second.published_window;
    }

    static uint32_t load_published_ema(const void *key) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return 0;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        return it == shard.records.end() ? 0 : it->second.published_ema;
    }

    static void add_window(const void *key, uint32_t delta) {
        if (delta == 0) return;
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        if (it == shard.records.end()) return;
        uint32_t &window = it->second.window;
        window = delta > CounterMax - window ? CounterMax : window + delta;
    }

    static bool try_add_window(const void *key, uint32_t delta) {
        if (delta == 0) return true;
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return false;
        Shard &shard = store->shards[shard_index(key)];
        std::unique_lock<std::mutex> lock(shard.mutex, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        auto it = shard.records.find(key);
        if (it == shard.records.end()) return false;
        uint32_t &window = it->second.window;
        window = delta > CounterMax - window ? CounterMax : window + delta;
        return true;
    }

    static uint32_t consume_window(const void *key) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return 0;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        if (it == shard.records.end()) return 0;
        const uint32_t old = it->second.window;
        it->second.window = 0;
        return old;
    }

    static void update_ema(const void *key, uint32_t observed_frequency,
                           uint32_t decay_shift) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        if (it == shard.records.end()) return;
        uint32_t current = it->second.ema;
        const uint32_t decay =
            decay_shift == 0 || decay_shift >= 32 ? 0u
                                                   : current >> decay_shift;
        uint32_t next = current - decay;
        next = observed_frequency > CounterMax - next
                   ? CounterMax
                   : next + observed_frequency;
        it->second.ema = next;
    }

    static void publish(const void *key, uint32_t window, uint32_t ema) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.records.find(key);
        if (it == shard.records.end()) return;
        it->second.published_window = window;
        it->second.published_ema = ema;
    }

    // Copy a source snapshot, then assign the destination separately. Never
    // hold two shard locks, so entries that hash to the same shard are safe.
    static void copy(const void *from, const void *to) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || from == nullptr || to == nullptr ||
            from == to) {
            return;
        }

        Record snapshot;
        bool source_exists = false;
        {
            Shard &source = store->shards[shard_index(from)];
            std::lock_guard<std::mutex> lock(source.mutex);
            auto it = source.records.find(from);
            if (it != source.records.end()) {
                snapshot = it->second;
                source_exists = true;
            }
        }

        Shard &destination = store->shards[shard_index(to)];
        std::lock_guard<std::mutex> lock(destination.mutex);
        if (!source_exists) {
            destination.records.erase(to);
            return;
        }
        try {
            auto [it, inserted] = destination.records.try_emplace(to);
            (void)inserted;
            it->second = snapshot;
        } catch (const std::bad_alloc &) {
            // Handle moves can be noexcept and already own allocator metadata.
            // Drop optional statistics, never abort a functional object move.
            destination.records.erase(to);
            allocation_failure_count().fetch_add(1, std::memory_order_relaxed);
        }
    }

    static void erase(const void *key) {
        Store *store = load_store_if_enabled();
        if (store == nullptr || key == nullptr) return;
        Shard &shard = store->shards[shard_index(key)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        shard.records.erase(key);
    }

    static size_t live_records() {
        Store *store = load_store_if_enabled();
        if (store == nullptr) return 0;
        size_t total = 0;
        for (Shard &shard : store->shards) {
            std::lock_guard<std::mutex> lock(shard.mutex);
            total += shard.records.size();
        }
        return total;
    }
};

}  // namespace FarLib::cache::detail
