#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>
#include <utility>
#include <memory>
#include <vector>

#include "async/context.hpp"
#include "cache/cache.hpp"
#include "utils/spinlock.hpp"
#include "utils/cpu_cycles.hpp"
#include "utils/parallel.hpp"

namespace FarLib {

template <typename Key, typename Val, typename Hash = std::hash<Key>>
class ConcurrentHashMap {
private:
    struct KVData {
        Key key;
        Val val;
    };

    struct BucketEntry {
        constexpr static uintptr_t kBusyPtr = 0x1;
        std::atomic_uint32_t bitmap;
        Spinlock spin;
        std::atomic_uint64_t timestamp;
        UniqueFarPtr<KVData> ptr;

        BucketEntry() : bitmap(0), timestamp(0) {}

        uint32_t load_bitmap() const {
            return bitmap.load(std::memory_order_acquire);
        }

        bool has_bitmap(uint32_t mask) const {
            return (load_bitmap() & mask) != 0;
        }

        void add_bitmap(uint32_t mask) {
            bitmap.fetch_or(mask, std::memory_order_acq_rel);
        }

        void remove_bitmap(uint32_t mask) {
            bitmap.fetch_and(~mask, std::memory_order_acq_rel);
        }

        uint64_t get_timestamp() {
            return timestamp.load(std::memory_order::seq_cst);
        }

        void set_timestamp(uint64_t v) {
            timestamp.store(v, std::memory_order::relaxed);
        }

        void bump_timestamp() {
            timestamp.fetch_add(1, std::memory_order_seq_cst);
        }

        bool cas_busy() { return ptr.cas_null_to_busy(); }
    };

    constexpr static uint32_t kNeighborhood = 32;
    constexpr static uint32_t kMaxRetries = 2;

    const uint32_t kHashMask_;
    const uint32_t kNumEntries_;
    std::unique_ptr<BucketEntry[]> buckets_;

    void do_remove(BucketEntry *bucket, BucketEntry *entry);

    size_t get_hash(const Key &key) { return Hash{}(key); }

    static inline std::atomic_uint64_t local_deref_count_{0};
    static inline std::atomic_uint64_t remote_deref_count_{0};

    static void record_deref(bool at_local) {
        if (at_local) {
            local_deref_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            remote_deref_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    LiteAccessor<KVData> access_entry(BucketEntry *entry, __DMH__,
                                      DereferenceScope &scope) {
        bool at_local = cache::at_local(entry->ptr.obj());
        record_deref(at_local);
        return entry->ptr.access(__on_miss__, scope);
    }

public:
    ConcurrentHashMap(uint32_t num_entries_shift);
    ~ConcurrentHashMap() = default;

    bool get(const Key &key, Val *val, __DMH__, DereferenceScope &scope);
    bool put(const Key &key, Val val, __DMH__, DereferenceScope &scope);
    bool remove(const Key &key, __DMH__, DereferenceScope &scope);
    template <typename Fn>
    void for_each_locked(Fn &&fn, __DMH__, DereferenceScope &scope);
    template <typename Fn>
    void for_each_snapshot(Fn &&fn, __DMH__, DereferenceScope &scope);
    // Quiescent-table API: no concurrent insert/remove/update is permitted.
    // Reads are parallel; callbacks run serially in the original bucket order.
    template <typename Fn>
    void for_each_readonly_parallel(Fn &&fn, size_t readers);
    static uint64_t local_deref_count() {
        return local_deref_count_.load(std::memory_order_relaxed);
    }
    static uint64_t remote_deref_count() {
        return remote_deref_count_.load(std::memory_order_relaxed);
    }
    static void reset_deref_counts() {
        local_deref_count_.store(0, std::memory_order_relaxed);
        remote_deref_count_.store(0, std::memory_order_relaxed);
    }

    bool get(const Key &key, Val *val, DereferenceScope &scope) {
        ON_MISS_BEGIN
        ON_MISS_END
        return get(key, val, __on_miss__, scope);
    }
    bool put(const Key &key, Val val, DereferenceScope &scope) {
        ON_MISS_BEGIN
        ON_MISS_END
        return put(key, val, __on_miss__, scope);
    }
    bool remove(const Key &key, DereferenceScope &scope) {
        ON_MISS_BEGIN
        ON_MISS_END
        return remove(key, __on_miss__, scope);
    }
    template <typename Fn>
    void for_each_locked(Fn &&fn, DereferenceScope &scope) {
        ON_MISS_BEGIN
        ON_MISS_END
        for_each_locked(std::forward<Fn>(fn), __on_miss__, scope);
    }
    template <typename Fn>
    void for_each_snapshot(Fn &&fn, DereferenceScope &scope) {
        ON_MISS_BEGIN
        ON_MISS_END
        for_each_snapshot(std::forward<Fn>(fn), __on_miss__, scope);
    }

    template <class Context>
        requires requires(Context context, const Val *value) {
            { context.get_key() } -> std::convertible_to<const Key *>;
            {
                context.get_hash_map()
            } -> std::convertible_to<ConcurrentHashMap<Key, Val, Hash> *>;
            context.make_result(value);
        }
    struct GetFrame : Context {
        // local vars
        BucketEntry *bucket;
        uint32_t bitmap;
        uint64_t timestamp;
        uint32_t retry_counter;
        uint32_t bucket_idx;
        LiteAccessor<KVData> data;
        enum { INIT, FETCH, FETCH_LOCK } status;

        size_t conflict_id() const { return bucket_idx; }
        bool fetched() { return cache::at_local(data); }
        void pin() { data.pin(); }
        void unpin() { data.unpin(); }
        void init() {
            std::construct_at(&data);
            status = INIT;
            retry_counter = 0;
            auto hash_map = Context::get_hash_map();
            uint32_t hash = hash_map->get_hash(*Context::get_key());
            bucket_idx = hash & hash_map->kHashMask_;
            bucket = &(hash_map->buckets_[bucket_idx]);
        }
        bool run(DereferenceScope &scope) {
            switch (status) {
            case FETCH:
                goto FETCH;
            case FETCH_LOCK:
                goto FETCH_LOCK;
            }
            // fast path
            do {
                timestamp = bucket->get_timestamp();
                bitmap = bucket->load_bitmap();
                while (bitmap) {
                    {
                        auto offset = __builtin_ctz(bitmap);
                        BucketEntry *entry = &bucket[offset];
                        bitmap ^= (1 << offset);
                        if (entry->ptr.is_null()) [[unlikely]]
                            continue;
                        bool at_local = data.async_fetch(entry->ptr, scope);
                        if (!at_local) {
                            status = FETCH;
                            return false;
                        }
                    }
                FETCH:
                    if (!data.is_null() && data->key == *Context::get_key()) {
                        Context::make_result(&(data->val));
                        return true;
                    }
                }
            } while (timestamp != bucket->get_timestamp() &&
                     retry_counter++ < kMaxRetries);
            if (timestamp == bucket->get_timestamp()) {
                // not found
                data = {};
                Context::make_result(nullptr);
                return true;
            }
            // slow path
            bucket->spin.lock(scope);
            bitmap = bucket->load_bitmap();
            while (bitmap) {
                {
                    auto offset = __builtin_ctz(bitmap);
                    BucketEntry *entry = &bucket[offset];
                    bitmap ^= (1 << offset);
                    if (entry->ptr.is_null()) [[unlikely]]
                        continue;
                    bool at_local = data.async_fetch(entry->ptr, scope);
                    if (!at_local) {
                        status = FETCH_LOCK;
                        return false;
                    }
                }
            FETCH_LOCK:
                if (data->key == *Context::get_key()) {
                    Context::make_result(&(data->val));
                    bucket->spin.unlock();
                    return true;
                }
            }
            bucket->spin.unlock();
            // not found
            data = {};
            Context::make_result(nullptr);
            return true;
        }
    };

    template <class Context>
        requires requires(Context context, bool result) {
            { context.get_key() } -> std::convertible_to<const Key *>;
            {
                context.get_hash_map()
            } -> std::convertible_to<ConcurrentHashMap<Key, Val, Hash> *>;
            context.make_result(result);
        }
    struct RemoveFrame : Context {
        // local vars
        BucketEntry *bucket;
        uint32_t bucket_idx;
        uint32_t bitmap;
        uint64_t timestamp;
        BucketEntry *entry;
        LiteAccessor<KVData> data;
        enum { INIT, FETCH } status;

        bool fetched() { return cache::at_local(data); }
        void pin() { data.pin(); }
        void unpin() { data.unpin(); }
        void init() {
            status = INIT;
            std::construct_at(&data);
            auto hash_map = Context::get_hash_map();
            uint32_t hash = hash_map->get_hash(*Context::get_key());
            bucket_idx = hash & hash_map->kHashMask_;
            bucket = &(hash_map->buckets_[bucket_idx]);
        }
        size_t conflict_id() const { return bucket_idx; }
        bool run(DereferenceScope &scope) {
            switch (status) {
            case FETCH:
                goto FETCH;
            }
            bucket->spin.lock(scope);
            bitmap = bucket->load_bitmap();
            while (bitmap) {
                {
                    auto offset = __builtin_ctz(bitmap);
                    entry = &bucket[offset];
                    bitmap ^= (1 << offset);
                    bool at_local = data.async_fetch(entry->ptr, scope);
                    if (!at_local) {
                        status = FETCH;
                        return false;
                    }
                }
            FETCH:
                if (data->key == *Context::get_key()) {
                    auto hash_map = Context::get_hash_map();
                    hash_map->size--;
                    hash_map->do_remove(bucket, entry);
                    bucket->spin.unlock();
                    Context::make_result(true);
                    return true;
                }
            }
            bucket->spin.unlock();
            // not found
            data = {};
            Context::make_result(false);
            return true;
        }
    };

    template <class Context>
        requires requires(Context context, bool result) {
            { context.get_key() } -> std::convertible_to<const Key *>;
            { context.get_value() } -> std::convertible_to<Val>;
            {
                context.get_hash_map()
            } -> std::convertible_to<ConcurrentHashMap<Key, Val, Hash> *>;
            context.make_result(result);
        }
    struct PutFrame : Context {
        // local vars
        BucketEntry *bucket;
        uint32_t bucket_idx;
        uint32_t orig_bucket_idx;
        uint32_t bitmap;
        uint64_t timestamp;
        LiteAccessor<KVData> data;

        enum { INIT, FETCH_LOCK } status;

        bool fetched() { return cache::at_local(data); }
        void pin() { data.pin(); }
        void unpin() { data.unpin(); }
        size_t conflict_id() const { return orig_bucket_idx; }
        void init() {
            status = INIT;
            std::construct_at(&data);
            auto hash_map = Context::get_hash_map();
            uint32_t hash = hash_map->get_hash(*Context::get_key());
            bucket_idx = hash & hash_map->kHashMask_;
            orig_bucket_idx = bucket_idx;
            bucket = &(hash_map->buckets_[bucket_idx]);
        }
        bool run(DereferenceScope &scope) {
            switch (status) {
            case FETCH_LOCK:
                goto FETCH_LOCK;
            }

            //! 1. key exists
            bucket->spin.lock(scope);
            bitmap = bucket->load_bitmap();
            while (bitmap) {
                {
                    auto offset = __builtin_ctz(bitmap);
                    BucketEntry *entry = &bucket[offset];
                    bitmap ^= (1 << offset);
                    bool at_local = data.async_fetch(entry->ptr, scope);
                    if (!at_local) {
                        status = FETCH_LOCK;
                        return false;
                    }
                }
            FETCH_LOCK:
                if (data->key == *Context::get_key()) {
                    data.as_mut()->val = Context::get_value();
                    Context::make_result(true);
                    bucket->spin.unlock();
                    return true;
                }
            }

            //! key not exists
            auto hash_map = Context::get_hash_map();
            while (bucket_idx < hash_map->kNumEntries_) {
                if (hash_map->buckets_[bucket_idx].cas_busy()) break;
                bucket_idx++;
            }
            if (bucket_idx == hash_map->kNumEntries_) {
                bucket->spin.unlock();
                Context::make_result(false);
                return true;
            }

            uint32_t distance_to_orig_bucket;
            while ((distance_to_orig_bucket = bucket_idx - orig_bucket_idx) >=
                   hash_map->kNeighborhood) {
                // Try to see if we can move things backward.
                uint32_t distance;
                for (distance = kNeighborhood - 1; distance > 0; distance--) {
                    uint32_t idx = bucket_idx - distance;
                    BucketEntry *anchor_entry = &(hash_map->buckets_[idx]);
                    if (!anchor_entry->load_bitmap()) {
                        continue;
                    }

                    // Lock and recheck bitmap.
                    anchor_entry->spin.lock(scope);
                    auto bitmap = anchor_entry->load_bitmap();
                    if (!bitmap) {
                        anchor_entry->spin.unlock();
                        continue;
                    }

                    // Get the offset of the first entry within the bucket.
                    auto offset = __builtin_ctz(bitmap);
                    if (idx + offset >= bucket_idx) {
                        anchor_entry->spin.unlock();
                        continue;
                    }

                    // Swap entry [closest_bucket + offset] and [bucket_idx]
                    auto *from_entry = &(hash_map->buckets_[idx + offset]);
                    auto *to_entry = &(hash_map->buckets_[bucket_idx]);

                    from_entry->ptr.atomic_move_to_and_set_busy(to_entry->ptr);
                    assert(!anchor_entry->has_bitmap(1u << distance));
                    anchor_entry->add_bitmap(1u << distance);
                    anchor_entry->set_timestamp(anchor_entry->get_timestamp() +
                                                1);

                    // from_entry->set_busy();
                    assert(anchor_entry->has_bitmap(1u << offset));
                    anchor_entry->remove_bitmap(1u << offset);

                    // Jump backward.
                    bucket_idx = idx + offset;
                    anchor_entry->spin.unlock();
                    break;
                }

                if (!distance) {
                    bucket->spin.unlock();
                    Context::make_result(false);
                    return true;
                }
            }

            // Allocate memory.
            BucketEntry *final_entry = &(hash_map->buckets_[bucket_idx]);
            LiteAccessor<KVData, true> data =
                final_entry->ptr.allocate_lite_from_busy(scope);
            hash_map->size++;

            // Write object.
            data->key = *Context::get_key();
            data->val = Context::get_value();

            // Update the bitmap of the final bucket.
            assert(!bucket->has_bitmap(1u << distance_to_orig_bucket));
            bucket->add_bitmap(1u << distance_to_orig_bucket);
            bucket->bump_timestamp();

            bucket->spin.unlock();
            Context::make_result(true);
            return true;
        }
    };

    std::atomic<size_t> size;

    template <bool Mut>
    struct DefaultGetCont {
        void operator()(LiteAccessor<Val, Mut> result) {}
    };

    template <bool Mut = false, typename Cont = DefaultGetCont<Mut>>
    struct GetContext : public async::GenericOrderedContext {
        using Map = ConcurrentHashMap<Key, Val, Hash>;
        using Result = LiteAccessor<Val, Mut>;
        const Key *key;
        BucketEntry *bucket;
        uint32_t bitmap;
        uint64_t timestamp;
        uint32_t retry_counter;
        LiteAccessor<KVData, Mut> data;
        Cont continuation;
        enum { INIT, FETCH, FETCH_LOCK } status;

        GetContext(Map *map, const Key *key, Cont &&cont = {})
            : key(key), data(), continuation(cont) {
            status = INIT;
            retry_counter = 0;
            uint32_t hash = map->get_hash(*key);
            uint32_t bucket_idx = hash & map->kHashMask_;
            set_conflict_id(bucket_idx);
            bucket = &(map->buckets_[bucket_idx]);
        }

        virtual bool fetched() const override { return cache::at_local(data); }
        virtual void pin() const override { data.pin(); }
        virtual void unpin() const override { data.unpin(); }
        virtual void destruct() override { std::destroy_at(&data); }
        void set_result() {
            if (!data.is_null()) {
                continuation(Result(data, &(data->val)));
            } else {
                continuation(Result{});
            }
        }

        virtual bool run(DereferenceScope &scope) override {
            switch (status) {
            case FETCH:
                goto FETCH;
            case FETCH_LOCK:
                goto FETCH_LOCK;
            }
            // fast path
            do {
                timestamp = bucket->get_timestamp();
                bitmap = bucket->load_bitmap();
                while (bitmap) {
                    {
                        auto offset = __builtin_ctz(bitmap);
                        BucketEntry *entry = &bucket[offset];
                        bitmap ^= (1 << offset);
                        if (entry->ptr.is_null()) [[unlikely]]
                            continue;
                        bool at_local = data.async_fetch(entry->ptr, scope);
                        if (!at_local) {
                            status = FETCH;
                            return false;
                        }
                    }
                FETCH:
                    if (!data.is_null() && data->key == *key) {
                        set_result();
                        return true;
                    }
                }
            } while (timestamp != bucket->get_timestamp() &&
                     retry_counter++ < kMaxRetries);
            if (timestamp == bucket->get_timestamp()) {
                // not found
                data = {};
                set_result();
                return true;
            }
            // slow path
            bucket->spin.lock(scope);
            bitmap = bucket->load_bitmap();
            while (bitmap) {
                {
                    auto offset = __builtin_ctz(bitmap);
                    BucketEntry *entry = &bucket[offset];
                    bitmap ^= (1 << offset);
                    if (entry->ptr.is_null()) [[unlikely]]
                        continue;
                    bool at_local = data.async_fetch(entry->ptr, scope);
                    if (!at_local) {
                        status = FETCH_LOCK;
                        return false;
                    }
                }
            FETCH_LOCK:
                if (data->key == *key) {
                    set_result();
                    bucket->spin.unlock();
                    return true;
                }
            }
            bucket->spin.unlock();
            // not found
            data = {};
            set_result();
            return true;
        }
    };

    struct DefaultDelCont {
        void operator()(bool result) {}
    };

    template <typename Cont = DefaultDelCont>
    struct RemoveContext : public async::GenericOrderedContext {
        using Map = ConcurrentHashMap<Key, Val, Hash>;
        using Result = bool;
        Map *map;
        const Key *key;
        BucketEntry *bucket;
        uint32_t bucket_idx;
        uint32_t bitmap;
        uint64_t timestamp;
        BucketEntry *entry;
        LiteAccessor<KVData> data;
        Cont cont;
        enum { INIT, FETCH } status;

        RemoveContext(Map *map, const Key *key, Cont &&cont = {})
            : map(map), key(key), data(), cont(cont) {
            status = INIT;
            uint32_t hash = map->get_hash(*key);
            uint32_t bucket_idx = hash & map->kHashMask_;
            set_conflict_id(bucket_idx);
            bucket = &(map->buckets_[bucket_idx]);
        }

        virtual bool fetched() const override { return cache::at_local(data); }
        virtual void pin() const override { data.pin(); }
        virtual void unpin() const override { data.unpin(); }
        virtual void destruct() override { std::destroy_at(&data); }
        virtual bool run(DereferenceScope &scope) override {
            switch (status) {
            case FETCH:
                goto FETCH;
            }
            bucket->spin.lock(scope);
            bitmap = bucket->load_bitmap();
            while (bitmap) {
                {
                    auto offset = __builtin_ctz(bitmap);
                    entry = &bucket[offset];
                    bitmap ^= (1 << offset);
                    bool at_local = data.async_fetch(entry->ptr, scope);
                    if (!at_local) {
                        status = FETCH;
                        return false;
                    }
                }
            FETCH:
                if (data->key == *key) {
                    map->size--;
                    map->do_remove(bucket, entry);
                    bucket->spin.unlock();
                    cont(true);
                    return true;
                }
            }
            bucket->spin.unlock();
            // not found
            data = {};
            cont(false);
            return true;
        }
    };

    struct DefaultPutCont {
        void operator()(bool result) {}
    };

    template <typename Cont = DefaultPutCont>
    struct PutContext : public async::GenericOrderedContext {
        using Map = ConcurrentHashMap<Key, Val, Hash>;
        using Result = bool;
        Map *map;
        Key key;
        Val val;
        BucketEntry *bucket;
        uint32_t bucket_idx;
        uint32_t orig_bucket_idx;
        uint32_t bitmap;
        uint64_t timestamp;
        LiteAccessor<KVData> data;
        Cont cont;
        enum { INIT, FETCH_LOCK } status;

        virtual bool fetched() const override { return cache::at_local(data); }
        virtual void pin() const override { data.pin(); }
        virtual void unpin() const override { data.unpin(); }
        virtual void destruct() override {
            std::destroy_at(&key);
            std::destroy_at(&val);
            std::destroy_at(&data);
        }
        PutContext(Map *map, const Key &key, const Val &val, Cont &&cont = {})
            : map(map), key(key), val(val), data(), cont(cont) {
            status = INIT;
            uint32_t hash = map->get_hash(key);
            bucket_idx = hash & map->kHashMask_;
            orig_bucket_idx = bucket_idx;
            bucket = &(map->buckets_[bucket_idx]);
            set_conflict_id(orig_bucket_idx);
        }
        virtual bool run(DereferenceScope &scope) override {
            switch (status) {
            case FETCH_LOCK:
                goto FETCH_LOCK;
            }

            //! 1. key exists
            bucket->spin.lock(scope);
            bitmap = bucket->load_bitmap();
            while (bitmap) {
                {
                    auto offset = __builtin_ctz(bitmap);
                    BucketEntry *entry = &bucket[offset];
                    bitmap ^= (1 << offset);
                    bool at_local = data.async_fetch(entry->ptr, scope);
                    if (!at_local) {
                        status = FETCH_LOCK;
                        return false;
                    }
                }
            FETCH_LOCK:
                if (data->key == key) {
                    data.as_mut()->val = std::move(val);
                    bucket->spin.unlock();
                    cont(true);
                    return true;
                }
            }

            //! key not exists
            // find a new bucket that not used
            while (bucket_idx < map->kNumEntries_) {
                if (map->buckets_[bucket_idx].cas_busy()) break;
                bucket_idx++;
            }
            if (bucket_idx == map->kNumEntries_) {
                bucket->spin.unlock();
                ERROR("map is full!");
                cont(false);
                return true;
            }

            uint32_t distance_to_orig_bucket;
            while ((distance_to_orig_bucket = bucket_idx - orig_bucket_idx) >=
                   map->kNeighborhood) {
                // Try to see if we can move things backward.
                uint32_t distance;
                for (distance = kNeighborhood - 1; distance > 0; distance--) {
                    uint32_t idx = bucket_idx - distance;
                    BucketEntry *anchor_entry = &(map->buckets_[idx]);
                    if (!anchor_entry->load_bitmap()) {
                        continue;
                    }

                    // Lock and recheck bitmap.
                    anchor_entry->spin.lock(scope);
                    auto bitmap = anchor_entry->load_bitmap();
                    if (!bitmap) {
                        anchor_entry->spin.unlock();
                        continue;
                    }

                    // Get the offset of the first entry within the bucket.
                    auto offset = __builtin_ctz(bitmap);
                    if (idx + offset >= bucket_idx) {
                        anchor_entry->spin.unlock();
                        continue;
                    }

                    // Swap entry [closest_bucket + offset] and [bucket_idx]
                    auto *from_entry = &(map->buckets_[idx + offset]);
                    auto *to_entry = &(map->buckets_[bucket_idx]);

                    from_entry->ptr.atomic_move_to_and_set_busy(to_entry->ptr);
                    assert(!anchor_entry->has_bitmap(1u << distance));
                    anchor_entry->add_bitmap(1u << distance);
                    anchor_entry->set_timestamp(anchor_entry->get_timestamp() +
                                                1);

                    // from_entry->set_busy();
                    assert(anchor_entry->has_bitmap(1u << offset));
                    anchor_entry->remove_bitmap(1u << offset);

                    // Jump backward.
                    bucket_idx = idx + offset;
                    anchor_entry->spin.unlock();
                    break;
                }

                if (!distance) {
                    bucket->spin.unlock();
                    ERROR("bucket cant move!");
                    cont(false);
                    return true;
                }
            }

            // Allocate memory.
            BucketEntry *final_entry = &(map->buckets_[bucket_idx]);
            LiteAccessor<KVData, true> data =
                final_entry->ptr.allocate_lite_from_busy(scope);
            map->size++;

            // Write object.
            data->key = std::move(key);
            data->val = std::move(val);

            // Update the bitmap of the final bucket.
            assert(!bucket->has_bitmap(1u << distance_to_orig_bucket));
            bucket->add_bitmap(1u << distance_to_orig_bucket);
            bucket->bump_timestamp();

            bucket->spin.unlock();
            cont(true);
            return true;
        }
    };
};

template <typename Key, typename Val, typename Hash>
inline ConcurrentHashMap<Key, Val, Hash>::ConcurrentHashMap(
    uint32_t num_entries_shift)
    : kHashMask_((1 << num_entries_shift) - 1),
      kNumEntries_((1 << num_entries_shift) + kNeighborhood), size(0) {
    assert(((kHashMask_ + 1) >> num_entries_shift) == 1);
    buckets_.reset(new BucketEntry[kNumEntries_]);
}

template <typename Key, typename Val, typename Hash>
inline void ConcurrentHashMap<Key, Val, Hash>::do_remove(BucketEntry *bucket,
                                                         BucketEntry *entry) {
    entry->ptr.reset();
    auto offset = entry - bucket;
    assert(bucket->has_bitmap(1u << offset));
    bucket->remove_bitmap(1u << offset);
    bucket->bump_timestamp();
}

template <typename Key, typename Val, typename Hash>
inline bool ConcurrentHashMap<Key, Val, Hash>::get(const Key &key, Val *val,
                                                   __DMH__,
                                                   DereferenceScope &scope) {
    uint32_t hash = get_hash(key);
    uint32_t bucket_idx = hash & kHashMask_;
    BucketEntry *bucket = &(buckets_[bucket_idx]);

    BucketEntry *entry;
    uint64_t timestamp;
    uint32_t retry_counter = 0;

    auto get_once = [&]<bool Lock>() -> bool {
        if constexpr (Lock) {
            bucket->spin.lock(scope);
        }

        timestamp = bucket->get_timestamp();
        uint32_t bitmap = bucket->load_bitmap();
        while (bitmap) {
            auto offset = __builtin_ctz(bitmap);
            entry = &buckets_[bucket_idx + offset];
            if (!entry->ptr.is_null()) [[likely]] {
                LiteAccessor<KVData> data = access_entry(entry, __on_miss__, scope);
                if (!data.is_null() && data->key == key) {
                    *val = data->val;
                    if constexpr (Lock) {
                        bucket->spin.unlock();
                    }
                    return true;
                }
            }

            bitmap ^= (1 << offset);
        }

        if constexpr (Lock) {
            bucket->spin.unlock();
        }
        return false;
    };

    // fast path
    do {
        if (get_once.template operator()<false>()) {
            return true;
        }
    } while (timestamp != bucket->get_timestamp() &&
             retry_counter++ < kMaxRetries);

    // slow path
    if (timestamp != bucket->get_timestamp()) {
        if (get_once.template operator()<true>()) {
            return true;
        }
    }

    return false;
}

template <typename Key, typename Val, typename Hash>
inline bool ConcurrentHashMap<Key, Val, Hash>::put(const Key &key, Val val,
                                                   __DMH__,
                                                   DereferenceScope &scope) {
    // 1. get bucket index
    uint32_t hash = get_hash(key);
    uint32_t bucket_idx = hash & kHashMask_;
    BucketEntry *bucket = &(buckets_[bucket_idx]);
    uint32_t orig_bucket_idx = bucket_idx;

    bucket->spin.lock(scope);

    uint32_t bitmap = bucket->load_bitmap();
    while (bitmap) {
        auto offset = __builtin_ctz(bitmap);
        BucketEntry *entry = bucket + offset;
        assert(!entry->ptr.is_null());
        LiteAccessor<KVData> data = access_entry(entry, __on_miss__, scope);

        // 2.1 key exists, update
        if (data->key == key) {
            // Re-enter through the runtime's mutable dereference path instead
            // of upgrading a read accessor in place.  Selective-backup mode
            // must invalidate a retained remote copy before this write; the
            // mutable path owns that state transition and its move-lock.
            bool at_local = cache::at_local(entry->ptr.obj());
            record_deref(at_local);
            auto mutable_data =
                entry->ptr.template access<true>(__on_miss__, scope);
            mutable_data->val = std::move(val);
            bucket->spin.unlock();
            return true;
        }
        bitmap ^= (1 << offset);
    }

    // 2.2 not exists, find empty slot to insert
    while (bucket_idx < kNumEntries_) {
        if (buckets_[bucket_idx].cas_busy()) break;
        bucket_idx++;
    }

    // 2.3 buckets full, can not insert
    if (bucket_idx == kNumEntries_) {
        bucket->spin.unlock();
        ERROR("bucket full!");
        return false;
    }

    // 2.4 buckets not full, move this bucket to the neighborhood
    uint32_t distance_to_orig_bucket;
    while ((distance_to_orig_bucket = bucket_idx - orig_bucket_idx) >=
           kNeighborhood) {
        // Try to see if we can move things backward.
        uint32_t distance;
        for (distance = kNeighborhood - 1; distance > 0; distance--) {
            uint32_t idx = bucket_idx - distance;
            BucketEntry *anchor_entry = &(buckets_[idx]);
            if (!anchor_entry->load_bitmap()) {
                continue;
            }

            // Lock and recheck bitmap.
            anchor_entry->spin.lock(scope);
            auto bitmap = anchor_entry->load_bitmap();
            if (!bitmap) {
                anchor_entry->spin.unlock();
                continue;
            }

            // Get the offset of the first entry within the bucket.
            auto offset = __builtin_ctz(bitmap);
            if (idx + offset >= bucket_idx) {
                anchor_entry->spin.unlock();
                continue;
            }

            // Swap entry [closest_bucket + offset] and [bucket_idx]
            auto *from_entry = &buckets_[idx + offset];
            auto *to_entry = &buckets_[bucket_idx];

            // now, to is busy, from is valid
            from_entry->ptr.atomic_move_to_and_set_busy(to_entry->ptr);
            // now, from is busy, to is valid
            assert(!anchor_entry->has_bitmap(1u << distance));
            anchor_entry->add_bitmap(1u << distance);
            anchor_entry->set_timestamp(anchor_entry->get_timestamp() + 1);

            assert(anchor_entry->has_bitmap(1u << offset));
            anchor_entry->remove_bitmap(1u << offset);

            // Jump backward.
            bucket_idx = idx + offset;
            anchor_entry->spin.unlock();
            break;
        }

        if (!distance) {
            bucket->spin.unlock();
            ERROR("bucket cant move!");
            return false;
        }
    }
    // now, buckets_[bucket_idx] is busy

    // Allocate memory.
    BucketEntry *final_entry = &buckets_[bucket_idx];
    LiteAccessor<KVData, true> data =
        final_entry->ptr.allocate_lite_from_busy(scope);
    size++;

    // Write object.
    data->key = key;
    data->val = std::move(val);

    // Update the bitmap of the final bucket.
    assert(!bucket->has_bitmap(1u << distance_to_orig_bucket));
    bucket->add_bitmap(1u << distance_to_orig_bucket);
    bucket->bump_timestamp();

    bucket->spin.unlock();

    return true;
}

template <typename Key, typename Val, typename Hash>
inline bool ConcurrentHashMap<Key, Val, Hash>::remove(const Key &key, __DMH__,
                                                      DereferenceScope &scope) {
    uint32_t hash = get_hash(key);
    uint32_t bucket_idx = hash & kHashMask_;
    auto *bucket = &(buckets_[bucket_idx]);

    bucket->spin.lock(scope);

    uint32_t bitmap = bucket->load_bitmap();
    while (bitmap) {
        auto offset = __builtin_ctz(bitmap);
        BucketEntry *entry = &buckets_[bucket_idx + offset];
        assert(!entry->ptr.is_null());
        LiteAccessor<KVData> data = access_entry(entry, __on_miss__, scope);

        if (data->key == key) {
            size--;
            do_remove(bucket, entry);
            bucket->spin.unlock();
            return true;
        }
        bitmap ^= (1 << offset);
    }

    bucket->spin.unlock();
    return false;
}

template <typename Key, typename Val, typename Hash>
template <typename Fn>
inline void ConcurrentHashMap<Key, Val, Hash>::for_each_locked(
    Fn &&fn, __DMH__, DereferenceScope &scope) {
    // Diagnostic-only timing: preserve the original serial traversal and calls.
    const char *diag_env = std::getenv("FARLIB_WC_REDUCE_DIAG");
    const bool timed = diag_env && diag_env[0] == '1';
    const auto wall_begin = std::chrono::steady_clock::now();
    const uint64_t ticks_begin = timed ? get_cycles() : 0;
    uint64_t access_ticks = 0, callback_ticks = 0, visited = 0;
    for (uint32_t bucket_idx = 0; bucket_idx <= kHashMask_; ++bucket_idx) {
        BucketEntry *bucket = &(buckets_[bucket_idx]);
        bucket->spin.lock(scope);
        uint32_t bitmap = bucket->load_bitmap();
        while (bitmap) {
            auto offset = __builtin_ctz(bitmap);
            BucketEntry *entry = &buckets_[bucket_idx + offset];
            if (!entry->ptr.is_null()) [[likely]] {
                const uint64_t access_begin = timed ? get_cycles() : 0;
                LiteAccessor<KVData> data = access_entry(entry, __on_miss__, scope);
                if (timed) access_ticks += get_cycles() - access_begin;
                if (!data.is_null()) {
                    const uint64_t callback_begin = timed ? get_cycles() : 0;
                    fn(data->key, data->val, scope);
                    if (timed) {
                        callback_ticks += get_cycles() - callback_begin;
                        ++visited;
                    }
                }
            }
            bitmap ^= (1 << offset);
        }
        bucket->spin.unlock();
    }
    if (timed) {
        const uint64_t ticks = get_cycles() - ticks_begin;
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_begin).count();
        const double hz = ticks / seconds;
        std::fprintf(stderr,
            "wc_locked_scan_diag visited=%lu elapsed_s=%.9f tsc_hz=%.3f "
            "total_ticks=%lu access_ticks=%lu callback_ticks=%lu "
            "access_s=%.9f callback_s=%.9f other_s=%.9f\n",
            static_cast<unsigned long>(visited), seconds, hz,
            static_cast<unsigned long>(ticks),
            static_cast<unsigned long>(access_ticks),
            static_cast<unsigned long>(callback_ticks),
            access_ticks / hz, callback_ticks / hz,
            (ticks - access_ticks - callback_ticks) / hz);
    }
}

template <typename Key, typename Val, typename Hash>
template <typename Fn>
inline void ConcurrentHashMap<Key, Val, Hash>::for_each_readonly_parallel(
    Fn &&fn, size_t readers) {
    assert(readers > 1);
    constexpr size_t batch_limit = 65536;
    using Item = std::pair<Key, Val>;
    std::vector<BucketEntry *> entries;
    entries.reserve(batch_limit);
    std::vector<Item> items(batch_limit);
    uint32_t next_bucket = 0;
    size_t visited = 0, batches = 0, peak_items = 0;
    double gather_s = 0, fetch_s = 0, callback_s = 0;
    int64_t fetch_reads = 0, fetch_read_bytes = 0;
    int64_t fetch_writes = 0, fetch_write_bytes = 0;
    const auto start = std::chrono::steady_clock::now();
    wc_object_diag::begin_capture();
    while (next_bucket <= kHashMask_) {
        entries.clear();
        auto part_start = std::chrono::steady_clock::now();
        {
            // Do not keep this scope alive while joining reader fibres: the
            // parent would otherwise prevent old epochs from draining.
            RootDereferenceScope scope;
            while (next_bucket <= kHashMask_ &&
                   entries.size() + kNeighborhood <= batch_limit) {
                BucketEntry *bucket = &buckets_[next_bucket];
                bucket->spin.lock(scope);
                uint32_t bitmap = bucket->load_bitmap();
                while (bitmap) {
                    const auto offset = __builtin_ctz(bitmap);
                    BucketEntry *entry = &buckets_[next_bucket + offset];
                    if (!entry->ptr.is_null()) entries.push_back(entry);
                    bitmap ^= (1u << offset);
                }
                bucket->spin.unlock();
                ++next_bucket;
            }
        }
        gather_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - part_start).count();
        if (entries.empty()) break;
        const auto r0 = profile::collect_rdma_read_post_count();
        const auto rb0 = profile::collect_rdma_read_post_bytes();
        const auto w0 = profile::collect_rdma_write_post_count();
        const auto wb0 = profile::collect_rdma_write_post_bytes();
        part_start = std::chrono::steady_clock::now();
        uthread::parallel_for_with_scope<1>(readers, entries.size(),
            [&](size_t i, auto &scope) {
                ON_MISS_BEGIN
                ON_MISS_END
                auto *wc_sample = wc_object_diag::begin(
                    visited + i, batches, entries[i]->ptr.obj().obj_id);
                LiteAccessor<KVData> data =
                    access_entry(entries[i], __on_miss__, scope);
                wc_object_diag::access_done(wc_sample);
                assert(!data.is_null());
                // Copy while the reader scope protects this payload; no far
                // pointer or accessor escapes to the serial consumer.
                items[i] = {data->key, data->val};
                wc_object_diag::finish(wc_sample);
            });
        const auto fetch_stop = std::chrono::steady_clock::now();
        fetch_s += std::chrono::duration<double>(fetch_stop - part_start).count();
        wc_object_diag::window("parallel_fetch", batches, part_start, fetch_stop);
        fetch_reads += profile::collect_rdma_read_post_count() - r0;
        fetch_read_bytes += profile::collect_rdma_read_post_bytes() - rb0;
        fetch_writes += profile::collect_rdma_write_post_count() - w0;
        fetch_write_bytes += profile::collect_rdma_write_post_bytes() - wb0;
        part_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < entries.size(); ++i) {
            fn(items[i].first, items[i].second);
        }
        callback_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - part_start).count();
        visited += entries.size();
        peak_items = std::max(peak_items, entries.size());
        ++batches;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    wc_object_diag::end_capture();
    std::fprintf(stderr,
        "wc_parallel_scan_diag readers=%zu visited=%zu batches=%zu "
        "batch_limit=%zu peak_items=%zu staging_capacity_bytes=%zu "
        "elapsed_s=%.9f gather_s=%.9f parallel_fetch_s=%.9f "
        "serial_callback_s=%.9f fetch_read_count=%ld fetch_read_bytes=%ld "
        "fetch_write_count=%ld fetch_write_bytes=%ld\n",
        readers, visited, batches, batch_limit, peak_items,
        items.capacity() * sizeof(Item) + entries.capacity() * sizeof(BucketEntry *),
        seconds, gather_s, fetch_s, callback_s, fetch_reads, fetch_read_bytes,
        fetch_writes, fetch_write_bytes);
}

template <typename Key, typename Val, typename Hash>
template <typename Fn>
inline void ConcurrentHashMap<Key, Val, Hash>::for_each_snapshot(
    Fn &&fn, __DMH__, DereferenceScope &scope) {
    static std::atomic_uint64_t diag_next_call{0};
    const bool diag_enabled =
        std::getenv("FARLIB_HASHMAP_SNAPSHOT_DIAG") != nullptr;
    const bool progress_enabled =
        std::getenv("FARLIB_HASHMAP_PROGRESS_DIAG") != nullptr;
    const bool phase_enabled =
        std::getenv("FARLIB_HASHMAP_PHASE_DIAG") != nullptr;
    auto env_u64 = [](const char *name, uint64_t fallback) {
        const char *value = std::getenv(name);
        return value ? std::strtoull(value, nullptr, 0) : fallback;
    };
    const uint64_t slow_phase_cycles = env_u64(
        "FARLIB_HASHMAP_PHASE_DIAG_SLOW_CYCLES", 1000000000ull);
    const uint64_t diag_call =
        (diag_enabled || progress_enabled || phase_enabled)
            ? diag_next_call.fetch_add(1, std::memory_order_relaxed)
            : 0;
    uint64_t visited = 0;
    uint64_t remote_seen = 0;
    uint64_t progress_start = progress_enabled ? __rdtsc() : 0;
    uint64_t progress_last = progress_start;
    auto log_phase = [&](const char *phase, uint32_t bucket, uint32_t offset,
                         uint64_t cycles, uint64_t extra0 = 0,
                         uint64_t extra1 = 0) {
        std::fprintf(stderr,
                     "hashmap_phase call=%lu map=%p phase=%s bucket=%u "
                     "offset=%u visited=%lu remote=%lu cycles=%lu extra0=%lu "
                     "extra1=%lu\n",
                     static_cast<unsigned long>(diag_call),
                     static_cast<void *>(this), phase, bucket,
                     static_cast<unsigned>(offset),
                     static_cast<unsigned long>(visited),
                     static_cast<unsigned long>(remote_seen),
                     static_cast<unsigned long>(cycles),
                     static_cast<unsigned long>(extra0),
                     static_cast<unsigned long>(extra1));
    };
    if (diag_enabled || progress_enabled || phase_enabled) {
        std::fprintf(stderr,
                     "hashmap_snapshot_diag begin call=%lu map=%p buckets=%u size=%zu\n",
                     static_cast<unsigned long>(diag_call),
                     static_cast<void *>(this), kNumEntries_,
                     size.load(std::memory_order_relaxed));
    }
    using Item = std::pair<Key, Val>;
    std::vector<Item> batch;
    batch.reserve(kNeighborhood);
    for (uint32_t bucket_idx = 0; bucket_idx <= kHashMask_; ++bucket_idx) {
        if (diag_enabled && ((bucket_idx & ((1u << 15) - 1)) == 0)) {
            std::fprintf(stderr,
                         "hashmap_snapshot_diag bucket call=%lu map=%p bucket=%u\n",
                         static_cast<unsigned long>(diag_call),
                         static_cast<void *>(this), bucket_idx);
        }
        if (progress_enabled && ((bucket_idx & ((1u << 15) - 1)) == 0)) {
            uint64_t now = __rdtsc();
            std::fprintf(stderr,
                         "hashmap_progress call=%lu map=%p bucket=%u visited=%lu remote=%lu delta_cycles=%lu total_cycles=%lu\n",
                         static_cast<unsigned long>(diag_call),
                         static_cast<void *>(this), bucket_idx,
                         static_cast<unsigned long>(visited),
                         static_cast<unsigned long>(remote_seen),
                         static_cast<unsigned long>(now - progress_last),
                         static_cast<unsigned long>(now - progress_start));
            progress_last = now;
        }
        BucketEntry *bucket = &(buckets_[bucket_idx]);
        uint64_t lock_start = phase_enabled ? __rdtsc() : 0;
        {
            SpinlockDiagScope lock_diag("hashmap_snapshot_bucket",
                                        reinterpret_cast<uintptr_t>(this),
                                        bucket_idx);
            bucket->spin.lock(scope);
        }
        if (phase_enabled) {
            uint64_t cycles = __rdtsc() - lock_start;
            if (cycles >= slow_phase_cycles) {
                log_phase("bucket_lock", bucket_idx, 0, cycles,
                          reinterpret_cast<uintptr_t>(&bucket->spin));
            }
        }
        batch.clear();
        uint32_t bitmap = bucket->load_bitmap();
        while (bitmap) {
            auto offset = __builtin_ctz(bitmap);
            BucketEntry *entry = &buckets_[bucket_idx + offset];
            if (!entry->ptr.is_null()) [[likely]] {
                ++visited;
                bool entry_at_local = cache::at_local(entry->ptr.obj());
                if (!entry_at_local) {
                    ++remote_seen;
                }
                if (diag_enabled && !entry_at_local) {
                    std::fprintf(stderr,
                                 "hashmap_snapshot_diag remote_before call=%lu map=%p bucket=%u offset=%u obj=%lu\n",
                                 static_cast<unsigned long>(diag_call),
                                 static_cast<void *>(this), bucket_idx,
                                 static_cast<unsigned>(offset),
                                 static_cast<unsigned long>(
                                     entry->ptr.obj().to_uint64_t()));
                }
                uint64_t access_start =
                    (progress_enabled || phase_enabled) ? __rdtsc() : 0;
                LiteAccessor<KVData> data;
                {
                    SpinlockDiagScope access_diag(
                        "hashmap_snapshot_access",
                        reinterpret_cast<uintptr_t>(this),
                        (static_cast<uint64_t>(bucket_idx) << 32) | offset);
                    data = access_entry(entry, __on_miss__, scope);
                }
                if (progress_enabled || phase_enabled) {
                    uint64_t access_cycles = __rdtsc() - access_start;
                    if (access_cycles >= slow_phase_cycles) {
                        if (progress_enabled) {
                            std::fprintf(stderr,
                                         "hashmap_slow_access call=%lu map=%p bucket=%u offset=%u obj=%lu was_local=%d cycles=%lu\n",
                                         static_cast<unsigned long>(diag_call),
                                         static_cast<void *>(this), bucket_idx,
                                         static_cast<unsigned>(offset),
                                         static_cast<unsigned long>(
                                             entry->ptr.obj().to_uint64_t()),
                                         entry_at_local ? 1 : 0,
                                         static_cast<unsigned long>(
                                             access_cycles));
                        }
                        if (phase_enabled) {
                            log_phase("entry_access", bucket_idx, offset,
                                      access_cycles,
                                      entry->ptr.obj().to_uint64_t(),
                                      entry_at_local ? 1 : 0);
                        }
                    }
                }
                if (diag_enabled && !cache::at_local(entry->ptr.obj())) {
                    std::fprintf(stderr,
                                 "hashmap_snapshot_diag remote_after call=%lu map=%p bucket=%u offset=%u obj=%lu null=%d\n",
                                 static_cast<unsigned long>(diag_call),
                                 static_cast<void *>(this), bucket_idx,
                                 static_cast<unsigned>(offset),
                                 static_cast<unsigned long>(
                                     entry->ptr.obj().to_uint64_t()),
                                 data.is_null() ? 1 : 0);
                }
                if (!data.is_null()) {
                    batch.emplace_back(data->key, data->val);
                }
            }
            bitmap ^= (1 << offset);
        }
        bucket->spin.unlock();
        for (auto &item : batch) {
            uint64_t callback_start = phase_enabled ? __rdtsc() : 0;
            {
                SpinlockDiagScope callback_diag(
                    "hashmap_snapshot_callback",
                    reinterpret_cast<uintptr_t>(this), bucket_idx);
                fn(item.first, item.second, scope);
            }
            if (phase_enabled) {
                uint64_t cycles = __rdtsc() - callback_start;
                if (cycles >= slow_phase_cycles) {
                    log_phase("callback", bucket_idx, 0, cycles, batch.size());
                }
            }
        }
    }
    if (diag_enabled || progress_enabled || phase_enabled) {
        std::fprintf(stderr,
                     "hashmap_snapshot_diag end call=%lu map=%p visited=%lu remote=%lu\n",
                     static_cast<unsigned long>(diag_call),
                     static_cast<void *>(this),
                     static_cast<unsigned long>(visited),
                     static_cast<unsigned long>(remote_seen));
    }
}

}  // namespace FarLib
