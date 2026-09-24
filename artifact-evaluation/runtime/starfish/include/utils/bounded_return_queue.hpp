#pragma once
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace FarLib {
// Owns pointers, not intrusive links. No callback, allocation or yielding while
// the ingress lock is held. A successful take transfers sole ownership to the
// consumer until it calls complete() after synchronous list publication.
template <class T, size_t Capacity>
class BoundedReturnQueue {
    std::array<T *, Capacity> slots_{};
    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    size_t head_ = 0, tail_ = 0;
    std::atomic<size_t> pending_{0}, inflight_{0};
    std::atomic<uint64_t> accepted_{0}, published_{0}, fallback_{0};
    std::atomic<size_t> high_water_{0};
    void lock() {
        while (lock_.test_and_set(std::memory_order_acquire)) {
            __builtin_ia32_pause();
        }
    }
    void unlock() { lock_.clear(std::memory_order_release); }
public:
    struct Snapshot {
        uint64_t accepted, published, fallback;
        size_t pending, inflight, high_water;
    };
    // All-or-nothing ownership transfer. On failure the caller still owns
    // every item. A requested wakeup must happen only after this call returns.
    bool enqueue(T **items, size_t n, bool &wake) {
        assert(n && n <= Capacity);
        wake = false;
        lock();
        const size_t current = pending_.load(std::memory_order_relaxed);
        if (n > Capacity - current) {
            fallback_.fetch_add(n, std::memory_order_relaxed);
            unlock();
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            assert(items[i]);
            slots_[tail_] = items[i];
            tail_ = (tail_ + 1) % Capacity;
        }
        accepted_.fetch_add(n, std::memory_order_relaxed);
        if (current + n > high_water_.load(std::memory_order_relaxed))
            high_water_.store(current + n, std::memory_order_relaxed);
        pending_.store(current + n, std::memory_order_release);
        wake = current == 0;
        unlock();
        return true;
    }
    // The consumer owns the returned items outside the ingress lock. Publish
    // them to their destination lists before calling complete().
    size_t take(T **out, size_t limit) {
        lock();
        const size_t current = pending_.load(std::memory_order_relaxed);
        const size_t n = current < limit ? current : limit;
        for (size_t i = 0; i < n; ++i) {
            out[i] = slots_[head_];
            slots_[head_] = nullptr;
            head_ = (head_ + 1) % Capacity;
        }
        inflight_.fetch_add(n, std::memory_order_relaxed);
        pending_.store(current - n, std::memory_order_release);
        unlock();
        return n;
    }
    void complete(size_t n) {
        published_.fetch_add(n, std::memory_order_relaxed);
        const size_t previous = inflight_.fetch_sub(n, std::memory_order_release);
        assert(previous >= n);
    }
    // Pending excludes items already taken by consumers; it is not an idle
    // check. Shutdown must join producers/consumers before testing conservation.
    bool pending() const { return pending_.load(std::memory_order_acquire) != 0; }
    // A live snapshot is approximate; conservation is asserted only quiescently.
    Snapshot snapshot() const {
        return {accepted_.load(), published_.load(), fallback_.load(),
                pending_.load(), inflight_.load(), high_water_.load()};
    }
};
}
