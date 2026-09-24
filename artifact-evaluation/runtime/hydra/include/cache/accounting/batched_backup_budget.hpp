#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace FarLib::cache {

// Strict capacity admission with cached credits. At all times:
// actual reservations + unused credits = issued <= limit.
// Acquiring consumes private credits; releasing can credit a different owner.
class BatchedBackupBudget {
    struct alignas(128) Account {
        std::atomic_flag busy = ATOMIC_FLAG_INIT;
        uint64_t credit = 0;
        // Modular deltas allow arbitrary cross-owner release and lifetime
        // turnover without signed overflow. Only the aggregate is a byte count.
        uint64_t net = 0;
        uint64_t acquires = 0, releases = 0, grants = 0, refunds = 0;
        uint64_t grant_attempts = 0, peak_attempts = 0, scans = 0;
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
    struct State {
        const uint64_t limit, quantum;
        alignas(128) std::atomic<uint64_t> issued{0};
        alignas(128) std::atomic<uint64_t> peak{0};
        mutable std::mutex registry;
        std::vector<std::unique_ptr<Account>> accounts;
        State(uint64_t cap, uint64_t q) : limit(cap), quantum(q) {
            assert(q != 0 && q <= std::numeric_limits<uint64_t>::max() / 2);
            assert(cap <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
        }
        Account *attach() {
            auto a = std::make_unique<Account>();
            auto *raw = a.get();
            std::lock_guard<std::mutex> guard(registry);
            accounts.push_back(std::move(a));
            return raw;
        }
        // Caller owns a's lock. Shared updates occur only in these grant/refund paths.
        bool grant(Account &a, uint64_t need) {
            uint64_t current = issued.load(std::memory_order_relaxed);
            for (;;) {
                assert(current <= limit);
                if (limit - current < need) return false;
                const uint64_t amount = std::min(limit - current, std::max(quantum, need));
                ++a.grant_attempts;
                if (issued.compare_exchange_weak(current, current + amount,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    a.credit += amount;
                    ++a.grants;
                    auto high = peak.load(std::memory_order_relaxed);
                    while (high < current + amount) {
                        ++a.peak_attempts;
                        if (peak.compare_exchange_weak(high, current + amount,
                                std::memory_order_relaxed, std::memory_order_relaxed)) break;
                    }
                    return true;
                }
            }
        }
        void refund(Account &a, uint64_t amount) {
            if (!amount) return;
            assert(amount <= a.credit);
            a.credit -= amount;
            const auto previous = issued.fetch_sub(amount, std::memory_order_relaxed);
            assert(previous >= amount);
            ++a.refunds;
        }
        void retire(Account *a) {
            if (!a) return;
            a->lock();
            refund(*a, a->credit);
            a->unlock();
        }
        void reclaim_idle() {
            // No caller may enter with an Account locked. Registration and
            // snapshots always use registry -> account order, never the reverse.
            std::lock_guard<std::mutex> guard(registry);
            for (auto &a : accounts) {
                a->lock();
                refund(*a, a->credit);
                a->unlock();
            }
        }
    };
public:
    struct Snapshot {
        int64_t actual_bytes = 0;
        uint64_t issued_bytes = 0, cached_bytes = 0, peak_issued_bytes = 0;
        uint64_t acquire_count = 0, release_count = 0, grant_count = 0, refund_count = 0;
        uint64_t grant_cas_attempts = 0, peak_cas_attempts = 0, pressure_scans = 0, handles = 0;
    };
    class Handle {
        friend class BatchedBackupBudget;
        std::shared_ptr<State> owner;
        Account *account = nullptr;
    public:
        Handle() = default;
        Handle(const Handle &) = delete;
        Handle &operator=(const Handle &) = delete;
        ~Handle() { if (owner) owner->retire(account); }
    };
private:
    std::shared_ptr<State> state;
    Account &local(Handle &handle) {
        // No shared_ptr refcount operations on the steady-state hot path.
        if (handle.owner.get() != state.get()) {
            Account *next = state->attach();
            if (handle.owner) handle.owner->retire(handle.account);
            handle.owner = state;
            handle.account = next;
        }
        return *handle.account;
    }
public:
    explicit BatchedBackupBudget(uint64_t limit, uint64_t quantum = 256 * 1024)
        : state(std::make_shared<State>(limit, quantum)) {}
    bool acquire(Handle &handle, uint64_t bytes) {
        if (bytes > state->limit) return false;
        auto &a = local(handle);
        for (unsigned attempt = 0; attempt != 2; ++attempt) {
            a.lock();
            if (a.credit >= bytes || state->grant(a, bytes - a.credit)) {
                a.credit -= bytes;
                a.net += bytes;
                ++a.acquires;
                a.unlock();
                return true;
            }
            if (attempt == 0) ++a.scans;
            a.unlock();
            if (attempt == 0) state->reclaim_idle();
        }
        return false;
    }
    void release(Handle &handle, uint64_t bytes) {
        auto &a = local(handle);
        a.lock();
        a.credit += bytes;
        a.net -= bytes;
        ++a.releases;
        // Retain at most one quantum after each flush. This also handles a
        // single object larger than the quantum without stranding its credits.
        if (a.credit >= state->quantum * 2)
            state->refund(a, a.credit - state->quantum);
        a.unlock();
    }
    void reclaim_idle() { state->reclaim_idle(); }
    Snapshot snapshot() const {
        Snapshot out;
        uint64_t actual = 0;
        std::lock_guard<std::mutex> guard(state->registry);
        for (auto &a : state->accounts) a->lock();
        for (auto &a : state->accounts) {
            actual += a->net;
            out.cached_bytes += a->credit;
            out.acquire_count += a->acquires;
            out.release_count += a->releases;
            out.grant_count += a->grants;
            out.refund_count += a->refunds;
            out.grant_cas_attempts += a->grant_attempts;
            out.peak_cas_attempts += a->peak_attempts;
            out.pressure_scans += a->scans;
        }
        out.handles = state->accounts.size();
        out.issued_bytes = state->issued.load(std::memory_order_relaxed);
        out.peak_issued_bytes = state->peak.load(std::memory_order_relaxed);
        assert(actual <= state->limit);
        out.actual_bytes = static_cast<int64_t>(actual);
        assert(out.actual_bytes >= 0);
        assert(static_cast<uint64_t>(out.actual_bytes) + out.cached_bytes == out.issued_bytes);
        assert(out.issued_bytes <= state->limit && out.peak_issued_bytes <= state->limit);
        for (auto it = state->accounts.rbegin(); it != state->accounts.rend(); ++it)
            (*it)->unlock();
        return out;
    }
};

// Acquisitions run on the same fibre context as remote fetch/local allocation.
// Releases also run on native planner threads: their adapter must not call
// fibre_self(). Both Handle lifetimes may safely extend beyond the Cache.
BatchedBackupBudget::Handle &current_backup_budget_handle();
BatchedBackupBudget::Handle &current_backup_thread_handle();
} // namespace FarLib::cache
