#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <utility>

namespace FarLib::rdma::read_batch {

enum class QueueStatus { Ok, Busy, Full };

// Fixed-capacity prefix queue. PostFn receives the current ordered prefix and
// returns the number accepted by the transport. Only that accepted prefix is
// removed; an unaccepted suffix remains in order for a later retry.
template <typename Request, std::size_t Capacity = 2>
class PrefixQueue {
public:
    static_assert(Capacity > 0);
    struct Result {
        QueueStatus status = QueueStatus::Ok;
        bool enqueued = false;
        bool submit_attempted = false;
        bool submit_suppressed = false;
        std::size_t accepted_count = 0;
        std::array<Request, Capacity> accepted{};
        std::size_t pending_after = 0;
    };

    template <typename PostFn>
    Result try_enqueue(const Request &request, PostFn &&post) {
        return try_enqueue_observed(
            request, std::forward<PostFn>(post),
            [](const Request &) {});
    }

    template <typename PostFn, typename EnqueueFn>
    Result try_enqueue_observed(const Request &request, PostFn &&post,
                                EnqueueFn &&on_enqueue) {
        return try_enqueue_controlled_at(
            request, Capacity,
            [](const Request *, std::size_t) { return true; },
            std::forward<PostFn>(post), std::forward<EnqueueFn>(on_enqueue));
    }

    template <typename SubmitIfFn, typename PostFn, typename EnqueueFn>
    Result try_enqueue_controlled(const Request &request,
                                  SubmitIfFn &&submit_if, PostFn &&post,
                                  EnqueueFn &&on_enqueue) {
        return try_enqueue_controlled_at(
            request, Capacity, std::forward<SubmitIfFn>(submit_if),
            std::forward<PostFn>(post), std::forward<EnqueueFn>(on_enqueue));
    }

    template <typename SubmitIfFn, typename PostFn, typename EnqueueFn>
    Result try_enqueue_controlled_at(const Request &request,
                                     std::size_t submit_at_count,
                                     SubmitIfFn &&submit_if, PostFn &&post,
                                     EnqueueFn &&on_enqueue) {
        if (submit_at_count == 0 || submit_at_count > Capacity) std::abort();
        TryGuard guard(lock_);
        if (!guard.owns_lock()) {
            return {.status = QueueStatus::Busy};
        }
        if (count_ >= submit_at_count) {
            return {.status = QueueStatus::Full,
                    .pending_after = count_};
        }
        requests_[count_++] = request;
        on_enqueue(request);
        Result result;
        result.enqueued = true;
        if (count_ >= submit_at_count) {
            if (submit_if(requests_.data(), count_)) {
                submit_locked(result, std::forward<PostFn>(post));
            } else {
                result.submit_suppressed = true;
                result.pending_after = count_;
            }
        } else {
            result.pending_after = count_;
        }
        return result;
    }

    template <typename SubmitIfFn, typename PostFn>
    Result try_flush_if(SubmitIfFn &&submit_if, PostFn &&post) {
        TryGuard guard(lock_);
        if (!guard.owns_lock()) {
            return {.status = QueueStatus::Busy};
        }
        Result result;
        if (count_ != 0) {
            if (submit_if(requests_.data(), count_)) {
                submit_locked(result, std::forward<PostFn>(post));
            } else {
                result.submit_suppressed = true;
                result.pending_after = count_;
            }
        }
        return result;
    }

    template <typename PostFn>
    Result try_flush(PostFn &&post) {
        TryGuard guard(lock_);
        if (!guard.owns_lock()) {
            return {.status = QueueStatus::Busy};
        }
        Result result;
        if (count_ != 0) {
            submit_locked(result, std::forward<PostFn>(post));
        } else {
            result.pending_after = 0;
        }
        return result;
    }

    Result try_observe() const {
        TryGuard guard(lock_);
        if (!guard.owns_lock()) {
            return {.status = QueueStatus::Busy};
        }
        return {.status = QueueStatus::Ok,
                .pending_after = count_};
    }

    template <typename InspectFn>
    Result try_inspect(InspectFn &&inspect) const {
        TryGuard guard(lock_);
        if (!guard.owns_lock()) {
            return {.status = QueueStatus::Busy};
        }
        inspect(count_);
        return {.status = QueueStatus::Ok,
                .pending_after = count_};
    }


    template <typename InspectFn>
    Result try_inspect_items(InspectFn &&inspect) const {
        TryGuard guard(lock_);
        if (!guard.owns_lock()) {
            return {.status = QueueStatus::Busy};
        }
        inspect(requests_.data(), count_);
        return {.status = QueueStatus::Ok,
                .pending_after = count_};
    }

private:
    template <typename PostFn>
    void submit_locked(Result &result, PostFn &&post) {
        result.submit_attempted = true;
        const std::size_t accepted = post(requests_.data(), count_);
        if (accepted > count_) {
            std::abort();
        }
        result.accepted_count = accepted;
        for (std::size_t i = 0; i < accepted && i < Capacity; ++i) {
            result.accepted[i] = requests_[i];
        }
        for (std::size_t i = accepted; i < count_ && i < Capacity; ++i) {
            requests_[i - accepted] = requests_[i];
        }
        count_ -= accepted;
        result.pending_after = count_;
    }

    class TryGuard {
    public:
        explicit TryGuard(std::atomic_flag &lock)
            : lock_(lock), owns_(!lock_.test_and_set(std::memory_order_acquire)) {}
        ~TryGuard() {
            if (owns_) lock_.clear(std::memory_order_release);
        }
        bool owns_lock() const { return owns_; }
    private:
        std::atomic_flag &lock_;
        bool owns_;
    };

    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    std::array<Request, Capacity> requests_{};
    std::size_t count_ = 0;
};

}  // namespace FarLib::rdma::read_batch
