#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace FarLib::recompute {

// A recipe callback rebuilds an immutable object into destination.  Returning
// false reports that the callback could not produce the requested bytes.
using EntryFn = bool (*)(void *destination, size_t bytes, const void *context,
                         uint64_t arg);

// The state shared by all Inputs wrappers for one immutable input set.  Entry
// bindings retain this object independently of the application wrapper, so a
// callback's context remains alive until every bound entry releases it.
struct InputState {
    std::shared_ptr<const void> context;
    std::atomic<size_t> refs{1};
    std::atomic<EntryFn> callback{nullptr};

    explicit InputState(std::shared_ptr<const void> context)
        : context(std::move(context)) {}

    InputState(const InputState &) = delete;
    InputState &operator=(const InputState &) = delete;

    void retain() noexcept {
        refs.fetch_add(1, std::memory_order_relaxed);
    }

    void release() noexcept {
        if (refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

    // The first binding fixes the callback for this input set.  Subsequent
    // bindings may reuse exactly that function, but a different function is
    // rejected without changing the registered callback.
    bool register_callback(EntryFn fn) noexcept {
        if (fn == nullptr) return false;
        EntryFn expected = nullptr;
        if (callback.compare_exchange_strong(
                expected, fn, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return true;
        }
        return expected == fn;
    }

    bool invoke(void *destination, size_t bytes, uint64_t arg) const {
        EntryFn fn = callback.load(std::memory_order_acquire);
        return fn != nullptr && fn(destination, bytes, context.get(), arg);
    }

private:
    ~InputState() = default;
};

using InputSet = InputState;

class Inputs {
private:
    struct AdoptRefTag {};

    InputState *state_ = nullptr;

    explicit Inputs(InputState *state, AdoptRefTag) noexcept : state_(state) {}

public:
    Inputs() noexcept = default;

    Inputs(const Inputs &other) noexcept : state_(other.state_) {
        if (state_ != nullptr) state_->retain();
    }

    Inputs(Inputs &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}

    Inputs &operator=(const Inputs &other) noexcept {
        if (this == &other) return *this;
        Inputs copy(other);
        swap(copy);
        return *this;
    }

    Inputs &operator=(Inputs &&other) noexcept {
        if (this == &other) return *this;
        reset();
        state_ = std::exchange(other.state_, nullptr);
        return *this;
    }

    ~Inputs() { reset(); }

    template <typename T>
    static Inputs immutable_compute_owned(std::shared_ptr<const T> context) {
        if (!context) return {};
        std::shared_ptr<const void> erased(std::move(context));
        return Inputs(new InputState(std::move(erased)), AdoptRefTag{});
    }

    // Convenience overload for callers that still hold a mutable shared_ptr;
    // the InputState stores it as shared_ptr<const T> and exposes only const
    // context to the callback.
    template <typename T>
        requires(!std::is_const_v<T>)
    static Inputs immutable_compute_owned(std::shared_ptr<T> context) {
        return immutable_compute_owned(
            std::shared_ptr<const T>(std::move(context)));
    }

    bool valid() const noexcept { return state_ != nullptr; }
    bool is_valid() const noexcept { return valid(); }
    explicit operator bool() const noexcept { return valid(); }

    InputState *state() const noexcept { return state_; }
    InputState *input_state() const noexcept { return state_; }
    InputState *owner() const noexcept { return state_; }

    void reset() noexcept {
        InputState *state = std::exchange(state_, nullptr);
        if (state != nullptr) state->release();
    }

    void swap(Inputs &other) noexcept { std::swap(state_, other.state_); }
};

// A Binding is deliberately a trivially destructible, non-owning pair.  The
// owning reference is held by the entry's InputState pointer, not by Binding.
struct Binding {
    InputState *inputs = nullptr;
    uint64_t arg = 0;
};

static_assert(sizeof(Binding) == 16);
static_assert(std::is_trivially_destructible_v<Binding>);

}  // namespace FarLib::recompute
