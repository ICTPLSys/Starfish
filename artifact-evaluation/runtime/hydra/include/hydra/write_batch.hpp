#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace FarLib::hydra {

// A request is only copied into the worker-owned batch.  The local address is
// never dereferenced by this container; the RDMA posting path owns its lifetime
// and registration checks.
struct PendingWrite {
    void *local_addr = nullptr;
    uint64_t remote_offset = 0;
    uint64_t wr_id = 0;
    uint32_t bytes = 0;
    uint32_t lkey = 0;
};

// One logical evacuator worker owns one WriteBatch.  There is deliberately no
// lock and no completion-side mutation here: the producer drains an endpoint
// before reusing it, while completions only consume the WRs that were copied to
// the transport by the caller.
class WriteBatch final {
public:
    static constexpr std::size_t kCapacity = 64;
    static constexpr std::size_t kUnboundContext =
        std::numeric_limits<std::size_t>::max();

    struct Endpoint {
        std::array<PendingWrite, kCapacity> requests{};
        std::size_t count = 0;
    };

    WriteBatch() = default;
    ~WriteBatch() = default;

    WriteBatch(const WriteBatch &) = delete;
    WriteBatch &operator=(const WriteBatch &) = delete;
    WriteBatch(WriteBatch &&) = delete;
    WriteBatch &operator=(WriteBatch &&) = delete;

    // Initialization is intentionally one-shot.  A second call throws even
    // after clear_endpoint(), so no caller can silently discard a batch by
    // replacing its backing array.
    void init(std::size_t endpoint_count) {
        if (initialized_) {
            throw std::logic_error("hydra::WriteBatch is already initialized");
        }
        if (endpoint_count == 0) {
            throw std::invalid_argument(
                "hydra::WriteBatch requires at least one endpoint");
        }

        auto endpoints = std::make_unique<Endpoint[]>(endpoint_count);
        endpoints_ = std::move(endpoints);
        endpoint_count_ = endpoint_count;
        initialized_ = true;
        context_bound_ = false;
        client_idx_ = kUnboundContext;
        qp_idx_ = kUnboundContext;
    }

    bool initialized() const noexcept { return initialized_; }

    std::size_t endpoint_count() const noexcept {
        return initialized_ ? endpoint_count_ : 0;
    }

    // A context may be changed only while no request is pending.  Rebinding to
    // the same context is always harmless and succeeds, including while the
    // endpoints contain requests.
    bool bind_context(std::size_t client_idx,
                      std::size_t qp_idx) noexcept {
        if (!initialized_ || client_idx == kUnboundContext ||
            qp_idx == kUnboundContext) {
            return false;
        }

        const bool same_context =
            context_bound_ && client_idx_ == client_idx && qp_idx_ == qp_idx;
        if (!same_context && pending() != 0) {
            return false;
        }

        client_idx_ = client_idx;
        qp_idx_ = qp_idx;
        context_bound_ = true;
        return true;
    }

    std::size_t client_idx() const noexcept {
        return context_bound_ ? client_idx_ : kUnboundContext;
    }

    std::size_t qp_idx() const noexcept {
        return context_bound_ ? qp_idx_ : kUnboundContext;
    }

    // Invalid endpoints and a full endpoint are ordinary posting refusals; no
    // request is copied and count is unchanged in either case.
    bool enqueue(std::size_t endpoint_index,
                 const PendingWrite &request) noexcept {
        if (!initialized_ || endpoint_index >= endpoint_count_) {
            return false;
        }
        Endpoint &target = endpoints_[endpoint_index];
        if (target.count >= kCapacity) {
            return false;
        }
        target.requests[target.count] = request;
        ++target.count;
        return true;
    }

    Endpoint &endpoint(std::size_t endpoint_index) {
        return checked_endpoint(endpoint_index);
    }

    const Endpoint &endpoint(std::size_t endpoint_index) const {
        return checked_endpoint(endpoint_index);
    }

    void clear_endpoint(std::size_t endpoint_index) {
        checked_endpoint(endpoint_index).count = 0;
    }

    std::size_t pending() const noexcept {
        if (!initialized_) return 0;
        std::size_t total = 0;
        for (std::size_t i = 0; i < endpoint_count_; ++i) {
            total += endpoints_[i].count;
        }
        return total;
    }

private:
    Endpoint &checked_endpoint(std::size_t endpoint_index) {
        if (!initialized_ || endpoint_index >= endpoint_count_) {
            throw std::out_of_range("hydra::WriteBatch endpoint index");
        }
        return endpoints_[endpoint_index];
    }

    const Endpoint &checked_endpoint(std::size_t endpoint_index) const {
        if (!initialized_ || endpoint_index >= endpoint_count_) {
            throw std::out_of_range("hydra::WriteBatch endpoint index");
        }
        return endpoints_[endpoint_index];
    }

    std::unique_ptr<Endpoint[]> endpoints_;
    std::size_t endpoint_count_ = 0;
    std::size_t client_idx_ = kUnboundContext;
    std::size_t qp_idx_ = kUnboundContext;
    bool initialized_ = false;
    bool context_bound_ = false;
};

// Return the number of requests accepted before ibv_post_send's bad_wr.
// `wr_array` is the base address of a contiguous array of `count` request
// objects, and `first` is the first request still being posted after earlier
// partial progress.  A successful post has no bad_wr and therefore needs no
// pointer validation: all [first,count) requests were accepted.  On failure,
// bad_wr must identify an aligned element in [first,count); otherwise the
// result is SIZE_MAX so the caller cannot silently treat an ambiguous failure
// as zero progress.
inline std::size_t accepted_write_prefix(std::uintptr_t wr_array,
                                         std::size_t stride,
                                         std::size_t count,
                                         std::size_t first,
                                         std::uintptr_t bad_wr,
                                         bool success) noexcept {
    const std::size_t kInvalid = std::numeric_limits<std::size_t>::max();
    if (first > count) return kInvalid;
    if (success) return count - first;
    if (first == count || wr_array == 0 || stride == 0) return kInvalid;

    // Check both the multiplication and the base-plus-span before forming
    // any address.  The end address is one-past-the-array and is not itself a
    // valid bad_wr location.
    const std::uintptr_t kAddressMax =
        std::numeric_limits<std::uintptr_t>::max();
    if (count > std::numeric_limits<std::size_t>::max() / stride ||
        count > kAddressMax / stride) {
        return kInvalid;
    }
    const std::uintptr_t span = static_cast<std::uintptr_t>(count) *
                                static_cast<std::uintptr_t>(stride);
    if (wr_array > kAddressMax - span) return kInvalid;
    const std::uintptr_t end = wr_array + span;

    if (bad_wr < wr_array || bad_wr >= end) return kInvalid;
    const std::uintptr_t offset = bad_wr - wr_array;
    if (offset % stride != 0) return kInvalid;
    const std::uintptr_t index_wide = offset / stride;
    if (index_wide > std::numeric_limits<std::size_t>::max()) {
        return kInvalid;
    }
    const std::size_t index = static_cast<std::size_t>(index_wide);
    if (index < first || index >= count) return kInvalid;
    return index - first;
}

}  // namespace FarLib::hydra
