#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <array>
#include <vector>

#include "cache/alloc/ec_batch_staging.hpp"

namespace FarLib::cache::ec_read_recovery {

// The adapter owns allocation and MR registration. Keeping verbs outside this
// class permits CPU tests to inject registration failure and track ownership.
// This remains the callback type used by the old recovery-only pool.
struct RecoveryScratchChunk {
    void *base = nullptr;
    uint32_t lkey = 0;
    void *registration = nullptr;
};

// Grow-on-demand registered temporary buffers shared by EC recovery reads and
// EC batch eviction writes.  Storage is partitioned by owner worker: a worker
// borrows from its own free list, while a completion thread returns a lease by
// pushing a stable node onto that owner's lock-free MPSC return stack.  The
// next borrow by the owner drains that stack under only its own mutex.
class WorkerTempBufferPool final : public ::FarLib::cache::WorkerTempBufferProvider {
private:
    struct OwnerState;
    struct Node;
    static constexpr size_t kFastOwnerSlots = 1024;
    static constexpr uint64_t kMaxLeaseGeneration =
        std::numeric_limits<uint64_t>::max() >> 1;
    static constexpr uint64_t kLeaseActiveBit = 1;

public:
    using Allocate = bool (*)(void *, size_t, RecoveryScratchChunk *);
    using Release = void (*)(void *, RecoveryScratchChunk);

    ~WorkerTempBufferPool() override {
        if (in_use_.load(std::memory_order_acquire) != 0) std::abort();

        std::vector<OwnerState *> owners;
        {
            std::lock_guard<std::mutex> lock(owner_init_mutex_);
            owners.reserve(owners_.size());
            for (const auto &owner : owners_) owners.push_back(owner.get());
        }
        for (OwnerState *owner : owners) {
            std::lock_guard<std::mutex> lock(owner->mutex);
            drain_returns_locked(*owner);
            for (const auto &chunk : owner->chunks) {
                release_(context_, chunk.chunk);
            }
        }
    }

    bool init(size_t slot_size, void *context, Allocate allocate,
              Release release, size_t explicit_limit = 0,
              size_t max_slot_size = ec_batch::kEcBatchStagingMaxSlotSize) {
        if (valid_ || slot_size == 0 ||
            max_slot_size == 0 || slot_size > max_slot_size ||
            max_slot_size > std::numeric_limits<uint32_t>::max() ||
            allocate == nullptr || release == nullptr) {
            return false;
        }
        slot_size_ = slot_size;
        context_ = context;
        allocate_ = allocate;
        release_ = release;
        limit_ = explicit_limit;
        max_slot_size_ = max_slot_size;
        valid_ = true;
        return true;
    }

    bool valid() const override { return valid_; }
    size_t slot_size() const override { return slot_size_; }
    size_t limit() const override { return limit_; }
    size_t depth() const override {
        return depth_.load(std::memory_order_acquire);
    }
    size_t bytes() const override {
        return bytes_.load(std::memory_order_acquire);
    }
    size_t in_use() const override {
        return in_use_.load(std::memory_order_acquire);
    }
    size_t peak_in_use() const override {
        return peak_.load(std::memory_order_acquire);
    }
    size_t growths() const override {
        return growths_.load(std::memory_order_acquire);
    }
    size_t allocation_failures() const override {
        return failures_.load(std::memory_order_acquire);
    }
    size_t available() const override {
        const size_t depth = depth_.load(std::memory_order_acquire);
        const size_t in_use = in_use_.load(std::memory_order_acquire);
        const size_t retired = retired_.load(std::memory_order_acquire);
        const size_t unavailable = in_use + retired;
        return depth > unavailable ? depth - unavailable : 0;
    }

    bool acquire(ec_batch::EcStagingGroupSlot *out,
                 size_t owner_idx = 0) override {
        if (out == nullptr || !valid_) return false;
        OwnerState *owner = nullptr;
        try {
            owner = owner_for(owner_idx);
        } catch (const std::bad_alloc &) {
            failures_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        std::lock_guard<std::mutex> lock(owner->mutex);
        drain_returns_locked(*owner);
        for (;;) {
            while (!owner->free.empty()) {
                Node *node = owner->free.back();
                owner->free.pop_back();
                const uint64_t state = node->lease_state.load(
                    std::memory_order_relaxed);
                uint64_t generation = state >> 1;
                if (generation >= kMaxLeaseGeneration) {
                    node->retired = true;
                    retired_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                ++generation;
                node->lease_state.store((generation << 1) | kLeaseActiveBit,
                                        std::memory_order_release);
                *out = node->value;
                out->generation = generation;
                const size_t live =
                    in_use_.fetch_add(1, std::memory_order_acq_rel) + 1;
                update_peak(live);
                return true;
            }
            if (!grow_locked(*owner)) return false;
            // The growth operation appends free nodes.  Loop so generation
            // retirement and return-stack draining share one path.
        }
    }

    // Completion-side release intentionally does not lock either the owner or
    // a global pool mutex.  It validates the immutable lease identity, flips
    // active with CAS (rejecting stale/double release), then publishes the
    // node onto the owner's MPSC return list.
    bool release(const ec_batch::EcStagingGroupSlot &value) override {
        Node *node = nullptr;
        if (!lookup_active_node(value, &node)) return false;

        uint64_t expected = (value.generation << 1) | kLeaseActiveBit;
        if (!node->lease_state.compare_exchange_strong(
                expected, value.generation << 1, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return false;
        }
        in_use_.fetch_sub(1, std::memory_order_acq_rel);
        const uint64_t generation = value.generation;
        if (generation >= kMaxLeaseGeneration) {
            node->retired = true;
            retired_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        OwnerState *owner = node->owner;
        Node *head = owner->returns.load(std::memory_order_relaxed);
        do {
            node->return_next.store(head, std::memory_order_relaxed);
        } while (!owner->returns.compare_exchange_weak(
            head, node, std::memory_order_release, std::memory_order_relaxed));
        return true;
    }

    bool owns_slot(const ec_batch::EcStagingGroupSlot &value) const override {
        Node *node = nullptr;
        return lookup_active_node(value, &node);
    }

    bool owns_buffer(const ec_batch::EcStagingGroupSlot &value,
                     const void *ptr, size_t bytes) const override {
        if (ptr == nullptr || bytes == 0) return false;
        Node *node = nullptr;
        if (!lookup_active_node(value, &node)) return false;
        const uintptr_t q = reinterpret_cast<uintptr_t>(ptr);
        for (size_t i = 0; i < ec_batch::kEcBatchDataSlots; ++i) {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(value.data[i]);
            if (q >= begin && q - begin <= value.slot_size &&
                bytes <= value.slot_size - (q - begin)) {
                return true;
            }
        }
        for (size_t i = 0; i < ec_batch::kEcBatchParitySlots; ++i) {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(value.parity[i]);
            if (q >= begin && q - begin <= value.slot_size &&
                bytes <= value.slot_size - (q - begin)) {
                return true;
            }
        }
        return false;
    }

private:
    struct Node {
        WorkerTempBufferPool *pool = nullptr;
        OwnerState *owner = nullptr;
        ec_batch::EcStagingGroupSlot value{};
        // state = (generation << 1) | active.  A single CAS on this word
        // makes a stale lease unable to release a reused node.
        std::atomic<uint64_t> lease_state{0};
        std::atomic<Node *> return_next{nullptr};
        bool retired = false;
    };

    struct ChunkRecord {
        RecoveryScratchChunk chunk{};
        size_t count = 0;
    };

    struct OwnerState {
        explicit OwnerState(size_t id) : owner_id(id) {}

        size_t owner_id = 0;
        mutable std::mutex mutex;
        std::atomic<Node *> returns{nullptr};
        std::vector<Node *> free;
        std::vector<std::unique_ptr<Node>> nodes;
        std::vector<ChunkRecord> chunks;
    };

    OwnerState *owner_for(size_t owner_idx) {
        if (owner_idx < kFastOwnerSlots) {
            OwnerState *owner =
                fast_owners_[owner_idx].load(std::memory_order_acquire);
            if (owner != nullptr) return owner;
            std::lock_guard<std::mutex> lock(owner_init_mutex_);
            owner = fast_owners_[owner_idx].load(std::memory_order_relaxed);
            if (owner == nullptr) {
                owners_.push_back(std::make_unique<OwnerState>(owner_idx));
                owner = owners_.back().get();
                fast_owners_[owner_idx].store(owner, std::memory_order_release);
            }
            return owner;
        }

        // Worker ids normally fit the fast table.  The overflow path keeps
        // the API unbounded while paying the initialization mutex only for a
        // genuinely new, out-of-range owner id.
        std::lock_guard<std::mutex> lock(owner_init_mutex_);
        for (const auto &owner : owners_) {
            if (owner->owner_id == owner_idx) return owner.get();
        }
        owners_.push_back(std::make_unique<OwnerState>(owner_idx));
        return owners_.back().get();
    }

    void drain_returns_locked(OwnerState &owner) const {
        Node *node = owner.returns.exchange(nullptr, std::memory_order_acquire);
        while (node != nullptr) {
            Node *next = node->return_next.load(std::memory_order_relaxed);
            if (!node->retired) owner.free.push_back(node);
            node = next;
        }
    }

    bool reserve_depth(size_t count, size_t *granted_out) {
        if (granted_out == nullptr) return false;
        size_t current = depth_.load(std::memory_order_acquire);
        for (;;) {
            if (limit_ != 0 && current >= limit_) return false;
            size_t grant = count;
            if (limit_ != 0) grant = std::min(grant, limit_ - current);
            if (grant == 0) return false;
            if (depth_.compare_exchange_weak(current, current + grant,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
                *granted_out = grant;
                return true;
            }
        }
    }

    void rollback_depth(size_t count) {
        depth_.fetch_sub(count, std::memory_order_acq_rel);
    }

    bool grow_locked(OwnerState &owner) {
        const size_t before = owner.nodes.size();
        size_t count = before == 0 ? 8 : std::min<size_t>(before, 64);
        if (!reserve_depth(count, &count)) return false;
        if (before > std::numeric_limits<uint32_t>::max() - count) {
            rollback_depth(count);
            failures_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const size_t segments = ec_batch::kEcBatchSegmentsPerGroup;
        if (slot_size_ > std::numeric_limits<size_t>::max() / segments ||
            count > std::numeric_limits<size_t>::max() /
                         (slot_size_ * segments)) {
            rollback_depth(count);
            failures_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const size_t stride = slot_size_ * segments;
        const size_t allocation_bytes = count * stride;
        try {
            owner.nodes.reserve(before + count);
            owner.free.reserve(owner.free.size() + count);
            owner.chunks.reserve(owner.chunks.size() + 1);
        } catch (const std::bad_alloc &) {
            rollback_depth(count);
            failures_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        RecoveryScratchChunk chunk{};
        if (!allocate_(context_, allocation_bytes, &chunk) ||
            chunk.base == nullptr) {
            if (chunk.base != nullptr || chunk.registration != nullptr) {
                release_(context_, chunk);
            }
            rollback_depth(count);
            failures_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        std::vector<std::unique_ptr<Node>> created;
        try {
            created.reserve(count);
            uint64_t first_index = 0;
            uint64_t current_index = next_index_.load(
                std::memory_order_relaxed);
            bool index_reserved = false;
            for (;;) {
                if (current_index > std::numeric_limits<uint32_t>::max() - count) {
                    break;
                }
                if (next_index_.compare_exchange_weak(
                        current_index, current_index + count,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    first_index = current_index;
                    index_reserved = true;
                    break;
                }
            }
            if (!index_reserved) {
                release_(context_, chunk);
                rollback_depth(count);
                failures_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            for (size_t i = 0; i < count; ++i) {
                auto node = std::make_unique<Node>();
                node->pool = this;
                node->owner = &owner;
                auto &value = node->value;
                value.slot_size = static_cast<uint32_t>(slot_size_);
                value.index = static_cast<uint32_t>(first_index + i);
                value.mr_offset = static_cast<uint64_t>(i * stride);
                value.lkey = chunk.lkey;
                value.pool_cookie = reinterpret_cast<uintptr_t>(this);
                value.lease_cookie = reinterpret_cast<uintptr_t>(node.get());
                auto *base = static_cast<unsigned char *>(chunk.base) + i * stride;
                for (size_t s = 0; s < ec_batch::kEcBatchDataSlots; ++s) {
                    value.data[s] = base + s * slot_size_;
                }
                for (size_t s = 0; s < ec_batch::kEcBatchParitySlots; ++s) {
                    value.parity[s] =
                        base + (ec_batch::kEcBatchDataSlots + s) * slot_size_;
                }
                created.push_back(std::move(node));
            }
            owner.chunks.push_back({chunk, count});
            for (auto &node : created) {
                owner.free.push_back(node.get());
                owner.nodes.push_back(std::move(node));
            }
        } catch (const std::bad_alloc &) {
            release_(context_, chunk);
            rollback_depth(count);
            failures_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        bytes_.fetch_add(allocation_bytes, std::memory_order_acq_rel);
        growths_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool lookup_active_node(const ec_batch::EcStagingGroupSlot &value,
                            Node **node_out) const {
        if (node_out == nullptr || !valid_ || !value.valid() ||
            value.pool_cookie != reinterpret_cast<uintptr_t>(this) ||
            value.lease_cookie == 0) {
            return false;
        }
        auto *node = reinterpret_cast<Node *>(value.lease_cookie);
        // pool_cookie is checked before dereference, so leases from another
        // pool are rejected without touching their node metadata.
        if (node == nullptr || node->pool != this || node->owner == nullptr ||
            node->value.lease_cookie != value.lease_cookie ||
            node->value.pool_cookie != value.pool_cookie ||
            node->value.index != value.index ||
            node->value.slot_size != value.slot_size ||
            node->value.lkey != value.lkey ||
            node->value.mr_offset != value.mr_offset) {
            return false;
        }
        if (node->lease_state.load(std::memory_order_acquire) !=
            ((value.generation << 1) | kLeaseActiveBit)) {
            return false;
        }
        for (size_t s = 0; s < ec_batch::kEcBatchDataSlots; ++s) {
            if (node->value.data[s] != value.data[s]) return false;
        }
        for (size_t s = 0; s < ec_batch::kEcBatchParitySlots; ++s) {
            if (node->value.parity[s] != value.parity[s]) return false;
        }
        *node_out = node;
        return true;
    }

    void update_peak(size_t value) {
        size_t peak = peak_.load(std::memory_order_relaxed);
        while (peak < value &&
               !peak_.compare_exchange_weak(peak, value,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
        }
    }

    bool valid_ = false;
    size_t slot_size_ = 0;
    size_t limit_ = 0;
    // Legacy recovery defaults to the 4 KiB staging bound. Split-object pools
    // opt into a larger size-class bound without changing existing callers.
    size_t max_slot_size_ = ec_batch::kEcBatchStagingMaxSlotSize;
    void *context_ = nullptr;
    Allocate allocate_ = nullptr;
    Release release_ = nullptr;

    mutable std::mutex owner_init_mutex_;
    std::array<std::atomic<OwnerState *>, kFastOwnerSlots> fast_owners_{};
    std::vector<std::unique_ptr<OwnerState>> owners_;
    std::atomic<uint64_t> next_index_{0};
    std::atomic<size_t> depth_{0};
    std::atomic<size_t> bytes_{0};
    std::atomic<size_t> in_use_{0};
    std::atomic<size_t> peak_{0};
    std::atomic<size_t> growths_{0};
    std::atomic<size_t> failures_{0};
    std::atomic<size_t> retired_{0};
};

// Keep the production name and all old callback/API call sites unchanged.
using EcRecoveryScratchPool = WorkerTempBufferPool;

}  // namespace FarLib::cache::ec_read_recovery
