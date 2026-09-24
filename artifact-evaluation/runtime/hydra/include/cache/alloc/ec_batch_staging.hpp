// ---------------------------------------------------------------------------
// ec_batch write path, stage A2a - client-side staging pool.
//
// Under ft_method=ec_batch an object is never written on its own.  Eviction
// collects four small objects into one *slot group* of the EC stripe allocator
// (include/cache/alloc/small_object_stripe.hpp: allocate_slot_group /
// seal_slot_group / mark_dead_group); the client then encodes RS(4,2) locally
// with the ported GF(2^8) codec (small_object_stripe_codec.hpp) and writes the
// six segments - four data slots plus two parity slots - to six distinct
// endpoints with single-sided RDMA WRITE.
//
// The six source buffers must live in *registered* memory and must stay alive
// until the last CQE of the group has been reaped: a stack buffer or a
// per-call allocation is a use-after-free waiting to happen.  This header owns
// that memory.  The pool is resident (built once, at initialization), covers a
// caller-provided range of an already registered MR (the pool itself allocates
// no memory and never registers anything), and reuses its group slots across
// groups instead of allocating per group.
//
//   EcStagingPool pool;
//   pool.init(mr_base, mr_bytes, slot_size, depth);   // once, at init
//   EcStagingGroupSlot slot;
//   pool.acquire(&slot);              // slot.data[0..3], slot.parity[0..1]
//   ...  stage the objects, encode the two parity slots  ...
//   ...  post the six RDMA WRITEs; keep `slot` alive until the last CQE  ...
//   pool.release(slot);               // only after the last CQE of the group
//
// Layout: depth group slots of six contiguous slot_size segments each, so one
// group is a single contiguous MR range and the offset of every segment is
// fixed:
//   group g, segment s  ->  mr_base + (g * 6 + s) * slot_size
// Segment 0..3 are the data shards, 4..5 the parity shards - the same order as
// SlotGroupHandle::segments[].
//
// Concurrency: one mutex covers the pool, so several groups (and several
// eviction threads) can acquire/release concurrently.  `depth` is the hard
// limit on how many groups can be staged (encoded but not yet released) at
// once; a further acquire() fails instead of overrunning the pool, and a
// release of a slot that is not currently borrowed fails instead of corrupting
// the free list.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace FarLib::cache::ec_batch {

// One ec_batch group is 4 data + 2 parity segments, one slot size for all six.
inline constexpr size_t kEcBatchDataSlots = 4;
inline constexpr size_t kEcBatchParitySlots = 2;
inline constexpr size_t kEcBatchSegmentsPerGroup =
    kEcBatchDataSlots + kEcBatchParitySlots;  // 6

// Resident depth of the pool: how many groups may be staged (encoded, not yet
// sent and released) at the same time.  Deliberately small and fixed: a slot
// holds a full group slot size per segment, and staging is bounded by the
// evict batch / qp_send_cap anyway.
inline constexpr size_t kEcBatchStagingDefaultDepth = 8;
inline constexpr size_t kEcBatchStagingMaxDepth = 64;

// Largest slot this pool accepts: ft_small_object_cutoff (4096) rounded up to
// the allocator's largest small-object bin.  Nothing in the ec_batch path may
// stage a bigger object - bigger objects are not group-allocated.
inline constexpr size_t kEcBatchStagingMaxSlotSize = 4096;

// Borrowed staging group slot.  All six pointers lie inside the registered MR
// the pool was initialized with.
struct EcStagingGroupSlot {
    void *data[kEcBatchDataSlots]{};        // 4 data slots, slot_size each
    void *parity[kEcBatchParitySlots]{};    // 2 parity slots, slot_size each
    uint32_t slot_size = 0;
    uint32_t index = 0;                     // group index inside the pool
    uint64_t mr_offset = 0;                 // byte offset of data[0] in the MR
    // Recovery may use separately registered, grow-on-demand chunks. The
    // write staging pool leaves these at zero and uses its original MR.
    uint32_t lkey = 0;
    uint64_t generation = 0;
    // Nonzero for a lease issued by either the fixed pool or a shared
    // provider.  The pool cookie is checked before dereferencing this value,
    // which rejects a lease copied from a foreign pool safely.
    uintptr_t lease_cookie = 0;
    uintptr_t pool_cookie = 0;

    bool valid() const {
        return slot_size != 0 && data[0] != nullptr && parity[0] != nullptr;
    }
};

}  // namespace FarLib::cache::ec_batch

// Keep the provider interface independent from the fixed-pool implementation;
// it only needs the completed slot type above.
#include "cache/alloc/worker_temp_buffers.hpp"

namespace FarLib::cache::ec_batch {

class EcStagingPool {
public:
    using OwnerIdCallback = std::function<size_t()>;

    // Bytes one resident pool of this shape occupies inside the MR.
    static size_t required_bytes(size_t slot_size, size_t depth) {
        return slot_size * kEcBatchSegmentsPerGroup * depth;
    }

    EcStagingPool() = default;
    EcStagingPool(void *mr_base, size_t mr_bytes, size_t slot_size,
                  size_t depth = kEcBatchStagingDefaultDepth) {
        init(mr_base, mr_bytes, slot_size, depth);
    }

    // Binds the pool to a range of an already registered MR.  Returns false and
    // stores a message in error() on an invalid shape; a pool that failed to
    // initialize refuses every acquire(), so a misconfigured run can never
    // write outside the MR.
    bool init(void *mr_base, size_t mr_bytes, size_t slot_size,
              size_t depth = kEcBatchStagingDefaultDepth) {
        std::lock_guard<std::mutex> lock(mutex_);
        valid_ = false;
        base_ = nullptr;
        bytes_ = 0;
        slot_size_ = 0;
        depth_ = 0;
        free_stack_.clear();
        error_.clear();
        if (mr_base == nullptr) {
            error_ = "ec_batch staging: MR base is null";
            return false;
        }
        if (slot_size == 0 || slot_size > kEcBatchStagingMaxSlotSize) {
            error_ = "ec_batch staging: slot_size must be in [1, " +
                     std::to_string(kEcBatchStagingMaxSlotSize) + "]";
            return false;
        }
        if (depth == 0 || depth > kEcBatchStagingMaxDepth) {
            error_ = "ec_batch staging: depth must be in [1, " +
                     std::to_string(kEcBatchStagingMaxDepth) + "]";
            return false;
        }
        if (mr_bytes < required_bytes(slot_size, depth)) {
            error_ = "ec_batch staging: MR range too small (" +
                     std::to_string(mr_bytes) + " bytes, need " +
                     std::to_string(required_bytes(slot_size, depth)) + ")";
            return false;
        }
        base_ = static_cast<uint8_t *>(mr_base);
        bytes_ = required_bytes(slot_size, depth);
        slot_size_ = slot_size;
        depth_ = depth;
        // Free list of group indices; the pool hands out the first free group.
        free_stack_.reserve(depth);
        for (size_t i = depth; i > 0; i--) {
            free_stack_.push_back(static_cast<uint32_t>(i - 1));
        }
        valid_ = true;
        peak_in_use_ = 0;
        return true;
    }

    // Switches acquisition/release and validation to a shared worker-owned
    // provider.  The original fixed-MR reservation remains initialized (and
    // therefore continues to be a valid fallback/API surface), but no new
    // lease uses it after this call.  Binding is an initialization-time
    // operation and must precede concurrent acquire/release calls.
    bool bind_shared_buffers(
        ::FarLib::cache::WorkerTempBufferProvider *provider,
        OwnerIdCallback owner_id_callback = OwnerIdCallback{}) {
        if (provider == nullptr || !provider->valid()) return false;
        if (slot_size_ != 0 && provider->slot_size() != slot_size_) {
            error_ = "ec_batch staging: shared provider slot_size mismatch";
            return false;
        }
        shared_provider_ = provider;
        owner_id_callback_ = std::move(owner_id_callback);
        if (!owner_id_callback_) owner_id_callback_ = [] { return size_t{0}; };
        if (slot_size_ == 0) slot_size_ = provider->slot_size();
        error_.clear();
        return true;
    }

    bool shared_buffers_bound() const { return shared_provider_ != nullptr; }
    ::FarLib::cache::WorkerTempBufferProvider *shared_provider() const {
        return shared_provider_;
    }

    bool valid() const {
        return shared_provider_ != nullptr ? shared_provider_->valid() : valid_;
    }
    const std::string &error() const { return error_; }
    size_t slot_size() const {
        return shared_provider_ != nullptr ? shared_provider_->slot_size()
                                           : slot_size_;
    }
    size_t depth() const {
        return shared_provider_ != nullptr ? shared_provider_->depth() : depth_;
    }
    size_t bytes() const {
        return shared_provider_ != nullptr ? shared_provider_->bytes() : bytes_;
    }
    void *mr_base() const { return base_; }

    // Bytes one group slot spans (6 * slot_size); the stride between groups.
    size_t group_stride() const { return slot_size_ * kEcBatchSegmentsPerGroup; }

    size_t in_use() const {
        if (shared_provider_ != nullptr) return shared_provider_->in_use();
        std::lock_guard<std::mutex> lock(mutex_);
        return valid_ ? depth_ - free_stack_.size() : 0;
    }

    size_t available() const {
        if (shared_provider_ != nullptr) return shared_provider_->available();
        std::lock_guard<std::mutex> lock(mutex_);
        return valid_ ? free_stack_.size() : 0;
    }

    size_t peak_in_use() const {
        if (shared_provider_ != nullptr) return shared_provider_->peak_in_use();
        std::lock_guard<std::mutex> lock(mutex_);
        return valid_ ? peak_in_use_ : 0;
    }

    size_t growths() const {
        if (shared_provider_ != nullptr) return shared_provider_->growths();
        std::lock_guard<std::mutex> lock(mutex_);
        return valid_ ? (depth_ != 0 ? 1 : 0) : 0;
    }

    size_t allocation_failures() const {
        if (shared_provider_ != nullptr) {
            return shared_provider_->allocation_failures();
        }
        return 0;
    }

    size_t limit() const {
        if (shared_provider_ != nullptr) return shared_provider_->limit();
        return depth_;
    }

    // Borrows one group slot (4 data + 2 parity pointers inside the MR).
    // Returns false when the pool is invalid or all `depth` slots are in use;
    // it never hands out a slot twice.
    bool acquire(EcStagingGroupSlot *out) {
        if (out == nullptr) return false;
        if (shared_provider_ != nullptr) {
            const size_t owner = owner_id_callback_ ? owner_id_callback_() : 0;
            return shared_provider_->acquire(out, owner);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid_ || free_stack_.empty()) return false;
        const uint32_t index = free_stack_.back();
        free_stack_.pop_back();
        uint8_t *base = base_ + static_cast<size_t>(index) * group_stride();
        out->slot_size = static_cast<uint32_t>(slot_size_);
        out->index = index;
        out->mr_offset = static_cast<uint64_t>(index) * group_stride();
        out->lease_cookie = reinterpret_cast<uintptr_t>(this);
        out->pool_cookie = reinterpret_cast<uintptr_t>(this);
        out->generation = 1;
        for (size_t i = 0; i < kEcBatchDataSlots; i++) {
            out->data[i] = base + i * slot_size_;
        }
        for (size_t i = 0; i < kEcBatchParitySlots; i++) {
            out->parity[i] = base + (kEcBatchDataSlots + i) * slot_size_;
        }
        const size_t live = depth_ - free_stack_.size();
        if (live > peak_in_use_) peak_in_use_ = live;
        return true;
    }

    // Returns a slot to the pool.  Must only be called after the last CQE of
    // the group that borrowed it.  Releasing a slot that is not currently
    // borrowed (stale copy, double release, foreign slot) returns false and
    // changes nothing.
    bool release(const EcStagingGroupSlot &slot) {
        if (shared_provider_ != nullptr) return shared_provider_->release(slot);
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid_ || !slot.valid()) return false;
        if (slot.pool_cookie != reinterpret_cast<uintptr_t>(this) ||
            slot.lease_cookie != reinterpret_cast<uintptr_t>(this)) {
            return false;
        }
        if (slot.slot_size != slot_size_ || slot.index >= depth_) return false;
        uint8_t *base = base_ + static_cast<size_t>(slot.index) * group_stride();
        for (size_t i = 0; i < kEcBatchDataSlots; i++) {
            if (slot.data[i] != static_cast<void *>(base + i * slot_size_)) {
                return false;
            }
        }
        for (size_t i = 0; i < kEcBatchParitySlots; i++) {
            if (slot.parity[i] !=
                static_cast<void *>(base + (kEcBatchDataSlots + i) * slot_size_)) {
                return false;
            }
        }
        for (size_t i = 0; i < free_stack_.size(); i++) {
            if (free_stack_[i] == slot.index) return false;  // double release
        }
        free_stack_.push_back(slot.index);
        return true;
    }

    // Byte offset of a staging pointer inside the MR (what a post-send needs
    // next to the MR lkey).  False when the pointer is outside this pool.
    bool offset_in_mr(const void *p, uint64_t *offset_out) const {
        if (p == nullptr || offset_out == nullptr || !valid_) return false;
        const uint8_t *q = static_cast<const uint8_t *>(p);
        if (q < base_ || q >= base_ + bytes_) return false;
        *offset_out = static_cast<uint64_t>(q - base_);
        return true;
    }

    // Validates a borrowed slot and one source segment without exposing the
    // implementation-specific MR/chunk layout to the EC path.
    bool owns_slot(const EcStagingGroupSlot &slot) const {
        if (shared_provider_ != nullptr) return shared_provider_->owns_slot(slot);
        if (!slot.valid() || slot.pool_cookie != reinterpret_cast<uintptr_t>(this) ||
            slot.lease_cookie != reinterpret_cast<uintptr_t>(this)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid_ || slot.slot_size != slot_size_ || slot.index >= depth_) {
            return false;
        }
        const uint8_t *base = base_ + static_cast<size_t>(slot.index) * group_stride();
        for (size_t i = 0; i < kEcBatchDataSlots; ++i) {
            if (slot.data[i] != static_cast<const void *>(base + i * slot_size_)) {
                return false;
            }
        }
        for (size_t i = 0; i < kEcBatchParitySlots; ++i) {
            if (slot.parity[i] != static_cast<const void *>(
                                      base + (kEcBatchDataSlots + i) * slot_size_)) {
                return false;
            }
        }
        for (const auto free_index : free_stack_) {
            if (free_index == slot.index) return false;
        }
        return true;
    }

    bool owns_buffer(const EcStagingGroupSlot &slot, const void *ptr,
                     size_t bytes) const {
        if (ptr == nullptr || bytes == 0 || !owns_slot(slot)) return false;
        const auto q = reinterpret_cast<uintptr_t>(ptr);
        for (size_t i = 0; i < kEcBatchDataSlots; ++i) {
            const auto begin = reinterpret_cast<uintptr_t>(slot.data[i]);
            if (q >= begin && q - begin <= slot.slot_size &&
                bytes <= slot.slot_size - (q - begin)) return true;
        }
        for (size_t i = 0; i < kEcBatchParitySlots; ++i) {
            const auto begin = reinterpret_cast<uintptr_t>(slot.parity[i]);
            if (q >= begin && q - begin <= slot.slot_size &&
                bytes <= slot.slot_size - (q - begin)) return true;
        }
        return false;
    }

private:
    uint8_t *base_ = nullptr;
    size_t bytes_ = 0;
    size_t slot_size_ = 0;
    size_t depth_ = 0;
    bool valid_ = false;
    size_t peak_in_use_ = 0;
    std::string error_;
    std::vector<uint32_t> free_stack_;  // indices of free group slots
    mutable std::mutex mutex_;
    ::FarLib::cache::WorkerTempBufferProvider *shared_provider_ = nullptr;
    OwnerIdCallback owner_id_callback_;
};

}  // namespace FarLib::cache::ec_batch
