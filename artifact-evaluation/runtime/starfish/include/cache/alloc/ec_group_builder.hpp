// ---------------------------------------------------------------------------
// ec_batch write path, stage A2b - group accumulator + local RS(4,2) encode.
//
// One EcGroupBuilderT accumulates up to four objects into one slot group of the
// EC stripe allocator, encodes the two parity segments locally and publishes the
// sealed group.  This stage does NOT post any RDMA work: it stops at the point
// where the six segments of a group are known (remote addr + endpoint per
// segment) and the six source buffers are staged inside the registered client
// MR.  The next stage consumes the published records and issues the six
// single-sided WRITEs.
//
// Per group:
//   1. add_object() borrows one staging group slot from EcStagingPool, asks the
//      allocator for a group of the object's size class (allocate_slot_group),
//      copies the object into data slot i (zero padded to the group's slot
//      size) and records the object pointer + size.
//   2. When the fourth object arrives (or on an explicit flush()), the holes
//      are zero filled, the two parity slots are computed with the ported
//      GF(2^8) codec over exactly slot_size bytes, and the group is sealed with
//      the 4-bit live mask (seal_slot_group).  Sealing is what makes the group
//      visible to the allocation side and irreversible here: after this call
//      the group takes no further objects and the record is immutable.
//   3. The sealed group is appended to a FIFO of pending groups.  The next
//      stage pops it with pop_pending(), posts the six writes and gives the
//      staging slot back with release_sent() after the last CQE.
//
// Consistency model: a group is allocated, sealed and reused as a whole.  There
// is no incremental update, no read-modify-write and no lock outside this class;
// an object that is not among the four live shards of its group is a zero-filled
// hole that the decoder still accounts for (live mask).
//
// Failure policy - nothing is silently dropped:
//   * staging pool exhausted  -> kStagingExhausted (no group was allocated)
//   * pending FIFO full       -> kPendingQueueFull, the *unsealed* group stays
//                                staged; retry flush() after pop_pending()
//   * allocator refused       -> kManagerRejected (staging slot returned)
//   * encode/seal refused     -> kEncodeRejected / kSealRejected; the staging
//                                slot is returned and the group is released as
//                                a whole (mark_dead_group) so no slot leaks
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>

#include "cache/alloc/ec_batch_staging.hpp"
#include "cache/alloc/small_object_stripe.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"

namespace FarLib::cache::ec_batch {

// The group shape, the staging shape and the codec shape must be the same
// 4 data + 2 parity / six-segment geometry.
static_assert(kEcBatchDataSlots == kSmallObjectStripeDataShards,
              "ec_batch staging must match the allocator's data shards");
static_assert(kEcBatchParitySlots == kSmallObjectStripeParityShards,
              "ec_batch staging must match the allocator's parity shards");
static_assert(kEcBatchSegmentsPerGroup == kSmallObjectStripeShardCount,
              "ec_batch staging must match the allocator's segment count");
static_assert(kEcBatchDataSlots == kStripeCodecDataShards,
              "ec_batch staging must match the RS(4,2) codec data shards");
static_assert(kEcBatchParitySlots == kStripeCodecParityShards,
              "ec_batch staging must match the RS(4,2) codec parity shards");

enum class EcBatchStatus : uint8_t {
    kOk = 0,
    kGroupFull,          // // the staged group already holds four objects
    kStagingExhausted,   // no free staging group slot (depth groups in flight)
    kPendingQueueFull,   // no room for another sealed group; retry after pop
    kManagerRejected,    // allocate_slot_group() refused to hand out a group
    kEncodeRejected,     // the RS(4,2) codec refused
    kSealRejected,       // seal_slot_group() refused (group not in progress)
    kObjectTooLarge,     // object does not fit the group's slot size
    kInvalidArgument,
};

inline const char *ec_batch_status_name(EcBatchStatus status) {
    switch (status) {
        case EcBatchStatus::kOk:
            return "ok";
        case EcBatchStatus::kGroupFull:
            return "group_full";
        case EcBatchStatus::kStagingExhausted:
            return "staging_exhausted";
        case EcBatchStatus::kPendingQueueFull:
            return "pending_queue_full";
        case EcBatchStatus::kManagerRejected:
            return "manager_rejected";
        case EcBatchStatus::kEncodeRejected:
            return "encode_rejected";
        case EcBatchStatus::kSealRejected:
            return "seal_rejected";
        case EcBatchStatus::kObjectTooLarge:
            return "object_too_large";
        case EcBatchStatus::kInvalidArgument:
            return "invalid_argument";
    }
    return "unknown";
}

// How many sealed groups may wait for their RDMA round.  Bounded on purpose:
// the staging pool (depth) is the real limiter, this FIFO only has to be big
// enough not to block a full eviction batch.
inline constexpr size_t kEcBatchPendingQueueDefaultDepth = 32;
inline constexpr size_t kEcBatchPendingQueueMaxDepth = 256;

template <class ManagerT>
class EcGroupBuilderT {
public:
    using Manager = ManagerT;
    using SlotGroupHandle = typename Manager::SlotGroupHandle;
    using SlotGroupId = typename Manager::SlotGroupId;
    using SlotGroupSegment = typename Manager::SlotGroupSegment;

    // One sealed, encoded group: the six remote segments (shard_idx,
    // endpoint_idx, offset, addr, slot_size), the live mask, the objects it
    // carries and the six local source buffers.  Produced by flush()/the fourth
    // add_object() and consumed by the next stage (the RDMA write round).
    struct SealedGroup {
        SlotGroupHandle group{};      // 6 segments: 0..3 data, 4..5 parity
        uint8_t live_mask = 0;        // bit i set <=> data slot i holds object i
        uint8_t live_count = 0;       // popcount(live_mask)
        uint32_t slot_size = 0;       // bytes per segment (== group.slot_size)
        const void *objects[kEcBatchDataSlots]{};  // nullptr in a hole
        uint32_t object_sizes[kEcBatchDataSlots]{};  // 0 in a hole
        EcStagingGroupSlot staging{};  // local source, inside the client MR
        uint64_t sequence = 0;         // seal order (FIFO identity)
        bool automatic = false;        // sealed by the 4th object, not by flush()

        bool full() const { return live_count == kEcBatchDataSlots; }

        // Remote segment i of this group (0..3 data, 4..5 parity): the RDMA
        // target is (endpoint_idx, addr).
        const SlotGroupSegment &segment(size_t i) const {
            return group.segments[i];
        }

        // Local source of segment i: inside the registered client MR, same
        // order as the segments (data first, then parity).
        uint64_t source_offset(size_t i) const {
            return staging.mr_offset + static_cast<uint64_t>(i) * slot_size;
        }
        uintptr_t source_addr(size_t i) const {
            return reinterpret_cast<uintptr_t>(staging.data[0]) +
                   static_cast<uint64_t>(i) * slot_size;
        }
    };

    EcGroupBuilderT() = default;
    EcGroupBuilderT(Manager *manager, EcStagingPool *staging,
                    size_t pending_capacity = kEcBatchPendingQueueDefaultDepth)
        : manager_(manager), staging_(staging) {
        if (pending_capacity == 0) pending_capacity = 1;
        if (pending_capacity > kEcBatchPendingQueueMaxDepth) {
            pending_capacity = kEcBatchPendingQueueMaxDepth;
        }
        pending_capacity_ = pending_capacity;
    }

    Manager *manager() const { return manager_; }
    EcStagingPool *staging() const { return staging_; }

    // A builder is usable only with a real allocator and a live staging pool.
    bool valid() const {
        return manager_ != nullptr && staging_ != nullptr && staging_->valid();
    }

    // ---------------------------- state queries ----------------------------
    bool group_open() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return group_open_;
    }
    size_t staged_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return staged_count_;
    }
    uint8_t staged_live_mask() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return live_mask_;
    }
    uint32_t current_slot_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return group_open_ ? handle_.slot_size : 0;
    }
    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size();
    }
    size_t pending_capacity() const { return pending_capacity_; }
    size_t staging_in_use() const {
        return staging_ != nullptr ? staging_->in_use() : 0;
    }
    uint64_t sealed_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sealed_count_;
    }
    uint64_t automatic_seal_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return automatic_seal_count_;
    }
    uint64_t partial_seal_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return partial_seal_count_;
    }
    // Bytes of parity actually produced (2 * slot_size per sealed group).
    uint64_t encoded_parity_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return encoded_parity_bytes_;
    }

    // Diagnostics only (observability of allocate_slot_group(), which itself
    // reports a plain bool): how often the allocator handed out a slot group and
    // how often it refused one.  Nothing in this class reads them.
    uint64_t group_alloc_ok_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return group_alloc_ok_;
    }
    uint64_t group_alloc_fail_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return group_alloc_fail_;
    }

    // ------------------------------ the API --------------------------------
    // Stages one object.  The object is copied into the next free data slot of
    // the group that is being accumulated, zero padded to the group's slot
    // size; the group is allocated on the first object of the group.
    //   - size > the staging slot size: kObjectTooLarge
    //   - the group is already full (four staged objects, waiting for flush()
    //     to find room in the pending FIFO): kGroupFull, nothing is written
    //   - the four staged objects are complete: the group is encoded, sealed
    //     and published, and the call returns kOk.
    EcBatchStatus add_object(const void *object, size_t size) {
        return add_object(object, size, nullptr);
    }

    // Same as add_object() above, but also reports the absolute remote address
    // of the data segment the object was copied into (one of the group's four
    // data shards, segment index 0..3).  The eviction path uses it as the
    // object's remote address: the report happens under the builder lock and
    // before the group can be sealed, so no segment of this group can be posted
    // (and no completion can run) before that address is published.
    EcBatchStatus add_object(const void *object, size_t size,
                             uint64_t *segment_addr_out) {
        if (object == nullptr || size == 0) {
            return EcBatchStatus::kInvalidArgument;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (manager_ == nullptr || staging_ == nullptr || !staging_->valid()) {
            return EcBatchStatus::kInvalidArgument;
        }
        if (size > staging_->slot_size()) return EcBatchStatus::kObjectTooLarge;
        if (group_open_ && staged_count_ >= kEcBatchDataSlots) {
            return EcBatchStatus::kGroupFull;
        }
        if (!group_open_) {
            // Borrow the source buffers first: a group allocated without a
            // staging slot would leak its four data slots, while a staging slot
            // without a group is simply released again.
            EcStagingGroupSlot slot;
            if (!staging_->acquire(&slot)) {
                return EcBatchStatus::kStagingExhausted;
            }
            SlotGroupHandle handle;
            if (!manager_->allocate_slot_group(size, &handle)) {
                group_alloc_fail_++;
                staging_->release(slot);
                return EcBatchStatus::kManagerRejected;
            }
            group_alloc_ok_++;
            if (handle.slot_size == 0 || handle.slot_size > slot.slot_size) {
                manager_->mark_dead_group(handle.id);
                staging_->release(slot);
                return EcBatchStatus::kObjectTooLarge;
            }
            current_slot_ = slot;
            handle_ = handle;
            group_open_ = true;
            staged_count_ = 0;
            live_mask_ = 0;
        }
        if (size > handle_.slot_size) {
            // The group's slot size was fixed by the first object of the group.
            return EcBatchStatus::kObjectTooLarge;
        }
        const size_t slot_index = staged_count_;
        uint8_t *dst = static_cast<uint8_t *>(current_slot_.data[slot_index]);
        std::memcpy(dst, object, size);
        if (size < handle_.slot_size) {
            std::memset(dst + size, 0, handle_.slot_size - size);
        }
        objects_[slot_index] = object;
        object_sizes_[slot_index] = static_cast<uint32_t>(size);
        live_mask_ |= static_cast<uint8_t>(1u << slot_index);
        staged_count_++;
        if (segment_addr_out != nullptr) {
            *segment_addr_out = handle_.segments[slot_index].addr;
        }
        if (staged_count_ == kEcBatchDataSlots) {
            return seal_locked(true);
        }
        return EcBatchStatus::kOk;
    }

    // Closes the group that is being accumulated: zero fills the holes, encodes
    // the two parity slots and seals the group.  Fewer than four objects is
    // legal - the live mask records how many data slots carry an object, the
    // rest are holes that read as zero.  With nothing staged it is a no-op.
    EcBatchStatus flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        return seal_locked(false);
    }

    // Gives up the group that is being accumulated (unsealed): the staging slot
    // is returned and the group is released as a whole.  Intended for teardown
    // and for an aborted eviction batch; any object already staged is dropped.
    EcBatchStatus drop_partial() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!group_open_) return EcBatchStatus::kOk;
        abandon_locked();
        return EcBatchStatus::kOk;
    }

    // --------------------- query interface for the next round --------------
    // Pops the oldest sealed group (FIFO).  The record keeps ownership of its
    // staging slot until release_sent() is called for it.
    bool pop_pending(SealedGroup *out) {
        if (out == nullptr) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_.empty()) return false;
        *out = pending_.front();
        pending_.pop_front();
        return true;
    }

    // Hands the staging slot of an already sent group back to the pool.  Only
    // after the last CQE of that group's six writes.
    bool release_sent(const SealedGroup &record) {
        if (staging_ == nullptr) return false;
        return staging_->release(record.staging);
    }

    // Teardown only (the cache is quiesced, nothing calls flush()/pop_pending()
    // concurrently): hands every sealed group that was never posted - and the
    // objects of the group that is still open - to `fn(const SealedGroup &)`
    // exactly once, gives the staging slot of every unposted group back and
    // marks the open group dead as a whole.  The caller owns the write
    // reference of the objects it is handed: no CQE will ever arrive for them.
    // Returns the number of groups dropped.
    template <typename Fn>
    size_t abandon_unposted(Fn &&fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t dropped = 0;
        while (!pending_.empty()) {
            const SealedGroup record = pending_.front();
            pending_.pop_front();
            fn(record);
            release_staging_locked(record.staging);
            ++dropped;
        }
        if (group_open_) {
            // The open group never reached the encoder: the objects only exist
            // in its staging slot and no completion can come for them, but the
            // write reference taken when they were staged still has to be
            // consumed by fn().  The staging slot itself belongs to the group
            // and goes back through abandon_locked() below, so the synthesized
            // record carries none.
            SealedGroup open_group;
            open_group.group = handle_;
            open_group.live_mask = live_mask_;
            open_group.live_count = static_cast<uint8_t>(
                __builtin_popcount(static_cast<unsigned>(live_mask_)));
            open_group.slot_size = handle_.slot_size;
            for (size_t i = 0; i < kEcBatchDataSlots; i++) {
                open_group.objects[i] = objects_[i];
                open_group.object_sizes[i] = object_sizes_[i];
            }
            fn(open_group);
            abandon_locked();
            ++dropped;
        }
        return dropped;
    }

private:
    EcBatchStatus seal_locked(bool automatic) {
        if (!group_open_) return EcBatchStatus::kOk;
        if (pending_.size() >= pending_capacity_) {
            // Refuse before sealing: a sealed group with nowhere to go would
            // strand its staging slot until teardown.
            return EcBatchStatus::kPendingQueueFull;
        }
        // A data slot without an object must read as zero for the encoder (and
        // stays zero on the remote side): the hole is defined, not garbage.
        for (size_t i = staged_count_; i < kEcBatchDataSlots; i++) {
            std::memset(current_slot_.data[i], 0, handle_.slot_size);
        }

        const void *data_shards[kEcBatchDataSlots];
        void *parity_shards[kEcBatchParitySlots];
        for (size_t i = 0; i < kEcBatchDataSlots; i++) {
            data_shards[i] = current_slot_.data[i];
        }
        for (size_t i = 0; i < kEcBatchParitySlots; i++) {
            parity_shards[i] = current_slot_.parity[i];
        }
        if (!small_object_stripe_encode_shards(
                data_shards, parity_shards, handle_.slot_size, 0)) {
            abandon_locked();
            return EcBatchStatus::kEncodeRejected;
        }
        if (!manager_->seal_slot_group(handle_.id, live_mask_)) {
            abandon_locked();
            return EcBatchStatus::kSealRejected;
        }

        SealedGroup record;
        record.group = handle_;
        record.live_mask = live_mask_;
        record.live_count = static_cast<uint8_t>(
            __builtin_popcount(static_cast<unsigned>(live_mask_)));
        record.slot_size = handle_.slot_size;
        for (size_t i = 0; i < kEcBatchDataSlots; i++) {
            record.objects[i] = objects_[i];
            record.object_sizes[i] = object_sizes_[i];
        }
        record.staging = current_slot_;
        record.sequence = next_sequence_++;
        record.automatic = automatic;
        pending_.push_back(record);
        sealed_count_++;
        if (automatic) {
            automatic_seal_count_++;
        } else {
            partial_seal_count_++;
        }
        encoded_parity_bytes_ +=
            static_cast<uint64_t>(record.slot_size) * kEcBatchParitySlots;
        reset_current_locked();
        return EcBatchStatus::kOk;
    }

    void abandon_locked() {
        if (manager_ != nullptr && handle_.id.valid()) {
            manager_->mark_dead_group(handle_.id);
        }
        release_current_staging_locked();
        reset_current_locked();
    }

    void release_staging_locked(const EcStagingGroupSlot &slot) {
        if (staging_ != nullptr && slot.valid()) {
            staging_->release(slot);
        }
    }

    void release_current_staging_locked() {
        release_staging_locked(current_slot_);
    }

    void reset_current_locked() {
        group_open_ = false;
        staged_count_ = 0;
        live_mask_ = 0;
        handle_ = SlotGroupHandle{};
        current_slot_ = EcStagingGroupSlot{};
        for (size_t i = 0; i < kEcBatchDataSlots; i++) {
            objects_[i] = nullptr;
            object_sizes_[i] = 0;
        }
    }

    Manager *manager_ = nullptr;
    EcStagingPool *staging_ = nullptr;
    size_t pending_capacity_ = kEcBatchPendingQueueDefaultDepth;
    mutable std::mutex mutex_;
    bool group_open_ = false;
    uint8_t staged_count_ = 0;
    uint8_t live_mask_ = 0;
    SlotGroupHandle handle_{};
    EcStagingGroupSlot current_slot_{};
    const void *objects_[kEcBatchDataSlots]{};
    uint32_t object_sizes_[kEcBatchDataSlots]{};
    std::deque<SealedGroup> pending_;
    uint64_t next_sequence_ = 0;
    uint64_t sealed_count_ = 0;
    uint64_t automatic_seal_count_ = 0;
    uint64_t partial_seal_count_ = 0;
    uint64_t encoded_parity_bytes_ = 0;
    // Diagnostics only (bodies of the getters above): how often the allocator
    // handed out / refused a slot group.  Not read by any decision here.
    uint64_t group_alloc_ok_ = 0;
    uint64_t group_alloc_fail_ = 0;
};

// The production instantiation: the allocator's own slot-group API.
using EcGroupBuilder = EcGroupBuilderT<::FarLib::cache::SmallObjectStripeManager>;
using EcGroupSendRecord =
    typename EcGroupBuilderT<::FarLib::cache::SmallObjectStripeManager>::SealedGroup;

}  // namespace FarLib::cache::ec_batch
