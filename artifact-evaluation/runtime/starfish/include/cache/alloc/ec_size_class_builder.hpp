// ---------------------------------------------------------------------------
// Worker-private whole-object EC builder keyed by semantic behavior group and
// allocator size class.
//
// Unlike EcGroupBuilderT (which owns the fixed <=4 KiB staging pool), this
// builder obtains one registered six-segment lease for the exact allocator bin
// returned by the size-class manager.  The six data/parity buffers are then
// encoded with the same ISA-L RS(4,2) codec and published as an ordinary
// EcGroupSendRecord.  It deliberately has no split-object path: each data
// shard names one complete object and all unused bytes in that shard are zero.
// This first integration is intentionally limited to the existing small-object
// cutoff (<= 4096 bytes, including exactly 4096); larger objects stay on the
// established four-fragment split path.
//
// A builder is worker-private.  There is no lock around the open-group lookup
// or the pending ring; the owner must not call add/flush/peek/commit/release
// concurrently from multiple producers.  The pool itself remains responsible
// for making registered leases safe across the sender/completion boundary.
// ---------------------------------------------------------------------------
#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "cache/alloc/ec_group_builder.hpp"
#include "cache/alloc/ec_split_buffers.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "cache/region_based_allocator.hpp"

namespace FarLib::cache::ec_batch {

static_assert(kEcBatchDataSlots == kStripeCodecDataShards,
              "size-class EC builder requires RS(4,2) data geometry");
static_assert(kEcBatchParitySlots == kStripeCodecParityShards,
              "size-class EC builder requires RS(4,2) parity geometry");

// Semantic behavior groups are dense 0..5.  The allocator currently exposes
// 56 bins, and the lookup is indexed as behavior * RegionBinCount + bin.
inline constexpr size_t kEcSizeClassBehaviorGroupCount = 6;
inline constexpr size_t kEcSizeClassOpenGroupLimit = 64;
inline constexpr size_t kEcSizeClassPendingQueueDefaultDepth = 128;
inline constexpr size_t kEcSizeClassPendingQueueMaxDepth = 128;
inline constexpr size_t kEcSizeClassMaxSlotSize =
    kEcBatchStagingMaxSlotSize;  // existing small-object cutoff (4096 bytes)

// Pool/manager failures use the existing ec_batch status vocabulary so the
// generic sender can handle both builders without a second status adapter.
template <class ManagerT, class PoolT>
class EcSizeClassBuilderT {
public:
    using Manager = ManagerT;
    using Pool = PoolT;
    using Record = EcGroupSendRecord;
    using SealedGroup = Record;
    using SlotGroupHandle = typename ManagerT::SlotGroupHandle;
    using SlotGroupId = typename ManagerT::SlotGroupId;
    using SlotGroupSegment = typename ManagerT::SlotGroupSegment;

private:
    struct OpenGroup {
        bool open = false;
        SlotGroupHandle handle{};
        EcStagingGroupSlot staging{};
        uint8_t staged_count = 0;
        uint8_t live_mask = 0;
        const void *objects[kEcBatchDataSlots]{};
        uint32_t object_sizes[kEcBatchDataSlots]{};
    };

    static constexpr size_t kBinCount = ::FarLib::allocator::RegionBinCount;
    static constexpr size_t kLookupCount =
        kEcSizeClassBehaviorGroupCount * kBinCount;
    static constexpr uint8_t kInvalidOpenIndex =
        std::numeric_limits<uint8_t>::max();

    struct PendingRing {
        std::array<Record, kEcSizeClassPendingQueueMaxDepth> records{};
        size_t head = 0;
        size_t count = 0;

        bool empty() const { return count == 0; }
        size_t size() const { return count; }
        Record &front() {
            assert(count != 0);
            return records[head];
        }
        const Record &front() const {
            assert(count != 0);
            return records[head];
        }
        void push_back(const Record &record) {
            assert(count < records.size());
            records[(head + count) % records.size()] = record;
            ++count;
        }
        void pop_front() {
            assert(count != 0);
            head = (head + 1) % records.size();
            --count;
        }
    };

    static size_t lookup_index(uint32_t behavior_group, uint32_t bin) {
        return static_cast<size_t>(behavior_group) * kBinCount + bin;
    }

    void initialize_lookup() { open_lookup_.fill(kInvalidOpenIndex); }

    OpenGroup *open_group_for(size_t index) {
        const uint8_t slot = open_lookup_[index];
        return slot == kInvalidOpenIndex ? nullptr : &open_storage_[slot];
    }

    const OpenGroup *open_group_for(size_t index) const {
        const uint8_t slot = open_lookup_[index];
        return slot == kInvalidOpenIndex ? nullptr : &open_storage_[slot];
    }

    OpenGroup *reserve_open_group(size_t index) {
        if (index >= kLookupCount || open_lookup_[index] != kInvalidOpenIndex ||
            open_count_ >= kEcSizeClassOpenGroupLimit) {
            return nullptr;
        }
        for (size_t slot = 0; slot < kEcSizeClassOpenGroupLimit; ++slot) {
            if (open_storage_[slot].open) continue;
            open_storage_[slot] = OpenGroup{};
            open_storage_[slot].open = true;
            open_lookup_[index] = static_cast<uint8_t>(slot);
            ++open_count_;
            return &open_storage_[slot];
        }
        return nullptr;
    }

    void close_open_group(size_t index) {
        assert(index < kLookupCount);
        const uint8_t slot = open_lookup_[index];
        if (slot == kInvalidOpenIndex) return;
        open_storage_[slot] = OpenGroup{};
        open_lookup_[index] = kInvalidOpenIndex;
        assert(open_count_ != 0);
        --open_count_;
    }

    static bool request_bin(size_t size, uint32_t *bin_out) {
        if (bin_out == nullptr || size == 0 ||
            size > kEcSizeClassMaxSlotSize) {
            return false;
        }
        const size_t words = ::FarLib::allocator::wsize_from_size(size);
        if (words == 0 || words > ::FarLib::allocator::MaxBinSize) {
            return false;
        }
        const size_t bin = ::FarLib::allocator::bin_from_wsize(words);
        if (bin >= kBinCount ||
            ::FarLib::allocator::get_bin_size(bin) < size) {
            return false;
        }
        *bin_out = static_cast<uint32_t>(bin);
        return true;
    }

    // BufferPools has the production acquire(requested, owner, out) API.
    // The one-argument form keeps this header usable with the fixed staging
    // pool in CPU tests and with a caller-provided compatible pool.  A fixed
    // pool is accepted only when it already has the exact requested stride;
    // source_addr() relies on that invariant when the sender posts a record.
    static bool acquire_pool(Pool *pool, size_t slot_size, size_t owner,
                             EcStagingGroupSlot *out) {
        if (pool == nullptr || out == nullptr) return false;
        if constexpr (requires(Pool &p, size_t s, size_t o,
                               EcStagingGroupSlot *q) {
                          p.acquire(s, o, q);
                      }) {
            return static_cast<bool>(pool->acquire(slot_size, owner, out));
        } else if constexpr (requires(Pool &p, size_t s,
                                      EcStagingGroupSlot *q) {
                                 p.acquire(s, q);
                             }) {
            return static_cast<bool>(pool->acquire(slot_size, out));
        } else if constexpr (requires(Pool &p, EcStagingGroupSlot *q) {
                                 p.acquire(q);
                             }) {
            if (!pool->acquire(out)) return false;
            if (out->slot_size != slot_size) {
                (void)pool->release(*out);
                *out = EcStagingGroupSlot{};
                return false;
            }
            return true;
        } else {
            return false;
        }
    }

    static bool pool_valid(const Pool *pool) {
        if (pool == nullptr) return false;
        if constexpr (requires(const Pool &p) { p.valid(); }) {
            return static_cast<bool>(pool->valid());
        } else {
            return false;
        }
    }

    static bool release_pool(Pool *pool, const EcStagingGroupSlot &slot) {
        if (pool == nullptr) return false;
        if constexpr (requires(Pool &p, const EcStagingGroupSlot &s) {
                          p.release(s);
                      }) {
            return static_cast<bool>(pool->release(slot));
        } else {
            return false;
        }
    }

    static size_t pool_in_use(const Pool *pool) {
        if (pool == nullptr) return 0;
        if constexpr (requires(const Pool &p) { p.in_use(); }) {
            return static_cast<size_t>(pool->in_use());
        } else {
            return 0;
        }
    }

public:
    EcSizeClassBuilderT() { initialize_lookup(); }

    // `owner` is forwarded to a grow-on-demand registered-buffer pool.  It is
    // intentionally not inferred from behavior_group: ownership is a worker
    // identity, while behavior_group is a semantic data-placement key.
    EcSizeClassBuilderT(Manager *manager, Pool *pool, size_t owner = 0,
                        size_t pending_capacity =
                            kEcSizeClassPendingQueueDefaultDepth)
        : manager_(manager), pool_(pool), owner_(owner) {
        initialize_lookup();
        if (pending_capacity == 0) pending_capacity = 1;
        if (pending_capacity > kEcSizeClassPendingQueueMaxDepth) {
            pending_capacity = kEcSizeClassPendingQueueMaxDepth;
        }
        pending_capacity_ = pending_capacity;
    }

    EcSizeClassBuilderT(const EcSizeClassBuilderT &) = delete;
    EcSizeClassBuilderT &operator=(const EcSizeClassBuilderT &) = delete;

    Manager *manager() const { return manager_; }
    Pool *pool() const { return pool_; }
    size_t owner() const { return owner_; }
    void set_breakdown(profile::evict_breakdown::Worker *p) { breakdown_ = p; }

    bool valid() const {
        return manager_ != nullptr && pool_ != nullptr && pool_valid(pool_);
    }

    bool group_open() const { return open_count_ != 0; }
    size_t open_group_count() const { return open_count_; }
    size_t pending_count() const { return pending_.size(); }
    size_t pending_capacity() const { return pending_capacity_; }
    size_t staging_in_use() const { return pool_in_use(pool_); }
    uint64_t sealed_count() const { return sealed_count_; }
    uint64_t automatic_seal_count() const { return automatic_seal_count_; }
    uint64_t partial_seal_count() const { return partial_seal_count_; }
    uint64_t encoded_parity_bytes() const { return encoded_parity_bytes_; }
    uint64_t group_alloc_ok_count() const { return group_alloc_ok_; }
    uint64_t group_alloc_fail_count() const { return group_alloc_fail_; }
    uint32_t last_behavior_group() const { return last_behavior_group_; }

    uint64_t sealed_count_for_behavior_group(uint32_t behavior_group) const {
        if (behavior_group >= kEcSizeClassBehaviorGroupCount) return 0;
        return sealed_by_behavior_group_[behavior_group];
    }

    uint64_t sealed_count_for_size_class(uint32_t bin) const {
        if (bin >= kBinCount) return 0;
        return sealed_by_bin_[bin];
    }

    uint64_t sealed_count_for_key(uint32_t behavior_group,
                                  uint32_t bin) const {
        if (behavior_group >= kEcSizeClassBehaviorGroupCount ||
            bin >= kBinCount) {
            return 0;
        }
        return sealed_by_key_[lookup_index(behavior_group, bin)];
    }

    size_t staged_count(uint32_t behavior_group, uint32_t bin) const {
        if (behavior_group >= kEcSizeClassBehaviorGroupCount ||
            bin >= kBinCount) {
            return 0;
        }
        const auto *entry =
            open_group_for(lookup_index(behavior_group, bin));
        return entry != nullptr ? entry->staged_count : 0;
    }

    // Stages one complete object.  The output address is written immediately
    // after the data copy and before an automatic seal can publish/post the
    // record.  If the pending ring is full, the fourth object remains consumed
    // in its open group and the caller receives kPendingQueueFull; it can retry
    // flush() after committing the pending front.
    EcBatchStatus add_object(const void *object, size_t size,
                             uint64_t *segment_addr_out,
                             uint32_t behavior_group) {
        if (object == nullptr || size == 0 ||
            size > std::numeric_limits<uint32_t>::max() ||
            behavior_group >= kEcSizeClassBehaviorGroupCount) {
            return EcBatchStatus::kInvalidArgument;
        }
        uint32_t bin = 0;
        if (!request_bin(size, &bin)) return EcBatchStatus::kObjectTooLarge;
        last_behavior_group_ = behavior_group;
        const size_t index = lookup_index(behavior_group, bin);
        OpenGroup *current = open_group_for(index);

        if (current == nullptr) {
            if (open_count_ >= kEcSizeClassOpenGroupLimit) {
                return EcBatchStatus::kStagingExhausted;
            }
            profile::evict_breakdown::Scope allocation_scope(
                breakdown_, profile::evict_breakdown::Stage::GroupAllocate);
            SlotGroupHandle handle;
            if (manager_ == nullptr ||
                !manager_->allocate_slot_group(size, &handle)) {
                ++group_alloc_fail_;
                return EcBatchStatus::kManagerRejected;
            }
            ++group_alloc_ok_;

            // The key is the allocator's bin for the request.  Refuse a
            // manager that hands back a different stride: combining it with
            // this key would violate the no-mixed-size-group invariant, and a
            // sender record assumes the staging and remote strides match.
            const size_t expected_slot_size =
                ::FarLib::allocator::get_bin_size(bin);
            if (handle.slot_size != expected_slot_size ||
                handle.slot_size > kEcSizeClassMaxSlotSize ||
                handle.slot_size == 0) {
                (void)manager_->mark_dead_group(handle.id);
                return EcBatchStatus::kObjectTooLarge;
            }

            EcStagingGroupSlot slot;
            if (!acquire_pool(pool_, handle.slot_size, owner_, &slot)) {
                (void)manager_->mark_dead_group(handle.id);
                return EcBatchStatus::kStagingExhausted;
            }
            if (!slot.valid() || slot.slot_size != handle.slot_size) {
                (void)release_pool(pool_, slot);
                (void)manager_->mark_dead_group(handle.id);
                return EcBatchStatus::kObjectTooLarge;
            }

            current = reserve_open_group(index);
            if (current == nullptr) {
                (void)release_pool(pool_, slot);
                (void)manager_->mark_dead_group(handle.id);
                return EcBatchStatus::kStagingExhausted;
            }
            current->handle = handle;
            current->staging = slot;
        }

        if (current->staged_count >= kEcBatchDataSlots) {
            return EcBatchStatus::kGroupFull;
        }
        if (size > current->handle.slot_size) {
            return EcBatchStatus::kObjectTooLarge;
        }

        const size_t slot_index = current->staged_count;
        auto *dst =
            static_cast<uint8_t *>(current->staging.data[slot_index]);
        {
            profile::evict_breakdown::Scope copy_scope(
                breakdown_, profile::evict_breakdown::Stage::CopyPad);
            std::memcpy(dst, object, size);
            std::memset(dst + size, 0,
                        static_cast<size_t>(current->handle.slot_size) - size);
        }
        current->objects[slot_index] = object;
        current->object_sizes[slot_index] = static_cast<uint32_t>(size);
        current->live_mask |= static_cast<uint8_t>(1u << slot_index);
        ++current->staged_count;

        if (segment_addr_out != nullptr) {
            // Publish the remote address before sealing, so a caller that
            // receives kPendingQueueFull can retain the consumed object's
            // destination and retry the queue operation safely.
            *segment_addr_out = current->handle.segments[slot_index].addr;
        }
        if (current->staged_count == kEcBatchDataSlots) {
            return seal_locked(index, true);
        }
        return EcBatchStatus::kOk;
    }

    EcBatchStatus add_object(const void *object, size_t size,
                             uint32_t behavior_group,
                             uint64_t *segment_addr_out = nullptr) {
        return add_object(object, size, segment_addr_out, behavior_group);
    }

    // Compatibility overload for callers that intentionally use behavior 0.
    EcBatchStatus add_object(const void *object, size_t size,
                             uint64_t *segment_addr_out = nullptr) {
        return add_object(object, size, segment_addr_out, 0);
    }

    // Seal every currently open (behavior, bin) group.  A pending-full result
    // leaves that group open and is returned after all other keys are visited.
    EcBatchStatus flush() {
        EcBatchStatus result = EcBatchStatus::kOk;
        for (size_t index = 0; index < kLookupCount; ++index) {
            if (open_group_for(index) == nullptr) continue;
            const EcBatchStatus status = seal_locked(index, false);
            if (status != EcBatchStatus::kOk && result == EcBatchStatus::kOk) {
                result = status;
            }
        }
        return result;
    }

    // Abandon unsealed groups only.  A sealed record stays owned by the
    // pending ring and must be committed/released or passed through
    // abandon_unposted().
    EcBatchStatus drop_partial() {
        for (size_t index = 0; index < kLookupCount; ++index) {
            if (open_group_for(index) != nullptr) abandon_locked(index);
        }
        return EcBatchStatus::kOk;
    }

    bool peek_pending(Record *out) const {
        if (out == nullptr || pending_.empty()) return false;
        *out = pending_.front();
        return true;
    }

    // Worker-private sender fast path: no descriptor copy is needed merely to
    // inspect the FIFO front.  The pointer remains valid until commit/pop.
    const Record *peek_private_pending() const {
        return pending_.empty() ? nullptr : &pending_.front();
    }

    bool commit_pending(uint64_t sequence) {
        if (pending_.empty() || pending_.front().sequence != sequence) {
            return false;
        }
        pending_.pop_front();
        return true;
    }

    bool pop_pending(Record *out) {
        if (out == nullptr || pending_.empty()) return false;
        *out = pending_.front();
        pending_.pop_front();
        return true;
    }

    // Called only after the sender's final six CQEs for this record.
    bool release_sent(const Record &record) {
        if (!record.size_class_staging) return false;
        return release_pool(pool_, record.staging);
    }

    template <typename Fn>
    size_t abandon_unposted(Fn &&fn) {
        size_t dropped = 0;
        while (!pending_.empty()) {
            const Record record = pending_.front();
            pending_.pop_front();
            fn(record);
            (void)release_pool(pool_, record.staging);
            ++dropped;
        }
        for (size_t index = 0; index < kLookupCount; ++index) {
            const OpenGroup *current = open_group_for(index);
            if (current == nullptr) continue;
            Record record{};
            record.group = current->handle;
            record.live_mask = current->live_mask;
            record.live_count = static_cast<uint8_t>(__builtin_popcount(
                static_cast<unsigned>(current->live_mask)));
            record.slot_size = current->handle.slot_size;
            record.behavior_group =
                static_cast<uint32_t>(index / kBinCount);
            record.staging = current->staging;
            record.split_object = false;
            record.size_class_staging = true;
            for (size_t i = 0; i < kEcBatchDataSlots; ++i) {
                record.objects[i] = current->objects[i];
                record.object_sizes[i] = current->object_sizes[i];
            }
            fn(record);
            abandon_locked(index);
            ++dropped;
        }
        return dropped;
    }

private:
    EcBatchStatus seal_locked(size_t index, bool automatic) {
        OpenGroup *current = open_group_for(index);
        if (current == nullptr) return EcBatchStatus::kOk;
        if (pending_.size() >= pending_capacity_) {
            return EcBatchStatus::kPendingQueueFull;
        }
        profile::evict_breakdown::Scope seal_scope(
            breakdown_, profile::evict_breakdown::Stage::SealQueue);

        {
            profile::evict_breakdown::Scope pad_scope(
                breakdown_, profile::evict_breakdown::Stage::CopyPad);
            for (size_t i = current->staged_count; i < kEcBatchDataSlots;
                 ++i) {
                std::memset(current->staging.data[i], 0,
                            current->handle.slot_size);
            }
        }

        const void *data_shards[kEcBatchDataSlots]{};
        void *parity_shards[kEcBatchParitySlots]{};
        for (size_t i = 0; i < kEcBatchDataSlots; ++i) {
            data_shards[i] = current->staging.data[i];
        }
        for (size_t i = 0; i < kEcBatchParitySlots; ++i) {
            parity_shards[i] = current->staging.parity[i];
        }
        bool encoded = false;
        {
            profile::evict_breakdown::Scope encode_scope(
                breakdown_, profile::evict_breakdown::Stage::Encode);
            encoded = small_object_stripe_encode_shards(
                data_shards, parity_shards, current->handle.slot_size, 0);
        }
        if (!encoded) {
            abandon_locked(index);
            return EcBatchStatus::kEncodeRejected;
        }
        if (!manager_->seal_slot_group(current->handle.id, current->live_mask,
                                       current->object_sizes,
                                       static_cast<uint32_t>(index / kBinCount))) {
            abandon_locked(index);
            return EcBatchStatus::kSealRejected;
        }

        Record record{};
        record.group = current->handle;
        record.live_mask = current->live_mask;
        record.live_count = static_cast<uint8_t>(__builtin_popcount(
            static_cast<unsigned>(current->live_mask)));
        record.slot_size = current->handle.slot_size;
        record.objects[0] = current->objects[0];
        record.objects[1] = current->objects[1];
        record.objects[2] = current->objects[2];
        record.objects[3] = current->objects[3];
        record.object_sizes[0] = current->object_sizes[0];
        record.object_sizes[1] = current->object_sizes[1];
        record.object_sizes[2] = current->object_sizes[2];
        record.object_sizes[3] = current->object_sizes[3];
        record.staging = current->staging;
        record.sequence = next_sequence_++;
        record.automatic = automatic;
        record.behavior_group =
            static_cast<uint32_t>(index / kBinCount);
        record.split_object = false;
        record.size_class_staging = true;
        pending_.push_back(record);

        ++sealed_count_;
        ++sealed_by_behavior_group_[record.behavior_group];
        const uint32_t bin = static_cast<uint32_t>(index % kBinCount);
        ++sealed_by_bin_[bin];
        ++sealed_by_key_[index];
        if (automatic) {
            ++automatic_seal_count_;
        } else {
            ++partial_seal_count_;
        }
        encoded_parity_bytes_ +=
            static_cast<uint64_t>(record.slot_size) * kEcBatchParitySlots;
        close_open_group(index);
        return EcBatchStatus::kOk;
    }

    void abandon_locked(size_t index) {
        OpenGroup *current = open_group_for(index);
        if (current == nullptr) return;
        if (manager_ != nullptr && current->handle.id.valid()) {
            (void)manager_->mark_dead_group(current->handle.id);
        }
        (void)release_pool(pool_, current->staging);
        close_open_group(index);
    }

    Manager *manager_ = nullptr;
    Pool *pool_ = nullptr;
    size_t owner_ = 0;
    profile::evict_breakdown::Worker *breakdown_ = nullptr;
    size_t pending_capacity_ = kEcSizeClassPendingQueueDefaultDepth;
    // Fixed bounded storage: no group-open allocation on the hot path.  The
    // lookup array is separate so only the selected bin/behavior slot is live;
    // unused keys cost one byte each rather than a constructed group object.
    std::array<OpenGroup, kEcSizeClassOpenGroupLimit> open_storage_{};
    std::array<uint8_t, kLookupCount> open_lookup_{};
    size_t open_count_ = 0;
    PendingRing pending_{};
    uint64_t next_sequence_ = 0;
    uint64_t sealed_count_ = 0;
    std::array<uint64_t, kEcSizeClassBehaviorGroupCount>
        sealed_by_behavior_group_{};
    std::array<uint64_t, kBinCount> sealed_by_bin_{};
    std::array<uint64_t, kLookupCount> sealed_by_key_{};
    uint64_t automatic_seal_count_ = 0;
    uint64_t partial_seal_count_ = 0;
    uint64_t encoded_parity_bytes_ = 0;
    uint64_t group_alloc_ok_ = 0;
    uint64_t group_alloc_fail_ = 0;
    uint32_t last_behavior_group_ = 0;
};

using EcSizeClassBuilder =
    EcSizeClassBuilderT<::FarLib::cache::SmallObjectStripeManager,
                        ::FarLib::cache::ec_split::BufferPools>;

}  // namespace FarLib::cache::ec_batch
