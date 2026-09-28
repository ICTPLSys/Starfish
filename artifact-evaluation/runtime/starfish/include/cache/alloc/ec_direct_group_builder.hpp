#pragma once
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "cache/alloc/ec_group_builder.hpp"
#include "cache/alloc/ec_direct_write_bank.hpp"
#include "cache/alloc/ec_direct_codec.hpp"

namespace FarLib::cache::ec_batch {

// One permanent builder per logical Evict worker. Data sources are borrowed,
// while each reserved bank position owns its parity and completion record.
class EcDirectGroupBuilder {
public:
    using Bank = EcDirectWriteBank;
    // The default keeps the original one-open-group-per-behavior contract.
    // Size-classed mode adds the allocator bin to the physical open-group key
    // without changing the public behavior-group accounting.
    EcDirectGroupBuilder(SmallObjectStripeManager *manager, Bank *bank,
                         size_t owner, bool size_classed = false)
        : manager_(manager),
          bank_(bank),
          owner_(owner),
          size_classed_(size_classed) {}
    void set_breakdown(profile::evict_breakdown::Worker *p) { breakdown_ = p; }
    bool group_open() const {
        for (size_t key = 0; key < active_open_count(); ++key) {
            if (open_[key].entry != nullptr) return true;
        }
        return false;
    }
    bool size_classed() const { return size_classed_; }
    size_t pending_count() const { return pending_count_; }
    uint64_t front_token() const {
        return pending_count_ ? pending_[pending_head_].token : 0;
    }
    void commit_pending() {
        assert(pending_count_);
        pending_[pending_head_] = {};
        pending_head_ = (pending_head_ + 1) % pending_.size();
        --pending_count_;
    }
    uint64_t sealed_count() const { return sealed_; }
    uint64_t group_alloc_ok_count() const { return allocated_; }
    uint64_t group_alloc_fail_count() const { return allocation_failed_; }
    uint64_t sealed_count_for_behavior_group(size_t behavior_group) const {
        if (behavior_group >= kEcBatchBehaviorGroupCount) return 0;
        return sealed_by_group_[behavior_group];
    }
    uint64_t object_count() const { return objects_; }
    uint64_t data_bytes() const { return data_bytes_; }

    // The direct bank deliberately has no behavior-specific fields.  Keep the
    // immutable identity in the builder's pending metadata instead, so a
    // caller that consumes a token can report its semantic behavior without
    // changing the bank record ABI.
    struct GroupMetadata {
        uint32_t behavior_group = 0;
        uint16_t allocator_bin = 0;
    };

    bool metadata_for_token(uint64_t token, GroupMetadata *metadata) const {
        if (metadata == nullptr || token == 0) return false;
        for (size_t key = 0; key < active_open_count(); ++key) {
            const auto &g = open_[key];
            if (g.entry != nullptr && g.token == token) {
                *metadata = g.metadata;
                return true;
            }
        }
        for (const auto &pending : pending_) {
            if (pending.token == token) {
                *metadata = pending.metadata;
                return true;
            }
        }
        return false;
    }

    EcBatchStatus add_object(const void *object, size_t size, uint64_t *address, size_t key) {
        if (manager_ == nullptr || bank_ == nullptr || object == nullptr ||
            address == nullptr || size == 0 || size > kEcDirectMaxSlotSize ||
            key >= kEcBatchBehaviorGroupCount) {
            return EcBatchStatus::kInvalidArgument;
        }
        const size_t allocator_bin = size_classed_ ? allocator_bin_for_size(size) : 0;
        const size_t physical_key = physical_key_for(key, allocator_bin);
        auto &g = open_[physical_key];
        if (!g.entry) {
            profile::evict_breakdown::Scope scope(breakdown_, profile::evict_breakdown::Stage::GroupAllocate);
            Bank::Lease lease;
            if (!bank_->reserve(owner_, &g.token, &g.entry, &lease)) return EcBatchStatus::kStagingExhausted;
            g.metadata.behavior_group = static_cast<uint32_t>(key);
            g.metadata.allocator_bin = static_cast<uint16_t>(allocator_bin);
            auto &r = g.entry->record;
            if (!manager_->allocate_slot_group(size, &r.group)) {
                ++allocation_failed_;
                bank_->cancel(g.token); g = {};
                return EcBatchStatus::kManagerRejected;
            }
            ++allocated_;
            r.slot_size = r.group.slot_size;
            r.parity[0] = lease.parity[0]; r.parity[1] = lease.parity[1];
            r.zero_pad = lease.zero_pad; r.parity_lkey = lease.lkey;
            r.owner = lease.owner; r.slot = lease.slot; r.generation = lease.generation;
        }
        auto &r = g.entry->record;
        if (size > r.slot_size || r.live_count >= 4) return EcBatchStatus::kObjectTooLarge;
        const size_t slot = r.live_count++;
        r.objects[slot] = object; r.object_sizes[slot] = static_cast<uint32_t>(size);
        r.live_mask |= static_cast<uint8_t>(1u << slot);
        *address = r.group.segments[slot].addr;
        ++objects_; data_bytes_ += size;
        return r.live_count == 4 ? seal(physical_key) : EcBatchStatus::kOk;
    }
    EcBatchStatus flush() {
        for (size_t key = 0; key < active_open_count(); ++key) {
            if (open_[key].entry) {
                auto result = seal(key);
                if (result != EcBatchStatus::kOk) return result;
            }
        }
        return EcBatchStatus::kOk;
    }
private:
    size_t active_open_count() const {
        return size_classed_ ? kOpenGroupCount : kEcBatchBehaviorGroupCount;
    }

    EcBatchStatus seal(size_t key) {
        profile::evict_breakdown::Scope seal_scope(breakdown_, profile::evict_breakdown::Stage::SealQueue);
        auto &g = open_[key];
        auto &r = g.entry->record;
        if (pending_count_ == pending_.size()) return EcBatchStatus::kPendingQueueFull;
        bool encoded;
        {
            profile::evict_breakdown::Scope scope(breakdown_, profile::evict_breakdown::Stage::Encode);
            encoded = encode_direct_shards(r.objects, r.object_sizes, r.parity, r.slot_size, r.zero_pad);
        }
        if (!encoded) return EcBatchStatus::kEncodeRejected;
        if (!manager_->seal_slot_group(r.group.id, r.live_mask, r.object_sizes,
                                      g.metadata.behavior_group)) return EcBatchStatus::kSealRejected;
        if (!bank_->publish(g.token)) return EcBatchStatus::kSealRejected;
        const size_t pending_index =
            (pending_head_ + pending_count_) % pending_.size();
        pending_[pending_index].token = g.token;
        pending_[pending_index].metadata = g.metadata;
        ++pending_count_;
        ++sealed_;
        ++sealed_by_group_[g.metadata.behavior_group];
        g = {};
        return EcBatchStatus::kOk;
    }
    static constexpr size_t kOpenGroupCount =
        kEcBatchBehaviorGroupCount * ::FarLib::allocator::RegionBinCount;

    struct Open {
        uint64_t token = 0;
        Bank::Entry *entry = nullptr;
        GroupMetadata metadata{};
    };

    struct Pending {
        uint64_t token = 0;
        GroupMetadata metadata{};
    };

    static size_t allocator_bin_for_size(size_t size) {
        const size_t bin = ::FarLib::allocator::bin_from_wsize(
            ::FarLib::allocator::wsize_from_size(size));
        assert(bin < ::FarLib::allocator::RegionBinCount);
        assert(::FarLib::allocator::get_bin_size(bin) >= size);
        return bin;
    }

    size_t physical_key_for(size_t behavior_group, size_t allocator_bin) const {
        if (!size_classed_) return behavior_group;
        assert(allocator_bin < ::FarLib::allocator::RegionBinCount);
        return behavior_group * ::FarLib::allocator::RegionBinCount +
               allocator_bin;
    }

    SmallObjectStripeManager *manager_;
    Bank *bank_;
    size_t owner_;
    profile::evict_breakdown::Worker *breakdown_ = nullptr;
    bool size_classed_ = false;
    std::array<Open, kOpenGroupCount> open_{};
    std::array<Pending, Bank::kSlotsPerOwner> pending_{};
    size_t pending_head_ = 0, pending_count_ = 0;
    uint64_t sealed_ = 0, allocated_ = 0, allocation_failed_ = 0, objects_ = 0, data_bytes_ = 0;
    std::array<uint64_t, kEcBatchBehaviorGroupCount> sealed_by_group_{};
};
} // namespace FarLib::cache::ec_batch
