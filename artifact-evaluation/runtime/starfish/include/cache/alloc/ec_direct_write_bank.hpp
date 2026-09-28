#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>

#include "cache/alloc/small_object_stripe.hpp"

namespace FarLib::cache::ec_batch {

// Persistent owner-local storage for a direct EC write round.  The bank owns
// only parity and a read-only zero page; the four data pointers in Record are
// the caller's original object addresses.  No object payload is copied here.
class EcDirectWriteBank final {
public:
    static constexpr size_t kMaxOwners = 64;
    static constexpr size_t kSlotsPerOwner = 128;
    static constexpr size_t kDataSlots = 4;
    static constexpr size_t kParitySlots = 2;
    static constexpr size_t kSegmentsPerGroup = kDataSlots + kParitySlots;
    static constexpr size_t kSlotBytes = 4096;
    static constexpr size_t kParityBytesPerSlot = kParitySlots * kSlotBytes;
    static constexpr size_t kZeroPadBytes = kSlotBytes;
    static constexpr size_t kBankBytes =
        kSlotsPerOwner * kParityBytesPerSlot + kZeroPadBytes;

    // The direct token occupies the low 56-bit EC token namespace.  The old
    // EC table must route this bit-55 namespace here before decoding its own
    // slot/generation layout.
    static constexpr uint64_t kTokenTagBit = uint64_t{1} << 55;
    static constexpr uint64_t kTokenMask = (uint64_t{1} << 56) - 1;
    static constexpr uint64_t kTokenSlotMask = 0xff;
    static constexpr uint64_t kTokenOwnerMask = 0xff;
    static constexpr unsigned kTokenOwnerShift = 8;
    static constexpr unsigned kTokenGenerationShift = 16;
    static constexpr size_t kGenerationBits = 39;
    static constexpr uint64_t kGenerationMask =
        (uint64_t{1} << kGenerationBits) - 1;
    static_assert(kMaxOwners <= 256 && kSlotsPerOwner <= 256);
    static_assert(kTokenGenerationShift + kGenerationBits == 55);

    struct RegisteredBank {
        void *base = nullptr;
        size_t bytes = 0;
        uint32_t lkey = 0;
        void *registration = nullptr;
    };

    using Allocate = bool (*)(void *, size_t, RegisteredBank *);
    using Free = void (*)(void *, RegisteredBank &);

    // The caller fills group/object metadata in-place between reserve() and
    // publish().  parity/zero_pad/lkey and owner/slot/generation are installed
    // by reserve() and are validated again by publish().
    struct Record {
        SmallObjectStripeManager::SlotGroupHandle group{};
        const void *objects[kDataSlots]{};
        uint32_t object_sizes[kDataSlots]{};
        void *parity[kParitySlots]{};
        void *zero_pad = nullptr;
        uint32_t parity_lkey = 0;
        uint32_t slot_size = static_cast<uint32_t>(kSlotBytes);
        uint8_t live_mask = 0;
        uint8_t live_count = 0;
        uint16_t owner = 0;
        uint16_t slot = 0;
        uint64_t generation = 0;
    };

    struct Lease {
        void *parity[kParitySlots]{};
        void *zero_pad = nullptr;
        uint32_t lkey = 0;
        size_t owner = 0;
        size_t slot = 0;
        uint64_t generation = 0;
    };

    // Entry is persistent metadata.  Its Record is written by the reserving
    // owner while the state is reserved, then becomes immutable at publish.
    // Completion threads only inspect it after complete_segment() returns the
    // finalizer pointer.
    // Completion CAS must not invalidate the next slot's group/record fields.
    // A few padding cache lines per owner are cheaper than cross-core ping-pong.
    struct alignas(64) Entry {
        Record record{};
        uint8_t pending = 0;  // published snapshot; becomes zero at final CQE
        uint8_t acked = 0;    // published snapshot of terminal segments
        uint8_t failed = 0;   // published snapshot of failed segments

        size_t segment_count() const noexcept { return kSegmentsPerGroup; }

        unsigned durable_segment_count() const noexcept {
            return static_cast<unsigned>(__builtin_popcount(
                static_cast<unsigned>(acked & static_cast<uint8_t>(~failed))));
        }

        bool recoverable() const noexcept {
            return pending == 0 &&
                   durable_segment_count() >= kDataSlots;
        }

    private:
        // [0,38] generation; [39] active; [40,45] completed; [46,51]
        // failed; [52] finalizer claimed; [53] reserved; [54] retired;
        // [55] publication in progress.
        std::atomic<uint64_t> state{0};
        friend class EcDirectWriteBank;
    };

    explicit EcDirectWriteBank(size_t owners)
        : owner_count_(checked_owner_count(owners)),
          entries_(std::make_unique<Entry[]>(owner_count_ * kSlotsPerOwner)),
          cursors_(std::make_unique<OwnerCursor[]>(owner_count_)) {}

    EcDirectWriteBank(const EcDirectWriteBank &) = delete;
    EcDirectWriteBank &operator=(const EcDirectWriteBank &) = delete;

    ~EcDirectWriteBank() { release_banks_if_quiescent(); }

    // Initialization is single-threaded and must precede reserve/publish.
    // Each owner receives one registered bank; a partial failure rolls back
    // all earlier banks and leaves the object retryable.
    bool init(void *context, Allocate allocate, Free free) {
        if (initialized_ || allocate == nullptr || free == nullptr) {
            return false;
        }
        size_t allocated = 0;
        for (; allocated < owner_count_; ++allocated) {
            RegisteredBank bank{};
            bank.bytes = kBankBytes;
            if (!allocate(context, kBankBytes, &bank) ||
                bank.base == nullptr || bank.bytes < kBankBytes) {
                if (bank.base != nullptr || bank.registration != nullptr) {
                    free(context, bank);
                }
                for (size_t owner = 0; owner < allocated; ++owner) {
                    free(context, banks_[owner]);
                    banks_[owner] = RegisteredBank{};
                }
                return false;
            }
            // Registration callbacks need not return zero-filled allocation.
            // This shared tail page is immutable throughout the bank lifetime.
            std::memset(static_cast<uint8_t *>(bank.base) +
                            kSlotsPerOwner * kParityBytesPerSlot,
                        0, kZeroPadBytes);
            banks_[allocated] = bank;
        }
        context_ = context;
        free_ = free;
        initialized_ = true;
        return true;
    }

    bool initialized() const noexcept { return initialized_; }
    size_t owner_count() const noexcept { return owner_count_; }
    size_t slots_per_owner() const noexcept { return kSlotsPerOwner; }

    static bool is_direct_token(uint64_t token) noexcept {
        if ((token & ~kTokenMask) != 0 ||
            (token & kTokenTagBit) == 0) {
            return false;
        }
        const uint64_t generation =
            (token >> kTokenGenerationShift) & kGenerationMask;
        return generation != 0;
    }

    // Reserves metadata and a fixed parity/zero-pad lease.  The returned
    // Entry remains invisible to get()/complete_segment() until publish().
    bool reserve(size_t owner, uint64_t *token_id_out, Entry **entry_out,
                 Lease *lease_out) {
        if (!initialized_ || token_id_out == nullptr || entry_out == nullptr ||
            lease_out == nullptr || owner >= owner_count_) {
            return false;
        }

        size_t slot = 0;
        uint64_t token = 0;
        Entry *entry = nullptr;
        if (!reserve_slot(owner, &slot, &token, &entry)) return false;

        entry->record = Record{};
        entry->record.slot_size = static_cast<uint32_t>(kSlotBytes);
        entry->record.owner = static_cast<uint16_t>(owner);
        entry->record.slot = static_cast<uint16_t>(slot);
        entry->record.generation = generation_from_token(token);

        *lease_out = make_lease(owner, slot, generation_from_token(token));
        for (size_t i = 0; i < kParitySlots; ++i) {
            entry->record.parity[i] = lease_out->parity[i];
        }
        entry->record.zero_pad = lease_out->zero_pad;
        entry->record.parity_lkey = lease_out->lkey;
        entry->pending = 0;
        entry->acked = 0;
        entry->failed = 0;
        *token_id_out = token;
        *entry_out = entry;
        return true;
    }

    // Publishes the caller-filled Entry in place.  No Record copy occurs.
    // Only fixed bank bindings and basic group metadata are checked here.
    bool publish(uint64_t token) noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return false;
        Entry &entry = entry_at(decoded.owner, decoded.slot);
        if (!record_belongs_to_lease(entry.record, decoded)) return false;
        uint64_t observed = entry.state.load(std::memory_order_acquire);
        for (;;) {
            if (!matches_reserved(observed, decoded.generation)) return false;
            const uint64_t preparing = observed | kStatePreparingBit;
            if (!entry.state.compare_exchange_weak(
                    observed, preparing, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            entry.pending = static_cast<uint8_t>(kSegmentsPerGroup);
            entry.acked = 0;
            entry.failed = 0;
            entry.state.store(
                pack_state(decoded.generation, true, 0, 0, false, false,
                           false),
                std::memory_order_release);
            return true;
        }
    }

    // Returns only a published active Entry.  The caller must not retain the
    // pointer across release or a subsequent reuse of the same token slot.
    Entry *get(uint64_t token) noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return nullptr;
        Entry &entry = entry_at(decoded.owner, decoded.slot);
        const uint64_t state = entry.state.load(std::memory_order_acquire);
        return matches_active(state, decoded.generation) ? &entry : nullptr;
    }

    const Entry *get(uint64_t token) const noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return nullptr;
        const Entry &entry = entry_at(decoded.owner, decoded.slot);
        const uint64_t state = entry.state.load(std::memory_order_acquire);
        return matches_active(state, decoded.generation) ? &entry : nullptr;
    }

    // The unique CAS winner of the sixth terminal segment receives the Entry.
    // Earlier callers and duplicate/stale completions receive nullptr.
    Entry *complete_segment(uint64_t token, uint8_t segment,
                            bool success = true) noexcept {
        if (segment >= kSegmentsPerGroup) return nullptr;
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return nullptr;
        Entry &entry = entry_at(decoded.owner, decoded.slot);
        const uint64_t bit = uint64_t{1} << segment;
        for (;;) {
            uint64_t observed = entry.state.load(std::memory_order_acquire);
            if (!matches_active(observed, decoded.generation)) return nullptr;
            const uint8_t completed = state_completed(observed);
            if ((completed & static_cast<uint8_t>(bit)) != 0) return nullptr;
            const uint8_t next_completed = static_cast<uint8_t>(
                completed | static_cast<uint8_t>(bit));
            const uint8_t next_failed = static_cast<uint8_t>(
                state_failed(observed) |
                (success ? 0 : static_cast<uint8_t>(bit)));
            const bool last = next_completed == kAllSegmentsMask;
            uint64_t desired = observed |
                               (bit << kStateCompletedShift);
            if (!success) desired |= bit << kStateFailedShift;
            if (last) desired |= kStateClaimedBit;
            if (!entry.state.compare_exchange_weak(
                    observed, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            if (last) {
                entry.pending = 0;
                entry.acked = next_completed;
                entry.failed = next_failed;
                return &entry;
            }
            return nullptr;
        }
    }

    // Only a finalizer after all six terminal outcomes may release a slot.
    bool release(uint64_t token) noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return false;
        Entry &entry = entry_at(decoded.owner, decoded.slot);
        for (;;) {
            uint64_t observed = entry.state.load(std::memory_order_acquire);
            if (!matches_generation_active(observed, decoded.generation) ||
                !state_claimed(observed) ||
                state_completed(observed) != kAllSegmentsMask ||
                entry.pending != 0) {
                return false;
            }
            const bool retire = decoded.generation == kGenerationMask;
            const uint64_t free_state = pack_state(
                decoded.generation, false, 0, 0, false, false, retire);
            if (entry.state.compare_exchange_weak(
                    observed, free_state, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
    }

    // Cancels a reservation that has not been published.  It cannot race a
    // completion because reserved entries are not visible to completion.
    bool cancel(uint64_t token) noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return false;
        Entry &entry = entry_at(decoded.owner, decoded.slot);
        for (;;) {
            uint64_t observed = entry.state.load(std::memory_order_acquire);
            if (!matches_reserved(observed, decoded.generation)) return false;
            const bool retire = decoded.generation == kGenerationMask;
            const uint64_t free_state = pack_state(
                decoded.generation, false, 0, 0, false, false, retire);
            if (entry.state.compare_exchange_weak(
                    observed, free_state, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
    }

    size_t in_use() const noexcept {
        size_t count = 0;
        for (size_t owner = 0; owner < owner_count_; ++owner) {
            for (size_t slot = 0; slot < kSlotsPerOwner; ++slot) {
                const uint64_t state =
                    entry_at(owner, slot).state.load(std::memory_order_acquire);
                if (state_active(state) || state_reserved(state)) ++count;
            }
        }
        return count;
    }

    size_t available(size_t owner) const noexcept {
        if (owner >= owner_count_) return 0;
        size_t count = 0;
        for (size_t slot = 0; slot < kSlotsPerOwner; ++slot) {
            const uint64_t state =
                entry_at(owner, slot).state.load(std::memory_order_acquire);
            if (!state_active(state) && !state_reserved(state) &&
                !state_retired(state)) {
                ++count;
            }
        }
        return count;
    }

private:
    static constexpr unsigned kStateActiveShift = 39;
    static constexpr unsigned kStateCompletedShift = 40;
    static constexpr unsigned kStateFailedShift = 46;
    static constexpr unsigned kStateClaimedShift = 52;
    static constexpr unsigned kStateReservedShift = 53;
    static constexpr unsigned kStateRetiredShift = 54;
    static constexpr uint64_t kStateActiveBit = uint64_t{1}
                                                 << kStateActiveShift;
    static constexpr uint64_t kStateClaimedBit = uint64_t{1}
                                                  << kStateClaimedShift;
    static constexpr uint64_t kStateReservedBit = uint64_t{1}
                                                   << kStateReservedShift;
    static constexpr uint64_t kStateRetiredBit = uint64_t{1}
                                                  << kStateRetiredShift;
    static constexpr uint64_t kStatePreparingBit = uint64_t{1} << 55;
    static constexpr uint8_t kAllSegmentsMask = static_cast<uint8_t>(
        (uint8_t{1} << kSegmentsPerGroup) - uint8_t{1});

    struct DecodedToken {
        size_t owner = 0;
        size_t slot = 0;
        uint64_t generation = 0;
    };

    // One logical eviction producer owns each owner cursor. Completion
    // threads never reserve slots, so this is a plain cursor rather than an
    // unnecessary shared RMW hotspot.
    struct OwnerCursor {
        size_t next = 0;
    };

    static size_t checked_owner_count(size_t owners) {
        if (owners == 0 || owners > kMaxOwners) {
            throw std::invalid_argument(
                "EcDirectWriteBank requires 1..64 owners");
        }
        return owners;
    }

    static uint64_t state_generation(uint64_t state) noexcept {
        return state & kGenerationMask;
    }
    static bool state_active(uint64_t state) noexcept {
        return (state & kStateActiveBit) != 0;
    }
    static bool state_claimed(uint64_t state) noexcept {
        return (state & kStateClaimedBit) != 0;
    }
    static bool state_reserved(uint64_t state) noexcept {
        return (state & kStateReservedBit) != 0;
    }
    static bool state_retired(uint64_t state) noexcept {
        return (state & kStateRetiredBit) != 0;
    }
    static bool state_preparing(uint64_t state) noexcept {
        return (state & kStatePreparingBit) != 0;
    }
    static uint8_t state_completed(uint64_t state) noexcept {
        return static_cast<uint8_t>((state >> kStateCompletedShift) &
                                    kAllSegmentsMask);
    }
    static uint8_t state_failed(uint64_t state) noexcept {
        return static_cast<uint8_t>((state >> kStateFailedShift) &
                                    kAllSegmentsMask);
    }

    static uint64_t pack_state(uint64_t generation, bool active,
                               uint8_t completed, uint8_t failed,
                               bool claimed, bool reserved,
                               bool retired) noexcept {
        uint64_t state = generation & kGenerationMask;
        state |= (static_cast<uint64_t>(completed) & kAllSegmentsMask)
                 << kStateCompletedShift;
        state |= (static_cast<uint64_t>(failed) & kAllSegmentsMask)
                 << kStateFailedShift;
        if (active) state |= kStateActiveBit;
        if (claimed) state |= kStateClaimedBit;
        if (reserved) state |= kStateReservedBit;
        if (retired) state |= kStateRetiredBit;
        return state;
    }

    static uint64_t make_token(size_t owner, size_t slot,
                               uint64_t generation) noexcept {
        return kTokenTagBit | static_cast<uint64_t>(slot) |
               (static_cast<uint64_t>(owner) << kTokenOwnerShift) |
               ((generation & kGenerationMask) << kTokenGenerationShift);
    }

    static uint64_t generation_from_token(uint64_t token) noexcept {
        return (token >> kTokenGenerationShift) & kGenerationMask;
    }

    bool decode_token(uint64_t token, DecodedToken *out) const noexcept {
        if (out == nullptr || !is_direct_token(token)) return false;
        const size_t owner = static_cast<size_t>(
            (token >> kTokenOwnerShift) & kTokenOwnerMask);
        const size_t slot = static_cast<size_t>(token & kTokenSlotMask);
        const uint64_t generation = generation_from_token(token);
        if (owner >= owner_count_ || slot >= kSlotsPerOwner || generation == 0) {
            return false;
        }
        out->owner = owner;
        out->slot = slot;
        out->generation = generation;
        return true;
    }

    static bool matches_active(uint64_t state, uint64_t generation) noexcept {
        return state_active(state) && !state_reserved(state) &&
               !state_retired(state) && !state_claimed(state) &&
               state_generation(state) == generation;
    }

    static bool matches_generation_active(uint64_t state,
                                          uint64_t generation) noexcept {
        return state_active(state) && !state_reserved(state) &&
               !state_retired(state) &&
               state_generation(state) == generation;
    }

    static bool matches_reserved(uint64_t state, uint64_t generation) noexcept {
        return state_reserved(state) && !state_active(state) &&
               !state_preparing(state) && !state_retired(state) &&
               state_generation(state) == generation;
    }

    Entry &entry_at(size_t owner, size_t slot) noexcept {
        return entries_[owner * kSlotsPerOwner + slot];
    }
    const Entry &entry_at(size_t owner, size_t slot) const noexcept {
        return entries_[owner * kSlotsPerOwner + slot];
    }

    Lease make_lease(size_t owner, size_t slot,
                    uint64_t generation) const noexcept {
        Lease lease{};
        auto *base = static_cast<uint8_t *>(banks_[owner].base);
        const size_t offset = slot * kParityBytesPerSlot;
        lease.parity[0] = base + offset;
        lease.parity[1] = base + offset + kSlotBytes;
        lease.zero_pad = base + kSlotsPerOwner * kParityBytesPerSlot;
        lease.lkey = banks_[owner].lkey;
        lease.owner = owner;
        lease.slot = slot;
        lease.generation = generation;
        return lease;
    }

    bool record_belongs_to_lease(const Record &record,
                                 const DecodedToken &decoded) const noexcept {
        if (record.owner != decoded.owner || record.slot != decoded.slot ||
            record.generation != decoded.generation ||
            record.slot_size == 0 || record.slot_size > kSlotBytes ||
            record.parity_lkey != banks_[decoded.owner].lkey) {
            return false;
        }
        const Lease expected = make_lease(decoded.owner, decoded.slot,
                                          decoded.generation);
        return record.parity[0] == expected.parity[0] &&
               record.parity[1] == expected.parity[1] &&
               record.zero_pad == expected.zero_pad;
    }

    bool reserve_slot(size_t owner, size_t *slot_out, uint64_t *token_out,
                      Entry **entry_out) noexcept {
        const size_t first = cursors_[owner].next % kSlotsPerOwner;
        for (size_t offset = 0; offset < kSlotsPerOwner; ++offset) {
            const size_t slot = (first + offset) % kSlotsPerOwner;
            Entry &entry = entry_at(owner, slot);
            uint64_t observed = entry.state.load(std::memory_order_acquire);
            for (;;) {
                if (state_active(observed) || state_reserved(observed) ||
                    state_retired(observed)) {
                    break;
                }
                const uint64_t generation = state_generation(observed);
                if (generation == kGenerationMask) {
                    const uint64_t retired = observed | kStateRetiredBit;
                    if (!entry.state.compare_exchange_weak(
                            observed, retired, std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
                        continue;
                    }
                    break;
                }
                const uint64_t next_generation = generation + 1;
                const uint64_t reserved = pack_state(
                    next_generation, false, 0, 0, false, true, false);
                if (!entry.state.compare_exchange_weak(
                        observed, reserved, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    continue;
                }
                cursors_[owner].next = (slot + 1) % kSlotsPerOwner;
                *slot_out = slot;
                *token_out = make_token(owner, slot, next_generation);
                *entry_out = &entry;
                return true;
            }
        }
        return false;
    }

    void release_banks_if_quiescent() noexcept {
        if (!initialized_) return;
        if (free_ == nullptr || in_use() != 0) std::terminate();
        for (size_t owner = 0; owner < owner_count_; ++owner) {
            free_(context_, banks_[owner]);
            banks_[owner] = RegisteredBank{};
        }
        initialized_ = false;
        context_ = nullptr;
        free_ = nullptr;
    }

    size_t owner_count_ = 0;
    std::unique_ptr<Entry[]> entries_;
    std::unique_ptr<OwnerCursor[]> cursors_;
    std::array<RegisteredBank, kMaxOwners> banks_{};
    void *context_ = nullptr;
    Free free_ = nullptr;
    bool initialized_ = false;
};

}  // namespace FarLib::cache::ec_batch
