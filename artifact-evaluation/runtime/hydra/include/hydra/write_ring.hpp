// Bounded owner-local completion rings for Hydra's six-segment writes.
//
// The inherited ec_batch token table is intentionally still used by the
// ordinary EC paths.  Hydra page writes use this ring instead: every logical
// eviction worker owns a fixed set of slots, so unrelated workers do not
// consume one global token quota or take a completion mutex.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

#include "cache/alloc/ec_batch_write.hpp"

namespace FarLib::hydra {

using EcGroupSendRecord = ::FarLib::cache::ec_batch::EcGroupSendRecord;
using EcStagingGroupSlot =
    ::FarLib::cache::ec_batch::EcStagingGroupSlot;

struct RegisteredParity {
    void *base = nullptr;
    uint32_t lkey = 0;
    void *registration = nullptr;
};

using Allocate = bool (*)(void *, size_t, RegisteredParity *);
using Free = void (*)(void *, RegisteredParity);

class WriteRing final {
public:
    static constexpr size_t kMaxOwners = 256;
    static constexpr size_t kMaxSlotsPerOwner = 256;
    // Banked mode fills A sequentially, then B; a bank is reused only after
    // all its page finalizers have finished. No free-slot search in this mode.
    static constexpr size_t kPagesPerBuffer = 32;
    static constexpr size_t kBuffersPerOwner = 2;
    static constexpr size_t kDoubleBufferedSlots = kPagesPerBuffer * kBuffersPerOwner;
    static constexpr size_t kLegacySlotsPerOwner = 64;
    static_assert(kDoubleBufferedSlots <= kMaxSlotsPerOwner);
    static constexpr size_t kGenerationBits = 39;
    static constexpr uint64_t kGenerationMask =
        (uint64_t{1} << kGenerationBits) - 1;

    // The existing ec_batch wr_id leaves bits 0..55 for its token.  Bit 55 is
    // reserved for this ring namespace; the remaining fields are deliberately
    // fixed-width so a token can be carried by the old encoder unchanged.
    static constexpr uint64_t kTokenTagBit = uint64_t{1} << 55;
    static constexpr uint64_t kTokenMask = (uint64_t{1} << 56) - 1;
    static constexpr uint64_t kTokenSlotMask = 0xff;
    static constexpr uint64_t kTokenOwnerMask = 0xff;
    static constexpr unsigned kTokenOwnerShift = 8;
    static constexpr unsigned kTokenGenerationShift = 16;
    static constexpr size_t kParitySegmentBytes = 2048;
    static constexpr size_t kParitySlotBytes = kParitySegmentBytes * 2;

    using RegisteredParity = ::FarLib::hydra::RegisteredParity;
    using EcStagingGroupSlot = ::FarLib::hydra::EcStagingGroupSlot;
    using Allocate = ::FarLib::hydra::Allocate;
    using Free = ::FarLib::hydra::Free;

    static_assert(::FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup == 6,
                  "Hydra write rings require six EC segments");

    // A slot's record is immutable from the moment acquire() publishes the
    // active state until release().  The byte fields are atomics so diagnostic
    // readers may inspect them while a completion round is settling; the
    // packed state below remains the source of truth for all transitions.
    struct Entry {
        EcGroupSendRecord record{};
        std::atomic<uint8_t> pending{0};
        std::atomic<uint8_t> acked{0};
        std::atomic<uint8_t> failed{0};

        size_t segment_count() const noexcept {
            return static_cast<size_t>(record.group.segments.size());
        }

        unsigned durable_segment_count() const noexcept {
            const uint8_t done = acked.load(std::memory_order_acquire);
            const uint8_t bad = failed.load(std::memory_order_acquire);
            return static_cast<unsigned>(__builtin_popcount(
                static_cast<unsigned>(done & static_cast<uint8_t>(~bad))));
        }

        bool recoverable() const noexcept {
            return pending.load(std::memory_order_acquire) == 0 &&
                   durable_segment_count() >=
                       ::FarLib::cache::ec_batch::kEcBatchDataSlots;
        }

    private:
        // [0,38] generation; [39] active; [40,45] completed; [46,51]
        // failed; [52] finalizer claimed; [53] acquire reservation; [54]
        // generation retired; [55] page-record publication in progress.
        std::atomic<uint64_t> state{0};
        friend class WriteRing;
    };

    explicit WriteRing(size_t owners, size_t slots_per_owner = kLegacySlotsPerOwner,
                       bool banked = false)
        : owner_count_(owners), slots_per_owner_(slots_per_owner), banked_(banked),
          entries_(std::make_unique<Entry[]>(
              checked_capacity(owners, slots_per_owner))),
          cursors_(std::make_unique<OwnerCursor[]>(owners)) {
        if (banked_ && slots_per_owner_ != kDoubleBufferedSlots)
            throw std::invalid_argument("Hydra banked WRITE requires two 32-page buffers");
        if (banked_) {
            bank_producers_ = std::make_unique<BankProducer[]>(owners);
            bank_counters_ = std::make_unique<BankCounter[]>(owners * kBuffersPerOwner);
        }
        parity_banks_ = std::make_unique<RegisteredParity[]>(owner_count_);
        for (size_t owner = 0; owner < owner_count_; ++owner) {
            cursors_[owner].next.store(0, std::memory_order_relaxed);
        }
    }

    WriteRing(const WriteRing &) = delete;
    WriteRing &operator=(const WriteRing &) = delete;

    ~WriteRing() { release_parity_buffers_if_quiescent(); }

    // Allocate one registered parity bank per logical owner.  A successful
    // call is one-shot; a failed call rolls back every bank allocated so far
    // and leaves the ring retryable.  The callback owns the registration
    // handle semantics; this ring only stores and compares the returned
    // metadata.
    bool init_parity_buffers(void *context, Allocate allocate,
                             Free free) {
        if (parity_initialized_ || allocate == nullptr || free == nullptr) {
            return false;
        }

        const size_t bytes = slots_per_owner_ * kParitySlotBytes;
        size_t allocated = 0;
        for (; allocated < owner_count_; ++allocated) {
            RegisteredParity bank{};
            if (!allocate(context, bytes, &bank) || bank.base == nullptr) {
                if (bank.base != nullptr || bank.registration != nullptr) {
                    free(context, bank);
                }
                for (size_t i = 0; i < allocated; ++i) {
                    free(context, parity_banks_[i]);
                    parity_banks_[i] = RegisteredParity{};
                }
                return false;
            }
            parity_banks_[allocated] = bank;
        }

        parity_context_ = context;
        parity_free_ = free;
        parity_initialized_ = true;
        return true;
    }

    bool parity_initialized() const noexcept { return parity_initialized_; }

    size_t parity_bytes() const noexcept {
        return parity_initialized_ ? capacity() * kParitySlotBytes : 0;
    }

    // True only for a syntactically valid ring token.  This function does not
    // turn any token bits into a pointer; bounds and generation checks happen
    // again against this ring before an Entry is returned.
    static bool is_token(uint64_t token) noexcept {
        if ((token & ~kTokenMask) != 0 || (token & kTokenTagBit) == 0) {
            return false;
        }
        return ((token >> kTokenGenerationShift) & kGenerationMask) != 0;
    }

    // Legacy mode scans slots. Banked mode requires one logical producer per
    // owner (the same contract as WriteBatch), directly indexes the next slot,
    // and only checks a whole-bank completion counter when wrapping to a bank.
    bool acquire(size_t owner, const EcGroupSendRecord &record,
                 uint64_t *token_id_out, Entry **entry_out) {
        size_t slot = 0;
        if (!reserve_slot(owner, token_id_out, entry_out, &slot)) {
            return false;
        }

        Entry &entry = **entry_out;
        entry.record = record;
        entry.pending.store(
            static_cast<uint8_t>(
                ::FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup),
            std::memory_order_relaxed);
        entry.acked.store(0, std::memory_order_relaxed);
        entry.failed.store(0, std::memory_order_relaxed);
        const uint64_t generation =
            (*token_id_out >> kTokenGenerationShift) & kGenerationMask;
        entry.state.store(pack_state(generation, true, 0, 0, false, false,
                                     false),
                          std::memory_order_release);
        return true;
    }

    // Reserve the persistent ring entry and its owner-local parity slot, but
    // do not publish a record yet.  Until publish_page(), completion callbacks
    // see only the reserved state and cannot dereference a half-built record.
    bool reserve_page(size_t owner, uint64_t *token_id_out,
                      Entry **entry_out,
                      EcStagingGroupSlot *scratch_out) {
        if (!parity_initialized_ || token_id_out == nullptr ||
            entry_out == nullptr || scratch_out == nullptr ||
            owner >= owner_count_ || parity_banks_[owner].base == nullptr) {
            return false;
        }

        size_t slot = 0;
        if (!reserve_slot(owner, token_id_out, entry_out, &slot)) {
            return false;
        }

        const uint64_t generation =
            (*token_id_out >> kTokenGenerationShift) & kGenerationMask;
        *scratch_out = make_parity_scratch(owner, slot, generation);
        return true;
    }

    // Publish a fully prepared record.  The supplied staging descriptor must
    // be the fixed parity slot returned by reserve_page; foreign pointers are
    // rejected before the record becomes visible to completion.
    bool publish_page(uint64_t token, const EcGroupSendRecord &record) {
        if (!parity_initialized_) return false;
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return false;
        Entry &entry = entries_[decoded.owner * slots_per_owner_ +
                                decoded.slot];

        if (!owns_parity(token, record.staging)) return false;

        uint64_t observed = entry.state.load(std::memory_order_acquire);
        for (;;) {
            if (!matches_reserved(observed, decoded.generation)) {
                return false;
            }
            const uint64_t preparing = observed | kStatePreparingBit;
            if (!entry.state.compare_exchange_weak(
                    observed, preparing, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }

            entry.record = record;
            entry.pending.store(
                static_cast<uint8_t>(
                    ::FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup),
                std::memory_order_relaxed);
            entry.acked.store(0, std::memory_order_relaxed);
            entry.failed.store(0, std::memory_order_relaxed);
            entry.state.store(pack_state(decoded.generation, true, 0, 0,
                                         false, false, false),
                              std::memory_order_release);
            return true;
        }
    }

    // Cancel is valid only for a page reservation that has not been
    // published.  The generation remains in the state so the canceled token
    // cannot match a later reuse of the fixed parity address.
    bool cancel_page(uint64_t token) noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return false;
        Entry &entry = entries_[decoded.owner * slots_per_owner_ +
                                decoded.slot];
        for (;;) {
            uint64_t observed = entry.state.load(std::memory_order_acquire);
            if (!matches_reserved(observed, decoded.generation)) return false;
            const bool retire = decoded.generation == kGenerationMask;
            const uint64_t canceled = pack_state(
                decoded.generation, false, 0, 0, false, false, retire);
            if (entry.state.compare_exchange_weak(
                    observed, canceled, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                finish_bank_page(decoded);
                return true;
            }
        }
    }

    // Validate only integer metadata and pointer values.  No caller-provided
    // address is dereferenced, and the token is checked against this ring
    // before it indexes any persistent metadata.
    bool owns_parity(uint64_t token,
                     const EcStagingGroupSlot &scratch) const noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded) || !parity_initialized_) {
            return false;
        }
        const uint64_t state = entries_[decoded.owner * slots_per_owner_ +
                                       decoded.slot]
                                   .state.load(std::memory_order_acquire);
        if (!matches_parity_lifetime(state, decoded.generation)) return false;

        const EcStagingGroupSlot expected =
            make_parity_scratch(decoded.owner, decoded.slot,
                                decoded.generation);
        for (size_t i = 0;
             i < ::FarLib::cache::ec_batch::kEcBatchDataSlots; ++i) {
            if (scratch.data[i] != expected.data[i]) return false;
        }
        return scratch.parity[0] == expected.parity[0] &&
               scratch.parity[1] == expected.parity[1] &&
               scratch.slot_size == expected.slot_size &&
               scratch.index == expected.index &&
               scratch.mr_offset == expected.mr_offset &&
               scratch.lkey == expected.lkey &&
               scratch.generation == expected.generation &&
               scratch.lease_cookie == expected.lease_cookie &&
               scratch.pool_cookie == expected.pool_cookie;
    }

    // Account one terminal segment.  The final completion sets the claimed
    // bit in the same CAS that sets its segment bit, so exactly one caller gets
    // the Entry pointer.  Duplicate, stale, malformed, and post-finalizer
    // callbacks are ignored.
    Entry *complete_segment(uint64_t token, uint8_t segment,
                            bool success = true) noexcept {
        if (segment >=
            ::FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup) {
            return nullptr;
        }
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return nullptr;
        Entry &entry = entries_[decoded.owner * slots_per_owner_ +
                                decoded.slot];
        const uint64_t bit = uint64_t{1} << segment;
        for (;;) {
            uint64_t observed =
                entry.state.load(std::memory_order_acquire);
            if (!matches_active(observed, decoded.generation)) return nullptr;
            const uint8_t completed = state_completed(observed);
            if ((completed & static_cast<uint8_t>(bit)) != 0) {
                return nullptr;
            }
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
                    observed, desired,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }

            // Only the final completion publishes the finalizer-visible
            // snapshot.  Earlier callbacks may finish in either order and
            // must never overwrite a newer mask, especially after release and
            // reuse of this persistent Entry.
            if (last) {
                entry.acked.store(next_completed, std::memory_order_release);
                entry.failed.store(next_failed, std::memory_order_release);
                entry.pending.store(
                    static_cast<uint8_t>(
                        ::FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup -
                        __builtin_popcount(
                            static_cast<unsigned>(next_completed))),
                    std::memory_order_release);
            }
            return last ? &entry : nullptr;
        }
    }

    // Only the finalizer that received the last-completion pointer may release
    // a live entry.  Releasing early is rejected, which keeps the record and
    // owner-local staging lease stable until all six terminal outcomes exist.
    bool release(uint64_t token) noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return false;
        Entry &entry = entries_[decoded.owner * slots_per_owner_ + decoded.slot];
        for (;;) {
            uint64_t observed =
                entry.state.load(std::memory_order_acquire);
            if (!matches_generation_active(observed, decoded.generation) ||
                !state_claimed(observed) ||
                state_completed(observed) != kAllSegmentsMask) {
                return false;
            }
            const bool retire = decoded.generation == kGenerationMask;
            const uint64_t desired = pack_state(
                decoded.generation, false, 0, 0, false, false, retire);
            if (!entry.state.compare_exchange_weak(
                    observed, desired,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            // Keep the completed masks available to a quiescent finalizer;
            // the next acquire resets them before publishing a new record.
            finish_bank_page(decoded);
            return true;
        }
    }

    size_t in_use() const noexcept {
        size_t count = 0;
        for (size_t i = 0; i < capacity(); ++i) {
            const uint64_t state = entries_[i].state.load(
                std::memory_order_acquire);
            if (state_active(state) || state_reserved(state)) ++count;
        }
        return count;
    }

    size_t capacity() const noexcept { return owner_count_ * slots_per_owner_; }

    size_t available(size_t owner) const noexcept {
        if (owner >= owner_count_) return 0;
        size_t count = 0;
        for (size_t slot = 0; slot < slots_per_owner_; ++slot) {
            const uint64_t state = entries_[owner * slots_per_owner_ + slot]
                                       .state.load(std::memory_order_acquire);
            if (!state_active(state) && !state_reserved(state) &&
                !state_retired(state)) {
                ++count;
            }
        }
        return count;
    }

    // Quiescent/debug lookup.  It performs the same token and generation
    // validation as completion; callers must not retain the pointer past
    // release or read record fields concurrently with the next acquire.
    Entry *get(uint64_t token) noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return nullptr;
        Entry &entry = entries_[decoded.owner * slots_per_owner_ + decoded.slot];
        const uint64_t state = entry.state.load(std::memory_order_acquire);
        return matches_generation_active(state, decoded.generation) ? &entry
                                                                     : nullptr;
    }

    const Entry *get(uint64_t token) const noexcept {
        DecodedToken decoded;
        if (!decode_token(token, &decoded)) return nullptr;
        const Entry &entry = entries_[decoded.owner * slots_per_owner_ + decoded.slot];
        const uint64_t state = entry.state.load(std::memory_order_acquire);
        return matches_generation_active(state, decoded.generation) ? &entry
                                                                     : nullptr;
    }

    size_t owner_count() const noexcept { return owner_count_; }
    size_t slots_per_owner() const noexcept { return slots_per_owner_; }
    bool banked() const noexcept { return banked_; }

    // Pure token arithmetic: safe even when an immediate completion already
    // released the page. The caller flushes after enqueueing all six segments.
    bool buffer_end(uint64_t token) const noexcept {
        DecodedToken decoded;
        return banked_ && decode_token(token, &decoded) &&
               decoded.slot % kPagesPerBuffer == kPagesPerBuffer - 1;
    }

private:
    static constexpr unsigned kStateActiveShift = 39;
    static constexpr unsigned kStateCompletedShift = 40;
    static constexpr unsigned kStateFailedShift = 46;
    static constexpr unsigned kStateClaimedShift = 52;
    static constexpr unsigned kStateReservedShift = 53;
    static constexpr unsigned kStateRetiredShift = 54;
    static constexpr uint64_t kStatePreparingBit = uint64_t{1} << 55;
    static constexpr uint64_t kStateActiveBit = uint64_t{1}
                                                 << kStateActiveShift;
    static constexpr uint64_t kStateClaimedBit = uint64_t{1}
                                                  << kStateClaimedShift;
    static constexpr uint64_t kStateReservedBit = uint64_t{1}
                                                   << kStateReservedShift;
    static constexpr uint64_t kStateRetiredBit = uint64_t{1}
                                                   << kStateRetiredShift;
    static constexpr uint8_t kAllSegmentsMask = static_cast<uint8_t>(
        (uint8_t{1} <<
         ::FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup) -
        uint8_t{1});

    struct DecodedToken {
        size_t owner = 0;
        size_t slot = 0;
        uint64_t generation = 0;
    };

    struct alignas(64) OwnerCursor {
        std::atomic<size_t> next{0};
    };

    struct alignas(64) BankProducer {
        size_t bank = 0;
        size_t next = 0;
        uint64_t generation[kBuffersPerOwner]{};
    };
    struct alignas(64) BankCounter {
        // Counts reserved/published pages, not WRs. Decrement happens only
        // after the page's six terminal CQEs AND its finalizer are finished.
        std::atomic<size_t> outstanding{0};
    };

    void finish_bank_page(const DecodedToken &token) noexcept {
        if (!banked_) return;
        auto &counter = bank_counters_[token.owner * kBuffersPerOwner +
                                      token.slot / kPagesPerBuffer].outstanding;
        if (counter.fetch_sub(1, std::memory_order_release) == 0)
            std::terminate();
    }

    bool reserve_banked_slot(size_t owner, uint64_t *token_id_out,
                             Entry **entry_out, size_t *slot_out) {
        auto &producer = bank_producers_[owner];
        const bool switch_bank = producer.next == kPagesPerBuffer;
        const size_t bank = switch_bank ? (producer.bank ^ 1) : producer.bank;
        auto &counter = bank_counters_[owner * kBuffersPerOwner + bank].outstanding;
        if (switch_bank || producer.next == 0) {
            // One acquire load per bank entry (or failed retry), never a scan.
            // A zero read synchronizes with every prior page finalizer in it.
            if (counter.load(std::memory_order_acquire) != 0) return false;
            if (producer.generation[bank] == kGenerationMask)
                throw std::overflow_error("Hydra WRITE bank generation exhausted");
            ++producer.generation[bank];
            producer.bank = bank;
            producer.next = 0;
        }
        const size_t slot = bank * kPagesPerBuffer + producer.next++;
        const uint64_t generation = producer.generation[bank];
        Entry &entry = entries_[owner * slots_per_owner_ + slot];
        // Account the reservation before it can be published or cancelled.
        counter.fetch_add(1, std::memory_order_relaxed);
        entry.pending.store(0, std::memory_order_relaxed);
        entry.acked.store(0, std::memory_order_relaxed);
        entry.failed.store(0, std::memory_order_relaxed);
        entry.state.store(pack_state(generation, false, 0, 0, false, true, false),
                          std::memory_order_release);
        *token_id_out = make_token(owner, slot, generation);
        *entry_out = &entry;
        *slot_out = slot;
        return true;
    }

    static size_t checked_capacity(size_t owners, size_t slots) {
        if (owners == 0 || owners > kMaxOwners || slots == 0 ||
            slots > kMaxSlotsPerOwner) {
            throw std::invalid_argument(
                "hydra::WriteRing requires 1..256 owners and slots");
        }
        return owners * slots;
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

    bool decode_token(uint64_t token, DecodedToken *out) const noexcept {
        if (out == nullptr || !is_token(token)) return false;
        const size_t owner = static_cast<size_t>(
            (token >> kTokenOwnerShift) & kTokenOwnerMask);
        const size_t slot = static_cast<size_t>(token & kTokenSlotMask);
        if (owner >= owner_count_ || slot >= slots_per_owner_) return false;
        out->owner = owner;
        out->slot = slot;
        out->generation = (token >> kTokenGenerationShift) & kGenerationMask;
        return out->generation != 0;
    }

    static bool matches_active(uint64_t state, uint64_t generation) noexcept {
        return matches_generation_active(state, generation) &&
               !state_claimed(state);
    }

    static bool matches_generation_active(uint64_t state,
                                          uint64_t generation) noexcept {
        return state_active(state) && !state_reserved(state) &&
               !state_retired(state) && state_generation(state) == generation;
    }

    static bool matches_reserved(uint64_t state,
                                 uint64_t generation) noexcept {
        return state_reserved(state) && !state_active(state) &&
               !state_preparing(state) && !state_retired(state) &&
               state_generation(state) == generation;
    }

    static bool matches_parity_lifetime(uint64_t state,
                                        uint64_t generation) noexcept {
        return (state_reserved(state) || state_active(state)) &&
               !state_retired(state) && state_generation(state) == generation;
    }

    bool reserve_slot(size_t owner, uint64_t *token_id_out,
                      Entry **entry_out, size_t *slot_out) {
        if (token_id_out == nullptr || entry_out == nullptr ||
            slot_out == nullptr || owner >= owner_count_) {
            return false;
        }
        if (banked_) return reserve_banked_slot(owner, token_id_out, entry_out, slot_out);

        const size_t first =
            cursors_[owner].next.fetch_add(1, std::memory_order_relaxed) %
            slots_per_owner_;
        for (size_t offset = 0; offset < slots_per_owner_; ++offset) {
            const size_t slot = (first + offset) % slots_per_owner_;
            Entry &entry = entries_[owner * slots_per_owner_ + slot];
            uint64_t observed = entry.state.load(std::memory_order_acquire);
            for (;;) {
                if (state_retired(observed) || state_active(observed) ||
                    state_reserved(observed)) {
                    break;
                }

                const uint64_t generation = state_generation(observed);
                if (generation == kGenerationMask) {
                    // Do not ever wrap the generation into a prior token.
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

                entry.pending.store(0, std::memory_order_relaxed);
                entry.acked.store(0, std::memory_order_relaxed);
                entry.failed.store(0, std::memory_order_relaxed);
                *token_id_out = make_token(owner, slot, next_generation);
                *entry_out = &entry;
                *slot_out = slot;
                return true;
            }
        }
        return false;
    }

    EcStagingGroupSlot make_parity_scratch(size_t owner, size_t slot,
                                           uint64_t generation) const noexcept {
        EcStagingGroupSlot value{};
        const auto *base =
            static_cast<const uint8_t *>(parity_banks_[owner].base);
        const size_t offset = slot * kParitySlotBytes;
        value.parity[0] = const_cast<uint8_t *>(base + offset);
        value.parity[1] = const_cast<uint8_t *>(
            base + offset + kParitySegmentBytes);
        value.slot_size = static_cast<uint32_t>(kParitySegmentBytes);
        value.index = static_cast<uint32_t>(slot);
        value.mr_offset = static_cast<uint64_t>(offset);
        value.lkey = parity_banks_[owner].lkey;
        value.generation = generation;
        return value;
    }

    void release_parity_buffers_if_quiescent() noexcept {
        if (!parity_initialized_) {
            return;
        }
        if (parity_free_ == nullptr || in_use() != 0) {
            std::terminate();
        }
        for (size_t owner = 0; owner < owner_count_; ++owner) {
            parity_free_(parity_context_, parity_banks_[owner]);
            parity_banks_[owner] = RegisteredParity{};
        }
        parity_context_ = nullptr;
        parity_free_ = nullptr;
        parity_initialized_ = false;
    }

    size_t owner_count_;
    size_t slots_per_owner_;
    bool banked_;
    std::unique_ptr<Entry[]> entries_;
    std::unique_ptr<OwnerCursor[]> cursors_;
    std::unique_ptr<BankProducer[]> bank_producers_;
    std::unique_ptr<BankCounter[]> bank_counters_;
    std::unique_ptr<RegisteredParity[]> parity_banks_;
    void *parity_context_ = nullptr;
    Free parity_free_ = nullptr;
    bool parity_initialized_ = false;
};

}  // namespace FarLib::hydra
