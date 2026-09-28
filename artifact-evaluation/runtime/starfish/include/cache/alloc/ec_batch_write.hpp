// ---------------------------------------------------------------------------
// ec_batch write path, stage B2 - group token + wr_id tag contract.
//
// A sealed group (EcGroupSendRecord, see ec_group_builder.hpp) is handed to
// the cache, which posts its six segments (4 data + 2 parity, one per
// endpoint) as single-sided RDMA WRITEs.  Six CQEs arrive later, possibly on
// six different completion threads (full_checker() polls other clients' CQs),
// so the identity of a group in flight has to live in *shared* state, not in a
// thread_local or on the posting thread's stack.
//
// This header owns that state: a bounded, allocation-free token table plus the
// wr_id encoding that carries (token, segment) inside one uint64_t.
//
// wr_id contract
// --------------
// The pre-existing contract is "wr_id == object pointer" (see
// concurrent_cache.hpp post_rdma_request / post_write_requests and
// complete_evict_writeback()).  It must keep working unchanged, so an ec_batch
// wr_id must be impossible to mistake for a pointer and vice versa:
//
//   bit 63      1 <=> ec_batch wr_id.  Pointers are canonical user addresses
//               and never have bit 63 set, so the two namespaces are disjoint
//               without assuming anything about the allocator's alignment.
//   bits 56..62 segment index 0..5 (4 data, then the 2 parity segments)
//   bits 0..55  token id (0 is never used)
//
// encode/decode are static_assert-protected; is_ec_batch_wr_id() is the single
// classifier the completion path uses and a wr_id that fails it keeps the old
// meaning ("it is an object pointer").
// ---------------------------------------------------------------------------
#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "cache/alloc/ec_group_builder.hpp"

namespace FarLib::cache::ec_batch {

// ------------------------------ wr_id tag ---------------------------------
inline constexpr uint64_t kEcBatchWrIdTagBit = 1ull << 63;
inline constexpr uint64_t kEcBatchWrIdSegmentShift = 56;
inline constexpr uint64_t kEcBatchWrIdSegmentMask = 0x7f;  // 7 bits, 0..5 used
// Bit 62 of the high half is reserved for the *other* tagged round that shares
// this wr_id contract: the degraded read of the EC recovery path (see
// recovery/ec_read_recovery.hpp).  The write encoder below never sets it -
// it puts the segment index in bits 56..62 and only 0..5 are used - so the two
// tag value sets are disjoint and is_ec_batch_wr_id() stays the single "is this
// a write tag" classifier.
inline constexpr uint64_t kEcBatchWrIdReservedTagBit = 1ull << 62;
inline constexpr uint64_t kEcBatchWrIdTokenMask =
    (1ull << kEcBatchWrIdSegmentShift) - 1;
inline constexpr uint64_t kEcBatchWrIdTokenMax = kEcBatchWrIdTokenMask;

static_assert(kEcBatchWrIdSegmentShift + 7 <= 63,
              "the ec_batch wr_id tag must stay inside the high half");
static_assert((kEcBatchWrIdReservedTagBit & kEcBatchWrIdTokenMask) == 0,
              "the reserved tag bit must not overlap the token field");
static_assert((kEcBatchWrIdReservedTagBit & kEcBatchWrIdTagBit) == 0,
              "the reserved tag bit is not the tag bit itself");
static_assert(kEcBatchSegmentsPerGroup <= kEcBatchWrIdSegmentMask + 1,
              "the segment field must hold all six segments");
static_assert((kEcBatchWrIdTagBit & kEcBatchWrIdTokenMask) == 0,
              "the tag bit must not overlap the token field");
static_assert(sizeof(uint64_t) == sizeof(void *),
              "a wr_id carries either a pointer or an ec_batch tag");

inline uint64_t encode_ec_batch_wr_id(uint64_t token_id, uint8_t segment_idx) {
    assert(token_id != 0 && token_id <= kEcBatchWrIdTokenMax);
    assert(segment_idx < kEcBatchSegmentsPerGroup);
    return kEcBatchWrIdTagBit |
           (static_cast<uint64_t>(segment_idx) << kEcBatchWrIdSegmentShift) |
           (token_id & kEcBatchWrIdTokenMask);
}

// True iff this wr_id belongs to the ec_batch write round.  A plain object
// pointer never satisfies it (bit 63 clear, canonical user address), and
// neither does a wr_id of the degraded read round (it sets the reserved tag
// bit, which the write encoder never sets).
inline bool is_ec_batch_wr_id(uint64_t wr_id) {
    return (wr_id & kEcBatchWrIdTagBit) != 0 &&
           (wr_id & kEcBatchWrIdReservedTagBit) == 0 &&
           (wr_id & kEcBatchWrIdTokenMask) != 0;
}

inline uint64_t ec_batch_wr_id_token(uint64_t wr_id) {
    return wr_id & kEcBatchWrIdTokenMask;
}

inline uint8_t ec_batch_wr_id_segment(uint64_t wr_id) {
    return static_cast<uint8_t>((wr_id >> kEcBatchWrIdSegmentShift) &
                                kEcBatchWrIdSegmentMask);
}

// ----------------------------- token table --------------------------------
// Token ownership is partitioned by posting owner (normally a client/QP
// index).  Completion may arrive on any worker, so the wr_id still names a
// global slot; acquisition, however, only scans the owner's local bitmap.
// The slot field remains inside the existing 56-bit token namespace and the
// generation occupies the remaining high bits.  There is deliberately no
// shared free-stack mutex on the hot path.
inline constexpr size_t kEcBatchTokenOwnerCount = 64;
inline constexpr size_t kEcBatchTokenSlotsPerOwner = 128;
inline constexpr size_t kEcBatchTokenOwnerWordCount =
    kEcBatchTokenSlotsPerOwner / 64;
inline constexpr size_t kEcBatchTokenSlotBits = 13;  // 64 owners * 128 slots
inline constexpr size_t kEcBatchTokenTableCapacity =
    size_t{1} << kEcBatchTokenSlotBits;
inline constexpr size_t kEcBatchTokenGenerationBits =
    kEcBatchWrIdSegmentShift - 1 - kEcBatchTokenSlotBits; // bit55: fixed direct bank
inline constexpr uint64_t kEcBatchTokenGenerationMax =
    (uint64_t{1} << kEcBatchTokenGenerationBits) - 1;
static_assert(kEcBatchTokenOwnerCount * kEcBatchTokenSlotsPerOwner ==
                  kEcBatchTokenTableCapacity,
              "owner partition must cover the token table");
static_assert(kEcBatchTokenOwnerWordCount * 64 ==
                  kEcBatchTokenSlotsPerOwner,
              "owner slot count must be bitmap aligned");
static_assert(kEcBatchTokenTableCapacity >= kEcBatchStagingDefaultDepth,
              "the token table must cover every group the staging pool holds");

// One group in flight: the sealed record (group handle / six segments / the
// four object pointers + sizes / the staging slot) plus the completion
// accounting of its six segments.
struct alignas(64) EcBatchToken {
    EcGroupSendRecord record{};
    uint64_t generation = 1;  // zero is reserved for an invalid token identity
    uint8_t pending = 0;      // final snapshot: segments not yet completed
    uint8_t acked = 0;        // final snapshot: bit s set <=> segment completed
    uint8_t failed = 0;       // final snapshot: failed/unposted segment mask
    bool in_use = false;

private:
    friend class EcBatchTokenTable;
    // Generation, pending count, acked mask, and failed mask are updated by
    // completion threads through one CAS word.  The public byte fields above
    // are written only at acquire/release and by the winner of the final-CQE
    // CAS, preserving the old observation API without racing partial CQEs.
    std::atomic<uint64_t> completion_state{0};
    uint16_t owner = 0;

public:
    size_t segment_count() const {
        return static_cast<size_t>(record.group.segments.size());
    }

    unsigned durable_segment_count() const {
        return static_cast<unsigned>(__builtin_popcount(
            static_cast<unsigned>(acked & ~failed)));
    }

    bool recoverable() const {
        return pending == 0 && durable_segment_count() >= kEcBatchDataSlots;
    }
};

// Bounded table of group tokens.  No allocation or global mutex is used on the
// hot path: each owner claims a bit in its private free bitmap, while the six
// CQEs of one group update one slot's atomic completion state concurrently.
class EcBatchTokenTable {
public:
    EcBatchTokenTable() {
        for (size_t owner = 0; owner < kEcBatchTokenOwnerCount; ++owner) {
            for (size_t word = 0; word < kEcBatchTokenOwnerWordCount;
                 ++word) {
                free_bits_[owner].words[word].store(
                    ~uint64_t{0}, std::memory_order_relaxed);
            }
        }
        for (size_t i = 0; i < kEcBatchTokenTableCapacity; ++i) {
            generation_counters_[i].store(1, std::memory_order_relaxed);
        }
    }

    static constexpr size_t capacity() { return kEcBatchTokenTableCapacity; }
    static constexpr size_t owner_count() { return kEcBatchTokenOwnerCount; }
    static constexpr bool valid_owner(size_t owner) {
        return owner < kEcBatchTokenOwnerCount;
    }
    static constexpr size_t owner_capacity() {
        return kEcBatchTokenSlotsPerOwner;
    }

    // Registers one sealed group.  Returns false (and changes nothing) when all
    // entries owned by `owner` are in flight; `token_id_out` is the encoded
    // (generation, global-slot) identity that goes into every wr_id of the
    // group.  The default owner preserves the old three-argument API.
    bool acquire(const EcGroupSendRecord &record, uint64_t *token_id_out,
                 EcBatchToken **entry_out, size_t owner = 0) {
        if (token_id_out == nullptr || entry_out == nullptr) return false;
        if (!valid_owner(owner)) return false;
        // The objects of a group are completed through
        // complete_evict_writeback() with their raw pointer as the key, so
        // those pointers must stay outside the ec_batch wr_id namespace.
        for (size_t i = 0; i < kEcBatchDataSlots; i++) {
            assert(record.objects[i] == nullptr ||
                       !is_ec_batch_wr_id(
                           reinterpret_cast<uint64_t>(record.objects[i])));
        }

        size_t local_slot = 0;
        if (!claim_slot(owner, &local_slot)) return false;
        const size_t slot = owner * kEcBatchTokenSlotsPerOwner + local_slot;
        EcBatchToken &entry = entries_[slot];
        const uint64_t generation =
            generation_counters_[slot].fetch_add(1, std::memory_order_relaxed);
        if (generation == 0 || generation > kEcBatchTokenGenerationMax) {
            // Do not recycle an exhausted generation: permanently reserving
            // this slot is preferable to ever allowing a token-id ABA.
            return false;
        }
        entry.record = record;
        entry.owner = static_cast<uint16_t>(owner);
        entry.generation = generation;
        entry.pending = static_cast<uint8_t>(kEcBatchSegmentsPerGroup);
        entry.acked = 0;
        entry.failed = 0;
        entry.in_use = true;
        // Publish the immutable record before the state becomes visible to a
        // completion thread.  A stale CQE for an older generation cannot
        // match this state word even if the slot was immediately reused.
        entry.completion_state.store(
            pack_state(generation,
                       static_cast<uint8_t>(kEcBatchSegmentsPerGroup), 0, 0),
            std::memory_order_release);
        *token_id_out = (generation << kEcBatchTokenSlotBits) |
                        static_cast<uint64_t>(slot);
        *entry_out = &entry;
        return true;
    }

    // Live entry of a token id; nullptr when the id is unknown, was already
    // released, or belongs to a previous generation of the same slot.
    EcBatchToken *get(uint64_t token_id) {
        size_t slot = 0;
        uint64_t generation = 0;
        if (!decode_token(token_id, &slot, &generation)) return nullptr;
        return find_active(slot, generation);
    }

    // Accounts one completed segment.  Returns the entry exactly once, to the
    // last completion of the group; every other caller gets nullptr (segments
    // still pending, duplicate/foreign segment, or unknown token).
    EcBatchToken *complete_segment(uint64_t token_id, uint8_t segment_idx,
                                  bool success = true) {
        if (segment_idx >= kEcBatchSegmentsPerGroup) return nullptr;
        size_t slot = 0;
        uint64_t generation = 0;
        if (!decode_token(token_id, &slot, &generation)) return nullptr;
        EcBatchToken &entry = entries_[slot];
        uint64_t state = entry.completion_state.load(std::memory_order_acquire);
        const uint8_t bit = static_cast<uint8_t>(1u << segment_idx);
        for (;;) {
            if (state == 0 || state_generation(state) != generation) {
                return nullptr;
            }
            const uint8_t pending = state_pending(state);
            const uint8_t acked = state_acked(state);
            const uint8_t failed = state_failed(state);
            if (pending == 0 || (acked & bit) != 0) return nullptr;
            const uint8_t next_acked = static_cast<uint8_t>(acked | bit);
            const uint8_t next_failed =
                static_cast<uint8_t>(failed | (success ? 0 : bit));
            const uint8_t next_pending = static_cast<uint8_t>(pending - 1);
            const uint64_t next = pack_state(generation, next_pending,
                                              next_acked, next_failed);
            if (!entry.completion_state.compare_exchange_weak(
                    state, next, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            if (next_pending == 0) {
                // Exactly one CAS can transition pending 1 -> 0.  Only that
                // winner publishes the legacy plain completion snapshot and
                // returns the token to the caller for object release.
                entry.pending = 0;
                entry.acked = next_acked;
                entry.failed = next_failed;
                return &entry;
            }
            return nullptr;
        }
    }

    // Returns a finished (or abandoned) token to the table.  The next acquire
    // advances the slot generation so a late CQE of the old group can never
    // match a new one.
    bool release(uint64_t token_id) {
        size_t slot = 0;
        uint64_t generation = 0;
        if (!decode_token(token_id, &slot, &generation)) return false;
        EcBatchToken &entry = entries_[slot];
        uint64_t state = entry.completion_state.load(std::memory_order_acquire);
        for (;;) {
            if (state == 0 || state_generation(state) != generation) {
                return false;
            }
            if (entry.completion_state.compare_exchange_weak(
                    state, 0, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                break;
            }
        }
        entry.in_use = false;
        entry.pending = 0;
        entry.acked = 0;
        entry.failed = 0;
        const size_t owner = entry.owner;
        const size_t local_slot = slot % kEcBatchTokenSlotsPerOwner;
        release_slot(owner, local_slot);
        return true;
    }

    // Teardown only (the cache is quiesced: no post is in flight and no
    // completion thread is running).  Hands every live token to
    // `fn(token_id, entry)` exactly once and then releases it.  The next
    // acquire advances that slot's generation, so a late CQE of an abandoned
    // group can never match its token again (find_active() rejects the old
    // generation), and can neither complete nor release the same objects /
    // staging slot a second time.  Returns the number of abandoned tokens.
    template <typename Fn>
    size_t abandon_all_in_use(Fn &&fn) {
        std::array<uint64_t, kEcBatchTokenTableCapacity> ids{};
        size_t count = 0;
        for (size_t slot = 0; slot < kEcBatchTokenTableCapacity; ++slot) {
            const uint64_t state =
                entries_[slot].completion_state.load(std::memory_order_acquire);
            if (state == 0) continue;
            ids[count++] = (state_generation(state) << kEcBatchTokenSlotBits) |
                           static_cast<uint64_t>(slot);
        }
        for (size_t i = 0; i < count; i++) {
            const EcBatchToken *entry = get(ids[i]);
            if (entry == nullptr) continue;  // released meanwhile
            fn(ids[i], *entry);
            release(ids[i]);
        }
        return count;
    }

    size_t in_use() const {
        return kEcBatchTokenTableCapacity - available();
    }

    size_t available() const {
        size_t total = 0;
        for (size_t owner = 0; owner < kEcBatchTokenOwnerCount; ++owner) {
            total += available(owner);
        }
        return total;
    }

    size_t in_use(size_t owner) const {
        if (!valid_owner(owner)) return 0;
        return kEcBatchTokenSlotsPerOwner - available(owner);
    }

    size_t available(size_t owner) const {
        if (!valid_owner(owner)) return 0;
        size_t total = 0;
        for (size_t word = 0; word < kEcBatchTokenOwnerWordCount; ++word) {
            total += static_cast<size_t>(__builtin_popcountll(
                free_bits_[owner].words[word].load(
                    std::memory_order_acquire)));
        }
        return total;
    }

private:
    static constexpr size_t kStatePendingBits = 4;
    static constexpr size_t kStateAckedShift = kStatePendingBits;
    static constexpr size_t kStateFailedShift =
        kStateAckedShift + kEcBatchSegmentsPerGroup;
    static constexpr size_t kStateGenerationShift =
        kStateFailedShift + kEcBatchSegmentsPerGroup;
    static constexpr uint64_t kStatePendingMask =
        (uint64_t{1} << kStatePendingBits) - 1;
    static constexpr uint64_t kStateSegmentMask =
        (uint64_t{1} << kEcBatchSegmentsPerGroup) - 1;

    static uint64_t pack_state(uint64_t generation, uint8_t pending,
                               uint8_t acked, uint8_t failed) {
        return (generation << kStateGenerationShift) |
               (static_cast<uint64_t>(failed & kStateSegmentMask)
                << kStateFailedShift) |
               (static_cast<uint64_t>(acked & kStateSegmentMask)
                << kStateAckedShift) |
               static_cast<uint64_t>(pending & kStatePendingMask);
    }

    static uint64_t state_generation(uint64_t state) {
        return state >> kStateGenerationShift;
    }
    static uint8_t state_pending(uint64_t state) {
        return static_cast<uint8_t>(state & kStatePendingMask);
    }
    static uint8_t state_acked(uint64_t state) {
        return static_cast<uint8_t>((state >> kStateAckedShift) &
                                    kStateSegmentMask);
    }
    static uint8_t state_failed(uint64_t state) {
        return static_cast<uint8_t>((state >> kStateFailedShift) &
                                    kStateSegmentMask);
    }

    static bool decode_token(uint64_t token_id, size_t *slot_out,
                             uint64_t *generation_out) {
        if (slot_out == nullptr || generation_out == nullptr || token_id == 0 ||
            (token_id & ~kEcBatchWrIdTokenMask) != 0) {
            return false;
        }
        const size_t slot = static_cast<size_t>(
            token_id & (kEcBatchTokenTableCapacity - 1));
        const uint64_t generation = token_id >> kEcBatchTokenSlotBits;
        if (slot >= kEcBatchTokenTableCapacity || generation == 0 ||
            generation > kEcBatchTokenGenerationMax) {
            return false;
        }
        *slot_out = slot;
        *generation_out = generation;
        return true;
    }

    EcBatchToken *find_active(size_t slot, uint64_t generation) {
        if (slot >= kEcBatchTokenTableCapacity) return nullptr;
        EcBatchToken &entry = entries_[slot];
        const uint64_t state =
            entry.completion_state.load(std::memory_order_acquire);
        if (state == 0 || state_generation(state) != generation) return nullptr;
        return &entry;
    }

    bool claim_slot(size_t owner, size_t *local_slot_out) {
        if (local_slot_out == nullptr || !valid_owner(owner)) {
            return false;
        }
        for (size_t word = 0; word < kEcBatchTokenOwnerWordCount; ++word) {
            uint64_t bits =
                free_bits_[owner].words[word].load(std::memory_order_acquire);
            while (bits != 0) {
                const uint64_t bit = bits & (~bits + 1);
                const uint64_t desired = bits & ~bit;
                if (free_bits_[owner].words[word].compare_exchange_weak(
                        bits, desired, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    *local_slot_out = word * 64 +
                                      static_cast<size_t>(__builtin_ctzll(bit));
                    return true;
                }
            }
        }
        return false;
    }

    void release_slot(size_t owner, size_t local_slot) {
        assert(valid_owner(owner));
        assert(local_slot < kEcBatchTokenSlotsPerOwner);
        const size_t word = local_slot / 64;
        const uint64_t bit = uint64_t{1} << (local_slot % 64);
        free_bits_[owner].words[word].fetch_or(bit,
                                               std::memory_order_release);
    }

    std::array<EcBatchToken, kEcBatchTokenTableCapacity> entries_{};
    std::array<std::atomic<uint64_t>, kEcBatchTokenTableCapacity>
        generation_counters_{};
    struct alignas(64) OwnerFreeBitmap {
        std::array<std::atomic<uint64_t>, kEcBatchTokenOwnerWordCount> words{};
    };
    std::array<OwnerFreeBitmap, kEcBatchTokenOwnerCount> free_bits_{};
};

// Thread-safety note: complete_segment() hands the entry back to exactly one
// completion through the slot CAS, and only that winner calls release(), so the
// six CQEs of a group can be reaped by six different threads without holding a
// lock across complete_evict_writeback().

}  // namespace FarLib::cache::ec_batch
