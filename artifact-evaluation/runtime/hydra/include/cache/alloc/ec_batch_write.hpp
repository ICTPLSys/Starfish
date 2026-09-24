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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>

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
// Capacity is the hard bound on groups in flight; it is a whole power of two so
// that the slot index fits `kEcBatchTokenSlotBits` bits of the token id.
inline constexpr size_t kEcBatchTokenSlotBits = 6;
inline constexpr size_t kEcBatchTokenTableCapacity =
    size_t{1} << kEcBatchTokenSlotBits;
static_assert(kEcBatchTokenTableCapacity >= kEcBatchStagingDefaultDepth,
              "the token table must cover every group the staging pool holds");

// One group in flight: the sealed record (group handle / six segments / the
// four object pointers + sizes / the staging slot) plus the completion
// accounting of its six segments.
struct EcBatchToken {
    EcGroupSendRecord record{};
    uint64_t generation = 1;  // zero is reserved for an invalid token identity
    uint8_t pending = 0;      // segments not yet completed (starts at 6)
    uint8_t acked = 0;        // bit s set <=> segment s completed
    uint8_t failed = 0;       // completed with error, or not posted to a dead node
    bool in_use = false;

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

// Bounded table of group tokens.  No allocation on the hot path: entries and
// the free list are fixed-size arrays, and only the short bookkeeping runs
// under the mutex - the six CQEs of one group may be reaped concurrently.
class EcBatchTokenTable {
public:
    EcBatchTokenTable() {
        for (size_t i = 0; i < kEcBatchTokenTableCapacity; i++) {
            free_stack_[i] = static_cast<uint32_t>(i);
        }
        free_count_ = kEcBatchTokenTableCapacity;
    }

    static constexpr size_t capacity() { return kEcBatchTokenTableCapacity; }

    // Registers one sealed group.  Returns false (and changes nothing) when all
    // capacity entries are in flight; `token_id_out` is then the encoded
    // (generation, slot) identity that goes into every wr_id of the group.
    bool acquire(const EcGroupSendRecord &record, uint64_t *token_id_out,
                 EcBatchToken **entry_out) {
        if (token_id_out == nullptr || entry_out == nullptr) return false;
        // The objects of a group are completed through
        // complete_evict_writeback() with their raw pointer as the key, so
        // those pointers must stay outside the ec_batch wr_id namespace.
        for (size_t i = 0; i < kEcBatchDataSlots; i++) {
            assert(record.objects[i] == nullptr ||
                   !is_ec_batch_wr_id(
                       reinterpret_cast<uint64_t>(record.objects[i])));
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_count_ == 0) return false;
        const size_t slot = free_stack_[--free_count_];
        EcBatchToken &entry = entries_[slot];
        entry.record = record;
        entry.pending = static_cast<uint8_t>(kEcBatchSegmentsPerGroup);
        entry.acked = 0;
        entry.failed = 0;
        entry.in_use = true;
        // Six low bits represent all 64 zero-based slots. The nonzero
        // generation keeps token id 0 invalid. Encoding slot+1 here loses
        // slot 63 because 64 does not fit in six bits.
        *token_id_out = (entry.generation << kEcBatchTokenSlotBits) |
                        static_cast<uint64_t>(slot);
        *entry_out = &entry;
        return true;
    }

    // Live entry of a token id; nullptr when the id is unknown, was already
    // released, or belongs to a previous generation of the same slot.
    EcBatchToken *get(uint64_t token_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return find_locked(token_id);
    }

    // Accounts one completed segment.  Returns the entry exactly once, to the
    // last completion of the group; every other caller gets nullptr (segments
    // still pending, duplicate/foreign segment, or unknown token).
    EcBatchToken *complete_segment(uint64_t token_id, uint8_t segment_idx,
                                  bool success = true) {
        if (segment_idx >= kEcBatchSegmentsPerGroup) return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        EcBatchToken *entry = find_locked(token_id);
        if (entry == nullptr) return nullptr;
        const uint8_t bit = static_cast<uint8_t>(1u << segment_idx);
        if ((entry->acked & bit) != 0) return nullptr;  // duplicate CQE
        if (entry->pending == 0) return nullptr;
        entry->acked |= bit;
        if (!success) entry->failed |= bit;
        entry->pending--;
        return entry->pending == 0 ? entry : nullptr;
    }

    // Returns a finished (or abandoned) token to the table and bumps the slot
    // generation so a late CQE of the old group can never match a new one.
    bool release(uint64_t token_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        EcBatchToken *entry = find_locked(token_id);
        if (entry == nullptr) return false;
        entry->in_use = false;
        entry->pending = 0;
        entry->acked = 0;
        entry->failed = 0;
        entry->generation++;
        free_stack_[free_count_++] = static_cast<uint32_t>(
            static_cast<size_t>(entry - entries_.data()));
        return true;
    }

    // Teardown only (the cache is quiesced: no post is in flight and no
    // completion thread is running).  Hands every live token to
    // `fn(token_id, entry)` exactly once and then releases it, which bumps the
    // slot generation: a late CQE of an abandoned group can never match its
    // token again (find_locked() rejects the old generation), so it can
    // neither complete nor release the same objects / staging slot a second
    // time.  Returns the number of abandoned tokens.
    template <typename Fn>
    size_t abandon_all_in_use(Fn &&fn) {
        uint64_t ids[kEcBatchTokenTableCapacity];
        size_t count = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t slot = 0; slot < kEcBatchTokenTableCapacity; slot++) {
                const EcBatchToken &entry = entries_[slot];
                if (!entry.in_use) continue;
                ids[count++] = (entry.generation << kEcBatchTokenSlotBits) |
                               static_cast<uint64_t>(slot);
            }
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
        std::lock_guard<std::mutex> lock(mutex_);
        return kEcBatchTokenTableCapacity - free_count_;
    }

    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_count_;
    }

private:
    // Zero-based slot index; the generation, not a shifted slot number,
    // distinguishes a live identity from token id zero.
    static size_t token_slot(uint64_t token_id) {
        return static_cast<size_t>(token_id & (kEcBatchTokenTableCapacity - 1));
    }

    EcBatchToken *find_locked(uint64_t token_id) {
        const size_t slot = token_slot(token_id);
        if (slot >= kEcBatchTokenTableCapacity) return nullptr;
        EcBatchToken &entry = entries_[slot];
        if (!entry.in_use) return nullptr;
        if (((entry.generation << kEcBatchTokenSlotBits) |
             static_cast<uint64_t>(slot)) != token_id) {
            return nullptr;
        }
        return &entry;
    }

    std::array<EcBatchToken, kEcBatchTokenTableCapacity> entries_{};
    std::array<uint32_t, kEcBatchTokenTableCapacity> free_stack_{};
    size_t free_count_ = 0;
    mutable std::mutex mutex_;
};

// Thread-safety note: complete_segment() hands the entry back to exactly one
// completion, and only that winner calls release(), so the six CQEs of a group
// can be reaped by six different threads without holding a lock across
// complete_evict_writeback().

}  // namespace FarLib::cache::ec_batch
