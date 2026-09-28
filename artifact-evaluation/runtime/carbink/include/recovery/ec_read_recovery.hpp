// ---------------------------------------------------------------------------
// ec_batch read path - the degraded read (endpoint-loss) contract.
//
// A small object that was evicted through the ec_batch path lives in the data
// shard of a slot group: 4 data slots plus 2 RS(4,2) parity slots, one 4 KB
// segment per shard, six distinct endpoints.  Once one of those servers is
// gone, an RDMA READ posted to its endpoint never completes normally (the QP
// goes to error and the CQE carries a failure status), so the fetch of every
// object whose segment sits on that endpoint would spin in FETCHING forever.
//
// The read of such an object can still be served: five of the six segments are
// readable, and the codec reconstructs the missing one from any four survivors
// (include/cache/alloc/small_object_stripe_codec.hpp).  This header owns the
// bookkeeping of that round, mirroring the write-side split of
// ec_batch_write.hpp:
//
//   * the wr_id tag contract of the degraded READ round (its own namespace, so
//     it can never be mistaken for a write tag or for an object pointer),
//   * the alive-mask / plan derivation from the per-segment endpoints,
//   * a growable token table (one entry per degraded read in flight, with a
//     generation so a late CQE cannot match a reused slot).
//
// The scratch buffers of a degraded read are borrowed from an EcStagingPool
// bound to the tail of the registered client MR (see the pool's own header):
// six slot-size segments in MR memory, so a single-sided read can land in them
// and the codec can rebuild in place.
//
// Nothing in this file is reachable while ft_method != ec_batch: the cache only
// consults it behind Configure::is_ec_batch_mode().
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "cache/alloc/ec_batch_staging.hpp"
#include "cache/alloc/ec_batch_write.hpp"
#include "recovery/ec_recovery_profile.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"

namespace FarLib::cache {

// Group view of one object (defined in recovery/ec_read_recovery.ipp):
// the six segments of the slot group its remote address belongs to plus the
// plan derived from the endpoints that survived.  Forward declared here because
// it is produced and consumed inside that file only.
struct EcRecoveryGroupView;

}  // namespace FarLib::cache

namespace FarLib::cache::ec_read_recovery {

// ------------------------------ wr_id tag ---------------------------------
// Same high-half convention as the ec_batch write round, in its own namespace:
//
//   bit 63      1 <=> a tagged wr_id.  Object pointers are canonical user
//               addresses and never have bit 63 set, so the two namespaces are
//               disjoint without assuming anything about the allocator.
//   bit 62      1 <=> the read-recovery namespace.  The write encoder puts the
//               segment index in bits 56..62 and only uses 0..5, so it never
//               sets this bit: the two tag value sets cannot collide.
//   bits 56..61 segment index 0..5 (data shards first, then the parity shards)
//   bits 0..55  token id (0 is never used)
inline constexpr uint64_t kEcReadWrIdTagBit = 1ull << 63;
inline constexpr uint64_t kEcReadWrIdNamespaceBit = 1ull << 62;
inline constexpr uint64_t kEcReadWrIdSegmentShift = 56;
inline constexpr uint64_t kEcReadWrIdSegmentMask = 0x3f;  // 6 bits, 0..5 used
inline constexpr uint64_t kEcReadWrIdTokenMask =
    (1ull << kEcReadWrIdSegmentShift) - 1;

// One degraded read covers the whole slot group: 4 data + 2 parity segments.
inline constexpr size_t kEcReadSegmentCount =
    ec_batch::kEcBatchSegmentsPerGroup;

static_assert(kEcReadSegmentCount == kStripeCodecShardCount,
              "one segment per shard, data shards first");
static_assert(kEcReadWrIdNamespaceBit == ec_batch::kEcBatchWrIdReservedTagBit,
              "the read namespace bit is the one the write tag reserves");
static_assert((kEcReadWrIdTagBit & kEcReadWrIdNamespaceBit) == 0,
              "the tag bit and the namespace bit are distinct");
static_assert((kEcReadWrIdTagBit & kEcReadWrIdTokenMask) == 0,
              "the tag bit must not overlap the token field");
static_assert((kEcReadWrIdNamespaceBit & kEcReadWrIdTokenMask) == 0,
              "the namespace bit must not overlap the token field");
static_assert(kEcReadWrIdSegmentShift + 6 <= 62,
              "the segment field must stay below the namespace bit");
static_assert(kEcReadSegmentCount <= kEcReadWrIdSegmentMask + 1,
              "the segment field must hold all six segments");
static_assert(sizeof(uint64_t) == sizeof(void *),
              "a wr_id carries either a pointer or a tagged round");

inline uint64_t encode_ec_read_wr_id(uint64_t token_id, uint8_t segment_idx) {
    assert(token_id != 0 && token_id <= kEcReadWrIdTokenMask);
    assert(segment_idx < kEcReadSegmentCount);
    return kEcReadWrIdTagBit | kEcReadWrIdNamespaceBit |
           (static_cast<uint64_t>(segment_idx) << kEcReadWrIdSegmentShift) |
           (token_id & kEcReadWrIdTokenMask);
}

// True iff this wr_id belongs to the degraded read round.  A plain object
// pointer never satisfies it, and neither does an ec_batch write tag (those
// leave the namespace bit clear): this classifier is the single test the
// completion path uses before it hands a wr_id to handle_rdma_read_complete().
inline bool is_ec_read_wr_id(uint64_t wr_id) {
    return (wr_id & kEcReadWrIdTagBit) != 0 &&
           (wr_id & kEcReadWrIdNamespaceBit) != 0 &&
           (wr_id & kEcReadWrIdTokenMask) != 0;
}

inline uint64_t ec_read_wr_id_token(uint64_t wr_id) {
    return wr_id & kEcReadWrIdTokenMask;
}

inline uint8_t ec_read_wr_id_segment(uint64_t wr_id) {
    return static_cast<uint8_t>((wr_id >> kEcReadWrIdSegmentShift) &
                                kEcReadWrIdSegmentMask);
}

// --------------------------- alive mask / plan -----------------------------
// Which of the six segments of a group are still readable, given the endpoint
// of every segment and a "this endpoint is dead" predicate.  Only the segments
// whose endpoint answered are set: a dead endpoint is never retried (its RC QP
// is in error and the tree has no reset/re-handshake path).
template <typename EndpointDeadFn>
inline uint8_t alive_mask_for_dead_endpoints(
    const uint32_t endpoint_idx[kEcReadSegmentCount],
    EndpointDeadFn &&endpoint_dead) {
    uint8_t alive_mask = 0;
    for (size_t s = 0; s < kEcReadSegmentCount; s++) {
        if (!endpoint_dead(static_cast<size_t>(endpoint_idx[s]))) {
            alive_mask |= static_cast<uint8_t>(1u << s);
        }
    }
    return alive_mask;
}

// What a degraded read of one object can do, derived from the alive mask and
// from the shard index of the object's own data segment (0..3, the index of
// `entry.remote_addr()` inside the group).
struct EcReadPlan {
    // Physical endpoint availability for the whole group.  This is retained
    // separately from read_mask: callers use it to describe the failure set
    // and diagnostics must not mistake an intentionally skipped survivor for
    // a dead endpoint.
    uint8_t alive_mask = 0;
    // The exact four segments fetched into scratch.  When five or six
    // segments are physically alive, only the lowest-indexed four are read;
    // the codec can reconstruct the requested shard from any four survivors.
    // Once a context is acquired this is the mask stored in its alive_mask:
    // that field means fetched/valid scratch buffers, not physical liveness.
    uint8_t read_mask = 0;
    uint8_t missing_mask = 0;
    uint8_t alive_count = 0;
    uint8_t missing_count = 0;
    uint8_t own_shard_idx = 0;
    bool own_shard_alive = false;
    bool own_shard_missing = false;
    // True when the codec can serve this object: at most two segments are gone
    // and at least four survive (the MDS code rebuilds any 1..2 missing shards
    // from four survivors).
    bool usable = false;

    // True when the bytes of the object have to be reconstructed rather than
    // copied out of its own (still readable) data segment.
    bool needs_rebuild() const { return usable && own_shard_missing; }
};

// Select the lowest four physically alive shards.  Keep this helper separate
// from alive_mask_for_dead_endpoints()/EcReadPlan::alive_mask: a fifth live
// shard is deliberately not fetched, and its scratch buffer must never be
// treated as valid by the completion/rebuild path.
inline uint8_t ec_read_selected_mask(uint8_t alive_mask) {
    constexpr uint8_t kAllSegments =
        static_cast<uint8_t>((1u << kEcReadSegmentCount) - 1u);
    const uint8_t physical_alive = static_cast<uint8_t>(alive_mask & kAllSegments);
    uint8_t selected = 0;
    uint8_t selected_count = 0;
    for (size_t segment = 0; segment < kEcReadSegmentCount &&
                              selected_count < kStripeCodecDataShards;
         ++segment) {
        const uint8_t bit = static_cast<uint8_t>(1u << segment);
        if ((physical_alive & bit) == 0) continue;
        selected |= bit;
        ++selected_count;
    }
    return selected;
}

inline EcReadPlan make_ec_read_plan(uint8_t alive_mask, uint8_t own_shard_idx) {
    constexpr uint8_t kAllSegments =
        static_cast<uint8_t>((1u << kEcReadSegmentCount) - 1u);
    EcReadPlan plan;
    plan.own_shard_idx = own_shard_idx < kStripeCodecDataShards
                             ? own_shard_idx
                             : static_cast<uint8_t>(0);
    plan.alive_mask = static_cast<uint8_t>(alive_mask & kAllSegments);
    plan.read_mask = ec_read_selected_mask(plan.alive_mask);
    plan.missing_mask = static_cast<uint8_t>((~plan.alive_mask) & kAllSegments);
    plan.alive_count = static_cast<uint8_t>(
        __builtin_popcount(static_cast<unsigned>(plan.alive_mask)));
    plan.missing_count =
        static_cast<uint8_t>(kEcReadSegmentCount - plan.alive_count);
    plan.own_shard_alive = (plan.alive_mask & (1u << plan.own_shard_idx)) != 0;
    plan.own_shard_missing = !plan.own_shard_alive;
    plan.usable = plan.missing_count <= kStripeCodecParityShards &&
                  plan.alive_count >= kStripeCodecDataShards;
    return plan;
}

// Number of segments a posting loop really sends for this alive mask: exactly
// the set bits below kEcReadSegmentCount.  A dead endpoint is skipped there,
// never retried, so kEcReadSegmentCount is the wrong pending count for a token
// as soon as one segment is gone (the token would wait for a completion that
// never comes).
inline uint8_t ec_read_posted_segment_count(uint8_t alive_mask) {
    constexpr uint8_t kAllSegments =
        static_cast<uint8_t>((1u << kEcReadSegmentCount) - 1u);
    return static_cast<uint8_t>(
        __builtin_popcount(static_cast<unsigned>(alive_mask & kAllSegments)));
}

// The set of segments a posting loop really sends; the winner condition of
// EcReadTokenTable::complete_segment().  Same set of bits as
// ec_read_posted_segment_count().
inline uint8_t ec_read_posted_segment_mask(uint8_t alive_mask) {
    constexpr uint8_t kAllSegments =
        static_cast<uint8_t>((1u << kEcReadSegmentCount) - 1u);
    return static_cast<uint8_t>(alive_mask & kAllSegments);
}

// Outcome of trying to serve a read from the surviving segments of its group.
enum class EcReadPostResult {
    kPosted,       // the surviving segments were posted to the scratch
    kInFlight,     // an identical degraded read is already registered
    kUnavailable,  // not stripe-owned / not small / not recoverable / no scratch
};

// Scratch buffer of one segment inside a borrowed staging slot; the order is
// the segment order of the group (4 data shards, then the 2 parity shards).
inline void *ec_read_scratch_segment(const ec_batch::EcStagingGroupSlot &slot,
                                     uint8_t segment) {
    if (segment < ec_batch::kEcBatchDataSlots) return slot.data[segment];
    return slot.parity[segment - ec_batch::kEcBatchDataSlots];
}

// ------------------------------ token table -------------------------------
// Initial token-table population.  This name is retained for source
// compatibility with callers that use it to size the initial scratch pool;
// it is not a limit on degraded reads in flight.  The token table grows when
// all currently allocated entries are in use.
inline constexpr size_t kEcReadRecoveryInFlight = 8;
inline constexpr size_t kEcReadTokenTableInitialCapacity =
    kEcReadRecoveryInFlight;

// A token id uses the 56 low bits of a tagged wr_id: 32 slot-index bits and a
// 24-bit generation.  The namespace is intentionally much larger than the
// initial population; an allocation failure simply makes acquire() return
// false, while already registered tokens remain live.
inline constexpr size_t kEcReadTokenSlotBits = 32;
inline constexpr size_t kEcReadTokenGenerationBits = 24;
inline constexpr uint64_t kEcReadTokenSlotMask =
    (uint64_t{1} << kEcReadTokenSlotBits) - 1;
inline constexpr uint64_t kEcReadTokenGenerationMask =
    (uint64_t{1} << kEcReadTokenGenerationBits) - 1;
inline constexpr uint64_t kEcReadTokenMaxGeneration =
    kEcReadTokenGenerationMask;
inline constexpr uint64_t kEcReadTokenIdMask =
    (uint64_t{1} << (kEcReadTokenSlotBits + kEcReadTokenGenerationBits)) - 1;
static_assert(kEcReadTokenSlotBits + kEcReadTokenGenerationBits ==
                  kEcReadWrIdSegmentShift,
              "token slot and generation fields must fill wr_id bits 0..55");
static_assert(kEcReadTokenMaxGeneration <=
                  std::numeric_limits<uint32_t>::max(),
              "the token generation must fit the entry field");
static_assert(kEcReadTokenIdMask == kEcReadWrIdTokenMask,
              "token id must occupy the complete low 56-bit wr_id field");

// One degraded read in flight: where to rebuild, what to rebuild and the six
// scratch segments the surviving shards are read into.
struct EcReadToken {
    uint64_t target_local_addr = 0;  // entry.local_addr(): object slot to rebuild
    uint64_t remote_addr = 0;        // entry.remote_addr(): own data segment
    uint32_t byte_count = 0;         // obj.size bytes to rebuild
    uint8_t alive_mask = 0;          // bit s set <=> segment s is readable
    uint8_t own_shard_idx = 0;       // 0..3, segment of target_local_addr
    // `alive_mask` is the set the posting loop intends to send.  The three
    // masks below are updated under EcReadTokenTable::mutex_: a completion may
    // race the posting fibre, so neither an integer pending count nor the
    // posting fibre's local counter is sufficient to decide when the MR can
    // be returned.
    uint8_t posted_mask = 0;          // successful post_read() calls
    uint8_t completed_mask = 0;       // success or failure CQEs observed
    uint8_t failed_mask = 0;          // completed requests with an error
    uint8_t pending = 0;              // posted bits without a final CQE
    uint8_t acked = 0;                // successful completion bits
    uint32_t slot_index = 0;          // stable index in the token-id namespace
    uint32_t generation = 1;         // reuse generation of the slot
    bool in_use = false;
    bool retired = false;             // generation namespace was exhausted
    bool posting_done = false;        // no more post_read() calls will occur
    // Set by complete_segment() when the token is handed to the winner: while
    // that fibre rebuilds, the token must not be reaped by anyone else.
    bool winner_handed_out = false;
    // Optional profiler timestamp.  It is initialized before the first
    // survivor post and is copied into EcReadTokenEvent before free_locked()
    // can clear/recycle this entry.  No caller may inspect it after a token
    // table transition can release the token.
    uint64_t profile_acquire_ns = 0;
    ec_batch::EcStagingGroupSlot scratch{};  // 6 registered MR buffers

    size_t segment_count() const { return kEcReadSegmentCount; }
};

// Result of one state-machine transition.  A scratch slot is returned only in
// kRelease, after posting is finished and every successfully posted request
// has a success or failure completion.  kWinner hands the live token to the
// sole rebuilding caller; that caller releases the token after rebuilding.
enum class EcReadTokenEventKind : uint8_t {
    kIgnored,
    kPending,
    kWinnerReady,  // posting ended; caller must claim via a completion probe
    kWinner,
    kRelease,
};

struct EcReadTokenEvent {
    EcReadTokenEventKind kind = EcReadTokenEventKind::kIgnored;
    EcReadToken *token = nullptr;
    ec_batch::EcStagingGroupSlot scratch{};
    uint8_t segment = 0;
    // Snapshot before a kRelease transition can recycle the token.  These are
    // diagnostics only; they never participate in the state machine.
    uint64_t profile_acquire_ns = 0;
    uint32_t profile_byte_count = 0;
};

// Growable table of degraded-read tokens.  Entries are individually owned so
// vector growth never moves an EcReadToken whose address is held by a winner
// or a CQ handler.  Growth is allocation-only when the warm free list is
// empty; ordinary acquire/release operations reuse slots without allocating.
// All short bookkeeping runs under the mutex, so CQEs of one read (and of
// several reads) can be reaped concurrently.
//
// The same target may be waited on by several fibres, and its degraded read
// must be posted exactly once: acquire() refuses a target that is already in
// flight, which is also the only guard the wait path needs.
class EcReadTokenTable {
public:
    EcReadTokenTable() noexcept {
        // Best-effort warm-up.  A constructor cannot report bad_alloc, so a
        // failed warm-up leaves an empty but usable table; acquire() retries
        // growth and reports false if allocation still cannot succeed.
        for (size_t i = 0; i < kEcReadTokenTableInitialCapacity; i++) {
            if (!grow_slot_locked()) break;
        }
    }

    // Number of allocated entries (including retired entries); this is a
    // runtime value because the table grows on demand.
    size_t capacity() const {
        EcRecoveryTokenMutexGuard lock(mutex_);
        return entries_.size();
    }

    // Registers one degraded read.  Returns false (and changes nothing) when
    // the target is already in flight, the validation predicate rejects it,
    // or table growth cannot allocate; `token_id_out` is then the encoded
    // (generation, slot) identity that goes into every wr_id of the read.
    bool acquire(uint64_t target_local_addr, uint64_t remote_addr,
                 uint32_t byte_count, uint8_t alive_mask,
                 uint8_t own_shard_idx,
                 const ec_batch::EcStagingGroupSlot &scratch,
                 uint64_t *token_id_out, EcReadToken **entry_out) {
        return acquire_if(target_local_addr, remote_addr, byte_count,
                          alive_mask, own_shard_idx, scratch,
                          token_id_out, entry_out, [] { return true; });
    }

    // Validate the fetch while holding the same lock that releases the prior
    // token. A waiter may have observed FETCHING before the prior winner
    // published LOCAL and released its token; checking only before acquire()
    // would let that waiter start a second rebuild over application memory.
    // The predicate must not yield or re-enter this table.
    template <typename StillFetching>
    bool acquire_if(uint64_t target_local_addr, uint64_t remote_addr,
                    uint32_t byte_count, uint8_t alive_mask,
                    uint8_t own_shard_idx,
                    const ec_batch::EcStagingGroupSlot &scratch,
                    uint64_t *token_id_out, EcReadToken **entry_out,
                    StillFetching &&still_fetching) {
        if (token_id_out == nullptr || entry_out == nullptr) return false;
        if (target_local_addr == 0) return false;
        // The object pointers are used as keys all over the cache, so they must
        // stay outside the tagged wr_id namespace.
        assert(!is_ec_read_wr_id(target_local_addr));
        EcRecoveryTokenMutexGuard lock(mutex_);
        for (const std::unique_ptr<EcReadToken> &owned_entry : entries_) {
            if (owned_entry != nullptr && owned_entry->in_use &&
                owned_entry->target_local_addr == target_local_addr) {
                return false;  // already posted for this object
            }
        }
        if (!still_fetching()) return false;
        // Grow only after all acquire_if validation has passed, so a rejected
        // duplicate/non-FETCHING request cannot consume allocation budget.
        if (free_count_ == 0 && !grow_slot_locked()) return false;
        const uint32_t slot = free_stack_[--free_count_];
        if (static_cast<size_t>(slot) >= entries_.size() ||
            entries_[slot] == nullptr || entries_[slot]->retired) {
            // This is an internal free-list invariant failure.  Do not expose
            // an invalid pointer or consume any existing live token.
            ++free_count_;
            return false;
        }
        EcReadToken &entry = *entries_[slot];
        entry.target_local_addr = target_local_addr;
        entry.remote_addr = remote_addr;
        entry.byte_count = byte_count;
        entry.alive_mask = alive_mask;
        entry.own_shard_idx = own_shard_idx;
        entry.scratch = scratch;
        // The posting loop records the actual successful posts below.  Do not
        // arm a pending count for the intended mask: a later post may fail,
        // and a CQE may arrive before the posting fibre records that post.
        entry.posted_mask = 0;
        entry.completed_mask = 0;
        entry.failed_mask = 0;
        entry.pending = 0;
        entry.acked = 0;
        entry.posting_done = false;
        entry.winner_handed_out = false;
        entry.profile_acquire_ns = ec_recovery_profile_enabled()
                                       ? ec_recovery_profile_now_ns()
                                       : 0;
        entry.in_use = true;
        ++in_use_count_;
        if (in_use_count_ > peak_in_use_) peak_in_use_ = in_use_count_;
        // The low bits hold the slot, so 0 is never a token id (generations
        // start at 1).
        *token_id_out = (static_cast<uint64_t>(entry.generation)
                         << kEcReadTokenSlotBits) |
                        static_cast<uint64_t>(slot);
        *entry_out = &entry;
        return true;
    }

    // Live entry of a token id; nullptr when the id is unknown, was already
    // released, or belongs to a previous generation of the same slot.
    EcReadToken *get(uint64_t token_id) {
        EcRecoveryTokenMutexGuard lock(mutex_);
        return find_locked(token_id);
    }

    // Compatibility wrapper for the old success-only API.  The state machine
    // now records success/failure and the posting boundary together; callers
    // that need scratch reclamation use complete_segment_event().
    EcReadToken *complete_segment(uint64_t token_id, uint8_t segment_idx) {
        const EcReadTokenEvent event =
            complete_segment_event(token_id, segment_idx, true);
        return event.kind == EcReadTokenEventKind::kWinner ? event.token
                                                            : nullptr;
    }

    // Record that one post_read() returned success.  This is deliberately a
    // separate transition from completion: the CQ can be reaped by another
    // thread before the posting fibre reaches this line.
    bool mark_segment_posted(uint64_t token_id, uint8_t segment_idx) {
        if (segment_idx >= kEcReadSegmentCount) return false;
        EcRecoveryTokenMutexGuard lock(mutex_);
        EcReadToken *entry = find_locked(token_id);
        if (entry == nullptr || entry->posting_done) return false;
        const uint8_t bit = static_cast<uint8_t>(1u << segment_idx);
        if ((entry->alive_mask & bit) == 0 ||
            (entry->posted_mask & bit) != 0) {
            return false;
        }
        entry->posted_mask |= bit;
        entry->pending = static_cast<uint8_t>(
            __builtin_popcount(static_cast<unsigned>(
                entry->posted_mask & ~entry->completed_mask)));
        return true;
    }

    // Complete one posted request.  `success=false` is a terminal error for
    // that request, not a reason to synthesize application success.  Completion
    // is accepted before mark_segment_posted() because a very fast CQ poll can
    // race the posting fibre; the later post mark closes that bookkeeping gap.
    EcReadTokenEvent complete_segment_event(uint64_t token_id,
                                            uint8_t segment_idx,
                                            bool success) {
        if (segment_idx >= kEcReadSegmentCount) {
            return {};
        }
        EcRecoveryTokenMutexGuard lock(mutex_);
        EcReadToken *entry = find_locked(token_id);
        if (entry == nullptr) return {};
        const uint8_t bit = static_cast<uint8_t>(1u << segment_idx);
        if ((entry->alive_mask & bit) == 0) return {};
        if ((entry->completed_mask & bit) != 0) {
            // A duplicate CQE is harmless, but after posting has ended it may
            // serve as the probe that claims a winner whose last CQE raced the
            // posting boundary.  Exactly one caller can set the latch.
            if (success && entry->posting_done &&
                entry->failed_mask == 0 &&
                entry->posted_mask == entry->alive_mask &&
                (entry->completed_mask & entry->posted_mask) ==
                    entry->posted_mask &&
                !entry->winner_handed_out) {
                entry->winner_handed_out = true;
                EcReadTokenEvent out;
                out.kind = EcReadTokenEventKind::kWinner;
                out.token = entry;
                out.segment = segment_idx;
                out.profile_acquire_ns = entry->profile_acquire_ns;
                out.profile_byte_count = entry->byte_count;
                return out;
            }
            return {};
        }
        entry->completed_mask |= bit;
        if (success) {
            entry->acked |= bit;
        } else {
            entry->failed_mask |= bit;
        }
        if ((entry->posted_mask & bit) != 0 && entry->pending != 0) {
            entry->pending--;
        }
        return finalize_locked(entry, true);
    }

    // Mark the posting phase complete.  No token or scratch slot is released
    // here unless all actual posts already have terminal CQEs.  A full-success
    // token is returned as kWinnerReady; the caller claims it through the
    // completion wrapper so winner handling remains in one place.
    EcReadTokenEvent finish_posting(uint64_t token_id) {
        EcRecoveryTokenMutexGuard lock(mutex_);
        EcReadToken *entry = find_locked(token_id);
        if (entry == nullptr) return {};
        entry->posting_done = true;
        return finalize_locked(entry, false);
    }

    // Cross-check of the posting path.  This keeps the old API but compares
    // the local post count with the authoritative posted mask.  It never
    // compares against `pending`, which decreases concurrently as CQEs arrive.
    bool pending_matches_posted(uint64_t token_id, uint8_t posted_count,
                                bool *token_gone_out) {
        if (token_gone_out != nullptr) *token_gone_out = false;
        EcRecoveryTokenMutexGuard lock(mutex_);
        EcReadToken *entry = find_locked(token_id);
        if (entry == nullptr) {
            if (token_gone_out != nullptr) *token_gone_out = true;
            return false;
        }
        return posted_count != 0 &&
               __builtin_popcount(static_cast<unsigned>(entry->posted_mask)) ==
                   posted_count;
    }

    // Returns a finished token to the table and bumps the slot generation so a
    // late CQE of the old read can never match a new one.  The winner of the
    // last segment calls this after it rebuilt the object.
    bool release(uint64_t token_id) { return abandon(token_id, nullptr); }

    // Same, but only succeeds after the state machine has reached a terminal
    // state.  In particular, it refuses to free a token while any posted WR
    // can still produce a CQE.
    bool abandon(uint64_t token_id,
                 ec_batch::EcStagingGroupSlot *scratch_out) {
        EcRecoveryTokenMutexGuard lock(mutex_);
        EcReadToken *entry = find_locked(token_id);
        if (entry == nullptr) return false;
        if (!entry->posting_done ||
            (entry->completed_mask & entry->posted_mask) !=
                entry->posted_mask) {
            return false;
        }
        free_locked(entry, scratch_out);
        return true;
    }

    // Compatibility safety net.  It is now intentionally strict: only a
    // posting-complete, all-terminal, non-success token may be reclaimed here.
    bool reap_all_posted_acked(uint64_t token_id,
                               ec_batch::EcStagingGroupSlot *scratch_out) {
        EcRecoveryTokenMutexGuard lock(mutex_);
        EcReadToken *entry = find_locked(token_id);
        if (entry == nullptr || entry->winner_handed_out) return false;
        const uint8_t posted = entry->posted_mask;
        if (!entry->posting_done || posted == 0 ||
            (entry->completed_mask & posted) != posted ||
            (entry->failed_mask == 0 && posted == entry->alive_mask)) {
            return false;
        }
        free_locked(entry, scratch_out);
        return true;
    }

    // True while a degraded read of this object slot is registered (posted or
    // being posted).  Used by the wait path to post each object only once.
    bool target_in_flight(uint64_t target_local_addr) const {
        if (target_local_addr == 0) return false;
        EcRecoveryTokenMutexGuard lock(mutex_);
        for (const std::unique_ptr<EcReadToken> &owned_entry : entries_) {
            if (owned_entry != nullptr && owned_entry->in_use &&
                owned_entry->target_local_addr == target_local_addr) {
                return true;
            }
        }
        return false;
    }

    size_t in_use() const {
        EcRecoveryTokenMutexGuard lock(mutex_);
        return in_use_count_;
    }

    size_t available() const {
        EcRecoveryTokenMutexGuard lock(mutex_);
        return free_count_;
    }

    // Monotonic high-water mark of simultaneously registered recovery reads.
    // This is diagnostic only and is intentionally independent of the current
    // free-list size, which may grow and then return to zero as reads finish.
    size_t peak_in_use() const {
        EcRecoveryTokenMutexGuard lock(mutex_);
        return peak_in_use_;
    }

private:
    // Add one stable-address entry and make it immediately available.  The
    // caller holds mutex_.  Both vectors reserve before mutating logical
    // state, so an allocation failure leaves every existing token unchanged.
    bool grow_slot_locked() noexcept {
        // The 32-bit slot field is the only namespace limit; no smaller
        // runtime cap is imposed by this table.
        constexpr uint64_t kSlotCount = uint64_t{1} << kEcReadTokenSlotBits;
        if (static_cast<uint64_t>(entries_.size()) >= kSlotCount) {
            return false;
        }
        try {
            std::unique_ptr<EcReadToken> fresh =
                std::make_unique<EcReadToken>();
            const size_t next_size = entries_.size() + 1;
            if (next_size > entries_.capacity()) {
                const size_t reserve = std::max<size_t>(
                    next_size, std::max<size_t>(8, entries_.capacity() * 2));
                entries_.reserve(reserve);
            }
            // Keep enough free-list capacity for every non-retired slot.  A
            // release then remains allocation-free even if all slots become
            // free after the table has grown.
            if (next_size > free_stack_.capacity())
                free_stack_.reserve(entries_.capacity());
            free_stack_.resize(next_size);
            const uint32_t slot = static_cast<uint32_t>(entries_.size());
            fresh->slot_index = slot;
            entries_.push_back(std::move(fresh));
            free_stack_[free_count_++] = slot;
            return true;
        } catch (...) {
            return false;
        }
    }

    // Hands one live entry back to the table: the slot leaves `in_use` (which
    // is the per-object latch target_in_flight() looks at, so the object can be
    // posted again) and the generation is bumped (or the slot retired when the
    // 24-bit namespace is exhausted) so that a late CQE of the old read can
    // never match a new one.  Every exit path of a degraded read goes through
    // here.
    void free_locked(EcReadToken *entry,
                     ec_batch::EcStagingGroupSlot *scratch_out) {
        EcRecoveryProfile::ScopedTimer release_timer(
            ::FarLib::cache::ec_recovery_profile(),
            EcRecoveryProfile::Stage::kTokenRelease);
        if (scratch_out != nullptr) *scratch_out = entry->scratch;
        entry->in_use = false;
        entry->posted_mask = 0;
        entry->completed_mask = 0;
        entry->failed_mask = 0;
        entry->winner_handed_out = false;
        entry->posting_done = false;
        entry->pending = 0;
        entry->acked = 0;
        entry->target_local_addr = 0;
        entry->byte_count = 0;
        entry->profile_acquire_ns = 0;
        if (in_use_count_ != 0) --in_use_count_;
        if (entry->generation == kEcReadTokenMaxGeneration) {
            // Never wrap the 24-bit generation: a stale CQE must not alias a
            // future token after namespace exhaustion.  The stable entry is
            // retained for diagnostics but is never put back on the free list.
            entry->retired = true;
        } else {
            ++entry->generation;
            // grow_slot_locked() reserves for every allocated entry, so this
            // push cannot allocate on a normal release path.
            free_stack_[free_count_++] = entry->slot_index;
        }
    }

    EcReadTokenEvent finalize_locked(EcReadToken *entry,
                                     bool hand_winner) {
        EcReadTokenEvent out;
        // Capture timestamp/size before any branch below can call
        // free_locked(), which recycles the token immediately.
        out.profile_acquire_ns = entry->profile_acquire_ns;
        out.profile_byte_count = entry->byte_count;
        if (!entry->posting_done) {
            out.kind = EcReadTokenEventKind::kPending;
            return out;
        }
        const uint8_t posted = entry->posted_mask;
        if (posted == 0) {
            // No post_read() succeeded, so there is no hardware request to
            // drain.  The scratch slot is safe to return immediately.
            out.kind = EcReadTokenEventKind::kRelease;
            free_locked(entry, &out.scratch);
            return out;
        }
        if ((entry->completed_mask & posted) != posted) {
            out.kind = EcReadTokenEventKind::kPending;
            return out;
        }
        if (entry->failed_mask != 0 || posted != entry->alive_mask) {
            out.kind = EcReadTokenEventKind::kRelease;
            free_locked(entry, &out.scratch);
            return out;
        }
        if (entry->winner_handed_out) {
            out.kind = EcReadTokenEventKind::kPending;
            return out;
        }
        out.segment = static_cast<uint8_t>(__builtin_ctz(
            static_cast<unsigned>(posted)));
        if (!hand_winner) {
            out.kind = EcReadTokenEventKind::kWinnerReady;
            return out;
        }
        entry->winner_handed_out = true;
        out.kind = EcReadTokenEventKind::kWinner;
        out.token = entry;
        return out;
    }

    EcReadToken *find_locked(uint64_t token_id) {
        // A CQE can contain an arbitrary/stale wr_id.  Validate the complete
        // 56-bit identity before converting its slot field to an index.
        if (token_id == 0 || (token_id & ~kEcReadTokenIdMask) != 0) {
            return nullptr;
        }
        const uint64_t generation = token_id >> kEcReadTokenSlotBits;
        if (generation == 0 || generation > kEcReadTokenMaxGeneration) {
            return nullptr;
        }
        const uint64_t slot_id = token_id & kEcReadTokenSlotMask;
        if (slot_id >= entries_.size()) return nullptr;
        const size_t slot = static_cast<size_t>(slot_id);
        if (entries_[slot] == nullptr) return nullptr;
        EcReadToken &entry = *entries_[slot];
        if (!entry.in_use || entry.retired) return nullptr;
        if (((static_cast<uint64_t>(entry.generation)
              << kEcReadTokenSlotBits) |
             static_cast<uint64_t>(entry.slot_index)) != token_id) {
            return nullptr;  // stale generation (or a foreign token id)
        }
        return &entry;
    }

    std::vector<std::unique_ptr<EcReadToken>> entries_;
    std::vector<uint32_t> free_stack_;
    size_t free_count_ = 0;
    size_t in_use_count_ = 0;
    size_t peak_in_use_ = 0;
    mutable std::mutex mutex_;
};

// Thread-safety note: complete_segment() hands the entry back to exactly one
// completion, and only that winner rebuilds and then calls release(), so the
// CQEs of a read can be reaped by different threads without holding a lock
// across the rebuild.

}  // namespace FarLib::cache::ec_read_recovery
