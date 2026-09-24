// ---------------------------------------------------------------------------
// Direct identity/state for the degraded EC-read path.
//
// A context address is the identity carried by every completion.  Contexts
// are individually allocated and remain at that address until the pool is
// destroyed; the low 40 bits of a context pointer shifted by eight, together
// with a 16-bit generation, fit in the existing 56-bit EC-read token field.
// The high bits of a WR id therefore retain the old
// encode_ec_read_wr_id(token, segment) contract.
//
// The state machine deliberately has no completion mutex.  One atomic word
// contains the generation, active bit, intended/already-posted/completed/
// failed masks, posting close, finalizer claim, and the retired bit.  A CQE
// can race the posting worker, and a late CQE can race reuse, without reading
// or changing a newly acquired context.
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#if __has_include(<infiniband/verbs.h>)
#include "recovery/ec_read_recovery.hpp"
#else
// The direct context pool itself only needs the staging-slot shape and the
// event enum.  Keep a dependency-light fallback so this header's unit tests
// can compile on a host without libibverbs; production builds include the
// existing recovery header above and therefore use its definitions.
#include "cache/alloc/ec_batch_staging.hpp"
#include "recovery/ec_recovery_profile.hpp"

namespace FarLib::cache::ec_read_recovery {

inline constexpr size_t kEcReadSegmentCount =
    ec_batch::kEcBatchSegmentsPerGroup;
inline constexpr uint64_t kEcReadWrIdTagBit = 1ull << 63;
inline constexpr uint64_t kEcReadWrIdNamespaceBit = 1ull << 62;
inline constexpr uint64_t kEcReadWrIdSegmentShift = 56;
inline constexpr uint64_t kEcReadWrIdSegmentMask = 0x3f;
inline constexpr uint64_t kEcReadWrIdTokenMask =
    (uint64_t{1} << kEcReadWrIdSegmentShift) - 1;

inline uint64_t encode_ec_read_wr_id(uint64_t token_id,
                                     uint8_t segment_idx) {
    assert(token_id != 0 && token_id <= kEcReadWrIdTokenMask);
    assert(segment_idx < kEcReadSegmentCount);
    return kEcReadWrIdTagBit | kEcReadWrIdNamespaceBit |
           (static_cast<uint64_t>(segment_idx) << kEcReadWrIdSegmentShift) |
           (token_id & kEcReadWrIdTokenMask);
}

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

enum class EcReadTokenEventKind : uint8_t {
    kIgnored,
    kPending,
    kWinnerReady,
    kWinner,
    kRelease,
};

}  // namespace FarLib::cache::ec_read_recovery
#endif

namespace FarLib::cache::ec_read_recovery {

// The existing wr_id encoder owns bits 56..63, leaving exactly 56 low bits.
// Contexts are alignas(256), so their low eight address bits are implicit.
inline constexpr size_t kEcReadContextPointerBits = 40;
inline constexpr size_t kEcReadContextGenerationBits = 16;
inline constexpr size_t kEcReadContextPointerShift = 8;
inline constexpr uint64_t kEcReadContextPointerMask =
    (uint64_t{1} << kEcReadContextPointerBits) - 1;
inline constexpr uint64_t kEcReadContextGenerationMask =
    (uint64_t{1} << kEcReadContextGenerationBits) - 1;
inline constexpr uint16_t kEcReadContextInitialGeneration = 1;
inline constexpr uint16_t kEcReadContextMaxGeneration =
    static_cast<uint16_t>(kEcReadContextGenerationMask);
inline constexpr uint64_t kEcReadContextTokenMask =
    (uint64_t{1} << (kEcReadContextPointerBits +
                     kEcReadContextGenerationBits)) - 1;

static_assert(kEcReadContextPointerBits + kEcReadContextGenerationBits ==
                  kEcReadWrIdSegmentShift,
              "the direct context token must fill the old 56-bit field");
static_assert(kEcReadContextTokenMask == kEcReadWrIdTokenMask,
              "direct context tokens must remain encode_ec_read_wr_id tokens");

class EcReadContextPool;

// One direct degraded-read context.  The fields below are immutable from the
// point of view of an active caller: acquire writes them before publishing the
// active state, and release only makes the context inactive.  The next owner
// acquire may overwrite them only after the previous generation is inactive.
// Callers must therefore inspect them only after receiving kWinner or use the
// copies carried by kRelease.
struct alignas(256) EcReadContext {
    uint64_t target_local_addr = 0;
    uint64_t remote_addr = 0;
    uint32_t byte_count = 0;
    // Segments fetched into this context's scratch slot.  This is the
    // selected/read mask (exactly four for the degraded path), not the full
    // physical endpoint-survivor mask; skipped scratch buffers are invalid.
    uint8_t alive_mask = 0;
    uint8_t own_shard_idx = 0;
    ec_batch::EcStagingGroupSlot scratch{};
    uint64_t profile_acquire_ns = 0;

    size_t segment_count() const { return kEcReadSegmentCount; }

    // This is useful to the posting path when it wants to construct a WR id
    // outside the pool API.  It reads only the atomic state word.
    uint64_t token_id() const noexcept;

private:
    friend class EcReadContextPool;

    // State layout (all fields are updated through compare_exchange):
    //
    //   [ 0,15] generation
    //   [   16] active
    //   [17,22] alive mask
    //   [23,28] posted mask
    //   [29,34] completed mask
    //   [35,40] failed mask
    //   [   41] posting closed
    //   [   42] finalizer claimed
    //   [   43] generation retired
    //
    // Bits 44..63 are reserved and always zero.
    std::atomic<uint64_t> state_{0};
    std::atomic<EcReadContext *> return_next_{nullptr};
    EcReadContextPool *pool_ = nullptr;
    size_t owner_index_ = 0;
};

static_assert(alignof(EcReadContext) >= 256,
              "direct context tokens require eight implicit low address bits");

// A terminal event carries a context pointer for kWinner/kRelease and a
// scratch/profile snapshot that remains valid even after the caller releases
// the context.  `token` is an intentional compatibility alias for code that
// used EcReadTokenEvent; both pointers are set to the same context.
struct EcReadContextEvent {
    EcReadTokenEventKind kind = EcReadTokenEventKind::kIgnored;
    EcReadContext *context = nullptr;
    EcReadContext *token = nullptr;
    ec_batch::EcStagingGroupSlot scratch{};
    uint8_t segment = 0;
    uint64_t profile_acquire_ns = 0;
    uint32_t profile_byte_count = 0;
};

class EcReadContextPool {
public:
    // owner_count is normally the number of RDMA client workers.  A default
    // of 256 keeps owner-side freelists independent without imposing a global
    // lock.  The second argument is useful for optional warm-up and defaults
    // to zero; storage grows lazily as contexts become live.
    explicit EcReadContextPool(size_t owner_count = 256,
                               size_t initial_capacity_per_owner = 0)
        : owner_count_(owner_count == 0 ? 1 : owner_count) {
        owners_.reserve(owner_count_);
        for (size_t i = 0; i < owner_count_; ++i) {
            auto owner = std::make_unique<Owner>();
            if (initial_capacity_per_owner != 0) {
                std::lock_guard<std::mutex> lock(owner->allocation_mutex);
                for (size_t n = 0; n < initial_capacity_per_owner; ++n) {
                    if (!grow_locked(*owner, i)) break;
                }
            }
            owners_.push_back(std::move(owner));
        }
    }

    EcReadContextPool(const EcReadContextPool &) = delete;
    EcReadContextPool &operator=(const EcReadContextPool &) = delete;

    ~EcReadContextPool() { assert(in_use() == 0); }

    // Register one read for owner_idx.  No target lookup or global object
    // latch is performed here; the caller owns the per-object claim.
    bool acquire(size_t owner_idx, uint64_t target_local_addr,
                 uint64_t remote_addr, uint32_t byte_count,
                 uint8_t alive_mask, uint8_t own_shard_idx,
                 const ec_batch::EcStagingGroupSlot &scratch,
                 uint64_t *token_id_out, EcReadContext **context_out) {
        if (token_id_out == nullptr || context_out == nullptr ||
            target_local_addr == 0 || owner_idx >= owner_count_) {
            return false;
        }

        Owner &owner = *owners_[owner_idx];
        std::lock_guard<std::mutex> lock(owner.allocation_mutex);
        drain_returns_locked(owner);

        EcReadContext *context = nullptr;
        while (!owner.free_list.empty()) {
            context = owner.free_list.back();
            owner.free_list.pop_back();
            const uint64_t state = context->state_.load(std::memory_order_acquire);
            if (is_free_state(state)) break;
            context = nullptr;
        }
        if (context == nullptr) {
            if (!grow_locked(owner, owner_idx)) return false;
            context = owner.free_list.back();
            owner.free_list.pop_back();
        }

        const uint64_t old_state = context->state_.load(std::memory_order_acquire);
        if (!is_free_state(old_state)) {
            // This should only be reachable after memory corruption or a
            // broken owner return stack.  Do not publish an active context.
            owner.free_list.push_back(context);
            return false;
        }
        const uint16_t generation = state_generation(old_state);
        if (state_retired(old_state) || generation == 0 ||
            !pointer_token_fits(context)) {
            return false;
        }

        constexpr uint8_t kAllSegments =
            static_cast<uint8_t>((1u << kEcReadSegmentCount) - 1u);
        const uint8_t intended = static_cast<uint8_t>(alive_mask & kAllSegments);

        // Metadata is initialized before the release-store that publishes the
        // active state.  No completion can read it before its winning CAS.
        context->target_local_addr = target_local_addr;
        context->remote_addr = remote_addr;
        context->byte_count = byte_count;
        context->alive_mask = intended;
        context->own_shard_idx = own_shard_idx;
        context->scratch = scratch;
        context->profile_acquire_ns =
            ec_recovery_profile_enabled() ? ec_recovery_profile_now_ns() : 0;

        const uint64_t active_state = pack_state(
            generation, true, intended, 0, 0, 0, false, false, false);
        context->state_.store(active_state, std::memory_order_release);

        const size_t owner_current =
            owner.in_use.fetch_add(1, std::memory_order_relaxed) + 1;
        const size_t current =
            in_use_total_.fetch_add(1, std::memory_order_relaxed) + 1;
        update_peak(owner.peak_in_use, owner_current);
        update_peak(total_peak_in_use_, current);
        *token_id_out = make_token_id(context, generation);
        *context_out = context;
        return true;
    }

    // Record a successful post.  Completion may have already set the
    // completed bit; this transition remains valid until posting is closed.
    bool mark_segment_posted(uint64_t token_id, uint8_t segment_idx) noexcept {
        if (segment_idx >= kEcReadSegmentCount) return false;
        EcReadContext *context = decode_context(token_id);
        if (context == nullptr || context->pool_ != this) return false;
        const uint8_t bit = static_cast<uint8_t>(1u << segment_idx);
        for (;;) {
            uint64_t state = context->state_.load(std::memory_order_acquire);
            if (!matches_active_generation(state, token_id) ||
                state_posting_closed(state) || state_finalizer_claimed(state)) {
                return false;
            }
            const uint8_t alive = state_alive_mask(state);
            const uint8_t posted = state_posted_mask(state);
            if ((alive & bit) == 0 || (posted & bit) != 0) return false;
            // The mask bit must be ORed at its segment position, not at the
            // first bit of the field.
            const uint64_t with_bit = state | (uint64_t{bit} << kPostedShift);
            if (context->state_.compare_exchange_weak(
                    state, with_bit, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
    }

    // Record one terminal CQE.  The completion path never takes an allocation
    // or pool mutex.  A completion before mark_segment_posted() is accepted;
    // the posting worker can subsequently set the posted bit.  Once posting
    // is closed, an unposted segment is ignored because it cannot belong to
    // this generation's hardware work.
    EcReadContextEvent complete_segment_event(uint64_t token_id,
                                              uint8_t segment_idx,
                                              bool success) noexcept {
        if (segment_idx >= kEcReadSegmentCount) return ignored_event();
        EcReadContext *context = decode_context(token_id);
        if (context == nullptr || context->pool_ != this) return ignored_event();
        const uint8_t bit = static_cast<uint8_t>(1u << segment_idx);
        for (;;) {
            uint64_t state = context->state_.load(std::memory_order_acquire);
            if (!matches_active_generation(state, token_id) ||
                state_finalizer_claimed(state)) {
                return ignored_event();
            }
            const uint8_t alive = state_alive_mask(state);
            if ((alive & bit) == 0) return ignored_event();
            const uint8_t posted = state_posted_mask(state);
            if (state_posting_closed(state) && (posted & bit) == 0) {
                return ignored_event();
            }
            if ((state_completed_mask(state) & bit) != 0) {
                return ignored_event();
            }

            uint64_t desired =
                state | (uint64_t{bit} << kCompletedShift);
            if (!success) desired |= uint64_t{bit} << kFailedShift;
            if (!context->state_.compare_exchange_weak(
                    state, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            return try_finalize(context, token_id, segment_idx);
        }
    }

    // Close the posting phase and, when all actually posted requests are
    // terminal, claim either the sole winner or the cleanup path directly.
    // There is no kWinnerReady probe in the direct pool.
    EcReadContextEvent finish_posting(uint64_t token_id) noexcept {
        EcReadContext *context = decode_context(token_id);
        if (context == nullptr || context->pool_ != this) return ignored_event();
        for (;;) {
            uint64_t state = context->state_.load(std::memory_order_acquire);
            if (!matches_active_generation(state, token_id)) {
                return ignored_event();
            }
            if (state_finalizer_claimed(state)) return ignored_event();
            if (state_posting_closed(state)) {
                return try_finalize(context, token_id, 0);
            }
            const uint64_t desired =
                state | (uint64_t{1} << kPostingClosedShift);
            if (!context->state_.compare_exchange_weak(
                    state, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            return try_finalize(context, token_id, 0);
        }
    }

    // A kWinner/kRelease caller invokes this after it has finished using the
    // context (and after a kRelease caller has returned its scratch lease).
    // The context itself is not freed by event production.  Releasing pushes
    // the stable address onto the owner's lock-free MPSC return stack.
    bool release(uint64_t token_id) noexcept {
        EcReadContext *context = decode_context(token_id);
        if (context == nullptr || context->pool_ != this) return false;
        Owner *owner = owners_[context->owner_index_].get();
        for (;;) {
            uint64_t state = context->state_.load(std::memory_order_acquire);
            if (!matches_active_generation(state, token_id) ||
                !state_posting_closed(state) ||
                !state_finalizer_claimed(state)) {
                return false;
            }
            const uint8_t posted = state_posted_mask(state);
            if ((state_completed_mask(state) & posted) != posted) return false;

            const uint16_t generation = state_generation(state);
            const bool retired = generation == kEcReadContextMaxGeneration;
            const uint16_t next_generation =
                retired ? generation : static_cast<uint16_t>(generation + 1);
            const uint64_t desired = pack_state(
                next_generation, false, 0, 0, 0, 0, false, false, retired);
            if (!context->state_.compare_exchange_weak(
                    state, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }

            owner->in_use.fetch_sub(1, std::memory_order_relaxed);
            in_use_total_.fetch_sub(1, std::memory_order_relaxed);
            if (!retired) {
                // Publish the node only after its generation became inactive;
                // an old CQE can therefore never race a new active state.
                EcReadContext *head =
                    owner->returned.load(std::memory_order_relaxed);
                do {
                    context->return_next_.store(head,
                                               std::memory_order_relaxed);
                } while (!owner->returned.compare_exchange_weak(
                    head, context, std::memory_order_release,
                    std::memory_order_relaxed));
            }
            return true;
        }
    }

    size_t in_use() const noexcept {
        return in_use_total_.load(std::memory_order_relaxed);
    }

    size_t capacity() const noexcept {
        size_t total = 0;
        for (const auto &owner : owners_) {
            total += owner->allocated.load(std::memory_order_relaxed);
        }
        return total;
    }

    size_t peak_in_use() const noexcept {
        return total_peak_in_use_.load(std::memory_order_relaxed);
    }

    size_t owner_count() const noexcept { return owner_count_; }

private:
    friend struct EcReadContext;

    static constexpr uint64_t kGenerationShift = 0;
    static constexpr uint64_t kActiveShift = 16;
    static constexpr uint64_t kAliveShift = 17;
    static constexpr uint64_t kPostedShift = 23;
    static constexpr uint64_t kCompletedShift = 29;
    static constexpr uint64_t kFailedShift = 35;
    static constexpr uint64_t kPostingClosedShift = 41;
    static constexpr uint64_t kFinalizerClaimShift = 42;
    static constexpr uint64_t kRetiredShift = 43;
    static constexpr uint64_t kSixBitMask = 0x3f;

    struct Owner {
        // This mutex is strictly owner-side allocation/free-list state.  It is
        // never taken by mark/complete/finish/release.
        mutable std::mutex allocation_mutex;
        std::vector<std::unique_ptr<EcReadContext>> storage;
        std::vector<EcReadContext *> free_list;
        std::atomic<EcReadContext *> returned{nullptr};
        std::atomic<size_t> allocated{0};
        std::atomic<size_t> in_use{0};
        std::atomic<size_t> peak_in_use{0};
    };

    static uint16_t state_generation(uint64_t state) noexcept {
        return static_cast<uint16_t>((state >> kGenerationShift) &
                                     kEcReadContextGenerationMask);
    }

    static bool state_active(uint64_t state) noexcept {
        return (state & (uint64_t{1} << kActiveShift)) != 0;
    }

    static uint8_t state_alive_mask(uint64_t state) noexcept {
        return static_cast<uint8_t>((state >> kAliveShift) & kSixBitMask);
    }

    static uint8_t state_posted_mask(uint64_t state) noexcept {
        return static_cast<uint8_t>((state >> kPostedShift) & kSixBitMask);
    }

    static uint8_t state_completed_mask(uint64_t state) noexcept {
        return static_cast<uint8_t>((state >> kCompletedShift) & kSixBitMask);
    }

    static uint8_t state_failed_mask(uint64_t state) noexcept {
        return static_cast<uint8_t>((state >> kFailedShift) & kSixBitMask);
    }

    static bool state_posting_closed(uint64_t state) noexcept {
        return (state & (uint64_t{1} << kPostingClosedShift)) != 0;
    }

    static bool state_finalizer_claimed(uint64_t state) noexcept {
        return (state & (uint64_t{1} << kFinalizerClaimShift)) != 0;
    }

    static bool state_retired(uint64_t state) noexcept {
        return (state & (uint64_t{1} << kRetiredShift)) != 0;
    }

    static bool is_free_state(uint64_t state) noexcept {
        return !state_active(state) && !state_retired(state) &&
               state_generation(state) != 0;
    }

    static uint64_t pack_state(uint16_t generation, bool active,
                               uint8_t alive, uint8_t posted,
                               uint8_t completed, uint8_t failed,
                               bool posting_closed, bool finalizer_claimed,
                               bool retired) noexcept {
        uint64_t state =
            static_cast<uint64_t>(generation) & kEcReadContextGenerationMask;
        if (active) state |= uint64_t{1} << kActiveShift;
        state |= (static_cast<uint64_t>(alive) & kSixBitMask) << kAliveShift;
        state |= (static_cast<uint64_t>(posted) & kSixBitMask) << kPostedShift;
        state |= (static_cast<uint64_t>(completed) & kSixBitMask)
                 << kCompletedShift;
        state |= (static_cast<uint64_t>(failed) & kSixBitMask) << kFailedShift;
        if (posting_closed) state |= uint64_t{1} << kPostingClosedShift;
        if (finalizer_claimed) state |= uint64_t{1} << kFinalizerClaimShift;
        if (retired) state |= uint64_t{1} << kRetiredShift;
        return state;
    }

    static bool pointer_token_fits(const EcReadContext *context) noexcept {
        const uintptr_t address = reinterpret_cast<uintptr_t>(context);
        return (address & ((uintptr_t{1} << kEcReadContextPointerShift) - 1)) ==
                   0 &&
               (address >> kEcReadContextPointerShift) <=
                   kEcReadContextPointerMask;
    }

    static uint64_t make_token_id(const EcReadContext *context,
                                  uint16_t generation) noexcept {
        const uint64_t pointer_part =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context) >>
                                  kEcReadContextPointerShift) &
            kEcReadContextPointerMask;
        return (static_cast<uint64_t>(generation)
                << kEcReadContextPointerBits) |
               pointer_part;
    }

    static EcReadContext *decode_context(uint64_t token_id) noexcept {
        if (token_id == 0 || (token_id & ~kEcReadContextTokenMask) != 0) {
            return nullptr;
        }
        const uint64_t pointer_part = token_id & kEcReadContextPointerMask;
        const uint16_t generation = static_cast<uint16_t>(
            (token_id >> kEcReadContextPointerBits) &
            kEcReadContextGenerationMask);
        if (pointer_part == 0 || generation == 0) return nullptr;
        const uintptr_t address = static_cast<uintptr_t>(pointer_part)
                                  << kEcReadContextPointerShift;
        if ((address & ((uintptr_t{1} << kEcReadContextPointerShift) - 1)) !=
            0) {
            return nullptr;
        }
        return reinterpret_cast<EcReadContext *>(address);
    }

    static bool matches_active_generation(uint64_t state,
                                         uint64_t token_id) noexcept {
        if (!state_active(state) || state_retired(state)) return false;
        const uint16_t token_generation = static_cast<uint16_t>(
            (token_id >> kEcReadContextPointerBits) &
            kEcReadContextGenerationMask);
        return token_generation != 0 &&
               state_generation(state) == token_generation;
    }

    static void update_peak(std::atomic<size_t> &peak,
                            size_t current) noexcept {
        size_t old = peak.load(std::memory_order_relaxed);
        while (old < current &&
               !peak.compare_exchange_weak(old, current,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
        }
    }

    static EcReadContextEvent ignored_event() noexcept { return {}; }

    EcReadContextEvent terminal_event(EcReadContext *context,
                                      EcReadTokenEventKind kind,
                                      uint8_t segment) const noexcept {
        EcReadContextEvent event;
        event.kind = kind;
        event.context = context;
        event.token = context;
        event.scratch = context->scratch;
        event.segment = segment;
        event.profile_acquire_ns = context->profile_acquire_ns;
        event.profile_byte_count = context->byte_count;
        return event;
    }

    EcReadContextEvent try_finalize(EcReadContext *context,
                                    uint64_t token_id,
                                    uint8_t segment) noexcept {
        for (;;) {
            uint64_t state = context->state_.load(std::memory_order_acquire);
            if (!matches_active_generation(state, token_id)) {
                return ignored_event();
            }
            if (!state_posting_closed(state)) {
                return EcReadContextEvent{
                    EcReadTokenEventKind::kPending, nullptr, nullptr, {},
                    segment, 0, 0};
            }
            if (state_finalizer_claimed(state)) return ignored_event();

            const uint8_t posted = state_posted_mask(state);
            const uint8_t completed = state_completed_mask(state);
            if ((completed & posted) != posted) {
                return EcReadContextEvent{
                    EcReadTokenEventKind::kPending, nullptr, nullptr, {},
                    segment, 0, 0};
            }

            const bool release =
                posted == 0 || state_failed_mask(state) != 0 ||
                posted != state_alive_mask(state);
            const uint64_t desired =
                state | (uint64_t{1} << kFinalizerClaimShift);
            if (!context->state_.compare_exchange_weak(
                    state, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            // The successful finalizer CAS is the publication point for every
            // non-atomic field copied below.  No other transition may release
            // or recycle this generation before this event is returned.
            return terminal_event(
                context,
                release ? EcReadTokenEventKind::kRelease
                        : EcReadTokenEventKind::kWinner,
                segment);
        }
    }

    bool grow_locked(Owner &owner, size_t owner_idx) {
        try {
            owner.storage.reserve(owner.storage.size() + 1);
            owner.free_list.reserve(owner.storage.size() + 1);
            auto context = std::make_unique<EcReadContext>();
            context->pool_ = this;
            context->owner_index_ = owner_idx;
            context->state_.store(
                pack_state(kEcReadContextInitialGeneration, false, 0, 0, 0,
                           0, false, false, false),
                std::memory_order_relaxed);
            EcReadContext *raw = context.get();
            owner.storage.push_back(std::move(context));
            owner.free_list.push_back(raw);
            owner.allocated.fetch_add(1, std::memory_order_relaxed);
            return true;
        } catch (...) {
            return false;
        }
    }

    static void drain_returns_locked(Owner &owner) noexcept {
        EcReadContext *head =
            owner.returned.exchange(nullptr, std::memory_order_acquire);
        while (head != nullptr) {
            EcReadContext *next =
                head->return_next_.load(std::memory_order_relaxed);
            head->return_next_.store(nullptr, std::memory_order_relaxed);
            owner.free_list.push_back(head);
            head = next;
        }
    }

    size_t owner_count_ = 0;
    std::vector<std::unique_ptr<Owner>> owners_;
    std::atomic<size_t> in_use_total_{0};
    std::atomic<size_t> total_peak_in_use_{0};
};

inline uint64_t EcReadContext::token_id() const noexcept {
    const uint64_t state = state_.load(std::memory_order_acquire);
    if (!EcReadContextPool::state_active(state) ||
        EcReadContextPool::state_retired(state)) {
        return 0;
    }
    return EcReadContextPool::make_token_id(this,
                                            EcReadContextPool::state_generation(
                                                state));
}

}  // namespace FarLib::cache::ec_read_recovery
