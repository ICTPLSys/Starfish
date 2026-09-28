#pragma once

// Erasure-coded (EC) stripe layout for small remote objects.
//
// Ported from the design-3 "sponge" prototype
// (starfish-design3-20260824/include/cache/alloc/small_object_stripe.hpp) with
// two deliberate adaptations:
//
//   1. The shard size is generalized to the region size of *this* tree
//      (::FarLib::allocator::RegionSize == 256 KiB, see
//      include/cache/region_based_allocator.hpp) instead of the prototype's
//      hard-coded 512 KiB.  A stripe is 4 data shards + 2 parity shards and
//      every shard occupies one whole region.
//   2. The old-runtime dependencies (mailbox/mutex profiling scaffolding,
//      FABRIC_* helpers, the prototype's region size) are dropped.  The
//      address reverse mapping (binding_for_addr / get_slot_layout) is kept
//      verbatim in spirit, because entry.hpp must stay untouched this phase and
//      therefore every object address has to identify its own shard/stripe
//      position.
//
// Phase 1 scope: layout, geometry, address decoding and the allocator-side
// dispatch hooks.  Phase 2 added the RS(4,2) parity codec
// (include/cache/alloc/small_object_stripe_codec.hpp) and the shard-level
// SmallObjectStripeEncoder at the bottom of this file: encode() produces the two
// parity shards from the four data shards, rebuild()/rebuild_one() recover any 1
// or 2 missing shards from the surviving four.  The write path and the ACK path
// are still NOT wired: nothing in the runtime calls the encoder yet.
//
// Nothing in this file runs while ft_method == none (the default): the stripe
// manager is only initialized by RemoteAllocator when ft_enabled() is true, and
// every public entry point short-circuits on enabled().
//
// ec_batch may configure one connected standby endpoint
// (Configure::ft_standby_endpoint).  It is excluded from new stripe/group
// placement until mark_endpoint_dead() observes a failure; existing stripes
// retain their original addresses for recovery.

#include <array>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "../../../../common/runtime_metadata.hpp"
#include "cache/alloc/region_remote_allocator.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"

#include "utils/ec_benchmark_phase.hpp"
#include "utils/ec_rmw_prepare_trace.hpp"

#include "rdma/config.hpp"
#include "utils/debug.hpp"

namespace FarLib::cache {

// One EC shard is one region.  The value must track the library's region size;
// the static_assert below makes any drift a compile-time error.
inline constexpr size_t SmallObjectStripeShardSize =
    ::FarLib::allocator::RegionSize;
static_assert(
    SmallObjectStripeShardSize == ::FarLib::rdma::Configure::ft_ec_shard_size_bytes,
    "EC shard size must equal the region size (and ft_small_stripe_shard_size_bytes)");

// Geometry: 4 data shards + 2 parity shards, one region each.
inline constexpr uint8_t kSmallObjectStripeDataShards = 4;
inline constexpr uint8_t kSmallObjectStripeParityShards = 2;
inline constexpr uint8_t kSmallObjectStripeShardCount =
    kSmallObjectStripeDataShards + kSmallObjectStripeParityShards;
static_assert(kSmallObjectStripeDataShards == 4 &&
                  kSmallObjectStripeParityShards == 2,
              "this phase only supports a 4+2 EC layout");

// A manager-side snapshot for one single-shard background rebuild.  The
// addresses in original_base remain logical: the manager keeps its existing
// bindings immutable and publishes a replacement only through
// resolve_rebuilt_addr().
struct BackgroundRebuildStripe {
    uint64_t stripe_id = std::numeric_limits<uint64_t>::max();
    uint8_t failed_shard = 0xff;
    uint32_t failed_endpoint = std::numeric_limits<uint32_t>::max();
    uint32_t slot_size = 0;
    uint64_t shard_bytes = 0;
    std::array<uint64_t, kSmallObjectStripeShardCount> original_base{};
    std::array<uint32_t, kSmallObjectStripeShardCount> endpoint{};
    uint64_t live_groups = 0;
    uint64_t live_objects = 0;
    // Valid, durable sealed groups whose codewords contain at least one live
    // object.  This is optional metadata for a worker doing bounded
    // codeword validation; free/dead slots are deliberately not listed.
    std::vector<uint32_t> live_slot_ids;
    BackgroundRebuildStripe() {
        original_base.fill(::FarLib::allocator::remote::InvalidRemoteAddr);
        endpoint.fill(std::numeric_limits<uint32_t>::max());
    }
};
enum class BackgroundRebuildStatus : uint8_t {
    Skip = 0,
    Busy = 1,
    Ready = 2,
    Unsupported = 3,
};
// Stripe geometry relative to a remote capacity (in shard/region units).
class SmallObjectStripeShardTable {
public:
    struct Location {
        uint64_t shard_region_id = 0;
        uint32_t slot_id = 0;
        uint32_t slot_size = 0;
        uint64_t slot_offset = 0;
    };

    SmallObjectStripeShardTable() = default;

    SmallObjectStripeShardTable(size_t remote_capacity_bytes, size_t shard_size) {
        init(remote_capacity_bytes, shard_size);
    }

    void init(size_t remote_capacity_bytes, size_t shard_size) {
        assert(shard_size != 0);
        assert((shard_size & (shard_size - 1)) == 0);
        remote_capacity_bytes_ = remote_capacity_bytes;
        shard_size_ = shard_size;
        shard_region_count_ = remote_capacity_bytes / shard_size_;
    }

    bool enabled() const { return shard_region_count_ != 0; }

    size_t remote_capacity_bytes() const { return remote_capacity_bytes_; }
    size_t shard_region_count() const { return shard_region_count_; }
    size_t shard_size() const { return shard_size_; }

    // Number of stripes that fit into the configured remote capacity.
    size_t stripe_count() const {
        return shard_region_count_ / kSmallObjectStripeShardCount;
    }

    // Address reverse mapping step 1: which shard (== region) does an address
    // belong to?  Pure arithmetic, valid for every address inside the stripe
    // range, independent of any runtime state.
    uint64_t shard_region_id_of_addr(uint64_t remote_addr) const {
        if (!enabled() || shard_size_ == 0 ||
            remote_addr >= remote_capacity_bytes_) {
            return kInvalidShardRegionId;
        }
        return remote_addr / shard_size_;
    }

    // Geometric shard label of a *contiguous* shard unit, i.e. of the
    // idealized addressing where shard unit n is simply the n-th 256 KiB extent
    // (so shard slot = n % 6).  This is the convention the prototype used.
    //
    // NOTE: under the region-pair allocator actually in use,
    // RemoteGlobalHeap::allocate_whole_region_on_endpoint() hands out whole
    // 512 KiB regions, so every shard base is at a 512 KiB boundary and its
    // 256 KiB region id is always even.  A stripe's 6 shards therefore live in
    // 6 *distinct* regions and are NOT contiguous, so shard index can no longer
    // be recovered from the region id alone.  The authoritative attribution is
    // the per-region binding table (binding_for_addr / get_slot_layout), which
    // records the exact (stripe_id, shard_idx, bin) for every data shard at
    // allocation time.  The two helpers below are kept only for the geometric
    // convention and must not be used to classify a live address.
    static uint8_t data_shard_index(uint64_t shard_region_id) {
        return static_cast<uint8_t>(shard_region_id % kSmallObjectStripeShardCount);
    }

    static bool is_parity_shard(uint8_t shard_index) {
        return shard_index >= kSmallObjectStripeDataShards;
    }
    // Slot-level decode: object address -> (shard region, slot, offset).
    Location decode(uint64_t remote_addr, uint32_t slot_size) const {
        assert(slot_size != 0);
        assert((slot_size & (slot_size - 1)) == 0);
        assert(shard_size_ != 0);
        assert((shard_size_ % slot_size) == 0);
        assert(remote_addr < remote_capacity_bytes_);
        Location loc;
        loc.shard_region_id = remote_addr / shard_size_;
        loc.slot_offset = remote_addr % shard_size_;
        loc.slot_id = static_cast<uint32_t>(loc.slot_offset / slot_size);
        loc.slot_size = slot_size;
        return loc;
    }

    static bool size_is_supported(const ::FarLib::rdma::Configure &config,
                                 size_t size) {
        return config.ft_small_object(size);
    }

    static constexpr uint64_t kInvalidShardRegionId =
        std::numeric_limits<uint64_t>::max();

private:
    size_t remote_capacity_bytes_ = 0;
    size_t shard_size_ = 0;
    size_t shard_region_count_ = 0;
};

class SmallObjectStripeManager {
public:
    using BackgroundRebuildStripe = ::FarLib::cache::BackgroundRebuildStripe;
    using BackgroundRebuildStatus = ::FarLib::cache::BackgroundRebuildStatus;
    using View = BackgroundRebuildStripe;
#ifdef FARLIB_REBUILD_TEST_HOOK
    inline static void (*background_test_after_create)(SmallObjectStripeManager *, uint64_t) = nullptr;
#endif
    // Aggregate counters only: do not collect example objects on the hot path.
    struct SpaceUsage {
        // Active means sealed or in-progress.  A dead group is retained as a
        // reusable reservation and is reported separately below.
        uint64_t occupied_group_bytes = 0;
        uint64_t sealed_group_bytes = 0;
        uint64_t live_payload_bytes = 0;
        uint64_t live_data_slot_bytes = 0;
        // Short name used by the reporting layer; equal to
        // live_data_slot_bytes in every snapshot.
        uint64_t live_slot_bytes = 0;
        uint64_t parity_bytes = 0;
        uint64_t trapped_hole_bytes = 0;
        uint64_t padding_bytes = 0;
        std::array<uint64_t, 5> group_counts_by_live{};
        uint64_t split_groups = 0;
        uint64_t in_progress_groups = 0;
        uint64_t reusable_groups = 0;
        uint64_t reusable_group_bytes = 0;
        uint64_t reserved_stripe_bytes = 0;
        uint64_t unknown_payload_objects = 0;
        uint64_t sealed_small_groups = 0;
        std::vector<uint64_t> endpoint_occupied_bytes;
    };

private:
    static constexpr uint8_t kDataShards = kSmallObjectStripeDataShards;
    static constexpr uint8_t kParityShards = kSmallObjectStripeParityShards;
    static constexpr uint8_t kShardCount = kSmallObjectStripeShardCount;
    static constexpr uint64_t kInvalidStripeId =
        std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t kInvalidRemoteAddr =
        ::FarLib::allocator::remote::InvalidRemoteAddr;
    static constexpr uint64_t kBitsPerWord = 64;
    static constexpr uint64_t kInvalidBinding = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t kStripeIdMask = (1ULL << 48) - 1ULL;

    struct Bitmap {
        std::vector<uint64_t> words;
        size_t next_word = 0;

        void init_full(size_t bit_count) {
            size_t word_count = (bit_count + kBitsPerWord - 1) / kBitsPerWord;
            words.assign(word_count, std::numeric_limits<uint64_t>::max());
            next_word = 0;
            if (bit_count == 0 || word_count == 0) return;
            size_t valid_bits = bit_count % kBitsPerWord;
            if (valid_bits != 0) {
                words[word_count - 1] = (1ULL << valid_bits) - 1ULL;
            }
        }

        void init_empty(size_t bit_count) {
            size_t word_count = (bit_count + kBitsPerWord - 1) / kBitsPerWord;
            words.assign(word_count, 0);
            next_word = 0;
        }

        bool test(size_t bit) const {
            return (words[bit / kBitsPerWord] &
                    (1ULL << (bit % kBitsPerWord))) != 0;
        }

        void set(size_t bit) {
            size_t word_idx = bit / kBitsPerWord;
            words[word_idx] |= (1ULL << (bit % kBitsPerWord));
            next_word = word_idx;
        }

        void reset(size_t bit) {
            words[bit / kBitsPerWord] &= ~(1ULL << (bit % kBitsPerWord));
        }

        bool any_set() const {
            for (uint64_t word : words) {
                if (word != 0) return true;
            }
            return false;
        }

        bool find_and_reset(size_t *bit_out) {
            if (words.empty()) return false;
            for (size_t i = 0; i < words.size(); i++) {
                size_t word_idx = next_word + i;
                if (word_idx >= words.size()) {
                    word_idx -= words.size();
                }
                uint64_t word = words[word_idx];
                if (word == 0) continue;
                int bit = __builtin_ffsl(word);
                assert(bit > 0);
                size_t bit_idx =
                    word_idx * kBitsPerWord + static_cast<size_t>(bit - 1);
                words[word_idx] = word & ~(1ULL << (bit - 1));
                if (words[word_idx] == 0) {
                    next_word = word_idx + 1;
                    if (next_word >= words.size()) {
                        next_word = 0;
                    }
                } else {
                    next_word = word_idx;
                }
                *bit_out = bit_idx;
                return true;
            }
            return false;
        }

        // Find a set bit without changing bitmap.  Candidate bitmaps are
        // protected by their stripe mutex, so this gives indexed reuse a
        // bounded cursor walk without an object/slot table scan.
        bool find_set(size_t start, size_t *bit_out) const {
            if (words.empty() || bit_out == nullptr) return false;
            const size_t word_count = words.size();
            const size_t first_word = (start / kBitsPerWord) % word_count;
            const size_t first_bit = start % kBitsPerWord;
            for (size_t i = 0; i < word_count; i++) {
                const size_t word_idx = (first_word + i) % word_count;
                uint64_t word = words[word_idx];
                if (i == 0 && first_bit != 0) {
                    word &= ~((1ULL << first_bit) - 1ULL);
                }
                if (word == 0) continue;
                const int bit = __builtin_ctzll(word);
                *bit_out = word_idx * kBitsPerWord +
                           static_cast<size_t>(bit);
                return true;
            }
            // With one word, the loop above masks bits before `start` and
            // never visits that same word again. Complete the wraparound
            // explicitly; otherwise a lone candidate below the cursor could
            // remain unreachable forever.
            if (first_bit != 0) {
                const uint64_t word =
                    words[first_word] & ((1ULL << first_bit) - 1ULL);
                if (word != 0) {
                    const int bit = __builtin_ctzll(word);
                    *bit_out = first_word * kBitsPerWord +
                               static_cast<size_t>(bit);
                    return true;
                }
            }
            return false;
        }
    };

    // Reuse keys are the six semantic behavior groups used by ec_batch.  The
    // key is intentionally fixed-size: index construction happens before the
    // first Work phase and never allocates on the RMW hot path.
    static constexpr uint8_t kReuseBehaviorGroupCount = 6;

    // Manager-wide stripe summary.  Bits are published with release ordering
    // after a per-stripe candidate bit is set, and cleared only while holding
    // that stripe's mutex.  A stale set bit is harmless; begin_slot_reuse_*()
    // revalidates under the same mutex before reserving.
    struct AtomicBitmap {
        std::unique_ptr<std::atomic<uint64_t>[]> words;
        size_t word_count = 0;
        size_t bit_count = 0;

        void init(size_t bits) {
            bit_count = bits;
            word_count = (bits + kBitsPerWord - 1) / kBitsPerWord;
            words.reset(word_count == 0
                            ? nullptr
                            : new std::atomic<uint64_t>[word_count]);
            for (size_t i = 0; i < word_count; i++) {
                words[i].store(0, std::memory_order_relaxed);
            }
        }

        void set(size_t bit) {
            if (bit >= bit_count || word_count == 0) return;
            words[bit / kBitsPerWord].fetch_or(
                1ULL << (bit % kBitsPerWord), std::memory_order_release);
        }

        void reset(size_t bit) {
            if (bit >= bit_count || word_count == 0) return;
            words[bit / kBitsPerWord].fetch_and(
                ~(1ULL << (bit % kBitsPerWord)), std::memory_order_release);
        }

        bool find_set(size_t start, size_t *bit_out) const {
            if (word_count == 0 || bit_out == nullptr) return false;
            const size_t first_word = (start / kBitsPerWord) % word_count;
            const size_t first_bit = start % kBitsPerWord;
            for (size_t i = 0; i < word_count; i++) {
                const size_t word_idx = (first_word + i) % word_count;
                uint64_t word = words[word_idx].load(std::memory_order_acquire);
                if (i == 0 && first_bit != 0) {
                    word &= ~((1ULL << first_bit) - 1ULL);
                }
                if (word == 0) continue;
                const int bit = __builtin_ctzll(word);
                const size_t candidate =
                    word_idx * kBitsPerWord + static_cast<size_t>(bit);
                if (candidate < bit_count) {
                    *bit_out = candidate;
                    return true;
                }
            }
            // Complete one-word wraparound; see Bitmap::find_set().
            if (first_bit != 0) {
                const uint64_t word =
                    words[first_word].load(std::memory_order_acquire) &
                    ((1ULL << first_bit) - 1ULL);
                if (word != 0) {
                    const int bit = __builtin_ctzll(word);
                    const size_t candidate =
                        first_word * kBitsPerWord + static_cast<size_t>(bit);
                    if (candidate < bit_count) {
                        *bit_out = candidate;
                        return true;
                    }
                }
            }
            return false;
        }
    };

    struct Stripe {
        uint64_t stripe_id = kInvalidStripeId;
        // Placement and group-only status are immutable after publication.
        bool group_stripe = false;
        // Protected by recovery_replacement_mutex_, not the stripe mutex.
        bool recovery_replacement_credited = false;
        uint16_t bin = 0;
        uint32_t slot_size = 0;
        uint32_t slots_per_shard = 0;
        uint32_t live_count = 0;
        uint32_t dead_count = 0;
        uint32_t free_count = 0;
        uint8_t next_data_shard = 0;
        bool in_pool = false;
        // --- slot-group metadata (EC group allocation; lazily materialised) --
        // group_state_words is the group state bitmap: 2 bits per slot_id
        // (slot_id -> bits [2*slot_id, 2*slot_id+1]); group_live_mask holds the
        // 4-bit live mask per slot_id (bit i = data shard i).
        bool in_group_pool = false;
        uint32_t next_group_slot = 0;   // group allocation cursor
        uint32_t reuse_scan_slot = 0;   // bounded retired-slot scan cursor
        uint32_t group_dead_count = 0;  // dead groups, reusable as a whole
        std::array<uint32_t, kShardCount> shard_endpoint{};
        std::vector<uint64_t> group_state_words;
        std::vector<uint8_t> group_live_mask;
        // A slot-reuse transaction reserves one retired data bit without
        // changing the committed live mask.  The coordinator publishes the
        // bit only after its three absolute writes have reached terminal
        // completion.  Keeping this separate is what prevents a pending
        // target from being mistaken for a readable object or a dead group.
        std::vector<uint8_t> group_pending_mask;
        std::vector<uint8_t> group_durable;
        std::vector<uint32_t> group_behavior_group;
        std::vector<uint64_t> group_generation;
        // Live *object* counter per slot_id: how many of the group's four data
        // segments still hold an object.  Maintained together with
        // group_live_mask (count == popcount(mask)) by seal_slot_group /
        // release_group_object / mark_dead_group.
        std::vector<uint8_t> group_live_objects;
        // A split object owns all four data fragments as one public object.
        // The live mask remains 0xf; this metadata distinguishes split
        // ownership from an ordinary batch with holes.
        std::vector<uint8_t> group_is_split;
        // Payload accounting is group-local so release can subtract the
        // exact object size without an address-keyed table or a global lock.
        std::vector<std::array<uint32_t, kDataShards>> group_payload_sizes;
        std::vector<uint8_t> group_payload_unknown;
        // Per-stripe accounting.  All fields below are plain values and are
        // read/written only while this stripe's existing mutex is held.
        uint64_t occupied_group_bytes = 0;
        uint64_t sealed_group_bytes = 0;
        uint64_t live_payload_bytes = 0;
        uint64_t live_data_slot_bytes = 0;
        uint64_t parity_bytes = 0;
        uint64_t trapped_hole_bytes = 0;
        uint64_t padding_bytes = 0;
        std::array<uint64_t, 5> group_counts_by_live{};
        uint64_t split_groups = 0;
        uint64_t in_progress_groups = 0;
        uint64_t reusable_groups = 0;
        uint64_t unknown_payload_objects = 0;
        uint64_t sealed_small_groups = 0;
        // Set (never cleared) when group metadata is materialised for this
        // stripe, so the group-aware release can tell "this stripe never served
        // a group" without taking the stripe mutex.
        std::atomic<bool> has_groups{false};
        // Conservative fast-path hint for begin_slot_reuse().  It may stay
        // true after a candidate is consumed, but it is cleared only by a
        // complete under-lock slot sweep; false is never published while an
        // eligible partial durable group can still exist.
        std::atomic<bool> reuse_candidate{false};
        // Exact candidate slots, partitioned by behavior key.  Each bit is
        // set only for a durable sealed non-split group with no pending
        // update and at least one retired data segment.  Access is serialized
        // by this stripe's mutex; manager-level summary bits point directly
        // to stripes whose bitmap for a requested key is non-empty.
        std::array<Bitmap, kReuseBehaviorGroupCount> reuse_bits;
        std::array<uint64_t, kShardCount> shard_base{};
        // Published replacement regions are physical-only redirects.  The
        // original shard_base values above are never rewritten.
        std::array<std::atomic<uint64_t>, kShardCount> rebuilt_base;

        std::array<Bitmap, kDataShards> free_bits;
        std::array<Bitmap, kDataShards> dead_bits;
        mutable std::mutex mutex;

        Stripe() {
            shard_base.fill(kInvalidRemoteAddr);
            shard_endpoint.fill(std::numeric_limits<uint32_t>::max());
            for (auto &base : rebuilt_base) {
                base.store(kInvalidRemoteAddr, std::memory_order_relaxed);
            }
        }
    };

    struct DecodedBinding {
        uint64_t stripe_id = kInvalidStripeId;
        uint8_t shard_idx = 0xff;
        uint16_t bin = 0;
    };

public:
    // Everything needed to locate one small object and its parity peers.
    // Phase 2 (parity encoding) consumes this; phase 1 only exposes it.
    struct SlotLayout {
        uint64_t stripe_id = kInvalidStripeId;
        uint16_t bin = 0;
        uint32_t slot_size = 0;
        uint32_t slots_per_shard = 0;
        uint32_t slot_id = 0;
        uint64_t slot_offset = 0;
        uint8_t data_shard_idx = 0xff;
        uint64_t data_addr = kInvalidRemoteAddr;
        uint64_t data_shard_base = kInvalidRemoteAddr;
        std::array<uint64_t, kParityShards> parity_addr{};
        std::array<uint64_t, kParityShards> parity_shard_base{};
        std::array<uint8_t, kParityShards> parity_shard_idx{};

        SlotLayout() {
            parity_addr.fill(kInvalidRemoteAddr);
            parity_shard_base.fill(kInvalidRemoteAddr);
            for (uint8_t i = 0; i < kParityShards; i++) {
                parity_shard_idx[i] = static_cast<uint8_t>(kDataShards + i);
            }
        }
    };


    // -------------------------------------------------------------------
    // Slot groups (EC group-granularity allocation)
    //
    // One group is one slot offset on all six shards: 4 data segments (one
    // per data shard, segments 0..3) plus the 2 parity segments (4..5) at
    // that same offset.  The six segments always live on six distinct
    // endpoints.  The four data slots of a group only ever exist together:
    // they are reserved atomically from the per-shard free bitmaps and can
    // only be reused as a whole group, never piecewise.
    // -------------------------------------------------------------------
    enum SlotGroupState : uint8_t {
        kSlotGroupFree = 0,
        kSlotGroupInProgress = 1,
        kSlotGroupSealed = 2,
        kSlotGroupDead = 3,
    };

    struct SlotGroupId {
        uint64_t stripe_id = kInvalidStripeId;
        uint32_t slot_id = 0;

        bool valid() const { return stripe_id != kInvalidStripeId; }
    };

    // One of the six segments of a group.  offset is the in-shard byte offset
    // and is identical for all six segments.
    struct SlotGroupSegment {
        uint8_t shard_idx = 0xff;  // 0..3 data, 4..5 parity
        uint32_t endpoint_idx = std::numeric_limits<uint32_t>::max();
        uint64_t offset = 0;
        uint64_t addr = kInvalidRemoteAddr;
        uint32_t slot_size = 0;
    };

    // Handle of one allocated group: the six segments plus the common slot
    // offset/size needed to re-derive the layout later.
    struct SlotGroupHandle {
        SlotGroupId id;
        uint16_t bin = 0;
        uint32_t slot_size = 0;
        uint32_t slots_per_shard = 0;
        uint64_t slot_offset = 0;
        std::array<SlotGroupSegment, kShardCount> segments{};
    };

    // Client-coordinated reuse of one retired data slot in an already sealed
    // group.  `generation` is an ABA guard for delayed commit/abort callbacks;
    // `data_shard` identifies the retired data segment that remains outside
    // the committed live mask until commit_slot_reuse().
    struct SlotReuse {
        SlotGroupHandle group;
        uint64_t generation = 0;
        uint8_t data_shard = 0xff;
        uint32_t size = 0;
        uint32_t behavior_group = 0;
    };
private:
    struct SizeClassPool {
        std::mutex mutex;
        std::vector<uint64_t> stripes_with_space;
    };

    struct LocalLeaseEntry {
        const SmallObjectStripeManager *owner;
        uint64_t stripe_id;

        LocalLeaseEntry() : owner(nullptr), stripe_id(kInvalidStripeId) {}
    };

    struct LocalLeases {
        std::array<LocalLeaseEntry, ::FarLib::allocator::RegionBinCount> bins;
    };

    SmallObjectStripeShardTable shard_table_;
    std::vector<std::unique_ptr<Stripe>> stripe_storage_;
    std::unique_ptr<std::atomic<Stripe *>[]> stripe_index_;
    std::unique_ptr<std::atomic<uint64_t>[]> shard_bindings_;
    size_t stripe_capacity_ = 0;
    size_t shard_binding_count_ = 0;
    std::atomic<uint64_t> stripe_count_{0};
    uint64_t backup_growth_limit_stripes_ = 0;
    std::atomic<uint64_t> backup_growth_stripes_{0};
    std::mutex recovery_replacement_mutex_;
    std::atomic<uint64_t> recovery_replacement_limit_stripes_{0};
    std::atomic<uint64_t> recovery_replacement_stripes_{0};
    std::array<SizeClassPool, ::FarLib::allocator::RegionBinCount> pools_;
    // Group allocations use their own candidate pool: group stripes stay out
    // of the single-object pool so the two paths cannot hand out the same
    // slot offset.
    std::array<SizeClassPool, ::FarLib::allocator::RegionBinCount> group_pools_;
    // Direct reuse index: one atomic stripe-summary bitmap per (size class,
    // behavior) key. Stripe-local candidate bitmaps carry exact slot IDs;
    // summaries avoid scanning unrelated stripes while remaining lock-free
    // for readers.
    std::array<std::array<AtomicBitmap, kReuseBehaviorGroupCount>,
               ::FarLib::allocator::RegionBinCount>
        reuse_stripe_summaries_;
    // create_stripe() and the rebuild scan use this barrier to ensure that a
    // pre-failure endpoint selection cannot be published after the scan limit
    // is captured.
    mutable std::mutex stripes_mutex_;
    std::atomic<uint64_t> next_endpoint_{0};
    // Endpoint liveness lives in RemoteGlobalHeap and is shared with ordinary
    // flat allocation.  A dead endpoint never changes the addresses already
    // published by a stripe; it only filters future placement and reusable
    // group-pool candidates here.
    inline static thread_local LocalLeases local_leases_;

    struct LocalReuseCursor {
        const SmallObjectStripeManager *owner;
        // One traversal cursor is sufficient across keys. Keep TLS compact:
        // libfibre idle pthreads have a small stack shared with static TLS.
        size_t next_stripe = 0;
        size_t legacy_next_stripe = 0;

        LocalReuseCursor() : owner(nullptr) {}
    };
    inline static thread_local LocalReuseCursor local_reuse_cursor_;

    static size_t bin_from_size(size_t size) {
        size_t wsize = ::FarLib::allocator::wsize_from_size(size);
        size_t bin = ::FarLib::allocator::bin_from_wsize(wsize);
        assert(::FarLib::allocator::get_bin_size(bin) >= size);
        return bin;
    }

    uint64_t allocate_region_for_endpoint(size_t endpoint_idx) {
        return ::FarLib::allocator::remote::remote_global_heap
            .allocate_whole_region_on_endpoint(endpoint_idx);
    }

    void release_allocated_shards(
        const std::array<uint64_t, kShardCount> &shard_base) {
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            if (shard_base[shard] == kInvalidRemoteAddr) {
                continue;
            }
            bool ok = ::FarLib::allocator::remote::remote_global_heap
                          .release_whole_region(shard_base[shard]);
            ASSERT(ok);
        }
    }
    // Background workers are joined before init() is used as manager reset.
    // Exchange each published target exactly once, then return its whole
    // region to the remote heap; the old logical region is never released by
    // this helper.
    void release_published_background_targets() {
        for (auto &stripe_ptr : stripe_storage_) {
            if (stripe_ptr == nullptr) continue;
            for (uint8_t shard = 0; shard < kShardCount; shard++) {
                const uint64_t target = stripe_ptr->rebuilt_base[shard].exchange(
                    kInvalidRemoteAddr, std::memory_order_acq_rel);
                if (target == kInvalidRemoteAddr) continue;
                const bool ok =
                    ::FarLib::allocator::remote::remote_global_heap
                        .release_whole_region(target);
                ASSERT(ok);
            }
        }
    }

    bool stripe_has_space_locked(const Stripe &stripe) const {
        return stripe.dead_count != 0 || stripe.free_count != 0;
    }

    Stripe *stripe_by_id(uint64_t stripe_id) {
        if (stripe_id >= stripe_capacity_) return nullptr;
        return stripe_index_[static_cast<size_t>(stripe_id)].load(
            std::memory_order_acquire);
    }

    const Stripe *stripe_by_id(uint64_t stripe_id) const {
        if (stripe_id >= stripe_capacity_) return nullptr;
        return stripe_index_[static_cast<size_t>(stripe_id)].load(
            std::memory_order_acquire);
    }

    static uint64_t encode_binding(uint64_t stripe_id, uint8_t shard_idx,
                                   uint16_t bin) {
        assert(stripe_id <= kStripeIdMask);
        assert(shard_idx < kDataShards);
        assert(bin < ::FarLib::allocator::RegionBinCount);
        return (stripe_id & kStripeIdMask) |
               (static_cast<uint64_t>(shard_idx) << 48) |
               (static_cast<uint64_t>(bin) << 56);
    }

    static DecodedBinding decode_binding(uint64_t raw) {
        DecodedBinding binding;
        binding.stripe_id = raw & kStripeIdMask;
        binding.shard_idx = static_cast<uint8_t>((raw >> 48) & 0xff);
        binding.bin = static_cast<uint16_t>((raw >> 56) & 0xff);
        return binding;
    }

    // Address reverse mapping step 2: addr -> (stripe, data shard, bin).
    // shard region id == addr / shard_size; the per-region binding records to
    // which stripe/shard that region belongs.  Only data shards are bound, so a
    // parity shard address is not "owned" by the stripe allocator.
    bool binding_for_addr(uint64_t addr, DecodedBinding *binding) const {
        if (!enabled() || addr >= shard_table_.remote_capacity_bytes()) {
            return false;
        }
        uint64_t region_id = addr / shard_table_.shard_size();
        if (region_id >= shard_binding_count_) {
            return false;
        }
        uint64_t raw = shard_bindings_[static_cast<size_t>(region_id)].load(
            std::memory_order_acquire);
        if (raw == kInvalidBinding) {
            return false;
        }
        if (binding != nullptr) {
            *binding = decode_binding(raw);
        }
        return true;
    }

    uint64_t pop_candidate(SizeClassPool &pool) {
        std::lock_guard<std::mutex> lock(pool.mutex);
        if (pool.stripes_with_space.empty()) {
            return kInvalidStripeId;
        }
        uint64_t stripe_id = pool.stripes_with_space.back();
        pool.stripes_with_space.pop_back();
        return stripe_id;
    }

    void push_candidate(uint16_t bin, uint64_t stripe_id) {
        auto &pool = pools_[bin];
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.stripes_with_space.push_back(stripe_id);
    }

    bool mark_queued_if_has_space_locked(Stripe &stripe) {
        if (!stripe_has_space_locked(stripe) || stripe.in_pool) {
            return false;
        }
        stripe.in_pool = true;
        return true;
    }

    bool allocate_from_stripe_locked(Stripe &stripe, uint64_t *addr_out,
                                     bool *dead_source_out = nullptr) {
        auto try_source = [&](std::array<Bitmap, kDataShards> &bits,
                              bool dead_source) -> bool {
            for (uint8_t attempt = 0; attempt < kDataShards; attempt++) {
                uint8_t shard =
                    static_cast<uint8_t>((stripe.next_data_shard + attempt) %
                                         kDataShards);
                size_t slot = 0;
                if (!bits[shard].find_and_reset(&slot)) {
                    continue;
                }
                if (slot >= stripe.slots_per_shard) {
                    continue;
                }
                stripe.next_data_shard =
                    static_cast<uint8_t>((shard + 1) % kDataShards);
                if (dead_source) {
                    assert(stripe.dead_count > 0);
                    stripe.dead_count--;
                } else {
                    assert(stripe.free_count > 0);
                    stripe.free_count--;
                }
                stripe.live_count++;
                *addr_out = stripe.shard_base[shard] +
                            slot * static_cast<uint64_t>(stripe.slot_size);
                if (dead_source_out != nullptr) {
                    *dead_source_out = dead_source;
                }
                return true;
            }
            return false;
        };

        if (try_source(stripe.dead_bits, true)) return true;
        return try_source(stripe.free_bits, false);
    }


    // -------------------------- slot-group helpers --------------------------
    // kGroupBitsPerSlot bits per slot_id in the group state bitmap, so
    // slot_id -> bit pair [2*slot_id, 2*slot_id+1] of group_state_words.
    static constexpr uint8_t kGroupBitsPerSlot = 2;
    static constexpr uint64_t kGroupStateMask = 3;
    static constexpr uint32_t kInvalidEndpointIdx =
        std::numeric_limits<uint32_t>::max();

    static size_t group_state_bit(size_t slot) {
        return slot * kGroupBitsPerSlot;
    }

    static uint8_t group_state_locked(const Stripe &stripe, uint32_t slot) {
        if (stripe.group_state_words.empty()) return kSlotGroupFree;
        size_t bit = group_state_bit(slot);
        uint64_t word = stripe.group_state_words[bit / kBitsPerWord];
        return static_cast<uint8_t>((word >> (bit % kBitsPerWord)) &
                                    kGroupStateMask);
    }

    static void set_group_state_locked(Stripe &stripe, uint32_t slot,
                                       uint8_t state) {
        size_t bit = group_state_bit(slot);
        uint64_t &word = stripe.group_state_words[bit / kBitsPerWord];
        size_t shift = bit % kBitsPerWord;
        word = (word & ~(kGroupStateMask << shift)) |
               ((static_cast<uint64_t>(state) & kGroupStateMask) << shift);
    }

    static bool group_owns_slot_locked(const Stripe &stripe, uint32_t slot) {
        return group_state_locked(stripe, slot) != kSlotGroupFree;
    }

    static uint8_t group_pending_mask_locked(const Stripe &stripe,
                                             uint32_t slot) {
        if (slot >= stripe.group_pending_mask.size()) return 0;
        return static_cast<uint8_t>(stripe.group_pending_mask[slot] & 0x0fu);
    }

    static bool group_update_pending_locked(const Stripe &stripe,
                                            uint32_t slot) {
        return group_pending_mask_locked(stripe, slot) != 0;
    }

    // Snapshot only durable sealed groups with at least one live object.
    // `busy` also covers sealed groups whose initial six-way write has not
    // yet become durable; rebuilding such a codeword would race its writer.
    static bool background_live_groups_locked(
        const Stripe &stripe, uint64_t *live_groups, uint64_t *live_objects,
        std::vector<uint32_t> *live_slots, bool *busy) {
        if (live_groups == nullptr || live_objects == nullptr ||
            live_slots == nullptr || busy == nullptr) {
            return false;
        }
        *live_groups = 0;
        *live_objects = 0;
        live_slots->clear();
        *busy = false;
        if (stripe.group_state_words.empty()) {
            return stripe.live_count == 0;
        }
        if (stripe.group_pending_mask.size() < stripe.slots_per_shard ||
            stripe.group_durable.size() < stripe.slots_per_shard ||
            stripe.group_live_mask.size() < stripe.slots_per_shard ||
            stripe.group_live_objects.size() < stripe.slots_per_shard) {
            return false;
        }
        for (uint32_t slot = 0; slot < stripe.slots_per_shard; slot++) {
            const uint8_t state = group_state_locked(stripe, slot);
            if (state == kSlotGroupInProgress) {
                *busy = true;
                continue;
            }
            if (stripe.group_pending_mask[slot] != 0) {
                *busy = true;
            }
            if (state != kSlotGroupSealed) continue;
            if (stripe.group_durable[slot] == 0) {
                *busy = true;
                continue;
            }
            const uint8_t live =
                static_cast<uint8_t>(stripe.group_live_mask[slot] & 0x0fu);
            if (live == 0) continue;
            live_slots->push_back(slot);
            ++(*live_groups);
            *live_objects += stripe.group_live_objects[slot];
        }
        return true;
    }
    // Behavior-independent aggregate hint. Exact candidate slots live in the
    // per-behavior bitmaps, so this helper never walks group records.
    static bool stripe_has_reuse_candidate_locked(const Stripe &stripe) {
        if (!::FarLib::ec_benchmark_phase::enabled()) {
            if (stripe.group_state_words.empty()) return false;
            for (uint32_t slot = 0; slot < stripe.slots_per_shard; slot++) {
                if (group_state_locked(stripe, slot) != kSlotGroupSealed ||
                    slot >= stripe.group_durable.size() ||
                    stripe.group_durable[slot] == 0 ||
                    slot >= stripe.group_is_split.size() ||
                    stripe.group_is_split[slot] != 0 ||
                    slot >= stripe.group_live_mask.size()) {
                    continue;
                }
                const uint8_t live = static_cast<uint8_t>(
                    stripe.group_live_mask[slot] & 0x0fu);
                if (live != 0x0fu) return true;
            }
            return false;
        }
        for (uint8_t behavior = 0; behavior < kReuseBehaviorGroupCount;
             behavior++) {
            if (stripe.reuse_bits[behavior].any_set()) return true;
        }
        return false;
    }

    static uint64_t next_group_generation_locked(Stripe &stripe,
                                                 uint32_t slot) {
        if (slot >= stripe.group_generation.size()) return 0;
        uint64_t next = stripe.group_generation[slot] + 1;
        if (next == 0) next = 1;
        stripe.group_generation[slot] = next;
        return next;
    }

    static bool group_endpoints_alive_for_reuse(const Stripe &stripe) {
        auto &heap = ::FarLib::allocator::remote::remote_global_heap;
        if (!heap.endpoint_liveness_enabled()) return true;
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            const uint32_t endpoint = stripe.shard_endpoint[shard];
            if (endpoint == kInvalidEndpointIdx ||
                !heap.endpoint_is_alive(endpoint)) {
                return false;
            }
        }
        return true;
    }

    // Invariant 3: the six shards of a group must sit on six distinct
    // endpoints; unset endpoints (kInvalidEndpointIdx) also fail here.
    static bool endpoints_distinct(const Stripe &stripe) {
        for (uint8_t a = 0; a < kShardCount; a++) {
            for (uint8_t b = static_cast<uint8_t>(a + 1); b < kShardCount;
                 b++) {
                if (stripe.shard_endpoint[a] == stripe.shard_endpoint[b]) {
                    return false;
                }
            }
        }
        return true;
    }

    // Group metadata costs 2 bits + 1 byte per slot, so it is materialised
    // lazily and only for stripes that actually serve group allocations.
    bool ensure_group_metadata_locked(Stripe &stripe) const {
        if (!stripe.group_state_words.empty()) return true;
        if (stripe.slots_per_shard == 0) return false;
        size_t bits = static_cast<size_t>(stripe.slots_per_shard) *
                      kGroupBitsPerSlot;
        stripe.group_state_words.assign(
            (bits + kBitsPerWord - 1) / kBitsPerWord, 0);
        stripe.group_live_mask.assign(stripe.slots_per_shard, 0);
        stripe.group_pending_mask.assign(stripe.slots_per_shard, 0);
        stripe.group_durable.assign(stripe.slots_per_shard, 0);
        stripe.group_behavior_group.assign(stripe.slots_per_shard, 0);
        stripe.group_generation.assign(stripe.slots_per_shard, 0);
        stripe.next_group_slot = 0;
        stripe.group_dead_count = 0;
        stripe.group_live_objects.assign(stripe.slots_per_shard, 0);
        stripe.group_is_split.assign(stripe.slots_per_shard, 0);
        stripe.group_payload_sizes.assign(
            stripe.slots_per_shard,
            std::array<uint32_t, kDataShards>{});
        stripe.group_payload_unknown.assign(stripe.slots_per_shard, 0);
        if (::FarLib::ec_benchmark_phase::enabled()) {
            for (uint8_t behavior = 0; behavior < kReuseBehaviorGroupCount;
                 behavior++) {
                stripe.reuse_bits[behavior].init_empty(
                    stripe.slots_per_shard);
            }
        }
        // Published before the first group slot address can exist: the release
        // path uses it to skip the group lookup for stripes that never served a
        // group, and it is never cleared again.
        stripe.has_groups.store(true, std::memory_order_release);
        return true;
    }

    static uint64_t group_reserved_bytes(const Stripe &stripe) {
        return static_cast<uint64_t>(kShardCount) * stripe.slot_size;
    }

    static uint64_t group_parity_bytes(const Stripe &stripe) {
        return static_cast<uint64_t>(kParityShards) * stripe.slot_size;
    }

    static uint8_t group_live_slots_locked(const Stripe &stripe,
                                           uint32_t slot) {
        if (slot >= stripe.group_live_mask.size()) return 0;
        return static_cast<uint8_t>(
            __builtin_popcount(static_cast<unsigned>(
                stripe.group_live_mask[slot] & 0x0fu)));
    }

    static uint32_t group_unknown_count_locked(const Stripe &stripe,
                                               uint32_t slot) {
        if (slot >= stripe.group_payload_unknown.size() ||
            slot >= stripe.group_live_mask.size()) {
            return 0;
        }
        return static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(
            stripe.group_payload_unknown[slot] &
            stripe.group_live_mask[slot] & 0x0fu)));
    }

    static uint64_t group_payload_bytes_locked(const Stripe &stripe,
                                               uint32_t slot) {
        if (slot >= stripe.group_payload_sizes.size() ||
            slot >= stripe.group_live_mask.size()) {
            return 0;
        }
        const uint8_t mask = stripe.group_live_mask[slot] & 0x0fu;
        uint64_t bytes = 0;
        for (uint8_t i = 0; i < kDataShards; i++) {
            if ((mask & static_cast<uint8_t>(1u << i)) != 0) {
                bytes += stripe.group_payload_sizes[slot][i];
            }
        }
        return bytes;
    }

    // Account a fresh or dead-group-backed allocation.  The caller owns the
    // stripe mutex; no global atomic or lock is involved.
    static void account_group_allocate_locked(Stripe &stripe,
                                               bool from_dead) {
        if (from_dead) {
            assert(stripe.reusable_groups > 0);
            stripe.reusable_groups--;
        }
        stripe.occupied_group_bytes += group_reserved_bytes(stripe);
        stripe.in_progress_groups++;
    }

    static void account_group_seal_locked(Stripe &stripe, uint32_t slot,
                                          bool split) {
        const uint64_t reserved = group_reserved_bytes(stripe);
        const uint64_t parity = group_parity_bytes(stripe);
        const uint8_t live = split ? kDataShards
                                   : group_live_slots_locked(stripe, slot);
        const uint64_t payload = group_payload_bytes_locked(stripe, slot);
        const uint32_t unknown = group_unknown_count_locked(stripe, slot);
        assert(stripe.in_progress_groups > 0);
        stripe.in_progress_groups--;
        stripe.sealed_group_bytes += reserved;
        stripe.parity_bytes += parity;
        stripe.live_data_slot_bytes +=
            static_cast<uint64_t>(live) * stripe.slot_size;
        if (!split) {
            stripe.trapped_hole_bytes +=
                static_cast<uint64_t>(kDataShards - live) * stripe.slot_size;
            assert(live <= kDataShards);
            stripe.group_counts_by_live[live]++;
            stripe.sealed_small_groups++;
        }
        stripe.live_payload_bytes += payload;
        const uint64_t live_capacity =
            static_cast<uint64_t>(live) * stripe.slot_size;
        assert(live_capacity >= payload);
        stripe.padding_bytes += live_capacity - payload;
        stripe.unknown_payload_objects += unknown;
        if (split) {
            stripe.split_groups++;
        }
    }

    static void account_group_dead_locked(Stripe &stripe, uint32_t slot,
                                          bool split) {
        const uint64_t reserved = group_reserved_bytes(stripe);
        const uint64_t parity = group_parity_bytes(stripe);
        const uint8_t live = split ? kDataShards
                                   : group_live_slots_locked(stripe, slot);
        const uint64_t payload = group_payload_bytes_locked(stripe, slot);
        const uint32_t unknown = group_unknown_count_locked(stripe, slot);
        assert(stripe.occupied_group_bytes >= reserved);
        stripe.occupied_group_bytes -= reserved;
        const uint8_t state = group_state_locked(stripe, slot);
        if (state == kSlotGroupInProgress) {
            assert(stripe.in_progress_groups > 0);
            stripe.in_progress_groups--;
        } else if (state == kSlotGroupSealed) {
            assert(stripe.sealed_group_bytes >= reserved);
            assert(stripe.parity_bytes >= parity);
            assert(stripe.live_data_slot_bytes >=
                   static_cast<uint64_t>(live) * stripe.slot_size);
            stripe.sealed_group_bytes -= reserved;
            stripe.parity_bytes -= parity;
            stripe.live_data_slot_bytes -=
                static_cast<uint64_t>(live) * stripe.slot_size;
            const uint64_t holes = split
                                       ? 0
                                       : static_cast<uint64_t>(kDataShards - live) *
                                             stripe.slot_size;
            assert(stripe.trapped_hole_bytes >= holes);
            stripe.trapped_hole_bytes -= holes;
            assert(stripe.live_payload_bytes >= payload);
            stripe.live_payload_bytes -= payload;
            const uint64_t live_capacity =
                static_cast<uint64_t>(live) * stripe.slot_size;
            assert(live_capacity >= payload);
            assert(stripe.padding_bytes >= live_capacity - payload);
            stripe.padding_bytes -= live_capacity - payload;
            assert(stripe.unknown_payload_objects >= unknown);
            stripe.unknown_payload_objects -= unknown;
            if (split) {
                assert(stripe.split_groups > 0);
                stripe.split_groups--;
            } else {
                assert(live <= kDataShards);
                assert(stripe.group_counts_by_live[live] > 0);
                stripe.group_counts_by_live[live]--;
                assert(stripe.sealed_small_groups > 0);
                stripe.sealed_small_groups--;
            }
        } else {
            // The helper is only called for a live group transition.  Keep an
            // assertion here so a future state-machine change cannot silently
            // corrupt the counters.
            assert(false && "account_group_dead_locked on non-live group");
        }
        stripe.reusable_groups++;
    }

    static void account_group_object_release_locked(Stripe &stripe,
                                                     uint32_t slot,
                                                     uint8_t shard) {
        assert(slot < stripe.group_payload_sizes.size());
        assert(shard < kDataShards);
        const uint32_t payload = stripe.group_payload_sizes[slot][shard];
        assert(stripe.live_data_slot_bytes >= stripe.slot_size);
        assert(stripe.live_payload_bytes >= payload);
        stripe.live_data_slot_bytes -= stripe.slot_size;
        stripe.live_payload_bytes -= payload;
        assert(stripe.padding_bytes >= stripe.slot_size - payload);
        stripe.padding_bytes -= stripe.slot_size - payload;
        stripe.trapped_hole_bytes += stripe.slot_size;
        const uint8_t bit = static_cast<uint8_t>(1u << shard);
        if ((stripe.group_payload_unknown[slot] & bit) != 0) {
            assert(stripe.unknown_payload_objects > 0);
            stripe.unknown_payload_objects--;
            stripe.group_payload_unknown[slot] = static_cast<uint8_t>(
                stripe.group_payload_unknown[slot] & ~bit);
        }
    }

    // Fill one previously retired data slot while keeping the six-segment
    // reservation and sealed-group parity accounting intact.  The caller has
    // already validated the pending generation and holds stripe.mutex.
    static void account_group_slot_reuse_commit_locked(
        Stripe &stripe, uint32_t slot, uint8_t shard, uint32_t payload) {
        assert(shard < kDataShards);
        assert(slot < stripe.group_live_mask.size());
        const uint8_t old_live = group_live_slots_locked(stripe, slot);
        assert(old_live < kDataShards);
        assert((stripe.group_live_mask[slot] &
                static_cast<uint8_t>(1u << shard)) == 0);
        assert(payload > 0 && payload <= stripe.slot_size);
        assert(stripe.trapped_hole_bytes >= stripe.slot_size);
        stripe.trapped_hole_bytes -= stripe.slot_size;
        stripe.live_data_slot_bytes += stripe.slot_size;
        stripe.live_payload_bytes += payload;
        stripe.padding_bytes += stripe.slot_size - payload;
        if (old_live <= kDataShards) {
            assert(stripe.group_counts_by_live[old_live] > 0);
            stripe.group_counts_by_live[old_live]--;
            stripe.group_counts_by_live[old_live + 1]++;
        }
    }

    // Invariants 1 and 2: a candidate offset is only taken when all four data
    // shards have that offset free *at the same time*; the four free bits are
    // then cleared together inside this single critical section.  A dead group
    // already has its four slots reserved, so it is reused as a whole.
    bool allocate_group_from_stripe_locked(Stripe &stripe,
                                           uint32_t *slot_id_out,
                                           bool *from_dead_out = nullptr) {
        // Creation publishes the empty stripe before its first group is
        // reserved. Recheck under the stripe lock: a fault can land between
        // those events, after the rebuild scan already skipped the empty stripe.
        if (::FarLib::get_config().ft_background_rebuild &&
            !stripe_eligible_for_new_group(stripe)) return false;
        if (!ensure_group_metadata_locked(stripe)) return false;
        if (stripe.slots_per_shard == 0) return false;

        for (uint32_t n = 0; n < stripe.slots_per_shard; n++) {
            uint64_t slot64 =
                static_cast<uint64_t>(stripe.next_group_slot) + n;
            if (slot64 >= stripe.slots_per_shard) {
                slot64 -= stripe.slots_per_shard;
            }
            uint32_t slot = static_cast<uint32_t>(slot64);
            uint8_t state = group_state_locked(stripe, slot);
            bool from_dead = false;
            if (state == kSlotGroupDead) {
                from_dead = true;
            } else if (state == kSlotGroupFree) {
                bool all_free = true;
                for (uint8_t shard = 0; shard < kDataShards; shard++) {
                    if (!stripe.free_bits[shard].test(slot)) {
                        all_free = false;
                        break;
                    }
                }
                if (!all_free) continue;
                for (uint8_t shard = 0; shard < kDataShards; shard++) {
                    stripe.free_bits[shard].reset(slot);
                }
                assert(stripe.free_count >= kDataShards);
                stripe.free_count -= kDataShards;
            } else {
                continue;
            }

            stripe.next_group_slot = slot + 1;
            if (stripe.next_group_slot >= stripe.slots_per_shard) {
                stripe.next_group_slot = 0;
            }
            if (from_dead) {
                assert(stripe.group_dead_count > 0);
                stripe.group_dead_count--;
            }
            set_group_state_locked(stripe, slot, kSlotGroupInProgress);
            stripe.group_live_mask[slot] = 0;
            stripe.group_pending_mask[slot] = 0;
            stripe.group_durable[slot] = 0;
            stripe.group_behavior_group[slot] = 0;
            (void)next_group_generation_locked(stripe, slot);
            stripe.group_live_objects[slot] = 0;
            stripe.group_is_split[slot] = 0;
            stripe.group_payload_sizes[slot].fill(0);
            stripe.group_payload_unknown[slot] = 0;
            account_group_allocate_locked(stripe, from_dead);
            stripe.live_count += kDataShards;
            *slot_id_out = slot;
            if (from_dead_out != nullptr) *from_dead_out = from_dead;
            return true;
        }
        return false;
    }

    bool group_has_space_locked(const Stripe &stripe) const {
        if (stripe.group_dead_count != 0) return true;
        // Necessary but not sufficient: a fresh group also needs the four free
        // bits to sit at one common offset.
        return stripe.free_count >= kDataShards;
    }

    bool mark_group_queued_if_has_space_locked(Stripe &stripe) {
        if (!group_has_space_locked(stripe) || stripe.in_group_pool) {
            return false;
        }
        stripe.in_group_pool = true;
        return true;
    }

    void push_group_candidate(uint16_t bin, uint64_t stripe_id) {
        auto &pool = group_pools_[bin];
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.stripes_with_space.push_back(stripe_id);
    }

    // Fills all six segments of a handle in one shot; never returns a partial
    // handle.  Also re-checks the six-distinct-endpoint invariant.
    bool fill_group_handle(const Stripe &stripe, uint16_t bin, uint32_t slot_id,
                           SlotGroupHandle *group_out) const {
        if (group_out == nullptr || stripe.slot_size == 0) return false;
        if (slot_id >= stripe.slots_per_shard) return false;
        if (!endpoints_distinct(stripe)) return false;
        const uint64_t slot_offset =
            static_cast<uint64_t>(slot_id) * stripe.slot_size;
        SlotGroupHandle handle;
        handle.id.stripe_id = stripe.stripe_id;
        handle.id.slot_id = slot_id;
        handle.bin = bin;
        handle.slot_size = stripe.slot_size;
        handle.slots_per_shard = stripe.slots_per_shard;
        handle.slot_offset = slot_offset;
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            uint64_t base = stripe.shard_base[shard];
            if (base == kInvalidRemoteAddr) return false;
            SlotGroupSegment &segment = handle.segments[shard];
            segment.shard_idx = shard;
            segment.endpoint_idx = stripe.shard_endpoint[shard];
            segment.offset = slot_offset;
            segment.slot_size = stripe.slot_size;
            segment.addr = base + slot_offset;
        }
        *group_out = handle;
        return true;
    }

    // Validate the immutable physical handle carried by a slot-reuse
    // callback.  The generation check protects allocator metadata from ABA;
    // this check additionally prevents a caller that retained or modified a
    // stale handle from committing metadata for a different physical layout.
    static bool same_group_handle(const SlotGroupHandle &a,
                                  const SlotGroupHandle &b) {
        if (a.id.stripe_id != b.id.stripe_id ||
            a.id.slot_id != b.id.slot_id || a.bin != b.bin ||
            a.slot_size != b.slot_size ||
            a.slots_per_shard != b.slots_per_shard ||
            a.slot_offset != b.slot_offset) {
            return false;
        }
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            const SlotGroupSegment &as = a.segments[shard];
            const SlotGroupSegment &bs = b.segments[shard];
            if (as.shard_idx != bs.shard_idx ||
                as.endpoint_idx != bs.endpoint_idx || as.offset != bs.offset ||
                as.addr != bs.addr || as.slot_size != bs.slot_size) {
                return false;
            }
        }
        return true;
    }

    void clear_local_lease(uint16_t bin) {
        auto &lease = local_leases_.bins[bin];
        if (lease.owner == this) {
            lease.owner = nullptr;
            lease.stripe_id = kInvalidStripeId;
        }
    }

    void set_local_lease(uint16_t bin, uint64_t stripe_id) {
        auto &lease = local_leases_.bins[bin];
        lease.owner = this;
        lease.stripe_id = stripe_id;
    }

    bool endpoint_alive_for_policy(size_t endpoint_idx) const {
        return ::FarLib::allocator::remote::remote_global_heap
            .endpoint_is_eligible_for_allocation(endpoint_idx);
    }

    bool standby_active_for_policy() const {
        return ::FarLib::allocator::remote::remote_global_heap
            .standby_active_for_allocation();
    }

    size_t count_endpoints_for_new_allocations() const {
        size_t count = 0;
        const size_t endpoint_count =
            ::FarLib::allocator::remote::remote_global_heap.endpoint_count();
        for (size_t endpoint = 0; endpoint < endpoint_count; endpoint++) {
            if (endpoint_alive_for_policy(endpoint)) count++;
        }
        return count;
    }

    // Select exactly six distinct endpoints for one new stripe.  Once the
    // standby is activated it is deliberately included in every newly chosen
    // six-endpoint group, even when server_count is larger than seven.
    std::vector<size_t> select_endpoints_for_new_stripe() {
        std::vector<size_t> regular;
        const bool standby_active = standby_active_for_policy();
        const int standby_endpoint =
            ::FarLib::allocator::remote::remote_global_heap.standby_endpoint();
        const size_t endpoint_count =
            ::FarLib::allocator::remote::remote_global_heap.endpoint_count();
        for (size_t endpoint = 0; endpoint < endpoint_count; endpoint++) {
            if (!endpoint_alive_for_policy(endpoint)) continue;
            if (standby_active &&
                endpoint == static_cast<size_t>(standby_endpoint)) {
                continue;
            }
            regular.push_back(endpoint);
        }

        std::vector<size_t> selected;
        if (standby_active) {
            if (regular.size() + 1 < kShardCount) return selected;
            selected.push_back(static_cast<size_t>(standby_endpoint));
        } else if (regular.size() < kShardCount) {
            return selected;
        }

        if (regular.empty()) return selected;
        const size_t start = static_cast<size_t>(
            next_endpoint_.fetch_add(1, std::memory_order_relaxed) %
            regular.size());
        while (selected.size() < kShardCount) {
            selected.push_back(regular[(start + selected.size() -
                                        (standby_active ? 1 : 0)) %
                                       regular.size()]);
        }
        return selected;
    }

    bool stripe_eligible_for_new_group(const Stripe &stripe) const {
        const bool standby_active = standby_active_for_policy();
        const int standby_endpoint =
            ::FarLib::allocator::remote::remote_global_heap.standby_endpoint();
        bool has_standby = false;
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            const uint32_t endpoint = stripe.shard_endpoint[shard];
            if (!endpoint_alive_for_policy(endpoint)) return false;
            if (standby_active &&
                endpoint == static_cast<uint32_t>(standby_endpoint)) {
                has_standby = true;
            }
        }
        // Background repair freezes only affected old stripes (rejected by
        // the liveness loop above). Healthy old stripes must remain reusable;
        // otherwise normal sources can starve once bounded growth is spent.
        // Fresh stripe selection still includes the standby. Retain the
        // legacy policy when background reconstruction is disabled.
        return !standby_active || has_standby ||
               ::FarLib::get_config().ft_background_rebuild;
    }

    static bool reuse_behavior_indexable(uint32_t behavior_group) {
        return behavior_group < kReuseBehaviorGroupCount;
    }

    AtomicBitmap &reuse_summary(uint16_t bin, uint32_t behavior_group) {
        assert(bin < ::FarLib::allocator::RegionBinCount);
        assert(reuse_behavior_indexable(behavior_group));
        return reuse_stripe_summaries_[bin][behavior_group];
    }

    const AtomicBitmap &reuse_summary(uint16_t bin,
                                      uint32_t behavior_group) const {
        assert(bin < ::FarLib::allocator::RegionBinCount);
        assert(reuse_behavior_indexable(behavior_group));
        return reuse_stripe_summaries_[bin][behavior_group];
    }

    static void refresh_reuse_hint_locked(Stripe &stripe) {
        bool any = false;
        for (uint8_t behavior = 0; behavior < kReuseBehaviorGroupCount;
             behavior++) {
            if (stripe.reuse_bits[behavior].any_set()) {
                any = true;
                break;
            }
        }
        stripe.reuse_candidate.store(any, std::memory_order_release);
    }

    void clear_reuse_summary_if_empty_locked(Stripe &stripe,
                                             uint32_t behavior_group) {
        if (!reuse_behavior_indexable(behavior_group)) return;
        if (!stripe.reuse_bits[behavior_group].any_set()) {
            reuse_summary(stripe.bin, behavior_group).reset(
                static_cast<size_t>(stripe.stripe_id));
        }
    }

    // Publish or clear one exact candidate bit. Caller holds stripe.mutex.
    // The summary is deliberately conservative: an occasional stale set bit
    // is revalidated and cleared by begin_slot_reuse_indexed().
    void update_reuse_index_locked(Stripe &stripe, uint32_t slot) {
        if (!::FarLib::ec_benchmark_phase::enabled()) return;
        if (slot >= stripe.slots_per_shard ||
            stripe.reuse_bits[0].words.empty()) {
            return;
        }
        uint32_t behavior = 0;
        bool eligible = false;
        if (group_state_locked(stripe, slot) == kSlotGroupSealed &&
            slot < stripe.group_durable.size() &&
            stripe.group_durable[slot] != 0 &&
            slot < stripe.group_is_split.size() &&
            stripe.group_is_split[slot] == 0 &&
            slot < stripe.group_live_mask.size() &&
            (stripe.group_live_mask[slot] & 0x0fu) != 0x0fu &&
            !group_update_pending_locked(stripe, slot) &&
            slot < stripe.group_behavior_group.size()) {
            behavior = stripe.group_behavior_group[slot];
            eligible = reuse_behavior_indexable(behavior) &&
                       endpoints_distinct(stripe) &&
                       group_endpoints_alive_for_reuse(stripe) &&
                       stripe_eligible_for_new_group(stripe);
        }

        if (eligible) {
            stripe.reuse_bits[behavior].set(slot);
            reuse_summary(stripe.bin, behavior).set(
                static_cast<size_t>(stripe.stripe_id));
        } else {
            // Behavior is immutable for a sealed group. For stale metadata,
            // clear every key defensively; this path is transition-only.
            for (uint8_t key = 0; key < kReuseBehaviorGroupCount; key++) {
                stripe.reuse_bits[key].reset(slot);
                clear_reuse_summary_if_empty_locked(stripe, key);
            }
        }
        refresh_reuse_hint_locked(stripe);
    }

    void clear_reuse_index_locked(Stripe &stripe, uint32_t slot) {
        if (!::FarLib::ec_benchmark_phase::enabled()) return;
        if (slot >= stripe.slots_per_shard ||
            stripe.reuse_bits[0].words.empty()) {
            return;
        }
        for (uint8_t key = 0; key < kReuseBehaviorGroupCount; key++) {
            stripe.reuse_bits[key].reset(slot);
            clear_reuse_summary_if_empty_locked(stripe, key);
        }
        refresh_reuse_hint_locked(stripe);
    }

    // Normal indexed lookup knows the immutable behavior key of the sealed
    // group.  Remove only that exact candidate bit; the aggregate hint is
    // deliberately conservative and is refreshed by batch operations (or by
    // full cleanup) rather than scanning all six keys per object.
    void clear_reuse_key_locked(Stripe &stripe, uint32_t slot,
                                uint32_t behavior_group) {
        if (!::FarLib::ec_benchmark_phase::enabled() ||
            !reuse_behavior_indexable(behavior_group) ||
            slot >= stripe.slots_per_shard ||
            stripe.reuse_bits[behavior_group].words.empty()) {
            return;
        }
        stripe.reuse_bits[behavior_group].reset(slot);
        clear_reuse_summary_if_empty_locked(stripe, behavior_group);
    }

    void clear_all_reuse_index_locked(Stripe &stripe) {
        if (!::FarLib::ec_benchmark_phase::enabled()) return;
        if (stripe.reuse_bits[0].words.empty()) return;
        for (uint8_t key = 0; key < kReuseBehaviorGroupCount; key++) {
            for (uint64_t &word : stripe.reuse_bits[key].words) word = 0;
            reuse_summary(stripe.bin, key).reset(
                static_cast<size_t>(stripe.stripe_id));
        }
        refresh_reuse_hint_locked(stripe);
    }

    bool allocate_from_local_lease(uint16_t bin, uint64_t *addr_out,
                                   uint32_t *accounted_size_out,
                                   bool *from_dead_out) {
        auto &lease = local_leases_.bins[bin];
        if (lease.owner != this || lease.stripe_id == kInvalidStripeId) {
            return false;
        }

        Stripe *stripe = stripe_by_id(lease.stripe_id);
        if (stripe == nullptr) {
            clear_local_lease(bin);
            return false;
        }

        uint64_t addr = kInvalidRemoteAddr;
        bool from_dead = false;
        bool has_space = false;
        bool ok = false;
        uint32_t accounted_size = 0;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (stripe->bin == bin && stripe_eligible_for_new_group(*stripe)) {
                ok = allocate_from_stripe_locked(*stripe, &addr, &from_dead);
                if (ok) {
                    accounted_size = stripe->slot_size;
                    has_space = stripe_has_space_locked(*stripe);
                }
            }
        }

        if (!ok) {
            clear_local_lease(bin);
            return false;
        }

        if (!has_space) {
            clear_local_lease(bin);
        }
        *addr_out = addr;
        *accounted_size_out = accounted_size;
        *from_dead_out = from_dead;
        return true;
    }

    // require_distinct_endpoints: a slot group needs all six shards on six
    // distinct endpoints, so group allocations create stripes with this set
    // (and bail out instead of building a stripe that cannot host a group).
    // One failed physical group stripe may fund at most one replacement.
    // Credits are discovered only on a fault-enabled allocation miss and
    // consumed only for fresh physical stripes, never for an object/RMW.
    bool reserve_recovery_replacement_stripe() {
        if (!recovery_replacement_enabled()) return false;
        auto reserve = [&] {
            auto used = recovery_replacement_stripes_.load(std::memory_order_relaxed);
            do {
                if (used >= recovery_replacement_limit_stripes_.load(std::memory_order_acquire))
                    return false;
            } while (!recovery_replacement_stripes_.compare_exchange_weak(
                used, used + 1, std::memory_order_relaxed));
            return true;
        };
        if (reserve()) return true;
        {
            std::lock_guard<std::mutex> lock(recovery_replacement_mutex_);
            const auto count = stripe_count_.load(std::memory_order_acquire);
            uint64_t credits = 0;
            for (uint64_t i = 0; i < count; ++i) {
                Stripe *s = stripe_by_id(i);
                if (s && s->group_stripe && !s->recovery_replacement_credited &&
                    !group_endpoints_alive_for_reuse(*s)) {
                    s->recovery_replacement_credited = true;
                    ++credits;
                }
            }
            recovery_replacement_limit_stripes_.fetch_add(credits, std::memory_order_release);
        }
        return reserve();
    }

    Stripe *create_stripe(size_t bin,
                          bool require_distinct_endpoints = false,
                          bool backup_growth = false,
                          bool recovery_replacement = false) {
        if (backup_growth && recovery_replacement) return nullptr;
        const auto &config = ::FarLib::get_config();
        std::array<size_t, kShardCount> endpoints{};
        const bool endpoint_liveness_enabled =
            ::FarLib::allocator::remote::remote_global_heap
                .endpoint_liveness_enabled();
        if (!endpoint_liveness_enabled) {
            // Preserve the pre-standby placement exactly for sponge and for
            // all legacy/non-ec_batch callers: the original modulo walk was
            // allowed to repeat endpoints when the caller did not require a
            // six-way group.
            if (require_distinct_endpoints &&
                (config.server_count < static_cast<int>(kShardCount) ||
                 config.server_count <= 0)) {
                return nullptr;
            }
            const uint64_t start =
                next_endpoint_.fetch_add(1, std::memory_order_relaxed);
            for (uint8_t shard = 0; shard < kShardCount; shard++) {
                endpoints[shard] = static_cast<size_t>(
                    (start + shard) % static_cast<size_t>(config.server_count));
            }
        } else {
            const auto selected = select_endpoints_for_new_stripe();
            if (selected.size() < kShardCount) return nullptr;
            for (uint8_t shard = 0; shard < kShardCount; shard++) {
                endpoints[shard] = selected[shard];
            }
        }
        std::array<uint64_t, kShardCount> shard_base;
        std::array<uint32_t, kShardCount> shard_endpoint;
        shard_base.fill(kInvalidRemoteAddr);
        shard_endpoint.fill(kInvalidEndpointIdx);

        // Charge physical capacity once per new stripe, never per RMW object.
        if (recovery_replacement && !reserve_recovery_replacement_stripe())
            return nullptr;
        if (backup_growth) {
            auto used = backup_growth_stripes_.load(std::memory_order_relaxed);
            do {
                if (used >= backup_growth_limit_stripes_) return nullptr;
            } while (!backup_growth_stripes_.compare_exchange_weak(
                used, used + 1, std::memory_order_relaxed));
        }
        auto refund_growth = [&] {
            if (backup_growth)
                backup_growth_stripes_.fetch_sub(1, std::memory_order_relaxed);
            if (recovery_replacement)
                recovery_replacement_stripes_.fetch_sub(1, std::memory_order_relaxed);
        };

        // Shards of one stripe live on six distinct live memory servers.  The
        // endpoint selector excludes a configured standby until failure and
        // includes it in every new post-failure stripe.
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            size_t endpoint = endpoints[shard];
            uint64_t base = allocate_region_for_endpoint(endpoint);
            if (base == kInvalidRemoteAddr) {
                release_allocated_shards(shard_base);
                refund_growth();
                return nullptr;
            }
            shard_base[shard] = base;
            shard_endpoint[shard] = static_cast<uint32_t>(endpoint);
        }

        auto stripe = std::make_unique<Stripe>();
        stripe->stripe_id = 0;
        stripe->group_stripe = require_distinct_endpoints;
        stripe->bin = static_cast<uint16_t>(bin);
        stripe->slot_size =
            static_cast<uint32_t>(::FarLib::allocator::get_bin_size(bin));
        stripe->slots_per_shard =
            static_cast<uint32_t>(shard_table_.shard_size() / stripe->slot_size);
        stripe->live_count = 0;
        stripe->dead_count = 0;
        stripe->free_count = stripe->slots_per_shard * kDataShards;
        stripe->next_data_shard = 0;
        stripe->shard_base = shard_base;
        stripe->shard_endpoint = shard_endpoint;
        for (uint8_t shard = 0; shard < kDataShards; shard++) {
            stripe->free_bits[shard].init_full(stripe->slots_per_shard);
            stripe->dead_bits[shard].init_empty(stripe->slots_per_shard);
        }

       Stripe *stripe_ptr = nullptr;
       uint64_t stripe_id = kInvalidStripeId;
       std::lock_guard<std::mutex> lock(stripes_mutex_);
        // Endpoint selection happened before physical allocation, so repeat
        // the liveness/policy check at this publication barrier.  In
        // particular, a selection made before failure must not be published
        // after the standby becomes active.
        if (endpoint_liveness_enabled && ::FarLib::get_config().ft_background_rebuild) {
            bool has_standby = false;
            const bool standby_active = standby_active_for_policy();
            const int standby_endpoint =
                ::FarLib::allocator::remote::remote_global_heap
                    .standby_endpoint();
            for (uint8_t a = 0; a < kShardCount; a++) {
                if (!endpoint_alive_for_policy(endpoints[a])) {
                    release_allocated_shards(shard_base);
                    refund_growth();
                    return nullptr;
                }
                if (standby_active &&
                    endpoints[a] == static_cast<size_t>(standby_endpoint)) {
                    has_standby = true;
                }
                if (require_distinct_endpoints) {
                    for (uint8_t b = 0; b < a; b++) {
                        if (endpoints[a] == endpoints[b]) {
                            release_allocated_shards(shard_base);
                            refund_growth();
                            return nullptr;
                        }
                    }
                }
            }
            if (standby_active && !has_standby) {
                release_allocated_shards(shard_base);
                refund_growth();
                return nullptr;
            }
        }

       stripe_id = stripe_count_.load(std::memory_order_relaxed);
        if (stripe_id >= stripe_capacity_) {
            release_allocated_shards(shard_base);
            refund_growth();
            return nullptr;
        }
        stripe->stripe_id = stripe_id;
        stripe_ptr = stripe.get();
        stripe_storage_.push_back(std::move(stripe));
        stripe_index_[static_cast<size_t>(stripe_id)].store(
            stripe_ptr, std::memory_order_release);
        // Only the data shards are registered for the address reverse mapping;
        // parity shards are reachable through the stripe object.
        for (uint8_t shard = 0; shard < kDataShards; shard++) {
            uint64_t region_id = shard_base[shard] / shard_table_.shard_size();
            assert(region_id < shard_binding_count_);
            shard_bindings_[static_cast<size_t>(region_id)].store(
                encode_binding(stripe_id, shard, static_cast<uint16_t>(bin)),
                std::memory_order_release);
        }
        stripe_count_.store(stripe_id + 1, std::memory_order_release);
        return stripe_ptr;
    }

public:
    SmallObjectStripeManager() = default;

    // remote_capacity_bytes: total remote capacity (server_count *
    // server_buffer_size); shard_size: one region, i.e. SmallObjectStripeShardSize.
   void init(size_t remote_capacity_bytes, size_t shard_size) {
        // The owning recovery worker is joined before manager reset.  Do not
        // release targets from any worker path; this is the sole owner-side
        // cleanup point and exchanges each target before storage is cleared.
        release_published_background_targets();
        shard_table_.init(remote_capacity_bytes, shard_size);
        shard_binding_count_ = shard_table_.shard_region_count();
        shard_bindings_.reset(new std::atomic<uint64_t>[shard_binding_count_]);
        for (size_t i = 0; i < shard_binding_count_; i++) {
            shard_bindings_[i].store(kInvalidBinding, std::memory_order_relaxed);
        }
        stripe_capacity_ = shard_binding_count_ / kShardCount;
        stripe_index_.reset(new std::atomic<Stripe *>[stripe_capacity_]);
        for (size_t i = 0; i < stripe_capacity_; i++) {
            stripe_index_[i].store(nullptr, std::memory_order_relaxed);
        }
        if (::FarLib::ec_benchmark_phase::enabled()) {
            for (size_t bin = 0; bin < ::FarLib::allocator::RegionBinCount;
                 bin++) {
                for (uint8_t behavior = 0;
                     behavior < kReuseBehaviorGroupCount; behavior++) {
                    reuse_stripe_summaries_[bin][behavior].init(
                        stripe_capacity_);
                }
            }
        }
        stripe_storage_.clear();
        stripe_storage_.reserve(stripe_capacity_);
        stripe_count_.store(0, std::memory_order_relaxed);
        backup_growth_limit_stripes_ = 0;
        backup_growth_stripes_.store(0, std::memory_order_relaxed);
        recovery_replacement_limit_stripes_.store(0, std::memory_order_relaxed);
        recovery_replacement_stripes_.store(0, std::memory_order_relaxed);
        next_endpoint_.store(0, std::memory_order_relaxed);
    }

    void configure_backup_growth(uint64_t data_bytes) {
        const uint64_t quantum = kDataShards * shard_table_.shard_size();
        backup_growth_limit_stripes_ =
            data_bytes / quantum + uint64_t(data_bytes % quantum != 0);
    }

    bool backup_growth_enabled() const {
        return backup_growth_limit_stripes_ != 0;
    }

    uint64_t backup_growth_data_limit_bytes() const {
        return backup_growth_limit_stripes_ * kDataShards *
               shard_table_.shard_size();
    }

    uint64_t backup_growth_data_bytes() const {
        return backup_growth_stripes_.load(std::memory_order_relaxed) *
               kDataShards * shard_table_.shard_size();
    }

    bool recovery_replacement_enabled() const {
        const auto &config = ::FarLib::get_config();
        return config.ft_rmw_read_failure_fallback &&
               config.ft_incremental_one_sided && config.server_count > 0 &&
               live_endpoint_count() < static_cast<size_t>(config.server_count);
    }
    uint64_t recovery_replacement_data_bytes() const {
        return recovery_replacement_stripes_.load(std::memory_order_relaxed) *
               kDataShards * shard_table_.shard_size();
    }
    uint64_t recovery_replacement_data_limit_bytes() const {
        return recovery_replacement_limit_stripes_.load(std::memory_order_acquire) *
               kDataShards * shard_table_.shard_size();
    }

    bool enabled() const { return shard_table_.enabled(); }

    ::FarLib::runtime_metadata::Snapshot metadata_usage() {
        using Snapshot = ::FarLib::runtime_metadata::Snapshot;
        Snapshot snapshot{};

        // Inline manager storage is counted once, including compiler padding.
        // The enclosing cache does not add sizeof(this manager) a second time.
        snapshot.region_bytes += sizeof(shard_table_);
        std::lock_guard<std::mutex> stripes_lock(stripes_mutex_);
        snapshot.mapping_bytes += sizeof(*this) - sizeof(shard_table_);
        snapshot.mapping_bytes +=
            static_cast<uint64_t>(stripe_storage_.capacity()) *
            sizeof(std::unique_ptr<Stripe>);
        if (stripe_index_ != nullptr) {
            snapshot.mapping_bytes +=
                static_cast<uint64_t>(stripe_capacity_) *
                sizeof(std::atomic<Stripe *>);
        }
        if (shard_bindings_ != nullptr) {
            snapshot.mapping_bytes +=
                static_cast<uint64_t>(shard_binding_count_) *
                sizeof(std::atomic<uint64_t>);
        }

        // Pool objects include their mutex/vector headers in sizeof(manager);
        // only the vector backing stores are added below.
        const auto add_pool_backing = [&](auto &pools) {
            for (auto &pool : pools) {
                std::lock_guard<std::mutex> lock(pool.mutex);
                snapshot.mapping_bytes +=
                    static_cast<uint64_t>(pool.stripes_with_space.capacity()) *
                    sizeof(uint64_t);
            }
        };
        add_pool_backing(pools_);
        add_pool_backing(group_pools_);

        // AtomicBitmap contains its unique_ptr/size fields in the manager.
        // Count only the allocated words as backing storage.
        for (const auto &by_behavior : reuse_stripe_summaries_) {
            for (const auto &summary : by_behavior) {
                snapshot.mapping_bytes +=
                    static_cast<uint64_t>(summary.word_count) *
                    sizeof(std::atomic<uint64_t>);
            }
        }

        const auto add_bitmap_backing = [](const Bitmap &bitmap) {
            return static_cast<uint64_t>(bitmap.words.capacity()) *
                   sizeof(uint64_t);
        };
        for (const auto &holder : stripe_storage_) {
            if (holder == nullptr) continue;
            const Stripe &stripe = *holder;
            std::lock_guard<std::mutex> stripe_lock(stripe.mutex);
            const uint64_t stripe_measurement_aux_bytes =
                sizeof(stripe.occupied_group_bytes) +
                sizeof(stripe.sealed_group_bytes) +
                sizeof(stripe.live_payload_bytes) +
                sizeof(stripe.live_data_slot_bytes) +
                sizeof(stripe.parity_bytes) +
                sizeof(stripe.trapped_hole_bytes) +
                sizeof(stripe.padding_bytes) +
                sizeof(stripe.group_counts_by_live) +
                sizeof(stripe.split_groups) +
                sizeof(stripe.in_progress_groups) +
                sizeof(stripe.reusable_groups) +
                sizeof(stripe.unknown_payload_objects) +
                sizeof(stripe.sealed_small_groups) +
                sizeof(stripe.group_payload_sizes) +
                sizeof(stripe.group_payload_unknown);
            ++snapshot.stripes;
            snapshot.stripe_bytes +=
                sizeof(Stripe) - stripe_measurement_aux_bytes;
            snapshot.measurement_aux_bytes += stripe_measurement_aux_bytes;

            for (const auto &bitmap : stripe.free_bits)
                snapshot.stripe_bytes += add_bitmap_backing(bitmap);
            for (const auto &bitmap : stripe.dead_bits)
                snapshot.stripe_bytes += add_bitmap_backing(bitmap);
            for (const auto &bitmap : stripe.reuse_bits)
                snapshot.group_bytes += add_bitmap_backing(bitmap);

            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_state_words.capacity()) *
                sizeof(uint64_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_live_mask.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_pending_mask.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_durable.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_behavior_group.capacity()) *
                sizeof(uint32_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_generation.capacity()) *
                sizeof(uint64_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_live_objects.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_is_split.capacity()) *
                sizeof(uint8_t);

            // Payload-size/unknown tables are useful for space measurement but
            // are not part of the five metadata categories.
            snapshot.measurement_aux_bytes +=
                static_cast<uint64_t>(stripe.group_payload_sizes.capacity()) *
                sizeof(std::array<uint32_t, kDataShards>);
            snapshot.measurement_aux_bytes +=
                static_cast<uint64_t>(stripe.group_payload_unknown.capacity()) *
                sizeof(uint8_t);
            snapshot.group_slots +=
                static_cast<uint64_t>(stripe.group_live_mask.capacity());
        }
        return snapshot;
    }

    const SmallObjectStripeShardTable &shard_table() const {
        return shard_table_;
    }

    // Mark one connected endpoint dead after a failed RDMA operation.  This
    // hook is intentionally independent of stripe state: existing groups keep
    // their original physical addresses for degraded recovery, while future
    // allocation filters the endpoint and (when configured) activates the
    // standby.  The transition is idempotent; false means invalid or already
    // dead, and no allocator state is changed in either case.
   bool mark_endpoint_dead(size_t endpoint_idx) {
       return ::FarLib::allocator::remote::remote_global_heap
           .mark_endpoint_dead(endpoint_idx);
   }
    // Capture an exclusive upper bound for a background scan.  The same
    // stripes_mutex_ is the create_stripe() publication barrier: a creator
    // that selected a pre-failure endpoint either appears before this bound
    // or is rejected before publication.
    uint64_t background_rebuild_scan_limit() const {
        std::lock_guard<std::mutex> lock(stripes_mutex_);
        return stripe_count_.load(std::memory_order_acquire);
    }
    // The endpoint overload documents the worker's fault domain; the limit
    // itself is the manager-wide stripe-id upper bound and is intentionally
    // captured under the same publication barrier.
    uint64_t background_rebuild_scan_limit(uint32_t /*failed_endpoint*/) const {
        return background_rebuild_scan_limit();
    }
    // Unsupported obligations must not disappear from completion accounting.
    uint64_t count_rebuild_remaining(uint32_t failed_endpoint) const {
        const uint64_t limit = background_rebuild_scan_limit();
        uint64_t remaining = 0;
        for (uint64_t stripe_id = 0; stripe_id < limit; stripe_id++) {
            BackgroundRebuildStripe view;
            const BackgroundRebuildStatus status = background_rebuild_view(
                stripe_id, failed_endpoint, &view);
            if (status != BackgroundRebuildStatus::Skip) {
                ++remaining;
            }
        }
        return remaining;
    }
    // Inspect one stripe without taking any I/O lock.  Stripe metadata is
    // protected by stripe->mutex; physical addresses and endpoint labels are
    // immutable after publication and are copied as logical bases.
    BackgroundRebuildStatus background_rebuild_view(
        uint64_t stripe_id, uint32_t failed_endpoint,
        BackgroundRebuildStripe *view_out) const {
        if (view_out == nullptr) return BackgroundRebuildStatus::Unsupported;
        *view_out = BackgroundRebuildStripe{};
        if (failed_endpoint >= ::FarLib::allocator::remote::remote_global_heap.endpoint_count() ||
            endpoint_is_alive(failed_endpoint)) return BackgroundRebuildStatus::Unsupported;
        const Stripe *stripe = stripe_by_id(stripe_id);
        if (stripe == nullptr) return BackgroundRebuildStatus::Skip;
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (stripe->stripe_id != stripe_id) return BackgroundRebuildStatus::Skip;
        uint8_t failed_shard = 0xff;
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            if (stripe->shard_endpoint[shard] != failed_endpoint) continue;
            if (failed_shard != 0xff) {
                // A group with two copies on one failed endpoint is outside
                // the single-fault contract, even if its metadata is live.
                return BackgroundRebuildStatus::Unsupported;
            }
            failed_shard = shard;
        }
        if (failed_shard == 0xff) return BackgroundRebuildStatus::Skip;
        if (!stripe->group_stripe) {
            return stripe->live_count == 0
                       ? BackgroundRebuildStatus::Skip
                       : BackgroundRebuildStatus::Unsupported;
        }
        if (!endpoints_distinct(*stripe)) {
            return BackgroundRebuildStatus::Unsupported;
        }
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            const uint64_t redirected =
                stripe->rebuilt_base[shard].load(std::memory_order_acquire);
            if (redirected == kInvalidRemoteAddr) continue;
            if (shard == failed_shard) return BackgroundRebuildStatus::Skip;
            // This manager supports one failed shard per stripe.  A second
            // published redirect needs a different recovery protocol.
            return BackgroundRebuildStatus::Unsupported;
        }
        uint64_t live_groups = 0;
        uint64_t live_objects = 0;
        std::vector<uint32_t> live_slots;
        bool busy = false;
        if (!background_live_groups_locked(*stripe, &live_groups,
                                           &live_objects, &live_slots,
                                           &busy)) {
            return BackgroundRebuildStatus::Unsupported;
        }
        // Zero *durable* groups does not mean empty: initial writes can still
        // publish their live groups. Wait for all such reservations first.
        if (busy) return BackgroundRebuildStatus::Busy;
        if (live_groups == 0) return BackgroundRebuildStatus::Skip;
        if (stripe->slot_size == 0 ||
            stripe->shard_base[failed_shard] == kInvalidRemoteAddr) {
            return BackgroundRebuildStatus::Unsupported;
        }
        BackgroundRebuildStripe snapshot;
        snapshot.stripe_id = stripe_id;
        snapshot.failed_shard = failed_shard;
        snapshot.failed_endpoint = failed_endpoint;
        snapshot.slot_size = stripe->slot_size;
        snapshot.shard_bytes = shard_table_.shard_size();
        snapshot.original_base = stripe->shard_base;
        snapshot.endpoint = stripe->shard_endpoint;
        snapshot.live_groups = live_groups;
        snapshot.live_objects = live_objects;
        snapshot.live_slot_ids = std::move(live_slots);
        *view_out = std::move(snapshot);
        return busy ? BackgroundRebuildStatus::Busy
                    : BackgroundRebuildStatus::Ready;
    }
    // Publish one caller-owned replacement region after the worker has copied
    // the full shard.  Returning false leaves target_base caller-owned, so the
    // caller can release it exactly once on every failed publication.
    bool publish_background_rebuild(const BackgroundRebuildStripe &view,
                                    uint64_t target_base) {
        if (target_base == kInvalidRemoteAddr ||
            view.failed_shard >= kShardCount ||
            view.stripe_id == kInvalidStripeId || view.slot_size == 0 ||
            view.shard_bytes != shard_table_.shard_size() ||
            (target_base % shard_table_.shard_size()) != 0 ||
            view.live_groups == 0 || view.live_groups != view.live_slot_ids.size() ||
            view.live_objects < view.live_groups) {
            return false;
        }
        const auto &cfg = ::FarLib::get_config();
        const auto mapped = cfg.map_remote_addr(target_base);
        if (cfg.ft_standby_endpoint < 0 ||
            mapped.first != static_cast<size_t>(cfg.ft_standby_endpoint) ||
            !endpoint_is_alive(mapped.first) ||
            view.failed_endpoint >= ::FarLib::allocator::remote::remote_global_heap.endpoint_count() ||
            endpoint_is_alive(view.failed_endpoint) ||
            !cfg.validate_mapping(target_base, view.shard_bytes)) return false;
        Stripe *stripe = stripe_by_id(view.stripe_id);
        if (stripe == nullptr) return false;
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (!stripe->group_stripe || stripe->stripe_id != view.stripe_id ||
            stripe->slot_size != view.slot_size ||
            stripe->shard_endpoint[view.failed_shard] !=
                view.failed_endpoint ||
            stripe->shard_base[view.failed_shard] == kInvalidRemoteAddr) {
            return false;
        }
        if (!endpoints_distinct(*stripe)) return false;
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            if (stripe->shard_base[shard] != view.original_base[shard] ||
                stripe->shard_endpoint[shard] != view.endpoint[shard]) {
                return false;
            }
            const uint64_t redirected =
                stripe->rebuilt_base[shard].load(std::memory_order_acquire);
            if (redirected != kInvalidRemoteAddr) return false;
            if (target_base == stripe->shard_base[shard]) return false;
        }
        // Target ownership comes from one fresh whole-region allocation by
        // the sole repair worker. Do not scan the concurrently growing vector.
        uint64_t live_groups = 0;
        uint64_t live_objects = 0;
        std::vector<uint32_t> live_slots;
        bool busy = false;
        if (!background_live_groups_locked(*stripe, &live_groups,
                                           &live_objects, &live_slots,
                                           &busy) ||
            busy || live_groups > view.live_groups ||
            live_objects > view.live_objects) {
            return false;
        }
        // Only releases can reduce the live set after a failed endpoint: old
        // stripes are ineligible for new allocation/reuse.  Ensure no new
        // live codeword appeared after the snapshot while allowing a worker
        // to finish if all objects in a snapshotted group were released.
        if (!std::is_sorted(view.live_slot_ids.begin(), view.live_slot_ids.end()))
            return false;
        for (uint32_t current_slot : live_slots)
            if (!std::binary_search(view.live_slot_ids.begin(),
                                    view.live_slot_ids.end(), current_slot))
                return false;
        stripe->rebuilt_base[view.failed_shard].store(
            target_base, std::memory_order_release);
        return true;
    }
    // Caller already has stripe/shard identity from the logical group view.
    // O(1), including parity: no reverse-binding changes or global scan.
    uint64_t resolve_rebuilt_addr(uint64_t stripe_id, uint8_t shard,
                                  uint64_t logical_addr) const {
        if (shard >= kShardCount) return logical_addr;
        const Stripe *stripe = stripe_by_id(stripe_id);
        if (!stripe) return logical_addr;
        const uint64_t base = stripe->shard_base[shard];
        if (base == kInvalidRemoteAddr || logical_addr < base ||
            logical_addr - base >= shard_table_.shard_size()) return logical_addr;
        const uint64_t target = stripe->rebuilt_base[shard].load(std::memory_order_acquire);
        return target == kInvalidRemoteAddr ? logical_addr : target + (logical_addr - base);
    }
    // Quiescent lifecycle only: repair fibre, application and eviction I/O
    // must all be joined/drained. Repaired regions are never freed on release.
    void release_background_targets_for_shutdown() {
        release_published_background_targets();
    }
    // Translate a logical address only after its stripe has published a
    // replacement for the containing shard.  The original binding table and
    // Stripe::shard_base are intentionally untouched.
    uint64_t resolve_rebuilt_addr(uint64_t logical_addr) const {
        const Stripe *stripe = nullptr;
        uint8_t shard_idx = 0xff;
        uint64_t logical_base = kInvalidRemoteAddr;
        DecodedBinding binding;
        if (binding_for_addr(logical_addr, &binding) &&
            binding.shard_idx < kDataShards) {
            const Stripe *candidate = stripe_by_id(binding.stripe_id);
            if (candidate != nullptr) {
                const uint64_t base = candidate->shard_base[binding.shard_idx];
                if (base != kInvalidRemoteAddr && logical_addr >= base &&
                    logical_addr - base < shard_table_.shard_size()) {
                    stripe = candidate;
                    shard_idx = binding.shard_idx;
                    logical_base = base;
                }
            }
        }
        // Parity regions are deliberately absent from shard_bindings_; locate
        // those logical addresses through immutable stripe geometry.
        if (stripe == nullptr) {
            const uint64_t count = stripe_count_.load(std::memory_order_acquire);
            for (uint64_t id = 0; id < count && stripe == nullptr; id++) {
                const Stripe *candidate = stripe_by_id(id);
                if (candidate == nullptr) continue;
                for (uint8_t shard = kDataShards; shard < kShardCount; shard++) {
                    const uint64_t base = candidate->shard_base[shard];
                    if (base != kInvalidRemoteAddr && logical_addr >= base &&
                        logical_addr - base < shard_table_.shard_size()) {
                        stripe = candidate;
                        shard_idx = shard;
                        logical_base = base;
                        break;
                    }
                }
            }
        }
        if (stripe == nullptr || shard_idx >= kShardCount ||
            logical_base == kInvalidRemoteAddr) {
            return logical_addr;
        }
        const uint64_t target =
            stripe->rebuilt_base[shard_idx].load(std::memory_order_acquire);
        if (target == kInvalidRemoteAddr) return logical_addr;
        return target + (logical_addr - logical_base);
    }

    // Read-only liveness diagnostics used by recovery/self-check code.
    bool endpoint_is_alive(size_t endpoint_idx) const {
        return ::FarLib::allocator::remote::remote_global_heap
            .endpoint_is_alive(endpoint_idx);
    }

    size_t live_endpoint_count() const {
        return ::FarLib::allocator::remote::remote_global_heap
            .live_endpoint_count();
    }

    // Address reverse mapping for the allocator side: true iff addr belongs to
    // a data shard handed out by this manager.
    bool owns(uint64_t addr) const { return binding_for_addr(addr, nullptr); }

    // Immutable placement remains valid after the old slot is retired.
    bool data_address_needs_replacement(uint64_t addr) const {
        if (!recovery_replacement_enabled()) return false;
        DecodedBinding binding;
        if (!binding_for_addr(addr, &binding) || binding.shard_idx >= kDataShards)
            return false;
        const Stripe *s = stripe_by_id(binding.stripe_id);
        return s && s->group_stripe && !group_endpoints_alive_for_reuse(*s);
    }


    // Full layout of a live data slot, including the parity slot addresses.
    bool get_slot_layout(uint64_t data_addr, SlotLayout *layout_out) const {
        DecodedBinding binding;
        if (!binding_for_addr(data_addr, &binding)) return false;
        if (binding.shard_idx >= kDataShards) return false;

        const Stripe *stripe = stripe_by_id(binding.stripe_id);
        if (stripe == nullptr) return false;
        uint64_t data_base = stripe->shard_base[binding.shard_idx];
        if (data_base == kInvalidRemoteAddr || data_addr < data_base) {
            return false;
        }
        uint64_t slot_offset = data_addr - data_base;
        if (stripe->slot_size == 0 || (slot_offset % stripe->slot_size) != 0) {
            return false;
        }
        uint64_t slot_id = slot_offset / stripe->slot_size;
        if (slot_id >= stripe->slots_per_shard) return false;

        if (layout_out != nullptr) {
            SlotLayout layout;
            layout.stripe_id = binding.stripe_id;
            layout.bin = binding.bin;
            layout.slot_size = stripe->slot_size;
            layout.slots_per_shard = stripe->slots_per_shard;
            layout.slot_id = static_cast<uint32_t>(slot_id);
            layout.slot_offset = slot_offset;
            layout.data_shard_idx = binding.shard_idx;
            layout.data_addr = data_addr;
            layout.data_shard_base = data_base;
            for (uint8_t i = 0; i < kParityShards; i++) {
                uint8_t parity_shard = static_cast<uint8_t>(kDataShards + i);
                uint64_t parity_base = stripe->shard_base[parity_shard];
                if (parity_base == kInvalidRemoteAddr) return false;
                layout.parity_shard_idx[i] = parity_shard;
                layout.parity_shard_base[i] = parity_base;
                layout.parity_addr[i] = parity_base + slot_offset;
            }
            *layout_out = layout;
        }
        return true;
    }

    uint64_t allocate(size_t size) {
        size_t bin = bin_from_size(size);
        const uint32_t slot_size =
            static_cast<uint32_t>(::FarLib::allocator::get_bin_size(bin));
        if (slot_size > shard_table_.shard_size()) {
            return kInvalidRemoteAddr;
        }

        uint64_t leased_addr = kInvalidRemoteAddr;
        uint32_t leased_accounted_size = 0;
        bool leased_from_dead = false;
        if (allocate_from_local_lease(static_cast<uint16_t>(bin), &leased_addr,
                                      &leased_accounted_size,
                                      &leased_from_dead)) {
            ::FarLib::allocator::remote::remote_global_heap.inc_used_bytes(
                leased_accounted_size, leased_addr);
            return leased_addr;
        }

        auto &pool = pools_[bin];
        while (true) {
            uint64_t stripe_id = pop_candidate(pool);
            if (stripe_id == kInvalidStripeId) break;
            Stripe *stripe = stripe_by_id(stripe_id);
            if (stripe == nullptr) continue;
            uint64_t addr = kInvalidRemoteAddr;
            uint32_t accounted_size = 0;
            bool should_enqueue = false;
            bool from_dead = false;
            {
                std::lock_guard<std::mutex> lock(stripe->mutex);
                if (!stripe->in_pool) continue;
                stripe->in_pool = false;
                if (!stripe_eligible_for_new_group(*stripe)) continue;
                if (!allocate_from_stripe_locked(*stripe, &addr, &from_dead)) {
                    continue;
                }
                accounted_size = stripe->slot_size;
                should_enqueue = stripe_has_space_locked(*stripe);
            }
            if (should_enqueue) {
                set_local_lease(static_cast<uint16_t>(bin), stripe_id);
            }
            ::FarLib::allocator::remote::remote_global_heap.inc_used_bytes(
                accounted_size, addr);
            return addr;
        }

        Stripe *stripe = create_stripe(bin);
        if (stripe == nullptr) return kInvalidRemoteAddr;
        uint64_t addr = kInvalidRemoteAddr;
        uint32_t accounted_size = 0;
        bool should_enqueue = false;
        bool from_dead = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            bool ok = allocate_from_stripe_locked(*stripe, &addr, &from_dead);
            assert(ok);
            accounted_size = stripe->slot_size;
            should_enqueue = stripe_has_space_locked(*stripe);
        }
        if (should_enqueue) {
            set_local_lease(static_cast<uint16_t>(bin), stripe->stripe_id);
        }
        ::FarLib::allocator::remote::remote_global_heap.inc_used_bytes(
            accounted_size, addr);
        return addr;
    }

    // Removes a small object from its stripe slot and re-queues the stripe if it
    // regained capacity.  Phase 2 will also schedule the parity update here.
    bool mark_dead(uint64_t addr) {
        DecodedBinding binding;
        if (!binding_for_addr(addr, &binding)) return false;
        Stripe *stripe = stripe_by_id(binding.stripe_id);
        if (stripe == nullptr || binding.shard_idx >= kDataShards) {
            return false;
        }

        bool should_enqueue = false;
        uint32_t accounted_size = 0;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            uint64_t base = stripe->shard_base[binding.shard_idx];
            if (base == kInvalidRemoteAddr || addr < base) return false;
            uint64_t offset = addr - base;
            if (stripe->slot_size == 0 || (offset % stripe->slot_size) != 0) {
                return false;
            }
            uint64_t slot = offset / stripe->slot_size;
            if (slot >= stripe->slots_per_shard) return false;
            // Slots owned by a slot group belong to the group and are only
            // released as a whole (mark_dead_group); never break a group up.
            if (group_owns_slot_locked(*stripe, static_cast<uint32_t>(slot))) {
                return false;
            }
            if (stripe->dead_bits[binding.shard_idx].test(slot) ||
                stripe->free_bits[binding.shard_idx].test(slot)) {
                return false;
            }
            stripe->dead_bits[binding.shard_idx].set(slot);
            assert(stripe->live_count > 0);
            stripe->live_count--;
            stripe->dead_count++;
            accounted_size = stripe->slot_size;
            should_enqueue = mark_queued_if_has_space_locked(*stripe);
        }
        ::FarLib::allocator::remote::remote_global_heap.dec_used_bytes(
            accounted_size, addr);
        if (should_enqueue) {
            push_candidate(binding.bin, binding.stripe_id);
        }
        return true;
    }

    // ---------------------------- slot-group API ----------------------------
    // Allocates one slot group whose size class contains `size`.  The handle
    // holds the six segments (0..3 data, 4..5 parity) and their common slot
    // offset/size; handle.id identifies the group for seal / release / layout
    // queries.  Returns false if the cluster has fewer than six endpoints or
    // if no stripe has four data slots free at one common offset.
    bool allocate_slot_group(size_t size, SlotGroupHandle *group_out,
                             bool backup_growth = false,
                             bool recovery_replacement = false) {
        if (group_out == nullptr) return false;
        if (recovery_replacement &&
            (backup_growth || !recovery_replacement_enabled())) return false;
        if (count_endpoints_for_new_allocations() < kShardCount) {
            // Six distinct live endpoints per group are impossible.  A
            // configured standby is counted only after failure activation.
            return false;
        }
        size_t bin = bin_from_size(size);
        const uint32_t slot_size = static_cast<uint32_t>(
            ::FarLib::allocator::get_bin_size(bin));
        if (slot_size > shard_table_.shard_size()) return false;

        auto &pool = group_pools_[bin];
        while (true) {
            uint64_t stripe_id = pop_candidate(pool);
            if (stripe_id == kInvalidStripeId) break;
            Stripe *stripe = stripe_by_id(stripe_id);
            if (stripe == nullptr) continue;
            uint32_t slot_id = 0;
            bool should_enqueue = false;
            {
                std::lock_guard<std::mutex> lock(stripe->mutex);
                if (!stripe->in_group_pool) continue;
                stripe->in_group_pool = false;
                // A released pre-failure group may still be present in the
                // candidate pool.  Never hand it out after one of its old
                // physical endpoints died; its handle remains available to
                // the recovery path at its original addresses.
                if (!stripe_eligible_for_new_group(*stripe)) continue;
                if (!endpoints_distinct(*stripe)) continue;
                if (!allocate_group_from_stripe_locked(*stripe, &slot_id)) {
                    continue;
                }
                should_enqueue =
                    mark_group_queued_if_has_space_locked(*stripe);
            }
            if (should_enqueue) {
                push_group_candidate(static_cast<uint16_t>(bin), stripe_id);
            }
            return fill_group_handle(*stripe, static_cast<uint16_t>(bin),
                                     slot_id, group_out);
        }

        Stripe *stripe = create_stripe(bin, true, backup_growth, recovery_replacement);
        if (stripe == nullptr) return false;
#ifdef FARLIB_REBUILD_TEST_HOOK
        if (background_test_after_create)
            background_test_after_create(this, stripe->stripe_id);
#endif
        uint32_t slot_id = 0;
        bool should_enqueue = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (!allocate_group_from_stripe_locked(*stripe, &slot_id)) {
                return false;
            }
            should_enqueue = mark_group_queued_if_has_space_locked(*stripe);
        }
        if (should_enqueue) {
            push_group_candidate(static_cast<uint16_t>(bin), stripe->stripe_id);
        }
        return fill_group_handle(*stripe, static_cast<uint16_t>(bin), slot_id,
                                 group_out);
    }

    // Reserve one retired data slot from a sealed, durable, non-split group.
    // Candidate lookup is two-level indexed: (bin, behavior) selects an
    // atomic stripe-summary bitmap, then the stripe-local candidate bitmap
    // selects a slot. The reservation leaves group_live_mask unchanged until
    // commit, so a concurrent reader/recovery path still describes old codeword.
    bool begin_slot_reuse_indexed(size_t size, uint32_t behavior_group,
                                  SlotReuse *reuse_out,
                                  bool *busy_out = nullptr,
                                  ::FarLib::profile::ec_rmw_prepare::Trace *trace =
                                      nullptr) {
        if (busy_out != nullptr) *busy_out = false;
        if (!::FarLib::ec_benchmark_phase::enabled()) return false;
        if (reuse_out == nullptr || size == 0) return false;
        *reuse_out = SlotReuse{};
        const size_t bin = bin_from_size(size);
        const uint32_t slot_size = static_cast<uint32_t>(
            ::FarLib::allocator::get_bin_size(bin));
        if (slot_size > shard_table_.shard_size() || size > slot_size ||
            size > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        if (!reuse_behavior_indexable(behavior_group) ||
            stripe_capacity_ == 0) {
            return false;
        }

        auto &cursor = local_reuse_cursor_;
        const size_t count = static_cast<size_t>(
            stripe_count_.load(std::memory_order_acquire));
        if (count == 0) return false;
        if (cursor.owner != this) {
            cursor.owner = this;
            cursor.next_stripe = 0;
            cursor.legacy_next_stripe = 0;
        }
        constexpr size_t kStripeCandidateBudget = 64;
        auto &summary = reuse_summary(static_cast<uint16_t>(bin),
                                       behavior_group);
        for (size_t visited = 0; visited < kStripeCandidateBudget; visited++) {
            size_t index = 0;
            bool summary_found = false;
            {
                ::FarLib::profile::ec_rmw_prepare::Scope summary_scope(
                    trace != nullptr ? &trace->summary : nullptr);
                summary_found =
                    summary.find_set(cursor.next_stripe, &index);
            }
            if (trace != nullptr) {
                trace->summary_calls++;
                if (!summary_found || index >= count) {
                    trace->summary_misses++;
                }
            }
            if (!summary_found ||
                index >= count) {
                return false;
            }
            cursor.next_stripe = (index + 1) % count;
            Stripe *stripe = stripe_by_id(index);
            if (stripe == nullptr) {
                summary.reset(index);
                continue;
            }

            ::FarLib::profile::ec_rmw_prepare::Scope lock_wait(
                trace != nullptr ? &trace->lock_wait : nullptr);
            std::lock_guard<std::mutex> lock(stripe->mutex);
            lock_wait.stop();
            ::FarLib::profile::ec_rmw_prepare::Scope stripe_held(
                trace != nullptr ? &trace->stripe_held : nullptr);
            if (stripe->bin != bin || stripe->slot_size != slot_size ||
                stripe->group_state_words.empty() ||
                !endpoints_distinct(*stripe) ||
                !group_endpoints_alive_for_reuse(*stripe) ||
                !stripe_eligible_for_new_group(*stripe)) {
                // The candidate may have become invalid after publication
                // (endpoint death/standby activation); clear exact bits while
                // holding the stripe lock so no later publication is lost.
                if (stripe->bin == bin && stripe->slot_size == slot_size) {
                    {
                        ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                            trace != nullptr ? &trace->index : nullptr);
                        clear_all_reuse_index_locked(*stripe);
                        summary.reset(index);
                    }
                } else {
                    summary.reset(index);
                }
                continue;
            }
            auto &candidates = stripe->reuse_bits[behavior_group];
            size_t slot = 0;
            if (!candidates.find_set(stripe->reuse_scan_slot, &slot) ||
                slot >= stripe->slots_per_shard) {
                {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_reuse_summary_if_empty_locked(*stripe,
                                                         behavior_group);
                    refresh_reuse_hint_locked(*stripe);
                }
                continue;
            }
            stripe->reuse_scan_slot =
                (static_cast<uint32_t>(slot) + 1) % stripe->slots_per_shard;
            const uint32_t slot_id = static_cast<uint32_t>(slot);
            if (group_state_locked(*stripe, slot_id) != kSlotGroupSealed ||
                slot_id >= stripe->group_durable.size() ||
                stripe->group_durable[slot_id] == 0 ||
                slot_id >= stripe->group_is_split.size() ||
                stripe->group_is_split[slot_id] != 0 ||
                slot_id >= stripe->group_behavior_group.size() ||
                stripe->group_behavior_group[slot_id] != behavior_group) {
                {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_reuse_key_locked(*stripe, slot_id, behavior_group);
                }
                continue;
            }
            const uint8_t live = static_cast<uint8_t>(
                stripe->group_live_mask[slot_id] & 0x0fu);
            const uint8_t holes = static_cast<uint8_t>(~live & 0x0fu);
            if (holes == 0) {
                {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_reuse_key_locked(*stripe, slot_id, behavior_group);
                }
                continue;
            }
            if (group_update_pending_locked(*stripe, slot_id)) {
                if (busy_out != nullptr) *busy_out = true;
                {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_reuse_key_locked(*stripe, slot_id, behavior_group);
                }
                continue;
            }
            const uint8_t data_shard = static_cast<uint8_t>(
                __builtin_ctz(static_cast<unsigned>(holes)));
            SlotGroupHandle handle;
            if (!fill_group_handle(*stripe, stripe->bin, slot_id, &handle)) {
                {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_reuse_key_locked(*stripe, slot_id, behavior_group);
                }
                continue;
            }
            if (slot_id >= stripe->group_generation.size()) {
                {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_reuse_key_locked(*stripe, slot_id, behavior_group);
                }
                continue;
            }
            const uint64_t generation =
                next_group_generation_locked(*stripe, slot_id);
            if (generation == 0) {
                {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_reuse_key_locked(*stripe, slot_id, behavior_group);
                }
                continue;
            }
            stripe->group_pending_mask[slot_id] =
                static_cast<uint8_t>(1u << data_shard);
            // Pending target is no longer readable/reusable. Clear exact
            // candidate bit and summary before exposing reservation.
            {
                ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                    trace != nullptr ? &trace->index : nullptr);
                clear_reuse_key_locked(*stripe, slot_id, behavior_group);
            }
            reuse_out->group = handle;
            reuse_out->generation = generation;
            reuse_out->data_shard = data_shard;
            reuse_out->size = static_cast<uint32_t>(size);
            reuse_out->behavior_group = behavior_group;
            return true;
        }
        return false;
    }

    // Reserve independent holes by metadata Stripe, holding each selected
    // Stripe's mutex only once.  Each output still owns its own handle, data
    // shard, generation and pending bit; this is an allocator metadata
    // optimization, not a wider EC transaction.
    //
    // The indexed bitmap is exact for the requested behavior key.  A stale
    // candidate is cleared only from that key in the normal path; the full
    // six-key cleanup remains reserved for an invalid Stripe/endpoint state.
    // The return value is a successful prefix; later output pointers are not
    // touched when candidate supply is exhausted.
    size_t begin_slot_reuse_indexed_batch(
        size_t size, uint32_t behavior_group, SlotReuse **outputs,
        size_t capacity, bool *busy_out = nullptr,
        ::FarLib::profile::ec_rmw_prepare::Trace *trace = nullptr) {
        if (busy_out != nullptr) *busy_out = false;
        if (outputs == nullptr || capacity == 0 || size == 0) return 0;
        for (size_t i = 0; i < capacity; ++i) {
            if (outputs[i] == nullptr) return 0;
        }
        if (!::FarLib::ec_benchmark_phase::enabled()) return 0;
        if (!reuse_behavior_indexable(behavior_group) ||
            stripe_capacity_ == 0) {
            return 0;
        }

        const size_t bin = bin_from_size(size);
        const uint32_t slot_size = static_cast<uint32_t>(
            ::FarLib::allocator::get_bin_size(bin));
        if (slot_size > shard_table_.shard_size() || size > slot_size ||
            size > std::numeric_limits<uint32_t>::max()) {
            return 0;
        }

        auto &cursor = local_reuse_cursor_;
        const size_t count = static_cast<size_t>(
            stripe_count_.load(std::memory_order_acquire));
        if (count == 0) return 0;
        if (cursor.owner != this) {
            cursor.owner = this;
            cursor.next_stripe = 0;
            cursor.legacy_next_stripe = 0;
        }

        constexpr size_t kStripeCandidateBudget = 64;
        auto &summary = reuse_summary(static_cast<uint16_t>(bin),
                                       behavior_group);
        size_t total_acquired = 0;
        for (size_t visited = 0; visited < kStripeCandidateBudget;
             ++visited) {
            size_t index = 0;
            bool summary_found = false;
            {
                ::FarLib::profile::ec_rmw_prepare::Scope summary_scope(
                    trace != nullptr ? &trace->summary : nullptr);
                summary_found = summary.find_set(cursor.next_stripe, &index);
            }
            if (trace != nullptr) {
                ++trace->summary_calls;
                if (!summary_found || index >= count) ++trace->summary_misses;
            }
            if (!summary_found || index >= count) return total_acquired;
            cursor.next_stripe = (index + 1) % count;
            Stripe *stripe = stripe_by_id(index);
            if (stripe == nullptr) {
                summary.reset(index);
                continue;
            }

            ::FarLib::profile::ec_rmw_prepare::Scope lock_wait(
                trace != nullptr ? &trace->lock_wait : nullptr);
            std::lock_guard<std::mutex> lock(stripe->mutex);
            lock_wait.stop();
            ::FarLib::profile::ec_rmw_prepare::Scope stripe_held(
                trace != nullptr ? &trace->stripe_held : nullptr);

            if (stripe->bin != bin || stripe->slot_size != slot_size ||
                stripe->group_state_words.empty() ||
                !endpoints_distinct(*stripe) ||
                !group_endpoints_alive_for_reuse(*stripe) ||
                !stripe_eligible_for_new_group(*stripe)) {
                // Endpoint/Stripe invalidation is the exceptional path where
                // all keys must be cleared while this lock is held.
                if (stripe->bin == bin && stripe->slot_size == slot_size) {
                    ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                        trace != nullptr ? &trace->index : nullptr);
                    clear_all_reuse_index_locked(*stripe);
                    summary.reset(index);
                } else {
                    summary.reset(index);
                }
                continue;
            }

            auto &candidates = stripe->reuse_bits[behavior_group];
            while (total_acquired < capacity) {
                    size_t slot = 0;
                    if (!candidates.find_set(stripe->reuse_scan_slot,
                                              &slot) ||
                        slot >= stripe->slots_per_shard) {
                        break;
                    }
                    stripe->reuse_scan_slot =
                        (static_cast<uint32_t>(slot) + 1) %
                        stripe->slots_per_shard;
                    const uint32_t slot_id = static_cast<uint32_t>(slot);

                    const bool structurally_valid =
                        group_state_locked(*stripe, slot_id) ==
                            kSlotGroupSealed &&
                        slot_id < stripe->group_durable.size() &&
                        stripe->group_durable[slot_id] != 0 &&
                        slot_id < stripe->group_is_split.size() &&
                        stripe->group_is_split[slot_id] == 0 &&
                        slot_id < stripe->group_live_mask.size() &&
                        slot_id < stripe->group_pending_mask.size() &&
                        slot_id < stripe->group_generation.size() &&
                        slot_id < stripe->group_behavior_group.size() &&
                        stripe->group_behavior_group[slot_id] ==
                            behavior_group;
                    if (!structurally_valid) {
                        // Behavior is immutable for a sealed group, so a
                        // stale indexed bit can be removed from this key only.
                        ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                            trace != nullptr ? &trace->index : nullptr);
                        candidates.reset(slot_id);
                        continue;
                    }

                    const uint8_t live = static_cast<uint8_t>(
                        stripe->group_live_mask[slot_id] & 0x0fu);
                    const uint8_t holes = static_cast<uint8_t>(~live & 0x0fu);
                    if (holes == 0) {
                        ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                            trace != nullptr ? &trace->index : nullptr);
                        candidates.reset(slot_id);
                        continue;
                    }
                    if (group_update_pending_locked(*stripe, slot_id)) {
                        if (busy_out != nullptr) *busy_out = true;
                        ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                            trace != nullptr ? &trace->index : nullptr);
                        candidates.reset(slot_id);
                        continue;
                    }

                    const uint8_t data_shard = static_cast<uint8_t>(
                        __builtin_ctz(static_cast<unsigned>(holes)));
                    SlotGroupHandle handle;
                    if (!fill_group_handle(*stripe, stripe->bin, slot_id,
                                            &handle)) {
                        ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                            trace != nullptr ? &trace->index : nullptr);
                        candidates.reset(slot_id);
                        continue;
                    }
                    const uint64_t generation =
                        next_group_generation_locked(*stripe, slot_id);
                    if (generation == 0) {
                        ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                            trace != nullptr ? &trace->index : nullptr);
                        candidates.reset(slot_id);
                        continue;
                    }

                    // The output pointer was checked before touching metadata;
                    // assigning it cannot leave a pending reservation without
                    // a caller-visible SlotReuse record.
                    stripe->group_pending_mask[slot_id] =
                        static_cast<uint8_t>(1u << data_shard);
                    {
                        ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                            trace != nullptr ? &trace->index : nullptr);
                        candidates.reset(slot_id);
                    }
                    SlotReuse &reuse = *outputs[total_acquired];
                    reuse = SlotReuse{};
                    reuse.group = handle;
                    reuse.generation = generation;
                    reuse.data_shard = data_shard;
                    reuse.size = static_cast<uint32_t>(size);
                    reuse.behavior_group = behavior_group;
                    ++total_acquired;
            }

            // Publish/clear the manager summary only once for this Stripe;
            // all exact candidate bits above were changed under stripe->mutex.
            {
                ::FarLib::profile::ec_rmw_prepare::Scope index_scope(
                    trace != nullptr ? &trace->index : nullptr);
                const bool remaining = candidates.any_set();
                if (!remaining) {
                    clear_reuse_summary_if_empty_locked(*stripe,
                                                         behavior_group);
                }
                refresh_reuse_hint_locked(*stripe);
            }
            if (total_acquired == capacity) return total_acquired;
        }
        return total_acquired;
    }

    // Compatibility entry point. All callers now share indexed lookup; no
    // phase flag is set. The legacy path intentionally preserves its bounded
    // hint-guided scan and all prior behavior when phased mode is disabled.
    bool begin_slot_reuse(size_t size, uint32_t behavior_group,
                          SlotReuse *reuse_out, bool *busy_out = nullptr) {
        if (busy_out != nullptr) *busy_out = false;
        if (reuse_out == nullptr || size == 0) return false;
        *reuse_out = SlotReuse{};
        const size_t bin = bin_from_size(size);
        const uint32_t slot_size = static_cast<uint32_t>(
            ::FarLib::allocator::get_bin_size(bin));
        if (slot_size > shard_table_.shard_size() || size > slot_size ||
            size > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        auto &cursor = local_reuse_cursor_;
        const size_t count = static_cast<size_t>(
            stripe_count_.load(std::memory_order_acquire));
        if (count == 0) return false;
        if (cursor.owner != this) {
            cursor.owner = this;
            cursor.next_stripe = 0;
            cursor.legacy_next_stripe = 0;
        }
        const size_t start = cursor.legacy_next_stripe % count;
        constexpr size_t kStripeScanBudget = 64;
        constexpr size_t kSlotScanBudget = 128;
        const size_t stripe_budget = std::min(count, kStripeScanBudget);
        for (size_t visited = 0; visited < stripe_budget; visited++) {
            const size_t index = (start + visited) % count;
            cursor.legacy_next_stripe = (index + 1) % count;
            Stripe *stripe = stripe_by_id(index);
            if (stripe == nullptr) continue;
            if (stripe->bin != bin || stripe->slot_size != slot_size ||
                !stripe->reuse_candidate.load(std::memory_order_acquire)) {
                continue;
            }

            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (stripe->group_state_words.empty() ||
                !endpoints_distinct(*stripe) ||
                !group_endpoints_alive_for_reuse(*stripe) ||
                !stripe_eligible_for_new_group(*stripe)) {
                continue;
            }
            const size_t slot_count = stripe->slots_per_shard;
            if (slot_count == 0) continue;
            const size_t slot_start = stripe->reuse_scan_slot % slot_count;
            const size_t slot_budget = std::min(slot_count, kSlotScanBudget);
            bool found_any_hole = false;
            for (size_t step = 0; step < slot_budget; step++) {
                const uint32_t slot = static_cast<uint32_t>(
                    (slot_start + step) % slot_count);
                stripe->reuse_scan_slot = (slot + 1) % stripe->slots_per_shard;
                if (group_state_locked(*stripe, slot) != kSlotGroupSealed ||
                    slot >= stripe->group_durable.size() ||
                    stripe->group_durable[slot] == 0 ||
                    slot >= stripe->group_is_split.size() ||
                    stripe->group_is_split[slot] != 0) {
                    continue;
                }
                const uint8_t live = static_cast<uint8_t>(
                    stripe->group_live_mask[slot] & 0x0fu);
                const uint8_t holes = static_cast<uint8_t>(~live & 0x0fu);
                if (holes == 0) continue;
                found_any_hole = true;
                if (slot >= stripe->group_behavior_group.size() ||
                    stripe->group_behavior_group[slot] != behavior_group) {
                    continue;
                }
                if (group_update_pending_locked(*stripe, slot)) {
                    if (busy_out != nullptr) *busy_out = true;
                    continue;
                }
                const uint8_t data_shard = static_cast<uint8_t>(
                    __builtin_ctz(static_cast<unsigned>(holes)));
                SlotGroupHandle handle;
                if (!fill_group_handle(*stripe, stripe->bin, slot, &handle)) {
                    continue;
                }
                if (slot >= stripe->group_generation.size()) continue;
                const uint64_t generation =
                    next_group_generation_locked(*stripe, slot);
                if (generation == 0) continue;
                stripe->group_pending_mask[slot] =
                    static_cast<uint8_t>(1u << data_shard);
                reuse_out->group = handle;
                reuse_out->generation = generation;
                reuse_out->data_shard = data_shard;
                reuse_out->size = static_cast<uint32_t>(size);
                reuse_out->behavior_group = behavior_group;
                return true;
            }
            if (!found_any_hole && !stripe_has_reuse_candidate_locked(*stripe)) {
                stripe->reuse_candidate.store(false,
                                               std::memory_order_release);
            }
        }
        return false;
    }

    // Commit a slot reuse after the coordinator has drained the three
    // absolute-write completions.  The generation and target bit are checked
    // under stripe.mutex so stale callbacks cannot publish an ABA-reused slot.
    bool commit_slot_reuse(const SlotReuse &reuse) {
        const SlotGroupId &id = reuse.group.id;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || !id.valid() ||
            id.slot_id >= stripe->slots_per_shard ||
            reuse.data_shard >= kDataShards || reuse.size == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) != kSlotGroupSealed ||
            id.slot_id >= stripe->group_pending_mask.size() ||
            stripe->group_pending_mask[id.slot_id] !=
                static_cast<uint8_t>(1u << reuse.data_shard) ||
            id.slot_id >= stripe->group_generation.size() ||
            stripe->group_generation[id.slot_id] != reuse.generation ||
            id.slot_id >= stripe->group_behavior_group.size() ||
            stripe->group_behavior_group[id.slot_id] != reuse.behavior_group ||
            reuse.group.slot_size != stripe->slot_size ||
            reuse.group.bin != stripe->bin || reuse.size > stripe->slot_size) {
            return false;
        }
        SlotGroupHandle current;
        if (!fill_group_handle(*stripe, stripe->bin, id.slot_id, &current) ||
            !same_group_handle(reuse.group, current)) {
            return false;
        }
        const uint8_t bit = static_cast<uint8_t>(1u << reuse.data_shard);
        const uint8_t old_live = group_live_slots_locked(*stripe, id.slot_id);
        if ((stripe->group_live_mask[id.slot_id] & bit) != 0 ||
            old_live >= kDataShards) {
            return false;
        }
        account_group_slot_reuse_commit_locked(
            *stripe, id.slot_id, reuse.data_shard, reuse.size);
        stripe->group_live_mask[id.slot_id] = static_cast<uint8_t>(
            stripe->group_live_mask[id.slot_id] | bit);
        stripe->group_live_objects[id.slot_id] =
            static_cast<uint8_t>(old_live + 1);
        stripe->group_payload_sizes[id.slot_id][reuse.data_shard] = reuse.size;
        stripe->group_payload_unknown[id.slot_id] = static_cast<uint8_t>(
            stripe->group_payload_unknown[id.slot_id] & ~bit);
        stripe->group_pending_mask[id.slot_id] = 0;
        stripe->live_count++;
        if (!::FarLib::ec_benchmark_phase::enabled() &&
            stripe->group_live_mask[id.slot_id] != 0x0fu) {
            stripe->reuse_candidate.store(true, std::memory_order_release);
        } else {
            update_reuse_index_locked(*stripe, id.slot_id);
        }
        return true;
    }

    // Commit a set of already-completed slot reuses.  Reservations are grouped
    // by metadata Stripe without allocating or holding more than one Stripe
    // lock at a time.  Each Stripe is fully preflighted before its live masks
    // or accounting are modified.
    bool commit_slot_reuse_batch(const SlotReuse *const *reuses,
                                 size_t count) {
        if (count == 0) return true;
        if (reuses == nullptr) return false;

        // The only production caller is the bounded worker batch (64
        // transactions).  Keeping the grouping marks on the stack avoids a
        // hot-path allocation and lets each metadata Stripe take one lock.
        constexpr size_t kCommitBatchCapacity = 64;
        if (count > kCommitBatchCapacity) return false;
        auto same_stripe = [](const SlotReuse &left,
                              const SlotReuse &right) {
            return left.group.id.stripe_id == right.group.id.stripe_id;
        };

        for (size_t i = 0; i < count; ++i) {
            if (reuses[i] == nullptr || !reuses[i]->group.id.valid() ||
                reuses[i]->generation == 0 || reuses[i]->size == 0 ||
                reuses[i]->data_shard >= kDataShards) {
                return false;
            }
            if (::FarLib::ec_benchmark_phase::enabled() &&
                !reuse_behavior_indexable(reuses[i]->behavior_group)) {
                return false;
            }
        }

        auto validate_locked = [&](Stripe &stripe,
                                   const SlotReuse &reuse) -> bool {
            const SlotGroupId &id = reuse.group.id;
            if (id.stripe_id != stripe.stripe_id ||
                id.slot_id >= stripe.slots_per_shard ||
                reuse.data_shard >= kDataShards || reuse.size == 0 ||
                reuse.group.slot_size != stripe.slot_size ||
                reuse.group.bin != stripe.bin ||
                reuse.size > stripe.slot_size ||
                group_state_locked(stripe, id.slot_id) != kSlotGroupSealed ||
                id.slot_id >= stripe.group_durable.size() ||
                stripe.group_durable[id.slot_id] == 0 ||
                id.slot_id >= stripe.group_is_split.size() ||
                stripe.group_is_split[id.slot_id] != 0 ||
                id.slot_id >= stripe.group_live_mask.size() ||
                id.slot_id >= stripe.group_live_objects.size() ||
                id.slot_id >= stripe.group_pending_mask.size() ||
                id.slot_id >= stripe.group_generation.size() ||
                id.slot_id >= stripe.group_behavior_group.size() ||
                id.slot_id >= stripe.group_payload_sizes.size() ||
                id.slot_id >= stripe.group_payload_unknown.size() ||
                stripe.group_pending_mask[id.slot_id] !=
                    static_cast<uint8_t>(1u << reuse.data_shard) ||
                stripe.group_generation[id.slot_id] != reuse.generation ||
                stripe.group_behavior_group[id.slot_id] !=
                    reuse.behavior_group) {
                return false;
            }
            SlotGroupHandle current;
            if (!fill_group_handle(stripe, stripe.bin, id.slot_id, &current) ||
                !same_group_handle(reuse.group, current)) {
                return false;
            }
            const uint8_t bit = static_cast<uint8_t>(1u << reuse.data_shard);
            const uint8_t old_live =
                group_live_slots_locked(stripe, id.slot_id);
            return (stripe.group_live_mask[id.slot_id] & bit) == 0 &&
                   old_live < kDataShards;
        };

        // Validate and apply each distinct Stripe under one short lock.  A
        // pending bit prevents another allocator path from changing these
        // reservations while this lock is held.
        std::array<bool, kCommitBatchCapacity> visited{};
        for (size_t i = 0; i < count; ++i) {
            if (visited[i]) continue;
            Stripe *stripe = stripe_by_id(reuses[i]->group.id.stripe_id);
            if (stripe == nullptr) return false;
            for (size_t j = i; j < count; ++j) {
                if (!visited[j] && same_stripe(*reuses[i], *reuses[j]))
                    visited[j] = true;
            }
            std::lock_guard<std::mutex> lock(stripe->mutex);
            for (size_t j = i; j < count; ++j) {
                if (same_stripe(*reuses[i], *reuses[j]) &&
                    !validate_locked(*stripe, *reuses[j])) {
                    return false;
                }
            }
            std::array<bool, kReuseBehaviorGroupCount> touched{};
            const bool index_eligible =
                endpoints_distinct(*stripe) &&
                group_endpoints_alive_for_reuse(*stripe) &&
                stripe_eligible_for_new_group(*stripe);
            for (size_t j = i; j < count; ++j) {
                if (!same_stripe(*reuses[i], *reuses[j])) continue;
                const SlotReuse &reuse = *reuses[j];
                const uint32_t slot = reuse.group.id.slot_id;
                const uint8_t bit =
                    static_cast<uint8_t>(1u << reuse.data_shard);
                const uint8_t old_live = group_live_slots_locked(*stripe, slot);
                account_group_slot_reuse_commit_locked(
                    *stripe, slot, reuse.data_shard, reuse.size);
                stripe->group_live_mask[slot] = static_cast<uint8_t>(
                    stripe->group_live_mask[slot] | bit);
                stripe->group_live_objects[slot] =
                    static_cast<uint8_t>(old_live + 1);
                stripe->group_payload_sizes[slot][reuse.data_shard] =
                    reuse.size;
                stripe->group_payload_unknown[slot] = static_cast<uint8_t>(
                    stripe->group_payload_unknown[slot] & ~bit);
                stripe->group_pending_mask[slot] = 0;
                stripe->live_count++;
                if (!::FarLib::ec_benchmark_phase::enabled()) {
                    if (stripe->group_live_mask[slot] != 0x0fu) {
                        stripe->reuse_candidate.store(
                            true, std::memory_order_release);
                    }
                } else {
                    touched[reuse.behavior_group] = true;
                    auto &candidates =
                        stripe->reuse_bits[reuse.behavior_group];
                    if (index_eligible &&
                        stripe->group_live_mask[slot] != 0x0fu) {
                        candidates.set(slot);
                    } else {
                        candidates.reset(slot);
                    }
                }
            }
            if (::FarLib::ec_benchmark_phase::enabled()) {
                if (!index_eligible) {
                    // Recovery/endpoint invalidation is exceptional: clear
                    // every key so no stale candidate survives the failure.
                    clear_all_reuse_index_locked(*stripe);
                } else {
                    for (uint32_t behavior = 0;
                         behavior < kReuseBehaviorGroupCount; ++behavior) {
                        if (!touched[behavior]) continue;
                        // Reservation may have removed the last candidate
                        // and cleared this summary bit. A committed group
                        // with another hole must become discoverable again.
                        auto &summary = reuse_summary(stripe->bin, behavior);
                        if (stripe->reuse_bits[behavior].any_set()) {
                            summary.set(static_cast<size_t>(stripe->stripe_id));
                        } else {
                            summary.reset(static_cast<size_t>(stripe->stripe_id));
                        }
                    }
                }
                refresh_reuse_hint_locked(*stripe);
            }
        }
        return true;
    }

    // Abort a pending reuse.  If the old group lost its final live object
    // while the transaction was in flight, retire the now-empty group only
    // after clearing the pending bit and revalidate the same generation.
    bool abort_slot_reuse(const SlotReuse &reuse) {
        const SlotGroupId &id = reuse.group.id;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || !id.valid() ||
            id.slot_id >= stripe->slots_per_shard ||
            reuse.data_shard >= kDataShards) {
            return false;
        }
        bool became_dead = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (group_state_locked(*stripe, id.slot_id) != kSlotGroupSealed ||
                id.slot_id >= stripe->group_pending_mask.size() ||
                stripe->group_pending_mask[id.slot_id] !=
                    static_cast<uint8_t>(1u << reuse.data_shard) ||
                id.slot_id >= stripe->group_generation.size() ||
                stripe->group_generation[id.slot_id] != reuse.generation ||
                id.slot_id >= stripe->group_behavior_group.size() ||
                stripe->group_behavior_group[id.slot_id] != reuse.behavior_group) {
                return false;
            }
            SlotGroupHandle current;
            if (!fill_group_handle(*stripe, stripe->bin, id.slot_id, &current) ||
                !same_group_handle(reuse.group, current)) {
                return false;
            }
            stripe->group_pending_mask[id.slot_id] = 0;
            if (stripe->group_live_mask[id.slot_id] == 0) {
                const bool retain_empty =
                    ::FarLib::ec_benchmark_phase::enabled() &&
                    id.slot_id < stripe->group_is_split.size() &&
                    stripe->group_is_split[id.slot_id] == 0;
                if (retain_empty) {
                    update_reuse_index_locked(*stripe, id.slot_id);
                } else {
                    became_dead = mark_dead_group_locked(*stripe, id);
                }
            } else {
                if (!::FarLib::ec_benchmark_phase::enabled() &&
                    stripe->group_live_mask[id.slot_id] != 0x0fu) {
                    stripe->reuse_candidate.store(true,
                                                   std::memory_order_release);
                } else {
                    update_reuse_index_locked(*stripe, id.slot_id);
                }
            }
        }
        if (became_dead) requeue_group_stripe(*stripe);
        return true;
    }

    bool slot_group_update_pending(const SlotGroupId &id) const {
        const Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        return group_update_pending_locked(*stripe, id.slot_id);
    }

    // Mark initial group publication durable after all six terminal write
    // outcomes are observed.  Slot reuse is intentionally restricted to this
    // state so its PREPARE reads never race an initial write round.
    void mark_slot_group_durable(const SlotGroupId &id) {
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) return;
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) == kSlotGroupSealed &&
            !group_update_pending_locked(*stripe, id.slot_id) &&
            id.slot_id < stripe->group_durable.size()) {
            stripe->group_durable[id.slot_id] = 1;
            if (!::FarLib::ec_benchmark_phase::enabled() &&
                id.slot_id < stripe->group_live_mask.size() &&
                (stripe->group_live_mask[id.slot_id] & 0x0fu) != 0x0fu &&
                id.slot_id < stripe->group_is_split.size() &&
                stripe->group_is_split[id.slot_id] == 0) {
                stripe->reuse_candidate.store(true,
                                               std::memory_order_release);
            } else {
                update_reuse_index_locked(*stripe, id.slot_id);
            }
        }
    }

    bool addr_update_pending(uint64_t addr) const {
        DecodedBinding binding;
        if (!binding_for_addr(addr, &binding) || binding.shard_idx >= kDataShards) {
            return false;
        }
        const Stripe *stripe = stripe_by_id(binding.stripe_id);
        if (stripe == nullptr || !stripe->has_groups.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        const uint64_t base = stripe->shard_base[binding.shard_idx];
        if (base == kInvalidRemoteAddr || addr < base || stripe->slot_size == 0) {
            return false;
        }
        const uint64_t offset = addr - base;
        if ((offset % stripe->slot_size) != 0) return false;
        const uint64_t slot64 = offset / stripe->slot_size;
        if (slot64 >= stripe->slots_per_shard) return false;
        const uint32_t slot = static_cast<uint32_t>(slot64);
        const uint8_t pending = group_pending_mask_locked(*stripe, slot);
        return (pending & static_cast<uint8_t>(1u << binding.shard_idx)) != 0;
    }

    // free -> in_progress -> sealed.  live_mask is a 4-bit mask (bit i = data
    // shard i); any value below 16 is accepted, including fewer than four live
    // objects.  Slots that are not live hold no object and their shard content
    // is defined to be zero (the writer guarantees it).  The group keeps owning
    // all four data slots, live or not, so they are never handed out singly.
    bool seal_slot_group(const SlotGroupId &id, uint8_t live_mask,
                         const uint32_t *sizes = nullptr,
                         uint32_t behavior_group = 0) {
        if ((live_mask & 0xf0u) != 0) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) != kSlotGroupInProgress) {
            return false;
        }
        // Validate before touching either state or accounting.  A supplied
        // size is an exact payload length, not a slot capacity; holes must be
        // represented by zero and live objects must be non-zero and fit.
        if (sizes != nullptr) {
            for (uint8_t i = 0; i < kDataShards; i++) {
                const bool live = (live_mask & static_cast<uint8_t>(1u << i)) != 0;
                const uint32_t payload = sizes[i];
                if ((live && (payload == 0 || payload > stripe->slot_size)) ||
                    (!live && payload != 0)) {
                    return false;
                }
            }
        }
        uint32_t live = static_cast<uint32_t>(
            __builtin_popcount(static_cast<unsigned>(live_mask)));
        assert(stripe->live_count >= kDataShards - live);
        stripe->live_count -= (kDataShards - live);
        stripe->group_live_mask[id.slot_id] = live_mask;
        stripe->group_pending_mask[id.slot_id] = 0;
        stripe->group_durable[id.slot_id] = 0;
        stripe->group_behavior_group[id.slot_id] = behavior_group;
        stripe->group_live_objects[id.slot_id] = static_cast<uint8_t>(live);
        stripe->group_payload_sizes[id.slot_id].fill(0);
        stripe->group_payload_unknown[id.slot_id] = 0;
        for (uint8_t i = 0; i < kDataShards; i++) {
            if ((live_mask & static_cast<uint8_t>(1u << i)) == 0) continue;
            if (sizes == nullptr) {
                stripe->group_payload_sizes[id.slot_id][i] = stripe->slot_size;
                stripe->group_payload_unknown[id.slot_id] |=
                    static_cast<uint8_t>(1u << i);
            } else {
                stripe->group_payload_sizes[id.slot_id][i] = sizes[i];
            }
        }
        set_group_state_locked(*stripe, id.slot_id, kSlotGroupSealed);
        account_group_seal_locked(*stripe, id.slot_id, false);
        update_reuse_index_locked(*stripe, id.slot_id);
        return true;
    }

    // Seal one large-object split group. All four data fragments are valid
    // internal shards, while data[0] remains the sole public owner.
    bool seal_split_slot_group(const SlotGroupId &id,
                               uint32_t payload_size = 0) {
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) != kSlotGroupInProgress ||
            id.slot_id >= stripe->group_is_split.size()) {
            return false;
        }
        const uint64_t max_payload =
            static_cast<uint64_t>(kDataShards) * stripe->slot_size;
        if (payload_size != 0 && payload_size > max_payload) return false;
        stripe->group_live_mask[id.slot_id] = 0xfu;
        stripe->group_pending_mask[id.slot_id] = 0;
        stripe->group_durable[id.slot_id] = 0;
        stripe->group_behavior_group[id.slot_id] = 0;
        stripe->group_live_objects[id.slot_id] = kDataShards;
        stripe->group_is_split[id.slot_id] = 1;
        stripe->group_payload_sizes[id.slot_id].fill(0);
        // A split group has one public object anchored at data[0].  With no
        // payload argument retain a conservative full-group estimate, but mark
        // it unknown so callers never mistake the padding ratio for exact data.
        stripe->group_payload_sizes[id.slot_id][0] =
            payload_size != 0 ? payload_size
                              : static_cast<uint32_t>(max_payload);
        stripe->group_payload_unknown[id.slot_id] =
            payload_size == 0 ? static_cast<uint8_t>(1u) : 0;
        set_group_state_locked(*stripe, id.slot_id, kSlotGroupSealed);
        account_group_seal_locked(*stripe, id.slot_id, true);
        update_reuse_index_locked(*stripe, id.slot_id);
        return true;
    }

    // Whole-group release: in_progress/sealed -> dead.  The four data slots
    // stay reserved inside the group, so they can only come back as one new
    // group (dead -> in_progress); the single-object path never sees them.
    bool mark_dead_group(const SlotGroupId &id) {
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            released = mark_dead_group_locked(*stripe, id);
        }
        if (released) requeue_group_stripe(*stripe);
        return released;
    }

    // State-machine core of a group release: in_progress/sealed -> dead, with
    // the group's remaining live objects taken out of the stripe's live count
    // (an in_progress group owns the four slots it never published).  dead is
    // the only legal terminal state a group can be reused from
    // (dead -> in_progress).  The caller holds stripe.mutex; an already
    // dead/free group is refused and left untouched.
    bool mark_dead_group_locked(Stripe &stripe, const SlotGroupId &id) {
        const uint8_t state = group_state_locked(stripe, id.slot_id);
        if (state != kSlotGroupInProgress && state != kSlotGroupSealed) {
            return false;
        }
        // A sealed group with an outstanding slot-reuse transaction still
        // owns the old codeword.  Do not retire it underneath the coordinator;
        // abort_slot_reuse() will clear the reservation and retry this
        // transition once the three-part update has settled.
        if (group_update_pending_locked(stripe, id.slot_id)) return false;
        // Retained empty groups carry an indexed RMW candidate. Remove that
        // publication before changing state to dead (shutdown path and the
        // legacy non-phased last-release path both use this helper).
        clear_reuse_index_locked(stripe, id.slot_id);
        uint32_t live = kDataShards;
        if (state == kSlotGroupSealed) {
            live = static_cast<uint32_t>(__builtin_popcount(
                static_cast<unsigned>(stripe.group_live_mask[id.slot_id])));
        }
        assert(stripe.live_count >= live);
        const bool split = id.slot_id < stripe.group_is_split.size() &&
                           stripe.group_is_split[id.slot_id] != 0;
        account_group_dead_locked(stripe, id.slot_id, split);
        stripe.live_count -= live;
        stripe.group_live_mask[id.slot_id] = 0;
        stripe.group_pending_mask[id.slot_id] = 0;
        stripe.group_durable[id.slot_id] = 0;
        stripe.group_behavior_group[id.slot_id] = 0;
        stripe.group_live_objects[id.slot_id] = 0;
        stripe.group_payload_sizes[id.slot_id].fill(0);
        stripe.group_payload_unknown[id.slot_id] = 0;
        if (id.slot_id < stripe.group_is_split.size()) {
            stripe.group_is_split[id.slot_id] = 0;
        }
        set_group_state_locked(stripe, id.slot_id, kSlotGroupDead);
        stripe.group_dead_count++;
        return true;
    }

    // A stripe that owns a reusable (dead) group slot can serve a group again:
    // re-queue it exactly like allocate_slot_group() does after an allocation.
    // Takes the stripe mutex, so the caller must not hold it.
    void requeue_group_stripe(Stripe &stripe) {
        bool should_enqueue = false;
        {
            std::lock_guard<std::mutex> lock(stripe.mutex);
            should_enqueue = mark_group_queued_if_has_space_locked(stripe);
        }
        if (should_enqueue) {
            push_group_candidate(stripe.bin, stripe.stripe_id);
        }
    }

    // Cold shutdown hook for phased mode. Producers must be quiesced before
    // calling: convert retained sealed-empty groups to ordinary dead groups
    // under their existing stripe mutex, without requeueing or recursively
    // retaining them. This restores normal group accounting before allocator
    // teardown releases backing regions.
    void release_retained_empty_groups_for_shutdown() {
        if (!::FarLib::ec_benchmark_phase::enabled()) return;
        const size_t count = static_cast<size_t>(
            stripe_count_.load(std::memory_order_acquire));
        for (size_t i = 0; i < count && i < stripe_capacity_; i++) {
            Stripe *stripe = stripe_by_id(i);
            if (stripe == nullptr || stripe->group_state_words.empty()) {
                continue;
            }
            std::lock_guard<std::mutex> lock(stripe->mutex);
            for (uint32_t slot = 0; slot < stripe->slots_per_shard; slot++) {
                if (group_state_locked(*stripe, slot) != kSlotGroupSealed ||
                    slot >= stripe->group_live_mask.size() ||
                    stripe->group_live_mask[slot] != 0 ||
                    slot >= stripe->group_is_split.size() ||
                    stripe->group_is_split[slot] != 0 ||
                    group_update_pending_locked(*stripe, slot)) {
                    continue;
                }
                SlotGroupId id;
                id.stripe_id = stripe->stripe_id;
                id.slot_id = slot;
                (void)mark_dead_group_locked(*stripe, id);
            }
        }
    }

    // ------------------------------------------------- per-object release --
    // Result of an address-based per-object release.
    enum GroupReleaseResult : uint8_t {
        // The address is not owned by a slot group: the caller must fall back
        // to mark_dead(), i.e. exactly the pre-group behaviour.
        kGroupReleaseNotGrouped = 0,
        // The address named a live object of a sealed group that still holds
        // other objects.  Only that one data segment was released; the group
        // keeps owning all four of its slots, so none of them becomes singly
        // allocatable.
        kGroupReleaseObjectReleased = 1,
        // The address named the group's last live object: the whole group is
        // dead now and can be reused as a group.
        kGroupReleaseGroupReleased = 2,
        // The address lies inside a slot group but does not name a live object
        // of it (double release, a zero-filled hole, a not yet sealed group, an
        // already dead group).  Nothing was modified.
        kGroupReleaseRejected = 3,
    };

    // Group-aware counterpart of mark_dead(): releases one object addressed by
    // the data-segment address it lives at.  The remote allocator dispatches on
    // this first and only falls back to mark_dead() for
    // kGroupReleaseNotGrouped.
    //
    // Invariants: the group's four data slots stay reserved (dead_bits /
    // free_bits are never touched here, so the single-object path cannot see
    // them), a sealed group is only released as a whole, and a group only
    // becomes reusable through the group allocator.  `remaining_out`, when
    // given, receives the group's live object count after the release.
    GroupReleaseResult release_group_object(uint64_t addr,
                                            uint32_t *remaining_out = nullptr) {
        if (remaining_out != nullptr) *remaining_out = 0;
        DecodedBinding binding;
        if (!binding_for_addr(addr, &binding)) {
            return kGroupReleaseNotGrouped;
        }
        Stripe *stripe = stripe_by_id(binding.stripe_id);
        if (stripe == nullptr || binding.shard_idx >= kDataShards) {
            return kGroupReleaseNotGrouped;
        }
        // A stripe that never served a group has no group state at all, so the
        // address is a plain single-object slot.  has_groups is published
        // before the first group slot address can exist, so this cannot turn a
        // group address into a single-object one.
        if (!stripe->has_groups.load(std::memory_order_acquire)) {
            return kGroupReleaseNotGrouped;
        }

        SlotGroupId id;
        id.stripe_id = binding.stripe_id;
        bool split_group_released = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            uint64_t base = stripe->shard_base[binding.shard_idx];
            if (base == kInvalidRemoteAddr || addr < base) {
                return kGroupReleaseNotGrouped;
            }
            uint64_t offset = addr - base;
            if (stripe->slot_size == 0 || (offset % stripe->slot_size) != 0) {
                return kGroupReleaseNotGrouped;
            }
            uint64_t slot64 = offset / stripe->slot_size;
            if (slot64 >= stripe->slots_per_shard) {
                return kGroupReleaseNotGrouped;
            }
            const uint32_t slot = static_cast<uint32_t>(slot64);
            id.slot_id = slot;
            const uint8_t state = group_state_locked(*stripe, slot);
            if (state == kSlotGroupFree) {
                // No group ever owned this offset: a plain single-object slot.
                return kGroupReleaseNotGrouped;
            }
            if (state == kSlotGroupInProgress) {
                // An in-progress group has not published a sealed live-mask;
                // releasing one of its addresses must not silently turn it
                // into a reusable/dead group.  The write/refcount owner must
                // settle the group through the normal sealed path first.
                return kGroupReleaseRejected;
            } else if (state != kSlotGroupSealed) {
                // Already dead: the object was released before (double
                // release) or the whole group was.
                return kGroupReleaseRejected;
            } else {
                // A split group has one public owner at data shard 0. Never
                // release an internal fragment as an independent object.
                if (slot < stripe->group_is_split.size() &&
                    stripe->group_is_split[slot] != 0) {
                    const uint64_t anchor =
                        stripe->shard_base[0] +
                        static_cast<uint64_t>(slot) * stripe->slot_size;
                    if (binding.shard_idx != 0 || addr != anchor) {
                        return kGroupReleaseRejected;
                    }
                    if (!mark_dead_group_locked(*stripe, id)) {
                        return kGroupReleaseRejected;
                    }
                    split_group_released = true;
                } else {
                const uint8_t bit =
                    static_cast<uint8_t>(1u << binding.shard_idx);
                if ((stripe->group_live_mask[slot] & bit) == 0) {
                    // Zero-filled hole or a second release of the same object.
                    return kGroupReleaseRejected;
                }
                assert(slot < stripe->group_live_objects.size());
                assert(stripe->group_live_objects[slot] > 0);
                const uint8_t prior_live = stripe->group_live_objects[slot];
                const uint32_t remaining = static_cast<uint32_t>(prior_live - 1);
                const bool update_pending =
                    group_update_pending_locked(*stripe, slot);
                if (remaining == 0) {
                    if (update_pending) {
                        // Keep the sealed reservation alive while the pending
                        // target is being prepared.  abort_slot_reuse() or
                        // commit_slot_reuse() will settle the empty-old-live
                        // case after this release has removed its metadata.
                        account_group_object_release_locked(
                            *stripe, slot, binding.shard_idx);
                        stripe->group_live_mask[slot] = static_cast<uint8_t>(
                            stripe->group_live_mask[slot] & ~bit);
                        stripe->group_live_objects[slot] = 0;
                        assert(stripe->live_count > 0);
                        stripe->live_count--;
                        assert(stripe->group_counts_by_live[prior_live] > 0);
                        stripe->group_counts_by_live[prior_live]--;
                        stripe->group_counts_by_live[0]++;
                        if (!::FarLib::ec_benchmark_phase::enabled()) {
                            stripe->reuse_candidate.store(
                                true, std::memory_order_release);
                        } else {
                            update_reuse_index_locked(*stripe, slot);
                        }
                        if (remaining_out != nullptr) *remaining_out = 0;
                        return kGroupReleaseObjectReleased;
                    }
                    // In phased benchmark mode, preserve an empty sealed
                    // non-split group as an RMW target.  Its sealed/parity
                    // accounting, group_counts_by_live[0], holes and live
                    // count remain coherent; only object payload ownership is
                    // removed.  Legacy modes still retire the group below.
                    const bool retain_empty =
                        ::FarLib::ec_benchmark_phase::enabled() &&
                        slot < stripe->group_is_split.size() &&
                        stripe->group_is_split[slot] == 0;
                    if (retain_empty) {
                        account_group_object_release_locked(
                            *stripe, slot, binding.shard_idx);
                        stripe->group_live_mask[slot] = static_cast<uint8_t>(
                            stripe->group_live_mask[slot] & ~bit);
                        stripe->group_live_objects[slot] = 0;
                        assert(stripe->live_count > 0);
                        stripe->live_count--;
                        assert(stripe->group_counts_by_live[prior_live] > 0);
                        stripe->group_counts_by_live[prior_live]--;
                        stripe->group_counts_by_live[0]++;
                        update_reuse_index_locked(*stripe, slot);
                        if (remaining_out != nullptr) *remaining_out = 0;
                        return kGroupReleaseObjectReleased;
                    }
                    // Retire the sealed group while its mask/payload metadata
                    // still describe the final live object.  This keeps the
                    // structural and payload counters balanced in one lock.
                    if (!mark_dead_group_locked(*stripe, id)) {
                        return kGroupReleaseRejected;
                    }
                    split_group_released = true;
                }
                if (!split_group_released) {
                    account_group_object_release_locked(*stripe, slot,
                                                         binding.shard_idx);
                    stripe->group_live_mask[slot] = static_cast<uint8_t>(
                        stripe->group_live_mask[slot] & ~bit);
                    stripe->group_live_objects[slot] =
                        static_cast<uint8_t>(remaining);
                    assert(remaining ==
                           static_cast<uint32_t>(__builtin_popcount(
                               static_cast<unsigned>(
                                   stripe->group_live_mask[slot]))));
                    assert(stripe->live_count > 0);
                    stripe->live_count--;
                    assert(stripe->group_counts_by_live[prior_live] > 0);
                    stripe->group_counts_by_live[prior_live]--;
                    stripe->group_counts_by_live[remaining]++;
                    if (!::FarLib::ec_benchmark_phase::enabled()) {
                        stripe->reuse_candidate.store(
                            true, std::memory_order_release);
                    } else {
                        update_reuse_index_locked(*stripe, slot);
                    }
                    if (remaining_out != nullptr) *remaining_out = remaining;
                    return kGroupReleaseObjectReleased;
                }
                }
            }
        }

        const bool group_dead = split_group_released || mark_dead_group(id);
        // False only if the same group was already released as a whole by a
        // concurrent path; the re-queue below is harmless then.
        if (!group_dead) return kGroupReleaseRejected;
        // A dead group means the stripe can serve a group again, so re-queue it
        // exactly like allocate_slot_group() does after an allocation: without
        // this the released slots would be unreachable and the group pool would
        // keep creating fresh stripes.
        requeue_group_stripe(*stripe);
        return kGroupReleaseGroupReleased;
    }

    // Settled-address query used by the remote allocator's per-object release:
    // true when the slot group this address belongs to holds no object any
    // more - it reached the terminal state dead, or (safety net) it is a
    // sealed group whose last live object is already back.  A
    // kGroupReleaseRejected answer for such an address means "the group is
    // already settled, this release has nothing to do" and not "invalid
    // input"; an in_progress group is never settled, because the group
    // allocator is still filling it.
    bool slot_group_addr_is_settled(uint64_t addr) const {
        DecodedBinding binding;
        if (!binding_for_addr(addr, &binding)) return false;
        const Stripe *stripe = stripe_by_id(binding.stripe_id);
        if (stripe == nullptr || binding.shard_idx >= kDataShards) return false;
        if (!stripe->has_groups.load(std::memory_order_acquire)) return false;
        std::lock_guard<std::mutex> lock(stripe->mutex);
        const uint64_t base = stripe->shard_base[binding.shard_idx];
        if (base == kInvalidRemoteAddr || addr < base) return false;
        const uint64_t offset = addr - base;
        if (stripe->slot_size == 0 || (offset % stripe->slot_size) != 0) {
            return false;
        }
        const uint64_t slot64 = offset / stripe->slot_size;
        if (slot64 >= stripe->slots_per_shard) return false;
        const uint32_t slot = static_cast<uint32_t>(slot64);
        const uint8_t state = group_state_locked(*stripe, slot);
        if (state == kSlotGroupDead) return true;
        if (state == kSlotGroupSealed) {
            if (group_update_pending_locked(*stripe, slot)) return false;
            return stripe->group_live_objects[slot] == 0;
        }
        return false;
    }

    // Group-level layout accessor: all six segments as
    // (endpoint_idx, offset, slot_size) plus their absolute addresses in one
    // call.  Valid for any non-free state; a dead group keeps its addresses
    // until the offset is reused by a new group.
    bool get_slot_group_layout(const SlotGroupId &id,
                               SlotGroupHandle *group_out) const {
        if (group_out == nullptr) return false;
        const Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) == kSlotGroupFree) {
            return false;
        }
        return fill_group_handle(*stripe, stripe->bin, id.slot_id, group_out);
    }

    // Current group state plus (for sealed groups) the 4-bit live mask.
    bool get_slot_group_state(const SlotGroupId &id, SlotGroupState *state_out,
                              uint8_t *live_mask_out = nullptr) const {
        const Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        uint8_t state = group_state_locked(*stripe, id.slot_id);
        if (state_out != nullptr) {
            *state_out = static_cast<SlotGroupState>(state);
        }
        if (live_mask_out != nullptr) {
            *live_mask_out = (state == kSlotGroupSealed &&
                              id.slot_id < stripe->group_live_mask.size())
                                 ? stripe->group_live_mask[id.slot_id]
                                 : 0;
        }
        return true;
    }
    // True only for a live/in-progress split group at a data-segment address.
    bool slot_group_is_split(uint64_t addr) const {
        DecodedBinding binding;
        if (!binding_for_addr(addr, &binding) ||
            binding.shard_idx >= kDataShards) {
            return false;
        }
        const Stripe *stripe = stripe_by_id(binding.stripe_id);
        if (stripe == nullptr ||
            !stripe->has_groups.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        const uint64_t base = stripe->shard_base[binding.shard_idx];
        if (base == kInvalidRemoteAddr || addr < base ||
            stripe->slot_size == 0) {
            return false;
        }
        const uint64_t offset = addr - base;
        if ((offset % stripe->slot_size) != 0) return false;
        const uint64_t slot64 = offset / stripe->slot_size;
        if (slot64 >= stripe->slots_per_shard ||
            slot64 >= stripe->group_is_split.size()) {
            return false;
        }
        return group_state_locked(*stripe, static_cast<uint32_t>(slot64)) !=
                   kSlotGroupFree &&
               stripe->group_is_split[static_cast<size_t>(slot64)] != 0;
    }

    // Live internal-segment counter of one group. For a split object it is
    // four fragments for one public owner.
    bool get_slot_group_object_count(const SlotGroupId &id,
                                     uint32_t *count_out) const {
        const Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (count_out != nullptr) {
            *count_out = (id.slot_id < stripe->group_live_objects.size())
                             ? stripe->group_live_objects[id.slot_id]
                             : 0;
        }
        return true;
    }

    // O(stripes) accounting snapshot.  In the exact/quiescent mode no stripe
    // mutates while this method runs, so the result is a coherent total.  If
    // allocation/release continues concurrently, each stripe is copied while
    // holding only that stripe's existing mutex; the returned total is a
    // rolling per-stripe snapshot (not a globally linearizable instant).
    SpaceUsage space_usage() const {
        SpaceUsage result;
        size_t endpoint_count =
            ::FarLib::allocator::remote::remote_global_heap.endpoint_count();
        if (endpoint_count == 0) {
            const int configured = ::FarLib::get_config().server_count;
            endpoint_count = configured > 0 ? static_cast<size_t>(configured) : 0;
        }
        result.endpoint_occupied_bytes.assign(endpoint_count, 0);

        const size_t count = static_cast<size_t>(
            stripe_count_.load(std::memory_order_acquire));
        for (size_t i = 0; i < count && i < stripe_capacity_; i++) {
            const Stripe *stripe = stripe_by_id(i);
            if (stripe == nullptr) continue;
            std::lock_guard<std::mutex> lock(stripe->mutex);
            result.reserved_stripe_bytes +=
                static_cast<uint64_t>(kShardCount) * shard_table_.shard_size();
            result.occupied_group_bytes += stripe->occupied_group_bytes;
            result.sealed_group_bytes += stripe->sealed_group_bytes;
            result.live_payload_bytes += stripe->live_payload_bytes;
            result.live_data_slot_bytes += stripe->live_data_slot_bytes;
            result.parity_bytes += stripe->parity_bytes;
            result.trapped_hole_bytes += stripe->trapped_hole_bytes;
            result.padding_bytes += stripe->padding_bytes;
            result.split_groups += stripe->split_groups;
            result.in_progress_groups += stripe->in_progress_groups;
            result.reusable_groups += stripe->reusable_groups;
            result.reusable_group_bytes +=
                stripe->reusable_groups * group_reserved_bytes(*stripe);
            result.unknown_payload_objects += stripe->unknown_payload_objects;
            result.sealed_small_groups += stripe->sealed_small_groups;
            for (size_t live = 0; live < result.group_counts_by_live.size();
                 live++) {
                result.group_counts_by_live[live] +=
                    stripe->group_counts_by_live[live];
            }

            // Endpoint occupancy is derived from active groups only.  Dead
            // groups are reusable reservations and are intentionally excluded;
            // including both their dead reservation and the next reuse would
            // double count one physical six-segment offset.
            const uint64_t group_bytes = group_reserved_bytes(*stripe);
            const uint64_t active_groups =
                group_bytes == 0 ? 0 : stripe->occupied_group_bytes / group_bytes;
            for (uint8_t shard = 0; shard < kShardCount; shard++) {
                const uint32_t endpoint = stripe->shard_endpoint[shard];
                if (endpoint >= result.endpoint_occupied_bytes.size()) continue;
                result.endpoint_occupied_bytes[endpoint] +=
                    active_groups * stripe->slot_size;
            }

        }
        result.live_slot_bytes = result.live_data_slot_bytes;
        return result;
    }

    size_t stripe_count() const {
        return stripe_count_.load(std::memory_order_acquire);
    }

    bool get_stripe_counts(uint64_t stripe_id, uint32_t *live, uint32_t *dead,
                           uint32_t *free_count) const {
        const Stripe *stripe = stripe_by_id(stripe_id);
        if (stripe == nullptr) return false;
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (live) *live = stripe->live_count;
        if (dead) *dead = stripe->dead_count;
        if (free_count) *free_count = stripe->free_count;
        return true;
    }
};

// ---------------------------------------------------------------------------
// RS(4,2) parity encoder / shard rebuild (phase 2).
//
// The GF(2^8) core (primitive polynomial 0x1d, G1's generator matrix, G1's
// 8/16-bit parity LUTs and G1's byte loops) lives in
// include/cache/alloc/small_object_stripe_codec.hpp.  This class is the
// shard-level facade the stripe manager and the self-check use.
//
// Shard representation: one shard is one SmallObjectStripeShardSize
// (== RegionSize, 256 KiB) contiguous buffer, i.e. exactly what
// Stripe::shard_base and SlotLayout::shard_base / SlotLayout::parity_shard_base
// describe for remote memory.  Every entry point takes a (shard_offset,
// byte_count) window, so only the dirty / missing part of a shard has to be
// touched instead of rewriting whole shards.
//
// The write path and the ACK path are still not wired to this class: nothing in
// the runtime calls it yet, so ft_method == none is unaffected.
// ---------------------------------------------------------------------------
class SmallObjectStripeEncoder {
public:
    static constexpr uint8_t kDataShards = kSmallObjectStripeDataShards;
    static constexpr uint8_t kParityShards = kSmallObjectStripeParityShards;
    static constexpr uint8_t kShardCount = kSmallObjectStripeShardCount;
    static constexpr size_t kShardSize = SmallObjectStripeShardSize;

    static_assert(kStripeCodecDataShards == kSmallObjectStripeDataShards &&
                      kStripeCodecParityShards == kSmallObjectStripeParityShards,
                  "the codec geometry and the stripe geometry must agree");

    // Parity encoding and shard rebuild are implemented (phase 2); the write
    // path/ACK that would consume them is still phase 3.
    static constexpr bool implemented() { return true; }

    // Encode the two parity shards of one stripe from its four data shards:
    //   parity[p][i] = sum_j coef[p][j] * data[j][i]      (GF(2^8), poly 0x1d)
    // with G1's coefficient table (parity0 = XOR of the four data shards,
    // parity1 = sum_j 2^j * data_j).  byte_count defaults to a whole shard.
    static bool encode(const void *const data_shards[kDataShards],
                       void *const parity_shards[kParityShards],
                       size_t byte_count = kShardSize, size_t shard_offset = 0) {
        return small_object_stripe_encode_shards(data_shards, parity_shards,
                                                byte_count, shard_offset);
    }

    // Rebuild the missing shards of one stripe in place.  shards[0..3] are the
    // data shards and shards[4..5] the parity shards, indexed exactly like
    // Stripe::shard_base / SlotLayout; alive_mask says which of them are alive
    // (readable).  The 1 or 2 missing shards named by the complement of
    // alive_mask are written in place, so all six pointers must be valid.  Works
    // for all 6 single-missing and all 15 double-missing sets.
    static bool rebuild(uint8_t alive_mask, void *const shards[kShardCount],
                        size_t byte_count = kShardSize, size_t shard_offset = 0) {
        return small_object_stripe_rebuild_shards(alive_mask, shards, byte_count,
                                                 shard_offset);
    }

    // Rebuild one missing shard from four explicitly listed survivors (repair
    // path: one dead endpoint, four live ones).
    static bool rebuild_one(
        uint8_t missing_shard_idx, const uint8_t survivor_idx[kDataShards],
        const void *const survivors[kDataShards], void *dst,
        size_t byte_count = kShardSize, size_t shard_offset = 0) {
        return small_object_stripe_rebuild_one(
            missing_shard_idx, survivor_idx, survivors, dst, byte_count,
            shard_offset);
    }
};

}  // namespace FarLib::cache
