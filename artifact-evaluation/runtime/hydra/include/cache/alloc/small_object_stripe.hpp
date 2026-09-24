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
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "cache/alloc/region_remote_allocator.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"

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
    };

    struct Stripe {
        uint64_t stripe_id = kInvalidStripeId;
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
        uint32_t group_dead_count = 0;  // dead groups, reusable as a whole
        std::array<uint32_t, kShardCount> shard_endpoint{};
        std::vector<uint64_t> group_state_words;
        std::vector<uint8_t> group_live_mask;
        // Live *object* counter per slot_id: how many of the group's four data
        // segments still hold an object.  Maintained together with
        // group_live_mask (count == popcount(mask)) by seal_slot_group /
        // release_group_object / mark_dead_group.
        std::vector<uint8_t> group_live_objects;
        // A split object owns all four data fragments as one public object.
        // The live mask remains 0xf; this metadata distinguishes split
        // ownership from an ordinary batch with holes.
        std::vector<uint8_t> group_is_split;
        // Set (never cleared) when group metadata is materialised for this
        // stripe, so the group-aware release can tell "this stripe never served
        // a group" without taking the stripe mutex.
        std::atomic<bool> has_groups{false};
        std::array<uint64_t, kShardCount> shard_base{};
        std::array<Bitmap, kDataShards> free_bits;
        std::array<Bitmap, kDataShards> dead_bits;
        mutable std::mutex mutex;

        Stripe() {
            shard_base.fill(kInvalidRemoteAddr);
            shard_endpoint.fill(std::numeric_limits<uint32_t>::max());
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
    std::array<SizeClassPool, ::FarLib::allocator::RegionBinCount> pools_;
    // Group allocations use their own candidate pool: group stripes stay out
    // of the single-object pool so the two paths cannot hand out the same
    // slot offset.
    std::array<SizeClassPool, ::FarLib::allocator::RegionBinCount> group_pools_;
    std::mutex stripes_mutex_;
    std::atomic<uint64_t> next_endpoint_{0};
    // Endpoint liveness lives in RemoteGlobalHeap and is shared with ordinary
    // flat allocation.  A dead endpoint never changes the addresses already
    // published by a stripe; it only filters future placement and reusable
    // group-pool candidates here.
    inline static thread_local LocalLeases local_leases_;

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
        stripe.next_group_slot = 0;
        stripe.group_dead_count = 0;
        stripe.group_live_objects.assign(stripe.slots_per_shard, 0);
        stripe.group_is_split.assign(stripe.slots_per_shard, 0);
        // Published before the first group slot address can exist: the release
        // path uses it to skip the group lookup for stripes that never served a
        // group, and it is never cleared again.
        stripe.has_groups.store(true, std::memory_order_release);
        return true;
    }

    // Invariants 1 and 2: a candidate offset is only taken when all four data
    // shards have that offset free *at the same time*; the four free bits are
    // then cleared together inside this single critical section.  A dead group
    // already has its four slots reserved, so it is reused as a whole.
    bool allocate_group_from_stripe_locked(Stripe &stripe,
                                           uint32_t *slot_id_out,
                                           bool *from_dead_out = nullptr) {
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
            stripe.group_live_objects[slot] = 0;
            stripe.group_is_split[slot] = 0;
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
        // A pre-failure stripe may remain valid for recovery, but after the
        // standby is activated it must not be recycled as a new group.  This
        // preserves its old physical addresses while ensuring all newly
        // allocated groups include the spare.
        return !standby_active || has_standby;
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
    Stripe *create_stripe(size_t bin,
                          bool require_distinct_endpoints = false) {
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

        // Shards of one stripe live on six distinct live memory servers.  The
        // endpoint selector excludes a configured standby until failure and
        // includes it in every new post-failure stripe.
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            size_t endpoint = endpoints[shard];
            uint64_t base = allocate_region_for_endpoint(endpoint);
            if (base == kInvalidRemoteAddr) {
                release_allocated_shards(shard_base);
                return nullptr;
            }
            shard_base[shard] = base;
            shard_endpoint[shard] = static_cast<uint32_t>(endpoint);
        }

        auto stripe = std::make_unique<Stripe>();
        stripe->stripe_id = 0;
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
        stripe_id = stripe_count_.load(std::memory_order_relaxed);
        if (stripe_id >= stripe_capacity_) {
            release_allocated_shards(shard_base);
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
        stripe_storage_.clear();
        stripe_storage_.reserve(stripe_capacity_);
        stripe_count_.store(0, std::memory_order_relaxed);
        next_endpoint_.store(0, std::memory_order_relaxed);
    }

    bool enabled() const { return shard_table_.enabled(); }

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
    bool allocate_slot_group(size_t size, SlotGroupHandle *group_out) {
        if (group_out == nullptr) return false;
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

        Stripe *stripe = create_stripe(bin, true);
        if (stripe == nullptr) return false;
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

    // free -> in_progress -> sealed.  live_mask is a 4-bit mask (bit i = data
    // shard i); any value below 16 is accepted, including fewer than four live
    // objects.  Slots that are not live hold no object and their shard content
    // is defined to be zero (the writer guarantees it).  The group keeps owning
    // all four data slots, live or not, so they are never handed out singly.
    bool seal_slot_group(const SlotGroupId &id, uint8_t live_mask) {
        if ((live_mask & 0xf0u) != 0) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) != kSlotGroupInProgress) {
            return false;
        }
        uint32_t live = static_cast<uint32_t>(
            __builtin_popcount(static_cast<unsigned>(live_mask)));
        assert(stripe->live_count >= kDataShards - live);
        stripe->live_count -= (kDataShards - live);
        stripe->group_live_mask[id.slot_id] = live_mask;
        stripe->group_live_objects[id.slot_id] = static_cast<uint8_t>(live);
        set_group_state_locked(*stripe, id.slot_id, kSlotGroupSealed);
        return true;
    }

    // Seal one large-object split group. All four data fragments are valid
    // internal shards, while data[0] remains the sole public owner.
    bool seal_split_slot_group(const SlotGroupId &id) {
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) != kSlotGroupInProgress ||
            id.slot_id >= stripe->group_is_split.size()) {
            return false;
        }
        stripe->group_live_mask[id.slot_id] = 0xfu;
        stripe->group_live_objects[id.slot_id] = kDataShards;
        stripe->group_is_split[id.slot_id] = 1;
        set_group_state_locked(*stripe, id.slot_id, kSlotGroupSealed);
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
        uint32_t live = kDataShards;
        if (state == kSlotGroupSealed) {
            live = static_cast<uint32_t>(__builtin_popcount(
                static_cast<unsigned>(stripe.group_live_mask[id.slot_id])));
        }
        assert(stripe.live_count >= live);
        stripe.live_count -= live;
        stripe.group_live_mask[id.slot_id] = 0;
        stripe.group_live_objects[id.slot_id] = 0;
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
                stripe->group_live_mask[slot] =
                    static_cast<uint8_t>(stripe->group_live_mask[slot] & ~bit);
                assert(slot < stripe->group_live_objects.size());
                assert(stripe->group_live_objects[slot] > 0);
                const uint32_t remaining =
                    static_cast<uint32_t>(--stripe->group_live_objects[slot]);
                assert(remaining ==
                       static_cast<uint32_t>(__builtin_popcount(
                           static_cast<unsigned>(stripe->group_live_mask[slot]))));
                assert(stripe->live_count > 0);
                stripe->live_count--;
                if (remaining != 0) {
                    if (remaining_out != nullptr) *remaining_out = remaining;
                    return kGroupReleaseObjectReleased;
                }
                // Last live object: the whole group is released below, outside
                // this critical section, because mark_dead_group() takes the
                // same mutex.  The group is still sealed here, so no allocator
                // can pick it up in between and no second thread can observe
                // the last object.
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
