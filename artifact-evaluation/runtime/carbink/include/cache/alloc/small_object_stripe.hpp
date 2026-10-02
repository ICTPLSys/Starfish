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

// A stable snapshot of one original stripe while its failed physical shard is
// reconstructed out of place.  The logical addresses remain immutable; the
// manager publishes a sidecar base only after the replacement WRITE completes.
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
    std::vector<uint32_t> live_slot_ids;

    BackgroundRebuildStripe() {
        original_base.fill(::FarLib::allocator::remote::InvalidRemoteAddr);
        endpoint.fill(std::numeric_limits<uint32_t>::max());
    }
};

struct BackgroundRebuildSet {
    uint64_t stripe_id = std::numeric_limits<uint64_t>::max();
    uint32_t slot_size = 0;
    uint64_t shard_bytes = 0;
    std::array<uint64_t, kSmallObjectStripeShardCount> original_base{};
    std::array<uint32_t, kSmallObjectStripeShardCount> endpoint{};
    uint8_t missing_mask = 0;
    uint64_t live_groups = 0;
    uint64_t live_objects = 0;
    std::vector<uint32_t> live_slot_ids;
    BackgroundRebuildSet() {
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
        uint64_t next_group_generation = 1;
        uint32_t group_dead_count = 0;  // dead groups, reusable as a whole
        std::array<uint32_t, kShardCount> shard_endpoint{};
        std::vector<uint64_t> group_state_words;
        std::vector<uint8_t> group_live_mask;
        // Live *object* counter per slot_id: how many of the group's four data
        // segments still hold an object.  Maintained together with
        // group_live_mask (count == popcount(mask)) by seal_slot_group /
        // release_group_object / mark_dead_group.
        std::vector<uint8_t> group_live_objects;
        // A write reservation spans group allocation through its terminal
        // CQE disposition.  Background scans treat this as Busy even after
        // the group has sealed, while the normal path leaves it zero.
        std::vector<uint8_t> group_pending_write;
        // A split object owns all four data fragments as one public object.
        // The live mask remains 0xf; this metadata distinguishes split
        // ownership from an ordinary batch with holes.
        std::vector<uint8_t> group_ready_after_write;
        std::vector<uint64_t> group_generation;
        std::vector<std::array<uintptr_t, kDataShards>> group_owners;
        std::vector<uint8_t> group_is_split;
        // Set (never cleared) when group metadata is materialised for this
        // stripe, so the group-aware release can tell "this stripe never served
        // a group" without taking the stripe mutex.
        std::atomic<bool> has_groups{false};
        // A claimed group is reserved by remote compaction while its owners
        // and six remote segments are in flight.  Claims are checked by both
        // allocation and release-side requeue paths.
        std::vector<uint8_t> group_compaction_claimed;
        std::vector<uint8_t> group_shadow_quarantined;
        // Observer-only mirror of the exact states counted by
        // live_group_bytes_by_endpoint(): in-progress/sealed and
        // dead+compaction-claimed. Writers update it under stripe.mutex.
        std::atomic<uint64_t> observer_reserved_group_count{0};
        std::array<uint64_t, kShardCount> shard_base{};
        // A rebuilt physical shard is published here only after its complete
        // replacement codeword has been written.  Logical addresses and their
        // generation/owner metadata never move.
        std::array<std::atomic<uint64_t>, kShardCount> rebuilt_base;
        // Shadow stripes are out-of-place compaction targets.  They use a
        // separate per-layout pool and are never ordinary allocation sources.
        bool shadow_only = false;
        bool shadow_quarantined = false;
        bool shadow_regions_released = false;
        std::array<uint32_t, kShardCount> shadow_layout{};
        std::array<Bitmap, kDataShards> free_bits;
        std::array<Bitmap, kDataShards> dead_bits;
        mutable std::mutex mutex;

        Stripe() {
            shard_base.fill(kInvalidRemoteAddr);
            shard_endpoint.fill(std::numeric_limits<uint32_t>::max());
            shadow_layout.fill(std::numeric_limits<uint32_t>::max());
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
        uint64_t generation = 0;
        uint16_t bin = 0;
        uint32_t slot_size = 0;
        uint32_t slots_per_shard = 0;
        uint64_t slot_offset = 0;
        std::array<SlotGroupSegment, kShardCount> segments{};
    };
    // Stable snapshot returned while a sealed group is claimed by a
    // compaction worker.  The handle may be used after the stripe mutex is
    // released as long as the claim remains held.
    struct CompactionGroupView {
        SlotGroupId id;
        SlotGroupHandle group;
        SlotGroupHandle handle;
        uint8_t live_mask = 0;
        uint8_t live_count = 0;
        uint64_t generation = 0;
        std::array<uintptr_t, kDataShards> owners{};
        bool shadow_only = false;
    };
private:
    struct SizeClassPool {
        std::mutex mutex;
        std::vector<uint64_t> stripes_with_space;
    };

    struct ShadowLayoutPool {
        std::array<uint32_t, kShardCount> layout{};
        SizeClassPool groups;
        size_t created_stripes = 0;
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
    mutable std::mutex stripes_mutex_;
    std::mutex shadow_layout_pools_mutex_;
    std::vector<std::unique_ptr<ShadowLayoutPool>> shadow_layout_pools_;
    static constexpr size_t kMaxShadowLayoutPools = 32;
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

    void release_published_background_targets_locked() {
        for (auto &stripe : stripe_storage_) {
            if (stripe == nullptr) continue;
            for (uint8_t shard = 0; shard < kShardCount; shard++) {
                const uint64_t target = stripe->rebuilt_base[shard].exchange(
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

    ShadowLayoutPool *shadow_layout_pool_for(
        const std::array<uint32_t, kShardCount> &layout, bool create) {
        std::lock_guard<std::mutex> lock(shadow_layout_pools_mutex_);
        for (auto &candidate : shadow_layout_pools_) {
            if (candidate != nullptr && candidate->layout == layout) {
                return candidate.get();
            }
        }
        if (!create || shadow_layout_pools_.size() >= kMaxShadowLayoutPools) {
            return nullptr;
        }
        auto candidate = std::make_unique<ShadowLayoutPool>();
        candidate->layout = layout;
        ShadowLayoutPool *result = candidate.get();
        shadow_layout_pools_.push_back(std::move(candidate));
        return result;
    }

    uint64_t pop_shadow_candidate(
        const std::array<uint32_t, kShardCount> &layout) {
        ShadowLayoutPool *pool = shadow_layout_pool_for(layout, false);
        if (pool == nullptr) return kInvalidStripeId;
        std::lock_guard<std::mutex> lock(pool->groups.mutex);
        if (pool->groups.stripes_with_space.empty()) {
            return kInvalidStripeId;
        }
        const uint64_t stripe_id = pool->groups.stripes_with_space.back();
        pool->groups.stripes_with_space.pop_back();
        return stripe_id;
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

    static bool observer_reserved_locked(const Stripe &stripe, uint32_t slot,
                                         uint8_t state) {
        return state == kSlotGroupInProgress ||
               state == kSlotGroupSealed ||
               (state == kSlotGroupDead &&
                slot < stripe.group_compaction_claimed.size() &&
                stripe.group_compaction_claimed[slot] != 0);
    }

    static void observer_adjust_reserved_locked(Stripe &stripe, bool before,
                                                bool after) {
        if (!before && after) {
            stripe.observer_reserved_group_count.fetch_add(
                1, std::memory_order_relaxed);
        } else if (before && !after) {
            const uint64_t old = stripe.observer_reserved_group_count.fetch_sub(
                1, std::memory_order_relaxed);
            assert(old > 0);
        }
    }

    static void set_group_state_locked(Stripe &stripe, uint32_t slot,
                                       uint8_t state) {
        const uint8_t previous = group_state_locked(stripe, slot);
        const bool counted_before =
            observer_reserved_locked(stripe, slot, previous);
        size_t bit = group_state_bit(slot);
        uint64_t &word = stripe.group_state_words[bit / kBitsPerWord];
        size_t shift = bit % kBitsPerWord;
        word = (word & ~(kGroupStateMask << shift)) |
               ((static_cast<uint64_t>(state) & kGroupStateMask) << shift);
        observer_adjust_reserved_locked(
            stripe, counted_before,
            observer_reserved_locked(stripe, slot, state));
    }

    static bool group_owns_slot_locked(const Stripe &stripe, uint32_t slot) {
        return group_state_locked(stripe, slot) != kSlotGroupFree;
    }
    static bool group_compaction_claimed_locked(const Stripe &stripe,
                                                uint32_t slot) {
        return slot < stripe.group_compaction_claimed.size() &&
               stripe.group_compaction_claimed[slot] != 0;
    }

    static void set_group_compaction_claimed_locked(
        Stripe &stripe, uint32_t slot, bool claimed) {
        if (slot >= stripe.group_compaction_claimed.size()) return;
        const uint8_t state = group_state_locked(stripe, slot);
        const bool counted_before =
            observer_reserved_locked(stripe, slot, state);
        stripe.group_compaction_claimed[slot] = claimed ? 1 : 0;
        observer_adjust_reserved_locked(
            stripe, counted_before,
            observer_reserved_locked(stripe, slot, state));
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
        if (stripe.group_live_mask.size() < stripe.slots_per_shard ||
            stripe.group_live_objects.size() < stripe.slots_per_shard ||
            stripe.group_pending_write.size() < stripe.slots_per_shard) {
            return false;
        }
        const bool background_enabled =
            ::FarLib::get_config().ft_background_rebuild;
        for (uint32_t slot = 0; slot < stripe.slots_per_shard; slot++) {
            const uint8_t state = group_state_locked(stripe, slot);
            if (state == kSlotGroupInProgress ||
                (background_enabled && stripe.group_pending_write[slot] != 0)) {
                *busy = true;
            }
            if (state != kSlotGroupSealed) continue;
            const uint8_t live =
                static_cast<uint8_t>(stripe.group_live_mask[slot] & 0x0fu);
            if (live == 0) continue;
            live_slots->push_back(slot);
            ++(*live_groups);
            *live_objects += stripe.group_live_objects[slot];
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
        stripe.group_pending_write.assign(stripe.slots_per_shard, 0);
        stripe.group_is_split.assign(stripe.slots_per_shard, 0);
        stripe.next_group_generation = 1;
        stripe.group_ready_after_write.assign(stripe.slots_per_shard, 0);
        stripe.group_generation.assign(stripe.slots_per_shard, 0);
        stripe.group_owners.resize(stripe.slots_per_shard);
        for (auto &owners : stripe.group_owners) owners.fill(0);
        stripe.group_compaction_claimed.assign(stripe.slots_per_shard, 0);
        stripe.group_shadow_quarantined.assign(stripe.slots_per_shard, 0);
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
            if (group_compaction_claimed_locked(stripe, slot)) continue;
            if (::FarLib::get_config().ft_background_rebuild &&
                slot < stripe.group_pending_write.size() &&
                stripe.group_pending_write[slot] != 0) {
                continue;
            }
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
            uint64_t generation = stripe.next_group_generation++;
            if (generation == 0) generation = stripe.next_group_generation++;
            if (stripe.next_group_generation == 0) {
                stripe.next_group_generation = 1;
            }
            stripe.group_generation[slot] = generation;
            stripe.group_ready_after_write[slot] = 0;
            stripe.group_owners[slot].fill(0);
            stripe.group_live_mask[slot] = 0;
            stripe.group_live_objects[slot] = 0;
            stripe.group_is_split[slot] = 0;
            if (::FarLib::get_config().ft_background_rebuild) {
                stripe.group_pending_write[slot] = 1;
            }
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
        if (!group_has_space_locked(stripe) || stripe.in_group_pool ||
            stripe.shadow_quarantined || stripe.shadow_regions_released) {
            return false;
        }
        stripe.in_group_pool = true;
        return true;
    }

    std::array<uint32_t, kShardCount> effective_shadow_layout(
        const Stripe &stripe) const {
        auto layout = stripe.shadow_layout;
        const auto &config = ::FarLib::get_config();
        for (uint8_t shard = 0; shard < kShardCount; ++shard)
            layout[shard] = static_cast<uint32_t>(config.map_remote_addr(
                resolve_rebuilt_addr(stripe.stripe_id, shard,
                                     stripe.shard_base[shard])).first);
        return layout;
    }

    void push_group_candidate(uint16_t bin, uint64_t stripe_id) {
        Stripe *stripe = stripe_by_id(stripe_id);
        if (stripe != nullptr && stripe->shadow_only) {
            ShadowLayoutPool *pool =
                shadow_layout_pool_for(effective_shadow_layout(*stripe), true);
            if (pool == nullptr) return;
            std::lock_guard<std::mutex> lock(pool->groups.mutex);
            pool->groups.stripes_with_space.push_back(stripe_id);
            return;
        }
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
        handle.generation = slot_id < stripe.group_generation.size()
                                ? stripe.group_generation[slot_id]
                                : 0;
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
        const auto &config = ::FarLib::get_config();
        const bool reuse_rebuilt =
            config.ft_background_rebuild && config.is_carbink_mode();
        const bool standby_active = standby_active_for_policy();
        const int standby_endpoint =
            ::FarLib::allocator::remote::remote_global_heap.standby_endpoint();
        bool has_standby = false;
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            // Keep logical addresses stable for owners and reverse lookup.
            // Only Carbink routes new WRs through the published repair map.
            const uint32_t endpoint = reuse_rebuilt
                ? static_cast<uint32_t>(config.map_remote_addr(
                      resolve_rebuilt_addr(stripe.stripe_id, shard,
                                           stripe.shard_base[shard])).first)
                : stripe.shard_endpoint[shard];
            if (!endpoint_alive_for_policy(endpoint)) return false;
            if (standby_active &&
                endpoint == static_cast<uint32_t>(standby_endpoint)) {
                has_standby = true;
            }
        }
        // Unrepaired stripes remain frozen. A fully published Carbink repair
        // may reuse dead/free groups, with the missing role on the spare.
        return !standby_active || has_standby;
    }

    bool stripe_effectively_healthy_locked(const Stripe &stripe) const {
        const auto &config = ::FarLib::get_config();
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            const uint64_t logical = stripe.shard_base[shard];
            const uint64_t base =
                resolve_rebuilt_addr(stripe.stripe_id, shard, logical);
            if (base == kInvalidRemoteAddr ||
                !config.validate_mapping(base, shard_table_.shard_size())) {
                return false;
            }
            const size_t endpoint = config.map_remote_addr(base).first;
            if (!endpoint_is_alive(endpoint)) return false;
        }
        return true;
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
                          bool require_distinct_endpoints = false,
                          uint32_t *first_group_slot = nullptr) {
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
        // Revalidate the selected physical layout at the publication
        // barrier.  A creator racing an endpoint failure must be discarded,
        // not published outside the background scan limit.
        if (endpoint_liveness_enabled) {
            bool has_standby = false;
            const bool standby_active = standby_active_for_policy();
            const int standby_endpoint =
                ::FarLib::allocator::remote::remote_global_heap
                    .standby_endpoint();
            for (uint8_t a = 0; a < kShardCount; a++) {
                if (!endpoint_alive_for_policy(endpoints[a])) {
                    release_allocated_shards(shard_base);
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
                            return nullptr;
                        }
                    }
                }
            }
            if (standby_active && !has_standby) {
                release_allocated_shards(shard_base);
                return nullptr;
            }
        }
        stripe_id = stripe_count_.load(std::memory_order_relaxed);
        if (stripe_id >= stripe_capacity_) {
            release_allocated_shards(shard_base);
            return nullptr;
        }
        stripe->stripe_id = stripe_id;
        if (first_group_slot != nullptr &&
            !allocate_group_from_stripe_locked(*stripe, first_group_slot)) {
            release_allocated_shards(shard_base);
            return nullptr;
        }
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

    Stripe *create_shadow_stripe(
        uint16_t bin, const std::array<uint32_t, kShardCount> &layout,
        uint32_t *first_group_slot) {
        if (first_group_slot == nullptr ||
            !::FarLib::get_config().ft_background_rebuild) {
            return nullptr;
        }
        const auto &config = ::FarLib::get_config();
        for (uint8_t a = 0; a < kShardCount; a++) {
            if (layout[a] >= static_cast<uint32_t>(config.server_count) ||
                !endpoint_alive_for_policy(layout[a])) {
                return nullptr;
            }
            for (uint8_t b = 0; b < a; b++) {
                if (layout[a] == layout[b]) return nullptr;
            }
        }
        std::array<uint64_t, kShardCount> shard_base;
        shard_base.fill(kInvalidRemoteAddr);
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            shard_base[shard] = allocate_region_for_endpoint(layout[shard]);
            if (shard_base[shard] == kInvalidRemoteAddr) {
                release_allocated_shards(shard_base);
                return nullptr;
            }
        }
        auto stripe = std::make_unique<Stripe>();
        stripe->bin = bin;
        stripe->shadow_only = true;
        stripe->shadow_layout = layout;
        stripe->slot_size =
            static_cast<uint32_t>(::FarLib::allocator::get_bin_size(bin));
        stripe->slots_per_shard =
            static_cast<uint32_t>(shard_table_.shard_size() / stripe->slot_size);
        stripe->free_count = stripe->slots_per_shard * kDataShards;
        stripe->shard_base = shard_base;
        stripe->shard_endpoint = layout;
        for (uint8_t shard = 0; shard < kDataShards; shard++) {
            stripe->free_bits[shard].init_full(stripe->slots_per_shard);
            stripe->dead_bits[shard].init_empty(stripe->slots_per_shard);
        }

        std::lock_guard<std::mutex> lock(stripes_mutex_);
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            if (!endpoint_alive_for_policy(layout[shard])) {
                release_allocated_shards(shard_base);
                return nullptr;
            }
        }
        const uint64_t stripe_id =
            stripe_count_.load(std::memory_order_relaxed);
        if (stripe_id >= stripe_capacity_) {
            release_allocated_shards(shard_base);
            return nullptr;
        }
        stripe->stripe_id = stripe_id;
        if (!allocate_group_from_stripe_locked(*stripe, first_group_slot)) {
            release_allocated_shards(shard_base);
            return nullptr;
        }
        std::array<size_t, kDataShards> data_region_ids{};
        for (uint8_t shard = 0; shard < kDataShards; shard++) {
            data_region_ids[shard] =
                shard_base[shard] / shard_table_.shard_size();
            if (data_region_ids[shard] >= shard_binding_count_) {
                release_allocated_shards(shard_base);
                return nullptr;
            }
        }
        Stripe *stripe_ptr = stripe.get();
        stripe_storage_.push_back(std::move(stripe));
        stripe_index_[static_cast<size_t>(stripe_id)].store(
            stripe_ptr, std::memory_order_release);
        for (uint8_t shard = 0; shard < kDataShards; shard++) {
            shard_bindings_[data_region_ids[shard]].store(
                encode_binding(stripe_id, shard, bin),
                std::memory_order_release);
        }
        stripe_count_.store(stripe_id + 1, std::memory_order_release);
        return stripe_ptr;
    }

public:
    SmallObjectStripeManager() = default;
    using BackgroundRebuildStripe = ::FarLib::cache::BackgroundRebuildStripe;
    using BackgroundRebuildSet = ::FarLib::cache::BackgroundRebuildSet;
    using BackgroundRebuildStatus = ::FarLib::cache::BackgroundRebuildStatus;
    using View = BackgroundRebuildStripe;

    // remote_capacity_bytes: total remote capacity (server_count *
    // server_buffer_size); shard_size: one region, i.e. SmallObjectStripeShardSize.
    void init(size_t remote_capacity_bytes, size_t shard_size) {
        release_background_targets_for_shutdown();
        release_quarantined_shadows_for_shutdown();
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
        {
            std::lock_guard<std::mutex> lock(shadow_layout_pools_mutex_);
            shadow_layout_pools_.clear();
        }
        stripe_count_.store(0, std::memory_order_relaxed);
        next_endpoint_.store(0, std::memory_order_relaxed);
    }

    bool enabled() const { return shard_table_.enabled(); }

    ::FarLib::runtime_metadata::Snapshot metadata_usage() {
        using Snapshot = ::FarLib::runtime_metadata::Snapshot;
        Snapshot snapshot{};

        snapshot.region_bytes += sizeof(shard_table_);
        std::lock_guard<std::mutex> stripes_lock(stripes_mutex_);
        // Includes inline pool/mapping headers and compiler padding once.
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

        const auto add_pool_backing = [&](auto &pool_array) {
            for (auto &pool : pool_array) {
                std::lock_guard<std::mutex> lock(pool.mutex);
                snapshot.mapping_bytes +=
                    static_cast<uint64_t>(pool.stripes_with_space.capacity()) *
                    sizeof(uint64_t);
            }
        };
        add_pool_backing(pools_);
        add_pool_backing(group_pools_);

        const auto add_bitmap_backing = [](const Bitmap &bitmap) {
            return static_cast<uint64_t>(bitmap.words.capacity()) *
                   sizeof(uint64_t);
        };
        for (const auto &holder : stripe_storage_) {
            if (holder == nullptr) continue;
            const Stripe &stripe = *holder;
            std::lock_guard<std::mutex> stripe_lock(stripe.mutex);
            ++snapshot.stripes;
            snapshot.stripe_bytes += sizeof(Stripe);

            for (const auto &bitmap : stripe.free_bits)
                snapshot.stripe_bytes += add_bitmap_backing(bitmap);
            for (const auto &bitmap : stripe.dead_bits)
                snapshot.stripe_bytes += add_bitmap_backing(bitmap);

            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_state_words.capacity()) *
                sizeof(uint64_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_live_mask.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_live_objects.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_ready_after_write.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_generation.capacity()) *
                sizeof(uint64_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_is_split.capacity()) *
                sizeof(uint8_t);
            snapshot.group_bytes +=
                static_cast<uint64_t>(stripe.group_compaction_claimed.capacity()) *
                sizeof(uint8_t);

            snapshot.mapping_bytes +=
                static_cast<uint64_t>(stripe.group_owners.capacity()) *
                sizeof(std::array<uintptr_t, kDataShards>);
            snapshot.group_slots +=
                static_cast<uint64_t>(stripe.group_live_mask.capacity());
        }
        return snapshot;
    }

    // Cold-path snapshot of the physical EC group ownership.  A group keeps
    // all six slot segments reserved from allocation while it is in progress
    // or sealed.  A dead group remains reserved only while a compaction claim
    // protects it; an unclaimed dead group is reusable and therefore omitted.
    // The caller samples this at phase boundaries, so taking the stripe and
    // stripe-list mutexes here does not add a hot-path counter or lock.
    std::vector<uint64_t> live_group_bytes_by_endpoint() const {
        const int configured_endpoint_count =
            ::FarLib::get_config().server_count;
        const size_t endpoint_count =
            configured_endpoint_count > 0
                ? static_cast<size_t>(configured_endpoint_count)
                : 0;
        std::vector<uint64_t> bytes(endpoint_count, 0);
        if (!enabled() || endpoint_count == 0) return bytes;

        std::lock_guard<std::mutex> stripes_lock(stripes_mutex_);
        for (const auto &stripe_holder : stripe_storage_) {
            if (stripe_holder == nullptr) continue;
            const Stripe &stripe = *stripe_holder;
            std::lock_guard<std::mutex> stripe_lock(stripe.mutex);
            if (stripe.group_state_words.empty()) continue;
            for (uint32_t slot = 0; slot < stripe.slots_per_shard; ++slot) {
                const uint8_t state = group_state_locked(stripe, slot);
                const bool reserved =
                    state == kSlotGroupInProgress ||
                    state == kSlotGroupSealed ||
                    (state == kSlotGroupDead &&
                     group_compaction_claimed_locked(stripe, slot));
                if (!reserved) continue;
                for (uint8_t shard = 0; shard < kShardCount; ++shard) {
                    const uint32_t endpoint = stripe.shard_endpoint[shard];
                    if (endpoint >= bytes.size()) continue;
                    bytes[endpoint] += stripe.slot_size;
                }
            }
        }
        return bytes;
    }

    // Nonblocking observer path for the same reservation metric as
    // live_group_bytes_by_endpoint(). stripe_index_ publication is
    // release/acquire and lifecycle reset/destruction is quiescent after the
    // sampler is shut down; no registry/stripe mutex or slot scan is needed.
    std::vector<uint64_t> observer_group_endpoint_bytes() const {
        const int configured_endpoint_count =
            ::FarLib::get_config().server_count;
        const size_t endpoint_count =
            configured_endpoint_count > 0
                ? static_cast<size_t>(configured_endpoint_count)
                : 0;
        std::vector<uint64_t> bytes(endpoint_count, 0);
        if (!enabled() || endpoint_count == 0) return bytes;
        const size_t count = static_cast<size_t>(std::min<uint64_t>(
            stripe_count_.load(std::memory_order_acquire),
            static_cast<uint64_t>(stripe_capacity_)));
        for (size_t stripe_id = 0; stripe_id < count; ++stripe_id) {
            const Stripe *stripe = stripe_by_id(stripe_id);
            if (stripe == nullptr) continue;
            const uint64_t groups = stripe->observer_reserved_group_count.load(
                std::memory_order_relaxed);
            if (groups == 0) continue;
            for (uint8_t shard = 0; shard < kShardCount; ++shard) {
                const uint32_t endpoint = stripe->shard_endpoint[shard];
                if (endpoint >= bytes.size()) continue;
                bytes[endpoint] += groups * stripe->slot_size;
            }
        }
        return bytes;
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

    uint64_t background_rebuild_scan_limit() const {
        std::lock_guard<std::mutex> lock(stripes_mutex_);
        return stripe_count_.load(std::memory_order_acquire);
    }

    uint64_t background_rebuild_scan_limit(uint32_t) const {
        return background_rebuild_scan_limit();
    }

    uint64_t count_rebuild_remaining(uint32_t failed_endpoint) const {
        const uint64_t limit = background_rebuild_scan_limit();
        uint64_t remaining = 0;
        for (uint64_t id = 0; id < limit; id++) {
            BackgroundRebuildStripe view;
            const auto status =
                background_rebuild_view(id, failed_endpoint, &view);
            if (status != BackgroundRebuildStatus::Skip) ++remaining;
        }
        return remaining;
    }

    BackgroundRebuildStatus background_rebuild_view(
        uint64_t stripe_id, uint32_t failed_endpoint,
        BackgroundRebuildStripe *view_out) const {
        if (view_out == nullptr) return BackgroundRebuildStatus::Unsupported;
        *view_out = BackgroundRebuildStripe{};
        if (!::FarLib::get_config().ft_background_rebuild) {
            return BackgroundRebuildStatus::Skip;
        }
        const Stripe *stripe = stripe_by_id(stripe_id);
        if (stripe == nullptr) return BackgroundRebuildStatus::Skip;
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (stripe->stripe_id != stripe_id) return BackgroundRebuildStatus::Skip;
        if (failed_endpoint >=
            ::FarLib::allocator::remote::remote_global_heap.endpoint_count()) {
            return BackgroundRebuildStatus::Unsupported;
        }
        uint8_t failed_shard = 0xff;
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            if (stripe->shard_endpoint[shard] != failed_endpoint) continue;
            if (failed_shard != 0xff) {
                return BackgroundRebuildStatus::Unsupported;
            }
            failed_shard = shard;
        }
        if (failed_shard == 0xff) return BackgroundRebuildStatus::Skip;
        if (endpoint_is_alive(failed_endpoint)) {
            return BackgroundRebuildStatus::Unsupported;
        }
        if (!stripe->has_groups.load(std::memory_order_acquire)) {
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
            return BackgroundRebuildStatus::Unsupported;
        }
        const auto &config = ::FarLib::get_config();
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            if (shard == failed_shard) continue;
            const uint64_t base = stripe->shard_base[shard];
            if (base == kInvalidRemoteAddr ||
                !config.validate_mapping(base, shard_table_.shard_size()) ||
                !endpoint_is_alive(config.map_remote_addr(base).first)) {
                return BackgroundRebuildStatus::Unsupported;
            }
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
        if (live_groups == 0 && config.is_carbink_mode()) {
            if (stripe->shadow_quarantined || stripe->shadow_regions_released)
                return BackgroundRebuildStatus::Skip;
            if (stripe->live_count != 0)
                return BackgroundRebuildStatus::Unsupported;
            for (uint32_t slot = 0; slot < stripe->slots_per_shard; ++slot) {
                if (group_compaction_claimed_locked(*stripe, slot)) busy = true;
                if (stripe->group_live_mask[slot] != 0 ||
                    stripe->group_live_objects[slot] != 0)
                    return BackgroundRebuildStatus::Unsupported;
            }
        }
        if (busy) return BackgroundRebuildStatus::Busy;
        if (live_groups == 0 &&
            (!config.is_carbink_mode() || stripe->shadow_quarantined))
            return BackgroundRebuildStatus::Skip;
        if (stripe->shard_base[failed_shard] == kInvalidRemoteAddr ||
            stripe->slot_size == 0) {
            return BackgroundRebuildStatus::Unsupported;
        }
        return BackgroundRebuildStatus::Ready;
    }

    bool publish_background_rebuild(const BackgroundRebuildStripe &view,
                                    uint64_t target_base) {
        if (!::FarLib::get_config().ft_background_rebuild ||
            target_base == kInvalidRemoteAddr ||
            view.stripe_id == kInvalidStripeId ||
            view.failed_shard >= kShardCount || view.slot_size == 0 ||
            view.shard_bytes != shard_table_.shard_size() ||
            (view.live_groups == 0 &&
             !::FarLib::get_config().is_carbink_mode()) ||
            view.live_groups != view.live_slot_ids.size() ||
            view.live_objects < view.live_groups) {
            return false;
        }
        const auto &config = ::FarLib::get_config();
        const auto mapped = config.map_remote_addr(target_base);
        if (config.ft_standby_endpoint < 0 ||
            mapped.first != static_cast<size_t>(config.ft_standby_endpoint) ||
            !endpoint_is_alive(mapped.first) ||
            view.failed_endpoint >=
                ::FarLib::allocator::remote::remote_global_heap.endpoint_count() ||
            endpoint_is_alive(view.failed_endpoint) ||
            !config.validate_mapping(target_base, view.shard_bytes)) {
            return false;
        }
        Stripe *stripe = stripe_by_id(view.stripe_id);
        if (stripe == nullptr) return false;
        std::unique_lock<std::mutex> lock(stripe->mutex);
        if (stripe->group_state_words.empty() ||
            stripe->stripe_id != view.stripe_id ||
            stripe->slot_size != view.slot_size ||
            stripe->shard_endpoint[view.failed_shard] !=
                view.failed_endpoint ||
            endpoint_is_alive(view.failed_endpoint) ||
            !endpoints_distinct(*stripe) ||
            stripe->shard_base[view.failed_shard] == kInvalidRemoteAddr) {
            return false;
        }
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            if (stripe->shard_base[shard] != view.original_base[shard] ||
                stripe->shard_endpoint[shard] != view.endpoint[shard]) {
                return false;
            }
            if (stripe->rebuilt_base[shard].load(
                    std::memory_order_acquire) != kInvalidRemoteAddr) {
                return false;
            }
            if (target_base == stripe->shard_base[shard]) return false;
        }
        uint64_t live_groups = 0;
        uint64_t live_objects = 0;
        std::vector<uint32_t> live_slots;
        bool busy = false;
        if (!background_live_groups_locked(*stripe, &live_groups,
                                           &live_objects, &live_slots,
                                           &busy) ||
            busy || live_groups > view.live_groups ||
            live_objects > view.live_objects ||
            !std::is_sorted(view.live_slot_ids.begin(),
                            view.live_slot_ids.end())) {
            return false;
        }
        for (uint32_t current_slot : live_slots) {
            if (!std::binary_search(view.live_slot_ids.begin(),
                                    view.live_slot_ids.end(), current_slot)) {
                return false;
            }
        }
        if (view.live_groups == 0) {
            if (stripe->live_count != 0 || stripe->shadow_quarantined ||
                stripe->shadow_regions_released) return false;
            for (uint32_t slot = 0; slot < stripe->slots_per_shard; ++slot)
                if (group_compaction_claimed_locked(*stripe, slot) ||
                    stripe->group_live_mask[slot] != 0 ||
                    stripe->group_live_objects[slot] != 0)
                    return false;
        }
        stripe->rebuilt_base[view.failed_shard].store(
            target_base, std::memory_order_release);
        // Publication follows the repair WRITE completion. New writes can
        // now use this codeword without racing its background snapshot.
        // The old shadow-layout pool may still contain a stale candidate.
        // Its effective-layout check will reject it without clearing the
        // new pool membership. Keep original addresses/layout immutable.
        if (stripe->shadow_only) stripe->in_group_pool = false;
        const bool requeue = config.is_carbink_mode() &&
            mark_group_queued_if_has_space_locked(*stripe);
        lock.unlock();
        if (requeue) push_group_candidate(stripe->bin, stripe->stripe_id);
        return true;
    }

    uint64_t count_rebuild_set_remaining(uint64_t mask) const {
        const uint64_t limit = background_rebuild_scan_limit();
        uint64_t remaining = 0;
        for (uint64_t id = 0; id < limit; ++id) {
            BackgroundRebuildSet view;
            if (background_rebuild_set_view(id, mask, &view) !=
                BackgroundRebuildStatus::Skip) {
                ++remaining;
            }
        }
        return remaining;
    }

    BackgroundRebuildStatus background_rebuild_set_view(
        uint64_t stripe_id, uint64_t mask,
        BackgroundRebuildSet *out) const {
        if (out == nullptr) return BackgroundRebuildStatus::Unsupported;
        *out = BackgroundRebuildSet{};
        const auto &config = ::FarLib::get_config();
        if (!config.ft_background_rebuild) return BackgroundRebuildStatus::Skip;
        const size_t endpoint_count =
            ::FarLib::allocator::remote::remote_global_heap.endpoint_count();
        if (endpoint_count == 0 || endpoint_count > 64 || mask == 0) {
            return BackgroundRebuildStatus::Unsupported;
        }
        const uint64_t valid_mask =
            endpoint_count == 64
                ? std::numeric_limits<uint64_t>::max()
                : ((uint64_t{1} << endpoint_count) - 1);
        if ((mask & ~valid_mask) != 0) return BackgroundRebuildStatus::Unsupported;
        const Stripe *stripe = stripe_by_id(stripe_id);
        if (stripe == nullptr) return BackgroundRebuildStatus::Skip;
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (stripe->stripe_id != stripe_id) return BackgroundRebuildStatus::Skip;
        if (!stripe->has_groups.load(std::memory_order_acquire) ||
            !endpoints_distinct(*stripe)) {
            return stripe->live_count == 0
                       ? BackgroundRebuildStatus::Skip
                       : BackgroundRebuildStatus::Unsupported;
        }
        uint8_t missing = 0, missing_count = 0;
        for (uint8_t shard = 0; shard < kShardCount; ++shard) {
            const uint32_t endpoint = stripe->shard_endpoint[shard];
            if (endpoint >= endpoint_count || endpoint >= 64 ||
                stripe->shard_base[shard] == kInvalidRemoteAddr) {
                return BackgroundRebuildStatus::Unsupported;
            }
            if ((mask & (uint64_t{1} << endpoint)) == 0) continue;
            if (endpoint_is_alive(endpoint)) return BackgroundRebuildStatus::Unsupported;
            missing = static_cast<uint8_t>(missing | (1u << shard));
            ++missing_count;
        }
        if (missing_count == 0) return BackgroundRebuildStatus::Skip;
        if (missing_count > 2 ||
            static_cast<uint8_t>(kShardCount - missing_count) < 4 ||
            (missing_count == 2 &&
             config.ft_background_rebuild_failures != 2)) {
            return BackgroundRebuildStatus::Unsupported;
        }
        for (uint8_t shard = 0; shard < kShardCount; ++shard) {
            if ((missing & static_cast<uint8_t>(1u << shard)) != 0) continue;
            const uint64_t base = stripe->shard_base[shard];
            if (!endpoint_is_alive(stripe->shard_endpoint[shard]) ||
                !config.validate_mapping(base, shard_table_.shard_size()) ||
                !endpoint_is_alive(config.map_remote_addr(base).first)) {
                return BackgroundRebuildStatus::Unsupported;
            }
        }
        uint8_t published = 0;
        for (uint8_t shard = 0; shard < kShardCount; ++shard) {
            const uint64_t redirected =
                stripe->rebuilt_base[shard].load(std::memory_order_acquire);
            if (redirected == kInvalidRemoteAddr) continue;
            if ((missing & static_cast<uint8_t>(1u << shard)) == 0) {
                return BackgroundRebuildStatus::Unsupported;
            }
            published = static_cast<uint8_t>(published | (1u << shard));
        }
        if (published == missing) return BackgroundRebuildStatus::Skip;
        if (published != 0) return BackgroundRebuildStatus::Unsupported;

        uint64_t live_groups = 0, live_objects = 0;
        std::vector<uint32_t> slots;
        bool busy = false;
        if (!background_live_groups_locked(*stripe, &live_groups, &live_objects,
                                            &slots, &busy)) {
            return BackgroundRebuildStatus::Unsupported;
        }
        if (live_groups == 0 && config.is_carbink_mode()) {
            if (stripe->shadow_quarantined || stripe->shadow_regions_released) {
                return BackgroundRebuildStatus::Skip;
            }
            if (stripe->live_count != 0) return BackgroundRebuildStatus::Unsupported;
            for (uint32_t slot = 0; slot < stripe->slots_per_shard; ++slot) {
                if (group_compaction_claimed_locked(*stripe, slot)) busy = true;
                if (stripe->group_live_mask[slot] != 0 ||
                    stripe->group_live_objects[slot] != 0) {
                    return BackgroundRebuildStatus::Unsupported;
                }
            }
        }
        if (busy) return BackgroundRebuildStatus::Busy;
        if (live_groups == 0 &&
            (!config.is_carbink_mode() || stripe->shadow_quarantined)) {
            return BackgroundRebuildStatus::Skip;
        }
        if (stripe->slot_size == 0) return BackgroundRebuildStatus::Unsupported;

        BackgroundRebuildSet snapshot;
        snapshot.stripe_id = stripe_id;
        snapshot.slot_size = stripe->slot_size;
        snapshot.shard_bytes = shard_table_.shard_size();
        snapshot.original_base = stripe->shard_base;
        snapshot.endpoint = stripe->shard_endpoint;
        snapshot.missing_mask = missing;
        snapshot.live_groups = live_groups;
        snapshot.live_objects = live_objects;
        snapshot.live_slot_ids = std::move(slots);
        *out = std::move(snapshot);
        return BackgroundRebuildStatus::Ready;
    }

    bool publish_background_rebuild_set(
        const BackgroundRebuildSet &view,
        const std::array<uint64_t, kSmallObjectStripeShardCount> &targets) {
        constexpr uint8_t all_mask =
            static_cast<uint8_t>((1u << kShardCount) - 1u);
        const auto &config = ::FarLib::get_config();
        if (!config.ft_background_rebuild ||
            view.stripe_id == kInvalidStripeId || view.missing_mask == 0 ||
            (view.missing_mask & static_cast<uint8_t>(~all_mask)) != 0 ||
            view.slot_size == 0 ||
            view.shard_bytes != shard_table_.shard_size() ||
            (view.live_groups == 0 && !config.is_carbink_mode()) ||
            view.live_groups != view.live_slot_ids.size() ||
            view.live_objects < view.live_groups) {
            return false;
        }
        uint8_t missing_count = 0;
        for (uint8_t shard = 0; shard < kShardCount; ++shard)
            if ((view.missing_mask & static_cast<uint8_t>(1u << shard)) != 0)
                ++missing_count;
        if (missing_count == 0 || missing_count > 2 ||
            static_cast<uint8_t>(kShardCount - missing_count) < 4 ||
            (missing_count == 2 &&
             config.ft_background_rebuild_failures != 2)) {
            return false;
        }
        const size_t endpoint_count =
            ::FarLib::allocator::remote::remote_global_heap.endpoint_count();
        if (endpoint_count == 0 || endpoint_count > 64) return false;
        Stripe *stripe = stripe_by_id(view.stripe_id);
        if (stripe == nullptr) return false;
        std::unique_lock<std::mutex> lock(stripe->mutex);
        if (stripe->group_state_words.empty() ||
            stripe->stripe_id != view.stripe_id ||
            stripe->slot_size != view.slot_size || stripe->slot_size == 0 ||
            !endpoints_distinct(*stripe)) {
            return false;
        }
        uint8_t actual = 0;
        for (uint8_t shard = 0; shard < kShardCount; ++shard) {
            if (stripe->shard_base[shard] != view.original_base[shard] ||
                stripe->shard_endpoint[shard] != view.endpoint[shard]) {
                return false;
            }
            const uint32_t endpoint = stripe->shard_endpoint[shard];
            if (endpoint >= endpoint_count || endpoint >= 64) return false;
            const uint8_t bit = static_cast<uint8_t>(1u << shard);
            if ((view.missing_mask & bit) != 0) {
                if (endpoint_is_alive(endpoint)) return false;
                actual = static_cast<uint8_t>(actual | bit);
            } else if (!endpoint_is_alive(endpoint) ||
                       !config.validate_mapping(stripe->shard_base[shard],
                                                 view.shard_bytes)) {
                return false;
            }
            if (stripe->rebuilt_base[shard].load(
                    std::memory_order_acquire) != kInvalidRemoteAddr) {
                return false;
            }
        }
        if (actual != view.missing_mask) return false;

        std::array<uint32_t, kSmallObjectStripeShardCount> target_endpoint{};
        target_endpoint.fill(std::numeric_limits<uint32_t>::max());
        for (uint8_t shard = 0; shard < kShardCount; ++shard) {
            const uint8_t bit = static_cast<uint8_t>(1u << shard);
            if ((view.missing_mask & bit) == 0) continue;
            const uint64_t target = targets[shard];
            if (target == kInvalidRemoteAddr ||
                target % view.shard_bytes != 0 ||
                !config.validate_mapping(target, view.shard_bytes)) {
                return false;
            }
            const auto mapped = config.map_remote_addr(target);
            if (mapped.first >= endpoint_count ||
                !endpoint_is_alive(mapped.first)) {
                return false;
            }
            if (missing_count == 1 &&
                config.ft_background_rebuild_failures != 2 &&
                (config.ft_standby_endpoint < 0 ||
                 mapped.first != static_cast<size_t>(config.ft_standby_endpoint))) {
                return false;
            }
            for (uint8_t prior = 0; prior < kShardCount; ++prior) {
                if (stripe->shard_base[prior] == target) return false;
                if ((view.missing_mask &
                     static_cast<uint8_t>(1u << prior)) != 0) {
                    if (prior < shard &&
                        target_endpoint[prior] == mapped.first) {
                        return false;
                    }
                } else if (stripe->shard_endpoint[prior] == mapped.first) {
                    return false;
                }
            }
            target_endpoint[shard] = static_cast<uint32_t>(mapped.first);
        }
        uint64_t live_groups = 0, live_objects = 0;
        std::vector<uint32_t> slots;
        bool busy = false;
        if (!background_live_groups_locked(*stripe, &live_groups, &live_objects,
                                            &slots, &busy) ||
            busy || live_groups > view.live_groups ||
            live_objects > view.live_objects ||
            !std::is_sorted(view.live_slot_ids.begin(), view.live_slot_ids.end())) {
            return false;
        }
        for (size_t i = 0; i < view.live_slot_ids.size(); ++i) {
            if (view.live_slot_ids[i] >= stripe->slots_per_shard ||
                (i != 0 && view.live_slot_ids[i] == view.live_slot_ids[i - 1])) {
                return false;
            }
        }
        for (uint32_t slot : slots)
            if (!std::binary_search(view.live_slot_ids.begin(),
                                    view.live_slot_ids.end(), slot)) {
                return false;
            }
        if (view.live_groups == 0) {
            if (stripe->live_count != 0 || stripe->shadow_quarantined ||
                stripe->shadow_regions_released) {
                return false;
            }
            for (uint32_t slot = 0; slot < stripe->slots_per_shard; ++slot) {
                if (group_compaction_claimed_locked(*stripe, slot) ||
                    stripe->group_live_mask[slot] != 0 ||
                    stripe->group_live_objects[slot] != 0) {
                    return false;
                }
            }
        }
        for (uint8_t shard = 0; shard < kShardCount; ++shard)
            if ((view.missing_mask & static_cast<uint8_t>(1u << shard)) != 0)
                stripe->rebuilt_base[shard].store(
                    targets[shard], std::memory_order_release);
        if (stripe->shadow_only) stripe->in_group_pool = false;
        const bool requeue =
            config.is_carbink_mode() &&
            mark_group_queued_if_has_space_locked(*stripe);
        lock.unlock();
        if (requeue) push_group_candidate(stripe->bin, stripe->stripe_id);
        return true;
    }
    uint64_t resolve_rebuilt_addr(uint64_t stripe_id, uint8_t shard_idx,
                                  uint64_t logical_addr) const {
        if (shard_idx >= kShardCount) return logical_addr;
        const Stripe *stripe = stripe_by_id(stripe_id);
        if (stripe == nullptr) return logical_addr;
        const uint64_t base = stripe->shard_base[shard_idx];
        if (base == kInvalidRemoteAddr || logical_addr < base ||
            logical_addr - base >= shard_table_.shard_size()) {
            return logical_addr;
        }
        const uint64_t target =
            stripe->rebuilt_base[shard_idx].load(std::memory_order_acquire);
        return target == kInvalidRemoteAddr
                   ? logical_addr
                   : target + (logical_addr - base);
    }

    uint64_t resolve_rebuilt_addr(uint64_t logical_addr) const {
        DecodedBinding binding;
        if (binding_for_addr(logical_addr, &binding) &&
            binding.shard_idx < kDataShards) {
            return resolve_rebuilt_addr(binding.stripe_id, binding.shard_idx,
                                        logical_addr);
        }
        const uint64_t count = stripe_count_.load(std::memory_order_acquire);
        for (uint64_t id = 0; id < count; id++) {
            const Stripe *stripe = stripe_by_id(id);
            if (stripe == nullptr) continue;
            for (uint8_t shard = kDataShards; shard < kShardCount; shard++) {
                const uint64_t base = stripe->shard_base[shard];
                if (base != kInvalidRemoteAddr && logical_addr >= base &&
                    logical_addr - base < shard_table_.shard_size()) {
                    return resolve_rebuilt_addr(id, shard, logical_addr);
                }
            }
        }
        return logical_addr;
    }

    bool physical_compaction_layout(
        const CompactionGroupView &view,
        std::array<uint32_t, kShardCount> *layout_out) const {
        if (layout_out == nullptr) return false;
        const auto &config = ::FarLib::get_config();
        std::array<uint32_t, kShardCount> layout{};
        for (uint8_t shard = 0; shard < kShardCount; shard++) {
            const uint64_t physical = resolve_rebuilt_addr(
                view.id.stripe_id, shard, view.group.segments[shard].addr);
            if (!config.validate_mapping(physical, shard_table_.shard_size())) {
                return false;
            }
            layout[shard] = static_cast<uint32_t>(
                config.map_remote_addr(physical).first);
            if (!endpoint_is_alive(layout[shard])) return false;
            for (uint8_t prior = 0; prior < shard; prior++) {
                if (layout[prior] == layout[shard]) return false;
            }
        }
        *layout_out = layout;
        return true;
    }

    // Called only after the owning background worker and quiesced evacuation
    // have joined.  Individual object release never frees a repaired region.
    void release_background_targets_for_shutdown() {
        std::lock_guard<std::mutex> lock(stripes_mutex_);
        release_published_background_targets_locked();
    }

    // Quarantined shadow groups retain their claim until this quiescent
    // shutdown point, so a late RPC cannot race a fresh shadow allocation.
    void release_quarantined_shadows_for_shutdown() {
        std::lock_guard<std::mutex> lock(stripes_mutex_);
        for (auto &owned : stripe_storage_) {
            Stripe *stripe = owned.get();
            if (stripe == nullptr || !stripe->shadow_only ||
                !stripe->shadow_quarantined ||
                stripe->shadow_regions_released) {
                continue;
            }
            const auto bases = stripe->shard_base;
            release_allocated_shards(bases);
            for (uint8_t shard = 0; shard < kShardCount; shard++) {
                if (shard < kDataShards &&
                    bases[shard] != kInvalidRemoteAddr) {
                    const uint64_t region_id =
                        bases[shard] / shard_table_.shard_size();
                    if (region_id < shard_binding_count_) {
                        shard_bindings_[region_id].store(
                            kInvalidBinding, std::memory_order_release);
                    }
                }
                stripe->shard_base[shard] = kInvalidRemoteAddr;
            }
            std::fill(stripe->group_compaction_claimed.begin(),
                      stripe->group_compaction_claimed.end(), 0);
            stripe->shadow_regions_released = true;
        }
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

        uint32_t slot_id = 0;
        const bool reserved_first =
            ::FarLib::get_config().ft_background_rebuild;
        Stripe *stripe = create_stripe(
            bin, true, reserved_first ? &slot_id : nullptr);
        if (stripe == nullptr) return false;
        bool should_enqueue = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (!reserved_first &&
                !allocate_group_from_stripe_locked(*stripe, &slot_id)) {
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
        stripe->group_ready_after_write[id.slot_id] = 0;
        stripe->group_owners[id.slot_id].fill(0);
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
        stripe->group_ready_after_write[id.slot_id] = 0;
        stripe->group_owners[id.slot_id].fill(0);
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
        bool should_requeue = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            released = mark_dead_group_locked(*stripe, id);
            should_requeue = released &&
                             !group_compaction_claimed_locked(*stripe,
                                                              id.slot_id);
        }
        if (should_requeue) requeue_group_stripe(*stripe);
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
        stripe.group_ready_after_write[id.slot_id] = 0;
        stripe.group_owners[id.slot_id].fill(0);
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

    bool finish_background_write(const SlotGroupId &id) {
        if (!::FarLib::get_config().ft_background_rebuild) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        bool should_enqueue = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (id.slot_id >= stripe->group_pending_write.size() ||
                stripe->group_pending_write[id.slot_id] == 0) {
                return false;
            }
            stripe->group_pending_write[id.slot_id] = 0;
            if (group_state_locked(*stripe, id.slot_id) == kSlotGroupDead &&
                !group_compaction_claimed_locked(*stripe, id.slot_id) &&
                stripe_eligible_for_new_group(*stripe)) {
                should_enqueue = mark_group_queued_if_has_space_locked(*stripe);
            }
        }
        if (should_enqueue) push_group_candidate(stripe->bin, stripe->stripe_id);
        return true;
    }

    bool cancel_background_write(const SlotGroupId &id) {
        if (!::FarLib::get_config().ft_background_rebuild) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        bool should_enqueue = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (id.slot_id >= stripe->group_pending_write.size() ||
                stripe->group_pending_write[id.slot_id] == 0) {
                return false;
            }
            stripe->group_pending_write[id.slot_id] = 0;
            if (group_state_locked(*stripe, id.slot_id) == kSlotGroupDead &&
                !group_compaction_claimed_locked(*stripe, id.slot_id) &&
                stripe_eligible_for_new_group(*stripe)) {
                should_enqueue = mark_group_queued_if_has_space_locked(*stripe);
            }
        }
        if (should_enqueue) push_group_candidate(stripe->bin, stripe->stripe_id);
        return true;
    }

    bool allocate_compaction_shadow(
        const std::array<uint32_t, kShardCount> &physical_layout,
        CompactionGroupView *view_out) {
        if (view_out == nullptr ||
            !::FarLib::get_config().ft_background_rebuild) {
            return false;
        }
        const size_t bin = bin_from_size(8192);
        if (::FarLib::allocator::get_bin_size(bin) != 8192) return false;
        ShadowLayoutPool *pool =
            shadow_layout_pool_for(physical_layout, true);
        if (pool == nullptr) return false;

        auto fill_shadow_view = [&](Stripe *stripe, uint32_t slot) {
            SlotGroupHandle handle;
            if (stripe == nullptr ||
                !fill_group_handle(*stripe, static_cast<uint16_t>(bin), slot,
                                   &handle)) {
                return false;
            }
            if (slot >= stripe->group_compaction_claimed.size()) return false;
            set_group_compaction_claimed_locked(*stripe, slot, true);
            view_out->id = handle.id;
            view_out->group = handle;
            view_out->handle = handle;
            view_out->live_mask = 0;
            view_out->live_count = 0;
            view_out->generation = handle.generation;
            view_out->owners.fill(0);
            view_out->shadow_only = true;
            return true;
        };

        while (true) {
            const uint64_t stripe_id = pop_shadow_candidate(physical_layout);
            if (stripe_id == kInvalidStripeId) break;
            Stripe *stripe = stripe_by_id(stripe_id);
            if (stripe == nullptr) continue;
            uint32_t slot_id = 0;
            bool should_enqueue = false;
            {
                std::lock_guard<std::mutex> lock(stripe->mutex);
                if (!stripe->shadow_only ||
                    stripe->shadow_regions_released ||
                    stripe->shadow_quarantined ||
                    effective_shadow_layout(*stripe) != physical_layout ||
                    !stripe->in_group_pool) {
                    continue;
                }
                stripe->in_group_pool = false;
                if (!stripe_eligible_for_new_group(*stripe)) continue;
                if (!allocate_group_from_stripe_locked(*stripe, &slot_id) ||
                    !fill_shadow_view(stripe, slot_id)) {
                    continue;
                }
                should_enqueue =
                    mark_group_queued_if_has_space_locked(*stripe);
            }
            if (should_enqueue) {
                push_group_candidate(static_cast<uint16_t>(bin), stripe_id);
            }
            return true;
        }

        uint32_t slot_id = 0;
        Stripe *stripe =
            create_shadow_stripe(static_cast<uint16_t>(bin), physical_layout,
                                 &slot_id);
        if (stripe == nullptr) return false;
        {
            std::lock_guard<std::mutex> lock(pool->groups.mutex);
            ++pool->created_stripes;
        }
        bool should_enqueue = false;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (!fill_shadow_view(stripe, slot_id)) return false;
            should_enqueue =
                mark_group_queued_if_has_space_locked(*stripe);
        }
        if (should_enqueue) {
            push_group_candidate(static_cast<uint16_t>(bin), stripe->stripe_id);
        }
        return true;
    }

    bool mark_compaction_shadow_durable(
        const CompactionGroupView &expected) {
        if (!::FarLib::get_config().ft_background_rebuild) return false;
        const SlotGroupId id =
            expected.id.valid() ? expected.id : expected.group.id;
        const uint64_t generation =
            expected.generation != 0 ? expected.generation
                                     : expected.group.generation;
        if (!id.valid() || generation == 0) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (!stripe->shadow_only ||
            !group_compaction_claimed_locked(*stripe, id.slot_id) ||
            group_state_locked(*stripe, id.slot_id) != kSlotGroupInProgress ||
            id.slot_id >= stripe->group_generation.size() ||
            stripe->group_generation[id.slot_id] != generation ||
            id.slot_id >= stripe->group_pending_write.size() ||
            stripe->group_pending_write[id.slot_id] == 0) {
            return false;
        }
        assert(stripe->live_count >= kDataShards);
        stripe->live_count -= kDataShards;
        stripe->group_live_mask[id.slot_id] = 0;
        stripe->group_live_objects[id.slot_id] = 0;
        stripe->group_ready_after_write[id.slot_id] = 1;
        stripe->group_owners[id.slot_id].fill(0);
        stripe->group_is_split[id.slot_id] = 0;
        set_group_state_locked(*stripe, id.slot_id, kSlotGroupSealed);
        return true;
    }

    bool quarantine_compaction_shadow(
        const CompactionGroupView &expected) {
        if (!::FarLib::get_config().ft_background_rebuild) return false;
        const SlotGroupId id =
            expected.id.valid() ? expected.id : expected.group.id;
        const uint64_t generation =
            expected.generation != 0 ? expected.generation
                                     : expected.group.generation;
        if (!id.valid() || generation == 0) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (!stripe->shadow_only ||
            !group_compaction_claimed_locked(*stripe, id.slot_id) ||
            id.slot_id >= stripe->group_generation.size() ||
            stripe->group_generation[id.slot_id] != generation) {
            return false;
        }
        const uint8_t state = group_state_locked(*stripe, id.slot_id);
        if (state != kSlotGroupInProgress &&
            state != kSlotGroupSealed) {
            return false;
        }
        if (id.slot_id >= stripe->group_live_mask.size() ||
            stripe->group_live_mask[id.slot_id] != 0) {
            // A shadow is quarantinable only before any owner has been
            // published into it; never erase a live destination's metadata.
            return false;
        }
        if (!mark_dead_group_locked(*stripe, id)) return false;
        stripe->group_pending_write[id.slot_id] = 0;
        stripe->group_ready_after_write[id.slot_id] = 0;
        stripe->group_owners[id.slot_id].fill(0);
        stripe->shadow_quarantined = true;
        stripe->group_shadow_quarantined[id.slot_id] = 1;
        stripe->in_group_pool = false;
        // The claim intentionally remains set until the quiescent shutdown
        // release; a late RPC can no longer make this slot reusable.
        return true;
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
        bool deferred_claim = false;
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
                stripe->group_owners[slot][binding.shard_idx] = 0;
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
                if (group_compaction_claimed_locked(*stripe, slot)) {
                    deferred_claim = true;
                }
                }
            }
        }

        if (deferred_claim) {
            return kGroupReleaseGroupReleased;
        }
        const bool group_dead = split_group_released || mark_dead_group(id);
        // False only if the same group was already released as a whole by a
        // concurrent path; the re-queue below is harmless then.
        if (!group_dead) return kGroupReleaseRejected;
        // A dead group means the stripe can serve a group again, so re-queue it
        // exactly like allocate_slot_group() does after an allocation: without
        // this the released slots would be unreachable and the group pool would
        // keep creating fresh stripes.
        if (!compaction_group_claimed(id)) requeue_group_stripe(*stripe);
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
    // Publication happens only after the caller has observed all six WRITE
    // completions.  The manager stores stable owner identities for the current
    // live mask; hole entries are deliberately ignored and cleared.
    bool publish_compaction_ready(
        const SlotGroupHandle &handle,
        const std::array<uintptr_t, kDataShards> &owners) {
        const SlotGroupId id = handle.id;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (!id.valid() || stripe == nullptr ||
            id.slot_id >= stripe->slots_per_shard ||
            handle.generation == 0 || handle.slot_size != 8192) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (group_state_locked(*stripe, id.slot_id) != kSlotGroupSealed ||
            stripe->slot_size != 8192 ||
            id.slot_id >= stripe->group_generation.size() ||
            stripe->group_generation[id.slot_id] != handle.generation ||
            id.slot_id >= stripe->group_is_split.size() ||
            stripe->group_is_split[id.slot_id] != 0 ||
            id.slot_id >= stripe->group_live_mask.size() ||
            id.slot_id >= stripe->group_ready_after_write.size()) {
            return false;
        }
        const uint8_t mask = stripe->group_live_mask[id.slot_id];
        if (mask == 0) return false;
        for (uint8_t slot = 0; slot < kDataShards; slot++) {
            if ((mask & static_cast<uint8_t>(1u << slot)) != 0 &&
                owners[slot] == 0) {
                return false;
            }
        }
        auto &stored = stripe->group_owners[id.slot_id];
        stored.fill(0);
        for (uint8_t slot = 0; slot < kDataShards; slot++) {
            if ((mask & static_cast<uint8_t>(1u << slot)) != 0) {
                stored[slot] = owners[slot];
            }
        }
        stripe->group_ready_after_write[id.slot_id] = 1;
        return true;
    }

    // Bounded round-robin scanner.  Cursors are caller-owned so separate
    // scanner fibres can keep progress without a global scan lock.
    bool collect_compaction_candidates(
        size_t &stripe_cursor, size_t &slot_cursor, size_t budget,
        std::vector<CompactionGroupView> &out) const {
        out.clear();
        const size_t total =
            static_cast<size_t>(stripe_count_.load(std::memory_order_acquire));
        if (budget == 0 || total == 0) return false;
        const size_t origin = stripe_cursor % total;
        size_t current = origin;
        size_t next_slot = slot_cursor;
        size_t visited = 0;
        while (visited < total && out.size() < budget) {
            const Stripe *stripe = stripe_by_id(current);
            bool stopped_inside = false;
            if (stripe != nullptr && stripe->slots_per_shard != 0) {
                std::lock_guard<std::mutex> lock(stripe->mutex);
                const bool physically_healthy =
                    stripe_effectively_healthy_locked(*stripe);
                const size_t slots = stripe->slots_per_shard;
                const size_t begin =
                    (current == origin ? next_slot : 0) % slots;
                for (size_t n = 0; n < slots && out.size() < budget; n++) {
                    const uint32_t slot =
                        static_cast<uint32_t>((begin + n) % slots);
                    if (physically_healthy &&
                        group_state_locked(*stripe, slot) ==
                            kSlotGroupSealed &&
                        stripe->slot_size == 8192 &&
                        slot < stripe->group_ready_after_write.size() &&
                        stripe->group_ready_after_write[slot] != 0 &&
                        !group_compaction_claimed_locked(*stripe, slot) &&
                        slot < stripe->group_is_split.size() &&
                        stripe->group_is_split[slot] == 0 &&
                        slot < stripe->group_live_mask.size()) {
                        const uint8_t mask = stripe->group_live_mask[slot];
                        if (mask != 0 && mask != 0xfu) {
                            SlotGroupHandle handle;
                            if (fill_group_handle(*stripe, stripe->bin,
                                                  slot, &handle)) {
                                CompactionGroupView view;
                                view.id = handle.id;
                                view.group = handle;
                                view.live_mask = mask;
                                view.live_count = static_cast<uint8_t>(
                                    __builtin_popcount(
                                        static_cast<unsigned>(mask)));
                                view.generation =
                                    stripe->group_generation[slot];
                                view.owners = stripe->group_owners[slot];
                                view.shadow_only = stripe->shadow_only;
                                out.push_back(view);
                            }
                        }
                    }
                    next_slot = (begin + n + 1) % slots;
                }
                stopped_inside = out.size() == budget;
            }
            if (stopped_inside) {
                stripe_cursor = current;
                slot_cursor = next_slot;
                return true;
            }
            current = (current + 1) % total;
            next_slot = 0;
            visited++;
        }
        stripe_cursor = current;
        slot_cursor = next_slot;
        return !out.empty();
    }
    bool try_claim_compaction_group(
        const CompactionGroupView &expected,
        CompactionGroupView *view_out = nullptr) {
        const SlotGroupId id =
            expected.id.valid() ? expected.id : expected.group.id;
        const uint64_t generation =
            expected.generation != 0 ? expected.generation
                                     : expected.group.generation;
        const bool wildcard_generation = generation == 0;
        if (!id.valid() ||
            expected.live_mask == 0 || expected.live_mask == 0xfu) {
            return false;
        }
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (!stripe_effectively_healthy_locked(*stripe) ||
            group_state_locked(*stripe, id.slot_id) != kSlotGroupSealed ||
            stripe->slot_size != 8192 ||
            id.slot_id >= stripe->group_generation.size() ||
            (!wildcard_generation &&
             stripe->group_generation[id.slot_id] != generation) ||
            id.slot_id >= stripe->group_ready_after_write.size() ||
            stripe->group_ready_after_write[id.slot_id] == 0 ||
            id.slot_id >= stripe->group_is_split.size() ||
            stripe->group_is_split[id.slot_id] != 0 ||
            id.slot_id >= stripe->group_live_mask.size() ||
            stripe->group_live_mask[id.slot_id] != expected.live_mask ||
            group_compaction_claimed_locked(*stripe, id.slot_id)) {
            return false;
        }
        SlotGroupHandle handle;
        if (!fill_group_handle(*stripe, stripe->bin, id.slot_id, &handle)) {
            return false;
        }
        set_group_compaction_claimed_locked(*stripe, id.slot_id, true);
        if (view_out != nullptr) {
            view_out->id = id;
            view_out->group = handle;
            view_out->live_mask = expected.live_mask;
            view_out->live_count = static_cast<uint8_t>(
                __builtin_popcount(static_cast<unsigned>(expected.live_mask)));
            view_out->generation = stripe->group_generation[id.slot_id];
            view_out->owners = stripe->group_owners[id.slot_id];
            view_out->shadow_only = stripe->shadow_only;
        }
        return true;
    }

    // True only for a live/in-progress split group at a data-segment address.
    // Claim/release and slot admission used by remote compaction.  A claim
    // is taken only for a sealed partial group and blocks both group reuse and
    // release-side requeue until the worker drops it.
    bool try_claim_compaction_group(
        const SlotGroupId &id, uint8_t expected_live_mask,
        CompactionGroupView *view_out = nullptr) {
        CompactionGroupView expected;
        expected.id = id;
        expected.live_mask = expected_live_mask;
        return try_claim_compaction_group(expected, view_out);
    }

    bool release_compaction_group_claim(
        const CompactionGroupView &expected) {
        const SlotGroupId id =
            expected.id.valid() ? expected.id : expected.group.id;
        const uint64_t generation =
            expected.generation != 0 ? expected.generation
                                     : expected.group.generation;
        if (!id.valid() || generation == 0) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        bool should_enqueue = false;
        uint16_t bin = 0;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (!group_compaction_claimed_locked(*stripe, id.slot_id) ||
                id.slot_id >= stripe->group_generation.size() ||
                stripe->group_generation[id.slot_id] != generation) {
                return false;
            }
            if (stripe->group_shadow_quarantined[id.slot_id]) return false;
            if (group_state_locked(*stripe, id.slot_id) ==
                    kSlotGroupSealed &&
                id.slot_id < stripe->group_live_mask.size() &&
                stripe->group_live_mask[id.slot_id] == 0) {
                (void)mark_dead_group_locked(*stripe, id);
            }
            set_group_compaction_claimed_locked(*stripe, id.slot_id, false);
            should_enqueue = mark_group_queued_if_has_space_locked(*stripe);
            bin = stripe->bin;
        }
        if (should_enqueue) push_group_candidate(bin, id.stripe_id);
        return true;
    }

    // Drop a compaction claim after all updates for the group have either
    // committed or rolled back.  Only here may a dead source group become a
    // reusable group-pool candidate.
    bool release_compaction_group_claim(const SlotGroupId &id) {
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        bool should_enqueue = false;
        uint16_t bin = 0;
        {
            std::lock_guard<std::mutex> lock(stripe->mutex);
            if (!group_compaction_claimed_locked(*stripe, id.slot_id)) {
                return false;
            }
            if (stripe->group_shadow_quarantined[id.slot_id]) return false;
            if (group_state_locked(*stripe, id.slot_id) ==
                    kSlotGroupSealed &&
                stripe->group_live_mask[id.slot_id] == 0) {
                (void)mark_dead_group_locked(*stripe, id);
            }
            set_group_compaction_claimed_locked(*stripe, id.slot_id, false);
            should_enqueue = mark_group_queued_if_has_space_locked(*stripe);
            bin = stripe->bin;
        }
        if (should_enqueue) push_group_candidate(bin, id.stripe_id);
        return true;
    }

    bool compaction_group_claimed(const SlotGroupId &id) const {
        const Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        return group_compaction_claimed_locked(*stripe, id.slot_id);
    }

    // Atomically transfer one owner slot between two claimed groups.  The
    // caller holds the corresponding EntryGuard BUSY window; no allocator
    // reuse can observe the intermediate source/destination masks.
    bool compaction_transfer_slot(const CompactionGroupView &src_view,
                                  const CompactionGroupView &dst_view,
                                  uint8_t data_slot,
                                  uintptr_t expected_owner) {
        const SlotGroupId src_id =
            src_view.id.valid() ? src_view.id : src_view.group.id;
        const SlotGroupId dst_id =
            dst_view.id.valid() ? dst_view.id : dst_view.group.id;
        const uint64_t src_generation =
            src_view.generation != 0 ? src_view.generation
                                     : src_view.group.generation;
        const uint64_t dst_generation =
            dst_view.generation != 0 ? dst_view.generation
                                     : dst_view.group.generation;
        if (!src_id.valid() || !dst_id.valid() ||
            (src_id.stripe_id == dst_id.stripe_id &&
             src_id.slot_id == dst_id.slot_id) ||
            data_slot >= kDataShards || expected_owner == 0 ||
            src_generation == 0 || dst_generation == 0) {
            return false;
        }
        Stripe *src = stripe_by_id(src_id.stripe_id);
        Stripe *dst = stripe_by_id(dst_id.stripe_id);
        if (src == nullptr || dst == nullptr ||
            src_id.slot_id >= src->slots_per_shard ||
            dst_id.slot_id >= dst->slots_per_shard) {
            return false;
        }

        std::unique_lock<std::mutex> first_lock;
        std::unique_lock<std::mutex> second_lock;
        if (src_id.stripe_id == dst_id.stripe_id) {
            first_lock = std::unique_lock<std::mutex>(src->mutex);
        } else if (src_id.stripe_id < dst_id.stripe_id) {
            first_lock = std::unique_lock<std::mutex>(src->mutex);
            second_lock = std::unique_lock<std::mutex>(dst->mutex);
        } else {
            first_lock = std::unique_lock<std::mutex>(dst->mutex);
            second_lock = std::unique_lock<std::mutex>(src->mutex);
        }

        const auto valid_group = [&](const Stripe &stripe,
                                     const SlotGroupId &id,
                                     uint64_t generation) {
            return stripe.slot_size == 8192 &&
                   group_state_locked(stripe, id.slot_id) ==
                       kSlotGroupSealed &&
                   id.slot_id < stripe.group_generation.size() &&
                   stripe.group_generation[id.slot_id] == generation &&
                   id.slot_id < stripe.group_ready_after_write.size() &&
                   stripe.group_ready_after_write[id.slot_id] != 0 &&
                   id.slot_id < stripe.group_is_split.size() &&
                   stripe.group_is_split[id.slot_id] == 0 &&
                   id.slot_id < stripe.group_live_mask.size() &&
                   id.slot_id < stripe.group_live_objects.size() &&
                   id.slot_id < stripe.group_owners.size() &&
                   group_compaction_claimed_locked(stripe, id.slot_id);
        };
        if (!valid_group(*src, src_id, src_generation) ||
            !valid_group(*dst, dst_id, dst_generation)) {
            return false;
        }
        const uint8_t bit = static_cast<uint8_t>(1u << data_slot);
        if ((src->group_live_mask[src_id.slot_id] & bit) == 0 ||
            (dst->group_live_mask[dst_id.slot_id] & bit) != 0 ||
            src->group_owners[src_id.slot_id][data_slot] != expected_owner ||
            src->group_live_objects[src_id.slot_id] == 0) {
            return false;
        }

        src->group_live_mask[src_id.slot_id] =
            static_cast<uint8_t>(src->group_live_mask[src_id.slot_id] & ~bit);
        src->group_live_objects[src_id.slot_id]--;
        src->live_count--;
        src->group_owners[src_id.slot_id][data_slot] = 0;

        dst->group_live_mask[dst_id.slot_id] |= bit;
        dst->group_live_objects[dst_id.slot_id]++;
        dst->live_count++;
        dst->group_owners[dst_id.slot_id][data_slot] = expected_owner;

        if (src->group_live_mask[src_id.slot_id] == 0) {
            src->group_ready_after_write[src_id.slot_id] = 0;
            src->group_owners[src_id.slot_id].fill(0);
            set_group_state_locked(*src, src_id.slot_id, kSlotGroupDead);
            src->group_dead_count++;
        }
        return true;
    }

    // Install one moved owner into a destination hole.  The source release is
    // deliberately separate: the caller must publish the new owner address
    // only after the transport reports all six writes complete.
    bool compaction_add_destination_slot(const SlotGroupId &id,
                                         uint8_t data_slot) {
        if (data_slot >= kDataShards) return false;
        Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (!group_compaction_claimed_locked(*stripe, id.slot_id) ||
            group_state_locked(*stripe, id.slot_id) != kSlotGroupSealed ||
            id.slot_id >= stripe->group_live_mask.size() ||
            (stripe->group_live_mask[id.slot_id] &
             static_cast<uint8_t>(1u << data_slot)) != 0) {
            return false;
        }
        stripe->group_live_mask[id.slot_id] |=
            static_cast<uint8_t>(1u << data_slot);
        if (id.slot_id < stripe->group_live_objects.size()) {
            stripe->group_live_objects[id.slot_id]++;
        }
        stripe->live_count++;
        return true;
    }

    // Resolve the source data address for a live slot while its claim is held.
    bool compaction_source_slot_addr(const SlotGroupId &id, uint8_t data_slot,
                                     uint64_t *addr_out) const {
        if (addr_out == nullptr || data_slot >= kDataShards) return false;
        const Stripe *stripe = stripe_by_id(id.stripe_id);
        if (stripe == nullptr || id.slot_id >= stripe->slots_per_shard) {
            return false;
        }
        std::lock_guard<std::mutex> lock(stripe->mutex);
        if (!group_compaction_claimed_locked(*stripe, id.slot_id) ||
            group_state_locked(*stripe, id.slot_id) != kSlotGroupSealed ||
            id.slot_id >= stripe->group_live_mask.size() ||
            (stripe->group_live_mask[id.slot_id] &
             static_cast<uint8_t>(1u << data_slot)) == 0) {
            return false;
        }
        SlotGroupHandle handle;
        if (!fill_group_handle(*stripe, stripe->bin, id.slot_id, &handle)) {
            return false;
        }
        *addr_out = resolve_rebuilt_addr(
            id.stripe_id, data_slot, handle.segments[data_slot].addr);
        return true;
    }
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
