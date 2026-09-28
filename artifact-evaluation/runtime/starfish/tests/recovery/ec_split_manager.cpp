// CPU/fixture lifecycle checks for SmallObjectStripeManager's split-group
// ownership.  The manager uses the production RemoteGlobalHeap, so a full
// run needs the same lightweight global-heap/config fixture as the EC batch
// self-check.  When libibverbs is unavailable, the test compiles to an
// explicit skip so the codec-only CPU test suite remains buildable.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_split_manager_config;
const rdma::Configure &get_config() { return g_ec_split_manager_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using FarLib::cache::SmallObjectStripeManager;
using GroupRelease = SmallObjectStripeManager::GroupReleaseResult;
using GroupState = SmallObjectStripeManager::SlotGroupState;

struct Checks {
    int count = 0;
    int failures = 0;

    void check(bool value, const char *what) {
        ++count;
        if (value) {
            std::printf("  ok   %s\n", what);
        } else {
            ++failures;
            std::printf("  FAIL %s\n", what);
        }
    }

    template <typename T, typename U>
    void equal(T got, U want, const char *what) {
        ++count;
        if (got == want) {
            std::printf("  ok   %s\n", what);
            return;
        }
        ++failures;
        std::printf("  FAIL %s (got=%llu want=%llu)\n", what,
                    static_cast<unsigned long long>(got),
                    static_cast<unsigned long long>(want));
    }
};

bool configure_remote_fixture() {
    auto &config = FarLib::g_ec_split_manager_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    config.server_count =
        static_cast<int>(FarLib::rdma::Configure::ft_ec_batch_endpoint_count);
    config.server_buffer_size = 4ull * 1024 * 1024;
    config.client_buffer_size = 4ull * 1024 * 1024;
    config.remote_total =
        config.server_buffer_size * static_cast<size_t>(config.server_count);
    config.mapping_type = FarLib::rdma::Configure::MAPPING_RANGE;
    config.ft_method_type = FarLib::rdma::Configure::FT_EC_BATCH;
    config.ft_method = "ec_batch";
    config.exclusive_cache = true;
    config.ft_ec_data_shards = 4;
    config.ft_ec_parity_shards = 2;
    config.ft_small_object_cutoff = 4096;
    config.ft_small_stripe_shard_size_bytes = shard_size;
    ::FarLib::allocator::remote::remote_global_heap.register_remote(
        config.remote_total);
    return config.is_ec_batch_mode();
}

bool find_reused_group(
    SmallObjectStripeManager &manager,
    const SmallObjectStripeManager::SlotGroupHandle &original,
    size_t group_size, SmallObjectStripeManager::SlotGroupHandle *reused) {
    if (reused == nullptr) return false;
    // The group candidate pool normally returns the released slot first. If
    // another candidate is ahead of it, retire that candidate as a whole and
    // keep looking; no single data segment is ever released here.
    const size_t attempts = static_cast<size_t>(original.slots_per_shard) + 8;
    for (size_t attempt = 0; attempt < attempts; ++attempt) {
        SmallObjectStripeManager::SlotGroupHandle candidate;
        if (!manager.allocate_slot_group(group_size, &candidate)) return false;
        if (candidate.id.stripe_id == original.id.stripe_id &&
            candidate.id.slot_id == original.id.slot_id) {
            *reused = candidate;
            return true;
        }
        if (!manager.mark_dead_group(candidate.id)) return false;
    }
    return false;
}

int run() {
    Checks checks;
    checks.check(configure_remote_fixture(), "ec_batch fixture configured");

    const auto &config = FarLib::g_ec_split_manager_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    SmallObjectStripeManager manager;
    manager.init(config.remote_total, shard_size);
    checks.check(manager.enabled(), "stripe manager enabled");

    constexpr size_t kSplitGroupBytes = 64 * 1024;
    SmallObjectStripeManager::SlotGroupHandle split;
    checks.check(manager.allocate_slot_group(kSplitGroupBytes, &split),
                 "allocate split group");
    if (!split.id.valid()) {
        checks.check(false, "split group id is valid");
        return checks.failures == 0 ? 0 : 1;
    }

    GroupState state = GroupState::kSlotGroupFree;
    uint8_t mask = 0;
    uint32_t count = 0;
    checks.check(manager.get_slot_group_state(split.id, &state, &mask) &&
                     state == GroupState::kSlotGroupInProgress && mask == 0,
                 "split group starts in_progress with empty mask");
    checks.check(manager.seal_split_slot_group(split.id),
                 "seal split group");
    checks.check(manager.get_slot_group_state(split.id, &state, &mask) &&
                     state == GroupState::kSlotGroupSealed && mask == 0xf,
                 "split group publishes live mask 0xF");
    checks.check(manager.get_slot_group_object_count(split.id, &count) &&
                     count == 4,
                 "split group internally counts four live fragments");
    for (size_t i = 0; i < 4; ++i) {
        checks.check(manager.slot_group_is_split(split.segments[i].addr),
                     "all four data addresses identify the split group");
    }

    // The public owner is data[0].  Every other data fragment must be
    // rejected without changing mask/count/state.
    for (size_t i = 1; i < 4; ++i) {
        uint32_t remaining = 0xffffffffu;
        checks.equal(
            static_cast<uint8_t>(manager.release_group_object(
                split.segments[i].addr, &remaining)),
            static_cast<uint8_t>(GroupRelease::kGroupReleaseRejected),
            "non-anchor split fragment release is rejected");
        checks.equal(remaining, 0u,
                     "non-anchor rejection reports no public remainder");
        checks.check(manager.get_slot_group_state(split.id, &state, &mask) &&
                         state == GroupState::kSlotGroupSealed && mask == 0xf,
                     "non-anchor rejection leaves split mask unchanged");
        checks.check(manager.get_slot_group_object_count(split.id, &count) &&
                         count == 4,
                     "non-anchor rejection leaves fragment count unchanged");
        checks.check(!manager.mark_dead(split.segments[i].addr),
                     "single-object mark_dead rejects split fragments");
    }
    checks.equal(
        static_cast<uint8_t>(manager.release_group_object(split.segments[4].addr)),
        static_cast<uint8_t>(GroupRelease::kGroupReleaseNotGrouped),
        "parity address remains outside data ownership mapping");

    // One anchor release retires all six segments in one transition.
    uint32_t remaining = 0xffffffffu;
    checks.equal(
        static_cast<uint8_t>(manager.release_group_object(
            split.segments[0].addr, &remaining)),
        static_cast<uint8_t>(GroupRelease::kGroupReleaseGroupReleased),
        "split anchor release retires the whole group");
    checks.equal(remaining, 0u, "whole split release has no remainder");
    checks.check(manager.get_slot_group_state(split.id, &state, &mask) &&
                     state == GroupState::kSlotGroupDead && mask == 0,
                 "whole split release leaves dead group with empty mask");
    checks.check(manager.get_slot_group_object_count(split.id, &count) &&
                     count == 0,
                 "dead split group has no live fragments");
    checks.check(!manager.slot_group_is_split(split.segments[0].addr),
                 "split flag clears after group retirement");
    checks.equal(
        static_cast<uint8_t>(manager.release_group_object(
            split.segments[0].addr)),
        static_cast<uint8_t>(GroupRelease::kGroupReleaseRejected),
        "second split anchor release is rejected");
    checks.check(manager.slot_group_addr_is_settled(split.segments[0].addr),
                 "dead split anchor is settled");

    // Reuse exactly the released offset, then exercise the ordinary small
    // group lifecycle on that same physical group.  This proves reuse clears
    // split metadata rather than inheriting the old public-owner rule.
    SmallObjectStripeManager::SlotGroupHandle reused;
    const bool found = find_reused_group(manager, split, kSplitGroupBytes,
                                         &reused);
    checks.check(found, "released group offset is reused as a whole");
    if (found) {
        checks.check(reused.slot_offset == split.slot_offset,
                     "reused group keeps the original slot offset");
        bool same_segments = true;
        for (size_t i = 0; i < split.segments.size(); ++i) {
            same_segments = same_segments &&
                            reused.segments[i].addr == split.segments[i].addr;
        }
        checks.check(same_segments,
                     "reuse returns all six original segment addresses");
        checks.check(!manager.slot_group_is_split(reused.segments[0].addr),
                     "group reuse clears split flag before sealing");
        checks.check(manager.get_slot_group_state(reused.id, &state, &mask) &&
                         state == GroupState::kSlotGroupInProgress && mask == 0,
                     "reused group starts as ordinary in_progress");
        checks.check(manager.seal_slot_group(reused.id, 0x03),
                     "ordinary small group still seals with a partial mask");
        checks.check(manager.get_slot_group_state(reused.id, &state, &mask) &&
                         state == GroupState::kSlotGroupSealed && mask == 0x03,
                     "ordinary small group preserves mask 0x03");
        checks.check(manager.get_slot_group_object_count(reused.id, &count) &&
                         count == 2,
                     "ordinary small group counts two live objects");
        checks.equal(
            static_cast<uint8_t>(manager.release_group_object(
                reused.segments[2].addr)),
            static_cast<uint8_t>(GroupRelease::kGroupReleaseRejected),
            "ordinary small group hole remains non-releasable");
        checks.equal(
            static_cast<uint8_t>(manager.release_group_object(
                reused.segments[0].addr, &remaining)),
            static_cast<uint8_t>(GroupRelease::kGroupReleaseObjectReleased),
            "ordinary small group releases one live object only");
        checks.equal(remaining, 1u, "ordinary group remainder is one");
        checks.equal(
            static_cast<uint8_t>(manager.release_group_object(
                reused.segments[1].addr, &remaining)),
            static_cast<uint8_t>(GroupRelease::kGroupReleaseGroupReleased),
            "ordinary small group releases on its last live object");
        checks.check(!manager.slot_group_is_split(reused.segments[0].addr),
                     "ordinary group remains non-split after release");
    }

    std::printf("EC_SPLIT_MANAGER checks=%d failures=%d\n", checks.count,
                checks.failures);
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_SPLIT_MANAGER_SKIP_NO_VERBS");
    return 0;
#endif
}
