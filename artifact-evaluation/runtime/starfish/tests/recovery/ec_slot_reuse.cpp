// CPU-only lifecycle checks for one-slot EC reuse.  No RDMA requests are
// posted: the test exercises the allocator's reservation, publication, ABA,
// and accounting gates that surround the three-part update protocol.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_slot_reuse_config;
const rdma::Configure &get_config() { return g_ec_slot_reuse_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using Manager = FarLib::cache::SmallObjectStripeManager;
using GroupState = Manager::SlotGroupState;
using Release = Manager::GroupReleaseResult;

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

bool configure_fixture() {
    auto &config = FarLib::g_ec_slot_reuse_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    config.server_count = static_cast<int>(
        FarLib::rdma::Configure::ft_ec_batch_endpoint_count);
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

bool seal_partial(Manager &manager, size_t size, uint32_t behavior,
                  uint8_t live_mask, Manager::SlotGroupHandle *group,
                  bool durable) {
    if (group == nullptr || !manager.allocate_slot_group(size, group)) {
        return false;
    }
    std::array<uint32_t, 4> payloads{};
    for (size_t shard = 0; shard < 4; ++shard) {
        if ((live_mask & static_cast<uint8_t>(1u << shard)) != 0) {
            payloads[shard] = group->slot_size;
        }
    }
    if (!manager.seal_slot_group(group->id, live_mask, payloads.data(),
                                 behavior)) {
        return false;
    }
    if (durable) manager.mark_slot_group_durable(group->id);
    return true;
}

int run() {
    Checks checks;
    checks.check(configure_fixture(), "ec_batch fixture configured");
    const auto &config = FarLib::g_ec_slot_reuse_config;
    Manager manager;
    manager.init(config.remote_total,
                 FarLib::rdma::Configure::ft_ec_shard_size_bytes);
    checks.check(manager.enabled(), "stripe manager initialized");
    if (!manager.enabled()) return checks.failures == 0 ? 0 : 1;

    constexpr size_t kGroupBytes = 64 * 1024;
    constexpr size_t kOtherClassBytes = 128 * 1024;

    Manager::SlotGroupHandle not_durable;
    checks.check(seal_partial(manager, kGroupBytes, 11, 0x03, &not_durable,
                              false),
                 "sealed partial group starts non-durable");
    checks.check(
        !manager.begin_slot_reuse(not_durable.slot_size, 11, nullptr),
        "null reuse output is rejected");
    Manager::SlotReuse reuse{};
    checks.check(!manager.begin_slot_reuse(not_durable.slot_size, 11, &reuse),
                 "non-durable group is not reusable");

    Manager::SlotGroupHandle split;
    checks.check(manager.allocate_slot_group(kGroupBytes, &split),
                 "allocate split candidate");
    checks.check(manager.seal_split_slot_group(split.id,
                                               static_cast<uint32_t>(split.slot_size)),
                 "seal split candidate");
    manager.mark_slot_group_durable(split.id);
    checks.check(!manager.begin_slot_reuse(split.slot_size, 0, &reuse),
                 "split group is not reusable");

    Manager::SlotGroupHandle wrong_class;
    checks.check(seal_partial(manager, kOtherClassBytes, 12, 0x01,
                              &wrong_class, true),
                 "seal durable different-class group");
    checks.check(!manager.begin_slot_reuse(not_durable.slot_size, 12, &reuse),
                 "canonical size class is required");

    Manager::SlotGroupHandle wrong_behavior;
    checks.check(seal_partial(manager, kGroupBytes, 13, 0x01,
                              &wrong_behavior, true),
                 "seal durable wrong-behavior group");
    checks.check(!manager.begin_slot_reuse(not_durable.slot_size, 14, &reuse),
                 "behavior group is required");

    // Release one old object before reserving its replacement.  The first
    // hole is shard 0 after this release, and the reservation must not expose
    // that bit in the committed mask until COMMIT.
    Manager::SlotGroupHandle partial;
    checks.check(seal_partial(manager, kGroupBytes, 21, 0x03, &partial, true),
                 "seal durable partial reuse candidate");
    uint32_t remaining = 0;
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     partial.segments[0].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "release one old object leaves a hole");
    checks.equal(remaining, 1u, "partial release reports one survivor");
    checks.check(manager.begin_slot_reuse(partial.slot_size, 21, &reuse),
                 "partial durable group reserves one hole");
    checks.equal(reuse.data_shard, static_cast<uint8_t>(0),
                 "reservation targets the retired data shard");
    checks.check(manager.slot_group_update_pending(partial.id),
                 "reservation publishes pending group gate");
    checks.check(manager.addr_update_pending(
                     reuse.group.segments[reuse.data_shard].addr),
                 "reservation publishes pending address gate");
    Manager::SlotGroupState state = GroupState::kSlotGroupFree;
    uint8_t live_mask = 0xff;
    checks.check(manager.get_slot_group_state(partial.id, &state, &live_mask) &&
                     state == GroupState::kSlotGroupSealed && live_mask == 0x02,
                 "committed live mask stays unchanged during PREPARE");
    Manager::SlotReuse second;
    bool busy = false;
    checks.check(!manager.begin_slot_reuse(partial.slot_size, 21, &second, &busy),
                 "second update skips busy group");
    checks.check(busy, "deferred sender can distinguish occupied update tag");
    busy = true;
    checks.check(!manager.begin_slot_reuse(partial.slot_size, 999, &second, &busy)
                     && !busy,
                 "wrong behavior does not report another class as busy");

    // The final old object may retire while PREPARE is outstanding.  COMMIT
    // then publishes the replacement as the sole live object and closes the
    // hole accounting without retiring the six-segment reservation.
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     partial.segments[1].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "last old object is deferred while update is pending");
    checks.equal(remaining, 0u, "deferred final release reports zero survivors");
    checks.check(manager.slot_group_update_pending(partial.id),
                 "pending survives final old-object release");
    checks.check(manager.get_slot_group_state(partial.id, &state, &live_mask) &&
                     state == GroupState::kSlotGroupSealed && live_mask == 0,
                 "deferred final release keeps group sealed");
    const auto before_commit = manager.space_usage();
    checks.check(manager.commit_slot_reuse(reuse),
                 "COMMIT publishes the replacement slot");
    checks.check(!manager.slot_group_update_pending(partial.id) &&
                     !manager.addr_update_pending(
                         reuse.group.segments[reuse.data_shard].addr),
                 "COMMIT clears both pending gates");
    checks.check(manager.get_slot_group_state(partial.id, &state, &live_mask) &&
                     state == GroupState::kSlotGroupSealed && live_mask == 0x01,
                 "COMMIT publishes only the replacement bit");
    const auto after_commit = manager.space_usage();
    checks.equal(after_commit.occupied_group_bytes,
                 before_commit.occupied_group_bytes,
                 "slot reuse preserves physical group occupancy");
    checks.equal(after_commit.live_data_slot_bytes,
                 before_commit.live_data_slot_bytes + partial.slot_size,
                 "slot reuse restores one live data slot");
    checks.equal(after_commit.trapped_hole_bytes,
                 before_commit.trapped_hole_bytes - partial.slot_size,
                 "slot reuse removes one trapped hole");
    checks.check(!manager.commit_slot_reuse(reuse),
                 "duplicate COMMIT is rejected");
    checks.check(!manager.abort_slot_reuse(reuse),
                 "duplicate ABORT is rejected");

    // A new generation on the same group fences a delayed callback from the
    // prior transaction.  Abort the new reservation to leave the group live.
    Manager::SlotReuse next_generation;
    checks.check(manager.begin_slot_reuse(partial.slot_size, 21,
                                          &next_generation),
                 "same group admits a later generation");
    checks.check(!manager.commit_slot_reuse(reuse),
                 "stale generation cannot COMMIT a later reservation");
    checks.check(manager.abort_slot_reuse(next_generation),
                 "later generation can ABORT");

    // The abort path must retire an empty old group only after clearing its
    // pending bit, and must close all physical accounting exactly once.
    Manager::SlotGroupHandle abort_group;
    checks.check(seal_partial(manager, kGroupBytes, 31, 0x01, &abort_group,
                              true),
                 "seal durable abort candidate");
    Manager::SlotReuse abort_reuse;
    checks.check(manager.begin_slot_reuse(abort_group.slot_size, 31,
                                          &abort_reuse),
                 "reserve abort candidate");
    const auto before_abort_release = manager.space_usage();
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     abort_group.segments[0].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "abort candidate final release is deferred");
    checks.check(manager.abort_slot_reuse(abort_reuse),
                 "ABORT retires empty old group");
    checks.check(!manager.slot_group_update_pending(abort_group.id) &&
                     manager.slot_group_addr_is_settled(
                         abort_group.segments[0].addr),
                 "ABORT closes pending and settled gates");
    checks.check(manager.get_slot_group_state(abort_group.id, &state,
                                              &live_mask) &&
                     state == GroupState::kSlotGroupDead,
                 "ABORT transitions empty group to dead");
    const auto after_abort = manager.space_usage();
    checks.equal(after_abort.occupied_group_bytes,
                 before_abort_release.occupied_group_bytes -
                     6ull * abort_group.slot_size,
                 "ABORT releases physical group occupancy once");

    // Failure is monotone: once one endpoint is fenced, no new independent
    // update may start, even for a different candidate group.
    Manager::SlotGroupHandle dead_endpoint_group;
    checks.check(seal_partial(manager, kGroupBytes, 41, 0x01,
                              &dead_endpoint_group, true),
                 "seal endpoint-dead candidate");
    const uint32_t dead_endpoint =
        dead_endpoint_group.segments[0].endpoint_idx;
    checks.check(manager.endpoint_is_alive(dead_endpoint),
                 "candidate endpoint starts alive");
    checks.check(manager.mark_endpoint_dead(dead_endpoint),
                 "fence candidate endpoint");
    checks.check(!manager.begin_slot_reuse(dead_endpoint_group.slot_size, 41,
                                           &reuse),
                 "endpoint failure blocks new slot reuse");

    std::printf("EC_SLOT_REUSE checks=%d failures=%d\n", checks.count,
                checks.failures);
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_SLOT_REUSE_SKIP_NO_VERBS");
    return 0;
#endif
}
