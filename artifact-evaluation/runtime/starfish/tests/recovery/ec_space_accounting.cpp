// CPU regression checks for bounded EC-space accounting.  The test deliberately
// exercises only the stripe manager state machine; no RDMA requests are posted.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib {
rdma::Configure g_ec_space_accounting_config;
const rdma::Configure &get_config() { return g_ec_space_accounting_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using Manager = FarLib::cache::SmallObjectStripeManager;
using Usage = Manager::SpaceUsage;
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
    auto &config = FarLib::g_ec_space_accounting_config;
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

uint64_t endpoint_sum(const Usage &usage) {
    uint64_t sum = 0;
    for (uint64_t bytes : usage.endpoint_occupied_bytes) sum += bytes;
    return sum;
}

bool same_core(const Usage &a, const Usage &b) {
    return a.occupied_group_bytes == b.occupied_group_bytes &&
           a.sealed_group_bytes == b.sealed_group_bytes &&
           a.live_payload_bytes == b.live_payload_bytes &&
           a.live_data_slot_bytes == b.live_data_slot_bytes &&
           a.parity_bytes == b.parity_bytes &&
           a.trapped_hole_bytes == b.trapped_hole_bytes &&
           a.padding_bytes == b.padding_bytes &&
           a.group_counts_by_live == b.group_counts_by_live &&
           a.split_groups == b.split_groups &&
           a.in_progress_groups == b.in_progress_groups &&
           a.reusable_groups == b.reusable_groups &&
           a.reusable_group_bytes == b.reusable_group_bytes &&
           a.unknown_payload_objects == b.unknown_payload_objects;
}

int run() {
    Checks checks;
    checks.check(configure_fixture(), "EC fixture configured");
    auto &config = FarLib::g_ec_space_accounting_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;

    Manager manager;
    manager.init(config.remote_total, shard_size);
    checks.check(manager.enabled(), "stripe manager enabled");

    constexpr size_t kGroupSize = 64 * 1024;
    Manager::SlotGroupHandle group;
    checks.check(manager.allocate_slot_group(kGroupSize, &group),
                 "allocate ordinary group");
    const uint64_t slot = group.slot_size;
    Usage usage = manager.space_usage();
    checks.equal(usage.occupied_group_bytes, 6 * slot,
                 "in-progress group occupies six segments");
    checks.equal(usage.in_progress_groups, 1u,
                 "in-progress count increments");
    checks.equal(usage.sealed_group_bytes, 0u,
                 "in-progress group is not sealed yet");
    checks.equal(endpoint_sum(usage), 6 * slot,
                 "endpoint bytes sum to active group occupancy");
    checks.equal(usage.endpoint_occupied_bytes.size(),
                 static_cast<size_t>(config.server_count),
                 "endpoint vector follows configured endpoint count");

    const Usage before_invalid = usage;
    std::array<uint32_t, 4> bad_sizes{{1, static_cast<uint32_t>(slot + 1), 0,
                                       0}};
    checks.check(!manager.seal_slot_group(group.id, 0x3, bad_sizes.data()),
                 "invalid payload size rejects seal");
    checks.check(same_core(manager.space_usage(), before_invalid),
                 "invalid seal leaves all counters unchanged");
    GroupState state = GroupState::kSlotGroupFree;
    uint8_t mask = 0xff;
    checks.check(manager.get_slot_group_state(group.id, &state, &mask) &&
                     state == GroupState::kSlotGroupInProgress && mask == 0,
                 "invalid seal leaves group in progress");

    std::array<uint32_t, 4> sizes{{100, 200, 0, 0}};
    checks.check(manager.seal_slot_group(group.id, 0x3, sizes.data()),
                 "seal ordinary group with exact payloads");
    usage = manager.space_usage();
    checks.equal(usage.occupied_group_bytes, 6 * slot,
                 "sealed group remains physically occupied");
    checks.equal(usage.sealed_group_bytes, 6 * slot,
                 "sealed group bytes include six segments");
    checks.equal(usage.live_payload_bytes, 300u, "payload total is exact");
    checks.equal(usage.live_data_slot_bytes, 2 * slot,
                 "two live data slots are counted");
    checks.equal(usage.parity_bytes, 2 * slot,
                 "sealed parity bytes are counted");
    checks.equal(usage.trapped_hole_bytes, 2 * slot,
                 "two sealed holes are trapped");
    checks.equal(usage.padding_bytes, 2 * slot - 300,
                 "slot padding excludes payload bytes");
    checks.equal(usage.group_counts_by_live[2], 1u,
                 "ordinary sealed histogram records two survivors");
    checks.equal(usage.unknown_payload_objects, 0u,
                 "explicit payload sizes are known");

    uint32_t remaining = 0;
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     group.segments[0].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "4-to-3 object release is accepted");
    checks.equal(remaining, 1u, "release reports one remaining object");
    usage = manager.space_usage();
    checks.equal(usage.live_payload_bytes, 200u,
                 "partial release subtracts only released payload");
    checks.equal(usage.live_data_slot_bytes, slot,
                 "partial release subtracts one data slot");
    checks.equal(usage.trapped_hole_bytes, 3 * slot,
                 "partial release turns one slot into a trapped hole");
    checks.equal(usage.group_counts_by_live[1], 1u,
                 "histogram moves from two to one survivor");
    const Usage before_double = usage;
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     group.segments[0].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseRejected),
                 "double release is rejected");
    checks.check(same_core(manager.space_usage(), before_double),
                 "double release leaves counters unchanged");

    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     group.segments[1].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseGroupReleased),
                 "last survivor retires whole group");
    usage = manager.space_usage();
    checks.equal(usage.occupied_group_bytes, 0u,
                 "retired group leaves active occupancy at zero");
    checks.equal(usage.reusable_groups, 1u, "retired group becomes reusable");
    checks.equal(usage.reusable_group_bytes, 6 * slot,
                 "reusable bytes retain six segment reservation");
    checks.equal(endpoint_sum(usage), 0u,
                 "reusable groups are excluded from endpoint active bytes");
    checks.equal(usage.group_counts_by_live[1], 0u,
                 "retired group leaves sealed histogram");

    // Exercise the complete 4 -> 3 -> 2 -> 1 -> 0 survivor transition on a
    // second ordinary group.  The group stays physically occupied until the
    // final object is released.
    Manager::SlotGroupHandle all_live;
    bool found_reuse = false;
    for (size_t attempt = 0; attempt < group.slots_per_shard + 8; ++attempt) {
        Manager::SlotGroupHandle candidate;
        if (!manager.allocate_slot_group(kGroupSize, &candidate)) break;
        if (candidate.id.stripe_id == group.id.stripe_id &&
            candidate.id.slot_id == group.id.slot_id) {
            all_live = candidate;
            found_reuse = true;
            break;
        }
        // Keep probing without leaking a fresh candidate; each candidate is
        // retired as a whole and remains eligible for a later group reuse.
        if (!manager.mark_dead_group(candidate.id)) break;
    }
    checks.check(found_reuse, "reuse returns a previously retired group");
    std::array<uint32_t, 4> all_sizes{{1, 2, 3, 4}};
    checks.check(manager.seal_slot_group(all_live.id, 0xf, all_sizes.data()),
                 "seal four-survivor group");
    usage = manager.space_usage();
    checks.equal(usage.group_counts_by_live[4], 1u,
                 "histogram records four survivors");
    for (size_t i = 0; i < 4; ++i) {
        const auto expected = i == 3
                                  ? Release::kGroupReleaseGroupReleased
                                  : Release::kGroupReleaseObjectReleased;
        checks.equal(static_cast<uint8_t>(manager.release_group_object(
                         all_live.segments[i].addr, &remaining)),
                     static_cast<uint8_t>(expected),
                     "survivor transition releases one object");
        if (i < 3) {
            checks.equal(remaining, static_cast<uint32_t>(3 - i),
                         "survivor transition reports remainder");
        }
        const size_t live_after = 3 - i;
        const auto transition = manager.space_usage();
        checks.equal(transition.group_counts_by_live[live_after],
                     i == 3 ? 0u : 1u,
                     "survivor histogram follows live count");
        checks.equal(transition.occupied_group_bytes, i == 3 ? 0u : 6 * slot,
                     "physical occupancy remains until last survivor dies");
        checks.equal(transition.sealed_group_bytes,
                     transition.live_payload_bytes + transition.padding_bytes +
                         transition.trapped_hole_bytes + transition.parity_bytes,
                     "payload padding holes and parity close physical total");
    }

    Manager::SlotGroupHandle reused;
    const Usage before_reuse = manager.space_usage();
    checks.check(manager.allocate_slot_group(kGroupSize, &reused),
                 "allocate after whole-group reuse");
    usage = manager.space_usage();
    checks.equal(usage.in_progress_groups, 1u,
                 "reused group is in progress exactly once");
    checks.check(usage.reusable_groups <= before_reuse.reusable_groups,
                 "reusable reservation accounting is bounded");
    checks.equal(endpoint_sum(usage), 6 * slot,
                 "reused group has one active endpoint contribution");
    const Usage before_bad_split = usage;
    checks.check(!manager.seal_split_slot_group(
                     reused.id, static_cast<uint32_t>(4 * slot + 1)),
                 "oversized split payload rejects seal");
    checks.check(same_core(manager.space_usage(), before_bad_split),
                 "invalid split seal leaves counters unchanged");
    checks.check(manager.seal_split_slot_group(reused.id, 123),
                 "seal split group with one public payload");
    usage = manager.space_usage();
    checks.equal(usage.split_groups, 1u, "split group has separate count");
    checks.equal(usage.group_counts_by_live[4], 0u,
                 "split group is excluded from small histogram");
    checks.equal(usage.live_data_slot_bytes, 4 * slot,
                 "split group has four live internal slots");
    checks.equal(usage.live_payload_bytes, 123u,
                 "split group counts one public payload");
    checks.equal(usage.padding_bytes, 4 * slot - 123,
                 "split padding spans four internal slots");
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     reused.segments[0].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseGroupReleased),
                 "split anchor release retires all fragments");
    usage = manager.space_usage();
    checks.equal(usage.split_groups, 0u, "split count clears on release");
    checks.check(usage.reusable_groups >= 1u,
                 "released split group is reusable");

    // Whole abort and idempotence: an in-progress group is released once, and
    // a second abort is rejected without changing the snapshot.
    Manager::SlotGroupHandle aborted;
    checks.check(manager.allocate_slot_group(kGroupSize, &aborted),
                 "allocate group for whole abort");
    checks.check(manager.mark_dead_group(aborted.id),
                 "whole abort retires in-progress group");
    const Usage after_abort = manager.space_usage();
    checks.check(!manager.mark_dead_group(aborted.id),
                 "double whole abort is rejected");
    checks.check(same_core(manager.space_usage(), after_abort),
                 "double whole abort leaves counters unchanged");

    Manager::SlotGroupHandle unknown;
    checks.check(manager.allocate_slot_group(kGroupSize, &unknown),
                 "allocate legacy unknown-size group");
    checks.check(manager.seal_slot_group(unknown.id, 0x3),
                 "legacy seal without payload sizes remains supported");
    checks.equal(manager.space_usage().unknown_payload_objects, 2u,
                 "missing sizes are explicit, never silently exact");
    checks.equal(static_cast<uint8_t>(manager.release_group_object(unknown.segments[0].addr)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "release unknown-size survivor");
    checks.equal(manager.space_usage().unknown_payload_objects, 1u,
                 "unknown-size count follows survivors");
    checks.check(manager.mark_dead_group(unknown.id), "retire remaining unknown group");
    checks.equal(manager.space_usage().unknown_payload_objects, 0u,
                 "unknown-size count returns to zero");
    checks.equal(manager.space_usage().occupied_group_bytes, 0u,
                 "all accounting returns to zero live occupancy");

    std::printf("EC_SPACE_ACCOUNTING checks=%d failures=%d\n", checks.count,
                checks.failures);
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_SPACE_ACCOUNTING_SKIP_NO_VERBS");
    return 0;
#endif
}
