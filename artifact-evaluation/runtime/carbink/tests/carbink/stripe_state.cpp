// Real CPU-only lifecycle checks for Carbink slot-group compaction state.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_carbink_state_config;
const rdma::Configure &get_config() { return g_carbink_state_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using Manager = FarLib::cache::SmallObjectStripeManager;
using Handle = Manager::SlotGroupHandle;
using View = Manager::CompactionGroupView;
using State = Manager::SlotGroupState;
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

bool same_id(const Manager::SlotGroupId &a, const Manager::SlotGroupId &b) {
    return a.stripe_id == b.stripe_id && a.slot_id == b.slot_id;
}

void make_owners(uintptr_t seed, std::array<uintptr_t, 4> *owners) {
    for (size_t i = 0; i < owners->size(); ++i) {
        (*owners)[i] = seed + static_cast<uintptr_t>(i + 1) * 0x100;
    }
}

bool configure_remote_fixture() {
    auto &config = FarLib::g_carbink_state_config;
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

bool collect_one(Manager &manager, const Manager::SlotGroupId &id,
                 View *view_out) {
    size_t stripe_cursor = 0;
    size_t slot_cursor = 0;
    std::vector<View> candidates;
    manager.collect_compaction_candidates(stripe_cursor, slot_cursor, 128,
                                           candidates);
    for (const View &view : candidates) {
        if (same_id(view.id, id)) {
            if (view_out != nullptr) *view_out = view;
            return true;
        }
    }
    return false;
}

bool find_reused_group(Manager &manager, const Handle &original,
                       size_t group_size, Handle *reused) {
    if (reused == nullptr) return false;
    const size_t attempts =
        static_cast<size_t>(original.slots_per_shard) + 128;
    for (size_t attempt = 0; attempt < attempts; ++attempt) {
        Handle candidate;
        if (!manager.allocate_slot_group(group_size, &candidate)) return false;
        if (same_id(candidate.id, original.id)) {
            *reused = candidate;
            return true;
        }
        if (!manager.mark_dead_group(candidate.id)) return false;
    }
    return false;
}

int run() {
    Checks checks;
    checks.check(configure_remote_fixture(), "Carbink fixture configured");

    const auto &config = FarLib::g_carbink_state_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    Manager manager;
    manager.init(config.remote_total, shard_size);
    checks.check(manager.enabled(), "stripe manager enabled");

    constexpr size_t kGroupBytes = 8192;
    constexpr size_t kGroupsPerStripe = 32;
    std::vector<Handle> groups;
    groups.reserve(kGroupsPerStripe);
    std::vector<std::array<uintptr_t, 4>> owners;
    owners.reserve(kGroupsPerStripe);

    for (size_t i = 0; i < kGroupsPerStripe; ++i) {
        Handle group;
        checks.check(manager.allocate_slot_group(kGroupBytes, &group),
                     "allocate real 8KiB slot group");
        if (!group.id.valid()) {
            checks.check(false, "allocated group has valid id");
            return 1;
        }
        if (!groups.empty()) {
            checks.check(group.id.stripe_id == groups.front().id.stripe_id,
                         "all 32 groups fit one stripe");
        }
        groups.push_back(group);
        std::array<uintptr_t, 4> group_owners{};
        make_owners(0x1000000 + i * 0x10000, &group_owners);
        owners.push_back(group_owners);
    }
    if (groups.size() != kGroupsPerStripe) {
        std::printf("CARBINK_STRIPE_LIFECYCLE checks=%d failures=%d\n",
                    checks.count, checks.failures + 1);
        return 1;
    }

    for (size_t i = 0; i < groups.size(); ++i) {
        checks.check(manager.seal_slot_group(groups[i].id, 0xf),
                     "seal full group before publish");
        checks.check(manager.publish_compaction_ready(groups[i], owners[i]),
                     "publish full group after six-write barrier");
    }

    size_t stripe_cursor = 0;
    size_t slot_cursor = 0;
    std::vector<View> candidates;
    checks.check(!manager.collect_compaction_candidates(
                     stripe_cursor, slot_cursor, 128, candidates) &&
                     candidates.empty(),
                 "full 0xf groups are not partial scan candidates");

    const Handle src = groups[0];
    const Handle dst = groups[1];
    checks.check(src.id.stripe_id == dst.id.stripe_id,
                 "source and destination share stripe for one-lock path");

    uint32_t remaining = 0;
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     src.segments[2].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "source release slot 2");
    checks.equal(remaining, 3u, "source has three objects after first release");
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     src.segments[3].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "source release slot 3");
    checks.equal(remaining, 2u, "source has two objects after second release");

    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     dst.segments[0].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "destination release slot 0");
    checks.equal(remaining, 3u, "destination has three objects after first release");
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     dst.segments[1].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "destination release slot 1");
    checks.equal(remaining, 2u, "destination has two objects after second release");

    State state = State::kSlotGroupFree;
    uint8_t mask = 0;
    checks.check(manager.get_slot_group_state(src.id, &state, &mask) &&
                     state == State::kSlotGroupSealed && mask == 0x3,
                 "source full publish becomes partial mask 0x3");
    checks.check(manager.get_slot_group_state(dst.id, &state, &mask) &&
                     state == State::kSlotGroupSealed && mask == 0xc,
                 "destination full publish becomes partial mask 0xc");

    View src_candidate;
    View dst_candidate;
    checks.check(collect_one(manager, src.id, &src_candidate) &&
                     src_candidate.live_mask == 0x3 &&
                     src_candidate.generation == src.generation &&
                     src_candidate.owners[0] == owners[0][0] &&
                     src_candidate.owners[1] == owners[0][1] &&
                     src_candidate.owners[2] == 0 &&
                     src_candidate.owners[3] == 0,
                 "scanner finds source with exact generation and owners");
    checks.check(collect_one(manager, dst.id, &dst_candidate) &&
                     dst_candidate.live_mask == 0xc &&
                     dst_candidate.generation == dst.generation &&
                     dst_candidate.owners[0] == 0 &&
                     dst_candidate.owners[1] == 0 &&
                     dst_candidate.owners[2] == owners[1][2] &&
                     dst_candidate.owners[3] == owners[1][3],
                 "scanner finds destination with complementary holes");

    View stale = src_candidate;
    stale.generation++;
    checks.check(!manager.try_claim_compaction_group(stale, nullptr),
                 "wrong-generation claim is rejected");

    View src_claim;
    View dst_claim;
    checks.check(manager.try_claim_compaction_group(src_candidate, &src_claim),
                 "source claim succeeds");
    checks.check(manager.try_claim_compaction_group(dst_candidate, &dst_claim),
                 "destination claim succeeds");

    Handle blocked_destination;
    checks.check(manager.allocate_slot_group(kGroupBytes, &blocked_destination) &&
                     !same_id(blocked_destination.id, dst.id),
                 "claimed destination is not reused");
    checks.check(manager.mark_dead_group(blocked_destination.id),
                 "cleanup allocation blocked by destination claim");

    checks.check(manager.compaction_transfer_slot(
                     src_claim, dst_claim, 0, owners[0][0]),
                 "transfer source slot 0 into complementary destination hole");
    checks.check(manager.compaction_transfer_slot(
                     src_claim, dst_claim, 1, owners[0][1]),
                 "transfer source slot 1 into complementary destination hole");
    checks.check(manager.get_slot_group_state(src.id, &state, &mask) &&
                     state == State::kSlotGroupDead && mask == 0,
                 "source becomes dead after two transfers");
    checks.check(manager.get_slot_group_state(dst.id, &state, &mask) &&
                     state == State::kSlotGroupSealed && mask == 0xf,
                 "destination becomes full after two transfers");
    uint32_t object_count = 0;
    checks.check(manager.get_slot_group_object_count(dst.id, &object_count) &&
                     object_count == 4,
                 "destination object count returns to four");

    Handle blocked_source;
    checks.check(manager.allocate_slot_group(kGroupBytes, &blocked_source) &&
                     !same_id(blocked_source.id, src.id),
                 "claimed dead source is not reused");
    checks.check(manager.mark_dead_group(blocked_source.id),
                 "cleanup allocation blocked by source claim");

    View stale_claim = src_claim;
    stale_claim.generation++;
    checks.check(!manager.release_compaction_group_claim(stale_claim),
                 "stale claim release is rejected");
    checks.check(manager.release_compaction_group_claim(dst_claim),
                 "destination claim release succeeds");
    checks.check(manager.release_compaction_group_claim(src_claim),
                 "source claim release succeeds");
    Handle reused;
    checks.check(find_reused_group(manager, src, kGroupBytes, &reused) &&
                     same_id(reused.id, src.id) &&
                     reused.generation > src.generation,
                 "released source offset reuses with new generation");

    checks.check(!manager.publish_compaction_ready(src, owners[0]),
                 "old source handle cannot publish after reuse");
    checks.check(manager.seal_slot_group(reused.id, 0x3),
                 "reuse seals a new partial group");
    std::array<uintptr_t, 4> reused_owners{};
    make_owners(0x7000000, &reused_owners);
    checks.check(manager.publish_compaction_ready(reused, reused_owners),
                 "new generation publishes");
    checks.check(!manager.try_claim_compaction_group(src_candidate, nullptr),
                 "old candidate handle cannot claim new generation");
    checks.check(!manager.release_compaction_group_claim(src_candidate),
                 "old candidate handle cannot release new generation");

    // Repeat the transfer with a destination that is emptied while claimed.
    // This covers the foreground-fetch race: Sealed+ready mask 0 remains
    // protected until claim release, and the worker may fill it afterwards.
    const Handle empty_dst = groups[2];
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     empty_dst.segments[0].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "empty-destination setup releases slot 0");
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     empty_dst.segments[1].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "empty-destination setup releases slot 1");
    View reused_candidate;
    View empty_dst_candidate;
    checks.check(collect_one(manager, reused.id, &reused_candidate) &&
                     reused_candidate.live_mask == 0x3 &&
                     reused_candidate.generation == reused.generation,
                 "new-generation source is scanner-visible");
    checks.check(collect_one(manager, empty_dst.id, &empty_dst_candidate) &&
                     empty_dst_candidate.live_mask == 0xc &&
                     empty_dst_candidate.generation == empty_dst.generation,
                 "second destination has complementary holes");
    View reused_claim;
    View empty_dst_claim;
    checks.check(manager.try_claim_compaction_group(reused_candidate,
                                                    &reused_claim),
                 "new-generation source claim succeeds");
    checks.check(manager.try_claim_compaction_group(empty_dst_candidate,
                                                    &empty_dst_claim),
                 "second destination claim succeeds");
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     empty_dst.segments[2].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseObjectReleased),
                 "claimed empty destination releases slot 2");
    checks.equal(static_cast<uint8_t>(manager.release_group_object(
                     empty_dst.segments[3].addr, &remaining)),
                 static_cast<uint8_t>(Release::kGroupReleaseGroupReleased),
                 "claimed empty destination releases to zero");
    checks.check(manager.get_slot_group_state(
                     empty_dst.id, &state, &mask) &&
                     state == State::kSlotGroupSealed && mask == 0 &&
                     manager.compaction_group_claimed(empty_dst.id),
                 "claimed empty destination stays sealed and protected");
    Handle blocked_empty_destination;
    checks.check(manager.allocate_slot_group(kGroupBytes,
                                             &blocked_empty_destination) &&
                     !same_id(blocked_empty_destination.id, empty_dst.id),
                 "empty claimed destination is not reused");
    checks.check(manager.mark_dead_group(blocked_empty_destination.id),
                 "cleanup allocation blocked by empty destination claim");
    checks.check(manager.compaction_transfer_slot(
                     reused_claim, empty_dst_claim, 0, reused_owners[0]),
                 "transfer into claimed empty destination slot 0");
    checks.check(manager.compaction_transfer_slot(
                     reused_claim, empty_dst_claim, 1, reused_owners[1]),
                 "transfer into claimed empty destination slot 1");
    checks.check(manager.get_slot_group_state(
                     empty_dst.id, &state, &mask) &&
                     state == State::kSlotGroupSealed && mask == 0x3,
                 "empty destination receives two transferred objects");
    checks.check(manager.get_slot_group_object_count(
                     empty_dst.id, &object_count) &&
                     object_count == 2,
                 "empty destination object count is two after transfer");
    checks.check(manager.get_slot_group_state(
                     reused.id, &state, &mask) &&
                     state == State::kSlotGroupDead && mask == 0,
                 "new-generation source becomes dead after empty transfer");
    checks.check(manager.release_compaction_group_claim(empty_dst_claim),
                 "second destination claim releases");
    checks.check(manager.release_compaction_group_claim(reused_claim),
                 "new-generation source claim releases");
    checks.check(manager.mark_dead_group(empty_dst.id),
                 "cleanup partially filled empty destination");

    Handle unpublished;
    checks.check(manager.allocate_slot_group(kGroupBytes, &unpublished),
                 "allocate group for pre-publish scan check");
    checks.check(manager.seal_slot_group(unpublished.id, 0x3),
                 "seal group before six-write publication");
    View should_not_scan;
    checks.check(!collect_one(manager, unpublished.id, &should_not_scan),
                 "sealed group before publish is absent from scanner");
    checks.check(manager.mark_dead_group(unpublished.id),
                 "cleanup unpublished group");

    std::printf("CARBINK_STRIPE_LIFECYCLE checks=%d failures=%d\n",
                checks.count, checks.failures);
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("CARBINK_STRIPE_LIFECYCLE_SKIP_NO_VERBS");
    return 0;
#endif
}
