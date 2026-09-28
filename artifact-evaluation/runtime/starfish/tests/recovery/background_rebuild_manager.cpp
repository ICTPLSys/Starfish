#define FARLIB_REBUILD_TEST_HOOK 1
#include "cache/alloc/small_object_stripe.hpp"
#include <cassert>
#include <iostream>

namespace FarLib {
static rdma::Configure config;
const rdma::Configure &get_config() { return config; }
}
namespace FarLib::allocator::remote { RemoteGlobalHeap remote_global_heap; }

int main(int argc, char **) {
    using M = FarLib::cache::SmallObjectStripeManager;
    using S = M::BackgroundRebuildStatus;
    auto &cfg = FarLib::config;
    cfg.server_count = 8;
    cfg.server_buffer_size = 4ull << 20;
    cfg.remote_total = cfg.server_count * cfg.server_buffer_size;
    cfg.mapping_type = FarLib::rdma::Configure::MAPPING_RANGE;
    cfg.ft_method_type = FarLib::rdma::Configure::FT_EC_BATCH;
    cfg.ft_method = "ec_batch"; cfg.exclusive_cache = true;
    cfg.ft_standby_endpoint = 7; cfg.ft_background_rebuild = true;
    auto &heap = FarLib::allocator::remote::remote_global_heap;
    heap.register_remote(cfg.remote_total);
    M m;
    m.init(cfg.remote_total, FarLib::cache::SmallObjectStripeShardSize);
    if (argc > 1) {
        M::background_test_after_create = +[](M *manager, uint64_t id) {
            M::background_test_after_create = nullptr;
            assert(manager->mark_endpoint_dead(0));
            assert(manager->background_rebuild_scan_limit() == 1);
            M::BackgroundRebuildStripe empty;
            assert(manager->background_rebuild_view(id, 0, &empty) == S::Skip);
        };
        M::SlotGroupHandle g;
        assert(!m.allocate_slot_group(128 << 10, &g));
        assert(m.count_rebuild_remaining(0) == 0);
        assert(m.allocate_slot_group(128 << 10, &g));
        for (const auto &s : g.segments) assert(s.endpoint_idx != 0);
        std::cout << "creation/failure interleaving gate passed\n";
        return 0;
    }
    auto seal = [&](M::SlotGroupHandle &g, uint8_t mask, bool durable) {
        uint32_t sizes[4]{};
        for (unsigned s = 0; s < 4; ++s) if (mask & (1u << s)) sizes[s] = g.slot_size;
        assert(m.seal_slot_group(g.id, mask, sizes, 11));
        if (durable) m.mark_slot_group_durable(g.id);
    };
    M::SlotGroupHandle a, b, other, parity;
    assert(m.allocate_slot_group(128 << 10, &a)); seal(a, 3, true);
    assert(m.allocate_slot_group(128 << 10, &b));
    assert(a.id.stripe_id == b.id.stripe_id);
    assert(m.allocate_slot_group(64 << 10, &other)); seal(other, 1, true);
    assert(m.allocate_slot_group(32 << 10, &parity)); seal(parity, 1, true);
    M::SlotGroupHandle empty_initial;
    assert(m.allocate_slot_group(16 << 10, &empty_initial));
    const auto failed = a.segments[0].endpoint_idx;
    M::BackgroundRebuildStripe v;
    assert(m.background_rebuild_view(a.id.stripe_id, failed, &v) == S::Unsupported);
    M::SlotReuse reuse;
    assert(m.begin_slot_reuse(a.slot_size, 11, &reuse));
    assert(m.mark_endpoint_dead(failed));
    // This stripe has zero durable groups, but its initial writer can still
    // publish data. It must not disappear from the repair obligation.
    assert(m.background_rebuild_view(empty_initial.id.stripe_id, failed, &v) == S::Busy);
    assert(m.mark_dead_group(empty_initial.id));
    assert(m.background_rebuild_view(empty_initial.id.stripe_id, failed, &v) == S::Skip);
    // Healthy pre-failure stripes must remain usable when bounded growth
    // runs out; only stripes containing the failed endpoint stay frozen.
    M::SlotReuse healthy_reuse;
    cfg.ft_background_rebuild = false;
    assert(!m.begin_slot_reuse(other.slot_size, 11, &healthy_reuse));
    cfg.ft_background_rebuild = true;
    assert(m.begin_slot_reuse(other.slot_size, 11, &healthy_reuse));
    assert(healthy_reuse.group.id.stripe_id == other.id.stripe_id);
    assert(m.abort_slot_reuse(healthy_reuse));
    assert(m.background_rebuild_view(a.id.stripe_id, failed, &v) == S::Busy);
    M::SlotReuse rejected;
    assert(!m.begin_slot_reuse(a.slot_size, 11, &rejected));
    // Abort the reservation saved before failure. begin_slot_reuse clears
    // its output even on rejection, so keep the two attempts separate.
    assert(m.abort_slot_reuse(reuse));
    assert(m.background_rebuild_view(b.id.stripe_id, failed, &v) == S::Busy);
    seal(b, 1, false);
    assert(m.background_rebuild_view(b.id.stripe_id, failed, &v) == S::Busy);
    m.mark_slot_group_durable(b.id);
    assert(m.background_rebuild_view(a.id.stripe_id, failed, &v) == S::Ready);
    assert(v.live_groups == 2 && v.live_objects == 3 && v.live_slot_ids.size() == 2);
    // Concurrent foreground release may shrink, but cannot replace, the set.
    assert(m.release_group_object(a.segments[0].addr) != M::kGroupReleaseRejected);
    assert(m.release_group_object(a.segments[1].addr) != M::kGroupReleaseRejected);
    const auto target = heap.allocate_whole_region_on_endpoint(7);
    assert(target != FarLib::allocator::remote::InvalidRemoteAddr);
    auto bad = v; bad.original_base[0]++;
    assert(!m.publish_background_rebuild(bad, target));
    assert(m.publish_background_rebuild(v, target));
    assert(!m.publish_background_rebuild(v, target));
    assert(m.background_rebuild_view(a.id.stripe_id, failed, &v) == S::Skip);
    const auto logical = b.segments[0].addr;
    const auto expected = target + b.slot_offset;
    assert(m.resolve_rebuilt_addr(b.id.stripe_id, 0, logical) == expected);
    assert(m.owns(logical) && !m.owns(target));
    M::SlotGroupHandle unchanged;
    assert(m.get_slot_group_layout(b.id, &unchanged));
    assert(unchanged.segments[0].addr == logical);
    assert(m.background_rebuild_view(parity.id.stripe_id, failed, &v) == S::Ready);
    assert(v.failed_shard >= 4);
    const auto parity_target = heap.allocate_whole_region_on_endpoint(7);
    assert(m.publish_background_rebuild(v, parity_target));
    assert(m.resolve_rebuilt_addr(v.stripe_id, v.failed_shard,
                                  v.original_base[v.failed_shard] + 7) == parity_target + 7);
    assert(m.count_rebuild_remaining(failed) == 0);
    M::SlotGroupHandle fresh;
    assert(m.allocate_slot_group(8 << 10, &fresh));
    bool has_spare = false;
    for (const auto &s : fresh.segments) {
        assert(s.endpoint_idx != failed);
        has_spare |= s.endpoint_idx == 7;
    }
    assert(has_spare);
    m.release_background_targets_for_shutdown();
    m.release_background_targets_for_shutdown(); // idempotent, no double free
    assert(m.resolve_rebuilt_addr(b.id.stripe_id, 0, logical) == logical);
    std::cout << "background rebuild manager lifecycle tests passed\n";
}
