// CPU integration checks for the behavior-group runtime switch and the
// protection-side backup contract.  No cache object, RDMA client, or network
// request is constructed here.  The manager fixture uses production
// SmallObjectStripeManager metadata and a supplied endpoint-liveness oracle.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "design2/behavior_group_runtime.hpp"
#include "design2/simple_region_budget.hpp"
#include "recovery/ec_backup_policy.hpp"

#if __has_include(<infiniband/verbs.h>)
#include "rdma/config.hpp"
#include "cache/alloc/small_object_stripe.hpp"
#include "design2/fixed_six_runtime.hpp"
#include "design2/simple_region_heat.hpp"
#include "utils/control.hpp"
#endif

#if __has_include(<infiniband/verbs.h>)
namespace FarLib {
rdma::Configure g_behavior_group_runtime_config;
const rdma::Configure &get_config() { return g_behavior_group_runtime_config; }
}  // namespace FarLib
#endif

#if __has_include(<infiniband/verbs.h>)
namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote
#endif

namespace {

using FarLib::cache::ec_backup_policy::backup_endpoint_is_dead;
using FarLib::cache::ec_backup_policy::can_reuse_clean_backup;

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

#if __has_include(<infiniband/verbs.h>)

void test_runtime_switch(Checks &checks) {
    auto &config = FarLib::g_behavior_group_runtime_config;
    config.behavior_group = false;
    FarLib::behavior_group_runtime::configure(false);
    checks.check(!FarLib::behavior_group_runtime::enabled(),
                 "behavior-group runtime defaults disabled");

    // This test is intentionally an env-free control.  If a caller injects a
    // legacy override, fail rather than silently relabeling the result.
    const bool env_free =
        std::getenv("FARLIB_SIMPLE_SIX_GROUPS") == nullptr &&
        std::getenv("FARLIB_SIMPLE_REGION_BUDGET") == nullptr &&
        std::getenv("FARLIB_SIMPLE_LOCAL_RESIDENT") == nullptr &&
        std::getenv("FARLIB_SIMPLE_REGION_HEAT") == nullptr &&
        std::getenv("FARLIB_SIMPLE_HOTCOLD_RELINK") == nullptr &&
        std::getenv("FARLIB_SIMPLE_LOCAL_ROUTING") == nullptr &&
        std::getenv("FARLIB_SIMPLE_REMOTE_HOTCOLD") == nullptr &&
        std::getenv("FARLIB_LIST_ONLY_SIX") == nullptr &&
        std::getenv("FARLIB_LIST_ONLY_SIX_GROUPS") == nullptr &&
        std::getenv("FARLIB_FIXED_SIX_GROUPS") == nullptr;
    checks.check(env_free, "semantic-six control has no environment override");
    if (env_free) {
        checks.check(!FarLib::simple_region_budget::six_enabled(),
                     "semantic-six remains two-way when the flag is off");
        checks.equal(FarLib::simple_region_budget::classes(), size_t{2},
                     "default-off budget exposes two legacy classes");
        checks.check(!FarLib::simple_region_budget::enabled(),
                     "default-off budget controller is disabled");
        checks.check(!FarLib::simple_region_heat::enabled(),
                     "default-off region heat is disabled");
        checks.check(!FarLib::simple_region_heat::grouping_enabled(),
                     "default-off heat grouping is disabled");
        checks.check(!FarLib::simple_region_heat::relink_enabled(),
                     "default-off heat relinking is disabled");
        checks.check(!FarLib::simple_region_heat::local_routing_enabled(),
                     "default-off local routing is disabled");
        checks.check(!FarLib::simple_region_heat::remote_grouping_enabled(),
                     "default-off remote grouping is disabled");
        checks.check(!FarLib::allocator::six_group::list_only_six::enabled(),
                     "default-off list-only six mode is disabled");
    }

    config.behavior_group = true;
    FarLib::behavior_group_runtime::configure(config.behavior_group);
    checks.check(FarLib::behavior_group_runtime::enabled(),
                 "one behavior-group flag enables runtime mode");
    checks.check(config.is_ec_batch_mode(),
                 "the same flag selects the EC batch family");
    checks.check(FarLib::simple_region_budget::six_enabled(),
                 "behavior-group mode enables semantic six groups");
    checks.equal(FarLib::simple_region_budget::classes(), size_t{6},
                 "behavior-group mode exposes six semantic classes");
    checks.check(FarLib::simple_region_budget::enabled(),
                 "behavior-group mode enables adaptive budget");
    checks.check(FarLib::simple_region_budget::mode() ==
                     FarLib::simple_region_budget::Mode::Adaptive,
                 "behavior-group mode selects adaptive budget");
    checks.check(FarLib::simple_region_budget::local_resident_enabled(),
                 "behavior-group mode enables local-resident semantics");
    checks.check(FarLib::simple_region_heat::enabled(),
                 "behavior-group mode enables region heat");
    checks.check(FarLib::simple_region_heat::grouping_enabled(),
                 "behavior-group mode enables heat grouping");
    checks.check(FarLib::simple_region_heat::relink_enabled(),
                 "behavior-group mode enables heat relinking");
    checks.check(FarLib::simple_region_heat::local_routing_enabled(),
                 "behavior-group mode enables local routing");
    checks.check(FarLib::simple_region_heat::remote_grouping_enabled(),
                 "behavior-group mode enables remote grouping");
    checks.check(FarLib::allocator::six_group::list_only_six::enabled(),
                 "behavior-group mode enables six-list compatibility");

    config.behavior_group = false;
    FarLib::behavior_group_runtime::configure(false);
}

void test_size_boundary(Checks &checks) {
    auto &config = FarLib::g_behavior_group_runtime_config;
    config.behavior_group = false;
    config.ft_method_type = FarLib::rdma::Configure::FT_EC_BATCH;
    config.ft_method = "ec_batch";
    config.ft_small_object_cutoff = 4096;
    checks.check(config.ft_small_object(4095),
                 "legacy default cutoff keeps 4095-byte object small");
    checks.check(!config.ft_small_object(4096),
                 "legacy default cutoff remains strict at 4096");
    checks.check(!config.ec_object_uses_split(4095),
                 "legacy default cutoff keeps 4095-byte object whole");
    checks.check(config.ec_object_uses_split(4096),
                 "legacy default cutoff splits 4096-byte object");

    config.behavior_group = true;
    config.ft_small_object_cutoff = 8192;  // Must not move the fixed boundary.
    checks.check(config.ft_small_object(4095),
                 "behavior-group 4095-byte object is whole-capable");
    checks.check(config.ft_small_object(4096),
                 "behavior-group 4096-byte object is whole-capable");
    checks.check(!config.ec_object_uses_split(4095),
                 "4095-byte behavior-group object is not split");
    checks.check(!config.ec_object_uses_split(4096),
                 "4096-byte behavior-group object is not split");
    checks.check(config.ec_object_uses_split(4097),
                 "4097-byte behavior-group object uses four-way split");

    // Legacy mode retains its configured small-object semantics, but the
    // fixed 4 KiB staging ceiling still prevents 4097 from entering it.
    config.behavior_group = false;
    // A permissive legacy cutoff must still not route >4 KiB through the
    // fixed-size whole-object staging path.
    checks.check(!config.ec_object_uses_split(4095),
                 "legacy 4095-byte object remains whole");
    checks.check(!config.ec_object_uses_split(4096),
                 "legacy 4096-byte object remains whole");
    checks.check(config.ec_object_uses_split(4097),
                 "legacy 4097-byte object avoids fixed staging");
    config.ft_method_type = FarLib::rdma::Configure::FT_NONE;
    config.ft_method = "none";
}

#endif  // __has_include(<infiniband/verbs.h>)

void test_clean_backup_guard(Checks &checks) {
    checks.check(can_reuse_clean_backup(true, false, true, true),
                 "clean valid reserved backup is reusable");
    checks.check(!can_reuse_clean_backup(true, false, true, false),
                 "clean valid unreserved backup requires protection");
    checks.check(!can_reuse_clean_backup(true, true, true, true),
                 "dirty backup cannot use clean reuse");
    checks.check(!can_reuse_clean_backup(true, false, false, true),
                 "invalid remote cannot use clean reuse");
    checks.check(!can_reuse_clean_backup(false, false, true, true),
                 "disabled backup policy cannot use clean reuse");
}

#if __has_include(<infiniband/verbs.h>)

bool configure_manager_fixture() {
    auto &config = FarLib::g_behavior_group_runtime_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    config.behavior_group = true;
    config.server_count = static_cast<int>(
        FarLib::rdma::Configure::ft_ec_batch_endpoint_count);
    config.server_buffer_size = 4ull * 1024 * 1024;
    config.client_buffer_size = 4ull * 1024 * 1024;
    config.remote_total = config.server_buffer_size *
                          static_cast<size_t>(config.server_count);
    config.mapping_type = FarLib::rdma::Configure::MAPPING_RANGE;
    config.ft_method_type = FarLib::rdma::Configure::FT_NONE;
    config.ft_method = "none";
    config.exclusive_cache = true;
    config.ft_ec_data_shards = 4;
    config.ft_ec_parity_shards = 2;
    config.ft_small_object_cutoff = 4096;
    config.ft_small_stripe_shard_size_bytes = shard_size;
    ::FarLib::allocator::remote::remote_global_heap.register_remote(
        config.remote_total);
    return config.is_ec_batch_mode();
}

void test_group_endpoint_contract(Checks &checks) {
    using Manager = FarLib::cache::SmallObjectStripeManager;
    checks.check(configure_manager_fixture(),
                 "behavior-group manager fixture selects EC mode");

    const auto &config = FarLib::g_behavior_group_runtime_config;
    Manager manager;
    manager.init(config.remote_total,
                 FarLib::rdma::Configure::ft_ec_shard_size_bytes);
    checks.check(manager.enabled(), "production stripe manager initialized");

    constexpr size_t kWholeBytes = 4096;
    constexpr size_t kSplitBytes = 64 * 1024;
    Manager::SlotGroupHandle whole;
    Manager::SlotGroupHandle split;
    checks.check(manager.allocate_slot_group(kWholeBytes, &whole),
                 "allocate exact-4KiB whole group");
    checks.check(manager.seal_slot_group(whole.id, 0x01),
                 "seal exact-4KiB ordinary group");
    checks.check(manager.allocate_slot_group(kSplitBytes, &split),
                 "allocate >4KiB split group");
    checks.check(manager.seal_split_slot_group(split.id,
                                               static_cast<uint32_t>(kSplitBytes)),
                 "seal >4KiB split group");

    const auto no_dead = [](uint64_t) { return false; };
    checks.check(!backup_endpoint_is_dead(manager, split.segments[0].addr,
                                          true, no_dead),
                 "healthy split group remains reusable");

    const auto data_non_anchor_dead = [&](uint64_t address) {
        return address == split.segments[1].addr;
    };
    checks.check(backup_endpoint_is_dead(manager, split.segments[0].addr,
                                         true, data_non_anchor_dead),
                 "dead non-anchor data endpoint invalidates split backup");

    const auto parity_dead = [&](uint64_t address) {
        return address == split.segments[4].addr;
    };
    checks.check(backup_endpoint_is_dead(manager, split.segments[0].addr,
                                         true, parity_dead),
                 "dead parity endpoint invalidates split backup");

    const auto own_whole_dead = [&](uint64_t address) {
        return address == whole.segments[0].addr;
    };
    const auto parity_only_dead = [&](uint64_t address) {
        return address == whole.segments[4].addr;
    };
    checks.check(backup_endpoint_is_dead(manager, whole.segments[0].addr,
                                         false, own_whole_dead),
                 "dead own endpoint invalidates whole backup");
    checks.check(!backup_endpoint_is_dead(manager, whole.segments[0].addr,
                                          false, parity_only_dead),
                 "whole backup ignores unrelated parity endpoint");

    checks.check(manager.release_group_object(split.segments[0].addr) ==
                     Manager::GroupReleaseResult::kGroupReleaseGroupReleased,
                 "release split group anchor after liveness checks");
    checks.check(manager.release_group_object(whole.segments[0].addr) ==
                     Manager::GroupReleaseResult::kGroupReleaseGroupReleased,
                 "release ordinary whole group after liveness checks");
}

#endif  // __has_include(<infiniband/verbs.h>)

}  // namespace

int main() {
    Checks checks;
#if __has_include(<infiniband/verbs.h>)
    test_runtime_switch(checks);
    test_size_boundary(checks);
#else
    std::puts("BEHAVIOR_GROUP_RUNTIME_SKIP_NO_VERBS");
#endif
    test_clean_backup_guard(checks);
#if __has_include(<infiniband/verbs.h>)
    test_group_endpoint_contract(checks);
#endif
    std::printf("BEHAVIOR_GROUP_RUNTIME checks=%d failures=%d\n", checks.count,
                checks.failures);
    if (checks.failures == 0) std::puts("BEHAVIOR_GROUP_RUNTIME_PASS");
    return checks.failures == 0 ? 0 : 1;
}
