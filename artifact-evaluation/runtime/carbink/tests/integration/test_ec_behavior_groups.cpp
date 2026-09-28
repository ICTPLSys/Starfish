// CPU-only EC builder contract test.
//
// The D2 eviction path supplies a normalized six-class behavior key.  An
// unsealed group must retain that key and must not mix objects from another
// class.  A pending-queue backpressure round must also retain a partially
// staged object exactly once until the post/commit owner succeeds.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/ec_batch_staging.hpp"
#include "cache/alloc/ec_group_builder.hpp"
#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_behavior_config;
const rdma::Configure &get_config() { return g_ec_behavior_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using FarLib::cache::SmallObjectStripeManager;
using FarLib::cache::ec_batch::EcBatchStatus;
using FarLib::cache::ec_batch::EcGroupBuilder;
using FarLib::cache::ec_batch::EcStagingPool;

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
    auto &config = FarLib::g_ec_behavior_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    config.server_count = static_cast<int>(
        FarLib::rdma::Configure::ft_ec_batch_endpoint_count);
    config.server_buffer_size = 4ull * 1024 * 1024;
    config.client_buffer_size = 4ull * 1024 * 1024;
    config.remote_total = config.server_buffer_size *
                          static_cast<size_t>(config.server_count);
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

bool add(EcGroupBuilder &builder, const std::array<uint8_t, 4080> &object,
         uint32_t key, uint64_t *address) {
    return builder.add_object(object.data(), object.size(), address, key) ==
           EcBatchStatus::kOk;
}

int run() {
    Checks checks;
    checks.check(configure_fixture(), "ec_batch behavior fixture configured");

    const auto &config = FarLib::g_ec_behavior_config;
    SmallObjectStripeManager manager;
    manager.init(config.remote_total,
                 FarLib::rdma::Configure::ft_ec_shard_size_bytes);
    checks.check(manager.enabled(), "stripe manager initialized");

    constexpr size_t kSlotSize = 4096;
    constexpr size_t kDepth = 8;
    std::vector<uint8_t> storage(EcStagingPool::required_bytes(kSlotSize,
                                                                 kDepth), 0);
    EcStagingPool staging;
    checks.check(staging.init(storage.data(), storage.size(), kSlotSize, kDepth),
                 "staging pool initialized");
    EcGroupBuilder builder(&manager, &staging, 1);
    checks.check(builder.valid(), "behavior-group builder initialized");

    std::array<std::array<uint8_t, 4080>, 4> objects{};
    for (size_t i = 0; i < objects.size(); ++i) {
        std::memset(objects[i].data(), static_cast<int>(0x20 + i),
                    objects[i].size());
    }
    uint64_t ignored = 0;
    // Interleave two classes. Neither class may enter the other's open group.
    checks.check(add(builder, objects[0], 1, &ignored),
                 "class 1 object 0 staged");
    checks.check(add(builder, objects[1], 4, &ignored),
                 "class 4 object 0 staged");
    checks.check(add(builder, objects[2], 1, &ignored),
                 "class 1 object 1 staged");
    checks.check(add(builder, objects[3], 4, &ignored),
                 "class 4 object 1 staged");
    checks.equal(builder.pending_count(), size_t{0},
                 "interleaved partial classes remain unsealed");
    checks.check(builder.flush() == EcBatchStatus::kPendingQueueFull,
                 "first flush reports pending-depth backpressure");
    checks.equal(builder.pending_count(), size_t{1},
                 "only one class seals before pending queue fills");
    checks.equal(builder.sealed_count_for_behavior_group(1), uint64_t{1},
                 "EC builder records one sealed class-1 group");
    checks.equal(builder.sealed_count_for_behavior_group(4), uint64_t{0},
                 "EC builder has not sealed class-4 partial group yet");

    EcGroupBuilder::SealedGroup first;
    checks.check(builder.peek_pending(&first), "first sealed class is peekable");
    checks.equal(first.behavior_group, uint32_t{1},
                 "sealed record keeps class identity");
    checks.equal(first.live_count, uint8_t{2}, "class 1 retains two objects");
    const uint64_t first_sequence = first.sequence;
    checks.check(builder.commit_pending(first_sequence),
                 "successful post commits class 1 exactly once");
    checks.check(!builder.commit_pending(first_sequence),
                 "duplicate post commit is rejected");
    checks.check(builder.release_sent(first),
                 "class 1 staging lease released after final post");
    checks.check(manager.mark_dead_group(first.group.id),
                 "class 1 remote group retired once");

    // The class-4 open group was retained while the queue was full. Its two
    // objects must now seal once, not be re-added by the retry path.
    checks.check(builder.flush() == EcBatchStatus::kOk,
                 "retained class 4 partial group seals after drain");
    checks.equal(builder.pending_count(), size_t{1},
                 "retained class 4 group appears exactly once");
    EcGroupBuilder::SealedGroup second;
    checks.check(builder.peek_pending(&second), "class 4 group is peekable");
    checks.equal(second.behavior_group, uint32_t{4},
                 "second sealed record keeps class 4 identity");
    checks.equal(second.live_count, uint8_t{2},
                 "class 4 retains exactly its two objects");
    checks.equal(builder.sealed_count_for_behavior_group(4), uint64_t{1},
                 "EC builder records one sealed class-4 group");
    checks.check(builder.commit_pending(second.sequence),
                 "class 4 post commits once");
    checks.check(builder.release_sent(second),
                 "class 4 staging lease released after final post");
    checks.check(manager.mark_dead_group(second.group.id),
                 "class 4 remote group retired once");
    checks.equal(builder.pending_count(), size_t{0},
                 "pending queue empty after both posts");
    checks.equal(staging.in_use(), size_t{0}, "no staging lease leaked");

    std::printf("EC_BEHAVIOR_GROUPS checks=%d failures=%d\n", checks.count,
                checks.failures);
    if (checks.failures == 0) std::puts("EC_BEHAVIOR_GROUPS_PASS");
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_BEHAVIOR_GROUPS_SKIP_NO_VERBS");
    return 0;
#endif
}
