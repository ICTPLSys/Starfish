// CPU/fixture regression for the ec_batch pending FIFO/token race.
//
// The old sender observed EcBatchTokenTable::available(), popped the sealed
// group, and only then acquired a token.  A split-object producer could claim
// the last token in between those operations.  This fixture models that exact
// interleaving and checks the peek/commit ownership contract: a failed token
// acquisition leaves the pending group and its staging lease untouched.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/ec_batch_staging.hpp"
#include "cache/alloc/ec_batch_write.hpp"
#include "cache/alloc/ec_group_builder.hpp"
#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_pending_config;
const rdma::Configure &get_config() { return g_ec_pending_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using FarLib::cache::SmallObjectStripeManager;
using FarLib::cache::ec_batch::EcBatchTokenTable;
using FarLib::cache::ec_batch::EcGroupBuilder;
using FarLib::cache::ec_batch::EcStagingGroupSlot;
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

bool configure_remote_fixture() {
    auto &config = FarLib::g_ec_pending_config;
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

bool complete_and_release(EcBatchTokenTable &tokens, uint64_t token_id) {
    FarLib::cache::ec_batch::EcBatchToken *last = nullptr;
    for (size_t segment = 0;
         segment < FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup;
         ++segment) {
        auto *finished = tokens.complete_segment(token_id,
                                                 static_cast<uint8_t>(segment));
        if (segment + 1 ==
            FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup) {
            last = finished;
        } else if (finished != nullptr) {
            return false;
        }
    }
    return last != nullptr && tokens.release(token_id);
}

int run() {
    Checks checks;
    checks.check(configure_remote_fixture(), "ec_batch fixture configured");

    const auto &config = FarLib::g_ec_pending_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    SmallObjectStripeManager manager;
    manager.init(config.remote_total, shard_size);
    checks.check(manager.enabled(), "stripe manager enabled");

    constexpr size_t kSlotSize = 4096;
    constexpr size_t kObjectBytes = 4080;
    constexpr size_t kPendingDepth = 2;
    std::vector<uint8_t> staging_storage(
        EcStagingPool::required_bytes(kSlotSize, 1), 0);
    EcStagingPool staging;
    checks.check(staging.init(staging_storage.data(), staging_storage.size(),
                              kSlotSize, 1),
                 "staging pool initialized");
    EcGroupBuilder builder(&manager, &staging, kPendingDepth);
    checks.check(builder.valid(), "group builder initialized");

    std::array<std::array<uint8_t, kObjectBytes>,
               FarLib::cache::ec_batch::kEcBatchDataSlots>
        objects{};
    for (size_t i = 0; i < objects.size(); ++i) {
        std::memset(objects[i].data(), static_cast<int>(0x10 + i),
                    objects[i].size());
        const auto status = builder.add_object(objects[i].data(),
                                               objects[i].size());
        checks.equal(static_cast<unsigned>(status),
                     static_cast<unsigned>(
                         FarLib::cache::ec_batch::EcBatchStatus::kOk),
                     "small object staged/sealed");
    }
    checks.equal(builder.pending_count(), size_t{1},
                 "one sealed group is pending");
    checks.equal(staging.in_use(), size_t{1},
                 "pending group retains its staging lease");

    EcGroupBuilder::SealedGroup pending;
    checks.check(builder.peek_pending(&pending),
                 "peek observes the sealed group without removing it");
    const uint64_t sequence = pending.sequence;
    checks.equal(pending.slot_size, static_cast<uint32_t>(kSlotSize),
                 "sealed group keeps the expected size class");

    // This is the stale available() observation from the old sender.  It sees
    // capacity before another producer fills all 64 token entries.
    checks.check(EcBatchTokenTable{}.available() > 0,
                 "available is positive before the competing producer");
    EcBatchTokenTable tokens;
    constexpr size_t kTokenCapacity = EcBatchTokenTable::capacity();
    std::array<uint64_t, kTokenCapacity> token_ids{};
    for (size_t i = 0; i < kTokenCapacity; ++i) {
        FarLib::cache::ec_batch::EcBatchToken *entry = nullptr;
        checks.check(tokens.acquire(pending, &token_ids[i], &entry) &&
                         entry != nullptr,
                     "competing producer acquires token");
    }
    checks.equal(tokens.available(), size_t{0},
                 "competing producer fills the token table");

    // A post attempt now fails only at acquire().  The pending record and the
    // staging lease must still be present; an old pop-before-acquire sender
    // would have made pending_count() zero here and stranded the lease.
    FarLib::cache::ec_batch::EcBatchToken *blocked_entry = nullptr;
    uint64_t blocked_id = 0;
    checks.check(!tokens.acquire(pending, &blocked_id, &blocked_entry),
                 "token acquisition fails under backpressure");
    checks.equal(builder.pending_count(), size_t{1},
                 "backpressure leaves pending group queued");
    checks.equal(staging.in_use(), size_t{1},
                 "backpressure leaves staging lease owned");

    checks.check(complete_and_release(tokens, token_ids[0]),
                 "one competing token completes six segments and releases");
    checks.equal(tokens.available(), size_t{1},
                 "one token becomes available again");

    uint64_t retry_id = 0;
    FarLib::cache::ec_batch::EcBatchToken *retry_entry = nullptr;
    checks.check(tokens.acquire(pending, &retry_id, &retry_entry) &&
                     retry_entry != nullptr,
                 "pending group acquires token after CQE release");
    checks.check(builder.commit_pending(sequence),
                 "successful post commits the pending sequence");
    checks.check(!builder.commit_pending(sequence),
                 "repeated commit of the same sequence is rejected");
    checks.equal(builder.pending_count(), size_t{0},
                 "pending FIFO is empty only after commit");
    checks.equal(staging.in_use(), size_t{1},
                 "committed group still owns staging until final CQE");

    // Complete all tokens, including the retried pending-group token.  The
    // token table must return to its exact initial state.
    checks.check(complete_and_release(tokens, retry_id),
                 "retried pending-group token completes and releases");
    for (size_t i = 1; i < kTokenCapacity; ++i) {
        checks.check(complete_and_release(tokens, token_ids[i]),
                     "competing token completes and releases");
    }
    checks.equal(tokens.in_use(), size_t{0},
                 "all token entries are released");
    checks.equal(tokens.available(), kTokenCapacity,
                 "token table returns to full capacity");

    checks.check(builder.release_sent(pending),
                 "final CQE releases the pending group's staging lease");
    checks.equal(staging.in_use(), size_t{0},
                 "staging pool has no leaked lease");
    checks.check(manager.mark_dead_group(pending.group.id),
                 "fixture marks the sent group dead for cleanup");

    std::printf(
        "EC_BATCH_PENDING_BACKPRESSURE checks=%d failures=%d\n",
        checks.count, checks.failures);
    if (checks.failures == 0) {
        std::puts("EC_BATCH_PENDING_BACKPRESSURE_PASS old-pop-would-drop-queued-group");
    }
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_BATCH_PENDING_BACKPRESSURE_SKIP_NO_VERBS");
    return 0;
#endif
}
