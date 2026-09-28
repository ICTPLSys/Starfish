// CPU-only contracts for worker-private EC builders and sharded group tokens.
//
// The test deliberately stops before RDMA: it checks the ownership boundary
// that the runtime's worker-private eviction path relies on, then exercises
// the shared token table's completion/generation contract.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/ec_batch_staging.hpp"
#include "cache/alloc/ec_batch_write.hpp"
#include "cache/alloc/ec_group_builder.hpp"
#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_worker_private_config;
const rdma::Configure &get_config() { return g_ec_worker_private_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using FarLib::cache::SmallObjectStripeManager;
using FarLib::cache::ec_batch::EcBatchStatus;
using FarLib::cache::ec_batch::EcBatchTokenTable;
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
    auto &config = FarLib::g_ec_worker_private_config;
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

template <size_t Bytes>
bool add(EcGroupBuilder &builder, const std::array<uint8_t, Bytes> &object,
         uint32_t behavior_group, uint64_t *address) {
    return builder.add_object(object.data(), object.size(), address,
                              behavior_group) == EcBatchStatus::kOk;
}

template <size_t Bytes>
void fill(std::array<std::array<uint8_t, Bytes>, 4> &objects,
          uint8_t base) {
    for (size_t i = 0; i < objects.size(); ++i) {
        std::memset(objects[i].data(), static_cast<int>(base + i),
                    objects[i].size());
    }
}

bool cleanup_group(SmallObjectStripeManager &manager, EcGroupBuilder &builder,
                   const EcGroupBuilder::SealedGroup &record) {
    return builder.commit_pending(record.sequence) &&
           builder.release_sent(record) &&
           manager.mark_dead_group(record.group.id);
}

int run() {
    Checks checks;
    checks.check(configure_fixture(), "ec_batch worker-private fixture configured");

    const auto &config = FarLib::g_ec_worker_private_config;
    SmallObjectStripeManager manager;
    manager.init(config.remote_total,
                 FarLib::rdma::Configure::ft_ec_shard_size_bytes);
    checks.check(manager.enabled(), "stripe manager initialized");

    constexpr size_t kSlotSize = 4096;
    constexpr size_t kObjectBytes = 4080;
    constexpr size_t kDepth = 8;
    std::vector<uint8_t> storage0(
        EcStagingPool::required_bytes(kSlotSize, kDepth), 0);
    std::vector<uint8_t> storage1(
        EcStagingPool::required_bytes(kSlotSize, kDepth), 0);
    EcStagingPool staging0;
    EcStagingPool staging1;
    checks.check(staging0.init(storage0.data(), storage0.size(), kSlotSize,
                               kDepth),
                 "worker 0 staging pool initialized");
    checks.check(staging1.init(storage1.data(), storage1.size(), kSlotSize,
                               kDepth),
                 "worker 1 staging pool initialized");

    EcGroupBuilder worker0(&manager, &staging0, 256, true);
    EcGroupBuilder worker1(&manager, &staging1, 256, true);
    checks.check(worker0.valid() && worker1.valid(),
                 "worker-private builders initialized");

    std::array<std::array<uint8_t, kObjectBytes>, 4> objects0{};
    std::array<std::array<uint8_t, kObjectBytes>, 4> objects1{};
    fill(objects0, 0x20);
    fill(objects1, 0x60);
    std::array<uint64_t, 4> addresses0{};
    std::array<uint64_t, 4> addresses1{};
    constexpr uint64_t invalid_addr = FarLib::allocator::remote::InvalidRemoteAddr;
    addresses0.fill(invalid_addr);
    addresses1.fill(invalid_addr);
    for (size_t i = 0; i < 4; ++i) {
        checks.check(add(worker0, objects0[i], 1, &addresses0[i]),
                     "worker 0 object staged");
        checks.check(add(worker1, objects1[i], 1, &addresses1[i]),
                     "worker 1 object staged");
    }

    EcGroupBuilder::SealedGroup record0;
    EcGroupBuilder::SealedGroup record1;
    checks.check(worker0.peek_pending(&record0),
                 "worker 0 sealed group is pending");
    checks.check(worker1.peek_pending(&record1),
                 "worker 1 sealed group is pending");
    checks.equal(record0.live_count, uint8_t{4},
                 "worker 0 group has four live objects");
    checks.equal(record1.live_count, uint8_t{4},
                 "worker 1 group has four live objects");
    checks.check(record0.group.id.stripe_id != record1.group.id.stripe_id ||
                     record0.group.id.slot_id != record1.group.id.slot_id,
                 "worker-private builders allocate distinct groups");
    for (size_t i = 0; i < 4; ++i) {
        checks.check(record0.objects[i] == objects0[i].data() &&
                         record1.objects[i] == objects1[i].data(),
                     "private groups do not mix worker object pointers");
        // The address returned while staging is the address retained by the
        // sealed record before a sender can consume it.
        checks.check(addresses0[i] == record0.group.segments[i].addr &&
                         addresses1[i] == record1.group.segments[i].addr &&
                         addresses0[i] != invalid_addr && addresses1[i] != invalid_addr,
                     "data address is published before simulated send");
    }
    checks.check(cleanup_group(manager, worker0, record0),
                 "worker 0 group released after simulated final CQE");
    checks.check(cleanup_group(manager, worker1, record1),
                 "worker 1 group released after simulated final CQE");
    checks.equal(staging0.in_use(), size_t{0}, "worker 0 staging lease released");
    checks.equal(staging1.in_use(), size_t{0}, "worker 1 staging lease released");

    // The fixed descriptor ring must remain reusable across multiple wraps;
    // source/sequence identity stays stable until the producer commits it.
    const EcGroupBuilder::SealedGroup *first_front = nullptr;
    uint64_t previous_sequence = record0.sequence;
    for (size_t round = 0; round < 600; ++round) {
        for (size_t i = 0; i < 4; ++i) {
            if (!add(worker0, objects0[i], 1, &addresses0[i])) {
                checks.check(false, "reused fixed ring accepts four objects");
                return 1;
            }
        }
        const auto *front = worker0.peek_private_pending();
        if (front == nullptr || front->sequence <= previous_sequence ||
            front->live_count != 4 || !staging0.owns_slot(front->staging)) {
            checks.check(false, "reused fixed ring preserves descriptor ownership");
            return 1;
        }
        if (round == 0) first_front = front;
        if (round == 256)
            checks.check(front == first_front, "fixed ring reuses the same descriptor after wrap");
        auto damaged = front->staging;
        damaged.data[3] = static_cast<uint8_t *>(damaged.data[3]) + 1;
        if (staging0.owns_slot(damaged)) {
            checks.check(false, "one lease validation rejects a changed data source");
            return 1;
        }
        damaged = front->staging;
        damaged.parity[1] = static_cast<uint8_t *>(damaged.parity[1]) + 1;
        if (staging0.owns_slot(damaged)) {
            checks.check(false, "one lease validation rejects a changed parity source");
            return 1;
        }
        const auto copy = *front;
        previous_sequence = copy.sequence;
        if (!cleanup_group(manager, worker0, copy) ||
            worker0.peek_private_pending() != nullptr) {
            checks.check(false, "fixed ring clears committed descriptor exactly once");
            return 1;
        }
    }
    checks.equal(staging0.in_use(), size_t{0}, "600 reused batches return every lease");

    // A tail batch with fewer than four objects must become one protected
    // partial group rather than being dropped at worker shutdown.
    std::array<std::array<uint8_t, kObjectBytes>, 4> tail_objects{};
    fill(tail_objects, 0x90);
    EcGroupBuilder tail(&manager, &staging0, 256, true);
    uint64_t ignored = 0;
    for (size_t i = 0; i < 3; ++i) {
        checks.check(add(tail, tail_objects[i], 2, &ignored),
                     "tail object staged");
    }
    checks.check(tail.flush() == EcBatchStatus::kOk,
                 "tail flush seals an underfull group");
    EcGroupBuilder::SealedGroup tail_record;
    checks.check(tail.peek_pending(&tail_record),
                 "tail partial group is pending");
    checks.equal(tail_record.live_count, uint8_t{3},
                 "tail group retains three live objects");
    checks.equal(tail_record.live_mask, uint8_t{0x7},
                 "tail group records the hole mask");
    checks.check(cleanup_group(manager, tail, tail_record),
                 "tail partial group released after simulated post");

    // Explicit teardown must hand both sealed pending groups and still-open
    // groups to the owner before the builder is destroyed.
    std::array<std::array<uint8_t, kObjectBytes>, 4> abandon_objects{};
    fill(abandon_objects, 0xc0);
    EcGroupBuilder abandoned(&manager, &staging0, 256, true);
    checks.check(add(abandoned, abandon_objects[0], 3, &ignored),
                 "abandon open object 0 staged");
    checks.check(add(abandoned, abandon_objects[1], 3, &ignored),
                 "abandon open object 1 staged");
    for (size_t i = 0; i < 4; ++i) {
        checks.check(add(abandoned, abandon_objects[i], 4, &ignored),
                     "abandon sealed object staged");
    }
    size_t abandoned_groups = 0;
    size_t abandoned_objects = 0;
    const size_t dropped = abandoned.abandon_unposted(
        [&](const EcGroupBuilder::SealedGroup &record) {
            ++abandoned_groups;
            abandoned_objects += record.live_count;
            checks.check(manager.mark_dead_group(record.group.id),
                         "abandoned group retired exactly once");
        });
    checks.equal(dropped, size_t{2},
                 "teardown returns pending and open groups");
    checks.equal(abandoned_groups, size_t{2},
                 "teardown callback observes both groups");
    checks.equal(abandoned_objects, size_t{6},
                 "teardown callback observes every staged object");
    checks.equal(staging0.in_use(), size_t{0},
                 "builder teardown releases all staging leases");

    // The token table is shared by completion threads but ownership is local
    // at acquisition.  Prefix/duplicate CQEs must not release the group.
    auto token_storage = std::make_unique<EcBatchTokenTable>();
    auto &tokens = *token_storage;
    uint64_t token_id = 0;
    FarLib::cache::ec_batch::EcBatchToken *token = nullptr;
    checks.check(tokens.acquire(record0, &token_id, &token, 7) &&
                     token != nullptr,
                 "owner 7 acquires a token");
    for (uint8_t segment = 0; segment < 5; ++segment) {
        checks.check(tokens.complete_segment(token_id, segment) == nullptr,
                     "prefix CQE does not release the group");
    }
    checks.check(tokens.complete_segment(token_id, 2) == nullptr,
                 "duplicate CQE is ignored");
    auto *last = tokens.complete_segment(token_id, 5);
    checks.check(last == token, "sixth CQE exclusively returns the token");
    checks.equal(last->pending, uint8_t{0},
                 "final token snapshot has no pending segments");
    checks.equal(last->acked, uint8_t{0x3f},
                 "final token snapshot records all six segments");
    checks.check(tokens.release(token_id), "completed token is released");
    checks.check(tokens.complete_segment(token_id, 5) == nullptr,
                 "stale CQE after release is ignored");
    checks.equal(tokens.in_use(7), size_t{0}, "owner 7 has no live token");

    // Completion threads may reap the six endpoint CQEs concurrently.  One
    // duplicate is deliberately injected by worker 0; only the CAS that
    // consumes the last distinct segment may return the live token.
    uint64_t concurrent_id = 0;
    FarLib::cache::ec_batch::EcBatchToken *concurrent_token = nullptr;
    checks.check(tokens.acquire(record0, &concurrent_id, &concurrent_token, 7) &&
                     concurrent_token != nullptr,
                 "owner 7 acquires the concurrent-completion token");
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<size_t> final_winners{0};
    std::atomic<size_t> wrong_winners{0};
    std::atomic<size_t> duplicate_ignored{0};
    std::array<std::thread, FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup>
        completers;
    for (size_t i = 0; i < completers.size(); ++i) {
        completers[i] = std::thread([&, i] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto *winner = tokens.complete_segment(
                concurrent_id, static_cast<uint8_t>(i));
            if (winner == concurrent_token) {
                final_winners.fetch_add(1, std::memory_order_relaxed);
            } else if (winner != nullptr) {
                wrong_winners.fetch_add(1, std::memory_order_relaxed);
            }
            if (i == 0 &&
                tokens.complete_segment(concurrent_id, 0) == nullptr) {
                duplicate_ignored.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != completers.size()) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto &completer : completers) completer.join();
    checks.equal(final_winners.load(std::memory_order_relaxed), size_t{1},
                 "exactly one concurrent CQE wins the final completion");
    checks.equal(wrong_winners.load(std::memory_order_relaxed), size_t{0},
                 "no non-final CQE returns the token");
    checks.equal(duplicate_ignored.load(std::memory_order_relaxed), size_t{1},
                 "duplicate concurrent CQE is ignored");
    checks.equal(concurrent_token->acked, uint8_t{0x3f},
                 "concurrent final snapshot records all segments");
    checks.check(tokens.release(concurrent_id),
                 "concurrent-completed token is released");

    // Reuse the same owner/slot and inject a CQE from the old generation.
    // It must not alter the newly acquired token's completion state.
    uint64_t reused_id = 0;
    FarLib::cache::ec_batch::EcBatchToken *reused_token = nullptr;
    checks.check(tokens.acquire(record1, &reused_id, &reused_token, 7) &&
                     reused_token != nullptr,
                 "owner 7 reuses the released token slot");
    checks.check(reused_token == concurrent_token && reused_id != concurrent_id,
                 "slot is reused with a new generation");
    checks.check(tokens.complete_segment(concurrent_id, 0) == nullptr,
                 "old-generation CQE is ignored after slot reuse");
    checks.equal(reused_token->pending, uint8_t{6},
                 "stale CQE cannot change new token pending count");
    checks.equal(reused_token->acked, uint8_t{0},
                 "stale CQE cannot change new token ack bitmap");
    for (uint8_t segment = 0; segment < 5; ++segment) {
        checks.check(tokens.complete_segment(reused_id, segment) == nullptr,
                     "new-generation prefix CQE remains non-final");
    }
    auto *reused_last = tokens.complete_segment(reused_id, 5);
    checks.check(reused_last == reused_token,
                 "new generation still has one final completion");
    checks.check(tokens.release(reused_id),
                 "new-generation token is released");

    uint64_t abandoned_ids[2]{};
    for (size_t i = 0; i < 2; ++i) {
        FarLib::cache::ec_batch::EcBatchToken *entry = nullptr;
        checks.check(tokens.acquire(record1, &abandoned_ids[i], &entry, 9 + i),
                     "owner-partitioned token acquired");
    }
    size_t abandoned_tokens = 0;
    checks.equal(tokens.abandon_all_in_use(
                     [&](uint64_t id, const FarLib::cache::ec_batch::EcBatchToken &entry) {
                         ++abandoned_tokens;
                         checks.check(id != 0 && entry.record.group.id.valid(),
                                      "abandon callback receives a live record");
                     }),
                 size_t{2}, "token teardown abandons every live token");
    checks.equal(abandoned_tokens, size_t{2},
                 "token teardown callback runs once per token");
    checks.equal(tokens.in_use(), size_t{0},
                 "token table has no inflight groups after teardown");

    std::printf("EC_BATCH_WORKER_PRIVATE checks=%d failures=%d\n",
                checks.count, checks.failures);
    if (checks.failures == 0) {
        std::puts("EC_BATCH_WORKER_PRIVATE_PASS");
    }
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_BATCH_WORKER_PRIVATE_SKIP_NO_VERBS");
    return 0;
#endif
}
