// CPU-only contract tests for the direct four-data/two-parity EC path.
//
// These tests deliberately use host-allocated registered-bank memory.  They
// exercise the ownership and completion protocol without posting RDMA work.
// In particular, the Record must retain the caller's four source pointers;
// only the two parity shards and the zero page belong to the persistent bank.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/ec_direct_group_builder.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_direct_group_config;
const rdma::Configure &get_config() { return g_ec_direct_group_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using FarLib::cache::SmallObjectStripeManager;
using FarLib::cache::ec_batch::EcBatchStatus;
using FarLib::cache::ec_batch::EcDirectGroupBuilder;
using FarLib::cache::ec_batch::EcDirectWriteBank;

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

bool host_bank_allocate(void *, size_t bytes,
                        EcDirectWriteBank::RegisteredBank *out) {
    if (out == nullptr || bytes == 0) return false;
    const size_t aligned = (bytes + 4095u) & ~size_t{4095u};
    void *base = std::aligned_alloc(4096, aligned);
    if (base == nullptr) return false;
    // The tail of every bank is the direct encoder's zero-pad source.  Do not
    // let an allocator's uninitialized bytes make short/hole parity tests
    // depend on heap history.
    std::memset(base, 0, aligned);
    out->base = base;
    out->bytes = aligned;
    // The CPU test does not post verbs.  A non-zero lkey still exercises the
    // fixed-bank binding checks performed by publish().
    out->lkey = 0xecd1u;
    out->registration = nullptr;
    return true;
}

void host_bank_free(void *, EcDirectWriteBank::RegisteredBank &bank) {
    std::free(bank.base);
    bank = EcDirectWriteBank::RegisteredBank{};
}

bool configure_fixture() {
    auto &config = FarLib::g_ec_direct_group_config;
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

bool parity_matches(const EcDirectWriteBank::Entry &entry) {
    if (entry.record.slot_size == 0 || entry.record.slot_size > 4096) {
        return false;
    }

    // This oracle uses the existing fixed-shard codec over local zero-filled
    // copies.  The production direct encoder is variable-length and may read
    // no copy at all; comparing against this independent fixed-shard call
    // catches parity errors for short objects and trailing holes.
    std::array<std::array<uint8_t, 4096>, 4> padded{};
    const void *data[4]{};
    for (size_t i = 0; i < 4; ++i) {
        const size_t size = entry.record.object_sizes[i];
        if (size > entry.record.slot_size) return false;
        if (size != 0 && entry.record.objects[i] == nullptr) return false;
        if (size != 0) {
            std::memcpy(padded[i].data(), entry.record.objects[i], size);
        }
        data[i] = padded[i].data();
    }

    std::array<std::array<uint8_t, 4096>, 2> expected{};
    void *parity[2] = {expected[0].data(), expected[1].data()};
    if (!FarLib::cache::small_object_stripe_encode_shards(
            data, parity, entry.record.slot_size, 0)) {
        return false;
    }
    for (size_t i = 0; i < 2; ++i) {
        if (entry.record.parity[i] == nullptr) return false;
        if (std::memcmp(entry.record.parity[i], expected[i].data(),
                        entry.record.slot_size) != 0) {
            return false;
        }
    }
    return true;
}

bool recovery_matches(const EcDirectWriteBank::Entry &entry) {
    if (entry.record.slot_size == 0 || entry.record.slot_size > 4096) {
        return false;
    }

    std::array<std::array<uint8_t, 4096>, 6> original{};
    for (size_t i = 0; i < 4; ++i) {
        const size_t size = entry.record.object_sizes[i];
        if (size > entry.record.slot_size) return false;
        if (size != 0 && entry.record.objects[i] == nullptr) return false;
        if (size != 0) {
            std::memcpy(original[i].data(), entry.record.objects[i], size);
        }
    }
    for (size_t i = 0; i < 2; ++i) {
        if (entry.record.parity[i] == nullptr) return false;
        std::memcpy(original[4 + i].data(), entry.record.parity[i],
                    entry.record.slot_size);
    }

    auto rebuild_case = [&](uint8_t alive_mask,
                            uint8_t missing_a,
                            uint8_t missing_b) {
        auto work = original;
        work[missing_a].fill(0);
        if (missing_b < 6) work[missing_b].fill(0);
        void *shards[6] = {work[0].data(), work[1].data(), work[2].data(),
                           work[3].data(), work[4].data(), work[5].data()};
        if (!FarLib::cache::small_object_stripe_rebuild_shards(
                alive_mask, shards, entry.record.slot_size, 0)) {
            return false;
        }
        if (std::memcmp(work[missing_a].data(), original[missing_a].data(),
                        entry.record.slot_size) != 0) {
            return false;
        }
        if (missing_b < 6 &&
            std::memcmp(work[missing_b].data(), original[missing_b].data(),
                        entry.record.slot_size) != 0) {
            return false;
        }
        return true;
    };

    // One missing data shard, then one data plus one parity shard.  The
    // second case exercises the same RS recovery oracle used by the runtime's
    // two-failure path, rather than only checking that parity was encoded.
    return rebuild_case(static_cast<uint8_t>(0x3f & ~uint8_t{1u << 1}), 1,
                        6) &&
           rebuild_case(static_cast<uint8_t>(0x3f &
                                              ~(uint8_t{1u << 0} |
                                                uint8_t{1u << 4})),
                        0, 4);
}

bool finish_group(EcDirectWriteBank &bank, SmallObjectStripeManager &manager,
                  uint64_t token, Checks &checks, const char *label) {
    auto *entry = bank.get(token);
    if (entry == nullptr) {
        checks.check(false, label);
        return false;
    }
    const auto group_id = entry->record.group.id;
    bool ok = true;
    for (uint8_t segment = 0; segment < 5; ++segment) {
        if (bank.complete_segment(token, segment, true) != nullptr) {
            ok = false;
        }
    }
    // A direct slot is not reusable merely because five of six CQEs arrived.
    ok = bank.release(token) == false && ok;
    auto *terminal = bank.complete_segment(token, 5, true);
    ok = terminal != nullptr && terminal->recoverable() && ok;
    ok = bank.complete_segment(token, 5, true) == nullptr && ok;
    ok = bank.release(token) && ok;
    ok = bank.get(token) == nullptr && ok;
    ok = manager.mark_dead_group(group_id) && ok;
    checks.check(ok, label);
    return ok;
}

void add_four_4072(EcDirectGroupBuilder &builder,
                   std::array<std::array<uint8_t, 4072>, 4> &objects,
                   size_t key, Checks &checks, const char *prefix) {
    for (size_t i = 0; i < objects.size(); ++i) {
        std::memset(objects[i].data(), static_cast<int>(0x31 + i),
                    objects[i].size());
        uint64_t address = 0;
        const auto status = builder.add_object(objects[i].data(),
                                               objects[i].size(), &address,
                                               key);
        const bool expected_ok = status == EcBatchStatus::kOk;
        checks.check(expected_ok, prefix);
        checks.check(address !=
                         FarLib::allocator::remote::InvalidRemoteAddr,
                     "direct group returns a valid remote data address");
    }
}

void test_four_object_group(SmallObjectStripeManager &manager, Checks &checks) {
    EcDirectWriteBank bank(1);
    checks.check(bank.init(nullptr, host_bank_allocate, host_bank_free),
                 "direct bank initializes one host owner");
    EcDirectGroupBuilder builder(&manager, &bank, 0);
    std::array<std::array<uint8_t, 4072>, 4> objects{};
    add_four_4072(builder, objects, 0, checks,
                  "four-object 4072-byte source accepted");
    checks.equal(builder.pending_count(), size_t{1},
                 "fourth object seals one direct group");
    checks.equal(builder.object_count(), uint64_t{4},
                 "direct builder counts four source objects");
    checks.equal(builder.data_bytes(), uint64_t{4 * 4072},
                 "direct builder counts exact source payload bytes");

    const uint64_t token = builder.front_token();
    auto *entry = bank.get(token);
    checks.check(entry != nullptr, "sealed direct group is visible to completion");
    if (entry == nullptr) return;
    for (size_t i = 0; i < objects.size(); ++i) {
        checks.check(entry->record.objects[i] == objects[i].data(),
                     "record retains the original object pointer");
        checks.equal(entry->record.object_sizes[i], uint32_t{4072},
                     "record retains the 4072-byte object length");
    }
    checks.check(parity_matches(*entry),
                 "four nonzero sources match the fixed-shard parity oracle");
    checks.check(recovery_matches(*entry),
                 "four nonzero sources pass one/two-shard recovery oracle");
    builder.commit_pending();
    checks.equal(builder.pending_count(), size_t{0},
                 "direct group leaves the builder after post handoff");
    finish_group(bank, manager, token, checks,
                 "six terminal completions are required before reuse");
}

void test_partial_and_interleaved(SmallObjectStripeManager &manager,
                                  Checks &checks) {
    EcDirectWriteBank bank(1);
    checks.check(bank.init(nullptr, host_bank_allocate, host_bank_free),
                 "partial-group bank initializes");
    EcDirectGroupBuilder builder(&manager, &bank, 0);

    std::array<std::array<uint8_t, 4072>, 4> full{};
    add_four_4072(builder, full, 0, checks,
                  "key 0 full group accepts each source");

    std::array<uint8_t, 4096> unequal_a{};
    std::array<uint8_t, 4096> unequal_b{};
    std::array<uint8_t, 4096> unequal_c{};
    std::array<uint8_t, 4096> unequal_d{};
    std::memset(unequal_a.data(), 0x42, unequal_a.size());
    std::memset(unequal_b.data(), 0x43, unequal_b.size());
    std::memset(unequal_c.data(), 0x44, unequal_c.size());
    std::memset(unequal_d.data(), 0x45, unequal_d.size());
    const size_t sizes[6][4] = {{123, 0, 0, 0},
                                {2048, 1024, 0, 0},
                                {3072, 511, 7, 0},
                                {4096, 4096, 4096, 4096},
                                {333, 222, 0, 0},
                                {0, 0, 0, 0}};
    const size_t keys[6] = {1, 2, 3, 4, 5, 5};
    const void *sources[6][4] = {
        {unequal_a.data(), nullptr, nullptr, nullptr},
        {unequal_b.data(), unequal_c.data(), nullptr, nullptr},
        {unequal_c.data(), unequal_b.data(), unequal_a.data(), nullptr},
        {unequal_a.data(), unequal_b.data(), unequal_c.data(), unequal_d.data()},
        {unequal_c.data(), unequal_d.data(), nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr},
    };
    for (size_t row = 0; row < 5; ++row) {
        for (size_t i = 0; i < 4 && sizes[row][i] != 0; ++i) {
            uint64_t address = 0;
            checks.check(builder.add_object(sources[row][i], sizes[row][i],
                                             &address, keys[row]) ==
                             EcBatchStatus::kOk,
                         "interleaved partial/unequal source accepted");
            checks.check(address !=
                             FarLib::allocator::remote::InvalidRemoteAddr,
                         "partial group returns a valid remote data address");
        }
    }
    checks.check(builder.flush() == EcBatchStatus::kOk,
                 "flush seals all partial and unequal groups");
    checks.equal(builder.pending_count(), size_t{6},
                 "two full and four partial groups are pending");
    for (size_t key = 0; key < 6; ++key) {
        checks.equal(builder.sealed_count_for_behavior_group(key), uint64_t{1},
                     "each behavior key owns exactly one direct group");
    }
    while (builder.pending_count() != 0) {
        const uint64_t token = builder.front_token();
        auto *entry = bank.get(token);
        checks.check(entry != nullptr, "interleaved pending group is visible");
        if (entry != nullptr) {
            checks.check(entry->record.live_count >= 1 &&
                             entry->record.live_count <= 4,
                         "partial group has a valid live-object count");
            checks.check(parity_matches(*entry),
                         "partial/unequal group matches parity oracle");
            checks.check(recovery_matches(*entry),
                         "partial/unequal group passes recovery oracle");
        }
        builder.commit_pending();
        finish_group(bank, manager, token, checks,
                     "partial group waits for all six terminal completions");
    }
    checks.equal(builder.pending_count(), size_t{0},
                 "interleaved pending queue drains exactly once");
    checks.equal(bank.in_use(), size_t{0},
                 "partial-group completion leaves no bank slot in use");
}

void test_slot_reuse(SmallObjectStripeManager &manager, Checks &checks) {
    EcDirectWriteBank bank(1);
    checks.check(bank.init(nullptr, host_bank_allocate, host_bank_free),
                 "reuse bank initializes");
    EcDirectGroupBuilder builder(&manager, &bank, 0);
    std::array<std::array<uint8_t, 4072>, 4> objects{};
    std::unordered_set<uint64_t> tokens;
    void *first_parity = nullptr;

    constexpr size_t kRounds = 300;
    for (size_t round = 0; round < kRounds; ++round) {
        for (size_t i = 0; i < objects.size(); ++i) {
            std::memset(objects[i].data(),
                        static_cast<int>((round + i) & 0xff),
                        objects[i].size());
        }
        uint64_t ignored = 0;
        for (size_t i = 0; i < objects.size(); ++i) {
            const auto status = builder.add_object(
                objects[i].data(), objects[i].size(), &ignored, round % 6);
            if (status != EcBatchStatus::kOk) {
                checks.check(false, "each reuse round seals four objects");
            }
        }
        const uint64_t token = builder.front_token();
        auto *entry = bank.get(token);
        checks.check(entry != nullptr, "reused direct slot is visible");
        if (entry == nullptr) break;
        checks.check(tokens.insert(token).second,
                     "slot generation produces a unique completion token");
        checks.check(parity_matches(*entry),
                     "reused slot parity matches the current source round");
        checks.check(recovery_matches(*entry),
                     "reused slot passes the recovery oracle");
        if (round == 0) {
            first_parity = entry->record.parity[0];
            checks.equal(entry->record.slot, uint16_t{0},
                         "first round starts at bank slot zero");
            checks.equal(entry->record.generation, uint64_t{1},
                         "first round starts at generation one");
        }
        if (round == EcDirectWriteBank::kSlotsPerOwner) {
            checks.equal(entry->record.slot, uint16_t{0},
                         "round 129 reuses fixed bank slot zero");
            checks.check(entry->record.generation > 1,
                         "reused bank slot advances its generation");
            checks.check(entry->record.parity[0] == first_parity,
                         "reused bank slot keeps a fixed parity address");
        }
        builder.commit_pending();
        finish_group(bank, manager, token, checks,
                     "reuse round cannot release before the sixth completion");
        checks.equal(bank.in_use(), size_t{0},
                     "each reuse round returns its bank slot");
    }
    checks.equal(tokens.size(), kRounds,
                 "all multiple-round tokens remain generation-distinct");
    checks.equal(builder.pending_count(), size_t{0},
                 "reuse builder has no pending groups");
}

int run() {
    Checks checks;
    checks.check(configure_fixture(), "direct-group fixture configured");
    const auto &config = FarLib::g_ec_direct_group_config;
    SmallObjectStripeManager manager;
    manager.init(config.remote_total,
                 FarLib::rdma::Configure::ft_ec_shard_size_bytes);
    checks.check(manager.enabled(), "stripe manager initialized for direct groups");
    if (manager.enabled()) {
        test_four_object_group(manager, checks);
        test_partial_and_interleaved(manager, checks);
        test_slot_reuse(manager, checks);
    }
    std::printf("EC_DIRECT_GROUP checks=%d failures=%d\n", checks.count,
                checks.failures);
    if (checks.failures == 0) std::puts("EC_DIRECT_GROUP_PASS");
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_DIRECT_GROUP_SKIP_NO_VERBS");
    return 0;
#endif
}
