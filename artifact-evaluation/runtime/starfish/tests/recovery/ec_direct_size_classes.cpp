// CPU contract for behavior-group direct EC size-class stripes.
//
// The test keeps the direct path's borrowed source pointers intact, checks the
// real allocator bins and exercises the ISA-L-backed parity/rebuild helpers.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/ec_direct_group_builder.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_direct_size_class_config;
const rdma::Configure &get_config() { return g_ec_direct_size_class_config; }
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
    std::memset(base, 0, aligned);
    out->base = base;
    out->bytes = aligned;
    out->lkey = 0xec51u;
    out->registration = nullptr;
    return true;
}

void host_bank_free(void *, EcDirectWriteBank::RegisteredBank &bank) {
    std::free(bank.base);
    bank = EcDirectWriteBank::RegisteredBank{};
}

bool configure_fixture() {
    auto &config = FarLib::g_ec_direct_size_class_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    config.server_count = static_cast<int>(
        FarLib::rdma::Configure::ft_ec_batch_endpoint_count);
    // The bank-exhaustion case deliberately opens 128 groups over many bins.
    config.server_buffer_size = 16ull * 1024 * 1024;
    config.client_buffer_size = config.server_buffer_size;
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

size_t allocator_bin(size_t size) {
    return ::FarLib::allocator::bin_from_wsize(
        ::FarLib::allocator::wsize_from_size(size));
}

bool parity_and_rebuild_match(const EcDirectWriteBank::Entry &entry) {
    const size_t slot_size = entry.record.slot_size;
    if (slot_size == 0 || slot_size > 4096) return false;

    std::array<std::array<uint8_t, 4096>, 6> original{};
    const void *data[4]{};
    for (size_t i = 0; i < 4; ++i) {
        const size_t size = entry.record.object_sizes[i];
        if (size > slot_size || (size != 0 && entry.record.objects[i] == nullptr)) {
            return false;
        }
        if (size != 0) {
            std::memcpy(original[i].data(), entry.record.objects[i], size);
        }
        data[i] = original[i].data();
    }

    std::array<std::array<uint8_t, 4096>, 2> expected{};
    void *expected_parity[2] = {expected[0].data(), expected[1].data()};
    if (!FarLib::cache::small_object_stripe_encode_shards(
            data, expected_parity, slot_size, 0)) {
        return false;
    }
    for (size_t i = 0; i < 2; ++i) {
        if (entry.record.parity[i] == nullptr ||
            std::memcmp(entry.record.parity[i], expected[i].data(), slot_size) != 0) {
            return false;
        }
        std::memcpy(original[4 + i].data(), entry.record.parity[i], slot_size);
    }

    auto work = original;
    void *shards[6] = {work[0].data(), work[1].data(), work[2].data(),
                       work[3].data(), work[4].data(), work[5].data()};
    std::memset(work[1].data(), 0, slot_size);
    std::memset(work[4].data(), 0, slot_size);
    const uint8_t alive = static_cast<uint8_t>(0x3f &
        ~(uint8_t{1u << 1} | uint8_t{1u << 4}));
    if (!FarLib::cache::small_object_stripe_rebuild_shards(
            alive, shards, slot_size, 0)) {
        return false;
    }
    return std::memcmp(work[1].data(), original[1].data(), slot_size) == 0 &&
           std::memcmp(work[4].data(), original[4].data(), slot_size) == 0;
}

bool finish_group(EcDirectWriteBank &bank, SmallObjectStripeManager &manager,
                  uint64_t token) {
    auto *entry = bank.get(token);
    if (entry == nullptr) return false;
    const auto id = entry->record.group.id;
    for (uint8_t segment = 0; segment < 5; ++segment) {
        if (bank.complete_segment(token, segment, true) != nullptr) return false;
    }
    if (bank.complete_segment(token, 5, true) == nullptr) return false;
    if (!bank.release(token) || bank.get(token) != nullptr) return false;
    return manager.mark_dead_group(id);
}

bool add_one(EcDirectGroupBuilder &builder, const void *object, size_t size,
             size_t behavior, Checks &checks, const char *label) {
    uint64_t address = ::FarLib::allocator::remote::InvalidRemoteAddr;
    const auto status = builder.add_object(object, size, &address, behavior);
    checks.check(status == EcBatchStatus::kOk, label);
    checks.check(address != ::FarLib::allocator::remote::InvalidRemoteAddr,
                 "accepted direct object has a published address");
    return status == EcBatchStatus::kOk;
}

void drain(EcDirectGroupBuilder &builder, EcDirectWriteBank &bank,
           SmallObjectStripeManager &manager, Checks &checks,
           bool verify_codec) {
    while (builder.pending_count() != 0) {
        const uint64_t token = builder.front_token();
        auto *entry = bank.get(token);
        checks.check(entry != nullptr, "pending direct token is visible");
        if (entry != nullptr && verify_codec) {
            checks.check(parity_and_rebuild_match(*entry),
                         "ISA-L parity and two-shard decode match");
        }
        builder.commit_pending();
        checks.check(finish_group(bank, manager, token),
                     "six completions safely consume direct group");
    }
}

void test_legacy_compatibility(SmallObjectStripeManager &manager,
                               EcDirectWriteBank &bank, Checks &checks) {
    EcDirectGroupBuilder builder(&manager, &bank, 0);
    checks.check(!builder.size_classed(), "legacy constructor keeps size classes disabled");
    std::vector<uint8_t> large(4080, 0x41);
    std::vector<uint8_t> small(64, 0x42);
    add_one(builder, large.data(), large.size(), 0, checks,
            "legacy builder accepts large object");
    add_one(builder, small.data(), small.size(), 0, checks,
            "legacy builder packs smaller object into open larger slot");
    checks.check(builder.flush() == EcBatchStatus::kOk,
                 "legacy flush seals one partial group");
    checks.equal(builder.pending_count(), size_t{1},
                 "legacy mode keeps one behavior-only group");
    if (builder.pending_count() != 0) {
        auto *entry = bank.get(builder.front_token());
        checks.check(entry != nullptr && entry->record.slot_size == 4096 &&
                         entry->record.live_count == 2 &&
                         entry->record.object_sizes[0] == 4080 &&
                         entry->record.object_sizes[1] == 64,
                     "legacy group retains its original slot semantics");
        drain(builder, bank, manager, checks, true);
    }
}

void test_size_classed_groups(SmallObjectStripeManager &manager,
                              EcDirectWriteBank &bank, Checks &checks) {
    EcDirectGroupBuilder builder(&manager, &bank, 0, true);
    checks.check(builder.size_classed(), "size-class constructor enables bin keys");
    const size_t sizes[] = {64, 128, 512, 4080};
    std::array<std::array<size_t, ::FarLib::allocator::RegionBinCount>, 6>
        expected_sizes{};
    std::array<std::vector<uint8_t>, 4> objects;
    for (size_t i = 0; i < 4; ++i) {
        objects[i].assign(sizes[i], static_cast<uint8_t>(0x50 + i));
        expected_sizes[2][allocator_bin(sizes[i])] = sizes[i];
        add_one(builder, objects[i].data(), sizes[i], 2, checks,
                "size-classed behavior accepts interleaved class");
    }
    std::vector<uint8_t> other(128, 0x61);
    expected_sizes[4][allocator_bin(other.size())] = other.size();
    add_one(builder, other.data(), other.size(), 4, checks,
            "different behavior opens an independent same-bin group");
    checks.equal(builder.sealed_count_for_behavior_group(2), uint64_t{0},
                 "interleaved partial groups remain unsealed before flush");
    checks.check(builder.flush() == EcBatchStatus::kOk,
                 "size-classed flush seals every partial class");
    checks.equal(builder.pending_count(), size_t{5},
                 "each behavior/bin pair has an isolated pending group");
    checks.equal(builder.sealed_count_for_behavior_group(2), uint64_t{4},
                 "seal count indexes behavior rather than physical key");
    checks.equal(builder.sealed_count_for_behavior_group(4), uint64_t{1},
                 "other behavior is counted independently");

    while (builder.pending_count() != 0) {
        const uint64_t token = builder.front_token();
        EcDirectGroupBuilder::GroupMetadata metadata{};
        checks.check(builder.metadata_for_token(token, &metadata),
                     "pending token retains immutable behavior metadata");
        auto *entry = bank.get(token);
        const bool metadata_in_range =
            metadata.behavior_group < 6 &&
            metadata.allocator_bin < ::FarLib::allocator::RegionBinCount;
        checks.check(entry != nullptr && metadata_in_range &&
                         entry->record.live_count == 1 &&
                         entry->record.slot_size ==
                             ::FarLib::allocator::get_bin_size(metadata.allocator_bin) &&
                         entry->record.object_sizes[0] != 0 &&
                         expected_sizes[metadata.behavior_group]
                             [metadata.allocator_bin] == entry->record.object_sizes[0],
                     "record slot size is the exact allocator class");
        if (entry != nullptr) {
            checks.check(parity_and_rebuild_match(*entry),
                         "size-classed group passes ISA-L parity/decode oracle");
        }
        builder.commit_pending();
        checks.check(finish_group(bank, manager, token),
                     "size-classed group consumes without bank leakage");
    }

    std::vector<uint8_t> invalid(4097, 0x7a);
    uint64_t address = 0;
    checks.check(builder.add_object(invalid.data(), invalid.size(), &address, 2) ==
                     EcBatchStatus::kInvalidArgument,
                 ">4KiB direct object is rejected for the whole-object path");
    checks.check(builder.add_object(objects[0].data(), objects[0].size(), nullptr, 2) ==
                     EcBatchStatus::kInvalidArgument,
                 "null address output is rejected without reserving a group");
    checks.check(builder.add_object(objects[0].data(), objects[0].size(), &address, 6) ==
                     EcBatchStatus::kInvalidArgument,
                 "invalid behavior identity is rejected");
    checks.equal(bank.in_use(), size_t{0},
                 "invalid size-class requests consume no bank slot");
}

void test_bank_exhaustion(SmallObjectStripeManager &manager,
                          EcDirectWriteBank &bank, Checks &checks) {
    EcDirectGroupBuilder builder(&manager, &bank, 0, true);
    std::array<std::array<bool, ::FarLib::allocator::RegionBinCount>, 6> used{};
    std::vector<std::vector<uint8_t>> sources;
    sources.reserve(EcDirectWriteBank::kSlotsPerOwner + 1);
    size_t opened = 0;
    for (size_t behavior = 0;
         behavior < 6 && opened < EcDirectWriteBank::kSlotsPerOwner;
         ++behavior) {
        for (size_t size = 1;
             size <= 4096 && opened < EcDirectWriteBank::kSlotsPerOwner;
             ++size) {
            const size_t bin = allocator_bin(size);
            if (used[behavior][bin]) continue;
            used[behavior][bin] = true;
            sources.emplace_back(size, static_cast<uint8_t>(opened));
            if (!add_one(builder, sources.back().data(), size, behavior, checks,
                         "bank-exhaustion group reserves one physical class")) {
                return;
            }
            ++opened;
        }
    }
    checks.equal(opened, EcDirectWriteBank::kSlotsPerOwner,
                 "128 distinct behavior/bin groups fit the direct bank");
    size_t extra_behavior = 0;
    size_t extra_size = 1;
    bool found = false;
    for (size_t behavior = 0; behavior < 6 && !found; ++behavior) {
        for (size_t size = 1; size <= 4096; ++size) {
            const size_t bin = allocator_bin(size);
            if (!used[behavior][bin]) {
                extra_behavior = behavior;
                extra_size = size;
                found = true;
                break;
            }
        }
    }
    checks.check(found, "an additional physical behavior/bin key exists");
    sources.emplace_back(extra_size, 0xa5);
    uint64_t ignored = 0;
    checks.check(builder.add_object(sources.back().data(), extra_size, &ignored,
                                   extra_behavior) ==
                     EcBatchStatus::kStagingExhausted,
                 "bank exhaustion is reported without consuming the extra source");
    checks.equal(bank.in_use(), EcDirectWriteBank::kSlotsPerOwner,
                 "reserve failure leaves all existing bank leases intact");
    checks.check(builder.flush() == EcBatchStatus::kOk,
                 "flush seals all bank-resident partial groups");
    checks.equal(builder.pending_count(), EcDirectWriteBank::kSlotsPerOwner,
                 "all 128 groups enter the bounded pending queue");
    drain(builder, bank, manager, checks, false);
    checks.equal(bank.in_use(), size_t{0},
                 "completion consumption returns every bank slot");
    checks.equal(builder.pending_count(), size_t{0},
                 "bank-exhaustion pending queue drains exactly once");
}

int run() {
    Checks checks;
    checks.check(configure_fixture(), "size-class fixture configured");
    const auto &config = FarLib::g_ec_direct_size_class_config;
    SmallObjectStripeManager manager;
    manager.init(config.remote_total,
                 FarLib::rdma::Configure::ft_ec_shard_size_bytes);
    checks.check(manager.enabled(), "stripe manager initialized for size classes");
    if (manager.enabled()) {
        EcDirectWriteBank bank(1);
        checks.check(bank.init(nullptr, host_bank_allocate, host_bank_free),
                     "direct bank initializes one owner");
        if (bank.initialized()) {
            test_legacy_compatibility(manager, bank, checks);
            test_size_classed_groups(manager, bank, checks);
            test_bank_exhaustion(manager, bank, checks);
        }
    }
    std::printf("EC_DIRECT_SIZE_CLASSES checks=%d failures=%d\n", checks.count,
                checks.failures);
    if (checks.failures == 0) std::puts("EC_DIRECT_SIZE_CLASSES_PASS");
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_DIRECT_SIZE_CLASSES_SKIP_NO_VERBS");
    return 0;
#endif
}
