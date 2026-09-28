// CPU contract for the whole-object, semantic-group EC builder.
//
// This test uses the real SmallObjectStripeManager and real ISA-L codec, while
// BufferPools receives malloc-backed fake registration callbacks.  It stops
// before RDMA posting and checks the exact record/staging boundary consumed by
// the runtime sender.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#if __has_include(<infiniband/verbs.h>)

#include "cache/alloc/ec_size_class_builder.hpp"
#include "cache/alloc/small_object_stripe.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "rdma/config.hpp"

namespace FarLib {
static rdma::Configure g_ec_size_class_config;
const rdma::Configure &get_config() { return g_ec_size_class_config; }
}  // namespace FarLib

namespace FarLib::allocator::remote {
RemoteGlobalHeap remote_global_heap;
}  // namespace FarLib::allocator::remote

namespace {

using FarLib::cache::SmallObjectStripeManager;
using FarLib::cache::ec_batch::EcBatchStatus;
using FarLib::cache::ec_batch::EcSizeClassBuilder;
using FarLib::cache::ec_split::BufferPools;
using FarLib::cache::ec_read_recovery::RecoveryScratchChunk;

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

struct Registration {
    uint32_t lkey = 0;
};

struct RegistrationContext {
    std::mutex mutex;
    std::vector<std::pair<void *, Registration *>> allocations;
    uint32_t next_lkey = 0x5000;
};

bool allocate_chunk(void *opaque, size_t bytes, RecoveryScratchChunk *out) {
    if (out == nullptr) return false;
    auto &context = *static_cast<RegistrationContext *>(opaque);
    void *base = std::malloc(bytes);
    if (base == nullptr) return false;
    auto *registration = new (std::nothrow) Registration{context.next_lkey++};
    if (registration == nullptr) {
        std::free(base);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(context.mutex);
        context.allocations.push_back({base, registration});
    }
    *out = {base, registration->lkey, registration};
    return true;
}

void release_chunk(void *opaque, RecoveryScratchChunk chunk) {
    auto &context = *static_cast<RegistrationContext *>(opaque);
    std::lock_guard<std::mutex> lock(context.mutex);
    for (auto it = context.allocations.begin(); it != context.allocations.end();
         ++it) {
        if (it->first != chunk.base || it->second != chunk.registration) {
            continue;
        }
        std::free(it->first);
        delete it->second;
        context.allocations.erase(it);
        return;
    }
    assert(false && "unknown fake registered chunk");
}

bool configure_fixture() {
    auto &config = FarLib::g_ec_size_class_config;
    const size_t shard_size = FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    config.server_count = static_cast<int>(
        FarLib::rdma::Configure::ft_ec_batch_endpoint_count);
    // Enough remote stripe capacity for the explicit 64-open-group bound.
    config.server_buffer_size = 64ull * 1024 * 1024;
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

size_t bin_for_size(size_t size) {
    return ::FarLib::allocator::bin_from_wsize(
        ::FarLib::allocator::wsize_from_size(size));
}

std::array<std::vector<uint8_t>, 6> copy_record_shards(
    const EcSizeClassBuilder::SealedGroup &record) {
    std::array<std::vector<uint8_t>, 6> shards;
    for (size_t i = 0; i < 4; ++i) {
        const auto *src = static_cast<const uint8_t *>(record.staging.data[i]);
        shards[i].assign(src, src + record.slot_size);
    }
    for (size_t i = 0; i < 2; ++i) {
        const auto *src =
            static_cast<const uint8_t *>(record.staging.parity[i]);
        shards[4 + i].assign(src, src + record.slot_size);
    }
    return shards;
}

bool recover_all_single_and_double_losses(
    const std::array<std::vector<uint8_t>, 6> &encoded,
    size_t slot_size) {
    for (uint8_t missing = 1; missing < (1u << 6); ++missing) {
        const unsigned count = __builtin_popcount(static_cast<unsigned>(missing));
        if (count != 1 && count != 2) continue;
        auto recovered = encoded;
        std::array<void *, 6> shards{};
        for (size_t i = 0; i < shards.size(); ++i) {
            shards[i] = recovered[i].data();
            if ((missing & static_cast<uint8_t>(1u << i)) != 0) {
                std::memset(shards[i], 0xcd, slot_size);
            }
        }
        const uint8_t alive = static_cast<uint8_t>(~missing) & 0x3fu;
        if (!FarLib::cache::small_object_stripe_rebuild_shards(
                alive, shards.data(), slot_size, 0)) {
            return false;
        }
        if (recovered != encoded) return false;
    }
    return true;
}

bool cleanup_group(SmallObjectStripeManager &manager, EcSizeClassBuilder &builder,
                   const EcSizeClassBuilder::SealedGroup &record) {
    return builder.commit_pending(record.sequence) &&
           builder.release_sent(record) &&
           manager.mark_dead_group(record.group.id);
}

std::array<size_t, 4> lengths_for_bin(size_t requested) {
    const size_t bin = bin_for_size(requested);
    const size_t slot_size = ::FarLib::allocator::get_bin_size(bin);
    const size_t lower = bin == 0 ? size_t{1}
                                  : ::FarLib::allocator::get_bin_size(bin - 1) + 1;
    return {slot_size, slot_size - 1,
            slot_size > lower + 7 ? slot_size - 7 : lower,
            lower};
}

void run_full_case(Checks &checks, SmallObjectStripeManager &manager,
                   BufferPools &pools, size_t requested, uint32_t behavior,
                   size_t owner) {
    const auto lengths = lengths_for_bin(requested);
    std::array<std::vector<uint8_t>, 4> objects;
    for (size_t i = 0; i < objects.size(); ++i) {
        objects[i].resize(lengths[i]);
        for (size_t j = 0; j < objects[i].size(); ++j) {
            objects[i][j] = static_cast<uint8_t>(
                (j * 17u + i * 53u + requested) & 0xffu);
        }
    }

    EcSizeClassBuilder builder(&manager, &pools, owner);
    checks.check(builder.valid(), "size-class builder is valid");
    std::array<uint64_t, 4> addresses{};
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto status = builder.add_object(
            objects[i].data(), objects[i].size(), &addresses[i], behavior);
        checks.check(status == EcBatchStatus::kOk,
                     "whole object staged/sealed with ISA-L backend");
    }
    checks.check(builder.pending_count() == 1, "full size-class group is pending");

    EcSizeClassBuilder::SealedGroup record;
    checks.check(builder.peek_pending(&record), "whole group is peekable");
    checks.equal(record.behavior_group, behavior,
                 "record preserves semantic behavior key");
    checks.equal(record.live_mask, uint8_t{0xf}, "whole group has four live slots");
    checks.check(!record.split_object && record.size_class_staging,
                 "record is whole-object size-class staging");
    checks.check(record.slot_size ==
                     ::FarLib::allocator::get_bin_size(bin_for_size(requested)),
                 "record slot size is the exact allocator bin");
    for (size_t i = 0; i < objects.size(); ++i) {
        checks.check(addresses[i] == record.group.segments[i].addr &&
                         addresses[i] !=
                             ::FarLib::allocator::remote::InvalidRemoteAddr,
                     "object address is published before pending post");
        checks.equal(record.object_sizes[i],
                     static_cast<uint32_t>(objects[i].size()),
                     "record keeps exact object payload size");
    }

    const auto encoded = copy_record_shards(record);
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto *data = static_cast<const uint8_t *>(record.staging.data[i]);
        bool contents_ok = true;
        for (size_t j = 0; j < objects[i].size(); ++j) {
            if (data[j] != objects[i][j]) contents_ok = false;
        }
        for (size_t j = objects[i].size(); j < record.slot_size; ++j) {
            if (data[j] != 0) contents_ok = false;
        }
        checks.check(contents_ok, "data shard copies and zero-pads payload");
    }
    checks.check(recover_all_single_and_double_losses(encoded, record.slot_size),
                 "ISA-L RS(4,2) recovers every single/double shard loss");
    checks.check(cleanup_group(manager, builder, record),
                 "whole group commits, releases staging, and retires");
}

void run_hole_case(Checks &checks, SmallObjectStripeManager &manager,
                   BufferPools &pools) {
    const auto lengths = lengths_for_bin(128);
    std::array<std::vector<uint8_t>, 2> objects;
    for (size_t i = 0; i < objects.size(); ++i) {
        objects[i].resize(lengths[i]);
        std::memset(objects[i].data(), static_cast<int>(0x70 + i),
                    objects[i].size());
    }
    EcSizeClassBuilder builder(&manager, &pools, 77);
    uint64_t ignored = 0;
    checks.check(builder.add_object(objects[0].data(), objects[0].size(),
                                   &ignored, 2) == EcBatchStatus::kOk,
                 "hole group object zero staged");
    checks.check(builder.add_object(objects[1].data(), objects[1].size(),
                                   &ignored, 2) == EcBatchStatus::kOk,
                 "hole group object one staged");
    checks.check(builder.flush() == EcBatchStatus::kOk,
                 "flush seals a partial group with holes");

    EcSizeClassBuilder::SealedGroup record;
    checks.check(builder.peek_pending(&record), "partial group is peekable");
    checks.equal(record.live_mask, uint8_t{0x3},
                 "partial group records two live data slots");
    checks.equal(record.live_count, uint8_t{2},
                 "partial group records two live objects");
    checks.equal(record.object_sizes[2], uint32_t{0}, "hole payload size is zero");
    const auto encoded = copy_record_shards(record);
    bool holes_zero = true;
    for (size_t shard = 2; shard < 4; ++shard) {
        for (uint8_t byte : encoded[shard]) {
            if (byte != 0) holes_zero = false;
        }
    }
    checks.check(holes_zero, "partial group zero-fills unused data shards");
    checks.check(recover_all_single_and_double_losses(encoded, record.slot_size),
                 "ISA-L recovery covers every loss mask with holes");
    checks.check(cleanup_group(manager, builder, record),
                 "partial group commits and retires cleanly");
}

void run_backpressure_case(Checks &checks, SmallObjectStripeManager &manager,
                           BufferPools &pools) {
    std::array<uint8_t, 128> object{};
    std::memset(object.data(), 0x5a, object.size());
    EcSizeClassBuilder builder(&manager, &pools, 88, 1);
    uint64_t address_a = 0;
    for (size_t i = 0; i < 4; ++i) {
        checks.check(builder.add_object(object.data(), object.size(),
                                        &address_a, 0) == EcBatchStatus::kOk,
                     "first pending-capacity group stages");
    }
    EcSizeClassBuilder::SealedGroup first;
    checks.check(builder.peek_pending(&first), "first group fills pending capacity");

    uint64_t address_b = 0;
    EcBatchStatus final_status = EcBatchStatus::kOk;
    for (size_t i = 0; i < 4; ++i) {
        final_status = builder.add_object(object.data(), object.size(),
                                          &address_b, 1);
    }
    checks.check(final_status == EcBatchStatus::kPendingQueueFull,
                 "fourth object reports pending backpressure after consumption");
    checks.check(address_b !=
                     ::FarLib::allocator::remote::InvalidRemoteAddr,
                 "backpressured fourth object retains published address");
    checks.check(builder.group_open() && builder.pending_count() == 1,
                 "full backpressured group remains open for retry");

    checks.check(cleanup_group(manager, builder, first),
                 "first pending group drains and retires");
    checks.check(builder.flush() == EcBatchStatus::kOk,
                 "retained full group seals after pending drain");
    EcSizeClassBuilder::SealedGroup second;
    checks.check(builder.peek_pending(&second), "retained group is sealed once");
    checks.equal(second.behavior_group, uint32_t{1},
                 "retained group keeps behavior identity");
    checks.check(cleanup_group(manager, builder, second),
                 "retained group commits and retires");
    checks.equal(builder.pending_count(), size_t{0},
                 "backpressure pending queue is empty");
}

void run_rejection_case(Checks &checks, SmallObjectStripeManager &manager,
                        BufferPools &pools) {
    EcSizeClassBuilder builder(&manager, &pools, 99);
    std::array<uint8_t, 4097> large{};
    std::array<uint8_t, 64> small{};
    checks.check(builder.add_object(nullptr, small.size(), nullptr, 0) ==
                     EcBatchStatus::kInvalidArgument,
                 "null object is rejected");
    checks.check(builder.add_object(small.data(), 0, nullptr, 0) ==
                     EcBatchStatus::kInvalidArgument,
                 "zero-sized object is rejected");
    checks.check(builder.add_object(large.data(), large.size(), nullptr, 0) ==
                     EcBatchStatus::kObjectTooLarge,
                 "object above 4096 remains on split path");
    checks.check(builder.add_object(small.data(), small.size(), nullptr, 6) ==
                     EcBatchStatus::kInvalidArgument,
                 "invalid behavior key is rejected");
    checks.check(!builder.group_open() && builder.pending_count() == 0 &&
                     builder.staging_in_use() == 0,
                 "rejections do not allocate remote groups or staging");
}

void run_open_limit_case(Checks &checks, SmallObjectStripeManager &manager,
                         BufferPools &pools) {
    EcSizeClassBuilder builder(&manager, &pools, 111);
    std::vector<std::vector<uint8_t>> objects;
    objects.reserve(64);
    // Low allocator IDs are not all canonical request classes: for wsize <= 8
    // bin_from_wsize() rounds several neighboring BinSize entries to the same
    // odd-word class.  Enumerate canonical bins instead of accidentally
    // reusing one (behavior,bin) key and mistaking allocator aliasing for the
    // builder's 64-open-group bound.
    std::array<size_t, 12> canonical_bins{};
    size_t canonical_count = 0;
    for (size_t bin = 0;
         bin < ::FarLib::allocator::RegionBinCount && canonical_count < 12;
         ++bin) {
        const size_t size = ::FarLib::allocator::get_bin_size(bin);
        if (size > FarLib::cache::ec_batch::kEcSizeClassMaxSlotSize) break;
        if (bin_for_size(size) == bin) {
            canonical_bins[canonical_count++] = bin;
        }
    }
    checks.equal(canonical_count, size_t{12},
                 "open-limit fixture found 12 canonical bins");
    if (canonical_count != 12) return;
    size_t added = 0;
    bool all_staged = true;
    for (uint32_t behavior = 0; behavior < 6 && added < 64; ++behavior) {
        for (size_t canonical = 0; canonical < 11; ++canonical) {
            if (added == 64) break;
            const size_t bin = canonical_bins[canonical];
            const size_t size = ::FarLib::allocator::get_bin_size(bin);
            objects.emplace_back(size, static_cast<uint8_t>(behavior + bin));
            uint64_t address = 0;
            if (builder.add_object(objects.back().data(), size, &address,
                                   behavior) != EcBatchStatus::kOk) {
                all_staged = false;
                break;
            }
            ++added;
        }
    }
    checks.check(all_staged && added == 64 && builder.open_group_count() == 64,
                 "builder admits exactly 64 bounded open groups");
    const size_t overflow_bin = canonical_bins[11];
    std::vector<uint8_t> overflow(
        ::FarLib::allocator::get_bin_size(overflow_bin), 0x3c);
    uint64_t overflow_address = 0;
    checks.check(builder.add_object(overflow.data(), overflow.size(),
                                   &overflow_address, 5) ==
                     EcBatchStatus::kStagingExhausted,
                 "65th distinct open key is bounded without allocation");
    checks.check(builder.drop_partial() == EcBatchStatus::kOk &&
                     builder.open_group_count() == 0 &&
                     builder.staging_in_use() == 0,
                 "bounded open groups release on partial teardown");
}

int run() {
    Checks checks;
    checks.check(configure_fixture(), "size-class EC fixture configured");
    const auto &config = FarLib::g_ec_size_class_config;
    SmallObjectStripeManager manager;
    manager.init(config.remote_total,
                 FarLib::rdma::Configure::ft_ec_shard_size_bytes);
    checks.check(manager.enabled(), "real stripe manager initialized");

    RegistrationContext registration;
    BufferPools pools;
    checks.check(pools.init(&registration, allocate_chunk, release_chunk),
                 "fake registered BufferPools initialized");
    checks.check(pools.valid(), "fake registered BufferPools valid");

    for (const auto requested : {size_t{64}, size_t{128}, size_t{512},
                                  size_t{4080}, size_t{4096}}) {
        run_full_case(checks, manager, pools, requested,
                      static_cast<uint32_t>(requested == 4096 ? 5 : 2),
                      requested + 1);
    }
    run_hole_case(checks, manager, pools);
    run_backpressure_case(checks, manager, pools);
    run_rejection_case(checks, manager, pools);
    run_open_limit_case(checks, manager, pools);

    checks.equal(pools.in_use(), size_t{0},
                 "all size-class staging leases are returned");
    std::printf("EC_SIZE_CLASS_BUILDER checks=%d failures=%d\n", checks.count,
                checks.failures);
    if (checks.failures == 0) std::puts("EC_SIZE_CLASS_BUILDER_PASS");
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_SIZE_CLASS_BUILDER_SKIP_NO_VERBS");
    return 0;
#endif
}
