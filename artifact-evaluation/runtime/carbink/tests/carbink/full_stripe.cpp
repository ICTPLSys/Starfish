// CPU-only checks for the Carbink page-sized full-stripe builder path.
//
// The fake manager deliberately implements only the group API consumed by
// EcGroupBuilderT.  No RemoteGlobalHeap, RDMA object, or worker is constructed.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/alloc/ec_group_builder.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "hydra/page_codec.hpp"
#include "hydra/page_layout.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using RealManager = FarLib::cache::SmallObjectStripeManager;

class FakeManager {
public:
    using SlotGroupId = RealManager::SlotGroupId;
    using SlotGroupSegment = RealManager::SlotGroupSegment;
    using SlotGroupHandle = RealManager::SlotGroupHandle;

    explicit FakeManager(uint32_t slot_size) : slot_size_(slot_size) {}

    bool allocate_slot_group(size_t requested, SlotGroupHandle *out) {
        if (out == nullptr || requested == 0 || requested > slot_size_) {
            return false;
        }
        const uint64_t group_number = next_group_++;
        SlotGroupHandle handle;
        handle.id.stripe_id = 0x100 + group_number;
        handle.id.slot_id = static_cast<uint32_t>(group_number);
        handle.bin = 0;
        handle.slot_size = slot_size_;
        handle.slots_per_shard = 32;
        handle.slot_offset = group_number * slot_size_;
        for (size_t i = 0; i < handle.segments.size(); ++i) {
            handle.segments[i].shard_idx = static_cast<uint8_t>(i);
            handle.segments[i].endpoint_idx = static_cast<uint32_t>(i);
            handle.segments[i].offset = handle.slot_offset;
            handle.segments[i].addr =
                0x100000000ull + group_number * 0x100000ull +
                static_cast<uint64_t>(i) * slot_size_;
            handle.segments[i].slot_size = slot_size_;
        }
        last_handle_ = handle;
        allocated_ = true;
        *out = handle;
        return true;
    }

    bool seal_slot_group(SlotGroupId id, uint8_t live_mask) {
        if (!allocated_ || id.stripe_id != last_handle_.id.stripe_id ||
            id.slot_id != last_handle_.id.slot_id) {
            return false;
        }
        sealed_ = true;
        sealed_id_ = id;
        sealed_mask_ = live_mask;
        return true;
    }

    bool mark_dead_group(SlotGroupId id) {
        if (!allocated_ || id.stripe_id != last_handle_.id.stripe_id ||
            id.slot_id != last_handle_.id.slot_id) {
            return false;
        }
        ++dead_calls_;
        return true;
    }

    const SlotGroupHandle &last_handle() const { return last_handle_; }
    bool sealed() const { return sealed_; }
    uint8_t sealed_mask() const { return sealed_mask_; }
    size_t dead_calls() const { return dead_calls_; }

private:
    uint32_t slot_size_;
    uint64_t next_group_ = 0;
    bool allocated_ = false;
    bool sealed_ = false;
    SlotGroupId sealed_id_{};
    uint8_t sealed_mask_ = 0;
    size_t dead_calls_ = 0;
    SlotGroupHandle last_handle_{};
};

using BuilderForFake = FarLib::cache::ec_batch::EcGroupBuilderT<FakeManager>;
using StagingPool = FarLib::cache::ec_batch::EcStagingPool;
using Status = FarLib::cache::ec_batch::EcBatchStatus;

void initialize_pool(StagingPool *pool, std::vector<uint8_t> *storage,
                     size_t depth = 2) {
    const size_t bytes = StagingPool::required_bytes(
        FarLib::hydra::kPageBytes, depth);
    storage->assign(bytes, 0xa5);
    assert(pool->init(storage->data(), storage->size(),
                      FarLib::hydra::kPageBytes, depth));
}

void compare_page_codec_with_existing_rs() {
    // The imported server builds its coefficients with gf_gen_rs_matrix.
    // Check that it uses the same two rows as Hydra's full-stripe writer.
    unsigned char legacy_matrix[6 * 4]{};
    const unsigned char coefficients[] = {1, 1, 1, 1, 1, 2, 4, 8};
    gf_gen_rs_matrix(legacy_matrix, 6, 4);
    assert(std::memcmp(legacy_matrix + 4 * 4, coefficients, sizeof(coefficients)) == 0);
    constexpr size_t kBytes = FarLib::hydra::kPageBytes;
    std::array<std::vector<uint8_t>, 4> data;
    std::array<std::vector<uint8_t>, 2> old_parity;
    std::array<std::vector<uint8_t>, 2> page_parity;
    for (size_t shard = 0; shard < data.size(); ++shard) {
        data[shard].resize(kBytes);
        for (size_t i = 0; i < kBytes; ++i) {
            data[shard][i] = static_cast<uint8_t>(
                (i * 17u + shard * 53u + 11u) & 0xffu);
        }
    }
    for (auto &shard : old_parity) shard.assign(kBytes, 0xa5);
    for (auto &shard : page_parity) shard.assign(kBytes, 0xa5);

    std::array<const void *, 4> data_ptrs{};
    std::array<void *, 2> old_ptrs{};
    std::array<void *, 2> page_ptrs{};
    for (size_t shard = 0; shard < data.size(); ++shard) {
        data_ptrs[shard] = data[shard].data();
    }
    for (size_t shard = 0; shard < 2; ++shard) {
        old_ptrs[shard] = old_parity[shard].data();
        page_ptrs[shard] = page_parity[shard].data();
    }

    assert(FarLib::cache::small_object_stripe_encode_shards(
        data_ptrs.data(), old_ptrs.data(), kBytes));
    assert(FarLib::hydra::page_codec_encode_4plus2(
        data_ptrs.data(), page_ptrs.data(), kBytes));
    assert(old_parity == page_parity);
}

void check_complete_8k_group() {
    constexpr size_t kBytes = FarLib::hydra::kPageBytes;
    FakeManager manager(static_cast<uint32_t>(kBytes));
    StagingPool pool;
    std::vector<uint8_t> storage;
    initialize_pool(&pool, &storage);
    BuilderForFake builder(
        &manager, &pool, 4, &FarLib::hydra::page_codec_encode_4plus2);

    std::array<std::vector<uint8_t>, 4> objects;
    for (size_t object = 0; object < objects.size(); ++object) {
        objects[object].resize(kBytes);
        for (size_t i = 0; i < kBytes; ++i) {
            objects[object][i] = static_cast<uint8_t>(
                (object * 31u + i * 7u + 1u) & 0xffu);
        }
        uint64_t remote_addr = 0;
        assert(builder.add_object(objects[object].data(), kBytes,
                                  &remote_addr) == Status::kOk);
        assert(remote_addr == manager.last_handle().segments[object].addr);
    }

    assert(manager.sealed());
    assert(manager.sealed_mask() == 0xf);
    assert(builder.automatic_seal_count() == 1);
    assert(builder.partial_seal_count() == 0);
    assert(builder.pending_count() == 1);

    BuilderForFake::SealedGroup record;
    assert(builder.pop_pending(&record));
    assert(record.group.slot_size == kBytes);
    assert(record.live_mask == 0xf);
    assert(record.live_count == 4);
    assert(!record.split_object);
    assert(!record.direct_page_data);
    for (size_t segment = 0; segment < 6; ++segment) {
        assert(record.group.segments[segment].slot_size == kBytes);
        assert(record.group.segments[segment].offset ==
               record.group.slot_offset);
    }
    for (size_t data = 0; data < 4; ++data) {
        assert(std::memcmp(record.staging.data[data],
                           objects[data].data(), kBytes) == 0);
    }
    assert(builder.release_sent(record));
    assert(pool.in_use() == 0);
}

void check_partial_group_zero_fill() {
    constexpr size_t kBytes = FarLib::hydra::kPageBytes;
    constexpr size_t kObjectBytes = 1237;
    FakeManager manager(static_cast<uint32_t>(kBytes));
    StagingPool pool;
    std::vector<uint8_t> storage;
    initialize_pool(&pool, &storage, 1);

    // Poison the only lease before returning it, so the test observes the
    // builder's explicit zero-fill rather than vector initialization.
    FarLib::cache::ec_batch::EcStagingGroupSlot poison;
    assert(pool.acquire(&poison));
    for (size_t shard = 0; shard < 4; ++shard) {
        std::memset(poison.data[shard], 0xa5, kBytes);
    }
    for (size_t shard = 0; shard < 2; ++shard) {
        std::memset(poison.parity[shard], 0xa5, kBytes);
    }
    assert(pool.release(poison));

    BuilderForFake builder(
        &manager, &pool, 2, &FarLib::hydra::page_codec_encode_4plus2);
    std::vector<uint8_t> object(kObjectBytes);
    for (size_t i = 0; i < object.size(); ++i) {
        object[i] = static_cast<uint8_t>((i * 19u + 3u) & 0xffu);
    }
    assert(builder.add_object(object.data(), object.size()) == Status::kOk);
    assert(builder.flush() == Status::kOk);
    assert(builder.partial_seal_count() == 1);

    BuilderForFake::SealedGroup record;
    assert(builder.peek_pending(&record));
    assert(record.live_mask == 0x1);
    assert(record.live_count == 1);
    const auto *first = static_cast<const uint8_t *>(record.staging.data[0]);
    for (size_t i = 0; i < object.size(); ++i) {
        assert(first[i] == object[i]);
    }
    for (size_t i = object.size(); i < kBytes; ++i) {
        assert(first[i] == 0);
    }
    for (size_t data = 1; data < 4; ++data) {
        const auto *hole =
            static_cast<const uint8_t *>(record.staging.data[data]);
        for (size_t i = 0; i < kBytes; ++i) assert(hole[i] == 0);
    }
    assert(builder.pop_pending(&record));
    assert(builder.release_sent(record));
    assert(pool.in_use() == 0);
}

}  // namespace

int main() {
    compare_page_codec_with_existing_rs();
    check_complete_8k_group();
    check_partial_group_zero_fill();
    std::puts("CARBINK_FULL_STRIPE_PASS");
    return 0;
}
