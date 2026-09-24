#include "hydra/page_write.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>

#if __has_include(<infiniband/verbs.h>)
#include "hydra/write_ring.hpp"
#endif

using namespace FarLib;

#if __has_include(<infiniband/verbs.h>)
namespace {
struct ParityAllocator {
    size_t calls = 0;
    size_t frees = 0;
    size_t bytes = 0;
};
bool allocate_parity(void *ctx, size_t bytes,
                     hydra::WriteRing::RegisteredParity *out) {
    auto &allocator = *static_cast<ParityAllocator *>(ctx);
    void *base = std::malloc(bytes);
    if (base == nullptr) return false;
    ++allocator.calls;
    allocator.bytes += bytes;
    *out = {base, 17, base};
    return true;
}
void free_parity(void *ctx, hydra::WriteRing::RegisteredParity value) {
    ++static_cast<ParityAllocator *>(ctx)->frees;
    std::free(value.base);
}
void test_fixed_parity_codec() {
    ParityAllocator allocator;
    {
        hydra::WriteRing ring(1, 1);
        assert(ring.init_parity_buffers(&allocator, allocate_parity, free_parity));
        assert(allocator.calls == 1 && allocator.bytes == 4096);
        std::array<unsigned char, hydra::kPageBytes> page{};
        std::array<std::array<unsigned char, 2048>, 6> expected{};
        cache::ec_batch::EcStagingGroupSlot copied;
        copied.slot_size = 2048;
        for (size_t i = 0; i < 4; ++i) copied.data[i] = expected[i].data();
        for (size_t i = 0; i < 2; ++i) copied.parity[i] = expected[i + 4].data();
        void *fixed_parity = nullptr;
        for (size_t round = 0; round < 128; ++round) {
            for (size_t i = 0; i < page.size(); ++i)
                page[i] = static_cast<unsigned char>(round * 13 + i * 71 + i / 29);
            const auto before = page;
            uint64_t token = 0;
            hydra::WriteRing::Entry *entry = nullptr;
            cache::ec_batch::EcStagingGroupSlot scratch;
            assert(ring.reserve_page(0, &token, &entry, &scratch));
            assert(ring.owns_parity(token, scratch));
            for (void *data : scratch.data) assert(data == nullptr);
            if (round == 0) fixed_parity = scratch.parity[0];
            assert(scratch.parity[0] == fixed_parity);
            assert(static_cast<unsigned char *>(scratch.parity[1]) ==
                   static_cast<unsigned char *>(scratch.parity[0]) + 2048);
            assert(hydra::encode_page_parity_only(page.data(), scratch));
            assert(cache::ec_split::encode(page.data(), page.size(), copied, round % 2));
            assert(page == before);
            for (size_t i = 0; i < 2; ++i)
                assert(std::memcmp(scratch.parity[i], expected[i + 4].data(), 2048) == 0);
            cache::ec_batch::EcGroupSendRecord record;
            record.split_object = true;
            record.direct_page_data = true;
            record.objects[0] = page.data();
            record.object_sizes[0] = page.size();
            record.slot_size = 2048;
            record.staging = scratch;
            for (size_t i = 0; i < 4; ++i)
                assert(record.source_addr(i) ==
                       reinterpret_cast<uintptr_t>(page.data() + i * 2048));
            for (size_t i = 0; i < 2; ++i) {
                assert(record.source_addr(i + 4) ==
                       reinterpret_cast<uintptr_t>(scratch.parity[i]));
                assert(record.source_offset(i + 4) == scratch.mr_offset + i * 2048);
            }
            assert(ring.publish_page(token, record));
            for (uint8_t segment = 0; segment < 5; ++segment)
                assert(ring.complete_segment(token, segment) == nullptr);
            assert(ring.available(0) == 0);
            assert(!ring.release(token));
            // The final parity completion may report failure; the fixed
            // buffer still cannot be reused before this sixth terminal event.
            assert(ring.complete_segment(token, 5, round % 2 == 0) == entry);
            assert(entry->recoverable());
            assert(ring.release(token));
            assert(allocator.calls == 1 && allocator.frees == 0);
        }
        assert(ring.in_use() == 0);
    }
    assert(allocator.frees == 1);
}

void test_double_buffer_inflight_and_reuse() {
    using Ring = hydra::WriteRing;
    constexpr size_t half = Ring::kPagesPerBuffer;
    constexpr size_t count = Ring::kDoubleBufferedSlots;
    static_assert(half == 32 && count == 64);
    ParityAllocator allocator;
    {
        Ring ring(2, count, true);
        assert(ring.init_parity_buffers(&allocator, allocate_parity, free_parity));
        assert(allocator.calls == 2 && allocator.bytes == 2 * count * 4096);
        using Page = std::array<unsigned char, hydra::kPageBytes>;
        using Parity = std::array<unsigned char, 4096>;
        auto pages = std::make_unique<Page[]>(count);
        auto parity = std::make_unique<Parity[]>(count);
        std::array<uint64_t, count> tokens{}, original_tokens{};
        std::array<cache::ec_batch::EcStagingGroupSlot, count> slots{};
        auto prepare = [&](size_t index, unsigned seed) {
            auto &page = pages[index];
            for (size_t i = 0; i < page.size(); ++i)
                page[i] = static_cast<unsigned char>(seed + 71 * i + i / 17);
            const auto before = page;
            Ring::Entry *entry = nullptr;
            assert(ring.reserve_page(0, &tokens[index], &entry, &slots[index]));
            assert(hydra::encode_page_parity_only(page.data(), slots[index]));
            assert(page == before);
            std::memcpy(parity[index].data(), slots[index].parity[0], 4096);
            cache::ec_batch::EcGroupSendRecord record{};
            record.split_object = record.direct_page_data = true;
            record.objects[0] = page.data();
            record.object_sizes[0] = page.size();
            record.slot_size = 2048;
            record.staging = slots[index];
            for (size_t s = 0; s < 4; ++s)
                assert(record.source_addr(s) == reinterpret_cast<uintptr_t>(page.data() + s * 2048));
            assert(ring.publish_page(tokens[index], record));
        };
        // Keep every first-half page in flight. The second half must be usable
        // without any completion or new allocation/registration.
        for (size_t i = 0; i < count; ++i) {
            prepare(i, i);
            assert(slots[i].index == i);
            assert(slots[i].mr_offset == i * 4096);
            assert(slots[i].lkey == slots[0].lkey);
            if (i == half - 1) assert(ring.available(0) == half);
        }
        original_tokens = tokens;
        assert(ring.in_use() == count && ring.available(0) == 0);
        Ring::Entry *entry = nullptr;
        uint64_t extra = 0;
        cache::ec_batch::EcStagingGroupSlot extra_slot{};
        assert(!ring.reserve_page(0, &extra, &entry, &extra_slot));
        assert(ring.reserve_page(1, &extra, &entry, &extra_slot));
        assert(ring.cancel_page(extra));
        for (size_t i = 0; i < count; ++i)
            assert(std::memcmp(parity[i].data(), slots[i].parity[0], 4096) == 0);
        // Five out-of-order completions cannot release any first-half slot.
        for (size_t i = 0; i < half; ++i) {
            for (uint8_t s : {uint8_t(5), uint8_t(3), uint8_t(1), uint8_t(4), uint8_t(0)})
                assert(ring.complete_segment(tokens[i], s) == nullptr);
            assert(!ring.release(tokens[i]));
        }
        assert(!ring.reserve_page(0, &extra, &entry, &extra_slot));
        for (size_t i = 0; i < half; ++i) {
            auto *done = ring.complete_segment(tokens[i], 2, i % 2 == 0);
            assert(done && done->recoverable());
            assert(ring.release(tokens[i]));
        }
        // Rewrite/re-encode first-half pages while all second-half pages are
        // still in flight. Ignore old completions; do not overwrite bank B.
        for (size_t i = 0; i < half; ++i) {
            prepare(i, 0xa7 + i);
            assert(slots[i].index < half);
            for (uint8_t s = 0; s < 6; ++s)
                assert(ring.complete_segment(original_tokens[i], s) == nullptr);
        }
        for (size_t i = half; i < count; ++i)
            assert(std::memcmp(parity[i].data(), slots[i].parity[0], 4096) == 0);
        assert(allocator.calls == 2 && allocator.frees == 0);
        for (size_t i = 0; i < count; ++i) {
            Ring::Entry *done = nullptr;
            for (int s = 5; s >= 0; --s) {
                auto *candidate = ring.complete_segment(tokens[i], s, s > 1);
                if (candidate) done = candidate;
            }
            assert(done && done->recoverable());
            assert(ring.release(tokens[i]));
        }
        assert(ring.in_use() == 0 && ring.available(0) == count);
    }
    assert(allocator.frees == 2);
}
} // namespace
#endif

int main() {
#if __has_include(<infiniband/verbs.h>)
    test_fixed_parity_codec();
    test_double_buffer_inflight_and_reuse();
#endif
    std::array<unsigned char, hydra::kPageBytes + 64> registered{};
    auto *page = registered.data() + 24;  // Like the runtime's block header.
    for (size_t i = 0; i < hydra::kPageBytes; ++i)
        page[i] = static_cast<unsigned char>((i * 73 + i / 31) ^ 0xa7);
    assert(hydra::page_source_in_registered_range(page, registered.data(), registered.size()));
    assert(!hydra::page_source_in_registered_range(nullptr, registered.data(), registered.size()));
    assert(!hydra::page_source_in_registered_range(page, nullptr, registered.size()));
    assert(!hydra::page_source_in_registered_range(page, registered.data(), hydra::kPageBytes));
    assert(!hydra::page_source_in_registered_range(registered.data(), page, registered.size()));
    assert(hydra::page_data_fragment(page, 4) == nullptr);

    using Shards = std::array<std::array<unsigned char, hydra::kPageWriteFragmentBytes>, 6>;
    Shards copied{}, direct{};
    for (auto &shard : direct) shard.fill(0x6d);
    auto lease = [](Shards &shards) {
        cache::ec_batch::EcStagingGroupSlot value;
        value.index = 0;
        value.slot_size = hydra::kPageWriteFragmentBytes;
        for (size_t i = 0; i < 4; ++i) value.data[i] = shards[i].data();
        for (size_t i = 0; i < 2; ++i) value.parity[i] = shards[i + 4].data();
        return value;
    };
    const auto source_snapshot = registered;
    auto copy_lease = lease(copied);
    auto direct_lease = lease(direct);
    for (bool vector_codec : {false, true}) {
        assert(cache::ec_split::encode(page, hydra::kPageBytes, copy_lease, vector_codec));
        assert(hydra::encode_page_parity_only(page, direct_lease));
        for (size_t i = 0; i < 4; ++i) {
            assert(hydra::page_data_fragment(page, i) == page + i * 2048);
            for (auto byte : direct[i]) assert(byte == 0x6d); // No data staging copy.
        }
        assert(direct[4] == copied[4] && direct[5] == copied[5]);
        assert(registered == source_snapshot);
    }
    auto invalid = direct_lease;
    invalid.slot_size = 4096;
    assert(!hydra::encode_page_parity_only(page, invalid));

    // Deterministic stale-cleanup race: G1 drops its write ref to0, then G2
    // starts and an accessor rescues it to LOCAL. The old cleanup must refuse.
    cache::EntryStateBits stale{};
    stale.state = cache::LOCAL;
    stale.ref_cnt = 0;
    assert(hydra::page_write_cleanup_may_lock(stale));
    auto next = stale;
    next.state = cache::EVICTING;
    next.inc_ref_cnt();
    assert(!next.can_evict());
    next.state = cache::LOCAL;
    next.dirty = 1;
    assert(!next.can_evict());
    assert(!hydra::page_write_cleanup_may_lock(next));
    next.dec_ref_cnt();
    assert(hydra::page_write_cleanup_may_lock(next));
    next.invalid = 1;
    assert(!hydra::page_write_cleanup_may_lock(next));
    next.invalid = 0;
    next.state = cache::REMOTE;
    assert(!hydra::page_write_cleanup_may_lock(next));
    std::puts("HYDRA_PAGE_WRITE_PASS fixed-worker-parity parity-only direct-data MR-bounds stale-cleanup-guard");
}
