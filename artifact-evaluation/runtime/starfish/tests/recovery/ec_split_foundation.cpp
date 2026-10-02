// CPU-only foundation checks for the large-object EC split layout and its
// size-classed registered scratch pools.  The remote allocator fixture is
// intentionally not constructed here; placement/reuse is covered by the
// allocator integration tests because it requires a configured remote heap.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/alloc/ec_split_buffers.hpp"
#include "cache/alloc/ec_split_layout.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::cache::ec_read_recovery::RecoveryScratchChunk;
using FarLib::cache::ec_split::BufferPools;

struct Registration {
    uint32_t lkey = 0;
};

struct Context {
    std::mutex mutex;
    std::vector<std::pair<void *, Registration *>> allocations;
    uint32_t next_lkey = 0x4000;
};

bool allocate_chunk(void *opaque, size_t bytes, RecoveryScratchChunk *out) {
    auto &context = *static_cast<Context *>(opaque);
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
    auto &context = *static_cast<Context *>(opaque);
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
    assert(false && "unknown registered chunk");
}

size_t rounded_slot_size(size_t bytes) {
    // The codec accepts a bin-rounded slot.  The real BufferPools test below
    // exercises allocator-bin rounding; keeping this CPU codec fixture at the
    // exact logical fragment size avoids pulling RDMA allocator headers into a
    // standalone unit test.
    return FarLib::cache::ec_split::fragment_size(bytes);
}

EcStagingGroupSlot slot_for(std::vector<uint8_t> &storage, size_t slot_size) {
    EcStagingGroupSlot slot;
    slot.slot_size = static_cast<uint32_t>(slot_size);
    for (size_t i = 0; i < FarLib::cache::ec_split::kDataSlots; ++i) {
        slot.data[i] = storage.data() + i * slot_size;
    }
    for (size_t i = 0; i < FarLib::cache::ec_split::kParitySlots; ++i) {
        slot.parity[i] = storage.data() +
                         (FarLib::cache::ec_split::kDataSlots + i) * slot_size;
    }
    return slot;
}

void test_all_survivor_sets(size_t bytes) {
    std::vector<uint8_t> object(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        object[i] = static_cast<uint8_t>((i * 131u + bytes * 17u) & 0xffu);
    }
    const size_t slot_size = rounded_slot_size(bytes);
    std::vector<uint8_t> encoded(
        FarLib::cache::ec_split::kSegments * slot_size, 0xa5);
    auto source = slot_for(encoded, slot_size);
    assert(FarLib::cache::ec_split::encode(object.data(), bytes, source));

    const size_t logical = FarLib::cache::ec_split::fragment_size(bytes);
    for (size_t data = 0; data < 4; ++data) {
        const size_t offset = data * logical;
        const size_t count = offset < bytes ?
                                 std::min(logical, bytes - offset) :
                                 size_t{0};
        for (size_t i = count; i < slot_size; ++i) {
            assert(static_cast<uint8_t *>(source.data[data])[i] == 0);
        }
    }

    // C(6,4) exact-survivor masks cover all single and double failure sets.
    for (uint8_t selected = 0; selected < (1u << 6); ++selected) {
        if (__builtin_popcount(static_cast<unsigned>(selected)) != 4) {
            continue;
        }
        std::vector<uint8_t> recovered = encoded;
        auto scratch = slot_for(recovered, slot_size);
        for (size_t segment = 0; segment < 6; ++segment) {
            if ((selected & static_cast<uint8_t>(1u << segment)) == 0) {
                void *ptr = segment < 4 ? scratch.data[segment]
                                        : scratch.parity[segment - 4];
                std::memset(ptr, 0xcd, slot_size);
            }
        }
        std::vector<uint8_t> output(bytes, 0);
        assert(FarLib::cache::ec_split::reconstruct(
            selected, scratch, output.data(), bytes));
        assert(output == object);
    }
}

void test_direct_read_fragment_ranges_and_canaries() {
    constexpr size_t kCanaryBytes = 32;
    constexpr uint8_t kCanary = 0xd3;
    const std::array<size_t, 6> sizes = {
        4097, 8168, 8169, 8192, 262143, 262144};
    for (const size_t bytes : sizes) {
        std::vector<uint8_t> source(bytes);
        for (size_t i = 0; i < bytes; ++i) {
            source[i] = static_cast<uint8_t>(0x4bu + 0x23u * i + (i >> 5));
        }
        std::vector<uint8_t> guarded(kCanaryBytes + bytes + kCanaryBytes,
                                     kCanary);
        auto *payload = guarded.data() + kCanaryBytes;
        std::memset(payload, 0xa6, bytes);

        size_t covered = 0;
        for (size_t fragment = 0;
             fragment < FarLib::cache::ec_split::kDataSlots; ++fragment) {
            FarLib::cache::ec_split::DirectReadFragment range;
            assert(FarLib::cache::ec_split::direct_read_fragment(
                bytes, fragment, &range));
            assert(range.bytes != 0);
            assert(range.offset + range.bytes <= bytes);
            if (bytes == 8169 && fragment == 3) {
                assert(range.offset == 6129 && range.bytes == 2040);
            }
            // Model the four accepted RDMA READs writing their exact logical
            // ranges into the final object; no padded tail is touched.
            std::memcpy(payload + range.offset, source.data() + range.offset,
                        range.bytes);
            covered += range.bytes;
        }
        assert(covered == bytes);
        assert(std::memcmp(payload, source.data(), bytes) == 0);
        for (size_t i = 0; i < kCanaryBytes; ++i) {
            assert(guarded[i] == kCanary);
            assert(guarded[kCanaryBytes + bytes + i] == kCanary);
        }
    }

    FarLib::cache::ec_split::DirectReadFragment untouched{17, 19};
    assert(!FarLib::cache::ec_split::direct_read_fragment(0, 0, &untouched));
    assert(!FarLib::cache::ec_split::direct_read_fragment(
        FarLib::cache::ec_split::kMaxObjectBytes + 1, 0, &untouched));
    assert(!FarLib::cache::ec_split::direct_read_fragment(4097, 4, &untouched));
    assert(!FarLib::cache::ec_split::direct_read_fragment(4097, 0, nullptr));

    std::array<uint8_t, 128> region{};
    assert(FarLib::cache::ec_split::address_range_within(
        region.data() + 8, 64, region.data(), region.size()));
    assert(!FarLib::cache::ec_split::address_range_within(
        region.data() + 8, 121, region.data(), region.size()));
    assert(!FarLib::cache::ec_split::address_range_within(
        region.data(), 0, region.data(), region.size()));
    constexpr uintptr_t kMax = std::numeric_limits<uintptr_t>::max();
    assert(!FarLib::cache::ec_split::address_range_within(
        reinterpret_cast<const void *>(kMax - 3), 8, region.data(),
        region.size()));
    assert(!FarLib::cache::ec_split::address_range_within(
        region.data(), 8, reinterpret_cast<const void *>(kMax - 3), 8));
}

void test_buffer_pools() {
    Context context;
    BufferPools pools;
    assert(pools.init(&context, allocate_chunk, release_chunk));
    assert(pools.valid());

    EcStagingGroupSlot lease;
    assert(pools.acquire(4097, 3, &lease));
    assert(lease.slot_size >= 4097);
    assert(pools.in_use() == 1);
    assert(pools.bytes() != 0);
    assert(pools.growths() != 0);
    assert(pools.owns_slot(lease));
    assert(pools.owns_buffer(lease, lease.data[0], lease.slot_size));
    assert(pools.owns_buffer(lease, lease.parity[1], lease.slot_size));
    uint8_t foreign_byte = 0;
    assert(!pools.owns_buffer(lease, &foreign_byte, 1));

    auto tampered = lease;
    tampered.pool_cookie ^= 1u;
    assert(!pools.owns_slot(tampered));
    assert(!pools.release(tampered));
    assert(pools.release(lease));
    assert(!pools.release(lease));
    assert(!pools.owns_slot(lease));

    // A distinct size class gets its own worker-local pool and lazy chunk.
    EcStagingGroupSlot large;
    assert(pools.acquire(65536, 7, &large));
    assert(large.slot_size == 65536);
    assert(pools.release(large));
    assert(pools.in_use() == 0);
}

}  // namespace

int main() {
    for (const size_t bytes : {size_t{4096}, size_t{4097}, size_t{16384},
                               size_t{20016}, size_t{65537},
                               size_t{256 * 1024}}) {
        test_all_survivor_sets(bytes);
    }
    test_direct_read_fragment_ranges_and_canaries();
    test_buffer_pools();
    std::cout << "EC_SPLIT_FOUNDATION_PASS\n";
    return 0;
}
