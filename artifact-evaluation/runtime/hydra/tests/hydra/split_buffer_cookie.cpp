// CPU-only validation of BufferPools' opaque lease-cookie boundary checks.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/alloc/ec_split_buffers.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::cache::ec_read_recovery::RecoveryScratchChunk;
using FarLib::cache::ec_split::BufferPools;

struct FakeContext {
    std::vector<void *> allocations;
    size_t release_count = 0;
    uint32_t next_lkey = 0x4000;
};

bool fake_allocate(void *opaque, size_t bytes, RecoveryScratchChunk *out) {
    auto &context = *static_cast<FakeContext *>(opaque);
    void *base = std::malloc(bytes);
    if (base == nullptr) return false;
    context.allocations.push_back(base);
    *out = {base, context.next_lkey++, nullptr};
    return true;
}

void fake_release(void *opaque, RecoveryScratchChunk chunk) {
    auto &context = *static_cast<FakeContext *>(opaque);
    auto it = std::find(context.allocations.begin(), context.allocations.end(),
                        chunk.base);
    assert(it != context.allocations.end());
    std::free(*it);
    context.allocations.erase(it);
    ++context.release_count;
}

void test_cookie_boundaries_and_lifecycle() {
    FakeContext context;
    BufferPools pools;
    assert(pools.init(&context, fake_allocate, fake_release));

    EcStagingGroupSlot lease;
    assert(pools.acquire(64, 7, &lease));
    assert(lease.pool_cookie != 0);
    assert(pools.owns_slot(lease));
    assert(pools.owns_buffer(lease, lease.data[0], lease.slot_size));

    const auto stale = lease;
    assert(pools.release(lease));
    assert(!pools.release(stale));
    assert(!pools.owns_slot(stale));
    assert(!pools.owns_buffer(stale, stale.data[0], stale.slot_size));

    EcStagingGroupSlot reused;
    assert(pools.acquire(64, 7, &reused));
    assert(reused.generation != stale.generation);

    // Zero, a misaligned address, and an address outside the cookie array must
    // all be rejected without dereferencing the supplied integer as a pointer.
    const uintptr_t cookie = reused.pool_cookie;
    for (const uintptr_t invalid : {
             uintptr_t{0}, cookie + 1,
             std::numeric_limits<uintptr_t>::max()}) {
        auto malformed = reused;
        malformed.pool_cookie = invalid;
        assert(!pools.release(malformed));
        assert(!pools.owns_slot(malformed));
        assert(!pools.owns_buffer(malformed, reused.data[0], 1));
    }

    BufferPools uninitialized;
    assert(!uninitialized.owns_slot(reused));
    assert(!uninitialized.release(reused));

    // A live lease from another BufferPools instance is foreign even when its
    // slot shape and owner index match.
    FakeContext foreign_context;
    BufferPools foreign;
    assert(foreign.init(&foreign_context, fake_allocate, fake_release));
    EcStagingGroupSlot foreign_lease;
    assert(foreign.acquire(64, 7, &foreign_lease));
    assert(!pools.owns_slot(foreign_lease));
    assert(!pools.release(foreign_lease));
    assert(foreign.release(foreign_lease));

    assert(pools.release(reused));
    assert(pools.in_use() == 0);
}

}  // namespace

int main() {
    test_cookie_boundaries_and_lifecycle();
    return 0;
}
