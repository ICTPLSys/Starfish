#ifdef NDEBUG
#undef NDEBUG
#endif
#include "rdma/ec2pc_compact_transport.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <unordered_set>

namespace {
using FarLib::rdma::compact::CompactMoveDescriptor;
using FarLib::rdma::compact::compact_transport_state_shard_index;

struct UpdateLike {
    CompactMoveDescriptor descriptor{};
    bool submitted = false;
    bool rollback = false;
    bool done = false;
    std::chrono::steady_clock::time_point posted{};
};

static_assert(sizeof(UpdateLike) == 88);
static_assert(alignof(UpdateLike) == 8);

}  // namespace

int main() {
    constexpr size_t kUpdateCount = 512;
    const size_t bytes = sizeof(UpdateLike) * kUpdateCount;
    void *raw = std::aligned_alloc(16, bytes);
    assert(raw != nullptr);
    assert((reinterpret_cast<uintptr_t>(raw) & 15u) == 0);

    auto *updates = static_cast<UpdateLike *>(raw);
    std::unordered_set<size_t> old_shards;
    std::unordered_set<size_t> mixed_shards;
    for (size_t i = 0; i < kUpdateCount; ++i) {
        std::construct_at(&updates[i]);
        const uint64_t wr_id = reinterpret_cast<uint64_t>(&updates[i]);
        old_shards.insert(static_cast<size_t>(wr_id & 63u));
        mixed_shards.insert(compact_transport_state_shard_index(wr_id));
    }

    // A real aligned Update-like array with the current 88-byte stride has
    // gcd(88,64)=8, so direct low-bit masking can select only eight buckets.
    assert(old_shards.size() == 8);
    // The mixed index must exercise most of the 64-shard state table.
    assert(mixed_shards.size() >= 48);

    std::printf("CARBINK_REQUEST_SHARDS_PASS old=%zu mixed=%zu stride=%zu\n",
                old_shards.size(), mixed_shards.size(), sizeof(UpdateLike));

    for (size_t i = 0; i < kUpdateCount; ++i)
        std::destroy_at(&updates[i]);
    std::free(raw);
    return 0;
}
