#include "hydra/page_layout.hpp"
#include <cassert>
#include <cstdio>

int main() {
    using namespace FarLib::hydra;
    assert(slot_bytes(0) == 0 && slot_bytes(8193) == 0);
    assert(capacity(0) == 0 && capacity(8193) == 0);
    for (size_t n = 1; n <= kPageBytes; ++n) {
        const auto slot = slot_bytes(n);
        assert(slot >= n && slot >= kMinSlotBytes && slot <= kPageBytes);
        assert((slot & (slot - 1)) == 0);
        assert(capacity(n) * slot == kPageBytes);
        assert((kMinSlotBytes << size_class(n)) == slot);
        for (size_t i = 0; i < capacity(n); ++i) {
            assert(i * slot + n <= kPageBytes);
        }
    }
    assert(capacity(512) == 16);
    assert(capacity(4096) == 2 && capacity(4097) == 1);
    assert(kDataShards * kShardBytes == kPageBytes);
    assert((kDataShards + kParityShards) * kShardBytes == 12288);
    std::puts("HYDRA_PAGE_LAYOUT_PASS sizes=1..8192 page=8192 stripe=4+2 shard=2048");
}
