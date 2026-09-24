#ifdef NDEBUG
#undef NDEBUG
#endif

#include "hydra/write_batch.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

using FarLib::hydra::PendingWrite;
using FarLib::hydra::WriteBatch;
using FarLib::hydra::accepted_write_prefix;

PendingWrite request(std::uintptr_t address, uint64_t id) {
    PendingWrite value;
    value.local_addr = reinterpret_cast<void *>(address);
    value.remote_offset = 0x1000 + id * 64;
    value.wr_id = id;
    value.bytes = 4096;
    value.lkey = static_cast<uint32_t>(0x200 + id);
    return value;
}

void test_lifecycle_and_context() {
    WriteBatch batch;
    assert(!batch.initialized());
    assert(batch.endpoint_count() == 0);
    assert(batch.pending() == 0);
    assert(batch.client_idx() == WriteBatch::kUnboundContext);
    assert(batch.qp_idx() == WriteBatch::kUnboundContext);
    assert(!batch.bind_context(0, 0));
    assert(!batch.enqueue(0, request(0x1000, 1)));

    bool threw = false;
    try {
        (void)batch.endpoint(0);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        batch.init(0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    batch.init(2);
    assert(batch.initialized());
    assert(batch.endpoint_count() == 2);
    assert(batch.bind_context(3, 5));
    assert(batch.client_idx() == 3);
    assert(batch.qp_idx() == 5);
    assert(batch.bind_context(3, 5));

    assert(batch.enqueue(0, request(0x2000, 2)));
    assert(batch.pending() == 1);
    assert(!batch.bind_context(4, 6));
    assert(batch.client_idx() == 3);
    assert(batch.qp_idx() == 5);
    assert(batch.bind_context(3, 5));

    batch.clear_endpoint(0);
    assert(batch.pending() == 0);
    assert(batch.bind_context(4, 6));
    assert(batch.client_idx() == 4);
    assert(batch.qp_idx() == 6);

    threw = false;
    try {
        batch.init(1);
    } catch (const std::logic_error &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        batch.clear_endpoint(2);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);
    assert(!batch.enqueue(2, request(0x3000, 3)));
}

void test_endpoint_capacity_and_independence() {
    WriteBatch batch;
    batch.init(2);
    const PendingWrite first = request(0x4000, 10);
    assert(batch.enqueue(0, first));

    for (std::size_t i = 0; i < WriteBatch::kCapacity; ++i) {
        assert(batch.enqueue(1, request(0x5000 + i * 64, 100 + i)));
    }
    assert(batch.endpoint(0).count == 1);
    assert(batch.endpoint(1).count == WriteBatch::kCapacity);
    assert(batch.pending() == WriteBatch::kCapacity + 1);

    const PendingWrite last_before =
        batch.endpoint(1).requests[WriteBatch::kCapacity - 1];
    assert(!batch.enqueue(1, request(0x9000, 999)));
    assert(batch.endpoint(1).count == WriteBatch::kCapacity);
    const PendingWrite last_after =
        batch.endpoint(1).requests[WriteBatch::kCapacity - 1];
    assert(last_after.local_addr == last_before.local_addr);
    assert(last_after.wr_id == last_before.wr_id);
    assert(last_after.remote_offset == last_before.remote_offset);

    assert(!batch.enqueue(2, request(0xa000, 1000)));
    batch.clear_endpoint(1);
    assert(batch.endpoint(1).count == 0);
    assert(batch.pending() == 1);
    assert(batch.enqueue(1, request(0xb000, 1001)));
    assert(batch.pending() == 2);
    assert(batch.endpoint(0).requests[0].wr_id == first.wr_id);
}

void test_accepted_write_prefix() {
    std::array<PendingWrite, 5> requests{};
    const std::uintptr_t base =
        reinterpret_cast<std::uintptr_t>(requests.data());
    const std::size_t stride = sizeof(PendingWrite);
    const std::size_t count = requests.size();
    const std::size_t invalid = std::numeric_limits<std::size_t>::max();

    assert(accepted_write_prefix(base, stride, count, 0, 0, true) == count);
    assert(accepted_write_prefix(base, stride, count, 2,
                                 std::numeric_limits<std::uintptr_t>::max(),
                                 true) == 3);

    // Zero progress, an interior partial post, and errors at the last entry.
    assert(accepted_write_prefix(base, stride, count, 0, base, false) == 0);
    assert(accepted_write_prefix(base, stride, count, 0,
                                 base + 2 * stride, false) == 2);
    assert(accepted_write_prefix(base, stride, count, 2,
                                 base + 2 * stride, false) == 0);
    assert(accepted_write_prefix(base, stride, count, 2,
                                 base + 4 * stride, false) == 2);
    assert(accepted_write_prefix(base, stride, count, 0,
                                 base + 4 * stride, false) == 4);

    // The bad pointer must be an aligned element in the unaccepted suffix.
    assert(accepted_write_prefix(base, stride, count, 2,
                                 base + stride, false) == invalid);
    assert(accepted_write_prefix(base, stride, count, 0,
                                 base + 5 * stride, false) == invalid);
    assert(accepted_write_prefix(base, stride, count, 0,
                                 base - 1, false) == invalid);
    assert(accepted_write_prefix(base, stride, count, 0,
                                 base + 1, false) == invalid);
    assert(accepted_write_prefix(base, stride, count, 0,
                                 base + 7 * stride, false) == invalid);

    assert(accepted_write_prefix(base, stride, count, count, base, true) ==
           0);
    assert(accepted_write_prefix(base, stride, count, count, base, false) ==
           invalid);
    assert(accepted_write_prefix(base, stride, count, count + 1, base, true) ==
           invalid);
    assert(accepted_write_prefix(base, 0, count, 0, base, false) == invalid);
    assert(accepted_write_prefix(0, stride, count, 0, 0, false) == invalid);

    const std::uintptr_t address_max =
        std::numeric_limits<std::uintptr_t>::max();
    assert(accepted_write_prefix(address_max - 1, 2, 2, 0,
                                 address_max - 1, false) == invalid);
    assert(accepted_write_prefix(1, address_max / 2 + 1, 2, 0, 1, false) ==
           invalid);
}

}  // namespace

int main() {
    test_lifecycle_and_context();
    test_endpoint_capacity_and_independence();
    test_accepted_write_prefix();
    return 0;
}
