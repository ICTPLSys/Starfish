#if __has_include(<infiniband/verbs.h>)
#include "hydra/write_ring.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <latch>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using FarLib::cache::ec_batch::EcGroupSendRecord;
using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::hydra::WriteRing;
using RegisteredParity = WriteRing::RegisteredParity;

struct AllocationState {
    size_t fail_at = std::numeric_limits<size_t>::max();
    size_t allocate_calls = 0;
    size_t free_calls = 0;
    std::vector<void *> live_bases;
};

bool allocate_parity(void *context, size_t bytes, RegisteredParity *out) {
    auto *state = static_cast<AllocationState *>(context);
    assert(out != nullptr);
    *out = RegisteredParity{};
    const size_t call = state->allocate_calls++;
    if (call == state->fail_at) return false;
    auto *memory = new uint8_t[bytes];
    state->live_bases.push_back(memory);
    out->base = memory;
    out->lkey = static_cast<uint32_t>(0x100 + call);
    out->registration = memory;
    return true;
}

void free_parity(void *context, RegisteredParity value) {
    auto *state = static_cast<AllocationState *>(context);
    assert(value.base != nullptr);
    const auto it = std::find(state->live_bases.begin(),
                              state->live_bases.end(), value.base);
    assert(it != state->live_bases.end());
    delete[] static_cast<uint8_t *>(value.base);
    state->live_bases.erase(it);
    ++state->free_calls;
}

void finish(WriteRing &ring, uint64_t token, uint8_t failed_mask = 0) {
    WriteRing::Entry *winner = nullptr;
    for (uint8_t segment = 0; segment < 6; ++segment) {
        WriteRing::Entry *candidate = ring.complete_segment(
            token, segment, (failed_mask & (uint8_t{1} << segment)) == 0);
        if (candidate != nullptr) {
            assert(winner == nullptr);
            winner = candidate;
        }
    }
    assert(winner != nullptr);
    assert(winner->pending.load() == 0);
    assert(ring.release(token));
}

void test_owner_capacity_and_reuse() {
    WriteRing ring(2, 2);
    EcGroupSendRecord record{};
    std::array<uint64_t, 2> ids{};
    for (size_t i = 0; i < 2; ++i) {
        WriteRing::Entry *entry = nullptr;
        assert(ring.acquire(0, record, &ids[i], &entry));
        assert(entry != nullptr);
    }
    assert(ring.available(0) == 0);
    assert(ring.available(1) == 2);
    uint64_t blocked = 0;
    WriteRing::Entry *entry = nullptr;
    assert(!ring.acquire(0, record, &blocked, &entry));
    assert(!ring.release(ids[0]));
    assert(ring.complete_segment(ids[0], 0) == nullptr);
    assert(ring.complete_segment(ids[0], 0) == nullptr);
    finish(ring, ids[0]);
    assert(ring.available(0) == 1);
    uint64_t reused = 0;
    assert(ring.acquire(0, record, &reused, &entry));
    assert(reused != ids[0]);
    assert(WriteRing::is_token(reused));
    assert(ring.complete_segment(ids[0], 0) == nullptr);
    finish(ring, ids[1]);
    finish(ring, reused);
    assert(ring.in_use() == 0);
}

void test_parity_initialization_and_rollback() {
    AllocationState state;
    {
        WriteRing ring(3, 2);
        assert(!ring.parity_initialized());
        assert(ring.parity_bytes() == 0);
        assert(!ring.init_parity_buffers(&state, nullptr, free_parity));

        state.fail_at = 1;
        assert(!ring.init_parity_buffers(&state, allocate_parity,
                                         free_parity));
        assert(!ring.parity_initialized());
        assert(state.allocate_calls == 2);
        assert(state.free_calls == 1);
        assert(state.live_bases.empty());

        state.fail_at = std::numeric_limits<size_t>::max();
        assert(ring.init_parity_buffers(&state, allocate_parity,
                                        free_parity));
        assert(ring.parity_initialized());
        assert(ring.parity_bytes() == 3 * 2 * WriteRing::kParitySlotBytes);
        assert(!ring.init_parity_buffers(&state, allocate_parity,
                                         free_parity));
    }
    assert(state.free_calls == 4);
    assert(state.live_bases.empty());
}

void test_direct_page_reservation_and_reuse() {
    AllocationState state;
    {
        WriteRing ring(2, 1);
        assert(ring.init_parity_buffers(&state, allocate_parity,
                                        free_parity));

        EcGroupSendRecord record{};
        uint64_t token = 0;
        WriteRing::Entry *entry = nullptr;
        EcStagingGroupSlot scratch{};
        assert(ring.reserve_page(0, &token, &entry, &scratch));
        assert(entry != nullptr);
        assert(WriteRing::is_token(token));
        assert(entry->pending.load() == 0);
        assert(scratch.slot_size == WriteRing::kParitySegmentBytes);
        assert(scratch.index == 0);
        assert(scratch.generation != 0);
        assert(scratch.parity[0] != nullptr);
        assert(scratch.parity[1] != nullptr);
        assert(static_cast<uint8_t *>(scratch.parity[1]) ==
               static_cast<uint8_t *>(scratch.parity[0]) +
                   WriteRing::kParitySegmentBytes);
        for (void *data : scratch.data) assert(data == nullptr);
        assert(ring.owns_parity(token, scratch));
        assert(ring.get(token) == nullptr);

        // CQEs arriving before publish must be ignored, and a foreign
        // staging descriptor cannot be published into this reservation.
        for (uint8_t segment = 0; segment < 6; ++segment) {
            assert(ring.complete_segment(token, segment) == nullptr);
        }
        assert(!ring.release(token));
        EcStagingGroupSlot foreign = scratch;
        foreign.parity[0] = reinterpret_cast<void *>(0x1234);
        assert(!ring.owns_parity(token, foreign));
        record.staging = foreign;
        assert(!ring.publish_page(token, record));
        assert(ring.owns_parity(token, scratch));

        record.staging = scratch;
        record.direct_page_data = true;
        assert(ring.publish_page(token, record));
        assert(entry->pending.load() == 6);
        assert(entry->record.staging.parity[0] == scratch.parity[0]);
        assert(ring.get(token) == entry);
        assert(ring.owns_parity(token, scratch));
        assert(!ring.cancel_page(token));
        finish(ring, token);
        assert(!ring.owns_parity(token, scratch));
        assert(!ring.publish_page(token, record));

        // The same fixed slot address is reused only with a new generation.
        uint64_t reused_token = 0;
        WriteRing::Entry *reused_entry = nullptr;
        EcStagingGroupSlot reused_scratch{};
        assert(ring.reserve_page(0, &reused_token, &reused_entry,
                                 &reused_scratch));
        assert(reused_scratch.parity[0] == scratch.parity[0]);
        assert(reused_scratch.parity[1] == scratch.parity[1]);
        assert(reused_scratch.generation != scratch.generation);
        assert(ring.complete_segment(token, 0) == nullptr);
        assert(ring.cancel_page(reused_token));
        assert(!ring.cancel_page(reused_token));
    }
    assert(state.live_bases.empty());
}

void test_direct_page_backpressure_and_owner_isolation() {
    AllocationState state;
    {
        WriteRing ring(2, 2);
        assert(ring.init_parity_buffers(&state, allocate_parity,
                                        free_parity));
        std::array<uint64_t, 4> tokens{};
        std::array<WriteRing::Entry *, 4> entries{};
        std::array<EcStagingGroupSlot, 4> scratches{};

        for (size_t i = 0; i < 2; ++i) {
            assert(ring.reserve_page(0, &tokens[i], &entries[i],
                                     &scratches[i]));
        }
        assert(ring.available(0) == 0);
        assert(!ring.reserve_page(0, &tokens[0], &entries[0],
                                  &scratches[0]));

        for (size_t i = 2; i < 4; ++i) {
            assert(ring.reserve_page(1, &tokens[i], &entries[i],
                                     &scratches[i]));
        }
        assert(ring.available(1) == 0);
        assert(ring.in_use() == 4);
        assert(scratches[0].parity[0] != scratches[2].parity[0]);
        assert(!ring.owns_parity(tokens[0], scratches[2]));

        for (uint64_t token : tokens) assert(ring.cancel_page(token));
        assert(ring.in_use() == 0);
        assert(ring.available(0) == 2);
        assert(ring.available(1) == 2);
    }
    assert(state.live_bases.empty());
}

void test_failure_masks() {
    EcGroupSendRecord record{};
    for (uint8_t failed_mask : {uint8_t{0}, uint8_t{1}, uint8_t{3},
                                uint8_t{7}}) {
        WriteRing ring(1, 1);
        uint64_t token = 0;
        WriteRing::Entry *entry = nullptr;
        assert(ring.acquire(0, record, &token, &entry));
        finish(ring, token, failed_mask);
        // The entry is still address-stable after release, and the finalizer
        // fields were populated before release.
        assert(entry->durable_segment_count() ==
               (6u - static_cast<unsigned>(__builtin_popcount(failed_mask))));
        assert(entry->recoverable() ==
               (__builtin_popcount(failed_mask) <= 2));
    }
}

void test_concurrent_last_completion() {
    AllocationState allocation;
    WriteRing ring(1, 1);
    assert(ring.init_parity_buffers(&allocation, allocate_parity, free_parity));
    EcGroupSendRecord record{};
    uint64_t token = 0;
    WriteRing::Entry *entry = nullptr;
    assert(ring.reserve_page(0, &token, &entry, &record.staging));
    record.direct_page_data = true;
    assert(ring.publish_page(token, record));
    std::array<WriteRing::Entry *, 6> winners{};
    std::vector<std::thread> workers;
    workers.reserve(6);
    for (uint8_t segment = 0; segment < 6; ++segment) {
        workers.emplace_back([&, segment] {
            winners[segment] =
                ring.complete_segment(token, segment, true);
        });
    }
    for (auto &worker : workers) worker.join();
    size_t winner_count = 0;
    for (auto *winner : winners) winner_count += winner != nullptr;
    assert(winner_count == 1);
    assert(ring.release(token));
}

void test_callbacks_race_reuse() {
    AllocationState allocation;
    WriteRing ring(1, 1);
    assert(ring.init_parity_buffers(&allocation, allocate_parity, free_parity));
    EcGroupSendRecord record{};
    uint64_t old_token = 0;
    WriteRing::Entry *entry = nullptr;
    assert(ring.reserve_page(0, &old_token, &entry, &record.staging));
    record.direct_page_data = true;
    assert(ring.publish_page(old_token, record));

    std::latch ready{5};
    std::latch let_return{1};
    std::vector<std::thread> callbacks;
    callbacks.reserve(5);
    for (uint8_t segment = 0; segment < 5; ++segment) {
        callbacks.emplace_back([&, segment] {
            assert(ring.complete_segment(old_token, segment) == nullptr);
            ready.count_down();
            let_return.wait();
            // A duplicate/late callback must not touch a reused generation.
            assert(ring.complete_segment(old_token, segment) == nullptr);
        });
    }
    ready.wait();
    assert(ring.complete_segment(old_token, 5) != nullptr);
    assert(ring.release(old_token));

    uint64_t new_token = 0;
    assert(ring.reserve_page(0, &new_token, &entry, &record.staging));
    assert(ring.publish_page(new_token, record));
    assert(new_token != old_token);
    for (uint8_t segment = 0; segment < 6; ++segment) {
        assert(ring.complete_segment(old_token, segment) == nullptr);
    }
    finish(ring, new_token);
    let_return.count_down();
    for (auto &callback : callbacks) callback.join();
}

}  // namespace

int main() {
    test_owner_capacity_and_reuse();
    test_parity_initialization_and_rollback();
    test_direct_page_reservation_and_reuse();
    test_direct_page_backpressure_and_owner_isolation();
    test_failure_masks();
    test_concurrent_last_completion();
    test_callbacks_race_reuse();
    return 0;
}
#else
int main() { return 0; }
#endif
