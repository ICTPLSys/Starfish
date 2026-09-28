#include <algorithm>
#include "hydra/write_ring.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
using FarLib::cache::ec_batch::EcGroupSendRecord;
using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::hydra::RegisteredParity;
using FarLib::hydra::WriteRing;

struct AllocationState {
    size_t allocate_calls = 0;
    size_t free_calls = 0;
    std::vector<void *> live_bases;
};

bool allocate_parity(void *context, size_t bytes, RegisteredParity *out) {
    auto *state = static_cast<AllocationState *>(context);
    assert(out != nullptr);
    *out = RegisteredParity{};
    auto *memory = new uint8_t[bytes];
    state->live_bases.push_back(memory);
    out->base = memory;
    out->lkey = static_cast<uint32_t>(0x700 + state->allocate_calls++);
    out->registration = memory;
    return true;
}

void free_parity(void *context, RegisteredParity value) {
    auto *state = static_cast<AllocationState *>(context);
    assert(value.base != nullptr);
    auto it = std::find(state->live_bases.begin(), state->live_bases.end(),
                        value.base);
    assert(it != state->live_bases.end());
    delete[] static_cast<uint8_t *>(value.base);
    state->live_bases.erase(it);
    ++state->free_calls;
}

struct Page {
    uint64_t token = 0;
    WriteRing::Entry *entry = nullptr;
    EcStagingGroupSlot scratch{};
};

void reserve(WriteRing &ring, size_t owner, Page &page) {
    assert(ring.reserve_page(owner, &page.token, &page.entry, &page.scratch));
    assert(page.entry != nullptr);
    assert(page.scratch.slot_size == 8192);
    assert(ring.owns_parity(page.token, page.scratch));
}

void publish(WriteRing &ring, const Page &page) {
    EcGroupSendRecord record{};
    record.direct_span_data = true;
    record.staging = page.scratch;
    assert(ring.publish_page(page.token, record));
}

void finish(WriteRing &ring, const Page &page,
            std::array<uint8_t, 6> order = {5, 1, 3, 0, 4, 2}) {
    WriteRing::Entry *winner = nullptr;
    for (uint8_t segment : order) {
        WriteRing::Entry *candidate =
            ring.complete_segment(page.token, segment, true);
        assert(candidate == nullptr || winner == nullptr);
        if (candidate != nullptr) winner = candidate;
        if (segment == order[0]) assert(ring.complete_segment(page.token, segment, true) == nullptr);
    }
    assert(winner == page.entry);
    assert(ring.release(page.token));
    assert(!ring.release(page.token));
    assert(ring.complete_segment(page.token, order[0]) == nullptr);
}

void cancel(WriteRing &ring, const Page &page) {
    assert(ring.cancel_page(page.token));
    assert(!ring.cancel_page(page.token));
}

void test_validation() {
    bool threw = false;
    try {
        WriteRing ring(1, 128, true, 4096, 64);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        WriteRing ring(1, 127, true, 8192, 64);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        WriteRing ring(1, 128, true, 8192, 0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

void test_carbink_banked_ring() {
    AllocationState state;
    {
        WriteRing ring(2, 128, true, 8192, 64);
        assert(ring.banked());
        assert(ring.owner_count() == 2);
        assert(ring.slots_per_owner() == 128);
        assert(ring.parity_segment_bytes() == 8192);
        assert(ring.groups_per_bank() == 64);
        assert(ring.init_parity_buffers(&state, allocate_parity, free_parity));
        assert(ring.parity_bytes() == 2 * 128 * 2 * 8192);

        std::array<Page, 128> pages{};
        for (size_t i = 0; i < pages.size(); ++i) {
            reserve(ring, 0, pages[i]);
            publish(ring, pages[i]);
            assert(pages[i].scratch.index == i);
            assert(pages[i].scratch.parity[1] ==
                   static_cast<uint8_t *>(pages[i].scratch.parity[0]) + 8192);
            assert(ring.buffer_end(pages[i].token) ==
                   (i == 63 || i == 127));
        }
        assert(!ring.buffer_end(pages[62].token));
        assert(!ring.buffer_end(pages[64].token));

        auto *owner0_slot0 =
            static_cast<uint8_t *>(pages[0].scratch.parity[0]);
        auto *owner0_slot1 =
            static_cast<uint8_t *>(pages[1].scratch.parity[0]);
        auto *owner0_slot64 =
            static_cast<uint8_t *>(pages[64].scratch.parity[0]);
        assert(owner0_slot1 == owner0_slot0 + 2 * 8192);
        assert(owner0_slot64 == owner0_slot0 + 64 * 2 * 8192);

        Page owner1;
        reserve(ring, 1, owner1);
        publish(ring, owner1);
        assert(owner1.scratch.index == 0);
        assert(owner1.scratch.parity[0] != pages[0].scratch.parity[0]);
        finish(ring, owner1);

        // A bank can be reused only after its last outstanding page finishes.
        for (size_t i = 0; i < 63; ++i) finish(ring, pages[i]);
        for (size_t i = 64; i < pages.size(); ++i) finish(ring, pages[i]);
        Page blocked;
        assert(!ring.reserve_page(0, &blocked.token, &blocked.entry,
                                  &blocked.scratch));
        assert(ring.available(0) == 127);

        finish(ring, pages[63]);
        Page reused;
        reserve(ring, 0, reused);
        assert(reused.scratch.index == 0);
        assert(reused.token != pages[0].token);
        publish(ring, reused);
        finish(ring, reused);
        assert(ring.in_use() == 0);
    }
    assert(state.free_calls == 2);
    assert(state.live_bases.empty());
}

}  // namespace

int main() {
    test_validation();
    test_carbink_banked_ring();
    return 0;
}
