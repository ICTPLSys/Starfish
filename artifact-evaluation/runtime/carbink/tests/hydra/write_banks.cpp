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
#include <thread>
#include <vector>

namespace {

using FarLib::cache::ec_batch::EcGroupSendRecord;
using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::hydra::WriteRing;
using RegisteredParity = WriteRing::RegisteredParity;

static_assert(WriteRing::kPagesPerBuffer == 32);
static_assert(WriteRing::kBuffersPerOwner == 2);
static_assert(WriteRing::kDoubleBufferedSlots == 64);

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
    out->lkey = static_cast<uint32_t>(0x200 + state->allocate_calls++);
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

struct Page {
    uint64_t token = 0;
    WriteRing::Entry *entry = nullptr;
    EcStagingGroupSlot scratch{};
};

void init(WriteRing &ring, AllocationState &state) {
    assert(ring.banked());
    assert(ring.init_parity_buffers(&state, allocate_parity, free_parity));
    assert(ring.parity_bytes() ==
           ring.capacity() * WriteRing::kParitySlotBytes);
}

void reserve(WriteRing &ring, size_t owner, Page &page) {
    assert(ring.reserve_page(owner, &page.token, &page.entry,
                             &page.scratch));
    assert(page.entry != nullptr);
    assert(page.scratch.index < WriteRing::kDoubleBufferedSlots);
}

void publish(WriteRing &ring, const Page &page) {
    EcGroupSendRecord record{};
    record.direct_page_data = true;
    record.staging = page.scratch;
    assert(ring.publish_page(page.token, record));
}

void finish(WriteRing &ring, const Page &page, uint8_t failed_mask = 0) {
    WriteRing::Entry *winner = nullptr;
    for (uint8_t segment = 0; segment < 6; ++segment) {
        auto *candidate = ring.complete_segment(
            page.token, segment,
            (failed_mask & (uint8_t{1} << segment)) == 0);
        if (candidate != nullptr) {
            assert(winner == nullptr);
            winner = candidate;
        }
    }
    assert(winner != nullptr);
    assert(ring.release(page.token));
    assert(!ring.release(page.token));
}

void cancel(WriteRing &ring, const Page &page) {
    assert(ring.cancel_page(page.token));
    assert(!ring.cancel_page(page.token));
}

void test_initial_banks_and_boundaries() {
    AllocationState state;
    {
        WriteRing ring(1, WriteRing::kDoubleBufferedSlots, true);
        init(ring, state);
        std::array<Page, WriteRing::kDoubleBufferedSlots> pages{};
        for (size_t i = 0; i < pages.size(); ++i) {
            reserve(ring, 0, pages[i]);
            assert(pages[i].scratch.index == i);
            assert(ring.buffer_end(pages[i].token) ==
                   (i == WriteRing::kPagesPerBuffer - 1 ||
                    i == WriteRing::kDoubleBufferedSlots - 1));
        }
        Page blocked;
        assert(!ring.reserve_page(0, &blocked.token, &blocked.entry,
                                  &blocked.scratch));
        assert(ring.in_use() == WriteRing::kDoubleBufferedSlots);
        assert(ring.available(0) == 0);
        for (const Page &page : pages) cancel(ring, page);
        assert(ring.in_use() == 0);
        assert(ring.available(0) == WriteRing::kDoubleBufferedSlots);
    }
    assert(state.free_calls == 1);
    assert(state.live_bases.empty());
}

void test_whole_bank_reuse_and_no_skip() {
    AllocationState state;
    {
        WriteRing ring(1, WriteRing::kDoubleBufferedSlots, true);
        init(ring, state);
        std::array<Page, WriteRing::kDoubleBufferedSlots> pages{};
        for (Page &page : pages) {
            reserve(ring, 0, page);
            publish(ring, page);
        }

        // B becoming completely free does not permit skipping an unfinished A.
        finish(ring, pages[0]);
        for (size_t i = WriteRing::kPagesPerBuffer;
             i < WriteRing::kDoubleBufferedSlots; ++i) {
            finish(ring, pages[i]);
        }
        assert(ring.available(0) == WriteRing::kPagesPerBuffer + 1);
        Page blocked;
        assert(!ring.reserve_page(0, &blocked.token, &blocked.entry,
                                  &blocked.scratch));

        // Once every A page is terminal and released, the cursor may wrap to A.
        for (size_t i = 1; i < WriteRing::kPagesPerBuffer; ++i) {
            finish(ring, pages[i]);
        }
        Page reused;
        reserve(ring, 0, reused);
        assert(reused.scratch.index == 0);
        assert(reused.token != pages[0].token);
        cancel(ring, reused);
        assert(ring.in_use() == 0);
    }
    assert(state.free_calls == 1);
    assert(state.live_bases.empty());
}

void test_partial_bank_cursor_and_cancel_position() {
    AllocationState state;
    {
        WriteRing ring(1, WriteRing::kDoubleBufferedSlots, true);
        init(ring, state);

        // Releasing a partially filled bank does not reset its producer cursor.
        std::array<Page, 5> first{};
        for (Page &page : first) {
            reserve(ring, 0, page);
            publish(ring, page);
        }
        for (const Page &page : first) finish(ring, page);

        std::array<Page, 3> second{};
        for (size_t i = 0; i < second.size(); ++i) {
            reserve(ring, 0, second[i]);
            assert(second[i].scratch.index == 5 + i);
            publish(ring, second[i]);
        }
        for (const Page &page : second) finish(ring, page);

        // Cancellation advances the cursor but decrements its bank exactly once.
        Page canceled0;
        Page canceled1;
        reserve(ring, 0, canceled0);
        reserve(ring, 0, canceled1);
        assert(canceled0.scratch.index == 8);
        assert(canceled1.scratch.index == 9);
        assert(ring.available(0) == 62);
        cancel(ring, canceled0);
        assert(ring.available(0) == 63);
        assert(ring.available(0) == 63);
        cancel(ring, canceled1);
        assert(ring.available(0) == 64);

        Page tail;
        reserve(ring, 0, tail);
        assert(tail.scratch.index == 10);
        cancel(ring, tail);
        assert(ring.in_use() == 0);
    }
    assert(state.free_calls == 1);
    assert(state.live_bases.empty());
}

void test_owner_isolation() {
    AllocationState state;
    {
        WriteRing ring(2, WriteRing::kDoubleBufferedSlots, true);
        init(ring, state);
        std::array<Page, WriteRing::kDoubleBufferedSlots> owner_zero{};
        for (Page &page : owner_zero) reserve(ring, 0, page);
        assert(ring.available(0) == 0);

        Page owner_one;
        reserve(ring, 1, owner_one);
        assert(owner_one.scratch.index == 0);
        assert(owner_zero[0].scratch.parity[0] != owner_one.scratch.parity[0]);
        assert(ring.available(1) == WriteRing::kDoubleBufferedSlots - 1);

        cancel(ring, owner_one);
        for (const Page &page : owner_zero) cancel(ring, page);
        assert(ring.available(0) == WriteRing::kDoubleBufferedSlots);
        assert(ring.available(1) == WriteRing::kDoubleBufferedSlots);
    }
    assert(state.free_calls == 2);
    assert(state.live_bases.empty());
}

void test_duplicate_failure_and_no_early_release() {
    AllocationState state;
    {
        WriteRing ring(1, WriteRing::kDoubleBufferedSlots, true);
        init(ring, state);
        Page page;
        reserve(ring, 0, page);
        publish(ring, page);
        assert(ring.complete_segment(page.token, 0, true) == nullptr);
        assert(ring.complete_segment(page.token, 0, true) == nullptr);
        assert(!ring.release(page.token));
        for (uint8_t segment = 1; segment < 6; ++segment) {
            const bool success = segment >= 4;
            assert(ring.complete_segment(page.token, segment, success) ==
                   (segment == 5 ? page.entry : nullptr));
        }
        assert(page.entry->durable_segment_count() == 3);
        assert(!page.entry->recoverable());
        assert(ring.release(page.token));
        assert(!ring.release(page.token));
        for (uint8_t segment = 0; segment < 6; ++segment)
            assert(ring.complete_segment(page.token, segment) == nullptr);
        assert(ring.in_use() == 0);
    }
    assert(state.free_calls == 1);
    assert(state.live_bases.empty());
}

void test_parallel_finalizer_and_stale_reuse() {
    AllocationState state;
    {
        WriteRing ring(1, WriteRing::kDoubleBufferedSlots, true);
        init(ring, state);
        std::array<Page, WriteRing::kDoubleBufferedSlots> pages{};
        for (size_t i = 0; i < pages.size(); ++i) {
            reserve(ring, 0, pages[i]);
            if (i == 0) {
                publish(ring, pages[i]);
            } else {
                cancel(ring, pages[i]);
            }
        }

        std::array<WriteRing::Entry *, 6> winners{};
        std::latch ready{6};
        std::latch start{1};
        std::vector<std::thread> completions;
        completions.reserve(6);
        for (uint8_t segment = 0; segment < 6; ++segment) {
            completions.emplace_back([&, segment] {
                ready.count_down();
                start.wait();
                winners[segment] =
                    ring.complete_segment(pages[0].token, segment, true);
            });
        }
        ready.wait();
        start.count_down();
        for (auto &thread : completions) thread.join();
        size_t winner_count = 0;
        for (auto *winner : winners) winner_count += winner != nullptr;
        assert(winner_count == 1);

        std::latch stale_ready{5};
        std::latch permit_late{1};
        std::vector<std::thread> stale;
        stale.reserve(5);
        for (uint8_t segment = 0; segment < 5; ++segment) {
            stale.emplace_back([&, segment] {
                assert(ring.complete_segment(pages[0].token, segment) == nullptr);
                stale_ready.count_down();
                permit_late.wait();
                assert(ring.complete_segment(pages[0].token, segment) == nullptr);
            });
        }
        stale_ready.wait();
        assert(ring.release(pages[0].token));

        Page reused;
        reserve(ring, 0, reused);
        assert(reused.scratch.index == 0);
        assert(reused.token != pages[0].token);
        publish(ring, reused);
        permit_late.count_down();
        for (auto &thread : stale) thread.join();
        for (uint8_t segment = 0; segment < 6; ++segment)
            assert(ring.complete_segment(pages[0].token, segment) == nullptr);
        finish(ring, reused);
        assert(ring.in_use() == 0);
    }
    assert(state.free_calls == 1);
    assert(state.live_bases.empty());
}

}  // namespace

int main() {
    test_initial_banks_and_boundaries();
    test_whole_bank_reuse_and_no_skip();
    test_partial_bank_cursor_and_cancel_position();
    test_owner_isolation();
    test_duplicate_failure_and_no_early_release();
    test_parallel_finalizer_and_stale_reuse();
    return 0;
}
#else
int main() { return 0; }
#endif
