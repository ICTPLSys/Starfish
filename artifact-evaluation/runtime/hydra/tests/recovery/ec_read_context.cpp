// The tests deliberately exercise calls inside assert; never elide them in a
// Release build configured with -DNDEBUG.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "recovery/ec_read_context.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::cache::ec_read_recovery::EcReadContext;
using FarLib::cache::ec_read_recovery::EcReadContextEvent;
using FarLib::cache::ec_read_recovery::EcReadContextPool;
using FarLib::cache::ec_read_recovery::EcReadTokenEventKind;
using FarLib::cache::ec_read_recovery::ec_read_wr_id_segment;
using FarLib::cache::ec_read_recovery::ec_read_wr_id_token;
using FarLib::cache::ec_read_recovery::encode_ec_read_wr_id;
using FarLib::cache::ec_read_recovery::kEcReadSegmentCount;

constexpr uint8_t kAllSegments =
    static_cast<uint8_t>((1u << kEcReadSegmentCount) - 1u);

EcStagingGroupSlot scratch_for(uintptr_t tag) {
    EcStagingGroupSlot slot;
    for (size_t i = 0; i < 4; ++i) {
        slot.data[i] = reinterpret_cast<void *>(tag + i * 0x10);
    }
    for (size_t i = 0; i < 2; ++i) {
        slot.parity[i] = reinterpret_cast<void *>(tag + 0x80 + i * 0x10);
    }
    slot.slot_size = 4096;
    slot.index = static_cast<uint32_t>(tag);
    slot.mr_offset = tag * 4096;
    slot.lkey = static_cast<uint32_t>(tag ^ 0x55aa);
    slot.generation = tag + 1;
    return slot;
}

void assert_pending(const EcReadContextEvent &event) {
    assert(event.kind == EcReadTokenEventKind::kPending);
    assert(event.context == nullptr);
    assert(event.token == nullptr);
}

void assert_ignored(const EcReadContextEvent &event) {
    assert(event.kind == EcReadTokenEventKind::kIgnored);
    assert(event.context == nullptr);
    assert(event.token == nullptr);
    assert(event.profile_acquire_ns == 0);
    assert(event.profile_byte_count == 0);
}

void assert_nonwinner(const EcReadContextEvent &event) {
    assert(event.kind == EcReadTokenEventKind::kPending ||
           event.kind == EcReadTokenEventKind::kIgnored);
    assert(event.context == nullptr);
    assert(event.token == nullptr);
}

void test_success_reorder_and_close() {
    EcReadContextPool pool(2, 0);
    const auto scratch = scratch_for(11);
    uint64_t id = 0;
    EcReadContext *context = nullptr;
    assert(pool.acquire(0, 0x100000, 0x200000, 1234, kAllSegments, 2,
                        scratch, &id, &context));
    assert(context != nullptr);
    assert((reinterpret_cast<uintptr_t>(context) & 255u) == 0);
    assert(id == context->token_id());
    const uint64_t wr_id = encode_ec_read_wr_id(id, 3);
    assert(ec_read_wr_id_token(wr_id) == id);
    assert(ec_read_wr_id_segment(wr_id) == 3);
    assert(context->target_local_addr == 0x100000);
    assert(context->remote_addr == 0x200000);
    assert(context->byte_count == 1234);
    assert(context->alive_mask == kAllSegments);
    assert(context->own_shard_idx == 2);
    assert(pool.in_use() == 1);

    for (uint8_t segment : {5, 1, 3, 0, 4, 2}) {
        assert(pool.mark_segment_posted(id, segment));
    }
    for (uint8_t segment : {3, 5, 0, 2, 4, 1}) {
        assert_pending(pool.complete_segment_event(id, segment, true));
    }

    const auto event = pool.finish_posting(id);
    assert(event.kind == EcReadTokenEventKind::kWinner);
    assert(event.context == context);
    assert(event.token == context);
    assert(event.scratch.index == scratch.index);
    assert(event.scratch.data[0] == scratch.data[0]);
    assert(event.profile_acquire_ns == context->profile_acquire_ns);
    assert(event.profile_byte_count == 1234);
    assert(pool.in_use() == 1);
    assert(pool.release(id));
    assert(pool.in_use() == 0);
    assert_ignored(pool.finish_posting(id));
}

void test_completion_before_post_mark() {
    EcReadContextPool pool(1, 0);
    const auto scratch = scratch_for(12);
    uint64_t id = 0;
    EcReadContext *context = nullptr;
    assert(pool.acquire(0, 0x300000, 0x400000, 64, 0x03, 0, scratch, &id,
                        &context));

    // A very fast CQE can precede the posting worker's bookkeeping mark.
    assert_pending(pool.complete_segment_event(id, 1, true));
    assert(pool.mark_segment_posted(id, 1));
    assert(pool.mark_segment_posted(id, 0));
    assert_pending(pool.complete_segment_event(id, 0, true));
    const auto event = pool.finish_posting(id);
    assert(event.kind == EcReadTokenEventKind::kWinner);
    assert(event.token == context);
    assert(pool.release(id));
}

void test_partial_no_post_and_error_cleanup() {
    const auto scratch = scratch_for(13);

    {
        EcReadContextPool pool(1, 0);
        uint64_t id = 0;
        EcReadContext *context = nullptr;
        assert(pool.acquire(0, 0x500000, 0x600000, 32, kAllSegments, 1,
                            scratch, &id, &context));
        for (uint8_t segment = 0; segment < 3; ++segment) {
            assert(pool.mark_segment_posted(id, segment));
            assert_pending(pool.complete_segment_event(id, segment, true));
        }
        const auto event = pool.finish_posting(id);
        assert(event.kind == EcReadTokenEventKind::kRelease);
        assert(event.token == context);
        assert(event.scratch.index == scratch.index);
        assert(pool.in_use() == 1);  // kRelease does not auto-free.
        assert(pool.release(id));
        assert(pool.in_use() == 0);
    }

    {
        EcReadContextPool pool(1, 0);
        uint64_t id = 0;
        EcReadContext *context = nullptr;
        assert(pool.acquire(0, 0x700000, 0x800000, 32, 0x03, 1, scratch, &id,
                            &context));
        const auto event = pool.finish_posting(id);
        assert(event.kind == EcReadTokenEventKind::kRelease);
        assert(event.token == context);
        assert(pool.release(id));
    }

    {
        EcReadContextPool pool(1, 0);
        uint64_t id = 0;
        EcReadContext *context = nullptr;
        assert(pool.acquire(0, 0x900000, 0xa00000, 32, 0x03, 0, scratch, &id,
                            &context));
        assert(pool.mark_segment_posted(id, 0));
        assert(pool.mark_segment_posted(id, 1));
        assert_pending(pool.complete_segment_event(id, 0, false));
        assert_pending(pool.complete_segment_event(id, 1, true));
        const auto event = pool.finish_posting(id);
        assert(event.kind == EcReadTokenEventKind::kRelease);
        assert(event.token == context);
        assert(pool.release(id));
    }
}

void test_duplicate_stale_and_reused_context() {
    EcReadContextPool pool(1, 0);
    const auto scratch = scratch_for(14);
    uint64_t old_id = 0;
    EcReadContext *old_context = nullptr;
    assert(pool.acquire(0, 0xb00000, 0xc00000, 16, 1, 0, scratch, &old_id,
                        &old_context));
    assert(pool.mark_segment_posted(old_id, 0));
    assert_pending(pool.complete_segment_event(old_id, 0, true));
    assert(pool.finish_posting(old_id).kind == EcReadTokenEventKind::kWinner);
    assert_ignored(pool.complete_segment_event(old_id, 0, true));
    assert(pool.release(old_id));

    uint64_t new_id = 0;
    EcReadContext *new_context = nullptr;
    const auto replacement = scratch_for(15);
    assert(pool.acquire(0, 0xd00000, 0xe00000, 17, 1, 3, replacement, &new_id,
                        &new_context));
    // The stable storage is reused, but its generation is not.
    assert(new_context == old_context);
    assert(new_id != old_id);
    assert_ignored(pool.complete_segment_event(old_id, 0, true));
    assert(pool.mark_segment_posted(new_id, 0));
    assert_pending(pool.complete_segment_event(new_id, 0, true));
    const auto event = pool.finish_posting(new_id);
    assert(event.kind == EcReadTokenEventKind::kWinner);
    assert(event.token == new_context);
    assert_ignored(pool.complete_segment_event(new_id, 0, true));
    assert(pool.release(new_id));
}

void test_concurrent_last_winner() {
    EcReadContextPool pool(1, 0);
    const auto scratch = scratch_for(16);
    uint64_t id = 0;
    EcReadContext *context = nullptr;
    assert(pool.acquire(0, 0xf00000, 0x1000000, 8, kAllSegments, 0, scratch,
                        &id, &context));
    for (uint8_t segment = 0; segment < kEcReadSegmentCount; ++segment) {
        assert(pool.mark_segment_posted(id, segment));
    }
    assert_pending(pool.finish_posting(id));

    std::array<EcReadContextEvent, kEcReadSegmentCount> events{};
    std::array<std::thread, kEcReadSegmentCount> threads;
    for (size_t i = 0; i < kEcReadSegmentCount; ++i) {
        threads[i] = std::thread([&, i] {
            events[i] = pool.complete_segment_event(id,
                                                    static_cast<uint8_t>(i),
                                                    true);
        });
    }
    for (auto &thread : threads) thread.join();
    size_t winners = 0;
    for (const auto &event : events) {
        if (event.kind == EcReadTokenEventKind::kWinner) {
            ++winners;
            assert(event.token == context);
        } else {
            assert_nonwinner(event);
        }
    }
    assert(winners == 1);
    assert(pool.release(id));
}

void test_growth_and_owner_return_stack() {
    EcReadContextPool pool(2, 0);
    const auto scratch = scratch_for(17);
    constexpr size_t live_count = 24;
    std::vector<uint64_t> ids;
    std::vector<EcReadContext *> contexts;
    ids.reserve(live_count);
    contexts.reserve(live_count);
    for (size_t i = 0; i < live_count; ++i) {
        uint64_t id = 0;
        EcReadContext *context = nullptr;
        assert(pool.acquire(0, 0x1100000 + i, 0x1200000 + i, 4,
                            static_cast<uint8_t>(1u << (i % 6)), 0, scratch,
                            &id, &context));
        ids.push_back(id);
        contexts.push_back(context);
        assert(pool.finish_posting(id).kind == EcReadTokenEventKind::kRelease);
    }
    assert(pool.in_use() == live_count);
    assert(pool.capacity() >= live_count);
    assert(pool.peak_in_use() >= live_count);
    for (uint64_t id : ids) assert(pool.release(id));
    assert(pool.in_use() == 0);

    // Returns are pushed by a completion-side thread and drained only by the
    // owner on its next acquire.
    uint64_t id = 0;
    EcReadContext *context = nullptr;
    assert(pool.acquire(1, 0x1300000, 0x1400000, 5, 1, 0, scratch, &id,
                        &context));
    assert(pool.finish_posting(id).kind == EcReadTokenEventKind::kRelease);
    std::atomic<bool> released{false};
    std::thread completer([&] {
        assert(pool.release(id));
        released.store(true, std::memory_order_release);
    });
    completer.join();
    assert(released.load(std::memory_order_acquire));

    uint64_t next_id = 0;
    EcReadContext *next_context = nullptr;
    assert(pool.acquire(1, 0x1500000, 0x1600000, 6, 1, 0, scratch, &next_id,
                        &next_context));
    assert(next_context == context);
    assert(next_id != id);
    assert(pool.finish_posting(next_id).kind == EcReadTokenEventKind::kRelease);
    assert(pool.release(next_id));
}

}  // namespace

int main() {
    test_success_reorder_and_close();
    test_completion_before_post_mark();
    test_partial_no_post_and_error_cleanup();
    test_duplicate_stale_and_reused_context();
    test_concurrent_last_winner();
    test_growth_and_owner_return_stack();
    return 0;
}
