#include "cache/carbink/write_fence.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

using Table = FarLib::cache::carbink::WriteFenceTable;

int main() {
    Table table(/*owner_count=*/2, /*endpoint_count=*/2);
    assert(table.in_use() == 0);
    assert(!Table::is_fence(0));
    assert(!Table::is_fence(0x8500000000000001ULL));

    std::array<uint64_t, 3> ids{11, 22, 33};
    const uint64_t first = table.acquire(0, 0, ids);
    assert(first != 0);
    assert(Table::is_fence(first));
    assert(table.in_use() == 1);
    assert(!table.release(first));  // early release is rejected

    const auto *entry = table.consume(first);
    assert(entry != nullptr);
    assert(entry->count == ids.size());
    assert(entry->ids[0] == ids[0] && entry->ids[2] == ids[2]);
    assert(entry->owner == 0 && entry->endpoint == 0);
    assert(table.consume(first) == nullptr);  // exact-once claim
    assert(table.release(first));
    assert(!table.release(first));            // duplicate release rejected

    const uint64_t reused = table.acquire(0, 0, ids);
    assert(reused != 0 && reused != first);
    assert(table.consume(first) == nullptr);  // stale generation
    assert(!table.release(first));
    assert(table.consume(reused) != nullptr);
    assert(table.release(reused));

    std::array<uint64_t, Table::kMaxIdsPerFence> max_ids{};
    for (size_t i = 0; i < max_ids.size(); ++i) max_ids[i] = 1000 + i;
    const uint64_t max_fence = table.acquire(1, 0, max_ids);
    assert(max_fence != 0);
    const auto *max_entry = table.consume(max_fence);
    assert(max_entry != nullptr && max_entry->count == max_ids.size());
    assert(max_entry->ids[max_ids.size() - 1] ==
           max_ids[max_ids.size() - 1]);
    assert(table.release(max_fence));

    std::array<uint64_t, Table::kMaxIdsPerFence + 1> too_many{};
    too_many.fill(1);
    assert(table.acquire(1, 0, too_many) == 0);
    std::array<uint64_t, 1> zero_id{0};
    assert(table.acquire(1, 0, zero_id) == 0);

    std::array<uint64_t, 1> one{77};
    const uint64_t cross_a = table.acquire(0, 0, one);
    const uint64_t cross_b = table.acquire(1, 1, one);
    assert(cross_a != 0 && cross_b != 0 && cross_a != cross_b);
    const auto *entry_b = table.consume(cross_b);  // out-of-order completion
    const auto *entry_a = table.consume(cross_a);
    assert(entry_a != nullptr && entry_b != nullptr);
    assert(entry_a->owner == 0 && entry_a->endpoint == 0);
    assert(entry_b->owner == 1 && entry_b->endpoint == 1);
    assert(table.release(cross_a) && table.release(cross_b));

    std::vector<uint64_t> held;
    held.reserve(Table::kSlotsPerLane);
    for (size_t i = 0; i < Table::kSlotsPerLane; ++i) {
        const uint64_t wrid = table.acquire(0, 1, one);
        assert(wrid != 0);
        held.push_back(wrid);
    }
    assert(table.acquire(0, 1, one) == 0);  // lane backpressure
    assert(table.consume(held.front()) != nullptr);
    assert(table.acquire(0, 1, one) == 0);  // consumed but unreleased
    assert(table.release(held.front()));
    const uint64_t after_release = table.acquire(0, 1, one);
    assert(after_release != 0 && after_release != held.front());
    for (size_t i = 1; i < held.size(); ++i) {
        assert(table.consume(held[i]) != nullptr);
        assert(table.release(held[i]));
    }
    assert(table.consume(after_release) != nullptr);
    assert(table.release(after_release));

    // Force repeated same-slot reuse. A stale WRID must never consume the
    // newly posted generation, which exercises the packed generation/state CAS.
    for (size_t round = 0; round < 4096; ++round) {
        const uint64_t old_wrid = table.acquire(0, 0, one);
        assert(old_wrid != 0);
        assert(table.consume(old_wrid) != nullptr);
        assert(table.release(old_wrid));
        const uint64_t new_wrid = table.acquire(0, 0, one);
        assert(new_wrid != 0 && new_wrid != old_wrid);
        assert(table.consume(old_wrid) == nullptr);
        assert(table.consume(new_wrid) != nullptr);
        assert(table.release(new_wrid));
    }

    // Concurrent consumers must still have exactly one winner across repeats.
    for (size_t round = 0; round < 64; ++round) {
        const uint64_t concurrent = table.acquire(0, 0, one);
        assert(concurrent != 0);
        std::atomic<unsigned> winners{0};
        std::vector<std::thread> consumers;
        for (size_t i = 0; i < 16; ++i) {
            consumers.emplace_back([&] {
                if (table.consume(concurrent) != nullptr) {
                    winners.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto &consumer : consumers) consumer.join();
        assert(winners.load(std::memory_order_relaxed) == 1);
        assert(table.release(concurrent));
    }

    assert(table.in_use() == 0);
    return 0;
}
