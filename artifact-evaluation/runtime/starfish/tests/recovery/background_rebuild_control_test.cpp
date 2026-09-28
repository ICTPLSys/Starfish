#include "recovery/ec_background_rebuild.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    using namespace FarLib::cache::ec_background;
    constexpr uint64_t kBytes = 262144;
    constexpr uint64_t kMbps = 2500;
    constexpr uint64_t kBase = 1000000000;
    constexpr uint64_t kDuration =
        (kBytes * 1000 + kMbps - 1) / kMbps;

    SharedState shared;
    assert(!shared.fully_rebuilt.load());
    shared.start_ns = kBase;
    shared.scan_limit = 11856;
    shared.ready.store(true, std::memory_order_release);
    assert(shared.ready.load(std::memory_order_acquire));

    // Every slot has its own generation stream and therefore a unique WR id.
    std::array<uint64_t, kSlots> ids{};
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
        ids[slot] = shared.completions[slot].arm(slot);
        assert(ids[slot] != 0 && is_wr_id(ids[slot]));
        assert(slot_from_wr_id(ids[slot]) == slot);
    }
    auto sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    assert(std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) ==
           sorted_ids.end());

    Completion c;
    const auto first = c.arm(7);
    assert(first && is_wr_id(first) && slot_from_wr_id(first) == 7 &&
           c.matches(first));
    assert(!is_wr_id(1ull << 60)); // RMW
    assert(!is_wr_id((1ull << 61) | 123)); // normal WRITE
    assert(!is_wr_id((1ull << 62) | 123)); // normal READ
    assert(!is_wr_id((1ull << 63) | 123)); // EC WRITE
    assert(!is_wr_id((3ull << 62) | 123)); // degraded READ
    assert(slot_from_wr_id((first & ~kSlotMask) | kSlots) == kSlots); // invalid slot
    assert(slot_from_wr_id(0) == kSlots);
    assert(c.arm(kSlots) == 0);

    std::atomic<int> payload{0};
    std::thread callback([&] {
        payload.store(42, std::memory_order_relaxed);
        assert(c.matches(first));
        c.complete(true);
    });
    while (!c.status.load(std::memory_order_acquire)) std::this_thread::yield();
    assert(payload.load(std::memory_order_relaxed) == 42);
    callback.join();
    const auto second = c.arm(7);
    assert(!c.matches(first) && c.matches(second));
    assert(slot_from_wr_id(second) == 7);
    assert(c.status.load() == 0);
    c.complete(false);
    assert(c.status.load() == -1);
    c.sequence = kSequenceMask;
    assert(c.arm(7) == 0);

    // Four workers share one READ ticket clock. The first 16 grants form one
    // burst, not four independent 2500 MB/s streams.
    constexpr uint32_t kBurstSlots =
        static_cast<uint32_t>(kBurstBytes / kBytes);
    constexpr uint32_t kCallsPerWorker = kBurstSlots / kWorkers;
    static_assert(kBurstSlots == 16);
    static_assert(kCallsPerWorker * kWorkers == kBurstSlots);
    std::array<std::array<uint64_t, kCallsPerWorker>, kWorkers> reservations{};
    std::array<std::array<bool, kCallsPerWorker>, kWorkers> granted{};
    std::array<std::thread, kWorkers> workers;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        workers[worker] = std::thread([&, worker] {
            shared.worker_ids.fetch_add(1, std::memory_order_relaxed);
            for (uint32_t i = 0; i < kCallsPerWorker; ++i) {
                granted[worker][i] = shared.reads.try_reserve(
                    kBase, kBytes, kMbps, kBurstBytes,
                    &reservations[worker][i]);
            }
            shared.workers_done.fetch_add(1, std::memory_order_release);
        });
    }
    for (auto &worker : workers) worker.join();
    assert(shared.worker_ids.load(std::memory_order_acquire) == kWorkers);
    assert(shared.workers_done.load(std::memory_order_acquire) == kWorkers);

    std::vector<uint64_t> read_tickets;
    read_tickets.reserve(kWorkers * kCallsPerWorker);
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        for (uint32_t i = 0; i < kCallsPerWorker; ++i) {
            assert(granted[worker][i]);
            read_tickets.push_back(reservations[worker][i]);
        }
    }
    std::sort(read_tickets.begin(), read_tickets.end());
    assert(read_tickets.front() == kBase);
    for (size_t i = 1; i < read_tickets.size(); ++i)
        assert(read_tickets[i] - read_tickets[i - 1] == kDuration);
    assert(shared.reads.next_ns.load(std::memory_order_acquire) ==
           kBase + kBurstSlots * kDuration);

    // A blocked caller receives the next admissible time without charging the
    // virtual clock; retrying exactly then grants one ticket at +dt.
    uint64_t ready = 0;
    const auto before = shared.reads.next_ns.load(std::memory_order_acquire);
    assert(!shared.reads.try_reserve(kBase, kBytes, kMbps, kBurstBytes,
                                     &ready));
    assert(ready == kBase + kDuration);
    assert(shared.reads.next_ns.load(std::memory_order_acquire) == before);
    uint64_t retry_ticket = 0;
    assert(shared.reads.try_reserve(ready, kBytes, kMbps, kBurstBytes,
                                     &retry_ticket));
    assert(retry_ticket == kBase + kBurstSlots * kDuration);

    // The stream obeys rate*elapsed plus the configured burst allowance.
    const uint64_t elapsed =
        retry_ticket - read_tickets.front() + kDuration;
    const uint64_t total_bytes =
        static_cast<uint64_t>(read_tickets.size() + 1) * kBytes;
    assert(total_bytes * 1000 <=
           elapsed * kMbps + kBurstBytes * 1000);

    // Check actual grant timestamps (not virtual future tickets) at every
    // prefix of a repeatedly replenished stream.
    Pacer bounded;
    uint64_t issued = 0;
    for (uint64_t step = 0; step < 2000; ++step) {
        const uint64_t now = kBase + step * 50000;
        while (bounded.try_reserve(now, kBytes, kMbps, kBurstBytes, &ready))
            issued += kBytes;
        assert(issued * 1000 <= (now - kBase) * kMbps + kBurstBytes * 1000);
    }
    assert(issued > 240000000);

    // WRITE tickets are independent of the READ clock.
    uint64_t write_ticket = 0;
    assert(shared.writes.try_reserve(kBase, kBytes, kMbps, kBurstBytes,
                                     &write_ticket));
    assert(write_ticket == kBase);

    std::cout << "background rebuild completion/shared-pipeline tests passed\n";
}
