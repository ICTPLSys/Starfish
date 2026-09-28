// CPU-only regression coverage for the Exclusive work-alignment port.
//
// This file intentionally calls the production GlobalHeap/RegionList helpers;
// it is not a replacement allocator model.  The parent build should add this
// file as recovery_exclusive_work_alignment after the runtime port lands.

#include <bits/stdc++.h> // include standard headers before private/public test access
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <unordered_set>
#include <vector>

#define private public
#include "cache/region_based_allocator.hpp"
#undef private

using FarLib::allocator::EvacuationEligibility;
using FarLib::allocator::EvictTask;
using FarLib::allocator::GlobalHeap;
using FarLib::allocator::IN_USE;
using FarLib::allocator::RegionHead;
using FarLib::allocator::RegionList;
using FarLib::allocator::RegionPlacement;
using FarLib::allocator::evict_region_admissible;
using FarLib::allocator::refund_legacy_mark_regions;
using FarLib::allocator::reserve_legacy_mark_regions;
using FarLib::allocator::refund_scan_visits;
using FarLib::allocator::reserve_scan_visits;

[[noreturn]] static void fail(const char *expression, int line) {
    std::cerr << "CHECK failed: " << expression << " line=" << line << '\n';
    std::exit(EXIT_FAILURE);
}

#define CHECK(expression) \
    do { \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

struct RegionPool {
    std::vector<RegionHead *> nodes;

    explicit RegionPool(size_t count) {
        nodes.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto *node = new RegionHead{};
            node->state.store(IN_USE, std::memory_order_relaxed);
            node->bin = 0;
            nodes.push_back(node);
        }
    }

    ~RegionPool() {
        for (auto *node : nodes) delete node;
    }
};

static auto &streaming_full(GlobalHeap &heap) {
    return heap.full_region_list[static_cast<size_t>(
        RegionPlacement::Streaming)];
}

static void push_all(RegionList &list,
                     const std::vector<RegionHead *> &nodes) {
    for (auto *node : nodes) list.push(node);
}

// All is the Exclusive eligibility contract: an unmarked, nonempty region is
// admissible.  LegacyConcurrent deliberately retains the old marked-list gate.
static void test_all_eligibility_is_not_legacy_concurrent() {
    RegionHead region{};
    // is_empty() is based on the list heads.  The sentinel is inspected only
    // by the predicate, never dereferenced.
    region.active_list = reinterpret_cast<FarLib::allocator::BlockHead *>(1);
    region.marked_list = nullptr;
    CHECK(evict_region_admissible(&region, EvacuationEligibility::All));
    CHECK(!evict_region_admissible(
        &region, EvacuationEligibility::LegacyConcurrent));
}

// Directly exercise the production RegionList cursor from native std::threads.
// GlobalHeap::collect_evict_tasks has an intentional Fibre yield after a visit
// budget exhaustion; calling that branch from a native test thread would enter
// Fred::yieldGlobal without a Fred and is not a valid unit-test context.  The
// cursor itself has the same budget/resume semantics and correctly falls back
// to std::this_thread::yield while its native-thread yield gate is disabled.
static void scan_cursor_with_six_workers(
    RegionList &list, uint32_t timestamp, uint64_t scan_generation,
    std::atomic_size_t *visit_budget, std::atomic_bool *pass_complete) {
    constexpr size_t kWorkers = 6;
    std::thread workers[kWorkers];
    for (auto &worker : workers) {
        worker = std::thread([&] {
            // Standalone CPU tests do not run inside a Fibre worker.  Keep the
            // production RegionList path from attempting a Fibre yield here.
            FarLib::allocator::disable_region_list_fibre_yield_for_current_thread();
            for (;;) {
                const size_t reserved =
                    reserve_scan_visits(visit_budget, 256);
                if (reserved == 0) break;
                RegionHead *batch[64];
                bool more = false;
                bool exhausted = false;
                size_t visited = 0;
                list.pop_matching_batch_evict_cursor(
                    timestamp, batch, 64, reserved, nullptr, &more,
                    &exhausted, [](RegionHead *) { return false; }, nullptr,
                    scan_generation, false, &visited);
                refund_scan_visits(visit_budget, reserved - visited);
                CHECK(!exhausted || more);
                if (!more) {
                    pass_complete->store(true, std::memory_order_relaxed);
                    break;
                }
                if (visited == 0) break;
            }
        });
    }
    for (auto &worker : workers) worker.join();
}

// A zero visit budget must not publish a completed pass.  Six workers then
// share a bounded budget, resume the same scan generation at a new safety
// timestamp, and consume only the unvisited suffix.
static void test_six_worker_evict_budget_resume_and_zero_guard() {
    GlobalHeap heap;
    RegionPool pool(700);
    push_all(streaming_full(heap), pool.nodes);

    std::vector<EvictTask> tasks;
    std::atomic_size_t zero_budget{0};
    std::atomic_bool complete{false};
    const bool stopped = heap.collect_evict_tasks(
        100, tasks, 64, 0, true, EvacuationEligibility::All, nullptr, 1,
        &zero_budget, &complete);
    CHECK(stopped);
    CHECK(tasks.empty());
    CHECK(!complete.load(std::memory_order_relaxed));

    // Six workers each reserve at most the production 256-visit cursor chunk;
    // the shared budget is deliberately not a multiple of that chunk.
    std::atomic_size_t first_budget{384};
    scan_cursor_with_six_workers(streaming_full(heap), 101, 1, &first_budget,
                                 &complete);
    CHECK(first_budget.load(std::memory_order_relaxed) == 0);
    CHECK(!complete.load(std::memory_order_relaxed));

    // Same scan generation carries the RegionList cursor over the safety
    // timestamp.  The remaining 316 regions close the pass and refund 196.
    std::atomic_size_t second_budget{512};
    scan_cursor_with_six_workers(streaming_full(heap), 102, 1, &second_budget,
                                 &complete);
    CHECK(complete.load(std::memory_order_relaxed));
    CHECK(second_budget.load(std::memory_order_relaxed) == 196);

    while (streaming_full(heap).pop() != nullptr) {}
}

static size_t claim_mark_regions_with_six_workers(
    RegionList &list, uint32_t epoch, std::atomic_size_t &budget,
    std::vector<RegionHead *> (&claimed)[6], bool per_worker = false) {
    constexpr size_t kWorkers = 6;
    std::atomic_bool start{false};
    std::thread workers[kWorkers];
    for (size_t worker = 0; worker < kWorkers; ++worker) {
        workers[worker] = std::thread([&, worker] {
            FarLib::allocator::disable_region_list_fibre_yield_for_current_thread();
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            RegionHead *batch[64];
            std::atomic_size_t local_budget{64};
            auto *active_budget = per_worker ? &local_budget : &budget;
            for (;;) {
                const size_t reserved =
                    reserve_legacy_mark_regions(active_budget, 64);
                if (reserved == 0) break;
                const size_t popped = list.pop_matching_batch_cursor(
                    epoch, batch, reserved, nullptr,
                    [](RegionHead *) { return true; });
                refund_legacy_mark_regions(active_budget, reserved - popped);
                claimed[worker].insert(claimed[worker].end(), batch,
                                       batch + popped);
                if (popped == 0) break;
            }
            if (per_worker) {
                CHECK(claimed[worker].size() <= 64);
                budget.fetch_sub(64 - local_budget.load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();
    size_t total = 0;
    for (const auto &worker : claimed) total += worker.size();
    return total;
}

// Preserve the historical per-worker 64-region cap, refund unused reservations,
// and verify that the next sweep reaches every suffix region exactly once.
static void test_mark_budget_refund_and_next_sweep() {
    constexpr uint32_t first_epoch = 7000;
    RegionPool pool(700);
    RegionList list;
    push_all(list, pool.nodes);

    std::atomic_size_t first_budget{512};
    std::vector<RegionHead *> first_claimed[6];
    CHECK(claim_mark_regions_with_six_workers(
              list, first_epoch, first_budget, first_claimed) == 512);
    CHECK(first_budget.load(std::memory_order_relaxed) == 0);
    CHECK(list.size() == 188);
    CHECK(reserve_legacy_mark_regions(&first_budget, 64) == 0);

    std::unordered_set<RegionHead *> unique;
    for (const auto &worker : first_claimed) {
        for (auto *node : worker) {
            CHECK(node->sweep_time_stamp == first_epoch);
            unique.insert(node);
        }
    }
    CHECK(unique.size() == 512);

    // The same sweep stamp does not reacquire marked regions.  A new sweep
    // timestamp reaches exactly the 188-region suffix and refunds 324.
    std::atomic_size_t second_budget{512};
    std::vector<RegionHead *> second_claimed[6];
    CHECK(claim_mark_regions_with_six_workers(
              list, first_epoch + 1, second_budget, second_claimed) == 188);
    CHECK(second_budget.load(std::memory_order_relaxed) == 324);
    CHECK(list.size() == 0);
    for (const auto &worker : second_claimed) {
        for (auto *node : worker) unique.insert(node);
    }
    CHECK(unique.size() == 700);
}

static void test_private_per_marker_cap_and_tail_refund() {
    RegionPool pool(700);
    RegionList list;
    push_all(list, pool.nodes);
    std::atomic_size_t first_remaining{6 * 64};
    std::vector<RegionHead *> first[6];
    CHECK(claim_mark_regions_with_six_workers(list, 8000, first_remaining, first, true) == 384);
    CHECK(first_remaining.load() == 0);
    for (const auto &worker : first) CHECK(worker.size() == 64);
    std::atomic_size_t second_remaining{6 * 64};
    std::vector<RegionHead *> second[6];
    CHECK(claim_mark_regions_with_six_workers(list, 8001, second_remaining, second, true) == 316);
    CHECK(second_remaining.load() == 68);
    for (const auto &worker : second) CHECK(worker.size() <= 64);
    CHECK(list.size() == 0);
}

int main() {
    test_all_eligibility_is_not_legacy_concurrent();
    test_six_worker_evict_budget_resume_and_zero_guard();
    test_mark_budget_refund_and_next_sweep();
    test_private_per_marker_cap_and_tail_refund();
    std::cout << "exclusive_work_alignment=passed workers=6 mark_batch=64 "
                 "evict_eligibility=All\n";
    return 0;
}
