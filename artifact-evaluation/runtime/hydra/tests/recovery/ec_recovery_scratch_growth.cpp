// The tests deliberately exercise calls inside assert; never elide them in a
// Release build configured with -DNDEBUG.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "recovery/ec_recovery_scratch.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

namespace {

using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::cache::ec_batch::EcStagingPool;
using FarLib::cache::ec_read_recovery::EcRecoveryScratchPool;
using FarLib::cache::ec_read_recovery::RecoveryScratchChunk;

struct FakeRegistration {
    uint32_t lkey;
};

struct FakeContext {
    struct Allocation {
        void *base;
        size_t bytes;
        uint32_t lkey;
        FakeRegistration *registration;
    };

    mutable std::mutex mutex;
    std::vector<Allocation> allocations;
    uint32_t next_lkey = 0x1000;
    size_t release_count = 0;
    bool fail_next = false;
};

bool fake_allocate(void *opaque, size_t bytes, RecoveryScratchChunk *out) {
    auto &context = *static_cast<FakeContext *>(opaque);
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.fail_next) {
        context.fail_next = false;
        return false;
    }
    void *base = std::malloc(bytes);
    if (base == nullptr) return false;
    auto *registration = new (std::nothrow) FakeRegistration{
        context.next_lkey++};
    if (registration == nullptr) {
        std::free(base);
        return false;
    }
    context.allocations.push_back(
        {base, bytes, registration->lkey, registration});
    *out = {base, registration->lkey, registration};
    return true;
}

void fake_release(void *opaque, RecoveryScratchChunk chunk) {
    auto &context = *static_cast<FakeContext *>(opaque);
    std::lock_guard<std::mutex> lock(context.mutex);
    bool found = false;
    for (auto it = context.allocations.begin();
         it != context.allocations.end(); ++it) {
        if (it->base != chunk.base || it->lkey != chunk.lkey ||
            it->registration != chunk.registration) {
            continue;
        }
        std::free(it->base);
        delete it->registration;
        context.allocations.erase(it);
        found = true;
        break;
    }
    assert(found);
    ++context.release_count;
}

void check_slot_layout(const EcStagingGroupSlot &slot,
                       const FakeContext &context) {
    assert(slot.valid());
    assert(slot.slot_size != 0);
    const FakeContext::Allocation *owner = nullptr;
    {
        std::lock_guard<std::mutex> lock(context.mutex);
        for (const auto &allocation : context.allocations) {
            if (allocation.lkey == slot.lkey) {
                owner = &allocation;
                break;
            }
        }
    }
    assert(owner != nullptr);
    const auto *base = static_cast<const unsigned char *>(owner->base);
    const auto *data0 = static_cast<const unsigned char *>(slot.data[0]);
    assert(data0 >= base && data0 < base + owner->bytes);
    assert(reinterpret_cast<uintptr_t>(data0) -
               reinterpret_cast<uintptr_t>(base) ==
           slot.mr_offset);
    for (size_t i = 0; i < FarLib::cache::ec_batch::kEcBatchDataSlots;
         ++i) {
        const auto *expected = data0 + i * slot.slot_size;
        assert(slot.data[i] == expected);
    }
    for (size_t i = 0; i < FarLib::cache::ec_batch::kEcBatchParitySlots;
         ++i) {
        const auto *expected =
            data0 + (FarLib::cache::ec_batch::kEcBatchDataSlots + i) *
                        slot.slot_size;
        assert(slot.parity[i] == expected);
    }
}

void test_growth_and_immutable_metadata() {
    constexpr size_t slot_size = 64;
    constexpr size_t lease_count = 321;
    FakeContext context;
    EcRecoveryScratchPool pool;
    assert(pool.init(slot_size, &context, fake_allocate, fake_release));

    std::vector<EcStagingGroupSlot> leases(lease_count);
    for (auto &lease : leases) assert(pool.acquire(&lease));

    // The old fixed pool depth was eight. Exercise both the first growth past
    // eight and multiple metadata-vector reallocations past 64.
    assert(pool.in_use() == lease_count);
    assert(pool.peak_in_use() == lease_count);
    assert(pool.depth() >= lease_count);
    assert(pool.growths() >= 6);

    for (const auto &lease : leases) check_slot_layout(lease, context);

    const auto first = leases.front();
    auto tampered = first;
    tampered.lkey ^= 1u;
    assert(!pool.release(tampered));
    assert(pool.in_use() == lease_count);
    assert(pool.release(first));
    assert(pool.in_use() == lease_count - 1);

    for (size_t i = 1; i < leases.size(); ++i) {
        assert(pool.release(leases[i]));
    }
    assert(pool.in_use() == 0);
    assert(context.release_count == 0);
}

void test_reuse_and_stale_generation() {
    FakeContext context;
    EcRecoveryScratchPool pool;
    assert(pool.init(128, &context, fake_allocate, fake_release));

    EcStagingGroupSlot old_lease;
    assert(pool.acquire(&old_lease));
    const auto stale = old_lease;
    assert(pool.release(old_lease));
    assert(!pool.release(old_lease));

    EcStagingGroupSlot current;
    assert(pool.acquire(&current));
    assert(current.index == stale.index);
    assert(current.generation != stale.generation);
    assert(!pool.release(stale));
    assert(pool.in_use() == 1);
    assert(pool.release(current));
    assert(pool.in_use() == 0);
}

void test_allocation_failure_preserves_and_retries() {
    FakeContext context;
    EcRecoveryScratchPool pool;
    assert(pool.init(256, &context, fake_allocate, fake_release));

    std::vector<EcStagingGroupSlot> leases(8);
    for (auto &lease : leases) assert(pool.acquire(&lease));
    assert(pool.depth() == 8);
    const auto first = leases.front();
    const size_t bytes_before = pool.bytes();
    const size_t allocations_before = context.allocations.size();

    context.fail_next = true;
    EcStagingGroupSlot failed;
    assert(!pool.acquire(&failed));
    assert(pool.depth() == 8);
    assert(pool.bytes() == bytes_before);
    assert(pool.in_use() == 8);
    assert(pool.allocation_failures() == 1);
    assert(context.allocations.size() == allocations_before);
    check_slot_layout(first, context);

    EcStagingGroupSlot retried;
    assert(pool.acquire(&retried));
    assert(pool.depth() == 16);
    assert(pool.in_use() == 9);
    assert(context.allocations.size() == allocations_before + 1);
    assert(pool.release(retried));
    for (const auto &lease : leases) assert(pool.release(lease));
    assert(pool.in_use() == 0);
}

void test_explicit_limit_eight() {
    FakeContext context;
    EcRecoveryScratchPool pool;
    assert(pool.init(64, &context, fake_allocate, fake_release, 8));
    assert(pool.limit() == 8);

    std::vector<EcStagingGroupSlot> leases(8);
    for (auto &lease : leases) assert(pool.acquire(&lease));
    EcStagingGroupSlot ninth;
    assert(!pool.acquire(&ninth));
    assert(pool.depth() == 8);
    assert(pool.in_use() == 8);
    assert(pool.bytes() ==
           8 * 64 * FarLib::cache::ec_batch::kEcBatchSegmentsPerGroup);
    for (const auto &lease : leases) assert(pool.release(lease));
    assert(pool.in_use() == 0);
}

void test_concurrent_acquire_release() {
    constexpr size_t thread_count = 8;
    constexpr size_t leases_per_thread = 40;
    FakeContext context;
    EcRecoveryScratchPool pool;
    assert(pool.init(96, &context, fake_allocate, fake_release));

    std::atomic<size_t> ready{0};
    std::atomic<size_t> acquired_threads{0};
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (size_t tid = 0; tid < thread_count; ++tid) {
        workers.emplace_back([&] {
            std::vector<EcStagingGroupSlot> leases(leases_per_thread);
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();

            size_t acquired = 0;
            for (; acquired < leases.size(); ++acquired) {
                if (!pool.acquire(&leases[acquired])) {
                    failed.store(true, std::memory_order_release);
                    break;
                }
            }
            acquired_threads.fetch_add(1, std::memory_order_release);
            while (acquired_threads.load(std::memory_order_acquire) !=
                   thread_count) {
                std::this_thread::yield();
            }
            for (size_t i = 0; i < acquired; ++i) {
                if (!pool.release(leases[i]))
                    failed.store(true, std::memory_order_release);
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != thread_count)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();

    assert(!failed.load(std::memory_order_acquire));
    assert(pool.in_use() == 0);
    assert(pool.peak_in_use() == thread_count * leases_per_thread);
    assert(pool.depth() >= thread_count * leases_per_thread);
}

void test_cross_worker_returns_and_shared_staging_adapter() {
    FakeContext context;
    EcRecoveryScratchPool pool;
    assert(pool.init(64, &context, fake_allocate, fake_release));

    EcStagingGroupSlot worker_three;
    EcStagingGroupSlot worker_four;
    assert(pool.acquire(&worker_three, 3));
    assert(pool.acquire(&worker_four, 4));
    const auto worker_three_copy = worker_three;
    // Release is intentionally performed without the acquiring worker id.
    // The lease's immutable owner routes it back to worker three's return
    // list; worker four must not be able to consume it.
    assert(pool.release(worker_three));
    assert(pool.in_use() == 1);
    EcStagingGroupSlot worker_three_again;
    assert(pool.acquire(&worker_three_again, 3));
    assert(worker_three_again.index == worker_three_copy.index);
    assert(pool.release(worker_three_again));
    assert(pool.release(worker_four));
    assert(pool.in_use() == 0);

    std::vector<unsigned char> fixed_mr(
        EcStagingPool::required_bytes(64, 1));
    EcStagingPool staging;
    assert(staging.init(fixed_mr.data(), fixed_mr.size(), 64, 1));
    size_t callback_owner = 11;
    assert(staging.bind_shared_buffers(
        &pool, [&callback_owner] { return callback_owner; }));
    assert(staging.shared_buffers_bound());

    EcStagingGroupSlot shared;
    assert(staging.acquire(&shared));
    assert(shared.lkey != 0);
    assert(staging.owns_slot(shared));
    assert(staging.owns_buffer(shared, shared.data[0], shared.slot_size));
    assert(staging.owns_buffer(shared, shared.parity[1], shared.slot_size));
    assert(!staging.owns_buffer(shared, fixed_mr.data(), shared.slot_size));
    assert(staging.release(shared));
    assert(!staging.owns_slot(shared));

    // A valid lease from the shared provider is still foreign to a second
    // provider, even when its shape and callback metadata match.
    EcRecoveryScratchPool foreign;
    assert(foreign.init(64, &context, fake_allocate, fake_release));
    EcStagingGroupSlot foreign_slot;
    assert(foreign.acquire(&foreign_slot, 11));
    assert(!pool.release(foreign_slot));
    assert(foreign.release(foreign_slot));
}

}  // namespace

int main() {
    test_growth_and_immutable_metadata();
    test_reuse_and_stale_generation();
    test_allocation_failure_preserves_and_retries();
    test_explicit_limit_eight();
    test_concurrent_acquire_release();
    test_cross_worker_returns_and_shared_staging_adapter();
    std::cout << "EC_RECOVERY_SCRATCH_GROWTH_PASS\n";
    return 0;
}
