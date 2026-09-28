// CPU-only tests for the per-block direct-EC source gate.
//
// A borrowed source is an existing object-header bit, not an entry move.  It
// must reject a new ordinary READ, must not be claimed while an ordinary READ
// pin is pending, and must become ordinary-READ-eligible after release.

#include <cstdint>
#include <cstdio>

#if __has_include(<infiniband/verbs.h>)

#include "cache/region_based_allocator.hpp"

namespace {

using FarLib::allocator::BlockHead;
using FarLib::allocator::acquire_normal_read_pin;
using FarLib::allocator::ec_write_source_borrowed;
using FarLib::allocator::normal_read_recovery_active;
using FarLib::allocator::release_ec_write_source;
using FarLib::allocator::try_borrow_ec_write_source;

struct Checks {
    int count = 0;
    int failures = 0;

    void check(bool value, const char *what) {
        ++count;
        if (value) {
            std::printf("  ok   %s\n", what);
        } else {
            ++failures;
            std::printf("  FAIL %s\n", what);
        }
    }
};

int run() {
    Checks checks;
    BlockHead block{};
    block.pending_rdma_reads.store(0, std::memory_order_relaxed);
    block.normal_read_recovery_active.store(0, std::memory_order_relaxed);

    const uint16_t pending_generation = acquire_normal_read_pin(&block);
    checks.check(pending_generation != 0,
                 "ordinary READ pin acquires a nonzero generation");
    checks.check(block.pending_rdma_reads.load(std::memory_order_acquire) == 1,
                 "ordinary READ pin is visible as pending");
    checks.check(!try_borrow_ec_write_source(&block),
                 "source borrow fails while an ordinary READ is pending");

    // This CPU test has no completion queue; drain the synthetic pin exactly
    // as the normal-read completion path would before retrying the gate.
    block.pending_rdma_reads.store(0, std::memory_order_release);
    checks.check(try_borrow_ec_write_source(&block),
                 "source borrow claims an unpinned block");
    checks.check(ec_write_source_borrowed(&block),
                 "source-borrowed bit is published in BlockHead");
    checks.check(acquire_normal_read_pin(&block) == 0,
                 "borrowed source rejects a new ordinary READ pin");
    checks.check(block.pending_rdma_reads.load(std::memory_order_acquire) == 0,
                 "rejected ordinary READ does not create a pin");

    // A failed try-lock must not clear a lock owned by another producer.
    block.rdma_read_pin_lock.test_and_set(std::memory_order_acquire);
    checks.check(!try_borrow_ec_write_source(&block),
                 "busy ordinary-READ gate rejects a source borrow");
    const bool remains_busy =
        block.rdma_read_pin_lock.test_and_set(std::memory_order_acquire);
    checks.check(remains_busy,
                 "failed source borrow preserves the busy pin lock");
    checks.check(block.normal_read_recovery_active.load(
                     std::memory_order_acquire) == 4 &&
                     ec_write_source_borrowed(&block),
                 "busy source-borrow retry preserves the borrow bit");
    block.rdma_read_pin_lock.clear(std::memory_order_release);

    // Existing recovery ownership bits are not overwritten by direct EC.
    block.normal_read_recovery_active.store(1, std::memory_order_release);
    checks.check(!try_borrow_ec_write_source(&block),
                 "direct source borrow does not overwrite READ-recovery bit");
    checks.check(block.normal_read_recovery_active.load(
                     std::memory_order_acquire) == 1 &&
                     normal_read_recovery_active(&block),
                 "READ-recovery bit remains observable after failed borrow");
    block.normal_read_recovery_active.store(2, std::memory_order_release);
    checks.check(!try_borrow_ec_write_source(&block),
                 "direct source borrow does not overwrite degraded-owner bit");
    checks.check(block.normal_read_recovery_active.load(
                     std::memory_order_acquire) == 2 &&
                     normal_read_recovery_active(&block),
                 "degraded-owner bit remains observable after failed borrow");
    block.normal_read_recovery_active.store(0, std::memory_order_release);

    // Fault-enabled admission can borrow a completed recovered object without
    // reopening its old ordinary-READ epoch. Live owners/pins still block it.
    block.normal_read_recovery_active.store(1, std::memory_order_release);
    checks.check(try_borrow_ec_write_source(&block, true),
                 "opt-in can borrow a recovered object with a closed READ gate");
    checks.check(block.normal_read_recovery_active.load() == 5,
                 "source borrow retains the closed READ gate");
    checks.check(acquire_normal_read_pin(&block) == 0,
                 "recovered borrowed source rejects late ordinary READ");
    release_ec_write_source(&block);
    checks.check(block.normal_read_recovery_active.load() == 1 &&
                 acquire_normal_read_pin(&block) == 0,
                 "source release does not reopen the recovered READ epoch");
    block.pending_rdma_reads.store(1);
    checks.check(!try_borrow_ec_write_source(&block, true),
                 "opt-in still waits for ordinary DMA pins");
    block.pending_rdma_reads.store(0);
    for (uint8_t flags : {uint8_t{2},uint8_t{3},uint8_t{4},uint8_t{5}}) {
        block.normal_read_recovery_active.store(flags);
        checks.check(!try_borrow_ec_write_source(&block, true) &&
                     block.normal_read_recovery_active.load() == flags,
                     "opt-in preserves an active recovery/write owner");
    }
    block.normal_read_recovery_active.store(0);

    release_ec_write_source(&block);
    checks.check(!ec_write_source_borrowed(&block),
                 "source release clears only the direct-borrow bit");
    checks.check(acquire_normal_read_pin(&block) != 0,
                 "ordinary READ pin is admitted after source release");
    checks.check(block.pending_rdma_reads.load(std::memory_order_acquire) == 1,
                 "post-release ordinary READ pin is accounted");
    block.pending_rdma_reads.store(0, std::memory_order_release);

    std::printf("EC_DIRECT_SOURCE_GATE checks=%d failures=%d\n", checks.count,
                checks.failures);
    if (checks.failures == 0) std::puts("EC_DIRECT_SOURCE_GATE_PASS");
    return checks.failures == 0 ? 0 : 1;
}

}  // namespace

#endif  // __has_include(<infiniband/verbs.h>)

int main() {
#if __has_include(<infiniband/verbs.h>)
    return run();
#else
    std::puts("EC_DIRECT_SOURCE_GATE_SKIP_NO_VERBS");
    return 0;
#endif
}
