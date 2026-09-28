#pragma once

// Shared interface for registered temporary buffers owned by worker threads.
//
// The EC write-side staging adapter deliberately depends on this small
// interface rather than on the read-recovery implementation.  That keeps the
// fixed-MR staging pool usable by itself while allowing recovery and eviction
// to bind to one grow-on-demand provider at startup.

#include <cstddef>

namespace FarLib::cache {

namespace ec_batch {
struct EcStagingGroupSlot;
}

class WorkerTempBufferProvider {
public:
    virtual ~WorkerTempBufferProvider() = default;

    virtual bool valid() const = 0;
    virtual size_t slot_size() const = 0;

    // The owner id is supplied by the caller's worker-local callback.  A
    // provider may use it to keep free lists and registration chunks local to
    // the worker that will consume the buffer.
    virtual bool acquire(ec_batch::EcStagingGroupSlot *out,
                         size_t owner_id) = 0;

    // Completion threads may be different from the acquiring worker.  The
    // provider is responsible for routing a returned lease to its immutable
    // owner without taking a global borrow/release mutex.
    virtual bool release(const ec_batch::EcStagingGroupSlot &slot) = 0;

    // Immutable ownership and active-lease validation used before posting a
    // WR.  `owns_buffer` additionally checks that [ptr, ptr + bytes) lies in
    // one of the six segments of `slot`.
    virtual bool owns_slot(const ec_batch::EcStagingGroupSlot &slot) const = 0;
    virtual bool owns_buffer(const ec_batch::EcStagingGroupSlot &slot,
                             const void *ptr, size_t bytes) const = 0;

    // Diagnostics are intentionally part of the provider contract so the
    // legacy EcStagingPool facade reports the same live shared pool when it is
    // bound to recovery scratch.
    virtual size_t depth() const = 0;
    virtual size_t bytes() const = 0;
    virtual size_t in_use() const = 0;
    virtual size_t available() const = 0;
    virtual size_t peak_in_use() const = 0;
    virtual size_t growths() const = 0;
    virtual size_t allocation_failures() const = 0;
    virtual size_t limit() const = 0;
};

}  // namespace FarLib::cache
