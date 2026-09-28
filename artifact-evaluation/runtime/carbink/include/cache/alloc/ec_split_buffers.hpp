#pragma once

// Lazy, size-classed registered scratch buffers for large-object EC splits.
// Each allocator bin has its own WorkerTempBufferPool, so a worker-local
// acquire/release path never contends on a new global buffer-pool mutex.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "recovery/ec_recovery_scratch.hpp"

#if __has_include(<infiniband/verbs.h>)
#include "cache/region_based_allocator.hpp"
#define FARLIB_EC_SPLIT_HAS_ALLOCATOR_BINS 1
#endif

namespace FarLib::cache::ec_split {

using WorkerTempBufferPool = ec_read_recovery::WorkerTempBufferPool;

inline constexpr size_t kMaxSplitSlotSize = 64 * 1024;
// Keep this header usable by CPU-only codec tests: region_based_allocator.hpp
// pulls in the RDMA verbs headers.  A full build uses the allocator's own
// physical bin table; the fallback below is the same table (BinSize[] entries
// multiplied by sizeof(void *)) for a standalone test build without verbs.
#if defined(FARLIB_EC_SPLIT_HAS_ALLOCATOR_BINS)
inline constexpr size_t kSplitPoolClassCount =
    ::FarLib::allocator::RegionBinCount;
inline size_t split_bin_size(size_t bin) {
    return ::FarLib::allocator::get_bin_size(bin);
}
#else
inline constexpr std::array<size_t, 56> kSplitBinSizes = {
    8,      16,     24,     32,     40,     48,     56,     64,
    80,     96,     112,    128,    160,    192,    224,    256,
    320,    384,    448,    512,    640,    768,    896,    1024,
    1280,   1536,   1792,   2048,   2560,   3072,   3584,   4096,
    5120,   6144,   7168,   8192,   10240,  12288,  14336,  16384,
    20480,  24576,  28672,  32768,  40960,  49152,  57344,  65536,
    81920,  98304,  114688, 131072, 163840, 196608, 229376, 262144};
inline constexpr size_t kSplitPoolClassCount = kSplitBinSizes.size();
inline constexpr size_t split_bin_size(size_t bin) {
    return kSplitBinSizes[bin];
}
#endif

class BufferPools final {
private:
    struct PoolCookie {
        BufferPools *owner = nullptr;
        size_t bin = 0;
    };

    static constexpr size_t kInvalidBin =
        std::numeric_limits<size_t>::max();

    static size_t bin_for_request(size_t requested) {
        if (requested == 0 || requested > kMaxSplitSlotSize) {
            return kInvalidBin;
        }
        for (size_t bin = 0; bin < kSplitPoolClassCount; ++bin) {
            const size_t slot_size = split_bin_size(bin);
            if (slot_size >= requested) {
                return slot_size <= kMaxSplitSlotSize ? bin : kInvalidBin;
            }
        }
        return kInvalidBin;
    }

    PoolCookie *cookie_for(uintptr_t raw) {
        // `raw` is caller-provided lease metadata.  Do all integer range,
        // alignment, and index checks before forming or dereferencing a
        // PoolCookie pointer; in particular, never probe an arbitrary address
        // while trying to discover which bin it might represent.
        if (!initialized_ || raw == 0) return nullptr;
        const uintptr_t begin =
            reinterpret_cast<uintptr_t>(cookies_.data());
        if (raw < begin) return nullptr;
        const uintptr_t offset = raw - begin;
        if (offset >= sizeof(cookies_) ||
            offset % sizeof(PoolCookie) != 0) {
            return nullptr;
        }
        const size_t bin = static_cast<size_t>(offset / sizeof(PoolCookie));
        if (bin >= kSplitPoolClassCount) return nullptr;
        PoolCookie *cookie = cookies_.data() + bin;
        if (cookie->owner != this || cookie->bin != bin ||
            !pools_[bin].valid()) {
            return nullptr;
        }
        return cookie;
    }

    const PoolCookie *cookie_for(uintptr_t raw) const {
        return const_cast<BufferPools *>(this)->cookie_for(raw);
    }

    static void forward_to_pool(const PoolCookie &cookie,
                                const ec_batch::EcStagingGroupSlot &in,
                                ec_batch::EcStagingGroupSlot *out) {
        *out = in;
        out->pool_cookie = reinterpret_cast<uintptr_t>(&cookie.owner->pools_[cookie.bin]);
    }

public:
    using Allocate = WorkerTempBufferPool::Allocate;
    using Release = WorkerTempBufferPool::Release;

    BufferPools() = default;
    BufferPools(const BufferPools &) = delete;
    BufferPools &operator=(const BufferPools &) = delete;

    // Initialize metadata for every split-capable allocator size class.  No
    // registered memory is allocated here: each worker pool grows its own
    // registered chunk lazily on the first acquire for that class.
    bool init(void *context, Allocate allocate, Release release) {
        if (initialized_ || allocate == nullptr || release == nullptr) {
            return false;
        }
        context_ = context;
        allocate_ = allocate;
        release_ = release;
        for (size_t bin = 0; bin < kSplitPoolClassCount; ++bin) {
            cookies_[bin].owner = this;
            cookies_[bin].bin = bin;
            const size_t slot_size = split_bin_size(bin);
            if (slot_size > kMaxSplitSlotSize) continue;
            if (!pools_[bin].init(slot_size, context_, allocate_, release_, 0,
                                  kMaxSplitSlotSize)) {
                return false;
            }
        }
        initialized_ = true;
        return true;
    }

    bool valid() const noexcept { return initialized_; }

    // Acquire from the allocator bin containing `requested_slot_size`.  The
    // returned lease carries a BufferPools cookie; release/ownership checks
    // recover the bin without a global lock and then restore the underlying
    // WorkerTempBufferPool cookie for its immutable lease validation.
    bool acquire(size_t requested_slot_size, size_t owner,
                 ec_batch::EcStagingGroupSlot *out) {
        if (out == nullptr || !initialized_) return false;
        const size_t bin = bin_for_request(requested_slot_size);
        if (bin == kInvalidBin) return false;
        if (!pools_[bin].acquire(out, owner)) return false;
        out->pool_cookie = reinterpret_cast<uintptr_t>(&cookies_[bin]);
        return true;
    }

    bool release(const ec_batch::EcStagingGroupSlot &slot) {
        const PoolCookie *cookie = cookie_for(slot.pool_cookie);
        if (cookie == nullptr) return false;
        ec_batch::EcStagingGroupSlot forwarded;
        forward_to_pool(*cookie, slot, &forwarded);
        return cookie->owner->pools_[cookie->bin].release(forwarded);
    }

    bool owns_slot(const ec_batch::EcStagingGroupSlot &slot) const {
        const PoolCookie *cookie = cookie_for(slot.pool_cookie);
        if (cookie == nullptr) return false;
        ec_batch::EcStagingGroupSlot forwarded;
        forward_to_pool(*cookie, slot, &forwarded);
        return cookie->owner->pools_[cookie->bin].owns_slot(forwarded);
    }

    bool owns_buffer(const ec_batch::EcStagingGroupSlot &slot,
                     const void *ptr, size_t bytes) const {
        const PoolCookie *cookie = cookie_for(slot.pool_cookie);
        if (cookie == nullptr) return false;
        ec_batch::EcStagingGroupSlot forwarded;
        forward_to_pool(*cookie, slot, &forwarded);
        return cookie->owner->pools_[cookie->bin].owns_buffer(forwarded, ptr,
                                                               bytes);
    }

    size_t in_use() const noexcept {
        size_t total = 0;
        for (const auto &pool : pools_) total += pool.in_use();
        return total;
    }

    size_t bytes() const noexcept {
        size_t total = 0;
        for (const auto &pool : pools_) total += pool.bytes();
        return total;
    }

    size_t growths() const noexcept {
        size_t total = 0;
        for (const auto &pool : pools_) total += pool.growths();
        return total;
    }

    size_t available() const noexcept {
        size_t total = 0;
        for (const auto &pool : pools_) total += pool.available();
        return total;
    }

    size_t allocation_failures() const noexcept {
        size_t total = 0;
        for (const auto &pool : pools_) total += pool.allocation_failures();
        return total;
    }

private:
    void *context_ = nullptr;
    Allocate allocate_ = nullptr;
    Release release_ = nullptr;
    bool initialized_ = false;
    std::array<WorkerTempBufferPool, kSplitPoolClassCount> pools_{};
    std::array<PoolCookie, kSplitPoolClassCount> cookies_{};
};

}  // namespace FarLib::cache::ec_split

#if defined(FARLIB_EC_SPLIT_HAS_ALLOCATOR_BINS)
#undef FARLIB_EC_SPLIT_HAS_ALLOCATOR_BINS
#endif
