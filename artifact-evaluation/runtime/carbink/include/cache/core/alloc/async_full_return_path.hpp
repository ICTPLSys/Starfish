#pragma once
#include "cache/region_based_allocator.hpp"

namespace FarLib::allocator {

inline bool GlobalHeap::async_full_returns_enabled() {
    static const bool enabled = [] {
        const char *v = std::getenv("FARLIB_ASYNC_FULL_RETURNS");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

inline void GlobalHeap::submit_full_regions_batch(RegionHead **regions,
                                                  size_t count) {
    if (count == 0) return;
    assert(count <= RegionReturnBatchSize);
    if (async_full_returns_enabled()) {
        bool wake = false;
        if (async_full_returns_.enqueue(regions, count, wake)) {
            // The batch remains IN_USE/unlinked. Queue ownership, rather than
            // a mutator slot, now keeps it alive until the original publisher.
            // notify_all takes eviction_mutex; never call it under ingress.
            if (wake) {
                async_full_return_wakes_.fetch_add(1, std::memory_order_relaxed);
                on_memory_low();
            }
            return;
        }
    }
    // Full queue: enqueue accepted nothing. Preserve the synchronous fallback.
    return_back_full_regions_batch(regions, count);
}

inline bool GlobalHeap::has_async_full_returns() const {
    return async_full_returns_enabled() && async_full_returns_.pending();
}

inline size_t GlobalHeap::drain_async_full_returns(size_t limit) {
    if (!has_async_full_returns()) return 0;
    const uint64_t start = get_cycles();
    size_t total = 0;
    RegionHead *batch[RegionReturnBatchSize];
    while (total < limit) {
        const size_t count = async_full_returns_.take(
            batch, std::min(RegionReturnBatchSize, limit - total));
        if (count == 0) break;
        // take released ingress before the original FULL-list/placement locks.
        return_back_full_regions_batch(batch, count);
        async_full_returns_.complete(count);
        total += count;
    }
    async_full_return_drain_cycles_.fetch_add(get_cycles() - start,
                                             std::memory_order_relaxed);
    return total;
}

inline void GlobalHeap::print_async_full_returns(const char *phase,
                                                 size_t fibres) const {
    if (!async_full_returns_enabled()) return;
    const auto s = async_full_returns_.snapshot();
    std::cerr << "allocator.async_full_returns phase=" << phase
              << " fibres=" << fibres << " accepted=" << s.accepted
              << " published=" << s.published << " fallback=" << s.fallback
              << " pending=" << s.pending << " inflight=" << s.inflight
              << " high_water=" << s.high_water
              << " wakes=" << async_full_return_wakes_.load()
              << " drain_cycles=" << async_full_return_drain_cycles_.load()
              << std::endl;
}

inline void GlobalHeap::finish_async_full_returns_for_shutdown() {
    if (!async_full_returns_enabled()) return;
    // Called after every producer and background consumer has quiesced and
    // the remaining ThreadHeap-owned regions have been released.
    while (drain_async_full_returns(512) != 0) {}
    const auto s = async_full_returns_.snapshot();
    if (s.pending != 0 || s.inflight != 0 || s.accepted != s.published)
        ERROR("async full-return shutdown ownership mismatch");
    print_async_full_returns("shutdown");
}

}  // namespace FarLib::allocator
