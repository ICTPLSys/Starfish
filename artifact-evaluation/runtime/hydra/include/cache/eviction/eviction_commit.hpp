#pragma once

// Small bookkeeping boundary shared by the production failed-WR path and
// CPU-only tests. The caller must already hold the entry's invalid/move lock;
// this helper only turns one pending-eviction flag into at most one callback.

#include <cstdint>
#include <utility>

namespace FarLib::design2 {

// `take_pending` must atomically exchange the entry's pending flag with zero.
// `commit` receives the dirty bit (kind==2) only when commit_event is true.
// A second invocation after the first successful take is a no-op, which keeps
// duplicate CQEs from double-counting one logical eviction.
template <typename TakePending, typename Commit>
inline bool consume_six_pending_evict_once(TakePending &&take_pending,
                                           Commit &&commit,
                                           bool commit_event) {
    const uint8_t kind = static_cast<uint8_t>(
        std::forward<TakePending>(take_pending)());
    if (kind == 0) return false;
    if (commit_event) {
        std::forward<Commit>(commit)(kind == 2);
    }
    return true;
}

template <typename TakePending>
inline bool discard_six_pending_evict_once(TakePending &&take_pending) {
    return consume_six_pending_evict_once(
        std::forward<TakePending>(take_pending),
        [](bool) {}, false);
}

}  // namespace FarLib::design2
