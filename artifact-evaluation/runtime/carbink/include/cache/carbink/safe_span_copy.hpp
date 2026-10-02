#pragma once
#include <cstring>
#include "cache/entry.hpp"
#include "utils/debug.hpp"
#include "utils/uthreads.hpp"

namespace FarLib::cache::carbink {
// Data and parity must describe the identical immutable snapshot. A borrowed
// span rescued to LOCAL can be modified while its old WRITE is pending; unlike
// Hydra's single-page codeword, other pages in this group still need parity.
inline bool capture_span_for_recovery(FarObjectEntry &entry,
                                      const void *source, void *target,
                                      size_t bytes) {
    auto observed = entry.load_state(std::memory_order_acquire);
    for (;;) {
        if (observed.invalid || observed.state == EntryState::BUSY) {
            uthread::yield();
            observed = entry.load_state(std::memory_order_acquire);
            continue;
        }
        if (observed.ref_cnt == 0)
            ERROR("carbink: snapshot lost WRITE ownership");
        if (observed.state == EntryState::LOCAL) {
            // Foreground already rescued it. The caller cancels this member
            // before remote binding and retires its reserved WRITE reference.
            std::memset(target, 0, bytes);
            return false;
        }
        if (observed.state != EntryState::EVICTING)
            ERROR("carbink: unexpected snapshot entry state");
        auto locked = observed;
        locked.state = EntryState::BUSY;
        if (!entry.cas_state_weak(observed, locked)) continue;
        std::memcpy(target, source, bytes);
        auto current = entry.load_state(std::memory_order_acquire);
        for (;;) {
            if (current.invalid || current.state != EntryState::BUSY ||
                current.ref_cnt == 0)
                ERROR("carbink: lost snapshot BUSY ownership");
            auto released = current;
            released.state = EntryState::EVICTING;
            if (entry.cas_state_weak(current, released)) break;
        }
        return true;
    }
}
} // namespace FarLib::cache::carbink
