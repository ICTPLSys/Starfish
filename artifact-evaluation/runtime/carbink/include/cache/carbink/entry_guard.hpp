#pragma once

#include "cache/entry.hpp"
#include "utils/debug.hpp"
#include "utils/uthreads.hpp"

namespace FarLib::cache::carbink {

// Port of the original remote-span compaction guard to Hydra's stable page
// owner. One reference prevents free/re-eviction while a foreground fetch is
// still allowed. BUSY is held only for local address inspection/publication,
// never over transport, polling, or a fibre yield.
class EntryGuard {
public:
    EntryGuard() = default;
    EntryGuard(const EntryGuard &) = delete;
    EntryGuard &operator=(const EntryGuard &) = delete;
    ~EntryGuard() { release(); }

    bool acquire(FarObjectEntry *entry, uint64_t expected_remote) {
        if (entry_ != nullptr || entry == nullptr) return false;
        auto observed = entry->load_state();
        for (;;) {
            if (observed.state != REMOTE || observed.invalid ||
                observed.ref_cnt != 0) return false;
            auto locked = observed;
            locked.state = BUSY;
            locked.inc_ref_cnt();
            if (!entry->cas_state_weak(observed, locked)) continue;
            if (entry->remote_addr() != expected_remote) {
                restore_remote(entry, true);
                return false;
            }
            entry_ = entry;
            source_addr_ = expected_remote;
            restore_remote(entry, false);
            return true;
        }
    }

    bool try_lock_publish() {
        if (entry_ == nullptr || publish_locked_) return false;
        auto observed = entry_->load_state();
        for (;;) {
            if (observed.state != REMOTE || observed.invalid ||
                observed.ref_cnt != 1) return false;
            auto locked = observed;
            locked.state = BUSY;
            if (!entry_->cas_state_weak(observed, locked)) continue;
            if (entry_->remote_addr() != source_addr_) {
                restore_remote(entry_, false);
                return false;
            }
            publish_locked_ = true;
            return true;
        }
    }

    // Call only after allocator metadata has transferred under the pair claim.
    void publish_locked(uint64_t destination_addr) {
        if (!publish_locked_) ERROR("carbink: publish without owner move lock");
        entry_->set_remote_addr(destination_addr);
        restore_remote(entry_, false);
        publish_locked_ = false;
    }

    void cancel_publish() {
        if (!publish_locked_) return;
        restore_remote(entry_, false);
        publish_locked_ = false;
    }

    void release() {
        if (entry_ == nullptr) return;
        cancel_publish();
        auto observed = entry_->load_state();
        for (;;) {
            if (observed.invalid || observed.state == BUSY) {
                uthread::yield();
                observed = entry_->load_state();
                continue;
            }
            if (observed.ref_cnt == 0 || observed.state == FREE)
                ERROR("carbink: lost compaction owner reference");
            auto released = observed;
            released.dec_ref_cnt();
            if (entry_->cas_state_weak(observed, released)) break;
        }
        entry_ = nullptr;
    }

    FarObjectEntry *entry() const { return entry_; }

private:
    static void restore_remote(FarObjectEntry *entry, bool drop_reference) {
        // Hydra aggregate pins may change ref_cnt even while BUSY. Preserve
        // their latest increments/decrements rather than storing an old word.
        auto observed = entry->load_state();
        for (;;) {
            if (observed.state != BUSY || observed.invalid ||
                observed.ref_cnt == 0)
                ERROR("carbink: lost owner BUSY publication");
            auto remote = observed;
            remote.state = REMOTE;
            if (drop_reference) remote.dec_ref_cnt();
            if (entry->cas_state_weak(observed, remote)) return;
        }
    }

    FarObjectEntry *entry_ = nullptr;
    uint64_t source_addr_ = FarObjectEntry::RemoteAddrInvalid48;
    bool publish_locked_ = false;
};

}  // namespace FarLib::cache::carbink
