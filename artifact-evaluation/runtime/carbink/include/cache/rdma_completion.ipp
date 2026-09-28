#pragma once
#include "cache/concurrent_cache.hpp"

namespace FarLib::cache {

inline void ConcurrentArrayCache::handle_rdma_write_complete(const ibv_wc &wc) {
    struct WriteCompleteCycleGuard {
        bool enabled;
        uint64_t start;
        ~WriteCompleteCycleGuard() {
            if (enabled) {
                profile::count_evac_write_complete_cycles(
                    (int64_t)(get_cycles() - start));
            }
        }
    } cycle_guard{profile::evac_fine_profile_runtime_enabled(), 0};
    if (cycle_guard.enabled) {
        cycle_guard.start = get_cycles();
    }
    void *local_ptr = reinterpret_cast<void *>(wc.wr_id);
    ::FarLib::allocator::BlockHead *block =
        static_cast<::FarLib::allocator::BlockHead *>(local_ptr) - 1;
retry:
    auto obj = block->obj_meta_data.load();
    auto &entry = get_entry_of(obj);
    auto old_state = entry.load_state();

    if (old_state.state == PINNED) return;
    // A deallocator has consumed the completion's ownership once FREE is
    // visible. Retrying cannot make progress and would spin forever.
    if (old_state.state == FREE) [[unlikely]] return;
    if (entry.local_addr() != local_ptr) [[unlikely]]
        goto retry;
    if ((old_state.state == EntryState::LOCAL ||
         old_state.state == EntryState::MARKED) &&
        old_state.ref_cnt == 0) {
        // A stale write completion can arrive after an interrupted eviction
        // has already been converted back to LOCAL and re-marked for a
        // future eviction cycle. The completion no longer owns a write ref
        // in this state, so it should not mutate the entry.
        return;
    }
    auto new_state = old_state;
    // each RDMA write will increase the reference count
    // to avoid deallocating the local buffer while RDMA write is not done
    new_state.dec_ref_cnt();
    switch (new_state.state) {
    case EntryState::EVICTING:
        if (new_state.ref_cnt == 0) {
            new_state.state = REMOTE;
            if (!entry.cas_state_weak(old_state, new_state)) goto retry;
            block->obj_meta_data = far_obj_t::null();
            // The completed block remains linked from marked_list until a
            // Region reclaim pass. Advertise it exactly like an explicit
            // local deallocation so shutdown cannot strand null marked
            // blocks after the final CQ drain.
            ::FarLib::allocator::block_to_region(block)
                ->pending_reclaims.fetch_add(
                    1, std::memory_order_release);
        } else {
            // maybe another rdma request (write back) in the progress
            // do not mark it as remote to prevent reallocating memory to
            // other object
            if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        }
        break;
    case EntryState::LOCAL:
        // another access interrupted the eviction
        // just save the ref count and return
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        if (new_state.ref_cnt != 0) {
            // Another write still owns the local buffer and remote slot.
            break;
        }
        if (::FarLib::get_config().exclusive_cache) {
            // Corner case: eviction was interrupted and the object stayed LOCAL.
            // The remote slot allocated for this eviction must be freed (after the
            // write completes) to preserve strict-exclusivity (LOCAL => no remote).
            while (true) {
                auto cleanup_state = entry.load_state();
                if (cleanup_state.state != EntryState::LOCAL &&
                    cleanup_state.state != EntryState::MARKED) {
                    // A new eviction or ownership move has taken over.
                    return;
                }
                if (cleanup_state.invalid) {
                    uthread::yield();
                    continue;
                }
                auto lock_state = cleanup_state;
                lock_state.invalid = 1;
                if (!entry.cas_state_weak(cleanup_state, lock_state)) {
                    continue;
                }
                uint64_t old_remote = entry.remote_addr();
                if (old_remote != FarObjectEntry::RemoteAddrInvalid48) {
                    entry.set_remote_invalid();
                    remote_allocator.deallocate(old_remote);
                    profile::count_excl_interrupted_evict_free(obj.size);
                }
                auto unlock_state = entry.load_state();
                while (unlock_state.invalid) {
                    auto final_state = unlock_state;
                    final_state.invalid = 0;
                    if (entry.cas_state_weak(unlock_state, final_state)) {
                        return;
                    }
                }
                return;
            }
        }
        break;
    case EntryState::BUSY:
        goto retry;
    default:
        ERROR("check_cq: invalid state when evict");
    }
}

inline void ConcurrentArrayCache::handle_rdma_read_complete(const ibv_wc &wc) {
    void *local_ptr = reinterpret_cast<void *>(wc.wr_id);
    ::FarLib::allocator::BlockHead *block =
        static_cast<::FarLib::allocator::BlockHead *>(local_ptr) - 1;
retry:
    auto obj = block->obj_meta_data.load();
    auto &entry = get_entry_of(obj);
    auto old_state = entry.load_state();
    if (old_state.state == FREE || entry.local_addr() != local_ptr) [[unlikely]] {
        goto retry;
    }
    if (old_state.state == EntryState::BUSY) [[unlikely]] {
        goto retry;
    }
    assert(old_state.state == EntryState::FETCHING);
    if (::FarLib::get_config().exclusive_cache) {
        if (old_state.invalid) [[unlikely]] {
            auto spin_start = get_cycles();
            profile::count_excl_move_lock_spin();
            profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
            goto retry;
        }

        auto lock_state = old_state;
        lock_state.invalid = 1;
        if (!entry.cas_state_weak(old_state, lock_state)) [[unlikely]] {
            goto retry;
        }

        uint64_t old_remote = entry.remote_addr();
        // T7: Invariant check - fetch from remote must have a valid remote address
        if (old_remote == FarObjectEntry::RemoteAddrInvalid48) {
            std::string err = "FETCH_FAIL: remote_addr is invalid for obj=" +
                              std::to_string(obj.obj_id) +
                              " state=" + std::to_string(old_state.state) + "\n";
            EXCLUSIVE_ERROR(err);
            std::abort();
        }

        bool retain_backup = entry.has_remote_backup_reservation();
        const bool placement_allows_backup =
            local_placement_allows_remote_backup(entry);
        if (retain_backup &&
            (lock_state.dirty || !placement_allows_backup)) {
            entry.set_remote_backup_reservation(false);
            release_remote_backup_budget(obj.size);
            if (lock_state.dirty) {
                profile::count_remote_backup_invalidated(obj.size);
            } else {
                resident_profile_released_backup_bytes.fetch_add(
                    obj.size, std::memory_order_relaxed);
            }
            retain_backup = false;
        } else if (!retain_backup && !lock_state.dirty &&
                   placement_allows_backup &&
                   selective_backup_enabled() &&
                   !segmented_backup_enabled()) {
            retain_backup = try_reserve_remote_backup(obj.size);
            entry.set_remote_backup_reservation(retain_backup);
        }
        if (!retain_backup) {
            entry.set_remote_backup_reservation(false);
            entry.set_remote_invalid();
        }

        auto new_state = lock_state;
        new_state.invalid = 0;
        new_state.state = LOCAL;
        auto expected = lock_state;
        ASSERT(entry.cas_state_strong(expected, new_state));

        if (retain_backup) {
            return;
        }

        auto free_start = get_cycles();
        remote_allocator.deallocate(old_remote);
        auto remote_free_cycles = get_cycles() - free_start;
        profile::count_excl_remote_free_on_fetch_cycles(remote_free_cycles);
        profile::count_excl_remote_free_on_fetch(obj.size);
        constexpr int64_t RemoteFreeLogThreshold = 10000000;  // 10M cycles
        if (remote_free_cycles > RemoteFreeLogThreshold) {
            profile::count_excl_remote_free_on_fetch_slow(remote_free_cycles,
                                                          obj.size);
        }
    } else {
        auto new_state = old_state;
        new_state.state = LOCAL;
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
    }
}

inline void ConcurrentArrayCache::handle_work_complete(const ibv_wc &wc) {
    if (wc.status != IBV_WC_SUCCESS) {
        static std::atomic<int> err_wc_log{0};
        if (err_wc_log.fetch_add(1, std::memory_order_relaxed) < 10) {
            std::cerr << "ERROR: rdma wc status=" << wc.status
                      << " opcode=" << wc.opcode << " wr_id=" << wc.wr_id
                      << " qp_num=" << wc.qp_num << std::endl;
        }
        return;
    }

    // Skip invalid opcodes (garbage from wrong CQ)
    if (wc.opcode > IBV_WC_RECV) {
        return;
    }

    if (wc.opcode == IBV_WC_RDMA_WRITE) {
        handle_rdma_write_complete(wc);
    } else if (wc.opcode == IBV_WC_RDMA_READ) {
        handle_rdma_read_complete(wc);
    } else if (wc.opcode == IBV_WC_SEND || wc.opcode == IBV_WC_RECV) {
        // Control QP operations - ignore
    } else {
        static std::atomic<int> bad_wc_log{0};
        if (bad_wc_log.fetch_add(1, std::memory_order_relaxed) < 10) {
            std::cerr << "WARN: unexpected rdma wc opcode: " << wc.opcode
                      << " status=" << wc.status << " wr_id=" << wc.wr_id
                      << " qp_num=" << wc.qp_num
                      << " wc_flags=" << wc.wc_flags
                      << " byte_len=" << wc.byte_len << std::endl;
        }
    }
}

}  // namespace FarLib::cache
