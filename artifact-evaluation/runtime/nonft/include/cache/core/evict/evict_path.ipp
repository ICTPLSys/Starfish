#pragma once
#include "cache/concurrent_cache.hpp"

namespace FarLib::cache {

inline EntryState ConcurrentArrayCache::try_mark(
    ::FarLib::allocator::BlockHead *block,
    profile::DualFrequencyHistogramSnapshot *frequency_histogram,
    bool update_ema_this_pass) {
    if (::FarLib::allocator::block_to_region(block)
            ->profile_pending_resident()) {
        resident_region_pending_mark_bypasses.fetch_add(
            1, std::memory_order_relaxed);
        return LOCAL;
    }
    uint32_t retry_count = 0;
    bool frequency_updated = false;
retry:
    if (++retry_count > 4096) {
        // Bound retries so one hot/churning object cannot stall an entire mark worker.
        return LOCAL;
    }
    if ((retry_count & 63u) == 0) {
        uthread::yield();
    }
    far_obj_t obj = block->obj_meta_data.load();
    if (obj.is_null()) {
        // this object is deallocated by user
        // mark it as garbage
        return FREE;
    }
    auto entry = &get_entry_of(obj);
    auto old_state = entry->load_state();
    if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
        auto spin_start = get_cycles();
        profile::count_excl_move_lock_spin();
        profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
        goto retry;
    }
    if (old_state.state == FREE ||
        entry->local_addr() != block->get_object_ptr()) [[unlikely]] {
        // The block no longer belongs to this entry (or was freed).
        // Retrying can livelock under concurrent migration/reuse.
        return FREE;
    }
    if (old_state.state == PINNED) {
        return old_state.state;
    }
    const auto region_placement =
        ::FarLib::allocator::block_to_region(block)->load_placement();
    const bool region_authoritative =
        region_resident_placement_enabled() &&
        region_placement !=
            ::FarLib::allocator::RegionPlacement::Unclassified &&
        region_placement !=
            ::FarLib::allocator::RegionPlacement::Mixed;
    if (region_authoritative) {
        if (region_placement ==
            ::FarLib::allocator::RegionPlacement::Resident) {
            return old_state.state;
        }
    } else {
        if (resident_local_budget_bytes != 0 &&
            entry->is_resident_local()) {
            if (try_demote_profiled_resident(*entry, obj.size)) {
                goto retry;
            }
            return old_state.state;
        }
        if (try_promote_profiled_resident(*entry, obj.size)) {
            return LOCAL;
        }
    }
    if (!frequency_updated) {
        const auto frequency_profile =
            update_frequency_profile_on_mark(*entry, update_ema_this_pass);
        if (frequency_histogram != nullptr && hybrid_profiling_enabled()) {
            frequency_histogram->window_histogram.add(
                frequency_profile.window_frequency);
            frequency_histogram->ema_histogram.add(
                frequency_profile.ema_frequency);
        }
        frequency_updated = true;
    }
    auto new_state = old_state;
    new_state.dec_hotness();
    if (new_state.can_evict()) {
        // mark this object
        new_state.state = MARKED;
        if (!entry->cas_state_weak(old_state, new_state)) [[unlikely]] {
            goto retry;
        }
        profile::count_mark(obj.size);
        return MARKED;
    } else {
        // not evictable, but its hotness is decreased
        if (!entry->cas_state_weak(old_state, new_state)) [[unlikely]] {
            goto retry;
        }
        profile::count_not_mark();
        return new_state.state;
    }
}

inline EntryState ConcurrentArrayCache::try_evict(
    ::FarLib::allocator::BlockHead *block, EvictBufferSet &buffer_set) {
    profile::count_evac_try_evict();
    struct TryEvictCycleGuard {
        bool enabled;
        uint64_t start;
        ~TryEvictCycleGuard() {
            if (enabled) {
                profile::count_evac_try_evict_cycles(
                    (int64_t)(get_cycles() - start));
            }
        }
    } cycle_guard{profile::evac_fine_profile_runtime_enabled(), 0};
    if (cycle_guard.enabled) {
        cycle_guard.start = get_cycles();
    }
    uint32_t retry_count = 0;
retry:
    if (++retry_count > 4096) {
        // Keep progress under extreme contention; retry in next cycle.
        return BUSY;
    }
    if ((retry_count & 63u) == 0) {
        uthread::yield();
    }
    far_obj_t obj = block->obj_meta_data.load();
    if (obj.is_null()) {
        // this object is deallocated by user
        profile::count_evac_try_evict_free_or_mismatch();
        return FREE;
    }
    auto entry = &get_entry_of(obj);
    auto old_state = entry->load_state();
    if (::FarLib::allocator::block_to_region(block)
            ->profile_pending_resident()) {
        if (old_state.state == MARKED) {
            auto local_state = old_state;
            local_state.state = LOCAL;
            if (!entry->cas_state_weak(old_state, local_state)) [[unlikely]] {
                goto retry;
            }
            resident_region_pending_unmarks.fetch_add(
                1, std::memory_order_relaxed);
            return LOCAL;
        }
        if (old_state.state == EVICTING) {
            resident_region_pending_evicting_seen.fetch_add(
                1, std::memory_order_relaxed);
        }
        return old_state.state;
    }
    if (old_state.state == PINNED) {
        profile::count_evac_try_evict_pinned();
        return PINNED;
    }
    if (old_state.state == FREE ||
        entry->local_addr() != block->get_object_ptr()) [[unlikely]] {
        profile::count_evac_try_evict_free_or_mismatch();
        return FREE;
    }
    auto new_state = old_state;
    switch (new_state.state) {
    case MARKED: {
        if (::FarLib::get_config().exclusive_cache) {
            if (selective_backup_enabled() && !old_state.dirty &&
                entry->has_remote() &&
                entry->has_remote_backup_reservation()) {
                if (old_state.invalid) [[unlikely]] {
                    return BUSY;
                }
                auto lock_state = old_state;
                lock_state.invalid = 1;
                if (!entry->cas_state_weak(old_state, lock_state)) [[unlikely]] {
                    goto retry;
                }

                auto remote_state = lock_state;
                remote_state.invalid = 0;
                remote_state.state = REMOTE;
                entry->set_remote_backup_reservation(false);
                auto expected = lock_state;
                ASSERT(entry->cas_state_strong(expected, remote_state));

                record_backup_group_eviction(obj.size, false);
                record_logical_object_eviction(*entry, obj.size, false);
                release_remote_backup_budget(obj.size);
                block->obj_meta_data.store(far_obj_t::null(),
                                           std::memory_order_relaxed);
                profile::count_remote_backup_reused(obj.size);
                profile::count_evac_try_evict_marked_clean();
                profile::count_obj_unmodifed();
                profile::count_clean_evict_bytes(obj.size);
                return FREE;
            }

            // Strict-exclusive: allocate remote at eviction time and always write.
            if (old_state.invalid) [[unlikely]] {
                return BUSY;
            }

            // Acquire move-lock (invalid bit) to serialize remote_addr mutation and
            // to prevent "interrupt eviction" once remote allocation begins.
            auto lock_state = old_state;
            lock_state.invalid = 1;
            if (!entry->cas_state_weak(old_state, lock_state)) [[unlikely]] {
                goto retry;
            }
            uint64_t remote_addr = entry->remote_addr();
            if (remote_addr == FarObjectEntry::RemoteAddrInvalid48) {
                auto alloc_start = get_cycles();
                remote_addr = allocate_remote(obj.size);
                profile::count_excl_remote_alloc_cycles(get_cycles() - alloc_start);
                profile::count_excl_remote_alloc(obj.size);
                entry->set_remote_addr(remote_addr);
            }

            auto evict_state = lock_state;
            evict_state.invalid = 0;
            evict_state.dirty = 0;
            evict_state.state = EVICTING;
            evict_state.inc_ref_cnt();  // dec on rdma work completed
            entry->set_client_idx(rdma::thread_info.thread_id);
            if (!entry->cas_state_weak(lock_state, evict_state)) [[unlikely]] {
                goto retry;
            }
            record_backup_group_eviction(obj.size, old_state.dirty);
            record_logical_object_eviction(*entry, obj.size,
                                           old_state.dirty);
            add_write_request(buffer_set, block->get_object_ptr(), remote_addr,
                              obj.size);
            profile::count_obj_modifed();
            if (old_state.dirty) {
                profile::count_dirty_evict_bytes(obj.size);
            } else {
                profile::count_clean_evict_bytes(obj.size);
            }
            return EVICTING;
        }

        if (old_state.dirty) {
            profile::count_evac_try_evict_marked_dirty();
            new_state.dirty = 0;
            new_state.state = EVICTING;
            new_state.inc_ref_cnt();  // dec on rdma work completed
            entry->set_client_idx(rdma::thread_info.thread_id);
            uint64_t remote_addr = entry->remote_addr();
            if (!entry->cas_state_weak(old_state, new_state)) [[unlikely]] {
                goto retry;
            }
            add_write_request(buffer_set, block->get_object_ptr(), remote_addr,
                              obj.size);
            profile::count_obj_modifed();
            profile::count_dirty_evict_bytes(obj.size);
            return EVICTING;
        } else {
            profile::count_evac_try_evict_marked_clean();
            assert(new_state.ref_cnt == 0);
            new_state.state = REMOTE;
            if (!entry->cas_state_weak(old_state, new_state)) [[unlikely]] {
                goto retry;
            }
            block->obj_meta_data.store(far_obj_t::null(), std::memory_order_relaxed);
            profile::count_obj_unmodifed();
            profile::count_clean_evict_bytes(obj.size);
            return FREE;
        }
        return new_state.state;
    }
    default:
        // this object is touched after marked
        // should not evict
        profile::count_evac_touched_after_mark();
        switch (new_state.state) {
        case LOCAL:
            profile::count_evac_try_evict_state_local();
            break;
        case FETCHING:
            profile::count_evac_try_evict_state_fetching();
            break;
        case REMOTE:
            profile::count_evac_try_evict_state_remote();
            break;
        case BUSY:
            profile::count_evac_try_evict_state_busy();
            break;
        default:
            profile::count_evac_try_evict_state_other();
            break;
        }
        return new_state.state;
    }
}

}  // namespace FarLib::cache
