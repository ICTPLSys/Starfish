#pragma once
#include "cache/concurrent_cache.hpp"

namespace FarLib::cache {

inline EntryState ConcurrentArrayCache::try_mark(
    ::FarLib::allocator::BlockHead *block,
    profile::DualFrequencyHistogramSnapshot *frequency_histogram,
    bool update_ema_this_pass) {
    if (six_dirty_routing_enabled() &&
        ::FarLib::allocator::block_to_region(block)
            ->pending_allocation_publications.load(std::memory_order_acquire) != 0)
        return LOCAL;
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
            if (old_state.invalid) [[unlikely]] {
                return BUSY;
            }

            const auto &ft_config = ::FarLib::get_config();
            // EC (ft_method=ec_batch) candidate gate: every small object must
            // either be staged into an EC group or abort.  Staging readiness is
            // checked after the move-lock by stage_ec_batch_object(); a failed
            // candidate must never fall through to an unprotected flat write.
            const bool ec_batch_candidate =
                ft_config.is_ec_batch_mode() &&
                ft_config.ft_small_object(obj.size);

            // Acquire move-lock (invalid bit) to serialize remote_addr mutation and
            // to prevent "interrupt eviction" once remote allocation begins.
            auto lock_state = old_state;
            lock_state.invalid = 1;
            if (!entry->cas_state_weak(old_state, lock_state)) [[unlikely]] {
                goto retry;
            }
            uint64_t remote_addr = entry->remote_addr();

            // Revalidate the clean backup while holding the same move-lock used
            // for every remote-address mutation.  A recovered object can still
            // carry a reservation for its former data segment; if that segment
            // is on the failed endpoint, discard only that slot and continue
            // through the ordinary EC/flat writeback path below.  Other dead
            // segments in the EC group do not affect this entry's own address.
            if (selective_backup_enabled() && !lock_state.dirty &&
                remote_addr != FarObjectEntry::RemoteAddrInvalid48 &&
                entry->has_remote_backup_reservation()) {
                if (!reuse_ec_recovery_endpoint_is_dead(remote_addr)) {
                    auto remote_state = lock_state;
                    remote_state.invalid = 0;
                    remote_state.state = REMOTE;
                    entry->set_remote_backup_reservation(false);
                    six_commit_evict(*entry, obj.size, false);
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

                // The backup reservation belongs to this exact remote slot.
                // Release it once, then let the already-held move-lock carry
                // the object into normal writeback without reacquiring it or
                // changing the object to dirty merely because the endpoint died.
                entry->set_remote_backup_reservation(false);
                release_remote_backup_budget(obj.size);
                profile::count_remote_backup_invalidated(obj.size);
                entry->set_remote_invalid();
                remote_allocator.deallocate(remote_addr);
                remote_addr = FarObjectEntry::RemoteAddrInvalid48;
            }

            // Diagnostics only: count candidates that reach the writeback path.
            // Keep the clean-backup reuse fast path out of these write-gate
            // counters; it does not evaluate or stage an EC write.
            if (ft_config.is_ec_batch_mode()) {
                ec_candidate_checked_.fetch_add(1, std::memory_order_relaxed);
                if (ec_batch_candidate) {
                    ec_candidate_true_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            if (!ec_batch_candidate &&
                remote_addr == FarObjectEntry::RemoteAddrInvalid48) {
                auto alloc_start = get_cycles();
                const auto behavior_group = ::FarLib::simple_region_budget::six_enabled()
                    ? ((::FarLib::simple_region_heat::is_hot(
                            ::FarLib::simple_region_heat::class_for(
                                reinterpret_cast<uintptr_t>(block)))
                            ? 3u
                            : 0u) +
                       entry->simple_dirty_class())
                    : ::FarLib::simple_region_heat::grouping_enabled()
                    ? ::FarLib::simple_region_heat::class_for(reinterpret_cast<uintptr_t>(block))
                    : current_behavior_group(*entry);
                remote_addr = allocate_remote(obj.size, behavior_group);
                profile::count_excl_remote_alloc_cycles(get_cycles() - alloc_start);
                profile::count_excl_remote_alloc(obj.size);
                entry->set_remote_addr(remote_addr);
                if(::FarLib::simple_region_heat::grouping_enabled())
                    entry->set_simple_heat_hot(
                        ::FarLib::simple_region_heat::is_hot(
                            ::FarLib::simple_region_heat::fetch_class_hint(
                                behavior_group,
                                ::FarLib::simple_region_heat::remote_grouping_enabled()
                                    ? ::FarLib::allocator::remote::remote_global_heap.simple_class_for(remote_addr)
                                    : behavior_group)));
            }

            auto evict_state = lock_state;
            evict_state.invalid = 0;
            evict_state.dirty = 0;
            evict_state.state = EVICTING;
            evict_state.inc_ref_cnt();  // dec on rdma work completed
            entry->set_client_idx(rdma::thread_info.thread_id);
            if (six_dirty_routing_enabled())
                entry->set_six_pending_evict(old_state.dirty);
            if (!entry->cas_state_strong(lock_state, evict_state)) [[unlikely]] {
                goto retry;
            }
            record_backup_group_eviction(obj.size, old_state.dirty);
            record_logical_object_eviction(*entry, obj.size,
                                           old_state.dirty);
            // The object is staged only after the EVICTING transition, so the
            // entry already owns its write reference before the group that
            // contains it can be posted.
            const bool ec_batch_grouped =
                ec_batch_candidate &&
                stage_ec_batch_object(block->get_object_ptr(), obj.size, entry);
            // Diagnostics only: candidate true, but staging/grouping failed.
            if (ec_batch_candidate && !ec_batch_grouped) {
                ec_diag_stage_failed_.fetch_add(1, std::memory_order_relaxed);
                ERROR("ec_batch: candidate staging failed; refusing flat fallback");
            }
            if (ec_batch_grouped) {
                // The bytes are in the group's data slot and the entry's
                // remote_addr points at that segment; the six segment writes are
                // posted by flush_ec_batch_groups().  An address the entry was
                // re-homed away from (it already had a remote copy) is released
                // through the allocator, exactly as in the dirty branch below; a
                // candidate never allocates a flat slot before this point, so
                // remote_addr is either that old copy or unset.
                if (remote_addr != FarObjectEntry::RemoteAddrInvalid48) {
                    remote_allocator.deallocate(remote_addr);
                }
            } else if (!ec_batch_candidate &&
                       ft_config.ft_small_object(obj.size) &&
                       buffer_set.sponge_buffers != nullptr &&
                       remote_allocator.small_object_stripe_manager().owns(
                           remote_addr)) {
                // EC (ft_method=sponge): the object lives in an EC stripe
                // slot, so the data shard and both parity shards have to be
                // updated as one three-segment commit RPC (see
                // post_sponge_commit_requests); the EVICTING state is only
                // released by the final ACK.
                const auto &stripe_manager =
                    remote_allocator.small_object_stripe_manager();
                SmallObjectStripeManager::SlotLayout layout;
                if (!stripe_manager.get_slot_layout(remote_addr, &layout)) {
                    ERROR("EC small-object eviction found no stripe slot layout");
                }
                rdma::SpongeCommitRecord record;
                if (!rdma::sponge_commit_record_from_layout(
                        record, layout, block->get_object_ptr(), obj.size,
                        reinterpret_cast<uint64_t>(block->get_object_ptr()),
                        false, ft_config)) {
                    ERROR("EC small-object commit record build failed");
                }
                add_sponge_commit_request(
                    buffer_set, ft_config.map_remote_addr(remote_addr).first,
                    record);
            } else {
                if (remote_addr == FarObjectEntry::RemoteAddrInvalid48) {
                    // ec_batch candidate whose grouping failed: keep the flat
                    // write path alive by allocating after the transition.
                    auto alloc_start = get_cycles();
                    remote_addr = allocate_remote(obj.size);
                    profile::count_excl_remote_alloc_cycles(get_cycles() - alloc_start);
                    profile::count_excl_remote_alloc(obj.size);
                    entry->set_remote_addr(remote_addr);
                }
                add_write_request(buffer_set, block->get_object_ptr(),
                                  remote_addr, obj.size);
            }
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
            if (six_dirty_routing_enabled())
                entry->set_six_pending_evict(old_state.dirty);
            uint64_t remote_addr = entry->remote_addr();
            if (!entry->cas_state_weak(old_state, new_state)) [[unlikely]] {
                goto retry;
            }
            const auto &ft_config = ::FarLib::get_config();
            // EC (ft_method=ec_batch): a dirty object is re-homed into a fresh
            // group slot instead of being rewritten in place, so the RS(4,2)
            // parity of the group it used to live in can never go stale.  The
            // entry's remote_addr is set to the new group's data segment.
            // Staging happens after the EVICTING transition, so the entry
            // already owns its write reference before the group is postable.
            const bool ec_batch_candidate =
                ft_config.is_ec_batch_mode() &&
                ft_config.ft_small_object(obj.size);
            // Diagnostics only: does this branch even evaluate the EC gate?
            if (ft_config.is_ec_batch_mode()) {
                ec_candidate_checked_.fetch_add(1, std::memory_order_relaxed);
                if (ec_batch_candidate) {
                    ec_candidate_true_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            const bool ec_batch_grouped =
                ec_batch_candidate &&
                stage_ec_batch_object(block->get_object_ptr(), obj.size, entry);
            if (ec_batch_candidate && !ec_batch_grouped) {
                ec_diag_stage_failed_.fetch_add(1, std::memory_order_relaxed);
                ERROR("ec_batch: candidate staging failed; refusing flat fallback");
            }
            if (ec_batch_grouped) {
                // Nothing to buffer here: the six segment writes of the group
                // are posted by flush_ec_batch_groups().  The object now lives
                // in the new group's data segment instead of the address it
                // used before, so that address is released through the
                // allocator: a group segment is released per object (the group
                // goes back only once its last live object is gone), a plain
                // slot goes through mark_dead().  No allocator metadata is
                // touched here.
                if (remote_addr != FarObjectEntry::RemoteAddrInvalid48) {
                    remote_allocator.deallocate(remote_addr);
                }
            } else if (!ec_batch_candidate &&
                       ft_config.ft_small_object(obj.size) &&
                       buffer_set.sponge_buffers != nullptr &&
                       remote_allocator.small_object_stripe_manager().owns(
                           remote_addr)) {
                // EC (ft_method=sponge): same three-segment commit as in the
                // strict-exclusive branch above; the EVICTING state is only
                // released by the data server's final ACK.
                const auto &stripe_manager =
                    remote_allocator.small_object_stripe_manager();
                SmallObjectStripeManager::SlotLayout layout;
                if (!stripe_manager.get_slot_layout(remote_addr, &layout)) {
                    ERROR("EC small-object eviction found no stripe slot layout");
                }
                rdma::SpongeCommitRecord record;
                if (!rdma::sponge_commit_record_from_layout(
                        record, layout, block->get_object_ptr(), obj.size,
                        reinterpret_cast<uint64_t>(block->get_object_ptr()),
                        false, ft_config)) {
                    ERROR("EC small-object commit record build failed");
                }
                add_sponge_commit_request(
                    buffer_set, ft_config.map_remote_addr(remote_addr).first,
                    record);
            } else {
                add_write_request(buffer_set, block->get_object_ptr(),
                                  remote_addr, obj.size);
            }
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
            six_commit_evict(*entry, obj.size, false);
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
