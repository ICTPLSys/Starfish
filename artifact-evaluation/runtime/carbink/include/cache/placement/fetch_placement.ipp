#pragma once

namespace FarLib::cache {

inline ::FarLib::allocator::RegionPlacement
ConcurrentArrayCache::record_region_fetch_request(
    ::FarLib::allocator::RegionPlacement placement) {
    if (!region_fetch_diagnostics_enabled()) {
        return placement;
    }
    if (placement == ::FarLib::allocator::RegionPlacement::Resident) {
        region_fetch_requested_resident_count.fetch_add(
            1, std::memory_order_relaxed);
    } else if (placement == ::FarLib::allocator::RegionPlacement::Streaming) {
        region_fetch_requested_streaming_count.fetch_add(
            1, std::memory_order_relaxed);
    }
    return placement;
}

inline void ConcurrentArrayCache::record_region_fetch_actual(
    ::FarLib::allocator::RegionPlacement requested_placement,
    ::FarLib::allocator::RegionPlacement actual_placement, size_t size,
    ::FarLib::allocator::RegionHead *actual_region) {
    const bool diagnostics = region_fetch_diagnostics_enabled();
    if (diagnostics) {
        if (actual_placement ==
            ::FarLib::allocator::RegionPlacement::Resident) {
            region_fetch_actual_resident_count.fetch_add(
                1, std::memory_order_relaxed);
        } else if (actual_placement ==
                   ::FarLib::allocator::RegionPlacement::Streaming) {
            region_fetch_actual_streaming_count.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    if (requested_placement != ::FarLib::allocator::RegionPlacement::Resident ||
        actual_placement != ::FarLib::allocator::RegionPlacement::Streaming) {
        return;
    }
    const uint64_t footprint = local_allocation_footprint(size);
    const size_t bin = resident_allocation_bin(size);
    if (diagnostics) {
        region_fetch_resident_to_streaming_fallback_count.fetch_add(
            1, std::memory_order_relaxed);
        region_fetch_resident_to_streaming_fallback_bytes.fetch_add(
            footprint, std::memory_order_relaxed);
    }
    // These window counters and the per-Region intent are planner inputs,
    // not diagnostics.  A diagnostics-only environment switch must not
    // silently disable physical R/S reclassification.
    if (!region_fetch_hotness_placement_enabled() || !profile::is_working()) {
        return;
    }
    region_fetch_window_resident_fallback_count[bin].fetch_add(
        1, std::memory_order_relaxed);
    region_fetch_window_resident_fallback_bytes[bin].fetch_add(
        footprint, std::memory_order_relaxed);
    ASSERT(actual_region != nullptr);
    ASSERT(actual_region->load_placement() ==
           ::FarLib::allocator::RegionPlacement::Streaming);
    actual_region->profile_window_resident_fetch_intent_bytes.fetch_add(
        footprint, std::memory_order_relaxed);
}

inline ::FarLib::allocator::RegionPlacement
ConcurrentArrayCache::region_fetch_placement(const FarObjectEntry &entry) {
    const bool count_placement = region_fetch_diagnostics_enabled() &&
                                 region_resident_placement_enabled() &&
                                 profile::is_working();
    struct RegionPlacementCycleGuard {
        bool enabled;
        uint64_t start;
        ~RegionPlacementCycleGuard() {
            if (enabled) {
                profile::count_resident_profile_region_placement(get_cycles() -
                                                                 start);
            }
        }
    } placement_cycle_guard{count_placement,
                            count_placement ? get_cycles() : 0};
    if (region_hotness_placement_enabled()) {
        if (!region_fetch_hotness_placement_enabled() ||
            !resident_region_fetch_plan_ready.load(std::memory_order_acquire)) {
            // With no learned behavior history, a fetched object first
            // joins Streaming. Physical-region demand samples may still
            // promote the complete region on a later planner pass.
            return record_region_fetch_request(
                ::FarLib::allocator::RegionPlacement::Streaming);
        }
        const uint32_t group_id = entry.resident_group_id();
        if (group_id == 0) {
            return record_region_fetch_request(
                ::FarLib::allocator::RegionPlacement::Streaming);
        }
        ASSERT(group_id < next_resident_placement_group_id.load(
                              std::memory_order_acquire));
        const auto &group = resident_placement_groups[group_id];
        const bool below_resident_target =
            group.resident_current_bytes.load(std::memory_order_relaxed) <
            group.resident_target_bytes.load(std::memory_order_acquire);
        return record_region_fetch_request(
            group.fetch_preferred_resident.load(std::memory_order_acquire) &&
                    below_resident_target
                ? ::FarLib::allocator::RegionPlacement::Resident
                : ::FarLib::allocator::RegionPlacement::Streaming);
    }
    const uint32_t group_id = entry.resident_group_id();
    if (!region_resident_placement_enabled() || group_id == 0) {
        return ::FarLib::allocator::RegionPlacement::Unclassified;
    }
    if (!resident_profile_plan_ready.load(std::memory_order_acquire)) {
        return ::FarLib::allocator::RegionPlacement::Streaming;
    }
    ASSERT(group_id <
           next_resident_placement_group_id.load(std::memory_order_acquire));
    const auto &profile = resident_placement_groups[group_id];
    return profile.resident_current_bytes.load(std::memory_order_relaxed) <
                   profile.resident_target_bytes.load(std::memory_order_acquire)
               ? ::FarLib::allocator::RegionPlacement::Resident
               : ::FarLib::allocator::RegionPlacement::Streaming;
}

inline ConcurrentArrayCache::RemoteFetchPlacement
ConcurrentArrayCache::prepare_remote_fetch_placement(FarObjectEntry &entry,
                                                     size_t size,
                                                     DereferenceScope &scope,
                                                     bool allow_backup) {
    using ::FarLib::allocator::RegionPlacement;

    RemoteFetchPlacement placement{
        .requested_placement = region_fetch_placement(entry),
        .allocation_group_id = 0,
        .backup_policy_admitted = false,
        .backup_reserved = false,
    };
    placement.backup_policy_admitted =
        allow_backup && profiled_backup_should_admit(entry, size);
    const bool requested_placement_allows_backup =
        !region_resident_placement_enabled() ||
        placement.requested_placement != RegionPlacement::Resident;
    if (segmented_backup_enabled() && placement.backup_policy_admitted &&
        requested_placement_allows_backup) {
        placement.backup_reserved =
            reserve_segmented_remote_backup(size, scope);
        if (!profiled_backup_enabled()) {
            ASSERT(placement.backup_reserved);
        }
    }
    if (placement.requested_placement == RegionPlacement::Resident &&
        region_group_binding_enabled()) {
        placement.allocation_group_id = entry.resident_group_id();
    }
    return placement;
}

inline ::FarLib::allocator::RegionPlacement
ConcurrentArrayCache::finalize_remote_fetch_placement(
    FarObjectEntry &entry, size_t size, void *local_ptr,
    RemoteFetchPlacement &placement) {
    using ::FarLib::allocator::BlockHead;
    using ::FarLib::allocator::RegionPlacement;

    auto *local_block = static_cast<BlockHead *>(local_ptr) - 1;
    auto *actual_region = ::FarLib::allocator::block_to_region(local_block);
    const RegionPlacement actual_placement = actual_region->load_placement();
    if (region_resident_placement_enabled()) {
        if (actual_placement == RegionPlacement::Resident &&
            placement.backup_reserved) {
            release_remote_backup_budget(size);
            placement.backup_reserved = false;
        } else if (actual_placement == RegionPlacement::Streaming &&
                   !placement.backup_reserved && segmented_backup_enabled() &&
                   placement.backup_policy_admitted) {
            placement.backup_reserved = try_reserve_remote_backup(size, true);
        }
    }

    entry.set_resident_local(actual_placement == RegionPlacement::Resident);
    record_region_fetch_actual(placement.requested_placement, actual_placement,
                               size, actual_region);
    if (entry.is_resident_local() && !region_group_binding_enabled()) {
        add_profiled_resident_bytes(entry, local_allocation_footprint(size));
    }
    return actual_placement;
}

inline uint32_t ConcurrentArrayCache::register_resident_placement_group(
    uint32_t logical_owner_id, size_t size, uint64_t footprint_bytes) {
    if (region_hotness_placement_enabled() &&
        !region_fetch_hotness_placement_enabled()) {
        return 0;
    }
    if (!automatic_resident_group_planner_enabled()) {
        return 0;
    }
    const uint32_t allocation_bin = resident_allocation_bin(size);
    const uint64_t region_group_capacity =
        ((::FarLib::allocator::RegionSize -
          sizeof(::FarLib::allocator::RegionHead)) /
         footprint_bytes) *
        footprint_bytes;
    const uint64_t group_capacity = region_group_binding_enabled()
                                        ? region_group_capacity
                                        : resident_profile_group_bytes;
    ASSERT(group_capacity != 0);
    const uint64_t key =
        (static_cast<uint64_t>(logical_owner_id) << 32) | allocation_bin;
    std::lock_guard<std::mutex> lock(resident_placement_group_mutex);
    auto &active = active_resident_placement_groups[key];
    if (active.group_id == 0 ||
        (active.filled_bytes != 0 &&
         active.filled_bytes + footprint_bytes > group_capacity)) {
        const uint32_t group_id = next_resident_placement_group_id.fetch_add(
            1, std::memory_order_relaxed);
        ASSERT(group_id < kMaxResidentPlacementGroupCount);
        active = {.group_id = group_id, .filled_bytes = 0};
        auto &profile = resident_placement_groups[group_id];
        profile.logical_owner_id = logical_owner_id;
        profile.allocation_bin = allocation_bin;
    }
    active.filled_bytes += footprint_bytes;
    auto &profile = resident_placement_groups[active.group_id];
    profile.live_objects.fetch_add(1, std::memory_order_relaxed);
    profile.live_footprint_bytes.fetch_add(footprint_bytes,
                                           std::memory_order_relaxed);
    return active.group_id;
}

inline void ConcurrentArrayCache::release_resident_placement_allocation(
    const FarObjectEntry &entry, uint64_t footprint_bytes) {
    const uint32_t group_id = entry.resident_group_id();
    if (group_id == 0) {
        return;
    }
    ASSERT(group_id <
           next_resident_placement_group_id.load(std::memory_order_acquire));
    auto &profile = resident_placement_groups[group_id];
    const uint64_t previous_objects =
        profile.live_objects.fetch_sub(1, std::memory_order_acq_rel);
    const uint64_t previous_bytes = profile.live_footprint_bytes.fetch_sub(
        footprint_bytes, std::memory_order_acq_rel);
    ASSERT(previous_objects != 0);
    ASSERT(previous_bytes >= footprint_bytes);
}

inline void ConcurrentArrayCache::handle_region_placement_transition_backup(
    FarObjectEntry &entry, size_t bytes,
    ::FarLib::allocator::RegionPlacement from,
    ::FarLib::allocator::RegionPlacement to,
    std::vector<uint64_t> *deferred_remote_frees) {
    using ::FarLib::allocator::RegionPlacement;

    // GlobalHeap invokes this callback while the entry is invalid-locked.
    // Keep the behavior-group's actual R occupancy synchronized with the
    // physical Region transition before changing backup ownership.
    const size_t footprint = local_allocation_footprint(bytes);
    if (from == RegionPlacement::Streaming && to == RegionPlacement::Resident) {
        add_profiled_resident_bytes(entry, footprint);
    } else if (from == RegionPlacement::Resident &&
               to == RegionPlacement::Streaming) {
        release_profiled_resident_bytes(entry, footprint);
    }

    if (!selective_backup_enabled() || from != RegionPlacement::Streaming ||
        to != RegionPlacement::Resident) {
        return;
    }

    const auto locked_state = entry.load_state(std::memory_order_acquire);
    ASSERT(locked_state.invalid);
    ASSERT(locked_state.state != FETCHING && locked_state.state != EVICTING &&
           locked_state.state != BUSY && locked_state.state != REMOTE &&
           locked_state.state != FREE && locked_state.ref_cnt == 0);
    // The invalid bit is the lock owned by GlobalHeap. Re-locking here
    // would always fail and used to leave retained backups attached to
    // physical R objects. Mutate the backup fields under that existing
    // lock; GlobalHeap restores the original state after this callback.
    if (!entry.has_remote_backup_reservation()) {
        return;
    }
    const uint64_t remote_addr = entry.remote_addr();
    ASSERT(remote_addr != FarObjectEntry::RemoteAddrInvalid48);
    entry.set_remote_backup_reservation(false);
    entry.set_remote_invalid();
    release_remote_backup_budget(bytes);
    resident_profile_released_backup_bytes.fetch_add(bytes,
                                                     std::memory_order_relaxed);
    if (deferred_remote_frees != nullptr) {
        deferred_remote_frees->push_back(remote_addr);
    } else {
        remote_allocator.deallocate(remote_addr);
    }
}

}  // namespace FarLib::cache
