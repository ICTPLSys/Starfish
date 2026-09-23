#pragma once
#include "cache/region_based_allocator.hpp"
#include "utils/inclusive_reclaim_diag.hpp"
#include "utils/wait_trace.hpp"

namespace FarLib::allocator {

inline bool optimized_skip_mark_pending_enabled() {
    static const bool enabled =
        std::getenv("FARLIB_OPT_SKIP_MARK_PENDING") != nullptr;
    return enabled;
}

inline bool optimized_mark_after_evict_enabled() {
    static const bool enabled =
        std::getenv("FARLIB_OPT_MARK_AFTER_EVICT") != nullptr;
    return enabled;
}

inline bool optimized_scan_past_blocked_region_enabled() {
    static const bool enabled =
        std::getenv("FARLIB_OPT_SCAN_PAST_BLOCKED_REGION") != nullptr;
    return enabled;
}

inline bool mark_region_admissible(
    RegionHead *region, EvacuationEligibility eligibility) {
    return eligibility != EvacuationEligibility::LegacyConcurrent ||
           region->marked_list == nullptr;
}

inline bool evict_region_admissible(
    RegionHead *region, EvacuationEligibility eligibility) {
    return eligibility != EvacuationEligibility::LegacyConcurrent ||
           region->marked_list != nullptr || region->is_empty();
}

namespace detail {

inline void emit_ready_task(std::vector<EvictTask> *ready_tasks,
                            RegionHead *region, uint32_t timestamp) {
    ready_tasks->emplace_back(region, nullptr, timestamp, true);
}

template <typename ReadyTaskSink>
inline void emit_ready_task(ReadyTaskSink *ready_tasks, RegionHead *region,
                            uint32_t timestamp) {
    ready_tasks->push_ready_task(
        EvictTask(region, nullptr, timestamp, true));
}

}  // namespace detail

inline bool GlobalHeap::adopt_collected_evict_task(EvictTask &task) {
    RegionHead *region = task.region;
    // Already-owned ready tasks must not be reserved a second time.
    if (region == nullptr || task.source_list == nullptr) {
        return task.reserved_free_size && region != nullptr;
    }
    // Failure leaves the collector task untouched for exactly one requeue.
    if (!task.placement_still_matches() ||
        region->load_placement() == RegionPlacement::Resident ||
        (region->profile_pending_resident() && region->marked_list == nullptr) ||
        region->state.load(std::memory_order_relaxed) == IN_USE) {
        return false;
    }
    // Collector-pop keeps these bytes in global free_size. Reserve the full
    // current value before the owned worker republishes final capacity.
    const size_t origin_free_size = region->free_size();
    if (origin_free_size != 0) {
        add_free_size(-static_cast<int64_t>(origin_free_size));
    }
    task.reserved_free_size = true;
    // Requeue must classify current placement/state, not the old source list.
    task.source_list = nullptr;
    return true;
}

inline GlobalHeap::GlobalHeap()
    : unallocated_offset(0),
      heap_size(0),
      memory_low_water_mark(0),
      memory_high_water_mark(0),
      memory_release_water_mark(0),
      heap(nullptr),
      is_dead(false),
      free_size(0),
      next_placement_epoch(1),
      resident_reserved_regions(0),
      streaming_reserved_regions(0),
      resident_counted_free_size(0),
      resident_region_budget(0),
      region_placement_enabled(false) {}

inline void GlobalHeap::register_heap(void *buffer, size_t size) {
    ASSERT(heap == nullptr);
    unallocated_offset = 0;
    heap_size = size;
    heap = buffer;
    free_size = heap_size;
    region_placement_enabled =
        FarLib::get_config().enable_region_resident_placement;
    ASSERT(!region_placement_enabled ||
           !FarLib::get_config().region_placement_bind_groups);
    resident_region_budget = region_placement_enabled
                                 ? std::min<size_t>(
                                       heap_size,
                                       FarLib::get_config()
                                           .local_resident_budget_bytes) /
                                       RegionSize
                                 : 0;
    resident_reserved_regions.store(0, std::memory_order_relaxed);
    streaming_reserved_regions.store(0, std::memory_order_relaxed);
    resident_counted_free_size.store(0, std::memory_order_relaxed);
    size_t low_pct = FarLib::get_config().evacuate_low_watermark_pct;
    if (low_pct > 95) low_pct = 95;
    if (low_pct < 1) low_pct = 1;
    size_t high_pct = FarLib::get_config().evacuate_high_watermark_pct;
    if (high_pct > 99) high_pct = 99;
    if (high_pct <= low_pct) {
        high_pct = std::min<size_t>(99, low_pct + 10);
    }
    size_t release_pct = FarLib::get_config().evacuate_release_watermark_pct;
    if (release_pct > 99) release_pct = 99;
    if (release_pct < 1) release_pct = 1;
    // Watermark percentages are defined over the entire local heap. R/S
    // placement changes which regions can be evicted, not the meaning of the
    // configured global memory-pressure thresholds.
    memory_low_water_mark =
        (int64_t)((heap_size * low_pct) / 100);
    memory_high_water_mark =
        (int64_t)((heap_size * high_pct) / 100);
    memory_release_water_mark =
        (int64_t)((heap_size * release_pct) / 100);
}

inline void GlobalHeap::destroy() { is_dead.store(true); }
inline bool GlobalHeap::dead() const { return is_dead.load(); }

inline void GlobalHeap::set_on_memory_low(std::function<void(void)> fn) {
    on_memory_low = std::move(fn);
}

inline bool GlobalHeap::need_evacuate(bool evacuator_waiting) {
    bool low = free_size.load(std::memory_order_relaxed) <
               memory_low_water_mark;
    return low || evacuator_waiting;
}

inline uint64_t GlobalHeap::allocate_placement_epoch() {
    return next_placement_epoch.fetch_add(1, std::memory_order_relaxed);
}

inline void GlobalHeap::account_region_claim(RegionPlacement placement) {
    if (placement == RegionPlacement::Resident) {
        resident_reserved_regions.fetch_add(1, std::memory_order_relaxed);
    } else if (placement == RegionPlacement::Streaming) {
        streaming_reserved_regions.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void GlobalHeap::account_region_release(RegionPlacement placement) {
    if (placement == RegionPlacement::Resident) {
        const size_t previous =
            resident_reserved_regions.fetch_sub(1, std::memory_order_relaxed);
        ASSERT(previous != 0);
    } else if (placement == RegionPlacement::Streaming) {
        const size_t previous = streaming_reserved_regions.fetch_sub(
            1, std::memory_order_relaxed);
        ASSERT(previous != 0);
    }
}

inline void GlobalHeap::add_resident_counted_free_size(int64_t bytes) {
    if (bytes != 0) {
        resident_counted_free_size.fetch_add(bytes,
                                             std::memory_order_relaxed);
    }
}

inline void GlobalHeap::sub_resident_counted_free_size(int64_t bytes) {
    if (bytes != 0) {
        const int64_t previous = resident_counted_free_size.fetch_sub(
            bytes, std::memory_order_relaxed);
        ASSERT(previous >= bytes);
    }
}

inline RegionPlacement GlobalHeap::resolve_allocation_placement(
    RegionPlacement requested_placement, uint32_t requested_group_id) const {
    if (!region_placement_enabled) {
        return RegionPlacement::Unclassified;
    }
    if (requested_placement != RegionPlacement::Resident) {
        return requested_placement;
    }
    if (resident_region_budget == 0) {
        return RegionPlacement::Streaming;
    }
    return resident_reserved_regions.load(std::memory_order_acquire) <
                   resident_region_budget
               ? RegionPlacement::Resident
               : RegionPlacement::Streaming;
}

inline RegionHead *GlobalHeap::allocate_region(
    size_t bin, RegionPlacement requested_placement,
    uint32_t requested_group_id) {
    if (!region_placement_enabled) {
        requested_placement = RegionPlacement::Unclassified;
        requested_group_id = 0;
    } else {
        ASSERT(requested_placement != RegionPlacement::Unclassified ||
               requested_group_id == 0);
    }
    RegionHead *region =
        allocate_region_impl(bin, requested_placement, requested_group_id);
    return region;
}

inline void GlobalHeap::add_free_size(int64_t size) {
    free_size.fetch_add(size, std::memory_order::relaxed);
}

inline void GlobalHeap::sub_free_size(int64_t size) {
    free_size.fetch_sub(size, std::memory_order::relaxed);
}

inline void GlobalHeap::check_free_size_negative_after_alloc(
    const char *where, size_t bin, void *region_ptr, int64_t sub_bytes,
    int64_t region_free_bytes, size_t unalloc_off, size_t heap_sz) {
    int64_t now = free_size.load(std::memory_order::relaxed);
    if (now >= 0) return;

    std::cerr << "[FREE_SIZE_NEG] where=" << where << "\n"
              << "  free_size_now=" << now << "\n"
              << "  bin=" << bin << "\n"
              << "  region=" << region_ptr << "\n"
              << "  sub_bytes=" << sub_bytes << "\n"
              << "  region_free_bytes=" << region_free_bytes << "\n"
              << "  unallocated_offset=" << unalloc_off
              << " heap_size=" << heap_sz << "\n";
    std::abort();
}

inline RegionHead *GlobalHeap::allocate_region_impl(
    size_t bin, RegionPlacement requested_placement,
    uint32_t requested_group_id) {
    const bool bind_groups = FarLib::get_config().region_placement_bind_groups;
    auto pop_usable = [&](RegionPlacement placement,
                          const char *debug_tag) -> RegionHead * {
        auto matches = [placement, requested_group_id,
                        bind_groups](RegionHead *candidate) {
            return bind_groups
                       ? candidate->placement_matches(placement,
                                                      requested_group_id)
                       : candidate->placement_class_matches(placement);
        };
        auto &list = usable_region_list[placement_index(placement)][bin];
        RegionHead *candidate =
            bind_groups ? list.pop_first_matching(matches) : list.pop();
        if (candidate == nullptr) {
            return nullptr;
        }
        ASSERT(candidate->placement_class_matches(placement));
        const int64_t region_free =
            static_cast<int64_t>(candidate->free_size());
        if (region_free < 0 || region_free > RegionSize) {
            ERROR("region_free is out of range");
        }
        sub_free_size(region_free);
        if (placement == RegionPlacement::Resident) {
            sub_resident_counted_free_size(region_free);
        }
        check_free_size_negative_after_alloc(
            debug_tag, bin, static_cast<void *>(candidate), region_free,
            region_free,
            unallocated_offset.load(std::memory_order_relaxed), heap_size);
        return candidate;
    };

    int active_fallback_direction = -1;
    auto start_fallback = [&](RegionPlacement from, RegionPlacement to) {
        if (from == RegionPlacement::Resident &&
            to == RegionPlacement::Streaming) {
            active_fallback_direction = 0;
        } else if (from == RegionPlacement::Streaming &&
                   to == RegionPlacement::Resident) {
            active_fallback_direction = 1;
        } else {
            ASSERT(false);
        }
        placement_fallback_attempts[active_fallback_direction][bin].fetch_add(
            1, std::memory_order_relaxed);
    };
    auto record_fallback_success = [&](size_t capacity_bytes) {
        ASSERT(active_fallback_direction >= 0);
        placement_fallback_successes[active_fallback_direction][bin].fetch_add(
            1, std::memory_order_relaxed);
        placement_fallback_capacity_bytes[active_fallback_direction][bin]
            .fetch_add(capacity_bytes, std::memory_order_relaxed);
    };

    RegionHead *region = pop_usable(requested_placement, "usable.pop");
    if (region != nullptr) {
        return region;
    }

    RegionPlacement actual_placement = requested_placement;
    bool reserved_new_resident_region = false;
    if (actual_placement == RegionPlacement::Resident) {
        size_t observed =
            resident_reserved_regions.load(std::memory_order_relaxed);
        while (observed < resident_region_budget &&
               !resident_reserved_regions.compare_exchange_weak(
                   observed, observed + 1, std::memory_order_acq_rel,
                   std::memory_order_relaxed)) {
        }
        if (observed < resident_region_budget) {
            reserved_new_resident_region = true;
        } else {
            actual_placement = RegionPlacement::Streaming;
            region = pop_usable(actual_placement,
                                "usable.resident_budget_streaming");
            if (region != nullptr) {
                return region;
            }
        }
    }

    region = static_cast<RegionHead *>(free_region_list.pop());
    if (region != nullptr) {
#ifdef FARLIB_ALLOC_DEBUG
        int64_t before = free_size.load(std::memory_order::relaxed);
        if (before < (int64_t)RegionSize) {
            alloc_debug_printf(
                "[FREE_POP_BEFORE] free_size=%lld (<RegionSize=%zu) bin=%zu region=%p unalloc=%zu heap=%zu\n",
                (long long)before, (size_t)RegionSize, (size_t)bin, (void *)region,
                (size_t)unallocated_offset.load(std::memory_order::relaxed),
                (size_t)heap_size);
        }
#endif
        sub_free_size(RegionSize);
        region->init(bin, allocate_placement_epoch());
        region->claim_placement(actual_placement,
                                bind_groups ? requested_group_id : 0,
                                allocate_placement_epoch());
        if (!reserved_new_resident_region) {
            account_region_claim(actual_placement);
        }
        if (active_fallback_direction >= 0) {
            record_fallback_success(RegionSize);
        }
        check_free_size_negative_after_alloc(
            "free.pop", bin, (void *)region, (int64_t)RegionSize,
            (int64_t)RegionSize,
            unallocated_offset.load(std::memory_order::relaxed), heap_size);
        return region;
    }

    size_t offset = unallocated_offset.load();
retry_alloc:
    if (offset + RegionSize <= heap_size) {
        if (!unallocated_offset.compare_exchange_weak(offset, offset + RegionSize))
            goto retry_alloc;
        region =
            reinterpret_cast<RegionHead *>(static_cast<char *>(heap) + offset);
        sub_free_size(RegionSize);
        region->init(bin, allocate_placement_epoch());
        region->claim_placement(actual_placement,
                                bind_groups ? requested_group_id : 0,
                                allocate_placement_epoch());
        if (!reserved_new_resident_region) {
            account_region_claim(actual_placement);
        }
        if (active_fallback_direction >= 0) {
            record_fallback_success(RegionSize);
        }
        check_free_size_negative_after_alloc(
            "bump.alloc", bin, (void *)region, (int64_t)RegionSize,
            (int64_t)RegionSize,
            unallocated_offset.load(std::memory_order::relaxed), heap_size);
        return region;
    }
    if (reserved_new_resident_region) {
        account_region_release(RegionPlacement::Resident);
    }
    if (actual_placement == requested_placement &&
        (requested_placement == RegionPlacement::Resident ||
         requested_placement == RegionPlacement::Streaming)) {
        const RegionPlacement fallback_placement =
            requested_placement == RegionPlacement::Resident
                ? RegionPlacement::Streaming
                : RegionPlacement::Resident;
        start_fallback(requested_placement, fallback_placement);
        region = pop_usable(fallback_placement,
                            requested_placement == RegionPlacement::Resident
                                ? "usable.resident_to_streaming_fallback"
                                : "usable.streaming_to_resident_fallback");
        if (region != nullptr) {
            record_fallback_success(region->free_size());
            return region;
        }
    }
    return nullptr;
}

inline void GlobalHeap::return_back_region(RegionHead *region) {
#ifdef FARLIB_ALLOC_DEBUG
    if (region->state.load(std::memory_order::relaxed) != IN_USE) {
        alloc_debug_printf(
            "[RETURN_BACK_BUG] return_back on non-IN_USE region=%p state=%u bin=%u stamp=%u enq_list=%p last_tag=%s\n",
            (void *)region,
            (unsigned)region->state.load(std::memory_order::relaxed),
            (unsigned)region->bin, (unsigned)region->sweep_time_stamp,
            region->dbg_enqueued_list.load(std::memory_order::relaxed),
            (region->dbg_last_push_tag ? region->dbg_last_push_tag : "(null)"));
        return;
    }
#endif
    if (region->is_empty()) [[unlikely]] {
        account_region_release(region->load_placement());
        region->reset_placement(allocate_placement_epoch());
        region->state.store(FREE, std::memory_order_relaxed);
        add_free_size((int64_t)RegionSize);
        free_region_list.push_dbg(region, "return_back.free");
    } else if (region->can_allocate()) [[unlikely]] {
        uint32_t bin = region->bin;
        region->state.store(USABLE, std::memory_order::relaxed);
        size_t region_free_size = region->free_size();
        add_free_size((int64_t)region_free_size);
        if (region->load_placement() == RegionPlacement::Resident) {
            add_resident_counted_free_size(
                static_cast<int64_t>(region_free_size));
        }
        usable_region_list[placement_index(region->load_placement())][bin]
            .push_dbg(region, "return_back.usable");
    } else {
        region->state.store(FULL, std::memory_order::relaxed);
        full_region_list[placement_index(region->load_placement())].push_dbg(
            region, "return_back.full");
    }
}


inline void GlobalHeap::return_back_full_regions_batch(RegionHead **regions,
                                                       size_t count) {
    if (count == 0) return;

    RegionHead *by_placement[RegionPlacementCount][RegionReturnBatchSize];
    size_t placement_counts[RegionPlacementCount]{};
    assert(count <= RegionReturnBatchSize);
    for (size_t i = 0; i < count; ++i) {
        RegionHead *region = regions[i];
        assert(region != nullptr);
        assert(!region->is_empty());
        assert(!region->can_allocate());
#ifdef FARLIB_ALLOC_DEBUG
        assert(region->state.load(std::memory_order_relaxed) == IN_USE);
#endif
        region->state.store(FULL, std::memory_order_relaxed);
        const size_t placement = placement_index(region->load_placement());
        by_placement[placement][placement_counts[placement]++] = region;
    }
    for (size_t placement = 0; placement < RegionPlacementCount;
         ++placement) {
        full_region_list[placement].push_batch_dbg(
            by_placement[placement], placement_counts[placement],
            "return_back.full_batch");
    }
}

inline void GlobalHeap::publish_owned_evict_region(RegionHead *region) {
    if (region == nullptr) return;
    if (region->is_empty()) [[unlikely]] {
        profile::count_evac_evict_region_to_free();
        account_region_release(region->load_placement());
        region->reset_placement(allocate_placement_epoch());
        region->state.store(FREE, std::memory_order::relaxed);
        add_free_size((int64_t)RegionSize);
#if defined(FARLIB_ALLOC_DEBUG) && defined(FARLIB_ALLOC_DEBUG_FENCE)
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
        free_region_list.push_dbg(region, "evict.ready_to_free");
    } else if (region->can_allocate()) {
        profile::count_evac_evict_region_to_usable();
        region->state.store(USABLE, std::memory_order::relaxed);
        int64_t region_free = (int64_t)region->free_size();
        if (region_free < 0 || region_free > (int64_t)RegionSize) {
            ERROR("region_free is out of range");
        }
        add_free_size(region_free);
        if (region->load_placement() == RegionPlacement::Resident) {
            add_resident_counted_free_size(region_free);
        }
#if defined(FARLIB_ALLOC_DEBUG) && defined(FARLIB_ALLOC_DEBUG_FENCE)
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
        usable_region_list[placement_index(region->load_placement())]
                          [region->bin]
                              .push_dbg(region, "evict.ready_to_usable");
    } else {
        profile::count_evac_evict_region_to_full();
        region->state.store(FULL, std::memory_order::relaxed);
        full_region_list[placement_index(region->load_placement())].push_dbg(
            region, "evict.ready_to_full");
    }
}

inline void GlobalHeap::requeue_evict_task(EvictTask &task) {
    RegionHead *region = task.region;
    if (region == nullptr) {
        return;
    }
    while (region->requeue_lock.test_and_set(std::memory_order_acquire)) {
        __builtin_ia32_pause();
    }
    if (region->placement_list_owned.load(std::memory_order_acquire)) {
        region->requeue_lock.clear(std::memory_order_release);
        return;
    }
    if (region->state.load(std::memory_order_acquire) == IN_USE) {
        region->requeue_lock.clear(std::memory_order_release);
        return;
    }
    const RegionPlacement placement = region->load_placement();
    const RegionState state =
        region->state.load(std::memory_order_acquire);
    const size_t current_free_size = region->free_size();
    if (region->is_empty()) {
        // A ready task may outlive its last object: shutdown can observe the
        // task after its RDMA completion or deallocator has emptied the
        // Region.  An empty Region must become a whole FREE Region, not an
        // USABLE Region whose free_size excludes sizeof(RegionHead).
        if (task.reserved_free_size) {
            // mark_list_to_sink reserved the old free bytes before emitting
            // this task; the whole Region is newly available now.
            task.reserved_free_size = false;
            add_free_size(static_cast<int64_t>(RegionSize));
        } else {
            // For a normal list task, the current usable free bytes were
            // already included in global free_size.  Publish the Region
            // header as the newly available remainder.
            if (state == USABLE) {
                add_free_size(static_cast<int64_t>(RegionSize -
                                                    current_free_size));
            } else if (current_free_size != 0) {
                // FULL should normally have no free bytes; preserve the
                // existing diagnostic rather than silently corrupting free
                // accounting if an anomalous FULL Region reaches here.
                full_region_free_size_violation_count.fetch_add(
                    1, std::memory_order_relaxed);
                add_free_size(static_cast<int64_t>(RegionSize));
            }
        }
        if (state == USABLE && placement == RegionPlacement::Resident) {
            sub_resident_counted_free_size(
                static_cast<int64_t>(current_free_size));
        }
        account_region_release(placement);
        region->reset_placement(allocate_placement_epoch());
        region->state.store(FREE, std::memory_order_release);
        free_region_list.push_dbg(region, "task.requeue.empty");
        region->requeue_lock.clear(std::memory_order_release);
        return;
    }
    if (task.reserved_free_size) {
        add_free_size(static_cast<int64_t>(region->free_size()));
        task.reserved_free_size = false;
    }
    if (task.source_list != nullptr) {
        const size_t placement = placement_index(region->load_placement());
        const RegionState state =
            region->state.load(std::memory_order_relaxed);
        if (state == USABLE) {
            usable_region_list[placement][region->bin].push_dbg(
                region, "task.requeue.source.usable");
        } else {
            ASSERT(state == FULL);
            full_region_list[placement].push_dbg(
                region, "task.requeue.source.full");
        }
        region->requeue_lock.clear(std::memory_order_release);
        return;
    }
    if (region->can_allocate()) {
        region->state.store(USABLE, std::memory_order_relaxed);
        usable_region_list[placement_index(region->load_placement())]
                          [region->bin]
                              .push_dbg(region, "task.requeue.usable");
    } else {
        region->state.store(FULL, std::memory_order_relaxed);
        full_region_list[placement_index(region->load_placement())].push_dbg(
            region, "task.requeue.full");
    }
    region->requeue_lock.clear(std::memory_order_release);
}

inline void GlobalHeap::reclaim_resident_region(
    RegionHead *region, RegionList &source_list) {
    const bool origin_free_was_counted =
        region->state.load(std::memory_order_relaxed) == USABLE;
    const size_t origin_free_size = region->free_size();
    if (!origin_free_was_counted && origin_free_size != 0) {
        full_region_free_size_violation_count.fetch_add(
            1, std::memory_order_relaxed);
    }
    if (region->reclaim_deallocated_blocks() == 0) {
        source_list.push_dbg(region, "resident.reclaim.noop");
        return;
    }
    if (origin_free_was_counted || origin_free_size != 0) {
        add_free_size(-static_cast<int64_t>(origin_free_size));
    }
    if (origin_free_was_counted) {
        sub_resident_counted_free_size(
            static_cast<int64_t>(origin_free_size));
    }
    publish_owned_evict_region(region);
}

inline void GlobalHeap::reclaim_all_deallocated_regions() {
    uint64_t empty_list_regions = 0;
    auto reclaim_list = [this, &empty_list_regions](RegionList &list,
                               bool origin_free_was_counted) {
        while (true) {
            RegionHead *region = list.pop_first_matching(
                [](RegionHead *candidate) {
                    return candidate->is_empty() ||
                           candidate->pending_reclaims.load(
                               std::memory_order_acquire) != 0 ||
                           candidate->has_deallocated_blocks();
                });
            if (region == nullptr) {
                break;
            }
            const size_t origin_free_size = region->free_size();
            if (!origin_free_was_counted && origin_free_size != 0) {
                full_region_free_size_violation_count.fetch_add(
                    1, std::memory_order_relaxed);
                shutdown_full_origin_free_bytes.fetch_add(
                    origin_free_size, std::memory_order_relaxed);
            }
            const bool was_empty = region->is_empty();
            if (region->reclaim_deallocated_blocks(true) == 0 &&
                !was_empty) {
                list.push_dbg(region, "shutdown.reclaim.noop");
                continue;
            }
            if (was_empty) {
                ++empty_list_regions;
            }
            // A FULL Region should have zero free bytes. If a prior mark made
            // deallocated blocks visible without reclassifying the list, that
            // anomalous free space was nevertheless already added globally.
            // Subtract it before publish_owned_evict_region() republishes the
            // complete post-reclaim free size.
            if (origin_free_was_counted || origin_free_size != 0) {
                add_free_size(-static_cast<int64_t>(origin_free_size));
                if (origin_free_was_counted &&
                    region->load_placement() ==
                        RegionPlacement::Resident) {
                    sub_resident_counted_free_size(
                        static_cast<int64_t>(origin_free_size));
                }
            }
            publish_owned_evict_region(region);
        }
    };
    constexpr RegionPlacement placement_order[] = {
        RegionPlacement::Streaming, RegionPlacement::Unclassified,
        RegionPlacement::Mixed, RegionPlacement::Resident};
    for (RegionPlacement placement_kind : placement_order) {
        const size_t placement = placement_index(placement_kind);
        reclaim_list(full_region_list[placement], false);
        for (size_t bin = 0; bin < RegionBinCount; ++bin) {
            reclaim_list(usable_region_list[placement][bin], true);
        }
    }
    const int64_t final_free = get_free_bytes();
    const int64_t final_resident_free =
        resident_counted_free_size.load(std::memory_order_relaxed);
    ASSERT(final_free >= 0);
    ASSERT(final_free <= static_cast<int64_t>(heap_size));
    ASSERT(final_resident_free >= 0);
    ASSERT(final_resident_free <= final_free);
    if (empty_list_regions != 0) {
        std::cerr << "shutdown.empty_list_reclaim count="
                  << empty_list_regions << std::endl;
    }
}

inline void GlobalHeap::verify_no_detached_regions_for_shutdown() const {
    uint64_t in_use_regions = 0;
    uint64_t detached_regions = 0;
    const size_t committed =
        unallocated_offset.load(std::memory_order_acquire);
    for (size_t offset = 0; offset < committed; offset += RegionSize) {
        auto *region = reinterpret_cast<RegionHead *>(
            static_cast<char *>(heap) + offset);
        if (region->state.load(std::memory_order_acquire) == IN_USE) {
            ++in_use_regions;
            continue;
        }
        if (!region->placement_list_owned.load(std::memory_order_acquire)) {
            ++detached_regions;
        }
    }
    std::cerr << "shutdown.region_ownership committed_regions="
              << committed / RegionSize
              << " in_use_regions=" << in_use_regions
              << " detached_regions=" << detached_regions << std::endl;
    ASSERT(in_use_regions == 0);
    ASSERT(detached_regions == 0);
}

inline void GlobalHeap::mark_streaming_region_shared(RegionHead *region) {
    if (region == nullptr || region->load_placement_group_id() == 0) {
        return;
    }
    region->lock_placement();
    ASSERT(!region->placement_list_owned.load(std::memory_order_acquire));
    ASSERT(region->load_placement() == RegionPlacement::Streaming);
    // Group zero is the shared transient generation. It stays uniformly
    // Streaming; only the profile-group homogeneity constraint is relaxed.
    region->placement_group_id.store(0, std::memory_order_release);
    region->placement_epoch.store(allocate_placement_epoch(),
                                  std::memory_order_release);
    region->unlock_placement();
}

inline void *GlobalHeap::get_heap() { return heap; }
inline size_t GlobalHeap::get_heap_size() const { return heap_size; }

inline int64_t GlobalHeap::get_free_bytes() const {
    return free_size.load(std::memory_order::relaxed);
}

inline ReclaimSupplySnapshot GlobalHeap::reclaim_supply_snapshot(
    size_t bin) const {
    ReclaimSupplySnapshot snapshot;
    snapshot.free_regions = free_region_list.size();
    if (bin >= RegionBinCount) return snapshot;
    for (size_t placement = 0; placement < RegionPlacementCount; ++placement) {
        snapshot.usable_regions += usable_region_list[placement][bin].size();
    }
    return snapshot;
}

inline int64_t GlobalHeap::get_used_bytes() const {
    return (int64_t)heap_size - get_free_bytes();
}

inline size_t GlobalHeap::get_committed_bytes() const {
    return unallocated_offset.load(std::memory_order::relaxed);
}

inline bool GlobalHeap::region_placement_is_enabled() const {
    return region_placement_enabled;
}

inline size_t GlobalHeap::get_resident_region_budget() const {
    return resident_region_budget;
}

inline size_t GlobalHeap::get_resident_reserved_regions() const {
    return resident_reserved_regions.load(std::memory_order_acquire);
}

inline size_t GlobalHeap::get_streaming_reserved_regions() const {
    return streaming_reserved_regions.load(std::memory_order_acquire);
}

inline void GlobalHeap::get_region_placement_group_snapshots(
    std::vector<RegionPlacementGroupSnapshot> &snapshots) const {
    if (!FarLib::get_config().region_placement_bind_groups) {
        for (auto &snapshot : snapshots) {
            snapshot = {};
        }
        return;
    }
    for (auto &snapshot : snapshots) {
        snapshot = {};
    }
    const size_t committed =
        unallocated_offset.load(std::memory_order_acquire);
    for (size_t offset = 0; offset < committed; offset += RegionSize) {
        auto *region = reinterpret_cast<RegionHead *>(
            static_cast<char *>(heap) + offset);
        const uint32_t group_id = region->load_placement_group_id();
        if (group_id == 0 || group_id >= snapshots.size()) {
            continue;
        }
        auto &snapshot = snapshots[group_id];
        ++snapshot.total_regions;
        const RegionPlacement placement = region->load_placement();
        if (placement == RegionPlacement::Resident) {
            ++snapshot.resident_regions;
        } else if (placement == RegionPlacement::Streaming) {
            ++snapshot.streaming_regions;
        }
    }
}

inline void GlobalHeap::collect_region_hotness_snapshots(
    std::vector<RegionHotnessSnapshot> &snapshots, size_t write_weight,
    size_t ema_decay_shift) {
    snapshots.clear();
    std::vector<RegionListSnapshotHandle> listed_regions;
    for (RegionPlacement placement : {RegionPlacement::Resident,
                                      RegionPlacement::Streaming}) {
        full_region_list[placement_index(placement)]
            .append_region_snapshot(listed_regions);
        for (size_t bin = 0; bin < RegionBinCount; ++bin) {
            usable_region_list[placement_index(placement)][bin]
                .append_region_snapshot(listed_regions);
        }
    }
    snapshots.reserve(listed_regions.size());
    for (const RegionListSnapshotHandle &handle : listed_regions) {
        RegionHead *region = handle.region;
        region->lock_placement();
        const RegionPlacement placement = region->load_placement();
        const uint64_t placement_epoch = region->load_placement_epoch();
        if (!region->placement_list_owned.load(std::memory_order_acquire) ||
            placement_epoch != handle.placement_epoch ||
            placement != handle.placement ||
            (placement != RegionPlacement::Resident &&
             placement != RegionPlacement::Streaming)) {
            region->unlock_placement();
            continue;
        }
        const uint64_t window_read =
            region->profile_window_read_references.exchange(
                0, std::memory_order_acq_rel);
        const uint64_t window_write =
            region->profile_window_write_references.exchange(
                0, std::memory_order_acq_rel);
        const uint64_t window_resident_fetch_intent_bytes =
            region->profile_window_resident_fetch_intent_bytes.exchange(
                0, std::memory_order_acq_rel);
        const uint64_t previous_read =
            region->profile_ema_read_references.load(
                std::memory_order_relaxed);
        const uint64_t previous_write =
            region->profile_ema_write_references.load(
                std::memory_order_relaxed);
        const uint64_t previous_resident_fetch_intent_bytes =
            region->profile_ema_resident_fetch_intent_bytes.load(
                std::memory_order_relaxed);
        const uint64_t carry_read =
            ema_decay_shift >= 64 ? 0 : previous_read >> ema_decay_shift;
        const uint64_t carry_write =
            ema_decay_shift >= 64 ? 0 : previous_write >> ema_decay_shift;
        const uint64_t carry_resident_fetch_intent_bytes =
            ema_decay_shift >= 64
                ? 0
                : previous_resident_fetch_intent_bytes >> ema_decay_shift;
        const uint64_t ema_read =
            window_read > std::numeric_limits<uint64_t>::max() - carry_read
                ? std::numeric_limits<uint64_t>::max()
                : window_read + carry_read;
        const uint64_t ema_write =
            window_write > std::numeric_limits<uint64_t>::max() - carry_write
                ? std::numeric_limits<uint64_t>::max()
                : window_write + carry_write;
        const uint64_t ema_resident_fetch_intent_bytes =
            window_resident_fetch_intent_bytes >
                    std::numeric_limits<uint64_t>::max() -
                        carry_resident_fetch_intent_bytes
                ? std::numeric_limits<uint64_t>::max()
                : window_resident_fetch_intent_bytes +
                      carry_resident_fetch_intent_bytes;
        region->profile_ema_read_references.store(
            ema_read, std::memory_order_relaxed);
        region->profile_ema_write_references.store(
            ema_write, std::memory_order_relaxed);
        region->profile_ema_resident_fetch_intent_bytes.store(
            ema_resident_fetch_intent_bytes,
            std::memory_order_relaxed);
        const __uint128_t weighted =
            static_cast<__uint128_t>(ema_write) * write_weight + ema_read;
        const uint64_t total_read =
            region->profile_total_read_references.load(
                std::memory_order_relaxed);
        const uint64_t total_write =
            region->profile_total_write_references.load(
                std::memory_order_relaxed);
        const __uint128_t total_weighted =
            static_cast<__uint128_t>(total_write) * write_weight +
            total_read;
        snapshots.push_back(
            {.region = region,
             .placement_epoch = placement_epoch,
             .placement = placement,
             .state = region->state.load(std::memory_order_acquire),
             .bin = region->bin,
             .free_bytes = region->free_size(),
             .resident_fetch_intent_bytes =
                 ema_resident_fetch_intent_bytes,
             .read_references = ema_read,
             .write_references = ema_write,
             .weighted_references =
                 weighted > std::numeric_limits<uint64_t>::max()
                     ? std::numeric_limits<uint64_t>::max()
                     : static_cast<uint64_t>(weighted),
             .total_weighted_references =
                 total_weighted >
                         std::numeric_limits<uint64_t>::max()
                     ? std::numeric_limits<uint64_t>::max()
                     : static_cast<uint64_t>(total_weighted),
             .list_owned = region->placement_list_owned.load(
                 std::memory_order_acquire)});
        region->unlock_placement();
    }
}

inline size_t GlobalHeap::prioritize_profile_pending_regions(
    uint64_t plan_id) {
    size_t prioritized = 0;
    auto prioritize_list = [&](RegionList &list) {
        std::vector<RegionHead *> selected;
        list.extract_all_matching(
            selected, [plan_id](RegionHead *region) {
                return region->profile_pending_plan_matches(plan_id);
            });
        std::stable_sort(
            selected.begin(), selected.end(),
            [](RegionHead *left, RegionHead *right) {
                return left->load_profile_pending_plan_rank() <
                       right->load_profile_pending_plan_rank();
            });
        prioritized += selected.size();
        list.push_front_batch_dbg(selected.data(), selected.size(),
                                  "profile.pending.prioritize");
    };
    for (RegionPlacement placement : {RegionPlacement::Resident,
                                      RegionPlacement::Streaming}) {
        prioritize_list(full_region_list[placement_index(placement)]);
        for (size_t bin = 0; bin < RegionBinCount; ++bin) {
            prioritize_list(
                usable_region_list[placement_index(placement)][bin]);
        }
    }
    return prioritized;
}

inline RegionReclassifyResult GlobalHeap::try_reclassify_hotness_region(
    RegionHead *region, uint64_t expected_placement_epoch,
    RegionPlacement from, RegionPlacement to,
    const RegionPlacementTransitionFn &on_entry_transition) {
    RegionReclassifyResult result;
    if (!region_placement_enabled ||
        FarLib::get_config().region_placement_bind_groups ||
        region == nullptr || from == to ||
        (from != RegionPlacement::Resident &&
         from != RegionPlacement::Streaming) ||
        (to != RegionPlacement::Resident &&
         to != RegionPlacement::Streaming)) {
        return result;
    }

    const RegionState expected_state =
        region->state.load(std::memory_order_acquire);
    RegionList *source_list = nullptr;
    if (expected_state == USABLE) {
        source_list =
            &usable_region_list[placement_index(from)][region->bin];
    } else if (expected_state == FULL) {
        source_list = &full_region_list[placement_index(from)];
    } else {
        return result;
    }

    RegionHead *owned = source_list->pop_first_matching(
        [region, expected_placement_epoch, expected_state,
         from](RegionHead *candidate) {
            return candidate == region &&
                   candidate->state.load(std::memory_order_relaxed) ==
                       expected_state &&
                   candidate->load_placement_epoch() ==
                       expected_placement_epoch &&
                   candidate->placement_class_matches(from);
        });
    if (owned == nullptr) {
        return result;
    }
    ASSERT(owned == region);

    auto restore_source = [&] {
        source_list->push_dbg(region, "hotness.reclass.restore");
    };

    region->lock_placement();
    bool transitionable =
        !region->placement_list_owned.load(std::memory_order_relaxed) &&
        region->state.load(std::memory_order_relaxed) == expected_state &&
        region->load_placement_epoch() == expected_placement_epoch &&
        region->placement_class_matches(from) &&
        region->marked_list == nullptr &&
        region->last_mark_epoch.load(std::memory_order_acquire) <=
            region->evict_drained_epoch.load(std::memory_order_acquire);
    struct LockedEntry {
        cache::FarObjectEntry *entry;
        cache::EntryStateBits original_state;
        size_t size;
    };
    std::vector<LockedEntry> locked_entries;
    auto unlock_entries = [&] {
        for (auto it = locked_entries.rbegin();
             it != locked_entries.rend(); ++it) {
            auto expected = it->original_state;
            expected.invalid = 1;
            auto restored = it->original_state;
            restored.invalid = 0;
            ASSERT(it->entry->cas_state_strong(expected, restored));
        }
        locked_entries.clear();
    };
    if (transitionable) {
        for (BlockHead *block = region->active_list; block != nullptr;
             block = block->next) {
            const cache::far_obj_t obj =
                block->obj_meta_data.load(std::memory_order_acquire);
            if (obj.is_null()) {
                continue;
            }
            auto *entry = obj.get_entry_ptr();
            auto state = entry->load_state(std::memory_order_acquire);
            if (state.invalid || state.state == cache::FETCHING ||
                state.state == cache::EVICTING ||
                state.state == cache::BUSY ||
                state.state == cache::REMOTE ||
                state.state == cache::FREE || state.ref_cnt != 0) {
                transitionable = false;
                break;
            }
            auto locked_state = state;
            locked_state.invalid = 1;
            if (!entry->cas_state_strong(state, locked_state)) {
                transitionable = false;
                break;
            }
            locked_entries.push_back(
                {.entry = entry, .original_state = state, .size = obj.size});
        }
    }
    result.transitionable = transitionable;
    if (!transitionable) {
        unlock_entries();
        region->unlock_placement();
        restore_source();
        return result;
    }

    bool budget_acquired = true;
    if (to == RegionPlacement::Resident) {
        size_t observed =
            resident_reserved_regions.load(std::memory_order_relaxed);
        while (observed < resident_region_budget &&
               !resident_reserved_regions.compare_exchange_weak(
                   observed, observed + 1, std::memory_order_acq_rel,
                   std::memory_order_relaxed)) {
        }
        budget_acquired = observed < resident_region_budget;
    }
    if (!budget_acquired) {
        unlock_entries();
        region->unlock_placement();
        restore_source();
        return result;
    }

    if (on_entry_transition) {
        for (const auto &locked : locked_entries) {
            on_entry_transition(*locked.entry, locked.size, from, to);
        }
    }

    const bool resident = to == RegionPlacement::Resident;
    for (const auto &locked : locked_entries) {
        locked.entry->set_resident_local(resident);
    }
    const int64_t counted_region_free =
        expected_state == USABLE
            ? static_cast<int64_t>(region->free_size())
            : 0;
    if (from == RegionPlacement::Resident) {
        sub_resident_counted_free_size(counted_region_free);
        account_region_release(RegionPlacement::Resident);
    } else {
        account_region_release(RegionPlacement::Streaming);
    }
    if (to == RegionPlacement::Streaming) {
        account_region_claim(RegionPlacement::Streaming);
    } else {
        add_resident_counted_free_size(counted_region_free);
    }
    region->placement_group_id.store(0, std::memory_order_relaxed);
    region->profile_pending_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Unclassified),
        std::memory_order_relaxed);
    region->profile_pending_plan_id.store(0, std::memory_order_relaxed);
    region->profile_pending_plan_rank.store(0,
                                            std::memory_order_relaxed);
    region->profile_window_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    region->profile_ema_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    region->placement.store(to, std::memory_order_release);
    region->placement_epoch.store(allocate_placement_epoch(),
                                  std::memory_order_release);
    region->profile_last_target_placement.store(
        static_cast<uint8_t>(to), std::memory_order_relaxed);
    region->profile_target_streak.store(0, std::memory_order_relaxed);
    unlock_entries();
    region->unlock_placement();

    RegionList *destination_list =
        expected_state == USABLE
            ? &usable_region_list[placement_index(to)][region->bin]
            : &full_region_list[placement_index(to)];
    destination_list->push_dbg(region, "hotness.reclass.commit");
    result.changed = true;
    result.changed_bytes = RegionSize;
    return result;
}

inline RegionExchangeResult GlobalHeap::try_exchange_hotness_regions(
    RegionHead *hot_region, uint64_t expected_hot_epoch,
    RegionHead *cold_region, uint64_t expected_cold_epoch,
    const RegionPlacementTransitionFn &on_entry_transition) {
    RegionExchangeResult result;
    if (!region_placement_enabled ||
        FarLib::get_config().region_placement_bind_groups ||
        hot_region == nullptr || cold_region == nullptr ||
        hot_region == cold_region) {
        result.failure = RegionExchangeFailure::InvalidInput;
        return result;
    }
    const RegionState hot_state =
        hot_region->state.load(std::memory_order_acquire);
    const RegionState cold_state =
        cold_region->state.load(std::memory_order_acquire);
    auto list_for = [this](RegionHead *region, RegionState state,
                           RegionPlacement placement) -> RegionList * {
        if (state == USABLE) {
            return &usable_region_list[placement_index(placement)]
                                      [region->bin];
        }
        if (state == FULL) {
            return &full_region_list[placement_index(placement)];
        }
        return nullptr;
    };
    RegionList *hot_source = list_for(
        hot_region, hot_state, RegionPlacement::Streaming);
    RegionList *cold_source = list_for(
        cold_region, cold_state, RegionPlacement::Resident);
    if (hot_source == nullptr) {
        result.failure = RegionExchangeFailure::HotInvalidRegionState;
        return result;
    }
    if (cold_source == nullptr) {
        result.failure = RegionExchangeFailure::ColdInvalidRegionState;
        return result;
    }

    RegionHead *owned_hot = hot_source->pop_first_matching(
        [hot_region, hot_state,
         expected_hot_epoch](RegionHead *candidate) {
            return candidate == hot_region &&
                   candidate->state.load(std::memory_order_relaxed) ==
                       hot_state &&
                   candidate->load_placement_epoch() ==
                       expected_hot_epoch &&
                   candidate->placement_class_matches(
                       RegionPlacement::Streaming);
        });
    if (owned_hot == nullptr) {
        result.failure = RegionExchangeFailure::HotRegionUnavailable;
        return result;
    }
    RegionHead *owned_cold = cold_source->pop_first_matching(
        [cold_region, cold_state,
         expected_cold_epoch](RegionHead *candidate) {
            return candidate == cold_region &&
                   candidate->state.load(std::memory_order_relaxed) ==
                       cold_state &&
                   candidate->load_placement_epoch() ==
                       expected_cold_epoch &&
                   candidate->placement_class_matches(
                       RegionPlacement::Resident);
        });
    if (owned_cold == nullptr) {
        hot_source->push_dbg(hot_region,
                             "hotness.exchange.restore.hot");
        result.failure = RegionExchangeFailure::ColdRegionUnavailable;
        return result;
    }

    RegionHead *first = hot_region;
    RegionHead *second = cold_region;
    if (std::less<RegionHead *>{}(second, first)) {
        std::swap(first, second);
    }
    first->lock_placement();
    second->lock_placement();

    struct LockedEntry {
        cache::FarObjectEntry *entry;
        cache::EntryStateBits original_state;
        size_t size;
        RegionPlacement from;
        RegionPlacement to;
    };
    std::vector<LockedEntry> locked_entries;
    auto unlock_entries = [&] {
        for (auto it = locked_entries.rbegin();
             it != locked_entries.rend(); ++it) {
            auto expected = it->original_state;
            expected.invalid = 1;
            auto restored = it->original_state;
            restored.invalid = 0;
            ASSERT(it->entry->cas_state_strong(expected, restored));
        }
        locked_entries.clear();
    };
    auto lock_region_entries = [&](RegionHead *region, RegionPlacement from,
                                   RegionPlacement to) {
        for (BlockHead *block = region->active_list; block != nullptr;
             block = block->next) {
            const cache::far_obj_t obj =
                block->obj_meta_data.load(std::memory_order_acquire);
            if (obj.is_null()) {
                continue;
            }
            auto *entry = obj.get_entry_ptr();
            auto state = entry->load_state(std::memory_order_acquire);
            if (state.invalid || state.state == cache::FETCHING ||
                state.state == cache::EVICTING ||
                state.state == cache::BUSY ||
                state.state == cache::REMOTE ||
                state.state == cache::FREE || state.ref_cnt != 0) {
                return false;
            }
            auto locked_state = state;
            locked_state.invalid = 1;
            if (!entry->cas_state_strong(state, locked_state)) {
                return false;
            }
            locked_entries.push_back(
                {.entry = entry,
                 .original_state = state,
                 .size = obj.size,
                 .from = from,
                 .to = to});
        }
        return true;
    };
    const bool regions_still_match =
        !hot_region->placement_list_owned.load(std::memory_order_relaxed) &&
        !cold_region->placement_list_owned.load(std::memory_order_relaxed) &&
        hot_region->state.load(std::memory_order_relaxed) == hot_state &&
        cold_region->state.load(std::memory_order_relaxed) == cold_state &&
        hot_region->load_placement_epoch() == expected_hot_epoch &&
        cold_region->load_placement_epoch() == expected_cold_epoch &&
        hot_region->placement_class_matches(RegionPlacement::Streaming) &&
        cold_region->placement_class_matches(RegionPlacement::Resident) &&
        hot_region->marked_list == nullptr &&
        cold_region->marked_list == nullptr &&
        hot_region->last_mark_epoch.load(std::memory_order_acquire) <=
            hot_region->evict_drained_epoch.load(std::memory_order_acquire) &&
        cold_region->last_mark_epoch.load(std::memory_order_acquire) <=
            cold_region->evict_drained_epoch.load(std::memory_order_acquire);
    bool cold_entries_locked = false;
    bool hot_entries_locked = false;
    if (regions_still_match) {
        cold_entries_locked = lock_region_entries(
            cold_region, RegionPlacement::Resident,
            RegionPlacement::Streaming);
        if (cold_entries_locked) {
            hot_entries_locked = lock_region_entries(
                hot_region, RegionPlacement::Streaming,
                RegionPlacement::Resident);
        }
    }
    const bool entries_locked =
        regions_still_match && cold_entries_locked && hot_entries_locked;
    if (!entries_locked) {
        result.failure =
            !regions_still_match
                ? RegionExchangeFailure::RegionChanged
                : (!cold_entries_locked
                       ? RegionExchangeFailure::ColdEntryBusy
                       : RegionExchangeFailure::HotEntryBusy);
        unlock_entries();
        second->unlock_placement();
        first->unlock_placement();
        cold_source->push_dbg(cold_region,
                              "hotness.exchange.restore.cold");
        hot_source->push_dbg(hot_region,
                             "hotness.exchange.restore.hot");
        return result;
    }


    if (on_entry_transition) {
        for (const auto &locked : locked_entries) {
            on_entry_transition(*locked.entry, locked.size, locked.from,
                                locked.to);
        }
    }

    for (BlockHead *block = cold_region->active_list; block != nullptr;
         block = block->next) {
        const cache::far_obj_t obj =
            block->obj_meta_data.load(std::memory_order_acquire);
        if (!obj.is_null()) {
            obj.get_entry_ptr()->set_resident_local(false);
        }
    }
    for (BlockHead *block = hot_region->active_list; block != nullptr;
         block = block->next) {
        const cache::far_obj_t obj =
            block->obj_meta_data.load(std::memory_order_acquire);
        if (!obj.is_null()) {
            obj.get_entry_ptr()->set_resident_local(true);
        }
    }
    const int64_t cold_counted_free =
        cold_state == USABLE
            ? static_cast<int64_t>(cold_region->free_size())
            : 0;
    const int64_t hot_counted_free =
        hot_state == USABLE
            ? static_cast<int64_t>(hot_region->free_size())
            : 0;
    sub_resident_counted_free_size(cold_counted_free);
    add_resident_counted_free_size(hot_counted_free);

    cold_region->placement_group_id.store(0, std::memory_order_relaxed);
    cold_region->profile_pending_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Unclassified),
        std::memory_order_relaxed);
    cold_region->profile_pending_plan_id.store(0,
                                               std::memory_order_relaxed);
    cold_region->profile_pending_plan_rank.store(0,
                                                 std::memory_order_relaxed);
    cold_region->profile_window_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    cold_region->profile_ema_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    cold_region->placement.store(RegionPlacement::Streaming,
                                 std::memory_order_release);
    cold_region->placement_epoch.store(allocate_placement_epoch(),
                                       std::memory_order_release);
    cold_region->profile_last_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Streaming),
        std::memory_order_relaxed);
    cold_region->profile_target_streak.store(0,
                                             std::memory_order_relaxed);
    hot_region->placement_group_id.store(0, std::memory_order_relaxed);
    hot_region->profile_pending_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Unclassified),
        std::memory_order_relaxed);
    hot_region->profile_pending_plan_id.store(0,
                                              std::memory_order_relaxed);
    hot_region->profile_pending_plan_rank.store(0,
                                                std::memory_order_relaxed);
    hot_region->profile_window_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    hot_region->profile_ema_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    hot_region->placement.store(RegionPlacement::Resident,
                                std::memory_order_release);
    hot_region->placement_epoch.store(allocate_placement_epoch(),
                                      std::memory_order_release);
    hot_region->profile_last_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Resident),
        std::memory_order_relaxed);
    hot_region->profile_target_streak.store(0,
                                            std::memory_order_relaxed);
    unlock_entries();
    second->unlock_placement();
    first->unlock_placement();

    RegionList *cold_destination = list_for(
        cold_region, cold_state, RegionPlacement::Streaming);
    RegionList *hot_destination = list_for(
        hot_region, hot_state, RegionPlacement::Resident);
    ASSERT(cold_destination != nullptr && hot_destination != nullptr);
    cold_destination->push_dbg(cold_region,
                               "hotness.exchange.commit.cold");
    hot_destination->push_dbg(hot_region,
                              "hotness.exchange.commit.hot");
    result.changed = true;
    result.promoted_bytes = RegionSize;
    result.demoted_bytes = RegionSize;
    result.failure = RegionExchangeFailure::None;
    return result;
}

inline size_t GlobalHeap::reclassify_region_placement_group(
    uint32_t group_id, RegionPlacement from, RegionPlacement to,
    size_t max_regions) {
    if (!FarLib::get_config().region_placement_bind_groups) {
        return 0;
    }
    ASSERT(group_id != 0);
    ASSERT(from != to);
    ASSERT(from != RegionPlacement::Unclassified);
    ASSERT(to != RegionPlacement::Unclassified);
    size_t changed = 0;
    const size_t committed =
        unallocated_offset.load(std::memory_order_acquire);
    for (size_t offset = 0;
         offset < committed && changed < max_regions;
         offset += RegionSize) {
        auto *region = reinterpret_cast<RegionHead *>(
            static_cast<char *>(heap) + offset);
        if (!region->placement_matches(from, group_id)) {
            continue;
        }
        region->lock_placement();
        bool transitionable =
            region->placement_list_owned.load(std::memory_order_acquire) &&
            region->state.load(std::memory_order_acquire) != IN_USE &&
            region->marked_list == nullptr &&
            region->last_mark_epoch.load(std::memory_order_acquire) <=
                region->evict_drained_epoch.load(std::memory_order_acquire) &&
            region->placement_matches(from, group_id);
        if (transitionable) {
            for (BlockHead *block = region->active_list; block != nullptr;
                 block = block->next) {
                const cache::far_obj_t obj =
                    block->obj_meta_data.load(std::memory_order_acquire);
                if (obj.is_null()) {
                    continue;
                }
                const auto state = obj.get_entry_ptr()->load_state(
                    std::memory_order_acquire);
                if (state.invalid || state.state == cache::FETCHING ||
                    state.state == cache::EVICTING ||
                    state.state == cache::BUSY) {
                    transitionable = false;
                    break;
                }
            }
        }
        if (!transitionable) {
            region->unlock_placement();
            continue;
        }

        bool budget_acquired = true;
        if (to == RegionPlacement::Resident) {
            size_t observed =
                resident_reserved_regions.load(std::memory_order_relaxed);
            while (observed < resident_region_budget &&
                   !resident_reserved_regions.compare_exchange_weak(
                       observed, observed + 1, std::memory_order_acq_rel,
                       std::memory_order_relaxed)) {
            }
            budget_acquired = observed < resident_region_budget;
        }
        if (!budget_acquired) {
            region->unlock_placement();
            break;
        }

        const bool resident = to == RegionPlacement::Resident;
        for (BlockHead *block = region->active_list; block != nullptr;
             block = block->next) {
            const cache::far_obj_t obj =
                block->obj_meta_data.load(std::memory_order_acquire);
            if (obj.is_null()) {
                continue;
            }
            auto *entry = obj.get_entry_ptr();
            if (entry->resident_group_id() == group_id) {
                entry->set_resident_local(resident);
            }
        }
        const int64_t counted_region_free =
            region->state.load(std::memory_order_relaxed) == USABLE
                ? static_cast<int64_t>(region->free_size())
                : 0;
        if (from == RegionPlacement::Resident) {
            sub_resident_counted_free_size(counted_region_free);
            account_region_release(RegionPlacement::Resident);
        } else {
            account_region_release(RegionPlacement::Streaming);
        }
        if (to == RegionPlacement::Streaming) {
            account_region_claim(RegionPlacement::Streaming);
        } else {
            add_resident_counted_free_size(counted_region_free);
        }
        region->placement.store(to, std::memory_order_release);
        region->placement_epoch.store(allocate_placement_epoch(),
                                      std::memory_order_release);
        region->unlock_placement();
        ++changed;
    }
    return changed;
}

inline void GlobalHeap::print_used_memory() {
    std::array<uint64_t, 8> remaining_entry_states{};
    uint64_t remaining_local_objects = 0;
    uint64_t remaining_local_object_bytes = 0;
    const size_t committed =
        unallocated_offset.load(std::memory_order_acquire);
    for (size_t offset = 0; offset < committed; offset += RegionSize) {
        auto *region = reinterpret_cast<RegionHead *>(
            static_cast<char *>(heap) + offset);
        auto count_list = [&](BlockHead *block) {
            for (; block != nullptr; block = block->next) {
                const cache::far_obj_t obj =
                    block->obj_meta_data.load(std::memory_order_acquire);
                if (obj.is_null()) {
                    continue;
                }
                ++remaining_local_objects;
                remaining_local_object_bytes += obj.size;
                const auto state = obj.get_entry_ptr()->load_state(
                    std::memory_order_acquire);
                ++remaining_entry_states[
                    static_cast<size_t>(state.state)];
            }
        };
        count_list(region->active_list);
        count_list(region->marked_list);
    }
    std::cout << "local.capacity_bytes: " << get_heap_size() << std::endl;
    std::cout << "local.committed_bytes: " << get_committed_bytes() << std::endl;
    std::cout << "local.free_bytes: " << get_free_bytes() << std::endl;
    std::cout << "local.streaming_free_bytes: "
              << get_streaming_free_size() << std::endl;
    std::cout << "local.resident_counted_free_bytes: "
              << resident_counted_free_size.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "local.memory_low_watermark_bytes: "
              << memory_low_water_mark << std::endl;
    std::cout << "local.memory_high_watermark_bytes: "
              << memory_high_water_mark << std::endl;
    std::cout << "local.memory_release_watermark_bytes: "
              << memory_release_water_mark << std::endl;
    std::cout << "local.used_bytes: " << get_used_bytes() << std::endl;
    std::cout << "local.remaining_object_count: "
              << remaining_local_objects << std::endl;
    std::cout << "local.remaining_object_bytes: "
              << remaining_local_object_bytes << std::endl;
    for (size_t state = 0; state < remaining_entry_states.size(); ++state) {
        if (remaining_entry_states[state] != 0) {
            std::cout << "local.remaining_entry_state_" << state << ": "
                      << remaining_entry_states[state] << std::endl;
        }
    }
    std::cout << "region_placement.enabled: " << region_placement_enabled
              << std::endl;
    std::cout << "region_placement.resident_budget_regions: "
              << resident_region_budget << std::endl;
    std::cout << "region_placement.resident_reserved_regions: "
              << get_resident_reserved_regions() << std::endl;
    std::cout << "region_placement.streaming_reserved_regions: "
              << get_streaming_reserved_regions() << std::endl;
    std::cout << "region_placement.full_to_usable_after_mark_count: "
              << full_to_usable_after_mark_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region_placement.full_region_free_size_violation_count: "
              << full_region_free_size_violation_count.load(
                     std::memory_order_relaxed)
              << std::endl;
    std::cout << "region_placement.shutdown_full_origin_free_bytes: "
              << shutdown_full_origin_free_bytes.load(
                     std::memory_order_relaxed)
              << std::endl;
    constexpr const char *placement_names[] = {
        "unclassified", "resident", "streaming", "mixed"};
    for (size_t placement = 0; placement < RegionPlacementCount;
         ++placement) {
        size_t usable_regions = 0;
        for (size_t bin = 0; bin < RegionBinCount; ++bin) {
            usable_regions += usable_region_list[placement][bin].size();
        }
        std::cout << "region_placement." << placement_names[placement]
                  << "_usable_regions: " << usable_regions << std::endl;
        std::cout << "region_placement." << placement_names[placement]
                  << "_full_regions: "
                  << full_region_list[placement].size() << std::endl;
    }
    constexpr const char *fallback_names[] = {"resident_to_streaming",
                                               "streaming_to_resident"};
    for (size_t direction = 0;
         direction < RegionFallbackDirectionCount; ++direction) {
        uint64_t total_attempts = 0;
        uint64_t total_successes = 0;
        uint64_t total_capacity_bytes = 0;
        for (size_t bin = 0; bin < RegionBinCount; ++bin) {
            const uint64_t attempts =
                placement_fallback_attempts[direction][bin].load(
                    std::memory_order_relaxed);
            const uint64_t successes =
                placement_fallback_successes[direction][bin].load(
                    std::memory_order_relaxed);
            const uint64_t capacity_bytes =
                placement_fallback_capacity_bytes[direction][bin].load(
                    std::memory_order_relaxed);
            total_attempts += attempts;
            total_successes += successes;
            total_capacity_bytes += capacity_bytes;
            if (attempts != 0 || successes != 0) {
                std::cout << "region_placement.fallback_"
                          << fallback_names[direction] << "_bin_" << bin
                          << "_attempts: " << attempts << std::endl;
                std::cout << "region_placement.fallback_"
                          << fallback_names[direction] << "_bin_" << bin
                          << "_successes: " << successes << std::endl;
                std::cout << "region_placement.fallback_"
                          << fallback_names[direction] << "_bin_" << bin
                          << "_capacity_bytes: " << capacity_bytes
                          << std::endl;
            }
        }
        std::cout << "region_placement.fallback_" << fallback_names[direction]
                  << "_attempts: " << total_attempts << std::endl;
        std::cout << "region_placement.fallback_" << fallback_names[direction]
                  << "_successes: " << total_successes << std::endl;
        std::cout << "region_placement.fallback_" << fallback_names[direction]
                  << "_capacity_bytes: " << total_capacity_bytes
                  << std::endl;
    }
}

inline size_t reserve_legacy_mark_regions(std::atomic_size_t *budget,
                                          size_t max_regions) {
    if (budget == nullptr) return max_regions;
    size_t observed = budget->load(std::memory_order_relaxed);
    while (observed != 0) {
        const size_t wanted = std::min(max_regions, observed);
        if (budget->compare_exchange_weak(observed, observed - wanted,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
            return wanted;
        }
    }
    return 0;
}

inline void refund_legacy_mark_regions(std::atomic_size_t *budget,
                                       size_t regions) {
    if (budget != nullptr && regions != 0) {
        budget->fetch_add(regions, std::memory_order_relaxed);
    }
}

inline size_t reserve_scan_visits(std::atomic_size_t *budget,
                                  size_t max_visits) {
    if (budget == nullptr) return max_visits;
    size_t observed = budget->load(std::memory_order_relaxed);
    while (observed != 0) {
        const size_t wanted = std::min(max_visits, observed);
        if (budget->compare_exchange_weak(observed, observed - wanted,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
            return wanted;
        }
    }
    return 0;
}

inline void refund_scan_visits(std::atomic_size_t *budget, size_t visits) {
    if (budget != nullptr && visits != 0) {
        budget->fetch_add(visits, std::memory_order_relaxed);
    }
}

template <BlockInvoker Fn>
inline size_t GlobalHeap::mark_list(Fn &&fn, RegionList &list,
                                    uint32_t timestamp,
                                    std::vector<EvictTask> *ready_tasks,
                                    std::atomic_size_t *ready_task_budget,
                                    bool legacy_cursors,
                                    EvacuationEligibility eligibility,
                                    MarkBreakdownStats *mark_diag,
                                    std::atomic_size_t *legacy_region_budget,
                                    MarkStopContext *stop_context) {
    size_t freed_size = 0;
    const size_t batch_limit = 64;
    RegionHead *batch[batch_limit];
    const bool scan_past_blocked =
        legacy_cursors ||
        eligibility == EvacuationEligibility::LegacyConcurrent ||
        region_placement_enabled ||
        (FarLib::get_config().optimized_evacuator &&
         optimized_scan_past_blocked_region_enabled());
    const bool segment_diag =
        mark_diag != nullptr && mark_diag->segment_enabled;
    while (true) {
        drain_async_full_returns(64);
        if (stop_context != nullptr && stop_context->should_stop()) {
            stop_context->stopped = true;
            break;
        }
        if (ready_task_budget != nullptr &&
            ready_task_budget->load(std::memory_order_relaxed) == 0) {
            break;
        }
        const size_t reserved_regions =
            reserve_legacy_mark_regions(legacy_region_budget, batch_limit);
        if (reserved_regions == 0) break;
        const size_t pop_limit = reserved_regions;
        int pop_reason = RegionList::PopUnmatchedEmpty;
        size_t popped = 0;
        const uint64_t pop_start = segment_diag ? get_cycles() : 0;
        if (scan_past_blocked) {
            auto eligible = [&](RegionHead *region) {
                if (mark_diag != nullptr) ++mark_diag->eligible_calls;
                if (region->profile_pending_resident()) {
                    if (mark_diag != nullptr) {
                        ++mark_diag->eligible_false;
                        ++mark_diag->eligible_pending_resident;
                    }
                    return false;
                }
                if (region->state.load(std::memory_order::relaxed) ==
                    IN_USE) {
                    if (mark_diag != nullptr) {
                        ++mark_diag->eligible_false;
                        ++mark_diag->eligible_in_use;
                    }
                    profile::count_evac_mark_region_skipped_in_use();
                    return false;
                }
                if (region->load_placement() ==
                    RegionPlacement::Resident) {
                    const bool has_reclaims = region->pending_reclaims.load(
                               std::memory_order_acquire) != 0;
                    if (!has_reclaims && mark_diag != nullptr) {
                        ++mark_diag->eligible_false;
                        ++mark_diag->eligible_resident_no_reclaim;
                    }
                    return has_reclaims;
                }
                if (FarLib::get_config().optimized_evacuator &&
                    optimized_skip_mark_pending_enabled() &&
                    region->marked_list != nullptr) {
                    if (mark_diag != nullptr) {
                        ++mark_diag->eligible_false;
                        ++mark_diag->eligible_stable;
                    }
                    profile::count_evac_mark_region_skipped_stable();
                    return false;
                }
                if (FarLib::get_config().optimized_evacuator &&
                    optimized_mark_after_evict_enabled() &&
                    region->last_mark_epoch.load(std::memory_order_relaxed) >
                        region->evict_drained_epoch.load(
                            std::memory_order_relaxed)) {
                    if (mark_diag != nullptr) {
                        ++mark_diag->eligible_false;
                        ++mark_diag->eligible_stable;
                    }
                    profile::count_evac_mark_region_skipped_stable();
                    return false;
                }
                const bool admissible =
                    mark_region_admissible(region, eligibility) &&
                    (!legacy_cursors || region->marked_list == nullptr);
                if (!admissible && mark_diag != nullptr) {
                    ++mark_diag->eligible_false;
                    ++mark_diag->eligible_inadmissible;
                }
                return admissible;
            };
            if (legacy_cursors) {
                popped = list.pop_matching_batch_cursor(
                    timestamp, batch, pop_limit, &pop_reason, eligible,
                    mark_diag);
            } else {
                popped = list.pop_matching_batch(timestamp, batch, pop_limit,
                                                 &pop_reason, eligible,
                                                 mark_diag);
            }
        } else {
            popped = list.pop_unmatched_batch(timestamp, batch, pop_limit,
                                              &pop_reason, 0, mark_diag,
                                              nullptr);
        }
        if (reserved_regions > popped) {
            refund_legacy_mark_regions(legacy_region_budget,
                                       reserved_regions - popped);
        }
        if (segment_diag) {
            mark_diag->pop_whole_cycles += get_cycles() - pop_start;
            ++mark_diag->pop_whole_calls;
        }
        if (popped == 0) break;
        size_t batch_selected_free[batch_limit];
        uint64_t batch_selected_free_total = 0;
        if (mark_diag != nullptr) {
            for (size_t i = 0; i < popped; ++i) {
                batch_selected_free[i] = batch[i]->free_size();
                batch_selected_free_total += batch_selected_free[i];
            }
            inclusive_reclaim_diag::begin_hold(
                mark_diag->held_slot, inclusive_reclaim_diag::HeldMark,
                popped, batch_selected_free_total);
        }
        for (size_t i = 0; i < popped; i++) {
            RegionHead *region = batch[i];
            if (ready_task_budget != nullptr &&
                ready_task_budget->load(std::memory_order_relaxed) == 0) {
                for (size_t j = i; j < popped; j++) {
                    list.push_dbg(batch[j], "mark.ready_budget_return");
                    if (mark_diag != nullptr) {
                        inclusive_reclaim_diag::release_one(
                            mark_diag->held_slot, batch_selected_free[j]);
                    }
                }
                return freed_size;
            }
            inclusive_reclaim_diag::HeldRegionGuard held_guard(
                mark_diag != nullptr,
                mark_diag == nullptr ? inclusive_reclaim_diag::kNoWorker
                                     : mark_diag->held_slot,
                batch_selected_free[i]);
            if (region->profile_pending_resident()) {
                list.push_dbg(region, "skip.pending_resident");
                continue;
            }
            if (region->load_placement() == RegionPlacement::Resident) {
                reclaim_resident_region(region, list);
                continue;
            }
            if (region->state.load(std::memory_order::relaxed) == IN_USE) [[unlikely]] {
                profile::count_evac_mark_region_skipped_in_use();
                list.push_dbg(region, "skip.in_use");
                continue;
            }
            if (FarLib::get_config().optimized_evacuator &&
                optimized_skip_mark_pending_enabled() &&
                region->marked_list != nullptr) {
                profile::count_evac_mark_region_skipped_stable();
                list.push_dbg(region, "skip.pending_marked");
                continue;
            }
            if (FarLib::get_config().optimized_evacuator &&
                optimized_mark_after_evict_enabled() &&
                region->last_mark_epoch.load(std::memory_order_relaxed) >
                    region->evict_drained_epoch.load(
                        std::memory_order_relaxed)) {
                profile::count_evac_mark_region_skipped_stable();
                list.push_dbg(region, "skip.pending_evict_epoch");
                continue;
            }
            if (!mark_region_admissible(region, eligibility) ||
                (legacy_cursors && region->marked_list != nullptr)) {
                list.push_dbg(region, "mark.skip.legacy_pending");
                continue;
            }
            profile::count_evac_mark_region_scanned();
            if (mark_diag != nullptr) ++mark_diag->regions_marked;
            int64_t mark_before = profile::get_tlpd().mark_count;
            const uint32_t used_before = region->used_count;
            const size_t slot_size = get_bin_size(region->bin);
            const bool was_full =
                region->state.load(std::memory_order_relaxed) == FULL;
            size_t origin_free_size = region->free_size();
            if (inclusive_reclaim_diag::enabled() &&
                inclusive_reclaim_diag::sample_region(region)) {
                wait_trace::emit("reclaim_mark_selected", region->bin,
                                 timestamp, region->placement_epoch.load(
                                                std::memory_order_relaxed),
                                 reinterpret_cast<uint64_t>(region));
            }
            if (segment_diag) {
                const uint64_t region_start = get_cycles();
                region->mark(fn);
                mark_diag->region_mark_whole_cycles +=
                    get_cycles() - region_start;
                ++mark_diag->region_mark_whole_calls;
            } else {
                region->mark(fn);
            }
            const uint32_t used_after = region->used_count;
            ASSERT(used_after <= used_before);
            const size_t reclaimed_slots = used_before - used_after;
            const bool reclaimed_slots_claimable =
                region->marked_list == nullptr && region->can_allocate();
            if (profile::get_tlpd().mark_count > mark_before) {
                profile::count_evac_mark_region_with_output();
            } else {
                profile::count_evac_mark_region_without_output();
            }
            region->note_mark_epoch(timestamp);
            int64_t delta = (int64_t)region->free_size() - (int64_t)origin_free_size;
            if (delta < 0) {
                ERROR("delta should >= 0");
            }
            const bool ready_for_evict = region->marked_list != nullptr;
            bool new_slots_published = false;
            if (ready_for_evict && ready_tasks != nullptr) {
                if (ready_task_budget != nullptr) {
                    size_t observed =
                        ready_task_budget->load(std::memory_order_relaxed);
                    while (observed != 0 &&
                           !ready_task_budget->compare_exchange_weak(
                               observed, observed - 1,
                               std::memory_order_relaxed,
                               std::memory_order_relaxed)) {
                    }
                }
                add_free_size(-(int64_t)origin_free_size);
                ready_tasks->emplace_back(region, nullptr, timestamp, true);
            } else {
                if (region->marked_list == nullptr) {
                    region->note_evict_drained_epoch(timestamp);
                }
                add_free_size(delta);
                if (was_full && region->marked_list == nullptr &&
                    region->can_allocate()) {
                    region->state.store(USABLE,
                                        std::memory_order_relaxed);
                    full_to_usable_after_mark_count.fetch_add(
                        1, std::memory_order_relaxed);
                    usable_region_list[placement_index(
                                           region->load_placement())]
                                      [region->bin]
                                          .push_dbg(
                                              region,
                                              "mark.full_to_usable");
                } else {
                    list.push(region);
                }
                new_slots_published = reclaimed_slots_claimable;
            }
            alloc_reclaim_rate_diag::record_reclaim(
                alloc_reclaim_rate_diag::ReclaimKind::Mark,
                reclaimed_slots,
                reclaimed_slots * slot_size,
                new_slots_published);
        }
        if (stop_context != nullptr) {
            ++stop_context->completed_batches;
            stop_context->completed_regions += popped;
        }
    }
    return freed_size;
}

template <BlockInvoker Fn>
inline size_t GlobalHeap::mark(Fn &&fn, uint32_t timestamp,
                               std::vector<EvictTask> *ready_tasks,
                               std::atomic_size_t *ready_task_budget,
                               bool legacy_cursors,
                               EvacuationEligibility eligibility,
                               MarkBreakdownStats *mark_diag,
                               std::atomic_size_t *legacy_region_budget,
                               MarkStopContext *stop_context) {
    size_t freed_size = 0;
    auto should_stop = [&] {
        if (stop_context == nullptr || !stop_context->should_stop()) {
            return false;
        }
        stop_context->stopped = true;
        return true;
    };
    constexpr RegionPlacement placement_order[] = {
        RegionPlacement::Streaming, RegionPlacement::Unclassified,
        RegionPlacement::Mixed, RegionPlacement::Resident};
    for (RegionPlacement placement_kind : placement_order) {
        const size_t placement = placement_index(placement_kind);
        freed_size += mark_list(fn, full_region_list[placement], timestamp,
                                ready_tasks, ready_task_budget, legacy_cursors,
                                eligibility, mark_diag, legacy_region_budget,
                                stop_context);
        if (should_stop()) return freed_size;
        if (EVACUATE_PRIORITY_LARGE_OBJ) {
            int64_t smaller_limit = 0;
            if (OBJECT_SIZE_THRESHOLD > 0) {
                smaller_limit = (int64_t)get_smaller_limit();
            }
            for (int64_t i = (int64_t)RegionBinCount - 1;
                 i > smaller_limit; i--) {
                freed_size += mark_list(
                    fn, usable_region_list[placement][i], timestamp,
                    ready_tasks, ready_task_budget, legacy_cursors,
                    eligibility, mark_diag, legacy_region_budget,
                    stop_context);
                if (should_stop()) return freed_size;
            }
        } else {
            for (size_t i = 0; i < RegionBinCount; i++) {
                freed_size += mark_list(
                    fn, usable_region_list[placement][i], timestamp,
                    ready_tasks, ready_task_budget, legacy_cursors,
                    eligibility, mark_diag, legacy_region_budget,
                    stop_context);
                if (should_stop()) return freed_size;
            }
        }
    }
    return freed_size;
}

template <BlockInvoker Fn, typename ReadyTaskSink>
inline size_t GlobalHeap::mark_list_to_sink(
    Fn &&fn, RegionList &list, uint32_t timestamp, ReadyTaskSink *ready_tasks,
    std::atomic_size_t *ready_task_budget) {
    size_t freed_size = 0;
    const size_t batch_limit = 64;
    RegionHead *batch[batch_limit];
    const bool scan_past_blocked =
        region_placement_enabled ||
        (FarLib::get_config().optimized_evacuator &&
         optimized_scan_past_blocked_region_enabled());
    while (true) {
        if (ready_task_budget != nullptr &&
            ready_task_budget->load(std::memory_order_relaxed) == 0) {
            break;
        }
        int pop_reason = RegionList::PopUnmatchedEmpty;
        size_t popped = 0;
        if (scan_past_blocked) {
            auto eligible = [&](RegionHead *region) {
                if (region->profile_pending_resident()) {
                    return false;
                }
                if (region->state.load(std::memory_order::relaxed) ==
                    IN_USE) {
                    profile::count_evac_mark_region_skipped_in_use();
                    return false;
                }
                if (region->load_placement() ==
                    RegionPlacement::Resident) {
                    return region->pending_reclaims.load(
                               std::memory_order_acquire) != 0;
                }
                if (FarLib::get_config().optimized_evacuator &&
                    optimized_skip_mark_pending_enabled() &&
                    region->marked_list != nullptr) {
                    profile::count_evac_mark_region_skipped_stable();
                    return false;
                }
                if (FarLib::get_config().optimized_evacuator &&
                    optimized_mark_after_evict_enabled() &&
                    region->last_mark_epoch.load(std::memory_order_relaxed) >
                        region->evict_drained_epoch.load(
                            std::memory_order_relaxed)) {
                    profile::count_evac_mark_region_skipped_stable();
                    return false;
                }
                return true;
            };
            popped = list.pop_matching_batch(timestamp, batch, batch_limit,
                                             &pop_reason, eligible);
        } else {
            popped = list.pop_unmatched_batch(timestamp, batch, batch_limit,
                                              &pop_reason);
        }
        if (popped == 0) break;
        for (size_t i = 0; i < popped; i++) {
            RegionHead *region = batch[i];
            if (ready_task_budget != nullptr &&
                ready_task_budget->load(std::memory_order_relaxed) == 0) {
                for (size_t j = i; j < popped; j++) {
                    list.push_dbg(batch[j], "mark.ready_budget_return");
                }
                return freed_size;
            }
            if (region->profile_pending_resident()) {
                list.push_dbg(region, "skip.pending_resident");
                continue;
            }
            if (region->load_placement() == RegionPlacement::Resident) {
                reclaim_resident_region(region, list);
                continue;
            }
            if (region->state.load(std::memory_order::relaxed) ==
                IN_USE) [[unlikely]] {
                profile::count_evac_mark_region_skipped_in_use();
                list.push_dbg(region, "skip.in_use");
                continue;
            }
            if (FarLib::get_config().optimized_evacuator &&
                optimized_skip_mark_pending_enabled() &&
                region->marked_list != nullptr) {
                profile::count_evac_mark_region_skipped_stable();
                list.push_dbg(region, "skip.pending_marked");
                continue;
            }
            if (FarLib::get_config().optimized_evacuator &&
                optimized_mark_after_evict_enabled() &&
                region->last_mark_epoch.load(std::memory_order_relaxed) >
                    region->evict_drained_epoch.load(
                        std::memory_order_relaxed)) {
                profile::count_evac_mark_region_skipped_stable();
                list.push_dbg(region, "skip.pending_evict_epoch");
                continue;
            }
            profile::count_evac_mark_region_scanned();
            int64_t mark_before = profile::get_tlpd().mark_count;
            const bool was_full =
                region->state.load(std::memory_order_relaxed) == FULL;
            size_t origin_free_size = region->free_size();
            region->mark(fn);
            if (profile::get_tlpd().mark_count > mark_before) {
                profile::count_evac_mark_region_with_output();
            } else {
                profile::count_evac_mark_region_without_output();
            }
            region->note_mark_epoch(timestamp);
            int64_t delta =
                (int64_t)region->free_size() - (int64_t)origin_free_size;
            if (delta < 0) {
                ERROR("delta should >= 0");
            }
            const bool ready_for_evict = region->marked_list != nullptr;
            if (ready_for_evict && ready_tasks != nullptr) {
                if (ready_task_budget != nullptr) {
                    size_t observed =
                        ready_task_budget->load(std::memory_order_relaxed);
                    while (observed != 0 &&
                           !ready_task_budget->compare_exchange_weak(
                               observed, observed - 1,
                               std::memory_order_relaxed,
                               std::memory_order_relaxed)) {
                    }
                }
                add_free_size(-(int64_t)origin_free_size);
                detail::emit_ready_task(ready_tasks, region, timestamp);
            } else {
                if (region->marked_list == nullptr) {
                    region->note_evict_drained_epoch(timestamp);
                }
                add_free_size(delta);
                if (was_full && region->marked_list == nullptr &&
                    region->can_allocate()) {
                    region->state.store(USABLE,
                                        std::memory_order_relaxed);
                    full_to_usable_after_mark_count.fetch_add(
                        1, std::memory_order_relaxed);
                    usable_region_list[placement_index(
                                           region->load_placement())]
                                      [region->bin]
                                          .push_dbg(
                                              region,
                                              "mark.full_to_usable");
                } else {
                    list.push(region);
                }
            }
        }
    }
    return freed_size;
}

template <BlockInvoker Fn, typename ReadyTaskSink>
inline size_t GlobalHeap::mark_to_sink(
    Fn &&fn, uint32_t timestamp, ReadyTaskSink *ready_tasks,
    std::atomic_size_t *ready_task_budget) {
    size_t freed_size = 0;
    constexpr RegionPlacement placement_order[] = {
        RegionPlacement::Streaming, RegionPlacement::Unclassified,
        RegionPlacement::Mixed, RegionPlacement::Resident};
    for (RegionPlacement placement_kind : placement_order) {
        const size_t placement = placement_index(placement_kind);
        freed_size += mark_list_to_sink(
            fn, full_region_list[placement], timestamp, ready_tasks,
            ready_task_budget);
        if (EVACUATE_PRIORITY_LARGE_OBJ) {
            int64_t smaller_limit = 0;
            if (OBJECT_SIZE_THRESHOLD > 0) {
                smaller_limit = (int64_t)get_smaller_limit();
            }
            for (int64_t i = (int64_t)RegionBinCount - 1;
                 i > smaller_limit; i--) {
                freed_size += mark_list_to_sink(
                    fn, usable_region_list[placement][i], timestamp,
                    ready_tasks, ready_task_budget);
            }
        } else {
            for (size_t i = 0; i < RegionBinCount; i++) {
                freed_size += mark_list_to_sink(
                    fn, usable_region_list[placement][i], timestamp,
                    ready_tasks, ready_task_budget);
            }
        }
    }
    return freed_size;
}

inline bool GlobalHeap::collect_evict_tasks_from_list(std::vector<EvictTask> &tasks,
                                                      RegionList &list,
                                                      uint32_t timestamp,
                                                      size_t max_tasks,
                                                      uint32_t max_mark_epoch,
                                                      bool legacy_cursors,
                                                      EvacuationEligibility eligibility,
                                                      EvictCollectorDiag *evict_diag,
                                                      uint64_t scan_generation,
                                                      std::atomic_size_t *visit_budget,
                                                      bool *list_complete) {
    const size_t batch_limit = 16;
    RegionHead *batch[batch_limit];
    const bool scan_past_blocked =
        legacy_cursors ||
        eligibility == EvacuationEligibility::LegacyConcurrent ||
        region_placement_enabled ||
        (FarLib::get_config().optimized_evacuator &&
         optimized_scan_past_blocked_region_enabled());
    if (list_complete != nullptr) *list_complete = false;
    while (true) {
        if (tasks.size() >= max_tasks) return true;
        size_t to_pop = std::min(batch_limit, max_tasks - tasks.size());
        int pop_reason = RegionList::PopUnmatchedEmpty;
        size_t popped = 0;
        bool has_unscanned = false;
        bool visit_budget_exhausted = false;
        if (scan_past_blocked) {
            auto eligible = [&](RegionHead *region) {
                if (region->profile_pending_resident() &&
                    region->marked_list == nullptr) {
                    return false;
                }
                if (region->state.load(std::memory_order::relaxed) ==
                    IN_USE) {
                    profile::count_evac_evict_region_skipped_in_use();
                    return false;
                }
                if (region->load_placement() ==
                    RegionPlacement::Resident) {
                    return false;
                }
                if (max_mark_epoch != 0 &&
                    region->last_mark_epoch.load(std::memory_order::relaxed) >
                        max_mark_epoch) {
                    return false;
                }
                return evict_region_admissible(region, eligibility) &&
                       (!legacy_cursors || region->marked_list != nullptr ||
                        region->is_empty());
            };
            if (legacy_cursors) {
                const size_t reserved_visits =
                    reserve_scan_visits(visit_budget, 256);
                if (reserved_visits == 0) return true;
                size_t visited = 0;
                popped = list.pop_matching_batch_evict_cursor(
                    timestamp, batch, to_pop, reserved_visits, &pop_reason,
                    &has_unscanned, &visit_budget_exhausted, eligible,
                    evict_diag, scan_generation, false, &visited);
                refund_scan_visits(visit_budget, reserved_visits - visited);
            } else {
                popped = list.pop_matching_batch(timestamp, batch, to_pop,
                                                 &pop_reason, eligible,
                                                 nullptr, evict_diag);
            }
        } else {
            popped =
                list.pop_unmatched_batch(timestamp, batch, to_pop, &pop_reason,
                                         max_mark_epoch, nullptr, evict_diag);
        }
        if (visit_budget_exhausted && has_unscanned) {
            // The list and placement locks have been released by the pop.
            uthread::yield();
        }
        if (popped == 0) {
            if (has_unscanned) continue;
            if (pop_reason == RegionList::PopUnmatchedEmpty) {
                profile::count_evac_evict_pop_empty();
            } else if (pop_reason == RegionList::PopUnmatchedHeadMatched) {
                profile::count_evac_evict_pop_head_matched();
            }
            break;
        }
        for (size_t i = 0; i < popped; i++) {
            if (batch[i]->profile_pending_resident() &&
                batch[i]->marked_list == nullptr) {
                list.push_dbg(batch[i], "collect.skip.pending_resident");
                continue;
            }
            if (batch[i]->load_placement() ==
                RegionPlacement::Resident) {
                list.push_dbg(batch[i], "collect.skip.resident");
                continue;
            }
            if (!evict_region_admissible(batch[i], eligibility) ||
                (legacy_cursors && batch[i]->marked_list == nullptr &&
                 !batch[i]->is_empty())) {
                list.push_dbg(batch[i], "collect.skip.legacy_unmarked");
                continue;
            }
            const bool unmarked_nonempty =
                batch[i]->marked_list == nullptr && !batch[i]->is_empty();
            if (evict_diag != nullptr && unmarked_nonempty) {
                ++evict_diag->selected_unmarked_nonempty;
            }
            if (inclusive_reclaim_diag::enabled() &&
                inclusive_reclaim_diag::sample_region(batch[i])) {
                wait_trace::emit(
                    unmarked_nonempty ? "reclaim_evict_select_unmarked"
                                      : "reclaim_evict_select_marked",
                    batch[i]->bin, timestamp,
                    batch[i]->placement_epoch.load(std::memory_order_relaxed),
                    reinterpret_cast<uint64_t>(batch[i]));
            }
            profile::count_evac_evict_pop_success();
            tasks.emplace_back(batch[i], &list, timestamp);
        }
    }
    if (list_complete != nullptr) *list_complete = true;
    return false;
}

inline bool GlobalHeap::collect_evict_tasks(uint32_t timestamp,
                                            std::vector<EvictTask> &tasks,
                                            size_t max_tasks,
                                            uint32_t max_mark_epoch,
                                            bool legacy_cursors,
                                            EvacuationEligibility eligibility,
                                            EvictCollectorDiag *evict_diag,
                                            uint64_t scan_generation,
                                            std::atomic_size_t *visit_budget,
                                            std::atomic_bool *pass_complete) {
    for (size_t placement = 0; placement < RegionPlacementCount;
         ++placement) {
        if (placement == placement_index(RegionPlacement::Resident)) {
            continue;
        }
        bool list_complete = false;
        if (collect_evict_tasks_from_list(
                tasks, full_region_list[placement], timestamp, max_tasks,
                max_mark_epoch, legacy_cursors, eligibility, evict_diag,
                scan_generation, visit_budget, &list_complete)) {
            return true;
        }
        if (EVACUATE_PRIORITY_LARGE_OBJ) {
            int64_t smaller_limit = 0;
            if (OBJECT_SIZE_THRESHOLD > 0) {
                smaller_limit = (int64_t)get_smaller_limit();
            }
            for (int64_t i = (int64_t)RegionBinCount - 1;
                 i > smaller_limit; i--) {
                if (collect_evict_tasks_from_list(
                        tasks, usable_region_list[placement][i], timestamp,
                        max_tasks, max_mark_epoch, legacy_cursors,
                        eligibility, evict_diag, scan_generation,
                        visit_budget, &list_complete)) {
                    return true;
                }
            }
        } else {
            for (size_t i = 0; i < RegionBinCount; i++) {
                if (collect_evict_tasks_from_list(
                        tasks, usable_region_list[placement][i], timestamp,
                        max_tasks, max_mark_epoch, legacy_cursors,
                        eligibility, evict_diag, scan_generation,
                        visit_budget, &list_complete)) {
                    return true;
                }
            }
        }
    }
    if (pass_complete != nullptr) {
        pass_complete->store(true, std::memory_order_relaxed);
    }
    return false;
}

template <BlockInvoker Fn>
inline void GlobalHeap::process_evict_task(EvictTask &task, Fn &&fn,
                                           EvictCollectorDiag *evict_diag) {
    RegionHead *region = task.region;
    RegionList *list = task.source_list;
    if (region == nullptr) return;
    if (!task.placement_still_matches() ||
        region->load_placement() == RegionPlacement::Resident) {
        requeue_evict_task(task);
        return;
    }
    if (list == nullptr) return;
    if (region->profile_pending_resident() &&
        region->marked_list == nullptr) {
        requeue_evict_task(task);
        return;
    }
    if (region->state.load(std::memory_order::relaxed) == IN_USE) [[unlikely]] {
        profile::count_evac_evict_region_skipped_in_use();
        requeue_evict_task(task);
        return;
    }
    size_t origin_free_size = region->free_size();
    const uint32_t used_before = region->used_count;
    const size_t slot_size = get_bin_size(region->bin);
#ifdef FARLIB_ALLOC_DEBUG
    if (origin_free_size > RegionSize) {
        alloc_debug_printf(
            "[EVICT_ORIGIN_RANGE] origin_free=%zu (>RegionSize=%zu) region=%p bin=%u used=%u off=%u\n",
            origin_free_size, (size_t)RegionSize, (void *)region, region->bin,
            region->used_count, region->unused_offset);
        assert(origin_free_size <= RegionSize);
    }
#endif
    region->evict(fn);
    region->note_evict_processed_epoch();
    const uint32_t used_after = region->used_count;
    ASSERT(used_after <= used_before);
    const size_t reclaimed_slots = used_before - used_after;
    if (evict_diag != nullptr) ++evict_diag->processed_regions;
    if (region->marked_list == nullptr) {
        region->note_evict_drained_epoch(
            region->last_mark_epoch.load(std::memory_order_relaxed));
    }
    const uint64_t diag_region_ptr = reinterpret_cast<uint64_t>(region);
    const uint64_t diag_region_bin = region->bin;
    const uint64_t diag_placement_epoch = task.placement_epoch;
    const bool diag_sample_region =
        inclusive_reclaim_diag::enabled() &&
        inclusive_reclaim_diag::sample_region(region);
    auto record_published = [&](const char *event, uint64_t reclaimed_bytes) {
        if (evict_diag != nullptr) {
            evict_diag->reclaimed_bytes += reclaimed_bytes;
            if (reclaimed_bytes == 0) {
                ++evict_diag->processing_zero_reclaim;
            }
        }
        if (diag_sample_region) {
            wait_trace::emit(event, diag_region_bin, task.epoch,
                             diag_placement_epoch, diag_region_ptr);
        }
    };
    bool new_slots_published = false;
    if (region->is_empty()) {
        profile::count_evac_evict_region_to_free();
        account_region_release(region->load_placement());
        region->reset_placement(allocate_placement_epoch());
        region->state.store(FREE, std::memory_order::relaxed);
        int64_t delta = (int64_t)RegionSize - (int64_t)origin_free_size;
        if (delta < 0) {
            ERROR("delta should >= 0");
        }
        add_free_size(delta);
#if defined(FARLIB_ALLOC_DEBUG) && defined(FARLIB_ALLOC_DEBUG_FENCE)
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
        free_region_list.push_dbg(region, "evict.to_free");
        record_published("reclaim_publish_free",
                         static_cast<uint64_t>(delta));
        new_slots_published = true;
    } else if (region->can_allocate()) {
        profile::count_evac_evict_region_to_usable();
        region->state.store(USABLE, std::memory_order::relaxed);
        int64_t delta = (int64_t)region->free_size() - (int64_t)origin_free_size;
        if (delta < 0) {
            ERROR("delta should >= 0");
        }
        add_free_size(delta);
#if defined(FARLIB_ALLOC_DEBUG) && defined(FARLIB_ALLOC_DEBUG_FENCE)
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
        usable_region_list[placement_index(region->load_placement())]
                          [region->bin]
                              .push_dbg(region, "evict.to_usable");
        record_published("reclaim_publish_usable",
                         static_cast<uint64_t>(delta));
        new_slots_published = true;
    } else {
        profile::count_evac_evict_region_to_full();
        region->state.store(FULL, std::memory_order::relaxed);
        full_region_list[placement_index(region->load_placement())].push_dbg(
            region, "evict.to_full");
        record_published("reclaim_publish_full", 0);
        assert(origin_free_size == 0);
    }
    alloc_reclaim_rate_diag::record_reclaim(
        alloc_reclaim_rate_diag::ReclaimKind::Evict, reclaimed_slots,
        reclaimed_slots * slot_size, new_slots_published);
}

template <BlockInvoker Fn>
inline size_t GlobalHeap::evict_list(Fn &&fn, RegionList &list,
                                     uint32_t timestamp,
                                     EvacuationEligibility eligibility,
                                     uint64_t scan_generation,
                                     std::atomic_size_t *visit_budget,
                                     bool *list_complete,
                                     bool require_evict_processed,
                                     GcProcessedFilterStats *filter_stats) {
    size_t freed_size = 0;
    const bool scan_past_blocked =
        eligibility == EvacuationEligibility::LegacyConcurrent ||
        region_placement_enabled ||
        (FarLib::get_config().optimized_evacuator &&
         optimized_scan_past_blocked_region_enabled());
    if (list_complete != nullptr) *list_complete = false;
    while (true) {
        int pop_reason = RegionList::PopUnmatchedEmpty;
        RegionHead *region = nullptr;
        if (scan_past_blocked) {
            auto eligible = [&](RegionHead *candidate) {
                if (candidate->profile_pending_resident() &&
                    candidate->marked_list == nullptr) {
                    return false;
                }
                if (candidate->state.load(std::memory_order::relaxed) ==
                    IN_USE) {
                    profile::count_evac_evict_region_skipped_in_use();
                    return false;
                }
                if (candidate->load_placement() ==
                    RegionPlacement::Resident) {
                    return false;
                }
                if (!evict_region_admissible(candidate, eligibility)) {
                    return false;
                }
                if (require_evict_processed &&
                    !candidate->evict_processed_for_last_mark()) {
                    if (filter_stats != nullptr) {
                        ++filter_stats->rejected_checks;
                    }
                    return false;
                }
                if (require_evict_processed && filter_stats != nullptr) {
                    ++filter_stats->accepted_checks;
                }
                return true;
            };
            if (scan_generation != 0) {
                const size_t reserved_visits =
                    reserve_scan_visits(visit_budget, 256);
                if (reserved_visits == 0) return freed_size;
                RegionHead *batch[1];
                bool has_unscanned = false;
                bool visit_budget_exhausted = false;
                size_t visited = 0;
                const size_t popped = list.pop_matching_batch_evict_cursor(
                    timestamp, batch, 1, reserved_visits, &pop_reason,
                    &has_unscanned, &visit_budget_exhausted, eligible,
                    nullptr, scan_generation, true, &visited);
                refund_scan_visits(visit_budget, reserved_visits - visited);
                if (popped != 0) region = batch[0];
                if (region == nullptr && has_unscanned) continue;
            } else {
                region = list.pop_matching(timestamp, &pop_reason, eligible);
            }
        } else {
            region = list.pop_unmatched(timestamp, &pop_reason);
        }
        if (region == nullptr) {
            if (pop_reason == RegionList::PopUnmatchedEmpty) {
                profile::count_evac_evict_pop_empty();
            } else if (pop_reason == RegionList::PopUnmatchedHeadMatched) {
                profile::count_evac_evict_pop_head_matched();
            }
            break;
        }
        profile::count_evac_evict_pop_success();
        if (region->profile_pending_resident() &&
            region->marked_list == nullptr) {
            list.push_dbg(region, "evict.skip.pending_resident");
            continue;
        }
        if (region->load_placement() == RegionPlacement::Resident) {
            list.push_dbg(region, "evict.skip.resident");
            continue;
        }
        if (region->state.load(std::memory_order::relaxed) == IN_USE) [[unlikely]] {
            list.push_dbg(region, "skip.in_use");
            continue;
        }
        if (!evict_region_admissible(region, eligibility)) {
            list.push_dbg(region, "evict.skip.legacy_unmarked");
            continue;
        }
        ASSERT(!require_evict_processed ||
               region->evict_processed_for_last_mark());
        size_t origin_free_size = region->free_size();
        const uint32_t used_before = region->used_count;
        const size_t slot_size = get_bin_size(region->bin);
#ifdef FARLIB_ALLOC_DEBUG
        if (origin_free_size > RegionSize) {
            alloc_debug_printf(
                "[EVICT_ORIGIN_RANGE] origin_free=%zu (>RegionSize=%zu) region=%p bin=%u used=%u off=%u\n",
                origin_free_size, (size_t)RegionSize, (void *)region, region->bin,
                region->used_count, region->unused_offset);
            assert(origin_free_size <= RegionSize);
        }
#endif
        region->evict(fn);
        const uint32_t used_after = region->used_count;
        ASSERT(used_after <= used_before);
        const size_t reclaimed_slots = used_before - used_after;
        if (region->marked_list == nullptr) {
            region->note_evict_drained_epoch(
                region->last_mark_epoch.load(std::memory_order_relaxed));
        }
        bool new_slots_published = false;
        if (region->is_empty()) {
            account_region_release(region->load_placement());
            region->reset_placement(allocate_placement_epoch());
            region->state.store(FREE, std::memory_order::relaxed);
            int64_t delta = (int64_t)RegionSize - (int64_t)origin_free_size;
            if (delta < 0) {
                ERROR("delta should >= 0");
            }
            add_free_size(delta);
#if defined(FARLIB_ALLOC_DEBUG) && defined(FARLIB_ALLOC_DEBUG_FENCE)
            std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
            free_region_list.push_dbg(region, "evict.to_free");
            new_slots_published = true;
        } else if (region->can_allocate()) {
            region->state.store(USABLE, std::memory_order::relaxed);
            int64_t delta = (int64_t)region->free_size() - (int64_t)origin_free_size;
            if (delta < 0) {
                ERROR("delta should >= 0");
            }
            add_free_size(delta);
#if defined(FARLIB_ALLOC_DEBUG) && defined(FARLIB_ALLOC_DEBUG_FENCE)
            std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
            usable_region_list[placement_index(region->load_placement())]
                              [region->bin]
                                  .push_dbg(region, "evict.to_usable");
            new_slots_published = true;
        } else {
            region->state.store(FULL, std::memory_order::relaxed);
            full_region_list[placement_index(region->load_placement())]
                .push_dbg(region, "evict.to_full");
            assert(origin_free_size == 0);
        }
        alloc_reclaim_rate_diag::record_reclaim(
            alloc_reclaim_rate_diag::ReclaimKind::GC, reclaimed_slots,
            reclaimed_slots * slot_size,
            new_slots_published);
    }
    if (list_complete != nullptr) *list_complete = true;
    return freed_size;
}

template <BlockInvoker Fn>
inline size_t GlobalHeap::evict(Fn &&fn, uint32_t timestamp,
                                EvacuationEligibility eligibility,
                                uint64_t scan_generation,
                                std::atomic_size_t *visit_budget,
                                std::atomic_bool *pass_complete,
                                bool require_evict_processed,
                                GcProcessedFilterStats *filter_stats) {
    size_t freed_size = 0;
    for (size_t placement = 0; placement < RegionPlacementCount;
         ++placement) {
        if (placement == placement_index(RegionPlacement::Resident)) {
            continue;
        }
        bool list_complete = false;
        freed_size += evict_list(fn, full_region_list[placement], timestamp,
                                 eligibility, scan_generation, visit_budget,
                                 &list_complete, require_evict_processed,
                                 filter_stats);
        if (scan_generation != 0 && !list_complete) return freed_size;
        if (EVACUATE_PRIORITY_LARGE_OBJ) {
            int64_t smaller_limit = 0;
            if (OBJECT_SIZE_THRESHOLD > 0) {
                smaller_limit = (int64_t)get_smaller_limit();
            }
            for (int64_t i = (int64_t)RegionBinCount - 1;
                 i > smaller_limit; i--) {
                freed_size += evict_list(
                    fn, usable_region_list[placement][i], timestamp,
                    eligibility, scan_generation, visit_budget,
                    &list_complete, require_evict_processed, filter_stats);
                if (scan_generation != 0 && !list_complete) return freed_size;
            }
        } else {
            for (size_t i = 0; i < RegionBinCount; i++) {
                freed_size += evict_list(
                    fn, usable_region_list[placement][i], timestamp,
                    eligibility, scan_generation, visit_budget,
                    &list_complete, require_evict_processed, filter_stats);
                if (scan_generation != 0 && !list_complete) return freed_size;
            }
        }
    }
    if (pass_complete != nullptr) {
        pass_complete->store(true, std::memory_order_relaxed);
    }
    return freed_size;
}

inline bool GlobalHeap::memory_low() {
    return free_size.load(std::memory_order_relaxed) <
           memory_low_water_mark;
}
inline bool GlobalHeap::evacuator_continue() {
    return free_size.load(std::memory_order_relaxed) >=
           memory_high_water_mark;
}
inline bool GlobalHeap::evacuator_should_start() {
    return free_size.load(std::memory_order_relaxed) <=
           memory_low_water_mark;
}
inline bool GlobalHeap::mutator_release_ready() {
    return free_size.load(std::memory_order_relaxed) >=
           memory_release_water_mark;
}
inline bool GlobalHeap::can_allocate_region_now(size_t bin) const {
    assert(bin < RegionBinCount);
    for (size_t placement = 0; placement < RegionPlacementCount;
         ++placement) {
        if (usable_region_list[placement][bin].nonempty()) return true;
    }
    if (free_region_list.nonempty()) return true;
    size_t offset = unallocated_offset.load(std::memory_order_acquire);
    return offset + RegionSize <= heap_size;
}
inline bool GlobalHeap::can_allocate_region_directly_now(
    size_t bin, RegionPlacement requested_placement) const {
    assert(bin < RegionBinCount);
    if (!region_placement_enabled) {
        requested_placement = RegionPlacement::Unclassified;
    }
    if (usable_region_list[placement_index(requested_placement)][bin]
            .nonempty()) {
        return true;
    }
    if (requested_placement == RegionPlacement::Resident &&
        resident_reserved_regions.load(std::memory_order_acquire) >=
            resident_region_budget) {
        return false;
    }
    if (free_region_list.nonempty()) {
        return true;
    }
    const size_t offset = unallocated_offset.load(std::memory_order_acquire);
    return offset + RegionSize <= heap_size;
}
inline GlobalHeapBinSnapshot GlobalHeap::get_bin_snapshot(size_t bin) const {
    assert(bin < RegionBinCount);
    GlobalHeapBinSnapshot snapshot;
    snapshot.free_regions = free_region_list.size();
    for (size_t placement = 0; placement < RegionPlacementCount;
         ++placement) {
        snapshot.target_usable_regions +=
            usable_region_list[placement][bin].size();
        snapshot.full_regions += full_region_list[placement].size();
        for (size_t i = 0; i < RegionBinCount; i++) {
            snapshot.total_usable_regions +=
                usable_region_list[placement][i].size();
        }
    }
    const size_t committed = unallocated_offset.load(std::memory_order_acquire);
    snapshot.committed_regions = committed / RegionSize;
    snapshot.unallocated_regions =
        committed >= heap_size ? 0 : (heap_size - committed) / RegionSize;
    snapshot.can_allocate_now = can_allocate_region_now(bin);
    return snapshot;
}
inline int64_t GlobalHeap::get_memory_low_water_mark() {
    return memory_low_water_mark;
}
inline int64_t GlobalHeap::get_memory_release_water_mark() {
    return memory_release_water_mark;
}
inline int64_t GlobalHeap::get_free_size() {
    return free_size.load(std::memory_order::relaxed);
}

inline int64_t GlobalHeap::get_streaming_free_size() const {
    return free_size.load(std::memory_order_relaxed) -
        resident_counted_free_size.load(std::memory_order_relaxed);
}

}  // namespace FarLib::allocator
