#pragma once
#include "cache/region_based_allocator.hpp"

namespace FarLib::allocator {

inline void RegionHead::init(size_t bin, uint64_t new_placement_epoch) {
    state.store(FREE, std::memory_order::relaxed);
    placement.store(RegionPlacement::Unclassified, std::memory_order::relaxed);
    placement_group_id.store(0, std::memory_order::relaxed);
    placement_epoch.store(new_placement_epoch, std::memory_order::relaxed);
    placement_list_owned.store(false, std::memory_order::relaxed);
    placement_lock.clear(std::memory_order::relaxed);
    requeue_lock.clear(std::memory_order::relaxed);
    pending_reclaims.store(0, std::memory_order::relaxed);
    free_list = nullptr;
    active_list = nullptr;
    marked_list = nullptr;
    used_count = 0;
    unused_offset = sizeof(RegionHead);
    next_region = nullptr;
    this->bin = bin;
    sweep_time_stamp = uint32_t(-1);
    last_mark_epoch.store(0, std::memory_order_relaxed);
    evict_drained_epoch.store(0, std::memory_order_relaxed);
    evict_processed_epoch.store(0, std::memory_order_relaxed);
    profile_window_read_references.store(0, std::memory_order_relaxed);
    profile_window_write_references.store(0, std::memory_order_relaxed);
    profile_ema_read_references.store(0, std::memory_order_relaxed);
    profile_ema_write_references.store(0, std::memory_order_relaxed);
    profile_total_read_references.store(0, std::memory_order_relaxed);
    profile_total_write_references.store(0, std::memory_order_relaxed);
    profile_window_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    profile_ema_resident_fetch_intent_bytes.store(
        0, std::memory_order_relaxed);
    profile_last_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Unclassified),
        std::memory_order_relaxed);
    profile_target_streak.store(0, std::memory_order_relaxed);
    profile_pending_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Unclassified),
        std::memory_order_relaxed);
    profile_pending_plan_id.store(0, std::memory_order_relaxed);
    profile_pending_plan_rank.store(0, std::memory_order_relaxed);
#ifdef FARLIB_ALLOC_DEBUG
    dbg_enqueued_list.store(nullptr, std::memory_order::relaxed);
    dbg_last_push_tag = nullptr;
#endif
}

inline void RegionHead::lock_placement() {
    while (placement_lock.test_and_set(std::memory_order_acquire)) {
        __builtin_ia32_pause();
    }
}

inline void RegionHead::unlock_placement() {
    placement_lock.clear(std::memory_order_release);
}

inline RegionPlacement RegionHead::load_placement() const {
    return placement.load(std::memory_order_acquire);
}

inline uint32_t RegionHead::load_placement_group_id() const {
    return placement_group_id.load(std::memory_order_acquire);
}

inline uint64_t RegionHead::load_placement_epoch() const {
    return placement_epoch.load(std::memory_order_acquire);
}

inline bool RegionHead::placement_class_matches(
    RegionPlacement expected_placement) const {
    return load_placement() == expected_placement;
}

inline bool RegionHead::placement_matches(
    RegionPlacement expected_placement, uint32_t expected_group_id) const {
    if (load_placement() != expected_placement) {
        return false;
    }
    return expected_placement == RegionPlacement::Streaming &&
                   expected_group_id == 0
               ? true
               : load_placement_group_id() == expected_group_id;
}

inline void RegionHead::claim_placement(RegionPlacement new_placement,
                                        uint32_t new_group_id,
                                        uint64_t new_placement_epoch) {
    ASSERT(new_placement != RegionPlacement::Unclassified ||
           new_group_id == 0);
    ASSERT(is_empty());
    evict_processed_epoch.store(0, std::memory_order_relaxed);
    placement_group_id.store(new_group_id, std::memory_order_relaxed);
    profile_pending_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Unclassified),
        std::memory_order_relaxed);
    profile_pending_plan_id.store(0, std::memory_order_relaxed);
    profile_pending_plan_rank.store(0, std::memory_order_relaxed);
    placement.store(new_placement, std::memory_order_release);
    placement_epoch.store(new_placement_epoch, std::memory_order_release);
}

inline void RegionHead::reset_placement(uint64_t new_placement_epoch) {
    ASSERT(is_empty());
    evict_processed_epoch.store(0, std::memory_order_relaxed);
    placement_group_id.store(0, std::memory_order_relaxed);
    profile_pending_target_placement.store(
        static_cast<uint8_t>(RegionPlacement::Unclassified),
        std::memory_order_relaxed);
    profile_pending_plan_id.store(0, std::memory_order_relaxed);
    profile_pending_plan_rank.store(0, std::memory_order_relaxed);
    placement.store(RegionPlacement::Unclassified, std::memory_order_release);
    placement_epoch.store(new_placement_epoch, std::memory_order_release);
}

inline bool RegionHead::try_set_profile_pending_target(
    uint64_t expected_placement_epoch,
    RegionPlacement expected_placement,
    RegionPlacement target_placement, uint64_t plan_id,
    uint32_t plan_rank) {
    ASSERT(plan_id != 0);
    ASSERT(plan_rank != 0);
    ASSERT(expected_placement != target_placement);
    ASSERT(target_placement == RegionPlacement::Resident ||
           target_placement == RegionPlacement::Streaming);
    lock_placement();
    const bool matches =
        load_placement_epoch() == expected_placement_epoch &&
        placement_class_matches(expected_placement);
    if (matches) {
        profile_pending_plan_id.store(plan_id,
                                      std::memory_order_relaxed);
        profile_pending_plan_rank.store(plan_rank,
                                        std::memory_order_relaxed);
        profile_pending_target_placement.store(
            static_cast<uint8_t>(target_placement),
            std::memory_order_release);
    }
    unlock_placement();
    return matches;
}

inline bool RegionHead::clear_profile_pending_target(
    uint64_t expected_plan_id) {
    lock_placement();
    const uint64_t current_plan =
        profile_pending_plan_id.load(std::memory_order_relaxed);
    const bool cleared = expected_plan_id == 0 ||
                         current_plan == expected_plan_id;
    if (cleared) {
        profile_pending_target_placement.store(
            static_cast<uint8_t>(RegionPlacement::Unclassified),
            std::memory_order_release);
        profile_pending_plan_id.store(0, std::memory_order_relaxed);
        profile_pending_plan_rank.store(0, std::memory_order_relaxed);
    }
    unlock_placement();
    return cleared;
}

inline bool RegionHead::profile_pending_resident() const {
    return static_cast<RegionPlacement>(
               profile_pending_target_placement.load(
                   std::memory_order_acquire)) ==
           RegionPlacement::Resident;
}

inline bool RegionHead::profile_pending_plan_matches(
    uint64_t plan_id) const {
    return profile_pending_plan_id.load(std::memory_order_acquire) ==
           plan_id;
}

inline uint32_t RegionHead::load_profile_pending_plan_rank() const {
    return profile_pending_plan_rank.load(std::memory_order_acquire);
}

inline BlockHead *RegionHead::allocate(cache::far_obj_t obj) {
    BlockHead *block = free_list;
    if (block != nullptr) {
        assert(block->obj_meta_data == cache::far_obj_t::null());
        used_count++;
        free_list = block->next;
        block->next = active_list;
        active_list = block;
        block->obj_meta_data.store(obj, std::memory_order::relaxed);
        return block;
    }
    size_t bin_size = get_bin_size(bin);
    if (unused_offset + bin_size <= RegionSize) {
        void *r = reinterpret_cast<char *>(this) + unused_offset;
        unused_offset += bin_size;
        block = static_cast<BlockHead *>(r);
        used_count++;
        block->next = active_list;
        active_list = block;
        block->obj_meta_data.store(obj, std::memory_order::relaxed);
        return block;
    }
    return nullptr;
}

inline void RegionHead::note_mark_epoch(uint32_t epoch) {
    evict_processed_epoch.store(0, std::memory_order_relaxed);
    last_mark_epoch.store(epoch, std::memory_order_release);
}

inline void RegionHead::note_evict_drained_epoch(uint32_t epoch) {
    evict_drained_epoch.store(epoch, std::memory_order_relaxed);
}

inline void RegionHead::note_evict_processed_epoch() {
    const uint32_t epoch = last_mark_epoch.load(std::memory_order_acquire);
    if (epoch != 0) {
        evict_processed_epoch.store(epoch, std::memory_order_release);
    }
}

inline bool RegionHead::evict_processed_for_last_mark() const {
    const uint32_t epoch = last_mark_epoch.load(std::memory_order_acquire);
    return epoch != 0 &&
           evict_processed_epoch.load(std::memory_order_acquire) == epoch;
}

inline bool RegionHead::can_allocate() const {
    return free_list != nullptr || unused_offset + get_bin_size(bin) <= RegionSize;
}

inline size_t RegionHead::free_size() const {
    size_t bin_size = get_bin_size(bin);
    size_t max_block_num = (RegionSize - sizeof(RegionHead)) / bin_size;
    return (max_block_num - used_count) * bin_size;
}

inline bool RegionHead::is_empty() const {
    return active_list == nullptr && marked_list == nullptr;
}

inline bool RegionHead::has_deallocated_blocks() const {
    for (BlockHead *block = active_list; block != nullptr;
         block = block->next) {
        if (block->obj_meta_data.load(std::memory_order_acquire).is_null()) {
            return true;
        }
    }
    for (BlockHead *block = marked_list; block != nullptr;
         block = block->next) {
        if (block->obj_meta_data.load(std::memory_order_acquire).is_null()) {
            return true;
        }
    }
    return false;
}

inline size_t RegionHead::reclaim_deallocated_blocks(bool force) {
    if (pending_reclaims.exchange(0, std::memory_order_acq_rel) == 0 &&
        !force) {
        return 0;
    }
    size_t reclaimed = 0;
    BlockHead *new_active_list = nullptr;
    BlockHead *new_free_list = free_list;
    for (BlockHead *block = active_list; block != nullptr;) {
        BlockHead *next = block->next;
        if (block->obj_meta_data.load(std::memory_order_acquire).is_null()) {
            block->next = new_free_list;
            new_free_list = block;
            ASSERT(used_count != 0);
            --used_count;
            ++reclaimed;
            profile::trace_dealloc(block, bin);
        } else {
            block->next = new_active_list;
            new_active_list = block;
        }
        block = next;
    }
    BlockHead *new_marked_list = nullptr;
    for (BlockHead *block = marked_list; block != nullptr;) {
        BlockHead *next = block->next;
        if (block->obj_meta_data.load(std::memory_order_acquire).is_null()) {
            block->next = new_free_list;
            new_free_list = block;
            ASSERT(used_count != 0);
            --used_count;
            ++reclaimed;
            profile::trace_dealloc(block, bin);
        } else {
            block->next = new_marked_list;
            new_marked_list = block;
        }
        block = next;
    }
    active_list = new_active_list;
    marked_list = new_marked_list;
    free_list = new_free_list;
    return reclaimed;
}

template <BlockInvoker Fn>
inline void RegionHead::mark(Fn &&fn) {
    pending_reclaims.exchange(0, std::memory_order_acq_rel);
    BlockHead *new_active_list = nullptr;
    BlockHead *new_marked_list = marked_list;
    BlockHead *new_free_list = free_list;
    for (BlockHead *block = active_list; block;) {
        BlockHead *next_block = block->next;
        switch (fn(block)) {
        case cache::FREE:
            block->next = new_free_list;
            new_free_list = block;
            profile::trace_dealloc(block, bin);
            used_count--;
            break;
        case cache::LOCAL:
        case cache::FETCHING:
        case cache::REMOTE:
        case cache::BUSY:
        case cache::PINNED:
            block->next = new_active_list;
            new_active_list = block;
            break;
        case cache::MARKED:
        case cache::EVICTING:
            block->next = new_marked_list;
            new_marked_list = block;
            break;
        }
        block = next_block;
    }
    active_list = new_active_list;
    marked_list = new_marked_list;
    free_list = new_free_list;
}

template <BlockInvoker Fn>
inline void RegionHead::evict(Fn &&fn) {
    BlockHead *new_active_list = active_list;
    BlockHead *new_marked_list = nullptr;
    BlockHead *new_free_list = free_list;
    for (BlockHead *block = marked_list; block;) {
        BlockHead *next_block = block->next;
        switch (fn(block)) {
        case cache::FREE:
            block->next = new_free_list;
            new_free_list = block;
            used_count--;
            profile::trace_dealloc(block, bin);
            break;
        case cache::PINNED:
            std::cout << "pinned object, skip" << std::endl;
            break;
        case cache::LOCAL:
        case cache::FETCHING:
        case cache::REMOTE:
        case cache::BUSY:
            block->next = new_active_list;
            new_active_list = block;
            break;
        case cache::MARKED:
        case cache::EVICTING:
            block->next = new_marked_list;
            new_marked_list = block;
            break;
        }
        block = next_block;
    }
    marked_list = new_marked_list;
    active_list = new_active_list;
    free_list = new_free_list;
}

}  // namespace FarLib::allocator
