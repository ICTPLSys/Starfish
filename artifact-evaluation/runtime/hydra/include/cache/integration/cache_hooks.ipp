#pragma once

namespace FarLib::cache {

inline bool ConcurrentArrayCache::six_dirty_routing_enabled() {
    return ::FarLib::simple_region_budget::six_enabled() ||
           ::FarLib::allocator::six_group::enabled();
}

inline uint32_t ConcurrentArrayCache::initial_simple_behavior_group(bool hot) {
    if (::FarLib::simple_region_budget::six_enabled())
        return (hot ? 3u : 0u) +
               FarObjectEntry::simple_dirty_class_for_q(
                   FarObjectEntry::SimpleDirtyUnknownQ);
    return hot ? ::FarLib::simple_region_heat::kHot
               : ::FarLib::simple_region_heat::kCold;
}

inline uint32_t ConcurrentArrayCache::current_behavior_group(const FarObjectEntry& entry) {
    if (::FarLib::simple_region_budget::six_enabled())
        return (entry.simple_heat_hot() ? 3u : 0u) +
               entry.simple_dirty_class();
    if(::FarLib::simple_region_heat::grouping_enabled())
        return entry.simple_heat_hot() ? ::FarLib::simple_region_heat::kHot : ::FarLib::simple_region_heat::kCold;
    return ::FarLib::allocator::six_group::enabled()
        ? ::FarLib::allocator::six_group::group_of(entry.six_record()) : 0;
}

// Caller holds the sampled object's trace lock across binding lookup and
// append.  Region locks alone cannot order a move between two Regions.
inline void ConcurrentArrayCache::trace_object_event_locked(const FarObjectEntry& entry,
                                     ::FarLib::object_group_trace::Kind kind) {
    const auto slot = entry.object_trace_slot();
    if (slot == 0) return;
    auto* record = entry.six_record();
    ASSERT(record != nullptr);
    // This diagnostic-only lock does not acquire allocator/list locks.
    // It makes the timestamped group snapshot agree with route publication.
    auto guard = ::FarLib::object_group_trace::lock_region(
        record->token.generation_id);
    ::FarLib::object_group_trace::note(slot, kind,
        record->token.generation_id,
        record->group.load(std::memory_order_acquire));
}

inline void ConcurrentArrayCache::trace_object_event(const FarObjectEntry& entry,
                              ::FarLib::object_group_trace::Kind kind) {
    const auto slot = entry.object_trace_slot();
    if (slot == 0) return;
    auto object_guard = ::FarLib::object_group_trace::lock_object(slot);
    ASSERT(entry.object_trace_slot() == slot);
    trace_object_event_locked(entry, kind);
}

inline void ConcurrentArrayCache::trace_object_allocation(FarObjectEntry& entry, size_t size) {
    if (!::FarLib::object_group_trace::enabled() ||
        !::FarLib::allocator::six_group::enabled()) return;
    auto* record = entry.six_record();
    ASSERT(record != nullptr);
    auto guard = ::FarLib::object_group_trace::lock_region(
        record->token.generation_id);
    entry.set_object_trace_slot(
        ::FarLib::object_group_trace::register_object(size, record->owner,
            record->family_bin, record->token.generation_id,
            record->group.load(std::memory_order_acquire)));
}

inline void ConcurrentArrayCache::six_bind_local(FarObjectEntry& entry, void* local, size_t size) {
    if (!::FarLib::allocator::six_group::enabled()) return;
    auto object_guard = ::FarLib::object_group_trace::lock_object(
        entry.object_trace_slot());
    auto* region = ::FarLib::allocator::block_to_region(
        static_cast<::FarLib::allocator::BlockHead*>(local) - 1);
    auto* record = region->six_record.load(std::memory_order_acquire);
    ASSERT(record != nullptr);
    ::FarLib::allocator::six_group::bind_primary(
        entry.six_binding(), record, size);
    trace_object_event_locked(entry, ::FarLib::object_group_trace::Kind::Bind);
}

inline void ConcurrentArrayCache::six_commit_fetch(FarObjectEntry& entry, void* local, size_t size) {
    if (::FarLib::simple_dirty_observer::objects_enabled())
        ::FarLib::simple_dirty_observer::monitor().fetch(
            entry.object_trace_slot(), reinterpret_cast<uintptr_t>(local),
            entry.simple_dirty_score_q(), entry.simple_dirty_score_known(),
            entry.simple_heat_hot());
    if (!::FarLib::allocator::six_group::enabled()) return;
    six_bind_local(entry, local, size);
    trace_object_event(entry, ::FarLib::object_group_trace::Kind::Fetch);
    ::FarLib::allocator::six_group::note_event(entry.six_record(),
        ::FarLib::allocator::six_group::Event::Fetch);
}

inline void ConcurrentArrayCache::six_commit_evict(FarObjectEntry& entry, size_t size, bool dirty) {
    if (::FarLib::simple_region_budget::six_enabled())
        entry.note_simple_dirty_eviction(dirty);
    // Commit hook runs under the entry move-lock in the supported exclusive
    // mode. Attribute one logical event to each endpoint, not each RDMA WR.
    if (::FarLib::simple_dirty_observer::enabled())
        ::FarLib::simple_dirty_observer::monitor().evict(
            entry.object_trace_slot(),
            reinterpret_cast<uintptr_t>(entry.local_addr()), entry.remote_addr(),
            size, dirty, entry.simple_dirty_score_q(),
            entry.simple_dirty_score_known(), entry.simple_heat_hot());
    if (!::FarLib::allocator::six_group::enabled()) return;
    auto object_guard = ::FarLib::object_group_trace::lock_object(
        entry.object_trace_slot());
    auto* remote = ::FarLib::allocator::remote::remote_global_heap.record_for(entry.remote_addr());
    ASSERT(remote != nullptr);
    // Attribute one committed eviction to the object's SOURCE group.
    // Region-level accounting below intentionally credits both endpoints.
    trace_object_event_locked(entry, dirty
        ? ::FarLib::object_group_trace::Kind::DirtyEvict
        : ::FarLib::object_group_trace::Kind::CleanEvict);
    // Both endpoint Regions observe the same logical eviction outcome.
    // Reusing a retained copy is still a CLEAN eviction, even without an
    // RDMA WRITE; excluding it biases remote Regions toward dirty arrivals.
    ::FarLib::allocator::six_group::note_eviction(
        entry.six_record(), remote, dirty);
    ::FarLib::allocator::six_group::bind_primary(entry.six_binding(), remote, size);
    trace_object_event_locked(entry, ::FarLib::object_group_trace::Kind::Bind);
}

inline void ConcurrentArrayCache::six_free(FarObjectEntry& entry, size_t size) {
    if (::FarLib::simple_dirty_observer::objects_enabled()) {
        ::FarLib::simple_dirty_observer::monitor().free_object(entry.object_trace_slot());
        entry.set_object_trace_slot(0);
    }
    if (!::FarLib::allocator::six_group::enabled()) {
        if (::FarLib::simple_region_budget::six_enabled())
            entry.take_six_pending_evict();
        return;
    }
    auto object_guard = ::FarLib::object_group_trace::lock_object(
        entry.object_trace_slot());
    trace_object_event_locked(entry, ::FarLib::object_group_trace::Kind::Free);
    entry.set_object_trace_slot(0);
    ::FarLib::allocator::six_group::note_event(entry.six_record(),
        ::FarLib::allocator::six_group::Event::Free);
    ::FarLib::allocator::six_group::clear_primary(entry.six_binding(), size);
    entry.take_six_pending_evict();
}

}  // namespace FarLib::cache
