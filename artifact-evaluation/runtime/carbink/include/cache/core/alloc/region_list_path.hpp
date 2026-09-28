#pragma once
#include "cache/region_based_allocator.hpp"
#include "utils/uthreads.hpp"

#include <thread>

namespace FarLib::allocator {

struct RegionListLockDiagnostics {
    std::atomic<uint64_t> list_lock_calls{0};
    std::atomic<uint64_t> list_lock_cycles{0};
    std::atomic<uint64_t> placement_lock_calls{0};
    std::atomic<uint64_t> placement_lock_cycles{0};
};

inline RegionListLockDiagnostics &region_list_lock_diagnostics() {
    static auto *diag = new RegionListLockDiagnostics;
    return *diag;
}

inline bool region_list_lock_diag_enabled() {
    static const bool enabled =
        std::getenv("FARLIB_REGION_LIST_LOCK_DIAG") != nullptr;
    return enabled;
}

// RegionList is used by both libfibre workers and native background threads.
// A native thread must never call Fibre::yieldGlobal(), which requires a
// current Fred. The online placement planner marks itself native here.
inline thread_local bool region_list_fibre_yield_allowed = true;

inline void disable_region_list_fibre_yield_for_current_thread() {
    region_list_fibre_yield_allowed = false;
}

inline void acquire_region_list_lock(std::atomic_flag &lock) {
    static const bool yield_enabled =
        std::getenv("FARLIB_REGION_LIST_YIELD_LOCK") != nullptr;
    const bool diag_enabled = region_list_lock_diag_enabled();
    const uint64_t start = diag_enabled ? get_cycles() : 0;
    uint32_t spins = 0;
    while (lock.test_and_set(std::memory_order_acquire)) {
        if (yield_enabled && ++spins == 64) {
            if (region_list_fibre_yield_allowed) {
                uthread::yield();
            } else {
                std::this_thread::yield();
            }
            spins = 0;
        }
    }
    if (diag_enabled) {
        auto &diag = region_list_lock_diagnostics();
        diag.list_lock_calls.fetch_add(1, std::memory_order_relaxed);
        diag.list_lock_cycles.fetch_add(get_cycles() - start,
                                        std::memory_order_relaxed);
    }
}

inline void acquire_region_placement_lock(RegionHead *region) {
    if (!region_list_lock_diag_enabled()) {
        region->lock_placement();
        return;
    }
    const uint64_t start = get_cycles();
    region->lock_placement();
    auto &diag = region_list_lock_diagnostics();
    diag.placement_lock_calls.fetch_add(1, std::memory_order_relaxed);
    diag.placement_lock_cycles.fetch_add(get_cycles() - start,
                                         std::memory_order_relaxed);
}

inline RegionList::RegionList()
    : head(&head),
      tail(&head),
      item_count(0),
      simple_budget_scan_cursor(&head),
      mark_scan_epoch(0),
      mark_scan_initialized(false),
      mark_scan_done(false),
      mark_scan_prev(&head),
      mark_scan_last(&head),
      evict_scan_epoch(0),
      evict_scan_initialized(false),
      evict_scan_done(false),
      evict_scan_prev(&head),
      evict_scan_last(&head),
      gc_scan_epoch(0),
      gc_scan_initialized(false),
      gc_scan_done(false),
      gc_scan_prev(&head),
      gc_scan_last(&head) {}

inline void RegionList::enable_six_group_index() {
    // Called during GlobalHeap registration before allocator threads can
    // publish Regions.  Keeping buckets in the map (rather than pointers to
    // map values) makes later rehashes harmless to live intrusive links.
    six_group_buckets.reserve(64);
    six_group_index_enabled = true;
}

inline void RegionList::enable_list_only_six_children() {
    ASSERT(item_count.load(std::memory_order_relaxed) == 0);
    ASSERT(!list_only_six_children && list_only_six_parent == nullptr);
    list_only_six_children =
        std::make_unique<RegionList[]>(list_only_six::kChildCount);
    for (size_t i = 0; i < list_only_six::kChildCount; ++i) {
        list_only_six_children[i].list_only_six_parent = this;
    }
}

inline RegionList *RegionList::list_only_six_child(size_t slot) const {
    if (!list_only_six_children || slot >= list_only_six::kChildCount)
        return nullptr;
    return &list_only_six_children[slot];
}

inline size_t RegionList::reclassify_simple_heat() {
    // With a soft supply budget, child-list membership is a persistent
    // allocation label.  Heat windows still update Classes::labels_, but
    // must not move live/public Regions between supply classes.
    if (simple_region_budget::enabled() ||
        simple_region_heat::local_resident_enabled())
        return 0;
    if(!list_only_six_children)return 0;
    size_t moved=0;
    for(size_t c=0;c<list_only_six::kChildCount;++c) {
        std::vector<RegionHead *> detached;
        list_only_six_children[c].extract_all_matching(detached,[c](RegionHead *r) {
            return simple_region_heat::class_for(reinterpret_cast<uintptr_t>(r))!=c;
        });
        moved+=detached.size();
        push_batch_dbg(detached.data(),detached.size(),"simple_heat.reclassify");
    }
    return moved;
}

inline void RegionList::add_six_group_index_locked(RegionHead *node,
                                                   bool front) {
    ASSERT(node->six_indexed_group_id == 0);
    if (!six_group_index_enabled) return;
    const uint32_t group = node->six_group_id();
    if (group == 0) return;
    auto &bucket = six_group_buckets[group];
    node->six_indexed_group_id = group;
    if (front) {
        node->six_group_prev = nullptr;
        node->six_group_next = bucket.first;
        if (bucket.first != nullptr) bucket.first->six_group_prev = node;
        else bucket.last = node;
        bucket.first = node;
    } else {
        node->six_group_prev = bucket.last;
        node->six_group_next = nullptr;
        if (bucket.last != nullptr) bucket.last->six_group_next = node;
        else bucket.first = node;
        bucket.last = node;
    }
}

inline void RegionList::remove_six_group_index_locked(RegionHead *node) {
    const uint32_t group = node->six_indexed_group_id;
    if (group == 0) return;
    auto found = six_group_buckets.find(group);
    if (found != six_group_buckets.end()) {
        auto &bucket = found->second;
        if (node->six_group_prev != nullptr)
            node->six_group_prev->six_group_next = node->six_group_next;
        else
            bucket.first = node->six_group_next;
        if (node->six_group_next != nullptr)
            node->six_group_next->six_group_prev = node->six_group_prev;
        else
            bucket.last = node->six_group_prev;
    }
    node->six_group_prev = nullptr;
    node->six_group_next = nullptr;
    node->six_indexed_group_id = 0;
}

inline void RegionList::reindex_six_group_locked(RegionHead *node) {
    ASSERT(node->owner_list.load(std::memory_order_relaxed) == this);
    remove_six_group_index_locked(node);
    add_six_group_index_locked(node);
}

inline void RegionList::set_owner_locked(RegionHead *node, bool front) {
    ASSERT(node->owner_list.load(std::memory_order_relaxed) == nullptr);
    node->owner_list.store(this, std::memory_order_release);
    add_six_group_index_locked(node, front);
}

inline void RegionList::clear_owner_locked(RegionHead *node) {
    ASSERT(node->owner_list.load(std::memory_order_relaxed) == this);
    simple_budget_cursor_on_unlink_locked(node, node->next_region);
    remove_six_group_index_locked(node);
    node->owner_list.store(nullptr, std::memory_order_release);
}

inline bool RegionList::mark_cursor_active_locked() const {
    return mark_scan_initialized && !mark_scan_done;
}

inline void RegionList::finish_mark_cursor_locked() {
    mark_scan_done = true;
    // Keep the epoch/done state, but release node references so a removed and
    // later re-enqueued RegionHead cannot be mistaken for unfinished work.
    mark_scan_prev = &head;
    mark_scan_last = &head;
}

inline void RegionList::mark_cursor_on_unlink_locked(
    RegionListNodeBase *prev, RegionListNodeBase *node) {
    if (!mark_cursor_active_locked()) return;

    const bool removed_prev = node == mark_scan_prev;
    const bool removed_last = node == mark_scan_last;
    if (removed_prev) {
        mark_scan_prev = prev;
    }
    if (removed_last) {
        // The removed boundary is no longer eligible for this pass.  Its
        // predecessor becomes the last surviving member of the captured set.
        // If that predecessor is exactly the cursor, the pass is complete.
        mark_scan_last = prev;
        if (removed_prev || mark_scan_prev == prev) {
            finish_mark_cursor_locked();
        }
    }
}

inline void RegionList::mark_cursor_on_prepend_locked(
    RegionListNodeBase *new_prefix_last) {
    if (mark_cursor_active_locked() && mark_scan_prev == &head) {
        // The prefix was not present at epoch initialization.  Keep it behind
        // the cursor so it is first considered in the next mark epoch.
        mark_scan_prev = new_prefix_last;
    }
}

inline bool RegionList::evict_cursor_active_locked() const {
    return evict_scan_initialized && !evict_scan_done;
}

inline void RegionList::finish_evict_cursor_locked() {
    evict_scan_done = true;
    evict_scan_prev = &head;
    evict_scan_last = &head;
}

inline void RegionList::evict_cursor_on_unlink_locked(
    RegionListNodeBase *prev, RegionListNodeBase *node) {
    if (!evict_cursor_active_locked()) return;

    const bool removed_prev = node == evict_scan_prev;
    const bool removed_last = node == evict_scan_last;
    if (removed_prev) {
        evict_scan_prev = prev;
    }
    if (removed_last) {
        evict_scan_last = prev;
        if (removed_prev || evict_scan_prev == prev) {
            finish_evict_cursor_locked();
        }
    }
}

inline void RegionList::evict_cursor_on_prepend_locked(
    RegionListNodeBase *new_prefix_last) {
    if (evict_cursor_active_locked() && evict_scan_prev == &head) {
        evict_scan_prev = new_prefix_last;
    }
}

inline bool RegionList::gc_cursor_active_locked() const {
    return gc_scan_initialized && !gc_scan_done;
}

inline void RegionList::finish_gc_cursor_locked() {
    gc_scan_done = true;
    gc_scan_prev = &head;
    gc_scan_last = &head;
}

inline void RegionList::gc_cursor_on_unlink_locked(
    RegionListNodeBase *prev, RegionListNodeBase *node) {
    if (!gc_cursor_active_locked()) return;
    const bool removed_prev = node == gc_scan_prev;
    const bool removed_last = node == gc_scan_last;
    if (removed_prev) gc_scan_prev = prev;
    if (removed_last) {
        gc_scan_last = prev;
        if (removed_prev || gc_scan_prev == prev) finish_gc_cursor_locked();
    }
}

inline void RegionList::gc_cursor_on_prepend_locked(
    RegionListNodeBase *new_prefix_last) {
    if (gc_cursor_active_locked() && gc_scan_prev == &head) {
        gc_scan_prev = new_prefix_last;
    }
}

inline void RegionList::simple_budget_cursor_on_unlink_locked(
    RegionListNodeBase *node, RegionListNodeBase *next) {
    if (simple_budget_scan_cursor == node) {
        simple_budget_scan_cursor = next == &head ? &head : next;
    }
}

inline void RegionList::push(RegionHead *node) { push_dbg(node, nullptr); }

inline void RegionList::push_dbg(RegionHead *node, const char *tag,
                                 MarkBreakdownStats *mark_diag) {
    if (simple_region_heat::local_resident_enabled() && node != nullptr) {
        const RegionPlacement placement = node->load_placement();
        if (placement == RegionPlacement::Resident ||
            placement == RegionPlacement::Streaming) {
            // Synchronize before taking either list lock. The heat ledger is
            // independent of intrusive-list ownership, and doing this before
            // the lock avoids a list-lock -> budget-lock inversion.
            simple_region_heat::sync_supply_placement(
                reinterpret_cast<uintptr_t>(node), node->bin,
                placement == RegionPlacement::Resident);
        }
    }
    if (list_only_six_children) {
        if (list_only_six::grouped()) {
            const size_t c = list_only_six_next_push.fetch_add(
                1, std::memory_order_relaxed) % list_only_six::kChildCount;
            list_only_six_children[c].push_dbg(node, tag, mark_diag);
            return;
        }
        // In the ordinary list-only mode, all normal Regions use fixed class
        // zero unless the independent Hot/Cold experiment is enabled.
        const size_t c=simple_region_heat::grouping_enabled()
            ? simple_region_heat::allocation_class_for(
                  reinterpret_cast<uintptr_t>(node))
            : 0;
        list_only_six_children[c].push_dbg(node, tag, mark_diag);
        return;
    }
    if (list_only_six_parent == nullptr) node->ensure_six_group();
    const bool segment_diag =
        mark_diag != nullptr && mark_diag->segment_enabled;
    const uint64_t before_list_lock = segment_diag ? get_cycles() : 0;
    acquire_region_list_lock(lock);
    const uint64_t after_list_lock = segment_diag ? get_cycles() : 0;
    if(simple_region_heat::grouping_enabled() && list_only_six_parent &&
       list_only_six_parent->list_only_six_child(
           simple_region_heat::allocation_class_for(
               reinterpret_cast<uintptr_t>(node))) != this) {
        lock.clear(std::memory_order_release);
        list_only_six_parent->push_dbg(node,tag,mark_diag);
        return;
    }
#ifdef FARLIB_ALLOC_DEBUG
    // Debug-only invariant: a node must not already be linked in any list.
    // We enforce this by setting node->next_region = nullptr on successful pop.
    if (node->next_region != nullptr) {
        alloc_debug_printf(
            "[REGION_LIST_BUG] double-push? list=%p tag=%s node=%p next=%p state=%u bin=%u stamp=%u last_tag=%s\n",
            (void *)this, (tag ? tag : "(null)"), (void *)node,
            (void *)node->next_region,
            (unsigned)node->state.load(std::memory_order::relaxed),
            (unsigned)node->bin, (unsigned)node->sweep_time_stamp,
            (node->dbg_last_push_tag ? node->dbg_last_push_tag : "(null)"));
        assert(node->next_region == nullptr &&
               "RegionHead double-push without pop");
    }
    // Cross-list double-enqueue detection.
    void *expected = nullptr;
    if (!node->dbg_enqueued_list.compare_exchange_strong(
            expected, (void *)this, std::memory_order::relaxed)) {
        alloc_debug_printf(
            "[REGION_LIST_BUG] already-enqueued list_prev=%p list_now=%p tag=%s node=%p state=%u bin=%u stamp=%u last_tag=%s\n",
            expected, (void *)this, (tag ? tag : "(null)"), (void *)node,
            (unsigned)node->state.load(std::memory_order::relaxed),
            (unsigned)node->bin, (unsigned)node->sweep_time_stamp,
            (node->dbg_last_push_tag ? node->dbg_last_push_tag : "(null)"));
        assert(false && "RegionHead enqueued in multiple lists");
    }
    node->dbg_last_push_tag = tag;
#endif
    assert(tail->next_region == &head);
    assert(node->prev_region == nullptr && node->next_region == nullptr);
    RegionListNodeBase *old_tail = tail;
    node->next_region = &head;
    node->prev_region = old_tail;
    old_tail->next_region = node;
    head.prev_region = node;
    tail = node;
    set_owner_locked(node);
    const uint64_t before_placement_lock = segment_diag ? get_cycles() : 0;
    acquire_region_placement_lock(node);
    const uint64_t after_placement_lock = segment_diag ? get_cycles() : 0;
    ASSERT(!node->placement_list_owned.load(std::memory_order_relaxed));
    node->placement_list_owned.store(true, std::memory_order_release);
    node->unlock_placement();
    item_count.fetch_add(1, std::memory_order_relaxed);
    const uint64_t before_list_unlock = segment_diag ? get_cycles() : 0;
    lock.clear();
    if (segment_diag) {
        ++mark_diag->push_list_lock_calls;
        mark_diag->push_list_lock_wait_cycles +=
            after_list_lock - before_list_lock;
        mark_diag->push_list_lock_hold_cycles +=
            before_list_unlock - after_list_lock;
        mark_diag->push_placement_lock_cycles +=
            after_placement_lock - before_placement_lock;
    }
}

inline void RegionList::push_batch_dbg(RegionHead **nodes, size_t count,
                                       const char *tag) {
    if (count == 0) return;

    if (simple_region_heat::local_resident_enabled()) {
        for (size_t i = 0; i < count; ++i) {
            RegionHead *node = nodes[i];
            if (node == nullptr) continue;
            const RegionPlacement placement = node->load_placement();
            if (placement == RegionPlacement::Resident ||
                placement == RegionPlacement::Streaming) {
                simple_region_heat::sync_supply_placement(
                    reinterpret_cast<uintptr_t>(node), node->bin,
                    placement == RegionPlacement::Resident);
            }
        }
    }

    if (list_only_six_children) {
        if (list_only_six::grouped()) {
            std::vector<RegionHead *> buckets[list_only_six::kChildCount];
            for (size_t i = 0; i < count; ++i) {
                const size_t c = list_only_six_next_push.fetch_add(
                    1, std::memory_order_relaxed) % list_only_six::kChildCount;
                buckets[c].push_back(nodes[i]);
            }
            for (size_t c = 0; c < list_only_six::kChildCount; ++c)
                list_only_six_children[c].push_batch_dbg(
                    buckets[c].data(), buckets[c].size(), tag);
            return;
        }
        if(simple_region_heat::grouping_enabled()) {
            // Keep batch locking; partition every semantic class before
            // enqueue.  In legacy mode allocation labels are 1/4; in six
            // mode they are the identity labels 0..5.
            std::vector<RegionHead *> buckets[list_only_six::kChildCount];
            for (size_t i = 0; i < count; ++i) {
                const size_t c = simple_region_heat::allocation_class_for(
                    reinterpret_cast<uintptr_t>(nodes[i]));
                ASSERT(c < list_only_six::kChildCount);
                buckets[c].push_back(nodes[i]);
            }
            for (size_t c = 0; c < list_only_six::kChildCount; ++c)
                list_only_six_children[c].push_batch_dbg(
                    buckets[c].data(), buckets[c].size(), tag);
            return;
        }
        list_only_six_children[0].push_batch_dbg(nodes, count, tag);
        return;
    }

    if (list_only_six_parent == nullptr) {
        for (size_t i = 0; i < count; ++i) {
            ASSERT(nodes[i] != nullptr);
            nodes[i]->ensure_six_group();
        }
    }

    acquire_region_list_lock(lock);
    if(simple_region_heat::grouping_enabled() && list_only_six_parent) {
        bool stale=false;
        for(size_t i=0;i<count;++i)
            stale |= list_only_six_parent->list_only_six_child(
                         simple_region_heat::allocation_class_for(
                             reinterpret_cast<uintptr_t>(nodes[i]))) != this;
        if(stale) {
            lock.clear(std::memory_order_release);
            list_only_six_parent->push_batch_dbg(nodes,count,tag);
            return;
        }
    }
    assert(tail->next_region == &head);
    for (size_t i = 0; i < count; ++i) {
        RegionHead *node = nodes[i];
        assert(node != nullptr);
#ifdef FARLIB_ALLOC_DEBUG
        if (node->next_region != nullptr) {
            alloc_debug_printf(
                "[REGION_LIST_BUG] batch double-push? list=%p tag=%s node=%p next=%p state=%u bin=%u stamp=%u last_tag=%s\n",
                (void *)this, (tag ? tag : "(null)"), (void *)node,
                (void *)node->next_region,
                (unsigned)node->state.load(std::memory_order_relaxed),
                (unsigned)node->bin, (unsigned)node->sweep_time_stamp,
                (node->dbg_last_push_tag ? node->dbg_last_push_tag
                                         : "(null)"));
            assert(node->next_region == nullptr &&
                   "RegionHead batch double-push without pop");
        }
        void *expected = nullptr;
        if (!node->dbg_enqueued_list.compare_exchange_strong(
                expected, (void *)this, std::memory_order_relaxed)) {
            alloc_debug_printf(
                "[REGION_LIST_BUG] batch already-enqueued list_prev=%p list_now=%p tag=%s node=%p state=%u bin=%u stamp=%u last_tag=%s\n",
                expected, (void *)this, (tag ? tag : "(null)"),
                (void *)node,
                (unsigned)node->state.load(std::memory_order_relaxed),
                (unsigned)node->bin, (unsigned)node->sweep_time_stamp,
                (node->dbg_last_push_tag ? node->dbg_last_push_tag
                                         : "(null)"));
            assert(false && "RegionHead enqueued in multiple lists");
        }
        node->dbg_last_push_tag = tag;
#endif
        ASSERT(node->prev_region == nullptr && node->next_region == nullptr);
        node->next_region = &head;
        node->prev_region = tail;
        tail->next_region = node;
        head.prev_region = node;
        tail = node;
        set_owner_locked(node);
        acquire_region_placement_lock(node);
        ASSERT(!node->placement_list_owned.load(std::memory_order_relaxed));
        node->placement_list_owned.store(true, std::memory_order_release);
        node->unlock_placement();
    }
    item_count.fetch_add(count, std::memory_order_relaxed);
    lock.clear();
}

inline RegionHead *RegionList::pop() {
    if (list_only_six_children) {
        if (list_only_six::grouped()) {
            const size_t start = list_only_six_next_pop.fetch_add(
                1, std::memory_order_relaxed) % list_only_six::kChildCount;
            for (size_t i = 0; i < list_only_six::kChildCount; ++i) {
                const size_t c = (start + i) % list_only_six::kChildCount;
                if (auto *r = list_only_six_children[c].pop()) return r;
            }
            return nullptr;
        }
        if(simple_region_heat::grouping_enabled()) {
            if (simple_region_budget::six_enabled()) {
                const size_t count = simple_region_budget::classes();
                const size_t start = list_only_six_next_pop.fetch_add(
                    1, std::memory_order_relaxed) % count;
                for (size_t i = 0; i < count; ++i) {
                    const size_t c = (start + i) % count;
                    if (auto *r = list_only_six_children[c].pop()) return r;
                }
                return nullptr;
            }
            if(auto *r=list_only_six_children[simple_region_heat::kCold].pop())return r;
            return list_only_six_children[simple_region_heat::kHot].pop();
        }
        return list_only_six_children[0].pop();
    }
    acquire_region_list_lock(lock);
    RegionListNodeBase *node = head.next_region;
    if (node == &head) {
        lock.clear();
        return nullptr;
    } else {
        auto *region = static_cast<RegionHead *>(node);
        acquire_region_placement_lock(region);
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(&head, node);
        evict_cursor_on_unlink_locked(&head, node);
        gc_cursor_on_unlink_locked(&head, node);
        head.next_region = node->next_region;
        node->next_region->prev_region = &head;
        if (tail == node) tail = &head;
        head.prev_region = tail;
        // Mark as unlinked so a later push can detect double-enqueue.
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        static_cast<RegionHead *>(node)->dbg_enqueued_list.store(
            nullptr, std::memory_order::relaxed);
#endif
        item_count.fetch_sub(1, std::memory_order_relaxed);
        lock.clear();
        return region;
    }
}

template <typename EligibleFn>
inline RegionHead *RegionList::pop_for_six_group(uint32_t group_id,
                                                 EligibleFn &&eligible) {
    if (list_only_six_children) {
        if (list_only_six::grouped()) {
            return list_only_six_children[group_id % list_only_six::kChildCount]
                .pop_first_matching(std::forward<EligibleFn>(eligible));
        }
        // List-only mode has no group metadata.  All Regions are class zero.
        return list_only_six_children[0].pop_for_six_group(
            group_id, std::forward<EligibleFn>(eligible));
    }
    if (group_id == 0 || !six_group_index_enabled) return nullptr;
    acquire_region_list_lock(lock);
    auto found = six_group_buckets.find(group_id);
    RegionHead *region = found == six_group_buckets.end()
                             ? nullptr
                             : found->second.first;
    while (region != nullptr) {
        RegionHead *next_group = region->six_group_next;
        if (region->owner_list.load(std::memory_order_relaxed) != this ||
            region->six_group_id() != group_id) {
            if (region->owner_list.load(std::memory_order_relaxed) == this) {
                remove_six_group_index_locked(region);
                add_six_group_index_locked(region);
            }
            region = next_group;
            continue;
        }
        acquire_region_placement_lock(region);
        if (!eligible(region)) {
            region->unlock_placement();
            region = next_group;
            continue;
        }
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        RegionListNodeBase *prev = region->prev_region;
        RegionListNodeBase *next = region->next_region;
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(prev, region);
        evict_cursor_on_unlink_locked(prev, region);
        gc_cursor_on_unlink_locked(prev, region);
        prev->next_region = next;
        next->prev_region = prev;
        if (tail == region) tail = prev;
        head.prev_region = tail;
        region->next_region = nullptr;
        region->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr, std::memory_order_relaxed);
#endif
        item_count.fetch_sub(1, std::memory_order_relaxed);
        lock.clear(std::memory_order_release);
        return region;
    }
    lock.clear(std::memory_order_release);
    return nullptr;
}

template <typename EligibleFn>
inline RegionHead *RegionList::pop_for_group(uint32_t group_id,
                                             EligibleFn &&eligible) {
    return pop_for_six_group(group_id, std::forward<EligibleFn>(eligible));
}

inline void RegionList::append_region_snapshot(
    std::vector<RegionListSnapshotHandle> &regions) {
    if (list_only_six_children) {
        for (size_t i = 0; i < list_only_six::kChildCount; ++i)
            list_only_six_children[i].append_region_snapshot(regions);
        return;
    }
    acquire_region_list_lock(lock);
    for (RegionListNodeBase *node = head.next_region; node != &head;
         node = node->next_region) {
        auto *region = static_cast<RegionHead *>(node);
        acquire_region_placement_lock(region);
        if (region->placement_list_owned.load(std::memory_order_relaxed)) {
            regions.push_back(
                {.region = region,
                 .placement_epoch = region->load_placement_epoch(),
                 .placement = region->load_placement()});
        }
        region->unlock_placement();
    }
    lock.clear(std::memory_order_release);
}

inline void RegionList::append_simple_budget_supply(
    simple_region_budget::Supply &supply) {
    if (list_only_six_children) {
        // Child lists are independently locked.  Recurse instead of reading
        // their intrusive links here so a concurrent pop cannot race the
        // free-size read.  This also handles the exploratory six-child shape
        // without assuming that only children 1 and 4 contain Regions.
        for (size_t i = 0; i < list_only_six::kChildCount; ++i)
            list_only_six_children[i].append_simple_budget_supply(supply);
        return;
    }

    acquire_region_list_lock(lock);
    for (RegionListNodeBase *node = head.next_region; node != &head;
         node = node->next_region) {
        auto *region = static_cast<RegionHead *>(node);
        acquire_region_placement_lock(region);
        if (region->placement_list_owned.load(std::memory_order_relaxed)) {
            const RegionState state =
                region->state.load(std::memory_order_relaxed);
            // FREE Regions contribute their whole physical extent.  USABLE
            // Regions contribute only allocator slack; FULL/IN_USE Regions
            // are not expected on a public list and are excluded defensively.
            const uint64_t bytes =
                state == FREE
                    ? static_cast<uint64_t>(RegionSize)
                    : (state == USABLE
                           ? static_cast<uint64_t>(region->free_size())
                           : uint64_t{0});
            if (bytes != 0) {
                if (region->bin < simple_region_budget::kBins) {
                    const size_t class_index = simple_region_budget::index(
                        simple_region_heat::allocation_class_for(
                            reinterpret_cast<uintptr_t>(region)));
                    supply[region->bin][class_index] += bytes;
                }
            }
        }
        region->unlock_placement();
    }
    lock.clear(std::memory_order_release);
}

inline size_t RegionList::extract_simple_budget_usable(
    std::vector<RegionHead *> &out, simple_region_budget::Supply &remaining,
    size_t bin, size_t max_count, size_t visit_limit,
    SimpleBudgetExtractStats &stats) {
    // Detach only list-owned USABLE Regions with remaining free slots.  The
    // caller must preserve their headers and republish them after relabeling.
    if (bin >= simple_region_budget::kBins || max_count == 0 || visit_limit == 0)
        return 0;
    if (list_only_six_children) {
        size_t extracted = 0;
        for (size_t donor = 0;
             donor < simple_region_budget::classes() && extracted < max_count &&
             stats.scanned < visit_limit;
             ++donor) {
            if (remaining[bin][donor] == 0) continue;
            const size_t child = simple_region_budget::label(donor);
            extracted += list_only_six_children[child]
                             .extract_simple_budget_usable(
                                 out, remaining, bin, max_count - extracted,
                                 visit_limit, stats);
        }
        return extracted;
    }

    // The native monitor must never wait on a Region placement lock while
    // holding the list lock: mutator/deallocator paths can hold the inverse
    // order.  Probe placement_lock directly and skip busy Regions instead.
    acquire_region_list_lock(lock);
    RegionListNodeBase *node = simple_budget_scan_cursor;
    if (node == nullptr || node == &head) node = head.next_region;
    size_t extracted = 0;
    while (node != &head && extracted < max_count &&
           stats.scanned < visit_limit) {
        RegionListNodeBase *next = node->next_region;
        ++stats.scanned;
        auto *region = static_cast<RegionHead *>(node);
        if (region->placement_lock.test_and_set(std::memory_order_acquire)) {
            ++stats.busy;
            node = next;
            continue;
        }

        const size_t region_bin = region->bin;
        const size_t region_class = simple_region_budget::index(
            simple_region_heat::allocation_class_for(
                reinterpret_cast<uintptr_t>(region)));
        const bool eligible =
            region->owner_list.load(std::memory_order_relaxed) == this &&
            region->placement_list_owned.load(std::memory_order_relaxed) &&
            region->state.load(std::memory_order_relaxed) == USABLE &&
            region_bin == bin && region->can_allocate() &&
            region->free_size() != 0 &&
            region_bin < simple_region_budget::kBins &&
            remaining[region_bin][region_class] != 0;
        if (!eligible) {
            region->unlock_placement();
            node = next;
            continue;
        }

        ++stats.eligible;
        --remaining[region_bin][region_class];
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(node->prev_region, node);
        evict_cursor_on_unlink_locked(node->prev_region, node);
        gc_cursor_on_unlink_locked(node->prev_region, node);
        RegionListNodeBase *prev = node->prev_region;
        prev->next_region = next;
        next->prev_region = prev;
        if (tail == node) tail = prev;
        head.prev_region = tail;
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr, std::memory_order_relaxed);
#endif
        out.push_back(region);
        ++extracted;
        node = next;
    }
    if (extracted != 0)
        item_count.fetch_sub(extracted, std::memory_order_relaxed);
    simple_budget_scan_cursor = node == &head ? &head : node;
    lock.clear(std::memory_order_release);
    return extracted;
}

template <typename EligibleFn>
inline RegionHead *RegionList::pop_first_matching(EligibleFn &&eligible) {
    if (list_only_six_children) {
        if (list_only_six::grouped()) {
            const size_t start = list_only_six_next_pop.fetch_add(
                1, std::memory_order_relaxed) % list_only_six::kChildCount;
            for (size_t i = 0; i < list_only_six::kChildCount; ++i) {
                const size_t c = (start + i) % list_only_six::kChildCount;
                if (auto *r = list_only_six_children[c].pop_first_matching(
                        std::forward<EligibleFn>(eligible))) return r;
            }
            return nullptr;
        }
        if(simple_region_heat::grouping_enabled()) {
            if (simple_region_budget::six_enabled()) {
                const size_t count = simple_region_budget::classes();
                const size_t start = list_only_six_next_pop.fetch_add(
                    1, std::memory_order_relaxed) % count;
                for (size_t i = 0; i < count; ++i) {
                    const size_t c = (start + i) % count;
                    if (auto *r = list_only_six_children[c].pop_first_matching(
                            std::forward<EligibleFn>(eligible))) return r;
                }
                return nullptr;
            }
            if(auto *r=list_only_six_children[simple_region_heat::kCold].pop_first_matching(eligible))return r;
            return list_only_six_children[simple_region_heat::kHot].pop_first_matching(eligible);
        }
        return list_only_six_children[0].pop_first_matching(
            std::forward<EligibleFn>(eligible));
    }
    acquire_region_list_lock(lock);
    RegionListNodeBase *prev = &head;
    RegionListNodeBase *node = head.next_region;
    while (node != &head) {
        auto *region = static_cast<RegionHead *>(node);
        RegionListNodeBase *next = node->next_region;
        acquire_region_placement_lock(region);
        if (!eligible(region)) {
            region->unlock_placement();
            prev = node;
            node = next;
            continue;
        }
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(prev, node);
        evict_cursor_on_unlink_locked(prev, node);
        gc_cursor_on_unlink_locked(prev, node);
        prev->next_region = next;
        next->prev_region = prev;
        if (tail == node) tail = prev;
        head.prev_region = tail;
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr, std::memory_order_relaxed);
#endif
        item_count.fetch_sub(1, std::memory_order_relaxed);
        lock.clear();
        return region;
    }
    lock.clear();
    return nullptr;
}

template <typename EligibleFn>
inline size_t RegionList::extract_all_matching(
    std::vector<RegionHead *> &out, EligibleFn &&eligible) {
    if (list_only_six_children) {
        size_t count = 0;
        for (size_t i = 0; i < list_only_six::kChildCount; ++i)
            count += list_only_six_children[i].extract_all_matching(
                out, std::forward<EligibleFn>(eligible));
        return count;
    }
    acquire_region_list_lock(lock);
    size_t count = 0;
    RegionListNodeBase *prev = &head;
    RegionListNodeBase *node = head.next_region;
    while (node != &head) {
        auto *region = static_cast<RegionHead *>(node);
        RegionListNodeBase *next = node->next_region;
        acquire_region_placement_lock(region);
        if (!eligible(region)) {
            region->unlock_placement();
            prev = node;
            node = next;
            continue;
        }
        ASSERT(region->placement_list_owned.load(
            std::memory_order_relaxed));
        region->placement_list_owned.store(false,
                                           std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(prev, node);
        evict_cursor_on_unlink_locked(prev, node);
        gc_cursor_on_unlink_locked(prev, node);
        prev->next_region = next;
        next->prev_region = prev;
        if (tail == node) tail = prev;
        head.prev_region = tail;
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr,
                                        std::memory_order_relaxed);
#endif
        out.push_back(region);
        ++count;
        node = next;
    }
    if (count != 0) {
        item_count.fetch_sub(count, std::memory_order_relaxed);
    }
    lock.clear(std::memory_order_release);
    return count;
}

inline void RegionList::push_front_batch_dbg(RegionHead **nodes,
                                             size_t count,
                                             const char *tag) {
    if (simple_region_heat::local_resident_enabled()) {
        for (size_t i = 0; i < count; ++i) {
            RegionHead *node = nodes[i];
            if (node == nullptr) continue;
            const RegionPlacement placement = node->load_placement();
            if (placement == RegionPlacement::Resident ||
                placement == RegionPlacement::Streaming) {
                simple_region_heat::sync_supply_placement(
                    reinterpret_cast<uintptr_t>(node), node->bin,
                    placement == RegionPlacement::Resident);
            }
        }
    }
    if (list_only_six_children) {
        if(simple_region_heat::grouping_enabled()) {
            std::vector<RegionHead *> buckets[list_only_six::kChildCount];
            for (size_t i = 0; i < count; ++i) {
                const size_t c = simple_region_heat::allocation_class_for(
                    reinterpret_cast<uintptr_t>(nodes[i]));
                ASSERT(c < list_only_six::kChildCount);
                buckets[c].push_back(nodes[i]);
            }
            for (size_t c = 0; c < list_only_six::kChildCount; ++c)
                list_only_six_children[c].push_front_batch_dbg(
                    buckets[c].data(), buckets[c].size(), tag);
            return;
        }
        list_only_six_children[0].push_front_batch_dbg(nodes, count, tag);
        return;
    }
    if (count == 0) return;
    if (list_only_six_parent == nullptr) {
        for (size_t i = 0; i < count; ++i) {
            ASSERT(nodes[i] != nullptr);
            nodes[i]->ensure_six_group();
        }
    }
    acquire_region_list_lock(lock);
    if(simple_region_heat::grouping_enabled() && list_only_six_parent) {
        bool stale=false;
        for(size_t i=0;i<count;++i)
            stale |= list_only_six_parent->list_only_six_child(
                         simple_region_heat::allocation_class_for(
                             reinterpret_cast<uintptr_t>(nodes[i]))) != this;
        if(stale) {
            lock.clear(std::memory_order_release);
            list_only_six_parent->push_front_batch_dbg(nodes,count,tag);
            return;
        }
    }
    RegionListNodeBase *old_first = head.next_region;
    for (size_t i = 0; i < count; ++i) {
        RegionHead *node = nodes[i];
        ASSERT(node != nullptr);
#ifdef FARLIB_ALLOC_DEBUG
        ASSERT(node->next_region == nullptr);
        void *expected = nullptr;
        ASSERT(node->dbg_enqueued_list.compare_exchange_strong(
            expected, static_cast<void *>(this),
            std::memory_order_relaxed));
        node->dbg_last_push_tag = tag;
#endif
        node->prev_region = i == 0 ? &head : nodes[i - 1];
        node->next_region = i + 1 < count ? nodes[i + 1] : old_first;
        if (i != 0) nodes[i - 1]->next_region = node;
        set_owner_locked(node, true);
        acquire_region_placement_lock(node);
        ASSERT(!node->placement_list_owned.load(
            std::memory_order_relaxed));
        node->placement_list_owned.store(true,
                                         std::memory_order_release);
        node->unlock_placement();
    }
    mark_cursor_on_prepend_locked(nodes[count - 1]);
    evict_cursor_on_prepend_locked(nodes[count - 1]);
    gc_cursor_on_prepend_locked(nodes[count - 1]);
    head.next_region = nodes[0];
    old_first->prev_region = nodes[count - 1];
    head.prev_region = tail;
    if (tail == &head) {
        tail = nodes[count - 1];
        head.prev_region = tail;
    }
    item_count.fetch_add(count, std::memory_order_relaxed);
    lock.clear(std::memory_order_release);
}

inline RegionHead *RegionList::pop_unmatched(uint32_t time_stamp, int *reason) {
    if (list_only_six_children) {
        for (size_t i = 0; i < list_only_six::kChildCount; ++i) {
            if (auto *node = list_only_six_children[i].pop_unmatched(
                    time_stamp, reason))
                return node;
        }
        if (reason)
            *reason = nonempty() ? PopUnmatchedHeadMatched
                                 : PopUnmatchedEmpty;
        return nullptr;
    }
    acquire_region_list_lock(lock);
    RegionListNodeBase *node = head.next_region;
    if (node == &head) {
        if (reason) *reason = PopUnmatchedEmpty;
        lock.clear();
        return nullptr;
    }
    if (static_cast<RegionHead *>(node)->sweep_time_stamp == time_stamp) {
        if (reason) *reason = PopUnmatchedHeadMatched;
        lock.clear();
        return nullptr;
    } else {
        if (reason) *reason = PopUnmatchedSuccess;
        if (node != nullptr) {
            static_cast<RegionHead *>(node)->sweep_time_stamp = time_stamp;
        }
        auto *region = static_cast<RegionHead *>(node);
        acquire_region_placement_lock(region);
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(&head, node);
        evict_cursor_on_unlink_locked(&head, node);
        gc_cursor_on_unlink_locked(&head, node);
        head.next_region = node->next_region;
        node->next_region->prev_region = &head;
        if (tail == node) tail = &head;
        head.prev_region = tail;
        // Mark as unlinked so a later push can detect double-enqueue.
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        static_cast<RegionHead *>(node)->dbg_enqueued_list.store(
            nullptr, std::memory_order::relaxed);
#endif
        item_count.fetch_sub(1, std::memory_order_relaxed);
        lock.clear();
        return region;
    }
}

template <typename EligibleFn>
inline size_t RegionList::pop_matching_batch(uint32_t time_stamp,
                                             RegionHead **out_batch,
                                             size_t max_count, int *reason,
                                             EligibleFn &&eligible,
                                             MarkBreakdownStats *mark_diag,
                                             EvictCollectorDiag *evict_diag) {
    if (list_only_six_children) {
        size_t count = 0;
        for (size_t i = 0; i < list_only_six::kChildCount &&
                            count < max_count;
             ++i) {
            count += list_only_six_children[i].pop_matching_batch(
                time_stamp, out_batch + count, max_count - count, reason,
                std::forward<EligibleFn>(eligible), mark_diag, evict_diag);
        }
        if (reason)
            *reason = count ? PopUnmatchedSuccess
                            : (nonempty() ? PopUnmatchedHeadMatched
                                          : PopUnmatchedEmpty);
        return count;
    }
    const bool sample = mark_diag != nullptr && mark_diag->sample_pop();
    const bool collect_diag =
        evict_diag != nullptr && evict_diag->segment_enabled;
    const uint64_t before_lock =
        sample || collect_diag ? get_cycles() : 0;
    acquire_region_list_lock(lock);
    const uint64_t after_lock =
        sample || collect_diag ? get_cycles() : 0;
    if (mark_diag != nullptr) ++mark_diag->pop_calls;
    if (collect_diag) ++evict_diag->calls;
    size_t count = 0;
    bool saw_node = false;
    RegionListNodeBase *prev = &head;
    RegionListNodeBase *node = head.next_region;
    while (node != &head && count < max_count) {
        saw_node = true;
        if (mark_diag != nullptr) ++mark_diag->nodes_examined;
        if (collect_diag) ++evict_diag->nodes_examined;
        RegionHead *region = static_cast<RegionHead *>(node);
        RegionListNodeBase *next = node->next_region;
        acquire_region_placement_lock(region);
        if (region->sweep_time_stamp == time_stamp) {
            if (mark_diag != nullptr) ++mark_diag->stamped_skips;
            if (collect_diag) ++evict_diag->stamped_skips;
            region->unlock_placement();
            prev = node;
            node = next;
            continue;
        }
        if (!eligible(region)) {
            if (collect_diag) ++evict_diag->eligible_false;
            region->unlock_placement();
            prev = node;
            node = next;
            continue;
        }

        region->sweep_time_stamp = time_stamp;
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(prev, node);
        evict_cursor_on_unlink_locked(prev, node);
        gc_cursor_on_unlink_locked(prev, node);
        prev->next_region = next;
        next->prev_region = prev;
        if (tail == node) tail = prev;
        head.prev_region = tail;
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr, std::memory_order::relaxed);
#endif
        out_batch[count++] = region;
        node = next;
    }
    if (count > 0) {
        item_count.fetch_sub(count, std::memory_order_relaxed);
    } else if (reason) {
        *reason = saw_node ? PopUnmatchedHeadMatched : PopUnmatchedEmpty;
    }
    if (mark_diag != nullptr) mark_diag->regions_popped += count;
    if (collect_diag) evict_diag->selected += count;
    const uint64_t before_unlock =
        sample || collect_diag ? get_cycles() : 0;
    lock.clear();
    if (sample) {
        const uint64_t wait_cycles = after_lock - before_lock;
        const uint64_t hold_cycles = before_unlock - after_lock;
        ++mark_diag->pop_samples;
        mark_diag->list_lock_wait_cycles += wait_cycles;
        mark_diag->list_lock_hold_cycles += hold_cycles;
        if (wait_cycles > mark_diag->list_lock_wait_max_cycles) {
            mark_diag->list_lock_wait_max_cycles = wait_cycles;
        }
        if (hold_cycles > mark_diag->list_lock_hold_max_cycles) {
            mark_diag->list_lock_hold_max_cycles = hold_cycles;
        }
    }
    if (collect_diag) {
        const uint64_t wait_cycles = after_lock - before_lock;
        const uint64_t hold_cycles = before_unlock - after_lock;
        evict_diag->list_lock_wait_cycles += wait_cycles;
        evict_diag->list_lock_hold_cycles += hold_cycles;
        if (hold_cycles > evict_diag->list_lock_hold_max_cycles) {
            evict_diag->list_lock_hold_max_cycles = hold_cycles;
        }
    }
    return count;
}

template <typename EligibleFn>
inline size_t RegionList::pop_matching_batch_cursor(
    uint32_t time_stamp, RegionHead **out_batch, size_t max_count, int *reason,
    EligibleFn &&eligible, MarkBreakdownStats *mark_diag) {
    if (list_only_six_children) {
        size_t count = 0;
        for (size_t i = 0; i < list_only_six::kChildCount &&
                            count < max_count;
             ++i) {
            count += list_only_six_children[i].pop_matching_batch_cursor(
                time_stamp, out_batch + count, max_count - count, reason,
                std::forward<EligibleFn>(eligible), mark_diag);
        }
        if (reason)
            *reason = count ? PopUnmatchedSuccess
                            : (nonempty() ? PopUnmatchedHeadMatched
                                          : PopUnmatchedEmpty);
        return count;
    }
    const bool sample = mark_diag != nullptr && mark_diag->sample_pop();
    const uint64_t before_lock = sample ? get_cycles() : 0;
    acquire_region_list_lock(lock);
    const uint64_t after_lock = sample ? get_cycles() : 0;
    if (mark_diag != nullptr) ++mark_diag->pop_calls;
    auto unlock_and_record = [&] {
        const uint64_t before_unlock = sample ? get_cycles() : 0;
        lock.clear();
        if (!sample) return;
        const uint64_t wait_cycles = after_lock - before_lock;
        const uint64_t hold_cycles = before_unlock - after_lock;
        ++mark_diag->pop_samples;
        mark_diag->list_lock_wait_cycles += wait_cycles;
        mark_diag->list_lock_hold_cycles += hold_cycles;
        if (wait_cycles > mark_diag->list_lock_wait_max_cycles) {
            mark_diag->list_lock_wait_max_cycles = wait_cycles;
        }
        if (hold_cycles > mark_diag->list_lock_hold_max_cycles) {
            mark_diag->list_lock_hold_max_cycles = hold_cycles;
        }
    };

    // A zero-sized request is a no-op: in particular, it must not initialize
    // an empty epoch and thereby hide nodes inserted before the real caller.
    if (max_count == 0) {
        if (reason) *reason = PopUnmatchedEmpty;
        unlock_and_record();
        return 0;
    }

    if (!mark_scan_initialized || mark_scan_epoch != time_stamp) {
        mark_scan_epoch = time_stamp;
        mark_scan_initialized = true;
        mark_scan_done = tail == &head;
        mark_scan_prev = &head;
        mark_scan_last = tail;
    }
    if (mark_scan_done) {
        if (reason) *reason = PopUnmatchedEmpty;
        unlock_and_record();
        return 0;
    }

    size_t count = 0;
    bool saw_node = false;
    RegionListNodeBase *node = mark_scan_prev->next_region;
    while (node != &head && count < max_count) {
        saw_node = true;
        if (mark_diag != nullptr) ++mark_diag->nodes_examined;
        RegionHead *region = static_cast<RegionHead *>(node);
        RegionListNodeBase *next = node->next_region;
        const bool is_last = node == mark_scan_last;
        acquire_region_placement_lock(region);
        if (region->sweep_time_stamp == time_stamp) {
            if (mark_diag != nullptr) ++mark_diag->stamped_skips;
            region->unlock_placement();
            mark_scan_prev = node;
            node = next;
            if (is_last) {
                finish_mark_cursor_locked();
                break;
            }
            continue;
        }
        if (!eligible(region)) {
            region->unlock_placement();
            mark_scan_prev = node;
            node = next;
            if (is_last) {
                finish_mark_cursor_locked();
                break;
            }
            continue;
        }

        region->sweep_time_stamp = time_stamp;
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        RegionListNodeBase *prev = mark_scan_prev;
        mark_cursor_on_unlink_locked(prev, node);
        evict_cursor_on_unlink_locked(prev, node);
        gc_cursor_on_unlink_locked(prev, node);
        prev->next_region = next;
        next->prev_region = prev;
        if (tail == node) tail = prev;
        head.prev_region = tail;
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr, std::memory_order_relaxed);
#endif
        out_batch[count++] = region;
        if (is_last) {
            // The unlink hook finishes and clears the cursor references.
            break;
        }
        node = next;
    }

    // Reaching the sentinel before the captured boundary should only be
    // possible when concurrent list operations removed the remaining suffix.
    // Their unlink hooks already adjusted the boundary; finish defensively so
    // this epoch cannot restart from the head.
    if (node == &head && mark_cursor_active_locked()) {
        finish_mark_cursor_locked();
    }
    if (count > 0) {
        item_count.fetch_sub(count, std::memory_order_relaxed);
    } else if (reason) {
        *reason = saw_node ? PopUnmatchedHeadMatched : PopUnmatchedEmpty;
    }
    if (mark_diag != nullptr) mark_diag->regions_popped += count;
    unlock_and_record();
    return count;
}

template <typename EligibleFn>
inline size_t RegionList::pop_matching_batch_evict_cursor(
    uint32_t time_stamp, RegionHead **out_batch, size_t max_count,
    size_t visit_limit, int *reason, bool *has_unscanned,
    bool *visit_budget_exhausted, EligibleFn &&eligible,
    EvictCollectorDiag *evict_diag, uint64_t scan_generation,
    bool gc_cursor, size_t *visited_out) {
    if (list_only_six_children) {
        size_t count = 0;
        size_t visited = 0;
        bool more = false;
        bool exhausted = false;
        size_t child = 0;
        for (; child < list_only_six::kChildCount &&
                    count < max_count && visited < visit_limit;
             ++child) {
            bool child_more = false;
            bool child_exhausted = false;
            size_t child_visited = 0;
            count += list_only_six_children[child]
                         .pop_matching_batch_evict_cursor(
                             time_stamp, out_batch + count,
                             max_count - count, visit_limit - visited, reason,
                             &child_more, &child_exhausted,
                             std::forward<EligibleFn>(eligible), evict_diag,
                             scan_generation, gc_cursor, &child_visited);
            visited += child_visited;
            more = more || child_more;
            exhausted = exhausted || child_exhausted;
        }
        // A child not visited in this call has an independent cursor and is
        // deliberately left for the next call.
        more = more || (child < list_only_six::kChildCount &&
                        max_count != 0 && visit_limit != 0);
        exhausted = exhausted ||
                    (child < list_only_six::kChildCount &&
                     visited == visit_limit && visit_limit != 0);
        if (has_unscanned) *has_unscanned = more;
        if (visit_budget_exhausted) *visit_budget_exhausted = exhausted;
        if (visited_out) *visited_out = visited;
        if (reason)
            *reason = count ? PopUnmatchedSuccess
                            : (nonempty() ? PopUnmatchedHeadMatched
                                          : PopUnmatchedEmpty);
        return count;
    }
    const bool collect_diag =
        evict_diag != nullptr && evict_diag->segment_enabled;
    const uint64_t before_lock = collect_diag ? get_cycles() : 0;
    acquire_region_list_lock(lock);
    const uint64_t after_lock = collect_diag ? get_cycles() : 0;
    if (collect_diag) ++evict_diag->calls;
    auto unlock_and_record = [&] {
        const uint64_t before_unlock = collect_diag ? get_cycles() : 0;
        lock.clear();
        if (!collect_diag) return;
        const uint64_t wait_cycles = after_lock - before_lock;
        const uint64_t hold_cycles = before_unlock - after_lock;
        evict_diag->list_lock_wait_cycles += wait_cycles;
        evict_diag->list_lock_hold_cycles += hold_cycles;
        if (hold_cycles > evict_diag->list_lock_hold_max_cycles) {
            evict_diag->list_lock_hold_max_cycles = hold_cycles;
        }
    };

    if (has_unscanned != nullptr) *has_unscanned = false;
    if (visited_out != nullptr) *visited_out = 0;
    if (visit_budget_exhausted != nullptr) {
        *visit_budget_exhausted = false;
    }
    if (max_count == 0 || visit_limit == 0) {
        if (reason) *reason = PopUnmatchedEmpty;
        unlock_and_record();
        return 0;
    }

    if (scan_generation == 0) scan_generation = time_stamp;
    uint64_t &cursor_epoch = gc_cursor ? gc_scan_epoch : evict_scan_epoch;
    bool &cursor_initialized =
        gc_cursor ? gc_scan_initialized : evict_scan_initialized;
    bool &cursor_done = gc_cursor ? gc_scan_done : evict_scan_done;
    RegionListNodeBase *&cursor_prev =
        gc_cursor ? gc_scan_prev : evict_scan_prev;
    RegionListNodeBase *&cursor_last =
        gc_cursor ? gc_scan_last : evict_scan_last;
    auto cursor_active = [&] { return cursor_initialized && !cursor_done; };
    auto finish_cursor = [&] {
        if (gc_cursor) {
            finish_gc_cursor_locked();
        } else {
            finish_evict_cursor_locked();
        }
    };
    if (!cursor_initialized || cursor_epoch != scan_generation) {
        cursor_epoch = scan_generation;
        cursor_initialized = true;
        cursor_done = tail == &head;
        cursor_prev = &head;
        cursor_last = tail;
    }
    if (cursor_done) {
        if (reason) *reason = PopUnmatchedEmpty;
        unlock_and_record();
        return 0;
    }

    size_t count = 0;
    size_t visited = 0;
    bool saw_node = false;
    RegionListNodeBase *node = cursor_prev->next_region;
    while (node != &head && count < max_count && visited < visit_limit) {
        saw_node = true;
        ++visited;
        if (collect_diag) ++evict_diag->nodes_examined;
        RegionHead *region = static_cast<RegionHead *>(node);
        RegionListNodeBase *next = node->next_region;
        const bool is_last = node == cursor_last;
        acquire_region_placement_lock(region);
        if (region->sweep_time_stamp == time_stamp) {
            if (collect_diag) ++evict_diag->stamped_skips;
            region->unlock_placement();
            cursor_prev = node;
            node = next;
            if (is_last) {
                finish_cursor();
                break;
            }
            continue;
        }
        if (!eligible(region)) {
            if (collect_diag) ++evict_diag->eligible_false;
            region->unlock_placement();
            cursor_prev = node;
            node = next;
            if (is_last) {
                finish_cursor();
                break;
            }
            continue;
        }

        region->sweep_time_stamp = time_stamp;
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        RegionListNodeBase *prev = cursor_prev;
        mark_cursor_on_unlink_locked(prev, node);
        evict_cursor_on_unlink_locked(prev, node);
        gc_cursor_on_unlink_locked(prev, node);
        prev->next_region = next;
        next->prev_region = prev;
        if (tail == node) tail = prev;
        head.prev_region = tail;
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr, std::memory_order_relaxed);
#endif
        out_batch[count++] = region;
        if (collect_diag) ++evict_diag->selected;
        if (is_last) break;
        node = next;
    }

    if (node == &head && cursor_active()) {
        finish_cursor();
    }
    if (count > 0) {
        item_count.fetch_sub(count, std::memory_order_relaxed);
    }
    const bool more = cursor_active();
    const bool exhausted = more && visited == visit_limit;
    if (has_unscanned != nullptr) *has_unscanned = more;
    if (visit_budget_exhausted != nullptr) {
        *visit_budget_exhausted = exhausted;
    }
    if (visited_out != nullptr) *visited_out = visited;
    if (collect_diag) {
        if (more) ++evict_diag->cursor_more_returns;
        if (exhausted) ++evict_diag->cursor_visit_budget_exhaustions;
    }
    if (count == 0 && reason != nullptr) {
        *reason = saw_node ? PopUnmatchedHeadMatched : PopUnmatchedEmpty;
    }
    unlock_and_record();
    return count;
}

template <typename EligibleFn>
inline RegionHead *RegionList::pop_matching(uint32_t time_stamp, int *reason,
                                            EligibleFn &&eligible) {
    RegionHead *region = nullptr;
    RegionHead *batch[1];
    size_t popped = pop_matching_batch(time_stamp, batch, 1, reason, eligible);
    if (popped != 0) {
        region = batch[0];
    }
    return region;
}

inline size_t RegionList::pop_unmatched_batch(uint32_t time_stamp,
                                              RegionHead **out_batch,
                                              size_t max_count, int *reason,
                                              uint32_t max_mark_epoch,
                                              MarkBreakdownStats *mark_diag,
                                              EvictCollectorDiag *evict_diag) {
    if (list_only_six_children) {
        size_t count = 0;
        for (size_t i = 0; i < list_only_six::kChildCount &&
                            count < max_count;
             ++i) {
            count += list_only_six_children[i].pop_unmatched_batch(
                time_stamp, out_batch + count, max_count - count, reason,
                max_mark_epoch, mark_diag, evict_diag);
        }
        if (reason)
            *reason = count ? PopUnmatchedSuccess
                            : (nonempty() ? PopUnmatchedHeadMatched
                                          : PopUnmatchedEmpty);
        return count;
    }
    const bool sample = mark_diag != nullptr && mark_diag->sample_pop();
    const bool collect_diag =
        evict_diag != nullptr && evict_diag->segment_enabled;
    const uint64_t before_lock =
        sample || collect_diag ? get_cycles() : 0;
    acquire_region_list_lock(lock);
    const uint64_t after_lock =
        sample || collect_diag ? get_cycles() : 0;
    if (mark_diag != nullptr) ++mark_diag->pop_calls;
    if (collect_diag) ++evict_diag->calls;
    size_t count = 0;
    while (count < max_count) {
        RegionListNodeBase *node = head.next_region;
        if (node == &head) {
            if (count == 0 && reason) *reason = PopUnmatchedEmpty;
            break;
        }
        RegionHead *region = static_cast<RegionHead *>(node);
        if (mark_diag != nullptr) ++mark_diag->nodes_examined;
        if (collect_diag) ++evict_diag->nodes_examined;
        if (region->sweep_time_stamp == time_stamp) {
            if (mark_diag != nullptr) ++mark_diag->stamped_skips;
            if (collect_diag) ++evict_diag->stamped_skips;
            if (count == 0 && reason) *reason = PopUnmatchedHeadMatched;
            break;
        }
        if (max_mark_epoch != 0 &&
            region->last_mark_epoch.load(std::memory_order_relaxed) >
                max_mark_epoch) {
            if (mark_diag != nullptr) ++mark_diag->eligible_false;
            if (collect_diag) ++evict_diag->eligible_false;
            if (count == 0 && reason) *reason = PopUnmatchedHeadMatched;
            break;
        }

        // Pop it
        region->sweep_time_stamp = time_stamp;
        acquire_region_placement_lock(region);
        ASSERT(region->placement_list_owned.load(std::memory_order_relaxed));
        region->placement_list_owned.store(false, std::memory_order_release);
        region->unlock_placement();
        clear_owner_locked(region);
        mark_cursor_on_unlink_locked(&head, node);
        evict_cursor_on_unlink_locked(&head, node);
        gc_cursor_on_unlink_locked(&head, node);
        head.next_region = node->next_region;
        node->next_region->prev_region = &head;
        if (tail == node) tail = &head;
        head.prev_region = tail;
        // Mark as unlinked so a later push can detect double-enqueue.
        node->next_region = nullptr;
        node->prev_region = nullptr;
#ifdef FARLIB_ALLOC_DEBUG
        region->dbg_enqueued_list.store(nullptr, std::memory_order::relaxed);
#endif
        out_batch[count++] = region;
    }
    if (count > 0) {
        item_count.fetch_sub(count, std::memory_order_relaxed);
    }
    if (mark_diag != nullptr) mark_diag->regions_popped += count;
    if (collect_diag) evict_diag->selected += count;
    const uint64_t before_unlock =
        sample || collect_diag ? get_cycles() : 0;
    lock.clear();
    if (sample) {
        const uint64_t wait_cycles = after_lock - before_lock;
        const uint64_t hold_cycles = before_unlock - after_lock;
        ++mark_diag->pop_samples;
        mark_diag->list_lock_wait_cycles += wait_cycles;
        mark_diag->list_lock_hold_cycles += hold_cycles;
        mark_diag->list_lock_wait_max_cycles =
            std::max(mark_diag->list_lock_wait_max_cycles, wait_cycles);
        mark_diag->list_lock_hold_max_cycles =
            std::max(mark_diag->list_lock_hold_max_cycles, hold_cycles);
    }
    if (collect_diag) {
        const uint64_t wait_cycles = after_lock - before_lock;
        const uint64_t hold_cycles = before_unlock - after_lock;
        evict_diag->list_lock_wait_cycles += wait_cycles;
        evict_diag->list_lock_hold_cycles += hold_cycles;
        evict_diag->list_lock_hold_max_cycles =
            std::max(evict_diag->list_lock_hold_max_cycles, hold_cycles);
    }
    return count;
}

}  // namespace FarLib::allocator
