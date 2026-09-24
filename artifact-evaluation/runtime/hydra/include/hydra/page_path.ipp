#pragma once

namespace FarLib::cache {

inline std::pair<far_obj_t, void *> ConcurrentArrayCache::hydra_allocate(
    FarObjectEntry *handle, size_t bytes, bool dirty, DereferenceScope &scope) {
    const auto bin = hydra::size_class(bytes);
    if (bin == hydra::kSizeClasses) ERROR("hydra: invalid packed object size");
    auto &lane = hydra_lanes_[rdma::thread_info.thread_id % hydra::kAllocationLanes];
    hydra::Page *page = nullptr;
    uint16_t offset = 0;
    bool release_lease = false;
    {
        // No yielding or RDMA while holding the slot-allocation mutex.
        std::lock_guard<std::mutex> lock(lane.mutex);
        page = lane.active[bin];
        if (page != nullptr) {
            offset = static_cast<uint16_t>(page->next_slot * page->slot_bytes);
            ++page->next_slot;
            page->references.fetch_add(1, std::memory_order_relaxed);
            if (page->next_slot == hydra::kPageBytes / page->slot_bytes) {
                lane.active[bin] = nullptr;
                release_lease = page->allocation_lease.exchange(false);
            }
        }
    }
    if (page == nullptr) {
        auto candidate = std::make_unique<hydra::Page>();
        candidate->slot_bytes = static_cast<uint16_t>(hydra::slot_bytes(bytes));
        candidate->next_slot = 1;
        candidate->references.store(2); // current handle + allocator lease
        page = candidate.get();
        auto allocation = allocate_runtime_object<true>(
            &page->owner, hydra::kPageBytes, true, scope, 0);
        page->owner.hydra_mark_page();
        std::memset(allocation.second, 0, hydra::kPageBytes);
        try {
            std::lock_guard<std::mutex> lock(hydra_pages_mutex_);
            hydra_pages_.push_back(std::move(candidate));
        } catch (...) {
            deallocate<false, false>(page->owner, hydra::kPageBytes);
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(lane.mutex);
            if (lane.active[bin] == nullptr && page->slot_bytes < hydra::kPageBytes) {
                lane.active[bin] = page;
            } else {
                // Another fibre filled the lane while allocation yielded.
                // This candidate remains valid for its one live handle.
                release_lease = page->allocation_lease.exchange(false);
            }
        }
        hydra_pages_created_.fetch_add(1, std::memory_order_relaxed);
    }
    if (release_lease) hydra_release_page_reference(page);
    // Reserve the slot/handle before fetching: other fibres may run while we wait.
    const far_obj_t page_obj{.size = hydra::kPageBytes,
        .obj_id = reinterpret_cast<uint64_t>(&page->owner)};
    ON_MISS_BEGIN
    ON_MISS_END
    void *base = fetch_lite<true>(page_obj, __on_miss__, scope);
    handle->hydra_attach(&page->owner, offset, static_cast<uint16_t>(bytes));
    if (dirty) page->owner.mark_dirty();
    hydra_objects_created_.fetch_add(1, std::memory_order_relaxed);
    return {{.size = bytes, .obj_id = reinterpret_cast<uint64_t>(handle)},
            static_cast<void *>(static_cast<char *>(base) + offset)};
}

inline void ConcurrentArrayCache::hydra_release_page_reference(hydra::Page *page) {
    const auto before = page->references.fetch_sub(1, std::memory_order_acq_rel);
    if (before == 0) ERROR("hydra: page reference underflow");
    if (before == 1) {
        deallocate<false, false>(page->owner, hydra::kPageBytes);
        hydra_pages_released_.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void ConcurrentArrayCache::hydra_release_handle(FarObjectEntry &handle) {
    auto *page = reinterpret_cast<hydra::Page *>(handle.hydra_owner());
    handle.set_free();
    hydra_objects_released_.fetch_add(1, std::memory_order_relaxed);
    hydra_release_page_reference(page);
}

inline void ConcurrentArrayCache::hydra_shutdown_pages() {
    if (hydra_pages_.empty()) return;
    // Called after mutators and evacuation producers have stopped. Graph bulk
    // teardown may deliberately abandon handles; the arena owns final cleanup.
    for (auto &lane : hydra_lanes_) lane.active.fill(nullptr);
    for (auto &page : hydra_pages_) {
        if (page->owner.load_state().state != FREE) {
            deallocate<false, false>(page->owner, hydra::kPageBytes);
            hydra_pages_released_.fetch_add(1, std::memory_order_relaxed);
        }
        page->references.store(0);
        page->allocation_lease.store(false);
    }
    std::cout << "hydra.pages created=" << hydra_pages_created_.load()
              << " released=" << hydra_pages_released_.load()
              << " objects_created=" << hydra_objects_created_.load()
              << " objects_released=" << hydra_objects_released_.load()
              << " page_bytes=" << hydra::kPageBytes
              << " data_shard_bytes=" << hydra::kShardBytes << '\n';
}
}  // namespace FarLib::cache
