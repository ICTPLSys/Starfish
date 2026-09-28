#pragma once

namespace FarLib::cache {

inline std::pair<far_obj_t, void *> ConcurrentArrayCache::hydra_allocate(
    FarObjectEntry *handle, size_t bytes, bool dirty, DereferenceScope &scope,
    uint8_t packing_class) {
    if (packing_class > 1) ERROR("hydra: invalid packing class");
    const auto bin = hydra::size_class(bytes);
    if (bin == hydra::kSizeClasses) ERROR("hydra: invalid packed object size");
    auto &lane = hydra_lanes_[rdma::thread_info.thread_id % hydra::kAllocationLanes];
    hydra::Page *page = nullptr;
    uint16_t offset = 0;
    bool release_lease = false;
    {
        // No yielding or RDMA while holding the slot-allocation mutex.
        std::lock_guard<std::mutex> lock(lane.mutex);
        page = lane.active[bin][packing_class];
        if (page != nullptr) {
            if (page->packing_class != packing_class) ERROR("hydra: mixed packing class");
            offset = static_cast<uint16_t>(page->next_slot * page->slot_bytes);
            ++page->next_slot;
            page->references.fetch_add(1, std::memory_order_relaxed);
            if (page->next_slot == hydra::kPageBytes / page->slot_bytes) {
                lane.active[bin][packing_class] = nullptr;
                release_lease = page->allocation_lease.exchange(false);
            }
        }
    }
    if (page == nullptr) {
        auto candidate = std::make_unique<hydra::Page>();
        candidate->slot_bytes = static_cast<uint16_t>(hydra::slot_bytes(bytes));
        candidate->packing_class = packing_class;
        candidate->next_slot = 1;
        if (hydra_runtime_packing_ && hydra_runtime_packing_->phase.load() == hydra::RuntimePacking::Learning)
            candidate->heat = std::make_unique<hydra::SlotHeat[]>(hydra::kPageBytes / candidate->slot_bytes);
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
            if (lane.active[bin][packing_class] == nullptr && page->slot_bytes < hydra::kPageBytes) {
                lane.active[bin][packing_class] = page;
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
    hydra_rebind_slot(*handle);
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
    if (page->heat) page->heat[handle.hydra_offset() / page->slot_bytes].handle.store(nullptr, std::memory_order_release);
    handle.set_free();
    hydra_objects_released_.fetch_add(1, std::memory_order_relaxed);
    hydra_release_page_reference(page);
}

inline void ConcurrentArrayCache::hydra_report_packing() {
    std::lock_guard<std::mutex> lock(hydra_pages_mutex_);
    uint64_t pages[2]{}, slots[2]{};
    for (const auto &page : hydra_pages_) {
        if (page->packing_class > 1) ERROR("hydra: invalid page packing class");
        ++pages[page->packing_class];
        slots[page->packing_class] += page->next_slot;
    }
    for (size_t c = 0; c < 2; ++c)
        std::cout << "hydra.packing class=" << c << " pages=" << pages[c]
                  << " slots=" << slots[c] << " page_bytes=" << hydra::kPageBytes << '\n';
}

inline void ConcurrentArrayCache::hydra_shutdown_pages() {
    if (hydra_pages_.empty()) return;
    // Called after mutators and evacuation producers have stopped. Graph bulk
    // teardown may deliberately abandon handles; the arena owns final cleanup.
    for (auto &lane : hydra_lanes_)
        for (auto &classes : lane.active) classes.fill(nullptr);
    for (auto &page : hydra_pages_) {
        if (page->owner.load_state().state != FREE) {
            deallocate<false, false>(page->owner, hydra::kPageBytes);
            hydra_pages_released_.fetch_add(1, std::memory_order_relaxed);
        }
        page->references.store(0);
        page->allocation_lease.store(false);
    }
    const bool carbink = ::FarLib::get_config().is_carbink_mode();
    std::cout << (carbink ? "carbink.pages created=" : "hydra.pages created=")
              << hydra_pages_created_.load()
              << " released=" << hydra_pages_released_.load()
              << " objects_created=" << hydra_objects_created_.load()
              << " objects_released=" << hydra_objects_released_.load()
              << " page_bytes=" << hydra::kPageBytes
              << " data_shard_bytes="
              << (carbink ? hydra::kPageBytes : hydra::kShardBytes) << '\n';
    if (hydra_runtime_packing_) {
        const auto &p = *hydra_runtime_packing_;
        std::cout << "hydra.runtime_packing_end phase=" << int(p.phase.load())
                  << " roots=" << p.active_roots()
                  << " promoted_objects=" << p.promoted_objects
                  << " promoted_bytes=" << p.promoted_bytes << '\n';
    }
}
}  // namespace FarLib::cache
