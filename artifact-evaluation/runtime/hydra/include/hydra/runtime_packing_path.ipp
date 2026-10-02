#pragma once

namespace FarLib::cache {
inline void ConcurrentArrayCache::hydra_rebind_slot(FarObjectEntry &handle) {
    if (!hydra_runtime_packing_ || !handle.hydra_packed()) return;
    auto *page = reinterpret_cast<hydra::Page *>(handle.hydra_owner());
    if (page->heat)
        page->heat[handle.hydra_offset() / page->slot_bytes].handle.store(&handle, std::memory_order_release);
}

inline void ConcurrentArrayCache::hydra_record_demand(const FarObjectEntry &handle) {
    auto *p = hydra_runtime_packing_.get();
    if (!p || p->phase.load(std::memory_order_relaxed) != hydra::RuntimePacking::Learning
        || !profile::is_working() || !handle.hydra_packed()) return;
    const auto *page = reinterpret_cast<const hydra::Page *>(handle.hydra_owner());
    if (!page->heat) return;
    auto &hits = page->heat[handle.hydra_offset() / page->slot_bytes].references;
    hits.fetch_add(1, std::memory_order_relaxed); // learning stops long before uint32 overflow
    auto &s = p->shards[hydra::RuntimePacking::native_shard()];
    if ((++s.references & 1023) == 0)
        p->published_references.fetch_add(1024, std::memory_order_relaxed);
}

inline int ConcurrentArrayCache::hydra_packing_root_enter() {
    auto *p = hydra_runtime_packing_.get();
    if (!p) return -1;
    const uintptr_t self = reinterpret_cast<uintptr_t>(fibre_self());
    for (;;) {
        auto phase = p->phase.load(std::memory_order_seq_cst);
        if (phase == hydra::RuntimePacking::Done) return -1;
        if (phase == hydra::RuntimePacking::Closing) {
            if (p->maintenance_fibre.load(std::memory_order_acquire) == self) return -1;
            uthread::yield(); continue;
        }
        if (p->published_references.load(std::memory_order_relaxed) >= hydra::RuntimePacking::LearnReferences) {
            if (p->phase.compare_exchange_strong(phase, hydra::RuntimePacking::Closing, std::memory_order_seq_cst)) {
                p->maintenance_fibre.store(self, std::memory_order_release);
                const auto start = std::chrono::steady_clock::now();
                while (p->active_roots() != 0) {
                    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(30)) {
                        std::cerr << "hydra.runtime_packing abort=drain_timeout roots=" << p->active_roots() << '\n';
                        p->phase.store(hydra::RuntimePacking::Done, std::memory_order_seq_cst);
                        return -1;
                    }
                    uthread::yield();
                }
                hydra_repack_learned();
                std::cout << "hydra.runtime_packing_pause seconds="
                          << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
                          << " includes_drain=1 includes_repack=1\n" << std::flush;
                p->phase.store(hydra::RuntimePacking::Done, std::memory_order_seq_cst);
                if (::FarLib::get_config().ft_background_rebuild) {
                    std::ostringstream out;
                    out << "hydra.runtime_packing_ready monotonic_ns="
                        << ec_recovery_profile_now_ns();
                    std::cout << (out.str() + "\n") << std::flush;
                }
                return -1;
            }
            continue;
        }
        const size_t shard = hydra::RuntimePacking::native_shard();
        p->shards[shard].roots.fetch_add(1, std::memory_order_seq_cst);
        if (p->phase.load(std::memory_order_seq_cst) != hydra::RuntimePacking::Closing)
            return static_cast<int>(shard);
        p->shards[shard].roots.fetch_sub(1, std::memory_order_seq_cst);
    }
}

inline void ConcurrentArrayCache::hydra_packing_root_exit(int token) {
    auto &counter = hydra_runtime_packing_->shards[static_cast<size_t>(token)].roots;
    if (counter.fetch_sub(1, std::memory_order_seq_cst) == 0) ERROR("hydra packing root underflow");
}

inline void ConcurrentArrayCache::hydra_repack_learned() {
    auto &p = *hydra_runtime_packing_;
    struct Candidate { hydra::Page *page; uint16_t slot; uint32_t hits; };
    const auto begin = std::chrono::steady_clock::now();
    std::vector<Candidate> candidates;
    uint64_t references = 0;
    for (const auto &s : p.shards) references += s.references;
    // Root admission is closed and drained: no app allocation/relocation/free.
    // GC operates on owner entries, never these per-slot handles.
    for (const auto &owned : hydra_pages_) {
        auto *page = owned.get();
        if (!page->heat || page->packing_class != 0) continue;
        p.slot_metadata_bytes += sizeof(hydra::SlotHeat) * hydra::capacity(page->slot_bytes);
        for (uint16_t slot = 0; slot < page->next_slot; ++slot) {
            const auto hits = page->heat[slot].references.load(std::memory_order_relaxed);
            if (hits) ++p.observed_objects;
            if (hits >= 2 && page->heat[slot].handle.load(std::memory_order_acquire))
                candidates.push_back({page,slot,hits});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
        if (a.hits != b.hits) return a.hits > b.hits;
        if (a.page != b.page) return std::less<hydra::Page *>{}(a.page,b.page);
        return a.slot < b.slot;
    });
    const uint64_t budget = p.hot_byte_budget;
    std::cout << "hydra.runtime_packing_begin refs=" << references
              << " observed=" << p.observed_objects << " candidates=" << candidates.size()
              << " budget_bytes=" << budget << " slot_metadata_bytes=" << p.slot_metadata_bytes
              << " roots=" << p.active_roots() << '\n' << std::flush;
    std::array<std::byte, hydra::kPageBytes> copy;
    for (const auto &c : candidates) {
        if (p.promoted_bytes + c.page->slot_bytes > budget) continue;
        auto *handle = c.page->heat[c.slot].handle.load(std::memory_order_acquire);
        if (!handle || !handle->hydra_packed() || handle->hydra_owner() != &c.page->owner
            || handle->hydra_offset() != c.slot * c.page->slot_bytes) ERROR("hydra packing stale slot registry");
        const uint16_t bytes = handle->object_size();
        FarObjectEntry replacement;
        {
            RootDereferenceScope scope;
            ON_MISS_BEGIN
            ON_MISS_END
            const far_obj_t obj{.size=bytes,.obj_id=reinterpret_cast<uint64_t>(handle)};
            void *source = fetch_with_miss_handler(obj,__on_miss__,scope);
            auto *owner = handle->hydra_owner();
            std::memcpy(copy.data(),source,bytes);
            owner->unpin();
        }
        {
            RootDereferenceScope scope;
            auto dest = hydra_allocate(&replacement,bytes,true,scope,1);
            ON_MISS_BEGIN
            ON_MISS_END
            void *pinned_dest = fetch_with_miss_handler(dest.first,__on_miss__,scope);
            auto *owner = replacement.hydra_owner();
            owner->mark_dirty();
            std::memcpy(pinned_dest,copy.data(),bytes);
            owner->unpin();
        }
        // No raw pointers survive either scope. Page owner can move/evict;
        // descriptors and handles remain stable behind the admission gate.
        hydra_release_handle(*handle);
        handle->hydra_take_handle(replacement);
        hydra_rebind_slot(*handle);
        ++p.promoted_objects; p.promoted_bytes += c.page->slot_bytes;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout << "hydra.runtime_packing_done promoted_objects=" << p.promoted_objects
              << " promoted_bytes=" << p.promoted_bytes << " seconds=" << seconds
              << " roots=" << p.active_roots() << " oracle=0\n" << std::flush;
    hydra_report_packing();
    // One-shot learning is finished. No root may access these arrays until
    // Done is published; GC never reads them. Future allocations omit them.
    for (auto &page : hydra_pages_) page->heat.reset();
    std::cout << "hydra.runtime_packing_metadata_released bytes=" << p.slot_metadata_bytes << '\n';
}
} // namespace FarLib::cache
