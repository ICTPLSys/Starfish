#pragma once

#include "cache/concurrent_cache.hpp"

namespace FarLib::cache {

namespace {

struct RecomputeRecipeSnapshot {
    recompute::InputState *inputs = nullptr;
    uint64_t arg = 0;
};

// Snapshot the binding while holding the existing entry move lock, then keep
// the InputState alive across the user callback. The callback is deliberately
// passed only immutable context and an argument; it must write the supplied
// destination and must not touch cache state.
inline bool snapshot_recompute_recipe(FarObjectEntry *entry,
                                      RecomputeRecipeSnapshot *snapshot) {
    if (entry == nullptr || snapshot == nullptr) return false;
    auto current = entry->load_state(std::memory_order_acquire);
    for (;;) {
        if (current.state != FETCHING) return false;
        if (current.invalid) {
            // Another publisher owns the move lock. Wait for its state
            // publication instead of returning with invalid=1 still held.
            uthread::yield();
            current = entry->load_state(std::memory_order_acquire);
            continue;
        }
        auto locked = current;
        locked.invalid = 1;
        if (!entry->cas_state_weak(current, locked)) continue;

        const auto binding = entry->recompute_binding();
        if (binding.inputs == nullptr) {
            auto unlocked = locked;
            unlocked.invalid = 0;
            (void)entry->cas_state_strong(locked, unlocked);
            return false;
        }
        binding.inputs->retain();
        snapshot->inputs = binding.inputs;
        snapshot->arg = binding.arg;

        auto unlocked = locked;
        unlocked.invalid = 0;
        if (!entry->cas_state_strong(locked, unlocked)) {
            snapshot->inputs->release();
            snapshot->inputs = nullptr;
            // The move lock normally makes this CAS unconditional, but a
            // concurrent publisher may have already converged the fetch. Do
            // not strand invalid=1: re-read state and retry under new owner.
            current = entry->load_state(std::memory_order_acquire);
            continue;
        }
        return true;
    }
}

inline void release_recompute_snapshot(RecomputeRecipeSnapshot &snapshot) {
    if (snapshot.inputs != nullptr) snapshot.inputs->release();
    snapshot = {};
}

inline FarObjectEntry *current_recipe_entry(
    ::FarLib::allocator::BlockHead *block) {
    if (block == nullptr) return nullptr;
    const auto obj = block->obj_meta_data.load(std::memory_order_acquire);
    return obj.is_null() ? nullptr : &ConcurrentArrayCache::get_entry_of(obj);
}

}  // namespace

inline bool ConcurrentArrayCache::recompute_recipe_assist_wait(
    FarObjectEntry *entry, size_t qp_idx, size_t client_idx) {
    (void)qp_idx;
    if (entry == nullptr || entry->is_local() ||
        entry->load_state(std::memory_order_acquire).state != FETCHING ||
        !entry->has_recompute_recipe()) {
        return false;
    }

    void *local_addr = entry->local_addr();
    if (local_addr == nullptr) return false;
    auto *block = static_cast<::FarLib::allocator::BlockHead *>(local_addr) - 1;

    // The failed WC or known-dead prepost path closes this gate. If it is not
    // closed, the normal READ path still owns the fetch epoch.
    if (!::FarLib::allocator::normal_read_recovery_active(block)) {
        return false;
    }

    // Pin before claiming the block owner. A fetch can be concurrently moved
    // or deallocated between the initial check and this assist.
    if (!pin_fetch_entry_for_recovery(entry)) return true;
    FarObjectEntry *pinned_entry = entry;
    const auto current_obj = block->obj_meta_data.load(
        std::memory_order_acquire);
    if (current_obj.is_null()) {
        unpin_fetch_entry_for_recovery(pinned_entry);
        return true;
    }
    pinned_entry = &get_entry_of(current_obj);
    const auto pinned_state =
        pinned_entry->load_state(std::memory_order_acquire);
    if (pinned_state.state != FETCHING || pinned_state.invalid ||
        pinned_entry->local_addr() != block->get_object_ptr() ||
        !pinned_entry->has_recompute_recipe()) {
        unpin_fetch_entry_for_recovery(pinned_entry);
        return true;
    }
    entry = pinned_entry;

    auto *client = rdma::get_client(client_idx);
    if (client == nullptr) {
        unpin_fetch_entry_for_recovery(entry);
        return false;
    }

    // Never rebuild over bytes that a prior ordinary READ may still write.
    if (block->pending_rdma_reads.load(std::memory_order_acquire) != 0) {
        if (!ec_recovery_drain_normal_reads_once(block)) {
            unpin_fetch_entry_for_recovery(entry);
            return true;
        }
    }

    // Reuse the allocator's single owner bit: bit 0 is ordinary-READ gate,
    // bit 1 is the one-winner recovery owner.
    if (!::FarLib::allocator::try_claim_degraded_read(block)) {
        unpin_fetch_entry_for_recovery(entry);
        uthread::yield();
        return true;
    }

    RecomputeRecipeSnapshot snapshot;
    if (!snapshot_recompute_recipe(entry, &snapshot)) {
        release_degraded_read_owner(block);
        FarObjectEntry *owner = current_recipe_entry(block);
        unpin_fetch_entry_for_recovery(owner != nullptr ? owner : entry);
        return true;
    }

    const auto stable_obj = block->obj_meta_data.load(std::memory_order_acquire);
    if (stable_obj.is_null()) {
        release_recompute_snapshot(snapshot);
        release_degraded_read_owner(block);
        unpin_fetch_entry_for_recovery(entry);
        return true;
    }
    const size_t bytes = stable_obj.size;
    bool rebuilt = false;
    try {
        rebuilt = snapshot.inputs->invoke(block->get_object_ptr(), bytes,
                                          snapshot.arg);
    } catch (...) {
        rebuilt = false;
    }
    release_recompute_snapshot(snapshot);

    if (!rebuilt) {
        recomputable_failures_.fetch_add(1, std::memory_order_relaxed);
        release_degraded_read_owner(block);
        FarObjectEntry *owner = current_recipe_entry(block);
        unpin_fetch_entry_for_recovery(owner != nullptr ? owner : entry);
        ERROR("recompute recipe callback failed");
    }

    ibv_wc synthetic{};
    synthetic.status = IBV_WC_SUCCESS;
    synthetic.opcode = IBV_WC_RDMA_READ;
    synthetic.wr_id = reinterpret_cast<uint64_t>(block->get_object_ptr());
    synthetic.byte_len = static_cast<uint32_t>(bytes);
    handle_rdma_read_complete(synthetic, false, true);

    FarObjectEntry *owner = current_recipe_entry(block);
    const bool published =
        owner != nullptr &&
        owner->load_state(std::memory_order_acquire).state == LOCAL &&
        owner->local_addr() == block->get_object_ptr();
    if (!published) {
        recomputable_failures_.fetch_add(1, std::memory_order_relaxed);
        release_degraded_read_owner(block);
        if (owner != nullptr) unpin_fetch_entry_for_recovery(owner);
        else unpin_fetch_entry_for_recovery(entry);
        ERROR("recompute recipe publication failed");
    }

    recomputable_restored_.fetch_add(1, std::memory_order_relaxed);
    release_degraded_read_owner(block);
    unpin_fetch_entry_for_recovery(owner != nullptr ? owner : entry);
    return true;
}

}  // namespace FarLib::cache
