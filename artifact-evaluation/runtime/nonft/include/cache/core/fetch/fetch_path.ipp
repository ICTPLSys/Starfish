#pragma once
#include "cache/concurrent_cache.hpp"
#include "utils/scope_diag.hpp"

namespace FarLib::cache {

template <bool IncreaseRefCount>
inline bool ConcurrentArrayCache::post_fetch(far_obj_t obj,
                                             DereferenceScope &scope,
                                             bool sync_batch_eligible) {
    auto &entry = get_entry_of(obj);
retry:
    auto old_state = entry.load_state();
    if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
        auto spin_start = get_cycles();
        profile::count_excl_move_lock_spin();
        profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
        goto retry;
    }
    auto new_state = old_state;
    if (IncreaseRefCount) {
        new_state.inc_ref_cnt();
    }
    new_state.inc_hotness();
    switch (new_state.state) {
    case LOCAL:
    case PINNED:
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        record_non_fast_path_access(entry, profile::ReferenceKind::Read);
        return true;
    case FETCHING:
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        record_non_fast_path_access(entry, profile::ReferenceKind::Read);
        return false;
    case MARKED:
    case EVICTING:
        if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
            auto spin_start = get_cycles();
            profile::count_excl_move_lock_spin();
            profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
            goto retry;
        }
        new_state.state = LOCAL;
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        record_non_fast_path_access(entry, profile::ReferenceKind::Read);
        return true;
    case REMOTE: {
        auto placement =
            prepare_remote_fetch_placement(entry, obj.size, scope, true);
        signal::disable_signal();
        void *local_ptr = this->allocate_local(
            obj.size, obj, scope, placement.requested_placement,
            placement.allocation_group_id);
        new_state.state = FETCHING;
        if (!entry.cas_state_weak(old_state, new_state)) {
            deallocate_local(local_ptr, entry);
            if (placement.backup_reserved) {
                release_remote_backup_budget(obj.size);
            }
            signal::enable_signal();
            goto retry;
        }
        assert(local_ptr);
        entry.set_local_addr(local_ptr);
        const auto actual_fetch_placement =
            finalize_remote_fetch_placement(entry, obj.size, local_ptr,
                                            placement);
        auto client_idx = rdma::thread_info.thread_id;
        entry.set_client_idx(client_idx);
        entry.set_remote_backup_reservation(placement.backup_reserved);
        assert(entry.local_addr());
        record_backup_group_fetch(obj.size);
        record_logical_object_fetch(entry, obj.size,
                                    actual_fetch_placement);
        record_profiled_backup_decision(entry, obj.size,
                                        placement.backup_reserved);
        auto *diag_fibre = fibre_self();
        scope_diag::set_pending(
            diag_fibre, reinterpret_cast<uintptr_t>(&entry));
        {
            scope_diag::Guard diag_guard(diag_fibre, scope_diag::RDMA_POST);
            this->post_read_request_with_idx(obj, client_idx,
                                             sync_batch_eligible);
            scope_diag::posted(diag_fibre);
        }
        signal::enable_signal();
        record_non_fast_path_access(entry, profile::ReferenceKind::Read);
        return false;
    }
    case BUSY:
        goto retry;
    case FREE:
        return true;
    default:
        ERROR("invalid entry state when fetching");
    }
}

template <bool Mut>
inline bool ConcurrentArrayCache::fetch_lite_fast_path(EntryStateBits state) {
    return state.is_deref_fast_path<Mut>();
}

template <bool Mut, bool Profile>
inline bool ConcurrentArrayCache::post_fetch_lite(far_obj_t obj,
                                                  DereferenceScope &scope,
                                                  bool sync_batch_eligible) {
    auto &entry = get_entry_of(obj);
    auto old_state = entry.load_state(std::memory_order::relaxed);
    if (fetch_lite_fast_path<Mut>(old_state)) [[likely]] {
        assert(entry.local_addr() != nullptr);
        if constexpr (Profile) {
            record_local_fast_path_access(entry, Mut ? profile::ReferenceKind::Write
                                                    : profile::ReferenceKind::Read);
        } else {
            record_local_fast_path_reference(entry,
                                             Mut ? profile::ReferenceKind::Write
                                                 : profile::ReferenceKind::Read);
        }
        return true;
    }
    return post_fetch_lite_slow_path<Mut, Profile>(
        entry, obj, scope, sync_batch_eligible);
}

template <bool Mut, bool Profile>
inline bool ConcurrentArrayCache::post_fetch_lite_slow_path(
    FarObjectEntry &entry, far_obj_t obj, DereferenceScope &scope,
    bool sync_batch_eligible) {
    auto old_state = entry.load_state(std::memory_order::relaxed);
retry:
    if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
        auto spin_start = get_cycles();
        profile::count_excl_move_lock_spin();
        old_state = entry.load_state(std::memory_order::relaxed);
        profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
        goto retry;
    }
    if constexpr (Mut) {
        if (invalidate_retained_backup_for_write(entry, obj.size)) {
            old_state = entry.load_state(std::memory_order::relaxed);
        }
    }
    auto new_state = old_state;
    if constexpr (Mut) {
        new_state.dirty = true;
    }
    new_state.inc_hotness();
    switch (new_state.state) {
    case LOCAL:
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        [[fallthrough]];
    case PINNED:
        if constexpr (Profile) {
            record_non_fast_path_access(entry,
                                        Mut ? profile::ReferenceKind::Write
                                            : profile::ReferenceKind::Read);
        } else {
            record_non_fast_path_reference(entry,
                                           Mut ? profile::ReferenceKind::Write
                                               : profile::ReferenceKind::Read);
        }
        return true;
    case FETCHING:
        if (!entry.cas_state_weak(old_state, new_state)) {
            goto retry;
        }
        if constexpr (Profile) {
            record_non_fast_path_access(entry,
                                        Mut ? profile::ReferenceKind::Write
                                            : profile::ReferenceKind::Read);
        } else {
            record_non_fast_path_reference(entry,
                                           Mut ? profile::ReferenceKind::Write
                                               : profile::ReferenceKind::Read);
        }
        return false;
    case MARKED:
    case EVICTING:
        if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
            auto spin_start = get_cycles();
            profile::count_excl_move_lock_spin();
            old_state = entry.load_state(std::memory_order::relaxed);
            profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
            goto retry;
        }
        new_state.state = LOCAL;
        if (!entry.cas_state_weak(old_state, new_state)) {
            goto retry;
        }
        assert(entry.local_addr());
        if constexpr (Profile) {
            record_non_fast_path_access(entry,
                                        Mut ? profile::ReferenceKind::Write
                                            : profile::ReferenceKind::Read);
        } else {
            record_non_fast_path_reference(entry,
                                           Mut ? profile::ReferenceKind::Write
                                               : profile::ReferenceKind::Read);
        }
        return true;
    case REMOTE: {
        auto placement = prepare_remote_fetch_placement(
            entry, obj.size, scope, !Mut);
        signal::disable_signal();
        void *local_ptr = this->allocate_local(
            obj.size, obj, scope, placement.requested_placement,
            placement.allocation_group_id);
        new_state.state = FETCHING;
        if (!entry.cas_state_weak(old_state, new_state)) {
            deallocate_local(local_ptr, entry);
            if (placement.backup_reserved) {
                release_remote_backup_budget(obj.size);
            }
            signal::enable_signal();
            goto retry;
        }
        assert(local_ptr != nullptr);
        entry.set_local_addr(local_ptr);
        const auto actual_fetch_placement =
            finalize_remote_fetch_placement(entry, obj.size, local_ptr,
                                            placement);
        assert(entry.local_addr());
        auto client_idx = rdma::thread_info.thread_id;
        entry.set_client_idx(client_idx);
        entry.set_remote_backup_reservation(placement.backup_reserved);
        record_backup_group_fetch(obj.size);
        record_logical_object_fetch(entry, obj.size,
                                    actual_fetch_placement);
        if constexpr (!Mut) {
            record_profiled_backup_decision(entry, obj.size,
                                             placement.backup_reserved);
        }
        auto *diag_fibre = fibre_self();
        scope_diag::set_pending(
            diag_fibre, reinterpret_cast<uintptr_t>(&entry));
        {
            scope_diag::Guard diag_guard(diag_fibre, scope_diag::RDMA_POST);
            this->post_read_request_with_idx(obj, client_idx,
                                             sync_batch_eligible);
            scope_diag::posted(diag_fibre);
        }
        signal::enable_signal();
        if constexpr (Profile) {
            record_non_fast_path_access(entry,
                                        Mut ? profile::ReferenceKind::Write
                                            : profile::ReferenceKind::Read);
        } else {
            record_non_fast_path_reference(entry,
                                           Mut ? profile::ReferenceKind::Write
                                               : profile::ReferenceKind::Read);
        }
        return false;
    }
    case BUSY:
        goto retry;
    case FREE:
        return true;
    default:
        ERROR("invalid entry state when fetching");
    }
}

inline bool ConcurrentArrayCache::check_fetch(FarObjectEntry *entry,
                                              fetch_ddl_t &ddl) {
    fetch_ddl_t now = __rdtsc();
    auto client_idx = entry->get_client_idx();
    auto *client = rdma::get_client(client_idx);
    size_t qp_idx = client->get_qp_idx();
    auto [endpoint_idx, _offset] =
        ::FarLib::get_config().map_remote_addr(entry->remote_addr());
    if (now >= ddl) [[unlikely]] {
        do {
            if (entry->is_local()) return true;
        } while (check_cq_idx_with_client_idx_endpoint(qp_idx, client_idx,
                                                        endpoint_idx));
        ddl = delay_fetch_ddl(ddl);
    }
    return false;
}

inline void ConcurrentArrayCache::fetch_wait_until_local(
    FarObjectEntry *entry, far_obj_t obj, size_t qp_idx, size_t client_idx,
    size_t endpoint_idx) {
    auto *diag_fibre = fibre_self();
    scope_diag::set_pending(
        diag_fibre, reinterpret_cast<uintptr_t>(entry));
    {
        scope_diag::Guard diag_guard(diag_fibre, scope_diag::RDMA_WAIT);
        while (!entry->is_local()) {
            check_cq_idx_with_client_idx_endpoint(qp_idx, client_idx,
                                                  endpoint_idx);
        }
    }
    if (auto *diag_slot = scope_diag::current(diag_fibre)) {
        request_interval_diag::requester_resumed(
            scope_diag::index(diag_slot),
            reinterpret_cast<uint64_t>(diag_fibre),
            reinterpret_cast<uint64_t>(entry));
    }
    scope_diag::clear_pending(diag_fibre);
}

inline void *ConcurrentArrayCache::fetch_with_miss_handler(
    far_obj_t obj, const DataMissHandler &handler, DereferenceScope &scope) {
    bool work_suspended = profile::suspend_work();
    bool at_local = post_fetch<true>(obj, scope, true);
    auto entry = &get_entry_of(obj);
    if (!at_local) [[unlikely]] {
        fetch_ddl_t ddl = create_fetch_ddl();
        profile::resume_work(work_suspended);
        handler(entry, ddl);
        auto client_idx = entry->get_client_idx();
        auto *client = rdma::get_client(client_idx);
        size_t qp_idx = client->get_qp_idx();
        auto [endpoint_idx, _offset] =
            ::FarLib::get_config().map_remote_addr(entry->remote_addr());
        fetch_wait_until_local(entry, obj, qp_idx, client_idx, endpoint_idx);
    }
    profile::resume_work(work_suspended);
    return entry->local_addr();
}

template <bool Mut>
inline void *ConcurrentArrayCache::fetch_lite(far_obj_t obj,
                                              const DataMissHandler &handler,
                                              DereferenceScope &scope) {
    auto entry = &get_entry_of(obj);
    auto state = entry->load_state(std::memory_order::relaxed);
    if (fetch_lite_fast_path<Mut>(state)) [[likely]] {
        record_local_fast_path_access(*entry,
                                      Mut ? profile::ReferenceKind::Write
                                          : profile::ReferenceKind::Read);
        return entry->local_addr();
    }
    return fetch_lite_slow_path<Mut>(obj, handler, scope);
}

template <bool Mut>
inline void *ConcurrentArrayCache::fetch_lite_no_profile(
    far_obj_t obj, const DataMissHandler &handler, DereferenceScope &scope) {
    auto entry = &get_entry_of(obj);
    auto state = entry->load_state(std::memory_order::relaxed);
    if (fetch_lite_fast_path<Mut>(state)) [[likely]] {
        record_local_fast_path_reference(*entry,
                                         Mut ? profile::ReferenceKind::Write
                                             : profile::ReferenceKind::Read);
        return entry->local_addr();
    }
    bool work_suspended = profile::suspend_work();
    bool at_local = post_fetch_lite_slow_path<Mut, false>(
        *entry, obj, scope, true);

    if (!at_local) [[unlikely]] {
        fetch_ddl_t ddl = create_fetch_ddl();
        handler(entry, ddl);
        auto client_idx = entry->get_client_idx();
        auto *client = rdma::get_client(client_idx);
        size_t qp_idx = client->get_qp_idx();
        auto [endpoint_idx, _offset] =
            ::FarLib::get_config().map_remote_addr(entry->remote_addr());

        fetch_wait_until_local(entry, obj, qp_idx, client_idx, endpoint_idx);
    }

    profile::resume_work(work_suspended);
    return entry->local_addr();
}

template <bool Mut>
inline void *ConcurrentArrayCache::fetch_lite_slow_path(
    far_obj_t obj, const DataMissHandler &handler, DereferenceScope &scope) {
    bool work_suspended = profile::suspend_work();
    auto entry = &get_entry_of(obj);
    bool at_local = post_fetch_lite_slow_path<Mut, true>(
        *entry, obj, scope, true);

    if (!at_local) [[unlikely]] {
        fetch_ddl_t ddl = create_fetch_ddl();
        handler(entry, ddl);
        auto client_idx = entry->get_client_idx();
        auto *client = rdma::get_client(client_idx);
        size_t qp_idx = client->get_qp_idx();
        auto [endpoint_idx, _offset] =
            ::FarLib::get_config().map_remote_addr(entry->remote_addr());

        fetch_wait_until_local(entry, obj, qp_idx, client_idx, endpoint_idx);
    }

    profile::resume_work(work_suspended);
    return entry->local_addr();
}

template <bool Mut>
inline std::pair<bool, void *> ConcurrentArrayCache::async_fetch_lite(
    far_obj_t obj, DereferenceScope &scope) {
    bool at_local = post_fetch_lite<Mut, true>(obj, scope);
    return {at_local, get_entry_of(obj).local_addr()};
}

template <bool Mut>
inline std::pair<bool, void *> ConcurrentArrayCache::async_fetch_lite_no_profile(
    far_obj_t obj, DereferenceScope &scope) {
    bool at_local = post_fetch_lite<Mut, false>(obj, scope);
    return {at_local, get_entry_of(obj).local_addr()};
}

}  // namespace FarLib::cache
