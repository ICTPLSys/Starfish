#pragma once
#include "cache/concurrent_cache.hpp"
#include "utils/scope_diag.hpp"

namespace FarLib::cache {

inline size_t ConcurrentArrayCache::check_cq() {
    // ft_method=sponge only: ec_batch never commits through the sponge path,
    // so it must not poll the control CQ from a data-CQ poller (24 pollers on
    // one control CQ livelocked libmlx5's pthread_spin_lock).
    if (::FarLib::get_config().is_sponge_mode()) check_sponge_ack_cq_ft();
    static std::atomic_flag ttas_lock = ATOMIC_FLAG_INIT;
    if (ttas_lock.test_and_set()) {
        return 0;
    }
    auto fid = fibre_self();
    scope_diag::Guard cq_guard(fid, scope_diag::Phase::CQ_POLL);
    auto client = rdma::Client::get_default();
    const bool read_batch = client->sync_read_batching_enabled();
    ibv_wc wc[rdma::CHECK_CQ_BATCH_SIZE];
    size_t total_cnt = 0;
    profile::start_check_cq();
    for (size_t endpoint = 0; endpoint < client->get_endpoint_count();
         ++endpoint) {
        const size_t cnt = client->check_cq_with_idx_endpoint(
            wc, rdma::CHECK_CQ_BATCH_SIZE, 0, endpoint);
        if (read_batch) client->note_cq_progress(endpoint, 0, cnt);
        for (size_t i = 0; i < cnt; i++) {
            {
                scope_diag::Guard wc_guard(fibre_self(),
                                           scope_diag::WC_HANDLE);
                handle_work_complete(wc[i]);
            }
        }
        if (read_batch) {
            (void)client->try_progress_sync_reads(endpoint, 0, false);
        }
        total_cnt += cnt;
    }
    profile::end_check_cq();
    scope_diag::cq_result(fid, total_cnt);
    ttas_lock.clear();
    return total_cnt;
}

inline size_t ConcurrentArrayCache::check_cq_idx(size_t qp_idx) {
    // ft_method=sponge only: ec_batch never commits through the sponge path,
    // so it must not poll the control CQ from a data-CQ poller (24 pollers on
    // one control CQ livelocked libmlx5's pthread_spin_lock).
    if (::FarLib::get_config().is_sponge_mode()) check_sponge_ack_cq_ft();
    auto client = rdma::Client::get_default();
    const bool read_batch = client->sync_read_batching_enabled();

    if (client->set_thread_lock()) {
        return 0;
    }
    auto fid = fibre_self();
    scope_diag::Guard cq_guard(fid, scope_diag::Phase::CQ_POLL);
    ibv_wc wc[rdma::CHECK_CQ_BATCH_SIZE];
    size_t total_cnt = 0;
    profile::start_check_cq();
    for (size_t endpoint = 0; endpoint < client->get_endpoint_count();
         ++endpoint) {
        const size_t cnt = client->check_cq_with_idx_endpoint(
            wc, rdma::CHECK_CQ_BATCH_SIZE, qp_idx, endpoint);
        if (read_batch) client->note_cq_progress(endpoint, qp_idx, cnt);
        for (size_t i = 0; i < cnt; i++) {
            {
                scope_diag::Guard wc_guard(fibre_self(),
                                           scope_diag::WC_HANDLE);
                handle_work_complete(wc[i]);
            }
        }
        if (read_batch) {
            (void)client->try_progress_sync_reads(endpoint, qp_idx, false);
        }
        total_cnt += cnt;
    }
    profile::end_check_cq();
    scope_diag::cq_result(fid, total_cnt);
    client->clear_thread_lock();

    return total_cnt;
}

inline size_t ConcurrentArrayCache::check_cq_idx_with_client_idx(
    size_t qp_idx, size_t thread_id) {
    // ft_method=sponge only: ec_batch never commits through the sponge path,
    // so it must not poll the control CQ from a data-CQ poller (24 pollers on
    // one control CQ livelocked libmlx5's pthread_spin_lock).
    if (::FarLib::get_config().is_sponge_mode()) check_sponge_ack_cq_ft();
    auto client = rdma::get_client(thread_id);
    const bool read_batch = client->sync_read_batching_enabled();
    auto fid = fibre_self();
    scope_diag::Guard cq_guard(fid, scope_diag::Phase::CQ_POLL);

    ibv_wc wc[rdma::CHECK_CQ_BATCH_SIZE];
    size_t total_cnt = 0;
    profile::start_check_cq();
    for (size_t endpoint = 0; endpoint < client->get_endpoint_count();
         ++endpoint) {
        const size_t cnt = client->check_cq_with_idx_endpoint(
            wc, rdma::CHECK_CQ_BATCH_SIZE, qp_idx, endpoint);
        if (read_batch) client->note_cq_progress(endpoint, qp_idx, cnt);
        for (size_t i = 0; i < cnt; i++) {
            {
                scope_diag::Guard wc_guard(fibre_self(),
                                           scope_diag::WC_HANDLE);
                handle_work_complete(wc[i]);
            }
        }
        if (read_batch) {
            (void)client->try_progress_sync_reads(endpoint, qp_idx, false);
        }
        total_cnt += cnt;
    }
    profile::end_check_cq();
    scope_diag::cq_result(fid, total_cnt);
    return total_cnt;
}

inline size_t ConcurrentArrayCache::check_cq_idx_with_client_idx_endpoint(
    size_t qp_idx, size_t thread_id, size_t endpoint_idx) {
    return poll_cq_idx_with_client_idx_endpoint_no_flush(
        qp_idx, thread_id, endpoint_idx);
}

inline size_t
ConcurrentArrayCache::poll_cq_idx_with_client_idx_endpoint_no_flush(
    size_t qp_idx, size_t thread_id, size_t endpoint_idx) {
    // ft_method=sponge only: ec_batch never commits through the sponge path,
    // so it must not poll the control CQ from a data-CQ poller (24 pollers on
    // one control CQ livelocked libmlx5's pthread_spin_lock).
    if (::FarLib::get_config().is_sponge_mode()) check_sponge_ack_cq_ft();
    auto client = rdma::get_client(thread_id);
    const bool read_batch = client->sync_read_batching_enabled();
    auto fid = fibre_self();
    scope_diag::Guard cq_guard(fid, scope_diag::Phase::CQ_POLL);

    ibv_wc wc[rdma::CHECK_CQ_BATCH_SIZE];
    profile::start_check_cq();
    size_t cnt = client->check_cq_with_idx_endpoint(
        wc, rdma::CHECK_CQ_BATCH_SIZE, qp_idx, endpoint_idx);
    scope_diag::cq_result(fid, cnt);
    if (read_batch) client->note_cq_progress(endpoint_idx, qp_idx, cnt);
    for (size_t i = 0; i < cnt; i++) {
        {
            scope_diag::Guard wc_guard(fibre_self(), scope_diag::WC_HANDLE);
            handle_work_complete(wc[i]);
        }
    }
    profile::end_check_cq();
    if (read_batch) {
        (void)client->try_progress_sync_reads(endpoint_idx, qp_idx, false);
    }
    return cnt;
}

inline void ConcurrentArrayCache::full_checker() {
    // ft_method=sponge only: ec_batch never commits through the sponge path,
    // so it must not poll the control CQ from a data-CQ poller (24 pollers on
    // one control CQ livelocked libmlx5's pthread_spin_lock).
    if (::FarLib::get_config().is_sponge_mode()) check_sponge_ack_cq_ft();
    for (size_t i = 0; i < rdma::get_client_count(); i++) {
        auto client = rdma::get_client(i);
        const bool read_batch = client->sync_read_batching_enabled();

        ibv_wc wc[rdma::CHECK_CQ_BATCH_SIZE];
        for (size_t endpoint = 0;
             endpoint < client->get_endpoint_count(); ++endpoint) {
            const size_t cnt = client->check_cq_with_idx_endpoint(
                wc, rdma::CHECK_CQ_BATCH_SIZE, 0, endpoint);
            if (read_batch) client->note_cq_progress(endpoint, 0, cnt);
            for (size_t j = 0; j < cnt; j++) {
                {
                    scope_diag::Guard wc_guard(fibre_self(),
                                               scope_diag::WC_HANDLE);
                    handle_work_complete(wc[j]);
                }
            }
            if (read_batch) {
                (void)client->try_progress_sync_reads(endpoint, 0, false);
            }
        }
    }
}

inline void ConcurrentArrayCache::pin(far_obj_t obj) {
    if (obj.is_null()) return;
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
    new_state.inc_ref_cnt();
    if (new_state.state == MARKED || new_state.state == EVICTING) {
        new_state.state = LOCAL;
    }
    if (!entry.cas_state_weak(old_state, new_state)) [[unlikely]] {
        goto retry;
    }
}

inline void ConcurrentArrayCache::unpin(far_obj_t obj) {
    if (obj.is_null()) return;
    auto &entry = get_entry_of(obj);
retry:
    auto old_state = entry.load_state();
    if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
        auto spin_start = get_cycles();
        profile::count_excl_move_lock_spin();
        profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
        goto retry;
    }
    assert(old_state.state != MARKED);
    assert(old_state.state != EVICTING);
    assert(old_state.state != REMOTE);
    if (old_state.ref_cnt == 0) {
        return;
    }
    auto new_state = old_state;
    new_state.dec_ref_cnt();
    if (!entry.cas_state_weak(old_state, new_state)) [[unlikely]] {
        goto retry;
    }
}

inline void ConcurrentArrayCache::mark_dirty(far_obj_t obj) {
    FarObjectEntry &entry = get_entry_of(obj);
    wait_ec_write_source(entry);
    record_non_fast_path_reference(entry, profile::ReferenceKind::Write);
    if (invalidate_retained_backup_for_write(entry, obj.size)) {
        return;
    }
    EntryStateBits old_state = entry.load_state();
    if (old_state.dirty) return;
    for (;;) {
        old_state = entry.load_state();
        // Keep this state snapshot from BEFORE the borrow check: a later
        // eviction/rescue adds a write ref and therefore makes this CAS fail.
        wait_ec_write_source(entry);
        if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
            auto spin_start = get_cycles();
            profile::count_excl_move_lock_spin();
            old_state = entry.load_state();
            profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
            continue;
        }
        auto new_state = old_state;
        new_state.dirty = 1;
        if (entry.cas_state_weak(old_state, new_state)) return;
    }
}

inline void ConcurrentArrayCache::release_cache(FarObjectEntry *entry, bool dirty) {
    if (dirty) {
        wait_ec_write_source(*entry);
        invalidate_retained_backup_for_write(
            *entry, entry->load_state(std::memory_order_relaxed).size);
    }
    for (;;) {
        auto old_state = entry->load_state();
        if (dirty) wait_ec_write_source(*entry);
        if (::FarLib::get_config().exclusive_cache && old_state.invalid) [[unlikely]] {
            auto spin_start = get_cycles();
            profile::count_excl_move_lock_spin();
            old_state = entry->load_state();
            profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
            continue;
        }
        auto new_state = old_state;
        if (new_state.state == FREE) [[unlikely]] {
            ERROR("release cache: try to release a free entry");
        }
        if (new_state.ref_cnt == 0) [[unlikely]] {
            if (dirty) new_state.dirty = 1;
            if (entry->cas_state_weak(old_state, new_state)) {
                return;
            }
            continue;
        }
        new_state.dec_ref_cnt();
        if (dirty) new_state.dirty = 1;
        if (entry->cas_state_weak(old_state, new_state)) return;
    }
}

inline void ConcurrentArrayCache::release_cache(far_obj_t obj, bool dirty) {
    release_cache(&get_entry_of(obj), dirty);
}

inline void ConcurrentArrayCache::enter_scope() {
    assert(uthread::get_tls()->scope_state == OutOfScope);
    if (scope_counter_shards_enabled) {
        const int token = scope_counters.enter();
        uthread::get_tls()->scope_state = token;
        scope_diag::on_enter(fibre_self(), ShardedScopeCounters::state(token));
        return;
    }
    auto state = mutator_states.global_state.load(std::memory_order::relaxed);
    uthread::get_tls()->scope_state = state;
    mutator_states.count[state].fetch_add(1, std::memory_order::relaxed);
    scope_diag::on_enter(fibre_self(), state);
}

inline void ConcurrentArrayCache::exit_scope() {
    auto prev_state = uthread::get_tls()->scope_state;
    assert(prev_state != OutOfScope);
    uthread::get_tls()->scope_state = OutOfScope;
    if (scope_counter_shards_enabled) {
        scope_counters.leave(prev_state);
        scope_diag::on_exit(fibre_self(), ShardedScopeCounters::state(prev_state));
        return;
    }
    mutator_states.count[prev_state].fetch_sub(1, std::memory_order::relaxed);
    scope_diag::on_exit(fibre_self(), prev_state);
}

inline bool ConcurrentArrayCache::evacuator_waiting() { return false; }

inline void ConcurrentArrayCache::update_scope(DereferenceScope &scope) {
    auto &state = uthread::get_tls()->scope_state;
    assert(state != OutOfScope);
    if (scope_counter_shards_enabled) {
        const int old_state = ShardedScopeCounters::state(state);
        if (scope_counters.current_state() != old_state) {
            const int old_token = state;
            scope_diag::before_update(fibre_self(), old_state, scope_counters.current_state());
            scope.recursive_pin();
            const int new_token = scope_counters.enter_on(ShardedScopeCounters::shard(old_token));
            state = new_token;
            scope_counters.leave(old_token);
            scope.recursive_unpin();
            scope_diag::after_update(fibre_self(), old_state, ShardedScopeCounters::state(new_token));
        }
        return;
    }
    auto new_state = mutator_states.global_state.load(std::memory_order::relaxed);
    if (new_state != state) {
        auto old_state = state;
        scope_diag::before_update(fibre_self(), old_state, new_state);
        // Pin every declared live object before releasing the old generation;
        // the new-generation count protects the chain until it is unpinned.
        scope.recursive_pin();
        mutator_states.count[new_state].fetch_add(1, std::memory_order::relaxed);
        state = new_state;
        mutator_states.count[old_state].fetch_sub(1, std::memory_order::release);
        scope.recursive_unpin();
        scope_diag::after_update(fibre_self(), old_state, new_state);
    }
}

}  // namespace FarLib::cache
