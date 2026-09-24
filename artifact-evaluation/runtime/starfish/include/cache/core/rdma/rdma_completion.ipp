#pragma once
#include "cache/concurrent_cache.hpp"

namespace FarLib::cache {

namespace detail {

struct NormalReadIdentity {
    ::FarLib::allocator::BlockHead *block = nullptr;
    void *local_addr = nullptr;
    uint16_t generation = 0;
    bool tagged = false;
};

// Decode and validate an ordinary READ token before forming a BlockHead
// pointer.  `allow_raw_pointer` is reserved for the EC recovery path's
// synthetic completion; hardware ordinary READ WCs must carry the generation
// token, so a stale raw address cannot pass the ABA check.
inline NormalReadIdentity decode_normal_read_wr_id(
    uint64_t wr_id, bool allow_raw_pointer = false) {
    NormalReadIdentity result;
    if (wr_id == 0) return result;
    const bool tagged =
        ::FarLib::allocator::is_normal_read_wr_id(wr_id);
    if (!tagged && !allow_raw_pointer) return result;
    const uintptr_t data_addr = reinterpret_cast<uintptr_t>(
        tagged ? ::FarLib::allocator::normal_read_wr_id_local_addr(wr_id)
               : reinterpret_cast<void *>(wr_id));
    const uintptr_t heap_addr = reinterpret_cast<uintptr_t>(
        ::FarLib::allocator::global_heap.get_heap());
    const size_t heap_size = ::FarLib::allocator::global_heap.get_heap_size();
    if (heap_addr == 0 || data_addr < heap_addr) return result;
    const uintptr_t offset = data_addr - heap_addr;
    if (offset < sizeof(::FarLib::allocator::BlockHead) ||
        offset >= heap_size) {
        return result;
    }
    result.block = reinterpret_cast<::FarLib::allocator::BlockHead *>(
        data_addr - sizeof(::FarLib::allocator::BlockHead));
    result.local_addr = reinterpret_cast<void *>(data_addr);
    result.tagged = tagged;
    result.generation = tagged
        ? ::FarLib::allocator::normal_read_wr_id_generation(wr_id)
        : 0;
    if (tagged &&
        (result.generation == 0 ||
         result.block->rdma_read_generation.load(std::memory_order_acquire) !=
             result.generation)) {
        return {};
    }
    return result;
}

// Drop exactly one ordinary READ lifetime pin.  A missing pin means that the
// completion is late/duplicate (or was a legacy request that was never pinned)
// and is deliberately ignored.  The object is inspected while the pin still
// holds the block in the allocator; this avoids racing reclamation after the
// final decrement.
inline bool release_normal_read_pin(uint64_t wr_id) {
    const auto identity = decode_normal_read_wr_id(wr_id, false);
    auto *block = identity.block;
    if (block == nullptr || !identity.tagged) return false;
    while (block->rdma_read_pin_lock.test_and_set(std::memory_order_acquire)) {
    }
    const uint16_t current_generation = block->rdma_read_generation.load(
        std::memory_order_relaxed);
    if (current_generation != identity.generation) {
        static std::atomic<uint64_t> generation_mismatch_logs{0};
        if (generation_mismatch_logs.fetch_add(1, std::memory_order_relaxed) <
            20) {
            std::cerr << "WARN: ordinary READ pin generation mismatch wr_id="
                      << wr_id << " token_generation=" << identity.generation
                      << " current_generation=" << current_generation
                      << " pending="
                      << block->pending_rdma_reads.load(
                             std::memory_order_relaxed)
                      << std::endl;
        }
        block->rdma_read_pin_lock.clear(std::memory_order_release);
        return false;
    }
    const uint32_t pending =
        block->pending_rdma_reads.load(std::memory_order_relaxed);
    if (pending == 0) {
        static std::atomic<uint64_t> empty_pin_logs{0};
        if (empty_pin_logs.fetch_add(1, std::memory_order_relaxed) < 20) {
            std::cerr << "WARN: ordinary READ pin already empty wr_id="
                      << wr_id << std::endl;
        }
        block->rdma_read_pin_lock.clear(std::memory_order_release);
        return false;
    }
    const bool was_null = block->obj_meta_data.load(
        std::memory_order_acquire).is_null();
    block->pending_rdma_reads.store(pending - 1, std::memory_order_release);
    block->rdma_read_pin_lock.clear(std::memory_order_release);
    if (pending == 1 && was_null) {
        // A mark/evict pass may have retained a FREE block solely for this
        // READ.  Make the next reclaim pass notice the transition to zero even
        // when no write completion is involved.
        ::FarLib::allocator::block_to_region(block)
            ->pending_reclaims.fetch_add(1, std::memory_order_release);
    }
    return true;
}

}  // namespace detail

inline void ConcurrentArrayCache::complete_evict_writeback(void *local_ptr) {
    ::FarLib::allocator::BlockHead *block =
        static_cast<::FarLib::allocator::BlockHead *>(local_ptr) - 1;
retry:
    auto obj = block->obj_meta_data.load(std::memory_order_acquire);
    // A duplicate or delayed completion may arrive after the first
    // completion has detached this block from its object.  The block can
    // still be physically present in a region, but there is no entry to
    // dereference anymore; treating the completion as stale is safe.
    if (obj.is_null()) [[unlikely]] return;
    auto &entry = get_entry_of(obj);
    auto old_state = entry.load_state();

    if (six_dirty_routing_enabled() && old_state.invalid) goto retry;

    if (old_state.state == PINNED) return;
    // FREE is terminal: the deallocator already consumed this completion's
    // ownership. Retrying cannot make the old local block valid again.
    if (old_state.state == FREE) [[unlikely]] return;
    if (entry.local_addr() != local_ptr) [[unlikely]] goto retry;
    if ((old_state.state == EntryState::LOCAL ||
         old_state.state == EntryState::MARKED) &&
        old_state.ref_cnt == 0) {
        // A stale write completion can arrive after an interrupted eviction
        // has already been converted back to LOCAL and re-marked for a
        // future eviction cycle. The completion no longer owns a write ref
        // in this state, so it should not mutate the entry.
        return;
    }
    auto new_state = old_state;
    // each RDMA write will increase the reference count
    // to avoid deallocating the local buffer while RDMA write is not done
    new_state.dec_ref_cnt();
    switch (new_state.state) {
    case EntryState::EVICTING:
        if (new_state.ref_cnt == 0) {
            if (six_dirty_routing_enabled()) {
                // Keep fetch/move/free out until binding and occupancy are committed.
                auto locked = new_state;
                locked.invalid = 1;
                if (!entry.cas_state_weak(old_state, locked)) goto retry;
                const auto kind = entry.take_six_pending_evict();
                ASSERT(kind != 0);
                six_commit_evict(entry, obj.size, kind == 2);
                new_state.state = REMOTE;
                auto expected = locked;
                ASSERT(entry.cas_state_strong(expected, new_state));
            } else {
                new_state.state = REMOTE;
                if (!entry.cas_state_weak(old_state, new_state)) goto retry;
            }
            block->obj_meta_data.store(far_obj_t::null(),
                                       std::memory_order_release);
            // The block remains linked from marked_list until a reclaim pass.
            // Publish the asynchronous completion so the allocator cannot
            // reclaim/reuse it before a late duplicate completion is seen.
            ::FarLib::allocator::block_to_region(block)
                ->pending_reclaims.fetch_add(
                    1, std::memory_order_release);
        } else {
            // maybe another rdma request (write back) in the progress
            // do not mark it as remote to prevent reallocating memory to
            // other object
            if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        }
        break;
    case EntryState::LOCAL:
        // another access interrupted the eviction
        // Consume this completion's write reference exactly once.
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        if (new_state.ref_cnt != 0) {
            // Another write still owns the local buffer and remote slot.
            break;
        }
        if (::FarLib::get_config().exclusive_cache) {
            // Corner case: eviction was interrupted and the object stayed LOCAL.
            // The remote slot allocated for this eviction must be freed (after the
            // write completes) to preserve strict-exclusivity (LOCAL => no remote).
            // Cleanup retries must not return to the decrement path above.
            while (true) {
                auto cleanup_state = entry.load_state();
                if (cleanup_state.state != EntryState::LOCAL &&
                    cleanup_state.state != EntryState::MARKED) {
                    return;
                }
                if (cleanup_state.invalid) {
                    uthread::yield();
                    continue;
                }
                auto lock_state = cleanup_state;
                lock_state.invalid = 1;
                if (!entry.cas_state_weak(cleanup_state, lock_state)) {
                    continue;
                }
                uint64_t old_remote = entry.remote_addr();
                if (six_dirty_routing_enabled())
                    entry.take_six_pending_evict();
                if (old_remote != FarObjectEntry::RemoteAddrInvalid48) {
                    const bool remote_endpoint_dead =
                        reuse_ec_recovery_endpoint_is_dead(old_remote);
                    const bool keep = all_nonresident_backup_mode &&
                        !cleanup_state.dirty && !entry.is_resident_local() &&
                        !remote_endpoint_dead &&
                        (entry.has_remote_backup_reservation() ||
                         try_reserve_remote_backup(obj.size));
                    if (keep) {
                        entry.set_remote_backup_reservation(true);
                        nonresident_interrupted_keep_count.fetch_add(1, std::memory_order_relaxed);
                        nonresident_interrupted_keep_bytes.fetch_add(obj.size, std::memory_order_relaxed);
                    } else {
                        if ((all_nonresident_backup_mode || remote_endpoint_dead) &&
                            entry.has_remote_backup_reservation()) {
                            entry.set_remote_backup_reservation(false);
                            release_remote_backup_budget(obj.size);
                            if (remote_endpoint_dead) {
                                profile::count_remote_backup_invalidated(obj.size);
                            }
                        }
                        entry.set_remote_invalid();
                        remote_allocator.deallocate(old_remote);
                        profile::count_excl_interrupted_evict_free(obj.size);
                    }
                }
                auto unlock_state = entry.load_state();
                while (unlock_state.invalid) {
                    auto final_state = unlock_state;
                    final_state.invalid = 0;
                    if (entry.cas_state_weak(unlock_state, final_state)) {
                        return;
                    }
                }
                return;
            }
        }
        break;
    case EntryState::BUSY:
        goto retry;
    default:
        ERROR("check_cq: invalid state when evict");
    }
}

inline void ConcurrentArrayCache::handle_rdma_write_complete(const ibv_wc &wc) {
    struct WriteCompleteCycleGuard {
        bool enabled;
        uint64_t start;
        ~WriteCompleteCycleGuard() {
            if (enabled) {
                profile::count_evac_write_complete_cycles(
                    (int64_t)(get_cycles() - start));
            }
        }
    } cycle_guard{profile::evac_fine_profile_runtime_enabled(), 0};
    if (cycle_guard.enabled) {
        cycle_guard.start = get_cycles();
    }
    // EC (ft_method=ec_batch): a wr_id of the six segment writes carries the
    // group token instead of an object pointer.  The tag bit keeps the two
    // namespaces disjoint, so the legacy "wr_id == object pointer" contract is
    // untouched (and nothing changes while ft_method == none: no ec_batch
    // wr_id can ever be posted in that mode).
    if (ec_batch::is_ec_batch_wr_id(wc.wr_id)) [[unlikely]] {
        handle_ec_batch_write_complete(wc.wr_id);
        return;
    }
    complete_evict_writeback(reinterpret_cast<void *>(wc.wr_id));
}

// EC (ft_method=sponge): the data server's final ACK for one commit.  It means
// the data shard plus both parity shards are in place, so the eviction may now
// go through the ordinary EVICTING -> REMOTE release.
inline void ConcurrentArrayCache::handle_sponge_commit_ack(uint64_t wr_id,
                                                           uint16_t status) {
    if (status != rdma::SPONGE_RPC_STATUS_OK) [[unlikely]] {
        ERROR("sponge commit failed");
    }
    complete_evict_writeback(reinterpret_cast<void *>(wr_id));
}

// EC (ft_method=sponge): decode one sponged ACK batch (a SEND from the data
// server), complete every record it carries and repost the receive slot.
// Returns false when the WC is not one of our ACK slots.
inline bool ConcurrentArrayCache::handle_sponge_ack_complete(const ibv_wc &wc) {
    rdma::SpongeAckBatchView view;
    rdma::SpongeAckWireRecvSlot *slot = nullptr;
    if (!rdma::Client::decode_sponge_ack_recv_slot(wc.wr_id, wc.byte_len, view,
                                                   &slot)) {
        if (slot == nullptr) {
            return false;
        }
        auto *control = rdma::ClientControl::get_default();
        if (control == nullptr ||
            !control->repost_sponge_ack_recv_slot(wc.wr_id)) [[unlikely]] {
            ERROR("failed to repost invalid sponge ACK recv slot");
        }
        return true;
    }
    for (size_t i = 0; i < view.wire.header->record_count; i++) {
        const auto &record = view.records()[i];
        handle_sponge_commit_ack(record.wr_id, record.status);
    }
    auto *control = rdma::ClientControl::get_default();
    if (control == nullptr ||
        !control->repost_sponge_ack_recv_slot(wc.wr_id)) [[unlikely]] {
        ERROR("failed to repost sponge ACK recv slot");
    }
    return true;
}

// EC (ft_method=sponge): drain the control CQ, which carries the commit RPC
// SEND completions (slot release) and the data server's ACK batches
// (EVICTING -> REMOTE).  Only called when ft_method != none.
inline void ConcurrentArrayCache::check_sponge_ack_cq_ft() {
    auto *client = rdma::Client::get_default();
    if (client == nullptr) {
        return;
    }
    ibv_wc wc[rdma::CHECK_CQ_BATCH_SIZE];
    const size_t cnt =
        client->check_sponge_ack_cq(wc, rdma::CHECK_CQ_BATCH_SIZE);
    for (size_t i = 0; i < cnt; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            static std::atomic<int> sponge_wc_error_log{0};
            if (sponge_wc_error_log.fetch_add(1, std::memory_order_relaxed) <
                10) {
                std::cerr << "ERROR: sponge control wc status=" << wc[i].status
                          << " opcode=" << wc[i].opcode
                          << " wr_id=" << wc[i].wr_id << std::endl;
            }
            continue;
        }
        if (wc[i].opcode == IBV_WC_SEND) {
            rdma::Client::release_sponge_commit_send_slot(wc[i].wr_id);
        } else if (wc[i].opcode == IBV_WC_RECV) {
            (void)handle_sponge_ack_complete(wc[i]);
        }
    }
}

inline void ConcurrentArrayCache::handle_rdma_read_complete(
    const ibv_wc &wc, bool release_read_pin) {
    const auto identity =
        detail::decode_normal_read_wr_id(wc.wr_id, !release_read_pin);
    void *local_ptr = identity.local_addr;
    ::FarLib::allocator::BlockHead *block = identity.block;
    if (block == nullptr) [[unlikely]] {
        static std::atomic<int> invalid_read_wr_id_log{0};
        if (invalid_read_wr_id_log.fetch_add(1, std::memory_order_relaxed) <
            10) {
            std::cerr << "WARN: dropping invalid ordinary READ wr_id="
                      << wc.wr_id << std::endl;
        }
        return;
    }
    if (release_read_pin && !identity.tagged) [[unlikely]] {
        return;
    }
    // A normal READ must hold a pin from generation through completion.  If
    // the pin is already gone, this is a delayed/duplicate WC and must not
    // inspect a block that may have been reclaimed and reused.
    if (release_read_pin &&
        block->pending_rdma_reads.load(std::memory_order_acquire) == 0) {
        return;
    }
    struct ReadPinGuard {
        bool enabled;
        uint64_t wr_id;
        ~ReadPinGuard() {
            if (enabled) {
                (void)detail::release_normal_read_pin(wr_id);
            }
        }
    } read_pin_guard{release_read_pin, wc.wr_id};
retry:
    auto obj = block->obj_meta_data.load(std::memory_order_acquire);
    if (obj.is_null()) [[unlikely]] return;
    auto &entry = get_entry_of(obj);
    auto old_state = entry.load_state();
    if (old_state.state == FREE || entry.local_addr() != local_ptr) [[unlikely]] {
        return;
    }
    if (old_state.state == EntryState::BUSY) [[unlikely]] {
        goto retry;
    }
    // LOCAL/MARKED/PINNED are terminal for this completion: another fetch
    // completion or a recovery path already won the race.  Retrying a stale WC
    // here can spin forever and can eventually touch a reused entry.
    if (old_state.state != EntryState::FETCHING) [[unlikely]] {
        return;
    }
    if (::FarLib::get_config().exclusive_cache) {
        if (old_state.invalid) [[unlikely]] {
            auto spin_start = get_cycles();
            profile::count_excl_move_lock_spin();
            profile::count_excl_move_lock_spin_cycles(get_cycles() - spin_start);
            goto retry;
        }

        auto lock_state = old_state;
        lock_state.invalid = 1;
        if (!entry.cas_state_weak(old_state, lock_state)) [[unlikely]] {
            goto retry;
        }

        uint64_t old_remote = entry.remote_addr();
        // T7: Invariant check - fetch from remote must have a valid remote address
        if (old_remote == FarObjectEntry::RemoteAddrInvalid48) {
            std::string err = "FETCH_FAIL: remote_addr is invalid for obj=" +
                              std::to_string(obj.obj_id) +
                              " state=" + std::to_string(old_state.state) + "\n";
            EXCLUSIVE_ERROR(err);
            std::abort();
        }

        const bool remote_endpoint_dead =
            reuse_ec_recovery_endpoint_is_dead(old_remote);
        bool retain_backup = entry.has_remote_backup_reservation();
        const bool placement_allows_backup =
            local_placement_allows_remote_backup(entry);
        // A degraded READ can reconstruct a clean object while its old data
        // segment still points at the failed endpoint.  Do not retain or
        // newly reserve that address: the move-lock above makes this the
        // current remote address for this entry, and the normal deallocation
        // below releases its slot exactly once.  Keep the fetched object
        // clean; endpoint failure is not a user write.
        if (remote_endpoint_dead) {
            if (retain_backup) {
                entry.set_remote_backup_reservation(false);
                release_remote_backup_budget(obj.size);
                profile::count_remote_backup_invalidated(obj.size);
            }
            retain_backup = false;
        } else if (retain_backup &&
                   (lock_state.dirty || !placement_allows_backup)) {
            entry.set_remote_backup_reservation(false);
            release_remote_backup_budget(obj.size);
            if (lock_state.dirty) {
                profile::count_remote_backup_invalidated(obj.size);
            } else {
                resident_profile_released_backup_bytes.fetch_add(
                    obj.size, std::memory_order_relaxed);
            }
            retain_backup = false;
        } else if (!retain_backup && !lock_state.dirty &&
                   placement_allows_backup &&
                   selective_backup_enabled() &&
                   !segmented_backup_enabled()) {
            retain_backup = try_reserve_remote_backup(obj.size);
            entry.set_remote_backup_reservation(retain_backup);
        }
        if (all_nonresident_backup_mode && !remote_endpoint_dead &&
            !lock_state.dirty && placement_allows_backup) {
            // Never silently turn a clean non-Resident fetch into strict-exclusive.
            ASSERT(retain_backup);
        }
        if (!retain_backup) {
            entry.set_remote_backup_reservation(false);
            entry.set_remote_invalid();
        }

        six_commit_fetch(entry, local_ptr, obj.size);
        auto new_state = lock_state;
        new_state.invalid = 0;
        new_state.state = LOCAL;
        auto expected = lock_state;
        uint32_t finalize_retry_count = 0;
        while (true) {
            request_interval_diag::Stamp local_before{};
            if (request_interval_diag::completion_context.active) {
                local_before = request_interval_diag::ordered_stamp();
            }
            if (entry.cas_state_strong(expected, new_state)) {
                request_interval_diag::Stamp local_after{};
                if (request_interval_diag::completion_context.active) {
                    local_after = request_interval_diag::ordered_stamp();
                    request_interval_diag::local_published(
                        wc.wr_id, local_before, local_after);
                }
                break;
            }
            const bool retryable =
                expected.invalid && expected.state == EntryState::FETCHING &&
                expected.dirty == lock_state.dirty &&
                expected.size == lock_state.size;
            if (!retryable || finalize_retry_count >= 64) {
                std::cerr << "FETCH_FINALIZE_STATE_CONFLICT"
                          << " obj=" << obj.obj_id
                          << " retry_count=" << finalize_retry_count
                          << " observed_state="
                          << static_cast<uint32_t>(expected.state)
                          << " observed_ref_cnt="
                          << static_cast<uint32_t>(expected.ref_cnt)
                          << std::endl;
                std::abort();
            }
            new_state = expected;
            new_state.invalid = 0;
            new_state.state = LOCAL;
            ++finalize_retry_count;
        }

        // Sparse observer timestamp after the LOCAL state is published.  The
        // requesting fibre may resume before the remote accounting below ends.
        read_supply_timeline::record_lifecycle_local(wc.wr_id);

        if (retain_backup) {
            return;
        }

        auto free_start = get_cycles();
        {
            scope_diag::Guard remote_free_guard(fibre_self(),
                                                scope_diag::REMOTE_FREE);
            remote_allocator.deallocate(old_remote);
        }
        auto remote_free_cycles = get_cycles() - free_start;
        profile::count_excl_remote_free_on_fetch_cycles(remote_free_cycles);
        profile::count_excl_remote_free_on_fetch(obj.size);
        constexpr int64_t RemoteFreeLogThreshold = 10000000;  // 10M cycles
        if (remote_free_cycles > RemoteFreeLogThreshold) {
            profile::count_excl_remote_free_on_fetch_slow(remote_free_cycles,
                                                          obj.size);
        }
    } else {
        auto new_state = old_state;
        new_state.state = LOCAL;
        if (!entry.cas_state_weak(old_state, new_state)) goto retry;
        read_supply_timeline::record_lifecycle_local(wc.wr_id);
    }
}

inline void ConcurrentArrayCache::handle_work_complete(const ibv_wc &wc) {
    if (wc.status != IBV_WC_SUCCESS) {
        // EC (ft_method=ec_batch) read-side recovery: a failed completion is
        // how an endpoint that went away announces itself.  Remember it before
        // the CQE is dropped - the QP of that endpoint is in error and is never
        // retried, but every later read of a segment that lives on it has to be
        // served from the surviving segments of its slot group instead.
        // Gated: with ft_method=none this branch is exactly the old one.
        if (::FarLib::get_config().is_ec_batch_mode()) [[unlikely]] {
            note_ec_recovery_error_wc(wc);
            if (ec_batch::is_ec_batch_wr_id(wc.wr_id)) {
                // Failed/flush completions also discharge hardware ownership.
                // The group succeeds only if at least four other durable
                // segments remain; it must never strand its write reference.
                handle_ec_batch_write_complete(wc.wr_id, false);
            }
        }
        // A failed ordinary READ still owns the allocator block until this
        // error WC is consumed.  EC-tagged reads use staging storage and do
        // not carry this pin.
        // Error WCs from mlx5 are not consistent about preserving opcode
        // (some flushed READs arrive with opcode=0).  The wr_id classifier in
        // release_normal_read_pin() rejects EC tags/control storage, so use
        // the pin itself as the ordinary-READ discriminator here.
        (void)detail::release_normal_read_pin(wc.wr_id);
        static std::atomic<int> err_wc_log{0};
        if (err_wc_log.fetch_add(1, std::memory_order_relaxed) < 10) {
            std::cerr << "ERROR: rdma wc status=" << wc.status
                      << " opcode=" << wc.opcode << " wr_id=" << wc.wr_id
                      << " qp_num=" << wc.qp_num << std::endl;
        }
        return;
    }

    // Skip invalid opcodes (garbage from wrong CQ)
    if (wc.opcode > IBV_WC_RECV) {
        return;
    }

    if (wc.opcode == IBV_WC_RDMA_WRITE) {
        handle_rdma_write_complete(wc);
    } else if (wc.opcode == IBV_WC_RDMA_READ) {
        // EC (ft_method=ec_batch) degraded read: a wr_id of the read recovery
        // round carries (token, segment) instead of an object pointer, so it
        // must never reach handle_rdma_read_complete(), which would dereference
        // it.  Gated: no such wr_id can exist unless ec_batch runs, and the
        // classifier also rejects an ordinary object pointer.
        if (ec_recovery_is_read_wr_id(wc.wr_id)) [[unlikely]] {
            handle_ec_read_segment_complete(wc.wr_id);
        } else {
            handle_rdma_read_complete(wc);
        }
    } else if (wc.opcode == IBV_WC_SEND) {
        // EC commit RPC: the SEND completion releases its client send slot.
        // Gated: with ft_method=none the condition is false and the WC is
        // ignored exactly as before.
        if (::FarLib::get_config().ft_enabled() &&
            rdma::Client::release_sponge_commit_send_slot(wc.wr_id)) {
            return;
        }
        // Control QP operations - ignore
    } else if (wc.opcode == IBV_WC_RECV) {
        // EC commit RPC: the data server's final ACK releases the eviction.
        if (::FarLib::get_config().ft_enabled() &&
            handle_sponge_ack_complete(wc)) {
            return;
        }
        // Control QP operations - ignore
        static std::atomic<int> bad_wc_log{0};
        if (bad_wc_log.fetch_add(1, std::memory_order_relaxed) < 10) {
            std::cerr << "WARN: unexpected rdma wc opcode: " << wc.opcode
                      << " status=" << wc.status << " wr_id=" << wc.wr_id
                      << " qp_num=" << wc.qp_num
                      << " wc_flags=" << wc.wc_flags
                      << " byte_len=" << wc.byte_len << std::endl;
        }
    }
}

}  // namespace FarLib::cache
