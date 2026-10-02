// ---------------------------------------------------------------------------
// ec_batch write path, stage B2 - the six-segment single-sided write round and
// the group-token completion path of ConcurrentArrayCache.
//
// Included at the end of concurrent_cache.hpp (like rdma_completion.ipp), so
// the class is complete here and every member access resolves.
//
// Flow:
//   post_ec_batch_group(record)      acquire a token, post six RDMA WRITEs
//                                    (4 data + 2 parity, one per endpoint)
//   ... six CQEs, possibly on six different completion threads ...
//   handle_ec_batch_write_complete() complete every live object of the group
//                                    through complete_evict_writeback(),
//                                    then return staging slot + token
//
// Only reachable while get_config().is_ec_batch_mode(): the staging pool is
// initialized by the constructor only in that mode, and post_ec_batch_group()
// returns false before touching anything otherwise.  ft_method=none keeps its
// old branch (handle_rdma_write_complete() only adds one high-bit test per
// write CQE, and no ec_batch wr_id can ever exist in that mode).
// ---------------------------------------------------------------------------
#pragma once
#include <sstream>

#include "cache/concurrent_cache.hpp"

namespace FarLib::cache::ec_batch {

// Local source of segment `segment` inside the record's staging slot: the four
// data slots first, then the two parity slots - the order of group.segments.
inline void *ec_batch_record_source(const EcGroupSendRecord &record,
                                    size_t segment) {
    if (record.source_mode == EcGroupSourceMode::kSplitDirect &&
        segment < kEcBatchDataSlots) {
        return const_cast<void *>(record.split_data_sources[segment]);
    }
    if (segment < kEcBatchDataSlots) return record.staging.data[segment];
    return record.staging.parity[segment - kEcBatchDataSlots];
}

inline uint32_t ec_batch_record_source_bytes(
    const EcGroupSendRecord &record) {
    return record.source_mode == EcGroupSourceMode::kSplitDirect
               ? record.split_direct_write_bytes
               : record.slot_size;
}

inline bool ec_batch_record_source_uses_staging(
    const EcGroupSendRecord &record, size_t segment,
    bool shared_staging_bound = false) {
    if (record.source_mode == EcGroupSourceMode::kSplitDirect)
        return segment >= kEcBatchDataSlots;
    return record.split_object || record.size_class_staging ||
           shared_staging_bound;
}

// A record is postable only when all six segments carry a usable remote target
// and a local source inside the resident staging MR.  Checked before the first
// post so a rejected group cannot leave part of its segments on the wire.
template <class Pool>
inline bool ec_batch_record_is_postable(const EcGroupSendRecord &record,
                                        const Pool &pool,
                                        size_t server_count) {
    if (record.slot_size == 0) return false;
    if (record.staging.slot_size < record.slot_size) return false;
    // The shared provider validates the immutable pool/node cookie, lease
    // generation, active state, and all six source pointers.  The fixed-MR
    // fallback performs the equivalent range/free-list checks.
    if (!pool.owns_slot(record.staging)) return false;
    if (record.group.segments.size() != kEcBatchSegmentsPerGroup) return false;
    if (record.split_object) {
        if (record.objects[0] == nullptr || record.object_sizes[0] == 0 ||
            ec_split::fragment_size(record.object_sizes[0]) > record.slot_size)
            return false;
        for (size_t i = 1; i < kEcBatchDataSlots; ++i)
            if (record.objects[i] != nullptr || record.object_sizes[i] != 0)
                return false;
        if (record.source_mode == EcGroupSourceMode::kSplitDirect) {
            const uint32_t bytes = record.split_direct_write_bytes;
            if (bytes == 0 || bytes > record.slot_size ||
                record.object_sizes[0] !=
                    static_cast<uint64_t>(bytes) * kEcBatchDataSlots) {
                return false;
            }
            for (size_t i = 0; i < kEcBatchDataSlots; ++i) {
                if (record.split_data_sources[i] == nullptr ||
                    record.split_data_lengths[i] != bytes) {
                    return false;
                }
            }
        }
    } else if (record.source_mode == EcGroupSourceMode::kSplitDirect) {
        return false;
    }
    for (size_t i = 0; !record.split_object && i < kEcBatchDataSlots; i++) {
        if (record.objects[i] != nullptr &&
            record.object_sizes[i] > record.slot_size) {
            return false;
        }
    }
    for (size_t segment = 0; segment < kEcBatchSegmentsPerGroup; segment++) {
        const auto &seg = record.group.segments[segment];
        if (seg.endpoint_idx >= server_count) return false;
        if (seg.addr == ::FarLib::allocator::remote::InvalidRemoteAddr) {
            return false;
        }
        if (seg.slot_size != record.slot_size) return false;
        // owns_slot above already validates every source pointer and the
        // active lease. Sources are exactly those segment bases, and the
        // length was bounded by staging.slot_size before that validation.
        // Repeating owns_buffer six times revalidated the same lease 7x.
    }
    return true;
}

}  // namespace FarLib::cache::ec_batch

namespace FarLib::cache {

// There is no cache-wide stage guard: each EvictBufferSet has one producer.

// Large split write batching is deliberately opt-in so the old per-group
// submission remains the exact default/control. Values in (1, 32] are
// accepted for diagnostics; the experiment uses 0 and 32.
inline size_t ec_split_write_batch_limit() {
    if (!ec_split::mg_optimizations_enabled) return 0;
    static const size_t limit = [] {
        const char *value = std::getenv("FARLIB_EC_SPLIT_WRITE_BATCH");
        if (value == nullptr || *value == '\0') return size_t{0};
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0) return size_t{0};
        if (parsed > 32) return size_t{32};
        return static_cast<size_t>(parsed);
    }();
    return limit;
}

// Fresh split-group allocation batching is deliberately narrower than the
// write batch: only the bounded values 0 and 8 are accepted, and the value is
// sampled once so the scalar default remains the exact control path.
inline size_t ec_split_group_batch_limit() {
    if (!ec_split::mg_optimizations_enabled) return 0;
    static const size_t limit = [] {
        if (::FarLib::get_config().ft_background_rebuild) return size_t{0};
        const char *value = std::getenv("FARLIB_EC_SPLIT_GROUP_BATCH");
        if (value == nullptr || *value == '\0') return size_t{0};
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || (parsed != 0 && parsed != 8))
            return size_t{0};
        return static_cast<size_t>(parsed);
    }();
    return limit;
}

// ---------------------------------------------------------------------------
// ec_batch diagnostics (observability only).
//
// One relaxed atomic increment per event that already existed, one gated print
// per teardown step.  No counter below is read by a decision, no return value
// and no ordering changes anywhere; ft_method=none never reaches the printing
// helpers (all of them are gated on get_config().is_ec_batch_mode()), so its
// output stays byte-identical.
// ---------------------------------------------------------------------------
inline void ConcurrentArrayCache::ec_batch_diag_note_status(
    ec_batch::EcBatchStatus status) {
    ec_diag_add_obj_calls_.fetch_add(1, std::memory_order_relaxed);
    switch (status) {
        case ec_batch::EcBatchStatus::kOk:
            ec_diag_add_obj_ok_.fetch_add(1, std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kGroupFull:
            ec_diag_status_group_full_.fetch_add(1, std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kStagingExhausted:
            ec_diag_status_staging_exhausted_.fetch_add(
                1, std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kPendingQueueFull:
            ec_diag_status_pending_queue_full_.fetch_add(
                1, std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kManagerRejected:
            ec_diag_status_manager_rejected_.fetch_add(
                1, std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kEncodeRejected:
            ec_diag_status_encode_rejected_.fetch_add(
                1, std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kSealRejected:
            ec_diag_status_seal_rejected_.fetch_add(1,
                                                    std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kObjectTooLarge:
            ec_diag_status_object_too_large_.fetch_add(
                1, std::memory_order_relaxed);
            break;
        case ec_batch::EcBatchStatus::kInvalidArgument:
            ec_diag_status_invalid_argument_.fetch_add(
                1, std::memory_order_relaxed);
            break;
    }
}

inline uint64_t ConcurrentArrayCache::ec_batch_diag_pending_groups() {
    // No shared pending queue exists. Worker-private queues are drained before
    // their owner returns; active_worker_buffers is reported separately.
    return 0;
}

// "INFO: ec_batch teardown step=<name> pending_groups=N in_flight_tokens=M
//  staging_in_use=K ..." - one flushed line per teardown step, only in
// ft_method=ec_batch.
inline void ConcurrentArrayCache::ec_batch_diag_step(const char *step) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    const uint64_t sealed_groups =
        ec_worker_sealed_groups_.load(std::memory_order_relaxed) +
        ec_split_groups_staged_.load(std::memory_order_relaxed);
    std::cout << "INFO: ec_batch teardown step=" << step
              << " pending_groups=" << ec_batch_diag_pending_groups()
              << " in_flight_tokens=" << ec_batch_tokens_.in_use() + (ec_direct_bank_ ? ec_direct_bank_->in_use() : 0)
              << " staging_in_use=" << ec_staging_pool_.in_use() + ec_split_buffers_.in_use()
              << " sealed_groups=" << sealed_groups
              << " posted_groups="
              << ec_batch_groups_posted_.load(std::memory_order_relaxed)
              << " completed_groups="
              << ec_batch_groups_completed_.load(std::memory_order_relaxed)
              << " stage_called="
              << ec_diag_stage_called_.load(std::memory_order_relaxed)
              << " stage_failed="
              << ec_diag_stage_failed_.load(std::memory_order_relaxed)
              << " active_worker_buffers=" << ec_worker_active_buffers_.load(std::memory_order_relaxed)
              << std::endl;
}
// ---------------------------------------------------------------------------
// ft_method=ec_batch teardown of a stuck write round.
//
// A group's six segment CQEs are reaped by whichever thread happens to poll
// the (client, QP, endpoint) CQ they land in, and every one of them is reaped
// by handle_rdma_write_complete() -> handle_ec_batch_write_complete().  A
// group whose CQEs nobody polls any more (the worker that posted it stopped
// polling, or the CQ is not the one of QP index 0 that full_checker() walks)
// stays in the token table forever: its four objects keep the write reference
// taken at eviction time, and deallocate() of such an object spins on
// ref_cnt != 0 without any completion to come.
//
// Two steps, both inert unless ft_method=ec_batch:
//   1. drain_ec_batch_in_flight_groups_for_shutdown(): a *bounded* drain that
//      polls every CQ of every client (all QP indices, not just 0) while the
//      workers still exist, so a group that is merely unobserved gets its real
//      completions.
//   2. settle_ec_batch_in_flight_groups(): only once the cache is quiesced,
//      write off what can never complete - by the group release path.
// ---------------------------------------------------------------------------

// Bounds of the shutdown drain (whichever comes first): it is a rescue, not a
// wait loop, and it must not delay a teardown that has nothing left to do.
inline constexpr uint64_t kEcBatchShutdownDrainMaxPasses = 4096;
inline constexpr uint64_t kEcBatchShutdownDrainTimeoutMs = 2000;

// Groups whose completion is still outstanding: posted tokens in flight plus
// sealed groups that have not been posted yet.  Diagnostics/decisions only.
inline size_t ConcurrentArrayCache::ec_batch_in_flight_group_count() {
    if (!::FarLib::get_config().is_ec_batch_mode()) return 0;
    size_t in_flight = ec_batch_tokens_.in_use() + (ec_direct_bank_ ? ec_direct_bank_->in_use() : 0);
    // Active owners may still hold private, unpublished groups. Their normal
    // batch-tail flush happens before the master joins during quiescence.
    in_flight += ec_worker_active_buffers_.load(std::memory_order_relaxed);
    return in_flight;
}

// One pass over every CQ this client set owns: all clients, every QP index of
// every endpoint.  Unlike full_checker() (QP index 0 only) this reaches the
// CQ of the QP a worker actually posted the six writes on.
inline size_t ConcurrentArrayCache::ec_batch_poll_all_cqs_once() {
    size_t completions = 0;
    const size_t client_count = rdma::get_client_count();
    for (size_t client_idx = 0; client_idx < client_count; ++client_idx) {
        auto *client = rdma::get_client(client_idx);
        if (client == nullptr) continue;
        const size_t qp_count = client->get_local_qp_count();
        const size_t endpoint_count = client->get_endpoint_count();
        for (size_t qp_idx = 0; qp_idx < qp_count; ++qp_idx) {
            for (size_t endpoint = 0; endpoint < endpoint_count; ++endpoint) {
                completions += check_cq_idx_with_client_idx_endpoint(
                    qp_idx, client_idx, endpoint);
            }
        }
    }
    return completions;
}

inline void
ConcurrentArrayCache::drain_ec_batch_in_flight_groups_for_shutdown() {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    const auto start = std::chrono::steady_clock::now();
    flush_ec_batch_groups(rdma::thread_info.thread_id);
    ec_batch_diag_step("ec_batch_shutdown_drain_begin");
    const auto deadline = start + std::chrono::milliseconds(
                                       kEcBatchShutdownDrainTimeoutMs);
    uint64_t passes = 0;
    uint64_t completions = 0;
    while (ec_batch_in_flight_group_count() != 0 &&
           passes < kEcBatchShutdownDrainMaxPasses &&
           std::chrono::steady_clock::now() < deadline) {
        const size_t in_flight_before = ec_batch_in_flight_group_count();
        completions += ec_batch_poll_all_cqs_once();
        ++passes;
        if (ec_batch_in_flight_group_count() == in_flight_before &&
            (passes & 15ull) == 0) {
            uthread::yield();
        }
    }
    const int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    std::cout << "INFO: ec_batch shutdown drain passes=" << passes
              << " completions=" << completions
              << " elapsed_ms=" << elapsed_ms
              << " remaining_in_flight=" << ec_batch_in_flight_group_count()
              << std::endl;
    ec_batch_diag_step("ec_batch_shutdown_drain_end");
}

// Shutdown integrity guard: stopping workers does not stop DMA. Groups must
// drain through real success/error completions. Never synthesize successful
// completion or recycle staging while the NIC might still use it.
inline size_t ConcurrentArrayCache::settle_ec_batch_in_flight_groups(
    const char *reason) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return 0;
    if (working.load(std::memory_order_acquire)) return 0;
    if (working.load(std::memory_order_acquire)) return 0;
    const size_t remaining = ec_batch_in_flight_group_count();
    if (remaining == 0) return 0;
    ec_batch_diag_step(reason);
    std::cerr << "ERROR: ec_batch shutdown still owns " << remaining
              << " groups after bounded drain; reason=" << reason << std::endl;
    ERROR("ec_batch: refusing to fabricate write completions during shutdown");
}


// The whole counter set, printed once per call site; only in ft_method=ec_batch.
inline void ConcurrentArrayCache::ec_batch_diag_report(const char *where) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    auto ld = [](const auto &value) {
        return value.load(std::memory_order_relaxed);
    };
    std::cout << "ec_direct diag [" << where << "] enabled=" << (ec_direct_bank_ ? 1 : 0)
              << " borrowed_data_bytes=" << ld(ec_direct_data_bytes_)
              << " copied_data_bytes=" << ld(ec_copy_data_bytes_)
              << " groups_posted=" << ld(ec_direct_groups_)
              << " mutable_waits=" << ld(ec_direct_write_waits_)
              << " work_write_posts=" << ld(ec_direct_work_write_posts_)
              << " work_write_bytes=" << ld(ec_direct_work_write_bytes_)
              << " bank_init_count=" << ec_persistent_owner_count_
              << " bank_in_use=" << (ec_direct_bank_ ? ec_direct_bank_->in_use() : 0)
              << std::endl;
    if (ec_split::mg_optimizations_enabled) {
    std::cout << "ec_split direct_write [" << where << "] enabled="
              << (ec_split::direct_write_enabled() ? 1 : 0)
              << " staged="
              << ec_split_direct_write_staged_.load(std::memory_order_relaxed)
              << " completed="
              << ec_split_direct_write_completed_.load(std::memory_order_relaxed)
              << " logical_bytes="
              << ec_split_direct_write_bytes_.load(std::memory_order_relaxed)
              << " borrowed_in_use="
              << ec_split_direct_write_borrowed_.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "ec_split batch [" << where << "] limit="
              << ec_split_write_batch_limit()
              << " flushes=" << ec_split_batch_flushes_.load(std::memory_order_relaxed)
              << " groups=" << ec_split_batch_groups_.load(std::memory_order_relaxed)
              << " max_groups=" << ec_split_batch_max_groups_.load(std::memory_order_relaxed)
              << " endpoint_chains="
              << ec_split_batch_endpoint_chains_.load(std::memory_order_relaxed)
              << " accepted_wrs="
              << ec_split_batch_accepted_wrs_.load(std::memory_order_relaxed)
              << std::endl;
    std::cout << "ec_split group_batch [" << where << "] limit="
              << ec_split_group_batch_limit()
              << " refills="
              << ld(ec_split_group_batch_refills_)
              << " groups=" << ld(ec_split_group_batch_groups_)
              << " consumed=" << ld(ec_split_group_batch_consumed_)
              << " released=" << ld(ec_split_group_batch_released_)
              << std::endl;
    }
    const uint64_t sealed_groups =
        ec_worker_sealed_groups_.load(std::memory_order_relaxed) +
        ec_split_groups_staged_.load(std::memory_order_relaxed);
    const uint64_t alloc_group_ok =
        ec_worker_alloc_ok_.load(std::memory_order_relaxed);
    const uint64_t alloc_group_fail =
        ec_worker_alloc_failed_.load(std::memory_order_relaxed);
    std::cout << "ec_batch diag [" << where
              << "] candidate: ec_candidate_checked="
              << ld(ec_candidate_checked_)
              << " ec_candidate_true=" << ld(ec_candidate_true_)
              << " stage_called=" << ld(ec_diag_stage_called_)
              << " stage_ok=" << ld(ec_diag_stage_ok_)
              << " stage_failed=" << ld(ec_diag_stage_failed_)
              << " stage_fail{not_ready="
              << ld(ec_diag_stage_fail_not_ready_)
              << " no_client=" << ld(ec_diag_stage_fail_no_client_)
              << " bad_segment_addr="
              << ld(ec_diag_stage_fail_bad_segment_addr_)
              << " rounds_exhausted="
              << ld(ec_diag_stage_fail_rounds_exhausted_) << "}" << std::endl;
    std::cout << "ec_batch diag [" << where
              << "] group: allocate_slot_group_ok=" << alloc_group_ok
              << " allocate_slot_group_failed=" << alloc_group_fail
              << " (allocate_slot_group() returns bool only: a refusal shows up "
                 "as status manager_rejected, a refusal before the allocator as "
                 "staging_exhausted)"
              << " add_object_calls=" << ld(ec_diag_add_obj_calls_)
              << " status{group_full=" << ld(ec_diag_status_group_full_)
              << " staging_exhausted=" << ld(ec_diag_status_staging_exhausted_)
              << " pending_queue_full="
              << ld(ec_diag_status_pending_queue_full_)
              << " manager_rejected=" << ld(ec_diag_status_manager_rejected_)
              << " encode_rejected=" << ld(ec_diag_status_encode_rejected_)
              << " seal_rejected=" << ld(ec_diag_status_seal_rejected_)
              << " object_too_large=" << ld(ec_diag_status_object_too_large_)
              << " invalid_argument=" << ld(ec_diag_status_invalid_argument_)
              << "}" << std::endl;
    // The builder is the only owner of behavior-keyed small-object groups.
    // Keep this diagnostic behind the ec_batch gate above: ft_method=none must
    // not pay for per-key reads on its hot path. Large split groups bypass the
    // builder and remain represented by the aggregate split counter below.
    const auto sealed_for_behavior_group = [this](uint32_t behavior_group) {
        return ec_worker_sealed_by_group_[behavior_group].load(std::memory_order_relaxed);
    };
    std::cout << "ec_batch diag [" << where
              << "] sealed_by_behavior_group="
              << "0:" << sealed_for_behavior_group(0)
              << ",1:" << sealed_for_behavior_group(1)
              << ",2:" << sealed_for_behavior_group(2)
              << ",3:" << sealed_for_behavior_group(3)
              << ",4:" << sealed_for_behavior_group(4)
              << ",5:" << sealed_for_behavior_group(5)
              << " split_groups="
              << ec_split_groups_staged_.load(std::memory_order_relaxed)
              << " (small-object keys; split groups separate)" << std::endl;
    std::cout << "ec_batch diag [" << where << "] post: sealed_groups="
              << sealed_groups << " flush_calls=" << ld(ec_diag_flush_calls_)
              << " flush_sealed_ok=" << ld(ec_diag_flush_sealed_ok_)
              << " posted_groups=" << ld(ec_batch_groups_posted_)
              << " post_rejects=" << ld(ec_batch_post_rejects_)
              << " post_failed{no_staging_or_mode="
              << ld(ec_diag_post_fail_no_mode_or_staging_)
              << " not_postable=" << ld(ec_diag_post_fail_not_postable_)
              << " no_client=" << ld(ec_diag_post_fail_no_client_)
              << " no_token=" << ld(ec_diag_post_fail_no_token_)
              << " pending_pop_post_failed="
              << ld(ec_diag_post_pending_failed_)
              << " pending_token_backpressure="
              << ld(ec_diag_pending_token_backpressure_)
              << " post_write_retries=" << ld(ec_diag_post_write_retries_)
              << "}" << std::endl;
    std::cout << "ec_batch diag [" << where
              << "] live: pending_groups=" << ec_batch_diag_pending_groups()
              << " in_flight_tokens=" << ec_batch_tokens_.in_use() + (ec_direct_bank_ ? ec_direct_bank_->in_use() : 0)
              << " token_table_capacity="
              << ec_batch::EcBatchTokenTable::capacity()
              << " staging_in_use=" << ec_staging_pool_.in_use() + ec_split_buffers_.in_use()
              << " completed_segments=" << ld(ec_diag_complete_cqes_)
              << " completed_groups=" << ld(ec_diag_complete_last_)
              << " groups_completed_counter="
              << ld(ec_batch_groups_completed_)
              << " complete_not_last_or_stale="
              << ld(ec_diag_complete_not_last_)
              << " staging_release_failures="
              << ld(ec_batch_staging_release_failures_) << std::endl;
}

// Reserves the resident ec_batch staging range inside the client MR and returns
// the heap size the object allocator may use.  The range is taken from the tail
// of the registered buffer (rounded up to whole allocator regions) and is *not*
// handed to ::FarLib::allocator::global_heap::register_heap(), so the cache can
// never distribute it as object memory.  The lkey of that memory is the client
// MR's lkey, i.e. exactly what build_send_wr() puts into the sge of the six
// writes.  Returns local_buf_size unchanged (staging disabled) unless
// ft_method=ec_batch.
inline size_t ConcurrentArrayCache::init_ec_batch_staging(void *local_buf,
                                                          size_t local_buf_size) {
    ec_staging_reserved_bytes_ = 0;
    ec_staging_slot_size_ = 0;
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode()) return local_buf_size;
    if (local_buf == nullptr || local_buf_size == 0) return local_buf_size;

    // Largest slot the group path can ask for: the bin that holds the small
    // object cutoff.  Objects are only group-allocated while they are small
    // (ft_small_object()), so this covers every group; the pool's hard ceiling
    // still applies.
    size_t slot_size = ::FarLib::allocator::MaxBinSize;
    const size_t cutoff = config.behavior_group
        ? rdma::Configure::behavior_group_max_whole_object_bytes
        : config.ft_small_object_cutoff;
    if (cutoff > 0 && cutoff < ::FarLib::allocator::MaxBinSize) {
        slot_size = ::FarLib::allocator::get_bin_size(
            ::FarLib::allocator::bin_from_wsize(
                ::FarLib::allocator::wsize_from_size(cutoff)));
    }
    if (slot_size > ec_batch::kEcBatchStagingMaxSlotSize) {
        slot_size = ec_batch::kEcBatchStagingMaxSlotSize;
    }
    const size_t depth = ec_batch::kEcBatchStagingDefaultDepth;
    const size_t required =
        ec_batch::EcStagingPool::required_bytes(slot_size, depth);
    const size_t region = ::FarLib::allocator::RegionSize;
    const size_t reserve = ((required + region - 1) / region) * region;
    if (reserve == 0 || reserve + region > local_buf_size) {
        std::cerr << "ec_batch staging: client MR too small (bytes="
                  << local_buf_size << ", need=" << reserve
                  << "); group writes disabled" << std::endl;
        return local_buf_size;
    }
    auto *pool_base = static_cast<uint8_t *>(local_buf) +
                      (local_buf_size - reserve);
    if (!ec_staging_pool_.init(pool_base, required, slot_size, depth)) {
        std::cerr << "ec_batch staging: pool init failed: "
                  << ec_staging_pool_.error() << std::endl;
        return local_buf_size;
    }
    if (auto *control = rdma::ClientControl::get_default()) {
        const auto *mr_base =
            static_cast<const uint8_t *>(control->get_buffer());
        const auto *pool_end = static_cast<const uint8_t *>(pool_base) + required;
        ASSERT(pool_base >= mr_base);
        ASSERT(pool_end <= mr_base + config.client_buffer_size);
    }
    ec_staging_reserved_bytes_ = reserve;
    ec_staging_slot_size_ = slot_size;
    std::cout << "ec_batch staging: reserved=" << reserve
              << " usable=" << required << " slot_size=" << slot_size
              << " depth=" << depth
              << " base=" << static_cast<void *>(pool_base)
              << " heap_bytes=" << (local_buf_size - reserve) << std::endl;
    std::cout << "ec_batch worker_private=1 object_batch=" << EvictBatchSize
              << " endpoint_wr_batch=" << EvictBatchSize
              << " tokens_per_owner=" << ec_batch::EcBatchTokenTable::owner_capacity()
              << std::endl;
    return local_buf_size - reserve;
}

// Posts the six single-sided RDMA WRITEs of one sealed group: four data slots
// and the two parity slots, each segment to its own endpoint, each with an
// ec_batch wr_id that carries (token, segment).  Either all six are posted and
// the group is registered in the token table, or nothing is posted at all (no
// token acquired, no WR built) - never a half-posted group reported as success.
inline bool ConcurrentArrayCache::post_ec_batch_group(
    const ec_batch::EcGroupSendRecord &record, size_t client_idx, size_t qp_idx,
    bool *token_backpressure) {
    if (token_backpressure != nullptr) *token_backpressure = false;
    if (!ec_batch::EcBatchTokenTable::valid_owner(client_idx)) {
        ERROR("ec_batch: RDMA client index exceeds token owner capacity");
    }
    const auto &config = ::FarLib::get_config();
    const bool staging_valid = (record.split_object || record.size_class_staging)
        ? ec_split_buffers_.valid() : ec_staging_pool_.valid();
    if (!config.is_ec_batch_mode() || !staging_valid) {
        ec_batch_post_rejects_.fetch_add(1, std::memory_order_relaxed);
        ec_diag_post_fail_no_mode_or_staging_.fetch_add(1,
                                                        std::memory_order_relaxed);
        return false;
    }
    const bool postable = (record.split_object || record.size_class_staging)
        ? ec_batch::ec_batch_record_is_postable(record, ec_split_buffers_, config.server_count)
        : ec_batch::ec_batch_record_is_postable(record, ec_staging_pool_, config.server_count);
    if (!postable) {
        ec_batch_post_rejects_.fetch_add(1, std::memory_order_relaxed);
        ec_diag_post_fail_not_postable_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto *client = rdma::get_client(client_idx);
    if (client == nullptr) {
        ec_batch_post_rejects_.fetch_add(1, std::memory_order_relaxed);
        ec_diag_post_fail_no_client_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    uint64_t token_id = 0;
    ec_batch::EcBatchToken *token = nullptr;
    if (!ec_batch_tokens_.acquire(record, &token_id, &token, client_idx)) {
        if (token_backpressure != nullptr) *token_backpressure = true;
        ec_batch_post_rejects_.fetch_add(1, std::memory_order_relaxed);
        ec_diag_post_fail_no_token_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    ASSERT(token != nullptr);
    for (size_t segment = 0; segment < ec_batch::kEcBatchSegmentsPerGroup;
         segment++) {
        const auto &seg = record.group.segments[segment];
        // The group's segment attribution and the address mapping must agree;
        // map_remote_addr() then gives the per-endpoint offset of the write.
        const auto mapped = config.map_remote_addr(seg.addr);
        ASSERT(mapped.first == seg.endpoint_idx);
        ASSERT(mapped.second + seg.slot_size <= config.server_buffer_size);
        const uint64_t wr_id = ec_batch::encode_ec_batch_wr_id(
            token_id, static_cast<uint8_t>(segment));
        void *src = ec_batch::ec_batch_record_source(record, segment);
        if (ec_recovery_endpoint_is_dead(seg.endpoint_idx)) {
            // A group allocated before failure keeps its original addresses.
            // Resolve the unposted segment as failed; the other five writes
            // still form a recoverable codeword. Never reuse staging until
            // every actually posted write has a hardware completion.
            handle_ec_batch_write_complete(wr_id, false);
            continue;
        }
        // A momentarily full send queue is backpressure, not failure: drain the
        // completions of this endpoint (including the segments already posted
        // for this group) and retry the same segment until it is accepted.
        const uint32_t *lkey_override =
            ec_batch_record_source_uses_staging(
                record, segment, ec_staging_pool_.shared_buffers_bound())
                ? &record.staging.lkey
                : nullptr;
        const uint32_t source_bytes = ec_batch_record_source_bytes(record);
        while (!client->post_write(mapped.second, src, source_bytes, wr_id, 0,
                                   qp_idx, seg.endpoint_idx, lkey_override)) {
            (void)this->check_cq_idx_with_client_idx_endpoint(
                qp_idx, client_idx, seg.endpoint_idx);
            ec_diag_post_write_retries_.fetch_add(1, std::memory_order_relaxed);
            if (ec_recovery_endpoint_is_dead(seg.endpoint_idx)) {
                handle_ec_batch_write_complete(wr_id, false);
                break;
            }
        }
    }
    ec_batch_groups_posted_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Only the owning eviction invocation consumes this builder. A whole batch is
// prepared before posting: every record already contains published addresses,
// and every token owns its staging lease until six real completions arrive.
inline void ConcurrentArrayCache::account_ec_worker_builder(EvictBufferSet &buffers) {
    if (buffers.direct_builder) {
        auto &b = *buffers.direct_builder;
        auto delta = [](auto &total, uint64_t &prior, uint64_t value) {
            if (value != prior) total.fetch_add(value - prior, std::memory_order_relaxed);
            prior = value;
        };
        delta(ec_worker_sealed_groups_, buffers.direct_reported_sealed, b.sealed_count());
        delta(ec_worker_alloc_ok_, buffers.direct_reported_alloc_ok, b.group_alloc_ok_count());
        delta(ec_worker_alloc_failed_, buffers.direct_reported_alloc_failed, b.group_alloc_fail_count());
        for (size_t k = 0; k < ec_batch::kEcBatchBehaviorGroupCount; ++k)
            delta(ec_worker_sealed_by_group_[k], buffers.direct_reported_by_group[k], b.sealed_count_for_behavior_group(k));
        const auto objects = b.object_count() - buffers.direct_reported_objects;
        // Publish per-worker totals once per batch, not four atomics per object.
        if (objects) {
            ec_diag_stage_called_.fetch_add(objects, std::memory_order_relaxed);
            ec_diag_stage_ok_.fetch_add(objects, std::memory_order_relaxed);
            ec_diag_add_obj_calls_.fetch_add(objects, std::memory_order_relaxed);
            ec_diag_add_obj_ok_.fetch_add(objects, std::memory_order_relaxed);
            buffers.direct_reported_objects = b.object_count();
        }
        delta(ec_direct_data_bytes_, buffers.direct_reported_bytes, b.data_bytes());
    }
    auto account = [&](auto &builder) {
    auto add_delta = [](auto &total, uint64_t &previous, uint64_t value) {
        if (value != previous) total.fetch_add(value - previous, std::memory_order_relaxed);
        previous = value;
    };
    add_delta(ec_worker_sealed_groups_, buffers.ec_reported_sealed, builder.sealed_count());
    add_delta(ec_worker_alloc_ok_, buffers.ec_reported_alloc_ok, builder.group_alloc_ok_count());
    add_delta(ec_worker_alloc_failed_, buffers.ec_reported_alloc_failed, builder.group_alloc_fail_count());
    for (size_t key = 0; key < ec_batch::kEcBatchBehaviorGroupCount; ++key) {
        add_delta(ec_worker_sealed_by_group_[key], buffers.ec_reported_by_group[key],
                  builder.sealed_count_for_behavior_group(key));
    }
    };
    if (buffers.size_class_builder) account(*buffers.size_class_builder);
    else if (buffers.ec_builder) account(*buffers.ec_builder);
}

inline size_t ConcurrentArrayCache::post_ec_batch_pending(
    EvictBufferSet &buffers, size_t client_idx, size_t qp_idx) {
    size_t groups = 0;
    if (ec_split::mg_optimizations_enabled && buffers.ec_split_builder)
        groups += post_ec_split_pending(buffers, client_idx, qp_idx);
    if (buffers.size_class_builder)
        return groups + post_ec_pending_impl(buffers, *buffers.size_class_builder,
                                             ec_split_buffers_, client_idx, qp_idx);
    if (buffers.ec_builder)
        return groups + post_ec_pending_impl(buffers, *buffers.ec_builder,
                                             ec_staging_pool_, client_idx, qp_idx);
    return groups;
}

template <class Builder, class Pool>
inline size_t ConcurrentArrayCache::post_ec_pending_impl(
    EvictBufferSet &buffers, Builder &builder, Pool &pool,
    size_t client_idx, size_t qp_idx, bool split_batch_observe) {
    profile::evict_breakdown::Scope prepare_scope(
        buffers.breakdown, profile::evict_breakdown::Stage::SendPrepare);
    if (!ec_batch::EcBatchTokenTable::valid_owner(client_idx)) {
        ERROR("ec_batch: RDMA client index exceeds token owner capacity");
    }
    auto *client = rdma::get_client(client_idx);
    const auto &config = ::FarLib::get_config();
    if (!buffers.ec_writes) {
        buffers.ec_writes.reset(new EvictBufferSet::EcEndpointWrites[buffers.server_count]);
    }
    for (size_t ep = 0; ep < buffers.server_count; ++ep) {
        ASSERT(buffers.ec_writes[ep].count == 0);
    }
    size_t groups = 0;
    while (groups < EvictBatchSize) {
        const auto *pending_record = builder.peek_private_pending();
        if (pending_record == nullptr) break;
        const auto &record = *pending_record;
        if (!ec_batch::ec_batch_record_is_postable(record, pool, config.server_count)) {
            ERROR("ec_batch: private worker produced an invalid send record");
        }
        // Keep endpoint chains bounded even if a future layout has more than
        // one segment on an endpoint. No token/queue ownership changes yet.
        bool fits = true;
        for (size_t i = 0; i < record.group.segments.size(); ++i) {
            const auto &seg = record.group.segments[i];
            if (ec_recovery_endpoint_is_dead(seg.endpoint_idx)) continue;
            size_t same_endpoint = 1;
            for (size_t j = 0; j < i; ++j)
                same_endpoint += record.group.segments[j].endpoint_idx == seg.endpoint_idx;
            if (buffers.ec_writes[seg.endpoint_idx].count + same_endpoint > EvictBatchSize)
                fits = false;
        }
        if (!fits) break;
        uint64_t token_id = 0;
        ec_batch::EcBatchToken *token = nullptr;
        if (!ec_batch_tokens_.acquire(record, &token_id, &token, client_idx)) break;
        ASSERT(token != nullptr);
        for (size_t segment = 0; segment < ec_batch::kEcBatchSegmentsPerGroup; ++segment) {
            const auto &seg = record.group.segments[segment];
            const uint64_t wr_id = ec_batch::encode_ec_batch_wr_id(token_id, segment);
            if (ec_recovery_endpoint_is_dead(seg.endpoint_idx)) {
                handle_ec_batch_write_complete(wr_id, false);
                continue;
            }
            auto &batch = buffers.ec_writes[seg.endpoint_idx];
            const size_t i = batch.count++;
            const auto mapped = config.map_remote_addr(seg.addr);
            ASSERT(mapped.first == seg.endpoint_idx);
            const uint32_t source_bytes =
                ec_batch::ec_batch_record_source_bytes(record);
            client->build_send_wr(batch.wrs[i], batch.sges[i], mapped.second,
                ec_batch::ec_batch_record_source(record, segment), source_bytes,
                wr_id, true, IBV_WR_RDMA_WRITE, seg.endpoint_idx);
            if (ec_batch::ec_batch_record_source_uses_staging(
                    record, segment, ec_staging_pool_.shared_buffers_bound()))
                batch.sges[i].lkey = record.staging.lkey;
            if (i != 0) batch.wrs[i - 1].next = &batch.wrs[i];
        }
        if (!builder.commit_pending(record.sequence)) {
            ERROR("ec_batch: private sender lost queue ownership");
        }
        ++groups;
    }
    for (size_t ep = 0; ep < buffers.server_count; ++ep) {
        auto &batch = buffers.ec_writes[ep];
        if (batch.count == 0) continue;
        if (split_batch_observe)
            ec_split_batch_endpoint_chains_.fetch_add(1, std::memory_order_relaxed);
        ibv_send_wr *remaining = &batch.wrs[0];
        size_t accepted_count = batch.count;
        // On ENOMEM the provider returns the first unaccepted WR. Never
        // resubmit the accepted prefix (its CQEs may already be processed).
        auto post_once = [&] {
            profile::evict_breakdown::Scope post_scope(
                buffers.breakdown, profile::evict_breakdown::Stage::PostSend);
            return client->post_writes(remaining, &remaining, qp_idx, ep);
        };
        while (!post_once()) {
            ec_diag_post_write_retries_.fetch_add(1, std::memory_order_relaxed);
            {
                profile::evict_breakdown::Scope poll_scope(
                    buffers.breakdown, profile::evict_breakdown::Stage::CqProcess);
                (void)check_cq_idx_with_client_idx_endpoint(qp_idx, client_idx, ep);
            }
            if (ec_recovery_endpoint_is_dead(ep)) {
                for (auto *wr = remaining; wr != nullptr; wr = wr->next) {
                    handle_ec_batch_write_complete(wr->wr_id, false);
                    --accepted_count;
                }
                break;
            }
        }
        size_t bytes = 0;
        for (size_t i = 0; i < accepted_count; ++i) {
            bytes += batch.sges[i].length;
            profile::count_rdma_write_post(batch.sges[i].length);
        }
        if (split_batch_observe)
            ec_split_batch_accepted_wrs_.fetch_add(
                accepted_count, std::memory_order_relaxed);
        profile::count_evacuation_bytes(bytes);
        profile::count_evac_flush(accepted_count, accepted_count == EvictBatchSize);
        batch.count = 0;
    }
    if (groups != 0) ec_batch_groups_posted_.fetch_add(groups, std::memory_order_relaxed);
    account_ec_worker_builder(buffers);
    return groups;
}

inline size_t ConcurrentArrayCache::post_ec_split_pending(
    EvictBufferSet &buffers, size_t client_idx, size_t qp_idx) {
    if (!buffers.ec_split_builder ||
        buffers.ec_split_builder->pending_count() == 0)
        return 0;
    const size_t groups = post_ec_pending_impl(
        buffers, *buffers.ec_split_builder, ec_split_buffers_, client_idx, qp_idx,
        true);
    if (groups != 0) {
        ec_split_batch_flushes_.fetch_add(1, std::memory_order_relaxed);
        ec_split_batch_groups_.fetch_add(groups, std::memory_order_relaxed);
        uint64_t previous = ec_split_batch_max_groups_.load(
            std::memory_order_relaxed);
        while (previous < groups &&
               !ec_split_batch_max_groups_.compare_exchange_weak(
                   previous, groups, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }
    return groups;
}

// One segment of a group completed.  The last of the six completions completes
// the group: every live object of the group is handed back through the ordinary
// EVICTING -> REMOTE release, then the staging slot and the token are returned.
// Runs on whatever thread reaped the CQE (the eviction threads or
// full_checker()), so all bookkeeping lives in the shared token table.
inline void ConcurrentArrayCache::handle_ec_batch_write_complete(uint64_t wr_id,
                                                                bool success) {
    const uint64_t token_id = ec_batch::ec_batch_wr_id_token(wr_id);
    const uint8_t segment = ec_batch::ec_batch_wr_id_segment(wr_id);
    if (ec_batch::EcDirectWriteBank::is_direct_token(token_id)) {
        complete_ec_direct_write(token_id, segment, success);
        return;
    }
    ec_diag_complete_cqes_.fetch_add(1, std::memory_order_relaxed);
    ec_batch::EcBatchToken *token =
        ec_batch_tokens_.complete_segment(token_id, segment, success);
    if (token == nullptr) {
        // Not the last segment of a live group: either a segment of a group
        // that is still in flight / a duplicate CQE / a stale CQE of an
        // already released token.  Nothing to do.
        ec_diag_complete_not_last_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const ec_batch::EcGroupSendRecord &record = token->record;
    ASSERT(token->pending == 0);
    const unsigned completed_ok = token->durable_segment_count();
    if (!token->recoverable()) {
        ERROR("ec_batch: fewer than four durable segments; refusing false write completion");
    }
    const int standby = ::FarLib::get_config().ft_standby_endpoint;
    if (standby >= 0) {
        for (size_t s = 0; s < record.group.segments.size(); ++s) {
            if (record.group.segments[s].endpoint_idx ==
                    static_cast<size_t>(standby) &&
                (token->failed & (1u << s)) == 0) {
                static std::atomic<bool> first_standby_write_reported{false};
                if (!first_standby_write_reported.exchange(true)) {
                    std::ostringstream line;
                    line << "INFO: ec_batch standby write_completed endpoint="
                         << standby << " token=" << token_id
                         << " bytes=" << record.group.segments[s].slot_size;
                    std::cout << line.str() << std::endl;
                }
                break;
            }
        }
    }
    if (token->failed != 0) {
        std::cout << "INFO: ec_batch degraded_write token=" << token_id
                  << " completed_ok=" << completed_ok
                  << " failed_mask=" << static_cast<unsigned>(token->failed)
                  << std::endl;
    }
    ec_diag_complete_last_.fetch_add(1, std::memory_order_relaxed);
    if (::FarLib::get_config().ft_incremental_update)
        remote_allocator.small_object_stripe_manager().mark_slot_group_durable(record.group.id);
    for (size_t i = 0; i < ec_batch::kEcBatchDataSlots; i++) {
        if (record.objects[i] != nullptr) {
            if (record.source_mode == ec_batch::EcGroupSourceMode::kSplitDirect) {
                auto *block = static_cast<::FarLib::allocator::BlockHead *>(
                    const_cast<void *>(record.objects[i])) - 1;
                ::FarLib::allocator::release_ec_write_source(block);
                ec_split_direct_write_borrowed_.fetch_sub(
                    1, std::memory_order_relaxed);
                ec_split_direct_write_completed_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            complete_evict_writeback(const_cast<void *>(record.objects[i]));
        }
    }
    const bool staging_released = (record.split_object || record.size_class_staging)
        ? ec_split_buffers_.release(record.staging)
        : ec_staging_pool_.release(record.staging);
    if (!staging_released) {
        ec_batch_staging_release_failures_.fetch_add(1,
                                                     std::memory_order_relaxed);
    }
    ec_batch_tokens_.release(token_id);
    ec_batch_groups_completed_.fetch_add(1, std::memory_order_relaxed);
}


// ---------------------------------------------------------------------------
// Group staging for the eviction path (try_evict()).
//
// A small object that is evicted is placed into a slot group and gets the
// address of its own data segment as the remote location; the group is encoded
// (RS(4,2)) and posted in endpoint chains by its private EvictBufferSet.
// Only the owning worker can consume its queue, and it publishes each object's
// remote address before calling the sender for the group that contains it:
// a completion (or a fetch of the object once it is REMOTE) can never observe a
// stale remote address.
// ---------------------------------------------------------------------------

// Private builders are created by, and remain owned by, the logical eviction
// invocation. Address publication always precedes any call that can post it.
inline constexpr uint64_t kEcBatchStageRetryTimeoutMs = 30000;

inline bool ConcurrentArrayCache::ec_batch_staging_ready() {
    return ::FarLib::get_config().is_ec_batch_mode() && ec_staging_pool_.valid();
}

inline bool ConcurrentArrayCache::stage_ec_batch_object(
    EvictBufferSet &buffers, void *local_addr, size_t size,
    FarObjectEntry *entry, uint32_t behavior_group, bool direct_split_write) {
    buffers.last_stage_owns_old_remote = false;
    if (direct_split_write) {
        return stage_ec_split_object(buffers, local_addr, size, entry,
                                     behavior_group, true);
    }
    // Route once on admission; already-started batches keep their protocol.
    if (::FarLib::ec_benchmark_phase::enabled() &&
        !ec_batch_uses_split(size) && size <= rdma::ec_rmw::kMaxBytes) {
        if (!buffers.direct_builder ||
            !::FarLib::allocator::ec_write_source_borrowed(
                static_cast<::FarLib::allocator::BlockHead *>(local_addr) - 1))
            ERROR("ec_benchmark_phase: small eviction lacks private builder/source ownership");
        if (!::FarLib::ec_benchmark_phase::steady())
            return stage_ec_direct_object(buffers, local_addr, size, entry, behavior_group);
        if (!stage_ec_rmw_object(buffers, local_addr, size, entry, behavior_group))
            ERROR("ec_benchmark_phase: incremental admission rejected after first Work");
        return true;
    }
    if (::FarLib::get_config().ft_incremental_one_sided &&
        stage_ec_rmw_object(buffers, local_addr, size, entry, behavior_group))
        return true;
    if (::FarLib::get_config().ft_incremental_update &&
        !::FarLib::get_config().ft_incremental_one_sided &&
        stage_ec_incremental_object(buffers, local_addr, size, entry,
                                    behavior_group))
        return true;
    if (buffers.direct_builder &&
        ::FarLib::allocator::ec_write_source_borrowed(
            static_cast<::FarLib::allocator::BlockHead *>(local_addr) - 1))
        return stage_ec_direct_object(buffers, local_addr, size, entry, behavior_group);
    profile::evict_breakdown::Scope stage_scope(
        buffers.breakdown, profile::evict_breakdown::Stage::StageControl);
    ec_diag_stage_called_.fetch_add(1, std::memory_order_relaxed);
    if (entry == nullptr || local_addr == nullptr) return false;
    ec_copy_data_bytes_.fetch_add(size, std::memory_order_relaxed);
    if (entry->is_recomputable()) ERROR("recomputable object must not enter EC");
    if (ec_batch_uses_split(size))
        return stage_ec_split_object(buffers, local_addr, size, entry,
                                     behavior_group);
    if (::FarLib::get_config().behavior_group)
        return stage_ec_size_class_object(buffers, local_addr, size, entry, behavior_group);
    if (!ec_batch_staging_ready()) {
        ec_diag_stage_fail_not_ready_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!buffers.ec_builder) {
        buffers.ec_builder.reset(new ec_batch::EcGroupBuilder(
            &remote_allocator.small_object_stripe_manager(), &ec_staging_pool_,
            ec_batch::kEcBatchPendingQueueMaxDepth, true));
        buffers.ec_builder->set_breakdown(buffers.breakdown);
        buffers.ec_cache = this;
    }
    auto &builder = *buffers.ec_builder;
    const size_t client_idx = rdma::thread_info.thread_id;
    auto *client = rdma::get_client(client_idx);
    const size_t qp_idx = client->get_qp_idx();
    // The deadline is only needed after actual backpressure. Avoid a clock
    // read for every normally admitted object.
    std::chrono::steady_clock::time_point deadline{};
    auto progress = [&] {
        const auto now = std::chrono::steady_clock::now();
        if (deadline == std::chrono::steady_clock::time_point{})
            deadline = now + std::chrono::milliseconds(kEcBatchStageRetryTimeoutMs);
        (void)post_ec_batch_pending(buffers, client_idx, qp_idx);
        {
            profile::evict_breakdown::Scope poll_scope(
                buffers.breakdown, profile::evict_breakdown::Stage::CqProcess);
            (void)check_cq_idx_with_client_idx(qp_idx, client_idx);
        }
        if (now >= deadline) {
            ERROR("ec_batch: private builder backpressure exceeded 30s");
        }
        {
            profile::evict_breakdown::Scope yield_scope(
                buffers.breakdown, profile::evict_breakdown::Stage::BackpressureYield);
            uthread::yield(); // owns no shared builder lock
        }
    };
    bool consumed = false;
    while (true) {
        if (consumed) {
            const auto status = builder.flush();
            if (status == ec_batch::EcBatchStatus::kOk) break;
            if (status != ec_batch::EcBatchStatus::kPendingQueueFull) {
                ERROR("ec_batch: private consumed object failed to seal");
            }
            progress();
            continue;
        }
        uint64_t addr = ::FarLib::allocator::remote::InvalidRemoteAddr;
        const auto status = builder.add_object(local_addr, size, &addr, behavior_group);
        ec_batch_diag_note_status(status);
        if (status == ec_batch::EcBatchStatus::kOk ||
            status == ec_batch::EcBatchStatus::kPendingQueueFull) {
            ASSERT(addr != ::FarLib::allocator::remote::InvalidRemoteAddr);
            // Even the fourth object's address is published BEFORE exposing
            // its sealed group to this worker's endpoint submission batches.
            entry->set_remote_addr(addr);
            consumed = true;
            if (status == ec_batch::EcBatchStatus::kOk) break;
            progress();
            continue;
        }
        if (status == ec_batch::EcBatchStatus::kEncodeRejected ||
            status == ec_batch::EcBatchStatus::kSealRejected ||
            status == ec_batch::EcBatchStatus::kInvalidArgument) {
            ERROR("ec_batch: private builder rejected candidate");
        }
        if (status == ec_batch::EcBatchStatus::kObjectTooLarge && size > ec_staging_pool_.slot_size()) {
            ERROR("ec_batch: candidate exceeds staging slot size");
        }
        const auto flushed = builder.flush();
        if (flushed != ec_batch::EcBatchStatus::kOk &&
            flushed != ec_batch::EcBatchStatus::kPendingQueueFull) {
            ERROR("ec_batch: private builder retry failed");
        }
        progress();
    }
    if (++buffers.ec_objects_since_post >= EvictBatchSize) {
        (void)post_ec_batch_pending(buffers, client_idx, qp_idx);
        buffers.ec_objects_since_post = 0;
    }
    ec_diag_stage_ok_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Whole objects are grouped only with the same size class and behavior key.
// The caller owns an EVICTING reference until all six writes complete; the
// registered payload lease is copied into the completion token, not freed at
// submission. Dirty/no-backup versions also use this safe fresh-group path.
inline bool ConcurrentArrayCache::stage_ec_size_class_object(
    EvictBufferSet &buffers, void *local_addr, size_t size,
    FarObjectEntry *entry, uint32_t behavior_group) {
    if (size == 0 || size > ec_batch::kEcBatchStagingMaxSlotSize ||
        behavior_group >= ec_batch::kEcBatchBehaviorGroupCount)
        ERROR("behavior_group: invalid whole-object protection request");
    const size_t client_idx = rdma::thread_info.thread_id;
    auto *client = rdma::get_client(client_idx);
    if (!client || !ec_split_buffers_.valid())
        ERROR("behavior_group: size-class buffers/client unavailable");
    if (!buffers.size_class_builder) {
        buffers.size_class_builder.reset(new ec_batch::EcSizeClassBuilder(
            &remote_allocator.small_object_stripe_manager(), &ec_split_buffers_,
            client_idx));
        buffers.ec_cache = this;
    }
    auto &builder = *buffers.size_class_builder;
    const size_t qp_idx = client->get_qp_idx();
    std::chrono::steady_clock::time_point deadline{};
    auto progress = [&] {
        if (deadline == std::chrono::steady_clock::time_point{})
            deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(kEcBatchStageRetryTimeoutMs);
        (void)post_ec_batch_pending(buffers, client_idx, qp_idx);
        if (buffers.direct_builder)
            (void)post_ec_direct_pending(buffers, client_idx, qp_idx);
        (void)check_cq_idx_with_client_idx(qp_idx, client_idx);
        if (std::chrono::steady_clock::now() >= deadline)
            ERROR("behavior_group: size-class backpressure exceeded 30s");
        uthread::yield();
    };
    bool consumed = false;
    while (true) {
        uint64_t address = ::FarLib::allocator::remote::InvalidRemoteAddr;
        const auto status = consumed ? builder.flush()
            : builder.add_object(local_addr, size, &address, behavior_group);
        if (!consumed && address != ::FarLib::allocator::remote::InvalidRemoteAddr) {
            entry->set_remote_addr(address);
            consumed = true;
        }
        if (status == ec_batch::EcBatchStatus::kOk) break;
        if (status == ec_batch::EcBatchStatus::kEncodeRejected ||
            status == ec_batch::EcBatchStatus::kSealRejected ||
            status == ec_batch::EcBatchStatus::kInvalidArgument ||
            status == ec_batch::EcBatchStatus::kObjectTooLarge)
            ERROR("behavior_group: whole-object encoding/sealing failed");
        if (!consumed) {
            const auto flushed = builder.flush();
            if (flushed != ec_batch::EcBatchStatus::kOk &&
                flushed != ec_batch::EcBatchStatus::kPendingQueueFull)
                ERROR("behavior_group: cannot seal pending size-class groups");
        }
        progress();
    }
    if (!consumed) ERROR("behavior_group: missing protected object address");
    if (++buffers.ec_objects_since_post >= EvictBatchSize) {
        (void)post_ec_batch_pending(buffers, client_idx, qp_idx);
        buffers.ec_objects_since_post = 0;
    }
    ec_diag_stage_ok_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline bool ConcurrentArrayCache::stage_ec_split_object(
    EvictBufferSet &buffers, void *local_addr, size_t size,
    FarObjectEntry *entry, uint32_t behavior_group, bool direct_split_write) {
    auto *breakdown = ec_split::mg_optimizations_enabled
                          ? buffers.breakdown : nullptr;
    const auto release_direct_source = [&] {
        if (!direct_split_write || local_addr == nullptr) return;
        auto *block = static_cast<::FarLib::allocator::BlockHead *>(local_addr) - 1;
        ::FarLib::allocator::release_ec_write_source(block);
        ec_split_direct_write_borrowed_.fetch_sub(1,
                                                  std::memory_order_relaxed);
    };
    if (local_addr == nullptr || entry == nullptr || size == 0 ||
        !ec_split_buffers_.valid()) {
        release_direct_source();
        return false;
    }
    if (size > ec_split::kMaxObjectBytes) {
        ERROR("ec_split: object exceeds the runtime's 256 KiB size limit");
        release_direct_source();
        return false;
    }
    if (direct_split_write && !ec_split::direct_write_shape(size)) {
        release_direct_source();
        return false;
    }
    const size_t fragment_bytes = ec_split::fragment_size(size);
    const size_t split_group_batch_limit = ec_split_group_batch_limit();
    auto &manager = remote_allocator.small_object_stripe_manager();
    const size_t client_idx = rdma::thread_info.thread_id;
    auto *client = rdma::get_client(client_idx);
    if (client == nullptr) {
        release_direct_source();
        return false;
    }
    const size_t qp_idx = client->get_qp_idx();
    const size_t split_batch_limit = ec_split_write_batch_limit();
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(kEcBatchStageRetryTimeoutMs);
    auto progress = [&] {
        if (split_batch_limit != 0)
            (void)post_ec_batch_pending(buffers, client_idx, qp_idx);
        {
            profile::evict_breakdown::Scope poll_scope(
                breakdown, profile::evict_breakdown::Stage::CqProcess);
            (void)ec_batch_poll_all_cqs_once();
        }
        if (std::chrono::steady_clock::now() >= deadline)
            ERROR("ec_split: unable to allocate or post protected object within 30s");
        {
            profile::evict_breakdown::Scope yield_scope(
                breakdown,
                profile::evict_breakdown::Stage::BackpressureYield);
            uthread::yield();
        }
    };
    auto flush_split_boundary = [&] {
        if (split_batch_limit == 0 || !buffers.ec_split_builder) return;
        while (buffers.ec_split_builder->pending_count() >= split_batch_limit) {
            // The normal threshold hit is a direct submit.  Yield/global poll
            // is reserved for a provider/token backpressure retry.
            (void)post_ec_batch_pending(buffers, client_idx, qp_idx);
            if (buffers.ec_split_builder->pending_count() < split_batch_limit)
                break;
            progress();
        }
    };
    if (split_batch_limit != 0) {
        if (!buffers.ec_split_builder)
            buffers.ec_split_builder.reset(new EvictBufferSet::EcSplitPendingBuilder());
        flush_split_boundary();
    }
    SmallObjectStripeManager::SlotGroupHandle group;
    {
        profile::evict_breakdown::Scope group_scope(
            breakdown, profile::evict_breakdown::Stage::GroupAllocate);
        if (split_group_batch_limit == 0) {
            while (!manager.allocate_slot_group(fragment_bytes, &group))
                progress();
        } else {
            auto &batch = buffers.ec_split_group_batch;
            const size_t bin = ::FarLib::allocator::bin_from_wsize(
                ::FarLib::allocator::wsize_from_size(fragment_bytes));
            const uint32_t expected_slot_size = static_cast<uint32_t>(
                ::FarLib::allocator::get_bin_size(bin));
            if (batch.count != 0 &&
                (batch.groups[0].bin != bin ||
                 batch.groups[0].slot_size != expected_slot_size)) {
                release_ec_split_group_batch(buffers);
            }
            while (batch.count == 0) {
                const size_t produced = manager.allocate_slot_group_batch(
                    fragment_bytes, batch.groups.data(), split_group_batch_limit);
                if (produced != 0) {
                    ASSERT(produced <= EvictBufferSet::EcSplitGroupBatch::kCapacity);
                    batch.count = produced;
                    ++buffers.ec_split_group_batch_refills;
                    buffers.ec_split_group_batch_groups += produced;
                    break;
                }
                progress();
            }
            if (!manager.group_handle_eligible_for_new_group(
                    batch.groups[batch.count - 1])) {
                release_ec_split_group_batch(buffers);
                // Retry this object through the existing endpoint-aware
                // scalar allocator after discarding stale reservations.
                while (!manager.allocate_slot_group(fragment_bytes, &group))
                    progress();
            } else {
                group = batch.groups[batch.count - 1];
                batch.groups[batch.count - 1] =
                    SmallObjectStripeManager::SlotGroupHandle{};
                --batch.count;
                ++buffers.ec_split_group_batch_consumed;
            }
        }
    }
    ec_batch::EcStagingGroupSlot staging;
    {
        profile::evict_breakdown::Scope buffer_scope(
            breakdown, profile::evict_breakdown::Stage::StageControl);
        while (!ec_split_buffers_.acquire(group.slot_size, client_idx, &staging))
            progress();
    }
    bool encoded = false;
    if (direct_split_write) {
        const size_t direct_fragment_bytes = ec_split::fragment_size(size);
        const void *sources[ec_split::kDataSlots]{};
        for (size_t i = 0; i < ec_split::kDataSlots; ++i) {
            ec_split::DirectReadFragment fragment;
            if (!ec_split::direct_read_fragment(size, i, &fragment) ||
                fragment.bytes != direct_fragment_bytes) {
                release_direct_source();
                (void)ec_split_buffers_.release(staging);
                (void)manager.mark_dead_group(group.id);
                return false;
            }
            sources[i] = static_cast<const uint8_t *>(local_addr) +
                         fragment.offset;
        }
        encoded = ec_split::encode_direct(sources, direct_fragment_bytes,
                                          staging, breakdown);
    } else {
        encoded = ec_split::encode(local_addr, size, staging);
    }
    bool sealed = false;
    if (encoded) {
        profile::evict_breakdown::Scope seal_scope(
            breakdown, profile::evict_breakdown::Stage::SealQueue);
        sealed = manager.seal_split_slot_group(
            group.id, static_cast<uint32_t>(size));
    }
    if (!encoded || !sealed) {
        (void)ec_split_buffers_.release(staging);
        (void)manager.mark_dead_group(group.id);
        release_direct_source();
        ERROR("ec_split: encode/seal failed; refusing unprotected fallback");
        return false;
    }
    ec_batch::EcGroupSendRecord record;
    {
        profile::evict_breakdown::Scope seal_scope(
            breakdown, profile::evict_breakdown::Stage::SealQueue);
        record.group = group;
        record.slot_size = group.slot_size;
        record.staging = staging;
        record.split_object = true;
        record.source_mode = direct_split_write
                                ? ec_batch::EcGroupSourceMode::kSplitDirect
                                : ec_batch::EcGroupSourceMode::kStaging;
        record.behavior_group = behavior_group;
        record.live_mask = 1;  // One application object; all four data fragments valid.
        record.live_count = 1;
        record.objects[0] = local_addr;
        record.object_sizes[0] = static_cast<uint32_t>(size);
        if (direct_split_write) {
            const size_t direct_fragment_bytes = ec_split::fragment_size(size);
            record.split_direct_write_bytes =
                static_cast<uint32_t>(direct_fragment_bytes);
            for (size_t i = 0; i < ec_split::kDataSlots; ++i) {
                ec_split::DirectReadFragment fragment;
                ASSERT(ec_split::direct_read_fragment(size, i, &fragment));
                record.split_data_sources[i] =
                    static_cast<const uint8_t *>(local_addr) + fragment.offset;
                record.split_data_lengths[i] =
                    static_cast<uint32_t>(fragment.bytes);
            }
        }
        record.sequence = ec_split_groups_staged_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (direct_split_write) {
            ec_split_direct_write_staged_.fetch_add(
                1, std::memory_order_relaxed);
            ec_split_direct_write_bytes_.fetch_add(
                size, std::memory_order_relaxed);
        }
    }
    // The caller already published EVICTING and holds its write reference.
    // Publish the anchor before any WR can complete; it represents the entire
    // split group, not a contiguous single-node copy of this large object.
    entry->set_remote_addr(group.segments[0].addr);
    if (split_batch_limit == 0) {
        while (!post_ec_batch_group(record, client_idx, qp_idx)) progress();
    } else {
        ASSERT(buffers.ec_split_builder != nullptr);
        {
            profile::evict_breakdown::Scope queue_scope(
                breakdown, profile::evict_breakdown::Stage::SealQueue);
            if (!buffers.ec_split_builder->push(record)) {
                (void)ec_split_buffers_.release(staging);
                (void)manager.mark_dead_group(group.id);
                release_direct_source();
                ERROR("ec_split: pending write batch overflow");
                return false;
            }
        }
        // Publish at the configured cross-object boundary. If token or CQ
        // pressure prevents a full drain, progress keeps the queue bounded
        // and preserves the existing global-poll + yield behavior.
        flush_split_boundary();
    }
    // No dereference of entry/local_addr after posting: completion may already
    // have released the write reference on another CQ-polling worker.
    ec_diag_stage_ok_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline void ConcurrentArrayCache::flush_ec_batch_groups(
    size_t client_idx, EvictBufferSet *buffers) {
    if (buffers && buffers->direct_builder && ec_rmw_workers_)
        flush_ec_rmw_objects(*buffers, client_idx);
    if (buffers && buffers->direct_builder && ec_update_workers_)
        flush_ec_incremental_objects(*buffers, client_idx);
    if (buffers && buffers->direct_builder) flush_ec_direct_groups(*buffers, client_idx);
    // Mutators/shutdown may help real completions, but cannot access another
    // worker's half-built groups. Every worker flushes its own batch tail.
    if (buffers == nullptr) return;
    profile::evict_breakdown::Scope tail_scope(
        buffers->breakdown, profile::evict_breakdown::Stage::StageControl);
    auto *client = rdma::get_client(client_idx);
    const size_t qp_idx = client->get_qp_idx();
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(kEcBatchStageRetryTimeoutMs);
    auto flush_builder = [&](auto &builder) {
    while (builder.group_open() || builder.pending_count() != 0) {
        if (builder.group_open()) {
            const auto status = builder.flush();
            ec_diag_flush_calls_.fetch_add(1, std::memory_order_relaxed);
            if (status == ec_batch::EcBatchStatus::kOk) {
                ec_diag_flush_sealed_ok_.fetch_add(1, std::memory_order_relaxed);
            } else if (status != ec_batch::EcBatchStatus::kPendingQueueFull) {
                ERROR("ec_batch: could not seal private batch tail");
            }
        }
        (void)post_ec_batch_pending(*buffers, client_idx, qp_idx);
        if (!builder.group_open() && builder.pending_count() == 0) break;
        {
            profile::evict_breakdown::Scope poll_scope(
                buffers->breakdown, profile::evict_breakdown::Stage::CqProcess);
            (void)check_cq_idx_with_client_idx(qp_idx, client_idx);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            ERROR("ec_batch: private tail could not be posted within 30s");
        }
        {
            profile::evict_breakdown::Scope yield_scope(
                buffers->breakdown, profile::evict_breakdown::Stage::BackpressureYield);
            uthread::yield();
        }
    }
    };
    auto flush_split_builder = [&] {
        if (!buffers->ec_split_builder) return;
        while (buffers->ec_split_builder->pending_count() != 0) {
            const size_t before = buffers->ec_split_builder->pending_count();
            (void)post_ec_batch_pending(*buffers, client_idx, qp_idx);
            if (buffers->ec_split_builder->pending_count() == 0) break;
            {
                profile::evict_breakdown::Scope poll_scope(
                    buffers->breakdown, profile::evict_breakdown::Stage::CqProcess);
                (void)check_cq_idx_with_client_idx(qp_idx, client_idx);
            }
            if (std::chrono::steady_clock::now() >= deadline)
                ERROR("ec_split: pending write batch could not be posted within 30s");
            {
                profile::evict_breakdown::Scope yield_scope(
                    buffers->breakdown, profile::evict_breakdown::Stage::BackpressureYield);
                uthread::yield();
            }
            if (buffers->ec_split_builder->pending_count() >= before &&
                ec_batch_in_flight_group_count() == 0)
                (void)ec_batch_poll_all_cqs_once();
        }
    };
    if (ec_split::mg_optimizations_enabled) {
        flush_split_builder();
        release_ec_split_group_batch(*buffers);
    }
    if (buffers->size_class_builder) flush_builder(*buffers->size_class_builder);
    else if (buffers->ec_builder) flush_builder(*buffers->ec_builder);
    buffers->ec_objects_since_post = 0;
    account_ec_worker_builder(*buffers);
}

inline void ConcurrentArrayCache::release_ec_split_group_batch(
    EvictBufferSet &buffers) {
    auto &batch = buffers.ec_split_group_batch;
    auto &manager = remote_allocator.small_object_stripe_manager();
    for (size_t i = 0; i < batch.count; ++i) {
        if (!manager.mark_dead_group(batch.groups[i].id)) {
            ERROR("ec_split: failed to retire unused preallocated group");
        }
        ++buffers.ec_split_group_batch_released;
    }
    if (buffers.ec_split_group_batch_groups !=
        buffers.ec_split_group_batch_consumed +
            buffers.ec_split_group_batch_released) {
        ERROR("ec_split: fresh group batch ownership did not close");
    }
    if (buffers.ec_split_group_batch_refills != 0) {
        ec_split_group_batch_refills_.fetch_add(
            buffers.ec_split_group_batch_refills, std::memory_order_relaxed);
        ec_split_group_batch_groups_.fetch_add(
            buffers.ec_split_group_batch_groups, std::memory_order_relaxed);
        ec_split_group_batch_consumed_.fetch_add(
            buffers.ec_split_group_batch_consumed, std::memory_order_relaxed);
        ec_split_group_batch_released_.fetch_add(
            buffers.ec_split_group_batch_released, std::memory_order_relaxed);
    }
    batch.clear();
    buffers.ec_split_group_batch_refills = 0;
    buffers.ec_split_group_batch_groups = 0;
    buffers.ec_split_group_batch_consumed = 0;
    buffers.ec_split_group_batch_released = 0;
}

// ---------------------------------------------------------------------------
// Diagnostics of the ref_cnt spin in ConcurrentArrayCache::deallocate()
// (concurrent_cache.hpp) while ft_method=ec_batch runs.
//
// Purpose: a teardown that hangs in that spin must say what it is waiting for
// *before* the cache is quiesced, i.e. while `working` is still true and the
// settle fallback cannot run.  This function only reads state and prints, one
// field per line, flushed per line; it never waits, settles, yields or
// releases anything, so the wait and every decision around it stay unchanged.
//
// `spin` is the spin counter of the reporting loop (0 for the per-object line
// printed when the wait is entered with the entry already EVICTING),
// `harvested` is the last value returned by ec_batch_poll_all_cqs_once() from
// that loop, `where` names the call site.  `damp_enter_report` throttles the
// per-object entry line (1 in 256) so a mass teardown cannot flood the log.
//
// The address reverse lookup is the same one the group-aware release uses: the
// stripe manager's binding_for_addr() (through owns()/get_slot_layout()) finds
// the data slot, its group is then read from get_slot_group_state() /
// get_slot_group_object_count().
// ---------------------------------------------------------------------------
inline void ConcurrentArrayCache::ec_batch_report_stuck_spin(
    FarObjectEntry &entry, uint64_t spin, uint64_t harvested, const char *where,
    bool damp_enter_report) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    if (damp_enter_report) {
        static std::atomic<uint64_t> enter_reports{0};
        if ((enter_reports.fetch_add(1, std::memory_order_relaxed) & 0xffull) !=
            0) {
            return;
        }
    }
    const auto bits = entry.load_state();
    const char *state_name = "UNKNOWN";
    switch (bits.state) {
    case FREE: state_name = "FREE"; break;
    case PINNED: state_name = "PINNED"; break;
    case LOCAL: state_name = "LOCAL"; break;
    case MARKED: state_name = "MARKED"; break;
    case EVICTING: state_name = "EVICTING"; break;
    case REMOTE: state_name = "REMOTE"; break;
    case FETCHING: state_name = "FETCHING"; break;
    case BUSY: state_name = "BUSY"; break;
    }
    const uint64_t remote_addr = entry.remote_addr();
    auto &stripes = remote_allocator.small_object_stripe_manager();
    using Stripes = std::remove_reference_t<decltype(stripes)>;
    const bool addr_owned = stripes.owns(remote_addr);
    Stripes::SlotLayout layout;
    const bool has_slot = stripes.get_slot_layout(remote_addr, &layout);
    bool in_group = false;
    const char *group_state_name = "none";
    uint32_t group_live = 0;
    uint8_t group_live_mask = 0;
    if (has_slot) {
        Stripes::SlotGroupId group_id;
        group_id.stripe_id = layout.stripe_id;
        group_id.slot_id = layout.slot_id;
        Stripes::SlotGroupState group_state = Stripes::kSlotGroupFree;
        if (stripes.get_slot_group_state(group_id, &group_state,
                                         &group_live_mask)) {
            switch (group_state) {
            case Stripes::kSlotGroupFree: group_state_name = "free"; break;
            case Stripes::kSlotGroupInProgress:
                group_state_name = "in_progress";
                break;
            case Stripes::kSlotGroupSealed: group_state_name = "sealed"; break;
            case Stripes::kSlotGroupDead: group_state_name = "dead"; break;
            }
            in_group = (group_state != Stripes::kSlotGroupFree);
        }
        (void)stripes.get_slot_group_object_count(group_id, &group_live);
    }
    const size_t in_flight_groups = ec_batch_in_flight_group_count();
    const size_t tokens_in_use = ec_batch_tokens_.in_use() + (ec_direct_bank_ ? ec_direct_bank_->in_use() : 0);
    const size_t pending_sealed_groups = ec_batch_diag_pending_groups();
    const size_t staging_in_use = ec_staging_pool_.in_use() + ec_split_buffers_.in_use();
    const uint64_t posted_groups =
        ec_batch_groups_posted_.load(std::memory_order_relaxed);
    const uint64_t completed_groups =
        ec_batch_groups_completed_.load(std::memory_order_relaxed);

    std::cout << "INFO: ec_stuck where=" << where << std::endl;
    std::cout << "INFO: ec_stuck spin=" << spin << std::endl;
    std::cout << "INFO: ec_stuck entry=" << static_cast<const void *>(&entry)
              << std::endl;
    std::cout << "INFO: ec_stuck ref_cnt=" << bits.ref_cnt << std::endl;
    std::cout << "INFO: ec_stuck state=" << state_name << std::endl;
    std::cout << "INFO: ec_stuck invalid=" << bits.invalid << std::endl;
    std::cout << "INFO: ec_stuck client_idx=" << entry.get_client_idx()
              << std::endl;
    std::cout << "INFO: ec_stuck remote_addr=0x" << std::hex << remote_addr
              << std::dec << std::endl;
    std::cout << "INFO: ec_stuck addr_owned_by_stripe_manager="
              << (addr_owned ? 1 : 0) << std::endl;
    std::cout << "INFO: ec_stuck addr_has_slot_layout=" << (has_slot ? 1 : 0)
              << std::endl;
    std::cout << "INFO: ec_stuck addr_is_group_data_slot="
              << (in_group ? 1 : 0) << std::endl;
    std::cout << "INFO: ec_stuck group_stripe_id="
              << (has_slot ? layout.stripe_id : uint64_t{0}) << std::endl;
    std::cout << "INFO: ec_stuck group_slot_index="
              << (has_slot ? layout.slot_id : uint32_t{0}) << std::endl;
    std::cout << "INFO: ec_stuck group_state=" << group_state_name
              << std::endl;
    std::cout << "INFO: ec_stuck group_live_count=" << group_live << std::endl;
    std::cout << "INFO: ec_stuck group_live_mask=0x" << std::hex
              << static_cast<unsigned>(group_live_mask) << std::dec
              << std::endl;
    std::cout << "INFO: ec_stuck ec_in_flight_groups=" << in_flight_groups
              << std::endl;
    std::cout << "INFO: ec_stuck ec_tokens_in_use=" << tokens_in_use
              << std::endl;
    std::cout << "INFO: ec_stuck ec_pending_sealed_groups="
              << pending_sealed_groups << std::endl;
    std::cout << "INFO: ec_stuck ec_staging_in_use=" << staging_in_use
              << std::endl;
    std::cout << "INFO: ec_stuck ec_last_poll_harvest=" << harvested
              << std::endl;
    std::cout << "INFO: ec_stuck ec_groups_posted=" << posted_groups
              << std::endl;
    std::cout << "INFO: ec_stuck ec_groups_completed=" << completed_groups
              << std::endl;
    std::cout << "INFO: ec_stuck end where=" << where << std::endl;
}
}  // namespace FarLib::cache
