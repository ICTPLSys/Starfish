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
    if (segment < kEcBatchDataSlots) return record.staging.data[segment];
    return record.staging.parity[segment - kEcBatchDataSlots];
}

// A record is postable only when all six segments carry a usable remote target
// and a local source inside the resident staging MR.  Checked before the first
// post so a rejected group cannot leave part of its segments on the wire.
inline bool ec_batch_record_is_postable(const EcGroupSendRecord &record,
                                        const EcStagingPool &pool,
                                        size_t server_count) {
    if (record.slot_size == 0) return false;
    if (record.staging.slot_size < record.slot_size) return false;
    // The shared provider validates the immutable pool/node cookie, lease
    // generation, active state, and all six source pointers.  The fixed-MR
    // fallback performs the equivalent range/free-list checks.
    if (!pool.owns_slot(record.staging)) return false;
    if (record.group.segments.size() != kEcBatchSegmentsPerGroup) return false;
    for (size_t i = 0; i < kEcBatchDataSlots; i++) {
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
        if (!pool.owns_buffer(record.staging,
                              ec_batch_record_source(record, segment),
                              record.slot_size)) {
            return false;
        }
    }
    return true;
}

}  // namespace FarLib::cache::ec_batch

namespace FarLib::cache {

class EcBatchStageGuard {
    uthread::Mutex &mutex_;
public:
    explicit EcBatchStageGuard(uthread::Mutex &mutex) : mutex_(mutex) {
        uthread::lock(&mutex_);
    }
    ~EcBatchStageGuard() { uthread::unlock(&mutex_); }
    EcBatchStageGuard(const EcBatchStageGuard &) = delete;
    EcBatchStageGuard &operator=(const EcBatchStageGuard &) = delete;
};
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
    if (ec_group_builder_ == nullptr) return 0;
    return ec_group_builder_->pending_count();
}

// "INFO: ec_batch teardown step=<name> pending_groups=N in_flight_tokens=M
//  staging_in_use=K ..." - one flushed line per teardown step, only in
// ft_method=ec_batch.
inline void ConcurrentArrayCache::ec_batch_diag_step(const char *step) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    const uint64_t sealed_groups =
        ec_group_builder_ != nullptr ? ec_group_builder_->sealed_count() : 0;
    std::cout << "INFO: ec_batch teardown step=" << step
              << " pending_groups=" << ec_batch_diag_pending_groups()
              << " in_flight_tokens=" << ec_batch_tokens_.in_use()
              << " staging_in_use=" << ec_staging_pool_.in_use()
              << " sealed_groups=" << sealed_groups
              << " posted_groups="
              << ec_batch_groups_posted_.load(std::memory_order_relaxed)
              << " completed_groups="
              << ec_batch_groups_completed_.load(std::memory_order_relaxed)
              << " stage_called="
              << ec_diag_stage_called_.load(std::memory_order_relaxed)
              << " stage_failed="
              << ec_diag_stage_failed_.load(std::memory_order_relaxed)
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
    size_t in_flight = ec_batch_tokens_.in_use();
    if (ec_group_builder_ != nullptr) {
        in_flight += ec_group_builder_->pending_count();
        in_flight += ec_group_builder_->group_open() ? 1 : 0;
    }
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
        const size_t in_flight_before = ec_batch_tokens_.in_use();
        completions += ec_batch_poll_all_cqs_once();
        ++passes;
        if (ec_batch_tokens_.in_use() == in_flight_before &&
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
    EcBatchStageGuard stage_lock(ec_batch_stage_mutex_);
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
    auto ld = [](const std::atomic<uint64_t> &value) {
        return value.load(std::memory_order_relaxed);
    };
    const uint64_t sealed_groups =
        ec_group_builder_ != nullptr ? ec_group_builder_->sealed_count() : 0;
    const uint64_t alloc_group_ok =
        ec_group_builder_ != nullptr
            ? ec_group_builder_->group_alloc_ok_count()
            : 0;
    const uint64_t alloc_group_fail =
        ec_group_builder_ != nullptr
            ? ec_group_builder_->group_alloc_fail_count()
            : 0;
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
              << " post_write_retries=" << ld(ec_diag_post_write_retries_)
              << "}" << std::endl;
    std::cout << "ec_batch diag [" << where
              << "] live: pending_groups=" << ec_batch_diag_pending_groups()
              << " in_flight_tokens=" << ec_batch_tokens_.in_use()
              << " token_table_capacity="
              << ec_batch::EcBatchTokenTable::capacity()
              << " staging_in_use=" << ec_staging_pool_.in_use()
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
    const size_t cutoff = config.ft_small_object_cutoff;
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
    return local_buf_size - reserve;
}

// Posts the six single-sided RDMA WRITEs of one sealed group: four data slots
// and the two parity slots, each segment to its own endpoint, each with an
// ec_batch wr_id that carries (token, segment).  Either all six are posted and
// the group is registered in the token table, or nothing is posted at all (no
// token acquired, no WR built) - never a half-posted group reported as success.
inline bool ConcurrentArrayCache::post_ec_batch_group(
    const ec_batch::EcGroupSendRecord &record, size_t client_idx, size_t qp_idx) {
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode() || !ec_staging_pool_.valid()) {
        ec_batch_post_rejects_.fetch_add(1, std::memory_order_relaxed);
        ec_diag_post_fail_no_mode_or_staging_.fetch_add(1,
                                                        std::memory_order_relaxed);
        return false;
    }
    if (!ec_batch::ec_batch_record_is_postable(record, ec_staging_pool_,
                                               config.server_count)) {
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
    if (!ec_batch_tokens_.acquire(record, &token_id, &token)) {
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
            ec_staging_pool_.shared_buffers_bound() ? &record.staging.lkey
                                                    : nullptr;
        while (!client->post_write(mapped.second, src, seg.slot_size, wr_id, 0,
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

// Pops every sealed group of `builder` and posts it.  A group is only popped
// when a token is available, so the builder's FIFO keeps the rest and nothing
// is dropped; a popped group that cannot be posted is a programming error and
// is reported, not silently discarded.
inline size_t ConcurrentArrayCache::post_ec_batch_pending(
    ec_batch::EcGroupBuilder &builder, size_t client_idx, size_t qp_idx) {
    if (!::FarLib::get_config().is_ec_batch_mode() || !ec_staging_pool_.valid()) {
        return 0;
    }
    size_t posted = 0;
    ec_batch::EcGroupSendRecord record;
    while (ec_batch_tokens_.available() > 0 && builder.pop_pending(&record)) {
        if (!post_ec_batch_group(record, client_idx, qp_idx)) {
            ERROR("ec_batch: sealed group could not be posted");
            ec_diag_post_pending_failed_.fetch_add(1, std::memory_order_relaxed);
        }
        posted++;
    }
    return posted;
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
    for (size_t i = 0; i < ec_batch::kEcBatchDataSlots; i++) {
        if (record.objects[i] != nullptr) {
            complete_evict_writeback(const_cast<void *>(record.objects[i]));
        }
    }
    if (!ec_staging_pool_.release(record.staging)) {
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
// (RS(4,2)) and posted as a whole at the flush point of the evacuation batch
// (EvictBufferSet::flush_all).  Staging and the flush are serialized by
// ec_batch_stage_mutex_, so every staged object has already published its
// remote address before the group that contains it can be sealed or posted:
// a completion (or a fetch of the object once it is REMOTE) can never observe a
// stale remote address.
// ---------------------------------------------------------------------------

// A candidate eviction must remain in the EC path while transient staging,
// pending-queue, token, or allocator backpressure clears.  The stage mutex is
// fibre-aware, so holding it while polling/yielding does not block the
// completion fibre that returns a token or staging slot.  A persistent
// inability to form the protection group is a correctness failure, not a
// reason to issue an unprotected flat write.
inline constexpr uint64_t kEcBatchStageRetryTimeoutMs = 30000;

inline bool ConcurrentArrayCache::ec_batch_staging_ready() {
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode()) return false;
    if (!ec_staging_pool_.valid()) return false;
    if (ec_group_builder_ != nullptr) return true;
    EcBatchStageGuard lock(ec_batch_stage_mutex_);
    if (ec_group_builder_ != nullptr) return true;
    std::unique_ptr<ec_batch::EcGroupBuilder> builder(
        new ec_batch::EcGroupBuilder(
            &remote_allocator.small_object_stripe_manager(),
            &ec_staging_pool_, ec_batch::kEcBatchPendingQueueDefaultDepth));
    if (!builder->valid()) {
        ERROR("ec_batch: group builder could not be initialized");
        return false;
    }
    ec_group_builder_ = std::move(builder);
    return true;
}

inline bool ConcurrentArrayCache::stage_ec_batch_object(void *local_addr,
                                                        size_t size,
                                                        FarObjectEntry *entry) {
    ec_diag_stage_called_.fetch_add(1, std::memory_order_relaxed);
    if (entry == nullptr || local_addr == nullptr) return false;
    if (!ec_batch_staging_ready()) {
        ec_diag_stage_fail_not_ready_.fetch_add(1,
                                                std::memory_order_relaxed);
        return false;
    }
    const size_t client_idx = rdma::thread_info.thread_id;
    auto *client = rdma::get_client(client_idx);
    if (client == nullptr) {
        ec_diag_stage_fail_no_client_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const size_t qp_idx = client->get_qp_idx();
    EcBatchStageGuard lock(ec_batch_stage_mutex_);
    auto &builder = *ec_group_builder_;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(
                              kEcBatchStageRetryTimeoutMs);
    // A fourth add can copy the object into the open group and then return
    // kPendingQueueFull before sealing.  In that case the object is already
    // consumed by the builder; never call add_object() on it again.
    bool object_consumed = false;
    uint64_t consumed_segment_addr =
        ::FarLib::allocator::remote::InvalidRemoteAddr;
    ec_batch::EcBatchStatus last_status = ec_batch::EcBatchStatus::kOk;

    auto check_deadline = [&]() {
        if (std::chrono::steady_clock::now() < deadline) return;
        if (last_status == ec_batch::EcBatchStatus::kManagerRejected) {
            ec_recovery_report_group_alloc_blocked(
                ec_batch::ec_batch_status_name(last_status));
        }
        ec_diag_stage_fail_rounds_exhausted_.fetch_add(
            1, std::memory_order_relaxed);
        ERROR("ec_batch: candidate staging retry exceeded 30s deadline");
    };

    auto retry_backpressure = [&]() {
        const ec_batch::EcBatchStatus flush_status = builder.flush();
        if (flush_status != ec_batch::EcBatchStatus::kOk &&
            flush_status != ec_batch::EcBatchStatus::kPendingQueueFull) {
            ERROR("ec_batch: staging retry could not seal the open group");
        }
        (void)post_ec_batch_pending(builder, client_idx, qp_idx);
        // The builder is shared by all eviction clients.  Poll every client's
        // QP/endpoint set, not just the caller's client.
        (void)ec_batch_poll_all_cqs_once();
        check_deadline();
        uthread::yield();
    };

    while (true) {
        check_deadline();
        if (object_consumed) {
            const ec_batch::EcBatchStatus flush_status = builder.flush();
            if (flush_status == ec_batch::EcBatchStatus::kOk) {
                if (consumed_segment_addr ==
                    ::FarLib::allocator::remote::InvalidRemoteAddr) {
                    ERROR("ec_batch: consumed object has no group address");
                }
                // Publish exactly once, after the group containing the object
                // has been sealed.  No later retry may add this object again.
                entry->set_remote_addr(consumed_segment_addr);
                ec_diag_stage_ok_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            if (flush_status != ec_batch::EcBatchStatus::kPendingQueueFull) {
                ERROR("ec_batch: consumed candidate group could not seal");
            }
            // The current group is still open and the object is already in it;
            // only drain older pending groups and retry the seal.
            (void)post_ec_batch_pending(builder, client_idx, qp_idx);
            (void)ec_batch_poll_all_cqs_once();
            check_deadline();
            uthread::yield();
            continue;
        }

        uint64_t segment_addr = 0;
        const ec_batch::EcBatchStatus status =
            builder.add_object(local_addr, size, &segment_addr);
        ec_batch_diag_note_status(status);
        last_status = status;
        if (status == ec_batch::EcBatchStatus::kOk) {
            if (segment_addr ==
                ::FarLib::allocator::remote::InvalidRemoteAddr) [[unlikely]] {
                ERROR("ec_batch: grouped data slot has no remote address");
                ec_diag_stage_fail_bad_segment_addr_.fetch_add(
                    1, std::memory_order_relaxed);
                return false;
            }
            // The object now lives in that data segment of the group.
            entry->set_remote_addr(segment_addr);
            ec_diag_stage_ok_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        if (status == ec_batch::EcBatchStatus::kPendingQueueFull) {
            // add_object() has already copied this object into slot 3 and
            // incremented staged_count_ before seal_locked() found the FIFO
            // full.  Preserve that exact-once ownership while older pending
            // groups drain; never blindly re-add this object.
            if (segment_addr ==
                    ::FarLib::allocator::remote::InvalidRemoteAddr ||
                !builder.group_open() ||
                builder.staged_count() != ec_batch::kEcBatchDataSlots) {
                ERROR("ec_batch: pending-queue-full lost fourth staged object");
            }
            object_consumed = true;
            consumed_segment_addr = segment_addr;
            continue;
        }
        if (status == ec_batch::EcBatchStatus::kInvalidArgument) {
            ERROR("ec_batch: staging rejected the eviction candidate");
        }
        if (status == ec_batch::EcBatchStatus::kObjectTooLarge) {
            // An open group inherits the size class of its first object. A
            // larger candidate can still fit the staging pool: flush the
            // smaller group and retry in a fresh group of the correct size.
            if (size > ec_staging_pool_.slot_size()) {
                ERROR("ec_batch: candidate object does not fit EC staging slot");
            }
            retry_backpressure();
            continue;
        }
        if (status == ec_batch::EcBatchStatus::kEncodeRejected ||
            status == ec_batch::EcBatchStatus::kSealRejected) {
            ERROR("ec_batch: candidate group encoding/sealing failed");
        }
        // No room right now (full open group, exhausted staging pool, or an
        // allocator refusal): close what is open, post what is sealed, help
        // every client's CQ, and retry this same object.
        retry_backpressure();
    }
}

inline void ConcurrentArrayCache::flush_ec_batch_groups(size_t client_idx) {
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode()) return;
    if (ec_group_builder_ == nullptr || !ec_staging_pool_.valid()) return;
    auto *client = rdma::get_client(client_idx);
    if (client == nullptr) return;
    const size_t qp_idx = client->get_qp_idx();
    ec_batch_diag_step("flush_ec_batch_groups_enter");
    EcBatchStageGuard lock(ec_batch_stage_mutex_);
    auto &builder = *ec_group_builder_;
    uint64_t spin = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(
                              kEcBatchStageRetryTimeoutMs);
    while (true) {
        if (builder.group_open()) {
            const ec_batch::EcBatchStatus status = builder.flush();
            ec_diag_flush_calls_.fetch_add(1, std::memory_order_relaxed);
            if (status == ec_batch::EcBatchStatus::kOk) {
                ec_diag_flush_sealed_ok_.fetch_add(1,
                                                   std::memory_order_relaxed);
            }
            if (status != ec_batch::EcBatchStatus::kOk &&
                status != ec_batch::EcBatchStatus::kPendingQueueFull) {
                ERROR("ec_batch: could not seal the open eviction group");
                break;
            }
        }
        if (builder.pending_count() == 0) break;
        (void)post_ec_batch_pending(builder, client_idx, qp_idx);
        if (builder.pending_count() == 0) break;
        // A pending group waits for a token that a completion must return.
        ++spin;
        (void)ec_batch_poll_all_cqs_once();
        if (std::chrono::steady_clock::now() >= deadline) [[unlikely]] {
            ERROR("ec_batch: sealed eviction group could not be posted within 30s");
        }
        if ((spin & 63ull) == 0) uthread::yield();
    }
    ec_batch_diag_step("flush_ec_batch_groups_exit");
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
    const size_t tokens_in_use = ec_batch_tokens_.in_use();
    const size_t pending_sealed_groups = ec_batch_diag_pending_groups();
    const size_t staging_in_use = ec_staging_pool_.in_use();
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
