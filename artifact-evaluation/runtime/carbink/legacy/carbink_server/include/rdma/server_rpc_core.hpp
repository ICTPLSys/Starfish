#pragma once

// NOTE: This file is included by `rdma/server.hpp` after `Server` class
// declaration, still inside namespace `FarLib::rdma`.

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

__attribute__((target("avx2"))) inline void
ec2pc_xor_payload_8k_avx2(uint8_t *__restrict__ dst,
                          const uint8_t *__restrict__ src) {
    constexpr size_t kVecBytes = sizeof(__m256i);
    constexpr size_t kVecCount = rdma::kEC2PCRpcPayloadBytes / kVecBytes;
    auto *dst_vec = reinterpret_cast<__m256i *>(dst);
    const auto *src_vec = reinterpret_cast<const __m256i *>(src);
    for (size_t i = 0; i < kVecCount; i++) {
        __m256i dst_v = _mm256_loadu_si256(dst_vec + i);
        __m256i src_v = _mm256_loadu_si256(src_vec + i);
        _mm256_storeu_si256(dst_vec + i, _mm256_xor_si256(dst_v, src_v));
    }
}

inline bool ec2pc_cpu_supports_avx2() {
    static const bool supported = __builtin_cpu_supports("avx2");
    return supported;
}
#endif

inline void ec2pc_xor_payload_8k(uint8_t *__restrict__ dst,
                                 const uint8_t *__restrict__ src) {
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_expect(ec2pc_cpu_supports_avx2(), 1)) {
        ec2pc_xor_payload_8k_avx2(dst, src);
        return;
    }
#endif
    constexpr size_t kWordBytes = sizeof(uint64_t);
    constexpr size_t kWordCount = rdma::kEC2PCRpcPayloadBytes / kWordBytes;
    auto *dst64 = reinterpret_cast<uint64_t *>(dst);
    const auto *src64 = reinterpret_cast<const uint64_t *>(src);
    for (size_t i = 0; i < kWordCount; i++) {
        dst64[i] ^= src64[i];
    }
}

// Build a plain EC2PC RPC header so call sites do not open-code the same field
// assignments for ACK/probe messages.
inline rdma::EC2PCRpcMessage make_ec2pc_rpc_message(
    uint16_t type, uint64_t wr_id, uint16_t target_endpoint,
    uint32_t payload_len = 0, uint16_t flags = 0,
    uint64_t target_offset = 0) {
    return {
        .type = type,
        .flags = flags,
        .payload_len = payload_len,
        .wr_id = wr_id,
        .target_endpoint = target_endpoint,
        .target_offset = target_offset,
        .status = rdma::kEC2PCStatusOK,
    };
}

inline void Server::drop_unposted_peer_work(size_t peer_idx) {
    if (!peer_recovery_enabled() || peer_idx >= peer_endpoint_count)
        return;

    auto subtract_pending = [&](size_t count) {
        size_t current = pending_peer_send_total.load(
            std::memory_order_relaxed);
        while (count != 0 && current != 0) {
            size_t next = current > count ? current - count : 0;
            if (pending_peer_send_total.compare_exchange_weak(
                    current, next, std::memory_order_relaxed,
                    std::memory_order_relaxed))
                break;
        }
    };
    auto release_external_owner = [&](BatchRpcRecvSlot *owner,
                                      uint64_t generation) {
        if (owner == nullptr || generation == 0 ||
            owner->generation.load(std::memory_order_acquire) != generation)
            return;
        uint8_t refs = owner->parity_send_refs.load(
            std::memory_order_acquire);
        while (refs != 0 &&
               !owner->parity_send_refs.compare_exchange_weak(
                   refs, static_cast<uint8_t>(refs - 1),
                   std::memory_order_acq_rel,
                   std::memory_order_acquire)) {
        }
        if (refs == 1)
            release_batch_rpc_recv_slot(*owner);
    };
    if (peer_idx < pending_peer_send_queues.size() &&
        pending_peer_send_queue_mutexes != nullptr) {
        std::lock_guard<std::mutex> lock(
            pending_peer_send_queue_mutexes[peer_idx]);
        size_t dropped = pending_peer_send_queues[peer_idx].size();
        pending_peer_send_queues[peer_idx].clear();
        subtract_pending(dropped);
    }
    if (peer_idx < pending_peer_ack_queues.size() &&
        pending_peer_ack_queue_mutexes != nullptr) {
        std::lock_guard<std::mutex> lock(
            pending_peer_ack_queue_mutexes[peer_idx]);
        pending_peer_ack_queues[peer_idx].clear();
    }

    size_t lane_owner_count = std::max<size_t>(1, rpc_worker_count);
    for (size_t worker_idx = 0; worker_idx < lane_owner_count; ++worker_idx) {
        size_t lane_idx = peer_idx * lane_owner_count + worker_idx;
        if (lane_idx >= peer_lane_send_queues.size() ||
            peer_lane_mutexes == nullptr)
            continue;
        std::lock_guard<std::mutex> lock(peer_lane_mutexes[lane_idx]);
        auto &lane = peer_lane_send_queues[lane_idx];
        lane.pending_head = 0;
        lane.pending_tail = 0;
        lane.pending_count = 0;
    }

    for (size_t shard_idx = 0;
         shard_idx < std::max<size_t>(1, peer_payload_qp_count);
         ++shard_idx) {
        size_t q_idx = probe_parity_batch_queue_index(peer_idx, shard_idx);
        if (q_idx >= probe_parity_batch_send_queues.size())
            continue;
        if (probe_parity_batch_pending_mutexes != nullptr) {
            std::lock_guard<std::mutex> lock(
                probe_parity_batch_pending_mutexes[q_idx]);
            auto &queue = probe_parity_batch_send_queues[q_idx];
            if (queue.pending_items != nullptr &&
                queue.pending_capacity != 0) {
                for (size_t n = 0; n < queue.pending_count; ++n) {
                    auto &item = queue.pending_items[
                        (queue.pending_head + n) % queue.pending_capacity];
                    release_external_owner(
                        item.batch_owner, item.batch_owner_generation);
                    item = ProbeParityBatchPendingItem{};
                }
            }
            queue.pending_head = 0;
            queue.pending_tail = 0;
            queue.pending_count = 0;
            if (probe_parity_batch_pending_space_cvs != nullptr)
                probe_parity_batch_pending_space_cvs[q_idx].notify_all();
        }
        if (probe_parity_batch_ready_mutexes != nullptr &&
            q_idx < probe_parity_batch_ready_queues.size()) {
            std::lock_guard<std::mutex> lock(
                probe_parity_batch_ready_mutexes[q_idx]);
            probe_parity_batch_ready_queues[q_idx].clear();
        }
        if (probe_parity_batch_send_mutexes != nullptr) {
            std::lock_guard<std::mutex> lock(
                probe_parity_batch_send_mutexes[q_idx]);
            auto &queue = probe_parity_batch_send_queues[q_idx];
            for (size_t slot_idx = 0;
                 queue.slots != nullptr && slot_idx < queue.depth;
                 ++slot_idx) {
                auto &slot = queue.slots[slot_idx];
                if (slot.state.load(std::memory_order_acquire) ==
                    kProbeBatchSlotReady) {
                    release_external_owner(
                        slot.batch_owner, slot.batch_owner_generation);
                    slot.reset_message_binding();
                    slot.in_use.store(false, std::memory_order_release);
                    slot.state.store(kProbeBatchSlotFree,
                                     std::memory_order_release);
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(
            probe_parity_batch_pair_pending_mutex);
        for (auto it = probe_parity_batch_pair_pending_queue.begin();
             it != probe_parity_batch_pair_pending_queue.end();) {
            if (it->peer_idx0 == peer_idx || it->peer_idx1 == peer_idx) {
                release_external_owner(
                    it->batch_owner, it->batch_owner_generation);
                release_external_owner(
                    it->batch_owner, it->batch_owner_generation);
                it = probe_parity_batch_pair_pending_queue.erase(it);
            } else
                ++it;
        }
    }
    {
        std::lock_guard<std::mutex> lock(
            runtime_parity_batch_pair_pending_mutex);
        for (auto it = runtime_parity_batch_pair_pending_queue.begin();
             it != runtime_parity_batch_pair_pending_queue.end();) {
            if (it->peer_idx0 == peer_idx || it->peer_idx1 == peer_idx)
                it = runtime_parity_batch_pair_pending_queue.erase(it);
            else
                ++it;
        }
    }
}

inline void Server::fail_tracked_requests_for_peer(
    size_t peer_idx, uint32_t status) {
    if (!peer_recovery_enabled())
        return;
    struct FailedRequest {
        uint64_t wr_id;
        size_t client_qp_local_idx;
        uint16_t completion_msg_type;
        uint16_t target_endpoint;
    };
    std::vector<FailedRequest> failed;
    {
        std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
        for (auto it = server_ec2pc_track_.begin();
             it != server_ec2pc_track_.end();) {
            auto &request = it->second;
            if (request.parity_peer0 != peer_idx &&
                request.parity_peer1 != peer_idx) {
                ++it;
                continue;
            }
            request.failed = true;
            request.failure_status = status;
            failed.push_back({it->first, request.client_qp_local_idx,
                              request.completion_msg_type,
                              request.target_endpoint});
            server_ec2pc_canceled_.insert(it->first);
            it = server_ec2pc_track_.erase(it);
        }
    }
    for (const auto &request : failed) {
        rdma::EC2PCRpcMessage ack = make_ec2pc_rpc_message(
            request.completion_msg_type, request.wr_id,
            request.target_endpoint);
        ack.status = status;
        if (request.completion_msg_type == rdma::EC2PC_MSG_ACK_BATCH) {
            ack.payload_len = static_cast<uint32_t>(sizeof(uint64_t));
            std::memcpy(ack.payload, &request.wr_id, sizeof(uint64_t));
        }
        // The client may already have quarantined this request.  A failed
        // ACK is still best-effort; healthy peer/server threads must not
        // abort merely because the client lane disappeared.
        (void)enqueue_rpc_send(request.client_qp_local_idx, ack);
    }
}

inline bool Server::mark_peer_dead(size_t peer_idx, uint32_t status) {
    if (!peer_recovery_enabled() || peer_alive_ == nullptr ||
        peer_idx >= peer_endpoint_count || peer_idx == local_server_index)
        return false;
    uint8_t expected = 1;
    if (!peer_alive_[peer_idx].compare_exchange_strong(
            expected, 0, std::memory_order_acq_rel))
        return false;
    drop_unposted_peer_work(peer_idx);
    fail_tracked_requests_for_peer(peer_idx, status);
    return true;
}

// Tag a parity-send slot with a fresh bind generation before it is published.
inline void Server::prepare_probe_parity_slot(
    ProbeParityBatchSendSlot &slot) {
    slot.bind_generation.store(
        probe_parity_bind_generation_.fetch_add(1, std::memory_order_relaxed) +
            1,
        std::memory_order_release);
}

// Kick one ready runtime/probe batch toward the target peer. When the caller
// leaves shard selection unspecified, walk shards until one post succeeds.
inline size_t Server::kick_probe_parity_ready_slots(size_t peer_idx,
                                                    size_t preferred_shard_idx,
                                                    size_t max_posts) {
    if (preferred_shard_idx != std::numeric_limits<size_t>::max()) {
        return flush_ready_probe_parity_batch_send_slots(peer_idx,
                                                         preferred_shard_idx,
                                                         max_posts);
    }
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    for (size_t shard_idx = 0; shard_idx < payload_qp_count; shard_idx++) {
        size_t posted = flush_ready_probe_parity_batch_send_slots(peer_idx,
                                                                  shard_idx,
                                                                  max_posts);
        if (posted != 0) {
            return posted;
        }
    }
    return 0;
}

// Publish one ready slot into the per-(peer, shard) queue while accounting for
// queue bookkeeping time in the existing runtime-parity metrics.
inline void Server::enqueue_probe_parity_ready_slot(size_t queue_idx,
                                                    uint16_t slot_index,
                                                    uint64_t bind_seq,
                                                    uint64_t ready_enqueue_time_ns) {
    if (queue_idx >= probe_parity_batch_ready_queues.size() ||
        probe_parity_batch_ready_mutexes == nullptr) {
        return;
    }
    if (ready_enqueue_time_ns == 0) {
        ready_enqueue_time_ns = Server::steady_clock_now_ns();
    }
    uint64_t queue_begin_ns = Server::steady_clock_now_ns();
    std::lock_guard<std::mutex> lock(probe_parity_batch_ready_mutexes[queue_idx]);
    probe_parity_batch_ready_queues[queue_idx].push_back(
        {.slot_index = slot_index,
         .bind_seq = bind_seq,
         .ready_enqueue_time_ns = ready_enqueue_time_ns});
    uint64_t queue_end_ns = Server::steady_clock_now_ns();
    ctr_runtime_parity_ready_queue_ns.fetch_add(
        queue_end_ns - queue_begin_ns, std::memory_order_relaxed);
    ctr_runtime_parity_ready_queue_cnt.fetch_add(1, std::memory_order_relaxed);
}

// Pop one ready slot from the per-(peer, shard) queue and charge the queue
// access time to the same runtime-parity bookkeeping counters.
inline bool Server::try_dequeue_probe_parity_ready_slot(
    size_t queue_idx, ProbeParityBatchReadyItem &item) {
    if (queue_idx >= probe_parity_batch_ready_queues.size() ||
        probe_parity_batch_ready_mutexes == nullptr) {
        return false;
    }
    uint64_t queue_begin_ns = Server::steady_clock_now_ns();
    bool have_slot = false;
    {
        std::lock_guard<std::mutex> lock(
            probe_parity_batch_ready_mutexes[queue_idx]);
        auto &ready_q = probe_parity_batch_ready_queues[queue_idx];
        if (!ready_q.empty()) {
            item = ready_q.front();
            ready_q.pop_front();
            have_slot = true;
        }
    }
    uint64_t queue_end_ns = Server::steady_clock_now_ns();
    ctr_runtime_parity_ready_queue_ns.fetch_add(
        queue_end_ns - queue_begin_ns, std::memory_order_relaxed);
    ctr_runtime_parity_ready_queue_cnt.fetch_add(1, std::memory_order_relaxed);
    return have_slot;
}

inline void Server::post_recv_slot(RpcRecvSlot &slot) {
    if (slot.has_batch_buffer()) {
        if (batch_rpc_mr == nullptr) {
            ERROR("server post_recv_slot: batch transport not ready");
        }
    }
    ibv_sge sge = {
        .addr = slot.has_batch_buffer()
                    ? reinterpret_cast<uint64_t>(&slot.message())
                    : reinterpret_cast<uint64_t>(&slot.msg),
        .length = slot.has_batch_buffer()
                      ? static_cast<uint32_t>(
                            sizeof(rdma::EC2PCDataReqBatchMessage))
                      : static_cast<uint32_t>(sizeof(rdma::EC2PCRpcMessage)),
        .lkey = slot.has_batch_buffer() ? batch_rpc_mr->lkey : rpc_mr->lkey,
    };
    ibv_recv_wr wr = {
        .wr_id = reinterpret_cast<uint64_t>(&slot),
        .next = nullptr,
        .sg_list = &sge,
        .num_sge = 1,
    };
    ibv_recv_wr *bad = nullptr;
    int ret = ibv_post_recv(qps[slot.qp_idx].queue_pair, &wr, &bad);
    if (ret != 0) {
        ctr_recv_repost_fail.fetch_add(1, std::memory_order_relaxed);
        ERROR("server post_recv_slot failed");
    }
}

inline void Server::release_batch_rpc_recv_slot(BatchRpcRecvSlot &slot) {
    uint64_t repost_time_ns = steady_clock_now_ns();
    uint64_t recv_wc_time_ns =
        slot.last_recv_wc_time_ns.exchange(0, std::memory_order_acq_rel);
    if (recv_wc_time_ns != 0 && repost_time_ns >= recv_wc_time_ns) {
        uint64_t delta_ns = repost_time_ns - recv_wc_time_ns;
        ctr_batch_recv_to_repost_ns.fetch_add(delta_ns,
                                              std::memory_order_relaxed);
        ctr_batch_recv_to_repost_cnt.fetch_add(1, std::memory_order_relaxed);
        update_peak(ctr_batch_recv_to_repost_max_ns, delta_ns);
    }
    ctr_batch_recv_release.fetch_add(1, std::memory_order_relaxed);
    uint64_t inflight_before =
        ctr_batch_recv_inflight.fetch_sub(1, std::memory_order_acq_rel);
    if (inflight_before == 0) {
        ctr_batch_recv_inflight.store(0, std::memory_order_relaxed);
    }
    slot.parity_send_refs.store(0, std::memory_order_release);
    slot.generation.fetch_add(1, std::memory_order_acq_rel);
    post_recv_slot(slot);
    ctr_recv_repost_ok.fetch_add(1, std::memory_order_relaxed);
}

inline void Server::release_probe_parity_batch_owner(
    BatchRpcRecvSlot *owner, uint64_t generation) {
    if (owner == nullptr || generation == 0 ||
        owner->generation.load(std::memory_order_acquire) != generation)
        return;
    uint8_t refs = owner->parity_send_refs.load(
        std::memory_order_acquire);
    while (refs != 0 &&
           !owner->parity_send_refs.compare_exchange_weak(
               refs, static_cast<uint8_t>(refs - 1),
               std::memory_order_acq_rel,
               std::memory_order_acquire)) {
    }
    if (refs == 1)
        release_batch_rpc_recv_slot(*owner);
}

inline Server::BatchRpcMessageBuffer *
Server::try_acquire_batch_rpc_message_buffer() {
    std::lock_guard<std::mutex> lock(batch_rpc_msg_pool_mutex);
    if (batch_rpc_msg_free_pool.empty()) {
        return nullptr;
    }
    BatchRpcMessageBuffer *buffer = batch_rpc_msg_free_pool.back();
    batch_rpc_msg_free_pool.pop_back();
    return buffer;
}

inline void Server::release_batch_rpc_message_buffer(
    BatchRpcMessageBuffer *buffer) {
    if (buffer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(batch_rpc_msg_pool_mutex);
    batch_rpc_msg_free_pool.push_back(buffer);
}

inline bool Server::acquire_runtime_parity_batch_pair_storage_slot(
    size_t worker_idx, uint32_t &slot_idx_out,
    RuntimeParityBatchPairStorageSlot *&slot_out) {
    slot_idx_out = 0;
    slot_out = nullptr;
    if (worker_idx >= runtime_parity_batch_pair_storage_by_worker.size()) {
        return false;
    }
    auto &storage = runtime_parity_batch_pair_storage_by_worker[worker_idx];
    if (storage.slots == nullptr || storage.free_bitmap == nullptr ||
        storage.capacity == 0 || storage.free_bitmap_words == 0) {
        return false;
    }
    size_t start_word = (storage.alloc_hint / 64) % storage.free_bitmap_words;
    for (size_t word_offset = 0; word_offset < storage.free_bitmap_words;
         word_offset++) {
        size_t word_idx =
            (start_word + word_offset) % storage.free_bitmap_words;
        uint64_t available = storage.free_bitmap[word_idx];
        while (available != 0) {
            unsigned bit = static_cast<unsigned>(__builtin_ctzll(available));
            size_t slot_idx = word_idx * 64 + bit;
            uint64_t mask = 1ull << bit;
            available &= ~mask;
            if (slot_idx >= storage.capacity ||
                (storage.free_bitmap[word_idx] & mask) == 0) {
                continue;
            }
            storage.free_bitmap[word_idx] &= ~mask;
            storage.alloc_hint = (slot_idx + 1) % storage.capacity;
            auto &slot = storage.slots[slot_idx];
            slot.in_use = true;
            slot_idx_out = static_cast<uint32_t>(slot_idx);
            slot_out = &slot;
            return true;
        }
    }
    return false;
}

inline void Server::release_runtime_parity_batch_pair_storage_slot(
    size_t worker_idx, uint32_t slot_idx) {
    auto *slot = get_runtime_parity_batch_pair_storage_slot(worker_idx, slot_idx);
    if (slot == nullptr) {
        return;
    }
    auto &storage = runtime_parity_batch_pair_storage_by_worker[worker_idx];
    slot->first_try_time_ns = 0;
    slot->pending_enqueue_time_ns = 0;
    slot->retry_count = 0;
    slot->in_use = false;
    size_t word_idx = static_cast<size_t>(slot_idx) / 64;
    size_t bit_idx = static_cast<size_t>(slot_idx) % 64;
    storage.free_bitmap[word_idx] |= (1ull << bit_idx);
    storage.alloc_hint = static_cast<size_t>(slot_idx);
}

inline Server::RuntimeParityBatchPairStorageSlot *
Server::get_runtime_parity_batch_pair_storage_slot(size_t worker_idx,
                                                   uint32_t slot_idx) {
    if (worker_idx >= runtime_parity_batch_pair_storage_by_worker.size()) {
        return nullptr;
    }
    auto &storage = runtime_parity_batch_pair_storage_by_worker[worker_idx];
    if (storage.slots == nullptr || slot_idx >= storage.capacity) {
        return nullptr;
    }
    auto &slot = storage.slots[slot_idx];
    return slot.in_use ? &slot : nullptr;
}

inline bool Server::complete_probe_parity_batch_send_slot(
    ProbeParityBatchSendSlot &slot, uint64_t cqe_time_ns) {
    record_probe_parity_batch_send_cqe(slot, cqe_time_ns);
    const auto *completed_msg = slot.active_msg();
    bool runtime_msg = completed_msg != nullptr &&
                       (completed_msg->flags & rdma::kEC2PCFlagProbeBatch) ==
                           0;
    auto *batch_owner = slot.batch_owner;
    uint64_t batch_generation = slot.batch_owner_generation;
    uint16_t peer_idx = slot.peer_idx;
    uint16_t shard_idx = slot.shard_idx;
    slot.post_time_ns = 0;
    slot.state.store(kProbeBatchSlotFree, std::memory_order_release);
    slot.in_use.store(false, std::memory_order_release);
    slot.reset_message_binding();
    if (batch_owner != nullptr && batch_generation != 0 &&
        batch_owner->generation.load(std::memory_order_acquire) ==
            batch_generation) {
        uint8_t refs = batch_owner->parity_send_refs.load(
            std::memory_order_acquire);
        while (refs != 0 &&
               !batch_owner->parity_send_refs.compare_exchange_weak(
                   refs, static_cast<uint8_t>(refs - 1),
                   std::memory_order_acq_rel,
                   std::memory_order_acquire)) {
        }
        if (refs == 1) {
            release_batch_rpc_recv_slot(*batch_owner);
        }
    }
    // Keep SEND-CQE handling lightweight. The main flush thread refills probe
    // and runtime parity transport state; doing it inline here can monopolize
    // the completion poller for milliseconds or worse.
    (void)runtime_msg;
    (void)peer_idx;
    (void)shard_idx;
    return true;
}

inline void Server::post_peer_recv_slot(PeerRpcRecvSlot &slot) {
    if (peer_recovery_enabled() && !peer_is_alive(slot.peer_idx))
        return;
    if (peer_rpc_mr == nullptr || slot.peer_idx >= peer_qps.size() ||
        peer_qps[slot.peer_idx].queue_pair == nullptr) {
        ERROR("server post_peer_recv_slot: peer transport not ready");
    }
    ibv_sge sge = {
        .addr = reinterpret_cast<uint64_t>(slot.bytes.data()),
        .length = static_cast<uint32_t>(rdma::kPeerRpcRecvBytes),
        .lkey = peer_rpc_mr->lkey,
    };
    ibv_recv_wr wr = {
        .wr_id = reinterpret_cast<uint64_t>(&slot),
        .next = nullptr,
        .sg_list = &sge,
        .num_sge = 1,
    };
    ibv_recv_wr *bad = nullptr;
    int ret = ibv_post_recv(peer_qps[slot.peer_idx].queue_pair, &wr, &bad);
    if (ret != 0) {
        ERROR("server post_peer_recv_slot failed");
    }
}

inline bool Server::post_probe_parity_batch_message(
    size_t peer_idx, const rdma::ProbeParityBatchMessage &msg,
    size_t preferred_shard_idx) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    bool runtime_msg = (msg.flags & rdma::kEC2PCFlagProbeBatch) == 0;
    uint64_t reserve_begin_ns = 0;
    uint64_t reserve_end_ns = 0;
    uint64_t copy_end_ns = 0;
    bool queued_pending = false;
    (void)reserve_begin_ns;
    (void)reserve_end_ns;
    (void)copy_end_ns;
    (void)queued_pending;
    if (runtime_msg) {
        reserve_begin_ns = steady_clock_now_ns();
    }
    auto *slot = reserve_probe_parity_batch_send_slot(
        peer_idx, false, preferred_shard_idx);
    if (slot == nullptr) {
        if (!enqueue_probe_parity_batch_pending(peer_idx, msg,
                                                preferred_shard_idx)) {
            if (runtime_msg) {
                ctr_runtime_parity_batch_reserve_fail.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return false;
        }
        queued_pending = true;
        if (runtime_msg) {
            ctr_runtime_parity_batch_reserve_fail.fetch_add(
                1, std::memory_order_relaxed);
            reserve_end_ns = steady_clock_now_ns();
            copy_end_ns = reserve_end_ns;
        }
    } else {
        uint32_t wire_bytes = rdma::probe_parity_batch_wire_bytes(msg.span_count);
        if (runtime_msg) {
            reserve_end_ns = steady_clock_now_ns();
        }
        slot->reset_message_binding();
        slot->last_flush_bookkeeping_ns = 0;
        slot->last_post_send_ns = 0;
        std::memcpy(&slot->msg, &msg, wire_bytes);
        slot->msg_ptr = &slot->msg;
        slot->msg_lkey = peer_rpc_mr != nullptr ? peer_rpc_mr->lkey : 0;
        prepare_probe_parity_slot(*slot);
        if (runtime_msg) {
            copy_end_ns = steady_clock_now_ns();
        }
        publish_probe_parity_batch_send_slot(*slot);
    }
    (void)runtime_msg;
    (void)reserve_begin_ns;
    (void)reserve_end_ns;
    (void)copy_end_ns;
    (void)queued_pending;
    (void)kick_probe_parity_ready_slots(peer_idx, preferred_shard_idx, 1);
    return true;
}

inline bool Server::post_probe_parity_batch_message_external(
    size_t peer_idx, rdma::ProbeParityBatchMessage *msg, uint32_t msg_lkey,
    BatchRpcRecvSlot *batch_owner, uint64_t batch_owner_generation,
    size_t preferred_shard_idx) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg == nullptr || msg_lkey == 0 || batch_owner == nullptr) {
        return false;
    }
    bool runtime_msg = (msg->flags & rdma::kEC2PCFlagProbeBatch) == 0;
    uint64_t reserve_begin_ns = runtime_msg ? steady_clock_now_ns() : 0;
    uint64_t reserve_end_ns = reserve_begin_ns;
    auto *slot = reserve_probe_parity_batch_send_slot(
        peer_idx, false, preferred_shard_idx);
    bool queued_pending = false;
    (void)reserve_begin_ns;
    (void)reserve_end_ns;
    (void)queued_pending;
    if (slot == nullptr) {
        if (!enqueue_probe_parity_batch_pending_external(
                peer_idx, msg, msg_lkey, batch_owner, batch_owner_generation,
                preferred_shard_idx)) {
            if (runtime_msg) {
                ctr_runtime_parity_batch_reserve_fail.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return false;
        }
        queued_pending = true;
        if (runtime_msg) {
            ctr_runtime_parity_batch_reserve_fail.fetch_add(
                1, std::memory_order_relaxed);
            reserve_end_ns = steady_clock_now_ns();
        }
    } else {
        if (runtime_msg) {
            reserve_end_ns = steady_clock_now_ns();
        }
        slot->reset_message_binding();
        slot->last_flush_bookkeeping_ns = 0;
        slot->last_post_send_ns = 0;
        slot->msg_ptr = msg;
        slot->msg_lkey = msg_lkey;
        slot->batch_owner = batch_owner;
        slot->batch_owner_generation = batch_owner_generation;
        prepare_probe_parity_slot(*slot);
        publish_probe_parity_batch_send_slot(*slot);
    }
    (void)runtime_msg;
    (void)reserve_begin_ns;
    (void)reserve_end_ns;
    (void)queued_pending;
    (void)kick_probe_parity_ready_slots(peer_idx, preferred_shard_idx, 1);
    return true;
}

// Publish the two runtime parity packets as one fanout unit. Either both get
// transport slots together, or the pair stays pending together.
inline bool Server::post_probe_parity_batch_message_pair_external(
    size_t peer_idx0, rdma::ProbeParityBatchMessage *msg0, size_t peer_idx1,
    rdma::ProbeParityBatchMessage *msg1, uint32_t msg_lkey,
    BatchRpcRecvSlot *batch_owner, uint64_t batch_owner_generation,
    size_t preferred_shard_idx0, size_t preferred_shard_idx1) {
    if ((peer_recovery_enabled() && !peer_is_alive(peer_idx0)) ||
        (peer_recovery_enabled() && !peer_is_alive(peer_idx1)))
        return false;
    if (msg0 == nullptr || msg1 == nullptr || msg_lkey == 0 ||
        batch_owner == nullptr) {
        return false;
    }
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    auto pick_shard = [&](size_t peer_idx,
                          size_t preferred_shard_idx) -> uint16_t {
        if (preferred_shard_idx == std::numeric_limits<size_t>::max()) {
            if (probe_parity_batch_next_shards == nullptr ||
                peer_idx >= peer_endpoint_count) {
                return std::numeric_limits<uint16_t>::max();
            }
            return static_cast<uint16_t>(
                probe_parity_batch_next_shards[peer_idx].fetch_add(
                    1, std::memory_order_relaxed) %
                payload_qp_count);
        }
        return static_cast<uint16_t>(preferred_shard_idx % payload_qp_count);
    };
    uint16_t chosen_shard0 = pick_shard(peer_idx0, preferred_shard_idx0);
    uint16_t chosen_shard1 = pick_shard(peer_idx1, preferred_shard_idx1);
    if (chosen_shard0 == std::numeric_limits<uint16_t>::max() ||
        chosen_shard1 == std::numeric_limits<uint16_t>::max()) {
        return false;
    }
    ProbeParityBatchPairPendingItem item{
        .peer_idx0 = peer_idx0,
        .peer_idx1 = peer_idx1,
        .preferred_shard_idx0 = chosen_shard0,
        .preferred_shard_idx1 = chosen_shard1,
        .msg_lkey = msg_lkey,
        .batch_owner = batch_owner,
        .batch_owner_generation = batch_owner_generation,
        .msg0 = msg0,
        .msg1 = msg1,
    };
    if (try_post_probe_parity_batch_pair_pending_item(item)) {
        return true;
    }
    std::lock_guard<std::mutex> lock(probe_parity_batch_pair_pending_mutex);
    item.pending_enqueue_time_ns = Server::steady_clock_now_ns();
    uint64_t depth_after_enqueue =
        probe_parity_batch_pair_pending_queue.size() + 1;
    ctr_runtime_parity_ext_pair_enqueue_depth_sum.fetch_add(
        depth_after_enqueue, std::memory_order_relaxed);
    ctr_runtime_parity_ext_pair_enqueue_depth_cnt.fetch_add(
        1, std::memory_order_relaxed);
    update_peak(ctr_runtime_parity_ext_pair_enqueue_depth_max,
                depth_after_enqueue);
    probe_parity_batch_pair_pending_queue.push_back(item);
    return true;
}

inline bool Server::enqueue_runtime_parity_batch_pair_pending(
    size_t peer_idx0, const rdma::ProbeParityBatchMessage &msg0,
    size_t preferred_shard_idx0, size_t peer_idx1,
    const rdma::ProbeParityBatchMessage &msg1, size_t preferred_shard_idx1) {
    if (peer_idx0 >= peer_endpoint_count || peer_idx1 >= peer_endpoint_count ||
        peer_idx0 == local_server_index || peer_idx1 == local_server_index) {
        return false;
    }
    if (peer_recovery_enabled() &&
        (!peer_is_alive(peer_idx0) || !peer_is_alive(peer_idx1)))
        return false;
    RuntimeParityBatchPairPendingItem item{};
    item.peer_idx0 = peer_idx0;
    item.peer_idx1 = peer_idx1;
    item.preferred_shard_idx0 = preferred_shard_idx0 ==
                                        std::numeric_limits<size_t>::max()
                                    ? std::numeric_limits<uint16_t>::max()
                                    : static_cast<uint16_t>(preferred_shard_idx0);
    item.preferred_shard_idx1 = preferred_shard_idx1 ==
                                        std::numeric_limits<size_t>::max()
                                    ? std::numeric_limits<uint16_t>::max()
                                    : static_cast<uint16_t>(preferred_shard_idx1);
    item.msg0 = msg0;
    item.msg1 = msg1;
    std::lock_guard<std::mutex> lock(runtime_parity_batch_pair_pending_mutex);
    if (peer_recovery_enabled() &&
        (!peer_is_alive(peer_idx0) || !peer_is_alive(peer_idx1)))
        return false;
    runtime_parity_batch_pair_pending_queue.push_back(std::move(item));
    return true;
}

inline bool Server::try_post_runtime_parity_batch_pair_pending_item(
    const RuntimeParityBatchPairPendingItem &item) {
    if (peer_rpc_mr == nullptr) {
        return false;
    }
    if (peer_recovery_enabled() &&
        (!peer_is_alive(item.peer_idx0) || !peer_is_alive(item.peer_idx1)))
        return true;
    auto preferred_shard = [](uint16_t shard) -> size_t {
        return shard == std::numeric_limits<uint16_t>::max()
                   ? std::numeric_limits<size_t>::max()
                   : static_cast<size_t>(shard);
    };
    auto *slot0 = reserve_probe_parity_batch_send_slot(
        item.peer_idx0, true, preferred_shard(item.preferred_shard_idx0));
    if (slot0 == nullptr) {
        ctr_runtime_parity_batch_reserve_fail.fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
    auto *slot1 = reserve_probe_parity_batch_send_slot(
        item.peer_idx1, true, preferred_shard(item.preferred_shard_idx1));
    if (slot1 == nullptr) {
        ctr_runtime_parity_batch_reserve_fail.fetch_add(
            1, std::memory_order_relaxed);
        release_reserved_probe_parity_batch_send_slot(*slot0);
        return false;
    }
    auto publish_slot = [&](ProbeParityBatchSendSlot &slot, size_t peer_idx,
                            const rdma::ProbeParityBatchMessage &msg) {
        uint32_t wire_bytes = rdma::probe_parity_batch_wire_bytes(msg.span_count);
        slot.reset_message_binding();
        slot.last_flush_bookkeeping_ns = 0;
        slot.last_post_send_ns = 0;
        slot.peer_idx = static_cast<uint16_t>(peer_idx);
        uint64_t memcpy_begin_ns = Server::steady_clock_now_ns();
        std::memcpy(&slot.msg, &msg, wire_bytes);
        uint64_t memcpy_end_ns = Server::steady_clock_now_ns();
        ctr_runtime_parity_slot_memcpy_ns.fetch_add(
            memcpy_end_ns - memcpy_begin_ns, std::memory_order_relaxed);
        ctr_runtime_parity_slot_memcpy_cnt.fetch_add(1,
                                                     std::memory_order_relaxed);
        slot.msg_ptr = &slot.msg;
        slot.msg_lkey = peer_rpc_mr->lkey;
        prepare_probe_parity_slot(slot);
        publish_probe_parity_batch_send_slot(slot);
    };
    publish_slot(*slot0, item.peer_idx0, item.msg0);
    publish_slot(*slot1, item.peer_idx1, item.msg1);
    (void)flush_ready_probe_parity_batch_send_slots(
        item.peer_idx0, preferred_shard(item.preferred_shard_idx0), 1);
    (void)flush_ready_probe_parity_batch_send_slots(
        item.peer_idx1, preferred_shard(item.preferred_shard_idx1), 1);
    return true;
}

inline bool Server::try_post_runtime_parity_batch_pair_storage_slot(
    size_t worker_idx, uint32_t slot_idx) {
    auto *item = get_runtime_parity_batch_pair_storage_slot(worker_idx, slot_idx);
    if (item == nullptr || peer_rpc_mr == nullptr) {
        return false;
    }
    if (peer_recovery_enabled() &&
        (!peer_is_alive(item->peer_idx0) || !peer_is_alive(item->peer_idx1)))
        return true;
    uint64_t try_begin_ns = Server::steady_clock_now_ns();
    if (item->first_try_time_ns == 0) {
        item->first_try_time_ns = try_begin_ns;
    }
    auto preferred_shard = [](uint16_t shard) -> size_t {
        return shard == std::numeric_limits<uint16_t>::max()
                   ? std::numeric_limits<size_t>::max()
                   : static_cast<size_t>(shard);
    };
    auto *slot0 = reserve_probe_parity_batch_send_slot(
        item->peer_idx0, true, preferred_shard(item->preferred_shard_idx0));
    if (slot0 == nullptr) {
        item->retry_count++;
        ctr_runtime_parity_pair_storage_try_fail.fetch_add(
            1, std::memory_order_relaxed);
        ctr_runtime_parity_batch_reserve_fail.fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
    auto *slot1 = reserve_probe_parity_batch_send_slot(
        item->peer_idx1, true, preferred_shard(item->preferred_shard_idx1));
    if (slot1 == nullptr) {
        item->retry_count++;
        ctr_runtime_parity_pair_storage_try_fail.fetch_add(
            1, std::memory_order_relaxed);
        ctr_runtime_parity_batch_reserve_fail.fetch_add(
            1, std::memory_order_relaxed);
        release_reserved_probe_parity_batch_send_slot(*slot0);
        return false;
    }
    auto publish_slot = [&](ProbeParityBatchSendSlot &slot, size_t peer_idx,
                            const rdma::ProbeParityBatchMessage &msg) {
        uint32_t wire_bytes = rdma::probe_parity_batch_wire_bytes(msg.span_count);
        slot.reset_message_binding();
        slot.last_flush_bookkeeping_ns = 0;
        slot.last_post_send_ns = 0;
        slot.peer_idx = static_cast<uint16_t>(peer_idx);
        uint64_t memcpy_begin_ns = Server::steady_clock_now_ns();
        std::memcpy(&slot.msg, &msg, wire_bytes);
        uint64_t memcpy_end_ns = Server::steady_clock_now_ns();
        ctr_runtime_parity_slot_memcpy_ns.fetch_add(
            memcpy_end_ns - memcpy_begin_ns, std::memory_order_relaxed);
        ctr_runtime_parity_slot_memcpy_cnt.fetch_add(1,
                                                     std::memory_order_relaxed);
        slot.msg_ptr = &slot.msg;
        slot.msg_lkey = peer_rpc_mr->lkey;
        prepare_probe_parity_slot(slot);
        publish_probe_parity_batch_send_slot(slot);
    };
    publish_slot(*slot0, item->peer_idx0, item->msg0);
    publish_slot(*slot1, item->peer_idx1, item->msg1);
    (void)flush_ready_probe_parity_batch_send_slots(
        item->peer_idx0, preferred_shard(item->preferred_shard_idx0), 1);
    (void)flush_ready_probe_parity_batch_send_slots(
        item->peer_idx1, preferred_shard(item->preferred_shard_idx1), 1);
    uint64_t success_time_ns = Server::steady_clock_now_ns();
    ctr_runtime_parity_pair_storage_try_success.fetch_add(
        1, std::memory_order_relaxed);
    if (item->pending_enqueue_time_ns != 0 &&
        success_time_ns >= item->pending_enqueue_time_ns) {
        uint64_t residency_ns =
            success_time_ns - item->pending_enqueue_time_ns;
        ctr_runtime_parity_pair_storage_pending_residency_ns.fetch_add(
            residency_ns, std::memory_order_relaxed);
        ctr_runtime_parity_pair_storage_pending_residency_cnt.fetch_add(
            1, std::memory_order_relaxed);
        update_peak(ctr_runtime_parity_pair_storage_pending_residency_max_ns,
                    residency_ns);
        ctr_runtime_parity_pair_storage_retry_sum.fetch_add(
            item->retry_count, std::memory_order_relaxed);
        ctr_runtime_parity_pair_storage_retry_cnt.fetch_add(
            1, std::memory_order_relaxed);
        update_peak(ctr_runtime_parity_pair_storage_retry_max,
                    item->retry_count);
    } else if (item->first_try_time_ns != 0 &&
               success_time_ns >= item->first_try_time_ns) {
        uint64_t direct_ns = success_time_ns - item->first_try_time_ns;
        ctr_runtime_parity_pair_storage_direct_success_ns.fetch_add(
            direct_ns, std::memory_order_relaxed);
        ctr_runtime_parity_pair_storage_direct_success_cnt.fetch_add(
            1, std::memory_order_relaxed);
        update_peak(ctr_runtime_parity_pair_storage_direct_success_max_ns,
                    direct_ns);
    }
    return true;
}

inline bool Server::enqueue_runtime_parity_batch_pair_pending_for_worker(
    size_t worker_idx, size_t peer_idx0, const rdma::ProbeParityBatchMessage &msg0,
    size_t preferred_shard_idx0, size_t peer_idx1,
    const rdma::ProbeParityBatchMessage &msg1, size_t preferred_shard_idx1) {
    if (peer_idx0 >= peer_endpoint_count || peer_idx1 >= peer_endpoint_count ||
        peer_idx0 == local_server_index || peer_idx1 == local_server_index) {
        return false;
    }
    if (peer_recovery_enabled() &&
        (!peer_is_alive(peer_idx0) || !peer_is_alive(peer_idx1)))
        return false;
    if (worker_idx < runtime_parity_batch_pair_storage_by_worker.size()) {
        uint32_t slot_idx = 0;
        RuntimeParityBatchPairStorageSlot *slot = nullptr;
        if (acquire_runtime_parity_batch_pair_storage_slot(worker_idx, slot_idx,
                                                           slot)) {
            slot->peer_idx0 = peer_idx0;
            slot->peer_idx1 = peer_idx1;
            slot->preferred_shard_idx0 =
                preferred_shard_idx0 == std::numeric_limits<size_t>::max()
                    ? std::numeric_limits<uint16_t>::max()
                    : static_cast<uint16_t>(preferred_shard_idx0);
            slot->preferred_shard_idx1 =
                preferred_shard_idx1 == std::numeric_limits<size_t>::max()
                    ? std::numeric_limits<uint16_t>::max()
                    : static_cast<uint16_t>(preferred_shard_idx1);
            uint64_t item_copy_begin_ns = Server::steady_clock_now_ns();
            slot->msg0 = msg0;
            slot->msg1 = msg1;
            uint64_t item_copy_end_ns = Server::steady_clock_now_ns();
            ctr_runtime_parity_item_copy_ns.fetch_add(
                item_copy_end_ns - item_copy_begin_ns,
                std::memory_order_relaxed);
            ctr_runtime_parity_item_copy_cnt.fetch_add(
                1, std::memory_order_relaxed);
            if (try_post_runtime_parity_batch_pair_storage_slot(worker_idx,
                                                                slot_idx)) {
                release_runtime_parity_batch_pair_storage_slot(worker_idx,
                                                               slot_idx);
                return true;
            }
            uint64_t queue_begin_ns = Server::steady_clock_now_ns();
            slot->pending_enqueue_time_ns = queue_begin_ns;
            uint64_t depth_after_enqueue =
                runtime_parity_batch_pair_pending_slot_ids_by_worker[worker_idx]
                    .size() +
                1;
            runtime_parity_batch_pair_pending_slot_ids_by_worker[worker_idx]
                .push_back(slot_idx);
            uint64_t queue_end_ns = Server::steady_clock_now_ns();
            ctr_runtime_parity_pair_storage_enqueue_depth_sum.fetch_add(
                depth_after_enqueue, std::memory_order_relaxed);
            ctr_runtime_parity_pair_storage_enqueue_depth_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_runtime_parity_pair_storage_enqueue_depth_max,
                        depth_after_enqueue);
            ctr_runtime_parity_pending_queue_ns.fetch_add(
                queue_end_ns - queue_begin_ns, std::memory_order_relaxed);
            ctr_runtime_parity_pending_queue_cnt.fetch_add(
                1, std::memory_order_relaxed);
            return true;
        }
    }
    return enqueue_runtime_parity_batch_pair_pending(
        peer_idx0, msg0, preferred_shard_idx0, peer_idx1, msg1,
        preferred_shard_idx1);
}

inline size_t Server::flush_runtime_parity_batch_pair_pending(size_t max_jobs) {
    if (max_jobs == 0) {
        return 0;
    }
    size_t initial_pending = 0;
    {
        std::lock_guard<std::mutex> lock(
            runtime_parity_batch_pair_pending_mutex);
        initial_pending = runtime_parity_batch_pair_pending_queue.size();
    }
    if (initial_pending == 0) {
        return 0;
    }
    size_t flushed = 0;
    size_t attempts = 0;
    while (flushed < max_jobs && attempts < initial_pending) {
        RuntimeParityBatchPairPendingItem item{};
        {
            std::lock_guard<std::mutex> lock(
                runtime_parity_batch_pair_pending_mutex);
            if (runtime_parity_batch_pair_pending_queue.empty()) {
                break;
            }
            item = std::move(runtime_parity_batch_pair_pending_queue.front());
            runtime_parity_batch_pair_pending_queue.pop_front();
        }
        if (!try_post_runtime_parity_batch_pair_pending_item(item)) {
            std::lock_guard<std::mutex> lock(
                runtime_parity_batch_pair_pending_mutex);
            runtime_parity_batch_pair_pending_queue.push_back(std::move(item));
        }
        else {
            flushed++;
        }
        attempts++;
    }
    return flushed;
}

inline size_t Server::flush_runtime_parity_batch_pair_pending_for_worker(
    size_t worker_idx, size_t max_jobs) {
    if (max_jobs == 0 ||
        worker_idx >= runtime_parity_batch_pair_pending_slot_ids_by_worker.size()) {
        return 0;
    }
    auto &queue =
        runtime_parity_batch_pair_pending_slot_ids_by_worker[worker_idx];
    size_t initial_pending = queue.size();
    if (initial_pending == 0) {
        return 0;
    }
    size_t flushed = 0;
    size_t attempts = 0;
    while (flushed < max_jobs && attempts < initial_pending && !queue.empty()) {
        uint64_t queue_pop_begin_ns = Server::steady_clock_now_ns();
        uint32_t slot_idx = queue.front();
        queue.pop_front();
        uint64_t queue_pop_end_ns = Server::steady_clock_now_ns();
        ctr_runtime_parity_pending_queue_ns.fetch_add(
            queue_pop_end_ns - queue_pop_begin_ns, std::memory_order_relaxed);
        ctr_runtime_parity_pending_queue_cnt.fetch_add(1,
                                                       std::memory_order_relaxed);
        if (try_post_runtime_parity_batch_pair_storage_slot(worker_idx,
                                                            slot_idx)) {
            release_runtime_parity_batch_pair_storage_slot(worker_idx, slot_idx);
            flushed++;
        } else {
            uint64_t queue_push_begin_ns = Server::steady_clock_now_ns();
            queue.push_back(slot_idx);
            uint64_t queue_push_end_ns = Server::steady_clock_now_ns();
            ctr_runtime_parity_pending_queue_ns.fetch_add(
                queue_push_end_ns - queue_push_begin_ns,
                std::memory_order_relaxed);
            ctr_runtime_parity_pending_queue_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
        attempts++;
    }
    return flushed;
}

inline size_t Server::runtime_parity_batch_index(size_t peer_idx,
                                                 size_t worker_idx) const {
    size_t lane_owner_count = std::max<size_t>(1, rpc_worker_count);
    return peer_idx * lane_owner_count + worker_idx;
}

inline void init_runtime_probe_parity_batch_message(
    rdma::ProbeParityBatchMessage &msg, uint16_t parity_idx, uint64_t wr_id) {
    msg.type = rdma::EC2PC_MSG_PARITY_PAYLOAD;
    msg.span_count = 0;
    msg.parity_idx = parity_idx;
    msg.flags = 0;
    msg.ack_count = 0;
    msg.reserved0 = 0;
    msg.status = rdma::kEC2PCStatusOK;
    msg.wr_id = wr_id;
    msg.send_time_ns = 0;
    msg.post_time_ns = 0;
}

inline bool Server::flush_runtime_parity_batch(size_t peer_idx,
                                               size_t worker_idx,
                                               bool force) {
    if (worker_idx >= std::max<size_t>(1, rpc_worker_count) ||
        peer_idx >= peer_endpoint_count || runtime_parity_batches.empty()) {
        return false;
    }
    size_t batch_idx = runtime_parity_batch_index(peer_idx, worker_idx);
    if (batch_idx >= runtime_parity_batches.size()) {
        return false;
    }
    auto &builder = runtime_parity_batches[batch_idx];
    if (builder.msg.span_count == 0) {
        return true;
    }
    if (!force && builder.first_enqueue_time_ns != 0) {
        uint64_t now_ns = steady_clock_now_ns();
        if (now_ns < builder.first_enqueue_time_ns ||
            now_ns - builder.first_enqueue_time_ns <
                kRuntimeParityBatchFlushNs) {
            return true;
        }
    }
    builder.msg.send_time_ns = steady_clock_now_ns();
    size_t preferred_shard_idx =
        worker_owned_peer_payload_shard(worker_idx, builder.msg.wr_id);
    if (!post_probe_parity_batch_message(peer_idx, builder.msg,
                                         preferred_shard_idx)) {
        return false;
    }
    builder.msg.span_count = 0;
    builder.msg.send_time_ns = 0;
    builder.msg.post_time_ns = 0;
    builder.msg.ack_count = 0;
    builder.peer_idx = peer_idx;
    builder.first_enqueue_time_ns = 0;
    return true;
}

inline size_t Server::flush_all_runtime_parity_batches_for_worker(
    size_t worker_idx, bool force) {
    if (worker_idx >= std::max<size_t>(1, rpc_worker_count) ||
        runtime_parity_batches.empty()) {
        return 0;
    }
    size_t flushed = 0;
    for (size_t peer_idx = 0; peer_idx < peer_endpoint_count; peer_idx++) {
        if (peer_idx == local_server_index) {
            continue;
        }
        size_t batch_idx = runtime_parity_batch_index(peer_idx, worker_idx);
        if (batch_idx >= runtime_parity_batches.size()) {
            continue;
        }
        if (runtime_parity_batches[batch_idx].msg.span_count == 0) {
            continue;
        }
        if (flush_runtime_parity_batch(peer_idx, worker_idx, force)) {
            flushed++;
        }
    }
    return flushed;
}

inline bool Server::append_runtime_parity_batch(size_t peer_idx,
                                                size_t worker_idx,
                                                uint64_t wr_id,
                                                uint16_t parity_idx,
                                                uint64_t target_offset,
                                                const uint8_t *payload) {
    uint8_t *reserved_payload = nullptr;
    bool flush_after_encode = false;
    if (!reserve_runtime_parity_batch_payload(peer_idx, worker_idx, wr_id,
                                              parity_idx, target_offset,
                                              &reserved_payload,
                                              &flush_after_encode)) {
        return false;
    }
    if (reserved_payload == nullptr || payload == nullptr) {
        return false;
    }
    std::memcpy(reserved_payload, payload, rdma::kEC2PCRpcPayloadBytes);
    if (flush_after_encode) {
        return flush_runtime_parity_batch(peer_idx, worker_idx, true);
    }
    return true;
}

inline bool Server::reserve_runtime_parity_batch_payload(
    size_t peer_idx, size_t worker_idx, uint64_t wr_id, uint16_t parity_idx,
    uint64_t target_offset, uint8_t **payload_out, bool *flush_after_encode) {
    if (worker_idx >= std::max<size_t>(1, rpc_worker_count) ||
        peer_idx >= peer_endpoint_count || peer_idx == local_server_index ||
        runtime_parity_batches.empty() || payload_out == nullptr ||
        flush_after_encode == nullptr) {
        return false;
    }
    size_t batch_idx = runtime_parity_batch_index(peer_idx, worker_idx);
    if (batch_idx >= runtime_parity_batches.size()) {
        return false;
    }
    auto &builder = runtime_parity_batches[batch_idx];
    if (builder.msg.span_count == 0) {
        builder.peer_idx = peer_idx;
        builder.first_enqueue_time_ns = steady_clock_now_ns();
        init_runtime_probe_parity_batch_message(builder.msg, parity_idx, wr_id);
    } else if (builder.msg.parity_idx != parity_idx) {
        if (!flush_runtime_parity_batch(peer_idx, worker_idx, true)) {
            return false;
        }
        builder.peer_idx = peer_idx;
        builder.first_enqueue_time_ns = steady_clock_now_ns();
        init_runtime_probe_parity_batch_message(builder.msg, parity_idx, wr_id);
    }
    if (builder.msg.span_count >= rdma::kProbeParityBatchMaxSpans) {
        if (!flush_runtime_parity_batch(peer_idx, worker_idx, true)) {
            return false;
        }
        builder.peer_idx = peer_idx;
        builder.first_enqueue_time_ns = steady_clock_now_ns();
        init_runtime_probe_parity_batch_message(builder.msg, parity_idx, wr_id);
    }
    size_t span_idx = builder.msg.span_count;
    builder.msg.wr_ids[span_idx] = wr_id;
    builder.msg.target_offsets[span_idx] = target_offset;
    *payload_out = builder.msg.payload[span_idx];
    builder.msg.span_count++;
    builder.msg.ack_count = builder.msg.span_count;
    *flush_after_encode =
        builder.msg.span_count >= rdma::kProbeParityBatchMaxSpans;
    return true;
}

inline Server::ProbeParityBatchSendSlot *
Server::reserve_probe_parity_batch_send_slot(size_t peer_idx,
                                             bool record_runtime_fail,
                                             size_t preferred_shard_idx) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx)) {
        if (record_runtime_fail)
            ctr_runtime_parity_reserve_fail_invalid.fetch_add(
                1, std::memory_order_relaxed);
        return nullptr;
    }
    if (peer_rpc_mr == nullptr || peer_idx >= peer_endpoint_count ||
        peer_idx == local_server_index) {
        if (record_runtime_fail) {
            ctr_runtime_parity_reserve_fail_invalid.fetch_add(
                1, std::memory_order_relaxed);
        }
        return nullptr;
    }
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    size_t shard_idx_value = preferred_shard_idx;
    if (shard_idx_value == std::numeric_limits<size_t>::max()) {
        if (probe_parity_batch_next_shards == nullptr) {
            if (record_runtime_fail) {
                ctr_runtime_parity_reserve_fail_no_shard.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return nullptr;
        }
        shard_idx_value =
            probe_parity_batch_next_shards[peer_idx].fetch_add(
                1, std::memory_order_relaxed) %
            payload_qp_count;
    } else {
        shard_idx_value %= payload_qp_count;
    }
    uint16_t shard_idx = static_cast<uint16_t>(shard_idx_value);
    size_t q_idx = probe_parity_batch_queue_index(peer_idx, shard_idx);
    if (q_idx >= probe_parity_batch_send_queues.size()) {
        if (record_runtime_fail) {
            ctr_runtime_parity_reserve_fail_bad_qidx.fetch_add(
                1, std::memory_order_relaxed);
        }
        return nullptr;
    }
    auto &queue = probe_parity_batch_send_queues[q_idx];
    if (queue.slots == nullptr || queue.depth == 0) {
        if (record_runtime_fail) {
            ctr_runtime_parity_reserve_fail_no_queue.fetch_add(
                1, std::memory_order_relaxed);
        }
        return nullptr;
    }
    if (probe_parity_batch_send_mutexes == nullptr) {
        if (record_runtime_fail) {
            ctr_runtime_parity_reserve_fail_no_mutex.fetch_add(
                1, std::memory_order_relaxed);
        }
        return nullptr;
    }
    auto reserve_lock_begin = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> queue_lock(
        probe_parity_batch_send_mutexes[q_idx]);
    auto reserve_lock_acquired = std::chrono::steady_clock::now();
    ctr_probe_parity_slot_reserve_lock_wait_ns.fetch_add(
        elapsed_ns(reserve_lock_begin, reserve_lock_acquired),
        std::memory_order_relaxed);
    ctr_probe_parity_slot_reserve_lock_wait_cnt.fetch_add(
        1, std::memory_order_relaxed);
    if (record_runtime_fail &&
        runtime_parity_batch_outstanding_by_qp != nullptr &&
        q_idx < peer_endpoint_count * payload_qp_count) {
        size_t outstanding_limit =
            configured_runtime_parity_batch_soft_limit();
        uint64_t outstanding =
            runtime_parity_batch_outstanding_by_qp[q_idx].load(
                std::memory_order_relaxed);
        if (outstanding >= outstanding_limit) {
            ctr_runtime_parity_reserve_fail_outstanding.fetch_add(
                1, std::memory_order_relaxed);
            ctr_runtime_parity_reserve_fail_outstanding_sum.fetch_add(
                outstanding, std::memory_order_relaxed);
            ctr_runtime_parity_reserve_fail_outstanding_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_runtime_parity_reserve_fail_outstanding_max,
                        outstanding);
            return nullptr;
        }
    }
    ProbeParityBatchSendSlot *picked = nullptr;
    size_t scan_steps = 0;
    for (size_t attempt = 0; attempt < queue.depth; attempt++) {
        scan_steps++;
        auto &slot = queue.slots[queue.next_slot];
        queue.next_slot = (queue.next_slot + 1) % queue.depth;
        bool expected = false;
        if (slot.in_use.compare_exchange_strong(expected, true,
                                                std::memory_order_acq_rel)) {
            picked = &slot;
            picked->state.store(kProbeBatchSlotBusy,
                                std::memory_order_release);
            picked->shard_idx = shard_idx;
            break;
        }
    }
    ctr_probe_parity_slot_reserve_scan_steps.fetch_add(
        scan_steps, std::memory_order_relaxed);
    ctr_probe_parity_slot_reserve_calls.fetch_add(
        1, std::memory_order_relaxed);
    if (picked == nullptr && record_runtime_fail) {
        ctr_runtime_parity_reserve_fail_no_free_slot.fetch_add(
            1, std::memory_order_relaxed);
    }
    return picked;
}

// Return a parity send slot to the free pool before it was posted.
inline void Server::release_reserved_probe_parity_batch_send_slot(
    ProbeParityBatchSendSlot &slot) {
    auto *batch_owner = slot.batch_owner;
    uint64_t batch_generation = slot.batch_owner_generation;
    slot.post_time_ns = 0;
    slot.last_flush_bookkeeping_ns = 0;
    slot.last_post_send_ns = 0;
    slot.state.store(kProbeBatchSlotFree, std::memory_order_release);
    slot.in_use.store(false, std::memory_order_release);
    slot.reset_message_binding();
    release_probe_parity_batch_owner(batch_owner, batch_generation);
}

// Try to reserve transport tokens for both runtime parity packets together.
inline bool Server::try_post_probe_parity_batch_pair_pending_item(
    ProbeParityBatchPairPendingItem &item) {
    auto abandon_pair = [&]() {
        release_probe_parity_batch_owner(
            item.batch_owner, item.batch_owner_generation);
        release_probe_parity_batch_owner(
            item.batch_owner, item.batch_owner_generation);
        item.msg0 = nullptr;
        item.msg1 = nullptr;
        item.batch_owner = nullptr;
        item.batch_owner_generation = 0;
        return true;
    };
    if (peer_recovery_enabled() &&
        (!peer_is_alive(item.peer_idx0) || !peer_is_alive(item.peer_idx1)))
        return abandon_pair();
    if (item.msg0 == nullptr || item.msg1 == nullptr ||
        item.batch_owner == nullptr || item.msg_lkey == 0) {
        return abandon_pair();
    }
    if (item.batch_owner_generation == 0 ||
        item.batch_owner->generation.load(std::memory_order_acquire) !=
            item.batch_owner_generation) {
        return abandon_pair();
    }
    uint64_t try_begin_ns = Server::steady_clock_now_ns();
    if (item.first_try_time_ns == 0) {
        item.first_try_time_ns = try_begin_ns;
    }
    auto *slot0 = reserve_probe_parity_batch_send_slot(
        item.peer_idx0, true, item.preferred_shard_idx0);
    if (slot0 == nullptr) {
        item.retry_count++;
        ctr_runtime_parity_ext_pair_try_fail.fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
    auto *slot1 = reserve_probe_parity_batch_send_slot(
        item.peer_idx1, true, item.preferred_shard_idx1);
    if (slot1 == nullptr) {
        item.retry_count++;
        ctr_runtime_parity_ext_pair_try_fail.fetch_add(
            1, std::memory_order_relaxed);
        release_reserved_probe_parity_batch_send_slot(*slot0);
        return false;
    }
    auto bind_slot = [&](ProbeParityBatchSendSlot &slot, size_t peer_idx,
                         rdma::ProbeParityBatchMessage *msg) {
        slot.reset_message_binding();
        slot.last_flush_bookkeeping_ns = 0;
        slot.last_post_send_ns = 0;
        slot.peer_idx = static_cast<uint16_t>(peer_idx);
        slot.msg_ptr = msg;
        slot.msg_lkey = item.msg_lkey;
        slot.batch_owner = item.batch_owner;
        slot.batch_owner_generation = item.batch_owner_generation;
        prepare_probe_parity_slot(slot);
    };
    bind_slot(*slot0, item.peer_idx0, item.msg0);
    bind_slot(*slot1, item.peer_idx1, item.msg1);
    publish_probe_parity_batch_send_slot(*slot0);
    publish_probe_parity_batch_send_slot(*slot1);
    (void)flush_ready_probe_parity_batch_send_slots(
        item.peer_idx0, item.preferred_shard_idx0, 1);
    (void)flush_ready_probe_parity_batch_send_slots(
        item.peer_idx1, item.preferred_shard_idx1, 1);
    uint64_t success_time_ns = Server::steady_clock_now_ns();
    ctr_runtime_parity_ext_pair_try_success.fetch_add(
        1, std::memory_order_relaxed);
    if (item.pending_enqueue_time_ns != 0 &&
        success_time_ns >= item.pending_enqueue_time_ns) {
        uint64_t residency_ns =
            success_time_ns - item.pending_enqueue_time_ns;
        ctr_runtime_parity_ext_pair_pending_residency_ns.fetch_add(
            residency_ns, std::memory_order_relaxed);
        ctr_runtime_parity_ext_pair_pending_residency_cnt.fetch_add(
            1, std::memory_order_relaxed);
        update_peak(ctr_runtime_parity_ext_pair_pending_residency_max_ns,
                    residency_ns);
        ctr_runtime_parity_ext_pair_retry_sum.fetch_add(
            item.retry_count, std::memory_order_relaxed);
        ctr_runtime_parity_ext_pair_retry_cnt.fetch_add(
            1, std::memory_order_relaxed);
        update_peak(ctr_runtime_parity_ext_pair_retry_max, item.retry_count);
    } else if (item.first_try_time_ns != 0 &&
               success_time_ns >= item.first_try_time_ns) {
        uint64_t direct_ns = success_time_ns - item.first_try_time_ns;
        ctr_runtime_parity_ext_pair_direct_success_ns.fetch_add(
            direct_ns, std::memory_order_relaxed);
        ctr_runtime_parity_ext_pair_direct_success_cnt.fetch_add(
            1, std::memory_order_relaxed);
        update_peak(ctr_runtime_parity_ext_pair_direct_success_max_ns,
                    direct_ns);
    }
    return true;
}

// Retry pending runtime parity fanout jobs until one blocks again.
inline size_t Server::flush_probe_parity_batch_pair_pending(size_t max_jobs) {
    size_t flushed = 0;
    while (flushed < max_jobs) {
        ProbeParityBatchPairPendingItem item{};
        bool have_item = false;
        {
            std::lock_guard<std::mutex> lock(
                probe_parity_batch_pair_pending_mutex);
            if (!probe_parity_batch_pair_pending_queue.empty()) {
                item = probe_parity_batch_pair_pending_queue.front();
                probe_parity_batch_pair_pending_queue.pop_front();
                have_item = true;
            }
        }
        if (!have_item) {
            break;
        }
        if (try_post_probe_parity_batch_pair_pending_item(item)) {
            flushed++;
            continue;
        }
        std::lock_guard<std::mutex> lock(probe_parity_batch_pair_pending_mutex);
        probe_parity_batch_pair_pending_queue.push_front(item);
        break;
    }
    return flushed;
}

inline bool Server::post_reserved_probe_parity_batch_send_slot(
    ProbeParityBatchSendSlot &slot) {
    if (peer_recovery_enabled() && !peer_is_alive(slot.peer_idx)) {
        release_reserved_probe_parity_batch_send_slot(slot);
        return false;
    }
    if (peer_rpc_mr == nullptr || slot.peer_idx >= peer_endpoint_count ||
        slot.shard_idx >= peer_payload_qp_count) {
        release_reserved_probe_parity_batch_send_slot(slot);
        return false;
    }
    size_t q_idx = peer_payload_qp_index(slot.peer_idx, slot.shard_idx);
    if (q_idx >= peer_payload_qps.size()) {
        release_reserved_probe_parity_batch_send_slot(slot);
        return false;
    }
    auto &qp = peer_payload_qps[q_idx];
    if (qp.queue_pair == nullptr) {
        release_reserved_probe_parity_batch_send_slot(slot);
        return false;
    }
    auto *msg = slot.active_msg();
    if (msg == nullptr) {
        release_reserved_probe_parity_batch_send_slot(slot);
        return false;
    }
    ibv_sge sge = {
        .addr = reinterpret_cast<uint64_t>(msg),
        .length = rdma::probe_parity_batch_wire_bytes(msg->span_count),
        .lkey = slot.msg_lkey != 0
                    ? slot.msg_lkey
                    : (peer_rpc_mr != nullptr ? peer_rpc_mr->lkey : 0),
    };
    ibv_send_wr wr{};
    wr.wr_id = reinterpret_cast<uint64_t>(&slot);
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    ibv_send_wr *bad = nullptr;
    uint64_t post_send_begin_ns = Server::steady_clock_now_ns();
    int ret = ibv_post_send(qp.queue_pair, &wr, &bad);
    slot.last_post_send_ns =
        Server::steady_clock_now_ns() - post_send_begin_ns;
    if (ret != 0) {
        if ((msg->flags & rdma::kEC2PCFlagProbeBatch) == 0) {
            ctr_runtime_parity_batch_post_fail.fetch_add(
                1, std::memory_order_relaxed);
        }
        const bool endpoint_dead =
            peer_recovery_enabled() && !peer_is_alive(slot.peer_idx);
        if (ret == ENOMEM && !endpoint_dead) {
            // Keep the BUSY binding intact.  The ready-flush caller may
            // safely return it to READY and retry without touching a freed
            // external message.
            return false;
        }
        if (endpoint_dead) {
            release_reserved_probe_parity_batch_send_slot(slot);
            return false;
        }
        ERROR("server probe parity ibv_post_send failed");
        return false;
    }
    uint64_t post_time_ns = Server::steady_clock_now_ns();
    slot.post_time_ns = post_time_ns;
    msg->post_time_ns = post_time_ns;
    if ((msg->reserved0 & rdma::kEC2PCFlagCompactReq) != 0) {
        ctr_compact_peer_post_ns.fetch_add(
            slot.last_post_send_ns, std::memory_order_relaxed);
        ctr_compact_peer_post_cnt.fetch_add(1, std::memory_order_relaxed);
        ctr_compact_peer_payload_send_bytes.fetch_add(
            static_cast<uint64_t>(sge.length), std::memory_order_relaxed);
    }
    if ((msg->flags & rdma::kEC2PCFlagProbeBatch) == 0) {
        note_server_ec2pc_parity_payload_posted(slot.peer_idx, *msg,
                                                post_time_ns);
    }
    slot.state.store(kProbeBatchSlotInflight, std::memory_order_release);
    if ((msg->flags & rdma::kEC2PCFlagProbeBatch) == 0 &&
        runtime_parity_batch_outstanding_by_qp != nullptr) {
        runtime_parity_batch_outstanding_by_qp[q_idx].fetch_add(
            1, std::memory_order_relaxed);
    }
    ctr_peer_enq_payload.fetch_add(msg->span_count,
                                   std::memory_order_relaxed);
    ctr_peer_flush_posts.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline bool Server::enqueue_probe_parity_batch_pending(
    size_t peer_idx, const rdma::ProbeParityBatchMessage &msg,
    size_t preferred_shard_idx) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (peer_idx >= peer_endpoint_count ||
        probe_parity_batch_pending_mutexes == nullptr ||
        probe_parity_batch_pending_space_cvs == nullptr) {
        return false;
    }
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    uint16_t chosen_shard = 0;
    if (preferred_shard_idx == std::numeric_limits<size_t>::max()) {
        if (probe_parity_batch_next_shards == nullptr) {
            return false;
        }
        chosen_shard = static_cast<uint16_t>(
            probe_parity_batch_next_shards[peer_idx].fetch_add(
                1, std::memory_order_relaxed) %
            payload_qp_count);
    } else {
        chosen_shard =
            static_cast<uint16_t>(preferred_shard_idx % payload_qp_count);
    }
    size_t q_idx = probe_parity_batch_queue_index(peer_idx, chosen_shard);
    if (q_idx >= probe_parity_batch_send_queues.size()) {
        return false;
    }
    auto &queue = probe_parity_batch_send_queues[q_idx];
    if (queue.pending_items == nullptr || queue.pending_capacity == 0) {
        return false;
    }
    std::unique_lock<std::mutex> lock(
        probe_parity_batch_pending_mutexes[q_idx]);
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    while (queue.pending_count >= queue.pending_capacity) {
        probe_parity_batch_pending_space_cvs[q_idx].wait_for(
            lock, std::chrono::microseconds(50));
    }
    auto &item = queue.pending_items[queue.pending_tail];
    item.preferred_shard_idx = chosen_shard;
    item.msg_ptr = nullptr;
    item.msg_lkey = 0;
    item.batch_owner = nullptr;
    item.msg = msg;
    queue.pending_tail = (queue.pending_tail + 1) % queue.pending_capacity;
    queue.pending_count++;
    return true;
}

inline bool Server::enqueue_probe_parity_batch_pending_external(
    size_t peer_idx, rdma::ProbeParityBatchMessage *msg, uint32_t msg_lkey,
    BatchRpcRecvSlot *batch_owner, uint64_t batch_owner_generation,
    size_t preferred_shard_idx) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg == nullptr || msg_lkey == 0 || batch_owner == nullptr ||
        peer_idx >= peer_endpoint_count ||
        probe_parity_batch_pending_mutexes == nullptr ||
        probe_parity_batch_pending_space_cvs == nullptr) {
        return false;
    }
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    uint16_t chosen_shard = 0;
    if (preferred_shard_idx == std::numeric_limits<size_t>::max()) {
        if (probe_parity_batch_next_shards == nullptr) {
            return false;
        }
        chosen_shard = static_cast<uint16_t>(
            probe_parity_batch_next_shards[peer_idx].fetch_add(
                1, std::memory_order_relaxed) %
            payload_qp_count);
    } else {
        chosen_shard =
            static_cast<uint16_t>(preferred_shard_idx % payload_qp_count);
    }
    size_t q_idx = probe_parity_batch_queue_index(peer_idx, chosen_shard);
    if (q_idx >= probe_parity_batch_send_queues.size()) {
        return false;
    }
    auto &queue = probe_parity_batch_send_queues[q_idx];
    if (queue.pending_items == nullptr || queue.pending_capacity == 0) {
        return false;
    }
    std::unique_lock<std::mutex> lock(
        probe_parity_batch_pending_mutexes[q_idx]);
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    while (queue.pending_count >= queue.pending_capacity) {
        probe_parity_batch_pending_space_cvs[q_idx].wait_for(
            lock, std::chrono::microseconds(50));
    }
    auto &item = queue.pending_items[queue.pending_tail];
    item.preferred_shard_idx = chosen_shard;
    item.msg_ptr = msg;
    item.msg_lkey = msg_lkey;
    item.batch_owner = batch_owner;
    item.batch_owner_generation = batch_owner_generation;
    item.msg = rdma::ProbeParityBatchMessage{};
    queue.pending_tail = (queue.pending_tail + 1) % queue.pending_capacity;
    queue.pending_count++;
    return true;
}

inline size_t Server::fill_probe_parity_batch_send_slots_from_pending(
    size_t peer_idx, size_t preferred_shard_idx, size_t max_fill) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx)) {
        drop_unposted_peer_work(peer_idx);
        return 0;
    }
    if (peer_idx >= peer_endpoint_count || max_fill == 0 ||
        probe_parity_batch_pending_mutexes == nullptr) {
        return 0;
    }
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    uint16_t target_shard = static_cast<uint16_t>(
        preferred_shard_idx == std::numeric_limits<size_t>::max()
            ? 0
            : (preferred_shard_idx % payload_qp_count));
    size_t q_idx = probe_parity_batch_queue_index(peer_idx, target_shard);
    if (q_idx >= probe_parity_batch_send_queues.size()) {
        return 0;
    }
    auto &queue = probe_parity_batch_send_queues[q_idx];
    if (queue.pending_items == nullptr || queue.pending_capacity == 0) {
        return 0;
    }
    size_t filled = 0;
    while (filled < max_fill) {
        // Empty polls do not construct/zero a 64KiB payload. Construct the
        // optional directly from the queued item while retaining the same
        // mutex, slot reset, FIFO and notification ownership as before.
        auto pending = [&] {
            std::lock_guard<std::mutex> lock(
                probe_parity_batch_pending_mutexes[q_idx]);
            return carbink::take_pending_item_locked(
                queue.pending_items, queue.pending_capacity,
                queue.pending_head, queue.pending_count);
        }();
        if (!pending) break;
        probe_parity_batch_pending_space_cvs[q_idx].notify_one();
        auto &item = *pending;
        auto *slot = reserve_probe_parity_batch_send_slot(
            peer_idx, false, target_shard);
        if (slot == nullptr) {
            std::lock_guard<std::mutex> lock(
                probe_parity_batch_pending_mutexes[q_idx]);
            queue.pending_items[queue.pending_tail] = item;
            queue.pending_tail =
                (queue.pending_tail + 1) % queue.pending_capacity;
            queue.pending_count++;
            break;
        }
        slot->reset_message_binding();
        slot->last_flush_bookkeeping_ns = 0;
        slot->last_post_send_ns = 0;
        if (item.msg_ptr != nullptr) {
            slot->msg_ptr = item.msg_ptr;
            slot->msg_lkey = item.msg_lkey;
            slot->batch_owner = item.batch_owner;
            slot->batch_owner_generation = item.batch_owner_generation;
        } else {
            uint32_t wire_bytes =
                rdma::probe_parity_batch_wire_bytes(item.msg.span_count);
            uint64_t memcpy_begin_ns = Server::steady_clock_now_ns();
            std::memcpy(&slot->msg, &item.msg, wire_bytes);
            uint64_t memcpy_end_ns = Server::steady_clock_now_ns();
            ctr_runtime_parity_slot_memcpy_ns.fetch_add(
                memcpy_end_ns - memcpy_begin_ns, std::memory_order_relaxed);
            ctr_runtime_parity_slot_memcpy_cnt.fetch_add(
                1, std::memory_order_relaxed);
            slot->msg_ptr = &slot->msg;
            slot->msg_lkey = peer_rpc_mr != nullptr ? peer_rpc_mr->lkey : 0;
        }
        prepare_probe_parity_slot(*slot);
        publish_probe_parity_batch_send_slot(*slot);
        filled++;
    }
    return filled;
}

inline void Server::publish_probe_parity_batch_send_slot(
    ProbeParityBatchSendSlot &slot) {
    auto *msg = slot.active_msg();
    if (msg == nullptr) {
        return;
    }
    msg->send_time_ns = steady_clock_now_ns();
    msg->post_time_ns = 0;
    msg->reserved0 &= rdma::kEC2PCFlagCompactReq;
    if ((msg->flags & rdma::kEC2PCFlagProbeBatch) == 0 &&
        runtime_parity_batch_ready_by_qp != nullptr &&
        slot.peer_idx < peer_endpoint_count &&
        slot.shard_idx < peer_payload_qp_count) {
        size_t q_idx = peer_payload_qp_index(slot.peer_idx, slot.shard_idx);
        if (q_idx < peer_endpoint_count *
                        std::max<size_t>(1, peer_payload_qp_count)) {
            runtime_parity_batch_ready_by_qp[q_idx].fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    slot.state.store(kProbeBatchSlotReady, std::memory_order_release);
    size_t q_idx = probe_parity_batch_queue_index(slot.peer_idx, slot.shard_idx);
    enqueue_probe_parity_ready_slot(
        q_idx, slot.slot_index,
        slot.bind_generation.load(std::memory_order_acquire));
}

inline size_t Server::flush_ready_probe_parity_batch_send_slots(
    size_t peer_idx, size_t preferred_shard_idx, size_t max_posts) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return 0;
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    size_t target_shard =
        preferred_shard_idx == std::numeric_limits<size_t>::max()
            ? 0
            : (preferred_shard_idx % payload_qp_count);
    size_t q_idx = probe_parity_batch_queue_index(peer_idx, target_shard);
    if (q_idx >= probe_parity_batch_send_queues.size() || max_posts == 0) {
        return 0;
    }
    auto &queue = probe_parity_batch_send_queues[q_idx];
    if (queue.slots == nullptr || queue.depth == 0) {
        return 0;
    }
    (void)fill_probe_parity_batch_send_slots_from_pending(peer_idx, target_shard,
                                                          max_posts);
    size_t posted = 0;
    size_t scanned = 0;
    size_t ready_gate_hits = 0;
    size_t shard_mismatch = 0;
    bool saw_runtime_msg = false;
    size_t attempt_budget = 0;
    if (q_idx < probe_parity_batch_ready_queues.size() &&
        probe_parity_batch_ready_mutexes != nullptr) {
        std::lock_guard<std::mutex> lock(
            probe_parity_batch_ready_mutexes[q_idx]);
        attempt_budget = probe_parity_batch_ready_queues[q_idx].size();
    }
    for (size_t attempt = 0; attempt < attempt_budget && posted < max_posts;
         attempt++) {
        ProbeParityBatchReadyItem ready_item{};
        uint64_t slot_dispatch_begin_ns = Server::steady_clock_now_ns();
        bool have_slot = try_dequeue_probe_parity_ready_slot(q_idx, ready_item);
        if (!have_slot) {
            break;
        }
        scanned++;
        auto &slot = queue.slots[ready_item.slot_index];
        uint64_t bind_seq = slot.bind_generation.load(std::memory_order_acquire);
        if (bind_seq != ready_item.bind_seq) {
            continue;
        }
        uint8_t expected = kProbeBatchSlotReady;
        if (!slot.state.compare_exchange_strong(expected, kProbeBatchSlotBusy,
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed)) {
            continue;
        }
        if (slot.shard_idx != target_shard) {
            shard_mismatch++;
            slot.state.store(kProbeBatchSlotReady, std::memory_order_release);
            enqueue_probe_parity_ready_slot(q_idx, slot.slot_index, bind_seq,
                                            ready_item.ready_enqueue_time_ns);
            continue;
        }
        size_t q_idx = peer_payload_qp_index(peer_idx, slot.shard_idx);
        auto *msg = slot.active_msg();
        if (msg == nullptr) {
            slot.state.store(kProbeBatchSlotFree, std::memory_order_release);
            slot.in_use.store(false, std::memory_order_release);
            slot.reset_message_binding();
            continue;
        }
        bool runtime_msg = (msg->flags & rdma::kEC2PCFlagProbeBatch) == 0;
        saw_runtime_msg = saw_runtime_msg || runtime_msg;
        if (runtime_msg && runtime_parity_batch_ready_by_qp != nullptr &&
            q_idx < peer_endpoint_count * payload_qp_count) {
            uint64_t cur_ready =
                runtime_parity_batch_ready_by_qp[q_idx].load(
                    std::memory_order_relaxed);
            while (cur_ready != 0 &&
                   !runtime_parity_batch_ready_by_qp[q_idx]
                        .compare_exchange_weak(cur_ready, cur_ready - 1,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
            }
        }
        if (runtime_msg &&
            runtime_parity_batch_outstanding_by_qp != nullptr &&
            q_idx < peer_endpoint_count * payload_qp_count) {
            size_t outstanding_limit =
                configured_runtime_parity_batch_soft_limit();
            uint64_t outstanding =
                runtime_parity_batch_outstanding_by_qp[q_idx].load(
                    std::memory_order_relaxed);
            if (outstanding >= outstanding_limit) {
                ready_gate_hits++;
                uint64_t ready_depth = 0;
                if (runtime_parity_batch_ready_by_qp != nullptr) {
                    ready_depth =
                        runtime_parity_batch_ready_by_qp[q_idx].load(
                            std::memory_order_relaxed) +
                        1;
                }
                ctr_runtime_parity_batch_ready_gate_fail.fetch_add(
                    1, std::memory_order_relaxed);
                ctr_runtime_parity_batch_ready_gate_slots.fetch_add(
                    ready_depth, std::memory_order_relaxed);
                ctr_runtime_parity_ready_gate_outstanding_sum.fetch_add(
                    outstanding, std::memory_order_relaxed);
                ctr_runtime_parity_ready_gate_outstanding_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                update_peak(ctr_runtime_parity_ready_gate_outstanding_max,
                            outstanding);
                uint64_t now_ns = Server::steady_clock_now_ns();
                if (ready_item.ready_enqueue_time_ns != 0 &&
                    now_ns >= ready_item.ready_enqueue_time_ns) {
                    uint64_t ready_wait_ns =
                        now_ns - ready_item.ready_enqueue_time_ns;
                    ctr_runtime_parity_ready_to_gate_fail_ns.fetch_add(
                        ready_wait_ns, std::memory_order_relaxed);
                    ctr_runtime_parity_ready_to_gate_fail_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    update_peak(ctr_runtime_parity_ready_to_gate_fail_max_ns,
                                ready_wait_ns);
                }
                if (runtime_parity_batch_ready_by_qp != nullptr) {
                    runtime_parity_batch_ready_by_qp[q_idx].fetch_add(
                        1, std::memory_order_relaxed);
                }
                slot.state.store(kProbeBatchSlotReady,
                                 std::memory_order_release);
                enqueue_probe_parity_ready_slot(q_idx, slot.slot_index, bind_seq,
                                                ready_item.ready_enqueue_time_ns);
                break;
            }
        }
        if (!post_reserved_probe_parity_batch_send_slot(slot)) {
            const bool binding_retained =
                slot.in_use.load(std::memory_order_acquire) &&
                slot.state.load(std::memory_order_acquire) ==
                    kProbeBatchSlotBusy &&
                slot.bind_generation.load(std::memory_order_acquire) ==
                    bind_seq &&
                slot.active_msg() != nullptr;
            if (binding_retained) {
                auto *retry_msg = slot.active_msg();
                if (runtime_msg &&
                    runtime_parity_batch_ready_by_qp != nullptr &&
                    q_idx < peer_endpoint_count * payload_qp_count) {
                    runtime_parity_batch_ready_by_qp[q_idx].fetch_add(
                        1, std::memory_order_relaxed);
                }
                slot.state.store(kProbeBatchSlotReady,
                                 std::memory_order_release);
                slot.post_time_ns = 0;
                retry_msg->post_time_ns = 0;
                enqueue_probe_parity_ready_slot(
                    q_idx, slot.slot_index, bind_seq,
                    ready_item.ready_enqueue_time_ns);
            }
            break;
        }
        uint64_t post_time_ns = slot.post_time_ns;
        if (runtime_msg && ready_item.ready_enqueue_time_ns != 0 &&
            post_time_ns >= ready_item.ready_enqueue_time_ns) {
            uint64_t ready_wait_ns =
                post_time_ns - ready_item.ready_enqueue_time_ns;
            ctr_runtime_parity_ready_to_post_ns.fetch_add(
                ready_wait_ns, std::memory_order_relaxed);
            ctr_runtime_parity_ready_to_post_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_runtime_parity_ready_to_post_max_ns,
                        ready_wait_ns);
        }
        uint64_t slot_dispatch_total_ns =
            Server::steady_clock_now_ns() - slot_dispatch_begin_ns;
        slot.last_flush_bookkeeping_ns =
            slot_dispatch_total_ns > slot.last_post_send_ns
                ? (slot_dispatch_total_ns - slot.last_post_send_ns)
                : 0;
        posted++;
    }
    (void)saw_runtime_msg;
    return posted;
}

inline void Server::post_peer_payload_recv_slot(PeerPayloadRecvSlot &slot) {
    if (peer_recovery_enabled() && !peer_is_alive(slot.peer_idx))
        return;
    if (peer_rpc_mr == nullptr || !peer_payload_transport_ready() ||
        slot.peer_idx >= peer_endpoint_count ||
        slot.shard_idx >= peer_payload_qp_count) {
        ERROR("server post_peer_payload_recv_slot: peer payload transport not ready");
    }
    size_t q_idx = peer_payload_qp_index(slot.peer_idx, slot.shard_idx);
    if (q_idx >= peer_payload_qps.size() ||
        peer_payload_qps[q_idx].queue_pair == nullptr) {
        ERROR("server post_peer_payload_recv_slot: invalid payload qp");
    }
    ibv_sge sge = {
        .addr = reinterpret_cast<uint64_t>(slot.bytes.data()),
        .length = static_cast<uint32_t>(rdma::kPeerRpcRecvBytes),
        .lkey = peer_rpc_mr->lkey,
    };
    ibv_recv_wr wr = {
        .wr_id = reinterpret_cast<uint64_t>(&slot),
        .next = nullptr,
        .sg_list = &sge,
        .num_sge = 1,
    };
    ibv_recv_wr *bad = nullptr;
    int ret = ibv_post_recv(peer_payload_qps[q_idx].queue_pair, &wr, &bad);
    if (ret != 0) {
        ERROR("server post_peer_payload_recv_slot failed");
    }
    if (peer_payload_recv_credits != nullptr &&
        q_idx < peer_payload_recv_credit_count) {
        peer_payload_recv_credits[q_idx].fetch_add(1, std::memory_order_relaxed);
    }
}

inline void Server::release_peer_send_slot(PeerRpcSendSlot &slot) {
    if (slot.magic != 0xEC2C2002u || slot.queue_index >= peer_rpc_queues.size()) {
        return;
    }
    auto &queue = peer_rpc_queues[slot.queue_index];
    std::lock_guard<std::mutex> lock(peer_send_mutexes[slot.queue_index]);
    bool expected = true;
    if (!slot.in_use.compare_exchange_strong(expected, false,
                                             std::memory_order_acq_rel)) {
        return;
    }
    if (queue.free_count >= kPeerRpcDepth) {
        ERROR("server release_peer_send_slot: free-list overflow");
    }
    queue.free_slot_indices[queue.free_tail] = slot.slot_index;
    queue.free_tail = (queue.free_tail + 1) % kPeerRpcDepth;
    queue.free_count++;
}

inline size_t Server::reclaim_inflight_peer_send_slots_upto(
    size_t peer_idx, PeerRpcSendSlot *completed_slot) {
    if (completed_slot == nullptr || peer_idx >= inflight_peer_send_slots.size()) {
        return 0;
    }
    auto &queue = peer_rpc_queues[peer_idx];
    auto release_under_send_lock = [&](PeerRpcSendSlot *slot) -> bool {
        if (slot == nullptr || slot->magic != 0xEC2C2002u ||
            slot->queue_index != peer_idx) {
            return false;
        }
        bool expected = true;
        if (!slot->in_use.compare_exchange_strong(expected, false,
                                                 std::memory_order_acq_rel)) {
            return false;
        }
        if (queue.free_count >= kPeerRpcDepth) {
            ERROR("server reclaim_inflight_peer_send_slots_upto: free-list overflow");
        }
        queue.free_slot_indices[queue.free_tail] = slot->slot_index;
        queue.free_tail = (queue.free_tail + 1) % kPeerRpcDepth;
        queue.free_count++;
        return true;
    };

    size_t reclaimed = 0;
    bool found = false;
    auto send_lock_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> send_lock(peer_send_mutexes[peer_idx]);
    auto send_lock_acquired = std::chrono::steady_clock::now();
    ctr_peer_send_lock_reclaim_wait_ns.fetch_add(
        elapsed_ns(send_lock_begin, send_lock_acquired),
        std::memory_order_relaxed);
    ctr_peer_send_lock_reclaim_ops.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> inflight_lock(
        inflight_peer_send_slot_mutexes[peer_idx]);
    auto &inflight = inflight_peer_send_slots[peer_idx];
    while (!inflight.empty()) {
        PeerRpcSendSlot *slot = inflight.front();
        inflight.pop_front();
        if (release_under_send_lock(slot)) {
            reclaimed++;
        }
        if (slot == completed_slot) {
            found = true;
            break;
        }
    }
    if (!found && release_under_send_lock(completed_slot)) {
        reclaimed++;
    }
    auto send_lock_release = std::chrono::steady_clock::now();
    ctr_peer_send_lock_reclaim_hold_ns.fetch_add(
        elapsed_ns(send_lock_acquired, send_lock_release),
        std::memory_order_relaxed);
    return reclaimed;
}

inline size_t Server::try_post_peer_send_batch_locked(
    size_t peer_idx, std::deque<rdma::EC2PCRpcMessage> &dq, size_t max_posts) {
    if (peer_idx >= peer_rpc_queues.size() || peer_idx >= peer_qps.size() ||
        peer_idx == local_server_index || max_posts == 0 || dq.empty()) {
        return 0;
    }
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx)) {
        return 0;
    }

    auto &qp = peer_qps[peer_idx];
    if (qp.queue_pair == nullptr) {
        return 0;
    }

    const size_t batch_cap = std::min(
        {max_posts, dq.size(), static_cast<size_t>(kPeerPostSendBatch)});
    if (batch_cap == 0) {
        return 0;
    }

    auto &queue = peer_rpc_queues[peer_idx];
    std::array<PeerRpcSendSlot *, kPeerPostSendBatch> slots{};
    std::array<ibv_sge, kPeerPostSendBatch> sges{};
    std::array<ibv_send_wr, kPeerPostSendBatch> wrs{};

    size_t prepared = 0;
    size_t posted = 0;
    {
        auto send_lock_begin = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> send_lock(peer_send_mutexes[peer_idx]);
        auto send_lock_acquired = std::chrono::steady_clock::now();
        ctr_peer_send_lock_batch_wait_ns.fetch_add(
            elapsed_ns(send_lock_begin, send_lock_acquired),
            std::memory_order_relaxed);
        ctr_peer_send_lock_batch_ops.fetch_add(1, std::memory_order_relaxed);
        for (; prepared < batch_cap; prepared++) {
            if (queue.free_count == 0) {
                break;
            }
            uint16_t slot_idx = queue.free_slot_indices[queue.free_head];
            queue.free_head = (queue.free_head + 1) % kPeerRpcDepth;
            queue.free_count--;
            auto &slot = queue.send_slots[slot_idx];
            bool expected = false;
            if (!slot.in_use.compare_exchange_strong(expected, true,
                                                     std::memory_order_acq_rel)) {
                ERROR("server peer batch: free-list returned busy slot");
            }
            slot.msg = dq[prepared];
            slots[prepared] = &slot;

            sges[prepared].addr = reinterpret_cast<uint64_t>(&slot.msg);
            sges[prepared].length = rdma::ec2pc_rpc_wire_bytes(slot.msg);
            sges[prepared].lkey = peer_rpc_mr->lkey;

            wrs[prepared] = {};
            wrs[prepared].wr_id = reinterpret_cast<uint64_t>(&slot);
            wrs[prepared].next = nullptr;
            wrs[prepared].sg_list = &sges[prepared];
            wrs[prepared].num_sge = 1;
            wrs[prepared].opcode = IBV_WR_SEND;
            wrs[prepared].send_flags = 0;
        }

        if (prepared == 0) {
            auto send_lock_release = std::chrono::steady_clock::now();
            ctr_peer_send_lock_batch_hold_ns.fetch_add(
                elapsed_ns(send_lock_acquired, send_lock_release),
                std::memory_order_relaxed);
            return 0;
        }

        for (size_t i = 0; i + 1 < prepared; i++) {
            wrs[i].next = &wrs[i + 1];
            if (((i + 1) % kPeerSendSignalStride) == 0) {
                wrs[i].send_flags = IBV_SEND_SIGNALED;
            }
        }
        wrs[prepared - 1].send_flags = IBV_SEND_SIGNALED;

        ibv_send_wr *bad = nullptr;
        int ret = ibv_post_send(qp.queue_pair, &wrs[0], &bad);
        posted = prepared;
        if (ret != 0) {
            posted = 0;
            if (bad != nullptr && bad >= &wrs[0] && bad <= &wrs[prepared - 1]) {
                posted = static_cast<size_t>(bad - &wrs[0]);
            }
            for (size_t i = posted; i < prepared; i++) {
                auto *slot = slots[i];
                bool busy = true;
                if (!slot->in_use.compare_exchange_strong(busy, false,
                                                          std::memory_order_acq_rel)) {
                    ERROR("server peer batch: send failure on non-busy slot");
                }
                if (queue.free_count >= kPeerRpcDepth) {
                    ERROR("server peer batch: free-list overflow on send failure");
                }
                queue.free_slot_indices[queue.free_tail] = slot->slot_index;
                queue.free_tail = (queue.free_tail + 1) % kPeerRpcDepth;
                queue.free_count++;
            }
            if (ret != ENOMEM) {
                static std::atomic<uint32_t> log_budget{0};
                uint32_t seq = log_budget.fetch_add(1, std::memory_order_relaxed);
                if (seq < 16) {
                    ibv_qp_attr qp_attr{};
                    ibv_qp_init_attr qp_init_attr{};
                    int qret = ibv_query_qp(qp.queue_pair, &qp_attr,
                                            IBV_QP_STATE | IBV_QP_DEST_QPN,
                                            &qp_init_attr);
                    std::cerr << "WARN: peer batch ibv_post_send failed"
                              << " local=" << local_server_index
                              << " peer=" << peer_idx
                              << " ret=" << ret
                              << " posted_before_fail=" << posted
                              << " qret=" << qret
                              << " qp_state="
                              << (qret == 0 ? qp_attr.qp_state : -1)
                              << " dest_qpn="
                              << (qret == 0 ? static_cast<int>(qp_attr.dest_qp_num)
                                            : -1)
                              << std::endl;
                }
                if (!peer_recovery_enabled() || peer_is_alive(peer_idx))
                    ERROR("server peer batch ibv_post_send failed");
            }
        }

        if (posted > 0) {
            std::lock_guard<std::mutex> inflight_lock(
                inflight_peer_send_slot_mutexes[peer_idx]);
            auto &inflight = inflight_peer_send_slots[peer_idx];
            for (size_t i = 0; i < posted; i++) {
                inflight.push_back(slots[i]);
            }
        }
        auto send_lock_release = std::chrono::steady_clock::now();
        ctr_peer_send_lock_batch_hold_ns.fetch_add(
            elapsed_ns(send_lock_acquired, send_lock_release),
            std::memory_order_relaxed);
    }

    for (size_t i = 0; i < posted; i++) {
        dq.pop_front();
    }
    if (posted > 0) {
        pending_peer_send_total.fetch_sub(posted, std::memory_order_relaxed);
    }
    return posted;
}

inline bool Server::enqueue_peer_rpc_send(size_t peer_idx,
                                          const rdma::EC2PCRpcMessage &msg) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg.type == rdma::EC2PC_MSG_PARITY_APPLY && msg.wr_id == 0) {
        ERROR("enqueue_peer_rpc_send zero parity wr_id");
    }
    if (peer_idx >= pending_peer_send_queues.size() || peer_idx >= peer_qps.size() ||
        peer_idx == local_server_index || peer_qps[peer_idx].queue_pair == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(pending_peer_send_queue_mutexes[peer_idx]);
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    auto &dq = pending_peer_send_queues[peer_idx];
    bool merged = false;
    if (msg.type == rdma::EC2PC_MSG_ACK_BATCH && !dq.empty()) {
        auto &tail = dq.back();
        if (tail.type == rdma::EC2PC_MSG_ACK_BATCH &&
            tail.payload_len + msg.payload_len <= rdma::kEC2PCRpcPayloadBytes) {
            std::memcpy(tail.payload + tail.payload_len, msg.payload,
                        msg.payload_len);
            tail.payload_len += msg.payload_len;
            merged = true;
        }
    }
    if (!merged) {
        if (dq.size() >= kPeerPendingQueueLimit) {
            return false;
        }
        dq.push_back(msg);
        pending_peer_send_total.fetch_add(1, std::memory_order_relaxed);
        update_peak(ctr_peer_pending_send_peak,
                    static_cast<uint64_t>(dq.size()));
    }
    if (msg.type == rdma::EC2PC_MSG_PARITY_APPLY) {
        ctr_peer_enq_payload.fetch_add(1, std::memory_order_relaxed);
    } else if (msg.type == rdma::EC2PC_MSG_ACK) {
        ctr_peer_enq_ack.fetch_add(1, std::memory_order_relaxed);
    } else if (msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
        uint32_t ack_cnt = msg.payload_len / static_cast<uint32_t>(sizeof(uint64_t));
        ctr_peer_enq_ack.fetch_add(ack_cnt, std::memory_order_relaxed);
    }
    return true;
}

inline size_t Server::flush_pending_peer_sends(size_t max_posts,
                                               size_t first_peer,
                                               size_t peer_stride) {
    ctr_peer_flush_rounds.fetch_add(1, std::memory_order_relaxed);
    if (peer_stride == 0 || pending_peer_send_queues.empty()) {
        return 0;
    }
    size_t posted = 0;
    auto flush_begin = std::chrono::steady_clock::now();
    auto budget_exhausted = [&]() {
        return elapsed_ns(flush_begin, std::chrono::steady_clock::now()) >=
               kFlushRoundBudgetNs;
    };
    auto finish_flush = [&](size_t total_posted) {
        if (total_posted > 0) {
            ctr_peer_flush_posts.fetch_add(total_posted, std::memory_order_relaxed);
        }
        return total_posted;
    };
    for (size_t peer_idx = first_peer; peer_idx < pending_peer_send_queues.size();
         peer_idx += peer_stride) {
        if (peer_idx == local_server_index) {
            continue;
        }
        if (peer_recovery_enabled() && !peer_is_alive(peer_idx)) {
            drop_unposted_peer_work(peer_idx);
            continue;
        }
        std::lock_guard<std::mutex> lock(
            pending_peer_send_queue_mutexes[peer_idx]);
        auto &dq = pending_peer_send_queues[peer_idx];
        while (!dq.empty()) {
            if (posted >= max_posts) {
                return finish_flush(posted);
            }
            if ((posted & 0x3fu) == 0u && budget_exhausted()) {
                return finish_flush(posted);
            }
            size_t just_posted =
                try_post_peer_send_batch_locked(peer_idx, dq, max_posts - posted);
            if (just_posted == 0) {
                break;
            }
            posted += just_posted;
        }
    }
    return finish_flush(posted);
}

inline bool Server::post_peer_ack_message(size_t peer_idx,
                                          const rdma::EC2PCRpcMessage &msg) {
    auto post_begin = std::chrono::steady_clock::now();
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg.type != rdma::EC2PC_MSG_ACK &&
        msg.type != rdma::EC2PC_MSG_ACK_BATCH) {
        return false;
    }
    if (peer_rpc_mr == nullptr || peer_idx >= peer_ack_send_queues.size() ||
        peer_idx == local_server_index || peer_idx >= peer_qps.size()) {
        return false;
    }
    auto &qp = peer_qps[peer_idx];
    if (qp.queue_pair == nullptr) {
        return false;
    }
    if (peer_ack_send_mutexes == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> send_lock(peer_ack_send_mutexes[peer_idx]);
    auto &queue = peer_ack_send_queues[peer_idx];
    if (queue.slots == nullptr || queue.depth == 0) {
        return false;
    }

    PeerAckSendSlot *picked = nullptr;
    size_t start = queue.next_slot;
    for (size_t attempt = 0; attempt < queue.depth; attempt++) {
        auto &slot = queue.slots[(start + attempt) % queue.depth];
        bool expected = false;
        if (slot.in_use.compare_exchange_strong(expected, true,
                                                std::memory_order_acq_rel)) {
            picked = &slot;
            queue.next_slot = (start + attempt + 1) % queue.depth;
            break;
        }
    }
    if (picked == nullptr) {
        ctr_peer_ack_direct_post_fail_cnt.fetch_add(1,
                                                    std::memory_order_relaxed);
        return false;
    }

    picked->msg = msg;
    ibv_sge sge = {
        .addr = reinterpret_cast<uint64_t>(&picked->msg),
        .length = rdma::ec2pc_rpc_wire_bytes(msg),
        .lkey = peer_rpc_mr->lkey,
    };
    ibv_send_wr wr{};
    wr.wr_id = reinterpret_cast<uint64_t>(picked);
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    ibv_send_wr *bad = nullptr;
    int ret = ibv_post_send(qp.queue_pair, &wr, &bad);
    if (ret != 0) {
        ctr_peer_ack_direct_post_fail_cnt.fetch_add(1,
                                                    std::memory_order_relaxed);
        picked->in_use.store(false, std::memory_order_release);
        if (ret != ENOMEM &&
            (!peer_recovery_enabled() || peer_is_alive(peer_idx)))
            ERROR("server peer ACK ibv_post_send failed");
        return false;
    }
    picked->post_time_ns = steady_clock_now_ns();
    uint32_t ack_cnt =
        msg.type == rdma::EC2PC_MSG_ACK
            ? 1u
            : (msg.payload_len / static_cast<uint32_t>(sizeof(uint64_t)));
    ctr_peer_enq_ack.fetch_add(ack_cnt, std::memory_order_relaxed);
    ctr_peer_flush_posts.fetch_add(1, std::memory_order_relaxed);
    auto post_end = std::chrono::steady_clock::now();
    ctr_peer_ack_direct_post_ns.fetch_add(
        elapsed_ns(post_begin, post_end),
        std::memory_order_relaxed);
    ctr_peer_ack_direct_post_cnt.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline bool Server::enqueue_peer_ack_send(size_t peer_idx,
                                          const rdma::EC2PCRpcMessage &msg) {
    auto enqueue_begin = std::chrono::steady_clock::now();
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg.type != rdma::EC2PC_MSG_ACK &&
        msg.type != rdma::EC2PC_MSG_ACK_BATCH) {
        return false;
    }
    if (peer_idx >= pending_peer_ack_queues.size()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(pending_peer_ack_queue_mutexes[peer_idx]);
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    auto &dq = pending_peer_ack_queues[peer_idx];
    bool merged = false;
    if (msg.type == rdma::EC2PC_MSG_ACK_BATCH && !dq.empty()) {
        auto &tail = dq.back();
        if (tail.type == rdma::EC2PC_MSG_ACK_BATCH &&
            tail.payload_len + msg.payload_len <= rdma::kEC2PCRpcPayloadBytes) {
            std::memcpy(tail.payload + tail.payload_len, msg.payload,
                        msg.payload_len);
            tail.payload_len += msg.payload_len;
            merged = true;
        }
    }
    if (!merged) {
        dq.push_back(msg);
    }
    auto enqueue_end = std::chrono::steady_clock::now();
    ctr_peer_ack_enqueue_ns.fetch_add(
        elapsed_ns(enqueue_begin, enqueue_end),
        std::memory_order_relaxed);
    ctr_peer_ack_enqueue_cnt.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline size_t Server::flush_pending_peer_acks(size_t max_posts,
                                              size_t first_peer,
                                              size_t peer_stride) {
    if (peer_stride == 0 || pending_peer_ack_queues.empty()) {
        return 0;
    }
    size_t posted = 0;
    for (size_t peer_idx = first_peer; peer_idx < pending_peer_ack_queues.size();
         peer_idx += peer_stride) {
        if (peer_idx == local_server_index) {
            continue;
        }
        if (peer_recovery_enabled() && !peer_is_alive(peer_idx)) {
            drop_unposted_peer_work(peer_idx);
            continue;
        }
        std::lock_guard<std::mutex> lock(
            pending_peer_ack_queue_mutexes[peer_idx]);
        auto &dq = pending_peer_ack_queues[peer_idx];
        while (!dq.empty() && posted < max_posts) {
            if (!post_peer_ack_message(peer_idx, dq.front())) {
                break;
            }
            dq.pop_front();
            posted++;
        }
    }
    return posted;
}

inline bool Server::post_peer_rpc_message(size_t peer_idx,
                                          const rdma::EC2PCRpcMessage &msg) {
    static std::atomic<uint32_t> log_budget{0};
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg.type == rdma::EC2PC_MSG_PARITY_APPLY && msg.wr_id == 0) {
        ERROR("post_peer_rpc_message zero parity wr_id");
    }
    if (peer_rpc_mr == nullptr || peer_idx >= peer_rpc_queues.size() ||
        peer_idx == local_server_index || peer_idx >= peer_qps.size()) {
        uint32_t seq = log_budget.fetch_add(1, std::memory_order_relaxed);
        if (seq < 16) {
            std::cerr << "WARN: post_peer_rpc_message precheck failed local="
                      << local_server_index << " peer=" << peer_idx
                      << " peer_rpc_mr=" << static_cast<void *>(peer_rpc_mr)
                      << " queue_count=" << peer_rpc_queues.size()
                      << " qps=" << peer_qps.size()
                      << " msg_type=" << msg.type << std::endl;
        }
        return false;
    }
    auto &qp = peer_qps[peer_idx];
    if (qp.queue_pair == nullptr) {
        uint32_t seq = log_budget.fetch_add(1, std::memory_order_relaxed);
        if (seq < 16) {
            std::cerr << "WARN: post_peer_rpc_message null qp local="
                      << local_server_index << " peer=" << peer_idx
                      << " msg_type=" << msg.type << std::endl;
        }
        return false;
    }
    auto &queue = peer_rpc_queues[peer_idx];
    auto send_lock_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(peer_send_mutexes[peer_idx]);
    auto send_lock_acquired = std::chrono::steady_clock::now();
    ctr_peer_send_lock_direct_wait_ns.fetch_add(
        elapsed_ns(send_lock_begin, send_lock_acquired),
        std::memory_order_relaxed);
    ctr_peer_send_lock_direct_ops.fetch_add(1, std::memory_order_relaxed);
    PeerRpcSendSlot *picked = nullptr;
    if (queue.free_count > 0) {
        uint16_t slot_idx = queue.free_slot_indices[queue.free_head];
        queue.free_head = (queue.free_head + 1) % kPeerRpcDepth;
        queue.free_count--;
        auto &slot = queue.send_slots[slot_idx];
        bool expected = false;
        if (!slot.in_use.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel)) {
            ERROR("server post_peer_rpc_message: free-list returned busy slot");
        }
        picked = &slot;
    }
    if (picked == nullptr) {
        auto send_lock_release = std::chrono::steady_clock::now();
        ctr_peer_send_lock_direct_hold_ns.fetch_add(
            elapsed_ns(send_lock_acquired, send_lock_release),
            std::memory_order_relaxed);
        uint32_t seq = log_budget.fetch_add(1, std::memory_order_relaxed);
        if (seq < 16) {
            std::cerr << "WARN: post_peer_rpc_message slot exhausted local="
                      << local_server_index << " peer=" << peer_idx
                      << " msg_type=" << msg.type << std::endl;
        }
        return false;
    }

    picked->msg = msg;
    ibv_sge sge = {
        .addr = reinterpret_cast<uint64_t>(&picked->msg),
        .length = rdma::ec2pc_rpc_wire_bytes(msg),
        .lkey = peer_rpc_mr->lkey,
    };
    ibv_send_wr wr{};
    wr.wr_id = reinterpret_cast<uint64_t>(picked);
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    ibv_send_wr *bad = nullptr;
    int ret = ibv_post_send(qp.queue_pair, &wr, &bad);
    if (ret != 0) {
        bool expected = true;
        if (!picked->in_use.compare_exchange_strong(expected, false,
                                                    std::memory_order_acq_rel)) {
            ERROR("server post_peer_rpc_message: send failure on non-busy slot");
        }
        if (queue.free_count >= kPeerRpcDepth) {
            ERROR("server post_peer_rpc_message: free-list overflow on send failure");
        }
        queue.free_slot_indices[queue.free_tail] = picked->slot_index;
        queue.free_tail = (queue.free_tail + 1) % kPeerRpcDepth;
        queue.free_count++;
        uint32_t seq = log_budget.fetch_add(1, std::memory_order_relaxed);
        if (seq < 16) {
            ibv_qp_attr qp_attr{};
            ibv_qp_init_attr qp_init_attr{};
            int qret = ibv_query_qp(qp.queue_pair, &qp_attr,
                                    IBV_QP_STATE | IBV_QP_DEST_QPN,
                                    &qp_init_attr);
            std::cerr << "WARN: post_peer_rpc_message ibv_post_send failed"
                      << " local=" << local_server_index
                      << " peer=" << peer_idx
                      << " msg_type=" << msg.type
                      << " ret=" << ret
                      << " qret=" << qret
                      << " qp_state="
                      << (qret == 0 ? qp_attr.qp_state : -1)
                      << " dest_qpn="
                      << (qret == 0 ? static_cast<int>(qp_attr.dest_qp_num) : -1)
                      << std::endl;
        }
        if (ret != ENOMEM &&
            (!peer_recovery_enabled() || peer_is_alive(peer_idx)))
            ERROR("server post_peer_rpc_message ibv_post_send failed");
        auto send_lock_release = std::chrono::steady_clock::now();
        ctr_peer_send_lock_direct_hold_ns.fetch_add(
            elapsed_ns(send_lock_acquired, send_lock_release),
            std::memory_order_relaxed);
        return false;
    }
    {
        std::lock_guard<std::mutex> inflight_lock(
            inflight_peer_send_slot_mutexes[peer_idx]);
        inflight_peer_send_slots[peer_idx].push_back(picked);
    }
    auto send_lock_release = std::chrono::steady_clock::now();
    ctr_peer_send_lock_direct_hold_ns.fetch_add(
        elapsed_ns(send_lock_acquired, send_lock_release),
        std::memory_order_relaxed);
    if (msg.type == rdma::EC2PC_MSG_PARITY_APPLY) {
        ctr_peer_enq_payload.fetch_add(1, std::memory_order_relaxed);
    } else if (msg.type == rdma::EC2PC_MSG_ACK) {
        ctr_peer_enq_ack.fetch_add(1, std::memory_order_relaxed);
    } else if (msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
        uint32_t ack_cnt = msg.payload_len / static_cast<uint32_t>(sizeof(uint64_t));
        ctr_peer_enq_ack.fetch_add(ack_cnt, std::memory_order_relaxed);
    }
    ctr_peer_flush_posts.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline bool Server::post_peer_parity_apply_lane(
    size_t peer_idx, size_t worker_idx, const rdma::EC2PCRpcMessage &msg) {
    static std::atomic<uint32_t> log_budget{0};
    auto lane_try_begin = std::chrono::steady_clock::now();
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg.type != rdma::EC2PC_MSG_PARITY_APPLY) {
        return false;
    }
    if (msg.wr_id == 0) {
        ERROR("post_peer_parity_apply_lane zero wr_id");
    }
    if (peer_rpc_mr == nullptr || peer_idx >= peer_endpoint_count ||
        peer_idx == local_server_index || peer_idx >= peer_lane_send_queues.size()) {
        return false;
    }
    if (worker_idx >= std::max<size_t>(1, rpc_worker_count)) {
        return false;
    }
    size_t lane_owner_count = std::max<size_t>(1, rpc_worker_count);
    size_t lane_idx = peer_idx * lane_owner_count + worker_idx;
    if (lane_idx >= peer_lane_send_queues.size()) {
        return false;
    }
    auto &lane = peer_lane_send_queues[lane_idx];
    if (lane.slots == nullptr) {
        return false;
    }
    size_t shard_idx = worker_owned_peer_payload_shard(worker_idx);
    size_t payload_q_idx = peer_payload_qp_index(peer_idx, shard_idx);
    if (payload_q_idx >= peer_payload_qps.size()) {
        return false;
    }
    auto &qp = peer_payload_qps[payload_q_idx];
    if (qp.queue_pair == nullptr) {
        return false;
    }

    size_t depth = std::max<size_t>(1, peer_lane_depth);
    auto &slot = lane.slots[lane.next_slot % depth];
    bool expected = false;
    if (!slot.in_use.compare_exchange_strong(expected, true,
                                             std::memory_order_acq_rel)) {
        auto lane_try_end = std::chrono::steady_clock::now();
        ctr_peer_lane_busy_ns.fetch_add(
            elapsed_ns(lane_try_begin, lane_try_end),
            std::memory_order_relaxed);
        ctr_peer_lane_busy_retries.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    slot.msg = msg;
    ibv_sge sge = {
        .addr = reinterpret_cast<uint64_t>(&slot.msg),
        .length = rdma::ec2pc_rpc_wire_bytes(slot.msg),
        .lkey = peer_rpc_mr->lkey,
    };
    ibv_send_wr wr{};
    wr.wr_id = reinterpret_cast<uint64_t>(&slot);
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    ibv_send_wr *bad = nullptr;
    int ret = ibv_post_send(qp.queue_pair, &wr, &bad);
    if (ret != 0) {
        slot.in_use.store(false, std::memory_order_release);
        ctr_peer_lane_post_fail.fetch_add(1, std::memory_order_relaxed);
        uint32_t seq = log_budget.fetch_add(1, std::memory_order_relaxed);
        if (ret != ENOMEM && seq < 16) {
            ibv_qp_attr qp_attr{};
            ibv_qp_init_attr qp_init_attr{};
            int qret = ibv_query_qp(qp.queue_pair, &qp_attr,
                                    IBV_QP_STATE | IBV_QP_DEST_QPN,
                                    &qp_init_attr);
            std::cerr << "WARN: post_peer_parity_apply_lane failed"
                      << " local=" << local_server_index
                      << " peer=" << peer_idx
                      << " lane_worker=" << worker_idx
                      << " parity_idx=" << msg.parity_idx
                      << " ret=" << ret
                      << " qret=" << qret
                      << " qp_state="
                      << (qret == 0 ? qp_attr.qp_state : -1)
                      << " dest_qpn="
                      << (qret == 0 ? static_cast<int>(qp_attr.dest_qp_num) : -1)
                      << std::endl;
        }
        if (ret != ENOMEM &&
            (!peer_recovery_enabled() || peer_is_alive(peer_idx)))
            ERROR("server post_peer_parity_apply_lane ibv_post_send failed");
        return false;
    }
    lane.next_slot = (lane.next_slot + 1) % depth;
    auto lane_try_end = std::chrono::steady_clock::now();
    ctr_peer_lane_post_ok_ns.fetch_add(
        elapsed_ns(lane_try_begin, lane_try_end),
        std::memory_order_relaxed);
    ctr_peer_lane_post_ok.fetch_add(1, std::memory_order_relaxed);
    ctr_peer_enq_payload.fetch_add(1, std::memory_order_relaxed);
    ctr_peer_flush_posts.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline size_t Server::flush_peer_parity_apply_lane(size_t peer_idx,
                                                   size_t worker_idx) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx)) {
        drop_unposted_peer_work(peer_idx);
        return 0;
    }
    if (worker_idx >= std::max<size_t>(1, rpc_worker_count)) {
        return 0;
    }
    size_t lane_owner_count = std::max<size_t>(1, rpc_worker_count);
    size_t lane_idx = peer_idx * lane_owner_count + worker_idx;
    if (lane_idx >= peer_lane_send_queues.size() || !peer_lane_mutexes) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(peer_lane_mutexes[lane_idx]);
    auto &lane = peer_lane_send_queues[lane_idx];
    if (lane.pending_count == 0 || lane.slots == nullptr || peer_rpc_mr == nullptr ||
        peer_idx >= peer_endpoint_count || peer_idx == local_server_index) {
        return 0;
    }
    size_t shard_idx = worker_owned_peer_payload_shard(worker_idx);
    size_t payload_q_idx = peer_payload_qp_index(peer_idx, shard_idx);
    if (payload_q_idx >= peer_payload_qps.size()) {
        return 0;
    }
    auto &qp = peer_payload_qps[payload_q_idx];
    if (qp.queue_pair == nullptr) {
        return 0;
    }

    constexpr size_t kBatchCap = kPeerPostSendBatch;
    std::array<PeerLaneSendSlot *, kBatchCap> slots{};
    std::array<ibv_sge, kBatchCap> sges{};
    std::array<ibv_send_wr, kBatchCap> wrs{};

    size_t depth = std::max<size_t>(1, peer_lane_depth);
    size_t prepared = 0;
    size_t local_next_slot = lane.next_slot;
    auto lane_try_begin = std::chrono::steady_clock::now();
    while (prepared < kBatchCap && prepared < lane.pending_count) {
        auto &slot = lane.slots[local_next_slot % depth];
        bool expected = false;
        if (!slot.in_use.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel)) {
            if (prepared == 0) {
                auto lane_try_end = std::chrono::steady_clock::now();
                ctr_peer_lane_busy_ns.fetch_add(
                    elapsed_ns(lane_try_begin, lane_try_end),
                    std::memory_order_relaxed);
                ctr_peer_lane_busy_retries.fetch_add(1, std::memory_order_relaxed);
            }
            break;
        }

        size_t msg_idx = (lane.pending_head + prepared) % lane.pending_capacity;
        slot.msg = lane.pending_msgs[msg_idx];
        slots[prepared] = &slot;

        sges[prepared].addr = reinterpret_cast<uint64_t>(&slot.msg);
        sges[prepared].length = rdma::ec2pc_rpc_wire_bytes(slot.msg);
        sges[prepared].lkey = peer_rpc_mr->lkey;

        wrs[prepared] = {};
        wrs[prepared].wr_id = reinterpret_cast<uint64_t>(&slot);
        wrs[prepared].next = nullptr;
        wrs[prepared].sg_list = &sges[prepared];
        wrs[prepared].num_sge = 1;
        wrs[prepared].opcode = IBV_WR_SEND;
        wrs[prepared].send_flags = IBV_SEND_SIGNALED;
        if (prepared > 0) {
            wrs[prepared - 1].next = &wrs[prepared];
        }

        prepared++;
        local_next_slot = (local_next_slot + 1) % depth;
    }

    if (prepared == 0) {
        return 0;
    }

    ibv_send_wr *bad = nullptr;
    int ret = ibv_post_send(qp.queue_pair, &wrs[0], &bad);
    size_t posted = prepared;
    if (ret != 0) {
        posted = 0;
        if (bad != nullptr && bad >= &wrs[0] && bad <= &wrs[prepared - 1]) {
            posted = static_cast<size_t>(bad - &wrs[0]);
        }
        for (size_t i = posted; i < prepared; i++) {
            slots[i]->in_use.store(false, std::memory_order_release);
        }
        ctr_peer_lane_post_fail.fetch_add(prepared - posted,
                                          std::memory_order_relaxed);
    }

    if (posted > 0) {
        lane.pending_head = (lane.pending_head + posted) % lane.pending_capacity;
        lane.pending_count -= posted;
        lane.next_slot = (lane.next_slot + posted) % depth;
    }

    auto lane_try_end = std::chrono::steady_clock::now();
    ctr_peer_lane_post_ok_ns.fetch_add(
        elapsed_ns(lane_try_begin, lane_try_end),
        std::memory_order_relaxed);
    ctr_peer_lane_post_ok.fetch_add(posted, std::memory_order_relaxed);
    ctr_peer_enq_payload.fetch_add(posted, std::memory_order_relaxed);
    ctr_peer_flush_posts.fetch_add(posted, std::memory_order_relaxed);
    return posted;
}

inline size_t Server::flush_all_peer_parity_apply_lanes(size_t max_posts) {
    if (peer_lane_send_queues.empty()) {
        return 0;
    }
    size_t posted = 0;
    size_t lane_owner_count = std::max<size_t>(1, rpc_worker_count);
    for (size_t peer_idx = 0; peer_idx < peer_qps.size(); peer_idx++) {
        if (peer_idx == local_server_index) {
            continue;
        }
        for (size_t worker_idx = 0; worker_idx < lane_owner_count; worker_idx++) {
            posted += flush_peer_parity_apply_lane(peer_idx, worker_idx);
            if (posted >= max_posts) {
                return posted;
            }
        }
    }
    return posted;
}

inline bool Server::enqueue_peer_parity_apply_lane(
    size_t peer_idx, size_t worker_idx, const rdma::EC2PCRpcMessage &msg) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return false;
    if (msg.type != rdma::EC2PC_MSG_PARITY_APPLY) {
        return false;
    }
    if (msg.wr_id == 0) {
        ERROR("enqueue_peer_parity_apply_lane zero wr_id");
    }
    if (worker_idx >= std::max<size_t>(1, rpc_worker_count)) {
        return false;
    }
    size_t lane_owner_count = std::max<size_t>(1, rpc_worker_count);
    if (peer_idx * lane_owner_count >= peer_lane_send_queues.size() ||
        !peer_lane_mutexes) {
        return false;
    }
    size_t preferred = worker_idx % lane_owner_count;
    for (size_t offset = 0; offset < lane_owner_count; offset++) {
        size_t candidate_worker = (preferred + offset) % lane_owner_count;
        size_t lane_idx = peer_idx * lane_owner_count + candidate_worker;
        if (lane_idx >= peer_lane_send_queues.size()) {
            continue;
        }
        std::lock_guard<std::mutex> lock(peer_lane_mutexes[lane_idx]);
        if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
            return false;
        auto &lane = peer_lane_send_queues[lane_idx];
        if (lane.pending_msgs == nullptr || lane.pending_capacity == 0) {
            continue;
        }
        if (lane.pending_count >= lane.pending_capacity) {
            continue;
        }
        lane.pending_msgs[lane.pending_tail] = msg;
        lane.pending_tail = (lane.pending_tail + 1) % lane.pending_capacity;
        lane.pending_count++;
        return true;
    }
    ctr_peer_lane_enqueue_full.fetch_add(1, std::memory_order_relaxed);
    return false;
}

inline size_t Server::reclaim_inflight_send_slots_upto(
    size_t data_qp_local_idx, RpcSendSlot *completed_slot) {
    if (completed_slot == nullptr || data_qp_local_idx >= inflight_send_slots.size()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(inflight_send_slot_mutexes[data_qp_local_idx]);
    auto &inflight = inflight_send_slots[data_qp_local_idx];
    size_t reclaimed = 0;
    bool found = false;
    while (!inflight.empty()) {
        RpcSendSlot *slot = inflight.front();
        inflight.pop_front();
        if (slot != nullptr) {
            slot->in_use.store(false, std::memory_order_release);
        }
        reclaimed++;
        if (slot == completed_slot) {
            found = true;
            break;
        }
    }
    if (!found) {
        completed_slot->in_use.store(false, std::memory_order_release);
        reclaimed = 1;
    }
    return reclaimed;
}

inline size_t Server::try_post_send_rpc_batch_locked(
    size_t data_qp_local_idx, std::deque<rdma::EC2PCRpcMessage> &dq,
    std::deque<PendingRpcSendMeta> &meta_q,
    size_t max_posts) {
    static constexpr size_t kRpcSendSlotProbeLimit = 8;
    if (data_qp_local_idx >= rpc_queues.size() || max_posts == 0 ||
        dq.empty()) {
        return 0;
    }

    const size_t batch_cap =
        std::min({max_posts, dq.size(), static_cast<size_t>(kRpcPostSendBatch)});
    if (batch_cap == 0) {
        return 0;
    }

    auto &queue = rpc_queues[data_qp_local_idx];
    std::array<RpcSendSlot *, kRpcPostSendBatch> slots{};
    std::array<ibv_sge, kRpcPostSendBatch> sges{};
    std::array<ibv_send_wr, kRpcPostSendBatch> wrs{};

    size_t prepared = 0;
    for (; prepared < batch_cap; prepared++) {
        RpcSendSlot *picked = nullptr;
        const size_t probe_limit =
            std::min(rpc_queue_depth, kRpcSendSlotProbeLimit);
        for (size_t attempt = 0; attempt < probe_limit; attempt++) {
            auto &slot = queue.send_slots[queue.send_head];
            queue.send_head = (queue.send_head + 1) % rpc_queue_depth;
            bool expected = false;
            if (slot.in_use.compare_exchange_strong(expected, true,
                                                    std::memory_order_acq_rel)) {
                picked = &slot;
                break;
            }
        }
        if (picked == nullptr) {
            break;
        }

        const auto &msg = dq[prepared];
        picked->msg = msg;
        picked->post_time_ns = 0;
        slots[prepared] = picked;

        sges[prepared].addr = reinterpret_cast<uint64_t>(&picked->msg);
        sges[prepared].length = rdma::ec2pc_rpc_wire_bytes(msg);
        sges[prepared].lkey = rpc_mr->lkey;

        wrs[prepared] = {};
        wrs[prepared].wr_id = reinterpret_cast<uint64_t>(picked);
        wrs[prepared].next = nullptr;
        wrs[prepared].sg_list = &sges[prepared];
        wrs[prepared].num_sge = 1;
        wrs[prepared].opcode = IBV_WR_SEND;
        wrs[prepared].send_flags = 0;
    }

    if (prepared == 0) {
        return 0;
    }

    for (size_t i = 0; i + 1 < prepared; i++) {
        wrs[i].next = &wrs[i + 1];
        if (((i + 1) % kRpcSendSignalStride) == 0) {
            wrs[i].send_flags = IBV_SEND_SIGNALED;
        }
    }
    // Keep a guaranteed completion per ibv_post_send call.
    wrs[prepared - 1].send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr *bad = nullptr;
    int ret = ibv_post_send(qps[data_qp_local_idx + 1].queue_pair, &wrs[0],
                            &bad);

    size_t posted = prepared;
    if (ret != 0) {
        // ibv_post_send may partially post a WR chain: all WRs before `bad`
        // are already accepted by HW and must be treated as in-flight.
        posted = 0;
        if (bad != nullptr && bad >= &wrs[0] && bad <= &wrs[prepared - 1]) {
            posted = static_cast<size_t>(bad - &wrs[0]);
        }
        if (ret != ENOMEM) {
            std::cerr << "server post_send failed ret=" << ret
                      << " posted_before_fail=" << posted
                      << std::endl;
            ERROR("server post_send unexpected failure");
        }
    }

    {
        uint64_t post_time_ns = steady_clock_now_ns();
        std::lock_guard<std::mutex> inflight_lock(
            inflight_send_slot_mutexes[data_qp_local_idx]);
        auto &inflight = inflight_send_slots[data_qp_local_idx];
        for (size_t i = 0; i < posted; i++) {
            slots[i]->post_time_ns = post_time_ns;
            if (slots[i]->msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
                if (i < meta_q.size()) {
                    const auto &meta = meta_q[i];
                    if (meta.ack_count != 0 &&
                        meta.server_recv_start_sum_ns != 0) {
                        uint64_t post_sum_ns = 0;
                        if (post_time_ns >=
                            meta.server_recv_start_sum_ns / meta.ack_count) {
                            if (meta.ack_count >=
                                std::numeric_limits<uint64_t>::max() /
                                    post_time_ns) {
                                post_sum_ns =
                                    std::numeric_limits<uint64_t>::max();
                            } else {
                                post_sum_ns = post_time_ns * meta.ack_count;
                            }
                        }
                        uint64_t delta_sum_ns =
                            post_sum_ns > meta.server_recv_start_sum_ns
                                ? (post_sum_ns -
                                   meta.server_recv_start_sum_ns)
                                : 0;
                        rdma::set_ec2pc_rpc_server_recv_to_ack_post_sum_ns(
                            slots[i]->msg, delta_sum_ns);
                    }
                }
                uint64_t ack_count = static_cast<uint64_t>(
                    std::max<uint32_t>(
                        1, slots[i]->msg.payload_len /
                               static_cast<uint32_t>(sizeof(uint64_t))));
                ctr_client_final_ack_posted.fetch_add(
                    ack_count, std::memory_order_relaxed);
                if (i < meta_q.size()) {
                    const auto &meta = meta_q[i];
                    uint64_t meta_ack_count = static_cast<uint64_t>(
                        meta.ack_count == 0 ? ack_count : meta.ack_count);
                    if (meta_ack_count != 0 &&
                        meta.enqueue_sum_ns != 0 &&
                        post_time_ns * meta_ack_count >= meta.enqueue_sum_ns) {
                        uint64_t total_delta_ns =
                            post_time_ns * meta_ack_count -
                            meta.enqueue_sum_ns;
                        ctr_client_final_ack_enqueue_to_post_ns.fetch_add(
                            total_delta_ns, std::memory_order_relaxed);
                        ctr_client_final_ack_enqueue_to_post_cnt.fetch_add(
                            meta_ack_count, std::memory_order_relaxed);
                        update_peak(
                            ctr_client_final_ack_enqueue_to_post_max_ns,
                            total_delta_ns / meta_ack_count);
                    }
                    if (meta_ack_count != 0 &&
                        meta.final_ack_ready_sum_ns != 0 &&
                        post_time_ns * meta_ack_count >=
                            meta.final_ack_ready_sum_ns) {
                        uint64_t total_delta_ns =
                            post_time_ns * meta_ack_count -
                            meta.final_ack_ready_sum_ns;
                        ctr_client_final_ack_done_to_post_ns.fetch_add(
                            total_delta_ns, std::memory_order_relaxed);
                        ctr_client_final_ack_done_to_post_cnt.fetch_add(
                            meta_ack_count, std::memory_order_relaxed);
                        update_peak(
                            ctr_client_final_ack_done_to_post_max_ns,
                            total_delta_ns / meta_ack_count);
                    }
                }
            }
            inflight.push_back(slots[i]);
            dq.pop_front();
            meta_q.pop_front();
        }
    }
    if (posted > 0) {
        pending_send_total.fetch_sub(posted, std::memory_order_relaxed);
        ctr_post_ok.fetch_add(posted, std::memory_order_relaxed);
        ctr_flush_posts.fetch_add(posted, std::memory_order_relaxed);
        if (ec2pc_preflight_debug_enabled() &&
            !slots.empty() &&
            slots[0] != nullptr &&
            slots[0]->msg.type == rdma::EC2PC_MSG_PROBE_ACK) {
            static std::atomic<uint64_t> seen{0};
            uint64_t seq = seen.fetch_add(1, std::memory_order_relaxed) + 1;
            if (seq <= 16) {
                std::cerr << "[ec2pc-preflight] server post probe-ack"
                          << " server=" << local_server_index
                          << " data_qp=" << data_qp_local_idx
                          << " posted=" << posted
                          << " qp_num="
                          << qps[data_qp_local_idx + 1].queue_pair->qp_num
                          << std::endl;
            }
        }
    }

    if (ret != 0) {
        const size_t failed = prepared - posted;
        for (size_t i = posted; i < prepared; i++) {
            slots[i]->in_use.store(false, std::memory_order_release);
        }
        if (failed > 0) {
            ctr_post_fail.fetch_add(failed, std::memory_order_relaxed);
        }
    }

    return posted;
}

inline bool Server::enqueue_rpc_send(size_t data_qp_local_idx,
                                     const rdma::EC2PCRpcMessage &msg,
                                     uint64_t final_ack_ready_time_ns,
                                     uint64_t server_recv_start_time_ns) {
    if (data_qp_local_idx >= pending_send_queues.size()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(pending_send_queue_mutexes[data_qp_local_idx]);
    auto &dq = pending_send_queues[data_qp_local_idx];
    auto &meta_q = pending_send_meta_queues[data_qp_local_idx];
    uint64_t enqueue_time_ns = steady_clock_now_ns();
    const bool single_ack_batch =
        msg.type == rdma::EC2PC_MSG_ACK_BATCH &&
        msg.payload_len == sizeof(uint64_t);

    if (single_ack_batch && final_ack_ready_time_ns != 0 &&
        enqueue_time_ns >= final_ack_ready_time_ns) {
        uint64_t delta_ns = enqueue_time_ns - final_ack_ready_time_ns;
        ctr_client_final_ack_done_to_enqueue_ns.fetch_add(
            delta_ns, std::memory_order_relaxed);
        ctr_client_final_ack_done_to_enqueue_cnt.fetch_add(
            1, std::memory_order_relaxed);
        update_peak(ctr_client_final_ack_done_to_enqueue_max_ns, delta_ns);
    }

    bool merged = false;
    if (single_ack_batch && !dq.empty()) {
        auto &tail = dq.back();
        auto &tail_meta = meta_q.back();
        if (tail.type == rdma::EC2PC_MSG_ACK_BATCH &&
            tail.payload_len + sizeof(uint64_t) <= rdma::kEC2PCRpcPayloadBytes) {
            std::memcpy(tail.payload + tail.payload_len, msg.payload,
                        sizeof(uint64_t));
            tail.payload_len += static_cast<uint32_t>(sizeof(uint64_t));
            tail_meta.enqueue_sum_ns += enqueue_time_ns;
            tail_meta.ack_count += 1;
            if (server_recv_start_time_ns != 0) {
                tail_meta.server_recv_start_sum_ns += server_recv_start_time_ns;
            }
            if (final_ack_ready_time_ns != 0) {
                tail_meta.final_ack_ready_sum_ns += final_ack_ready_time_ns;
            }
            merged = true;
        }
    }

    if (!merged) {
        dq.push_back(msg);
        PendingRpcSendMeta meta{};
        if (msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
            meta.enqueue_sum_ns = enqueue_time_ns;
            meta.ack_count = static_cast<uint32_t>(
                std::max<uint32_t>(
                    1, msg.payload_len /
                           static_cast<uint32_t>(sizeof(uint64_t))));
            meta.server_recv_start_sum_ns = server_recv_start_time_ns;
            if (single_ack_batch && final_ack_ready_time_ns != 0) {
                meta.final_ack_ready_sum_ns = final_ack_ready_time_ns;
            }
        }
        meta_q.push_back(meta);
        pending_send_total.fetch_add(1, std::memory_order_relaxed);
    }

    if (msg.type == rdma::EC2PC_MSG_PARITY_PAYLOAD) {
        ctr_enq_payload.fetch_add(1, std::memory_order_relaxed);
    } else if (msg.type == rdma::EC2PC_MSG_ACK) {
        ctr_enq_ack.fetch_add(1, std::memory_order_relaxed);
    } else if (msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
        uint32_t ack_cnt = msg.payload_len / static_cast<uint32_t>(sizeof(uint64_t));
        ctr_enq_ack.fetch_add(ack_cnt, std::memory_order_relaxed);
    }
    update_peak(ctr_pending_send_peak, static_cast<uint64_t>(dq.size()));
    return true;
}

inline size_t Server::flush_pending_sends(size_t max_posts, size_t first_q,
                                          size_t q_stride) {
    ctr_flush_rounds.fetch_add(1, std::memory_order_relaxed);
    size_t posted = 0;
    if (q_stride == 0) {
        return 0;
    }
    auto flush_begin = std::chrono::steady_clock::now();
    auto budget_exhausted = [&]() {
        return elapsed_ns(flush_begin, std::chrono::steady_clock::now()) >=
               kFlushRoundBudgetNs;
    };
    for (size_t q = first_q; q < pending_send_queues.size(); q += q_stride) {
        std::lock_guard<std::mutex> lock(pending_send_queue_mutexes[q]);
        auto &dq = pending_send_queues[q];
        auto &meta_q = pending_send_meta_queues[q];
        while (!dq.empty()) {
            if (posted >= max_posts) {
                return posted;
            }
            if ((posted & 0x3fu) == 0u && budget_exhausted()) {
                return posted;
            }
            size_t just_posted =
                try_post_send_rpc_batch_locked(q, dq, meta_q,
                                               max_posts - posted);
            if (just_posted == 0) {
                break;
            }
            posted += just_posted;
        }
    }
    return posted;
}
