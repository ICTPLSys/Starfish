#pragma once

// NOTE: This file is included by `rdma/server.hpp` after `server_rpc_core.hpp`,
// still inside namespace `FarLib::rdma`.

inline bool Server::flush_peer_payload_transport_once(size_t max_batches) {
    static constexpr size_t kPeerLaneFlushBudget = 1024;
    static constexpr size_t kRuntimePairFlushBudget = 64;
    static constexpr size_t kProbePairFlushBudget = 64;
    static constexpr size_t kProbeReadySweepBudget = 64;
    bool made_progress = false;
    if (!peer_payload_transport_ready()) {
        return made_progress;
    }
    auto flush_probe_payload_send_path = [&](size_t budget) -> bool {
        bool progress = false;
        size_t pair_budget =
            budget == SIZE_MAX ? kProbePairFlushBudget
                               : std::max<size_t>(1, budget);
        progress =
            flush_probe_parity_batch_pair_pending(pair_budget) > 0 || progress;

        size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
        size_t active_peers =
            peer_endpoint_count > 0 ? peer_endpoint_count - 1 : 0;
        size_t sweep_space = active_peers * payload_qp_count;
        if (sweep_space == 0) {
            return progress;
        }
        size_t sweep_budget =
            budget == SIZE_MAX ? kProbeReadySweepBudget
                               : std::max<size_t>(1, budget);
        static thread_local size_t ready_rr_cursor = 0;
        for (size_t attempt = 0; attempt < sweep_budget; attempt++) {
            size_t rr = (ready_rr_cursor + attempt) % sweep_space;
            size_t peer_rank = rr / payload_qp_count;
            size_t shard_idx = rr % payload_qp_count;
            size_t peer_idx = peer_rank;
            if (peer_idx >= local_server_index) {
                peer_idx++;
            }
            if (peer_idx >= peer_endpoint_count) {
                continue;
            }
            progress = flush_ready_probe_parity_batch_send_slots(
                           peer_idx, shard_idx, 1) > 0 ||
                       progress;
        }
        ready_rr_cursor = (ready_rr_cursor + sweep_budget) % sweep_space;
        return progress;
    };
    size_t probe_pair_budget =
        max_batches == SIZE_MAX
            ? kProbePairFlushBudget
            : std::max(kProbePairFlushBudget, std::max<size_t>(1, max_batches));
    made_progress =
        flush_probe_payload_send_path(probe_pair_budget) || made_progress;
    size_t runtime_pair_budget =
        max_batches == SIZE_MAX
            ? kRuntimePairFlushBudget
            : std::max(kRuntimePairFlushBudget,
                       std::max<size_t>(1, max_batches));
    made_progress =
        flush_runtime_parity_batch_pair_pending(runtime_pair_budget) > 0 ||
        made_progress;
    made_progress =
        flush_all_peer_parity_apply_lanes(kPeerLaneFlushBudget) > 0 ||
        made_progress;
    made_progress = flush_probe_payload_send_path(max_batches) || made_progress;
    made_progress = flush_runtime_parity_batch_pair_pending(
                        max_batches == SIZE_MAX
                            ? kRuntimePairFlushBudget
                            : std::max<size_t>(1, max_batches)) > 0 ||
                    made_progress;
    return made_progress;
}

inline bool Server::poll_peer_payload_cq_only_once(size_t max_batches) {
    bool made_progress = false;
    uint64_t round_begin_ns = steady_clock_now_ns();
    uint64_t cq_lock_wait_ns = 0;
    uint64_t cq_poll_sys_ns = 0;
    uint64_t cq_handle_ns = 0;
    uint64_t poll_calls = 0;
    uint64_t nonempty_polls = 0;
    uint64_t wc_total = 0;
    if (!peer_payload_transport_ready()) {
        return made_progress;
    }
    uint64_t poll_enter_ns = steady_clock_now_ns();
    uint64_t prev_poll_ns =
        last_peer_payload_poll_time_ns_.exchange(
            poll_enter_ns, std::memory_order_relaxed);
    if (prev_poll_ns != 0 && poll_enter_ns >= prev_poll_ns) {
        uint64_t gap_ns = poll_enter_ns - prev_poll_ns;
        ctr_peer_payload_poll_gap_ns.fetch_add(
            gap_ns, std::memory_order_relaxed);
        ctr_peer_payload_poll_gap_cnt.fetch_add(1, std::memory_order_relaxed);
        update_peak(ctr_peer_payload_poll_gap_max_ns, gap_ns);
    }
    for (size_t cq_idx = 0; cq_idx < peer_payload_cqs.size(); cq_idx++) {
        auto *payload_cq = peer_payload_cqs[cq_idx].get();
        if (payload_cq == nullptr) {
            continue;
        }
        size_t completion_budget =
            max_batches == SIZE_MAX ? SIZE_MAX : std::max<size_t>(1, max_batches);
        size_t completions_processed = 0;
        while (true) {
            int poll_cap = 64;
            if (completion_budget != SIZE_MAX) {
                size_t remaining = completion_budget - completions_processed;
                if (remaining == 0) {
                    break;
                }
                poll_cap = static_cast<int>(std::min<size_t>(remaining, 64));
            }
            ibv_wc wc[64];
            uint64_t poll_seq =
                ctr_peer_payload_poll_calls.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            int poll_ret = 0;
            uint64_t cq_poll_enter_ns = 0;
            uint64_t cq_lock_wait_begin_ns = steady_clock_now_ns();
            uint64_t lock_wait_ns = 0;
            if (peer_payload_cq_poll_mutexes != nullptr) {
                std::unique_lock<std::mutex> lock(
                    peer_payload_cq_poll_mutexes[cq_idx]);
                lock_wait_ns = steady_clock_now_ns() - cq_lock_wait_begin_ns;
                cq_lock_wait_ns += lock_wait_ns;
                cq_poll_enter_ns = steady_clock_now_ns();
                poll_ret =
                    ibv_poll_cq(payload_cq->complete_queue, poll_cap, wc);
            } else {
                lock_wait_ns = steady_clock_now_ns() - cq_lock_wait_begin_ns;
                cq_lock_wait_ns += lock_wait_ns;
                cq_poll_enter_ns = steady_clock_now_ns();
                poll_ret =
                    ibv_poll_cq(payload_cq->complete_queue, poll_cap, wc);
            }
            uint64_t cq_poll_exit_ns = steady_clock_now_ns();
            poll_calls++;
            uint64_t poll_sys_ns = cq_poll_exit_ns - cq_poll_enter_ns;
            cq_poll_sys_ns += poll_sys_ns;
            ctr_peer_payload_cq_lock_wait_ns.fetch_add(
                lock_wait_ns, std::memory_order_relaxed);
            ctr_peer_payload_cq_lock_wait_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_peer_payload_cq_lock_wait_max_ns,
                        lock_wait_ns);
            ctr_peer_payload_cq_poll_sys_ns.fetch_add(
                poll_sys_ns, std::memory_order_relaxed);
            ctr_peer_payload_cq_poll_sys_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_peer_payload_cq_poll_sys_max_ns, poll_sys_ns);
            ASSERT(poll_ret >= 0 && poll_ret <= 64);
            if (poll_ret == 0) {
                ctr_peer_payload_poll_empty.fetch_add(
                    1, std::memory_order_relaxed);
                break;
            }
            ctr_peer_payload_poll_nonempty.fetch_add(
                1, std::memory_order_relaxed);
            ctr_peer_payload_poll_wc_total.fetch_add(
                static_cast<uint64_t>(poll_ret), std::memory_order_relaxed);
            nonempty_polls++;
            wc_total += static_cast<uint64_t>(poll_ret);
            update_peak(ctr_peer_payload_poll_batch_max,
                        static_cast<uint64_t>(poll_ret));
            if (poll_ret == 64) {
                ctr_peer_payload_poll_full_batch.fetch_add(
                    1, std::memory_order_relaxed);
            }
            made_progress = true;
            uint64_t cq_handle_begin_ns = steady_clock_now_ns();
            for (int i = 0; i < poll_ret; i++) {
                auto &work_completion = wc[i];
                if (work_completion.status != IBV_WC_SUCCESS) {
                    std::cerr << "server peer payload completion error"
                              << " status="
                              << ibv_wc_status_str(work_completion.status)
                              << " opcode="
                              << static_cast<int>(work_completion.opcode)
                              << " vendor_err=" << work_completion.vendor_err
                              << " wr_id=0x" << std::hex
                              << work_completion.wr_id << std::dec
                              << " qp_num=" << work_completion.qp_num
                              << std::endl;
                    if (peer_recovery_enabled()) {
                        size_t failed_peer =
                            std::numeric_limits<size_t>::max();
                        auto *lane_slot =
                            reinterpret_cast<PeerLaneSendSlot *>(
                                work_completion.wr_id);
                        if (lane_slot != nullptr &&
                            lane_slot->magic == 0xEC2C2003u) {
                            failed_peer = lane_slot->peer_idx;
                        } else {
                            auto *probe_slot =
                                reinterpret_cast<ProbeParityBatchSendSlot *>(
                                    work_completion.wr_id);
                            if (probe_slot != nullptr &&
                                probe_slot->magic == 0xEC2C2005u) {
                                failed_peer = probe_slot->peer_idx;
                            } else {
                                auto *recv_slot =
                                    reinterpret_cast<PeerPayloadRecvSlot *>(
                                        work_completion.wr_id);
                                if (recv_slot != nullptr &&
                                    recv_slot->magic == 0xEC2C2004u)
                                    failed_peer = recv_slot->peer_idx;
                            }
                        }
                        if (failed_peer < peer_endpoint_count) {
                            mark_peer_dead(
                                failed_peer,
                                rdma::kEC2PCStatusPeerDead);
                            auto *failed_lane =
                                reinterpret_cast<PeerLaneSendSlot *>(
                                    work_completion.wr_id);
                            if (failed_lane != nullptr &&
                                failed_lane->magic == 0xEC2C2003u) {
                                failed_lane->in_use.store(
                                    false, std::memory_order_release);
                                continue;
                            }
                            auto *failed_probe =
                                reinterpret_cast<ProbeParityBatchSendSlot *>(
                                    work_completion.wr_id);
                            if (failed_probe != nullptr &&
                                failed_probe->magic == 0xEC2C2005u) {
                                (void)complete_probe_parity_batch_send_slot(
                                    *failed_probe, steady_clock_now_ns());
                                continue;
                            }
                            auto *failed_recv =
                                reinterpret_cast<PeerPayloadRecvSlot *>(
                                    work_completion.wr_id);
                            if (failed_recv != nullptr &&
                                failed_recv->magic == 0xEC2C2004u)
                                continue;
                            continue;
                        } else {
                            // An unknown failed CQE is not proof of a peer
                            // failure; preserve the legacy assertion.
                            ERROR("server peer payload completion failed");
                        }
                    } else {
                        ERROR("server peer payload completion failed");
                    }
                }
                if (work_completion.opcode == IBV_WC_SEND) {
                    ctr_peer_send_wc.fetch_add(1, std::memory_order_relaxed);
                    auto *lane_slot = reinterpret_cast<PeerLaneSendSlot *>(
                        work_completion.wr_id);
                    if (lane_slot != nullptr &&
                        lane_slot->magic == 0xEC2C2003u) {
                        lane_slot->in_use.store(false, std::memory_order_release);
                        continue;
                    }
                    auto *probe_batch_slot =
                        reinterpret_cast<ProbeParityBatchSendSlot *>(
                            work_completion.wr_id);
                    if (probe_batch_slot != nullptr &&
                        probe_batch_slot->magic == 0xEC2C2005u) {
                        (void)complete_probe_parity_batch_send_slot(
                            *probe_batch_slot, steady_clock_now_ns());
                        continue;
                    }
                    ERROR("server peer payload send: invalid slot");
                }
                if (work_completion.opcode != IBV_WC_RECV) {
                    continue;
                }
                ctr_peer_recv_wc.fetch_add(1, std::memory_order_relaxed);
                uint64_t recv_wc_time_ns = steady_clock_now_ns();
                uint64_t prev_recv_ns =
                    last_peer_payload_recv_time_ns_.exchange(
                        recv_wc_time_ns, std::memory_order_relaxed);
                if (prev_recv_ns != 0 && recv_wc_time_ns >= prev_recv_ns) {
                    uint64_t gap_ns = recv_wc_time_ns - prev_recv_ns;
                    ctr_peer_payload_recv_gap_ns.fetch_add(
                        gap_ns, std::memory_order_relaxed);
                    ctr_peer_payload_recv_gap_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    update_peak(ctr_peer_payload_recv_gap_max_ns, gap_ns);
                }
                auto *payload_slot = reinterpret_cast<PeerPayloadRecvSlot *>(
                    work_completion.wr_id);
                if (payload_slot == nullptr ||
                    payload_slot->magic != 0xEC2C2004u) {
                    ERROR("server peer payload recv: invalid slot");
                }
                size_t payload_q_idx = peer_payload_qp_index(
                    static_cast<size_t>(payload_slot->peer_idx),
                    static_cast<size_t>(payload_slot->shard_idx));
                if (peer_payload_recv_credits != nullptr &&
                    payload_q_idx < peer_payload_recv_credit_count) {
                    uint32_t before = peer_payload_recv_credits[payload_q_idx]
                                          .load(std::memory_order_relaxed);
                    while (true) {
                        if (before == 0) {
                            ctr_peer_payload_recv_credit_zero.fetch_add(
                                1, std::memory_order_relaxed);
                            break;
                        }
                        if (peer_payload_recv_credits[payload_q_idx]
                                .compare_exchange_weak(
                                    before, before - 1,
                                    std::memory_order_relaxed)) {
                            uint64_t after = static_cast<uint64_t>(before - 1);
                            update_min(ctr_peer_payload_recv_credit_min, after);
                            if (after == 0) {
                                ctr_peer_payload_recv_credit_zero.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                            break;
                        }
                    }
                }
                size_t peer_idx = payload_slot->peer_idx;
                uint16_t msg_type = payload_slot->message_type();
                if (msg_type == rdma::EC2PC_MSG_PARITY_PAYLOAD) {
                    const rdma::ProbeParityBatchMessage &payload_msg =
                        *payload_slot->probe_batch_msg();
                    if ((payload_msg.reserved0 & rdma::kEC2PCFlagCompactReq) != 0) {
                        ctr_compact_peer_recv_wc.fetch_add(
                            1, std::memory_order_relaxed);
                        ctr_compact_peer_payload_recv_bytes.fetch_add(
                            rdma::probe_parity_batch_wire_bytes(
                                payload_msg.span_count),
                            std::memory_order_relaxed);
                    }
                    dispatch_peer_probe_parity_batch_task(*payload_slot,
                                                          recv_wc_time_ns);
                    continue;
                }
                const rdma::EC2PCRpcMessage msg = *payload_slot->rpc_msg();
                post_peer_payload_recv_slot(*payload_slot);
                handle_peer_rpc_message(msg, peer_idx, recv_wc_time_ns);
            }
            uint64_t handle_ns = steady_clock_now_ns() - cq_handle_begin_ns;
            cq_handle_ns += handle_ns;
            ctr_peer_payload_cq_handle_ns.fetch_add(
                handle_ns, std::memory_order_relaxed);
            ctr_peer_payload_cq_handle_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_peer_payload_cq_handle_max_ns, handle_ns);
            completions_processed += static_cast<size_t>(poll_ret);
            if (completion_budget != SIZE_MAX &&
                completions_processed >= completion_budget) {
                break;
            }
        }
    }

    uint64_t round_total_ns = steady_clock_now_ns() - round_begin_ns;
    ctr_peer_payload_round_ns.fetch_add(round_total_ns,
                                        std::memory_order_relaxed);
    ctr_peer_payload_round_cnt.fetch_add(1, std::memory_order_relaxed);
    update_peak(ctr_peer_payload_round_max_ns, round_total_ns);
    (void)cq_lock_wait_ns;
    (void)cq_poll_sys_ns;
    (void)cq_handle_ns;
    (void)poll_calls;
    (void)nonempty_polls;
    (void)wc_total;
    return made_progress;
}

// Poll one round of peer-payload CQ work and dispatch parity-side handlers.
inline bool Server::drain_peer_payload_cq_once(size_t max_batches) {
    bool made_progress = flush_peer_payload_transport_once(max_batches);
    made_progress = poll_peer_payload_cq_only_once(max_batches) || made_progress;
    made_progress = flush_peer_payload_transport_once(max_batches) || made_progress;
    return made_progress;
}

// Queue or inline-handle one peer parity batch. The queued path transfers
// recv-slot ownership to the worker, mirroring client batch RPC handling.
inline void Server::dispatch_peer_probe_parity_batch_task(
    PeerPayloadRecvSlot &slot, uint64_t recv_wc_time_ns) {
    if (!should_queue_peer_probe_parity_batch()) {
        const rdma::ProbeParityBatchMessage &msg = *slot.probe_batch_msg();
        bool is_compact =
            (msg.reserved0 & rdma::kEC2PCFlagCompactReq) != 0;
        auto dispatch_begin = std::chrono::steady_clock::now();
        handle_peer_probe_parity_batch_message(
            msg, static_cast<size_t>(slot.peer_idx), recv_wc_time_ns,
            slot.shard_idx);
        auto dispatch_end = std::chrono::steady_clock::now();
        uint64_t dispatch_ns = elapsed_ns(dispatch_begin, dispatch_end);
        ctr_peer_payload_dispatch_ns.fetch_add(
            dispatch_ns, std::memory_order_relaxed);
        ctr_peer_payload_dispatch_cnt.fetch_add(1, std::memory_order_relaxed);
        if (is_compact) {
            ctr_compact_peer_payload_dispatch_ns.fetch_add(
                dispatch_ns, std::memory_order_relaxed);
            ctr_compact_peer_payload_dispatch_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
        auto repost_begin = std::chrono::steady_clock::now();
        post_peer_payload_recv_slot(slot);
        auto repost_end = std::chrono::steady_clock::now();
        uint64_t repost_ns = elapsed_ns(repost_begin, repost_end);
        ctr_peer_payload_repost_ns.fetch_add(
            repost_ns, std::memory_order_relaxed);
        ctr_peer_payload_repost_cnt.fetch_add(1, std::memory_order_relaxed);
        if (is_compact) {
            ctr_compact_peer_payload_repost_ns.fetch_add(
                repost_ns, std::memory_order_relaxed);
            ctr_compact_peer_payload_repost_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
        return;
    }
    const rdma::ProbeParityBatchMessage &msg = *slot.probe_batch_msg();
    bool is_compact = (msg.reserved0 & rdma::kEC2PCFlagCompactReq) != 0;
    size_t worker_idx = peer_probe_parity_batch_worker_idx(slot, msg);
    auto dispatch_begin = std::chrono::steady_clock::now();
    uint16_t peer_idx = slot.peer_idx;
    uint16_t shard_idx = slot.shard_idx;
    uint64_t enqueue_time_ns = steady_clock_now_ns();
    bool enqueued = false;
    auto make_parity_task = [&](size_t target_worker) {
        RpcTask task{};
        task.kind = RpcTaskKind::PeerProbeParityBatch;
        task.peer_payload_slot = &slot;
        task.peer_payload_peer_idx = peer_idx;
        task.peer_payload_shard_idx = shard_idx;
        task.worker_idx = target_worker;
        task.enqueue_time_ns = enqueue_time_ns;
        task.recv_wc_time_ns = recv_wc_time_ns;
        return task;
    };
    {
        std::lock_guard<std::mutex> lock(rpc_task_mutexes[worker_idx]);
        if (rpc_task_queues[worker_idx].size() < kRpcTaskQueueLimit) {
            rpc_task_queues[worker_idx].push_back(make_parity_task(worker_idx));
            enqueued = true;
        }
    }
    if (!enqueued) {
        for (size_t w = 0; w < rpc_worker_count; w++) {
            if (w == worker_idx) {
                continue;
            }
            std::lock_guard<std::mutex> lock(rpc_task_mutexes[w]);
            if (rpc_task_queues[w].size() < kRpcTaskQueueLimit) {
                rpc_task_queues[w].push_back(make_parity_task(w));
                worker_idx = w;
                enqueued = true;
                break;
            }
        }
    }
    if (enqueued) {
        bool is_probe_batch =
            (msg.flags & rdma::kEC2PCFlagProbeBatch) != 0;
        if (!is_probe_batch && recv_wc_time_ns != 0 &&
            enqueue_time_ns >= recv_wc_time_ns) {
            uint64_t recv_to_enqueue_ns = enqueue_time_ns - recv_wc_time_ns;
            ctr_runtime_parity_batch_recv_to_enqueue_ns.fetch_add(
                recv_to_enqueue_ns, std::memory_order_relaxed);
            ctr_runtime_parity_batch_recv_to_enqueue_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_runtime_parity_batch_recv_to_enqueue_max_ns,
                        recv_to_enqueue_ns);
            if (is_compact) {
                ctr_compact_peer_recv_to_enqueue_ns.fetch_add(
                    recv_to_enqueue_ns, std::memory_order_relaxed);
                ctr_compact_peer_recv_to_enqueue_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        ctr_dispatch_enqueued.fetch_add(1, std::memory_order_relaxed);
        rpc_task_cvs[worker_idx].notify_one();
        auto dispatch_end = std::chrono::steady_clock::now();
        uint64_t dispatch_ns = elapsed_ns(dispatch_begin, dispatch_end);
        ctr_peer_payload_dispatch_ns.fetch_add(
            dispatch_ns, std::memory_order_relaxed);
        ctr_peer_payload_dispatch_cnt.fetch_add(1, std::memory_order_relaxed);
        if (is_compact) {
            ctr_compact_peer_payload_dispatch_ns.fetch_add(
                dispatch_ns, std::memory_order_relaxed);
            ctr_compact_peer_payload_dispatch_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
        return;
    }
    ctr_dispatch_queue_full.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(rpc_overflow_mutex);
        rpc_overflow_queue.push_back(make_parity_task(worker_idx));
        rpc_overflow_depth.fetch_add(1, std::memory_order_relaxed);
    }
    ctr_dispatch_overflow.fetch_add(1, std::memory_order_relaxed);
    for (size_t w = 0; w < rpc_worker_count; w++) {
        rpc_task_cvs[w].notify_one();
    }
    auto dispatch_end = std::chrono::steady_clock::now();
    uint64_t dispatch_ns = elapsed_ns(dispatch_begin, dispatch_end);
    ctr_peer_payload_dispatch_ns.fetch_add(
        dispatch_ns, std::memory_order_relaxed);
    ctr_peer_payload_dispatch_cnt.fetch_add(1, std::memory_order_relaxed);
    if (is_compact) {
        ctr_compact_peer_payload_dispatch_ns.fetch_add(
            dispatch_ns, std::memory_order_relaxed);
        ctr_compact_peer_payload_dispatch_cnt.fetch_add(
            1, std::memory_order_relaxed);
    }
}

// Handle one peer RPC message on the parity side, including ACK fan-in.
inline void Server::handle_peer_rpc_message(const rdma::EC2PCRpcMessage &msg,
                                            size_t peer_idx,
                                            uint64_t recv_wc_time_ns) {
    if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
        return;
    if (msg.type == rdma::EC2PC_MSG_PARITY_APPLY) {
        if (msg.wr_id == 0) {
            ERROR("parity apply recv zero wr_id");
        }
        ctr_parity_apply.fetch_add(1, std::memory_order_relaxed);
        auto lock_begin = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> addr_lock(
            address_mutexes[address_lock_shard(msg.target_offset)]);
        auto lock_acquired = std::chrono::steady_clock::now();
        uint64_t wait_ns = elapsed_ns(lock_begin, lock_acquired);
        ctr_addr_lock_wait_ns_parity_apply.fetch_add(
            wait_ns, std::memory_order_relaxed);
        update_peak(ctr_addr_lock_wait_max_ns, wait_ns);
        auto *parity_ptr = static_cast<uint8_t *>(get(msg.target_offset));
        ec2pc_xor_payload_8k(parity_ptr, msg.payload);
        addr_lock.unlock();
        auto lock_release = std::chrono::steady_clock::now();
        uint64_t hold_ns = elapsed_ns(lock_acquired, lock_release);
        ctr_addr_lock_hold_ns_parity_apply.fetch_add(
            hold_ns, std::memory_order_relaxed);
        update_peak(ctr_addr_lock_hold_max_ns, hold_ns);

        rdma::EC2PCRpcMessage ack = make_ec2pc_rpc_message(
            rdma::EC2PC_MSG_ACK_BATCH, 0, 0,
            static_cast<uint32_t>(sizeof(uint64_t)),
            rdma::kEC2PCFlagParityWrite);
        std::memcpy(ack.payload, &msg.wr_id, sizeof(uint64_t));
        static constexpr uint32_t kPeerRetryLimit = 1u << 24;
        if (!post_peer_ack_message(peer_idx, ack)) {
            if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
                return;
            uint32_t retry = 0;
            while (!enqueue_peer_ack_send(peer_idx, ack)) {
                if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
                    return;
                retry++;
                if (retry >= kPeerRetryLimit) {
                    if (!peer_recovery_enabled() || peer_is_alive(peer_idx))
                        ERROR("server parity_apply: peer ack enqueue retry exhausted");
                    return;
                }
            }
        }
        return;
    }

    if (msg.type == rdma::EC2PC_MSG_ACK || msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
        uint64_t ack_dispatch_begin_ns =
            recv_wc_time_ns != 0 ? recv_wc_time_ns : steady_clock_now_ns();
        if (msg.target_endpoint >= peer_endpoint_count)
            ERROR("server peer ack target endpoint is invalid");
        if (msg.status != rdma::kEC2PCStatusOK) {
            if (peer_recovery_enabled()) {
                fail_tracked_requests_for_peer(peer_idx, msg.status);
                return;
            }
            ERROR("server peer ack reported error status");
        }
        if ((msg.flags & rdma::kEC2PCFlagParityWrite) != 0) {
            return;
        }
        uint32_t len = msg.payload_len;
        if (msg.type == rdma::EC2PC_MSG_ACK) {
            len = static_cast<uint32_t>(sizeof(uint64_t));
        }
        if (len > rdma::kEC2PCRpcPayloadBytes ||
            (len % static_cast<uint32_t>(sizeof(uint64_t))) != 0) {
            ERROR("server peer ack payload length invalid");
        }
        const size_t n = len / sizeof(uint64_t);
        for (size_t i = 0; i < n; i++) {
            uint64_t wr_id = (msg.type == rdma::EC2PC_MSG_ACK) ? msg.wr_id : 0;
            if (msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
                std::memcpy(&wr_id, msg.payload + i * sizeof(uint64_t),
                            sizeof(uint64_t));
            }
            uint64_t consume_begin_ns = steady_clock_now_ns();
            bool done = false;
            size_t client_qp_local_idx = 0;
            uint16_t completion_msg_type = rdma::EC2PC_MSG_ACK_BATCH;
            uint16_t target_endpoint = 0;
            uint64_t request_start_time_ns = 0;
            bool consumed = consume_server_ec2pc_ack(
                wr_id, done, client_qp_local_idx, completion_msg_type,
                target_endpoint, request_start_time_ns, peer_idx);
            uint64_t consume_end_ns = steady_clock_now_ns();
            if (consume_end_ns >= consume_begin_ns) {
                ctr_peer_ack_consume_ns.fetch_add(
                    consume_end_ns - consume_begin_ns,
                    std::memory_order_relaxed);
                ctr_peer_ack_consume_cnt.fetch_add(1,
                                                   std::memory_order_relaxed);
            }
            if (!consumed) {
                continue;
            }
            if (!done) {
                ctr_peer_ack_partial_seen.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
            }
            ctr_peer_ack_done_seen.fetch_add(1, std::memory_order_relaxed);
            rdma::EC2PCRpcMessage client_ack = make_ec2pc_rpc_message(
                completion_msg_type, 0, target_endpoint, 0, 0,
                wall_clock_now_ns());
            if (completion_msg_type == rdma::EC2PC_MSG_PROBE_ACK) {
                client_ack.wr_id = wr_id;
                client_ack.payload_len = 0;
            } else {
                client_ack.payload_len =
                    static_cast<uint32_t>(sizeof(uint64_t));
                std::memcpy(client_ack.payload, &wr_id, sizeof(uint64_t));
            }
            if (!enqueue_rpc_send(client_qp_local_idx, client_ack,
                                  consume_end_ns,
                                  request_start_time_ns)) {
                ERROR("server peer ack: client ack enqueue failed");
            }
            ctr_client_final_ack_enqueued.fetch_add(
                1, std::memory_order_relaxed);
        }
        uint64_t ack_dispatch_end_ns = steady_clock_now_ns();
        if (ack_dispatch_end_ns >= ack_dispatch_begin_ns) {
            ctr_peer_ack_recv_dispatch_ns.fetch_add(
                ack_dispatch_end_ns - ack_dispatch_begin_ns,
                std::memory_order_relaxed);
            ctr_peer_ack_recv_dispatch_cnt.fetch_add(1,
                                                     std::memory_order_relaxed);
        }
        return;
    }

    ERROR("server peer rpc: unexpected message type");
}

// Apply one peer parity batch and send the aggregated peer ACK.
inline void Server::handle_peer_probe_parity_batch_message(
    const rdma::ProbeParityBatchMessage &msg, size_t peer_idx,
    uint64_t recv_wc_time_ns, uint16_t recv_shard_idx) {
    if (msg.type != rdma::EC2PC_MSG_PARITY_PAYLOAD) {
        ERROR("server peer probe parity batch: unexpected message type");
    }
    bool is_probe_batch = (msg.flags & rdma::kEC2PCFlagProbeBatch) != 0;
    bool is_compact = (msg.reserved0 & rdma::kEC2PCFlagCompactReq) != 0;
    auto batch_begin = std::chrono::steady_clock::now();
    uint64_t batch_begin_ns = steady_clock_now_ns();
    if (is_probe_batch && msg.send_time_ns != 0 &&
        batch_begin_ns >= msg.send_time_ns) {
        ctr_probe_parity_batch_send_to_handle_ns.fetch_add(
            batch_begin_ns - msg.send_time_ns, std::memory_order_relaxed);
        ctr_probe_parity_batch_send_to_handle_cnt.fetch_add(
            1, std::memory_order_relaxed);
    } else if (is_probe_batch && recv_wc_time_ns != 0 &&
               batch_begin_ns >= recv_wc_time_ns) {
        ctr_probe_parity_batch_send_to_handle_ns.fetch_add(
            batch_begin_ns - recv_wc_time_ns, std::memory_order_relaxed);
        ctr_probe_parity_batch_send_to_handle_cnt.fetch_add(
            1, std::memory_order_relaxed);
    } else if (!is_probe_batch && msg.send_time_ns != 0 &&
               batch_begin_ns >= msg.send_time_ns) {
        ctr_runtime_parity_batch_send_to_handle_ns.fetch_add(
            batch_begin_ns - msg.send_time_ns, std::memory_order_relaxed);
        ctr_runtime_parity_batch_send_to_handle_cnt.fetch_add(
            1, std::memory_order_relaxed);
    }
    if (!is_probe_batch && recv_wc_time_ns != 0 && msg.send_time_ns != 0 &&
        recv_wc_time_ns >= msg.send_time_ns) {
        ctr_runtime_parity_batch_send_to_recv_wc_ns.fetch_add(
            recv_wc_time_ns - msg.send_time_ns, std::memory_order_relaxed);
        ctr_runtime_parity_batch_send_to_recv_wc_cnt.fetch_add(
            1, std::memory_order_relaxed);
    }
    if (!is_probe_batch && recv_wc_time_ns != 0 && msg.post_time_ns != 0 &&
        recv_wc_time_ns >= msg.post_time_ns) {
        uint64_t delta_ns = recv_wc_time_ns - msg.post_time_ns;
        ctr_runtime_parity_batch_post_to_recv_wc_ns.fetch_add(
            delta_ns, std::memory_order_relaxed);
        ctr_runtime_parity_batch_post_to_recv_wc_cnt.fetch_add(
            1, std::memory_order_relaxed);
        (void)delta_ns;
    }
    (void)recv_shard_idx;
    if (!is_probe_batch && recv_wc_time_ns != 0 &&
        batch_begin_ns >= recv_wc_time_ns) {
        uint64_t recv_to_handle_ns = batch_begin_ns - recv_wc_time_ns;
        ctr_runtime_parity_batch_recv_to_handle_ns.fetch_add(
            recv_to_handle_ns, std::memory_order_relaxed);
        ctr_runtime_parity_batch_recv_to_handle_cnt.fetch_add(
            1, std::memory_order_relaxed);
        if (is_compact) {
            ctr_compact_peer_recv_to_handle_ns.fetch_add(
                recv_to_handle_ns, std::memory_order_relaxed);
            ctr_compact_peer_recv_to_handle_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    size_t span_count =
        std::min<size_t>(msg.span_count, rdma::kProbeParityBatchMaxSpans);
    uint16_t ack_count =
        std::min<uint16_t>(msg.ack_count, rdma::kProbeParityBatchMaxSpans);
    int zero_wr_idx = -1;
    for (uint16_t i = 0; i < ack_count; i++) {
        if (msg.wr_ids[i] == 0) {
            zero_wr_idx = static_cast<int>(i);
            break;
        }
    }
    if (ack_count > span_count || zero_wr_idx >= 0) {
        ERROR("peer parity batch bad ack metadata");
    }
    ctr_parity_apply.fetch_add(span_count, std::memory_order_relaxed);
    if (is_probe_batch) {
        ctr_probe_parity_batch_msgs.fetch_add(1, std::memory_order_relaxed);
        ctr_probe_parity_batch_span_ops.fetch_add(span_count,
                                                  std::memory_order_relaxed);
    }
    if (is_compact) {
        ctr_compact_peer_payload_msgs.fetch_add(1, std::memory_order_relaxed);
        ctr_compact_peer_payload_span_ops.fetch_add(
            span_count, std::memory_order_relaxed);
    }
    for (size_t span_idx = 0; span_idx < span_count; span_idx++) {
        uint64_t target_offset = msg.target_offsets[span_idx];
        auto lock_begin = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> addr_lock(
            address_mutexes[address_lock_shard(target_offset)]);
        auto lock_acquired = std::chrono::steady_clock::now();
        uint64_t wait_ns = elapsed_ns(lock_begin, lock_acquired);
        ctr_addr_lock_wait_ns_parity_apply.fetch_add(
            wait_ns, std::memory_order_relaxed);
        update_peak(ctr_addr_lock_wait_max_ns, wait_ns);
        auto get_begin = std::chrono::steady_clock::now();
        auto *parity_ptr = static_cast<uint8_t *>(get(target_offset));
        auto get_end = std::chrono::steady_clock::now();
        uint64_t get_ns = elapsed_ns(get_begin, get_end);
        ctr_probe_parity_batch_get_ns.fetch_add(
            get_ns, std::memory_order_relaxed);
        if (is_compact) {
            ctr_compact_parity_get_ns.fetch_add(
                get_ns, std::memory_order_relaxed);
        }
        auto xor_begin = get_end;
        ec2pc_xor_payload_8k(parity_ptr, msg.payload[span_idx]);
        auto xor_end = std::chrono::steady_clock::now();
        uint64_t xor_ns = elapsed_ns(xor_begin, xor_end);
        ctr_probe_parity_batch_xor_ns.fetch_add(
            xor_ns, std::memory_order_relaxed);
        if (is_compact) {
            ctr_compact_parity_xor_ns.fetch_add(
                xor_ns, std::memory_order_relaxed);
        }
        addr_lock.unlock();
        auto lock_release = std::chrono::steady_clock::now();
        uint64_t hold_ns = elapsed_ns(lock_acquired, lock_release);
        ctr_addr_lock_hold_ns_parity_apply.fetch_add(
            hold_ns, std::memory_order_relaxed);
        update_peak(ctr_addr_lock_hold_max_ns, hold_ns);
    }
    auto batch_apply_end = std::chrono::steady_clock::now();
    if (is_probe_batch) {
        ctr_probe_parity_batch_apply_ns.fetch_add(
            elapsed_ns(batch_begin, batch_apply_end),
            std::memory_order_relaxed);
    }
    if (is_compact) {
        ctr_compact_parity_apply_ns.fetch_add(
            elapsed_ns(batch_begin, batch_apply_end),
            std::memory_order_relaxed);
    }

    rdma::EC2PCRpcMessage ack{};
    ack.type = rdma::EC2PC_MSG_ACK_BATCH;
    ack.status = rdma::kEC2PCStatusOK;
    if (ack_count == 0) {
        ack_count = 1;
        std::memcpy(ack.payload, &msg.wr_id, sizeof(uint64_t));
    } else {
        ack.payload_len = static_cast<uint32_t>(ack_count * sizeof(uint64_t));
        std::memcpy(ack.payload, msg.wr_ids, ack.payload_len);
    }
    if (ack.payload_len == 0) {
        ack.payload_len = static_cast<uint32_t>(sizeof(uint64_t));
    }
    static constexpr uint32_t kPeerRetryLimit = 1u << 24;
    auto ack_post_begin = std::chrono::steady_clock::now();
    if (!post_peer_ack_message(peer_idx, ack)) {
        if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
            return;
        uint32_t retry = 0;
        while (!enqueue_peer_ack_send(peer_idx, ack)) {
            if (peer_recovery_enabled() && !peer_is_alive(peer_idx))
                return;
            retry++;
            if (retry >= kPeerRetryLimit) {
                if (!peer_recovery_enabled() || peer_is_alive(peer_idx))
                    ERROR("server peer probe parity batch: peer ack enqueue retry exhausted");
                return;
            }
        }
    }
    auto ack_post_end = std::chrono::steady_clock::now();
    uint64_t ack_post_end_ns = steady_clock_now_ns();
    uint64_t ack_post_ns = elapsed_ns(ack_post_begin, ack_post_end);
    if (is_compact) {
        ctr_compact_parity_ack_post_ns.fetch_add(
            ack_post_ns, std::memory_order_relaxed);
    }
    if (is_probe_batch) {
        ctr_probe_parity_batch_ack_post_ns.fetch_add(
            elapsed_ns(ack_post_begin, ack_post_end),
            std::memory_order_relaxed);
        if (msg.send_time_ns != 0 && ack_post_end_ns >= msg.send_time_ns) {
            ctr_probe_parity_batch_send_to_ack_post_ns.fetch_add(
                ack_post_end_ns - msg.send_time_ns, std::memory_order_relaxed);
            ctr_probe_parity_batch_send_to_ack_post_cnt.fetch_add(
                1, std::memory_order_relaxed);
        } else if (recv_wc_time_ns != 0 && ack_post_end_ns >= recv_wc_time_ns) {
            ctr_probe_parity_batch_send_to_ack_post_ns.fetch_add(
                ack_post_end_ns - recv_wc_time_ns, std::memory_order_relaxed);
            ctr_probe_parity_batch_send_to_ack_post_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
    } else {
        if (msg.send_time_ns != 0 && ack_post_end_ns >= msg.send_time_ns) {
            ctr_runtime_parity_batch_send_to_ack_post_ns.fetch_add(
                ack_post_end_ns - msg.send_time_ns, std::memory_order_relaxed);
            ctr_runtime_parity_batch_send_to_ack_post_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (recv_wc_time_ns != 0 && ack_post_end_ns >= recv_wc_time_ns) {
            uint64_t recv_to_ack_post_ns = ack_post_end_ns - recv_wc_time_ns;
            ctr_runtime_parity_batch_recv_to_ack_post_ns.fetch_add(
                recv_to_ack_post_ns, std::memory_order_relaxed);
            ctr_runtime_parity_batch_recv_to_ack_post_cnt.fetch_add(
                1, std::memory_order_relaxed);
            if (is_compact) {
                ctr_compact_peer_recv_to_ack_post_ns.fetch_add(
                    recv_to_ack_post_ns, std::memory_order_relaxed);
                ctr_compact_peer_recv_to_ack_post_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
    }
}
