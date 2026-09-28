#pragma once

// NOTE: This file is included by `rdma/server.hpp` after `server_rpc_core.hpp`,
// still inside namespace `FarLib::rdma`.

// Queue one client-facing batch RPC onto a worker or overflow queue.
inline void Server::dispatch_batch_rpc_task(BatchRpcRecvSlot &slot,
                                            uint32_t recv_byte_len) {
    if (slot.magic != 0xEC2C1001u) {
        ERROR("server batch recv: invalid batch slot");
    }
    const rdma::EC2PCDataReqBatchMessage msg = slot.message();
    if (recv_byte_len != 0) {
        uint32_t expected_byte_len = rdma::ec2pc_data_req_batch_wire_bytes(msg);
        if (recv_byte_len != expected_byte_len) {
            ERROR("server batch recv byte_len mismatch");
        }
    }
    ctr_batch_recv_dispatch.fetch_add(1, std::memory_order_relaxed);
    update_peak(ctr_batch_recv_inflight_max,
                ctr_batch_recv_inflight.fetch_add(
                    1, std::memory_order_relaxed) +
                    1);
    uint64_t recv_wc_time_ns = steady_clock_now_ns();
    slot.last_recv_wc_time_ns.store(recv_wc_time_ns,
                                    std::memory_order_release);
    if (rpc_workers.empty()) {
        ctr_dispatch_inline.fetch_add(1, std::memory_order_relaxed);
        auto inline_begin = std::chrono::steady_clock::now();
        handle_batch_rpc_message(slot, 0, recv_wc_time_ns);
        auto inline_end = std::chrono::steady_clock::now();
        uint64_t inline_ns = elapsed_ns(inline_begin, inline_end);
        ctr_dispatch_inline_ns.fetch_add(inline_ns, std::memory_order_relaxed);
        update_peak(ctr_dispatch_inline_max_ns, inline_ns);
        if (inline_ns >= 100000) {
            ctr_dispatch_inline_slow.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    size_t worker_idx = rpc_task_worker_idx(msg);
    uint64_t enqueue_time_ns = steady_clock_now_ns();
    if (recv_wc_time_ns != 0 && enqueue_time_ns >= recv_wc_time_ns) {
        ctr_client_batch_recv_to_enqueue_ns.fetch_add(
            enqueue_time_ns - recv_wc_time_ns, std::memory_order_relaxed);
        ctr_client_batch_recv_to_enqueue_cnt.fetch_add(
            1, std::memory_order_relaxed);
    }
    bool enqueued = false;
    auto make_batch_task = [&](size_t target_worker) {
        RpcTask task{};
        task.kind = RpcTaskKind::ClientBatchRpc;
        task.batch_slot = &slot;
        task.batch_msg_buffer = nullptr;
        task.worker_idx = target_worker;
        task.enqueue_time_ns = enqueue_time_ns;
        task.recv_wc_time_ns = recv_wc_time_ns;
        return task;
    };
    {
        auto lock_begin = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(rpc_task_mutexes[worker_idx]);
        auto lock_acquired = std::chrono::steady_clock::now();
        ctr_dispatch_batch_lock_wait_ns.fetch_add(
            elapsed_ns(lock_begin, lock_acquired), std::memory_order_relaxed);
        ctr_dispatch_batch_lock_wait_cnt.fetch_add(1,
                                                   std::memory_order_relaxed);
        if (rpc_task_queues[worker_idx].size() < kRpcTaskQueueLimit) {
            rpc_task_queues[worker_idx].push_back(make_batch_task(worker_idx));
            enqueued = true;
        }
    }
    if (!enqueued) {
        for (size_t w = 0; w < rpc_worker_count; w++) {
            if (w == worker_idx) {
                continue;
            }
            auto lock_begin = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lock(rpc_task_mutexes[w]);
            auto lock_acquired = std::chrono::steady_clock::now();
            ctr_dispatch_batch_lock_wait_ns.fetch_add(
                elapsed_ns(lock_begin, lock_acquired),
                std::memory_order_relaxed);
            ctr_dispatch_batch_lock_wait_cnt.fetch_add(
                1, std::memory_order_relaxed);
            if (rpc_task_queues[w].size() < kRpcTaskQueueLimit) {
                rpc_task_queues[w].push_back(make_batch_task(w));
                worker_idx = w;
                enqueued = true;
                break;
            }
        }
    }
    if (enqueued) {
        ctr_dispatch_enqueued.fetch_add(1, std::memory_order_relaxed);
        rpc_task_cvs[worker_idx].notify_one();
        return;
    }
    ctr_dispatch_queue_full.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(rpc_overflow_mutex);
        rpc_overflow_queue.push_back(make_batch_task(worker_idx));
        rpc_overflow_depth.fetch_add(1, std::memory_order_relaxed);
    }
    ctr_dispatch_overflow.fetch_add(1, std::memory_order_relaxed);
    for (size_t w = 0; w < rpc_worker_count; w++) {
        rpc_task_cvs[w].notify_one();
    }
}

// Handle one client-facing RPC message from the data QPs.
inline void Server::handle_rpc_message(const rdma::EC2PCRpcMessage &msg,
                                       size_t data_qp_local_idx,
                                       size_t worker_idx) {
    if (msg.type == rdma::EC2PC_MSG_PROBE_REQ) {
        ctr_probe_req.fetch_add(1, std::memory_order_relaxed);
        size_t logical_span_batch =
            std::max<size_t>(1, static_cast<size_t>(msg.payload_len));
        if (config.probe_real_parity_fanout &&
            (msg.flags & rdma::kEC2PCFlagProbeBypassCompute) == 0) {
            size_t data_endpoint_count = probe_real_parity_data_endpoint_count();
            if (peer_endpoint_count < 3 ||
                data_endpoint_count + 2 > peer_endpoint_count) {
                ERROR("server probe_req: real parity fanout requires 4 data + 2 parity style topology");
            }
            if (local_server_index >= data_endpoint_count) {
                ERROR("server probe_req: parity-only endpoint received probe request");
            }
            if (logical_span_batch > rdma::kProbeParityBatchMaxSpans) {
                ERROR("server probe_req: logical span batch exceeds probe parity batch max");
            }
            struct ProbeBatchState {
                std::vector<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>>
                    payload_bank;
                size_t batch = 0;
                bool seeded = false;
            };
            thread_local ProbeBatchState state;
            thread_local volatile uint64_t probe_sink = 0;
            if (state.batch != logical_span_batch) {
                state.payload_bank.resize(logical_span_batch);
                state.batch = logical_span_batch;
                state.seeded = false;
            }
            if (!state.seeded) {
                for (size_t slot = 0; slot < state.batch; slot++) {
                    for (size_t i = 0; i < rdma::kEC2PCRpcPayloadBytes; i++) {
                        state.payload_bank[slot][i] = static_cast<uint8_t>(
                            (i * (29u + static_cast<uint32_t>(slot) * 7u) + 7u +
                             static_cast<uint32_t>(slot) * 19u) &
                            0xffu);
                    }
                }
                state.seeded = true;
            }

            const size_t span_slots =
                std::max<size_t>(1, config.server_buffer_size /
                                        rdma::kEC2PCRpcPayloadBytes);
            const uint16_t parity_endpoint0 =
                static_cast<uint16_t>(data_endpoint_count);
            const uint16_t parity_endpoint1 =
                static_cast<uint16_t>(data_endpoint_count + 1);
            size_t expected_peer_acks =
                2 * static_cast<size_t>(config.probe_sim_compute_rounds);
            if (expected_peer_acks > std::numeric_limits<uint8_t>::max()) {
                ERROR("server probe_req: expected peer ack count overflow");
            }
            track_server_ec2pc_request(
                msg.wr_id, data_qp_local_idx,
                static_cast<uint8_t>(expected_peer_acks),
                rdma::EC2PC_MSG_PROBE_ACK, msg.target_endpoint,
                parity_endpoint0, parity_endpoint1);

            uint64_t seed =
                msg.wr_id ^ (static_cast<uint64_t>(worker_idx) << 32);
            uint64_t probe_real_span_ops = 0;
            uint64_t probe_real_addr_lock_ns = 0;
            uint64_t probe_real_get_ns = 0;
            uint64_t probe_real_encode_ns = 0;
            uint64_t probe_real_memcpy_ns = 0;
            auto probe_begin = std::chrono::steady_clock::now();
            for (uint32_t round = 0; round < config.probe_sim_compute_rounds;
                 round++) {
                static constexpr uint32_t kProbeBatchRetryLimit = 1u << 24;
                auto reserve_begin = std::chrono::steady_clock::now();
                auto *p0_slot = reserve_probe_parity_batch_send_slot(
                    parity_endpoint0);
                uint32_t reserve_retry0 = 0;
                while (p0_slot == nullptr) {
                    reserve_retry0++;
                    if (reserve_retry0 >= kProbeBatchRetryLimit) {
                        ERROR("server probe_req: batched parity0 reserve retry exhausted");
                    }
                    p0_slot =
                        reserve_probe_parity_batch_send_slot(parity_endpoint0);
                }
                auto *p1_slot = reserve_probe_parity_batch_send_slot(
                    parity_endpoint1);
                uint32_t reserve_retry1 = 0;
                while (p1_slot == nullptr) {
                    reserve_retry1++;
                    if (reserve_retry1 >= kProbeBatchRetryLimit) {
                        p0_slot->in_use.store(false, std::memory_order_release);
                        ERROR("server probe_req: batched parity1 reserve retry exhausted");
                    }
                    p1_slot =
                        reserve_probe_parity_batch_send_slot(parity_endpoint1);
                }
                auto reserve_end = std::chrono::steady_clock::now();
                ctr_probe_real_reserve_ns.fetch_add(
                    elapsed_ns(reserve_begin, reserve_end),
                    std::memory_order_relaxed);
                auto &p0 = p0_slot->msg;
                auto &p1 = p1_slot->msg;
                p0 = rdma::ProbeParityBatchMessage{};
                p1 = rdma::ProbeParityBatchMessage{};
                p0.type = rdma::EC2PC_MSG_PARITY_PAYLOAD;
                p0.span_count = static_cast<uint16_t>(logical_span_batch);
                p0.parity_idx = 0;
                p0.flags = rdma::kEC2PCFlagProbeBatch;
                p0.ack_count = 1;
                p0.wr_id = msg.wr_id;
                p0.status = rdma::kEC2PCStatusOK;
                p0.send_time_ns = steady_clock_now_ns();
                p0.wr_ids[0] = msg.wr_id;

                p1.type = rdma::EC2PC_MSG_PARITY_PAYLOAD;
                p1.span_count = static_cast<uint16_t>(logical_span_batch);
                p1.parity_idx = 1;
                p1.flags = rdma::kEC2PCFlagProbeBatch;
                p1.ack_count = 1;
                p1.wr_id = msg.wr_id;
                p1.status = rdma::kEC2PCStatusOK;
                p1.send_time_ns = p0.send_time_ns;
                p1.wr_ids[0] = msg.wr_id;

                for (size_t slot = 0; slot < logical_span_batch; slot++) {
                    probe_real_span_ops++;
                    uint64_t span_id = msg.wr_id * logical_span_batch + slot;
                    uint64_t span_idx = span_id % span_slots;
                    uint64_t data_offset =
                        span_idx * rdma::kEC2PCRpcPayloadBytes;
                    uint64_t parity_offset =
                        span_idx * rdma::kEC2PCRpcPayloadBytes;
                    uint64_t lock_begin_ns = steady_clock_now_ns();
                    std::unique_lock<std::mutex> addr_lock(
                        address_mutexes[address_lock_shard(data_offset)]);
                    uint64_t locked_ns = steady_clock_now_ns();
                    auto *old_data = static_cast<uint8_t *>(get(data_offset));
                    uint64_t got_ns = steady_clock_now_ns();
                    const auto &new_buf =
                        state.payload_bank[static_cast<size_t>(
                            (seed + round + slot) % state.payload_bank.size())];
                    uint8_t slot_idx = static_cast<uint8_t>(
                        span_idx %
                        static_cast<uint64_t>(FarLib::cache::EC_N));

                    p0.target_offsets[slot] = parity_offset;
                    p1.target_offsets[slot] = parity_offset;
                    FarLib::cache::ec_span_delta_encode_update(
                        slot_idx, old_data, new_buf.data(), p0.payload[slot],
                        p1.payload[slot]);
                    uint64_t encoded_ns = steady_clock_now_ns();
                    uint64_t memcpy_ns = encoded_ns;
                    addr_lock.unlock();

                    probe_real_addr_lock_ns += (locked_ns - lock_begin_ns);
                    probe_real_get_ns += (got_ns - locked_ns);
                    probe_real_encode_ns += (encoded_ns - got_ns);
                    probe_real_memcpy_ns += (memcpy_ns - encoded_ns);

                    seed += static_cast<uint64_t>(
                                p0.payload[slot][(round + slot) & 0xffu]) +
                            (static_cast<uint64_t>(
                                 p1.payload[slot][(round + slot + 1) & 0xffu])
                             << 8);
                }

                auto post_begin = std::chrono::steady_clock::now();
                uint32_t retry0 = 0;
                while (!post_reserved_probe_parity_batch_send_slot(*p0_slot)) {
                    retry0++;
                    if (retry0 >= kProbeBatchRetryLimit) {
                        ERROR("server probe_req: batched parity0 send retry exhausted");
                    }
                }
                uint32_t retry1 = 0;
                while (!post_reserved_probe_parity_batch_send_slot(*p1_slot)) {
                    retry1++;
                    if (retry1 >= kProbeBatchRetryLimit) {
                        ERROR("server probe_req: batched parity1 send retry exhausted");
                    }
                }
                auto post_end = std::chrono::steady_clock::now();
                ctr_probe_real_post_ns.fetch_add(
                    elapsed_ns(post_begin, post_end),
                    std::memory_order_relaxed);
            }
            note_server_ec2pc_fanout_ready(msg.wr_id);
            auto probe_end = std::chrono::steady_clock::now();
            ctr_probe_compute_ns.fetch_add(
                elapsed_ns(probe_begin, probe_end),
                std::memory_order_relaxed);
            ctr_probe_real_span_ops.fetch_add(probe_real_span_ops,
                                              std::memory_order_relaxed);
            ctr_probe_real_addr_lock_ns.fetch_add(probe_real_addr_lock_ns,
                                                  std::memory_order_relaxed);
            ctr_probe_real_get_ns.fetch_add(probe_real_get_ns,
                                            std::memory_order_relaxed);
            ctr_probe_real_encode_ns.fetch_add(probe_real_encode_ns,
                                               std::memory_order_relaxed);
            ctr_probe_real_memcpy_ns.fetch_add(probe_real_memcpy_ns,
                                               std::memory_order_relaxed);
            probe_sink += seed;
            return;
        }
        if (config.probe_sim_compute_rounds > 0 &&
            (msg.flags & rdma::kEC2PCFlagProbeBypassCompute) == 0) {
            struct ProbeBatchState {
                std::vector<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>>
                    old_bufs;
                std::vector<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>>
                    p0_bufs;
                std::vector<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>>
                    p1_bufs;
                std::array<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>, 4>
                    payload_bank{};
                size_t batch = 0;
                bool seeded = false;
            };
            thread_local ProbeBatchState state;
            thread_local volatile uint64_t probe_sink = 0;

            size_t batch = logical_span_batch;
            if (state.batch != batch) {
                state.old_bufs.resize(batch);
                state.p0_bufs.resize(batch);
                state.p1_bufs.resize(batch);
                state.batch = batch;
                state.seeded = false;
            }

            if (!state.seeded) {
                for (size_t i = 0; i < rdma::kEC2PCRpcPayloadBytes; i++) {
                    for (size_t slot = 0; slot < state.batch; slot++) {
                        state.old_bufs[slot][i] = static_cast<uint8_t>(
                            (i * 17u + 13u + slot * 23u) & 0xffu);
                    }
                    for (size_t b = 0; b < state.payload_bank.size(); b++) {
                        state.payload_bank[b][i] = static_cast<uint8_t>(
                            (i * (29u + static_cast<uint32_t>(b) * 7u) +
                             7u + static_cast<uint32_t>(b) * 19u) &
                            0xffu);
                    }
                }
                state.seeded = true;
            }

            auto lut = FarLib::cache::ec_span_slot_coeff_lut(
                static_cast<uint8_t>(msg.wr_id %
                                     static_cast<uint64_t>(FarLib::cache::EC_N)));
            uint64_t seed =
                msg.wr_id ^ (static_cast<uint64_t>(worker_idx) << 32);
            auto probe_begin = std::chrono::steady_clock::now();
            for (uint32_t round = 0; round < config.probe_sim_compute_rounds;
                 round++) {
                for (size_t slot = 0; slot < state.batch; slot++) {
                    const auto &new_buf =
                        state.payload_bank[static_cast<size_t>(
                            (seed + round + slot) &
                            (state.payload_bank.size() - 1))];
                    auto &old_buf = state.old_bufs[slot];
                    auto &p0_buf = state.p0_bufs[slot];
                    auto &p1_buf = state.p1_bufs[slot];
                    for (size_t i = 0; i < rdma::kEC2PCRpcPayloadBytes; i++) {
                        uint8_t d =
                            static_cast<uint8_t>(old_buf[i] ^ new_buf[i]);
                        p0_buf[i] = lut.mul0[d];
                        p1_buf[i] = lut.mul1[d];
                    }
                    std::memcpy(old_buf.data(), new_buf.data(),
                                rdma::kEC2PCRpcPayloadBytes);
                    seed += static_cast<uint64_t>(
                                p0_buf[(round + slot) & 0xffu]) +
                            (static_cast<uint64_t>(
                                 p1_buf[(round + slot + 1) & 0xffu])
                             << 8);
                }
            }
            auto probe_end = std::chrono::steady_clock::now();
            ctr_probe_compute_ns.fetch_add(
                elapsed_ns(probe_begin, probe_end),
                std::memory_order_relaxed);
            probe_sink += seed + state.p0_bufs[0][0] +
                          (static_cast<uint64_t>(state.p1_bufs[0][1]) << 8);
        }
        rdma::EC2PCRpcMessage ack{};
        ack.type = rdma::EC2PC_MSG_PROBE_ACK;
        ack.wr_id = msg.wr_id;
        ack.status = rdma::kEC2PCStatusOK;
        ack.target_endpoint = msg.target_endpoint;
        ack.payload_len = 0;
        if (!enqueue_rpc_send(data_qp_local_idx, ack)) {
            ERROR("server probe_req: probe_ack enqueue failed");
        }
        if (ec2pc_preflight_debug_enabled()) {
            static std::atomic<uint64_t> seen{0};
            uint64_t seq = seen.fetch_add(1, std::memory_order_relaxed) + 1;
            if (seq <= 16) {
                std::cerr << "[ec2pc-preflight] server enqueue probe-ack"
                          << " server=" << local_server_index
                          << " worker=" << worker_idx
                          << " data_qp=" << data_qp_local_idx
                          << " wr_id=0x" << std::hex << msg.wr_id
                          << std::dec << std::endl;
            }
        }
        return;
    }

    if (msg.type == rdma::EC2PC_MSG_DATA_REQ) {
        ctr_data_req.fetch_add(1, std::memory_order_relaxed);
        if (msg.payload_len != rdma::kEC2PCRpcPayloadBytes) {
            ERROR("server data_req: unexpected payload length");
        }
        auto lock_begin = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> addr_lock(
            address_mutexes[address_lock_shard(msg.data_offset)]);
        auto lock_acquired = std::chrono::steady_clock::now();
        uint64_t wait_ns = elapsed_ns(lock_begin, lock_acquired);
        ctr_addr_lock_wait_ns_data_req.fetch_add(wait_ns,
                                                 std::memory_order_relaxed);
        update_peak(ctr_addr_lock_wait_max_ns, wait_ns);
        auto *old_data = static_cast<uint8_t *>(get(msg.data_offset));
        const uint8_t *new_data = msg.payload;
        bool init_from_zero = (msg.flags & rdma::kEC2PCFlagInitFromZero) != 0;
        size_t peer_idx0 = static_cast<size_t>(msg.parity_endpoint0);
        size_t peer_idx1 = static_cast<size_t>(msg.parity_endpoint1);
        auto peer_enqueue_begin = std::chrono::steady_clock::now();
        size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
        size_t preferred_shard_idx = config.peer_payload_runtime_rr_shard
                                         ? std::numeric_limits<size_t>::max()
                                         : (worker_idx % payload_qp_count);
        auto choose_runtime_shard = [&](size_t peer_idx) -> size_t {
            if (preferred_shard_idx != std::numeric_limits<size_t>::max()) {
                return preferred_shard_idx % payload_qp_count;
            }
            if (probe_parity_batch_next_shards == nullptr) {
                ERROR("server data_req: probe parity shard picker missing");
            }
            return static_cast<size_t>(
                probe_parity_batch_next_shards[peer_idx].fetch_add(
                    1, std::memory_order_relaxed) %
                payload_qp_count);
        };
        size_t chosen_shard0 = choose_runtime_shard(peer_idx0);
        size_t chosen_shard1 = choose_runtime_shard(peer_idx1);
        rdma::ProbeParityBatchMessage p0{};
        rdma::ProbeParityBatchMessage p1{};
        init_runtime_probe_parity_batch_message(p0, 0, msg.wr_id);
        init_runtime_probe_parity_batch_message(p1, 1, msg.wr_id);
        p0.span_count = 1;
        p0.ack_count = 1;
        p0.wr_ids[0] = msg.wr_id;
        p0.target_offsets[0] = msg.parity_offset0;
        p1.span_count = 1;
        p1.ack_count = 1;
        p1.wr_ids[0] = msg.wr_id;
        p1.target_offsets[0] = msg.parity_offset1;

        auto delta_begin = std::chrono::steady_clock::now();
        FarLib::cache::ec_span_delta_encode_update(
            static_cast<uint8_t>(msg.data_slot), old_data, new_data,
            p0.payload[0], p1.payload[0], init_from_zero);
        auto delta_end = std::chrono::steady_clock::now();
        ctr_data_req_delta_ns.fetch_add(
            elapsed_ns(delta_begin, delta_end),
            std::memory_order_relaxed);

        auto copy_begin = delta_end;
        auto copy_end = delta_end;
        ctr_data_req_copy_ns.fetch_add(
            elapsed_ns(copy_begin, copy_end),
            std::memory_order_relaxed);

        addr_lock.unlock();
        auto lock_release = std::chrono::steady_clock::now();
        uint64_t hold_ns = elapsed_ns(lock_acquired, lock_release);
        ctr_addr_lock_hold_ns_data_req.fetch_add(hold_ns,
                                                 std::memory_order_relaxed);
        update_peak(ctr_addr_lock_hold_max_ns, hold_ns);
        track_server_ec2pc_request(msg.wr_id, data_qp_local_idx, 2,
                                   rdma::EC2PC_MSG_ACK_BATCH, 0,
                                   msg.parity_endpoint0, msg.parity_endpoint1);
        if (!enqueue_runtime_parity_batch_pair_pending_for_worker(
                worker_idx, peer_idx0, p0, chosen_shard0, peer_idx1, p1,
                chosen_shard1)) {
            ERROR("server data_req: runtime parity enqueue failed");
        }
        auto peer_enqueue_end = std::chrono::steady_clock::now();
        ctr_data_req_peer_enqueue_ns.fetch_add(
            elapsed_ns(peer_enqueue_begin, peer_enqueue_end),
            std::memory_order_relaxed);
        note_server_ec2pc_fanout_ready(msg.wr_id, steady_clock_now_ns());
        return;
    }

    if (msg.type == rdma::EC2PC_MSG_PARITY_APPLY) {
        ERROR("server parity_apply on client-facing QP is unsupported");
        return;
    }

    if (msg.type == rdma::EC2PC_MSG_COMPACT_REQ) {
        // msg.data_offset  = src slot offset (live data, will be freed)
        // msg.target_offset = dst slot offset (dead slot, will receive data)
        // msg.data_slot     = EC slot index (same for both, same position in spanset)
        ctr_compact_data_req.fetch_add(1, std::memory_order_relaxed);
        const uint64_t src_offset = msg.data_offset;
        const uint64_t dst_offset = msg.target_offset;
        const uint8_t slot_idx = static_cast<uint8_t>(msg.data_slot);
        size_t peer_idx0 = static_cast<size_t>(msg.parity_endpoint0);
        size_t peer_idx1 = static_cast<size_t>(msg.parity_endpoint1);

        auto lock_begin = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> addr_lock(
            address_mutexes[address_lock_shard(dst_offset)]);
        auto lock_acquired = std::chrono::steady_clock::now();
        uint64_t wait_ns = elapsed_ns(lock_begin, lock_acquired);
        ctr_addr_lock_wait_ns_data_req.fetch_add(wait_ns,
                                                 std::memory_order_relaxed);
        update_peak(ctr_addr_lock_wait_max_ns, wait_ns);

        auto get_begin = std::chrono::steady_clock::now();
        auto *old_data = static_cast<uint8_t *>(get(dst_offset));
        auto *src_data = static_cast<uint8_t *>(get(src_offset));
        auto get_end = std::chrono::steady_clock::now();
        ctr_compact_data_req_get_ns.fetch_add(
            elapsed_ns(get_begin, get_end), std::memory_order_relaxed);

        rdma::ProbeParityBatchMessage p0{};
        rdma::ProbeParityBatchMessage p1{};
        auto msg_init_begin = std::chrono::steady_clock::now();
        init_runtime_probe_parity_batch_message(p0, 0, msg.wr_id);
        init_runtime_probe_parity_batch_message(p1, 1, msg.wr_id);
        p0.reserved0 = rdma::kEC2PCFlagCompactReq;
        p1.reserved0 = rdma::kEC2PCFlagCompactReq;
        p0.span_count = 1;
        p0.ack_count = 1;
        p0.wr_ids[0] = msg.wr_id;
        p0.target_offsets[0] = msg.parity_offset0;
        p1.span_count = 1;
        p1.ack_count = 1;
        p1.wr_ids[0] = msg.wr_id;
        p1.target_offsets[0] = msg.parity_offset1;
        auto msg_init_end = std::chrono::steady_clock::now();
        ctr_compact_data_req_msg_init_ns.fetch_add(
            elapsed_ns(msg_init_begin, msg_init_end),
            std::memory_order_relaxed);

        const bool init_from_zero =
            (msg.flags & rdma::kEC2PCFlagInitFromZero) != 0;
        if (!init_from_zero) {
            ERROR("server compact_req: missing init-from-zero flag");
        }
        // Compact dst slot is logically empty; stale bytes at dst offset are
        // not part of the old parity value.
        uint64_t compact_prepare_ns = 0;
        uint64_t compact_encode_ns = 0;
        auto delta_begin = std::chrono::steady_clock::now();
        FarLib::cache::ec_span_delta_encode_update(
            slot_idx, old_data, src_data,
            p0.payload[0], p1.payload[0], init_from_zero,
            &compact_prepare_ns, &compact_encode_ns);
        auto delta_end = std::chrono::steady_clock::now();
        ctr_compact_data_req_delta_ns.fetch_add(
            elapsed_ns(delta_begin, delta_end), std::memory_order_relaxed);
        ctr_compact_data_req_delta_prepare_ns.fetch_add(
            compact_prepare_ns, std::memory_order_relaxed);
        ctr_compact_data_req_delta_encode_ns.fetch_add(
            compact_encode_ns, std::memory_order_relaxed);
        auto copy_begin = std::chrono::steady_clock::now();
        std::memcpy(old_data, src_data, rdma::kEC2PCRpcPayloadBytes);
        auto copy_end = std::chrono::steady_clock::now();
        ctr_compact_data_req_copy_ns.fetch_add(
            elapsed_ns(copy_begin, copy_end), std::memory_order_relaxed);

        addr_lock.unlock();
        auto lock_release = std::chrono::steady_clock::now();
        uint64_t hold_ns = elapsed_ns(lock_acquired, lock_release);
        ctr_addr_lock_hold_ns_data_req.fetch_add(hold_ns,
                                                 std::memory_order_relaxed);
        update_peak(ctr_addr_lock_hold_max_ns, hold_ns);

        size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
        size_t chosen_shard0 = worker_idx % payload_qp_count;
        size_t chosen_shard1 = worker_idx % payload_qp_count;
        auto peer_enqueue_begin = std::chrono::steady_clock::now();
        track_server_ec2pc_request(msg.wr_id, data_qp_local_idx, 2,
                                   rdma::EC2PC_MSG_ACK_BATCH, 0,
                                   msg.parity_endpoint0, msg.parity_endpoint1);
        if (!enqueue_runtime_parity_batch_pair_pending_for_worker(
                worker_idx, peer_idx0, p0, chosen_shard0, peer_idx1, p1,
                chosen_shard1)) {
            ERROR("server compact_req: runtime parity enqueue failed");
        }
        auto peer_enqueue_end = std::chrono::steady_clock::now();
        ctr_compact_data_req_peer_enqueue_ns.fetch_add(
            elapsed_ns(peer_enqueue_begin, peer_enqueue_end),
            std::memory_order_relaxed);
        auto fanout_ready_begin = std::chrono::steady_clock::now();
        note_server_ec2pc_fanout_ready(msg.wr_id, steady_clock_now_ns());
        auto fanout_ready_end = std::chrono::steady_clock::now();
        ctr_compact_data_req_fanout_ready_ns.fetch_add(
            elapsed_ns(fanout_ready_begin, fanout_ready_end),
            std::memory_order_relaxed);
        return;
    }
}

// Handle one queued client batch RPC copy and publish parity payload batches.
inline void Server::handle_batch_rpc_message(BatchRpcRecvSlot &slot,
                                             size_t worker_idx,
                                             uint64_t recv_wc_time_ns) {
    rdma::EC2PCDataReqBatchMessage &msg = slot.message();
    auto batch_begin = std::chrono::steady_clock::now();
    (void)batch_begin;
    uint64_t worker_begin_time_ns = steady_clock_now_ns();
    uint16_t span_count =
        std::min<uint16_t>(msg.span_count, rdma::kEC2PCDataReqBatchMaxSpans);
    if (span_count == 0) {
        release_batch_rpc_recv_slot(slot);
        return;
    }
    const bool compact_batch = msg.type == rdma::EC2PC_MSG_COMPACT_REQ_BATCH;
    if (msg.type != rdma::EC2PC_MSG_DATA_REQ_BATCH && !compact_batch) {
        ERROR("server batch rpc: unsupported message type");
    }
    int zero_recv_wr_idx = -1;
    for (uint16_t i = 0; i < span_count; i++) {
        if (msg.wr_ids[i] == 0) {
            zero_recv_wr_idx = static_cast<int>(i);
            break;
        }
    }
    if (zero_recv_wr_idx >= 0) {
        ERROR("server batch recv contains zero wr_id");
    }
    size_t client_qp_local_idx = static_cast<size_t>(msg.ack_data_qp_idx);
    if (client_qp_local_idx >= client_data_qp_count()) {
        ERROR("server batch rpc: ack_data_qp_idx out of range");
    }
    ctr_data_req.fetch_add(span_count, std::memory_order_relaxed);
    if (config.ec2pc_direct_final_ack_probe) {
        uint64_t ready_time_ns = steady_clock_now_ns();
        rdma::EC2PCRpcMessage ack = make_ec2pc_rpc_message(
            rdma::EC2PC_MSG_ACK_BATCH, 0, 0,
            static_cast<uint32_t>(span_count * sizeof(uint64_t)));
        std::memcpy(ack.payload, msg.wr_ids,
                    static_cast<size_t>(ack.payload_len));
        if (!enqueue_rpc_send(client_qp_local_idx, ack, ready_time_ns,
                              recv_wc_time_ns)) {
            ERROR("server batch rpc: direct final ack enqueue failed");
        }
        ctr_client_final_ack_enqueued.fetch_add(span_count,
                                                std::memory_order_relaxed);
        release_batch_rpc_recv_slot(slot);
        return;
    }
    size_t peer_idx0 = static_cast<size_t>(msg.parity_endpoint0);
    size_t peer_idx1 = static_cast<size_t>(msg.parity_endpoint1);
    if (peer_idx0 >= peer_endpoint_count || peer_idx1 >= peer_endpoint_count ||
        peer_idx0 == local_server_index || peer_idx1 == local_server_index) {
        ERROR("server batch rpc: invalid parity endpoint");
    }
    uint64_t batch_wait_ns = 0;
    uint64_t batch_hold_ns = 0;
    uint64_t batch_msg_init_ns = 0;
    uint64_t batch_meta_fill_ns = 0;
    uint64_t batch_get_ns = 0;
    uint64_t batch_delta_ns = 0;
    uint64_t batch_delta_prepare_ns = 0;
    uint64_t batch_delta_encode_ns = 0;
    uint64_t batch_copy_ns = 0;
    uint64_t batch_fanout_ready_ns = 0;
    uint64_t batch_reserve_ns = 0;
    uint64_t batch_publish_ns = 0;
    size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
    bool runtime_rr_shard = config.peer_payload_runtime_rr_shard ||
                            payload_qp_count >
                                std::max<size_t>(1, rpc_worker_count);
    size_t preferred_shard_idx = runtime_rr_shard
                                     ? std::numeric_limits<size_t>::max()
                                     : (worker_idx % payload_qp_count);
    auto choose_runtime_shard = [&](size_t peer_idx) -> size_t {
        if (preferred_shard_idx != std::numeric_limits<size_t>::max()) {
            return preferred_shard_idx % payload_qp_count;
        }
        if (probe_parity_batch_next_shards == nullptr) {
            ERROR("server batch rpc: probe parity shard picker missing");
        }
        return static_cast<size_t>(
            probe_parity_batch_next_shards[peer_idx].fetch_add(
                1, std::memory_order_relaxed) %
            payload_qp_count);
    };
    size_t chosen_shard0 = choose_runtime_shard(peer_idx0);
    size_t chosen_shard1 = choose_runtime_shard(peer_idx1);
    auto &p0 = slot.msg_buffer->parity_msgs[0];
    auto &p1 = slot.msg_buffer->parity_msgs[1];
    p0 = rdma::ProbeParityBatchMessage{};
    p1 = rdma::ProbeParityBatchMessage{};
    auto msg_init_begin = std::chrono::steady_clock::now();
    init_runtime_probe_parity_batch_message(p0, 0, msg.wr_ids[0]);
    p0.span_count = span_count;
    p0.ack_count = span_count;
    init_runtime_probe_parity_batch_message(p1, 1, msg.wr_ids[0]);
    p1.span_count = span_count;
    p1.ack_count = span_count;
    if (compact_batch) {
        p0.reserved0 = rdma::kEC2PCFlagCompactReq;
        p1.reserved0 = rdma::kEC2PCFlagCompactReq;
    }
    auto msg_init_end = std::chrono::steady_clock::now();
    batch_msg_init_ns += elapsed_ns(msg_init_begin, msg_init_end);
    for (uint16_t i = 0; i < span_count; i++) {
        auto meta_fill_begin = std::chrono::steady_clock::now();
        uint64_t data_offset = msg.data_offsets[i];
        uint64_t target_offset = compact_batch ? msg.target_offsets[i]
                                               : data_offset;
        uint64_t parity_offset0 =
            compact_batch ? msg.parity_offsets0[i]
                          : rdma::ec2pc_batch_parity_offset_from_data(data_offset);
        uint64_t parity_offset1 =
            compact_batch ? msg.parity_offsets1[i]
                          : rdma::ec2pc_batch_parity_offset_from_data(data_offset);
        uint64_t wr_id = msg.wr_ids[i];
        p0.wr_ids[i] = wr_id;
        p1.wr_ids[i] = wr_id;
        p0.target_offsets[i] = parity_offset0;
        p1.target_offsets[i] = parity_offset1;
        auto meta_fill_end = std::chrono::steady_clock::now();
        batch_meta_fill_ns += elapsed_ns(meta_fill_begin, meta_fill_end);
        auto lock_begin = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> addr_lock(
            address_mutexes[address_lock_shard(target_offset)]);
        auto lock_acquired = std::chrono::steady_clock::now();
        batch_wait_ns += elapsed_ns(lock_begin, lock_acquired);

        auto get_begin = std::chrono::steady_clock::now();
        auto *old_data = static_cast<uint8_t *>(get(target_offset));
        auto *new_data =
            compact_batch ? static_cast<uint8_t *>(get(data_offset))
                          : msg.payload[i];
        auto get_end = std::chrono::steady_clock::now();
        batch_get_ns += elapsed_ns(get_begin, get_end);
        bool init_from_zero = msg.init_from_zero[i] != 0;
        if (compact_batch && !init_from_zero) {
            ERROR("server compact_batch: missing init-from-zero flag");
        }
        auto delta_begin = get_end;
        FarLib::cache::ec_span_delta_encode_update(
            static_cast<uint8_t>(msg.data_slots[i]), old_data, new_data,
            p0.payload[i], p1.payload[i], init_from_zero,
            &batch_delta_prepare_ns, &batch_delta_encode_ns);
        if (compact_batch) {
            auto copy_begin = std::chrono::steady_clock::now();
            std::memcpy(old_data, new_data, rdma::kEC2PCRpcPayloadBytes);
            auto copy_end = std::chrono::steady_clock::now();
            batch_copy_ns += elapsed_ns(copy_begin, copy_end);
        }
        auto delta_end = std::chrono::steady_clock::now();
        batch_delta_ns += elapsed_ns(delta_begin, delta_end);
        addr_lock.unlock();
        auto lock_release = std::chrono::steady_clock::now();
        batch_hold_ns += elapsed_ns(lock_acquired, lock_release);
    }
    auto track_begin = std::chrono::steady_clock::now();
    track_server_ec2pc_batch_requests(
        msg.wr_ids, span_count, client_qp_local_idx, 2,
        rdma::EC2PC_MSG_ACK_BATCH, 0, msg.parity_endpoint0,
        msg.parity_endpoint1, worker_begin_time_ns, recv_wc_time_ns);
    auto track_end = std::chrono::steady_clock::now();

    ctr_addr_lock_wait_ns_data_req.fetch_add(batch_wait_ns,
                                             std::memory_order_relaxed);
    ctr_addr_lock_hold_ns_data_req.fetch_add(batch_hold_ns,
                                             std::memory_order_relaxed);
    ctr_data_req_msg_init_ns.fetch_add(batch_msg_init_ns,
                                       std::memory_order_relaxed);
    ctr_data_req_meta_fill_ns.fetch_add(batch_meta_fill_ns,
                                        std::memory_order_relaxed);
    ctr_data_req_get_ns.fetch_add(batch_get_ns, std::memory_order_relaxed);
    update_peak(ctr_addr_lock_wait_max_ns, batch_wait_ns);
    update_peak(ctr_addr_lock_hold_max_ns, batch_hold_ns);
    ctr_data_req_delta_ns.fetch_add(batch_delta_ns, std::memory_order_relaxed);
    ctr_data_req_delta_prepare_ns.fetch_add(batch_delta_prepare_ns,
                                            std::memory_order_relaxed);
    ctr_data_req_delta_encode_ns.fetch_add(batch_delta_encode_ns,
                                           std::memory_order_relaxed);
    if (compact_batch) {
        ctr_compact_data_req.fetch_add(span_count, std::memory_order_relaxed);
        ctr_compact_data_req_msg_init_ns.fetch_add(
            batch_msg_init_ns, std::memory_order_relaxed);
        ctr_compact_data_req_meta_fill_ns.fetch_add(
            batch_meta_fill_ns, std::memory_order_relaxed);
        ctr_compact_data_req_get_ns.fetch_add(batch_get_ns,
                                              std::memory_order_relaxed);
        ctr_compact_data_req_delta_ns.fetch_add(batch_delta_ns,
                                                std::memory_order_relaxed);
        ctr_compact_data_req_delta_prepare_ns.fetch_add(
            batch_delta_prepare_ns, std::memory_order_relaxed);
        ctr_compact_data_req_delta_encode_ns.fetch_add(
            batch_delta_encode_ns, std::memory_order_relaxed);
        ctr_compact_data_req_copy_ns.fetch_add(batch_copy_ns,
                                               std::memory_order_relaxed);
    }
    ctr_batch_rpc_reserve_ns.fetch_add(batch_reserve_ns,
                                       std::memory_order_relaxed);
    ctr_batch_rpc_reserve_cnt.fetch_add(1, std::memory_order_relaxed);
    ctr_batch_rpc_track_ns.fetch_add(
        elapsed_ns(track_begin, track_end),
        std::memory_order_relaxed);
    ctr_batch_rpc_track_cnt.fetch_add(1, std::memory_order_relaxed);

    auto publish_begin = std::chrono::steady_clock::now();
    uint64_t batch_generation = slot.generation.load(std::memory_order_acquire);
    slot.parity_send_refs.store(2, std::memory_order_release);
    if (!post_probe_parity_batch_message_pair_external(
            peer_idx0, &p0, peer_idx1, &p1, batch_rpc_mr->lkey, &slot,
            batch_generation, chosen_shard0, chosen_shard1)) {
        ERROR("server batch rpc: runtime parity enqueue failed");
    }
    auto publish_end = std::chrono::steady_clock::now();
    batch_publish_ns += elapsed_ns(publish_begin, publish_end);
    ctr_batch_rpc_publish_ns.fetch_add(batch_publish_ns,
                                       std::memory_order_relaxed);
    ctr_batch_rpc_publish_cnt.fetch_add(1, std::memory_order_relaxed);
    if (compact_batch) {
        ctr_compact_data_req_publish_ns.fetch_add(
            batch_publish_ns, std::memory_order_relaxed);
    }

    auto fanout_ready_begin = std::chrono::steady_clock::now();
    uint64_t ready_time_ns = steady_clock_now_ns();
    note_server_ec2pc_batch_fanout_ready(msg.wr_ids, span_count,
                                         ready_time_ns);
    auto fanout_ready_end = std::chrono::steady_clock::now();
    batch_fanout_ready_ns += elapsed_ns(fanout_ready_begin, fanout_ready_end);
    ctr_data_req_fanout_ready_ns.fetch_add(batch_fanout_ready_ns,
                                           std::memory_order_relaxed);
    if (compact_batch) {
        ctr_compact_data_req_fanout_ready_ns.fetch_add(
            batch_fanout_ready_ns, std::memory_order_relaxed);
    }
}

// Handle the synthetic probe batch path on the client-facing side.
inline void Server::handle_probe_batch(const rdma::EC2PCRpcMessage *msgs,
                                       const size_t *data_qp_local_idxs,
                                       size_t count,
                                       size_t worker_idx) {
    if (count == 0) {
        return;
    }

    if (config.probe_real_parity_fanout) {
        for (size_t idx = 0; idx < count; idx++) {
            handle_rpc_message(msgs[idx], data_qp_local_idxs[idx], worker_idx);
        }
        return;
    }

    ctr_probe_req.fetch_add(count, std::memory_order_relaxed);

    if (config.probe_sim_compute_rounds > 0) {
        struct ProbeSpanBatchState {
            std::vector<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>>
                old_bufs;
            std::vector<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>>
                p0_bufs;
            std::vector<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>>
                p1_bufs;
            std::array<std::array<uint8_t, rdma::kEC2PCRpcPayloadBytes>, 4>
                payload_bank{};
            size_t batch = 0;
            bool seeded = false;
        };
        thread_local ProbeSpanBatchState state;
        thread_local volatile uint64_t probe_sink = 0;

        auto probe_begin = std::chrono::steady_clock::now();
        for (size_t idx = 0; idx < count; idx++) {
            if ((msgs[idx].flags & rdma::kEC2PCFlagProbeBypassCompute) != 0) {
                continue;
            }
            size_t logical_span_batch =
                std::max<size_t>(1, static_cast<size_t>(msgs[idx].payload_len));
            if (state.batch != logical_span_batch) {
                state.old_bufs.resize(logical_span_batch);
                state.p0_bufs.resize(logical_span_batch);
                state.p1_bufs.resize(logical_span_batch);
                state.batch = logical_span_batch;
                state.seeded = false;
            }
            if (!state.seeded) {
                for (size_t i = 0; i < rdma::kEC2PCRpcPayloadBytes; i++) {
                    for (size_t slot = 0; slot < state.batch; slot++) {
                        state.old_bufs[slot][i] = static_cast<uint8_t>(
                            (i * 17u + 13u + slot * 23u) & 0xffu);
                    }
                    for (size_t b = 0; b < state.payload_bank.size(); b++) {
                        state.payload_bank[b][i] = static_cast<uint8_t>(
                            (i * (29u + static_cast<uint32_t>(b) * 7u) +
                             7u + static_cast<uint32_t>(b) * 19u) &
                            0xffu);
                    }
                }
                state.seeded = true;
            }
            auto lut = FarLib::cache::ec_span_slot_coeff_lut(
                static_cast<uint8_t>(msgs[idx].wr_id %
                                     static_cast<uint64_t>(FarLib::cache::EC_N)));
            uint64_t seed =
                msgs[idx].wr_id ^ (static_cast<uint64_t>(worker_idx) << 32);
            for (uint32_t round = 0; round < config.probe_sim_compute_rounds;
                 round++) {
                size_t logical_span_batch =
                    std::max<size_t>(1, static_cast<size_t>(msgs[idx].payload_len));
                for (size_t slot = 0; slot < logical_span_batch; slot++) {
                    const auto &new_buf =
                        state.payload_bank[static_cast<size_t>(
                            (seed + round + slot) &
                            (state.payload_bank.size() - 1))];
                    auto &old_buf = state.old_bufs[slot];
                    auto &p0_buf = state.p0_bufs[slot];
                    auto &p1_buf = state.p1_bufs[slot];
                    for (size_t i = 0; i < rdma::kEC2PCRpcPayloadBytes; i++) {
                        uint8_t d =
                            static_cast<uint8_t>(old_buf[i] ^ new_buf[i]);
                        p0_buf[i] = lut.mul0[d];
                        p1_buf[i] = lut.mul1[d];
                    }
                    std::memcpy(old_buf.data(), new_buf.data(),
                                rdma::kEC2PCRpcPayloadBytes);
                    seed += static_cast<uint64_t>(
                                p0_buf[(round + slot) & 0xffu]) +
                            (static_cast<uint64_t>(
                                 p1_buf[(round + slot + 1) & 0xffu])
                             << 8);
                }
            }
            probe_sink += seed + state.p0_bufs[0][0] +
                          (static_cast<uint64_t>(state.p1_bufs[0][1]) << 8);
        }
        auto probe_end = std::chrono::steady_clock::now();
        ctr_probe_compute_ns.fetch_add(
            elapsed_ns(probe_begin, probe_end),
            std::memory_order_relaxed);
    }

    for (size_t idx = 0; idx < count; idx++) {
        rdma::EC2PCRpcMessage ack = make_ec2pc_rpc_message(
            rdma::EC2PC_MSG_PROBE_ACK, msgs[idx].wr_id,
            msgs[idx].target_endpoint);
        if (!enqueue_rpc_send(data_qp_local_idxs[idx], ack)) {
            ERROR("server probe batch: probe_ack enqueue failed");
        }
    }
}
