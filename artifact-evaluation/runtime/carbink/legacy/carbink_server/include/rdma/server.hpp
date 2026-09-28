#pragma once

#include <infiniband/verbs.h>
#include <algorithm>
#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <deque>

#include "rdma/exchange_msg.hpp"
#include "rdma/pending_item.hpp"
#include "cache/ec_split_codec.hpp"
#include "rdma/rdma.hpp"
#include "utils/cpu_cycles.hpp"
#include "utils/debug.hpp"
#include "utils/defer.hpp"

namespace FarLib {
namespace rdma {

inline bool ec2pc_preflight_debug_enabled() {
    static const bool enabled =
        std::getenv("FARLIB_DEBUG_EC2PC_PREFLIGHT") != nullptr;
    return enabled;
}

inline bool ec2pc_server_summary_enabled() {
    static const bool enabled =
        std::getenv("FARLIB_PRINT_SERVER_EC2PC_SUMMARY") != nullptr;
    return enabled;
}

inline std::atomic_bool &server_summary_stop_requested() {
    static std::atomic_bool requested{false};
    return requested;
}

class Server {
private:
    Configure config;
    void *buffer;

    Context ctx;
    ProtectionDomain pd;
    MemoryRegion mr;
    CompleteChannel channel;
    CompleteQueue control_cq;
    std::vector<std::unique_ptr<CompleteQueue>> data_cqs;
    std::vector<size_t> data_qp_to_cq_idx;
    size_t data_cq_count = 0;

    // qps[0]: control qp
    // qps[1...]: data qp
    std::vector<QueuePair> qps;

    struct BatchRpcMessageBuffer {
        rdma::EC2PCDataReqBatchMessage msg{};
        rdma::ProbeParityBatchMessage parity_msgs[2]{};
    };

    struct RpcRecvSlot {
        uint32_t magic = 0xEC2C1001u;
        uint16_t qp_idx = 0;
        uint16_t reserved = 0;
        rdma::EC2PCRpcMessage msg{};
        std::atomic<uint8_t> parity_send_refs{0};
        std::atomic<uint64_t> generation{1};
        std::atomic<uint64_t> last_recv_wc_time_ns{0};
        BatchRpcMessageBuffer *msg_buffer = nullptr;

        bool has_batch_buffer() const { return msg_buffer != nullptr; }

        rdma::EC2PCDataReqBatchMessage &message() {
            if (msg_buffer == nullptr) {
                ERROR("rpc recv slot missing batch msg buffer");
            }
            return msg_buffer->msg;
        }

        const rdma::EC2PCDataReqBatchMessage &message() const {
            if (msg_buffer == nullptr) {
                ERROR("rpc recv slot missing batch msg buffer");
            }
            return msg_buffer->msg;
        }

        rdma::EC2PCRpcMessage &rpc_message() {
            return has_batch_buffer()
                       ? *reinterpret_cast<rdma::EC2PCRpcMessage *>(&msg_buffer->msg)
                       : msg;
        }

        const rdma::EC2PCRpcMessage &rpc_message() const {
            return has_batch_buffer()
                       ? *reinterpret_cast<const rdma::EC2PCRpcMessage *>(
                             &msg_buffer->msg)
                       : msg;
        }

        uint16_t message_type() const { return rpc_message().type; }
    };

    struct RpcSendSlot {
        std::atomic<bool> in_use{false};
        uint32_t magic = 0xEC2C1002u;
        uint16_t qp_idx = 0;
        uint16_t reserved = 0;
        uint64_t post_time_ns = 0;
        rdma::EC2PCRpcMessage msg{};
    };

    struct RpcQueue {
        RpcRecvSlot *recv_slots = nullptr;
        RpcSendSlot *send_slots = nullptr;
        size_t send_head = 0;
    };

    struct PendingRpcSendMeta {
        uint64_t enqueue_sum_ns = 0;
        uint64_t final_ack_ready_sum_ns = 0;
        uint64_t server_recv_start_sum_ns = 0;
        uint32_t ack_count = 0;
    };

    using BatchRpcRecvSlot = RpcRecvSlot;

    struct PeerRpcRecvSlot {
        uint32_t magic = 0xEC2C2001u;
        uint16_t peer_idx = 0;
        uint16_t reserved = 0;
        alignas(64) std::array<uint8_t, rdma::kPeerRpcRecvBytes> bytes{};

        uint16_t message_type() const {
            return *reinterpret_cast<const uint16_t *>(bytes.data());
        }

        rdma::EC2PCRpcMessage *rpc_msg() {
            return reinterpret_cast<rdma::EC2PCRpcMessage *>(bytes.data());
        }

        const rdma::EC2PCRpcMessage *rpc_msg() const {
            return reinterpret_cast<const rdma::EC2PCRpcMessage *>(bytes.data());
        }

        rdma::ProbeParityBatchMessage *probe_batch_msg() {
            return reinterpret_cast<rdma::ProbeParityBatchMessage *>(
                bytes.data());
        }

        const rdma::ProbeParityBatchMessage *probe_batch_msg() const {
            return reinterpret_cast<const rdma::ProbeParityBatchMessage *>(
                bytes.data());
        }
    };

    struct PeerPayloadRecvSlot {
        uint32_t magic = 0xEC2C2004u;
        uint16_t peer_idx = 0;
        uint16_t shard_idx = 0;
        alignas(64) std::array<uint8_t, rdma::kPeerRpcRecvBytes> bytes{};

        uint16_t message_type() const {
            return *reinterpret_cast<const uint16_t *>(bytes.data());
        }

        rdma::EC2PCRpcMessage *rpc_msg() {
            return reinterpret_cast<rdma::EC2PCRpcMessage *>(bytes.data());
        }

        const rdma::EC2PCRpcMessage *rpc_msg() const {
            return reinterpret_cast<const rdma::EC2PCRpcMessage *>(bytes.data());
        }

        rdma::ProbeParityBatchMessage *probe_batch_msg() {
            return reinterpret_cast<rdma::ProbeParityBatchMessage *>(
                bytes.data());
        }

        const rdma::ProbeParityBatchMessage *probe_batch_msg() const {
            return reinterpret_cast<const rdma::ProbeParityBatchMessage *>(
                bytes.data());
        }
    };

    struct PeerRpcSendSlot {
        std::atomic<bool> in_use{false};
        uint32_t magic = 0xEC2C2002u;
        uint16_t peer_idx = 0;
        uint16_t slot_index = 0;
        uint32_t queue_index = 0;
        rdma::EC2PCRpcMessage msg{};
    };

    struct PeerRpcQueue {
        PeerRpcRecvSlot *recv_slots = nullptr;
        PeerRpcSendSlot *send_slots = nullptr;
        uint16_t *free_slot_indices = nullptr;
        size_t free_head = 0;
        size_t free_tail = 0;
        size_t free_count = 0;
    };

    struct PeerPayloadRecvQueue {
        PeerPayloadRecvSlot *recv_slots = nullptr;
    };

    struct PeerLaneSendSlot {
        std::atomic<bool> in_use{false};
        uint32_t magic = 0xEC2C2003u;
        uint16_t peer_idx = 0;
        uint16_t lane_data_qp_idx = 0;
        uint16_t slot_index = 0;
        uint16_t reserved = 0;
        rdma::EC2PCRpcMessage msg{};
    };

    struct PeerLaneSendQueue {
        PeerLaneSendSlot *slots = nullptr;
        size_t next_slot = 0;
        size_t next_shard = 0;
        rdma::EC2PCRpcMessage *pending_msgs = nullptr;
        size_t pending_capacity = 0;
        size_t pending_head = 0;
        size_t pending_tail = 0;
        size_t pending_count = 0;
    };

    struct ProbeParityBatchSendSlot {
        std::atomic<bool> in_use{false};
        std::atomic<uint8_t> state{0};
        uint32_t magic = 0xEC2C2005u;
        uint16_t peer_idx = 0;
        uint16_t shard_idx = 0;
        uint16_t slot_index = 0;
        std::atomic<uint64_t> bind_generation{0};
        uint64_t post_time_ns = 0;
        uint64_t last_flush_bookkeeping_ns = 0;
        uint64_t last_post_send_ns = 0;
        rdma::ProbeParityBatchMessage *msg_ptr = nullptr;
        uint32_t msg_lkey = 0;
        BatchRpcRecvSlot *batch_owner = nullptr;
        uint64_t batch_owner_generation = 0;
        rdma::ProbeParityBatchMessage msg{};

        rdma::ProbeParityBatchMessage *active_msg() {
            return msg_ptr != nullptr ? msg_ptr : &msg;
        }

        const rdma::ProbeParityBatchMessage *active_msg() const {
            return msg_ptr != nullptr ? msg_ptr : &msg;
        }

        void reset_message_binding() {
            msg_ptr = nullptr;
            msg_lkey = 0;
            batch_owner = nullptr;
            batch_owner_generation = 0;
        }
    };

    struct ProbeParityBatchReadyItem {
        uint16_t slot_index = 0;
        uint64_t bind_seq = 0;
        uint64_t ready_enqueue_time_ns = 0;
    };

    struct ProbeParityBatchPendingItem {
        uint16_t preferred_shard_idx = 0;
        rdma::ProbeParityBatchMessage *msg_ptr = nullptr;
        uint32_t msg_lkey = 0;
        BatchRpcRecvSlot *batch_owner = nullptr;
        uint64_t batch_owner_generation = 0;
        rdma::ProbeParityBatchMessage msg{};
    };

    struct ProbeParityBatchSendQueue {
        ProbeParityBatchSendSlot *slots = nullptr;
        size_t depth = 0;
        size_t next_slot = 0;
        size_t next_shard = 0;
        size_t next_ready = 0;
        ProbeParityBatchPendingItem *pending_items = nullptr;
        size_t pending_capacity = 0;
        size_t pending_head = 0;
        size_t pending_tail = 0;
        size_t pending_count = 0;
    };

    // Keep the two runtime parity packets as one transport-owned fanout unit.
    struct ProbeParityBatchPairPendingItem {
        size_t peer_idx0 = 0;
        size_t peer_idx1 = 0;
        uint16_t preferred_shard_idx0 =
            std::numeric_limits<uint16_t>::max();
        uint16_t preferred_shard_idx1 =
            std::numeric_limits<uint16_t>::max();
        uint32_t msg_lkey = 0;
        BatchRpcRecvSlot *batch_owner = nullptr;
        uint64_t batch_owner_generation = 0;
        rdma::ProbeParityBatchMessage *msg0 = nullptr;
        rdma::ProbeParityBatchMessage *msg1 = nullptr;
        uint64_t first_try_time_ns = 0;
        uint64_t pending_enqueue_time_ns = 0;
        uint64_t retry_count = 0;
    };

    // Transport-owned by-value queue item for runtime parity fanout generated
    // by data workers. The worker can return immediately after enqueueing it.
    struct RuntimeParityBatchPairPendingItem {
        size_t peer_idx0 = 0;
        size_t peer_idx1 = 0;
        uint16_t preferred_shard_idx0 =
            std::numeric_limits<uint16_t>::max();
        uint16_t preferred_shard_idx1 =
            std::numeric_limits<uint16_t>::max();
        rdma::ProbeParityBatchMessage msg0{};
        rdma::ProbeParityBatchMessage msg1{};
    };

    struct RuntimeParityBatchPairStorageSlot {
        bool in_use = false;
        size_t peer_idx0 = 0;
        size_t peer_idx1 = 0;
        uint16_t preferred_shard_idx0 =
            std::numeric_limits<uint16_t>::max();
        uint16_t preferred_shard_idx1 =
            std::numeric_limits<uint16_t>::max();
        uint64_t first_try_time_ns = 0;
        uint64_t pending_enqueue_time_ns = 0;
        uint64_t retry_count = 0;
        rdma::ProbeParityBatchMessage msg0{};
        rdma::ProbeParityBatchMessage msg1{};
    };

    struct RuntimeParityBatchPairPendingStorage {
        std::unique_ptr<RuntimeParityBatchPairStorageSlot[]> slots;
        std::unique_ptr<uint64_t[]> free_bitmap;
        size_t capacity = 0;
        size_t free_bitmap_words = 0;
        size_t alloc_hint = 0;
    };

    static constexpr uint8_t kProbeBatchSlotFree = 0;
    static constexpr uint8_t kProbeBatchSlotBusy = 1;
    static constexpr uint8_t kProbeBatchSlotReady = 2;
    static constexpr uint8_t kProbeBatchSlotInflight = 3;

    struct PeerAckSendSlot {
        std::atomic<bool> in_use{false};
        uint32_t magic = 0xEC2C2006u;
        uint16_t peer_idx = 0;
        uint16_t slot_index = 0;
        uint64_t post_time_ns = 0;
        rdma::EC2PCRpcMessage msg{};
    };

    struct PeerAckSendQueue {
        PeerAckSendSlot *slots = nullptr;
        size_t depth = 0;
        size_t next_slot = 0;
    };

    std::vector<RpcQueue> rpc_queues; // index = data qp local index [0, data_qp_count)
    std::vector<uint16_t> recv_credits;
    std::unique_ptr<std::atomic<uint32_t>[]> peer_payload_recv_credits;
    size_t peer_payload_recv_credit_count = 0;
    size_t peer_payload_recv_depth = 0;
    std::vector<uint64_t> recv_qp_wc_counts;
    std::vector<uint64_t> last_recv_qp_wc_counts;
    std::vector<uint8_t> recv_qp_first_seen;
    std::vector<std::deque<rdma::EC2PCRpcMessage>> pending_send_queues;
    std::vector<std::deque<PendingRpcSendMeta>> pending_send_meta_queues;
    std::vector<std::deque<RpcSendSlot *>> inflight_send_slots;
    std::unique_ptr<std::mutex[]> pending_send_queue_mutexes;
    std::unique_ptr<std::mutex[]> inflight_send_slot_mutexes;
    std::atomic<size_t> pending_send_total{0};
    void *rpc_mem = nullptr;
    ibv_mr *rpc_mr = nullptr;

    std::unique_ptr<CompleteQueue> peer_cq;
    std::vector<std::unique_ptr<CompleteQueue>> peer_payload_cqs;
    std::unique_ptr<std::mutex[]> peer_payload_cq_poll_mutexes;
    std::vector<QueuePair> peer_qps;
    std::vector<QueuePair> peer_payload_qps;
    std::vector<PeerRpcQueue> peer_rpc_queues;
    std::vector<PeerPayloadRecvQueue> peer_payload_recv_queues;
    std::vector<PeerLaneSendQueue> peer_lane_send_queues;
    std::vector<ProbeParityBatchSendQueue> probe_parity_batch_send_queues;
    std::vector<std::deque<ProbeParityBatchReadyItem>>
        probe_parity_batch_ready_queues;
    std::vector<PeerAckSendQueue> peer_ack_send_queues;
    size_t peer_lane_depth = 1;
    size_t peer_lane_pending_depth = 0;
    size_t peer_payload_qp_count = 1;
    size_t probe_parity_batch_send_depth = 0;
    size_t probe_parity_batch_pending_depth = 0;
    size_t peer_ack_send_depth = 0;
    std::unique_ptr<rdma::EC2PCRpcMessage[]> peer_lane_pending_mem;
    std::unique_ptr<ProbeParityBatchPendingItem[]> probe_parity_batch_pending_mem;
    std::unique_ptr<std::mutex[]> peer_lane_mutexes;
    std::unique_ptr<std::mutex[]> peer_send_mutexes;
    std::unique_ptr<std::mutex[]> probe_parity_batch_send_mutexes;
    std::unique_ptr<std::mutex[]> probe_parity_batch_pending_mutexes;
    std::unique_ptr<std::condition_variable[]> probe_parity_batch_pending_space_cvs;
    std::unique_ptr<std::mutex[]> probe_parity_batch_ready_mutexes;
    std::unique_ptr<std::atomic<uint32_t>[]> probe_parity_batch_next_shards;
    std::deque<ProbeParityBatchPairPendingItem>
        probe_parity_batch_pair_pending_queue;
    std::mutex probe_parity_batch_pair_pending_mutex;
    std::deque<RuntimeParityBatchPairPendingItem>
        runtime_parity_batch_pair_pending_queue;
    std::mutex runtime_parity_batch_pair_pending_mutex;
    static constexpr size_t kRuntimeParityPendingStorageSlotsPerWorker = 64;
    std::vector<RuntimeParityBatchPairPendingStorage>
        runtime_parity_batch_pair_storage_by_worker;
    std::vector<std::deque<uint32_t>>
        runtime_parity_batch_pair_pending_slot_ids_by_worker;
    std::vector<std::deque<rdma::EC2PCRpcMessage>> pending_peer_send_queues;
    std::vector<std::deque<rdma::EC2PCRpcMessage>> pending_peer_ack_queues;
    std::vector<std::deque<PeerRpcSendSlot *>> inflight_peer_send_slots;
    std::unique_ptr<std::mutex[]> pending_peer_send_queue_mutexes;
    std::unique_ptr<std::mutex[]> peer_ack_send_mutexes;
    std::unique_ptr<std::mutex[]> pending_peer_ack_queue_mutexes;
    std::unique_ptr<std::mutex[]> inflight_peer_send_slot_mutexes;
    void *peer_rpc_mem = nullptr;
    ibv_mr *peer_rpc_mr = nullptr;
    std::atomic<bool> peer_poller_stop{false};
    std::vector<std::thread> peer_pollers;
    std::atomic<size_t> pending_peer_send_total{0};
    size_t peer_endpoint_count = 0;
    size_t local_server_index = 0;
    static constexpr size_t kWorkerTaskBatchPopLimit = 8;

    enum class RpcTaskKind : uint8_t {
        ClientRpc = 0,
        PeerProbeParityBatch = 1,
        ClientBatchRpc = 2,
    };

    struct RpcTask {
        RpcTaskKind kind = RpcTaskKind::ClientRpc;
        rdma::EC2PCRpcMessage msg{};
        BatchRpcRecvSlot *batch_slot = nullptr;
        BatchRpcMessageBuffer *batch_msg_buffer = nullptr;
        PeerPayloadRecvSlot *peer_payload_slot = nullptr;
        uint16_t peer_payload_peer_idx = 0;
        uint16_t peer_payload_shard_idx = 0;
        size_t data_qp_local_idx = 0;
        size_t worker_idx = 0;
        uint64_t enqueue_time_ns = 0;
        uint64_t recv_wc_time_ns = 0;
    };

    struct ServerEC2PCTrackedRequest {
        size_t client_qp_local_idx = 0;
        uint8_t pending_parity_acks = 0;
        uint16_t completion_msg_type = rdma::EC2PC_MSG_ACK_BATCH;
        uint16_t target_endpoint = 0;
        uint64_t start_time_ns = 0;
        uint64_t worker_begin_time_ns = 0;
        uint64_t fanout_ready_time_ns = 0;
        uint64_t first_ack_time_ns = 0;
        uint16_t parity_peer0 = std::numeric_limits<uint16_t>::max();
        uint16_t parity_peer1 = std::numeric_limits<uint16_t>::max();
        uint64_t parity_peer0_post_time_ns = 0;
        uint64_t parity_peer1_post_time_ns = 0;
        uint64_t parity_peer0_ack_time_ns = 0;
        uint64_t parity_peer1_ack_time_ns = 0;
    };
    std::mutex server_ec2pc_track_mutex_;
    std::unordered_map<uint64_t, ServerEC2PCTrackedRequest> server_ec2pc_track_;

    std::atomic<uint64_t> probe_parity_bind_generation_{0};

    struct RuntimeParityBatchBuilder {
        size_t peer_idx = 0;
        rdma::ProbeParityBatchMessage msg{};
        uint64_t first_enqueue_time_ns = 0;
    };
    std::vector<RuntimeParityBatchBuilder> runtime_parity_batches;

    std::deque<RpcTask> rpc_overflow_queue;
    std::mutex rpc_overflow_mutex;
    std::atomic<size_t> rpc_overflow_depth{0};

    size_t rpc_worker_count = 1;
    std::atomic<bool> rpc_worker_stop{false};
    std::vector<std::thread> rpc_workers;
    std::unique_ptr<std::deque<RpcTask>[]> rpc_task_queues;
    std::unique_ptr<std::mutex[]> rpc_task_mutexes;
    std::unique_ptr<std::condition_variable[]> rpc_task_cvs;
    std::unique_ptr<std::atomic<uint64_t>[]> worker_deq_counts;
    std::vector<uint64_t> last_worker_deq_counts;
    std::unique_ptr<std::atomic<uint64_t>[]>
        runtime_parity_batch_ready_by_qp;
    std::unique_ptr<std::atomic<uint64_t>[]>
        runtime_parity_batch_outstanding_by_qp;

    static constexpr size_t kAddressLockShardCount = 4096;
    static constexpr uint64_t kAddressLockGranularityShift = 13; // 8KB span
    std::array<std::mutex, kAddressLockShardCount> address_mutexes;
    void *batch_rpc_mem = nullptr;
    ibv_mr *batch_rpc_mr = nullptr;
    std::vector<BatchRpcMessageBuffer *> batch_rpc_msg_free_pool;
    std::mutex batch_rpc_msg_pool_mutex;
    size_t batch_rpc_msg_buffer_count = 0;
    size_t rpc_queue_depth = kRpcDepth;

    #define FARLIB_SERVER_RPC_SNAPSHOT_FIELDS(X)                             \
        X(recv_wc)                                                           \
        X(send_wc)                                                           \
        X(peer_recv_wc)                                                      \
        X(peer_send_wc)                                                      \
        X(data_req)                                                          \
        X(probe_req)                                                         \
        X(parity_apply)                                                      \
        X(dispatch_enqueued)                                                 \
        X(dispatch_inline)                                                   \
        X(dispatch_overflow)                                                 \
        X(dispatch_inline_ns)                                                \
        X(dispatch_inline_max_ns)                                            \
        X(dispatch_inline_slow)                                              \
        X(dispatch_queue_full)                                               \
        X(worker_task_dequeued)                                              \
        X(worker_queue_wait_ns)                                              \
        X(worker_service_ns)                                                 \
        X(worker_dequeue_lock_wait_ns)                                       \
        X(worker_dequeue_lock_wait_cnt)                                      \
        X(worker_intertask_gap_ns)                                           \
        X(worker_intertask_gap_cnt)                                          \
        X(worker_post_service_queue_empty)                                   \
        X(worker_post_service_queue_nonempty)                                \
        X(worker_post_service_overflow_nonempty)                             \
        X(worker_post_service_queue_depth_sum)                               \
        X(worker_post_service_probe_ns)                                      \
        X(worker_post_service_probe_cnt)                                     \
        X(worker_post_service_probe_lock_wait_ns)                            \
        X(worker_post_service_probe_lock_wait_cnt)                           \
        X(worker_post_task_flush_ns)                                         \
        X(worker_post_task_flush_cnt)                                        \
        X(worker_idle_flush_ns)                                              \
        X(worker_idle_flush_cnt)                                             \
        X(client_batch_task_dequeued)                                        \
        X(client_batch_worker_service_ns)                                    \
        X(client_batch_worker_service_cnt)                                   \
        X(client_batch_recv_to_enqueue_ns)                                   \
        X(client_batch_recv_to_enqueue_cnt)                                  \
        X(client_batch_recv_to_dequeue_ns)                                   \
        X(client_batch_recv_to_dequeue_cnt)                                  \
        X(client_rpc_task_dequeued)                                          \
        X(client_rpc_data_req_task_dequeued)                                 \
        X(client_rpc_probe_req_task_dequeued)                                \
        X(client_rpc_worker_service_ns)                                      \
        X(client_rpc_worker_service_cnt)                                     \
        X(peer_probe_batch_task_dequeued)                                    \
        X(addr_lock_wait_ns_data_req)                                        \
        X(addr_lock_hold_ns_data_req)                                        \
        X(data_req_msg_init_ns)                                              \
        X(data_req_meta_fill_ns)                                             \
        X(data_req_get_ns)                                                   \
        X(data_req_delta_ns)                                                 \
        X(data_req_delta_prepare_ns)                                         \
        X(data_req_delta_encode_ns)                                          \
        X(data_req_fanout_ready_ns)                                          \
        X(data_req_copy_ns)                                                  \
        X(data_req_peer_enqueue_ns)                                          \
        X(compact_data_req)                                                  \
        X(compact_data_req_msg_init_ns)                                      \
        X(compact_data_req_meta_fill_ns)                                     \
        X(compact_data_req_get_ns)                                           \
        X(compact_data_req_delta_ns)                                         \
        X(compact_data_req_delta_prepare_ns)                                 \
        X(compact_data_req_delta_encode_ns)                                  \
        X(compact_data_req_copy_ns)                                          \
        X(compact_data_req_peer_enqueue_ns)                                  \
        X(compact_data_req_publish_ns)                                       \
        X(compact_data_req_fanout_ready_ns)                                  \
        X(compact_peer_payload_msgs)                                         \
        X(compact_peer_payload_span_ops)                                     \
        X(compact_peer_payload_send_bytes)                                   \
        X(compact_peer_payload_recv_bytes)                                   \
        X(compact_peer_post_ns)                                              \
        X(compact_peer_post_cnt)                                             \
        X(compact_peer_send_wc)                                              \
        X(compact_peer_send_cqe_reclaim_ns)                                  \
        X(compact_peer_send_cqe_reclaim_cnt)                                 \
        X(compact_peer_recv_wc)                                              \
        X(compact_peer_recv_to_enqueue_ns)                                   \
        X(compact_peer_recv_to_enqueue_cnt)                                  \
        X(compact_peer_enqueue_to_dequeue_ns)                                \
        X(compact_peer_enqueue_to_dequeue_cnt)                               \
        X(compact_peer_recv_to_handle_ns)                                    \
        X(compact_peer_recv_to_handle_cnt)                                   \
        X(compact_peer_recv_to_ack_post_ns)                                  \
        X(compact_peer_recv_to_ack_post_cnt)                                 \
        X(compact_peer_payload_dispatch_ns)                                  \
        X(compact_peer_payload_dispatch_cnt)                                 \
        X(compact_peer_payload_repost_ns)                                    \
        X(compact_peer_payload_repost_cnt)                                   \
        X(compact_peer_worker_service_ns)                                    \
        X(compact_peer_worker_service_cnt)                                   \
        X(compact_parity_apply_ns)                                           \
        X(compact_parity_get_ns)                                             \
        X(compact_parity_xor_ns)                                             \
        X(compact_parity_ack_post_ns)                                        \
        X(batch_rpc_reserve_ns)                                              \
        X(batch_rpc_reserve_cnt)                                             \
        X(batch_rpc_track_ns)                                                \
        X(batch_rpc_track_cnt)                                               \
        X(batch_rpc_publish_ns)                                              \
        X(batch_rpc_publish_cnt)                                             \
        X(probe_parity_slot_reserve_lock_wait_ns)                            \
        X(probe_parity_slot_reserve_lock_wait_cnt)                           \
        X(probe_parity_slot_reserve_scan_steps)                              \
        X(probe_parity_slot_reserve_calls)                                   \
        X(dispatch_batch_lock_wait_ns)                                       \
        X(dispatch_batch_lock_wait_cnt)                                      \
        X(probe_compute_ns)                                                  \
        X(probe_real_reserve_ns)                                             \
        X(probe_real_post_ns)                                                \
        X(probe_real_span_ops)                                               \
        X(probe_real_addr_lock_ns)                                           \
        X(probe_real_get_ns)                                                 \
        X(probe_real_encode_ns)                                              \
        X(probe_real_memcpy_ns)                                              \
        X(probe_parity_batch_msgs)                                           \
        X(probe_parity_batch_apply_ns)                                       \
        X(probe_parity_batch_ack_post_ns)                                    \
        X(probe_parity_batch_worker_service_ns)                              \
        X(probe_parity_batch_worker_service_cnt)                             \
        X(probe_parity_batch_repost_ns)                                      \
        X(probe_parity_batch_repost_cnt)                                     \
        X(probe_parity_batch_span_ops)                                       \
        X(probe_parity_batch_get_ns)                                         \
        X(probe_parity_batch_xor_ns)                                         \
        X(probe_parity_batch_send_to_handle_ns)                              \
        X(probe_parity_batch_send_to_handle_cnt)                             \
        X(probe_parity_batch_send_to_ack_post_ns)                            \
        X(probe_parity_batch_send_to_ack_post_cnt)                           \
        X(probe_parity_batch_post_to_send_cqe_ns)                            \
        X(probe_parity_batch_post_to_send_cqe_cnt)                           \
        X(runtime_parity_batch_send_to_handle_ns)                            \
        X(runtime_parity_batch_send_to_handle_cnt)                           \
        X(runtime_parity_batch_prepost_ns)                                   \
        X(runtime_parity_batch_prepost_cnt)                                  \
        X(runtime_parity_batch_send_to_recv_wc_ns)                           \
        X(runtime_parity_batch_send_to_recv_wc_cnt)                          \
        X(runtime_parity_batch_post_to_recv_wc_ns)                           \
        X(runtime_parity_batch_post_to_recv_wc_cnt)                          \
        X(runtime_parity_batch_recv_to_enqueue_ns)                           \
        X(runtime_parity_batch_recv_to_enqueue_cnt)                          \
        X(runtime_parity_batch_recv_to_enqueue_max_ns)                       \
        X(runtime_parity_batch_enqueue_to_dequeue_ns)                        \
        X(runtime_parity_batch_enqueue_to_dequeue_cnt)                       \
        X(runtime_parity_batch_enqueue_to_dequeue_max_ns)                    \
        X(runtime_parity_batch_recv_to_handle_ns)                            \
        X(runtime_parity_batch_recv_to_handle_cnt)                           \
        X(runtime_parity_batch_worker_service_ns)                            \
        X(runtime_parity_batch_worker_service_cnt)                           \
        X(runtime_parity_batch_worker_service_max_ns)                        \
        X(runtime_parity_batch_send_to_ack_post_ns)                          \
        X(runtime_parity_batch_send_to_ack_post_cnt)                         \
        X(runtime_parity_batch_recv_to_ack_post_ns)                          \
        X(runtime_parity_batch_recv_to_ack_post_cnt)                         \
        X(runtime_parity_batch_post_to_send_cqe_ns)                          \
        X(runtime_parity_batch_post_to_send_cqe_cnt)                         \
        X(runtime_parity_batch_post_to_send_cqe_max_ns)                      \
        X(runtime_parity_batch_post_fail)                                    \
        X(runtime_parity_batch_reserve_fail)                                 \
        X(runtime_parity_reserve_fail_invalid)                               \
        X(runtime_parity_reserve_fail_no_shard)                              \
        X(runtime_parity_reserve_fail_bad_qidx)                              \
        X(runtime_parity_reserve_fail_no_queue)                              \
        X(runtime_parity_reserve_fail_no_mutex)                              \
        X(runtime_parity_reserve_fail_outstanding)                           \
        X(runtime_parity_reserve_fail_no_free_slot)                          \
        X(runtime_parity_reserve_fail_outstanding_sum)                       \
        X(runtime_parity_reserve_fail_outstanding_cnt)                       \
        X(runtime_parity_reserve_fail_outstanding_max)                       \
        X(runtime_parity_batch_ready_gate_fail)                              \
        X(runtime_parity_batch_ready_gate_slots)                             \
        X(runtime_parity_item_copy_ns)                                       \
        X(runtime_parity_item_copy_cnt)                                      \
        X(runtime_parity_slot_memcpy_ns)                                     \
        X(runtime_parity_slot_memcpy_cnt)                                    \
        X(runtime_parity_pending_queue_ns)                                   \
        X(runtime_parity_pending_queue_cnt)                                  \
        X(runtime_parity_ready_queue_ns)                                     \
        X(runtime_parity_ready_queue_cnt)                                    \
        X(runtime_parity_ready_to_post_ns)                                   \
        X(runtime_parity_ready_to_post_cnt)                                  \
        X(runtime_parity_ready_to_post_max_ns)                               \
        X(runtime_parity_ready_to_gate_fail_ns)                              \
        X(runtime_parity_ready_to_gate_fail_cnt)                             \
        X(runtime_parity_ready_to_gate_fail_max_ns)                          \
        X(runtime_parity_ready_gate_outstanding_sum)                         \
        X(runtime_parity_ready_gate_outstanding_cnt)                         \
        X(runtime_parity_ready_gate_outstanding_max)                         \
        X(runtime_parity_send_cqe_reclaim_ns)                                \
        X(runtime_parity_send_cqe_reclaim_cnt)                               \
        X(runtime_parity_send_cqe_reclaim_max_ns)                            \
        X(runtime_parity_outstanding_dec_before_sum)                         \
        X(runtime_parity_outstanding_dec_before_cnt)                         \
        X(runtime_parity_outstanding_dec_before_max)                         \
        X(runtime_parity_pair_storage_try_success)                           \
        X(runtime_parity_pair_storage_try_fail)                              \
        X(runtime_parity_pair_storage_direct_success_ns)                     \
        X(runtime_parity_pair_storage_direct_success_cnt)                    \
        X(runtime_parity_pair_storage_direct_success_max_ns)                 \
        X(runtime_parity_pair_storage_pending_residency_ns)                  \
        X(runtime_parity_pair_storage_pending_residency_cnt)                 \
        X(runtime_parity_pair_storage_pending_residency_max_ns)              \
        X(runtime_parity_pair_storage_retry_sum)                             \
        X(runtime_parity_pair_storage_retry_cnt)                             \
        X(runtime_parity_pair_storage_retry_max)                             \
        X(runtime_parity_pair_storage_enqueue_depth_sum)                     \
        X(runtime_parity_pair_storage_enqueue_depth_cnt)                     \
        X(runtime_parity_pair_storage_enqueue_depth_max)                     \
        X(runtime_parity_ext_pair_try_success)                               \
        X(runtime_parity_ext_pair_try_fail)                                  \
        X(runtime_parity_ext_pair_direct_success_ns)                         \
        X(runtime_parity_ext_pair_direct_success_cnt)                        \
        X(runtime_parity_ext_pair_direct_success_max_ns)                     \
        X(runtime_parity_ext_pair_pending_residency_ns)                      \
        X(runtime_parity_ext_pair_pending_residency_cnt)                     \
        X(runtime_parity_ext_pair_pending_residency_max_ns)                  \
        X(runtime_parity_ext_pair_retry_sum)                                 \
        X(runtime_parity_ext_pair_retry_cnt)                                 \
        X(runtime_parity_ext_pair_retry_max)                                 \
        X(runtime_parity_ext_pair_enqueue_depth_sum)                         \
        X(runtime_parity_ext_pair_enqueue_depth_cnt)                         \
        X(runtime_parity_ext_pair_enqueue_depth_max)                         \
        X(peer_payload_poll_gap_ns)                                          \
        X(peer_payload_poll_gap_cnt)                                         \
        X(peer_payload_poll_gap_max_ns)                                      \
        X(peer_payload_cq_lock_wait_ns)                                      \
        X(peer_payload_cq_lock_wait_cnt)                                     \
        X(peer_payload_cq_lock_wait_max_ns)                                  \
        X(peer_payload_cq_poll_sys_ns)                                       \
        X(peer_payload_cq_poll_sys_cnt)                                      \
        X(peer_payload_cq_poll_sys_max_ns)                                   \
        X(peer_payload_cq_handle_ns)                                         \
        X(peer_payload_cq_handle_cnt)                                        \
        X(peer_payload_cq_handle_max_ns)                                     \
        X(peer_payload_poll_calls)                                           \
        X(peer_payload_poll_empty)                                           \
        X(peer_payload_poll_nonempty)                                        \
        X(peer_payload_poll_wc_total)                                        \
        X(peer_payload_poll_batch_max)                                       \
        X(peer_payload_poll_full_batch)                                      \
        X(peer_payload_round_ns)                                             \
        X(peer_payload_round_cnt)                                            \
        X(peer_payload_round_max_ns)                                         \
        X(peer_payload_repost_ns)                                            \
        X(peer_payload_repost_cnt)                                           \
        X(peer_payload_dispatch_ns)                                          \
        X(peer_payload_dispatch_cnt)                                         \
        X(peer_payload_lane_flush_ns)                                        \
        X(peer_payload_lane_flush_cnt)                                       \
        X(peer_payload_recv_gap_ns)                                          \
        X(peer_payload_recv_gap_cnt)                                         \
        X(peer_payload_recv_gap_max_ns)                                      \
        X(peer_ack_direct_post_ns)                                           \
        X(peer_ack_direct_post_cnt)                                          \
        X(peer_ack_direct_post_fail_cnt)                                     \
        X(peer_ack_enqueue_ns)                                               \
        X(peer_ack_enqueue_cnt)                                              \
        X(peer_ack_send_cqe_ns)                                              \
        X(peer_ack_send_cqe_cnt)                                             \
        X(peer_ack_recv_dispatch_ns)                                         \
        X(peer_ack_recv_dispatch_cnt)                                        \
        X(peer_ack_consume_ns)                                               \
        X(peer_ack_consume_cnt)                                              \
        X(client_final_ack_posted)                                           \
        X(client_final_ack_done_to_enqueue_ns)                               \
        X(client_final_ack_done_to_enqueue_cnt)                              \
        X(client_final_ack_done_to_enqueue_max_ns)                           \
        X(client_final_ack_enqueue_to_post_ns)                               \
        X(client_final_ack_enqueue_to_post_cnt)                              \
        X(client_final_ack_enqueue_to_post_max_ns)                           \
        X(client_final_ack_done_to_post_ns)                                  \
        X(client_final_ack_done_to_post_cnt)                                 \
        X(client_final_ack_done_to_post_max_ns)                              \
        X(client_final_ack_post_to_send_cqe_ns)                              \
        X(client_final_ack_post_to_send_cqe_cnt)                             \
        X(client_final_ack_post_to_send_cqe_max_ns)                          \
        X(client_batch_send_to_recv_wc_ns)                                   \
        X(client_batch_send_to_recv_wc_cnt)                                  \
        X(server_ec2pc_complete_ns)                                          \
        X(server_ec2pc_complete_cnt)                                         \
        X(server_ec2pc_fanout_ns)                                            \
        X(server_ec2pc_fanout_cnt)                                           \
        X(server_ec2pc_first_ack_after_fanout_ns)                            \
        X(server_ec2pc_first_ack_after_fanout_cnt)                           \
        X(server_ec2pc_parity0_ack_after_fanout_ns)                          \
        X(server_ec2pc_parity0_ack_after_fanout_cnt)                         \
        X(server_ec2pc_parity1_ack_after_fanout_ns)                          \
        X(server_ec2pc_parity1_ack_after_fanout_cnt)                         \
        X(server_ec2pc_final_ack_after_fanout_ns)                            \
        X(server_ec2pc_final_ack_after_fanout_cnt)                           \
        X(server_ec2pc_fanout_to_parity_post_ns)                             \
        X(server_ec2pc_fanout_to_parity_post_cnt)                            \
        X(server_ec2pc_parity_post_to_ack_ns)                                \
        X(server_ec2pc_parity_post_to_ack_cnt)                               \
        X(server_ec2pc_first_to_final_ack_ns)                                \
        X(server_ec2pc_first_to_final_ack_cnt)                               \
        X(server_ec2pc_track_lock_wait_ns)                                   \
        X(server_ec2pc_track_lock_hold_ns)                                   \
        X(server_ec2pc_track_lock_ops)                                       \
        X(peer_lane_busy_ns)                                                 \
        X(peer_lane_busy_retries)                                            \
        X(peer_lane_post_ok_ns)                                              \
        X(peer_lane_post_ok)                                                 \
        X(peer_lane_post_fail)                                               \
        X(peer_lane_enqueue_full)                                            \
        X(peer_send_lock_direct_wait_ns)                                     \
        X(peer_send_lock_direct_hold_ns)                                     \
        X(peer_send_lock_direct_ops)                                         \
        X(peer_send_lock_batch_wait_ns)                                      \
        X(peer_send_lock_batch_hold_ns)                                      \
        X(peer_send_lock_batch_ops)                                          \
        X(peer_send_lock_reclaim_wait_ns)                                    \
        X(peer_send_lock_reclaim_hold_ns)                                    \
        X(peer_send_lock_reclaim_ops)                                        \
        X(addr_lock_wait_ns_parity_apply)                                    \
        X(addr_lock_hold_ns_parity_apply)                                    \
        X(addr_lock_wait_max_ns)                                             \
        X(addr_lock_hold_max_ns)                                             \
        X(enq_payload)                                                       \
        X(enq_ack)                                                           \
        X(post_ok)                                                           \
        X(post_fail)                                                         \
        X(flush_rounds)                                                      \
        X(flush_posts)                                                       \
        X(pending_send_peak)                                                 \
        X(peer_enq_payload)                                                  \
        X(peer_enq_ack)                                                      \
        X(peer_flush_rounds)                                                 \
        X(peer_flush_posts)                                                  \
        X(peer_pending_send_peak)                                            \
        X(poll_loop_count)                                                   \
        X(poll_loop_ns)                                                      \
        X(poll_loop_max_ns)                                                  \
        X(unified_data_phase_ns)                                             \
        X(unified_data_phase_max_ns)                                         \
        X(unified_peer_phase_ns)                                             \
        X(unified_peer_phase_max_ns)                                         \
        X(main_runtime_pair_flush_calls)                                     \
        X(main_runtime_pair_flush_ns)                                        \
        X(main_runtime_pair_flush_max_ns)                                    \
        X(main_runtime_pair_flush_gap_ns)                                    \
        X(main_runtime_pair_flush_gap_cnt)                                   \
        X(main_runtime_pair_flush_gap_max_ns)                                \
        X(main_runtime_pair_flush_jobs)                                      \
        X(main_runtime_pair_flush_active_calls)                              \
        X(main_runtime_pair_flush_last_start_ns)                             \
        X(main_peer_payload_transport_calls)                                 \
        X(main_peer_payload_transport_ns)                                    \
        X(main_peer_payload_transport_max_ns)                                \
        X(main_peer_payload_transport_gap_ns)                                \
        X(main_peer_payload_transport_gap_cnt)                               \
        X(main_peer_payload_transport_gap_max_ns)                            \
        X(main_peer_payload_transport_progress_calls)                        \
        X(main_peer_payload_transport_active_calls)                          \
        X(main_peer_payload_transport_last_start_ns)                         \
        X(main_client_ack_flush_calls)                                       \
        X(main_client_ack_flush_ns)                                          \
        X(main_client_ack_flush_max_ns)                                      \
        X(main_client_ack_flush_gap_ns)                                      \
        X(main_client_ack_flush_gap_cnt)                                     \
        X(main_client_ack_flush_gap_max_ns)                                  \
        X(main_client_ack_flush_posts)                                       \
        X(main_client_ack_flush_active_calls)                                \
        X(main_client_ack_flush_last_start_ns)                               \
        X(completion_poller_loop_ns)                                         \
        X(completion_poller_loop_cnt)                                        \
        X(completion_poller_loop_max_ns)                                     \
        X(completion_poller_gap_ns)                                          \
        X(completion_poller_gap_cnt)                                         \
        X(completion_poller_gap_max_ns)                                      \
        X(completion_poller_last_start_ns)                                   \
        X(completion_poller_data_drain_ns)                                   \
        X(completion_poller_data_drain_cnt)                                  \
        X(completion_poller_data_drain_max_ns)                               \
        X(completion_poller_data_drain_gt1ms)                                \
        X(completion_poller_data_drain_gt10ms)                               \
        X(completion_poller_data_drain_gt100ms)                              \
        X(completion_poller_data_drain_gt500ms)                              \
        X(completion_poller_data_drain_gt1000ms)                             \
        X(completion_poller_peer_ack_drain_ns)                               \
        X(completion_poller_peer_ack_drain_cnt)                              \
        X(completion_poller_peer_ack_drain_max_ns)                           \
        X(completion_poller_payload_poll_ns)                                 \
        X(completion_poller_payload_poll_cnt)                                \
        X(completion_poller_payload_poll_max_ns)                             \
        X(data_drain_poll_sys_ns)                                            \
        X(data_drain_poll_sys_cnt)                                           \
        X(data_drain_poll_sys_max_ns)                                        \
        X(data_drain_handle_ns)                                              \
        X(data_drain_handle_cnt)                                             \
        X(data_drain_handle_max_ns)                                          \
        X(data_drain_dispatch_batch_ns)                                      \
        X(data_drain_dispatch_batch_cnt)                                     \
        X(data_drain_dispatch_batch_max_ns)                                  \
        X(data_drain_wc_total)                                               \
        X(data_drain_call_wc_max)                                            \
        X(data_drain_poll_full_batch)                                        \
        X(poll_calls)                                                        \
        X(poll_nonempty)                                                     \
        X(poll_empty)                                                        \
        X(poll_wc_total)                                                     \
        X(poll_wc_max_batch)                                                 \
        X(batch_recv_poll_calls)                                             \
        X(batch_recv_poll_wc_total)                                          \
        X(batch_recv_poll_wc_max_batch)                                      \
        X(data_cq_poll_calls)                                                \
        X(data_cq_poll_nonempty)                                             \
        X(data_cq_poll_gap_ns)                                               \
        X(data_cq_poll_gap_cnt)                                              \
        X(data_cq_poll_gap_max_ns)                                           \
        X(poll_ns)                                                           \
        X(flush_ns)                                                          \
        X(peer_flush_ns)                                                     \
        X(recv_path_ns)                                                      \
        X(send_path_ns)                                                      \
        X(recv_repost_ok)                                                    \
        X(recv_repost_fail)                                                  \
        X(recv_slot_qp_mismatch)                                             \
        X(recv_credit_zero)                                                  \
        X(peer_payload_recv_credit_zero)                                     \
        X(async_total)                                                       \
        X(async_cq_err)                                                      \
        X(async_qp_fatal)                                                    \
        X(async_qp_req_err)                                                  \
        X(async_qp_access_err)                                               \
        X(async_other)

    #define FARLIB_SERVER_RPC_EXTRA_COUNTER_FIELDS(X)                        \
        X(client_final_ack_enqueued)                                         \
        X(peer_ack_partial_seen)                                             \
        X(peer_ack_done_seen)                                                \
        X(batch_recv_dispatch)                                               \
        X(batch_recv_release)                                                \
        X(batch_recv_inflight)                                               \
        X(batch_recv_inflight_max)

    // Server-side progress counters for identifying whether bottleneck is in
    // worker compute, enqueueing, or poller send progress.
    #define FARLIB_SERVER_DECLARE_RPC_COUNTER(field)                         \
        std::atomic<uint64_t> ctr_##field{0};
    FARLIB_SERVER_RPC_SNAPSHOT_FIELDS(FARLIB_SERVER_DECLARE_RPC_COUNTER)
    FARLIB_SERVER_RPC_EXTRA_COUNTER_FIELDS(FARLIB_SERVER_DECLARE_RPC_COUNTER)
    #undef FARLIB_SERVER_DECLARE_RPC_COUNTER
    std::atomic<uint64_t> ctr_batch_recv_to_repost_ns{0};
    std::atomic<uint64_t> ctr_batch_recv_to_repost_cnt{0};
    std::atomic<uint64_t> ctr_batch_recv_to_repost_max_ns{0};
    std::atomic<uint64_t> last_peer_payload_poll_time_ns_{0};
    std::atomic<uint64_t> last_peer_payload_recv_time_ns_{0};
    std::atomic<uint64_t> ctr_recv_credit_min{
        std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> ctr_peer_payload_recv_credit_min{
        std::numeric_limits<uint64_t>::max()};
    struct CpuUsageSample {
        uint64_t user_ns = 0;
        uint64_t sys_ns = 0;
        bool valid = false;
    };
    struct ThreadClockSample {
        clockid_t clock_id{};
        uint64_t baseline_ns = 0;
        bool valid = false;
    };
    clockid_t main_thread_clock_id{};
    bool main_thread_clock_valid = false;
    uint64_t remote_cpu_reset_wall_ns = 0;
    uint64_t remote_cpu_reset_tsc = 0;
    CpuUsageSample remote_cpu_reset_process{};
    uint64_t remote_cpu_reset_main_thread_ns = 0;
    std::vector<ThreadClockSample> remote_cpu_rpc_worker_clocks;
    std::vector<ThreadClockSample> remote_cpu_completion_poller_clocks;
    uint64_t last_ec2pc_periodic_summary_ns = 0;

public:
    Server(const Configure &config)
        : config(config),
          buffer(
              std::aligned_alloc(FarLib::PAGE_SIZE, config.server_buffer_size)),
          ctx(),
          pd(ctx),
          mr(pd, buffer, config.server_buffer_size),
          channel(ctx),
          control_cq(ctx, config, channel) {
#if FARLIB_EC2PC_VERBOSE_DIAG
        std::cout << "INFO: MR registered. addr=" << buffer
                  << " size=" << config.server_buffer_size
                  << " rkey=" << mr.memory_region->rkey << std::endl;
#endif
        if (buffer != nullptr && config.server_buffer_size != 0) {
            std::memset(buffer, 0, config.server_buffer_size);
        }
    }

    Server(const Server &) = delete;

    ~Server() {
        release_peer_resources();
        release_rpc_resources();
        std::free(buffer);
    }

    void *get(size_t offset) { return static_cast<char *>(buffer) + offset; }

    void reset_ec2pc_measurement_counters() {
        reset_ec2pc_base_counters();
        zero_vectors(last_worker_deq_counts);
        if (worker_deq_counts) {
            for (size_t i = 0; i < rpc_worker_count; i++) {
                worker_deq_counts[i].store(0, std::memory_order_relaxed);
            }
        }

        size_t qp_metric_count =
            peer_endpoint_count * std::max<size_t>(1, peer_payload_qp_count);
        reset_counter_arrays(qp_metric_count, 0,
                             runtime_parity_batch_ready_by_qp,
                             runtime_parity_batch_outstanding_by_qp);
        reset_remote_cpu_profile();
    }

    void start() {
        init_main_thread_cpu_clock();
        // RPC SENDs use IBV_SEND_SIGNALED (not solicited), so the server must
        // subscribe to all CQ events here; solicited-only notify can block
        // forever and prevent RECV handlers from running.
        control_cq.req_notify(false);
        int fd_flags = fcntl(ctx.context->async_fd, F_GETFL, 0);
        if (fd_flags >= 0) {
            (void)fcntl(ctx.context->async_fd, F_SETFL, fd_flags | O_NONBLOCK);
        }
        connect();
        connect_peer_servers();
        init_peer_rpc_resources();
        reset_ec2pc_measurement_counters();
        std::cout << "server started" << std::endl;

        auto post_control_recv = [&]() {
            ibv_recv_wr recv_stop_wr = {
            .wr_id = RQ_STOP,
            .next = nullptr,
            .sg_list = nullptr,
            .num_sge = 0,
        };

            ibv_recv_wr *bad_wr = nullptr;
            QueuePair &qp = qps[0];
            ibv_post_recv(qp.queue_pair, &recv_stop_wr, &bad_wr);
        };
        post_control_recv();

        auto handle_data_completion = [&](const ibv_wc &work_completion) {
            if (work_completion.status != IBV_WC_SUCCESS) {
                std::cerr << "server data completion error"
                          << " status="
                          << ibv_wc_status_str(work_completion.status)
                          << " opcode=" << static_cast<int>(work_completion.opcode)
                          << " vendor_err=" << work_completion.vendor_err
                          << " wr_id=0x" << std::hex << work_completion.wr_id
                          << std::dec
                          << " qp_num=" << work_completion.qp_num
                          << std::endl;
                ERROR("server data completion failed");
            }
            if (work_completion.opcode == IBV_WC_RECV) {
                auto recv_begin = std::chrono::steady_clock::now();
                ctr_recv_wc.fetch_add(1, std::memory_order_relaxed);
                if (work_completion.wr_id == 0) {
                    ERROR("server recv: null wr_id");
                }
                uint32_t slot_magic =
                    *reinterpret_cast<const uint32_t *>(work_completion.wr_id);
                if (slot_magic == 0xEC2C1001u) {
                    auto *slot =
                        reinterpret_cast<RpcRecvSlot *>(work_completion.wr_id);
                    if (slot->qp_idx == 0 ||
                        static_cast<size_t>(slot->qp_idx) >= qps.size()) {
                        ERROR("server recv: invalid slot qp_idx");
                    }
                    size_t data_qp_local_idx =
                        static_cast<size_t>(slot->qp_idx - 1);
                    uint32_t expected_qpn = qps[slot->qp_idx].queue_pair->qp_num;
                    if (work_completion.qp_num != expected_qpn) {
                        ctr_recv_slot_qp_mismatch.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    if (data_qp_local_idx < recv_credits.size()) {
                        if (recv_credits[data_qp_local_idx] == 0) {
                            ctr_recv_credit_zero.fetch_add(
                                1, std::memory_order_relaxed);
                        } else {
                            recv_credits[data_qp_local_idx]--;
                            update_min(ctr_recv_credit_min,
                                       static_cast<uint64_t>(
                                           recv_credits[data_qp_local_idx]));
                        }
                    }
                    if (data_qp_local_idx < recv_qp_wc_counts.size()) {
                        recv_qp_wc_counts[data_qp_local_idx]++;
                        if (data_qp_local_idx < recv_qp_first_seen.size() &&
                            recv_qp_first_seen[data_qp_local_idx] == 0) {
                            recv_qp_first_seen[data_qp_local_idx] = 1;
                        }
                    }
                    if (rdma::ec2pc_is_client_batch_type(slot->message_type())) {
                        record_client_batch_post_wall_to_recv_wc_wall_ns(
                            slot->message());
                        dispatch_batch_rpc_task(*slot, work_completion.byte_len);
                    } else {
                        const rdma::EC2PCRpcMessage msg = slot->rpc_message();
                        post_recv_slot(*slot);
                        ctr_recv_repost_ok.fetch_add(1,
                                                     std::memory_order_relaxed);
                        if (data_qp_local_idx < recv_credits.size() &&
                            recv_credits[data_qp_local_idx] <
                                static_cast<uint16_t>(rpc_queue_depth)) {
                            recv_credits[data_qp_local_idx]++;
                        }
                        dispatch_rpc_task(msg, data_qp_local_idx);
                    }
                } else {
                    ERROR("server recv: invalid slot magic");
                }
                auto recv_end = std::chrono::steady_clock::now();
                ctr_recv_path_ns.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             recv_end - recv_begin)
                                             .count()),
                    std::memory_order_relaxed);
            } else if (work_completion.opcode == IBV_WC_SEND) {
                auto send_begin = std::chrono::steady_clock::now();
                ctr_send_wc.fetch_add(1, std::memory_order_relaxed);
                auto *slot = reinterpret_cast<RpcSendSlot *>(work_completion.wr_id);
                if (slot != nullptr && slot->magic == 0xEC2C1002u) {
                    if (slot->msg.type == rdma::EC2PC_MSG_ACK_BATCH &&
                        slot->post_time_ns != 0) {
                        uint64_t cqe_time_ns = steady_clock_now_ns();
                        if (cqe_time_ns >= slot->post_time_ns) {
                            uint64_t ack_count = static_cast<uint64_t>(
                                std::max<uint32_t>(
                                    1, slot->msg.payload_len /
                                           static_cast<uint32_t>(
                                               sizeof(uint64_t))));
                            uint64_t delta_ns = cqe_time_ns - slot->post_time_ns;
                            ctr_client_final_ack_post_to_send_cqe_ns.fetch_add(
                                delta_ns * ack_count,
                                std::memory_order_relaxed);
                            ctr_client_final_ack_post_to_send_cqe_cnt.fetch_add(
                                ack_count, std::memory_order_relaxed);
                            update_peak(
                                ctr_client_final_ack_post_to_send_cqe_max_ns,
                                delta_ns);
                        }
                    }
                    size_t reclaimed = 0;
                    if (slot->qp_idx > 0) {
                        size_t data_qp_local_idx =
                            static_cast<size_t>(slot->qp_idx - 1);
                        reclaimed = reclaim_inflight_send_slots_upto(
                            data_qp_local_idx, slot);
                    }
                    if (reclaimed == 0) {
                        slot->in_use.store(false, std::memory_order_release);
                    }
                }
                auto send_end = std::chrono::steady_clock::now();
                ctr_send_path_ns.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             send_end - send_begin)
                                             .count()),
                    std::memory_order_relaxed);
            }
        };

        auto drain_data_cq = [&](size_t cq_idx,
                                 size_t max_batches = SIZE_MAX) {
            bool made_progress = false;
            if (cq_idx >= data_cqs.size() || data_cqs[cq_idx] == nullptr) {
                return made_progress;
            }
            const bool probe_batch_enabled =
                rpc_worker_count <= 1 && config.probe_packet_batch_max > 1;
            std::vector<rdma::EC2PCRpcMessage> probe_batch_msgs;
            std::vector<size_t> probe_batch_qps;
            if (probe_batch_enabled) {
                probe_batch_msgs.resize(
                    static_cast<size_t>(config.probe_packet_batch_max));
                probe_batch_qps.resize(
                    static_cast<size_t>(config.probe_packet_batch_max));
            }
            size_t batch_count = 0;
            size_t call_wc_total = 0;
            while (true) {
                ibv_wc wc[64];
                size_t probe_batch_count = 0;
                auto flush_probe_batch = [&]() {
                    if (probe_batch_count == 0) {
                        return;
                    }
                    ctr_dispatch_inline.fetch_add(probe_batch_count,
                                                  std::memory_order_relaxed);
                    auto inline_begin = std::chrono::steady_clock::now();
                    handle_probe_batch(probe_batch_msgs.data(),
                                       probe_batch_qps.data(),
                                       probe_batch_count, 0);
                    auto inline_end = std::chrono::steady_clock::now();
                    uint64_t inline_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            inline_end - inline_begin)
                            .count());
                    ctr_dispatch_inline_ns.fetch_add(
                        inline_ns, std::memory_order_relaxed);
                    update_peak(ctr_dispatch_inline_max_ns, inline_ns);
                    uint64_t per_msg_ns =
                        inline_ns / std::max<uint64_t>(1, probe_batch_count);
                    if (per_msg_ns >= 100000) {
                        ctr_dispatch_inline_slow.fetch_add(
                            probe_batch_count, std::memory_order_relaxed);
                    }
                    probe_batch_count = 0;
                };
                auto poll_begin = std::chrono::steady_clock::now();
                ctr_data_cq_poll_calls.fetch_add(1,
                                                 std::memory_order_relaxed);
                int poll_ret =
                    ibv_poll_cq(data_cqs[cq_idx]->complete_queue, 64, wc);
                auto poll_end = std::chrono::steady_clock::now();
                uint64_t data_poll_ns =
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            poll_end - poll_begin)
                            .count());
                ctr_data_drain_poll_sys_ns.fetch_add(
                    data_poll_ns, std::memory_order_relaxed);
                ctr_data_drain_poll_sys_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                update_peak(ctr_data_drain_poll_sys_max_ns, data_poll_ns);
                ctr_poll_calls.fetch_add(1, std::memory_order_relaxed);
                ctr_poll_ns.fetch_add(
                    data_poll_ns,
                    std::memory_order_relaxed);
                ASSERT(poll_ret >= 0 && poll_ret <= 64);
                if (poll_ret == 0) {
                    ctr_poll_empty.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                made_progress = true;
                ctr_data_cq_poll_nonempty.fetch_add(1,
                                                    std::memory_order_relaxed);
                ctr_poll_nonempty.fetch_add(1, std::memory_order_relaxed);
                ctr_poll_wc_total.fetch_add(static_cast<uint64_t>(poll_ret),
                                            std::memory_order_relaxed);
                ctr_data_drain_wc_total.fetch_add(
                    static_cast<uint64_t>(poll_ret), std::memory_order_relaxed);
                call_wc_total += static_cast<size_t>(poll_ret);
                update_peak(ctr_poll_wc_max_batch,
                            static_cast<uint64_t>(poll_ret));
                if (poll_ret == 64) {
                    ctr_data_drain_poll_full_batch.fetch_add(
                        1, std::memory_order_relaxed);
                }
                for (int i = 0; i < poll_ret; i++) {
                    uint64_t handle_begin_ns = steady_clock_now_ns();
                    auto &work_completion = wc[i];
                    if (work_completion.status != IBV_WC_SUCCESS) {
                        std::cerr << "server data completion error"
                                  << " status="
                                  << ibv_wc_status_str(work_completion.status)
                                  << " opcode="
                                  << static_cast<int>(work_completion.opcode)
                                  << " vendor_err=" << work_completion.vendor_err
                                  << " wr_id=0x" << std::hex
                                  << work_completion.wr_id << std::dec
                                  << " qp_num=" << work_completion.qp_num
                                  << std::endl;
                        ERROR("server data completion failed");
                    }
                    if (work_completion.opcode == IBV_WC_RECV) {
                        auto recv_begin = std::chrono::steady_clock::now();
                        ctr_recv_wc.fetch_add(1, std::memory_order_relaxed);
                        if (work_completion.wr_id == 0) {
                            ERROR("server recv: null wr_id");
                        }
                        uint32_t slot_magic =
                            *reinterpret_cast<const uint32_t *>(
                                work_completion.wr_id);
                        if (slot_magic == 0xEC2C1001u) {
                            auto *slot = reinterpret_cast<RpcRecvSlot *>(
                                work_completion.wr_id);
                            if (slot->qp_idx == 0 ||
                                static_cast<size_t>(slot->qp_idx) >= qps.size()) {
                                ERROR("server recv: invalid slot qp_idx");
                            }
                            size_t data_qp_local_idx =
                                static_cast<size_t>(slot->qp_idx - 1);
                            uint32_t expected_qpn =
                                qps[slot->qp_idx].queue_pair->qp_num;
                            if (work_completion.qp_num != expected_qpn) {
                                ctr_recv_slot_qp_mismatch.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                            if (data_qp_local_idx < recv_credits.size()) {
                                if (recv_credits[data_qp_local_idx] == 0) {
                                    ctr_recv_credit_zero.fetch_add(
                                        1, std::memory_order_relaxed);
                                } else {
                                    recv_credits[data_qp_local_idx]--;
                                    update_min(
                                        ctr_recv_credit_min,
                                        static_cast<uint64_t>(
                                            recv_credits[data_qp_local_idx]));
                                }
                            }
                            if (data_qp_local_idx < recv_qp_wc_counts.size()) {
                                recv_qp_wc_counts[data_qp_local_idx]++;
                                if (data_qp_local_idx <
                                        recv_qp_first_seen.size() &&
                                    recv_qp_first_seen[data_qp_local_idx] == 0) {
                                    recv_qp_first_seen[data_qp_local_idx] = 1;
                                }
                            }
                            if (rdma::ec2pc_is_client_batch_type(
                                    slot->message_type())) {
                                record_client_batch_post_wall_to_recv_wc_wall_ns(
                                    slot->message());
                                flush_probe_batch();
                                uint64_t dispatch_begin_ns =
                                    steady_clock_now_ns();
                                dispatch_batch_rpc_task(
                                    *slot, work_completion.byte_len);
                                uint64_t dispatch_ns =
                                    steady_clock_now_ns() - dispatch_begin_ns;
                                ctr_data_drain_dispatch_batch_ns.fetch_add(
                                    dispatch_ns, std::memory_order_relaxed);
                                ctr_data_drain_dispatch_batch_cnt.fetch_add(
                                    1, std::memory_order_relaxed);
                                update_peak(
                                    ctr_data_drain_dispatch_batch_max_ns,
                                    dispatch_ns);
                            } else {
                                const rdma::EC2PCRpcMessage msg =
                                    slot->rpc_message();
                                post_recv_slot(*slot);
                                ctr_recv_repost_ok.fetch_add(
                                    1, std::memory_order_relaxed);
                                if (ec2pc_preflight_debug_enabled() &&
                                    msg.type == rdma::EC2PC_MSG_PROBE_REQ) {
                                    static std::atomic<uint64_t> seen{0};
                                    uint64_t seq =
                                        seen.fetch_add(1, std::memory_order_relaxed) + 1;
                                    if (seq <= 16) {
                                        std::cerr << "[ec2pc-preflight] server recv probe"
                                                  << " server="
                                                  << local_server_index
                                                  << " cq=" << cq_idx
                                                  << " data_qp="
                                                  << data_qp_local_idx
                                                  << " qp_num="
                                                  << work_completion.qp_num
                                                  << " wr_id=0x" << std::hex
                                                  << msg.wr_id << std::dec
                                                  << std::endl;
                                    }
                                }
                                if (data_qp_local_idx < recv_credits.size() &&
                                    recv_credits[data_qp_local_idx] <
                                        static_cast<uint16_t>(rpc_queue_depth)) {
                                    recv_credits[data_qp_local_idx]++;
                                }
                                bool batch_probe_inline =
                                    probe_batch_enabled &&
                                    msg.type == rdma::EC2PC_MSG_PROBE_REQ;
                                if (batch_probe_inline) {
                                    probe_batch_msgs[probe_batch_count] = msg;
                                    probe_batch_qps[probe_batch_count] =
                                        data_qp_local_idx;
                                    probe_batch_count++;
                                    if (probe_batch_count >=
                                        static_cast<size_t>(
                                            config.probe_packet_batch_max)) {
                                        flush_probe_batch();
                                    }
                                } else {
                                    flush_probe_batch();
                                    dispatch_rpc_task(msg, data_qp_local_idx);
                                }
                            }
                        } else {
                            ERROR("server recv: invalid slot magic");
                        }
                        auto recv_end = std::chrono::steady_clock::now();
                        ctr_recv_path_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<
                                    std::chrono::nanoseconds>(recv_end -
                                                              recv_begin)
                                    .count()),
                            std::memory_order_relaxed);
                    } else {
                        flush_probe_batch();
                        handle_data_completion(work_completion);
                    }
                    uint64_t handle_ns = steady_clock_now_ns() - handle_begin_ns;
                    ctr_data_drain_handle_ns.fetch_add(
                        handle_ns, std::memory_order_relaxed);
                    ctr_data_drain_handle_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    update_peak(ctr_data_drain_handle_max_ns, handle_ns);
                }
                flush_probe_batch();
                batch_count++;
                if (batch_count >= max_batches) {
                    break;
                }

            }
            update_peak(ctr_data_drain_call_wc_max,
                        static_cast<uint64_t>(call_wc_total));
            return made_progress;
        };

        auto drain_peer_cq = [&](size_t max_batches = SIZE_MAX) {
            bool made_progress = false;
            if (peer_cq == nullptr || peer_rpc_queues.empty()) {
                return made_progress;
            }
            size_t batch_count = 0;
            while (true) {
                ibv_wc wc[64];
                int poll_ret = ibv_poll_cq(peer_cq->complete_queue, 64, wc);
                ASSERT(poll_ret >= 0 && poll_ret <= 64);
                if (poll_ret == 0) {
                    break;
                }
                made_progress = true;
                for (int i = 0; i < poll_ret; i++) {
                    auto &work_completion = wc[i];
                    if (work_completion.status != IBV_WC_SUCCESS) {
                        std::cerr << "server peer completion error"
                                  << " status="
                                  << ibv_wc_status_str(work_completion.status)
                                  << " opcode="
                                  << static_cast<int>(work_completion.opcode)
                                  << " vendor_err="
                                  << work_completion.vendor_err
                                  << " wr_id=0x" << std::hex
                                  << work_completion.wr_id << std::dec
                                  << " qp_num=" << work_completion.qp_num
                                  << std::endl;
                        ERROR("server peer completion failed");
                    }
                    if (work_completion.opcode == IBV_WC_SEND) {
                        ctr_peer_send_wc.fetch_add(1,
                                                   std::memory_order_relaxed);
                        auto *slot = reinterpret_cast<PeerRpcSendSlot *>(
                            work_completion.wr_id);
                        if (slot != nullptr && slot->magic == 0xEC2C2002u) {
                            size_t reclaimed =
                                reclaim_inflight_peer_send_slots_upto(
                                    static_cast<size_t>(slot->queue_index), slot);
                            if (reclaimed == 0) {
                                release_peer_send_slot(*slot);
                            }
                        } else {
                            auto *ack_slot =
                                reinterpret_cast<PeerAckSendSlot *>(
                                    work_completion.wr_id);
                            if (ack_slot != nullptr &&
                                ack_slot->magic == 0xEC2C2006u) {
                                uint64_t cqe_time_ns = steady_clock_now_ns();
                                if (ack_slot->post_time_ns != 0 &&
                                    cqe_time_ns >= ack_slot->post_time_ns) {
                                    ctr_peer_ack_send_cqe_ns.fetch_add(
                                        cqe_time_ns - ack_slot->post_time_ns,
                                        std::memory_order_relaxed);
                                    ctr_peer_ack_send_cqe_cnt.fetch_add(
                                        1, std::memory_order_relaxed);
                                }
                                ack_slot->post_time_ns = 0;
                                ack_slot->in_use.store(
                                    false, std::memory_order_release);
                                (void)flush_pending_peer_acks(
                                    std::max<size_t>(1, peer_ack_send_depth),
                                    static_cast<size_t>(ack_slot->peer_idx),
                                    peer_endpoint_count);
                            } else {
                            auto *lane_slot =
                                reinterpret_cast<PeerLaneSendSlot *>(
                                    work_completion.wr_id);
                            if (lane_slot != nullptr &&
                                lane_slot->magic == 0xEC2C2003u) {
                                lane_slot->in_use.store(
                                    false, std::memory_order_release);
                            } else {
                                auto *probe_batch_slot =
                                    reinterpret_cast<ProbeParityBatchSendSlot *>(
                                        work_completion.wr_id);
                                if (probe_batch_slot != nullptr &&
                                    probe_batch_slot->magic ==
                                        0xEC2C2005u) {
                                    (void)complete_probe_parity_batch_send_slot(
                                        *probe_batch_slot,
                                        steady_clock_now_ns());
                                }
                            }
                            }
                        }
                    } else if (work_completion.opcode == IBV_WC_RECV) {
                        ctr_peer_recv_wc.fetch_add(1,
                                                   std::memory_order_relaxed);
                        uint64_t recv_wc_time_ns = steady_clock_now_ns();
                        auto *slot = reinterpret_cast<PeerRpcRecvSlot *>(
                            work_completion.wr_id);
                        if (slot != nullptr && slot->magic == 0xEC2C2001u) {
                            size_t peer_idx = slot->peer_idx;
                            uint16_t msg_type = slot->message_type();
                            if (msg_type == rdma::EC2PC_MSG_PARITY_PAYLOAD) {
                                const rdma::ProbeParityBatchMessage msg =
                                    *slot->probe_batch_msg();
                                post_peer_recv_slot(*slot);
                                handle_peer_probe_parity_batch_message(
                                    msg, peer_idx, recv_wc_time_ns);
                                continue;
                            }
                            const rdma::EC2PCRpcMessage msg = *slot->rpc_msg();
                            post_peer_recv_slot(*slot);
                            handle_peer_rpc_message(msg, peer_idx,
                                                    recv_wc_time_ns);
                        } else {
                            auto *payload_slot =
                                reinterpret_cast<PeerPayloadRecvSlot *>(
                                    work_completion.wr_id);
                            if (payload_slot == nullptr ||
                                payload_slot->magic != 0xEC2C2004u) {
                                ERROR("server peer recv: invalid slot");
                            }
                            size_t peer_idx = payload_slot->peer_idx;
                            uint16_t msg_type = payload_slot->message_type();
                            if (msg_type == rdma::EC2PC_MSG_PARITY_PAYLOAD) {
                                dispatch_peer_probe_parity_batch_task(
                                    *payload_slot, recv_wc_time_ns);
                            } else {
                                const rdma::EC2PCRpcMessage msg =
                                    *payload_slot->rpc_msg();
                                post_peer_payload_recv_slot(*payload_slot);
                                handle_peer_rpc_message(msg, peer_idx,
                                                        recv_wc_time_ns);
                            }
                        }
                    }
                }
                batch_count++;
                if (batch_count >= max_batches) {
                    break;
                }
            }

            if (!pending_peer_ack_queues.empty()) {
                (void)flush_pending_peer_acks(256, 0, 1);
            }

            size_t peer_pending_hint =
                pending_peer_send_total.load(std::memory_order_relaxed);
            if (peer_pending_hint > 0) {
                size_t peer_budget = 256;
                if (peer_pending_hint >= 32768) {
                    peer_budget = 4096;
                } else if (peer_pending_hint >= 8192) {
                    peer_budget = 2048;
                } else if (peer_pending_hint >= 1024) {
                    peer_budget = 1024;
                }
                auto flush_begin = std::chrono::steady_clock::now();
                (void)flush_pending_peer_sends(peer_budget, 0, 1);
                auto flush_end = std::chrono::steady_clock::now();
                ctr_peer_flush_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            flush_end - flush_begin)
                            .count()),
                    std::memory_order_relaxed);
            }

            return made_progress;
        };

        pin_current_thread(0, "poller", 0);

        auto stop_completion_pollers = [&]() {
            peer_poller_stop.store(true, std::memory_order_release);
            for (auto &th : peer_pollers) {
                if (th.joinable()) {
                    th.join();
                }
            }
            peer_pollers.clear();
        };

        auto poll_completion_cqs_once = [&]() {
            static constexpr size_t kCompletionPollerDataBudget = 8;
            static constexpr size_t kCompletionPollerPeerBudget = 8;
            static constexpr size_t kCompletionPollerPeerPayloadBudget = 64;
            uint64_t loop_begin_ns = steady_clock_now_ns();
            uint64_t prev_loop_ns =
                ctr_completion_poller_last_start_ns.exchange(
                    loop_begin_ns, std::memory_order_relaxed);
            if (prev_loop_ns != 0 && loop_begin_ns >= prev_loop_ns) {
                uint64_t gap_ns = loop_begin_ns - prev_loop_ns;
                ctr_completion_poller_gap_ns.fetch_add(
                    gap_ns, std::memory_order_relaxed);
                ctr_completion_poller_gap_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                update_peak(ctr_completion_poller_gap_max_ns, gap_ns);
            }
            bool made_progress = false;
            uint64_t data_begin_ns = steady_clock_now_ns();
            for (size_t cq_idx = 0; cq_idx < data_cqs.size(); cq_idx++) {
                made_progress =
                    drain_data_cq(cq_idx, kCompletionPollerDataBudget) ||
                    made_progress;
            }
            uint64_t data_ns = steady_clock_now_ns() - data_begin_ns;
            ctr_completion_poller_data_drain_ns.fetch_add(
                data_ns, std::memory_order_relaxed);
            ctr_completion_poller_data_drain_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_completion_poller_data_drain_max_ns, data_ns);
            if (data_ns >= 1000ull * 1000) {
                ctr_completion_poller_data_drain_gt1ms.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (data_ns >= 10ull * 1000 * 1000) {
                ctr_completion_poller_data_drain_gt10ms.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (data_ns >= 100ull * 1000 * 1000) {
                ctr_completion_poller_data_drain_gt100ms.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (data_ns >= 500ull * 1000 * 1000) {
                ctr_completion_poller_data_drain_gt500ms.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (data_ns >= 1000ull * 1000 * 1000) {
                ctr_completion_poller_data_drain_gt1000ms.fetch_add(
                    1, std::memory_order_relaxed);
            }
            uint64_t peer_ack_begin_ns = steady_clock_now_ns();
            made_progress =
                drain_peer_cq(kCompletionPollerPeerBudget) || made_progress;
            uint64_t peer_ack_ns = steady_clock_now_ns() - peer_ack_begin_ns;
            ctr_completion_poller_peer_ack_drain_ns.fetch_add(
                peer_ack_ns, std::memory_order_relaxed);
            ctr_completion_poller_peer_ack_drain_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_completion_poller_peer_ack_drain_max_ns,
                        peer_ack_ns);
            uint64_t payload_begin_ns = steady_clock_now_ns();
            made_progress =
                poll_peer_payload_cq_only_once(
                    kCompletionPollerPeerPayloadBudget) ||
                made_progress;
            uint64_t payload_ns = steady_clock_now_ns() - payload_begin_ns;
            ctr_completion_poller_payload_poll_ns.fetch_add(
                payload_ns, std::memory_order_relaxed);
            ctr_completion_poller_payload_poll_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_completion_poller_payload_poll_max_ns, payload_ns);
            uint64_t loop_ns = steady_clock_now_ns() - loop_begin_ns;
            ctr_completion_poller_loop_ns.fetch_add(
                loop_ns, std::memory_order_relaxed);
            ctr_completion_poller_loop_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_completion_poller_loop_max_ns, loop_ns);
            return made_progress;
        };

        peer_poller_stop.store(false, std::memory_order_release);
        peer_pollers.clear();
        if (dedicated_completion_poller_enabled()) {
            size_t poller_count = dedicated_completion_poller_count();
            peer_pollers.reserve(poller_count);
            for (size_t poller_idx = 0; poller_idx < poller_count;
                 poller_idx++) {
                peer_pollers.emplace_back([&, poller_idx]() {
                    pin_current_thread(
                        peer_payload_poller_thread_slot(poller_idx),
                        "completion-poller", poller_idx);
                    while (!peer_poller_stop.load(std::memory_order_acquire)) {
                        (void)poll_completion_cqs_once();
                    }
                });
            }
        }

        while (true) {
            auto loop_begin = std::chrono::steady_clock::now();
            drain_async_events_nonblocking();

            if (handle_main_loop_control_cq(post_control_recv,
                                            stop_completion_pollers)) {
                return;
            }
            run_main_loop_data_phase(drain_data_cq);
            run_main_loop_peer_phase(drain_peer_cq);
            flush_unified_pending_client_acks();
            maybe_print_periodic_ec2pc_server_summary();
            if (server_summary_stop_requested().load(
                    std::memory_order_acquire)) {
                print_ec2pc_server_summary();
                std::cout << "server stopped" << std::endl;
                stop_completion_pollers();
                release_rpc_resources();
                release_peer_resources();
                qps.clear();
                data_cqs.clear();
                data_qp_to_cq_idx.clear();
                data_cq_count = 0;
                std::exit(0);
            }
            finalize_main_poll_loop_iteration(loop_begin);
        }
    }


private:
    static constexpr size_t kRpcDepth = 1024;
    static constexpr size_t kBatchRpcDepth = 1024;
    static constexpr size_t kPeerRpcDepth = 4096;
    static constexpr size_t kPeerPendingQueueLimit = 32768;
    static constexpr size_t kRpcTaskQueueLimit = 8192;
    static constexpr int kPeerPortOffset = 10000;

    template <typename PostControlRecvFn, typename StopCompletionPollersFn>
    bool handle_main_loop_control_cq(
        PostControlRecvFn &post_control_recv,
        StopCompletionPollersFn &stop_completion_pollers) {
        static constexpr int kControlPollCap = 8;
        ibv_wc ctrl_wc[kControlPollCap];
        auto ctrl_poll_begin = std::chrono::steady_clock::now();
        int ctrl_poll_ret =
            ibv_poll_cq(control_cq.complete_queue, kControlPollCap, ctrl_wc);
        auto ctrl_poll_end = std::chrono::steady_clock::now();
        ctr_poll_calls.fetch_add(1, std::memory_order_relaxed);
        ctr_poll_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    ctrl_poll_end - ctrl_poll_begin)
                    .count()),
            std::memory_order_relaxed);
        ASSERT(ctrl_poll_ret >= 0 && ctrl_poll_ret <= kControlPollCap);
        if (ctrl_poll_ret == 0) {
            ctr_poll_empty.fetch_add(1, std::memory_order_relaxed);
        } else {
            ctr_poll_nonempty.fetch_add(1, std::memory_order_relaxed);
            ctr_poll_wc_total.fetch_add(static_cast<uint64_t>(ctrl_poll_ret),
                                        std::memory_order_relaxed);
            update_peak(ctr_poll_wc_max_batch,
                        static_cast<uint64_t>(ctrl_poll_ret));
        }
        for (int i = 0; i < ctrl_poll_ret; i++) {
            auto &work_completion = ctrl_wc[i];
            ASSERT(work_completion.status == IBV_WC_SUCCESS);
            if (work_completion.opcode != IBV_WC_RECV ||
                work_completion.wr_id != RQ_STOP) {
                continue;
            }
            uint32_t control_imm = CONTROL_IMM_STOP;
            if ((work_completion.wc_flags & IBV_WC_WITH_IMM) != 0) {
                control_imm = ntohl(work_completion.imm_data);
            }
            if (control_imm == CONTROL_IMM_RESET_EC2PC_STATS) {
                reset_ec2pc_measurement_counters();
                post_control_recv();
                continue;
            }
            print_ec2pc_server_summary();
            std::cout << "server stopped" << std::endl;
            stop_completion_pollers();
            release_rpc_resources();
            release_peer_resources();
            qps.clear();
            data_cqs.clear();
            data_qp_to_cq_idx.clear();
            data_cq_count = 0;
            return true;
        }
        return false;
    }

    template <typename DrainDataFn>
    void run_main_loop_data_phase(DrainDataFn &drain_data_cq) {
        static constexpr size_t kUnifiedDataPollBudget = 8;
        auto unified_data_phase_begin = std::chrono::steady_clock::now();
        // Poller-owned client/data CQ progress. This path should only drain
        // completions and enqueue worker tasks; it must not run heavy RPC work.
        if (!dedicated_completion_poller_enabled()) {
            for (size_t cq_idx = 0; cq_idx < data_cqs.size(); cq_idx++) {
                (void)drain_data_cq(cq_idx, kUnifiedDataPollBudget);
            }
        }
        auto unified_data_phase_end = std::chrono::steady_clock::now();
        uint64_t delta_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                unified_data_phase_end - unified_data_phase_begin)
                .count());
        ctr_unified_data_phase_ns.fetch_add(delta_ns,
                                            std::memory_order_relaxed);
        update_peak(ctr_unified_data_phase_max_ns, delta_ns);
    }

    template <typename DrainPeerFn>
    void run_main_loop_peer_phase(DrainPeerFn &drain_peer_cq) {
        static constexpr size_t kUnifiedPeerAckPollBudget = 32;
        static constexpr size_t kUnifiedPeerPayloadPollBudget = 64;
        static constexpr size_t kUnifiedPeerPayloadFlushBudget = 64;
        static constexpr size_t kUnifiedRuntimePairFlushBudget = 128;
        static constexpr size_t kUnifiedPeerDrainRounds = 8;
        auto unified_peer_phase_begin = std::chrono::steady_clock::now();
        const bool completion_poller_enabled =
            dedicated_completion_poller_enabled();
        for (size_t round = 0; round < kUnifiedPeerDrainRounds; round++) {
            bool made_progress = false;
            // Poller owns runtime parity transport publication. Workers only
            // publish by-value batch descriptors into the pending queue.
            uint64_t runtime_pair_begin_ns = steady_clock_now_ns();
            size_t runtime_pair_jobs = flush_runtime_parity_batch_pair_pending(
                kUnifiedRuntimePairFlushBudget);
            uint64_t runtime_pair_end_ns = steady_clock_now_ns();
            note_main_loop_stage(
                ctr_main_runtime_pair_flush_calls,
                ctr_main_runtime_pair_flush_ns,
                ctr_main_runtime_pair_flush_max_ns,
                ctr_main_runtime_pair_flush_gap_ns,
                ctr_main_runtime_pair_flush_gap_cnt,
                ctr_main_runtime_pair_flush_gap_max_ns,
                ctr_main_runtime_pair_flush_last_start_ns,
                ctr_main_runtime_pair_flush_jobs,
                ctr_main_runtime_pair_flush_active_calls,
                runtime_pair_begin_ns, runtime_pair_end_ns,
                static_cast<uint64_t>(runtime_pair_jobs));
            made_progress = runtime_pair_jobs > 0 || made_progress;
            if (completion_poller_enabled) {
                uint64_t transport_begin_ns = steady_clock_now_ns();
                bool transport_progress = flush_peer_payload_transport_once(
                    kUnifiedPeerPayloadFlushBudget);
                uint64_t transport_end_ns = steady_clock_now_ns();
                note_main_loop_stage(
                    ctr_main_peer_payload_transport_calls,
                    ctr_main_peer_payload_transport_ns,
                    ctr_main_peer_payload_transport_max_ns,
                    ctr_main_peer_payload_transport_gap_ns,
                    ctr_main_peer_payload_transport_gap_cnt,
                    ctr_main_peer_payload_transport_gap_max_ns,
                    ctr_main_peer_payload_transport_last_start_ns,
                    ctr_main_peer_payload_transport_progress_calls,
                    ctr_main_peer_payload_transport_active_calls,
                    transport_begin_ns, transport_end_ns,
                    transport_progress ? 1u : 0u);
                made_progress = transport_progress || made_progress;
            } else {
                made_progress =
                    drain_peer_cq(kUnifiedPeerAckPollBudget) || made_progress;
                made_progress =
                    drain_peer_payload_cq_once(kUnifiedPeerPayloadPollBudget) ||
                    made_progress;
            }
            if (!made_progress) {
                break;
            }
        }
        auto unified_peer_phase_end = std::chrono::steady_clock::now();
        uint64_t delta_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                unified_peer_phase_end - unified_peer_phase_begin)
                .count());
        ctr_unified_peer_phase_ns.fetch_add(delta_ns,
                                            std::memory_order_relaxed);
        update_peak(ctr_unified_peer_phase_max_ns, delta_ns);
    }

    void flush_unified_pending_client_acks() {
        uint64_t flush_begin_ns = steady_clock_now_ns();
        uint64_t posted_total = 0;
        if (data_cq_count == 0) {
            note_main_loop_stage(
                ctr_main_client_ack_flush_calls, ctr_main_client_ack_flush_ns,
                ctr_main_client_ack_flush_max_ns,
                ctr_main_client_ack_flush_gap_ns,
                ctr_main_client_ack_flush_gap_cnt,
                ctr_main_client_ack_flush_gap_max_ns,
                ctr_main_client_ack_flush_last_start_ns,
                ctr_main_client_ack_flush_posts,
                ctr_main_client_ack_flush_active_calls, flush_begin_ns,
                steady_clock_now_ns(), posted_total);
            return;
        }
        // Poller-owned final ACK posting. Workers only enqueue ACK messages.
        size_t pending_hint = pending_send_total.load(std::memory_order_relaxed);
        if (pending_hint == 0) {
            note_main_loop_stage(
                ctr_main_client_ack_flush_calls, ctr_main_client_ack_flush_ns,
                ctr_main_client_ack_flush_max_ns,
                ctr_main_client_ack_flush_gap_ns,
                ctr_main_client_ack_flush_gap_cnt,
                ctr_main_client_ack_flush_gap_max_ns,
                ctr_main_client_ack_flush_last_start_ns,
                ctr_main_client_ack_flush_posts,
                ctr_main_client_ack_flush_active_calls, flush_begin_ns,
                steady_clock_now_ns(), posted_total);
            return;
        }
        size_t flush_budget = 256;
        if (pending_hint >= 32768) {
            flush_budget = 4096;
        } else if (pending_hint >= 8192) {
            flush_budget = 2048;
        } else if (pending_hint >= 1024) {
            flush_budget = 1024;
        }
        auto flush_begin = std::chrono::steady_clock::now();
        for (size_t cq_idx = 0; cq_idx < data_cq_count; cq_idx++) {
            posted_total += flush_pending_sends(flush_budget, cq_idx,
                                                std::max<size_t>(1, data_cq_count));
        }
        auto flush_end = std::chrono::steady_clock::now();
        ctr_flush_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    flush_end - flush_begin)
                    .count()),
            std::memory_order_relaxed);
        note_main_loop_stage(
            ctr_main_client_ack_flush_calls, ctr_main_client_ack_flush_ns,
            ctr_main_client_ack_flush_max_ns, ctr_main_client_ack_flush_gap_ns,
            ctr_main_client_ack_flush_gap_cnt,
            ctr_main_client_ack_flush_gap_max_ns,
            ctr_main_client_ack_flush_last_start_ns,
            ctr_main_client_ack_flush_posts,
            ctr_main_client_ack_flush_active_calls, flush_begin_ns,
            steady_clock_now_ns(), posted_total);
    }

    void finalize_main_poll_loop_iteration(
        std::chrono::steady_clock::time_point loop_begin) {
        auto loop_end = std::chrono::steady_clock::now();
        uint64_t loop_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end -
                                                                  loop_begin)
                .count());
        ctr_poll_loop_count.fetch_add(1, std::memory_order_relaxed);
        ctr_poll_loop_ns.fetch_add(loop_ns, std::memory_order_relaxed);
        update_peak(ctr_poll_loop_max_ns, loop_ns);
    }

    size_t address_lock_shard(uint64_t offset) const {
        // Offsets are typically 8KB-aligned; avoid low-bit masking which
        // collapses all traffic to one shard.
        uint64_t key = offset >> kAddressLockGranularityShift;
        key ^= (key >> 17);
        key ^= (key >> 33);
        return static_cast<size_t>(key & (kAddressLockShardCount - 1));
    }

    static void update_peak(std::atomic<uint64_t> &peak, uint64_t candidate) {
        uint64_t cur = peak.load(std::memory_order_relaxed);
        while (cur < candidate &&
               !peak.compare_exchange_weak(cur, candidate,
                                           std::memory_order_relaxed)) {
        }
    }

    static void update_min(std::atomic<uint64_t> &min_val, uint64_t candidate) {
        uint64_t cur = min_val.load(std::memory_order_relaxed);
        while (candidate < cur &&
               !min_val.compare_exchange_weak(cur, candidate,
                                              std::memory_order_relaxed)) {
        }
    }

    void reset_ec2pc_base_counters() {
#define FARLIB_SERVER_RESET_EC2PC_BASE_COUNTERS(X)                           \
        X(ctr_client_batch_task_dequeued, 0)                                 \
        X(ctr_client_batch_worker_service_ns, 0)                             \
        X(ctr_client_batch_worker_service_cnt, 0)                            \
        X(ctr_client_batch_recv_to_enqueue_ns, 0)                            \
        X(ctr_client_batch_recv_to_enqueue_cnt, 0)                           \
        X(ctr_client_batch_recv_to_dequeue_ns, 0)                            \
        X(ctr_client_batch_recv_to_dequeue_cnt, 0)                           \
        X(ctr_batch_recv_to_repost_ns, 0)                                    \
        X(ctr_batch_recv_to_repost_cnt, 0)                                   \
        X(ctr_batch_recv_to_repost_max_ns, 0)                                \
        X(ctr_batch_recv_poll_calls, 0)                                      \
        X(ctr_batch_recv_poll_wc_total, 0)                                   \
        X(ctr_batch_recv_poll_wc_max_batch, 0)                               \
        X(ctr_client_rpc_task_dequeued, 0)                                   \
        X(ctr_client_rpc_data_req_task_dequeued, 0)                          \
        X(ctr_client_rpc_probe_req_task_dequeued, 0)                         \
        X(ctr_client_rpc_worker_service_ns, 0)                               \
        X(ctr_client_rpc_worker_service_cnt, 0)                              \
        X(ctr_peer_probe_batch_task_dequeued, 0)                             \
        X(ctr_probe_parity_batch_worker_service_ns, 0)                       \
        X(ctr_probe_parity_batch_worker_service_cnt, 0)                      \
        X(ctr_enq_ack, 0)                                                    \
        X(ctr_post_ok, 0)                                                    \
        X(ctr_send_wc, 0)                                                    \
        X(ctr_batch_recv_dispatch, 0)                                        \
        X(ctr_batch_recv_release, 0)                                         \
        X(ctr_batch_recv_inflight, 0)                                        \
        X(ctr_batch_recv_inflight_max, 0)                                    \
        X(ctr_peer_enq_ack, 0)                                               \
        X(ctr_peer_send_wc, 0)                                               \
        X(ctr_worker_task_dequeued, 0)                                       \
        X(ctr_worker_queue_wait_ns, 0)                                       \
        X(ctr_worker_service_ns, 0)                                          \
        X(ctr_worker_dequeue_lock_wait_ns, 0)                                \
        X(ctr_worker_dequeue_lock_wait_cnt, 0)                               \
        X(ctr_worker_intertask_gap_ns, 0)                                    \
        X(ctr_worker_intertask_gap_cnt, 0)                                   \
        X(ctr_worker_post_service_queue_empty, 0)                            \
        X(ctr_worker_post_service_queue_nonempty, 0)                         \
        X(ctr_worker_post_service_overflow_nonempty, 0)                      \
        X(ctr_worker_post_service_queue_depth_sum, 0)                        \
        X(ctr_worker_post_service_probe_ns, 0)                               \
        X(ctr_worker_post_service_probe_cnt, 0)                              \
        X(ctr_worker_post_service_probe_lock_wait_ns, 0)                     \
        X(ctr_worker_post_service_probe_lock_wait_cnt, 0)                    \
        X(ctr_batch_rpc_publish_ns, 0)                                       \
        X(ctr_batch_rpc_publish_cnt, 0)                                      \
        X(ctr_dispatch_batch_lock_wait_ns, 0)                                \
        X(ctr_dispatch_batch_lock_wait_cnt, 0)                               \
        X(ctr_batch_rpc_track_ns, 0)                                         \
        X(ctr_batch_rpc_track_cnt, 0)                                        \
        X(ctr_data_req_msg_init_ns, 0)                                       \
        X(ctr_data_req_meta_fill_ns, 0)                                      \
        X(ctr_data_req_get_ns, 0)                                            \
        X(ctr_data_req_delta_ns, 0)                                          \
        X(ctr_data_req_delta_prepare_ns, 0)                                  \
        X(ctr_data_req_delta_encode_ns, 0)                                   \
        X(ctr_data_req_fanout_ready_ns, 0)                                   \
        X(ctr_addr_lock_wait_ns_data_req, 0)                                 \
        X(ctr_addr_lock_hold_ns_data_req, 0)                                 \
        X(ctr_addr_lock_wait_ns_parity_apply, 0)                             \
        X(ctr_addr_lock_hold_ns_parity_apply, 0)                             \
        X(ctr_addr_lock_wait_max_ns, 0)                                      \
        X(ctr_addr_lock_hold_max_ns, 0)                                      \
        X(ctr_data_req_copy_ns, 0)                                           \
        X(ctr_data_req_peer_enqueue_ns, 0)                                   \
        X(ctr_compact_data_req, 0)                                           \
        X(ctr_compact_data_req_msg_init_ns, 0)                               \
        X(ctr_compact_data_req_meta_fill_ns, 0)                              \
        X(ctr_compact_data_req_get_ns, 0)                                    \
        X(ctr_compact_data_req_delta_ns, 0)                                  \
        X(ctr_compact_data_req_delta_prepare_ns, 0)                          \
        X(ctr_compact_data_req_delta_encode_ns, 0)                           \
        X(ctr_compact_data_req_copy_ns, 0)                                   \
        X(ctr_compact_data_req_peer_enqueue_ns, 0)                           \
        X(ctr_compact_data_req_publish_ns, 0)                                \
        X(ctr_compact_data_req_fanout_ready_ns, 0)                           \
        X(ctr_compact_peer_payload_msgs, 0)                                  \
        X(ctr_compact_peer_payload_span_ops, 0)                              \
        X(ctr_compact_peer_payload_send_bytes, 0)                            \
        X(ctr_compact_peer_payload_recv_bytes, 0)                            \
        X(ctr_compact_peer_post_ns, 0)                                       \
        X(ctr_compact_peer_post_cnt, 0)                                      \
        X(ctr_compact_peer_send_wc, 0)                                       \
        X(ctr_compact_peer_send_cqe_reclaim_ns, 0)                           \
        X(ctr_compact_peer_send_cqe_reclaim_cnt, 0)                          \
        X(ctr_compact_peer_recv_wc, 0)                                       \
        X(ctr_compact_peer_recv_to_enqueue_ns, 0)                            \
        X(ctr_compact_peer_recv_to_enqueue_cnt, 0)                           \
        X(ctr_compact_peer_enqueue_to_dequeue_ns, 0)                         \
        X(ctr_compact_peer_enqueue_to_dequeue_cnt, 0)                        \
        X(ctr_compact_peer_recv_to_handle_ns, 0)                             \
        X(ctr_compact_peer_recv_to_handle_cnt, 0)                            \
        X(ctr_compact_peer_recv_to_ack_post_ns, 0)                           \
        X(ctr_compact_peer_recv_to_ack_post_cnt, 0)                          \
        X(ctr_compact_peer_payload_dispatch_ns, 0)                           \
        X(ctr_compact_peer_payload_dispatch_cnt, 0)                          \
        X(ctr_compact_peer_payload_repost_ns, 0)                             \
        X(ctr_compact_peer_payload_repost_cnt, 0)                            \
        X(ctr_compact_peer_worker_service_ns, 0)                             \
        X(ctr_compact_peer_worker_service_cnt, 0)                            \
        X(ctr_compact_parity_apply_ns, 0)                                    \
        X(ctr_compact_parity_get_ns, 0)                                      \
        X(ctr_compact_parity_xor_ns, 0)                                      \
        X(ctr_compact_parity_ack_post_ns, 0)                                 \
        X(ctr_runtime_parity_batch_send_to_handle_ns, 0)                     \
        X(ctr_runtime_parity_batch_send_to_handle_cnt, 0)                    \
        X(ctr_runtime_parity_batch_send_to_recv_wc_ns, 0)                    \
        X(ctr_runtime_parity_batch_send_to_recv_wc_cnt, 0)                   \
        X(ctr_runtime_parity_batch_recv_to_enqueue_ns, 0)                    \
        X(ctr_runtime_parity_batch_recv_to_enqueue_cnt, 0)                   \
        X(ctr_runtime_parity_batch_recv_to_enqueue_max_ns, 0)                \
        X(ctr_runtime_parity_batch_enqueue_to_dequeue_ns, 0)                 \
        X(ctr_runtime_parity_batch_enqueue_to_dequeue_cnt, 0)                \
        X(ctr_runtime_parity_batch_enqueue_to_dequeue_max_ns, 0)             \
        X(ctr_runtime_parity_batch_worker_service_ns, 0)                     \
        X(ctr_runtime_parity_batch_worker_service_cnt, 0)                    \
        X(ctr_runtime_parity_batch_worker_service_max_ns, 0)                 \
        X(ctr_runtime_parity_batch_send_to_ack_post_ns, 0)                   \
        X(ctr_runtime_parity_batch_send_to_ack_post_cnt, 0)                  \
        X(ctr_runtime_parity_batch_recv_to_ack_post_ns, 0)                   \
        X(ctr_runtime_parity_batch_recv_to_ack_post_cnt, 0)                  \
        X(ctr_runtime_parity_batch_post_to_send_cqe_ns, 0)                   \
        X(ctr_runtime_parity_batch_post_to_send_cqe_cnt, 0)                  \
        X(ctr_runtime_parity_batch_post_to_send_cqe_max_ns, 0)               \
        X(ctr_runtime_parity_batch_post_fail, 0)                             \
        X(ctr_runtime_parity_batch_reserve_fail, 0)                          \
        X(ctr_runtime_parity_reserve_fail_invalid, 0)                        \
        X(ctr_runtime_parity_reserve_fail_no_shard, 0)                       \
        X(ctr_runtime_parity_reserve_fail_bad_qidx, 0)                       \
        X(ctr_runtime_parity_reserve_fail_no_queue, 0)                       \
        X(ctr_runtime_parity_reserve_fail_no_mutex, 0)                       \
        X(ctr_runtime_parity_reserve_fail_outstanding, 0)                    \
        X(ctr_runtime_parity_reserve_fail_no_free_slot, 0)                   \
        X(ctr_runtime_parity_reserve_fail_outstanding_sum, 0)                \
        X(ctr_runtime_parity_reserve_fail_outstanding_cnt, 0)                \
        X(ctr_runtime_parity_reserve_fail_outstanding_max, 0)                \
        X(ctr_runtime_parity_batch_ready_gate_fail, 0)                       \
        X(ctr_runtime_parity_batch_ready_gate_slots, 0)                      \
        X(ctr_runtime_parity_ready_to_post_ns, 0)                            \
        X(ctr_runtime_parity_ready_to_post_cnt, 0)                           \
        X(ctr_runtime_parity_ready_to_post_max_ns, 0)                        \
        X(ctr_runtime_parity_ready_to_gate_fail_ns, 0)                       \
        X(ctr_runtime_parity_ready_to_gate_fail_cnt, 0)                      \
        X(ctr_runtime_parity_ready_to_gate_fail_max_ns, 0)                   \
        X(ctr_runtime_parity_ready_gate_outstanding_sum, 0)                  \
        X(ctr_runtime_parity_ready_gate_outstanding_cnt, 0)                  \
        X(ctr_runtime_parity_ready_gate_outstanding_max, 0)                  \
        X(ctr_runtime_parity_send_cqe_reclaim_ns, 0)                         \
        X(ctr_runtime_parity_send_cqe_reclaim_cnt, 0)                        \
        X(ctr_runtime_parity_send_cqe_reclaim_max_ns, 0)                     \
        X(ctr_peer_ack_recv_dispatch_ns, 0)                                  \
        X(ctr_peer_ack_recv_dispatch_cnt, 0)                                 \
        X(ctr_peer_ack_consume_ns, 0)                                        \
        X(ctr_peer_ack_consume_cnt, 0)                                       \
        X(ctr_client_final_ack_posted, 0)                                    \
        X(ctr_client_final_ack_enqueued, 0)                                  \
        X(ctr_client_final_ack_done_to_enqueue_ns, 0)                        \
        X(ctr_client_final_ack_done_to_enqueue_cnt, 0)                       \
        X(ctr_client_final_ack_done_to_enqueue_max_ns, 0)                    \
        X(ctr_client_final_ack_enqueue_to_post_ns, 0)                        \
        X(ctr_client_final_ack_enqueue_to_post_cnt, 0)                       \
        X(ctr_client_final_ack_enqueue_to_post_max_ns, 0)                    \
        X(ctr_client_final_ack_done_to_post_ns, 0)                           \
        X(ctr_client_final_ack_done_to_post_cnt, 0)                          \
        X(ctr_client_final_ack_done_to_post_max_ns, 0)                       \
        X(ctr_client_final_ack_post_to_send_cqe_ns, 0)                       \
        X(ctr_client_final_ack_post_to_send_cqe_cnt, 0)                      \
        X(ctr_client_final_ack_post_to_send_cqe_max_ns, 0)                   \
        X(ctr_client_batch_send_to_recv_wc_ns, 0)                            \
        X(ctr_client_batch_send_to_recv_wc_cnt, 0)                           \
        X(ctr_server_ec2pc_complete_ns, 0)                                   \
        X(ctr_server_ec2pc_complete_cnt, 0)                                  \
        X(ctr_server_ec2pc_fanout_ns, 0)                                     \
        X(ctr_server_ec2pc_fanout_cnt, 0)                                    \
        X(ctr_server_ec2pc_first_ack_after_fanout_ns, 0)                     \
        X(ctr_server_ec2pc_first_ack_after_fanout_cnt, 0)                    \
        X(ctr_server_ec2pc_parity0_ack_after_fanout_ns, 0)                   \
        X(ctr_server_ec2pc_parity0_ack_after_fanout_cnt, 0)                  \
        X(ctr_server_ec2pc_parity1_ack_after_fanout_ns, 0)                   \
        X(ctr_server_ec2pc_parity1_ack_after_fanout_cnt, 0)                  \
        X(ctr_server_ec2pc_final_ack_after_fanout_ns, 0)                     \
        X(ctr_server_ec2pc_final_ack_after_fanout_cnt, 0)                    \
        X(ctr_server_ec2pc_fanout_to_parity_post_ns, 0)                      \
        X(ctr_server_ec2pc_fanout_to_parity_post_cnt, 0)                     \
        X(ctr_server_ec2pc_parity_post_to_ack_ns, 0)                         \
        X(ctr_server_ec2pc_parity_post_to_ack_cnt, 0)                        \
        X(ctr_server_ec2pc_first_to_final_ack_ns, 0)                         \
        X(ctr_server_ec2pc_first_to_final_ack_cnt, 0)                        \
        X(ctr_server_ec2pc_track_lock_wait_ns, 0)                            \
        X(ctr_server_ec2pc_track_lock_hold_ns, 0)                            \
        X(ctr_server_ec2pc_track_lock_ops, 0)                                \
        X(ctr_peer_lane_busy_ns, 0)                                          \
        X(ctr_peer_lane_busy_retries, 0)                                     \
        X(ctr_peer_lane_post_ok_ns, 0)                                       \
        X(ctr_peer_lane_post_ok, 0)                                          \
        X(ctr_peer_lane_post_fail, 0)                                        \
        X(ctr_peer_lane_enqueue_full, 0)                                     \
        X(ctr_peer_send_lock_direct_wait_ns, 0)                              \
        X(ctr_peer_send_lock_direct_hold_ns, 0)                              \
        X(ctr_peer_send_lock_direct_ops, 0)                                  \
        X(ctr_peer_send_lock_batch_wait_ns, 0)                               \
        X(ctr_peer_send_lock_batch_hold_ns, 0)                               \
        X(ctr_peer_send_lock_batch_ops, 0)                                   \
        X(ctr_peer_send_lock_reclaim_wait_ns, 0)                             \
        X(ctr_peer_send_lock_reclaim_hold_ns, 0)                             \
        X(ctr_peer_send_lock_reclaim_ops, 0)                                 \
        X(ctr_main_runtime_pair_flush_calls, 0)                              \
        X(ctr_main_runtime_pair_flush_ns, 0)                                 \
        X(ctr_main_runtime_pair_flush_max_ns, 0)                             \
        X(ctr_main_runtime_pair_flush_jobs, 0)                               \
        X(ctr_main_runtime_pair_flush_active_calls, 0)                       \
        X(ctr_main_peer_payload_transport_calls, 0)                          \
        X(ctr_main_peer_payload_transport_ns, 0)                             \
        X(ctr_main_peer_payload_transport_max_ns, 0)                         \
        X(ctr_main_peer_payload_transport_progress_calls, 0)                 \
        X(ctr_main_peer_payload_transport_active_calls, 0)                   \
        X(ctr_main_client_ack_flush_calls, 0)                                \
        X(ctr_main_client_ack_flush_ns, 0)                                   \
        X(ctr_main_client_ack_flush_max_ns, 0)                               \
        X(ctr_main_client_ack_flush_posts, 0)                                \
        X(ctr_main_client_ack_flush_active_calls, 0)
#define FARLIB_SERVER_RESET_ONE(field, value) relaxed_store(field, value);
        FARLIB_SERVER_RESET_EC2PC_BASE_COUNTERS(FARLIB_SERVER_RESET_ONE)
#undef FARLIB_SERVER_RESET_ONE
#undef FARLIB_SERVER_RESET_EC2PC_BASE_COUNTERS
    }

    void print_ec2pc_server_summary() {
        if (!ec2pc_server_summary_enabled()) {
            return;
        }
        auto load = [](const auto &counter) -> uint64_t {
            return counter.load(std::memory_order_relaxed);
        };
        const char *port_env = std::getenv("SERVER_PORT");
        std::ostream &out = std::cout;
        out << "SERVER_EC2PC_SUMMARY_BEGIN server=" << local_server_index
            << " port=" << (port_env != nullptr ? port_env : "<unset>")
            << " rpc_workers=" << rpc_worker_count
            << " peer_payload_qps=" << peer_payload_qp_count << std::endl;
        auto metric = [&](const char *name, uint64_t value) {
            out << "SERVER_EC2PC_SUMMARY server=" << local_server_index
                << " metric=" << name << " value=" << value << std::endl;
        };
        auto avg_metric = [&](const char *name, uint64_t total,
                              uint64_t count) {
            metric(name, total);
            metric((std::string(name) + "_cnt").c_str(), count);
            if (count != 0) {
                metric((std::string(name) + "_avg_ns").c_str(), total / count);
            }
        };
        auto saturating_sub = [](uint64_t lhs, uint64_t rhs) {
            return lhs > rhs ? lhs - rhs : 0;
        };
        uint64_t profile_failures = 0;
        uint64_t wall_ns = 0;
        uint64_t tsc_delta = 0;
        uint64_t main_thread_ns = 0;
        uint64_t worker_thread_ns = 0;
        uint64_t completion_poller_thread_ns = 0;
        uint64_t process_user_ns = 0;
        uint64_t process_sys_ns = 0;
        uint64_t process_total_ns = 0;
        bool profile_valid = remote_cpu_reset_wall_ns != 0;
        if (profile_valid) {
            const uint64_t now_wall_ns = steady_clock_now_ns();
            wall_ns = now_wall_ns >= remote_cpu_reset_wall_ns
                          ? now_wall_ns - remote_cpu_reset_wall_ns
                          : 0;
            const uint64_t now_tsc = get_cycles();
            tsc_delta = now_tsc >= remote_cpu_reset_tsc
                            ? now_tsc - remote_cpu_reset_tsc
                            : 0;
            CpuUsageSample process_now = sample_process_cpu();
            if (process_now.valid && remote_cpu_reset_process.valid &&
                process_now.user_ns >= remote_cpu_reset_process.user_ns &&
                process_now.sys_ns >= remote_cpu_reset_process.sys_ns) {
                process_user_ns =
                    process_now.user_ns - remote_cpu_reset_process.user_ns;
                process_sys_ns =
                    process_now.sys_ns - remote_cpu_reset_process.sys_ns;
                process_total_ns = process_user_ns + process_sys_ns;
            } else {
                profile_valid = false;
                profile_failures++;
            }
            uint64_t main_now_ns = 0;
            if (main_thread_clock_valid &&
                sample_thread_cpu(main_thread_clock_id, &main_now_ns) &&
                main_now_ns >= remote_cpu_reset_main_thread_ns) {
                main_thread_ns =
                    main_now_ns - remote_cpu_reset_main_thread_ns;
            } else {
                profile_failures++;
            }
            worker_thread_ns = sample_thread_group_cpu_ns(
                remote_cpu_rpc_worker_clocks, &profile_failures);
            completion_poller_thread_ns = sample_thread_group_cpu_ns(
                remote_cpu_completion_poller_clocks, &profile_failures);
        }
        const uint64_t profiled_thread_ns =
            main_thread_ns + worker_thread_ns + completion_poller_thread_ns;
        const uint64_t unattributed_ns =
            process_total_ns > profiled_thread_ns
                ? process_total_ns - profiled_thread_ns
                : 0;
        const uint64_t network_thread_ns =
            main_thread_ns + completion_poller_thread_ns;

        metric("remote_cpu_profile_valid", profile_valid ? 1 : 0);
        metric("remote_cpu_sample_failures", profile_failures);
        metric("remote_cpu_wall_ns", wall_ns);
        metric("remote_cpu_process_user_ns", process_user_ns);
        metric("remote_cpu_process_sys_ns", process_sys_ns);
        metric("remote_cpu_process_total_ns", process_total_ns);
        metric("remote_cpu_process_util_pct_x100",
               pct_x100(process_total_ns, wall_ns));
        metric("remote_cpu_main_thread_ns", main_thread_ns);
        metric("remote_cpu_main_thread_util_pct_x100",
               pct_x100(main_thread_ns, wall_ns));
        metric("remote_cpu_rpc_worker_threads",
               remote_cpu_rpc_worker_clocks.size());
        metric("remote_cpu_rpc_worker_thread_ns", worker_thread_ns);
        metric("remote_cpu_rpc_worker_util_pct_x100",
               pct_x100(worker_thread_ns, wall_ns));
        metric("remote_cpu_completion_poller_threads",
               remote_cpu_completion_poller_clocks.size());
        metric("remote_cpu_completion_poller_thread_ns",
               completion_poller_thread_ns);
        metric("remote_cpu_completion_poller_util_pct_x100",
               pct_x100(completion_poller_thread_ns, wall_ns));
        metric("remote_cpu_profiled_thread_ns", profiled_thread_ns);
        metric("remote_cpu_profiled_thread_util_pct_x100",
               pct_x100(profiled_thread_ns, wall_ns));
        metric("remote_cpu_unattributed_ns", unattributed_ns);
        metric("remote_cpu_unattributed_util_pct_x100",
               pct_x100(unattributed_ns, wall_ns));
        metric("remote_cpu_network_thread_ns", network_thread_ns);
        metric("remote_cpu_network_thread_share_pct_x100",
               pct_x100(network_thread_ns, profiled_thread_ns));

        const uint64_t ecspan_peer_send_ns =
            load(ctr_main_peer_payload_transport_ns) +
            load(ctr_peer_lane_post_ok_ns) +
            load(ctr_runtime_parity_ready_to_post_ns) +
            load(ctr_runtime_parity_send_cqe_reclaim_ns);
        const uint64_t ecspan_peer_recv_ns =
            load(ctr_completion_poller_payload_poll_ns);
        const uint64_t ecspan_peer_memcpy_ns =
            load(ctr_runtime_parity_item_copy_ns) +
            load(ctr_runtime_parity_slot_memcpy_ns);
        const uint64_t ecspan_parity_apply_ns =
            load(ctr_probe_parity_batch_xor_ns);
        const uint64_t ecspan_peer_cq_poll_ns =
            load(ctr_completion_poller_data_drain_ns) +
            load(ctr_completion_poller_peer_ack_drain_ns) +
            load(ctr_completion_poller_payload_poll_ns);
        const uint64_t compact_encode_ns =
            load(ctr_compact_data_req_delta_encode_ns);
        const uint64_t compact_encode_prepare_ns =
            load(ctr_compact_data_req_delta_prepare_ns);
        const uint64_t compact_memcpy_ns =
            load(ctr_compact_data_req_copy_ns);
        const uint64_t compact_parity_xor_ns =
            load(ctr_compact_parity_xor_ns);
        const uint64_t compact_parity_apply_ns =
            load(ctr_compact_parity_apply_ns);
        const uint64_t compact_peer_send_ns =
            load(ctr_compact_peer_post_ns) +
            load(ctr_compact_peer_send_cqe_reclaim_ns);
        const uint64_t compact_peer_recv_ns =
            load(ctr_compact_peer_recv_to_enqueue_ns) +
            load(ctr_compact_peer_payload_dispatch_ns);
        const uint64_t compact_peer_ack_ns =
            load(ctr_compact_parity_ack_post_ns) +
            load(ctr_compact_peer_payload_repost_ns);
        const uint64_t compact_profiled_cpu_ns =
            compact_encode_ns + compact_memcpy_ns + compact_parity_xor_ns +
            compact_peer_send_ns + compact_peer_recv_ns + compact_peer_ack_ns;
        const uint64_t main_poll_overhead_ns =
            saturating_sub(load(ctr_poll_loop_ns),
                           load(ctr_unified_data_phase_ns) +
                               load(ctr_unified_peer_phase_ns) +
                               load(ctr_main_client_ack_flush_ns));
        const uint64_t completion_poller_spin_ns =
            saturating_sub(load(ctr_completion_poller_loop_ns),
                           load(ctr_completion_poller_data_drain_ns) +
                               load(ctr_completion_poller_peer_ack_drain_ns) +
                               load(ctr_completion_poller_payload_poll_ns));
        metric("ecspan_encode_cycles",
               ns_to_cycles(load(ctr_data_req_delta_encode_ns), wall_ns,
                            tsc_delta));
        metric("ecspan_peer_send_cycles",
               ns_to_cycles(ecspan_peer_send_ns, wall_ns, tsc_delta));
        metric("ecspan_peer_recv_cycles",
               ns_to_cycles(ecspan_peer_recv_ns, wall_ns, tsc_delta));
        metric("ecspan_peer_memcpy_cycles",
               ns_to_cycles(ecspan_peer_memcpy_ns, wall_ns, tsc_delta));
        metric("ecspan_parity_apply_cycles",
               ns_to_cycles(ecspan_parity_apply_ns, wall_ns, tsc_delta));
        metric("ecspan_peer_cq_poll_cycles",
               ns_to_cycles(ecspan_peer_cq_poll_ns, wall_ns, tsc_delta));
        metric("compact_data_req", load(ctr_compact_data_req));
        metric("compact_peer_payload_msgs",
               load(ctr_compact_peer_payload_msgs));
        metric("compact_peer_payload_span_ops",
               load(ctr_compact_peer_payload_span_ops));
        metric("compact_peer_payload_send_bytes",
               load(ctr_compact_peer_payload_send_bytes));
        metric("compact_peer_payload_recv_bytes",
               load(ctr_compact_peer_payload_recv_bytes));
        metric("compact_encode_ns", compact_encode_ns);
        metric("compact_encode_cycles",
               ns_to_cycles(compact_encode_ns, wall_ns, tsc_delta));
        metric("compact_encode_prepare_ns", compact_encode_prepare_ns);
        metric("compact_encode_prepare_cycles",
               ns_to_cycles(compact_encode_prepare_ns, wall_ns, tsc_delta));
        metric("compact_memcpy_ns", compact_memcpy_ns);
        metric("compact_memcpy_cycles",
               ns_to_cycles(compact_memcpy_ns, wall_ns, tsc_delta));
        metric("compact_parity_xor_ns", compact_parity_xor_ns);
        metric("compact_parity_xor_cycles",
               ns_to_cycles(compact_parity_xor_ns, wall_ns, tsc_delta));
        metric("compact_parity_apply_ns", compact_parity_apply_ns);
        metric("compact_parity_apply_cycles",
               ns_to_cycles(compact_parity_apply_ns, wall_ns, tsc_delta));
        metric("compact_peer_send_ns", compact_peer_send_ns);
        metric("compact_peer_send_cycles",
               ns_to_cycles(compact_peer_send_ns, wall_ns, tsc_delta));
        metric("compact_peer_recv_ns", compact_peer_recv_ns);
        metric("compact_peer_recv_cycles",
               ns_to_cycles(compact_peer_recv_ns, wall_ns, tsc_delta));
        metric("compact_peer_ack_ns", compact_peer_ack_ns);
        metric("compact_peer_ack_cycles",
               ns_to_cycles(compact_peer_ack_ns, wall_ns, tsc_delta));
        metric("compact_profiled_cpu_ns", compact_profiled_cpu_ns);
        metric("compact_profiled_cycles",
               ns_to_cycles(compact_profiled_cpu_ns, wall_ns, tsc_delta));
        metric("compact_peer_post_cnt", load(ctr_compact_peer_post_cnt));
        metric("compact_peer_send_wc", load(ctr_compact_peer_send_wc));
        metric("compact_peer_recv_wc", load(ctr_compact_peer_recv_wc));
        avg_metric("compact_data_req_msg_init_ns",
                   load(ctr_compact_data_req_msg_init_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_meta_fill_ns",
                   load(ctr_compact_data_req_meta_fill_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_get_ns",
                   load(ctr_compact_data_req_get_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_delta_ns",
                   load(ctr_compact_data_req_delta_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_delta_encode_ns",
                   load(ctr_compact_data_req_delta_encode_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_copy_ns",
                   load(ctr_compact_data_req_copy_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_peer_enqueue_ns",
                   load(ctr_compact_data_req_peer_enqueue_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_publish_ns",
                   load(ctr_compact_data_req_publish_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_data_req_fanout_ready_ns",
                   load(ctr_compact_data_req_fanout_ready_ns),
                   load(ctr_compact_data_req));
        avg_metric("compact_parity_get_ns",
                   load(ctr_compact_parity_get_ns),
                   load(ctr_compact_peer_payload_span_ops));
        avg_metric("compact_parity_xor_ns",
                   load(ctr_compact_parity_xor_ns),
                   load(ctr_compact_peer_payload_span_ops));
        avg_metric("compact_parity_ack_post_ns",
                   load(ctr_compact_parity_ack_post_ns),
                   load(ctr_compact_peer_payload_msgs));
        avg_metric("compact_peer_recv_to_enqueue_ns",
                   load(ctr_compact_peer_recv_to_enqueue_ns),
                   load(ctr_compact_peer_recv_to_enqueue_cnt));
        avg_metric("compact_peer_enqueue_to_dequeue_ns",
                   load(ctr_compact_peer_enqueue_to_dequeue_ns),
                   load(ctr_compact_peer_enqueue_to_dequeue_cnt));
        avg_metric("compact_peer_recv_to_handle_ns",
                   load(ctr_compact_peer_recv_to_handle_ns),
                   load(ctr_compact_peer_recv_to_handle_cnt));
        avg_metric("compact_peer_recv_to_ack_post_ns",
                   load(ctr_compact_peer_recv_to_ack_post_ns),
                   load(ctr_compact_peer_recv_to_ack_post_cnt));
        avg_metric("compact_peer_payload_dispatch_ns",
                   load(ctr_compact_peer_payload_dispatch_ns),
                   load(ctr_compact_peer_payload_dispatch_cnt));
        avg_metric("compact_peer_payload_repost_ns",
                   load(ctr_compact_peer_payload_repost_ns),
                   load(ctr_compact_peer_payload_repost_cnt));
        avg_metric("compact_peer_worker_service_ns",
                   load(ctr_compact_peer_worker_service_ns),
                   load(ctr_compact_peer_worker_service_cnt));
        avg_metric("compact_peer_send_cqe_reclaim_ns",
                   load(ctr_compact_peer_send_cqe_reclaim_ns),
                   load(ctr_compact_peer_send_cqe_reclaim_cnt));
        metric("remote_cpu_main_poll_overhead_wall_ns",
               main_poll_overhead_ns);
        metric("remote_cpu_main_poll_overhead_wall_pct_x100",
               pct_x100(main_poll_overhead_ns, wall_ns));
        metric("remote_cpu_completion_poller_spin_wall_ns",
               completion_poller_spin_ns);
        metric("remote_cpu_completion_poller_spin_wall_pct_x100",
               pct_x100(completion_poller_spin_ns, wall_ns));

        metric("recv_wc", load(ctr_recv_wc));
        metric("send_wc", load(ctr_send_wc));
        metric("peer_recv_wc", load(ctr_peer_recv_wc));
        metric("peer_send_wc", load(ctr_peer_send_wc));
        metric("data_req", load(ctr_data_req));
        metric("dispatch_enqueued", load(ctr_dispatch_enqueued));
        metric("dispatch_overflow", load(ctr_dispatch_overflow));
        metric("dispatch_queue_full", load(ctr_dispatch_queue_full));
        metric("client_batch_task_dequeued",
               load(ctr_client_batch_task_dequeued));
        avg_metric("client_batch_worker_service_ns",
                   load(ctr_client_batch_worker_service_ns),
                   load(ctr_client_batch_worker_service_cnt));
        avg_metric("client_batch_recv_to_enqueue_ns",
                   load(ctr_client_batch_recv_to_enqueue_ns),
                   load(ctr_client_batch_recv_to_enqueue_cnt));
        avg_metric("client_batch_recv_to_dequeue_ns",
                   load(ctr_client_batch_recv_to_dequeue_ns),
                   load(ctr_client_batch_recv_to_dequeue_cnt));
        avg_metric("worker_service_ns", load(ctr_worker_service_ns),
                   load(ctr_worker_task_dequeued));
        avg_metric("worker_queue_wait_ns", load(ctr_worker_queue_wait_ns),
                   load(ctr_worker_task_dequeued));
        avg_metric("dispatch_batch_lock_wait_ns",
                   load(ctr_dispatch_batch_lock_wait_ns),
                   load(ctr_dispatch_batch_lock_wait_cnt));
        avg_metric("addr_lock_wait_ns_data_req",
                   load(ctr_addr_lock_wait_ns_data_req), load(ctr_data_req));
        avg_metric("addr_lock_hold_ns_data_req",
                   load(ctr_addr_lock_hold_ns_data_req), load(ctr_data_req));
        metric("addr_lock_wait_max_ns", load(ctr_addr_lock_wait_max_ns));
        metric("addr_lock_hold_max_ns", load(ctr_addr_lock_hold_max_ns));
        avg_metric("data_req_msg_init_ns", load(ctr_data_req_msg_init_ns),
                   load(ctr_client_batch_task_dequeued));
        avg_metric("data_req_meta_fill_ns", load(ctr_data_req_meta_fill_ns),
                   load(ctr_data_req));
        avg_metric("data_req_get_ns", load(ctr_data_req_get_ns),
                   load(ctr_data_req));
        avg_metric("data_req_delta_ns", load(ctr_data_req_delta_ns),
                   load(ctr_data_req));
        avg_metric("data_req_delta_prepare_ns",
                   load(ctr_data_req_delta_prepare_ns), load(ctr_data_req));
        avg_metric("data_req_delta_encode_ns",
                   load(ctr_data_req_delta_encode_ns), load(ctr_data_req));
        avg_metric("data_req_copy_ns", load(ctr_data_req_copy_ns),
                   load(ctr_data_req));
        avg_metric("data_req_peer_enqueue_ns",
                   load(ctr_data_req_peer_enqueue_ns), load(ctr_data_req));
        avg_metric("batch_rpc_track_ns", load(ctr_batch_rpc_track_ns),
                   load(ctr_batch_rpc_track_cnt));
        avg_metric("batch_rpc_publish_ns", load(ctr_batch_rpc_publish_ns),
                   load(ctr_batch_rpc_publish_cnt));
        avg_metric("data_req_fanout_ready_ns",
                   load(ctr_data_req_fanout_ready_ns),
                   load(ctr_client_batch_task_dequeued));
        avg_metric("runtime_parity_send_to_handle_ns",
                   load(ctr_runtime_parity_batch_send_to_handle_ns),
                   load(ctr_runtime_parity_batch_send_to_handle_cnt));
        avg_metric("runtime_parity_recv_to_enqueue_ns",
                   load(ctr_runtime_parity_batch_recv_to_enqueue_ns),
                   load(ctr_runtime_parity_batch_recv_to_enqueue_cnt));
        metric("runtime_parity_recv_to_enqueue_max_ns",
               load(ctr_runtime_parity_batch_recv_to_enqueue_max_ns));
        avg_metric("runtime_parity_enqueue_to_dequeue_ns",
                   load(ctr_runtime_parity_batch_enqueue_to_dequeue_ns),
                   load(ctr_runtime_parity_batch_enqueue_to_dequeue_cnt));
        metric("runtime_parity_enqueue_to_dequeue_max_ns",
               load(ctr_runtime_parity_batch_enqueue_to_dequeue_max_ns));
        avg_metric("runtime_parity_worker_service_ns",
                   load(ctr_runtime_parity_batch_worker_service_ns),
                   load(ctr_runtime_parity_batch_worker_service_cnt));
        metric("runtime_parity_worker_service_max_ns",
               load(ctr_runtime_parity_batch_worker_service_max_ns));
        avg_metric("runtime_parity_recv_to_ack_post_ns",
                   load(ctr_runtime_parity_batch_recv_to_ack_post_ns),
                   load(ctr_runtime_parity_batch_recv_to_ack_post_cnt));
        avg_metric("runtime_parity_post_to_send_cqe_ns",
                   load(ctr_runtime_parity_batch_post_to_send_cqe_ns),
                   load(ctr_runtime_parity_batch_post_to_send_cqe_cnt));
        metric("runtime_parity_post_to_send_cqe_max_ns",
               load(ctr_runtime_parity_batch_post_to_send_cqe_max_ns));
        metric("runtime_parity_post_fail",
               load(ctr_runtime_parity_batch_post_fail));
        metric("runtime_parity_reserve_fail",
               load(ctr_runtime_parity_batch_reserve_fail));
        metric("runtime_parity_reserve_fail_outstanding_count",
               load(ctr_runtime_parity_reserve_fail_outstanding));
        metric("runtime_parity_reserve_fail_no_free_slot",
               load(ctr_runtime_parity_reserve_fail_no_free_slot));
        avg_metric("runtime_parity_reserve_fail_outstanding",
                   load(ctr_runtime_parity_reserve_fail_outstanding_sum),
                   load(ctr_runtime_parity_reserve_fail_outstanding_cnt));
        metric("runtime_parity_reserve_fail_outstanding_max",
               load(ctr_runtime_parity_reserve_fail_outstanding_max));
        metric("runtime_parity_ready_gate_fail",
               load(ctr_runtime_parity_batch_ready_gate_fail));
        metric("runtime_parity_ready_gate_slots",
               load(ctr_runtime_parity_batch_ready_gate_slots));
        avg_metric("runtime_parity_ready_to_post_ns",
                   load(ctr_runtime_parity_ready_to_post_ns),
                   load(ctr_runtime_parity_ready_to_post_cnt));
        metric("runtime_parity_ready_to_post_max_ns",
               load(ctr_runtime_parity_ready_to_post_max_ns));
        avg_metric("runtime_parity_ready_to_gate_fail_ns",
                   load(ctr_runtime_parity_ready_to_gate_fail_ns),
                   load(ctr_runtime_parity_ready_to_gate_fail_cnt));
        metric("runtime_parity_ready_to_gate_fail_max_ns",
               load(ctr_runtime_parity_ready_to_gate_fail_max_ns));
        avg_metric("runtime_parity_ready_gate_outstanding",
                   load(ctr_runtime_parity_ready_gate_outstanding_sum),
                   load(ctr_runtime_parity_ready_gate_outstanding_cnt));
        metric("runtime_parity_ready_gate_outstanding_max",
               load(ctr_runtime_parity_ready_gate_outstanding_max));
        avg_metric("runtime_parity_send_cqe_reclaim_ns",
                   load(ctr_runtime_parity_send_cqe_reclaim_ns),
                   load(ctr_runtime_parity_send_cqe_reclaim_cnt));
        metric("runtime_parity_send_cqe_reclaim_max_ns",
               load(ctr_runtime_parity_send_cqe_reclaim_max_ns));
        avg_metric("server_ec2pc_complete_ns",
                   load(ctr_server_ec2pc_complete_ns),
                   load(ctr_server_ec2pc_complete_cnt));
        avg_metric("server_ec2pc_fanout_ns",
                   load(ctr_server_ec2pc_fanout_ns),
                   load(ctr_server_ec2pc_fanout_cnt));
        avg_metric("server_ec2pc_first_ack_after_fanout_ns",
                   load(ctr_server_ec2pc_first_ack_after_fanout_ns),
                   load(ctr_server_ec2pc_first_ack_after_fanout_cnt));
        avg_metric("server_ec2pc_final_ack_after_fanout_ns",
                   load(ctr_server_ec2pc_final_ack_after_fanout_ns),
                   load(ctr_server_ec2pc_final_ack_after_fanout_cnt));
        avg_metric("server_ec2pc_fanout_to_parity_post_ns",
                   load(ctr_server_ec2pc_fanout_to_parity_post_ns),
                   load(ctr_server_ec2pc_fanout_to_parity_post_cnt));
        avg_metric("server_ec2pc_parity_post_to_ack_ns",
                   load(ctr_server_ec2pc_parity_post_to_ack_ns),
                   load(ctr_server_ec2pc_parity_post_to_ack_cnt));
        avg_metric("server_ec2pc_first_to_final_ack_ns",
                   load(ctr_server_ec2pc_first_to_final_ack_ns),
                   load(ctr_server_ec2pc_first_to_final_ack_cnt));
        avg_metric("server_ec2pc_track_lock_wait_ns",
                   load(ctr_server_ec2pc_track_lock_wait_ns),
                   load(ctr_server_ec2pc_track_lock_ops));
        avg_metric("server_ec2pc_track_lock_hold_ns",
                   load(ctr_server_ec2pc_track_lock_hold_ns),
                   load(ctr_server_ec2pc_track_lock_ops));
        avg_metric("peer_lane_busy_ns", load(ctr_peer_lane_busy_ns),
                   load(ctr_peer_lane_busy_retries));
        metric("peer_lane_post_ok", load(ctr_peer_lane_post_ok));
        avg_metric("peer_lane_post_ok_ns", load(ctr_peer_lane_post_ok_ns),
                   load(ctr_peer_lane_post_ok));
        metric("peer_lane_post_fail", load(ctr_peer_lane_post_fail));
        metric("peer_lane_enqueue_full", load(ctr_peer_lane_enqueue_full));
        avg_metric("peer_ack_recv_dispatch_ns",
                   load(ctr_peer_ack_recv_dispatch_ns),
                   load(ctr_peer_ack_recv_dispatch_cnt));
        avg_metric("peer_ack_consume_ns", load(ctr_peer_ack_consume_ns),
                   load(ctr_peer_ack_consume_cnt));
        metric("client_final_ack_enqueued",
               load(ctr_client_final_ack_enqueued));
        metric("client_final_ack_posted", load(ctr_client_final_ack_posted));
        avg_metric("client_final_ack_done_to_enqueue_ns",
                   load(ctr_client_final_ack_done_to_enqueue_ns),
                   load(ctr_client_final_ack_done_to_enqueue_cnt));
        metric("client_final_ack_done_to_enqueue_max_ns",
               load(ctr_client_final_ack_done_to_enqueue_max_ns));
        avg_metric("client_final_ack_enqueue_to_post_ns",
                   load(ctr_client_final_ack_enqueue_to_post_ns),
                   load(ctr_client_final_ack_enqueue_to_post_cnt));
        metric("client_final_ack_enqueue_to_post_max_ns",
               load(ctr_client_final_ack_enqueue_to_post_max_ns));
        avg_metric("client_final_ack_done_to_post_ns",
                   load(ctr_client_final_ack_done_to_post_ns),
                   load(ctr_client_final_ack_done_to_post_cnt));
        metric("client_final_ack_done_to_post_max_ns",
               load(ctr_client_final_ack_done_to_post_max_ns));
        avg_metric("client_final_ack_post_to_send_cqe_ns",
                   load(ctr_client_final_ack_post_to_send_cqe_ns),
                   load(ctr_client_final_ack_post_to_send_cqe_cnt));
        metric("client_final_ack_post_to_send_cqe_max_ns",
               load(ctr_client_final_ack_post_to_send_cqe_max_ns));
        metric("poll_loop_count", load(ctr_poll_loop_count));
        avg_metric("poll_loop_ns", load(ctr_poll_loop_ns),
                   load(ctr_poll_loop_count));
        metric("poll_loop_max_ns", load(ctr_poll_loop_max_ns));
        avg_metric("unified_data_phase_ns",
                   load(ctr_unified_data_phase_ns),
                   load(ctr_poll_loop_count));
        metric("unified_data_phase_max_ns",
               load(ctr_unified_data_phase_max_ns));
        avg_metric("unified_peer_phase_ns",
                   load(ctr_unified_peer_phase_ns),
                   load(ctr_poll_loop_count));
        metric("unified_peer_phase_max_ns",
               load(ctr_unified_peer_phase_max_ns));
        avg_metric("main_runtime_pair_flush_ns",
                   load(ctr_main_runtime_pair_flush_ns),
                   load(ctr_main_runtime_pair_flush_calls));
        avg_metric("main_runtime_pair_flush_gap_ns",
                   load(ctr_main_runtime_pair_flush_gap_ns),
                   load(ctr_main_runtime_pair_flush_gap_cnt));
        metric("main_runtime_pair_flush_gap_max_ns",
               load(ctr_main_runtime_pair_flush_gap_max_ns));
        metric("main_runtime_pair_flush_active_calls",
               load(ctr_main_runtime_pair_flush_active_calls));
        metric("main_runtime_pair_flush_jobs",
               load(ctr_main_runtime_pair_flush_jobs));
        avg_metric("main_peer_payload_transport_ns",
                   load(ctr_main_peer_payload_transport_ns),
                   load(ctr_main_peer_payload_transport_calls));
        avg_metric("main_peer_payload_transport_gap_ns",
                   load(ctr_main_peer_payload_transport_gap_ns),
                   load(ctr_main_peer_payload_transport_gap_cnt));
        metric("main_peer_payload_transport_gap_max_ns",
               load(ctr_main_peer_payload_transport_gap_max_ns));
        metric("main_peer_payload_transport_active_calls",
               load(ctr_main_peer_payload_transport_active_calls));
        avg_metric("main_client_ack_flush_ns",
                   load(ctr_main_client_ack_flush_ns),
                   load(ctr_main_client_ack_flush_calls));
        avg_metric("main_client_ack_flush_gap_ns",
                   load(ctr_main_client_ack_flush_gap_ns),
                   load(ctr_main_client_ack_flush_gap_cnt));
        metric("main_client_ack_flush_gap_max_ns",
               load(ctr_main_client_ack_flush_gap_max_ns));
        metric("main_client_ack_flush_posts",
               load(ctr_main_client_ack_flush_posts));
        metric("main_client_ack_flush_active_calls",
               load(ctr_main_client_ack_flush_active_calls));
        metric("completion_poller_loop_count",
               load(ctr_completion_poller_loop_cnt));
        avg_metric("completion_poller_loop_ns",
                   load(ctr_completion_poller_loop_ns),
                   load(ctr_completion_poller_loop_cnt));
        metric("completion_poller_loop_max_ns",
               load(ctr_completion_poller_loop_max_ns));
        avg_metric("completion_poller_gap_ns",
                   load(ctr_completion_poller_gap_ns),
                   load(ctr_completion_poller_gap_cnt));
        metric("completion_poller_gap_max_ns",
               load(ctr_completion_poller_gap_max_ns));
        avg_metric("completion_poller_data_drain_ns",
                   load(ctr_completion_poller_data_drain_ns),
                   load(ctr_completion_poller_data_drain_cnt));
        metric("completion_poller_data_drain_max_ns",
               load(ctr_completion_poller_data_drain_max_ns));
        avg_metric("completion_poller_peer_ack_drain_ns",
                   load(ctr_completion_poller_peer_ack_drain_ns),
                   load(ctr_completion_poller_peer_ack_drain_cnt));
        metric("completion_poller_peer_ack_drain_max_ns",
               load(ctr_completion_poller_peer_ack_drain_max_ns));
        avg_metric("completion_poller_payload_poll_ns",
                   load(ctr_completion_poller_payload_poll_ns),
                   load(ctr_completion_poller_payload_poll_cnt));
        metric("completion_poller_payload_poll_max_ns",
               load(ctr_completion_poller_payload_poll_max_ns));
        out << "SERVER_EC2PC_SUMMARY_END server=" << local_server_index
            << std::endl;
    }

    void maybe_print_periodic_ec2pc_server_summary() {
        if (!ec2pc_server_summary_enabled()) {
            return;
        }
        const char *interval_env =
            std::getenv("FARLIB_PRINT_SERVER_EC2PC_SUMMARY_INTERVAL_S");
        if (interval_env == nullptr) {
            return;
        }
        char *end = nullptr;
        unsigned long interval_s = std::strtoul(interval_env, &end, 10);
        if (end == interval_env || *end != '\0' || interval_s == 0) {
            return;
        }
        const uint64_t now_ns = steady_clock_now_ns();
        if (last_ec2pc_periodic_summary_ns == 0) {
            last_ec2pc_periodic_summary_ns = now_ns;
            return;
        }
        const uint64_t interval_ns = interval_s * 1000000000ull;
        if (now_ns - last_ec2pc_periodic_summary_ns < interval_ns) {
            return;
        }
        last_ec2pc_periodic_summary_ns = now_ns;
        print_ec2pc_server_summary();
    }

    void reset_rpc_runtime_counters() {
#define FARLIB_SERVER_RESET_RPC_RUNTIME_COUNTERS(X)                           \
        X(pending_send_total, 0)                                              \
        X(ctr_peer_recv_wc, 0)                                                \
        X(ctr_peer_send_wc, 0)                                                \
        X(ctr_probe_req, 0)                                                   \
        X(ctr_peer_enq_payload, 0)                                            \
        X(ctr_peer_enq_ack, 0)                                                \
        X(ctr_peer_flush_rounds, 0)                                           \
        X(ctr_peer_flush_posts, 0)                                            \
        X(ctr_peer_pending_send_peak, 0)                                      \
        X(ctr_recv_repost_ok, 0)                                              \
        X(ctr_recv_repost_fail, 0)                                            \
        X(ctr_recv_credit_zero, 0)                                            \
        X(ctr_recv_credit_min, rpc_queue_depth)                               \
        X(ctr_peer_payload_recv_credit_zero, 0)                               \
        X(ctr_peer_payload_recv_credit_min, std::numeric_limits<uint64_t>::max()) \
        X(ctr_dispatch_inline_ns, 0)                                          \
        X(ctr_dispatch_inline_max_ns, 0)                                      \
        X(ctr_dispatch_inline_slow, 0)                                        \
        X(ctr_dispatch_overflow, 0)                                           \
        X(ctr_client_batch_recv_to_enqueue_ns, 0)                             \
        X(ctr_client_batch_recv_to_enqueue_cnt, 0)                            \
        X(ctr_client_batch_recv_to_dequeue_ns, 0)                             \
        X(ctr_client_batch_recv_to_dequeue_cnt, 0)                            \
        X(ctr_worker_queue_wait_ns, 0)                                        \
        X(ctr_worker_service_ns, 0)                                           \
        X(ctr_worker_dequeue_lock_wait_ns, 0)                                 \
        X(ctr_worker_dequeue_lock_wait_cnt, 0)                                \
        X(ctr_worker_intertask_gap_ns, 0)                                     \
        X(ctr_worker_intertask_gap_cnt, 0)                                    \
        X(ctr_worker_post_service_queue_empty, 0)                             \
        X(ctr_worker_post_service_queue_nonempty, 0)                          \
        X(ctr_worker_post_service_overflow_nonempty, 0)                       \
        X(ctr_worker_post_service_queue_depth_sum, 0)                         \
        X(ctr_worker_post_service_probe_ns, 0)                                \
        X(ctr_worker_post_service_probe_cnt, 0)                               \
        X(ctr_worker_post_service_probe_lock_wait_ns, 0)                      \
        X(ctr_worker_post_service_probe_lock_wait_cnt, 0)                     \
        X(ctr_worker_post_task_flush_ns, 0)                                   \
        X(ctr_worker_post_task_flush_cnt, 0)                                  \
        X(ctr_worker_idle_flush_ns, 0)                                        \
        X(ctr_worker_idle_flush_cnt, 0)                                       \
        X(ctr_addr_lock_wait_ns_data_req, 0)                                  \
        X(ctr_addr_lock_hold_ns_data_req, 0)                                  \
        X(ctr_data_req_msg_init_ns, 0)                                        \
        X(ctr_data_req_meta_fill_ns, 0)                                       \
        X(ctr_data_req_get_ns, 0)                                             \
        X(ctr_data_req_delta_ns, 0)                                           \
        X(ctr_data_req_delta_prepare_ns, 0)                                   \
        X(ctr_data_req_delta_encode_ns, 0)                                    \
        X(ctr_data_req_fanout_ready_ns, 0)                                    \
        X(ctr_data_req_copy_ns, 0)                                            \
        X(ctr_data_req_peer_enqueue_ns, 0)                                    \
        X(ctr_dispatch_batch_lock_wait_ns, 0)                                 \
        X(ctr_dispatch_batch_lock_wait_cnt, 0)                                \
        X(ctr_batch_rpc_reserve_ns, 0)                                        \
        X(ctr_batch_rpc_reserve_cnt, 0)                                       \
        X(ctr_batch_rpc_track_ns, 0)                                          \
        X(ctr_batch_rpc_track_cnt, 0)                                         \
        X(ctr_batch_rpc_publish_ns, 0)                                        \
        X(ctr_batch_rpc_publish_cnt, 0)                                       \
        X(ctr_probe_compute_ns, 0)                                            \
        X(ctr_probe_real_reserve_ns, 0)                                       \
        X(ctr_probe_real_post_ns, 0)                                          \
        X(ctr_probe_real_span_ops, 0)                                         \
        X(ctr_probe_real_addr_lock_ns, 0)                                     \
        X(ctr_probe_real_get_ns, 0)                                           \
        X(ctr_probe_real_encode_ns, 0)                                        \
        X(ctr_probe_real_memcpy_ns, 0)                                        \
        X(ctr_probe_parity_batch_msgs, 0)                                     \
        X(ctr_probe_parity_batch_apply_ns, 0)                                 \
        X(ctr_probe_parity_batch_ack_post_ns, 0)                              \
        X(ctr_probe_parity_batch_worker_service_ns, 0)                        \
        X(ctr_probe_parity_batch_worker_service_cnt, 0)                       \
        X(ctr_probe_parity_batch_repost_ns, 0)                                \
        X(ctr_probe_parity_batch_repost_cnt, 0)                               \
        X(ctr_probe_parity_batch_span_ops, 0)                                 \
        X(ctr_probe_parity_batch_get_ns, 0)                                   \
        X(ctr_probe_parity_batch_xor_ns, 0)                                   \
        X(ctr_probe_parity_batch_send_to_handle_ns, 0)                        \
        X(ctr_probe_parity_batch_send_to_handle_cnt, 0)                       \
        X(ctr_probe_parity_batch_send_to_ack_post_ns, 0)                      \
        X(ctr_probe_parity_batch_send_to_ack_post_cnt, 0)                     \
        X(ctr_probe_parity_batch_post_to_send_cqe_ns, 0)                      \
        X(ctr_probe_parity_batch_post_to_send_cqe_cnt, 0)                     \
        X(ctr_runtime_parity_batch_send_to_handle_ns, 0)                      \
        X(ctr_runtime_parity_batch_send_to_handle_cnt, 0)                     \
        X(ctr_runtime_parity_batch_prepost_ns, 0)                             \
        X(ctr_runtime_parity_batch_prepost_cnt, 0)                            \
        X(ctr_runtime_parity_batch_send_to_recv_wc_ns, 0)                     \
        X(ctr_runtime_parity_batch_send_to_recv_wc_cnt, 0)                    \
        X(ctr_runtime_parity_batch_post_to_recv_wc_ns, 0)                     \
        X(ctr_runtime_parity_batch_post_to_recv_wc_cnt, 0)                    \
        X(ctr_runtime_parity_batch_recv_to_enqueue_ns, 0)                     \
        X(ctr_runtime_parity_batch_recv_to_enqueue_cnt, 0)                    \
        X(ctr_runtime_parity_batch_recv_to_enqueue_max_ns, 0)                 \
        X(ctr_runtime_parity_batch_enqueue_to_dequeue_ns, 0)                  \
        X(ctr_runtime_parity_batch_enqueue_to_dequeue_cnt, 0)                 \
        X(ctr_runtime_parity_batch_enqueue_to_dequeue_max_ns, 0)              \
        X(ctr_runtime_parity_batch_recv_to_handle_ns, 0)                      \
        X(ctr_runtime_parity_batch_recv_to_handle_cnt, 0)                     \
        X(ctr_runtime_parity_batch_worker_service_ns, 0)                      \
        X(ctr_runtime_parity_batch_worker_service_cnt, 0)                     \
        X(ctr_runtime_parity_batch_worker_service_max_ns, 0)                  \
        X(ctr_runtime_parity_batch_send_to_ack_post_ns, 0)                    \
        X(ctr_runtime_parity_batch_send_to_ack_post_cnt, 0)                   \
        X(ctr_runtime_parity_batch_recv_to_ack_post_ns, 0)                    \
        X(ctr_runtime_parity_batch_recv_to_ack_post_cnt, 0)                   \
        X(ctr_runtime_parity_batch_post_to_send_cqe_ns, 0)                    \
        X(ctr_runtime_parity_batch_post_to_send_cqe_cnt, 0)                   \
        X(ctr_runtime_parity_batch_post_to_send_cqe_max_ns, 0)                \
        X(ctr_runtime_parity_batch_post_fail, 0)                              \
        X(ctr_runtime_parity_batch_reserve_fail, 0)                           \
        X(ctr_runtime_parity_reserve_fail_invalid, 0)                         \
        X(ctr_runtime_parity_reserve_fail_no_shard, 0)                        \
        X(ctr_runtime_parity_reserve_fail_bad_qidx, 0)                        \
        X(ctr_runtime_parity_reserve_fail_no_queue, 0)                        \
        X(ctr_runtime_parity_reserve_fail_no_mutex, 0)                        \
        X(ctr_runtime_parity_reserve_fail_outstanding, 0)                     \
        X(ctr_runtime_parity_reserve_fail_no_free_slot, 0)                    \
        X(ctr_runtime_parity_reserve_fail_outstanding_sum, 0)                 \
        X(ctr_runtime_parity_reserve_fail_outstanding_cnt, 0)                 \
        X(ctr_runtime_parity_reserve_fail_outstanding_max, 0)                 \
        X(ctr_runtime_parity_batch_ready_gate_fail, 0)                        \
        X(ctr_runtime_parity_batch_ready_gate_slots, 0)                       \
        X(ctr_runtime_parity_item_copy_ns, 0)                                 \
        X(ctr_runtime_parity_item_copy_cnt, 0)                                \
        X(ctr_runtime_parity_slot_memcpy_ns, 0)                               \
        X(ctr_runtime_parity_slot_memcpy_cnt, 0)                              \
        X(ctr_runtime_parity_pending_queue_ns, 0)                             \
        X(ctr_runtime_parity_pending_queue_cnt, 0)                            \
        X(ctr_runtime_parity_ready_queue_ns, 0)                               \
        X(ctr_runtime_parity_ready_queue_cnt, 0)                              \
        X(ctr_runtime_parity_ready_to_post_ns, 0)                             \
        X(ctr_runtime_parity_ready_to_post_cnt, 0)                            \
        X(ctr_runtime_parity_ready_to_post_max_ns, 0)                         \
        X(ctr_runtime_parity_ready_to_gate_fail_ns, 0)                        \
        X(ctr_runtime_parity_ready_to_gate_fail_cnt, 0)                       \
        X(ctr_runtime_parity_ready_to_gate_fail_max_ns, 0)                    \
        X(ctr_runtime_parity_ready_gate_outstanding_sum, 0)                   \
        X(ctr_runtime_parity_ready_gate_outstanding_cnt, 0)                   \
        X(ctr_runtime_parity_ready_gate_outstanding_max, 0)                   \
        X(ctr_runtime_parity_send_cqe_reclaim_ns, 0)                          \
        X(ctr_runtime_parity_send_cqe_reclaim_cnt, 0)                         \
        X(ctr_runtime_parity_send_cqe_reclaim_max_ns, 0)                      \
        X(ctr_runtime_parity_outstanding_dec_before_sum, 0)                   \
        X(ctr_runtime_parity_outstanding_dec_before_cnt, 0)                   \
        X(ctr_runtime_parity_outstanding_dec_before_max, 0)                   \
        X(ctr_runtime_parity_pair_storage_try_success, 0)                     \
        X(ctr_runtime_parity_pair_storage_try_fail, 0)                        \
        X(ctr_runtime_parity_pair_storage_direct_success_ns, 0)               \
        X(ctr_runtime_parity_pair_storage_direct_success_cnt, 0)              \
        X(ctr_runtime_parity_pair_storage_direct_success_max_ns, 0)           \
        X(ctr_runtime_parity_pair_storage_pending_residency_ns, 0)            \
        X(ctr_runtime_parity_pair_storage_pending_residency_cnt, 0)           \
        X(ctr_runtime_parity_pair_storage_pending_residency_max_ns, 0)        \
        X(ctr_runtime_parity_pair_storage_retry_sum, 0)                       \
        X(ctr_runtime_parity_pair_storage_retry_cnt, 0)                       \
        X(ctr_runtime_parity_pair_storage_retry_max, 0)                       \
        X(ctr_runtime_parity_pair_storage_enqueue_depth_sum, 0)               \
        X(ctr_runtime_parity_pair_storage_enqueue_depth_cnt, 0)               \
        X(ctr_runtime_parity_pair_storage_enqueue_depth_max, 0)               \
        X(ctr_runtime_parity_ext_pair_try_success, 0)                         \
        X(ctr_runtime_parity_ext_pair_try_fail, 0)                            \
        X(ctr_runtime_parity_ext_pair_direct_success_ns, 0)                   \
        X(ctr_runtime_parity_ext_pair_direct_success_cnt, 0)                  \
        X(ctr_runtime_parity_ext_pair_direct_success_max_ns, 0)               \
        X(ctr_runtime_parity_ext_pair_pending_residency_ns, 0)                \
        X(ctr_runtime_parity_ext_pair_pending_residency_cnt, 0)               \
        X(ctr_runtime_parity_ext_pair_pending_residency_max_ns, 0)            \
        X(ctr_runtime_parity_ext_pair_retry_sum, 0)                           \
        X(ctr_runtime_parity_ext_pair_retry_cnt, 0)                           \
        X(ctr_runtime_parity_ext_pair_retry_max, 0)                           \
        X(ctr_runtime_parity_ext_pair_enqueue_depth_sum, 0)                   \
        X(ctr_runtime_parity_ext_pair_enqueue_depth_cnt, 0)                   \
        X(ctr_runtime_parity_ext_pair_enqueue_depth_max, 0)
#define FARLIB_SERVER_RESET_ONE(field, value) relaxed_store(field, value);
        FARLIB_SERVER_RESET_RPC_RUNTIME_COUNTERS(FARLIB_SERVER_RESET_ONE)
#undef FARLIB_SERVER_RESET_ONE
#undef FARLIB_SERVER_RESET_RPC_RUNTIME_COUNTERS
    }

    void reset_rpc_post_init_counters() {
#define FARLIB_SERVER_RESET_RPC_POST_INIT_COUNTERS(X)                         \
        X(ctr_peer_payload_poll_gap_ns, 0)                                    \
        X(ctr_peer_payload_poll_gap_cnt, 0)                                   \
        X(ctr_peer_payload_poll_gap_max_ns, 0)                                \
        X(ctr_peer_payload_cq_lock_wait_ns, 0)                                \
        X(ctr_peer_payload_cq_lock_wait_cnt, 0)                               \
        X(ctr_peer_payload_cq_lock_wait_max_ns, 0)                            \
        X(ctr_peer_payload_cq_poll_sys_ns, 0)                                 \
        X(ctr_peer_payload_cq_poll_sys_cnt, 0)                                \
        X(ctr_peer_payload_cq_poll_sys_max_ns, 0)                             \
        X(ctr_peer_payload_cq_handle_ns, 0)                                   \
        X(ctr_peer_payload_cq_handle_cnt, 0)                                  \
        X(ctr_peer_payload_cq_handle_max_ns, 0)                               \
        X(ctr_peer_payload_poll_calls, 0)                                     \
        X(ctr_peer_payload_poll_empty, 0)                                     \
        X(ctr_peer_payload_poll_nonempty, 0)                                  \
        X(ctr_peer_payload_poll_wc_total, 0)                                  \
        X(ctr_peer_payload_poll_batch_max, 0)                                 \
        X(ctr_peer_payload_poll_full_batch, 0)                                \
        X(ctr_peer_payload_round_ns, 0)                                       \
        X(ctr_peer_payload_round_cnt, 0)                                      \
        X(ctr_peer_payload_round_max_ns, 0)                                   \
        X(ctr_peer_payload_recv_gap_ns, 0)                                    \
        X(ctr_peer_payload_recv_gap_cnt, 0)                                   \
        X(ctr_peer_payload_recv_gap_max_ns, 0)                                \
        X(last_peer_payload_poll_time_ns_, 0)                                 \
        X(last_peer_payload_recv_time_ns_, 0)                                 \
        X(ctr_peer_ack_direct_post_ns, 0)                                     \
        X(ctr_peer_ack_direct_post_cnt, 0)                                    \
        X(ctr_peer_ack_direct_post_fail_cnt, 0)                               \
        X(ctr_peer_ack_enqueue_ns, 0)                                         \
        X(ctr_peer_ack_enqueue_cnt, 0)                                        \
        X(ctr_peer_ack_send_cqe_ns, 0)                                        \
        X(ctr_peer_ack_send_cqe_cnt, 0)                                       \
        X(ctr_peer_ack_recv_dispatch_ns, 0)                                   \
        X(ctr_peer_ack_recv_dispatch_cnt, 0)                                  \
        X(ctr_peer_ack_consume_ns, 0)                                         \
        X(ctr_peer_ack_consume_cnt, 0)                                        \
        X(ctr_client_final_ack_posted, 0)                                     \
        X(ctr_client_final_ack_done_to_enqueue_ns, 0)                         \
        X(ctr_client_final_ack_done_to_enqueue_cnt, 0)                        \
        X(ctr_client_final_ack_done_to_enqueue_max_ns, 0)                     \
        X(ctr_client_final_ack_enqueue_to_post_ns, 0)                         \
        X(ctr_client_final_ack_enqueue_to_post_cnt, 0)                        \
        X(ctr_client_final_ack_enqueue_to_post_max_ns, 0)                     \
        X(ctr_client_final_ack_done_to_post_ns, 0)                            \
        X(ctr_client_final_ack_done_to_post_cnt, 0)                           \
        X(ctr_client_final_ack_done_to_post_max_ns, 0)                        \
        X(ctr_client_final_ack_post_to_send_cqe_ns, 0)                        \
        X(ctr_client_final_ack_post_to_send_cqe_cnt, 0)                       \
        X(ctr_client_final_ack_post_to_send_cqe_max_ns, 0)                    \
        X(ctr_client_batch_send_to_recv_wc_ns, 0)                             \
        X(ctr_client_batch_send_to_recv_wc_cnt, 0)                            \
        X(ctr_server_ec2pc_complete_ns, 0)                                    \
        X(ctr_server_ec2pc_complete_cnt, 0)                                   \
        X(ctr_server_ec2pc_fanout_ns, 0)                                      \
        X(ctr_server_ec2pc_fanout_cnt, 0)                                     \
        X(ctr_server_ec2pc_first_ack_after_fanout_ns, 0)                      \
        X(ctr_server_ec2pc_first_ack_after_fanout_cnt, 0)                     \
        X(ctr_server_ec2pc_parity0_ack_after_fanout_ns, 0)                    \
        X(ctr_server_ec2pc_parity0_ack_after_fanout_cnt, 0)                   \
        X(ctr_server_ec2pc_parity1_ack_after_fanout_ns, 0)                    \
        X(ctr_server_ec2pc_parity1_ack_after_fanout_cnt, 0)                   \
        X(ctr_server_ec2pc_final_ack_after_fanout_ns, 0)                      \
        X(ctr_server_ec2pc_final_ack_after_fanout_cnt, 0)                     \
        X(ctr_server_ec2pc_fanout_to_parity_post_ns, 0)                       \
        X(ctr_server_ec2pc_fanout_to_parity_post_cnt, 0)                      \
        X(ctr_server_ec2pc_parity_post_to_ack_ns, 0)                          \
        X(ctr_server_ec2pc_parity_post_to_ack_cnt, 0)                         \
        X(ctr_server_ec2pc_first_to_final_ack_ns, 0)                          \
        X(ctr_server_ec2pc_first_to_final_ack_cnt, 0)                         \
        X(ctr_server_ec2pc_track_lock_wait_ns, 0)                             \
        X(ctr_server_ec2pc_track_lock_hold_ns, 0)                             \
        X(ctr_server_ec2pc_track_lock_ops, 0)                                 \
        X(ctr_peer_lane_busy_ns, 0)                                           \
        X(ctr_peer_lane_busy_retries, 0)                                      \
        X(ctr_peer_lane_post_ok_ns, 0)                                        \
        X(ctr_peer_lane_post_ok, 0)                                           \
        X(ctr_peer_lane_post_fail, 0)                                         \
        X(ctr_peer_lane_enqueue_full, 0)                                      \
        X(ctr_peer_send_lock_direct_wait_ns, 0)                               \
        X(ctr_peer_send_lock_direct_hold_ns, 0)                               \
        X(ctr_peer_send_lock_direct_ops, 0)                                   \
        X(ctr_peer_send_lock_batch_wait_ns, 0)                                \
        X(ctr_peer_send_lock_batch_hold_ns, 0)                                \
        X(ctr_peer_send_lock_batch_ops, 0)                                    \
        X(ctr_peer_send_lock_reclaim_wait_ns, 0)                              \
        X(ctr_peer_send_lock_reclaim_hold_ns, 0)                              \
        X(ctr_peer_send_lock_reclaim_ops, 0)                                  \
        X(ctr_addr_lock_wait_ns_parity_apply, 0)                              \
        X(ctr_addr_lock_hold_ns_parity_apply, 0)                              \
        X(ctr_addr_lock_wait_max_ns, 0)                                       \
        X(ctr_addr_lock_hold_max_ns, 0)                                       \
        X(ctr_poll_loop_count, 0)                                             \
        X(ctr_poll_loop_ns, 0)                                                \
        X(ctr_poll_loop_max_ns, 0)                                            \
        X(ctr_unified_data_phase_ns, 0)                                       \
        X(ctr_unified_data_phase_max_ns, 0)                                   \
        X(ctr_unified_peer_phase_ns, 0)                                       \
        X(ctr_unified_peer_phase_max_ns, 0)                                   \
        X(ctr_main_runtime_pair_flush_calls, 0)                               \
        X(ctr_main_runtime_pair_flush_ns, 0)                                  \
        X(ctr_main_runtime_pair_flush_max_ns, 0)                              \
        X(ctr_main_runtime_pair_flush_gap_ns, 0)                              \
        X(ctr_main_runtime_pair_flush_gap_cnt, 0)                             \
        X(ctr_main_runtime_pair_flush_gap_max_ns, 0)                          \
        X(ctr_main_runtime_pair_flush_jobs, 0)                                \
        X(ctr_main_runtime_pair_flush_active_calls, 0)                        \
        X(ctr_main_runtime_pair_flush_last_start_ns, 0)                       \
        X(ctr_main_peer_payload_transport_calls, 0)                           \
        X(ctr_main_peer_payload_transport_ns, 0)                              \
        X(ctr_main_peer_payload_transport_max_ns, 0)                          \
        X(ctr_main_peer_payload_transport_gap_ns, 0)                          \
        X(ctr_main_peer_payload_transport_gap_cnt, 0)                         \
        X(ctr_main_peer_payload_transport_gap_max_ns, 0)                      \
        X(ctr_main_peer_payload_transport_progress_calls, 0)                  \
        X(ctr_main_peer_payload_transport_active_calls, 0)                    \
        X(ctr_main_peer_payload_transport_last_start_ns, 0)                   \
        X(ctr_main_client_ack_flush_calls, 0)                                 \
        X(ctr_main_client_ack_flush_ns, 0)                                    \
        X(ctr_main_client_ack_flush_max_ns, 0)                                \
        X(ctr_main_client_ack_flush_gap_ns, 0)                                \
        X(ctr_main_client_ack_flush_gap_cnt, 0)                               \
        X(ctr_main_client_ack_flush_gap_max_ns, 0)                            \
        X(ctr_main_client_ack_flush_posts, 0)                                 \
        X(ctr_main_client_ack_flush_active_calls, 0)                          \
        X(ctr_main_client_ack_flush_last_start_ns, 0)                         \
        X(ctr_completion_poller_loop_ns, 0)                                   \
        X(ctr_completion_poller_loop_cnt, 0)                                  \
        X(ctr_completion_poller_loop_max_ns, 0)                               \
        X(ctr_completion_poller_gap_ns, 0)                                    \
        X(ctr_completion_poller_gap_cnt, 0)                                   \
        X(ctr_completion_poller_gap_max_ns, 0)                                \
        X(ctr_completion_poller_last_start_ns, 0)                             \
        X(ctr_completion_poller_data_drain_ns, 0)                             \
        X(ctr_completion_poller_data_drain_cnt, 0)                            \
        X(ctr_completion_poller_data_drain_max_ns, 0)                         \
        X(ctr_completion_poller_data_drain_gt1ms, 0)                          \
        X(ctr_completion_poller_data_drain_gt10ms, 0)                         \
        X(ctr_completion_poller_data_drain_gt100ms, 0)                        \
        X(ctr_completion_poller_data_drain_gt500ms, 0)                        \
        X(ctr_completion_poller_data_drain_gt1000ms, 0)                       \
        X(ctr_completion_poller_peer_ack_drain_ns, 0)                         \
        X(ctr_completion_poller_peer_ack_drain_cnt, 0)                        \
        X(ctr_completion_poller_peer_ack_drain_max_ns, 0)                     \
        X(ctr_completion_poller_payload_poll_ns, 0)                           \
        X(ctr_completion_poller_payload_poll_cnt, 0)                          \
        X(ctr_completion_poller_payload_poll_max_ns, 0)                       \
        X(ctr_data_drain_poll_sys_ns, 0)                                      \
        X(ctr_data_drain_poll_sys_cnt, 0)                                     \
        X(ctr_data_drain_poll_sys_max_ns, 0)                                  \
        X(ctr_data_drain_handle_ns, 0)                                        \
        X(ctr_data_drain_handle_cnt, 0)                                       \
        X(ctr_data_drain_handle_max_ns, 0)                                    \
        X(ctr_data_drain_dispatch_batch_ns, 0)                                \
        X(ctr_data_drain_dispatch_batch_cnt, 0)                               \
        X(ctr_data_drain_dispatch_batch_max_ns, 0)                            \
        X(ctr_data_drain_wc_total, 0)                                         \
        X(ctr_data_drain_call_wc_max, 0)                                      \
        X(ctr_data_drain_poll_full_batch, 0)                                  \
        X(ctr_batch_recv_poll_calls, 0)                                       \
        X(ctr_batch_recv_poll_wc_total, 0)                                    \
        X(ctr_batch_recv_poll_wc_max_batch, 0)                                \
        X(ctr_data_cq_poll_calls, 0)                                          \
        X(ctr_data_cq_poll_nonempty, 0)                                       \
        X(ctr_data_cq_poll_gap_ns, 0)                                         \
        X(ctr_data_cq_poll_gap_cnt, 0)                                        \
        X(ctr_data_cq_poll_gap_max_ns, 0)                                     \
        X(ctr_async_total, 0)                                                 \
        X(ctr_async_cq_err, 0)                                                \
        X(ctr_async_qp_fatal, 0)                                              \
        X(ctr_async_qp_req_err, 0)                                            \
        X(ctr_async_qp_access_err, 0)                                         \
        X(ctr_async_other, 0)                                                 \
        X(ctr_peer_flush_ns, 0)
#define FARLIB_SERVER_RESET_ONE(field, value) relaxed_store(field, value);
        FARLIB_SERVER_RESET_RPC_POST_INIT_COUNTERS(FARLIB_SERVER_RESET_ONE)
#undef FARLIB_SERVER_RESET_ONE
#undef FARLIB_SERVER_RESET_RPC_POST_INIT_COUNTERS
    }

    template <typename T, typename U>
    static void relaxed_store(std::atomic<T> &counter, U value) {
        counter.store(static_cast<T>(value), std::memory_order_relaxed);
    }

    template <typename T>
    static T relaxed_load(const std::atomic<T> &counter) {
        return counter.load(std::memory_order_relaxed);
    }

    template <typename T, typename U>
    static void relaxed_store_if(
        const std::unique_ptr<std::atomic<T>[]> &counters, size_t idx,
        U value) {
        if (counters != nullptr) {
            counters[idx].store(static_cast<T>(value),
                                std::memory_order_relaxed);
        }
    }

    template <typename TimePoint>
    static uint64_t elapsed_ns(const TimePoint &begin,
                               const TimePoint &end) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count());
    }

    template <typename... CounterArrays>
    static void reset_counter_arrays(size_t count, uint64_t value,
                                     const CounterArrays &...arrays) {
        for (size_t idx = 0; idx < count; idx++) {
            (relaxed_store_if(arrays, idx, value), ...);
        }
    }

    template <typename... CounterArrays>
    static bool all_counter_arrays_ready(const CounterArrays &...arrays) {
        return (... && (arrays != nullptr));
    }

    template <typename... Vectors>
    static void zero_vectors(Vectors &...vectors) {
        (std::fill(vectors.begin(), vectors.end(), uint64_t{0}), ...);
    }

    static void note_main_loop_stage(std::atomic<uint64_t> &call_ctr,
                                     std::atomic<uint64_t> &total_ns_ctr,
                                     std::atomic<uint64_t> &max_ns_ctr,
                                     std::atomic<uint64_t> &gap_ns_ctr,
                                     std::atomic<uint64_t> &gap_cnt_ctr,
                                     std::atomic<uint64_t> &gap_max_ns_ctr,
                                     std::atomic<uint64_t> &last_start_ns_ctr,
                                     std::atomic<uint64_t> &work_total_ctr,
                                     std::atomic<uint64_t> &active_call_ctr,
                                     uint64_t start_ns, uint64_t end_ns,
                                     uint64_t work_units) {
        call_ctr.fetch_add(1, std::memory_order_relaxed);
        if (end_ns >= start_ns) {
            uint64_t delta_ns = end_ns - start_ns;
            total_ns_ctr.fetch_add(delta_ns, std::memory_order_relaxed);
            update_peak(max_ns_ctr, delta_ns);
        }
        uint64_t prev_start =
            last_start_ns_ctr.exchange(start_ns, std::memory_order_relaxed);
        if (prev_start != 0 && start_ns > prev_start) {
            uint64_t gap_ns = start_ns - prev_start;
            gap_ns_ctr.fetch_add(gap_ns, std::memory_order_relaxed);
            gap_cnt_ctr.fetch_add(1, std::memory_order_relaxed);
            update_peak(gap_max_ns_ctr, gap_ns);
        }
        work_total_ctr.fetch_add(work_units, std::memory_order_relaxed);
        if (work_units != 0) {
            active_call_ctr.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static uint64_t steady_clock_now_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    static uint64_t wall_clock_now_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    static uint64_t timeval_to_ns(const timeval &tv) {
        return static_cast<uint64_t>(tv.tv_sec) * 1000000000ull +
               static_cast<uint64_t>(tv.tv_usec) * 1000ull;
    }

    static CpuUsageSample sample_process_cpu() {
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) {
            return {};
        }
        return {
            .user_ns = timeval_to_ns(usage.ru_utime),
            .sys_ns = timeval_to_ns(usage.ru_stime),
            .valid = true,
        };
    }

    static bool sample_thread_cpu(clockid_t clock_id, uint64_t *ns_out) {
        timespec ts{};
        if (clock_gettime(clock_id, &ts) != 0) {
            return false;
        }
        *ns_out = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
                  static_cast<uint64_t>(ts.tv_nsec);
        return true;
    }

    static ThreadClockSample capture_thread_clock(pthread_t thread) {
        ThreadClockSample sample{};
        if (pthread_getcpuclockid(thread, &sample.clock_id) != 0) {
            return sample;
        }
        sample.valid =
            sample_thread_cpu(sample.clock_id, &sample.baseline_ns);
        return sample;
    }

    void init_main_thread_cpu_clock() {
        main_thread_clock_valid =
            pthread_getcpuclockid(pthread_self(), &main_thread_clock_id) == 0;
    }

    void reset_remote_cpu_profile() {
        if (!main_thread_clock_valid) {
            init_main_thread_cpu_clock();
        }
        remote_cpu_reset_wall_ns = steady_clock_now_ns();
        remote_cpu_reset_tsc = get_cycles();
        remote_cpu_reset_process = sample_process_cpu();
        remote_cpu_reset_main_thread_ns = 0;
        if (main_thread_clock_valid) {
            (void)sample_thread_cpu(main_thread_clock_id,
                                    &remote_cpu_reset_main_thread_ns);
        }

        remote_cpu_rpc_worker_clocks.clear();
        remote_cpu_rpc_worker_clocks.reserve(rpc_workers.size());
        for (auto &worker : rpc_workers) {
            remote_cpu_rpc_worker_clocks.push_back(
                capture_thread_clock(worker.native_handle()));
        }

        remote_cpu_completion_poller_clocks.clear();
        remote_cpu_completion_poller_clocks.reserve(peer_pollers.size());
        for (auto &poller : peer_pollers) {
            remote_cpu_completion_poller_clocks.push_back(
                capture_thread_clock(poller.native_handle()));
        }
    }

    uint64_t sample_thread_group_cpu_ns(
        const std::vector<ThreadClockSample> &samples,
        uint64_t *sample_failures) const {
        uint64_t total_ns = 0;
        for (const auto &sample : samples) {
            uint64_t now_ns = 0;
            if (!sample.valid ||
                !sample_thread_cpu(sample.clock_id, &now_ns) ||
                now_ns < sample.baseline_ns) {
                (*sample_failures)++;
                continue;
            }
            total_ns += now_ns - sample.baseline_ns;
        }
        return total_ns;
    }

    static uint64_t pct_x100(uint64_t numerator, uint64_t denominator) {
        if (denominator == 0) {
            return 0;
        }
        return static_cast<uint64_t>(
            (static_cast<unsigned __int128>(numerator) * 10000u) /
            denominator);
    }

    uint64_t ns_to_cycles(uint64_t ns, uint64_t wall_ns,
                          uint64_t tsc_delta) const {
        if (wall_ns == 0 || tsc_delta == 0 || ns == 0) {
            return 0;
        }
        return static_cast<uint64_t>(
            (static_cast<unsigned __int128>(ns) * tsc_delta) / wall_ns);
    }

    void record_client_batch_post_wall_to_recv_wc_wall_ns(
        const rdma::EC2PCDataReqBatchMessage &msg) {
        uint64_t post_wall_time_ns = msg.post_wall_time_ns;
        if (post_wall_time_ns == 0) {
            return;
        }
        uint64_t recv_wc_wall_time_ns = wall_clock_now_ns();
        if (recv_wc_wall_time_ns < post_wall_time_ns) {
            return;
        }
        uint64_t delta_ns = recv_wc_wall_time_ns - post_wall_time_ns;
        ctr_client_batch_send_to_recv_wc_ns.fetch_add(
            delta_ns, std::memory_order_relaxed);
        ctr_client_batch_send_to_recv_wc_cnt.fetch_add(1,
                                                       std::memory_order_relaxed);
    }

#undef FARLIB_SERVER_RPC_EXTRA_COUNTER_FIELDS
#undef FARLIB_SERVER_RPC_SNAPSHOT_FIELDS

    size_t pending_send_depth_total() {
        return pending_send_total.load(std::memory_order_relaxed);
    }

    size_t pending_peer_send_depth_total() {
        return pending_peer_send_total.load(std::memory_order_relaxed);
    }

    size_t rpc_task_depth_total() {
        if (rpc_worker_count <= 1 || !rpc_task_queues || !rpc_task_mutexes) {
            return 0;
        }
        size_t total = 0;
        for (size_t w = 0; w < rpc_worker_count; w++) {
            std::lock_guard<std::mutex> lock(rpc_task_mutexes[w]);
            total += rpc_task_queues[w].size();
        }
        return total;
    }

    void drain_async_events_nonblocking() {
        while (true) {
            ibv_async_event event{};
            int ret = ibv_get_async_event(ctx.context, &event);
            if (ret != 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;
                }
                break;
            }
            ctr_async_total.fetch_add(1, std::memory_order_relaxed);
            switch (event.event_type) {
            case IBV_EVENT_CQ_ERR:
                ctr_async_cq_err.fetch_add(1, std::memory_order_relaxed);
                break;
            case IBV_EVENT_QP_FATAL:
                ctr_async_qp_fatal.fetch_add(1, std::memory_order_relaxed);
                break;
            case IBV_EVENT_QP_REQ_ERR:
                ctr_async_qp_req_err.fetch_add(1, std::memory_order_relaxed);
                break;
            case IBV_EVENT_QP_ACCESS_ERR:
                ctr_async_qp_access_err.fetch_add(1, std::memory_order_relaxed);
                break;
            default:
                ctr_async_other.fetch_add(1, std::memory_order_relaxed);
                break;
            }

            ibv_ack_async_event(&event);
        }
    }

    bool dedicated_peer_payload_worker_enabled() const {
        return is_probe_real_parity_only_endpoint() &&
               peer_payload_transport_ready();
    }

    static size_t normalize_worker_count(size_t worker_count) {
        return std::max<size_t>(1, worker_count);
    }

    size_t effective_rpc_worker_count() const {
        return normalize_worker_count(config.rpc_worker_count);
    }

    size_t configured_endpoint_count() const {
        return config.all_server_port_list.empty()
                   ? std::max<size_t>(1, static_cast<size_t>(config.server_count))
                   : config.all_server_port_list.size();
    }

    size_t probe_real_parity_data_endpoint_count() const {
        size_t endpoint_count = configured_endpoint_count();
        return endpoint_count > 2 ? endpoint_count - 2 : endpoint_count;
    }

    bool is_probe_real_parity_only_endpoint() const {
        return config.probe_real_parity_fanout &&
               local_server_index >= probe_real_parity_data_endpoint_count();
    }

    size_t parse_server_index_from_port() const {
        const char *port_env = std::getenv("SERVER_PORT");
        if (port_env == nullptr || port_env[0] == '\0') {
            return config.local_server_index;
        }
        char *end = nullptr;
        unsigned long port = std::strtoul(port_env, &end, 10);
        if (end == port_env || end == nullptr || *end != '\0') {
            return config.local_server_index;
        }
        if (!config.all_server_port_list.empty()) {
            for (size_t i = 0; i < config.all_server_port_list.size(); i++) {
                if (config.all_server_port_list[i] == port_env) {
                    return i;
                }
            }
        }
        return config.local_server_index;
    }

    size_t dedicated_completion_poller_count() const {
        if (is_probe_real_parity_only_endpoint()) {
            return 0;
        }
        // EC2PC and EC-span compaction use the same completion-sensitive RPC
        // path. Keep CQ poll/repost/reclaim off the flush/control thread.
        if (config.is_ec_2pc_mode() || config.is_ec_span_mode()) {
            return 1;
        }
        return 0;
    }

    bool dedicated_completion_poller_enabled() const {
        return dedicated_completion_poller_count() != 0 &&
               (!data_cqs.empty() || peer_cq != nullptr ||
                peer_payload_transport_ready());
    }

    size_t client_data_qp_count() const {
        return qps.size() > 0 ? (qps.size() - 1) : 0;
    }

    size_t configured_rpc_queue_depth() const {
        const size_t recv_cap =
            std::max<size_t>(1, static_cast<size_t>(config.qp_recv_cap));
        const size_t send_cap =
            std::max<size_t>(1, static_cast<size_t>(config.qp_send_cap));
        return std::min(kRpcDepth, std::min(recv_cap, send_cap));
    }

    size_t configured_peer_lane_depth() const {
        size_t workers = std::max<size_t>(1, rpc_worker_count);
        size_t qp_window = std::max<size_t>(
            1, std::min<size_t>(
                   {static_cast<size_t>(config.qp_send_cap),
                    static_cast<size_t>(config.qp_recv_cap), kPeerRpcDepth}));
        return std::max<size_t>(1, (qp_window + workers - 1) / workers);
    }

    size_t configured_peer_lane_pending_depth() const {
        size_t data_qp_count = client_data_qp_count();
        size_t max_outstanding =
            std::max<size_t>(1, data_qp_count) *
            std::max<size_t>(1, static_cast<size_t>(config.qp_send_cap));
        return std::max<size_t>(16384, max_outstanding);
    }

    size_t configured_peer_payload_qp_count() const {
        const char *forced_qp_count =
            std::getenv("FARLIB_FORCE_PEER_PAYLOAD_QP_COUNT");
        if (forced_qp_count != nullptr && forced_qp_count[0] != '\0') {
            char *end = nullptr;
            unsigned long parsed =
                std::strtoul(forced_qp_count, &end, 10);
            if (end != forced_qp_count && end != nullptr && *end == '\0') {
                return std::max<size_t>(1, static_cast<size_t>(parsed));
            }
        }
        if (config.peer_payload_qp_count_override != 0) {
            return std::max<size_t>(1, config.peer_payload_qp_count_override);
        }
        size_t worker_count =
            std::max<size_t>(1, normalize_worker_count(config.rpc_worker_count));
        size_t target = worker_count;
        return std::max<size_t>(1, std::min<size_t>(8, target));
    }

    bool dedicated_peer_payload_cq_per_qp() const {
        // Runtime EC2PC parity traffic now uses the same peer-payload transport
        // as the older parity-fanout probe path. Once we have more than one
        // payload QP per peer, funneling all of them back into one CQ creates
        // a single full-batch drain point on the parity side. Keep each
        // payload QP on its own CQ for EC2PC so the poller can drain them
        // independently.
        return config.is_ec_2pc_mode() &&
               std::max<size_t>(1, peer_payload_qp_count) > 1;
    }

    size_t worker_owned_peer_payload_qp_span() const {
        return 1;
    }

    size_t worker_owned_peer_payload_shard(size_t worker_idx,
                                           uint64_t selector = 0) const {
        size_t payload_qp_count = std::max<size_t>(1, peer_payload_qp_count);
        size_t shard_span = worker_owned_peer_payload_qp_span();
        size_t shard_base = (worker_idx * shard_span) % payload_qp_count;
        if (shard_span <= 1) {
            return shard_base;
        }
        return (shard_base + (selector % shard_span)) % payload_qp_count;
    }

    size_t configured_probe_parity_batch_send_depth() const {
        size_t qp_window = std::max<size_t>(
            1, std::min<size_t>(static_cast<size_t>(config.qp_send_cap), 1024));
        return std::max<size_t>(128, qp_window);
    }

    size_t configured_probe_parity_batch_pending_depth() const {
        size_t send_depth = configured_probe_parity_batch_send_depth();
        return std::min<size_t>(512, std::max<size_t>(128, send_depth * 2));
    }

    size_t configured_runtime_parity_batch_outstanding_limit() const {
        const char *forced_limit =
            std::getenv("FARLIB_FORCE_RUNTIME_PARITY_OUTSTANDING_LIMIT");
        if (forced_limit != nullptr && forced_limit[0] != '\0') {
            char *end = nullptr;
            unsigned long parsed = std::strtoul(forced_limit, &end, 10);
            if (end != forced_limit && end != nullptr && *end == '\0') {
                return std::max<size_t>(1, static_cast<size_t>(parsed));
            }
        }
        size_t send_cap =
            std::max<size_t>(1, static_cast<size_t>(config.qp_send_cap));
        // Keep each payload QP well below the full verbs window. The current
        // EC2PC runtime path still drives the hot parity peer into long
        // sender/remote-CQ queueing at 512 outstanding per QP, so shrink the
        // default window to 256. With two runtime payload QPs per peer, that
        // still leaves a 512-message per-peer transport window while cutting
        // the worst queueing depth in half.
        return std::max<size_t>(64, std::min<size_t>(256, send_cap));
    }

    size_t configured_runtime_parity_batch_soft_limit() const {
        const char *forced_limit =
            std::getenv("FARLIB_FORCE_RUNTIME_PARITY_OUTSTANDING_SOFT_LIMIT");
        if (forced_limit != nullptr && forced_limit[0] != '\0') {
            char *end = nullptr;
            unsigned long parsed = std::strtoul(forced_limit, &end, 10);
            if (end != forced_limit && end != nullptr && *end == '\0') {
                return std::max<size_t>(1, static_cast<size_t>(parsed));
            }
        }
        size_t hard_limit = configured_runtime_parity_batch_outstanding_limit();
        return std::max<size_t>(64, std::min<size_t>(128, hard_limit));
    }

    size_t configured_peer_ack_send_depth() const {
        size_t qp_window = std::max<size_t>(
            1, std::min<size_t>(static_cast<size_t>(config.qp_send_cap), 1024));
        return std::max<size_t>(64, qp_window);
    }

    size_t peer_payload_qp_index(size_t peer_idx, size_t shard_idx) const {
        return peer_idx * peer_payload_qp_count + shard_idx;
    }

    size_t peer_payload_cq_group_index(size_t peer_idx,
                                       size_t shard_idx = 0) const {
        if (dedicated_peer_payload_cq_per_qp()) {
            return peer_payload_qp_index(peer_idx, shard_idx);
        }
        (void)shard_idx;
        return config.peer_payload_cq_per_peer ? peer_idx : 0;
    }

    CompleteQueue *peer_payload_cq_for(size_t peer_idx,
                                       size_t shard_idx = 0) const {
        size_t cq_idx = peer_payload_cq_group_index(peer_idx, shard_idx);
        if (cq_idx >= peer_payload_cqs.size()) {
            return nullptr;
        }
        return peer_payload_cqs[cq_idx].get();
    }

    bool peer_payload_transport_ready() const {
        return !peer_payload_cqs.empty() && !peer_payload_recv_queues.empty();
    }

    size_t probe_parity_batch_queue_index(size_t peer_idx,
                                          size_t shard_idx) const {
        return peer_payload_qp_index(peer_idx, shard_idx);
    }

    size_t configured_server_thread_slots() const {
        return 1 + effective_rpc_worker_count() +
               dedicated_completion_poller_count();
    }

    size_t worker_thread_slot(size_t worker_idx) const {
        return 1 + dedicated_completion_poller_count() + worker_idx;
    }

    size_t peer_payload_poller_thread_slot(size_t poller_idx) const {
        return 1 + poller_idx;
    }

    bool lookup_controlled_ec2pc_pin_core(size_t thread_slot,
                                          size_t *core_out) const {
        size_t endpoint_count = config.all_server_port_list.empty()
                                    ? static_cast<size_t>(config.server_count)
                                    : config.all_server_port_list.size();
        if (endpoint_count != 6) {
            return false;
        }
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        if (cpu_count <= 0) {
            return false;
        }
        size_t logical_cpu_count = static_cast<size_t>(cpu_count);
        size_t server_idx = parse_server_index_from_port();
        if (server_idx >= endpoint_count) {
            return false;
        }
        if (config.get_configured_server_pin_core(server_idx, thread_slot,
                                                  core_out)) {
            if (*core_out >= logical_cpu_count) {
                return false;
            }
            return true;
        }
        // Host 10.156.112.101 exposes 20 physical cores with SMT siblings
        // at +20. Keep the prior EC2PC runtime pinning layout here because it
        // delivered the better end-to-end throughput on the current topology,
        // even though some data servers share cores.
        static constexpr size_t kPinnedCores[6][4] = {
            {0, 1, 2, 3},      // data 0
            {3, 4, 5, 6},      // data 1
            {6, 7, 8, 9},      // data 2
            {9, 10, 11, 0},    // data 3
            {12, 13, 14, 15},  // parity 4
            {16, 17, 18, 19},  // parity 5
        };
        if (thread_slot >= std::size(kPinnedCores[0])) {
            return false;
        }
        if (server_idx >= std::size(kPinnedCores)) {
            return false;
        }
        size_t core = kPinnedCores[server_idx][thread_slot];
        if (core >= logical_cpu_count) {
            return false;
        }
        *core_out = core;
        return true;
    }

    size_t pin_slot_to_core(size_t thread_slot) const {
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        if (cpu_count <= 0) {
            return 0;
        }
        size_t controlled_core = 0;
        if (lookup_controlled_ec2pc_pin_core(thread_slot, &controlled_core)) {
            return controlled_core;
        }
        size_t per_server_threads = configured_server_thread_slots();
        size_t server_idx = parse_server_index_from_port();
        size_t global_slot = server_idx * per_server_threads + thread_slot;
        return global_slot % static_cast<size_t>(cpu_count);
    }

    void pin_current_thread(size_t thread_slot, const char *role,
                            size_t role_idx) const {
        (void)role;
        (void)role_idx;
        const char *disable_pin = std::getenv("FARLIB_DISABLE_SERVER_PIN");
        if (disable_pin != nullptr && disable_pin[0] == '1') {
            return;
        }
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        if (cpu_count <= 0) {
            return;
        }
        size_t controlled_core = 0;
        bool controlled_pin =
            lookup_controlled_ec2pc_pin_core(thread_slot, &controlled_core);
        (void)controlled_pin;
        size_t core = pin_slot_to_core(thread_slot);
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(static_cast<int>(core), &set);
        int ret = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if (ret != 0) {
            return;
        }
#if FARLIB_EC2PC_VERBOSE_DIAG
        std::cout << "INFO: server pin server="
                  << parse_server_index_from_port()
                  << " role=" << role
                  << " role_idx=" << role_idx
                  << " slot=" << thread_slot
                  << " cpu=" << core
                  << " controlled_pin=" << (controlled_pin ? 1 : 0)
                  << " actual_cpu=" << sched_getcpu() << std::endl;
#endif
    }

    int controlled_peer_payload_comp_vector() const {
        return 0;
    }

    bool try_pop_worker_task(size_t worker_idx, RpcTask &task) {
        if (!rpc_task_queues || !rpc_task_mutexes || worker_idx >= rpc_worker_count) {
            return false;
        }
        std::lock_guard<std::mutex> lock(rpc_task_mutexes[worker_idx]);
        if (rpc_task_queues[worker_idx].empty()) {
            return false;
        }
        task = std::move(rpc_task_queues[worker_idx].front());
        rpc_task_queues[worker_idx].pop_front();
        return true;
    }

    bool worker_has_pending_task(size_t worker_idx) {
        if (!rpc_task_queues || !rpc_task_mutexes || worker_idx >= rpc_worker_count) {
            return false;
        }
        std::lock_guard<std::mutex> lock(rpc_task_mutexes[worker_idx]);
        return !rpc_task_queues[worker_idx].empty();
    }

    bool poll_peer_payload_cq_only_once(size_t max_batches = SIZE_MAX);
    bool flush_peer_payload_transport_once(size_t max_batches = SIZE_MAX);
    bool drain_peer_payload_cq_once(size_t max_batches = SIZE_MAX);

    void track_server_ec2pc_request(uint64_t wr_id, size_t client_qp_local_idx,
                                    uint8_t pending_parity_acks,
                                    uint16_t completion_msg_type =
                                        rdma::EC2PC_MSG_ACK_BATCH,
                                    uint16_t target_endpoint = 0,
                                    uint16_t parity_peer0 =
                                        std::numeric_limits<uint16_t>::max(),
                                    uint16_t parity_peer1 =
                                        std::numeric_limits<uint16_t>::max(),
                                    uint64_t start_time_ns = 0) {
        if (start_time_ns == 0) {
            start_time_ns = steady_clock_now_ns();
        }
        std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
        server_ec2pc_track_[wr_id] = {client_qp_local_idx, pending_parity_acks,
                                      completion_msg_type, target_endpoint,
                                      start_time_ns,
                                      0,
                                      0,
                                      0,
                                      parity_peer0,
                                      parity_peer1,
                                      0,
                                      0,
                                      0,
                                      0};
    }

    // Record one whole client batch into the server-side EC2PC tracker under
    // one mutex acquisition.
    void track_server_ec2pc_batch_requests(
        const uint64_t *wr_ids, uint16_t count, size_t client_qp_local_idx,
        uint8_t pending_parity_acks,
        uint16_t completion_msg_type = rdma::EC2PC_MSG_ACK_BATCH,
        uint16_t target_endpoint = 0,
        uint16_t parity_peer0 = std::numeric_limits<uint16_t>::max(),
        uint16_t parity_peer1 = std::numeric_limits<uint16_t>::max(),
        uint64_t worker_begin_time_ns = 0, uint64_t start_time_ns = 0) {
        if (wr_ids == nullptr || count == 0) {
            return;
        }
        if (start_time_ns == 0) {
            start_time_ns = steady_clock_now_ns();
        }
        if (worker_begin_time_ns == 0) {
            worker_begin_time_ns = steady_clock_now_ns();
        }
        std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
        for (uint16_t i = 0; i < count; i++) {
            server_ec2pc_track_[wr_ids[i]] = {
                client_qp_local_idx, pending_parity_acks, completion_msg_type,
                target_endpoint,  start_time_ns,         worker_begin_time_ns,
                0,                0,                     parity_peer0,
                parity_peer1,     0,                     0,
                0,                0};
        }
    }

    void note_server_ec2pc_worker_begin(uint64_t wr_id,
                                        uint64_t worker_begin_time_ns = 0) {
        if (worker_begin_time_ns == 0) {
            worker_begin_time_ns = steady_clock_now_ns();
        }
        std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
        auto it = server_ec2pc_track_.find(wr_id);
        if (it != server_ec2pc_track_.end() &&
            it->second.worker_begin_time_ns == 0) {
            it->second.worker_begin_time_ns = worker_begin_time_ns;
        }
    }

    void note_server_ec2pc_fanout_ready(uint64_t wr_id,
                                        uint64_t ready_time_ns = 0) {
        if (ready_time_ns == 0) {
            ready_time_ns = steady_clock_now_ns();
        }
        std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
        auto it = server_ec2pc_track_.find(wr_id);
        if (it != server_ec2pc_track_.end() &&
            it->second.fanout_ready_time_ns == 0) {
            it->second.fanout_ready_time_ns = ready_time_ns;
            if (ready_time_ns >= it->second.start_time_ns) {
                ctr_server_ec2pc_fanout_ns.fetch_add(
                    ready_time_ns - it->second.start_time_ns,
                    std::memory_order_relaxed);
                ctr_server_ec2pc_fanout_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
    }

    // Mark one whole client batch as ready-for-fanout under one mutex
    // acquisition.
    void note_server_ec2pc_batch_fanout_ready(const uint64_t *wr_ids,
                                              uint16_t count,
                                              uint64_t ready_time_ns = 0) {
        if (wr_ids == nullptr || count == 0) {
            return;
        }
        if (ready_time_ns == 0) {
            ready_time_ns = steady_clock_now_ns();
        }
        uint64_t batch_fanout_ns = 0;
        uint64_t batch_fanout_cnt = 0;
        {
            std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
            for (uint16_t i = 0; i < count; i++) {
                auto it = server_ec2pc_track_.find(wr_ids[i]);
                if (it == server_ec2pc_track_.end() ||
                    it->second.fanout_ready_time_ns != 0) {
                    continue;
                }
                it->second.fanout_ready_time_ns = ready_time_ns;
                if (ready_time_ns >= it->second.start_time_ns) {
                    batch_fanout_ns += ready_time_ns - it->second.start_time_ns;
                    batch_fanout_cnt++;
                }
            }
        }
        if (batch_fanout_cnt > 0) {
            ctr_server_ec2pc_fanout_ns.fetch_add(batch_fanout_ns,
                                                 std::memory_order_relaxed);
            ctr_server_ec2pc_fanout_cnt.fetch_add(batch_fanout_cnt,
                                                  std::memory_order_relaxed);
        }
    }

    void note_server_ec2pc_parity_payload_posted(
        size_t peer_idx, const rdma::ProbeParityBatchMessage &msg,
        uint64_t post_time_ns) {
        if (post_time_ns == 0) {
            return;
        }
        uint16_t count =
            std::min<uint16_t>(msg.ack_count, rdma::kProbeParityBatchMaxSpans);
        if (count == 0) {
            count = std::min<uint16_t>(msg.span_count,
                                       rdma::kProbeParityBatchMaxSpans);
        }
        std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
        for (uint16_t i = 0; i < count; i++) {
            uint64_t wr_id = msg.wr_ids[i];
            if (wr_id == 0) {
                continue;
            }
            auto it = server_ec2pc_track_.find(wr_id);
            if (it == server_ec2pc_track_.end()) {
                continue;
            }
            uint64_t *post_slot = nullptr;
            if (peer_idx == it->second.parity_peer0) {
                post_slot = &it->second.parity_peer0_post_time_ns;
            } else if (peer_idx == it->second.parity_peer1) {
                post_slot = &it->second.parity_peer1_post_time_ns;
            }
            if (post_slot == nullptr || *post_slot != 0) {
                continue;
            }
            *post_slot = post_time_ns;
            if (it->second.fanout_ready_time_ns != 0 &&
                post_time_ns >= it->second.fanout_ready_time_ns) {
                ctr_server_ec2pc_fanout_to_parity_post_ns.fetch_add(
                    post_time_ns - it->second.fanout_ready_time_ns,
                    std::memory_order_relaxed);
                ctr_server_ec2pc_fanout_to_parity_post_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
    }

    bool consume_server_ec2pc_ack(uint64_t wr_id, bool &done,
                                  size_t &client_qp_local_idx,
                                  uint16_t &completion_msg_type,
                                  uint16_t &target_endpoint,
                                  uint64_t &request_start_time_ns,
                                  size_t peer_idx) {
        done = false;
        client_qp_local_idx = 0;
        completion_msg_type = rdma::EC2PC_MSG_ACK_BATCH;
        target_endpoint = 0;
        request_start_time_ns = 0;
        std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
        auto it = server_ec2pc_track_.find(wr_id);
        if (it == server_ec2pc_track_.end()) {
            static std::atomic<uint64_t> peer_ack_track_miss_seen{0};
            uint64_t miss_seq =
                peer_ack_track_miss_seen.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            if (miss_seq <= 64 || (miss_seq % 1000) == 0) {
                std::cerr << "WARN: server peer ACK unknown wr_id"
                          << " server=" << local_server_index
                          << " peer_idx=" << peer_idx
                          << " miss_seq=" << miss_seq
                          << " wr_id=0x" << std::hex << wr_id << std::dec
                          << std::endl;
            }
            return false;
        }
        uint64_t now_ns = steady_clock_now_ns();
        uint64_t anchor_ns = it->second.fanout_ready_time_ns != 0
                                 ? it->second.fanout_ready_time_ns
                                 : it->second.start_time_ns;
        bool is_peer0 = peer_idx == it->second.parity_peer0;
        bool is_peer1 = peer_idx == it->second.parity_peer1;
        bool peer_matches = is_peer0 || is_peer1;
        if (!peer_matches) {
            static std::atomic<uint64_t> peer_ack_wrong_peer_seen{0};
            uint64_t wrong_seq =
                peer_ack_wrong_peer_seen.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            if (wrong_seq <= 64 || (wrong_seq % 1000) == 0) {
                std::cerr << "WARN: server peer ACK wrong peer"
                          << " server=" << local_server_index
                          << " peer_idx=" << peer_idx
                          << " wrong_seq=" << wrong_seq
                          << " wr_id=0x" << std::hex << wr_id << std::dec
                          << " expect0="
                          << static_cast<unsigned>(it->second.parity_peer0)
                          << " expect1="
                          << static_cast<unsigned>(it->second.parity_peer1)
                          << " pending="
                          << static_cast<unsigned>(
                                 it->second.pending_parity_acks)
                          << std::endl;
            }
            return false;
        }
        bool duplicate_peer_ack =
            (is_peer0 && it->second.parity_peer0_ack_time_ns != 0) ||
            (is_peer1 && it->second.parity_peer1_ack_time_ns != 0);
        if (now_ns >= anchor_ns) {
            if (is_peer0 && !duplicate_peer_ack) {
                it->second.parity_peer0_ack_time_ns = now_ns;
                if (it->second.parity_peer0_post_time_ns != 0 &&
                    now_ns >= it->second.parity_peer0_post_time_ns) {
                    ctr_server_ec2pc_parity_post_to_ack_ns.fetch_add(
                        now_ns - it->second.parity_peer0_post_time_ns,
                        std::memory_order_relaxed);
                    ctr_server_ec2pc_parity_post_to_ack_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                }
                ctr_server_ec2pc_parity0_ack_after_fanout_ns.fetch_add(
                    now_ns - anchor_ns, std::memory_order_relaxed);
                ctr_server_ec2pc_parity0_ack_after_fanout_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            } else if (is_peer1 && !duplicate_peer_ack) {
                it->second.parity_peer1_ack_time_ns = now_ns;
                if (it->second.parity_peer1_post_time_ns != 0 &&
                    now_ns >= it->second.parity_peer1_post_time_ns) {
                    ctr_server_ec2pc_parity_post_to_ack_ns.fetch_add(
                        now_ns - it->second.parity_peer1_post_time_ns,
                        std::memory_order_relaxed);
                    ctr_server_ec2pc_parity_post_to_ack_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                }
                ctr_server_ec2pc_parity1_ack_after_fanout_ns.fetch_add(
                    now_ns - anchor_ns, std::memory_order_relaxed);
                ctr_server_ec2pc_parity1_ack_after_fanout_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            } else if (duplicate_peer_ack) {
                static std::atomic<uint64_t> peer_ack_duplicate_seen{0};
                uint64_t dup_seq =
                    peer_ack_duplicate_seen.fetch_add(
                        1, std::memory_order_relaxed) +
                    1;
                if (dup_seq <= 64 || (dup_seq % 1000) == 0) {
                    std::cerr << "WARN: server peer ACK duplicate"
                              << " server=" << local_server_index
                              << " peer_idx=" << peer_idx
                              << " dup_seq=" << dup_seq
                              << " wr_id=0x" << std::hex << wr_id << std::dec
                              << " peer0_seen="
                              << (it->second.parity_peer0_ack_time_ns != 0)
                              << " peer1_seen="
                              << (it->second.parity_peer1_ack_time_ns != 0)
                              << " pending="
                              << static_cast<unsigned>(
                                     it->second.pending_parity_acks)
                              << std::endl;
                }
            }
        }
        if (duplicate_peer_ack) {
            return false;
        }
        if (it->second.first_ack_time_ns == 0 && now_ns >= anchor_ns) {
            it->second.first_ack_time_ns = now_ns;
            ctr_server_ec2pc_first_ack_after_fanout_ns.fetch_add(
                now_ns - anchor_ns, std::memory_order_relaxed);
            ctr_server_ec2pc_first_ack_after_fanout_cnt.fetch_add(
                1, std::memory_order_relaxed);
        }
        client_qp_local_idx = it->second.client_qp_local_idx;
        completion_msg_type = it->second.completion_msg_type;
        target_endpoint = it->second.target_endpoint;
        request_start_time_ns = it->second.start_time_ns;
        if (it->second.pending_parity_acks == 0) {
            static std::atomic<uint64_t> peer_ack_overcomplete_seen{0};
            uint64_t seq =
                peer_ack_overcomplete_seen.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            if (seq <= 64 || (seq % 1000) == 0) {
                std::cerr << "WARN: server peer ACK after completion"
                          << " server=" << local_server_index
                          << " peer_idx=" << peer_idx
                          << " seq=" << seq
                          << " wr_id=0x" << std::hex << wr_id << std::dec
                          << std::endl;
            }
            return false;
        }
        it->second.pending_parity_acks--;
        if (it->second.pending_parity_acks == 0) {
            if (now_ns >= it->second.start_time_ns) {
                ctr_server_ec2pc_complete_ns.fetch_add(
                    now_ns - it->second.start_time_ns,
                    std::memory_order_relaxed);
                ctr_server_ec2pc_complete_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (now_ns >= anchor_ns) {
                ctr_server_ec2pc_final_ack_after_fanout_ns.fetch_add(
                    now_ns - anchor_ns, std::memory_order_relaxed);
                ctr_server_ec2pc_final_ack_after_fanout_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (it->second.first_ack_time_ns != 0 &&
                now_ns >= it->second.first_ack_time_ns) {
                ctr_server_ec2pc_first_to_final_ack_ns.fetch_add(
                    now_ns - it->second.first_ack_time_ns,
                    std::memory_order_relaxed);
                ctr_server_ec2pc_first_to_final_ack_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
            server_ec2pc_track_.erase(it);
            done = true;
        }
        return true;
    }

    size_t rpc_task_worker_idx(const rdma::EC2PCRpcMessage &msg) const {
        if (rpc_worker_count <= 1) {
            return 0;
        }
        uint64_t key = msg.data_offset ^ (msg.target_offset << 1) ^
                       (msg.wr_id >> 6) ^
                       (static_cast<uint64_t>(msg.type) << 48);
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ULL;
        key ^= key >> 33;
        return static_cast<size_t>(key % rpc_worker_count);
    }

    size_t rpc_task_worker_idx(
        const rdma::EC2PCDataReqBatchMessage &msg) const {
        if (rpc_worker_count <= 1) {
            return 0;
        }
        uint16_t span_count =
            std::min<uint16_t>(msg.span_count, rdma::kEC2PCDataReqBatchMaxSpans);
        uint64_t first_offset = span_count == 0 ? 0 : msg.data_offsets[0];
        uint64_t first_wr_id = span_count == 0 ? 0 : msg.wr_ids[0];
        uint64_t key = first_offset ^ (first_wr_id >> 6) ^
                       (static_cast<uint64_t>(msg.ack_data_qp_idx) << 16) ^
                       (static_cast<uint64_t>(msg.parity_endpoint0) << 32) ^
                       (static_cast<uint64_t>(msg.parity_endpoint1) << 48);
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ULL;
        key ^= key >> 33;
        return static_cast<size_t>(key % rpc_worker_count);
    }

    size_t peer_probe_parity_batch_worker_idx(
        const PeerPayloadRecvSlot &slot,
        const rdma::ProbeParityBatchMessage &msg) const {
        if (rpc_worker_count <= 1) {
            return 0;
        }
        uint64_t first_target =
            msg.span_count == 0 ? 0 : msg.target_offsets[0];
        uint16_t scheduling_flags =
            static_cast<uint16_t>(msg.flags & ~rdma::kEC2PCFlagCompactReq);
        uint64_t key = first_target ^ (msg.wr_id << 1) ^
                       (static_cast<uint64_t>(slot.peer_idx) << 16) ^
                       (static_cast<uint64_t>(slot.shard_idx) << 32) ^
                       (static_cast<uint64_t>(scheduling_flags) << 48);
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ULL;
        key ^= key >> 33;
        return static_cast<size_t>(key % rpc_worker_count);
    }

    bool should_queue_peer_probe_parity_batch() const {
        return config.is_ec_2pc_mode() && !rpc_workers.empty();
    }

    bool try_pop_overflow_task(RpcTask &task) {
        std::lock_guard<std::mutex> lock(rpc_overflow_mutex);
        if (rpc_overflow_queue.empty()) {
            return false;
        }
        task = std::move(rpc_overflow_queue.front());
        rpc_overflow_queue.pop_front();
        rpc_overflow_depth.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    void worker_loop(size_t worker_idx) {
        pin_current_thread(worker_thread_slot(worker_idx), "worker", worker_idx);
        uint64_t last_service_end_ns = 0;
        std::array<RpcTask, kWorkerTaskBatchPopLimit> local_tasks{};
        size_t local_task_idx = 0;
        size_t local_task_count = 0;
        static constexpr size_t kWorkerRuntimePairFlushBudget = 2;
        static constexpr size_t kWorkerRuntimePairIdleFlushBudget = 1;
        while (true) {
            RpcTask task{};
            bool has_task = false;
            if (local_task_idx < local_task_count) {
                task = std::move(local_tasks[local_task_idx++]);
                has_task = true;
            } else {
                local_task_idx = 0;
                local_task_count = 0;
                auto worker_lock_begin = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(rpc_task_mutexes[worker_idx]);
                auto worker_lock_acquired = std::chrono::steady_clock::now();
                ctr_worker_dequeue_lock_wait_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            worker_lock_acquired - worker_lock_begin)
                            .count()),
                    std::memory_order_relaxed);
                ctr_worker_dequeue_lock_wait_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                if (!rpc_task_queues[worker_idx].empty()) {
                    size_t queue_depth_before = rpc_task_queues[worker_idx].size();
                    local_task_count = std::min(kWorkerTaskBatchPopLimit,
                                                queue_depth_before);
                    for (size_t i = 0; i < local_task_count; i++) {
                        local_tasks[i] = std::move(rpc_task_queues[worker_idx].front());
                        rpc_task_queues[worker_idx].pop_front();
                    }
                    task = std::move(local_tasks[0]);
                    local_task_idx = 1;
                    has_task = true;
                }
            }
            if (!has_task) {
                auto idle_flush_begin = std::chrono::steady_clock::now();
                size_t idle_flushed = flush_runtime_parity_batch_pair_pending_for_worker(
                    worker_idx, kWorkerRuntimePairIdleFlushBudget);
                auto idle_flush_end = std::chrono::steady_clock::now();
                if (idle_flushed != 0) {
                    ctr_worker_idle_flush_ns.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                idle_flush_end - idle_flush_begin)
                                .count()),
                        std::memory_order_relaxed);
                    ctr_worker_idle_flush_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    continue;
                }
                if (!try_pop_overflow_task(task)) {
                    if (rpc_worker_stop.load(std::memory_order_acquire)) {
                        if (rpc_overflow_depth.load(std::memory_order_relaxed) == 0) {
                            return;
                        }
                    }
                    continue;
                }
                // Overflow tasks may be stolen by a different worker; rebind the
                // execution context so per-worker shards/QPs follow the worker
                // that actually runs the task.
                task.worker_idx = worker_idx;
                has_task = true;
            }
            uint64_t dequeue_ns = steady_clock_now_ns();
            if (task.enqueue_time_ns != 0 && dequeue_ns >= task.enqueue_time_ns) {
                ctr_worker_queue_wait_ns.fetch_add(dequeue_ns - task.enqueue_time_ns,
                                                   std::memory_order_relaxed);
            }
            if (task.kind == RpcTaskKind::ClientBatchRpc &&
                task.recv_wc_time_ns != 0 && dequeue_ns >= task.recv_wc_time_ns) {
                ctr_client_batch_recv_to_dequeue_ns.fetch_add(
                    dequeue_ns - task.recv_wc_time_ns,
                    std::memory_order_relaxed);
                ctr_client_batch_recv_to_dequeue_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
            uint64_t worker_task_seq =
                ctr_worker_task_dequeued.fetch_add(1, std::memory_order_relaxed) +
                1;
            if (last_service_end_ns != 0 && dequeue_ns >= last_service_end_ns) {
                uint64_t gap_ns = dequeue_ns - last_service_end_ns;
                ctr_worker_intertask_gap_ns.fetch_add(
                    gap_ns, std::memory_order_relaxed);
                ctr_worker_intertask_gap_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (worker_deq_counts && worker_idx < rpc_worker_count) {
                worker_deq_counts[worker_idx].fetch_add(1,
                                                       std::memory_order_relaxed);
            }
            auto service_begin = std::chrono::steady_clock::now();
            if (task.kind == RpcTaskKind::PeerProbeParityBatch) {
                ctr_peer_probe_batch_task_dequeued.fetch_add(
                    1, std::memory_order_relaxed);
                auto parity_service_begin = std::chrono::steady_clock::now();
                if (task.peer_payload_slot == nullptr) {
                    ERROR("server rpc worker: missing peer payload recv slot");
                }
                const rdma::ProbeParityBatchMessage *parity_msg =
                    task.peer_payload_slot->probe_batch_msg();
                bool is_probe_batch =
                    (parity_msg->flags & rdma::kEC2PCFlagProbeBatch) != 0;
                bool is_compact =
                    (parity_msg->reserved0 & rdma::kEC2PCFlagCompactReq) != 0;
                if (!is_probe_batch && task.enqueue_time_ns != 0 &&
                    dequeue_ns >= task.enqueue_time_ns) {
                    uint64_t queue_ns = dequeue_ns - task.enqueue_time_ns;
                    ctr_runtime_parity_batch_enqueue_to_dequeue_ns.fetch_add(
                        queue_ns, std::memory_order_relaxed);
                    ctr_runtime_parity_batch_enqueue_to_dequeue_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    update_peak(
                        ctr_runtime_parity_batch_enqueue_to_dequeue_max_ns,
                        queue_ns);
                    if (is_compact) {
                        ctr_compact_peer_enqueue_to_dequeue_ns.fetch_add(
                            queue_ns, std::memory_order_relaxed);
                        ctr_compact_peer_enqueue_to_dequeue_cnt.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
                handle_peer_probe_parity_batch_message(
                    *parity_msg,
                    static_cast<size_t>(task.peer_payload_peer_idx),
                    task.recv_wc_time_ns, task.peer_payload_shard_idx);
                if (task.peer_payload_slot != nullptr) {
                    auto repost_begin = std::chrono::steady_clock::now();
                    post_peer_payload_recv_slot(*task.peer_payload_slot);
                    auto repost_end = std::chrono::steady_clock::now();
                    uint64_t repost_ns = elapsed_ns(repost_begin, repost_end);
                    ctr_peer_payload_repost_ns.fetch_add(
                        repost_ns, std::memory_order_relaxed);
                    ctr_peer_payload_repost_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    if (is_compact) {
                        ctr_compact_peer_payload_repost_ns.fetch_add(
                            repost_ns, std::memory_order_relaxed);
                        ctr_compact_peer_payload_repost_cnt.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
                auto service_end = std::chrono::steady_clock::now();
                uint64_t service_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        service_end - parity_service_begin)
                        .count());
                ctr_probe_parity_batch_repost_ns.fetch_add(
                    0,
                    std::memory_order_relaxed);
                ctr_probe_parity_batch_repost_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                ctr_probe_parity_batch_worker_service_ns.fetch_add(
                    service_ns,
                    std::memory_order_relaxed);
                ctr_probe_parity_batch_worker_service_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                if (!is_probe_batch) {
                    ctr_runtime_parity_batch_worker_service_ns.fetch_add(
                        service_ns, std::memory_order_relaxed);
                    ctr_runtime_parity_batch_worker_service_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    update_peak(ctr_runtime_parity_batch_worker_service_max_ns,
                                service_ns);
                }
                if (is_compact) {
                    ctr_compact_peer_worker_service_ns.fetch_add(
                        service_ns, std::memory_order_relaxed);
                    ctr_compact_peer_worker_service_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                }
            } else if (task.kind == RpcTaskKind::ClientBatchRpc) {
                ctr_client_batch_task_dequeued.fetch_add(
                    1, std::memory_order_relaxed);
                auto client_batch_service_begin = std::chrono::steady_clock::now();
                handle_batch_rpc_message(*task.batch_slot, worker_idx,
                                         task.recv_wc_time_ns);
                auto client_batch_service_end = std::chrono::steady_clock::now();
                ctr_client_batch_worker_service_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            client_batch_service_end - client_batch_service_begin)
                            .count()),
                    std::memory_order_relaxed);
                ctr_client_batch_worker_service_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            } else {
                ctr_client_rpc_task_dequeued.fetch_add(
                    1, std::memory_order_relaxed);
                if (task.msg.type == rdma::EC2PC_MSG_DATA_REQ) {
                    ctr_client_rpc_data_req_task_dequeued.fetch_add(
                        1, std::memory_order_relaxed);
                } else if (task.msg.type == rdma::EC2PC_MSG_PROBE_REQ) {
                    ctr_client_rpc_probe_req_task_dequeued.fetch_add(
                        1, std::memory_order_relaxed);
                    if (ec2pc_preflight_debug_enabled()) {
                        static std::atomic<uint64_t> seen{0};
                        uint64_t seq =
                            seen.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (seq <= 16) {
                            std::cerr << "[ec2pc-preflight] server worker probe"
                                      << " server=" << local_server_index
                                      << " worker=" << worker_idx
                                      << " data_qp=" << task.data_qp_local_idx
                                      << " wr_id=0x" << std::hex
                                      << task.msg.wr_id << std::dec
                                      << std::endl;
                        }
                    }
                }
                auto client_rpc_service_begin = std::chrono::steady_clock::now();
                handle_rpc_message(task.msg, task.data_qp_local_idx,
                                   worker_idx);
                auto client_rpc_service_end = std::chrono::steady_clock::now();
                ctr_client_rpc_worker_service_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            client_rpc_service_end - client_rpc_service_begin)
                            .count()),
                    std::memory_order_relaxed);
                ctr_client_rpc_worker_service_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
            auto service_end = std::chrono::steady_clock::now();
            ctr_worker_service_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        service_end - service_begin)
                        .count()),
                std::memory_order_relaxed);
            auto post_task_flush_begin = std::chrono::steady_clock::now();
            size_t post_task_flushed =
                flush_runtime_parity_batch_pair_pending_for_worker(
                    worker_idx, kWorkerRuntimePairFlushBudget);
            auto post_task_flush_end = std::chrono::steady_clock::now();
            if (post_task_flushed != 0) {
                ctr_worker_post_task_flush_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            post_task_flush_end - post_task_flush_begin)
                            .count()),
                    std::memory_order_relaxed);
                ctr_worker_post_task_flush_cnt.fetch_add(
                    1, std::memory_order_relaxed);
            }
            last_service_end_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    service_end.time_since_epoch())
                    .count());
            if (worker_task_seq == 1 || (worker_task_seq % 5000) == 0) {
                auto post_service_probe_begin = std::chrono::steady_clock::now();
                size_t queue_depth_after_service = 0;
                {
                    auto probe_lock_begin = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lock(rpc_task_mutexes[worker_idx]);
                    auto probe_lock_acquired = std::chrono::steady_clock::now();
                    ctr_worker_post_service_probe_lock_wait_ns.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                probe_lock_acquired - probe_lock_begin)
                                .count()),
                        std::memory_order_relaxed);
                    ctr_worker_post_service_probe_lock_wait_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                    queue_depth_after_service = rpc_task_queues[worker_idx].size();
                }
                auto post_service_probe_end = std::chrono::steady_clock::now();
                ctr_worker_post_service_probe_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            post_service_probe_end - post_service_probe_begin)
                            .count()),
                    std::memory_order_relaxed);
                ctr_worker_post_service_probe_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                ctr_worker_post_service_queue_depth_sum.fetch_add(
                    static_cast<uint64_t>(queue_depth_after_service),
                    std::memory_order_relaxed);
                if (queue_depth_after_service == 0) {
                    ctr_worker_post_service_queue_empty.fetch_add(
                        1, std::memory_order_relaxed);
                    if (rpc_overflow_depth.load(std::memory_order_relaxed) != 0) {
                        ctr_worker_post_service_overflow_nonempty.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                } else {
                    ctr_worker_post_service_queue_nonempty.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        }
    }

    void init_rpc_workers() {
        stop_rpc_workers();
        rpc_worker_count = effective_rpc_worker_count();
        rpc_worker_stop.store(false, std::memory_order_release);
        size_t endpoint_count = config.all_server_port_list.empty()
                                    ? std::max<size_t>(
                                          1, static_cast<size_t>(config.server_count))
                                    : config.all_server_port_list.size();
        runtime_parity_batches.assign(
            endpoint_count * std::max<size_t>(1, rpc_worker_count),
            RuntimeParityBatchBuilder{});
        size_t worker_count = std::max<size_t>(1, rpc_worker_count);
        runtime_parity_batch_pair_pending_slot_ids_by_worker.assign(
            worker_count, {});
        runtime_parity_batch_pair_storage_by_worker.clear();
        runtime_parity_batch_pair_storage_by_worker.resize(worker_count);
        for (size_t w = 0; w < worker_count; w++) {
            auto &storage = runtime_parity_batch_pair_storage_by_worker[w];
            storage.capacity = kRuntimeParityPendingStorageSlotsPerWorker;
            storage.free_bitmap_words = (storage.capacity + 63) / 64;
            storage.alloc_hint = 0;
            storage.slots.reset(
                new RuntimeParityBatchPairStorageSlot[storage.capacity]);
            storage.free_bitmap.reset(new uint64_t[storage.free_bitmap_words]);
            for (size_t word_idx = 0; word_idx < storage.free_bitmap_words;
                 word_idx++) {
                storage.free_bitmap[word_idx] = ~0ull;
            }
            size_t used_bits_last_word = storage.capacity % 64;
            if (used_bits_last_word != 0) {
                storage.free_bitmap[storage.free_bitmap_words - 1] =
                    (1ull << used_bits_last_word) - 1ull;
            }
        }
        {
            std::lock_guard<std::mutex> lock(rpc_overflow_mutex);
            rpc_overflow_queue.clear();
        }
        rpc_overflow_depth.store(0, std::memory_order_relaxed);
        rpc_task_queues.reset(new std::deque<RpcTask>[rpc_worker_count]);
        rpc_task_mutexes.reset(new std::mutex[rpc_worker_count]);
        rpc_task_cvs.reset(new std::condition_variable[rpc_worker_count]);
        worker_deq_counts.reset(new std::atomic<uint64_t>[rpc_worker_count]);
        last_worker_deq_counts.assign(rpc_worker_count, 0);
        for (size_t w = 0; w < rpc_worker_count; w++) {
            worker_deq_counts[w].store(0, std::memory_order_relaxed);
        }
        rpc_workers.reserve(rpc_worker_count);
        for (size_t w = 0; w < rpc_worker_count; w++) {
            rpc_workers.emplace_back([this, w] { worker_loop(w); });
        }
    }

    void stop_rpc_workers() {
        if (rpc_worker_count <= 1 && rpc_workers.empty()) {
            rpc_task_queues.reset();
            rpc_task_mutexes.reset();
            rpc_task_cvs.reset();
            worker_deq_counts.reset();
            last_worker_deq_counts.clear();
            return;
        }
        rpc_worker_stop.store(true, std::memory_order_release);
        if (rpc_task_cvs) {
            for (size_t w = 0; w < rpc_worker_count; w++) {
                rpc_task_cvs[w].notify_all();
            }
        }
        for (auto &th : rpc_workers) {
            if (th.joinable()) {
                th.join();
            }
        }
        rpc_workers.clear();
        rpc_task_queues.reset();
        rpc_task_mutexes.reset();
        rpc_task_cvs.reset();
        worker_deq_counts.reset();
        last_worker_deq_counts.clear();
        runtime_parity_batch_pair_pending_slot_ids_by_worker.clear();
        runtime_parity_batch_pair_storage_by_worker.clear();
        rpc_worker_count = 1;
    }

    void dispatch_rpc_task(const rdma::EC2PCRpcMessage &msg,
                           size_t data_qp_local_idx) {
        if (rpc_workers.empty()) {
            ctr_dispatch_inline.fetch_add(1, std::memory_order_relaxed);
            handle_rpc_message_inline_profiled(msg, data_qp_local_idx, 0);
            return;
        }
        size_t worker_idx = rpc_task_worker_idx(msg);
        uint64_t enqueue_time_ns = steady_clock_now_ns();
        bool enqueued = false;
        {
            std::lock_guard<std::mutex> lock(rpc_task_mutexes[worker_idx]);
            if (rpc_task_queues[worker_idx].size() < kRpcTaskQueueLimit) {
                RpcTask task{};
                task.kind = RpcTaskKind::ClientRpc;
                task.msg = msg;
                task.data_qp_local_idx = data_qp_local_idx;
                task.worker_idx = worker_idx;
                task.enqueue_time_ns = enqueue_time_ns;
                rpc_task_queues[worker_idx].push_back(std::move(task));
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
                    RpcTask task{};
                    task.kind = RpcTaskKind::ClientRpc;
                    task.msg = msg;
                    task.data_qp_local_idx = data_qp_local_idx;
                    task.worker_idx = w;
                    task.enqueue_time_ns = enqueue_time_ns;
                    rpc_task_queues[w].push_back(std::move(task));
                    worker_idx = w;
                    enqueued = true;
                    break;
                }
            }
        }
        if (enqueued) {
            ctr_dispatch_enqueued.fetch_add(1, std::memory_order_relaxed);
            if (ec2pc_preflight_debug_enabled() &&
                msg.type == rdma::EC2PC_MSG_PROBE_REQ) {
                static std::atomic<uint64_t> seen{0};
                uint64_t seq =
                    seen.fetch_add(1, std::memory_order_relaxed) + 1;
                if (seq <= 16) {
                    std::cerr << "[ec2pc-preflight] server dispatch probe"
                              << " server=" << local_server_index
                              << " worker=" << worker_idx
                              << " data_qp=" << data_qp_local_idx
                              << " wr_id=0x" << std::hex << msg.wr_id
                              << std::dec << std::endl;
                }
            }
            rpc_task_cvs[worker_idx].notify_one();
            return;
        }
        // When all worker queues are full, push into overflow queue so poller
        // never executes business logic inline (which can block CQ progress).
        ctr_dispatch_queue_full.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(rpc_overflow_mutex);
            RpcTask task{};
            task.kind = RpcTaskKind::ClientRpc;
            task.msg = msg;
            task.data_qp_local_idx = data_qp_local_idx;
            task.worker_idx = worker_idx;
            task.enqueue_time_ns = enqueue_time_ns;
            rpc_overflow_queue.push_back(std::move(task));
            rpc_overflow_depth.fetch_add(1, std::memory_order_relaxed);
        }
        ctr_dispatch_overflow.fetch_add(1, std::memory_order_relaxed);
        for (size_t w = 0; w < rpc_worker_count; w++) {
            rpc_task_cvs[w].notify_one();
        }
    }

    void dispatch_peer_probe_parity_batch_task(PeerPayloadRecvSlot &slot,
                                               uint64_t recv_wc_time_ns);

    void dispatch_batch_rpc_task(BatchRpcRecvSlot &slot,
                                 uint32_t recv_byte_len = 0);

    void handle_rpc_message_inline_profiled(const rdma::EC2PCRpcMessage &msg,
                                            size_t data_qp_local_idx,
                                            size_t worker_idx) {
        auto inline_begin = std::chrono::steady_clock::now();
        handle_rpc_message(msg, data_qp_local_idx, worker_idx);
        auto inline_end = std::chrono::steady_clock::now();
        uint64_t inline_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(inline_end -
                                                                  inline_begin)
                .count());
        ctr_dispatch_inline_ns.fetch_add(inline_ns, std::memory_order_relaxed);
        update_peak(ctr_dispatch_inline_max_ns, inline_ns);
        if (inline_ns >= 100000) {  // >=100us
            ctr_dispatch_inline_slow.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void release_rpc_resources() {
        stop_rpc_workers();
        if (rpc_mr != nullptr) {
            ibv_dereg_mr(rpc_mr);
            rpc_mr = nullptr;
        }
        if (batch_rpc_mr != nullptr) {
            ibv_dereg_mr(batch_rpc_mr);
            batch_rpc_mr = nullptr;
        }
        if (rpc_mem != nullptr) {
            std::free(rpc_mem);
            rpc_mem = nullptr;
        }
        rpc_queues.clear();
        if (batch_rpc_mem != nullptr) {
            std::free(batch_rpc_mem);
            batch_rpc_mem = nullptr;
        }
        batch_rpc_msg_free_pool.clear();
        batch_rpc_msg_buffer_count = 0;
        recv_credits.clear();
        peer_payload_recv_credits.reset();
        peer_payload_recv_credit_count = 0;
        peer_payload_recv_depth = 0;
        recv_qp_wc_counts.clear();
        last_recv_qp_wc_counts.clear();
        recv_qp_first_seen.clear();
        pending_send_queues.clear();
        pending_send_meta_queues.clear();
        inflight_send_slots.clear();
        pending_send_queue_mutexes.reset();
        inflight_send_slot_mutexes.reset();
        pending_send_total.store(0, std::memory_order_relaxed);
    }

    void init_rpc_resources() {
        release_rpc_resources();
        if (qps.size() <= 1) {
            return;
        }
        size_t data_qp_count = client_data_qp_count();
        rpc_queue_depth = configured_rpc_queue_depth();
        size_t recv_count = data_qp_count * rpc_queue_depth;
        size_t send_count = data_qp_count * rpc_queue_depth;
        size_t recv_bytes = recv_count * sizeof(RpcRecvSlot);
        size_t send_bytes = send_count * sizeof(RpcSendSlot);
        size_t total_bytes = recv_bytes + send_bytes;
        size_t aligned_bytes = ((total_bytes + 63) / 64) * 64;
        rpc_mem = std::aligned_alloc(64, aligned_bytes);
        if (rpc_mem == nullptr) {
            ERROR("server rpc_mem alloc failed");
        }
        std::memset(rpc_mem, 0, total_bytes);
        rpc_mr = ibv_reg_mr(pd.protection_domain, rpc_mem, total_bytes,
                            IBV_ACCESS_LOCAL_WRITE);
        if (rpc_mr == nullptr) {
            ERROR("server rpc_mr register failed");
        }

        auto *recv_base = reinterpret_cast<RpcRecvSlot *>(rpc_mem);
        auto *send_base = reinterpret_cast<RpcSendSlot *>(
            reinterpret_cast<uint8_t *>(rpc_mem) + recv_bytes);

        size_t batch_msg_count = recv_count * 2;
        size_t batch_msg_bytes = batch_msg_count * sizeof(BatchRpcMessageBuffer);
        size_t batch_aligned_bytes = ((batch_msg_bytes + 63) / 64) * 64;
        batch_rpc_mem = std::aligned_alloc(64, batch_aligned_bytes);
        if (batch_rpc_mem == nullptr) {
            ERROR("server batch_rpc_mem alloc failed");
        }
        std::memset(batch_rpc_mem, 0, batch_msg_bytes);
        batch_rpc_mr = ibv_reg_mr(pd.protection_domain, batch_rpc_mem,
                                  batch_msg_bytes, IBV_ACCESS_LOCAL_WRITE);
        if (batch_rpc_mr == nullptr) {
            ERROR("server batch_rpc_mr register failed");
        }
        auto *batch_msg_base =
            reinterpret_cast<BatchRpcMessageBuffer *>(batch_rpc_mem);
        batch_rpc_msg_free_pool.clear();
        batch_rpc_msg_free_pool.reserve(batch_msg_count);
        batch_rpc_msg_buffer_count = batch_msg_count;
        for (size_t i = recv_count; i < batch_msg_count; i++) {
            batch_rpc_msg_free_pool.push_back(&batch_msg_base[i]);
        }

        rpc_queues.assign(data_qp_count, RpcQueue{});
        recv_credits.assign(data_qp_count,
                            static_cast<uint16_t>(rpc_queue_depth));
        recv_qp_wc_counts.assign(data_qp_count, 0);
        last_recv_qp_wc_counts.assign(data_qp_count, 0);
        recv_qp_first_seen.assign(data_qp_count, 0);
        pending_send_queues.assign(data_qp_count,
                                  std::deque<rdma::EC2PCRpcMessage>{});
        pending_send_meta_queues.assign(
            data_qp_count, std::deque<PendingRpcSendMeta>{});
        inflight_send_slots.assign(data_qp_count, std::deque<RpcSendSlot *>{});
        pending_send_queue_mutexes.reset(new std::mutex[data_qp_count]);
        inflight_send_slot_mutexes.reset(new std::mutex[data_qp_count]);
        reset_rpc_runtime_counters();
        if (all_counter_arrays_ready(runtime_parity_batch_ready_by_qp,
                                     runtime_parity_batch_outstanding_by_qp)) {
            size_t qp_metric_count =
                peer_endpoint_count * std::max<size_t>(1, peer_payload_qp_count);
            reset_counter_arrays(qp_metric_count, 0,
                                 runtime_parity_batch_ready_by_qp,
                                 runtime_parity_batch_outstanding_by_qp);
        }
        reset_rpc_post_init_counters();
        for (size_t q = 0; q < data_qp_count; q++) {
            auto &queue = rpc_queues[q];
            queue.recv_slots = recv_base + q * rpc_queue_depth;
            queue.send_slots = send_base + q * rpc_queue_depth;
            queue.send_head = 0;
            for (size_t i = 0; i < rpc_queue_depth; i++) {
                auto &recv_slot = queue.recv_slots[i];
                new (&recv_slot) RpcRecvSlot();
                recv_slot.magic = 0xEC2C1001u;
                recv_slot.qp_idx = static_cast<uint16_t>(q + 1);
                recv_slot.msg_buffer =
                    &batch_msg_base[q * rpc_queue_depth + i];
                post_recv_slot(recv_slot);

                auto &send_slot = queue.send_slots[i];
                new (&send_slot) RpcSendSlot();
                send_slot.magic = 0xEC2C1002u;
                send_slot.qp_idx = static_cast<uint16_t>(q + 1);
                send_slot.in_use.store(false, std::memory_order_relaxed);
            }
        }
        if (ec2pc_preflight_debug_enabled()) {
            std::cerr << "[ec2pc-preflight] server recv posted"
                      << " server=" << local_server_index
                      << " port=" << config.server_port
                      << " data_qp_count=" << data_qp_count
                      << " rpc_queue_depth=" << rpc_queue_depth
                      << " data_cq_count=" << data_cq_count
                      << " first_data_qpn="
                      << (qps.size() > 1 ? qps[1].queue_pair->qp_num : 0)
                      << " last_data_qpn="
                      << (qps.size() > 1 ? qps.back().queue_pair->qp_num : 0)
                      << std::endl;
            for (size_t q = 0; q < std::min<size_t>(4, data_qp_count); q++) {
                std::cerr << "[ec2pc-preflight] server data qp"
                          << " port=" << config.server_port
                          << " data_qp=" << q
                          << " qpn=" << qps[q + 1].queue_pair->qp_num
                          << std::endl;
            }
        }
        init_rpc_workers();
    }

    void post_recv_slot(RpcRecvSlot &slot);

    void post_peer_recv_slot(PeerRpcRecvSlot &slot);
    void post_peer_payload_recv_slot(PeerPayloadRecvSlot &slot);

    void release_peer_send_slot(PeerRpcSendSlot &slot);

    size_t reclaim_inflight_peer_send_slots_upto(size_t peer_idx,
                                                 PeerRpcSendSlot *completed_slot);

    size_t reclaim_inflight_send_slots_upto(size_t data_qp_local_idx,
                                            RpcSendSlot *completed_slot);

    static constexpr size_t kRpcPostSendBatch = 32;
    static constexpr size_t kRpcSendSignalStride = 4;
    static constexpr size_t kPeerPostSendBatch = 32;
    static constexpr size_t kPeerSendSignalStride = 4;
    static constexpr uint64_t kFlushRoundBudgetNs = 500000; // 0.5 ms
    static constexpr uint64_t kRuntimeParityBatchFlushNs = 50000;

    size_t try_post_send_rpc_batch_locked(
        size_t data_qp_local_idx, std::deque<rdma::EC2PCRpcMessage> &dq,
        std::deque<PendingRpcSendMeta> &meta_q,
        size_t max_posts);

    bool enqueue_rpc_send(size_t data_qp_local_idx,
                          const rdma::EC2PCRpcMessage &msg,
                          uint64_t final_ack_ready_time_ns = 0,
                          uint64_t server_recv_start_time_ns = 0);

    size_t flush_pending_sends(size_t max_posts = SIZE_MAX,
                               size_t first_q = 0, size_t q_stride = 1);

    void handle_rpc_message(const rdma::EC2PCRpcMessage &msg,
                            size_t data_qp_local_idx,
                            size_t worker_idx);
    void handle_batch_rpc_message(BatchRpcRecvSlot &slot, size_t worker_idx,
                                  uint64_t recv_wc_time_ns = 0);
    void handle_probe_batch(const rdma::EC2PCRpcMessage *msgs,
                            const size_t *data_qp_local_idxs,
                            size_t count,
                            size_t worker_idx);

    void handle_peer_rpc_message(const rdma::EC2PCRpcMessage &msg,
                                 size_t peer_idx,
                                 uint64_t recv_wc_time_ns = 0);
    void handle_peer_probe_parity_batch_message(
        const rdma::ProbeParityBatchMessage &msg, size_t peer_idx,
        uint64_t recv_wc_time_ns = 0,
        uint16_t recv_shard_idx = std::numeric_limits<uint16_t>::max());

    size_t try_post_peer_send_batch_locked(
        size_t peer_idx, std::deque<rdma::EC2PCRpcMessage> &dq,
        size_t max_posts);

    bool enqueue_peer_rpc_send(size_t peer_idx,
                               const rdma::EC2PCRpcMessage &msg);

    size_t flush_pending_peer_sends(size_t max_posts = SIZE_MAX,
                                    size_t first_peer = 0,
                                    size_t peer_stride = 1);

    bool post_peer_ack_message(size_t peer_idx,
                               const rdma::EC2PCRpcMessage &msg);

    bool enqueue_peer_ack_send(size_t peer_idx,
                               const rdma::EC2PCRpcMessage &msg);

    size_t flush_pending_peer_acks(size_t max_posts = SIZE_MAX,
                                   size_t first_peer = 0,
                                   size_t peer_stride = 1);

    size_t runtime_parity_batch_index(size_t peer_idx, size_t worker_idx) const;
    bool reserve_runtime_parity_batch_payload(size_t peer_idx, size_t worker_idx,
                                              uint64_t wr_id,
                                              uint16_t parity_idx,
                                              uint64_t target_offset,
                                              uint8_t **payload_out,
                                              bool *flush_after_encode);
    bool append_runtime_parity_batch(size_t peer_idx, size_t worker_idx,
                                     uint64_t wr_id, uint16_t parity_idx,
                                     uint64_t target_offset,
                                     const uint8_t *payload);
    bool flush_runtime_parity_batch(size_t peer_idx, size_t worker_idx,
                                    bool force);
    size_t flush_all_runtime_parity_batches_for_worker(size_t worker_idx,
                                                       bool force);

    bool enqueue_peer_parity_apply_lane(size_t peer_idx, size_t worker_idx,
                                        const rdma::EC2PCRpcMessage &msg);

    size_t flush_peer_parity_apply_lane(size_t peer_idx, size_t worker_idx);
    size_t flush_all_peer_parity_apply_lanes(size_t max_posts = SIZE_MAX);

    bool post_peer_rpc_message(size_t peer_idx,
                               const rdma::EC2PCRpcMessage &msg);

    bool post_peer_parity_apply_lane(size_t peer_idx, size_t worker_idx,
                                     const rdma::EC2PCRpcMessage &msg);
    bool post_probe_parity_batch_message(
        size_t peer_idx, const rdma::ProbeParityBatchMessage &msg,
        size_t preferred_shard_idx = std::numeric_limits<size_t>::max());
    bool post_probe_parity_batch_message_external(
        size_t peer_idx, rdma::ProbeParityBatchMessage *msg, uint32_t msg_lkey,
        BatchRpcRecvSlot *batch_owner, uint64_t batch_owner_generation,
        size_t preferred_shard_idx = std::numeric_limits<size_t>::max());
    bool post_probe_parity_batch_message_pair_external(
        size_t peer_idx0, rdma::ProbeParityBatchMessage *msg0,
        size_t peer_idx1, rdma::ProbeParityBatchMessage *msg1,
        uint32_t msg_lkey, BatchRpcRecvSlot *batch_owner,
        uint64_t batch_owner_generation,
        size_t preferred_shard_idx0 = std::numeric_limits<size_t>::max(),
        size_t preferred_shard_idx1 = std::numeric_limits<size_t>::max());
    bool enqueue_runtime_parity_batch_pair_pending(
        size_t peer_idx0, const rdma::ProbeParityBatchMessage &msg0,
        size_t preferred_shard_idx0, size_t peer_idx1,
        const rdma::ProbeParityBatchMessage &msg1,
        size_t preferred_shard_idx1);
    bool enqueue_runtime_parity_batch_pair_pending_for_worker(
        size_t worker_idx, size_t peer_idx0,
        const rdma::ProbeParityBatchMessage &msg0, size_t preferred_shard_idx0,
        size_t peer_idx1, const rdma::ProbeParityBatchMessage &msg1,
        size_t preferred_shard_idx1);
    size_t flush_runtime_parity_batch_pair_pending(size_t max_jobs);
    size_t flush_runtime_parity_batch_pair_pending_for_worker(size_t worker_idx,
                                                              size_t max_jobs);
    bool try_post_runtime_parity_batch_pair_pending_item(
        const RuntimeParityBatchPairPendingItem &item);
    bool acquire_runtime_parity_batch_pair_storage_slot(
        size_t worker_idx, uint32_t &slot_idx_out,
        RuntimeParityBatchPairStorageSlot *&slot_out);
    void release_runtime_parity_batch_pair_storage_slot(size_t worker_idx,
                                                        uint32_t slot_idx);
    RuntimeParityBatchPairStorageSlot *get_runtime_parity_batch_pair_storage_slot(
        size_t worker_idx, uint32_t slot_idx);
    bool try_post_runtime_parity_batch_pair_storage_slot(size_t worker_idx,
                                                         uint32_t slot_idx);
    ProbeParityBatchSendSlot *reserve_probe_parity_batch_send_slot(
        size_t peer_idx, bool record_runtime_fail = false,
        size_t preferred_shard_idx = std::numeric_limits<size_t>::max());
    void release_reserved_probe_parity_batch_send_slot(
        ProbeParityBatchSendSlot &slot);
    bool post_reserved_probe_parity_batch_send_slot(
        ProbeParityBatchSendSlot &slot);
    bool try_post_probe_parity_batch_pair_pending_item(
        ProbeParityBatchPairPendingItem &item);
    size_t flush_probe_parity_batch_pair_pending(size_t max_jobs);
    bool enqueue_probe_parity_batch_pending(
        size_t peer_idx, const rdma::ProbeParityBatchMessage &msg,
        size_t preferred_shard_idx);
    bool enqueue_probe_parity_batch_pending_external(
        size_t peer_idx, rdma::ProbeParityBatchMessage *msg, uint32_t msg_lkey,
        BatchRpcRecvSlot *batch_owner, uint64_t batch_owner_generation,
        size_t preferred_shard_idx);
    size_t fill_probe_parity_batch_send_slots_from_pending(
        size_t peer_idx, size_t preferred_shard_idx, size_t max_fill);
    void prepare_probe_parity_slot(ProbeParityBatchSendSlot &slot);
    size_t kick_probe_parity_ready_slots(size_t peer_idx,
                                         size_t preferred_shard_idx,
                                         size_t max_posts);
    void enqueue_probe_parity_ready_slot(size_t queue_idx, uint16_t slot_index,
                                         uint64_t bind_seq,
                                         uint64_t ready_enqueue_time_ns = 0);
    bool try_dequeue_probe_parity_ready_slot(size_t queue_idx,
                                             ProbeParityBatchReadyItem &item);
    void publish_probe_parity_batch_send_slot(ProbeParityBatchSendSlot &slot);
    size_t flush_ready_probe_parity_batch_send_slots(
        size_t peer_idx, size_t preferred_shard_idx, size_t max_posts);
    // Complete one parity SEND only if the CQE still belongs to the current
    // recv-slot generation; this prevents stale completions from re-releasing
    // a reused BatchRpcRecvSlot.
    bool complete_probe_parity_batch_send_slot(ProbeParityBatchSendSlot &slot,
                                               uint64_t cqe_time_ns);
    void release_batch_rpc_recv_slot(BatchRpcRecvSlot &slot);
    BatchRpcMessageBuffer *try_acquire_batch_rpc_message_buffer();
    void release_batch_rpc_message_buffer(BatchRpcMessageBuffer *buffer);

    void record_probe_parity_batch_send_cqe(
        const ProbeParityBatchSendSlot &slot, uint64_t cqe_time_ns) {
        if (slot.post_time_ns == 0 || cqe_time_ns < slot.post_time_ns) {
            return;
        }
        uint64_t delta_ns = cqe_time_ns - slot.post_time_ns;
        const auto *msg = slot.active_msg();
        if (msg == nullptr) {
            return;
        }
        bool is_probe_batch =
            (msg->flags & rdma::kEC2PCFlagProbeBatch) != 0;
        bool is_compact =
            (msg->reserved0 & rdma::kEC2PCFlagCompactReq) != 0;
        size_t peer_idx = static_cast<size_t>(slot.peer_idx);
        size_t shard_idx = static_cast<size_t>(slot.shard_idx);
        size_t q_idx = peer_payload_qp_index(peer_idx, shard_idx);
        if (is_compact) {
            ctr_compact_peer_send_wc.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (!is_probe_batch) {
            ctr_runtime_parity_batch_post_to_send_cqe_ns.fetch_add(
                delta_ns, std::memory_order_relaxed);
            ctr_runtime_parity_batch_post_to_send_cqe_cnt.fetch_add(
                1, std::memory_order_relaxed);
            update_peak(ctr_runtime_parity_batch_post_to_send_cqe_max_ns,
                        delta_ns);
            if (runtime_parity_batch_outstanding_by_qp != nullptr &&
                q_idx < peer_endpoint_count *
                            std::max<size_t>(1, peer_payload_qp_count)) {
                uint64_t reclaim_begin_ns = steady_clock_now_ns();
                uint64_t cur_outstanding =
                    runtime_parity_batch_outstanding_by_qp[q_idx].load(
                        std::memory_order_relaxed);
                ctr_runtime_parity_outstanding_dec_before_sum.fetch_add(
                    cur_outstanding, std::memory_order_relaxed);
                ctr_runtime_parity_outstanding_dec_before_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                update_peak(ctr_runtime_parity_outstanding_dec_before_max,
                            cur_outstanding);
                while (cur_outstanding != 0 &&
                       !runtime_parity_batch_outstanding_by_qp[q_idx]
                            .compare_exchange_weak(
                                cur_outstanding, cur_outstanding - 1,
                                std::memory_order_relaxed,
                                std::memory_order_relaxed)) {
                }
                uint64_t reclaim_ns = steady_clock_now_ns() - reclaim_begin_ns;
                ctr_runtime_parity_send_cqe_reclaim_ns.fetch_add(
                    reclaim_ns, std::memory_order_relaxed);
                ctr_runtime_parity_send_cqe_reclaim_cnt.fetch_add(
                    1, std::memory_order_relaxed);
                update_peak(ctr_runtime_parity_send_cqe_reclaim_max_ns,
                            reclaim_ns);
                if (is_compact) {
                    ctr_compact_peer_send_cqe_reclaim_ns.fetch_add(
                        reclaim_ns, std::memory_order_relaxed);
                    ctr_compact_peer_send_cqe_reclaim_cnt.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
            return;
        }
        ctr_probe_parity_batch_post_to_send_cqe_ns.fetch_add(
            delta_ns, std::memory_order_relaxed);
        ctr_probe_parity_batch_post_to_send_cqe_cnt.fetch_add(
            1, std::memory_order_relaxed);
    }

    std::string peer_port_for_index(size_t idx) const {
        auto [_addr, base_port] = config.get_full_endpoint(idx);
        char *end = nullptr;
        long port = std::strtol(base_port.c_str(), &end, 10);
        if (end == nullptr || *end != '\0' || port <= 0) {
            ERROR("server peer port parse failed");
        }
        return std::to_string(port + kPeerPortOffset);
    }

    void connect_peer_servers() {
        peer_endpoint_count = config.all_server_addr_list.empty()
                                  ? static_cast<size_t>(config.server_count)
                                  : config.all_server_addr_list.size();
        local_server_index =
            peer_endpoint_count == 0
                ? 0
                : std::min(config.local_server_index, peer_endpoint_count - 1);
        peer_payload_qp_count = configured_peer_payload_qp_count();
        std::cout << "INFO: peer payload qp count server="
                  << parse_server_index_from_port()
                  << " env_force="
                  << (std::getenv("FARLIB_FORCE_PEER_PAYLOAD_QP_COUNT") != nullptr
                          ? std::getenv("FARLIB_FORCE_PEER_PAYLOAD_QP_COUNT")
                          : "<unset>")
                  << " cfg_override="
                  << config.peer_payload_qp_count_override
                  << " runtime_rr="
                  << (config.peer_payload_runtime_rr_shard ? 1 : 0)
                  << " effective_workers="
                  << effective_rpc_worker_count()
                  << " peer_payload_qp_count="
                  << peer_payload_qp_count << std::endl;
        peer_qps.clear();
        peer_payload_qps.clear();
        peer_cq.reset();
        peer_payload_cqs.clear();
        peer_payload_cq_poll_mutexes.reset();
        if (peer_endpoint_count <= 1) {
            return;
        }

        peer_cq = std::make_unique<CompleteQueue>(ctx, config, 0);
        size_t payload_cq_count = dedicated_peer_payload_cq_per_qp()
                                      ? peer_endpoint_count *
                                            peer_payload_qp_count
                                      : (config.peer_payload_cq_per_peer
                                             ? peer_endpoint_count
                                             : 1);
        peer_payload_cqs.clear();
        peer_payload_cqs.reserve(payload_cq_count);
        peer_payload_cq_poll_mutexes.reset(new std::mutex[payload_cq_count]);
        for (size_t cq_idx = 0; cq_idx < payload_cq_count; cq_idx++) {
            peer_payload_cqs.push_back(std::make_unique<CompleteQueue>(
                ctx, config, controlled_peer_payload_comp_vector()));
        }
        peer_qps.resize(peer_endpoint_count);
        peer_payload_qps.resize(peer_endpoint_count * peer_payload_qp_count);
        for (size_t peer_idx = 0; peer_idx < peer_endpoint_count; peer_idx++) {
            if (peer_idx == local_server_index) {
                continue;
            }
            peer_qps[peer_idx].init(ctx, *peer_cq, pd, config);
            for (size_t shard_idx = 0; shard_idx < peer_payload_qp_count;
                 shard_idx++) {
                auto *payload_cq = peer_payload_cq_for(peer_idx, shard_idx);
                if (payload_cq == nullptr) {
                    ERROR("server peer payload cq missing");
                }
                peer_payload_qps[peer_payload_qp_index(peer_idx, shard_idx)]
                    .init(ctx, *payload_cq, pd, config);
            }
        }

        ibv_port_attr port_attr;
        ibv_query_port(ctx.context, config.ib_port, &port_attr);
        uint32_t psn = lrand48() & 0xffffff;

        auto build_local_info = [&](size_t peer_idx) {
            size_t qp_count = 1 + peer_payload_qp_count;
            size_t bytes = sizeof(rdma::ServerPeerConnectionInfo) +
                           qp_count * sizeof(uint32_t);
            std::vector<uint8_t> buf(bytes, 0);
            auto *info =
                reinterpret_cast<rdma::ServerPeerConnectionInfo *>(buf.data());
            info->psn = psn;
            info->lid = port_attr.lid;
            info->endpoint_idx = static_cast<uint16_t>(local_server_index);
            info->qp_count = static_cast<uint16_t>(qp_count);
            info->qpn[0] = peer_qps[peer_idx].queue_pair->qp_num;
            for (size_t shard_idx = 0; shard_idx < peer_payload_qp_count;
                 shard_idx++) {
                info->qpn[1 + shard_idx] =
                    peer_payload_qps[peer_payload_qp_index(peer_idx, shard_idx)]
                        .queue_pair->qp_num;
            }
            return buf;
        };

        auto format_peer_qpn_list = [&](const rdma::ServerPeerConnectionInfo &info) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < info.qp_count; i++) {
                if (i != 0) {
                    oss << ",";
                }
                oss << info.qpn[i];
            }
            oss << "]";
            return oss.str();
        };
        (void)format_peer_qpn_list;

        auto finish_peer_connect = [&](size_t peer_idx,
                                       const rdma::ServerPeerConnectionInfo &remote,
                                       const rdma::ServerPeerConnectionInfo &local,
                                       const char *connect_role) {
            (void)connect_role;
            if (remote.qp_count != 1 + peer_payload_qp_count ||
                local.qp_count != 1 + peer_payload_qp_count) {
                ERROR("server peer connect: qp_count mismatch");
            }
            peer_qps[peer_idx].ready_to_recv(remote.lid, remote.psn,
                                             remote.qpn[0], config);
            peer_qps[peer_idx].ready_to_send(local.psn, config);
            ibv_qp_attr qp_attr{};
            ibv_qp_init_attr qp_init_attr{};
            int qret = ibv_query_qp(peer_qps[peer_idx].queue_pair, &qp_attr,
                                    IBV_QP_STATE | IBV_QP_DEST_QPN,
                                    &qp_init_attr);
            if (qret != 0 || qp_attr.qp_state != IBV_QPS_RTS ||
                qp_attr.dest_qp_num != remote.qpn[0]) {
                ERROR("server peer qp verify failed");
            }
            for (size_t shard_idx = 0; shard_idx < peer_payload_qp_count;
                 shard_idx++) {
                auto &payload_qp =
                    peer_payload_qps[peer_payload_qp_index(peer_idx, shard_idx)];
                payload_qp.ready_to_recv(remote.lid, remote.psn,
                                         remote.qpn[1 + shard_idx], config);
                payload_qp.ready_to_send(local.psn, config);
                ibv_qp_attr payload_attr{};
                ibv_qp_init_attr payload_init_attr{};
                qret = ibv_query_qp(payload_qp.queue_pair, &payload_attr,
                                    IBV_QP_STATE | IBV_QP_DEST_QPN,
                                    &payload_init_attr);
                if (qret != 0 || payload_attr.qp_state != IBV_QPS_RTS ||
                    payload_attr.dest_qp_num != remote.qpn[1 + shard_idx]) {
                    ERROR("server peer payload qp verify failed");
                }
            }
#if FARLIB_EC2PC_VERBOSE_DIAG
            std::cout << "INFO: peer-connect"
                      << " local=" << local_server_index
                      << " peer=" << peer_idx
                      << " role=" << connect_role
                      << " local_qpn=" << format_peer_qpn_list(local)
                      << " remote_qpn=" << format_peer_qpn_list(remote)
                      << std::endl;
#endif
        };

        auto [local_addr, _local_port] = config.get_full_endpoint(local_server_index);
        std::string listen_port = peer_port_for_index(local_server_index);
        sockaddr_in listen_addr{};
        listen_addr.sin_family = AF_INET;
        listen_addr.sin_port =
            htons(static_cast<in_port_t>(std::atoi(listen_port.c_str())));
        listen_addr.sin_addr = {.s_addr = inet_addr(local_addr.c_str())};
        int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        ASSERT(listen_fd >= 0);
        DEFER({ CHECK_ERR(close(listen_fd)); });
        int on = 1;
        CHECK_ERR(setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &on,
                             sizeof(on)));
        CHECK_ERR(bind(listen_fd, reinterpret_cast<sockaddr *>(&listen_addr),
                       sizeof(listen_addr)));
        CHECK_ERR(
            listen(listen_fd, static_cast<int>(std::max<size_t>(1, peer_endpoint_count))));

        for (size_t peer_idx = 0; peer_idx < local_server_index; peer_idx++) {
            auto [peer_addr, _peer_port] = config.get_full_endpoint(peer_idx);
            std::string peer_port = peer_port_for_index(peer_idx);
            int conn_fd = -1;
            for (size_t attempt = 0; attempt < 300; attempt++) {
                conn_fd =
                    tcp::connect_to_server(peer_addr.c_str(), peer_port.c_str());
                if (conn_fd >= 0) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (conn_fd < 0) {
                ERROR("server peer connect failed");
            }
            DEFER({ CHECK_ERR(close(conn_fd)); });
            auto local_info_buf = build_local_info(peer_idx);
            auto *local_info = reinterpret_cast<rdma::ServerPeerConnectionInfo *>(
                local_info_buf.data());
            size_t local_info_bytes = local_info_buf.size();
            ssize_t sent =
                tcp::send_all(conn_fd, local_info, local_info_bytes, 0);
            ASSERT(sent == static_cast<ssize_t>(local_info_bytes));
            rdma::ServerPeerConnectionInfo remote_head{};
            ssize_t recvd = tcp::recieve_all(conn_fd, &remote_head,
                                             sizeof(remote_head), 0);
            ASSERT(recvd == static_cast<ssize_t>(sizeof(remote_head)));
            size_t remote_info_bytes = sizeof(rdma::ServerPeerConnectionInfo) +
                                       remote_head.qp_count * sizeof(uint32_t);
            std::vector<uint8_t> remote_info_buf(remote_info_bytes, 0);
            std::memcpy(remote_info_buf.data(), &remote_head, sizeof(remote_head));
            if (remote_head.qp_count > 0) {
                ssize_t recvd_tail = tcp::recieve_all(
                    conn_fd, remote_info_buf.data() + sizeof(remote_head),
                    remote_head.qp_count * sizeof(uint32_t), 0);
                ASSERT(recvd_tail ==
                       static_cast<ssize_t>(remote_head.qp_count * sizeof(uint32_t)));
            }
            auto *remote_info =
                reinterpret_cast<rdma::ServerPeerConnectionInfo *>(
                    remote_info_buf.data());
            if (remote_info->endpoint_idx != peer_idx) {
                ERROR("server peer connect: remote endpoint mismatch");
            }
            finish_peer_connect(peer_idx, *remote_info, *local_info, "active");
        }

        std::vector<uint8_t> accepted(peer_endpoint_count, 0);
        for (size_t accepted_cnt = 0;
             accepted_cnt + local_server_index + 1 < peer_endpoint_count;) {
            sockaddr_in remote_addr{};
            socklen_t remote_len = sizeof(remote_addr);
            int conn_fd = accept(listen_fd, reinterpret_cast<sockaddr *>(&remote_addr),
                                 &remote_len);
            ASSERT(conn_fd >= 0);
            DEFER({ CHECK_ERR(close(conn_fd)); });
            rdma::ServerPeerConnectionInfo remote_head{};
            ssize_t recvd =
                tcp::recieve_all(conn_fd, &remote_head, sizeof(remote_head), 0);
            ASSERT(recvd == static_cast<ssize_t>(sizeof(remote_head)));
            size_t remote_info_bytes = sizeof(rdma::ServerPeerConnectionInfo) +
                                       remote_head.qp_count * sizeof(uint32_t);
            std::vector<uint8_t> remote_info_buf(remote_info_bytes, 0);
            std::memcpy(remote_info_buf.data(), &remote_head, sizeof(remote_head));
            if (remote_head.qp_count > 0) {
                ssize_t recvd_tail = tcp::recieve_all(
                    conn_fd, remote_info_buf.data() + sizeof(remote_head),
                    remote_head.qp_count * sizeof(uint32_t), 0);
                ASSERT(recvd_tail ==
                       static_cast<ssize_t>(remote_head.qp_count * sizeof(uint32_t)));
            }
            auto *remote_info =
                reinterpret_cast<rdma::ServerPeerConnectionInfo *>(
                    remote_info_buf.data());
            size_t peer_idx = remote_info->endpoint_idx;
            if (peer_idx >= peer_endpoint_count || peer_idx == local_server_index ||
                accepted[peer_idx] != 0) {
                ERROR("server peer accept: invalid endpoint");
            }
            auto local_info_buf = build_local_info(peer_idx);
            auto *local_info = reinterpret_cast<rdma::ServerPeerConnectionInfo *>(
                local_info_buf.data());
            size_t local_info_bytes = local_info_buf.size();
            ssize_t sent =
                tcp::send_all(conn_fd, local_info, local_info_bytes, 0);
            ASSERT(sent == static_cast<ssize_t>(local_info_bytes));
            finish_peer_connect(peer_idx, *remote_info, *local_info, "passive");
            accepted[peer_idx] = 1;
            accepted_cnt++;
        }
    }

    void init_peer_rpc_resources() {
        if (peer_qps.empty()) {
            return;
        }
        size_t queue_count = peer_qps.size();
        size_t payload_queue_count = peer_qps.size() * peer_payload_qp_count;
        size_t peer_recv_depth =
            std::max<size_t>(1, static_cast<size_t>(config.qp_recv_cap));
        peer_payload_recv_credit_count = payload_queue_count;
        peer_payload_recv_depth = peer_recv_depth;
        size_t recv_count = queue_count * peer_recv_depth;
        size_t payload_recv_count = payload_queue_count * peer_recv_depth;
        size_t send_count = queue_count * kPeerRpcDepth;
        size_t free_index_count = queue_count * kPeerRpcDepth;
        size_t lane_owner_count = std::max<size_t>(1, rpc_worker_count);
        peer_lane_depth = configured_peer_lane_depth();
        peer_lane_pending_depth = configured_peer_lane_pending_depth();
        probe_parity_batch_send_depth = configured_probe_parity_batch_send_depth();
        probe_parity_batch_pending_depth =
            configured_probe_parity_batch_pending_depth();
        peer_ack_send_depth = configured_peer_ack_send_depth();
        size_t lane_send_count = queue_count * lane_owner_count * peer_lane_depth;
        size_t probe_batch_send_count =
            payload_queue_count * probe_parity_batch_send_depth;
        size_t ack_send_count = queue_count * peer_ack_send_depth;
        size_t recv_bytes = recv_count * sizeof(PeerRpcRecvSlot);
        size_t payload_recv_bytes = payload_recv_count * sizeof(PeerPayloadRecvSlot);
        size_t send_bytes = send_count * sizeof(PeerRpcSendSlot);
        size_t free_index_bytes = free_index_count * sizeof(uint16_t);
        size_t lane_send_bytes = lane_send_count * sizeof(PeerLaneSendSlot);
        size_t probe_batch_send_bytes =
            probe_batch_send_count * sizeof(ProbeParityBatchSendSlot);
        size_t ack_send_bytes = ack_send_count * sizeof(PeerAckSendSlot);
        size_t total_bytes =
            recv_bytes + payload_recv_bytes + send_bytes + free_index_bytes +
            lane_send_bytes + probe_batch_send_bytes + ack_send_bytes;
        size_t aligned_bytes = ((total_bytes + 63) / 64) * 64;
        peer_rpc_mem = std::aligned_alloc(64, aligned_bytes);
        if (peer_rpc_mem == nullptr) {
            ERROR("server peer rpc_mem alloc failed");
        }
        std::memset(peer_rpc_mem, 0, total_bytes);
        peer_rpc_mr =
            ibv_reg_mr(pd.protection_domain, peer_rpc_mem, total_bytes,
                       IBV_ACCESS_LOCAL_WRITE);
        if (peer_rpc_mr == nullptr) {
            ERROR("server peer rpc_mr register failed");
        }

        auto *recv_base = reinterpret_cast<PeerRpcRecvSlot *>(peer_rpc_mem);
        auto *payload_recv_base = reinterpret_cast<PeerPayloadRecvSlot *>(
            reinterpret_cast<uint8_t *>(peer_rpc_mem) + recv_bytes);
        auto *send_base = reinterpret_cast<PeerRpcSendSlot *>(
            reinterpret_cast<uint8_t *>(peer_rpc_mem) + recv_bytes +
            payload_recv_bytes);
        auto *free_index_base = reinterpret_cast<uint16_t *>(
            reinterpret_cast<uint8_t *>(peer_rpc_mem) + recv_bytes +
            payload_recv_bytes +
            send_bytes);
        auto *lane_send_base = reinterpret_cast<PeerLaneSendSlot *>(
            reinterpret_cast<uint8_t *>(peer_rpc_mem) + recv_bytes +
            payload_recv_bytes + send_bytes + free_index_bytes);
        auto *probe_batch_send_base = reinterpret_cast<ProbeParityBatchSendSlot *>(
            reinterpret_cast<uint8_t *>(peer_rpc_mem) + recv_bytes +
            payload_recv_bytes + send_bytes + free_index_bytes +
            lane_send_bytes);
        auto *ack_send_base = reinterpret_cast<PeerAckSendSlot *>(
            reinterpret_cast<uint8_t *>(peer_rpc_mem) + recv_bytes +
            payload_recv_bytes + send_bytes + free_index_bytes +
            lane_send_bytes + probe_batch_send_bytes);
        peer_rpc_queues.assign(queue_count, PeerRpcQueue{});
        peer_payload_recv_queues.assign(payload_queue_count,
                                        PeerPayloadRecvQueue{});
        peer_lane_send_queues.assign(queue_count * lane_owner_count,
                                     PeerLaneSendQueue{});
        probe_parity_batch_send_queues.assign(payload_queue_count,
                                              ProbeParityBatchSendQueue{});
        probe_parity_batch_ready_queues.assign(
            payload_queue_count, std::deque<ProbeParityBatchReadyItem>{});
        peer_ack_send_queues.assign(queue_count, PeerAckSendQueue{});
        peer_lane_pending_mem.reset(new rdma::EC2PCRpcMessage[
            queue_count * lane_owner_count * peer_lane_pending_depth]);
        probe_parity_batch_pending_mem.reset(new ProbeParityBatchPendingItem[
            payload_queue_count * probe_parity_batch_pending_depth]);
        peer_lane_mutexes.reset(
            new std::mutex[queue_count * lane_owner_count]);
        peer_send_mutexes.reset(new std::mutex[queue_count]);
        probe_parity_batch_send_mutexes.reset(new std::mutex[payload_queue_count]);
        probe_parity_batch_pending_mutexes.reset(
            new std::mutex[payload_queue_count]);
        probe_parity_batch_pending_space_cvs.reset(
            new std::condition_variable[payload_queue_count]);
        probe_parity_batch_ready_mutexes.reset(
            new std::mutex[payload_queue_count]);
        probe_parity_batch_next_shards.reset(
            new std::atomic<uint32_t>[queue_count]);
        pending_peer_send_queues.assign(
            queue_count, std::deque<rdma::EC2PCRpcMessage>{});
        pending_peer_ack_queues.assign(
            queue_count, std::deque<rdma::EC2PCRpcMessage>{});
        inflight_peer_send_slots.assign(queue_count,
                                        std::deque<PeerRpcSendSlot *>{});
        pending_peer_send_queue_mutexes.reset(new std::mutex[queue_count]);
        peer_ack_send_mutexes.reset(new std::mutex[queue_count]);
        pending_peer_ack_queue_mutexes.reset(new std::mutex[queue_count]);
        inflight_peer_send_slot_mutexes.reset(new std::mutex[queue_count]);
        runtime_parity_batch_ready_by_qp.reset(
            new std::atomic<uint64_t>[payload_queue_count]);
        runtime_parity_batch_outstanding_by_qp.reset(
            new std::atomic<uint64_t>[payload_queue_count]);
        peer_payload_recv_credits.reset(
            new std::atomic<uint32_t>[payload_queue_count]);
        pending_peer_send_total.store(0, std::memory_order_relaxed);
        for (size_t peer_idx = 0; peer_idx < queue_count; peer_idx++) {
            if (probe_parity_batch_next_shards) {
                probe_parity_batch_next_shards[peer_idx].store(
                    0, std::memory_order_relaxed);
            }
        }
        for (size_t q_idx = 0; q_idx < payload_queue_count; q_idx++) {
            peer_payload_recv_credits[q_idx].store(0, std::memory_order_relaxed);
            runtime_parity_batch_ready_by_qp[q_idx].store(
                0, std::memory_order_relaxed);
            runtime_parity_batch_outstanding_by_qp[q_idx].store(
                0, std::memory_order_relaxed);
        }
        for (size_t peer_idx = 0; peer_idx < queue_count; peer_idx++) {
            auto &queue = peer_rpc_queues[peer_idx];
            queue.recv_slots = recv_base + peer_idx * peer_recv_depth;
            queue.send_slots = send_base + peer_idx * kPeerRpcDepth;
            queue.free_slot_indices = free_index_base + peer_idx * kPeerRpcDepth;
            queue.free_head = 0;
            queue.free_tail = 0;
            queue.free_count = kPeerRpcDepth;
            if (peer_idx == local_server_index ||
                peer_qps[peer_idx].queue_pair == nullptr) {
                continue;
            }
            for (size_t i = 0; i < peer_recv_depth; i++) {
                auto &recv_slot = queue.recv_slots[i];
                new (&recv_slot) PeerRpcRecvSlot();
                recv_slot.magic = 0xEC2C2001u;
                recv_slot.peer_idx = static_cast<uint16_t>(peer_idx);
                post_peer_recv_slot(recv_slot);
            }

            for (size_t shard_idx = 0; shard_idx < peer_payload_qp_count;
                 shard_idx++) {
                size_t payload_q_idx =
                    peer_payload_qp_index(peer_idx, shard_idx);
                auto &payload_queue = peer_payload_recv_queues[payload_q_idx];
                payload_queue.recv_slots =
                    payload_recv_base + payload_q_idx * peer_recv_depth;
                for (size_t i = 0; i < peer_recv_depth; i++) {
                    auto &recv_slot = payload_queue.recv_slots[i];
                    new (&recv_slot) PeerPayloadRecvSlot();
                    recv_slot.magic = 0xEC2C2004u;
                    recv_slot.peer_idx = static_cast<uint16_t>(peer_idx);
                    recv_slot.shard_idx = static_cast<uint16_t>(shard_idx);
                    post_peer_payload_recv_slot(recv_slot);
                }
            }

            for (size_t i = 0; i < kPeerRpcDepth; i++) {
                auto &send_slot = queue.send_slots[i];
                new (&send_slot) PeerRpcSendSlot();
                send_slot.magic = 0xEC2C2002u;
                send_slot.peer_idx = static_cast<uint16_t>(peer_idx);
                send_slot.slot_index = static_cast<uint16_t>(i);
                send_slot.queue_index = static_cast<uint32_t>(peer_idx);
                send_slot.in_use.store(false, std::memory_order_relaxed);
                queue.free_slot_indices[i] = static_cast<uint16_t>(i);
            }
            queue.free_tail = kPeerRpcDepth % kPeerRpcDepth;

            for (size_t worker_idx = 0; worker_idx < lane_owner_count;
                 worker_idx++) {
                size_t lane_idx = peer_idx * lane_owner_count + worker_idx;
                auto &lane = peer_lane_send_queues[lane_idx];
                lane.slots = lane_send_base +
                             (peer_idx * lane_owner_count + worker_idx) *
                                 peer_lane_depth;
                lane.next_slot = 0;
                lane.next_shard = 0;
                lane.pending_msgs =
                    peer_lane_pending_mem.get() +
                    (peer_idx * lane_owner_count + worker_idx) *
                        peer_lane_pending_depth;
                lane.pending_capacity = peer_lane_pending_depth;
                lane.pending_head = 0;
                lane.pending_tail = 0;
                lane.pending_count = 0;
                for (size_t slot_idx = 0; slot_idx < peer_lane_depth; slot_idx++) {
                    auto &lane_slot = lane.slots[slot_idx];
                    new (&lane_slot) PeerLaneSendSlot();
                    lane_slot.magic = 0xEC2C2003u;
                    lane_slot.peer_idx = static_cast<uint16_t>(peer_idx);
                    lane_slot.lane_data_qp_idx =
                        static_cast<uint16_t>(worker_idx);
                    lane_slot.slot_index = static_cast<uint16_t>(slot_idx);
                    lane_slot.in_use.store(false, std::memory_order_relaxed);
                }
            }

            for (size_t shard_idx = 0; shard_idx < peer_payload_qp_count;
                 shard_idx++) {
                size_t probe_q_idx =
                    probe_parity_batch_queue_index(peer_idx, shard_idx);
                auto &probe_batch_queue =
                    probe_parity_batch_send_queues[probe_q_idx];
                probe_batch_queue.slots =
                    probe_batch_send_base +
                    probe_q_idx * probe_parity_batch_send_depth;
                probe_batch_queue.depth = probe_parity_batch_send_depth;
                probe_batch_queue.next_slot = 0;
                probe_batch_queue.next_shard = shard_idx;
                probe_batch_queue.next_ready = 0;
                probe_batch_queue.pending_items =
                    probe_parity_batch_pending_mem.get() +
                    probe_q_idx * probe_parity_batch_pending_depth;
                probe_batch_queue.pending_capacity =
                    probe_parity_batch_pending_depth;
                probe_batch_queue.pending_head = 0;
                probe_batch_queue.pending_tail = 0;
                probe_batch_queue.pending_count = 0;
                for (size_t i = 0; i < probe_parity_batch_send_depth; i++) {
                    auto &slot = probe_batch_queue.slots[i];
                    new (&slot) ProbeParityBatchSendSlot();
                    slot.magic = 0xEC2C2005u;
                    slot.peer_idx = static_cast<uint16_t>(peer_idx);
                    slot.shard_idx = static_cast<uint16_t>(shard_idx);
                    slot.slot_index = static_cast<uint16_t>(i);
                    slot.state.store(kProbeBatchSlotFree,
                                     std::memory_order_relaxed);
                    slot.in_use.store(false, std::memory_order_relaxed);
                }
            }

            auto &ack_queue = peer_ack_send_queues[peer_idx];
            ack_queue.slots = ack_send_base + peer_idx * peer_ack_send_depth;
            ack_queue.depth = peer_ack_send_depth;
            ack_queue.next_slot = 0;
            for (size_t i = 0; i < peer_ack_send_depth; i++) {
                auto &slot = ack_queue.slots[i];
                new (&slot) PeerAckSendSlot();
                slot.magic = 0xEC2C2006u;
                slot.peer_idx = static_cast<uint16_t>(peer_idx);
                slot.slot_index = static_cast<uint16_t>(i);
                slot.in_use.store(false, std::memory_order_relaxed);
            }
        }
    }

    void release_peer_resources() {
        peer_poller_stop.store(true, std::memory_order_release);
        for (auto &th : peer_pollers) {
            if (th.joinable()) {
                th.join();
            }
        }
        peer_pollers.clear();
        {
            std::lock_guard<std::mutex> lock(server_ec2pc_track_mutex_);
            server_ec2pc_track_.clear();
        }
        if (peer_rpc_mr != nullptr) {
            ibv_dereg_mr(peer_rpc_mr);
            peer_rpc_mr = nullptr;
        }
        if (peer_rpc_mem != nullptr) {
            std::free(peer_rpc_mem);
            peer_rpc_mem = nullptr;
        }
        pending_peer_send_queues.clear();
        pending_peer_ack_queues.clear();
        inflight_peer_send_slots.clear();
        {
            std::lock_guard<std::mutex> lock(
                runtime_parity_batch_pair_pending_mutex);
            runtime_parity_batch_pair_pending_queue.clear();
        }
        runtime_parity_batch_pair_pending_slot_ids_by_worker.clear();
        runtime_parity_batch_pair_storage_by_worker.clear();
        peer_payload_recv_queues.clear();
        peer_lane_send_queues.clear();
        probe_parity_batch_send_queues.clear();
        probe_parity_batch_ready_queues.clear();
        peer_ack_send_queues.clear();
        runtime_parity_batches.clear();
        peer_lane_depth = 1;
        peer_lane_pending_depth = 0;
        peer_payload_qp_count = 1;
        probe_parity_batch_send_depth = 0;
        probe_parity_batch_pending_depth = 0;
        peer_ack_send_depth = 0;
        peer_lane_pending_mem.reset();
        probe_parity_batch_pending_mem.reset();
        peer_lane_mutexes.reset();
        probe_parity_batch_send_mutexes.reset();
        probe_parity_batch_pending_mutexes.reset();
        probe_parity_batch_pending_space_cvs.reset();
        probe_parity_batch_ready_mutexes.reset();
        probe_parity_batch_next_shards.reset();
        pending_peer_send_queue_mutexes.reset();
        peer_ack_send_mutexes.reset();
        pending_peer_ack_queue_mutexes.reset();
        inflight_peer_send_slot_mutexes.reset();
        runtime_parity_batch_ready_by_qp.reset();
        runtime_parity_batch_outstanding_by_qp.reset();
        pending_peer_send_total.store(0, std::memory_order_relaxed);
        peer_rpc_queues.clear();
        peer_send_mutexes.reset();
        peer_qps.clear();
        peer_payload_qps.clear();
        peer_cq.reset();
        peer_payload_cqs.clear();
        peer_payload_cq_poll_mutexes.reset();
        peer_endpoint_count = 0;
        local_server_index = 0;
    }

    void connect() {
        srand48(time(nullptr));
        uint32_t psn = lrand48() & 0xffffff;
        ibv_port_attr port_attr;
        ibv_query_port(ctx.context, config.ib_port, &port_attr);

        int sock_fd;
        int conn_fd = tcp::listen_for_client(
            config.server_addr.c_str(), config.server_port.c_str(), sock_fd);
        DEFER({ CHECK_ERR(close(conn_fd)); });
        DEFER({ CHECK_ERR(close(sock_fd)); });

        ClientConnectionInfo client_info_head;
        ssize_t recv_size =
            tcp::recieve_all(conn_fd, &client_info_head, sizeof(ClientConnectionInfo), 0);
        ASSERT(recv_size >= 0 && static_cast<size_t>(recv_size) == sizeof(ClientConnectionInfo));

        size_t qp_count = client_info_head.qp_count;
        ASSERT(qp_count > 0);
        std::unique_ptr<uint32_t[]> client_qpn(new uint32_t[qp_count]);
        size_t qpn_size_in_byte = sizeof(uint32_t) * qp_count;
        recv_size = tcp::recieve_all(conn_fd, client_qpn.get(), qpn_size_in_byte, 0);
        ASSERT(recv_size >= 0 && static_cast<size_t>(recv_size) == qpn_size_in_byte);

        qps.clear();
        qps.reserve(qp_count);
        qps.emplace_back(ctx, control_cq, pd, config);

        size_t data_qp_count = qp_count - 1;
        data_cqs.clear();
        data_qp_to_cq_idx.clear();
        data_cq_count = 0;
        if (data_qp_count > 0) {
            size_t requested_cq_count = std::max<size_t>(1, config.server_cq_count);
            data_cq_count = std::min(data_qp_count, requested_cq_count);
            data_cq_count = std::max<size_t>(1, data_cq_count);
            data_cqs.reserve(data_cq_count);
            for (size_t i = 0; i < data_cq_count; i++) {
                data_cqs.emplace_back(std::make_unique<CompleteQueue>(ctx, config));
            }
            data_qp_to_cq_idx.assign(data_qp_count, 0);
            for (size_t i = 0; i < data_qp_count; i++) {
                size_t cq_idx = i % data_cq_count;
                data_qp_to_cq_idx[i] = cq_idx;
                qps.emplace_back(ctx, *data_cqs[cq_idx], pd, config);
            }
        }

        config.qp_max_rd_atomic = client_info_head.max_rd_atomic;
        config.qp_mtu = client_info_head.mtu;
        ASSERT(config.server_buffer_size >=
               client_info_head.server_buffer_size);

        size_t server_info_size =
            sizeof(ServerConnectionInfo) + sizeof(uint32_t) * qp_count;
        auto server_info =
            static_cast<ServerConnectionInfo *>(std::malloc(server_info_size));
        DEFER({ std::free(server_info); });

        server_info->addr = reinterpret_cast<uint64_t>(buffer);
        server_info->lid = port_attr.lid;
        server_info->psn = psn;
        server_info->qp_count = qp_count;
        server_info->rkey = mr.memory_region->rkey;

        for (size_t i = 0; i < qp_count; i++) {
            server_info->qpn[i] = qps[i].queue_pair->qp_num;
        }

        // modify state to RTR/RTS
        for (size_t i = 0; i < qp_count; i++) {
            qps[i].ready_to_recv(client_info_head.lid, client_info_head.psn,
                                 client_qpn[i], config);
            qps[i].ready_to_send(psn, config);
        }

        // Full QP verification (control + all data QPs): state must be RTS and
        // destination QPN must match peer-announced QPN.
        for (size_t i = 0; i < qp_count; i++) {
            ibv_qp_attr qp_attr{};
            ibv_qp_init_attr qp_init_attr{};
            int qret = ibv_query_qp(qps[i].queue_pair, &qp_attr,
                                    IBV_QP_STATE | IBV_QP_DEST_QPN,
                                    &qp_init_attr);
            if (qret != 0) {
                std::cerr << "[RDMA-server-verify] query failed port="
                          << config.server_port << " qp=" << i
                          << " ret=" << qret << std::endl;
                ERROR("RDMA server qp verify: ibv_query_qp failed");
            }
            if (qp_attr.qp_state != IBV_QPS_RTS ||
                qp_attr.dest_qp_num != client_qpn[i]) {
                std::cerr << "[RDMA-server-verify] mismatch port="
                          << config.server_port << " qp=" << i
                          << " state=" << qp_attr.qp_state
                          << " dest_qpn=" << qp_attr.dest_qp_num
                          << " expected_dest_qpn=" << client_qpn[i]
                          << std::endl;
                ERROR("RDMA server qp verify: state/dest mismatch");
            }
        }

        // Make the local receive rings and RPC workers ready before the
        // client sees server_info and starts preflight probing the data lanes.
        init_rpc_resources();

        ssize_t sent_size =
            tcp::send_all(conn_fd, server_info, server_info_size, 0);
        ASSERT(sent_size >= 0 && static_cast<size_t>(sent_size) == server_info_size);

    }
};

#include "server_rpc_core.hpp"
#include "server_rpc_data.hpp"
#include "server_rpc_parity.hpp"

}  // namespace rdma
}  // namespace FarLib
