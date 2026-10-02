/**
 * RAII style structures to initialize RDMA connection
 */

#pragma once
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "config.hpp"
#include "utils/debug.hpp"

namespace FarLib {

namespace rdma {
inline uint8_t checked_qp_rd_atomic(const Configure &config) {
    ASSERT(config.qp_max_rd_atomic <= std::numeric_limits<uint8_t>::max());
    return static_cast<uint8_t>(config.qp_max_rd_atomic);
}

enum RequestType {
    RQ_STOP = 0,
};

enum ControlImmType : uint32_t {
    CONTROL_IMM_STOP = 1,
    CONTROL_IMM_RESET_EC2PC_STATS = 2,
};

constexpr uint32_t kEC2PCRpcPayloadBytes = 8192;
constexpr uint32_t kProbeParityBatchMaxSpans = 8;
constexpr uint32_t kEC2PCDataReqBatchMaxSpans = 8;
constexpr uint32_t kEC2PCClientBatchQpCount = 1;

enum EC2PCRpcType : uint16_t {
    EC2PC_MSG_INVALID = 0,
    EC2PC_MSG_DATA_REQ = 1,
    EC2PC_MSG_PARITY_PAYLOAD = 2,
    EC2PC_MSG_PARITY_APPLY = 3,
    EC2PC_MSG_ACK = 4,
    EC2PC_MSG_ACK_BATCH = 5,
    EC2PC_MSG_PROBE_REQ = 6,
    EC2PC_MSG_PROBE_ACK = 7,
    EC2PC_MSG_DATA_REQ_BATCH = 8,
    EC2PC_MSG_COMPACT_REQ = 9,
    EC2PC_MSG_COMPACT_REQ_BATCH = 10,
};

constexpr uint16_t kEC2PCFlagInitFromZero = 1u << 0;
constexpr uint16_t kEC2PCFlagParityWrite = 1u << 1;
constexpr uint16_t kEC2PCFlagProbeBypassCompute = 1u << 2;
constexpr uint16_t kEC2PCFlagProbeBatch = 1u << 3;
constexpr uint16_t kEC2PCFlagCompactReq = 1u << 4;
constexpr uint32_t kEC2PCStatusOK = 0;
// Failure status propagated through the existing EC2PC ACK wire format when a
// peer is quarantined by the opt-in recovery path.
constexpr uint32_t kEC2PCStatusPeerDead = 1;

constexpr bool ec2pc_is_client_batch_type(uint16_t type) {
    return type == EC2PC_MSG_DATA_REQ_BATCH ||
           type == EC2PC_MSG_COMPACT_REQ_BATCH;
}

struct alignas(64) EC2PCRpcMessage {
    uint16_t type = EC2PC_MSG_INVALID;
    uint16_t flags = 0;
    uint16_t parity_idx = 0;
    uint16_t data_slot = 0;
    uint32_t payload_len = 0;
    uint64_t wr_id = 0;

    uint64_t data_offset = 0;
    uint64_t parity_offset0 = 0;
    uint64_t parity_offset1 = 0;

    uint16_t parity_endpoint0 = 0;
    uint16_t parity_endpoint1 = 0;
    uint16_t target_endpoint = 0;
    uint16_t reserved0 = 0;

    uint64_t target_offset = 0;
    uint32_t status = kEC2PCStatusOK;
    uint32_t reserved1 = 0;

    uint8_t payload[kEC2PCRpcPayloadBytes] = {0};
};

// Piggyback the server-local recv_wc->client_ack_post interval into ACK headers
// using microseconds so the value remains meaningful without cross-host clock
// sync and still fits in the 32-bit reserved field.
inline uint64_t ec2pc_rpc_server_recv_to_ack_post_sum_ns(
    const EC2PCRpcMessage &msg) {
    return static_cast<uint64_t>(msg.reserved1) * 1000ull;
}

inline void set_ec2pc_rpc_server_recv_to_ack_post_sum_ns(
    EC2PCRpcMessage &msg, uint64_t ns) {
    uint64_t us = (ns + 999ull) / 1000ull;
    if (us > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        us = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    }
    msg.reserved1 = static_cast<uint32_t>(us);
}

struct alignas(64) EC2PCDataReqBatchMessage {
    uint16_t type = EC2PC_MSG_DATA_REQ_BATCH;
    uint16_t span_count = 0;
    uint16_t parity_endpoint0 = 0;
    uint16_t parity_endpoint1 = 0;
    uint16_t ack_data_qp_idx = 0;
    uint16_t reserved0 = 0;
    uint32_t status = kEC2PCStatusOK;
    uint64_t send_time_ns = 0;
    uint64_t post_wall_time_ns = 0;
    uint64_t wr_ids[kEC2PCDataReqBatchMaxSpans] = {0};
    uint64_t data_offsets[kEC2PCDataReqBatchMaxSpans] = {0};
    uint64_t target_offsets[kEC2PCDataReqBatchMaxSpans] = {0};
    uint64_t parity_offsets0[kEC2PCDataReqBatchMaxSpans] = {0};
    uint64_t parity_offsets1[kEC2PCDataReqBatchMaxSpans] = {0};
    uint8_t data_slots[kEC2PCDataReqBatchMaxSpans] = {0};
    uint8_t init_from_zero[kEC2PCDataReqBatchMaxSpans] = {0};
    uint16_t reserved1[kEC2PCDataReqBatchMaxSpans] = {0};
    uint8_t payload[kEC2PCDataReqBatchMaxSpans][kEC2PCRpcPayloadBytes] = {{0}};
};

struct alignas(64) ProbeParityBatchMessage {
    uint16_t type = EC2PC_MSG_PARITY_PAYLOAD;
    uint16_t span_count = 0;
    uint16_t parity_idx = 0;
    uint16_t flags = 0;
    uint16_t ack_count = 0;
    uint16_t reserved0 = 0;
    uint32_t status = kEC2PCStatusOK;
    uint64_t wr_id = 0;
    uint64_t send_time_ns = 0;
    uint64_t post_time_ns = 0;
    uint64_t wr_ids[kProbeParityBatchMaxSpans] = {0};
    uint64_t target_offsets[kProbeParityBatchMaxSpans] = {0};
    uint8_t payload[kProbeParityBatchMaxSpans][kEC2PCRpcPayloadBytes] = {{0}};
};

constexpr uint32_t probe_parity_batch_header_bytes() {
    return static_cast<uint32_t>(offsetof(ProbeParityBatchMessage, payload));
}

constexpr uint32_t probe_parity_batch_wire_bytes(uint16_t span_count) {
    uint32_t bounded =
        std::min<uint32_t>(span_count, kProbeParityBatchMaxSpans);
    return probe_parity_batch_header_bytes() +
           bounded * kEC2PCRpcPayloadBytes;
}

constexpr uint64_t ec2pc_batch_parity_offset_from_data(uint64_t data_offset) {
    return data_offset;
}

constexpr uint32_t ec2pc_data_req_batch_header_bytes() {
    return static_cast<uint32_t>(
        offsetof(EC2PCDataReqBatchMessage, payload));
}

constexpr uint32_t ec2pc_data_req_batch_wire_bytes(
    const EC2PCDataReqBatchMessage &msg) {
    if (msg.type == rdma::EC2PC_MSG_COMPACT_REQ_BATCH) {
        return ec2pc_data_req_batch_header_bytes();
    }
    uint32_t bounded =
        std::min<uint32_t>(msg.span_count, kEC2PCDataReqBatchMaxSpans);
    return ec2pc_data_req_batch_header_bytes() +
           bounded * kEC2PCRpcPayloadBytes;
}

constexpr size_t kPeerRpcRecvBytes =
    sizeof(ProbeParityBatchMessage) > sizeof(EC2PCRpcMessage)
        ? sizeof(ProbeParityBatchMessage)
        : sizeof(EC2PCRpcMessage);

constexpr size_t kEC2PCAckBatchWrIdBytes = sizeof(uint64_t);
constexpr size_t kEC2PCAckBatchMaxCount =
    kEC2PCRpcPayloadBytes / kEC2PCAckBatchWrIdBytes;

constexpr uint32_t ec2pc_rpc_header_bytes() {
    return static_cast<uint32_t>(offsetof(EC2PCRpcMessage, payload));
}

constexpr uint32_t ec2pc_rpc_wire_bytes(const EC2PCRpcMessage &msg) {
    // ACK/ACK_BATCH only carry header + payload_len bytes.
    if (msg.type == rdma::EC2PC_MSG_ACK || msg.type == rdma::EC2PC_MSG_ACK_BATCH) {
        uint32_t bounded = msg.payload_len;
        if (bounded > kEC2PCRpcPayloadBytes) {
            bounded = kEC2PCRpcPayloadBytes;
        }
        return ec2pc_rpc_header_bytes() + bounded;
    }
    // COMPACT_REQ carries no payload.
    if (msg.type == rdma::EC2PC_MSG_COMPACT_REQ) {
        return ec2pc_rpc_header_bytes();
    }
    return static_cast<uint32_t>(sizeof(EC2PCRpcMessage));
}

struct ClientConnectionInfo {
    size_t server_buffer_size;  // in byte
    uint32_t psn;               // Packet Sequence Number
    uint16_t lid;               // Local IDentifier
    uint16_t qp_count;          // QP Count
    uint8_t max_rd_atomic;      // max_rd_atomic & max_dest_rd_atomic
    int mtu;                    // ibv_mtu
    uint32_t qpn[0];            // QP Numbers
};

static_assert(offsetof(ClientConnectionInfo, qpn) ==
              sizeof(ClientConnectionInfo));

struct ServerConnectionInfo {
    uint32_t psn;       // Packet Sequence Number
    uint16_t lid;       // Local IDentifier
    uint16_t qp_count;  // QP Count
    uint32_t rkey;
    uint64_t addr;
    uint32_t qpn[0];  // QP Numbers
};

struct ServerPeerConnectionInfo {
    uint32_t psn;
    uint16_t lid;
    uint16_t endpoint_idx;
    uint16_t qp_count;
    uint32_t qpn[0];
};

// use RAII to manage resources
struct Context {
    ibv_context *context;
    uint32_t num_comp_vectors = 1;
    uint32_t max_qp_rd_atom = 0;
    uint32_t max_qp_init_rd_atom = 0;

    Context() {
        int num_devices;
        ibv_device **dev_list = ibv_get_device_list(&num_devices);
        ASSERT(dev_list && num_devices > 0);
        const char *requested_device = std::getenv("FARLIB_RDMA_DEVICE");
        if (requested_device != nullptr && requested_device[0] != '\0') {
            ibv_device *device = nullptr;
            for (int i = 0; i < num_devices; i++) {
                const char *device_name = ibv_get_device_name(dev_list[i]);
                if (device_name != nullptr &&
                    std::string(device_name) == requested_device) {
                    device = dev_list[i];
                    break;
                }
            }
            if (device == nullptr) {
                std::ostringstream oss;
                oss << "RDMA device '" << requested_device
                    << "' not found. available=";
                for (int i = 0; i < num_devices; i++) {
                    if (i > 0) {
                        oss << ",";
                    }
                    const char *device_name = ibv_get_device_name(dev_list[i]);
                    oss << (device_name != nullptr ? device_name : "<null>");
                }
                std::cerr << oss.str() << std::endl;
                std::abort();
            }
            context = ibv_open_device(device);
        } else {
            context = nullptr;
            for (int i = 0; i < num_devices; i++) {
                ibv_context *candidate = ibv_open_device(dev_list[i]);
                if (candidate == nullptr) {
                    continue;
                }
                ibv_port_attr attr{};
                if (ibv_query_port(candidate, 1, &attr) == 0 &&
                    attr.state == IBV_PORT_ACTIVE) {
                    context = candidate;
                    break;
                }
                CHECK_ERR(ibv_close_device(candidate));
            }
            if (context == nullptr) {
                context = ibv_open_device(dev_list[0]);
            }
        }
        ibv_free_device_list(dev_list);
        ASSERT(context);
        ibv_device_attr device_attr{};
        CHECK_ERR(ibv_query_device(context, &device_attr));
        max_qp_rd_atom = static_cast<uint32_t>(device_attr.max_qp_rd_atom);
        max_qp_init_rd_atom =
            static_cast<uint32_t>(device_attr.max_qp_init_rd_atom);
        if (context->num_comp_vectors > 0) {
            num_comp_vectors =
                static_cast<uint32_t>(context->num_comp_vectors);
        }
    }

    Context(const Context &) = delete;
    Context(Context &&) = delete;

    ~Context() { CHECK_ERR(ibv_close_device(context)); }
};

struct ProtectionDomain {
    ibv_pd *protection_domain;

    explicit ProtectionDomain(Context &ctx) {
        protection_domain = ibv_alloc_pd(ctx.context);
        ASSERT(protection_domain);
    }

    ProtectionDomain(const ProtectionDomain &) = delete;
    ProtectionDomain(ProtectionDomain &&) = delete;

    ~ProtectionDomain() { CHECK_ERR(ibv_dealloc_pd(protection_domain)); }
};

struct MemoryRegion {
    ibv_mr *memory_region;

    MemoryRegion(ProtectionDomain &pd, void *buffer, size_t buffer_size) {
        int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE;
        ASSERT(buffer != nullptr);
        memory_region =
            ibv_reg_mr(pd.protection_domain, buffer, buffer_size, access_flags);
        ASSERT(memory_region);
    }

    MemoryRegion(const MemoryRegion &) = delete;
    MemoryRegion(MemoryRegion &&other) {
        memory_region = other.memory_region;
        other.memory_region = nullptr;
    }

    ~MemoryRegion() {
        if (memory_region != nullptr) {
            CHECK_ERR(ibv_dereg_mr(memory_region));
        }
    }
};

struct CompleteChannel {
    ibv_comp_channel *channel;
    CompleteChannel(const CompleteChannel &) = delete;
    CompleteChannel(CompleteChannel &&) = delete;

    CompleteChannel(Context &ctx) {
        channel = ibv_create_comp_channel(ctx.context);
        ASSERT(channel != nullptr);
    }
    ~CompleteChannel() { CHECK_ERR(ibv_destroy_comp_channel(channel)); }

    void wait_cq_event() {
        ibv_cq *ev_cq;
        void *ev_ctx;
        CHECK_ERR(ibv_get_cq_event(channel, &ev_cq, &ev_ctx));
        ibv_ack_cq_events(ev_cq, 1);
    }
};

struct CompleteQueue {
    ibv_cq *complete_queue;

    CompleteQueue(const CompleteQueue &) = delete;
    CompleteQueue(CompleteQueue &&) = delete;

    CompleteQueue(Context &ctx, const Configure &config, int comp_vector = 0) {
        int cq_vector = 0;
        if (ctx.num_comp_vectors > 0) {
            cq_vector =
                comp_vector % static_cast<int>(ctx.num_comp_vectors);
            if (cq_vector < 0) {
                cq_vector += static_cast<int>(ctx.num_comp_vectors);
            }
        }
        complete_queue = ibv_create_cq(ctx.context, config.cq_entries, nullptr,
                                       nullptr, cq_vector);
        ASSERT(complete_queue);
    }

    CompleteQueue(Context &ctx, const Configure &config,
                  const CompleteChannel &channel, int comp_vector = 0) {
        int cq_vector = 0;
        if (ctx.num_comp_vectors > 0) {
            cq_vector =
                comp_vector % static_cast<int>(ctx.num_comp_vectors);
            if (cq_vector < 0) {
                cq_vector += static_cast<int>(ctx.num_comp_vectors);
            }
        }
        complete_queue = ibv_create_cq(ctx.context, config.cq_entries, nullptr,
                                       channel.channel, cq_vector);
        ASSERT(complete_queue);
    }

    ~CompleteQueue() { CHECK_ERR(ibv_destroy_cq(complete_queue)); }

    void req_notify(bool solicited_only) {
        CHECK_ERR(ibv_req_notify_cq(complete_queue, solicited_only));
    }
};

struct QueuePair {
    ibv_qp *queue_pair;
    uint32_t max_qp_rd_atom = 0;
    uint32_t max_qp_init_rd_atom = 0;

    QueuePair() = default;

    QueuePair(Context &ctx, CompleteQueue &cq, ProtectionDomain &pd,
              const Configure &config) {
        init(ctx, cq, pd, config);
    }

    QueuePair(const QueuePair &) = delete;
    QueuePair(QueuePair &&other) {
        queue_pair = other.queue_pair;
        other.queue_pair = nullptr;
    }

    ~QueuePair() {
        if (queue_pair != nullptr) {
            CHECK_ERR(ibv_destroy_qp(queue_pair));
        }
    }

    void init(Context &ctx, CompleteQueue &cq, ProtectionDomain &pd,
              const Configure &config) {
        if (config.qp_max_rd_atomic > ctx.max_qp_rd_atom ||
            config.qp_max_rd_atomic > ctx.max_qp_init_rd_atom) {
            std::cerr << "qp_max_rd_atomic=" << config.qp_max_rd_atomic
                      << " exceeds device caps max_qp_rd_atom="
                      << ctx.max_qp_rd_atom << " max_qp_init_rd_atom="
                      << ctx.max_qp_init_rd_atom << std::endl;
            std::abort();
        }
        max_qp_rd_atom = ctx.max_qp_rd_atom;
        max_qp_init_rd_atom = ctx.max_qp_init_rd_atom;
        ibv_qp_cap qp_cap = {
            .max_send_wr = config.qp_send_cap,
            .max_recv_wr = config.qp_recv_cap,
            .max_send_sge = config.qp_max_send_sge,
            .max_recv_sge = config.qp_max_recv_sge,
            .max_inline_data = 0,
        };
        ibv_qp_init_attr qp_init_attr = {
            .qp_context = ctx.context,
            .send_cq = cq.complete_queue,
            .recv_cq = cq.complete_queue,
            .srq = nullptr,
            .cap = qp_cap,
            .qp_type = IBV_QPT_RC,
            .sq_sig_all =
                0,  // will NOT submit work completion for all requests
        };
        queue_pair = ibv_create_qp(pd.protection_domain, &qp_init_attr);
        int attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
        ibv_qp_attr qp_attr = {
            .qp_state = IBV_QPS_INIT,
            .qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_ATOMIC,
            .pkey_index = 0,
            .port_num = config.ib_port,
        };
#pragma GCC diagnostic pop
        CHECK_ERR(ibv_modify_qp(queue_pair, &qp_attr, attr_mask));
    }

    void ready_to_recv(uint16_t lid, uint32_t psn, uint32_t qpn,
                       const Configure &config) {
        const uint8_t qp_max_rd_atomic = checked_qp_rd_atomic(config);
        int attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
        ibv_ah_attr ah_attr = {
            .dlid = lid,
            .sl = 0,  // service level
            .src_path_bits = 0,
            .is_global = 0,
            .port_num = config.ib_port,
        };
        ibv_qp_attr qp_attr = {
            .qp_state = IBV_QPS_RTR,
            .path_mtu = static_cast<ibv_mtu>(config.qp_mtu),
            .rq_psn = psn,
            .dest_qp_num = qpn,
            .ah_attr = ah_attr,
            .max_dest_rd_atomic = qp_max_rd_atomic,
            .min_rnr_timer = config.qp_min_rnr_timer,
        };
#pragma GCC diagnostic pop
        CHECK_ERR(ibv_modify_qp(queue_pair, &qp_attr, attr_mask));
    }

    void ready_to_send(uint32_t psn, const Configure &config) {
        const uint8_t qp_max_rd_atomic = checked_qp_rd_atomic(config);
        int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                        IBV_QP_MAX_QP_RD_ATOMIC;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
        ibv_qp_attr qp_attr = {
            .qp_state = IBV_QPS_RTS,
            .sq_psn = psn,
            .max_rd_atomic = qp_max_rd_atomic,
            .timeout = config.qp_timeout,
            .retry_cnt = config.qp_retry_cnt,
            .rnr_retry = config.qp_rnr_retry,
        };
#pragma GCC diagnostic pop
        CHECK_ERR(ibv_modify_qp(queue_pair, &qp_attr, attr_mask));
    }
};

}  // namespace rdma

}  // namespace FarLib
