#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <infiniband/verbs.h>
#include "rdma/rdma.hpp"

namespace FarLib::rdma::compact {

constexpr uint16_t kCompactRpcDataReq = 1;
constexpr uint16_t kCompactRpcDataReqBatch = 8;
constexpr uint16_t kCompactRpcCompactReqBatch = 10;
constexpr uint16_t kCompactRpcParityPayloadBatch = 2;
constexpr uint16_t kCompactRpcAck = 4;
constexpr uint16_t kCompactRpcAckBatch = 5;
constexpr uint16_t kCompactRpcStatusOk = 0;
// Nonzero status used when a compact request is abandoned because one of the
// participating endpoints failed.  It is deliberately distinct from the
// legacy success status and is carried in the existing ACK wire field.
constexpr uint16_t kCompactRpcStatusPeerDead = 1;
constexpr uint16_t kCompactRpcFlagInitFromZero = 1u << 0;
constexpr uint16_t kCompactRpcDataReqBatchMaxSpans = 8;
constexpr uint32_t kCompactRpcPayloadBytes = 8192;

inline constexpr size_t kCompactTransportStateShardCount = 64;

// Pointer wr_ids are naturally aligned and often have a fixed allocation
// stride. Mix higher address bits before selecting a shard; masking the raw
// low bits collapses the Update array onto only a few mutexes.
inline size_t compact_transport_state_shard_index(uint64_t wr_id) noexcept {
    uint64_t mixed = wr_id;
    mixed ^= mixed >> 6;
    mixed ^= mixed >> 12;
    mixed ^= mixed >> 24;
    return static_cast<size_t>(
        mixed & (kCompactTransportStateShardCount - 1));
}

struct alignas(64) CompactReqBatchMessage {
    uint16_t type = kCompactRpcDataReqBatch;
    uint16_t span_count = 0;
    uint16_t parity_endpoint0 = 0;
    uint16_t parity_endpoint1 = 0;
    uint16_t ack_data_qp_idx = 0;
    uint16_t reserved0 = 0;
    uint32_t status = kCompactRpcStatusOk;
    uint64_t send_time_ns = 0;
    uint64_t post_wall_time_ns = 0;
    uint64_t wr_ids[kCompactRpcDataReqBatchMaxSpans] = {};
    uint64_t data_offsets[kCompactRpcDataReqBatchMaxSpans] = {};
    uint64_t target_offsets[kCompactRpcDataReqBatchMaxSpans] = {};
    uint64_t parity_offsets0[kCompactRpcDataReqBatchMaxSpans] = {};
    uint64_t parity_offsets1[kCompactRpcDataReqBatchMaxSpans] = {};
    uint8_t data_slots[kCompactRpcDataReqBatchMaxSpans] = {};
    uint8_t init_from_zero[kCompactRpcDataReqBatchMaxSpans] = {};
    uint16_t reserved1[kCompactRpcDataReqBatchMaxSpans] = {};
};
inline constexpr size_t compact_req_batch_header_bytes() {
    return sizeof(CompactReqBatchMessage);
}
inline constexpr size_t compact_req_batch_wire_bytes(
    const CompactReqBatchMessage &m) {
    const size_t count = std::min<size_t>(
        m.span_count, kCompactRpcDataReqBatchMaxSpans);
    return m.type == kCompactRpcDataReqBatch
               ? sizeof(CompactReqBatchMessage) +
                     count * kCompactRpcPayloadBytes
               : sizeof(CompactReqBatchMessage);
}

struct alignas(64) CompactParityPayloadBatchMessage {
    uint16_t type = kCompactRpcParityPayloadBatch;
    uint16_t span_count = 0;
    uint16_t parity_idx = 0;
    uint16_t flags = 0;
    uint16_t ack_count = 0;
    uint16_t reserved0 = 0;
    uint32_t status = kCompactRpcStatusOk;
    uint64_t wr_id = 0;
    uint64_t send_time_ns = 0;
    uint64_t post_time_ns = 0;
    uint64_t wr_ids[kCompactRpcDataReqBatchMaxSpans] = {};
    uint64_t target_offsets[kCompactRpcDataReqBatchMaxSpans] = {};
    uint8_t payload[kCompactRpcDataReqBatchMaxSpans]
                  [kCompactRpcPayloadBytes] = {};
};
struct alignas(64) CompactAckBatchMessage {
    uint16_t type = kCompactRpcAckBatch;
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
    uint32_t status = kCompactRpcStatusOk;
    uint32_t reserved1 = 0;
    uint8_t payload[kCompactRpcPayloadBytes] = {};
};
static_assert(offsetof(CompactReqBatchMessage, wr_ids) == 32);
static_assert(offsetof(CompactReqBatchMessage, reserved1) == 368);
static_assert(sizeof(CompactReqBatchMessage) == 384);
static_assert(offsetof(CompactParityPayloadBatchMessage, payload) == 168);
static_assert(sizeof(CompactParityPayloadBatchMessage) == 65728);
static_assert(offsetof(CompactAckBatchMessage, payload) == 72);
static_assert(sizeof(CompactAckBatchMessage) == 8320);

inline constexpr size_t compact_ack_header_bytes() {
    return offsetof(CompactAckBatchMessage, payload);
}
inline bool compact_ack_message_shape_valid(
    const CompactAckBatchMessage &message, size_t received_bytes) {
    if (message.payload_len > kCompactRpcPayloadBytes ||
        received_bytes != compact_ack_header_bytes() + message.payload_len)
        return false;
    if (message.type == kCompactRpcAck)
        return message.payload_len == 0 && message.wr_id != 0;
    if (message.type == kCompactRpcAckBatch)
        return message.payload_len != 0 &&
               (message.payload_len % sizeof(uint64_t)) == 0;
    return false;
}
inline size_t compact_ack_message_count(
    const CompactAckBatchMessage &message, size_t received_bytes) {
    if (!compact_ack_message_shape_valid(message, received_bytes))
        return 0;
    return message.type == kCompactRpcAck
               ? 1
               : message.payload_len / sizeof(uint64_t);
}

inline bool compact_req_batch_type(uint16_t type) {
    return type == kCompactRpcDataReqBatch ||
           type == kCompactRpcCompactReqBatch;
}

struct CompactMoveDescriptor {
    uint64_t wr_id = 0;
    uint64_t src_offset = 0;
    uint64_t dst_offset = 0;
    uint64_t parity0_offset = 0;
    uint64_t parity1_offset = 0;
    uint16_t parity_endpoint0 = 0;
    uint16_t parity_endpoint1 = 0;
    uint16_t data_slot = 0;
    uint16_t ack_data_qp_idx = 0;
    size_t endpoint_idx = 0;
    size_t qp_idx = 0;
    bool init_from_zero = false;
};
// DATA_REQ_BATCH on the unmodified legacy server derives both parity offsets
// from data_offset. Only use it when that exact layout precondition holds.
// COMPACT_REQ_BATCH already carries explicit offsets and is unaffected.
inline uint16_t compact_data_request_type(const CompactMoveDescriptor &d) {
    return d.src_offset == d.parity0_offset && d.src_offset == d.parity1_offset
        ? kCompactRpcDataReqBatch : kCompactRpcDataReq;
}
struct CompactDataReqHeader {
    uint16_t type = kCompactRpcDataReq;
    uint16_t flags = 0, parity_idx = 0, data_slot = 0;
    uint32_t payload_len = kCompactRpcPayloadBytes;
    uint64_t wr_id = 0, data_offset = 0, parity_offset0 = 0, parity_offset1 = 0;
    uint16_t parity_endpoint0 = 0, parity_endpoint1 = 0;
    uint16_t target_endpoint = 0, reserved0 = 0;
    uint64_t target_offset = 0;
    uint32_t status = kCompactRpcStatusOk, reserved1 = 0;
};
static_assert(sizeof(CompactDataReqHeader) == compact_ack_header_bytes());
static_assert(offsetof(CompactDataReqHeader, wr_id) ==
              offsetof(CompactAckBatchMessage, wr_id));
static_assert(offsetof(CompactDataReqHeader, parity_offset0) ==
              offsetof(CompactAckBatchMessage, parity_offset0));
static_assert(offsetof(CompactDataReqHeader, reserved1) ==
              offsetof(CompactAckBatchMessage, reserved1));
inline CompactDataReqHeader compact_data_request_header(
    const CompactMoveDescriptor &d) {
    CompactDataReqHeader h{};
    h.flags = d.init_from_zero ? kCompactRpcFlagInitFromZero : 0;
    h.data_slot = d.data_slot; h.wr_id = d.wr_id;
    h.data_offset = d.src_offset;
    h.parity_offset0 = d.parity0_offset; h.parity_offset1 = d.parity1_offset;
    h.parity_endpoint0 = d.parity_endpoint0;
    h.parity_endpoint1 = d.parity_endpoint1;
    h.target_endpoint = static_cast<uint16_t>(d.endpoint_idx);
    h.target_offset = d.dst_offset;
    return h;
}

struct CompactDataDescriptor : CompactMoveDescriptor {
    const uint8_t *payload = nullptr;
};
struct CompactRequestHandle {
    uint64_t wr_id = 0;
    uint64_t lane = 0;
};
enum class CompactSubmitStatus : uint8_t {
    Accepted = 0, NeedFlush = 1, Backpressure = 2, Invalid = 3,
    EndpointDead = 4
};
struct CompactTransportEndpoint {
    QueuePair *qp = nullptr;
    CompleteQueue *cq = nullptr;
};

#ifdef FARLIB_CARBINK_COMPACT_TRANSPORT_TEST
struct CompactReqBatchTransportTest;
#endif

class CompactReqBatchTransport {
    static constexpr uint32_t kSendMagic = 0xEC2CC101u;
    static constexpr uint32_t kAckMagic = 0xEC2CC102u;
    struct SendSlot {
        uint32_t magic = kSendMagic;
        uint16_t lane = 0;
        uint16_t slot = 0;
        bool in_use = false;
        CompactReqBatchMessage message{};
        CompactDataReqHeader single_header{};
        // Legacy DATA_REQ transmits sizeof(EC2PCRpcMessage), including its
        // 56-byte alignment tail. Keep the zero-copy payload as a separate SGE.
        std::array<uint8_t, sizeof(CompactAckBatchMessage) -
            sizeof(CompactDataReqHeader) - kCompactRpcPayloadBytes> single_tail{};
        const uint8_t *payload[kCompactRpcDataReqBatchMaxSpans] = {};
        uint32_t payload_lkeys[kCompactRpcDataReqBatchMaxSpans] = {};
    };
    struct AckSlot {
        uint32_t magic = kAckMagic;
        uint16_t lane = 0;
        uint16_t slot = 0;
        CompactAckBatchMessage message{};
    };
    struct Lane {
        std::mutex mutex;
        int active = -1;
        std::vector<uint16_t> free_slots;
        std::vector<uint16_t> inflight_slots;
    };
    struct RequestState {
        bool send_done = false;
        bool ack_done = false;
        bool failed = false;
        uint32_t ack_status = std::numeric_limits<uint32_t>::max();
        size_t target_endpoint = 0;
        size_t parity_endpoint0 = 0;
        size_t parity_endpoint1 = 0;
        uint64_t creation_epoch = 0;
    };
    std::vector<CompactTransportEndpoint> endpoints_;
    size_t qps_per_endpoint_ = 0;
    size_t send_depth_ = 256;
    size_t ack_depth_ = 16;
    std::unique_ptr<Lane[]> lanes_;
    std::vector<SendSlot> send_slots_;
    std::vector<AckSlot> ack_slots_;
    ibv_mr *send_mr_ = nullptr;
    ibv_mr *ack_mr_ = nullptr;
    ibv_mr *zero_mr_ = nullptr;
    bool recv_posts_ready_ = false;
    std::unique_ptr<std::atomic<uint8_t>[]> endpoint_state_;
    std::atomic<bool> failure_recovery_enabled_{false};
    std::atomic<uint64_t> failure_epoch_{0};
    std::array<uint8_t, kCompactRpcPayloadBytes> zero_page_{};
    uint32_t payload_lkey_ = 0;
    std::atomic<uint64_t> explicit_data_requests_{0};
    std::atomic<uint64_t> batched_data_requests_{0};
    static constexpr size_t kStateShardCount =
        kCompactTransportStateShardCount;
    struct StateShard {
        mutable std::mutex mutex;
        std::unordered_map<uint64_t, RequestState> states;
    };
    std::array<StateShard, kStateShardCount> state_shards_{};
#ifdef FARLIB_CARBINK_COMPACT_TRANSPORT_TEST
    friend struct CompactReqBatchTransportTest;
#endif
    StateShard &state_shard(uint64_t wr_id) {
        return state_shards_[compact_transport_state_shard_index(wr_id)];
    }
    const StateShard &state_shard(uint64_t wr_id) const {
        return state_shards_[compact_transport_state_shard_index(wr_id)];
    }

    size_t lane_for(size_t endpoint, size_t qp) const {
        return endpoint * qps_per_endpoint_ + qp;
    }
    bool valid_lane(size_t lane) const {
        return lane < endpoints_.size() &&
               endpoints_[lane].qp != nullptr &&
               endpoints_[lane].cq != nullptr &&
               endpoints_[lane].qp->queue_pair != nullptr;
    }
    size_t endpoint_count() const {
        return qps_per_endpoint_ == 0 ? 0 :
            endpoints_.size() / qps_per_endpoint_;
    }
    size_t endpoint_for_lane(size_t lane) const {
        return qps_per_endpoint_ == 0 ? endpoint_count() :
            lane / qps_per_endpoint_;
    }
    bool endpoint_alive_internal(size_t endpoint) const {
        if (!failure_recovery_enabled_.load(std::memory_order_acquire))
            return true;
        if (endpoint >= endpoint_count() || endpoint_state_ == nullptr)
            return true;
        return endpoint_state_[endpoint].load(std::memory_order_acquire) == 0;
    }
    static void mark_request_failed(
        RequestState &state, uint32_t status, bool send_done) {
        state.failed = true;
        state.ack_done = true;
        state.ack_status = status;
        if (send_done) state.send_done = true;
    }
    void mark_slot_failed_locked(
        SendSlot &slot, uint32_t status, bool send_done,
        uint64_t failure_epoch = std::numeric_limits<uint64_t>::max()) {
        const size_t count = std::min<size_t>(
            slot.message.span_count, kCompactRpcDataReqBatchMaxSpans);
        for (size_t i = 0; i < count; ++i) {
            StateShard &shard = state_shard(slot.message.wr_ids[i]);
            std::lock_guard<std::mutex> state_lock(shard.mutex);
            auto it = shard.states.find(slot.message.wr_ids[i]);
            if (it != shard.states.end() &&
                (failure_epoch == std::numeric_limits<uint64_t>::max() ||
                 it->second.creation_epoch < failure_epoch))
                mark_request_failed(it->second, status, send_done);
        }
    }
    bool lane_has_request_before_epoch(
        const SendSlot &slot, uint64_t failure_epoch) {
        const size_t count = std::min<size_t>(
            slot.message.span_count, kCompactRpcDataReqBatchMaxSpans);
        for (size_t i = 0; i < count; ++i) {
            const StateShard &shard = state_shard(slot.message.wr_ids[i]);
            std::lock_guard<std::mutex> state_lock(shard.mutex);
            auto it = shard.states.find(slot.message.wr_ids[i]);
            if (it != shard.states.end() &&
                it->second.creation_epoch < failure_epoch)
                return true;
        }
        return false;
    }
    void fail_unposted_lane(
        size_t lane, uint32_t status,
        uint64_t failure_epoch = std::numeric_limits<uint64_t>::max()) {
        if (lanes_ == nullptr || lane >= endpoints_.size()) return;
        Lane &lane_state = lanes_[lane];
        std::lock_guard<std::mutex> lane_lock(lane_state.mutex);
        if (lane_state.active < 0) return;
        SendSlot &slot =
            send_slots_[lane * send_depth_ + lane_state.active];
        if (failure_epoch != std::numeric_limits<uint64_t>::max() &&
            !lane_has_request_before_epoch(slot, failure_epoch))
            return;
        // The transport slot is indivisible.  If it contains any request
        // predating the barrier, retire every request batched in that slot;
        // otherwise a post-barrier request would be left without a SEND CQE.
        mark_slot_failed_locked(slot, status, true);
        slot.in_use = false;
        slot.message = CompactReqBatchMessage{};
        for (auto &p : slot.payload) p = nullptr;
        for (auto &k : slot.payload_lkeys) k = 0;
        lane_state.active = -1;
        lane_state.free_slots.push_back(slot.slot);
    }
    void mark_requests_failed_for_endpoint(
        size_t endpoint, uint32_t status) {
        for (StateShard &shard : state_shards_) {
            std::lock_guard<std::mutex> lock(shard.mutex);
            for (auto &entry : shard.states) {
                RequestState &state = entry.second;
                if (state.target_endpoint == endpoint ||
                    state.parity_endpoint0 == endpoint ||
                    state.parity_endpoint1 == endpoint)
                    mark_request_failed(state, status, false);
            }
        }
    }
    void mark_all_requests_failed_before(
        uint64_t failure_epoch, uint32_t status) {
        for (StateShard &shard : state_shards_) {
            std::lock_guard<std::mutex> lock(shard.mutex);
            for (auto &entry : shard.states) {
                if (entry.second.creation_epoch < failure_epoch)
                    mark_request_failed(entry.second, status, false);
            }
        }
    }
    bool is_send_slot(uint64_t id) const {
        const uintptr_t value = static_cast<uintptr_t>(id);
        const uintptr_t begin =
            reinterpret_cast<uintptr_t>(send_slots_.data());
        const uintptr_t end = begin + send_slots_.size() * sizeof(SendSlot);
        return !send_slots_.empty() && value >= begin && value < end &&
               ((value - begin) % sizeof(SendSlot) == 0);
    }
    bool is_ack_slot(uint64_t id) const {
        const uintptr_t value = static_cast<uintptr_t>(id);
        const uintptr_t begin =
            reinterpret_cast<uintptr_t>(ack_slots_.data());
        const uintptr_t end = begin + ack_slots_.size() * sizeof(AckSlot);
        return !ack_slots_.empty() && value >= begin && value < end &&
               ((value - begin) % sizeof(AckSlot) == 0);
    }
    bool post_ack_recv(AckSlot &slot) {
        if (!valid_lane(slot.lane) || ack_mr_ == nullptr ||
            !endpoint_alive_internal(endpoint_for_lane(slot.lane)))
            return false;
        ibv_sge sge{reinterpret_cast<uint64_t>(&slot.message),
                    static_cast<uint32_t>(sizeof(slot.message)), ack_mr_->lkey};
        ibv_recv_wr wr{};
        wr.wr_id = reinterpret_cast<uint64_t>(&slot);
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ibv_recv_wr *bad = nullptr;
        return ibv_post_recv(endpoints_[slot.lane].qp->queue_pair, &wr,
                             &bad) == 0;
    }
    CompactSubmitStatus submit_impl(
        uint16_t type, const CompactMoveDescriptor &d, const uint8_t *payload,
        uint32_t payload_lkey, CompactRequestHandle *handle) {
        if (handle != nullptr) *handle = {};
        if (d.wr_id == 0 ||
            (!compact_req_batch_type(type) && type != kCompactRpcDataReq) ||
            ((type == kCompactRpcDataReqBatch || type == kCompactRpcDataReq) &&
             payload == nullptr) ||
            d.endpoint_idx * qps_per_endpoint_ + d.qp_idx >=
                endpoints_.size())
            return CompactSubmitStatus::Invalid;
        if (!endpoint_alive_internal(d.endpoint_idx) ||
            !endpoint_alive_internal(d.parity_endpoint0) ||
            !endpoint_alive_internal(d.parity_endpoint1))
            return CompactSubmitStatus::EndpointDead;
        const size_t lane = lane_for(d.endpoint_idx, d.qp_idx);
        Lane &lane_state = lanes_[lane];
        std::lock_guard<std::mutex> lane_lock(lane_state.mutex);
        if (!endpoint_alive_internal(d.endpoint_idx) ||
            !endpoint_alive_internal(d.parity_endpoint0) ||
            !endpoint_alive_internal(d.parity_endpoint1))
            return CompactSubmitStatus::EndpointDead;
        if (lane_state.active < 0) {
            if (lane_state.free_slots.empty())
                return CompactSubmitStatus::Backpressure;
            lane_state.active = lane_state.free_slots.back();
            lane_state.free_slots.pop_back();
            SendSlot &slot =
                send_slots_[lane * send_depth_ + lane_state.active];
            slot.in_use = true;
            slot.message = CompactReqBatchMessage{};
            slot.message.type = type;
            slot.message.parity_endpoint0 = d.parity_endpoint0;
            slot.message.parity_endpoint1 = d.parity_endpoint1;
            slot.message.ack_data_qp_idx = d.ack_data_qp_idx;
        }
        SendSlot &slot =
            send_slots_[lane * send_depth_ + lane_state.active];
        const size_t span_limit = type == kCompactRpcDataReq
            ? 1 : kCompactRpcDataReqBatchMaxSpans;
        if (slot.message.type != type ||
            slot.message.span_count >= span_limit ||
            slot.message.parity_endpoint0 != d.parity_endpoint0 ||
            slot.message.parity_endpoint1 != d.parity_endpoint1)
            return CompactSubmitStatus::NeedFlush;
        {
            StateShard &shard = state_shard(d.wr_id);
            std::lock_guard<std::mutex> state_lock(shard.mutex);
            if (shard.states.find(d.wr_id) != shard.states.end())
                return CompactSubmitStatus::Invalid;
            RequestState state{};
            state.target_endpoint = d.endpoint_idx;
            state.parity_endpoint0 = d.parity_endpoint0;
            state.parity_endpoint1 = d.parity_endpoint1;
            state.creation_epoch =
                failure_epoch_.load(std::memory_order_acquire);
            shard.states.emplace(d.wr_id, state);
        }
        const size_t i = slot.message.span_count++;
        slot.message.wr_ids[i] = d.wr_id;
        slot.message.data_offsets[i] = d.src_offset;
        slot.message.target_offsets[i] = d.dst_offset;
        slot.message.parity_offsets0[i] = d.parity0_offset;
        slot.message.parity_offsets1[i] = d.parity1_offset;
        slot.message.data_slots[i] = static_cast<uint8_t>(d.data_slot);
        slot.message.init_from_zero[i] = d.init_from_zero ? 1 : 0;
        slot.payload[i] = payload;
        slot.payload_lkeys[i] = payload_lkey;
        if (type == kCompactRpcDataReq) {
            slot.single_header = compact_data_request_header(d);
            explicit_data_requests_.fetch_add(1, std::memory_order_relaxed);
        } else if (type == kCompactRpcDataReqBatch) {
            batched_data_requests_.fetch_add(1, std::memory_order_relaxed);
        }
        if (handle != nullptr)
            *handle = {d.wr_id, static_cast<uint64_t>(lane)};
        return CompactSubmitStatus::Accepted;
    }
    void complete_send(SendSlot &slot) {
        const size_t count = std::min<size_t>(
            slot.message.span_count, kCompactRpcDataReqBatchMaxSpans);
        // submit_impl takes the lane lock before the state-shard lock. Keep
        // this order on retirement to avoid a lane/state lock inversion.
        Lane &lane = lanes_[slot.lane];
        std::lock_guard<std::mutex> lane_lock(lane.mutex);
        for (size_t i = 0; i < count; ++i) {
            StateShard &shard = state_shard(slot.message.wr_ids[i]);
            std::lock_guard<std::mutex> state_lock(shard.mutex);
            auto it = shard.states.find(slot.message.wr_ids[i]);
            if (it != shard.states.end()) it->second.send_done = true;
        }
        auto it = std::find(lane.inflight_slots.begin(),
                            lane.inflight_slots.end(), slot.slot);
        if (it != lane.inflight_slots.end()) lane.inflight_slots.erase(it);
        slot.in_use = false;
        slot.message = CompactReqBatchMessage{};
        for (auto &p : slot.payload) p = nullptr;
        for (auto &k : slot.payload_lkeys) k = 0;
        lane.free_slots.push_back(slot.slot);
    }
    void complete_ack(AckSlot &slot, uint32_t received_bytes) {
        constexpr size_t header_bytes = offsetof(CompactAckBatchMessage, payload);
        const uint32_t payload_bytes = slot.message.payload_len;
        if (payload_bytes > kCompactRpcPayloadBytes ||
            received_bytes != header_bytes + payload_bytes) {
            ERROR("Carbink compact ACK receive length is invalid");
        }

        uint64_t single_id = 0;
        const uint64_t *ids = nullptr;
        size_t count = 0;
        if (slot.message.type == kCompactRpcAck) {
            // EC2PC_MSG_ACK carries its one request id in wr_id; its payload
            // is empty on the legacy wire.
            if (payload_bytes != 0 || slot.message.wr_id == 0)
                ERROR("Carbink compact single ACK shape is invalid");
            single_id = slot.message.wr_id;
            ids = &single_id;
            count = 1;
        } else if (slot.message.type == kCompactRpcAckBatch) {
            if (payload_bytes == 0 ||
                (payload_bytes % sizeof(uint64_t)) != 0)
                ERROR("Carbink compact ACK_BATCH shape is invalid");
            ids = reinterpret_cast<const uint64_t *>(slot.message.payload);
            count = payload_bytes / sizeof(uint64_t);
            // ACK_BATCH may contain up to the complete 8192-byte payload
            // (1024 request ids), unlike a client request batch (8 spans).
            for (size_t i = 0; i < count; ++i)
                if (ids[i] == 0)
                    ERROR("Carbink compact ACK_BATCH contains zero wr_id");
        } else {
            ERROR("Carbink compact receive has unexpected message type");
        }

        for (size_t i = 0; i < count; ++i) {
            StateShard &shard = state_shard(ids[i]);
            std::lock_guard<std::mutex> lock(shard.mutex);
            auto it = shard.states.find(ids[i]);
            if (it != shard.states.end() && !it->second.failed) {
                it->second.ack_done = true;
                it->second.ack_status = slot.message.status;
                if (slot.message.status != kCompactRpcStatusOk)
                    it->second.failed = true;
            }
        }
    }

public:
#ifdef FARLIB_CARBINK_COMPACT_TRANSPORT_TEST
    // Verbs-free constructor used only by the CPU recovery-state test.  It
    // intentionally leaves MRs/QPs absent; the friend test hook exercises
    // request/epoch/CQE bookkeeping without claiming verbs coverage.
    explicit CompactReqBatchTransport(
        size_t test_endpoint_count, size_t test_send_depth = 2)
        : endpoints_(test_endpoint_count),
          qps_per_endpoint_(1),
          send_depth_(std::max<size_t>(1, test_send_depth)),
          ack_depth_(1) {
        endpoint_state_ = std::make_unique<std::atomic<uint8_t>[]>(
            endpoint_count());
        send_slots_.resize(endpoints_.size() * send_depth_);
        ack_slots_.resize(endpoints_.size() * ack_depth_);
        lanes_ = std::make_unique<Lane[]>(endpoints_.size());
        for (size_t lane = 0; lane < endpoints_.size(); ++lane) {
            for (size_t i = 0; i < send_depth_; ++i) {
                SendSlot &slot = send_slots_[lane * send_depth_ + i];
                slot.lane = static_cast<uint16_t>(lane);
                slot.slot = static_cast<uint16_t>(i);
                lanes_[lane].free_slots.push_back(
                    static_cast<uint16_t>(send_depth_ - i - 1));
            }
            AckSlot &slot = ack_slots_[lane];
            slot.lane = static_cast<uint16_t>(lane);
            slot.slot = 0;
            endpoint_state_[lane].store(0, std::memory_order_relaxed);
        }
    }
#endif
    CompactReqBatchTransport(
        ibv_pd *pd, std::vector<CompactTransportEndpoint> endpoints,
        size_t qps_per_endpoint, uint32_t payload_lkey = 0,
        size_t send_depth = 256, size_t ack_depth = 16)
        : endpoints_(std::move(endpoints)),
          qps_per_endpoint_(qps_per_endpoint),
          send_depth_(std::max<size_t>(1, send_depth)),
          ack_depth_(std::max<size_t>(1, ack_depth)),
          payload_lkey_(payload_lkey) {
        if (pd == nullptr || qps_per_endpoint_ == 0 ||
            endpoints_.empty())
            return;
        endpoint_state_ = std::make_unique<std::atomic<uint8_t>[]>(
            endpoint_count());
        for (size_t ep = 0; ep < endpoint_count(); ++ep)
            endpoint_state_[ep].store(0, std::memory_order_relaxed);
        zero_page_.fill(0);
        send_slots_.resize(endpoints_.size() * send_depth_);
        ack_slots_.resize(endpoints_.size() * ack_depth_);
        lanes_ = std::make_unique<Lane[]>(endpoints_.size());
        for (size_t lane = 0; lane < endpoints_.size(); ++lane) {
            for (size_t i = 0; i < send_depth_; ++i) {
                SendSlot &slot = send_slots_[lane * send_depth_ + i];
                slot.lane = static_cast<uint16_t>(lane);
                slot.slot = static_cast<uint16_t>(i);
                lanes_[lane].free_slots.push_back(
                    static_cast<uint16_t>(send_depth_ - i - 1));
            }
            for (size_t i = 0; i < ack_depth_; ++i) {
                AckSlot &slot = ack_slots_[lane * ack_depth_ + i];
                slot.lane = static_cast<uint16_t>(lane);
                slot.slot = static_cast<uint16_t>(i);
            }
        }
        send_mr_ = ibv_reg_mr(
            pd, send_slots_.data(),
            send_slots_.size() * sizeof(SendSlot), IBV_ACCESS_LOCAL_WRITE);
        ack_mr_ = ibv_reg_mr(
            pd, ack_slots_.data(),
            ack_slots_.size() * sizeof(AckSlot), IBV_ACCESS_LOCAL_WRITE);
        zero_mr_ = ibv_reg_mr(
            pd, zero_page_.data(), zero_page_.size(), IBV_ACCESS_LOCAL_WRITE);
        if (send_mr_ == nullptr || ack_mr_ == nullptr ||
            zero_mr_ == nullptr)
            return;
        recv_posts_ready_ = true;
        for (size_t lane = 0; lane < endpoints_.size(); ++lane) {
            for (size_t i = 0; i < ack_depth_; ++i) {
                if (!post_ack_recv(ack_slots_[lane * ack_depth_ + i])) {
                    recv_posts_ready_ = false;
                    return;
                }
            }
        }
    }
    ~CompactReqBatchTransport() {
        if (zero_mr_ != nullptr) ibv_dereg_mr(zero_mr_);
        if (send_mr_ != nullptr) ibv_dereg_mr(send_mr_);
        if (ack_mr_ != nullptr) ibv_dereg_mr(ack_mr_);
    }
    void enable_failure_recovery(bool enabled) {
        failure_recovery_enabled_.store(enabled, std::memory_order_release);
    }
    bool failure_recovery_enabled() const {
        return failure_recovery_enabled_.load(std::memory_order_acquire);
    }
    bool endpoint_is_alive(size_t endpoint) const {
        return endpoint_alive_internal(endpoint);
    }
    // Expose the endpoint encoded by a transport-owned CQE so the outer
    // client/router can notify cache recovery before it consumes the WC.
    size_t endpoint_for_completion(uint64_t wr_id) const {
        if (is_send_slot(wr_id)) {
            const auto *slot = reinterpret_cast<const SendSlot *>(wr_id);
            return endpoint_for_lane(slot->lane);
        }
        if (is_ack_slot(wr_id)) {
            const auto *slot = reinterpret_cast<const AckSlot *>(wr_id);
            return endpoint_for_lane(slot->lane);
        }
        return std::numeric_limits<size_t>::max();
    }
    bool mark_endpoint_failed(
        size_t endpoint, uint32_t status = kCompactRpcStatusPeerDead) {
        if (!failure_recovery_enabled() || endpoint >= endpoint_count() ||
            endpoint_state_ == nullptr)
            return false;
        uint8_t expected = 0;
        if (!endpoint_state_[endpoint].compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel))
            return false;
        uint64_t expected_epoch = 0;
        if (failure_epoch_.compare_exchange_strong(
                expected_epoch, 1, std::memory_order_acq_rel)) {
            for (size_t lane = 0; lane < endpoints_.size(); ++lane)
                fail_unposted_lane(lane, status, 1);
            mark_all_requests_failed_before(1, status);
        }
        mark_requests_failed_for_endpoint(endpoint, status);
        return true;
    }
    void fail_all_requests(
        uint32_t status = kCompactRpcStatusPeerDead) {
        if (!failure_recovery_enabled()) return;
        uint64_t failure_epoch =
            failure_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
        for (size_t lane = 0; lane < endpoints_.size(); ++lane)
            fail_unposted_lane(lane, status, failure_epoch);
        mark_all_requests_failed_before(failure_epoch, status);
    }
    bool request_failed(uint64_t wr_id, uint32_t *status = nullptr) const {
        const StateShard &shard = state_shard(wr_id);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.states.find(wr_id);
        if (it == shard.states.end()) return false;
        if (status != nullptr) *status = it->second.ack_status;
        return it->second.failed;
    }
    bool ready() const {
        return send_mr_ != nullptr && ack_mr_ != nullptr &&
               zero_mr_ != nullptr && lanes_ != nullptr && recv_posts_ready_;
    }
    CompactSubmitStatus submit(
        const CompactMoveDescriptor &d,
        CompactRequestHandle *handle = nullptr) {
        return submit_impl(kCompactRpcCompactReqBatch, d, nullptr, 0, handle);
    }
    CompactSubmitStatus submit_data(
        const CompactDataDescriptor &d,
        CompactRequestHandle *handle = nullptr) {
        return submit_impl(compact_data_request_type(d), d, d.payload,
                           payload_lkey_, handle);
    }
    CompactSubmitStatus submit_zero(
        const CompactMoveDescriptor &d,
        CompactRequestHandle *handle = nullptr) {
        return submit_impl(compact_data_request_type(d), d, zero_page_.data(),
                           zero_mr_ == nullptr ? 0 : zero_mr_->lkey, handle);
    }
    uint64_t explicit_data_requests() const {
        return explicit_data_requests_.load(std::memory_order_relaxed);
    }
    uint64_t batched_data_requests() const {
        return batched_data_requests_.load(std::memory_order_relaxed);
    }
    bool flush(size_t endpoint_idx, size_t qp_idx) {
        const size_t lane = lane_for(endpoint_idx, qp_idx);
        if (!ready() || !valid_lane(lane)) return false;
        if (!endpoint_alive_internal(endpoint_idx)) {
            fail_unposted_lane(lane, kCompactRpcStatusPeerDead);
            return true;
        }
        Lane &lane_state = lanes_[lane];
        std::unique_lock<std::mutex> lock(lane_state.mutex);
        if (lane_state.active < 0) return true;
        if (!endpoint_alive_internal(endpoint_idx)) {
            SendSlot &dead_slot =
                send_slots_[lane * send_depth_ + lane_state.active];
            mark_slot_failed_locked(
                dead_slot, kCompactRpcStatusPeerDead, true);
            dead_slot.in_use = false;
            dead_slot.message = CompactReqBatchMessage{};
            for (auto &p : dead_slot.payload) p = nullptr;
            for (auto &k : dead_slot.payload_lkeys) k = 0;
            lane_state.active = -1;
            lane_state.free_slots.push_back(dead_slot.slot);
            return true;
        }
        SendSlot &slot =
            send_slots_[lane * send_depth_ + lane_state.active];
        const size_t count = std::min<size_t>(
            slot.message.span_count, kCompactRpcDataReqBatchMaxSpans);
        ibv_sge sges[1 + kCompactRpcDataReqBatchMaxSpans]{};
        sges[0] = {reinterpret_cast<uint64_t>(&slot.message),
                   static_cast<uint32_t>(sizeof(CompactReqBatchMessage)),
                   send_mr_->lkey};
        size_t sge_count = 1;
        if (slot.message.type == kCompactRpcDataReq) {
            if (count != 1 || slot.payload[0] == nullptr ||
                slot.payload_lkeys[0] == 0) return false;
            sges[0] = {reinterpret_cast<uint64_t>(&slot.single_header),
                       static_cast<uint32_t>(sizeof(slot.single_header)),
                       send_mr_->lkey};
            sges[sge_count++] = {reinterpret_cast<uint64_t>(slot.payload[0]),
                                 kCompactRpcPayloadBytes, slot.payload_lkeys[0]};
            sges[sge_count++] = {reinterpret_cast<uint64_t>(slot.single_tail.data()),
                                 static_cast<uint32_t>(slot.single_tail.size()),
                                 send_mr_->lkey};
        } else if (slot.message.type == kCompactRpcDataReqBatch) {
            for (size_t i = 0; i < count; ++i) {
                if (slot.payload[i] == nullptr || slot.payload_lkeys[i] == 0)
                    return false;
                sges[sge_count++] = {
                    reinterpret_cast<uint64_t>(slot.payload[i]),
                    kCompactRpcPayloadBytes, slot.payload_lkeys[i]};
            }
        }
        ibv_send_wr wr{};
        wr.wr_id = reinterpret_cast<uint64_t>(&slot);
        wr.sg_list = sges;
        wr.num_sge = static_cast<int>(sge_count);
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ibv_send_wr *bad = nullptr;
        const int post_result = ibv_post_send(
            endpoints_[lane].qp->queue_pair, &wr, &bad);
        if (post_result == ENOMEM) return false;
        if (post_result != 0) {
            if (failure_recovery_enabled()) {
                mark_slot_failed_locked(
                    slot, kCompactRpcStatusPeerDead, true);
                slot.in_use = false;
                slot.message = CompactReqBatchMessage{};
                for (auto &p : slot.payload) p = nullptr;
                for (auto &k : slot.payload_lkeys) k = 0;
                lane_state.active = -1;
                lane_state.free_slots.push_back(slot.slot);
                lock.unlock();
                if (endpoint_is_alive(endpoint_idx)) {
                    std::cerr << "carbink: ibv_post_send error="
                              << post_result << '\n';
                    ERROR("Carbink compact SEND posting failed");
                }
                return true;
            }
            std::cerr << "carbink: ibv_post_send error=" << post_result << '\n';
            ERROR("Carbink compact SEND posting failed");
        }
        lane_state.inflight_slots.push_back(
            static_cast<uint16_t>(lane_state.active));
        lane_state.active = -1;
        return true;
    }
    // Flush only this worker's global QP lane at every endpoint.  The
    // all-lane helper remains available for diagnostics and setup tests.
    bool flush_qp(size_t global_qp) {
        if (!ready() || global_qp >= qps_per_endpoint_) return false;
        bool ok = true;
        const size_t endpoint_count =
            qps_per_endpoint_ == 0 ? 0 : endpoints_.size() / qps_per_endpoint_;
        for (size_t ep = 0; ep < endpoint_count; ++ep)
            ok = flush(ep, global_qp) && ok;
        return ok;
    }
    bool flush_all() {
        bool ok = true;
        if (qps_per_endpoint_ == 0) return false;
        for (size_t ep = 0; ep < endpoints_.size() / qps_per_endpoint_;
             ++ep)
            for (size_t qp = 0; qp < qps_per_endpoint_; ++qp)
                ok = flush(ep, qp) && ok;
        return ok;
    }
    bool consume_completion(const ibv_wc &wc) {
        // An error WC need not retain a meaningful opcode. Ownership comes
        // from the registered slot range, not from SEND/RECV classification.
        if (wc.status != IBV_WC_SUCCESS && is_send_slot(wc.wr_id)) {
            auto *slot = reinterpret_cast<SendSlot *>(wc.wr_id);
            if (failure_recovery_enabled()) {
                mark_endpoint_failed(
                    endpoint_for_lane(slot->lane), kCompactRpcStatusPeerDead);
                complete_send(*slot);
                return true;
            }
            ERROR("Carbink transport completion failed");
        }
        if (wc.status != IBV_WC_SUCCESS && is_ack_slot(wc.wr_id)) {
            auto *slot = reinterpret_cast<AckSlot *>(wc.wr_id);
            if (failure_recovery_enabled()) {
                mark_endpoint_failed(
                    endpoint_for_lane(slot->lane), kCompactRpcStatusPeerDead);
                slot->message = CompactAckBatchMessage{};
                return true;
            }
            ERROR("Carbink transport completion failed");
        }
        if (wc.opcode == IBV_WC_SEND && is_send_slot(wc.wr_id)) {
            if (wc.status != IBV_WC_SUCCESS)
                ERROR("Carbink compact SEND completion failed");
            auto *slot = reinterpret_cast<SendSlot *>(wc.wr_id);
            complete_send(*slot);
            return true;
        }
        if (wc.opcode == IBV_WC_RECV && is_ack_slot(wc.wr_id)) {
            if (wc.status != IBV_WC_SUCCESS)
                ERROR("Carbink compact ACK receive completion failed");
            auto *slot = reinterpret_cast<AckSlot *>(wc.wr_id);
            complete_ack(*slot, wc.byte_len);
            slot->message = CompactAckBatchMessage{};
            const bool reposted = post_ack_recv(*slot);
            if (!reposted) {
                if (failure_recovery_enabled())
                    mark_endpoint_failed(
                        endpoint_for_lane(slot->lane),
                        kCompactRpcStatusPeerDead);
                else
                    ERROR("Carbink compact ACK receive repost failed");
            }
            return true;
        }
        return false;
    }
    bool send_complete(uint64_t wr_id) const {
        const StateShard &shard = state_shard(wr_id);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.states.find(wr_id);
        return it != shard.states.end() && it->second.send_done;
    }
    bool final_ack_done(uint64_t wr_id, uint32_t *status = nullptr) const {
        const StateShard &shard = state_shard(wr_id);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.states.find(wr_id);
        if (it == shard.states.end()) return false;
        if (status != nullptr) *status = it->second.ack_status;
        return it->second.ack_done;
    }
    bool erase(uint64_t wr_id) {
        StateShard &shard = state_shard(wr_id);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.states.find(wr_id);
        if (it == shard.states.end() || !it->second.send_done ||
            !it->second.ack_done)
            return false;
        shard.states.erase(it);
        return true;
    }
};

}  // namespace FarLib::rdma::compact
