#pragma once

#include "rdma/ec_update_participant.hpp"
#include "rdma/rdma.hpp"
#include "utils/debug.hpp"

#include <infiniband/verbs.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace FarLib::rdma::ec_update {

// Server-side RDMA adapter for Participant.  The adapter deliberately uses
// the data QPs already exchanged by Server::connect(): no peer QP or TCP
// control path is introduced for incremental updates.
//
// Every receive slot has one corresponding response slot.  A receive is not
// reposted until the response SEND completion for that pair is observed.  The
// fixed 64/64 banks therefore bound both registered memory and outstanding
// responses even when the client outpaces the participant.
class ServerTransport {
private:
    struct QpState {
        ibv_qp *qp = nullptr;
        std::unique_ptr<Message[]> receives;
        std::unique_ptr<Message[]> responses;
        ibv_mr *receive_mr = nullptr;
        ibv_mr *response_mr = nullptr;
        bool failed = false;
    };

    static constexpr size_t kFirstDataQp = 1;
    static constexpr uint64_t kSlotMask = 0xffffffffULL;

    ProtectionDomain &pd_;
    std::vector<QueuePair> &qps_;
    CompleteQueue &receive_cq_;
    CompleteQueue &send_cq_;
    Participant participant_;
    std::vector<QpState> states_;
    bool armed_ = false;

    static uint64_t work_id(size_t qp_index, size_t slot) {
        return (static_cast<uint64_t>(qp_index) << 32) |
               static_cast<uint64_t>(slot);
    }

    bool decode_work_id(uint64_t id, size_t &qp_index, size_t &slot,
                        QpState *&state) {
        qp_index = static_cast<size_t>(id >> 32);
        slot = static_cast<size_t>(id & kSlotMask);
        if (qp_index < kFirstDataQp || qp_index >= qps_.size() ||
            slot >= kReceiveDepth) {
            return false;
        }
        state = &states_[qp_index - kFirstDataQp];
        return state->qp == qps_[qp_index].queue_pair;
    }

    bool post_receive(size_t qp_index, size_t slot, QpState &state) {
        // Only the header needs clearing.  A valid payload SEND covers every
        // byte that the participant reads; a short malformed SEND is rejected
        // before payload access.
        std::memset(&state.receives[slot], 0, kHeaderBytes);
        ibv_sge sge{
            reinterpret_cast<uint64_t>(&state.receives[slot]),
            static_cast<uint32_t>(sizeof(Message)), state.receive_mr->lkey};
        ibv_recv_wr wr{};
        wr.wr_id = work_id(qp_index, slot);
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ibv_recv_wr *bad = nullptr;
        return ibv_post_recv(state.qp, &wr, &bad) == 0;
    }

    void post_all_receives() {
        for (size_t state_index = 0; state_index < states_.size();
             ++state_index) {
            QpState &state = states_[state_index];
            const size_t qp_index = state_index + kFirstDataQp;
            for (size_t slot = 0; slot < kReceiveDepth; ++slot)
                ASSERT(post_receive(qp_index, slot, state));
        }
        armed_ = true;
    }

    void post_response(size_t qp_index, size_t slot, QpState &state,
                       const Message &response) {
        Message &wire = state.responses[slot];
        std::memset(&wire, 0, kHeaderBytes);
        std::memcpy(&wire, &response, kHeaderBytes);
        if (carries_payload(wire))
            std::memcpy(wire.payload, response.payload, wire.bytes);
        const size_t bytes = wire_bytes(wire);
        ASSERT(bytes >= kHeaderBytes && bytes <= sizeof(Message));

        ibv_sge sge{reinterpret_cast<uint64_t>(&wire),
                    static_cast<uint32_t>(bytes), state.response_mr->lkey};
        ibv_send_wr wr{};
        wr.wr_id = work_id(qp_index, slot);
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        // A completion is required before this receive slot can be reused.
        wr.send_flags = IBV_SEND_SIGNALED;
        ibv_send_wr *bad = nullptr;
        if (ibv_post_send(state.qp, &wr, &bad) != 0) {
            // With qp_send_cap >= 64 there can be at most 64 outstanding
            // responses, but a transport-level failure still belongs to this
            // lane rather than aborting the whole server process.
            state.failed = true;
        }
    }

    void handle_receive(const ibv_wc &wc) {
        size_t qp_index = 0, slot = 0;
        QpState *state = nullptr;
        ASSERT(decode_work_id(wc.wr_id, qp_index, slot, state));
        if (state->failed || wc.status != IBV_WC_SUCCESS) {
            state->failed = true;
            return;
        }

        // The receive bank was zeroed before posting, so a short malformed
        // header can still be turned into a fixed-header INVALID response
        // without reading beyond the CQE-provided bytes.
        const Message &request = state->receives[slot];
        Message response{};
        participant_.handle(request, wc.byte_len, response);
        post_response(qp_index, slot, *state, response);
    }

    void handle_send(const ibv_wc &wc) {
        size_t qp_index = 0, slot = 0;
        QpState *state = nullptr;
        ASSERT(decode_work_id(wc.wr_id, qp_index, slot, state));
        if (wc.status != IBV_WC_SUCCESS) {
            state->failed = true;
            return;
        }
        // An error completion still means the SEND WQE has left the SQ.  The
        // client will observe the failed QP.  Do not post a new receive on an
        // error QP; the lane is fenced by refusing all future work.
        if (!state->failed) state->failed = !post_receive(qp_index, slot, *state);
    }

    static int poll(ibv_cq *cq, ibv_wc *work_completions, int count) {
        const int result = ibv_poll_cq(cq, count, work_completions);
        ASSERT(result >= 0);
        return result;
    }

public:
    ServerTransport(const Configure &config, ProtectionDomain &pd,
                    std::vector<QueuePair> &qps, CompleteQueue &receive_cq,
                    CompleteQueue &send_cq, void *buffer, size_t capacity)
        : pd_(pd),
          qps_(qps),
          receive_cq_(receive_cq),
          send_cq_(send_cq),
          participant_(buffer, capacity, config.evacuate_thread_cnt) {
        ASSERT(config.evacuate_thread_cnt > 0);
        ASSERT(config.evacuate_thread_cnt <=
               static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
        ASSERT(config.qp_recv_cap >= kReceiveDepth);
        ASSERT(config.qp_send_cap >= kReceiveDepth);
        ASSERT(qps_.size() >= kFirstDataQp);
        ASSERT(qps_.size() - kFirstDataQp <=
               static_cast<size_t>(std::numeric_limits<uint32_t>::max()));

        states_.resize(qps_.size() - kFirstDataQp);
        for (size_t state_index = 0; state_index < states_.size();
             ++state_index) {
            QpState &state = states_[state_index];
            state.qp = qps_[state_index + kFirstDataQp].queue_pair;
            state.receives = std::make_unique<Message[]>(kReceiveDepth);
            state.responses = std::make_unique<Message[]>(kReceiveDepth);
            state.receive_mr = ibv_reg_mr(
                pd_.protection_domain, state.receives.get(),
                sizeof(Message) * kReceiveDepth, IBV_ACCESS_LOCAL_WRITE);
            state.response_mr = ibv_reg_mr(
                pd_.protection_domain, state.responses.get(),
                sizeof(Message) * kReceiveDepth, IBV_ACCESS_LOCAL_WRITE);
            ASSERT(state.receive_mr != nullptr);
            ASSERT(state.response_mr != nullptr);
        }
    }

    ServerTransport(const ServerTransport &) = delete;
    ServerTransport &operator=(const ServerTransport &) = delete;

    ~ServerTransport() {
        // Server::start destroys the data QPs before this object so all posted
        // receives have been flushed before their MRs are deregistered.
        for (QpState &state : states_) {
            if (state.response_mr != nullptr)
                CHECK_ERR(ibv_dereg_mr(state.response_mr));
            if (state.receive_mr != nullptr)
                CHECK_ERR(ibv_dereg_mr(state.receive_mr));
        }
    }

    void arm() {
        ASSERT(!armed_);
        post_all_receives();
    }

    // Active-poll data receive/send CQs and the existing control stop CQ.
    // The caller must keep this adapter alive until after the data QPs have
    // been destroyed; that ordering flushes any still-posted receives before
    // the registered banks are deregistered.
    void run_until_stop(ibv_cq *control_cq, uint64_t stop_wr_id) {
        ASSERT(control_cq != nullptr);
        if (!armed_) arm();

        std::array<ibv_wc, kReceiveDepth> completions{};
        while (true) {
            ibv_wc control_wc{};
            const int control_count = poll(control_cq, &control_wc, 1);
            if (control_count != 0) {
                ASSERT(control_wc.wr_id == stop_wr_id);
                ASSERT(control_wc.status == IBV_WC_SUCCESS);
                return;
            }

            const int send_count =
                poll(send_cq_.complete_queue, completions.data(),
                     static_cast<int>(completions.size()));
            for (int i = 0; i < send_count; ++i)
                handle_send(completions[static_cast<size_t>(i)]);

            const int receive_count =
                poll(receive_cq_.complete_queue, completions.data(),
                     static_cast<int>(completions.size()));
            for (int i = 0; i < receive_count; ++i)
                handle_receive(completions[static_cast<size_t>(i)]);

        }
    }

    // QP destruction flushes outstanding WQEs into the shared data CQs.  Drain
    // those CQEs before this adapter deregisters its banks or a later
    // connection reuses the same CQs with fresh work-id ranges.
    void drain_after_qp_destroy() {
        std::array<ibv_wc, kReceiveDepth> completions{};
        while (ibv_poll_cq(send_cq_.complete_queue,
                           static_cast<int>(completions.size()),
                           completions.data()) > 0) {
        }
        while (ibv_poll_cq(receive_cq_.complete_queue,
                           static_cast<int>(completions.size()),
                           completions.data()) > 0) {
        }
    }

    Participant &participant() { return participant_; }
    const Participant &participant() const { return participant_; }
};

using ServerAdapter = ServerTransport;

}  // namespace FarLib::rdma::ec_update
