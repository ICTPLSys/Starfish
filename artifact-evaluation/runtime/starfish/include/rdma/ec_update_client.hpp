#pragma once
#include "rdma/ec_update_protocol.hpp"
#include "utils/debug.hpp"
#include <infiniband/verbs.h>
#include <atomic>
#include <cerrno>
#include <memory>
#include <vector>

namespace FarLib::rdma::ec_update {

// SEND/RECV use the evacuator's existing data QP/CQ, not an application CQ or
// the shared control CQ. The logical worker owns request construction; any CQ
// poller may deliver completion through the two release/acquire flags.
class ClientTransport {
public:
    struct alignas(64) Exchange {
        Message request{}, reply{};
        ibv_send_wr wr{};
        ibv_sge sge{};
        std::atomic<uint8_t> response_state{0}; // idle, waiting, writing, ready
        std::atomic<bool> send_done{true};
        std::atomic<bool> send_failed{false};
        bool posted = false; // logical worker only
    };
private:
    struct alignas(64) Receive {
        Message message{};
        ibv_qp *qp = nullptr;
    };
    size_t owners_ = 0, exchange_count_ = 0, receive_count_ = 0;
    std::unique_ptr<Exchange[]> exchanges_;
    std::unique_ptr<Receive[]> receives_;
    ibv_mr *send_mr_ = nullptr, *recv_mr_ = nullptr;

    template<class T> static T *owned(uint64_t id, T *base, size_t count) {
        const auto start = reinterpret_cast<uintptr_t>(base);
        if (!base || id < start || id - start >= count * sizeof(T) ||
            (id - start) % sizeof(T)) return nullptr;
        return base + (id - start) / sizeof(T);
    }
    bool repost(Receive &r) {
        ibv_sge sge{reinterpret_cast<uint64_t>(&r.message), sizeof(Message), recv_mr_->lkey};
        ibv_recv_wr wr{};
        wr.wr_id = reinterpret_cast<uint64_t>(&r);
        wr.sg_list = &sge; wr.num_sge = 1;
        ibv_recv_wr *bad = nullptr;
        return ibv_post_recv(r.qp, &wr, &bad) == 0;
    }
public:
    // The owner MUST destroy/stop its data QPs before destroying this object;
    // receive buffers remain posted for the whole connection lifetime.
    ~ClientTransport() {
        if (recv_mr_) CHECK_ERR(ibv_dereg_mr(recv_mr_));
        if (send_mr_) CHECK_ERR(ibv_dereg_mr(send_mr_));
    }
    void init(ibv_pd *pd, size_t owners, const std::vector<ibv_qp *> &qps) {
        owners_ = owners;
        exchange_count_ = owners * kSlots * 3;
        receive_count_ = qps.size() * kReceiveDepth;
        exchanges_.reset(new Exchange[exchange_count_]);
        receives_.reset(new Receive[receive_count_]);
        send_mr_ = ibv_reg_mr(pd, exchanges_.get(), exchange_count_ * sizeof(Exchange), IBV_ACCESS_LOCAL_WRITE);
        recv_mr_ = ibv_reg_mr(pd, receives_.get(), receive_count_ * sizeof(Receive), IBV_ACCESS_LOCAL_WRITE);
        if (!send_mr_ || !recv_mr_) ERROR("ec_update: register client RPC bank failed");
        for (size_t lane = 0; lane < qps.size(); ++lane)
            for (size_t i = 0; i < kReceiveDepth; ++i) {
                auto &r = receives_[lane * kReceiveDepth + i];
                r.qp = qps[lane];
                if (!repost(r)) ERROR("ec_update: post client reply receive failed");
            }
    }
    Exchange &get(size_t owner, size_t slot, size_t participant) {
        ASSERT(owner < owners_ && slot < kSlots && participant < 3);
        return exchanges_[(owner * kSlots + slot) * 3 + participant];
    }
    void begin(Exchange &x) {
        ASSERT(x.send_done.load(std::memory_order_acquire));
        // Exclude a CQ poller copying an old response before changing the
        // identity. This is per-entry completion ownership, not a shared pool.
        uint8_t state = x.response_state.load(std::memory_order_acquire);
        for (;;) {
            if (state == 2) { state = x.response_state.load(std::memory_order_acquire); continue; }
            if (x.response_state.compare_exchange_weak(state, 0, std::memory_order_acq_rel)) break;
        }
    }
    void arm(Exchange &x) {
        // Previous SEND ownership must have drained before request reuse.
        ASSERT(x.send_done.load(std::memory_order_acquire));
        x.posted = false;
        x.send_failed.store(false, std::memory_order_relaxed);
        x.send_done.store(false, std::memory_order_relaxed);
        x.response_state.store(1, std::memory_order_release);
    }
    // One verbs submission per endpoint batch. On SQ pressure only the accepted
    // prefix becomes posted; the caller polls and resubmits the remaining tail.
    size_t post(Exchange **list, size_t count, ibv_qp *qp) {
        if (!count) return 0;
        for (size_t i = 0; i < count; ++i) {
            auto &x = *list[i];
            x.sge = {reinterpret_cast<uint64_t>(&x.request),
                     static_cast<uint32_t>(wire_bytes(x.request)), send_mr_->lkey};
            x.wr = {};
            x.wr.wr_id = reinterpret_cast<uint64_t>(&x);
            x.wr.sg_list = &x.sge; x.wr.num_sge = 1;
            x.wr.opcode = IBV_WR_SEND; x.wr.send_flags = IBV_SEND_SIGNALED;
            x.wr.next = i + 1 < count ? &list[i + 1]->wr : nullptr;
        }
        ibv_send_wr *bad = nullptr;
        const int rc = ibv_post_send(qp, &list[0]->wr, &bad);
        size_t accepted = count;
        if (rc) {
            accepted = 0;
            while (accepted < count && &list[accepted]->wr != bad) ++accepted;
            if (!bad || accepted == count) ERROR("ec_update: invalid partial SEND result");
            if (rc != ENOMEM && rc != EAGAIN) ERROR("ec_update: SEND failed outside SQ backpressure");
        }
        for (size_t i = 0; i < accepted; ++i) list[i]->posted = true;
        return accepted;
    }
    bool handle(const ibv_wc &wc) {
        if (auto *x = owned(wc.wr_id, exchanges_.get(), exchange_count_)) {
            x->send_failed.store(wc.status != IBV_WC_SUCCESS, std::memory_order_relaxed);
            x->send_done.store(true, std::memory_order_release);
            return true;
        }
        auto *r = owned(wc.wr_id, receives_.get(), receive_count_);
        if (!r) return false;
        if (wc.status != IBV_WC_SUCCESS) return true;
        const auto &m = r->message;
        if (!valid(m, wc.byte_len) || !m.response || m.owner >= owners_)
            ERROR("ec_update: malformed participant response");
        auto &x = get(m.owner, m.slot, m.participant);
        uint8_t waiting = 1;
        if (x.response_state.compare_exchange_strong(waiting, 2, std::memory_order_acq_rel)) {
            if (m.generation == x.request.generation && m.op == x.request.op &&
                m.offset == x.request.offset && m.bytes == x.request.bytes) {
                std::memcpy(&x.reply, &m, wc.byte_len);
                x.response_state.store(3, std::memory_order_release);
            } else {
                x.response_state.store(1, std::memory_order_release);
            }
        }
        if (!repost(*r)) ERROR("ec_update: repost participant response failed");
        return true;
    }
};
} // namespace FarLib::rdma::ec_update
