#pragma once
#include <cerrno>
#include <mutex>
#include <string>
#include <unordered_map>
#include "rdma/client.hpp"
#include "cache/carbink/shadow_io.hpp"
#include "utils/debug.hpp"

namespace FarLib::cache::carbink::shadow {
namespace detail {
struct Registry {
    std::mutex mutex;
    std::unordered_map<uint64_t, IoBatch *> owners;
    std::atomic<uint64_t> sequence{1};
};
inline Registry &registry() {
    static Registry *value = new Registry();
    return *value;
}
inline uint64_t next_id(uint8_t slot) {
    auto &r = registry();
    for (;;) {
        const uint64_t sequence =
            r.sequence.fetch_add(1, std::memory_order_relaxed) &
            kShadowWrIdSequenceMask;
        if (sequence == 0) continue;
        return kShadowWrIdTag | (sequence << 8) |
               static_cast<uint64_t>(slot);
    }
}
}  // namespace detail

inline uint64_t IoBatch::arm(size_t slot) {
    if (slot >= kShadowZeroSlots) return 0;
    std::lock_guard<std::mutex> lock(detail::registry().mutex);
    if (slots_[slot].expected.load(std::memory_order_acquire) != 0)
        return 0;
    const uint64_t id = detail::next_id(static_cast<uint8_t>(slot));
    slots_[slot].status.store(0, std::memory_order_relaxed);
    slots_[slot].expected.store(id, std::memory_order_release);
    detail::registry().owners.emplace(id, this);
    return id;
}

inline void IoBatch::cancel(size_t slot) {
    if (slot >= kShadowZeroSlots) return;
    std::lock_guard<std::mutex> lock(detail::registry().mutex);
    const uint64_t id =
        slots_[slot].expected.exchange(0, std::memory_order_acq_rel);
    if (id != 0) {
        auto it = detail::registry().owners.find(id);
        if (it != detail::registry().owners.end() && it->second == this)
            detail::registry().owners.erase(it);
    }
    // Keep this release store last while holding the registry lock.  A CQ
    // consumer cannot retain this object's pointer past the lock boundary.
    slots_[slot].status.store(-1, std::memory_order_release);
}

inline int IoBatch::status(size_t slot) const {
    if (slot >= kShadowZeroSlots) return -1;
    return slots_[slot].status.load(std::memory_order_acquire);
}

inline bool IoBatch::all_terminal() const {
    for (size_t slot = 0; slot < kShadowZeroSlots; ++slot)
        if (status(slot) == 0) return false;
    return true;
}

inline bool IoBatch::all_success() const {
    for (size_t slot = 0; slot < kShadowZeroSlots; ++slot)
        if (status(slot) != 1) return false;
    return true;
}

inline void IoBatch::complete(uint64_t wr_id, bool success) {
    const size_t slot =
        static_cast<size_t>(wr_id & kShadowWrIdSlotMask);
    if (slot >= kShadowZeroSlots) return;
    const uint64_t expected =
        slots_[slot].expected.load(std::memory_order_acquire);
    if (expected != wr_id) return;
    auto it = detail::registry().owners.find(wr_id);
    if (it != detail::registry().owners.end() && it->second == this)
        detail::registry().owners.erase(it);
    slots_[slot].expected.store(0, std::memory_order_release);
    // This is intentionally the final access to *this; the caller holds the
    // registry mutex until after this release store.
    slots_[slot].status.store(success ? 1 : -1, std::memory_order_release);
}

inline bool IoBatch::consume_completion(const ibv_wc &wc) {
    if (!is_wr_id(wc.wr_id)) return false;
    std::lock_guard<std::mutex> lock(detail::registry().mutex);
    auto it = detail::registry().owners.find(wc.wr_id);
    if (it == detail::registry().owners.end()) return false;
    it->second->complete(wc.wr_id, wc.status == IBV_WC_SUCCESS);
    return true;
}

inline uint64_t IoBatch::next_rpc_wr_id() {
    return detail::next_id(0xff);
}

inline PostResult IoBatch::post_zero(
    rdma::Client &client, size_t qp_idx, size_t endpoint_idx,
    uint64_t remote_offset, void *zero_page, uint32_t zero_lkey,
    uint32_t bytes, size_t slot) {
    const uint64_t wr_id = arm(slot);
    if (wr_id == 0) return PostResult::kFailed;
    if (!client.carbink_transport().endpoint_is_alive(endpoint_idx)) {
        cancel(slot);
        return PostResult::kFailed;
    }
    auto *qp = client.get_endpoint_data_qp_local(endpoint_idx, qp_idx);
    if (qp == nullptr || rdma::ClientControl::get_default() == nullptr) {
        cancel(slot);
        return PostResult::kFailed;
    }
    ibv_sge sge{};
    ibv_send_wr wr{}, *bad = nullptr;
    client.build_send_wr(wr, sge, remote_offset, zero_page, bytes, wr_id,
                         true, IBV_WR_RDMA_WRITE, endpoint_idx);
    sge.lkey = zero_lkey;
    const int rc = ibv_post_send(qp->queue_pair, &wr, &bad);
    if (rc == ENOMEM) {
        cancel(slot);
        return PostResult::kQueueFull;
    }
    if (rc != 0) {
        cancel(slot);
        ERROR((std::string("carbink: shadow zero ibv_post_send failed rc=") +
               std::to_string(rc)).c_str());
    }
    return PostResult::kAccepted;
}

inline void IoBatch::forget_all() {
    std::lock_guard<std::mutex> lock(detail::registry().mutex);
    for (auto &slot : slots_) {
        const uint64_t id = slot.expected.exchange(
            0, std::memory_order_acq_rel);
        if (id != 0) {
            auto it = detail::registry().owners.find(id);
            if (it != detail::registry().owners.end() && it->second == this)
                detail::registry().owners.erase(it);
        }
        slot.status.store(-1, std::memory_order_release);
    }
}

}  // namespace FarLib::cache::carbink::shadow
