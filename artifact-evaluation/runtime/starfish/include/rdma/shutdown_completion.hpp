#pragma once

#include <infiniband/verbs.h>

#include <cstdint>

namespace FarLib::rdma {

enum class StopCompletionKind : uint8_t {
    kIgnore = 0,
    kSuccess,
    kError,
};

// Error WCs may carry a provider-dependent opcode.  A successful STOP WC
// must be the SEND from this control QP; an error WC for that STOP/QP is
// terminal regardless of opcode.
inline StopCompletionKind classify_stop_completion(const ibv_wc &wc,
                                                   uint64_t stop_wr_id,
                                                   uint32_t control_qp_num) noexcept {
    if (wc.wr_id != stop_wr_id || wc.qp_num != control_qp_num) {
        return StopCompletionKind::kIgnore;
    }
    if (wc.status == IBV_WC_SUCCESS) {
        return wc.opcode == IBV_WC_SEND ? StopCompletionKind::kSuccess
                                        : StopCompletionKind::kIgnore;
    }
    return StopCompletionKind::kError;
}

}  // namespace FarLib::rdma
