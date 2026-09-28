#include "rdma/shutdown_completion.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    constexpr uint64_t kStop = 0;
    constexpr uint32_t kControlQp = 77;
    constexpr uint32_t kOtherQp = 78;
    using FarLib::rdma::StopCompletionKind;
    using FarLib::rdma::classify_stop_completion;

    ibv_wc wc{};
    wc.wr_id = kStop;
    wc.qp_num = kControlQp;
    wc.status = IBV_WC_SUCCESS;
    wc.opcode = IBV_WC_SEND;
    assert(classify_stop_completion(wc, kStop, kControlQp) ==
           StopCompletionKind::kSuccess);

    wc.opcode = IBV_WC_RDMA_WRITE;
    assert(classify_stop_completion(wc, kStop, kControlQp) ==
           StopCompletionKind::kIgnore);

    wc.status = IBV_WC_RETRY_EXC_ERR;
    wc.opcode = static_cast<ibv_wc_opcode>(0xffff);
    assert(classify_stop_completion(wc, kStop, kControlQp) ==
           StopCompletionKind::kError);

    wc.qp_num = kOtherQp;
    assert(classify_stop_completion(wc, kStop, kControlQp) ==
           StopCompletionKind::kIgnore);

    wc.qp_num = kControlQp;
    wc.wr_id = 123;
    assert(classify_stop_completion(wc, kStop, kControlQp) ==
           StopCompletionKind::kIgnore);

    std::cout << "shutdown completion classifier tests passed" << std::endl;
}
