#ifdef NDEBUG
#undef NDEBUG
#endif
#include "rdma/ec2pc_compact_transport.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
namespace c = FarLib::rdma::compact;
int main() {
    for (uint64_t data : {uint64_t{0}, uint64_t{8192}, uint64_t{1}<<32}) {
        for (unsigned unequal=0; unequal<4; ++unequal) {
            c::CompactMoveDescriptor d{};
            d.wr_id=123; d.src_offset=data; d.dst_offset=data;
            d.parity0_offset=data+((unequal&1)?8192:0);
            d.parity1_offset=data+((unequal&2)?16384:0);
            d.parity_endpoint0=4; d.parity_endpoint1=5;
            d.data_slot=3; d.endpoint_idx=2;
            const auto type=c::compact_data_request_type(d);
            assert(type==(unequal?c::kCompactRpcDataReq:c::kCompactRpcDataReqBatch));
            const auto h=c::compact_data_request_header(d);
            assert(h.type==c::kCompactRpcDataReq && h.flags==0);
            assert(h.payload_len==8192 && h.wr_id==123 && h.data_slot==3);
            assert(h.data_offset==data && h.target_offset==data);
            assert(h.parity_offset0==d.parity0_offset);
            assert(h.parity_offset1==d.parity1_offset);
            assert(h.parity_endpoint0==4 && h.parity_endpoint1==5);
            assert(h.target_endpoint==2);
            c::CompactAckBatchMessage legacy{};
            std::memcpy(&legacy,&h,sizeof(h));
            assert(legacy.data_offset==data);
            assert(legacy.parity_offset0==d.parity0_offset);
            assert(legacy.parity_offset1==d.parity1_offset);
            d.init_from_zero=true;
            assert(c::compact_data_request_header(d).flags==
                   c::kCompactRpcFlagInitFromZero);
        }
    }
    assert(sizeof(c::CompactDataReqHeader)==72);
    assert(sizeof(c::CompactAckBatchMessage)==8320);
    assert(!c::compact_req_batch_type(c::kCompactRpcDataReq));
    assert(c::compact_req_batch_type(c::kCompactRpcCompactReqBatch));
    std::cout<<"CARBINK_ROLLBACK_LAYOUT_PASS cases=12 explicit_wire=8320\n";
}
