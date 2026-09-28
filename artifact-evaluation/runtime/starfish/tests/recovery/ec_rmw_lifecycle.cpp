// Bounded-bank lifecycle + RS delta math. Registers memory on mlx5_1, but
// posts NO network WRs: a local mock validates partial-post bookkeeping.
#include "rdma/ec_rmw_client.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include <array>
#include <cstdio>
#include <cstring>

using namespace FarLib::rdma::ec_rmw;
static size_t accept_limit = 0;
static int fake_post(ibv_qp *, ibv_send_wr *wr, ibv_send_wr **bad) {
    size_t accepted = 0;
    while (wr && accepted < accept_limit) { wr = wr->next; ++accepted; }
    *bad = wr;
    return wr ? ENOMEM : 0;
}
static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "RMW_LIFECYCLE_FAIL %s\n", what); std::abort(); }
}
int main() {
    int count = 0;
    ibv_device **devices = ibv_get_device_list(&count);
    check(devices != nullptr, "device list");
    ibv_context *context = nullptr;
    for (int i = 0; i < count; ++i)
        if (std::strcmp(ibv_get_device_name(devices[i]), "mlx5_1") == 0)
            context = ibv_open_device(devices[i]);
    check(context != nullptr, "open compute22 mlx5_1");
    ibv_pd *pd = ibv_alloc_pd(context);
    check(pd != nullptr, "protection domain");
    {
        ClientTransport transport;
        transport.init(pd, 2);
        check(transport.exchange_count() == 2 * 4 * 64 * 3, "fixed layout");
        check(&transport.get(0, 0, 0, 0) != &transport.get(1, 0, 0, 0), "private owners");
        check(!ClientTransport::is_wr_id(1ull << 62), "ordinary read disjoint");
        check(!ClientTransport::is_wr_id(1ull << 61), "ordinary write disjoint");
        check(!ClientTransport::is_wr_id(3ull << 62), "recovery disjoint");
        std::array<Exchange *, 6> requests{};
        for (size_t i = 0; i < requests.size(); ++i) {
            requests[i] = &transport.get(0, 0, i / 3, i % 3);
            transport.arm_read(*requests[i], 4096 * (i + 1), 123, 4096);
        }
        ibv_context fake_context{};
        fake_context.ops.post_send = fake_post;
        ibv_qp fake_qp{};
        fake_qp.context = &fake_context;
        accept_limit = 2;
        check(transport.post(requests.data(), requests.size(), &fake_qp) == 2, "accepted prefix");
        check(requests[0]->request.posted && requests[1]->request.posted &&
              !requests[2]->request.posted, "only accepted prefix owned by NIC");
        accept_limit = 10;
        check(transport.post(requests.data() + 2, 4, &fake_qp) == 4, "retry suffix only");
        uint64_t old_id = requests[0]->request.wr_id;
        for (auto *request : requests) {
            ibv_wc wc{};
            wc.wr_id = request->request.wr_id;
            wc.status = IBV_WC_SUCCESS;
            check(transport.handle(wc), "receive read completion");
            check(request->request.load_state() == CompletionState::success, "read success");
            request->request.posted = false;
        }
        auto &x = *requests[0];
        transport.arm_write(x, 4096, 123, 4096);
        ibv_wc stale{};
        stale.wr_id = old_id;
        stale.status = IBV_WC_SUCCESS;
        check(transport.handle(stale), "stale namespace recognized");
        check(x.request.load_state() == CompletionState::pending, "stale CQE cannot finish new phase");
        ibv_wc failure{};
        failure.wr_id = x.request.wr_id;
        failure.status = IBV_WC_WR_FLUSH_ERR;
        check(transport.handle(failure), "error completion ownership");
        check(x.request.load_state() == CompletionState::error, "error is terminal");
        failure.status = IBV_WC_SUCCESS;
        transport.handle(failure);
        check(x.request.load_state() == CompletionState::error, "duplicate cannot change terminal status");
        std::array<uint8_t, 4000> source{};
        ibv_mr *source_mr = ibv_reg_mr(pd, source.data(), source.size(), IBV_ACCESS_LOCAL_WRITE);
        check(source_mr != nullptr, "register source");
        transport.arm_write(x, 4096, 123, 4096, source.data(), source_mr->lkey, source.size());
        check(x.request.wr.num_sge == 2 && x.request.sges[0].length == 4000 &&
              x.request.sges[1].length == 96, "zero-copy data plus registered padding");
        check(ibv_dereg_mr(source_mr) == 0, "release mock source MR");
        std::printf("RMW_BANK_SMOKE registered_bytes=%zu owners=2 batch_depth=4 batch_objects=64 network_wrs=0\n",
                    transport.registered_bytes());
    }
    check(ibv_dealloc_pd(pd) == 0, "release protection domain");
    check(ibv_close_device(context) == 0, "close device");
    ibv_free_device_list(devices);

    alignas(64) uint8_t data[4][4096]{}, before[2][4096]{}, expected[2][4096]{};
    alignas(64) uint8_t delta[4096]{}, parity_delta[2][4096]{}, replacement[4096]{};
    for (size_t size : {size_t{8}, size_t{64}, size_t{512}, size_t{4096}}) {
        for (uint8_t changed = 0; changed < 4; ++changed) {
            for (size_t s = 0; s < 4; ++s)
                for (size_t b = 0; b < size; ++b) data[s][b] = uint8_t(s * 71 + b * 13);
            const void *inputs[4] = {data[0], data[1], data[2], data[3]};
            void *old_parities[2] = {before[0], before[1]};
            void *new_parities[2] = {expected[0], expected[1]};
            check(FarLib::cache::small_object_stripe_encode_shards(inputs, old_parities, size), "old encode");
            for (size_t b = 0; b < size; ++b) {
                replacement[b] = uint8_t(199 + b * 17);
                delta[b] = data[changed][b] ^ replacement[b];
            }
            check(FarLib::cache::small_object_stripe_encode_parity_delta(changed, delta, size,
                      parity_delta[0], parity_delta[1]), "delta ISA-L encode");
            std::memcpy(data[changed], replacement, size);
            check(FarLib::cache::small_object_stripe_encode_shards(inputs, new_parities, size), "new encode");
            for (size_t p = 0; p < 2; ++p)
                for (size_t b = 0; b < size; ++b)
                    check(uint8_t(before[p][b] ^ parity_delta[p][b]) == expected[p][b], "absolute parity update");
        }
    }
    std::puts("RMW_LIFECYCLE_AND_DELTA_PASS");
}
