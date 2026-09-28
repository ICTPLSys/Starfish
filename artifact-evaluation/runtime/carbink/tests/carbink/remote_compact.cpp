// Real two-host Carbink compact protocol smoke test.
//
// This is deliberately an integration test, not a fake transport test.  It
// requires six configured servers and a real ClientControl/legacy Carbink
// server pair.  It is compiled here but never started by the build audit.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "rdma/client.hpp"
#include "rdma/config.hpp"
#include "utils/control.hpp"
#include "hydra/page_codec.hpp"
#include "hydra/page_layout.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {

namespace compact = FarLib::rdma::compact;
using FarLib::rdma::Client;
using FarLib::rdma::ClientControl;
using FarLib::rdma::Configure;

constexpr size_t kStripeCount = 32;
constexpr size_t kDataSlots = 4;
constexpr size_t kParitySlots = 2;
constexpr size_t kSpanBytes = FarLib::hydra::kPageBytes;
constexpr size_t kSlotStride = 2 * kSpanBytes;
constexpr uint64_t kSourceBase = 16ull * 1024 * 1024;
constexpr uint64_t kTargetBase = 256ull * 1024 * 1024;
constexpr size_t kBufferOffset = 2ull * 1024 * 1024;
constexpr uint64_t kInitWrBase = UINT64_C(0x5100000000000000);
constexpr uint64_t kMoveWrBase = UINT64_C(0x5200000000000000);
constexpr uint64_t kRollbackWrBase = UINT64_C(0x5300000000000000);

[[noreturn]] void fail(const char *message) {
    std::cerr << "CARBINK_REMOTE_COMPACT_FAIL " << message << std::endl;
    std::exit(2);
}

struct BufferLayout {
    uint8_t *base = nullptr;
    size_t source_data_offset = 0;
    size_t expected_parity_offset = 0;
    size_t zero_offset = 0;
    size_t scratch_offset = 0;
    size_t canary_offset = 0;

    uint8_t *source_data(size_t stripe, size_t slot) const {
        const size_t index = stripe * kDataSlots + slot;
        return base + source_data_offset + index * kSpanBytes;
    }
    uint8_t *expected_parity(size_t stripe, size_t parity) const {
        return base + expected_parity_offset +
               (stripe * kParitySlots + parity) * kSpanBytes;
    }
    uint8_t *zero_span() const { return base + zero_offset; }
    uint8_t *scratch() const { return base + scratch_offset; }
    uint8_t *canary() const { return base + canary_offset; }

    size_t required_bytes() const {
        return canary_offset + kSpanBytes;
    }
};

struct RemoteLayout {
    const char *name = nullptr;
    uint64_t parity_delta0 = 0;
    uint64_t parity_delta1 = 0;

    bool unequal_offsets() const {
        return parity_delta0 != 0 || parity_delta1 != 0;
    }
    uint64_t source_offset(size_t stripe) const {
        return kSourceBase + stripe * kSlotStride;
    }
    uint64_t target_data_offset(size_t stripe) const {
        return kTargetBase + stripe * kSlotStride;
    }
    uint64_t parity_offset(size_t stripe, size_t parity) const {
        return target_data_offset(stripe) +
               (parity == 0 ? parity_delta0 : parity_delta1);
    }
};

struct MoveCase {
    compact::CompactMoveDescriptor descriptor{};
    bool submitted = false;
    bool completed = false;
};
void check_ack_parser_cpu() {
    compact::CompactAckBatchMessage message{};
    message.type = compact::kCompactRpcAckBatch;
    for (size_t count : {size_t{9}, size_t{32}, size_t{1024}}) {
        message.payload_len =
            static_cast<uint32_t>(count * sizeof(uint64_t));
        auto *ids = reinterpret_cast<uint64_t *>(message.payload);
        for (size_t i = 0; i < count; ++i) ids[i] = i + 1;
        const size_t received =
            compact::compact_ack_header_bytes() + message.payload_len;
        assert(compact::compact_ack_message_shape_valid(message, received));
        assert(compact::compact_ack_message_count(message, received) == count);
    }
    message = compact::CompactAckBatchMessage{};
    message.type = compact::kCompactRpcAck;
    message.wr_id = 1;
    assert(compact::compact_ack_message_shape_valid(
        message, compact::compact_ack_header_bytes()));
    assert(compact::compact_ack_message_count(
               message, compact::compact_ack_header_bytes()) == 1);
    message.type = compact::kCompactRpcAckBatch;
    message.payload_len = 7;
    assert(!compact::compact_ack_message_shape_valid(
        message, compact::compact_ack_header_bytes() + 7));
    message.payload_len = sizeof(uint64_t);
    assert(!compact::compact_ack_message_shape_valid(
        message, compact::compact_ack_header_bytes()));
}

uint64_t remote_addr(const Configure &config, size_t endpoint,
                     uint64_t offset) {
    if (endpoint >= static_cast<size_t>(config.server_count) ||
        offset + kSpanBytes > config.server_buffer_size) {
        fail("remote address outside configured endpoint");
    }
    return static_cast<uint64_t>(endpoint) * config.server_buffer_size + offset;
}

void assert_same_endpoint(const Configure &config, size_t endpoint,
                          uint64_t source_offset, uint64_t target_offset) {
    const auto source = config.map_remote_addr(
        remote_addr(config, endpoint, source_offset));
    const auto target = config.map_remote_addr(
        remote_addr(config, endpoint, target_offset));
    assert(source.first == endpoint);
    assert(target.first == endpoint);
    assert(source.second == source_offset);
    assert(target.second == target_offset);
}

size_t poll_data_cq(Client *client, ibv_wc *work_completions,
                    size_t capacity) {
    // check_cq_with_idx() polls every endpoint's ordinary data CQ.  Its
    // route_carbink_completions() call invokes Client::consume_carbink_completion
    // for SEND/RECV completions and leaves ordinary WRITE/READ CQEs in the
    // returned array for this test to inspect.
    return client->check_cq_with_idx(work_completions, capacity, 0);
}

void reject_unexpected_completion(const ibv_wc &wc) {
    if (wc.status != IBV_WC_SUCCESS) {
        std::cerr << "unexpected CQE status=" << wc.status
                  << " opcode=" << wc.opcode << " wr_id=" << wc.wr_id
                  << std::endl;
        fail("ordinary CQE failed");
    }
    if (wc.opcode != IBV_WC_RDMA_WRITE &&
        wc.opcode != IBV_WC_RDMA_READ) {
        std::cerr << "unexpected ordinary opcode=" << wc.opcode
                  << " wr_id=" << wc.wr_id << std::endl;
        fail("unexpected ordinary CQE");
    }
}

void wait_one_ordinary(Client *client, uint64_t wr_id, uint32_t expected_opcode) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    ibv_wc completions[64]{};
    for (;;) {
        const size_t count = poll_data_cq(client, completions, 64);
        for (size_t i = 0; i < count; ++i) {
            reject_unexpected_completion(completions[i]);
            if (completions[i].wr_id != wr_id) continue;
            if (completions[i].opcode != expected_opcode)
                fail("ordinary CQE opcode mismatch");
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            fail("ordinary CQE timeout");
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void post_write_and_wait(Client *client, const Configure &config,
                         uint64_t address, uint8_t *payload,
                         uint64_t wr_id) {
    auto [endpoint, offset] = config.map_remote_addr(address);
    (void)offset;
    if (!client->post_write_with_addr(address, payload, kSpanBytes, wr_id,
                                      wr_id, 0)) {
        fail("initialization WRITE was rejected");
    }
    assert(endpoint < static_cast<size_t>(config.server_count));
    wait_one_ordinary(client, wr_id, IBV_WC_RDMA_WRITE);
}

void read_and_check(Client *client, const Configure &config, uint64_t address,
                    uint8_t *scratch, const uint8_t *expected) {
    if (!client->post_read_with_addr(address, scratch, kSpanBytes,
                                     kInitWrBase ^ address, address, 0)) {
        fail("verification READ was rejected");
    }
    wait_one_ordinary(client, kInitWrBase ^ address, IBV_WC_RDMA_READ);
    if (std::memcmp(scratch, expected, kSpanBytes) != 0) {
        const auto mapped = config.map_remote_addr(address);
        size_t byte=0;
        while (byte<kSpanBytes && scratch[byte]==expected[byte]) ++byte;
        std::cerr << "CARBINK_REMOTE_BYTES_MISMATCH endpoint=" << mapped.first
                  << " offset=" << mapped.second << " byte=" << byte
                  << " expected=" << unsigned(expected[byte])
                  << " actual=" << unsigned(scratch[byte]) << std::endl;
        fail("remote bytes differ from expected");
    }
}

void fill_inputs(const Configure &config, const BufferLayout &buffers) {
    std::memset(buffers.zero_span(), 0, kSpanBytes);
    for (size_t byte = 0; byte < kSpanBytes; ++byte) {
        buffers.canary()[byte] = static_cast<uint8_t>(
            0xa5u ^ ((byte * 29u + 7u) & 0xffu));
    }
    for (size_t stripe = 0; stripe < kStripeCount; ++stripe) {
        const void *data[kDataSlots]{};
        for (size_t slot = 0; slot < kDataSlots; ++slot) {
            uint8_t *payload = buffers.source_data(stripe, slot);
            for (size_t byte = 0; byte < kSpanBytes; ++byte) {
                payload[byte] = static_cast<uint8_t>(
                    (17u + stripe * 29u + slot * 47u + byte * 13u) & 0xffu);
            }
            data[slot] = payload;
        }
        void *parity[] = {buffers.expected_parity(stripe, 0),
                          buffers.expected_parity(stripe, 1)};
        if (!FarLib::hydra::page_codec_encode_4plus2(data, parity, kSpanBytes))
            fail("Hydra full-stripe codec rejected inputs");
    }
    (void)config;
}

void initialize_remote_layout(Client *client, const Configure &config,
                              const BufferLayout &buffers,
                              const RemoteLayout &layout) {
    uint64_t wr_id = kInitWrBase;
    for (size_t stripe = 0; stripe < kStripeCount; ++stripe) {
        const uint64_t source_offset = layout.source_offset(stripe);
        const uint64_t target_offset = layout.target_data_offset(stripe);
        for (size_t slot = 0; slot < kDataSlots; ++slot) {
            assert_same_endpoint(config, slot, source_offset, target_offset);
            post_write_and_wait(
                client, config, remote_addr(config, slot, source_offset),
                buffers.source_data(stripe, slot), ++wr_id);
            post_write_and_wait(
                client, config, remote_addr(config, slot, target_offset),
                buffers.zero_span(), ++wr_id);
        }
        for (size_t parity = 0; parity < kParitySlots; ++parity) {
            const size_t endpoint = kDataSlots + parity;
            post_write_and_wait(
                client, config, remote_addr(config, endpoint, source_offset),
                buffers.expected_parity(stripe, parity), ++wr_id);
            post_write_and_wait(
                client, config,
                remote_addr(config, endpoint,
                            layout.parity_offset(stripe, parity)),
                buffers.zero_span(), ++wr_id);
            if (layout.unequal_offsets()) {
                // The legacy DATA_REQ rollback path derives parity from
                // data_offset. Keep that wrong address nonzero so a bad
                // rollback cannot hide by writing into an already-zero slot.
                post_write_and_wait(
                    client, config,
                    remote_addr(config, endpoint, target_offset),
                    buffers.canary(), ++wr_id);
            }
        }
    }
}

std::vector<MoveCase> make_moves(const Configure &config,
                                 const RemoteLayout &layout,
                                 bool rollback, size_t transport_qp) {
    std::vector<MoveCase> moves;
    moves.reserve(kStripeCount * kDataSlots);
    // Keep one endpoint lane's 32 moves adjacent: every eight accepted
    // descriptors form one transport batch, exercising the real 8-span limit.
    for (size_t slot = 0; slot < kDataSlots; ++slot) {
        for (size_t stripe = 0; stripe < kStripeCount; ++stripe) {
            const uint64_t source_offset = layout.source_offset(stripe);
            const uint64_t target_offset = layout.target_data_offset(stripe);
            MoveCase move;
            auto &d = move.descriptor;
            d.wr_id = (rollback ? kRollbackWrBase : kMoveWrBase) +
                      slot * kStripeCount + stripe + 1;
            d.src_offset = rollback ? target_offset : source_offset;
            d.dst_offset = target_offset;
            d.parity0_offset = layout.parity_offset(stripe, 0);
            d.parity1_offset = layout.parity_offset(stripe, 1);
            d.parity_endpoint0 = kDataSlots;
            d.parity_endpoint1 = kDataSlots + 1;
            d.data_slot = static_cast<uint16_t>(slot);
            d.ack_data_qp_idx = static_cast<uint16_t>(transport_qp);
            d.endpoint_idx = slot;
            d.qp_idx = transport_qp;
            d.init_from_zero = !rollback;
            assert_same_endpoint(config, slot, source_offset, target_offset);
            moves.push_back(move);
        }
    }
    return moves;
}

void submit_and_wait(Client *client, compact::CompactReqBatchTransport &transport,
                     std::vector<MoveCase> &moves, bool rollback) {
    for (auto &move : moves) {
        for (;;) {
            compact::CompactRequestHandle handle{};
            const auto status = rollback
                ? transport.submit_zero(move.descriptor, &handle)
                : transport.submit(move.descriptor, &handle);
            if (status == compact::CompactSubmitStatus::Accepted) {
                if (handle.wr_id != move.descriptor.wr_id)
                    fail("transport returned wrong request handle");
                move.submitted = true;
                break;
            }
            if (status == compact::CompactSubmitStatus::NeedFlush) {
                if (!transport.flush(move.descriptor.endpoint_idx,
                                     move.descriptor.qp_idx))
                    fail("transport lane flush failed");
            } else if (status == compact::CompactSubmitStatus::Backpressure) {
                // Polling here is also required to reclaim SEND slots before
                // more descriptors are admitted.
            } else {
                fail("transport rejected compact descriptor");
            }
            ibv_wc completions[64]{};
            const size_t count = poll_data_cq(client, completions, 64);
            for (size_t i = 0; i < count; ++i)
                reject_unexpected_completion(completions[i]);
        }
    }
    if (!transport.flush_all())
        fail("transport flush_all failed");

    size_t completed = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (completed != moves.size()) {
        ibv_wc completions[64]{};
        const size_t count = poll_data_cq(client, completions, 64);
        for (size_t i = 0; i < count; ++i)
            reject_unexpected_completion(completions[i]);

        for (auto &move : moves) {
            if (!move.submitted || move.completed) continue;
            uint32_t status = UINT32_MAX;
            if (!transport.final_ack_done(move.descriptor.wr_id, &status) ||
                !transport.send_complete(move.descriptor.wr_id)) {
                continue;
            }
            if (status != compact::kCompactRpcStatusOk)
                fail(rollback ? "rollback ACK failed" : "compact ACK failed");
            if (!transport.erase(move.descriptor.wr_id))
                fail("transport request erased before completion");
            move.completed = true;
            ++completed;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            fail(rollback ? "rollback completion timeout"
                          : "compact completion timeout");
        if (completed != moves.size())
            std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void verify_target_after_move(Client *client, const Configure &config,
                              const BufferLayout &buffers,
                              const RemoteLayout &layout) {
    for (size_t stripe = 0; stripe < kStripeCount; ++stripe) {
        const uint64_t target_offset = layout.target_data_offset(stripe);
        for (size_t slot = 0; slot < kDataSlots; ++slot) {
            read_and_check(client, config,
                           remote_addr(config, slot, target_offset),
                           buffers.scratch(), buffers.source_data(stripe, slot));
        }
        for (size_t parity = 0; parity < kParitySlots; ++parity) {
            const size_t endpoint = kDataSlots + parity;
            read_and_check(
                client, config,
                remote_addr(config, endpoint,
                            layout.parity_offset(stripe, parity)),
                buffers.scratch(), buffers.expected_parity(stripe, parity));
            if (layout.unequal_offsets()) {
                read_and_check(
                    client, config, remote_addr(config, endpoint, target_offset),
                    buffers.scratch(), buffers.canary());
            }
        }
    }
}

void verify_after_rollback(Client *client, const Configure &config,
                           const BufferLayout &buffers,
                           const RemoteLayout &layout) {
    std::array<uint8_t, kSpanBytes> zero{};
    for (size_t stripe = 0; stripe < kStripeCount; ++stripe) {
        const uint64_t source_offset = layout.source_offset(stripe);
        const uint64_t target_offset = layout.target_data_offset(stripe);
        for (size_t slot = 0; slot < kDataSlots; ++slot) {
            read_and_check(client, config,
                           remote_addr(config, slot, source_offset),
                           buffers.scratch(), buffers.source_data(stripe, slot));
            read_and_check(client, config,
                           remote_addr(config, slot, target_offset),
                           buffers.scratch(), zero.data());
        }
        for (size_t parity = 0; parity < kParitySlots; ++parity) {
            const size_t endpoint = kDataSlots + parity;
            read_and_check(
                client, config,
                remote_addr(config, endpoint, source_offset),
                buffers.scratch(), buffers.expected_parity(stripe, parity));
            read_and_check(
                client, config,
                remote_addr(config, endpoint,
                            layout.parity_offset(stripe, parity)),
                buffers.scratch(), zero.data());
            if (layout.unequal_offsets()) {
                // Explicit p0/p1 must be zero again; the old derived D
                // address is a canary and must never be touched.
                read_and_check(
                    client, config, remote_addr(config, endpoint, target_offset),
                    buffers.scratch(), buffers.canary());
            }
        }
    }
}

int run(const char *config_path) {
    check_ack_parser_cpu();
    Configure config;
    config.from_file(config_path);
    if (!config.is_carbink_mode() || config.server_count != 6)
        fail("config must select carbink/ec_span with six servers");
    if (config.client_buffer_size <
        kBufferOffset + 2 * kStripeCount * kDataSlots * kSpanBytes +
            kStripeCount * kParitySlots * kSpanBytes + 3 * kSpanBytes)
        fail("client buffer is too small for registered test payloads");

    FarLib::runtime_init(config, false);
    ClientControl *control = ClientControl::get_default();
    if (control == nullptr) fail("ClientControl initialization failed");
    const size_t logical_worker = config.compaction_client_base();
    Client *client = Client::get_specific_client(logical_worker);
    if (client == nullptr || client->get_endpoint_count() != 6)
        fail("client endpoint setup failed");
    std::cout << "CARBINK_REMOTE_COMPACT_SETUP logical_worker=" << logical_worker
              << " global_qp=" << client->carbink_qp_index(0)
              << " stripes=" << kStripeCount
              << " layouts=equal,unequal"
              << " move_updates=128 rollback_updates=128" << std::endl;

    BufferLayout buffers;
    buffers.base = static_cast<uint8_t *>(control->get_buffer()) + kBufferOffset;
    buffers.source_data_offset = 0;
    buffers.expected_parity_offset =
        kStripeCount * kDataSlots * kSpanBytes;
    buffers.zero_offset = buffers.expected_parity_offset +
                          kStripeCount * kParitySlots * kSpanBytes;
    buffers.scratch_offset = buffers.zero_offset + kSpanBytes;
    buffers.canary_offset = buffers.scratch_offset + kSpanBytes;
    fill_inputs(config, buffers);

    auto &transport = client->carbink_transport();
    if (!transport.ready()) fail("Carbink transport is not ready");
    const size_t transport_qp = client->carbink_qp_index(0);
    const std::array<RemoteLayout, 2> layouts{{
        {"equal-offset", 0, 0},
        {"unequal-offset", 64ull * 1024 * 1024,
                           128ull * 1024 * 1024},
    }};

    for (const auto &layout : layouts) {
        std::cout << "CARBINK_REMOTE_COMPACT_LAYOUT name=" << layout.name
                  << " stage=initialize" << std::endl;
        initialize_remote_layout(client, config, buffers, layout);

        std::cout << "CARBINK_REMOTE_COMPACT_LAYOUT name=" << layout.name
                  << " stage=move_submit updates=128" << std::endl;
        auto moves = make_moves(config, layout, false, transport_qp);
        submit_and_wait(client, transport, moves, false);
        verify_target_after_move(client, config, buffers, layout);
        std::cout << "CARBINK_REMOTE_COMPACT_LAYOUT name=" << layout.name
                  << " stage=move_verify PASS" << std::endl;

        std::cout << "CARBINK_REMOTE_COMPACT_LAYOUT name=" << layout.name
                  << " stage=rollback_submit updates=128" << std::endl;
        auto rollback = make_moves(config, layout, true, transport_qp);
        submit_and_wait(client, transport, rollback, true);
        verify_after_rollback(client, config, buffers, layout);
        std::cout << "CARBINK_REMOTE_COMPACT_LAYOUT name=" << layout.name
                  << " stage=rollback_verify PASS" << std::endl;
    }

    std::puts("CARBINK_REMOTE_COMPACT_PASS");
    FarLib::runtime_destroy();
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <config>" << std::endl;
        return 2;
    }
    return run(argv[1]);
}
