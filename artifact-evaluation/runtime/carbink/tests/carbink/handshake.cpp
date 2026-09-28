#include "rdma/carbink_handshake.hpp"
#include <array>
#include <cassert>
#include <cstdio>
using namespace FarLib::rdma;

int main() {
    constexpr uint16_t count = 37;
    std::array<std::byte, sizeof(ClientConnectionInfo) + 4 * count> packed{};
    auto *client = reinterpret_cast<ClientConnectionInfo *>(packed.data());
    client->server_buffer_size = 10118758400ull;
    client->psn = 0x123456;
    client->lid = 0x1234;
    client->qp_count = count;
    client->max_rd_atomic = 16;
    client->mtu = IBV_MTU_4096;
    for (size_t i = 0; i < count; ++i) client->qpn[i] = 0x4000 + i;
    const auto wire = compact::encode_legacy_client_connection(*client);
    assert(wire.size() == 24 + 4 * count);
    compact::LegacyClientConnectionInfo header{};
    std::memcpy(&header, wire.data(), sizeof(header));
    assert(header.server_buffer_size == client->server_buffer_size);
    assert(header.psn == client->psn && header.lid == client->lid);
    assert(header.qp_count == count && header.max_rd_atomic == 16);
    assert(header.mtu == IBV_MTU_4096);
    for (size_t i = 0; i < count; ++i) {
        uint32_t qpn = 0;
        std::memcpy(&qpn, wire.data() + 24 + 4 * i, 4);
        assert(qpn == 0x4000 + i);
    }
    // Old server's full response, including its native padding.
    compact::LegacyServerConnectionInfo response{};
    response.psn = 0xabcdef;
    response.lid = 0x5678;
    response.qp_count = count;
    response.rkey = 0xabcd1234;
    response.addr = 0x1020304050607080ull;
    std::array<std::byte, 24 + 4 * count> reply{};
    std::memcpy(reply.data(), &response, sizeof(response));
    for (size_t i = 0; i < count; ++i) {
        uint32_t qpn = 0x5000 + i;
        std::memcpy(reply.data() + 24 + 4 * i, &qpn, 4);
    }
    std::array<std::byte, sizeof(ServerConnectionInfo) + 4 * count> decoded{};
    auto *server = reinterpret_cast<ServerConnectionInfo *>(decoded.data());
    assert(compact::decode_legacy_server_connection(
        reply.data(), reply.size(), count, server));
    assert(server->psn == response.psn && server->lid == response.lid);
    assert(server->rkey == response.rkey && server->addr == response.addr);
    assert(server->qp_count == count);
    for (size_t i = 0; i < count; ++i) assert(server->qpn[i] == 0x5000 + i);
    assert(!compact::decode_legacy_server_connection(
        reply.data(), reply.size() - 1, count, server));
    assert(!compact::decode_legacy_server_connection(
        reply.data(), reply.size(), count - 1, server));
    reply[6] = std::byte{0}; reply[7] = std::byte{0};
    assert(!compact::decode_legacy_server_connection(
        reply.data(), reply.size(), count, server));
    std::puts("CARBINK_HANDSHAKE_PASS packed=21/20 legacy=24/24 qps=37");
}
