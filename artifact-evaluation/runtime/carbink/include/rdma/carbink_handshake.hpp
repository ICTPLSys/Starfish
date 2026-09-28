#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "rdma/rdma.hpp"

namespace FarLib::rdma::compact {
// The original Carbink TCP bootstrap used native x86-64 alignment, unlike
// Hydra's packed 21/20-byte headers. Keep Hydra's wire layout unchanged.
struct LegacyClientConnectionInfo {
    size_t server_buffer_size;
    uint32_t psn;
    uint16_t lid;
    uint16_t qp_count;
    uint8_t max_rd_atomic;
    int mtu;
};
struct LegacyServerConnectionInfo {
    uint32_t psn;
    uint16_t lid;
    uint16_t qp_count;
    uint32_t rkey;
    uint64_t addr;
};
static_assert(sizeof(LegacyClientConnectionInfo) == 24);
static_assert(offsetof(LegacyClientConnectionInfo, mtu) == 20);
static_assert(sizeof(LegacyServerConnectionInfo) == 24);
static_assert(offsetof(LegacyServerConnectionInfo, addr) == 16);
static_assert(sizeof(ClientConnectionInfo) == 21);
static_assert(sizeof(ServerConnectionInfo) == 20);

inline std::vector<std::byte> encode_legacy_client_connection(
    const ClientConnectionInfo &packed) {
    LegacyClientConnectionInfo legacy{};
    legacy.server_buffer_size = packed.server_buffer_size;
    legacy.psn = packed.psn;
    legacy.lid = packed.lid;
    legacy.qp_count = packed.qp_count;
    legacy.max_rd_atomic = packed.max_rd_atomic;
    legacy.mtu = packed.mtu;
    std::vector<std::byte> bytes(sizeof(legacy) +
                                sizeof(uint32_t) * packed.qp_count);
    std::memcpy(bytes.data(), &legacy, sizeof(legacy));
    std::memcpy(bytes.data() + sizeof(legacy), packed.qpn,
                sizeof(uint32_t) * packed.qp_count);
    return bytes;
}

inline bool decode_legacy_server_connection(
    const std::byte *bytes, size_t size, uint16_t expected_qps,
    ServerConnectionInfo *packed) {
    if (size != sizeof(LegacyServerConnectionInfo) +
                    sizeof(uint32_t) * expected_qps || packed == nullptr)
        return false;
    LegacyServerConnectionInfo legacy{};
    std::memcpy(&legacy, bytes, sizeof(legacy));
    if (legacy.qp_count != expected_qps) return false;
    packed->psn = legacy.psn;
    packed->lid = legacy.lid;
    packed->qp_count = legacy.qp_count;
    packed->rkey = legacy.rkey;
    packed->addr = legacy.addr;
    std::memcpy(packed->qpn, bytes + sizeof(legacy),
                sizeof(uint32_t) * expected_qps);
    return true;
}
} // namespace FarLib::rdma::compact
