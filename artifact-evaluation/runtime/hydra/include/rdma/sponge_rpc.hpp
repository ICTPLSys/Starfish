#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/config.hpp"

namespace FarLib::rdma {

constexpr uint16_t kSpongeRpcMagic = 0x5B9E;
constexpr uint16_t kSpongeRpcVersion = 1;
constexpr size_t kSpongeSmallObjectMaxBytes = 4096;
constexpr size_t kSpongeBatchPayloadBytes = 8 * kSpongeSmallObjectMaxBytes;
constexpr size_t kSpongeMinSlotBytes = 8;
constexpr size_t kSpongeBatchMaxRecords =
    kSpongeBatchPayloadBytes / kSpongeMinSlotBytes;
constexpr size_t kSpongeDataShards = 4;
constexpr size_t kSpongeParityShards = 2;
constexpr size_t kSpongeRpcQueueDepth = 4;
constexpr size_t kSpongeClientCommitSendQueueDepth = 4;
constexpr size_t kSpongeClientAckRecvQueueDepth = 16;
constexpr size_t kSpongeServerAckQueueDepth = 64;
constexpr size_t kSpongePeerRpcQueueDepth = 64;
constexpr uint8_t kSpongeGfPrimitivePolynomial = 0x1d;
constexpr size_t kSpongeGfPairLutSize = 1u << 16;
static_assert(kSpongeBatchMaxRecords <= UINT16_MAX);

inline size_t sponge_env_depth_or_default(const char *name,
                                          size_t default_value,
                                          size_t max_value = 0) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0') || parsed == 0) {
        return default_value;
    }
    size_t depth = static_cast<size_t>(parsed);
    if (max_value != 0 && depth > max_value) {
        depth = max_value;
    }
    return depth;
}

enum SpongeRpcType : uint16_t {
    SPONGE_RPC_INVALID = 0,
    SPONGE_RPC_COMMIT_BATCH = 1,
    SPONGE_RPC_PARITY_APPLY_BATCH = 2,
    SPONGE_RPC_ACK_BATCH = 3,
};

enum SpongeCommitFlags : uint16_t {
    SPONGE_COMMIT_FLAG_REUSED_DEAD = 1u << 0,
};

enum SpongeRpcStatus : uint16_t {
    SPONGE_RPC_STATUS_OK = 0,
    SPONGE_RPC_STATUS_INVALID = 1,
    SPONGE_RPC_STATUS_RETRY = 2,
};

enum SpongeParityApplyFlags : uint8_t {
    SPONGE_PARITY_APPLY_FLAG_BATCH_ACK = 1u << 0,
};

enum SpongeAckFlags : uint16_t {
    SPONGE_ACK_FLAG_PARITY_BATCH = 1u << 0,
};

inline bool sponge_slot_size_supported(size_t slot_size) {
    return slot_size > 0 && slot_size <= kSpongeSmallObjectMaxBytes;
}

inline bool sponge_batch_payload_size_supported(size_t byte_count) {
    return byte_count > 0 && byte_count <= kSpongeBatchPayloadBytes;
}

struct SpongeRpcHeader {
    uint16_t magic = kSpongeRpcMagic;
    uint16_t version = kSpongeRpcVersion;
    uint16_t type = SPONGE_RPC_INVALID;
    uint16_t record_count = 0;
    uint32_t payload_bytes = 0;
    uint32_t reserved = 0;
};

struct alignas(64) SpongeCommitRecord {
    uint64_t wr_id = 0;
    uint64_t data_offset = 0;
    uint64_t parity_offset[2] = {0, 0};
    uint64_t stripe_id = 0;
    uint32_t slot_id = 0;
    uint16_t obj_size = 0;
    uint16_t slot_size = 0;
    uint16_t bin = 0;
    uint16_t parity_endpoint[2] = {0, 0};
    uint8_t data_shard_idx = 0;
    uint8_t flags = 0;
    uint8_t payload[kSpongeSmallObjectMaxBytes] = {0};
};

struct SpongeCommitDescriptor {
    uint64_t wr_id = 0;
    uint64_t data_offset = 0;
    uint64_t parity_offset[2] = {0, 0};
    uint64_t stripe_id = 0;
    uint32_t slot_id = 0;
    uint32_t payload_offset = 0;
    uint16_t obj_size = 0;
    uint16_t slot_size = 0;
    uint16_t bin = 0;
    uint16_t parity_endpoint[2] = {0, 0};
    uint8_t data_shard_idx = 0;
    uint8_t flags = 0;
};

struct alignas(64) SpongeParityApplyRecord {
    uint64_t wr_id = 0;
    uint64_t parity_offset = 0;
    uint16_t slot_size = 0;
    uint8_t payload[kSpongeSmallObjectMaxBytes] = {0};
};

struct SpongeParityApplyDescriptor {
    uint64_t wr_id = 0;
    uint64_t parity_offset = 0;
    uint32_t payload_offset = 0;
    uint16_t slot_size = 0;
    uint8_t parity_idx = 0;
    uint8_t flags = 0;
};

struct SpongeAckRecord {
    uint64_t wr_id = 0;
    uint16_t status = SPONGE_RPC_STATUS_OK;
    uint16_t flags = 0;
    uint32_t reserved = 0;
};

inline bool sponge_commit_descriptor_valid(
    const SpongeCommitDescriptor &desc) {
    return sponge_slot_size_supported(desc.slot_size) &&
           desc.obj_size <= desc.slot_size &&
           desc.data_shard_idx < kSpongeDataShards;
}

inline bool sponge_parity_apply_descriptor_valid(
    const SpongeParityApplyDescriptor &desc) {
    return sponge_slot_size_supported(desc.slot_size) &&
           desc.parity_idx < kSpongeParityShards;
}

struct SpongePendingCommit {
    void *local_addr = nullptr;
    void *entry = nullptr;
    uint64_t obj_id = 0;
    uint64_t remote_addr = 0;
    uint32_t obj_size = 0;
    uint16_t client_idx = 0;
    uint16_t endpoint_idx = 0;
    uint16_t qp_idx = 0;
    std::atomic<uint8_t> completed{0};
    uint16_t status = SPONGE_RPC_STATUS_INVALID;

    uint64_t wr_id() const {
        return reinterpret_cast<uint64_t>(this);
    }

    void reset(void *local, void *entry_ptr, uint64_t object_id,
               uint64_t remote, uint32_t size, uint16_t client,
               uint16_t endpoint, uint16_t qp) {
        local_addr = local;
        entry = entry_ptr;
        obj_id = object_id;
        remote_addr = remote;
        obj_size = size;
        client_idx = client;
        endpoint_idx = endpoint;
        qp_idx = qp;
        status = SPONGE_RPC_STATUS_INVALID;
        completed.store(0, std::memory_order_relaxed);
    }

    bool complete(uint16_t completion_status) {
        uint8_t expected = 0;
        if (!completed.compare_exchange_strong(
                expected, 2, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return false;
        }
        status = completion_status;
        completed.store(1, std::memory_order_release);
        return true;
    }
};

inline size_t sponge_rpc_descriptor_bytes(uint16_t type,
                                          size_t record_count) {
    switch (type) {
    case SPONGE_RPC_COMMIT_BATCH:
        return record_count * sizeof(SpongeCommitDescriptor);
    case SPONGE_RPC_PARITY_APPLY_BATCH:
        return record_count * sizeof(SpongeParityApplyDescriptor);
    case SPONGE_RPC_ACK_BATCH:
        return record_count * sizeof(SpongeAckRecord);
    default:
        return 0;
    }
}

inline size_t sponge_rpc_meta_bytes(const SpongeRpcHeader &header) {
    return sizeof(SpongeRpcHeader) +
           sponge_rpc_descriptor_bytes(header.type, header.record_count);
}

inline size_t sponge_rpc_wire_bytes(const SpongeRpcHeader &header) {
    return sponge_rpc_meta_bytes(header) + header.payload_bytes;
}

inline size_t sponge_rpc_max_wire_bytes(uint16_t type) {
    SpongeRpcHeader header;
    header.type = type;
    header.record_count = static_cast<uint16_t>(kSpongeBatchMaxRecords);
    header.payload_bytes =
        type == SPONGE_RPC_ACK_BATCH ? 0 : kSpongeBatchPayloadBytes;
    return sponge_rpc_wire_bytes(header);
}

constexpr size_t kSpongeCommitBatchMaxWireBytes =
    sizeof(SpongeRpcHeader) +
    kSpongeBatchMaxRecords * sizeof(SpongeCommitDescriptor) +
    kSpongeBatchPayloadBytes;
constexpr size_t kSpongeParityApplyBatchMaxWireBytes =
    sizeof(SpongeRpcHeader) +
    kSpongeBatchMaxRecords * sizeof(SpongeParityApplyDescriptor) +
    kSpongeBatchPayloadBytes;
constexpr size_t kSpongeAckBatchMaxWireBytes =
    sizeof(SpongeRpcHeader) +
    kSpongeBatchMaxRecords * sizeof(SpongeAckRecord);

inline bool sponge_rpc_header_valid(const SpongeRpcHeader &header,
                                    uint16_t expected_type) {
    return header.magic == kSpongeRpcMagic &&
           header.version == kSpongeRpcVersion &&
           header.type == expected_type &&
           header.record_count <= kSpongeBatchMaxRecords &&
           header.payload_bytes <= kSpongeBatchPayloadBytes;
}

struct SpongeWireView {
    const SpongeRpcHeader *header = nullptr;
    const uint8_t *descriptors = nullptr;
    const uint8_t *payload = nullptr;
    size_t meta_bytes = 0;
    size_t wire_bytes = 0;

    template <typename Descriptor>
    const Descriptor *descriptor_array() const {
        return reinterpret_cast<const Descriptor *>(descriptors);
    }
};

inline bool sponge_rpc_parse_wire(const void *wire, size_t byte_len,
                                  uint16_t expected_type,
                                  SpongeWireView &view) {
    view = SpongeWireView{};
    if (wire == nullptr || byte_len < sizeof(SpongeRpcHeader)) {
        return false;
    }
    const auto *header = reinterpret_cast<const SpongeRpcHeader *>(wire);
    if (!sponge_rpc_header_valid(*header, expected_type)) {
        return false;
    }
    size_t meta_bytes = sponge_rpc_meta_bytes(*header);
    size_t wire_bytes = meta_bytes + header->payload_bytes;
    if (meta_bytes > byte_len || wire_bytes > byte_len ||
        wire_bytes > sponge_rpc_max_wire_bytes(expected_type)) {
        return false;
    }
    view.header = header;
    view.descriptors = reinterpret_cast<const uint8_t *>(wire) +
                       sizeof(SpongeRpcHeader);
    view.payload = reinterpret_cast<const uint8_t *>(wire) + meta_bytes;
    view.meta_bytes = meta_bytes;
    view.wire_bytes = wire_bytes;
    return true;
}

inline bool sponge_rpc_pack_wire(const SpongeRpcHeader &header,
                                 const void *descriptors,
                                 const void *payload, uint8_t *dst,
                                 size_t dst_capacity) {
    if (!sponge_rpc_header_valid(header, header.type) || dst == nullptr) {
        return false;
    }
    size_t descriptor_bytes =
        sponge_rpc_descriptor_bytes(header.type, header.record_count);
    size_t meta_bytes = sizeof(SpongeRpcHeader) + descriptor_bytes;
    size_t wire_bytes = meta_bytes + header.payload_bytes;
    if (wire_bytes > dst_capacity ||
        wire_bytes > sponge_rpc_max_wire_bytes(header.type)) {
        return false;
    }
    if (descriptor_bytes != 0 && descriptors == nullptr) {
        return false;
    }
    if (header.payload_bytes != 0 && payload == nullptr) {
        return false;
    }
    std::memcpy(dst, &header, sizeof(header));
    if (descriptor_bytes != 0) {
        std::memcpy(dst + sizeof(SpongeRpcHeader), descriptors,
                    descriptor_bytes);
    }
    if (header.payload_bytes != 0) {
        std::memcpy(dst + meta_bytes, payload, header.payload_bytes);
    }
    return true;
}

struct SpongeCommitBatchBuffer {
    SpongeRpcHeader header;
    std::array<SpongeCommitDescriptor, kSpongeBatchMaxRecords> descriptors;
    std::array<uint8_t, kSpongeBatchPayloadBytes> payload;

    SpongeCommitBatchBuffer() { reset(); }

    void reset() {
        header = SpongeRpcHeader{};
        header.type = SPONGE_RPC_COMMIT_BATCH;
    }

    bool append(const SpongeCommitRecord &record) {
        if (!sponge_slot_size_supported(record.slot_size) ||
            record.obj_size > record.slot_size ||
            record.data_shard_idx >= kSpongeDataShards ||
            header.record_count >= descriptors.size() ||
            header.payload_bytes + record.slot_size >
                kSpongeBatchPayloadBytes) {
            return false;
        }

        uint16_t idx = header.record_count;
        auto &desc = descriptors[idx];
        desc = SpongeCommitDescriptor{};
        desc.wr_id = record.wr_id;
        desc.data_offset = record.data_offset;
        desc.parity_offset[0] = record.parity_offset[0];
        desc.parity_offset[1] = record.parity_offset[1];
        desc.stripe_id = record.stripe_id;
        desc.slot_id = record.slot_id;
        desc.payload_offset = header.payload_bytes;
        desc.obj_size = record.obj_size;
        desc.slot_size = record.slot_size;
        desc.bin = record.bin;
        desc.parity_endpoint[0] = record.parity_endpoint[0];
        desc.parity_endpoint[1] = record.parity_endpoint[1];
        desc.data_shard_idx = record.data_shard_idx;
        desc.flags = record.flags;
        std::memcpy(payload.data() + desc.payload_offset, record.payload,
                    record.slot_size);
        header.payload_bytes += record.slot_size;
        header.record_count++;
        return true;
    }

    const uint8_t *payload_for(const SpongeCommitDescriptor &desc) const {
        if (desc.payload_offset + desc.slot_size > header.payload_bytes) {
            return nullptr;
        }
        return payload.data() + desc.payload_offset;
    }

    bool expand(size_t idx, SpongeCommitRecord &record) const {
        if (idx >= header.record_count ||
            !sponge_rpc_header_valid(header, SPONGE_RPC_COMMIT_BATCH)) {
            return false;
        }
        const auto &desc = descriptors[idx];
        const uint8_t *src = payload_for(desc);
        if (src == nullptr || !sponge_slot_size_supported(desc.slot_size) ||
            desc.obj_size > desc.slot_size ||
            desc.data_shard_idx >= kSpongeDataShards) {
            return false;
        }
        record = SpongeCommitRecord{};
        record.wr_id = desc.wr_id;
        record.data_offset = desc.data_offset;
        record.parity_offset[0] = desc.parity_offset[0];
        record.parity_offset[1] = desc.parity_offset[1];
        record.stripe_id = desc.stripe_id;
        record.slot_id = desc.slot_id;
        record.obj_size = desc.obj_size;
        record.slot_size = desc.slot_size;
        record.bin = desc.bin;
        record.parity_endpoint[0] = desc.parity_endpoint[0];
        record.parity_endpoint[1] = desc.parity_endpoint[1];
        record.data_shard_idx = desc.data_shard_idx;
        record.flags = desc.flags;
        std::memcpy(record.payload, src, desc.slot_size);
        return true;
    }

    size_t meta_bytes() const { return sponge_rpc_meta_bytes(header); }

    size_t wire_bytes() const { return sponge_rpc_wire_bytes(header); }

    bool pack_wire(uint8_t *dst, size_t dst_capacity) const {
        return sponge_rpc_pack_wire(header, descriptors.data(), payload.data(),
                                    dst, dst_capacity);
    }
};

struct SpongeCommitBatchView {
    SpongeWireView wire;

    bool parse(const void *bytes, size_t byte_len) {
        return sponge_rpc_parse_wire(bytes, byte_len,
                                     SPONGE_RPC_COMMIT_BATCH, wire);
    }

    const SpongeCommitDescriptor *descriptors() const {
        return wire.descriptor_array<SpongeCommitDescriptor>();
    }

    const uint8_t *payload_for(const SpongeCommitDescriptor &desc) const {
        if (wire.header == nullptr ||
            desc.payload_offset + desc.slot_size >
                wire.header->payload_bytes) {
            return nullptr;
        }
        return wire.payload + desc.payload_offset;
    }

    bool expand(size_t idx, SpongeCommitRecord &record) const {
        if (wire.header == nullptr || idx >= wire.header->record_count) {
            return false;
        }
        const auto &desc = descriptors()[idx];
        const uint8_t *src = payload_for(desc);
        if (src == nullptr || !sponge_slot_size_supported(desc.slot_size) ||
            desc.obj_size > desc.slot_size ||
            desc.data_shard_idx >= kSpongeDataShards) {
            return false;
        }
        record = SpongeCommitRecord{};
        record.wr_id = desc.wr_id;
        record.data_offset = desc.data_offset;
        record.parity_offset[0] = desc.parity_offset[0];
        record.parity_offset[1] = desc.parity_offset[1];
        record.stripe_id = desc.stripe_id;
        record.slot_id = desc.slot_id;
        record.obj_size = desc.obj_size;
        record.slot_size = desc.slot_size;
        record.bin = desc.bin;
        record.parity_endpoint[0] = desc.parity_endpoint[0];
        record.parity_endpoint[1] = desc.parity_endpoint[1];
        record.data_shard_idx = desc.data_shard_idx;
        record.flags = desc.flags;
        std::memcpy(record.payload, src, desc.slot_size);
        return true;
    }
};

struct SpongeParityApplyBatchBuffer {
    SpongeRpcHeader header;
    std::array<SpongeParityApplyDescriptor, kSpongeBatchMaxRecords> descriptors;
    std::array<uint8_t, kSpongeBatchPayloadBytes> payload;

    SpongeParityApplyBatchBuffer() { reset(); }

    void reset() {
        header = SpongeRpcHeader{};
        header.type = SPONGE_RPC_PARITY_APPLY_BATCH;
    }

    bool append(const SpongeParityApplyRecord &record, uint8_t parity_idx) {
        if (!sponge_slot_size_supported(record.slot_size) ||
            header.record_count >= descriptors.size() ||
            header.payload_bytes + record.slot_size >
                kSpongeBatchPayloadBytes ||
            parity_idx >= kSpongeParityShards) {
            return false;
        }

        uint16_t idx = header.record_count;
        auto &desc = descriptors[idx];
        desc = SpongeParityApplyDescriptor{};
        desc.wr_id = record.wr_id;
        desc.parity_offset = record.parity_offset;
        desc.payload_offset = header.payload_bytes;
        desc.slot_size = record.slot_size;
        desc.parity_idx = parity_idx;
        std::memcpy(payload.data() + desc.payload_offset, record.payload,
                    record.slot_size);
        header.payload_bytes += record.slot_size;
        header.record_count++;
        return true;
    }

    uint8_t *append_uninitialized(uint64_t wr_id, uint64_t parity_offset,
                                  uint16_t slot_size, uint8_t parity_idx,
                                  uint8_t flags = 0) {
        if (!sponge_slot_size_supported(slot_size) ||
            header.record_count >= descriptors.size() ||
            header.payload_bytes + slot_size > kSpongeBatchPayloadBytes ||
            parity_idx >= kSpongeParityShards) {
            return nullptr;
        }

        uint16_t idx = header.record_count;
        auto &desc = descriptors[idx];
        desc = SpongeParityApplyDescriptor{};
        desc.wr_id = wr_id;
        desc.parity_offset = parity_offset;
        desc.payload_offset = header.payload_bytes;
        desc.slot_size = slot_size;
        desc.parity_idx = parity_idx;
        desc.flags = flags;
        uint8_t *dst = payload.data() + desc.payload_offset;
        header.payload_bytes += slot_size;
        header.record_count++;
        return dst;
    }

    const uint8_t *payload_for(
        const SpongeParityApplyDescriptor &desc) const {
        if (desc.payload_offset + desc.slot_size > header.payload_bytes) {
            return nullptr;
        }
        return payload.data() + desc.payload_offset;
    }

    bool expand(size_t idx, SpongeParityApplyRecord &record) const {
        if (idx >= header.record_count ||
            !sponge_rpc_header_valid(header,
                                     SPONGE_RPC_PARITY_APPLY_BATCH)) {
            return false;
        }
        const auto &desc = descriptors[idx];
        const uint8_t *src = payload_for(desc);
        if (src == nullptr || !sponge_slot_size_supported(desc.slot_size) ||
            desc.parity_idx >= kSpongeParityShards) {
            return false;
        }
        record = SpongeParityApplyRecord{};
        record.wr_id = desc.wr_id;
        record.parity_offset = desc.parity_offset;
        record.slot_size = desc.slot_size;
        std::memcpy(record.payload, src, desc.slot_size);
        return true;
    }

    size_t meta_bytes() const { return sponge_rpc_meta_bytes(header); }

    size_t wire_bytes() const { return sponge_rpc_wire_bytes(header); }

    bool pack_wire(uint8_t *dst, size_t dst_capacity) const {
        return sponge_rpc_pack_wire(header, descriptors.data(), payload.data(),
                                    dst, dst_capacity);
    }
};

struct SpongeParityApplyBatchView {
    SpongeWireView wire;

    bool parse(const void *bytes, size_t byte_len) {
        return sponge_rpc_parse_wire(bytes, byte_len,
                                     SPONGE_RPC_PARITY_APPLY_BATCH, wire);
    }

    const SpongeParityApplyDescriptor *descriptors() const {
        return wire.descriptor_array<SpongeParityApplyDescriptor>();
    }

    const uint8_t *payload_for(
        const SpongeParityApplyDescriptor &desc) const {
        if (wire.header == nullptr ||
            desc.payload_offset + desc.slot_size >
                wire.header->payload_bytes) {
            return nullptr;
        }
        return wire.payload + desc.payload_offset;
    }

    bool expand(size_t idx, SpongeParityApplyRecord &record) const {
        if (wire.header == nullptr || idx >= wire.header->record_count) {
            return false;
        }
        const auto &desc = descriptors()[idx];
        const uint8_t *src = payload_for(desc);
        if (src == nullptr || !sponge_slot_size_supported(desc.slot_size) ||
            desc.parity_idx >= kSpongeParityShards) {
            return false;
        }
        record = SpongeParityApplyRecord{};
        record.wr_id = desc.wr_id;
        record.parity_offset = desc.parity_offset;
        record.slot_size = desc.slot_size;
        std::memcpy(record.payload, src, desc.slot_size);
        return true;
    }
};

struct SpongeAckBatchBuffer {
    SpongeRpcHeader header;
    std::array<SpongeAckRecord, kSpongeBatchMaxRecords> records;

    SpongeAckBatchBuffer() { reset(); }

    void reset() {
        header = SpongeRpcHeader{};
        header.type = SPONGE_RPC_ACK_BATCH;
    }

    bool append(uint64_t wr_id, uint16_t status = SPONGE_RPC_STATUS_OK,
                uint16_t flags = 0) {
        if (wr_id == 0 || header.record_count >= records.size()) {
            return false;
        }
        auto &record = records[header.record_count];
        record = SpongeAckRecord{};
        record.wr_id = wr_id;
        record.status = status;
        record.flags = flags;
        header.record_count++;
        return true;
    }

    size_t meta_bytes() const { return sponge_rpc_meta_bytes(header); }

    size_t wire_bytes() const { return sponge_rpc_wire_bytes(header); }

    bool pack_wire(uint8_t *dst, size_t dst_capacity) const {
        return sponge_rpc_pack_wire(header, records.data(), nullptr, dst,
                                    dst_capacity);
    }
};

struct SpongeAckBatchView {
    SpongeWireView wire;

    bool parse(const void *bytes, size_t byte_len) {
        return sponge_rpc_parse_wire(bytes, byte_len, SPONGE_RPC_ACK_BATCH,
                                     wire);
    }

    const SpongeAckRecord *records() const {
        return wire.descriptor_array<SpongeAckRecord>();
    }
};

template <typename Batch, uint32_t MagicValue>
struct SpongeRpcBatchSlot {
    uint32_t magic = MagicValue;
    std::atomic<bool> in_use{false};
    uint16_t endpoint_idx = 0;
    uint16_t qp_idx = 0;
    Batch batch;

    void reset() {
        magic = MagicValue;
        endpoint_idx = 0;
        qp_idx = 0;
        batch.reset();
        in_use.store(false, std::memory_order_release);
    }
};

using SpongeClientCommitSendSlot =
    SpongeRpcBatchSlot<SpongeCommitBatchBuffer, 0x5b9e1001u>;
using SpongeServerCommitRecvSlot =
    SpongeRpcBatchSlot<SpongeCommitBatchBuffer, 0x5b9e1002u>;
using SpongeParityApplySendSlot =
    SpongeRpcBatchSlot<SpongeParityApplyBatchBuffer, 0x5b9e1003u>;
using SpongeParityApplyRecvSlot =
    SpongeRpcBatchSlot<SpongeParityApplyBatchBuffer, 0x5b9e1004u>;
using SpongeAckSendSlot =
    SpongeRpcBatchSlot<SpongeAckBatchBuffer, 0x5b9e1005u>;
using SpongeAckRecvSlot =
    SpongeRpcBatchSlot<SpongeAckBatchBuffer, 0x5b9e1006u>;

template <size_t MaxWireBytes, uint32_t MagicValue>
struct SpongeRpcWireSlot {
    uint32_t magic = MagicValue;
    std::atomic<bool> in_use{false};
    uint16_t endpoint_idx = 0;
    uint16_t qp_idx = 0;
    uint32_t byte_len = 0;
    std::atomic<uint32_t> profile_record_count{0};
    std::atomic<uint64_t> profile_post_time_ns{0};
    alignas(64) std::array<uint8_t, MaxWireBytes> bytes{};

    void reset() {
        magic = MagicValue;
        endpoint_idx = 0;
        qp_idx = 0;
        byte_len = 0;
        profile_record_count.store(0, std::memory_order_relaxed);
        profile_post_time_ns.store(0, std::memory_order_relaxed);
        in_use.store(false, std::memory_order_release);
    }
};

// Named magic constants so the client transport can verify slot ownership
// without duplicating the literals (the aliases below stay the definition).
inline constexpr uint32_t kSpongeClientCommitWireSendSlotMagic = 0x5b9e2001u;
inline constexpr uint32_t kSpongeAckWireRecvSlotMagic = 0x5b9e2006u;

using SpongeClientCommitWireSendSlot =
    SpongeRpcWireSlot<kSpongeCommitBatchMaxWireBytes,
                      kSpongeClientCommitWireSendSlotMagic>;
using SpongeServerCommitWireRecvSlot =
    SpongeRpcWireSlot<kSpongeCommitBatchMaxWireBytes, 0x5b9e2002u>;
using SpongeParityApplyWireSendSlot =
    SpongeRpcWireSlot<kSpongeParityApplyBatchMaxWireBytes, 0x5b9e2003u>;
using SpongeParityApplyWireRecvSlot =
    SpongeRpcWireSlot<kSpongeParityApplyBatchMaxWireBytes, 0x5b9e2004u>;
using SpongeAckWireSendSlot =
    SpongeRpcWireSlot<kSpongeAckBatchMaxWireBytes, 0x5b9e2005u>;
using SpongeAckWireRecvSlot =
    SpongeRpcWireSlot<kSpongeAckBatchMaxWireBytes, kSpongeAckWireRecvSlotMagic>;
using SpongePeerWireRecvSlot =
    SpongeRpcWireSlot<kSpongeParityApplyBatchMaxWireBytes, 0x5b9e2007u>;

template <typename Slot>
class SpongeRpcSlotPool {
private:
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<Slot *> free_slots_;
    std::mutex mutex_;

public:
    void init(size_t depth) {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.clear();
        free_slots_.clear();
        slots_.reserve(depth);
        free_slots_.reserve(depth);
        for (size_t i = 0; i < depth; i++) {
            auto slot = std::make_unique<Slot>();
            slot->reset();
            free_slots_.push_back(slot.get());
            slots_.push_back(std::move(slot));
        }
    }

    Slot *try_acquire(uint16_t endpoint_idx = 0, uint16_t qp_idx = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_slots_.empty()) {
            return nullptr;
        }
        Slot *slot = free_slots_.back();
        free_slots_.pop_back();
        slot->endpoint_idx = endpoint_idx;
        slot->qp_idx = qp_idx;
        slot->in_use.store(true, std::memory_order_release);
        return slot;
    }

    void release(Slot *slot) {
        if (slot == nullptr) return;
        slot->reset();
        std::lock_guard<std::mutex> lock(mutex_);
        free_slots_.push_back(slot);
    }

    size_t free_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_slots_.size();
    }

    size_t capacity() const { return slots_.size(); }
};

struct SpongeShardParityLut {
    uint8_t coef[kSpongeParityShards] = {0, 0};
    uint8_t mul[kSpongeParityShards][256] = {};
    uint16_t mul16[kSpongeParityShards][kSpongeGfPairLutSize] = {};
};

struct SpongeParityCodecCache {
    SpongeShardParityLut shard[kSpongeDataShards];

    SpongeParityCodecCache() {
        uint8_t matrix[(kSpongeDataShards + kSpongeParityShards) *
                       kSpongeDataShards] = {};
        gen_rs_matrix(matrix, kSpongeDataShards + kSpongeParityShards,
                      kSpongeDataShards);
        for (size_t data_shard = 0; data_shard < kSpongeDataShards;
             data_shard++) {
            auto &lut = shard[data_shard];
            for (size_t parity_idx = 0; parity_idx < kSpongeParityShards;
                 parity_idx++) {
                lut.coef[parity_idx] =
                    matrix[(kSpongeDataShards + parity_idx) *
                               kSpongeDataShards +
                           data_shard];
                for (size_t v = 0; v < 256; v++) {
                    lut.mul[parity_idx][v] = gf_mul(
                        lut.coef[parity_idx], static_cast<uint8_t>(v));
                }
                for (size_t v = 0; v < kSpongeGfPairLutSize; v++) {
                    uint16_t in_pair = static_cast<uint16_t>(v);
                    uint8_t in[sizeof(in_pair)] = {};
                    uint8_t out[sizeof(in_pair)] = {};
                    std::memcpy(in, &in_pair, sizeof(in_pair));
                    out[0] = lut.mul[parity_idx][in[0]];
                    out[1] = lut.mul[parity_idx][in[1]];
                    std::memcpy(&lut.mul16[parity_idx][v], out, sizeof(out));
                }
            }
        }
    }

private:
    static uint8_t gf_mul(uint8_t a, uint8_t b) {
        uint8_t result = 0;
        while (b != 0) {
            if ((b & 1u) != 0) {
                result ^= a;
            }
            bool carry = (a & 0x80u) != 0;
            a = static_cast<uint8_t>(a << 1);
            if (carry) {
                a ^= kSpongeGfPrimitivePolynomial;
            }
            b = static_cast<uint8_t>(b >> 1);
        }
        return result;
    }

    static uint8_t gf_pow(uint8_t a, size_t power) {
        uint8_t result = 1;
        for (size_t i = 0; i < power; i++) {
            result = gf_mul(result, a);
        }
        return result;
    }

    static void gen_rs_matrix(uint8_t *matrix, size_t rows, size_t cols) {
        assert(rows >= cols);
        for (size_t row = 0; row < cols; row++) {
            for (size_t col = 0; col < cols; col++) {
                matrix[row * cols + col] = (row == col) ? 1 : 0;
            }
        }
        for (size_t row = cols; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {
                matrix[row * cols + col] = gf_pow(
                    static_cast<uint8_t>(row - cols + 1), col);
            }
        }
    }
};

inline const SpongeParityCodecCache &sponge_parity_codec_cache() {
    static const SpongeParityCodecCache cache;
    return cache;
}

inline const SpongeShardParityLut &sponge_shard_parity_lut(
    uint8_t data_shard_idx) {
    assert(data_shard_idx < kSpongeDataShards);
    return sponge_parity_codec_cache().shard[data_shard_idx];
}

inline void sponge_encode_parity_bytes_with_lut(const uint8_t *src,
                                                size_t byte_count,
                                                const uint8_t *lut8,
                                                const uint16_t *lut16,
                                                uint8_t *dst) {
    size_t i = 0;
    for (; i + sizeof(uint16_t) <= byte_count; i += sizeof(uint16_t)) {
        uint16_t in_pair = 0;
        std::memcpy(&in_pair, src + i, sizeof(in_pair));
        uint16_t out_pair = lut16[in_pair];
        std::memcpy(dst + i, &out_pair, sizeof(out_pair));
    }
    for (; i < byte_count; i++) {
        dst[i] = lut8[src[i]];
    }
}

inline bool sponge_encode_parity_delta_bytes(uint8_t data_shard_idx,
                                             const uint8_t *data_delta,
                                             size_t byte_count,
                                             uint8_t *parity0_delta,
                                             uint8_t *parity1_delta) {
    if (!sponge_batch_payload_size_supported(byte_count) ||
        data_shard_idx >= kSpongeDataShards || data_delta == nullptr ||
        parity0_delta == nullptr || parity1_delta == nullptr) {
        return false;
    }
    const auto &lut = sponge_shard_parity_lut(data_shard_idx);
    if (lut.coef[0] == 1 && lut.coef[1] == 1) {
        std::memcpy(parity0_delta, data_delta, byte_count);
        std::memcpy(parity1_delta, data_delta, byte_count);
        return true;
    }
    if (lut.coef[0] == 1) {
        std::memcpy(parity0_delta, data_delta, byte_count);
        sponge_encode_parity_bytes_with_lut(data_delta, byte_count, lut.mul[1],
                                            lut.mul16[1], parity1_delta);
        return true;
    }
    sponge_encode_parity_bytes_with_lut(data_delta, byte_count, lut.mul[0],
                                        lut.mul16[0], parity0_delta);
    sponge_encode_parity_bytes_with_lut(data_delta, byte_count, lut.mul[1],
                                        lut.mul16[1], parity1_delta);
    return true;
}

inline bool sponge_commit_record_from_layout(
    SpongeCommitRecord &record,
    const cache::SmallObjectStripeManager::SlotLayout &layout,
    const void *payload, size_t obj_size, uint64_t wr_id, bool reused_dead,
    const Configure &config) {
    if (!sponge_slot_size_supported(layout.slot_size) ||
        obj_size > layout.slot_size || payload == nullptr || wr_id == 0 ||
        layout.data_shard_idx >= kSpongeDataShards) {
        return false;
    }
    if (!config.validate_mapping(layout.data_addr, layout.slot_size) ||
        !config.validate_mapping(layout.parity_addr[0], layout.slot_size) ||
        !config.validate_mapping(layout.parity_addr[1], layout.slot_size)) {
        return false;
    }
    auto [data_ep, data_offset] = config.map_remote_addr(layout.data_addr);
    auto [p0_ep, p0_offset] = config.map_remote_addr(layout.parity_addr[0]);
    auto [p1_ep, p1_offset] = config.map_remote_addr(layout.parity_addr[1]);
    (void)data_ep;

    record = SpongeCommitRecord{};
    record.wr_id = wr_id;
    record.data_offset = data_offset;
    record.parity_offset[0] = p0_offset;
    record.parity_offset[1] = p1_offset;
    record.stripe_id = layout.stripe_id;
    record.slot_id = layout.slot_id;
    record.obj_size = static_cast<uint16_t>(obj_size);
    record.slot_size = layout.slot_size;
    record.bin = layout.bin;
    record.parity_endpoint[0] = static_cast<uint16_t>(p0_ep);
    record.parity_endpoint[1] = static_cast<uint16_t>(p1_ep);
    record.data_shard_idx = layout.data_shard_idx;
    record.flags = reused_dead ? SPONGE_COMMIT_FLAG_REUSED_DEAD : 0;
    std::memcpy(record.payload, payload, obj_size);
    if (obj_size < layout.slot_size) {
        std::memset(record.payload + obj_size, 0, layout.slot_size - obj_size);
    }
    return true;
}

inline bool sponge_apply_commit_payload_locally(
    const SpongeCommitDescriptor &desc, const uint8_t *payload,
    uint8_t *data_slot, uint8_t *delta_out,
    bool *delta_nonzero_out = nullptr) {
    if (!sponge_commit_descriptor_valid(desc) || payload == nullptr ||
        data_slot == nullptr || delta_out == nullptr) {
        return false;
    }
    uint64_t delta_or = 0;
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= desc.slot_size; i += sizeof(uint64_t)) {
        uint64_t old_word = 0;
        uint64_t new_word = 0;
        std::memcpy(&old_word, data_slot + i, sizeof(old_word));
        std::memcpy(&new_word, payload + i, sizeof(new_word));
        uint64_t delta_word = old_word ^ new_word;
        delta_or |= delta_word;
        std::memcpy(data_slot + i, &new_word, sizeof(new_word));
        std::memcpy(delta_out + i, &delta_word, sizeof(delta_word));
    }
    for (; i < desc.slot_size; i++) {
        uint8_t old_byte = data_slot[i];
        uint8_t new_byte = payload[i];
        uint8_t delta = static_cast<uint8_t>(old_byte ^ new_byte);
        delta_or |= delta;
        delta_out[i] = delta;
        data_slot[i] = new_byte;
    }
    if (delta_nonzero_out != nullptr) {
        *delta_nonzero_out = delta_or != 0;
    }
    return true;
}

inline bool sponge_apply_commit_record_locally(const SpongeCommitRecord &record,
                                               uint8_t *data_slot,
                                               uint8_t *delta_out) {
    if (!sponge_slot_size_supported(record.slot_size) ||
        record.obj_size > record.slot_size || data_slot == nullptr ||
        delta_out == nullptr || record.data_shard_idx >= kSpongeDataShards) {
        return false;
    }
    for (size_t i = 0; i < record.slot_size; i++) {
        uint8_t old_byte = data_slot[i];
        uint8_t new_byte = record.payload[i];
        delta_out[i] = static_cast<uint8_t>(old_byte ^ new_byte);
        data_slot[i] = new_byte;
    }
    return true;
}

inline bool sponge_apply_commit_payload_and_encode_parity(
    const SpongeCommitDescriptor &desc, const uint8_t *payload,
    uint8_t *data_slot, uint8_t *parity0_delta,
    uint8_t *parity1_delta) {
    if (!sponge_commit_descriptor_valid(desc) || payload == nullptr ||
        data_slot == nullptr || parity0_delta == nullptr ||
        parity1_delta == nullptr) {
        return false;
    }
    const auto &lut = sponge_shard_parity_lut(desc.data_shard_idx);
    if (lut.coef[0] == 1 && lut.coef[1] == 1) {
        size_t i = 0;
        for (; i + sizeof(uint64_t) <= desc.slot_size; i += sizeof(uint64_t)) {
            uint64_t old_word = 0;
            uint64_t new_word = 0;
            std::memcpy(&old_word, data_slot + i, sizeof(old_word));
            std::memcpy(&new_word, payload + i, sizeof(new_word));
            uint64_t delta_word = old_word ^ new_word;
            std::memcpy(data_slot + i, &new_word, sizeof(new_word));
            std::memcpy(parity0_delta + i, &delta_word, sizeof(delta_word));
            std::memcpy(parity1_delta + i, &delta_word, sizeof(delta_word));
        }
        for (; i < desc.slot_size; i++) {
            uint8_t old_byte = data_slot[i];
            uint8_t new_byte = payload[i];
            uint8_t delta = static_cast<uint8_t>(old_byte ^ new_byte);
            data_slot[i] = new_byte;
            parity0_delta[i] = delta;
            parity1_delta[i] = delta;
        }
        return true;
    }
    if (lut.coef[0] == 1) {
        size_t i = 0;
        for (; i + sizeof(uint64_t) <= desc.slot_size; i += sizeof(uint64_t)) {
            uint64_t old_word = 0;
            uint64_t new_word = 0;
            std::memcpy(&old_word, data_slot + i, sizeof(old_word));
            std::memcpy(&new_word, payload + i, sizeof(new_word));
            uint64_t delta_word = old_word ^ new_word;
            std::memcpy(data_slot + i, &new_word, sizeof(new_word));
            std::memcpy(parity0_delta + i, &delta_word, sizeof(delta_word));
            const uint8_t *delta_bytes =
                reinterpret_cast<const uint8_t *>(&delta_word);
            for (size_t j = 0; j < sizeof(delta_word); j += sizeof(uint16_t)) {
                uint16_t in_pair = 0;
                std::memcpy(&in_pair, delta_bytes + j, sizeof(in_pair));
                uint16_t out_pair = lut.mul16[1][in_pair];
                std::memcpy(parity1_delta + i + j, &out_pair, sizeof(out_pair));
            }
        }
        for (; i < desc.slot_size; i++) {
            uint8_t old_byte = data_slot[i];
            uint8_t new_byte = payload[i];
            uint8_t delta = static_cast<uint8_t>(old_byte ^ new_byte);
            data_slot[i] = new_byte;
            parity0_delta[i] = delta;
            parity1_delta[i] = lut.mul[1][delta];
        }
        return true;
    }
    for (size_t i = 0; i < desc.slot_size; i++) {
        uint8_t old_byte = data_slot[i];
        uint8_t new_byte = payload[i];
        uint8_t delta = static_cast<uint8_t>(old_byte ^ new_byte);
        data_slot[i] = new_byte;
        parity0_delta[i] = lut.mul[0][delta];
        parity1_delta[i] = lut.mul[1][delta];
    }
    return true;
}

inline bool sponge_apply_commit_payload_without_parity(
    const SpongeCommitDescriptor &desc, const uint8_t *payload,
    uint8_t *data_slot) {
    if (!sponge_commit_descriptor_valid(desc) || payload == nullptr ||
        data_slot == nullptr) {
        return false;
    }
    std::memcpy(data_slot, payload, desc.slot_size);
    return true;
}

inline bool sponge_encode_parity_deltas(uint8_t data_shard_idx,
                                        const uint8_t *data_delta,
                                        size_t slot_size,
                                        uint8_t *parity0_delta,
                                        uint8_t *parity1_delta) {
    if (!sponge_slot_size_supported(slot_size) ||
        data_shard_idx >= kSpongeDataShards || data_delta == nullptr ||
        parity0_delta == nullptr || parity1_delta == nullptr) {
        return false;
    }
    return sponge_encode_parity_delta_bytes(data_shard_idx, data_delta,
                                            slot_size, parity0_delta,
                                            parity1_delta);
}

inline bool sponge_encode_parity_deltas(const SpongeCommitRecord &record,
                                        const uint8_t *data_delta,
                                        uint8_t *parity0_delta,
                                        uint8_t *parity1_delta) {
    return sponge_encode_parity_deltas(record.data_shard_idx, data_delta,
                                       record.slot_size, parity0_delta,
                                       parity1_delta);
}

inline bool sponge_encode_parity_delta(uint8_t data_shard_idx,
                                       uint8_t parity_idx,
                                       const uint8_t *data_delta,
                                       size_t slot_size,
                                       uint8_t *parity_delta) {
    if (!sponge_slot_size_supported(slot_size) ||
        data_shard_idx >= kSpongeDataShards ||
        parity_idx >= kSpongeParityShards || data_delta == nullptr ||
        parity_delta == nullptr) {
        return false;
    }
    const auto &lut = sponge_shard_parity_lut(data_shard_idx);
    if (lut.coef[parity_idx] == 1) {
        std::memcpy(parity_delta, data_delta, slot_size);
        return true;
    }
    for (size_t i = 0; i < slot_size; i++) {
        parity_delta[i] = lut.mul[parity_idx][data_delta[i]];
    }
    return true;
}

inline bool sponge_prepare_parity_apply_records(
    const SpongeCommitRecord &record, const uint8_t *data_delta,
    SpongeParityApplyRecord &parity0_record,
    SpongeParityApplyRecord &parity1_record) {
    if (!sponge_slot_size_supported(record.slot_size)) {
        return false;
    }
    parity0_record = SpongeParityApplyRecord{};
    parity1_record = SpongeParityApplyRecord{};
    parity0_record.wr_id = record.wr_id;
    parity1_record.wr_id = record.wr_id;
    parity0_record.parity_offset = record.parity_offset[0];
    parity1_record.parity_offset = record.parity_offset[1];
    parity0_record.slot_size = record.slot_size;
    parity1_record.slot_size = record.slot_size;
    return sponge_encode_parity_deltas(record, data_delta,
                                       parity0_record.payload,
                                       parity1_record.payload);
}

inline bool sponge_apply_parity_payload_locally(uint16_t slot_size,
                                                const uint8_t *payload,
                                                uint8_t *parity_slot) {
    if (!sponge_slot_size_supported(slot_size) || payload == nullptr ||
        parity_slot == nullptr) {
        return false;
    }
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= slot_size; i += sizeof(uint64_t)) {
        uint64_t old_word = 0;
        uint64_t delta_word = 0;
        std::memcpy(&old_word, parity_slot + i, sizeof(old_word));
        std::memcpy(&delta_word, payload + i, sizeof(delta_word));
        old_word ^= delta_word;
        std::memcpy(parity_slot + i, &old_word, sizeof(old_word));
    }
    for (; i < slot_size; i++) {
        parity_slot[i] ^= payload[i];
    }
    return true;
}

inline bool sponge_apply_parity_record_locally(
    const SpongeParityApplyRecord &record, uint8_t *parity_slot) {
    if (!sponge_slot_size_supported(record.slot_size) ||
        parity_slot == nullptr) {
        return false;
    }
    return sponge_apply_parity_payload_locally(record.slot_size,
                                               record.payload, parity_slot);
}

}  // namespace FarLib::rdma
