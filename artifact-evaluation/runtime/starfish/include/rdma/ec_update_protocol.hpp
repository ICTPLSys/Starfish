#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace FarLib::rdma::ec_update {

// One compute process coordinates a transaction over one dead data slot and
// its two parity slots. PREPARE never modifies the visible registered memory.
// A COMMIT decision is irrevocable; all surviving participants must acknowledge
// it before the allocator exposes the replacement. Compute-process restart is
// outside this in-memory protocol's failure model.
inline constexpr uint32_t kMagic = 0x53553250;
inline constexpr uint16_t kVersion = 1;
inline constexpr size_t kMaxBytes = 4096;
inline constexpr size_t kSlots = 64;
inline constexpr size_t kReceiveDepth = 64;

enum class Op : uint8_t { DataPrepare = 1, ParityPrepare = 2, Commit = 3, Abort = 4 };
enum class Status : uint8_t { Ok = 0, Invalid = 1, Conflict = 2, Stale = 3 };

struct Message {
    uint32_t magic = kMagic;
    uint16_t version = kVersion;
    Op op = Op::DataPrepare;
    Status status = Status::Ok;
    uint64_t generation = 0;
    uint64_t offset = 0;
    uint32_t owner = 0;  // logical Evict worker; survives fibre migration
    uint16_t slot = 0;  // worker-private transaction position
    uint16_t bytes = 0;
    uint8_t participant = 0; // 0=data, 1=parity0, 2=parity1
    uint8_t response = 0;
    uint16_t reserved = 0;
    uint8_t payload[kMaxBytes]{};
};

inline constexpr size_t kHeaderBytes = offsetof(Message, payload);
inline bool carries_payload(const Message &m) {
    return (!m.response && (m.op == Op::DataPrepare || m.op == Op::ParityPrepare)) ||
           (m.response && m.op == Op::DataPrepare && m.status == Status::Ok);
}
inline size_t wire_bytes(const Message &m) {
    return kHeaderBytes + (carries_payload(m) ? m.bytes : 0);
}
inline bool valid(const Message &m, size_t wire_size) {
    return m.magic == kMagic && m.version == kVersion && m.generation != 0 &&
           m.slot < kSlots && m.participant < 3 && m.response <= 1 &&
           m.op >= Op::DataPrepare && m.op <= Op::Abort &&
           m.bytes > 0 && m.bytes <= kMaxBytes && wire_size == wire_bytes(m);
}

} // namespace FarLib::rdma::ec_update
