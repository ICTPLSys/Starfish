#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <infiniband/verbs.h>

namespace FarLib::rdma {
class Client;
}

namespace FarLib::cache::carbink::shadow {

// Ordinary READs occupy every top nibble from 0x4 through 0x7: their
// generation spans bits 48..61. In particular, 0x5 is not a free namespace.
// Use 0x1, disjoint from raw pointers (0x0), ordinary WRITEs (0x2),
// background rebuild (0x3), ordinary READs (0x4..0x7), and EC (bit 63).
inline constexpr uint64_t kShadowWrIdTag = 0x1ull << 60;
inline constexpr uint64_t kShadowWrIdTagMask = 0xfull << 60;
inline constexpr uint64_t kShadowWrIdSlotMask = 0xffull;
inline constexpr uint64_t kShadowWrIdSequenceMask = (1ull << 52) - 1;
inline constexpr size_t kShadowZeroSlots = 6;
static_assert((kShadowWrIdTag & (0x7ull << 61)) == 0,
              "shadow completions must not overlap ordinary READ/WRITE or EC");
static_assert(((kShadowWrIdSequenceMask << 8) & kShadowWrIdTagMask) == 0,
              "shadow sequences must not overwrite the namespace tag");

inline bool is_wr_id(uint64_t id) {
    return (id & kShadowWrIdTagMask) == kShadowWrIdTag;
}

enum class PostResult : uint8_t {
    kAccepted,
    kQueueFull,
    kFailed,
};

// Tracks the six zero-WR completions of one unpublished shadow group.  The
// registry is process-global only for CQ dispatch; ownership remains with the
// compaction fibre that created this object.  IDs are opaque monotonic values,
// never pointers or Update addresses.
class IoBatch {
public:
    IoBatch() = default;
    IoBatch(const IoBatch &) = delete;
    IoBatch &operator=(const IoBatch &) = delete;
    ~IoBatch() { forget_all(); }

    uint64_t arm(size_t slot);
    void cancel(size_t slot);
    int status(size_t slot) const;
    bool all_terminal() const;
    bool all_success() const;

    PostResult post_zero(rdma::Client &client, size_t qp_idx,
                         size_t endpoint_idx, uint64_t remote_offset,
                         void *zero_page, uint32_t zero_lkey,
                         uint32_t bytes, size_t slot);

    static bool consume_completion(const ibv_wc &wc);
    static uint64_t next_rpc_wr_id();

private:
    struct Slot {
        std::atomic<uint64_t> expected{0};
        std::atomic<int> status{0};
    };
    std::array<Slot, kShadowZeroSlots> slots_{};

    void complete(uint64_t wr_id, bool success);
    void forget_all();
};

}  // namespace FarLib::cache::carbink::shadow
