#pragma once

// CPU-side layout and codec helpers for one large object represented as a
// 4+2 EC group.  Small objects continue to use the ec_batch path unchanged;
// this header only handles a caller-provided large-object byte string and a
// six-segment EcStagingGroupSlot.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cache/alloc/ec_batch_staging.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "hydra/page_codec.hpp"

namespace FarLib::cache::ec_split {

inline constexpr size_t kDataSlots = ec_batch::kEcBatchDataSlots;
inline constexpr size_t kParitySlots = ec_batch::kEcBatchParitySlots;
inline constexpr size_t kSegments = ec_batch::kEcBatchSegmentsPerGroup;
inline constexpr size_t kSplitObjectCutoff = 4096;
inline constexpr size_t kMaxObjectBytes = 256 * 1024;
inline constexpr size_t kMaxFragmentBytes = kMaxObjectBytes / kDataSlots;

// The four logical fragments have equal length.  The last fragment is
// zero-padded by encode(); expressing the calculation without bytes + 3 also
// keeps size_t overflow from turning an invalid request into a small one.
inline constexpr size_t fragment_size(size_t object_bytes) noexcept {
    return object_bytes / kDataSlots +
           (object_bytes % kDataSlots == 0 ? 0 : 1);
}

namespace detail {

inline bool valid_scratch(const ec_batch::EcStagingGroupSlot &scratch) {
    if (scratch.slot_size == 0 || scratch.slot_size < fragment_size(1) ||
        scratch.data[0] == nullptr || scratch.parity[0] == nullptr) {
        return false;
    }
    for (size_t i = 0; i < kDataSlots; ++i) {
        if (scratch.data[i] == nullptr) return false;
    }
    for (size_t i = 0; i < kParitySlots; ++i) {
        if (scratch.parity[i] == nullptr) return false;
    }
    return true;
}

inline const void *segment(const ec_batch::EcStagingGroupSlot &scratch,
                           uint8_t index) {
    if (index < kDataSlots) return scratch.data[index];
    return scratch.parity[index - kDataSlots];
}

inline void *data_segment(const ec_batch::EcStagingGroupSlot &scratch,
                          uint8_t index) {
    return scratch.data[index];
}

}  // namespace detail

// Copy and zero-pad one object into four equal logical data fragments, then
// encode both parity fragments over the complete allocator slot.  `scratch`
// must be a live registered lease; ownership/registration validation remains
// the responsibility of the pool that issued it.
inline bool encode(const void *object, size_t bytes,
                   const ec_batch::EcStagingGroupSlot &scratch,
                   bool use_page_codec = false) {
    if (!detail::valid_scratch(scratch)) return false;
    if (bytes > kMaxObjectBytes) return false;
    const size_t logical = fragment_size(bytes);
    if (logical > scratch.slot_size) return false;
    if (bytes != 0 && object == nullptr) return false;

    const auto *source = static_cast<const uint8_t *>(object);
    for (size_t i = 0; i < kDataSlots; ++i) {
        auto *dst = static_cast<uint8_t *>(scratch.data[i]);
        // Zero the complete bin-rounded slot.  This covers both the final
        // logical fragment's padding and allocator-bin tail bytes before the
        // parity codec consumes the full slot_size window.
        std::memset(dst, 0, scratch.slot_size);
        const size_t offset = i * logical;
        if (offset >= bytes) continue;
        const size_t count = std::min(logical, bytes - offset);
        if (count != 0) std::memcpy(dst, source + offset, count);
    }

    const void *data[kDataSlots] = {scratch.data[0], scratch.data[1],
                                    scratch.data[2], scratch.data[3]};
    void *parity[kParitySlots] = {scratch.parity[0], scratch.parity[1]};
    if (use_page_codec)
        return ::FarLib::hydra::page_codec_encode_4plus2(data, parity, scratch.slot_size);
    return small_object_stripe_encode_shards(data, parity, scratch.slot_size);
}

// Reconstruct an object from exactly four selected survivors.  Bits set in
// `selected_mask` identify the four readable data/parity segments (bits 0..3
// are data, bits 4..5 are parity).  Missing data segments are rebuilt in place
// in their existing scratch.data[i] buffers using only those four survivors;
// no newly rebuilt segment is used as a fifth survivor and parity holes are
// never fetched or rebuilt.  Only the original `bytes` are copied to `out`.
inline bool reconstruct(
    uint8_t selected_mask, const ec_batch::EcStagingGroupSlot &scratch,
    void *out, size_t bytes) {
    if (!detail::valid_scratch(scratch)) return false;
    if (bytes > kMaxObjectBytes) return false;
    if (bytes != 0 && out == nullptr) return false;
    const uint8_t all_mask = static_cast<uint8_t>((1u << kSegments) - 1u);
    if ((selected_mask & static_cast<uint8_t>(~all_mask)) != 0 ||
        __builtin_popcount(static_cast<unsigned>(selected_mask)) !=
            kDataSlots) {
        return false;
    }

    const size_t logical = fragment_size(bytes);
    if (logical > scratch.slot_size) return false;

    std::array<uint8_t, kDataSlots> survivor_idx{};
    std::array<const void *, kDataSlots> survivor_ptrs{};
    size_t survivor_count = 0;
    for (uint8_t segment = 0; segment < kSegments; ++segment) {
        if ((selected_mask & static_cast<uint8_t>(1u << segment)) == 0) {
            continue;
        }
        survivor_idx[survivor_count] = segment;
        survivor_ptrs[survivor_count] = detail::segment(scratch, segment);
        ++survivor_count;
    }
    if (survivor_count != kDataSlots) return false;

    // Rebuild every missing data segment independently from the four original
    // survivors.  For a double data failure, the first reconstruction is not
    // added to survivor_ptrs, preserving the exact-four-read contract.
    for (uint8_t data = 0; data < kDataSlots; ++data) {
        if ((selected_mask & static_cast<uint8_t>(1u << data)) != 0) {
            continue;
        }
        if (!small_object_stripe_rebuild_one(
                data, survivor_idx.data(), survivor_ptrs.data(),
                detail::data_segment(scratch, data), logical)) {
            return false;
        }
    }

    auto *destination = static_cast<uint8_t *>(out);
    for (size_t data = 0; data < kDataSlots; ++data) {
        const size_t offset = data * logical;
        if (offset >= bytes) continue;
        const size_t count = std::min(logical, bytes - offset);
        if (count != 0) {
            std::memcpy(destination + offset, scratch.data[data], count);
        }
    }
    return true;
}

}  // namespace FarLib::cache::ec_split
