#pragma once

// Direct encoder for one four-object small-object EC group.
//
// Unlike EcGroupBuilder, this API does not copy objects into a staging group.
// Each non-zero object prefix is passed directly to ISA-L; bytes after an
// object's declared size are represented by the caller-provided zero_pad
// buffer.  The output parity buffers are written in place.

#include "cache/alloc/small_object_stripe_codec.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace FarLib::cache {

inline constexpr size_t kEcDirectDataShards = 4;
inline constexpr size_t kEcDirectParityShards = 2;
inline constexpr size_t kEcDirectMaxSlotSize = 4096;

// Encode four variable-length objects into two parity slots without creating a
// data staging buffer.  Each data object occupies the prefix [0, sizes[i]) of
// a slot of slot_size bytes.  A null data[i] is valid only when sizes[i] == 0;
// such a slot is a hole.  All bytes outside an object's prefix are zero and
// are read from zero_pad, which must provide slot_size readable zero bytes when
// any object is short.  parity[0] and parity[1] each require slot_size writable
// bytes.
//
// The function returns false without touching parity when the arguments are
// invalid.  A non-zero slot_size is required; an all-zero four-object group is
// valid and is handled by clearing both parity slots.  The only data copied by
// this function is the stack-resident pointer arrays passed to ISA-L; object
// bytes are never copied to a staging buffer.
inline bool encode_direct_shards(
    const void *const data[kEcDirectDataShards],
    const uint32_t sizes[kEcDirectDataShards],
    void *const parity[kEcDirectParityShards], size_t slot_size,
    const void *zero_pad) {
    if (data == nullptr || sizes == nullptr || parity == nullptr ||
        slot_size == 0 || slot_size > kEcDirectMaxSlotSize) {
        return false;
    }

    bool needs_zero_pad = false;
    bool any_object = false;
    for (size_t i = 0; i < kEcDirectDataShards; ++i) {
        if (sizes[i] > slot_size) return false;
        if (sizes[i] != 0) {
            any_object = true;
            if (data[i] == nullptr) return false;
        }
        if (sizes[i] < slot_size) needs_zero_pad = true;
    }
    for (size_t i = 0; i < kEcDirectParityShards; ++i) {
        if (parity[i] == nullptr) return false;
    }
    // An entirely empty group never reads zero_pad because its parity is
    // cleared directly; short non-empty groups do require the zero source.
    if (needs_zero_pad && any_object && zero_pad == nullptr) return false;

    auto *zero = static_cast<const uint8_t *>(zero_pad);
    if (!any_object) {
        // The empty group is a valid all-zero stripe.  Avoid an unnecessary
        // ISA-L call and make its output deterministic even if zero_pad is a
        // caller-provided read-only page.
        for (size_t i = 0; i < kEcDirectParityShards; ++i) {
            std::memset(parity[i], 0, slot_size);
        }
        return true;
    }

    size_t offset = 0;
    while (offset < slot_size) {
        size_t next = slot_size;
        bool active = false;
        for (size_t i = 0; i < kEcDirectDataShards; ++i) {
            if (sizes[i] > offset) {
                active = true;
                if (sizes[i] < next) next = sizes[i];
            }
        }
        if (!active) {
            // Every remaining input is padding.  Do not read zero_pad and do
            // not issue a second ISA-L call for a zero-only tail.
            const size_t tail = slot_size - offset;
            for (size_t i = 0; i < kEcDirectParityShards; ++i) {
                auto *dst = static_cast<uint8_t *>(parity[i]);
                std::memset(dst + offset, 0, tail);
            }
            break;
        }

        const size_t segment_size = next - offset;
        if (segment_size == 0) return false;

        const void *segment_data[kEcDirectDataShards];
        void *segment_parity[kEcDirectParityShards];
        for (size_t i = 0; i < kEcDirectDataShards; ++i) {
            if (sizes[i] > offset) {
                segment_data[i] = static_cast<const uint8_t *>(data[i]) + offset;
            } else {
                segment_data[i] = zero + offset;
            }
        }
        for (size_t i = 0; i < kEcDirectParityShards; ++i) {
            segment_parity[i] = static_cast<uint8_t *>(parity[i]) + offset;
        }

        if (!small_object_stripe_encode_shards(
                segment_data, segment_parity, segment_size, 0)) {
            return false;
        }
        offset = next;
    }
    return true;
}

// Descriptive alias for call sites that use the existing small-object codec
// naming convention.
inline bool small_object_stripe_encode_direct(
    const void *const data[kEcDirectDataShards],
    const uint32_t sizes[kEcDirectDataShards],
    void *const parity[kEcDirectParityShards], size_t slot_size,
    const void *zero_pad) {
    return encode_direct_shards(data, sizes, parity, slot_size, zero_pad);
}

inline bool ec_direct_encode_4(
    const void *const data[kEcDirectDataShards],
    const uint32_t sizes[kEcDirectDataShards],
    void *const parity[kEcDirectParityShards], size_t slot_size,
    const void *zero_pad) {
    return encode_direct_shards(data, sizes, parity, slot_size, zero_pad);
}

}  // namespace FarLib::cache
