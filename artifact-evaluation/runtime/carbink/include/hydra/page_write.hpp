#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "cache/alloc/ec_split_layout.hpp"
#include "cache/entry.hpp"
#include "hydra/page_layout.hpp"

namespace FarLib::hydra {

inline constexpr size_t kPageWriteFragmentBytes = kPageBytes / 4;

inline bool page_source_in_registered_range(const void *page,
                                            const void *registered_base,
                                            size_t registered_bytes) noexcept {
    if (page == nullptr || registered_base == nullptr ||
        registered_bytes < kPageBytes) return false;
    const auto source = reinterpret_cast<uintptr_t>(page);
    const auto base = reinterpret_cast<uintptr_t>(registered_base);
    // Subtraction after the ordering check avoids base+length overflow.
    return source >= base && source - base <= registered_bytes - kPageBytes &&
           source <= std::numeric_limits<uintptr_t>::max() - (kPageBytes - 1);
}

// A prior completion may lose its last write reference just before a new
// eviction starts and is rescued back to LOCAL. It must not invalidate that
// new eviction's remote anchor. The later invalid-bit CAS rechecks these bits.
inline bool page_write_cleanup_may_lock(const cache::EntryStateBits &state) noexcept {
    return state.ref_cnt == 0 && !state.invalid &&
           (state.state == cache::LOCAL || state.state == cache::MARKED);
}

inline const void *page_data_fragment(const void *page, size_t segment) noexcept {
    if (page == nullptr || segment >= 4) return nullptr;
    return static_cast<const uint8_t *>(page) + segment * kPageWriteFragmentBytes;
}

// Borrow the already-registered page's four data fragments. Only the two parity
// outputs are written; the lease's four data buffers remain untouched. The
// caller owns the page write reference until all six terminal completions.
inline bool encode_page_parity_only(
    const void *page, const cache::ec_batch::EcStagingGroupSlot &scratch) {
    if (page == nullptr || scratch.slot_size != kPageWriteFragmentBytes ||
        scratch.parity[0] == nullptr || scratch.parity[1] == nullptr) return false;
    const void *data[4];
    for (size_t i = 0; i < 4; ++i) data[i] = page_data_fragment(page, i);
    void *parity[2] = {scratch.parity[0], scratch.parity[1]};
    return page_codec_encode_4plus2(data, parity, kPageWriteFragmentBytes);
}

}  // namespace FarLib::hydra
