#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <erasure_code.h>

namespace FarLib::hydra {

inline constexpr std::size_t kPageCodecDataShards = 4;
inline constexpr std::size_t kPageCodecParityShards = 2;

namespace detail {

struct PageCodecTables {
    // Preserve the existing systematic RS(4,2) codeword: the data rows are
    // identity, and the parity rows are [1,1,1,1] and [1,2,4,8] over GF(2^8).
    // ISA-L uses the same primitive polynomial (0x11d). Do not substitute a
    // Cauchy matrix: existing remote data and the recovery codec must agree.
    alignas(64) std::array<unsigned char, 32 * kPageCodecDataShards *
                                               kPageCodecParityShards> bytes;

    PageCodecTables() noexcept {
        unsigned char coefficients[] = {1, 1, 1, 1, 1, 2, 4, 8};
        ec_init_tables(kPageCodecDataShards, kPageCodecParityShards,
                       coefficients, bytes.data());
    }
};

inline const PageCodecTables &page_codec_tables() noexcept {
    // C++ inline-function static: one thread-safe initialization per process,
    // shared by every worker and translation unit; never per page or request.
    static const PageCodecTables tables;
    return tables;
}

inline bool page_codec_valid_inputs(
    const void *const data_shards[kPageCodecDataShards],
    void *const parity_shards[kPageCodecParityShards]) noexcept {
    if (data_shards == nullptr || parity_shards == nullptr) return false;
    for (std::size_t i = 0; i < kPageCodecDataShards; ++i)
        if (data_shards[i] == nullptr) return false;
    for (std::size_t i = 0; i < kPageCodecParityShards; ++i)
        if (parity_shards[i] == nullptr) return false;
    return true;
}

}  // namespace detail

inline void page_codec_initialize() noexcept {
    // Warm the single immutable table during cache construction, before Work.
    (void)detail::page_codec_tables();
}

inline bool page_codec_encode_4plus2(
    const void *const data_shards[kPageCodecDataShards],
    void *const parity_shards[kPageCodecParityShards],
    std::size_t byte_count) noexcept {
    if (!detail::page_codec_valid_inputs(data_shards, parity_shards) ||
        byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return false;
    if (byte_count == 0) return true;

    unsigned char *data[kPageCodecDataShards];
    unsigned char *parity[kPageCodecParityShards];
    for (std::size_t i = 0; i < kPageCodecDataShards; ++i)
        data[i] = const_cast<unsigned char *>(
            static_cast<const unsigned char *>(data_shards[i]));
    for (std::size_t i = 0; i < kPageCodecParityShards; ++i)
        parity[i] = static_cast<unsigned char *>(parity_shards[i]);

    // ISA-L's C API omits const qualifiers, but source buffers and expanded
    // tables are inputs only. Concurrent workers share tables, not outputs.
    auto *tables = const_cast<unsigned char *>(
        detail::page_codec_tables().bytes.data());
    const int length = static_cast<int>(byte_count);
    if (byte_count < 64) {
        // Use ISA-L's own baseline for short fragments, below the SIMD
        // kernels' minimum lengths. Normal 8KiB pages have 2048B fragments.
        ec_encode_data_base(length, kPageCodecDataShards,
                            kPageCodecParityShards, tables, data, parity);
    } else {
        ec_encode_data(length, kPageCodecDataShards,
                       kPageCodecParityShards, tables, data, parity);
    }
    return true;
}

}  // namespace FarLib::hydra
