/**
 * EC-Split erasure codec for per-span 4+2 Reed-Solomon encoding.
 *
 * One 8KB span is split into N=4 data shards (2KB each) and K=2 parity shards.
 * Uses Intel ISA-L for encode/decode. Requires exclusive cache and server_count >= 6.
 */
#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <isa-l/erasure_code.h>

namespace FarLib {
namespace cache {

// EC-Split parameters (N data, K parity)
constexpr int EC_N = 4;   // data shards
constexpr int EC_K = 2;   // parity shards
constexpr int EC_TOTAL = EC_N + EC_K;  // 6 shards
#ifndef FARLIB_SPAN_SIZE
#define FARLIB_SPAN_SIZE 8192u
#endif
constexpr size_t EC_SHARD_SIZE = FARLIB_SPAN_SIZE / EC_N;
static_assert(FARLIB_SPAN_SIZE % EC_N == 0,
              "EC-Split span size must divide evenly into data shards");

struct ECSpanSlotCoeffLut {
    uint8_t coef0 = 0;
    uint8_t coef1 = 0;
    uint8_t mul0[256] = {0};
    uint8_t mul1[256] = {0};
};

struct ECSpanCodecCache {
    ECSpanSlotCoeffLut lut[EC_N];
    unsigned char encode_tables[32 * EC_N * EC_K] = {0};

    ECSpanCodecCache() {
        unsigned char encode_matrix[EC_TOTAL * EC_N];
        gf_gen_rs_matrix(encode_matrix, EC_TOTAL, EC_N);
        ec_init_tables(EC_N, EC_K, encode_matrix + EC_N * EC_N,
                       encode_tables);
        for (int s = 0; s < EC_N; s++) {
            auto &entry = lut[s];
            entry.coef0 = encode_matrix[(EC_N + 0) * EC_N + s];
            entry.coef1 = encode_matrix[(EC_N + 1) * EC_N + s];
            for (int v = 0; v < 256; v++) {
                entry.mul0[v] =
                    gf_mul(entry.coef0, static_cast<unsigned char>(v));
                entry.mul1[v] =
                    gf_mul(entry.coef1, static_cast<unsigned char>(v));
            }
        }
    }
};

inline const ECSpanCodecCache &ec_span_codec_cache() {
    static const ECSpanCodecCache cache;
    return cache;
}

inline const ECSpanSlotCoeffLut &ec_span_slot_coeff_lut(uint8_t slot) {
    assert(slot < EC_N);
    return ec_span_codec_cache().lut[slot];
}

// Build the 8KB delta while updating the local data copy in the same pass.
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) inline void
ec_span_prepare_delta_and_update_old_avx2(
    uint8_t *__restrict__ old_data, const uint8_t *__restrict__ new_data,
    uint8_t *__restrict__ delta) {
    constexpr size_t kVecBytes = sizeof(__m256i);
    constexpr size_t kVecCount = (EC_N * EC_SHARD_SIZE) / kVecBytes;
    auto *old_vec = reinterpret_cast<__m256i *>(old_data);
    const auto *new_vec = reinterpret_cast<const __m256i *>(new_data);
    auto *delta_vec = reinterpret_cast<__m256i *>(delta);
    for (size_t i = 0; i < kVecCount; i++) {
        __m256i old_v = _mm256_loadu_si256(old_vec + i);
        __m256i new_v = _mm256_loadu_si256(new_vec + i);
        _mm256_storeu_si256(delta_vec + i, _mm256_xor_si256(old_v, new_v));
        _mm256_storeu_si256(old_vec + i, new_v);
    }
}

inline bool ec_span_cpu_supports_avx2() {
    static const bool supported = __builtin_cpu_supports("avx2");
    return supported;
}
#endif

inline void ec_span_prepare_delta_and_update_old(
    uint8_t *__restrict__ old_data, const uint8_t *__restrict__ new_data,
    uint8_t *__restrict__ delta) {
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_expect(ec_span_cpu_supports_avx2(), 1)) {
        ec_span_prepare_delta_and_update_old_avx2(old_data, new_data, delta);
        return;
    }
#endif
    constexpr size_t kWordBytes = sizeof(uint64_t);
    constexpr size_t kWordCount = (EC_N * EC_SHARD_SIZE) / kWordBytes;
    auto *old64 = reinterpret_cast<uint64_t *>(old_data);
    const auto *new64 = reinterpret_cast<const uint64_t *>(new_data);
    auto *delta64 = reinterpret_cast<uint64_t *>(delta);
    for (size_t i = 0; i < kWordCount; i++) {
        uint64_t new_v = new64[i];
        delta64[i] = old64[i] ^ new_v;
        old64[i] = new_v;
    }
}

/**
 * Delta-encode one 8KB span update and optionally expose the local-prep vs
 * ISA-L update split for profiling.
 */
inline void ec_span_delta_encode_update(uint8_t slot, uint8_t *old_data,
                                        const uint8_t *new_data,
                                        uint8_t *delta0, uint8_t *delta1,
                                        bool init_from_zero = false,
                                        uint64_t *prepare_ns = nullptr,
                                        uint64_t *encode_ns = nullptr) {
    assert(slot < EC_N);
    alignas(64) thread_local std::array<uint8_t, EC_N * EC_SHARD_SIZE> delta{};
    auto prepare_begin = std::chrono::steady_clock::now();
    if (init_from_zero) {
        std::memcpy(delta.data(), new_data, EC_N * EC_SHARD_SIZE);
        std::memcpy(old_data, new_data, EC_N * EC_SHARD_SIZE);
    } else {
        ec_span_prepare_delta_and_update_old(old_data, new_data, delta.data());
    }
    auto prepare_end = std::chrono::steady_clock::now();
    unsigned char *coding_ptrs[EC_K] = {delta0, delta1};
    auto &cache = ec_span_codec_cache();
    auto encode_begin = prepare_end;
    ec_encode_data_update(static_cast<int>(EC_N * EC_SHARD_SIZE), EC_N, EC_K,
                          static_cast<int>(slot),
                          const_cast<unsigned char *>(cache.encode_tables),
                          delta.data(), coding_ptrs);
    auto encode_end = std::chrono::steady_clock::now();
    if (prepare_ns != nullptr) {
        *prepare_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                prepare_end - prepare_begin)
                .count());
    }
    if (encode_ns != nullptr) {
        *encode_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                encode_end - encode_begin)
                .count());
    }
}

inline void ec_span_full_stripe_encode(uint8_t k,
                                       const uint8_t *span_ptrs[EC_N],
                                       uint8_t *parity0,
                                       uint8_t *parity1) {
    assert(k > 0 && k <= EC_N);
    assert(parity0 != nullptr && parity1 != nullptr);
    constexpr size_t kSpanBytes = EC_N * EC_SHARD_SIZE;
    alignas(64) static std::array<unsigned char, kSpanBytes> zero_span{};

    unsigned char *data_ptrs[EC_N];
    (void)k;
    for (uint8_t slot = 0; slot < EC_N; slot++) {
        const uint8_t *src = span_ptrs[slot] != nullptr
                                 ? span_ptrs[slot]
                                 : reinterpret_cast<const uint8_t *>(
                                       zero_span.data());
        data_ptrs[slot] = const_cast<unsigned char *>(
            reinterpret_cast<const unsigned char *>(src));
    }
    unsigned char *coding_ptrs[EC_K] = {parity0, parity1};
    auto &cache = ec_span_codec_cache();
    ec_encode_data(static_cast<int>(kSpanBytes), EC_N, EC_K,
                   const_cast<unsigned char *>(cache.encode_tables), data_ptrs,
                   coding_ptrs);
}

inline bool ec_span_recover_missing_shard(
    uint8_t missing_shard, const uint8_t *shard_ptrs[EC_TOTAL],
    uint8_t *out) {
    if (missing_shard >= EC_TOTAL || shard_ptrs == nullptr || out == nullptr) {
        return false;
    }
    constexpr size_t kSpanBytes = EC_N * EC_SHARD_SIZE;
    if (missing_shard >= EC_N) {
        for (uint8_t i = 0; i < EC_N; i++) {
            if (shard_ptrs[i] == nullptr) {
                return false;
            }
        }
        unsigned char *data_ptrs[EC_N];
        for (uint8_t i = 0; i < EC_N; i++) {
            data_ptrs[i] =
                const_cast<unsigned char *>(shard_ptrs[i]);
        }
        alignas(64) std::array<uint8_t, kSpanBytes> scratch{};
        unsigned char *coding_ptrs[EC_K] = {
            missing_shard == EC_N ? out : scratch.data(),
            missing_shard == EC_N + 1 ? out : scratch.data(),
        };
        auto &cache = ec_span_codec_cache();
        ec_encode_data(static_cast<int>(kSpanBytes), EC_N, EC_K,
                       const_cast<unsigned char *>(cache.encode_tables),
                       data_ptrs, coding_ptrs);
        return true;
    }

    unsigned char encode_matrix[EC_TOTAL * EC_N];
    unsigned char decode_matrix[EC_N * EC_N];
    unsigned char invert_matrix[EC_N * EC_N];
    unsigned char decode_coeff[EC_N];
    gf_gen_rs_matrix(encode_matrix, EC_TOTAL, EC_N);

    unsigned char *decode_ptrs[EC_N];
    uint8_t used = 0;
    for (uint8_t row = 0; row < EC_TOTAL && used < EC_N; row++) {
        if (row == missing_shard || shard_ptrs[row] == nullptr) {
            continue;
        }
        std::memcpy(decode_matrix + used * EC_N,
                    encode_matrix + row * EC_N, EC_N);
        decode_ptrs[used] = const_cast<unsigned char *>(shard_ptrs[row]);
        used++;
    }
    if (used != EC_N) {
        return false;
    }
    if (gf_invert_matrix(decode_matrix, invert_matrix, EC_N) < 0) {
        return false;
    }
    std::memcpy(decode_coeff, invert_matrix + missing_shard * EC_N, EC_N);

    unsigned char decode_tables[32 * EC_N];
    ec_init_tables(EC_N, 1, decode_coeff, decode_tables);
    unsigned char *recover_ptrs[1] = {out};
    ec_encode_data(static_cast<int>(kSpanBytes), EC_N, 1, decode_tables,
                   decode_ptrs, recover_ptrs);
    return true;
}

inline bool ec_span_recover_missing_shards(
    const uint8_t *missing_shards, uint8_t missing_count,
    const uint8_t *shard_ptrs[EC_TOTAL], uint8_t *out_ptrs[EC_K]) {
    if (missing_count == 0) {
        return true;
    }
    if (missing_shards == nullptr || shard_ptrs == nullptr ||
        out_ptrs == nullptr || missing_count > EC_K) {
        return false;
    }
    constexpr size_t kSpanBytes = EC_N * EC_SHARD_SIZE;
    bool missing[EC_TOTAL] = {};
    uint8_t missing_data[EC_K] = {};
    uint8_t missing_data_count = 0;
    uint8_t missing_parity_count = 0;
    for (uint8_t i = 0; i < missing_count; i++) {
        uint8_t shard = missing_shards[i];
        if (shard >= EC_TOTAL || out_ptrs[i] == nullptr || missing[shard]) {
            return false;
        }
        missing[shard] = true;
        if (shard < EC_N) {
            missing_data[missing_data_count++] = i;
        } else {
            missing_parity_count++;
        }
    }

    if (missing_data_count != 0) {
        unsigned char encode_matrix[EC_TOTAL * EC_N];
        unsigned char decode_matrix[EC_N * EC_N];
        unsigned char invert_matrix[EC_N * EC_N];
        unsigned char decode_coeff[EC_K * EC_N];
        gf_gen_rs_matrix(encode_matrix, EC_TOTAL, EC_N);

        unsigned char *decode_ptrs[EC_N];
        uint8_t used = 0;
        for (uint8_t row = 0; row < EC_TOTAL && used < EC_N; row++) {
            if (missing[row] || shard_ptrs[row] == nullptr) {
                continue;
            }
            std::memcpy(decode_matrix + used * EC_N,
                        encode_matrix + row * EC_N, EC_N);
            decode_ptrs[used] = const_cast<unsigned char *>(shard_ptrs[row]);
            used++;
        }
        if (used != EC_N) {
            return false;
        }
        if (gf_invert_matrix(decode_matrix, invert_matrix, EC_N) < 0) {
            return false;
        }

        unsigned char *recover_ptrs[EC_K] = {};
        for (uint8_t i = 0; i < missing_data_count; i++) {
            uint8_t missing_idx = missing_data[i];
            uint8_t data_shard = missing_shards[missing_idx];
            std::memcpy(decode_coeff + i * EC_N,
                        invert_matrix + data_shard * EC_N, EC_N);
            recover_ptrs[i] = out_ptrs[missing_idx];
        }
        unsigned char decode_tables[32 * EC_N * EC_K];
        ec_init_tables(EC_N, missing_data_count, decode_coeff, decode_tables);
        ec_encode_data(static_cast<int>(kSpanBytes), EC_N,
                       missing_data_count, decode_tables, decode_ptrs,
                       recover_ptrs);
    }

    if (missing_parity_count != 0) {
        unsigned char *data_ptrs[EC_N] = {};
        for (uint8_t shard = 0; shard < EC_N; shard++) {
            if (!missing[shard]) {
                if (shard_ptrs[shard] == nullptr) {
                    return false;
                }
                data_ptrs[shard] =
                    const_cast<unsigned char *>(shard_ptrs[shard]);
                continue;
            }
            bool found = false;
            for (uint8_t i = 0; i < missing_count; i++) {
                if (missing_shards[i] == shard) {
                    data_ptrs[shard] = out_ptrs[i];
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        alignas(64) std::array<uint8_t, kSpanBytes> scratch0{};
        alignas(64) std::array<uint8_t, kSpanBytes> scratch1{};
        unsigned char *coding_ptrs[EC_K] = {scratch0.data(), scratch1.data()};
        for (uint8_t i = 0; i < missing_count; i++) {
            if (missing_shards[i] == EC_N) {
                coding_ptrs[0] = out_ptrs[i];
            } else if (missing_shards[i] == EC_N + 1) {
                coding_ptrs[1] = out_ptrs[i];
            }
        }
        auto &cache = ec_span_codec_cache();
        ec_encode_data(static_cast<int>(kSpanBytes), EC_N, EC_K,
                       const_cast<unsigned char *>(cache.encode_tables),
                       data_ptrs, coding_ptrs);
    }
    return true;
}

/**
 * Encodes one 8KB span into 4 data + 2 parity shards using ISA-L.
 *
 * @param span_data Pointer to 8KB contiguous span (must be 8KB)
 * @param data_shards [out] Array of 4 pointers, each to 2KB buffer (can be same as span chunks)
 * @param parity_shards [out] Array of 2 pointers, each to 2KB buffer (must be pre-allocated)
 *
 * Data shards are contiguous chunks: data_shards[i] = span_data + i * EC_SHARD_SIZE.
 * Parity shards are written by this function.
 */
inline void ec_split_encode(const uint8_t *span_data,
                            uint8_t *data_shards[EC_N],
                            uint8_t *parity_shards[EC_K]) {
    // Set up data pointers (may alias span_data)
    unsigned char *data_ptrs[EC_N];
    for (int i = 0; i < EC_N; i++) {
        data_ptrs[i] = const_cast<unsigned char *>(span_data + i * EC_SHARD_SIZE);
        if (data_shards[i] != data_ptrs[i]) {
            std::memcpy(data_shards[i], data_ptrs[i], EC_SHARD_SIZE);
        }
    }
    // If caller passed span_data chunks as data_shards, no copy needed
    unsigned char *coding_ptrs[EC_K];
    for (int i = 0; i < EC_K; i++) {
        coding_ptrs[i] = parity_shards[i];
    }

    // Reed-Solomon matrix: 6 rows (data+parity) x 4 cols (data)
    unsigned char encode_matrix[EC_TOTAL * EC_N];
    gf_gen_rs_matrix(encode_matrix, EC_TOTAL, EC_N);
    // Rows 4,5 are parity coefficients
    unsigned char encode_tables[32 * EC_N * EC_K];
    ec_init_tables(EC_N, EC_K, encode_matrix + EC_N * EC_N, encode_tables);

    ec_encode_data(static_cast<int>(EC_SHARD_SIZE), EC_N, EC_K,
                   encode_tables, data_ptrs, coding_ptrs);
}

/**
 * Encodes one 8KB span in-place: data shards are span chunks, parity written to output.
 *
 * @param span_data Pointer to 8KB span (read; data shards are span[0:2K], [2K:4K], [4K:6K], [6K:8K])
 * @param parity_buf Pointer to 4KB buffer for 2 parity shards (must be pre-allocated)
 */
inline void ec_split_encode_inplace(const uint8_t *span_data, uint8_t *parity_buf) {
    unsigned char *data_ptrs[EC_N];
    for (int i = 0; i < EC_N; i++) {
        data_ptrs[i] = const_cast<unsigned char *>(span_data + i * EC_SHARD_SIZE);
    }
    unsigned char *coding_ptrs[EC_K] = {
        parity_buf,
        parity_buf + EC_SHARD_SIZE,
    };

    unsigned char encode_matrix[EC_TOTAL * EC_N];
    gf_gen_rs_matrix(encode_matrix, EC_TOTAL, EC_N);
    unsigned char encode_tables[32 * EC_N * EC_K];
    ec_init_tables(EC_N, EC_K, encode_matrix + EC_N * EC_N, encode_tables);

    ec_encode_data(static_cast<int>(EC_SHARD_SIZE), EC_N, EC_K,
                   encode_tables, data_ptrs, coding_ptrs);
}

/**
 * Decode from N available shards back to 8KB span (for degraded fetch).
 * Stub: returns false. Full implementation in later batch.
 */
inline bool ec_split_decode(uint8_t *span_data,
                            const uint8_t *shard_ptrs[EC_TOTAL],
                            const uint8_t shard_valid[EC_TOTAL]) {
    (void)span_data;
    (void)shard_ptrs;
    (void)shard_valid;
    return false;  // Degraded path: stub only
}

}  // namespace cache
}  // namespace FarLib
