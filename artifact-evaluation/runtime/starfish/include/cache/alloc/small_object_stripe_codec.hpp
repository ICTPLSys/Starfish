#pragma once

// GF(2^8) Reed-Solomon codec for the small-object EC stripe (4 data + 2 parity
// shards, one shard == one remote region).
//
// This is the *physical* parity layer that phase 1 deliberately left as a stub
// (see SmallObjectStripeEncoder at the bottom of small_object_stripe.hpp).  It
// is a port of G1
// (starfish-design3-20260824/include/rdma/sponge_rpc.hpp); the mapping is:
//
//   G1 site                                       here
//   sponge_rpc.hpp:26-27   4 data + 2 parity      kStripeCodec*Shards
//   sponge_rpc.hpp:33      GF(2^8), poly 0x1d     SmallObjectStripeGf256
//   sponge_rpc.hpp:34      1<<16 pair LUT         kStripeCodecGfPairLutSize
//   sponge_rpc.hpp:822-836 gf_mul (bitwise)       Gf256::mul              [copy]
//   sponge_rpc.hpp:838-844 gf_pow                 Gf256::pow              [copy]
//   sponge_rpc.hpp:846-859 gen_rs_matrix          Gf256::build_generator_matrix
//   sponge_rpc.hpp:781-785 SpongeShardParityLut   SmallObjectStripeShardParityLut
//   sponge_rpc.hpp:787-819 SpongeParityCodecCache SmallObjectStripeParityCodecCache
//   sponge_rpc.hpp:873-888 ..._bytes_with_lut     ..._apply_bytes_with_lut [copy]
//   sponge_rpc.hpp:890-917 ..._parity_delta_bytes ..._encode_parity_delta  [copy]
//
// The copied pieces are kept identical on purpose: same primitive polynomial
// (0x1d), same generator-matrix construction (rows 4/5 are the Vandermonde rows
// with bases 1 and 2, i.e. parity0 = XOR of the four data shards and
// parity1 = sum_j 2^j * data_j), same 256-entry and 65536-entry LUT layout and
// the same pair-LUT construction.  The resulting coef/mul/mul16 tables are
// therefore byte-for-byte G1's tables (the stripe self-check compares them and
// the encoded parity against a verbatim copy of the G1 reference code).
//
// The two deliberate deviations from G1 are marked [new] below:
//
//   * the full-stripe *encode* (assign semantics) and the XOR-accumulating twin
//     of G1's byte loop.  G1's sponge_encode_parity_delta_bytes() already covers
//     the delta case; encode is that plus the "the first contributing shard
//     assigns instead of XORs" rule, which is exactly G1's coef == 1 fast path.
//   * decode/rebuild.  G1 has no decoder at all (it only ever recomputed
//     parity), so there is no G1 table to align with here; the arithmetic and
//     the LUT construction are the ones above, parameterized by the
//     reconstruction coefficient.
//
// Nothing in this file is reachable while ft_method == none: the codec is only
// called from the stripe write/ACK path, which phase 1 already gates on
// Configure::ft_enabled().

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace FarLib::cache {

// Geometry of the ported code; small_object_stripe.hpp static_asserts that this
// agrees with its own kSmallObjectStripe* constants.
inline constexpr size_t kStripeCodecDataShards = 4;
inline constexpr size_t kStripeCodecParityShards = 2;
inline constexpr size_t kStripeCodecShardCount =
    kStripeCodecDataShards + kStripeCodecParityShards;
inline constexpr size_t kStripeCodecGfPairLutSize = 1u << 16;
static_assert(kStripeCodecDataShards == 4 && kStripeCodecParityShards == 2,
              "the ported codec is fixed at 4 data + 2 parity shards");

// ---------------------------------------------------------------------------
// GF(2^8) with primitive polynomial 0x1d (G1 sponge_rpc.hpp:33).  The
// arithmetic is G1's bitwise Russian-peasant form, not a log/antilog table, so
// the two stay comparable operation for operation.
// ---------------------------------------------------------------------------
class SmallObjectStripeGf256 {
public:
    static constexpr uint8_t kPrimitivePolynomial = 0x1d;
    static constexpr size_t kFieldSize = 256;

    // G1 sponge_rpc.hpp:822-836, copied verbatim.
    static uint8_t mul(uint8_t a, uint8_t b) {
        uint8_t result = 0;
        while (b != 0) {
            if ((b & 1u) != 0) {
                result ^= a;
            }
            bool carry = (a & 0x80u) != 0;
            a = static_cast<uint8_t>(a << 1);
            if (carry) {
                a ^= kPrimitivePolynomial;
            }
            b = static_cast<uint8_t>(b >> 1);
        }
        return result;
    }

    // G1 sponge_rpc.hpp:838-844, copied verbatim.
    static uint8_t pow(uint8_t a, size_t power) {
        uint8_t result = 1;
        for (size_t i = 0; i < power; i++) {
            result = mul(result, a);
        }
        return result;
    }

    // [new] Multiplicative inverse (G1 never needed one).  GF(2^8)* is cyclic of
    // order 255, so a^254 == a^-1 for a != 0.
    static uint8_t inv(uint8_t a) {
        assert(a != 0);
        return pow(a, 254);
    }

    // G1 sponge_rpc.hpp:846-859, copied verbatim (keeping G1's rows/cols names):
    // the first k rows are the identity, row k+i is the Vandermonde row with
    // base (i + 1).  With k = 4 that yields coefficients {1,1,1,1} for parity 0
    // and {1,2,4,8} for parity 1.
    static void build_generator_matrix(
        uint8_t *matrix /* [kStripeCodecShardCount][kStripeCodecDataShards] */) {
        constexpr size_t rows = kStripeCodecShardCount;
        constexpr size_t cols = kStripeCodecDataShards;
        for (size_t row = 0; row < cols; row++) {
            for (size_t col = 0; col < cols; col++) {
                matrix[row * cols + col] = (row == col) ? 1 : 0;
            }
        }
        for (size_t row = cols; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {
                matrix[row * cols + col] =
                    pow(static_cast<uint8_t>(row - cols + 1), col);
            }
        }
    }

    // lut8[v] = coef * v: the 256-entry table G1 keeps in lut.mul
    // (sponge_rpc.hpp:804-807).  Also used by the [new] rebuild path, where the
    // coefficient is a reconstruction coefficient, not an encode coefficient.
    static void build_mul_lut(uint8_t coef, uint8_t *lut8) {
        for (size_t v = 0; v < kFieldSize; v++) {
            lut8[v] = mul(coef, static_cast<uint8_t>(v));
        }
    }

    // lut16[v] = lut8[low byte of v] | (lut8[high byte of v] << 8): G1's
    // pair-LUT construction (sponge_rpc.hpp:808-816), endian-agnostic, because
    // it reproduces the input byte order in the output.
    static void build_pair_lut(const uint8_t *lut8, uint16_t *lut16) {
        for (size_t v = 0; v < kStripeCodecGfPairLutSize; v++) {
            uint16_t in_pair = static_cast<uint16_t>(v);
            uint8_t in[sizeof(in_pair)] = {};
            uint8_t out[sizeof(in_pair)] = {};
            std::memcpy(in, &in_pair, sizeof(in_pair));
            out[0] = lut8[in[0]];
            out[1] = lut8[in[1]];
            std::memcpy(&lut16[v], out, sizeof(out));
        }
    }
};

// G1 SpongeShardParityLut (sponge_rpc.hpp:781-785).
struct SmallObjectStripeShardParityLut {
    uint8_t coef[kStripeCodecParityShards] = {0, 0};
    uint8_t mul[kStripeCodecParityShards][SmallObjectStripeGf256::kFieldSize] = {};
    uint16_t mul16[kStripeCodecParityShards][kStripeCodecGfPairLutSize] = {};
};

// G1 SpongeParityCodecCache (sponge_rpc.hpp:787-819): one entry per data shard.
// Constructing it builds the four parity LUTs on first use (1 MiB of tables);
// nothing here is constructed unless the encoder is actually called, so the
// ft_method == none path never touches it.
struct SmallObjectStripeParityCodecCache {
    SmallObjectStripeShardParityLut shard[kStripeCodecDataShards];

    SmallObjectStripeParityCodecCache() {
        uint8_t matrix[kStripeCodecShardCount * kStripeCodecDataShards] = {};
        SmallObjectStripeGf256::build_generator_matrix(matrix);
        for (size_t data_shard = 0; data_shard < kStripeCodecDataShards;
             data_shard++) {
            auto &lut = shard[data_shard];
            for (size_t parity_idx = 0; parity_idx < kStripeCodecParityShards;
                 parity_idx++) {
                lut.coef[parity_idx] =
                    matrix[(kStripeCodecDataShards + parity_idx) *
                               kStripeCodecDataShards +
                           data_shard];
                SmallObjectStripeGf256::build_mul_lut(lut.coef[parity_idx],
                                                      lut.mul[parity_idx]);
                SmallObjectStripeGf256::build_pair_lut(lut.mul[parity_idx],
                                                       lut.mul16[parity_idx]);
            }
        }
    }
};

// G1 sponge_parity_codec_cache() (sponge_rpc.hpp:862-865).
inline const SmallObjectStripeParityCodecCache &
small_object_stripe_parity_codec_cache() {
    static const SmallObjectStripeParityCodecCache cache;
    return cache;
}

// G1 sponge_shard_parity_lut() (sponge_rpc.hpp:867-871).
inline const SmallObjectStripeShardParityLut &small_object_stripe_shard_parity_lut(
    uint8_t data_shard_idx) {
    assert(data_shard_idx < kStripeCodecDataShards);
    return small_object_stripe_parity_codec_cache()
        .shard[data_shard_idx % kStripeCodecDataShards];
}

// ---------------------------------------------------------------------------
// Byte loops: dst[i] = lut(src[i]), dst[i] ^= lut(src[i]), dst[i] ^= src[i].
// ---------------------------------------------------------------------------

// G1 sponge_encode_parity_bytes_with_lut (sponge_rpc.hpp:873-888), copied
// verbatim: 16-bit pairs through the pair LUT, tail byte through the 8-bit LUT.
inline void small_object_stripe_apply_bytes_with_lut(const uint8_t *src,
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

// [new] XOR-accumulating twin of the loop above (same tables, same pair trick).
// G1's delta helper always wrote a fresh delta buffer, so it only needed the
// assigning form; the full-stripe encoder adds into an accumulator.
inline void small_object_stripe_xor_bytes_with_lut(const uint8_t *src,
                                                   size_t byte_count,
                                                   const uint8_t *lut8,
                                                   const uint16_t *lut16,
                                                   uint8_t *dst) {
    size_t i = 0;
    for (; i + sizeof(uint16_t) <= byte_count; i += sizeof(uint16_t)) {
        uint16_t in_pair = 0;
        uint16_t acc_pair = 0;
        std::memcpy(&in_pair, src + i, sizeof(in_pair));
        std::memcpy(&acc_pair, dst + i, sizeof(acc_pair));
        acc_pair ^= lut16[in_pair];
        std::memcpy(dst + i, &acc_pair, sizeof(acc_pair));
    }
    for (; i < byte_count; i++) {
        dst[i] ^= lut8[src[i]];
    }
}

// [new] 8-bit-only variants: the rebuild path has no cached pair LUT, because
// its coefficient depends on the missing-shard set rather than on a data shard.
inline void small_object_stripe_apply_bytes_with_lut8(const uint8_t *src,
                                                      size_t byte_count,
                                                      const uint8_t *lut8,
                                                      uint8_t *dst) {
    for (size_t i = 0; i < byte_count; i++) {
        dst[i] = lut8[src[i]];
    }
}

inline void small_object_stripe_xor_bytes_with_lut8(const uint8_t *src,
                                                    size_t byte_count,
                                                    const uint8_t *lut8,
                                                    uint8_t *dst) {
    for (size_t i = 0; i < byte_count; i++) {
        dst[i] ^= lut8[src[i]];
    }
}

// coef == 1 fast path (G1 sponge_rpc.hpp:901-910 memcpy/XOR), word at a time.
inline void small_object_stripe_xor_bytes(const uint8_t *src, size_t byte_count,
                                          uint8_t *dst) {
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= byte_count; i += sizeof(uint64_t)) {
        uint64_t in_word = 0;
        uint64_t acc_word = 0;
        std::memcpy(&in_word, src + i, sizeof(in_word));
        std::memcpy(&acc_word, dst + i, sizeof(acc_word));
        acc_word ^= in_word;
        std::memcpy(dst + i, &acc_word, sizeof(acc_word));
    }
    for (; i < byte_count; i++) {
        dst[i] ^= src[i];
    }
}

// dst = sum_k coefs[k] * srcs[k] over one contiguous byte window, GF(2^8).
// Used by the rebuild path; `dst` must not alias `srcs`.
inline void small_object_stripe_combine_bytes(const uint8_t *const *srcs,
                                              const uint8_t *coefs,
                                              size_t source_count,
                                              uint8_t *dst,
                                              size_t byte_count) {
    bool assigned = false;
    for (size_t k = 0; k < source_count; k++) {
        const uint8_t coef = coefs[k];
        if (coef == 0) {
            continue;
        }
        const uint8_t *src = srcs[k];
        if (coef == 1) {
            if (!assigned) {
                std::memcpy(dst, src, byte_count);
            } else {
                small_object_stripe_xor_bytes(src, byte_count, dst);
            }
        } else {
            uint8_t lut8[SmallObjectStripeGf256::kFieldSize] = {};
            SmallObjectStripeGf256::build_mul_lut(coef, lut8);
            if (!assigned) {
                small_object_stripe_apply_bytes_with_lut8(src, byte_count, lut8,
                                                          dst);
            } else {
                small_object_stripe_xor_bytes_with_lut8(src, byte_count, lut8,
                                                        dst);
            }
        }
        assigned = true;
    }
    if (!assigned) {
        // Unreachable for a valid plan (every survivor has a non-zero
        // coefficient), but keeps the output deterministic if it ever happens.
        std::memset(dst, 0, byte_count);
    }
}

// ---------------------------------------------------------------------------
// Encode.
// ---------------------------------------------------------------------------

// G1 sponge_encode_parity_delta_bytes (sponge_rpc.hpp:890-917), copied apart
// from the G1-only byte-count guard (G1 caps a batch at 32 KiB; a stripe shard
// is 256 KiB here).
inline bool small_object_stripe_encode_parity_delta(uint8_t data_shard_idx,
                                                    const uint8_t *data_delta,
                                                    size_t byte_count,
                                                    uint8_t *parity0_delta,
                                                    uint8_t *parity1_delta) {
    if (data_shard_idx >= kStripeCodecDataShards || data_delta == nullptr ||
        parity0_delta == nullptr || parity1_delta == nullptr) {
        return false;
    }
    const auto &lut = small_object_stripe_shard_parity_lut(data_shard_idx);
    if (lut.coef[0] == 1 && lut.coef[1] == 1) {
        std::memcpy(parity0_delta, data_delta, byte_count);
        std::memcpy(parity1_delta, data_delta, byte_count);
        return true;
    }
    if (lut.coef[0] == 1) {
        std::memcpy(parity0_delta, data_delta, byte_count);
        small_object_stripe_apply_bytes_with_lut(data_delta, byte_count,
                                                 lut.mul[1], lut.mul16[1],
                                                 parity1_delta);
        return true;
    }
    small_object_stripe_apply_bytes_with_lut(data_delta, byte_count, lut.mul[0],
                                             lut.mul16[0], parity0_delta);
    small_object_stripe_apply_bytes_with_lut(data_delta, byte_count, lut.mul[1],
                                             lut.mul16[1], parity1_delta);
    return true;
}

// [new] Full-stripe encode with assign semantics:
//   parity[p][i] = sum_j coef[p][j] * data[j][i]     (GF(2^8), poly 0x1d)
// with G1's coef table, touching only [shard_offset, shard_offset + byte_count)
// of every shard.  Equivalent to running G1's delta helper once per data shard
// into a zeroed parity shard; "the first contributing shard assigns" is G1's
// coef == 1 fast path and saves the extra zero-fill pass.
inline bool small_object_stripe_encode_shards(
    const void *const data_shards[kStripeCodecDataShards],
    void *const parity_shards[kStripeCodecParityShards], size_t byte_count,
    size_t shard_offset = 0) {
    for (size_t j = 0; j < kStripeCodecDataShards; j++) {
        if (data_shards[j] == nullptr) {
            return false;
        }
    }
    for (size_t p = 0; p < kStripeCodecParityShards; p++) {
        if (parity_shards[p] == nullptr) {
            return false;
        }
    }

    for (size_t p = 0; p < kStripeCodecParityShards; p++) {
        uint8_t *dst = static_cast<uint8_t *>(parity_shards[p]) + shard_offset;
        bool assigned = false;
        for (size_t j = 0; j < kStripeCodecDataShards; j++) {
            const auto &lut = small_object_stripe_shard_parity_lut(
                static_cast<uint8_t>(j));
            const uint8_t coef = lut.coef[p];
            if (coef == 0) {
                continue;
            }
            const uint8_t *src =
                static_cast<const uint8_t *>(data_shards[j]) + shard_offset;
            if (coef == 1) {
                if (assigned) {
                    small_object_stripe_xor_bytes(src, byte_count, dst);
                } else {
                    std::memcpy(dst, src, byte_count);
                }
            } else if (assigned) {
                small_object_stripe_xor_bytes_with_lut(
                    src, byte_count, lut.mul[p], lut.mul16[p], dst);
            } else {
                small_object_stripe_apply_bytes_with_lut(
                    src, byte_count, lut.mul[p], lut.mul16[p], dst);
            }
            assigned = true;
        }
        if (!assigned) {
            std::memset(dst, 0, byte_count);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// [new] Decode / rebuild.  G1 has no decoder: it recomputed parity deltas on the
// write path and never had to recover a shard.  The scheme here is the standard
// systematic-RS reconstruction on G1's own generator matrix: for the surviving
// rows S, invert the 4x4 submatrix M = G[S] and reconstruct
//   missing shard m = sum_k ( sum_j G[m][j] * M^-1[j][k] ) * survivor[k],
// i.e. every survivor contributes with one precomputed GF coefficient.  All 6
// single-missing and all 15 double-missing shard sets are solvable because the
// generator matrix is MDS: its identity rows are the four "infinity" points and
// rows 4/5 are the evaluation points 1 and 2, so any 4 of its 6 rows are
// independent.
// ---------------------------------------------------------------------------
class SmallObjectStripeRebuildTable {
public:
    static constexpr size_t kMaskCount = size_t{1} << kStripeCodecShardCount;

    struct Plan {
        bool valid = false;
        uint8_t missing_count = 0;
        uint8_t missing[kStripeCodecParityShards] = {0, 0};
        uint8_t survivor[kStripeCodecDataShards] = {0, 0, 0, 0};
        // coef[m][k]: coefficient of survivor k for the m-th missing shard.
        uint8_t coef[kStripeCodecParityShards][kStripeCodecDataShards] = {};
    };

    // Plan for one missing set: bit i of `missing_mask` means "shard i is
    // missing".  Masks with 0 or more than 2 missing bits yield an invalid plan;
    // the survivors used are the four lowest-indexed alive shards.  The
    // 64-entry table is built on first use.
    static const Plan &for_mask(uint8_t missing_mask) {
        static const std::array<Plan, kMaskCount> plans = build_all();
        return plans[missing_mask & static_cast<uint8_t>(kMaskCount - 1)];
    }

private:
    static std::array<Plan, kMaskCount> build_all() {
        std::array<Plan, kMaskCount> plans{};
        for (size_t mask = 0; mask < kMaskCount; mask++) {
            build_plan(static_cast<uint8_t>(mask), &plans[mask]);
        }
        return plans;
    }

    // Gauss-Jordan inversion of a 4x4 GF(2^8) matrix; false iff singular.
    static bool invert_matrix(const uint8_t *m, uint8_t *inv) {
        constexpr size_t n = kStripeCodecDataShards;
        uint8_t a[n][2 * n];
        for (size_t r = 0; r < n; r++) {
            for (size_t c = 0; c < n; c++) {
                a[r][c] = m[r * n + c];
                a[r][n + c] = (r == c) ? 1 : 0;
            }
        }
        for (size_t col = 0; col < n; col++) {
            size_t pivot = col;
            while (pivot < n && a[pivot][col] == 0) {
                pivot++;
            }
            if (pivot == n) {
                return false;
            }
            if (pivot != col) {
                for (size_t c = 0; c < 2 * n; c++) {
                    uint8_t tmp = a[col][c];
                    a[col][c] = a[pivot][c];
                    a[pivot][c] = tmp;
                }
            }
            const uint8_t pivot_value = a[col][col];
            if (pivot_value != 1) {
                const uint8_t pivot_inv =
                    SmallObjectStripeGf256::inv(pivot_value);
                for (size_t c = 0; c < 2 * n; c++) {
                    a[col][c] =
                        SmallObjectStripeGf256::mul(a[col][c], pivot_inv);
                }
            }
            for (size_t r = 0; r < n; r++) {
                if (r == col) {
                    continue;
                }
                const uint8_t factor = a[r][col];
                if (factor == 0) {
                    continue;
                }
                for (size_t c = 0; c < 2 * n; c++) {
                    a[r][c] ^= SmallObjectStripeGf256::mul(factor, a[col][c]);
                }
            }
        }
        for (size_t r = 0; r < n; r++) {
            for (size_t c = 0; c < n; c++) {
                inv[r * n + c] = a[r][n + c];
            }
        }
        return true;
    }

    static void build_plan(uint8_t mask, Plan *plan) {
        *plan = Plan{};
        uint8_t survivor_idx[kStripeCodecDataShards] = {0, 0, 0, 0};
        size_t survivor_count = 0;
        size_t missing_count = 0;
        for (uint8_t shard = 0; shard < kStripeCodecShardCount; shard++) {
            if ((mask & (1u << shard)) != 0) {
                if (missing_count < kStripeCodecParityShards) {
                    plan->missing[missing_count] = shard;
                }
                missing_count++;
            } else if (survivor_count < kStripeCodecDataShards) {
                survivor_idx[survivor_count++] = shard;
            }
        }
        if (missing_count == 0 || missing_count > kStripeCodecParityShards ||
            survivor_count < kStripeCodecDataShards) {
            return;  // 0 or more than 2 missing shards, or fewer than 4 alive
        }

        uint8_t matrix[kStripeCodecShardCount * kStripeCodecDataShards] = {};
        SmallObjectStripeGf256::build_generator_matrix(matrix);
        // Any 4 of the alive rows work (the generator matrix is MDS), so the
        // four lowest-indexed alive shards are used.
        uint8_t sub[kStripeCodecDataShards * kStripeCodecDataShards] = {};
        for (size_t k = 0; k < kStripeCodecDataShards; k++) {
            for (size_t j = 0; j < kStripeCodecDataShards; j++) {
                sub[k * kStripeCodecDataShards + j] =
                    matrix[survivor_idx[k] * kStripeCodecDataShards + j];
            }
        }
        uint8_t sub_inv[kStripeCodecDataShards * kStripeCodecDataShards] = {};
        if (!invert_matrix(sub, sub_inv)) {
            return;
        }

        plan->valid = true;
        plan->missing_count = static_cast<uint8_t>(missing_count);
        for (size_t k = 0; k < kStripeCodecDataShards; k++) {
            plan->survivor[k] = survivor_idx[k];
        }
        for (size_t m = 0; m < missing_count; m++) {
            for (size_t k = 0; k < kStripeCodecDataShards; k++) {
                uint8_t acc = 0;
                for (size_t j = 0; j < kStripeCodecDataShards; j++) {
                    acc ^= SmallObjectStripeGf256::mul(
                        matrix[plan->missing[m] * kStripeCodecDataShards + j],
                        sub_inv[j * kStripeCodecDataShards + k]);
                }
                plan->coef[m][k] = acc;
            }
        }
    }
};

// Rebuild one shard from an explicit plan; writes only
// [shard_offset, shard_offset + byte_count) of `dst`.
inline bool small_object_stripe_rebuild_target(
    const SmallObjectStripeRebuildTable::Plan &plan, uint8_t missing_idx,
    const void *const survivor_ptrs[kStripeCodecDataShards], void *dst,
    size_t byte_count, size_t shard_offset = 0) {
    if (!plan.valid || dst == nullptr) {
        return false;
    }
    size_t slot = plan.missing_count;
    for (size_t m = 0; m < plan.missing_count; m++) {
        if (plan.missing[m] == missing_idx) {
            slot = m;
            break;
        }
    }
    if (slot == plan.missing_count) {
        return false;
    }
    const uint8_t *srcs[kStripeCodecDataShards];
    uint8_t coefs[kStripeCodecDataShards];
    for (size_t k = 0; k < kStripeCodecDataShards; k++) {
        if (survivor_ptrs[k] == nullptr) {
            return false;
        }
        srcs[k] = static_cast<const uint8_t *>(survivor_ptrs[k]) + shard_offset;
        coefs[k] = plan.coef[slot][k];
    }
    small_object_stripe_combine_bytes(srcs, coefs, kStripeCodecDataShards,
                                      static_cast<uint8_t *>(dst) + shard_offset,
                                      byte_count);
    return true;
}

// Rebuild every missing shard of one stripe.
//   alive_mask  one bit per shard: bit s set means shard s is alive (readable);
//               its complement must name exactly 1 or 2 shards to rebuild.
//   shards      one buffer per shard, data shards first (0..3) and parity
//               shards last (4..5), indexed exactly like
//               SmallObjectStripeManager::Stripe::shard_base / SlotLayout.
//               The alive entries are read, the missing ones are written in
//               place, so all six pointers must be non-null.
// One call covers all 6 single-missing and all 15 double-missing sets; the
// survivors actually used are the four lowest-indexed alive shards, and any
// four alive shards are enough (the generator matrix is MDS).
inline bool small_object_stripe_rebuild_shards(
    uint8_t alive_mask, void *const shards[kStripeCodecShardCount],
    size_t byte_count, size_t shard_offset = 0) {
    const uint8_t missing_mask =
        static_cast<uint8_t>(~alive_mask) &
        static_cast<uint8_t>((1u << kStripeCodecShardCount) - 1u);
    if (missing_mask == 0) {
        return false;  // nothing to rebuild
    }
    size_t alive = 0;
    for (size_t s = 0; s < kStripeCodecShardCount; s++) {
        if (shards[s] == nullptr) {
            return false;  // a missing shard still needs a write buffer
        }
        if ((alive_mask & (1u << s)) != 0) {
            alive++;
        }
    }
    if (alive < kStripeCodecDataShards) {
        return false;  // fewer than the four shards the code needs
    }
    const auto &plan = SmallObjectStripeRebuildTable::for_mask(missing_mask);
    if (!plan.valid) {
        return false;  // more than 2 missing, or fewer than 4 alive
    }
    const void *survivors[kStripeCodecDataShards];
    for (size_t k = 0; k < kStripeCodecDataShards; k++) {
        survivors[k] = shards[plan.survivor[k]];
    }
    for (size_t m = 0; m < plan.missing_count; m++) {
        if (!small_object_stripe_rebuild_target(
                plan, plan.missing[m], survivors, shards[plan.missing[m]],
                byte_count, shard_offset)) {
            return false;
        }
    }
    return true;
}

// Rebuild a single missing shard from four explicitly listed survivors (the
// repair path: one dead endpoint next to four live ones).  Any input order is
// accepted; the survivors are sorted internally, because the plan's
// coefficients and the plan's survivor slots are both in ascending shard order.
inline bool small_object_stripe_rebuild_one(
    uint8_t missing_shard_idx,
    const uint8_t survivor_idx[kStripeCodecDataShards],
    const void *const survivors[kStripeCodecDataShards], void *dst,
    size_t byte_count, size_t shard_offset = 0) {
    if (missing_shard_idx >= kStripeCodecShardCount || dst == nullptr) {
        return false;
    }
    uint8_t sorted_idx[kStripeCodecDataShards] = {0, 0, 0, 0};
    const void *sorted_ptrs[kStripeCodecDataShards] = {nullptr, nullptr,
                                                       nullptr, nullptr};
    uint8_t mask = static_cast<uint8_t>((1u << kStripeCodecShardCount) - 1u);
    for (size_t k = 0; k < kStripeCodecDataShards; k++) {
        if (survivor_idx[k] >= kStripeCodecShardCount ||
            survivors[k] == nullptr || survivor_idx[k] == missing_shard_idx) {
            return false;
        }
        sorted_idx[k] = survivor_idx[k];
        sorted_ptrs[k] = survivors[k];
        mask &= static_cast<uint8_t>(~(1u << survivor_idx[k]));
    }
    for (size_t k = 0; k < kStripeCodecDataShards; k++) {
        for (size_t l = k + 1; l < kStripeCodecDataShards; l++) {
            if (sorted_idx[l] < sorted_idx[k]) {
                const uint8_t tmp_idx = sorted_idx[k];
                sorted_idx[k] = sorted_idx[l];
                sorted_idx[l] = tmp_idx;
                const void *tmp_ptr = sorted_ptrs[k];
                sorted_ptrs[k] = sorted_ptrs[l];
                sorted_ptrs[l] = tmp_ptr;
            }
        }
    }
    for (size_t k = 0; k + 1 < kStripeCodecDataShards; k++) {
        if (sorted_idx[k] == sorted_idx[k + 1]) {
            return false;  // duplicate survivor
        }
    }
    if ((mask & (1u << missing_shard_idx)) == 0) {
        return false;  // the target is not actually missing
    }
    const auto &plan = SmallObjectStripeRebuildTable::for_mask(mask);
    return small_object_stripe_rebuild_target(plan, missing_shard_idx,
                                              sorted_ptrs, dst, byte_count,
                                              shard_offset);
}

}  // namespace FarLib::cache
