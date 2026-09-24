#include "cache/alloc/small_object_stripe_codec.hpp"
#include "hydra/page_codec.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <latch>
#include <random>
#include <thread>
#include <vector>

extern "C" void __real_ec_init_tables(int k, int m, unsigned char *matrix,
                                       unsigned char *tables);

namespace {

std::atomic<unsigned> g_ec_init_tables_calls{0};

}  // namespace

extern "C" void __wrap_ec_init_tables(int k, int m, unsigned char *matrix,
                                       unsigned char *tables) {
    g_ec_init_tables_calls.fetch_add(1, std::memory_order_relaxed);
    __real_ec_init_tables(k, m, matrix, tables);
}

namespace {

enum class Pattern { zero, ff, random };

constexpr std::array<std::size_t, 14> kByteCounts = {
    0, 1, 2, 3, 15, 16, 31, 32, 63, 64, 65, 2048, 4096, 65536};

void fill_data(std::array<std::vector<std::uint8_t>, 4> &raw,
               const std::array<std::size_t, 4> &offsets,
               std::size_t byte_count, Pattern pattern,
               std::mt19937 &random) {
    for (std::size_t shard = 0; shard < raw.size(); ++shard) {
        std::fill(raw[shard].begin(), raw[shard].end(), 0xa5);
        for (std::size_t i = 0; i < byte_count; ++i) {
            if (pattern == Pattern::zero) {
                raw[shard][offsets[shard] + i] = 0;
            } else if (pattern == Pattern::ff) {
                raw[shard][offsets[shard] + i] = 0xff;
            } else {
                raw[shard][offsets[shard] + i] =
                    static_cast<std::uint8_t>(random() & 0xffu);
            }
        }
    }
}

void initialize_parity(std::array<std::vector<std::uint8_t>, 2> &raw,
                       std::size_t storage_bytes) {
    for (auto &shard : raw) {
        shard.assign(storage_bytes, 0xa5);
    }
}

void assert_data_guards(const std::array<std::vector<std::uint8_t>, 4> &raw,
                        const std::array<std::size_t, 4> &offsets,
                        std::size_t byte_count) {
    for (std::size_t shard = 0; shard < raw.size(); ++shard) {
        for (std::size_t i = 0; i < offsets[shard]; ++i) {
            assert(raw[shard][i] == 0xa5);
        }
        for (std::size_t i = offsets[shard] + byte_count;
             i < raw[shard].size(); ++i) {
            assert(raw[shard][i] == 0xa5);
        }
    }
}

void run_lut_equivalence_case(std::size_t byte_count, Pattern pattern,
                              std::mt19937 &random) {
    constexpr std::array<std::size_t, 4> kDataOffsets = {1, 3, 5, 7};
    constexpr std::array<std::size_t, 2> kParityOffsets = {9, 11};
    constexpr std::size_t kGuardBytes = 64;

    std::array<std::vector<std::uint8_t>, 4> data_raw;
    for (auto &shard : data_raw) shard.resize(byte_count + kGuardBytes);
    fill_data(data_raw, kDataOffsets, byte_count, pattern, random);
    const auto data_before = data_raw;

    std::array<const void *, 4> data_ptrs{};
    for (std::size_t shard = 0; shard < data_raw.size(); ++shard) {
        data_ptrs[shard] = data_raw[shard].data() + kDataOffsets[shard];
    }

    std::array<std::vector<std::uint8_t>, 2> old_raw;
    std::array<std::vector<std::uint8_t>, 2> isal_raw;
    initialize_parity(old_raw, byte_count + kGuardBytes);
    initialize_parity(isal_raw, byte_count + kGuardBytes);

    std::array<void *, 2> old_parity{};
    std::array<void *, 2> isal_parity{};
    for (std::size_t shard = 0; shard < 2; ++shard) {
        old_parity[shard] = old_raw[shard].data() + kParityOffsets[shard];
        isal_parity[shard] = isal_raw[shard].data() + kParityOffsets[shard];
    }

    assert(FarLib::cache::small_object_stripe_encode_shards(
        data_ptrs.data(), old_parity.data(), byte_count));
    assert(FarLib::hydra::page_codec_encode_4plus2(
        data_ptrs.data(), isal_parity.data(), byte_count));

    for (std::size_t shard = 0; shard < 2; ++shard) {
        assert(old_raw[shard] == isal_raw[shard]);
    }
    assert(data_raw == data_before);
    assert_data_guards(data_raw, kDataOffsets, byte_count);
}

unsigned bit_count(unsigned value) {
    unsigned count = 0;
    while (value != 0) {
        count += value & 1u;
        value >>= 1;
    }
    return count;
}

void run_recovery_masks() {
    constexpr std::size_t kByteCount = 2048;
    std::mt19937 random(0x5eed1234u);

    std::array<std::vector<std::uint8_t>, 4> data;
    std::array<const void *, 4> data_ptrs{};
    for (std::size_t shard = 0; shard < data.size(); ++shard) {
        data[shard].resize(kByteCount);
        for (auto &byte : data[shard]) {
            byte = static_cast<std::uint8_t>(random() & 0xffu);
        }
        data_ptrs[shard] = data[shard].data();
    }

    std::array<std::vector<std::uint8_t>, 2> parity;
    for (auto &shard : parity) shard.resize(kByteCount);
    std::array<void *, 2> parity_ptrs = {parity[0].data(), parity[1].data()};
    assert(FarLib::hydra::page_codec_encode_4plus2(
        data_ptrs.data(), parity_ptrs.data(), kByteCount));

    std::array<std::vector<std::uint8_t>, 6> encoded;
    for (std::size_t shard = 0; shard < 4; ++shard) encoded[shard] = data[shard];
    for (std::size_t shard = 0; shard < 2; ++shard) {
        encoded[4 + shard] = parity[shard];
    }

    unsigned single_masks = 0;
    unsigned double_masks = 0;
    for (unsigned missing_mask = 1; missing_mask < (1u << 6); ++missing_mask) {
        const unsigned missing = bit_count(missing_mask);
        if (missing == 1) {
            ++single_masks;
        } else if (missing == 2) {
            ++double_masks;
        } else {
            continue;
        }

        auto work = encoded;
        std::array<void *, 6> shard_ptrs{};
        for (std::size_t shard = 0; shard < work.size(); ++shard) {
            shard_ptrs[shard] = work[shard].data();
            if ((missing_mask & (1u << shard)) != 0) {
                std::fill(work[shard].begin(), work[shard].end(), 0xcc);
            }
        }
        const std::uint8_t alive_mask = static_cast<std::uint8_t>(
            ((1u << 6) - 1u) ^ missing_mask);
        assert(FarLib::cache::small_object_stripe_rebuild_shards(
            alive_mask, shard_ptrs.data(), kByteCount));
        for (std::size_t shard = 0; shard < work.size(); ++shard) {
            assert(work[shard] == encoded[shard]);
        }
    }
    assert(single_masks == 6);
    assert(double_masks == 15);
}

void run_invalid_inputs() {
    std::array<std::array<std::uint8_t, 64>, 4> data{};
    std::array<std::array<std::uint8_t, 64>, 2> parity{};
    std::array<const void *, 4> data_ptrs{};
    std::array<void *, 2> parity_ptrs{};
    for (std::size_t i = 0; i < data.size(); ++i) data_ptrs[i] = data[i].data();
    for (std::size_t i = 0; i < parity.size(); ++i) {
        parity_ptrs[i] = parity[i].data();
    }

    assert(!FarLib::hydra::page_codec_encode_4plus2(
        nullptr, parity_ptrs.data(), 1));
    assert(!FarLib::hydra::page_codec_encode_4plus2(
        data_ptrs.data(), nullptr, 1));

    auto bad_data = data_ptrs;
    bad_data[2] = nullptr;
    assert(!FarLib::hydra::page_codec_encode_4plus2(
        bad_data.data(), parity_ptrs.data(), 1));
    auto bad_parity = parity_ptrs;
    bad_parity[1] = nullptr;
    assert(!FarLib::hydra::page_codec_encode_4plus2(
        data_ptrs.data(), bad_parity.data(), 1));

    assert(FarLib::hydra::page_codec_encode_4plus2(
        data_ptrs.data(), parity_ptrs.data(), 0));
    assert(!FarLib::hydra::page_codec_encode_4plus2(
        data_ptrs.data(), parity_ptrs.data(),
        static_cast<std::size_t>(INT_MAX) + 1));
}

void run_concurrent_first_call() {
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kBytes = 65;
    std::latch ready(kThreads + 1);
    std::latch start(1);
    std::array<bool, kThreads> results{};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (std::size_t thread_index = 0; thread_index < kThreads;
         ++thread_index) {
        workers.emplace_back([&, thread_index] {
            std::array<std::array<std::uint8_t, kBytes>, 4> data{};
            std::array<std::array<std::uint8_t, kBytes>, 2> parity{};
            std::array<std::array<std::uint8_t, kBytes>, 2> expected{};
            for (std::size_t shard = 0; shard < data.size(); ++shard) {
                for (std::size_t i = 0; i < kBytes; ++i) {
                    data[shard][i] = static_cast<std::uint8_t>(
                        thread_index * 17 + shard * 5 + i);
                }
            }
            std::array<const void *, 4> data_ptrs{};
            std::array<void *, 2> parity_ptrs{};
            for (std::size_t shard = 0; shard < data.size(); ++shard) {
                data_ptrs[shard] = data[shard].data();
            }
            for (std::size_t shard = 0; shard < parity.size(); ++shard) {
                parity_ptrs[shard] = parity[shard].data();
            }

            // Warm the independent LUT oracle before releasing the new-codec
            // start gate.  This keeps the first concurrent call focused on
            // ISA-L's shared table initialization.
            std::array<void *, 2> expected_ptrs{};
            for (std::size_t shard = 0; shard < expected.size(); ++shard) {
                expected_ptrs[shard] = expected[shard].data();
            }
            assert(FarLib::cache::small_object_stripe_encode_shards(
                data_ptrs.data(), expected_ptrs.data(), kBytes));

            ready.count_down();
            start.wait();
            const bool encoded = FarLib::hydra::page_codec_encode_4plus2(
                data_ptrs.data(), parity_ptrs.data(), kBytes);
            results[thread_index] = encoded && parity == expected;
        });
    }

    ready.count_down();
    ready.wait();
    start.count_down();
    for (auto &worker : workers) worker.join();
    for (bool result : results) assert(result);
    assert(g_ec_init_tables_calls.load(std::memory_order_acquire) == 1);
}

}  // namespace

int main() {
    // This must be the first codec call: it proves concurrent first use shares
    // one ISA-L matrix/table initialization.
    run_concurrent_first_call();

    std::mt19937 random(0x12345678u);
    for (const auto byte_count : kByteCounts) {
        run_lut_equivalence_case(byte_count, Pattern::zero, random);
        run_lut_equivalence_case(byte_count, Pattern::ff, random);
        run_lut_equivalence_case(byte_count, Pattern::random, random);
    }
    run_invalid_inputs();
    run_recovery_masks();
    assert(g_ec_init_tables_calls.load(std::memory_order_acquire) == 1);
    std::printf("HYDRA_PAGE_CODEC_PASS isa_l=1 ec_init_tables=1 "
                "masks=6+15 sizes=%zu patterns=3\n",
                kByteCounts.size());
}
