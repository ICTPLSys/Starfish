// CPU-only regression coverage for the ISA-L small-object stripe encoder.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/alloc/small_object_stripe_codec.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

extern "C" void __real_ec_init_tables(int k, int m, unsigned char *a,
                                       unsigned char *g_tbls);

namespace {

std::atomic<unsigned> g_ec_init_tables_calls{0};

constexpr size_t kDataShards = FarLib::cache::kStripeCodecDataShards;
constexpr size_t kParityShards = FarLib::cache::kStripeCodecParityShards;
constexpr size_t kShardCount = FarLib::cache::kStripeCodecShardCount;
constexpr std::array<std::array<uint8_t, kDataShards>, kParityShards>
    kParityMatrix{{{{1, 1, 1, 1}}, {{1, 2, 4, 8}}}};

uint8_t gf_mul(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    while (b != 0) {
        if ((b & 1u) != 0) result ^= a;
        const bool carry = (a & 0x80u) != 0;
        a = static_cast<uint8_t>(a << 1);
        if (carry) a ^= 0x1d;
        b = static_cast<uint8_t>(b >> 1);
    }
    return result;
}

uint8_t canary(size_t shard, size_t byte) {
    return static_cast<uint8_t>(0xa5u ^ (0x2du * shard) ^
                                (0x17u * byte));
}

uint8_t data_byte(size_t shard, size_t byte) {
    return static_cast<uint8_t>(0x13u + 0x37u * shard + 0x0bu * byte +
                                ((shard + 1u) * (byte + 3u) >> 1));
}

size_t popcount(uint8_t value) {
    size_t count = 0;
    for (size_t bit = 0; bit < kShardCount; ++bit) {
        count += (value & static_cast<uint8_t>(1u << bit)) != 0;
    }
    return count;
}

void assert_outside_window_unchanged(const std::vector<uint8_t> &actual,
                                     const std::vector<uint8_t> &before,
                                     size_t begin, size_t length) {
    assert(actual.size() == before.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (i < begin || i >= begin + length) assert(actual[i] == before[i]);
    }
}

void test_cold_initialization_is_once() {
    assert(g_ec_init_tables_calls.load(std::memory_order_relaxed) == 0);

    constexpr size_t kThreadCount = 8;
    constexpr size_t kByteCount = 4096;
    std::atomic<size_t> ready{0};
    std::atomic<bool> go{false};
    std::atomic<size_t> failures{0};
    std::array<std::thread, kThreadCount> workers;

    for (size_t tid = 0; tid < kThreadCount; ++tid) {
        workers[tid] = std::thread([&, tid] {
            std::array<std::array<uint8_t, kByteCount>, kShardCount> shards{};
            for (size_t shard = 0; shard < kDataShards; ++shard) {
                for (size_t byte = 0; byte < kByteCount; ++byte) {
                    shards[shard][byte] = static_cast<uint8_t>(
                        data_byte(shard, byte) ^ static_cast<uint8_t>(tid));
                }
            }
            for (size_t shard = kDataShards; shard < kShardCount; ++shard) {
                std::fill(shards[shard].begin(), shards[shard].end(),
                          static_cast<uint8_t>(0xc0u + shard + tid));
            }
            const void *data[kDataShards] = {shards[0].data(), shards[1].data(),
                                             shards[2].data(), shards[3].data()};
            void *parity[kParityShards] = {shards[4].data(), shards[5].data()};

            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) != kThreadCount) {
                std::this_thread::yield();
            }
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (!FarLib::cache::small_object_stripe_encode_shards(
                    data, parity, kByteCount, 0)) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kThreadCount) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();

    assert(failures.load(std::memory_order_relaxed) == 0);
    assert(g_ec_init_tables_calls.load(std::memory_order_relaxed) == 1);

    FarLib::cache::small_object_stripe_initialize_encoder();
    assert(g_ec_init_tables_calls.load(std::memory_order_relaxed) == 1);
}

void test_encode_oracle_for_length(size_t byte_count, size_t shard_offset) {
    constexpr size_t kPointerBias = 1;
    constexpr size_t kSuffix = 11;
    const size_t window_begin = kPointerBias + shard_offset;
    const size_t storage_size = window_begin + byte_count + kSuffix;

    std::array<std::vector<uint8_t>, kDataShards> data_storage;
    std::array<std::vector<uint8_t>, kParityShards> parity_storage;
    for (size_t shard = 0; shard < kDataShards; ++shard) {
        data_storage[shard].resize(storage_size);
        for (size_t byte = 0; byte < storage_size; ++byte) {
            data_storage[shard][byte] = canary(shard, byte);
        }
        for (size_t byte = 0; byte < byte_count; ++byte) {
            data_storage[shard][window_begin + byte] =
                data_byte(shard, byte);
        }
    }
    for (size_t parity = 0; parity < kParityShards; ++parity) {
        parity_storage[parity].resize(storage_size);
        for (size_t byte = 0; byte < storage_size; ++byte) {
            parity_storage[parity][byte] = canary(kDataShards + parity, byte);
        }
    }

    const auto data_before = data_storage;
    const auto parity_before = parity_storage;
    const void *data[kDataShards] = {
        data_storage[0].data() + kPointerBias,
        data_storage[1].data() + kPointerBias,
        data_storage[2].data() + kPointerBias,
        data_storage[3].data() + kPointerBias};
    void *parity[kParityShards] = {
        parity_storage[0].data() + kPointerBias,
        parity_storage[1].data() + kPointerBias};

    assert(FarLib::cache::small_object_stripe_encode_shards(
        data, parity, byte_count, shard_offset));
    assert(g_ec_init_tables_calls.load(std::memory_order_relaxed) == 1);

    for (size_t shard = 0; shard < kDataShards; ++shard) {
        assert(data_storage[shard] == data_before[shard]);
    }
    for (size_t parity_idx = 0; parity_idx < kParityShards; ++parity_idx) {
        assert_outside_window_unchanged(parity_storage[parity_idx],
                                        parity_before[parity_idx], window_begin,
                                        byte_count);
        for (size_t byte = 0; byte < byte_count; ++byte) {
            uint8_t expected = 0;
            for (size_t shard = 0; shard < kDataShards; ++shard) {
                expected ^= gf_mul(
                    kParityMatrix[parity_idx][shard],
                    data_storage[shard][window_begin + byte]);
            }
            assert(parity_storage[parity_idx][window_begin + byte] == expected);
        }
    }
}

void test_lengths_offsets_and_canaries() {
    constexpr std::array<size_t, 13> kLengths = {
        0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 4072, 4096, 65536};
    for (const size_t length : kLengths) {
        test_encode_oracle_for_length(length, 3);
    }
    test_encode_oracle_for_length(4096, 0);
}

void test_invalid_arguments_do_not_touch_buffers() {
    std::array<uint8_t, 32> data_buffer{};
    std::array<uint8_t, 32> parity_buffer{};
    for (size_t i = 0; i < data_buffer.size(); ++i) {
        data_buffer[i] = static_cast<uint8_t>(0x31u + i);
        parity_buffer[i] = static_cast<uint8_t>(0x91u + i);
    }
    const auto data_before = data_buffer;
    const auto parity_before = parity_buffer;
    const void *data[kDataShards] = {data_buffer.data(), data_buffer.data(),
                                     data_buffer.data(), data_buffer.data()};
    void *parity[kParityShards] = {parity_buffer.data(), parity_buffer.data()};

    assert(!FarLib::cache::small_object_stripe_encode_shards(
        static_cast<const void *const *>(nullptr), parity, 1, 0));
    assert(!FarLib::cache::small_object_stripe_encode_shards(
        data, static_cast<void *const *>(nullptr), 1, 0));
    assert(data_buffer == data_before);
    assert(parity_buffer == parity_before);

    const void *bad_data[kDataShards] = {data_buffer.data(), data_buffer.data(),
                                         nullptr, data_buffer.data()};
    assert(!FarLib::cache::small_object_stripe_encode_shards(
        bad_data, parity, 1, 0));
    assert(data_buffer == data_before);
    assert(parity_buffer == parity_before);

    void *bad_parity[kParityShards] = {parity_buffer.data(), nullptr};
    assert(!FarLib::cache::small_object_stripe_encode_shards(
        data, bad_parity, 1, 0));
    assert(data_buffer == data_before);
    assert(parity_buffer == parity_before);

    constexpr size_t kTooLong =
        static_cast<size_t>(std::numeric_limits<int>::max()) + 1u;
    assert(!FarLib::cache::small_object_stripe_encode_shards(
        data, parity, kTooLong, 0));
    assert(data_buffer == data_before);
    assert(parity_buffer == parity_before);
}

void test_rebuild_all_single_and_double_erasures() {
    constexpr size_t kByteCount = 4096;
    std::array<std::vector<uint8_t>, kShardCount> original;
    for (size_t shard = 0; shard < kShardCount; ++shard) {
        original[shard].resize(kByteCount);
        for (size_t byte = 0; byte < kByteCount; ++byte) {
            original[shard][byte] = canary(shard, byte) ^
                                    static_cast<uint8_t>(0x5du * shard);
        }
    }
    const void *data[kDataShards] = {original[0].data(), original[1].data(),
                                     original[2].data(), original[3].data()};
    void *parity[kParityShards] = {original[4].data(), original[5].data()};
    assert(FarLib::cache::small_object_stripe_encode_shards(
        data, parity, kByteCount, 0));

    for (uint8_t missing = 1; missing < (1u << kShardCount); ++missing) {
        const size_t missing_count = popcount(missing);
        if (missing_count != 1 && missing_count != 2) continue;

        auto rebuilt = original;
        for (size_t shard = 0; shard < kShardCount; ++shard) {
            if ((missing & static_cast<uint8_t>(1u << shard)) != 0) {
                std::fill(rebuilt[shard].begin(), rebuilt[shard].end(),
                          static_cast<uint8_t>(0xe0u + shard));
            }
        }
        void *shards[kShardCount] = {rebuilt[0].data(), rebuilt[1].data(),
                                     rebuilt[2].data(), rebuilt[3].data(),
                                     rebuilt[4].data(), rebuilt[5].data()};
        const uint8_t alive = static_cast<uint8_t>(
            (~missing) & static_cast<uint8_t>((1u << kShardCount) - 1u));
        assert(FarLib::cache::small_object_stripe_rebuild_shards(
            alive, shards, kByteCount, 0));
        for (size_t shard = 0; shard < kShardCount; ++shard) {
            assert(rebuilt[shard] == original[shard]);
        }
    }
}

}  // namespace

extern "C" void __wrap_ec_init_tables(int k, int m, unsigned char *a,
                                        unsigned char *g_tbls) {
    g_ec_init_tables_calls.fetch_add(1, std::memory_order_relaxed);
    __real_ec_init_tables(k, m, a, g_tbls);
}

int main() {
    test_cold_initialization_is_once();
    test_lengths_offsets_and_canaries();
    test_invalid_arguments_do_not_touch_buffers();
    test_rebuild_all_single_and_double_erasures();
    assert(g_ec_init_tables_calls.load(std::memory_order_relaxed) == 1);
    std::cout << "EC_ISAL_CODEC_PASS\n";
    return 0;
}
