#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/alloc/ec_direct_codec.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

constexpr size_t kDataShards = FarLib::cache::kEcDirectDataShards;
constexpr size_t kParityShards = FarLib::cache::kEcDirectParityShards;
constexpr size_t kMaxSlotSize = FarLib::cache::kEcDirectMaxSlotSize;
constexpr size_t kGuardBytes = 23;

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
    return static_cast<uint8_t>(0x91u ^ (0x27u * shard) ^
                                (0x0du * byte));
}

uint8_t object_byte(size_t shard, size_t byte) {
    return static_cast<uint8_t>(0x17u + 0x31u * shard + 0x0bu * byte +
                                ((shard + 3u) * (byte + 5u) >> 1));
}

void assert_padding_unchanged(const std::vector<uint8_t> &actual,
                              const std::vector<uint8_t> &before,
                              size_t begin, size_t length) {
    assert(actual.size() == before.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (i < begin || i >= begin + length) assert(actual[i] == before[i]);
    }
}

void make_oracle(const std::array<std::vector<uint8_t>, kDataShards> &data,
                 const std::array<uint32_t, kDataShards> &sizes,
                 size_t slot_size, size_t data_begin,
                 std::array<std::vector<uint8_t>, kParityShards> &expected) {
    for (size_t parity = 0; parity < kParityShards; ++parity) {
        expected[parity].assign(slot_size, 0);
        for (size_t byte = 0; byte < slot_size; ++byte) {
            uint8_t value = 0;
            for (size_t shard = 0; shard < kDataShards; ++shard) {
                const uint8_t source =
                    byte < sizes[shard]
                        ? data[shard][data_begin + byte]
                        : static_cast<uint8_t>(0);
                value ^= gf_mul(kParityMatrix[parity][shard], source);
            }
            expected[parity][byte] = value;
        }
    }
}

void test_variable_lengths_and_padding_snapshot(
    const std::array<uint32_t, kDataShards> &sizes,
    bool use_null_for_holes, const void *zero_pad_override = nullptr) {
    constexpr size_t slot_size = kMaxSlotSize;
    constexpr size_t data_begin = kGuardBytes;
    const size_t storage_size = data_begin + slot_size + kGuardBytes;

    std::array<std::vector<uint8_t>, kDataShards> data_storage;
    for (size_t shard = 0; shard < kDataShards; ++shard) {
        data_storage[shard].resize(storage_size);
        for (size_t byte = 0; byte < storage_size; ++byte) {
            data_storage[shard][byte] = canary(shard, byte);
        }
        for (size_t byte = 0; byte < sizes[shard]; ++byte) {
            data_storage[shard][data_begin + byte] = object_byte(shard, byte);
        }
    }
    const auto data_before = data_storage;

    std::array<std::vector<uint8_t>, kParityShards> parity_storage;
    for (size_t parity = 0; parity < kParityShards; ++parity) {
        parity_storage[parity].resize(storage_size);
        for (size_t byte = 0; byte < storage_size; ++byte) {
            parity_storage[parity][byte] = canary(kDataShards + parity, byte);
        }
    }
    const auto parity_before = parity_storage;

    std::vector<uint8_t> zero_storage(storage_size, 0);
    for (size_t byte = 0; byte < data_begin; ++byte) {
        zero_storage[byte] = canary(kDataShards + kParityShards, byte);
        zero_storage[data_begin + slot_size + byte] =
            canary(kDataShards + kParityShards, data_begin + slot_size + byte);
    }

    const void *data[kDataShards] = {};
    for (size_t shard = 0; shard < kDataShards; ++shard) {
        if (sizes[shard] == 0 && use_null_for_holes) {
            data[shard] = nullptr;
        } else {
            data[shard] = data_storage[shard].data() + data_begin;
        }
    }
    void *parity[kParityShards] = {
        parity_storage[0].data() + data_begin,
        parity_storage[1].data() + data_begin};
    const void *zero_pad = zero_pad_override != nullptr
                               ? zero_pad_override
                               : zero_storage.data() + data_begin;

    std::array<std::vector<uint8_t>, kParityShards> expected;
    make_oracle(data_storage, sizes, slot_size, data_begin, expected);

    assert(FarLib::cache::encode_direct_shards(
        data, sizes.data(), parity, slot_size, zero_pad));

    for (size_t shard = 0; shard < kDataShards; ++shard) {
        assert(data_storage[shard] == data_before[shard]);
    }
    for (size_t parity_idx = 0; parity_idx < kParityShards; ++parity_idx) {
        assert_padding_unchanged(parity_storage[parity_idx],
                                  parity_before[parity_idx], data_begin,
                                  slot_size);
        for (size_t byte = 0; byte < slot_size; ++byte) {
            assert(parity_storage[parity_idx][data_begin + byte] ==
                   expected[parity_idx][byte]);
        }
    }
}

void test_invalid_arguments_do_not_touch_parity() {
    constexpr size_t slot_size = 128;
    std::array<std::array<uint8_t, slot_size>, kDataShards> data_storage{};
    std::array<std::array<uint8_t, slot_size>, kParityShards> parity_storage{};
    for (size_t shard = 0; shard < kDataShards; ++shard) {
        for (size_t byte = 0; byte < slot_size; ++byte) {
            data_storage[shard][byte] = object_byte(shard, byte);
        }
    }
    for (size_t parity = 0; parity < kParityShards; ++parity) {
        for (size_t byte = 0; byte < slot_size; ++byte) {
            parity_storage[parity][byte] = canary(kDataShards + parity, byte);
        }
    }
    const auto parity_before = parity_storage;

    const void *data[kDataShards] = {data_storage[0].data(),
                                     data_storage[1].data(),
                                     data_storage[2].data(),
                                     data_storage[3].data()};
    uint32_t sizes[kDataShards] = {slot_size, slot_size, slot_size, slot_size};
    void *parity[kParityShards] = {parity_storage[0].data(),
                                   parity_storage[1].data()};
    std::array<uint8_t, slot_size> zero_pad{};

    sizes[1] = static_cast<uint32_t>(slot_size + 1);
    assert(!FarLib::cache::encode_direct_shards(
        data, sizes, parity, slot_size, zero_pad.data()));
    assert(parity_storage == parity_before);
    sizes[1] = slot_size;

    const void *missing_data[kDataShards] = {nullptr, data[1], data[2], data[3]};
    assert(!FarLib::cache::encode_direct_shards(
        missing_data, sizes, parity, slot_size, zero_pad.data()));
    assert(parity_storage == parity_before);

    sizes[0] = 1;
    assert(!FarLib::cache::encode_direct_shards(
        data, sizes, parity, slot_size, nullptr));
    assert(parity_storage == parity_before);
    sizes[0] = slot_size;

    assert(!FarLib::cache::encode_direct_shards(
        data, sizes, parity, 0, zero_pad.data()));
    assert(parity_storage == parity_before);

    void *missing_parity[kParityShards] = {parity_storage[0].data(), nullptr};
    assert(!FarLib::cache::encode_direct_shards(
        data, sizes, missing_parity, slot_size, zero_pad.data()));
    assert(parity_storage == parity_before);
}

void test_empty_group_clears_parity() {
    constexpr size_t slot_size = 4072;
    std::array<std::array<uint8_t, slot_size>, kParityShards> parity_storage{};
    for (size_t parity = 0; parity < kParityShards; ++parity) {
        parity_storage[parity].fill(0xa7u);
    }
    const std::array<uint32_t, kDataShards> sizes = {0, 0, 0, 0};
    const void *data[kDataShards] = {nullptr, nullptr, nullptr, nullptr};
    void *parity[kParityShards] = {parity_storage[0].data(),
                                   parity_storage[1].data()};
    assert(FarLib::cache::encode_direct_shards(
        data, sizes.data(), parity, slot_size, nullptr));
    for (const auto &buffer : parity_storage) {
        for (uint8_t byte : buffer) assert(byte == 0);
    }
}

#if defined(__linux__)
class GuardedPage {
public:
    GuardedPage() = default;
    GuardedPage(const GuardedPage &) = delete;
    GuardedPage &operator=(const GuardedPage &) = delete;

    ~GuardedPage() {
        if (mapping_ != nullptr) {
            ::munmap(mapping_, 2 * page_size_);
        }
    }

    bool allocate(size_t length) {
        page_size_ = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        if (page_size_ == 0 || length > page_size_) return false;
        mapping_ = static_cast<uint8_t *>(::mmap(
            nullptr, 2 * page_size_, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            return false;
        }
        if (::mprotect(mapping_ + page_size_, page_size_, PROT_NONE) != 0) {
            ::munmap(mapping_, 2 * page_size_);
            mapping_ = nullptr;
            return false;
        }
        pointer_ = mapping_ + page_size_ - length;
        return true;
    }

    uint8_t *data() { return pointer_; }
    const uint8_t *data() const { return pointer_; }

private:
    size_t page_size_ = 0;
    uint8_t *mapping_ = nullptr;
    uint8_t *pointer_ = nullptr;
};

void test_guard_pages_reject_overread() {
    constexpr size_t slot_size = kMaxSlotSize;
    const std::array<uint32_t, kDataShards> sizes = {4096, 4072, 2048, 0};
    std::array<GuardedPage, kDataShards> data_pages;
    std::array<GuardedPage, kParityShards> parity_pages;
    GuardedPage zero_page;
    for (size_t shard = 0; shard < kDataShards - 1; ++shard) {
        assert(data_pages[shard].allocate(sizes[shard]));
        for (size_t byte = 0; byte < sizes[shard]; ++byte) {
            data_pages[shard].data()[byte] = object_byte(shard, byte);
        }
    }
    assert(zero_page.allocate(slot_size));
    std::memset(zero_page.data(), 0, slot_size);
    for (size_t parity = 0; parity < kParityShards; ++parity) {
        assert(parity_pages[parity].allocate(slot_size));
        std::memset(parity_pages[parity].data(), 0xcd, slot_size);
    }

    const void *data[kDataShards] = {data_pages[0].data(),
                                     data_pages[1].data(),
                                     data_pages[2].data(), nullptr};
    void *parity[kParityShards] = {parity_pages[0].data(),
                                   parity_pages[1].data()};
    assert(FarLib::cache::encode_direct_shards(
        data, sizes.data(), parity, slot_size, zero_page.data()));

    std::array<std::vector<uint8_t>, kDataShards> snapshots;
    for (size_t shard = 0; shard < kDataShards; ++shard) {
        snapshots[shard].resize(sizes[shard]);
        if (sizes[shard] != 0) {
            std::memcpy(snapshots[shard].data(), data_pages[shard].data(),
                        sizes[shard]);
        }
    }
    std::array<std::vector<uint8_t>, kParityShards> expected;
    // The oracle accepts vectors for holes as well; their zero-sized vectors
    // are never indexed because the corresponding size is zero.
    make_oracle(snapshots, sizes, slot_size, 0, expected);
    for (size_t parity_idx = 0; parity_idx < kParityShards; ++parity_idx) {
        assert(std::memcmp(parity_pages[parity_idx].data(),
                           expected[parity_idx].data(), slot_size) == 0);
    }
}
#endif

}  // namespace

int main() {
    test_variable_lengths_and_padding_snapshot(
        {4096, 4096, 4096, 4096}, false);
    test_variable_lengths_and_padding_snapshot(
        {4072, 4072, 4072, 4072}, false);
    test_variable_lengths_and_padding_snapshot(
        {4096, 3073, 129, 0}, true);
    test_variable_lengths_and_padding_snapshot(
        {1, 0, 0, 0}, true);
    test_empty_group_clears_parity();
    test_invalid_arguments_do_not_touch_parity();
#if defined(__linux__)
    test_guard_pages_reject_overread();
#endif
    std::cout << "EC_DIRECT_CODEC_PASS\n";
    return 0;
}
