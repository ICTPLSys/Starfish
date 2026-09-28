#include "cache/ec_split_codec.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using FarLib::cache::EC_N;

constexpr std::size_t kSpanBytes = FarLib::cache::FARLIB_SPAN_SIZE;
constexpr std::size_t kGuardBytes = 64;

struct GuardedBuffer {
    std::array<uint8_t, kSpanBytes + 2 * kGuardBytes> bytes{};

    uint8_t *data() { return bytes.data() + kGuardBytes; }
    const uint8_t *data() const { return bytes.data() + kGuardBytes; }

    void fill_all(uint8_t value) { bytes.fill(value); }

    void fill_payload(uint8_t value) {
        std::fill(data(), data() + kSpanBytes, value);
    }

    bool guards_equal(uint8_t value) const {
        for (std::size_t i = 0; i < kGuardBytes; ++i) {
            if (bytes[i] != value ||
                bytes[kGuardBytes + kSpanBytes + i] != value) {
                return false;
            }
        }
        return true;
    }
};

void assert_bytes_equal(const uint8_t *lhs, const uint8_t *rhs,
                        std::size_t bytes) {
    for (std::size_t i = 0; i < bytes; ++i) {
        assert(lhs[i] == rhs[i]);
    }
}

void expected_single_slot_delta(uint8_t slot, const uint8_t *delta,
                                std::array<uint8_t, kSpanBytes> &parity0,
                                std::array<uint8_t, kSpanBytes> &parity1) {
    std::array<std::array<uint8_t, kSpanBytes>, EC_N> data{};
    std::copy(delta, delta + kSpanBytes, data[slot].begin());
    const uint8_t *data_ptrs[EC_N] = {};
    for (int i = 0; i < EC_N; ++i) {
        data_ptrs[i] = data[i].data();
    }
    parity0.fill(0);
    parity1.fill(0);
    FarLib::cache::ec_span_full_stripe_encode(
        static_cast<uint8_t>(EC_N), data_ptrs, parity0.data(), parity1.data());
}

void test_init_from_zero_all_slots() {
    for (uint8_t slot = 0; slot < EC_N; ++slot) {
        GuardedBuffer src;
        GuardedBuffer stale_dst;
        GuardedBuffer parity0;
        GuardedBuffer parity1;
        src.fill_all(0xE1);
        src.fill_payload(static_cast<uint8_t>(0x21 + slot));
        stale_dst.fill_all(0xD2);
        stale_dst.fill_payload(static_cast<uint8_t>(0x91 + slot));
        parity0.fill_all(0xC3);
        parity1.fill_all(0xB4);

        std::array<uint8_t, kSpanBytes> src_before{};
        std::copy(src.data(), src.data() + kSpanBytes, src_before.begin());
        std::array<uint8_t, kSpanBytes> expected0{};
        std::array<uint8_t, kSpanBytes> expected1{};
        expected_single_slot_delta(slot, src.data(), expected0, expected1);

        FarLib::cache::ec_span_delta_encode_update(
            slot, stale_dst.data(), src.data(), parity0.data(), parity1.data(),
            true);

        assert_bytes_equal(stale_dst.data(), src.data(), kSpanBytes);
        assert_bytes_equal(src.data(), src_before.data(), kSpanBytes);
        assert_bytes_equal(parity0.data(), expected0.data(), kSpanBytes);
        assert_bytes_equal(parity1.data(), expected1.data(), kSpanBytes);
        assert(src.guards_equal(0xE1));
        assert(stale_dst.guards_equal(0xD2));
        assert(parity0.guards_equal(0xC3));
        assert(parity1.guards_equal(0xB4));
    }
}

void test_delta_update_false_all_slots() {
    for (uint8_t slot = 0; slot < EC_N; ++slot) {
        GuardedBuffer old_data;
        GuardedBuffer new_data;
        GuardedBuffer parity0;
        GuardedBuffer parity1;
        old_data.fill_all(0xA1);
        old_data.fill_payload(static_cast<uint8_t>(0x31 + slot));
        new_data.fill_all(0xB2);
        new_data.fill_payload(static_cast<uint8_t>(0x71 + slot));
        parity0.fill_all(0xC4);
        parity1.fill_all(0xD5);

        std::array<uint8_t, kSpanBytes> old_before{};
        std::array<uint8_t, kSpanBytes> new_before{};
        std::array<uint8_t, kSpanBytes> delta{};
        for (std::size_t i = 0; i < kSpanBytes; ++i) {
            old_before[i] = old_data.data()[i];
            new_before[i] = new_data.data()[i];
            delta[i] = old_before[i] ^ new_before[i];
        }
        std::array<uint8_t, kSpanBytes> expected0{};
        std::array<uint8_t, kSpanBytes> expected1{};
        expected_single_slot_delta(slot, delta.data(), expected0, expected1);

        FarLib::cache::ec_span_delta_encode_update(
            slot, old_data.data(), new_data.data(), parity0.data(), parity1.data(),
            false);

        assert_bytes_equal(old_data.data(), new_before.data(), kSpanBytes);
        assert_bytes_equal(new_data.data(), new_before.data(), kSpanBytes);
        assert_bytes_equal(parity0.data(), expected0.data(), kSpanBytes);
        assert_bytes_equal(parity1.data(), expected1.data(), kSpanBytes);
        assert(old_data.guards_equal(0xA1));
        assert(new_data.guards_equal(0xB2));
        assert(parity0.guards_equal(0xC4));
        assert(parity1.guards_equal(0xD5));
    }
}

}  // namespace

int main() {
    test_init_from_zero_all_slots();
    test_delta_update_false_all_slots();
    std::cout << "CARBINK_SERVER_COMPACT_CODEC_PASS\n";
    return 0;
}
