#include "rdma/pending_item.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>

namespace {

struct PendingItem {
    static inline std::size_t default_constructions = 0;
    static constexpr std::size_t kPayloadBytes = 8 * 8192;

    uint16_t preferred_shard_idx = 0;
    void *msg_ptr = nullptr;
    uint32_t msg_lkey = 0;
    void *batch_owner = nullptr;
    uint64_t batch_owner_generation = 0;
    uint64_t sequence = 0;
    std::array<uint8_t, kPayloadBytes> payload{};

    PendingItem() { ++default_constructions; }
    PendingItem(const PendingItem &) = default;
    PendingItem &operator=(const PendingItem &) = default;
};

void reset_counter() { PendingItem::default_constructions = 0; }

void fill_item(PendingItem &item, uint64_t sequence, uint8_t pattern,
               void *msg_ptr, void *batch_owner, uint64_t generation) {
    item.preferred_shard_idx = static_cast<uint16_t>(sequence + 7);
    item.msg_ptr = msg_ptr;
    item.msg_lkey = static_cast<uint32_t>(0x1000 + sequence);
    item.batch_owner = batch_owner;
    item.batch_owner_generation = generation;
    item.sequence = sequence;
    item.payload.fill(pattern);
}

void assert_reset(const PendingItem &item) {
    assert(item.preferred_shard_idx == 0);
    assert(item.msg_ptr == nullptr);
    assert(item.msg_lkey == 0);
    assert(item.batch_owner == nullptr);
    assert(item.batch_owner_generation == 0);
    assert(item.sequence == 0);
    for (uint8_t byte : item.payload) {
        assert(byte == 0);
    }
}

void test_empty_does_not_construct() {
    std::array<PendingItem, 2> slots{};
    reset_counter();
    std::size_t head = 0;
    std::size_t count = 0;
    auto taken = FarLib::rdma::carbink::take_pending_item_locked(
        slots.data(), slots.size(), head, count);
    assert(!taken.has_value());
    assert(PendingItem::default_constructions == 0);
    assert(head == 0);
    assert(count == 0);
}

void test_fifo_wrap_payload_and_reset() {
    std::array<PendingItem, 4> slots{};
    int msg_a = 0;
    int owner_a = 0;
    int msg_b = 0;
    int owner_b = 0;
    fill_item(slots[3], 10, 0xA5, &msg_a, &owner_a, 1010);
    fill_item(slots[0], 11, 0x5A, &msg_b, &owner_b, 1011);

    reset_counter();
    std::size_t head = 3;
    std::size_t count = 2;
    auto first = FarLib::rdma::carbink::take_pending_item_locked(
        slots.data(), slots.size(), head, count);
    assert(first.has_value());
    assert(first->sequence == 10);
    assert(first->preferred_shard_idx == 17);
    assert(first->msg_ptr == &msg_a);
    assert(first->msg_lkey == 0x100A);
    assert(first->batch_owner == &owner_a);
    assert(first->batch_owner_generation == 1010);
    assert(first->payload.front() == 0xA5);
    assert(first->payload.back() == 0xA5);
    assert_reset(slots[3]);
    assert(head == 0);
    assert(count == 1);

    auto second = FarLib::rdma::carbink::take_pending_item_locked(
        slots.data(), slots.size(), head, count);
    assert(second.has_value());
    assert(second->sequence == 11);
    assert(second->msg_ptr == &msg_b);
    assert(second->batch_owner == &owner_b);
    assert(second->batch_owner_generation == 1011);
    assert(second->payload.front() == 0x5A);
    assert(second->payload.back() == 0x5A);
    assert_reset(slots[0]);
    assert(head == 1);
    assert(count == 0);
}

}  // namespace

int main() {
    test_empty_does_not_construct();
    test_fifo_wrap_payload_and_reset();
    std::cout << "CARBINK_SERVER_PENDING_ITEM_PASS\n";
    return 0;
}
