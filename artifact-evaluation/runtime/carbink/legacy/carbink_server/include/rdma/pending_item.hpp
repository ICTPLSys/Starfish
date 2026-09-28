#pragma once

#include <cstddef>
#include <optional>
#include <exception>

namespace FarLib::rdma::carbink {

// The caller owns the pending-queue mutex.
template <typename Item>
std::optional<Item> take_pending_item_locked(Item *items, std::size_t capacity,
                                             std::size_t &head,
                                             std::size_t &count) {
    if (count == 0) return std::nullopt;
    // A corrupt/nonexistent nonempty queue must not look like an empty one.
    if (items == nullptr || capacity == 0 || head >= capacity || count > capacity)
        std::terminate();
    std::optional<Item> item(std::in_place, items[head]);
    items[head] = Item{};
    head = (head + 1) % capacity;
    --count;
    return item;
}

}  // namespace FarLib::rdma::carbink
