#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <type_traits>
#include "cache/entry.hpp"
#include "hydra/page_layout.hpp"

namespace FarLib::hydra {
struct SlotHeat {
    std::atomic<cache::FarObjectEntry *> handle{nullptr};
    std::atomic<uint32_t> references{0};
};
// Descriptors have stable native addresses; managed page data may move/evict.
// Keep owner first so a subobject's owner pointer identifies its descriptor.
struct Page {
    cache::FarObjectEntry owner;
    cache::HydraPagePins pins;
    std::atomic<uint32_t> references{1}; // one allocation lease plus live handles
    std::atomic<bool> allocation_lease{true};
    uint16_t slot_bytes = 0;
    uint16_t next_slot = 0; // protected by the lane mutex
    uint8_t packing_class = 0; // immutable: cold/default=0, hot=1
    std::unique_ptr<SlotHeat[]> heat; // allocated only with runtime packing ON
};
static_assert(std::is_standard_layout_v<Page>);
static_assert(offsetof(Page, owner) == 0);
static_assert(offsetof(Page, pins) == sizeof(cache::FarObjectEntry));

struct Lane {
    std::mutex mutex;
    std::array<std::array<Page *, 2>, kSizeClasses> active{};
};
}  // namespace FarLib::hydra
