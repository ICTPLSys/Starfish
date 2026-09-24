#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include "cache/entry.hpp"
#include "hydra/page_layout.hpp"

namespace FarLib::hydra {
// Descriptors have stable native addresses; managed page data may move/evict.
// Keep owner first so a subobject's owner pointer identifies its descriptor.
struct Page {
    cache::FarObjectEntry owner;
    cache::HydraPagePins pins;
    std::atomic<uint32_t> references{1}; // one allocation lease plus live handles
    std::atomic<bool> allocation_lease{true};
    uint16_t slot_bytes = 0;
    uint16_t next_slot = 0; // protected by the lane mutex
};
static_assert(std::is_standard_layout_v<Page>);
static_assert(offsetof(Page, owner) == 0);
static_assert(offsetof(Page, pins) == sizeof(cache::FarObjectEntry));

struct Lane {
    std::mutex mutex;
    std::array<Page *, kSizeClasses> active{};
};
}  // namespace FarLib::hydra
