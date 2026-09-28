#pragma once

#include <cstddef>
#include <cstdint>

namespace FarLib::hydra {
inline constexpr std::size_t kPageBytes = 8192;
inline constexpr std::size_t kDataShards = 4;
inline constexpr std::size_t kParityShards = 2;
inline constexpr std::size_t kShardBytes = kPageBytes / kDataShards;
inline constexpr std::size_t kMinSlotBytes = 8;
inline constexpr std::size_t kSizeClasses = 11;
inline constexpr std::size_t kAllocationLanes = 64;

constexpr std::size_t slot_bytes(std::size_t requested) {
    if (requested == 0 || requested > kPageBytes) return 0;
    std::size_t slot = kMinSlotBytes;
    while (slot < requested) slot *= 2;
    return slot;
}
constexpr std::size_t size_class(std::size_t requested) {
    const auto slot = slot_bytes(requested);
    if (slot == 0) return kSizeClasses;
    std::size_t index = 0;
    for (auto size = kMinSlotBytes; size < slot; size *= 2) ++index;
    return index;
}
constexpr std::size_t capacity(std::size_t requested) {
    const auto slot = slot_bytes(requested);
    return slot == 0 ? 0 : kPageBytes / slot;
}
static_assert(kShardBytes == 2048);
static_assert(capacity(512) == 16 && capacity(8192) == 1);
static_assert(size_class(8192) + 1 == kSizeClasses);
}  // namespace FarLib::hydra
