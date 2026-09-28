#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace FarLib::cache::carbink {

// Metadata identity for one endpoint's linked WRITE batch fence.
//
// WRID layout (after the fixed 0x86 high byte):
//   bits [0, 3]   slot (16 slots per owner/endpoint lane)
//   bits [4, 15]  lane (owner * endpoint_count + endpoint, <= 4096)
//   bits [16,55]  generation (40 bits, lane-local and never wrapped)
class WriteFenceTable {
public:
    static constexpr size_t kSlotsPerLane = 16;
    static constexpr size_t kMaxIdsPerFence = 64;

    struct Entry {
        uint8_t count = 0;
        uint8_t reserved[7]{};
        uint64_t ids[kMaxIdsPerFence]{};
        size_t owner = 0;
        size_t endpoint = 0;
        uint8_t slot = 0;
        uint8_t reserved2[7]{};
        uint64_t generation = 0;
    };

private:
    static constexpr uint64_t kFenceTag = 0x8600000000000000ULL;
    static constexpr uint64_t kFenceTagMask = 0xff00000000000000ULL;
    static constexpr unsigned kSlotBits = 4;
    static constexpr unsigned kLaneBits = 12;
    static constexpr unsigned kGenerationShift = kSlotBits + kLaneBits;
    static constexpr unsigned kPackedStateBits = 3;
    static constexpr uint64_t kSlotMask = (uint64_t{1} << kSlotBits) - 1;
    static constexpr uint64_t kLaneMask = (uint64_t{1} << kLaneBits) - 1;
    static constexpr uint64_t kGenerationMask =
        (uint64_t{1} << (56 - kGenerationShift)) - 1;
    static constexpr uint64_t kMaxLaneCount = uint64_t{1} << kLaneBits;
    static constexpr uint64_t kInitialGeneration = 1;

    enum class State : uint8_t {
        kFree = 0,
        kFilling = 1,
        kPosted = 2,
        kConsumed = 3,
        kRetired = 4,
    };

    static constexpr uint64_t pack(uint64_t generation, State state) {
        return (generation << kPackedStateBits) |
               static_cast<uint64_t>(state);
    }

    static constexpr State packed_state(uint64_t packed) {
        return static_cast<State>(packed & ((uint64_t{1} << kPackedStateBits) - 1));
    }

    static constexpr uint64_t packed_generation(uint64_t packed) {
        return packed >> kPackedStateBits;
    }

    struct Slot {
        std::atomic<uint64_t> packed{pack(kInitialGeneration, State::kFree)};
        Entry entry{};
    };

    struct Lane {
        std::array<Slot, kSlotsPerLane> slots{};
        std::atomic<size_t> in_use{0};
    };

    static uint64_t make_wrid(uint64_t generation, size_t lane,
                              size_t slot) {
        return kFenceTag | (generation << kGenerationShift) |
               (static_cast<uint64_t>(lane) << kSlotBits) |
               static_cast<uint64_t>(slot);
    }

    static bool decode_wrid(uint64_t wrid, uint64_t *generation,
                            size_t *lane, size_t *slot) {
        if (generation == nullptr || lane == nullptr || slot == nullptr ||
            !is_fence(wrid)) {
            return false;
        }
        const uint64_t low = wrid & ((uint64_t{1} << 56) - 1);
        const uint64_t decoded_generation = low >> kGenerationShift;
        const size_t decoded_lane =
            static_cast<size_t>((low >> kSlotBits) & kLaneMask);
        const size_t decoded_slot = static_cast<size_t>(low & kSlotMask);
        if (decoded_generation == 0 || decoded_lane >= kMaxLaneCount) {
            return false;
        }
        *generation = decoded_generation;
        *lane = decoded_lane;
        *slot = decoded_slot;
        return true;
    }

    Slot *slot_for_wrid(uint64_t wrid, uint64_t *generation,
                        size_t *lane_index) const {
        size_t lane = 0;
        size_t slot_index = 0;
        if (!decode_wrid(wrid, generation, &lane, &slot_index) ||
            lane >= lane_count_) {
            return nullptr;
        }
        if (lane_index != nullptr) *lane_index = lane;
        return &lanes_[lane].slots[slot_index];
    }

public:
    WriteFenceTable(size_t owner_count, size_t endpoint_count)
        : owner_count_(owner_count), endpoint_count_(endpoint_count) {
        if (owner_count_ == 0 || endpoint_count_ == 0 ||
            owner_count_ >
                std::numeric_limits<size_t>::max() / endpoint_count_) {
            owner_count_ = 0;
            endpoint_count_ = 0;
            lane_count_ = 0;
            return;
        }
        lane_count_ = owner_count_ * endpoint_count_;
        if (lane_count_ > kMaxLaneCount) {
            owner_count_ = 0;
            endpoint_count_ = 0;
            lane_count_ = 0;
            return;
        }
        lanes_ = std::make_unique<Lane[]>(lane_count_);
    }

    WriteFenceTable(const WriteFenceTable &) = delete;
    WriteFenceTable &operator=(const WriteFenceTable &) = delete;

    static bool is_fence(uint64_t wrid) {
        return (wrid & kFenceTagMask) == kFenceTag;
    }

    size_t owner_count() const { return owner_count_; }
    size_t endpoint_count() const { return endpoint_count_; }

    // One producer owns each lane. Completion consumers may call consume/release
    // concurrently for any returned fence.
    uint64_t acquire(size_t owner, size_t endpoint,
                     std::span<const uint64_t> ids) {
        if (lanes_ == nullptr || owner >= owner_count_ ||
            endpoint >= endpoint_count_ || ids.empty() ||
            ids.size() > kMaxIdsPerFence) {
            return 0;
        }
        for (uint64_t id : ids) {
            if (id == 0) return 0;
        }

        const size_t lane_index = owner * endpoint_count_ + endpoint;
        Lane &lane = lanes_[lane_index];
        for (size_t slot_index = 0; slot_index < kSlotsPerLane;
             ++slot_index) {
            Slot &slot = lane.slots[slot_index];
            uint64_t expected = slot.packed.load(std::memory_order_acquire);
            if (packed_state(expected) != State::kFree) continue;
            const uint64_t generation = packed_generation(expected);
            if (generation == 0 || generation > kGenerationMask) continue;
            if (!slot.packed.compare_exchange_strong(
                    expected, pack(generation, State::kFilling),
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                continue;
            }

            slot.entry.count = static_cast<uint8_t>(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                slot.entry.ids[i] = ids[i];
            }
            for (size_t i = ids.size(); i < kMaxIdsPerFence; ++i) {
                slot.entry.ids[i] = 0;
            }
            slot.entry.owner = owner;
            slot.entry.endpoint = endpoint;
            slot.entry.slot = static_cast<uint8_t>(slot_index);
            slot.entry.generation = generation;
            lane.in_use.fetch_add(1, std::memory_order_acq_rel);
            slot.packed.store(pack(generation, State::kPosted),
                              std::memory_order_release);
            return make_wrid(generation, lane_index, slot_index);
        }
        return 0;
    }

    // Claims a completed fence exactly once. Entry remains immutable until
    // release(wrid) succeeds.
    const Entry *consume(uint64_t wrid) const {
        uint64_t generation = 0;
        size_t lane_index = 0;
        Slot *slot = slot_for_wrid(wrid, &generation, &lane_index);
        if (slot == nullptr) return nullptr;
        uint64_t expected = pack(generation, State::kPosted);
        if (!slot->packed.compare_exchange_strong(
                expected, pack(generation, State::kConsumed),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return nullptr;
        }
        return &slot->entry;
    }

    // Only a consumed/claimed fence can be released. Generation increments in
    // the same packed CAS; stale consumers cannot claim a reused slot.
    bool release(uint64_t wrid) {
        uint64_t generation = 0;
        size_t lane_index = 0;
        Slot *slot = slot_for_wrid(wrid, &generation, &lane_index);
        if (slot == nullptr) return false;

        const bool retire = generation == kGenerationMask;
        const uint64_t next_generation = generation + (retire ? 0 : 1);
        const State next_state = retire ? State::kRetired : State::kFree;
        uint64_t expected = pack(generation, State::kConsumed);
        if (!slot->packed.compare_exchange_strong(
                expected, pack(next_generation, next_state),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return false;
        }
        lanes_[lane_index].in_use.fetch_sub(1, std::memory_order_acq_rel);
        return true;
    }

    // Diagnostic count; sums lane-local counters and avoids a global hot-path
    // atomic.
    size_t in_use() const {
        size_t total = 0;
        for (size_t i = 0; i < lane_count_; ++i) {
            total += lanes_[i].in_use.load(std::memory_order_acquire);
        }
        return total;
    }

private:
    size_t owner_count_ = 0;
    size_t endpoint_count_ = 0;
    size_t lane_count_ = 0;
    std::unique_ptr<Lane[]> lanes_;
};

}  // namespace FarLib::cache::carbink

namespace FarLib::cache {
using CarbinkWriteFenceTable = carbink::WriteFenceTable;
using WriteFenceTable = carbink::WriteFenceTable;
}  // namespace FarLib::cache
