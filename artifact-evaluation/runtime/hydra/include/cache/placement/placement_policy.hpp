#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace FarLib::cache::design1 {

// The placement policy only ranks behavior groups. Region allocation and
// reclassification are separate runtime concerns.
enum class PlacementPolicy : uint8_t {
    Hotness,
    Design1Gain,
};

inline constexpr std::string_view to_string(PlacementPolicy policy) {
    switch (policy) {
    case PlacementPolicy::Hotness:
        return "hotness";
    case PlacementPolicy::Design1Gain:
        return "design1_gain";
    }
    return "unknown";
}

inline constexpr std::optional<PlacementPolicy>
parse_placement_policy(std::string_view value) {
    if (value == "hotness") {
        return PlacementPolicy::Hotness;
    }
    if (value == "design1_gain") {
        return PlacementPolicy::Design1Gain;
    }
    return std::nullopt;
}

// All byte fields below represent sampled or accumulated object footprint.
// A score is "byte-work": bytes multiplied by expected access events.
struct GroupPlacementInput {
    uint64_t ema_read_references{0};
    uint64_t ema_write_references{0};
    uint64_t write_weight{0};
    uint64_t live_bytes{0};
    uint64_t live_objects{0};
    uint64_t total_fetch_bytes{0};
    uint64_t total_clean_evict_bytes{0};
    uint64_t total_dirty_evict_bytes{0};
    uint64_t ema_clean_evict_bytes{0};
    uint64_t ema_dirty_evict_bytes{0};
};

struct GroupPlacementScore {
    uint64_t weighted_references{0};
    uint64_t resident_gain_byte_work{0};
    uint64_t streaming_gain_byte_work{0};
    long double resident_advantage_per_live_byte{0.0L};
};

inline uint64_t saturating_u64(__uint128_t value) {
    return value > std::numeric_limits<uint64_t>::max()
               ? std::numeric_limits<uint64_t>::max()
               : static_cast<uint64_t>(value);
}

inline GroupPlacementScore score_group(const GroupPlacementInput &input) {
    GroupPlacementScore score;
    score.weighted_references =
        saturating_u64(static_cast<__uint128_t>(input.ema_write_references) *
                           input.write_weight +
                       input.ema_read_references);

    const uint64_t total_evict_bytes =
        input.total_clean_evict_bytes + input.total_dirty_evict_bytes;
    const long double refetch_probability =
        total_evict_bytes == 0
            ? 0.0L
            : static_cast<long double>(
                  std::min(input.total_fetch_bytes, total_evict_bytes)) /
                  total_evict_bytes;
    const long double clean_probability =
        total_evict_bytes == 0
            ? 0.0L
            : static_cast<long double>(input.total_clean_evict_bytes) /
                  total_evict_bytes;
    const uint64_t ema_evict_bytes =
        input.ema_clean_evict_bytes + input.ema_dirty_evict_bytes;
    const long double streaming_gain =
        static_cast<long double>(ema_evict_bytes) * refetch_probability *
        clean_probability;
    score.streaming_gain_byte_work =
        streaming_gain >=
                static_cast<long double>(std::numeric_limits<uint64_t>::max())
            ? std::numeric_limits<uint64_t>::max()
            : static_cast<uint64_t>(streaming_gain);

    // Resident gain estimates counterfactual avoided fetch work. A Resident
    // group has few observed misses by construction, so sampled demand
    // references are multiplied by its mean object footprint instead.
    const uint64_t average_footprint =
        input.live_objects == 0 ? 0 : input.live_bytes / input.live_objects;
    score.resident_gain_byte_work =
        saturating_u64(static_cast<__uint128_t>(score.weighted_references) *
                       average_footprint);
    if (input.live_bytes != 0) {
        score.resident_advantage_per_live_byte =
            (static_cast<long double>(score.resident_gain_byte_work) -
             static_cast<long double>(score.streaming_gain_byte_work)) /
            input.live_bytes;
    }
    return score;
}

struct RankedGroup {
    uint32_t group_id{0};
    uint64_t live_bytes{0};
    bool currently_resident_preferred{false};
    GroupPlacementScore score;
};

inline bool ranks_before(PlacementPolicy policy, const RankedGroup &left,
                         const RankedGroup &right) {
    if (policy == PlacementPolicy::Design1Gain &&
        left.score.resident_advantage_per_live_byte !=
            right.score.resident_advantage_per_live_byte) {
        return left.score.resident_advantage_per_live_byte >
               right.score.resident_advantage_per_live_byte;
    }

    const __uint128_t left_hotness =
        static_cast<__uint128_t>(left.score.weighted_references) *
        right.live_bytes;
    const __uint128_t right_hotness =
        static_cast<__uint128_t>(right.score.weighted_references) *
        left.live_bytes;
    if (left_hotness != right_hotness) {
        return left_hotness > right_hotness;
    }
    if (left.currently_resident_preferred !=
        right.currently_resident_preferred) {
        return left.currently_resident_preferred;
    }
    return left.group_id < right.group_id;
}

}  // namespace FarLib::cache::design1
