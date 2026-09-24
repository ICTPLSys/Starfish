#pragma once

// Observation-only behavior-group refinement.
//
// This header deliberately has no dependency on the cache allocator, object
// entries, RDMA, or any Rep/EC implementation.  A caller supplies a stable
// (address, generation) region token and a complete-window profile; the
// registry only changes the region-to-group metadata recorded here.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__SIZEOF_INT128__)
using starfish_refinement_wide_uint = unsigned __int128;

// A small fixed-width unsigned integer for exact products of three uint64
// values.  The persistent range policy needs up to 192 bits when a ratio is
// applied; keeping this local avoids a heavyweight dependency on platforms
// that already provide native __int128.
struct starfish_refinement_persistent_big_uint {
    std::array<std::uint64_t, 4> limbs{};

    constexpr starfish_refinement_persistent_big_uint() = default;
    constexpr explicit starfish_refinement_persistent_big_uint(
        std::uint64_t value) {
        limbs[0] = value;
    }

    friend bool operator==(const starfish_refinement_persistent_big_uint &lhs,
                           const starfish_refinement_persistent_big_uint &rhs) {
        return lhs.limbs == rhs.limbs;
    }
    friend bool operator!=(const starfish_refinement_persistent_big_uint &lhs,
                           const starfish_refinement_persistent_big_uint &rhs) {
        return !(lhs == rhs);
    }
    friend bool operator<(const starfish_refinement_persistent_big_uint &lhs,
                          const starfish_refinement_persistent_big_uint &rhs) {
        for (std::size_t index = lhs.limbs.size(); index-- > 0;) {
            if (lhs.limbs[index] != rhs.limbs[index])
                return lhs.limbs[index] < rhs.limbs[index];
        }
        return false;
    }
    friend bool operator>(const starfish_refinement_persistent_big_uint &lhs,
                          const starfish_refinement_persistent_big_uint &rhs) {
        return rhs < lhs;
    }
    friend bool operator<=(const starfish_refinement_persistent_big_uint &lhs,
                           const starfish_refinement_persistent_big_uint &rhs) {
        return !(rhs < lhs);
    }
    friend bool operator>=(const starfish_refinement_persistent_big_uint &lhs,
                           const starfish_refinement_persistent_big_uint &rhs) {
        return !(lhs < rhs);
    }

    friend starfish_refinement_persistent_big_uint operator*(
        starfish_refinement_persistent_big_uint lhs, std::uint64_t rhs) {
        starfish_refinement_persistent_big_uint result;
        unsigned __int128 carry = 0;
        for (std::size_t index = 0; index < lhs.limbs.size(); ++index) {
            const unsigned __int128 product =
                static_cast<unsigned __int128>(lhs.limbs[index]) * rhs + carry;
            result.limbs[index] = static_cast<std::uint64_t>(product);
            carry = product >> 64;
        }
        return result;
    }
    friend starfish_refinement_persistent_big_uint operator*(
        std::uint64_t lhs, starfish_refinement_persistent_big_uint rhs) {
        return rhs * lhs;
    }
    friend starfish_refinement_persistent_big_uint operator-(
        const starfish_refinement_persistent_big_uint &lhs,
        const starfish_refinement_persistent_big_uint &rhs) {
        starfish_refinement_persistent_big_uint result;
        bool borrow = false;
        for (std::size_t index = 0; index < lhs.limbs.size(); ++index) {
            const unsigned __int128 right =
                static_cast<unsigned __int128>(rhs.limbs[index]) + borrow;
            const unsigned __int128 left = lhs.limbs[index];
            result.limbs[index] = static_cast<std::uint64_t>(left - right);
            borrow = left < right;
        }
        return result;
    }
};
using starfish_refinement_persistent_wide_uint =
    starfish_refinement_persistent_big_uint;
#else
#include <boost/multiprecision/cpp_int.hpp>
using starfish_refinement_wide_uint = boost::multiprecision::uint128_t;
using starfish_refinement_persistent_wide_uint =
    boost::multiprecision::uint256_t;
#endif

namespace FarLib {
namespace cache {
namespace refinement {

constexpr std::uint64_t kPartsPerMillion = 1'000'000;
// Lifetime predicates are intentionally conservative even though v2 no longer
// requires lifetime evidence in the common eligibility gate.
constexpr std::uint64_t kMinLifetimeSamplesForPredicate = 2;
using GroupId = std::uint64_t;
using RootGroupId = GroupId;

enum class IntrinsicSignature : std::uint8_t {
    Unclassified = 0,
    Transient = 1,
    FrequentLifecycle = 2,
    StableCold = 3,
    Default = 4,
};

inline const char *to_string(IntrinsicSignature signature) {
    switch (signature) {
    case IntrinsicSignature::Unclassified:
        return "Unclassified";
    case IntrinsicSignature::Transient:
        return "Transient";
    case IntrinsicSignature::FrequentLifecycle:
        return "FrequentLifecycle";
    case IntrinsicSignature::StableCold:
        return "StableCold";
    case IntrinsicSignature::Default:
        return "Default";
    }
    return "Unknown";
}

struct IntrinsicThresholds {
    // These values are intentionally injected by the experiment preset.  The
    // semantic freeze does not invent numeric theta values.
    std::uint64_t theta_dirty_per_million = 0;
    std::uint64_t theta_lifecycle_rate_per_million = 0;
    std::uint64_t theta_activity_rate_per_million = 0;
    std::uint64_t theta_short_lifetime_ns = 0;
    std::uint64_t theta_long_lifetime_ns = 0;

    bool valid() const {
        return theta_dirty_per_million <= kPartsPerMillion &&
               theta_short_lifetime_ns > 0 &&
               theta_long_lifetime_ns > theta_short_lifetime_ns;
    }
};

enum class IntrinsicClassifierVersion : std::uint8_t {
    V1 = 1,
    V2 = 2,
    V3 = 3,
};

struct EvidenceGate {
    // v1/v2 use these as common evidence floors.  v3 keeps the L/E floors as
    // per-predicate thresholds: its common gate is only complete_window,
    // L >= 8, and X > 0.  A is therefore never a v3 common floor.
    std::uint64_t min_lifecycle_events = 8;
    std::uint64_t min_eviction_events = 8;
    std::uint64_t min_demand_references = 64;
    std::uint64_t min_completed_lifetime_samples = 0;
    // Keep aggregate initialization with the historical four fields source
    // compatible; a bare EvidenceGate{} remains the v2 shape, while all
    // classifier/runtime defaults select v3 explicitly below.
    IntrinsicClassifierVersion version = IntrinsicClassifierVersion::V2;

    static constexpr EvidenceGate frozen_v1() {
        EvidenceGate gate;
        gate.min_completed_lifetime_samples = 2;
        gate.version = IntrinsicClassifierVersion::V1;
        return gate;
    }

    static constexpr EvidenceGate frozen_v2() {
        EvidenceGate gate;
        gate.version = IntrinsicClassifierVersion::V2;
        return gate;
    }

    static constexpr EvidenceGate frozen_v3() {
        EvidenceGate gate;
        gate.version = IntrinsicClassifierVersion::V3;
        return gate;
    }
};

struct IntrinsicClassifierConfig {
    IntrinsicThresholds thresholds;
    // Runtime/default construction selects the revised v3 gate.  Callers that
    // need the historical v1/v2 common gates can pass their frozen helper.
    EvidenceGate evidence = EvidenceGate::frozen_v3();

    bool valid() const { return thresholds.valid(); }
};

// Exact, unsampled counters for one region in one profiling window.  The
// caller sets complete_window only for a full 4,096-transition window;
// partial/final windows therefore remain Unclassified.
struct RegionProfile {
    bool complete_window = false;

    std::uint64_t fetch_commit = 0;
    std::uint64_t clean_evict_commit = 0;
    std::uint64_t dirty_evict_commit = 0;
    std::uint64_t free_commit = 0;

    std::uint64_t demand_read_reference = 0;
    std::uint64_t demand_write_reference = 0;
    std::uint64_t live_object_exposure = 0;

    std::uint64_t completed_lifetime_sum_ns = 0;
    std::uint64_t completed_lifetime_samples = 0;

    // Persistent physical regions keep their behavior identity across a free
    // interval.  These endpoint fields are supplied by the runtime when it
    // has an authoritative occupancy census; the legacy classifier ignores
    // them.  They intentionally follow all historical fields so old
    // aggregate initialization remains source-compatible.  `live_object_time_ns`
    // is the common time denominator for the persistent range-policy hotness
    // metric.
    std::uint64_t live_objects_at_end = 0;
    bool occupancy_known = false;
    std::uint64_t live_object_time_ns = 0;

    // Capacity grouping uses an authoritative census of the current logical
    // primary object set.  These fields are appended so all historical
    // aggregate initializers remain source-compatible.  A false census bit
    // means that the numeric fields are not routing evidence.
    bool primary_census_known = false;
    std::uint64_t primary_live_objects = 0;
    std::uint64_t primary_live_bytes = 0;
    std::uint64_t primary_object_time_ns = 0;

    std::uint64_t lifecycle_events() const {
        return saturated_add(
            saturated_add(fetch_commit, clean_evict_commit),
            saturated_add(dirty_evict_commit, free_commit));
    }

    std::uint64_t eviction_events() const {
        return saturated_add(clean_evict_commit, dirty_evict_commit);
    }

    std::uint64_t demand_references() const {
        return saturated_add(demand_read_reference, demand_write_reference);
    }

    std::uint64_t lifecycle_pressure() const {
        return saturated_add(
            saturated_add(fetch_commit, dirty_evict_commit), free_commit);
    }

private:
    static std::uint64_t saturated_add(std::uint64_t lhs,
                                       std::uint64_t rhs) {
        constexpr auto max = std::numeric_limits<std::uint64_t>::max();
        return lhs > max - rhs ? max : lhs + rhs;
    }
};

struct RawPredicateVector {
    bool transient = false;
    bool frequent_lifecycle = false;
    bool stable_cold = false;

    std::array<bool, 3> as_array() const {
        return {transient, frequent_lifecycle, stable_cold};
    }

    bool operator==(const RawPredicateVector &other) const {
        return transient == other.transient &&
               frequent_lifecycle == other.frequent_lifecycle &&
               stable_cold == other.stable_cold;
    }
};

struct IntrinsicClassification {
    IntrinsicSignature signature = IntrinsicSignature::Unclassified;
    RawPredicateVector raw;
    bool evidence_qualified = false;
    bool complete_window = false;

    std::uint64_t lifecycle_events = 0;
    std::uint64_t eviction_events = 0;
    std::uint64_t demand_references = 0;
    std::uint64_t lifecycle_pressure = 0;
    std::uint64_t dirty_fraction_per_million = 0;
    std::uint64_t lifecycle_rate_per_million = 0;
    std::uint64_t activity_rate_per_million = 0;
    std::uint64_t mean_completed_lifetime_ns = 0;

    bool eligible() const { return evidence_qualified; }
};

namespace detail {

using wide_uint = starfish_refinement_wide_uint;

inline std::uint64_t scaled_floor(std::uint64_t numerator,
                                  std::uint64_t denominator) {
    if (denominator == 0) {
        return 0;
    }
    const wide_uint value =
        (static_cast<wide_uint>(numerator) * kPartsPerMillion) /
        static_cast<wide_uint>(denominator);
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    return value > static_cast<wide_uint>(max)
               ? max
               : static_cast<std::uint64_t>(value);
}

inline bool scaled_ratio_at_least(std::uint64_t numerator,
                                  std::uint64_t denominator,
                                  std::uint64_t threshold_per_million) {
    if (denominator == 0) {
        return false;
    }
    return static_cast<wide_uint>(numerator) * kPartsPerMillion >=
           static_cast<wide_uint>(denominator) * threshold_per_million;
}

inline bool scaled_ratio_at_most(std::uint64_t numerator,
                                 std::uint64_t denominator,
                                 std::uint64_t threshold_per_million) {
    if (denominator == 0) {
        return false;
    }
    return static_cast<wide_uint>(numerator) * kPartsPerMillion <=
           static_cast<wide_uint>(denominator) * threshold_per_million;
}

inline bool mean_at_most(std::uint64_t sum, std::uint64_t samples,
                         std::uint64_t threshold_ns) {
    return samples != 0 &&
           static_cast<wide_uint>(sum) <=
               static_cast<wide_uint>(samples) * threshold_ns;
}

inline bool mean_at_least(std::uint64_t sum, std::uint64_t samples,
                          std::uint64_t threshold_ns) {
    return samples != 0 &&
           static_cast<wide_uint>(sum) >=
               static_cast<wide_uint>(samples) * threshold_ns;
}

} // namespace detail

// Pure policy function.  It reads only the supplied profile and injected
// classifier config; no physical placement, budget, object, or recovery state
// can affect the result.
inline IntrinsicClassification intrinsic_behavior_signature(
    const RegionProfile &profile, const IntrinsicClassifierConfig &config) {
    if (!config.valid()) {
        throw std::invalid_argument("invalid intrinsic classifier config");
    }

    IntrinsicClassification result;
    result.complete_window = profile.complete_window;
    result.lifecycle_events = profile.lifecycle_events();
    result.eviction_events = profile.eviction_events();
    result.demand_references = profile.demand_references();
    result.lifecycle_pressure = profile.lifecycle_pressure();
    result.dirty_fraction_per_million = detail::scaled_floor(
        profile.dirty_evict_commit, result.eviction_events);
    result.lifecycle_rate_per_million = detail::scaled_floor(
        result.lifecycle_pressure, profile.live_object_exposure);
    result.activity_rate_per_million = detail::scaled_floor(
        result.demand_references, profile.live_object_exposure);
    if (profile.completed_lifetime_samples != 0) {
        result.mean_completed_lifetime_ns =
            profile.completed_lifetime_sum_ns /
            profile.completed_lifetime_samples;
    }

    const auto &gate = config.evidence;
    const bool v3_gate =
        gate.version == IntrinsicClassifierVersion::V3;
    // A bare four-field EvidenceGate{...} historically represented v1 when
    // its lifetime floor was nonzero; preserve that source-compatible shape.
    const bool v1_gate =
        gate.version == IntrinsicClassifierVersion::V1 ||
        (gate.version == IntrinsicClassifierVersion::V2 &&
         gate.min_completed_lifetime_samples != 0);

    // v3 removes the erroneous conjunction of all predicate evidence.  Its
    // common gate is complete_window, L >= 8, and X > 0; E/A/lifetime floors
    // are applied only when their corresponding predicates are evaluated.
    result.evidence_qualified =
        profile.complete_window &&
        result.lifecycle_events >= gate.min_lifecycle_events &&
        profile.live_object_exposure > 0 &&
        (v3_gate ||
         (result.eviction_events >= gate.min_eviction_events &&
          result.demand_references >= gate.min_demand_references &&
          (!v1_gate ||
           profile.completed_lifetime_samples >=
               gate.min_completed_lifetime_samples)));
    if (!result.evidence_qualified) {
        return result;
    }

    const auto &thresholds = config.thresholds;
    const bool has_lifetime_evidence =
        profile.completed_lifetime_samples >=
        kMinLifetimeSamplesForPredicate;
    // v3 evaluates each signal with only its own evidence floor.  The v1/v2
    // common gate already implies these floors, so retaining the checks here
    // is behavior-preserving and makes the v3 rule explicit.
    result.raw.transient =
        (result.eviction_events >= gate.min_eviction_events &&
         detail::scaled_ratio_at_least(
             profile.dirty_evict_commit, result.eviction_events,
             thresholds.theta_dirty_per_million)) ||
        (has_lifetime_evidence &&
         detail::mean_at_most(profile.completed_lifetime_sum_ns,
                              profile.completed_lifetime_samples,
                              thresholds.theta_short_lifetime_ns));
    result.raw.frequent_lifecycle =
        result.lifecycle_events >= gate.min_lifecycle_events &&
        detail::scaled_ratio_at_least(
            result.lifecycle_pressure, profile.live_object_exposure,
            thresholds.theta_lifecycle_rate_per_million);
    result.raw.stable_cold =
        has_lifetime_evidence &&
        detail::mean_at_least(profile.completed_lifetime_sum_ns,
                              profile.completed_lifetime_samples,
                              thresholds.theta_long_lifetime_ns) &&
        detail::scaled_ratio_at_most(
            result.demand_references, profile.live_object_exposure,
            thresholds.theta_activity_rate_per_million);

    if (result.raw.transient) {
        result.signature = IntrinsicSignature::Transient;
    } else if (result.raw.frequent_lifecycle) {
        result.signature = IntrinsicSignature::FrequentLifecycle;
    } else if (result.raw.stable_cold) {
        result.signature = IntrinsicSignature::StableCold;
    } else {
        result.signature = IntrinsicSignature::Default;
    }
    return result;
}

// Convenient spelling matching the policy plan.  EvidenceGate remains
// injectable for small deterministic fixtures; its default is the revised v3.
inline IntrinsicClassification intrinsic_behavior_signature(
    const RegionProfile &profile, const IntrinsicThresholds &thresholds,
    const EvidenceGate &evidence = EvidenceGate::frozen_v3()) {
    return intrinsic_behavior_signature(
        profile, IntrinsicClassifierConfig{thresholds, evidence});
}

inline IntrinsicClassification classify_intrinsic_behavior(
    const RegionProfile &profile, const IntrinsicClassifierConfig &config) {
    return intrinsic_behavior_signature(profile, config);
}

struct RegionToken {
    // address is an opaque stable region identity.  generation_id prevents a
    // recycled address from inheriting an archived generation's metadata.
    std::uint64_t address = 0;
    std::uint64_t generation_id = 0;

    bool operator==(const RegionToken &other) const {
        return address == other.address && generation_id == other.generation_id;
    }
    bool operator!=(const RegionToken &other) const { return !(*this == other); }
    bool operator<(const RegionToken &other) const {
        return address < other.address ||
               (address == other.address && generation_id < other.generation_id);
    }
};

struct RegionTokenHash {
    std::size_t operator()(const RegionToken &token) const noexcept {
        const auto h1 = std::hash<std::uint64_t>{}(token.address);
        const auto h2 = std::hash<std::uint64_t>{}(token.generation_id);
        return h1 ^ (h2 + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) +
                     (h1 << 6) + (h1 >> 2));
    }
};

struct InitialGroupKey {
    std::uint64_t logical_owner_id = 0;
    std::uint64_t allocator_size_bin = 0;

    bool operator==(const InitialGroupKey &other) const {
        return logical_owner_id == other.logical_owner_id &&
               allocator_size_bin == other.allocator_size_bin;
    }
    bool operator<(const InitialGroupKey &other) const {
        return logical_owner_id < other.logical_owner_id ||
               (logical_owner_id == other.logical_owner_id &&
                allocator_size_bin < other.allocator_size_bin);
    }
};

struct InitialGroupKeyHash {
    std::size_t operator()(const InitialGroupKey &key) const noexcept {
        const auto h1 = std::hash<std::uint64_t>{}(key.logical_owner_id);
        const auto h2 = std::hash<std::uint64_t>{}(key.allocator_size_bin);
        return h1 ^ (h2 + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) +
                     (h1 << 6) + (h1 >> 2));
    }
};

struct RegionObservation {
    RegionToken token;
    RegionProfile profile;
};

// Optional policy for persistent physical regions.  Unlike the historical
// classifier above, this policy compares ranges of directly observed metrics
// and deliberately has no lifecycle/window/stability floor.  It is disabled
// by default; the runtime adapter opts in explicitly.
struct PersistentRangeConfig {
    bool enabled = false;
    std::uint64_t max_dirty_fraction_range_per_million = 200'000;
    std::uint64_t max_hotness_ratio = 2;
    std::uint64_t max_lifetime_ratio = 2;
    bool allow_root_targets = true;

    bool valid() const {
        return max_dirty_fraction_range_per_million <= kPartsPerMillion &&
               max_hotness_ratio >= 1 && max_lifetime_ratio >= 1;
    }
};

// Opt-in bounded policy for grouping current logical-primary regions by
// primary occupancy and dirty tier.  This policy is intentionally independent
// of the historical intrinsic classifier and persistent-range policy.
struct CapacityGroupingConfig {
    bool enabled = false;
    std::uint64_t local_capacity_bytes = 0;
    std::uint64_t dirty_low_per_million = 330'000;
    std::uint64_t dirty_high_per_million = 670'000;
    // Appended for aggregate-initializer compatibility. Share the capacity
    // classifier, but manage actual split/merge groups instead of routing
    // each region to six preassigned canonical slots.
    bool dynamic_split_merge = false;
    // Explicit fixed-six policy.  The historical fixed-slot capacity mode
    // remains selected by dynamic_split_merge == false; this bit is only for
    // the new bounded policy whose root is metadata-only and whose bootstrap
    // route is ColdMedium.
    bool fixed_six_groups = false;
    bool carry_dirty_class = true;

    bool valid() const {
        return (!dynamic_split_merge || enabled) &&
               (!fixed_six_groups || (enabled && !dynamic_split_merge)) &&
               dirty_low_per_million < dirty_high_per_million &&
               dirty_high_per_million <= kPartsPerMillion;
    }
};

enum class CapacityDirtyClass : std::uint8_t {
    Unknown = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Med = Medium,
};

enum class CapacityHeatClass : std::uint8_t {
    Unknown = 0,
    Cold = 1,
    Hot = 2,
    NoHeat = Unknown,
};

// The six non-retired canonical target groups under one initial root.  In the
// historical fixed-slot mode Unknown may leave a new region in its root
// holding group; fixed-six mode uses ColdMedium as a temporary bootstrap and
// never routes through an Unknown target.
enum class CapacityGroupClass : std::uint8_t {
    Unknown = 0,
    ColdLow = 1,
    ColdMedium = 2,
    ColdHigh = 3,
    HotLow = 4,
    HotMedium = 5,
    HotHigh = 6,
    ColdMed = ColdMedium,
    HotMed = HotMedium,
};

using CapacityDirtyBand = CapacityDirtyClass;
using CapacityHeatBand = CapacityHeatClass;
using CapacityClass = CapacityGroupClass;

inline const char *to_string(CapacityDirtyClass value) {
    switch (value) {
    case CapacityDirtyClass::Unknown:
        return "Unknown";
    case CapacityDirtyClass::Low:
        return "Low";
    case CapacityDirtyClass::Medium:
        return "Medium";
    case CapacityDirtyClass::High:
        return "High";
    }
    return "Unknown";
}

inline const char *to_string(CapacityHeatClass value) {
    switch (value) {
    case CapacityHeatClass::Unknown:
        return "Unknown";
    case CapacityHeatClass::Cold:
        return "Cold";
    case CapacityHeatClass::Hot:
        return "Hot";
    }
    return "Unknown";
}

inline const char *to_string(CapacityGroupClass value) {
    switch (value) {
    case CapacityGroupClass::Unknown:
        return "Unknown";
    case CapacityGroupClass::ColdLow:
        return "ColdLow";
    case CapacityGroupClass::ColdMedium:
        return "ColdMedium";
    case CapacityGroupClass::ColdHigh:
        return "ColdHigh";
    case CapacityGroupClass::HotLow:
        return "HotLow";
    case CapacityGroupClass::HotMedium:
        return "HotMedium";
    case CapacityGroupClass::HotHigh:
        return "HotHigh";
    }
    return "Unknown";
}

inline CapacityGroupClass capacity_group_class(CapacityHeatClass heat,
                                               CapacityDirtyClass dirty) {
    if (heat == CapacityHeatClass::Cold) {
        switch (dirty) {
        case CapacityDirtyClass::Low:
            return CapacityGroupClass::ColdLow;
        case CapacityDirtyClass::Medium:
            return CapacityGroupClass::ColdMedium;
        case CapacityDirtyClass::High:
            return CapacityGroupClass::ColdHigh;
        default:
            return CapacityGroupClass::Unknown;
        }
    }
    if (heat == CapacityHeatClass::Hot) {
        switch (dirty) {
        case CapacityDirtyClass::Low:
            return CapacityGroupClass::HotLow;
        case CapacityDirtyClass::Medium:
            return CapacityGroupClass::HotMedium;
        case CapacityDirtyClass::High:
            return CapacityGroupClass::HotHigh;
        default:
            return CapacityGroupClass::Unknown;
        }
    }
    return CapacityGroupClass::Unknown;
}

inline bool capacity_group_class_known(CapacityGroupClass value) {
    return value != CapacityGroupClass::Unknown;
}

inline std::size_t capacity_group_slot(CapacityGroupClass value) {
    if (!capacity_group_class_known(value)) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(value) - 1;
}

// Fixed-six routing always has one deterministic bootstrap slot.  Dirty is
// deliberately unknown until an eviction denominator is observed; the
// ColdMedium label is therefore only a temporary placement default, not a
// fabricated dirty observation.
inline constexpr CapacityGroupClass kFixedSixBootstrapClass =
    CapacityGroupClass::ColdMedium;
inline constexpr std::size_t kFixedSixGroupCount = 6;

enum class PersistentRangeMetric : std::uint8_t {
    DirtyFraction = 0,
    Hotness = 1,
    MeanLifetime = 2,
};

struct PersistentRangeCompatibility {
    bool has_occupied_profile = false;
    bool has_common_metric = false;
    bool compatible = false;
    std::size_t occupied_profile_count = 0;
    std::array<bool, 3> comparable_metrics{};
    std::array<bool, 3> metric_ranges_valid{};
};

inline bool persistent_profile_occupied(const RegionProfile &profile) {
    return profile.occupancy_known && profile.live_objects_at_end != 0;
}

namespace detail {

struct persistent_metric_value {
    std::uint64_t numerator = 0;
    std::uint64_t denominator = 0;
};

// A pair of uint64 counters can require 128 bits just for a cross product;
// multiplying by the configured ratio requires another 64 bits.  Keep the
// policy comparison exact for saturated-looking inputs instead of relying on
// a floating-point approximation or a wrapping uint128 product.
using persistent_big_uint = starfish_refinement_persistent_wide_uint;

struct persistent_metric_summary {
    bool has_value = false;
    std::size_t value_count = 0;
    persistent_metric_value minimum;
    persistent_metric_value maximum;
};

struct persistent_range_summary {
    bool has_occupied_profile = false;
    std::size_t occupied_profile_count = 0;
    std::array<persistent_metric_summary, 3> metrics{};
};

inline bool persistent_metric_known(const RegionProfile &profile,
                                    PersistentRangeMetric metric,
                                    persistent_metric_value &value) {
    switch (metric) {
    case PersistentRangeMetric::DirtyFraction: {
        const auto denominator = profile.eviction_events();
        if (denominator == 0) return false;
        value = {profile.dirty_evict_commit, denominator};
        return true;
    }
    case PersistentRangeMetric::Hotness:
        if (profile.live_object_time_ns == 0) return false;
        value = {profile.demand_references(), profile.live_object_time_ns};
        return true;
    case PersistentRangeMetric::MeanLifetime:
        if (profile.completed_lifetime_samples == 0) return false;
        value = {profile.completed_lifetime_sum_ns,
                 profile.completed_lifetime_samples};
        return true;
    }
    return false;
}

inline int persistent_compare_ratio(const persistent_metric_value &lhs,
                                    const persistent_metric_value &rhs) {
    const persistent_big_uint left =
        static_cast<persistent_big_uint>(lhs.numerator) * rhs.denominator;
    const persistent_big_uint right =
        static_cast<persistent_big_uint>(rhs.numerator) * lhs.denominator;
    return left < right ? -1 : (left > right ? 1 : 0);
}

inline void persistent_summary_add(
    persistent_range_summary &summary, const RegionProfile &profile) {
    if (!persistent_profile_occupied(profile)) return;
    summary.has_occupied_profile = true;
    if (summary.occupied_profile_count != std::numeric_limits<std::size_t>::max())
        ++summary.occupied_profile_count;
    for (std::size_t index = 0; index < summary.metrics.size(); ++index) {
        persistent_metric_value value;
        if (!persistent_metric_known(
                profile, static_cast<PersistentRangeMetric>(index), value)) {
            continue;
        }
        auto &metric = summary.metrics[index];
        if (!metric.has_value) {
            metric.minimum = value;
            metric.maximum = value;
            metric.has_value = true;
        } else {
            if (persistent_compare_ratio(value, metric.minimum) < 0)
                metric.minimum = value;
            if (persistent_compare_ratio(value, metric.maximum) > 0)
                metric.maximum = value;
        }
        if (metric.value_count != std::numeric_limits<std::size_t>::max())
            ++metric.value_count;
    }
}

inline void persistent_summary_merge(persistent_range_summary &destination,
                                     const persistent_range_summary &source) {
    destination.has_occupied_profile =
        destination.has_occupied_profile || source.has_occupied_profile;
    const auto max = std::numeric_limits<std::size_t>::max();
    destination.occupied_profile_count =
        source.occupied_profile_count > max - destination.occupied_profile_count
            ? max
            : destination.occupied_profile_count + source.occupied_profile_count;
    for (std::size_t index = 0; index < destination.metrics.size(); ++index) {
        const auto &rhs = source.metrics[index];
        if (!rhs.has_value) continue;
        auto &lhs = destination.metrics[index];
        if (!lhs.has_value) {
            lhs = rhs;
            continue;
        }
        if (persistent_compare_ratio(rhs.minimum, lhs.minimum) < 0)
            lhs.minimum = rhs.minimum;
        if (persistent_compare_ratio(rhs.maximum, lhs.maximum) > 0)
            lhs.maximum = rhs.maximum;
        const auto max = std::numeric_limits<std::size_t>::max();
        lhs.value_count = rhs.value_count > max - lhs.value_count
                              ? max
                              : lhs.value_count + rhs.value_count;
    }
}

inline bool persistent_ratio_within(
    const persistent_metric_value &lhs,
    const persistent_metric_value &rhs, std::uint64_t ratio) {
    // A known zero is distinct from a known positive value.  This prevents a
    // no-demand profile from sharing with a positive-demand profile merely
    // because integer scaling rounded the former to zero.
    if (lhs.numerator == 0 || rhs.numerator == 0) {
        return lhs.numerator == rhs.numerator;
    }
    const persistent_big_uint left =
        static_cast<persistent_big_uint>(lhs.numerator) * rhs.denominator;
    const persistent_big_uint right =
        static_cast<persistent_big_uint>(rhs.numerator) * lhs.denominator *
        ratio;
    if (left > right) return false;
    const persistent_big_uint reverse_left =
        static_cast<persistent_big_uint>(rhs.numerator) * lhs.denominator;
    const persistent_big_uint reverse_right =
        static_cast<persistent_big_uint>(lhs.numerator) * rhs.denominator *
        ratio;
    return reverse_left <= reverse_right;
}

inline bool persistent_dirty_range_within(
    const persistent_metric_value &lhs,
    const persistent_metric_value &rhs, std::uint64_t range_per_million) {
    const persistent_big_uint lhs_cross =
        static_cast<persistent_big_uint>(lhs.numerator) * rhs.denominator;
    const persistent_big_uint rhs_cross =
        static_cast<persistent_big_uint>(rhs.numerator) * lhs.denominator;
    const persistent_big_uint difference = lhs_cross >= rhs_cross
                                               ? lhs_cross - rhs_cross
                                               : rhs_cross - lhs_cross;
    const persistent_big_uint scaled_difference = difference * kPartsPerMillion;
    const persistent_big_uint allowance =
        static_cast<persistent_big_uint>(range_per_million) * lhs.denominator *
        rhs.denominator;
    return scaled_difference <= allowance;
}

inline PersistentRangeCompatibility persistent_summary_compatibility(
    const persistent_range_summary &summary,
    const PersistentRangeConfig &config) {
    PersistentRangeCompatibility result;
    result.has_occupied_profile = summary.has_occupied_profile;
    result.occupied_profile_count = summary.occupied_profile_count;
    for (std::size_t index = 0; index < summary.metrics.size(); ++index) {
        const auto &metric = summary.metrics[index];
        if (metric.value_count < 2) {
            // No pair exists for this metric; that is unknown evidence, not a
            // range violation.  It still cannot establish a merge by itself.
            result.metric_ranges_valid[index] = true;
            continue;
        }
        result.has_common_metric = true;
        result.comparable_metrics[index] = true;
        if (index == static_cast<std::size_t>(
                         PersistentRangeMetric::DirtyFraction)) {
            result.metric_ranges_valid[index] = persistent_dirty_range_within(
                metric.minimum, metric.maximum,
                config.max_dirty_fraction_range_per_million);
        } else {
            const auto ratio = index == static_cast<std::size_t>(
                                          PersistentRangeMetric::Hotness)
                                   ? config.max_hotness_ratio
                                   : config.max_lifetime_ratio;
            result.metric_ranges_valid[index] =
                persistent_ratio_within(metric.minimum, metric.maximum, ratio);
        }
    }
    // A set is internally compatible whenever none of its known metric
    // ranges violates the configured bound.  Missing/disjoint metrics remain
    // unknown rather than forcing a split; the side-aware merge predicate
    // below separately requires one metric known on both groups.
    bool all_ranges_valid = true;
    for (const bool valid : result.metric_ranges_valid)
        all_ranges_valid = all_ranges_valid && valid;
    result.compatible = result.has_occupied_profile && all_ranges_valid;
    return result;
}

inline PersistentRangeCompatibility persistent_summary_cross_compatibility(
    const persistent_range_summary &lhs, const persistent_range_summary &rhs,
    const PersistentRangeConfig &config) {
    persistent_range_summary union_summary = lhs;
    persistent_summary_merge(union_summary, rhs);
    auto result = persistent_summary_compatibility(union_summary, config);
    // A metric known only on one side is not evidence that the two groups are
    // comparable.  Require a known value on both occupied sides explicitly;
    // this avoids merging an observed group with a data-less group.
    result.has_common_metric = false;
    result.comparable_metrics.fill(false);
    for (std::size_t index = 0; index < union_summary.metrics.size(); ++index) {
        if (lhs.metrics[index].value_count != 0 &&
            rhs.metrics[index].value_count != 0) {
            result.has_common_metric = true;
            result.comparable_metrics[index] = true;
        }
    }
    result.compatible = result.has_occupied_profile &&
                        result.has_common_metric;
    for (std::size_t index = 0; index < union_summary.metrics.size(); ++index) {
        if (!result.metric_ranges_valid[index]) result.compatible = false;
    }
    return result;
}

} // namespace detail

// The one compatibility predicate used by both persistent splitting and
// persistent merging.  Idle and occupancy-unknown profiles are excluded from
// comparisons.  At least one metric must be known for two occupied profiles;
// otherwise a merge would be based solely on missing data.
inline PersistentRangeCompatibility persistent_profiles_compatibility(
    const std::vector<RegionProfile> &profiles,
    const PersistentRangeConfig &config = PersistentRangeConfig{}) {
    if (!config.valid()) {
        throw std::invalid_argument("invalid persistent range config");
    }
    detail::persistent_range_summary summary;
    for (const auto &profile : profiles)
        detail::persistent_summary_add(summary, profile);
    return detail::persistent_summary_compatibility(summary, config);
}

inline bool persistent_profiles_share(
    const std::vector<RegionProfile> &profiles,
    const PersistentRangeConfig &config = PersistentRangeConfig{}) {
    return persistent_profiles_compatibility(profiles, config).compatible;
}

// Side-aware form used for merge decisions.  The one-vector form answers
// whether a set is internally within range; this overload additionally
// requires a comparable metric on each side, so one data-less group cannot
// be merged merely because the other side has observations.
inline PersistentRangeCompatibility persistent_profiles_compatibility(
    const std::vector<RegionProfile> &lhs,
    const std::vector<RegionProfile> &rhs,
    const PersistentRangeConfig &config = PersistentRangeConfig{}) {
    detail::persistent_range_summary lhs_summary;
    detail::persistent_range_summary rhs_summary;
    for (const auto &profile : lhs)
        detail::persistent_summary_add(lhs_summary, profile);
    for (const auto &profile : rhs)
        detail::persistent_summary_add(rhs_summary, profile);
    return detail::persistent_summary_cross_compatibility(lhs_summary,
                                                           rhs_summary,
                                                           config);
}

inline bool persistent_profiles_share(
    const std::vector<RegionProfile> &lhs,
    const std::vector<RegionProfile> &rhs,
    const PersistentRangeConfig &config = PersistentRangeConfig{}) {
    return persistent_profiles_compatibility(lhs, rhs, config).compatible;
}

struct RegistryConfig {
    IntrinsicClassifierConfig classifier;
    // Both bounds include roots.  They are finite by construction so a
    // failed metadata allocation can never affect the workload itself.
    std::size_t max_regions = 4096;
    std::size_t max_groups = 4096;

    // Merge is deliberately opt-in.  Keeping this bit false preserves the
    // v3 split-only registry's behavior for existing callers.  A candidate
    // must be observed in this many consecutive complete windows before it
    // can participate in a merge.
    bool enable_merge = false;
    std::size_t merge_min_stable_windows = 2;
    // Boundary values are global lifecycle-event ordinals in the runtime.
    // Standalone callers using window indices may explicitly select stride 1.
    std::uint64_t merge_boundary_stride = 4096;
    // Coverage is an optional audit stream.  It is disabled by default so
    // existing split/merge callers retain their legacy metadata footprint.
    bool record_merge_coverage = false;
    // The persistent policy is opt-in and leaves the historical classifier
    // and merge gate untouched when disabled.
    PersistentRangeConfig persistent_range;
    // The capacity policy is a separate opt-in mode.  It is mutually
    // exclusive with persistent_range so the two policies cannot silently
    // compete for the same boundary.
    CapacityGroupingConfig capacity_grouping;
    // Capacity boundaries retain lightweight aggregate summaries by default.
    // Set this false for a runtime that publishes only aggregate summaries and
    // actual reassignments; no per-region CapacityRegionDecision audit rows
    // are then constructed or retained.
    bool record_capacity_details = true;

    bool valid() const {
        // The persistent range policy deliberately does not evaluate the
        // historical classifier.  Keep its legacy configuration requirement
        // when that policy is disabled, but allow a range-only caller to
        // leave classifier thresholds at their unused defaults.
        return (persistent_range.enabled || capacity_grouping.enabled ||
                classifier.valid()) &&
               max_regions != 0 && max_groups != 0 &&
               persistent_range.valid() &&
               capacity_grouping.valid() &&
               !(persistent_range.enabled && capacity_grouping.enabled) &&
               (!enable_merge || persistent_range.enabled ||
                capacity_grouping.enabled ||
                (merge_min_stable_windows != 0 &&
                                  merge_boundary_stride != 0));
    }
};

struct GroupSnapshot {
    GroupId id = 0;
    RootGroupId root_group_id = 0;
    GroupId parent_group_id = 0;
    std::uint32_t depth = 0;
    InitialGroupKey initial_key;
    bool has_signature = false;
    IntrinsicSignature signature = IntrinsicSignature::Unclassified;
    std::size_t member_count = 0;
    std::vector<GroupId> child_group_ids;

    // A retired group remains in the historical registry for auditability,
    // but is never a current routing target.  merged_into_group_id is a
    // historical edge, not an alias to follow while routing a region.
    GroupId merged_into_group_id = 0;
    std::uint64_t merged_at_boundary = 0;
    bool retired = false;
    bool is_active_region_group = false;

    // Capacity target groups are canonical, non-retired children whose
    // identity is reused across boundaries.  The root itself remains the
    // Unknown/new-allocation holding group.
    bool capacity_group = false;
    CapacityGroupClass capacity_class = CapacityGroupClass::Unknown;
    // Dynamic labels summarize this boundary's observed evidence; they are
    // not permanent group identities and do not claim unknown members have
    // a measured label.  Fixed-slot capacity_group keeps its old meaning.
    bool dynamic_capacity_group = false;
    CapacityHeatClass current_heat_class = CapacityHeatClass::Unknown;
    CapacityDirtyClass current_dirty_class = CapacityDirtyClass::Unknown;
    bool heat_mixed = false;
    bool dirty_mixed = false;
    std::uint64_t evidence_boundary = 0;
};

struct RegionSnapshot {
    RegionToken token;
    InitialGroupKey initial_key;
    RootGroupId root_group_id = 0;
    GroupId current_group_id = 0;
};

struct SplitRecord {
    std::uint64_t boundary = 0;
    RootGroupId root_group_id = 0;
    GroupId parent_group_id = 0;
    std::uint32_t parent_depth = 0;
    bool successful = false;
    std::string failure_reason;
    std::size_t qualified_region_count = 0;
    std::vector<IntrinsicSignature> conflicting_signatures;
    std::vector<GroupId> child_group_ids;
    // These fields are populated only by the opt-in persistent range policy;
    // legacy records retain their historical classifier interpretation.
    bool persistent_range_policy = false;
    std::size_t occupied_region_count = 0;
    std::size_t idle_region_count = 0;
    std::size_t unknown_region_count = 0;
    bool dynamic_capacity_policy = false;
};

// Immutable description of one successful online merge.  The pooled profile
// is the exact checked sum of the source and target profiles from the
// boundary; no ratio/mean is averaged in the registry.
struct MergeRecord {
    std::uint64_t boundary = 0;
    RootGroupId root_group_id = 0;
    GroupId source_group_id = 0;
    GroupId target_group_id = 0;
    IntrinsicSignature signature = IntrinsicSignature::Unclassified;
    std::size_t moved_region_count = 0;
    RegionProfile pooled_profile;
    IntrinsicClassification pooled_classification;

    // Short aliases make the record convenient for standalone serializers
    // while the *_group_id names remain the canonical API.
    GroupId source = 0;
    GroupId target = 0;
    // Persistent range-policy audit fields.  Existing merge consumers may
    // ignore these appended fields without changing the legacy ABI shape.
    bool persistent_range_policy = false;
    bool persistent_has_common_metric = false;
    std::array<bool, 3> persistent_metric_ranges_valid{};
    std::size_t persistent_occupied_region_count = 0;
    std::size_t persistent_idle_region_count = 0;
    std::size_t persistent_unknown_region_count = 0;
    std::vector<RegionToken> persistent_source_member_tokens;
    std::vector<RegionToken> persistent_target_member_tokens;
    bool dynamic_capacity_policy = false;
};

using MergeOperation = MergeRecord;

// Per-boundary audit of the current non-empty region groups immediately after
// split processing and immediately before merge eligibility is evaluated.
// The counters form a total partition of member_count in this order:
// missing_or_duplicate > incomplete > below_lifecycle_floor > zero_exposure
// > other_unqualified > qualified.
struct MergeCoverageRecord {
    std::uint64_t boundary = 0;
    GroupId group_id = 0;
    RootGroupId root_group_id = 0;
    std::size_t member_count = 0;
    std::size_t present_nonduplicate_count = 0;
    std::size_t qualified_count = 0;
    std::size_t missing_or_duplicate_count = 0;
    std::size_t incomplete_count = 0;
    std::size_t below_lifecycle_floor_count = 0;
    std::size_t zero_exposure_count = 0;
    std::size_t other_unqualified_count = 0;
    bool split_touched = false;
    bool root_excluded = false;
};

enum class PersistentRangeDecisionKind : std::uint8_t {
    Split = 0,
    Merge = 1,
};

inline const char *to_string(PersistentRangeDecisionKind kind) {
    return kind == PersistentRangeDecisionKind::Split ? "range_split"
                                                       : "range_merge";
}

// Audit record for the opt-in persistent policy.  The token lists are the
// exact pre-mutation memberships, so a decision remains checkable even when
// a source group is immediately retired or a boundary contains a split.
struct PersistentRangeDecisionRecord {
    std::uint64_t boundary = 0;
    PersistentRangeDecisionKind kind = PersistentRangeDecisionKind::Split;
    RootGroupId root_group_id = 0;
    GroupId source_group_id = 0;
    GroupId target_group_id = 0;
    std::vector<GroupId> child_group_ids;
    std::vector<RegionToken> source_member_tokens;
    std::vector<RegionToken> target_member_tokens;
    std::size_t occupied_region_count = 0;
    std::size_t idle_region_count = 0;
    std::size_t unknown_region_count = 0;
    bool has_common_metric = false;
    bool compatible = false;
    std::array<bool, 3> comparable_metrics{};
    std::array<bool, 3> metric_ranges_valid{};
};

struct RegionRebindRecord {
    RegionToken token;
    InitialGroupKey previous_initial_key;
    InitialGroupKey target_initial_key;
    RootGroupId previous_root_group_id = 0;
    RootGroupId target_root_group_id = 0;
    GroupId previous_group_id = 0;
    GroupId target_group_id = 0;
};

// One auditable routing decision made by the bounded capacity policy.  A
// decision is emitted for every accepted, unique observation; deferred
// observations retain their previous group and carry an explicit reason via
// the evidence booleans below rather than being assigned a fabricated class.
struct CapacityRegionDecision {
    std::uint64_t boundary = 0;
    RegionToken token;
    InitialGroupKey initial_key;
    RootGroupId root_group_id = 0;

    GroupId previous_group_id = 0;
    GroupId new_group_id = 0;
    // Short aliases are convenient for serializers and callers that describe
    // a transition as prev/new.
    GroupId previous_group = 0;
    GroupId new_group = 0;

    CapacityGroupClass group_class = CapacityGroupClass::Unknown;
    CapacityHeatClass heat_class = CapacityHeatClass::Unknown;
    CapacityDirtyClass dirty_class = CapacityDirtyClass::Unknown;
    bool dirty_class_current_known = false;
    bool dirty_class_carried = false;
    // Compact aliases for the two dirty-evidence states above.
    bool current_known = false;
    bool carried = false;

    bool primary_census_known = false;
    bool heat_evidence_known = false;
    bool hot_candidate = false;
    bool hot_selected = false;
    bool first_crossing = false;
    bool deferred = false;
    bool moved = false;
    bool overflow = false;

    std::uint64_t demand_references = 0;
    std::uint64_t hot_rank_numerator = 0;
    std::uint64_t hot_rank_denominator = 0;
    std::uint64_t primary_live_objects = 0;
    std::uint64_t primary_live_bytes = 0;
    std::uint64_t primary_object_time_ns = 0;

    std::uint64_t cumulative_hot_bytes = 0;
    std::uint64_t hot_budget_bytes = 0;
    std::uint64_t total_primary_bytes = 0;
    std::uint64_t crossing_overshoot_bytes = 0;
    std::size_t moved_region_count = 0;
};

// Per-boundary summary for the capacity policy.  When
// RegistryConfig::record_capacity_details is true, `decisions` is retained in
// the summary as well as in BoundaryResult.capacity_decisions so a standalone
// serializer can consume one record without joining two streams.  Lean mode
// leaves this vector empty while preserving all aggregate fields.
struct CapacityBoundaryRecord {
    std::uint64_t boundary = 0;
    std::uint64_t local_capacity_bytes = 0;
    std::uint64_t hot_budget_bytes = 0;
    std::uint64_t total_primary_bytes = 0;
    std::uint64_t cumulative_hot_bytes = 0;
    std::uint64_t first_crossing_overshoot_bytes = 0;

    bool has_first_crossing = false;
    RegionToken first_crossing_token;
    bool total_primary_bytes_overflow = false;
    bool cumulative_hot_bytes_overflow = false;

    std::size_t observed_region_count = 0;
    std::size_t known_primary_region_count = 0;
    std::size_t idle_region_count = 0;
    std::size_t unknown_region_count = 0;
    std::size_t hot_candidate_count = 0;
    std::size_t hot_region_count = 0;
    std::size_t deferred_region_count = 0;
    std::size_t moved_region_count = 0;
    std::size_t moved_group_count = 0;
    std::size_t created_group_count = 0;
    bool overflow = false;

    std::vector<CapacityRegionDecision> decisions;
};

// Compact capacity-routing delta for runtimes that do not retain the full
// per-region CapacityRegionDecision audit stream.  An entry exists only when
// the registry actually changes a region's current group.
struct CapacityRegionReassignment {
    RegionToken token;
    GroupId previous_group_id = 0;
    GroupId new_group_id = 0;
};

enum class DynamicCapacityDecisionKind : std::uint8_t { Split = 0, Merge = 1 };

inline const char *to_string(DynamicCapacityDecisionKind value) {
    return value == DynamicCapacityDecisionKind::Split ? "dynamic_split"
                                                      : "dynamic_merge";
}

struct DynamicCapacityPartitionRecord {
    GroupId group_id = 0;
    CapacityHeatClass heat_class = CapacityHeatClass::Unknown;
    CapacityDirtyClass dirty_class = CapacityDirtyClass::Unknown;
    std::vector<RegionToken> member_tokens;
};

// Exact pre-mutation memberships and child partitions permit replay of all
// operations, including a child that merges again within the same boundary.
// Partial labels constrain only their known axis.  The counts are disjoint:
// evidence + idle + unknown = source members + target members (for merge).
struct DynamicCapacityDecisionRecord {
    std::uint64_t boundary = 0;
    DynamicCapacityDecisionKind kind = DynamicCapacityDecisionKind::Split;
    RootGroupId root_group_id = 0;
    GroupId source_group_id = 0;
    GroupId target_group_id = 0;
    std::vector<RegionToken> source_member_tokens;
    std::vector<RegionToken> target_member_tokens;
    std::vector<DynamicCapacityPartitionRecord> children;
    std::vector<RegionToken> parent_retained_tokens;
    CapacityHeatClass source_heat_class = CapacityHeatClass::Unknown;
    CapacityDirtyClass source_dirty_class = CapacityDirtyClass::Unknown;
    bool source_heat_mixed = false;
    bool source_dirty_mixed = false;
    CapacityHeatClass target_heat_class = CapacityHeatClass::Unknown;
    CapacityDirtyClass target_dirty_class = CapacityDirtyClass::Unknown;
    CapacityHeatClass pooled_heat_class = CapacityHeatClass::Unknown;
    CapacityDirtyClass pooled_dirty_class = CapacityDirtyClass::Unknown;
    bool has_common_heat = false;
    bool has_common_dirty = false;
    bool compatible = false;
    std::size_t evidence_region_count = 0;
    std::size_t idle_region_count = 0;
    std::size_t unknown_region_count = 0;
};

struct RegistrySnapshot {
    RegistryConfig config;
    std::vector<GroupSnapshot> groups;
    std::vector<RegionSnapshot> regions;
    std::vector<SplitRecord> splits;
    std::vector<MergeRecord> merges;
    std::vector<MergeCoverageRecord> merge_coverage;
    std::vector<PersistentRangeDecisionRecord> persistent_decisions;
    std::vector<RegionRebindRecord> rebinds;
    std::vector<CapacityRegionDecision> capacity_decisions;
    std::vector<CapacityBoundaryRecord> capacity_boundaries;
    std::vector<DynamicCapacityDecisionRecord> dynamic_capacity_decisions;

    std::size_t initial_group_count = 0;
    std::size_t historical_group_count = 0;
    std::size_t active_group_count = 0;
    std::size_t refined_initial_group_count = 0;
    std::size_t refinement_event_count = 0;
    std::size_t child_group_count = 0;
    std::size_t max_refinements_per_initial_group = 0;
    std::uint32_t max_refinement_depth = 0;
    std::size_t failed_refinement_event_count = 0;
    std::size_t owner_mismatch_count = 0;
    std::size_t ignored_observation_count = 0;
    std::size_t duplicate_observation_count = 0;

    std::size_t merged_group_count = 0;
    std::size_t merge_event_count = 0;
    std::size_t moved_region_count = 0;
    std::size_t merge_missing_evidence_count = 0;
    std::size_t merge_mixed_label_count = 0;
    std::size_t merge_aggregate_changed_count = 0;
    std::size_t merge_overflow_count = 0;
    std::size_t merge_stability_wait_count = 0;

    // Capacity-policy counters intentionally stay separate from legacy split,
    // merge, and persistent-range counters.
    std::size_t capacity_boundary_count = 0;
    std::size_t capacity_decision_count = 0;
    std::size_t capacity_group_count = 0;
    std::size_t capacity_hot_candidate_count = 0;
    std::size_t capacity_hot_region_count = 0;
    std::size_t capacity_deferred_region_count = 0;
    std::size_t capacity_moved_region_count = 0;
    std::size_t capacity_moved_group_count = 0;
    std::size_t capacity_idle_region_count = 0;
    std::size_t capacity_unknown_region_count = 0;
    std::size_t capacity_overflow_count = 0;
};

enum class RegionRegistrationStatus : std::uint8_t {
    Created = 0,
    AlreadyRegistered = 1,
    KeyMismatch = 2,
    MetadataExhausted = 3,
    Rebound = 4,
    UnknownRegion = 5,
};

struct RegionRegistrationResult {
    RegionRegistrationStatus status =
        RegionRegistrationStatus::MetadataExhausted;
    GroupId current_group_id = 0;
    GroupId previous_group_id = 0;

    bool success() const {
        return status == RegionRegistrationStatus::Created ||
               status == RegionRegistrationStatus::AlreadyRegistered ||
               status == RegionRegistrationStatus::Rebound;
    }
    bool created() const {
        return status == RegionRegistrationStatus::Created;
    }
    bool rebound() const { return status == RegionRegistrationStatus::Rebound; }
};

enum class SplitStatus : std::uint8_t {
    NoConflict = 0,
    SplitSucceeded = 1,
    MetadataExhausted = 2,
    BoundaryAlreadyProcessed = 3,
};

struct SplitOperation {
    std::uint64_t boundary = 0;
    RootGroupId root_group_id = 0;
    GroupId parent_group_id = 0;
    SplitStatus status = SplitStatus::NoConflict;
    std::size_t qualified_region_count = 0;
    std::size_t unclassified_region_count = 0;
    std::vector<IntrinsicSignature> conflicting_signatures;
    std::vector<GroupId> child_group_ids;
};

struct BoundaryResult {
    std::uint64_t boundary = 0;
    bool already_processed = false;
    std::size_t observation_count = 0;
    std::size_t accepted_observation_count = 0;
    std::size_t ignored_observation_count = 0;
    std::size_t duplicate_observation_count = 0;
    std::vector<SplitOperation> operations;
    std::vector<MergeRecord> merges;
    std::vector<MergeCoverageRecord> merge_coverage;
    std::vector<PersistentRangeDecisionRecord> persistent_decisions;
    std::vector<CapacityRegionDecision> capacity_decisions;
    std::vector<CapacityRegionReassignment> capacity_reassignments;
    std::vector<CapacityBoundaryRecord> capacity_boundaries;
    std::vector<DynamicCapacityDecisionRecord> dynamic_capacity_decisions;

    std::size_t successful_event_count() const {
        std::size_t result = 0;
        for (const auto &operation : operations) {
            result += operation.status == SplitStatus::SplitSucceeded;
        }
        return result;
    }

    std::size_t successful_merge_event_count() const {
        return merges.size();
    }
};

// Metadata-only, bounded region-group registry.  All mutating operations are
// serialized under one mutex.  This makes a boundary's split linearization
// point explicit and keeps concurrent observers from creating duplicate
// events or partially materialized child groups.
class BehaviorGroupRegistry {
private:
    struct GroupState {
        GroupSnapshot snapshot;
        std::unordered_set<RegionToken, RegionTokenHash> members;
        std::map<IntrinsicSignature, GroupId> children;
        // children is the current routing pointer for each signature.  Keep
        // every allocated child separately so replacing a retired pointer
        // does not erase historical lineage from the snapshot.
        std::vector<GroupId> historical_child_group_ids;
    };

    struct RegionState {
        InitialGroupKey initial_key;
        RootGroupId root_group_id = 0;
        GroupId current_group_id = 0;
    };

    struct MergeStabilityState {
        IntrinsicSignature signature = IntrinsicSignature::Unclassified;
        std::vector<RegionToken> member_tokens;
        std::size_t consecutive_windows = 0;
        std::uint64_t last_boundary = 0;
    };

    struct MergeCandidate {
        GroupId group_id = 0;
        RootGroupId root_group_id = 0;
        InitialGroupKey initial_key;
        IntrinsicSignature signature = IntrinsicSignature::Unclassified;
        std::vector<RegionToken> member_tokens;
        RegionProfile pooled_profile;
        IntrinsicClassification pooled_classification;
        bool stable = false;
    };

    mutable std::mutex mutex_;
    RegistryConfig config_;
    GroupId next_group_id_ = 1;
    std::map<GroupId, GroupState> groups_;
    std::unordered_map<InitialGroupKey, RootGroupId, InitialGroupKeyHash>
        roots_by_key_;
    std::unordered_map<RegionToken, RegionState, RegionTokenHash> regions_;
    std::set<std::pair<std::uint64_t, GroupId>> processed_boundary_parents_;
    std::set<std::uint64_t> processed_boundaries_;
    std::optional<std::uint64_t> latest_accepted_boundary_;
    std::vector<SplitRecord> splits_;
    std::vector<MergeRecord> merges_;
    std::vector<MergeCoverageRecord> merge_coverage_;
    std::vector<PersistentRangeDecisionRecord> persistent_decisions_;
    std::vector<RegionRebindRecord> rebinds_;
    static constexpr std::size_t kCapacityTargetGroupCount = 6;
    using CapacityTargetGroups =
        std::array<GroupId, kCapacityTargetGroupCount>;
    // A root owns one stable slot per known heat/dirty label.  Slots are
    // allocated lazily and are never retired or replaced by later flips.
    std::unordered_map<RootGroupId, CapacityTargetGroups>
        capacity_target_groups_;
    std::unordered_map<RegionToken, CapacityDirtyClass, RegionTokenHash>
        capacity_last_dirty_class_;
    std::vector<CapacityRegionDecision> capacity_decisions_;
    std::vector<CapacityBoundaryRecord> capacity_boundaries_;
    std::vector<DynamicCapacityDecisionRecord> dynamic_capacity_decisions_;
    std::unordered_map<RootGroupId, std::size_t> root_event_counts_;
    std::set<RootGroupId> refined_roots_;
    std::unordered_map<GroupId, MergeStabilityState> merge_stability_;

    std::size_t refinement_event_count_ = 0;
    std::size_t child_group_count_ = 0;
    std::size_t failed_refinement_event_count_ = 0;
    std::size_t owner_mismatch_count_ = 0;
    std::size_t ignored_observation_count_ = 0;
    std::size_t duplicate_observation_count_ = 0;
    std::size_t merged_group_count_ = 0;
    std::size_t merge_event_count_ = 0;
    std::size_t moved_region_count_ = 0;
    std::size_t merge_missing_evidence_count_ = 0;
    std::size_t merge_mixed_label_count_ = 0;
    std::size_t merge_aggregate_changed_count_ = 0;
    std::size_t merge_overflow_count_ = 0;
    std::size_t merge_stability_wait_count_ = 0;

    std::size_t capacity_boundary_count_ = 0;
    std::size_t capacity_decision_count_ = 0;
    std::size_t capacity_group_count_ = 0;
    std::size_t capacity_hot_candidate_count_ = 0;
    std::size_t capacity_hot_region_count_ = 0;
    std::size_t capacity_deferred_region_count_ = 0;
    std::size_t capacity_moved_region_count_ = 0;
    std::size_t capacity_moved_group_count_ = 0;
    std::size_t capacity_idle_region_count_ = 0;
    std::size_t capacity_unknown_region_count_ = 0;
    std::size_t capacity_overflow_count_ = 0;

    static bool has_capacity(std::size_t current, std::size_t limit,
                             std::size_t additional) {
        return current <= limit && additional <= limit - current;
    }

    GroupId allocate_group_id_locked() {
        if (next_group_id_ == 0) {
            return 0;
        }
        const GroupId id = next_group_id_++;
        if (id == 0) {
            return 0;
        }
        return id;
    }

    GroupId canonical_group_locked(GroupId id) const {
        while (id != 0) {
            const auto it = groups_.find(id);
            if (it == groups_.end()) return 0;
            const auto next = it->second.snapshot.merged_into_group_id;
            if (next == 0) return id;
            if (next >= id) return 0;
            id = next;
        }
        return 0;
    }

    GroupId ensure_root_locked(const InitialGroupKey &key) {
        const auto found = roots_by_key_.find(key);
        if (found != roots_by_key_.end()) return found->second;
        if (groups_.size() >= config_.max_groups) return 0;
        const auto root = allocate_group_id_locked();
        if (!root) return 0;
        GroupState state;
        state.snapshot.id = root;
        state.snapshot.root_group_id = root;
        state.snapshot.initial_key = key;
        groups_.emplace(root, std::move(state));
        roots_by_key_.emplace(key, root);
        root_event_counts_.emplace(root, 0);
        return root;
    }

    GroupId capacity_target_group_locked(RootGroupId root,
                                         CapacityGroupClass group_class) const {
        const auto slot = capacity_group_slot(group_class);
        if (slot >= kCapacityTargetGroupCount) {
            return 0;
        }
        const auto root_it = capacity_target_groups_.find(root);
        if (root_it == capacity_target_groups_.end()) {
            return 0;
        }
        return root_it->second[slot];
    }

    // Allocate all missing canonical slots for one root transactionally.  A
    // failed preflight or ID allocation leaves both the registry and slot
    // table unchanged, so a capacity boundary never partially repartitions a
    // root because metadata was exhausted.
    bool ensure_capacity_target_groups_locked(
        RootGroupId root, const std::set<CapacityGroupClass> &requested) {
        if (requested.empty()) {
            return true;
        }
        auto root_it = groups_.find(root);
        if (root_it == groups_.end()) {
            return false;
        }

        auto &slots = capacity_target_groups_[root];
        std::size_t missing = 0;
        for (const auto group_class : requested) {
            const auto slot = capacity_group_slot(group_class);
            if (slot < kCapacityTargetGroupCount && slots[slot] == 0) {
                ++missing;
            }
        }
        if (!has_capacity(groups_.size(), config_.max_groups, missing)) {
            return false;
        }

        std::vector<std::pair<std::size_t, GroupId>> allocated;
        allocated.reserve(missing);
        for (const auto group_class : requested) {
            const auto slot = capacity_group_slot(group_class);
            if (slot >= kCapacityTargetGroupCount || slots[slot] != 0) {
                continue;
            }
            const auto id = allocate_group_id_locked();
            if (id == 0) {
                for (const auto &[allocated_slot, allocated_id] : allocated) {
                    slots[allocated_slot] = 0;
                    groups_.erase(allocated_id);
                    if (capacity_group_count_ != 0) {
                        --capacity_group_count_;
                    }
                }
                return false;
            }

            GroupState child;
            child.snapshot.id = id;
            child.snapshot.root_group_id = root;
            child.snapshot.parent_group_id = root;
            child.snapshot.depth = root_it->second.snapshot.depth + 1;
            child.snapshot.initial_key = root_it->second.snapshot.initial_key;
            child.snapshot.capacity_group = true;
            child.snapshot.capacity_class = group_class;
            groups_.emplace(id, std::move(child));
            root_it->second.historical_child_group_ids.push_back(id);
            slots[slot] = id;
            allocated.emplace_back(slot, id);
            ++capacity_group_count_;
        }
        return true;
    }

    bool fixed_six_mode_locked() const {
        return config_.capacity_grouping.enabled &&
               config_.capacity_grouping.fixed_six_groups;
    }

    GroupId ensure_fixed_six_group_locked(
        RootGroupId root, CapacityGroupClass group_class) {
        if (!fixed_six_mode_locked() ||
            !capacity_group_class_known(group_class)) {
            return 0;
        }
        const std::set<CapacityGroupClass> requested{group_class};
        if (!ensure_capacity_target_groups_locked(root, requested)) {
            return 0;
        }
        return capacity_target_group_locked(root, group_class);
    }

    std::array<GroupId, kFixedSixGroupCount>
    ensure_fixed_six_family_locked(RootGroupId root) {
        std::array<GroupId, kFixedSixGroupCount> result{};
        if (!fixed_six_mode_locked()) {
            return result;
        }
        const std::set<CapacityGroupClass> requested{
            CapacityGroupClass::ColdLow,
            CapacityGroupClass::ColdMedium,
            CapacityGroupClass::ColdHigh,
            CapacityGroupClass::HotLow,
            CapacityGroupClass::HotMedium,
            CapacityGroupClass::HotHigh,
        };
        if (!ensure_capacity_target_groups_locked(root, requested)) {
            return result;
        }
        for (std::size_t slot = 0; slot < result.size(); ++slot) {
            result[slot] = capacity_target_group_locked(
                root, static_cast<CapacityGroupClass>(slot + 1));
        }
        return result;
    }

    static bool checked_add_u64(std::uint64_t lhs, std::uint64_t rhs,
                                std::uint64_t &result) {
        constexpr auto max = std::numeric_limits<std::uint64_t>::max();
        if (lhs > max - rhs) {
            return false;
        }
        result = lhs + rhs;
        return true;
    }

    // RegionProfile's public derived accessors intentionally saturate for
    // ordinary classification.  Merge cannot use those saturated values:
    // an apparently valid pooled profile would otherwise silently lose
    // evidence.  Recompute every derived sufficient statistic with checked
    // arithmetic and reject malformed lifetime sums as well.
    static bool profile_derived_totals_fit(const RegionProfile &profile) {
        std::uint64_t ignored = 0;
        return checked_add_u64(profile.fetch_commit,
                               profile.clean_evict_commit, ignored) &&
               checked_add_u64(ignored, profile.dirty_evict_commit, ignored) &&
               checked_add_u64(ignored, profile.free_commit, ignored) &&
               checked_add_u64(profile.clean_evict_commit,
                               profile.dirty_evict_commit, ignored) &&
               checked_add_u64(profile.demand_read_reference,
                               profile.demand_write_reference, ignored) &&
               checked_add_u64(profile.fetch_commit,
                               profile.dirty_evict_commit, ignored) &&
               checked_add_u64(ignored, profile.free_commit, ignored) &&
               (profile.completed_lifetime_samples != 0 ||
                profile.completed_lifetime_sum_ns == 0);
    }

    static bool checked_add_profile(const RegionProfile &lhs,
                                   const RegionProfile &rhs,
                                   RegionProfile &result) {
        if (!profile_derived_totals_fit(lhs) ||
            !profile_derived_totals_fit(rhs)) {
            return false;
        }
        RegionProfile sum;
        sum.complete_window = lhs.complete_window && rhs.complete_window;
        if (!checked_add_u64(lhs.fetch_commit, rhs.fetch_commit,
                             sum.fetch_commit) ||
            !checked_add_u64(lhs.clean_evict_commit, rhs.clean_evict_commit,
                             sum.clean_evict_commit) ||
            !checked_add_u64(lhs.dirty_evict_commit, rhs.dirty_evict_commit,
                             sum.dirty_evict_commit) ||
            !checked_add_u64(lhs.free_commit, rhs.free_commit,
                             sum.free_commit) ||
            !checked_add_u64(lhs.demand_read_reference,
                             rhs.demand_read_reference,
                             sum.demand_read_reference) ||
            !checked_add_u64(lhs.demand_write_reference,
                             rhs.demand_write_reference,
                             sum.demand_write_reference) ||
            !checked_add_u64(lhs.live_object_exposure,
                             rhs.live_object_exposure,
                             sum.live_object_exposure) ||
            !checked_add_u64(lhs.live_object_time_ns,
                             rhs.live_object_time_ns,
                             sum.live_object_time_ns) ||
            !checked_add_u64(lhs.completed_lifetime_sum_ns,
                             rhs.completed_lifetime_sum_ns,
                             sum.completed_lifetime_sum_ns) ||
            !checked_add_u64(lhs.completed_lifetime_samples,
                             rhs.completed_lifetime_samples,
                             sum.completed_lifetime_samples) ||
            !profile_derived_totals_fit(sum)) {
            return false;
        }
        // Endpoint occupancy is meaningful for a pooled audit profile only
        // when both inputs came from authoritative censuses.  Unknown must
        // never become a fabricated zero through aggregation.
        sum.occupancy_known = lhs.occupancy_known && rhs.occupancy_known;
        if (sum.occupancy_known &&
            !checked_add_u64(lhs.live_objects_at_end, rhs.live_objects_at_end,
                             sum.live_objects_at_end)) {
            return false;
        }
        if (!sum.occupancy_known) sum.live_objects_at_end = 0;
        result = sum;
        return true;
    }

    static bool same_members(const std::vector<RegionToken> &lhs,
                             const std::vector<RegionToken> &rhs) {
        return lhs == rhs;
    }

    static bool is_retired(const GroupState &state) {
        return state.snapshot.retired ||
               state.snapshot.merged_into_group_id != 0;
    }

    void reset_merge_stability_for_group_locked(GroupId group_id) {
        merge_stability_.erase(group_id);
    }

    // Validate all arithmetic that can affect capacity routing before the
    // boundary is entered into processed_boundaries_.  A malformed profile is
    // rejected as a whole; saturating it to UINT64_MAX would fabricate a
    // dirty tier, rank, or cache-budget witness.
    void validate_capacity_boundary_locked(
        const std::vector<RegionObservation> &observations) const {
        std::unordered_set<RegionToken, RegionTokenHash> seen;
        std::uint64_t total_primary_bytes = 0;
        for (const auto &observation : observations) {
            if (regions_.find(observation.token) == regions_.end()) {
                continue;
            }
            if (!seen.insert(observation.token).second) {
                continue;
            }
            const auto &profile = observation.profile;
            std::uint64_t ignored = 0;
            if (!checked_add_u64(profile.clean_evict_commit,
                                 profile.dirty_evict_commit, ignored)) {
                throw std::invalid_argument(
                    "capacity eviction counter overflow");
            }
            if (!checked_add_u64(profile.demand_read_reference,
                                 profile.demand_write_reference, ignored)) {
                throw std::invalid_argument(
                    "capacity demand counter overflow");
            }
            if (profile.primary_census_known &&
                !checked_add_u64(total_primary_bytes,
                                 profile.primary_live_bytes,
                                 total_primary_bytes)) {
                throw std::invalid_argument(
                    "capacity primary-byte total overflow");
            }
        }
    }

    // A dynamic whole-group move must account for every physical member.
    // Reject incomplete/ambiguous input before marking a boundary processed,
    // changing dirty carry state, or mutating membership. Fixed mode retains
    // its historical first-observation behavior.
    void validate_dynamic_capacity_input_locked(
        const std::vector<RegionObservation> &observations) const {
        std::unordered_set<RegionToken, RegionTokenHash> seen;
        for (const auto &observation : observations) {
            if (regions_.find(observation.token) == regions_.end())
                throw std::invalid_argument("dynamic capacity unknown region");
            if (!seen.insert(observation.token).second)
                throw std::invalid_argument("dynamic capacity duplicate region");
        }
        if (seen.size() != regions_.size())
            throw std::invalid_argument("dynamic capacity missing region");
    }

    // Pool raw sufficient statistics, not ratios or categorical labels.
    // This separate helper leaves legacy profile aggregation unchanged and
    // explicitly preserves every newly added primary-census field. Lifetime
    // is retained for audit only; no lifetime threshold or shape gate applies.
    static bool checked_add_dynamic_profile(const RegionProfile &lhs,
                                            const RegionProfile &rhs,
                                            RegionProfile &result) {
        RegionProfile sum;
        sum.complete_window = lhs.complete_window && rhs.complete_window;
#define FARLIB_DYNAMIC_SUM(field)                                      \
        if (!checked_add_u64(lhs.field, rhs.field, sum.field)) return false
        FARLIB_DYNAMIC_SUM(fetch_commit);
        FARLIB_DYNAMIC_SUM(clean_evict_commit);
        FARLIB_DYNAMIC_SUM(dirty_evict_commit);
        FARLIB_DYNAMIC_SUM(free_commit);
        FARLIB_DYNAMIC_SUM(demand_read_reference);
        FARLIB_DYNAMIC_SUM(demand_write_reference);
        FARLIB_DYNAMIC_SUM(live_object_exposure);
        FARLIB_DYNAMIC_SUM(live_object_time_ns);
        FARLIB_DYNAMIC_SUM(completed_lifetime_sum_ns);
        FARLIB_DYNAMIC_SUM(completed_lifetime_samples);
        sum.occupancy_known = lhs.occupancy_known && rhs.occupancy_known;
        if (sum.occupancy_known) {
            FARLIB_DYNAMIC_SUM(live_objects_at_end);
        }
        sum.primary_census_known =
            lhs.primary_census_known && rhs.primary_census_known;
        if (sum.primary_census_known) {
            FARLIB_DYNAMIC_SUM(primary_live_objects);
            FARLIB_DYNAMIC_SUM(primary_live_bytes);
            FARLIB_DYNAMIC_SUM(primary_object_time_ns);
        }
#undef FARLIB_DYNAMIC_SUM
        std::uint64_t ignored = 0;
        if (!checked_add_u64(sum.clean_evict_commit, sum.dirty_evict_commit, ignored) ||
            !checked_add_u64(sum.demand_read_reference, sum.demand_write_reference, ignored))
            return false;
        result = sum;
        return true;
    }

    // Classifier output is shared with fixed mode. This helper changes only
    // the management mechanism: partition existing groups, then merge whole
    // compatible groups. Known axes constrain compatibility independently.
    template <typename PreparedObservation>
    std::size_t refine_dynamic_capacity_groups_locked(
        std::uint64_t boundary,
        const std::vector<PreparedObservation> &prepared,
        BoundaryResult &result) {
        struct Evidence {
            CapacityHeatClass heat = CapacityHeatClass::Unknown;
            CapacityDirtyClass dirty = CapacityDirtyClass::Unknown;
            bool heat_mixed = false;
            bool dirty_mixed = false;
            std::size_t occupied_count = 0;
            std::size_t evidence_count = 0;
            std::size_t idle_count = 0;
            std::size_t unknown_count = 0;
        };
        struct Partition {
            std::vector<RegionToken> tokens;
            Evidence evidence;
        };
        std::unordered_map<RegionToken, const PreparedObservation *,
                           RegionTokenHash> by_token;
        for (const auto &entry : prepared) by_token.emplace(entry.token, &entry);
        const auto tokens_for = [](const GroupState &state) {
            std::vector<RegionToken> tokens(state.members.begin(),
                                            state.members.end());
            std::sort(tokens.begin(), tokens.end());
            return tokens;
        };
        const auto combine = [](Evidence lhs, const Evidence &rhs) {
            if (rhs.heat != CapacityHeatClass::Unknown) {
                if (lhs.heat == CapacityHeatClass::Unknown) lhs.heat = rhs.heat;
                else if (lhs.heat != rhs.heat) lhs.heat_mixed = true;
            }
            if (rhs.dirty != CapacityDirtyClass::Unknown) {
                if (lhs.dirty == CapacityDirtyClass::Unknown) lhs.dirty = rhs.dirty;
                else if (lhs.dirty != rhs.dirty) lhs.dirty_mixed = true;
            }
            lhs.heat_mixed = lhs.heat_mixed || rhs.heat_mixed;
            lhs.dirty_mixed = lhs.dirty_mixed || rhs.dirty_mixed;
            lhs.occupied_count += rhs.occupied_count;
            lhs.evidence_count += rhs.evidence_count;
            lhs.idle_count += rhs.idle_count;
            lhs.unknown_count += rhs.unknown_count;
            return lhs;
        };
        const auto member_evidence = [&](const RegionToken &token) {
            Evidence evidence;
            const auto &entry = *by_token.at(token);
            if (entry.primary_known && !entry.known_primary) {
                evidence.idle_count = 1;
            } else if (!entry.known_primary) {
                evidence.unknown_count = 1;
            } else {
                evidence.occupied_count = 1;
                evidence.heat = entry.heat_class;
                evidence.dirty = entry.dirty_class;
                if (evidence.heat != CapacityHeatClass::Unknown ||
                    evidence.dirty != CapacityDirtyClass::Unknown)
                    evidence.evidence_count = 1;
                else evidence.unknown_count = 1;
            }
            return evidence;
        };
        const auto summarize = [&](const std::vector<RegionToken> &tokens) {
            Evidence evidence;
            for (const auto &token : tokens)
                evidence = combine(evidence, member_evidence(token));
            return evidence;
        };
        const auto compatible = [&](const Evidence &lhs, const Evidence &rhs,
                                    bool require_common) {
            const auto pooled = combine(lhs, rhs);
            const bool common =
                (lhs.heat != CapacityHeatClass::Unknown &&
                 rhs.heat != CapacityHeatClass::Unknown) ||
                (lhs.dirty != CapacityDirtyClass::Unknown &&
                 rhs.dirty != CapacityDirtyClass::Unknown);
            return !pooled.heat_mixed && !pooled.dirty_mixed &&
                   (!require_common || common);
        };
        const auto stamp_source = [](DynamicCapacityDecisionRecord &record,
                                      const Evidence &evidence) {
            record.source_heat_class = evidence.heat_mixed
                ? CapacityHeatClass::Unknown : evidence.heat;
            record.source_dirty_class = evidence.dirty_mixed
                ? CapacityDirtyClass::Unknown : evidence.dirty;
            record.source_heat_mixed = evidence.heat_mixed;
            record.source_dirty_mixed = evidence.dirty_mixed;
            record.evidence_region_count = evidence.evidence_count;
            record.idle_region_count = evidence.idle_count;
            record.unknown_region_count = evidence.unknown_count;
        };

        std::size_t created_group_count = 0;
        std::set<GroupId> split_touched;
        std::vector<GroupId> original_groups;
        for (const auto &[id, state] : groups_)
            if (!is_retired(state) && !state.members.empty())
                original_groups.push_back(id);
        for (const auto parent_id : original_groups) {
            auto &parent = groups_.at(parent_id);
            const auto original_tokens = tokens_for(parent);
            const auto original_evidence = summarize(original_tokens);
            SplitOperation operation;
            operation.boundary = boundary;
            operation.parent_group_id = parent_id;
            operation.root_group_id = parent.snapshot.root_group_id;
            operation.qualified_region_count = original_evidence.evidence_count;
            operation.unclassified_region_count = original_evidence.idle_count +
                                                  original_evidence.unknown_count;
            std::vector<Partition> partitions;
            std::vector<RegionToken> retained;
            for (const auto &token : original_tokens) {
                const auto one = member_evidence(token);
                if (one.evidence_count == 0) {
                    retained.push_back(token);
                    continue;
                }
                bool placed = false;
                for (auto &partition : partitions) {
                    if (!compatible(partition.evidence, one, false)) continue;
                    partition.tokens.push_back(token);
                    partition.evidence = combine(partition.evidence, one);
                    placed = true;
                    break;
                }
                if (!placed) partitions.push_back(Partition{{token}, one});
            }
            if (partitions.size() < 2) {
                operation.status = SplitStatus::NoConflict;
                result.operations.push_back(std::move(operation));
                continue;
            }

            const auto max_id = std::numeric_limits<GroupId>::max();
            const bool id_room = next_group_id_ != 0 &&
                partitions.size() - 1 <= max_id - next_group_id_;
            const bool depth_room = parent.snapshot.depth !=
                std::numeric_limits<std::uint32_t>::max();
            if (!has_capacity(groups_.size(), config_.max_groups,
                              partitions.size()) || !id_room || !depth_room) {
                operation.status = SplitStatus::MetadataExhausted;
                ++failed_refinement_event_count_;
                SplitRecord record;
                record.boundary = boundary;
                record.root_group_id = operation.root_group_id;
                record.parent_group_id = parent_id;
                record.parent_depth = parent.snapshot.depth;
                record.failure_reason = "metadata_exhausted";
                record.qualified_region_count = original_evidence.evidence_count;
                record.occupied_region_count = original_evidence.occupied_count;
                record.idle_region_count = original_evidence.idle_count;
                record.unknown_region_count = original_evidence.unknown_count;
                record.dynamic_capacity_policy = true;
                splits_.push_back(std::move(record));
                result.operations.push_back(std::move(operation));
                continue;
            }

            DynamicCapacityDecisionRecord decision;
            decision.boundary = boundary;
            decision.kind = DynamicCapacityDecisionKind::Split;
            decision.root_group_id = operation.root_group_id;
            decision.source_group_id = parent_id;
            decision.source_member_tokens = original_tokens;
            decision.parent_retained_tokens = retained;
            stamp_source(decision, original_evidence);
            // All finite metadata/ID/depth checks precede mutation. IDs and
            // ancestry are new for this actual split; no six-slot reuse.
            for (const auto &partition : partitions) {
                const auto child_id = allocate_group_id_locked();
                GroupState child;
                child.snapshot.id = child_id;
                child.snapshot.root_group_id = operation.root_group_id;
                child.snapshot.parent_group_id = parent_id;
                child.snapshot.depth = parent.snapshot.depth + 1;
                child.snapshot.initial_key = parent.snapshot.initial_key;
                child.snapshot.dynamic_capacity_group = true;
                child.members.insert(partition.tokens.begin(), partition.tokens.end());
                groups_.emplace(child_id, std::move(child));
                parent.historical_child_group_ids.push_back(child_id);
                operation.child_group_ids.push_back(child_id);
                decision.children.push_back(DynamicCapacityPartitionRecord{
                    child_id, partition.evidence.heat, partition.evidence.dirty,
                    partition.tokens});
                for (const auto &token : partition.tokens) {
                    parent.members.erase(token);
                    regions_.at(token).current_group_id = child_id;
                }
                split_touched.insert(child_id);
                ++child_group_count_;
                ++created_group_count;
            }
            split_touched.insert(parent_id);
            operation.status = SplitStatus::SplitSucceeded;
            ++refinement_event_count_;
            ++root_event_counts_[operation.root_group_id];
            refined_roots_.insert(operation.root_group_id);
            SplitRecord record;
            record.boundary = boundary;
            record.root_group_id = operation.root_group_id;
            record.parent_group_id = parent_id;
            record.parent_depth = parent.snapshot.depth;
            record.successful = true;
            record.qualified_region_count = original_evidence.evidence_count;
            record.child_group_ids = operation.child_group_ids;
            record.occupied_region_count = original_evidence.occupied_count;
            record.idle_region_count = original_evidence.idle_count;
            record.unknown_region_count = original_evidence.unknown_count;
            record.dynamic_capacity_policy = true;
            splits_.push_back(std::move(record));
            dynamic_capacity_decisions_.push_back(decision);
            result.dynamic_capacity_decisions.push_back(std::move(decision));
            result.operations.push_back(std::move(operation));
        }

        // Evidence is recomputed on post-split memberships, never inherited
        // from a parent's prior label. Newly split children may merge with a
        // third compatible group, but their conflicting sibling cannot pass.
        std::map<RootGroupId, std::vector<GroupId>> candidates_by_root;
        for (const auto &[id, state] : groups_) {
            if (is_retired(state) || state.members.empty()) continue;
            const auto evidence = summarize(tokens_for(state));
            if (config_.record_merge_coverage) {
                MergeCoverageRecord coverage;
                coverage.boundary = boundary;
                coverage.group_id = id;
                coverage.root_group_id = state.snapshot.root_group_id;
                coverage.member_count = state.members.size();
                coverage.present_nonduplicate_count = state.members.size();
                coverage.qualified_count = evidence.evidence_count;
                coverage.zero_exposure_count = evidence.idle_count;
                coverage.other_unqualified_count = evidence.unknown_count;
                coverage.split_touched = split_touched.count(id) != 0;
                merge_coverage_.push_back(coverage);
                result.merge_coverage.push_back(std::move(coverage));
            }
            if (evidence.evidence_count != 0 &&
                !evidence.heat_mixed && !evidence.dirty_mixed)
                candidates_by_root[state.snapshot.root_group_id].push_back(id);
        }

        if (config_.enable_merge) {
            for (const auto &[root, candidates] : candidates_by_root) {
                (void)root;
                std::vector<GroupId> targets;
                // map iteration made candidates ascending. Every source is
                // newer than its target, so canonical aliases are acyclic and
                // the root is never retired (it may absorb descendants).
                for (const auto source_id : candidates) {
                    bool merged = false;
                    for (const auto target_id : targets) {
                        auto &source = groups_.at(source_id);
                        auto &target = groups_.at(target_id);
                        if (source.snapshot.root_group_id != target.snapshot.root_group_id ||
                            !(source.snapshot.initial_key == target.snapshot.initial_key))
                            continue;
                        const auto source_tokens = tokens_for(source);
                        const auto target_tokens = tokens_for(target);
                        const auto source_evidence = summarize(source_tokens);
                        const auto target_evidence = summarize(target_tokens);
                        if (!compatible(source_evidence, target_evidence, true)) continue;

                        RegionProfile pooled_profile;
                        bool have_pooled = false;
                        bool pooled_ok = true;
                        const auto pool = [&](const std::vector<RegionToken> &tokens) {
                            for (const auto &token : tokens) {
                                const auto &profile = by_token.at(token)->profile;
                                if (!have_pooled) {
                                    pooled_profile = profile;
                                    have_pooled = true;
                                } else if (!checked_add_dynamic_profile(
                                               pooled_profile, profile, pooled_profile)) {
                                    pooled_ok = false;
                                    return;
                                }
                            }
                        };
                        pool(target_tokens);
                        if (pooled_ok) pool(source_tokens);
                        if (!pooled_ok || !have_pooled) {
                            ++merge_overflow_count_;
                            continue;
                        }
                        const auto pooled_evidence = combine(target_evidence, source_evidence);
                        DynamicCapacityDecisionRecord decision;
                        decision.boundary = boundary;
                        decision.kind = DynamicCapacityDecisionKind::Merge;
                        decision.root_group_id = target.snapshot.root_group_id;
                        decision.source_group_id = source_id;
                        decision.target_group_id = target_id;
                        decision.source_member_tokens = source_tokens;
                        decision.target_member_tokens = target_tokens;
                        stamp_source(decision, source_evidence);
                        decision.target_heat_class = target_evidence.heat;
                        decision.target_dirty_class = target_evidence.dirty;
                        decision.pooled_heat_class = pooled_evidence.heat;
                        decision.pooled_dirty_class = pooled_evidence.dirty;
                        decision.has_common_heat = source_evidence.heat != CapacityHeatClass::Unknown &&
                            target_evidence.heat != CapacityHeatClass::Unknown;
                        decision.has_common_dirty = source_evidence.dirty != CapacityDirtyClass::Unknown &&
                            target_evidence.dirty != CapacityDirtyClass::Unknown;
                        decision.compatible = true;
                        decision.evidence_region_count = pooled_evidence.evidence_count;
                        decision.idle_region_count = pooled_evidence.idle_count;
                        decision.unknown_region_count = pooled_evidence.unknown_count;

                        for (const auto &token : source_tokens) {
                            target.members.insert(token);
                            regions_.at(token).current_group_id = target_id;
                        }
                        source.members.clear();
                        source.snapshot.merged_into_group_id = target_id;
                        source.snapshot.merged_at_boundary = boundary;
                        source.snapshot.retired = true;
                        source.snapshot.is_active_region_group = false;
                        MergeRecord record;
                        record.boundary = boundary;
                        record.root_group_id = target.snapshot.root_group_id;
                        record.source_group_id = record.source = source_id;
                        record.target_group_id = record.target = target_id;
                        record.moved_region_count = source_tokens.size();
                        record.pooled_profile = pooled_profile;
                        record.dynamic_capacity_policy = true;
                        merges_.push_back(record);
                        result.merges.push_back(std::move(record));
                        dynamic_capacity_decisions_.push_back(decision);
                        result.dynamic_capacity_decisions.push_back(std::move(decision));
                        ++merged_group_count_;
                        ++merge_event_count_;
                        moved_region_count_ += source_tokens.size();
                        merged = true;
                        break;
                    }
                    if (!merged) targets.push_back(source_id);
                }
            }
        }

        for (auto &[id, state] : groups_) {
            (void)id;
            const auto evidence = summarize(tokens_for(state));
            state.snapshot.dynamic_capacity_group = true;
            state.snapshot.current_heat_class = evidence.heat_mixed
                ? CapacityHeatClass::Unknown : evidence.heat;
            state.snapshot.current_dirty_class = evidence.dirty_mixed
                ? CapacityDirtyClass::Unknown : evidence.dirty;
            state.snapshot.heat_mixed = evidence.heat_mixed;
            state.snapshot.dirty_mixed = evidence.dirty_mixed;
            state.snapshot.evidence_boundary = boundary;
        }
        return created_group_count;
    }

    // Capacity classifier shared by fixed-slot and dynamic management. The
    // caller holds mutex_. Neither mode consults legacy classification,
    // lifetime gates, or persistent-range thresholds.
    BoundaryResult refine_capacity_boundary_locked(
        std::uint64_t boundary,
        const std::vector<RegionObservation> &observations,
        BoundaryResult result) {
        struct PreparedObservation {
            RegionToken token;
            InitialGroupKey initial_key;
            RootGroupId root_group_id = 0;
            GroupId previous_group_id = 0;
            RegionProfile profile;

            CapacityDirtyClass dirty_class = CapacityDirtyClass::Unknown;
            bool dirty_class_current_known = false;
            bool dirty_class_carried = false;
            bool primary_known = false;
            bool known_primary = false;
            bool heat_evidence_known = false;
            bool hot_candidate = false;
            bool hot_selected = false;
            bool first_crossing = false;
            bool overflow = false;

            std::uint64_t demand_references = 0;
            std::uint64_t hot_rank_numerator = 0;
            std::uint64_t hot_rank_denominator = 0;
            CapacityHeatClass heat_class = CapacityHeatClass::Unknown;
            CapacityGroupClass group_class = CapacityGroupClass::Unknown;
        };

        std::vector<PreparedObservation> prepared;
        prepared.reserve(observations.size());
        std::unordered_set<RegionToken, RegionTokenHash> seen;

        for (const auto &observation : observations) {
            const auto region_it = regions_.find(observation.token);
            if (region_it == regions_.end()) {
                ++result.ignored_observation_count;
                ++ignored_observation_count_;
                continue;
            }
            if (!seen.insert(observation.token).second) {
                ++result.duplicate_observation_count;
                ++duplicate_observation_count_;
                continue;
            }
            ++result.accepted_observation_count;
            PreparedObservation entry;
            entry.token = observation.token;
            entry.initial_key = region_it->second.initial_key;
            entry.root_group_id = region_it->second.root_group_id;
            entry.previous_group_id = region_it->second.current_group_id;
            entry.profile = observation.profile;
            prepared.push_back(std::move(entry));
        }

        // Audit order is stable even when the runtime presents observations in
        // a different order.  The ranking tie-breaker below uses this same
        // RegionToken order.
        std::sort(prepared.begin(), prepared.end(),
                  [](const PreparedObservation &lhs,
                     const PreparedObservation &rhs) {
                      return lhs.token < rhs.token;
                  });

        const auto &capacity_config = config_.capacity_grouping;
        const auto saturating_add_size = [](std::size_t &value,
                                            std::size_t amount) {
            const auto max_size = std::numeric_limits<std::size_t>::max();
            value = amount > max_size - value ? max_size : value + amount;
        };
        std::uint64_t total_primary_bytes = 0;
        std::size_t known_primary_region_count = 0;
        std::size_t idle_region_count = 0;
        std::size_t unknown_region_count = 0;
        for (auto &entry : prepared) {
            RegionProfile &profile = entry.profile;
            std::uint64_t demand_references = 0;
            if (!checked_add_u64(profile.demand_read_reference,
                                 profile.demand_write_reference,
                                 demand_references)) {
                // validate_capacity_boundary_locked() runs before this
                // helper.  Keep this guard for callers that are refactored
                // into the helper later; never route a saturated rank.
                throw std::invalid_argument(
                    "capacity demand counter overflow");
            }
            entry.demand_references = demand_references;
            entry.hot_rank_numerator = demand_references;
            entry.hot_rank_denominator = profile.primary_object_time_ns;

            // E is the exact clean+dirty eviction denominator.  Validation
            // above guarantees that this checked sum, rather than a
            // saturated accessor, drives the tier boundary.
            std::uint64_t eviction_events = 0;
            if (!checked_add_u64(profile.clean_evict_commit,
                                 profile.dirty_evict_commit,
                                 eviction_events)) {
                throw std::invalid_argument(
                    "capacity eviction counter overflow");
            }
            if (eviction_events != 0) {
                const auto numerator =
                    static_cast<detail::persistent_big_uint>(
                        profile.dirty_evict_commit) * kPartsPerMillion;
                const auto low_denominator =
                    static_cast<detail::persistent_big_uint>(eviction_events) *
                    capacity_config.dirty_low_per_million;
                const auto high_denominator =
                    static_cast<detail::persistent_big_uint>(eviction_events) *
                    capacity_config.dirty_high_per_million;
                if (numerator < low_denominator) {
                    entry.dirty_class = CapacityDirtyClass::Low;
                } else if (numerator < high_denominator) {
                    entry.dirty_class = CapacityDirtyClass::Medium;
                } else {
                    entry.dirty_class = CapacityDirtyClass::High;
                }
                entry.dirty_class_current_known = true;
                capacity_last_dirty_class_[entry.token] = entry.dirty_class;
            } else if (capacity_config.carry_dirty_class) {
                const auto previous_dirty =
                    capacity_last_dirty_class_.find(entry.token);
                if (previous_dirty != capacity_last_dirty_class_.end()) {
                    entry.dirty_class = previous_dirty->second;
                    entry.dirty_class_carried = true;
                }
            }

            entry.primary_known = profile.primary_census_known;
            if (!entry.primary_known) {
                ++unknown_region_count;
            } else {
                // The total is based only on the current logical primary
                // census; backups and unknown census values never contribute.
                if (!checked_add_u64(total_primary_bytes,
                                     profile.primary_live_bytes,
                                     total_primary_bytes)) {
                    throw std::invalid_argument(
                        "capacity primary-byte total overflow");
                }
                if (profile.primary_live_objects == 0 ||
                    profile.primary_live_bytes == 0) {
                    ++idle_region_count;
                } else {
                    entry.known_primary = true;
                    ++known_primary_region_count;
                    if (profile.primary_object_time_ns == 0) {
                        ++unknown_region_count;
                    }
                }
            }

            // A heat rank exists only for an authoritative, nonempty primary
            // census with positive object time.  Zero demand is measured cold,
            // not an invitation to fill an arbitrary part of the cache.
            entry.heat_evidence_known =
                entry.primary_known && profile.primary_live_objects != 0 &&
                profile.primary_live_bytes != 0 &&
                profile.primary_object_time_ns != 0;
            if (entry.heat_evidence_known) {
                entry.heat_class = CapacityHeatClass::Cold;
                entry.hot_candidate = demand_references != 0;
            }
        }

        // Rank all roots together.  A rank is a rational number; comparing
        // cross-products in the existing fixed-width big integer avoids
        // overflow and floating-point tie drift.
        std::vector<std::size_t> hot_candidates;
        hot_candidates.reserve(prepared.size());
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            if (prepared[index].hot_candidate) {
                hot_candidates.push_back(index);
            }
        }
        const auto compare_rank = [&](std::size_t lhs_index,
                                      std::size_t rhs_index) {
            const auto &lhs = prepared[lhs_index];
            const auto &rhs = prepared[rhs_index];
            const auto lhs_cross =
                static_cast<detail::persistent_big_uint>(
                    lhs.hot_rank_numerator) * rhs.hot_rank_denominator;
            const auto rhs_cross =
                static_cast<detail::persistent_big_uint>(
                    rhs.hot_rank_numerator) * lhs.hot_rank_denominator;
            if (lhs_cross != rhs_cross) {
                return lhs_cross > rhs_cross;
            }
            return lhs.token < rhs.token;
        };
        std::sort(hot_candidates.begin(), hot_candidates.end(), compare_rank);

        std::uint64_t cumulative_hot_bytes = 0;
        bool cumulative_hot_bytes_overflow = false;
        bool has_first_crossing = false;
        RegionToken first_crossing_token;
        std::uint64_t first_crossing_overshoot = 0;
        if (capacity_config.local_capacity_bytes != 0) {
            for (const auto index : hot_candidates) {
                auto &entry = prepared[index];
                if (has_first_crossing) {
                    break;
                }
                entry.hot_selected = true;
                if (!checked_add_u64(cumulative_hot_bytes,
                                     entry.profile.primary_live_bytes,
                                     cumulative_hot_bytes)) {
                    throw std::invalid_argument(
                        "capacity hot-byte prefix overflow");
                }
                if (cumulative_hot_bytes >=
                    capacity_config.local_capacity_bytes) {
                    has_first_crossing = true;
                    entry.first_crossing = true;
                    first_crossing_token = entry.token;
                    first_crossing_overshoot =
                        cumulative_hot_bytes -
                        capacity_config.local_capacity_bytes;
                }
            }
        }
        (void)cumulative_hot_bytes_overflow;

        std::map<RootGroupId, std::set<CapacityGroupClass>>
            requested_target_groups;
        for (auto &entry : prepared) {
            if (entry.hot_candidate) {
                entry.heat_class = entry.hot_selected
                                       ? CapacityHeatClass::Hot
                                       : CapacityHeatClass::Cold;
            }
            entry.group_class =
                capacity_group_class(entry.heat_class, entry.dirty_class);
            if (capacity_group_class_known(entry.group_class)) {
                requested_target_groups[entry.root_group_id].insert(
                    entry.group_class);
            }
        }

        std::map<RootGroupId, bool> target_groups_ready;
        std::size_t created_group_count = 0;
        if (capacity_config.dynamic_split_merge) {
            created_group_count = refine_dynamic_capacity_groups_locked(
                boundary, prepared, result);
        } else {
            for (const auto &[root_group_id, requested] : requested_target_groups) {
                const std::size_t before = capacity_group_count_;
                const bool ready =
                    ensure_capacity_target_groups_locked(root_group_id, requested);
                target_groups_ready[root_group_id] = ready;
                if (capacity_group_count_ >= before) {
                    saturating_add_size(created_group_count,
                                        capacity_group_count_ - before);
                }
            }
        }

        CapacityBoundaryRecord boundary_record;
        boundary_record.boundary = boundary;
        boundary_record.local_capacity_bytes =
            capacity_config.local_capacity_bytes;
        boundary_record.hot_budget_bytes =
            capacity_config.local_capacity_bytes;
        boundary_record.total_primary_bytes = total_primary_bytes;
        boundary_record.cumulative_hot_bytes = cumulative_hot_bytes;
        boundary_record.first_crossing_overshoot_bytes =
            first_crossing_overshoot;
        boundary_record.has_first_crossing = has_first_crossing;
        boundary_record.first_crossing_token = first_crossing_token;
        boundary_record.total_primary_bytes_overflow = false;
        boundary_record.cumulative_hot_bytes_overflow =
            cumulative_hot_bytes_overflow;
        boundary_record.observed_region_count = prepared.size();
        boundary_record.known_primary_region_count =
            known_primary_region_count;
        boundary_record.idle_region_count = idle_region_count;
        boundary_record.unknown_region_count = unknown_region_count;
        boundary_record.hot_candidate_count = hot_candidates.size();
        boundary_record.created_group_count = created_group_count;
        boundary_record.overflow = false;

        std::set<std::pair<GroupId, GroupId>> moved_group_pairs;
        for (auto &entry : prepared) {
            const bool label_known =
                capacity_group_class_known(entry.group_class);
            const auto ready_it = target_groups_ready.find(entry.root_group_id);
            const bool root_ready =
                ready_it != target_groups_ready.end() && ready_it->second;
            GroupId target_group_id = entry.previous_group_id;
            bool deferred = !label_known || !root_ready;
            if (capacity_config.dynamic_split_merge) {
                target_group_id = regions_.at(entry.token).current_group_id;
                // Deferred denotes absence of individual evidence, not a
                // promise to remain stationary: whole-group merge carries
                // idle and unknown members with their group.
                deferred = !entry.known_primary ||
                    (entry.heat_class == CapacityHeatClass::Unknown &&
                     entry.dirty_class == CapacityDirtyClass::Unknown);
            } else if (!deferred) {
                target_group_id = capacity_target_group_locked(
                    entry.root_group_id, entry.group_class);
                if (target_group_id == 0) {
                    target_group_id = entry.previous_group_id;
                    deferred = true;
                }
            }

            bool moved = false;
            if (capacity_config.dynamic_split_merge) {
                moved = target_group_id != entry.previous_group_id;
                if (moved) moved_group_pairs.emplace(entry.previous_group_id,
                                                     target_group_id);
            } else if (!deferred && target_group_id != entry.previous_group_id) {
                auto previous_it = groups_.find(entry.previous_group_id);
                auto target_it = groups_.find(target_group_id);
                if (previous_it == groups_.end() || target_it == groups_.end() ||
                    target_it->second.snapshot.root_group_id !=
                        entry.root_group_id) {
                    target_group_id = entry.previous_group_id;
                    deferred = true;
                } else {
                    previous_it->second.members.erase(entry.token);
                    target_it->second.members.insert(entry.token);
                    auto region_it = regions_.find(entry.token);
                    if (region_it != regions_.end()) {
                        region_it->second.current_group_id = target_group_id;
                    }
                    moved = true;
                    moved_group_pairs.emplace(entry.previous_group_id,
                                              target_group_id);
                }
            }

            if (entry.hot_selected) {
                saturating_add_size(boundary_record.hot_region_count, 1);
            }
            if (deferred) {
                saturating_add_size(boundary_record.deferred_region_count, 1);
            }
            if (moved) {
                saturating_add_size(boundary_record.moved_region_count, 1);
                result.capacity_reassignments.push_back(
                    CapacityRegionReassignment{entry.token,
                                                entry.previous_group_id,
                                                target_group_id});
            }

            if (config_.record_capacity_details) {
                CapacityRegionDecision decision;
                decision.boundary = boundary;
                decision.token = entry.token;
                decision.initial_key = entry.initial_key;
                decision.root_group_id = entry.root_group_id;
                decision.previous_group_id = entry.previous_group_id;
                decision.new_group_id = target_group_id;
                decision.previous_group = entry.previous_group_id;
                decision.new_group = target_group_id;
                decision.group_class = entry.group_class;
                decision.heat_class = entry.heat_class;
                decision.dirty_class = entry.dirty_class;
                decision.dirty_class_current_known =
                    entry.dirty_class_current_known;
                decision.dirty_class_carried = entry.dirty_class_carried;
                decision.current_known = entry.dirty_class_current_known;
                decision.carried = entry.dirty_class_carried;
                decision.primary_census_known = entry.primary_known;
                decision.heat_evidence_known = entry.heat_evidence_known;
                decision.hot_candidate = entry.hot_candidate;
                decision.hot_selected = entry.hot_selected;
                decision.first_crossing = entry.first_crossing;
                decision.deferred = deferred;
                decision.moved = moved;
                decision.overflow = entry.overflow;
                decision.demand_references = entry.demand_references;
                decision.hot_rank_numerator = entry.hot_rank_numerator;
                decision.hot_rank_denominator = entry.hot_rank_denominator;
                decision.primary_live_objects =
                    entry.profile.primary_live_objects;
                decision.primary_live_bytes = entry.profile.primary_live_bytes;
                decision.primary_object_time_ns =
                    entry.profile.primary_object_time_ns;
                decision.cumulative_hot_bytes = cumulative_hot_bytes;
                decision.hot_budget_bytes =
                    capacity_config.local_capacity_bytes;
                decision.total_primary_bytes = total_primary_bytes;
                decision.crossing_overshoot_bytes =
                    entry.first_crossing ? first_crossing_overshoot : 0;
                decision.moved_region_count = moved ? 1 : 0;

                result.capacity_decisions.push_back(decision);
                capacity_decisions_.push_back(decision);
                boundary_record.decisions.push_back(std::move(decision));
            }
        }

        boundary_record.moved_group_count = moved_group_pairs.size();
        boundary_record.overflow = false;
        saturating_add_size(capacity_boundary_count_, 1);
        saturating_add_size(capacity_decision_count_, prepared.size());
        saturating_add_size(capacity_hot_candidate_count_,
                            hot_candidates.size());
        saturating_add_size(capacity_hot_region_count_,
                            boundary_record.hot_region_count);
        saturating_add_size(capacity_deferred_region_count_,
                            boundary_record.deferred_region_count);
        saturating_add_size(capacity_moved_region_count_,
                            boundary_record.moved_region_count);
        saturating_add_size(capacity_moved_group_count_,
                            boundary_record.moved_group_count);
        saturating_add_size(capacity_idle_region_count_, idle_region_count);
        saturating_add_size(capacity_unknown_region_count_, unknown_region_count);
        capacity_boundaries_.push_back(boundary_record);
        result.capacity_boundaries.push_back(std::move(boundary_record));
        return result;
    }

public:
    explicit BehaviorGroupRegistry(RegistryConfig config)
        : config_(std::move(config)) {
        if (!config_.valid()) {
            throw std::invalid_argument("invalid behavior-group registry config");
        }
    }

    BehaviorGroupRegistry(const BehaviorGroupRegistry &) = delete;
    BehaviorGroupRegistry &operator=(const BehaviorGroupRegistry &) = delete;

    GroupId canonical_group_id(GroupId id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return canonical_group_locked(id);
    }

    std::vector<std::pair<GroupId, GroupId>> routing_alias_snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<GroupId, GroupId>> result;
        result.reserve(groups_.size());
        for (const auto &[id, group] : groups_)
            result.emplace_back(id, canonical_group_locked(id));
        return result;
    }

    GroupId ensure_initial_group(const InitialGroupKey &key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return ensure_root_locked(key);
    }

    // Materialize one fixed-six target for an initial key.  This is separate
    // from ensure_initial_group(): callers that only need the root metadata
    // retain the historical root return value, while fixed-six observers can
    // explicitly obtain the bootstrap child without making the root routable.
    GroupId ensure_fixed_six_group(const InitialGroupKey &key,
                                   CapacityGroupClass group_class) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto root = ensure_root_locked(key);
        if (!root) return 0;
        return ensure_fixed_six_group_locked(root, group_class);
    }

    // Return all six stable target IDs for one initial key.  IDs are allocated
    // transactionally and remain stable for the lifetime of the registry.
    // An all-zero result denotes an unknown root or exhausted metadata.
    std::array<GroupId, kFixedSixGroupCount>
    ensure_fixed_six_group_family(const InitialGroupKey &key) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto root = ensure_root_locked(key);
        if (!root) return {};
        return ensure_fixed_six_family_locked(root);
    }

    // Resolve a group (root or one of its fixed children) to its initial-key
    // family.  This helper is intentionally a metadata-only operation; the
    // observer layer owns any runtime/atomic publication.
    std::array<GroupId, kFixedSixGroupCount>
    ensure_fixed_six_group_family(GroupId group_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto canonical = canonical_group_locked(group_id);
        const auto it = groups_.find(canonical);
        if (it == groups_.end()) return {};
        return ensure_fixed_six_family_locked(
            it->second.snapshot.root_group_id);
    }

    // Return a canonical capacity target when that slot has already been
    // materialized for the root.  Slots are intentionally lazy: an absent
    // value is not a metadata error and simply means no region has required
    // that label yet.
    std::optional<GroupId> capacity_group_id(
        RootGroupId root, CapacityGroupClass group_class) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto id = capacity_target_group_locked(root, group_class);
        if (id == 0) {
            return std::nullopt;
        }
        return id;
    }

    std::array<GroupId, 6> capacity_group_ids(RootGroupId root) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = capacity_target_groups_.find(root);
        if (it == capacity_target_groups_.end()) {
            return std::array<GroupId, 6>{};
        }
        return it->second;
    }

    RegionRegistrationResult register_region(const RegionToken &token,
                                              const InitialGroupKey &key,
                                              GroupId requested_group_id = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        GroupId target = requested_group_id ? canonical_group_locked(requested_group_id) : 0;
        if (requested_group_id && (!target || !(groups_.at(target).snapshot.initial_key == key)))
            return {RegionRegistrationStatus::KeyMismatch, 0};

        // In fixed-six mode a root is an initial-key metadata record only.
        // Convert an explicit root request to the bootstrap child before
        // checking an existing token, and create the same child for a new
        // token with no requested group.  The dirty axis remains Unknown in
        // the capacity carry map until a real eviction denominator arrives.
        if (fixed_six_mode_locked() && target != 0) {
            const auto target_it = groups_.find(target);
            if (target_it == groups_.end())
                return {RegionRegistrationStatus::KeyMismatch, 0};
            if (!target_it->second.snapshot.capacity_group) {
                target = ensure_fixed_six_group_locked(
                    target_it->second.snapshot.root_group_id,
                    kFixedSixBootstrapClass);
                if (!target)
                    return {RegionRegistrationStatus::MetadataExhausted, 0};
            }
        }
        auto existing = regions_.find(token);
        if (existing != regions_.end()) {
            if (!(existing->second.initial_key == key) ||
                (target && existing->second.current_group_id != target)) {
                ++owner_mismatch_count_;
                return {RegionRegistrationStatus::KeyMismatch,
                        existing->second.current_group_id};
            }
            return {RegionRegistrationStatus::AlreadyRegistered,
                    existing->second.current_group_id};
        }
        if (regions_.size() >= config_.max_regions) {
            return {RegionRegistrationStatus::MetadataExhausted, 0};
        }

        const RootGroupId root = target ? groups_.at(target).snapshot.root_group_id : ensure_root_locked(key);
        if (!root) return {RegionRegistrationStatus::MetadataExhausted, 0};
        if (!target) {
            target = fixed_six_mode_locked()
                         ? ensure_fixed_six_group_locked(
                               root, kFixedSixBootstrapClass)
                         : root;
            if (!target)
                return {RegionRegistrationStatus::MetadataExhausted, 0};
        }

        RegionState region;
        region.initial_key = key;
        region.root_group_id = root;
        region.current_group_id = target;
        regions_.emplace(token, region);
        groups_.at(target).members.insert(token);
        // A registration changes the membership universe used by every
        // stability witness for this root.  Other current groups have a
        // disjoint membership universe and retain their witnesses.
        reset_merge_stability_for_group_locked(target);
        return {RegionRegistrationStatus::Created, target};
    }

    // Rebind an existing physical generation without destroying its token or
    // historical group metadata.  This is the only persistent-reuse path:
    // an explicit target selects an existing compatible group, while target
    // zero selects (or creates) the root for the supplied initial key.
    RegionRegistrationResult rebind_region(const RegionToken &token,
                                            const InitialGroupKey &key,
                                            GroupId requested_group_id = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto region_it = regions_.find(token);
        if (region_it == regions_.end()) {
            return {RegionRegistrationStatus::UnknownRegion, 0, 0};
        }

        GroupId target = 0;
        if (requested_group_id != 0) {
            target = canonical_group_locked(requested_group_id);
            if (!target || groups_.find(target) == groups_.end() ||
                !(groups_.at(target).snapshot.initial_key == key)) {
                ++owner_mismatch_count_;
                return {RegionRegistrationStatus::KeyMismatch,
                        region_it->second.current_group_id,
                        region_it->second.current_group_id};
            }
        } else {
            target = ensure_root_locked(key);
            if (!target) {
                return {RegionRegistrationStatus::MetadataExhausted, 0, 0};
            }
        }

        if (fixed_six_mode_locked()) {
            const auto target_it = groups_.find(target);
            if (target_it == groups_.end()) {
                return {RegionRegistrationStatus::MetadataExhausted, 0,
                        region_it->second.current_group_id};
            }
            if (!target_it->second.snapshot.capacity_group) {
                target = ensure_fixed_six_group_locked(
                    target_it->second.snapshot.root_group_id,
                    kFixedSixBootstrapClass);
                if (!target)
                    return {RegionRegistrationStatus::MetadataExhausted, 0,
                            region_it->second.current_group_id};
            }
        }

        RegionState &region = region_it->second;
        const GroupId previous_group = region.current_group_id;
        const RootGroupId previous_root = region.root_group_id;
        const InitialGroupKey previous_key = region.initial_key;
        if (previous_group == target && previous_key == key) {
            return {RegionRegistrationStatus::AlreadyRegistered, target,
                    previous_group};
        }

        auto previous_group_it = groups_.find(previous_group);
        if (previous_group_it != groups_.end()) {
            previous_group_it->second.members.erase(token);
        }
        auto target_group_it = groups_.find(target);
        if (target_group_it == groups_.end()) {
            // ensure_root_locked/canonical_group_locked above should make
            // this unreachable; leave the original membership intact if a
            // future implementation changes that invariant.
            if (previous_group_it != groups_.end())
                previous_group_it->second.members.insert(token);
            return {RegionRegistrationStatus::MetadataExhausted, 0,
                    previous_group};
        }
        target_group_it->second.members.insert(token);

        region.initial_key = key;
        region.root_group_id = target_group_it->second.snapshot.root_group_id;
        region.current_group_id = target;

        // An explicit rebind starts a new placement decision.  Do not let a
        // dirty tier observed under the prior placement silently carry across
        // that explicit user/runtime choice.
        capacity_last_dirty_class_.erase(token);

        RegionRebindRecord record;
        record.token = token;
        record.previous_initial_key = previous_key;
        record.target_initial_key = key;
        record.previous_root_group_id = previous_root;
        record.target_root_group_id = region.root_group_id;
        record.previous_group_id = previous_group;
        record.target_group_id = target;
        rebinds_.push_back(record);

        // Keep the legacy witness invalidation conservative.  The persistent
        // policy does not consult this map, while old merge mode must not
        // retain a witness over a membership mutation.
        reset_merge_stability_for_group_locked(previous_group);
        reset_merge_stability_for_group_locked(target);
        return {RegionRegistrationStatus::Rebound, target, previous_group};
    }

    bool unregister_region(const RegionToken &token) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = regions_.find(token);
        if (it == regions_.end()) {
            return false;
        }
        const GroupId previous_group_id = it->second.current_group_id;
        auto group_it = groups_.find(previous_group_id);
        if (group_it != groups_.end()) {
            group_it->second.members.erase(token);
        }
        regions_.erase(it);
        capacity_last_dirty_class_.erase(token);
        reset_merge_stability_for_group_locked(previous_group_id);
        return true;
    }

    std::optional<GroupId> current_group(const RegionToken &token) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = regions_.find(token);
        if (it == regions_.end()) {
            return std::nullopt;
        }
        return it->second.current_group_id;
    }

    std::optional<RegionSnapshot> region_snapshot(
        const RegionToken &token) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = regions_.find(token);
        if (it == regions_.end()) {
            return std::nullopt;
        }
        RegionSnapshot result;
        result.token = token;
        result.initial_key = it->second.initial_key;
        result.root_group_id = it->second.root_group_id;
        result.current_group_id = it->second.current_group_id;
        return result;
    }

    // A call carries the complete profiles observed at one boundary.  The
    // registry groups those profiles by each region's current group before any
    // rebinding, so all child creation and routing are one atomic metadata
    // operation.  A given (boundary,parent) pair is idempotent.
    BoundaryResult refine_boundary(
        std::uint64_t boundary,
        const std::vector<RegionObservation> &observations) {
        std::lock_guard<std::mutex> lock(mutex_);

        BoundaryResult result;
        result.boundary = boundary;
        result.observation_count = observations.size();
        if (processed_boundaries_.find(boundary) != processed_boundaries_.end()) {
            result.already_processed = true;
            return result;
        }
        // Do not let stale profiles earn a stability witness or mutate routing.
        // Previously processed boundaries remain idempotent, even if replayed.
        if ((config_.enable_merge || config_.persistent_range.enabled ||
             config_.capacity_grouping.enabled) &&
            latest_accepted_boundary_.has_value() &&
            boundary <= *latest_accepted_boundary_) {
            throw std::invalid_argument("merge boundaries must increase monotonically");
        }
        if (config_.capacity_grouping.enabled) {
            // Reject malformed capacity arithmetic before any boundary or
            // routing state is mutated.  Legacy and persistent-range modes
            // retain their historical validation paths below.
            validate_capacity_boundary_locked(observations);
            if (config_.capacity_grouping.dynamic_split_merge)
                validate_dynamic_capacity_input_locked(observations);
        }
        processed_boundaries_.insert(boundary);
        latest_accepted_boundary_ = boundary;

        // Capacity mode is a complete policy alternative.  Return before
        // preparing legacy classifications so lifetime evidence, intrinsic
        // thresholds, persistent-range state, and legacy merge logs cannot
        // influence capacity routing or counters.
        if (config_.capacity_grouping.enabled) {
            return refine_capacity_boundary_locked(boundary, observations,
                                                   std::move(result));
        }

        struct PreparedObservation {
            RegionToken token;
            RegionProfile profile;
            IntrinsicClassification classification;
        };
        std::map<GroupId, std::vector<PreparedObservation>> by_group;
        std::unordered_map<RegionToken, PreparedObservation, RegionTokenHash>
            prepared_by_token;
        std::unordered_set<RegionToken, RegionTokenHash> seen;
        std::unordered_set<RegionToken, RegionTokenHash> duplicate_tokens;
        for (const auto &observation : observations) {
            auto region_it = regions_.find(observation.token);
            if (region_it == regions_.end()) {
                ++result.ignored_observation_count;
                ++ignored_observation_count_;
                continue;
            }
            if (!seen.insert(observation.token).second) {
                duplicate_tokens.insert(observation.token);
                ++result.duplicate_observation_count;
                ++duplicate_observation_count_;
                continue;
            }
            ++result.accepted_observation_count;
            PreparedObservation prepared{observation.token, observation.profile,
                                          IntrinsicClassification{}};
            // Do not even invoke the legacy classifier in persistent mode:
            // its labels and evidence gates are not inputs to the range
            // policy, and a range-only configuration need not provide them.
            if (!config_.persistent_range.enabled) {
                prepared.classification = intrinsic_behavior_signature(
                    observation.profile, config_.classifier);
            }
            by_group[region_it->second.current_group_id].push_back(prepared);
            prepared_by_token.emplace(observation.token, std::move(prepared));
        }

        // Successful splits are excluded from merge consideration below.  In
        // particular, a freshly-created child must not be merged back into a
        // sibling using the same boundary's observations.
        std::set<GroupId> split_touched_groups;

        if (config_.persistent_range.enabled) {
            // The persistent policy intentionally bypasses all historical
            // complete-window/classifier/stability gates below.  It uses one
            // deterministic streaming partition per current group and the
            // same extrema predicate again for current-group merge pairing.
            struct PersistentPartition {
                std::vector<RegionToken> tokens;
                detail::persistent_range_summary summary;
            };
            struct PersistentGroupEvidence {
                GroupId group_id = 0;
                RootGroupId root_group_id = 0;
                InitialGroupKey initial_key;
                std::vector<RegionToken> member_tokens;
                std::vector<RegionProfile> observed_profiles;
                detail::persistent_range_summary summary;
                std::size_t occupied_region_count = 0;
                std::size_t idle_region_count = 0;
                std::size_t unknown_region_count = 0;
            };

            const auto has_known_metric = [](const RegionProfile &profile) {
                for (std::size_t index = 0; index < 3; ++index) {
                    detail::persistent_metric_value value;
                    if (detail::persistent_metric_known(
                            profile,
                            static_cast<PersistentRangeMetric>(index),
                            value)) {
                        return true;
                    }
                }
                return false;
            };

            const auto member_tokens_for =
                [](const GroupState &state) {
                    std::vector<RegionToken> tokens(state.members.begin(),
                                                    state.members.end());
                    std::sort(tokens.begin(), tokens.end());
                    return tokens;
                };

            const auto make_evidence = [&](GroupId group_id) {
                PersistentGroupEvidence evidence;
                evidence.group_id = group_id;
                const auto group_it = groups_.find(group_id);
                if (group_it == groups_.end()) return evidence;
                const auto &state = group_it->second;
                evidence.root_group_id = state.snapshot.root_group_id;
                evidence.initial_key = state.snapshot.initial_key;
                evidence.member_tokens = member_tokens_for(state);
                for (const auto &token : evidence.member_tokens) {
                    const auto prepared_it = prepared_by_token.find(token);
                    if (prepared_it == prepared_by_token.end() ||
                        duplicate_tokens.find(token) != duplicate_tokens.end()) {
                        ++evidence.unknown_region_count;
                        continue;
                    }
                    const auto &profile = prepared_it->second.profile;
                    if (!persistent_profile_occupied(profile)) {
                        if (profile.occupancy_known &&
                            profile.live_objects_at_end == 0) {
                            ++evidence.idle_region_count;
                        } else {
                            ++evidence.unknown_region_count;
                        }
                        continue;
                    }
                    ++evidence.occupied_region_count;
                    if (!has_known_metric(profile)) {
                        ++evidence.unknown_region_count;
                        continue;
                    }
                    detail::persistent_summary_add(evidence.summary, profile);
                    evidence.observed_profiles.push_back(profile);
                }
                return evidence;
            };

            // First split only profiles with authoritative, nonzero
            // occupancy and at least one known metric.  Idle and unknown
            // members stay in their parent and therefore remain persistent
            // members of whatever later merge absorbs that parent.
            for (auto &[parent_id, members] : by_group) {
                SplitOperation operation;
                operation.boundary = boundary;
                operation.parent_group_id = parent_id;
                auto parent_it = groups_.find(parent_id);
                if (parent_it == groups_.end()) continue;
                operation.root_group_id =
                    parent_it->second.snapshot.root_group_id;

                const auto boundary_parent = std::make_pair(boundary, parent_id);
                if (!processed_boundary_parents_.insert(boundary_parent).second) {
                    operation.status = SplitStatus::BoundaryAlreadyProcessed;
                    result.operations.push_back(std::move(operation));
                    continue;
                }

                std::sort(members.begin(), members.end(),
                          [](const PreparedObservation &lhs,
                             const PreparedObservation &rhs) {
                              return lhs.token < rhs.token;
                          });
                std::vector<PersistentPartition> partitions;
                std::size_t occupied_count = 0;
                std::size_t idle_count = 0;
                std::size_t unknown_count = 0;
                for (const auto &member : members) {
                    if (!persistent_profile_occupied(member.profile)) {
                        if (member.profile.occupancy_known &&
                            member.profile.live_objects_at_end == 0) {
                            ++idle_count;
                        } else {
                            ++unknown_count;
                        }
                        continue;
                    }
                    ++occupied_count;
                    if (!has_known_metric(member.profile)) {
                        ++unknown_count;
                        continue;
                    }

                    bool appended = false;
                    for (auto &partition : partitions) {
                        auto trial = partition.summary;
                        detail::persistent_range_summary candidate_summary;
                        detail::persistent_summary_add(candidate_summary,
                                                       member.profile);
                        detail::persistent_summary_merge(trial,
                                                         candidate_summary);
                        const auto compatibility =
                            detail::persistent_summary_compatibility(
                                trial, config_.persistent_range);
                        // A singleton is valid by definition; adding a
                        // profile is valid whenever no known shared range
                        // violates its bound.  Disjoint metrics are unknown
                        // for splitting and therefore remain together.
                        if (compatibility.compatible) {
                            partition.tokens.push_back(member.token);
                            partition.summary = std::move(trial);
                            appended = true;
                            break;
                        }
                    }
                    if (!appended) {
                        PersistentPartition partition;
                        partition.tokens.push_back(member.token);
                        detail::persistent_summary_add(partition.summary,
                                                       member.profile);
                        partitions.push_back(std::move(partition));
                    }
                }

                operation.qualified_region_count = occupied_count;
                operation.unclassified_region_count =
                    members.size() >= occupied_count
                        ? members.size() - occupied_count
                        : 0;
                if (partitions.size() < 2) {
                    operation.status = SplitStatus::NoConflict;
                    result.operations.push_back(std::move(operation));
                    continue;
                }

                const std::size_t missing_children = partitions.size();
                if (!has_capacity(groups_.size(), config_.max_groups,
                                  missing_children)) {
                    operation.status = SplitStatus::MetadataExhausted;
                    ++failed_refinement_event_count_;
                    SplitRecord record;
                    record.boundary = boundary;
                    record.root_group_id = operation.root_group_id;
                    record.parent_group_id = parent_id;
                    record.parent_depth = parent_it->second.snapshot.depth;
                    record.successful = false;
                    record.failure_reason = "metadata_exhausted";
                    record.qualified_region_count = occupied_count;
                    record.persistent_range_policy = true;
                    record.occupied_region_count = occupied_count;
                    record.idle_region_count = idle_count;
                    record.unknown_region_count = unknown_count;
                    splits_.push_back(record);
                    result.operations.push_back(std::move(operation));
                    continue;
                }

                std::vector<GroupId> child_ids;
                child_ids.reserve(partitions.size());
                for (std::size_t index = 0; index < partitions.size();
                     ++index) {
                    const GroupId child_id = allocate_group_id_locked();
                    if (child_id == 0) {
                        // The configured max_groups preflight normally makes
                        // this impossible.  Keep the operation atomic if the
                        // monotonically increasing ID ever wraps.
                        for (const auto allocated : child_ids) {
                            groups_.erase(allocated);
                            --child_group_count_;
                        }
                        operation.status = SplitStatus::MetadataExhausted;
                        ++failed_refinement_event_count_;
                        result.operations.push_back(std::move(operation));
                        child_ids.clear();
                        break;
                    }
                    GroupState child;
                    child.snapshot.id = child_id;
                    child.snapshot.root_group_id = operation.root_group_id;
                    child.snapshot.parent_group_id = parent_id;
                    child.snapshot.depth = parent_it->second.snapshot.depth + 1;
                    child.snapshot.initial_key =
                        parent_it->second.snapshot.initial_key;
                    // A persistent child has no legacy creation label.  The
                    // policy decision is stored separately below.
                    child.snapshot.has_signature = false;
                    groups_.emplace(child_id, std::move(child));
                    parent_it->second.historical_child_group_ids.push_back(
                        child_id);
                    child_ids.push_back(child_id);
                    ++child_group_count_;
                }
                if (child_ids.size() != partitions.size()) {
                    operation.status = SplitStatus::MetadataExhausted;
                    result.operations.push_back(std::move(operation));
                    continue;
                }

                const auto parent_tokens =
                    member_tokens_for(parent_it->second);
                operation.status = SplitStatus::SplitSucceeded;
                operation.child_group_ids = child_ids;
                split_touched_groups.insert(parent_id);
                for (const auto child_id : child_ids)
                    split_touched_groups.insert(child_id);
                for (std::size_t index = 0; index < partitions.size();
                     ++index) {
                    for (const auto &token : partitions[index].tokens) {
                        auto region_it = regions_.find(token);
                        if (region_it == regions_.end() ||
                            region_it->second.current_group_id != parent_id) {
                            continue;
                        }
                        parent_it->second.members.erase(token);
                        groups_.at(child_ids[index]).members.insert(token);
                        region_it->second.current_group_id = child_ids[index];
                    }
                }

                ++refinement_event_count_;
                ++root_event_counts_[operation.root_group_id];
                refined_roots_.insert(operation.root_group_id);

                SplitRecord record;
                record.boundary = boundary;
                record.root_group_id = operation.root_group_id;
                record.parent_group_id = parent_id;
                record.parent_depth = parent_it->second.snapshot.depth;
                record.successful = true;
                record.qualified_region_count = occupied_count;
                record.child_group_ids = child_ids;
                record.persistent_range_policy = true;
                record.occupied_region_count = occupied_count;
                record.idle_region_count = idle_count;
                record.unknown_region_count = unknown_count;
                splits_.push_back(record);
                PersistentRangeDecisionRecord decision;
                decision.boundary = boundary;
                decision.kind = PersistentRangeDecisionKind::Split;
                decision.root_group_id = operation.root_group_id;
                decision.source_group_id = parent_id;
                decision.child_group_ids = child_ids;
                decision.source_member_tokens = parent_tokens;
                decision.occupied_region_count = occupied_count;
                decision.idle_region_count = idle_count;
                decision.unknown_region_count = unknown_count;
                persistent_decisions_.push_back(decision);
                result.persistent_decisions.push_back(std::move(decision));
                result.operations.push_back(std::move(operation));
            }

            // Rebuild evidence after splitting.  This ensures source and
            // target token lists below describe the exact current routing
            // membership.  Persistent range compatibility itself prevents a
            // just-created incompatible child from being merged straight
            // back; a split child may still merge with another compatible
            // current group at this same boundary.
            for (const auto &[group_id, state] : groups_) {
                if (state.members.empty()) continue;
                MergeCoverageRecord coverage;
                coverage.boundary = boundary;
                coverage.group_id = group_id;
                coverage.root_group_id = state.snapshot.root_group_id;
                coverage.member_count = state.members.size();
                coverage.split_touched =
                    split_touched_groups.find(group_id) !=
                    split_touched_groups.end();
                coverage.root_excluded =
                    !config_.persistent_range.allow_root_targets &&
                    group_id == state.snapshot.root_group_id;
                for (const auto &member_token : state.members) {
                    const auto prepared_it = prepared_by_token.find(member_token);
                    if (prepared_it == prepared_by_token.end() ||
                        duplicate_tokens.find(member_token) !=
                            duplicate_tokens.end()) {
                        ++coverage.missing_or_duplicate_count;
                        continue;
                    }
                    ++coverage.present_nonduplicate_count;
                    const auto &profile = prepared_it->second.profile;
                    if (profile.occupancy_known &&
                        profile.live_objects_at_end == 0) {
                        ++coverage.zero_exposure_count;
                    } else if (persistent_profile_occupied(profile) &&
                               has_known_metric(profile)) {
                        ++coverage.qualified_count;
                    } else {
                        ++coverage.other_unqualified_count;
                    }
                }
                merge_coverage_.push_back(coverage);
                result.merge_coverage.push_back(std::move(coverage));
            }

            if (!config_.enable_merge) return result;
            std::map<RootGroupId,
                     std::map<InitialGroupKey,
                              std::vector<PersistentGroupEvidence>>>
                grouped_evidence;
            for (const auto &[group_id, state] : groups_) {
                if (state.members.empty() || is_retired(state)) {
                    continue;
                }
                if (group_id == state.snapshot.root_group_id &&
                    !config_.persistent_range.allow_root_targets) {
                    continue;
                }
                auto evidence = make_evidence(group_id);
                const auto own = detail::persistent_summary_compatibility(
                    evidence.summary, config_.persistent_range);
                // A group with multiple occupied profiles that already
                // violates its internal range is not a merge candidate.  A
                // singleton (or an idle/unknown-only group) is not enough to
                // establish cross-group evidence and is filtered below.
                if (!own.compatible) {
                    continue;
                }
                grouped_evidence[evidence.root_group_id][evidence.initial_key]
                    .push_back(std::move(evidence));
            }

            for (auto &[root_id, by_key] : grouped_evidence) {
                (void)root_id;
                for (auto &[key, candidates] : by_key) {
                    (void)key;
                    std::sort(candidates.begin(), candidates.end(),
                              [](const PersistentGroupEvidence &lhs,
                                 const PersistentGroupEvidence &rhs) {
                                  return lhs.group_id < rhs.group_id;
                              });
                    struct MergeCluster {
                        std::vector<std::size_t> candidate_indexes;
                        detail::persistent_range_summary summary;
                    };
                    std::vector<MergeCluster> clusters;
                    for (std::size_t index = 0; index < candidates.size();
                         ++index) {
                        const auto &candidate = candidates[index];
                        if (candidate.summary.occupied_profile_count == 0) {
                            continue;
                        }
                        bool appended = false;
                        for (auto &cluster : clusters) {
                            const auto compatibility =
                                detail::persistent_summary_cross_compatibility(
                                    cluster.summary, candidate.summary,
                                    config_.persistent_range);
                            if (!compatibility.compatible) continue;
                            cluster.candidate_indexes.push_back(index);
                            detail::persistent_summary_merge(cluster.summary,
                                                             candidate.summary);
                            appended = true;
                            break;
                        }
                        if (!appended) {
                            MergeCluster cluster;
                            cluster.candidate_indexes.push_back(index);
                            cluster.summary = candidate.summary;
                            clusters.push_back(std::move(cluster));
                        }
                    }

                    for (const auto &cluster : clusters) {
                        if (cluster.candidate_indexes.size() < 2) continue;
                        GroupId target_id = 0;
                        for (const auto index : cluster.candidate_indexes) {
                            const auto candidate_id = candidates[index].group_id;
                            if (candidate_id == candidates[index].root_group_id &&
                                config_.persistent_range.allow_root_targets) {
                                target_id = candidate_id;
                                break;
                            }
                        }
                        if (target_id == 0) {
                            target_id = candidates[cluster.candidate_indexes[0]]
                                            .group_id;
                        }

                        for (const auto source_index : cluster.candidate_indexes) {
                            const auto source_id = candidates[source_index].group_id;
                            if (source_id == target_id) continue;
                            auto target_it = groups_.find(target_id);
                            auto source_it = groups_.find(source_id);
                            if (target_it == groups_.end() ||
                                source_it == groups_.end() ||
                                is_retired(target_it->second) ||
                                is_retired(source_it->second) ||
                                target_it->second.members.empty() ||
                                source_it->second.members.empty()) {
                                continue;
                            }

                            const auto target_tokens =
                                member_tokens_for(target_it->second);
                            const auto source_tokens =
                                member_tokens_for(source_it->second);
                            const auto target_evidence =
                                make_evidence(target_id);
                            const auto source_evidence =
                                make_evidence(source_id);
                            const auto compatibility =
                                detail::persistent_summary_cross_compatibility(
                                    target_evidence.summary,
                                    source_evidence.summary,
                                    config_.persistent_range);
                            if (!compatibility.compatible) continue;

                            RegionProfile pooled_profile;
                            bool have_pooled_profile = false;
                            const auto add_profile = [&](const RegionProfile &p) {
                                if (!have_pooled_profile) {
                                    pooled_profile = p;
                                    have_pooled_profile = true;
                                    return true;
                                }
                                return checked_add_profile(pooled_profile, p,
                                                           pooled_profile);
                            };
                            for (const auto &profile :
                                 target_evidence.observed_profiles) {
                                if (!add_profile(profile)) {
                                    ++merge_overflow_count_;
                                    have_pooled_profile = false;
                                    break;
                                }
                            }
                            if (have_pooled_profile) {
                                for (const auto &profile :
                                     source_evidence.observed_profiles) {
                                    if (!add_profile(profile)) {
                                        ++merge_overflow_count_;
                                        have_pooled_profile = false;
                                        break;
                                    }
                                }
                            }
                            if (!have_pooled_profile) {
                                // The routing decision must never proceed
                                // with a fabricated zero pooled profile when
                                // its audit sufficient statistics overflow.
                                continue;
                            }

                            const std::size_t moved = source_tokens.size();
                            for (const auto &token : source_tokens) {
                                target_it->second.members.insert(token);
                                auto region_it = regions_.find(token);
                                if (region_it != regions_.end())
                                    region_it->second.current_group_id =
                                        target_id;
                            }
                            source_it->second.members.clear();
                            source_it->second.snapshot.merged_into_group_id =
                                target_id;
                            source_it->second.snapshot.merged_at_boundary =
                                boundary;
                            source_it->second.snapshot.retired = true;
                            source_it->second.snapshot.is_active_region_group =
                                false;

                            MergeRecord record;
                            record.boundary = boundary;
                            record.root_group_id =
                                target_it->second.snapshot.root_group_id;
                            record.source_group_id = source_id;
                            record.target_group_id = target_id;
                            record.source = source_id;
                            record.target = target_id;
                            record.signature =
                                IntrinsicSignature::Unclassified;
                            record.moved_region_count = moved;
                            record.pooled_profile = pooled_profile;
                            record.pooled_classification =
                                IntrinsicClassification{};
                            record.persistent_range_policy = true;
                            record.persistent_has_common_metric =
                                compatibility.has_common_metric;
                            record.persistent_metric_ranges_valid =
                                compatibility.metric_ranges_valid;
                            record.persistent_occupied_region_count =
                                target_evidence.occupied_region_count +
                                source_evidence.occupied_region_count;
                            record.persistent_idle_region_count =
                                target_evidence.idle_region_count +
                                source_evidence.idle_region_count;
                            record.persistent_unknown_region_count =
                                target_evidence.unknown_region_count +
                                source_evidence.unknown_region_count;
                            record.persistent_source_member_tokens =
                                source_tokens;
                            record.persistent_target_member_tokens =
                                target_tokens;
                            merges_.push_back(record);
                            result.merges.push_back(record);

                            PersistentRangeDecisionRecord decision;
                            decision.boundary = boundary;
                            decision.kind = PersistentRangeDecisionKind::Merge;
                            decision.root_group_id = record.root_group_id;
                            decision.source_group_id = source_id;
                            decision.target_group_id = target_id;
                            decision.source_member_tokens = source_tokens;
                            decision.target_member_tokens = target_tokens;
                            decision.occupied_region_count =
                                record.persistent_occupied_region_count;
                            decision.idle_region_count =
                                record.persistent_idle_region_count;
                            decision.unknown_region_count =
                                record.persistent_unknown_region_count;
                            decision.has_common_metric =
                                compatibility.has_common_metric;
                            decision.compatible = compatibility.compatible;
                            decision.comparable_metrics =
                                compatibility.comparable_metrics;
                            decision.metric_ranges_valid =
                                compatibility.metric_ranges_valid;
                            persistent_decisions_.push_back(decision);
                            result.persistent_decisions.push_back(
                                std::move(decision));

                            ++merged_group_count_;
                            ++merge_event_count_;
                            moved_region_count_ += moved;
                        }
                    }
                }
            }
            return result;
        }

        for (auto &[parent_id, members] : by_group) {
            SplitOperation operation;
            operation.boundary = boundary;
            operation.parent_group_id = parent_id;
            operation.root_group_id = groups_.at(parent_id).snapshot.root_group_id;

            const auto boundary_parent = std::make_pair(boundary, parent_id);
            if (!processed_boundary_parents_.insert(boundary_parent).second) {
                operation.status = SplitStatus::BoundaryAlreadyProcessed;
                result.operations.push_back(std::move(operation));
                continue;
            }

            std::map<IntrinsicSignature, std::vector<RegionToken>> partitions;
            for (const auto &member : members) {
                if (member.classification.evidence_qualified) {
                    partitions[member.classification.signature].push_back(
                        member.token);
                    ++operation.qualified_region_count;
                } else {
                    ++operation.unclassified_region_count;
                }
            }
            if (partitions.size() < 2) {
                operation.status = SplitStatus::NoConflict;
                result.operations.push_back(std::move(operation));
                continue;
            }

            for (const auto &[signature, tokens] : partitions) {
                (void)tokens;
                operation.conflicting_signatures.push_back(signature);
            }

            GroupState &parent = groups_.at(parent_id);
            std::size_t missing_children = 0;
            for (const auto &[signature, tokens] : partitions) {
                (void)tokens;
                auto child_it = parent.children.find(signature);
                if (child_it == parent.children.end() ||
                    is_retired(groups_.at(child_it->second))) {
                    ++missing_children;
                }
            }

            if (!has_capacity(groups_.size(), config_.max_groups,
                              missing_children)) {
                operation.status = SplitStatus::MetadataExhausted;
                ++failed_refinement_event_count_;
                SplitRecord record;
                record.boundary = boundary;
                record.root_group_id = operation.root_group_id;
                record.parent_group_id = parent_id;
                record.parent_depth = parent.snapshot.depth;
                record.successful = false;
                record.failure_reason = "metadata_exhausted";
                record.qualified_region_count = operation.qualified_region_count;
                record.conflicting_signatures = operation.conflicting_signatures;
                splits_.push_back(std::move(record));
                result.operations.push_back(std::move(operation));
                continue;
            }

            std::map<IntrinsicSignature, GroupId> child_ids;
            struct NewlyCreatedChild {
                IntrinsicSignature signature = IntrinsicSignature::Unclassified;
                GroupId child_id = 0;
                GroupId replaced_child_id = 0;
            };
            std::vector<NewlyCreatedChild> newly_created_children;
            for (const auto &[signature, tokens] : partitions) {
                (void)tokens;
                auto child_it = parent.children.find(signature);
                if (child_it != parent.children.end() &&
                    !is_retired(groups_.at(child_it->second))) {
                    child_ids.emplace(signature, child_it->second);
                    continue;
                }

                const GroupId child_id = allocate_group_id_locked();
                if (child_id == 0) {
                    // The overflow case is practically unreachable, but keep
                    // it transactional just like finite metadata exhaustion.
                    operation.status = SplitStatus::MetadataExhausted;
                    ++failed_refinement_event_count_;
                    SplitRecord record;
                    record.boundary = boundary;
                    record.root_group_id = operation.root_group_id;
                    record.parent_group_id = parent_id;
                    record.parent_depth = parent.snapshot.depth;
                    record.successful = false;
                    record.failure_reason = "group_id_exhausted";
                    record.qualified_region_count =
                        operation.qualified_region_count;
                    record.conflicting_signatures =
                        operation.conflicting_signatures;
                    splits_.push_back(std::move(record));
                    result.operations.push_back(std::move(operation));
                    break;
                }

                GroupState child;
                child.snapshot.id = child_id;
                child.snapshot.root_group_id = parent.snapshot.root_group_id;
                child.snapshot.parent_group_id = parent_id;
                child.snapshot.depth = parent.snapshot.depth + 1;
                child.snapshot.initial_key = parent.snapshot.initial_key;
                child.snapshot.has_signature = true;
                child.snapshot.signature = signature;
                const GroupId replaced_child_id =
                    child_it != parent.children.end() ? child_it->second : 0;
                groups_.emplace(child_id, std::move(child));
                if (child_it != parent.children.end()) {
                    // The old child remains historical; only the current
                    // routing pointer is replaced.
                    child_it->second = child_id;
                } else {
                    parent.children.emplace(signature, child_id);
                }
                parent.historical_child_group_ids.push_back(child_id);
                child_ids.emplace(signature, child_id);
                newly_created_children.push_back(
                    {signature, child_id, replaced_child_id});
                ++child_group_count_;
            }

            // Only the ID-overflow branch above can leave child_ids empty after
            // preflight; roll back any children allocated in that impossible
            // branch so exhaustion is conservative even then.
            if (child_ids.size() != partitions.size()) {
                for (const auto &created : newly_created_children) {
                    auto child_it = parent.children.find(created.signature);
                    if (child_it != parent.children.end() &&
                        child_it->second == created.child_id) {
                        if (created.replaced_child_id != 0) {
                            child_it->second = created.replaced_child_id;
                        } else {
                            parent.children.erase(child_it);
                        }
                    }
                    parent.historical_child_group_ids.erase(
                        std::remove(parent.historical_child_group_ids.begin(),
                                    parent.historical_child_group_ids.end(),
                                    created.child_id),
                        parent.historical_child_group_ids.end());
                    groups_.erase(created.child_id);
                    --child_group_count_;
                }
                continue;
            }

            operation.status = SplitStatus::SplitSucceeded;
            split_touched_groups.insert(parent_id);
            for (const auto &[signature, child_id] : child_ids) {
                operation.child_group_ids.push_back(child_id);
                split_touched_groups.insert(child_id);
                for (const auto &token : partitions.at(signature)) {
                    auto region_it = regions_.find(token);
                    if (region_it == regions_.end() ||
                        region_it->second.current_group_id != parent_id) {
                        continue;
                    }
                    parent.members.erase(token);
                    groups_.at(child_id).members.insert(token);
                    region_it->second.current_group_id = child_id;
                }
            }

            ++refinement_event_count_;
            auto root_events = ++root_event_counts_[operation.root_group_id];
            (void)root_events;
            refined_roots_.insert(operation.root_group_id);
            reset_merge_stability_for_group_locked(parent_id);
            for (const auto &[signature, child_id] : child_ids) {
                (void)signature;
                reset_merge_stability_for_group_locked(child_id);
            }

            SplitRecord record;
            record.boundary = boundary;
            record.root_group_id = operation.root_group_id;
            record.parent_group_id = parent_id;
            record.parent_depth = parent.snapshot.depth;
            record.successful = true;
            record.qualified_region_count = operation.qualified_region_count;
            record.conflicting_signatures = operation.conflicting_signatures;
            record.child_group_ids = operation.child_group_ids;
            splits_.push_back(std::move(record));
            result.operations.push_back(std::move(operation));
        }

        if (config_.record_merge_coverage) {
            // Capture the exact post-split/pre-merge census.  Iterate every
            // current non-empty group, rather than only groups represented by
            // observations, so a missing group member is visible as missing
            // evidence instead of a fabricated zero-valued profile.
            for (const auto &[group_id, state] : groups_) {
                if (state.members.empty()) {
                    continue;
                }

                MergeCoverageRecord coverage;
                coverage.boundary = boundary;
                coverage.group_id = group_id;
                coverage.root_group_id = state.snapshot.root_group_id;
                coverage.member_count = state.members.size();
                coverage.split_touched =
                    split_touched_groups.find(group_id) !=
                    split_touched_groups.end();
                coverage.root_excluded =
                    group_id == state.snapshot.root_group_id;

                for (const auto &member_token : state.members) {
                    auto prepared_it = prepared_by_token.find(member_token);
                    if (prepared_it == prepared_by_token.end() ||
                        duplicate_tokens.find(member_token) !=
                            duplicate_tokens.end()) {
                        ++coverage.missing_or_duplicate_count;
                        continue;
                    }

                    ++coverage.present_nonduplicate_count;
                    const auto &prepared = prepared_it->second;
                    if (!prepared.profile.complete_window) {
                        ++coverage.incomplete_count;
                    } else if (prepared.classification.lifecycle_events <
                               config_.classifier.evidence
                                   .min_lifecycle_events) {
                        ++coverage.below_lifecycle_floor_count;
                    } else if (prepared.profile.live_object_exposure == 0) {
                        ++coverage.zero_exposure_count;
                    } else if (prepared.classification.evidence_qualified) {
                        ++coverage.qualified_count;
                    } else {
                        ++coverage.other_unqualified_count;
                    }
                }

                merge_coverage_.push_back(coverage);
                result.merge_coverage.push_back(std::move(coverage));
            }
        }

        if (config_.enable_merge) {
            std::vector<MergeCandidate> candidates;
            std::set<GroupId> observed_current_groups;
            for (const auto &[member_token, prepared] : prepared_by_token) {
                (void)prepared;
                auto region_it = regions_.find(member_token);
                if (region_it != regions_.end()) {
                    observed_current_groups.insert(
                        region_it->second.current_group_id);
                }
            }

            // Any witness for a group that is not represented by this
            // boundary's current observations is no longer consecutive.
            for (auto stability_it = merge_stability_.begin();
                 stability_it != merge_stability_.end();) {
                if (observed_current_groups.find(stability_it->first) ==
                    observed_current_groups.end()) {
                    stability_it = merge_stability_.erase(stability_it);
                } else {
                    ++stability_it;
                }
            }
            candidates.reserve(observed_current_groups.size());

            // Build candidates from current routing membership after all
            // splits in this boundary.  This second pass is intentional: a
            // split-created/rebound group is excluded, and every other group
            // is checked against the complete current membership set rather
            // than just the observations that happened to share its old
            // parent before the split.
            for (const auto group_id : observed_current_groups) {
                auto group_it = groups_.find(group_id);
                if (group_it == groups_.end()) {
                    continue;
                }
                const auto &state = group_it->second;
                if (group_id == state.snapshot.root_group_id ||
                    is_retired(state) || state.members.empty()) {
                    continue;
                }
                if (split_touched_groups.find(group_id) !=
                    split_touched_groups.end()) {
                    reset_merge_stability_for_group_locked(group_id);
                    continue;
                }

                MergeCandidate candidate;
                candidate.group_id = group_id;
                candidate.root_group_id = state.snapshot.root_group_id;
                candidate.initial_key = state.snapshot.initial_key;
                candidate.member_tokens.assign(state.members.begin(),
                                               state.members.end());
                std::sort(candidate.member_tokens.begin(),
                          candidate.member_tokens.end());

                bool has_signature = false;
                bool missing_evidence = false;
                bool mixed_label = false;
                bool overflow = false;
                IntrinsicSignature signature =
                    IntrinsicSignature::Unclassified;
                RegionProfile aggregate;
                bool have_aggregate = false;

                for (const auto &member_token : candidate.member_tokens) {
                    auto prepared_it = prepared_by_token.find(member_token);
                    if (prepared_it == prepared_by_token.end() ||
                        duplicate_tokens.find(member_token) !=
                            duplicate_tokens.end()) {
                        missing_evidence = true;
                        continue;
                    }

                    const auto &prepared = prepared_it->second;
                    if (!prepared.classification.complete_window ||
                        !prepared.classification.evidence_qualified) {
                        missing_evidence = true;
                        continue;
                    }

                    if (!has_signature) {
                        signature = prepared.classification.signature;
                        has_signature = true;
                    } else if (signature != prepared.classification.signature) {
                        mixed_label = true;
                    }

                    if (!have_aggregate) {
                        if (!profile_derived_totals_fit(prepared.profile)) {
                            overflow = true;
                            continue;
                        }
                        aggregate = prepared.profile;
                        aggregate.complete_window = true;
                        have_aggregate = true;
                    } else if (!checked_add_profile(aggregate,
                                                    prepared.profile,
                                                    aggregate)) {
                        overflow = true;
                    }
                }

                if (overflow) {
                    ++merge_overflow_count_;
                    reset_merge_stability_for_group_locked(group_id);
                    continue;
                }
                if (missing_evidence || !has_signature || !have_aggregate) {
                    ++merge_missing_evidence_count_;
                    reset_merge_stability_for_group_locked(group_id);
                    continue;
                }
                if (mixed_label) {
                    ++merge_mixed_label_count_;
                    reset_merge_stability_for_group_locked(group_id);
                    continue;
                }

                const auto aggregate_classification =
                    intrinsic_behavior_signature(aggregate,
                                                 config_.classifier);
                if (!aggregate_classification.complete_window ||
                    !aggregate_classification.evidence_qualified ||
                    aggregate_classification.signature != signature) {
                    ++merge_aggregate_changed_count_;
                    reset_merge_stability_for_group_locked(group_id);
                    continue;
                }

                auto stability_it = merge_stability_.find(group_id);
                if (stability_it == merge_stability_.end() ||
                    stability_it->second.signature != signature ||
                    boundary <= stability_it->second.last_boundary ||
                    boundary - stability_it->second.last_boundary !=
                        config_.merge_boundary_stride ||
                    !same_members(stability_it->second.member_tokens,
                                  candidate.member_tokens)) {
                    MergeStabilityState stability;
                    stability.signature = signature;
                    stability.member_tokens = candidate.member_tokens;
                    stability.consecutive_windows = 1;
                    stability.last_boundary = boundary;
                    stability_it = merge_stability_
                                       .insert_or_assign(group_id,
                                                         std::move(stability))
                                       .first;
                } else {
                    if (stability_it->second.consecutive_windows !=
                        std::numeric_limits<std::size_t>::max()) {
                        ++stability_it->second.consecutive_windows;
                    }
                    stability_it->second.last_boundary = boundary;
                }

                candidate.signature = signature;
                candidate.pooled_profile = aggregate;
                candidate.pooled_classification = aggregate_classification;
                candidate.stable =
                    stability_it->second.consecutive_windows >=
                    config_.merge_min_stable_windows;
                if (!candidate.stable) {
                    ++merge_stability_wait_count_;
                }
                candidates.push_back(std::move(candidate));
            }

            // A sorted first-fit pass gives a deterministic smallest target
            // while considering only current eligible candidates.  Pairing is
            // deliberately disjoint: a target that just absorbed a source
            // must earn a fresh stability witness before another merge.
            std::sort(candidates.begin(), candidates.end(),
                      [](const MergeCandidate &lhs,
                         const MergeCandidate &rhs) {
                          if (lhs.root_group_id != rhs.root_group_id) {
                              return lhs.root_group_id < rhs.root_group_id;
                          }
                          if (!(lhs.initial_key == rhs.initial_key)) {
                              return lhs.initial_key < rhs.initial_key;
                          }
                          if (lhs.signature != rhs.signature) {
                              return static_cast<std::uint8_t>(lhs.signature) <
                                     static_cast<std::uint8_t>(rhs.signature);
                          }
                          return lhs.group_id < rhs.group_id;
                      });

            std::vector<bool> consumed(candidates.size(), false);
            for (std::size_t index = 0; index < candidates.size(); ++index) {
                if (consumed[index] || !candidates[index].stable) {
                    continue;
                }

                // First-fit over unconsumed candidates keeps the smallest
                // available ID as target while allowing a failed pooled
                // profile with one source to try the next relevant source.
                // A target is consumed at most once in this boundary.
                for (std::size_t source_index = index + 1;
                     source_index < candidates.size(); ++source_index) {
                    if (consumed[source_index] ||
                        !candidates[source_index].stable) {
                        continue;
                    }
                    auto &target = candidates[index];
                    auto &source = candidates[source_index];
                    const bool compatible =
                        target.root_group_id == source.root_group_id &&
                        target.initial_key == source.initial_key &&
                        target.signature == source.signature;
                    if (!compatible) {
                        continue;
                    }

                    RegionProfile pooled_profile;
                    if (!checked_add_profile(target.pooled_profile,
                                             source.pooled_profile,
                                             pooled_profile)) {
                        ++merge_overflow_count_;
                        continue;
                    }
                    const auto pooled_classification =
                        intrinsic_behavior_signature(pooled_profile,
                                                     config_.classifier);
                    if (!pooled_classification.complete_window ||
                        !pooled_classification.evidence_qualified ||
                        pooled_classification.signature != target.signature) {
                        ++merge_aggregate_changed_count_;
                        continue;
                    }

                    target.pooled_profile = pooled_profile;
                    target.pooled_classification = pooled_classification;

                    auto target_it = groups_.find(target.group_id);
                    auto source_it = groups_.find(source.group_id);
                    if (target_it == groups_.end() ||
                        source_it == groups_.end() ||
                        is_retired(target_it->second) ||
                        is_retired(source_it->second) ||
                        target_it->second.members.empty() ||
                        source_it->second.members.empty()) {
                        continue;
                    }

                    // Group IDs are monotonically allocated and candidates
                    // are sorted by ID within a lineage, so the target is
                    // always the smaller ID.  No alias is installed:
                    // every region state is rewritten directly to target.
                    const std::size_t moved =
                        source_it->second.members.size();
                    std::vector<RegionToken> moved_tokens(
                        source_it->second.members.begin(),
                        source_it->second.members.end());
                    for (const auto &member_token : moved_tokens) {
                        target_it->second.members.insert(member_token);
                        auto region_it = regions_.find(member_token);
                        if (region_it != regions_.end()) {
                            region_it->second.current_group_id =
                                target.group_id;
                        }
                    }
                    source_it->second.members.clear();
                    source_it->second.snapshot.merged_into_group_id =
                        target.group_id;
                    source_it->second.snapshot.merged_at_boundary = boundary;
                    source_it->second.snapshot.retired = true;
                    source_it->second.snapshot.is_active_region_group = false;

                    MergeRecord record;
                    record.boundary = boundary;
                    record.root_group_id = target.root_group_id;
                    record.source_group_id = source.group_id;
                    record.target_group_id = target.group_id;
                    record.signature = target.signature;
                    record.moved_region_count = moved;
                    record.pooled_profile = target.pooled_profile;
                    record.pooled_classification =
                        target.pooled_classification;
                    record.source = record.source_group_id;
                    record.target = record.target_group_id;
                    merges_.push_back(record);
                    result.merges.push_back(record);

                    ++merged_group_count_;
                    ++merge_event_count_;
                    moved_region_count_ += moved;
                    reset_merge_stability_for_group_locked(target.group_id);
                    reset_merge_stability_for_group_locked(source.group_id);
                    consumed[index] = true;
                    consumed[source_index] = true;
                    break;
                }
            }
        }
        return result;
    }

    std::optional<GroupSnapshot> group_snapshot(GroupId id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = groups_.find(id);
        if (it == groups_.end()) {
            return std::nullopt;
        }
        GroupSnapshot result = it->second.snapshot;
        result.member_count = it->second.members.size();
        result.child_group_ids = it->second.historical_child_group_ids;
        result.is_active_region_group =
            !is_retired(it->second) && !it->second.members.empty();
        return result;
    }

    RegistrySnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        RegistrySnapshot result;
        result.config = config_;
        result.initial_group_count = roots_by_key_.size();
        result.historical_group_count = groups_.size();
        result.refined_initial_group_count = refined_roots_.size();
        result.refinement_event_count = refinement_event_count_;
        result.child_group_count = child_group_count_;
        result.failed_refinement_event_count = failed_refinement_event_count_;
        result.owner_mismatch_count = owner_mismatch_count_;
        result.ignored_observation_count = ignored_observation_count_;
        result.duplicate_observation_count = duplicate_observation_count_;
        result.merged_group_count = merged_group_count_;
        result.merge_event_count = merge_event_count_;
        result.moved_region_count = moved_region_count_;
        result.merge_missing_evidence_count = merge_missing_evidence_count_;
        result.merge_mixed_label_count = merge_mixed_label_count_;
        result.merge_aggregate_changed_count =
            merge_aggregate_changed_count_;
        result.merge_overflow_count = merge_overflow_count_;
        result.merge_stability_wait_count = merge_stability_wait_count_;
        result.capacity_boundary_count = capacity_boundary_count_;
        result.capacity_decision_count = capacity_decision_count_;
        result.capacity_group_count = capacity_group_count_;
        result.capacity_hot_candidate_count = capacity_hot_candidate_count_;
        result.capacity_hot_region_count = capacity_hot_region_count_;
        result.capacity_deferred_region_count =
            capacity_deferred_region_count_;
        result.capacity_moved_region_count = capacity_moved_region_count_;
        result.capacity_moved_group_count = capacity_moved_group_count_;
        result.capacity_idle_region_count = capacity_idle_region_count_;
        result.capacity_unknown_region_count = capacity_unknown_region_count_;
        result.capacity_overflow_count = capacity_overflow_count_;

        for (const auto &[id, state] : groups_) {
            (void)id;
            GroupSnapshot group = state.snapshot;
            group.member_count = state.members.size();
            group.child_group_ids = state.historical_child_group_ids;
            group.is_active_region_group =
                !is_retired(state) && !state.members.empty();
            if (group.is_active_region_group) {
                ++result.active_group_count;
            }
            result.max_refinement_depth =
                std::max(result.max_refinement_depth, group.depth);
            result.groups.push_back(std::move(group));
        }

        result.regions.reserve(regions_.size());
        for (const auto &[token, state] : regions_) {
            result.regions.push_back(
                RegionSnapshot{token, state.initial_key, state.root_group_id,
                               state.current_group_id});
        }
        std::sort(result.regions.begin(), result.regions.end(),
                  [](const RegionSnapshot &lhs, const RegionSnapshot &rhs) {
                      return lhs.token < rhs.token;
                  });

        result.splits = splits_;
        result.merges = merges_;
        result.merge_coverage = merge_coverage_;
        result.persistent_decisions = persistent_decisions_;
        result.rebinds = rebinds_;
        result.capacity_decisions = capacity_decisions_;
        result.capacity_boundaries = capacity_boundaries_;
        result.dynamic_capacity_decisions = dynamic_capacity_decisions_;
        for (const auto &[root, count] : root_event_counts_) {
            (void)root;
            result.max_refinements_per_initial_group =
                std::max(result.max_refinements_per_initial_group, count);
        }
        return result;
    }
};

using RegionGroupRegistry = BehaviorGroupRegistry;
using RegionRefinementRegistry = BehaviorGroupRegistry;

} // namespace refinement
} // namespace cache
} // namespace FarLib
