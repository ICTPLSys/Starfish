#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace FarLib::cache::design2::behavior_group {

// The behavior class is a dense six-way identity used for routing and
// accounting.  It is deliberately independent from ProtectionDecision:
// classification chooses a class, while policy chooses a recovery backend.
enum class BehaviorClass : std::uint8_t {
    ColdLow = 0,
    ColdMedium = 1,
    ColdHigh = 2,
    HotLow = 3,
    HotMedium = 4,
    HotHigh = 5,
};

inline constexpr std::size_t kBehaviorClassCount = 6;

constexpr std::uint8_t behavior_class_id(BehaviorClass value) noexcept {
    return static_cast<std::uint8_t>(value);
}

constexpr bool valid_behavior_class(BehaviorClass value) noexcept {
    return behavior_class_id(value) < kBehaviorClassCount;
}

constexpr bool valid_behavior_class_id(std::uint32_t value) noexcept {
    return value < kBehaviorClassCount;
}

constexpr std::optional<BehaviorClass> behavior_class_from_id(
    std::uint32_t value) noexcept {
    if (!valid_behavior_class_id(value)) return std::nullopt;
    return static_cast<BehaviorClass>(value);
}

constexpr const char *to_string(BehaviorClass value) noexcept {
    switch (value) {
    case BehaviorClass::ColdLow:
        return "cold_low";
    case BehaviorClass::ColdMedium:
        return "cold_medium";
    case BehaviorClass::ColdHigh:
        return "cold_high";
    case BehaviorClass::HotLow:
        return "hot_low";
    case BehaviorClass::HotMedium:
        return "hot_medium";
    case BehaviorClass::HotHigh:
        return "hot_high";
    }
    return "unknown";
}

// The policy consumes explicitly classified predicates.  Thresholds and
// locality interpretation belong to the caller/profiler; this header does
// not infer them from a BehaviorClass or from a local-resident mode.
struct ProtectionPredicates {
    bool transient = false;
    bool replication_budget_slack = false;
    bool frequent_lifecycle = false;
    bool stable_cold = false;
    bool large_object = false;
    bool low_fanout_sensitivity = false;
    bool batched_region_scan_profitable = false;
};

enum class ProtectionMethod : std::uint8_t {
    Replication = 0,
    ErasureCoding = 1,
};

enum class EcGranularity : std::uint8_t {
    NotApplicable = 0,
    Fine = 1,
    SizeClassPacked = 2,
    CoarsePacked = 3,
};

enum class ObjectLayout : std::uint8_t {
    Whole = 0,
    Split = 1,
};

struct ProtectionDecision {
    ProtectionMethod method = ProtectionMethod::ErasureCoding;
    EcGranularity ec_granularity = EcGranularity::SizeClassPacked;
    ObjectLayout object_layout = ObjectLayout::Whole;
    bool use_batched_region_scan = false;

    friend constexpr bool operator==(const ProtectionDecision &,
                                     const ProtectionDecision &) = default;
};

constexpr bool valid_protection_method(ProtectionMethod value) noexcept {
    return value == ProtectionMethod::Replication ||
           value == ProtectionMethod::ErasureCoding;
}

constexpr bool valid_ec_granularity(EcGranularity value) noexcept {
    return value == EcGranularity::NotApplicable ||
           value == EcGranularity::Fine ||
           value == EcGranularity::SizeClassPacked ||
           value == EcGranularity::CoarsePacked;
}

constexpr bool valid_object_layout(ObjectLayout value) noexcept {
    return value == ObjectLayout::Whole || value == ObjectLayout::Split;
}

// Implements the paper's branch ordering.  Predicate thresholds are supplied
// by the caller; no numeric threshold is invented here.
constexpr ProtectionDecision choose_protection(
    const ProtectionPredicates &predicates) noexcept {
    if (predicates.transient && predicates.replication_budget_slack) {
        return {
            .method = ProtectionMethod::Replication,
            .ec_granularity = EcGranularity::NotApplicable,
            .object_layout = ObjectLayout::Whole,
            .use_batched_region_scan = false,
        };
    }

    ProtectionDecision decision;
    if (predicates.frequent_lifecycle) {
        decision.ec_granularity = EcGranularity::Fine;
    } else if (predicates.stable_cold) {
        decision.ec_granularity = EcGranularity::CoarsePacked;
    } else {
        decision.ec_granularity = EcGranularity::SizeClassPacked;
    }

    if (predicates.large_object && predicates.low_fanout_sensitivity) {
        decision.object_layout = ObjectLayout::Split;
    }
    decision.use_batched_region_scan =
        predicates.batched_region_scan_profitable;
    return decision;
}

constexpr bool valid_decision(const ProtectionDecision &decision) noexcept {
    if (!valid_protection_method(decision.method) ||
        !valid_ec_granularity(decision.ec_granularity) ||
        !valid_object_layout(decision.object_layout)) {
        return false;
    }
    if (decision.method == ProtectionMethod::Replication) {
        return decision.ec_granularity == EcGranularity::NotApplicable &&
               decision.object_layout == ObjectLayout::Whole &&
               !decision.use_batched_region_scan;
    }
    return decision.ec_granularity != EcGranularity::NotApplicable;
}

constexpr const char *to_string(ProtectionMethod method) noexcept {
    switch (method) {
    case ProtectionMethod::Replication:
        return "replication";
    case ProtectionMethod::ErasureCoding:
        return "erasure_coding";
    }
    return "unknown";
}

constexpr const char *to_string(EcGranularity granularity) noexcept {
    switch (granularity) {
    case EcGranularity::NotApplicable:
        return "not_applicable";
    case EcGranularity::Fine:
        return "fine";
    case EcGranularity::SizeClassPacked:
        return "size_class_packed";
    case EcGranularity::CoarsePacked:
        return "coarse_packed";
    }
    return "unknown";
}

constexpr const char *to_string(ObjectLayout layout) noexcept {
    switch (layout) {
    case ObjectLayout::Whole:
        return "whole";
    case ObjectLayout::Split:
        return "split";
    }
    return "unknown";
}

}  // namespace FarLib::cache::design2::behavior_group
