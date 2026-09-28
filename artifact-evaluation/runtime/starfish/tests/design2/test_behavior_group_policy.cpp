#include "design2/behavior_group_policy.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace policy = FarLib::cache::design2::behavior_group;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "check failed: " << #condition << " at "          \
                      << __FILE__ << ':' << __LINE__ << '\n';                \
            return false;                                                      \
        }                                                                      \
    } while (false)

policy::ProtectionPredicates predicates_for_mask(std::uint8_t mask) {
    return {
        .transient = (mask & (1u << 0)) != 0,
        .replication_budget_slack = (mask & (1u << 1)) != 0,
        .frequent_lifecycle = (mask & (1u << 2)) != 0,
        .stable_cold = (mask & (1u << 3)) != 0,
        .large_object = (mask & (1u << 4)) != 0,
        .low_fanout_sensitivity = (mask & (1u << 5)) != 0,
        .batched_region_scan_profitable = (mask & (1u << 6)) != 0,
    };
}

policy::ProtectionDecision expected_for(
    const policy::ProtectionPredicates &predicates) {
    if (predicates.transient && predicates.replication_budget_slack) {
        return {
            .method = policy::ProtectionMethod::Replication,
            .ec_granularity = policy::EcGranularity::NotApplicable,
            .object_layout = policy::ObjectLayout::Whole,
            .use_batched_region_scan = false,
        };
    }

    policy::ProtectionDecision expected;
    expected.ec_granularity = predicates.frequent_lifecycle
                                  ? policy::EcGranularity::Fine
                              : predicates.stable_cold
                                  ? policy::EcGranularity::CoarsePacked
                                  : policy::EcGranularity::SizeClassPacked;
    expected.object_layout =
        predicates.large_object && predicates.low_fanout_sensitivity
            ? policy::ObjectLayout::Split
            : policy::ObjectLayout::Whole;
    expected.use_batched_region_scan =
        predicates.batched_region_scan_profitable;
    return expected;
}

bool test_all_predicate_combinations() {
    for (std::uint16_t mask = 0; mask < (1u << 7); ++mask) {
        const auto predicates = predicates_for_mask(
            static_cast<std::uint8_t>(mask));
        const auto actual = policy::choose_protection(predicates);
        CHECK(actual == expected_for(predicates));
        CHECK(policy::valid_decision(actual));
    }
    return true;
}

bool test_dense_behavior_classes() {
    constexpr std::array expected{
        policy::BehaviorClass::ColdLow,
        policy::BehaviorClass::ColdMedium,
        policy::BehaviorClass::ColdHigh,
        policy::BehaviorClass::HotLow,
        policy::BehaviorClass::HotMedium,
        policy::BehaviorClass::HotHigh,
    };
    for (std::uint8_t id = 0; id < expected.size(); ++id) {
        CHECK(policy::valid_behavior_class_id(id));
        CHECK(policy::valid_behavior_class(expected[id]));
        CHECK(policy::behavior_class_id(expected[id]) == id);
        const auto parsed = policy::behavior_class_from_id(id);
        CHECK(parsed.has_value() && *parsed == expected[id]);
    }
    CHECK(!policy::valid_behavior_class_id(6));
    CHECK(!policy::valid_behavior_class_id(255));
    CHECK(!policy::valid_behavior_class_id(256));
    CHECK(!policy::valid_behavior_class_id(UINT32_MAX));
    CHECK(!policy::valid_behavior_class(
        static_cast<policy::BehaviorClass>(6)));
    CHECK(!policy::valid_behavior_class(
        static_cast<policy::BehaviorClass>(255)));
    CHECK(!policy::behavior_class_from_id(6).has_value());
    CHECK(!policy::behavior_class_from_id(255).has_value());
    CHECK(!policy::behavior_class_from_id(256).has_value());
    CHECK(!policy::behavior_class_from_id(UINT32_MAX).has_value());

    // There is intentionally no local-resident input in this API: the six
    // dense IDs are stable identities, while locality is an external
    // classification concern.
    CHECK(policy::to_string(policy::BehaviorClass::ColdLow) ==
          std::string_view("cold_low"));
    CHECK(policy::to_string(policy::BehaviorClass::HotHigh) ==
          std::string_view("hot_high"));
    CHECK(policy::to_string(static_cast<policy::BehaviorClass>(6)) ==
          std::string_view("unknown"));
    return true;
}

bool test_invalid_decision_enums() {
    policy::ProtectionDecision decision;
    CHECK(policy::valid_decision(decision));

    decision.method = static_cast<policy::ProtectionMethod>(255);
    CHECK(!policy::valid_decision(decision));

    decision = {};
    decision.ec_granularity = static_cast<policy::EcGranularity>(255);
    CHECK(!policy::valid_decision(decision));

    decision = {};
    decision.object_layout = static_cast<policy::ObjectLayout>(255);
    CHECK(!policy::valid_decision(decision));

    CHECK(std::string_view(policy::to_string(
              static_cast<policy::ProtectionMethod>(255))) == "unknown");
    CHECK(std::string_view(policy::to_string(
              static_cast<policy::EcGranularity>(255))) == "unknown");
    CHECK(std::string_view(policy::to_string(
              static_cast<policy::ObjectLayout>(255))) == "unknown");
    return true;
}

}  // namespace

int main() {
    if (!test_all_predicate_combinations() ||
        !test_dense_behavior_classes() || !test_invalid_decision_enums()) {
        return 1;
    }
    std::cout << "BEHAVIOR_GROUP_POLICY_TEST_OK predicates=128\n";
    return 0;
}
