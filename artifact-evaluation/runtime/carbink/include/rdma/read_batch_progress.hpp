#pragma once

#include <cstddef>
#include <cstdint>

namespace FarLib::rdma::read_batch {

enum class ProgressReason {
    None,
    ImmediateFirst,
    Full,
    Age,
    IdleNoSwitch,
    Credit,
};

struct ProgressDecision {
    bool submit = false;
    ProgressReason reason = ProgressReason::None;
};

// This state is always accessed while the owning PrefixQueue guard is held.
// cq_generation itself is maintained atomically by the integration layer.
class CreditAgePolicy {
public:
    ProgressDecision on_full(std::uint64_t cq_generation) const {
        if (!blocked_on_credit_) return {true, ProgressReason::Full};
        if (cq_generation != blocked_generation_) {
            return {true, ProgressReason::Credit};
        }
        return {};
    }

    ProgressDecision on_immediate_first(
        std::uint64_t cq_generation) const {
        ProgressDecision decision = on_full(cq_generation);
        if (decision.reason == ProgressReason::Full) {
            decision.reason = ProgressReason::ImmediateFirst;
        }
        return decision;
    }

    ProgressDecision on_progress(std::uint64_t cq_generation,
                                 std::uint64_t oldest_enqueue_ns,
                                 std::uint64_t now_ns,
                                 std::uint64_t max_delay_ns,
                                 bool idle_no_switch) const {
        if (blocked_on_credit_) {
            if (cq_generation != blocked_generation_) {
                return {true, ProgressReason::Credit};
            }
            return {};
        }
        if (idle_no_switch) {
            return {true, ProgressReason::IdleNoSwitch};
        }
        if (now_ns >= oldest_enqueue_ns &&
            now_ns - oldest_enqueue_ns >= max_delay_ns) {
            return {true, ProgressReason::Age};
        }
        return {};
    }

    void on_submission_result(std::uint64_t generation_before,
                              std::size_t accepted,
                              std::size_t submitted) {
        if (accepted == submitted) {
            blocked_on_credit_ = false;
            blocked_generation_ = 0;
        } else {
            blocked_on_credit_ = true;
            blocked_generation_ = generation_before;
        }
    }

    bool blocked_on_credit() const { return blocked_on_credit_; }
    std::uint64_t blocked_generation() const {
        return blocked_generation_;
    }
    void reset() {
        blocked_on_credit_ = false;
        blocked_generation_ = 0;
    }

private:
    bool blocked_on_credit_ = false;
    std::uint64_t blocked_generation_ = 0;
};

}  // namespace FarLib::rdma::read_batch
