// Optional, diagnostics-only profiling for EC degraded-read recovery.
//
// This header deliberately owns no cache state.  The profile is a process-wide
// set of relaxed atomic aggregates so the benchmark can take a snapshot before
// and after a failure wave without changing ConcurrentArrayCache's layout or
// adding a lock to the recovery path.  Every stage timer is guarded by the
// FARLIB_EC_RECOVERY_PROFILE environment flag; the default path pays only the
// branch in the small ScopedTimer constructor at instrumented call sites.
//
// Elapsed fields are summed operation elapsed time.  They are not additive wall
// time: recovery operations overlap across fibres and several stages are nested
// in the inclusive post-attempt timer.  Process CPU time is measured separately
// by the benchmark.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

namespace FarLib::cache {

inline bool ec_recovery_profile_env_flag(const char *name,
                                         bool default_value) noexcept {
    const char *value = std::getenv(name);
    if (value == nullptr) return default_value;
    return std::strtol(value, nullptr, 0) != 0;
}

// This clock is also used for the first endpoint-dead timestamp when profiling
// is disabled.  That one timestamp is intentionally captured on a rare error
// path so an external benchmark can anchor a failure wave without changing
// ordinary recovery behavior.
inline uint64_t ec_recovery_profile_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline bool ec_recovery_profile_enabled() noexcept {
    static const bool enabled =
        ec_recovery_profile_env_flag("FARLIB_EC_RECOVERY_PROFILE", false);
    return enabled;
}

// Suppress only successful INFO-style recovery chatter.  Error and failure
// lines remain visible so a quiet benchmark cannot hide an invalid recovery.
inline bool ec_recovery_diag_quiet() noexcept {
    static const bool quiet =
        ec_recovery_profile_env_flag("FARLIB_EC_RECOVERY_DIAG_QUIET", true);
    return quiet;
}

// The byte oracle in the benchmark remains mandatory.  This flag controls only
// the runtime's optional re-encode observer after a successful reconstruction.
inline bool ec_recovery_verify_enabled() noexcept {
    static const bool enabled =
        ec_recovery_profile_env_flag("FARLIB_EC_RECOVERY_VERIFY", true);
    return enabled;
}

struct EcRecoveryProfileSnapshot {
    bool enabled = false;
    uint64_t first_dead_ns = 0;
    uint64_t endpoint_dead_cas = 0;

    uint64_t post_attempt_calls = 0;
    uint64_t post_attempt_elapsed_ns_sum = 0;
    uint64_t post_success = 0;
    uint64_t post_duplicate = 0;
    uint64_t post_unavailable = 0;
    uint64_t post_in_flight = 0;

    uint64_t group_view_calls = 0;
    uint64_t group_view_elapsed_ns_sum = 0;
    uint64_t target_in_flight_calls = 0;
    uint64_t target_in_flight_elapsed_ns_sum = 0;
    uint64_t target_in_flight_true = 0;
    uint64_t scratch_acquire_calls = 0;
    uint64_t scratch_acquire_elapsed_ns_sum = 0;
    uint64_t scratch_acquire_success = 0;
    uint64_t token_acquire_calls = 0;
    uint64_t token_acquire_elapsed_ns_sum = 0;
    uint64_t token_acquire_success = 0;
    uint64_t segment_post_mark_calls = 0;
    uint64_t segment_post_mark_elapsed_ns_sum = 0;

    uint64_t accepted_survivor_segments = 0;
    uint64_t accepted_survivor_bytes = 0;

    uint64_t ordinary_read_drain_calls = 0;
    uint64_t ordinary_read_drain_elapsed_ns_sum = 0;
    uint64_t segment_completion_calls = 0;
    uint64_t segment_completion_elapsed_ns_sum = 0;
    uint64_t segment_completion_success = 0;
    uint64_t segment_completion_failure = 0;

    uint64_t decode_calls = 0;
    uint64_t decode_elapsed_ns_sum = 0;
    uint64_t verify_calls = 0;
    uint64_t verify_elapsed_ns_sum = 0;
    uint64_t verify_pass = 0;
    uint64_t verify_fail = 0;
    uint64_t verify_skips = 0;

    uint64_t publication_remote_calls = 0;
    uint64_t publication_remote_elapsed_ns_sum = 0;
    uint64_t scratch_release_calls = 0;
    uint64_t scratch_release_elapsed_ns_sum = 0;
    uint64_t token_release_calls = 0;
    uint64_t token_release_elapsed_ns_sum = 0;

    uint64_t recovery_log_format_calls = 0;
    uint64_t recovery_log_format_elapsed_ns_sum = 0;
    uint64_t recovery_log_output_calls = 0;
    uint64_t recovery_log_output_elapsed_ns_sum = 0;
    uint64_t recovery_log_suppressed = 0;

    uint64_t assist_poll_all_endpoints_calls = 0;
    uint64_t assist_poll_all_endpoints_elapsed_ns_sum = 0;
    uint64_t assist_poll_endpoints = 0;

    uint64_t winner_calls = 0;
    uint64_t winner_latency_ns_sum = 0;
    uint64_t winner_latency_ns_max = 0;
    uint64_t recovered_objects = 0;
    uint64_t recovered_object_bytes = 0;
    uint64_t token_abandon_latency_calls = 0;
    uint64_t token_abandon_latency_ns_sum = 0;
    uint64_t token_mutex_acquire_calls = 0;
    uint64_t token_mutex_wait_elapsed_ns_sum = 0;
    uint64_t token_mutex_critical_elapsed_ns_sum = 0;
};

class EcRecoveryProfile {
public:
    enum class Stage : uint8_t {
        kGroupView,
        kTargetInFlight,
        kScratchAcquire,
        kTokenAcquire,
        kSegmentPostMark,
        kOrdinaryReadDrain,
        kSegmentCompletion,
        kDecode,
        kVerify,
        kPublicationRemote,
        kScratchRelease,
        kTokenRelease,
        kLogFormat,
        kLogOutput,
        kAssistPollAllEndpoints,
    };

    enum class PostOutcome : uint8_t {
        kUnknown,
        kPosted,
        kDuplicate,
        kUnavailable,
        kInFlight,
    };

    bool enabled() const noexcept { return ec_recovery_profile_enabled(); }

    // The first timestamp is intentionally recorded even when `enabled()` is
    // false.  It is written by the winning endpoint-dead CAS branch, before
    // allocator bookkeeping and before the recovery log.
    void note_endpoint_dead(uint64_t timestamp_ns) noexcept {
        uint64_t expected = 0;
        first_dead_ns_.compare_exchange_strong(expected, timestamp_ns,
                                               std::memory_order_relaxed);
        if (!enabled()) return;
        endpoint_dead_cas_.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t first_dead_ns() const noexcept {
        return first_dead_ns_.load(std::memory_order_relaxed);
    }

    void note_stage_call(Stage stage) noexcept {
        if (!enabled()) return;
        stage_counter(stage).fetch_add(1, std::memory_order_relaxed);
    }

    void note_stage_elapsed(Stage stage, uint64_t elapsed_ns) noexcept {
        if (!enabled()) return;
        stage_elapsed(stage).fetch_add(elapsed_ns, std::memory_order_relaxed);
    }

    void note_target_in_flight_result(bool result) noexcept {
        if (enabled() && result)
            target_in_flight_true_.fetch_add(1, std::memory_order_relaxed);
    }

    void note_scratch_acquire_result(bool result) noexcept {
        if (enabled() && result)
            scratch_acquire_success_.fetch_add(1, std::memory_order_relaxed);
    }

    void note_token_acquire_result(bool result) noexcept {
        if (enabled() && result)
            token_acquire_success_.fetch_add(1, std::memory_order_relaxed);
    }

    void note_assist_poll_endpoints(uint64_t endpoint_count) noexcept {
        if (enabled())
            assist_poll_endpoints_.fetch_add(endpoint_count,
                                             std::memory_order_relaxed);
    }

    void note_accepted_survivor_segment(uint32_t byte_count) noexcept {
        if (!enabled()) return;
        accepted_survivor_segments_.fetch_add(1, std::memory_order_relaxed);
        accepted_survivor_bytes_.fetch_add(byte_count, std::memory_order_relaxed);
    }

    void note_segment_completion(bool success) noexcept {
        if (!enabled()) return;
        auto &counter = success ? segment_completion_success_
                                : segment_completion_failure_;
        counter.fetch_add(1, std::memory_order_relaxed);
    }

    void note_verify_skip() noexcept {
        if (enabled()) verify_skips_.fetch_add(1, std::memory_order_relaxed);
    }

    void note_verify_result(bool pass) noexcept {
        if (!enabled()) return;
        auto &counter = pass ? verify_pass_ : verify_fail_;
        counter.fetch_add(1, std::memory_order_relaxed);
    }

    void note_post_attempt(uint64_t elapsed_ns, PostOutcome outcome,
                           uint32_t byte_count) noexcept {
        if (!enabled()) return;
        post_attempt_calls_.fetch_add(1, std::memory_order_relaxed);
        post_attempt_elapsed_ns_sum_.fetch_add(elapsed_ns,
                                               std::memory_order_relaxed);
        std::atomic<uint64_t> *counter = nullptr;
        switch (outcome) {
        case PostOutcome::kPosted:
            counter = &post_success_;
            break;
        case PostOutcome::kDuplicate:
            counter = &post_duplicate_;
            break;
        case PostOutcome::kUnavailable:
            counter = &post_unavailable_;
            break;
        case PostOutcome::kInFlight:
            counter = &post_in_flight_;
            break;
        case PostOutcome::kUnknown:
            break;
        }
        if (counter != nullptr) counter->fetch_add(1, std::memory_order_relaxed);
    }

    void note_winner(uint64_t acquire_ns, uint32_t byte_count) noexcept {
        if (!enabled()) return;
        winner_calls_.fetch_add(1, std::memory_order_relaxed);
        if (acquire_ns != 0) {
            const uint64_t now = ec_recovery_profile_now_ns();
            const uint64_t elapsed = now >= acquire_ns ? now - acquire_ns : 0;
            winner_latency_ns_sum_.fetch_add(elapsed, std::memory_order_relaxed);
            uint64_t old = winner_latency_ns_max_.load(std::memory_order_relaxed);
            while (old < elapsed &&
                   !winner_latency_ns_max_.compare_exchange_weak(
                       old, elapsed, std::memory_order_relaxed,
                       std::memory_order_relaxed)) {
            }
        }
    }

    void note_recovered_object(uint32_t byte_count) noexcept {
        if (!enabled()) return;
        recovered_objects_.fetch_add(1, std::memory_order_relaxed);
        recovered_object_bytes_.fetch_add(byte_count, std::memory_order_relaxed);
    }

    void note_token_abandon(uint64_t acquire_ns) noexcept {
        if (!enabled() || acquire_ns == 0) return;
        const uint64_t now = ec_recovery_profile_now_ns();
        const uint64_t elapsed = now >= acquire_ns ? now - acquire_ns : 0;
        token_abandon_latency_calls_.fetch_add(1, std::memory_order_relaxed);
        token_abandon_latency_ns_sum_.fetch_add(elapsed,
                                                std::memory_order_relaxed);
    }

    void note_token_mutex_acquire(uint64_t wait_elapsed_ns,
                                  uint64_t critical_elapsed_ns) noexcept {
        if (!enabled()) return;
        token_mutex_acquire_calls_.fetch_add(1, std::memory_order_relaxed);
        token_mutex_wait_elapsed_ns_sum_.fetch_add(wait_elapsed_ns,
                                                   std::memory_order_relaxed);
        token_mutex_critical_elapsed_ns_sum_.fetch_add(
            critical_elapsed_ns, std::memory_order_relaxed);
    }

    void add_token_mutex_critical_elapsed(uint64_t elapsed_ns) noexcept {
        if (enabled()) {
            token_mutex_critical_elapsed_ns_sum_.fetch_add(
                elapsed_ns, std::memory_order_relaxed);
        }
    }

    void note_log_suppressed() noexcept {
        if (enabled()) recovery_log_suppressed_.fetch_add(1,
                                                          std::memory_order_relaxed);
    }

    EcRecoveryProfileSnapshot snapshot() const noexcept {
        EcRecoveryProfileSnapshot out;
        out.enabled = enabled();
        out.first_dead_ns = first_dead_ns();
        out.endpoint_dead_cas = load(endpoint_dead_cas_);
        out.post_attempt_calls = load(post_attempt_calls_);
        out.post_attempt_elapsed_ns_sum = load(post_attempt_elapsed_ns_sum_);
        out.post_success = load(post_success_);
        out.post_duplicate = load(post_duplicate_);
        out.post_unavailable = load(post_unavailable_);
        out.post_in_flight = load(post_in_flight_);
        out.group_view_calls = load(group_view_calls_);
        out.group_view_elapsed_ns_sum = load(group_view_elapsed_ns_sum_);
        out.target_in_flight_calls = load(target_in_flight_calls_);
        out.target_in_flight_elapsed_ns_sum =
            load(target_in_flight_elapsed_ns_sum_);
        out.target_in_flight_true = load(target_in_flight_true_);
        out.scratch_acquire_calls = load(scratch_acquire_calls_);
        out.scratch_acquire_elapsed_ns_sum =
            load(scratch_acquire_elapsed_ns_sum_);
        out.scratch_acquire_success = load(scratch_acquire_success_);
        out.token_acquire_calls = load(token_acquire_calls_);
        out.token_acquire_elapsed_ns_sum = load(token_acquire_elapsed_ns_sum_);
        out.token_acquire_success = load(token_acquire_success_);
        out.segment_post_mark_calls = load(segment_post_mark_calls_);
        out.segment_post_mark_elapsed_ns_sum =
            load(segment_post_mark_elapsed_ns_sum_);
        out.accepted_survivor_segments = load(accepted_survivor_segments_);
        out.accepted_survivor_bytes = load(accepted_survivor_bytes_);
        out.ordinary_read_drain_calls = load(ordinary_read_drain_calls_);
        out.ordinary_read_drain_elapsed_ns_sum =
            load(ordinary_read_drain_elapsed_ns_sum_);
        out.segment_completion_calls = load(segment_completion_calls_);
        out.segment_completion_elapsed_ns_sum =
            load(segment_completion_elapsed_ns_sum_);
        out.segment_completion_success = load(segment_completion_success_);
        out.segment_completion_failure = load(segment_completion_failure_);
        out.decode_calls = load(decode_calls_);
        out.decode_elapsed_ns_sum = load(decode_elapsed_ns_sum_);
        out.verify_calls = load(verify_calls_);
        out.verify_elapsed_ns_sum = load(verify_elapsed_ns_sum_);
        out.verify_pass = load(verify_pass_);
        out.verify_fail = load(verify_fail_);
        out.verify_skips = load(verify_skips_);
        out.publication_remote_calls = load(publication_remote_calls_);
        out.publication_remote_elapsed_ns_sum =
            load(publication_remote_elapsed_ns_sum_);
        out.scratch_release_calls = load(scratch_release_calls_);
        out.scratch_release_elapsed_ns_sum = load(scratch_release_elapsed_ns_sum_);
        out.token_release_calls = load(token_release_calls_);
        out.token_release_elapsed_ns_sum = load(token_release_elapsed_ns_sum_);
        out.recovery_log_format_calls = load(recovery_log_format_calls_);
        out.recovery_log_format_elapsed_ns_sum =
            load(recovery_log_format_elapsed_ns_sum_);
        out.recovery_log_output_calls = load(recovery_log_output_calls_);
        out.recovery_log_output_elapsed_ns_sum =
            load(recovery_log_output_elapsed_ns_sum_);
        out.recovery_log_suppressed = load(recovery_log_suppressed_);
        out.assist_poll_all_endpoints_calls =
            load(assist_poll_all_endpoints_calls_);
        out.assist_poll_all_endpoints_elapsed_ns_sum =
            load(assist_poll_all_endpoints_elapsed_ns_sum_);
        out.assist_poll_endpoints = load(assist_poll_endpoints_);
        out.winner_calls = load(winner_calls_);
        out.winner_latency_ns_sum = load(winner_latency_ns_sum_);
        out.winner_latency_ns_max = load(winner_latency_ns_max_);
        out.recovered_objects = load(recovered_objects_);
        out.recovered_object_bytes = load(recovered_object_bytes_);
        out.token_abandon_latency_calls = load(token_abandon_latency_calls_);
        out.token_abandon_latency_ns_sum =
            load(token_abandon_latency_ns_sum_);
        out.token_mutex_acquire_calls = load(token_mutex_acquire_calls_);
        out.token_mutex_wait_elapsed_ns_sum =
            load(token_mutex_wait_elapsed_ns_sum_);
        out.token_mutex_critical_elapsed_ns_sum =
            load(token_mutex_critical_elapsed_ns_sum_);
        return out;
    }

    // Compute a cumulative delta without resetting global counters.  Callers
    // should take the first snapshot before any in-flight recovery and the
    // second after all recovery work has drained; no reset is needed.
    static EcRecoveryProfileSnapshot delta(
        const EcRecoveryProfileSnapshot &after,
        const EcRecoveryProfileSnapshot &before) noexcept {
        EcRecoveryProfileSnapshot out = after;
        out.first_dead_ns = after.first_dead_ns;
#define FARLIB_EC_PROFILE_SUBTRACT(field)                                      \
        out.field = after.field >= before.field ? after.field - before.field : 0
        FARLIB_EC_PROFILE_SUBTRACT(endpoint_dead_cas);
        FARLIB_EC_PROFILE_SUBTRACT(post_attempt_calls);
        FARLIB_EC_PROFILE_SUBTRACT(post_attempt_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(post_success);
        FARLIB_EC_PROFILE_SUBTRACT(post_duplicate);
        FARLIB_EC_PROFILE_SUBTRACT(post_unavailable);
        FARLIB_EC_PROFILE_SUBTRACT(post_in_flight);
        FARLIB_EC_PROFILE_SUBTRACT(group_view_calls);
        FARLIB_EC_PROFILE_SUBTRACT(group_view_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(target_in_flight_calls);
        FARLIB_EC_PROFILE_SUBTRACT(target_in_flight_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(target_in_flight_true);
        FARLIB_EC_PROFILE_SUBTRACT(scratch_acquire_calls);
        FARLIB_EC_PROFILE_SUBTRACT(scratch_acquire_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(scratch_acquire_success);
        FARLIB_EC_PROFILE_SUBTRACT(token_acquire_calls);
        FARLIB_EC_PROFILE_SUBTRACT(token_acquire_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(token_acquire_success);
        FARLIB_EC_PROFILE_SUBTRACT(segment_post_mark_calls);
        FARLIB_EC_PROFILE_SUBTRACT(segment_post_mark_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(accepted_survivor_segments);
        FARLIB_EC_PROFILE_SUBTRACT(accepted_survivor_bytes);
        FARLIB_EC_PROFILE_SUBTRACT(ordinary_read_drain_calls);
        FARLIB_EC_PROFILE_SUBTRACT(ordinary_read_drain_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(segment_completion_calls);
        FARLIB_EC_PROFILE_SUBTRACT(segment_completion_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(segment_completion_success);
        FARLIB_EC_PROFILE_SUBTRACT(segment_completion_failure);
        FARLIB_EC_PROFILE_SUBTRACT(decode_calls);
        FARLIB_EC_PROFILE_SUBTRACT(decode_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(verify_calls);
        FARLIB_EC_PROFILE_SUBTRACT(verify_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(verify_pass);
        FARLIB_EC_PROFILE_SUBTRACT(verify_fail);
        FARLIB_EC_PROFILE_SUBTRACT(verify_skips);
        FARLIB_EC_PROFILE_SUBTRACT(publication_remote_calls);
        FARLIB_EC_PROFILE_SUBTRACT(publication_remote_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(scratch_release_calls);
        FARLIB_EC_PROFILE_SUBTRACT(scratch_release_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(token_release_calls);
        FARLIB_EC_PROFILE_SUBTRACT(token_release_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(recovery_log_format_calls);
        FARLIB_EC_PROFILE_SUBTRACT(recovery_log_format_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(recovery_log_output_calls);
        FARLIB_EC_PROFILE_SUBTRACT(recovery_log_output_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(recovery_log_suppressed);
        FARLIB_EC_PROFILE_SUBTRACT(assist_poll_all_endpoints_calls);
        FARLIB_EC_PROFILE_SUBTRACT(assist_poll_all_endpoints_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(assist_poll_endpoints);
        FARLIB_EC_PROFILE_SUBTRACT(winner_calls);
        FARLIB_EC_PROFILE_SUBTRACT(winner_latency_ns_sum);
        // max is not additive; keep the after/before-safe maximum from the
        // interval (the caller can interpret it as cumulative if unchanged).
        out.winner_latency_ns_max = after.winner_latency_ns_max;
        FARLIB_EC_PROFILE_SUBTRACT(recovered_objects);
        FARLIB_EC_PROFILE_SUBTRACT(recovered_object_bytes);
        FARLIB_EC_PROFILE_SUBTRACT(token_abandon_latency_calls);
        FARLIB_EC_PROFILE_SUBTRACT(token_abandon_latency_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(token_mutex_acquire_calls);
        FARLIB_EC_PROFILE_SUBTRACT(token_mutex_wait_elapsed_ns_sum);
        FARLIB_EC_PROFILE_SUBTRACT(token_mutex_critical_elapsed_ns_sum);
#undef FARLIB_EC_PROFILE_SUBTRACT
        return out;
    }

    void report(const char *label) const { print_snapshot(label, snapshot()); }

    void report_delta(const char *label,
                      const EcRecoveryProfileSnapshot &before) const {
        print_snapshot(label, delta(snapshot(), before));
    }

    class ScopedTimer {
    public:
        ScopedTimer(EcRecoveryProfile &profile, Stage stage) noexcept
            : profile_(&profile), stage_(stage), active_(profile.enabled()) {
            if (active_) {
                profile_->note_stage_call(stage_);
                start_ns_ = ec_recovery_profile_now_ns();
            }
        }
        ~ScopedTimer() {
            if (active_) {
                profile_->note_stage_elapsed(
                    stage_, ec_recovery_profile_now_ns() - start_ns_);
            }
        }
        ScopedTimer(const ScopedTimer &) = delete;
        ScopedTimer &operator=(const ScopedTimer &) = delete;

    private:
        EcRecoveryProfile *profile_;
        Stage stage_;
        uint64_t start_ns_ = 0;
        bool active_ = false;
    };

    class PostAttemptScope {
    public:
        explicit PostAttemptScope(EcRecoveryProfile &profile) noexcept
            : profile_(&profile), active_(profile.enabled()) {
            if (active_) start_ns_ = ec_recovery_profile_now_ns();
        }
        ~PostAttemptScope() {
            if (active_) {
                profile_->note_post_attempt(
                    ec_recovery_profile_now_ns() - start_ns_, outcome_,
                    byte_count_);
            }
        }
        void set_byte_count(uint32_t byte_count) noexcept {
            if (active_) byte_count_ = byte_count;
        }
        void set_outcome(PostOutcome outcome) noexcept {
            if (active_) outcome_ = outcome;
        }
        PostAttemptScope(const PostAttemptScope &) = delete;
        PostAttemptScope &operator=(const PostAttemptScope &) = delete;

    private:
        EcRecoveryProfile *profile_;
        uint64_t start_ns_ = 0;
        uint32_t byte_count_ = 0;
        PostOutcome outcome_ = PostOutcome::kUnknown;
        bool active_ = false;
    };

private:
    static uint64_t load(const std::atomic<uint64_t> &value) noexcept {
        return value.load(std::memory_order_relaxed);
    }

    std::atomic<uint64_t> &stage_counter(Stage stage) noexcept {
        switch (stage) {
        case Stage::kGroupView: return group_view_calls_;
        case Stage::kTargetInFlight: return target_in_flight_calls_;
        case Stage::kScratchAcquire: return scratch_acquire_calls_;
        case Stage::kTokenAcquire: return token_acquire_calls_;
        case Stage::kSegmentPostMark: return segment_post_mark_calls_;
        case Stage::kOrdinaryReadDrain: return ordinary_read_drain_calls_;
        case Stage::kSegmentCompletion: return segment_completion_calls_;
        case Stage::kDecode: return decode_calls_;
        case Stage::kVerify: return verify_calls_;
        case Stage::kPublicationRemote: return publication_remote_calls_;
        case Stage::kScratchRelease: return scratch_release_calls_;
        case Stage::kTokenRelease: return token_release_calls_;
        case Stage::kLogFormat: return recovery_log_format_calls_;
        case Stage::kLogOutput: return recovery_log_output_calls_;
        case Stage::kAssistPollAllEndpoints:
            return assist_poll_all_endpoints_calls_;
        }
        return group_view_calls_;
    }

    std::atomic<uint64_t> &stage_elapsed(Stage stage) noexcept {
        switch (stage) {
        case Stage::kGroupView: return group_view_elapsed_ns_sum_;
        case Stage::kTargetInFlight: return target_in_flight_elapsed_ns_sum_;
        case Stage::kScratchAcquire: return scratch_acquire_elapsed_ns_sum_;
        case Stage::kTokenAcquire: return token_acquire_elapsed_ns_sum_;
        case Stage::kSegmentPostMark: return segment_post_mark_elapsed_ns_sum_;
        case Stage::kOrdinaryReadDrain:
            return ordinary_read_drain_elapsed_ns_sum_;
        case Stage::kSegmentCompletion:
            return segment_completion_elapsed_ns_sum_;
        case Stage::kDecode: return decode_elapsed_ns_sum_;
        case Stage::kVerify: return verify_elapsed_ns_sum_;
        case Stage::kPublicationRemote:
            return publication_remote_elapsed_ns_sum_;
        case Stage::kScratchRelease: return scratch_release_elapsed_ns_sum_;
        case Stage::kTokenRelease: return token_release_elapsed_ns_sum_;
        case Stage::kLogFormat: return recovery_log_format_elapsed_ns_sum_;
        case Stage::kLogOutput: return recovery_log_output_elapsed_ns_sum_;
        case Stage::kAssistPollAllEndpoints:
            return assist_poll_all_endpoints_elapsed_ns_sum_;
        }
        return group_view_elapsed_ns_sum_;
    }

    static void print_snapshot(const char *label,
                               const EcRecoveryProfileSnapshot &s) {
        std::ostringstream line;
        line << "ec_recovery_profile label=" << (label != nullptr ? label : "unknown")
             << " enabled=" << (s.enabled ? 1 : 0)
             << " elapsed_units=summed_operation_elapsed_ns_not_additive_walltime"
             << " first_dead_ns=" << s.first_dead_ns
             << " endpoint_dead_cas=" << s.endpoint_dead_cas
             << " post_attempt_calls=" << s.post_attempt_calls
             << " post_attempt_elapsed_ns_sum=" << s.post_attempt_elapsed_ns_sum
             << " post_success=" << s.post_success
             << " post_duplicate=" << s.post_duplicate
             << " post_unavailable=" << s.post_unavailable
             << " post_in_flight=" << s.post_in_flight
             << " group_view_calls=" << s.group_view_calls
             << " group_view_elapsed_ns_sum=" << s.group_view_elapsed_ns_sum
             << " target_in_flight_calls=" << s.target_in_flight_calls
             << " target_in_flight_elapsed_ns_sum="
             << s.target_in_flight_elapsed_ns_sum
             << " target_in_flight_true=" << s.target_in_flight_true
             << " scratch_acquire_calls=" << s.scratch_acquire_calls
             << " scratch_acquire_elapsed_ns_sum="
             << s.scratch_acquire_elapsed_ns_sum
             << " scratch_acquire_success=" << s.scratch_acquire_success
             << " token_acquire_calls=" << s.token_acquire_calls
             << " token_acquire_elapsed_ns_sum="
             << s.token_acquire_elapsed_ns_sum
             << " token_acquire_success=" << s.token_acquire_success
             << " segment_post_mark_calls=" << s.segment_post_mark_calls
             << " segment_post_mark_elapsed_ns_sum="
             << s.segment_post_mark_elapsed_ns_sum
             << " accepted_survivor_segments=" << s.accepted_survivor_segments
             << " accepted_survivor_bytes=" << s.accepted_survivor_bytes
             << " ordinary_read_drain_calls=" << s.ordinary_read_drain_calls
             << " ordinary_read_drain_elapsed_ns_sum="
             << s.ordinary_read_drain_elapsed_ns_sum
             << " segment_completion_calls=" << s.segment_completion_calls
             << " segment_completion_elapsed_ns_sum="
             << s.segment_completion_elapsed_ns_sum
             << " segment_completion_success=" << s.segment_completion_success
             << " segment_completion_failure=" << s.segment_completion_failure
             << " decode_calls=" << s.decode_calls
             << " decode_elapsed_ns_sum=" << s.decode_elapsed_ns_sum
             << " verify_calls=" << s.verify_calls
             << " verify_elapsed_ns_sum=" << s.verify_elapsed_ns_sum
             << " verify_pass=" << s.verify_pass
             << " verify_fail=" << s.verify_fail
             << " verify_skips=" << s.verify_skips
             << " publication_remote_calls=" << s.publication_remote_calls
             << " publication_remote_elapsed_ns_sum="
             << s.publication_remote_elapsed_ns_sum
             << " scratch_release_calls=" << s.scratch_release_calls
             << " scratch_release_elapsed_ns_sum="
             << s.scratch_release_elapsed_ns_sum
             << " token_release_calls=" << s.token_release_calls
             << " token_release_elapsed_ns_sum="
             << s.token_release_elapsed_ns_sum
             << " recovery_log_format_calls=" << s.recovery_log_format_calls
             << " recovery_log_format_elapsed_ns_sum="
             << s.recovery_log_format_elapsed_ns_sum
             << " recovery_log_output_calls=" << s.recovery_log_output_calls
             << " recovery_log_output_elapsed_ns_sum="
             << s.recovery_log_output_elapsed_ns_sum
             << " recovery_log_suppressed=" << s.recovery_log_suppressed
             << " assist_poll_all_endpoints_calls="
             << s.assist_poll_all_endpoints_calls
             << " assist_poll_all_endpoints_elapsed_ns_sum="
             << s.assist_poll_all_endpoints_elapsed_ns_sum
             << " assist_poll_endpoints=" << s.assist_poll_endpoints
             << " winner_calls=" << s.winner_calls
             << " winner_latency_ns_sum=" << s.winner_latency_ns_sum
             << " winner_latency_ns_max=" << s.winner_latency_ns_max
             << " recovered_objects=" << s.recovered_objects
             << " recovered_object_bytes=" << s.recovered_object_bytes
             << " token_abandon_latency_calls="
             << s.token_abandon_latency_calls
             << " token_abandon_latency_ns_sum="
             << s.token_abandon_latency_ns_sum
             << " token_mutex_acquire_calls=" << s.token_mutex_acquire_calls
             << " token_mutex_wait_elapsed_ns_sum="
             << s.token_mutex_wait_elapsed_ns_sum
             << " token_mutex_critical_elapsed_ns_sum="
             << s.token_mutex_critical_elapsed_ns_sum;
        std::cout << line.str() << std::endl;
    }

    std::atomic<uint64_t> first_dead_ns_{0};
    std::atomic<uint64_t> endpoint_dead_cas_{0};
    std::atomic<uint64_t> post_attempt_calls_{0};
    std::atomic<uint64_t> post_attempt_elapsed_ns_sum_{0};
    std::atomic<uint64_t> post_success_{0};
    std::atomic<uint64_t> post_duplicate_{0};
    std::atomic<uint64_t> post_unavailable_{0};
    std::atomic<uint64_t> post_in_flight_{0};
    std::atomic<uint64_t> group_view_calls_{0};
    std::atomic<uint64_t> group_view_elapsed_ns_sum_{0};
    std::atomic<uint64_t> target_in_flight_calls_{0};
    std::atomic<uint64_t> target_in_flight_elapsed_ns_sum_{0};
    std::atomic<uint64_t> target_in_flight_true_{0};
    std::atomic<uint64_t> scratch_acquire_calls_{0};
    std::atomic<uint64_t> scratch_acquire_elapsed_ns_sum_{0};
    std::atomic<uint64_t> scratch_acquire_success_{0};
    std::atomic<uint64_t> token_acquire_calls_{0};
    std::atomic<uint64_t> token_acquire_elapsed_ns_sum_{0};
    std::atomic<uint64_t> token_acquire_success_{0};
    std::atomic<uint64_t> segment_post_mark_calls_{0};
    std::atomic<uint64_t> segment_post_mark_elapsed_ns_sum_{0};
    std::atomic<uint64_t> accepted_survivor_segments_{0};
    std::atomic<uint64_t> accepted_survivor_bytes_{0};
    std::atomic<uint64_t> ordinary_read_drain_calls_{0};
    std::atomic<uint64_t> ordinary_read_drain_elapsed_ns_sum_{0};
    std::atomic<uint64_t> segment_completion_calls_{0};
    std::atomic<uint64_t> segment_completion_elapsed_ns_sum_{0};
    std::atomic<uint64_t> segment_completion_success_{0};
    std::atomic<uint64_t> segment_completion_failure_{0};
    std::atomic<uint64_t> decode_calls_{0};
    std::atomic<uint64_t> decode_elapsed_ns_sum_{0};
    std::atomic<uint64_t> verify_calls_{0};
    std::atomic<uint64_t> verify_elapsed_ns_sum_{0};
    std::atomic<uint64_t> verify_pass_{0};
    std::atomic<uint64_t> verify_fail_{0};
    std::atomic<uint64_t> verify_skips_{0};
    std::atomic<uint64_t> publication_remote_calls_{0};
    std::atomic<uint64_t> publication_remote_elapsed_ns_sum_{0};
    std::atomic<uint64_t> scratch_release_calls_{0};
    std::atomic<uint64_t> scratch_release_elapsed_ns_sum_{0};
    std::atomic<uint64_t> token_release_calls_{0};
    std::atomic<uint64_t> token_release_elapsed_ns_sum_{0};
    std::atomic<uint64_t> recovery_log_format_calls_{0};
    std::atomic<uint64_t> recovery_log_format_elapsed_ns_sum_{0};
    std::atomic<uint64_t> recovery_log_output_calls_{0};
    std::atomic<uint64_t> recovery_log_output_elapsed_ns_sum_{0};
    std::atomic<uint64_t> recovery_log_suppressed_{0};
    std::atomic<uint64_t> assist_poll_all_endpoints_calls_{0};
    std::atomic<uint64_t> assist_poll_all_endpoints_elapsed_ns_sum_{0};
    std::atomic<uint64_t> assist_poll_endpoints_{0};
    std::atomic<uint64_t> winner_calls_{0};
    std::atomic<uint64_t> winner_latency_ns_sum_{0};
    std::atomic<uint64_t> winner_latency_ns_max_{0};
    std::atomic<uint64_t> recovered_objects_{0};
    std::atomic<uint64_t> recovered_object_bytes_{0};
    std::atomic<uint64_t> token_abandon_latency_calls_{0};
    std::atomic<uint64_t> token_abandon_latency_ns_sum_{0};
    std::atomic<uint64_t> token_mutex_acquire_calls_{0};
    std::atomic<uint64_t> token_mutex_wait_elapsed_ns_sum_{0};
    std::atomic<uint64_t> token_mutex_critical_elapsed_ns_sum_{0};
};

inline EcRecoveryProfile &ec_recovery_profile() noexcept {
    static EcRecoveryProfile profile;
    return profile;
}

// A lock_guard-equivalent used only by EcReadTokenTable.  It calls the same
// blocking mutex.lock()/unlock() pair as std::lock_guard.  With profiling off,
// the only added work is the cached flag branch around those calls.  With
// profiling on, timestamps are taken immediately before/after lock() and
// immediately before unlock(); the atomic aggregates never participate in the
// mutex's correctness protocol.
class EcRecoveryTokenMutexGuard {
public:
    explicit EcRecoveryTokenMutexGuard(std::mutex &mutex)
        : mutex_(&mutex), profile_enabled_(ec_recovery_profile_enabled()) {
        const uint64_t wait_start =
            profile_enabled_ ? ec_recovery_profile_now_ns() : 0;
        mutex_->lock();
        if (profile_enabled_) {
            const uint64_t locked = ec_recovery_profile_now_ns();
            ec_recovery_profile().note_token_mutex_acquire(
                locked >= wait_start ? locked - wait_start : 0, 0);
            critical_start_ns_ = locked;
        }
    }

    ~EcRecoveryTokenMutexGuard() noexcept {
        if (profile_enabled_) {
            const uint64_t before_unlock = ec_recovery_profile_now_ns();
            const uint64_t critical =
                before_unlock >= critical_start_ns_
                    ? before_unlock - critical_start_ns_
                    : 0;
            // The acquire call above already counted the lock.  Add only the
            // critical interval here without creating another lock event.
            ec_recovery_profile().add_token_mutex_critical_elapsed(critical);
        }
        mutex_->unlock();
    }

    EcRecoveryTokenMutexGuard(const EcRecoveryTokenMutexGuard &) = delete;
    EcRecoveryTokenMutexGuard &operator=(const EcRecoveryTokenMutexGuard &) = delete;

private:
    std::mutex *mutex_;
    uint64_t critical_start_ns_ = 0;
    bool profile_enabled_ = false;
};

}  // namespace FarLib::cache
