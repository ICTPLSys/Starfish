#pragma once

// Opt-in, benchmark-start anchored remote-memory samples shared by every
// runtime.  This header deliberately knows nothing about a particular cache
// implementation: each runtime supplies a fixed metric name and an accessor
// that returns one occupied-byte value per configured remote endpoint.

#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace FarLib::benchmark_memory {

inline constexpr std::array<uint64_t, 5> kScheduledSeconds = {10, 20, 30, 40,
                                                               50};
inline constexpr const char *kSamplingWindow =
    "benchmark_start_10_20_30_40_50s";

inline uint64_t monotonic_ns() {
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    if (value < 0) return 0;
    return static_cast<uint64_t>(value);
}

inline uint64_t parse_uint64_env(const char *name, bool required) {
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        if (required) {
            throw std::runtime_error(std::string("missing ") + name);
        }
        return 0;
    }
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            throw std::runtime_error(std::string("invalid ") + name);
        }
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(value);
}

inline bool enabled_from_environment() {
    const char *text = std::getenv("FARLIB_REMOTE_MEMORY_SAMPLES");
    if (text == nullptr) return false;
    if (std::string(text) == "1") return true;
    if (std::string(text) == "0") return false;
    throw std::runtime_error(
        "FARLIB_REMOTE_MEMORY_SAMPLES must be 0 or 1");
}

struct DuePoint {
    size_t index = 0;
    uint64_t scheduled_elapsed_s = 0;
    uint64_t scheduled_ns = 0;
    uint64_t observed_ns = 0;
    bool missed_notready = false;
};

// Small fake-clock-friendly scheduler.  It is intentionally independent of
// threads and I/O so the exact early-cache and late-observer cases can be
// unit-tested without sleeping for benchmark-scale durations.
class DeadlineSchedule {
public:
    explicit DeadlineSchedule(uint64_t start_ns) : start_ns_(start_ns) {
        constexpr uint64_t kLast = kScheduledSeconds.back() * 1'000'000'000ULL;
        if (start_ns_ > std::numeric_limits<uint64_t>::max() - kLast) {
            throw std::runtime_error("benchmark start monotonic timestamp overflows");
        }
    }

    std::vector<DuePoint> take_due(uint64_t now_ns, uint64_t ready_ns) {
        std::vector<DuePoint> result;
        while (next_ < kScheduledSeconds.size()) {
            const uint64_t scheduled_ns =
                start_ns_ + kScheduledSeconds[next_] * 1'000'000'000ULL;
            if (scheduled_ns > now_ns) break;
            result.push_back(DuePoint{
                next_, kScheduledSeconds[next_], scheduled_ns, now_ns,
                scheduled_ns < ready_ns});
            ++next_;
        }
        return result;
    }

    bool done() const { return next_ == kScheduledSeconds.size(); }
    uint64_t next_deadline_ns() const {
        if (done()) return 0;
        return start_ns_ + kScheduledSeconds[next_] * 1'000'000'000ULL;
    }
    size_t next_index() const { return next_; }

private:
    uint64_t start_ns_;
    size_t next_ = 0;
};

class Sampler {
public:
    using SnapshotFn = std::function<std::vector<uint64_t>()>;

    Sampler(size_t expected_endpoint_count, const char *metric,
            const char *source, SnapshotFn snapshot)
        : expected_endpoint_count_(expected_endpoint_count),
          metric_(metric),
          source_(source),
          snapshot_(std::move(snapshot)),
          enabled_(enabled_from_environment()) {
        if (metric_.empty() || source_.empty() || !snapshot_) {
            throw std::runtime_error("benchmark memory sampler has invalid configuration");
        }
        if (enabled_ && expected_endpoint_count_ == 0) {
            throw std::runtime_error(
                "benchmark memory sampler requires remote endpoints");
        }
    }

    Sampler(const Sampler &) = delete;
    Sampler &operator=(const Sampler &) = delete;

    ~Sampler() { stop(); }

    bool enabled() const { return enabled_; }

    // Called only after the owning cache has completed construction and all
    // accessor state is ready.  The timestamp is intentionally after cache
    // construction; scheduled points before it are reported as missed-notready
    // instead of being reconstructed from a later snapshot.
    void start() {
        if (!enabled_ || started_) return;
        start_ns_ = parse_uint64_env("FARLIB_BENCHMARK_START_MONOTONIC_NS", true);
        if (start_ns_ > monotonic_ns()) {
            throw std::runtime_error(
                "FARLIB_BENCHMARK_START_MONOTONIC_NS is in the future");
        }
        observer_cpu_ = parse_uint64_env("FARLIB_REMOTE_MEMORY_OBSERVER_CPU", true);
        if (observer_cpu_ >= CPU_SETSIZE) {
            throw std::runtime_error(
                "FARLIB_REMOTE_MEMORY_OBSERVER_CPU exceeds CPU_SETSIZE");
        }
        ready_ns_ = monotonic_ns();
        schedule_ = std::make_unique<DeadlineSchedule>(start_ns_);
        started_ = true;
        try {
            observer_ = std::thread([this] { run(); });
        } catch (...) {
            started_ = false;
            schedule_.reset();
            throw;
        }
    }

    // Idempotent.  Cache destructors call this as their first statement so
    // the observer cannot access remote allocator or EC state during teardown.
    void stop() noexcept {
        if (!started_) return;
        {
            std::lock_guard<std::mutex> lock(wait_mutex_);
            stop_requested_ = true;
        }
        wake_.notify_all();
        if (observer_.joinable()) observer_.join();
        started_ = false;
        if (!ended_) emit_end();
    }

private:
    static double elapsed_s(uint64_t now_ns, uint64_t start_ns) {
        return now_ns < start_ns
                   ? 0.0
                   : static_cast<double>(now_ns - start_ns) / 1'000'000'000.0;
    }

    static void append_elapsed(std::ostringstream &line, double value) {
        line << std::fixed << std::setprecision(9) << value
             << std::defaultfloat;
    }

    void pin_observer() {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(static_cast<int>(observer_cpu_), &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            throw std::runtime_error("sched_setaffinity failed for memory observer");
        }
    }

    bool stopping() {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        return stop_requested_;
    }

    void emit_line(const std::string &line) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::cerr << line << std::endl;
    }

    void emit_missed(const DuePoint &point, const char *status) {
        std::ostringstream line;
        line << "runtime_remote_memory_missing schema_version=2"
             << " sample=" << point.index
             << " scheduled_elapsed_s=" << point.scheduled_elapsed_s
             << " observed_elapsed_s=";
        append_elapsed(line, elapsed_s(point.observed_ns, start_ns_));
        line << " status=" << status
             << " metric=" << metric_
             << " source=" << source_
             << " start_monotonic_ns=" << start_ns_
             << " window=" << kSamplingWindow;
        emit_line(line.str());
        ++missing_points_;
        if (point.missed_notready) ++missed_notready_points_;
    }

    void sample(const DuePoint &point) {
        const uint64_t begin_ns = monotonic_ns();
        try {
            const auto values = snapshot_();
            const uint64_t end_ns = monotonic_ns();
            if (values.size() != expected_endpoint_count_) {
                emit_missed(point, "endpoint-count-mismatch");
                return;
            }
            uint64_t occupied = 0;
            for (const uint64_t value : values) {
                if (occupied > std::numeric_limits<uint64_t>::max() - value) {
                    emit_missed(point, "occupied-bytes-overflow");
                    return;
                }
                occupied += value;
            }
            std::ostringstream line;
            line << "runtime_remote_memory schema_version=2"
                 << " sample=" << point.index
                 << " scheduled_elapsed_s=" << point.scheduled_elapsed_s
                 << " observed_elapsed_s=";
            append_elapsed(line, elapsed_s(begin_ns, start_ns_));
            line << " occupied_bytes=" << occupied
                 << " metric=" << metric_
                 << " source=" << source_
                 << " start_monotonic_ns=" << start_ns_
                 << " window=" << kSamplingWindow
                 << " snapshot_start_monotonic_ns=" << begin_ns
                 << " snapshot_ns=" << (end_ns - begin_ns)
                 << " endpoint_count=" << values.size()
                 << " endpoint_bytes=";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i) line << ',';
                line << i << ':' << values[i];
            }
            line << " consistency=rolling_allocator_snapshot";
            emit_line(line.str());
            ++samples_;
        } catch (const std::exception &) {
            emit_missed(point, "snapshot-error");
        } catch (...) {
            emit_missed(point, "snapshot-error");
        }
    }

    void emit_end() noexcept {
        if (ended_) return;
        std::ostringstream line;
        line << "runtime_remote_memory_end schema_version=2"
             << " samples=" << samples_
             << " sampling=" << kSamplingWindow
             << " metric=" << metric_
             << " source=" << source_
             << " start_monotonic_ns=" << start_ns_
             << " expected_samples=" << kScheduledSeconds.size()
             << " missed_notready=" << missed_notready_points_
             << " missing=" << missing_points_;
        emit_line(line.str());
        ended_ = true;
    }

    void run() noexcept {
        try {
            pin_observer();
            while (!schedule_->done()) {
                if (stopping()) break;
                const uint64_t now = monotonic_ns();
                const auto due = schedule_->take_due(now, ready_ns_);
                if (!due.empty()) {
                    // Never catch up several historical slots.  Keep only the
                    // newest due slot; older slots are explicitly late/missing.
                    for (size_t i = 0; i + 1 < due.size(); ++i) {
                        emit_missed(due[i], due[i].missed_notready
                                                   ? "missed-notready"
                                                   : "missed-late");
                    }
                    const auto &point = due.back();
                    if (point.missed_notready) {
                        emit_missed(point, "missed-notready");
                    } else {
                        sample(point);
                    }
                }
                if (schedule_->done()) break;
                const uint64_t deadline = schedule_->next_deadline_ns();
                std::unique_lock<std::mutex> lock(wait_mutex_);
                if (stop_requested_) break;
                const uint64_t current = monotonic_ns();
                if (current < deadline) {
                    wake_.wait_for(lock,
                                   std::chrono::nanoseconds(deadline - current),
                                   [this] { return stop_requested_; });
                }
            }
        } catch (const std::exception &error) {
            emit_line(std::string("runtime_remote_memory_missing schema_version=2") +
                      " sample=-1 status=observer-error detail=" + error.what());
        } catch (...) {
            emit_line("runtime_remote_memory_missing schema_version=2 sample=-1"
                      " status=observer-error detail=unknown");
        }
        emit_end();
    }

    const size_t expected_endpoint_count_;
    const std::string metric_;
    const std::string source_;
    const SnapshotFn snapshot_;
    const bool enabled_;
    uint64_t start_ns_ = 0;
    uint64_t ready_ns_ = 0;
    uint64_t observer_cpu_ = 0;
    size_t samples_ = 0;
    size_t missing_points_ = 0;
    size_t missed_notready_points_ = 0;
    bool started_ = false;
    bool ended_ = false;
    std::unique_ptr<DeadlineSchedule> schedule_;
    std::thread observer_;
    std::mutex wait_mutex_;
    std::condition_variable wake_;
    bool stop_requested_ = false;
    std::mutex output_mutex_;
};

}  // namespace FarLib::benchmark_memory