#pragma once

// Work-boundary anchored remote-memory samples shared by every runtime.  The
// cache arms one observer thread during construction; profile::start_work and
// profile::end_work only publish timestamped boundary events.  Sampling,
// snapshotting, and output stay on the observer thread so the Work hooks never
// wait for remote state or for I/O.

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <cstring>

namespace FarLib::benchmark_memory {

inline constexpr std::array<uint64_t, 5> kScheduledSeconds = {10, 20, 30,
                                                               40, 50};
inline constexpr std::array<uint64_t, 1> kBfsScheduledSeconds = {3};
inline constexpr const char *kNormalSamplingWindow =
    "work_start_10_20_30_40_50s";
inline constexpr const char *kBfsSamplingWindow = "work_start_3s";
// Compatibility for small downstream schedule fixtures; production records
// use the explicit work_start_* names above.
inline constexpr const char *kSamplingWindow = kNormalSamplingWindow;

struct SamplingProfile {
    std::vector<uint64_t> offsets_s;
    const char *window = kNormalSamplingWindow;
};

inline SamplingProfile normal_sampling_profile() {
    return {std::vector<uint64_t>(kScheduledSeconds.begin(),
                                  kScheduledSeconds.end()),
            kNormalSamplingWindow};
}

inline SamplingProfile bfs_sampling_profile() {
    return {std::vector<uint64_t>(kBfsScheduledSeconds.begin(),
                                  kBfsScheduledSeconds.end()),
            kBfsSamplingWindow};
}

inline SamplingProfile sampling_profile_from_environment() {
    const char *text = std::getenv("FARLIB_REMOTE_MEMORY_SAMPLING_PROFILE");
    const std::string value = text == nullptr ? "normal" : text;
    if (value == "normal" || value == "work_10_20_30_40_50s")
        return normal_sampling_profile();
    if (value == "bfs_work_3s") return bfs_sampling_profile();
    throw std::runtime_error(
        "FARLIB_REMOTE_MEMORY_SAMPLING_PROFILE must be normal or bfs_work_3s");
}

inline bool explicit_kvs_work_origin_from_environment() {
    const char *text = std::getenv("FARLIB_REMOTE_MEMORY_WORK_ORIGIN");
    const std::string value = text == nullptr ? "profile_start_work" : text;
    if (value == "profile_start_work") return false;
    if (value == "kvs_request_start") return true;
    throw std::runtime_error(
        "FARLIB_REMOTE_MEMORY_WORK_ORIGIN must be profile_start_work or kvs_request_start");
}

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
        if (required) throw std::runtime_error(std::string("missing ") + name);
        return 0;
    }
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9')
            throw std::runtime_error(std::string("invalid ") + name);
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value > std::numeric_limits<uint64_t>::max())
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<uint64_t>(value);
}

inline bool enabled_from_environment() {
    const char *text = std::getenv("FARLIB_REMOTE_MEMORY_SAMPLES");
    if (text == nullptr) return false;
    if (std::string(text) == "1") return true;
    if (std::string(text) == "0") return false;
    throw std::runtime_error("FARLIB_REMOTE_MEMORY_SAMPLES must be 0 or 1");
}

struct DuePoint {
    size_t index = 0;
    uint64_t scheduled_elapsed_s = 0;
    uint64_t scheduled_ns = 0;
    uint64_t observed_ns = 0;
    bool missed_notready = false;
};

// Small fake-clock-friendly scheduler. It has no I/O or thread state, which
// keeps exact boundary tests independent of 50-second wall-clock waits.
class DeadlineSchedule {
public:
    explicit DeadlineSchedule(uint64_t start_ns)
        : DeadlineSchedule(start_ns, std::vector<uint64_t>(
                                         kScheduledSeconds.begin(),
                                         kScheduledSeconds.end())) {}

    DeadlineSchedule(uint64_t start_ns, std::vector<uint64_t> offsets_s)
        : start_ns_(start_ns), offsets_s_(std::move(offsets_s)) {
        if (offsets_s_.empty())
            throw std::runtime_error("empty sampling schedule");
        for (size_t i = 1; i < offsets_s_.size(); ++i) {
            if (offsets_s_[i] <= offsets_s_[i - 1])
                throw std::runtime_error("sampling schedule must be increasing");
        }
        const uint64_t last = offsets_s_.back() * 1'000'000'000ULL;
        if (start_ns_ > std::numeric_limits<uint64_t>::max() - last)
            throw std::runtime_error(
                "benchmark start monotonic timestamp overflows");
    }

    std::vector<DuePoint> take_due(uint64_t now_ns, uint64_t ready_ns) {
        std::vector<DuePoint> result;
        while (next_ < offsets_s_.size()) {
            const uint64_t scheduled_ns =
                start_ns_ + offsets_s_[next_] * 1'000'000'000ULL;
            if (scheduled_ns > now_ns) break;
            result.push_back(DuePoint{next_, offsets_s_[next_], scheduled_ns,
                                      now_ns, scheduled_ns < ready_ns});
            ++next_;
        }
        return result;
    }

    bool done() const { return next_ == offsets_s_.size(); }
    uint64_t next_deadline_ns() const {
        if (done()) return 0;
        return start_ns_ + offsets_s_[next_] * 1'000'000'000ULL;
    }
    size_t next_index() const { return next_; }

private:
    uint64_t start_ns_;
    std::vector<uint64_t> offsets_s_;
    size_t next_ = 0;
};

// Keep each diagnostic record on the same stdout stream as application
// output. The leading newline repairs a preceding writer that omitted its
// terminator; one fwrite keeps the record itself from field-level interleave.
inline void emit_stdout_record(const std::string &line) {
    const std::string record = "\n" + line + "\n";
    std::fwrite(record.data(), 1, record.size(), stdout);
    std::fflush(stdout);
}

class Sampler {
public:
    using SnapshotFn = std::function<std::vector<uint64_t>()>;

    Sampler(size_t expected_endpoint_count, const char *metric,
            const char *source, SnapshotFn snapshot)
        : expected_endpoint_count_(expected_endpoint_count),
          metric_(metric),
          source_(source),
          snapshot_(std::move(snapshot)),
          enabled_(enabled_from_environment()),
          profile_(normal_sampling_profile()) {
        if (metric_.empty() || source_.empty() || !snapshot_)
            throw std::runtime_error(
                "benchmark memory sampler has invalid configuration");
        if (enabled_ && expected_endpoint_count_ == 0)
            throw std::runtime_error(
                "benchmark memory sampler requires remote endpoints");
    }

    Sampler(const Sampler &) = delete;
    Sampler &operator=(const Sampler &) = delete;

    ~Sampler() { shutdown(); }

    bool enabled() const { return enabled_; }

    // Start the observer before the first Work boundary. This does not start a
    // sampling clock; the clock is created only after a Begin event arrives.
    void arm() {
        if (!enabled_ || started_) return;
        profile_ = sampling_profile_from_environment();
        explicit_kvs_work_origin_ =
            explicit_kvs_work_origin_from_environment();
        const char *cpu_text = std::getenv("FARLIB_REMOTE_MEMORY_OBSERVER_CPU");
        observer_cpu_configured_ = cpu_text != nullptr && *cpu_text != '\0';
        if (observer_cpu_configured_) {
            observer_cpu_ = parse_uint64_env(
                "FARLIB_REMOTE_MEMORY_OBSERVER_CPU", true);
            if (observer_cpu_ >= CPU_SETSIZE)
                throw std::runtime_error(
                    "FARLIB_REMOTE_MEMORY_OBSERVER_CPU exceeds CPU_SETSIZE");
        }
        stop_requested_.store(false, std::memory_order_release);
        started_ = true;
        try {
            observer_ = std::thread([this] { run(); });
        } catch (...) {
            started_ = false;
            throw;
        }
    }

    // Compatibility aliases for out-of-tree fixtures. Production caches use
    // arm()/shutdown() so a cache destructor is explicit about its lifecycle.
    void start() { arm(); }
    void stop() noexcept { shutdown(); }

    // Called by profile::start_work/end_work through static callbacks. These
    // methods perform only clock reads, atomics, and a bounded queue publish.
    void begin_work() noexcept {
        if (explicit_kvs_work_origin_) return;
        const uint64_t now = monotonic_ns();
        begin_work_at(now, "profile_start_work");
    }
    void end_work() noexcept {
        if (explicit_kvs_work_origin_) return;
        end_work_at(monotonic_ns(), "profile_start_work");
    }

    // Explicit origins are used by KVS, whose request-ready barrier is the
    // approved measurement boundary rather than the earlier generic hook.
    void begin_work_at(uint64_t start_ns, const char *origin) noexcept {
        begin_work_at_impl(start_ns, start_ns, origin_from_text(origin));
    }
    void end_work_at(uint64_t end_ns, const char *origin) noexcept {
        end_work_at_impl(end_ns, origin_from_text(origin));
    }

    static void begin_work_callback(void *context) noexcept {
        if (context != nullptr)
            static_cast<Sampler *>(context)->begin_work();
    }
    static void end_work_callback(void *context) noexcept {
        if (context != nullptr)
            static_cast<Sampler *>(context)->end_work();
    }
    static void begin_work_at_callback(void *context, uint64_t start_ns,
                                       const char *origin) noexcept {
        if (context != nullptr)
            static_cast<Sampler *>(context)->begin_work_at(start_ns, origin);
    }
    static void end_work_at_callback(void *context, uint64_t end_ns,
                                     const char *origin) noexcept {
        if (context != nullptr)
            static_cast<Sampler *>(context)->end_work_at(end_ns, origin);
    }

    // Test-only fake-clock entry points. They use the same event queue as the
    // production callbacks, so tests exercise the nonblocking boundary path.
    void begin_work_for_test(uint64_t start_ns, uint64_t ready_ns) noexcept {
        begin_work_at_impl(start_ns, ready_ns, Origin::ProfileStartWork);
    }
    void tick_for_test(uint64_t now_ns) noexcept {
        const uint64_t id = active_window_id_.load(std::memory_order_acquire);
        if (id != 0)
            publish(Event{EventKind::Tick, id, now_ns, 0,
                          Origin::ProfileStartWork});
    }
    void end_work_for_test(uint64_t end_ns) noexcept {
        end_work_at_impl(end_ns, Origin::ProfileStartWork);
    }

    void shutdown() noexcept {
        if (!started_) return;
        stop_requested_.store(true, std::memory_order_release);
        wake_.notify_all();
        if (observer_.joinable()) observer_.join();
        started_ = false;
    }

private:
    static constexpr size_t kEventCapacity = 64;
    static constexpr uint64_t kNoPublishedSequence =
        std::numeric_limits<uint64_t>::max();

    enum class EventKind : uint8_t { Begin, Tick, End };
    enum class Origin : uint8_t { ProfileStartWork, KvsRequestStart };
    struct Event {
        EventKind kind;
        uint64_t window_id;
        uint64_t timestamp_ns;
        uint64_t ready_ns;
        Origin origin;
    };
    struct EventSlot {
        Event event{EventKind::Tick, 0, 0, 0, Origin::ProfileStartWork};
        std::atomic<uint64_t> published{kNoPublishedSequence};
    };

    static double elapsed_s(uint64_t now_ns, uint64_t start_ns) {
        return now_ns < start_ns
                   ? 0.0
                   : static_cast<double>(now_ns - start_ns) /
                         1'000'000'000.0;
    }

    static void append_elapsed(std::ostringstream &line, double value) {
        line << std::fixed << std::setprecision(9) << value
             << std::defaultfloat;
    }

    static Origin origin_from_text(const char *origin) noexcept {
        if (origin != nullptr && std::strcmp(origin, "kvs_request_start") == 0)
            return Origin::KvsRequestStart;
        return Origin::ProfileStartWork;
    }

    static const char *origin_name(Origin origin) noexcept {
        return origin == Origin::KvsRequestStart ? "kvs_request_start"
                                                  : "profile_start_work";
    }

    void pin_observer() {
        if (!observer_cpu_configured_) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(static_cast<int>(observer_cpu_), &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            throw std::runtime_error("sched_setaffinity failed for memory observer");
    }

    void begin_work_at_impl(uint64_t start_ns, uint64_t ready_ns,
                            Origin origin) noexcept {
        if (!enabled_ || !started_) return;
        if (explicit_kvs_work_origin_ !=
            (origin == Origin::KvsRequestStart))
            return;
        const uint64_t id =
            next_window_id_.fetch_add(1, std::memory_order_relaxed) + 1;
        active_start_ns_.store(start_ns, std::memory_order_release);
        active_ready_ns_.store(ready_ns, std::memory_order_release);
        active_window_id_.store(id, std::memory_order_release);
        publish(Event{EventKind::Begin, id, start_ns, ready_ns, origin});
    }

    void end_work_at_impl(uint64_t end_ns, Origin origin) noexcept {
        if (!enabled_ || !started_) return;
        if (explicit_kvs_work_origin_ !=
            (origin == Origin::KvsRequestStart))
            return;
        const uint64_t id = active_window_id_.load(std::memory_order_acquire);
        if (id == 0) return;
        end_state_sequence_.fetch_add(1, std::memory_order_acq_rel);
        std::atomic_thread_fence(std::memory_order_release);
        end_window_id_.store(id, std::memory_order_relaxed);
        end_ns_.store(end_ns, std::memory_order_relaxed);
        end_state_sequence_.fetch_add(1, std::memory_order_release);
        publish(Event{EventKind::End, id, end_ns, 0, origin});
    }

    // There is one producer (the profile boundary thread) and one consumer
    // (the observer). CAS reserves a slot without waiting when the ring is
    // full; an overflow is reflected in the final Work status.
    void publish(const Event &event) noexcept {
        uint64_t sequence = producer_sequence_.load(std::memory_order_relaxed);
        for (;;) {
            const uint64_t consumed =
                consumed_sequence_.load(std::memory_order_acquire);
            if (sequence - consumed >= kEventCapacity) {
                event_overflowed_.store(true, std::memory_order_release);
                wake_.notify_one();
                return;
            }
            if (producer_sequence_.compare_exchange_weak(
                    sequence, sequence + 1, std::memory_order_relaxed,
                    std::memory_order_relaxed))
                break;
        }
        EventSlot &slot = event_ring_[sequence % kEventCapacity];
        slot.event = event;
        slot.published.store(sequence, std::memory_order_release);
        wake_.notify_one();
    }

    bool event_available() const {
        return consumed_sequence_.load(std::memory_order_acquire) <
               producer_sequence_.load(std::memory_order_acquire);
    }

    bool pop_event(Event &event) {
        const uint64_t sequence = consumer_sequence_;
        if (sequence >= producer_sequence_.load(std::memory_order_acquire))
            return false;
        EventSlot &slot = event_ring_[sequence % kEventCapacity];
        if (slot.published.load(std::memory_order_acquire) != sequence)
            return false;  // producer is still copying its payload
        event = slot.event;
        consumer_sequence_ = sequence + 1;
        consumed_sequence_.store(consumer_sequence_,
                                 std::memory_order_release);
        return true;
    }

    bool end_seen_for(uint64_t window_id, uint64_t *end_ns) const {
        for (;;) {
            const uint64_t before =
                end_state_sequence_.load(std::memory_order_acquire);
            if (before & 1) continue;
            const uint64_t observed_id =
                end_window_id_.load(std::memory_order_relaxed);
            const uint64_t observed_ns =
                end_ns_.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint64_t after =
                end_state_sequence_.load(std::memory_order_acquire);
            if (before != after) continue;
            if (observed_id != window_id) return false;
            *end_ns = observed_ns;
            return true;
        }
    }

    void emit_line(const std::string &line) {
        pending_lines_.push_back(line);
    }

    void flush_lines() {
        if (pending_lines_.empty()) return;
        size_t bytes = 0;
        for (const auto &line : pending_lines_) bytes += line.size() + 2;
        std::string batch;
        batch.reserve(bytes);
        for (const auto &line : pending_lines_) {
            batch.push_back('\n');
            batch += line;
            batch.push_back('\n');
        }
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::fwrite(batch.data(), 1, batch.size(), stdout);
        std::fflush(stdout);
        pending_lines_.clear();
    }

    void emit_missed(const DuePoint &point, const char *status) {
        std::ostringstream line;
        line << "runtime_remote_memory_missing schema_version=3"
             << " sample=" << point.index
             << " work_window_id=" << current_window_id_
             << " origin=" << origin_name(current_origin_)
             << " scheduled_elapsed_s=" << point.scheduled_elapsed_s
             << " observed_elapsed_s=";
        append_elapsed(line, elapsed_s(point.observed_ns, current_start_ns_));
        line << " status=" << status << " metric=" << metric_
             << " source=" << source_ << " start_monotonic_ns="
             << current_start_ns_ << " window=" << profile_.window
             << " sampling=" << profile_.window;
        emit_line(line.str());
        ++missing_points_;
        if (point.missed_notready) ++missed_notready_points_;
    }

    void sample(const DuePoint &point) {
        const uint64_t window_id = current_window_id_;
        const uint64_t start_ns = current_start_ns_;
        const uint64_t begin_ns = monotonic_ns();
        uint64_t end_ns = 0;
        if (active_window_id_.load(std::memory_order_acquire) != window_id ||
            (end_seen_for(window_id, &end_ns) && end_ns <= begin_ns)) {
            emit_missed(point, "work-end");
            return;
        }
        try {
            const auto values = snapshot_();
            const uint64_t snapshot_end_ns = monotonic_ns();
            // A Work end racing a blocking snapshot invalidates that sample;
            // the observer never substitutes an end-of-work snapshot.
            const bool ended =
                active_window_id_.load(std::memory_order_acquire) != window_id ||
                (end_seen_for(window_id, &end_ns) && end_ns <= snapshot_end_ns);
            if (ended) {
                emit_missed(point, "work-end");
                return;
            }
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
            line << "runtime_remote_memory schema_version=3"
                 << " sample=" << point.index
                 << " work_window_id=" << window_id
                 << " origin=" << origin_name(current_origin_)
                 << " scheduled_elapsed_s=" << point.scheduled_elapsed_s
                 << " observed_elapsed_s=";
            append_elapsed(line, elapsed_s(begin_ns, start_ns));
            line << " occupied_bytes=" << occupied << " metric=" << metric_
                 << " source=" << source_ << " start_monotonic_ns="
                 << start_ns << " window=" << profile_.window
                 << " sampling=" << profile_.window
                 << " snapshot_start_monotonic_ns=" << begin_ns
                 << " snapshot_ns=" << (snapshot_end_ns - begin_ns)
                 << " endpoint_count=" << values.size() << " endpoint_bytes=";
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

    size_t scheduled_by(uint64_t end_ns) const {
        if (end_ns < current_start_ns_) return 0;
        size_t count = 0;
        for (const uint64_t offset : profile_.offsets_s) {
            const uint64_t delta = offset * 1'000'000'000ULL;
            if (current_start_ns_ > std::numeric_limits<uint64_t>::max() - delta)
                break;
            if (current_start_ns_ + delta <= end_ns)
                ++count;
        }
        return count;
    }

    void emit_end(uint64_t end_ns, bool confirmed_end) noexcept {
        if (end_emitted_) return;
        const size_t scheduled = scheduled_by(end_ns);
        const size_t expected = profile_.offsets_s.size();
        const char *status = nullptr;
        if (!confirmed_end)
            status = "work-end-missing";
        else if (event_overflowed_.load(std::memory_order_acquire))
            status = "unsupported";
        else if (scheduled == 0)
            status = "short-work";
        else if (samples_ == expected && missing_points_ == 0)
            status = "complete";
        else
            status = "partial-work";
        std::ostringstream line;
        line << "runtime_remote_memory_end schema_version=3"
             << " work_window_id=" << current_window_id_
             << " origin=" << origin_name(current_origin_)
             << " end_monotonic_ns=" << end_ns
             << " samples=" << samples_ << " scheduled_samples=" << scheduled
             << " missing=" << missing_points_ << " sampling=" << profile_.window
             << " window=" << profile_.window << " metric=" << metric_
             << " source=" << source_ << " start_monotonic_ns="
             << current_start_ns_ << " expected_samples=" << expected
             << " missed_notready=" << missed_notready_points_
             << " status=" << status;
        emit_line(line.str());
        flush_lines();
        end_emitted_ = true;
    }

    void finish_window(uint64_t end_ns, bool confirmed_end = true) {
        if (!current_window_active_ || !schedule_) return;
        // Consume all points that were due by the real Work end as explicit
        // missing records. Points after end remain unobserved, not failures.
        const auto due = schedule_->take_due(end_ns, current_ready_ns_);
        for (const auto &point : due) emit_missed(point, "work-end");
        emit_end(end_ns, confirmed_end);
        uint64_t expected_active = current_window_id_;
        active_window_id_.compare_exchange_strong(
            expected_active, 0, std::memory_order_acq_rel);
        current_window_active_ = false;
        schedule_.reset();
    }

    void process_due(uint64_t now_ns) {
        if (!current_window_active_ || !schedule_) return;
        uint64_t end_ns = 0;
        if (end_seen_for(current_window_id_, &end_ns) && now_ns >= end_ns) {
            finish_window(end_ns);
            return;
        }
        const auto due = schedule_->take_due(now_ns, current_ready_ns_);
        if (!due.empty()) {
            for (size_t i = 0; i + 1 < due.size(); ++i) {
                emit_missed(due[i], due[i].missed_notready
                                      ? "missed-notready"
                                      : "missed-late");
            }
            const auto &point = due.back();
            if (point.missed_notready)
                emit_missed(point, "missed-notready");
            else
                sample(point);
        }
        if (end_seen_for(current_window_id_, &end_ns) && now_ns >= end_ns)
            finish_window(end_ns);
    }

    void process_event(const Event &event) {
        if (event.kind == EventKind::Begin) {
            if (current_window_active_)
                finish_window(event.timestamp_ns, false);
            current_window_id_ = event.window_id;
            current_start_ns_ = event.timestamp_ns;
            current_ready_ns_ = event.ready_ns;
            current_origin_ = event.origin;
            pending_lines_.clear();
            pending_lines_.reserve(profile_.offsets_s.size() + 1);
            samples_ = 0;
            missing_points_ = 0;
            missed_notready_points_ = 0;
            end_emitted_ = false;
            schedule_ = std::make_unique<DeadlineSchedule>(
                current_start_ns_, profile_.offsets_s);
            current_window_active_ = true;
            return;
        }
        if (!current_window_active_ || event.window_id != current_window_id_)
            return;
        if (event.kind == EventKind::Tick) {
            process_due(event.timestamp_ns);
        } else {
            finish_window(event.timestamp_ns);
        }
    }

    void run() noexcept {
        try {
            pin_observer();
            for (;;) {
                Event event{EventKind::Tick, 0, 0, 0,
                            Origin::ProfileStartWork};
                while (pop_event(event)) process_event(event);

                if (stop_requested_.load(std::memory_order_acquire) &&
                    !event_available()) {
                    if (current_window_active_) finish_window(monotonic_ns(), false);
                    break;
                }

                if (current_window_active_ && schedule_ &&
                    !schedule_->done()) {
                    const uint64_t now_ns = monotonic_ns();
                    if (now_ns >= schedule_->next_deadline_ns()) {
                        process_due(now_ns);
                        continue;
                    }
                }

                std::unique_lock<std::mutex> lock(wait_mutex_);
                if (stop_requested_.load(std::memory_order_acquire) ||
                    event_available())
                    continue;
                if (current_window_active_ && schedule_ &&
                    !schedule_->done()) {
                    const uint64_t now_ns = monotonic_ns();
                    const uint64_t deadline_ns = schedule_->next_deadline_ns();
                    const uint64_t remaining_ns =
                        deadline_ns > now_ns ? deadline_ns - now_ns : 0;
                    const auto wait_ns = std::chrono::nanoseconds(std::min<uint64_t>(
                        remaining_ns, 10'000'000ULL));
                    wake_.wait_for(lock, wait_ns, [this] {
                        return stop_requested_.load(std::memory_order_acquire) ||
                               event_available();
                    });
                } else {
                    // The producer deliberately does not take wait_mutex_:
                    // hooks must not block. A short bounded poll closes the
                    // notify-before-wait race without an unbounded sleep.
                    wake_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                        return stop_requested_.load(std::memory_order_acquire) ||
                               event_available();
                    });
                }
            }
        } catch (...) {
            // A pinning/observer failure is represented by the active window's
            // unsupported status, never by output before a Work boundary.
            event_overflowed_.store(true, std::memory_order_release);
            if (current_window_active_) finish_window(monotonic_ns(), false);
        }
    }

    const size_t expected_endpoint_count_;
    const std::string metric_;
    const std::string source_;
    const SnapshotFn snapshot_;
    const bool enabled_;
    SamplingProfile profile_;
    bool explicit_kvs_work_origin_ = false;

    std::atomic<uint64_t> next_window_id_{0};
    std::atomic<uint64_t> active_window_id_{0};
    std::atomic<uint64_t> active_start_ns_{0};
    std::atomic<uint64_t> active_ready_ns_{0};
    std::atomic<uint64_t> end_state_sequence_{0};
    std::atomic<uint64_t> end_window_id_{0};
    std::atomic<uint64_t> end_ns_{0};

    std::array<EventSlot, kEventCapacity> event_ring_{};
    std::atomic<uint64_t> producer_sequence_{0};
    std::atomic<uint64_t> consumed_sequence_{0};
    uint64_t consumer_sequence_ = 0;
    std::atomic<bool> event_overflowed_{false};

    uint64_t current_window_id_ = 0;
    uint64_t current_start_ns_ = 0;
    uint64_t current_ready_ns_ = 0;
    Origin current_origin_ = Origin::ProfileStartWork;
    size_t samples_ = 0;
    size_t missing_points_ = 0;
    size_t missed_notready_points_ = 0;
    bool current_window_active_ = false;
    bool end_emitted_ = false;
    std::unique_ptr<DeadlineSchedule> schedule_;

    uint64_t observer_cpu_ = 0;
    bool observer_cpu_configured_ = false;
    bool started_ = false;
    std::thread observer_;
    std::mutex wait_mutex_;
    std::condition_variable wake_;
    std::atomic<bool> stop_requested_{false};
    std::vector<std::string> pending_lines_;
    std::mutex output_mutex_;
};

// Explicit application-owned boundaries (currently KVS request start/drain)
// use this small bridge because the application does not own the cache's
// Sampler object. The cache installs the bridge while constructing its
// observer; calls remain a bounded atomic/event-queue operation.
using ExplicitBeginObserver =
    void (*)(void *, uint64_t, const char *) noexcept;
using ExplicitEndObserver =
    void (*)(void *, uint64_t, const char *) noexcept;
inline void *explicit_work_origin_context = nullptr;
inline ExplicitBeginObserver explicit_work_begin_observer = nullptr;
inline ExplicitEndObserver explicit_work_end_observer = nullptr;

inline void begin_work_at(uint64_t start_ns, const char *origin) noexcept {
    if (explicit_work_begin_observer != nullptr)
        explicit_work_begin_observer(explicit_work_origin_context, start_ns,
                                     origin);
}

inline void end_work_at(uint64_t end_ns, const char *origin) noexcept {
    if (explicit_work_end_observer != nullptr)
        explicit_work_end_observer(explicit_work_origin_context, end_ns,
                                   origin);
}

}  // namespace FarLib::benchmark_memory
