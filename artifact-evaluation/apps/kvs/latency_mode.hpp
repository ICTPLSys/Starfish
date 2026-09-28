#pragma once

#include <hdr/hdr_histogram.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace kv_latency {

inline uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        throw std::overflow_error("KV latency nanosecond overflow");
    return a + b;
}

inline uint64_t parse_unsigned(const char* text, const char* name,
                               bool allow_zero = false) {
    if (!text || !*text)
        throw std::invalid_argument(std::string("missing ") + name);
    uint64_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9')
            throw std::invalid_argument(std::string("invalid integer ") + name);
        const uint64_t digit = *p - '0';
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            throw std::overflow_error(std::string("integer overflow ") + name);
        value = value * 10 + digit;
    }
    if (!allow_zero && value == 0)
        throw std::invalid_argument(std::string("zero ") + name);
    return value;
}

inline uint64_t milliseconds_to_ns(uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() / 1'000'000ULL)
        throw std::overflow_error("KV latency duration overflow");
    return value * 1'000'000ULL;
}

inline uint64_t microseconds_to_ns(uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() / 1'000ULL)
        throw std::overflow_error("KV queue deadline overflow");
    return value * 1'000ULL;
}

inline uint64_t mix_seed(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint64_t lane_seed(uint64_t base, uint64_t phase, uint64_t lane) {
    return mix_seed(base ^ mix_seed(phase + 0x6172726976616cULL) ^
                    mix_seed(lane + 0x9e3779b97f4a7c15ULL));
}

// The next arrival depends only on the seed and load, never on service time.
// Fractional nanoseconds are accumulated before conversion, avoiding a minimum
// one-nanosecond interval or per-request truncation of the exponential samples.
class PoissonArrivals {
    static long double rate_parameter(uint64_t aggregate_ops, size_t lanes) {
        if (aggregate_ops == 0 || lanes == 0)
            throw std::invalid_argument("KV latency rate/lanes must be positive");
        return static_cast<long double>(aggregate_ops) /
               (1'000'000'000.0L * static_cast<long double>(lanes));
    }
    std::mt19937_64 random_;
    std::exponential_distribution<long double> interval_;
    long double elapsed_ns_ = 0;
    bool ended_ = false;

public:
    PoissonArrivals(uint64_t aggregate_ops, size_t lanes, uint64_t seed)
        : random_(seed), interval_(rate_parameter(aggregate_ops, lanes)) {}

    bool next(uint64_t window_ns, uint64_t& offset_ns) {
        if (ended_) return false;
        long double delta;
        do {
            delta = interval_(random_);
        } while (delta == 0);
        const long double next = elapsed_ns_ + delta;
        if (!std::isfinite(next) || next >= static_cast<long double>(window_ns)) {
            ended_ = true;
            return false;
        }
        if (!(next > elapsed_ns_))
            throw std::overflow_error("KV latency arrival precision exhausted");
        elapsed_ns_ = next;
        offset_ns = static_cast<uint64_t>(elapsed_ns_);
        return true;
    }
};

struct LaneCounts {
    uint64_t scheduled = 0;
    uint64_t started = 0;
    uint64_t completed = 0;
    uint64_t completed_in_window = 0;
    uint64_t deadline_dropped = 0;
    uint64_t completed_after_deadline = 0;
    uint64_t skipped = 0;
    bool drain_timeout = false;
};

class LaneSchedule {
    PoissonArrivals arrivals_;
    uint64_t start_ns_;
    uint64_t window_ns_;
    uint64_t window_end_ns_;
    uint64_t drain_end_ns_;
    uint64_t current_due_ns_ = 0;
    uint64_t max_queue_delay_ns_;

public:
    enum class StartDecision { Started, DeadlineDropped, DrainSkipped };
    LaneCounts counts;

    LaneSchedule(uint64_t start_ns, uint64_t window_ns, uint64_t drain_ns,
                 uint64_t aggregate_ops, size_t lanes, uint64_t seed,
                 uint64_t max_queue_delay_ns = 0)
        : arrivals_(aggregate_ops, lanes, seed), start_ns_(start_ns),
          window_ns_(window_ns), window_end_ns_(checked_add(start_ns, window_ns)),
          drain_end_ns_(checked_add(window_end_ns_, drain_ns)),
          max_queue_delay_ns_(max_queue_delay_ns) {}

    bool next(uint64_t& due_ns) {
        uint64_t offset;
        if (!arrivals_.next(window_ns_, offset)) return false;
        counts.scheduled = checked_add(counts.scheduled, 1);
        due_ns = current_due_ns_ = checked_add(start_ns_, offset);
        return true;
    }

    StartDecision start_decision(uint64_t now_ns) {
        if (now_ns < current_due_ns_)
            throw std::logic_error("KV latency request started before arrival");
        if (now_ns >= drain_end_ns_) {
            ++counts.skipped;
            counts.drain_timeout = true;
            return StartDecision::DrainSkipped;
        }
        // Only admission is bounded. Once started, a slow request must finish
        // and contribute its full latency; no timeout clipping or cancellation.
        if (max_queue_delay_ns_ != 0 &&
            now_ns - current_due_ns_ > max_queue_delay_ns_) {
            ++counts.deadline_dropped;
            return StartDecision::DeadlineDropped;
        }
        ++counts.started;
        return StartDecision::Started;
    }

    bool start(uint64_t now_ns) {
        return start_decision(now_ns) == StartDecision::Started;
    }

    void complete(uint64_t now_ns) {
        ++counts.completed;
        if (now_ns < window_end_ns_) ++counts.completed_in_window;
        if (max_queue_delay_ns_ != 0 &&
            now_ns - current_due_ns_ > max_queue_delay_ns_)
            ++counts.completed_after_deadline;
        if (now_ns > drain_end_ns_) counts.drain_timeout = true;
    }

    uint64_t deadline() const { return drain_end_ns_; }
    uint64_t window_end() const { return window_end_ns_; }
};

class HistogramSet {
    struct Close {
        void operator()(hdr_histogram* h) const { if (h) hdr_close(h); }
    };
    std::array<std::unique_ptr<hdr_histogram, Close>, 3> hist_;
    uint64_t dropped_ = 0;

public:
    enum Kind { Total = 0, Service = 1, Dispatch = 2 };
    explicit HistogramSet(uint64_t maximum_ns) {
        if (maximum_ns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            throw std::overflow_error("KV latency HDR range overflow");
        for (auto& h : hist_) {
            hdr_histogram* raw = nullptr;
            if (hdr_init(1, static_cast<int64_t>(maximum_ns < 2 ? 2 : maximum_ns),
                         3, &raw) != 0)
                throw std::runtime_error("KV latency HDR initialization failed");
            h.reset(raw);
        }
    }

    void record(uint64_t due, uint64_t begin, uint64_t end) {
        if (begin < due || end < begin)
            throw std::logic_error("KV latency non-monotonic clock");
        const uint64_t values[] = {end - due, end - begin, begin - due};
        for (size_t i = 0; i < 3; ++i) {
            if (hist_[i]->total_count == std::numeric_limits<int64_t>::max() ||
                values[i] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                !hdr_record_value(hist_[i].get(), static_cast<int64_t>(values[i])))
                dropped_ = checked_add(dropped_, 1);
        }
    }

    uint64_t merge(const HistogramSet& from) {
        uint64_t merge_dropped = 0;
        // Validate every destination before mutating any histogram.
        for (size_t i = 0; i < 3; ++i) {
            const int64_t left = hist_[i]->total_count;
            const int64_t right = from.hist_[i]->total_count;
            if (left < 0 || right < 0 ||
                right > std::numeric_limits<int64_t>::max() - left)
                throw std::overflow_error("KV latency HDR count overflow");
        }
        for (size_t i = 0; i < 3; ++i)
            merge_dropped = checked_add(merge_dropped,
                static_cast<uint64_t>(hdr_add(hist_[i].get(), from.hist_[i].get())));
        dropped_ = checked_add(dropped_, from.dropped_);
        return merge_dropped;
    }

    uint64_t dropped() const { return dropped_; }
    uint64_t count(Kind kind) const { return hist_[kind]->total_count; }
    uint64_t p99(Kind kind) const {
        return hdr_value_at_percentile(hist_[kind].get(), 99);
    }
    hdr_histogram* get(Kind kind) const { return hist_[kind].get(); }
};

inline bool complete_measurement(const LaneCounts& counts,
                                 const HistogramSet& hist,
                                 uint64_t merge_dropped) {
    return !counts.drain_timeout && counts.completed != 0 &&
        counts.scheduled >= counts.deadline_dropped &&
        counts.scheduled - counts.deadline_dropped == counts.started &&
        counts.started == counts.completed && counts.skipped == 0 &&
        hist.dropped() == 0 && merge_dropped == 0 &&
        hist.count(HistogramSet::Total) == counts.completed &&
        hist.count(HistogramSet::Service) == counts.completed &&
        hist.count(HistogramSet::Dispatch) == counts.completed;
}

}  // namespace kv_latency
