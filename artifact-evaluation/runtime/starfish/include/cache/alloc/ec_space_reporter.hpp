#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "cache/alloc/small_object_stripe.hpp"

namespace FarLib::cache {

// Diagnostics only. Writers update stripe-local counters under their existing
// stripe mutex; this observer never adds a lock/RMW to the object hot path.
// A rolling snapshot is exact per stripe, not a simultaneous global snapshot.
// Report sampled peaks honestly; never advertise them as exact live-byte peaks.
class EcSpaceReporter {
    using Usage = SmallObjectStripeManager::SpaceUsage;
    SmallObjectStripeManager &manager_;
    std::mutex observer_mutex_;
    std::condition_variable wake_;
    std::thread thread_;
    bool stop_ = false;
    bool work_ = false;
    uint64_t interval_ms_ = 1000;
    uint64_t work_samples_ = 0;
    uint64_t peak_occupied_ = 0;
    uint64_t peak_live_payload_ = 0;
    double peak_amplification_ = 0;
    std::chrono::steady_clock::time_point epoch_ = std::chrono::steady_clock::now();

    static double amplification(const Usage &s) {
        return s.live_payload_bytes != 0 && s.unknown_payload_objects == 0
                   ? static_cast<double>(s.sealed_group_bytes) / s.live_payload_bytes
                   : 0.0;
    }

    // observer_mutex_ serializes ONLY cold observers (timer and Work boundary).
    // No allocation/release ever acquires it.
    void sample_locked(const char *phase, bool include_in_work) {
        const auto begin = std::chrono::steady_clock::now();
        const auto s = manager_.space_usage();
        const auto end = std::chrono::steady_clock::now();
        const auto &heap = ::FarLib::allocator::remote::remote_global_heap;
        const uint64_t committed = heap.get_ec_shard_region_count() *
                                   ::FarLib::allocator::remote::RegionSize;
        const double amp = amplification(s);
        if (include_in_work) {
            ++work_samples_;
            peak_occupied_ = std::max(peak_occupied_, s.occupied_group_bytes);
            peak_live_payload_ = std::max(peak_live_payload_, s.live_payload_bytes);
            peak_amplification_ = std::max(peak_amplification_, amp);
        }
        std::ostringstream line;
        line << "ec_space phase=" << phase
             << " elapsed_ms="
             << std::chrono::duration_cast<std::chrono::milliseconds>(begin - epoch_).count()
             << " snapshot_us="
             << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
             << " consistency=rolling_per_stripe"
             << " interval_ms=" << interval_ms_
             << " occupied_group_bytes=" << s.occupied_group_bytes
             << " sealed_group_bytes=" << s.sealed_group_bytes
             << " live_payload_bytes=" << s.live_payload_bytes
             << " live_slot_bytes=" << s.live_slot_bytes
             << " parity_bytes=" << s.parity_bytes
             << " trapped_hole_bytes=" << s.trapped_hole_bytes
             << " padding_bytes=" << s.padding_bytes
             << " unknown_payload_objects=" << s.unknown_payload_objects
             << " in_progress_groups=" << s.in_progress_groups
             << " reusable_groups=" << s.reusable_groups
             << " reusable_group_bytes=" << s.reusable_group_bytes
             << " reserved_stripe_bytes=" << s.reserved_stripe_bytes
             << " ec_pool_committed_bytes=" << committed
             << " backup_growth_data_bytes="
             << manager_.backup_growth_data_bytes()
             << " backup_growth_data_limit_bytes="
             << manager_.backup_growth_data_limit_bytes()
             << " recovery_replacement_data_bytes="
             << manager_.recovery_replacement_data_bytes()
             << " recovery_replacement_data_limit_bytes="
             << manager_.recovery_replacement_data_limit_bytes()
             << " split_groups=" << s.split_groups
             << " small_groups_by_live=";
        for (size_t i = 0; i < s.group_counts_by_live.size(); ++i) {
            if (i) line << ',';
            line << i << ':' << s.group_counts_by_live[i];
        }
        line << " endpoint_occupied_bytes=";
        for (size_t i = 0; i < s.endpoint_occupied_bytes.size(); ++i) {
            if (i) line << ',';
            line << i << ':' << s.endpoint_occupied_bytes[i];
        }
        line << " sealed_space_amplification=";
        if (s.live_payload_bytes && s.unknown_payload_objects == 0) line << amp;
        else line << "NA";
        line << " work_samples=" << work_samples_
             << " work_sampled_peak_occupied_bytes=" << peak_occupied_
             << " work_sampled_peak_live_payload_bytes=" << peak_live_payload_
             << " work_sampled_peak_sealed_amplification=";
        if (peak_amplification_ != 0) line << peak_amplification_;
        else line << "NA";
        // One insertion avoids fragmenting a row across concurrent log writers.
        std::cout << line.str() + '\n';
    }

public:
    explicit EcSpaceReporter(SmallObjectStripeManager &manager) : manager_(manager) {}
    EcSpaceReporter(const EcSpaceReporter &) = delete;
    EcSpaceReporter &operator=(const EcSpaceReporter &) = delete;
    ~EcSpaceReporter() { stop(); }

    void start() {
        // Enabled by default. Zero disables only periodic sampling; boundary
        // reports and accounting remain enabled and the log declares the value.
        if (const char *text = std::getenv("FARLIB_EC_SPACE_SAMPLE_MS")) {
            char *end = nullptr;
            const auto value = std::strtoull(text, &end, 10);
            if (text[0] == '-' || end == text || *end != '\0' ||
                (value != 0 && (value < 100 || value > 60000))) {
                ERROR("FARLIB_EC_SPACE_SAMPLE_MS must be 0 or 100..60000");
            }
            interval_ms_ = value;
        }
        if (interval_ms_ == 0) return;
        thread_ = std::thread([this] {
            std::unique_lock<std::mutex> lock(observer_mutex_);
            while (!wake_.wait_for(lock, std::chrono::milliseconds(interval_ms_),
                                   [this] { return stop_; })) {
                if (work_) sample_locked("work_sample", true);
            }
        });
    }

    void begin_work() {
        std::lock_guard<std::mutex> lock(observer_mutex_);
        work_samples_ = peak_occupied_ = peak_live_payload_ = 0;
        peak_amplification_ = 0;
        epoch_ = std::chrono::steady_clock::now();
        work_ = true;
        sample_locked("work_begin", true);
    }

    void end_work() {
        std::lock_guard<std::mutex> lock(observer_mutex_);
        work_ = false;
        sample_locked("work_end", true);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(observer_mutex_);
            stop_ = true;
            work_ = false;
        }
        wake_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void final_report() {
        stop();
        std::lock_guard<std::mutex> lock(observer_mutex_);
        sample_locked("allocator_destroy", false);
    }
};

}  // namespace FarLib::cache
