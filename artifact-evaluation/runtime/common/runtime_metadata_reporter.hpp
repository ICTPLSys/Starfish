#pragma once

#include "runtime_metadata.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace FarLib::runtime_metadata {

using MeasurementAuxReader = uint64_t (*)(void *);

inline bool enabled_from_environment() {
    const char *value = std::getenv("FARLIB_RUNTIME_METADATA");
    if (value == nullptr || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::runtime_error("FARLIB_RUNTIME_METADATA must be 0 or 1");
}

// Construction/destruction and Work boundaries belong to the application
// control thread. The accessor itself locks mutable component storage.
inline void *observer_context = nullptr;
inline void (*work_end_observer)(void *) = nullptr;
inline void *measurement_aux_context = nullptr;
inline MeasurementAuxReader measurement_aux_reader = nullptr;

inline void install_measurement_aux_reader(void *context,
                                           MeasurementAuxReader reader) {
    if (context == nullptr || reader == nullptr)
        throw std::runtime_error("invalid measurement auxiliary reader");
    if (measurement_aux_reader != nullptr &&
        (measurement_aux_context != context || measurement_aux_reader != reader))
        throw std::runtime_error("runtime metadata auxiliary reader already installed");
    measurement_aux_context = context;
    measurement_aux_reader = reader;
}

inline void end_work() {
    if (work_end_observer != nullptr) work_end_observer(observer_context);
}

class Reporter {
public:
    using ReadSnapshot = Snapshot (*)(void *);

    Reporter(const char *system, void *context, ReadSnapshot reader)
        : system_(system), context_(context), reader_(reader) {
        if (observer_context != nullptr || reader == nullptr)
            throw std::runtime_error("runtime metadata observer already installed");
        observer_context = this;
        work_end_observer = +[](void *self) {
            static_cast<Reporter *>(self)->capture();
        };
    }

    Reporter(const Reporter &) = delete;
    Reporter &operator=(const Reporter &) = delete;

    ~Reporter() {
        if (observer_context == this) {
            work_end_observer = nullptr;
            observer_context = nullptr;
        }
        // No Work means no metric. Never turn an unobserved run into zero.
        if (sequence_ == 0) return;
        std::ostringstream line;
        line << "runtime_metadata schema_version=1"
             << " system=" << system_
             << " phase=work_end snapshot=last_completed_work"
             << " scope=region_group_stripe_span"
             << " accounting=sizeof_plus_capacity"
             << " consistency=component_locked"
             << " boundary_sequence=" << sequence_
             << " snapshot_us=" << snapshot_us_
             << " metadata_bytes=" << last_.metadata_bytes()
             << " measurement_aux_bytes=" << last_.measurement_aux_bytes
             << " accounted_bytes="
             << last_.metadata_bytes() + last_.measurement_aux_bytes
             << " region_bytes=" << last_.region_bytes
             << " group_bytes=" << last_.group_bytes
             << " stripe_bytes=" << last_.stripe_bytes
             << " mapping_bytes=" << last_.mapping_bytes
             << " span_bytes=" << last_.span_bytes
             << " local_regions=" << last_.local_regions
             << " remote_regions=" << last_.remote_regions
             << " stripes=" << last_.stripes
             << " group_slots=" << last_.group_slots
             << " spans=" << last_.spans << '\n';
        std::cout << line.str() << std::flush;
    }

    const Snapshot &last_snapshot() const { return last_; }
    uint64_t boundary_count() const { return sequence_; }

private:
    void capture() {
        const auto begin = std::chrono::steady_clock::now();
        last_ = reader_(context_);
        // This observer is measurement state, never application metadata.
        last_.measurement_aux_bytes +=
            sizeof(*this) + sizeof(observer_context) + sizeof(work_end_observer);
        if (measurement_aux_reader != nullptr)
            last_.measurement_aux_bytes +=
                measurement_aux_reader(measurement_aux_context);
        snapshot_us_ = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - begin).count());
        ++sequence_;
    }

    const char *system_;
    void *context_;
    ReadSnapshot reader_;
    Snapshot last_{};
    uint64_t sequence_ = 0;
    uint64_t snapshot_us_ = 0;
};

}  // namespace FarLib::runtime_metadata
