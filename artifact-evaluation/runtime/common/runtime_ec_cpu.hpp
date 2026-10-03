#pragma once

#include "runtime_metadata_reporter.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <x86intrin.h>

namespace FarLib::profile::runtime_ec_cpu {

enum class Operation : uint8_t { Encode, Decode, Xor, Update, Setup };

enum class Phase : uint8_t { None, Initialization, Work };

struct Counters {
    uint64_t cycles = 0;
    uint64_t scopes = 0;
    void reset() noexcept { cycles = scopes = 0; }
    void add(const Counters &other) noexcept {
        cycles += other.cycles;
        scopes += other.scopes;
    }
    void record(Operation, uint64_t elapsed) noexcept {
        cycles += elapsed;
        ++scopes;
    }
};

struct ThreadState {
    Counters initialization{};
    Counters work{};
    uint64_t initialization_epoch = 0;
    uint64_t work_epoch = 0;
    bool retired = false;

    Counters &counters_for(Phase phase) noexcept {
        if (phase == Phase::Initialization) return initialization;
        if (phase == Phase::Work) return work;
        std::abort();
    }
    const Counters &counters_for(Phase phase) const noexcept {
        if (phase == Phase::Initialization) return initialization;
        if (phase == Phase::Work) return work;
        std::abort();
    }
};

struct Registry {
    std::mutex mutex;
    std::vector<ThreadState *> states;
    uint64_t allocated_bytes() const noexcept {
        return static_cast<uint64_t>(sizeof(Registry)) +
               static_cast<uint64_t>(states.capacity()) * sizeof(ThreadState *) +
               static_cast<uint64_t>(states.size()) * sizeof(ThreadState);
    }
};

// Keep the registry alive until after thread-local lease destructors run.
inline Registry &registry() {
    static Registry *instance = new Registry();
    return *instance;
}

inline std::atomic_bool profiler_enabled{false};
// Initialization and Work have independent admission gates and counters.
// High 32 bits are the generation; bit 31 closes the selected gate and the
// low 31 bits count admitted outer scopes.  A Scope remembers its selected
// phase, so publishing Work never makes old initialization scopes wait for the
// new phase (or vice versa).
inline constexpr uint64_t closed_bit = uint64_t{1} << 31;
inline constexpr uint64_t count_mask = closed_bit - 1;
inline constexpr uint64_t generation_mask = ~((uint64_t{1} << 32) - 1);
inline std::atomic<uint64_t> initialization_gate{closed_bit};
inline std::atomic<uint64_t> work_gate{closed_bit};
inline std::atomic<uint64_t> boundary_sequence{0};
inline std::atomic<uint64_t> current_sequence{0};
// This is also the active admission bank; None is the closed active gate.
inline std::atomic<Phase> current_phase{Phase::None};
inline std::atomic_bool initialization_started{false};
inline std::atomic_bool initialization_finished{false};
inline const char *current_system = nullptr;
inline std::mutex boundary_mutex;
inline thread_local unsigned scope_depth = 0;
inline thread_local Phase scope_phase = Phase::None;

inline std::atomic<uint64_t> &gate_for(Phase phase) noexcept {
    if (phase == Phase::Initialization) return initialization_gate;
    if (phase == Phase::Work) return work_gate;
    std::abort();
}

inline bool try_enter(std::atomic<uint64_t> &gate,
                      uint64_t observed) noexcept {
    const uint64_t generation = observed & generation_mask;
    for (;;) {
        if ((observed & closed_bit) ||
            (observed & generation_mask) != generation) return false;
        if ((observed & count_mask) == count_mask) {
            std::fputs("runtime_ec_cpu: in-flight counter exhausted\n", stderr);
            std::abort();
        }
        if (gate.compare_exchange_weak(observed, observed + 1,
                                       std::memory_order_acq_rel,
                                       std::memory_order_acquire)) return true;
    }
}

inline void leave(Phase phase) noexcept {
    gate_for(phase).fetch_sub(1, std::memory_order_release);
}

inline bool parse_enabled() {
    static const bool value = [] {
        const char *text = std::getenv("FARLIB_RUNTIME_EC_CPU");
        if (text == nullptr || *text == '\0' || std::strcmp(text, "0") == 0)
            return false;
        if (std::strcmp(text, "1") == 0) return true;
        throw std::runtime_error("FARLIB_RUNTIME_EC_CPU must be 0 or 1");
    }();
    return value;
}

inline bool enabled() { return parse_enabled(); }

inline void retire_state(ThreadState *state) noexcept {
    if (state == nullptr) return;
    Registry &r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    state->retired = true;
}

struct ThreadStateLease {
    ThreadState *state = nullptr;
    ~ThreadStateLease() { retire_state(state); }
};

inline ThreadState &thread_state(Phase phase) {
    thread_local ThreadStateLease lease;
    if (lease.state == nullptr) {
        auto *state = new ThreadState();
        if (phase == Phase::Initialization)
            state->initialization_epoch = 0;
        else if (phase == Phase::Work)
            state->work_epoch =
                current_sequence.load(std::memory_order_acquire);
        else
            std::abort();
        Registry &r = registry();
        try {
            std::lock_guard<std::mutex> lock(r.mutex);
            r.states.push_back(state);
            lease.state = state;
        } catch (...) {
            delete state;
            throw;
        }
    }
    return *lease.state;
}

inline bool gate_counts_zero() noexcept {
    return (initialization_gate.load(std::memory_order_acquire) &
            count_mask) == 0 &&
           (work_gate.load(std::memory_order_acquire) & count_mask) == 0;
}

inline void prepare_registry(Phase phase, uint64_t sequence,
                             bool reclaim_retired) {
    Registry &r = registry();
    // During init->Work publication the initialization gate is still open.
    // Do not reclaim a retired TLS state until that bank has drained; its
    // initialization counters may still be written by an admitted Scope.
    const bool can_reclaim = reclaim_retired && gate_counts_zero();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto it = r.states.begin();
    while (it != r.states.end()) {
        ThreadState *state = *it;
        if (state->retired && can_reclaim) {
            delete state;
            it = r.states.erase(it);
            continue;
        }
        state->counters_for(phase).reset();
        if (phase == Phase::Initialization)
            state->initialization_epoch = sequence;
        else
            state->work_epoch = sequence;
        ++it;
    }
}

inline void validate_system(const char *system) {
    if (system == nullptr ||
        (std::strcmp(system, "starfish") != 0 &&
         std::strcmp(system, "carbink") != 0)) {
        throw std::runtime_error("runtime EC profiler received invalid system");
    }
}

inline Counters collect_counters(Phase phase) {
    Counters total;
    Registry &r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (const ThreadState *state : r.states)
        total.add(state->counters_for(phase));
    return total;
}

inline void drain_admitted_scopes(Phase phase) {
    while ((gate_for(phase).load(std::memory_order_acquire) & count_mask) != 0)
        std::this_thread::yield();
}

inline void print_record(const char *phase, uint64_t sequence,
                         const Counters &total) {
    std::ostringstream line;
    line << "runtime_ec_cpu schema_version=2 system="
         << (current_system == nullptr ? "unknown" : current_system)
         << " phase=" << phase
         << " scope=compute_ec clock=tsc boundary_sequence="
         << sequence << " cycles=" << total.cycles
         << " scopes=" << total.scopes
         << " boundary_semantics=admitted_scope_drain\n";
    std::cout << line.str() << std::flush;
}

inline void begin_initialization(const char *system) {
    if (!parse_enabled()) return;
    if (scope_depth != 0)
        throw std::runtime_error("runtime EC initialization entered from an active scope");
    validate_system(system);
    std::lock_guard<std::mutex> boundary_lock(boundary_mutex);
    if (initialization_started.load(std::memory_order_acquire)) {
        throw std::runtime_error("runtime EC profiler initialization re-entered");
    }
    if (current_phase.load(std::memory_order_acquire) != Phase::None ||
        !gate_counts_zero() ||
        boundary_sequence.load(std::memory_order_acquire) != 0) {
        throw std::runtime_error("runtime EC profiler initialization overlaps");
    }
    const uint64_t gate = work_gate.load(std::memory_order_acquire);
    const uint64_t initialization =
        initialization_gate.load(std::memory_order_acquire);
    if ((gate & closed_bit) == 0 || (initialization & closed_bit) == 0) {
        throw std::runtime_error("runtime EC profiler initialization overlaps");
    }
    bool expected = false;
    if (!initialization_started.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        throw std::runtime_error("runtime EC profiler initialization re-entered");
    }
    current_system = system;
    current_sequence.store(0, std::memory_order_release);
    prepare_registry(Phase::Initialization, 0, true);
    profiler_enabled.store(true, std::memory_order_release);
    // Open the bank before publishing it as active.  A Scope that observes
    // Initialization therefore always has a gate it can admit to.
    initialization_gate.store(0, std::memory_order_release);
    current_phase.store(Phase::Initialization, std::memory_order_release);
}

inline uint64_t measurement_aux_bytes(void *context) noexcept {
    if (context == nullptr) return 0;
    auto *r = static_cast<Registry *>(context);
    std::lock_guard<std::mutex> lock(r->mutex);
    return r->allocated_bytes();
}

inline void install_measurement_aux_reader() {
    static std::atomic_bool installed{false};
    bool expected = false;
    if (installed.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
        ::FarLib::runtime_metadata::install_measurement_aux_reader(
            &registry(), &measurement_aux_bytes);
    }
}

inline void begin_work(const char *system) {
    if (!parse_enabled()) return;
    if (scope_depth != 0)
        throw std::runtime_error("runtime EC Work boundary entered from an active scope");
    validate_system(system);
    std::lock_guard<std::mutex> boundary_lock(boundary_mutex);
    if (!initialization_started.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            "runtime EC profiler Work requires begin_initialization");
    }
    if (current_system == nullptr ||
        std::strcmp(current_system, system) != 0) {
        throw std::runtime_error(
            "runtime EC profiler system changed across initialization");
    }
    if (!initialization_finished.load(std::memory_order_acquire)) {
        if (scope_depth != 0) {
            throw std::runtime_error(
                "runtime EC profiler Work boundary entered from an active scope");
        }
        if (current_phase.load(std::memory_order_acquire) !=
            Phase::Initialization) {
            throw std::runtime_error(
                "runtime EC profiler Work transition overlaps");
        }
        const uint64_t initialization =
            initialization_gate.load(std::memory_order_acquire);
        const uint64_t work = work_gate.load(std::memory_order_acquire);
        if ((initialization & closed_bit) ||
            (work & closed_bit) == 0 || (work & count_mask) != 0) {
            throw std::runtime_error(
                "runtime EC profiler Work transition overlaps");
        }
        const uint64_t previous =
            boundary_sequence.load(std::memory_order_acquire);
        const uint64_t sequence = previous + 1;
        if (sequence > UINT32_MAX) {
            throw std::runtime_error(
                "runtime EC profiler Work generation exhausted");
        }
        // Work counters are independent, so they can be prepared before
        // closing Initialization.  Retired states are retained until the old
        // bank drains.
        prepare_registry(Phase::Work, sequence, false);
        const uint64_t prior =
            initialization_gate.fetch_or(closed_bit, std::memory_order_acq_rel);
        if (prior & closed_bit) {
            throw std::runtime_error(
                "runtime EC profiler Work transition overlaps");
        }
        boundary_sequence.store(sequence, std::memory_order_release);
        current_sequence.store(sequence, std::memory_order_release);
        install_measurement_aux_reader();
        work_gate.store(sequence << 32, std::memory_order_release);
        // Publish Work immediately after closing Initialization.  New outer
        // scopes now use the independent Work bank while old Initialization
        // scopes drain without blocking any Work caller.
        current_phase.store(Phase::Work, std::memory_order_release);
        initialization_finished.store(true, std::memory_order_release);
        drain_admitted_scopes(Phase::Initialization);
        const Counters initialization_counters =
            collect_counters(Phase::Initialization);
        print_record("initialization", 0, initialization_counters);
        return;
    }

    if (current_phase.load(std::memory_order_acquire) != Phase::None) {
        throw std::runtime_error("runtime EC profiler Work boundary overlaps");
    }
    const uint64_t initialization =
        initialization_gate.load(std::memory_order_acquire);
    const uint64_t work = work_gate.load(std::memory_order_acquire);
    if ((initialization & closed_bit) == 0 ||
        (initialization & count_mask) != 0 ||
        (work & closed_bit) == 0 || (work & count_mask) != 0) {
        throw std::runtime_error("runtime EC profiler Work boundary overlaps");
    }
    const uint64_t previous =
        boundary_sequence.load(std::memory_order_acquire);
    const uint64_t sequence = previous + 1;
    if (sequence > UINT32_MAX) {
        throw std::runtime_error("runtime EC profiler Work generation exhausted");
    }
    prepare_registry(Phase::Work, sequence, true);
    boundary_sequence.store(sequence, std::memory_order_release);
    current_sequence.store(sequence, std::memory_order_release);
    install_measurement_aux_reader();
    work_gate.store(sequence << 32, std::memory_order_release);
    current_phase.store(Phase::Work, std::memory_order_release);
}

inline void end_work() {
    if (!parse_enabled()) return;
    if (scope_depth != 0)
        throw std::runtime_error("runtime EC Work end entered from an active scope");
    std::lock_guard<std::mutex> boundary_lock(boundary_mutex);
    if (!initialization_started.load(std::memory_order_acquire) ||
        !initialization_finished.load(std::memory_order_acquire) ||
        !profiler_enabled.load(std::memory_order_acquire) ||
        current_phase.load(std::memory_order_acquire) != Phase::Work) {
        throw std::runtime_error(
            "runtime EC profiler Work end overlaps or lacks initialization");
    }
    const uint64_t prior =
        work_gate.fetch_or(closed_bit, std::memory_order_acq_rel);
    if (prior & closed_bit) {
        current_phase.store(Phase::None, std::memory_order_release);
        throw std::runtime_error("runtime EC profiler Work end repeated");
    }
    // Stop admission before draining; a Scope that observed Work concurrently
    // either entered before this close and is drained, or retries against the
    // now-closed active bank.
    current_phase.store(Phase::None, std::memory_order_release);
    drain_admitted_scopes(Phase::Work);
    const Counters total = collect_counters(Phase::Work);
    const uint64_t sequence =
        current_sequence.load(std::memory_order_acquire);
    print_record("work", sequence, total);
}

class Scope {
public:
    explicit Scope(Operation operation) noexcept : operation_(operation) {
        // begin_initialization publishes an active bank only after its gate is
        // open.  A failed admission retries from the current active bank so a
        // phase switch cannot silently drop a measurement.
        for (;;) {
            if (scope_depth != 0) {
                ++scope_depth;
                phase_ = scope_phase;
                nested_ = true;
                return;
            }
            const Phase active =
                current_phase.load(std::memory_order_acquire);
            if (active == Phase::None) return;
            std::atomic<uint64_t> &gate = gate_for(active);
            const uint64_t observed = gate.load(std::memory_order_acquire);
            if (!try_enter(gate, observed)) continue;
            phase_ = active;
            break;
        }
        try {
            state_ = &thread_state(phase_);
        } catch (...) {
            leave(phase_);
            // Never publish a plausible, silently undercounted measurement.
            std::fputs("runtime_ec_cpu: cannot allocate/register counter\n", stderr);
            std::abort();
        }
        scope_depth = 1;
        scope_phase = phase_;
        start_ = __rdtscp(&aux_);
        active_ = true;
    }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    ~Scope() noexcept {
        if (nested_) {
            --scope_depth;
            return;
        }
        if (!active_) return;
        const uint64_t end = __rdtscp(&aux_);
        state_->counters_for(phase_).record(operation_, end - start_);
        --scope_depth;
        if (scope_depth == 0) scope_phase = Phase::None;
        leave(phase_);
    }

private:
    Operation operation_;
    Phase phase_ = Phase::None;
    ThreadState *state_ = nullptr;
    uint64_t start_ = 0;
    unsigned aux_ = 0;
    bool active_ = false;
    bool nested_ = false;
};

}  // namespace FarLib::profile::runtime_ec_cpu
