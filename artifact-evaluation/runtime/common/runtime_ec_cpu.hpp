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

enum class Operation : uint8_t { Encode, Decode, Xor, Update };

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
    Counters counters{};
    uint64_t epoch = 0;
    bool retired = false;
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
// One atomic modification order covers admission, closing and draining.
// High 32 bits are the generation; low 31 bits count admitted outer scopes.
// The closed bit prevents new admission. Generation prevents delayed entrants
// from mistaking the next Work's OPEN/0 state for the previous Work (ABA).
inline constexpr uint64_t closed_bit = uint64_t{1} << 31;
inline constexpr uint64_t count_mask = closed_bit - 1;
inline constexpr uint64_t generation_mask = ~((uint64_t{1} << 32) - 1);
inline std::atomic<uint64_t> work_gate{closed_bit};
inline std::atomic<uint64_t> boundary_sequence{0};
inline std::atomic<uint64_t> current_sequence{0};
inline const char *current_system = nullptr;
inline thread_local unsigned scope_depth = 0;

inline bool try_enter(uint64_t observed) noexcept {
    const uint64_t generation = observed & generation_mask;
    for (;;) {
        if ((observed & closed_bit) ||
            (observed & generation_mask) != generation) return false;
        if ((observed & count_mask) == count_mask) {
            std::fputs("runtime_ec_cpu: in-flight counter exhausted\n", stderr);
            std::abort();
        }
        if (work_gate.compare_exchange_weak(observed, observed + 1,
                                           std::memory_order_acq_rel,
                                           std::memory_order_acquire)) return true;
    }
}

inline void leave() noexcept {
    work_gate.fetch_sub(1, std::memory_order_release);
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

inline ThreadState &thread_state() {
    thread_local ThreadStateLease lease;
    if (lease.state == nullptr) {
        auto *state = new ThreadState();
        state->epoch = current_sequence.load(std::memory_order_acquire);
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

inline void prepare_registry_for_work(uint64_t sequence) {
    Registry &r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto it = r.states.begin();
    while (it != r.states.end()) {
        ThreadState *state = *it;
        if (state->retired) {
            delete state;
            it = r.states.erase(it);
            continue;
        }
        state->counters.reset();
        state->epoch = sequence;
        ++it;
    }
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
    if (system == nullptr ||
        (std::strcmp(system, "starfish") != 0 &&
         std::strcmp(system, "carbink") != 0)) {
        throw std::runtime_error("runtime EC profiler received invalid system");
    }
    const uint64_t gate = work_gate.load(std::memory_order_acquire);
    if (!(gate & closed_bit) || (gate & count_mask) != 0) {
        throw std::runtime_error("runtime EC profiler Work boundary overlaps");
    }
    const uint64_t sequence =
        boundary_sequence.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (sequence > UINT32_MAX) {
        throw std::runtime_error("runtime EC profiler Work generation exhausted");
    }
    current_sequence.store(sequence, std::memory_order_release);
    current_system = system;
    prepare_registry_for_work(sequence);
    install_measurement_aux_reader();
    profiler_enabled.store(true, std::memory_order_release);
    work_gate.store(sequence << 32, std::memory_order_release);
}

inline void end_work() {
    if (!profiler_enabled.load(std::memory_order_acquire)) return;
    if (work_gate.fetch_or(closed_bit, std::memory_order_acq_rel) & closed_bit) return;
    while ((work_gate.load(std::memory_order_acquire) & count_mask) != 0)
        std::this_thread::yield();
    Counters total;
    {
        Registry &r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        for (const ThreadState *state : r.states) total.add(state->counters);
    }
    std::ostringstream line;
    line << "runtime_ec_cpu schema_version=1 system="
              << (current_system == nullptr ? "unknown" : current_system)
              << " phase=work scope=compute_ec clock=tsc boundary_sequence="
              << current_sequence.load(std::memory_order_acquire)
              << " cycles=" << total.cycles << " scopes=" << total.scopes
              << '\n';
    std::cout << line.str() << std::flush;
}

class Scope {
public:
    explicit Scope(Operation operation) noexcept : operation_(operation) {
        // begin_work validates the environment; inactive is the disabled fast path.
        const uint64_t observed = work_gate.load(std::memory_order_acquire);
        if (observed & closed_bit) return;
        if (scope_depth != 0) {
            ++scope_depth;
            nested_ = true;
            return;
        }
        if (!try_enter(observed)) return;
        try {
            state_ = &thread_state();
        } catch (...) {
            leave();
            // Never publish a plausible, silently undercounted measurement.
            std::fputs("runtime_ec_cpu: cannot allocate/register counter\n", stderr);
            std::abort();
        }
        scope_depth = 1;
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
        state_->counters.record(operation_, end - start_);
        --scope_depth;
        leave();
    }

private:
    Operation operation_;
    ThreadState *state_ = nullptr;
    uint64_t start_ = 0;
    unsigned aux_ = 0;
    bool active_ = false;
    bool nested_ = false;
};

}  // namespace FarLib::profile::runtime_ec_cpu
