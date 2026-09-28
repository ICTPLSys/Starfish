#pragma once

#include "utils/cpu_cycles.hpp"

#include <cstdint>

namespace FarLib::profile::ec_rmw_prepare {

struct Trace {
    uint64_t summary = 0;
    uint64_t lock_wait = 0;
    uint64_t stripe_held = 0;
    uint64_t index = 0;
    uint64_t summary_calls = 0;
    uint64_t summary_misses = 0;
};

class Scope {
public:
    explicit Scope(uint64_t *counter)
        : counter_(counter), start_(counter != nullptr ? ::get_cycles() : 0) {}

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    ~Scope() { stop(); }

    void stop() {
        if (counter_ == nullptr) return;
        *counter_ += ::get_cycles() - start_;
        counter_ = nullptr;
    }

private:
    uint64_t *counter_ = nullptr;
    uint64_t start_ = 0;
};

}  // namespace FarLib::profile::ec_rmw_prepare
