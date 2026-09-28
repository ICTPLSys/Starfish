#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace FarLib::profile::read_size_profile {

// Diagnostic only: exact 8-byte-aligned READ payload lengths representable by
// FarObject's 16-bit size. No allocation, locking, or I/O on the recording path.
inline constexpr size_t Quantum = 8;
inline constexpr size_t BucketCount = 65536 / Quantum;

inline bool enabled() {
    static const bool value = [] {
        const char *text = std::getenv("FARLIB_READ_SIZE_PROFILE");
        if (text == nullptr || *text == '\0' ||
            (text[0] == '0' && text[1] == '\0')) return false;
        if (text[0] == '1' && text[1] == '\0') return true;
        std::abort();
    }();
    return value;
}

struct Histogram {
    std::array<uint64_t, BucketCount> counts{};
    uint64_t unbucketed_count = 0;
    uint64_t unbucketed_bytes = 0;

    void reset() {
        counts.fill(0);
        unbucketed_count = 0;
        unbucketed_bytes = 0;
    }

    void record_single(size_t bytes) {
        if ((bytes % Quantum) != 0 || bytes / Quantum >= BucketCount) {
            ++unbucketed_count;
            unbucketed_bytes += bytes;
            return;
        }
        ++counts[bytes / Quantum];
    }

    // A batch's aggregate bytes do not reveal its individual request sizes.
    void record_aggregate(size_t count, size_t bytes) {
        if (count == 1) {
            record_single(bytes);
        } else {
            unbucketed_count += count;
            unbucketed_bytes += bytes;
        }
    }

    void merge(const Histogram &other) {
        for (size_t i = 0; i < BucketCount; ++i) counts[i] += other.counts[i];
        unbucketed_count += other.unbucketed_count;
        unbucketed_bytes += other.unbucketed_bytes;
    }
};

}  // namespace FarLib::profile::read_size_profile
