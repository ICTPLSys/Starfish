#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace FarLib::hydra {
inline bool runtime_packing_enabled() {
    static const bool enabled = [] {
        const char *s = std::getenv("FARLIB_HYDRA_RUNTIME_HOT_PACKING");
        return s && s[0] == '1' && s[1] == '\0';
    }();
    return enabled;
}

// One-shot admission gate. Tracks ROOT LIFETIMES, not GC epoch membership:
// allocation may drop its epoch while still holding pinned accessors.
struct RuntimePacking {
    static constexpr size_t Shards = 256;
    static constexpr uint64_t LearnReferences = 1ULL << 25;
    enum Phase : uint8_t { Learning, Closing, Done };
    struct alignas(64) Shard {
        std::atomic<uint64_t> roots{0};
        uint64_t references = 0; // only its native worker writes during learning
    };
    std::array<Shard, Shards> shards{};
    std::atomic<Phase> phase{Learning};
    std::atomic<uintptr_t> maintenance_fibre{0};
    std::atomic<uint64_t> published_references{0};
    uint64_t promoted_objects = 0, promoted_bytes = 0;
    uint64_t observed_objects = 0, slot_metadata_bytes = 0;
    uint64_t hot_byte_budget = 0;

    static size_t native_shard() {
        static std::atomic<size_t> next{0};
        static thread_local const size_t id = next.fetch_add(1);
        if (id >= Shards) std::abort();
        return id;
    }
    uint64_t active_roots() const {
        uint64_t n = 0;
        for (const auto &s : shards) n += s.roots.load(std::memory_order_seq_cst);
        return n;
    }
};
} // namespace FarLib::hydra
