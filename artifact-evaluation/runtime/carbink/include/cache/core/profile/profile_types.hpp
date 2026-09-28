#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "utils/stats.hpp"

namespace FarLib {
namespace cache {

class ConcurrentArrayCache;
class FarObjectEntry;

namespace detail {

struct FrequencyProfileMarkResult {
    uint32_t window_frequency = 0;
    uint32_t ema_frequency = 0;
};

struct FrequencyMarkPassContext {
    uint32_t mark_pass_ordinal = 0;
    size_t ema_mark_interval = 1;
    bool update_ema_this_pass = true;
};

struct FullPopulationFrequencyTracker {
    struct LiveEntryRegistryShard {
        std::mutex mutex;
        std::vector<FarObjectEntry *> entries;
        std::unordered_map<FarObjectEntry *, size_t> indices;
    };

    static constexpr size_t ShardCount = 64;

    std::array<LiveEntryRegistryShard, ShardCount> live_entry_registry;
    std::mutex scan_mutex;
    std::condition_variable scan_cond;
    std::thread scanner_thread;
    bool scanner_stop = false;
    std::atomic<uint32_t> scan_sequence{0};

    size_t shard_index(const FarObjectEntry *entry) const;
    profile::FullPopulationFrequencySnapshot collect_snapshot(
        const ConcurrentArrayCache &cache, uint32_t current_scan_sequence);
    void run(ConcurrentArrayCache &cache);
    void start(ConcurrentArrayCache &cache);
    void stop();
    void register_entry(FarObjectEntry *entry);
    void unregister_entry(FarObjectEntry *entry);
    void move_entry(FarObjectEntry *from, FarObjectEntry *to);
};

}  // namespace detail

}  // namespace cache
}  // namespace FarLib
