#pragma once

namespace FarLib::cache {

inline bool ConcurrentArrayCache::full_population_frequency_stats_enabled() {
    return hybrid_profiling_enabled() &&
           ::FarLib::get_config().enable_full_population_frequency_stats;
}

inline size_t ConcurrentArrayCache::full_population_frequency_scan_period_ms() {
    return ::FarLib::get_config().full_population_frequency_scan_period_ms;
}

inline size_t detail::FullPopulationFrequencyTracker::shard_index(
    const FarObjectEntry *entry) const {
    return (reinterpret_cast<uintptr_t>(entry) >> 3) % ShardCount;
}

inline profile::FullPopulationFrequencySnapshot
detail::FullPopulationFrequencyTracker::collect_snapshot(
    const ConcurrentArrayCache &cache, uint32_t current_scan_sequence) {
    const auto scan_start = std::chrono::steady_clock::now();
    profile::FullPopulationFrequencySnapshot snapshot;
    snapshot.scan_sequence = current_scan_sequence;
    snapshot.observed_mark_pass_ordinal =
        cache.hybrid_profile_mark_pass_counter.load(std::memory_order_relaxed);
    snapshot.scan_period_ms =
        ConcurrentArrayCache::full_population_frequency_scan_period_ms();
    for (auto &shard : live_entry_registry) {
        std::lock_guard<std::mutex> lock(shard.mutex);
        for (auto *entry : shard.entries) {
            if (entry == nullptr) {
                continue;
            }
            auto state = entry->load_state(std::memory_order_relaxed);
            if (state.state == FREE || state.state == BUSY) {
                continue;
            }
            snapshot.window_histogram.add(
                entry->load_published_window_frequency());
            snapshot.ema_histogram.add(entry->load_published_ema_frequency());
        }
    }
    snapshot.scan_duration_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - scan_start)
            .count());
    return snapshot;
}

inline void detail::FullPopulationFrequencyTracker::run(
    ConcurrentArrayCache &cache) {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(scan_mutex);
            if (scan_cond.wait_for(
                    lock,
                    std::chrono::milliseconds(
                        ConcurrentArrayCache::full_population_frequency_scan_period_ms()),
                    [this] { return scanner_stop; })) {
                return;
            }
        }
        const uint32_t scan_sequence =
            this->scan_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        if (!profile::should_record_frequency_output()) {
            continue;
        }
        auto snapshot = collect_snapshot(cache, scan_sequence);
        std::cerr << "\nfullpop.scan seq=" << snapshot.scan_sequence
                  << " us=" << snapshot.scan_duration_us
                  << " observed_ordinal="
                  << snapshot.observed_mark_pass_ordinal << std::endl;
        profile::record_full_population_frequency_snapshot(snapshot);
    }
}

inline void detail::FullPopulationFrequencyTracker::start(
    ConcurrentArrayCache &cache) {
    scanner_stop = false;
    scanner_thread = std::thread([this, &cache] { run(cache); });
}

inline void detail::FullPopulationFrequencyTracker::stop() {
    if (!scanner_thread.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(scan_mutex);
        scanner_stop = true;
    }
    scan_cond.notify_one();
    scanner_thread.join();
}

inline void detail::FullPopulationFrequencyTracker::register_entry(
    FarObjectEntry *entry) {
    if (!ConcurrentArrayCache::full_population_frequency_stats_enabled() ||
        entry == nullptr) {
        return;
    }
    auto &shard = live_entry_registry[shard_index(entry)];
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto [it, inserted] = shard.indices.emplace(entry, shard.entries.size());
    if (!inserted) {
        return;
    }
    shard.entries.push_back(entry);
}

inline void
detail::FullPopulationFrequencyTracker::unregister_entry(
    FarObjectEntry *entry) {
    if (!ConcurrentArrayCache::full_population_frequency_stats_enabled() ||
        entry == nullptr) {
        return;
    }
    auto &shard = live_entry_registry[shard_index(entry)];
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto it = shard.indices.find(entry);
    if (it == shard.indices.end()) {
        return;
    }
    const size_t idx = it->second;
    FarObjectEntry *tail = shard.entries.back();
    shard.entries[idx] = tail;
    shard.entries.pop_back();
    shard.indices.erase(it);
    if (tail != entry) {
        shard.indices[tail] = idx;
    }
}

inline void detail::FullPopulationFrequencyTracker::move_entry(
    FarObjectEntry *from, FarObjectEntry *to) {
    if (!ConcurrentArrayCache::full_population_frequency_stats_enabled() ||
        from == nullptr || to == nullptr || from == to) {
        return;
    }
    const size_t from_idx = shard_index(from);
    const size_t to_idx = shard_index(to);
    auto move_entry = [&](LiveEntryRegistryShard &from_shard,
                          LiveEntryRegistryShard &to_shard) {
        auto from_it = from_shard.indices.find(from);
        if (from_it == from_shard.indices.end()) {
            return;
        }
        if (&from_shard == &to_shard) {
            const size_t idx = from_it->second;
            from_shard.entries[idx] = to;
            from_shard.indices.erase(from_it);
            from_shard.indices[to] = idx;
            return;
        }
        const size_t idx = from_it->second;
        FarObjectEntry *tail = from_shard.entries.back();
        from_shard.entries[idx] = tail;
        from_shard.entries.pop_back();
        from_shard.indices.erase(from_it);
        if (tail != from) {
            from_shard.indices[tail] = idx;
        }
        auto [to_it, inserted] =
            to_shard.indices.emplace(to, to_shard.entries.size());
        if (inserted) {
            to_shard.entries.push_back(to);
        } else {
            to_shard.entries[to_it->second] = to;
        }
    };
    if (from_idx == to_idx) {
        auto &shard = live_entry_registry[from_idx];
        std::lock_guard<std::mutex> lock(shard.mutex);
        move_entry(shard, shard);
        return;
    }
    auto &first = live_entry_registry[std::min(from_idx, to_idx)];
    auto &second = live_entry_registry[std::max(from_idx, to_idx)];
    std::lock(first.mutex, second.mutex);
    std::lock_guard<std::mutex> first_lock(first.mutex, std::adopt_lock);
    std::lock_guard<std::mutex> second_lock(second.mutex, std::adopt_lock);
    if (from_idx < to_idx) {
        move_entry(first, second);
    } else {
        move_entry(second, first);
    }
}

inline void ConcurrentArrayCache::register_live_entry(FarObjectEntry *entry) {
    full_population_frequency_tracker.register_entry(entry);
}

inline void ConcurrentArrayCache::unregister_live_entry(FarObjectEntry *entry) {
    full_population_frequency_tracker.unregister_entry(entry);
}

inline void ConcurrentArrayCache::move_live_entry(FarObjectEntry *from,
                                                  FarObjectEntry *to) {
    full_population_frequency_tracker.move_entry(from, to);
}

inline const std::string &ConcurrentArrayCache::profiling_method() {
    return ::FarLib::get_config().profiling_method;
}

inline bool ConcurrentArrayCache::hybrid_profiling_enabled() {
    return ::FarLib::get_config().profiling_enabled;
}

inline bool ConcurrentArrayCache::hybrid_profile_object_access_enabled() {
    return ::FarLib::get_config().profiling_object_access_enabled;
}

inline bool ConcurrentArrayCache::hybrid_profile_dereference_access_enabled() {
    return ::FarLib::get_config().profiling_dereference_enabled;
}

inline uint32_t ConcurrentArrayCache::local_fast_path_sample_weight() {
    const size_t shift =
        ::FarLib::get_config().hybrid_profile_local_fast_path_sample_shift;
    constexpr size_t MaxShift = std::numeric_limits<uint32_t>::digits - 2;
    if (shift >= MaxShift) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(1u << shift);
}

inline bool ConcurrentArrayCache::should_sample_local_fast_path() {
    const size_t shift =
        ::FarLib::get_config().hybrid_profile_local_fast_path_sample_shift;
    if (shift == 0) {
        return true;
    }
    constexpr size_t MaxShift = std::numeric_limits<uint32_t>::digits - 2;
    if (shift >= MaxShift) {
        return true;
    }
    thread_local uint32_t sample_counter = 0;
    const uint32_t mask = (1u << shift) - 1u;
    return ((sample_counter++) & mask) == 0;
}

inline profile::ReferenceHeatClass
ConcurrentArrayCache::published_reference_heat_class(
    const FarObjectEntry &entry) {
    const size_t bucket =
        profile::FrequencyHistogramSnapshot::bucket_index(
            entry.load_published_ema_frequency());
    if (bucket <= profile::current_reference_heat_cold_max_bucket()) {
        return profile::ReferenceHeatClass::Cold;
    }
    if (bucket <= profile::current_reference_heat_warm_max_bucket()) {
        return profile::ReferenceHeatClass::Warm;
    }
    return profile::ReferenceHeatClass::Hot;
}

inline void ConcurrentArrayCache::record_reference_access(
    const FarObjectEntry &entry, profile::ReferenceKind kind,
    int64_t weight) const {
    if (!hybrid_profiling_enabled()) {
        return;
    }
    const auto state = entry.load_state(std::memory_order_relaxed);
    profile::count_reference_access(published_reference_heat_class(entry), kind,
                                    state.dirty, weight);
}

inline void ConcurrentArrayCache::record_local_fast_path_reference(
    const FarObjectEntry &entry, profile::ReferenceKind kind) const {
    const_cast<ConcurrentArrayCache *>(this)->record_resident_profile_reference(
        entry, kind);
    if (!hybrid_profiling_enabled()) {
        return;
    }
    if (!should_sample_local_fast_path()) {
        return;
    }
    record_reference_access(entry, kind, local_fast_path_sample_weight());
}

inline void ConcurrentArrayCache::record_non_fast_path_reference(
    const FarObjectEntry &entry, profile::ReferenceKind kind) const {
    const_cast<ConcurrentArrayCache *>(this)->record_resident_profile_reference(
        entry, kind);
    record_reference_access(entry, kind, 1);
}

inline void ConcurrentArrayCache::record_local_fast_path_access(
    const FarObjectEntry &entry, profile::ReferenceKind kind) const {
    const_cast<ConcurrentArrayCache *>(this)->record_resident_profile_reference(
        entry, kind);
    if (!hybrid_profiling_enabled()) {
        return;
    }
    if (!should_sample_local_fast_path()) {
        return;
    }
    const auto weight = local_fast_path_sample_weight();
    if (hybrid_profile_object_access_enabled()) {
        entry.add_window_frequency(weight);
    }
    record_reference_access(entry, kind, weight);
}

inline void ConcurrentArrayCache::record_non_fast_path_access(
    const FarObjectEntry &entry, profile::ReferenceKind kind) const {
    const_cast<ConcurrentArrayCache *>(this)->record_resident_profile_reference(
        entry, kind);
    if (hybrid_profile_object_access_enabled()) {
        entry.add_window_frequency(1);
    }
    record_reference_access(entry, kind, 1);
}

inline ConcurrentArrayCache::FrequencyProfileMarkResult
ConcurrentArrayCache::update_frequency_profile_on_mark(
    const FarObjectEntry &entry, bool update_ema_this_pass) const {
    FrequencyProfileMarkResult result;
    if (!hybrid_profiling_enabled()) {
        return result;
    }
    if (update_ema_this_pass) {
        result.window_frequency = entry.consume_window_frequency();
        entry.update_ema_frequency(
            result.window_frequency,
            ::FarLib::get_config().hybrid_profile_frequency_ema_shift);
    } else {
        result.window_frequency = entry.load_window_frequency();
    }
    result.ema_frequency = entry.load_ema_frequency();
    entry.publish_frequency_profile(result.window_frequency,
                                    result.ema_frequency);
    return result;
}

}  // namespace FarLib::cache
