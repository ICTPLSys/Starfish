#pragma once

// A deliberately small runtime adapter for the fixed-six capacity policy.
//
// The policy metadata lives in behavior_group_refinement.hpp, while this
// adapter owns stable physical-region records and the relaxed counters that
// feed one ten-second boundary.  Disabled mode must not touch either object
// or registry state: callers are expected to gate their hooks with enabled().

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "design2/behavior_group_refinement.hpp"
#include "design2/object_group_trace.hpp"

namespace FarLib::allocator::six_group {

namespace refinement = ::FarLib::cache::refinement;

using GroupId = std::uint32_t;

inline bool enabled() noexcept {
    static const bool value = [] {
#if defined(_WIN32)
        char *raw = nullptr;
        std::size_t length = 0;
        if (_dupenv_s(&raw, &length, "FARLIB_FIXED_SIX_GROUPS") != 0 ||
            raw == nullptr) {
            return false;
        }
        const bool enabled_value = std::string(raw) == "1";
        std::free(raw);
        return enabled_value;
#else
        const char *raw = std::getenv("FARLIB_FIXED_SIX_GROUPS");
        return raw != nullptr && std::string(raw) == "1";
#endif
    }();
    return value;
}

// The list-only ablation is deliberately independent of the fixed-six
// registry/runtime.  It only changes the physical usable-list shape: normal
// allocation routes to child zero and the other children remain available to
// the collector for traversal tests.  In particular, enabling this switch
// must not register Regions or create per-object/group metadata.
namespace list_only_six {

inline constexpr std::size_t kChildCount = 6;

// Exploratory structural mode: keep six independent intrusive lists per
// size/placement parent, without enabling the old fixed-six policy metadata.
// Regions are assigned round-robin by the parent list itself.
inline bool grouped() noexcept {
    static const bool value = [] {
#if defined(_WIN32)
        char *raw = nullptr;
        std::size_t length = 0;
        if (_dupenv_s(&raw, &length, "FARLIB_LIST_ONLY_SIX_GROUPS") != 0 ||
            raw == nullptr) {
            return false;
        }
        const bool enabled_value = std::string(raw) == "1";
        std::free(raw);
        return enabled_value;
#else
        const char *raw = std::getenv("FARLIB_LIST_ONLY_SIX_GROUPS");
        return raw != nullptr && std::string(raw) == "1";
#endif
    }();
    return value;
}

inline bool enabled() noexcept {
    static const bool value = [] {
#if defined(_WIN32)
        char *raw = nullptr;
        std::size_t length = 0;
        if (_dupenv_s(&raw, &length, "FARLIB_LIST_ONLY_SIX") != 0 ||
            raw == nullptr) {
            return false;
        }
        const bool enabled_value = std::string(raw) == "1";
        std::free(raw);
        return enabled_value;
#else
        const char *raw = std::getenv("FARLIB_LIST_ONLY_SIX");
        return raw != nullptr && std::string(raw) == "1";
#endif
    }();
    return value;
}

}  // namespace list_only_six

struct Record;
using Publish = void (*)(Record *, GroupId);
using WorkPhaseGuard = bool (*)();
using WorkerInit = void (*)();

enum class Event : std::uint8_t {
    Fetch = 0,
    CleanEvict = 1,
    DirtyEvict = 2,
    Free = 3,
};

struct Record {
    refinement::RegionToken token;
    bool remote = false;
    std::atomic<GroupId> group{0};
    // Publication/index maintenance only.  Event and demand counters never
    // acquire this mutex.
    std::mutex routing_mutex;

    std::atomic<std::uint64_t> primary_objects{0};
    std::atomic<std::uint64_t> primary_bytes{0};
    std::atomic<std::uint64_t> demand_reads{0};
    std::atomic<std::uint64_t> demand_writes{0};
    std::array<std::atomic<std::uint64_t>, 4> lifecycle{};
    // Object-time is object-count * elapsed nanoseconds for the current
    // classification window.  Keeping it per record avoids a global counter.
    std::atomic<std::uint64_t> object_time_ns{0};

    std::uint64_t owner = 0;
    std::uint32_t bin = 0;
    // The physical allocator bin is deliberately separate from the initial
    // family key.  Local and remote caches can use different physical bins
    // for the same six-group family.
    std::uint32_t family_bin = 0;
    // Cached stable family IDs avoid a policy metadata lookup on allocation
    // and are refreshed only when a physical region is (re)registered.
    std::array<GroupId, 6> family_groups{};
    Publish publish = nullptr;

    Record(refinement::RegionToken region_token, bool is_remote,
           std::uint64_t logical_owner, std::uint32_t physical_bin,
           std::uint32_t logical_family_bin, Publish callback)
        : token(region_token),
          remote(is_remote),
          owner(logical_owner),
          bin(physical_bin),
          family_bin(logical_family_bin),
          publish(callback) {
        for (auto &counter : lifecycle) counter.store(0, std::memory_order_relaxed);
    }

    Record(const Record &) = delete;
    Record &operator=(const Record &) = delete;
};

struct GroupDistribution {
    GroupId group = 0;
    std::uint8_t family_slot = 0;
    std::uint64_t family_owner = 0;
    std::uint32_t family_bin = 0;
    std::uint64_t regions = 0;
    std::uint64_t objects = 0;
    std::uint64_t bytes = 0;
    std::uint64_t dirty_evidence_regions = 0;
    std::uint64_t dirty_unknown_regions = 0;
    std::uint64_t dirty_evidence_objects = 0;
    std::uint64_t dirty_unknown_objects = 0;
};

struct BoundarySummary {
    std::uint64_t boundary = 0;
    std::array<GroupDistribution, 6> six_groups{};
    std::uint64_t unknown_regions = 0;
    std::uint64_t unknown_objects = 0;
    std::uint64_t unknown_bytes = 0;
    std::uint64_t deferred_regions = 0;
    std::uint64_t moved_regions = 0;
    std::uint64_t moved_groups = 0;
    std::uint64_t classification_failures = 0;
    std::uint64_t active_window_ns = 0;
    std::uint64_t classification_duration_ns = 0;
    // Unlike six_groups, this retains one entry per actual stable group ID,
    // including multiple owner/bin families that share the six slots.
    std::vector<GroupDistribution> distributions;
};

class Registry {
public:
    Registry() = default;
    Registry(const Registry &) = delete;
    Registry &operator=(const Registry &) = delete;
    ~Registry() { stop(); }

    GroupId initial_group(std::uint32_t owner, std::uint32_t bin) {
        if (!enabled()) return 0;
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        ensure_policy_locked();
        const auto id = policy_->ensure_fixed_six_group(
            refinement::InitialGroupKey{owner, bin},
            refinement::kFixedSixBootstrapClass);
        return narrow_group(id);
    }

    std::array<GroupId, 6> group_family(GroupId group) {
        std::array<GroupId, 6> result{};
        if (!enabled() || group == 0) return result;
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        ensure_policy_locked();
        const auto family = policy_->ensure_fixed_six_group_family(group);
        for (std::size_t i = 0; i < family.size(); ++i)
            result[i] = narrow_group(family[i]);
        return result;
    }

    Record *register_region(std::uintptr_t address, bool remote,
                            std::uint32_t bin, GroupId preferred,
                            Publish publish) {
        if (!enabled() || address == 0) return nullptr;
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        ensure_policy_locked();

        const PhysicalKey physical_key{static_cast<std::uintptr_t>(address),
                                       remote};
        const auto existing_it = records_by_physical_.find(physical_key);
        if (existing_it != records_by_physical_.end()) {
            // RegionHead is cleared on physical reuse, but the physical
            // identity is not.  Return the original Record and retain its
            // token; an explicit preferred group may still request a
            // key/class rebind while the caller holds the empty-region
            // guarantee.
            Record *record = existing_it->second;
            if (publish != nullptr) record->publish = publish;
            if (preferred != 0) {
                const auto snapshot = policy_->group_snapshot(
                    static_cast<refinement::GroupId>(preferred));
                if (!snapshot.has_value()) return nullptr;
                const auto result = policy_->rebind_region(
                    record->token, snapshot->initial_key,
                    static_cast<refinement::GroupId>(preferred));
                if (!result.success()) return nullptr;
                record->owner = snapshot->initial_key.logical_owner_id;
                record->family_bin = static_cast<std::uint32_t>(
                    snapshot->initial_key.allocator_size_bin);
                record->family_groups = narrow_family(
                    policy_->ensure_fixed_six_group_family(
                        snapshot->initial_key));
                record->bin = bin;
                publish_locked(record, narrow_group(result.current_group_id));
            } else {
                record->bin = bin;
            }
            return record;
        }

        const refinement::RegionToken token{
            static_cast<std::uint64_t>(address), next_region_id_++};
        // `bin` identifies the physical cache bin.  When a caller supplies a
        // preferred group, its snapshot is the authoritative family key: a
        // remote 8KiB bin and a local 10KiB bin may intentionally share one
        // behavior family.  Falling back to a fresh key is retained for
        // first registration without a bootstrap group.
        refinement::InitialGroupKey key{};
        std::uint64_t logical_owner = 0;
        std::uint32_t logical_family_bin = bin;
        if (preferred != 0) {
            const auto snapshot = policy_->group_snapshot(
                static_cast<refinement::GroupId>(preferred));
            if (!snapshot.has_value()) return nullptr;
            key = snapshot->initial_key;
            logical_owner = key.logical_owner_id;
            logical_family_bin = static_cast<std::uint32_t>(
                key.allocator_size_bin);
        } else {
            // Unpreferred registration is the bootstrap path.  All such
            // regions in one physical bin intentionally share owner zero;
            // callers that need a different family pass its preferred
            // stable group explicitly.
            logical_owner = 0;
            key = refinement::InitialGroupKey{logical_owner, bin};
        }
        const auto result = policy_->register_region(
            token, key, static_cast<refinement::GroupId>(preferred));
        if (!result.success()) return nullptr;
        const GroupId current = narrow_group(result.current_group_id);
        auto record = std::make_unique<Record>(token, remote, logical_owner,
                                                bin, logical_family_bin, publish);
        Record *raw = record.get();
        raw->group.store(current, std::memory_order_release);
        raw->family_groups = narrow_family(
            policy_->ensure_fixed_six_group_family(key));
        records_.push_back(std::move(record));
        records_by_token_.emplace(token, raw);
        records_by_physical_.emplace(physical_key, raw);
        publish_locked(raw, current);
        return raw;
    }

    // The caller guarantees that the physical Region is empty and has no
    // remote backup.  The stable token and Record survive the key/class update.
    void rebind_empty(Record *record, std::uint32_t bin,
                      GroupId preferred) {
        if (!enabled() || record == nullptr) return;
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        ensure_policy_locked();
        refinement::InitialGroupKey key{record->owner, record->family_bin};
        std::uint32_t logical_family_bin = record->family_bin;
        if (preferred != 0) {
            const auto snapshot = policy_->group_snapshot(
                static_cast<refinement::GroupId>(preferred));
            if (!snapshot.has_value()) return;
            key = snapshot->initial_key;
            logical_family_bin = static_cast<std::uint32_t>(
                key.allocator_size_bin);
        }
        const auto result = policy_->rebind_region(
            record->token, key, static_cast<refinement::GroupId>(preferred));
        if (!result.success()) return;
        record->bin = bin;
        record->owner = key.logical_owner_id;
        record->family_bin = logical_family_bin;
        record->family_groups = narrow_family(
            policy_->ensure_fixed_six_group_family(key));
        publish_locked(record, narrow_group(result.current_group_id));
    }

    void start(std::uint64_t local_capacity_bytes,
               WorkPhaseGuard work_phase_guard = nullptr,
               WorkerInit worker_init = nullptr) {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lock(worker_mutex_);
        local_capacity_bytes_ = local_capacity_bytes;
        work_phase_guard_.store(work_phase_guard, std::memory_order_release);
        worker_init_.store(worker_init, std::memory_order_release);
        if (worker_.joinable()) return;
        {
            std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
            ensure_policy_locked();
        }
        stop_requested_.store(false, std::memory_order_release);
        running_.store(true, std::memory_order_release);
        worker_ = std::thread([this] { worker_loop(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            if (!worker_.joinable()) {
                running_.store(false, std::memory_order_release);
                return;
            }
            stop_requested_.store(true, std::memory_order_release);
        }
        worker_wakeup_.notify_all();
        worker_.join();
        running_.store(false, std::memory_order_release);
    }

    void dump() {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        std::cout << "six_group.records\t" << records_.size() << '\n';
        std::uint64_t objects = 0;
        std::uint64_t bytes = 0;
        for (const auto &record : records_) {
            objects = saturated_add(objects,
                                    record->primary_objects.load(std::memory_order_relaxed));
            bytes = saturated_add(bytes,
                                  record->primary_bytes.load(std::memory_order_relaxed));
        }
        std::cout << "six_group.primary_objects\t" << objects << '\n'
                  << "six_group.primary_bytes\t" << bytes << '\n'
                  << "six_group.boundaries\t" << boundaries_.size() << '\n'
                  << "six_group.classification_failures\t"
                  << classification_failures_.load(std::memory_order_relaxed)
                  << '\n'
                  << "six_group.counter_underflows\t"
                  << counter_underflows_.load(std::memory_order_relaxed)
                  << '\n';
        for (const auto &boundary : boundaries_) {
            std::cout << "six_group.boundary\t" << boundary.boundary
                      << "\tactive_window_ns=" << boundary.active_window_ns
                      << "\tclassification_ns="
                      << boundary.classification_duration_ns
                      << "\tdeferred=" << boundary.deferred_regions
                      << "\tmoved_regions=" << boundary.moved_regions
                      << "\tmoved_groups=" << boundary.moved_groups
                      << "\tunknown=" << boundary.unknown_regions << '\n';
            for (const auto &group : boundary.distributions) {
                std::cout << "six_group.distribution\t" << group.group
                          << "\tfamily_slot="
                          << static_cast<unsigned>(group.family_slot)
                          << "\tfamily_owner=" << group.family_owner
                          << "\tfamily_bin=" << group.family_bin
                          << "\tregions=" << group.regions
                          << "\tobjects=" << group.objects
                          << "\tbytes=" << group.bytes
                          << "\tdirty_evidence_regions="
                          << group.dirty_evidence_regions
                          << "\tdirty_unknown_regions="
                          << group.dirty_unknown_regions
                          << "\tdirty_evidence_objects="
                          << group.dirty_evidence_objects
                          << "\tdirty_unknown_objects="
                          << group.dirty_unknown_objects << '\n';
            }
        }
    }

    std::size_t boundary_count() const {
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        return boundaries_.size();
    }

    BoundarySummary latest_boundary() const {
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        return boundaries_.empty() ? BoundarySummary{} : boundaries_.back();
    }

    std::uint64_t classification_failures() const noexcept {
        return classification_failures_.load(std::memory_order_relaxed);
    }

    GroupId group_of(const Record *record) const noexcept {
        return record == nullptr ? 0
                                 : record->group.load(std::memory_order_acquire);
    }

    void note_demand(Record *record, bool write) noexcept {
        if (!enabled() || record == nullptr || !work_phase_active()) return;
        auto &counter = write ? record->demand_writes : record->demand_reads;
        counter.fetch_add(1, std::memory_order_relaxed);
    }

    void note_event(Record *record, Event event) noexcept {
        // Initialization/validation history is not active-work evidence.
        if (!enabled() || record == nullptr || !work_phase_active()) return;
        record->lifecycle[static_cast<std::size_t>(event)].fetch_add(
            1, std::memory_order_relaxed);
    }

    void note_eviction(Record *source, Record *destination,
                       bool dirty) noexcept {
        if (!enabled() || (source == nullptr && destination == nullptr) ||
            !work_phase_active())
            return;
        const auto index = static_cast<std::size_t>(
            dirty ? Event::DirtyEvict : Event::CleanEvict);
        if (source != nullptr)
            source->lifecycle[index].fetch_add(1, std::memory_order_relaxed);
        if (destination != nullptr && destination != source)
            destination->lifecycle[index].fetch_add(1,
                                                    std::memory_order_relaxed);
    }

    void bind_primary(std::atomic<Record *> &entry_binding, Record *record,
                      std::uint64_t bytes) noexcept {
        if (!enabled() || record == nullptr) return;
        Record *old = entry_binding.exchange(record, std::memory_order_acq_rel);
        if (old == record) return;
        if (old != nullptr) decrement_primary(old, bytes);
        record->primary_objects.fetch_add(1, std::memory_order_relaxed);
        record->primary_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    void clear_primary(std::atomic<Record *> &entry_binding,
                       std::uint64_t bytes) noexcept {
        if (!enabled()) return;
        Record *record = entry_binding.exchange(nullptr, std::memory_order_acq_rel);
        if (record == nullptr) return;
        decrement_primary(record, bytes);
    }

private:
    struct PhysicalKey {
        std::uintptr_t address = 0;
        bool remote = false;

        bool operator==(const PhysicalKey &other) const noexcept {
            return address == other.address && remote == other.remote;
        }
    };

    struct PhysicalKeyHash {
        std::size_t operator()(const PhysicalKey &key) const noexcept {
            const auto address_hash = std::hash<std::uintptr_t>{}(key.address);
            const auto remote_hash = std::hash<bool>{}(key.remote);
            return address_hash ^ (remote_hash + static_cast<std::size_t>(
                                       0x9e3779b9U) +
                                   (address_hash << 6) + (address_hash >> 2));
        }
    };

    static constexpr std::uint64_t kBoundaryNs = 10'000'000'000ULL;
    static constexpr std::uint64_t kSampleNs = 10'000'000ULL;

    static std::uint64_t saturated_add(std::uint64_t lhs,
                                       std::uint64_t rhs) noexcept {
        return rhs > std::numeric_limits<std::uint64_t>::max() - lhs
                   ? std::numeric_limits<std::uint64_t>::max()
                   : lhs + rhs;
    }

    static std::uint64_t saturated_mul(std::uint64_t lhs,
                                       std::uint64_t rhs) noexcept {
        if (lhs == 0 || rhs == 0) return 0;
        return lhs > std::numeric_limits<std::uint64_t>::max() / rhs
                   ? std::numeric_limits<std::uint64_t>::max()
                   : lhs * rhs;
    }

    void decrement_counter(std::atomic<std::uint64_t> &counter,
                           std::uint64_t amount) noexcept {
        auto current = counter.load(std::memory_order_relaxed);
        while (true) {
            if (current < amount) {
                if (counter.compare_exchange_weak(
                        current, 0, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    counter_underflows_.fetch_add(1,
                                                  std::memory_order_relaxed);
                    return;
                }
                continue;
            }
            const auto next = current - amount;
            if (counter.compare_exchange_weak(current, next,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed))
                return;
        }
    }

    static GroupId narrow_group(refinement::GroupId group) {
        if (group > std::numeric_limits<GroupId>::max())
            throw std::overflow_error("fixed-six group id exceeds uint32");
        return static_cast<GroupId>(group);
    }

    static std::array<GroupId, 6> narrow_family(
        const std::array<refinement::GroupId, 6> &family) {
        std::array<GroupId, 6> result{};
        for (std::size_t i = 0; i < family.size(); ++i)
            result[i] = narrow_group(family[i]);
        return result;
    }

    static refinement::RegistryConfig make_policy_config(
        std::uint64_t local_capacity_bytes) {
        refinement::RegistryConfig config;
        config.max_regions = 1'000'000;
        config.max_groups = 1'000'000;
        config.enable_merge = false;
        config.record_capacity_details = false;
        config.capacity_grouping.enabled = true;
        config.capacity_grouping.local_capacity_bytes = local_capacity_bytes;
        config.capacity_grouping.dirty_low_per_million = 330'000;
        config.capacity_grouping.dirty_high_per_million = 670'000;
        config.capacity_grouping.dynamic_split_merge = false;
        config.capacity_grouping.fixed_six_groups = true;
        config.capacity_grouping.carry_dirty_class = false;
        config.persistent_range.enabled = false;
        return config;
    }

    void ensure_policy_locked() {
        if (!policy_) {
            policy_ = std::make_unique<refinement::BehaviorGroupRegistry>(
                make_policy_config(local_capacity_bytes_));
        }
    }

    static void add_object_time(Record *record, std::uint64_t previous_objects,
                                std::uint64_t current_objects,
                                std::uint64_t delta_ns) noexcept {
        if (record == nullptr || delta_ns == 0) return;
        // Trapezoidal integration: (previous + current) * dt / 2.  Split the
        // sum before multiplication so a large, but valid, interval does not
        // spuriously saturate at twice the intended area.
        const auto sum = saturated_add(previous_objects, current_objects);
        const auto half_sum_area = saturated_mul(sum / 2, delta_ns);
        const auto odd_sum_area = (sum & 1U) == 0
                                      ? 0
                                      : delta_ns / 2;
        const auto increment = saturated_add(half_sum_area, odd_sum_area);
        auto current = record->object_time_ns.load(std::memory_order_relaxed);
        while (true) {
            const auto next = saturated_add(current, increment);
            if (record->object_time_ns.compare_exchange_weak(
                    current, next, std::memory_order_relaxed,
                    std::memory_order_relaxed))
                return;
        }
    }

    void sample_object_time(std::uint64_t now_ns,
                            std::uint64_t &last_sample_ns) {
        if (now_ns <= last_sample_ns) return;
        const auto delta = now_ns - last_sample_ns;
        last_sample_ns = now_ns;
        std::vector<Record *> records;
        {
            std::lock_guard<std::mutex> lock(metadata_mutex_);
            records.reserve(records_.size());
            for (const auto &record : records_) records.push_back(record.get());
        }
        for (Record *record : records) {
            const auto current_objects = record->primary_objects.load(
                std::memory_order_relaxed);
            const auto previous_it = previous_sample_objects_.find(record);
            if (previous_it == previous_sample_objects_.end()) {
                previous_sample_objects_.emplace(record, current_objects);
                continue;
            }
            add_object_time(record, previous_it->second, current_objects, delta);
            previous_it->second = current_objects;
        }
    }

    bool work_phase_active() const noexcept {
        const auto guard = work_phase_guard_.load(std::memory_order_acquire);
        return guard == nullptr || guard();
    }

    void publish_locked(Record *record, GroupId group) {
        if (record == nullptr) return;
        std::lock_guard<std::mutex> route_lock(record->routing_mutex);
        {
            // Sampled event timestamps and this group publication share only
            // a diagnostic lock; the allocator callback runs after releasing it.
            auto trace_guard = ::FarLib::object_group_trace::lock_region(
                record->token.generation_id);
            record->group.store(group, std::memory_order_release);
            if (::FarLib::object_group_trace::enabled()) {
                const auto slot = std::find(record->family_groups.begin(),
                                            record->family_groups.end(), group);
                ::FarLib::object_group_trace::route_change(
                    record->token.generation_id, group,
                    slot == record->family_groups.end() ? 255 :
                        static_cast<std::uint8_t>(slot - record->family_groups.begin()),
                    record->remote, record->owner, record->family_bin);
            }
        }
        if (record->publish != nullptr) record->publish(record, group);
    }

    void decrement_primary(Record *record, std::uint64_t bytes) noexcept {
        decrement_counter(record->primary_objects, 1);
        decrement_counter(record->primary_bytes, bytes);
    }

    void classify_boundary(std::uint64_t boundary,
                           std::uint64_t active_window_ns) {
        const auto classification_start = std::chrono::steady_clock::now();
        std::vector<Record *> records;
        {
            std::lock_guard<std::mutex> lock(metadata_mutex_);
            records.reserve(records_.size());
            for (const auto &record : records_) records.push_back(record.get());
        }

        std::vector<refinement::RegionObservation> observations;
        observations.reserve(records.size());
        for (Record *record : records) {
            refinement::RegionProfile profile;
            profile.complete_window = true;
            profile.fetch_commit = record->lifecycle[0].exchange(
                0, std::memory_order_acq_rel);
            profile.clean_evict_commit = record->lifecycle[1].exchange(
                0, std::memory_order_acq_rel);
            profile.dirty_evict_commit = record->lifecycle[2].exchange(
                0, std::memory_order_acq_rel);
            profile.free_commit = record->lifecycle[3].exchange(
                0, std::memory_order_acq_rel);
            profile.demand_read_reference = record->demand_reads.exchange(
                0, std::memory_order_acq_rel);
            profile.demand_write_reference = record->demand_writes.exchange(
                0, std::memory_order_acq_rel);
            profile.live_object_exposure = record->object_time_ns.exchange(
                0, std::memory_order_acq_rel);
            profile.primary_census_known = true;
            profile.primary_live_objects = record->primary_objects.load(
                std::memory_order_acquire);
            profile.primary_live_bytes = record->primary_bytes.load(
                std::memory_order_acquire);
            profile.primary_object_time_ns = profile.live_object_exposure;
            observations.push_back({record->token, profile});
        }

        std::unordered_map<refinement::RegionToken,
                           refinement::RegionProfile,
                           refinement::RegionTokenHash>
            boundary_profiles;
        boundary_profiles.reserve(observations.size());
        for (const auto &observation : observations)
            boundary_profiles.emplace(observation.token, observation.profile);

        refinement::BoundaryResult result;
        try {
            std::lock_guard<std::mutex> lock(metadata_mutex_);
            ensure_policy_locked();
            result = policy_->refine_boundary(boundary, observations);
            // Keep registration/rebind and publication serialized with the
            // policy update.  Otherwise an empty-region rebind could race a
            // boundary move and publish a stale group after the rebind.
            for (const auto &move : result.capacity_reassignments) {
                const auto it = records_by_token_.find(move.token);
                if (it != records_by_token_.end()) {
                    publish_locked(it->second, narrow_group(move.new_group_id));
                }
            }

            BoundarySummary summary;
            summary.boundary = boundary;
            summary.active_window_ns = active_window_ns;
            if (!result.capacity_boundaries.empty()) {
                const auto &record = result.capacity_boundaries.back();
                summary.deferred_regions = record.deferred_region_count;
                summary.moved_regions = record.moved_region_count;
                summary.moved_groups = record.moved_group_count;
                summary.unknown_regions = record.unknown_region_count;
            }
            summary.moved_regions = std::max<std::uint64_t>(
                summary.moved_regions, result.capacity_reassignments.size());

            std::uint64_t observed_unknown_regions = 0;
            std::uint64_t observed_unknown_objects = 0;
            std::uint64_t observed_unknown_bytes = 0;
            for (Record *record : records) {
                const GroupId group = record->group.load(
                    std::memory_order_acquire);
                const auto objects = record->primary_objects.load(
                    std::memory_order_relaxed);
                const auto bytes = record->primary_bytes.load(
                    std::memory_order_relaxed);
                const auto family_slot = std::find(record->family_groups.begin(),
                                                   record->family_groups.end(),
                                                   group);
                if (group == 0 || family_slot == record->family_groups.end()) {
                    observed_unknown_regions = saturated_add(
                        observed_unknown_regions, 1);
                    observed_unknown_objects = saturated_add(
                        observed_unknown_objects, objects);
                    observed_unknown_bytes = saturated_add(
                        observed_unknown_bytes, bytes);
                    continue;
                }
                const auto slot_index = static_cast<std::size_t>(
                    std::distance(record->family_groups.begin(), family_slot));
                auto &compact_slot = summary.six_groups[slot_index];
                if (compact_slot.group == 0) compact_slot.group = group;
                if (compact_slot.group != group) compact_slot.group = 0;
                ++compact_slot.regions;
                compact_slot.objects = saturated_add(compact_slot.objects,
                                                    objects);
                compact_slot.bytes = saturated_add(compact_slot.bytes, bytes);

                auto distribution = std::find_if(
                    summary.distributions.begin(), summary.distributions.end(),
                    [group](const GroupDistribution &candidate) {
                        return candidate.group == group;
                    });
                if (distribution == summary.distributions.end()) {
                    GroupDistribution candidate;
                    candidate.group = group;
                    candidate.family_slot = static_cast<std::uint8_t>(
                        slot_index);
                    candidate.family_owner = record->owner;
                    candidate.family_bin = record->family_bin;
                    summary.distributions.push_back(candidate);
                    distribution = std::prev(summary.distributions.end());
                }
                ++distribution->regions;
                distribution->objects = saturated_add(distribution->objects,
                                                     objects);
                distribution->bytes = saturated_add(distribution->bytes,
                                                    bytes);
                const auto profile_it = boundary_profiles.find(record->token);
                const bool dirty_evidence =
                    profile_it != boundary_profiles.end() &&
                    (profile_it->second.clean_evict_commit != 0 ||
                     profile_it->second.dirty_evict_commit != 0);
                if (dirty_evidence) {
                    distribution->dirty_evidence_regions = saturated_add(
                        distribution->dirty_evidence_regions, 1);
                    distribution->dirty_evidence_objects = saturated_add(
                        distribution->dirty_evidence_objects, objects);
                } else {
                    distribution->dirty_unknown_regions = saturated_add(
                        distribution->dirty_unknown_regions, 1);
                    distribution->dirty_unknown_objects = saturated_add(
                        distribution->dirty_unknown_objects, objects);
                }
            }
            summary.unknown_regions = std::max(summary.unknown_regions,
                                               observed_unknown_regions);
            summary.unknown_objects = std::max(summary.unknown_objects,
                                               observed_unknown_objects);
            summary.unknown_bytes = std::max(summary.unknown_bytes,
                                             observed_unknown_bytes);
            summary.classification_duration_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - classification_start)
                    .count());
            boundaries_.push_back(summary);
        } catch (const std::exception &error) {
            classification_failures_.fetch_add(1, std::memory_order_relaxed);
            std::cerr << "fixed-six classification failed: " << error.what()
                      << '\n';
            return;
        }
    }

    void worker_loop() {
        using clock = std::chrono::steady_clock;
        const auto worker_init = worker_init_.load(std::memory_order_acquire);
        if (worker_init != nullptr) worker_init();
        const auto initial_now = clock::now();
        std::uint64_t last_sample_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                initial_now.time_since_epoch())
                .count());
        bool active_window = false;
        std::uint64_t active_elapsed_ns = 0;
        std::uint64_t boundary = 1;
        while (!stop_requested_.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> wake_lock(worker_wait_mutex_);
            worker_wakeup_.wait_for(wake_lock,
                                    std::chrono::nanoseconds(kSampleNs), [this] {
                                        return stop_requested_.load(
                                            std::memory_order_acquire);
                                    });
            wake_lock.unlock();
            const auto now = clock::now();
            const auto now_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now.time_since_epoch()).count());
            if (!work_phase_active()) {
                // Do not charge inactive wall time to object exposure.  Keep
                // the next active sample's origin current without allocating
                // or touching policy metadata.
                last_sample_ns = now_ns;
                active_window = false;
                previous_sample_objects_.clear();
                continue;
            }
            if (!active_window) {
                active_window = true;
                last_sample_ns = now_ns;
                continue;
            }
            const auto sample_start_ns = last_sample_ns;
            sample_object_time(now_ns, last_sample_ns);
            active_elapsed_ns = saturated_add(
                active_elapsed_ns, now_ns - sample_start_ns);
            while (active_elapsed_ns >= kBoundaryNs) {
                active_elapsed_ns -= kBoundaryNs;
                classify_boundary(boundary++, kBoundaryNs);
            }
        }
    }

    mutable std::mutex metadata_mutex_;
    std::unique_ptr<refinement::BehaviorGroupRegistry> policy_;
    std::unordered_map<refinement::RegionToken, Record *,
                       refinement::RegionTokenHash>
        records_by_token_;
    std::unordered_map<PhysicalKey, Record *, PhysicalKeyHash>
        records_by_physical_;
    std::vector<std::unique_ptr<Record>> records_;
    std::vector<BoundarySummary> boundaries_;
    std::atomic<std::uint64_t> classification_failures_{0};
    std::uint64_t next_region_id_ = 1;
    std::uint64_t local_capacity_bytes_ = 0;

    std::mutex worker_mutex_;
    std::mutex worker_wait_mutex_;
    std::condition_variable worker_wakeup_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<WorkPhaseGuard> work_phase_guard_{nullptr};
    std::atomic<WorkerInit> worker_init_{nullptr};
    std::unordered_map<Record *, std::uint64_t> previous_sample_objects_;
    std::atomic<std::uint64_t> counter_underflows_{0};
};

inline Registry &registry() {
    static Registry instance;
    return instance;
}

inline GroupId group_of(const Record *record) noexcept {
    return registry().group_of(record);
}

inline void note_demand(Record *record, bool write) noexcept {
    registry().note_demand(record, write);
}

inline void note_event(Record *record, Event event) noexcept {
    registry().note_event(record, event);
}

inline void note_eviction(Record *source, Record *destination,
                          bool dirty) noexcept {
    registry().note_eviction(source, destination, dirty);
}

inline void bind_primary(std::atomic<Record *> &entry_binding, Record *record,
                         std::uint64_t bytes) noexcept {
    registry().bind_primary(entry_binding, record, bytes);
}

inline void clear_primary(std::atomic<Record *> &entry_binding,
                          std::uint64_t bytes) noexcept {
    registry().clear_primary(entry_binding, bytes);
}

} // namespace FarLib::allocator::six_group
