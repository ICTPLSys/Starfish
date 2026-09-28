#pragma once

// Observation-only dirty evidence for the simple Region experiment.  This
// header intentionally has no cache or allocator dependency: cache code passes
// committed local/remote addresses and the metadata visible at that point.
//
// The observer is deliberately bounded.  Region counters are one packed
// atomic (low 32 bits clean events, high 32 bits dirty events) per physical
// Region slot.  The object table keeps metadata and counters only; it never
// retains an Entry, allocator descriptor, RDMA handle, or other object pointer.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace FarLib::simple_dirty_observer {

enum class Mode : std::uint8_t { Off = 0, Regions = 1, Both = 2 };

namespace detail {

constexpr std::uint64_t kSamplingSeed = 20260917ULL;
constexpr std::uint32_t kDefaultObjectSampleShift = 12;
constexpr std::uint32_t kMinimumObjectSampleShift = 8;
constexpr std::uint32_t kMaximumObjectSampleShift = 20;
constexpr std::size_t kObjectSlotCapacity = 16384;
constexpr std::uint32_t kInvalidRegion = 0x00ffffffU;

// These bounds keep an accidentally large configured heap from turning a
// diagnostic into an unbounded allocation.  They still cover 4 GiB at 4 KiB
// Regions and 64 GiB at 512 KiB Regions, and are larger than the experiment's
// usual 256 KiB/512 KiB physical Region arrays.
constexpr std::size_t kMaximumLocalRegionSlots = 1U << 20;
constexpr std::size_t kMaximumRemoteRegionSlots = 1U << 20;

inline std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

inline Mode parse_mode() {
    const char *raw = std::getenv("FARLIB_SIMPLE_DIRTY_OBSERVE");
    if (raw == nullptr || *raw == '\0' || std::strcmp(raw, "0") == 0)
        return Mode::Off;
    if (std::strcmp(raw, "regions") == 0)
        return Mode::Regions;
    if (std::strcmp(raw, "both") == 0)
        return Mode::Both;
    throw std::invalid_argument(
        "FARLIB_SIMPLE_DIRTY_OBSERVE must be 0, regions, or both");
}

inline Mode configured_mode() {
    static const Mode value = parse_mode();
    return value;
}

inline std::uint32_t parse_object_shift() {
    const char *raw = std::getenv("FARLIB_SIMPLE_DIRTY_OBJECT_SHIFT");
    if (raw == nullptr || *raw == '\0') return kDefaultObjectSampleShift;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' ||
        parsed < kMinimumObjectSampleShift ||
        parsed > kMaximumObjectSampleShift) {
        throw std::invalid_argument(
            "FARLIB_SIMPLE_DIRTY_OBJECT_SHIFT must be an integer in [8,20]");
    }
    return static_cast<std::uint32_t>(parsed);
}

inline std::string output_prefix() {
    const char *raw = std::getenv("FARLIB_SIMPLE_DIRTY_OBSERVE_PREFIX");
    return raw == nullptr || *raw == '\0' ? std::string("simple_dirty_observe")
                                          : std::string(raw);
}

inline std::size_t ceil_slots(std::size_t bytes, std::size_t region_bytes) {
    if (bytes == 0) return 0;
    return (bytes - 1) / region_bytes + 1;
}

inline std::uint32_t clamp_region(std::size_t index) noexcept {
    return index > static_cast<std::size_t>(kInvalidRegion)
               ? kInvalidRegion
               : static_cast<std::uint32_t>(index);
}

inline std::uint32_t packed32_clean(std::uint64_t value) noexcept {
    return static_cast<std::uint32_t>(value & 0xffffffffULL);
}

inline std::uint32_t packed32_dirty(std::uint64_t value) noexcept {
    return static_cast<std::uint32_t>(value >> 32U);
}

inline std::uint16_t packed16(std::uint64_t value, unsigned lane) noexcept {
    return static_cast<std::uint16_t>((value >> (lane * 16U)) & 0xffffULL);
}

inline void increment_packed32(std::atomic<std::uint64_t> &counter,
                               bool dirty) noexcept {
    const std::uint64_t lane = dirty ? (1ULL << 32U) : 1ULL;
    const std::uint64_t mask = dirty ? 0xffffffff00000000ULL : 0xffffffffULL;
    std::uint64_t old = counter.load(std::memory_order_relaxed);
    for (;;) {
        const std::uint32_t current = dirty
                                           ? static_cast<std::uint32_t>(old >> 32U)
                                           : static_cast<std::uint32_t>(old);
        if (current == std::numeric_limits<std::uint32_t>::max()) return;
        const std::uint64_t next = (old & ~mask) | ((old & mask) + lane);
        if (counter.compare_exchange_weak(old, next,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed))
            return;
    }
}

inline void increment_packed16(std::atomic<std::uint64_t> &counter,
                               unsigned lane) noexcept {
    const std::uint64_t mask = 0xffffULL << (lane * 16U);
    std::uint64_t old = counter.load(std::memory_order_relaxed);
    for (;;) {
        const std::uint16_t current = packed16(old, lane);
        if (current == std::numeric_limits<std::uint16_t>::max()) return;
        const std::uint64_t next =
            (old & ~mask) | ((old & mask) + (1ULL << (lane * 16U)));
        if (counter.compare_exchange_weak(old, next,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed))
            return;
    }
}

inline const char *mode_name(Mode mode) noexcept {
    switch (mode) {
    case Mode::Regions: return "regions";
    case Mode::Both: return "both";
    default: return "0";
    }
}

inline const char *class_name(std::uint64_t clean, std::uint64_t dirty,
                              bool *confident = nullptr,
                              const char **evidence = nullptr,
                              bool require_evidence = true) noexcept {
    const std::uint64_t total = clean + dirty;
    if (confident != nullptr) *confident = total >= 8;
    if (evidence != nullptr) {
        if (total == 0)
            *evidence = "no_evidence";
        else if (total < 8)
            *evidence = "insufficient";
        else
            *evidence = "sufficient";
    }
    if (total == 0 || (require_evidence && total < 8)) return "unknown";
    // 4/15 and 11/15 are the low/high histogram boundaries.  Cross-multiply
    // to keep the decision deterministic without floating-point rounding.
    if (dirty * 15ULL <= total * 4ULL) return "low";
    if (dirty * 15ULL >= total * 11ULL) return "high";
    return "medium";
}

inline const char *runtime_class(std::uint8_t q, bool known) noexcept {
    if (!known) return "unknown";
    if (q <= 4) return "low";
    if (q >= 11) return "high";
    return "medium";
}

} // namespace detail

inline bool enabled() { return detail::configured_mode() != Mode::Off; }
inline bool regions_enabled() { return detail::configured_mode() != Mode::Off; }
inline bool objects_enabled() { return detail::configured_mode() == Mode::Both; }

class Monitor {
    struct RegionTotals {
        std::uint64_t clean = 0;
        std::uint64_t dirty = 0;
    };

    // One cache-line-independent metadata record per possible sampled slot.
    // The record contains no pointer to the cache object.
    struct ObjectSlot {
        std::atomic<bool> claimed{false};
        std::uint64_t ordinal = 0;
        std::uint64_t bytes = 0;
        // local region [0,23], remote region [24,47], domain [48,49], q
        // [50,57], known [58], hot [59], alive [60].
        std::atomic<std::uint64_t> location{0};
        // Accesses: reads in low 32 bits, writes in high 32 bits.
        std::atomic<std::uint64_t> accesses{0};
        // Evicts: work-clean, work-dirty, init-clean, init-dirty, four 16-bit
        // lanes.  Counts saturate rather than carry into another class.
        std::atomic<std::uint64_t> evicts{0};
    };

    using Clock = std::chrono::steady_clock;

    const Mode mode_ = detail::configured_mode();
    std::mutex snapshot_mutex_;
    std::function<std::uint8_t(bool, std::size_t)> class_reader_;

    bool configured_ = false;
    std::atomic<bool> active_{false};
    std::atomic<bool> started_{false};
    bool had_begin_ = false;
    bool final_emitted_ = false;
    bool summary_emitted_ = false;
    std::uint64_t last_window_ = 0;
    std::uint64_t begin_ns_ = 0;
    Clock::time_point epoch_ = Clock::now();
    Clock::time_point begin_time_ = epoch_;

    uintptr_t local_base_ = 0;
    std::size_t local_bytes_ = 0;
    std::size_t local_region_bytes_ = 1;
    std::size_t remote_bytes_ = 0;
    std::size_t remote_region_bytes_ = 1;
    std::size_t local_slots_ = 0;
    std::size_t remote_slots_ = 0;
    std::size_t local_capacity_slots_ = 0;
    std::size_t remote_capacity_slots_ = 0;

    std::unique_ptr<std::atomic<std::uint64_t>[]> local_pending_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> remote_pending_;
    std::unique_ptr<RegionTotals[]> local_totals_;
    std::unique_ptr<RegionTotals[]> remote_totals_;

    std::array<ObjectSlot, detail::kObjectSlotCapacity + 1> objects_{};
    std::atomic<std::uint64_t> creation_ordinal_{0};
    std::atomic<std::uint32_t> next_slot_{1};
    std::atomic<std::uint64_t> eligible_allocations_{0};
    std::atomic<std::uint64_t> sampled_objects_{0};
    std::atomic<std::uint64_t> sample_saturation_{0};
    std::uint32_t object_sample_shift_ = detail::kDefaultObjectSampleShift;

    std::atomic<std::uint64_t> unmapped_local_{0};
    std::atomic<std::uint64_t> unmapped_remote_{0};
    std::uint64_t callback_errors_ = 0;
    std::uint64_t counter_saturated_rows_ = 0;
    std::uint64_t counter_saturated_snapshots_ = 0;

    std::ofstream region_output_;
    std::ofstream object_output_;
    std::string prefix_;

    std::uint64_t elapsed_ns(Clock::time_point now) const noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - epoch_)
                .count());
    }

    static std::uint64_t monotonic_ns(Clock::time_point now) noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch())
                .count());
    }

    std::uint64_t work_elapsed_ns(Clock::time_point now) const noexcept {
        if (!had_begin_) return 0;
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - begin_time_)
                .count());
    }

    std::size_t local_region(uintptr_t address) const noexcept {
        if (local_slots_ == 0 || address < local_base_) return detail::kInvalidRegion;
        const std::size_t offset = static_cast<std::size_t>(address - local_base_);
        if (offset >= local_bytes_) return detail::kInvalidRegion;
        const std::size_t index = offset / local_region_bytes_;
        return index < local_capacity_slots_ ? index : detail::kInvalidRegion;
    }

    std::size_t remote_region(std::uint64_t address) const noexcept {
        if (remote_slots_ == 0 || address >= remote_bytes_)
            return detail::kInvalidRegion;
        const std::size_t index = static_cast<std::size_t>(address / remote_region_bytes_);
        return index < remote_capacity_slots_ ? index : detail::kInvalidRegion;
    }

    static std::uint64_t pack_location(std::uint32_t local,
                                       std::uint32_t remote,
                                       std::uint8_t domain, std::uint8_t q,
                                       bool known, bool hot, bool alive) noexcept {
        return (static_cast<std::uint64_t>(local) & 0x00ffffffULL) |
               ((static_cast<std::uint64_t>(remote) & 0x00ffffffULL) << 24U) |
               ((static_cast<std::uint64_t>(domain) & 0x3ULL) << 48U) |
               ((static_cast<std::uint64_t>(q) & 0xffULL) << 50U) |
               (static_cast<std::uint64_t>(known) << 58U) |
               (static_cast<std::uint64_t>(hot) << 59U) |
               (static_cast<std::uint64_t>(alive) << 60U);
    }

    static std::uint32_t location_local(std::uint64_t value) noexcept {
        return static_cast<std::uint32_t>(value & 0x00ffffffULL);
    }
    static std::uint32_t location_remote(std::uint64_t value) noexcept {
        return static_cast<std::uint32_t>((value >> 24U) & 0x00ffffffULL);
    }
    static std::uint8_t location_domain(std::uint64_t value) noexcept {
        return static_cast<std::uint8_t>((value >> 48U) & 0x3ULL);
    }
    static std::uint8_t location_q(std::uint64_t value) noexcept {
        return static_cast<std::uint8_t>((value >> 50U) & 0xffULL);
    }
    static bool location_known(std::uint64_t value) noexcept {
        return ((value >> 58U) & 1ULL) != 0;
    }
    static bool location_hot(std::uint64_t value) noexcept {
        return ((value >> 59U) & 1ULL) != 0;
    }
    static bool location_alive(std::uint64_t value) noexcept {
        return ((value >> 60U) & 1ULL) != 0;
    }

    ObjectSlot *object(std::uint16_t slot) noexcept {
        if (slot == 0 || slot > detail::kObjectSlotCapacity) return nullptr;
        ObjectSlot &record = objects_[slot];
        return record.claimed.load(std::memory_order_acquire) ? &record : nullptr;
    }
    const ObjectSlot *object(std::uint16_t slot) const noexcept {
        if (slot == 0 || slot > detail::kObjectSlotCapacity) return nullptr;
        const ObjectSlot &record = objects_[slot];
        return record.claimed.load(std::memory_order_acquire) ? &record : nullptr;
    }

    void update_location(ObjectSlot &record, std::size_t local,
                         std::size_t remote, std::uint8_t domain,
                         std::uint8_t q, bool known, bool hot, bool alive,
                         bool replace_local, bool replace_remote) noexcept {
        const std::uint64_t old = record.location.load(std::memory_order_relaxed);
        const std::uint32_t old_local = location_local(old);
        const std::uint32_t old_remote = location_remote(old);
        const std::uint32_t next_local =
            replace_local && local != detail::kInvalidRegion
                ? detail::clamp_region(local)
                : old_local;
        const std::uint32_t next_remote =
            replace_remote && remote != detail::kInvalidRegion
                ? detail::clamp_region(remote)
                : old_remote;
        record.location.store(pack_location(next_local, next_remote, domain, q,
                                             known, hot, alive),
                              std::memory_order_release);
    }

    void note_region(std::size_t local, std::size_t remote, bool dirty) noexcept {
        if (!active_.load(std::memory_order_acquire) ||
            mode_ == Mode::Off) return;
        if (local != detail::kInvalidRegion && local < local_capacity_slots_)
            detail::increment_packed32(local_pending_[local], dirty);
        else
            unmapped_local_.fetch_add(1, std::memory_order_relaxed);
        if (remote != detail::kInvalidRegion && remote < remote_capacity_slots_)
            detail::increment_packed32(remote_pending_[remote], dirty);
        else
            unmapped_remote_.fetch_add(1, std::memory_order_relaxed);
    }

    void reset_measurement_counters() noexcept {
        for (std::size_t i = 0; i < local_capacity_slots_; ++i) {
            local_pending_[i].store(0, std::memory_order_relaxed);
            local_totals_[i] = {};
        }
        for (std::size_t i = 0; i < remote_capacity_slots_; ++i) {
            remote_pending_[i].store(0, std::memory_order_relaxed);
            remote_totals_[i] = {};
        }
        for (std::size_t i = 1; i <= detail::kObjectSlotCapacity; ++i) {
            if (!objects_[i].claimed.load(std::memory_order_acquire)) continue;
            objects_[i].accesses.store(0, std::memory_order_relaxed);
            objects_[i].evicts.fetch_and(0xffffffff00000000ULL,
                                         std::memory_order_relaxed);
        }
        unmapped_local_.store(0, std::memory_order_relaxed);
        unmapped_remote_.store(0, std::memory_order_relaxed);
    }

    void reset_object_table() noexcept {
        creation_ordinal_.store(0, std::memory_order_relaxed);
        next_slot_.store(1, std::memory_order_relaxed);
        eligible_allocations_.store(0, std::memory_order_relaxed);
        sampled_objects_.store(0, std::memory_order_relaxed);
        sample_saturation_.store(0, std::memory_order_relaxed);
        for (std::size_t i = 1; i <= detail::kObjectSlotCapacity; ++i) {
            objects_[i].claimed.store(false, std::memory_order_relaxed);
            objects_[i].ordinal = 0;
            objects_[i].bytes = 0;
            objects_[i].location.store(0, std::memory_order_relaxed);
            objects_[i].accesses.store(0, std::memory_order_relaxed);
            objects_[i].evicts.store(0, std::memory_order_relaxed);
        }
    }

    void write_region_header() {
        if (!region_output_) return;
        region_output_
            << "window\tpartial\tmonotonic_ns\tbegin_ns\telapsed_ns\twork_duration_ns"
               "\tdomain\tremote\tendpoint\tregion_index\tarray_index\tactive\thas_evidence"
               "\tclean_events\tdirty_events\twindow_clean_events\twindow_dirty_events\tcumulative_clean_events"
               "\tcumulative_dirty_events\tdirty_fraction\tdirty_class"
               "\tclass_confident\tevidence\tsupply_class\tcounter_saturated\n";
    }

    void write_object_header() {
        if (!object_output_) return;
        object_output_
            << "window\tpartial\tmonotonic_ns\tbegin_ns\telapsed_ns\twork_duration_ns"
               "\tsample_slot\tslot\tcreation_ordinal\tsize_bytes\tbytes\talive\tcurrent_domain"
               "\tlocal_region\tremote_region\tq\tq_known\thot\truntime_class"
               "\tknown\treads\twrites\taccesses\twork_clean_evictions\twork_dirty_evictions"
               "\tclean_evictions\tdirty_evictions\tinit_clean_evictions\tinit_dirty_evictions\tinit_evictions\tempirical_dirty_class"
               "\tempirical_class_confident\tempirical_evidence\tlocal_supply_class"
               "\tremote_supply_class\tcounter_saturated\n";
    }

    std::uint8_t supply_class(bool remote, std::size_t index) noexcept {
        if (!class_reader_ || index == detail::kInvalidRegion) return 0;
        try {
            return class_reader_(remote, index);
        } catch (...) {
            ++callback_errors_;
            return 0;
        }
    }

    void write_region_rows(std::uint64_t window, bool partial,
                           std::uint64_t monotonic, std::uint64_t elapsed,
                           std::uint64_t work, std::uint64_t &local_clean,
                           std::uint64_t &local_dirty,
                           std::uint64_t &remote_clean,
                           std::uint64_t &remote_dirty,
                           std::uint64_t &local_active,
                           std::uint64_t &remote_active) {
        for (std::size_t i = 0; i < local_capacity_slots_; ++i) {
            const std::uint64_t value =
                local_pending_[i].exchange(0, std::memory_order_acq_rel);
            const std::uint64_t clean = detail::packed32_clean(value);
            const std::uint64_t dirty = detail::packed32_dirty(value);
            local_totals_[i].clean += clean;
            local_totals_[i].dirty += dirty;
            local_clean += clean;
            local_dirty += dirty;
            if (clean != 0 || dirty != 0) ++local_active;
            if (!region_output_ || (clean == 0 && dirty == 0 &&
                                    local_totals_[i].clean == 0 &&
                                    local_totals_[i].dirty == 0))
                continue;
            bool confident = false;
            const char *evidence = nullptr;
            const char *klass = detail::class_name(
                local_totals_[i].clean, local_totals_[i].dirty, &confident,
                &evidence, false);
            const std::uint64_t total = local_totals_[i].clean +
                                        local_totals_[i].dirty;
            const bool saturated =
                clean == std::numeric_limits<std::uint32_t>::max() ||
                dirty == std::numeric_limits<std::uint32_t>::max();
            if (saturated) ++counter_saturated_rows_;
            const double fraction = total == 0
                                        ? 0.0
                                        : static_cast<double>(local_totals_[i].dirty) /
                                              static_cast<double>(total);
            region_output_ << window << '\t' << (partial ? 1 : 0) << '\t'
                           << monotonic << '\t' << begin_ns_ << '\t' << elapsed
                           << '\t' << work << "\tlocal\t0\tlocal\t" << i << '\t' << i
                           << '\t' << ((clean != 0 || dirty != 0) ? 1 : 0)
                           << '\t' << (total != 0 ? 1 : 0) << '\t' << clean << '\t'
                           << dirty << '\t' << clean << '\t' << dirty << '\t'
                           << local_totals_[i].clean << '\t'
                           << local_totals_[i].dirty << '\t' << std::setprecision(17)
                           << fraction << '\t' << klass << '\t' << (confident ? 1 : 0)
                           << '\t' << evidence << '\t' << static_cast<unsigned>(
                                  supply_class(false, i))
                           << '\t' << (saturated ? 1 : 0)
                           << '\n';
        }
        for (std::size_t i = 0; i < remote_capacity_slots_; ++i) {
            const std::uint64_t value =
                remote_pending_[i].exchange(0, std::memory_order_acq_rel);
            const std::uint64_t clean = detail::packed32_clean(value);
            const std::uint64_t dirty = detail::packed32_dirty(value);
            remote_totals_[i].clean += clean;
            remote_totals_[i].dirty += dirty;
            remote_clean += clean;
            remote_dirty += dirty;
            if (clean != 0 || dirty != 0) ++remote_active;
            if (!region_output_ || (clean == 0 && dirty == 0 &&
                                    remote_totals_[i].clean == 0 &&
                                    remote_totals_[i].dirty == 0))
                continue;
            bool confident = false;
            const char *evidence = nullptr;
            const char *klass = detail::class_name(
                remote_totals_[i].clean, remote_totals_[i].dirty, &confident,
                &evidence, false);
            const std::uint64_t total = remote_totals_[i].clean +
                                        remote_totals_[i].dirty;
            const bool saturated =
                clean == std::numeric_limits<std::uint32_t>::max() ||
                dirty == std::numeric_limits<std::uint32_t>::max();
            if (saturated) ++counter_saturated_rows_;
            const double fraction = total == 0
                                        ? 0.0
                                        : static_cast<double>(remote_totals_[i].dirty) /
                                              static_cast<double>(total);
            region_output_ << window << '\t' << (partial ? 1 : 0) << '\t'
                           << monotonic << '\t' << begin_ns_ << '\t' << elapsed
                           << '\t' << work << "\tremote\t1\tremote\t" << i << '\t' << i
                           << '\t' << ((clean != 0 || dirty != 0) ? 1 : 0)
                           << '\t' << (total != 0 ? 1 : 0) << '\t' << clean << '\t'
                           << dirty << '\t' << clean << '\t' << dirty << '\t'
                           << remote_totals_[i].clean << '\t'
                           << remote_totals_[i].dirty << '\t' << std::setprecision(17)
                           << fraction << '\t' << klass << '\t' << (confident ? 1 : 0)
                           << '\t' << evidence << '\t' << static_cast<unsigned>(
                                  supply_class(true, i))
                           << '\t' << (saturated ? 1 : 0)
                           << '\n';
        }
    }

    static const char *domain_name(std::uint8_t domain) noexcept {
        if (domain == 1) return "local";
        if (domain == 2) return "remote";
        return "none";
    }

    void write_object_rows(std::uint64_t window, bool partial,
                           std::uint64_t monotonic, std::uint64_t elapsed,
                           std::uint64_t work) {
        if (!object_output_) return;
        for (std::size_t i = 1; i <= detail::kObjectSlotCapacity; ++i) {
            const ObjectSlot &record = objects_[i];
            if (!record.claimed.load(std::memory_order_acquire)) continue;
            const std::uint64_t location =
                record.location.load(std::memory_order_acquire);
            const std::uint64_t accesses =
                record.accesses.load(std::memory_order_relaxed);
            const std::uint64_t evicts = record.evicts.load(std::memory_order_relaxed);
            const std::uint64_t work_clean = detail::packed16(evicts, 0);
            const std::uint64_t work_dirty = detail::packed16(evicts, 1);
            const std::uint64_t init_clean = detail::packed16(evicts, 2);
            const std::uint64_t init_dirty = detail::packed16(evicts, 3);
            const bool saturated =
                detail::packed32_clean(accesses) ==
                    std::numeric_limits<std::uint32_t>::max() ||
                detail::packed32_dirty(accesses) ==
                    std::numeric_limits<std::uint32_t>::max() ||
                work_clean == std::numeric_limits<std::uint16_t>::max() ||
                work_dirty == std::numeric_limits<std::uint16_t>::max() ||
                init_clean == std::numeric_limits<std::uint16_t>::max() ||
                init_dirty == std::numeric_limits<std::uint16_t>::max();
            if (saturated) ++counter_saturated_rows_;
            bool confident = false;
            const char *evidence = nullptr;
            const char *klass = detail::class_name(work_clean, work_dirty,
                                                   &confident, &evidence);
            object_output_ << window << '\t' << (partial ? 1 : 0) << '\t'
                           << monotonic << '\t' << begin_ns_ << '\t' << elapsed
                           << '\t' << work << '\t' << i << '\t' << i << '\t'
                           << record.ordinal << '\t' << record.bytes << '\t'
                           << record.bytes << '\t'
                           << (location_alive(location) ? 1 : 0) << '\t'
                           << domain_name(location_domain(location)) << '\t'
                           << ((location_local(location) == detail::kInvalidRegion)
                                   ? -1
                                   : static_cast<long long>(location_local(location)))
                           << '\t'
                           << ((location_remote(location) == detail::kInvalidRegion)
                                   ? -1
                                   : static_cast<long long>(location_remote(location)))
                           << '\t' << static_cast<unsigned>(location_q(location))
                           << '\t' << (location_known(location) ? 1 : 0) << '\t'
                           << (location_hot(location) ? 1 : 0) << '\t'
                           << detail::runtime_class(location_q(location),
                                                    location_known(location))
                           << '\t' << (location_known(location) ? 1 : 0)
                           << '\t' << detail::packed32_clean(accesses) << '\t'
                           << detail::packed32_dirty(accesses) << '\t'
                           << (static_cast<std::uint64_t>(detail::packed32_clean(accesses)) +
                               detail::packed32_dirty(accesses))
                           << '\t' << work_clean << '\t' << work_dirty
                           << '\t' << work_clean << '\t' << work_dirty
                           << '\t' << init_clean << '\t' << init_dirty
                           << '\t' << (init_clean + init_dirty) << '\t' << klass << '\t'
                           << (confident ? 1 : 0) << '\t' << evidence << '\t'
                           << static_cast<unsigned>(supply_class(
                                  false, location_local(location)))
                           << '\t'
                           << static_cast<unsigned>(supply_class(
                                  true, location_remote(location)))
                           << '\t' << (saturated ? 1 : 0)
                           << '\n';
        }
    }

    void snapshot_impl(std::uint64_t window, bool partial) {
        const Clock::time_point started = Clock::now();
        const std::uint64_t monotonic = monotonic_ns(started);
        const std::uint64_t elapsed = work_elapsed_ns(started);
        std::uint64_t local_clean = 0, local_dirty = 0, remote_clean = 0,
                      remote_dirty = 0, local_active = 0, remote_active = 0;
        if (mode_ != Mode::Off) {
            write_region_rows(window, partial, monotonic, elapsed, elapsed,
                              local_clean, local_dirty, remote_clean,
                              remote_dirty, local_active, remote_active);
            write_object_rows(window, partial, monotonic, elapsed, elapsed);
        }
        if (region_output_) region_output_.flush();
        if (object_output_) object_output_.flush();
        const std::uint64_t overhead = monotonic_ns(Clock::now()) - monotonic;
        if (counter_saturated_rows_ != 0) ++counter_saturated_snapshots_;
        std::cerr << "simple_dirty_observe.window index=" << window
                  << " partial=" << (partial ? 1 : 0)
                  << " monotonic_ns=" << monotonic
                  << " begin_ns=" << begin_ns_ << " elapsed_ns=" << elapsed
                  << " work_duration_ns=" << elapsed
                  << " local_clean=" << local_clean
                  << " local_dirty=" << local_dirty
                  << " remote_clean=" << remote_clean
                  << " remote_dirty=" << remote_dirty
                  << " local_active=" << local_active
                  << " remote_active=" << remote_active
                  << " eligible_allocations="
                  << eligible_allocations_.load(std::memory_order_relaxed)
                  << " sampled_objects="
                  << sampled_objects_.load(std::memory_order_relaxed)
                  << " sample_saturation="
                  << sample_saturation_.load(std::memory_order_relaxed)
                  << " counter_saturated="
                  << (counter_saturated_rows_ != 0 ? 1 : 0)
                  << " counter_saturated_rows=" << counter_saturated_rows_
                  << " snapshot_overhead_ns=" << overhead
                  << " unmapped_local="
                  << unmapped_local_.load(std::memory_order_relaxed)
                  << " unmapped_remote="
                  << unmapped_remote_.load(std::memory_order_relaxed) << '\n';
        last_window_ = window;
    }

    void emit_summary_locked() {
        if (summary_emitted_) return;
        const std::size_t region_memory =
            (local_capacity_slots_ + remote_capacity_slots_) * sizeof(std::uint64_t) +
            (local_capacity_slots_ + remote_capacity_slots_) * sizeof(RegionTotals);
        const std::size_t object_memory = objects_.size() * sizeof(ObjectSlot);
        std::cerr << "simple_dirty_observe.summary mode="
                  << detail::mode_name(mode_)
                  << " creations="
                  << eligible_allocations_.load(std::memory_order_relaxed)
                  << " sampled="
                  << sampled_objects_.load(std::memory_order_relaxed)
                  << " sample_saturation="
                  << sample_saturation_.load(std::memory_order_relaxed)
                  << " callback_errors=" << callback_errors_
                  << " unmapped_local="
                  << unmapped_local_.load(std::memory_order_relaxed)
                  << " unmapped_remote="
                  << unmapped_remote_.load(std::memory_order_relaxed)
                  << " counter_saturated="
                  << (counter_saturated_rows_ != 0 ? 1 : 0)
                  << " counter_saturated_rows=" << counter_saturated_rows_
                  << " counter_saturated_snapshots="
                  << counter_saturated_snapshots_
                  << " region_counter_memory_bytes=" << region_memory
                  << " object_metadata_memory_bytes=" << object_memory << '\n';
        summary_emitted_ = true;
    }

public:
    Monitor() = default;
    ~Monitor() { end(); }

    void configure(uintptr_t local_base, std::size_t local_bytes,
                   std::size_t local_region_bytes, std::size_t remote_bytes,
                   std::size_t remote_region_bytes) {
        if (mode_ == Mode::Off) return;
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        if (active_.load(std::memory_order_acquire))
            throw std::logic_error("simple dirty observer configure while active");
        if (local_region_bytes == 0 || (remote_bytes != 0 && remote_region_bytes == 0))
            throw std::invalid_argument("simple dirty observer requires nonzero Region size");

        local_base_ = local_base;
        local_bytes_ = local_bytes;
        local_region_bytes_ = local_region_bytes;
        remote_bytes_ = remote_bytes;
        remote_region_bytes_ = remote_region_bytes == 0 ? 1 : remote_region_bytes;
        local_slots_ = detail::ceil_slots(local_bytes_, local_region_bytes_);
        remote_slots_ = detail::ceil_slots(remote_bytes_, remote_region_bytes_);
        local_capacity_slots_ =
            std::min(local_slots_, detail::kMaximumLocalRegionSlots);
        remote_capacity_slots_ =
            std::min(remote_slots_, detail::kMaximumRemoteRegionSlots);

        if (mode_ == Mode::Both) object_sample_shift_ = detail::parse_object_shift();
        if (region_output_) region_output_.close();
        if (object_output_) object_output_.close();
        reset_object_table();
        local_pending_.reset();
        remote_pending_.reset();
        local_totals_.reset();
        remote_totals_.reset();
        if (local_capacity_slots_ != 0) {
            local_pending_ = std::make_unique<std::atomic<std::uint64_t>[]>(
                local_capacity_slots_);
            local_totals_ = std::make_unique<RegionTotals[]>(local_capacity_slots_);
            for (std::size_t i = 0; i < local_capacity_slots_; ++i)
                local_pending_[i].store(0, std::memory_order_relaxed);
        }
        if (remote_capacity_slots_ != 0) {
            remote_pending_ = std::make_unique<std::atomic<std::uint64_t>[]>(
                remote_capacity_slots_);
            remote_totals_ = std::make_unique<RegionTotals[]>(remote_capacity_slots_);
            for (std::size_t i = 0; i < remote_capacity_slots_; ++i)
                remote_pending_[i].store(0, std::memory_order_relaxed);
        }

        configured_ = true;
        had_begin_ = false;
        started_.store(false, std::memory_order_release);
        final_emitted_ = false;
        summary_emitted_ = false;
        last_window_ = 0;
        begin_ns_ = 0;
        counter_saturated_rows_ = 0;
        counter_saturated_snapshots_ = 0;
        epoch_ = Clock::now();
        begin_time_ = epoch_;
        prefix_ = detail::output_prefix();
        if (mode_ == Mode::Regions || mode_ == Mode::Both) {
            region_output_.open(prefix_ + ".regions.tsv", std::ios::out | std::ios::trunc);
            if (!region_output_)
                throw std::runtime_error("cannot open simple dirty Region output");
            write_region_header();
        }
        if (mode_ == Mode::Both) {
            object_output_.open(prefix_ + ".objects.tsv", std::ios::out | std::ios::trunc);
            if (!object_output_)
                throw std::runtime_error("cannot open simple dirty object output");
            write_object_header();
        }

        const std::size_t region_memory =
            (local_capacity_slots_ + remote_capacity_slots_) * sizeof(std::uint64_t) +
            (local_capacity_slots_ + remote_capacity_slots_) * sizeof(RegionTotals);
        const std::size_t object_memory =
            objects_.size() * sizeof(ObjectSlot);
        std::cerr << "simple_dirty_observe.config mode="
                  << detail::mode_name(mode_)
                  << " sampling_algorithm=splitmix64_topbits_v1"
                  << " sampling_seed=" << detail::kSamplingSeed
                  << " sample_shift=" << object_sample_shift_
                  << " sample_probability=1/" << (1ULL << object_sample_shift_)
                  << " object_slot_cap=" << detail::kObjectSlotCapacity
                  << " local_slots=" << local_slots_
                  << " remote_slots=" << remote_slots_
                  << " local_slots_tracked=" << local_capacity_slots_
                  << " remote_slots_tracked=" << remote_capacity_slots_
                  << " region_counter_memory_bytes=" << region_memory
                  << " object_metadata_memory_bytes=" << object_memory
                  << " region_counter_width=packed32_clean_dirty_events"
                  << " region_counts_are_events=1 occupancy_bytes=unavailable\n";
    }

    void set_class_reader(std::function<std::uint8_t(bool, std::size_t)> reader) {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        if (active_.load(std::memory_order_acquire))
            throw std::logic_error("simple dirty observer class reader while active");
        class_reader_ = std::move(reader);
    }

    std::uint16_t register_object(std::size_t bytes, uintptr_t local_addr,
                                  std::uint8_t q, bool known, bool hot) {
        if (mode_ != Mode::Both) return 0;
        const std::uint64_t ordinal =
            creation_ordinal_.fetch_add(1, std::memory_order_relaxed) + 1;
        eligible_allocations_.fetch_add(1, std::memory_order_relaxed);
        const std::uint64_t hash = detail::splitmix64(
            detail::kSamplingSeed ^ ordinal);
        if ((hash >> (64U - object_sample_shift_)) != 0) return 0;

        const std::uint32_t candidate =
            next_slot_.fetch_add(1, std::memory_order_relaxed);
        if (candidate == 0 || candidate > detail::kObjectSlotCapacity) {
            sample_saturation_.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        ObjectSlot &record = objects_[candidate];
        record.ordinal = ordinal;
        record.bytes = bytes;
        record.accesses.store(0, std::memory_order_relaxed);
        record.evicts.store(0, std::memory_order_relaxed);
        const std::size_t local = local_region(local_addr);
        record.location.store(
            pack_location(detail::clamp_region(local), detail::kInvalidRegion,
                          local == detail::kInvalidRegion ? 0 : 1, q, known, hot,
                          true),
            std::memory_order_relaxed);
        record.claimed.store(true, std::memory_order_release);
        sampled_objects_.fetch_add(1, std::memory_order_relaxed);
        return static_cast<std::uint16_t>(candidate);
    }

    void access(std::uint16_t slot, bool write) noexcept {
        if (!active_.load(std::memory_order_acquire)) return;
        ObjectSlot *record = object(slot);
        if (record == nullptr) return;
        detail::increment_packed32(record->accesses, write);
    }

    void fetch(std::uint16_t slot, uintptr_t local_addr, std::uint8_t q,
               bool known, bool hot) noexcept {
        ObjectSlot *record = object(slot);
        if (record == nullptr) return;
        const std::size_t local = local_region(local_addr);
        update_location(*record, local, detail::kInvalidRegion,
                        local == detail::kInvalidRegion ? 0 : 1, q, known, hot,
                        true, true, false);
    }

    void evict(std::uint16_t slot, uintptr_t local_addr,
               std::uint64_t remote_addr, std::size_t /*bytes*/, bool dirty,
               std::uint8_t q, bool known, bool hot) noexcept {
        ObjectSlot *record = object(slot);
        const std::size_t local = local_region(local_addr);
        const std::size_t remote = remote_region(remote_addr);
        // Object location/evidence is retained even before begin() or after
        // active observation has ended: this preserves cold-remote residency.
        if (record != nullptr) {
            const bool work = active_.load(std::memory_order_acquire);
            update_location(*record, local, remote,
                            remote == detail::kInvalidRegion ? 1 : 2, q, known,
                            hot, true, true, true);
            // Initialization evidence belongs only to the pre-first-begin
            // phase.  Events racing with/after end() still update residency,
            // but must not be relabeled as initialization or work evidence.
            if (work || !started_.load(std::memory_order_acquire)) {
                detail::increment_packed16(
                    record->evicts,
                    dirty ? (work ? 1U : 3U) : (work ? 0U : 2U));
            }
        }
        note_region(local, remote, dirty);
    }

    void free_object(std::uint16_t slot) noexcept {
        ObjectSlot *record = object(slot);
        if (record == nullptr) return;
        const std::uint64_t old = record->location.load(std::memory_order_relaxed);
        record->location.store(
            pack_location(location_local(old), location_remote(old),
                          location_domain(old), location_q(old),
                          location_known(old), location_hot(old), false),
            std::memory_order_release);
    }

    void begin() {
        if (mode_ == Mode::Off) return;
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        if (!configured_) throw std::logic_error("simple dirty observer not configured");
        if (active_.load(std::memory_order_acquire))
            throw std::logic_error("simple dirty observer already active");
        reset_measurement_counters();
        begin_time_ = Clock::now();
        begin_ns_ = monotonic_ns(begin_time_);
        had_begin_ = true;
        started_.store(true, std::memory_order_release);
        final_emitted_ = false;
        last_window_ = 0;
        active_.store(true, std::memory_order_release);
    }

    void snapshot(std::uint64_t window, bool partial = false) {
        if (mode_ == Mode::Off) return;
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        if (!configured_ || !had_begin_) return;
        snapshot_impl(window, partial);
    }

    void end() {
        if (mode_ == Mode::Off) return;
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        if (!configured_ || !had_begin_) {
            active_.store(false, std::memory_order_release);
            if (configured_) emit_summary_locked();
            return;
        }
        const bool was_active = active_.exchange(false, std::memory_order_acq_rel);
        if (was_active && !final_emitted_) {
            snapshot_impl(last_window_ + 1, true);
            final_emitted_ = true;
        }
        emit_summary_locked();
        if (region_output_) {
            region_output_.flush();
            region_output_.close();
        }
        if (object_output_) {
            object_output_.flush();
            object_output_.close();
        }
    }

    bool active() const noexcept { return active_.load(std::memory_order_acquire); }

    // Read-only diagnostic/test view; order is work clean/dirty, init clean/dirty.
    std::array<std::uint64_t,4> eviction_counts(std::uint16_t slot) const noexcept {
        const auto *record=object(slot);
        if (!record) return {};
        const auto packed=record->evicts.load(std::memory_order_relaxed);
        return {detail::packed16(packed,0),detail::packed16(packed,1),
                detail::packed16(packed,2),detail::packed16(packed,3)};
    }
};

inline Monitor &monitor() {
    static Monitor value;
    return value;
}

inline std::uint16_t register_object(std::size_t bytes, uintptr_t local_addr,
                                     std::uint8_t q, bool known, bool hot) {
    return monitor().register_object(bytes, local_addr, q, known, hot);
}
inline void access(std::uint16_t slot, bool write) {
    monitor().access(slot, write);
}
inline void fetch(std::uint16_t slot, uintptr_t local_addr, std::uint8_t q,
                  bool known, bool hot) {
    monitor().fetch(slot, local_addr, q, known, hot);
}
inline void evict(std::uint16_t slot, uintptr_t local_addr,
                  std::uint64_t remote_addr, std::size_t bytes, bool dirty,
                  std::uint8_t q, bool known, bool hot) {
    monitor().evict(slot, local_addr, remote_addr, bytes, dirty, q, known, hot);
}
inline void free_object(std::uint16_t slot) { monitor().free_object(slot); }
inline void begin() { monitor().begin(); }
inline void snapshot(std::uint64_t window, bool partial = false) {
    monitor().snapshot(window, partial);
}
inline void end() { monitor().end(); }

} // namespace FarLib::simple_dirty_observer
