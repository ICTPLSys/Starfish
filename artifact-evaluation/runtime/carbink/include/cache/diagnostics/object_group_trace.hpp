#pragma once

// Bounded sampled object trace.  This deliberately has no cache dependency:
// cache code supplies a physical region token and the group visible at the
// point of the event.  The trace is diagnostic only; callers must retain the
// returned region lock across the corresponding routing publication/event.

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace FarLib::object_group_trace {

enum class Kind : std::uint8_t {
    Allocate = 0,
    Bind = 1,
    Read = 2,
    Write = 3,
    Fetch = 4,
    CleanEvict = 5,
    DirtyEvict = 6,
    Free = 7,
};

namespace detail {

constexpr std::uint64_t kSeed = 20260917ULL;
constexpr std::uint64_t kTrackedBytes = 8192ULL;
constexpr std::uint32_t kDefaultSampleShift = 7;
constexpr std::size_t kDefaultPerObjectCap = 1048576;
constexpr std::size_t kDefaultTotalCapBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::size_t kBudgetChunk = 4096;

inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

inline std::uint64_t env_u64(const char* name, std::uint64_t fallback,
                             std::uint64_t maximum) noexcept {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed > maximum) return fallback;
    return parsed;
}

struct Event {
    std::uint64_t ns;
    std::uint64_t region;
    std::uint32_t group;
    Kind kind;
};

struct Record {
    const std::uint16_t slot;
    const std::uint64_t ordinal;
    const std::uint64_t bytes;
    const std::uint64_t owner;
    const std::uint32_t bin;
    std::mutex mutex;
    std::vector<Event> events;
    std::size_t budgeted_events = 0;
    std::size_t next_budget = 8;

    Record(std::uint16_t stable_slot, std::uint64_t allocation_ordinal,
           std::uint64_t allocation_bytes, std::uint64_t logical_owner,
           std::uint32_t physical_bin)
        : slot(stable_slot), ordinal(allocation_ordinal), bytes(allocation_bytes),
          owner(logical_owner), bin(physical_bin) {}
};

struct Route {
    std::uint64_t ns;
    std::uint64_t region;
    std::uint64_t owner;
    std::uint32_t group;
    std::uint32_t bin;
    std::uint8_t family_slot;
    bool remote;
};

struct Phase {
    std::uint64_t id;
    std::uint64_t start_ns;
    std::uint64_t end_ns;
};

struct State {
    const bool enabled;
    const std::uint32_t sample_shift;
    const std::size_t per_object_cap;
    const std::size_t total_cap_bytes;
    // Vector capacity can grow by nearly 2x.  Half the byte budget therefore
    // bounds actual event-vector allocation below the configured safety cap.
    const std::size_t total_event_cap;
    const std::string prefix;
    const std::chrono::steady_clock::time_point epoch;

    std::array<std::atomic<Record*>, 65536> slots{};
    std::atomic<std::uint64_t> eligible_allocations{0};
    std::atomic<std::uint32_t> next_slot{1};
    std::atomic<std::size_t> budgeted_events{0};
    std::atomic<std::uint64_t> dropped_events{0};
    std::atomic<std::uint64_t> cap_exceeded{0};
    std::atomic<std::uint64_t> slot_overflow{0};
    std::atomic<std::uint64_t> phase_errors{0};
    std::atomic<bool> output_error{false};
    std::atomic<bool> dumped{false};

    std::mutex metadata_mutex;
    std::vector<Record*> records;
    std::vector<Route> routes;
    std::vector<Phase> phases;
    std::uint64_t next_phase = 1;
    std::uint64_t active_phase = 0;
    std::uint64_t active_start_ns = 0;
    // Object and region diagnostic locks are deliberately separate.  Callers
    // that need a coherent binding snapshot always acquire object first,
    // then region; publication needs only region.  Neither lock is used by
    // cache/list/routing machinery.
    std::array<std::mutex, 128> object_mutexes;
    std::array<std::mutex, 128> region_mutexes;

    State()
        : enabled([] {
              const char* raw = std::getenv("FARLIB_OBJECT_GROUP_TRACE");
              return raw != nullptr && std::string(raw) == "1";
          }()),
          sample_shift(static_cast<std::uint32_t>(env_u64(
              "FARLIB_OBJECT_GROUP_TRACE_SAMPLE_SHIFT", kDefaultSampleShift, 63))),
          per_object_cap(static_cast<std::size_t>(env_u64(
              "FARLIB_OBJECT_GROUP_TRACE_PER_OBJECT_CAP", kDefaultPerObjectCap,
              std::numeric_limits<std::uint32_t>::max()))),
          total_cap_bytes(static_cast<std::size_t>(env_u64(
              "FARLIB_OBJECT_GROUP_TRACE_TOTAL_CAP_BYTES", kDefaultTotalCapBytes,
              std::numeric_limits<std::uint32_t>::max()))),
          total_event_cap((total_cap_bytes / sizeof(Event)) / 2),
          prefix([] {
              const char* raw = std::getenv("FARLIB_OBJECT_GROUP_TRACE_PREFIX");
              return raw == nullptr || *raw == '\0' ? std::string("object_group_trace")
                                                    : std::string(raw);
          }()),
          epoch(std::chrono::steady_clock::now()) {}
};

inline State& state() {
    static State value;
    return value;
}

inline std::uint64_t now_ns(const State& s) noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - s.epoch).count());
}

inline bool reserve_event_budget(State& s, Record& record) noexcept {
    if (record.events.size() < record.budgeted_events) return true;
    if (record.budgeted_events >= s.per_object_cap) {
        s.cap_exceeded.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::size_t wanted = std::min(record.next_budget,
                                        s.per_object_cap - record.budgeted_events);
    std::size_t current = s.budgeted_events.load(std::memory_order_relaxed);
    for (;;) {
        if (wanted > s.total_event_cap || current > s.total_event_cap - wanted) {
            s.cap_exceeded.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (s.budgeted_events.compare_exchange_weak(current, current + wanted,
                                                     std::memory_order_relaxed,
                                                     std::memory_order_relaxed)) {
            record.budgeted_events += wanted;
            record.next_budget = std::min(kBudgetChunk, record.next_budget * 2);
            return true;
        }
    }
}

inline void append(Record* record, Kind kind, std::uint64_t region,
                   std::uint32_t group) noexcept {
    if (record == nullptr) return;
    State& s = state();
    std::lock_guard<std::mutex> lock(record->mutex);
    // The timestamp deliberately occurs under this lock: dump sees events in
    // append order and per-object timestamps are nondecreasing (ties allowed).
    const std::uint64_t ns = now_ns(s);
    if (!reserve_event_budget(s, *record)) {
        s.dropped_events.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    try {
        record->events.push_back(Event{ns, region, group, kind});
    } catch (...) {
        // A failure must be visible to offline analysis; never silently omit it.
        s.dropped_events.fetch_add(1, std::memory_order_relaxed);
        s.cap_exceeded.fetch_add(1, std::memory_order_relaxed);
    }
}

inline bool write_header(std::ofstream& out, const char* header) {
    out << header << '\n';
    return static_cast<bool>(out);
}

inline bool invalid(const State& s, std::uint64_t unclosed) noexcept {
    return s.dropped_events.load(std::memory_order_relaxed) != 0 ||
           s.cap_exceeded.load(std::memory_order_relaxed) != 0 ||
           s.slot_overflow.load(std::memory_order_relaxed) != 0 ||
           s.phase_errors.load(std::memory_order_relaxed) != 0 || unclosed != 0 ||
           s.output_error.load(std::memory_order_relaxed);
}

}  // namespace detail

inline bool enabled() noexcept { return detail::state().enabled; }

// Exposed for deterministic tests and offline sampling audits.  Ordinal is
// one-based and only counts allocations whose byte size is exactly 8192.
inline bool would_sample(std::uint64_t ordinal) noexcept {
    const auto shift = detail::state().sample_shift;
    if (shift == 0) return true;
    const std::uint64_t hash = detail::splitmix64(detail::kSeed ^ ordinal);
    return (hash >> (64U - shift)) == 0;
}

inline std::unique_lock<std::mutex> lock_region(std::uint64_t region) {
    if (!enabled()) return {};
    return std::unique_lock<std::mutex>(detail::state().region_mutexes[region % 128U]);
}

// Serialize sampled-object binding publication with its reference snapshot.
// Required lock ordering is lock_object(slot), then lock_region(region), then
// the logger's private per-record mutex.  Slot zero is intentionally a no-op.
inline std::unique_lock<std::mutex> lock_object(std::uint16_t slot) {
    if (!enabled() || slot == 0) return {};
    return std::unique_lock<std::mutex>(detail::state().object_mutexes[slot % 128U]);
}

inline std::uint16_t register_object(std::uint64_t bytes, std::uint64_t owner,
                                     std::uint32_t bin, std::uint64_t region,
                                     std::uint32_t group) noexcept {
    if (!enabled() || bytes != detail::kTrackedBytes) return 0;
    detail::State& s = detail::state();
    const std::uint64_t ordinal = s.eligible_allocations.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!would_sample(ordinal)) return 0;
    const std::uint32_t candidate = s.next_slot.fetch_add(1, std::memory_order_relaxed);
    if (candidate == 0 || candidate > std::numeric_limits<std::uint16_t>::max()) {
        s.slot_overflow.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    detail::Record* record = nullptr;
    try {
        record = new detail::Record(static_cast<std::uint16_t>(candidate), ordinal,
                                    bytes, owner, bin);
        {
            std::lock_guard<std::mutex> lock(s.metadata_mutex);
            s.records.push_back(record);
        }
        s.slots[candidate].store(record, std::memory_order_release);
        detail::append(record, Kind::Allocate, region, group);
        return static_cast<std::uint16_t>(candidate);
    } catch (...) {
        delete record;
        s.dropped_events.fetch_add(1, std::memory_order_relaxed);
        s.cap_exceeded.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
}

inline void note(std::uint16_t slot, Kind kind, std::uint64_t region,
                 std::uint32_t group) noexcept {
    if (!enabled() || slot == 0) return;
    detail::append(detail::state().slots[slot].load(std::memory_order_acquire),
                   kind, region, group);
}

inline void route_change(std::uint64_t region, std::uint32_t group,
                         std::uint8_t family_slot, bool remote,
                         std::uint64_t owner, std::uint32_t bin) noexcept {
    if (!enabled()) return;
    detail::State& s = detail::state();
    try {
        // This serializes route rows.  The caller holds lock_region(region),
        // so this timestamp is ordered with the matching group publication.
        std::lock_guard<std::mutex> lock(s.metadata_mutex);
        s.routes.push_back(detail::Route{detail::now_ns(s), region, owner, group,
                                         bin, family_slot, remote});
    } catch (...) {
        s.dropped_events.fetch_add(1, std::memory_order_relaxed);
        s.cap_exceeded.fetch_add(1, std::memory_order_relaxed);
    }
}

inline std::uint64_t phase_start() noexcept {
    if (!enabled()) return 0;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.metadata_mutex);
    if (s.active_phase != 0) {
        s.phase_errors.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    s.active_phase = s.next_phase++;
    s.active_start_ns = detail::now_ns(s);
    return s.active_phase;
}

inline void phase_end() noexcept {
    if (!enabled()) return;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.metadata_mutex);
    if (s.active_phase == 0) {
        s.phase_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    try {
        s.phases.push_back(detail::Phase{s.active_phase, s.active_start_ns, detail::now_ns(s)});
    } catch (...) {
        s.phase_errors.fetch_add(1, std::memory_order_relaxed);
    }
    s.active_phase = 0;
    s.active_start_ns = 0;
}

inline bool dump() noexcept {
    if (!enabled()) return true;
    detail::State& s = detail::state();
    bool expected = false;
    if (!s.dumped.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return false;

    std::vector<detail::Record*> records;
    std::vector<detail::Route> routes;
    std::vector<detail::Phase> phases;
    std::uint64_t unclosed = 0;
    try {
        std::lock_guard<std::mutex> lock(s.metadata_mutex);
        records = s.records;
        routes = s.routes;
        phases = s.phases;
        if (s.active_phase != 0) {
            phases.push_back(detail::Phase{s.active_phase, s.active_start_ns, 0});
            unclosed = 1;
        }
    } catch (...) {
        s.output_error.store(true, std::memory_order_relaxed);
        std::cerr << "object_group_trace: failed to snapshot trace output\n";
        return false;
    }

    bool ok = true;
    std::uint64_t recorded_event_count = 0;
    auto fail = [&] { ok = false; s.output_error.store(true, std::memory_order_relaxed); };
    try {
        std::ofstream objects(s.prefix + ".objects.tsv");
        if (!objects || !detail::write_header(objects, "sample\tordinal\tbytes\towner\tbin")) fail();
        for (auto* record : records) {
            if (objects) objects << record->slot << '\t' << record->ordinal << '\t' << record->bytes
                                << '\t' << record->owner << '\t' << record->bin << '\n';
        }
        objects.flush();
        if (!objects) fail();
        objects.close();
        if (!objects) fail();

        std::ofstream events(s.prefix + ".events.tsv");
        if (!events || !detail::write_header(events, "sample\tns\tkind\tregion\tgroup")) fail();
        for (auto* record : records) {
            std::lock_guard<std::mutex> lock(record->mutex);
            for (const auto& event : record->events) {
                if (events) events << record->slot << '\t' << event.ns << '\t'
                                   << static_cast<unsigned>(event.kind) << '\t' << event.region
                                   << '\t' << event.group << '\n';
                ++recorded_event_count;
            }
        }
        events.flush();
        if (!events) fail();
        events.close();
        if (!events) fail();

        std::ofstream route_file(s.prefix + ".routes.tsv");
        if (!route_file || !detail::write_header(route_file,
                "ns\tregion\tgroup\tfamily_slot\tremote\towner\tbin")) fail();
        for (const auto& route : routes) {
            if (route_file) route_file << route.ns << '\t' << route.region << '\t' << route.group
                                       << '\t' << static_cast<unsigned>(route.family_slot) << '\t'
                                       << (route.remote ? 1 : 0) << '\t' << route.owner << '\t'
                                       << route.bin << '\n';
        }
        route_file.flush();
        if (!route_file) fail();
        route_file.close();
        if (!route_file) fail();

        std::ofstream phase_file(s.prefix + ".phases.tsv");
        if (!phase_file || !detail::write_header(phase_file, "phase\tstart_ns\tend_ns")) fail();
        for (const auto& phase : phases) {
            if (phase_file) phase_file << phase.id << '\t' << phase.start_ns << '\t'
                                       << phase.end_ns << '\n';
        }
        phase_file.flush();
        if (!phase_file) fail();
        phase_file.close();
        if (!phase_file) fail();

        std::ofstream status(s.prefix + ".status.tsv");
        if (!status || !detail::write_header(status, "key\tvalue")) fail();
        const auto drops = s.dropped_events.load(std::memory_order_relaxed);
        const auto cap = s.cap_exceeded.load(std::memory_order_relaxed);
        const auto overflow = s.slot_overflow.load(std::memory_order_relaxed);
        const auto phase_errors = s.phase_errors.load(std::memory_order_relaxed);
        const bool trace_invalid = detail::invalid(s, unclosed);
        if (status) {
            status << "enabled\t1\n"
                   << "sampling_algorithm\tsplitmix64_topbits_v1\n"
                   << "sampling_seed\t20260917\n"
                   << "eligible_allocations\t" << s.eligible_allocations.load(std::memory_order_relaxed) << '\n'
                   << "sampled_objects\t" << records.size() << '\n'
                   << "sample_shift\t" << s.sample_shift << '\n'
                   << "per_object_event_cap\t" << s.per_object_cap << '\n'
                   << "total_event_cap\t" << s.total_event_cap << '\n'
                   << "total_event_budget\t" << s.budgeted_events.load(std::memory_order_relaxed) << '\n'
                   << "recorded_events\t" << recorded_event_count << '\n'
                   << "dropped_events\t" << drops << '\n'
                   << "cap_exceeded\t" << cap << '\n'
                   << "slot_overflow\t" << overflow << '\n'
                   << "phases_unclosed\t" << unclosed << '\n'
                   << "phase_errors\t" << phase_errors << '\n'
                   << "output_error\t" << (s.output_error.load(std::memory_order_relaxed) ? 1 : 0) << '\n'
                   << "invalid\t" << (trace_invalid ? 1 : 0) << '\n'
                   << "kind_0\tAllocate\nkind_1\tBind\nkind_2\tRead\nkind_3\tWrite\n"
                   << "kind_4\tFetch\nkind_5\tCleanEvict\nkind_6\tDirtyEvict\nkind_7\tFree\n";
        }
        status.flush();
        if (!status) fail();
        status.close();
        if (!status) fail();
    } catch (...) {
        fail();
    }
    if (!ok) std::cerr << "object_group_trace: failed to write trace output\n";
    return ok;
}

}  // namespace FarLib::object_group_trace
