#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace FarLib::profile::work_traffic {

// Independent of legacy profile resets. Count at the existing call sites:
// ordinary buffered WRITEs remain enqueue-accounted by design. RMW WRITEs
// already reach the ordinary WRITE hook; only RMW READ needs another hook.
struct Totals {
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    uint64_t rmw_read_bytes = 0;
};

struct alignas(64) Shard {
    std::atomic<uint64_t> read_bytes{0};
    std::atomic<uint64_t> write_bytes{0};
    std::atomic<uint64_t> rmw_read_bytes{0};
};

class Registry {
    std::mutex mutex_;
    std::vector<std::unique_ptr<Shard>> shards_;
public:
    Shard *register_thread() {
        std::lock_guard<std::mutex> lock(mutex_);
        shards_.push_back(std::make_unique<Shard>());
        return shards_.back().get();
    }

    // Rolling per-worker snapshot, not a simultaneous global transaction.
    // Retain exited workers' shards so their bytes cannot disappear.
    Totals snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        Totals result;
        for (const auto &shard : shards_) {
            result.read_bytes += shard->read_bytes.load(std::memory_order_relaxed);
            result.write_bytes += shard->write_bytes.load(std::memory_order_relaxed);
            result.rmw_read_bytes += shard->rmw_read_bytes.load(std::memory_order_relaxed);
        }
        return result;
    }
};

inline Registry &registry() {
    // Process lifetime, including late TLS destructors and worker exits.
    static Registry *value = new Registry;
    return *value;
}

inline Shard &local_shard() {
    static thread_local Shard *value = registry().register_thread();
    return *value;
}

inline void add_local(std::atomic<uint64_t> &value, uint64_t bytes) {
    // The owning OS thread is the only writer; snapshots only read. Avoid
    // a locked atomic RMW on each transfer while keeping snapshot reads safe.
    value.store(value.load(std::memory_order_relaxed) + bytes,
                std::memory_order_relaxed);
}
inline void count_read(uint64_t bytes) {
    if (bytes) add_local(local_shard().read_bytes, bytes);
}
inline void count_write(uint64_t bytes) {
    if (bytes) add_local(local_shard().write_bytes, bytes);
}
inline void count_rmw_read(uint64_t bytes) {
    if (bytes) add_local(local_shard().rmw_read_bytes, bytes);
}

inline uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Record {
    uint64_t window_id;
    uint64_t start_ns;
    uint64_t end_ns;
    Totals bytes;

    uint64_t fetch_bytes() const { return bytes.read_bytes; }
    uint64_t eviction_bytes() const { return bytes.write_bytes + bytes.rmw_read_bytes; }

    std::string line() const {
        std::ostringstream out;
        out << "runtime_traffic_work schema_version=1 scope=client_rdma phase=work"
            << " window_id=" << window_id
            << " start_monotonic_ns=" << start_ns
            << " end_monotonic_ns=" << end_ns
            << " elapsed_ns=" << end_ns - start_ns
            << " read_bytes=" << bytes.read_bytes
            << " write_bytes=" << bytes.write_bytes
            << " rmw_read_bytes=" << bytes.rmw_read_bytes
            << " fetch_bytes=" << fetch_bytes()
            << " eviction_bytes=" << eviction_bytes()
            << " snapshot=rolling_per_worker time_bounds=snapshot_bracket"
            << " accounting=existing_counter_sites";
        return out.str();
    }
};

class Window {
    std::mutex mutex_;
    bool active_ = false;
    uint64_t id_ = 0;
    uint64_t start_ns_ = 0;
    Totals before_;
public:
    void begin() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) std::abort();
        start_ns_ = monotonic_ns();
        before_ = registry().snapshot();
        ++id_;
        active_ = true;
    }

    Record end() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_) std::abort();
        const Totals after = registry().snapshot();
        // Timestamp bounds include both rolling scans. They do not claim
        // an instantaneous cut across active background producers.
        const uint64_t end_ns = monotonic_ns();
        if (after.read_bytes < before_.read_bytes ||
            after.write_bytes < before_.write_bytes ||
            after.rmw_read_bytes < before_.rmw_read_bytes) std::abort();
        Record result{id_, start_ns_, end_ns,
            {after.read_bytes - before_.read_bytes,
             after.write_bytes - before_.write_bytes,
             after.rmw_read_bytes - before_.rmw_read_bytes}};
        active_ = false;
        return result;
    }
};

inline Window &window() {
    static Window *value = new Window;
    return *value;
}
inline void begin() { window().begin(); }
inline Record end() { return window().end(); }

inline void print(const Record &record) {
    // One short write normally emits the entire row, even alongside legacy
    // buffered stdout. A leading newline also separates partial legacy rows.
    const std::string text = "\n" + record.line() + "\n";
    size_t offset = 0;
    while (offset < text.size()) {
        const ssize_t written = ::write(STDERR_FILENO, text.data() + offset,
                                        text.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) std::abort();
        offset += static_cast<size_t>(written);
    }
}
}  // namespace FarLib::profile::work_traffic
