#include <hdr/hdr_histogram.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/timer/progress_display.hpp>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "latency_mode.hpp"
#include "logging.hpp"
#include "../../runtime/common/benchmark_memory.hpp"
#include "completion_series.hpp"
#include "async/stream_runner.hpp"
#include "cache/cache.hpp"
#include "data_structure/concurrent_hashmap.hpp"
#include "test/fixed_size_string.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/perf.hpp"
#include "utils/rank_permutation.hpp"
#include "utils/stats.hpp"
#include "utils/fork_join.hpp"
#include "utils/threads.hpp"
#include "utils/uthreads.hpp"
#include "utils/zipfian.hpp"

#if __has_include("utils/request_interval_diag.hpp")
#include "utils/request_interval_diag.hpp"
#define FARLIB_KVS_HAS_REQUEST_INTERVAL_DIAG 1
#endif

using namespace FarLib;
using namespace FarLib::rdma;
using namespace std::chrono_literals;

#define ENABLE_BENCHMARK

#ifndef FARLIB_KVS_OBJECT_BYTES
#define FARLIB_KVS_OBJECT_BYTES 8192
#endif
#ifndef FARLIB_KVS_DATA_SIZE_SHIFT
#define FARLIB_KVS_DATA_SIZE_SHIFT 27
#endif

constexpr bool EnableLocalBench = false;

using str_key_t = FixedSizeString<32>;
constexpr size_t KVSObjectBytes = FARLIB_KVS_OBJECT_BYTES;
static_assert(sizeof(str_key_t) + 1 < KVSObjectBytes);
using str_value_t =
    FixedSizeString<KVSObjectBytes - sizeof(str_key_t) - 1>;
static_assert(sizeof(str_key_t) + sizeof(str_value_t) == KVSObjectBytes);
static_assert(sizeof(str_value_t) >= sizeof(uint64_t) * 2);

namespace {

constexpr uint64_t KVSDefaultSeed = 20260917ULL;
constexpr size_t KVSVersionBytes = sizeof(uint64_t) * 2;
constexpr size_t KVSValidationMaxAttempts = 64;

bool mutating_values_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("FARLIB_KVS_MUTATING_VALUES");
        return value == nullptr || std::strtoull(value, nullptr, 0) != 0;
    }();
    return enabled;
}

uint64_t mix_version(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

void stamp_version(str_value_t& value, uint64_t version) {
    ASSERT(version != 0);
    const uint64_t check = mix_version(version ^ 0xd3c15a5eedULL);
    std::memcpy(value.str.data(), &version, sizeof(version));
    std::memcpy(value.str.data() + sizeof(version), &check, sizeof(check));
}

str_value_t make_versioned_value(const str_value_t& base, uint64_t version) {
    str_value_t value = base;
    stamp_version(value, version);
    return value;
}

bool valid_versioned_value(const str_value_t& value, const str_value_t& base) {
    uint64_t version = 0;
    uint64_t check = 0;
    std::memcpy(&version, value.str.data(), sizeof(version));
    std::memcpy(&check, value.str.data() + sizeof(version), sizeof(check));
    return version != 0 && check == mix_version(version ^ 0xd3c15a5eedULL) &&
           std::memcmp(value.str.data() + KVSVersionBytes,
                       base.str.data() + KVSVersionBytes,
                       value.str.size() - KVSVersionBytes) == 0;
}

uint64_t mutation_version(size_t generator_id, uint64_t put_sequence) {
    ASSERT(generator_id < (1ULL << 15));
    ASSERT(put_sequence < (1ULL << 48) - 1);
    return ((static_cast<uint64_t>(generator_id) + 1) << 48) |
           (put_sequence + 1);
}

uint64_t steady_time_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(std::chrono::steady_clock::now()
                                      .time_since_epoch()).count());
}

struct LatencyOptions {
    uint64_t offered_load_ops;
    uint64_t warmup_ns;
    uint64_t measurement_ns;
    uint64_t drain_timeout_ns;
    uint64_t max_queue_delay_ns;
    std::filesystem::path output_dir;
};

std::optional<LatencyOptions> latency_options_from_env() {
    const char* rate = std::getenv("FARLIB_KVS_OFFERED_LOAD_OPS");
    if (rate == nullptr) return std::nullopt;
    const auto duration = [](const char* name) {
        return kv_latency::milliseconds_to_ns(
            kv_latency::parse_unsigned(std::getenv(name), name));
    };
    LatencyOptions options{
        kv_latency::parse_unsigned(rate, "FARLIB_KVS_OFFERED_LOAD_OPS"),
        duration("FARLIB_KVS_LATENCY_WARMUP_MS"),
        duration("FARLIB_KVS_LATENCY_MEASURE_MS"),
        duration("FARLIB_KVS_DRAIN_TIMEOUT_MS"),
        0,
        {}
    };
    if (const char* deadline = std::getenv("FARLIB_KVS_MAX_QUEUE_DELAY_US"))
        options.max_queue_delay_ns = kv_latency::microseconds_to_ns(
            kv_latency::parse_unsigned(deadline, "FARLIB_KVS_MAX_QUEUE_DELAY_US", true));
    const char* output = std::getenv("FARLIB_KVS_LATENCY_OUTPUT_DIR");
    if (!output || !*output)
        throw std::invalid_argument("missing FARLIB_KVS_LATENCY_OUTPUT_DIR");
    options.output_dir = output;
    const uint64_t range = kv_latency::checked_add(
        std::max(options.warmup_ns, options.measurement_ns),
        options.drain_timeout_ns);
    if (range > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("KV latency duration exceeds histogram range");
    kv_latency::checked_add(steady_time_ns(), range);
    return options;
}

// Each strict parser record is submitted as one stdio operation. Background
// runtime output must not split the fields of a measurement record.
void latency_log(const std::string& line) {
    const std::string record = "\n" + line + "\n";
    std::fputs(record.c_str(), stderr);
}

}  // namespace

using LocalHashTable = std::unordered_map<str_key_t, str_value_t>;
using RemoteHashTable = ConcurrentHashMap<str_key_t, str_value_t>;

class Server;

class Workload {
public:
    enum OpType { GET, PUT, REMOVE };

    struct Request {
        OpType op_type;
        const str_key_t& key;
        const str_value_t& value;
        uint64_t mutation_version;
        size_t queue_index;
        uint64_t request_start_ns;

        Request(OpType op_type, const str_key_t& key, const str_value_t& value,
                uint64_t mutation_version = 0)
            : op_type(op_type), key(key), value(value),
              mutation_version(mutation_version), queue_index(0) {}
    };

    struct Response {
        uint64_t in_queue_lat_ns;
        uint64_t service_latency_ns;
    };

    struct Config {
        size_t n_server_thread;
        size_t n_client_thread;
        double put_ratio;
        double remove_ratio;
        double zipfian_constant;
        std::chrono::nanoseconds max_runtime;
        uint64_t max_serve_count;
        uint64_t hotset_shift_ns;
        size_t hotset_shift_offset;
        uint64_t random_seed;
        bool deterministic_random;
        bool fixed_request_count;
        bool drain_all_requests;
        uint64_t drain_timeout_ns;
    };

    struct Receipt {
        std::array<uint64_t, 3> generated{};
        std::array<uint64_t, 3> accepted{};
        uint64_t phase_a = 0;
        uint64_t phase_b = 0;
        uint64_t fingerprint = 0;

        uint64_t total_generated() const {
            return generated[GET] + generated[PUT] + generated[REMOVE];
        }
        uint64_t total_accepted() const {
            return accepted[GET] + accepted[PUT] + accepted[REMOVE];
        }
    };

    // 10GB
    // static constexpr size_t InitialDataCount = 32 * 1024 * 1024;
    static constexpr size_t DefaultInitialDataCount = 1024 * 1024 * 2;
    static constexpr size_t DataSizeShift = FARLIB_KVS_DATA_SIZE_SHIFT;
    static constexpr size_t DataSize = 1 << DataSizeShift;
    static_assert(DefaultInitialDataCount < DataSize);

public:
    Workload()
        : rank_to_id(std::make_shared<const benchmark::RankPermutation>(
              initial_data_count(), KVSDefaultSeed)),
          remote_hash_table(DataSizeShift) {
        const auto audit = rank_to_id->audit();
        ASSERT(audit.valid() && audit.unique_count == initial_data_count());
        kvs_logging::write_record(stdout, "kvs_rank_permutation size=",
                                  rank_to_id->size(), " seed=", KVSDefaultSeed,
                                  " checksum=", audit.checksum);
        init_data();
    }

    LocalHashTable* get_local_hash_table() { return &local_hash_table; }

    RemoteHashTable* get_remote_hash_table() { return &remote_hash_table; }

    static size_t initial_data_count() {
        const char* env = std::getenv("FARLIB_KVS_INITIAL_DATA_COUNT");
        if (env == nullptr || *env == '\0') return DefaultInitialDataCount;
        size_t value = std::strtoull(env, nullptr, 10);
        ASSERT(value > 0 && value < DataSize);
        return value;
    }

    void reset_phase_counts() {
        phase_b_announced.clear(std::memory_order_relaxed);
    }

    void print_phase_counts(const char* name, const Receipt& receipt) const {
        kvs_logging::write_record(
            stdout, "kvs_hotset_phases name=",
            (name == nullptr ? "unknown" : name), " phase_a_requests=",
            receipt.phase_a, " phase_b_requests=", receipt.phase_b,
            " request_fingerprint=", receipt.fingerprint);
    }

    void verify_mutating_values(size_t requested_samples) {
        if (!mutating_values_enabled() || requested_samples == 0) return;
        const size_t samples = std::min(requested_samples, data.size());
        RootDereferenceScope scope;
        for (size_t i = 0; i < samples; ++i) {
            const size_t idx = (i * 11400714819323198485ULL) % data.size();
            str_value_t value;
            ASSERT(remote_hash_table.get(data[idx].first, &value, scope));
            ASSERT(valid_versioned_value(value, data[idx].second));
        }
        kvs_logging::write_record(stdout, "kvs_post_verify checked=", samples,
                                  " failures=0");
    }

    uint64_t gen_requests(Server& server, const Config& config, size_t qi);

    Receipt gen_requests_max_speed(Server& server, const Config& config,
                                   size_t client_index, size_t qi_start,
                                   size_t qi_end);

    void run_direct(const Config& config, size_t fibres, bool miss_yield);
    template<bool RecordCompletions>
    void run_direct_observed(const Config& config, size_t fibres, bool miss_yield);

    bool run_latency(const Config& config, size_t fibres,
                     const LatencyOptions& options);

    void run_direct(const char* name, const Config& config, size_t fibre_count,
                    bool miss_yield);

private:
    void init_data() {
        const size_t initial_count = initial_data_count();
        const bool verbose = kvs_logging::verbose_enabled();
        if (verbose)
            kvs_logging::write_record(stdout, "Initialize: loading ",
                                      initial_count, " K-V pairs");
        data.reserve(initial_count);
        local_hash_table.reserve(initial_count);
        std::optional<boost::timer::progress_display> progress;
        if (verbose) {
            kvs_logging::write_record(stdout, "Loading Data...");
            progress.emplace(initial_count);
        }
        RootDereferenceScope scope;
        for (size_t i = 0; i < initial_count; i++) {
            str_key_t key = str_key_t::random();
            str_value_t value = str_value_t::random();
            str_value_t stored_value = value;
            if (mutating_values_enabled()) {
                stamp_version(stored_value, (1ULL << 63) | (i + 1));
            }
            ASSERT(remote_hash_table.put(key, stored_value, scope));
            data.push_back({key, value});
            if constexpr (EnableLocalBench) {
                local_hash_table[key] = stored_value;
            }
            if (progress) ++*progress;
        }
    }

private:
    std::vector<std::pair<str_key_t, str_value_t>> data;
    std::shared_ptr<const benchmark::RankPermutation> rank_to_id;
    LocalHashTable local_hash_table;
    RemoteHashTable remote_hash_table;
    std::atomic_flag phase_b_announced = ATOMIC_FLAG_INIT;
};

hdr_histogram* hist = nullptr;

void init_hist() {
    int err = hdr_init(1, 1'000'000, 3, &hist);
    if (err != 0) abort();
}

void print_hist() {
    double mean = hdr_mean(hist);
    uint64_t p50 = hdr_value_at_percentile(hist, 50);
    uint64_t p90 = hdr_value_at_percentile(hist, 90);
    uint64_t p95 = hdr_value_at_percentile(hist, 95);
    uint64_t p99 = hdr_value_at_percentile(hist, 99);
    printf("%f %lu %lu %lu %lu\n", mean, p50, p90, p95, p99);
    FILE* f = fopen("kvs.hist", "w");
    hdr_percentiles_print(hist, f, 100, 1, format_type::CLASSIC);
    fclose(f);
    hdr_close(hist);
}

class Server {
public:
    static constexpr size_t QueueCap = 64;
    using RequestQueue =
        boost::lockfree::spsc_queue<Workload::Request*,
                                    boost::lockfree::capacity<QueueCap>>;

protected:
    static constexpr size_t MaxNQueue = 32;

    size_t n_queue;
    std::atomic_bool serving;
    RequestQueue request_queue[MaxNQueue];
    std::array<std::array<uint64_t, 3>, MaxNQueue> completed_by_queue{};

public:
    Server(size_t n_queue) : n_queue(n_queue) {
        ASSERT(n_queue <= MaxNQueue);
        serving = true;
    }

    ~Server() {}

    virtual void serve(size_t qi) = 0;

    bool send_request(size_t qi, Workload::Request* req) {
        req->queue_index = qi;
        req->request_start_ns = get_time_ns();
        bool sent = request_queue[qi].push(req);
        door_bell(qi);
        return sent;
    }

    virtual void door_bell(size_t qi) {}

    virtual void finish_serving() { serving.store(false); }

    bool has_pending_requests(size_t qi) {
        return !request_queue[qi].empty();
    }

    std::array<uint64_t, 3> completed_counts() const {
        std::array<uint64_t, 3> totals{};
        for (const auto& queue_counts : completed_by_queue) {
            for (size_t op = 0; op < totals.size(); ++op) {
                totals[op] += queue_counts[op];
            }
        }
        return totals;
    }

    uint64_t completed_total() const {
        const auto totals = completed_counts();
        return totals[Workload::GET] + totals[Workload::PUT] +
               totals[Workload::REMOVE];
    }

protected:
    bool get_request(size_t qi, Workload::Request*& request,
                     Workload::Response& response) {
        if (request_queue[qi].pop(request)) {
            response.in_queue_lat_ns =
                get_time_ns() - request->request_start_ns;
            return true;
        }
        return false;
    }

    void send_response(Workload::Request* request,
                       Workload::Response& response) {
        response.service_latency_ns = get_time_ns() - request->request_start_ns;
        ++completed_by_queue[request->queue_index][request->op_type];
        delete request;
    }

    bool is_serving() { return serving.load(std::memory_order::relaxed); }
};

class DryServer : public Server {
public:
    DryServer(size_t n_queue, Workload& workload) : Server(n_queue) {}

    void serve(size_t qi) override {
        while (is_serving() || has_pending_requests(qi)) {
            Workload::Request* request;
            Workload::Response response;
            if (get_request(qi, request, response)) {
                send_response(request, response);
            }
        }
    }
};

class LocalServer : public Server {
public:
    LocalServer(size_t n_queue, Workload& workload) : Server(n_queue) {
        hash_table = workload.get_local_hash_table();
    }

    void serve(size_t qi) override {
        while (is_serving() || has_pending_requests(qi)) {
            Workload::Request* request;
            Workload::Response response;
            if (get_request(qi, request, response)) {
                switch (request->op_type) {
                case Workload::GET: {
                    auto it = hash_table->find(request->key);
                    ASSERT(it == hash_table->end() ||
                           !mutating_values_enabled() ||
                           valid_versioned_value(it->second, request->value));
                    break;
                }
                case Workload::PUT: {
                    hash_table->insert_or_assign(
                        request->key, mutating_values_enabled()
                            ? make_versioned_value(request->value,
                                                   request->mutation_version)
                            : request->value);
                    break;
                }
                case Workload::REMOVE: {
                    hash_table->erase(request->key);
                    break;
                }
                }
                send_response(request, response);
            }
        }
    }

protected:
    LocalHashTable* hash_table;
};

class RemotableServer : public Server {
public:
    RemotableServer(size_t n_queue, Workload& workload) : Server(n_queue) {
        hash_table = workload.get_remote_hash_table();
    }

protected:
    RemoteHashTable* hash_table;
};

class SyncServer : public RemotableServer {
public:
    SyncServer(size_t n_queue, Workload& workload)
        : RemotableServer(n_queue, workload) {}

    void serve(size_t qi) override {
        RootDereferenceScope scope;
        while (is_serving() || has_pending_requests(qi)) {
            Workload::Request* request;
            Workload::Response response;
            if (get_request(qi, request, response)) {
                auto start = get_time_ns();
                switch (request->op_type) {
                case Workload::GET: {
                    str_value_t value;
                    for (size_t attempt = 0; attempt < KVSValidationMaxAttempts;
                         ++attempt) {
                        const bool found =
                            hash_table->get(request->key, &value, scope);
                        if (!found || !mutating_values_enabled() ||
                            valid_versioned_value(value, request->value)) {
                            break;
                        }
                        ASSERT(attempt + 1 < KVSValidationMaxAttempts);
                        uthread::yield();
                    }
                    break;
                }
                case Workload::PUT: {
                    ASSERT(!mutating_values_enabled() ||
                           request->mutation_version != 0);
                    hash_table->put(
                        request->key, mutating_values_enabled()
                            ? make_versioned_value(request->value,
                                                   request->mutation_version)
                            : request->value,
                        scope);
                    break;
                }
                case Workload::REMOVE: {
                    hash_table->remove(request->key, scope);
                }
                }
                auto end = get_time_ns();
                hdr_record_value_atomic(hist, end - start);
                send_response(request, response);
            }
        }
    }
};

class UThreadServer : public RemotableServer {
public:
    UThreadServer(size_t n_queue, Workload& workload)
        : RemotableServer(n_queue, workload), request_cond(n_queue) {}

    void door_bell(size_t qi) override {
        uthread::notify(&request_cond[qi], &request_cond_mutex);
    }

    void finish_serving() override {
        serving.store(false);
        for (auto& c : request_cond) {
            uthread::notify_all(&c, &request_cond_mutex);
        }
    }

    void serve(size_t qi) override { serve_thread(qi); }

private:
    void serve_thread(size_t qi) {
        ON_MISS_BEGIN
            uthread::yield();
        ON_MISS_END
        RootDereferenceScope scope;
        while (is_serving() || has_pending_requests(qi)) {
            Workload::Request* request;
            Workload::Response response;
            while (get_request(qi, request, response)) {
                switch (request->op_type) {
                case Workload::GET: {
                    str_value_t value;
                    bool found = hash_table->get(request->key, &value,
                                                 __on_miss__, scope);
                    ASSERT(!found || !mutating_values_enabled() ||
                           valid_versioned_value(value, request->value));
                    break;
                }
                case Workload::PUT: {
                    hash_table->put(
                        request->key, mutating_values_enabled()
                            ? make_versioned_value(request->value,
                                                   request->mutation_version)
                            : request->value,
                        __on_miss__, scope);
                    break;
                }
                case Workload::REMOVE: {
                    hash_table->remove(request->key, __on_miss__, scope);
                    break;
                }
                }
                send_response(request, response);
            }
            if (serving.load() && !has_pending_requests(qi))
                uthread::wait(&request_cond[qi], &request_cond_mutex);
        }
    }

private:
    std::vector<uthread::Condition> request_cond;
    uthread::Mutex request_cond_mutex;
};

class PararoutineServer : public RemotableServer {
    struct Context {
        PararoutineServer* server;
        Workload::OpType op_type;
        Workload::Request* request;
        Workload::Response response;

        RemoteHashTable* get_hash_map() { return server->hash_table; }
        const str_key_t* get_key() const { return &(request->key); }
        str_value_t get_value() const {
            return mutating_values_enabled()
                ? make_versioned_value(request->value,
                                       request->mutation_version)
                : request->value;
        }
        // get
        void make_result(const str_value_t* v) {
            ASSERT(v == nullptr || !mutating_values_enabled() ||
                   valid_versioned_value(*v, request->value));
            server->send_response(request, response);
        }
        // put & remove
        void make_result(bool r) { server->send_response(request, response); }
    };
    using GetFrame = RemoteHashTable::GetFrame<Context>;
    using PutFrame = RemoteHashTable::PutFrame<Context>;
    using RemoveFrame = RemoteHashTable::RemoveFrame<Context>;
    union Frame {
        Context base_frame;
        GetFrame get_frame;
        PutFrame put_frame;
        RemoveFrame remove_frame;

        void init() {
            base_frame.op_type = base_frame.request->op_type;
            switch (base_frame.op_type) {
            case Workload::GET:
                get_frame.init();
                break;
            case Workload::PUT:
                put_frame.init();
                break;
            case Workload::REMOVE:
                remove_frame.init();
                break;
            }
        }
        size_t conflict_id() const {
            switch (base_frame.op_type) {
            case Workload::GET:
                return get_frame.conflict_id();
            case Workload::PUT:
                return put_frame.conflict_id();
            case Workload::REMOVE:
                return remove_frame.conflict_id();
            }
            __builtin_unreachable();
            return 0;
        }
        ~Frame() {
            switch (base_frame.op_type) {
            case Workload::GET:
                std::destroy_at(&get_frame);
                break;
            case Workload::PUT:
                std::destroy_at(&put_frame);
                break;
            case Workload::REMOVE:
                std::destroy_at(&remove_frame);
                break;
            }
        }

        bool fetched() {
            switch (base_frame.op_type) {
            case Workload::GET:
                return get_frame.fetched();
            case Workload::PUT:
                return put_frame.fetched();
            case Workload::REMOVE:
                return remove_frame.fetched();
            }
            __builtin_unreachable();
            return true;
        }
        void pin() {
            switch (base_frame.op_type) {
            case Workload::GET:
                get_frame.pin();
                return;
            case Workload::PUT:
                put_frame.pin();
                return;
            case Workload::REMOVE:
                remove_frame.pin();
                return;
            }
            __builtin_unreachable();
        }
        void unpin() {
            switch (base_frame.op_type) {
            case Workload::GET:
                get_frame.unpin();
                return;
            case Workload::PUT:
                put_frame.unpin();
                return;
            case Workload::REMOVE:
                remove_frame.unpin();
                return;
            }
            __builtin_unreachable();
        }
        bool run(DereferenceScope& scope) {
            switch (base_frame.op_type) {
            case Workload::GET:
                return get_frame.run(scope);
            case Workload::PUT:
                return put_frame.run(scope);
            case Workload::REMOVE:
                return remove_frame.run(scope);
            }
            __builtin_unreachable();
        }
    };
    struct ReqStream {
        using Context = Frame;
        PararoutineServer* server;
        size_t qi;

        ReqStream(PararoutineServer* server, size_t qi)
            : server(server), qi(qi) {}

        async::StreamState get(Context* ctx) {
            if (!server->get_request(qi, ctx->base_frame.request,
                                     ctx->base_frame.response)) {
                return server->is_serving() ? async::StreamState::WAITING
                                            : async::StreamState::FINISHED;
            }
            ctx->base_frame.server = server;
            ctx->init();
            return async::StreamState::READY;
        }
    };

public:
    PararoutineServer(size_t n_queue, Workload& workload)
        : RemotableServer(n_queue, workload) {}

    void serve(size_t qi) override {
        process_pararoutine_stream<ReqStream, true>(ReqStream(this, qi));
    }
};

static Workload::OpType get_op_type(float rand_num,
                                    const Workload::Config& config) {
    if (rand_num < config.put_ratio) {
        return Workload::PUT;
    } else if (rand_num < config.put_ratio + config.remove_ratio) {
        return Workload::REMOVE;
    } else {
        return Workload::GET;
    }
}

static std::default_random_engine make_request_random_engine(
    const Workload::Config& config, size_t stream_id) {
    if (!config.deterministic_random) {
        return std::default_random_engine(std::random_device{}());
    }
    const uint64_t mixed =
        config.random_seed +
        0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(stream_id + 1);
    const uint64_t stream = static_cast<uint64_t>(stream_id);
    std::seed_seq sequence{
        static_cast<uint32_t>(mixed),
        static_cast<uint32_t>(mixed >> 32),
        static_cast<uint32_t>(stream),
        static_cast<uint32_t>(stream >> 32)};
    return std::default_random_engine(sequence);
}

static std::default_random_engine make_operation_random_engine(
    const Workload::Config& config, size_t stream_id) {
    if (!config.deterministic_random) {
        return std::default_random_engine(std::random_device{}());
    }
    const uint64_t mixed = (config.random_seed ^ 0xd3c15a5eed5a17ULL) +
        0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(stream_id + 1);
    std::seed_seq sequence{static_cast<uint32_t>(mixed),
                           static_cast<uint32_t>(mixed >> 32),
                           static_cast<uint32_t>(stream_id),
                           static_cast<uint32_t>(stream_id >> 32),
                           0xd3c15a5eU, 0xed5a17U};
    return std::default_random_engine(sequence);
}

// main funtion for client
uint64_t Workload::gen_requests(Server& server, const Config& config,
                                size_t qi) {
    uint64_t max_runtime_ns = config.max_runtime.count();
    uint64_t max_serve_count = config.max_serve_count;
    std::default_random_engine random_engine =
        make_request_random_engine(config, qi);
    std::default_random_engine operation_random_engine =
        make_operation_random_engine(config, qi);
    std::uniform_real_distribution<float> op_type_dist(0.0, 1.0);
    ZipfianGenerator<true> rank_generator(Workload::initial_data_count(),
                                           config.zipfian_constant);

    uint64_t period_start_time = get_time_ns();
    uint64_t final_deadline = period_start_time + max_runtime_ns;
    uint64_t op_count = 0;
    uint64_t local_fingerprint =
        1469598103934665603ULL ^ static_cast<uint64_t>(qi);

    while (true) {
        const uint64_t now = get_time_ns();
        const bool phase_b = config.hotset_shift_ns != 0 &&
            now - period_start_time >= config.hotset_shift_ns;
        int rank = rank_generator(random_engine);
        int idx = static_cast<int>((*rank_to_id)[rank]);
        if (phase_b) {
            idx = (idx + config.hotset_shift_offset) % data.size();
            if (!phase_b_announced.test_and_set(std::memory_order_relaxed)) {
                kvs_logging::write_record(stdout, "kvs_hotset_shift phase=B offset=",
                                          config.hotset_shift_offset);
            }
        }
        ASSERT(idx >= 0 && idx < data.size());
        float op_type_v = op_type_dist(operation_random_engine);
        const OpType op_type = get_op_type(op_type_v, config);
        local_fingerprint ^= static_cast<uint64_t>(idx);
        local_fingerprint *= 1099511628211ULL;
        local_fingerprint ^= static_cast<uint64_t>(op_type);
        local_fingerprint *= 1099511628211ULL;
        const uint64_t version = op_type == PUT && mutating_values_enabled()
            ? mutation_version(qi, op_count) : 0;
        Request* request = new Request(op_type, data[idx].first,
                                       data[idx].second, version);
        // send request until success
        bool sent;
        do {
            sent = server.send_request(qi, request);
        } while (!sent);
        op_count++;
        uint64_t current_time = get_time_ns();
        // break if timeout
        if (current_time >= final_deadline) break;
        if (op_count >= max_serve_count) break;
    }
    return op_count;
}

template <std::integral T>
inline T div_ceil(T a, T b) {
    return (a + b - 1) / b;
}

Workload::Receipt Workload::gen_requests_max_speed(
    Server& server, const Config& config, size_t client_index,
    size_t qi_start, size_t qi_end) {
    uint64_t max_runtime_ns = config.max_runtime.count();
    std::default_random_engine random_engine =
        make_request_random_engine(config, client_index);
    std::default_random_engine operation_random_engine =
        make_operation_random_engine(config, client_index);
    std::uniform_real_distribution<float> op_type_dist(0.0, 1.0);
    ZipfianGenerator<true> rank_generator(Workload::initial_data_count(),
                                           config.zipfian_constant);

    const uint64_t phase_start_ns = get_time_ns();
    uint64_t final_deadline = phase_start_ns + max_runtime_ns;
    Receipt receipt;
    uint64_t op_count = 0;
    uint64_t put_sequence = 0;
    const uint64_t fixed_base =
        config.max_serve_count / config.n_client_thread;
    const uint64_t fixed_remainder =
        config.max_serve_count % config.n_client_thread;
    const uint64_t fixed_quota =
        fixed_base + (client_index < fixed_remainder ? 1 : 0);
    size_t qi = qi_start;
    uint64_t local_fingerprint =
        1469598103934665603ULL ^ static_cast<uint64_t>(client_index);
    while (config.fixed_request_count ? op_count < fixed_quota
                                      : get_time_ns() < final_deadline) {
        const uint64_t now = get_time_ns();
        const bool phase_b = config.hotset_shift_ns != 0 &&
            now - phase_start_ns >= config.hotset_shift_ns;
        const int rank = rank_generator(random_engine);
        int idx = static_cast<int>((*rank_to_id)[rank]);
        if (phase_b) {
            idx = (idx + config.hotset_shift_offset) % data.size();
            ++receipt.phase_b;
            if (!phase_b_announced.test_and_set(std::memory_order_relaxed)) {
                kvs_logging::write_record(stdout, "kvs_hotset_shift phase=B offset=",
                                          config.hotset_shift_offset);
            }
        } else {
            ++receipt.phase_a;
        }
        ASSERT(idx >= 0 && idx < data.size());
        float op_type_v = op_type_dist(operation_random_engine);
        const OpType op_type = get_op_type(op_type_v, config);
        ++receipt.generated[op_type];
        local_fingerprint ^= static_cast<uint64_t>(idx);
        local_fingerprint *= 1099511628211ULL;
        local_fingerprint ^= static_cast<uint64_t>(op_type);
        local_fingerprint *= 1099511628211ULL;
        const uint64_t version = op_type == PUT && mutating_values_enabled()
            ? mutation_version(client_index, put_sequence++) : 0;
        Request* request = new Request(op_type, data[idx].first,
                                       data[idx].second, version);
        // send request until success
        bool request_sent = false;
        while (true) {
            assert(qi >= qi_start && qi < qi_end);
            bool sent = server.send_request(qi, request);
            qi = qi + 1 >= qi_end ? qi_start : qi + 1;
            if (sent) {
                request_sent = true;
                break;
            }
            if (!config.fixed_request_count &&
                get_time_ns() >= final_deadline) [[unlikely]] {
                delete request;
                break;
            }
            uthread::yield();
        }
        if (!request_sent) {
            break;
        }
        op_count++;
        ++receipt.accepted[op_type];
    }
    ASSERT(!config.fixed_request_count || op_count == fixed_quota);
    receipt.fingerprint = local_fingerprint;
    return receipt;
}

const char* const ColumnNames[] = {"name",        "op-duration", "run-time",
                                   "req-cnt",     "serv-cnt",    "instructions",
                                   "l2-miss",     "l3-miss",     "async-total",
                                   "async-sched", "gc-mark",     "gc-evict"};
constexpr size_t ColumnWidth = 16;
constexpr size_t ColumnCount = sizeof(ColumnNames) / sizeof(*ColumnNames);

void run(const char* name, Server* server, Workload* workload,
         const Workload::Config& config) {
    init_hist();
    async::StreamRunnerProfiler::reset();
    profile::reset_all();
    workload->reset_phase_counts();
    void (*serve_fn)(std::pair<Server*, size_t>*) =
        [](std::pair<Server*, size_t>* args) {
            args->first->serve(args->second);
        };
    std::vector<std::pair<Server*, size_t>> server_args(config.n_server_thread);
    std::vector<std::unique_ptr<UThread>> server_threads(
        config.n_server_thread);
    for (size_t i = 0; i < config.n_server_thread; i++) {
        server_args[i] = {server, i};
        server_threads[i] = uthread::create(serve_fn, &(server_args[i]));
    }
    std::vector<Workload::Receipt> receipts(config.n_client_thread);
    size_t n_server_per_client =
        div_ceil(config.n_server_thread, config.n_client_thread);
    std::function<void(size_t)> gen_fn = [&](size_t i) {
        size_t qi_start = n_server_per_client * i;
        size_t qi_end =
            std::min(n_server_per_client * (i + 1), config.n_server_thread);
        receipts[i] = workload->gen_requests_max_speed(
            *server, config, i, qi_start, qi_end);
    };
    const uint64_t manual_replan_ms = [] {
        const char* env = std::getenv("FARLIB_KVS_MANUAL_REPLAN_MS");
        return env == nullptr ? uint64_t{0}
                              : std::strtoull(env, nullptr, 10);
    }();
    std::atomic_bool replan_done{false};
    std::thread replan_thread;
    if (manual_replan_ms != 0) {
        replan_thread = std::thread([&] {
            uint64_t tick = 0;
            while (!replan_done.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(manual_replan_ms));
                if (replan_done.load(std::memory_order_acquire)) {
                    break;
                }
                Cache::get_default()->publish_resident_group_plan_now();
                kvs_logging::write_verbose_record(stdout, "kvs_manual_replan tick=",
                                                  ++tick);
            }
        });
    }
    const uint64_t request_start_ns = steady_time_ns();
    kvs_logging::write_record(
        stdout, "kvs_phase name=", (name == nullptr ? "unknown" : name),
        " event=request_start monotonic_ns=", request_start_ns,
        " runtime_limit_ns=", config.max_runtime.count());
    auto perf_result = perf_profile([&] {
        uthread::fork_join(config.n_client_thread, gen_fn);
    });
    Workload::Receipt total_receipt;
    for (const auto& receipt : receipts) {
        for (size_t op = 0; op < total_receipt.generated.size(); ++op) {
            total_receipt.generated[op] += receipt.generated[op];
            total_receipt.accepted[op] += receipt.accepted[op];
        }
        total_receipt.phase_a += receipt.phase_a;
        total_receipt.phase_b += receipt.phase_b;
        total_receipt.fingerprint ^= receipt.fingerprint;
    }
    const uint64_t request_end_ns = steady_time_ns();
    kvs_logging::write_record(
        stdout, "kvs_phase name=", (name == nullptr ? "unknown" : name),
        " event=request_generation_end monotonic_ns=", request_end_ns,
        " generated=", total_receipt.total_generated(), " accepted=",
        total_receipt.total_accepted());
    replan_done.store(true, std::memory_order_release);
    if (replan_thread.joinable()) {
        replan_thread.join();
    }
    server->finish_serving();
    for (auto& t : server_threads) {
        uthread::join(std::move(t));
    }
    const auto completed = server->completed_counts();
    const uint64_t completed_total = server->completed_total();
    const uint64_t drain_end_ns = steady_time_ns();
    kvs_logging::write_record(
        stdout, "kvs_phase name=", (name == nullptr ? "unknown" : name),
        " event=request_drain_end monotonic_ns=", drain_end_ns,
        " completed=", completed_total);
    kvs_logging::write_record(
        stdout, "kvs_receipt name=", (name == nullptr ? "unknown" : name),
        " generated_get=", total_receipt.generated[Workload::GET],
        " generated_put=", total_receipt.generated[Workload::PUT],
        " generated_remove=", total_receipt.generated[Workload::REMOVE],
        " accepted_get=", total_receipt.accepted[Workload::GET],
        " accepted_put=", total_receipt.accepted[Workload::PUT],
        " accepted_remove=", total_receipt.accepted[Workload::REMOVE],
        " completed_get=", completed[Workload::GET], " completed_put=",
        completed[Workload::PUT], " completed_remove=", completed[Workload::REMOVE],
        " accepted_equals_completed=",
        (total_receipt.total_accepted() == completed_total),
        " request_fingerprint=", total_receipt.fingerprint);
    ASSERT(!config.drain_all_requests ||
           total_receipt.total_accepted() == completed_total);
    if (mutating_values_enabled() && config.remove_ratio == 0.0) {
        const char* env = std::getenv("FARLIB_KVS_POST_VERIFY_SAMPLES");
        const size_t samples = env == nullptr ? size_t{4096}
            : static_cast<size_t>(std::strtoull(env, nullptr, 0));
        workload->verify_mutating_values(samples);
    }
    workload->print_phase_counts(name, total_receipt);
    print_hist();
    if (name) {
        std::ostringstream line;
        // PerfResult::print() intentionally leaves cout's fixed formatting in
        // place; preserve that state when moving this row to one fwrite.
        line.copyfmt(std::cout);
        line << std::setw(ColumnWidth) << name;
        line << std::setw(ColumnWidth) << "max";
        line << std::setw(ColumnWidth) << perf_result.runtime_ms / 1e3;
        line << std::setw(ColumnWidth) << total_receipt.total_accepted();
        line << std::setw(ColumnWidth) << completed_total;
        line << std::setw(ColumnWidth) << perf_result.instructions;
        line << std::setw(ColumnWidth) << perf_result.l2_cache_miss;
        line << std::setw(ColumnWidth) << perf_result.l3_cache_miss;
        line << std::setw(ColumnWidth)
             << async::StreamRunnerProfiler::get_total_cycles();
        line << std::setw(ColumnWidth)
             << async::StreamRunnerProfiler::get_sched_cycles();
        line << std::setw(ColumnWidth) << profile::collect_mark_cycles();
        line << std::setw(ColumnWidth) << profile::collect_evict_cycles();
        kvs_logging::write_record(stdout, line.str());
    }
}

void Workload::run_direct(const Config& config, size_t fibres, bool miss_yield) {
    const bool record = kvs_observation::completion_series_enabled(
        std::getenv("FARLIB_KVS_COMPLETION_SERIES"));
    if (record) return run_direct_observed<true>(config, fibres, miss_yield);
    return run_direct_observed<false>(config, fibres, miss_yield);
}

template<bool RecordCompletions>
void Workload::run_direct_observed(const Config& config, size_t fibres, bool miss_yield) {
    ASSERT(fibres > 0 && fibres < (1ULL << 15));
    const char* lock_env = std::getenv("FARLIB_KVS_DEBUG_LOCKED_GET");
    const bool shared_get_lock = lock_env != nullptr && std::strtoull(lock_env, nullptr, 10) != 0;
    kvs_logging::write_record(stdout, "kvs_get_lock enabled=", shared_get_lock);
    kvs_logging::write_record(stdout, "kvs_deref_stats enabled=",
                              RemoteHashTable::deref_stats_enabled());
    RemoteHashTable::reset_deref_counts();
    const char* validate_env = std::getenv("FARLIB_KVS_VALIDATE_GET");
    const bool validate_get = validate_env == nullptr ||
        std::strtoull(validate_env, nullptr, 10) != 0;
    kvs_logging::write_record(stdout, "kvs_validation_config get_content=",
                              validate_get, " real_put=", mutating_values_enabled());
    const char* hist_env = std::getenv("FARLIB_KVS_HIST_SAMPLE_PERIOD");
    const uint64_t hist_period = hist_env == nullptr ? 1024 : std::strtoull(hist_env, nullptr, 10);
    ASSERT(hist_period == 1 || hist_period == 1024);
    kvs_logging::write_record(
        stdout, "kvs_hist_sampling period=", hist_period,
        " selector=one_per_block_rotating_offset timestamps_only_when_selected=1");
    init_hist();
    reset_phase_counts();
    std::vector<Receipt> receipts(fibres);
    std::vector<std::array<uint64_t, 3>> completed(fibres);
    std::vector<std::array<uint64_t, 2>> hist_counts(fibres);
    std::atomic_size_t ready{0};
    std::atomic_bool go{false};
    std::atomic<uint64_t> shared_start{0};
    std::atomic<uint64_t> shared_deadline{0};
    std::unique_ptr<kvs_observation::CompletionSeries> completion_series;
    if constexpr (RecordCompletions) {
        const char *cpu = std::getenv("FARLIB_KVS_COMPLETION_SERIES_CPU");
        if (!cpu || !*cpu) ERROR("completion series requires an explicit observer CPU");
        char *end = nullptr;
        long number = std::strtol(cpu, &end, 10);
        if (*end || number < 0 || number >= CPU_SETSIZE)
            ERROR("invalid completion series CPU");
        completion_series = std::make_unique<kvs_observation::CompletionSeries>(
            fibres, static_cast<int>(number));
        kvs_logging::write_record(
            stdout, "kvs_completion_series enabled=1 interval_ms=100 collector_cpu=",
            number, " file=kvs.completed-100ms.csv");
    }
    kvs_logging::write_record(
        stdout, "kvs_direct_config fibres=", fibres, " miss_yield=", miss_yield,
        " fairness_yield_every=64 shared_deadline=1 object_bytes=", KVSObjectBytes,
        " initial_count=", data.size(), " put_ratio=", config.put_ratio);
    auto worker = [&](size_t tid) {
        auto random_engine = make_request_random_engine(config, tid);
        auto operation_engine = make_operation_random_engine(config, tid);
        std::uniform_real_distribution<float> op_type_dist(0.0, 1.0);
        ZipfianGenerator<true> rank_generator(initial_data_count(),
                                             config.zipfian_constant);
        Receipt& receipt = receipts[tid];
        auto& done = completed[tid];
        uint64_t sequence = 0;
        uint64_t operations = 0;
        uint64_t hist_selected = 0, hist_retained = 0;
        uint64_t fingerprint = 1469598103934665603ULL ^ tid;
        const uint64_t quota = config.max_serve_count / fibres +
                              (tid < config.max_serve_count % fibres ? 1 : 0);
        // All operation fibres arrive cooperatively before one common clock
        // starts. Never hold a far-memory scope while waiting at this gate.
        if (ready.fetch_add(1, std::memory_order_acq_rel) + 1 == fibres) {
            const uint64_t start = steady_time_ns();
            shared_start.store(start, std::memory_order_relaxed);
            shared_deadline.store(start + config.max_runtime.count(),
                                  std::memory_order_relaxed);
            benchmark_memory::begin_work_at(start, "kvs_request_start");
            if constexpr (RecordCompletions) completion_series->start(start);
            kvs_logging::write_record(
                stdout, "kvs_phase name=direct event=request_start monotonic_ns=",
                start, " runtime_limit_ns=", config.max_runtime.count());
            go.store(true, std::memory_order_release);
        }
        while (!go.load(std::memory_order_acquire)) uthread::yield();
        const uint64_t start = shared_start.load(std::memory_order_relaxed);
        const uint64_t deadline = shared_deadline.load(std::memory_order_relaxed);
        ON_MISS_BEGIN
            if (miss_yield) uthread::yield();
        ON_MISS_END
        while (config.fixed_request_count ? operations < quota
                                          : steady_time_ns() < deadline) {
            const uint64_t now = steady_time_ns();
            const bool phase_b = config.hotset_shift_ns != 0 &&
                                 now - start >= config.hotset_shift_ns;
            const int rank = rank_generator(random_engine);
            size_t idx = (*rank_to_id)[rank];
            if (phase_b) {
                idx = (idx + config.hotset_shift_offset) % data.size();
                ++receipt.phase_b;
            } else {
                ++receipt.phase_a;
            }
            ASSERT(idx < data.size());
            const OpType op = get_op_type(op_type_dist(operation_engine), config);
            ++receipt.generated[op];
            ++receipt.accepted[op];
            fingerprint ^= idx; fingerprint *= 1099511628211ULL;
            fingerprint ^= static_cast<uint64_t>(op);
            fingerprint *= 1099511628211ULL;
            const uint64_t version = op == PUT && mutating_values_enabled()
                ? mutation_version(tid, sequence++) : 0;
            // One sample per 1024-request block per fibre, with a rotating
            // offset so sampling does not stay aligned with the 64-op yield.
            // No shared sampling counter or request-path RNG is introduced.
            const bool sample_hist = hist_period == 1 ||
                (operations & 1023ULL) == (((operations >> 10) * 17 + tid * 37) & 1023ULL);
            const uint64_t operation_start = sample_hist ? get_time_ns() : 0;
            {
                RootDereferenceScope scope;
                const auto& [key, base] = data[idx];
                switch (op) {
                case GET: {
                    str_value_t value;
                    for (size_t attempt = 0; attempt < KVSValidationMaxAttempts;
                         ++attempt) {
                        const bool found = remote_hash_table.get(
                            key, &value, __on_miss__, scope, shared_get_lock);
                        ASSERT(found);
                        if (!found || !validate_get || !mutating_values_enabled() ||
                            valid_versioned_value(value, base)) break;
                        ASSERT(attempt + 1 < KVSValidationMaxAttempts);
                        uthread::yield();
                    }
                    break;
                }
                case PUT:
                    ASSERT(!mutating_values_enabled() || version != 0);
                    remote_hash_table.put(key,
                        mutating_values_enabled() ? make_versioned_value(base, version)
                                                  : base,
                        __on_miss__, scope);
                    break;
                case REMOVE:
                    remote_hash_table.remove(key, __on_miss__, scope);
                    break;
                }
            }
            if (sample_hist) {
                ++hist_selected;
                hist_retained += hdr_record_value_atomic(hist, get_time_ns() - operation_start) ? 1 : 0;
            }
            ++done[op];
            ++operations;
            if constexpr (RecordCompletions)
                completion_series->publish(tid, operations);
            // Bound non-yielding local-hit streaks without retaining a scope.
            if ((operations & 63) == 0) uthread::yield();
        }
        receipt.fingerprint = fingerprint;
        hist_counts[tid] = {hist_selected, hist_retained};
    };
    uthread::fork_join(fibres, worker, "kvs_direct");
    const uint64_t end = steady_time_ns();
    benchmark_memory::end_work_at(end, "kvs_request_start");
    Receipt total;
    std::array<uint64_t, 3> done{};
    uint64_t hist_selected_total = 0, hist_retained_total = 0;
    for (size_t tid = 0; tid < fibres; ++tid) {
        for (size_t op = 0; op < 3; ++op) {
            total.generated[op] += receipts[tid].generated[op];
            total.accepted[op] += receipts[tid].accepted[op];
            done[op] += completed[tid][op];
            ASSERT(receipts[tid].accepted[op] == completed[tid][op]);
        }
        total.phase_a += receipts[tid].phase_a;
        total.phase_b += receipts[tid].phase_b;
        total.fingerprint ^= receipts[tid].fingerprint;
        hist_selected_total += hist_counts[tid][0];
        hist_retained_total += hist_counts[tid][1];
        kvs_logging::write_verbose_record(
            stdout, "kvs_hist_fibre id=", tid, " operations=",
            receipts[tid].total_accepted(), " selected=", hist_counts[tid][0],
            " retained=", hist_counts[tid][1], " dropped=",
            hist_counts[tid][0] - hist_counts[tid][1]);
        kvs_logging::write_verbose_record(
            stdout, "kvs_direct_fibre id=", tid, " completed=",
            receipts[tid].total_accepted(), " completed_get=", completed[tid][GET],
            " completed_put=", completed[tid][PUT]);
    }
    const uint64_t completed_total = done[GET] + done[PUT] + done[REMOVE];
    if constexpr (RecordCompletions) {
        completion_series->finish(end, completed_total);
        completion_series->write("kvs.completed-100ms");
    }
    kvs_logging::write_record(
        stdout, "kvs_phase name=direct event=request_generation_end monotonic_ns=",
        end, " generated=", total.total_generated(), " accepted=",
        total.total_accepted());
    kvs_logging::write_record(
        stdout, "kvs_phase name=direct event=request_drain_end monotonic_ns=", end,
        " completed=", completed_total);
    kvs_logging::write_record(
        stdout, "kvs_receipt name=direct generated_get=", total.generated[GET],
        " generated_put=", total.generated[PUT], " generated_remove=",
        total.generated[REMOVE], " accepted_get=", total.accepted[GET],
        " accepted_put=", total.accepted[PUT], " accepted_remove=",
        total.accepted[REMOVE], " completed_get=", done[GET], " completed_put=",
        done[PUT], " completed_remove=", done[REMOVE],
        " accepted_equals_completed=", (total.total_accepted() == completed_total),
        " request_fingerprint=", total.fingerprint);
    ASSERT(total.total_generated() == completed_total);
    ASSERT(!config.fixed_request_count || completed_total == config.max_serve_count);
    ASSERT(static_cast<uint64_t>(hist->total_count) == hist_retained_total);
    kvs_logging::write_record(
        stdout, "kvs_hist_receipt period=", hist_period, " operations=",
        completed_total, " selected=", hist_selected_total, " retained=",
        hist_retained_total, " dropped=", hist_selected_total - hist_retained_total,
        " histogram_total=", hist->total_count);
    kvs_logging::write_record(stdout, "kvs_deref_receipt local=",
                              RemoteHashTable::local_deref_count(), " remote=",
                              RemoteHashTable::remote_deref_count());
    if (mutating_values_enabled() && config.remove_ratio == 0.0) {
        const char* env = std::getenv("FARLIB_KVS_POST_VERIFY_SAMPLES");
        verify_mutating_values(env == nullptr ? 4096 : std::strtoull(env, nullptr, 0));
    }
    print_phase_counts("direct", total);
    print_hist();
}

bool Workload::run_latency(const Config& config, size_t fibres,
                           const LatencyOptions& options) {
    ASSERT(fibres == 48 && uthread::get_worker_count() == 24);
    ASSERT(KVSObjectBytes == 512 && data.size() == 33554432);
    ASSERT(config.put_ratio == 0.05 && config.remove_ratio == 0.0);
    ASSERT(config.zipfian_constant == 0.99 && config.hotset_shift_ns == 0);
    ASSERT(config.deterministic_random && config.random_seed == KVSDefaultSeed);
    ASSERT(mutating_values_enabled());
    const char* verify = std::getenv("FARLIB_KVS_POST_VERIFY_SAMPLES");
    ASSERT(verify == nullptr ||
           kv_latency::parse_unsigned(verify, "FARLIB_KVS_POST_VERIFY_SAMPLES") == 4096);
    std::error_code directory_error;
    std::filesystem::create_directories(options.output_dir, directory_error);
    if (directory_error) {
        latency_log("kvs_latency_error reason=histogram_directory");
        return false;
    }
    const uint64_t arrival_seed = config.random_seed;
    {
        std::ostringstream line;
        line << "kvs_latency_config offered_load_ops=" << options.offered_load_ops
             << " fibres=" << fibres << " os_workers=" << uthread::get_worker_count()
             << " warmup_ns=" << options.warmup_ns
             << " measurement_ns=" << options.measurement_ns
             << " drain_timeout_ns=" << options.drain_timeout_ns
             << " max_queue_delay_ns=" << options.max_queue_delay_ns
             << " deadline_action=" << (options.max_queue_delay_ns
                 ? "drop_before_execution" : "disabled")
             << " arrival=poisson_per_fibre queue_model=fifo_independent_lanes"
             << " include_queue_wait=1 hist_sample_period=1"
             << " object_bytes=" << KVSObjectBytes << " initial_count=" << data.size()
             << " put_ratio=" << config.put_ratio << " zipf=" << config.zipfian_constant
             << " random_seed=" << config.random_seed << " arrival_seed=" << arrival_seed
             << " locked_get=1 real_put=1 get_content_validation=0";
        latency_log(line.str());
    }
    latency_log("kvs_get_lock enabled=1 mode=open_loop");
    latency_log("kvs_validation_config get_content=0 real_put=1 mode=open_loop");
    {
        std::ostringstream line;
        line << "kvs_direct_config fibres=" << fibres
             << " miss_yield=1 fairness_yield_every=64 shared_deadline=1"
             << " object_bytes=" << KVSObjectBytes << " initial_count=" << data.size()
             << " put_ratio=" << config.put_ratio << " mode=open_loop";
        latency_log(line.str());
    }
    RemoteHashTable::reset_deref_counts();
    struct RandomState {
        std::default_random_engine rank;
        std::default_random_engine operation;
        ZipfianGenerator<true> rank_generator;
        uint64_t put_sequence = 0;
    };
    // zeta initialization scans the full key domain. Do it once before warmup,
    // then copy its parameters into private distributions; no expensive setup
    // or shared mutable distribution is introduced at the phase boundary.
    const ZipfianGenerator<true> rank_template(initial_data_count(),
                                               config.zipfian_constant);
    std::vector<RandomState> random;
    random.reserve(fibres);
    for (size_t lane = 0; lane < fibres; ++lane)
        random.push_back({make_request_random_engine(config, lane),
                          make_operation_random_engine(config, lane),
                          rank_template, 0});

    // Arrival streams are phase-local; operation RNGs and mutation versions
    // continue across the fully drained warmup/measurement boundary.
    const auto phase = [&](const char* name, uint64_t phase_id, uint64_t window_ns) {
        struct Lane {
            kv_latency::LaneCounts counts;
            std::array<uint64_t, 3> generated{};
            std::array<uint64_t, 3> completed{};
            std::array<uint64_t, 3> dropped{};
            std::array<uint64_t, 3> skipped{};
            uint64_t fingerprint = 0;
            std::unique_ptr<kv_latency::HistogramSet> histogram;
        };
        const uint64_t histogram_range =
            kv_latency::checked_add(window_ns, options.drain_timeout_ns);
        std::vector<Lane> lanes(fibres);
        for (auto& lane : lanes)
            lane.histogram = std::make_unique<kv_latency::HistogramSet>(histogram_range);
        std::atomic_size_t ready{0};
        std::atomic_bool go{false};
        std::atomic<uint64_t> epoch{0};
        auto worker = [&](size_t tid) {
            auto& lane = lanes[tid];
            auto& rng = random[tid];
            std::uniform_real_distribution<float> op_dist(0.0, 1.0);
            lane.fingerprint = 1469598103934665603ULL ^ tid;
            if (ready.fetch_add(1, std::memory_order_acq_rel) + 1 == fibres) {
                const uint64_t start = steady_time_ns();
                kv_latency::checked_add(start, histogram_range);
                epoch.store(start, std::memory_order_relaxed);
                std::ostringstream line;
                line << "kvs_latency_phase name=" << name
                     << " event=start monotonic_ns=" << start;
                latency_log(line.str());
                go.store(true, std::memory_order_release);
            }
            while (!go.load(std::memory_order_acquire)) uthread::yield();
            kv_latency::LaneSchedule schedule(
                epoch.load(std::memory_order_relaxed), window_ns,
                options.drain_timeout_ns, options.offered_load_ops, fibres,
                kv_latency::lane_seed(arrival_seed, phase_id, tid),
                options.max_queue_delay_ns);
            ON_MISS_BEGIN
                uthread::yield();
            ON_MISS_END
            uint64_t due;
            while (schedule.next(due)) {
                // Materialize every scheduled request, including dropped/skipped ones,
                // so its operation counters/fingerprint remain replayable.
                const int rank = rng.rank_generator(rng.rank);
                const size_t idx = (*rank_to_id)[rank];
                ASSERT(idx < data.size());
                const OpType op = get_op_type(op_dist(rng.operation), config);
                ++lane.generated[op];
                lane.fingerprint ^= idx;
                lane.fingerprint *= 1099511628211ULL;
                lane.fingerprint ^= static_cast<uint64_t>(op);
                lane.fingerprint *= 1099511628211ULL;
                const uint64_t version =
                    op == PUT ? mutation_version(tid, rng.put_sequence++) : 0;
                // This wait has no far-memory dereference scope. A delayed
                // request retains its original due time: no coordinated omission.
                uint64_t begin = steady_time_ns();
                while (begin < due) {
                    uthread::yield();
                    begin = steady_time_ns();
                }
                const auto admission = schedule.start_decision(begin);
                if (admission != kv_latency::LaneSchedule::StartDecision::Started) {
                    if (admission == kv_latency::LaneSchedule::StartDecision::DeadlineDropped)
                        ++lane.dropped[op];
                    else
                        ++lane.skipped[op];
                    if ((schedule.counts.scheduled & 63) == 0) uthread::yield();
                    continue;
                }
                {
                    RootDereferenceScope scope;
                    const auto& [key, base] = data[idx];
                    if (op == GET) {
                        str_value_t value;
                        const bool found = remote_hash_table.get(
                            key, &value, __on_miss__, scope, true);
                        ASSERT(found);
                        // The benchmark performs the full 479-byte value copy,
                        // even though content verification is outside timing.
                        asm volatile("" : : "m"(value) : "memory");
                    } else {
                        ASSERT(op == PUT && version != 0);
                        remote_hash_table.put(
                            key, make_versioned_value(base, version),
                            __on_miss__, scope);
                    }
                }
                const uint64_t complete = steady_time_ns();
                schedule.complete(complete);
                ++lane.completed[op];
                lane.histogram->record(due, begin, complete);
                if ((schedule.counts.scheduled & 63) == 0) uthread::yield();
            }
            // The arrival window is fixed even when this lane's final request
            // completes early. All lanes drain before the next phase starts.
            while (steady_time_ns() < schedule.window_end()) uthread::yield();
            lane.counts = schedule.counts;
        };
        uthread::fork_join(fibres, worker, "kvs_open_loop");
        const uint64_t end = steady_time_ns();
        const uint64_t start = epoch.load(std::memory_order_relaxed);
        {
            std::ostringstream line;
            line << "kvs_latency_phase name=" << name
                 << " event=end monotonic_ns=" << end;
            latency_log(line.str());
        }
        kv_latency::LaneCounts total;
        std::array<uint64_t, 3> generated{}, completed{}, dropped{}, skipped{};
        uint64_t fingerprint = 0, merge_dropped = 0;
        kv_latency::HistogramSet histogram(histogram_range);
        for (const auto& lane : lanes) {
            total.scheduled = kv_latency::checked_add(total.scheduled, lane.counts.scheduled);
            total.started = kv_latency::checked_add(total.started, lane.counts.started);
            total.completed = kv_latency::checked_add(total.completed, lane.counts.completed);
            total.completed_in_window = kv_latency::checked_add(
                total.completed_in_window, lane.counts.completed_in_window);
            total.deadline_dropped = kv_latency::checked_add(
                total.deadline_dropped, lane.counts.deadline_dropped);
            total.completed_after_deadline = kv_latency::checked_add(
                total.completed_after_deadline, lane.counts.completed_after_deadline);
            total.skipped = kv_latency::checked_add(total.skipped, lane.counts.skipped);
            total.drain_timeout |= lane.counts.drain_timeout;
            for (size_t op = 0; op < 3; ++op) {
                generated[op] = kv_latency::checked_add(generated[op], lane.generated[op]);
                completed[op] = kv_latency::checked_add(completed[op], lane.completed[op]);
                dropped[op] = kv_latency::checked_add(dropped[op], lane.dropped[op]);
                skipped[op] = kv_latency::checked_add(skipped[op], lane.skipped[op]);
                ASSERT(lane.generated[op] ==
                       lane.completed[op] + lane.dropped[op] + lane.skipped[op]);
            }
            fingerprint ^= lane.fingerprint;
            merge_dropped = kv_latency::checked_add(
                merge_dropped, histogram.merge(*lane.histogram));
        }
        ASSERT(total.scheduled == generated[GET] + generated[PUT] + generated[REMOVE]);
        ASSERT(total.completed == completed[GET] + completed[PUT] + completed[REMOVE]);
        ASSERT(total.started == total.completed);
        ASSERT(total.scheduled == total.started + total.deadline_dropped + total.skipped);
        ASSERT(total.deadline_dropped == dropped[GET] + dropped[PUT] + dropped[REMOVE]);
        ASSERT(total.skipped == skipped[GET] + skipped[PUT] + skipped[REMOVE]);
        ASSERT(total.completed_after_deadline <= total.completed);
        for (size_t op = 0; op < 3; ++op)
            ASSERT(generated[op] == completed[op] + dropped[op] + skipped[op]);
        {
            std::ostringstream line;
            line << "kvs_latency_receipt name=" << name
                 << " scheduled=" << total.scheduled << " started=" << total.started
                 << " completed=" << total.completed
                 << " completed_in_window=" << total.completed_in_window
                 << " deadline_dropped=" << total.deadline_dropped
                 << " completed_after_deadline=" << total.completed_after_deadline
                 << " skipped=" << total.skipped
                 << " generated_get=" << generated[GET] << " generated_put=" << generated[PUT]
                 << " generated_remove=" << generated[REMOVE]
                 << " completed_get=" << completed[GET] << " completed_put=" << completed[PUT]
                 << " completed_remove=" << completed[REMOVE]
                 << " dropped_get=" << dropped[GET] << " dropped_put=" << dropped[PUT]
                 << " dropped_remove=" << dropped[REMOVE]
                 << " skipped_get=" << skipped[GET] << " skipped_put=" << skipped[PUT]
                 << " skipped_remove=" << skipped[REMOVE]
                 << " request_fingerprint=" << fingerprint;
            latency_log(line.str());
        }
        uint64_t histogram_write_errors = 0;
        const char* labels[] = {"total", "service", "dispatch"};
        for (size_t i = 0; i < 3; ++i) {
            const auto path = options.output_dir /
                (std::string(name) + "." + labels[i] + ".hgrm");
            FILE* file = std::fopen(path.c_str(), "wx");
            if (!file) {
                ++histogram_write_errors;
                continue;
            }
            if (hdr_percentiles_print(histogram.get(
                    static_cast<kv_latency::HistogramSet::Kind>(i)),
                    file, 100, 1, format_type::CLASSIC) != 0)
                ++histogram_write_errors;
            if (std::fclose(file) != 0) ++histogram_write_errors;
        }
        const bool passed = histogram_write_errors == 0 &&
            kv_latency::complete_measurement(total, histogram, merge_dropped);
        {
            std::ostringstream line;
            line << "kvs_latency_result name=" << name
                 << " status=" << (passed ? "passed" : "invalid")
                 << " arrival_window_ns=" << window_ns
                 << " drain_elapsed_ns=" << end - kv_latency::checked_add(start, window_ns)
                 << " drain_timeout=" << total.drain_timeout
                 << " hist_count=" << histogram.count(kv_latency::HistogramSet::Total)
                 << " hist_service_count=" << histogram.count(kv_latency::HistogramSet::Service)
                 << " hist_dispatch_count=" << histogram.count(kv_latency::HistogramSet::Dispatch)
                 << " hist_dropped=" << histogram.dropped()
                 << " merge_dropped=" << merge_dropped
                 << " hist_write_errors=" << histogram_write_errors;
            // Incomplete or truncated histograms are retained for diagnosis,
            // but never exposed as an apparently valid P99 measurement.
            if (passed) {
                line << " p99_ns=" << histogram.p99(kv_latency::HistogramSet::Total)
                     << " p99_service_ns=" << histogram.p99(kv_latency::HistogramSet::Service)
                     << " p99_dispatch_ns=" << histogram.p99(kv_latency::HistogramSet::Dispatch);
            }
            latency_log(line.str());
        }
        return passed;
    };
    const bool warmup_ok = phase("warmup", 0, options.warmup_ns);
    const bool measurement_ok =
        warmup_ok && phase("measurement", 1, options.measurement_ns);
    verify_mutating_values(4096);
    return warmup_ok && measurement_ok;
}

bool run(size_t n_server_core, const std::optional<LatencyOptions>& latency_options) {
    constexpr size_t NEvalRepeat = 1;
    Workload workload;
    Workload::Config config{
        .n_server_thread = n_server_core,
        .n_client_thread = n_server_core / 2,
        .put_ratio = 0.5,
        .remove_ratio = 0.0,
        .zipfian_constant = 0.99,
        .max_runtime = 60s,
        .max_serve_count = 50'000'000,
        .hotset_shift_ns = 0,
        .hotset_shift_offset = 0,
        .random_seed = KVSDefaultSeed,
        .deterministic_random = true,
        .fixed_request_count = false,
        .drain_all_requests = true,
        .drain_timeout_ns = 30'000'000'000ULL,
    };
    if (const char* env = std::getenv("FARLIB_KVS_MAX_RUNTIME_MS")) {
        uint64_t value = std::strtoull(env, nullptr, 10);
        ASSERT(value > 0);
        config.max_runtime = std::chrono::milliseconds(value);
    }
    if (const char* env = std::getenv("FARLIB_KVS_MAX_SERVE_COUNT")) {
        uint64_t value = std::strtoull(env, nullptr, 10);
        ASSERT(value > 0);
        config.max_serve_count = value;
    }
    if (const char* env = std::getenv("FARLIB_KVS_PUT_RATIO")) {
        config.put_ratio = std::atof(env);
        ASSERT(config.put_ratio >= 0.0 && config.put_ratio <= 1.0);
    }
    if (const char* env = std::getenv("FARLIB_KVS_REMOVE_RATIO")) {
        config.remove_ratio = std::atof(env);
        ASSERT(config.remove_ratio >= 0.0 && config.remove_ratio <= 1.0);
    }
    ASSERT(config.put_ratio + config.remove_ratio <= 1.0);
    if (const char* env = std::getenv("FARLIB_KVS_ZIPFIAN_CONSTANT")) {
        config.zipfian_constant = std::atof(env);
        ASSERT(config.zipfian_constant >= 0.0);
    }
    if (const char* env = std::getenv("FARLIB_KVS_HOTSET_SHIFT_MS")) {
        config.hotset_shift_ns =
            std::strtoull(env, nullptr, 10) * 1'000'000ULL;
    }
    if (const char* env = std::getenv("FARLIB_KVS_RANDOM_SEED")) {
        config.random_seed = std::strtoull(env, nullptr, 0);
        config.deterministic_random = true;
    }
    if (const char* env = std::getenv("FARLIB_KVS_FIXED_REQUEST_COUNT")) {
        config.fixed_request_count = std::strtoull(env, nullptr, 0) != 0;
    }
    if (const char* env = std::getenv("FARLIB_KVS_DRAIN_ALL_REQUESTS")) {
        config.drain_all_requests = std::strtoull(env, nullptr, 0) != 0;
    }
    if (const char* env = std::getenv("FARLIB_KVS_DRAIN_TIMEOUT_MS")) {
        config.drain_timeout_ns =
            std::strtoull(env, nullptr, 0) * 1'000'000ULL;
    }
    ASSERT(!config.fixed_request_count || config.deterministic_random);
    config.hotset_shift_offset = Workload::initial_data_count() / 2;
    bool run_async = false;
    if (const char* env = std::getenv("FARLIB_KVS_RUN_ASYNC")) {
        run_async = std::strtoull(env, nullptr, 0) != 0;
    }
    ASSERT(!run_async);

    const char* execution_mode = std::getenv("FARLIB_KVS_EXECUTION_MODE");
    ASSERT(execution_mode == nullptr || std::strcmp(execution_mode, "queue") == 0 ||
           std::strcmp(execution_mode, "direct") == 0);
    const bool direct = latency_options.has_value() ||
                        (execution_mode != nullptr &&
                         std::strcmp(execution_mode, "direct") == 0);

    if (!direct) {
    kvs_logging::write_record(stdout, "setup: ", config.n_server_thread,
                              " server cores; ", config.n_client_thread,
                              " clients");
    } else if (latency_options) {
        latency_log("setup: open-loop KV; 48 independent FIFO logical lanes; "
                    "implicit backlog; no cross-lane stealing");
    } else {
        kvs_logging::write_record(
            stdout, "setup: self-generated direct KV operations; no request queues");
    }
    kvs_logging::write_record(stdout, "kvs.object_bytes: ", KVSObjectBytes);
    kvs_logging::write_record(stdout, "kvs.put_ratio: ", config.put_ratio);
    kvs_logging::write_record(stdout, "kvs.remove_ratio: ", config.remove_ratio);
    kvs_logging::write_record(stdout, "kvs.zipfian: ", config.zipfian_constant);
    kvs_logging::write_record(stdout, "kvs.hotset_shift_ns: ", config.hotset_shift_ns,
                              " offset=", config.hotset_shift_offset);
    kvs_logging::write_record(
        stdout, "kvs.random_seed: ",
        (config.deterministic_random ? std::to_string(config.random_seed)
                                     : std::string("random_device")));
    kvs_logging::write_record(stdout, "kvs.fixed_request_count: ",
                              config.fixed_request_count);
    kvs_logging::write_record(stdout, "kvs.drain_all_requests: ",
                              config.drain_all_requests,
                              " timeout_ns=", config.drain_timeout_ns);
    kvs_logging::write_record(stdout, "kvs.mutating_values: ",
                              mutating_values_enabled(), " post_verify_default=4096");
    if (!direct) {
    kvs_logging::write_record(
        stdout, "kvs.workers: total=",
        config.n_server_thread + config.n_client_thread,
        " generators=", config.n_client_thread, " sync_service_fibres=",
        config.n_server_thread);
    } else {
        kvs_logging::write_record(
            stdout, "kvs.workers: scheduler=", uthread::get_worker_count(),
            " generators=0 sync_service_fibres=0"
            " operation_fibres_reported_in=kvs_direct_config");
    }

    std::ostringstream column_header;
    column_header.copyfmt(std::cout);
    for (auto name : ColumnNames) {
        column_header << std::setw(ColumnWidth) << name;
    }
    kvs_logging::write_record(stdout, column_header.str());

    // warm up

    bool successful = true;
    #ifdef ENABLE_BENCHMARK
    if (latency_options) {
        const char* fibre_env = std::getenv("FARLIB_KVS_DIRECT_FIBRES");
        const size_t fibres = fibre_env == nullptr ? 48 :
            kv_latency::parse_unsigned(fibre_env, "FARLIB_KVS_DIRECT_FIBRES");
        profile::reset_all();
        perf_profile([&] {
            profile::start_work();
            profile::thread_start_work();
            successful = workload.run_latency(config, fibres, *latency_options);
            profile::thread_end_work();
            profile::end_work();
        }).print();
        profile::print_profile_data();
        allocator::remote::remote_global_heap.print_used_memory();
    } else if (direct) {
        const char* fibre_env = std::getenv("FARLIB_KVS_DIRECT_FIBRES");
        const size_t fibres = fibre_env == nullptr ? n_server_core
            : std::strtoull(fibre_env, nullptr, 10);
        ASSERT(fibres > 0 && fibres < (1ULL << 15));
        const char* yield_env = std::getenv("FARLIB_KVS_DIRECT_MISS_YIELD");
        const bool miss_yield = yield_env == nullptr || std::strtoull(yield_env, nullptr, 10) != 0;
#ifdef FARLIB_KVS_HAS_REQUEST_INTERVAL_DIAG
        request_interval_diag::begin_stage(fibres);
#endif
        profile::reset_all();
        perf_profile([&] {
            profile::start_work();
            profile::thread_start_work();
            workload.run_direct(config, fibres, miss_yield);
            profile::thread_end_work();
            profile::end_work();
#ifdef FARLIB_KVS_HAS_REQUEST_INTERVAL_DIAG
            request_interval_diag::end_stage();
#endif
        }).print();
        profile::print_profile_data();
        allocator::remote::remote_global_heap.print_used_memory();
    } else {
    SyncServer server(n_server_core, workload);

    profile::reset_all();
    perf_profile([&] {
        profile::start_work();
        profile::thread_start_work();
        run("warm-up", &server, &workload, config);
        profile::thread_end_work();
        profile::end_work();
    }).print();
    profile::print_profile_data();
    allocator::remote::remote_global_heap.print_used_memory();
    // run("warm-up", &server, &workload, config);
    if (run_async) {
        for (size_t i = 0; i < NEvalRepeat; i++) {
            PararoutineServer server(n_server_core, workload);
            run("async", &server, &workload, config);
        }
    }
    }
    #endif

    // No worker accesses the KV after this point. Stop background evacuation
    // before Workload destroys its far-memory objects.
    runtime_quiesce_cache();
    return successful;
}

int main(int argc, char* argv[]) {
    if (argc == 2 && std::strcmp(argv[1], "--describe-latency-mode") == 0) {
        std::fputs("kvs_latency_capability schema=1 arrival=poisson_per_fibre "
                   "include_queue_wait=1 hist_sample_period=1 "
                   "queue_deadline=drop_before_execution\n", stdout);
        return 0;
    }
    if (argc != 3) {
        std::cout << "usage: " << argv[0]
                  << " <configure file> <local memory (GB)>" << std::endl;
        return -1;
    }

    std::optional<LatencyOptions> latency_options;
    try {
        latency_options = latency_options_from_env();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "kvs_latency_error %s\n", error.what());
        return 2;
    }
    perf_init();
    Configure config;
    config.from_file(argv[1]);
    double local_mem = std::atof(argv[2]);
    ASSERT(config.max_thread_cnt >= 3);
    ASSERT(config.max_thread_cnt % 3 == 0);  // client : server = 1 : 1
    ASSERT(local_mem > 0);
    size_t n_server_core = config.max_thread_cnt / 3 * 2;
    config.client_buffer_size = local_mem * (1LL << 30);
    kvs_logging::write_record(stdout, "config: client buffer size = ",
                              config.client_buffer_size);
    runtime_init(config);
#ifndef FARLIB_KVS_NONFT_COMPAT
    kvs_logging::write_record(
        stdout, "kvs_scope_shards enabled=",
        Cache::get_default()->scope_counter_shards_enabled,
        " max_shards=", cache::ShardedScopeCounters::MaxShards,
        " bytes_per_shard=", sizeof(cache::ShardedScopeCounters::Shard));
    kvs_logging::write_record(
        stdout, "kvs_lock_wait_scope_yield enabled=",
        FarLib::detail::lock_wait_scope_yield_enabled());
#else
    kvs_logging::write_record(stdout, "kvs_scope_shards enabled=0 supported=0");
    kvs_logging::write_record(
        stdout, "kvs_lock_wait_scope_yield supported=0 legacy_check_memory_low=1");
#endif
    const bool successful = run(n_server_core, latency_options);
#ifndef FARLIB_KVS_NONFT_COMPAT
    if (Cache::get_default()->scope_counter_shards_enabled) {
        const auto v0 = Cache::get_default()->scope_counters.old_count(1);
        const auto v1 = Cache::get_default()->scope_counters.old_count(2);
        kvs_logging::write_record(
            stdout, "kvs_scope_shards_end registered=",
            cache::ShardedScopeCounters::registered_shards(), " v0=", v0,
            " v1=", v1);
        ASSERT(v0 == 0 && v1 == 0);
    }
#endif
    runtime_destroy();
    return successful ? 0 : 2;
}
