#pragma once

#include <netdb.h>
#include <sys/mman.h>
#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <sched.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#include "cache/region_based_allocator.hpp"
#include "config.hpp"
#include "read_batch_progress.hpp"
#include "read_batch_queue.hpp"
#include "rdma.hpp"
#include "rdma/sponge_rpc.hpp"
#include "utils/debug.hpp"
#include "utils/request_interval_diag.hpp"
#include "utils/wc_object_diag.hpp"
#include "utils/stats.hpp"

namespace FarLib {

namespace rdma {

constexpr size_t CHECK_CQ_BATCH_SIZE = 16;
constexpr size_t CHECK_CQ_BATCH_SIZE_SMALL = 1;

inline size_t configured_read_batch_size() {
    static const size_t value = [] {
        const char *text = std::getenv("FARLIB_RDMA_READ_BATCH");
        if (text == nullptr || *text == '\0') return size_t{1};
        const size_t parsed = std::strtoull(text, nullptr, 0);
        if (parsed != 1 && parsed != 2) {
            std::cerr << "FARLIB_RDMA_READ_BATCH must be 1 or 2" << std::endl;
            std::abort();
        }
        return parsed;
    }();
    return value;
}

inline bool read_batch_attribution_immediate_first() {
    static const bool enabled = [] {
        const char *text =
            std::getenv("FARLIB_RDMA_READ_BATCH_ATTRIBUTION");
        if (text == nullptr || *text == '\0' ||
            std::strcmp(text, "none") == 0) {
            return false;
        }
        if (std::strcmp(text, "immediate_first") == 0) {
            if (configured_read_batch_size() != 2) {
                std::cerr << "immediate_first requires READ batch 2"
                          << std::endl;
                std::abort();
            }
            return true;
        }
        std::cerr << "FARLIB_RDMA_READ_BATCH_ATTRIBUTION must be none or "
                     "immediate_first"
                  << std::endl;
        std::abort();
    }();
    return enabled;
}

inline const char *read_batch_attribution_mode() {
    return read_batch_attribution_immediate_first()
        ? "immediate_first"
        : "none";
}

inline constexpr uint64_t read_batch_max_delay_us() { return 5; }
inline constexpr uint64_t read_batch_max_delay_ns() {
    return read_batch_max_delay_us() * 1000;
}

inline uint64_t read_batch_monotonic_ns() {
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) std::abort();
    return static_cast<uint64_t>(now.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(now.tv_nsec);
}

struct ReadBatchSnapshot {
    uint64_t generated_reads = 0;
    uint64_t generated_bytes = 0;
    uint64_t physical_submit_calls = 0;
    uint64_t immediate_submit_calls = 0;
    uint64_t immediate_first_submit_attempts = 0;
    uint64_t attempted_chain_1 = 0;
    uint64_t attempted_chain_2 = 0;
    uint64_t accepted_prefix_0 = 0;
    uint64_t accepted_prefix_1 = 0;
    uint64_t accepted_prefix_2 = 0;
    uint64_t accepted_reads = 0;
    uint64_t accepted_bytes = 0;
    uint64_t partial_submit_calls = 0;
    uint64_t runtime_cooperative_yields = 0;
    uint64_t full_submit_attempts = 0;
    uint64_t age_flush_attempts = 0;
    uint64_t idle_no_switch_flush_attempts = 0;
    uint64_t credit_retry_attempts = 0;
    uint64_t blocked_submit_suppressed = 0;
    uint64_t age_progress_checks = 0;
    uint64_t no_ready_hook_calls = 0;
    uint64_t backpressure_yields = 0;
    uint64_t cq_progress_events = 0;
    uint64_t cq_progress_completions = 0;
    uint64_t enqueue_to_accept_cycles_sum = 0;
    uint64_t enqueue_to_accept_cycles_max = 0;
    uint64_t pending_reads = 0;

    ReadBatchSnapshot &operator+=(const ReadBatchSnapshot &other) {
        generated_reads += other.generated_reads;
        generated_bytes += other.generated_bytes;
        physical_submit_calls += other.physical_submit_calls;
        immediate_submit_calls += other.immediate_submit_calls;
        immediate_first_submit_attempts +=
            other.immediate_first_submit_attempts;
        attempted_chain_1 += other.attempted_chain_1;
        attempted_chain_2 += other.attempted_chain_2;
        accepted_prefix_0 += other.accepted_prefix_0;
        accepted_prefix_1 += other.accepted_prefix_1;
        accepted_prefix_2 += other.accepted_prefix_2;
        accepted_reads += other.accepted_reads;
        accepted_bytes += other.accepted_bytes;
        partial_submit_calls += other.partial_submit_calls;
        runtime_cooperative_yields += other.runtime_cooperative_yields;
        full_submit_attempts += other.full_submit_attempts;
        age_flush_attempts += other.age_flush_attempts;
        idle_no_switch_flush_attempts +=
            other.idle_no_switch_flush_attempts;
        credit_retry_attempts += other.credit_retry_attempts;
        blocked_submit_suppressed += other.blocked_submit_suppressed;
        age_progress_checks += other.age_progress_checks;
        no_ready_hook_calls += other.no_ready_hook_calls;
        backpressure_yields += other.backpressure_yields;
        cq_progress_events += other.cq_progress_events;
        cq_progress_completions += other.cq_progress_completions;
        enqueue_to_accept_cycles_sum += other.enqueue_to_accept_cycles_sum;
        enqueue_to_accept_cycles_max = std::max(
            enqueue_to_accept_cycles_max,
            other.enqueue_to_accept_cycles_max);
        pending_reads += other.pending_reads;
        return *this;
    }
};

struct ReadBatchOperation {
    read_batch::QueueStatus status = read_batch::QueueStatus::Ok;
    bool enqueued = false;
    bool submit_attempted = false;
    bool submit_suppressed = false;
    size_t accepted_count = 0;
    size_t pending_after = 0;
};

struct CompleteQueueArrayDeleter {
    size_t count = 0;
    void operator()(CompleteQueue *ptr) const {
        if (!ptr) return;
        for (size_t i = 0; i < count; i++) {
            ptr[i].~CompleteQueue();
        }
        std::free(ptr);
    }
};

using CompleteQueueArrayPtr = std::unique_ptr<CompleteQueue[], CompleteQueueArrayDeleter>;

// Endpoint control structure for multi-server support
struct EndpointControl {
    uint64_t remote_base_addr;
    uint32_t remote_key;
    size_t endpoint_idx;
    
    EndpointControl() : remote_base_addr(0), remote_key(0), endpoint_idx(0) {}
};

struct sg_entry {
    void *addr;
    uint32_t length;
    uint32_t lkey;
};

class Client;

// For now it is an arena allocator
class ConnectionPool {
private:
    struct Node {
        Node *next;
        size_t qp_idx;
    };

private:
    QueuePair *qps;
    CompleteQueue *cqs;
    std::atomic_size_t alloc_idx;
    std::atomic_size_t thread_id;
    size_t qp_count;
    size_t qp_per_client;

public:
    ConnectionPool()
        : qps(nullptr), alloc_idx(0), qp_count(0), qp_per_client(0) {}
    void init(QueuePair *qps, CompleteQueue *cqs, size_t qp_count, size_t qp_per_client) {
        this->qps = qps;
        this->cqs = cqs;
        this->alloc_idx = 0;
        this->qp_count = qp_count;
        this->qp_per_client = qp_per_client;
        this->thread_id = -1;
    }
    void add_client(Client *client);
};

extern ConnectionPool conn_pool;

class ClientControl {
private:
    void *buffer;
    Context ctx;
    ProtectionDomain pd;
    CompleteQueue control_cq;
    MemoryRegion mr;
    
    // Multi-endpoint support: vector of endpoint controls
    std::vector<EndpointControl> endpoints;
    size_t server_count;
    
    // Per-endpoint QPs/CQs: for each endpoint, we have control_qp, data_qps, data_cqs
    struct EndpointQPs {
        size_t data_qp_count;
        CompleteQueueArrayPtr data_cqs;
        std::unique_ptr<QueuePair[]> data_qps;
        std::unique_ptr<QueuePair> control_qp;
        
        EndpointQPs()
            : data_qp_count(0),
              data_cqs(nullptr),
              data_qps(nullptr),
              control_qp(nullptr) {}
    };
    std::vector<EndpointQPs> endpoint_qps;

    // --- EC (ft_method=sponge) commit RPC state ---------------------------
    // Everything below stays null/zero while ft_method == none: nothing
    // allocates it and no path reads it, so the default build is untouched.
    static constexpr size_t kSpongeCommitSendDepthPerEndpoint = 2;
    static constexpr size_t kSpongeAckRecvDepthPerEndpoint = 4;
    void *sponge_rpc_mem = nullptr;
    ibv_mr *sponge_rpc_mr = nullptr;
    SpongeClientCommitWireSendSlot *sponge_commit_send_base = nullptr;
    size_t sponge_commit_send_count = 0;
    SpongeAckWireRecvSlot *sponge_ack_recv_base = nullptr;
    size_t sponge_ack_recv_count = 0;
    // Per-endpoint rings of commit send slots, reused across commits.
    std::vector<std::vector<SpongeClientCommitWireSendSlot *>>
        sponge_commit_send_rings;
    std::vector<size_t> sponge_commit_send_heads;
    std::mutex sponge_send_mutex;

    void init_sponge_rpc_resources();
    void release_sponge_rpc_resources();
    bool post_sponge_ack_recv_slot_internal(SpongeAckWireRecvSlot &slot);
    SpongeClientCommitWireSendSlot *acquire_sponge_commit_send_slot(
        size_t endpoint_idx);
    bool owns_sponge_commit_send_slot(uint64_t wr_id) const;
    bool owns_sponge_ack_recv_slot(uint64_t wr_id) const;

    QueuePair *get_endpoint_control_qp(size_t endpoint_idx) const {
        if (endpoint_idx >= endpoint_qps.size()) return nullptr;
        QueuePair *qp = endpoint_qps[endpoint_idx].control_qp.get();
        if (qp == nullptr) return nullptr;
        return qp;
    }
    
    // Get endpoint-specific QP for data operations
    QueuePair* get_endpoint_data_qp(size_t endpoint_idx, size_t qp_idx) const {
        if (endpoint_idx >= endpoint_qps.size() || endpoint_qps[endpoint_idx].data_qps == nullptr) {
            return nullptr;
        }
        if (qp_idx >= endpoint_qps[endpoint_idx].data_qp_count) {
            qp_idx = 0;
        }
        return &endpoint_qps[endpoint_idx].data_qps[qp_idx];
    }

    CompleteQueue* get_endpoint_data_cq(size_t endpoint_idx, size_t qp_idx) const {
        if (endpoint_idx >= endpoint_qps.size() || endpoint_qps[endpoint_idx].data_cqs == nullptr) {
            return nullptr;
        }
        if (qp_idx >= endpoint_qps[endpoint_idx].data_qp_count) {
            return nullptr;
        }
        return &endpoint_qps[endpoint_idx].data_cqs[qp_idx];
    }
    
    // Get the data QP count per endpoint
    size_t get_endpoint_data_qp_count(size_t endpoint_idx) const {
        if (endpoint_idx >= endpoint_qps.size()) {
            return 0;
        }
        return endpoint_qps[endpoint_idx].data_qp_count;
    }
    
    // Get the total data QP/CQ count
    size_t get_data_cq_count() const {
        if (endpoint_qps.empty()) return 0;
        return endpoint_qps[0].data_qp_count;
    }
    
    // Legacy single-endpoint accessors (for backward compatibility)
    uint64_t remote_base_addr;
    uint32_t remote_key;
    const Configure &config;

    static std::unique_ptr<ClientControl> default_instance;

private:
    static bool check_ibv_post_send_ret(int ret) {
        switch (ret) {
        case 0:
            return true;
        case EINVAL:
            std::cerr << "ibv_post_send: Invalid value provided in wr"
                      << std::endl;
            break;
        case ENOMEM:
            return false;
        case EFAULT:
            std::cerr << "ibv_post_send: Invalid value provided in qp"
                      << std::endl;
            break;
        default:
            std::cerr << "ibv_post_send: failed, errno = " << errno
                      << std::endl;
            break;
        }
        abort();
    }

    void post_stop();

public:
    // Drain the control CQ (shared by every endpoint's control QP).  Only
    // called when ft_method != none; see Client::check_sponge_ack_cq().
    size_t poll_control_cq(ibv_wc *wc, size_t wc_length) {
        if (wc == nullptr || wc_length == 0) return 0;
        if (control_cq.complete_queue == nullptr) return 0;
        return ibv_poll_cq(control_cq.complete_queue, wc_length, wc);
    }

    bool post_sponge_commit_batch(size_t endpoint_idx, size_t qp_idx,
                                  const SpongeCommitBatchBuffer &batch);
    bool repost_sponge_ack_recv_slot(uint64_t wr_id);
    static bool release_sponge_commit_send_slot(uint64_t wr_id);
    static bool decode_sponge_ack_recv_slot(
        uint64_t wr_id, uint32_t byte_len, SpongeAckBatchView &view,
        SpongeAckWireRecvSlot **slot_out = nullptr);

private:

    static size_t mmap_length(size_t size) {
        const size_t page_size = 1 << 21;
        const size_t alignment = std::max(allocator::RegionSize, page_size);
        return (size % alignment == 0) ? size
                                       : (size / alignment + 1) * alignment;
    }

    static void *allocate_buffer(size_t size) {
        void *ptr =
            mmap(nullptr, mmap_length(size), PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT),
                 -1, 0);
        if (ptr == MAP_FAILED) {
            ERROR("mmap failed");
        }
        return ptr;
    }

    static void deallocate_buffer(void *ptr, size_t size) {
        CHECK_ERR(munmap(ptr, mmap_length(size)));
    }

public:
    ClientControl(const Configure &config);

    ~ClientControl() {
        INFO("destroying rdma client");
        post_stop();
        release_sponge_rpc_resources();
        deallocate_buffer(buffer, config.client_buffer_size);
    }

    static ClientControl *init_default(const Configure &config) {
#ifndef NO_REMOTE
        default_instance.reset(new ClientControl(config));
#endif
        return default_instance.get();
    }

    static ClientControl *get_default() { return default_instance.get(); }

    static void destroy_default() { default_instance.reset(); }

    QueuePair *get_endpoint_data_qps_ptr(size_t endpoint_idx) const {
        if (endpoint_idx >= endpoint_qps.size()) {
            return nullptr;
        }
        return endpoint_qps[endpoint_idx].data_qps.get();
    }

    CompleteQueue *get_endpoint_data_cqs_ptr(size_t endpoint_idx) const {
        if (endpoint_idx >= endpoint_qps.size()) {
            return nullptr;
        }
        return endpoint_qps[endpoint_idx].data_cqs.get();
    }

    void *get_buffer() const { return buffer; }
    ibv_pd *get_protection_domain() const { return pd.protection_domain; }
    
    // Multi-endpoint accessors
    size_t get_server_count() const { return server_count; }
    
    const EndpointControl& get_endpoint(size_t idx) const {
        if (idx >= server_count) {
            std::cerr << "Error: endpoint index out of range" << std::endl;
            std::abort();
        }
        return endpoints[idx];
    }
    
    // Get endpoint by remote_addr using config's mapping
    std::pair<size_t, uint64_t> get_endpoint_and_offset(uint64_t remote_addr) const {
        auto [endpoint, offset] = config.map_remote_addr(remote_addr);
        return {endpoint, offset};
    }
    
    // Legacy accessor for backward compatibility
    uint64_t get_remote_base_addr() const { return remote_base_addr; }
    uint32_t get_remote_key() const { return remote_key; }

    // DEBUG ONLY
    void check_event() {
        // only for debug
        std::cout << "check event" << std::endl;
        ibv_async_event event;
        int ret = ibv_get_async_event(ctx.context, &event);
        ibv_ack_async_event(&event);
        std::cout << event.event_type << std::endl;
    }

    friend class Client;
};

// extern thread_local Client client;
extern std::vector<Client> clients;
extern std::atomic_size_t thread_id_global;
bool thread_id_registry_enabled();
size_t get_registered_thread_id();
void reset_thread_id_registry(size_t app_worker_count);
void disable_thread_id_registry();
void register_thread_id(pthread_t tid, size_t thread_id);
void seal_thread_id_registry();

struct ThreadInfo {
    ThreadInfo()
        : thread_id(thread_id_registry_enabled()
                        ? get_registered_thread_id()
                        : thread_id_global.fetch_add(
                              1, std::memory_order_relaxed)) {
        const char *pin = std::getenv("FARLIB_PIN_RDMA_THREAD");
        if (pin && pin[0] == '1') {
            long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
            if (cpu_count > 0) {
                int cpu = static_cast<int>(thread_id % static_cast<size_t>(cpu_count));
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cpu, &set);
                (void)sched_setaffinity(0, sizeof(set), &set);
            }
        }
    }
    ~ThreadInfo() {}
    size_t thread_id;
};

extern thread_local ThreadInfo thread_info;

Client *get_client(size_t thread_id);
size_t get_client_count();
void install_read_batch_yield_progress();
void remove_read_batch_yield_progress();

class Client {
private:
    struct QueuedRead {
        std::size_t remote_offset = 0;
        void *local_addr = nullptr;
        uint32_t length = 0;
        uint64_t wr_id = 0;
        uint64_t obj_id = 0;
        size_t output_qp_idx = 0;
        size_t endpoint_idx = 0;
        uint64_t enqueue_tsc = 0;
        uint64_t enqueue_ns = 0;
    };
    using ReadQueue = read_batch::PrefixQueue<QueuedRead, 2>;
    struct ReadQueueState {
        ReadQueue queue;
        ReadBatchSnapshot diag;
        std::atomic<uint64_t> cq_progress_generation{0};
        std::atomic<uint64_t> cq_progress_events{0};
        std::atomic<uint64_t> cq_progress_completions{0};
        read_batch::CreditAgePolicy progress_policy;
    };

    QueuePair *qps; // QP belong to this POSIX thread
    QueuePair *global_qps; // QP belong to all POSIX threads
    CompleteQueue *cqs;
    size_t qp_count;
    size_t qp_idx;
    size_t thread_id; // also the global qp_idx
    int64_t in_flight_pkt_num;
    // client local lock, each thread has its own lock
    std::atomic_flag thread_lock;
    std::vector<QueuePair *> endpoint_data_qps;
    std::vector<CompleteQueue *> endpoint_data_cqs;
    std::unique_ptr<ReadQueueState[]> read_queue_states;
    size_t read_queue_state_count;
    ReadBatchSnapshot immediate_read_diag;
    std::atomic<uint64_t> no_ready_hook_calls{0};

    ReadQueueState *get_read_queue_state(size_t endpoint_idx,
                                         size_t local_qp_idx) {
        if (read_queue_states == nullptr || local_qp_idx >= qp_count) {
            return nullptr;
        }
        const size_t index = endpoint_idx * qp_count + local_qp_idx;
        if (index >= read_queue_state_count) {
            return nullptr;
        }
        return &read_queue_states[index];
    }

    size_t post_queued_read_prefix(ReadQueueState &state,
                                   const QueuedRead *requests,
                                   size_t count,
                                   read_batch::ProgressReason reason) {
        if (count == 0 || count > 2) {
            std::abort();
        }
        ibv_sge read_sges[2]{};
        ibv_send_wr read_wrs[2]{};
        const size_t endpoint_idx = requests[0].endpoint_idx;
        const size_t output_qp_idx = requests[0].output_qp_idx;
        for (size_t i = 0; i < count; ++i) {
            if (requests[i].endpoint_idx != endpoint_idx ||
                requests[i].output_qp_idx != output_qp_idx) {
                std::abort();
            }
            build_send_wr(read_wrs[i], read_sges[i],
                          requests[i].remote_offset,
                          requests[i].local_addr, requests[i].length,
                          requests[i].wr_id, true, IBV_WR_RDMA_READ,
                          endpoint_idx);
            if (i != 0) {
                read_wrs[i - 1].next = &read_wrs[i];
            }
        }

        ensure_endpoint_views();
        QueuePair *endpoint_qp =
            get_endpoint_data_qp_local(endpoint_idx, output_qp_idx);
        if (endpoint_qp == nullptr) {
            endpoint_qp = &qps[output_qp_idx];
        }
        ibv_send_wr *bad_wr = nullptr;
        const uint64_t generation_before =
            state.cq_progress_generation.load(std::memory_order_acquire);
        ++state.diag.physical_submit_calls;
        if (count == 1) ++state.diag.attempted_chain_1;
        if (count == 2) ++state.diag.attempted_chain_2;
        switch (reason) {
        case read_batch::ProgressReason::ImmediateFirst:
            ++state.diag.immediate_first_submit_attempts;
            break;
        case read_batch::ProgressReason::Full:
            ++state.diag.full_submit_attempts;
            break;
        case read_batch::ProgressReason::Age:
            ++state.diag.age_flush_attempts;
            break;
        case read_batch::ProgressReason::IdleNoSwitch:
            ++state.diag.idle_no_switch_flush_attempts;
            break;
        case read_batch::ProgressReason::Credit:
            ++state.diag.credit_retry_attempts;
            break;
        case read_batch::ProgressReason::None:
            std::abort();
        }
        const int send_ret =
            ibv_post_send(endpoint_qp->queue_pair, read_wrs, &bad_wr);

        size_t accepted = count;
        if (send_ret != 0) {
            (void)check_ibv_post_send_ret(send_ret);
            accepted = count;
            if (bad_wr == nullptr) {
                std::cerr << "ibv_post_send failed without bad_wr" << std::endl;
                std::abort();
            } else {
                bool found = false;
                for (size_t i = 0; i < count; ++i) {
                    if (bad_wr == &read_wrs[i]) {
                        accepted = i;
                        found = true;
                        break;
                    }
                }
                if (!found) std::abort();
            }
        }
        if (accepted == 0) ++state.diag.accepted_prefix_0;
        if (accepted == 1) ++state.diag.accepted_prefix_1;
        if (accepted == 2) ++state.diag.accepted_prefix_2;
        if (accepted != 0 && accepted != count) {
            ++state.diag.partial_submit_calls;
        }
        // The generation is captured before posting. If a different worker
        // polls the sole freeing completion during post_send, the current
        // generation will differ and the suffix is immediately eligible for
        // a credit-triggered retry.
        state.progress_policy.on_submission_result(generation_before,
                                                   accepted, count);

        const uint64_t accepted_tsc = __rdtsc();
        size_t accepted_bytes = 0;
        for (size_t i = 0; i < accepted; ++i) {
            accepted_bytes += requests[i].length;
            const uint64_t latency = accepted_tsc - requests[i].enqueue_tsc;
            state.diag.enqueue_to_accept_cycles_sum += latency;
            state.diag.enqueue_to_accept_cycles_max = std::max(
                state.diag.enqueue_to_accept_cycles_max, latency);
        }
        state.diag.accepted_reads += accepted;
        state.diag.accepted_bytes += accepted_bytes;
        profile::count_rdma_read_posts(accepted, accepted_bytes);
        return accepted;
    }

    ReadBatchOperation apply_queue_result(
        const typename ReadQueue::Result &result) const {
        return {.status = result.status,
                .enqueued = result.enqueued,
                .submit_attempted = result.submit_attempted,
                .submit_suppressed = result.submit_suppressed,
                .accepted_count = result.accepted_count,
                .pending_after = result.pending_after};
    }

public:
    // TODO: can we remove this TLS variable constructor?
    Client() : qps(nullptr), read_queue_state_count(0) {
        conn_pool.add_client(this);
    }
    ~Client() {}
    
    // Delete copy constructor and copy assignment operator since std::atomic_flag is not copyable
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    
    // Add move constructor and move assignment operator
    Client(Client&& other) noexcept 
        : qps(other.qps), global_qps(other.global_qps), cqs(other.cqs),
          qp_count(other.qp_count), qp_idx(other.qp_idx), 
          thread_id(other.thread_id), in_flight_pkt_num(other.in_flight_pkt_num),
          endpoint_data_qps(std::move(other.endpoint_data_qps)),
          endpoint_data_cqs(std::move(other.endpoint_data_cqs)),
          read_queue_states(std::move(other.read_queue_states)),
          read_queue_state_count(other.read_queue_state_count),
          immediate_read_diag(other.immediate_read_diag),
          no_ready_hook_calls(other.no_ready_hook_calls.load(
              std::memory_order_relaxed)) {
        // atomic_flag is default-initialized to false, reset to match other if needed
        bool locked = other.thread_lock.test_and_set();
        if (!locked) {
            thread_lock.clear();
        } else {
            thread_lock.test_and_set();
        }
        conn_pool.add_client(this);
    }
    
    Client& operator=(Client&& other) noexcept {
        if (this != &other) {
            qps = other.qps;
            global_qps = other.global_qps;
            cqs = other.cqs;
            qp_count = other.qp_count;
            qp_idx = other.qp_idx;
            thread_id = other.thread_id;
            in_flight_pkt_num = other.in_flight_pkt_num;
            endpoint_data_qps = std::move(other.endpoint_data_qps);
            endpoint_data_cqs = std::move(other.endpoint_data_cqs);
            read_queue_states = std::move(other.read_queue_states);
            read_queue_state_count = other.read_queue_state_count;
            immediate_read_diag = other.immediate_read_diag;
            no_ready_hook_calls.store(
                other.no_ready_hook_calls.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            // atomic_flag: clear and set to match other
            thread_lock.clear();
            if (other.thread_lock.test()) {
                thread_lock.test_and_set();
            }
        }
        return *this;
    }
    void init() {
        if (qps == nullptr) conn_pool.add_client(this);
    }
    void init(QueuePair *qps, CompleteQueue *cqs, size_t qp_count, size_t thread_id) {
        if (this->qps != nullptr) {
            ERROR("client double init");
        }
        this->qps = qps;
        this->cqs = cqs;
        this->qp_count = qp_count;
        this->qp_idx = 0;
        this->thread_id = thread_id;
        this->thread_lock.clear();
        this->in_flight_pkt_num = 0;
        this->read_queue_state_count = 0;
        this->immediate_read_diag = {};
        this->no_ready_hook_calls.store(0, std::memory_order_relaxed);
        setup_endpoint_views();
    }

    int64_t get_in_flight_pkt_num() const { return in_flight_pkt_num; }

    void add_in_flight_pkt_num(int64_t num) { in_flight_pkt_num += num; }
    void sub_in_flight_pkt_num(int64_t num) { in_flight_pkt_num -= num; }
    bool test_thread_lock() { return thread_lock.test(); }
    bool set_thread_lock() { return thread_lock.test_and_set(); }
    void clear_thread_lock() { thread_lock.clear(); }

    __attribute__((noinline)) size_t get_thread_id() const { return thread_id; }
    __attribute__((noinline)) size_t get_qp_idx() const { return qp_idx; }
    void *get_buffer() { return ClientControl::get_default()->get_buffer(); }

    static bool check_ibv_post_send_ret(int ret) {
        return ClientControl::check_ibv_post_send_ret(ret);
    }

    void setup_endpoint_views() {
        auto *control = ClientControl::get_default();
        if (control == nullptr) {
            return;
        }
        size_t endpoint_count = control->get_server_count();
        if (endpoint_count == 0) {
            return;
        }
        endpoint_data_qps.resize(endpoint_count, nullptr);
        endpoint_data_cqs.resize(endpoint_count, nullptr);
        for (size_t ep = 0; ep < endpoint_count; ep++) {
            QueuePair *base_qps = control->get_endpoint_data_qps_ptr(ep);
            CompleteQueue *base_cqs = control->get_endpoint_data_cqs_ptr(ep);
            if (base_qps != nullptr) {
                endpoint_data_qps[ep] = base_qps + thread_id;
            }
            if (base_cqs != nullptr) {
                endpoint_data_cqs[ep] = base_cqs + thread_id;
            }
        }
        if (configured_read_batch_size() == 2 &&
            read_queue_states == nullptr && qp_count != 0) {
            read_queue_state_count = endpoint_count * qp_count;
            read_queue_states =
                std::make_unique<ReadQueueState[]>(read_queue_state_count);
        }
    }

    void ensure_endpoint_views() {
        if (endpoint_data_qps.empty() || endpoint_data_cqs.empty()) {
            setup_endpoint_views();
        }
    }

    bool sync_read_batching_enabled() const {
        return configured_read_batch_size() == 2;
    }

    size_t get_local_qp_count() const { return qp_count; }
    size_t get_endpoint_count() const { return endpoint_data_qps.size(); }

    ReadBatchOperation enqueue_sync_read(
        std::size_t remote_offset, void *local_addr, uint32_t length,
        uint64_t wr_id, uint64_t obj_id, size_t output_qp_idx,
        size_t endpoint_idx) {
        ensure_endpoint_views();
        ReadQueueState *state =
            get_read_queue_state(endpoint_idx, output_qp_idx);
        if (state == nullptr) std::abort();
        const QueuedRead request{remote_offset, local_addr, length, wr_id,
                                 obj_id, output_qp_idx, endpoint_idx,
                                 __rdtsc(), read_batch_monotonic_ns()};
        read_batch::ProgressReason reason = read_batch::ProgressReason::Full;
        const bool immediate_first =
            read_batch_attribution_immediate_first();
        const size_t submit_at_count = immediate_first ? 1 : 2;
        auto result = state->queue.try_enqueue_controlled_at(
            request, submit_at_count, [&](const QueuedRead *, size_t) {
                const uint64_t generation =
                    state->cq_progress_generation.load(
                        std::memory_order_acquire);
                const auto decision = immediate_first
                    ? state->progress_policy.on_immediate_first(generation)
                    : state->progress_policy.on_full(generation);
                reason = decision.reason;
                if (!decision.submit) {
                    ++state->diag.blocked_submit_suppressed;
                }
                return decision.submit;
            }, [&](const QueuedRead *requests, size_t count) {
                return post_queued_read_prefix(*state, requests, count,
                                               reason);
            }, [&](const QueuedRead &enqueued) {
                ++state->diag.generated_reads;
                state->diag.generated_bytes += enqueued.length;
            });
        return apply_queue_result(result);
    }

    ReadBatchOperation try_progress_sync_reads(
        size_t endpoint_idx, size_t output_qp_idx,
        bool idle_no_switch) {
        if (!sync_read_batching_enabled()) return {};
        ensure_endpoint_views();
        ReadQueueState *state =
            get_read_queue_state(endpoint_idx, output_qp_idx);
        if (state == nullptr) {
            return {};
        }
        read_batch::ProgressReason reason = read_batch::ProgressReason::None;
        auto result = state->queue.try_flush_if(
            [&](const QueuedRead *requests, size_t count) {
                if (count == 0) return false;
                const uint64_t generation =
                    state->cq_progress_generation.load(
                        std::memory_order_acquire);
                uint64_t now_ns = 0;
                if (!state->progress_policy.blocked_on_credit() &&
                    !idle_no_switch) {
                    ++state->diag.age_progress_checks;
                    now_ns = read_batch_monotonic_ns();
                }
                const auto decision = state->progress_policy.on_progress(
                    generation, requests[0].enqueue_ns, now_ns,
                    read_batch_max_delay_ns(), idle_no_switch);
                reason = decision.reason;
                if (!decision.submit &&
                    state->progress_policy.blocked_on_credit()) {
                    ++state->diag.blocked_submit_suppressed;
                }
                return decision.submit;
            }, [&](const QueuedRead *requests, size_t count) {
                return post_queued_read_prefix(*state, requests, count,
                                               reason);
            });
        return apply_queue_result(result);
    }

    void note_cq_progress(size_t endpoint_idx, size_t output_qp_idx,
                          size_t completed) {
        if (!sync_read_batching_enabled() || completed == 0) return;
        ReadQueueState *state =
            get_read_queue_state(endpoint_idx, output_qp_idx);
        if (state == nullptr) return;
        state->cq_progress_generation.fetch_add(1,
                                                 std::memory_order_release);
        state->cq_progress_events.fetch_add(1, std::memory_order_relaxed);
        state->cq_progress_completions.fetch_add(
            completed, std::memory_order_relaxed);
    }

    void progress_all_sync_reads(bool idle_no_switch) {
        if (!sync_read_batching_enabled()) return;
        ensure_endpoint_views();
        if (idle_no_switch) {
            no_ready_hook_calls.fetch_add(1, std::memory_order_relaxed);
        }
        for (size_t endpoint_idx = 0;
             endpoint_idx < get_endpoint_count(); ++endpoint_idx) {
            for (size_t output_qp_idx = 0;
                 output_qp_idx < get_local_qp_count(); ++output_qp_idx) {
                (void)try_progress_sync_reads(endpoint_idx, output_qp_idx,
                                              idle_no_switch);
            }
        }
    }

    ReadBatchOperation try_observe_sync_reads(size_t endpoint_idx,
                                              size_t output_qp_idx) const {
        if (read_queue_states == nullptr || output_qp_idx >= qp_count) {
            return {};
        }
        const size_t index = endpoint_idx * qp_count + output_qp_idx;
        if (index >= read_queue_state_count) return {};
        return apply_queue_result(read_queue_states[index].queue.try_observe());
    }

    void record_immediate_read_generated(size_t bytes) {
        auto &diag = get_client(thread_info.thread_id)->immediate_read_diag;
        ++diag.generated_reads;
        diag.generated_bytes += bytes;
    }

    void record_backpressure_yield() {
        auto &diag = get_client(thread_info.thread_id)->immediate_read_diag;
        ++diag.backpressure_yields;
    }

    void reset_read_batch_diagnostics() {
        if (read_queue_states != nullptr) {
            for (size_t i = 0; i < read_queue_state_count; ++i) {
                bool reset = false;
                bool nonempty = false;
                for (size_t attempt = 0; attempt < 10000 && !reset; ++attempt) {
                    auto inspected = read_queue_states[i].queue.try_inspect(
                        [&](size_t pending) {
                            nonempty = pending != 0;
                            if (!nonempty) {
                                read_queue_states[i].diag = {};
                                read_queue_states[i].progress_policy.reset();
                                read_queue_states[i].cq_progress_events.store(
                                    0, std::memory_order_relaxed);
                                read_queue_states[i].cq_progress_completions.store(
                                    0, std::memory_order_relaxed);
                            }
                        });
                    if (inspected.status == read_batch::QueueStatus::Ok) {
                        reset = true;
                    } else {
                        std::this_thread::yield();
                    }
                }
                if (!reset || nonempty) {
                    std::cerr << "read batch diagnostic reset with active queue"
                              << std::endl;
                    std::abort();
                }
            }
        }
        immediate_read_diag = {};
        no_ready_hook_calls.store(0, std::memory_order_relaxed);
    }

    ReadBatchSnapshot snapshot_read_batch_diagnostics() const {
        ReadBatchSnapshot result = immediate_read_diag;
        result.immediate_submit_calls =
            immediate_read_diag.physical_submit_calls;
        result.no_ready_hook_calls +=
            no_ready_hook_calls.load(std::memory_order_relaxed);
        if (read_queue_states != nullptr) {
            for (size_t i = 0; i < read_queue_state_count; ++i) {
                bool copied = false;
                ReadBatchSnapshot queue_diag;
                size_t pending = 0;
                for (size_t attempt = 0; attempt < 10000 && !copied; ++attempt) {
                    auto inspected = read_queue_states[i].queue.try_inspect(
                        [&](size_t count) {
                            queue_diag = read_queue_states[i].diag;
                            queue_diag.cq_progress_events +=
                                read_queue_states[i].cq_progress_events.load(
                                    std::memory_order_relaxed);
                            queue_diag.cq_progress_completions +=
                                read_queue_states[i]
                                    .cq_progress_completions.load(
                                        std::memory_order_relaxed);
                            pending = count;
                        });
                    if (inspected.status == read_batch::QueueStatus::Ok) {
                        copied = true;
                    } else {
                        std::this_thread::yield();
                    }
                }
                if (!copied) {
                    std::cerr << "read batch diagnostic snapshot raced queue"
                              << std::endl;
                    std::abort();
                }
                result += queue_diag;
                result.pending_reads += pending;
            }
        }
        return result;
    }

    QueuePair *get_endpoint_data_qp_local(size_t endpoint_idx, size_t local_qp_idx) {
        if (endpoint_idx >= endpoint_data_qps.size()) {
            return nullptr;
        }
        QueuePair *base = endpoint_data_qps[endpoint_idx];
        if (base == nullptr) {
            return nullptr;
        }
        if (local_qp_idx >= qp_count) {
            local_qp_idx = 0;
        }
        return &base[local_qp_idx];
    }

    CompleteQueue *get_endpoint_data_cq_local(size_t endpoint_idx, size_t local_qp_idx) {
        if (endpoint_idx >= endpoint_data_cqs.size()) {
            return nullptr;
        }
        CompleteQueue *base = endpoint_data_cqs[endpoint_idx];
        if (base == nullptr) {
            return nullptr;
        }
        if (local_qp_idx >= qp_count) {
            return nullptr;
        }
        return &base[local_qp_idx];
    }

    uint32_t registered_buffer_lkey() const {
        return ClientControl::get_default()->mr.memory_region->lkey;
    }

    void build_send_wr(ibv_send_wr &wr, ibv_sge &sge, std::size_t remote_offset,
                       void *local_addr, uint32_t length, uint64_t wr_id,
                       bool signal = true,
                       ibv_wr_opcode opcode = IBV_WR_RDMA_READ,
                       size_t endpoint_idx = 0) {
        MemoryRegion &mr = ClientControl::get_default()->mr;
        sge.addr = reinterpret_cast<uint64_t>(local_addr);
        sge.length = length;
        sge.lkey = mr.memory_region->lkey;
        
        wr.wr_id = wr_id;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = opcode;
        wr.send_flags = signal ? IBV_SEND_SIGNALED : 0;
        
        // Multi-endpoint: use endpoint-specific remote address and key
        const auto& endpoint = ClientControl::get_default()->get_endpoint(endpoint_idx);
        wr.wr.rdma = {
            .remote_addr = endpoint.remote_base_addr + remote_offset,
            .rkey = endpoint.remote_key};
    }
    
    // Legacy version for backward compatibility (uses endpoint 0)
    void build_send_wr_legacy(ibv_send_wr &wr, ibv_sge &sge, std::size_t remote_offset,
                       void *local_addr, uint32_t length, uint64_t wr_id,
                       bool signal = true,
                       ibv_wr_opcode opcode = IBV_WR_RDMA_READ) {
        build_send_wr(wr, sge, remote_offset, local_addr, length, wr_id, signal, opcode, 0);
    }

    // sg_entry and sge have the same length (sge_num)
    void prepare_sge_list(ibv_sge *sge, sg_entry *sg_entry, size_t sge_num, void *local_addr, uint32_t length) {
        MemoryRegion &mr = ClientControl::get_default()->mr;
        for (size_t i = 0; i < sge_num; i++) {
            sge[i].addr = reinterpret_cast<uint64_t>(sg_entry[i].addr);
            sge[i].length = sg_entry[i].length;
            sge[i].lkey = mr.memory_region->lkey;
        }
    }

    void build_send_wr_with_multi_sges(ibv_send_wr &wr, ibv_sge *sge, size_t sge_num, std::size_t remote_offset,
                       uint64_t wr_id, bool signal = true, 
                       ibv_wr_opcode opcode = IBV_WR_RDMA_READ,
                       size_t endpoint_idx = 0) {
        wr.wr_id = wr_id;
        wr.next = nullptr;
        wr.sg_list = sge;
        wr.num_sge = sge_num;
        wr.opcode = opcode;
        wr.send_flags = signal ? IBV_SEND_SIGNALED : 0;
        
        // Multi-endpoint: use endpoint-specific remote address and key
        const auto& endpoint = ClientControl::get_default()->get_endpoint(endpoint_idx);
        wr.wr.rdma = {
            .remote_addr = endpoint.remote_base_addr + remote_offset,
            .rkey = endpoint.remote_key};
    }

    void update_qp_idx() { qp_idx = (qp_idx + 1 >= qp_count) ? 0 : qp_idx + 1; }

    // return true if success
    // return false if send queue if full
    // abort if other error occurs
    template <bool Signal = true>
    bool post_read(std::size_t remote_offset, void *local_addr, uint32_t length,
                   uint64_t wr_id, uint64_t obj_id, size_t output_qp_idx = 0,
                   size_t endpoint_idx = 0,
                   const uint32_t *local_lkey_override = nullptr) {
        const uint64_t enqueue_tsc = __rdtsc();
        ibv_sge read_sge;
        ibv_send_wr read_wr;
        ibv_send_wr *bad_wr;
        build_send_wr(read_wr, read_sge, remote_offset, local_addr, length,
                      wr_id, Signal, IBV_WR_RDMA_READ, endpoint_idx);
        // Only grow-on-demand recovery buffers use a different registered MR.
        // Ordinary calls keep the nullptr default (folded away when inlined).
        if (local_lkey_override != nullptr) read_sge.lkey = *local_lkey_override;
        
        // Use endpoint-specific QP
        ensure_endpoint_views();
        QueuePair* endpoint_qp = get_endpoint_data_qp_local(endpoint_idx, output_qp_idx);
        if (endpoint_qp == nullptr) {
            // Fallback to old behavior for single-server mode
            endpoint_qp = &qps[output_qp_idx];
        }
        auto &diag = get_client(thread_info.thread_id)->immediate_read_diag;
        ++diag.physical_submit_calls;
        ++diag.attempted_chain_1;
        ::FarLib::request_interval_diag::post_attempt_begin();
        auto *wc_sample = ::FarLib::wc_object_diag::find_object(obj_id);
        const uint64_t wc_submit_begin = wc_sample ? ::FarLib::wc_object_diag::stamp() : 0;
        int send_ret = ibv_post_send(endpoint_qp->queue_pair, &read_wr, &bad_wr);
        const uint64_t wc_submit_return = wc_sample ? ::FarLib::wc_object_diag::stamp() : 0;
        bool ret = check_ibv_post_send_ret(send_ret);
        if (wc_sample) {
            wc_sample->post_attempts.fetch_add(1);
            if (ret) {
                wc_sample->submit_begin.store(wc_submit_begin);
                wc_sample->submit_return.store(wc_submit_return);
            }
        }
        ::FarLib::request_interval_diag::post_attempt_result(ret);
        if (ret) {
            ++diag.accepted_prefix_1;
            ++diag.accepted_reads;
            diag.accepted_bytes += length;
            const uint64_t latency = __rdtsc() - enqueue_tsc;
            diag.enqueue_to_accept_cycles_sum += latency;
            diag.enqueue_to_accept_cycles_max = std::max(
                diag.enqueue_to_accept_cycles_max, latency);
            profile::count_rdma_read_post(length);
        } else {
            ++diag.accepted_prefix_0;
        }
        return ret;
    }
    
    // Endpoint-aware version: derives endpoint from remote_addr
    template <bool Signal = true>
    bool post_read_with_addr(uint64_t remote_addr, void *local_addr, uint32_t length,
                             uint64_t wr_id, uint64_t obj_id, size_t output_qp_idx) {
        auto [endpoint_idx, offset] = ClientControl::get_default()->get_endpoint_and_offset(remote_addr);
        return post_read<Signal>(offset, local_addr, length, wr_id, obj_id, output_qp_idx, endpoint_idx);
    }

    // return true if success
    // return false if send queue if full
    // abort if other error occurs
    template <bool Signal = true>
    bool post_write(std::size_t remote_offset, void *local_addr,
                    uint32_t length, uint64_t wr_id, uint64_t obj_id,
                    size_t output_qp_idx = 0, size_t endpoint_idx = 0,
                    const uint32_t *local_lkey_override = nullptr) {
        ibv_sge write_sge;
        ibv_send_wr write_wr;
        ibv_send_wr *bad_wr;
        build_send_wr(write_wr, write_sge, remote_offset, local_addr, length,
                      wr_id, Signal, IBV_WR_RDMA_WRITE, endpoint_idx);
        // Grow-on-demand recovery/staging buffers may be registered in a
        // provider-owned MR rather than the client's original MR.
        if (local_lkey_override != nullptr) {
            write_sge.lkey = *local_lkey_override;
        }
        
        // Use endpoint-specific QP
        ensure_endpoint_views();
        QueuePair* endpoint_qp = get_endpoint_data_qp_local(endpoint_idx, output_qp_idx);
        if (endpoint_qp == nullptr) {
            // Fallback to old behavior for single-server mode
            endpoint_qp = &qps[output_qp_idx];
        }
        int send_ret = ibv_post_send(endpoint_qp->queue_pair, &write_wr, &bad_wr);
        bool ret = check_ibv_post_send_ret(send_ret);
        return ret;
    }

    // Endpoint-aware version: derives endpoint from remote_addr
    template <bool Signal = true>
    bool post_write_with_addr(uint64_t remote_addr, void *local_addr,
                              uint32_t length, uint64_t wr_id, uint64_t obj_id,
                              size_t output_qp_idx = 0,
                              const uint32_t *local_lkey_override = nullptr) {
        auto [endpoint_idx, offset] = ClientControl::get_default()->get_endpoint_and_offset(remote_addr);
        return post_write<Signal>(offset, local_addr, length, wr_id, obj_id,
                                  output_qp_idx, endpoint_idx,
                                  local_lkey_override);
    }

    // return true if success
    // return false if send queue if full
    // abort if other error occurs
    bool post_writes(ibv_send_wr *write_wr, ibv_send_wr **bad_wr,
                     size_t output_qp_idx = 0, size_t endpoint_idx = 0) {
        // Use endpoint-specific QP
        ensure_endpoint_views();
        QueuePair* endpoint_qp = get_endpoint_data_qp_local(endpoint_idx, output_qp_idx);
        if (endpoint_qp == nullptr) {
            // Fallback to old behavior for single-server mode
            endpoint_qp = &qps[output_qp_idx];
        }
        int send_ret = ibv_post_send(endpoint_qp->queue_pair, write_wr, bad_wr);
        bool return_value = check_ibv_post_send_ret(send_ret);
        return return_value;
    }

    // --- EC (ft_method=sponge) commit RPC forwarding ----------------------
    // Thin per-thread front ends over ClientControl.  All of them are dead
    // code while ft_method == none (see the gating in ConcurrentArrayCache).
    bool post_sponge_commit_batch(size_t endpoint_idx, size_t local_qp_idx,
                                  const SpongeCommitBatchBuffer &batch) {
        auto *control = ClientControl::get_default();
        if (control == nullptr) return false;
        return control->post_sponge_commit_batch(endpoint_idx, local_qp_idx,
                                                 batch);
    }

    bool repost_sponge_ack_recv_slot(uint64_t wr_id) {
        auto *control = ClientControl::get_default();
        if (control == nullptr) return false;
        return control->repost_sponge_ack_recv_slot(wr_id);
    }

    static bool release_sponge_commit_send_slot(uint64_t wr_id) {
        auto *control = ClientControl::get_default();
        if (control == nullptr) return false;
        return control->release_sponge_commit_send_slot(wr_id);
    }

    static bool decode_sponge_ack_recv_slot(
        uint64_t wr_id, uint32_t byte_len, SpongeAckBatchView &view,
        SpongeAckWireRecvSlot **slot_out = nullptr) {
        return ClientControl::decode_sponge_ack_recv_slot(wr_id, byte_len, view,
                                                          slot_out);
    }

    // Drain the control CQ (sponge ACK batches and commit SEND completions).
    size_t check_sponge_ack_cq(ibv_wc *wc, size_t wc_length) {
        auto *control = ClientControl::get_default();
        if (control == nullptr) return 0;
        return control->poll_control_cq(wc, wc_length);
    }

    static size_t check_cq(ibv_wc *wc, size_t wc_length) {
        std::cout << "check_cq, should not be called" << std::endl;
        auto *control = ClientControl::get_default();
        auto *cq = control->get_endpoint_data_cq(0, 0);
        if (cq == nullptr) return 0;
        return ibv_poll_cq(cq->complete_queue, wc_length, wc);
    }

    // Check CQ for a specific QP index on a specific endpoint.
    size_t check_cq_with_idx_endpoint(ibv_wc *wc, size_t wc_length,
                                      size_t input_qp_idx, size_t endpoint_idx) {
        ensure_endpoint_views();
        auto *cq = get_endpoint_data_cq_local(endpoint_idx, input_qp_idx);
        if (cq == nullptr || cq->complete_queue == nullptr) {
            return 0;
        }
        return ibv_poll_cq(cq->complete_queue, wc_length, wc);
    }

    // Check CQ for a specific QP index across all endpoints.
    size_t check_cq_with_idx(ibv_wc *wc, size_t wc_length,
                                    size_t input_qp_idx) {
        ensure_endpoint_views();
        auto* control = ClientControl::get_default();
        size_t data_cq_count = control->get_data_cq_count();
        if (input_qp_idx >= data_cq_count) {
            return 0;
        }

        size_t total_cnt = 0;
        size_t endpoint_count = control->get_server_count();
        for (size_t ep_idx = 0; ep_idx < endpoint_count && total_cnt < wc_length; ep_idx++) {
            auto *cq = get_endpoint_data_cq_local(ep_idx, input_qp_idx);
            if (cq == nullptr || cq->complete_queue == nullptr) {
                continue;
            }
            size_t cnt = ibv_poll_cq(cq->complete_queue, wc_length - total_cnt, wc + total_cnt);
            total_cnt += cnt;
        }
        return total_cnt;
    }

    void handle_rdma_complete_simple(const ibv_wc &wc) {
        assert(wc.status == IBV_WC_SUCCESS);
        void *local_ptr = reinterpret_cast<void *>(wc.wr_id);
        // set the 8 bytes start from local_ptr to 1
        *reinterpret_cast<uint64_t *>(local_ptr) = 1;
    }

    void simple_handler(const ibv_wc &wc) {
        assert(wc.status == IBV_WC_SUCCESS);
        handle_rdma_complete_simple(wc);
    }

    size_t check_cq_idx_with_client_idx_simple_handler(size_t qp_idx, size_t thread_id) {
        // if (client->test_thread_lock() || client->set_thread_lock()) {
        //     return 0;
        // }
        ibv_wc wc[rdma::CHECK_CQ_BATCH_SIZE];
        size_t cnt = check_cq_with_idx(wc, rdma::CHECK_CQ_BATCH_SIZE, qp_idx);
        for (size_t i = 0; i < cnt; i++) {
            simple_handler(wc[i]);
        }
        return cnt;
    }

    __attribute__((noinline)) static Client *get_default() { return get_client(thread_info.thread_id); }

    __attribute__((noinline)) static Client *get_specific_client(size_t thread_id) { return get_client(thread_id); }


    void check_qp_state(size_t qp_idx) {
        if (qp_idx >= qp_count) {
            std::cout << "QP index " << qp_idx << " out of range (max: " << qp_count << ")" << std::endl;
            return;
        }
        
        ibv_qp_attr attr;
        ibv_qp_init_attr init_attr;
        int ret = ibv_query_qp(qps[qp_idx].queue_pair, &attr, IBV_QP_STATE, &init_attr);
        if (ret == 0) {
            std::cout << "QP " << qp_idx << " state: " << attr.qp_state << std::endl;
        } else {
            std::cout << "Failed to query QP " << qp_idx << " state, ret=" << ret << std::endl;
        }
    }
};

void reset_read_batch_diagnostics();
ReadBatchSnapshot collect_read_batch_diagnostics();
void print_read_batch_diagnostics();
}  // namespace rdma

}  // namespace FarLib
