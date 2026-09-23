#include "rdma/client.hpp"

#include <infiniband/verbs.h>
#include <cstdio>
#include <thread>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <sys/select.h>
#include <sys/syscall.h>
#include <unordered_map>
#include <unistd.h>

#include "rdma/exchange_msg.hpp"
#include "rdma/rdma.hpp"
#include "utils/debug.hpp"
#include "utils/defer.hpp"
#include "utils/uthreads.hpp"

namespace FarLib {
namespace rdma {

// Helper function to connect with retry and wait for connection
int connect_with_retry(const char* server_addr, const char* server_port, int max_retries = 60, int retry_interval_ms = 1000) {
    int conn_fd = -1;
    int retries = 0;
    
    while (conn_fd < 0 && retries < max_retries) {
        conn_fd = tcp::connect_to_server(server_addr, server_port);
        if (conn_fd < 0) {
            retries++;
            std::cout << "Waiting for server " << server_addr << ":" << server_port 
                      << " (attempt " << retries << "/" << max_retries << ")..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_ms));
        } else {
            // Wait for connection to complete using select
            fd_set write_fds;
            FD_ZERO(&write_fds);
            FD_SET(conn_fd, &write_fds);
            struct timeval timeout;
            timeout.tv_sec = 5;
            timeout.tv_usec = 0;
            
            int sel_ret = select(conn_fd + 1, nullptr, &write_fds, nullptr, &timeout);
            if (sel_ret <= 0) {
                // Connection failed or timeout
                close(conn_fd);
                conn_fd = -1;
                retries++;
                std::cout << "Connection to " << server_addr << ":" << server_port 
                          << " failed, retrying..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_ms));
            } else {
                // Check if connection succeeded or failed
                int so_error;
                socklen_t len = sizeof(so_error);
                getsockopt(conn_fd, SOL_SOCKET, SO_ERROR, &so_error, &len);
                if (so_error != 0) {
                    close(conn_fd);
                    conn_fd = -1;
                    retries++;
                    std::cout << "Connection to " << server_addr << ":" << server_port 
                              << " failed (error " << so_error << "), retrying..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_ms));
                }
                // Connection succeeded!
            }
        }
    }
    
    if (conn_fd < 0) {
        std::cerr << "Failed to connect to server after " << max_retries << " attempts" << std::endl;
        ERROR("Cannot connect to server");
    }
    
    return conn_fd;
}

ClientControl::ClientControl(const Configure &config)
    : config(config),
      buffer(allocate_buffer(config.client_buffer_size)),
      ctx(config),
      pd(ctx),
      mr(pd, buffer, config.client_buffer_size),
      control_cq(ctx, config),
      server_count(config.server_count),
      remote_base_addr(0),
      remote_key(0) {
    INFO("initializing rdma client");

    size_t data_qp_count = 0;
    if (config.enable_eager_evict) {
        data_qp_count = (config.max_thread_cnt + config.evacuate_thread_cnt + 1) * config.qp_count;
    }
    else {
        data_qp_count = (config.max_thread_cnt + 1) * config.qp_count;
    }
    size_t used_qp_count = data_qp_count + 1;

    // Initialize endpoint QPs vector
    endpoint_qps.resize(server_count);
    
    // prepare port attr & psn
    ibv_port_attr port_attr;
    ibv_query_port(ctx.context, config.ib_port, &port_attr);
    srand48(time(nullptr));
    uint32_t psn = lrand48() & 0xffffff;

    // allocate tcp message buffer
    size_t client_info_size =
        sizeof(ClientConnectionInfo) + sizeof(uint32_t) * used_qp_count;
    size_t server_info_size =
        sizeof(ServerConnectionInfo) + sizeof(uint32_t) * used_qp_count;
    auto client_info =
        static_cast<ClientConnectionInfo *>(std::malloc(client_info_size));
    DEFER({ std::free(client_info); });
    auto server_info =
        static_cast<ServerConnectionInfo *>(std::malloc(server_info_size));
    DEFER({ std::free(server_info); });

    // Initialize endpoints vector
    endpoints.resize(server_count);
    
    // Connect to all endpoints
    for (size_t ep_idx = 0; ep_idx < server_count; ep_idx++) {
        // Create QPs for this endpoint
        endpoint_qps[ep_idx].control_qp =
            std::make_unique<QueuePair>(ctx, control_cq, pd, config);
        auto *raw_cqs = static_cast<CompleteQueue *>(
            std::aligned_alloc(alignof(CompleteQueue),
                               sizeof(CompleteQueue) * data_qp_count));
        endpoint_qps[ep_idx].data_cqs = CompleteQueueArrayPtr(
            raw_cqs, CompleteQueueArrayDeleter{data_qp_count});
        for (size_t i = 0; i < data_qp_count; i++) {
            new (&endpoint_qps[ep_idx].data_cqs[i]) CompleteQueue(ctx, config);
        }
        endpoint_qps[ep_idx].data_qps = std::make_unique<QueuePair[]>(data_qp_count);
        for (size_t i = 0; i < data_qp_count; i++) {
            endpoint_qps[ep_idx].data_qps[i].init(ctx,
                                                 endpoint_qps[ep_idx].data_cqs[i],
                                                 pd, config);
        }
        endpoint_qps[ep_idx].data_qp_count = data_qp_count;
        
        auto [server_addr, server_port] = config.get_endpoint(ep_idx);
        
        INFO(("rdma.endpoint.connect.start endpoint=" + std::to_string(ep_idx) + 
             " addr=" + server_addr + " port=" + server_port).c_str());
        
        // Prepare client info with this endpoint's QP numbers
        client_info->psn = psn;
        client_info->lid = port_attr.lid;
        client_info->max_rd_atomic = config.qp_max_rd_atomic;
        client_info->mtu = config.qp_mtu;
        client_info->server_buffer_size = config.server_buffer_size;
        client_info->qp_count = used_qp_count;
        client_info->qpn[0] = endpoint_qps[ep_idx].control_qp->queue_pair->qp_num;
        for (size_t i = 0; i < data_qp_count; i++) {
            client_info->qpn[i + 1] = endpoint_qps[ep_idx].data_qps[i].queue_pair->qp_num;
        }
        
        // set up tcp connection with retry logic
        int conn_fd = connect_with_retry(server_addr.c_str(), server_port.c_str());
        ssize_t sent_size =
            tcp::send_all(conn_fd, client_info, client_info_size, 0);
        if (sent_size != (ssize_t)client_info_size) {
            std::cerr << "Error: sent_size (" << sent_size << ") != client_info_size (" << client_info_size << ")" << std::endl;
            std::abort();
        }
        ssize_t recv_size =
            tcp::recieve_all(conn_fd, server_info, server_info_size, 0);
        if (recv_size != (ssize_t)server_info_size) {
            std::cerr << "Error: recv_size (" << recv_size << ") != server_info_size (" << server_info_size << ")" << std::endl;
            std::abort();
        }
        close(conn_fd);
        ASSERT(server_info->qp_count == used_qp_count);

        // Store endpoint info
        endpoints[ep_idx].remote_base_addr = server_info->addr;
        endpoints[ep_idx].remote_key = server_info->rkey;
        endpoints[ep_idx].endpoint_idx = ep_idx;
        
        INFO(("rdma.endpoint.connect.ok endpoint=" + std::to_string(ep_idx) + 
             " base_addr=" + std::to_string(server_info->addr) +
             " rkey=" + std::to_string(server_info->rkey)).c_str());

        // set up queue pairs for this endpoint
        endpoint_qps[ep_idx].control_qp->ready_to_recv(server_info->lid, server_info->psn,
                                 server_info->qpn[0], config);
        endpoint_qps[ep_idx].control_qp->ready_to_send(psn, config);
        for (size_t i = 0; i < data_qp_count; i++) {
            endpoint_qps[ep_idx].data_qps[i].ready_to_recv(server_info->lid, server_info->psn,
                                      server_info->qpn[i + 1], config);
            endpoint_qps[ep_idx].data_qps[i].ready_to_send(psn, config);
        }
    }

    // For backward compatibility, set legacy remote_base_addr/remote_key from first endpoint
    if (server_count > 0) {
        remote_base_addr = endpoints[0].remote_base_addr;
        remote_key = endpoints[0].remote_key;
    }

    // set up connection pool
    if (server_count > 0) {
        conn_pool.init(endpoint_qps[0].data_qps.get(),
                       endpoint_qps[0].data_cqs.get(),
                       data_qp_count, config.qp_count);
    }
}

void ClientControl::post_stop() {
    const ReadBatchSnapshot read_batch = collect_read_batch_diagnostics();
    if (read_batch.pending_reads != 0) {
        std::cerr << "RDMA_READ_BATCH_SHUTDOWN_PENDING count="
                  << read_batch.pending_reads << std::endl;
        std::abort();
    }
    // Send stop message to all endpoints
    for (size_t ep_idx = 0; ep_idx < server_count; ep_idx++) {
        if (endpoint_qps[ep_idx].control_qp == nullptr) continue;
        
        ibv_send_wr stop_wr = {
            .wr_id = RQ_STOP,
            .next = nullptr,
            .sg_list = nullptr,
            .num_sge = 0,
            .opcode = IBV_WR_SEND,
            .send_flags = IBV_SEND_SIGNALED | IBV_SEND_SOLICITED,
        };
        ibv_send_wr *bad_wr;
    retry:
        int send_ret = ibv_post_send(endpoint_qps[ep_idx].control_qp->queue_pair, &stop_wr, &bad_wr);
        bool posted = check_ibv_post_send_ret(send_ret);
        ibv_wc wc;
        if (!posted) {
            ibv_poll_cq(control_cq.complete_queue, 1, &wc);
            goto retry;
        }
        while (true) {
            std::size_t n = ibv_poll_cq(control_cq.complete_queue, 1, &wc);
            ASSERT(n <= 1);
            if (n == 1) {
                if (wc.wr_id == stop_wr.wr_id && wc.opcode == IBV_WC_SEND) {
                    assert(wc.status == IBV_WC_SUCCESS);
                    break;
                }
            }
        }
    }
}

void ConnectionPool::add_client(Client *client) {
    if (qp_count == 0 || qp_per_client == 0) {
        // connection pool not initialized yet
        client->init(nullptr, nullptr, 0, 0);
        return;
    }
    size_t qp_idx = alloc_idx.load();
retry:
    if (qp_idx + qp_per_client > qp_count + 1) {
        std::cout << "qp_idx: " << qp_idx << ", qp_per_client: " << qp_per_client << ", qp_count: " << qp_count << std::endl;
        ERROR("Can not allocate connection");
    }
    if (!alloc_idx.compare_exchange_weak(qp_idx, qp_idx + qp_per_client))
        goto retry;

    client->init(qps + qp_idx, cqs + qp_idx, qp_per_client, qp_idx);
}

ConnectionPool conn_pool;
std::unique_ptr<ClientControl> ClientControl::default_instance;
// thread_local Client client;
std::vector<Client> clients;
std::atomic_size_t thread_id_global;
thread_local ThreadInfo thread_info;

namespace {
std::mutex thread_id_registry_mutex;
std::unordered_map<pthread_t, size_t> thread_id_registry;
std::atomic_bool use_thread_id_registry{false};
std::atomic_bool thread_id_registry_sealed{false};
size_t registered_app_worker_count = 0;
}

bool thread_id_registry_enabled() {
    return use_thread_id_registry.load(std::memory_order_acquire);
}

size_t get_registered_thread_id() {
    const pthread_t pthread_id = pthread_self();
    std::lock_guard<std::mutex> lock(thread_id_registry_mutex);
    auto it = thread_id_registry.find(pthread_id);
    if (it == thread_id_registry.end()) {
        std::cerr << "ERROR: unregistered RDMA worker pthread="
                  << static_cast<unsigned long>(pthread_id)
                  << " sealed=" << thread_id_registry_sealed.load()
                  << std::endl;
        ERROR("unregistered RDMA worker");
    }
    const size_t client_id = it->second;
    if (client_id >= clients.size()) {
        ERROR("registered RDMA client id out of range");
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    (void)pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed);
    int allowed_cpu = -1;
    size_t allowed_count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            allowed_cpu = cpu;
            ++allowed_count;
        }
    }
    char record[256];
    const bool app = client_id < registered_app_worker_count;
    const size_t role_index = app ? client_id
                                  : client_id - registered_app_worker_count;
    const int n = std::snprintf(
        record, sizeof(record),
        "worker_role role=%s index=%zu client_id=%zu tid=%ld cpu=%d "
        "allowed=%d allowed_count=%zu\n",
        app ? "app" : "background", role_index, client_id,
        static_cast<long>(::syscall(SYS_gettid)), ::sched_getcpu(),
        allowed_cpu, allowed_count);
    if (n > 0) {
        const size_t bytes = static_cast<size_t>(n) < sizeof(record)
                                 ? static_cast<size_t>(n)
                                 : sizeof(record) - 1;
        (void)::write(STDERR_FILENO, record, bytes);
    }
    return client_id;
}

void reset_thread_id_registry(size_t app_worker_count) {
    std::lock_guard<std::mutex> lock(thread_id_registry_mutex);
    thread_id_registry.clear();
    registered_app_worker_count = app_worker_count;
    thread_id_global.store(0, std::memory_order_relaxed);
    thread_id_registry_sealed.store(false, std::memory_order_release);
    use_thread_id_registry.store(true, std::memory_order_release);
}

void disable_thread_id_registry() {
    std::lock_guard<std::mutex> lock(thread_id_registry_mutex);
    use_thread_id_registry.store(false, std::memory_order_release);
    thread_id_registry_sealed.store(false, std::memory_order_release);
    thread_id_registry.clear();
    registered_app_worker_count = 0;
    thread_id_global.store(0, std::memory_order_relaxed);
}

void register_thread_id(pthread_t tid, size_t thread_id) {
    std::lock_guard<std::mutex> lock(thread_id_registry_mutex);
    if (!use_thread_id_registry.load(std::memory_order_relaxed) ||
        thread_id_registry_sealed.load(std::memory_order_relaxed) ||
        thread_id >= clients.size()) {
        ERROR("invalid RDMA thread id registration");
    }
    auto [it, inserted] = thread_id_registry.emplace(tid, thread_id);
    if (!inserted && it->second != thread_id) {
        ERROR("conflicting RDMA thread id registration");
    }
    size_t expected = thread_id_global.load(std::memory_order_relaxed);
    while (expected <= thread_id &&
           !thread_id_global.compare_exchange_weak(
               expected, thread_id + 1, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

void seal_thread_id_registry() {
    std::lock_guard<std::mutex> lock(thread_id_registry_mutex);
    if (thread_id_registry.size() != clients.size()) {
        ERROR("RDMA thread registry/client count mismatch");
    }
    thread_id_registry_sealed.store(true, std::memory_order_release);
}

Client *get_client(size_t thread_id) {
    return &clients[thread_id];
}

size_t get_client_count() {
    return clients.size();
}

namespace {
void progress_read_batch_when_yield_finds_no_ready_fibre() {
    if (configured_read_batch_size() != 2) return;
    const size_t client_idx = thread_info.thread_id;
    if (client_idx >= clients.size()) {
        std::cerr << "read batch yield hook client index out of range"
                  << std::endl;
        std::abort();
    }
    clients[client_idx].progress_all_sync_reads(true);
}
}  // namespace

void install_read_batch_yield_progress() {
    if (configured_read_batch_size() == 2) {
        uthread::install_yield_no_ready_progress(
            progress_read_batch_when_yield_finds_no_ready_fibre);
    }
}

void remove_read_batch_yield_progress() {
    if (configured_read_batch_size() == 2) {
        uthread::remove_yield_no_ready_progress();
    }
}

void reset_read_batch_diagnostics() {
    for (auto &client : clients) {
        client.reset_read_batch_diagnostics();
    }
}

ReadBatchSnapshot collect_read_batch_diagnostics() {
    ReadBatchSnapshot result;
    for (const auto &client : clients) {
        result += client.snapshot_read_batch_diagnostics();
    }
    return result;
}

void print_read_batch_diagnostics() {
    const ReadBatchSnapshot diag = collect_read_batch_diagnostics();
    const double mean_cycles = diag.accepted_reads == 0
        ? 0.0
        : static_cast<double>(diag.enqueue_to_accept_cycles_sum) /
              static_cast<double>(diag.accepted_reads);
    std::cout << "rdma_read_batch_diag"
              << " configured_batch_size=" << configured_read_batch_size()
              << " attribution_mode=" << read_batch_attribution_mode()
              << " generated_reads=" << diag.generated_reads
              << " generated_bytes=" << diag.generated_bytes
              << " physical_submit_calls=" << diag.physical_submit_calls
              << " immediate_submit_calls=" << diag.immediate_submit_calls
              << " immediate_first_submit_attempts="
              << diag.immediate_first_submit_attempts
              << " attempted_chain_1=" << diag.attempted_chain_1
              << " attempted_chain_2=" << diag.attempted_chain_2
              << " accepted_prefix_0=" << diag.accepted_prefix_0
              << " accepted_prefix_1=" << diag.accepted_prefix_1
              << " accepted_prefix_2=" << diag.accepted_prefix_2
              << " accepted_reads=" << diag.accepted_reads
              << " accepted_bytes=" << diag.accepted_bytes
              << " partial_submit_calls=" << diag.partial_submit_calls
              << " runtime_cooperative_yields="
              << diag.runtime_cooperative_yields
              << " full_submit_attempts=" << diag.full_submit_attempts
              << " age_flush_attempts=" << diag.age_flush_attempts
              << " idle_no_switch_flush_attempts="
              << diag.idle_no_switch_flush_attempts
              << " credit_retry_attempts=" << diag.credit_retry_attempts
              << " blocked_submit_suppressed="
              << diag.blocked_submit_suppressed
              << " age_progress_checks=" << diag.age_progress_checks
              << " no_ready_hook_calls=" << diag.no_ready_hook_calls
              << " backpressure_yields=" << diag.backpressure_yields
              << " cq_progress_events=" << diag.cq_progress_events
              << " cq_progress_completions="
              << diag.cq_progress_completions
              << " cq_progress_sample=independent_atomics"
              << " max_delay_us=" << read_batch_max_delay_us()
              << " age_clock=clock_monotonic_raw"
              << " pending_reads=" << diag.pending_reads
              << " enqueue_to_accept_mean_cycles=" << mean_cycles
              << " enqueue_to_accept_max_cycles="
              << diag.enqueue_to_accept_cycles_max
              << std::endl;
    if (diag.pending_reads != 0 ||
        diag.generated_reads != diag.accepted_reads + diag.pending_reads ||
        diag.physical_submit_calls !=
            diag.attempted_chain_1 + diag.attempted_chain_2 ||
        diag.accepted_reads !=
            diag.accepted_prefix_1 + 2 * diag.accepted_prefix_2 ||
        (configured_read_batch_size() == 2 &&
         (diag.runtime_cooperative_yields != 0 ||
          diag.physical_submit_calls !=
              diag.immediate_submit_calls +
                  diag.immediate_first_submit_attempts +
                  diag.full_submit_attempts +
                  diag.age_flush_attempts +
                  diag.idle_no_switch_flush_attempts +
                  diag.credit_retry_attempts))) {
        std::cerr << "RDMA_READ_BATCH_DIAG_FAIL generated/accepted/pending"
                  << std::endl;
        std::abort();
    }
}

Client *get_specific_client(size_t thread_id) {
    return get_client(thread_id);
}

}  // namespace rdma
}  // namespace FarLib
