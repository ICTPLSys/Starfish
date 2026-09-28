#include "rdma/server.hpp"

#include <iostream>
#include <cstdlib>
#include <csignal>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

using namespace FarLib::rdma;

namespace {

void handle_summary_stop_signal(int) {
    FarLib::rdma::server_summary_stop_requested().store(
        true, std::memory_order_release);
}

// Infer the logical server index early, before the Server object exists, so
// bootstrap CPU pinning can use the same endpoint ordering as runtime setup.
size_t parse_server_index_for_bootstrap_pin(const Configure &config) {
    if (!config.all_server_port_list.empty()) {
        return config.local_server_index;
    }
    char *end = nullptr;
    long port = std::strtol(config.server_port.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') {
        return 0;
    }
    if (port >= 1400 && port < 2400) {
        return static_cast<size_t>(port - 1400);
    }
    return static_cast<size_t>(port >= 0 ? port : 0);
}

// Pin the server main thread before RDMA/bootstrap buffer first-touch in the
// 6-server EC layouts.
// A config-provided core mapping wins; the hard-coded table is only the legacy
// local fallback for old launch scripts.
void maybe_pin_server_main_for_bootstrap(const Configure &config) {
    const char *disable_pin = std::getenv("FARLIB_DISABLE_SERVER_PIN");
    if (disable_pin != nullptr && disable_pin[0] == '1') {
        return;
    }
    size_t endpoint_count = config.all_server_port_list.empty()
                                ? static_cast<size_t>(config.server_count)
                                : config.all_server_port_list.size();
    if (endpoint_count != 6 ||
        (!config.is_ec_2pc_mode() && !config.is_ec_span_mode())) {
        return;
    }

    static constexpr size_t kPinnedWorkerCores[6] = {
        0, 4, 8, 12, 16, 18,
    };
    size_t server_idx = parse_server_index_for_bootstrap_pin(config);
    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    size_t core = 0;
    if (config.get_configured_server_pin_core(server_idx, 0, &core)) {
        if (cpu_count <= 0 || core >= static_cast<size_t>(cpu_count)) {
            return;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(static_cast<int>(core), &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
            return;
        }
        EC2PC_DIAG(std::cout << "INFO: server bootstrap-pin server="
                             << server_idx << " cpu=" << core
                             << " actual_cpu=" << sched_getcpu()
                             << " source=config" << std::endl;);
        return;
    }
    if (server_idx >= std::size(kPinnedWorkerCores)) {
        return;
    }
    core = kPinnedWorkerCores[server_idx];
    if (cpu_count <= 0 || core >= static_cast<size_t>(cpu_count)) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(core), &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        return;
    }
    EC2PC_DIAG(std::cout << "INFO: server bootstrap-pin server=" << server_idx
                         << " cpu=" << core
                         << " actual_cpu=" << sched_getcpu() << std::endl;);
}

}  // namespace

int main(int argc, const char *const argv[]) {
    if (argc != 2) {
        std::cerr << argv[0] << " <configure file>" << std::endl;
        std::abort();
    }
    const char *configure_file_name = argv[1];
    Configure config;
    config.from_file(configure_file_name);
    
    // Allow overriding server_port via environment variable or command line
    // Priority: SERVER_PORT env > argv[2] > config file
    const char* env_port = std::getenv("SERVER_PORT");
    if (env_port != nullptr) {
        config.server_port = env_port;
        config.post_process_config();
    } else if (argc >= 3) {
        config.server_port = argv[2];
        config.post_process_config();
    }
    
    std::cout << "Starting server on port " << config.server_port << std::endl;

    std::signal(SIGTERM, handle_summary_stop_signal);
    maybe_pin_server_main_for_bootstrap(config);
    Server server(config);
    while (true) {
        server.start();
    }
    std::cout << "Bye!" << std::endl;
    return 0;
}
