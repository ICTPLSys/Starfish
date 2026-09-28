/**
 * RDMA Configuration
 */
#pragma once

#include <infiniband/verbs.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "utils/debug.hpp"

namespace FarLib {

// Basic parameters, not configurable
constexpr size_t PAGE_SIZE = 4096;

namespace rdma {

struct Configure {
#define CONFIG(TYPE, VAR, DEFAULT) TYPE VAR = DEFAULT;
#include "config.def"
#undef CONFIG
    static constexpr uint8_t MODE_ENABLE_CACHE = 0x01;
    static constexpr uint8_t MODE_ENABLE_UTHREAD = 0x02;
    
    // Derived values
    size_t remote_total = 0;  // server_count * server_buffer_size
    
    // Parsed endpoint addresses (populated after from_file)
    std::vector<std::string> server_addr_list;
    std::vector<std::string> server_port_list;
    std::vector<std::string> all_server_addr_list;
    std::vector<std::string> all_server_port_list;
    std::vector<std::vector<size_t>> server_pin_core_table;
    size_t local_server_index = 0;
    
    // Mapping type enum
    enum MappingType { MAPPING_RANGE, MAPPING_STRIPE };
    MappingType mapping_type = MAPPING_RANGE;
    enum FTMethod { FT_NONE, FT_REPLICA, FT_EC_SPLIT, FT_EC_SPAN, FT_EC_2PC };
    FTMethod ft_method_type = FT_NONE;

    size_t configured_compaction_worker_count() const {
        return std::max<size_t>(1, compaction_worker_count);
    }

    bool uses_ec_span_compaction_workers() const {
        return ft_method_type == FT_EC_SPAN && exclusive_cache &&
               ft_method_type != FT_EC_2PC;
    }

    size_t background_worker_count() const {
        const size_t compaction_slots =
            uses_ec_span_compaction_workers()
                ? configured_compaction_worker_count() + 1
                : 0;
        return evacuate_thread_cnt + compaction_slots;
    }

    size_t rdma_data_client_count() const {
        return max_thread_cnt + background_worker_count();
    }
    
    void from_file(const char *filename) {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
            std::cerr << "Error: can not open configuration file: " << filename
                      << std::endl;
            std::abort();
        }
        std::string name;
        while (ifs >> name) {
            if (name.size() > 0 && name[0] == '#') {
                while (true) {
                    int ch = ifs.get();
                    if (ch == '\n' || ch == EOF) break;
                }
                continue;
            }
#define CONFIG(TYPE, VAR, DEFAULT)                                      \
    if (name == #VAR) {                                                 \
        if (!(ifs >> this->VAR)) {                                      \
            std::cerr << "Error when reading configuration of " << name \
                      << ". Expected type is " << #TYPE << std::endl;   \
            std::abort();                                               \
        }                                                               \
        continue;                                                       \
    }
#include "config.def"
#undef CONFIG
            std::cerr << "Unknown configuration name: " << name << std::endl;
            std::abort();
        }
        
        // Post-processing for multi-endpoint configuration
        post_process_config();
    }
    
    void post_process_config() {
        if (!rdma_device_name.empty()) {
            if (setenv("FARLIB_RDMA_DEVICE", rdma_device_name.c_str(), 1) != 0) {
                std::perror("setenv FARLIB_RDMA_DEVICE");
                std::abort();
            }
        }

        // Calculate remote_total first - now server_buffer_size means TOTAL capacity
        // Only do this once to avoid repeated division if called multiple times
        if (remote_total == 0) {
            remote_total = server_buffer_size;
            if (server_count > 0) {
                server_buffer_size = remote_total / server_count;
                
                // CRITICAL: server_buffer_size must be a multiple of RegionSize (512KB)
                // Align to the next multiple of 512KB (ceiling alignment).
                const size_t RegionSize = 512 * 1024;
                server_buffer_size = (server_buffer_size + RegionSize - 1) / RegionSize * RegionSize;
                remote_total = server_buffer_size * server_count;
            }
        }
        
        // Parse server addresses and ports
        if (server_count > 1 || (!server_addrs.empty() && !server_ports.empty())) {
            // Multi-endpoint mode: require server_addrs and server_ports
            if (server_addrs.empty() || server_ports.empty()) {
                std::cerr << "Error: server_count > 1 requires server_addrs and server_ports" << std::endl;
                std::abort();
            }
            server_addr_list = parse_string_list(server_addrs);
            server_port_list = parse_string_list(server_ports);
            if (server_addr_list.size() != server_port_list.size()) {
                std::cerr << "Error: server_addrs/server_ports count must match" << std::endl;
                std::abort();
            }
            if (server_count > 1 &&
                server_addr_list.size() != static_cast<size_t>(server_count)) {
                std::cerr << "Error: server_addrs/server_ports count must match server_count" << std::endl;
                std::abort();
            }
            all_server_addr_list = server_addr_list;
            all_server_port_list = server_port_list;
            local_server_index = 0;
            
            // Allow SERVER_PORT env to override: find matching port in the list
            const char* env_port = std::getenv("SERVER_PORT");
            if (env_port != nullptr) {
                bool found = false;
                for (size_t i = 0; i < server_port_list.size(); i++) {
                    if (server_port_list[i] == env_port) {
                        local_server_index = i;
                        // Save the matching address before clearing
                        std::string matching_addr = server_addr_list[i];
                        // Use only this port for this server instance
                        server_port_list.clear();
                        server_port_list.push_back(env_port);
                        server_addr_list.clear();
                        server_addr_list.push_back(matching_addr);
                        // Also update the singular server_addr for backward compatibility
                        server_addr = matching_addr;
                        server_port = env_port;
                        server_count = 1;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "Error: SERVER_PORT=" << env_port 
                              << " not found in server_ports=" << server_ports << std::endl;
                    std::abort();
                }
            }
        } else {
            // Backward compatible: use single server_addr/server_port
            server_addr_list = {server_addr};
            server_port_list = {server_port};
            all_server_addr_list = server_addr_list;
            all_server_port_list = server_port_list;
            local_server_index = 0;
        }

        parse_server_pin_core_table();
        
        // Parse mapping type
        if (remote_mapping == "stripe") {
            mapping_type = MAPPING_STRIPE;
        } else {
            mapping_type = MAPPING_RANGE;  // default
        }

        if (ft_method == "none") {
            ft_method_type = FT_NONE;
        } else if (ft_method == "replica") {
            ft_method_type = FT_REPLICA;
        } else if (ft_method == "ec_split") {
            ft_method_type = FT_EC_SPLIT;
        } else if (ft_method == "ec_span") {
            ft_method_type = FT_EC_SPAN;
        } else if (ft_method == "ec_2pc") {
            ft_method_type = FT_EC_2PC;
        } else {
            std::cerr << "Error: unsupported ft_method: " << ft_method
                      << " (supported: none, replica, ec_split, ec_span, ec_2pc)"
                      << std::endl;
            std::abort();
        }
    }
    
    std::vector<std::string> parse_string_list(const std::string& s) {
        std::vector<std::string> result;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // Trim whitespace
            size_t start = item.find_first_not_of(" \t");
            size_t end = item.find_last_not_of(" \t");
            if (start != std::string::npos) {
                result.push_back(item.substr(start, end - start + 1));
            }
        }
        return result;
    }

    void parse_server_pin_core_table() {
        server_pin_core_table.clear();
        if (server_pin_cores.empty()) {
            return;
        }
        std::vector<std::string> groups = parse_string_list(server_pin_cores);
        size_t endpoint_count =
            all_server_port_list.empty()
                ? static_cast<size_t>(std::max(server_count, 0))
                : all_server_port_list.size();
        if (groups.size() != endpoint_count) {
            std::cerr << "Error: server_pin_cores group count "
                      << groups.size() << " does not match endpoint count "
                      << endpoint_count << std::endl;
            std::abort();
        }
        server_pin_core_table.reserve(groups.size());
        for (const auto &group : groups) {
            std::vector<size_t> cores;
            std::stringstream ss(group);
            std::string item;
            while (std::getline(ss, item, ':')) {
                size_t start = item.find_first_not_of(" \t");
                size_t end = item.find_last_not_of(" \t");
                if (start == std::string::npos) {
                    continue;
                }
                std::string trimmed = item.substr(start, end - start + 1);
                char *parse_end = nullptr;
                unsigned long parsed =
                    std::strtoul(trimmed.c_str(), &parse_end, 10);
                if (parse_end == trimmed.c_str() || parse_end == nullptr ||
                    *parse_end != '\0') {
                    std::cerr << "Error: invalid server_pin_cores CPU id: "
                              << trimmed << std::endl;
                    std::abort();
                }
                cores.push_back(static_cast<size_t>(parsed));
            }
            if (cores.empty()) {
                std::cerr << "Error: empty server_pin_cores group" << std::endl;
                std::abort();
            }
            server_pin_core_table.push_back(std::move(cores));
        }
    }

    bool get_configured_server_pin_core(size_t server_idx, size_t thread_slot,
                                        size_t *core_out) const {
        if (core_out == nullptr || server_pin_core_table.empty()) {
            return false;
        }
        if (server_idx >= server_pin_core_table.size()) {
            return false;
        }
        const auto &cores = server_pin_core_table[server_idx];
        if (thread_slot >= cores.size()) {
            return false;
        }
        *core_out = cores[thread_slot];
        return true;
    }
    
    // Get endpoint address and port by index
    std::pair<std::string, std::string> get_endpoint(size_t idx) const {
        if (idx >= server_addr_list.size()) {
            std::cerr << "Error: endpoint index out of range" << std::endl;
            std::abort();
        }
        return {server_addr_list[idx], server_port_list[idx]};
    }

    std::pair<std::string, std::string> get_full_endpoint(size_t idx) const {
        if (idx >= all_server_addr_list.size()) {
            std::cerr << "Error: full endpoint index out of range" << std::endl;
            std::abort();
        }
        return {all_server_addr_list[idx], all_server_port_list[idx]};
    }
    
    // Mapping functions: remote_addr -> (endpoint_idx, offset)
    // Range mapping: endpoint = addr / server_buffer_size, offset = addr % server_buffer_size
    std::pair<size_t, size_t> map_range(uint64_t remote_addr) const {
        size_t endpoint = remote_addr / server_buffer_size;
        size_t offset = remote_addr % server_buffer_size;
        return {endpoint, offset};
    }
    
    // Stripe mapping: stripe = addr / stripe_size, endpoint = stripe % server_count
    // offset = (stripe / server_count) * stripe_size + (addr % stripe_size)
    std::pair<size_t, size_t> map_stripe(uint64_t remote_addr) const {
        if (stripe_size_bytes == 0) {
            std::cerr << "Error: stripe_size_bytes must be set for stripe mapping" << std::endl;
            std::abort();
        }
        uint64_t stripe = remote_addr / stripe_size_bytes;
        size_t endpoint = stripe % server_count;
        uint64_t stripe_idx = stripe / server_count;
        size_t offset = stripe_idx * stripe_size_bytes + (remote_addr % stripe_size_bytes);
        return {endpoint, offset};
    }
    
    // Main mapping function based on configured mapping type
    std::pair<size_t, size_t> map_remote_addr(uint64_t remote_addr) const {
        if (mapping_type == MAPPING_STRIPE) {
            return map_stripe(remote_addr);
        }
        return map_range(remote_addr);
    }
    
    // Validate mapping invariants
    bool validate_mapping(uint64_t remote_addr, size_t size) const {
        auto [endpoint, offset] = map_remote_addr(remote_addr);
        // I1: endpoint < server_count
        if (endpoint >= static_cast<size_t>(server_count)) {
            return false;
        }
        // I2: offset + size <= server_buffer_size
        if (offset + size > server_buffer_size) {
            return false;
        }
        return true;
    }
    
    void self_check() const {
        std::cerr << max_thread_cnt * qp_count * qp_send_cap << std::endl;
        // ASSERT(cq_entries >= max_thread_cnt * qp_count * qp_send_cap);
        // ASSERT(server_buffer_size >= client_buffer_size);
        std::cout << "cq_entries " << cq_entries << std::endl;
        ASSERT(PAGE_SIZE <= this->server_buffer_size);
        ASSERT(static_cast<int>(this->check_cq_batch_size) <= this->cq_entries);
        
        // Validate server_count
        if (server_count < 1) {
            std::cerr << "Error: server_count must be >= 1" << std::endl;
            std::abort();
        }
        
        // Validate remote_total
        if (remote_total == 0) {
            std::cerr << "Error: remote_total not computed" << std::endl;
            std::abort();
        }
        
        // Validate stripe_size for stripe mapping
        if (mapping_type == MAPPING_STRIPE && stripe_size_bytes == 0) {
            std::cerr << "Error: stripe_size_bytes must be set for stripe mapping" << std::endl;
            std::abort();
        }

        if (ft_method_type == FT_REPLICA) {
            if (!exclusive_cache) {
                std::cerr << "Error: ft_method=replica requires exclusive_cache=1"
                          << std::endl;
                std::abort();
            }
            if (server_count < 3) {
                std::cerr << "Error: ft_method=replica requires server_count >= 3"
                          << std::endl;
                std::abort();
            }
            if (mapping_type != MAPPING_RANGE) {
                std::cerr << "Error: ft_method=replica currently requires "
                             "remote_mapping=range"
                          << std::endl;
                std::abort();
            }
        }

        if (ft_method_type == FT_EC_SPAN || ft_method_type == FT_EC_2PC) {
            if (!exclusive_cache) {
                std::cerr << "Error: ft_method=ec_span/ec_2pc requires exclusive_cache=1"
                          << std::endl;
                std::abort();
            }
            if (server_count < 6) {
                std::cerr << "Error: ft_method=ec_span/ec_2pc requires server_count >= 6 "
                             "(4 data + 2 parity shards)"
                          << std::endl;
                std::abort();
            }
            if (mapping_type != MAPPING_RANGE) {
                std::cerr << "Error: ft_method=ec_span/ec_2pc currently requires "
                             "remote_mapping=range"
                          << std::endl;
                std::abort();
            }
        }
        
        // Print mapping info
        std::cout << "server_count: " << server_count << std::endl;
        std::cout << "server_buffer_size: " << server_buffer_size << std::endl;
        std::cout << "remote_total: " << remote_total << std::endl;
        std::cout << "remote_mapping: " << (mapping_type == MAPPING_RANGE ? "range" : "stripe") << std::endl;
        std::cout << "ft_method: "
                  << (ft_method_type == FT_REPLICA ? "replica"
                      : ft_method_type == FT_EC_SPLIT ? "ec_split"
                      : ft_method_type == FT_EC_SPAN ? "ec_span"
                      : ft_method_type == FT_EC_2PC ? "ec_2pc" : "none")
                  << std::endl;
    }
    
    // Fast mapping self-check for verification
    void verify_mapping() const {
        std::cout << "=== Mapping Self-Check ===" << std::endl;
        
        // Test range mapping
        if (mapping_type == MAPPING_RANGE) {
            for (size_t ep = 0; ep < static_cast<size_t>(server_count); ep++) {
                uint64_t base = ep * server_buffer_size;
                uint64_t end = (ep + 1) * server_buffer_size - 1;
                
                auto [mapped_ep, offset] = map_range(base);
                ASSERT(mapped_ep == ep && offset == 0);
                
                auto [mapped_ep2, offset2] = map_range(end);
                ASSERT(mapped_ep2 == ep && offset2 == server_buffer_size - 1);
            }
            std::cout << "Range mapping check: PASSED" << std::endl;
        }
        
        // Test stripe mapping if enabled
        if (mapping_type == MAPPING_STRIPE && stripe_size_bytes > 0) {
            // Test stripe boundaries
            for (uint64_t addr = 0; addr < remote_total; addr += stripe_size_bytes) {
                auto [ep, offset] = map_stripe(addr);
                ASSERT(ep < static_cast<size_t>(server_count));
                ASSERT(offset < server_buffer_size);
            }
            std::cout << "Stripe mapping check: PASSED" << std::endl;
        }
        
        // Verify invariants I1-I3
        // I1: endpoint < server_count for all valid addresses
        for (uint64_t addr = 0; addr < remote_total; addr += 4096) {
            auto [ep, offset] = map_remote_addr(addr);
            ASSERT(ep < static_cast<size_t>(server_count));
            // I2: offset < server_buffer_size (offset is always within per-server buffer)
            ASSERT(offset < server_buffer_size);
        }
        std::cout << "Invariant checks: PASSED" << std::endl;
        std::cout << "=========================" << std::endl;
    }

    bool is_replica_mode() const { return ft_method_type == FT_REPLICA; }
    bool is_ec_split_mode() const { return ft_method_type == FT_EC_SPLIT; }
    bool is_ec_span_mode() const {
        return ft_method_type == FT_EC_SPAN || ft_method_type == FT_EC_2PC;
    }
    bool is_ec_2pc_mode() const { return ft_method_type == FT_EC_2PC; }
};

}  // namespace rdma
}  // namespace FarLib
