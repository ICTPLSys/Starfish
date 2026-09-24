/**
 * RDMA Configuration
 */
#pragma once

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
    
    // Mapping type enum
    enum MappingType { MAPPING_RANGE, MAPPING_STRIPE };
    MappingType mapping_type = MAPPING_RANGE;

    // Derived profiling flags
    bool profiling_enabled = false;
    bool profiling_object_access_enabled = false;
    bool profiling_dereference_enabled = false;
    
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
            if (name == "include") {
                std::string include_filename;
                if (!(ifs >> include_filename)) {
                    std::cerr << "Error: 'include' directive requires a filename" << std::endl;
                    std::abort();
                }
                // Handle relative paths for include
                std::string full_include_path = include_filename;
                if (include_filename[0] != '/') {
                    std::string dir = "";
                    std::string s_filename = filename;
                    size_t last_slash = s_filename.find_last_of('/');
                    if (last_slash != std::string::npos) {
                        dir = s_filename.substr(0, last_slash + 1);
                    }
                    full_include_path = dir + include_filename;
                }
                from_file(full_include_path.c_str());
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
            std::cerr << "Unknown configuration name: " << name << " in file: " << filename << std::endl;
            std::abort();
        }
        
        // Post-processing for multi-endpoint configuration
        post_process_config();
    }
    
    void post_process_config() {
        if (profiling_method == "fine-grained") {
            profiling_method = "fine_grained";
        } else if (profiling_method == "coarse-grained" ||
                   profiling_method == "coars-grained" ||
                   profiling_method == "coars_grained") {
            profiling_method = "coarse_grained";
        }
        if (reference_heat_split_mode == "access-weighted" ||
            reference_heat_split_mode == "weighted" ||
            reference_heat_split_mode == "ema_weighted") {
            reference_heat_split_mode = "access_weighted";
        } else if (reference_heat_split_mode == "object-count" ||
                   reference_heat_split_mode == "count") {
            reference_heat_split_mode = "object_count";
        }
        // Calculate remote_total first - now server_buffer_size means per-server capacity
        // Only do this once to avoid repeated division if called multiple times
        // Align server_buffer_size to RegionSize (512KB)
        const size_t RegionSize = 512 * 1024;
        server_buffer_size = (server_buffer_size + RegionSize - 1) / RegionSize * RegionSize;
        
        if (server_count > 0) {
            remote_total = server_buffer_size * server_count;
        } else {
            remote_total = server_buffer_size;
        }
        
        // Parse server addresses and ports
        if (server_count > 1) {
            // Multi-endpoint mode: require server_addrs and server_ports
            if (server_addrs.empty() || server_ports.empty()) {
                std::cerr << "Error: server_count > 1 requires server_addrs and server_ports" << std::endl;
                std::abort();
            }
            server_addr_list = parse_string_list(server_addrs);
            server_port_list = parse_string_list(server_ports);
            if (server_addr_list.size() != static_cast<size_t>(server_count) ||
                server_port_list.size() != static_cast<size_t>(server_count)) {
                std::cerr << "Error: server_addrs/server_ports count must match server_count" << std::endl;
                std::abort();
            }
            
            // Allow SERVER_PORT env to override: find matching port in the list
            const char* env_port = std::getenv("SERVER_PORT");
            if (env_port != nullptr) {
                bool found = false;
                for (size_t i = 0; i < server_port_list.size(); i++) {
                    if (server_port_list[i] == env_port) {
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
        }
        
        // Parse mapping type
        if (remote_mapping == "stripe") {
            mapping_type = MAPPING_STRIPE;
        } else {
            mapping_type = MAPPING_RANGE;  // default
        }

        profiling_enabled = profiling_method != "disabled";
        profiling_object_access_enabled =
            profiling_method == "coarse_grained";
        profiling_dereference_enabled =
            profiling_method == "fine_grained";
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
    
    // Get endpoint address and port by index
    std::pair<std::string, std::string> get_endpoint(size_t idx) const {
        if (idx >= server_addr_list.size()) {
            std::cerr << "Error: endpoint index out of range" << std::endl;
            std::abort();
        }
        return {server_addr_list[idx], server_port_list[idx]};
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
    
    void self_check() {
        post_process_config();
        if (ft_method != "none") {
            std::cerr << "Error: Non-FT only supports ft_method none" << std::endl;
            std::abort();
        }
        // ASSERT(cq_entries >= max_thread_cnt * qp_count * qp_send_cap);
        // ASSERT(server_buffer_size >= client_buffer_size);
        std::cout << "cq_entries " << cq_entries << std::endl;
        ASSERT(PAGE_SIZE <= this->server_buffer_size);
        ASSERT(this->check_cq_batch_size <= this->cq_entries);
        
        // Validate server_count
        if (server_count < 1) {
            std::cerr << "Error: server_count must be >= 1" << std::endl;
            std::abort();
        }
        if (mark_thread_cnt > evacuate_thread_cnt) {
            std::cerr << "Error: mark_thread_cnt must be <= evacuate_thread_cnt"
                      << std::endl;
            std::abort();
        }
        if (optimized_evacuator &&
            (mark_thread_cnt == 0 || mark_thread_cnt >= evacuate_thread_cnt)) {
            std::cerr
                << "Error: optimized_evacuator requires 0 < mark_thread_cnt < evacuate_thread_cnt"
                << std::endl;
            std::abort();
        }
        if (hybrid_profile_local_fast_path_sample_shift > 20) {
            std::cerr
                << "Error: hybrid_profile_local_fast_path_sample_shift must be <= 20"
                << std::endl;
            std::abort();
        }
        if (profiling_method != "disabled" &&
            profiling_method != "fine_grained" &&
            profiling_method != "coarse_grained") {
            std::cerr
                << "Error: profiling_method must be one of disabled, fine_grained, coarse_grained"
                << std::endl;
            std::abort();
        }
        if (reference_heat_split_mode != "object_count" &&
            reference_heat_split_mode != "access_weighted") {
            std::cerr
                << "Error: reference_heat_split_mode must be one of object_count, access_weighted"
                << std::endl;
            std::abort();
        }
        if (hybrid_profile_frequency_ema_shift == 0 ||
            hybrid_profile_frequency_ema_shift > 31) {
            std::cerr
                << "Error: hybrid_profile_frequency_ema_shift must be in [1, 31]"
                << std::endl;
            std::abort();
        }
        if (hybrid_profile_frequency_ema_mark_interval == 0) {
            std::cerr
                << "Error: hybrid_profile_frequency_ema_mark_interval must be >= 1"
                << std::endl;
            std::abort();
        }
        if (full_population_frequency_scan_period_ms == 0) {
            std::cerr
                << "Error: full_population_frequency_scan_period_ms must be >= 1"
                << std::endl;
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
        
        // Print mapping info
        std::cout << "server_count: " << server_count << std::endl;
        std::cout << "server_buffer_size: " << server_buffer_size << std::endl;
        std::cout << "remote_total: " << remote_total << std::endl;
        std::cout << "remote_mapping: " << (mapping_type == MAPPING_RANGE ? "range" : "stripe") << std::endl;
    }
    
    // Fast mapping self-check for verification
    void verify_mapping() const {
        std::cout << "=== Mapping Self-Check ===" << std::endl;
        
        // Test range mapping
        if (mapping_type == MAPPING_RANGE) {
            for (size_t ep = 0; ep < server_count; ep++) {
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
                ASSERT(ep < server_count);
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
};

}  // namespace rdma
}  // namespace FarLib
