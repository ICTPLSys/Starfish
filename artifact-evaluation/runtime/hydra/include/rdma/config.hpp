/**
 * RDMA Configuration
 */
#pragma once

#include <infiniband/verbs.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "utils/debug.hpp"

namespace FarLib {

// Basic parameters, not configurable
constexpr size_t PAGE_SIZE = 4096;

namespace rdma {

namespace detail {

template <typename T>
bool read_config_value(std::istream &input, T &value) {
    return static_cast<bool>(input >> value);
}

// uint8_t is commonly an unsigned char, so operator>> would consume one
// character instead of a decimal value. Validate the token and range first.
inline bool read_config_value(std::istream &input, uint8_t &value) {
    std::string token;
    if (!(input >> token)) return false;
    for (const char character : token) {
        if (character < '0' || character > '9') {
            input.setstate(std::ios::failbit);
            return false;
        }
    }
    unsigned int parsed = 0;
    const char *first = token.data();
    const char *last = first + token.size();
    const auto result = std::from_chars(first, last, parsed, 10);
    if (result.ec != std::errc{} || result.ptr != last ||
        parsed > static_cast<unsigned int>(
                     std::numeric_limits<uint8_t>::max())) {
        input.setstate(std::ios::failbit);
        return false;
    }
    value = static_cast<uint8_t>(parsed);
    return true;
}

}  // namespace detail

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

    // Fault-tolerance method selection (ft_method).  FT_SPONGE is the ported
    // single-object EC stripe layout.  FT_EC_BATCH is the batched group path:
    // eviction collects four small objects, the client-side RS(4,2) codec
    // encodes two parity slots and the six segments are written to six
    // distinct endpoints.  A legal ft_method != none combination is accepted by
    // self_check(); only illegal combinations fail fast, and the RDMA write
    // path / ACK are not wired yet.
    enum FaultToleranceMethod { FT_NONE, FT_SPONGE, FT_EC_BATCH };
    FaultToleranceMethod ft_method_type = FT_NONE;

    // Endpoints one ec_batch group needs: 4 data + 2 parity segments, each on
    // its own endpoint.  Kept next to the enum because is_ec_batch_mode()'s
    // server_count check is derived from it.
    static constexpr size_t ft_ec_batch_endpoint_count = 6;
    // A configured standby needs one additional endpoint beyond the six
    // endpoints occupied by a group.  The standby remains connected and
    // registered, but allocation excludes it until a failure is reported.
    static constexpr size_t ft_ec_batch_standby_server_count =
        ft_ec_batch_endpoint_count + 1;

    // Canonical EC shard size.  This must stay equal to the library's region
    // size; cache/alloc/small_object_stripe.hpp static_asserts that it matches
    // ::FarLib::allocator::RegionSize, so the two cannot drift apart.
    static constexpr size_t ft_ec_shard_size_bytes = 256 * 1024;

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
        if (!detail::read_config_value(ifs, this->VAR)) {               \
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

        // Parse ft_method: none | sponge | ec_batch.  Anything else is a hard
        // error; the supported-method and parameter checks live in
        // self_check().
        if (ft_method == "none" || ft_method.empty()) {
            ft_method_type = FT_NONE;
        } else if (ft_method == "sponge") {
            ft_method_type = FT_SPONGE;
        } else if (ft_method == "ec_batch" || ft_method == "hydra" ||
                   ft_method == "ec_split") {
            ft_method_type = FT_EC_BATCH;
        } else {
            config_error(
                std::string(
                    "ft_method must be one of none, sponge, ec_batch, hydra, ec_split (got \"") +
                ft_method + "\")");
        }

        profiling_enabled = profiling_method != "disabled";
        profiling_object_access_enabled =
            profiling_method == "coarse_grained";
        profiling_dereference_enabled =
            profiling_method == "fine_grained";
    }

    // Report a configuration error and stop immediately, without a core dump.
    // (The pre-existing checks below use std::abort(); the new ft_* checks use
    // this helper so that an invalid run fails fast with a clear message and a
    // normal non-zero exit status.)
    [[noreturn]] static void config_error(const std::string &message) {
        std::cerr << "Error: " << message << std::endl;
        std::cerr.flush();
        std::exit(2);
    }

    bool ft_enabled() const { return ft_method_type != FT_NONE; }

    // ft_method=ec_batch: the batched four-object group write path (staging +
    // local RS(4,2) encode + six single-sided writes).  Every ec_batch-only
    // code path must be guarded by this predicate so that ft_method=none (and
    // sponge) keep their behaviour unchanged.
    bool is_ec_batch_mode() const {
        return ft_method_type == FT_EC_BATCH;
    }
    bool is_hydra_mode() const {
        return ft_method == "hydra" || ft_method == "ec_split";
    }
    // ft_method=sponge only: the sponge commit path (control CQ ACKs) is the
    // only user of check_sponge_ack_cq_ft(); ec_batch never commits through it,
    // so polling must stay off for ec_batch and none.
    bool is_sponge_mode() const { return ft_method_type == FT_SPONGE; }

    // True for objects that the EC stripe layout takes over.
    bool ft_small_object(size_t size) const {
        return ft_enabled() && size < ft_small_object_cutoff;
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
        std::cerr << max_thread_cnt * qp_count * qp_send_cap << std::endl;
        post_process_config();
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

        // Fault-tolerance / EC layout validation.  Everything below only runs
        // when ft_method != none, so the default path is untouched.
        if (ft_enabled()) {
            if (is_hydra_mode() &&
                (enable_selective_backup || enable_logical_object_profile ||
                 enable_resident_profile_planner || region_placement_bind_groups ||
                 enable_region_hotness_placement || enable_region_fetch_hotness_placement)) {
                config_error("hydra has no retained backup or behavior-group planner; disable these options (a fixed Resident budget is supported)");
            }
            const size_t ec_shard_count = ft_ec_data_shards + ft_ec_parity_shards;
            if (ft_standby_endpoint < -1) {
                config_error("ft_standby_endpoint must be -1 (disabled) or a "
                             "non-negative endpoint index, got " +
                             std::to_string(ft_standby_endpoint));
            }
            if (!exclusive_cache) {
                config_error("ft_method=" + ft_method +
                             " requires exclusive_cache=1");
            }

            // ec_batch needs one endpoint per segment: the four data and two
            // parity segments of a group are written to six *distinct*
            // endpoints.  Checked before the generic per-shard check so an
            // ec_batch run reports its own requirement, not a stripe one.
            if (is_ec_batch_mode() &&
                server_count <
                    static_cast<int>(ft_ec_batch_endpoint_count)) {
                config_error(
                    "ft_method=" + ft_method + " requires server_count >= " +
                    std::to_string(ft_ec_batch_endpoint_count) +
                    " (the 4 data + 2 parity segments of one group must land "
                    "on 6 distinct endpoints), got " +
                    std::to_string(server_count));
            }
            if (server_count < static_cast<int>(ec_shard_count)) {
                config_error("ft_method=" + ft_method +
                             " requires server_count >= " +
                             std::to_string(ec_shard_count) +
                             " (one memory server per EC shard), got " +
                             std::to_string(server_count));
            }
            // The standby is an ordinary configured/connected endpoint.  It
            // only changes allocation placement for ec_batch, where one
            // additional endpoint is required beyond the six group slots.
            // Keep ft_method=none (and the existing sponge/no-standby path)
            // byte-for-byte compatible when the option is left at -1.
            if (is_ec_batch_mode() && ft_standby_endpoint >= 0) {
                if (server_count <
                    static_cast<int>(ft_ec_batch_standby_server_count)) {
                    config_error(
                        "ft_method=" + ft_method +
                        " with ft_standby_endpoint requires server_count >= " +
                        std::to_string(ft_ec_batch_standby_server_count) +
                        " (six active endpoints plus one standby), got " +
                        std::to_string(server_count));
                }
                if (ft_standby_endpoint >= server_count) {
                    config_error(
                        "ft_standby_endpoint must be in [0, server_count), got " +
                        std::to_string(ft_standby_endpoint) +
                        " for server_count=" + std::to_string(server_count));
                }
            }
            if (ft_ec_data_shards != 4 || ft_ec_parity_shards != 2) {
                config_error(
                    "ft_method=" + ft_method +
                    " supports only a 4+2 EC layout, got " +
                    std::to_string(ft_ec_data_shards) + "+" +
                    std::to_string(ft_ec_parity_shards));
            }
            if (ft_small_object_cutoff == 0) {
                config_error("ft_small_object_cutoff must be > 0");
            }
            if (ft_small_stripe_shard_size_bytes != ft_ec_shard_size_bytes) {
                config_error(
                    "ft_small_stripe_shard_size_bytes must equal the region "
                    "size (" +
                    std::to_string(ft_ec_shard_size_bytes) + "), got " +
                    std::to_string(ft_small_stripe_shard_size_bytes));
            }
            // The keys above are *validated* here and nowhere else: a legal
            // ft_method != none combination is accepted, an illegal one fails
            // fast above.  Note that the EC stripe layout, the allocator-side
            // dispatch and the address reverse-mapping are the only parts that
            // are ported so far; parity encoding, the write path and the ACK
            // path are not wired yet, so such a run is only usable up to the
            // point where the write path is needed.  This is informational, not
            // an error - the configuration itself is legal.
            if (is_ec_batch_mode()) {
                std::cerr
                    << "FT runtime: ft_method=" << ft_method
                    << " RS(4,2), six single-sided segment writes, client-side recovery"
                    << std::endl;
                if (is_hydra_mode())
                    std::cerr << "hydra.layout page_bytes=8192 data_shards=4 parity_shards=2 shard_bytes=2048 max_packed_object_bytes=8192" << std::endl;
            } else {
                std::cerr << "Warning: ft_method=" << ft_method
                          << " accepted (EC layout only: parity encoding, "
                             "write path and ACK are not implemented yet)"
                          << std::endl;
            }
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
        std::cout << "ft_method: " << (ft_method.empty() ? "none" : ft_method)
                  << std::endl;
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
