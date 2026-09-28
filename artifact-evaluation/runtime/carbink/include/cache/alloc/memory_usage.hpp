#pragma once
#include <cstdint>
#include <iostream>

#include "cache/region_based_allocator.hpp"
#include "cache/alloc/remote_allocator.hpp"
#include "utils/control.hpp"

constexpr size_t MB = 1024 * 1024;
constexpr size_t GB = 1024 * 1024 * 1024;

namespace FarLib {
namespace allocator {

inline void print_combined_memory_usage() {
    uint64_t remote_exact = 0;
#ifdef REMOTE_REGION_ALLOCATOR
    remote_exact = remote::remote_global_heap.get_used_bytes();
#endif
    int64_t local_used = global_heap.get_used_bytes();
    uint64_t combined = remote_exact + static_cast<uint64_t>(local_used);

    std::cout << "remote.exact_allocated_bytes (in GB): " << static_cast<double>(remote_exact) / GB << std::endl;
#ifdef REMOTE_REGION_ALLOCATOR
    size_t server_count = FarLib::get_config().server_count;
    for (size_t i = 0; i < server_count; i++) {
        uint64_t s_used = remote::remote_global_heap.get_server_used_bytes(i);
        std::cout << "remote.server[" << i << "].used_bytes (in GB): " << static_cast<double>(s_used) / GB << std::endl;
    }
#endif
    std::cout << "local.used_bytes (in GB): " << static_cast<double>(local_used) / GB << std::endl;
    std::cout << "combined.used_bytes (in GB): " << static_cast<double>(combined) / GB << std::endl;
}

}  // namespace allocator
}  // namespace FarLib
