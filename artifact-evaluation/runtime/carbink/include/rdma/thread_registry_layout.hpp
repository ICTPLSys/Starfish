#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace FarLib::rdma {
// The prefix [0, worker_count) belongs to OS workers. Remaining logical
// clients may be reserved for migrating compaction fibres.
inline bool valid_worker_client_registry(std::span<const size_t> ids,
                                        size_t worker_count,
                                        size_t client_count) {
    if (worker_count > client_count || ids.size() != worker_count) return false;
    std::vector<bool> seen(worker_count, false);
    for (const size_t id : ids) {
        if (id >= worker_count || seen[id]) return false;
        seen[id] = true;
    }
    return true;
}
} // namespace FarLib::rdma
