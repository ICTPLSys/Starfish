#include "rdma/thread_registry_layout.hpp"
#include <cassert>
#include <cstdio>
#include <numeric>
#include <vector>

int main() {
    using FarLib::rdma::valid_worker_client_registry;
    std::vector<size_t> ids(35);
    std::iota(ids.begin(), ids.end(), 0);
    assert(valid_worker_client_registry(ids, 35, 39)); // four private RPC clients
    assert(valid_worker_client_registry(ids, 35, 35)); // Hydra unchanged
    assert(!valid_worker_client_registry(ids, 35, 34));
    assert(!valid_worker_client_registry(ids, 34, 39));
    ids[34] = 35; // private logical client must not belong to an OS worker
    assert(!valid_worker_client_registry(ids, 35, 39));
    ids[34] = 33; // reject two threads sharing one ordinary client
    assert(!valid_worker_client_registry(ids, 35, 39));
    std::puts("CARBINK_THREAD_REGISTRY_PASS workers=35 clients=39 private=4");
}
