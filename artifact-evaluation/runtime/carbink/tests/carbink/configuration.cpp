// CPU-only checks for the Carbink configuration boundary.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "rdma/config.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>

namespace {

using FarLib::rdma::Configure;

Configure make_config(const char *method) {
    Configure config;
    config.ft_method = method;
    config.server_count = 1;
    config.server_addr = "127.0.0.1";
    config.server_port = "1";
    config.server_buffer_size = 1ull << 20;
    config.max_thread_cnt = 24;
    config.evacuate_thread_cnt = 4;
    config.compaction_worker_count = 3;
    config.enable_eager_evict = true;
    config.post_process_config();
    return config;
}

void check_carbink_alias(const char *method) {
    const Configure config = make_config(method);
    assert(config.is_carbink_mode());
    assert(config.is_ec_batch_mode());
    assert(!config.is_hydra_mode());
    assert(config.is_page_mode());
    assert(config.ft_small_object(8192));
    assert(!config.ft_small_object(8193));
    assert(config.background_worker_count() == 4 + 3 + 1);
    assert(config.runtime_worker_count() == 24 + 4 + 3 + 1);
    // Cluster selection does not pin a fibre to an OS worker. Compaction
    // clients must not overlap ANY ordinary background-worker client.
    assert(config.compaction_client_base() >= config.runtime_worker_count());
    assert(config.runtime_client_count() ==
           config.runtime_worker_count() + 3 + 4);
}

void check_non_carbink_worker_accounting(const char *method) {
    const Configure config = make_config(method);
    assert(!config.is_carbink_mode());
    assert(config.is_ec_batch_mode());
    assert(config.background_worker_count() == 4);
    assert(config.runtime_client_count() == 24 + 4);
    assert(config.ft_small_object(4095));
    assert(!config.ft_small_object(4096));
}

void check_eager_evict_gate() {
    Configure config = make_config("carbink");
    config.enable_eager_evict = false;
    config.post_process_config();
    // Capacity accounting remains configured even when runtime_init disables
    // the actual eager-eviction pools.
    assert(config.background_worker_count() == 4 + 3 + 1);
    assert(config.runtime_worker_count() == 24 + 4 + 3 + 1);
    assert(config.runtime_client_count() == 24 + 4 + 3 + 1 + 3 + 4);
    assert(config.carbink_evict_client_base() ==
           config.compaction_client_base() + 3);
}

}  // namespace

int main() {
    check_carbink_alias("carbink");
    check_carbink_alias("ec_span");

    const Configure hydra = make_config("hydra");
    assert(hydra.is_hydra_mode());
    assert(!hydra.is_carbink_mode());
    assert(hydra.background_worker_count() == 4);
    assert(hydra.runtime_client_count() == 28);

    check_non_carbink_worker_accounting("ec_batch");
    check_eager_evict_gate();

    std::puts("CARBINK_CONFIGURATION_PASS");
    return 0;
}
