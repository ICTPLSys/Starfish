#include "utils/control.hpp"

#include "cache/cache.hpp"
#include "rdma/config.hpp"
#include "utils/fork_join.hpp"
#include "utils/uthreads.hpp"

#include <vector>

#ifndef NO_REMOTE

#include "rdma/client.hpp"

namespace FarLib {
using namespace FarLib::rdma;
static uint8_t mode;
static Configure global_config;

std::atomic_size_t thread_id_global;

namespace {
void register_current_cluster_thread_ids(size_t count) {
    std::vector<pthread_t> tids(count);
    const size_t actual =
        uthread::get_current_worker_sys_ids(tids.data(), tids.size());
    ASSERT(actual == count);
    for (size_t i = 0; i < count; ++i) {
        rdma::register_thread_id(tids[i], i);
    }
}
}

void runtime_init(const rdma::Configure &config, bool enable_cache) {
    mode = 0;
    global_config = config;
    global_config.self_check();
    
    // Run mapping self-check (fast verification)
    global_config.verify_mapping();
    const bool separate_background =
        uthread::separate_background_cluster_enabled();
    if (separate_background &&
        (!enable_cache || !config.enable_eager_evict ||
         config.evacuate_thread_cnt == 0)) {
        ERROR("separate background cluster requires eager cache eviction");
    }
    const char *legacy_pin = std::getenv("FARLIB_PIN_RDMA_THREAD");
    if (separate_background && legacy_pin != nullptr &&
        legacy_pin[0] == '1') {
        ERROR("separate background cluster requires FARLIB_PIN_RDMA_THREAD=0");
    }
    if (separate_background) {
        rdma::reset_thread_id_registry(config.max_thread_cnt);
    } else {
        rdma::disable_thread_id_registry();
    }
    if (enable_cache) {
        mode |= Configure::MODE_ENABLE_CACHE | Configure::MODE_ENABLE_UTHREAD;
    }
    ClientControl::init_default(config);
    // client.init();

    if (config.enable_eager_evict) {
        std::cout << "enable eager evict" << std::endl;
        clients.resize(config.max_thread_cnt + config.evacuate_thread_cnt);
    }
    else {
        clients.resize(config.max_thread_cnt);
    }
    if (!separate_background) {
        install_read_batch_yield_progress();
    }
    
    if (mode & Configure::MODE_ENABLE_UTHREAD) {
        if (config.enable_eager_evict) {
            uthread::runtime_init(
                separate_background
                    ? config.max_thread_cnt
                    : config.max_thread_cnt + config.evacuate_thread_cnt);
        }
        else {
            uthread::runtime_init(config.max_thread_cnt);
        }
        if (separate_background) {
            register_current_cluster_thread_ids(config.max_thread_cnt);
        }
    }
    if (mode & Configure::MODE_ENABLE_CACHE) {
        Cache::init_default(ClientControl::get_default()->get_buffer(),
                            config.client_buffer_size,
                            config.remote_total, config.evict_batch_size);
    }
    if (separate_background) {
        rdma::seal_thread_id_registry();
        // Install only after both clusters and every client id are registered;
        // the B2 yield hook may instantiate ThreadInfo on its first use.
        install_read_batch_yield_progress();
    }
}

void runtime_destroy() {
    // Removal happens while Client storage is still alive. The application
    // fibres have joined before runtime destruction; cache teardown follows.
    remove_read_batch_yield_progress();
    if (mode & Configure::MODE_ENABLE_CACHE) {
        Cache::destroy_default();
    }
    if (mode & Configure::MODE_ENABLE_UTHREAD) {
        uthread::runtime_destroy();
    }
    ClientControl::destroy_default();
}

void runtime_quiesce_cache() {
    if (mode & Configure::MODE_ENABLE_CACHE) {
        Cache::quiesce_default();
        allocator::remote::flush_all_registered_thread_heaps();
    }
}

const Configure &get_config() { return global_config; }
}  // namespace FarLib

#else

namespace FarLib {
using namespace FarLib::rdma;
static Configure global_config;
void runtime_init(const rdma::Configure &config, bool enable_cache) {
    global_config = config;
    uthread::runtime_init(config.max_thread_cnt);
#ifdef USE_BUMP_ALLOCATOR
    void *heap_ptr = mmap(
        nullptr, allocator::HeapSize, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANON | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT), -1, 0);
    if (heap_ptr == MAP_FAILED) {
        ERROR("mmap failed");
    }
    allocator::heap = static_cast<std::byte *>(heap_ptr);
#endif
}

void runtime_destroy() {
#ifdef USE_BUMP_ALLOCATOR
    uthread::runtime_destroy();
    munmap(allocator::heap, allocator::HeapSize);
#endif
}

void runtime_quiesce_cache() {}

const Configure &get_config() { return global_config; }
}  // namespace FarLib

#endif
