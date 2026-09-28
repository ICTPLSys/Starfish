// Ported from nq-resident-six r2/t8; assertions intentionally retained.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cache/accessor.hpp"
#include "cache/alloc/region_remote_allocator.hpp"
#include "cache/cache.hpp"
#include "rdma/config.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/parallel.hpp"
#include "utils/stats.hpp"

using namespace FarLib;
using namespace FarLib::cache;
using namespace FarLib::rdma;
using namespace std::chrono_literals;

namespace {

// This is a manually-run external-RDMA test.  Keep the object population
// small enough for a pair-B smoke run while allocating enough pressure data
// to cross the local Streaming/Resident boundary.
constexpr size_t kThreadCount = 2;
// Checked payload must exceed the 16MiB local heap. A tiny checked cohort
// could remain local while only the unchecked pressure cohort was evicted.
constexpr size_t kObjectCount = 2048;
constexpr size_t kPressureCount = 1536;
constexpr size_t kRegionBytes = FarLib::allocator::RegionSize;

struct Object {
    UniqueFarPtr<void> ptr;
    size_t bytes = 0;
    uint64_t version = 0;
};

struct ResidentCounts {
    size_t remote = 0;
    size_t resident = 0;
    size_t backed = 0;
};

Configure config;
std::atomic<size_t> allocation_progress{0}, read_progress{0}, pressure_progress{0};

void phase(const char *name) {
    std::cerr << "SIMPLE_RESIDENT_PHASE " << name << std::endl;
}

uint64_t payload_word(size_t object_index, uint64_t version,
                      size_t word_index) {
    uint64_t value = 0x9e3779b97f4a7c15ULL;
    value ^= static_cast<uint64_t>(object_index + 1) *
             0xd6e8feb86659fd93ULL;
    value ^= (version + 1) * 0xa0761d6478bd642fULL;
    value ^= static_cast<uint64_t>(word_index + 1) *
             0xe7037ed1a0b428dULL;
    value ^= value >> 29;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

void fill_payload(void *payload, size_t bytes, size_t object_index,
                  uint64_t version) {
    ASSERT(payload != nullptr);
    ASSERT(bytes != 0 && bytes % sizeof(uint64_t) == 0);
    auto *words = static_cast<uint64_t *>(payload);
    const size_t word_count = bytes / sizeof(uint64_t);
    for (size_t word = 0; word < word_count; ++word) {
        words[word] = payload_word(object_index, version, word);
    }
}

void verify_payload(const void *payload, size_t bytes, size_t object_index,
                    uint64_t version) {
    ASSERT(payload != nullptr);
    ASSERT(bytes != 0 && bytes % sizeof(uint64_t) == 0);
    const auto *words = static_cast<const uint64_t *>(payload);
    const size_t word_count = bytes / sizeof(uint64_t);
    for (size_t word = 0; word < word_count; ++word) {
        ASSERT(words[word] == payload_word(object_index, version, word));
    }
}

size_t object_size(size_t index) {
    // Every size is a multiple of eight and exercises several allocator bins.
    return 4096 * (1 + (index % 5));
}

void read_one(const UniqueFarPtr<void> &object, size_t bytes,
              size_t object_index, uint64_t version, DereferenceScope &scope) {
    auto *cache = Cache::get_default();
    void *payload = cache->sync_fetch(object.obj(), scope);
    verify_payload(payload, bytes, object_index, version);
    cache->release_cache(object.obj(), false);
    read_progress.fetch_add(1, std::memory_order_relaxed);
}

void read_all(const std::vector<Object> &objects) {
    uthread::parallel_for_with_scope<1>(
        kThreadCount, objects.size(),
        [&](size_t index, DereferenceScope &scope) {
            const auto &object = objects[index];
            read_one(object.ptr, object.bytes, index, object.version, scope);
        });
}

void write_one(Object &object, size_t object_index, DereferenceScope &scope) {
    auto *cache = Cache::get_default();
    void *payload = cache->sync_fetch(object.ptr.obj(), scope);
    ++object.version;
    fill_payload(payload, object.bytes, object_index, object.version);
    cache->release_cache(object.ptr.obj(), true);
}

void write_objects(std::vector<Object> &objects,
                   const std::vector<size_t> &indices) {
    uthread::parallel_for_with_scope<1>(
        kThreadCount, indices.size(),
        [&](size_t position, DereferenceScope &scope) {
            const size_t index = indices[position];
            write_one(objects[index], index, scope);
        });
}

void allocate_objects(std::vector<Object> &objects) {
    uthread::parallel_for_with_scope<1>(
        kThreadCount, objects.size(),
        [&](size_t index, DereferenceScope &scope) {
            Object &object = objects[index];
            object.bytes = object_size(index);

            // Half the objects start clean so the first eviction writes a
            // clean remote copy.  Initialization is metadata-clean by design;
            // the dirty half exercises the ordinary write-backed path.
            if ((index & 1u) == 0) {
                auto view = object.ptr.template allocate_lite<false>(
                    object.bytes, scope);
                fill_payload(const_cast<void *>(view.as_ptr()), object.bytes,
                             index, object.version);
            } else {
                auto view = object.ptr.template allocate_lite<true>(
                    object.bytes, scope);
                fill_payload(view.as_ptr(), object.bytes, index,
                             object.version);
            }
            allocation_progress.fetch_add(1, std::memory_order_relaxed);
        });
}

void allocate_pressure(std::vector<UniqueFarPtr<void>> &pressure,
                       std::vector<size_t> &sizes) {
    sizes.resize(pressure.size());
    uthread::parallel_for_with_scope<1>(
        kThreadCount, pressure.size(),
        [&](size_t index, DereferenceScope &scope) {
            sizes[index] = object_size(index + kObjectCount);
            auto view = pressure[index].template allocate_lite<true>(
                sizes[index], scope);
            std::memset(view.as_ptr(), 0xa5, sizes[index]);
            pressure_progress.fetch_add(1, std::memory_order_relaxed);
        });
}

void drive_evictions() {
    // Allocation pressure above already drives eviction. The legacy
    // invoke_eviction() is an allocation-wait primitive, not a synchronous
    // "evict one round" API: calling it after free space has recovered can
    // wait forever because that path does not publish an explicit request.
    // Let in-flight completions settle; assertions below still require the
    // clean/dirty/remote events to have actually happened.
    for (size_t pass = 0; pass < 100; ++pass)
        uthread::yield();
}

ResidentCounts count_entries(const std::vector<Object> &objects) {
    ResidentCounts counts;
    for (const Object &object : objects) {
        const auto &entry = object.ptr.get_entry();
        const auto state = entry.load_state();
        if (entry.is_resident_local()) {
            ++counts.resident;
        }
        if (entry.has_remote()) {
            ++counts.remote;
            // The remote class is measured remote heat, not a local Resident
            // placement request.  The allocator unit test checks the full
            // class/permutation table; this RDMA test checks every backed
            // address remains a valid six-class remote allocation.
            const uint8_t remote_class =
                allocator::remote::remote_global_heap.simple_class_for(
                    entry.remote_addr());
            ASSERT(remote_class < 6);
        }
        if (entry.has_remote_backup_reservation()) {
            ++counts.backed;
            ASSERT(entry.has_remote());
            ASSERT(state.state != REMOTE || !entry.is_resident_local());
        }
        if (state.state == REMOTE) {
            ASSERT(entry.has_remote());
            ASSERT(!entry.is_resident_local());
        }
    }
    return counts;
}

void check_snapshot(const Cache::SimpleResidentStateSnapshot &state,
                    bool compare_region_ledgers) {
    ASSERT(state.backup_used_bytes <= state.backup_budget_bytes);
    ASSERT(state.backup_peak_bytes >= state.backup_used_bytes);
    ASSERT(state.resident_regions <= state.resident_budget_regions);
    ASSERT(state.resident_bytes == state.resident_regions * kRegionBytes);
    ASSERT(state.resident_bytes <= state.resident_budget_bytes);
    if (compare_region_ledgers && state.local_resident_mapping) {
        ASSERT(state.local_hot_regions == state.resident_regions);
        ASSERT(state.local_cold_regions == state.streaming_regions);
    }
}

bool is_config_file(const char *argument) {
    std::ifstream input(argument);
    return input.good();
}

void set_default_external_config(const char *server_address,
                                 const char *server_port) {
    config.server_addr = server_address;
    config.server_port = server_port;
    config.server_buffer_size = 128ULL * 1024 * 1024;
    config.client_buffer_size = 16ULL * 1024 * 1024;
    config.evict_batch_size = 64 * 1024;
    config.max_thread_cnt = kThreadCount;
    config.evacuate_thread_cnt = 1;
    config.enable_eager_evict = true;
    config.exclusive_cache = true;
    config.enable_selective_backup = true;
    config.remote_backup_budget_bytes = 4ULL * 1024 * 1024;
    config.remote_backup_mode = "segmented";
    config.local_resident_budget_bytes = 2 * kRegionBytes;
    config.enable_region_resident_placement = true;
    config.region_placement_bind_groups = false;
    config.enable_logical_object_profile = true;
    config.logical_object_profile_sample_shift = 0;
    config.post_process_config();
}

void run_test() {
    const char *resident_env = std::getenv("FARLIB_SIMPLE_LOCAL_RESIDENT");
    ASSERT(resident_env != nullptr);
    const bool grouped = std::strcmp(resident_env, "1") == 0;
    ASSERT(Cache::get_default()->simple_resident_state_snapshot()
               .local_resident_mapping == grouped);
    ASSERT(config.exclusive_cache);
    ASSERT(config.enable_selective_backup);
    ASSERT(config.enable_region_resident_placement);
    ASSERT(config.local_resident_budget_bytes >= kRegionBytes);

    auto *cache = Cache::get_default();
    std::vector<Object> objects(kObjectCount);
    allocate_objects(objects);
    phase("objects_allocated");
    read_all(objects);
    phase("objects_initial_read");

    std::vector<UniqueFarPtr<void>> pressure(kPressureCount);
    std::vector<size_t> pressure_sizes;
    allocate_pressure(pressure, pressure_sizes);
    phase("pressure_allocated_1");
    drive_evictions();
    phase("first_evictions_done");

    const int64_t clean_before = profile::collect_clean_evict_bytes();
    const int64_t dirty_before = profile::collect_dirty_evict_bytes();
    const ResidentCounts first = count_entries(objects);
    ASSERT(first.resident > 0);
    ASSERT(first.remote > 0);

    // Pressure objects are no longer needed after they have forced the first
    // remote wave.  Free them before fetching the checked object population.
    for (auto &object : pressure) object.reset();
    phase("pressure_freed_1");

    read_all(objects);
    phase("first_remote_fetch_done");
    auto fetched_state = cache->simple_resident_state_snapshot();
    check_snapshot(fetched_state, false);
    ASSERT(fetched_state.backup_used_bytes > 0);

    std::vector<size_t> backed_indices;
    for (size_t index = 0; index < objects.size(); ++index) {
        if (objects[index].ptr.get_entry().has_remote_backup_reservation()) {
            backed_indices.push_back(index);
        }
    }
    ASSERT(!backed_indices.empty());

    // Writes invalidate retained clean backups before changing the payload.
    std::vector<size_t> write_indices;
    for (size_t position = 0; position < backed_indices.size(); ++position) {
        if ((position & 1u) == 0) write_indices.push_back(backed_indices[position]);
    }
    if (write_indices.empty()) write_indices.push_back(backed_indices.front());
    const int64_t invalidated_before =
        profile::collect_remote_backup_invalidated_bytes();
    write_objects(objects, write_indices);
    phase("backup_writes_done");
    ASSERT(profile::collect_remote_backup_invalidated_bytes() >
           invalidated_before);

    // Refill the pressure population after the clean backups are resident.
    // This second pressure wave is what makes the following clean-backup
    // reuse and dirty write-backed evictions deterministic even though the
    // first pressure population has been freed.
    allocate_pressure(pressure, pressure_sizes);
    phase("pressure_allocated_2");
    drive_evictions();
    phase("second_evictions_done");
    ASSERT(profile::collect_clean_evict_bytes() > clean_before);
    ASSERT(profile::collect_dirty_evict_bytes() > dirty_before);

    const ResidentCounts second = count_entries(objects);
    ASSERT(second.remote > 0);
    const auto second_state = cache->simple_resident_state_snapshot();
    check_snapshot(second_state, false);
    ASSERT(second_state.backup_peak_bytes >= fetched_state.backup_used_bytes);

    // Move a remote object through the public UniqueFarPtr move path and
    // verify that both the entry metadata and the payload survive relocation.
    // Keep the pressure population alive until this lookup: once it is freed,
    // the object population may fit in the available local buffer and no
    // longer contain a REMOTE entry.
    size_t move_index = objects.size();
    for (size_t index = 0; index < objects.size(); ++index) {
        if (objects[index].ptr.get_entry().load_state().state == REMOTE) {
            move_index = index;
            break;
        }
    }
    ASSERT(move_index < objects.size());
    UniqueFarPtr<void> moved;
    moved = std::move(objects[move_index].ptr);
    ASSERT(objects[move_index].ptr.get_entry().load_state().state == FREE);
    {
        RootDereferenceScope scope;
        read_one(moved, objects[move_index].bytes, move_index,
                 objects[move_index].version, scope);
    }
    objects[move_index].ptr = std::move(moved);
    ASSERT(moved.get_entry().load_state().state == FREE);
    phase("remote_move_done");

    read_all(objects);
    for (auto &object : pressure) object.reset();
    phase("final_reads_pressure_freed");

    // Stop background movement before comparing the independently locked
    // resident and budget ledgers.  Promotion/demotion success itself is
    // workload-dependent and is covered by the deterministic allocator unit
    // test; this integration test checks the resulting hard invariants.
    cache->quiesce_background_evacuation();
    phase("quiesced");
    const auto quiesced = cache->simple_resident_state_snapshot();
    check_snapshot(quiesced, true);
    ASSERT(quiesced.resident_regions > 0);
    ASSERT(quiesced.streaming_regions > 0);

    for (auto &object : objects) {
        auto &entry = object.ptr.get_entry();
        object.ptr.reset();
        ASSERT(entry.load_state().state == FREE);
        // REMOTE -> FREE may retain historical heat/q bits until reset().
        // Only ownership bits must be clear once resources are released.
        ASSERT(!entry.is_resident_local());
        ASSERT(!entry.has_remote_backup_reservation());
    }

    // Match Cache teardown: the direct read/move on the main fibre can leave
    // a private Region, invisible to public-list reclamation until returned.
    allocator::release_all_thread_heap_regions_for_shutdown();
    allocator::global_heap.finish_async_full_returns_for_shutdown();
    allocator::global_heap.reclaim_all_deallocated_regions();
    const auto final_state = cache->simple_resident_state_snapshot();
    check_snapshot(final_state, true);
    ASSERT(final_state.backup_used_bytes == 0);
    ASSERT(allocator::remote::remote_global_heap.get_used_bytes() == 0);
    ASSERT(allocator::global_heap.get_used_bytes() == 0);
    phase("cleanup_verified");
    std::cout << "SIMPLE_RESIDENT_RDMA_INTEGRATION_OK\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: test_simple_resident_integration "
                     "<config-file|server-host> [server-port]\n";
        return 2;
    }

    if (is_config_file(argv[1])) {
        ASSERT(argc == 2);
        config.from_file(argv[1]);
    } else {
        const char *port = argc == 3 ? argv[2] : "50000";
        set_default_external_config(argv[1], port);
    }

    FarLib::runtime_init(config);
    std::jthread reporter([](std::stop_token stop) {
        if (std::getenv("INTEGRATION_PROGRESS") == nullptr) return;
        while (!stop.stop_requested()) {
            auto s = Cache::get_default()->simple_resident_state_snapshot();
            std::cerr << "TEST_PROGRESS alloc=" << allocation_progress.load()
                      << " read=" << read_progress.load()
                      << " pressure=" << pressure_progress.load()
                      << " free_bytes=" << allocator::global_heap.get_free_size()
                      << " backup=" << s.backup_used_bytes
                      << " R=" << s.resident_regions
                      << " S=" << s.streaming_regions << std::endl;
            for (size_t bin : {32u, 36u, 38u, 40u, 41u}) {
                auto b = allocator::global_heap.get_bin_snapshot(bin);
                std::cerr << "TEST_BIN bin=" << bin
                          << " free=" << b.free_regions
                          << " usable=" << b.target_usable_regions
                          << " full=" << b.full_regions
                          << " ready=" << b.can_allocate_now << std::endl;
            }
            for (size_t i=0; i<10 && !stop.stop_requested(); ++i)
                std::this_thread::sleep_for(100ms);
        }
    });
    run_test();
    reporter.request_stop();
    reporter.join();
    FarLib::runtime_destroy();
    return 0;
}
