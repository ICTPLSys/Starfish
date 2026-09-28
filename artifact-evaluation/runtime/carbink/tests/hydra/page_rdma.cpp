// Explicit two-physical-host test; never registered as an automatic CTest.
#include "cache/accessor.hpp"
#include "cache/cache.hpp"
#include "utils/parallel.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using namespace FarLib;
using namespace FarLib::cache;
namespace {
constexpr size_t sizes[] = {1, 7, 8, 9, 511, 512, 513, 2047, 2048,
                            2049, 4095, 4096, 4097, 8191, 8192};
constexpr size_t count = 8192;
constexpr size_t workers = 4;
unsigned char pattern(size_t id, size_t offset, unsigned version) {
    uint64_t x = (id + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= (offset + 1) * 0xbf58476d1ce4e5b9ULL;
    x ^= (version + 1) * 0x94d049bb133111ebULL;
    return static_cast<unsigned char>((x ^ (x >> 37)) >> 19);
}
void bytes(void *ptr, size_t id, size_t n, unsigned version, bool write) {
    auto *p = static_cast<unsigned char *>(ptr);
    for (size_t j = 0; j < n; ++j) {
        if (write) p[j] = pattern(id, j, version);
        else if (p[j] != pattern(id, j, version)) {
            std::cerr << "HYDRA_BYTE_MISMATCH id=" << id << " offset=" << j
                      << " bytes=" << n << " version=" << version << '\n';
            std::abort();
        }
    }
}
void phase(const char *name) { std::cerr << "HYDRA_RDMA_PHASE " << name << std::endl; }
void fault_gate() {
    const char *path = std::getenv("HYDRA_FAULT_GATE");
    if (!path) return;
    std::ofstream(std::string(path) + ".ready") << "remote-data-ready\n";
    phase("fault_ready");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::ifstream(std::string(path) + ".go").good()) {
        if (std::chrono::steady_clock::now() >= deadline) std::abort();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
void run() {
    auto *cache = Cache::get_default();
    std::vector<UniqueFarPtr<void>> objects(count);
    std::vector<unsigned> versions(count, 0);
    std::array<UniqueFarPtr<void>, 16> neighbours;
    {
        RootDereferenceScope scope;
        for (size_t i = 0; i < neighbours.size(); ++i) {
            auto a = neighbours[i].allocate_lite<true>(512, scope);
            bytes(a.as_ptr(), count + i, 512, 0, true);
            assert(neighbours[i].size() == 512);
            assert(a.get_obj() == neighbours[i].obj());
            if (i) assert(neighbours[i].get_entry().hydra_owner() == neighbours[0].get_entry().hydra_owner());
        }
        // Exercise a dynamically allocated handle and the raw free API too.
        auto [obj, p] = cache->allocate<true>(513, true, scope);
        bytes(p, count + 16, 513, 0, true);
        cache->deallocate(obj);
        std::array<far_obj_t, 17> raw;
        for (size_t i = 0; i < raw.size(); ++i) {
            auto a = LiteAccessor<void, true>::allocate(512, scope);
            raw[i] = a.get_obj();
            assert(raw[i].size == 512 && raw[i].get_entry_ptr()->hydra_packed());
            auto typed = a.as<uint64_t>();
            assert(typed.get_obj() == raw[i]);
            *typed = i;
        }
        for (auto o : raw) cache->deallocate(o);
    }
    uthread::parallel_for_with_scope<1>(workers, count,
        [&](size_t i, DereferenceScope &scope) {
            auto a = objects[i].allocate_lite<true>(sizes[i % std::size(sizes)], scope);
            bytes(a.as_ptr(), i, objects[i].size(), 0, true);
            assert(objects[i].get_entry().hydra_packed());
        });
    phase("allocated");
    auto pressure = [&] {
        std::vector<UniqueFarPtr<void>> scratch(count);
        uthread::parallel_for_with_scope<1>(workers, count,
            [&](size_t i, DereferenceScope &scope) {
                auto a = scratch[i].allocate_lite<true>(8192, scope);
                std::memset(a.as_ptr(), 0xa5, 8192);
            });
        size_t remote = 0;
        for (const auto &obj : objects)
            remote += obj.get_entry().load_state().state == REMOTE;
        std::cerr << "HYDRA_REMOTE_HANDLES " << remote << std::endl;
        assert(remote > 0);
        return scratch;
    };
    auto scratch = pressure();
    phase("evicted");
    // Keep pressure alive through the fault gate so checked pages are remote.
    fault_gate();
    auto scan = [&](bool mutate) {
        uthread::parallel_for_with_scope<1>(workers, count,
            [&](size_t i, DereferenceScope &scope) {
                if (mutate) {
                    auto a = objects[i].access<true>(scope);
                    bytes(a.as_ptr(), i, objects[i].size(), versions[i], false);
                    bytes(a.as_ptr(), i, objects[i].size(), ++versions[i], true);
                } else {
                    // Mix old pinned and Lite APIs against the same page owners.
                    if (i & 1) {
                        void *p = cache->sync_fetch(objects[i].obj(), scope);
                        bytes(p, i, objects[i].size(), versions[i], false);
                        cache->release_cache(objects[i].obj(), false);
                    } else {
                        auto a = objects[i].access(scope);
                        bytes(const_cast<void *>(a.as_ptr()), i, objects[i].size(), versions[i], false);
                    }
                }
            });
    };
    UniqueFarPtr<void> moved(std::move(objects[3]));
    assert(objects[3].is_null() && moved.size() == sizes[3]);
    objects[3] = std::move(moved);
    scan(false);
    phase("full_read_verified");
    uthread::parallel_for_with_scope<1>(workers, neighbours.size(),
        [&](size_t i, DereferenceScope &scope) {
            auto a = neighbours[i].access<true>(scope);
            bytes(a.as_ptr(), count + i, 512, 0, false);
            bytes(a.as_ptr(), count + i, 512, 1, true);
        });
    neighbours[3].reset();
    neighbours[11].reset();
    scan(true);
    phase("rewritten");
    scratch.clear();
    scratch = pressure();
    scan(false);
    {
        RootDereferenceScope scope;
        for (size_t i = 0; i < neighbours.size(); ++i) {
            if (neighbours[i].is_null()) continue;
            auto a = neighbours[i].access(scope);
            bytes(const_cast<void *>(a.as_ptr()), count + i, 512, 1, false);
        }
    }
    phase("rewrite_refetch_verified");
    scratch.clear();
    objects.clear();
    for (auto &p : neighbours) p.reset();
    phase("released");
}
} // namespace
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    rdma::Configure config;
    config.from_file(argv[1]);
    assert(config.is_hydra_mode() && config.max_thread_cnt == workers);
    runtime_init(config);
    if (std::getenv("OBJECT_WAIT_TRACE")) wait_trace::start();
    run();
    if (std::getenv("OBJECT_WAIT_TRACE")) wait_trace::stop_and_dump();
    runtime_destroy();
    std::cout << "HYDRA_PAGE_RDMA_PASS complete-bytes boundaries shared-page move free rewrite exit\n";
}
