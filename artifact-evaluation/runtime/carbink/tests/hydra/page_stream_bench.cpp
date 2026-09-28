// Manual, two-physical-host streaming diagnostic. This is NOT LLaMA/BFS.
#include "cache/accessor.hpp"
#include "cache/cache.hpp"
#include "utils/parallel.hpp"
#include "utils/wait_trace.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace FarLib;
using namespace FarLib::cache;
namespace {
constexpr size_t page_bytes = 8192;
constexpr size_t words = page_bytes / sizeof(uint64_t);
uint64_t value(size_t page, size_t word) {
    return (uint64_t(page) * 0x9e3779b97f4a7c15ULL) ^
           (uint64_t(word) * 0xd6e8feb86659fd93ULL) ^ 0x123456789abcdef0ULL;
}
size_t parameter(const char *name, size_t fallback, size_t maximum) {
    const char *text = std::getenv(name);
    if (text == nullptr) return fallback;
    char *end = nullptr;
    const auto n = std::strtoull(text, &end, 10);
    if (end == text || *end != 0 || n == 0 || n > maximum) std::abort();
    return n;
}
void run(size_t workers) {
    const size_t count = parameter("HYDRA_STREAM_PAGES", 65536, 1048576);
    const size_t passes = parameter("HYDRA_STREAM_PASSES", 64, 1024);
    std::vector<UniqueFarPtr<void>> objects(count);
    std::vector<uint64_t> sums(count, 0);
    uthread::parallel_for_with_scope<1>(workers, count,
        [&](size_t i, DereferenceScope &scope) {
            auto a = objects[i].allocate_lite<true>(page_bytes, scope);
            auto *p = static_cast<uint64_t *>(a.as_ptr());
            for (size_t j = 0; j < words; ++j) p[j] = value(i, j);
        });
    auto verify = [&] {
        uthread::parallel_for_with_scope<1>(workers, count,
            [&](size_t i, DereferenceScope &scope) {
                auto a = objects[i].access(scope);
                const auto *p = static_cast<const uint64_t *>(a.as_ptr());
                for (size_t j = 0; j < words; ++j) {
                    if (p[j] != value(i, j)) {
                        std::cerr << "HYDRA_STREAM_BYTE_MISMATCH page=" << i
                                  << " word=" << j << std::endl;
                        std::abort();
                    }
                }
            });
    };
    verify(); // Complete-byte warm-up check, excluded from Work.
    const bool diagnostic = std::getenv("HYDRA_STREAM_DIAG") != nullptr &&
        std::strcmp(std::getenv("HYDRA_STREAM_DIAG"), "1") == 0;
    if (diagnostic) {
        profile::begin_allocation_wait_diagnostics(workers);
        wait_trace::start();
    }
    const auto reads_before = profile::collect_rdma_read_post_bytes();
    const auto writes_before = profile::collect_rdma_write_post_bytes();
    profile::start_work();
    profile::thread_start_work();
    const auto begin = std::chrono::steady_clock::now();
    for (size_t pass = 0; pass < passes; ++pass) {
        uthread::parallel_for_with_scope<1>(workers, count,
            [&](size_t i, DereferenceScope &scope) {
                auto a = objects[i].access(scope);
                const auto *p = static_cast<const uint64_t *>(a.as_ptr());
                sums[i] += p[pass % words];
            });
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    profile::thread_end_work();
    profile::end_work();
    const auto reads = profile::collect_rdma_read_post_bytes() - reads_before;
    const auto writes = profile::collect_rdma_write_post_bytes() - writes_before;
    if (diagnostic) {
        (void)profile::end_allocation_wait_diagnostics();
        wait_trace::stop_and_dump();
        profile::print_allocation_wait_diagnostics();
        profile::print_profile_data();
    }
    for (size_t i = 0; i < count; ++i) {
        uint64_t expected = 0;
        for (size_t pass = 0; pass < passes; ++pass) expected += value(i, pass % words);
        if (sums[i] != expected) std::abort();
    }
    verify(); // Full byte oracle again, including pages evicted during Work.
    const auto &c = get_config();
    std::printf("HYDRA_STREAM_RESULT mode=%s pages=%zu page_bytes=%zu passes=%zu "
                "workers=%zu local_bytes=%zu resident_bytes=%zu seconds=%.9f "
                "read_bytes=%lld write_bytes=%lld read_gbps=%.6f write_gbps=%.6f "
                "correctness=full_bytes_before_after_and_work_checksum\n",
                c.ft_method.c_str(), count, page_bytes, passes, workers,
                c.client_buffer_size, c.local_resident_budget_bytes, seconds,
                static_cast<long long>(reads), static_cast<long long>(writes),
                reads * 8.0 / seconds / 1e9, writes * 8.0 / seconds / 1e9);
    objects.clear();
}
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    rdma::Configure config;
    config.from_file(argv[1]);
    if ((!config.is_hydra_mode() && config.ft_enabled()) ||
        config.enable_selective_backup || config.enable_resident_profile_planner ||
        config.max_thread_cnt == 0) return 2;
    runtime_init(config);
    run(config.max_thread_cnt);
    runtime_destroy();
    std::cout << "HYDRA_STREAM_PASS exit cleanup\n";
}
