// Optional control-plane CPU diagnostic. No network or payload is transferred.
#include "cache/alloc/ec_batch_write.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main() {
    using namespace FarLib::cache::ec_batch;
    constexpr size_t groups_per_worker = 65536;
    for (unsigned workers : {1, 4, 8}) {
        EcBatchTokenTable table;
        std::atomic<unsigned> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < workers; ++w) threads.emplace_back([&] {
            EcGroupSendRecord record;
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (size_t i = 0; i < groups_per_worker; ++i) {
                uint64_t id;
                EcBatchToken *token;
                while (!table.acquire(record, &id, &token)) std::this_thread::yield();
                for (uint8_t segment = 0; segment < 6; ++segment) {
                    auto *done = table.complete_segment(id, segment);
                    if ((done != nullptr) != (segment == 5)) std::abort();
                }
                if (!table.release(id)) std::abort();
            }
        });
        while (ready.load() != workers) std::this_thread::yield();
        const auto begin = std::chrono::steady_clock::now();
        go.store(true, std::memory_order_release);
        for (auto &thread : threads) thread.join();
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        if (table.in_use() != 0) std::abort();
        const size_t groups = workers * groups_per_worker;
        std::printf("token_control workers=%u groups=%zu seconds=%.9f "
                    "groups_per_s=%.3f capacity=%zu\n", workers, groups, seconds,
                    groups / seconds, table.capacity());
    }
}
