// Synthetic accounting tests. No RDMA device, application or server is used.
#include "utils/stats.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

// Exercise the real inline legacy hooks/reset without starting libfibre.
namespace FarLib::profile {
void ThreadLocalProfileData::register_thread() {}
void ThreadLocalProfileData::unregister_thread() {}
ThreadLocalProfileData &get_tlpd() {
    static thread_local ThreadLocalProfileData value;
    return value;
}
}
namespace traffic = FarLib::profile::work_traffic;
namespace profile = FarLib::profile;

int main() {
    profile::count_rdma_read_post(1000);
    profile::count_rdma_write_post(2000);
    traffic::count_rmw_read(3000);
    traffic::begin();
    profile::count_rdma_read_post(10);
    profile::count_rdma_read_posts(2, 30);
    profile::count_rdma_write_post(50); // Includes any RMW WRITE once.
    traffic::count_rmw_read(7);
    profile::get_tlpd().reset(); // Actual legacy reset must not erase the window.
    profile::count_rdma_read_post(2);
    const auto first = traffic::end();
    assert(first.window_id == 1);
    assert(first.bytes.read_bytes == 42);
    assert(first.bytes.write_bytes == 50);
    assert(first.bytes.rmw_read_bytes == 7);
    assert(first.fetch_bytes() == 42 && first.eviction_bytes() == 57);
    const std::string frozen = first.line();
    profile::count_rdma_read_post(9000);
    profile::count_rdma_write_post(9000);
    traffic::count_rmw_read(9000);
    assert(first.line() == frozen);
    assert(frozen.find("scope=client_rdma") != std::string::npos);
    assert(frozen.find("schema_version=1") != std::string::npos);
    assert(frozen.find("eviction_bytes=57") != std::string::npos);

    traffic::begin();
    const auto zero = traffic::end();
    assert(zero.window_id == 2);
    assert(zero.fetch_bytes() == 0 && zero.eviction_bytes() == 0);

    traffic::begin();
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < 8; ++t) {
        workers.emplace_back([] {
            for (unsigned i = 0; i < 1000; ++i) {
                profile::count_rdma_read_post(3);
                profile::count_rdma_write_post(5);
                traffic::count_rmw_read(7);
            }
        });
    }
    for (auto &worker : workers) worker.join();
    const auto parallel = traffic::end(); // Exited/new workers remain represented.
    assert(parallel.window_id == 3);
    assert(parallel.bytes.read_bytes == 24000);
    assert(parallel.bytes.write_bytes == 40000);
    assert(parallel.bytes.rmw_read_bytes == 56000);
    assert(parallel.eviction_bytes() == 96000);
    profile::get_tlpd().reset();
    traffic::begin();
    profile::count_rdma_write_post(11);
    const auto next = traffic::end();
    assert(next.window_id == 4 && next.eviction_bytes() == 11);
    assert(next.fetch_bytes() == 0 && next.bytes.rmw_read_bytes == 0);
    traffic::print(first);
    traffic::print(zero);
    traffic::print(parallel);
    traffic::print(next);
    std::cout << "work traffic tests passed\n";
}
