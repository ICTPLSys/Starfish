#include "hydra/page_pool.hpp"
#include <atomic>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

int main() {
    using namespace FarLib;
    using namespace FarLib::cache;
    static_assert(sizeof(FarObjectEntry) == 48);
    hydra::Page page;
    std::array<unsigned char, hydra::kPageBytes> data{}, relocated{};
    page.owner.reset<true>(data.data(), FarObjectEntry::RemoteAddrInvalid48,
                           data.size());
    page.owner.hydra_mark_page();
    std::array<FarObjectEntry, 16> handles;
    for (size_t i = 0; i < handles.size(); ++i) {
        handles[i].hydra_attach(&page.owner, i * 512, 511);
        assert(handles[i].object_size() == 511);
        assert(handles[i].load_state().size == hydra::kPageBytes);
        std::memset(handles[i].local_addr(), static_cast<int>(i + 1), 511);
        assert(!handles[i].try_mark_recomputable_before_eviction());
    }
    handles[7].mark_dirty();
    assert(page.owner.load_state().dirty && handles[0].load_state().dirty);
    std::memcpy(relocated.data(), data.data(), data.size());
    page.owner.set_local_addr(relocated.data());
    for (size_t i = 0; i < handles.size(); ++i) {
        auto *p = static_cast<unsigned char *>(handles[i].local_addr());
        assert(p == relocated.data() + i * 512);
        for (size_t j = 0; j < 511; ++j) assert(p[j] == i + 1);
        assert(p[511] == 0);
    }
    FarObjectEntry moved;
    moved.hydra_take_handle(handles[7]);
    assert(handles[7].load_state().state == FREE);
    assert(moved.object_size() == 511 && moved.hydra_offset() == 7 * 512);
    assert(moved.hydra_owner() == &page.owner);
    moved.set_free();
    assert(page.owner.load_state().state == LOCAL);

    // One shared fetch winner; aliases cannot launch independent fetches.
    auto remote = page.owner.load_state();
    remote.state = REMOTE;
    page.owner.set_state(remote);
    std::atomic<size_t> winners{0};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 16; ++i) {
        threads.emplace_back([&, i] {
            if (i == 7) return;
            auto old = handles[i].load_state();
            while (old.state == REMOTE) {
                auto fetching = old;
                fetching.state = FETCHING;
                if (handles[i].cas_state_weak(old, fetching)) {
                    ++winners;
                    break;
                }
            }
        });
    }
    for (auto &thread : threads) thread.join();
    assert(winners == 1 && page.owner.load_state().state == FETCHING);
    remote.state = LOCAL;
    remote.ref_cnt = 1; // independent runtime I/O reference
    page.owner.set_state(remote);
    // A page-level aggregate pin can race the BUSY -> FETCHING publication.
    // The publisher must retain both the independent I/O reference and the
    // aggregate pin while preserving the other state fields.
    auto busy_with_io_ref = page.owner.load_state();
    busy_with_io_ref.state = BUSY;
    busy_with_io_ref.ref_cnt = 1;
    busy_with_io_ref.dirty = 1;
    busy_with_io_ref.hotness = 2;
    page.owner.set_state(busy_with_io_ref);
    page.owner.pin();
    assert(page.owner.load_state().ref_cnt == 2);
    assert(page.owner.hydra_publish_fetching());
    auto published = page.owner.load_state();
    assert(published.state == FETCHING && published.ref_cnt == 2);
    assert(published.dirty == 1 && published.hotness == 2);
    page.owner.unpin();
    published = page.owner.load_state();
    assert(published.state == FETCHING && published.ref_cnt == 1);

    // Exercise both orderings of the publisher and a page pin/unpin.  The
    // phase barrier keeps the owner reset outside the other thread's state
    // operations without requiring a C++20 std::barrier.
    constexpr unsigned kPublishPinRounds = 4096;
    std::atomic<unsigned> barrier_arrived{0};
    std::atomic<unsigned> barrier_phase{0};
    auto phase_barrier = [&](unsigned phase) {
        if (barrier_arrived.fetch_add(1, std::memory_order_acq_rel) == 1) {
            barrier_arrived.store(0, std::memory_order_release);
            barrier_phase.store(phase + 1, std::memory_order_release);
        } else {
            while (barrier_phase.load(std::memory_order_acquire) <= phase)
                std::this_thread::yield();
        }
    };
    std::thread publisher([&] {
        for (unsigned round = 0; round < kPublishPinRounds; ++round) {
            auto busy = page.owner.load_state();
            busy.state = BUSY;
            busy.ref_cnt = 1;
            busy.dirty = 1;
            busy.hotness = static_cast<uint32_t>(round & 3u);
            page.owner.set_state(busy);
            phase_barrier(round * 2);
            assert(page.owner.hydra_publish_fetching());
            phase_barrier(round * 2 + 1);
            const auto settled = page.owner.load_state();
            assert(settled.state == FETCHING && settled.ref_cnt == 1);
            assert(settled.dirty == 1 && settled.hotness == (round & 3u));
            assert(page.pins.count == 0);
        }
    });
    std::thread pinner([&] {
        for (unsigned round = 0; round < kPublishPinRounds; ++round) {
            phase_barrier(round * 2);
            page.owner.pin();
            page.owner.unpin();
            phase_barrier(round * 2 + 1);
        }
    });
    publisher.join();
    pinner.join();
    auto stress_final = page.owner.load_state();
    assert(stress_final.state == FETCHING && stress_final.ref_cnt == 1);
    assert(page.pins.count == 0);

    // 1024 subobject pins must not wrap the runtime's 8-bit refcount.
    for (size_t i = 0; i < 1024; ++i) handles[0].pin();
    assert(page.pins.count == 1024 && page.owner.load_state().ref_cnt == 2);
    for (size_t i = 0; i < 1024; ++i) handles[1].unpin();
    assert(page.pins.count == 0 && page.owner.load_state().ref_cnt == 1);
    threads.clear();
    for (size_t i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            for (size_t j = 0; j < 10000; ++j) {
                handles[0].pin();
                handles[1].unpin();
            }
        });
    }
    for (auto &thread : threads) thread.join();
    assert(page.pins.count == 0 && page.owner.load_state().ref_cnt == 1);
    std::puts("HYDRA_PAGE_HANDLES_PASS offsets bytes shared-fetch move 1024-pins concurrent-pins");
}
