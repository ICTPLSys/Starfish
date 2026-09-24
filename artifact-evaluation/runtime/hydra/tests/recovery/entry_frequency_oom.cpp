#include "cache/entry.hpp"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <new>

static std::atomic<bool> fail_next_allocation{false};
// Keep the replacement pair opaque to GCC's mismatched-allocation heuristic;
// both sides intentionally use malloc/free only in this fault-injection test.
[[gnu::noinline]] void *operator new(std::size_t size) {
    if (fail_next_allocation.exchange(false)) throw std::bad_alloc();
    if (void *p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void *p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void *p, std::size_t) noexcept { std::free(p); }

int main() {
    using FarLib::cache::FarObjectEntry;
    using Store = FarLib::cache::detail::EntryFrequencyStore;
    Store::set_enabled(true);
    {
        FarObjectEntry source, target;
        fail_next_allocation.store(true);
        source.reset<true>(reinterpret_cast<void *>(0x1000),
                          FarObjectEntry::RemoteAddrInvalid48, 528);
        assert(source.is_local());
        assert(source.local_addr() == reinterpret_cast<void *>(0x1000));
        assert(Store::allocation_failures() == 1);
        assert(Store::live_records() == 0);
        // A subsequent ordinary reset may resume profiling normally.
        source.reset<true>(reinterpret_cast<void *>(0x1000),
                          FarObjectEntry::RemoteAddrInvalid48, 528);
        source.add_window_frequency(19);
        assert(source.load_window_frequency() == 19);
        fail_next_allocation.store(true);
        // Matches the noexcept handle-move requirement for its metadata helper.
        auto copy = [&]() noexcept { target.copy_frequency_profile_from(source); };
        copy();
        assert(Store::allocation_failures() == 2);
        assert(source.load_window_frequency() == 19);
        assert(target.load_window_frequency() == 0);
        assert(Store::live_records() == 1);
        source.set_free();
    }
    assert(Store::live_records() == 0);
    Store::set_enabled(false);
    std::puts("ENTRY_FREQUENCY_OOM_PASS failures=2 records=0");
}
