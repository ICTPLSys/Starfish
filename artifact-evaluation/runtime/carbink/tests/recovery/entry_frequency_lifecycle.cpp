#include "cache/entry.hpp"
#include <cassert>
#include <cstdio>

int main() {
    using FarLib::cache::FarObjectEntry;
    using Store = FarLib::cache::detail::EntryFrequencyStore;
static_assert(sizeof(FarObjectEntry) == 48);
    Store::set_enabled(false);
    {
        FarObjectEntry entry;
        entry.reset<true>(reinterpret_cast<void *>(0x1000),
                         FarObjectEntry::RemoteAddrInvalid48, 528);
        entry.add_window_frequency(7);
        assert(entry.load_window_frequency() == 0);
        assert(Store::live_records() == 0);
    }
    Store::set_enabled(true);
    {
        FarObjectEntry from, to;
        assert(Store::live_records() == 0);
        from.reset<true>(reinterpret_cast<void *>(0x1000),
                        FarObjectEntry::RemoteAddrInvalid48, 528);
        assert(Store::live_records() == 1);
        from.add_window_frequency(17);
        from.update_ema_frequency(from.consume_window_frequency(), 3);
        from.publish_frequency_profile(17, from.load_ema_frequency());
        assert(from.load_published_window_frequency() == 17);
        assert(from.load_published_ema_frequency() == 17);
        to.copy_frequency_profile_from(from);
        from.set_free();
        assert(Store::live_records() == 1);
        assert(to.load_ema_frequency() == 17);
        // Late diagnostic updates must not recreate a freed entry's record.
        from.add_window_frequency(99);
        from.publish_frequency_profile(99, 99);
        assert(Store::live_records() == 1);
        from.reset<true>(reinterpret_cast<void *>(0x2000),
                        FarObjectEntry::RemoteAddrInvalid48, 528);
        assert(from.load_window_frequency() == 0);
        assert(from.load_ema_frequency() == 0);
        assert(Store::live_records() == 2);
    }
    assert(Store::live_records() == 0);
    Store::set_enabled(false);
std::puts("ENTRY_FREQUENCY_LIFECYCLE_PASS size=48 records=0");
}
