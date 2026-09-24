#include "cache/entry.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
    using namespace FarLib::cache;
    // The annotation consumes an existing placement flag, not extra metadata.
static_assert(sizeof(FarObjectEntry) == 48);
    FarObjectEntry a, b;
    a.reset<true>(reinterpret_cast<void *>(0x1000),
                  FarObjectEntry::RemoteAddrInvalid48, 4080, true, true);
    assert(!a.is_recomputable());
    a.set_remote_backup_reservation(true);
    bool newly_marked = false;
    assert(a.try_mark_recomputable_before_eviction(&newly_marked));
    assert(newly_marked);
    assert(a.try_mark_recomputable_before_eviction(&newly_marked));
    assert(!newly_marked);
    assert(a.is_recomputable() && a.is_resident_local() &&
           a.has_remote_backup_reservation());
    a.set_resident_local(false);
    a.set_remote_backup_reservation(false);
    assert(a.is_recomputable());
    auto state = a.load_state();
    for (auto next : {MARKED, EVICTING, REMOTE, FETCHING, LOCAL}) {
        state.state = next;
        a.set_state(state);
        assert(a.is_recomputable());
    }
    b.reset<true>(reinterpret_cast<void *>(0x2000),
                  FarObjectEntry::RemoteAddrInvalid48, 4080);
    b.store_placement_flags(a.load_placement_flags());
    a.set_free();
    assert(!a.is_recomputable() && b.is_recomputable());
    b.reset<true>(reinterpret_cast<void *>(0x2000),
                  FarObjectEntry::RemoteAddrInvalid48, 4080);
    assert(!b.is_recomputable());
    for (auto next : {FREE, BUSY, EVICTING, REMOTE, FETCHING}) {
        state = b.load_state();
        state.state = next;
        b.set_state(state);
        assert(!b.try_mark_recomputable_before_eviction());
        assert(!b.is_recomputable());
    }
    b.reset<true>(reinterpret_cast<void *>(0x2000), 0x4000, 4080);
    assert(!b.try_mark_recomputable_before_eviction());
    assert(b.remote_addr() == 0x4000 && !b.load_state().invalid);
    b.reset<true>(reinterpret_cast<void *>(0x2000),
                  FarObjectEntry::RemoteAddrInvalid48, 4080);
    state = b.load_state(); state.state = MARKED; b.set_state(state);
    assert(b.try_mark_recomputable_before_eviction());
    assert(b.load_state().state == LOCAL && !b.load_state().invalid);
std::puts("RECOMPUTABLE_ENTRY_PASS size=48");
}
