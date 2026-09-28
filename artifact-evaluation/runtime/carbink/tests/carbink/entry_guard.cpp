// CPU-only lifecycle checks for the Carbink compaction EntryGuard.
//
// The tail HydraPagePins storage is placed immediately after FarObjectEntry,
// matching the production page-owner allocation.  These tests intentionally
// interleave pin/unpin with BUSY publication without starting RDMA or fibres.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/carbink/entry_guard.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

using FarLib::cache::BUSY;
using FarLib::cache::EntryStateBits;
using FarLib::cache::FETCHING;
using FarLib::cache::FarObjectEntry;
using FarLib::cache::HydraPagePins;
using FarLib::cache::LOCAL;
using FarLib::cache::REMOTE;
using FarLib::cache::carbink::EntryGuard;

struct PageStorage {
    FarObjectEntry entry;
    HydraPagePins pins;
};

static_assert(offsetof(PageStorage, pins) == sizeof(FarObjectEntry));

constexpr uint64_t kSourceAddr = 0x12345000;
constexpr uint64_t kDestinationAddr = 0x23456000;

void initialize_remote(PageStorage *page, uint64_t remote_addr = kSourceAddr) {
    page->entry.set_remote_addr(remote_addr);
    EntryStateBits state{};
    state.state = REMOTE;
    state.size = 8192;
    page->entry.set_state(state);
    page->entry.hydra_mark_page();
}

template <typename Mutator>
void update_state(FarObjectEntry *entry, Mutator mutator) {
    auto observed = entry->load_state();
    for (;;) {
        auto desired = observed;
        mutator(desired);
        if (entry->cas_state_weak(observed, desired)) return;
    }
}

void test_stale_expected_address_does_not_leak_reference() {
    PageStorage page;
    initialize_remote(&page, kSourceAddr);
    EntryGuard guard;

    assert(!guard.acquire(&page.entry, kSourceAddr + 1));
    assert(guard.entry() == nullptr);
    const auto state = page.entry.load_state();
    assert(state.state == REMOTE);
    assert(state.ref_cnt == 0);
    assert(page.pins.count == 0);
}

void test_publish_preserves_busy_pin_dirty_and_hot() {
    PageStorage page;
    initialize_remote(&page);
    EntryGuard guard;
    assert(guard.acquire(&page.entry, kSourceAddr));
    assert(guard.try_lock_publish());

    // A Hydra page pin is allowed to add/remove an aggregate reference while
    // the guard owns BUSY.  Exercise both directions before publication, then
    // keep one extra pin alive across publish.
    page.entry.pin();
    page.entry.unpin();
    page.entry.pin();
    page.entry.mark_dirty();
    update_state(&page.entry, [](EntryStateBits &state) {
        state.hotness = EntryStateBits::HOTNESS_MAX;
    });

    auto busy = page.entry.load_state();
    assert(busy.state == BUSY);
    assert(busy.ref_cnt == 2);
    assert(busy.dirty != 0);
    assert(busy.hotness == EntryStateBits::HOTNESS_MAX);

    guard.publish_locked(kDestinationAddr);
    auto remote = page.entry.load_state();
    assert(remote.state == REMOTE);
    assert(remote.ref_cnt == 2);
    assert(remote.dirty != 0);
    assert(remote.hotness == EntryStateBits::HOTNESS_MAX);
    assert(page.entry.remote_addr() == kDestinationAddr);

    page.entry.unpin();
    assert(page.entry.load_state().ref_cnt == 1);
    guard.release();
    remote = page.entry.load_state();
    assert(remote.state == REMOTE);
    assert(remote.ref_cnt == 0);
    assert(page.pins.count == 0);
}

void test_cancel_preserves_latest_state() {
    PageStorage page;
    initialize_remote(&page);
    EntryGuard guard;
    assert(guard.acquire(&page.entry, kSourceAddr));
    assert(guard.try_lock_publish());

    page.entry.pin();
    page.entry.mark_dirty();
    update_state(&page.entry, [](EntryStateBits &state) {
        state.hotness = EntryStateBits::HOTNESS_MAX;
    });
    guard.cancel_publish();

    const auto remote = page.entry.load_state();
    assert(remote.state == REMOTE);
    assert(remote.ref_cnt == 2);
    assert(remote.dirty != 0);
    assert(remote.hotness == EntryStateBits::HOTNESS_MAX);
    assert(page.entry.remote_addr() == kSourceAddr);

    page.entry.unpin();
    guard.release();
    assert(page.entry.load_state().state == REMOTE);
    assert(page.entry.load_state().ref_cnt == 0);
    assert(page.pins.count == 0);
}

void test_fetch_transition_then_guard_release() {
    PageStorage page;
    initialize_remote(&page);
    EntryGuard guard;
    assert(guard.acquire(&page.entry, kSourceAddr));

    update_state(&page.entry, [](EntryStateBits &state) {
        state.state = FETCHING;
    });
    assert(!guard.try_lock_publish());

    // The foreground fetch has now installed its local buffer and published
    // LOCAL.  Releasing the compaction guard must only drop its own reference.
    page.entry.set_local_addr(reinterpret_cast<void *>(0x70000000));
    update_state(&page.entry, [](EntryStateBits &state) {
        state.state = LOCAL;
    });
    guard.release();

    const auto local = page.entry.load_state();
    assert(local.state == LOCAL);
    assert(local.ref_cnt == 0);
    assert(guard.entry() == nullptr);
    assert(page.pins.count == 0);
}

}  // namespace

int main() {
    test_stale_expected_address_does_not_leak_reference();
    test_publish_preserves_busy_pin_dirty_and_hot();
    test_cancel_preserves_latest_state();
    test_fetch_transition_then_guard_release();
    std::puts("CARBINK_ENTRY_GUARD_PASS");
    return 0;
}
