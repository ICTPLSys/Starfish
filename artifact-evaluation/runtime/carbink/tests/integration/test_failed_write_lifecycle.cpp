// CPU bookkeeping regression for D2 pending-eviction handling on failed
// ordinary WRITEs. This intentionally does not instantiate ConcurrentCache or
// RDMA; it exercises the same thin consume/commit boundary used by the
// production completion path.

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "cache/eviction/eviction_commit.hpp"

namespace {

enum class State : uint8_t { EVICTING, LOCAL, REMOTE };

struct FakeEntry {
    State state = State::EVICTING;
    bool invalid = false;
    bool dirty = false;
    uint32_t refs = 1;
    uint8_t pending = 0;
    uint32_t commit_count = 0;
    bool committed_dirty = false;

    uint8_t take_six_pending_evict() {
        const uint8_t value = pending;
        pending = 0;
        return value;
    }

    void six_commit_evict(bool is_dirty) {
        ++commit_count;
        committed_dirty = is_dirty;
    }
};

void emulate_virtual_failed_write(FakeEntry &entry) {
    assert(entry.state == State::EVICTING);
    assert(entry.refs != 0);
    entry.invalid = true;
    --entry.refs;
    assert(entry.refs == 0);
    entry.state = State::REMOTE;
    const bool consumed = FarLib::design2::consume_six_pending_evict_once(
        [&entry] { return entry.take_six_pending_evict(); },
        [&entry](bool dirty) { entry.six_commit_evict(dirty); }, true);
    assert(consumed);
    entry.invalid = false;
}

void emulate_failed_write_to_local(FakeEntry &entry, bool had_remote_slot) {
    assert(entry.refs != 0);
    entry.invalid = true;
    --entry.refs;
    entry.state = State::LOCAL;
    entry.dirty = true;
    if (entry.refs == 0 && had_remote_slot) {
        const bool consumed = FarLib::design2::discard_six_pending_evict_once(
            [&entry] { return entry.take_six_pending_evict(); });
        assert(consumed);
    } else if (entry.refs == 0) {
        const bool consumed = FarLib::design2::discard_six_pending_evict_once(
            [&entry] { return entry.take_six_pending_evict(); });
        assert(consumed);
    }
    entry.invalid = false;
}

int run() {
    // Recipe-backed dead remote: commit the logical eviction exactly once.
    {
        FakeEntry entry;
        entry.pending = 2;  // dirty eviction
        emulate_virtual_failed_write(entry);
        assert(entry.state == State::REMOTE);
        assert(entry.pending == 0);
        assert(entry.commit_count == 1);
        assert(entry.committed_dirty);
        assert(!FarLib::design2::consume_six_pending_evict_once(
            [&entry] { return entry.take_six_pending_evict(); },
            [&entry](bool dirty) { entry.six_commit_evict(dirty); }, true));
        assert(entry.commit_count == 1);
    }

    // Two write references: the first failure leaves pending ownership for the
    // final completion; only the final failure discards it.
    {
        FakeEntry entry;
        entry.refs = 2;
        entry.pending = 1;
        emulate_failed_write_to_local(entry, true);
        assert(entry.refs == 1);
        assert(entry.pending == 1);
        assert(entry.commit_count == 0);
        emulate_failed_write_to_local(entry, true);
        assert(entry.refs == 0);
        assert(entry.pending == 0);
        assert(entry.commit_count == 0);
    }

    // Flag-only recomputable state has no callback. A failed write remains a
    // dirty LOCAL retry and must never be reported as a virtual REMOTE commit.
    {
        FakeEntry entry;
        entry.pending = 2;
        emulate_failed_write_to_local(entry, true);
        assert(entry.state == State::LOCAL);
        assert(entry.dirty);
        assert(entry.pending == 0);
        assert(entry.commit_count == 0);
    }

    std::puts("FAILED_WRITE_DESIGN2_PASS");
    return 0;
}

}  // namespace

int main() { return run(); }
