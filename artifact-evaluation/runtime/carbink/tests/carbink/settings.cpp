#include "cache/carbink/settings.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main(int argc, char **argv) {
    using namespace FarLib::cache::carbink;
    if (argc == 2) {
        assert(max_updates_per_sec() == 50000);
        assert(inflight_update_limit() == 64);
        assert(drain_task_budget(4) == 7);
        assert(drain_task_budget(1) == std::numeric_limits<size_t>::max());
    } else {
        assert(max_updates_per_sec() == 0);
        assert(inflight_update_limit() == 128);
        assert(drain_task_budget(4) == std::numeric_limits<size_t>::max());
    }
    setenv("FARLIB_COMPACTION_INFLIGHT_UPDATES", "invalid", 1);
    assert(inflight_update_limit() == 128);
    setenv("FARLIB_COMPACTION_INFLIGHT_UPDATES", "0", 1);
    assert(inflight_update_limit() == 128);
    setenv("FARLIB_COMPACTION_DRAIN_TASK_BUDGET", "0", 1);
    assert(drain_task_budget(4) == std::numeric_limits<size_t>::max());
    std::puts("CARBINK_SETTINGS_PASS");
}
