#include "runtime_metadata_reporter.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace FarLib::runtime_metadata;

int main() {
    unsetenv("FARLIB_RUNTIME_METADATA");
    assert(!enabled_from_environment());
    setenv("FARLIB_RUNTIME_METADATA", "1", 1);
    assert(enabled_from_environment());
    setenv("FARLIB_RUNTIME_METADATA", "invalid", 1);
    bool invalid = false;
    try { (void)enabled_from_environment(); }
    catch (const std::runtime_error &) { invalid = true; }
    assert(invalid);
    Snapshot state;
    state.region_bytes = 10;
    state.group_bytes = 20;
    state.stripe_bytes = 30;
    state.mapping_bytes = 40;
    state.span_bytes = 50;
    state.measurement_aux_bytes = 900;
    assert(state.metadata_bytes() == 150);
    Snapshot aggregate;
    aggregate.add(state);
    aggregate.add(state);
    assert(aggregate.metadata_bytes() == 300);
    assert(aggregate.measurement_aux_bytes == 1800);
    std::ostringstream captured;
    auto *old = std::cout.rdbuf(captured.rdbuf());
    const auto reader = +[](void *context) {
        return *static_cast<Snapshot *>(context);
    };
    { Reporter unused("starfish", &state, reader); }
    assert(captured.str().empty());
    {
        Reporter reporter("starfish", &state, reader);
        end_work();
        state.region_bytes = 7;
        end_work();
        assert(reporter.boundary_count() == 2);
        assert(reporter.last_snapshot().metadata_bytes() == 147);
        // Simulate application cleanup; report final=147, not peak=150/zero.
        state = {};
        bool duplicate = false;
        try { Reporter nested("carbink", &state, reader); }
        catch (const std::runtime_error &) { duplicate = true; }
        assert(duplicate);
    }
    std::cout.rdbuf(old);
    assert(observer_context == nullptr && work_end_observer == nullptr);
    const auto line = captured.str();
    assert(line.find("metadata_bytes=147 ") != std::string::npos);
    assert(line.find("boundary_sequence=2 ") != std::string::npos);
    assert(line.find("metadata_bytes=0 ") == std::string::npos);
    assert(line.find('\n') == line.size() - 1);
    std::cout << line << "metadata fixture passed\n";
}
