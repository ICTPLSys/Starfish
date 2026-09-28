#include "runtime_ec_cpu.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using FarLib::profile::runtime_ec_cpu::Operation;
using FarLib::profile::runtime_ec_cpu::Scope;

static void burn() {
    uint64_t value = 0;
    for (uint64_t i = 0; i < 10000; ++i) value += i;
    asm volatile("" : "+r"(value) : : "memory");
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string mode(argv[1]);
    using namespace FarLib::profile::runtime_ec_cpu;
    if (mode == "disabled") {
        unsetenv("FARLIB_RUNTIME_EC_CPU");
        FarLib::profile::runtime_ec_cpu::begin_work("starfish");
        { Scope scope(Operation::Encode); burn(); }
        FarLib::profile::runtime_ec_cpu::end_work();
        assert(FarLib::runtime_metadata::measurement_aux_reader == nullptr);
        std::cout << "disabled passed\n";
        return 0;
    }
    if (mode == "invalid") {
        setenv("FARLIB_RUNTIME_EC_CPU", "invalid", 1);
        try {
            FarLib::profile::runtime_ec_cpu::begin_work("starfish");
            return 2;
        } catch (const std::runtime_error &) {
            std::cout << "invalid rejected\n";
            return 0;
        }
    }
    if (mode == "gate") {
        setenv("FARLIB_RUNTIME_EC_CPU", "1", 1);
        begin_work("starfish");
        const uint64_t stale = work_gate.load(std::memory_order_acquire);
        end_work();
        begin_work("starfish");
        // Deterministic reproduction: an entrant paused before CAS cannot
        // join the next Work even when both windows originally had count=0.
        assert(!try_enter(stale));
        assert((work_gate.load() & count_mask) == 0);
        std::atomic_bool entered{false}, release{false}, finished{false};
        std::thread worker([&] {
            { Scope scope(Operation::Encode);
              entered.store(true, std::memory_order_release);
              while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
              burn(); }
            finished.store(true, std::memory_order_release);
        });
        while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
        std::thread releaser([&] {
            while (!(work_gate.load(std::memory_order_acquire) & closed_bit))
                std::this_thread::yield();
            release.store(true, std::memory_order_release);
        });
        end_work();  // Must close admission AND wait for the admitted worker.
        worker.join();
        releaser.join();
        assert(finished.load() && (work_gate.load() & count_mask) == 0);
        assert(!try_enter(work_gate.load()));
        return 0;
    }
    if (mode == "stress") {
        setenv("FARLIB_RUNTIME_EC_CPU", "1", 1);
        std::atomic_bool stop{false};
        std::vector<std::thread> workers;
        for (unsigned i = 0; i < 4; ++i) workers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                Scope outer(Operation::Encode);
                Scope inner(Operation::Xor);
                burn();
            }
        });
        for (unsigned work = 0; work < 100; ++work) {
            begin_work("starfish");
            for (unsigned yield = 0; yield < 20; ++yield) std::this_thread::yield();
            end_work();
            assert((work_gate.load() & count_mask) == 0);
        }
        stop.store(true, std::memory_order_release);
        for (auto &worker : workers) worker.join();
        return 0;
    }
    assert(mode == "enabled");
    setenv("FARLIB_RUNTIME_EC_CPU", "1", 1);
    FarLib::runtime_metadata::Snapshot empty;
    FarLib::runtime_metadata::Reporter metadata("starfish", &empty, +[](void *) {
        return FarLib::runtime_metadata::Snapshot{};
    });
    begin_work("starfish");
    {
        Scope outer(Operation::Encode);
        {
            Scope nested(Operation::Update);
            burn();
        }
        std::thread worker([] {
            Scope scope(Operation::Xor);
            burn();
        });
        worker.join();
        burn();
    }
    end_work();
    FarLib::runtime_metadata::end_work();

    { Scope inactive(Operation::Xor); burn(); }
    begin_work("carbink");
    {
        Scope scope(Operation::Decode);
        burn();
    }
    end_work();
    FarLib::runtime_metadata::end_work();
    const auto expected_aux = sizeof(metadata) +
        sizeof(FarLib::runtime_metadata::observer_context) +
        sizeof(FarLib::runtime_metadata::work_end_observer) +
        measurement_aux_bytes(&registry());
    assert(metadata.last_snapshot().measurement_aux_bytes == expected_aux);
    return 0;
}
