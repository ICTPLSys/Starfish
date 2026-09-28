#include "cache/entry_frequency.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

using Store = FarLib::cache::detail::EntryFrequencyStore;

int main() {
    int first = 0;
    int second = 0;
    const void *a = &first;
    const void *b = &second;

    Store::set_enabled(false);
    Store::register_entry(a);
    Store::add_window(a, 10);
    Store::update_ema(a, 10, 0);
    Store::publish(a, 10, 10);
    assert(Store::load_window(a) == 0);
    assert(Store::load_ema(a) == 0);
    assert(Store::consume_window(a) == 0);
    assert(!Store::try_add_window(a, 1));
    assert(Store::live_records() == 0);

    Store::set_enabled(true);
    Store::register_entry(a);
    Store::register_entry(b);
    assert(Store::live_records() == 2);

    Store::add_window(a, 5);
    Store::add_window(a, Store::CounterMax);
    assert(Store::load_window(a) == Store::CounterMax);
    assert(Store::consume_window(a) == Store::CounterMax);
    assert(Store::load_window(a) == 0);

    Store::update_ema(a, 100, 0);
    Store::update_ema(a, 20, 2);  // 100 - 25 + 20 = 95
    assert(Store::load_ema(a) == 95);
    Store::publish(a, 7, 95);
    assert(Store::load_published_window(a) == 7);
    assert(Store::load_published_ema(a) == 95);

    Store::copy(a, b);
    assert(Store::load_ema(b) == 95);
    assert(Store::load_published_window(b) == 7);
    assert(Store::load_published_ema(b) == 95);
    Store::erase(a);
    assert(Store::live_records() == 1);
    Store::copy(a, b);
    assert(Store::live_records() == 0);
    assert(Store::load_ema(b) == 0);

    Store::register_entry(a);
    constexpr size_t thread_count = 8;
    constexpr size_t increments = 5000;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        workers.emplace_back([a] {
            for (size_t j = 0; j < increments; ++j) Store::add_window(a, 1);
        });
    }
    for (auto &worker : workers) worker.join();
    assert(Store::load_window(a) == thread_count * increments);

    Store::add_window(a, Store::CounterMax);
    assert(Store::load_window(a) == Store::CounterMax);
    Store::set_enabled(false);
    assert(Store::live_records() == 0);
    assert(Store::load_window(a) == 0);
    std::cout << "ENTRY_FREQUENCY_PASS\n";
}
