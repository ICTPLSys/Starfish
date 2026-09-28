#pragma once
#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <type_traits>

#include "utils/debug.hpp"
#include "uthreads.hpp"
#include "utils/stats.hpp"

namespace FarLib {
namespace uthread {

inline void fork_join_work_list(size_t work_count,
                                std::function<void(size_t)> &fn) {
    if (work_count == 1) {
        fn(0);
        return;
    }
    std::atomic_size_t next_work = 0;
    struct UThreadTask {
        Fibre fibre;
        std::atomic_size_t *next_work_ptr;
        std::atomic_size_t work_count;
        std::function<void(size_t)> *func;
        std::atomic_uint8_t state;
    };
    typedef void (*uthread_fn_t)(UThreadTask *);
    uthread_fn_t uthread_fn = [](UThreadTask *task) {
        task->state.store(1, std::memory_order_relaxed);
        profile::thread_start_work();
        while (true) {
            size_t work_idx =
                task->next_work_ptr->fetch_add(1, std::memory_order::relaxed);
            if (work_idx >= task->work_count) break;
            (*task->func)(work_idx);
        }
        profile::thread_end_work();
        task->state.store(2, std::memory_order_relaxed);
    };
    size_t thread_count = get_worker_count();
    std::unique_ptr<UThreadTask[]> tasks =
        std::make_unique<UThreadTask[]>(thread_count);
    for (size_t i = 0; i < thread_count; i++) {
        tasks[i].next_work_ptr = &next_work;
        tasks[i].work_count = work_count;
        tasks[i].func = &fn;
        tasks[i].state.store(0, std::memory_order_relaxed);
        tasks[i].fibre.run(uthread_fn, &tasks[i]);
    }
    bool suspended = profile::suspend_work();
    profile::start_fork_join();
    for (size_t i = 0; i < thread_count; i++) {
        tasks[i].fibre.join();
    }
    profile::end_fork_join();
    profile::resume_work(suspended);
}

template <bool HighPriority = false>
inline void fork_join(size_t thread_count, auto &&fn, const std::string &name = "fork_join") {
    if (thread_count == 1) {
        fn(0);
        return;
    }
    struct UThreadTask {
        Fibre fibre;
        size_t thread_id;
        std::remove_reference_t<decltype(fn)> *func;
        std::atomic_uint8_t state;
    };
    typedef void (*uthread_fn_t)(UThreadTask *);
    uthread_fn_t uthread_fn = [](UThreadTask *task) {
        task->state.store(1, std::memory_order_relaxed);
        profile::thread_start_work();
        (*task->func)(task->thread_id);
        profile::thread_end_work();
        task->state.store(2, std::memory_order_relaxed);
    };
    std::unique_ptr<UThreadTask[]> tasks =
        std::make_unique<UThreadTask[]>(thread_count);
    for (size_t i = 0; i < thread_count; i++) {
        tasks[i].thread_id = i;
        tasks[i].func = &fn;
        tasks[i].state.store(0, std::memory_order_relaxed);
        if constexpr (HighPriority) {
            tasks[i].fibre.setPriority(Fibre::TopPriority);
        }
        tasks[i].fibre.setName(name);
        tasks[i].fibre.run(uthread_fn, &tasks[i]);
    }
    bool suspended = profile::suspend_work();
    profile::start_fork_join();
    for (size_t i = 0; i < thread_count; i++) {
        tasks[i].fibre.join();
    }
    profile::end_fork_join();
    profile::resume_work(suspended);
}
}  // namespace uthread
}  // namespace FarLib
