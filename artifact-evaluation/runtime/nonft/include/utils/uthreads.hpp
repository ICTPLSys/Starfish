#pragma once
#include <sched.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "libfibre/fibre.h"
#include "utils/scope_diag.hpp"
#include "utils/stats.hpp"

namespace FarLib {

namespace uthread {

using UThread = Fibre;
using Mutex = FredMutex;
using Condition = FredCondition;
using UThreadLocal = FibreLocalVariables;
using Cluster = ::Cluster;
extern EventScope *es;

using YieldNoReadyProgress = void (*)();
inline std::atomic<YieldNoReadyProgress> yield_no_ready_progress{nullptr};

inline void install_yield_no_ready_progress(YieldNoReadyProgress progress) {
    if (progress == nullptr) std::abort();
    yield_no_ready_progress.store(progress, std::memory_order_release);
}

inline void remove_yield_no_ready_progress() {
    // Installation/removal is a quiescent lifecycle operation: no worker may
    // be inside yield() while the owning runtime is being destroyed.
    yield_no_ready_progress.store(nullptr, std::memory_order_release);
}

inline void runtime_init(size_t workers = 1) { es = FibreInit(1, workers); }

inline void runtime_destroy() {}

inline bool separate_background_cluster_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_SEPARATE_BACKGROUND_CLUSTER");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

// libfibre does not currently provide a safe Cluster teardown. A separately
// created cluster therefore has process lifetime and is intended for the
// runtime's single init/destroy cycle.
inline Cluster &create_cluster(size_t workers) {
    auto *cluster = new Cluster(1);
    cluster->addWorkers(workers);
    return *cluster;
}

inline size_t get_worker_count() {
    return ::Context::CurrCluster().getWorkerSysIDs(nullptr, 0);
}

inline size_t get_worker_sys_ids(Cluster &cluster, pthread_t *tids,
                                 size_t count) {
    return cluster.getWorkerSysIDs(tids, count);
}

inline size_t get_current_worker_sys_ids(pthread_t *tids, size_t count) {
    return ::Context::CurrCluster().getWorkerSysIDs(tids, count);
}
/* clang-format off */
inline size_t get_thread_count() { return 16; }
/* clang-format on */

inline std::unique_ptr<UThread> create(void (*func)()) {
    auto uthread = std::make_unique<UThread>();
    uthread->run(func);
    return uthread;
}

template <bool HighPriority = false, typename T>
inline std::unique_ptr<UThread> create(void (*func)(T *), T *arg,
                                       std::string name = "") {
    auto uthread = std::make_unique<UThread>();
    if constexpr (HighPriority) {
        uthread->setPriority(Fibre::TopPriority);
    }
    uthread->setName(name);
    uthread->run(func, arg);
    return uthread;
}

template <bool HighPriority = false, typename T>
inline std::unique_ptr<UThread> create_on(Cluster &cluster, void (*func)(T *),
                                          T *arg, std::string name = "") {
    auto uthread = std::make_unique<UThread>(cluster);
    if constexpr (HighPriority) {
        uthread->setPriority(Fibre::TopPriority);
    }
    uthread->setName(name);
    uthread->run(func, arg);
    return uthread;
}

inline void join(std::unique_ptr<UThread> uthread) { uthread.reset(); }

inline bool yield() {
    scope_diag::Guard scope_diag_guard(fibre_self(), scope_diag::Phase::YIELD);
    bool suspended = profile::suspend_work();
    profile::start_yield();
    const auto progress =
        yield_no_ready_progress.load(std::memory_order_acquire);
    bool yielded = Fibre::yieldGlobal();
    if (!yielded && progress != nullptr) {
        // The hook must be bounded and non-yielding. A false return means this
        // fibre did not switch or migrate and no local/stealable ready fibre
        // was found at that instant; it is not a global quiescence proof.
        progress();
    }
    profile::end_yield();
    profile::resume_work(suspended);
    return yielded;
}

inline void lock(Mutex *mutex) {
    scope_diag::Guard scope_diag_guard(fibre_self(), scope_diag::Phase::MUTEX);
    mutex->acquire();
}

inline bool try_lock(Mutex *mutex) {
    scope_diag::Guard scope_diag_guard(fibre_self(), scope_diag::Phase::MUTEX);
    return mutex->tryAcquire();
}

inline void unlock(Mutex *mutex) { mutex->release(); }

inline void wait_locked(Condition *cond, Mutex *mutex) {
    scope_diag::Guard scope_diag_guard(fibre_self(), scope_diag::Phase::COND);
    // printf("%p wait\n", fibre_self());
    cond->wait(*mutex);
    // printf("%p wake up\n", fibre_self());
}

inline void wait(Condition *cond, Mutex *mutex) {
    scope_diag::Guard scope_diag_guard(fibre_self(), scope_diag::Phase::COND);
    // printf("%p wait\n", fibre_self());
    mutex->acquire();
    cond->wait(*mutex);
    // printf("%p wake up\n", fibre_self());
}

template <class Pred>
inline void wait_for(Condition *cond, Mutex *mutex, Pred pred) {
    scope_diag::Guard scope_diag_guard(fibre_self(), scope_diag::Phase::COND);
    mutex->acquire();
    while (!pred()) {
        cond->wait(*mutex);
        mutex->acquire();
    }
    mutex->release();
}

inline void notify(Condition *cond, Mutex *mutex) {
    mutex->acquire();
    cond->signal();
    mutex->release();
}

inline void notify_all(Condition *cond, Mutex *mutex) {
    mutex->acquire();
    cond->signal<true>();
    mutex->release();
}

inline void notify_locked(Condition *cond) { cond->signal(); }

inline void notify_all_locked(Condition *cond) { cond->signal<true>(); }

inline UThreadLocal *get_tls() {
    return static_cast<UThreadLocal *>(fibre_self());
}

inline void set_default_priority() {
    fibre_self()->setPriority(Fibre::DefaultPriority);
}

inline void set_high_priority() {
    fibre_self()->setPriority(Fibre::TopPriority);
}

}  // namespace uthread

using uthread::UThread;

}  // namespace FarLib
