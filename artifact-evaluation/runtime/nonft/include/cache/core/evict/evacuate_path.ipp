#pragma once
#include "cache/concurrent_cache.hpp"
#include "utils/scope_diag.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>

namespace FarLib::cache {

// Experimental single-variable restoration. Inclusive never takes this path.
inline bool exclusive_owned_batch_reclaim_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_EXCLUSIVE_OWNED_BATCH");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled && ::FarLib::get_config().exclusive_cache;
}

inline bool legacy_scan_cursors_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_LEGACY_SCAN_CURSORS");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool inclusive_single_wait_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_INCLUSIVE_SINGLE_WAIT");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool inclusive_legacy_eligibility_enabled() {
    static const bool enabled = [] {
        const char *value =
            std::getenv("FARLIB_INCLUSIVE_LEGACY_ELIGIBILITY");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool inclusive_legacy_cursors_enabled() {
    static const bool enabled = [] {
        const char *value =
            std::getenv("FARLIB_INCLUSIVE_LEGACY_CURSORS");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool inclusive_mark_stop_evict_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_INCLUSIVE_MARK_STOP_EVICT");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool inclusive_gc_processed_filter_enabled() {
    static const bool enabled = [] {
        const char *value =
            std::getenv("FARLIB_INCLUSIVE_GC_PROCESSED_FILTER");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool inclusive_evict_stop_mark_enabled() {
    static const bool enabled = [] {
        const char *value =
            std::getenv("FARLIB_INCLUSIVE_EVICT_STOP_MARK");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline size_t inclusive_mark_round_regions() {
    static const size_t limit = [] {
        const char *value =
            std::getenv("FARLIB_INCLUSIVE_MARK_ROUND_REGIONS");
        if (value == nullptr || *value == '\0') return size_t{0};
        size_t parsed = 0;
        for (const char *p = value; *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') return size_t{0};
            const size_t digit = static_cast<size_t>(*p - '0');
            if (parsed > (std::numeric_limits<size_t>::max() - digit) / 10) {
                return size_t{0};
            }
            parsed = parsed * 10 + digit;
        }
        return parsed;
    }();
    return limit;
}

inline bool inclusive_mark_one_batch_per_worker_enabled() {
    static const bool enabled = [] {
        const char *value =
            std::getenv("FARLIB_INCLUSIVE_MARK_ONE_BATCH_PER_WORKER");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool inclusive_resume_scan_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("FARLIB_INCLUSIVE_RESUME_SCAN");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline size_t inclusive_scan_visit_limit() {
    static const size_t limit = [] {
        const char *value = std::getenv("FARLIB_INCLUSIVE_SCAN_VISITS");
        if (value == nullptr || *value == '\0') return size_t{4096};
        size_t parsed = 0;
        for (const char *p = value; *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') return size_t{4096};
            const size_t digit = static_cast<size_t>(*p - '0');
            if (parsed > (std::numeric_limits<size_t>::max() - digit) / 10) {
                return size_t{4096};
            }
            parsed = parsed * 10 + digit;
        }
        return parsed == 0 ? size_t{4096} : parsed;
    }();
    return limit;
}

inline size_t ConcurrentArrayCache::get_mark_worker_count() const {
        if (!::FarLib::get_config().optimized_evacuator) return 0;
        size_t configured_mark_workers = ::FarLib::get_config().mark_thread_cnt;
        if (configured_mark_workers == 0) return 0;
        if (configured_mark_workers >= evacuate_thread_cnt) {
            return evacuate_thread_cnt;
        }
        return configured_mark_workers;
    }

inline size_t ConcurrentArrayCache::get_evict_worker_count(
    size_t mark_workers) const {
        return evacuate_thread_cnt - mark_workers;
    }

inline ConcurrentArrayCache::FrequencyMarkPassContext
ConcurrentArrayCache::begin_frequency_mark_pass() {
        FrequencyMarkPassContext context;
        context.ema_mark_interval =
            ::FarLib::get_config().hybrid_profile_frequency_ema_mark_interval;
        if (!hybrid_profiling_enabled()) {
            context.update_ema_this_pass = false;
            return context;
        }
        context.mark_pass_ordinal =
            hybrid_profile_mark_pass_counter.fetch_add(
                1, std::memory_order_relaxed) +
            1;
        context.update_ema_this_pass =
            context.ema_mark_interval <= 1 ||
            (context.mark_pass_ordinal % context.ema_mark_interval) == 0;
        return context;
    }

inline void ConcurrentArrayCache::run_streaming_mark_master(
    StreamingMarkMasterArgs *args) {
        if (args->worker_count == 0) return;

        args->cache->log_cache_progress(
            "streaming_mark_begin", kAnyAllocBin, args->timestamp,
            args->worker_count);
        auto mark_start = get_cycles();
        if (args->ready_tasks_out != nullptr) {
            std::vector<std::vector<::FarLib::allocator::EvictTask>>
                worker_ready(args->worker_count);
            std::function<void(size_t)> fn_mark =
                [cache = args->cache, timestamp = args->timestamp,
                 frequency_mark_pass = args->frequency_mark_pass,
                 ready_task_budget = args->ready_task_budget,
                 legacy_region_budget = args->legacy_region_budget,
                 legacy_region_budget_per_worker =
                     args->legacy_region_budget_per_worker,
                 mark_stop_after_evict = args->mark_stop_after_evict,
                 &worker_ready](size_t worker_id) {
                    cache->log_cache_progress(
                        "mark_worker_begin", kAnyAllocBin, timestamp,
                        worker_id);
                    std::atomic_size_t worker_region_budget{64};
                    auto *region_budget = legacy_region_budget_per_worker
                                              ? &worker_region_budget
                                              : legacy_region_budget;
                    cache->mark_phase(timestamp, frequency_mark_pass,
                                      &worker_ready[worker_id],
                                      ready_task_budget, worker_id,
                                      region_budget,
                                      mark_stop_after_evict);
                    if (legacy_region_budget_per_worker &&
                        legacy_region_budget != nullptr) {
                        legacy_region_budget->fetch_sub(
                            64 - worker_region_budget.load(
                                     std::memory_order_relaxed),
                            std::memory_order_relaxed);
                    }
                    cache->log_cache_progress(
                        "mark_worker_end", kAnyAllocBin, timestamp,
                        worker_id);
                };
            uthread::fork_join<true>(args->worker_count, fn_mark,
                                     "streaming_mark");
            args->cache->log_cache_progress(
                "mark_scan_end", kAnyAllocBin, args->timestamp,
                args->worker_count);
            args->ready_tasks_out->clear();
            size_t total_ready = 0;
            for (auto &batch : worker_ready) {
                total_ready += batch.size();
            }
            args->ready_tasks_out->reserve(total_ready);
            for (auto &batch : worker_ready) {
                args->ready_tasks_out->insert(args->ready_tasks_out->end(),
                                              std::make_move_iterator(
                                                  batch.begin()),
                                              std::make_move_iterator(
                                                  batch.end()));
            }
        } else {
            std::function<void(size_t)> fn_mark =
                [cache = args->cache, timestamp = args->timestamp,
                 frequency_mark_pass = args->frequency_mark_pass,
                 legacy_region_budget =
                     args->legacy_region_budget,
                 legacy_region_budget_per_worker =
                     args->legacy_region_budget_per_worker,
                 mark_stop_after_evict =
                     args->mark_stop_after_evict](size_t worker_id) {
                    cache->log_cache_progress(
                        "mark_worker_begin", kAnyAllocBin, timestamp,
                        worker_id);
                    std::atomic_size_t worker_region_budget{64};
                    auto *region_budget = legacy_region_budget_per_worker
                                              ? &worker_region_budget
                                              : legacy_region_budget;
                    cache->mark_phase(timestamp, frequency_mark_pass, nullptr,
                                      nullptr, worker_id,
                                      region_budget,
                                      mark_stop_after_evict);
                    if (legacy_region_budget_per_worker &&
                        legacy_region_budget != nullptr) {
                        legacy_region_budget->fetch_sub(
                            64 - worker_region_budget.load(
                                     std::memory_order_relaxed),
                            std::memory_order_relaxed);
                    }
                    cache->log_cache_progress(
                        "mark_worker_end", kAnyAllocBin, timestamp,
                        worker_id);
            };
            uthread::fork_join<true>(args->worker_count, fn_mark,
                                     "streaming_mark");
            args->cache->log_cache_progress(
                "mark_scan_end", kAnyAllocBin, args->timestamp,
                args->worker_count);
        }
        profile::get_tlpd().evac_mark_phase_cycles +=
            (int64_t)(get_cycles() - mark_start);

        if (args->flip_after_mark) {
            auto flip_start_ph = get_cycles();
            auto flip_scope_start = profile::start_flip_scope();
            args->cache->flip_scope_state(args->timestamp);
            profile::end_flip_scope(flip_scope_start);
            profile::get_tlpd().evac_flip_scope_cycles +=
                (int64_t)(get_cycles() - flip_start_ph);
        }
        args->cache->log_cache_progress(
            "streaming_mark_end", kAnyAllocBin, args->timestamp,
            args->worker_count);
        if (args->evict_stop_after_mark != nullptr) {
            args->cache->log_cache_progress(
                "mark_evict_stop", kAnyAllocBin, args->timestamp);
            args->evict_stop_after_mark->store(true,
                                                std::memory_order_release);
        }
    }

inline void ConcurrentArrayCache::run_streaming_evict_master(
    StreamingEvictMasterArgs *args) {
        if (args->worker_count == 0) return;

        args->cache->log_cache_progress(
            "streaming_evict_begin", kAnyAllocBin, args->evict_timestamp,
            args->worker_count);
        const uint32_t required_safe_epoch = args->safe_epoch_to_wait;
        if (args->cache->safe_epoch.load(std::memory_order_acquire) <
            required_safe_epoch) {
            auto wait_start = get_cycles();
            while (args->cache->safe_epoch.load(std::memory_order_acquire) <
                   required_safe_epoch) {
                uthread::yield();
            }
            auto wait_cycles = (int64_t)(get_cycles() - wait_start);
            profile::get_tlpd().evac_streaming_wait_safe_epoch_cycles +=
                wait_cycles;
            profile::get_tlpd().evac_pipeline_idle_cycles += wait_cycles;
        }

        auto evict_start = get_cycles();
        if (args->ready_tasks_in != nullptr) {
            std::vector<std::vector<::FarLib::allocator::EvictTask>>
                worker_deferred(args->worker_count);
            std::atomic_size_t next_task_idx{0};
            std::function<void(size_t)> fn_evict =
                [cache = args->cache, ready_tasks = args->ready_tasks_in,
                 current_safe_epoch = required_safe_epoch,
                 posted_wrs_total = args->posted_wrs_total,
                 &next_task_idx, &worker_deferred](size_t worker_id) {
                    cache->evict_ready_worker_logic(
                        *ready_tasks, next_task_idx, current_safe_epoch,
                        worker_deferred[worker_id], *posted_wrs_total);
            };
            uthread::fork_join<true>(args->worker_count, fn_evict,
                                     "streaming_evict");
            if (args->deferred_tasks_out != nullptr) {
                args->deferred_tasks_out->clear();
                size_t total_deferred = 0;
                for (auto &batch : worker_deferred) {
                    total_deferred += batch.size();
                }
                args->deferred_tasks_out->reserve(total_deferred);
                for (auto &batch : worker_deferred) {
                    args->deferred_tasks_out->insert(
                        args->deferred_tasks_out->end(),
                        std::make_move_iterator(batch.begin()),
                        std::make_move_iterator(batch.end()));
                }
            }
        } else {
            const bool ignore_safe_epoch =
                args->safe_epoch_to_wait == 0 &&
                args->ready_tasks_in == nullptr &&
                optimized_legacy_exclusive_pipeline_enabled();
            const auto eligibility =
                ignore_safe_epoch && inclusive_legacy_eligibility_enabled() &&
                        !::FarLib::get_config().exclusive_cache
                    ? ::FarLib::allocator::
                          EvacuationEligibility::LegacyConcurrent
                    : ::FarLib::allocator::EvacuationEligibility::All;
            const bool owned_reclaim =
                ignore_safe_epoch && exclusive_owned_batch_reclaim_enabled();
            std::function<void(size_t)> fn_evict =
                [cache = args->cache, timestamp = args->evict_timestamp,
                 gc_timestamp = args->gc_timestamp,
                 posted_wrs_total = args->posted_wrs_total,
                 ignore_safe_epoch, eligibility, owned_reclaim,
                 resume_scan = args->resume_scan,
                 evict_stop_after_mark =
                     args->evict_stop_after_mark,
                 remaining_evict_post_workers =
                     args->remaining_evict_post_workers,
                 mark_stop_after_evict =
                     args->mark_stop_after_evict,
                 worker_count = args->worker_count](size_t worker_id) {
                    auto post_start = get_cycles();
                    cache->evict_post_worker_logic(timestamp,
                                                   *posted_wrs_total,
                                                   ignore_safe_epoch,
                                                   cache->get_mark_worker_count() +
                                                   worker_id,
                                                   eligibility,
                                                   resume_scan,
                                                   evict_stop_after_mark);
                    if (!owned_reclaim) {
                        profile::get_tlpd().evac_evict_post_cycles +=
                            (int64_t)(get_cycles() - post_start);
                    }
                    cache->log_cache_progress(
                        "evict_post_worker_end", kAnyAllocBin, timestamp,
                        worker_id);
                    if (remaining_evict_post_workers != nullptr &&
                        mark_stop_after_evict != nullptr &&
                        remaining_evict_post_workers->fetch_sub(
                            1, std::memory_order_acq_rel) == 1) {
                        cache->log_cache_progress(
                            "evict_mark_stop", kAnyAllocBin, timestamp,
                            worker_count);
                        mark_stop_after_evict->store(
                            true, std::memory_order_release);
                    }
                    if (!owned_reclaim) {
                        cache->evict_drain_phase();
                        cache->gc_phase(gc_timestamp, eligibility,
                                        resume_scan);
                    }
            };
            uthread::fork_join<true>(args->worker_count, fn_evict,
                                     "streaming_evict");
        }

        auto elapsed = (int64_t)(get_cycles() - evict_start);
        int64_t total_wrs = args->posted_wrs_total->load(std::memory_order_relaxed);
        profile::get_tlpd().evac_evict_rounds++;
        if (total_wrs > 0) {
            profile::get_tlpd().evac_evict_active_rounds++;
            profile::get_tlpd().evac_evict_active_cycles += elapsed;
            profile::get_tlpd().evac_evict_active_wrs += total_wrs;
        } else {
            profile::get_tlpd().evac_evict_empty_rounds++;
            profile::get_tlpd().evac_evict_empty_cycles += elapsed;
        }
        profile::get_tlpd().evac_evict_phase_cycles += elapsed;
        args->cache->log_cache_progress(
            "streaming_evict_end", kAnyAllocBin, args->evict_timestamp,
            static_cast<uint64_t>(total_wrs));
        args->cache->log_cache_progress(
            "tail_evict_end_free", kAnyAllocBin, args->evict_timestamp,
            ::FarLib::allocator::global_heap.get_free_size());
    }

inline void ConcurrentArrayCache::wait_for_eviction_trigger(
    uint64_t handled_eviction_request_seq) {
        log_cache_progress("evac_wait_trigger_enter");
        uthread::lock(&eviction_mutex);
        mutator_can_not_allocate.store(false);
        uthread::notify_all_locked(&mutator_cond);
        while (working.load(std::memory_order_acquire) &&
               !::FarLib::allocator::global_heap.has_async_full_returns() &&
               !::FarLib::allocator::global_heap.evacuator_should_start() &&
               !mutator_waiters_need_progress() &&
               eviction_request_seq.load(std::memory_order_acquire) ==
                   handled_eviction_request_seq) {
            log_cache_progress("evac_wait_trigger_sleep", kAnyAllocBin,
                               handled_eviction_request_seq);
            log_cache_progress("tail_sleep_state", kAnyAllocBin,
                               ::FarLib::allocator::global_heap.get_free_size(),
                               ::FarLib::allocator::global_heap.get_memory_low_water_mark());
            uthread::wait_locked(&eviction_cond, &eviction_mutex);
            // FredCondition::wait returns without keeping the mutex acquired.
            // Re-acquire it before checking the predicate again.
            uthread::lock(&eviction_mutex);
            log_cache_progress("tail_wake_state", kAnyAllocBin,
                               ::FarLib::allocator::global_heap.get_free_size(),
                               ::FarLib::allocator::global_heap.get_memory_low_water_mark());
        }
        uthread::unlock(&eviction_mutex);
        log_cache_progress("evac_wait_trigger_exit", kAnyAllocBin,
                           handled_eviction_request_seq);
    }

inline void ConcurrentArrayCache::notify_mutators_ready() {
        if (!mutators_can_resume()) {
            return;
        }
        log_cache_progress("notify_mutators_ready");
        uthread::lock(&eviction_mutex);
        mutator_can_not_allocate.store(false);
        log_cache_progress("notify_locked");
        uthread::notify_all_locked(&mutator_cond);
        uthread::unlock(&eviction_mutex);
    }

inline void ConcurrentArrayCache::evacuate_work() {
        uint32_t timestamp = 0;
        uint32_t pipeline_safe_mark_timestamp = 0;
        uint32_t pipeline_evict_timestamp = 0;
        uint32_t pipeline_gc_timestamp = 0;
        bool pipeline_has_safe_mark = false;
        std::vector<::FarLib::allocator::EvictTask> pipeline_ready_tasks;
        std::vector<std::vector<::FarLib::allocator::EvictTask>>
            ready_pipeline_worker_ready_tasks;
        const bool optimized_enabled =
            ::FarLib::get_config().optimized_evacuator && evacuate_thread_cnt > 1;
        const size_t configured_mark_workers = get_mark_worker_count();
        const size_t evict_workers = get_evict_worker_count(configured_mark_workers);
        const size_t mark_workers = configured_mark_workers;
        std::cerr << "runtime.exclusive_owned_batch="
                  << (optimized_enabled &&
                      optimized_legacy_exclusive_pipeline_enabled() &&
                      exclusive_owned_batch_reclaim_enabled())
                  << " mark_workers=" << mark_workers
                  << " evict_workers=" << evict_workers << std::endl;
        uint64_t handled_eviction_request_seq = 0;
        auto requeue_pending_pipeline_tasks = [&] {
            const size_t pending = pipeline_ready_tasks.size();
            for (auto &task : pipeline_ready_tasks) {
                ::FarLib::allocator::global_heap.requeue_evict_task(task);
            }
            pipeline_ready_tasks.clear();
            for (auto &worker_tasks : ready_pipeline_worker_ready_tasks) {
                for (auto &task : worker_tasks) {
                    ::FarLib::allocator::global_heap.requeue_evict_task(task);
                }
                worker_tasks.clear();
            }
            std::cerr << "shutdown.pipeline_requeue tasks=" << pending
                      << std::endl;
        };
        while (true) {
            if (!working.load()) {
                requeue_pending_pipeline_tasks();
                return;
            }
            // Publish ingress without forcing an otherwise unnecessary round.
            ::FarLib::allocator::global_heap.drain_async_full_returns(512);
            const uint64_t pending_eviction_request_seq =
                eviction_request_seq.load(std::memory_order_acquire);
            const bool explicit_eviction_requested =
                pending_eviction_request_seq != handled_eviction_request_seq;

            if (optimized_enabled) {
                if (optimized_legacy_exclusive_pipeline_enabled()) {
                    if (!::FarLib::allocator::global_heap.evacuator_should_start() &&
                        !mutator_waiters_need_progress() &&
                        !explicit_eviction_requested) {
                        wait_for_eviction_trigger(
                            handled_eviction_request_seq);
                        continue;
                    }

                    auto evac_master_start_cycles = get_cycles();
                    profile::count_evacuation();
                    log_cache_progress("evac_round_begin_legacy_opt",
                                       kAnyAllocBin, timestamp);

                    timestamp++;
                    std::atomic_int64_t posted_wrs_total{0};
                    const bool mark_one_batch_per_worker =
                        !::FarLib::get_config().exclusive_cache &&
                        legacy_scan_cursors_enabled() &&
                        inclusive_legacy_eligibility_enabled() &&
                        inclusive_legacy_cursors_enabled() &&
                        inclusive_mark_one_batch_per_worker_enabled();
                    const size_t mark_round_limit =
                        mark_one_batch_per_worker
                            ? size_t{64} * mark_workers
                            : !::FarLib::get_config().exclusive_cache &&
                                legacy_scan_cursors_enabled() &&
                                inclusive_legacy_eligibility_enabled() &&
                                inclusive_legacy_cursors_enabled()
                            ? inclusive_mark_round_regions()
                            : 0;
                    std::atomic_size_t mark_region_budget{mark_round_limit};
                    const bool resume_scan =
                        mark_round_limit != 0 &&
                        !::FarLib::get_config().exclusive_cache &&
                        legacy_scan_cursors_enabled() &&
                        inclusive_legacy_eligibility_enabled() &&
                        inclusive_legacy_cursors_enabled() &&
                        inclusive_resume_scan_enabled();
                    const size_t scan_visit_limit =
                        resume_scan ? inclusive_scan_visit_limit() : 0;
                    std::atomic_size_t evict_visit_budget{scan_visit_limit};
                    std::atomic_size_t gc_visit_budget{scan_visit_limit};
                    std::atomic_bool evict_pass_complete{false};
                    std::atomic_bool gc_pass_complete{false};
                    std::atomic_bool evict_stop_after_mark{false};
                    std::atomic_bool mark_stop_after_evict{false};
                    std::atomic_size_t remaining_evict_post_workers{
                        evict_workers};
                    StreamingEvictMasterArgs::ResumeScanRound scan_round{
                        inclusive_evict_scan_generation,
                        inclusive_gc_scan_generation,
                        &evict_visit_budget,
                        &gc_visit_budget,
                        &evict_pass_complete,
                        &gc_pass_complete};
                    StreamingMarkMasterArgs mark_args{
                        this, timestamp, mark_workers, begin_frequency_mark_pass(),
                        true, nullptr, nullptr};
                    mark_args.legacy_region_budget =
                        mark_round_limit == 0 ? nullptr : &mark_region_budget;
                    mark_args.legacy_region_budget_per_worker =
                        mark_one_batch_per_worker;
                    StreamingEvictMasterArgs evict_args{
                        this, 0, timestamp, timestamp, evict_workers,
                        &posted_wrs_total, nullptr, nullptr};
                    evict_args.resume_scan = resume_scan ? &scan_round : nullptr;
                    const bool stop_evict_after_mark =
                        !::FarLib::get_config().exclusive_cache &&
                        inclusive_legacy_eligibility_enabled() &&
                        inclusive_mark_stop_evict_enabled();
                    if (stop_evict_after_mark) {
                        mark_args.evict_stop_after_mark =
                            &evict_stop_after_mark;
                        evict_args.evict_stop_after_mark =
                            &evict_stop_after_mark;
                    }
                    const bool stop_mark_after_evict =
                        evict_workers != 0 && mark_round_limit != 0 &&
                        resume_scan &&
                        !::FarLib::get_config().exclusive_cache &&
                        inclusive_legacy_eligibility_enabled() &&
                        inclusive_legacy_cursors_enabled() &&
                        inclusive_evict_stop_mark_enabled();
                    if (stop_mark_after_evict) {
                        mark_args.mark_stop_after_evict =
                            &mark_stop_after_evict;
                        evict_args.remaining_evict_post_workers =
                            &remaining_evict_post_workers;
                        evict_args.mark_stop_after_evict =
                            &mark_stop_after_evict;
                    }

                    log_cache_progress("mark_dispatch", kAnyAllocBin,
                                       timestamp, mark_workers);
                    auto mark_master = uthread::create<true>(
                        &ConcurrentArrayCache::run_streaming_mark_master,
                        &mark_args, "streaming_mark_master");
                    auto evict_master = uthread::create<true>(
                        &ConcurrentArrayCache::run_streaming_evict_master,
                        &evict_args, "streaming_evict_master");
                    log_cache_progress("tail_join_mark_begin", kAnyAllocBin, timestamp);
                    uthread::join(std::move(mark_master));
                    log_cache_progress("tail_join_mark_end", kAnyAllocBin, timestamp);
                    if (mark_round_limit != 0) {
                        log_cache_progress(
                            "mark_round_region_budget", kAnyAllocBin,
                            mark_round_limit,
                            mark_round_limit - mark_region_budget.load(
                                                   std::memory_order_relaxed));
                    }
                    log_cache_progress("tail_join_evict_begin", kAnyAllocBin, timestamp);
                    uthread::join(std::move(evict_master));
                    log_cache_progress("tail_join_evict_end", kAnyAllocBin, timestamp);
                    if (resume_scan) {
                        const bool evict_complete = evict_pass_complete.load(
                            std::memory_order_relaxed);
                        const bool gc_complete = gc_pass_complete.load(
                            std::memory_order_relaxed);
                        log_cache_progress(
                            "resume_evict_scan", evict_complete ? 1 : 0,
                            scan_round.evict_generation,
                            scan_visit_limit - evict_visit_budget.load(
                                                   std::memory_order_relaxed));
                        log_cache_progress(
                            "resume_gc_scan", gc_complete ? 1 : 0,
                            scan_round.gc_generation,
                            scan_visit_limit - gc_visit_budget.load(
                                                   std::memory_order_relaxed));
                        if (evict_complete) ++inclusive_evict_scan_generation;
                        if (gc_complete) ++inclusive_gc_scan_generation;
                    }

                    profile::get_tlpd().evacuator_master_cycles +=
                        (int64_t)(get_cycles() - evac_master_start_cycles);
                    log_cache_progress("evac_round_end_legacy_opt",
                                       kAnyAllocBin, timestamp,
                                       posted_wrs_total.load(
                                           std::memory_order_relaxed));
                    notify_mutators_ready();
                    handled_eviction_request_seq =
                        pending_eviction_request_seq;
                    continue;
                }

                if (!::FarLib::allocator::global_heap.evacuator_should_start() &&
                    !mutator_waiters_need_progress() &&
                    !explicit_eviction_requested &&
                    (!optimized_ready_queue_enabled() ||
                     pipeline_ready_tasks.empty())) {
                    wait_for_eviction_trigger(
                        handled_eviction_request_seq);
                    continue;
                }

                auto evac_master_start_cycles = get_cycles();
                profile::count_evacuation();
                log_cache_progress("evac_round_begin", kAnyAllocBin,
                                   pipeline_safe_mark_timestamp);

                if (optimized_ready_queue_enabled()) {
                    const size_t ready_mark_limit =
                        optimized_ready_mark_task_limit();
                    const bool ready_full_workers =
                        optimized_ready_full_workers_enabled();
                    const bool need_new_mark =
                        ::FarLib::allocator::global_heap.evacuator_should_start() ||
                        mutator_waiters_need_progress() ||
                        explicit_eviction_requested;
                    if (!need_new_mark && pipeline_ready_tasks.empty()) {
                        wait_for_eviction_trigger(
                            handled_eviction_request_seq);
                        continue;
                    }

                    if (!pipeline_has_safe_mark) {
                        pipeline_safe_mark_timestamp = 1;
                        const size_t prime_mark_workers =
                            (optimized_full_prime_mark_enabled() ||
                             ready_full_workers)
                                ? evacuate_thread_cnt
                                : std::max<size_t>(
                                      size_t(1),
                                      std::min<size_t>(mark_workers,
                                                       evacuate_thread_cnt - 1));
                        std::atomic_size_t prime_ready_budget{
                            ready_mark_limit};
                        StreamingMarkMasterArgs prime_mark_args{
                            this, pipeline_safe_mark_timestamp,
                            prime_mark_workers, begin_frequency_mark_pass(),
                            true, &pipeline_ready_tasks,
                            ready_mark_limit == 0 ? nullptr
                                                  : &prime_ready_budget};
                        auto prime_mark_master = uthread::create<true>(
                            &ConcurrentArrayCache::run_streaming_mark_master,
                            &prime_mark_args, "streaming_mark_master");
                        uthread::join(std::move(prime_mark_master));
                        pipeline_evict_timestamp = pipeline_safe_mark_timestamp + 1;
                        pipeline_gc_timestamp = pipeline_safe_mark_timestamp + 2;
                        pipeline_has_safe_mark = true;
                    }

                    const uint32_t next_mark_timestamp =
                        pipeline_safe_mark_timestamp + 3;
                    size_t overlap_mark_workers = 0;
                    if (need_new_mark) {
                        overlap_mark_workers = mark_workers;
                        if (overlap_mark_workers == 0) {
                            overlap_mark_workers = 1;
                        }
                        if (overlap_mark_workers >= evacuate_thread_cnt) {
                            overlap_mark_workers = evacuate_thread_cnt - 1;
                        }
                    }
                    const size_t overlap_evict_workers = need_new_mark
                                                              ? (evacuate_thread_cnt -
                                                                 overlap_mark_workers)
                                                              : evacuate_thread_cnt;

                    std::atomic_int64_t posted_wrs_total{0};
                    std::vector<::FarLib::allocator::EvictTask> mark_ready_tasks;
                    std::vector<::FarLib::allocator::EvictTask> evict_deferred_tasks;
                    std::atomic_size_t overlap_ready_budget{ready_mark_limit};
                    const FrequencyMarkPassContext frequency_mark_pass =
                        need_new_mark ? begin_frequency_mark_pass()
                                      : FrequencyMarkPassContext{};
                    std::unique_ptr<UThread> mark_master;
                    std::unique_ptr<UThread> evict_master;
                    StreamingMarkMasterArgs mark_args{
                        this, next_mark_timestamp, overlap_mark_workers,
                        frequency_mark_pass, false, &mark_ready_tasks,
                        ready_mark_limit == 0 ? nullptr
                                              : &overlap_ready_budget};
                    StreamingEvictMasterArgs evict_args{
                        this, pipeline_safe_mark_timestamp,
                        pipeline_evict_timestamp, pipeline_gc_timestamp,
                        overlap_evict_workers, &posted_wrs_total,
                        &pipeline_ready_tasks, &evict_deferred_tasks};
                    auto update_max =
                        [](std::atomic_int64_t &target, int64_t value) {
                            int64_t observed =
                                target.load(std::memory_order_relaxed);
                            while (observed < value &&
                                   !target.compare_exchange_weak(
                                       observed, value,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
                            }
                        };
                    bool ready_pipeline_nonserialize = false;
                    int64_t ready_pipeline_evict_elapsed = 0;
                    auto overlap_start = get_cycles();
                    if (optimized_serialize_mark_evict_enabled()) {
                        const size_t serialize_evict_workers =
                            ready_full_workers ? evacuate_thread_cnt
                                               : overlap_evict_workers;
                        const size_t serialize_mark_workers =
                            ready_full_workers ? evacuate_thread_cnt
                                               : overlap_mark_workers;
                        evict_args.worker_count = serialize_evict_workers;
                        mark_args.worker_count = serialize_mark_workers;
                        if (serialize_evict_workers > 0) {
                            evict_master = uthread::create<true>(
                                &ConcurrentArrayCache::run_streaming_evict_master,
                                &evict_args, "streaming_evict_master");
                            uthread::join(std::move(evict_master));
                        }
                        if (need_new_mark && serialize_mark_workers > 0) {
                            mark_master = uthread::create<true>(
                                &ConcurrentArrayCache::run_streaming_mark_master,
                                &mark_args, "streaming_mark_master");
                            uthread::join(std::move(mark_master));
                        }
                        profile::get_tlpd().evac_overlap_mark_cycles +=
                            (int64_t)(get_cycles() - overlap_start);
                    } else {
                        ready_pipeline_nonserialize = true;
                        ready_pipeline_worker_ready_tasks.clear();
                        ready_pipeline_worker_ready_tasks.resize(
                            evacuate_thread_cnt);
                        std::vector<std::vector<::FarLib::allocator::EvictTask>>
                            worker_deferred(
                            evacuate_thread_cnt);
                        std::atomic_size_t next_task_idx{0};
                        std::atomic_int64_t safe_evict_elapsed_max{0};
                        std::atomic_int64_t mark_elapsed_max{0};
                        std::function<void(size_t)> fn_ready_pipeline =
                            [this, &pipeline_ready_tasks, &next_task_idx,
                             &ready_pipeline_worker_ready_tasks,
                             &posted_wrs_total, pipeline_safe_mark_timestamp,
                             need_new_mark, next_mark_timestamp,
                             frequency_mark_pass, ready_mark_limit,
                             &overlap_ready_budget, &worker_deferred,
                             &safe_evict_elapsed_max,
                             &mark_elapsed_max,
                             &update_max](size_t worker_id) {
                                if (!pipeline_ready_tasks.empty()) {
                                    auto evict_worker_start = get_cycles();
                                    evict_ready_worker_logic(
                                        pipeline_ready_tasks, next_task_idx,
                                        pipeline_safe_mark_timestamp,
                                        worker_deferred[worker_id],
                                        posted_wrs_total);
                                    update_max(
                                        safe_evict_elapsed_max,
                                        (int64_t)(get_cycles() -
                                                  evict_worker_start));
                                }
                                if (need_new_mark) {
                                    auto &worker_ready =
                                        ready_pipeline_worker_ready_tasks[
                                            worker_id];
                                    worker_ready.clear();
                                    worker_ready.reserve(64);
                                    auto mark_worker_start = get_cycles();
                                    mark_phase(
                                        next_mark_timestamp,
                                        frequency_mark_pass, &worker_ready,
                                        ready_mark_limit == 0
                                            ? nullptr
                                            : &overlap_ready_budget);
                                    update_max(
                                        mark_elapsed_max,
                                        (int64_t)(get_cycles() -
                                                  mark_worker_start));
                                }
                            };
                        uthread::fork_join<true>(
                            evacuate_thread_cnt, fn_ready_pipeline,
                            "ready_work_conserving");
                        ready_pipeline_evict_elapsed +=
                            safe_evict_elapsed_max.load(
                                std::memory_order_relaxed);
                        if (need_new_mark) {
                            profile::get_tlpd().evac_mark_phase_cycles +=
                                mark_elapsed_max.load(
                                    std::memory_order_relaxed);
                        }
                        size_t future_ready_total = 0;
                        for (auto &batch : ready_pipeline_worker_ready_tasks) {
                            future_ready_total += batch.size();
                        }
                        mark_ready_tasks.reserve(mark_ready_tasks.size() +
                                                 future_ready_total);
                        for (auto &batch : ready_pipeline_worker_ready_tasks) {
                            mark_ready_tasks.insert(
                                mark_ready_tasks.end(),
                                std::make_move_iterator(batch.begin()),
                                std::make_move_iterator(batch.end()));
                            batch.clear();
                        }
                        for (auto &batch : worker_deferred) {
                            evict_deferred_tasks.insert(
                                evict_deferred_tasks.end(),
                                std::make_move_iterator(batch.begin()),
                                std::make_move_iterator(batch.end()));
                            batch.clear();
                        }
                        profile::get_tlpd().evac_overlap_mark_cycles +=
                            (int64_t)(get_cycles() - overlap_start);
                    }

                    if (need_new_mark && overlap_mark_workers > 0) {
                        auto flip_start_ph = get_cycles();
                        auto flip_scope_start = profile::start_flip_scope();
                        flip_scope_state(next_mark_timestamp);
                        profile::end_flip_scope(flip_scope_start);
                        profile::get_tlpd().evac_flip_scope_cycles +=
                            (int64_t)(get_cycles() - flip_start_ph);

                        pipeline_safe_mark_timestamp = next_mark_timestamp;
                        pipeline_evict_timestamp =
                            pipeline_safe_mark_timestamp + 1;
                        pipeline_gc_timestamp = pipeline_safe_mark_timestamp + 2;
                    }

                    if (ready_pipeline_nonserialize && need_new_mark &&
                        !mark_ready_tasks.empty()) {
                        std::vector<
                            std::vector<::FarLib::allocator::EvictTask>>
                            future_worker_deferred(evacuate_thread_cnt);
                        std::atomic_size_t future_next_task_idx{0};
                        std::atomic_int64_t future_evict_elapsed_max{0};
                        std::function<void(size_t)> fn_future_evict =
                            [this, &mark_ready_tasks, &future_next_task_idx,
                             pipeline_safe_mark_timestamp,
                             &future_worker_deferred, &posted_wrs_total,
                             &future_evict_elapsed_max,
                             &update_max](size_t worker_id) {
                                auto evict_worker_start = get_cycles();
                                evict_ready_worker_logic(
                                    mark_ready_tasks, future_next_task_idx,
                                    pipeline_safe_mark_timestamp,
                                    future_worker_deferred[worker_id],
                                    posted_wrs_total);
                                update_max(
                                    future_evict_elapsed_max,
                                    (int64_t)(get_cycles() -
                                              evict_worker_start));
                            };
                        uthread::fork_join<true>(
                            evacuate_thread_cnt, fn_future_evict,
                            "ready_future_evict");
                        ready_pipeline_evict_elapsed +=
                            future_evict_elapsed_max.load(
                                std::memory_order_relaxed);
                        for (auto &batch : future_worker_deferred) {
                            evict_deferred_tasks.insert(
                                evict_deferred_tasks.end(),
                                std::make_move_iterator(batch.begin()),
                                std::make_move_iterator(batch.end()));
                            batch.clear();
                        }
                        mark_ready_tasks.clear();
                    }

                    if (ready_pipeline_nonserialize) {
                        const int64_t evict_elapsed =
                            ready_pipeline_evict_elapsed;
                        const int64_t total_wrs =
                            posted_wrs_total.load(std::memory_order_relaxed);
                        profile::get_tlpd().evac_evict_rounds++;
                        if (total_wrs > 0) {
                            profile::get_tlpd().evac_evict_active_rounds++;
                            profile::get_tlpd().evac_evict_active_cycles +=
                                evict_elapsed;
                            profile::get_tlpd().evac_evict_active_wrs +=
                                total_wrs;
                        } else {
                            profile::get_tlpd().evac_evict_empty_rounds++;
                            profile::get_tlpd().evac_evict_empty_cycles +=
                                evict_elapsed;
                        }
                        profile::get_tlpd().evac_evict_phase_cycles +=
                            evict_elapsed;
                    }

                    std::vector<::FarLib::allocator::EvictTask> next_ready_tasks;
                    next_ready_tasks.reserve(evict_deferred_tasks.size() +
                                             mark_ready_tasks.size());
                    next_ready_tasks.insert(
                        next_ready_tasks.end(),
                        std::make_move_iterator(evict_deferred_tasks.begin()),
                        std::make_move_iterator(evict_deferred_tasks.end()));
                    next_ready_tasks.insert(
                        next_ready_tasks.end(),
                        std::make_move_iterator(mark_ready_tasks.begin()),
                        std::make_move_iterator(mark_ready_tasks.end()));
                    pipeline_ready_tasks.swap(next_ready_tasks);

                    profile::get_tlpd().evac_pipeline_rounds++;
                    profile::get_tlpd().evacuator_master_cycles +=
                        (int64_t)(get_cycles() - evac_master_start_cycles);
                    log_cache_progress("evac_round_end", kAnyAllocBin,
                                       pipeline_safe_mark_timestamp,
                                       posted_wrs_total.load(
                                           std::memory_order_relaxed));
                    notify_mutators_ready();
                    handled_eviction_request_seq =
                        pending_eviction_request_seq;
                    continue;
                }

                if (!pipeline_has_safe_mark) {
                    pipeline_safe_mark_timestamp = 1;
                    const size_t prime_mark_workers =
                        optimized_full_prime_mark_enabled()
                            ? evacuate_thread_cnt
                            : mark_workers;
                    StreamingMarkMasterArgs mark_args{
                        this, pipeline_safe_mark_timestamp, prime_mark_workers,
                        begin_frequency_mark_pass(), true, nullptr, nullptr};
                    auto mark_master = uthread::create<true>(
                        &ConcurrentArrayCache::run_streaming_mark_master,
                        &mark_args, "streaming_mark_master");
                    uthread::join(std::move(mark_master));

                    pipeline_evict_timestamp = pipeline_safe_mark_timestamp + 1;
                    pipeline_gc_timestamp = pipeline_safe_mark_timestamp + 2;
                    pipeline_has_safe_mark = true;
                }

                const uint32_t next_mark_timestamp =
                    pipeline_safe_mark_timestamp + 3;
                std::atomic_int64_t posted_wrs_total{0};
                StreamingMarkMasterArgs mark_args{
                    this, next_mark_timestamp, mark_workers,
                    begin_frequency_mark_pass(), false, nullptr, nullptr};
                StreamingEvictMasterArgs evict_args{
                    this, pipeline_safe_mark_timestamp, pipeline_evict_timestamp,
                    pipeline_gc_timestamp, evict_workers, &posted_wrs_total,
                    nullptr, nullptr};
                auto overlap_start = get_cycles();
                if (optimized_serialize_mark_evict_enabled()) {
                    auto evict_master = uthread::create<true>(
                        &ConcurrentArrayCache::run_streaming_evict_master,
                        &evict_args, "streaming_evict_master");
                    uthread::join(std::move(evict_master));
                    auto mark_master = uthread::create<true>(
                        &ConcurrentArrayCache::run_streaming_mark_master,
                        &mark_args, "streaming_mark_master");
                    uthread::join(std::move(mark_master));
                    profile::get_tlpd().evac_overlap_mark_cycles +=
                        (int64_t)(get_cycles() - overlap_start);
                } else {
                    auto mark_master = uthread::create<true>(
                        &ConcurrentArrayCache::run_streaming_mark_master,
                        &mark_args, "streaming_mark_master");
                    auto evict_master = uthread::create<true>(
                        &ConcurrentArrayCache::run_streaming_evict_master,
                        &evict_args, "streaming_evict_master");
                    uthread::join(std::move(mark_master));
                    profile::get_tlpd().evac_overlap_mark_cycles +=
                        (int64_t)(get_cycles() - overlap_start);
                    uthread::join(std::move(evict_master));
                }

                auto flip_start_ph = get_cycles();
                auto flip_scope_start = profile::start_flip_scope();
                flip_scope_state(next_mark_timestamp);
                profile::end_flip_scope(flip_scope_start);
                profile::get_tlpd().evac_flip_scope_cycles +=
                    (int64_t)(get_cycles() - flip_start_ph);

                pipeline_safe_mark_timestamp = next_mark_timestamp;
                pipeline_evict_timestamp = pipeline_safe_mark_timestamp + 1;
                pipeline_gc_timestamp = pipeline_safe_mark_timestamp + 2;
                profile::get_tlpd().evac_pipeline_rounds++;
                profile::get_tlpd().evacuator_master_cycles +=
                    (int64_t)(get_cycles() - evac_master_start_cycles);
                log_cache_progress("evac_round_end", kAnyAllocBin,
                                   pipeline_safe_mark_timestamp,
                                   posted_wrs_total.load(
                                       std::memory_order_relaxed));
                notify_mutators_ready();
                handled_eviction_request_seq =
                    pending_eviction_request_seq;
                continue;
            }

            if (!::FarLib::allocator::global_heap.evacuator_should_start() &&
                !mutator_waiters_need_progress() &&
                !explicit_eviction_requested) {
                wait_for_eviction_trigger(handled_eviction_request_seq);
                continue;
            }

            auto evac_master_start_cycles = get_cycles();
            profile::count_evacuation();
            log_cache_progress("evac_round_begin_legacy", kAnyAllocBin,
                               timestamp);

            timestamp++;
            auto mark_start_ph = get_cycles();
            const auto frequency_mark_pass = begin_frequency_mark_pass();
            std::function<void(size_t)> fn_mark =
                [this, &timestamp, frequency_mark_pass](size_t) {
                    mark_phase(timestamp, frequency_mark_pass);
            };
            uthread::fork_join<true>(evacuate_thread_cnt, fn_mark);
            auto mark_elapsed = (int64_t)(get_cycles() - mark_start_ph);
            profile::get_tlpd().evac_mark_phase_cycles += mark_elapsed;

            auto flip_start_ph = get_cycles();
            auto flip_scope_start = profile::start_flip_scope();
            flip_scope_state(0);
            profile::end_flip_scope(flip_scope_start);
            auto flip_elapsed = (int64_t)(get_cycles() - flip_start_ph);
            profile::get_tlpd().evac_flip_scope_cycles += flip_elapsed;

            timestamp++;
            auto evict_start_ph = get_cycles();
            std::function<void(size_t)> fn_evacuate = [this, &timestamp](size_t) {
                evacuate_phase(timestamp);
            };
            uthread::fork_join<true>(evacuate_thread_cnt, fn_evacuate);
            auto evict_elapsed = (int64_t)(get_cycles() - evict_start_ph);
            profile::get_tlpd().evac_evict_phase_cycles += evict_elapsed;

            timestamp++;
            auto gc_start_ph = get_cycles();
            std::function<void(size_t)> fn_gc = [this, &timestamp](size_t) {
                gc_phase(timestamp);
            };
            uthread::fork_join<true>(evacuate_thread_cnt, fn_gc);
            auto gc_elapsed = (int64_t)(get_cycles() - gc_start_ph);
            profile::get_tlpd().evac_gc_phase_cycles += gc_elapsed;

            profile::get_tlpd().evacuator_master_cycles +=
                (int64_t)(get_cycles() - evac_master_start_cycles);
            log_cache_progress("evac_round_end_legacy", kAnyAllocBin,
                               timestamp);
            notify_mutators_ready();
            handled_eviction_request_seq = pending_eviction_request_seq;
        }
    }
template <typename ReadyTaskSink>
inline void ConcurrentArrayCache::mark_phase_to_sink(
    uint32_t timestamp, const FrequencyMarkPassContext &frequency_mark_pass,
    ReadyTaskSink *ready_tasks, std::atomic_size_t *ready_task_budget) {
        profile::DualFrequencyHistogramSnapshot local_frequency_histogram;
        local_frequency_histogram.timestamp = timestamp;
        local_frequency_histogram.mark_pass_ordinal =
            frequency_mark_pass.mark_pass_ordinal;
        local_frequency_histogram.ema_mark_interval =
            frequency_mark_pass.ema_mark_interval;
        local_frequency_histogram.ema_updated =
            frequency_mark_pass.update_ema_this_pass;
        auto mark =
            [this, &local_frequency_histogram, frequency_mark_pass](
                        ::FarLib::allocator::BlockHead *b) {
            return this->try_mark(
                b, &local_frequency_histogram,
                frequency_mark_pass.update_ema_this_pass);
        };
        int64_t mark_start = profile::start_mark();
        ::FarLib::allocator::global_heap.mark_to_sink(
            mark, timestamp, ready_tasks, ready_task_budget);
        profile::end_mark(mark_start);
        if (hybrid_profiling_enabled()) {
            profile::record_frequency_histogram_pass(
                timestamp, local_frequency_histogram);
        }
    }

inline void ConcurrentArrayCache::mark_phase(
    uint32_t timestamp, const FrequencyMarkPassContext &frequency_mark_pass,
    std::vector<::FarLib::allocator::EvictTask> *ready_tasks,
    std::atomic_size_t *ready_task_budget, size_t diag_worker_slot,
    std::atomic_size_t *legacy_region_budget,
    std::atomic_bool *mark_stop_after_evict) {
        // Invocation-local (not OS-thread-local): safe across fibre migration.
        const bool nq_result_diag = inclusive_reclaim_diag::enabled() &&
            wait_trace::nq_supply_only.load(std::memory_order_relaxed);
        uint64_t nq_mark_results[8]{};
        profile::DualFrequencyHistogramSnapshot local_frequency_histogram;
        local_frequency_histogram.timestamp = timestamp;
        local_frequency_histogram.mark_pass_ordinal =
            frequency_mark_pass.mark_pass_ordinal;
        local_frequency_histogram.ema_mark_interval =
            frequency_mark_pass.ema_mark_interval;
        local_frequency_histogram.ema_updated =
            frequency_mark_pass.update_ema_this_pass;
        auto mark =
            [this, &local_frequency_histogram, frequency_mark_pass,
             nq_result_diag, &nq_mark_results](
                        ::FarLib::allocator::BlockHead *b) {
            const auto result = this->try_mark(
                b, &local_frequency_histogram,
                frequency_mark_pass.update_ema_this_pass);
            if (nq_result_diag) ++nq_mark_results[static_cast<size_t>(result)];
            return result;
        };
        int64_t mark_start = profile::start_mark();
        const bool legacy_cursors =
            legacy_scan_cursors_enabled() &&
            ::FarLib::get_config().optimized_evacuator &&
            optimized_legacy_exclusive_pipeline_enabled() &&
            ready_tasks == nullptr && ready_task_budget == nullptr &&
            (::FarLib::get_config().exclusive_cache ||
             (inclusive_legacy_eligibility_enabled() &&
              inclusive_legacy_cursors_enabled()));
        const auto eligibility =
            inclusive_legacy_eligibility_enabled() &&
                    ::FarLib::get_config().optimized_evacuator &&
                    !::FarLib::get_config().exclusive_cache &&
                    optimized_legacy_exclusive_pipeline_enabled() &&
                    ready_tasks == nullptr && ready_task_budget == nullptr
                ? ::FarLib::allocator::
                      EvacuationEligibility::LegacyConcurrent
                : ::FarLib::allocator::EvacuationEligibility::All;
        ::FarLib::allocator::MarkBreakdownStats mark_diag;
        mark_diag.held_slot = diag_worker_slot;
        mark_diag.segment_enabled = inclusive_reclaim_diag::enabled();
        auto *mark_diag_ptr = mark_diag.segment_enabled ? &mark_diag : nullptr;
        const uint64_t segment_start =
            mark_diag.segment_enabled ? get_cycles() : 0;
        ::FarLib::allocator::MarkStopContext mark_stop_context;
        mark_stop_context.requested = mark_stop_after_evict;
        ::FarLib::allocator::global_heap.mark(mark, timestamp, ready_tasks,
                                              ready_task_budget, legacy_cursors,
                                              eligibility, mark_diag_ptr,
                                              legacy_cursors &&
                                                      eligibility ==
                                                          ::FarLib::allocator::
                                                              EvacuationEligibility::
                                                                  LegacyConcurrent
                                                  ? legacy_region_budget
                                                  : nullptr,
                                              mark_stop_after_evict == nullptr
                                                  ? nullptr
                                                  : &mark_stop_context);
        if (mark_stop_context.stopped) {
            log_cache_progress("mark_scan_stop", kAnyAllocBin, timestamp,
                               mark_stop_context.completed_regions);
        }
        if (mark_diag.segment_enabled) {
            mark_diag.segment_body_cycles = get_cycles() - segment_start;
            inclusive_reclaim_diag::clear_hold(diag_worker_slot);
            wait_trace::emit("reclaim_mark_scan", timestamp,
                             mark_diag.nodes_examined,
                             mark_diag.stamped_skips, diag_worker_slot);
            wait_trace::emit("reclaim_mark_regions", timestamp,
                             mark_diag.regions_popped,
                             mark_diag.regions_marked, diag_worker_slot);
            wait_trace::emit("reclaim_mark_lock", timestamp,
                             mark_diag.list_lock_wait_cycles,
                             mark_diag.list_lock_hold_cycles,
                             diag_worker_slot);
            wait_trace::emit("reclaim_mark_lock_samples", timestamp,
                             mark_diag.pop_calls, mark_diag.pop_samples,
                             diag_worker_slot);
            wait_trace::emit("reclaim_mark_cycles", timestamp,
                             mark_diag.region_mark_whole_cycles,
                             mark_diag.pop_whole_cycles, diag_worker_slot);
            wait_trace::emit("reclaim_mark_body", timestamp,
                             mark_diag.segment_body_cycles,
                             mark_diag.region_mark_whole_calls,
                             diag_worker_slot);
            wait_trace::emit("reclaim_mark_eligible", timestamp,
                             mark_diag.eligible_calls,
                             mark_diag.eligible_false, diag_worker_slot);
            if (nq_result_diag) {
                uint64_t total = 0;
                for (uint64_t n : nq_mark_results) total += n;
                wait_trace::emit("reclaim_mark_results", timestamp,
                                 nq_mark_results[MARKED], total, diag_worker_slot);
                wait_trace::emit("reclaim_mark_local_pinned", timestamp,
                                 nq_mark_results[LOCAL], nq_mark_results[PINNED],
                                 diag_worker_slot);
                wait_trace::emit("reclaim_mark_free_busy", timestamp,
                                 nq_mark_results[FREE], nq_mark_results[BUSY],
                                 diag_worker_slot);
            }
        }
        profile::end_mark(mark_start);
        if (hybrid_profiling_enabled()) {
            profile::record_frequency_histogram_pass(
                timestamp, local_frequency_histogram);
        }
    }
inline void ConcurrentArrayCache::evict_post_phase(uint32_t timestamp) {
        EvictThreadWorkGuard guard;
        auto phase_start = get_cycles();
        auto &tlpd = profile::get_tlpd();
        int64_t posted_start = tlpd.rdma_write_post_count;
        EvictBufferSet buffer_set;
        buffer_set.init(::FarLib::get_config().server_count);

        const bool epoch_release_enabled = ::FarLib::get_config().optimized_evacuator;
        const uint32_t current_safe_epoch =
            epoch_release_enabled ? safe_epoch.load(std::memory_order_acquire) : 0;

        auto evict = [this, &buffer_set, current_safe_epoch](::FarLib::allocator::BlockHead *b) {
            if (current_safe_epoch > 0) {
                auto *region = ::FarLib::allocator::block_to_region(b);
                uint32_t mark_epoch =
                    region->last_mark_epoch.load(std::memory_order_relaxed);
                if (mark_epoch > current_safe_epoch) {
                // Not safe yet.
                auto block_start = get_cycles();
                profile::get_tlpd().evac_streaming_post_blocked_cycles +=
                    (int64_t)(get_cycles() - block_start);
                return cache::MARKED;
                }
            }
            return this->try_evict(b, buffer_set);
        };
        ::FarLib::allocator::global_heap.evict(evict, timestamp);
        auto client_idx = rdma::thread_info.thread_id;

        auto flush_start = get_cycles();
        buffer_set.flush_all(client_idx, this);
        profile::get_tlpd().evac_flush_cycles += (int64_t)(get_cycles() - flush_start);
        auto elapsed = (int64_t)(get_cycles() - phase_start);
        int64_t posted_wrs = tlpd.rdma_write_post_count - posted_start;
        profile::get_tlpd().evac_evict_rounds++;
        if (posted_wrs > 0) {
            profile::get_tlpd().evac_evict_active_rounds++;
            profile::get_tlpd().evac_evict_active_cycles += elapsed;
            profile::get_tlpd().evac_evict_active_wrs += posted_wrs;
        } else {
            profile::get_tlpd().evac_evict_empty_rounds++;
            profile::get_tlpd().evac_evict_empty_cycles += elapsed;
        }
    }

inline void ConcurrentArrayCache::evict_post_worker_logic(
    uint32_t timestamp, std::atomic_int64_t &posted_wrs_total,
    bool ignore_safe_epoch, size_t diag_worker_slot,
    ::FarLib::allocator::EvacuationEligibility eligibility,
    StreamingEvictMasterArgs::ResumeScanRound *resume_scan,
    std::atomic_bool *evict_stop_after_mark) {
        if (ignore_safe_epoch && exclusive_owned_batch_reclaim_enabled()) {
            std::vector<::FarLib::allocator::EvictTask> collected_tasks;
            constexpr size_t batch_limit = 64;
            while (true) {
                collected_tasks.clear();
                // Keep the current collector eligibility and cursor policy.
                ::FarLib::allocator::global_heap.collect_evict_tasks(
                    timestamp, collected_tasks, batch_limit, 0,
                    legacy_scan_cursors_enabled(), eligibility, nullptr,
                    resume_scan == nullptr ? 0 : resume_scan->evict_generation,
                    resume_scan == nullptr ? nullptr : resume_scan->evict_visit_budget,
                    resume_scan == nullptr ? nullptr : resume_scan->evict_pass_complete);
                if (collected_tasks.empty()) break;

                std::vector<::FarLib::allocator::EvictTask> owned_tasks;
                owned_tasks.reserve(collected_tasks.size());
                for (auto &task : collected_tasks) {
                    if (!::FarLib::allocator::global_heap
                             .adopt_collected_evict_task(task)) {
                        ::FarLib::allocator::global_heap.requeue_evict_task(task);
                        continue;
                    }
                    owned_tasks.push_back(std::move(task));
                }
                if (owned_tasks.empty()) continue;

                std::atomic_size_t next_task_idx{0};
                std::vector<::FarLib::allocator::EvictTask> deferred_tasks;
                // Preserve legacy epoch=0 semantics. Completion polling is
                // not an all-WR barrier: the ready worker only frees FREE
                // slots, and retains tasks with remaining marked work.
                evict_ready_worker_logic(owned_tasks, next_task_idx, 0,
                                         deferred_tasks, posted_wrs_total);
                for (auto &task : deferred_tasks) {
                    ::FarLib::allocator::global_heap.requeue_evict_task(task);
                }
                log_cache_progress("exclusive_owned_batch", kAnyAllocBin,
                                   timestamp, owned_tasks.size());
            }
            return;
        }

        EvictThreadWorkGuard guard;
        auto &tlpd = profile::get_tlpd();
        const int64_t posted_start = tlpd.rdma_write_post_count;
        EvictBufferSet buffer_set;
        buffer_set.init(::FarLib::get_config().server_count);

        const bool epoch_release_enabled =
            ::FarLib::get_config().optimized_evacuator && !ignore_safe_epoch;
        const uint32_t current_safe_epoch =
            epoch_release_enabled ? safe_epoch.load(std::memory_order_acquire) : 0;
        if (epoch_release_enabled && current_safe_epoch == 0) {
            profile::get_tlpd().evac_streaming_safe_epoch_zero_rounds++;
        }

        std::vector<::FarLib::allocator::EvictTask> tasks;
        std::vector<size_t> task_selected_free;
        const size_t batch_limit = 64;
        ::FarLib::allocator::EvictCollectorDiag evict_diag;
        evict_diag.held_slot = diag_worker_slot;
        evict_diag.segment_enabled = inclusive_reclaim_diag::enabled();
        auto *evict_diag_ptr =
            evict_diag.segment_enabled ? &evict_diag : nullptr;
        size_t processed_batches = 0;

        while (true) {
            tasks.clear();
            ::FarLib::allocator::global_heap.collect_evict_tasks(
                timestamp, tasks, batch_limit, current_safe_epoch,
                ignore_safe_epoch && legacy_scan_cursors_enabled() &&
                    (::FarLib::get_config().exclusive_cache ||
                     (eligibility == ::FarLib::allocator::
                                         EvacuationEligibility::LegacyConcurrent &&
                      inclusive_legacy_cursors_enabled())),
                eligibility,
                evict_diag_ptr,
                resume_scan == nullptr ? 0
                                       : resume_scan->evict_generation,
                resume_scan == nullptr
                    ? nullptr
                    : resume_scan->evict_visit_budget,
                resume_scan == nullptr
                    ? nullptr
                    : resume_scan->evict_pass_complete);
            if (tasks.empty()) break;
            task_selected_free.clear();
            uint64_t selected_free_total = 0;
            if (evict_diag_ptr != nullptr) {
                task_selected_free.reserve(tasks.size());
                for (const auto &task : tasks) {
                    const size_t selected_free =
                        task.region == nullptr ? 0 : task.region->free_size();
                    task_selected_free.push_back(selected_free);
                    selected_free_total += selected_free;
                }
                inclusive_reclaim_diag::begin_hold(
                    diag_worker_slot, inclusive_reclaim_diag::HeldEvict,
                    tasks.size(), selected_free_total);
            }
            if (epoch_release_enabled && current_safe_epoch == 0) {
                profile::get_tlpd().evac_streaming_safe_epoch_zero_tasks +=
                    (int64_t)tasks.size();
            }

            for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
                auto &task = tasks[task_idx];
                inclusive_reclaim_diag::HeldRegionGuard held_guard(
                    evict_diag_ptr != nullptr, diag_worker_slot,
                    evict_diag_ptr == nullptr
                        ? 0
                        : task_selected_free[task_idx]);
                // Safety check: only evict if the task's epoch is safe.
                // current_safe_epoch == 0 means we are in round-based mode or
                // streaming is not yet fully active.
                if (current_safe_epoch > 0 &&
                    task.region->last_mark_epoch.load(std::memory_order_relaxed) >
                        current_safe_epoch) {
                    auto block_start = get_cycles();
                    // In a real streaming mode, we might want to yield or wait here.
                    // Keep the region visible for a later safe epoch.
                    task.source_list->push_dbg(task.region,
                                               "evict.defer_unsafe_epoch");
                    profile::get_tlpd().evac_streaming_post_blocked_cycles +=
                        (int64_t)(get_cycles() - block_start);
                    continue;
                }

                auto evict = [this, &buffer_set](::FarLib::allocator::BlockHead *b) {
                    return this->try_evict(b, buffer_set);
                };
                const uint64_t process_start =
                    evict_diag.segment_enabled ? get_cycles() : 0;
                ::FarLib::allocator::global_heap.process_evict_task(
                    task, evict, evict_diag_ptr);
                if (evict_diag.segment_enabled) {
                    const uint64_t process_cycles =
                        get_cycles() - process_start;
                    evict_diag.processing_cycles += process_cycles;
                    evict_diag.processing_max_cycles = std::max(
                        evict_diag.processing_max_cycles, process_cycles);
                }
            }
            if (evict_diag_ptr != nullptr) {
                inclusive_reclaim_diag::clear_hold(diag_worker_slot);
            }

            size_t pending = buffer_set.pending_count();
            if (pending > 0) {
                auto client_idx = rdma::thread_info.thread_id;
                auto flush_start = get_cycles();
                buffer_set.flush_all(client_idx, this);
                profile::get_tlpd().evac_flush_cycles +=
                    (int64_t)(get_cycles() - flush_start);
            }
            ++processed_batches;
            if (evict_stop_after_mark != nullptr &&
                evict_stop_after_mark->load(std::memory_order_acquire)) {
                log_cache_progress("evict_post_stop", kAnyAllocBin,
                                   timestamp, processed_batches);
                break;
            }
        }
        posted_wrs_total.fetch_add(tlpd.rdma_write_post_count - posted_start,
                                   std::memory_order_relaxed);
        if (evict_diag.segment_enabled) {
            inclusive_reclaim_diag::clear_hold(diag_worker_slot);
            wait_trace::emit("reclaim_evict_scan", timestamp,
                             evict_diag.nodes_examined,
                             evict_diag.stamped_skips, diag_worker_slot);
            wait_trace::emit("reclaim_evict_regions", timestamp,
                             evict_diag.selected,
                             evict_diag.selected_unmarked_nonempty,
                             diag_worker_slot);
            wait_trace::emit("reclaim_evict_lock", timestamp,
                             evict_diag.list_lock_wait_cycles,
                             evict_diag.list_lock_hold_cycles,
                             diag_worker_slot);
            wait_trace::emit("reclaim_evict_process", timestamp,
                             evict_diag.processed_regions,
                             evict_diag.processing_zero_reclaim,
                             diag_worker_slot);
            wait_trace::emit("reclaim_evict_cycles", timestamp,
                             evict_diag.processing_cycles,
                             evict_diag.reclaimed_bytes,
                             diag_worker_slot);
        }
    }

inline void ConcurrentArrayCache::evict_ready_worker_logic(
    const std::vector<::FarLib::allocator::EvictTask> &ready_tasks,
    std::atomic_size_t &next_task_idx, uint32_t current_safe_epoch,
    std::vector<::FarLib::allocator::EvictTask> &deferred_tasks,
    std::atomic_int64_t &posted_wrs_total) {
        std::vector<::FarLib::allocator::EvictTask> processed_tasks;
        processed_tasks.reserve(ready_tasks.size());

        {
            EvictThreadWorkGuard guard;
            auto post_start = get_cycles();
            auto &tlpd = profile::get_tlpd();
            const int64_t posted_start = tlpd.rdma_write_post_count;
            EvictBufferSet buffer_set;
            buffer_set.init(::FarLib::get_config().server_count);

            while (true) {
                size_t task_idx =
                    next_task_idx.fetch_add(1, std::memory_order_relaxed);
                if (task_idx >= ready_tasks.size()) break;

                auto task = ready_tasks[task_idx];
                auto *region = task.region;
                if (region == nullptr) continue;
                if (!task.placement_still_matches() ||
                    region->load_placement() ==
                        ::FarLib::allocator::RegionPlacement::Resident) {
                    ::FarLib::allocator::global_heap.requeue_evict_task(task);
                    continue;
                }

                uint32_t region_epoch =
                    region->last_mark_epoch.load(std::memory_order_relaxed);
                if (region_epoch == 0) {
                    region_epoch = task.epoch;
                }
                if (current_safe_epoch > 0 &&
                    region_epoch > current_safe_epoch) {
                    auto block_start = get_cycles();
                    task.epoch = region_epoch;
                    deferred_tasks.push_back(task);
                    profile::get_tlpd().evac_streaming_post_blocked_cycles +=
                        (int64_t)(get_cycles() - block_start);
                    continue;
                }
                if (region->state.load(std::memory_order::relaxed) ==
                    ::FarLib::allocator::IN_USE) [[unlikely]] {
                    profile::count_evac_evict_region_skipped_in_use();
                    task.epoch = region_epoch;
                    deferred_tasks.push_back(task);
                    continue;
                }

                auto evict = [this, &buffer_set](
                                 ::FarLib::allocator::BlockHead *b) {
                    return this->try_evict(b, buffer_set);
                };
                region->evict(evict);
                task.epoch = region_epoch;
                processed_tasks.push_back(task);
            }

            size_t pending = buffer_set.pending_count();
            if (pending > 0) {
                auto client_idx = rdma::thread_info.thread_id;
                auto flush_start = get_cycles();
                buffer_set.flush_all(client_idx, this);
                profile::get_tlpd().evac_flush_cycles +=
                    (int64_t)(get_cycles() - flush_start);
            }
            posted_wrs_total.fetch_add(tlpd.rdma_write_post_count - posted_start,
                                       std::memory_order_relaxed);
            profile::get_tlpd().evac_evict_post_cycles +=
                (int64_t)(get_cycles() - post_start);
        }

        evict_drain_phase();

        auto gc = [this](::FarLib::allocator::BlockHead *b) {
            retry:
                auto obj = b->obj_meta_data.load();
                if (obj.is_null()) return FREE;
                auto entry = obj.get_entry_ptr();
                auto state = entry->load_state().state;
                if (state == FREE || entry->local_addr() != b->get_object_ptr())
                    [[unlikely]]
                    return FREE;
                return entry->load_state().state;
        };

        int64_t gc_start = profile::start_gc();
        for (auto &task : processed_tasks) {
            auto *region = task.region;
            if (region == nullptr) continue;
            if (!task.placement_still_matches() ||
                region->load_placement() ==
                    ::FarLib::allocator::RegionPlacement::Resident) {
                ::FarLib::allocator::global_heap.requeue_evict_task(task);
                continue;
            }
            uint32_t region_epoch =
                region->last_mark_epoch.load(std::memory_order_relaxed);
            if (region_epoch == 0) {
                region_epoch = task.epoch;
            }
            if (current_safe_epoch > 0 && region_epoch > current_safe_epoch) {
                task.epoch = region_epoch;
                deferred_tasks.push_back(task);
                continue;
            }
            region->evict(gc);
            if (region->marked_list == nullptr) {
                region->note_evict_drained_epoch(region_epoch);
                ::FarLib::allocator::global_heap.publish_owned_evict_region(
                    region);
            } else {
                task.epoch = region_epoch;
                deferred_tasks.push_back(task);
            }
        }
        profile::end_gc(gc_start);
        profile::count_gc();
    }

inline void ConcurrentArrayCache::evict_drain_phase() {
        log_cache_progress("drain_begin");
        EvictThreadWorkGuard guard;
        auto client_idx = rdma::thread_info.thread_id;
        auto *client = rdma::get_client(client_idx);
        size_t qp_idx = client->get_qp_idx();

        auto cq_wait_start = get_cycles();
        while (check_cq_idx_with_client_idx(qp_idx, client_idx));
        auto elapsed = (int64_t)(get_cycles() - cq_wait_start);
        profile::get_tlpd().evac_check_cq_wait_cycles += elapsed;
        profile::get_tlpd().evac_drain_cycles += elapsed;
        log_cache_progress("drain_end", kAnyAllocBin,
                           static_cast<uint64_t>(elapsed));
    }

inline void ConcurrentArrayCache::drain_eviction_completions_for_shutdown() {
        using ::FarLib::allocator::BlockHead;
        using ::FarLib::allocator::RegionHead;
        using ::FarLib::allocator::RegionSize;

        const size_t client_count = rdma::get_client_count();
        std::vector<uint8_t> pending_clients(client_count, 0);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(30);
        uint64_t passes = 0;
        uint64_t completions = 0;
        while (true) {
            std::fill(pending_clients.begin(), pending_clients.end(), 0);
            uint64_t pending_entries = 0;
            const size_t committed =
                ::FarLib::allocator::global_heap.get_committed_bytes();
            auto *heap = static_cast<char *>(
                ::FarLib::allocator::global_heap.get_heap());
            for (size_t offset = 0; offset < committed;
                 offset += RegionSize) {
                auto *region =
                    reinterpret_cast<RegionHead *>(heap + offset);
                auto scan_list = [&](BlockHead *block) {
                    for (; block != nullptr; block = block->next) {
                        const auto obj = block->obj_meta_data.load(
                            std::memory_order_acquire);
                        if (obj.is_null()) {
                            continue;
                        }
                        auto &entry = get_entry_of(obj);
                        const auto state = entry.load_state(
                            std::memory_order_acquire);
                        if (state.state != EVICTING) {
                            continue;
                        }
                        ++pending_entries;
                        const size_t client_idx = entry.get_client_idx();
                        ASSERT(client_idx < pending_clients.size());
                        pending_clients[client_idx] = 1;
                    }
                };
                scan_list(region->active_list);
                scan_list(region->marked_list);
            }
            if (pending_entries == 0) {
                std::cerr << "shutdown.rdma_drain passes=" << passes
                          << " completions=" << completions << std::endl;
                return;
            }
            size_t progressed = 0;
            for (size_t client_idx = 0; client_idx < client_count;
                 ++client_idx) {
                if (pending_clients[client_idx] != 0) {
                    progressed +=
                        check_cq_idx_with_client_idx(0, client_idx);
                }
            }
            completions += progressed;
            ++passes;
            if (std::chrono::steady_clock::now() >= deadline) {
                std::cerr
                    << "shutdown.rdma_drain timeout pending_entries="
                    << pending_entries << " passes=" << passes
                    << " completions=" << completions << std::endl;
                ERROR("shutdown RDMA completion drain timed out");
            }
            if (progressed == 0) {
                uthread::yield();
            }
        }
    }

inline void ConcurrentArrayCache::evacuate_phase(uint32_t timestamp) {
        int64_t evict_start = profile::start_evict();
        evict_post_phase(timestamp);
        evict_drain_phase();
        profile::end_evict(evict_start);
    }
inline void ConcurrentArrayCache::gc_phase(
    uint32_t timestamp,
    ::FarLib::allocator::EvacuationEligibility eligibility,
    StreamingEvictMasterArgs::ResumeScanRound *resume_scan) {
        log_cache_progress("gc_begin", kAnyAllocBin, timestamp);
        uint64_t object_checks = 0;
        uint64_t object_free = 0;
        uint64_t object_marked = 0;
        uint64_t object_evicting = 0;
        auto gc = [this, &object_checks, &object_free, &object_marked,
                   &object_evicting](::FarLib::allocator::BlockHead *b) {
            retry:
                ++object_checks;
                auto obj = b->obj_meta_data.load();
                if (obj.is_null()) {
                    ++object_free;
                    return FREE;
                }
                auto entry = obj.get_entry_ptr();
                auto state = entry->load_state().state;
                if (state == FREE || entry->local_addr() != b->get_object_ptr())
                    [[unlikely]] {
                    ++object_free;
                    return FREE;
                }
                const auto result = entry->load_state().state;
                if (result == FREE) ++object_free;
                if (result == MARKED) ++object_marked;
                if (result == EVICTING) ++object_evicting;
                return result;
        };
        const bool require_evict_processed =
            eligibility ==
                ::FarLib::allocator::EvacuationEligibility::LegacyConcurrent &&
            !::FarLib::get_config().exclusive_cache &&
            inclusive_gc_processed_filter_enabled();
        ::FarLib::allocator::GcProcessedFilterStats filter_stats;
        int64_t gc_start = profile::start_gc();
        ::FarLib::allocator::global_heap.evict(
            gc, timestamp, eligibility,
            resume_scan == nullptr ? 0 : resume_scan->gc_generation,
            resume_scan == nullptr ? nullptr : resume_scan->gc_visit_budget,
            resume_scan == nullptr ? nullptr : resume_scan->gc_pass_complete,
            require_evict_processed,
            require_evict_processed ? &filter_stats : nullptr);
        profile::end_gc(gc_start);
        profile::count_gc();
        log_cache_progress("gc_filter_accept", kAnyAllocBin, timestamp,
                           filter_stats.accepted_checks);
        log_cache_progress("gc_filter_reject", kAnyAllocBin, timestamp,
                           filter_stats.rejected_checks);
        log_cache_progress("gc_objects_checks", kAnyAllocBin, timestamp,
                           object_checks);
        log_cache_progress("gc_objects_free", kAnyAllocBin, timestamp,
                           object_free);
        log_cache_progress("gc_objects_marked", kAnyAllocBin, timestamp,
                           object_marked);
        log_cache_progress("gc_objects_evicting", kAnyAllocBin, timestamp,
                           object_evicting);
        log_cache_progress("gc_end", kAnyAllocBin, timestamp);
    }

inline void ConcurrentArrayCache::flip_scope_state(uint32_t epoch_to_release) {
        log_cache_progress("flip_scope_begin", kAnyAllocBin, epoch_to_release);
        auto old_state =
            mutator_states.global_state.load(std::memory_order::relaxed);
        auto new_state = old_state == InScopeV0 ? InScopeV1 : InScopeV0;
        // Do not pre-wait on new_state. Under heavy contention this can livelock
        // and block progress; the required safety barrier is draining old_state
        // after publishing the new global state.
        mutator_states.global_state.store(new_state);
        scope_diag::begin_flip(epoch_to_release, old_state);
        int64_t old_entry_count =
            (int64_t)mutator_states.count[old_state].load(std::memory_order::acquire);
        if (old_entry_count > 0) {
            profile::count_evac_flip_wait_old_blocked_flip(old_entry_count);
        }
        while (true) {
            int64_t observed_old =
                (int64_t)mutator_states.count[old_state].load(std::memory_order::acquire);
            scope_diag::snapshot(epoch_to_release, observed_old);
            if (observed_old == 0) break;
            static thread_local uint64_t flip_wait_loop = 0;
            ++flip_wait_loop;
            profile::count_evac_flip_wait_old_loop();
            profile::count_evac_flip_wait_old_max(observed_old);
            profile::count_evac_flip_wait_old_observed(observed_old);
            auto start_yield = get_cycles();
            uthread::yield();
            auto yield_cycles = get_cycles() - start_yield;
            profile::count_flip_scope_yield(yield_cycles);
            profile::count_evac_flip_wait_old_cycles(yield_cycles);
        }
        if (epoch_to_release > 0) {
            safe_epoch.store(epoch_to_release, std::memory_order_release);
        }
        scope_diag::end_flip(epoch_to_release);
        log_cache_progress("flip_scope_end", kAnyAllocBin, epoch_to_release);
    }

inline void ConcurrentArrayCache::on_demand_invoke_eviction(size_t alloc_bin) {
        const auto cur_scope_state = uthread::get_tls()->scope_state;
        assert(cur_scope_state == OutOfScope);
        log_cache_progress("ondemand_enter", alloc_bin);
        if (use_legacy_baseline_ondemand()) {
            uthread::lock(&eviction_mutex);
            uthread::set_high_priority();
            mutator_can_not_allocate.store(true, std::memory_order_release);
            log_cache_progress("ondemand_legacy_sleep", alloc_bin);
            uthread::notify_all_locked(&eviction_cond);
            uthread::wait_locked(&mutator_cond, &eviction_mutex);
            uthread::set_default_priority();
            log_cache_progress("ondemand_legacy_exit", alloc_bin);
            return;
        }
        if (optimized_legacy_exclusive_pipeline_enabled() &&
            (::FarLib::get_config().exclusive_cache ||
             (alloc_bin < kAnyAllocBin &&
              inclusive_single_wait_enabled()))) {
            uthread::lock(&eviction_mutex);
            uthread::set_high_priority();
            mutator_can_not_allocate.store(true, std::memory_order_release);
            eviction_request_seq.fetch_add(1, std::memory_order_acq_rel);
            mutator_waiters.fetch_add(1, std::memory_order_relaxed);
            log_cache_progress("ondemand_legacy_exclusive_sleep", alloc_bin);
            uthread::notify_all_locked(&eviction_cond);
            uthread::set_default_priority();
            const bool condition_wait_diag_active =
                profile::diagnostics_active();
            const bool ready_before_sleep =
                condition_wait_diag_active && alloc_bin_ready(alloc_bin);
            const uint64_t condition_wait_start =
                condition_wait_diag_active ? get_cycles() : 0;
            log_cache_progress("wait_sleep", alloc_bin);
            uthread::wait_locked(&mutator_cond, &eviction_mutex);
            log_cache_progress("wait_return", alloc_bin);
            const uint64_t condition_wait_end =
                condition_wait_diag_active ? get_cycles() : 0;
            uthread::lock(&eviction_mutex);
            log_cache_progress("wait_relocked", alloc_bin,
                               static_cast<uint64_t>(alloc_bin_ready(alloc_bin)),
                               static_cast<uint64_t>(
                                   ::FarLib::allocator::global_heap
                                       .get_free_size()));
            const bool ready_after_wake =
                condition_wait_diag_active && alloc_bin_ready(alloc_bin);
            if (condition_wait_diag_active) {
                profile::record_allocation_condition_wait(
                    condition_wait_end - condition_wait_start,
                    ready_before_sleep, ready_after_wake);
            }
            mutator_waiters.fetch_sub(1, std::memory_order_relaxed);
            uthread::unlock(&eviction_mutex);
            log_cache_progress("ondemand_legacy_exclusive_exit", alloc_bin);
            return;
        }
        uthread::lock(&eviction_mutex);
        uthread::set_high_priority();
        mutator_can_not_allocate.store(true, std::memory_order_release);
        eviction_request_seq.fetch_add(1, std::memory_order_acq_rel);
        mutator_waiters.fetch_add(1, std::memory_order_relaxed);
        // can not allocate now
        log_cache_progress("ondemand_notify", alloc_bin);
        uthread::notify_all_locked(&eviction_cond);
        // Waiters must not stay high-priority while blocked on eviction.
        // Under all-worker allocation pressure this can starve the normal
        // priority evacuator uthread that is supposed to wake them.
        uthread::set_default_priority();
        bool blocked_alloc =
            mutator_can_not_allocate.load(std::memory_order_acquire);
        bool bin_ready = alloc_bin_ready(alloc_bin);
        uint64_t wait_loops = 0;
        while (blocked_alloc || !bin_ready) {
            ++wait_loops;
            uthread::wait_locked(&mutator_cond, &eviction_mutex);
            // FredCondition::wait returns without keeping the mutex acquired.
            // Re-acquire it before checking the predicate again.
            uthread::lock(&eviction_mutex);
            blocked_alloc =
                mutator_can_not_allocate.load(std::memory_order_acquire);
            bin_ready = alloc_bin_ready(alloc_bin);
        }
        mutator_waiters.fetch_sub(1, std::memory_order_relaxed);
        uthread::unlock(&eviction_mutex);
    }

}  // namespace FarLib::cache
