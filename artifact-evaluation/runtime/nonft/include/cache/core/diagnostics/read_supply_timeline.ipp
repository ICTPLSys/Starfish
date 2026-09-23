#pragma once

#include <pthread.h>
#include <sched.h>

namespace FarLib::cache {

inline void ConcurrentArrayCache::start_read_supply_timeline() {
    const char *configured_path =
        std::getenv("FARLIB_READ_SUPPLY_TIMELINE_PATH");
    if (configured_path == nullptr || configured_path[0] == '\0') return;

    const std::string path(configured_path);
    read_supply_timeline_stop.store(false, std::memory_order_release);
    read_supply_timeline_thread = std::thread([this, path] {
        const char *cpu_text =
            std::getenv("FARLIB_READ_SUPPLY_TIMELINE_CPU");
        const int requested_cpu = cpu_text == nullptr ? 0 : std::atoi(cpu_text);
        int affinity_rc = -1;
        if (requested_cpu >= 0 && requested_cpu < CPU_SETSIZE) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(requested_cpu, &set);
            affinity_rc = pthread_setaffinity_np(pthread_self(), sizeof(set),
                                                 &set);
        }

        const size_t client_count = rdma::get_client_count();
        if (client_count > read_supply_timeline::kMaxClients) {
            std::fprintf(stderr,
                         "read_supply_timeline client_count=%zu exceeds max=%zu\n",
                         client_count, read_supply_timeline::kMaxClients);
            std::abort();
        }
        size_t qp_count = 1;
        for (size_t client_idx = 0; client_idx < client_count; ++client_idx) {
            auto *client = rdma::get_client(client_idx);
            if (client->get_endpoint_count() != 1) {
                std::fprintf(stderr,
                             "read_supply_timeline requires one endpoint; "
                             "client=%zu endpoints=%zu\n",
                             client_idx, client->get_endpoint_count());
                std::abort();
            }
            const size_t local_qps = client->get_local_qp_count();
            if (local_qps > read_supply_timeline::kMaxQpsPerClient) {
                std::fprintf(stderr,
                             "read_supply_timeline client=%zu qps=%zu exceeds "
                             "max=%zu\n",
                             client_idx, local_qps,
                             read_supply_timeline::kMaxQpsPerClient);
                std::abort();
            }
            qp_count = std::max(qp_count, local_qps);
        }

        FILE *output = std::fopen(path.c_str(), "w");
        if (output == nullptr) {
            std::perror("FARLIB_READ_SUPPLY_TIMELINE_PATH");
            return;
        }
        (void)std::setvbuf(output, nullptr, _IOFBF, 1 << 20);
        std::fprintf(
            output,
            "mono_ns,realtime_ns,tsc,sampler_cpu,phase_active,cache_working,"
            "stw_active,mutator_waiters,mutator_cannot_allocate,memory_low,"
            "free_bytes,rate_diag_enabled,rate_shards,rate_dropped,"
            "alloc_slots,alloc_slot_bytes,mark_reclaimed_slots,"
            "mark_reclaimed_slot_bytes,evict_reclaimed_slots,"
            "evict_reclaimed_slot_bytes,gc_reclaimed_slots,"
            "gc_reclaimed_slot_bytes,reclaimed_slots,reclaimed_slot_bytes,"
            "published_new_slots,published_new_slot_bytes,"
            "reclaim_diag_enabled,reclaim_alloc_bin,"
            "reclaim_usable_regions,reclaim_free_regions,"
            "reclaim_mark_held_regions,"
            "reclaim_mark_held_selected_free_bytes,"
            "reclaim_evict_held_regions,"
            "reclaim_evict_held_selected_free_bytes,"
            "reclaim_diag_slot_errors,reclaim_diag_snapshot_unstable,"
            "total_accepted,total_reaped,total_pending,"
            "negative_slot_count,max_negative,overflow_guard,total_write_posts,"
            "total_cq_empty_polls,total_cq_nonempty_polls,total_cq_read_wcs,"
            "total_cq_write_wcs,scope_registered,logical_pending_fibres,"
            "phase_unknown,phase_compute,phase_access,phase_alloc,phase_refill,"
            "phase_rdma_post,phase_rdma_wait,phase_cq_poll,phase_yield,"
            "phase_mutex,phase_cond,phase_wc_handle,phase_remote_free,"
            "pending_state_free,pending_state_pinned,pending_state_local,"
            "pending_state_marked,pending_state_evicting,pending_state_remote,"
            "pending_state_fetching,pending_state_busy,pending_state_invalid,"
            "pending_state_unstable,yield_pending_state_free,"
            "yield_pending_state_pinned,yield_pending_state_local,"
            "yield_pending_state_marked,yield_pending_state_evicting,"
            "yield_pending_state_remote,yield_pending_state_fetching,"
            "yield_pending_state_busy,yield_pending_state_invalid,"
            "yield_pending_state_unstable");
        for (size_t client_idx = 0; client_idx < client_count; ++client_idx) {
            for (size_t qp_idx = 0; qp_idx < qp_count; ++qp_idx) {
                std::fprintf(output,
                             ",c%zuq%zu_accepted,c%zuq%zu_reaped,"
                             "c%zuq%zu_pending",
                             client_idx, qp_idx, client_idx, qp_idx,
                             client_idx, qp_idx);
            }
        }
        for (size_t fibre_idx = 0; fibre_idx < 48; ++fibre_idx) {
            std::fprintf(output, ",f%zu_fid,f%zu_phase,f%zu_pending",
                         fibre_idx, fibre_idx, fibre_idx);
        }
        std::fputc('\n', output);

        std::fprintf(stderr,
                     "read_supply_timeline enabled path=%s interval_us=1000 "
                     "requested_cpu=%d affinity_rc=%d actual_cpu=%d clients=%zu "
                     "qps=%zu packed_read_halves=32 cq_publish_batch=%llu\n",
                     path.c_str(), requested_cpu, affinity_rc, sched_getcpu(),
                     client_count, qp_count,
                     static_cast<unsigned long long>(
                         read_supply_timeline::kPollPublishBatch));

        std::vector<read_supply_timeline::SlotSnapshot> snapshots(
            client_count * qp_count);
        auto next = std::chrono::steady_clock::now();
        uint64_t sample_count = 0;
        bool prior_phase_active = false;
        uint32_t completed_phase_windows = 0;
        while (!read_supply_timeline_stop.load(std::memory_order_acquire)) {
            struct timespec mono {};
            struct timespec realtime {};
            (void)clock_gettime(CLOCK_MONOTONIC, &mono);
            (void)clock_gettime(CLOCK_REALTIME, &realtime);
            const uint64_t mono_ns =
                static_cast<uint64_t>(mono.tv_sec) * 1000000000ULL +
                static_cast<uint64_t>(mono.tv_nsec);
            const uint64_t realtime_ns =
                static_cast<uint64_t>(realtime.tv_sec) * 1000000000ULL +
                static_cast<uint64_t>(realtime.tv_nsec);
            const uint64_t tsc = __rdtsc();
            const bool phase_active = profile::is_working();
            const bool reclaim_diag_enabled =
                inclusive_reclaim_diag::enabled();
            size_t reclaim_alloc_bin = inclusive_reclaim_diag::kNoAllocBin;
            ::FarLib::allocator::ReclaimSupplySnapshot reclaim_supply;
            inclusive_reclaim_diag::HeldSnapshot reclaim_held;
            const auto rate_snapshot = alloc_reclaim_rate_diag::snapshot();
            if (reclaim_diag_enabled) {
                reclaim_alloc_bin = reclaim_diag_last_alloc_bin.load(
                    std::memory_order_relaxed);
                reclaim_supply =
                    ::FarLib::allocator::global_heap.reclaim_supply_snapshot(
                        reclaim_alloc_bin);
                reclaim_held = inclusive_reclaim_diag::snapshot();
            }

            uint64_t total_accepted = 0;
            uint64_t total_reaped = 0;
            int64_t total_pending = 0;
            uint64_t total_write_posts = 0;
            uint64_t total_cq_empty_polls = 0;
            uint64_t total_cq_nonempty_polls = 0;
            uint64_t total_cq_read_wcs = 0;
            uint64_t total_cq_write_wcs = 0;
            uint64_t negative_slot_count = 0;
            int64_t max_negative = 0;
            uint64_t overflow_guard = 0;
            std::array<uint64_t, scope_diag::REMOTE_FREE + 1> phase_counts{};
            std::array<uint64_t, 48> fibre_fids{};
            std::array<uint64_t, 48> fibre_phases{};
            std::array<uint64_t, 48> fibre_pending{};
            std::array<uint64_t, 8> pending_states{};
            std::array<uint64_t, 8> yield_pending_states{};
            uint64_t pending_state_invalid = 0;
            uint64_t pending_state_unstable = 0;
            uint64_t yield_pending_state_invalid = 0;
            uint64_t yield_pending_state_unstable = 0;
            uint64_t scope_registered = 0;
            uint64_t logical_pending_fibres = 0;
            size_t snapshot_idx = 0;
            for (size_t client_idx = 0; client_idx < client_count;
                 ++client_idx) {
                for (size_t qp_idx = 0; qp_idx < qp_count; ++qp_idx) {
                    auto snapshot = read_supply_timeline::snapshot_slot(
                        client_idx, qp_idx);
                    snapshots[snapshot_idx++] = snapshot;
                    total_accepted += snapshot.accepted;
                    total_reaped += snapshot.reaped;
                    total_pending += snapshot.pending;
                    total_write_posts += snapshot.write_posts;
                    total_cq_empty_polls += snapshot.cq_empty_polls;
                    total_cq_nonempty_polls += snapshot.cq_nonempty_polls;
                    total_cq_read_wcs += snapshot.cq_read_wcs;
                    total_cq_write_wcs += snapshot.cq_write_wcs;
                    if (snapshot.pending < 0) {
                        ++negative_slot_count;
                        max_negative = std::min(max_negative,
                                                snapshot.pending);
                    }
                    if (snapshot.accepted >= 0xf0000000U ||
                        snapshot.reaped >= 0xf0000000U) {
                        overflow_guard = 1;
                    }
                }
            }
            for (size_t fibre_idx = 0; fibre_idx < 48; ++fibre_idx) {
                const auto &slot = scope_diag::slots[fibre_idx];
                const uint64_t fid = slot.fid.load(std::memory_order_acquire);
                const uint64_t phase =
                    slot.phase.load(std::memory_order_relaxed);
                const uint64_t pending =
                    slot.pending.load(std::memory_order_relaxed);
                fibre_fids[fibre_idx] = fid;
                fibre_phases[fibre_idx] = phase;
                fibre_pending[fibre_idx] = pending;
                if (fid != 0) {
                    ++scope_registered;
                    if (phase <= scope_diag::REMOTE_FREE) {
                        ++phase_counts[phase];
                    }
                    if (pending != 0) ++logical_pending_fibres;
                }
                if (phase_active && fid != 0 && pending != 0 &&
                    scope_diag::inspect_entry != nullptr) {
                    const uint64_t entry_state =
                        scope_diag::inspect_entry(pending);
                    const uint64_t fid_after =
                        slot.fid.load(std::memory_order_acquire);
                    const uint64_t phase_after =
                        slot.phase.load(std::memory_order_relaxed);
                    const uint64_t pending_after =
                        slot.pending.load(std::memory_order_relaxed);
                    if (fid_after != fid || phase_after != phase ||
                        pending_after != pending) {
                        ++pending_state_unstable;
                        if (phase == scope_diag::YIELD) {
                            ++yield_pending_state_unstable;
                        }
                    } else {
                        const uint64_t state = entry_state & 0xffffffffULL;
                        const bool invalid = (entry_state >> 32) != 0;
                        if (state < pending_states.size()) {
                            ++pending_states[state];
                            if (phase == scope_diag::YIELD) {
                                ++yield_pending_states[state];
                            }
                        }
                        if (invalid) {
                            ++pending_state_invalid;
                            if (phase == scope_diag::YIELD) {
                                ++yield_pending_state_invalid;
                            }
                        }
                    }
                }
            }

            std::fprintf(
                output,
                "%llu,%llu,%llu,%d,%d,%d,%d,%u,%d,%d,%lld",
                static_cast<unsigned long long>(mono_ns),
                static_cast<unsigned long long>(realtime_ns),
                static_cast<unsigned long long>(tsc), sched_getcpu(),
                phase_active ? 1 : 0,
                working.load(std::memory_order_relaxed) ? 1 : 0,
                stw_active.load(std::memory_order_relaxed) ? 1 : 0,
                mutator_waiters.load(std::memory_order_relaxed),
                mutator_can_not_allocate.load(std::memory_order_relaxed) ? 1
                                                                         : 0,
                ::FarLib::allocator::global_heap.memory_low() ? 1 : 0,
                static_cast<long long>(
                    ::FarLib::allocator::global_heap.get_free_size()));
            const uint64_t reclaimed_slots =
                rate_snapshot.mark_slots + rate_snapshot.evict_slots +
                rate_snapshot.gc_slots;
            const uint64_t reclaimed_slot_bytes =
                rate_snapshot.mark_slot_bytes +
                rate_snapshot.evict_slot_bytes +
                rate_snapshot.gc_slot_bytes;
            std::fprintf(
                output,
                ",%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
                "%llu,%llu,%llu,%llu",
                alloc_reclaim_rate_diag::enabled() ? 1 : 0,
                static_cast<unsigned long long>(rate_snapshot.shard_count),
                static_cast<unsigned long long>(rate_snapshot.dropped),
                static_cast<unsigned long long>(rate_snapshot.alloc_slots),
                static_cast<unsigned long long>(rate_snapshot.alloc_slot_bytes),
                static_cast<unsigned long long>(rate_snapshot.mark_slots),
                static_cast<unsigned long long>(rate_snapshot.mark_slot_bytes),
                static_cast<unsigned long long>(rate_snapshot.evict_slots),
                static_cast<unsigned long long>(rate_snapshot.evict_slot_bytes),
                static_cast<unsigned long long>(rate_snapshot.gc_slots),
                static_cast<unsigned long long>(rate_snapshot.gc_slot_bytes),
                static_cast<unsigned long long>(reclaimed_slots),
                static_cast<unsigned long long>(reclaimed_slot_bytes),
                static_cast<unsigned long long>(rate_snapshot.published_slots),
                static_cast<unsigned long long>(
                    rate_snapshot.published_slot_bytes));
            std::fprintf(
                output,
                ",%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
                reclaim_diag_enabled ? 1 : 0,
                static_cast<unsigned long long>(reclaim_alloc_bin),
                static_cast<unsigned long long>(
                    reclaim_supply.usable_regions),
                static_cast<unsigned long long>(reclaim_supply.free_regions),
                static_cast<unsigned long long>(reclaim_held.mark_regions),
                static_cast<unsigned long long>(
                    reclaim_held.mark_selected_free_bytes),
                static_cast<unsigned long long>(reclaim_held.evict_regions),
                static_cast<unsigned long long>(
                    reclaim_held.evict_selected_free_bytes),
                static_cast<unsigned long long>(reclaim_held.slot_errors),
                static_cast<unsigned long long>(
                    reclaim_held.snapshot_unstable));
            std::fprintf(
                output,
                ",%llu,%llu,%lld,%llu,%lld,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
                static_cast<unsigned long long>(total_accepted),
                static_cast<unsigned long long>(total_reaped),
                static_cast<long long>(total_pending),
                static_cast<unsigned long long>(negative_slot_count),
                static_cast<long long>(max_negative),
                static_cast<unsigned long long>(overflow_guard),
                static_cast<unsigned long long>(total_write_posts),
                static_cast<unsigned long long>(total_cq_empty_polls),
                static_cast<unsigned long long>(total_cq_nonempty_polls),
                static_cast<unsigned long long>(total_cq_read_wcs),
                static_cast<unsigned long long>(total_cq_write_wcs),
                static_cast<unsigned long long>(scope_registered),
                static_cast<unsigned long long>(logical_pending_fibres));
            for (const uint64_t phase_count : phase_counts) {
                std::fprintf(output, ",%llu",
                             static_cast<unsigned long long>(phase_count));
            }
            for (const uint64_t state_count : pending_states) {
                std::fprintf(output, ",%llu",
                             static_cast<unsigned long long>(state_count));
            }
            std::fprintf(output, ",%llu,%llu",
                         static_cast<unsigned long long>(
                             pending_state_invalid),
                         static_cast<unsigned long long>(
                             pending_state_unstable));
            for (const uint64_t state_count : yield_pending_states) {
                std::fprintf(output, ",%llu",
                             static_cast<unsigned long long>(state_count));
            }
            std::fprintf(output, ",%llu,%llu",
                         static_cast<unsigned long long>(
                             yield_pending_state_invalid),
                         static_cast<unsigned long long>(
                             yield_pending_state_unstable));
            for (const auto &snapshot : snapshots) {
                std::fprintf(output, ",%u,%u,%lld", snapshot.accepted,
                             snapshot.reaped,
                             static_cast<long long>(snapshot.pending));
            }
            for (size_t fibre_idx = 0; fibre_idx < 48; ++fibre_idx) {
                std::fprintf(output, ",%llu,%llu,%llu",
                             static_cast<unsigned long long>(
                                 fibre_fids[fibre_idx]),
                             static_cast<unsigned long long>(
                                 fibre_phases[fibre_idx]),
                             static_cast<unsigned long long>(
                                 fibre_pending[fibre_idx]));
            }
            std::fputc('\n', output);
            if (++sample_count % 500 == 0) std::fflush(output);
            if (prior_phase_active && !phase_active) {
                ++completed_phase_windows;
            }
            prior_phase_active = phase_active;
            if (completed_phase_windows == 3) break;
            next += std::chrono::milliseconds(1);
            std::this_thread::sleep_until(next);
        }
        std::fflush(output);
        std::fclose(output);
        std::fprintf(stderr,
                     "read_supply_timeline stopped samples=%llu actual_cpu=%d "
                     "completed_phase_windows=%u\n",
                     static_cast<unsigned long long>(sample_count),
                     sched_getcpu(), completed_phase_windows);
        const auto final_rate = alloc_reclaim_rate_diag::snapshot();
        std::fprintf(
            stderr,
            "alloc_reclaim_rate_diag enabled=%d shards=%llu dropped=%llu "
            "alloc_slots=%llu alloc_slot_bytes=%llu mark_slots=%llu "
            "mark_slot_bytes=%llu evict_slots=%llu evict_slot_bytes=%llu "
            "gc_slots=%llu gc_slot_bytes=%llu published_new_slots=%llu "
            "published_new_slot_bytes=%llu\n",
            alloc_reclaim_rate_diag::enabled() ? 1 : 0,
            static_cast<unsigned long long>(final_rate.shard_count),
            static_cast<unsigned long long>(final_rate.dropped),
            static_cast<unsigned long long>(final_rate.alloc_slots),
            static_cast<unsigned long long>(final_rate.alloc_slot_bytes),
            static_cast<unsigned long long>(final_rate.mark_slots),
            static_cast<unsigned long long>(final_rate.mark_slot_bytes),
            static_cast<unsigned long long>(final_rate.evict_slots),
            static_cast<unsigned long long>(final_rate.evict_slot_bytes),
            static_cast<unsigned long long>(final_rate.gc_slots),
            static_cast<unsigned long long>(final_rate.gc_slot_bytes),
            static_cast<unsigned long long>(final_rate.published_slots),
            static_cast<unsigned long long>(
                final_rate.published_slot_bytes));
    });
}

inline void ConcurrentArrayCache::stop_read_supply_timeline() {
    read_supply_timeline_stop.store(true, std::memory_order_release);
    if (read_supply_timeline_thread.joinable()) {
        read_supply_timeline_thread.join();
    }
    if (!read_supply_timeline::nowait_gap_enabled()) return;

    const char *configured_path =
        std::getenv("FARLIB_READ_SUPPLY_TIMELINE_PATH");
    if (configured_path == nullptr || configured_path[0] == '\0') return;
    const std::string latency_path =
        std::string(configured_path) + ".latency.csv";
    FILE *latency = std::fopen(latency_path.c_str(), "w");
    if (latency == nullptr) {
        std::perror("read_supply_timeline latency");
        return;
    }
    std::fprintf(
        latency,
        "generation,client_idx,qp_idx,wr_id,accepted_tsc,reaped_tsc,"
        "local_tsc,done_tsc,first_poll_tsc,last_empty_poll_tsc,"
        "empty_polls_before_reap,accept_to_first_poll_cycles,"
        "accept_to_reap_cycles,last_empty_to_reap_cycles,"
        "reap_to_local_cycles,local_to_done_cycles,reap_to_done_cycles\n");
    const uint64_t reserved =
        read_supply_timeline::lifecycle_record_count.load(
            std::memory_order_acquire);
    const uint64_t count =
        std::min<uint64_t>(reserved,
                          read_supply_timeline::kLifecycleRecordCapacity);
    for (uint64_t i = 0; i < count; ++i) {
        const auto &record = read_supply_timeline::lifecycle_records[i];
        std::fprintf(
            latency,
            "%llu,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%llu,"
            "%llu,%llu,%llu,%llu,%llu\n",
            static_cast<unsigned long long>(record.generation),
            record.client_idx, record.qp_idx,
            static_cast<unsigned long long>(record.wr_id),
            static_cast<unsigned long long>(record.accepted_tsc),
            static_cast<unsigned long long>(record.reaped_tsc),
            static_cast<unsigned long long>(record.local_tsc),
            static_cast<unsigned long long>(record.done_tsc),
            static_cast<unsigned long long>(record.first_poll_tsc),
            static_cast<unsigned long long>(record.last_empty_poll_tsc),
            record.empty_polls_before_reap,
            static_cast<unsigned long long>(
                record.first_poll_tsc == 0 ? 0 :
                record.first_poll_tsc - record.accepted_tsc),
            static_cast<unsigned long long>(record.reaped_tsc -
                                            record.accepted_tsc),
            static_cast<unsigned long long>(
                record.last_empty_poll_tsc == 0 ? 0 :
                record.reaped_tsc - record.last_empty_poll_tsc),
            static_cast<unsigned long long>(record.local_tsc -
                                            record.reaped_tsc),
            static_cast<unsigned long long>(record.done_tsc -
                                            record.local_tsc),
            static_cast<unsigned long long>(record.done_tsc -
                                            record.reaped_tsc));
    }
    std::fclose(latency);

    uint64_t active_entries = 0;
    for (const auto &entry : read_supply_timeline::lifecycle_table) {
        if (entry.state.load(std::memory_order_acquire) !=
            read_supply_timeline::LifecycleEmpty) {
            ++active_entries;
        }
    }
    std::fprintf(
        stderr,
        "read_supply_lifecycle path=%s period=%llu selected=%llu claimed=%llu "
        "collisions=%llu early_reap=%llu reaped=%llu local=%llu completed=%llu "
        "missing_reap=%llu missing_local=%llu out_of_order=%llu overflow=%llu "
        "active_entries=%llu\n",
        latency_path.c_str(),
        static_cast<unsigned long long>(
            read_supply_timeline::kLifecycleSamplePeriod),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_selected.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_claimed.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_collisions.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_early_reap.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_reaped_matches.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_local_matches.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_completed.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_missing_reap.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_missing_local.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_out_of_order.load()),
        static_cast<unsigned long long>(
            read_supply_timeline::lifecycle_ring_overflow.load()),
        static_cast<unsigned long long>(active_entries));
}

}  // namespace FarLib::cache
