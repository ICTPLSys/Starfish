#pragma once
#include <cerrno>
#include <cstring>
#include <sstream>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

namespace FarLib::cache {
inline void ConcurrentArrayCache::start_background_rebuild() {
    if (!::FarLib::get_config().ft_background_rebuild) return;
    if (!background_cluster_)
        ERROR("ec_background_rebuild requires the separate background cluster");
    for (auto &thread : background_rebuild_threads_)
        thread = uthread::create_on<true>(
            *background_cluster_,
            +[](ConcurrentArrayCache *cache) {
                if (::FarLib::get_config().ft_background_rebuild_failures == 2)
                    cache->background_rebuild_work_dual();
                else
                    cache->background_rebuild_work();
            },
            this, "background EC rebuild");
}
inline void ConcurrentArrayCache::stop_background_rebuild() {
    if (!background_rebuild_threads_[0]) return;
    background_rebuild_stop_.store(true, std::memory_order_release);
    uthread::notify_all(&background_rebuild_cond_, &background_rebuild_mutex_);
    for (auto &thread : background_rebuild_threads_) uthread::join(std::move(thread));
    std::ostringstream out;
    out << "INFO: ec_background_rebuild event=joined workers=" << ec_background::kWorkers
        << " redirected_reads=" << background_rebuild_redirected_reads_.load();
    std::cout << (out.str() + "\n") << std::flush;
}
inline bool ConcurrentArrayCache::handle_background_rebuild_complete(const ibv_wc &wc) {
    if (!::FarLib::get_config().ft_background_rebuild ||
        !ec_background::is_wr_id(wc.wr_id)) return false;
    const auto slot = ec_background::slot_from_wr_id(wc.wr_id);
    if (slot >= ec_background::kSlots)
        ERROR("ec_background_rebuild: invalid completion slot");
    auto &completion = background_rebuild_state_.completions[slot];
    if (!completion.matches(wc.wr_id)) return true;
    if (wc.status != IBV_WC_SUCCESS) note_ec_recovery_error_wc(wc);
    completion.complete(wc.status == IBV_WC_SUCCESS); // last callback access
    return true;
}
inline void ConcurrentArrayCache::background_rebuild_work() {
    using namespace ec_background;
    using Manager = SmallObjectStripeManager;
    using Status = Manager::BackgroundRebuildStatus;
    auto &shared = background_rebuild_state_;
    const uint32_t worker = shared.worker_ids.fetch_add(1);
    if (worker >= kWorkers) ERROR("ec_background_rebuild: too many workers");
    uthread::wait_for(&background_rebuild_cond_, &background_rebuild_mutex_, [&] {
        return background_rebuild_stop_.load(std::memory_order_acquire) ||
               background_rebuild_failed_endpoint_.load(std::memory_order_acquire) >= 0;
    });
    if (background_rebuild_stop_.load(std::memory_order_acquire)) return;
    const auto &cfg = ::FarLib::get_config();
    const size_t failed_endpoint = background_rebuild_failed_endpoint_.load();
    const size_t standby = static_cast<size_t>(cfg.ft_standby_endpoint);
    auto &manager = remote_allocator.small_object_stripe_manager();
    bool unclaimed = false;
    if (shared.claimed.compare_exchange_strong(unclaimed, true)) {
        shared.start_ns = ec_recovery_profile_now_ns();
        shared.scan_limit = manager.background_rebuild_scan_limit();
        std::ostringstream out;
        out << "INFO: ec_background_rebuild event=start monotonic_ns=" << shared.start_ns
            << " failed_endpoint=" << failed_endpoint << " standby=" << standby
            << " read_cap_MBps=" << cfg.ft_rebuild_bandwidth_mbps
            << " write_cap_MBps=" << cfg.ft_rebuild_bandwidth_mbps
            << " tid=" << syscall(SYS_gettid) << " cpu=" << sched_getcpu()
            << " workers=" << kWorkers << " pipeline_depth=" << kPipelineDepth
            << " io_slots_per_job=" << kOpsPerJob
            << " scratch_bytes=" << kWorkers * kPipelineDepth * 5 * SmallObjectStripeShardSize
            << " pacing_burst_bytes=" << kBurstBytes
            << " verification=rdma_write_completion read_policy=four_survivors"
            << " range_policy=" << (cfg.is_carbink_mode() ? "live_groups" : "whole_shard");
        std::cout << (out.str() + "\n") << std::flush;
        shared.ready.store(true, std::memory_order_release);
    } else while (!shared.ready.load(std::memory_order_acquire)) uthread::yield();
    {
        std::ostringstream out;
        out << "INFO: ec_background_rebuild event=worker_start worker=" << worker
            << " tid=" << syscall(SYS_gettid) << " cpu=" << sched_getcpu();
        std::cout << (out.str() + "\n") << std::flush;
    }
    if (standby == failed_endpoint || ec_recovery_endpoint_is_dead(standby))
        ERROR("ec_background_rebuild: no healthy standby");
    constexpr size_t parts_per_job = 5; // four survivors and one output
    const size_t bytes = SmallObjectStripeShardSize;
    const size_t scratch_bytes = kPipelineDepth * parts_per_job * bytes;
    void *storage = nullptr;
    if (posix_memalign(&storage, 4096, scratch_bytes) != 0)
        ERROR("ec_background_rebuild: scratch allocation failed");
    std::memset(storage, 0, scratch_bytes);
    auto *mr = ibv_reg_mr(rdma::ClientControl::get_default()->get_protection_domain(),
                         storage, scratch_bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) { std::free(storage); ERROR("ec_background_rebuild: MR failed"); }
    struct Io {
        uint32_t slot = 0;
        size_t client = 0, qp = 0, endpoint = 0;
        uint64_t deadline = 0, wr_id = 0;
        size_t byte_count = 0;
        bool accepted = false, write = false, ticket = false, armed = false;
        int result = 0;
    };
    enum class Stage { Empty, Snapshot, ReadPost, ReadWait, WritePost, WriteWait };
    struct Range { size_t offset, length; };
    struct Job {
        Manager::BackgroundRebuildStripe view;
        std::array<Io, kOpsPerJob> io;
        std::array<void *, parts_per_job> parts{};
        std::array<uint8_t, 4> selected{};
        uint64_t stripe_id = 0;
        uint64_t target = ::FarLib::allocator::remote::InvalidRemoteAddr;
        Stage stage = Stage::Empty;
        uint8_t posted_reads = 0;
        std::vector<Range> ranges;
        size_t range_index = 0, selected_bytes = 0;
    };
    std::array<Job, kPipelineDepth> jobs;
    for (size_t j = 0; j < jobs.size(); ++j) {
        for (size_t s = 0; s < parts_per_job; ++s)
            jobs[j].parts[s] = static_cast<uint8_t *>(storage) + (j * parts_per_job + s) * bytes;
        for (size_t s = 0; s < kOpsPerJob; ++s)
            jobs[j].io[s].slot = worker * kPipelineDepth * kOpsPerJob + j * kOpsPerJob + s;
    }
    auto reset_range_io = [&](Job &job) {
        for (auto &io : job.io) {
            if (io.accepted) ERROR("ec_background_rebuild: buffer recycled with live DMA");
            const auto slot = io.slot;
            io = Io{};
            io.slot = slot;
        }
        job.posted_reads = 0;
        job.stage = Stage::ReadPost;
    };
    uint64_t decode_ns = 0, post_attempts = 0, pace_blocked = 0;
    uint64_t sq_full = 0, cq_polls = 0;
    uint64_t next_wake = UINT64_MAX;
    auto stopping = [&] {
        if (ec_recovery_dead_endpoints_.load(std::memory_order_acquire) != 1)
            shared.failed.store(true, std::memory_order_release);
        return shared.failed.load(std::memory_order_acquire) ||
               background_rebuild_stop_.load(std::memory_order_acquire);
    };
    auto poll_all = [&] {
        size_t pending = 0;
        struct CqKey { size_t client, qp, endpoint; };
        std::array<CqKey, kPipelineDepth * kOpsPerJob> polled{};
        size_t polled_count = 0;
        for (auto &job : jobs) for (auto &io : job.io) {
            if (!io.accepted) continue;
            auto &completion = shared.completions[io.slot];
            int status = completion.status.load(std::memory_order_acquire);
            if (!status) {
                bool seen = false;
                for (size_t i = 0; i < polled_count; ++i)
                    if (polled[i].client == io.client && polled[i].qp == io.qp &&
                        polled[i].endpoint == io.endpoint) { seen = true; break; }
                if (!seen) {
                    polled[polled_count++] = {io.client, io.qp, io.endpoint};
                    ++cq_polls;
                    (void)check_cq_idx_with_client_idx_endpoint(io.qp, io.client, io.endpoint);
                }
                status = completion.status.load(std::memory_order_acquire);
            }
            if (!status) {
                if (ec_recovery_profile_now_ns() > io.deadline)
                    ERROR("ec_background_rebuild: accepted WR failed to drain");
                ++pending;
                continue;
            }
            io.result = status;
            io.accepted = false;
            shared.in_flight.fetch_sub(1, std::memory_order_relaxed);
            if (status < 0) shared.failed.store(true, std::memory_order_release);
            else (io.write ? shared.write_bytes : shared.read_bytes)
                     .fetch_add(io.byte_count, std::memory_order_relaxed);
        }
        return pending;
    };
    // Never park a whole fibre on one job's pacing, READ, or WRITE.
    // A denied pacing attempt owns no future credit. SQ-full retry retains
    // one already-charged ticket, so the intended WR is charged only once.
    auto try_post = [&](Io &io, uint64_t address, void *buffer, bool write,
                        size_t byte_count) {
        if (io.accepted || io.result) ERROR("ec_background_rebuild: occupied I/O slot");
        if (byte_count == 0 || byte_count > bytes)
            ERROR("ec_background_rebuild: invalid I/O range");
        if (stopping()) return false;
        if (!io.ticket) {
            auto &pacer = write ? shared.writes : shared.reads;
            uint64_t ready = 0;
            if (!pacer.try_reserve(ec_recovery_profile_now_ns(), byte_count,
                                  cfg.ft_rebuild_bandwidth_mbps, kBurstBytes, &ready)) {
                ++pace_blocked;
                next_wake = std::min(next_wake, ready);
                return false;
            }
            io.ticket = true;
        }
        if (!io.armed) {
            const auto [endpoint, offset] = cfg.map_remote_addr(address);
            (void)offset;
            if (!cfg.validate_mapping(address, byte_count) || ec_recovery_endpoint_is_dead(endpoint)) {
                shared.failed.store(true); return false;
            }
            io.client = rdma::thread_info.thread_id;
            auto *client = rdma::get_client(io.client);
            if (!client) ERROR("ec_background_rebuild: unregistered client");
            io.qp = client->get_qp_idx();
            io.endpoint = endpoint;
            io.write = write;
            io.byte_count = byte_count;
            io.wr_id = shared.completions[io.slot].arm(io.slot);
            if (!io.wr_id) ERROR("ec_background_rebuild: token exhausted");
            io.armed = true;
        }
        auto *client = rdma::get_client(io.client);
        auto *qp = client->get_endpoint_data_qp_local(io.endpoint, io.qp);
        if (!qp) { shared.failed.store(true); return false; }
        const auto &remote = rdma::ClientControl::get_default()->get_endpoint(io.endpoint);
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uintptr_t>(buffer);
        sge.length = byte_count; sge.lkey = mr->lkey;
        ibv_send_wr wr{}, *bad = nullptr;
        wr.wr_id = io.wr_id;
        wr.sg_list = &sge; wr.num_sge = 1; wr.send_flags = IBV_SEND_SIGNALED;
        wr.opcode = write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
        wr.wr.rdma.remote_addr = remote.remote_base_addr + cfg.map_remote_addr(address).second;
        wr.wr.rdma.rkey = remote.remote_key;
        ++post_attempts;
        const int rc = ibv_post_send(qp->queue_pair, &wr, &bad);
        if (rc == ENOMEM) {
            ++sq_full; ++cq_polls;
            (void)check_cq_idx_with_client_idx_endpoint(io.qp, io.client, io.endpoint);
            return false;
        }
        if (rc) { shared.failed.store(true); return false; }
        io.deadline = ec_recovery_profile_now_ns() + 30000000000ull;
        io.accepted = true;
        const auto in_flight = shared.in_flight.fetch_add(1) + 1;
        auto peak = shared.max_in_flight.load();
        while (peak < in_flight && !shared.max_in_flight.compare_exchange_weak(peak, in_flight)) {}
        return true;
    };
    bool exhausted = false;
    size_t active_jobs = 0, peak_jobs = 0, rotate = 0;
    uint64_t local_completed = 0, progress_at = shared.start_ns;
    while (!stopping()) {
        (void)poll_all();
        if (stopping()) break;
        bool progressed = false;
        next_wake = UINT64_MAX;
        for (size_t n = 0; n < jobs.size(); ++n) {
            if (stopping()) break;
            Job &job = jobs[(rotate + n) % jobs.size()];
            if (job.stage == Stage::Empty && !exhausted) {
                const auto id = shared.cursor.fetch_add(1);
                if (id >= shared.scan_limit) exhausted = true;
                else {
                    job.stripe_id = id;
                    job.stage = Stage::Snapshot;
                    ++active_jobs;
                    peak_jobs = std::max(peak_jobs, active_jobs);
                    progressed = true;
                }
            }
            if (job.stage == Stage::Snapshot) {
                const auto state = manager.background_rebuild_view(
                    job.stripe_id, failed_endpoint, &job.view);
                if (state == Status::Skip) {
                    job.stage = Stage::Empty;
                    --active_jobs;
                    progressed = true;
                    continue;
                }
                if (state == Status::Busy) {
                    shared.busy_visits.fetch_add(1);
                    next_wake = std::min<uint64_t>(next_wake, ec_recovery_profile_now_ns() + 1000000ull);
                    continue;
                }
                if (state == Status::Unsupported || job.view.shard_bytes != bytes) {
                    shared.failed.store(true); break;
                }
                if (cfg.is_carbink_mode() && job.view.live_groups == 0) {
                    // This frozen stripe has no live owner and no pending
                    // WRITE. No data needs reconstruction: reserve its spare
                    // role, then revalidate emptiness before publication.
                    // Future normal groups overwrite all six fragments;
                    // shadow groups are zeroed before compaction writes.
                    job.target = ::FarLib::allocator::remote::remote_global_heap
                                     .allocate_whole_region_on_endpoint(standby);
                    if (job.target == ::FarLib::allocator::remote::InvalidRemoteAddr ||
                        !manager.publish_background_rebuild(job.view, job.target)) {
                        shared.failed.store(true); break;
                    }
                    job.target = ::FarLib::allocator::remote::InvalidRemoteAddr;
                    shared.empty_reactivated.fetch_add(1);
                    job.stage = Stage::Empty;
                    --active_jobs;
                    progressed = true;
                    continue;
                }
                size_t selected = 0;
                for (uint8_t s = 0; s < 6 && selected < 4; ++s)
                    if (s != job.view.failed_shard) job.selected[selected++] = s;
                if (selected != 4) { shared.failed.store(true); break; }
                job.ranges.clear();
                job.range_index = 0;
                job.selected_bytes = 0;
                if (cfg.is_carbink_mode()) {
                    // Groups are independent codewords. Keep the original
                    // offsets and coalesce only adjacent live slots; never
                    // include a free/dead group just to enlarge a request.
                    const auto slot_size = job.view.slot_size;
                    const auto &slots = job.view.live_slot_ids;
                    if (slot_size == 0 || slot_size > bytes || bytes % slot_size ||
                        slots.empty() || slots.size() != job.view.live_groups ||
                        !std::is_sorted(slots.begin(), slots.end()) ||
                        std::adjacent_find(slots.begin(), slots.end()) != slots.end()) {
                        shared.failed.store(true); break;
                    }
                    for (const auto slot : slots) {
                        if (slot >= bytes / slot_size) {
                            shared.failed.store(true); break;
                        }
                        const size_t offset = slot * slot_size;
                        if (!job.ranges.empty() &&
                            job.ranges.back().offset + job.ranges.back().length == offset)
                            job.ranges.back().length += slot_size;
                        else job.ranges.push_back({offset, slot_size});
                        job.selected_bytes += slot_size;
                    }
                    if (shared.failed.load()) break;
                } else {
                    // Preserve the existing non-Carbink recovery path.
                    job.ranges.push_back({0, bytes});
                    job.selected_bytes = bytes;
                }
                reset_range_io(job);
                progressed = true;
            }
            if (job.stage == Stage::ReadPost) {
                const auto range = job.ranges[job.range_index];
                while (job.posted_reads < 4) {
                    const size_t i = job.posted_reads;
                    if (!try_post(job.io[i], job.view.original_base[job.selected[i]] + range.offset,
                                  static_cast<uint8_t *>(job.parts[i]) + range.offset,
                                  false, range.length)) break;
                    ++job.posted_reads;
                    progressed = true;
                }
                if (job.posted_reads == 4) job.stage = Stage::ReadWait;
            }
            if (job.stage == Stage::ReadWait) {
                bool ready = true;
                for (size_t i = 0; i < 4; ++i) ready = ready && job.io[i].result == 1;
                if (ready) {
                    const auto range = job.ranges[job.range_index];
                    const void *survivors[4]{job.parts[0], job.parts[1],
                                             job.parts[2], job.parts[3]};
                    const auto begin = ec_recovery_profile_now_ns();
                    const bool ok = SmallObjectStripeEncoder::rebuild_one(
                        job.view.failed_shard, job.selected.data(), survivors,
                        job.parts[4], range.length, range.offset);
                    decode_ns += ec_recovery_profile_now_ns() - begin;
                    if (!ok) { shared.failed.store(true); break; }
                    job.stage = Stage::WritePost;
                    progressed = true;
                }
            }
            if (job.stage == Stage::WritePost) {
                if (job.target == ::FarLib::allocator::remote::InvalidRemoteAddr)
                    job.target = ::FarLib::allocator::remote::remote_global_heap
                                     .allocate_whole_region_on_endpoint(standby);
                if (job.target == ::FarLib::allocator::remote::InvalidRemoteAddr) {
                    shared.failed.store(true); break;
                }
                const auto range = job.ranges[job.range_index];
                if (try_post(job.io[4], job.target + range.offset,
                             static_cast<uint8_t *>(job.parts[4]) + range.offset,
                             true, range.length)) {
                    job.stage = Stage::WriteWait;
                    progressed = true;
                }
            }
            if (job.stage == Stage::WriteWait && job.io[4].result == 1) {
                if (stopping()) break;
                shared.completed_ranges.fetch_add(1);
                if (++job.range_index < job.ranges.size()) {
                    reset_range_io(job);
                    progressed = true;
                    continue;
                }
                // Publish only after ALL selected ranges have successful
                // WRITE CQEs. The manager rechecks that no new live slot has
                // appeared while the stripe was frozen. Skipped slots are
                // overwritten by normal allocation or zeroed by shadow reuse.
                if (!manager.publish_background_rebuild(job.view, job.target)) {
                    shared.failed.store(true); break;
                }
                job.target = ::FarLib::allocator::remote::InvalidRemoteAddr;
                job.stage = Stage::Empty;
                --active_jobs;
                ++local_completed;
                shared.completed.fetch_add(1);
                shared.write_completed.fetch_add(1);
                shared.selected_bytes.fetch_add(job.selected_bytes);
                shared.skipped_bytes.fetch_add(bytes - job.selected_bytes);
                shared.live_groups.fetch_add(job.view.live_groups);
                shared.live_objects.fetch_add(job.view.live_objects);
                progressed = true;
            }
        }
        rotate = (rotate + 1) % jobs.size();
        const auto pending = poll_all();
        if (exhausted && active_jobs == 0 && pending == 0) break;
        const auto now = ec_recovery_profile_now_ns();
        if (now - progress_at >= 1000000000ull) {
            std::ostringstream out;
            out << "INFO: ec_background_rebuild event=progress monotonic_ns=" << now
                << " worker=" << worker << " completed_stripes=" << shared.completed.load()
                << " scan_index=" << shared.cursor.load() << " scan_limit=" << shared.scan_limit
                << " read_bytes=" << shared.read_bytes.load()
                << " write_bytes=" << shared.write_bytes.load();
            out << " empty_reactivated=" << shared.empty_reactivated.load();
            std::cout << (out.str() + "\n") << std::flush;
            progress_at = now;
        }
        // A bounded burst replenishes while parked; unlike future-ticket
        // sleeps, scheduling delays do not create permanent submission holes.
        if (!progressed && next_wake != UINT64_MAX && next_wake > now)
            Fibre::usleep(std::min<uint64_t>(100, (next_wake - now + 999) / 1000));
        else uthread::yield();
    }
    while (poll_all()) uthread::yield();
    for (auto &job : jobs)
        if (job.target != ::FarLib::allocator::remote::InvalidRemoteAddr)
            ::FarLib::allocator::remote::remote_global_heap.release_whole_region(job.target);
    if (ibv_dereg_mr(mr) != 0) ERROR("ec_background_rebuild: MR deregistration failed");
    std::free(storage);
    shared.decode_ns.fetch_add(decode_ns);
    shared.post_attempts.fetch_add(post_attempts);
    shared.pace_blocked.fetch_add(pace_blocked);
    shared.sq_full.fetch_add(sq_full);
    shared.cq_polls.fetch_add(cq_polls);
    {
        std::ostringstream out;
        out << "INFO: ec_background_rebuild event=worker_done worker=" << worker
            << " completed_stripes=" << local_completed << " peak_active_jobs=" << peak_jobs
            << " decode_elapsed_ns=" << decode_ns << " post_attempts=" << post_attempts
            << " pace_blocked=" << pace_blocked << " sq_full=" << sq_full
            << " cq_polls=" << cq_polls << " in_flight=0";
        std::cout << (out.str() + "\n") << std::flush;
    }
    if (shared.workers_done.fetch_add(1, std::memory_order_acq_rel) + 1 != kWorkers) return;
    const bool cancelled = background_rebuild_stop_.load(std::memory_order_acquire);
    if (ec_recovery_dead_endpoints_.load(std::memory_order_acquire) != 1) shared.failed.store(true);
    const auto remaining = manager.count_rebuild_remaining(failed_endpoint);
    if (remaining && !cancelled) shared.failed.store(true);
    const bool failed = shared.failed.load(std::memory_order_acquire);
    const auto done = ec_recovery_profile_now_ns();
    if (shared.in_flight.load() != 0) ERROR("ec_background_rebuild: workers joined with live I/O");
    if (!failed && !cancelled && remaining == 0)
        shared.fully_rebuilt.store(true, std::memory_order_release);
    std::ostringstream out;
    out << "INFO: ec_background_rebuild event="
        << (failed ? "failed" : cancelled ? "cancelled" : "done")
        << " monotonic_ns=" << done << " start_ns=" << shared.start_ns
        << " duration_ns=" << done - shared.start_ns
        << " failed_endpoint=" << failed_endpoint << " standby=" << standby
        << " completed_stripes=" << shared.completed.load()
        << " write_completed_stripes=" << shared.write_completed.load()
        << " verified_stripes=0"
        << " live_groups_at_copy=" << shared.live_groups.load()
        << " live_internal_fragments_at_copy=" << shared.live_objects.load()
        << " read_bytes=" << shared.read_bytes.load()
        << " write_bytes=" << shared.write_bytes.load()
        << " range_policy=" << (cfg.is_carbink_mode() ? "live_groups" : "whole_shard")
        << " selected_bytes=" << shared.selected_bytes.load()
        << " skipped_bytes=" << shared.skipped_bytes.load()
        << " completed_ranges=" << shared.completed_ranges.load()
        << " empty_reactivated=" << shared.empty_reactivated.load()
        << " scan_limit=" << shared.scan_limit << " passes=1"
        << " busy_visits=" << shared.busy_visits.load()
        << " remaining=" << remaining << " failed=" << (failed ? 1 : 0)
        << " stage=final_scan workers=" << kWorkers << " pipeline_depth=" << kPipelineDepth
        << " max_in_flight=" << shared.max_in_flight.load() << " prefetched=0"
        << " io_slots_per_job=" << kOpsPerJob
        << " scratch_bytes=" << kWorkers * scratch_bytes
        << " pacing_burst_bytes=" << kBurstBytes
        << " decode_elapsed_ns_sum=" << shared.decode_ns.load()
        << " post_attempts=" << shared.post_attempts.load()
        << " pace_blocked=" << shared.pace_blocked.load()
        << " sq_full=" << shared.sq_full.load()
        << " cq_polls=" << shared.cq_polls.load() << " in_flight=0";
    std::cout << (out.str() + "\n") << std::flush;
}
} // namespace FarLib::cache
