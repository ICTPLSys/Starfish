#pragma once
#include <cerrno>
#include <cstring>
#include <sstream>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

namespace FarLib::cache {
inline void ConcurrentArrayCache::background_rebuild_work_dual() {
    using namespace ec_background;
    using Manager = SmallObjectStripeManager;
    using Status = Manager::BackgroundRebuildStatus;
    auto &shared = background_rebuild_state_;
    const uint32_t worker = shared.worker_ids.fetch_add(1);
    if (worker >= kWorkers) ERROR("ec_background_rebuild: too many workers");
    uthread::wait_for(&background_rebuild_cond_, &background_rebuild_mutex_, [&] {
        return background_rebuild_stop_.load(std::memory_order_acquire) ||
               ec_recovery_dead_endpoints_.load(std::memory_order_acquire) >= 2;
    });
    if (background_rebuild_stop_.load(std::memory_order_acquire)) return;

    const auto &cfg = ::FarLib::get_config();
    if (ec_endpoint_dead_count_ > 64 ||
        ec_recovery_dead_endpoints_.load(std::memory_order_acquire) != 2)
        ERROR("ec_background_rebuild: expected exactly two failed endpoints");
    uint64_t failed_mask = 0;
    for (size_t e = 0; e < ec_endpoint_dead_count_; ++e)
        if (ec_recovery_endpoint_is_dead(e))
            failed_mask |= uint64_t(1) << e;
    if (__builtin_popcountll(failed_mask) != 2)
        ERROR("ec_background_rebuild: inconsistent dual failure mask");

    const size_t endpoint_count = ec_endpoint_dead_count_;
    const size_t standby = cfg.ft_standby_endpoint >= 0
                                ? static_cast<size_t>(cfg.ft_standby_endpoint)
                                : endpoint_count;
    const size_t standby2 = cfg.ft_standby_endpoint2 >= 0
                                ? static_cast<size_t>(cfg.ft_standby_endpoint2)
                                : endpoint_count;
    if (standby >= endpoint_count || standby2 >= endpoint_count ||
        standby == standby2 ||
        (failed_mask & (uint64_t(1) << standby)) != 0 ||
        (failed_mask & (uint64_t(1) << standby2)) != 0)
        ERROR("ec_background_rebuild: no healthy dual standby pair");

    auto &manager = remote_allocator.small_object_stripe_manager();
    bool unclaimed = false;
    if (shared.claimed.compare_exchange_strong(unclaimed, true)) {
        shared.start_ns = ec_recovery_profile_now_ns();
        shared.scan_limit = manager.background_rebuild_scan_limit();
        for (size_t failed_endpoint = 0; failed_endpoint < endpoint_count;
             ++failed_endpoint) {
            if (!(failed_mask & (uint64_t(1) << failed_endpoint))) continue;
            std::ostringstream out;
            out << "INFO: ec_background_rebuild event=start monotonic_ns="
                << shared.start_ns << " failed_endpoint=" << failed_endpoint
                << " standby=" << standby << " standby2=" << standby2
                << " read_cap_MBps=" << cfg.ft_rebuild_bandwidth_mbps
                << " write_cap_MBps=" << cfg.ft_rebuild_bandwidth_mbps
                << " tid=" << syscall(SYS_gettid)
                << " cpu=" << sched_getcpu() << " workers=" << kWorkers
                << " pipeline_depth=" << kPipelineDepth
                << " io_slots_per_job=" << kDualOpsPerJob
                << " scratch_bytes="
                << kWorkers * kPipelineDepth * 6 * SmallObjectStripeShardSize
                << " pacing_burst_bytes=" << kBurstBytes
                << " verification=rdma_write_completion"
                << " read_policy=four_survivors_shared"
                << " range_policy=live_groups"
                << " expected_failures=2 failed_mask=" << failed_mask
                << " replacement_policy=spare_then_unused_healthy"
                << " read_accounting=first_missing_role";
            std::cout << (out.str() + "\n") << std::flush;
        }
        shared.ready.store(true, std::memory_order_release);
    } else {
        while (!shared.ready.load(std::memory_order_acquire)) uthread::yield();
    }
    {
        std::ostringstream out;
        out << "INFO: ec_background_rebuild event=worker_start worker=" << worker
            << " tid=" << syscall(SYS_gettid) << " cpu=" << sched_getcpu();
        std::cout << (out.str() + "\n") << std::flush;
    }

    constexpr size_t parts_per_job = 6; // four survivors and two outputs
    const size_t bytes = SmallObjectStripeShardSize;
    const size_t scratch_bytes = kPipelineDepth * parts_per_job * bytes;
    void *storage = nullptr;
    if (posix_memalign(&storage, 4096, scratch_bytes) != 0)
        ERROR("ec_background_rebuild: scratch allocation failed");
    std::memset(storage, 0, scratch_bytes);
    auto *mr = ibv_reg_mr(
        rdma::ClientControl::get_default()->get_protection_domain(), storage,
        scratch_bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) {
        std::free(storage);
        ERROR("ec_background_rebuild: MR failed");
    }

    struct Io {
        uint32_t slot = 0;
        size_t client = 0, qp = 0, endpoint = 0, charge_endpoint = 0;
        uint64_t deadline = 0, wr_id = 0;
        size_t byte_count = 0;
        bool accepted = false, write = false, ticket = false, armed = false;
        int result = 0;
    };
    enum class Stage { Empty, Snapshot, ReadPost, ReadWait, WritePost, WriteWait };
    struct Range {
        size_t offset = 0;
        size_t length = 0;
    };
    struct Job {
        Manager::BackgroundRebuildSet view;
        std::array<Io, kDualOpsPerJob> io;
        std::array<void *, parts_per_job> parts{};
        std::array<uint8_t, 4> selected{};
        std::array<uint64_t, kSmallObjectStripeShardCount> targets;
        std::array<uint8_t, 2> missing{};
        std::array<size_t, 2> target_endpoints{};
        uint8_t missing_count = 0;
        uint8_t posted_reads = 0;
        uint8_t posted_writes = 0;
        uint64_t stripe_id = 0;
        Stage stage = Stage::Empty;
        std::vector<Range> ranges;
        size_t range_index = 0;
        size_t selected_bytes = 0;
        Job() {
            targets.fill(::FarLib::allocator::remote::InvalidRemoteAddr);
            target_endpoints.fill(std::numeric_limits<size_t>::max());
        }
    };
    std::array<Job, kPipelineDepth> jobs;
    for (size_t j = 0; j < jobs.size(); ++j) {
        for (size_t s = 0; s < parts_per_job; ++s) {
            jobs[j].parts[s] = static_cast<uint8_t *>(storage) +
                (j * parts_per_job + s) * bytes;
        }
        for (size_t s = 0; s < kDualOpsPerJob; ++s) {
            jobs[j].io[s].slot =
                worker * kPipelineDepth * kDualOpsPerJob +
                j * kDualOpsPerJob + s;
        }
    }

    uint64_t decode_ns = 0, post_attempts = 0, pace_blocked = 0;
    uint64_t sq_full = 0, cq_polls = 0;
    uint64_t next_wake = UINT64_MAX;
    auto stopping = [&] {
        if (ec_recovery_dead_endpoints_.load(std::memory_order_acquire) != 2)
            shared.failed.store(true, std::memory_order_release);
        return shared.failed.load(std::memory_order_acquire) ||
               background_rebuild_stop_.load(std::memory_order_acquire);
    };

    auto poll_all = [&] {
        size_t pending = 0;
        struct CqKey {
            size_t client, qp, endpoint;
        };
        std::array<CqKey, kPipelineDepth * kDualOpsPerJob> polled{};
        size_t polled_count = 0;
        for (auto &job : jobs) {
            for (auto &io : job.io) {
                if (!io.accepted) continue;
                auto &completion = shared.completions[io.slot];
                int status = completion.status.load(std::memory_order_acquire);
                if (!status) {
                    bool seen = false;
                    for (size_t i = 0; i < polled_count; ++i) {
                        if (polled[i].client == io.client &&
                            polled[i].qp == io.qp &&
                            polled[i].endpoint == io.endpoint) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen) {
                        polled[polled_count++] =
                            {io.client, io.qp, io.endpoint};
                        ++cq_polls;
                        (void)check_cq_idx_with_client_idx_endpoint(
                            io.qp, io.client, io.endpoint);
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
                if (status < 0) {
                    shared.failed.store(true, std::memory_order_release);
                } else {
                    auto &total = io.write ? shared.write_bytes
                                           : shared.read_bytes;
                    total.fetch_add(io.byte_count, std::memory_order_relaxed);
                    if (io.charge_endpoint < shared.endpoint_read_bytes.size()) {
                        auto &per_endpoint = io.write
                            ? shared.endpoint_write_bytes[io.charge_endpoint]
                            : shared.endpoint_read_bytes[io.charge_endpoint];
                        per_endpoint.fetch_add(io.byte_count,
                                               std::memory_order_relaxed);
                    }
                }
            }
        }
        return pending;
    };

    auto try_post = [&](Io &io, uint64_t address, void *buffer, bool write,
                        size_t byte_count, size_t charge_endpoint) {
        if (io.accepted || io.result)
            ERROR("ec_background_rebuild: occupied I/O slot");
        if (byte_count == 0 || byte_count > bytes)
            ERROR("ec_background_rebuild: invalid I/O range");
        if (stopping()) return false;
        if (!io.ticket) {
            auto &pacer = write ? shared.writes : shared.reads;
            uint64_t ready = 0;
            if (!pacer.try_reserve(ec_recovery_profile_now_ns(), byte_count,
                                   cfg.ft_rebuild_bandwidth_mbps, kBurstBytes,
                                   &ready)) {
                ++pace_blocked;
                next_wake = std::min(next_wake, ready);
                return false;
            }
            io.ticket = true;
        }
        if (!io.armed) {
            const auto [endpoint, offset] = cfg.map_remote_addr(address);
            (void)offset;
            if (!cfg.validate_mapping(address, byte_count) ||
                ec_recovery_endpoint_is_dead(endpoint)) {
                shared.failed.store(true);
                return false;
            }
            io.client = rdma::thread_info.thread_id;
            auto *client = rdma::get_client(io.client);
            if (!client) ERROR("ec_background_rebuild: unregistered client");
            io.qp = client->get_qp_idx();
            io.endpoint = endpoint;
            io.charge_endpoint = charge_endpoint;
            io.write = write;
            io.byte_count = byte_count;
            io.wr_id = shared.completions[io.slot].arm(io.slot);
            if (!io.wr_id) ERROR("ec_background_rebuild: token exhausted");
            io.armed = true;
        }
        auto *client = rdma::get_client(io.client);
        auto *qp = client->get_endpoint_data_qp_local(io.endpoint, io.qp);
        if (!qp) {
            shared.failed.store(true);
            return false;
        }
        const auto &remote =
            rdma::ClientControl::get_default()->get_endpoint(io.endpoint);
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uintptr_t>(buffer);
        sge.length = byte_count;
        sge.lkey = mr->lkey;
        ibv_send_wr wr{}, *bad = nullptr;
        wr.wr_id = io.wr_id;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.opcode = write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
        wr.wr.rdma.remote_addr =
            remote.remote_base_addr + cfg.map_remote_addr(address).second;
        wr.wr.rdma.rkey = remote.remote_key;
        ++post_attempts;
        const int rc = ibv_post_send(qp->queue_pair, &wr, &bad);
        if (rc == ENOMEM) {
            ++sq_full;
            ++cq_polls;
            (void)check_cq_idx_with_client_idx_endpoint(io.qp, io.client,
                                                        io.endpoint);
            return false;
        }
        if (rc) {
            shared.failed.store(true);
            return false;
        }
        io.deadline = ec_recovery_profile_now_ns() + 30000000000ull;
        io.accepted = true;
        const auto in_flight = shared.in_flight.fetch_add(1) + 1;
        auto peak = shared.max_in_flight.load();
        while (peak < in_flight &&
               !shared.max_in_flight.compare_exchange_weak(peak, in_flight)) {
        }
        return true;
    };

    auto reset_range_io = [&](Job &job) {
        for (auto &io : job.io) {
            if (io.accepted)
                ERROR("ec_background_rebuild: buffer recycled with live DMA");
            const auto slot = io.slot;
            io = Io{};
            io.slot = slot;
        }
        job.posted_reads = 0;
        job.posted_writes = 0;
        job.stage = Stage::ReadPost;
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
                if (id >= shared.scan_limit) {
                    exhausted = true;
                } else {
                    job.stripe_id = id;
                    job.stage = Stage::Snapshot;
                    ++active_jobs;
                    peak_jobs = std::max(peak_jobs, active_jobs);
                    progressed = true;
                }
            }
            if (job.stage == Stage::Snapshot) {
                const auto state = manager.background_rebuild_set_view(
                    job.stripe_id, failed_mask, &job.view);
                if (state == Status::Skip) {
                    job.stage = Stage::Empty;
                    --active_jobs;
                    progressed = true;
                    continue;
                }
                if (state == Status::Busy) {
                    shared.busy_visits.fetch_add(1);
                    next_wake = std::min<uint64_t>(
                        next_wake, ec_recovery_profile_now_ns() + 1000000ull);
                    continue;
                }
                if (state == Status::Unsupported ||
                    job.view.shard_bytes != bytes) {
                    shared.failed.store(true);
                    break;
                }

                size_t selected = 0;
                job.missing_count = 0;
                uint64_t occupied_endpoints = failed_mask;
                for (uint8_t s = 0; s < kSmallObjectStripeShardCount; ++s) {
                    if (job.view.missing_mask & (1u << s)) {
                        if (job.missing_count >= 2) {
                            shared.failed.store(true);
                            break;
                        }
                        job.missing[job.missing_count++] = s;
                    } else {
                        if (job.view.endpoint[s] >= endpoint_count ||
                            job.view.endpoint[s] >= 64) {
                            shared.failed.store(true);
                            break;
                        }
                        occupied_endpoints |=
                            uint64_t(1) << job.view.endpoint[s];
                        if (selected < 4) job.selected[selected++] = s;
                    }
                }
                if (shared.failed.load(std::memory_order_acquire) ||
                    selected != 4 || job.missing_count == 0) {
                    shared.failed.store(true);
                    break;
                }

                std::array<size_t, 2> candidate_targets{standby, standby2};
                for (size_t m = 0; m < job.missing_count; ++m) {
                    size_t target = endpoint_count;
                    for (const size_t candidate : candidate_targets) {
                        if (candidate >= endpoint_count ||
                            (occupied_endpoints & (uint64_t(1) << candidate)) ||
                            ec_recovery_endpoint_is_dead(candidate)) {
                            continue;
                        }
                        target = candidate;
                        break;
                    }
                    if (target == endpoint_count) {
                        for (size_t candidate = 0; candidate < endpoint_count;
                             ++candidate) {
                            if (ec_recovery_endpoint_is_dead(candidate) ||
                                (occupied_endpoints &
                                 (uint64_t(1) << candidate))) {
                                continue;
                            }
                            target = candidate;
                            break;
                        }
                    }
                    if (target == endpoint_count) {
                        shared.failed.store(true);
                        break;
                    }
                    job.target_endpoints[m] = target;
                    occupied_endpoints |= uint64_t(1) << target;
                }
                if (shared.failed.load(std::memory_order_acquire)) break;

                job.ranges.clear();
                job.range_index = 0;
                job.selected_bytes = 0;
                if (job.view.live_groups == 0) {
                    for (size_t m = 0; m < job.missing_count; ++m) {
                        const uint8_t shard = job.missing[m];
                        auto &target = job.targets[shard];
                        target = ::FarLib::allocator::remote::remote_global_heap
                            .allocate_whole_region_on_endpoint(
                                job.target_endpoints[m]);
                        if (target ==
                            ::FarLib::allocator::remote::InvalidRemoteAddr) {
                            shared.failed.store(true);
                            break;
                        }
                    }
                    if (shared.failed.load(std::memory_order_acquire) ||
                        !manager.publish_background_rebuild_set(
                            job.view, job.targets)) {
                        shared.failed.store(true);
                        break;
                    }
                    for (size_t m = 0; m < job.missing_count; ++m) {
                        const uint8_t shard = job.missing[m];
                        const size_t endpoint = job.view.endpoint[shard];
                        if (endpoint < shared.endpoint_shards.size()) {
                            shared.endpoint_empty_reactivated[endpoint].fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        if (job.target_endpoints[m] == standby ||
                            job.target_endpoints[m] == standby2) {
                            shared.spare_shards.fetch_add(
                                1, std::memory_order_relaxed);
                        } else {
                            shared.regular_shards.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        job.targets[shard] =
                            ::FarLib::allocator::remote::InvalidRemoteAddr;
                    }
                    shared.empty_reactivated.fetch_add(1,
                                                        std::memory_order_relaxed);
                    job.stage = Stage::Empty;
                    --active_jobs;
                    progressed = true;
                    continue;
                }

                const size_t slot_size = job.view.slot_size;
                const auto &slots = job.view.live_slot_ids;
                if (slot_size == 0 || slot_size > bytes ||
                    bytes % slot_size != 0 || slots.empty() ||
                    slots.size() != job.view.live_groups ||
                    !std::is_sorted(slots.begin(), slots.end()) ||
                    std::adjacent_find(slots.begin(), slots.end()) !=
                        slots.end()) {
                    shared.failed.store(true);
                    break;
                }
                for (const uint32_t slot : slots) {
                    if (slot >= bytes / slot_size) {
                        shared.failed.store(true);
                        break;
                    }
                    const size_t offset = static_cast<size_t>(slot) * slot_size;
                    if (!job.ranges.empty() &&
                        job.ranges.back().offset +
                                job.ranges.back().length == offset) {
                        job.ranges.back().length += slot_size;
                    } else {
                        job.ranges.push_back({offset, slot_size});
                    }
                    job.selected_bytes += slot_size;
                }
                if (shared.failed.load(std::memory_order_acquire)) break;
                reset_range_io(job);
                progressed = true;
            }
            if (job.stage == Stage::ReadPost) {
                const auto range = job.ranges[job.range_index];
                while (job.posted_reads < 4) {
                    const size_t i = job.posted_reads;
                    const size_t charge_endpoint =
                        job.view.endpoint[job.missing[0]];
                    if (!try_post(
                            job.io[i],
                            job.view.original_base[job.selected[i]] +
                                range.offset,
                            static_cast<uint8_t *>(job.parts[i]) +
                                range.offset,
                            false, range.length, charge_endpoint)) {
                        break;
                    }
                    ++job.posted_reads;
                    progressed = true;
                }
                if (job.posted_reads == 4) job.stage = Stage::ReadWait;
            }
            if (job.stage == Stage::ReadWait) {
                bool ready = true;
                for (size_t i = 0; i < 4; ++i)
                    ready = ready && job.io[i].result == 1;
                if (ready) {
                    const auto range = job.ranges[job.range_index];
                    const void *survivors[4]{job.parts[0], job.parts[1],
                                             job.parts[2], job.parts[3]};
                    const auto begin = ec_recovery_profile_now_ns();
                    bool ok = true;
                    for (size_t m = 0; m < job.missing_count; ++m) {
                        ok = ok && SmallObjectStripeEncoder::rebuild_one(
                            job.missing[m], job.selected.data(), survivors,
                            job.parts[4 + m], range.length, range.offset);
                    }
                    decode_ns += ec_recovery_profile_now_ns() - begin;
                    if (!ok) {
                        shared.failed.store(true);
                        break;
                    }
                    job.stage = Stage::WritePost;
                    progressed = true;
                }
            }
            if (job.stage == Stage::WritePost) {
                const auto range = job.ranges[job.range_index];
                while (job.posted_writes < job.missing_count) {
                    const size_t m = job.posted_writes;
                    const uint8_t shard = job.missing[m];
                    auto &target = job.targets[shard];
                    if (target ==
                        ::FarLib::allocator::remote::InvalidRemoteAddr) {
                        target = ::FarLib::allocator::remote::remote_global_heap
                            .allocate_whole_region_on_endpoint(
                                job.target_endpoints[m]);
                    }
                    if (target ==
                        ::FarLib::allocator::remote::InvalidRemoteAddr) {
                        shared.failed.store(true);
                        break;
                    }
                    if (!try_post(
                            job.io[4 + m], target + range.offset,
                            static_cast<uint8_t *>(job.parts[4 + m]) +
                                range.offset,
                            true, range.length, job.view.endpoint[shard])) {
                        break;
                    }
                    ++job.posted_writes;
                    progressed = true;
                }
                if (job.posted_writes == job.missing_count)
                    job.stage = Stage::WriteWait;
            }
            if (job.stage == Stage::WriteWait) {
                bool ready = true;
                for (size_t m = 0; m < job.missing_count; ++m)
                    ready = ready && job.io[4 + m].result == 1;
                if (!ready) continue;
                if (stopping()) break;

                shared.completed_ranges.fetch_add(1, std::memory_order_relaxed);
                for (size_t m = 0; m < job.missing_count; ++m) {
                    const size_t endpoint = job.view.endpoint[job.missing[m]];
                    if (endpoint < shared.endpoint_completed_ranges.size())
                        shared.endpoint_completed_ranges[endpoint].fetch_add(
                            1, std::memory_order_relaxed);
                }
                if (++job.range_index < job.ranges.size()) {
                    reset_range_io(job);
                    progressed = true;
                    continue;
                }

                if (!manager.publish_background_rebuild_set(job.view,
                                                              job.targets)) {
                    shared.failed.store(true);
                    break;
                }
                for (size_t m = 0; m < job.missing_count; ++m) {
                    const uint8_t shard = job.missing[m];
                    const size_t endpoint = job.view.endpoint[shard];
                    if (endpoint < shared.endpoint_shards.size()) {
                        shared.endpoint_shards[endpoint].fetch_add(
                            1, std::memory_order_relaxed);
                        shared.endpoint_selected_bytes[endpoint].fetch_add(
                            job.selected_bytes, std::memory_order_relaxed);
                        shared.endpoint_skipped_bytes[endpoint].fetch_add(
                            bytes - job.selected_bytes,
                            std::memory_order_relaxed);
                        shared.endpoint_live_groups[endpoint].fetch_add(
                            job.view.live_groups, std::memory_order_relaxed);
                        shared.endpoint_live_objects[endpoint].fetch_add(
                            job.view.live_objects, std::memory_order_relaxed);
                    }
                    if (job.target_endpoints[m] == standby ||
                        job.target_endpoints[m] == standby2) {
                        shared.spare_shards.fetch_add(
                            1, std::memory_order_relaxed);
                    } else {
                        shared.regular_shards.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    job.targets[shard] =
                        ::FarLib::allocator::remote::InvalidRemoteAddr;
                }
                shared.completed.fetch_add(1, std::memory_order_relaxed);
                shared.write_completed.fetch_add(job.missing_count,
                                                std::memory_order_relaxed);
                shared.selected_bytes.fetch_add(job.selected_bytes,
                                                std::memory_order_relaxed);
                shared.skipped_bytes.fetch_add(bytes - job.selected_bytes,
                                              std::memory_order_relaxed);
                shared.live_groups.fetch_add(job.view.live_groups,
                                             std::memory_order_relaxed);
                shared.live_objects.fetch_add(job.view.live_objects,
                                              std::memory_order_relaxed);
                job.stage = Stage::Empty;
                --active_jobs;
                ++local_completed;
                progressed = true;
            }
        }
        rotate = (rotate + 1) % jobs.size();
        const auto pending = poll_all();
        if (exhausted && active_jobs == 0 && pending == 0) break;
        const auto now = ec_recovery_profile_now_ns();
        if (now - progress_at >= 1000000000ull) {
            std::ostringstream out;
            out << "INFO: ec_background_rebuild event=progress monotonic_ns="
                << now << " worker=" << worker
                << " completed_stripes=" << shared.completed.load()
                << " scan_index=" << shared.cursor.load()
                << " scan_limit=" << shared.scan_limit
                << " read_bytes=" << shared.read_bytes.load()
                << " write_bytes=" << shared.write_bytes.load()
                << " union_selected_bytes=" << shared.selected_bytes.load();
            std::cout << (out.str() + "\n") << std::flush;
            progress_at = now;
        }
        if (!progressed && next_wake != UINT64_MAX && next_wake > now)
            Fibre::usleep(std::min<uint64_t>(
                100, (next_wake - now + 999) / 1000));
        else
            uthread::yield();
    }

    while (poll_all()) uthread::yield();
    for (auto &job : jobs) {
        for (const auto target : job.targets) {
            if (target != ::FarLib::allocator::remote::InvalidRemoteAddr)
                ::FarLib::allocator::remote::remote_global_heap
                    .release_whole_region(target);
        }
    }
    if (ibv_dereg_mr(mr) != 0)
        ERROR("ec_background_rebuild: MR deregistration failed");
    std::free(storage);
    shared.decode_ns.fetch_add(decode_ns, std::memory_order_relaxed);
    shared.post_attempts.fetch_add(post_attempts,
                                   std::memory_order_relaxed);
    shared.pace_blocked.fetch_add(pace_blocked,
                                  std::memory_order_relaxed);
    shared.sq_full.fetch_add(sq_full, std::memory_order_relaxed);
    shared.cq_polls.fetch_add(cq_polls, std::memory_order_relaxed);
    {
        std::ostringstream out;
        out << "INFO: ec_background_rebuild event=worker_done worker=" << worker
            << " completed_stripes=" << local_completed
            << " peak_active_jobs=" << peak_jobs
            << " decode_elapsed_ns=" << decode_ns
            << " post_attempts=" << post_attempts
            << " pace_blocked=" << pace_blocked << " sq_full=" << sq_full
            << " cq_polls=" << cq_polls << " in_flight=0";
        std::cout << (out.str() + "\n") << std::flush;
    }

    if (shared.workers_done.fetch_add(1, std::memory_order_acq_rel) + 1 !=
        kWorkers)
        return;
    const bool cancelled =
        background_rebuild_stop_.load(std::memory_order_acquire);
    if (ec_recovery_dead_endpoints_.load(std::memory_order_acquire) != 2)
        shared.failed.store(true);
    const auto remaining = manager.count_rebuild_set_remaining(failed_mask);
    if (remaining && !cancelled) shared.failed.store(true);
    const bool failed = shared.failed.load(std::memory_order_acquire);
    const auto done = ec_recovery_profile_now_ns();
    if (shared.in_flight.load() != 0)
        ERROR("ec_background_rebuild: workers joined with live I/O");
    if (!failed && !cancelled && remaining == 0)
        shared.fully_rebuilt.store(true, std::memory_order_release);

    for (size_t failed_endpoint = 0; failed_endpoint < endpoint_count;
         ++failed_endpoint) {
        if (!(failed_mask & (uint64_t(1) << failed_endpoint))) continue;
        std::ostringstream out;
        out << "INFO: ec_background_rebuild event="
            << (failed ? "failed" : cancelled ? "cancelled" : "done")
            << " monotonic_ns=" << done << " start_ns=" << shared.start_ns
            << " duration_ns=" << done - shared.start_ns
            << " failed_endpoint=" << failed_endpoint << " standby=" << standby
            << " standby2=" << standby2
            << " completed_stripes="
            << shared.endpoint_shards[failed_endpoint].load()
            << " write_completed_stripes="
            << shared.endpoint_shards[failed_endpoint].load()
            << " verified_stripes=0"
            << " live_groups_at_copy="
            << shared.endpoint_live_groups[failed_endpoint].load()
            << " live_internal_fragments_at_copy="
            << shared.endpoint_live_objects[failed_endpoint].load()
            << " selected_bytes="
            << shared.endpoint_selected_bytes[failed_endpoint].load()
            << " read_bytes="
            << shared.endpoint_read_bytes[failed_endpoint].load()
            << " write_bytes="
            << shared.endpoint_write_bytes[failed_endpoint].load()
            << " skipped_bytes="
            << shared.endpoint_skipped_bytes[failed_endpoint].load()
            << " completed_ranges="
            << shared.endpoint_completed_ranges[failed_endpoint].load()
            << " empty_reactivated="
            << shared.endpoint_empty_reactivated[failed_endpoint].load()
            << " union_stripes=" << shared.completed.load()
            << " union_selected_bytes=" << shared.selected_bytes.load()
            << " union_skipped_bytes=" << shared.skipped_bytes.load()
            << " global_read_bytes=" << shared.read_bytes.load()
            << " global_write_bytes=" << shared.write_bytes.load()
            << " target_spare_shards=" << shared.spare_shards.load()
            << " target_regular_shards=" << shared.regular_shards.load()
            << " scan_limit=" << shared.scan_limit
            << " passes=1 busy_visits=" << shared.busy_visits.load()
            << " remaining=" << remaining << " failed=" << (failed ? 1 : 0)
            << " stage=final_scan workers=" << kWorkers
            << " pipeline_depth=" << kPipelineDepth
            << " max_in_flight=" << shared.max_in_flight.load()
            << " prefetched=0 io_slots_per_job=" << kDualOpsPerJob
            << " scratch_bytes=" << kWorkers * scratch_bytes
            << " pacing_burst_bytes=" << kBurstBytes
            << " decode_elapsed_ns_sum=" << shared.decode_ns.load()
            << " post_attempts=" << shared.post_attempts.load()
            << " pace_blocked=" << shared.pace_blocked.load()
            << " sq_full=" << shared.sq_full.load()
            << " cq_polls=" << shared.cq_polls.load()
            << " in_flight=0 expected_failures=2 failed_mask=" << failed_mask
            << " read_policy=four_survivors_shared"
            << " read_accounting=first_missing_role range_policy=live_groups";
        std::cout << (out.str() + "\n") << std::flush;
    }
}
} // namespace FarLib::cache
