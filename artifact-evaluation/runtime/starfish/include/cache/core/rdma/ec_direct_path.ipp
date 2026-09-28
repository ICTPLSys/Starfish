#pragma once

namespace FarLib::cache {

inline void ConcurrentArrayCache::init_ec_direct_writes() {
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode() || !config.exclusive_cache) return;
#ifdef ASSERT_ALL_LOCAL
    return; // This debug mode bypasses the mutable-access slow path.
#endif
    if (const char *value = std::getenv("FARLIB_EC_DIRECT_WRITE"))
        if (std::strcmp(value, "0") == 0) return;
    auto *control = rdma::ClientControl::get_default();
    if (!control) ERROR("ec_direct: missing RDMA protection domain");
    using Bank = ec_batch::EcDirectWriteBank;
    const auto allocate = [](void *context, size_t bytes, Bank::RegisteredBank *out) {
        void *base = nullptr;
        if (posix_memalign(&base, 4096, bytes) != 0) return false;
        // First-touch the permanent bank on the application's bound NUMA node.
        std::memset(base, 0, bytes);
        auto *mr = ibv_reg_mr(static_cast<ibv_pd *>(context), base, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr) { std::free(base); return false; }
        out->base = base; out->bytes = bytes; out->lkey = mr->lkey; out->registration = mr;
        return true;
    };
    const auto release = [](void *, Bank::RegisteredBank &bank) {
        if (ibv_dereg_mr(static_cast<ibv_mr *>(bank.registration)) != 0)
            ERROR("ec_direct: cannot deregister parity bank");
        std::free(bank.base);
    };
    ec_persistent_owner_count_ = std::max<size_t>(1, config.evacuate_thread_cnt);
    ec_direct_bank_.reset(new Bank(ec_persistent_owner_count_));
    if (!ec_direct_bank_->init(control->get_protection_domain(), allocate, release))
        ERROR("ec_direct: permanent parity bank initialization failed");
    ec_persistent_buffers_.reset(new EvictBufferSet[ec_persistent_owner_count_]);
    for (size_t owner = 0; owner < ec_persistent_owner_count_; ++owner) {
        auto &buffers = ec_persistent_buffers_[owner];
        buffers.init(config.server_count);
        buffers.ec_cache = this;
        buffers.direct_owner = owner;
        buffers.ec_writes.reset(new EvictBufferSet::EcEndpointWrites[config.server_count]);
        buffers.direct_builder.reset(new ec_batch::EcDirectGroupBuilder(
            &remote_allocator.small_object_stripe_manager(), ec_direct_bank_.get(), owner,
            config.behavior_group));
    }
    std::cout << "INFO: ec_direct enabled=1 data_copy=0 persistent_rounds=1"
              << " fixed_parity_token=1 owners=" << ec_persistent_owner_count_
              << " slots_per_owner=128 parity_slot_bytes=8192 zero_pad_bytes=4096"
              << " bank_init_count=" << ec_persistent_owner_count_ << std::endl;
}

inline ConcurrentArrayCache::EvictBufferSet *
ConcurrentArrayCache::persistent_evict_buffers(size_t owner) {
    if (!ec_persistent_buffers_) return nullptr;
    if (owner >= ec_persistent_owner_count_) ERROR("ec_direct: invalid logical worker");
    return &ec_persistent_buffers_[owner];
}

// Only conflicts with a borrowed source take this path. Normal mutable LOCAL
// accesses already have dirty=1 and never reach it. Rescue is kept clean until
// all six DMA operations finish; read-only rescue remains entirely nonblocking.
inline void ConcurrentArrayCache::wait_ec_write_source(FarObjectEntry &entry) {
    if (!ec_direct_bank_) return;
    const auto initial = entry.load_state(std::memory_order_acquire);
    if (initial.state != LOCAL && initial.state != MARKED && initial.state != EVICTING) return;
    auto *address = entry.local_addr();
    if (!address) return;
    auto *block = static_cast<::FarLib::allocator::BlockHead *>(address) - 1;
    if (!::FarLib::allocator::ec_write_source_borrowed(block)) return;
    ec_direct_write_waits_.fetch_add(1, std::memory_order_relaxed);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    size_t passes = 0;
    while (::FarLib::allocator::ec_write_source_borrowed(block)) {
        auto state = entry.load_state(std::memory_order_acquire);
        if (!state.invalid && state.state == EVICTING) {
            auto rescued = state;
            rescued.state = LOCAL;
            rescued.inc_hotness();
            (void)entry.cas_state_weak(state, rescued);
        }
        const size_t owner_client = entry.get_client_idx();
        if (auto *client = rdma::get_client(owner_client)) {
            for (size_t qp = 0; qp < client->get_local_qp_count(); ++qp)
                for (size_t ep = 0; ep < client->get_endpoint_count(); ++ep)
                    (void)check_cq_idx_with_client_idx_endpoint(qp, owner_client, ep);
        }
        if ((++passes & 63u) == 0) {
            // A logical producer can migrate before posting a partial group.
            // Its original entry client hint then need not name the sender CQ.
            (void)ec_batch_poll_all_cqs_once();
            if (std::chrono::steady_clock::now() >= deadline)
                ERROR("ec_direct: mutable source wait exceeded 30s");
            // A producer can still own an unsealed group. In a shared fibre
            // pool it must be allowed to run; never spin all workers forever.
            uthread::yield();
        }
    }
}

inline bool ConcurrentArrayCache::stage_ec_direct_object(
    EvictBufferSet &buffers, void *data, size_t size, FarObjectEntry *entry,
    uint32_t behavior_group, bool allow_no_capacity, bool recovery_fallback) {
    profile::evict_breakdown::Scope scope(buffers.breakdown,
        profile::evict_breakdown::Stage::StageControl);
    auto &builder = *buffers.direct_builder;
    const size_t client_idx = rdma::thread_info.thread_id;
    auto *client = rdma::get_client(client_idx);
    if (!client) ERROR("ec_direct: missing client");
    const size_t qp_idx = client->get_qp_idx();
    if (recovery_fallback) {
        const auto &config = ::FarLib::get_config();
        if (!config.ft_rmw_read_failure_fallback || !config.ft_incremental_one_sided)
            ERROR("ec_direct: recovery admission is disabled");
        // Keep failure-only replacement capacity separate from a partially
        // filled normal backup-growth group. This posts only already-published
        // sources; the new source is not yet in the builder.
        flush_ec_direct_groups(buffers, client_idx);
    }
    bool waiting = false, consumed = false;
    size_t progress_passes = 0;
    std::chrono::steady_clock::time_point deadline;
    auto progress = [&] {
        (void)post_ec_direct_pending(buffers, client_idx, qp_idx);
        {
            profile::evict_breakdown::Scope polling(buffers.breakdown,
                profile::evict_breakdown::Stage::CqProcess);
            (void)check_cq_idx_with_client_idx(qp_idx, client_idx);
            if ((++progress_passes & 7u) == 0)
                (void)ec_batch_poll_all_cqs_once();
        }
        const auto now = std::chrono::steady_clock::now();
        if (!waiting) { waiting = true; deadline = now + std::chrono::seconds(30); }
        if (now >= deadline) ERROR("ec_direct: bank/group backpressure exceeded 30s");
        profile::evict_breakdown::Scope yielding(buffers.breakdown,
            profile::evict_breakdown::Stage::BackpressureYield);
        uthread::yield();
    };
    while (true) {
        uint64_t address = ::FarLib::allocator::remote::InvalidRemoteAddr;
        const auto status = consumed ? builder.flush()
            : builder.add_object(data, size, &address, behavior_group, recovery_fallback);
        if (allow_no_capacity && !consumed &&
            address == ::FarLib::allocator::remote::InvalidRemoteAddr &&
            status == ec_batch::EcBatchStatus::kManagerRejected)
            return false;
        if (!consumed && address != ::FarLib::allocator::remote::InvalidRemoteAddr) {
            if (::FarLib::get_config().ft_incremental_one_sided) {
                // Deferred sources may have moved to a new FarObjectEntry.
                // Address publication must precede every possible WRITE post.
                while (!try_publish_ec_rmw_address(data, address, client_idx)) {
                    (void)check_cq_idx_with_client_idx(qp_idx, client_idx);
                }
            } else {
                entry->set_remote_addr(address);
            }
            consumed = true;
        }
        if (status == ec_batch::EcBatchStatus::kOk) break;
        if (status == ec_batch::EcBatchStatus::kEncodeRejected ||
            status == ec_batch::EcBatchStatus::kSealRejected ||
            status == ec_batch::EcBatchStatus::kInvalidArgument)
            ERROR("ec_direct: failed to encode/seal borrowed sources");
        if (!consumed) {
            const auto flushed = builder.flush();
            if (flushed != ec_batch::EcBatchStatus::kOk &&
                flushed != ec_batch::EcBatchStatus::kPendingQueueFull)
                ERROR("ec_direct: cannot flush blocked group");
        }
        progress();
    }
    if (!consumed) ERROR("ec_direct: successful add without a remote address");
    if (recovery_fallback) {
        // A failure may hand off only one object. Seal/post its partial group
        // now; the ordinary immutable completion retains/releases its borrow.
        // Never wait for future admissions to complete this source.
        flush_ec_direct_groups(buffers, client_idx);
        return true;
    }
    if (++buffers.ec_objects_since_post >= EvictBatchSize) {
        (void)post_ec_direct_pending(buffers, client_idx, qp_idx);
        buffers.ec_objects_since_post = 0;
    }
    return true;
}

inline size_t ConcurrentArrayCache::post_ec_direct_pending(
    EvictBufferSet &buffers, size_t client_idx, size_t qp_idx) {
    profile::evict_breakdown::Scope prepare(buffers.breakdown,
        profile::evict_breakdown::Stage::SendPrepare);
    auto &builder = *buffers.direct_builder;
    auto *client = rdma::get_client(client_idx);
    const auto &config = ::FarLib::get_config();
    for (size_t ep = 0; ep < buffers.server_count; ++ep) ASSERT(buffers.ec_writes[ep].count == 0);
    size_t groups = 0;
    while (groups < EvictBatchSize && builder.pending_count()) {
        const uint64_t token = builder.front_token();
        auto *bank_entry = ec_direct_bank_->get(token);
        if (!bank_entry) ERROR("ec_direct: stale pending token");
        const auto &record = bank_entry->record;
        bool fits = true;
        for (size_t s = 0; s < 6; ++s) {
            const auto &segment = record.group.segments[s];
            if (segment.endpoint_idx >= buffers.server_count || segment.slot_size != record.slot_size)
                ERROR("ec_direct: invalid group layout");
            size_t same_endpoint = 1;
            for (size_t j = 0; j < s; ++j)
                same_endpoint += record.group.segments[j].endpoint_idx == segment.endpoint_idx;
            if (buffers.ec_writes[segment.endpoint_idx].count + same_endpoint > EvictBatchSize) fits = false;
        }
        if (!fits) break;
        for (size_t s = 0; s < 6; ++s) {
            const auto &segment = record.group.segments[s];
            const uint64_t wr_id = ec_batch::encode_ec_batch_wr_id(token, s);
            if (ec_recovery_endpoint_is_dead(segment.endpoint_idx)) {
                complete_ec_direct_write(token, s, false);
                continue;
            }
            auto &batch = buffers.ec_writes[segment.endpoint_idx];
            const size_t i = batch.count++;
            auto *sges = batch.direct_sges[i];
            const auto mapped = config.map_remote_addr(segment.addr);
            ASSERT(mapped.first == segment.endpoint_idx);
            ASSERT(mapped.second + record.slot_size <= config.server_buffer_size);
            if (s < 4 && record.objects[s]) {
                client->build_send_wr(batch.wrs[i], sges[0], mapped.second,
                    const_cast<void *>(record.objects[s]), record.object_sizes[s],
                    wr_id, true, IBV_WR_RDMA_WRITE, segment.endpoint_idx);
                if (record.object_sizes[s] < record.slot_size) {
                    sges[1].addr = reinterpret_cast<uintptr_t>(record.zero_pad);
                    sges[1].length = record.slot_size - record.object_sizes[s];
                    sges[1].lkey = record.parity_lkey;
                    batch.wrs[i].num_sge = 2;
                }
            } else {
                void *source = s < 4 ? record.zero_pad : record.parity[s - 4];
                client->build_send_wr(batch.wrs[i], sges[0], mapped.second,
                    source, record.slot_size, wr_id, true, IBV_WR_RDMA_WRITE, segment.endpoint_idx);
                sges[0].lkey = record.parity_lkey;
            }
            if (i) batch.wrs[i - 1].next = &batch.wrs[i];
        }
        builder.commit_pending();
        ++groups;
    }
    for (size_t ep = 0; ep < buffers.server_count; ++ep) {
        auto &batch = buffers.ec_writes[ep];
        if (!batch.count) continue;
        ibv_send_wr *remaining = &batch.wrs[0];
        size_t accepted = batch.count;
        auto post = [&] {
            profile::evict_breakdown::Scope scope(buffers.breakdown,
                profile::evict_breakdown::Stage::PostSend);
            return client->post_writes(remaining, &remaining, qp_idx, ep);
        };
        while (!post()) {
            ec_diag_post_write_retries_.fetch_add(1, std::memory_order_relaxed);
            {
                profile::evict_breakdown::Scope polling(buffers.breakdown,
                    profile::evict_breakdown::Stage::CqProcess);
                (void)check_cq_idx_with_client_idx_endpoint(qp_idx, client_idx, ep);
            }
            if (ec_recovery_endpoint_is_dead(ep)) {
                for (auto *wr = remaining; wr; wr = wr->next) {
                    handle_ec_batch_write_complete(wr->wr_id, false);
                    --accepted;
                }
                break;
            }
        }
        size_t bytes = 0;
        for (size_t i = 0; i < accepted; ++i) {
            size_t wr_bytes = 0;
            for (int j = 0; j < batch.wrs[i].num_sge; ++j) wr_bytes += batch.direct_sges[i][j].length;
            profile::count_rdma_write_post(wr_bytes);
            bytes += wr_bytes;
        }
        if (profile::work_phase_active.load(std::memory_order_relaxed)) {
            ec_direct_work_write_posts_.fetch_add(accepted, std::memory_order_relaxed);
            ec_direct_work_write_bytes_.fetch_add(bytes, std::memory_order_relaxed);
        }
        profile::count_evacuation_bytes(bytes);
        profile::count_evac_flush(accepted, accepted == EvictBatchSize);
        batch.count = 0;
    }
    if (groups) {
        ec_batch_groups_posted_.fetch_add(groups, std::memory_order_relaxed);
        ec_direct_groups_.fetch_add(groups, std::memory_order_relaxed);
    }
    account_ec_worker_builder(buffers);
    return groups;
}

inline void ConcurrentArrayCache::flush_ec_direct_groups(EvictBufferSet &buffers, size_t client_idx) {
    auto &builder = *buffers.direct_builder;
    const size_t qp = rdma::get_client(client_idx)->get_qp_idx();
    while (builder.group_open() || builder.pending_count()) {
        const auto status = builder.flush();
        if (status != ec_batch::EcBatchStatus::kOk && status != ec_batch::EcBatchStatus::kPendingQueueFull)
            ERROR("ec_direct: failed to seal worker tail");
        (void)post_ec_direct_pending(buffers, client_idx, qp);
    }
    buffers.ec_objects_since_post = 0;
    account_ec_worker_builder(buffers);
}

inline void ConcurrentArrayCache::complete_ec_direct_write(uint64_t token, uint8_t segment, bool success) {
    ec_diag_complete_cqes_.fetch_add(1, std::memory_order_relaxed);
    auto *entry = ec_direct_bank_->complete_segment(token, segment, success);
    if (!entry) {
        ec_diag_complete_not_last_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (!entry->recoverable()) ERROR("ec_direct: fewer than four durable segments");
    const int standby = ::FarLib::get_config().ft_standby_endpoint;
    if (standby >= 0) {
        for (size_t s = 0; s < 6; ++s) {
            if (entry->record.group.segments[s].endpoint_idx == static_cast<size_t>(standby) &&
                (entry->failed & (1u << s)) == 0) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true))
                    std::cout << "INFO: ec_batch standby write_completed endpoint=" << standby
                              << " token=" << token << " bytes=" << entry->record.slot_size << std::endl;
                break;
            }
        }
    }
    if (entry->failed)
        std::cout << "INFO: ec_batch degraded_write token=" << token
                  << " completed_ok=" << entry->durable_segment_count()
                  << " failed_mask=" << static_cast<unsigned>(entry->failed) << std::endl;
    ec_diag_complete_last_.fetch_add(1, std::memory_order_relaxed);
    if (::FarLib::get_config().ft_incremental_update)
        remote_allocator.small_object_stripe_manager().mark_slot_group_durable(entry->record.group.id);
    for (size_t i = 0; i < 4; ++i) {
        if (auto *object = const_cast<void *>(entry->record.objects[i])) {
            auto *block = static_cast<::FarLib::allocator::BlockHead *>(object) - 1;
            ::FarLib::allocator::release_ec_write_source(block);
            // Resolve the CURRENT entry via the block; FarPtr moves can change
            // its metadata owner while the original data address stays pinned.
            complete_evict_writeback(object);
        }
    }
    if (!ec_direct_bank_->release(token)) ERROR("ec_direct: terminal token release failed");
    ec_batch_groups_completed_.fetch_add(1, std::memory_order_relaxed);
}
} // namespace FarLib::cache
