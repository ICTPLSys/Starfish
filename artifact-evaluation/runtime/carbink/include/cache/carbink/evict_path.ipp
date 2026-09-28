#pragma once
#include <chrono>
#include <span>

namespace FarLib::cache {
inline void ConcurrentArrayCache::init_carbink_evict_buffers() {
    const auto &config = ::FarLib::get_config();
    if (!config.is_carbink_mode()) return;
    const size_t owners = std::max<size_t>(1, config.evacuate_thread_cnt);
    carbink_evict_owners_ = std::make_unique<CarbinkEvictOwner[]>(owners);
    // Two banks of 64 complete 4+2 groups per logical eviction worker.
    // The tested Hydra completion primitive is parameterized, not its
    // split-page data path: Carbink retains its own ring, batches and owners.
    carbink_write_ring_ = std::make_unique<hydra::WriteRing>(
        owners, 128, true, hydra::kPageBytes, 64);
    carbink_write_fences_ = std::make_unique<carbink::WriteFenceTable>(
        owners, config.server_count);
    auto *control = rdma::ClientControl::get_default();
    const auto allocate = [](void *pd, size_t bytes, hydra::RegisteredParity *out) {
        void *buffer = nullptr;
        if (posix_memalign(&buffer, 4096, bytes) != 0) return false;
        auto *mr = ibv_reg_mr(static_cast<ibv_pd *>(pd), buffer, bytes,
                             IBV_ACCESS_LOCAL_WRITE);
        if (!mr) { std::free(buffer); return false; }
        *out = {buffer, mr->lkey, mr};
        return true;
    };
    const auto release = [](void *, hydra::RegisteredParity buffer) {
        if (ibv_dereg_mr(static_cast<ibv_mr *>(buffer.registration)) != 0)
            ERROR("carbink: deregister parity bank failed");
        std::free(buffer.base);
    };
    if (!carbink_write_ring_->init_parity_buffers(
            control->get_protection_domain(), allocate, release) ||
        !allocate(control->get_protection_domain(), hydra::kPageBytes,
                  &carbink_zero_page_))
        ERROR("carbink: initialize registered eviction buffers failed");
    std::memset(carbink_zero_page_.base, 0, hydra::kPageBytes);
    for (size_t owner = 0; owner < owners; ++owner) {
        auto &batch = carbink_evict_owners_[owner].batch;
        batch.init(config.server_count);
        if (!batch.bind_context(config.carbink_evict_client_base() + owner, 0))
            ERROR("carbink: bind owner WRITE queue failed");
    }
    std::cout << "carbink.evict owners=" << owners
              << " groups_per_bank=64 banks_per_owner=2 batch_wrs=64"
              << " data=borrowed_full_8192 parity=fence"
              << " logical_client_base=" << config.carbink_evict_client_base()
              << " parity_bytes=" << carbink_write_ring_->parity_bytes()
              << " os_workers=" << config.runtime_worker_count()
              << " rdma_clients=" << config.runtime_client_count() << '\n';
}

inline void ConcurrentArrayCache::progress_carbink_writes(size_t owner) {
    auto &state = carbink_evict_owners_[owner];
    ++state.polls;
    (void)check_cq_idx_with_client_idx(0, state.batch.client_idx());
}

inline bool ConcurrentArrayCache::stage_carbink_span(
    void *local, FarObjectEntry *entry, size_t owner) {
    if (!carbink_write_ring_ || owner >= carbink_write_ring_->owner_count())
        ERROR("carbink: no logical eviction owner");
    auto &state = carbink_evict_owners_[owner];
    const auto &config = ::FarLib::get_config();
    auto *control = rdma::ClientControl::get_default();
    if (!hydra::page_source_in_registered_range(
            local, control->get_buffer(), config.client_buffer_size))
        ERROR("carbink: source span outside registered application memory");
    auto &manager = remote_allocator.small_object_stripe_manager();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    auto progress = [&] {
        flush_carbink_writes(owner);
        progress_carbink_writes(owner);
        if (std::chrono::steady_clock::now() >= deadline)
            ERROR("carbink: eviction owner buffer/remote allocation deadline");
    };
    if (state.count == 0) {
        state.open = {};
        hydra::WriteRing::Entry *slot = nullptr;
        while (!carbink_write_ring_->reserve_page(owner, &state.token, &slot,
                                                  &state.open.staging))
            progress();
        while (!manager.allocate_slot_group(hydra::kPageBytes, &state.open.group))
            progress();
        state.open.slot_size = hydra::kPageBytes;
        state.open.direct_span_data = true;
        state.open.zero_data = carbink_zero_page_.base;
        state.open.zero_lkey = carbink_zero_page_.lkey;
    }
    const size_t index = state.count++;
    state.open.objects[index] = local;
    state.open.object_sizes[index] = hydra::kPageBytes;
    state.open.live_mask |= static_cast<uint8_t>(1u << index);
    state.open.live_count = static_cast<uint8_t>(state.count);
    // The caller has already set EVICTING and holds a WRITE reference. Bind
    // the hint to the stable logical client actually responsible for the DMA.
    entry->set_client_idx(state.batch.client_idx());
    entry->set_remote_addr(state.open.group.segments[index].addr);
    if (state.count == 4) seal_carbink_group(owner);
    // The fourth add may already have completed; do not touch entry here.
    ec_diag_stage_ok_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline void ConcurrentArrayCache::seal_carbink_group(size_t owner) {
    auto &state = carbink_evict_owners_[owner];
    if (!state.count) return;
    const auto &config = ::FarLib::get_config();
    auto &record = state.open;
    const void *data[4];
    void *parity[2] = {record.staging.parity[0], record.staging.parity[1]};
    for (size_t i=0; i<4; ++i)
        data[i] = record.objects[i] ? record.objects[i] : record.zero_data;
    if (!hydra::page_codec_encode_4plus2(data, parity, hydra::kPageBytes) ||
        !remote_allocator.small_object_stripe_manager().seal_slot_group(
            record.group.id, record.live_mask))
        ERROR("carbink: full-span group encoding/sealing failed");
    for (const auto &segment : record.group.segments) {
        if (segment.slot_size != hydra::kPageBytes ||
            !config.validate_mapping(segment.addr, segment.slot_size) ||
            config.map_remote_addr(segment.addr).first != segment.endpoint_idx ||
            ec_recovery_endpoint_is_dead(segment.endpoint_idx))
            ERROR("carbink: invalid/unavailable full-stripe WRITE endpoint");
    }
    const uint64_t token = state.token;
    if (!carbink_write_ring_->publish_page(token, record))
        ERROR("carbink: publish owner WRITE record failed");
    state.in_flight.fetch_add(1, std::memory_order_relaxed);
    ++state.groups;
    state.groups_work += profile::is_working();
    auto *client = rdma::get_client(state.batch.client_idx());
    bool full = false;
    for (size_t segment=0; segment<6; ++segment) {
        const auto &dst = record.group.segments[segment];
        const auto mapped = config.map_remote_addr(dst.addr);
        const bool data_live = segment<4 && record.objects[segment]!=nullptr;
        const uint32_t lkey = segment<4
            ? (data_live ? client->registered_buffer_lkey() : record.zero_lkey)
            : record.staging.lkey;
        const hydra::PendingWrite request{
            ec_batch::ec_batch_record_source(record, segment), mapped.second,
            ec_batch::encode_ec_batch_wr_id(token, segment),
            static_cast<uint32_t>(hydra::kPageBytes), lkey};
        if (!state.batch.enqueue(dst.endpoint_idx, request))
            ERROR("carbink: owner WRITE batch overflow");
        full |= state.batch.endpoint(dst.endpoint_idx).count ==
                hydra::WriteBatch::kCapacity;
    }
    state.count = 0;
    state.token = 0;
    ec_batch_groups_posted_.fetch_add(1, std::memory_order_relaxed);
    if (full || carbink_write_ring_->buffer_end(token)) flush_carbink_writes(owner);
}

inline void ConcurrentArrayCache::flush_carbink_writes(size_t owner) {
    auto &state = carbink_evict_owners_[owner];
    auto &batch = state.batch;
    auto *client = rdma::get_client(batch.client_idx());
    // Diagnostic control: keep the same chain/ownership, but request a CQE
    // for every parity WRITE to isolate fence aggregation from data lifetime.
    static const bool aggregate_parity =
        env_flag_or_default("FARLIB_CARBINK_PARITY_FENCE", true);
    for (size_t endpoint=0; endpoint<batch.endpoint_count(); ++endpoint) {
        auto &queue = batch.endpoint(endpoint);
        const size_t count = queue.count;
        if (!count) continue;
        uint64_t parity_ids[hydra::WriteBatch::kCapacity];
        size_t parity_count=0, last_parity=SIZE_MAX;
        for (size_t i=0; i<count; ++i) {
            if (aggregate_parity &&
                ec_batch::ec_batch_wr_id_segment(queue.requests[i].wr_id)>=4) {
                parity_ids[parity_count++] = queue.requests[i].wr_id;
                last_parity=i;
            }
        }
        uint64_t fence=0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        if (parity_count) {
            while (!(fence=carbink_write_fences_->acquire(
                        owner,endpoint,std::span<const uint64_t>(parity_ids,parity_count)))) {
                (void)check_cq_idx_with_client_idx_endpoint(0,batch.client_idx(),endpoint);
                if (std::chrono::steady_clock::now()>=deadline)
                    ERROR("carbink: parity fence pool deadline");
            }
            ++state.fences;
            state.fences_work += profile::is_working();
        }
        ibv_send_wr wrs[hydra::WriteBatch::kCapacity]{};
        ibv_sge sges[hydra::WriteBatch::kCapacity]{};
        for (size_t i=0; i<count; ++i) {
            const auto &request=queue.requests[i];
            const bool parity=ec_batch::ec_batch_wr_id_segment(request.wr_id)>=4;
            client->build_send_wr(wrs[i],sges[i],request.remote_offset,
                request.local_addr,request.bytes,i==last_parity?fence:request.wr_id,
                !aggregate_parity || !parity || i==last_parity,
                IBV_WR_RDMA_WRITE,endpoint);
            sges[i].lkey=request.lkey;
            if (i) wrs[i-1].next=&wrs[i];
        }
        size_t first=0;
        while (first<count) {
            if (ec_recovery_endpoint_is_dead(endpoint))
                ERROR("carbink: failed WRITE endpoint; no synthetic completion");
            ibv_send_wr *bad=nullptr;
            ++state.post_calls;
            state.post_calls_work += profile::is_working();
            const bool posted=client->post_writes(&wrs[first],&bad,0,endpoint);
            const size_t accepted=hydra::accepted_write_prefix(
                reinterpret_cast<uintptr_t>(wrs),sizeof(wrs[0]),count,first,
                reinterpret_cast<uintptr_t>(bad),posted);
            if (accepted==SIZE_MAX) ERROR("carbink: invalid partial WRITE prefix");
            for (size_t i=first; i<first+accepted; ++i)
                profile::count_rdma_write_post(queue.requests[i].bytes);
            state.posted_wrs+=accepted;
            if (profile::is_working()) state.posted_wrs_work+=accepted;
            first+=accepted;
            if (!posted) {
                ++state.partial_posts;
                (void)check_cq_idx_with_client_idx_endpoint(0,batch.client_idx(),endpoint);
                if (std::chrono::steady_clock::now()>=deadline)
                    ERROR("carbink: batch WRITE credit deadline");
            }
        }
        batch.clear_endpoint(endpoint);
    }
}

inline void ConcurrentArrayCache::flush_carbink_owner(size_t owner, bool drain) {
    if (!carbink_write_ring_) return;
    if (owner>=carbink_write_ring_->owner_count())
        ERROR("carbink: flush without logical eviction owner");
    seal_carbink_group(owner);
    flush_carbink_writes(owner);
    if (!drain) return;
    auto &state=carbink_evict_owners_[owner];
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while (state.in_flight.load(std::memory_order_acquire)!=0) {
        progress_carbink_writes(owner);
        if (std::chrono::steady_clock::now()>=deadline)
            ERROR("carbink: owner WRITE drain deadline");
    }
}

inline void ConcurrentArrayCache::report_carbink_writes() {
    if (!carbink_write_ring_) return;
    uint64_t groups=0,posts=0,wrs=0,fences=0,polls=0,partial=0;
    uint64_t wg=0,wp=0,ww=0,wf=0;
    for (size_t owner=0; owner<carbink_write_ring_->owner_count(); ++owner) {
        const auto &s=carbink_evict_owners_[owner];
        if (s.count || s.batch.pending() || s.in_flight.load()!=0)
            ERROR("carbink: eviction producer not drained at shutdown");
        groups+=s.groups; posts+=s.post_calls; wrs+=s.posted_wrs;
        fences+=s.fences; polls+=s.polls; partial+=s.partial_posts;
        wg+=s.groups_work; wp+=s.post_calls_work;
        ww+=s.posted_wrs_work; wf+=s.fences_work;
    }
    std::cout<<"carbink.evict_total groups="<<groups<<" post_calls="<<posts
             <<" write_wrs="<<wrs<<" parity_fences="<<fences
             <<" owner_poll_passes="<<polls<<" partial_posts="<<partial
             <<" ring_in_use="<<carbink_write_ring_->in_use()
             <<" fences_in_use="<<carbink_write_fences_->in_use()<<'\n';
    std::cout<<"carbink.evict_work groups="<<wg<<" post_calls="<<wp
             <<" write_wrs="<<ww<<" parity_fences="<<wf
             <<" wrs_per_post="<<(wp?static_cast<double>(ww)/wp:0)<<'\n';
}

inline void ConcurrentArrayCache::release_carbink_evict_buffers() {
    if (!carbink_write_ring_) return;
    if (carbink_write_ring_->in_use() || carbink_write_fences_->in_use())
        ERROR("carbink: registered eviction buffers still owned");
    carbink_write_ring_.reset();
    carbink_write_fences_.reset();
    if (ibv_dereg_mr(static_cast<ibv_mr *>(carbink_zero_page_.registration))!=0)
        ERROR("carbink: deregister zero page failed");
    std::free(carbink_zero_page_.base);
    carbink_zero_page_={};
}
} // namespace FarLib::cache
