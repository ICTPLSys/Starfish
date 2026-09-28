#pragma once
#include <chrono>
#include <map>
#include "cache/carbink/entry_guard.hpp"
#include "cache/carbink/settings.hpp"

namespace FarLib::cache {
inline void ConcurrentArrayCache::carbink_publish_ready(
    const ec_batch::EcGroupSendRecord &r) {
    if (r.split_object || r.slot_size != hydra::kPageBytes) return;
    std::array<uintptr_t, 4> owners{};
    for (size_t i = 0; i < 4; ++i) {
        if (!r.objects[i]) continue;
        auto *block = static_cast<const allocator::BlockHead *>(r.objects[i]) - 1;
        auto obj = block->obj_meta_data.load(std::memory_order_acquire);
        if (obj.is_null()) continue;
        auto &entry = get_entry_of(obj);
        if (!entry.hydra_page()) ERROR("carbink: non-page span owner");
        owners[i] = reinterpret_cast<uintptr_t>(&entry);
    }
    if (!remote_allocator.small_object_stripe_manager()
             .publish_compaction_ready(r.group, owners))
        ERROR("carbink: completed full-stripe publication failed");
}

inline void ConcurrentArrayCache::carbink_start_workers() {
    if (!::FarLib::get_config().is_carbink_mode()) return;
    carbink_worker_count_ = ::FarLib::get_config().configured_compaction_worker_count();
    carbink_queues_ = std::make_unique<CarbinkQueue[]>(carbink_worker_count_);
    carbink_counters_ = std::make_unique<CarbinkWorkerCounters[]>(carbink_worker_count_);
    carbink_worker_launch_ = std::make_unique<CarbinkWorkerLaunch[]>(carbink_worker_count_);
    carbink_running_.store(true, std::memory_order_release);
    auto fn = +[](CarbinkWorkerLaunch *l) { l->cache->carbink_compact_work(l->worker_id); };
    for (size_t i = 0; i < carbink_worker_count_; ++i) {
        carbink_worker_launch_[i] = {this, i};
        carbink_workers_.push_back(background_cluster_
            ? uthread::create_on<true>(*background_cluster_, fn,
                  &carbink_worker_launch_[i], "carbink compaction")
            : uthread::create<true>(fn, &carbink_worker_launch_[i], "carbink compaction"));
    }
    auto scan = +[](ConcurrentArrayCache *c) { c->carbink_scan_work(); };
    carbink_scanner_ = background_cluster_
        ? uthread::create_on<true>(*background_cluster_, scan, this, "carbink scanner")
        : uthread::create<true>(scan, this, "carbink scanner");
    std::fprintf(stderr,
                 "\ncarbink.compaction workers=%zu scanner=1 batch_max_spans=8 queue_capacity=1024"
                 " scan_low_watermark=512 logical_client_base=%zu os_workers=%zu rdma_clients=%zu\n",
                 static_cast<size_t>(carbink_worker_count_),
                 static_cast<size_t>(::FarLib::get_config().compaction_client_base()),
                 static_cast<size_t>(::FarLib::get_config().runtime_worker_count()),
                 static_cast<size_t>(::FarLib::get_config().runtime_client_count()));
}

inline void ConcurrentArrayCache::carbink_stop_workers() {
    if (!carbink_running_.exchange(false, std::memory_order_acq_rel)) return;
    if (carbink_scanner_) uthread::join(std::move(carbink_scanner_));
    for (auto &w : carbink_workers_) uthread::join(std::move(w));
    carbink_workers_.clear();
    uint64_t tasks=0, reclaimed=0, moved=0, rollback=0, rejected=0;
    uint64_t wt=0, wr=0, wm=0, wb=0;
    for (size_t i=0; i<carbink_worker_count_; ++i) {
        const auto &c=carbink_counters_[i];
        tasks+=c.drained.total; reclaimed+=c.reclaimed.total;
        moved+=c.moved.total; rollback+=c.rollback.total; rejected+=c.claim_failed.total;
        wt+=c.drained.work; wr+=c.reclaimed.work; wm+=c.moved.work; wb+=c.rollback.work;
    }
    std::cout << "carbink.compaction_total task_drained=" << tasks
              << " reclaim_stripes=" << reclaimed
              << " reclaim_bytes=" << reclaimed*(6*hydra::kPageBytes)
              << " moved_spans=" << moved << " rollback_spans=" << rollback
              << " claim_failed=" << rejected
              << " queue_remaining=" << carbink_queue_depth_.load() << '\n';
    std::cout << "carbink.compaction_work task_drained=" << wt
              << " reclaim_stripes=" << wr << " moved_spans=" << wm
              << " rollback_spans=" << wb << '\n';
    const auto &transport = rdma::get_client(
        ::FarLib::get_config().compaction_client_base())->carbink_transport();
    std::cout << "carbink.transport data_req_explicit="
              << transport.explicit_data_requests()
              << " data_req_batched_spans=" << transport.batched_data_requests()
              << '\n';
}

inline void ConcurrentArrayCache::carbink_scan_work() {
    auto &m=remote_allocator.small_object_stripe_manager();
    size_t stripe=0, slot=0, dispatch=0;
    // Diagnostic only: retain the same worker/client topology while isolating
    // full-stripe eviction from remote compaction traffic.
    const bool scan_enabled = env_flag_or_default("FARLIB_CARBINK_COMPACTION_SCAN", true);
    while (carbink_running_.load(std::memory_order_acquire)) {
        if (!scan_enabled) { uthread::yield(); continue; }
        if (carbink_queue_depth_.load(std::memory_order_relaxed)>=512) {
            uthread::yield(); continue;
        }
        std::vector<CarbinkGroupView> snapshot;
        m.collect_compaction_candidates(stripe, slot, 4096, snapshot);
        // Only equal endpoint-role layouts permit a server-local span move.
        using Layout=std::array<uint32_t,6>;
        std::map<Layout,std::array<std::vector<CarbinkGroupView>,16>> buckets;
        for (const auto &v:snapshot) {
            if (!v.live_mask || v.live_mask>=15) continue;
            Layout l{};
            for (size_t i=0;i<6;++i) l[i]=v.group.segments[i].endpoint_idx;
            buckets[l][v.live_mask].push_back(v);
        }
        for (auto &[layout, masks]:buckets) {
            (void)layout;
            for (uint8_t mask=1;mask<8;++mask) {
                auto &a=masks[mask], &b=masks[15^mask];
                for (size_t i=0,n=std::min(a.size(),b.size());i<n;++i) {
                    if (!carbink_running_.load(std::memory_order_relaxed) ||
                        carbink_queue_depth_.load(std::memory_order_relaxed)>=1024) break;
                    CarbinkTask task{a[i],b[i]};
                    if (task.dst.live_count<task.src.live_count) std::swap(task.dst,task.src);
                    auto &q=carbink_queues_[dispatch++%carbink_worker_count_];
                    std::lock_guard<std::mutex> lock(q.mutex);
                    q.tasks.push_back(std::move(task));
                    carbink_queue_depth_.fetch_add(1,std::memory_order_relaxed);
                }
            }
        }
        uthread::yield();
    }
}
inline void ConcurrentArrayCache::carbink_compact_work(size_t worker_id) {
    using Status=rdma::compact::CompactSubmitStatus;
    using Descriptor=rdma::compact::CompactMoveDescriptor;
    const auto &config=::FarLib::get_config();
    const size_t client_idx=config.compaction_client_base()+worker_id;
    constexpr size_t qp=0;
    auto *client=rdma::get_client(client_idx);
    if (!client) ERROR("carbink: missing compaction client");
    auto &transport=client->carbink_transport();
    const size_t transport_qp=client->carbink_qp_index(qp);
    auto &m=remote_allocator.small_object_stripe_manager();
    auto &q=carbink_queues_[worker_id];
    auto &stats=carbink_counters_[worker_id];
    const size_t limit=carbink::inflight_update_limit();
    const size_t task_budget=carbink::drain_task_budget(carbink_worker_count_);
    size_t drain_admitted=0;
    struct Update {
        Descriptor d{};
        bool submitted=false, rollback=false, done=false, rate_reserved=false;
        std::chrono::steady_clock::time_point posted{};
    };
    struct Active {
        CarbinkTask task;
        std::array<carbink::EntryGuard,4> src_guard,dst_guard;
        std::array<Update,4> updates;
        size_t pending=0;
    };
    std::vector<std::unique_ptr<Active>> active;
    size_t inflight=0;
    auto unguard=[](Active &t) {
        for (auto &g:t.src_guard) g.release();
        for (auto &g:t.dst_guard) g.release();
    };
    auto unclaim=[&](Active &t) {
        if (!m.release_compaction_group_claim(t.task.src) ||
            !m.release_compaction_group_claim(t.task.dst))
            ERROR("carbink: lost claimed generation");
    };
    auto prepare=[&](const CarbinkTask &queued)->std::unique_ptr<Active> {
        auto t=std::make_unique<Active>();
        if (!m.try_claim_compaction_group(queued.dst,&t->task.dst)) return nullptr;
        if (!m.try_claim_compaction_group(queued.src,&t->task.src)) {
            m.release_compaction_group_claim(t->task.dst); return nullptr;
        }
        if ((t->task.src.live_mask ^ t->task.dst.live_mask)!=15 ||
            (t->task.src.live_mask & t->task.dst.live_mask)!=0) {
            unclaim(*t); return nullptr;
        }
        for (uint8_t s=0;s<4;++s) {
            const bool source=t->task.src.live_mask & (1u<<s);
            const auto &v=source?t->task.src:t->task.dst;
            auto &g=source?t->src_guard[s]:t->dst_guard[s];
            if (!g.acquire(reinterpret_cast<FarObjectEntry *>(v.owners[s]),
                           v.group.segments[s].addr)) {
                unguard(*t); unclaim(*t); return nullptr;
            }
        }
        for (uint8_t s=0;s<4;++s) {
            if (!(t->task.src.live_mask & (1u<<s))) continue;
            auto &u=t->updates[s]; auto &d=u.d;
            auto src=config.map_remote_addr(t->task.src.group.segments[s].addr);
            auto dst=config.map_remote_addr(t->task.dst.group.segments[s].addr);
            auto p0=config.map_remote_addr(t->task.dst.group.segments[4].addr);
            auto p1=config.map_remote_addr(t->task.dst.group.segments[5].addr);
            if (src.first!=dst.first) ERROR("carbink: cross-endpoint move");
            d.wr_id=reinterpret_cast<uint64_t>(&u);
            d.src_offset=src.second; d.dst_offset=dst.second;
            d.endpoint_idx=dst.first; d.qp_idx=transport_qp;
            d.parity_endpoint0=p0.first; d.parity_endpoint1=p1.first;
            d.parity0_offset=p0.second; d.parity1_offset=p1.second;
            d.data_slot=s;
            d.ack_data_qp_idx=static_cast<uint16_t>(transport_qp);
            // Preserve original init_from_zero semantics, not a new protocol.
            d.init_from_zero=true;
            ++t->pending;
        }
        return t;
    };
    auto progress=[&] { (void)check_cq_idx_with_client_idx(qp,client_idx); };
    while (carbink_running_.load(std::memory_order_acquire) || !active.empty()) {
        progress();
        // Admit multiple 1-2-span tasks before flushing endpoint batches.
        while (carbink_running_.load(std::memory_order_relaxed) &&
               drain_admitted<task_budget) {
            CarbinkTask queued;
            {
                std::lock_guard<std::mutex> lock(q.mutex);
                if (q.tasks.empty()) break;
                auto expected=q.tasks.front().src.live_count;
                if (!active.empty() && inflight+expected>limit) break;
                queued=q.tasks.front(); q.tasks.pop_front();
                carbink_queue_depth_.fetch_sub(1,std::memory_order_relaxed);
            }
            ++drain_admitted;
            auto t=prepare(queued);
            if (!t) { stats.claim_failed.add(); continue; }
            inflight+=t->pending;
            active.push_back(std::move(t));
        }
        for (auto &t:active) for (uint8_t s=0;s<4;++s) {
            if (!(t->task.src.live_mask & (1u<<s))) continue;
            auto &u=t->updates[s]; auto &d=u.d;
            if (u.done || u.submitted) continue;
            if (!u.rollback && !u.rate_reserved) {
                carbink::rate_limit_one_update();
                u.rate_reserved=true;
            }
            rdma::compact::CompactRequestHandle handle{};
            auto status=u.rollback?transport.submit_zero(d,&handle):transport.submit(d,&handle);
            if (status==Status::NeedFlush) {
                (void)transport.flush(d.endpoint_idx,d.qp_idx); continue;
            }
            if (status==Status::Backpressure) continue;
            if (status!=Status::Accepted) ERROR("carbink: compact/rollback submission rejected");
            u.submitted=true; u.posted=std::chrono::steady_clock::now();
        }
        (void)transport.flush_qp(transport_qp);
        progress();
        for (auto &t:active) for (uint8_t s=0;s<4;++s) {
            if (!(t->task.src.live_mask & (1u<<s))) continue;
            auto &u=t->updates[s]; auto &d=u.d;
            if (u.done || !u.submitted) continue;
            uint32_t status=0;
            bool ack=transport.final_ack_done(d.wr_id,&status);
            // Independent events, checked without blocking the other tasks.
            if (!ack || !transport.send_complete(d.wr_id)) {
                if (std::chrono::steady_clock::now()-u.posted>std::chrono::seconds(30))
                    ERROR("carbink: compact/rollback ACK deadline exceeded");
                continue;
            }
            if (status!=0) ERROR("carbink: remote compact/rollback failed");
            if (!transport.erase(d.wr_id)) ERROR("carbink: premature request retirement");
            if (!u.rollback) {
                auto &g=t->src_guard[s];
                if (g.try_lock_publish()) {
                    if (!m.compaction_transfer_slot(t->task.src,t->task.dst,s,
                                                   reinterpret_cast<uintptr_t>(g.entry()))) {
                        g.cancel_publish(); ERROR("carbink: guarded metadata transfer failed");
                    }
                    g.publish_locked(t->task.dst.group.segments[s].addr);
                    stats.moved.add();
                } else {
                    // Foreground won; undo data/parity before dropping claims.
                    d.wr_id=reinterpret_cast<uint64_t>(&u)|1u;
                    d.src_offset=d.dst_offset; d.init_from_zero=false;
                    u.rollback=true; u.submitted=false; continue;
                }
            } else stats.rollback.add();
            u.done=true; --t->pending; --inflight;
        }
        for (auto it=active.begin();it!=active.end();) {
            auto &t=**it;
            if (t.pending) { ++it; continue; }
            unguard(t);
            SmallObjectStripeManager::SlotGroupState state{};
            uint8_t mask=0;
            if (!m.get_slot_group_state(t.task.src.id,&state,&mask))
                ERROR("carbink: source disappeared under claim");
            if (!mask && (state==SmallObjectStripeManager::kSlotGroupDead ||
                          state==SmallObjectStripeManager::kSlotGroupSealed))
                stats.reclaimed.add();
            unclaim(t); stats.drained.add();
            it=active.erase(it);
        }
        // A finite drain budget is renewed only after the admitted tasks
        // (including any rollback) have finished, as in the original worker.
        if (active.empty()) drain_admitted=0;
        uthread::yield();
    }
    // Queued work never held claims. Only active RPCs must drain at shutdown.
    std::lock_guard<std::mutex> lock(q.mutex);
    carbink_queue_depth_.fetch_sub(q.tasks.size(),std::memory_order_relaxed);
    q.tasks.clear();
}
} // namespace FarLib::cache
