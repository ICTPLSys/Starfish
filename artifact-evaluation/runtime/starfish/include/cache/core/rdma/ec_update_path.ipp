#pragma once
#include "cache/concurrent_cache.hpp"

namespace FarLib::cache {

bool ConcurrentArrayCache::handle_ec_incremental_complete(const ibv_wc &wc) {
    if (!::FarLib::get_config().ft_incremental_update) return false;
    auto *transport = rdma::ClientControl::get_default()->ec_update_transport();
    if (!transport || !transport->handle(wc)) return false;
    if (wc.status != IBV_WC_SUCCESS) note_ec_recovery_error_wc(wc);
    return true;
}

bool ConcurrentArrayCache::stage_ec_incremental_object(
    EvictBufferSet &buffers, void *source, size_t size, FarObjectEntry *entry,
    uint32_t behavior) {
    if (!ec_update_workers_ || !buffers.direct_builder || ec_batch_uses_split(size)) return false;
    auto *block = static_cast<::FarLib::allocator::BlockHead *>(source) - 1;
    if (!::FarLib::allocator::ec_write_source_borrowed(block)) return false;
    // Failure is monotone in this runtime. Once degraded reads can exist, do
    // not start new in-place updates; use the existing fresh-group writer.
    for (size_t ep = 0; ep < ec_recovery_endpoint_count(); ++ep)
        if (ec_recovery_endpoint_is_dead(ep)) return false;
    auto &worker = ec_update_workers_[buffers.direct_owner];
    if (worker.count == rdma::ec_update::kSlots)
        flush_ec_incremental_objects(buffers, rdma::thread_info.thread_id);
    SmallObjectStripeManager::SlotReuse reservation;
    {
        profile::evict_breakdown::Scope timer(buffers.breakdown, profile::evict_breakdown::Stage::GroupAllocate);
        if (!remote_allocator.small_object_stripe_manager().begin_slot_reuse(size, behavior, &reservation))
            return false;
    }
    const size_t position = worker.count++;
    auto &transaction = worker.transactions[position];
    transaction.reservation = reservation;
    transaction.source = source;
    transaction.prepared = false;
    transaction.prepared_mask = 0;
    if (++worker.generations[position] == 0) ERROR("ec_update: transaction generation exhausted");
    transaction.generation = worker.generations[position];
    // Only a reservation is visible here, not a REMOTE object. EVICTING's write
    // ref and the borrowed source prevent free/mutation until COMMIT finishes.
    entry->set_remote_addr(reservation.group.segments[reservation.data_shard].addr);
    if (worker.count == rdma::ec_update::kSlots)
        flush_ec_incremental_objects(buffers, rdma::thread_info.thread_id);
    return true;
}

void ConcurrentArrayCache::flush_ec_incremental_objects(
    EvictBufferSet &buffers, size_t client_idx) {
    using namespace rdma::ec_update;
    if (!ec_update_workers_) return;
    const size_t owner = buffers.direct_owner;
    auto &worker = ec_update_workers_[owner];
    if (!worker.count) return;
    auto *control = rdma::ClientControl::get_default();
    auto *transport = control->ec_update_transport();
    auto *client = rdma::get_client(client_idx);
    if (!transport || !client) ERROR("ec_update: missing transport/client");
    const size_t qp_idx = client->get_qp_idx();
    const auto &config = ::FarLib::get_config();
    auto &manager = remote_allocator.small_object_stripe_manager();
    auto segment = [&](size_t i, size_t p) -> const SmallObjectStripeManager::SlotGroupSegment & {
        const auto &r = worker.transactions[i].reservation;
        return r.group.segments[p == 0 ? r.data_shard : p + 3];
    };
    auto run_phase = [&](Op op, auto selected) {
        bool active[kSlots][3]{};
        for (size_t i = 0; i < worker.count; ++i) {
            auto &t = worker.transactions[i];
            for (size_t p = 0; p < 3; ++p) {
                if (!selected(t, p)) continue;
                active[i][p] = true;
                auto &x = transport->get(owner, i, p);
                transport->begin(x);
                auto &m = x.request;
                m.magic = kMagic; m.version = kVersion; m.op = op; m.status = Status::Ok;
                m.generation = t.generation; m.owner = owner; m.slot = i;
                m.participant = p; m.response = 0; m.reserved = 0;
                m.offset = config.map_remote_addr(segment(i, p).addr).second;
                m.bytes = t.reservation.group.slot_size;
                if (op == Op::DataPrepare) {
                    std::memcpy(m.payload, t.source, t.reservation.size);
                    std::memset(m.payload + t.reservation.size, 0, m.bytes - t.reservation.size);
                }
                transport->arm(x);
            }
        }
        ++worker.phases;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            for (size_t ep = 0; ep < static_cast<size_t>(config.server_count); ++ep) {
                if (ec_recovery_endpoint_is_dead(ep)) continue;
                ClientTransport::Exchange *pending[kSlots * 3];
                size_t count = 0;
                for (size_t i = 0; i < worker.count; ++i)
                    for (size_t p = 0; p < 3; ++p) {
                        auto &x = transport->get(owner, i, p);
                        if (active[i][p] && !x.posted && segment(i, p).endpoint_idx == ep)
                            pending[count++] = &x;
                    }
                if (count) {
                    profile::evict_breakdown::Scope timer(buffers.breakdown, profile::evict_breakdown::Stage::PostSend);
                    const size_t accepted = transport->post(pending, count,
                        client->get_endpoint_data_qp_local(ep, qp_idx)->queue_pair);
                    for (size_t j = 0; j < accepted; ++j)
                        worker.request_bytes += wire_bytes(pending[j]->request);
                }
            }
            {
                profile::evict_breakdown::Scope timer(buffers.breakdown, profile::evict_breakdown::Stage::CqProcess);
                (void)check_cq_idx_with_client_idx(qp_idx, client_idx);
            }
            bool done = true;
            for (size_t i = 0; i < worker.count; ++i)
                for (size_t p = 0; p < 3; ++p) {
                    if (!active[i][p]) continue;
                    auto &x = transport->get(owner, i, p);
                    const bool dead = ec_recovery_endpoint_is_dead(segment(i, p).endpoint_idx);
                    if (dead && !x.posted) x.send_done.store(true, std::memory_order_release);
                    if (!x.send_done.load(std::memory_order_acquire) ||
                        (!dead && x.response_state.load(std::memory_order_acquire) != 3)) done = false;
                }
            if (done) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                // A SEND CQE only proves transport delivery, not participant
                // execution. Fence an unresponsive endpoint before excluding
                // it, and drain flushed SENDs before recycling local storage.
                bool fenced = false;
                for (size_t i = 0; i < worker.count; ++i)
                    for (size_t p = 0; p < 3; ++p) {
                        if (!active[i][p]) continue;
                        auto &x = transport->get(owner, i, p);
                        const size_t ep = segment(i, p).endpoint_idx;
                        if (ec_recovery_endpoint_is_dead(ep) ||
                            (x.send_done.load(std::memory_order_acquire) &&
                             x.response_state.load(std::memory_order_acquire) == 3)) continue;
                        control->fence_ec_update_endpoint(ep);
                        ibv_wc failure{};
                        failure.status = IBV_WC_RETRY_EXC_ERR;
                        failure.qp_num = client->get_endpoint_data_qp_local(ep, qp_idx)->queue_pair->qp_num;
                        note_ec_recovery_error_wc(failure);
                        fenced = true;
                    }
                if (!fenced) ERROR("ec_update: fenced QP failed to drain");
                deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            }
        }
    };
    auto ok = [&](size_t i, size_t p) {
        auto &x = transport->get(owner, i, p);
        return !ec_recovery_endpoint_is_dead(segment(i, p).endpoint_idx) &&
               x.response_state.load(std::memory_order_acquire) == 3 && x.reply.status == Status::Ok;
    };

    run_phase(Op::DataPrepare, [](const auto &, size_t p) { return p == 0; });
    for (size_t i = 0; i < worker.count; ++i) {
        auto &t = worker.transactions[i];
        if (!ok(i, 0)) continue;
        t.prepared_mask = 1;
        auto &p0 = transport->get(owner, i, 1);
        auto &p1 = transport->get(owner, i, 2);
        transport->begin(p0); transport->begin(p1);
        profile::evict_breakdown::Scope timer(buffers.breakdown, profile::evict_breakdown::Stage::Encode);
        if (!small_object_stripe_encode_parity_delta(t.reservation.data_shard,
                transport->get(owner, i, 0).reply.payload, t.reservation.group.slot_size,
                p0.request.payload, p1.request.payload)) ERROR("ec_update: ISA-L delta encode rejected");
    }
    run_phase(Op::ParityPrepare, [](const auto &t, size_t p) { return t.prepared_mask == 1 && p != 0; });
    for (size_t i = 0; i < worker.count; ++i) {
        auto &t = worker.transactions[i];
        if (!(t.prepared_mask & 1)) continue;
        for (size_t p = 1; p < 3; ++p) if (ok(i, p)) t.prepared_mask |= 1u << p;
        t.prepared = t.prepared_mask == 7;
    }
    run_phase(Op::Abort, [](const auto &t, size_t p) {
        return !t.prepared && (t.prepared_mask & (1u << p));
    });
    for (size_t i = 0; i < worker.count; ++i) {
        const auto &t = worker.transactions[i];
        if (t.prepared) continue;
        for (size_t p = 0; p < 3; ++p)
            if ((t.prepared_mask & (1u << p)) &&
                !ec_recovery_endpoint_is_dead(segment(i, p).endpoint_idx) && !ok(i, p))
                ERROR("ec_update: prepared participant refused ABORT");
    }
    // The coordinator's decision is now COMMIT; never abort these transactions,
    // even when an endpoint fails between the following participant ACKs.
    run_phase(Op::Commit, [](const auto &t, size_t) { return t.prepared; });
    for (size_t i = 0; i < worker.count; ++i) {
        auto &t = worker.transactions[i];
        if (t.prepared) {
            unsigned survivors = 0;
            for (const auto &s : t.reservation.group.segments)
                survivors += !ec_recovery_endpoint_is_dead(s.endpoint_idx);
            if (survivors < 4) ERROR("ec_update: fewer than four coherent survivors");
            for (size_t p = 0; p < 3; ++p)
                if (!ec_recovery_endpoint_is_dead(segment(i, p).endpoint_idx) && !ok(i, p))
                    ERROR("ec_update: participant refused irrevocable COMMIT");
            if (!manager.commit_slot_reuse(t.reservation)) ERROR("ec_update: stale commit reservation");
            ++worker.committed; worker.payload_bytes += t.reservation.size;
            auto *block = static_cast<::FarLib::allocator::BlockHead *>(t.source) - 1;
            ::FarLib::allocator::release_ec_write_source(block);
            complete_evict_writeback(t.source);
        } else {
            if (!manager.abort_slot_reuse(t.reservation)) ERROR("ec_update: stale abort reservation");
            ++worker.aborted;
            auto *block = static_cast<::FarLib::allocator::BlockHead *>(t.source) - 1;
            const auto object = block->obj_meta_data.load(std::memory_order_acquire);
            if (object.is_null()) ERROR("ec_update: source lost while write ref held");
            auto &entry = get_entry_of(object);
            // Keep the same borrowed source/write reference. Only the failed
            // reservation is retired; the immutable fresh-group path finishes
            // this eviction and publishes its replacement exactly once.
            stage_ec_direct_object(buffers, t.source, t.reservation.size, &entry, t.reservation.behavior_group);
        }
        t = {};
    }
    worker.count = 0;
}
} // namespace FarLib::cache
