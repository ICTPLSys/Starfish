#pragma once
#include "cache/concurrent_cache.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "utils/ec_rmw_timing.hpp"
#include "cache/alloc/ec_rmw_fallback_policy.hpp"

namespace FarLib::cache {
void prepare_ec_rmw_timing(size_t owners) {
    (void)profile::ec_rmw_timing::owner(0, owners);
}
namespace ec_rmw_detail {
using Transaction = ec_rmw_runtime::Transaction;
using Completion = rdma::ec_rmw::CompletionState;
constexpr uint8_t All = 7;
inline const SmallObjectStripeManager::SlotGroupSegment &segment(const Transaction &t, size_t p) {
    return t.reservation.group.segments[p == 0 ? t.reservation.data_shard : p + 3];
}
inline void collect(rdma::ec_rmw::ClientTransport &transport, size_t owner,
                    size_t batch, size_t slot, Transaction &t, bool read,
                    profile::ec_rmw_timing::Object *timing = nullptr) {
    const uint8_t posted = read ? t.read_posted : t.write_posted;
    uint8_t &terminal = read ? t.read_terminal : t.write_terminal;
    uint8_t &errors = read ? t.read_error : t.write_error;
    for (size_t p = 0; p < 3; ++p) {
        const uint8_t bit = uint8_t(1u << p);
        if (!(posted & bit) || (terminal & bit)) continue;
        auto &request = transport.get(owner, batch, slot, p).request;
        const auto state = request.load_state();
        if (state == Completion::pending) continue;
        terminal |= bit;
        if (state == Completion::error) errors |= bit;
        else if (!read) t.write_success |= bit;
        // Producer retires hardware ownership only after acquiring its CQE.
        request.posted = false;
    }
    // Cancel unaccepted READs only. Every accepted READ must drain before abort.
    if (read && errors) terminal |= uint8_t(All & ~posted);
    if (timing && terminal == All && !errors) {
        auto &end = read ? timing->read_end : timing->write_end;
        if (!end) end = get_cycles();
    }
}
} // namespace ec_rmw_detail

bool ConcurrentArrayCache::handle_ec_rmw_complete(const ibv_wc &wc) {
    if (!::FarLib::get_config().ft_incremental_one_sided ||
        !rdma::ec_rmw::ClientTransport::is_wr_id(wc.wr_id)) return false;
    auto *transport = rdma::ClientControl::get_default()->ec_rmw_transport();
    if (!transport) return false;
    // Failure must be visible before a coordinator sees terminal error state.
    if (wc.status != IBV_WC_SUCCESS) note_ec_recovery_error_wc(wc);
    return transport->handle(wc);
}

bool ConcurrentArrayCache::try_publish_ec_rmw_address(
    void *source, uint64_t address, size_t client_idx, const uint64_t *expected_old) {
    auto *block = static_cast<::FarLib::allocator::BlockHead *>(source) - 1;
    const auto obj = block->obj_meta_data.load(std::memory_order_acquire);
    if (obj.is_null()) ERROR("ec_rmw: object disappeared while source borrowed");
    auto *entry = obj.get_entry_ptr();
    auto state = entry->load_state(std::memory_order_acquire);
    if (state.invalid || entry->local_addr() != source) return false;
    if (state.state == FREE || state.ref_cnt == 0)
        ERROR("ec_rmw: source lost its write reference");
    auto locked = state;
    locked.invalid = 1;
    if (!entry->cas_state_weak(state, locked)) return false;
    const auto current = block->obj_meta_data.load(std::memory_order_acquire);
    const bool matches = !current.is_null() && current.get_entry_ptr() == entry &&
                         entry->local_addr() == source;
    if (matches) {
        if (expected_old && entry->remote_addr() != *expected_old)
            ERROR("ec_rmw: old remote address ownership changed");
        entry->set_remote_addr(address);
        entry->set_client_idx(client_idx);
    }
    auto unlocked = locked;
    unlocked.invalid = 0;
    ASSERT(entry->cas_state_strong(locked, unlocked));
    return matches;
}

bool ConcurrentArrayCache::stage_ec_rmw_object(
    EvictBufferSet &buffers, void *source, size_t size, FarObjectEntry *entry,
    uint32_t behavior) {
    // Consume the per-object handoff exactly once; persistent buffers must
    // not carry eligibility into another object or non-exclusive path.
    const bool repair_eligible = ::FarLib::get_config().exclusive_cache &&
                                 buffers.staging_repair_eligible;
    buffers.staging_repair_eligible = false;
    if (!ec_rmw_workers_ || !buffers.direct_builder || !source || !entry ||
        ec_batch_uses_split(size) || size == 0 || size > rdma::ec_rmw::kMaxBytes ||
        !::FarLib::allocator::ec_write_source_borrowed(
            static_cast<::FarLib::allocator::BlockHead *>(source) - 1)) return false;
    const size_t owner = buffers.direct_owner;
    if (owner >= ec_persistent_owner_count_) ERROR("ec_rmw: invalid logical owner");
    profile::evict_breakdown::Scope progress_scope(
        buffers.breakdown, profile::evict_breakdown::Stage::RmwControl);
    auto &worker = ec_rmw_workers_[owner];
    // Captured by try_evict under its move lock, not reread from a stale entry.
    const uint64_t original = buffers.staging_original_remote_addr;
    auto *batch = worker.find_fill_batch();
    while (!batch) {
        // Genuine bounded-bank backpressure progresses all batches/owner CQs.
        flush_ec_rmw_objects(buffers, rdma::thread_info.thread_id);
        batch = worker.find_fill_batch();
        // After a fault, all producer banks may wait for replacement capacity.
        // Do not pin every eviction OS worker in a tight retry loop while the
        // repair fibres sharing this cluster need to finish reconstruction.
        if (ec_rmw_runtime::allow_background_bank_yield(
                ::FarLib::get_config().ft_background_rebuild,
                ec_recovery_dead_endpoints_.load(std::memory_order_acquire) != 0,
                batch != nullptr))
            uthread::yield();
    }
    profile::evict_breakdown::Scope enqueue_scope(
        buffers.breakdown, profile::evict_breakdown::Stage::RmwPrepare);
    auto *timing_owner = profile::ec_rmw_timing::owner(owner, ec_persistent_owner_count_);
    const size_t batch_id = size_t(batch - worker.batches.data());
    if (timing_owner && batch->count == 0) timing_owner->begin(batch_id, 0);
    auto *timing = timing_owner && timing_owner->batch[batch_id].selected
        ? &timing_owner->batch[batch_id] : nullptr;
    const uint64_t stage_begin = timing ? get_cycles() : 0;
    auto &t = batch->transactions[batch->count++];
    t.source = source;
    t.block = static_cast<::FarLib::allocator::BlockHead *>(source) - 1;
    t.original_remote = original;
    t.repair_eligible = repair_eligible;
    t.size = uint32_t(size);
    t.behavior_group = behavior;
    ++worker.attempted;
    ++worker.stages;
    buffers.last_stage_owns_old_remote = true;
    worker.inflight_highwater = std::max<uint64_t>(worker.inflight_highwater,
                                                 worker.inflight_batches());
    if (timing) {
        timing->objects = batch->count;
        const uint64_t elapsed = get_cycles() - stage_begin;
        timing->object[batch->count - 1].prepare += elapsed;
        timing->object[batch->count - 1].prepare_enqueue += elapsed;
    }
    if (batch->count == rdma::ec_rmw::kBatchObjects) {
        ++worker.batches_full;
        flush_ec_rmw_objects(buffers, rdma::thread_info.thread_id);
    }
    return true;
}

void ConcurrentArrayCache::flush_ec_rmw_objects(EvictBufferSet &buffers, size_t client_idx) {
    using namespace ec_rmw_detail;
    using BatchState = ec_rmw_runtime::BatchState;
    using PendingList = ec_rmw_runtime::PendingList;
    if (!ec_rmw_workers_ || buffers.direct_owner >= ec_persistent_owner_count_) return;
    profile::evict_breakdown::Scope progress_scope(
        buffers.breakdown, profile::evict_breakdown::Stage::RmwControl);
    const size_t owner = buffers.direct_owner;
    auto &worker = ec_rmw_workers_[owner];
    auto *control = rdma::ClientControl::get_default();
    auto *transport = control->ec_rmw_transport();
    if (!transport) ERROR("ec_rmw: transport unavailable");
    auto &manager = remote_allocator.small_object_stripe_manager();
    const auto &config = ::FarLib::get_config();
    auto *timing_owner = profile::ec_rmw_timing::owner(owner, ec_persistent_owner_count_);
    ++worker.polls;
    // Preserve the submitting CQ when a logical producer migrates.
    for (size_t b = 0; b < worker.batches.size(); ++b) {
        auto &batch = worker.batches[b];
        if (batch.state == BatchState::Free) continue;
        if (batch.count == 0) { batch.reset(); continue; }
        if (batch.state == BatchState::Fill) {
            if (batch.count < rdma::ec_rmw::kBatchObjects) ++worker.batches_tail;
            batch.state = BatchState::Reading;
            batch.submit_client = client_idx;
            batch.submit_qp = rdma::get_client(client_idx)->get_qp_idx();
        }
        bool polled = false;
        for (size_t previous = 0; previous < b; ++previous) {
            const auto &other = worker.batches[previous];
            polled |= other.state != BatchState::Free &&
                      other.submit_client == batch.submit_client && other.submit_qp == batch.submit_qp;
        }
        if (!polled) {
            profile::evict_breakdown::Scope scope(buffers.breakdown,
                profile::evict_breakdown::Stage::CqProcess);
            (void)check_cq_idx_with_client_idx(batch.submit_qp, batch.submit_client);
        }
    }
    for (size_t b = 0; b < worker.batches.size(); ++b) {
        auto &batch = worker.batches[b];
        if (batch.state == BatchState::Free) continue;
        auto *client = rdma::get_client(batch.submit_client);
        if (!client) ERROR("ec_rmw: missing submitting client");
        auto *timing = timing_owner && timing_owner->batch[b].selected
            ? &timing_owner->batch[b] : nullptr;
        auto retire_old = [&](Transaction &t) {
            if (!t.old_remote_released) {
                if (t.original_remote != FarObjectEntry::RemoteAddrInvalid48)
                    remote_allocator.deallocate(t.original_remote);
                t.old_remote_released = true;
            }
        };
        auto fallback = [&](Transaction &t, bool abort) {
            const bool fault_fallback = abort &&
                ec_rmw_runtime::allow_read_failure_fallback(
                    config.ft_rmw_read_failure_fallback,
                    config.ft_incremental_one_sided, t.read_error,
                    t.read_terminal, t.writes_decided);
            if (::FarLib::ec_benchmark_phase::enabled() && !fault_fallback)
                ERROR("ec_benchmark_phase: fresh fallback prohibited after first Work; failed READ requires recovery");
            if (abort && (t.read_terminal != All || t.read_error == 0))
                ERROR("ec_rmw: failure fallback requires terminal failed READs");
            if (timing) timing->object[size_t(&t - batch.transactions.data())].fallback = true;
            if (t.writes_decided) ERROR("ec_rmw: abort after write decision");
            if (t.has_reservation()) {
                if (t.read_terminal != All && t.read_posted != 0)
                    ERROR("ec_rmw: abort before accepted reads drained");
                if (!manager.abort_slot_reuse(t.reservation)) ERROR("ec_rmw: stale abort reservation");
            }
            // Never restore original_remote: it may already have been freed.
            // Transfer the same borrowed source/ref to the immutable writer.
            // In one-sided mode the immutable writer also resolves the entry
            // from the borrowed source at publication, not a cached pointer.
            if (!stage_ec_direct_object(buffers, t.source, t.size, nullptr,
                                        t.behavior_group, fault_fallback, fault_fallback))
                ERROR("ec_rmw: immutable fallback rejected borrowed source");
            retire_old(t);
            ++worker.fallback;
            if (abort) ++worker.aborted;
            if (fault_fallback) {
                ++worker.replacement_objects;
                // Rare error-path evidence, once per logical owner. The existing
                // worker-local counters retain the complete fallback total.
                if (worker.aborted == 1)
                    std::cout << "INFO: ec_rmw read_failure_fallback owner=" << owner
                              << " read_error_mask=" << unsigned(t.read_error)
                              << " source_bytes=" << t.size
                              << " reads_drained=1 write_decision=0" << std::endl;
            }
            t.clear();
        };
        bool grew_backup = false;
        auto grow_backup = [&](Transaction &t) {
            if (!manager.backup_growth_enabled()) return false;
            if (t.has_reservation() || t.read_posted || t.write_posted)
                ERROR("ec_rmw: growth requires an unreserved transaction");
            if (!stage_ec_direct_object(buffers, t.source, t.size, nullptr,
                                        t.behavior_group, true))
                return false;
            retire_old(t);
            ++worker.growth_objects;
            worker.growth_payload_bytes += t.size;
            t.clear();
            grew_backup = true;
            return true;
        };
        auto replace_failed_capacity = [&](Transaction &t) {
            const bool source_eligible = t.repair_eligible ||
                manager.data_address_needs_replacement(t.original_remote);
            const bool background_rebuild = ::FarLib::get_config().ft_background_rebuild &&
                background_rebuild_state_.fully_rebuilt.load(std::memory_order_acquire);
            if (!ec_rmw_runtime::allow_capacity_replacement(
                    manager.recovery_replacement_enabled(), source_eligible,
                    background_rebuild))
                return false;
            if (t.has_reservation() || t.read_posted || t.write_posted ||
                t.writes_decided)
                ERROR("ec_rmw: replacement requires unreserved, unposted source");
            if (!stage_ec_direct_object(buffers, t.source, t.size, nullptr,
                                        t.behavior_group, true, true))
                ERROR("ec_rmw: repair_credit_or_capacity_exhausted");
            if (!source_eligible && background_rebuild) {
                static std::atomic<bool> reported{false};
                if (!reported.load(std::memory_order_relaxed) &&
                    !reported.exchange(true, std::memory_order_relaxed))
                    std::cout << "INFO: ec_rmw background_capacity_liveness"
                              << " source_eligible=0 bounded_failed_stripe_credit=1"
                              << " no_posted_io=1" << std::endl;
            }
            retire_old(t);
            ++worker.fallback;
            ++worker.replacement_objects;
            if (++worker.capacity_replacement_objects == 1)
                std::cout << "INFO: ec_rmw recovery_replacement owner=" << owner
                          << " source_bytes=" << t.size
                          << " no_posted_io=1" << std::endl;
            t.clear();
            return true;
        };
        const bool bulk_reuse = ::FarLib::ec_benchmark_phase::enabled();
        if (timing) timing->bulk_allocator = bulk_reuse;
        // Add each newly-ready participant exactly once.  The request is
        // armed later, inside submit(), so this organization pass does not
        // initialize a write descriptor while any READ for the transaction is
        // still in flight.
        auto queue_pending = [&](Transaction &t, size_t i, bool read) {
            profile::evict_breakdown::Scope queue_scope(
                buffers.breakdown, profile::evict_breakdown::Stage::RmwSubmit);
            auto &queued = read ? t.read_queued : t.write_queued;
            auto &pending = read ? batch.read_pending : batch.write_pending;
            for (size_t p = 0; p < 3; ++p) {
                const uint8_t bit = uint8_t(1u << p);
                if (queued & bit) continue;
                queued |= bit;
                const uint8_t posted = read ? t.read_posted : t.write_posted;
                const uint8_t terminal = read ? t.read_terminal : t.write_terminal;
                if ((posted | terminal) & bit) continue;
                const auto &s = segment(t, p);
                auto &x = transport->get(owner, b, i, p);
                pending.append(s.endpoint_idx, &x,
                               uint16_t(i * rdma::ec_rmw::kParticipants + p));
            }
        };
        if (bulk_reuse) {
            profile::evict_breakdown::Scope prepare_scope(
                buffers.breakdown, profile::evict_breakdown::Stage::RmwPrepare);
            // The allocator groups independent codewords by their metadata
            // Stripe. Requests keep individual generations/update ownership;
            // no lock survives this call and no full batch is required.
            bool visited[rdma::ec_rmw::kBatchObjects]{};
            uint16_t indices[rdma::ec_rmw::kBatchObjects];
            SmallObjectStripeManager::SlotReuse *outputs[rdma::ec_rmw::kBatchObjects];
            auto account = [&](uint64_t elapsed,
                               const profile::ec_rmw_prepare::Trace &trace,
                               size_t first, size_t count, bool other) {
                if (!timing || count == 0) return;
                if (trace.index > trace.stripe_held ||
                    trace.summary + trace.lock_wait + trace.stripe_held > elapsed)
                    ERROR("ec_rmw: invalid bulk preparation accounting");
                const uint64_t values[] = {
                    trace.summary, trace.lock_wait, trace.stripe_held - trace.index,
                    trace.index,
                    elapsed - trace.summary - trace.lock_wait - trace.stripe_held,
                    trace.summary_calls, trace.summary_misses};
                uint64_t quotient[7], remainder[7];
                for (size_t k = 0; k < 7; ++k) {
                    quotient[k] = values[k] / count;
                    remainder[k] = values[k] % count;
                }
                for (size_t n = 0; n < count; ++n) {
                    auto &p = timing->object[indices[first + n]];
                    uint64_t v[7];
                    for (size_t k = 0; k < 7; ++k)
                        v[k] = quotient[k] + uint64_t(n < remainder[k]);
                    const uint64_t total = v[0] + v[1] + v[2] + v[3] + v[4];
                    p.prepare += total;
                    p.prepare_trace.summary += v[0];
                    p.prepare_trace.lock_wait += v[1];
                    p.prepare_trace.stripe_held += v[2] + v[3];
                    p.prepare_trace.index += v[3];
                    p.prepare_trace.summary_calls += v[5];
                    p.prepare_trace.summary_misses += v[6];
                    if (other) {
                        p.prepare_other_group += total;
                        ++p.other_group_calls;
                    } else {
                        p.prepare_same_group += total;
                    }
                }
            };
            for (size_t seed = 0; seed < batch.count; ++seed) {
                auto &first = batch.transactions[seed];
                if (visited[seed] || !first.source || first.has_reservation()) continue;
                const uint64_t gather_begin = timing ? get_cycles() : 0;
                size_t count = 0;
                for (size_t i = seed; i < batch.count; ++i) {
                    auto &t = batch.transactions[i];
                    if (visited[i] || !t.source || t.has_reservation() ||
                        t.size != first.size || t.behavior_group != first.behavior_group) continue;
                    visited[i] = true;
                    indices[count] = uint16_t(i);
                    outputs[count++] = &t.reservation;
                }
                if (timing) {
                    const uint64_t elapsed = get_cycles() - gather_begin;
                    const uint64_t q = elapsed / count, r = elapsed % count;
                    for (size_t n = 0; n < count; ++n)
                        timing->object[indices[n]].prepare += q + uint64_t(n < r);
                }
                size_t acquired = 0;
                bool busy = false;
                constexpr uint32_t order[] = {0, 3, 1, 4, 2, 5};
                for (uint32_t offset : order) {
                    const size_t pending = count - acquired;
                    if (pending == 0) break;
                    const uint32_t key = (first.behavior_group + offset) %
                        ec_batch::kEcBatchBehaviorGroupCount;
                    profile::ec_rmw_prepare::Trace trace;
                    bool attempt_busy = false;
                    const uint64_t before = timing ? get_cycles() : 0;
                    size_t n;
                    {
                        profile::evict_breakdown::Scope scope(buffers.breakdown,
                            profile::evict_breakdown::Stage::GroupAllocate);
                        n = manager.begin_slot_reuse_indexed_batch(
                            first.size, key, outputs + acquired, pending,
                            &attempt_busy, timing ? &trace : nullptr);
                    }
                    if (n > pending) ERROR("ec_rmw: allocator exceeded batch capacity");
                    if (timing)
                        account(get_cycles() - before, trace, acquired, pending, offset != 0);
                    acquired += n;
                    busy |= attempt_busy;
                }
                const uint64_t prepared_at = timing ? get_cycles() : 0;
                for (size_t n = 0; n < count; ++n) {
                    auto &t = batch.transactions[indices[n]];
                    if (n < acquired) {
                        t.data_shard = t.reservation.data_shard;
                        if (timing && timing->object[indices[n]].busy_begin)
                            timing->object[indices[n]].busy_end = prepared_at;
                    } else if (!busy && (grow_backup(t) || replace_failed_capacity(t))) {
                        continue;
                    } else if (timing && busy) {
                        ++timing->busy_attempts;
                        auto &p = timing->object[indices[n]];
                        if (!p.busy_begin) p.busy_begin = prepared_at;
                    }
                }
            }
        }
        if (grew_backup) {
            const auto status = buffers.direct_builder->flush();
            if (status != ec_batch::EcBatchStatus::kOk)
                ERROR("ec_rmw: cannot seal backup growth groups");
            auto *growth_client = rdma::get_client(client_idx);
            if (!growth_client) ERROR("ec_rmw: missing growth client");
            (void)post_ec_direct_pending(buffers, client_idx,
                                         growth_client->get_qp_idx());
        }
        // Sole reservation/tag step, in batch preparation. Busy entries do
        // not block other transactions in the batch from completing writes.
        {
        profile::evict_breakdown::Scope collect_scope(
            buffers.breakdown, profile::evict_breakdown::Stage::RmwCollect);
        for (size_t i = 0; i < batch.count; ++i) {
            auto &t = batch.transactions[i];
            if (!t.source) continue;
            if (!t.has_reservation()) {
                profile::evict_breakdown::Scope prepare_scope(
                    buffers.breakdown, profile::evict_breakdown::Stage::RmwPrepare);
                if (bulk_reuse) {
                    ++worker.deferred;
                    continue;
                }
                bool busy = false, acquired;
                const uint64_t before = timing ? get_cycles() : 0;
                {
                    profile::evict_breakdown::Scope scope(buffers.breakdown,
                        profile::evict_breakdown::Stage::GroupAllocate);
                    acquired = ::FarLib::ec_benchmark_phase::enabled()
                        ? manager.begin_slot_reuse_indexed(t.size, t.behavior_group, &t.reservation, &busy,
                            timing ? &timing->object[i].prepare_trace : nullptr)
                        : manager.begin_slot_reuse(t.size, t.behavior_group, &t.reservation, &busy);
                    if (timing) timing->object[i].prepare_same_group += get_cycles() - before;
                    if (::FarLib::ec_benchmark_phase::enabled() && !acquired) {
                        const uint64_t other_begin = timing ? get_cycles() : 0;
                        // Heat/dirty labels are placement preferences, not
                        // different EC protection levels. The fixed initialized
                        // layout must remain reusable when an object's label
                        // changes in Work. Reservation carries the target's
                        // original group label for commit validation.
                        constexpr uint32_t order[] = {3, 1, 4, 2, 5};
                        for (uint32_t offset : order) {
                            bool other_busy = false;
                            const uint32_t key = (t.behavior_group + offset) %
                                ec_batch::kEcBatchBehaviorGroupCount;
                            acquired = manager.begin_slot_reuse_indexed(
                                t.size, key, &t.reservation, &other_busy,
                                timing ? &timing->object[i].prepare_trace : nullptr);
                            if (timing) ++timing->object[i].other_group_calls;
                            busy |= other_busy;
                            if (acquired) break;
                        }
                        if (timing) timing->object[i].prepare_other_group += get_cycles() - other_begin;
                    }
                }
                if (timing) {
                    const uint64_t now = get_cycles();
                    auto &p = timing->object[i];
                    p.prepare += now - before;
                    if (busy && !acquired) {
                        ++timing->busy_attempts;
                        if (!p.busy_begin) p.busy_begin = now;
                    } else if (acquired && p.busy_begin) p.busy_end = now;
                }
                if (!acquired) {
                    if (::FarLib::ec_benchmark_phase::enabled()) {
                        ++worker.deferred;
                        continue;
                    }
                    if (busy) { ++worker.deferred; continue; }
                    fallback(t, false);
                    continue;
                }
                t.data_shard = t.reservation.data_shard;
            }
            if (!t.target_published) {
                profile::evict_breakdown::Scope publish_scope(
                    buffers.breakdown, profile::evict_breakdown::Stage::RmwPrepare);
                const uint64_t before = timing ? get_cycles() : 0;
                const bool published = try_publish_ec_rmw_address(t.source, segment(t, 0).addr,
                                                batch.submit_client, &t.original_remote);
                const uint64_t publish_end = timing ? get_cycles() : 0;
                if (timing) timing->object[i].prepare_publish += publish_end - before;
                if (!published) {
                    if (timing) timing->object[i].prepare += publish_end - before;
                    continue;
                }
                t.target_published = true;
                const uint64_t retire_begin = timing ? get_cycles() : 0;
                retire_old(t);
                if (timing) {
                    const uint64_t end = get_cycles();
                    timing->object[i].prepare_retire += end - retire_begin;
                    timing->object[i].prepare += end - before;
                }
            }
            if (!t.writes_decided && t.target_published && t.read_queued != All) {
                const uint64_t queue_begin = timing ? get_cycles() : 0;
                queue_pending(t, i, true);
                if (timing) timing->read_submit += get_cycles() - queue_begin;
            }
            collect(*transport, owner, b, i, t, !t.writes_decided,
                    timing ? &timing->object[i] : nullptr);
        }
        }
        auto submit = [&](bool read) {
            profile::evict_breakdown::Scope submit_scope(
                buffers.breakdown, profile::evict_breakdown::Stage::RmwSubmit);
            const uint64_t submit_begin = timing ? get_cycles() : 0;
            // Pending entries were organized when each transaction became
            // newly ready.  Walk only those entries now; no endpoint x object
            // x participant scan is needed on a fully-posted flush.
            auto &pending_list = read ? batch.read_pending : batch.write_pending;
            if (pending_list.empty()) {
                if (timing) (read ? timing->read_submit : timing->write_submit) +=
                    get_cycles() - submit_begin;
                return;
            }
            rdma::ec_rmw::Exchange *post_list[PendingList::kCapacity];
            uint16_t identities[PendingList::kCapacity];
            uint16_t nodes[PendingList::kCapacity];
            for (size_t group = 0; group < pending_list.group_count; ++group) {
                const size_t ep = pending_list.group_endpoint[group];
                const uint16_t head = pending_list.group_head[group];
                if (head == PendingList::kInvalid) continue;
                if (ec_recovery_endpoint_is_dead(ep)) {
                    for (uint16_t node = head; node != PendingList::kInvalid;
                         node = pending_list.entries[node].next) {
                        const size_t identity = pending_list.entries[node].identity;
                        const size_t i = identity / rdma::ec_rmw::kParticipants;
                        const size_t p = identity % rdma::ec_rmw::kParticipants;
                        auto &t = batch.transactions[i];
                        const uint8_t bit = uint8_t(1u << p);
                        (read ? t.read_terminal : t.write_terminal) |= bit;
                        (read ? t.read_error : t.write_error) |= bit;
                    }
                    pending_list.group_head[group] = PendingList::kInvalid;
                    pending_list.group_tail[group] = PendingList::kInvalid;
                    continue;
                }
                size_t count = 0;
                uint16_t previous = PendingList::kInvalid;
                for (uint16_t node = head; node != PendingList::kInvalid;) {
                    const auto &entry = pending_list.entries[node];
                    const uint16_t next = entry.next;
                    const size_t identity = entry.identity;
                    const size_t i = identity / rdma::ec_rmw::kParticipants;
                    const size_t p = identity % rdma::ec_rmw::kParticipants;
                    auto &t = batch.transactions[i];
                    const uint8_t bit = uint8_t(1u << p);
                    uint8_t &terminal = read ? t.read_terminal : t.write_terminal;
                    uint8_t &armed = read ? t.read_armed : t.write_armed;
                    const uint8_t posted = read ? t.read_posted : t.write_posted;
                    // collect() can cancel unaccepted READs, and fallback()
                    // can clear their transaction before this queue is visited.
                    // Unlink those nodes before touching reservation/Exchange.
                    if (!t.source || !t.has_reservation() || !t.target_published ||
                        (read ? t.writes_decided || t.read_error != 0 : !t.writes_decided) ||
                        ((posted | terminal) & bit)) {
                        if (previous == PendingList::kInvalid)
                            pending_list.group_head[group] = next;
                        else
                            pending_list.entries[previous].next = next;
                        if (pending_list.group_tail[group] == node)
                            pending_list.group_tail[group] = previous;
                        node = next;
                        continue;
                    }
                    const auto &s = segment(t, p);
                    if (s.endpoint_idx != ep)
                        ERROR("ec_rmw: pending endpoint mismatch");
                    auto &x = *entry.exchange;
                    if (!(armed & bit)) {
                        const auto [mapped, offset] = config.map_remote_addr(s.addr);
                        if (mapped != ep) ERROR("ec_rmw: endpoint mapping mismatch");
                        const auto &endpoint = control->get_endpoint(ep);
                        const uint64_t address = endpoint.remote_base_addr + offset;
                        const uint32_t bytes = t.reservation.group.slot_size;
                        if (read) transport->arm_read(x, address, endpoint.remote_key, bytes);
                        else if (p != 0) transport->arm_write(x, address, endpoint.remote_key, bytes);
                        else {
                            ibv_send_wr scratch_wr{};
                            ibv_sge scratch_sge{};
                            client->build_send_wr(scratch_wr, scratch_sge, 0, t.source, 1, 0,
                                                 false, IBV_WR_RDMA_WRITE, ep);
                            transport->arm_write(x, address, endpoint.remote_key, bytes,
                                                 t.source, scratch_sge.lkey, t.size);
                        }
                        armed |= bit;
                    }
                    post_list[count] = &x;
                    identities[count] = uint16_t(identity);
                    nodes[count++] = node;
                    previous = node;
                    node = next;
                }
                if (count == 0) continue;
                auto *qp = client->get_endpoint_data_qp_local(ep, batch.submit_qp);
                if (!qp) ERROR("ec_rmw: missing endpoint QP");
                size_t accepted;
                const uint64_t post_begin = timing ? get_cycles() : 0;
                {
                    profile::evict_breakdown::Scope scope(buffers.breakdown,
                        profile::evict_breakdown::Stage::PostSend);
                    accepted = transport->post(post_list, count, qp->queue_pair,
                        timing ? (read ? &timing->read_verbs : &timing->write_verbs) : nullptr);
                }
                if (timing) {
                    (read ? timing->read_transport : timing->write_transport) +=
                        get_cycles() - post_begin;
                    ++(read ? timing->read_post_calls : timing->write_post_calls);
                    (read ? timing->read_post_attempted_wrs : timing->write_post_attempted_wrs) += count;
                    (read ? timing->read_post_accepted_wrs : timing->write_post_accepted_wrs) += accepted;
                    if (accepted != count)
                        ++(read ? timing->read_post_partial_calls : timing->write_post_partial_calls);
                    auto &maximum = read ? timing->read_post_max_chain : timing->write_post_max_chain;
                    maximum = std::max<uint64_t>(maximum, count);
                }
                uint64_t written_bytes = 0;
                for (size_t n = 0; n < accepted; ++n) {
                    const size_t identity = identities[n];
                    const size_t i = identity / rdma::ec_rmw::kParticipants;
                    const size_t p = identity % rdma::ec_rmw::kParticipants;
                    auto &t = batch.transactions[i];
                    const uint8_t bit = uint8_t(1u << p);
                    const uint32_t bytes = post_list[n]->request.bytes;
                    if (timing) {
                        auto &begin = read ? timing->object[i].read_begin : timing->object[i].write_begin;
                        if (!begin) begin = post_begin;
                    }
                    if (read) {
                        t.read_posted |= bit;
                        ++worker.read_posts; worker.read_bytes += bytes;
                        profile::work_traffic::count_rmw_read(bytes);
                        profile::count_evac_phase_rdma_read(profile::EvacRdmaTraffic::EcBatch, bytes);
                    } else {
                        t.write_posted |= bit;
                        ++worker.write_posts; worker.write_bytes += bytes;
                        written_bytes += bytes;
                        profile::count_rdma_write_post(bytes);
                        profile::count_evac_phase_rdma_write(profile::EvacRdmaTraffic::EcBatch, bytes);
                    }
                }
                if (accepted != 0) {
                    const uint16_t new_head = accepted == count
                        ? PendingList::kInvalid : nodes[accepted];
                    pending_list.group_head[group] = new_head;
                    if (new_head == PendingList::kInvalid)
                        pending_list.group_tail[group] = PendingList::kInvalid;
                }
                if (!read && accepted) {
                    profile::count_evacuation_bytes(written_bytes);
                    profile::count_evac_flush(accepted, accepted == EvictBatchSize);
                }
            }
            if (timing) (read ? timing->read_submit : timing->write_submit) += get_cycles() - submit_begin;
        };
        submit(true);
        bool ready = true;
        {
        profile::evict_breakdown::Scope collect_scope(
            buffers.breakdown, profile::evict_breakdown::Stage::RmwCollect);
        for (size_t i = 0; i < batch.count; ++i) {
            auto &t = batch.transactions[i];
            if (!t.source || !t.has_reservation() || !t.target_published || t.writes_decided) continue;
            collect(*transport, owner, b, i, t, true, timing ? &timing->object[i] : nullptr);
            if (t.read_error && t.read_terminal == All) { fallback(t, true); continue; }
            if (t.read_terminal != All) ready = false;
        }
        }
        if (ready) {
            profile::evict_breakdown::Scope scope(buffers.breakdown,
                profile::evict_breakdown::Stage::RmwEncode);
            for (size_t i = 0; i < batch.count; ++i) {
                auto &t = batch.transactions[i];
                if (!t.source || !t.has_reservation() || !t.target_published ||
                    t.writes_decided || t.read_terminal != All || t.read_error) continue;
                if (timing) timing->object[i].encode_begin = get_cycles();
                const size_t bytes = t.reservation.group.slot_size;
                auto &data = transport->get(owner, b, i, 0);
                auto &p0 = transport->get(owner, b, i, 1);
                auto &p1 = transport->get(owner, b, i, 2);
                const auto *source = static_cast<const uint8_t *>(t.source);
                const uint64_t diff_begin = timing ? get_cycles() : 0;
                // Only the live prefix needs XOR. The tail already holds
                // old_data XOR zero, so never zero it when the new object is
                // shorter than the slot. The word-wise helper vectorizes.
                small_object_stripe_xor_bytes(source, t.size, data.payload);
                const uint64_t diff_end = timing ? get_cycles() : 0;
                if (!small_object_stripe_update_parity(t.reservation.data_shard,
                        data.payload, bytes, p0.payload, p1.payload))
                    ERROR("ec_rmw: ISA-L in-place parity update rejected");
                const uint64_t update_end = timing ? get_cycles() : 0;
                t.encoded = true;
                t.writes_decided = true; // Never abort after fixing new absolute images.
                batch.state = BatchState::Writing;
                if (timing) {
                    auto &p = timing->object[i];
                    const uint64_t end = get_cycles();
                    p.encode = end - p.encode_begin;
                    p.encode_diff = diff_end - diff_begin;
                    p.encode_update = update_end - diff_end;
                }
                const uint64_t queue_begin = timing ? get_cycles() : 0;
                queue_pending(t, i, false);
                if (timing) timing->write_submit += get_cycles() - queue_begin;
            }
        }
        submit(false);
        bool live = false;
        uint16_t completed_indices[rdma::ec_rmw::kBatchObjects];
        const SmallObjectStripeManager::SlotReuse *completed[rdma::ec_rmw::kBatchObjects];
        size_t completed_count = 0;
        {
        profile::evict_breakdown::Scope collect_scope(
            buffers.breakdown, profile::evict_breakdown::Stage::RmwCollect);
        for (size_t i = 0; i < batch.count; ++i) {
            auto &t = batch.transactions[i];
            if (!t.source) continue;
            if (t.writes_decided) {
                collect(*transport, owner, b, i, t, false, timing ? &timing->object[i] : nullptr);
                if (t.write_terminal == All) {
                    profile::evict_breakdown::Scope commit_scope(
                        buffers.breakdown, profile::evict_breakdown::Stage::RmwCommit);
                    const uint64_t commit_begin = timing ? get_cycles() : 0;
                    unsigned survivors = 0;
                    for (size_t shard = 0; shard < 6; ++shard) {
                        if (ec_recovery_endpoint_is_dead(t.reservation.group.segments[shard].endpoint_idx)) continue;
                        if (shard < 4 && shard != t.reservation.data_shard) ++survivors;
                        else {
                            const size_t p = shard == t.reservation.data_shard ? 0 : shard - 3;
                            if (t.write_success & (1u << p)) ++survivors;
                            else ERROR("ec_rmw: surviving participant lacks successful WRITE");
                        }
                    }
                    if (survivors < 4) ERROR("ec_rmw: fewer than four coherent surviving shards");
                    if (bulk_reuse) {
                        completed_indices[completed_count] = uint16_t(i);
                        completed[completed_count++] = &t.reservation;
                        if (timing) timing->object[i].commit += get_cycles() - commit_begin;
                        continue;
                    }
                    if (!manager.commit_slot_reuse(t.reservation)) ERROR("ec_rmw: stale stripe reservation");
                    ::FarLib::allocator::release_ec_write_source(
                        static_cast<::FarLib::allocator::BlockHead *>(t.block));
                    complete_evict_writeback(t.source);
                    ++worker.committed;
                    worker.payload_bytes += t.size;
                    t.clear();
                    if (timing) {
                        timing->object[i].commit = get_cycles() - commit_begin;
                        timing->object[i].committed = true;
                    }
                    continue;
                }
            }
            live = true;
        }
        }
        if (completed_count) {
            profile::evict_breakdown::Scope commit_scope(
                buffers.breakdown, profile::evict_breakdown::Stage::RmwCommit);
            const uint64_t commit_begin = timing ? get_cycles() : 0;
            if (!manager.commit_slot_reuse_batch(completed, completed_count))
                ERROR("ec_rmw: stale bulk stripe reservation");
            if (timing) {
                const uint64_t elapsed = get_cycles() - commit_begin;
                const uint64_t q = elapsed / completed_count, r = elapsed % completed_count;
                for (size_t n = 0; n < completed_count; ++n)
                    timing->object[completed_indices[n]].commit += q + uint64_t(n < r);
            }
            for (size_t n = 0; n < completed_count; ++n) {
                const size_t i = completed_indices[n];
                auto &t = batch.transactions[i];
                const uint64_t release_begin = timing ? get_cycles() : 0;
                ::FarLib::allocator::release_ec_write_source(
                    static_cast<::FarLib::allocator::BlockHead *>(t.block));
                complete_evict_writeback(t.source);
                ++worker.committed;
                worker.payload_bytes += t.size;
                t.clear();
                if (timing) {
                    timing->object[i].commit += get_cycles() - release_begin;
                    timing->object[i].committed = true;
                }
            }
        }
        if (!live) {
            if (timing_owner) timing_owner->finish(b);
            batch.reset();
        }
    }
}

void ConcurrentArrayCache::drain_ec_rmw_worker(EvictBufferSet &buffers, size_t client_idx) {
    if (!ec_rmw_workers_ || buffers.direct_owner >= ec_persistent_owner_count_) return;
    while (ec_rmw_workers_[buffers.direct_owner].pending_count() != 0)
        flush_ec_rmw_objects(buffers, client_idx);
}
} // namespace FarLib::cache
