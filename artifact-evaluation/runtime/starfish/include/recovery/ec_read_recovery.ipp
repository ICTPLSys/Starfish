// ---------------------------------------------------------------------------
// ec_batch read path, stage C1 - the degraded read (read-side recovery).
//
// Included at the end of cache/concurrent_cache.hpp (next to ec_batch_path.ipp),
// so the class is complete here and every member access resolves.
//
// Problem: a small object evicted through the ec_batch path lives in one data
// segment of a slot group (4 data + 2 parity segments, one 4 KB segment per
// shard, six distinct endpoints).  When the server behind one of those
// endpoints is gone, the RDMA READ that fetched the object ends in a *failed*
// completion (retry exhausted, then flushed), which handle_work_complete()
// drops; the entry stays FETCHING and fetch_wait_until_local() spins forever.
//
// This file makes that read recoverable:
//
//   note_ec_recovery_error_wc()   the failed CQE names its endpoint (qp_num);
//                                 that endpoint is remembered as dead and is
//                                 never posted to again (its RC QP is in error
//                                 and the tree has no reset path)
//   post_ec_degraded_read()       reverse-maps the object's remote address to
//                                 its slot group, reads the surviving segments
//                                 into registered scratch buffers and posts one
//                                 tagged RDMA READ per surviving segment
//   handle_ec_read_segment_complete()  the last tagged CQE rebuilds the missing
//                                 segment straight into the object's local slot
//                                 (RS(4,2), any four survivors) and converges
//                                 the entry to LOCAL exactly like a completed
//                                 single-sided read
//   ec_recovery_assist_wait()     the in-flight case: a read posted before the
//                                 failure will never complete, so the fetch
//                                 wait/check loops re-post the same entry once
//                                 as a degraded read
//
// Everything here is gated on get_config().is_ec_batch_mode(): the endpoint
// bitmap and the scratch pool are only built in that mode, no tagged read wr_id
// can exist otherwise, and ft_method=none keeps its behaviour verbatim (its
// dispatch adds one bit test that can never be true).
// ---------------------------------------------------------------------------
#pragma once

#include "cache/concurrent_cache.hpp"

namespace FarLib::cache {

// How long a degraded read may wait for temporary memory or for a send
// queue slot before it gives up and leaves the entry to the legacy path.  The
// wait path reaps CQEs, which is what returns scratch slots, so this is a
// backstop and not a rate limiter.
inline constexpr uint64_t kEcRecoveryScratchSpinLimit = 1ull << 22;

// One entry the degraded read path needs: the six segments of the group of the
// object whose remote address is `remote_addr`, plus the plan derived from the
// endpoints that are still alive.
struct EcRecoveryGroupView {
    SmallObjectStripeManager::SlotGroupHandle group;
    uint8_t own_shard_idx = 0;
    uint8_t alive_mask = 0;
    ec_read_recovery::EcReadPlan plan;
    bool valid = false;
};

inline size_t ConcurrentArrayCache::ec_recovery_endpoint_count() const {
    return ec_endpoint_dead_count_;
}

inline bool ConcurrentArrayCache::ec_recovery_endpoint_is_dead(
    size_t endpoint_idx) const {
    if (ec_endpoint_dead_ == nullptr || endpoint_idx >= ec_endpoint_dead_count_) {
        return false;
    }
    return ec_endpoint_dead_[endpoint_idx].load(std::memory_order_acquire);
}

// wr_id classifier of the degraded read round; false for every ordinary object
// pointer and for every ec_batch write tag.
inline bool ConcurrentArrayCache::ec_recovery_is_read_wr_id(
    uint64_t wr_id) const {
    if (!::FarLib::get_config().is_ec_batch_mode()) return false;
    return ec_read_recovery::is_ec_read_wr_id(wr_id);
}

// Endpoint of a data QP, looked up by the qp_num of a completion.  The CQEs of
// one endpoint are reaped by several threads (and full_checker() reaps other
// clients' CQs), so the QP number - not the polling thread - is the identity.
// Returns -1 when no data QP carries that number (e.g. a control QP).
inline int ConcurrentArrayCache::ec_recovery_endpoint_for_qp_num(
    uint32_t qp_num) const {
    auto *control = rdma::ClientControl::get_default();
    if (control == nullptr) return -1;
    // One endpoint owns a contiguous array of data QPs (one per thread and
    // qp_count); its length is the one ClientControl built it with.  The
    // accessor for it is private, so it is recomputed from the same config
    // fields (client.cpp: data_qp_count).
    const auto &config = ::FarLib::get_config();
    size_t qp_total = config.max_thread_cnt + 1;
    if (config.enable_eager_evict) qp_total += config.evacuate_thread_cnt;
    qp_total *= static_cast<size_t>(config.qp_count);
    const size_t endpoints = control->get_server_count();
    for (size_t endpoint = 0; endpoint < endpoints; endpoint++) {
        rdma::QueuePair *qps = control->get_endpoint_data_qps_ptr(endpoint);
        if (qps == nullptr) continue;
        for (size_t qp = 0; qp < qp_total; qp++) {
            if (qps[qp].queue_pair == nullptr) continue;
            if (qps[qp].queue_pair->qp_num == qp_num) {
                return static_cast<int>(endpoint);
            }
        }
    }
    return -1;
    return -1;
}

// A failed completion: remember the endpoint that produced it.  The marking is
// one-way (a dead endpoint is never revived), printed once per endpoint, and
// the CQE is otherwise dropped exactly as before.
inline void ConcurrentArrayCache::note_ec_recovery_error_wc(const ibv_wc &wc) {
    ec_recovery_error_wcs_.fetch_add(1, std::memory_order_relaxed);
    const int endpoint_idx = ec_recovery_endpoint_for_qp_num(wc.qp_num);
    if (endpoint_idx >= 0 &&
        static_cast<size_t>(endpoint_idx) < ec_endpoint_dead_count_) {
        bool expected = false;
        if (ec_endpoint_dead_[endpoint_idx].compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            // Anchor the failure wave at the winning CAS.  This is deliberately
            // before allocator bookkeeping and before any log formatting, and
            // the timestamp is retained even when optional profiling is off.
            ec_recovery_profile().note_endpoint_dead(
                ec_recovery_profile_now_ns());
            remote_allocator.small_object_stripe_manager().mark_endpoint_dead(
                static_cast<size_t>(endpoint_idx));
            ec_recovery_dead_endpoints_.fetch_add(1, std::memory_order_relaxed);
            std::cout << "INFO: ec_recovery endpoint_dead endpoint="
                      << endpoint_idx << " qp_num=" << wc.qp_num
                      << " wc_status=" << wc.status << " wr_id=" << wc.wr_id
                      << std::endl;
        }
    } else if (endpoint_idx < 0) {
        static std::atomic<int> unmapped_qp_log{0};
        if (unmapped_qp_log.fetch_add(1, std::memory_order_relaxed) < 10) {
            std::cout << "INFO: ec_recovery error_wc_unmapped_qp qp_num="
                      << wc.qp_num << " wc_status=" << wc.status
                      << " opcode=" << wc.opcode << std::endl;
        }
    }
    // A tagged read failure is one terminal completion of one posted WR, not a
    // reason to return the scratch immediately: other WRs of this same read
    // may still be writing into their temporary segments. The request context
    // returns kRelease only after posting is finished and every actual post has a
    // success or failure completion.
    if (!ec_read_recovery::is_ec_read_wr_id(wc.wr_id)) return;
    const uint64_t token_id = ec_read_recovery::ec_read_wr_id_token(wc.wr_id);
    const uint8_t segment =
        ec_read_recovery::ec_read_wr_id_segment(wc.wr_id);
    ec_read_recovery::EcReadContextEvent event;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kSegmentCompletion);
        event = ec_read_tokens_.complete_segment_event(token_id, segment, false);
    }
    ec_recovery_profile().note_segment_completion(false);
    if (event.kind == ec_read_recovery::EcReadTokenEventKind::kRelease) {
        finish_ec_read_context(token_id, event);
        std::cout << "INFO: ec_recovery read_abandoned token=" << token_id
                  << " qp_num=" << wc.qp_num << " wc_status=" << wc.status
                  << " segment="
                  << static_cast<unsigned>(segment)
                  << std::endl;
    }
}

// Reverse map of one object into its slot group plus the read plan of that
// group.  False when the object is not a group-allocated small object, when the
// group is already settled (its addresses may be reused), when the group lost
// too much to be rebuilt, or when the object's own segment is still readable.
inline bool ConcurrentArrayCache::ec_recovery_group_view(
    FarObjectEntry *entry, uint32_t byte_count, EcRecoveryGroupView *view_out) {
    if (entry == nullptr || view_out == nullptr) return false;
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode() || config.server_count <= 1) return false;
    if (byte_count == 0 || !config.ft_small_object(byte_count)) {
        return false;  // flat / large object path: not recoverable
    }
    const uint64_t remote_addr = entry->remote_addr();
    if (remote_addr == FarObjectEntry::RemoteAddrInvalid48) return false;
    auto &stripe_manager = remote_allocator.small_object_stripe_manager();
    if (!stripe_manager.owns(remote_addr)) return false;
    if (stripe_manager.slot_group_addr_is_settled(remote_addr)) {
        return false;  // dead group: the segments of this offset are gone
    }
    SmallObjectStripeManager::SlotLayout layout;
    if (!stripe_manager.get_slot_layout(remote_addr, &layout)) return false;
    const uint8_t own_shard = layout.data_shard_idx;
    if (own_shard >= kStripeCodecDataShards) return false;
    if (layout.slot_size == 0 || byte_count > layout.slot_size) return false;

    SmallObjectStripeManager::SlotGroupId group_id;
    group_id.stripe_id = layout.stripe_id;
    group_id.slot_id = layout.slot_id;
    EcRecoveryGroupView view;
    if (!stripe_manager.get_slot_group_layout(group_id, &view.group)) {
        return false;  // the group is free again: nothing to rebuild from
    }
    if (view.group.segments[own_shard].shard_idx != own_shard ||
        view.group.segments[own_shard].addr != remote_addr) {
        return false;  // the entry does not point at its own data segment
    }
    uint32_t endpoints[ec_read_recovery::kEcReadSegmentCount];
    for (size_t segment = 0; segment < ec_read_recovery::kEcReadSegmentCount;
         segment++) {
        const auto &seg = view.group.segments[segment];
        if (seg.slot_size < byte_count) return false;
        // The group's segment attribution and the address mapping must agree
        // (the same invariant the group write path asserts before posting).
        if (config.map_remote_addr(seg.addr).first != seg.endpoint_idx) {
            return false;
        }
        endpoints[segment] = seg.endpoint_idx;
    }
    view.own_shard_idx = own_shard;
    view.alive_mask = ec_read_recovery::alive_mask_for_dead_endpoints(
        endpoints, [this](size_t endpoint) {
            return ec_recovery_endpoint_is_dead(endpoint);
        });
    view.plan = ec_read_recovery::make_ec_read_plan(view.alive_mask, own_shard);
    view.valid = view.plan.usable && view.plan.own_shard_missing;
    *view_out = view;
    return view.valid;
}

// ---------------------------------------------------------------------------
// Bounded posting of a degraded read.
//
// post_ec_degraded_read_entry() bounds retries while a send queue
// is exhausted, and it used to poll the CQs in every one of those rounds
// (kEcRecoveryScratchSpinLimit rounds).  While an endpoint is dead the pool can
// stay empty for seconds, so many fetch waits ended up inside that loop at the
// same time.  The two limits below cap a single posting call: at most
// kEcRecoveryPostAttemptLimit rounds and at most kEcRecoveryPostBudgetUs
// microseconds of wall clock.  When the budget is out the call releases the
// token and the scratch slot it took (if any) and returns kUnavailable; the
// caller (ec_recovery_assist_wait() <- fetch_wait_until_local()) retries on a
// later wait round or lets that fetch fail normally, and nothing waits forever.
// The CQ drain of this path is the data CQ of this client only.
// ---------------------------------------------------------------------------
inline constexpr uint64_t kEcRecoveryPostAttemptLimit = 64;
inline constexpr uint64_t kEcRecoveryPostBudgetUs = 2000;  // 2 ms

inline uint64_t ec_recovery_now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline uint64_t ec_recovery_post_deadline_us() {
    return ec_recovery_now_us() + kEcRecoveryPostBudgetUs;
}

// True once this posting call has used up its budget: `attempts` rounds done or
// the wall-clock deadline reached.
inline bool ec_recovery_post_budget_out(uint64_t attempts,
                                        uint64_t deadline_us) {
    if (attempts >= kEcRecoveryPostAttemptLimit) return true;
    return ec_recovery_now_us() >= deadline_us;
}
// Posts the degraded read of one object: the surviving segments of its slot
// group are read into the resident scratch buffers, each with a wr_id that
// carries (token, segment). The ordinary-READ gate stays closed on retries.
// kInFlight also covers the old-READ drain and a fetch already completed by
// another waiter. No survivor round starts until the old READs have ended.
// The only block-owned recovery state: gate bit plus one owner/lifetime pin.
// The allocator observes the owner bit just as it observes ordinary READ pins.
inline bool pin_fetch_entry_for_recovery(FarObjectEntry *entry) {
    auto state = entry->load_state(std::memory_order_acquire);
    for (;;) {
        if (state.state != EntryState::FETCHING || state.invalid ||
            state.ref_cnt == 255) return false;
        auto pinned = state;
        pinned.inc_ref_cnt();
        if (entry->cas_state_weak(state, pinned)) return true;
    }
}

inline void unpin_fetch_entry_for_recovery(FarObjectEntry *entry) {
    auto state = entry->load_state(std::memory_order_acquire);
    for (;;) {
        // Respect the existing per-entry publication lock. Never modify the
        // publisher's expected state while it owns that lock.
        if (state.invalid) {
            state = entry->load_state(std::memory_order_acquire);
            continue;
        }
        ASSERT(state.ref_cnt != 0);
        auto unpinned = state;
        unpinned.dec_ref_cnt();
        if (entry->cas_state_weak(state, unpinned)) return;
    }
}

inline void release_degraded_read_owner(::FarLib::allocator::BlockHead *block) {
    // The matching entry reference is still held. Its later release permits
    // deallocate(), which supplies the normal allocator notification.
    ASSERT(!block->obj_meta_data.load(std::memory_order_acquire).is_null());
    block->normal_read_recovery_active.fetch_and(
        uint8_t{1}, std::memory_order_release);
}

inline ec_read_recovery::EcReadPostResult
ConcurrentArrayCache::post_ec_degraded_read_entry(FarObjectEntry *entry,
                                                  size_t client_idx,
                                                  void *expected_local_addr) {
    using Result = ec_read_recovery::EcReadPostResult;
    using Outcome = EcRecoveryProfile::PostOutcome;
    EcRecoveryProfile::PostAttemptScope profile_post_attempt(ec_recovery_profile());
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode() || entry == nullptr) {
        profile_post_attempt.set_outcome(Outcome::kUnavailable);
        return Result::kUnavailable;
    }
    // Pin the object before inspecting its local block. In particular Lite
    // accessors do not otherwise carry a reference through the drain/claim
    // window. Existing deallocate() waits for this reference to be returned.
    if (!pin_fetch_entry_for_recovery(entry)) {
        profile_post_attempt.set_outcome(Outcome::kInFlight);
        return Result::kInFlight;
    }
    struct EntryPinGuard {
        FarObjectEntry *entry;
        bool active = true;
        ~EntryPinGuard() { if (active) unpin_fetch_entry_for_recovery(entry); }
    } entry_pin{entry};
    void *local_addr = entry->local_addr();
    if (expected_local_addr != nullptr && expected_local_addr != local_addr) {
        profile_post_attempt.set_outcome(Outcome::kUnavailable);
        return Result::kUnavailable;
    }
    if (local_addr == nullptr ||
        entry->load_state(std::memory_order_acquire).state != EntryState::FETCHING) {
        profile_post_attempt.set_outcome(Outcome::kInFlight);
        return Result::kInFlight;
    }
    auto *target_block =
        static_cast<::FarLib::allocator::BlockHead *>(local_addr) - 1;
    // Cheap object-local duplicate test, before group lookup or buffer borrowing.
    if (::FarLib::allocator::degraded_read_owned(target_block)) {
        ec_recovery_post_duplicates_.fetch_add(1, std::memory_order_relaxed);
        profile_post_attempt.set_outcome(Outcome::kDuplicate);
        return Result::kInFlight;
    }
    ::FarLib::allocator::begin_normal_read_recovery(target_block);
    if (!ec_recovery_drain_normal_reads_once(target_block)) {
        profile_post_attempt.set_outcome(Outcome::kInFlight);
        return Result::kInFlight;
    }
    const uint64_t remote_addr = entry->remote_addr();
    const auto still_fetching = [&] {
        return entry->load_state(std::memory_order_acquire).state == EntryState::FETCHING &&
               entry->local_addr() == local_addr &&
               entry->remote_addr() == remote_addr;
    };
    if (!still_fetching() ||
        !::FarLib::allocator::try_claim_degraded_read(target_block)) {
        profile_post_attempt.set_outcome(Outcome::kInFlight);
        return Result::kInFlight;
    }
    struct OwnerGuard {
        ::FarLib::allocator::BlockHead *block;
        bool active = true;
        ~OwnerGuard() { if (active) release_degraded_read_owner(block); }
    } owner{target_block};
    // Serialize against the prior winner by claiming the block, then checking
    // again. A waiter that observed an old FETCHING cannot rebuild over LOCAL.
    if (!still_fetching()) {
        profile_post_attempt.set_outcome(Outcome::kInFlight);
        return Result::kInFlight;
    }
    const uint32_t byte_count =
        static_cast<uint32_t>(entry->load_state(std::memory_order_relaxed).size);
    profile_post_attempt.set_byte_count(byte_count);
    EcRecoveryGroupView view;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kGroupView);
        if (!ec_recovery_group_view(entry, byte_count, &view)) {
            ec_recovery_post_unavailable_.fetch_add(1, std::memory_order_relaxed);
            profile_post_attempt.set_outcome(Outcome::kUnavailable);
            return Result::kUnavailable;
        }
    }
    auto *client = rdma::get_client(client_idx);
    if (client == nullptr || !ec_read_scratch_pool_.valid()) {
        profile_post_attempt.set_outcome(Outcome::kUnavailable);
        return Result::kUnavailable;
    }
    ec_batch::EcStagingGroupSlot scratch;
    bool acquired = false;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kScratchAcquire);
        acquired = ec_read_scratch_pool_.acquire(&scratch, client_idx);
    }
    ec_recovery_profile().note_scratch_acquire_result(acquired);
    if (!acquired) {
        ec_recovery_scratch_backpressure_.fetch_add(1, std::memory_order_relaxed);
        profile_post_attempt.set_outcome(Outcome::kUnavailable);
        return Result::kUnavailable;
    }
    const uint64_t target = reinterpret_cast<uintptr_t>(local_addr);
    uint64_t token_id = 0;
    ec_read_recovery::EcReadContext *context = nullptr;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kTokenAcquire);
        acquired = ec_read_tokens_.acquire(
            client_idx, target, remote_addr, byte_count, view.plan.read_mask,
            view.own_shard_idx, scratch, &token_id, &context);
    }
    ec_recovery_profile().note_token_acquire_result(acquired);
    if (!acquired) {
        (void)ec_read_scratch_pool_.release(scratch);
        ec_recovery_post_unavailable_.fetch_add(1, std::memory_order_relaxed);
        profile_post_attempt.set_outcome(Outcome::kUnavailable);
        return Result::kUnavailable;
    }
    // From here the posting-close / final-CQE winner owns both resources and
    // the block pin, even if a later post fails.
    owner.active = false;
    entry_pin.active = false;
    const size_t qp_idx = client->get_qp_idx();
    const uint64_t post_deadline_us = ec_recovery_post_deadline_us();
    bool all_posted = true;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kSegmentPostMark);
        for (size_t segment = 0; segment < ec_read_recovery::kEcReadSegmentCount;
             ++segment) {
            // The context's alive_mask is the fetched/selected set, not the
            // physical endpoint-survivor set.  Exactly four reads are posted;
            // a fifth physically alive shard is intentionally left untouched.
            if ((view.plan.read_mask & (1u << segment)) == 0) continue;
            const auto &seg = view.group.segments[segment];
            const auto slot_index = static_cast<uint8_t>(segment);
            void *dst = ec_read_recovery::ec_read_scratch_segment(scratch, slot_index);
            const uint64_t wr_id =
                ec_read_recovery::encode_ec_read_wr_id(token_id, slot_index);
            const auto mapped = config.map_remote_addr(seg.addr);
            ASSERT(mapped.first == seg.endpoint_idx);
            ASSERT(mapped.second + seg.slot_size <= config.server_buffer_size);
            uint64_t spin = 0;
            while (!client->post_read(mapped.second, dst, byte_count, wr_id, 0,
                                     qp_idx, seg.endpoint_idx, &scratch.lkey)) {
                (void)check_cq_idx_with_client_idx_endpoint(
                    qp_idx, client_idx, seg.endpoint_idx);
                if (++spin > kEcRecoveryScratchSpinLimit ||
                    ec_recovery_post_budget_out(spin, post_deadline_us) ||
                    ec_recovery_endpoint_is_dead(seg.endpoint_idx)) {
                    all_posted = false;
                    break;
                }
                ec_recovery_scratch_backpressure_.fetch_add(1, std::memory_order_relaxed);
                uthread::yield();
            }
            if (!all_posted) break;
            ec_recovery_profile().note_accepted_survivor_segment(byte_count);
            // This is a request-local atomic update, not a shared-table lookup.
            // CQEs may beat this mark; posting close prevents premature release.
            if (!ec_read_tokens_.mark_segment_posted(token_id, slot_index)) {
                ERROR("ec_recovery: accepted READ lost its live context");
            }
        }
    }
    if (!all_posted) {
        ec_recovery_post_unavailable_.fetch_add(1, std::memory_order_relaxed);
        profile_post_attempt.set_outcome(Outcome::kUnavailable);
    } else {
        ec_recovery_posts_.fetch_add(1, std::memory_order_relaxed);
        profile_post_attempt.set_outcome(Outcome::kPosted);
        ec_recovery_report_degraded_read(entry, view);
    }
    const auto finish = ec_read_tokens_.finish_posting(token_id);
    // Closing may release the object reference on this or another thread.
    // No object/context access (including diagnostics) after this boundary.
    finish_ec_read_context(token_id, finish);
    return all_posted ? Result::kPosted : Result::kUnavailable;
}

// The spec'd entry point: by object, with the local slot the caller expects the
// bytes in.  A mismatch (the entry moved on) leaves the read alone.
inline ec_read_recovery::EcReadPostResult
ConcurrentArrayCache::post_ec_degraded_read(far_obj_t obj, size_t client_idx,
                                            void *local_addr) {
    if (!::FarLib::get_config().is_ec_batch_mode()) {
        return ec_read_recovery::EcReadPostResult::kUnavailable;
    }
    auto &entry = get_entry_of(obj);
    return post_ec_degraded_read_entry(&entry, client_idx, local_addr);
}

// One tagged read segment completed.  The last of them rebuilds the object into
// its local slot and converges the entry, exactly like a completed
// single-sided read of the object's own segment.
inline void ConcurrentArrayCache::handle_ec_read_segment_complete(
    uint64_t wr_id) {
    const uint64_t token_id = ec_read_recovery::ec_read_wr_id_token(wr_id);
    const uint8_t segment = ec_read_recovery::ec_read_wr_id_segment(wr_id);
    ec_read_recovery::EcReadContextEvent event;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kSegmentCompletion);
        event = ec_read_tokens_.complete_segment_event(token_id, segment, true);
    }
    ec_recovery_profile().note_segment_completion(true);
    finish_ec_read_context(token_id, event);
}

inline void ConcurrentArrayCache::finish_ec_read_context(
    uint64_t token_id, const ec_read_recovery::EcReadContextEvent &event) {
    if (event.kind == ec_read_recovery::EcReadTokenEventKind::kRelease) {
        auto *block = static_cast<::FarLib::allocator::BlockHead *>(
            reinterpret_cast<void *>(event.token->target_local_addr)) - 1;
        auto *entry = &get_entry_of(block->obj_meta_data.load(std::memory_order_acquire));
        ec_recovery_profile().note_token_abandon(event.profile_acquire_ns);
        {
            EcRecoveryProfile::ScopedTimer timer(
                ec_recovery_profile(), EcRecoveryProfile::Stage::kScratchRelease);
            if (ec_read_scratch_pool_.valid() &&
                !ec_read_scratch_pool_.release(event.scratch)) {
                ec_recovery_scratch_release_failures_.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        ec_recovery_abandons_.fetch_add(1, std::memory_order_relaxed);
        if (!ec_read_tokens_.release(token_id))
            ERROR("ec_recovery: cannot release failed context");
        release_degraded_read_owner(block);
        unpin_fetch_entry_for_recovery(entry);
        return;
    }
    if (event.kind != ec_read_recovery::EcReadTokenEventKind::kWinner) return;
    ec_read_recovery::EcReadContext *token = event.token;
    ec_recovery_completions_.fetch_add(1, std::memory_order_relaxed);
    // Copy what is needed out of the token before it is returned: after
    // release() the slot can be borrowed by another read.
    const uint32_t byte_count = token->byte_count;
    const uint8_t alive_mask = token->alive_mask;
    const uint8_t own_shard = token->own_shard_idx;
    const uint64_t target = token->target_local_addr;
    const ec_batch::EcStagingGroupSlot scratch = token->scratch;
    ec_recovery_profile().note_winner(event.profile_acquire_ns, byte_count);
    void *dst = reinterpret_cast<void *>(static_cast<uintptr_t>(target));
    auto *target_block =
        static_cast<::FarLib::allocator::BlockHead *>(dst) - 1;
    const auto owner_obj = target_block->obj_meta_data.load(std::memory_order_acquire);
    ASSERT(!owner_obj.is_null());
    auto *owner_entry = &get_entry_of(owner_obj);
    if (owner_entry->load_state(std::memory_order_acquire).state != EntryState::FETCHING ||
        owner_entry->local_addr() != dst || owner_entry->remote_addr() != token->remote_addr) {
        (void)ec_read_scratch_pool_.release(scratch);
        if (!ec_read_tokens_.release(token_id))
            ERROR("ec_recovery: cannot release obsolete context");
        release_degraded_read_owner(target_block);
        unpin_fetch_entry_for_recovery(owner_entry);
        ec_recovery_abandons_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // The ordinary-READ gate was closed and all pre-existing ordinary READs
    // were drained before this survivor round was posted.  Do not poll here:
    // this function runs inside a CQ handler, and recursively polling the CQ
    // can strand a completion that is already in the caller's WC batch.
    ASSERT(target_block->pending_rdma_reads.load(std::memory_order_acquire) ==
           0);

    bool rebuilt;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kDecode);
    if ((alive_mask & (1u << own_shard)) == 0) {
        // The object's own data segment is the missing one (the segment the
        // dead endpoint served).  The survivor reads remain in scratch, but
        // the reconstructed bytes are now written directly to dst after the
        // ordinary-READ drain above.
        uint8_t survivor_idx[kStripeCodecDataShards] = {};
        const void *survivors[kStripeCodecDataShards] = {};
        size_t survivor_count = 0;
        for (size_t s = 0;
             s < ec_read_recovery::kEcReadSegmentCount &&
             survivor_count < kStripeCodecDataShards;
             ++s) {
            if ((alive_mask & (1u << s)) == 0) continue;
            survivor_idx[survivor_count] = static_cast<uint8_t>(s);
            survivors[survivor_count] =
                ec_read_recovery::ec_read_scratch_segment(
                    scratch, static_cast<uint8_t>(s));
            ++survivor_count;
        }
        // The selected read mask has exactly four bits.  Rebuild only the
        // requested data shard; do not treat an intentionally skipped fifth
        // physical survivor as a missing output that must also be repaired.
        rebuilt = survivor_count == kStripeCodecDataShards &&
                  small_object_stripe_rebuild_one(
                      own_shard, survivor_idx, survivors, dst, byte_count, 0);
    } else {
        // Every data shard of this object is alive (only a parity segment is
        // gone): publish the already-read own data after the drain.
        std::memcpy(dst,
                    ec_read_recovery::ec_read_scratch_segment(scratch,
                                                              own_shard),
                    byte_count);
        rebuilt = true;
    }
    }
    if (!rebuilt) {
        ec_recovery_rebuild_failures_.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "ERROR: ec_recovery rebuild_failed token=" << token_id
                  << " segment=" << static_cast<unsigned>(event.segment)
                  << " alive_mask=" << static_cast<unsigned>(alive_mask)
                  << " own_shard=" << static_cast<unsigned>(own_shard)
                  << " bytes=" << byte_count << std::endl;
    }
    if (rebuilt) {
        // Runtime self-check of the rebuilt shard, after the safe publication
        // point; observation only, the bytes below are handed over whether it
        // passes or fails.
        ec_recovery_verify_rebuild(*token);
    }
    // Return the scratch segment first; the token still pins the target, so no
    // second read of this object can start in between.
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kScratchRelease);
        if (!ec_read_scratch_pool_.release(scratch)) {
            ec_recovery_scratch_release_failures_.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    if (rebuilt &&
        owner_entry->load_state(std::memory_order_acquire).state == EntryState::FETCHING &&
        owner_entry->local_addr() == dst &&
        owner_entry->remote_addr() == token->remote_addr) {
        ec_recovery_rebuilds_.fetch_add(1, std::memory_order_relaxed);
        // Converge the entry exactly like a read completion: FETCHING -> LOCAL
        // plus, under exclusive_cache, the backup bookkeeping and the
        // group-aware release of the dead segment's remote slot.
        ibv_wc synthetic{};
        synthetic.status = IBV_WC_SUCCESS;
        synthetic.opcode = IBV_WC_RDMA_READ;
        synthetic.wr_id = target;
        synthetic.byte_len = byte_count;
        // The original ordinary READ, if any, was drained before the survivor
        // round was posted.  This synthetic completion only converges the
        // entry after reconstruction.
        {
            EcRecoveryProfile::ScopedTimer timer(
                ec_recovery_profile(), EcRecoveryProfile::Stage::kPublicationRemote);
            handle_rdma_read_complete(synthetic, false);
        }
        ec_recovery_profile().note_recovered_object(byte_count);
    }
    if (!ec_read_tokens_.release(token_id))
        ERROR("ec_recovery: cannot release completed context");
    release_degraded_read_owner(target_block);
    unpin_fetch_entry_for_recovery(owner_entry);
}

// Help all data CQs once, then defer recovery if any ordinary READ still owns
// the block. Never clear pins on a timeout. Returning to the fetch wait keeps
// other fibres and its diagnostics deadline live; completion callbacks never
// wait for a CQE that may already be in their caller's WC batch.
inline bool ConcurrentArrayCache::ec_recovery_drain_normal_reads_once(
    ::FarLib::allocator::BlockHead *block) {
    EcRecoveryProfile::ScopedTimer timer(
        ec_recovery_profile(), EcRecoveryProfile::Stage::kOrdinaryReadDrain);
    if (block->pending_rdma_reads.load(std::memory_order_acquire) == 0)
        return true;
    (void)ec_batch_poll_all_cqs_once();
    return block->pending_rdma_reads.load(std::memory_order_acquire) == 0;
}

// ---------------------------------------------------------------------------
// Runtime correctness self-check of one rebuilt shard.
//
// The degraded read covers exactly the shape this check can use: the object's
// own data segment is gone, so the three surviving data shards plus the two
// parity shards are read back and the missing data shard is reconstructed.
// Re-encoding the four data shards (the rebuilt one plus the three read back)
// yields two parity shards that must reproduce the two parity shards just read
// off the disk, byte for byte; a wrongly rebuilt shard cannot, because that
// would need the GF(2^8) parity of `byte_count` bytes to collide.  Cheap and
// sufficient, and it needs no state beyond what the read already has.
//
// Observation only: it bumps two counters and prints one line, and it never
// changes what the access path receives - a FAIL is reported, not acted on,
// the rebuilt bytes are handed over exactly as before, and the entry
// convergence below is untouched.
// ---------------------------------------------------------------------------
inline void ConcurrentArrayCache::ec_recovery_verify_rebuild(
    const ec_read_recovery::EcReadContext &token) {
    EcRecoveryProfile::ScopedTimer verify_timer(
        ec_recovery_profile(), EcRecoveryProfile::Stage::kVerify);
    auto verify_skip = [&] { ec_recovery_profile().note_verify_skip(); };
    if (!::FarLib::get_config().is_ec_batch_mode()) {
        verify_skip();
        return;
    }
    if (!ec_recovery_verify_enabled()) {
        // The benchmark's independent byte oracle remains mandatory; this only
        // disables the runtime re-encode observer.
        verify_skip();
        return;
    }
    if (token.own_shard_idx >= kStripeCodecDataShards) {
        verify_skip();  // nothing rebuilt
        return;
    }
    // The runtime recovery round fetches exactly four survivors.  Re-encoding
    // requires both parity references, including any fifth physical survivor
    // that was intentionally skipped, so this observer cannot validate this
    // path.  The independent byte oracle covers the four-read codec contract.
    if (__builtin_popcount(static_cast<unsigned>(token.alive_mask)) ==
        kStripeCodecDataShards) {
        verify_skip();
        return;
    }
    const uint8_t all_segments = static_cast<uint8_t>(
        (1u << ec_read_recovery::kEcReadSegmentCount) - 1u);
    // Every segment but the rebuilt one must have come back from the disk,
    // both parity shards included: those two are the reference to compare
    // against.  Any other missing set is a partial group and is left alone.
    if ((token.alive_mask | static_cast<uint8_t>(1u << token.own_shard_idx)) !=
        all_segments) {
        verify_skip();
        return;
    }
    const uint32_t byte_count = token.byte_count;
    if (byte_count == 0 || byte_count > ec_batch::kEcBatchStagingMaxSlotSize) {
        verify_skip();
        return;  // does not fit the resident two-segment re-encode scratch
    }
    const auto &scratch = token.scratch;
    const void *data[kStripeCodecDataShards];
    for (size_t j = 0; j < kStripeCodecDataShards; j++) {
        data[j] = (j == token.own_shard_idx)
                      ? reinterpret_cast<const void *>(
                            static_cast<uintptr_t>(token.target_local_addr))
                      : ec_read_recovery::ec_read_scratch_segment(
                            scratch, static_cast<uint8_t>(j));
    }
    // The freshly encoded parity goes into its own two-segment scratch: the
    // read scratch still holds the parity read back off the disk.
    alignas(64) uint8_t reencoded[kStripeCodecParityShards]
                                 [ec_batch::kEcBatchStagingMaxSlotSize];
    void *parity[kStripeCodecParityShards];
    for (size_t p = 0; p < kStripeCodecParityShards; p++) {
        parity[p] = reencoded[p];
    }
    // The one encode call of this check: G1's generator-matrix encode, i.e.
    // parity[p] = sum_j coef[p][j] * data[j], no GF arithmetic of our own.
    if (!small_object_stripe_encode_shards(data, parity, byte_count, 0)) {
        verify_skip();
        return;  // codec refused the shape: not this check's business
    }
    bool pass = true;
    for (size_t p = 0; p < kStripeCodecParityShards; p++) {
        const void *on_disk = ec_read_recovery::ec_read_scratch_segment(
            scratch, static_cast<uint8_t>(kStripeCodecDataShards + p));
        if (std::memcmp(reencoded[p], on_disk, byte_count) != 0) pass = false;
    }
    if (pass) {
        ec_recovery_verify_pass_.fetch_add(1, std::memory_order_relaxed);
    } else {
        ec_recovery_verify_fail_.fetch_add(1, std::memory_order_relaxed);
    }
    ec_recovery_profile().note_verify_result(pass);
    // Throttled like the other per-event lines (the first 20 one by one, then
    // one every 100) - except a FAIL, which is the signal this check exists for
    // and is printed for every single occurrence.
    if (pass && !ec_recovery_throttled(ec_recovery_verify_events_, 20, 100)) {
        return;
    }
    if (pass && ec_recovery_diag_quiet()) {
        ec_recovery_profile().note_log_suppressed();
        return;
    }
    // group=(stripe,slot) of the rebuilt segment, reverse-mapped from the read
    // token (the token keeps the segment address, not the group id).
    uint64_t stripe_id = 0;
    uint64_t slot_id = 0;
    SmallObjectStripeManager::SlotLayout layout;
    if (remote_allocator.small_object_stripe_manager().get_slot_layout(
            token.remote_addr, &layout)) {
        stripe_id = layout.stripe_id;
        slot_id = layout.slot_id;
    }
    std::string line;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogFormat);
        std::ostringstream formatted;
        formatted << "INFO: ec_recovery verify group=(" << stripe_id << ","
                  << slot_id
                  << ") missing_shard="
                  << static_cast<unsigned>(token.own_shard_idx)
                  << " live_mask=0x" << std::hex
                  << static_cast<unsigned>(token.alive_mask) << std::dec
                  << " result=" << (pass ? "PASS" : "FAIL");
        line = formatted.str();
    }
    // One write call: several threads must not interleave inside one line.
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogOutput);
        std::cout << line << std::endl;
    }
}

// The in-flight case: while a fetch waits for an entry whose own data segment
// sits on an endpoint that is gone, the originally posted read can never
// complete.  Re-post the entry once as a degraded read and reap the CQEs of
// every endpoint of this client (the five reads of the group do not arrive on
// the dead endpoint).  Returns true when the caller should skip its own poll of
// the dead endpoint for this round; false leaves the legacy wait unchanged.
inline bool ConcurrentArrayCache::ec_recovery_assist_wait(
    FarObjectEntry *entry, size_t qp_idx, size_t client_idx) {
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode()) return false;
    if (entry == nullptr) return false;
    if (entry->is_local()) return false;
    if (entry->load_state(std::memory_order::relaxed).state !=
        EntryState::FETCHING) {
        return false;
    }
    if (config.server_count <= 1) return false;
    const uint64_t remote_addr = entry->remote_addr();
    if (remote_addr == FarObjectEntry::RemoteAddrInvalid48) return false;
    const size_t endpoint_idx = config.map_remote_addr(remote_addr).first;
    if (!ec_recovery_endpoint_is_dead(endpoint_idx)) return false;
    const auto result = post_ec_degraded_read_entry(entry, client_idx);
    if (result == ec_read_recovery::EcReadPostResult::kUnavailable) {
        return false;  // keep the legacy poll of the dead endpoint
    }
    ec_recovery_wait_assists_.fetch_add(1, std::memory_order_relaxed);
    // post_ec_degraded_read_entry() above returns within its posting budget
    // (kEcRecoveryPostAttemptLimit rounds / kEcRecoveryPostBudgetUs); the
    // kUnavailable case is a normal outcome and just leaves this wait round to
    // the legacy poll.  The drain below is one pass over this client's
    // endpoints and reaps data CQEs only (never the sponge control CQ).
    const size_t endpoints = ec_recovery_endpoint_count();
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(),
            EcRecoveryProfile::Stage::kAssistPollAllEndpoints);
        for (size_t endpoint = 0; endpoint < endpoints; endpoint++) {
            (void)check_cq_idx_with_client_idx_endpoint(qp_idx, client_idx,
                                                        endpoint);
        }
    }
    ec_recovery_profile().note_assist_poll_endpoints(endpoints);
    return true;
}

// Read-only diagnostics of the degraded read path; only called from the
// ec_batch teardown trace, so it prints nothing unless ec_batch runs.
inline void ConcurrentArrayCache::ec_read_recovery_diag_report(
    const char *where) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    auto ld = [](const std::atomic<uint64_t> &v) {
        return v.load(std::memory_order_relaxed);
    };
    size_t dead = 0;
    for (size_t endpoint = 0; endpoint < ec_endpoint_dead_count_; endpoint++) {
        if (ec_endpoint_dead_[endpoint].load(std::memory_order_relaxed)) {
            dead++;
        }
    }
    std::string line;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogFormat);
        std::ostringstream formatted;
        formatted << "ec_read_recovery diag [" << where
                  << "]: dead_endpoints=" << dead << "/"
                  << ec_endpoint_dead_count_
                  << " error_wcs=" << ld(ec_recovery_error_wcs_)
                  << " routed_reads=" << ld(ec_recovery_routed_reads_)
                  << " posts=" << ld(ec_recovery_posts_)
                  << " duplicates=" << ld(ec_recovery_post_duplicates_)
                  << " unavailable=" << ld(ec_recovery_post_unavailable_)
                  << " wait_assists=" << ld(ec_recovery_wait_assists_)
                  << " completions=" << ld(ec_recovery_completions_)
                  << " rebuild_failures=" << ld(ec_recovery_rebuild_failures_)
                  << " verify_pass=" << ld(ec_recovery_verify_pass_)
                  << " verify_fail=" << ld(ec_recovery_verify_fail_)
                  << " abandons=" << ld(ec_recovery_abandons_)
                  << " scratch_backpressure="
                  << ld(ec_recovery_scratch_backpressure_)
                  << " scratch_in_use=" << ec_read_scratch_pool_.in_use()
                  << " scratch_depth=" << ec_read_scratch_pool_.depth()
                  << " tokens_in_use=" << ec_read_tokens_.in_use()
                  << " token_capacity=" << ec_read_tokens_.capacity()
                  << " token_peak_in_use=" << ec_read_tokens_.peak_in_use()
                  << " scratch_peak_in_use="
                  << ec_read_scratch_pool_.peak_in_use()
                  << " scratch_bytes=" << ec_read_scratch_pool_.bytes()
                  << " scratch_growths=" << ec_read_scratch_pool_.growths()
                  << " scratch_allocation_failures="
                  << ec_read_scratch_pool_.allocation_failures()
              << " recovery_limit=" << ec_read_scratch_pool_.limit();
        line = formatted.str();
    }
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogOutput);
        std::cout << line << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Observability of the degraded read path, moved in front of the teardown.
//
// The counters of this file used to be readable only from the cache_dtor_begin
// report, which a hung teardown never reaches, so "did the degraded read ever
// run" was structurally unanswerable.  The lines below are printed from the
// events themselves: every real post of a degraded read, every refused group
// allocation of the write side while an endpoint is dead, and one counter
// summary every 30 s.  All of them are throttled, flushed line by line and
// gated on ft_method=ec_batch; none of them changes an offset, an order, a wait
// or a return value.
// ---------------------------------------------------------------------------

// Throttle of the per-event lines: event number n of `counter` is printed when
// n <= first (one line each) or when n % every == 0 (heartbeat); the counter
// counts every event either way.
inline bool ConcurrentArrayCache::ec_recovery_throttled(
    std::atomic<uint64_t> &counter, uint64_t first, uint64_t every) {
    const uint64_t n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= first) return true;
    if (every == 0) return false;
    return (n % every) == 0;
}

// Number of endpoints currently marked dead (the same quantity
// ec_read_recovery_diag_report() prints as dead_endpoints).
inline size_t ConcurrentArrayCache::ec_recovery_dead_endpoint_count() const {
    size_t dead = 0;
    for (size_t endpoint = 0; endpoint < ec_endpoint_dead_count_; endpoint++) {
        if (ec_endpoint_dead_[endpoint].load(std::memory_order_relaxed)) dead++;
    }
    return dead;
}

// One counter line every 30 s, checked at the two event points below (so it
// costs nothing while nothing happens).  routed/posted/completed/wait_assists
// are the existing degraded-read counters, rebuilt the successful rebuilds,
// group_alloc_blocked the refusals counted by
// ec_recovery_report_group_alloc_blocked().
inline void ConcurrentArrayCache::ec_recovery_report_counters() {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    constexpr uint64_t interval_ms = 30000;
    const uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    uint64_t last = ec_recovery_last_summary_ms_.load(std::memory_order_relaxed);
    if (last != 0 && now_ms < last + interval_ms) return;
    if (!ec_recovery_last_summary_ms_.compare_exchange_strong(
            last, now_ms, std::memory_order_relaxed)) {
        return;  // another thread already printed this interval
    }
    if (ec_recovery_diag_quiet()) {
        ec_recovery_profile().note_log_suppressed();
        return;
    }
    auto ld = [](const std::atomic<uint64_t> &v) {
        return v.load(std::memory_order_relaxed);
    };
    std::string line;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogFormat);
        std::ostringstream formatted;
        formatted << "INFO: ec_recovery counters routed="
                  << ld(ec_recovery_routed_reads_)
                  << " posted=" << ld(ec_recovery_posts_)
                  << " completed=" << ld(ec_recovery_completions_)
                  << " rebuilt=" << ld(ec_recovery_rebuilds_)
                  << " verify_pass=" << ld(ec_recovery_verify_pass_)
                  << " verify_fail=" << ld(ec_recovery_verify_fail_)
                  << " wait_assists=" << ld(ec_recovery_wait_assists_)
                  << " group_alloc_blocked="
                  << ld(ec_recovery_group_alloc_blocked_)
                  << " endpoint_dead_count="
                  << ec_recovery_dead_endpoint_count();
        line = formatted.str();
    }
    // One write call: several threads must not interleave inside one line.
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogOutput);
        std::cout << line << std::endl;
    }
}

// One line for a degraded read that really went out (all surviving segments of
// its group posted to the scratch).  obj is the object's local address
// (entry->local_addr()), entry its FarObjectEntry, group=(stripe,slot) the slot
// group served, missing_shard the segment missing because its endpoint is gone
// (endpoint_dead is that endpoint's index).  Throttled: the first 20 posts one
// by one, then one line every 100 posts.
inline void ConcurrentArrayCache::ec_recovery_report_degraded_read(
    FarObjectEntry *entry, const EcRecoveryGroupView &view) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    ec_recovery_report_counters();
    if (!ec_recovery_throttled(ec_recovery_degraded_read_events_, 20, 100)) {
        return;
    }
    if (ec_recovery_diag_quiet()) {
        ec_recovery_profile().note_log_suppressed();
        return;
    }
    const size_t missing_shard = view.own_shard_idx;
    const auto &dead_segment = view.group.segments[missing_shard];
    std::string line;
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogFormat);
        std::ostringstream formatted;
        formatted << "INFO: ec_recovery degraded_read obj=0x" << std::hex
                  << reinterpret_cast<uintptr_t>(entry->local_addr())
                  << " entry=0x" << reinterpret_cast<uintptr_t>(entry)
                  << std::dec << " group=(" << view.group.id.stripe_id << ","
                  << view.group.id.slot_id
                  << ") missing_shard="
                  << static_cast<unsigned>(missing_shard)
                  << " live_mask=0x" << std::hex
                  << static_cast<unsigned>(view.alive_mask) << std::dec
                  << " endpoint_dead=" << dead_segment.endpoint_idx;
        line = formatted.str();
    }
    {
        EcRecoveryProfile::ScopedTimer timer(
            ec_recovery_profile(), EcRecoveryProfile::Stage::kLogOutput);
        std::cout << line << std::endl;
    }
}

// The write side could not open a new group.  Counted and printed only when
// endpoints are actually missing and fewer than the six segments of a group are
// still reachable, which is exactly the "one endpoint down, no new group any
// more" hypothesis: with six live endpoints the refusal has another cause and
// this line stays silent.  Throttled: the first refusal, then every 1000th.
inline void ConcurrentArrayCache::ec_recovery_report_group_alloc_blocked(
    const char *reason) {
    if (!::FarLib::get_config().is_ec_batch_mode()) return;
    ec_recovery_report_counters();
    const size_t endpoints = ec_recovery_endpoint_count();
    size_t dead = 0;
    for (size_t endpoint = 0; endpoint < endpoints; endpoint++) {
        if (ec_recovery_endpoint_is_dead(endpoint)) dead++;
    }
    if (dead == 0) return;  // not an endpoint loss: not this diagnostic
    const size_t live = endpoints - dead;
    if (live >= ec_read_recovery::kEcReadSegmentCount) return;  // 6 still fit
    if (!ec_recovery_throttled(ec_recovery_group_alloc_blocked_, 1, 1000)) {
        return;
    }
    std::ostringstream line;
    line << "INFO: ec_recovery group_alloc_blocked live_endpoints=" << live
         << " dead=[";
    bool first_dead = true;
    for (size_t endpoint = 0; endpoint < endpoints; endpoint++) {
        if (!ec_recovery_endpoint_is_dead(endpoint)) continue;
        if (!first_dead) line << ",";
        first_dead = false;
        line << endpoint;
    }
    line << "] reason=" << (reason != nullptr ? reason : "unknown");
    // One write call: several threads must not interleave inside one line.
    std::cout << line.str() << std::endl;
}

// Sizes the endpoint liveness bitmap (config.server_count, not the six segments
// of a group) and carves the resident scratch range of the degraded reads out
// of the tail of the registered client MR.  Like the write-side staging range,
// the scratch is *not* registered as object memory, so the allocator can never
// hand those bytes out; the mr lkey is the client MR's lkey.  Returns the heap
// size the object allocator may use; unchanged (nothing reserved) unless
// ft_method=ec_batch.
inline size_t ConcurrentArrayCache::init_ec_read_recovery(void *local_buf,
                                                          size_t local_buf_size) {
    ec_endpoint_dead_.reset();
    ec_endpoint_dead_count_ = 0;
    const auto &config = ::FarLib::get_config();
    if (!config.is_ec_batch_mode()) return local_buf_size;
    const size_t endpoints = static_cast<size_t>(config.server_count);
    if (endpoints > 0) {
        ec_endpoint_dead_.reset(new std::atomic<bool>[endpoints]);
        for (size_t endpoint = 0; endpoint < endpoints; endpoint++) {
            ec_endpoint_dead_[endpoint].store(false, std::memory_order_relaxed);
        }
        ec_endpoint_dead_count_ = endpoints;
    }
    if (local_buf == nullptr || local_buf_size == 0) return local_buf_size;
    if (!ec_staging_pool_.valid()) {
        std::cerr << "ec_read_recovery scratch: no staging range to mirror; "
                     "degraded reads disabled"
                  << std::endl;
        return local_buf_size;
    }
    const size_t slot_size = ec_staging_pool_.slot_size();
    auto *control = rdma::ClientControl::get_default();
    if (control == nullptr) return local_buf_size;
    // Zero/default removes the arbitrary recovery-depth ceiling. A positive
    // value is an explicit experiment/resource budget, useful for same-binary
    // comparisons against the former 8-slot behavior.
    size_t limit = 0;
    if (const char *value = std::getenv("FARLIB_EC_RECOVERY_MAX_IN_FLIGHT")) {
        char *end = nullptr;
        errno = 0;
        const auto parsed = std::strtoull(value, &end, 10);
        if (value[0] == '-' || end == value || *end != '\0' || errno != 0 ||
            parsed > std::numeric_limits<size_t>::max())
            ERROR("invalid FARLIB_EC_RECOVERY_MAX_IN_FLIGHT");
        limit = static_cast<size_t>(parsed);
    }
    const auto allocate = [](void *context, size_t bytes,
                             ec_read_recovery::RecoveryScratchChunk *out) {
        void *buffer = nullptr;
        if (posix_memalign(&buffer, 4096, bytes) != 0) return false;
        auto *mr = ibv_reg_mr(static_cast<ibv_pd *>(context), buffer, bytes,
                              IBV_ACCESS_LOCAL_WRITE);
        if (mr == nullptr) {
            std::free(buffer);
            return false;
        }
        *out = {buffer, mr->lkey, mr};
        return true;
    };
    const auto release = [](void *, ec_read_recovery::RecoveryScratchChunk chunk) {
        if (ibv_dereg_mr(static_cast<ibv_mr *>(chunk.registration)) != 0)
            ERROR("cannot deregister recovery scratch MR");
        std::free(chunk.base);
    };
    if (!ec_read_scratch_pool_.init(slot_size, control->get_protection_domain(),
                                    allocate, release, limit))
        ERROR("cannot initialize grow-on-demand recovery scratch");
    // Evacuation and recovery borrow the same registered temporary chunks.
    // Ownership follows the issuing worker; no recovery-only global pool.
    if (!ec_staging_pool_.bind_shared_buffers(&ec_read_scratch_pool_, [] {
        return static_cast<size_t>(rdma::thread_info.thread_id);
    })) ERROR("cannot bind shared evacuation/recovery temporary buffers");
    std::cout << "ec_read_recovery scratch: mode=shared_worker_temp_buffers"
              << " slot_size=" << slot_size << " depth=0 limit=" << limit
              << " heap_bytes=" << local_buf_size
              << " endpoints=" << ec_endpoint_dead_count_
              << " legacy_reserved_bytes=" << ec_staging_reserved_bytes_
              << " context=direct_per_request" << std::endl;
    return local_buf_size;
}

}  // namespace FarLib::cache
