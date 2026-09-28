#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/ec_rmw_client.hpp"

namespace FarLib::cache::ec_rmw_runtime {

// A transaction owns the SlotReuse reservation (when one has been acquired),
// the borrowed local source, and the three requests in one RMW codeword.  The
// coordinator is the only writer of the masks below; CQ pollers only publish
// the transport request completion state.
struct Transaction {
    SmallObjectStripeManager::SlotReuse reservation{};
    void *source = nullptr;
    void *block = nullptr;
    uint64_t original_remote = 0;
    uint32_t size = 0;
    uint32_t behavior_group = 0;

    // `data_shard == 0xff` means that the transaction is still deferred and
    // owns the original remote slot.  A nonzero generation marks a live
    // reservation, including a reservation whose target has not yet been
    // published because the entry is moving.
    uint8_t data_shard = 0xff;
    uint8_t read_armed = 0;
    uint8_t write_armed = 0;
    uint8_t read_posted = 0;
    uint8_t read_terminal = 0;
    uint8_t read_error = 0;
    uint8_t write_posted = 0;
    uint8_t write_terminal = 0;
    uint8_t write_success = 0;
    uint8_t write_error = 0;
    // A queued bit means that the coordinator has added this participant to
    // the batch-private pending list.  It is separate from `*_armed`: a
    // queued request is armed only inside the submit parent, preserving the
    // existing timing boundary and avoiding write preparation while reads are
    // still outstanding.
    uint8_t read_queued = 0;
    uint8_t write_queued = 0;
    bool target_published = false;
    bool writes_decided = false;
    bool old_remote_released = false;
    bool repair_eligible = false;
    bool encoded = false;

    bool has_reservation() const noexcept {
        return data_shard < 4 &&
               reservation.generation != 0;
    }

    void clear() noexcept { *this = Transaction{}; }
};

enum class BatchState : uint8_t { Free = 0, Fill = 1, Reading = 2, Writing = 3 };

// Fixed-capacity, batch-private submission lists.  Entries are linked within
// endpoint groups so a partial ibv_post_send acceptance advances only the
// accepted prefix and leaves the untouched suffix for a later retry.  Endpoint
// keys are stored as size_t rather than using a server-count-sized array: the
// configured endpoint count remains unrestricted by this optimization.
struct PendingEntry {
    rdma::ec_rmw::Exchange *exchange = nullptr;
    uint16_t identity = 0;
    uint16_t next = 0xffff;
};

struct PendingList {
    static constexpr size_t kCapacity =
        rdma::ec_rmw::kBatchObjects * rdma::ec_rmw::kParticipants;
    static constexpr uint16_t kInvalid = 0xffff;

    std::array<PendingEntry, kCapacity> entries{};
    std::array<size_t, kCapacity> group_endpoint{};
    std::array<uint16_t, kCapacity> group_head{};
    std::array<uint16_t, kCapacity> group_tail{};
    uint16_t entry_count = 0;
    uint16_t group_count = 0;

    void reset() noexcept {
        entry_count = 0;
        group_count = 0;
    }

    bool empty() const noexcept {
        for (size_t g = 0; g < group_count; ++g)
            if (group_head[g] != kInvalid) return false;
        return true;
    }

    uint16_t find_group(size_t endpoint) {
        for (uint16_t g = 0; g < group_count; ++g)
            if (group_endpoint[g] == endpoint) return g;
        if (group_count == kCapacity) ERROR("ec_rmw: pending endpoint groups overflow");
        const uint16_t g = group_count++;
        group_endpoint[g] = endpoint;
        group_head[g] = kInvalid;
        group_tail[g] = kInvalid;
        return g;
    }

    void append(size_t endpoint, rdma::ec_rmw::Exchange *exchange,
                uint16_t identity) {
        if (entry_count == kCapacity) ERROR("ec_rmw: pending list overflow");
        const uint16_t group = find_group(endpoint);
        const uint16_t index = entry_count++;
        entries[index] = PendingEntry{exchange, identity, kInvalid};
        if (group_head[group] == kInvalid) {
            group_head[group] = index;
        } else {
            entries[group_tail[group]].next = index;
        }
        group_tail[group] = index;
    }
};

struct Batch {
    BatchState state = BatchState::Free;
    std::array<Transaction, rdma::ec_rmw::kBatchObjects> transactions{};
    // READ and WRITE lists must remain independent: a batch may contain a
    // transaction whose READs are still pending beside another transaction
    // that was already encoded and is retrying a partial WRITE prefix.
    PendingList read_pending{};
    PendingList write_pending{};
    size_t count = 0;
    size_t submit_client = static_cast<size_t>(-1);
    size_t submit_qp = static_cast<size_t>(-1);

    void reset() noexcept {
        for (auto &transaction : transactions) transaction.clear();
        read_pending.reset();
        write_pending.reset();
        count = 0;
        submit_client = static_cast<size_t>(-1);
        submit_qp = static_cast<size_t>(-1);
        state = BatchState::Free;
    }
};

// This object is allocated once per logical evacuation owner.  The bank of
// RDMA exchanges is owned by rdma::ec_rmw::ClientTransport; these records hold
// only coordinator state and therefore never allocate on the update path.
struct Worker {
    std::array<Batch, rdma::ec_rmw::kBatchDepth> batches{};
    // ISA-L delta outputs are reused one transaction at a time.  They are
    // worker-private and never enter the registered RDMA exchange bank.
    alignas(64) std::array<uint8_t, rdma::ec_rmw::kMaxBytes> parity_delta0{};
    alignas(64) std::array<uint8_t, rdma::ec_rmw::kMaxBytes> parity_delta1{};
    size_t fill_batch = 0;

    uint64_t attempted = 0;
    uint64_t committed = 0;
    uint64_t payload_bytes = 0;
    uint64_t aborted = 0;
    uint64_t fallback = 0;
    uint64_t growth_objects = 0;
    uint64_t growth_payload_bytes = 0;
    uint64_t replacement_objects = 0;
    uint64_t capacity_replacement_objects = 0;
    uint64_t deferred = 0;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    uint64_t read_posts = 0;
    uint64_t write_posts = 0;
    uint64_t batches_full = 0;
    uint64_t batches_tail = 0;
    uint64_t polls = 0;
    uint64_t stages = 0;
    uint64_t inflight_highwater = 0;

    size_t pending_count() const noexcept {
        size_t total = 0;
        for (const auto &batch : batches) total += batch.count;
        return total;
    }

    size_t inflight_batches() const noexcept {
        size_t total = 0;
        for (const auto &batch : batches)
            if (batch.state != BatchState::Free) ++total;
        return total;
    }

    Batch *find_fill_batch() noexcept {
        for (size_t n = 0; n < batches.size(); ++n) {
            const size_t index = (fill_batch + n) % batches.size();
            if (batches[index].state == BatchState::Fill &&
                batches[index].count < batches[index].transactions.size()) {
                fill_batch = index;
                return &batches[index];
            }
        }
        for (size_t n = 0; n < batches.size(); ++n) {
            const size_t index = (fill_batch + n) % batches.size();
            if (batches[index].state == BatchState::Free) {
                fill_batch = index;
                batches[index].state = BatchState::Fill;
                return &batches[index];
            }
        }
        return nullptr;
    }
};

}  // namespace FarLib::cache::ec_rmw_runtime
