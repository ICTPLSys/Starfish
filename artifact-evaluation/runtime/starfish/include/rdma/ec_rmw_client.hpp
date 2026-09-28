#pragma once

#include <infiniband/verbs.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "utils/debug.hpp"
#include "utils/cpu_cycles.hpp"

namespace FarLib::rdma::ec_rmw {

// One bounded incremental-EC round consists of four batches, with one entry
// for every object and three participants (old data/diff, parity 0, parity 1).
// Keeping these values here makes the client and coordinator share one layout
// without constructing a run-time work queue or a global staging pool.
inline constexpr std::size_t kBatchObjects = 64;
inline constexpr std::size_t kBatchDepth = 4;
inline constexpr std::size_t kMaxBytes = 4096;
inline constexpr std::size_t kParticipants = 3;

// Existing completion paths reserve the high-half namespaces as follows:
//   10xx = EC writes, 11xx = degraded EC reads, 010x = ordinary reads,
//   001x = ordinary writes.  RMW uses the disjoint 0001 nibble.  The low
// 60 bits carry an exchange index and a generation, so a late CQE cannot
// complete a newly armed occupant of the same bounded bank entry.
inline constexpr std::uint64_t kWrIdTag = std::uint64_t{1} << 60;
inline constexpr std::uint64_t kWrIdTopMask = std::uint64_t{0xf} << 60;
inline constexpr unsigned kWrIdSlotBits = 32;
inline constexpr std::uint64_t kWrIdSlotMask =
    (std::uint64_t{1} << kWrIdSlotBits) - 1;
inline constexpr unsigned kWrIdGenerationBits = 60 - kWrIdSlotBits;
inline constexpr std::uint32_t kWrIdGenerationMask =
    (std::uint32_t{1} << kWrIdGenerationBits) - 1;

static_assert((kWrIdTag & kWrIdTopMask) == kWrIdTag,
              "RMW wr_id tag must occupy exactly one top-nibble value");
static_assert((kWrIdTag & (std::uint64_t{1} << 61)) == 0,
              "RMW wr_id must remain disjoint from ordinary WRITE IDs");
static_assert((kWrIdTag & (std::uint64_t{1} << 62)) == 0,
              "RMW wr_id must remain disjoint from ordinary READ IDs");
static_assert((kWrIdTag & (std::uint64_t{1} << 63)) == 0,
              "RMW wr_id must remain disjoint from EC IDs");

enum class CompletionState : std::uint8_t {
    pending = 0,
    success = 1,
    error = 2,
};

// Per-exchange request state.  `posted` is deliberately non-atomic: the
// coordinator is the sole producer and owner of posting/reuse decisions.
// Completion pollers only publish `completion_word` and never touch `posted`
// or wait for a state transition.
struct Request {
    using State = CompletionState;

    // The generation and state share one atomic word.  A CQ poller can then
    // CAS (old-generation,pending) to (old-generation,result) without a
    // generation-load/state-CAS TOCTOU window during coordinator re-arm.
    std::atomic<std::uint64_t> completion_word{0x1};
    bool posted = false;
    std::atomic<std::uint32_t> generation{0};

    // The verbs objects are owned by this bounded entry.  No post path needs
    // to allocate or borrow a work-request object from a global pool.
    ibv_send_wr wr{};
    ibv_sge sges[2]{};
    std::uint32_t sge_count = 0;
    std::uint32_t bytes = 0;
    std::uint32_t source_bytes = 0;
    std::uint64_t wr_id = 0;

    static constexpr std::uint64_t pack_completion_word(
        std::uint32_t generation, CompletionState state) noexcept {
        return (static_cast<std::uint64_t>(generation) << 2) |
               static_cast<std::uint64_t>(state);
    }

    CompletionState load_state(
        std::memory_order order = std::memory_order_acquire) const noexcept {
        return static_cast<CompletionState>(
            completion_word.load(order) & std::uint64_t{3});
    }

    std::uint32_t load_completion_generation(
        std::memory_order order = std::memory_order_acquire) const noexcept {
        return static_cast<std::uint32_t>(completion_word.load(order) >> 2);
    }

    Request() noexcept = default;
    Request(const Request &) = delete;
    Request &operator=(const Request &) = delete;
};

struct alignas(64) Exchange {
    // The payload is the only portion used as an RDMA destination/source.  It
    // remains fixed-size so every entry can be registered in one contiguous
    // bank before worker execution starts.
    std::uint8_t payload[kMaxBytes]{};
    Request request{};

    Exchange() = default;
    Exchange(const Exchange &) = delete;
    Exchange &operator=(const Exchange &) = delete;
};

class ClientTransport {
private:
    std::size_t owners_ = 0;
    std::size_t exchange_count_ = 0;
    std::size_t exchange_mr_bytes_ = 0;
    std::size_t payload_bytes_ = 0;

    std::unique_ptr<Exchange[]> exchanges_;
    std::unique_ptr<std::uint8_t[]> zero_block_;
    ibv_pd *pd_ = nullptr;
    ibv_mr *exchange_mr_ = nullptr;
    ibv_mr *zero_mr_ = nullptr;

    static std::uint64_t encode_wr_id(std::size_t index,
                                      std::uint32_t generation) noexcept {
        return kWrIdTag |
               (static_cast<std::uint64_t>(generation)
                << kWrIdSlotBits) |
               static_cast<std::uint64_t>(index);
    }

    static bool decode_wr_id(std::uint64_t wr_id, std::size_t *index,
                             std::uint32_t *generation) noexcept {
        if (!is_wr_id(wr_id)) return false;
        const std::uint64_t raw_generation =
            (wr_id >> kWrIdSlotBits) & kWrIdGenerationMask;
        if (raw_generation == 0) return false;
        if (index != nullptr) {
            *index = static_cast<std::size_t>(wr_id & kWrIdSlotMask);
        }
        if (generation != nullptr) {
            *generation = static_cast<std::uint32_t>(raw_generation);
        }
        return true;
    }

    static std::size_t checked_exchange_count(std::size_t owners) {
        constexpr std::size_t per_owner = kBatchDepth * kBatchObjects *
                                           kParticipants;
        if (owners == 0 || owners > kWrIdSlotMask / per_owner) {
            ERROR("ec_rmw: invalid owner count for bounded exchange bank");
        }
        const std::size_t count = owners * per_owner;
        if (count > kWrIdSlotMask) {
            ERROR("ec_rmw: exchange bank exceeds wr_id slot field");
        }
        return count;
    }

    std::size_t index_of(const Exchange &exchange) const {
        const auto *base = exchanges_.get();
        const auto *ptr = &exchange;
        if (base == nullptr || ptr < base || ptr >= base + exchange_count_) {
            ERROR("ec_rmw: exchange does not belong to this transport");
        }
        const std::ptrdiff_t distance = ptr - base;
        if (distance < 0) {
            ERROR("ec_rmw: invalid exchange index");
        }
        return static_cast<std::size_t>(distance);
    }

    std::uint32_t arm_generation(Request &request) {
        const std::uint32_t old =
            request.generation.load(std::memory_order_relaxed);
        if (old >= kWrIdGenerationMask) {
            // A generation wrap would make a delayed CQE indistinguishable
            // from a current one.  This bounded transport intentionally
            // refuses that unsafe reuse instead of allowing ABA.
            ERROR("ec_rmw: request generation exhausted");
        }
        const std::uint32_t next = old + 1;
        request.generation.store(next, std::memory_order_release);
        return next;
    }

    void arm_common(Exchange &exchange, std::uint64_t remote_absolute_addr,
                    std::uint32_t rkey, std::uint32_t bytes,
                    ibv_wr_opcode opcode, const void *local_override,
                    std::uint32_t override_lkey, std::size_t source_bytes) {
        if (bytes == 0 || bytes > kMaxBytes) {
            ERROR("ec_rmw: request size exceeds kMaxBytes");
        }
        if (local_override == nullptr && source_bytes != 0) {
            ERROR("ec_rmw: source_bytes requires a local override");
        }
        if (local_override != nullptr && override_lkey == 0) {
            ERROR("ec_rmw: local override requires a registered lkey");
        }
        if (source_bytes > bytes) {
            ERROR("ec_rmw: source_bytes exceeds request bytes");
        }

        Request &request = exchange.request;
        if (request.posted &&
            request.load_state(std::memory_order_acquire) ==
                CompletionState::pending) {
            ERROR("ec_rmw: rearming a request before its posted CQE");
        }
        // The producer owns this lifecycle bit.  A completed request can be
        // rearmed by clearing it here; CQ pollers never touch it.
        request.posted = false;
        const std::size_t index = index_of(exchange);
        const std::uint32_t generation = arm_generation(request);

        request.wr = {};
        request.sges[0] = {};
        request.sges[1] = {};
        request.bytes = bytes;
        request.source_bytes = 0;
        request.wr_id = encode_wr_id(index, generation);
        request.wr.wr_id = request.wr_id;
        request.wr.opcode = opcode;
        request.wr.send_flags = IBV_SEND_SIGNALED;
        request.wr.wr.rdma.remote_addr = remote_absolute_addr;
        request.wr.wr.rdma.rkey = rkey;

        if (opcode == IBV_WR_RDMA_READ) {
            request.sges[0].addr = reinterpret_cast<std::uint64_t>(
                exchange.payload);
            request.sges[0].length = bytes;
            request.sges[0].lkey = exchange_mr_->lkey;
            request.sge_count = 1;
        } else {
            std::size_t source = source_bytes;
            if (local_override != nullptr && source == 0) source = bytes;
            if (local_override == nullptr) {
                request.sges[0].addr = reinterpret_cast<std::uint64_t>(
                    exchange.payload);
                request.sges[0].length = bytes;
                request.sges[0].lkey = exchange_mr_->lkey;
                request.sge_count = 1;
            } else {
                request.sges[0].addr =
                    reinterpret_cast<std::uint64_t>(local_override);
                request.sges[0].length = static_cast<std::uint32_t>(source);
                request.sges[0].lkey = override_lkey;
                request.source_bytes = static_cast<std::uint32_t>(source);
                request.sge_count = 1;
                if (source < bytes) {
                    // The data object may be shorter than the fixed wire
                    // width.  The second SGE supplies registered zero bytes,
                    // avoiding a copy into the bounded parity/data bank.
                    request.sges[1].addr = reinterpret_cast<std::uint64_t>(
                        zero_block_.get());
                    request.sges[1].length =
                        static_cast<std::uint32_t>(bytes - source);
                    request.sges[1].lkey = zero_mr_->lkey;
                    request.sge_count = 2;
                }
            }
        }
        request.wr.sg_list = request.sges;
        request.wr.num_sge = request.sge_count;
        request.wr.next = nullptr;

        // Publish pending only after the complete WR and generation are
        // initialized.  CQ pollers can therefore safely publish completion
        // with release semantics and ignore an older generation.
        request.completion_word.store(
            Request::pack_completion_word(generation,
                                          CompletionState::pending),
            std::memory_order_release);
    }

public:
    ClientTransport() = default;
    ClientTransport(const ClientTransport &) = delete;
    ClientTransport &operator=(const ClientTransport &) = delete;

    // The owner must stop/destroy every data QP before this object is
    // destroyed; posted WQEs still refer to the registered banks.
    ~ClientTransport() {
        if (zero_mr_ != nullptr) {
            CHECK_ERR(ibv_dereg_mr(zero_mr_));
            zero_mr_ = nullptr;
        }
        if (exchange_mr_ != nullptr) {
            CHECK_ERR(ibv_dereg_mr(exchange_mr_));
            exchange_mr_ = nullptr;
        }
        zero_block_.reset();
        exchanges_.reset();
        pd_ = nullptr;
    }

    // Allocate and register the complete bounded bank once, before Work (the
    // execution phase) starts.  `owners` is normally evacuate_thread_cnt.
    void init(ibv_pd *pd, std::size_t owners) {
        ASSERT(pd != nullptr);
        ASSERT(exchanges_ == nullptr);
        owners_ = owners;
        exchange_count_ = checked_exchange_count(owners);
        if (exchange_count_ > std::numeric_limits<std::size_t>::max() /
                                sizeof(Exchange)) {
            ERROR("ec_rmw: exchange bank byte size overflow");
        }
        exchange_mr_bytes_ = exchange_count_ * sizeof(Exchange);
        payload_bytes_ = exchange_count_ * kMaxBytes;

        exchanges_ = std::make_unique<Exchange[]>(exchange_count_);
        zero_block_ = std::make_unique<std::uint8_t[]>(kMaxBytes);
        std::memset(zero_block_.get(), 0, kMaxBytes);
        pd_ = pd;
        exchange_mr_ = ibv_reg_mr(pd_, exchanges_.get(), exchange_mr_bytes_,
                                  IBV_ACCESS_LOCAL_WRITE);
        if (exchange_mr_ == nullptr) {
            ERROR("ec_rmw: register exchange bank failed");
        }
        zero_mr_ = ibv_reg_mr(pd_, zero_block_.get(), kMaxBytes,
                              IBV_ACCESS_LOCAL_WRITE);
        if (zero_mr_ == nullptr) {
            CHECK_ERR(ibv_dereg_mr(exchange_mr_));
            exchange_mr_ = nullptr;
            ERROR("ec_rmw: register zero padding block failed");
        }
    }

    std::size_t owners() const noexcept { return owners_; }
    std::size_t exchange_count() const noexcept { return exchange_count_; }
    // `registered_bytes` reports the bytes covered by both MRs.  More precise
    // accessors are provided for diagnostics and tests of the fixed bank.
    std::size_t registered_bytes() const noexcept {
        return (exchange_mr_ != nullptr ? exchange_mr_bytes_ : 0) +
               (zero_mr_ != nullptr ? kMaxBytes : 0);
    }
    std::size_t registered_payload_bytes() const noexcept {
        return payload_bytes_;
    }
    std::size_t registered_exchange_bytes() const noexcept {
        return exchange_mr_ != nullptr ? exchange_mr_bytes_ : 0;
    }
    std::size_t registered_zero_bytes() const noexcept {
        return zero_mr_ != nullptr ? kMaxBytes : 0;
    }

    Exchange &get(std::size_t owner, std::size_t batch, std::size_t slot,
                  std::size_t participant) {
        ASSERT(owner < owners_);
        ASSERT(batch < kBatchDepth);
        ASSERT(slot < kBatchObjects);
        ASSERT(participant < kParticipants);
        const std::size_t index =
            (((owner * kBatchDepth) + batch) * kBatchObjects + slot) *
            kParticipants + participant;
        return exchanges_[index];
    }

    const Exchange &get(std::size_t owner, std::size_t batch,
                        std::size_t slot, std::size_t participant) const {
        return const_cast<ClientTransport *>(this)->get(owner, batch, slot,
                                                        participant);
    }

    void arm_read(Exchange &exchange, std::uint64_t remote_absolute_addr,
                  std::uint32_t rkey, std::uint32_t bytes) {
        arm_common(exchange, remote_absolute_addr, rkey, bytes,
                   IBV_WR_RDMA_READ, nullptr, 0, 0);
    }

    void arm_write(Exchange &exchange, std::uint64_t remote_absolute_addr,
                   std::uint32_t rkey, std::uint32_t bytes,
                   const void *local_override = nullptr,
                   std::uint32_t override_lkey = 0,
                   std::size_t source_bytes = 0) {
        arm_common(exchange, remote_absolute_addr, rkey, bytes,
                   IBV_WR_RDMA_WRITE, local_override, override_lkey,
                   source_bytes);
    }

    // Submit exactly one linked verbs chain.  A provider may accept only a
    // prefix under SQ pressure; ENOMEM/EAGAIN are returned as that prefix so
    // the coordinator can retry the untouched suffix without any hidden
    // allocation or wait.
    std::size_t post(Exchange **list, std::size_t count, ibv_qp *qp,
                     std::uint64_t *verbs_cycles = nullptr) {
        if (list == nullptr || count == 0 || qp == nullptr) return 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (list[i] == nullptr) ERROR("ec_rmw: null exchange in post");
            Request &request = list[i]->request;
            if (request.posted) {
                ERROR("ec_rmw: request posted twice without coordinator reuse");
            }
            request.wr.next =
                i + 1 < count ? &list[i + 1]->request.wr : nullptr;
        }
        ibv_send_wr *bad = nullptr;
        const std::uint64_t verbs_begin = verbs_cycles ? ::get_cycles() : 0;
        const int rc = ibv_post_send(qp, &list[0]->request.wr, &bad);
        if (verbs_cycles) *verbs_cycles += ::get_cycles() - verbs_begin;
        std::size_t accepted = count;
        if (rc != 0) {
            if (rc != ENOMEM && rc != EAGAIN) {
                ERROR("ec_rmw: ibv_post_send failed outside SQ backpressure");
            }
            if (bad == nullptr) {
                ERROR("ec_rmw: backpressure result did not return bad WR");
            }
            accepted = 0;
            while (accepted < count && &list[accepted]->request.wr != bad) {
                ++accepted;
            }
            if (accepted == count) {
                ERROR("ec_rmw: provider returned an unknown bad WR");
            }
        }
        // Only the producer writes this field.  Completion code must not
        // inspect or mutate it, since a CQE can race a retry/reuse decision.
        for (std::size_t i = 0; i < accepted; ++i) {
            list[i]->request.posted = true;
        }
        return accepted;
    }

    // Return true for a CQE in this bounded transport's namespace/range.  A
    // stale generation is still ours, but is intentionally ignored.  The CAS
    // makes duplicate CQEs unable to overwrite an already published result.
    bool handle(const ibv_wc &wc) {
        if (!is_wr_id(wc.wr_id)) return false;
        std::size_t index = 0;
        std::uint32_t generation = 0;
        if (!decode_wr_id(wc.wr_id, &index, &generation) ||
            index >= exchange_count_) {
            return false;
        }
        Request &request = exchanges_[index].request;
        // Read and claim the generation/state as one word.  If arm_common()
        // races this handler, the CAS below fails instead of publishing an
        // old CQE into the newly armed request.
        std::uint64_t expected = request.completion_word.load(
            std::memory_order_acquire);
        const std::uint32_t active_generation =
            static_cast<std::uint32_t>(expected >> 2);
        const CompletionState active_state = static_cast<CompletionState>(
            expected & std::uint64_t{3});
        if (active_generation != generation ||
            active_state != CompletionState::pending) {
            return true;
        }
        const CompletionState result =
            wc.status == IBV_WC_SUCCESS ? CompletionState::success
                                        : CompletionState::error;
        request.completion_word.compare_exchange_strong(
            expected,
            Request::pack_completion_word(generation, result),
            std::memory_order_acq_rel, std::memory_order_acquire);
        return true;
    }

    static bool is_wr_id(std::uint64_t wr_id) noexcept {
        if ((wr_id & kWrIdTopMask) != kWrIdTag) return false;
        return ((wr_id >> kWrIdSlotBits) & kWrIdGenerationMask) != 0;
    }
};

}  // namespace FarLib::rdma::ec_rmw
