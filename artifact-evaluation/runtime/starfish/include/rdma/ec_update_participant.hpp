#pragma once

#include "rdma/ec_update_protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>

namespace FarLib::rdma::ec_update {

// The participant is deliberately single-threaded.  The owner of the data
// receive CQ calls handle() and is also the only code that touches the
// prepared-image table or the registered server memory through this class.
// The table stores images rather than pointers into an incoming SEND buffer;
// a client may reuse its SEND slot as soon as the response is received.
class Participant {
public:
    enum class State : uint8_t { Prepared = 1, Committed = 2, Aborted = 3 };

private:
    struct TxnKey {
        uint32_t owner = 0;
        uint16_t slot = 0;

        bool operator==(const TxnKey &other) const {
            return owner == other.owner && slot == other.slot;
        }
    };

    struct TxnKeyHash {
        size_t operator()(const TxnKey &key) const noexcept {
            size_t value = static_cast<size_t>(key.owner);
            value = value * 1315423911u + key.slot;
            return value;
        }
    };

    struct Entry {
        uint64_t generation = 0;
        uint64_t offset = 0;
        uint16_t bytes = 0;
        uint8_t participant = 0;
        State state = State::Aborted;
        size_t capacity = 0;
        // For DataPrepare this is the requested new data; for ParityPrepare
        // it is the requested parity delta.  Keeping it makes a duplicate
        // PREPARE compareable without touching visible memory.
        std::unique_ptr<uint8_t[]> request;
        // DataPrepare returns old^new.  Retain that delta so a duplicate
        // request after COMMIT can be acknowledged without rereading memory
        // that may already belong to a newer generation.
        std::unique_ptr<uint8_t[]> delta;
        // The image copied to visible memory only after COMMIT.
        std::unique_ptr<uint8_t[]> image;
    };

    uint8_t *memory_ = nullptr;
    size_t capacity_ = 0;
    size_t owner_limit_ = 0;
    std::unordered_map<TxnKey, Entry, TxnKeyHash> entries_;

    static bool ranges_overlap(uint64_t left_offset, size_t left_bytes,
                               uint64_t right_offset, size_t right_bytes) {
        // Use subtraction so even a maximal configured capacity cannot make
        // an end-offset addition wrap before the overlap check.
        if (left_offset <= right_offset)
            return right_offset - left_offset < left_bytes;
        return left_offset - right_offset < right_bytes;
    }

    bool valid_request(const Message &request, size_t wire_size) const {
        if (!valid(request, wire_size) || request.response != 0 ||
            memory_ == nullptr || request.owner >= owner_limit_ ||
            request.offset > capacity_ ||
            static_cast<size_t>(request.bytes) > capacity_ - request.offset) {
            return false;
        }

        if (request.op == Op::DataPrepare) {
            if (request.participant != 0) return false;
        } else if (request.op == Op::ParityPrepare) {
            if (request.participant == 0) return false;
        } else if (request.op == Op::Commit || request.op == Op::Abort) {
            // Commit and abort carry only the fixed header.  bytes remains in
            // the header so that the participant can fence a slot identity.
            if (wire_size != kHeaderBytes) return false;
        } else {
            return false;
        }
        return true;
    }

    static Message make_response(const Message &request, Status status) {
        Message response{};
        response.op = (static_cast<uint8_t>(request.op) >=
                           static_cast<uint8_t>(Op::DataPrepare) &&
                       static_cast<uint8_t>(request.op) <=
                           static_cast<uint8_t>(Op::Abort))
                          ? request.op
                          : Op::DataPrepare;
        response.status = status;
        response.generation = request.generation;
        response.offset = request.offset;
        response.owner = request.owner;
        response.slot = request.slot;
        response.bytes = request.bytes == 0
                             ? 1
                             : static_cast<uint16_t>(request.bytes > kMaxBytes
                                                         ? kMaxBytes
                                                         : request.bytes);
        response.participant = request.participant < 3 ? request.participant : 0;
        response.response = 1;
        return response;
    }

    static TxnKey key_for(const Message &request) {
        return TxnKey{request.owner, request.slot};
    }

    bool overlaps_prepared(const Message &request, const TxnKey &key) const {
        for (const auto &item : entries_) {
            const TxnKey &other_key = item.first;
            const Entry &other = item.second;
            if (other.state != State::Prepared ||
                !ranges_overlap(other.offset, other.bytes, request.offset,
                                request.bytes)) {
                continue;
            }
            // A participant cannot prepare a second image for the same
            // address while another owner/slot transaction is pending.
            if (!(other_key == key && other.generation == request.generation)) {
                return true;
            }
        }
        return false;
    }

    static bool payload_matches(const Entry &entry, const Message &request) {
        return entry.bytes == request.bytes && entry.request != nullptr &&
               std::memcmp(entry.request.get(), request.payload,
                           request.bytes) == 0;
    }

    Status prepare(const Message &request, Message &response) {
        const TxnKey key = key_for(request);
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            Entry &entry = found->second;
            if (request.generation < entry.generation) {
                response.status = Status::Stale;
                return response.status;
            }
            if (request.generation == entry.generation) {
                if (entry.participant != request.participant ||
                    entry.offset != request.offset ||
                    entry.bytes != request.bytes ||
                    !payload_matches(entry, request)) {
                    response.status = Status::Conflict;
                    return response.status;
                }
                if (entry.state == State::Aborted) {
                    response.status = Status::Conflict;
                    return response.status;
                }
                response.status = Status::Ok;
                response.bytes = entry.bytes;
                if (request.op == Op::DataPrepare &&
                    entry.state != State::Aborted && entry.delta != nullptr) {
                    std::memcpy(response.payload, entry.delta.get(),
                                entry.bytes);
                }
                return response.status;
            }
            if (entry.state == State::Prepared) {
                // A slot cannot be reused while its previous prepare is
                // undecided.  The coordinator must COMMIT or ABORT first.
                response.status = Status::Conflict;
                return response.status;
            }
        }

        if (overlaps_prepared(request, key)) {
            response.status = Status::Conflict;
            return response.status;
        }

        // Only overwrite a terminal entry after all conflict checks pass.  A
        // rejected newer generation must leave the previous generation fence
        // intact so a delayed old COMMIT remains stale.
        auto [iterator, inserted] = entries_.try_emplace(key);
        (void)inserted;
        Entry &entry = iterator->second;
        entry.generation = request.generation;
        entry.offset = request.offset;
        entry.bytes = request.bytes;
        entry.participant = request.participant;
        entry.state = State::Prepared;
        if (entry.request == nullptr || entry.capacity < entry.bytes) {
            entry.request = std::make_unique<uint8_t[]>(entry.bytes);
            entry.delta = std::make_unique<uint8_t[]>(entry.bytes);
            entry.image = std::make_unique<uint8_t[]>(entry.bytes);
            entry.capacity = entry.bytes;
        }
        std::memcpy(entry.request.get(), request.payload, entry.bytes);

        auto *current = memory_ + entry.offset;
        if (request.op == Op::DataPrepare) {
            for (size_t i = 0; i < entry.bytes; ++i) {
                entry.image[i] = request.payload[i];
                entry.request[i] = request.payload[i];
                entry.delta[i] = current[i] ^ request.payload[i];
                response.payload[i] = entry.delta[i];
            }
        } else {
            // The parity request is already old^new.  Cache the absolute new
            // image while leaving the registered parity bytes untouched.
            for (size_t i = 0; i < entry.bytes; ++i) {
                entry.delta[i] = request.payload[i];
                entry.image[i] = current[i] ^ request.payload[i];
            }
        }

        response.status = Status::Ok;
        response.bytes = entry.bytes;
        return response.status;
    }

    Status commit(const Message &request, Message &response) {
        const TxnKey key = key_for(request);
        auto found = entries_.find(key);
        if (found == entries_.end()) {
            response.status = Status::Invalid;
            return response.status;
        }
        Entry &entry = found->second;
        if (request.generation < entry.generation) {
            response.status = Status::Stale;
            return response.status;
        }
        if (request.generation > entry.generation) {
            response.status = Status::Invalid;
            return response.status;
        }
        if (entry.participant != request.participant ||
            entry.offset != request.offset || entry.bytes != request.bytes) {
            response.status = Status::Conflict;
            return response.status;
        }
        if (entry.state == State::Committed) {
            response.status = Status::Ok;
            return response.status;
        }
        if (entry.state == State::Aborted || entry.image == nullptr) {
            response.status = Status::Conflict;
            return response.status;
        }

        std::memcpy(memory_ + entry.offset, entry.image.get(), entry.bytes);
        entry.state = State::Committed;
        response.status = Status::Ok;
        return response.status;
    }

    Status abort(const Message &request, Message &response) {
        const TxnKey key = key_for(request);
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            Entry &entry = found->second;
            if (request.generation < entry.generation) {
                response.status = Status::Stale;
                return response.status;
            }
            if (request.generation == entry.generation) {
                if (entry.participant != request.participant ||
                    entry.offset != request.offset ||
                    entry.bytes != request.bytes) {
                    response.status = Status::Conflict;
                    return response.status;
                }
                if (entry.state == State::Committed) {
                    // A commit decision is irrevocable; do not overwrite the
                    // committed image with a compensating write.
                    response.status = Status::Conflict;
                    return response.status;
                }
                if (entry.state == State::Aborted) {
                    response.status = Status::Ok;
                    return response.status;
                }
                entry.state = State::Aborted;
                response.status = Status::Ok;
                return response.status;
            }
            if (entry.state == State::Prepared) {
                response.status = Status::Conflict;
                return response.status;
            }
        }

        // Record an abort even when PREPARE never reached this participant.
        // That prevents a delayed PREPARE of the same generation from
        // resurrecting a transaction after the coordinator decided to abort.
        if (overlaps_prepared(request, key)) {
            response.status = Status::Conflict;
            return response.status;
        }
        auto [iterator, inserted] = entries_.try_emplace(key);
        (void)inserted;
        Entry &entry = iterator->second;
        entry.generation = request.generation;
        entry.offset = request.offset;
        entry.bytes = request.bytes;
        entry.participant = request.participant;
        entry.state = State::Aborted;
        response.status = Status::Ok;
        return response.status;
    }

public:
    Participant(void *memory, size_t capacity, size_t owner_limit)
        : memory_(static_cast<uint8_t *>(memory)),
          capacity_(capacity),
          owner_limit_(owner_limit) {}

    Participant(const Participant &) = delete;
    Participant &operator=(const Participant &) = delete;

    // Returns the status placed in response.  wire_size must be the exact
    // byte length reported by the receive CQE; malformed requests still get a
    // fixed-header INVALID response so the client can release its SEND slot.
    Status handle(const Message &request, size_t wire_size, Message &response) {
        response = make_response(request, Status::Invalid);
        if (!valid_request(request, wire_size)) return response.status;

        if (request.op == Op::DataPrepare || request.op == Op::ParityPrepare)
            return prepare(request, response);
        if (request.op == Op::Commit) return commit(request, response);
        if (request.op == Op::Abort) return abort(request, response);
        return response.status;
    }

    size_t prepared_entry_count() const { return entries_.size(); }
};

}  // namespace FarLib::rdma::ec_update
