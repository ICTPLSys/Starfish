#ifdef NDEBUG
#undef NDEBUG
#endif

#include "cache/alloc/small_object_stripe_codec.hpp"
#include "rdma/ec_update_participant.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using FarLib::cache::small_object_stripe_encode_parity_delta;
using FarLib::cache::small_object_stripe_encode_shards;
using FarLib::cache::small_object_stripe_initialize_encoder;
using FarLib::cache::small_object_stripe_rebuild_shards;
using FarLib::rdma::ec_update::Message;
using FarLib::rdma::ec_update::Op;
using FarLib::rdma::ec_update::Participant;
using FarLib::rdma::ec_update::Status;

Message make_request(Op op, uint64_t generation, uint32_t owner,
                     uint16_t slot, uint8_t participant, uint64_t offset,
                     uint16_t bytes) {
    Message message{};
    message.op = op;
    message.generation = generation;
    message.owner = owner;
    message.slot = slot;
    message.participant = participant;
    message.offset = offset;
    message.bytes = bytes;
    return message;
}

Status submit(Participant &participant, const Message &message,
              Message &response) {
    return participant.handle(message,
                              FarLib::rdma::ec_update::wire_bytes(message),
                              response);
}

void fill_payload(Message &message, uint8_t seed) {
    for (size_t i = 0; i < message.bytes; ++i)
        message.payload[i] = static_cast<uint8_t>(seed + 3u * i);
}

void test_normal_replay_abort_and_fences() {
    std::array<uint8_t, 128> memory{};
    for (size_t i = 0; i < memory.size(); ++i)
        memory[i] = static_cast<uint8_t>(0x20u + i);
    const auto before = memory;
    Participant participant(memory.data(), memory.size(), 2);
    Message response{};

    Message data = make_request(Op::DataPrepare, 1, 0, 0, 0, 8, 16);
    fill_payload(data, 0xa0);
    assert(submit(participant, data, response) == Status::Ok);
    for (size_t i = 0; i < data.bytes; ++i)
        assert(response.payload[i] == (before[data.offset + i] ^ data.payload[i]));
    assert(memory == before);

    Message replay{};
    assert(submit(participant, data, replay) == Status::Ok);
    assert(std::memcmp(replay.payload, response.payload, data.bytes) == 0);
    assert(memory == before);

    Message conflicting = data;
    conflicting.payload[0] ^= 1;
    assert(submit(participant, conflicting, response) == Status::Conflict);
    assert(memory == before);

    Message commit = data;
    commit.op = Op::Commit;
    std::memset(commit.payload, 0, sizeof(commit.payload));
    assert(submit(participant, commit, response) == Status::Ok);
    for (size_t i = 0; i < data.bytes; ++i)
        assert(memory[data.offset + i] ==
               static_cast<uint8_t>(0xa0u + 3u * i));
    const auto committed = memory;
    assert(submit(participant, commit, response) == Status::Ok);
    assert(memory == committed);

    // The same logical owner/slot may change endpoint role only at a newer
    // generation.  A delayed old role is stale, not a fresh transaction.
    Message changed_role = make_request(Op::ParityPrepare, 2, 0, 0, 1, 80, 16);
    fill_payload(changed_role, 0x31);
    assert(submit(participant, changed_role, response) == Status::Ok);
    Message old_commit = commit;
    assert(submit(participant, old_commit, response) == Status::Stale);
    Message role_commit = changed_role;
    role_commit.op = Op::Commit;
    std::memset(role_commit.payload, 0, sizeof(role_commit.payload));
    assert(submit(participant, role_commit, response) == Status::Ok);
    Message forbidden_abort = role_commit;
    forbidden_abort.op = Op::Abort;
    assert(submit(participant, forbidden_abort, response) == Status::Conflict);

    Message aborted = make_request(Op::DataPrepare, 3, 0, 0, 0, 8, 16);
    fill_payload(aborted, 0x44);
    assert(submit(participant, aborted, response) == Status::Ok);
    const auto before_abort = memory;
    Message abort = aborted;
    abort.op = Op::Abort;
    std::memset(abort.payload, 0, sizeof(abort.payload));
    assert(submit(participant, abort, response) == Status::Ok);
    assert(submit(participant, abort, response) == Status::Ok);
    assert(memory == before_abort);
    assert(submit(participant, aborted, response) == Status::Conflict);

    // A newer abort fences an old late phase, while a newer prepare may reuse
    // the slot after the terminal abort.
    Message abort_newer = make_request(Op::Abort, 4, 0, 0, 0, 8, 16);
    assert(submit(participant, abort_newer, response) == Status::Ok);
    Message stale_abort = abort_newer;
    stale_abort.generation = 3;
    assert(submit(participant, stale_abort, response) == Status::Stale);
    Message reuse = make_request(Op::DataPrepare, 5, 0, 0, 0, 8, 16);
    fill_payload(reuse, 0x55);
    assert(submit(participant, reuse, response) == Status::Ok);
    Message reuse_abort = reuse;
    reuse_abort.op = Op::Abort;
    assert(submit(participant, reuse_abort, response) == Status::Ok);

}

void test_overlap_and_failure_paths() {
    std::array<uint8_t, 64> memory{};
    Participant participant(memory.data(), memory.size(), 2);
    Message response{};

    Message first = make_request(Op::DataPrepare, 3, 0, 2, 0, 4, 16);
    fill_payload(first, 0x11);
    assert(submit(participant, first, response) == Status::Ok);

    Message overlap = make_request(Op::DataPrepare, 1, 1, 3, 0, 12, 16);
    fill_payload(overlap, 0x22);
    assert(submit(participant, overlap, response) == Status::Conflict);

    // A failed second participant must not cause a speculative image write.
    Message missing_commit = make_request(Op::Commit, 9, 1, 3, 0, 12, 16);
    assert(submit(participant, missing_commit, response) == Status::Invalid);
    for (uint8_t byte : memory) assert(byte == 0);

    Message stale = first;
    stale.generation = 2;
    assert(submit(participant, stale, response) == Status::Stale);

    Message out_of_bounds = make_request(Op::DataPrepare, 1, 0, 4, 0, 60, 8);
    fill_payload(out_of_bounds, 0x33);
    assert(submit(participant, out_of_bounds, response) == Status::Invalid);

    Message bad_owner = first;
    bad_owner.owner = 2;
    assert(submit(participant, bad_owner, response) == Status::Invalid);
}

void assert_rebuilds_all_single_and_double_losses(
    const std::array<std::vector<uint8_t>, 6> &expected, size_t bytes) {
    for (uint8_t alive = 0; alive < 64; ++alive) {
        const unsigned missing = 6u - std::popcount(static_cast<unsigned>(alive));
        if (missing == 0 || missing > 2) continue;

        std::array<std::vector<uint8_t>, 6> recovered = expected;
        std::array<void *, 6> shard_ptrs{};
        for (size_t shard = 0; shard < 6; ++shard) {
            if ((alive & (1u << shard)) == 0)
                std::fill(recovered[shard].begin(), recovered[shard].end(), 0);
            shard_ptrs[shard] = recovered[shard].data();
        }
        assert(small_object_stripe_rebuild_shards(alive, shard_ptrs.data(),
                                                  bytes));
        for (size_t shard = 0; shard < 6; ++shard)
            assert(recovered[shard] == expected[shard]);
    }
}

void test_delta_full_encode_and_commit_orders() {
    constexpr std::array<size_t, 5> kSizes{{8, 24, 64, 512, 4096}};

    for (size_t bytes : kSizes) {
        for (uint8_t data_idx = 0; data_idx < 4; ++data_idx) {
            for (size_t order_seed = 0; order_seed < 6; ++order_seed) {
                std::array<std::vector<uint8_t>, 6> shards;
                for (auto &shard : shards) shard.resize(bytes);
                for (size_t shard = 0; shard < 4; ++shard)
                    for (size_t i = 0; i < bytes; ++i)
                        shards[shard][i] = static_cast<uint8_t>(
                            0x10u + 13u * shard + 7u * i + order_seed);

                const void *data_ptrs[4] = {shards[0].data(), shards[1].data(),
                                             shards[2].data(), shards[3].data()};
                void *parity_ptrs[2] = {shards[4].data(), shards[5].data()};
                assert(small_object_stripe_encode_shards(data_ptrs,
                                                         parity_ptrs, bytes));
                const auto initial = shards;

                std::array<std::vector<uint8_t>, 6> expected = initial;
                for (size_t i = 0; i < bytes; ++i)
                    expected[data_idx][i] ^= static_cast<uint8_t>(0x5du + i);
                const void *expected_data[4] = {
                    expected[0].data(), expected[1].data(), expected[2].data(),
                    expected[3].data()};
                std::array<std::vector<uint8_t>, 2> expected_parity;
                expected_parity[0].resize(bytes);
                expected_parity[1].resize(bytes);
                void *expected_parity_ptrs[2] = {expected_parity[0].data(),
                                                 expected_parity[1].data()};
                assert(small_object_stripe_encode_shards(
                    expected_data, expected_parity_ptrs, bytes));

                Participant data_part(shards[data_idx].data(), bytes, 1);
                Participant p0_part(shards[4].data(), bytes, 1);
                Participant p1_part(shards[5].data(), bytes, 1);
                Message data = make_request(Op::DataPrepare, 1, 0, 0, 0, 0,
                                            static_cast<uint16_t>(bytes));
                std::memcpy(data.payload, expected[data_idx].data(), bytes);
                Message response{};
                assert(submit(data_part, data, response) == Status::Ok);
                const auto data_delta = response;
                std::array<uint8_t, 4096> p0_delta{}, p1_delta{};
                assert(small_object_stripe_encode_parity_delta(
                    data_idx, data_delta.payload, bytes, p0_delta.data(),
                    p1_delta.data()));
                Message p0 = make_request(Op::ParityPrepare, 1, 0, 0, 1, 0,
                                           static_cast<uint16_t>(bytes));
                Message p1 = make_request(Op::ParityPrepare, 1, 0, 0, 2, 0,
                                           static_cast<uint16_t>(bytes));
                std::memcpy(p0.payload, p0_delta.data(), bytes);
                std::memcpy(p1.payload, p1_delta.data(), bytes);
                assert(submit(p0_part, p0, response) == Status::Ok);
                assert(submit(p1_part, p1, response) == Status::Ok);
                assert(shards[data_idx] == initial[data_idx]);
                assert(shards[4] == initial[4]);
                assert(shards[5] == initial[5]);

                Message duplicate{};
                assert(submit(data_part, data, duplicate) == Status::Ok);
                assert(std::memcmp(duplicate.payload, data_delta.payload,
                                   bytes) == 0);
                assert(submit(p0_part, p0, duplicate) == Status::Ok);
                assert(submit(p1_part, p1, duplicate) == Status::Ok);

                std::array<size_t, 3> order{{0, 1, 2}};
                for (size_t i = 0; i < order_seed; ++i)
                    std::next_permutation(order.begin(), order.end());
                Participant *participants[3] = {&data_part, &p0_part,
                                                &p1_part};
                Message commits[3] = {data, p0, p1};
                for (size_t position : order) {
                    commits[position].op = Op::Commit;
                    std::memset(commits[position].payload, 0,
                                sizeof(commits[position].payload));
                    assert(submit(*participants[position], commits[position],
                                  response) == Status::Ok);
                }
                assert(submit(data_part, commits[0], response) == Status::Ok);
                assert(submit(p0_part, commits[1], response) == Status::Ok);
                assert(submit(p1_part, commits[2], response) == Status::Ok);

                std::array<std::vector<uint8_t>, 6> live;
                live[0] = expected[0]; live[1] = expected[1];
                live[2] = expected[2]; live[3] = expected[3];
                live[4] = expected_parity[0]; live[5] = expected_parity[1];
                assert(shards[data_idx] == live[data_idx]);
                assert(shards[4] == live[4]);
                assert(shards[5] == live[5]);
                assert_rebuilds_all_single_and_double_losses(live, bytes);
            }
        }
    }
}

// Model a memory endpoint disappearing after the COMMIT decision but before
// its image was installed (or after installation with its ACK lost). Only
// surviving participants are required to finish; recovery must match a fresh
// full encode, even when failed nodes still physically contain OLD bytes.
void test_failure_during_commit() {
    for (size_t bytes : {size_t{64}, size_t{4096}}) {
        for (uint8_t target = 0; target < 4; ++target) {
            for (uint8_t alive = 0; alive < 64; ++alive) {
                const auto survivors = std::popcount(static_cast<unsigned>(alive));
                if (survivors != 4 && survivors != 5) continue;
                std::array<std::vector<uint8_t>, 6> shards;
                for (size_t s = 0; s < 6; ++s) shards[s].resize(bytes);
                for (size_t s = 0; s < 4; ++s)
                    for (size_t b = 0; b < bytes; ++b)
                        shards[s][b] = static_cast<uint8_t>(s * 17 + b * 3);
                const void *data[4] = {shards[0].data(), shards[1].data(), shards[2].data(), shards[3].data()};
                void *parity[2] = {shards[4].data(), shards[5].data()};
                assert(small_object_stripe_encode_shards(data, parity, bytes));
                auto expected = shards;
                for (auto &byte : expected[target]) byte ^= 0x7b;
                const void *next_data[4] = {expected[0].data(), expected[1].data(), expected[2].data(), expected[3].data()};
                void *next_parity[2] = {expected[4].data(), expected[5].data()};
                assert(small_object_stripe_encode_shards(next_data, next_parity, bytes));

                Participant d(shards[target].data(), bytes, 1);
                Participant p0(shards[4].data(), bytes, 1);
                Participant p1(shards[5].data(), bytes, 1);
                Participant *participants[3] = {&d, &p0, &p1};
                const uint8_t positions[3] = {target, 4, 5};
                Message requests[3] = {
                    make_request(Op::DataPrepare, 1, 0, 0, 0, 0, bytes),
                    make_request(Op::ParityPrepare, 1, 0, 0, 1, 0, bytes),
                    make_request(Op::ParityPrepare, 1, 0, 0, 2, 0, bytes)};
                std::memcpy(requests[0].payload, expected[target].data(), bytes);
                Message response{};
                assert(submit(d, requests[0], response) == Status::Ok);
                assert(small_object_stripe_encode_parity_delta(target, response.payload, bytes,
                    requests[1].payload, requests[2].payload));
                assert(submit(p0, requests[1], response) == Status::Ok);
                assert(submit(p1, requests[2], response) == Status::Ok);
                for (size_t p = 0; p < 3; ++p) {
                    requests[p].op = Op::Commit;
                    if (!(alive & (1u << positions[p]))) continue;
                    assert(submit(*participants[p], requests[p], response) == Status::Ok);
                    // An ACK can be lost; retry must not apply a delta twice.
                    assert(submit(*participants[p], requests[p], response) == Status::Ok);
                }
                void *rebuild[6];
                for (size_t s = 0; s < 6; ++s) {
                    if (!(alive & (1u << s))) std::fill(shards[s].begin(), shards[s].end(), 0);
                    rebuild[s] = shards[s].data();
                }
                assert(small_object_stripe_rebuild_shards(alive, rebuild, bytes));
                assert(shards == expected);
            }
        }
    }
}

}  // namespace

int main() {
    small_object_stripe_initialize_encoder();
    test_normal_replay_abort_and_fences();
    test_overlap_and_failure_paths();
    test_delta_full_encode_and_commit_orders();
    test_failure_during_commit();
    return 0;
}
