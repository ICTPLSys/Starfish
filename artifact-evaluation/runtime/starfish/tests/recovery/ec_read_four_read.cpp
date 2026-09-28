// CPU-only four-read EC recovery oracle.
//
// This test deliberately uses the byte-level codec as the independent oracle:
// every one- and two-shard failure set is exercised, including missing data
// shards, and every possible group of four survivors is checked.  It also
// drives the direct completion context with exactly four selected reads,
// including a CQE that arrives before its post mark and a reordered CQE set.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "recovery/ec_read_context.hpp"
#include "recovery/ec_read_recovery.hpp"
#include "cache/alloc/small_object_stripe_codec.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

using FarLib::cache::ec_batch::EcStagingGroupSlot;
using FarLib::cache::ec_read_recovery::EcReadContext;
using FarLib::cache::ec_read_recovery::EcReadContextEvent;
using FarLib::cache::ec_read_recovery::EcReadContextPool;
using FarLib::cache::ec_read_recovery::EcReadTokenEventKind;
using FarLib::cache::ec_read_recovery::ec_read_selected_mask;
using FarLib::cache::ec_read_recovery::kEcReadSegmentCount;
using FarLib::cache::ec_read_recovery::make_ec_read_plan;
using FarLib::cache::kStripeCodecDataShards;
using FarLib::cache::kStripeCodecShardCount;
using FarLib::cache::small_object_stripe_encode_shards;
using FarLib::cache::small_object_stripe_rebuild_one;

constexpr size_t kByteCount = 257;
constexpr uint8_t kAllSegments =
    static_cast<uint8_t>((1u << kEcReadSegmentCount) - 1u);

using Shard = std::array<uint8_t, kByteCount>;
using Stripe = std::array<Shard, kStripeCodecShardCount>;

Stripe make_stripe() {
    Stripe stripe{};
    for (size_t shard = 0; shard < kStripeCodecDataShards; ++shard) {
        for (size_t byte = 0; byte < kByteCount; ++byte) {
            // Include both shard and byte position so a wrong survivor order
            // or a stale scratch buffer is caught by the oracle.
            stripe[shard][byte] = static_cast<uint8_t>(
                0x13u + 0x37u * shard + 0x0bu * byte +
                ((shard + 1u) * (byte + 3u) >> 1));
        }
    }
    const void *data[kStripeCodecDataShards] = {};
    void *parity[2] = {};
    for (size_t shard = 0; shard < kStripeCodecDataShards; ++shard) {
        data[shard] = stripe[shard].data();
    }
    for (size_t parity_shard = 0; parity_shard < 2; ++parity_shard) {
        parity[parity_shard] =
            stripe[kStripeCodecDataShards + parity_shard].data();
    }
    assert(small_object_stripe_encode_shards(data, parity, kByteCount, 0));
    return stripe;
}

size_t popcount(uint8_t mask) {
    size_t count = 0;
    for (size_t bit = 0; bit < kStripeCodecShardCount; ++bit) {
        count += (mask & static_cast<uint8_t>(1u << bit)) != 0;
    }
    return count;
}

void test_plan_preserves_physical_alive_and_selects_four() {
    for (uint8_t missing = 1; missing <= kAllSegments; ++missing) {
        const size_t missing_count = popcount(missing);
        if (missing_count != 1 && missing_count != 2) continue;
        const uint8_t physical_alive = static_cast<uint8_t>(
            (~missing) & kAllSegments);
        const auto plan = make_ec_read_plan(physical_alive, 0);
        assert(plan.alive_mask == physical_alive);
        assert(plan.missing_mask == missing);
        assert(plan.alive_count == 6 - missing_count);
        assert(plan.missing_count == missing_count);
        assert(plan.usable);
        assert(plan.read_mask == ec_read_selected_mask(physical_alive));
        assert(popcount(plan.read_mask) == kStripeCodecDataShards);
        assert((plan.read_mask & static_cast<uint8_t>(~physical_alive)) == 0);

        // Every missing data shard must remain absent from the selected set;
        // the runtime can therefore rebuild it from the four fetched shards.
        for (uint8_t own = 0; own < kStripeCodecDataShards; ++own) {
            if ((missing & static_cast<uint8_t>(1u << own)) == 0) continue;
            const auto own_plan = make_ec_read_plan(physical_alive, own);
            assert(own_plan.own_shard_missing);
            assert(own_plan.needs_rebuild());
            assert((own_plan.read_mask & static_cast<uint8_t>(1u << own)) ==
                   0);
        }
    }
}

void test_all_failure_sets_and_any_four_survivors() {
    const Stripe original = make_stripe();
    for (uint8_t missing = 1; missing <= kAllSegments; ++missing) {
        const size_t missing_count = popcount(missing);
        if (missing_count != 1 && missing_count != 2) continue;

        // For one failure there are five physical survivors, so enumerate all
        // five possible four-survivor choices.  For two failures this visits
        // the unique four-survivor set.  Rebuild every requested shard in the
        // failure set, including each missing data shard.
        for (uint8_t target = 0; target < kStripeCodecShardCount; ++target) {
            if ((missing & static_cast<uint8_t>(1u << target)) == 0) continue;
            for (uint8_t survivor_mask = 0; survivor_mask <= kAllSegments;
                 ++survivor_mask) {
                if (popcount(survivor_mask) != kStripeCodecDataShards ||
                    (survivor_mask & missing) != 0) {
                    continue;
                }
                uint8_t survivor_idx[kStripeCodecDataShards] = {};
                const void *survivors[kStripeCodecDataShards] = {};
                size_t survivor_count = 0;
                for (uint8_t shard = 0; shard < kStripeCodecShardCount;
                     ++shard) {
                    if ((survivor_mask & static_cast<uint8_t>(1u << shard)) ==
                        0) {
                        continue;
                    }
                    survivor_idx[survivor_count] = shard;
                    survivors[survivor_count] = original[shard].data();
                    ++survivor_count;
                }
                assert(survivor_count == kStripeCodecDataShards);

                Stripe rebuilt = original;
                std::memset(rebuilt[target].data(), 0, kByteCount);
                assert(small_object_stripe_rebuild_one(
                    target, survivor_idx, survivors, rebuilt[target].data(),
                    kByteCount, 0));
                assert(std::memcmp(rebuilt[target].data(),
                                   original[target].data(), kByteCount) == 0);
            }
        }
    }
}

void assert_pending(const EcReadContextEvent &event) {
    assert(event.kind == EcReadTokenEventKind::kPending);
    assert(event.context == nullptr);
    assert(event.token == nullptr);
}

void test_four_read_context_early_and_reordered() {
    EcReadContextPool pool(1, 0);
    const uint8_t read_mask = static_cast<uint8_t>((1u << 0) |
                                                   (1u << 2) |
                                                   (1u << 3) |
                                                   (1u << 5));
    uint64_t token_id = 0;
    EcReadContext *context = nullptr;
    EcStagingGroupSlot scratch;
    assert(pool.acquire(0, 0x100000, 0x200000, kByteCount, read_mask, 1,
                        scratch, &token_id, &context));
    assert(context != nullptr);
    assert(context->alive_mask == read_mask);

    // A skipped physical survivor is not a valid completion for this
    // generation, even though the six-shard namespace still has its bit.
    const auto skipped = pool.complete_segment_event(token_id, 4, true);
    assert(skipped.kind == EcReadTokenEventKind::kIgnored);

    // Segment 5 completes before its post bookkeeping mark.  The remaining
    // remaining three CQEs are deliberately reordered, and completion must wait for all
    // four posted bits before producing the sole winner.
    assert_pending(pool.complete_segment_event(token_id, 5, true));
    assert(pool.mark_segment_posted(token_id, 3));
    assert(pool.mark_segment_posted(token_id, 5));
    assert(pool.mark_segment_posted(token_id, 0));
    assert(pool.mark_segment_posted(token_id, 2));
    assert_pending(pool.complete_segment_event(token_id, 0, true));
    assert_pending(pool.complete_segment_event(token_id, 3, true));
    assert_pending(pool.complete_segment_event(token_id, 2, true));
    const auto winner = pool.finish_posting(token_id);
    assert(winner.kind == EcReadTokenEventKind::kWinner);
    assert(winner.context == context);
    assert(pool.release(token_id));
}

}  // namespace

int main() {
    test_plan_preserves_physical_alive_and_selects_four();
    test_all_failure_sets_and_any_four_survivors();
    test_four_read_context_early_and_reordered();
    std::cout << "EC_READ_FOUR_READ_PASS\n";
    return 0;
}
