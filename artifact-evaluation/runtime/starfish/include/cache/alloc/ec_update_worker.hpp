#pragma once
#include "cache/alloc/small_object_stripe.hpp"
#include "rdma/ec_update_protocol.hpp"
#include <array>

namespace FarLib::cache::ec_update_runtime {
struct Transaction {
    SmallObjectStripeManager::SlotReuse reservation;
    void *source = nullptr;
    uint64_t generation = 0;
    bool prepared = false;
    uint8_t prepared_mask = 0;
};
// Heap-allocated once per logical evacuator. No global assembly queue, per-
// object allocation, shared candidate pool, or new scheduler yield is needed.
struct Worker {
    std::array<Transaction, rdma::ec_update::kSlots> transactions{};
    std::array<uint64_t, rdma::ec_update::kSlots> generations{};
    size_t count = 0;
    uint64_t committed = 0, aborted = 0, payload_bytes = 0;
    uint64_t request_bytes = 0, phases = 0;
};
} // namespace FarLib::cache::ec_update_runtime
