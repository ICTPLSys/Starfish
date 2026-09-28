#include "cache/alloc/ec_direct_write_bank.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <thread>

namespace {

using Bank = FarLib::cache::ec_batch::EcDirectWriteBank;

struct CallbackState {
    size_t allocations = 0;
    size_t frees = 0;
};

bool allocate_bank(void *opaque, size_t bytes,
                   Bank::RegisteredBank *out) {
    if (opaque == nullptr || out == nullptr) return false;
    auto &state = *static_cast<CallbackState *>(opaque);
    auto *memory = new (std::nothrow) uint8_t[bytes];
    if (memory == nullptr) return false;
    ++state.allocations;
    out->base = memory;
    out->bytes = bytes;
    out->lkey = 0x4000u + static_cast<uint32_t>(state.allocations);
    out->registration = memory;
    return true;
}

void free_bank(void *opaque, Bank::RegisteredBank &bank) {
    auto &state = *static_cast<CallbackState *>(opaque);
    delete[] static_cast<uint8_t *>(bank.base);
    ++state.frees;
}

bool check(bool condition, const char *message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

void fill_record(Bank::Entry *entry,
                 std::array<std::array<uint8_t, Bank::kSlotBytes>,
                            Bank::kDataSlots> &payloads) {
    entry->record.group.slot_size = static_cast<uint32_t>(Bank::kSlotBytes);
    entry->record.live_mask = 0xf;
    entry->record.live_count = 4;
    for (size_t i = 0; i < Bank::kDataSlots; ++i) {
        entry->record.objects[i] = payloads[i].data();
        entry->record.object_sizes[i] = static_cast<uint32_t>(Bank::kSlotBytes);
    }
}

}  // namespace

int main() {
    static_assert(Bank::kSlotsPerOwner == 128);
    static_assert(Bank::kSlotBytes == 4096);

    CallbackState callbacks;
    {
        Bank bank(2);
        if (!check(bank.init(&callbacks, allocate_bank, free_bank),
                   "one-shot bank initialization") ||
            !check(callbacks.allocations == 2,
                   "one registered bank per owner") ||
            !check(bank.available(0) == Bank::kSlotsPerOwner,
                   "all owner slots initially available")) {
            return 1;
        }

        std::array<std::array<uint8_t, Bank::kSlotBytes>, Bank::kDataSlots>
            payloads{};
        uint64_t token = 0;
        Bank::Entry *entry = nullptr;
        Bank::Lease lease{};
        if (!check(bank.reserve(0, &token, &entry, &lease),
                   "reserve returns a persistent entry") ||
            !check(Bank::is_direct_token(token), "direct token namespace") ||
            !check(bank.get(token) == nullptr,
                   "reserved entry is invisible to completion") ||
            !check(entry->record.parity[0] == lease.parity[0] &&
                       entry->record.parity[1] == lease.parity[1] &&
                       entry->record.zero_pad == lease.zero_pad &&
                       entry->record.parity_lkey == lease.lkey,
                   "reserve fixes parity and zero-pad bindings")) {
            return 1;
        }
        fill_record(entry, payloads);
        if (!check(bank.publish(token), "publish reserved record") ||
            !check(bank.get(token) == entry, "published entry is visible") ||
            !check(bank.complete_segment(token, 0) == nullptr,
                   "non-final completion has no finalizer")) {
            return 1;
        }
        for (uint8_t segment = 1; segment < 6; ++segment) {
            Bank::Entry *winner = bank.complete_segment(token, segment);
            if (segment != 5 &&
                !check(winner == nullptr, "only sixth completion wins")) {
                return 1;
            }
            if (segment == 5 &&
                !check(winner == entry && entry->recoverable(),
                       "final completion publishes recoverability")) {
                return 1;
            }
        }
        if (!check(bank.release(token), "finalizer releases completed slot") ||
            !check(bank.get(token) == nullptr,
                   "released token is stale") ||
            !check(bank.available(0) == Bank::kSlotsPerOwner,
                   "released slot becomes available")) {
            return 1;
        }

        // Six completion threads may race; exactly one receives the finalizer.
        uint64_t concurrent_token = 0;
        Bank::Entry *concurrent_entry = nullptr;
        Bank::Lease concurrent_lease{};
        if (!bank.reserve(0, &concurrent_token, &concurrent_entry,
                          &concurrent_lease)) {
            std::fprintf(stderr, "FAIL: concurrent reserve\n");
            return 1;
        }
        fill_record(concurrent_entry, payloads);
        if (!check(bank.publish(concurrent_token),
                   "publish concurrent completion record")) {
            return 1;
        }
        std::atomic<size_t> winners{0};
        std::array<std::thread, 6> completion_threads;
        for (size_t segment = 0; segment < 6; ++segment) {
            completion_threads[segment] = std::thread(
                [&, segment] {
                    if (bank.complete_segment(concurrent_token,
                                              static_cast<uint8_t>(segment)) !=
                        nullptr) {
                        winners.fetch_add(1, std::memory_order_relaxed);
                    }
                });
        }
        for (auto &thread : completion_threads) thread.join();
        if (!check(winners.load(std::memory_order_relaxed) == 1,
                   "exactly one concurrent finalizer") ||
            !check(concurrent_entry->recoverable(),
                   "concurrent finalizer sees recoverable record") ||
            !check(bank.release(concurrent_token),
                   "concurrent finalizer releases slot")) {
            return 1;
        }

        // Cancel is reserved-only, and a later reuse advances generation so
        // the old token cannot complete the newly reserved slot.
        uint64_t old_token = 0;
        Bank::Entry *old_entry = nullptr;
        Bank::Lease old_lease{};
        if (!bank.reserve(1, &old_token, &old_entry, &old_lease) ||
            !check(bank.cancel(old_token), "cancel unpublished reservation") ||
            !check(bank.complete_segment(old_token, 0) == nullptr,
                   "canceled token cannot complete")) {
            return 1;
        }
        std::array<uint64_t, Bank::kSlotsPerOwner - 1> held_tokens{};
        std::array<Bank::Entry *, Bank::kSlotsPerOwner - 1> held_entries{};
        std::array<Bank::Lease, Bank::kSlotsPerOwner - 1> held_leases{};
        for (size_t i = 0; i < held_tokens.size(); ++i) {
            if (!bank.reserve(1, &held_tokens[i], &held_entries[i],
                              &held_leases[i])) {
                std::fprintf(stderr, "FAIL: fill owner slots\n");
                return 1;
            }
        }
        uint64_t reused_token = 0;
        Bank::Entry *reused_entry = nullptr;
        Bank::Lease reused_lease{};
        if (!bank.reserve(1, &reused_token, &reused_entry, &reused_lease) ||
            !check(reused_token != old_token,
                   "slot reuse advances generation")) {
            return 1;
        }
        if (!check(bank.cancel(reused_token), "cancel reused reservation")) {
            return 1;
        }
        for (uint64_t held_token : held_tokens) {
            if (!bank.cancel(held_token)) {
                std::fprintf(stderr, "FAIL: cancel held reservation\n");
                return 1;
            }
        }
        if (!check(bank.available(1) == Bank::kSlotsPerOwner,
                   "all canceled slots become available")) {
            return 1;
        }
    }

    if (!check(callbacks.frees == callbacks.allocations,
               "registered owner banks freed after quiescence")) {
        return 1;
    }
    std::printf("ec_direct_write_bank PASS\n");
    return 0;
}
