// Ported from nq-resident-six r2/t8; assertions intentionally retained.
#include "cache/placement/simple_region_budget.hpp"
#include "cache/placement/simple_region_heat.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

// Keep these checks active in the Release build used by the repository tests.
#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "check failed: " << #condition << " at " << __FILE__ \
                      << ':' << __LINE__ << '\n';                             \
            return false;                                                      \
        }                                                                      \
    } while (false)

namespace {

using FarLib::simple_region_budget::Controller;
using FarLib::simple_region_budget::Counts;
using FarLib::simple_region_budget::Mode;
using FarLib::simple_region_budget::Snapshot;
using FarLib::simple_region_budget::Supply;

constexpr std::size_t kBin = 7;
constexpr std::size_t kRegionBytes = 4096;

std::uint64_t total(const Counts &counts) {
    return FarLib::simple_region_budget::sum(counts);
}

std::uint64_t band_total(const Counts &counts, std::size_t band) {
    const std::size_t first = band == 0 ? 0 : 3;
    return counts[first] + counts[first + 1] + counts[first + 2];
}

void seed(Controller &controller, const Counts &counts) {
    for (std::size_t klass = 0;
         klass < FarLib::simple_region_budget::classes(); ++klass)
        for (std::uint64_t n = 0; n < counts[klass]; ++n)
            controller.claim(kBin,
                             FarLib::simple_region_budget::label(klass));
}

Supply supply_for(std::uint8_t klass, std::uint64_t regions) {
    Supply supply{};
    supply[kBin][FarLib::simple_region_budget::index(klass)] =
        regions * kRegionBytes;
    return supply;
}

void request(Controller &controller, std::uint8_t klass,
             std::uint64_t attempts, std::uint64_t misses) {
    for (std::uint64_t n = 0; n < attempts; ++n)
        controller.note_request(kBin, klass, n < misses);
}

bool test_api_and_local_fallback() {
    CHECK(FarLib::simple_region_heat::local_resident_enabled());
    for (std::uint8_t klass = 0; klass < 6; ++klass) {
        CHECK(FarLib::simple_region_heat::class_for_placement(false, klass) ==
              klass % 3);
        CHECK(FarLib::simple_region_heat::class_for_placement(true, klass) ==
              3 + klass % 3);
    }

    const auto streaming = FarLib::simple_region_heat::local_fallback_order(2);
    CHECK(streaming[0] == 2 && streaming[1] == 0 && streaming[2] == 1);
    CHECK(streaming[3] >= 3 && streaming[3] < 6);
    const auto resident = FarLib::simple_region_heat::local_fallback_order(4);
    CHECK(resident[0] == 4 && resident[1] == 3 && resident[2] == 5);

    // fallback_order is the measured-heat/remote order and remains the old
    // six-way requested/opposite sequence even in local Resident mode.
    const auto measured = FarLib::simple_region_heat::fallback_order(2);
    CHECK(measured[0] == 2 && measured[1] == 5);
    return true;
}

bool test_local_band_budget_isolation() {
    Controller controller("resident-local", Mode::Adaptive, true);
    controller.configure(kRegionBytes);
    Counts initial{};
    initial.fill(12);
    seed(controller, initial);
    controller.begin();

    // Class 4 is short, but class 3 is the only public donor.  Any transfer
    // must stay inside the Resident band [3,5].
    const auto supply = supply_for(3, 8);
    request(controller, 4, 100, 20);
    controller.tick(supply, 1);
    request(controller, 4, 100, 20);
    controller.tick(supply, 2);
    const Snapshot planned = controller.snapshot(kBin);
    CHECK(total(planned.actual) == 72);
    CHECK(total(planned.target) == 72);
    CHECK(band_total(planned.actual, 0) == 36);
    CHECK(band_total(planned.target, 0) == 36);
    CHECK(band_total(planned.actual, 1) == 36);
    CHECK(band_total(planned.target, 1) == 36);
    CHECK(planned.transferred > 0);
    CHECK(planned.target[4] > planned.actual[4]);
    CHECK(planned.target[3] < planned.actual[3]);

    // Settle any public donor debt and verify that the local controller never
    // creates a cross-band pending transfer.
    const auto pending = controller.pending_transfers();
    for (std::size_t donor = 0; donor < 6; ++donor)
        for (std::uint64_t n = 0; n < pending[kBin][donor]; ++n)
            CHECK(controller.reassign_public(
                      kBin, FarLib::simple_region_budget::label(donor)) !=
                  FarLib::simple_region_budget::label(donor));
    const auto settled = controller.snapshot(kBin);
    CHECK(settled.actual == settled.target);
    CHECK(band_total(settled.actual, 0) == 36);
    CHECK(band_total(settled.actual, 1) == 36);
    return true;
}

bool test_sync_pending_free_and_recycle() {
    Controller controller("resident-sync-local", Mode::Adaptive, true);
    controller.configure(kRegionBytes);
    Counts initial{};
    initial[0] = 4;
    initial[1] = 4;
    initial[2] = 4;
    initial[3] = 4;
    initial[4] = 4;
    initial[5] = 4;
    seed(controller, initial);
    controller.begin();

    // Create a pending dirty-class transfer in the Streaming band, then move
    // one committed Region across R/S.  The target ledger remains conserved
    // and does not underflow despite the pending target.
    request(controller, 1, 100, 20);
    controller.tick(supply_for(0, 8), 1);
    request(controller, 1, 100, 20);
    controller.tick(supply_for(0, 8), 2);
    const Snapshot before_sync = controller.snapshot(kBin);
    CHECK(total(before_sync.actual) == 24 && total(before_sync.target) == 24);
    CHECK(controller.reclassify_supply(kBin, 1, 4) == 4);
    const Snapshot after_sync = controller.snapshot(kBin);
    CHECK(total(after_sync.actual) == 24 && total(after_sync.target) == 24);
    CHECK(after_sync.actual[1] + 1 == before_sync.actual[1]);
    CHECK(after_sync.actual[4] == before_sync.actual[4] + 1);
    CHECK(band_total(after_sync.actual, 0) == 11);
    CHECK(band_total(after_sync.actual, 1) == 13);

    // Freeing a committed Region unregisters one actual and one target owner;
    // it must not leave an orphaned target or change total conservation.
    CHECK(controller.release_supply(kBin, 4));
    const Snapshot after_release = controller.snapshot(kBin);
    CHECK(total(after_release.actual) == 23);
    CHECK(total(after_release.target) == 23);

    // Foreground recycle into a different actual placement band must not
    // preserve the old Streaming class.
    Controller recycle("resident-recycle-local", Mode::Adaptive, true);
    recycle.configure(kRegionBytes);
    CHECK(recycle.claim(kBin, 1) == 1);
    recycle.begin();
    CHECK(recycle.claim(kBin, 4, static_cast<int>(kBin), 1) == 4);
    const Snapshot recycled = recycle.snapshot(kBin);
    CHECK(recycled.actual[1] == 0 && recycled.actual[4] == 1);
    CHECK(recycled.target[1] == 0 && recycled.target[4] == 1);

    // A cross-bin recycle with a zero target in the old class still removes
    // ownership from that old placement band, never from the opposite band.
    Controller cross("resident-cross-bin", Mode::Adaptive, true);
    cross.configure(kRegionBytes);
    CHECK(cross.claim(kBin, 0) == 0);
    CHECK(cross.claim(kBin, 0) == 0);
    CHECK(cross.claim(kBin, 1) == 1);
    CHECK(cross.claim(kBin, 1) == 1);
    cross.begin();
    Supply cross_supply{};
    cross_supply[kBin][0] = 8 * kRegionBytes;
    request(cross, 1, 100, 20);
    cross.tick(cross_supply, 1);
    request(cross, 1, 100, 20);
    cross.tick(cross_supply, 2);
    CHECK(cross.reclassify_supply(kBin, 0, 3) == 3);
    const Snapshot pending_cross = cross.snapshot(kBin);
    CHECK(pending_cross.target[0] == 0);
    CHECK(cross.claim(kBin + 1, 4, static_cast<int>(kBin), 0) == 4);
    const Snapshot old_bin = cross.snapshot(kBin);
    const Snapshot new_bin = cross.snapshot(kBin + 1);
    CHECK(band_total(old_bin.actual, 0) == band_total(old_bin.target, 0));
    CHECK(band_total(old_bin.actual, 1) == band_total(old_bin.target, 1));
    CHECK(total(new_bin.actual) == 1 && total(new_bin.target) == 1);
    return true;
}

bool test_remote_controller_remains_six_way() {
    Controller remote("remote", Mode::Adaptive, false);
    remote.configure(kRegionBytes);
    Counts initial{};
    initial.fill(10);
    seed(remote, initial);
    remote.begin();
    const auto supply = supply_for(1, 8);
    request(remote, 4, 100, 20);
    remote.tick(supply, 1);
    request(remote, 4, 100, 20);
    remote.tick(supply, 2);
    const auto snapshot = remote.snapshot(kBin);
    CHECK(snapshot.target[4] > snapshot.actual[4]);
    CHECK(snapshot.target[1] < snapshot.actual[1]);
    CHECK(total(snapshot.actual) == 60 && total(snapshot.target) == 60);
    return true;
}

bool test_heat_and_supply_labels_are_separate() {
    FarLib::simple_region_budget::local().configure(kRegionBytes);
    std::array<std::uint8_t, kRegionBytes * 2> storage{};
    FarLib::simple_region_heat::Classes labels;
    const auto base = reinterpret_cast<std::uintptr_t>(storage.data());
    labels.configure(base, storage.size(), kRegionBytes);
    labels.set(base, FarLib::simple_region_heat::kCold);
    CHECK(labels.sync_supply_placement(base, kBin, true) == 4);
    CHECK(labels.get(base) == FarLib::simple_region_heat::kCold);
    CHECK(labels.allocation_class(base) == 4);
    CHECK(labels.release_supply(base));
    CHECK(labels.allocation_class(base) == FarLib::simple_region_heat::kCold);
    labels.set(base, FarLib::simple_region_heat::kHot);
    CHECK(labels.sync_supply_placement(base, kBin, false) == 1);
    CHECK(labels.get(base) == FarLib::simple_region_heat::kHot);
    CHECK(labels.allocation_class(base) == 1);
    CHECK(labels.release_supply(base));
    return true;
}

}  // namespace

int main() {
    if (!test_api_and_local_fallback() || !test_local_band_budget_isolation() ||
        !test_sync_pending_free_and_recycle() ||
        !test_remote_controller_remains_six_way() ||
        !test_heat_and_supply_labels_are_separate())
        return 1;
    std::cout << "SIMPLE_RESIDENT_BUDGET_TEST_OK\n";
    return 0;
}
