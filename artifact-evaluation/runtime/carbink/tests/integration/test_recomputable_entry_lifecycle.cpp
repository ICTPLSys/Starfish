#include "cache/entry.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

namespace {

struct Context {
    explicit Context(int value) : value(value) {}
    ~Context() { ++destroyed; }
    int value;
    static int destroyed;
};

int Context::destroyed = 0;

bool rebuild(void *destination, size_t bytes, const void *opaque, uint64_t arg) {
    if (destination == nullptr || opaque == nullptr || bytes != sizeof(int)) {
        return false;
    }
    *static_cast<int *>(destination) =
        static_cast<const Context *>(opaque)->value + static_cast<int>(arg);
    return true;
}

}  // namespace

int main() {
    using FarLib::cache::detail::EntryFrequencyStore;
    using FarLib::cache::FarObjectEntry;
    using FarLib::cache::LOCAL;

    // Recovery's 48B Entry keeps Recomputable in placement bit 2.  D2 heat
    // uses the independent routing byte; frequency remains external.
    static_assert(sizeof(FarObjectEntry) == 48);

    EntryFrequencyStore::set_enabled(true);
    Context::destroyed = 0;

    auto context = std::make_shared<Context>(40);
    FarLib::recompute::Inputs inputs =
        FarLib::recompute::Inputs::immutable_compute_owned(context);

    FarObjectEntry source;
    source.reset<true>(reinterpret_cast<void *>(0x1000),
                       FarObjectEntry::RemoteAddrInvalid48, sizeof(int));
    source.set_simple_heat_hot(true);
    source.mark_dirty();
    source.set_resident_local(true);
    source.set_remote_backup_reservation(true);
    source.note_simple_dirty_eviction(true);
    source.note_simple_dirty_eviction(true);
    assert(source.simple_dirty_score_known());
    assert(source.simple_dirty_score_q() == 7);
    source.add_window_frequency(3);
    source.update_ema_frequency(5, 2);
    source.publish_frequency_profile(3, 5);

    assert(source.try_bind_recompute_recipe(inputs, rebuild, 2));
    const uint8_t placement_snapshot = source.load_placement_flags();
    assert(source.is_recomputable());
    assert((source.load_placement_flags() & (1u << 2)) != 0);
    assert(source.has_recompute_recipe());
    assert(source.simple_heat_hot());
    assert(source.load_state().dirty);
    assert(source.is_resident_local());
    assert(source.has_remote_backup_reservation());
    assert(source.simple_dirty_score_q() == 7);
    assert(source.simple_dirty_score_known());
    // Resident exchange restores the complete Recovery placement byte, which
    // includes Recomputable bit 2; routing heat remains independent.
    source.store_placement_flags(placement_snapshot);
    assert(source.is_recomputable() && source.has_recompute_recipe());
    assert(source.load_window_frequency() == 3);
    assert(source.load_ema_frequency() == 5);

    int value = 0;
    auto binding = source.recompute_binding();
    assert(binding.inputs->invoke(&value, sizeof(value), binding.arg));
    assert(value == 42);

    FarObjectEntry moved;
    moved.take_recompute_recipe_from(source);
    assert(!source.is_recomputable() && !source.has_recompute_recipe());
    assert(moved.is_recomputable() && moved.has_recompute_recipe());
    assert(source.simple_heat_hot());
    assert(source.load_state().dirty);
    assert(source.is_resident_local());
    assert(source.has_remote_backup_reservation());

    context.reset();
    inputs.reset();
    assert(Context::destroyed == 0);
    moved.set_free();
    assert(!moved.is_recomputable() && !moved.has_recompute_recipe());
    assert(Context::destroyed == 1);
    source.set_free();

    FarObjectEntry flag_only;
    flag_only.reset<true>(reinterpret_cast<void *>(0x2000),
                          FarObjectEntry::RemoteAddrInvalid48, sizeof(int));
    auto state = flag_only.load_state();
    state.state = LOCAL;
    flag_only.set_state(state);
    bool newly_marked = false;
    assert(flag_only.try_mark_recomputable_before_eviction(&newly_marked));
    assert(newly_marked && flag_only.is_recomputable());
    FarObjectEntry flag_moved;
    flag_moved.take_recompute_recipe_from(flag_only);
    assert(!flag_only.is_recomputable());
    assert(flag_moved.is_recomputable() && !flag_moved.has_recompute_recipe());
    flag_moved.clear_recompute_metadata();
    assert(!flag_moved.is_recomputable());
    flag_only.set_free();
    assert(!flag_only.is_recomputable());

    // Legacy fixed-six compatibility uses a lazy, address-keyed sidecar.  It
    // must transfer the non-owning Record pointer with Entry relocation and
    // leave no map node after the destination is freed.
    using SixBindingStore = FarLib::cache::detail::EntrySixBindingStore;
    using SixRecord = FarLib::allocator::six_group::Record;
    FarObjectEntry six_source;
    FarObjectEntry six_destination;
    six_source.reset<true>(reinterpret_cast<void *>(0x3000),
                           FarObjectEntry::RemoteAddrInvalid48, sizeof(int));
    six_destination.reset<true>(reinterpret_cast<void *>(0x4000),
                                FarObjectEntry::RemoteAddrInvalid48,
                                sizeof(int));
    auto *fake_record = reinterpret_cast<SixRecord *>(0x12345000);
    six_source.six_binding().store(fake_record);
    six_source.set_simple_heat_hot(true);
    six_source.set_six_pending_evict(true);
    assert(six_source.six_record() == fake_record);
    six_destination.take_six_metadata_from(six_source);
    assert(six_source.six_record() == nullptr);
    assert(six_destination.six_record() == fake_record);
    assert(!six_source.simple_heat_hot());
    assert(six_destination.simple_heat_hot());
    assert(six_destination.take_six_pending_evict() == 2);
    six_destination.six_binding().store(nullptr);
    six_destination.set_free();
    six_source.set_free();
    assert(SixBindingStore::live_records() == 0);

    assert(EntryFrequencyStore::live_records() == 0);
    EntryFrequencyStore::set_enabled(false);
    std::puts("RECOMPUTABLE_DESIGN2_ENTRY_PASS size=48");
    return 0;
}
