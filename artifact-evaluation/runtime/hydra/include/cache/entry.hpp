#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cache/entry_frequency.hpp"
#include "cache/recompute_recipe.hpp"

namespace FarLib {
namespace allocator::six_group { struct Record; }

namespace cache {

enum EntryState {
    FREE,      // this entry is not used
    PINNED,    // the entry is pinned, do not evacuate
    LOCAL,     // the object is at local and not evicting
    MARKED,    // the object is marked to evict
    EVICTING,  // the object is evicting to remote
    REMOTE,    // the object is at remote, no local buffer used
    FETCHING,  // the object is being fetched
    BUSY,      // the entry is modifying, e.g. moving
};

struct EntryStateBits {
    uint32_t invalid : 1;
    uint32_t dirty : 1;
    EntryState state : 3;
    uint32_t hotness : 3;
    uint32_t ref_cnt : 8;
    uint32_t size : 16;

static constexpr uint32_t HOTNESS_MAX = 3;
    static constexpr uint32_t COLD = 0;
    static_assert(COLD < HOTNESS_MAX);

    void inc_hotness() { hotness = HOTNESS_MAX; }

    void dec_hotness() {
        if (hotness != 0) hotness--;
    }

    void inc_ref_cnt() { ref_cnt++; }

    void dec_ref_cnt() { ref_cnt--; }

    __attribute__((always_inline)) bool can_evict() const {
        return hotness == 0 && ref_cnt == 0 &&
               state < EVICTING && state != PINNED /* FREE, LOCAL or MARKED_CLEAN */;
    }

    template <bool Mut>
    __attribute__((always_inline)) bool is_deref_fast_path() const {
#ifdef ASSERT_ALL_LOCAL
        return true;
#else
        if (state == PINNED) {
            return true;
        }
        return (state == LOCAL) && (hotness > EntryStateBits::COLD) &&
               ((!Mut) || dirty);
#endif
    }
};

static_assert(sizeof(EntryStateBits) == sizeof(uint32_t));
static_assert(std::atomic<EntryStateBits>::is_always_lock_free);

// Native-only tail of a Hydra page descriptor. Many subobjects may be pinned
// at once; the existing 8-bit entry refcount holds ONE aggregate application
// pin, leaving its remaining values available for runtime I/O references.
struct HydraPagePins {
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    uint32_t count = 0;
};

class FarObjectEntry {
private:
    std::atomic<EntryStateBits> state;
    uint32_t local_addr_low;
    uint32_t remote_addr_low;
    uint16_t local_addr_high;
    uint16_t remote_addr_high;
    uint8_t client_idx = 0;
    enum PlacementFlag : uint8_t {
        ResidentLocal = 1u << 0,
        RemoteBackupReserved = 1u << 1,
        // Recovery owns bit 2.  Design-2 dirty evidence uses bits 3-7;
        // keeping all of these in one byte preserves resident snapshots.
        Recomputable = 1u << 2,
        SimpleDirtyScoreMask = 0x0fu << 3,
        SimpleDirtyScoreKnown = 1u << 7,
    };
    std::atomic<uint8_t> placement_flags{0};
    // Heat is a routing hint, not placement identity.  Keep it in the one-byte
    // alignment gap so Recovery's placement bit 2 remains Recomputable.
    enum RoutingFlag : uint8_t {
        SimpleHeatHot = 1u << 0,
        HydraSubobject = 1u << 1,
        HydraPage = 1u << 2,
    };
    std::atomic<uint8_t> routing_flags_{0};
    uint32_t logical_owner_id_{0};
    uint32_t resident_group_id_{0};
    std::atomic<uint8_t> six_pending_evict_{0};
    bool six_remote_committed_{false};
    std::atomic<uint16_t> object_trace_slot_{0};
    // Zero means unsampled; the stable handle follows the logical object
    // across entries.
    recompute::Binding recompute_binding_{};

    static constexpr uint64_t MASK_HIGH = ((1ULL << 16) - 1) << 32;
    static constexpr uint64_t MASK_LOW = (1ULL << 32) - 1;
    static constexpr uint32_t OFFS_LOW = 32;

public:
    FarObjectEntry() noexcept : recompute_binding_{} { set_free(); }

    // A packed handle uses its otherwise-unused address words for page owner
    // and offset. Only the page owner participates in the runtime state machine.
    // Keep the 48-byte layout shared by all consumers of this runtime.
    bool hydra_packed() const {
        return (routing_flags_.load(std::memory_order_acquire) & HydraSubobject) != 0;
    }
    bool hydra_page() const {
        return (routing_flags_.load(std::memory_order_relaxed) & HydraPage) != 0;
    }
    void hydra_mark_page() { update_routing_flag(HydraPage, true); }
    // Publish a Hydra page fetch only after its local buffer and completion
    // client have been initialized.  A concurrent page-level pin may have
    // changed ref_cnt while the owner was BUSY, so do not store a stale
    // snapshot here; reload and CAS the current state while changing only the
    // state tag.  The release publication pairs with waiters that observe
    // FETCHING and then read the published address/client fields.
    bool hydra_publish_fetching() {
        assert(hydra_page());
        auto observed = state.load(std::memory_order_acquire);
        for (;;) {
            if (observed.state != BUSY) return false;
            auto desired = observed;
            desired.state = FETCHING;
            if (state.compare_exchange_weak(
                    observed, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
    }
    void hydra_pin_page(bool acquire);
    FarObjectEntry *hydra_owner() {
        return const_cast<FarObjectEntry *>(static_cast<const FarObjectEntry &>(*this).hydra_owner());
    }
    const FarObjectEntry *hydra_owner() const {
        if (!hydra_packed()) return this;
        return reinterpret_cast<const FarObjectEntry *>(
            (static_cast<uint64_t>(local_addr_high) << 32) | local_addr_low);
    }
    uint16_t hydra_offset() const {
        return hydra_packed() ? static_cast<uint16_t>(remote_addr_low) : 0;
    }
    // Application-visible length is independent of the page state/size.
    uint16_t object_size() const { return state.load(std::memory_order_relaxed).size; }
    void hydra_attach(FarObjectEntry *owner, uint16_t offset, uint16_t size) {
        assert(owner != nullptr && owner != this && !owner->hydra_packed());
        assert(size != 0 && static_cast<uint32_t>(offset) + size <= 8192);
        set_free();
        const auto address = reinterpret_cast<uint64_t>(owner);
        local_addr_low = static_cast<uint32_t>(address);
        local_addr_high = static_cast<uint16_t>(address >> 32);
        remote_addr_low = offset;
        remote_addr_high = 0;
        EntryStateBits handle_state{};
        handle_state.state = LOCAL;
        handle_state.size = size;
        state.store(handle_state, std::memory_order_relaxed);
        routing_flags_.store(HydraSubobject, std::memory_order_release);
    }
    void hydra_take_handle(FarObjectEntry &other) {
        assert(other.hydra_packed());
        hydra_attach(other.hydra_owner(), other.hydra_offset(),
                     static_cast<uint16_t>(other.state.load().size));
        other.set_free();
    }

    ~FarObjectEntry() {
        release_recompute_recipe();
        detail::EntrySixBindingStore::erase(this);
        reset_frequency_profile();
    }

    FarObjectEntry(const FarObjectEntry &) = delete;
    FarObjectEntry &operator=(const FarObjectEntry &) = delete;

    // One routing hint in an existing flag byte, not an object access counter.
    // Written under eviction move-lock; safe to read before the fetch CAS.
    bool simple_heat_hot() const {
        return (routing_flags_.load(std::memory_order_relaxed) & SimpleHeatHot) != 0;
    }
    void set_simple_heat_hot(bool hot) { update_routing_flag(SimpleHeatHot, hot); }

    // The dirty dimension is a committed logical-eviction EWMA packed into
    // the spare placement-flag bits.  Zero is the Low prior: the first dirty
    // eviction reaches q=4 (still Low), without a first-eviction special case.
    static constexpr uint8_t SimpleDirtyLow = 0;
    static constexpr uint8_t SimpleDirtyMedium = 1;
    static constexpr uint8_t SimpleDirtyHigh = 2;
    static constexpr uint8_t SimpleDirtyUnknownQ = 0;
    static constexpr uint8_t SimpleDirtyScoreShift = 3;

    uint8_t simple_dirty_score_q() const {
        const uint8_t flags = placement_flags.load(std::memory_order_acquire);
        if ((flags & SimpleDirtyScoreKnown) == 0) return SimpleDirtyUnknownQ;
        return static_cast<uint8_t>((flags & SimpleDirtyScoreMask) >>
                                    SimpleDirtyScoreShift);
    }

    bool simple_dirty_score_known() const {
        return (placement_flags.load(std::memory_order_acquire) &
                SimpleDirtyScoreKnown) != 0;
    }

    static constexpr uint8_t simple_dirty_class_for_q(uint8_t q) {
        if (q <= 4) return SimpleDirtyLow;
        if (q >= 11) return SimpleDirtyHigh;
        return SimpleDirtyMedium;
    }

    uint8_t simple_dirty_class() const {
        // Unknown scores use the same Low prior as the first EWMA update.
        // Keep the known bit clear until an eviction actually commits.
        return simple_dirty_class_for_q(simple_dirty_score_q());
    }

    // Record exactly one committed logical eviction.  This deliberately
    // updates only the dirty score; heat, resident, and backup flag bits are
    // preserved by the atomic read/modify/write.
    void note_simple_dirty_eviction(bool dirty) {
        uint8_t current = placement_flags.load(std::memory_order_relaxed);
        for (;;) {
            const uint8_t q = (current & SimpleDirtyScoreKnown) == 0
                                  ? SimpleDirtyUnknownQ
                                  : static_cast<uint8_t>(
                                        (current & SimpleDirtyScoreMask) >>
                                        SimpleDirtyScoreShift);
            const uint8_t next_q = static_cast<uint8_t>(
                (3u * q + (dirty ? 15u : 0u) + 2u) / 4u);
            const uint8_t desired = static_cast<uint8_t>(
                (current & static_cast<uint8_t>(
                               ~(SimpleDirtyScoreMask | SimpleDirtyScoreKnown))) |
                static_cast<uint8_t>(next_q << SimpleDirtyScoreShift) |
                SimpleDirtyScoreKnown);
            if (placement_flags.compare_exchange_weak(
                    current, desired, std::memory_order_acq_rel,
                    std::memory_order_relaxed))
                return;
        }
    }
    uint16_t object_trace_slot() const {
        return object_trace_slot_.load(std::memory_order_relaxed);
    }
    void set_object_trace_slot(uint16_t slot) {
        object_trace_slot_.store(slot, std::memory_order_relaxed);
    }
    // Compatibility reference for the legacy fixed-six registry.  The
    // semantic-six path does not call this method and therefore never creates
    // the sidecar or takes its shard mutex.
    auto& six_binding() { return detail::EntrySixBindingStore::binding(this); }
    allocator::six_group::Record* six_record() const {
        return detail::EntrySixBindingStore::load(this);
    }
    void set_six_pending_evict(bool dirty) {
        six_pending_evict_.store(dirty ? 2 : 1, std::memory_order_release);
    }
    uint8_t take_six_pending_evict() {
        return six_pending_evict_.exchange(0, std::memory_order_acq_rel);
    }
    bool six_first_remote_commit() {
        const bool first = !six_remote_committed_;
        six_remote_committed_ = true;
        return first;
    }
    // Entry relocation changes no physical object or Region occupancy.
    void take_six_metadata_from(FarObjectEntry& other) {
        // The sidecar is keyed by Entry address; move the non-owning Record
        // pointer before the source address is recycled.
        auto *binding = detail::EntrySixBindingStore::exchange(&other, nullptr);
        detail::EntrySixBindingStore::erase(this);
        if (binding != nullptr)
            detail::EntrySixBindingStore::store(this, binding);
        // Accessor relocation uses this routine as the common metadata move;
        // heat must follow the logical object even though it no longer shares
        // the Recovery placement byte.
        routing_flags_.store(other.routing_flags_.exchange(0),
                             std::memory_order_release);
        six_pending_evict_.store(other.six_pending_evict_.exchange(0),
                                 std::memory_order_release);
        six_remote_committed_ = other.six_remote_committed_;
        other.six_remote_committed_ = false;
        object_trace_slot_.store(other.object_trace_slot_.exchange(0),
                                 std::memory_order_release);
    }
    static constexpr uint64_t RemoteAddrInvalid48 = (1ULL << 48) - 1;
    static constexpr uint32_t FrequencyCounterMax =
        std::numeric_limits<uint32_t>::max();

    // tell poller which client to poll from
    inline uint8_t get_client_idx() const {
        if (hydra_packed()) return hydra_owner()->get_client_idx();
        return client_idx;
    }

    inline void set_client_idx(uint8_t idx) {
        client_idx = idx;
    }

    inline void *local_addr() const {
        if (hydra_packed()) {
            void *base = hydra_owner()->local_addr();
            return base == nullptr ? nullptr
                : static_cast<void *>(static_cast<char *>(base) + hydra_offset());
        }
        return reinterpret_cast<void *>(((uint64_t)(local_addr_high) << 32) |
                                        local_addr_low);
    }

    void set_local_addr(void *p) {
        uint64_t local_addr = reinterpret_cast<uint64_t>(p);
        local_addr_low = local_addr & MASK_LOW;
        local_addr_high = local_addr >> OFFS_LOW;
    }

    inline size_t remote_addr() const {
        if (hydra_packed()) return hydra_owner()->remote_addr();
        return (((uint64_t)remote_addr_high) << 32) | remote_addr_low;
    }

    inline bool remote_invalid() const { return remote_addr() == RemoteAddrInvalid48; }

    inline bool has_remote() const { return !remote_invalid(); }

    void set_remote_addr(size_t remote_addr) {
        remote_addr_low = remote_addr & MASK_LOW;
        remote_addr_high = remote_addr >> OFFS_LOW;
    }

    void set_remote_invalid() {
        set_remote_addr(RemoteAddrInvalid48);
        six_remote_committed_ = false;
    }

    bool is_resident_local() const {
        return (placement_flags.load(std::memory_order_acquire) &
                ResidentLocal) != 0;
    }

    bool has_remote_backup_reservation() const {
        return (placement_flags.load(std::memory_order_acquire) &
                RemoteBackupReserved) != 0;
    }

    void set_resident_local(bool resident) {
        update_placement_flag(ResidentLocal, resident);
    }

    void set_remote_backup_reservation(bool reserved) {
        update_placement_flag(RemoteBackupReserved, reserved);
    }

    bool is_recomputable() const {
        return (placement_flags.load(std::memory_order_acquire) &
                Recomputable) != 0;
    }

    bool has_recompute_recipe() const {
        return recompute_binding_.inputs != nullptr;
    }

    recompute::Binding recompute_binding() const {
        return recompute_binding_;
    }

    // Bind one immutable input set and callback before the entry has acquired
    // its first remote address. The entry invalid bit is the existing per-entry
    // move lock used to serialize this publication with eviction/moves.
    bool try_bind_recompute_recipe(const recompute::Inputs &inputs,
                                   recompute::EntryFn callback,
                                   uint64_t arg) {
        // A subobject recipe cannot describe the other live slots in its page.
        if (hydra_packed()) return false;
        if (!inputs.valid() || callback == nullptr) return false;
        recompute::InputState *input_state = inputs.input_state();
        auto current = load_state();
        for (;;) {
            if (current.invalid) return false;
            if (current.state != LOCAL && current.state != MARKED &&
                current.state != PINNED) {
                return false;
            }

            auto locked = current;
            locked.invalid = 1;
            if (!cas_state_weak(current, locked)) continue;

            bool accepted = false;
            if (remote_invalid() && !has_recompute_recipe() &&
                input_state->register_callback(callback)) {
                input_state->retain();
                recompute_binding_ = {input_state, arg};
                set_recomputable();
                accepted = true;
            }

            auto unlocked = locked;
            unlocked.invalid = 0;
            if (accepted && unlocked.state == MARKED) {
                unlocked.state = LOCAL;
                unlocked.inc_hotness();
            }
            const bool restored = cas_state_strong(locked, unlocked);
            assert(restored);
            (void)restored;
            return accepted;
        }
    }

    // Drop the entry's intrusive InputState reference. The placement flag is
    // intentionally left alone: reset/free clear flags as a separate step.
    void release_recompute_recipe() noexcept {
        recompute::InputState *input_state = recompute_binding_.inputs;
        recompute_binding_ = {};
        if (input_state != nullptr) input_state->release();
    }

    void clear_recompute_metadata() noexcept {
        release_recompute_recipe();
        update_placement_flag(Recomputable, false);
    }

    // Move both the recipe reference and Recovery's placement bit. The
    // reference is transferred, rather than retained/released, before the
    // source entry is cleared by an accessor move/free path.
    void take_recompute_recipe_from(FarObjectEntry &source) noexcept {
        if (this == &source) return;
        release_recompute_recipe();
        recompute_binding_ = source.recompute_binding_;
        source.recompute_binding_ = {};
        const bool source_recomputable = source.is_recomputable();
        update_placement_flag(Recomputable, source_recomputable);
        source.update_placement_flag(Recomputable, false);
    }

    // Publication is serialized by ConcurrentArrayCache's recovery entry
    // operation. Recovery reserves placement_flags bit 2 for this identity.
    void set_recomputable() {
        update_placement_flag(Recomputable, true);
    }

    bool try_mark_recomputable_before_eviction(bool *newly_marked = nullptr) {
        if (newly_marked != nullptr) *newly_marked = false;
        if (hydra_packed()) return false;
        auto current = load_state();
        for (;;) {
            if (current.state == FREE || current.state == BUSY) return false;
            if (is_recomputable()) return true;
            if (current.invalid || (current.state != LOCAL &&
                current.state != MARKED && current.state != PINNED)) {
                return false;
            }
            auto locked = current;
            locked.invalid = 1;
            if (!cas_state_weak(current, locked)) continue;
            const bool accepted = remote_invalid();
            if (accepted) set_recomputable();
            auto unlocked = locked;
            unlocked.invalid = 0;
            if (accepted && unlocked.state == MARKED) {
                unlocked.state = LOCAL;
                unlocked.inc_hotness();
            }
            const bool restored = cas_state_strong(locked, unlocked);
            assert(restored);
            (void)restored;
            if (newly_marked != nullptr) *newly_marked = accepted;
            return accepted;
        }
    }

    uint8_t load_placement_flags() const {
        return placement_flags.load(std::memory_order_acquire);
    }

    void store_placement_flags(uint8_t flags) {
        placement_flags.store(flags, std::memory_order_release);
    }

    uint32_t logical_owner_id() const { return logical_owner_id_; }

    void set_logical_owner_id(uint32_t owner_id) {
        logical_owner_id_ = owner_id;
    }
    uint32_t resident_group_id() const { return resident_group_id_; }
    void set_resident_group_id(uint32_t group_id) {
        resident_group_id_ = group_id;
    }

private:
    void update_placement_flag(uint8_t flag, bool enabled) {
        uint8_t current = placement_flags.load(std::memory_order_relaxed);
        uint8_t desired;
        do {
            desired = enabled ? static_cast<uint8_t>(current | flag)
                              : static_cast<uint8_t>(current & ~flag);
        } while (!placement_flags.compare_exchange_weak(
            current, desired, std::memory_order_acq_rel,
            std::memory_order_relaxed));
    }

    void update_routing_flag(uint8_t flag, bool enabled) {
        uint8_t current = routing_flags_.load(std::memory_order_relaxed);
        uint8_t desired;
        do {
            desired = enabled ? static_cast<uint8_t>(current | flag)
                              : static_cast<uint8_t>(current & ~flag);
        } while (!routing_flags_.compare_exchange_weak(
            current, desired, std::memory_order_acq_rel,
            std::memory_order_relaxed));
    }

public:

    void set_pinned(bool pinned = false) {
        EntryStateBits old_state = load_state();
        EntryStateBits new_state = old_state;
        new_state.state = pinned ? PINNED : LOCAL;
        while (!cas_state_weak(old_state, new_state)) {
            old_state = load_state();
            new_state = old_state;  
            new_state.state = pinned ? PINNED : LOCAL;
        }
    }

    bool is_pinned() const {
        return load_state().state == PINNED;
    }

    uint32_t load_window_frequency(
        std::memory_order order = std::memory_order::relaxed) const {
        (void)order;
        return detail::EntryFrequencyStore::load_window(this);
    }

    uint32_t load_ema_frequency(
        std::memory_order order = std::memory_order::relaxed) const {
        (void)order;
        return detail::EntryFrequencyStore::load_ema(this);
    }

    uint32_t load_published_window_frequency(
        std::memory_order order = std::memory_order::relaxed) const {
        (void)order;
        return detail::EntryFrequencyStore::load_published_window(this);
    }

    uint32_t load_published_ema_frequency(
        std::memory_order order = std::memory_order::relaxed) const {
        (void)order;
        return detail::EntryFrequencyStore::load_published_ema(this);
    }

    void add_window_frequency(uint32_t delta) const {
        detail::EntryFrequencyStore::add_window(this, delta);
    }

    bool try_add_window_frequency(uint32_t delta) const {
        return detail::EntryFrequencyStore::try_add_window(this, delta);
    }

    uint32_t consume_window_frequency() const {
        return detail::EntryFrequencyStore::consume_window(this);
    }

    void update_ema_frequency(uint32_t observed_frequency,
                              size_t ema_decay_shift) const {
        detail::EntryFrequencyStore::update_ema(
            this, observed_frequency, static_cast<uint32_t>(ema_decay_shift));
    }

    void publish_frequency_profile(uint32_t window, uint32_t ema) const {
        detail::EntryFrequencyStore::publish(this, window, ema);
    }

    void copy_frequency_profile_from(const FarObjectEntry &other) {
        detail::EntryFrequencyStore::copy(&other, this);
    }

    void reset_frequency_profile() {
        detail::EntryFrequencyStore::erase(this);
    }

public:
    EntryStateBits load_state(
        std::memory_order order = std::memory_order::seq_cst) const {
        if (hydra_packed()) return hydra_owner()->load_state(order);
        return state.load(order);
    }

    void set_state(EntryStateBits bits) { state.store(bits); }

    bool cas_state_weak(EntryStateBits &expected, EntryStateBits desired) {
        if (hydra_packed()) return hydra_owner()->cas_state_weak(expected, desired);
        return state.compare_exchange_weak(expected, desired);
    }

    bool cas_state_strong(EntryStateBits &expected, EntryStateBits desired) {
        if (hydra_packed()) return hydra_owner()->cas_state_strong(expected, desired);
        return state.compare_exchange_strong(expected, desired);
    }

    bool is_local() const { return load_state().state <= EVICTING; }

    template <bool Lite>
    void reset(void *local_ptr, size_t remote_ptr, size_t size,
               bool dirty = false, bool resident_local = false,
               uint32_t logical_owner_id = 0,
               uint32_t resident_group_id = 0) {
        release_recompute_recipe();
        EntryStateBits reset_state = {
            .invalid = 0,
            .dirty = dirty ? 1u : 0u,
            .state = LOCAL,
            .hotness = 1,
            .ref_cnt = Lite ? 0 : 1,
            .size = static_cast<uint32_t>(size),
        };
        placement_flags.store(resident_local ? ResidentLocal : 0,
                              std::memory_order_release);
        routing_flags_.store(0, std::memory_order_release);
        logical_owner_id_ = logical_owner_id;
        resident_group_id_ = resident_group_id;
        detail::EntrySixBindingStore::erase(this);
        six_pending_evict_.store(0, std::memory_order_relaxed);
        six_remote_committed_ = false;
        state.store(reset_state, std::memory_order::relaxed);
        object_trace_slot_.store(0, std::memory_order_relaxed);
        set_local_addr(local_ptr);
        set_remote_addr(remote_ptr);
        reset_frequency_profile();
        detail::EntryFrequencyStore::register_entry(this);
    }

    void set_free() {
        release_recompute_recipe();
        EntryStateBits free_state = {
            .invalid = 0,
            .dirty = 0,
            .state = FREE,
            .hotness = 0,
            .ref_cnt = 0,
            .size = 0,
        };
        state.store(free_state);
        placement_flags.store(0, std::memory_order_release);
        routing_flags_.store(0, std::memory_order_release);
        logical_owner_id_ = 0;
        resident_group_id_ = 0;
        assert(detail::EntrySixBindingStore::load(this) == nullptr);
        detail::EntrySixBindingStore::erase(this);
        six_pending_evict_.store(0, std::memory_order_relaxed);
        set_local_addr(nullptr);
        object_trace_slot_.store(0, std::memory_order_relaxed);
        set_remote_invalid();
        reset_frequency_profile();
    }

    void pin() {
        if (hydra_packed()) { hydra_owner()->pin(); return; }
        if (hydra_page()) { hydra_pin_page(true); return; }
        auto old_state = load_state();
        assert(old_state.state != MARKED);
        assert(old_state.state != EVICTING);
        assert(old_state.state != REMOTE);
    retry:
        auto new_state = old_state;
        new_state.inc_ref_cnt();
        if (new_state.state == MARKED || new_state.state == EVICTING) {
            new_state.state = LOCAL;
        }
        if (!cas_state_weak(old_state, new_state)) goto retry;
    }

    void unpin() {
        if (hydra_packed()) { hydra_owner()->unpin(); return; }
        if (hydra_page()) { hydra_pin_page(false); return; }
        auto old_state = load_state();
        assert(old_state.state != MARKED);
        assert(old_state.state != EVICTING);
        assert(old_state.state != REMOTE);
    retry:
        if (old_state.ref_cnt == 0) {
            return;
        }
        auto new_state = old_state;
        new_state.dec_ref_cnt();
        if (!cas_state_weak(old_state, new_state)) goto retry;
    }

    void mark_dirty() {
        EntryStateBits old_state = load_state();
        if (old_state.dirty) return;
        EntryStateBits new_state;
        do {
            new_state = old_state;
            new_state.dirty = 1;
        } while (!cas_state_weak(old_state, new_state));
    }

    friend class ConcurrentArrayCache;
};

// Recovery's ABI-critical Entry remains 48B: the legacy six binding is lazy
// sidecar state, while D2's routing/pending/trace metadata occupies the
// otherwise unused bytes before the 16B recompute Binding.
static_assert(sizeof(FarObjectEntry) == 48);

inline void FarObjectEntry::hydra_pin_page(bool acquire) {
    auto &pins = *reinterpret_cast<HydraPagePins *>(
        reinterpret_cast<char *>(this) + sizeof(FarObjectEntry));
    while (pins.lock.test_and_set(std::memory_order_acquire)) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    if (acquire) {
        assert(pins.count != std::numeric_limits<uint32_t>::max());
        if (pins.count++ == 0) {
            auto before = state.load();
            for (;;) {
                if (before.invalid) { before = state.load(); continue; }
                assert(before.state != REMOTE && before.state != FREE);
                auto after = before;
                after.inc_ref_cnt();
                if (after.state == MARKED || after.state == EVICTING)
                    after.state = LOCAL;
                if (state.compare_exchange_weak(before, after)) break;
            }
        }
    } else {
        assert(pins.count != 0);
        if (--pins.count == 0) {
            auto before = state.load();
            for (;;) {
                if (before.invalid) { before = state.load(); continue; }
                assert(before.ref_cnt != 0);
                auto after = before;
                after.dec_ref_cnt();
                if (state.compare_exchange_weak(before, after)) break;
            }
        }
    }
    pins.lock.clear(std::memory_order_release);
}

class DereferenceScope;

}  // namespace cache

}  // namespace FarLib
