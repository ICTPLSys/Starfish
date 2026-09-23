#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace FarLib {

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
    };
    std::atomic<uint8_t> placement_flags{0};
    uint32_t logical_owner_id_{0};
    uint32_t resident_group_id_{0};
    mutable std::atomic<uint32_t> window_frequency{0};
    mutable std::atomic<uint32_t> ema_frequency{0};
    mutable std::atomic<uint32_t> published_window_frequency{0};
    mutable std::atomic<uint32_t> published_ema_frequency{0};

    static constexpr uint64_t MASK_HIGH = ((1L << 16) - 1) << 32;
    static constexpr uint64_t MASK_LOW = (1L << 32) - 1;
    static constexpr uint32_t OFFS_LOW = 32;

public:
    static constexpr uint64_t RemoteAddrInvalid48 = (1ULL << 48) - 1;
    static constexpr uint32_t FrequencyCounterMax =
        std::numeric_limits<uint32_t>::max();

    // tell poller which client to poll from
    inline uint8_t get_client_idx() const {
        return client_idx;
    }

    inline void set_client_idx(uint8_t idx) {
        client_idx = idx;
    }

    inline void *local_addr() const {
        return reinterpret_cast<void *>(((uint64_t)(local_addr_high) << 32) |
                                        local_addr_low);
    }

    void set_local_addr(void *p) {
        uint64_t local_addr = reinterpret_cast<uint64_t>(p);
        local_addr_low = local_addr & MASK_LOW;
        local_addr_high = local_addr >> OFFS_LOW;
    }

    inline size_t remote_addr() const {
        return (((uint64_t)remote_addr_high) << 32) | remote_addr_low;
    }

    inline bool remote_invalid() const { return remote_addr() == RemoteAddrInvalid48; }

    inline bool has_remote() const { return !remote_invalid(); }

    void set_remote_addr(size_t remote_addr) {
        remote_addr_low = remote_addr & MASK_LOW;
        remote_addr_high = remote_addr >> OFFS_LOW;
    }

    void set_remote_invalid() { set_remote_addr(RemoteAddrInvalid48); }

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
        return window_frequency.load(order);
    }

    uint32_t load_ema_frequency(
        std::memory_order order = std::memory_order::relaxed) const {
        return ema_frequency.load(order);
    }

    uint32_t load_published_window_frequency(
        std::memory_order order = std::memory_order::relaxed) const {
        return published_window_frequency.load(order);
    }

    uint32_t load_published_ema_frequency(
        std::memory_order order = std::memory_order::relaxed) const {
        return published_ema_frequency.load(order);
    }

    void add_window_frequency(uint32_t delta) const {
        if (delta == 0) {
            return;
        }
        auto current = window_frequency.load(std::memory_order_relaxed);
        while (true) {
            uint32_t next = current > FrequencyCounterMax - delta
                                ? FrequencyCounterMax
                                : current + delta;
            if (window_frequency.compare_exchange_weak(
                    current, next, std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                return;
            }
        }
    }

    bool try_add_window_frequency(uint32_t delta) const {
        if (delta == 0) {
            return true;
        }
        auto current = window_frequency.load(std::memory_order_relaxed);
        uint32_t next = current > FrequencyCounterMax - delta
                            ? FrequencyCounterMax
                            : current + delta;
        return window_frequency.compare_exchange_weak(
            current, next, std::memory_order_relaxed,
            std::memory_order_relaxed);
    }

    uint32_t consume_window_frequency() const {
        return window_frequency.exchange(0, std::memory_order_relaxed);
    }

    void update_ema_frequency(uint32_t observed_frequency,
                              size_t ema_decay_shift) const {
        auto current = ema_frequency.load(std::memory_order_relaxed);
        while (true) {
            uint32_t decay =
                ema_decay_shift == 0 ? 0u : (current >> ema_decay_shift);
            uint32_t next = current - decay;
            next = next > FrequencyCounterMax - observed_frequency
                       ? FrequencyCounterMax
                       : next + observed_frequency;
            if (ema_frequency.compare_exchange_weak(
                    current, next, std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                return;
            }
        }
    }

    void publish_frequency_profile(uint32_t window, uint32_t ema) const {
        published_window_frequency.store(window, std::memory_order_relaxed);
        published_ema_frequency.store(ema, std::memory_order_relaxed);
    }

    void copy_frequency_profile_from(const FarObjectEntry &other) {
        window_frequency.store(other.load_window_frequency(),
                               std::memory_order_relaxed);
        ema_frequency.store(other.load_ema_frequency(),
                            std::memory_order_relaxed);
        published_window_frequency.store(other.load_published_window_frequency(),
                                         std::memory_order_relaxed);
        published_ema_frequency.store(other.load_published_ema_frequency(),
                                      std::memory_order_relaxed);
    }

    void reset_frequency_profile() {
        window_frequency.store(0, std::memory_order_relaxed);
        ema_frequency.store(0, std::memory_order_relaxed);
        published_window_frequency.store(0, std::memory_order_relaxed);
        published_ema_frequency.store(0, std::memory_order_relaxed);
    }

public:
    EntryStateBits load_state(
        std::memory_order order = std::memory_order::seq_cst) const {
        return state.load(order);
    }

    void set_state(EntryStateBits bits) { state.store(bits); }

    bool cas_state_weak(EntryStateBits &expected, EntryStateBits desired) {
        return state.compare_exchange_weak(expected, desired);
    }

    bool cas_state_strong(EntryStateBits &expected, EntryStateBits desired) {
        return state.compare_exchange_strong(expected, desired);
    }

    bool is_local() const { return state.load().state <= EVICTING; }

    template <bool Lite>
    void reset(void *local_ptr, size_t remote_ptr, size_t size,
               bool dirty = false, bool resident_local = false,
               uint32_t logical_owner_id = 0,
               uint32_t resident_group_id = 0) {
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
        logical_owner_id_ = logical_owner_id;
        resident_group_id_ = resident_group_id;
        state.store(reset_state, std::memory_order::relaxed);
        set_local_addr(local_ptr);
        set_remote_addr(remote_ptr);
        reset_frequency_profile();
    }

    void set_free() {
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
        logical_owner_id_ = 0;
        resident_group_id_ = 0;
        set_local_addr(nullptr);
        set_remote_invalid();
        reset_frequency_profile();
    }

    void pin() {
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

// static_assert(sizeof(FarObjectEntry) == 16);

class DereferenceScope;

}  // namespace cache

}  // namespace FarLib
