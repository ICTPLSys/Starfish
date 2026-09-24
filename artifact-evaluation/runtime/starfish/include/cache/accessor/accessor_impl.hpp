#pragma once
#include "cache/accessor.hpp"

namespace FarLib::cache {

template <typename T, typename Impl>
inline void UniqueFarPtrBase<T, Impl>::move(FarObjectEntry *other) {
    auto *cache = Cache::get_default();
    EntryStateBits state = other->load_state();
retry:
    if (::FarLib::get_config().exclusive_cache && state.invalid) {
        state = other->load_state();
        goto retry;
    }
    switch (state.state) {
    case FREE:
        entry.set_free();
        return;
    case REMOTE:
        entry.store_placement_flags(other->load_placement_flags());
        entry.set_logical_owner_id(other->logical_owner_id());
        entry.set_resident_group_id(other->resident_group_id());
        entry.set_state(state);
        entry.set_local_addr(nullptr);
        entry.set_remote_addr(other->remote_addr());
        entry.copy_frequency_profile_from(*other);
        entry.take_six_metadata_from(*other);
        break;
    case LOCAL:
    case PINNED:
    case FETCHING:
    case MARKED:
    case EVICTING: {
        EntryStateBits pin_state = state;
        if (state.state != FETCHING) {
            pin_state.state = LOCAL;
        }
        pin_state.ref_cnt++;
        if (!other->cas_state_weak(state, pin_state)) [[unlikely]] {
            goto retry;
        }
        void *local_addr = other->local_addr();
        EntryStateBits unpin_state = pin_state;
        unpin_state.ref_cnt--;
        entry.store_placement_flags(other->load_placement_flags());
        entry.set_logical_owner_id(other->logical_owner_id());
        entry.set_resident_group_id(other->resident_group_id());
        entry.set_state(unpin_state);
        entry.set_local_addr(local_addr);
        entry.set_remote_addr(other->remote_addr());
        entry.copy_frequency_profile_from(*other);
        entry.take_six_metadata_from(*other);
        auto block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        far_obj_t obj = block->obj_meta_data.load(std::memory_order::relaxed);
        obj.obj_id = reinterpret_cast<uint64_t>(this);
        block->obj_meta_data.store(obj, std::memory_order::relaxed);
        break;
    }
    case BUSY:
    default:
        ERROR("invalid state");
    }
    if (cache != nullptr) {
        cache->move_live_entry(other, &entry);
    }
    other->set_free();
}

template <typename T, typename Impl>
inline bool UniqueFarPtrBase<T, Impl>::cas_null_to_busy() {
    EntryStateBits prev_state = entry.load_state(std::memory_order::relaxed);
    if (prev_state.state != FREE) return false;
    EntryStateBits busy_state = {.dirty = 0, .state = BUSY, .hotness = 0, .ref_cnt = 0};
    return entry.cas_state_strong(prev_state, busy_state);
}

template <typename T, typename Impl>
inline void UniqueFarPtrBase<T, Impl>::atomic_move_to_and_set_busy(
    UniqueFarPtrBase<T, Impl> &to) {
    auto *cache = Cache::get_default();
    assert(to.entry.load_state(std::memory_order::relaxed).state == BUSY);
retry:
    EntryStateBits prev_state = entry.load_state(std::memory_order::relaxed);
    EntryStateBits busy_state = {.dirty = 0, .state = BUSY, .hotness = 0, .ref_cnt = 0};
    if (!entry.cas_state_weak(prev_state, busy_state)) goto retry;

    switch (prev_state.state) {
    case FREE:
        to.entry.set_free();
        return;
    case REMOTE:
        to.entry.store_placement_flags(entry.load_placement_flags());
        to.entry.set_logical_owner_id(entry.logical_owner_id());
        to.entry.set_resident_group_id(entry.resident_group_id());
        to.entry.set_state(prev_state);
        to.entry.set_local_addr(nullptr);
        to.entry.set_remote_addr(entry.remote_addr());
        to.entry.copy_frequency_profile_from(entry);
        to.entry.take_six_metadata_from(entry);
        break;
    case LOCAL:
    case PINNED:
    case FETCHING:
    case MARKED:
    case EVICTING: {
        void *local_addr = entry.local_addr();
        to.entry.store_placement_flags(entry.load_placement_flags());
        to.entry.set_logical_owner_id(entry.logical_owner_id());
        to.entry.set_resident_group_id(entry.resident_group_id());
        to.entry.set_state(prev_state);
        to.entry.set_local_addr(local_addr);
        to.entry.set_remote_addr(entry.remote_addr());
        to.entry.copy_frequency_profile_from(entry);
        to.entry.take_six_metadata_from(entry);
        auto block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        far_obj_t obj = block->obj_meta_data.load(std::memory_order::relaxed);
        obj.obj_id = reinterpret_cast<uint64_t>(&to);
        block->obj_meta_data.store(obj, std::memory_order::relaxed);
        break;
    }
    case BUSY:
    default:
        ERROR("invalid state");
    }
    if (cache != nullptr) {
        cache->move_live_entry(&entry, &to.entry);
    }
    entry.set_local_addr(nullptr);
    entry.set_remote_invalid();
    entry.store_placement_flags(0);
    entry.set_logical_owner_id(0);
    entry.set_resident_group_id(0);
    entry.reset_frequency_profile();
}

template <typename T>
template <typename... Args>
inline LiteAccessor<T, true> UniqueFarPtr<T>::allocate_lite(
    DereferenceScope &scope, Args &&...args) {
    this->reset();
    auto cache = Cache::get_default();
    cache->template allocate<true>(&entry, sizeof(T), true, scope);
    T *p = static_cast<T *>(entry.local_addr());
    std::construct_at(p, std::forward<Args>(args)...);
    LiteAccessor<T, true> accessor;
    accessor.block = reinterpret_cast<allocator::BlockHead *>(p) - 1;
    accessor.local_ptr = p;
    return accessor;
}

template <typename T>
template <typename... Args>
inline LiteAccessor<T, true> UniqueFarPtr<T>::allocate_lite_with_size(
    size_t size, DereferenceScope &scope, Args &&...args) {
    this->reset();
    size_64 = size;
    auto cache = Cache::get_default();
    cache->template allocate<true>(&entry, size, true, scope);
    T *p = static_cast<T *>(entry.local_addr());
    std::construct_at(p, std::forward<Args>(args)...);
    LiteAccessor<T, true> accessor;
    accessor.block = reinterpret_cast<allocator::BlockHead *>(p) - 1;
    accessor.local_ptr = p;
    return accessor;
}

template <typename T>
template <typename... Args>
inline LiteAccessor<T, true> UniqueFarPtr<T>::allocate_lite_with_size_and_pin(
    size_t size, DereferenceScope &scope, Args &&...args) {
    this->reset();
    size_64 = size;
    auto cache = Cache::get_default();
    cache->template allocate<true>(&entry, size, true, scope);
    T *p = static_cast<T *>(entry.local_addr());
    std::construct_at(p, std::forward<Args>(args)...);
    entry.set_pinned(true);
    LiteAccessor<T, true> accessor;
    accessor.block = reinterpret_cast<allocator::BlockHead *>(p) - 1;
    accessor.local_ptr = p;
    return accessor;
}

template <typename T>
template <bool Mut, typename... Args>
inline LiteAccessor<T, Mut> UniqueFarPtr<T>::allocate_lite_uninitialized(
    DereferenceScope &scope, uint32_t logical_owner_id) {
    this->reset();
    auto cache = Cache::get_default();
    cache->template allocate<true>(&entry, sizeof(T), Mut, scope,
                                   logical_owner_id);
    T *p = static_cast<T *>(entry.local_addr());
    LiteAccessor<T, Mut> accessor;
    accessor.block = reinterpret_cast<allocator::BlockHead *>(p) - 1;
    accessor.local_ptr = p;
    return accessor;
}

template <typename T>
template <typename... Args>
inline LiteAccessor<T, true> UniqueFarPtr<T>::allocate_lite_from_busy(
    DereferenceScope &scope, Args &&...args) {
    assert(entry.load_state().state == BUSY);
    auto cache = Cache::get_default();
    cache->template allocate<true>(&entry, sizeof(T), true, scope);
    T *p = static_cast<T *>(entry.local_addr());
    std::construct_at(p, std::forward<Args>(args)...);
    LiteAccessor<T, true> accessor;
    accessor.block = reinterpret_cast<allocator::BlockHead *>(p) - 1;
    accessor.local_ptr = p;
    return accessor;
}

template <typename T>
template <bool Mut>
inline LiteAccessor<T[], Mut> UniqueFarPtr<T[]>::allocate_lite(
    size_t n, DereferenceScope &scope) {
    this->reset();
    auto cache = Cache::get_default();
    constexpr bool dirty = !std::is_trivially_constructible<T>::value || Mut;
    cache->allocate<true>(&entry, n * sizeof(T), dirty, scope);
    if constexpr (dirty) {
        T *base_addr = entry.local_addr();
        for (size_t i = 0; i < n; i++) {
            new (base_addr + i) T;
        }
    }
    LiteAccessor<T, Mut> accessor;
    void *p = entry.local_addr();
    accessor.block = reinterpret_cast<allocator::BlockHead *>(p) - 1;
    accessor.local_ptr = p;
    return accessor;
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(far_obj_t obj, DereferenceScope &scope) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite<Mut>(obj, __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(far_obj_t obj, DereferenceScope &scope,
                                          NoProfileTag) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite_no_profile<Mut>(
        obj, __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(far_obj_t obj, __DMH__,
                                          DereferenceScope &scope) {
    void *ptr = Cache::get_default()->template fetch_lite<Mut>(obj, __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(far_obj_t obj, __DMH__,
                                          DereferenceScope &scope,
                                          NoProfileTag) {
    void *ptr = Cache::get_default()->template fetch_lite_no_profile<Mut>(
        obj, __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
template <typename Impl>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr,
                                          DereferenceScope &scope) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite<Mut>(uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
template <typename Impl>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr,
                                          DereferenceScope &scope,
                                          NoProfileTag) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite_no_profile<Mut>(
        uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
template <typename Impl>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr,
                                          __DMH__, DereferenceScope &scope) {
    void *ptr = Cache::get_default()->template fetch_lite<Mut>(uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
template <typename Impl>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr,
                                          __DMH__, DereferenceScope &scope,
                                          NoProfileTag) {
    void *ptr = Cache::get_default()->template fetch_lite_no_profile<Mut>(
        uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtr<void> &uptr,
                                          DereferenceScope &scope) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite<Mut>(uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtr<void> &uptr,
                                          DereferenceScope &scope,
                                          NoProfileTag) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite_no_profile<Mut>(
        uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtr<void> &uptr,
                                          __DMH__, DereferenceScope &scope) {
    void *ptr = Cache::get_default()->template fetch_lite<Mut>(uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline LiteAccessor<T, Mut>::LiteAccessor(const UniqueFarPtr<void> &uptr,
                                          __DMH__, DereferenceScope &scope,
                                          NoProfileTag) {
    void *ptr = Cache::get_default()->template fetch_lite_no_profile<Mut>(
        uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = static_cast<T *>(ptr);
    check(block, local_ptr);
}

template <bool Mut>
template <typename Impl>
inline LiteAccessor<void, Mut>::LiteAccessor(
    const UniqueFarPtrBase<void, Impl> &uptr, DereferenceScope &scope) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite<Mut>(uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = ptr;
    check(block, local_ptr);
}

template <bool Mut>
template <typename Impl>
inline LiteAccessor<void, Mut>::LiteAccessor(
    const UniqueFarPtrBase<void, Impl> &uptr, DereferenceScope &scope,
    NoProfileTag) {
    ON_MISS_BEGIN
    ON_MISS_END
    void *ptr = Cache::get_default()->template fetch_lite_no_profile<Mut>(
        uptr.obj(), __on_miss__, scope);
    block = static_cast<allocator::BlockHead *>(ptr) - 1;
    local_ptr = ptr;
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline bool LiteAccessor<T, Mut>::async_fetch_slow_path(far_obj_t obj,
                                                        DereferenceScope &scope) {
    auto [at_local, local_addr] = Cache::get_default()->template async_fetch_lite<Mut>(obj, scope);
    this->local_ptr = static_cast<Pointer>(local_addr);
    this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
    check(block, local_ptr);
    return at_local;
}

template <typename T, bool Mut>
inline bool LiteAccessor<T, Mut>::async_fetch(far_obj_t obj,
                                              DereferenceScope &scope) {
    auto entry = obj.get_entry_ptr();
    bool fast = entry->load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_access(
            *entry, Mut ? profile::ReferenceKind::Write
                        : profile::ReferenceKind::Read);
        void *local_addr = entry->local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    return async_fetch_slow_path(obj, scope);
}

template <typename T, bool Mut>
inline bool LiteAccessor<T, Mut>::async_fetch_no_profile(
    far_obj_t obj, DereferenceScope &scope) {
    auto entry = obj.get_entry_ptr();
    bool fast = entry->load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_reference(
            *entry, Mut ? profile::ReferenceKind::Write
                        : profile::ReferenceKind::Read);
        void *local_addr = entry->local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    auto [at_local, local_addr] =
        Cache::get_default()->template async_fetch_lite_no_profile<Mut>(obj,
                                                                        scope);
    this->local_ptr = static_cast<Pointer>(local_addr);
    this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
    check(block, local_ptr);
    return at_local;
}

template <typename T, bool Mut>
inline bool LiteAccessor<T, Mut>::async_fetch(const UniqueFarPtr<T> &uptr,
                                              DereferenceScope &scope) {
    bool fast = uptr.get_entry().load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_access(
            uptr.get_entry(), Mut ? profile::ReferenceKind::Write
                                  : profile::ReferenceKind::Read);
        void *local_addr = uptr.get_entry().local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    return async_fetch_slow_path(uptr.obj(), scope);
}

template <typename T, bool Mut>
inline bool LiteAccessor<T, Mut>::async_fetch_no_profile(
    const UniqueFarPtr<T> &uptr, DereferenceScope &scope) {
    bool fast = uptr.get_entry().load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_reference(
            uptr.get_entry(), Mut ? profile::ReferenceKind::Write
                                  : profile::ReferenceKind::Read);
        void *local_addr = uptr.get_entry().local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    auto [at_local, local_addr] = Cache::get_default()
                                      ->template async_fetch_lite_no_profile<Mut>(
                                          uptr.obj(), scope);
    this->local_ptr = static_cast<Pointer>(local_addr);
    this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
    check(block, local_ptr);
    return at_local;
}

template <typename T, bool Mut>
inline void LiteAccessor<T, Mut>::sync() {
    this->local_ptr = static_cast<Pointer>(Cache::get_entry_of(get_obj()).local_addr());
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline void LiteAccessor<T, Mut>::pin() const {
    assert(local_ptr == nullptr || block != nullptr);
    if (block != nullptr) Cache::get_default()->pin(get_obj());
    check(block, local_ptr);
}

template <typename T, bool Mut>
inline void LiteAccessor<T, Mut>::unpin() const {
    assert(local_ptr == nullptr || block != nullptr);
    if (block != nullptr) Cache::get_default()->unpin(get_obj());
    check(block, local_ptr);
}

template <bool Mut>
inline void LiteAccessor<void, Mut>::pin() const {
    assert(local_ptr == nullptr || block != nullptr);
    if (block != nullptr) Cache::get_default()->pin(get_obj());
    check(block, local_ptr);
}

template <bool Mut>
inline void LiteAccessor<void, Mut>::unpin() const {
    assert(local_ptr == nullptr || block != nullptr);
    if (block != nullptr) Cache::get_default()->unpin(get_obj());
    check(block, local_ptr);
}

template <bool Mut>
inline bool LiteAccessor<void, Mut>::async_fetch_slow_path(
    far_obj_t obj, DereferenceScope &scope) {
    auto [at_local, local_addr] = Cache::get_default()->template async_fetch_lite<Mut>(obj, scope);
    this->local_ptr = static_cast<Pointer>(local_addr);
    this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
    check(block, local_ptr);
    return at_local;
}

template <bool Mut>
inline bool LiteAccessor<void, Mut>::async_fetch(far_obj_t obj,
                                                 DereferenceScope &scope) {
    auto entry = obj.get_entry_ptr();
    bool fast = entry->load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_access(
            *entry, Mut ? profile::ReferenceKind::Write
                        : profile::ReferenceKind::Read);
        void *local_addr = entry->local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    return async_fetch_slow_path(obj, scope);
}

template <bool Mut>
inline bool LiteAccessor<void, Mut>::async_fetch_no_profile(
    far_obj_t obj, DereferenceScope &scope) {
    auto entry = obj.get_entry_ptr();
    bool fast = entry->load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_reference(
            *entry, Mut ? profile::ReferenceKind::Write
                        : profile::ReferenceKind::Read);
        void *local_addr = entry->local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    auto [at_local, local_addr] =
        Cache::get_default()->template async_fetch_lite_no_profile<Mut>(obj,
                                                                        scope);
    this->local_ptr = static_cast<Pointer>(local_addr);
    this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
    check(block, local_ptr);
    return at_local;
}

template <bool Mut>
inline bool LiteAccessor<void, Mut>::async_fetch(const UniqueFarPtr<void> &uptr,
                                                 DereferenceScope &scope) {
    bool fast = uptr.get_entry().load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_access(
            uptr.get_entry(), Mut ? profile::ReferenceKind::Write
                                  : profile::ReferenceKind::Read);
        void *local_addr = uptr.get_entry().local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    return async_fetch_slow_path(uptr.obj(), scope);
}

template <bool Mut>
inline bool LiteAccessor<void, Mut>::async_fetch_no_profile(
    const UniqueFarPtr<void> &uptr, DereferenceScope &scope) {
    bool fast = uptr.get_entry().load_state().template is_deref_fast_path<Mut>();
    if (fast) [[likely]] {
        Cache::get_default()->record_local_fast_path_reference(
            uptr.get_entry(), Mut ? profile::ReferenceKind::Write
                                  : profile::ReferenceKind::Read);
        void *local_addr = uptr.get_entry().local_addr();
        this->local_ptr = static_cast<Pointer>(local_addr);
        this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
        check(block, local_ptr);
        return true;
    }
    auto [at_local, local_addr] = Cache::get_default()
                                      ->template async_fetch_lite_no_profile<Mut>(
                                          uptr.obj(), scope);
    this->local_ptr = static_cast<Pointer>(local_addr);
    this->block = static_cast<allocator::BlockHead *>(local_addr) - 1;
    check(block, local_ptr);
    return at_local;
}

template <bool Mut>
inline LiteAccessor<void, Mut> UniqueFarPtr<void>::allocate_lite(
    size_t nbytes, DereferenceScope &scope) {
    this->reset();
    auto cache = Cache::get_default();
    cache->allocate<true>(&entry, nbytes, Mut, scope);
    LiteAccessor<void, Mut> accessor;
    void *p = entry.local_addr();
    accessor.block = static_cast<allocator::BlockHead *>(p) - 1;
    accessor.local_ptr = p;
    return accessor;
}

template <typename T, bool Mut>
inline bool at_local(const LiteAccessor<T, Mut> &accessor) {
    return at_local(accessor.get_obj());
}

inline void check_memory_low(DereferenceScope &scope) {
    auto cache = Cache::get_default();
    cache->check_memory_low(scope);
}

inline void DereferenceScope::enter() { Cache::get_default()->enter_scope(); }
inline void DereferenceScope::exit() { Cache::get_default()->exit_scope(); }

}  // namespace FarLib::cache
