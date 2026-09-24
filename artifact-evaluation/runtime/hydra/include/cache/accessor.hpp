#pragma once
#include "cache/region_based_allocator.hpp"
#include "concurrent_cache.hpp"
#include "cache/base/scope.hpp"
namespace FarLib {
using Cache = cache::ConcurrentArrayCache;
}

namespace FarLib {

namespace cache {

bool check_fetch(FarObjectEntry *entry, fetch_ddl_t &ddl);
size_t check_cq();
bool at_local(far_obj_t obj);

template <typename T, bool Mut = false>
class LiteAccessor;

struct NoProfileTag {};
inline constexpr NoProfileTag kNoProfile{};

template <typename T, bool Mut>
bool at_local(const LiteAccessor<T, Mut> &accessor);
void check_memory_low(DereferenceScope &scope);

template <typename T, typename Impl>
struct UniqueFarPtrBase {
private:
    Impl *get_impl() { return reinterpret_cast<Impl *>(this); }
    const Impl *get_impl() const {
        return reinterpret_cast<const Impl *>(this);
    }

public:
    FarObjectEntry entry;

    FarObjectEntry &get_entry() { return entry; }
    const FarObjectEntry &get_entry() const { return entry; }

    UniqueFarPtrBase() { entry.set_free(); }
    UniqueFarPtrBase(const UniqueFarPtrBase &) = delete;
    UniqueFarPtrBase(UniqueFarPtrBase &&other) noexcept { move(&other.entry); }
    ~UniqueFarPtrBase() { reset(); }
    UniqueFarPtrBase &operator=(const UniqueFarPtrBase &) = delete;
    UniqueFarPtrBase &operator=(UniqueFarPtrBase &&other) {
        reset();
        move(&other.entry);
        return *this;
    }
    void reset() {
        if (!is_null()) {
            Cache::get_default()->deallocate_unique(entry, get_impl()->size());
        }
    }
    far_obj_t obj() const {
        return {.size = get_impl()->size(),
                .obj_id = reinterpret_cast<uint64_t>(&entry)};
    }

    bool is_null() const { return entry.load_state().state == FREE; }

    // Call immediately after allocation, in that DereferenceScope. Returns
    // false without changing protection if the first eviction already began.
    bool mark_recomputable() {
        return Cache::get_default()->mark_recomputable(obj());
    }

    bool is_recomputable() const {
        return !is_null() && entry.is_recomputable();
    }

    template <bool Mut = false, typename Scope>
    __attribute__((always_inline)) LiteAccessor<T, Mut> access(
        Scope &&scope) const {
        if constexpr (Mut) {
            Cache::get_default()->guard_recompute_mutation(entry);
        }
        bool fast = entry.load_state().is_deref_fast_path<Mut>();
        if (fast) [[likely]] {
            Cache::get_default()->record_local_fast_path_access(
                entry, Mut ? profile::ReferenceKind::Write
                           : profile::ReferenceKind::Read);
            void *p = entry.local_addr();
            Cache::get_default()->record_simple_local_access(p);
            auto block = Cache::object_block(&entry, p);
            return LiteAccessor<T, Mut>(block, p, obj());
        } else {
            return LiteAccessor<T, Mut>(*this, std::forward<Scope>(scope));
        }
    }

    template <bool Mut = false, typename Scope>
    __attribute__((always_inline)) LiteAccessor<T, Mut> access(
        __DMH__, Scope &&scope) const {
        if constexpr (Mut) {
            Cache::get_default()->guard_recompute_mutation(entry);
        }
        bool fast = entry.load_state().is_deref_fast_path<Mut>();
        if (fast) [[likely]] {
            Cache::get_default()->record_local_fast_path_access(
                entry, Mut ? profile::ReferenceKind::Write
                           : profile::ReferenceKind::Read);
            void *p = entry.local_addr();
            Cache::get_default()->record_simple_local_access(p);
            auto block = Cache::object_block(&entry, p);
            return LiteAccessor<T, Mut>(block, p, obj());
        } else {
            return LiteAccessor<T, Mut>(*this, __on_miss__,
                                        std::forward<Scope>(scope));
        }
    }

    template <typename Scope>
    bool prefetch(Scope &&scope) const {
        return Cache::get_default()->prefetch(obj(), scope);
    }

    // !!! this is NOT thread safe
    // the src and dst entry should not be accessed concurrently
    //
    // move another entry to this, will rewrite the object header
    // dereferencing unique ptr when moving is UB for mutators
    // only eviction may be concurrent with this
    void move(FarObjectEntry *other);
    operator bool() const { return !is_null(); }

    bool cas_null_to_busy();

    // move this to `to`
    // set this as busy & null
    void atomic_move_to_and_set_busy(UniqueFarPtrBase<T, Impl> &to);
};

template <typename T>
struct UniqueFarPtr : public UniqueFarPtrBase<T, UniqueFarPtr<T>> {
private:
    using Base = UniqueFarPtrBase<T, UniqueFarPtr<T>>;
    using Base::entry;

public:
    uint64_t size_64 = 0;
    size_t size() const { return std::max(sizeof(T), size_64); }

    template <typename... Args>
    LiteAccessor<T, true> allocate_lite(DereferenceScope &scope, Args &&...args);

    template <typename... Args>
    LiteAccessor<T, true> allocate_lite_with_size(size_t size,
                                                  DereferenceScope &scope,
                                                  Args &&...args);

    template <typename... Args>
    LiteAccessor<T, true> allocate_lite_with_size_and_pin(
        size_t size, DereferenceScope &scope, Args &&...args);

    template <bool Mut = false, typename... Args>
    LiteAccessor<T, Mut> allocate_lite_uninitialized(
        DereferenceScope &scope, uint32_t logical_owner_id = 0);

    template <typename... Args>
    LiteAccessor<T, true> allocate_lite_from_busy(DereferenceScope &scope,
                                                  Args &&...args);
};

template <typename T>
struct UniqueFarPtr<T[]> : public UniqueFarPtrBase<T[], UniqueFarPtr<T[]>> {
private:
    using Base = UniqueFarPtrBase<T[], UniqueFarPtr<T[]>>;
    using Base::entry;

public:
    size_t size() const { return entry.object_size(); }
    size_t element_count() const { return size() / sizeof(T); }

    template <bool Mut = false>
    LiteAccessor<T[], Mut> allocate_lite(size_t n, DereferenceScope &scope);
};

template <>
struct UniqueFarPtr<void> : public UniqueFarPtrBase<void, UniqueFarPtr<void>> {
private:
    using Base = UniqueFarPtrBase<void, UniqueFarPtr<void>>;
    using Base::entry;

public:
    size_t size() const { return entry.object_size(); }

    template <bool Mut = false>
    LiteAccessor<void, Mut> allocate_lite(size_t nbytes,
                                          DereferenceScope &scope);
};

// LiteAccessor mut be used in a dereference scope
template <typename T, bool Mut>
class LiteAccessor {
    using Pointer = std::conditional_t<Mut, T *, const T *>;
    using Reference = std::conditional_t<Mut, T &, const T &>;

    allocator::BlockHead *block;
    Pointer local_ptr;
    far_obj_t logical_obj = far_obj_t::null();
    void remember_object(far_obj_t obj) {
        logical_obj = !obj.is_null() && obj.get_entry_ptr()->hydra_packed()
            ? obj : far_obj_t::null();
    }

    template <typename U, bool M>
    friend class LiteAccessor;

    template <typename U>
    friend class UniqueFarPtr;

public:
    void check() const {
        if (is_null()) [[unlikely]] {
            return;
        }
        check(block, local_ptr);
    }
    void check(allocator::BlockHead *block) const {
        // assert((block->obj_meta_data.load().get_entry_ptr()->local_addr()) ==
        //        block + 1);
    }

    void check(allocator::BlockHead *block, Pointer local_addr) const {
        if (block == nullptr) { assert(local_addr == nullptr); return; }
        check(block);
        assert(reinterpret_cast<const char *>(local_addr) >= static_cast<const char *>(block->get_object_ptr()));
        assert(reinterpret_cast<const char *>(local_addr) <= static_cast<const char *>(block->get_object_ptr()) + block->obj_meta_data.load().size);
    }

public:
    LiteAccessor() : block(nullptr), local_ptr(nullptr) {}

    // unsafe!
    LiteAccessor(allocator::BlockHead *block, void *local_ptr,
                 far_obj_t obj = far_obj_t::null())
        : block(block), local_ptr(static_cast<Pointer>(local_ptr)) {
        remember_object(obj);
    }

    LiteAccessor(far_obj_t obj, DereferenceScope &scope);

    LiteAccessor(far_obj_t obj, DereferenceScope &scope, NoProfileTag);

    LiteAccessor(far_obj_t obj, __DMH__, DereferenceScope &scope);

    LiteAccessor(far_obj_t obj, __DMH__, DereferenceScope &scope,
                 NoProfileTag);

    template <typename Impl>
    LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr, DereferenceScope &scope);

    template <typename Impl>
    LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr, DereferenceScope &scope,
                 NoProfileTag);

    template <typename Impl>
    LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr, __DMH__,
                 DereferenceScope &scope);

    template <typename Impl>
    LiteAccessor(const UniqueFarPtrBase<T, Impl> &uptr, __DMH__,
                 DereferenceScope &scope, NoProfileTag);

    LiteAccessor(const UniqueFarPtr<void> &uptr, DereferenceScope &scope);

    LiteAccessor(const UniqueFarPtr<void> &uptr, DereferenceScope &scope,
                 NoProfileTag);

    LiteAccessor(const UniqueFarPtr<void> &uptr, __DMH__,
                 DereferenceScope &scope);

    LiteAccessor(const UniqueFarPtr<void> &uptr, __DMH__,
                 DereferenceScope &scope, NoProfileTag);

    LiteAccessor(const LiteAccessor<T, Mut> &) = default;
    LiteAccessor(LiteAccessor<T, Mut> &&) = default;
    LiteAccessor<T, Mut> &operator=(const LiteAccessor<T, Mut> &) = default;
    LiteAccessor<T, Mut> &operator=(LiteAccessor<T, Mut> &&) = default;

    template <typename U>
    LiteAccessor(LiteAccessor<U, Mut> other, Pointer local_ptr)
        : block(other.block), local_ptr(local_ptr), logical_obj(other.logical_obj) {
        check(block, local_ptr);
    }

    ~LiteAccessor() = default;

    LiteAccessor<T, true> as_mut() const {
        check(block, local_ptr);
        Cache::get_default()->guard_recompute_mutation(*get_obj().get_entry_ptr());
        if constexpr (!Mut) {
            Cache::get_default()->mark_dirty(get_obj());
        }
        LiteAccessor<T, true> mut_accessor;
        mut_accessor.block = block;
        mut_accessor.logical_obj = logical_obj;
        mut_accessor.local_ptr = const_cast<T *>(local_ptr);
        return mut_accessor;
    }

    bool is_null() const {
        check(block, local_ptr);
        return block == nullptr;
    }

    far_obj_t get_obj() const {
        check(block, local_ptr);
        return logical_obj.is_null() ? block->obj_meta_data.load() : logical_obj;
    }

    Pointer as_ptr() const {
        check(block, local_ptr);
        return local_ptr;
    }

    Pointer operator->() const {
        check(block, local_ptr);
        return local_ptr;
    }

    Reference operator*() const {
        check(block, local_ptr);
        return *local_ptr;
    }

    bool async_fetch_slow_path(far_obj_t obj, DereferenceScope &scope);
    __attribute__((always_inline)) bool async_fetch(far_obj_t obj,
                                                    DereferenceScope &scope);
    __attribute__((always_inline)) bool async_fetch_no_profile(
        far_obj_t obj, DereferenceScope &scope);
    __attribute__((always_inline)) bool async_fetch(const UniqueFarPtr<T> &uptr,
                                                    DereferenceScope &scope);
    __attribute__((always_inline)) bool async_fetch_no_profile(
        const UniqueFarPtr<T> &uptr, DereferenceScope &scope);

    void sync();

    // when a new eviction phase starts, LiteAccessors should be pinned
    void pin() const;

    void unpin() const;
};

template <bool Mut>
class LiteAccessor<void, Mut> {
    using Pointer = std::conditional_t<Mut, void *, const void *>;

    allocator::BlockHead *block;
    Pointer local_ptr;
    far_obj_t logical_obj = far_obj_t::null();
    void remember_object(far_obj_t obj) {
        logical_obj = !obj.is_null() && obj.get_entry_ptr()->hydra_packed()
            ? obj : far_obj_t::null();
    }

    template <typename U, bool M>
    friend class LiteAccessor;

    template <typename U>
    friend class UniqueFarPtr;

public:
    void check() const { check(block, local_ptr); }
    void check(allocator::BlockHead *block) const {
        assert((block->obj_meta_data.load().get_entry_ptr()->local_addr()) ==
               block + 1);
    }

    void check(allocator::BlockHead *block, Pointer local_addr) const {
        if (block == nullptr) { assert(local_addr == nullptr); return; }
        check(block);
        assert(static_cast<const char *>(local_addr) >= static_cast<const char *>(block->get_object_ptr()));
        assert(static_cast<const char *>(local_addr) <= static_cast<const char *>(block->get_object_ptr()) + block->obj_meta_data.load().size);
    }

public:
    LiteAccessor() : block(nullptr), local_ptr(nullptr) {}
    LiteAccessor(const LiteAccessor<void, Mut> &) = default;
    LiteAccessor(LiteAccessor<void, Mut> &&) = default;
    LiteAccessor<void, Mut> &operator=(const LiteAccessor<void, Mut> &) =
        default;
    LiteAccessor<void, Mut> &operator=(LiteAccessor<void, Mut> &&) = default;
    template <typename U>
    LiteAccessor(LiteAccessor<U, Mut> other, Pointer local_ptr)
        : block(other.block), local_ptr(local_ptr), logical_obj(other.logical_obj) {
        check(block, local_ptr);
    }
    // unsafe!
    LiteAccessor(allocator::BlockHead *block, void *local_ptr,
                 far_obj_t obj = far_obj_t::null())
        : block(block), local_ptr(static_cast<Pointer>(local_ptr)) {
        remember_object(obj);
    }
    template <typename Impl>
    LiteAccessor(const UniqueFarPtrBase<void, Impl> &uptr,
                 DereferenceScope &scope);
    template <typename Impl>
    LiteAccessor(const UniqueFarPtrBase<void, Impl> &uptr,
                 DereferenceScope &scope, NoProfileTag);
    ~LiteAccessor() = default;
    bool is_null() const {
        check(block, local_ptr);
        return block == nullptr;
    }
    far_obj_t get_obj() const {
        check(block, local_ptr);
        return logical_obj.is_null() ? block->obj_meta_data.load() : logical_obj;
    }
    Pointer as_ptr() {
        check(block, local_ptr);
        return local_ptr;
    }
    Pointer operator->() {
        check(block, local_ptr);
        return local_ptr;
    }
    template <typename T>
    LiteAccessor<T, Mut> as() {
        check(block, local_ptr);
        using Dest = LiteAccessor<T, Mut>;
        return Dest(std::move(*this),
                    static_cast<typename Dest::Pointer>(local_ptr));
    }

    static LiteAccessor<void, Mut> allocate(size_t size,
                                            DereferenceScope &scope) {
        auto accessor = LiteAccessor<void, Mut>();
        auto cache = Cache::get_default();
        auto [o, p] = cache->allocate<true>(size, Mut, scope);
        accessor.block = Cache::object_block(o.get_entry_ptr(), p);
        accessor.remember_object(o);
        accessor.local_ptr = p;
        return accessor;
    }

    void pin() const;

    void unpin() const;

    LiteAccessor<void, true> as_mut() const {
        check(block, local_ptr);
        Cache::get_default()->guard_recompute_mutation(*get_obj().get_entry_ptr());
        if constexpr (!Mut) {
            Cache::get_default()->mark_dirty(get_obj());
        }
        LiteAccessor<void, true> mut_accessor;
        mut_accessor.block = block;
        mut_accessor.logical_obj = logical_obj;
        mut_accessor.local_ptr = const_cast<void *>(local_ptr);
        return mut_accessor;
    }

    bool async_fetch_slow_path(far_obj_t obj, DereferenceScope &scope);

    __attribute__((always_inline)) bool async_fetch(far_obj_t obj,
                                                    DereferenceScope &scope);
    __attribute__((always_inline)) bool async_fetch_no_profile(
        far_obj_t obj, DereferenceScope &scope);

    __attribute__((always_inline)) bool async_fetch(
        const UniqueFarPtr<void> &uptr, DereferenceScope &scope);
    __attribute__((always_inline)) bool async_fetch_no_profile(
        const UniqueFarPtr<void> &uptr, DereferenceScope &scope);
};


}  // namespace cache

using cache::DereferenceScope;
using cache::far_obj_t;
using cache::LiteAccessor;
using cache::RootDereferenceScope;
using cache::UniqueFarPtr;

static_assert(sizeof(far_obj_t) == sizeof(uint64_t));

// Recipe declaration follows the output/inputs/callback/argument order.
inline bool cache::DereferenceScope::mark_recomputable(
    far_obj_t output, const recompute::Inputs &inputs,
    recompute::EntryFn entry, uint64_t entry_args) {
    return Cache::get_default()->register_recompute_recipe(
        output, inputs, entry, entry_args);
}

// Applications promise that marked data can be regenerated. This annotation
// skips EC; it does not register or automatically execute a rebuild callback.
inline bool mark_recomputable(far_obj_t obj) {
    return Cache::get_default()->mark_recomputable(obj);
}

inline bool is_recomputable(far_obj_t obj) {
    return Cache::get_default()->is_recomputable(obj);
}

template <typename T>
inline bool mark_recomputable(UniqueFarPtr<T> &obj) {
    return obj.mark_recomputable();
}

template <typename T>
inline bool is_recomputable(const UniqueFarPtr<T> &obj) {
    return obj.is_recomputable();
}

inline LiteAccessor<void, true> alloc_uninitialized(size_t size,
                                                    DereferenceScope &scope) {
    return LiteAccessor<void, true>::allocate(size, scope);
}

}  // namespace FarLib

#include "cache/accessor/accessor_impl.hpp"
#include "cache/accessor/entry_helpers.hpp"
