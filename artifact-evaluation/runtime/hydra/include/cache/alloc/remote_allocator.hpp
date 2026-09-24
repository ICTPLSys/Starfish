#pragma once

#include <cstddef>
#include <cstdint>

#include <iostream>

#include "cache/alloc/small_object_stripe.hpp"
#include "utils/debug.hpp"
#define REMOTE_REGION_ALLOCATOR

#ifdef REMOTE_REGION_ALLOCATOR
#include "cache/alloc/region_remote_allocator.hpp"
#endif
namespace FarLib {

namespace cache {

#ifdef REMOTE_REGION_ALLOCATOR
using FarLib::allocator::remote::InvalidRemoteAddr;
using FarLib::allocator::remote::remote_global_heap;
using FarLib::allocator::remote::remote_thread_heap;
class RemoteAllocator {
private:
    // EC stripe layout for small objects.  Only initialized when
    // ft_method != none; while ft_method == none every dispatch below takes the
    // plain remote_thread_heap path and no stripe code executes.
    SmallObjectStripeManager small_object_stripes;

    // Cached fault-tolerance state.  The configuration is frozen before the
    // cache is constructed, so these are constant for the process lifetime.
    // With ft_method == none they stay "off", which reduces both dispatches
    // below to a single never-taken branch, i.e. no stripe state is read and no
    // stripe code can run.  (Configure::self_check() rejects
    // ft_small_object_cutoff == 0, so a zero cutoff implies "ft off".)
    bool ft_enabled_ = false;
    size_t ft_cutoff_ = 0;

public:
    RemoteAllocator(size_t buffer_size) {
        remote_global_heap.register_remote(buffer_size);
        const auto &config = ::FarLib::get_config();
        ft_enabled_ = config.ft_enabled();
        if (ft_enabled_) {
            if (config.ft_small_stripe_shard_size_bytes !=
                SmallObjectStripeShardSize) {
                ERROR("ft_small_stripe_shard_size_bytes must match remote RegionSize");
            }
            ft_cutoff_ = config.ft_small_object_cutoff;
            small_object_stripes.init(buffer_size, SmallObjectStripeShardSize);
        }
    }

    // The two hot dispatches below stay tiny: with ft_method == none,
    // ft_cutoff_ == 0 and ft_enabled_ == false, so the small-object branch is
    // never taken and the stripe code is never reached.  The stripe bodies are
    // kept out of line (noinline) on purpose: the cache is heavily templated
    // and inlining the ~800 line stripe path into every instantiation bloated
    // the binary by ~130 KB and cost ~4% runtime on the llama workload even
    // with ft_method=none.
    uint64_t allocate(size_t size,
                      allocator::six_group::GroupId behavior_group_id = 0) {
        if (size < ft_cutoff_) return allocate_small_object(size);
        return allocate_flat(size, behavior_group_id);
    }

    // Explicit flat storage for recomputable objects, retaining Design2 routing.
    uint64_t allocate_flat(size_t size,
                           allocator::six_group::GroupId behavior_group_id = 0) {
        uint64_t addr = remote_thread_heap.allocate(size, behavior_group_id);
        if (addr == InvalidRemoteAddr) [[unlikely]] {
            info();
            ERROR("Out Of Remote Memory!");
        }
        return addr;
    }

    void deallocate(uint64_t addr) {
        if (ft_enabled_ && small_object_stripes.owns(addr)) {
            deallocate_small_object(addr);
            return;
        }
        remote_thread_heap.deallocate(addr);
    }

    void deallocate_batch(std::vector<uint64_t> &addrs) {
        if (!ft_enabled_) {
            remote_thread_heap.deallocate_batch(addrs);
            return;
        }
        deallocate_batch_ft(addrs);
    }

    void info() { remote_global_heap.info(); }

    const SmallObjectStripeShardTable &small_stripe_shard_table() const {
        return small_object_stripes.shard_table();
    }

    // Non-const accessor: the ec_batch group path allocates and seals slot
    // groups through the manager's own API.
    SmallObjectStripeManager &small_object_stripe_manager() {
        return small_object_stripes;
    }

    const SmallObjectStripeManager &small_object_stripe_manager() const {
        return small_object_stripes;
    }

private:
    [[gnu::noinline]] uint64_t allocate_small_object(size_t size) {
        uint64_t addr = small_object_stripes.allocate(size);
        if (addr == InvalidRemoteAddr) [[unlikely]] {
            info();
            ERROR("Out Of Remote Memory!");
        }
        return addr;
    }

    [[gnu::noinline]] void deallocate_small_object(uint64_t addr) {
        deallocate_stripe_addr(addr);
    }

    // One stripe-owned address.  An address that lives in a slot group is
    // released per object through the manager's group-aware entry point (the
    // group itself goes back only when its last live object is gone); every
    // other address keeps the original single-object mark_dead() semantics.
    // ft_method != ec_batch never materialises group state, so there the two
    // are equivalent and the behaviour is unchanged.
    [[gnu::noinline]] void deallocate_stripe_addr(uint64_t addr) {
        const auto result = small_object_stripes.release_group_object(addr);
        if (result == SmallObjectStripeManager::kGroupReleaseNotGrouped) {
            if (!small_object_stripes.mark_dead(addr)) [[unlikely]] {
                ERROR("invalid stripe small-object deallocate");
            }
            return;
        }
        if (result == SmallObjectStripeManager::kGroupReleaseRejected) [[unlikely]] {
            // A release aimed at a group that is already settled - all its live
            // objects are back, because the bounded dealloc settle consumed the
            // last one, because its group was retired to the terminal state
            // dead, or because the whole group was released - has nothing left
            // to do.  Releasing one of its addresses again is idempotent, not
            // invalid input, so it is reported once and ignored instead of
            // aborting the process.  Every other rejection (a zero-filled hole,
            // a second release inside a group that still owns live objects)
            // stays an error.
            if (small_object_stripes.slot_group_addr_is_settled(addr)) {
                std::cout << "INFO: stripe slot-group release ignored addr=0x"
                          << std::hex << addr << std::dec
                          << " reason=group_already_settled" << std::endl;
                return;
            }
            ERROR("invalid stripe slot-group deallocate");
        }
    }

    [[gnu::noinline]] void deallocate_batch_ft(std::vector<uint64_t> &addrs) {
        // ft_method != none: stripe-owned addresses are not bin allocations.
        std::vector<uint64_t> bin_addrs;
        bin_addrs.reserve(addrs.size());
        std::vector<uint64_t> stripe_addrs;
        for (uint64_t addr : addrs) {
            if (small_object_stripes.owns(addr)) {
                stripe_addrs.push_back(addr);
            } else {
                bin_addrs.push_back(addr);
            }
        }
        for (uint64_t addr : stripe_addrs) {
            deallocate_stripe_addr(addr);
        }
        remote_thread_heap.deallocate_batch(bin_addrs);
        addrs.clear();
    }
};
#else
class RemoteAllocator {
private:
    size_t buffer_size;
    std::atomic_uint64_t allocated_offset = 0;

public:
    RemoteAllocator(size_t buffer_size) : buffer_size(buffer_size) {}

    size_t get_allocated() const { return allocated_offset.load(); }

    uint64_t allocate(size_t size,
                      allocator::six_group::GroupId = 0) {
        uint64_t ptr = allocated_offset.fetch_add(size);
        ASSERT(ptr + size <= buffer_size);
        return ptr;
    }

    // Do we need a size?
    void deallocate(uint64_t location) {
        // Do nothing now
    }

    void deallocate_batch(std::vector<uint64_t> &locations) {
        locations.clear();
    }
};
#endif
}  // namespace cache

}  // namespace FarLib
