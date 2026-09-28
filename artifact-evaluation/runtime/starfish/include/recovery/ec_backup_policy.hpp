#pragma once

// Small, dependency-free predicates shared by the EC backup/recovery path and
// its CPU fixtures.  The caller supplies the concrete stripe-manager type and
// the endpoint-liveness lookup so this header does not construct RDMA state or
// depend on the cache object.

#include <cstdint>

namespace FarLib::cache::ec_backup_policy {

// Return whether the remote copy cannot be reused because one of the physical
// endpoints serving it is dead.  A split object's public address is data[0],
// so a split group must be reverse-mapped and all six segments inspected.
// Ordinary whole-object addresses intentionally consult only their own
// endpoint.  If a supposedly split address is owned by the manager but its
// group metadata is gone or inconsistent, fail closed and require a rewrite.
template <typename StripeManager, typename DeadAddressPredicate>
inline bool backup_endpoint_is_dead(const StripeManager &manager,
                                    std::uint64_t remote_addr,
                                    bool split_object,
                                    DeadAddressPredicate dead_address) {
    if (split_object && manager.owns(remote_addr)) {
        typename StripeManager::SlotLayout layout;
        typename StripeManager::SlotGroupHandle group;
        if (!manager.slot_group_is_split(remote_addr) ||
            !manager.get_slot_layout(remote_addr, &layout)) {
            return true;
        }
        const typename StripeManager::SlotGroupId group_id{
            layout.stripe_id, layout.slot_id};
        if (!manager.get_slot_group_layout(group_id, &group)) return true;
        for (const auto &segment : group.segments) {
            if (dead_address(segment.addr)) return true;
        }
        return false;
    }
    return dead_address(remote_addr);
}

// Clean-backup reuse is legal only when the policy is enabled, the object is
// clean, the remote address is still valid, and its reservation is present.
// Keeping this as a constexpr predicate makes the no-write guard testable
// without constructing ConcurrentArrayCache or an RDMA completion path.
constexpr bool can_reuse_clean_backup(bool policy_enabled, bool dirty,
                                      bool valid_remote,
                                      bool has_reservation) noexcept {
    return policy_enabled && !dirty && valid_remote && has_reservation;
}

}  // namespace FarLib::cache::ec_backup_policy
