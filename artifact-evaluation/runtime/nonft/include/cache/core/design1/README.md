# Design 1 runtime code

This directory contains Region-level placement policy and its direct runtime
integration. The Non-FT recipe disables selective backup by design.

## Data flow

1. Each object stores a stable behavior-group id.
2. Demand accesses, fetches, and evictions update that group's profile.
3. `placement_policy.hpp` ranks immutable group snapshots. It must not read
   global configuration, mutate runtime state, or emit diagnostics.
4. `placement_planner.ipp` publishes group R/S targets and asks GlobalHeap to
   reclassify whole Regions.
5. `fetch_placement.ipp` converts a group target into a requested placement,
   records the allocator's actual placement, and reconciles remote backup
   ownership.

## Invariants

- A requested placement is a preference; the allocator's actual placement is
  authoritative.
- Only actual Resident occupancy contributes to a group's current R bytes.
- A Resident object must not retain a remote-valid streaming backup.
- Diagnostics may observe policy state but must never enable or disable a
  placement action.
- Region snapshots are hints. A consumer must revalidate placement and epoch
  while holding the Region placement lock before changing a Region.
- Physical Region reclassification and object-level fetch placement use
  different names. Do not call both operations promotion or demotion.
