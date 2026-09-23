# Design 1 runtime code

This directory contains only the Region-level Design 1 policy and its direct
runtime integration. Generic cache, GC, RDMA, and Region-list mechanisms stay
outside this directory.

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

## Local style

- Types use `PascalCase`; functions and variables use `snake_case`; constants
  use `kPascalCase`.
- Quantities carry units in their names: `_bytes`, `_byte_work`, `_count`,
  `_cycles`, or `_pct`.
- Pure policy functions take explicit inputs and return explicit results.
- Runtime methods separate decision, execution, accounting, and diagnostics.
- New functions should normally remain below 100 lines.
- Functional atomics document why acquire/release ordering is required;
  statistics use relaxed ordering.
- Existing configuration names and output keys remain stable for experiment
  reproducibility.

## Tests

Pure policy tests must not require RDMA hardware. Runtime integration tests
must cover requested-versus-actual placement, backup reconciliation, Region
reclassification accounting, and shutdown ownership invariants.
