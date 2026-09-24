# Selective remote backup

- `batched_backup_budget.hpp` controls admission against the remote-backup
  byte budget. Its fibre-local and native-thread-local handles are defined in
  `src/design1/backup_budget.cpp`.
- `backup_usage_shards.hpp` collects backup-usage counters.
- `placement_policy.hpp` and `runtime_types.hpp` define placement decisions
  and state.
- `placement_planner.ipp`, `fetch_placement.ipp`, and `diagnostics.ipp` integrate
  those decisions with the cache and remote-backup ownership.

The `.ipp` files define inline `ConcurrentArrayCache` methods and are included
by `cache/concurrent_cache.hpp`.
