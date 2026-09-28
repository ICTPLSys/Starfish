#pragma once

namespace FarLib::behavior_group_runtime {

// Written only by runtime_init before it starts workers. Immutable throughout
// the runtime lifetime; no environment mutation or hot-path atomic counter.
inline bool active = false;
inline void configure(bool enabled) noexcept { active = enabled; }
inline bool enabled() noexcept { return active; }

}  // namespace FarLib::behavior_group_runtime
