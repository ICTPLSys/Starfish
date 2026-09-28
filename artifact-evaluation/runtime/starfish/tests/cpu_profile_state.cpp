// Standalone CPU fixtures do not link stats.cpp or construct a runtime.
// The optimized allocator's optional contention observer still references
// these phase flags. Keep the observer dormant without stubbing any allocator,
// codec, completion, or protection behavior under test.
#include <atomic>
#include <cstdint>

namespace FarLib::profile {
std::atomic_bool work_phase_active{false};
std::atomic<uint8_t> evac_phase_active_mask{0};
}
