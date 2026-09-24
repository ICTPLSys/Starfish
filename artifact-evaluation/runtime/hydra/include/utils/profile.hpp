#include "async/stream_runner.hpp"
#include "stats.hpp"

// Print remote allocator usage when available
#include "cache/region_based_allocator.hpp"
#include "cache/alloc/remote_allocator.hpp"

namespace FarLib {

namespace profile {

template <typename Fn>
inline void beehive_profile(Fn &&fn) {
    async::StreamRunnerProfiler::reset();
    profile::reset_all();
    profile::start_work();
    // profile::thread_start_work();
    fn();
    // profile::thread_end_work();
    profile::end_work();
    profile::print_profile_data();

    FarLib::allocator::global_heap.print_used_memory();
#ifdef REMOTE_REGION_ALLOCATOR
    FarLib::allocator::remote::remote_global_heap.print_used_memory();
#endif

    std::cout << "stream runner: total cycles = "
              << async::StreamRunnerProfiler::get_app_cycles() << std::endl;
    std::cout << "stream runner: schedule cycles = "
              << async::StreamRunnerProfiler::get_sched_cycles() << std::endl;
}
}  // namespace profile
}  // namespace FarLib
