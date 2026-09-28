#pragma once

#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <string>
#include <system_error>

// Capture this in main(), BEFORE runtime_init pins libfibre's OS workers.
// Native graph-loader threads otherwise inherit one worker's single-CPU mask.
// Only the native loader threads apply this mask: never widen a fibre worker,
// and never infer a machine-wide mask that would escape the launcher's cpuset.
class GraphLoadAffinity {
public:
    GraphLoadAffinity() {
        CPU_ZERO(&cpus_);
        if (sched_getaffinity(0, sizeof(cpus_), &cpus_) != 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "capture graph-loader launch affinity");
        }
    }

    int apply_to_current_thread() const {
        const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpus_),
                                              &cpus_);
        if (rc != 0) return rc;
        cpu_set_t effective;
        CPU_ZERO(&effective);
        if (sched_getaffinity(0, sizeof(effective), &effective) != 0) return errno;
        // Do not silently claim parallel loading after a cpuset/hotplug change.
        return CPU_EQUAL(&effective, &cpus_) ? 0 : EINVAL;
    }

    int cpu_count() const { return CPU_COUNT(&cpus_); }

    std::string cpu_list() const {
        std::string result;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &cpus_)) continue;
            if (!result.empty()) result += ',';
            result += std::to_string(cpu);
        }
        return result;
    }

private:
    cpu_set_t cpus_;
};
