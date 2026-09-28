#pragma once

#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

// Capture this before runtime_init() starts and pins libfibre workers.  The
// NQ graph loader uses native std::threads; each one receives only the
// launcher's affinity intersected with FibreCpuSet.  Fibre workers are never
// widened or otherwise modified by this helper.
class GraphLoadAffinity {
public:
    GraphLoadAffinity() {
        CPU_ZERO(&launch_cpus_);
        if (sched_getaffinity(0, sizeof(launch_cpus_), &launch_cpus_) != 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "capture graph-loader launch affinity");
        }

        target_cpus_ = launch_cpus_;
        const char *raw = std::getenv("FibreCpuSet");
        if (raw != nullptr && raw[0] != '\0') {
            cpu_set_t requested;
            CPU_ZERO(&requested);
            parse_cpu_list(raw, requested);
            CPU_AND(&target_cpus_, &launch_cpus_, &requested);
        }
        if (CPU_COUNT(&target_cpus_) == 0) {
            throw std::invalid_argument(
                "FibreCpuSet has no CPUs in the graph-loader launch mask");
        }

        log("capture", -1, 0, launch_cpus_, target_cpus_);
    }

    void apply(const char *phase, int shard) const {
        const int set_rc = pthread_setaffinity_np(
            pthread_self(), sizeof(target_cpus_), &target_cpus_);

        cpu_set_t actual;
        CPU_ZERO(&actual);
        const int get_rc = sched_getaffinity(0, sizeof(actual), &actual) == 0
                               ? 0
                               : errno;
        log(phase, shard, set_rc != 0 ? set_rc : get_rc, target_cpus_, actual);

        if (set_rc != 0) {
            throw std::system_error(set_rc, std::generic_category(),
                                    "set graph-loader affinity");
        }
        if (get_rc != 0) {
            throw std::system_error(get_rc, std::generic_category(),
                                    "read graph-loader affinity");
        }

        // Do not silently change the approved placement after cgroup/hotplug
        // changes. This also rejects any CPU outside the captured intersection.
        if (!CPU_EQUAL(&actual, &target_cpus_)) {
            throw std::runtime_error(
                "graph-loader affinity differs from the requested CPU mask");
        }
    }

    std::string target_cpu_list() const { return cpu_list(target_cpus_); }

private:
    static unsigned parse_number(std::string_view value) {
        if (value.empty()) throw std::invalid_argument("empty CPU in FibreCpuSet");
        unsigned result = 0;
        for (const char c : value) {
            if (c < '0' || c > '9')
                throw std::invalid_argument("invalid CPU in FibreCpuSet");
            const unsigned digit = static_cast<unsigned>(c - '0');
            if (result > (static_cast<unsigned>(CPU_SETSIZE - 1) - digit) / 10)
                throw std::invalid_argument("CPU is out of range in FibreCpuSet");
            result = result * 10 + digit;
        }
        if (result >= CPU_SETSIZE)
            throw std::invalid_argument("CPU is out of range in FibreCpuSet");
        return result;
    }

    static void parse_cpu_list(const char *raw, cpu_set_t &result) {
        const std::string text(raw);
        std::size_t begin = 0;
        bool any = false;
        while (begin < text.size()) {
            std::size_t end = text.find(',', begin);
            if (end == std::string::npos) end = text.size();
            std::string_view item(text.data() + begin, end - begin);
            while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
                item.remove_prefix(1);
            while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
                item.remove_suffix(1);
            if (item.empty())
                throw std::invalid_argument("empty item in FibreCpuSet");

            const std::size_t dash = item.find('-');
            const unsigned first = parse_number(
                item.substr(0, dash == std::string_view::npos ? item.size() : dash));
            const unsigned last = dash == std::string_view::npos
                                      ? first
                                      : parse_number(item.substr(dash + 1));
            if (last < first)
                throw std::invalid_argument("descending range in FibreCpuSet");
            for (unsigned cpu = first; cpu <= last; ++cpu) CPU_SET(cpu, &result);
            any = true;
            begin = end == text.size() ? text.size() : end + 1;
        }
        if (!any) throw std::invalid_argument("empty FibreCpuSet");
    }

    static std::string cpu_list(const cpu_set_t &mask) {
        std::string result;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &mask)) continue;
            if (!result.empty()) result += ',';
            result += std::to_string(cpu);
        }
        return result;
    }

    static long tid() {
#ifdef SYS_gettid
        return static_cast<long>(syscall(SYS_gettid));
#else
        return static_cast<long>(::getpid());
#endif
    }

    static void log(const char *phase, int shard, int rc,
                    const cpu_set_t &requested, const cpu_set_t &actual) {
        std::lock_guard<std::mutex> lock(log_mutex());
        std::cerr << "nq_graph_loader_affinity"
                  << " phase=" << phase
                  << " shard=" << shard
                  << " tid=" << tid()
                  << " requested=" << cpu_list(requested)
                  << " actual=" << cpu_list(actual)
                  << " rc=" << rc << std::endl;
    }

    static std::mutex &log_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    cpu_set_t launch_cpus_{};
    cpu_set_t target_cpus_{};
};
