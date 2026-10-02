#pragma once
#include <sched.h>
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace FarLib::uthread {
inline std::vector<unsigned> background_cpu_assignment(
    const char *base_text, const char *list_text, size_t workers) {
    if (workers == 0 || workers > CPU_SETSIZE)
        throw std::invalid_argument("invalid background worker count");
    std::vector<unsigned> cpus;
    cpus.reserve(workers);
    if (list_text != nullptr) {
        // An explicit, ordered list can avoid application cores on both NUMA nodes.
        std::string_view list(list_text);
        if (list.empty())
            throw std::invalid_argument("empty FARLIB_BACKGROUND_CPU_LIST");
        const char *next = list.data();
        const char *end = next + list.size();
        cpu_set_t seen;
        CPU_ZERO(&seen);
        while (next != end) {
            unsigned cpu = 0;
            const auto parsed = std::from_chars(next, end, cpu);
            if (parsed.ec != std::errc{} || parsed.ptr == next ||
                cpu >= CPU_SETSIZE || CPU_ISSET(cpu, &seen) ||
                cpus.size() == workers)
                throw std::invalid_argument("invalid FARLIB_BACKGROUND_CPU_LIST");
            CPU_SET(cpu, &seen);
            cpus.push_back(cpu);
            if (parsed.ptr == end) break;
            if (*parsed.ptr != ',' || parsed.ptr + 1 == end)
                throw std::invalid_argument("invalid FARLIB_BACKGROUND_CPU_LIST");
            next = parsed.ptr + 1;
        }
        if (cpus.size() != workers)
            throw std::invalid_argument("background CPU list must match worker count");
        return cpus;
    }
    if (base_text == nullptr || base_text[0] == '\0')
        throw std::invalid_argument("FARLIB_BACKGROUND_CPU_BASE is required");
    char *end = nullptr;
    const auto base = std::strtoul(base_text, &end, 10);
    if (end == base_text || *end != '\0' || base >= CPU_SETSIZE ||
        workers > CPU_SETSIZE - base)
        throw std::invalid_argument("invalid FARLIB_BACKGROUND_CPU_BASE");
    for (size_t i = 0; i < workers; ++i)
        cpus.push_back(static_cast<unsigned>(base + i));
    return cpus;
}
}  // namespace FarLib::uthread
