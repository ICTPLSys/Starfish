#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

namespace kvs_logging {

// Verbose output is deliberately opt-in so benchmark stdout remains a
// machine-readable stream of configuration, profile, and receipt records.
inline bool verbose_enabled() {
    const char* value = std::getenv("FARLIB_VERBOSE_LOG");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

template <typename... Args>
std::string format_record(const Args&... args) {
    std::ostringstream line;
    (line << ... << args);
    return line.str();
}

// Build the complete line before taking the stdio lock. One fwrite keeps
// concurrent diagnostic writers from interleaving the fields of a record.
template <typename... Args>
void write_record(FILE* stream, const Args&... args) {
    std::string record = format_record(args...);
    record.push_back('\n');
    std::fwrite(record.data(), 1, record.size(), stream);
    std::fflush(stream);
}

template <typename... Args>
void write_verbose_record(FILE* stream, const Args&... args) {
    if (verbose_enabled()) write_record(stream, args...);
}

}  // namespace kvs_logging
