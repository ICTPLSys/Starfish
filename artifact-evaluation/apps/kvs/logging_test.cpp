#include "logging.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "kvs_logging_test failure: %s\n", message);
        std::abort();
    }
}

std::string read_file(FILE* file) {
    std::rewind(file);
    std::string output;
    char buffer[256];
    while (const size_t count = std::fread(buffer, 1, sizeof(buffer), file))
        output.append(buffer, count);
    return output;
}

void test_verbose_switch() {
    unsetenv("FARLIB_VERBOSE_LOG");
    require(!kvs_logging::verbose_enabled(), "unset verbose switch");
    setenv("FARLIB_VERBOSE_LOG", "1", 1);
    require(kvs_logging::verbose_enabled(), "enabled verbose switch");
    setenv("FARLIB_VERBOSE_LOG", "0", 1);
    require(!kvs_logging::verbose_enabled(), "disabled verbose switch");
    unsetenv("FARLIB_VERBOSE_LOG");
}

void test_complete_records() {
    FILE* file = std::tmpfile();
    require(file != nullptr, "create record file");
    kvs_logging::write_record(file, "kvs_phase name=direct event=request_start value=", 7);
    require(read_file(file) ==
                "kvs_phase name=direct event=request_start value=7\n",
            "complete record contents");
    setenv("FARLIB_VERBOSE_LOG", "0", 1);
    kvs_logging::write_verbose_record(file, "kvs_verbose hidden=1");
    require(read_file(file) ==
                "kvs_phase name=direct event=request_start value=7\n",
            "quiet verbose record");
    setenv("FARLIB_VERBOSE_LOG", "1", 1);
    kvs_logging::write_verbose_record(file, "kvs_verbose visible=1");
    require(read_file(file) ==
                "kvs_phase name=direct event=request_start value=7\n"
                "kvs_verbose visible=1\n",
            "enabled verbose record");
    unsetenv("FARLIB_VERBOSE_LOG");
    std::fclose(file);
}

void test_concurrent_records() {
    FILE* file = std::tmpfile();
    require(file != nullptr, "create concurrent record file");
    constexpr size_t threads = 8;
    constexpr size_t records_per_thread = 32;
    std::vector<std::thread> writers;
    for (size_t tid = 0; tid < threads; ++tid) {
        writers.emplace_back([file, tid] {
            for (size_t seq = 0; seq < records_per_thread; ++seq) {
                if ((tid + seq) & 1) {
                    kvs_logging::write_record(file, "kvs_receipt tid=", tid,
                                              " seq=", seq, " accepted=1");
                } else {
                    std::fprintf(file, "kvs_receipt tid=%zu seq=%zu accepted=1\n",
                                 tid, seq);
                    std::fflush(file);
                }
            }
        });
    }
    for (auto& writer : writers) writer.join();
    const std::string output = read_file(file);
    std::set<std::pair<size_t, size_t>> seen;
    size_t lines = 0;
    size_t begin = 0;
    while (begin < output.size()) {
        const size_t end = output.find('\n', begin);
        require(end != std::string::npos, "newline-terminated record");
        const std::string line = output.substr(begin, end - begin);
        size_t tid = 0, seq = 0;
        int consumed = 0;
        const int matched = std::sscanf(
            line.c_str(), "kvs_receipt tid=%zu seq=%zu accepted=1%n", &tid, &seq,
            &consumed);
        require(matched == 2 && consumed == static_cast<int>(line.size()),
                "exact concurrent record fields");
        require(tid < threads && seq < records_per_thread,
                "concurrent record range");
        require(seen.emplace(tid, seq).second, "unique concurrent record");
        ++lines;
        begin = end + 1;
    }
    require(lines == threads * records_per_thread, "concurrent record count");
    require(seen.size() == threads * records_per_thread,
            "all concurrent records present");
    std::fclose(file);
}

}  // namespace

int main() {
    test_verbose_switch();
    test_complete_records();
    test_concurrent_records();
    std::puts("kvs_logging_test status=pass");
}
