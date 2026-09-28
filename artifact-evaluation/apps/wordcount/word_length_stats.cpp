#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

struct MappedFile {
    int fd{-1};
    const char *data{nullptr};
    size_t size{0};

    explicit MappedFile(const char *path) {
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(std::string("open failed: ") + strerror(errno));
        }
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            throw std::runtime_error(std::string("fstat failed: ") + strerror(errno));
        }
        size = static_cast<size_t>(st.st_size);
        if (size == 0) {
            return;
        }
        void *mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
        }
        data = static_cast<const char *>(mapped);
    }

    ~MappedFile() {
        if (data != nullptr) {
            munmap(const_cast<char *>(data), size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
};

struct Chunk {
    size_t begin{0};
    size_t end{0};
};

struct Stats {
    uint64_t total{0};
    uint64_t max_len{0};
    uint64_t over_64{0};
    uint64_t over_128{0};
    uint64_t over_256{0};
    uint64_t over_512{0};
    uint64_t over_1024{0};
    std::vector<uint64_t> hist;

    explicit Stats(size_t hist_size = 4097) : hist(hist_size, 0) {}

    void add_len(size_t len) {
        ++total;
        max_len = std::max<uint64_t>(max_len, len);
        over_64 += len > 64;
        over_128 += len > 128;
        over_256 += len > 256;
        over_512 += len > 512;
        over_1024 += len > 1024;
        if (len >= hist.size()) {
            ++hist.back();
        } else {
            ++hist[len];
        }
    }

    void merge(const Stats &other) {
        total += other.total;
        max_len = std::max(max_len, other.max_len);
        over_64 += other.over_64;
        over_128 += other.over_128;
        over_256 += other.over_256;
        over_512 += other.over_512;
        over_1024 += other.over_1024;
        for (size_t i = 0; i < hist.size(); ++i) {
            hist[i] += other.hist[i];
        }
    }
};

bool is_word_char(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

size_t advance_to_delimiter(const char *data, size_t pos, size_t limit) {
    while (pos < limit && is_word_char(static_cast<unsigned char>(data[pos]))) {
        ++pos;
    }
    return pos;
}

std::vector<Chunk> make_chunks(const char *data, size_t size, size_t threads) {
    std::vector<Chunk> chunks;
    if (size == 0 || threads == 0) {
        return chunks;
    }
    threads = std::min(threads, size);
    chunks.reserve(threads);
    size_t begin = 0;
    for (size_t i = 0; i < threads; ++i) {
        size_t end = (i + 1 == threads) ? size : ((i + 1) * size / threads);
        if (i + 1 != threads) {
            end = advance_to_delimiter(data, end, size);
        }
        chunks.push_back({begin, end});
        begin = end;
    }
    return chunks;
}

Stats scan_chunk(const char *data, Chunk chunk) {
    Stats stats;
    size_t pos = chunk.begin;
    while (pos < chunk.end) {
        while (pos < chunk.end &&
               !is_word_char(static_cast<unsigned char>(data[pos]))) {
            ++pos;
        }
        size_t len = 0;
        while (pos < chunk.end &&
               is_word_char(static_cast<unsigned char>(data[pos]))) {
            ++len;
            ++pos;
        }
        if (len != 0) {
            stats.add_len(len);
        }
    }
    return stats;
}

size_t parse_size_arg(const char *text, const char *name) {
    char *end = nullptr;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<size_t>(value);
}

uint64_t percentile_len(const Stats &stats, long double percentile) {
    if (stats.total == 0) {
        return 0;
    }
    long double target_f =
        percentile * static_cast<long double>(stats.total) / 100.0L;
    uint64_t target = static_cast<uint64_t>(target_f);
    if (static_cast<long double>(target) < target_f) {
        ++target;
    }
    if (target == 0) {
        target = 1;
    }
    uint64_t cumulative = 0;
    for (size_t len = 0; len < stats.hist.size(); ++len) {
        cumulative += stats.hist[len];
        if (cumulative >= target) {
            return len;
        }
    }
    return stats.max_len;
}

void print_usage(const char *argv0) {
    std::cerr << "usage: " << argv0 << " <dataset> [threads] [max_bytes]\n";
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        print_usage(argv[0]);
        return 2;
    }
    try {
        const char *path = argv[1];
        size_t threads = std::thread::hardware_concurrency();
        if (threads == 0) {
            threads = 1;
        }
        if (argc >= 3) {
            threads = parse_size_arg(argv[2], "threads");
        }
        size_t max_bytes = 0;
        if (argc >= 4) {
            max_bytes = parse_size_arg(argv[3], "max_bytes");
        }

        MappedFile file(path);
        size_t size = file.size;
        if (max_bytes != 0) {
            size = std::min(size, max_bytes);
        }
        auto chunks = make_chunks(file.data, size, threads);
        std::vector<Stats> partials;
        partials.reserve(chunks.size());
        for (size_t i = 0; i < chunks.size(); ++i) {
            partials.emplace_back();
        }
        std::vector<std::thread> workers;
        workers.reserve(chunks.size());

        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < chunks.size(); ++i) {
            workers.emplace_back([&, i] {
                partials[i] = scan_chunk(file.data, chunks[i]);
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }
        Stats total;
        for (const auto &partial : partials) {
            total.merge(partial);
        }
        auto stop = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(stop - start).count();

        std::cout << "word_length_stats file=" << path
                  << " bytes=" << size
                  << " threads=" << chunks.size()
                  << " total_words=" << total.total
                  << " p50=" << percentile_len(total, 50.0L)
                  << " p90=" << percentile_len(total, 90.0L)
                  << " p99=" << percentile_len(total, 99.0L)
                  << " p99_9=" << percentile_len(total, 99.9L)
                  << " p99_99=" << percentile_len(total, 99.99L)
                  << " p99_999=" << percentile_len(total, 99.999L)
                  << " max_len=" << total.max_len
                  << " over_64=" << total.over_64
                  << " over_128=" << total.over_128
                  << " over_256=" << total.over_256
                  << " over_512=" << total.over_512
                  << " over_1024=" << total.over_1024
                  << " elapsed_s=" << elapsed_s
                  << "\n";
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
