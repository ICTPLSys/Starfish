#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

using CountMap = std::unordered_map<std::string, uint64_t>;

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
        if (st.st_size < 0) {
            throw std::runtime_error("negative file size");
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

struct WordCountResult {
    uint64_t total_words{0};
    uint64_t too_long_words{0};
    uint64_t max_word_len{0};
    CountMap counts;
};

bool is_word_char(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

char lower_ascii(unsigned char c) {
    if (c >= 'A' && c <= 'Z') {
        return static_cast<char>(c - 'A' + 'a');
    }
    return static_cast<char>(c);
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

size_t env_size(const char *name, size_t fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::strtoull(value, nullptr, 0);
}

WordCountResult count_chunk(const char *data, Chunk chunk, size_t max_word_bytes) {
    WordCountResult result;
    std::string word;
    word.reserve(64);

    size_t pos = chunk.begin;
    while (pos < chunk.end) {
        while (pos < chunk.end &&
               !is_word_char(static_cast<unsigned char>(data[pos]))) {
            ++pos;
        }
        word.clear();
        size_t len = 0;
        while (pos < chunk.end &&
               is_word_char(static_cast<unsigned char>(data[pos]))) {
            if (max_word_bytes == 0 || len < max_word_bytes) {
                word.push_back(lower_ascii(static_cast<unsigned char>(data[pos])));
            }
            ++len;
            ++pos;
        }
        result.max_word_len = std::max<uint64_t>(result.max_word_len, len);
        if (max_word_bytes != 0 && len > max_word_bytes) {
            ++result.too_long_words;
        } else if (!word.empty()) {
            ++result.counts[word];
            ++result.total_words;
        }
    }
    return result;
}

void merge_counts(CountMap &dst, CountMap &src) {
    for (auto &entry : src) {
        dst[entry.first] += entry.second;
    }
}

uint64_t fnv1a_bytes(uint64_t hash, const void *ptr, size_t len) {
    const auto *bytes = static_cast<const unsigned char *>(ptr);
    for (size_t i = 0; i < len; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t checksum_counts(const std::vector<std::pair<std::string, uint64_t>> &items) {
    uint64_t hash = 1469598103934665603ull;
    for (const auto &item : items) {
        hash = fnv1a_bytes(hash, item.first.data(), item.first.size());
        const char sep = '\0';
        hash = fnv1a_bytes(hash, &sep, sizeof(sep));
        uint64_t count = item.second;
        hash = fnv1a_bytes(hash, &count, sizeof(count));
    }
    return hash;
}

void print_usage(const char *argv0) {
    std::cerr << "usage: " << argv0
              << " <dataset> [threads] [max_bytes] [top_k]\n";
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

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
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
        size_t top_k = 20;
        if (argc >= 5) {
            top_k = parse_size_arg(argv[4], "top_k");
        }
        size_t max_word_bytes = env_size("WORDCOUNT_MAX_WORD_BYTES", 0);
        if (threads == 0) {
            throw std::runtime_error("threads must be > 0");
        }

        MappedFile file(path);
        size_t size = file.size;
        if (max_bytes != 0) {
            size = std::min(size, max_bytes);
        }

        auto start = std::chrono::steady_clock::now();
        auto chunks = make_chunks(file.data, size, threads);
        std::vector<WordCountResult> partials(chunks.size());
        std::vector<std::thread> workers;
        workers.reserve(chunks.size());
        for (size_t i = 0; i < chunks.size(); ++i) {
            workers.emplace_back([&, i] {
                partials[i] =
                    count_chunk(file.data, chunks[i], max_word_bytes);
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }

        CountMap global;
        uint64_t total_words = 0;
        uint64_t too_long_words = 0;
        uint64_t max_word_len = 0;
        for (auto &partial : partials) {
            total_words += partial.total_words;
            too_long_words += partial.too_long_words;
            max_word_len = std::max(max_word_len, partial.max_word_len);
            merge_counts(global, partial.counts);
        }

        std::vector<std::pair<std::string, uint64_t>> sorted;
        sorted.reserve(global.size());
        for (auto &entry : global) {
            sorted.emplace_back(entry.first, entry.second);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto &a, const auto &b) {
                      return a.first < b.first;
                  });
        uint64_t checksum = checksum_counts(sorted);

        std::vector<std::pair<std::string_view, uint64_t>> top;
        top.reserve(sorted.size());
        for (const auto &entry : sorted) {
            top.emplace_back(std::string_view(entry.first), entry.second);
        }
        std::sort(top.begin(), top.end(),
                  [](const auto &a, const auto &b) {
                      if (a.second != b.second) {
                          return a.second > b.second;
                      }
                      return a.first < b.first;
                  });

        auto stop = std::chrono::steady_clock::now();
        double seconds =
            std::chrono::duration<double>(stop - start).count();

        std::cout << "wordcount_native"
                  << " file=" << path
                  << " bytes=" << size
                  << " threads=" << chunks.size()
                  << " total_words=" << total_words
                  << " unique_words=" << sorted.size()
                  << " checksum=0x" << std::hex << checksum << std::dec
                  << " max_word_bytes=" << max_word_bytes
                  << " too_long_words=" << too_long_words
                  << " max_word_len=" << max_word_len
                  << " elapsed_s=" << seconds
                  << " words_s=" << (seconds > 0 ? total_words / seconds : 0)
                  << "\n";

        size_t printed = std::min(top_k, top.size());
        for (size_t i = 0; i < printed; ++i) {
            std::cout << "top" << (i + 1)
                      << " word=" << top[i].first
                      << " count=" << top[i].second
                      << "\n";
        }
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
