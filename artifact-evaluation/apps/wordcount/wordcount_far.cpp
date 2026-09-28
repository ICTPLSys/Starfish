#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
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

#include "cache/alloc/region_remote_allocator.hpp"
#include "cache/cache.hpp"
#include "data_structure/concurrent_hashmap.hpp"
#include "rdma/config.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/parallel.hpp"
#include "utils/spinlock.hpp"
#include "utils/stats.hpp"

#ifndef FARLIB_WORDCOUNT_STARFISH
#define FARLIB_WORDCOUNT_STARFISH 0
#endif
#ifndef FARLIB_WORDCOUNT_RECOVERY_TEST
#define FARLIB_WORDCOUNT_RECOVERY_TEST 0
#endif
#if FARLIB_WORDCOUNT_RECOVERY_TEST && !FARLIB_WORDCOUNT_STARFISH
#error "Recovery validation requires the Starfish Wordcount build"
#endif

namespace {

constexpr size_t kMaxWordBytes = 512;
constexpr size_t kGlobalLockCount = 4096;

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

struct FixedWord {
    uint16_t len{0};
    std::array<char, kMaxWordBytes> bytes{};

    std::string_view view() const { return {bytes.data(), len}; }

    bool operator==(const FixedWord &other) const {
        return len == other.len &&
               std::memcmp(bytes.data(), other.bytes.data(), len) == 0;
    }
};

struct FixedWordHash {
    size_t operator()(const FixedWord &word) const {
        uint64_t hash = 1469598103934665603ull;
        for (uint16_t i = 0; i < word.len; ++i) {
            hash ^= static_cast<unsigned char>(word.bytes[i]);
            hash *= 1099511628211ull;
        }
        return static_cast<size_t>(hash);
    }
};

struct CountValue {
    uint64_t count{0};
};

// All runtimes share counting; only Starfish records a recomputation locator.
struct NativeCount {
    uint64_t count{0};
#if FARLIB_WORDCOUNT_STARFISH
    uint64_t first_offset{0};
#endif
};

using NativeCountMap =
    std::unordered_map<FixedWord, NativeCount, FixedWordHash>;

using FarWordMap = FarLib::ConcurrentHashMap<FixedWord, CountValue, FixedWordHash>;

#if FARLIB_WORDCOUNT_STARFISH
void mark_wordcount_recomputable(FarLib::far_obj_t obj) {
    ASSERT(FarLib::mark_recomputable(obj));
    ASSERT(FarLib::is_recomputable(obj));
}
#endif

struct PhaseStats {
    int64_t rdma_read_bytes;
    int64_t rdma_write_bytes;
};

struct ThreadStats {
    uint64_t total_words{0};
    uint64_t too_long_words{0};
    uint64_t max_word_len{0};
    uint64_t native_unique_words{0};
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

size_t parse_size_arg(const char *text, const char *name) {
    char *end = nullptr;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<size_t>(value);
}

PhaseStats collect_phase_stats() {
    return {FarLib::profile::collect_rdma_read_post_bytes(),
            FarLib::profile::collect_rdma_write_post_bytes()};
}

PhaseStats operator-(const PhaseStats &after, const PhaseStats &before) {
    return {after.rdma_read_bytes - before.rdma_read_bytes,
            after.rdma_write_bytes - before.rdma_write_bytes};
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

void fill_payload(CountValue &value, const FixedWord &word) {
    (void)value;
    (void)word;
}

bool read_word(const char *data, size_t &pos, size_t end, FixedWord &word,
               ThreadStats &stats, size_t *word_offset = nullptr) {
    while (pos < end && !is_word_char(static_cast<unsigned char>(data[pos]))) {
        ++pos;
    }
    if (pos >= end) {
        return false;
    }

    const size_t begin = pos;
    if (word_offset != nullptr) {
        *word_offset = begin;
    }

    word = {};
    size_t len = 0;
    while (pos < end && is_word_char(static_cast<unsigned char>(data[pos]))) {
        if (len < kMaxWordBytes) {
            word.bytes[len] = lower_ascii(static_cast<unsigned char>(data[pos]));
        }
        ++len;
        ++pos;
    }
    stats.max_word_len = std::max<uint64_t>(stats.max_word_len, len);
    if (len > kMaxWordBytes) {
        ++stats.too_long_words;
        return false;
    }
    word.len = static_cast<uint16_t>(len);
    ++stats.total_words;
    return true;
}

CountValue make_count_value(const FixedWord &word, uint64_t count) {
    CountValue value{};
    value.count = count;
    fill_payload(value, word);
    return value;
}

void add_to_map(FarWordMap &map, const FixedWord &word, uint64_t delta,
                FarLib::DereferenceScope &scope) {
    CountValue value{};
    uint64_t count = delta;
    if (map.get(word, &value, scope)) {
        count += value.count;
    }
    ASSERT(map.put(word, make_count_value(word, count), scope));
}

void count_chunk_native(NativeCountMap &counts, const char *data, Chunk chunk,
                        ThreadStats &stats, bool cooperative_yield = false) {
    FixedWord word;
    size_t pos = chunk.begin;
#if FARLIB_WORDCOUNT_STARFISH
    size_t words_since_yield = 0;
#endif
    while (pos < chunk.end) {
#if FARLIB_WORDCOUNT_STARFISH
        size_t word_offset = 0;
        if (read_word(data, pos, chunk.end, word, stats, &word_offset)) {
            auto [it, inserted] = counts.try_emplace(
                word, NativeCount{1, static_cast<uint64_t>(word_offset)});
            if (!inserted) ++it->second.count;
        }
        if (cooperative_yield && ++words_since_yield == (1u << 16)) {
            words_since_yield = 0;
            FarLib::uthread::yield();
        }
#else
        if (read_word(data, pos, chunk.end, word, stats)) ++counts[word].count;
#endif
    }
    stats.native_unique_words = counts.size();
}

#if FARLIB_WORDCOUNT_RECOVERY_TEST
std::atomic<uint64_t> recipe_restored_objects{0};
std::atomic<uint64_t> recipe_partition_rebuilds{0};
std::atomic<uint64_t> recipe_failures{0};
#endif

#if FARLIB_WORDCOUNT_STARFISH
// This is the only compute-side state retained by a recipe.  It owns the
// immutable mapped input and partition bounds, and lazily materializes a
// derived count cache only when a lost object actually invokes the callback.
struct WordRecipeInput {
    std::shared_ptr<const MappedFile> file;
    Chunk chunk;
    size_t max_word_bytes{0};
    mutable std::atomic<uint8_t> cache_state{0}; // empty, building, ready, failed
    mutable std::unique_ptr<NativeCountMap> cached_counts;

    WordRecipeInput(std::shared_ptr<const MappedFile> file, Chunk chunk,
                    size_t max_word_bytes)
        : file(std::move(file)), chunk(chunk), max_word_bytes(max_word_bytes) {}

    const NativeCountMap *ensure_counts() const {
        uint8_t expected = 0;
        if (cache_state.compare_exchange_strong(
                expected, static_cast<uint8_t>(1),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            try {
                auto built = std::make_unique<NativeCountMap>();
                built->reserve(1 << 20);
                ThreadStats stats;
                count_chunk_native(*built, file->data, chunk, stats, true);
                cached_counts = std::move(built);
#if FARLIB_WORDCOUNT_RECOVERY_TEST
                recipe_partition_rebuilds.fetch_add(1,
                                                    std::memory_order_relaxed);
#endif
                cache_state.store(2, std::memory_order_release);
            } catch (...) {
                cache_state.store(3, std::memory_order_release);
            }
        } else {
            while (expected == 1) {
                FarLib::uthread::yield();
                expected = cache_state.load(std::memory_order_acquire);
            }
        }
        return cache_state.load(std::memory_order_acquire) == 2
                   ? cached_counts.get()
                   : nullptr;
    }
};

using RecipeInputs = decltype(
    FarLib::recompute::Inputs::immutable_compute_owned(
        std::declval<std::shared_ptr<const WordRecipeInput>>()));

struct RecipeBuildHookContext {
    std::shared_ptr<const WordRecipeInput> input;
    RecipeInputs inputs;
    std::atomic<const NativeCountMap *> native_counts{nullptr};

    explicit RecipeBuildHookContext(std::shared_ptr<const WordRecipeInput> input)
        : input(std::move(input)),
          inputs(FarLib::recompute::Inputs::immutable_compute_owned(this->input)) {}
};

bool rebuild_wordcount_kv(void *dst, size_t bytes, const void *ctx,
                          uint64_t arg);

void register_wordcount_recipe(FarLib::far_obj_t obj,
                               FarLib::DereferenceScope &scope,
                               const void *raw_context, uint64_t arg) {
    auto *build = static_cast<const RecipeBuildHookContext *>(raw_context);
    if (build == nullptr || build->input == nullptr ||
        build->native_counts.load(std::memory_order_acquire) == nullptr) {
#if FARLIB_WORDCOUNT_RECOVERY_TEST
        recipe_failures.fetch_add(1, std::memory_order_relaxed);
#endif
        ASSERT(false && "invalid wordcount recipe build context");
        std::abort();
    }
    ASSERT(scope.mark_recomputable(obj, build->inputs, &rebuild_wordcount_kv,
                                   arg));
}
#endif

void load_native_counts_to_far(
    FarWordMap &map, const NativeCountMap &counts,
    FarLib::DereferenceScope &scope) {
    for (const auto &entry : counts) {
        ASSERT(map.put(entry.first,
                       make_count_value(entry.first, entry.second.count),
                       scope));
    }
}

#if FARLIB_WORDCOUNT_RECOVERY_TEST
uint64_t checksum_native_counts(const NativeCountMap &counts) {
    std::vector<std::pair<std::string, uint64_t>> sorted;
    sorted.reserve(counts.size());
    for (const auto &entry : counts) {
        sorted.emplace_back(std::string(entry.first.view()), entry.second.count);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    return checksum_counts(sorted);
}
#endif

#if FARLIB_WORDCOUNT_STARFISH
bool rebuild_wordcount_kv(void *dst, size_t bytes, const void *ctx,
                          uint64_t arg) {
    auto *input = static_cast<const WordRecipeInput *>(ctx);
    auto fail = [] {
#if FARLIB_WORDCOUNT_RECOVERY_TEST
        recipe_failures.fetch_add(1, std::memory_order_relaxed);
#endif
        return false;
    };
    if (input == nullptr || input->file == nullptr || dst == nullptr ||
        bytes != FarWordMap::kv_data_size() || arg < input->chunk.begin ||
        arg >= input->chunk.end) {
        return fail();
    }

    const NativeCountMap *counts = input->ensure_counts();
    if (counts == nullptr) {
        return fail();
    }

    FixedWord word;
    size_t pos = static_cast<size_t>(arg);
    size_t word_offset = 0;
    ThreadStats parse_stats;
    if (!read_word(input->file->data, pos, input->chunk.end, word,
                   parse_stats, &word_offset) || word_offset != arg) {
        return fail();
    }
    const auto it = counts->find(word);
    if (it == counts->end() || it->second.first_offset != arg) {
        return fail();
    }

    // Use the canonical key from the rebuilt partition map.  The parser word
    // is only the lookup probe; copying the stored key preserves the complete
    // FixedWord object representation while leaving its 514-byte payload out
    // of the recipe metadata.
    const FixedWord &restored_word = it->first;
    CountValue value = make_count_value(restored_word, it->second.count);
    std::array<unsigned char, FarWordMap::kv_data_size()> image{};
    std::memcpy(image.data() + FarWordMap::kv_key_offset(), &restored_word,
                sizeof(restored_word));
    std::memcpy(image.data() + FarWordMap::kv_value_offset(), &value,
                sizeof(value));
    std::memcpy(dst, image.data(), image.size());
#if FARLIB_WORDCOUNT_RECOVERY_TEST
    recipe_restored_objects.fetch_add(1, std::memory_order_relaxed);
#endif
    return true;
}
#endif

#if FARLIB_WORDCOUNT_RECOVERY_TEST
void write_recipe_marker(const std::string &path, const std::string &line) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("open recipe marker failed: " + path +
                                 ": " + strerror(errno));
    }
    const char *data = line.data();
    size_t left = line.size();
    while (left != 0) {
        const ssize_t written = write(fd, data, left);
        if (written <= 0) {
            close(fd);
            throw std::runtime_error("write recipe marker failed: " + path +
                                     ": " + strerror(errno));
        }
        data += written;
        left -= static_cast<size_t>(written);
    }
    close(fd);
}

bool wait_for_recipe_marker(const std::string &path) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(90);
    while (std::chrono::steady_clock::now() < deadline) {
        if (access(path.c_str(), F_OK) == 0) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return access(path.c_str(), F_OK) == 0;
}
#endif

void print_phase_result(const char *name, double elapsed_s, size_t ops,
                        const PhaseStats &delta) {
    std::cout << "wordcount_far_phase phase=" << name
              << " elapsed_s=" << elapsed_s
              << " ops=" << ops
              << " rdma_read_bytes=" << delta.rdma_read_bytes
              << " rdma_write_bytes=" << delta.rdma_write_bytes
              << "\n";
}

void print_usage(const char *argv0) {
    std::cerr << "usage: " << argv0
              << " <config> <dataset> [threads] [max_bytes] [top_k]\n";
}

} // namespace

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc < 3 || argc > 6) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        FarLib::rdma::Configure config;
        config.from_file(argv[1]);

        size_t threads = std::min<size_t>(env_size("FARLIB_WORDCOUNT_THREADS", 24),
                                          config.max_thread_cnt);
        if (argc >= 4) {
            threads = std::min(parse_size_arg(argv[3], "threads"),
                               static_cast<size_t>(config.max_thread_cnt));
        }
        size_t max_bytes = 0;
        if (argc >= 5) {
            max_bytes = parse_size_arg(argv[4], "max_bytes");
        }
        size_t top_k = 20;
        if (argc >= 6) {
            top_k = parse_size_arg(argv[5], "top_k");
        }
        size_t global_map_shift = env_size("FARLIB_WORDCOUNT_GLOBAL_MAP_SHIFT", 25);
        bool keep_local_far_maps =
            env_size("FARLIB_WORDCOUNT_KEEP_LOCAL_FAR_MAPS", 0) != 0;
        size_t local_map_shift = env_size("FARLIB_WORDCOUNT_LOCAL_MAP_SHIFT", 21);
        size_t map_uthreads =
            std::max(env_size("FARLIB_WORDCOUNT_UTHREADS", threads), threads);
#if FARLIB_WORDCOUNT_STARFISH
        const bool recipe_mode =
            env_size("FARLIB_WORDCOUNT_RECIPES", 0) != 0;
        const bool recomputable_mode =
            env_size("FARLIB_WORDCOUNT_RECOMPUTABLE", 0) != 0;
#else
        constexpr bool recipe_mode = false;
        constexpr bool recomputable_mode = false;
#endif

        ASSERT(threads > 0);
        if (recipe_mode && !keep_local_far_maps) {
            throw std::runtime_error(
                "FARLIB_WORDCOUNT_RECIPES=1 requires "
                "FARLIB_WORDCOUNT_KEEP_LOCAL_FAR_MAPS=1");
        }
#if FARLIB_WORDCOUNT_STARFISH
        auto file_owner = std::make_shared<MappedFile>(argv[2]);
        const MappedFile &file = *file_owner;
#else
        MappedFile file(argv[2]);
#endif
        size_t size = file.size;
        if (max_bytes != 0) {
            size = std::min(size, max_bytes);
        }
        auto chunks = make_chunks(file.data, size, threads);
        size_t subtasks_per_chunk =
            (map_uthreads + chunks.size() - 1) / chunks.size();
        size_t map_task_count = chunks.size() * subtasks_per_chunk;
#if FARLIB_WORDCOUNT_RECOVERY_TEST
        const char *ready_env = std::getenv("FARLIB_WC_RECIPE_READY_PATH");
        const char *go_env = std::getenv("FARLIB_WC_RECIPE_GO_PATH");
        const bool recipe_gate =
            recipe_mode && ready_env != nullptr && ready_env[0] != '\0' &&
            go_env != nullptr && go_env[0] != '\0';
#endif

        std::cout << "wordcount_far start file=" << argv[2]
                  << " bytes=" << size
                  << " threads=" << chunks.size()
                  << " requested_map_uthreads=" << map_uthreads
                  << " map_task_count=" << map_task_count
                  << " subtasks_per_chunk=" << subtasks_per_chunk
                  << " global_map_shift=" << global_map_shift
                  << " keep_local_far_maps=" << keep_local_far_maps
                  << " local_map_shift=" << local_map_shift
                  << " starfish_hooks=" << FARLIB_WORDCOUNT_STARFISH
                  << " recipe_mode=" << recipe_mode
                  << " sizeof_key=" << sizeof(FixedWord)
                  << " sizeof_value=" << sizeof(CountValue)
                  << "\n";
        if (recomputable_mode) {
            std::cout << "wordcount_far_recomputable start scope=all_maps\n";
        }
        if (recipe_mode) {
            std::cout << "wordcount_far_recipe start scope=local_maps\n";
        }

        FarLib::runtime_init(config);
        {
            std::vector<std::unique_ptr<FarWordMap>> local_maps;
            if (keep_local_far_maps) {
                local_maps.reserve(map_task_count);
                for (size_t i = 0; i < map_task_count; ++i) {
#if FARLIB_WORDCOUNT_STARFISH
                    local_maps.emplace_back(
                        std::make_unique<FarWordMap>(
                            local_map_shift,
                            (!recipe_mode && recomputable_mode)
                                ? &mark_wordcount_recomputable
                                : nullptr,
                            recipe_mode ? &register_wordcount_recipe
                                         : nullptr));
#else
                    local_maps.emplace_back(std::make_unique<FarWordMap>(local_map_shift));
#endif
                }
            }
#if FARLIB_WORDCOUNT_STARFISH
            // Legacy mode keeps its historical all-map annotation.  Recipe
            // mode deliberately leaves the global aggregate protected by EC;
            // only local partial-map objects carry replayable recipes.
            FarWordMap global_map(
                global_map_shift,
                (!recipe_mode && recomputable_mode)
                    ? &mark_wordcount_recomputable
                    : nullptr);
#else
            FarWordMap global_map(global_map_shift);
#endif
            std::array<FarLib::Spinlock, kGlobalLockCount> global_locks;
            std::vector<ThreadStats> task_stats(map_task_count);
            std::vector<std::unique_ptr<
                NativeCountMap>>
                native_partials(map_task_count);
#if FARLIB_WORDCOUNT_RECOVERY_TEST
            std::vector<uint64_t> native_partition_checksums(map_task_count, 0);
#endif
#if FARLIB_WORDCOUNT_STARFISH
            std::vector<std::shared_ptr<RecipeBuildHookContext>>
                recipe_bindings(map_task_count);
#endif

            auto preagg_stats_before = collect_phase_stats();
            auto preagg_start = std::chrono::steady_clock::now();
            FarLib::uthread::parallel_for_with_scope<1>(
                map_task_count, map_task_count, [&](size_t task_id, auto &scope) {
                    (void)scope;
                    size_t chunk_id = task_id % chunks.size();
                    size_t subtask_id = task_id / chunks.size();
                    Chunk parent = chunks[chunk_id];
                    size_t begin =
                        parent.begin +
                        (parent.end - parent.begin) * subtask_id /
                            subtasks_per_chunk;
                    size_t end =
                        parent.begin +
                        (parent.end - parent.begin) * (subtask_id + 1) /
                            subtasks_per_chunk;
                    if (subtask_id != 0) {
                        begin = advance_to_delimiter(file.data, begin, parent.end);
                    }
                    if (subtask_id + 1 != subtasks_per_chunk) {
                        end = advance_to_delimiter(file.data, end, parent.end);
                    }
                    if (begin > parent.end) {
                        begin = parent.end;
                    }
                    if (end > parent.end) {
                        end = parent.end;
                    }
                    Chunk subchunk{begin, end};
                    auto native_counts = std::make_unique<NativeCountMap>();
                    native_counts->reserve(1 << 20);
                    count_chunk_native(*native_counts, file.data, subchunk,
                                       task_stats[task_id]);
#if FARLIB_WORDCOUNT_RECOVERY_TEST
                    if (recipe_gate) {
                        native_partition_checksums[task_id] =
                            checksum_native_counts(*native_counts);
                    }
#endif
#if FARLIB_WORDCOUNT_STARFISH
                    if (recipe_mode) {
                        auto input = std::make_shared<const WordRecipeInput>(
                            file_owner, subchunk, kMaxWordBytes);
                        recipe_bindings[task_id] =
                            std::make_shared<RecipeBuildHookContext>(
                                std::move(input));
                        recipe_bindings[task_id]->native_counts.store(
                            native_counts.get(), std::memory_order_release);
                    }
#endif
                    native_partials[task_id] = std::move(native_counts);
                });
            auto preagg_stop = std::chrono::steady_clock::now();
            double preagg_elapsed =
                std::chrono::duration<double>(preagg_stop - preagg_start)
                    .count();
            uint64_t total_words = 0;
            uint64_t too_long_words = 0;
            uint64_t max_word_len = 0;
            size_t total_native_unique = 0;
            for (const auto &stat : task_stats) {
                total_words += stat.total_words;
                too_long_words += stat.too_long_words;
                max_word_len = std::max(max_word_len, stat.max_word_len);
                total_native_unique += stat.native_unique_words;
            }
            print_phase_result("native_preagg", preagg_elapsed, total_words,
                               collect_phase_stats() - preagg_stats_before);

            // Measure only the remote map/merge/reduce portion. Keep input
            // parsing and native preaggregation outside the planner window.
            FarLib::profile::start_work();
            std::cout << "wordcount_far_work_phase_start phase=map_local\n";
            auto map_stats_before = collect_phase_stats();
            auto map_start = std::chrono::steady_clock::now();
            auto map_one_task = [&](size_t task_id, auto &scope) {
                auto &native_counts = native_partials[task_id];
#if FARLIB_WORDCOUNT_STARFISH
                if (recipe_mode) {
                    auto &map = *local_maps[task_id];
                    auto &build = recipe_bindings[task_id];
                    ASSERT(build != nullptr);
                    build->native_counts.store(native_counts.get(),
                                               std::memory_order_release);
                    map.set_recipe_context(build.get());
                    for (const auto &entry : *native_counts) {
                        ASSERT(map.put_recipe(
                            entry.first,
                            make_count_value(entry.first, entry.second.count),
                            entry.second.first_offset, scope));
                    }
                    // The build-only lookup is never retained by the recipe;
                    // only the immutable input owner and offset are bound.
                    map.set_recipe_context(nullptr);
                    build->native_counts.store(nullptr,
                                               std::memory_order_release);
                } else
#endif
                if (keep_local_far_maps) {
                    load_native_counts_to_far(*local_maps[task_id],
                                              *native_counts, scope);
                } else {
                    for (const auto &entry : *native_counts) {
                        size_t lock_idx =
                            FixedWordHash{}(entry.first) &
                            (kGlobalLockCount - 1);
                        global_locks[lock_idx].lock(scope);
                        add_to_map(global_map, entry.first, entry.second.count,
                                   scope);
                        global_locks[lock_idx].unlock();
                    }
                }
                native_counts.reset();
            };
            auto run_map_range = [&](size_t begin, size_t end) {
                if (begin >= end) return;
                const size_t count = end - begin;
                FarLib::uthread::parallel_for_with_scope<1>(
                    count, count, [&](size_t offset, auto &scope) {
                        map_one_task(begin + offset, scope);
                    });
            };

#if FARLIB_WORDCOUNT_RECOVERY_TEST
            if (recipe_gate) {
                const size_t completed_partitions = (map_task_count * 3) / 4;
                run_map_range(0, completed_partitions);
                const std::string ready_line =
                    "wordcount_recipe_fault_ready phase=map_local "
                    "completed_partitions=" +
                    std::to_string(completed_partitions) +
                    " total_partitions=" + std::to_string(map_task_count) +
                    " completed_native_inputs_live=0";
                std::cout << ready_line << "\n";
                write_recipe_marker(ready_env, ready_line + "\n");
                if (!wait_for_recipe_marker(go_env)) {
#if FARLIB_WORDCOUNT_RECOVERY_TEST
                    recipe_failures.fetch_add(1, std::memory_order_relaxed);
#endif
                    const auto runtime_recipe_stats =
                        FarLib::Cache::get_default()->recomputable_stats();
                    std::cout
                        << "wordcount_recipe_local_verify status=fail "
                           "reason=go_timeout\n";
                    std::cout << "wordcount_recipe_summary restored_objects="
                              << recipe_restored_objects.load()
                              << " partition_rebuilds="
                              << recipe_partition_rebuilds.load()
                              << " failed=" << recipe_failures.load()
                              << " runtime_registered="
                              << runtime_recipe_stats.registered
                              << " runtime_restored="
                              << runtime_recipe_stats.restored
                              << " runtime_failures="
                              << runtime_recipe_stats.failures
                              << " virtual_evictions="
                              << runtime_recipe_stats.virtual_evictions
                              << " failed_writes="
                              << runtime_recipe_stats.failed_writes << "\n";
                    throw std::runtime_error(
                        "timed out waiting for FARLIB_WC_RECIPE_GO_PATH");
                }

                bool verify_ok = true;
                size_t verified_objects = 0;
                for (size_t task_id = 0; task_id < completed_partitions;
                     ++task_id) {
                    std::vector<std::pair<std::string, uint64_t>> actual;
                    actual.reserve(task_stats[task_id].native_unique_words);
                    FarLib::RootDereferenceScope verify_scope;
                    local_maps[task_id]->for_each_locked(
                        [&](const FixedWord &word, const CountValue &value,
                            auto &) {
                            actual.emplace_back(std::string(word.view()),
                                                value.count);
                            ++verified_objects;
                        },
                        verify_scope);
                    std::sort(actual.begin(), actual.end(),
                              [](const auto &a, const auto &b) {
                                  return a.first < b.first;
                              });
                    uint64_t actual_words = 0;
                    for (const auto &entry : actual) actual_words += entry.second;
                    const bool partition_ok =
                        actual.size() == task_stats[task_id].native_unique_words &&
                        actual_words == task_stats[task_id].total_words &&
                        checksum_counts(actual) ==
                            native_partition_checksums[task_id];
                    verify_ok = verify_ok && partition_ok;
                }
                const auto runtime_recipe_stats =
                    FarLib::Cache::get_default()->recomputable_stats();
                const uint64_t callback_restored =
                    recipe_restored_objects.load(std::memory_order_relaxed);
                const bool runtime_recovery_ok =
                    runtime_recipe_stats.restored == callback_restored;
                verify_ok = verify_ok && runtime_recovery_ok;
                if (!verify_ok) {
#if FARLIB_WORDCOUNT_RECOVERY_TEST
                    recipe_failures.fetch_add(1, std::memory_order_relaxed);
#endif
                }
                std::cout << "wordcount_recipe_local_verify status="
                          << (verify_ok ? "pass" : "fail")
                          << " partitions=" << completed_partitions
                          << " objects=" << verified_objects
                          << " restored_objects="
                          << callback_restored
                          << " partition_rebuilds="
                          << recipe_partition_rebuilds.load()
                          << " runtime_restored="
                          << runtime_recipe_stats.restored << "\n";
                if (!verify_ok) {
                    std::cout << "wordcount_recipe_summary restored_objects="
                              << recipe_restored_objects.load()
                              << " partition_rebuilds="
                              << recipe_partition_rebuilds.load()
                              << " failed=" << recipe_failures.load()
                              << " runtime_registered="
                              << runtime_recipe_stats.registered
                              << " runtime_restored="
                              << runtime_recipe_stats.restored
                              << " runtime_failures="
                              << runtime_recipe_stats.failures
                              << " virtual_evictions="
                              << runtime_recipe_stats.virtual_evictions
                              << " failed_writes="
                              << runtime_recipe_stats.failed_writes << "\n";
                    throw std::runtime_error(
                        "recipe local-map verification failed");
                }
                run_map_range(completed_partitions, map_task_count);
            } else {
                run_map_range(0, map_task_count);
            }
#endif
#if !FARLIB_WORDCOUNT_RECOVERY_TEST
            run_map_range(0, map_task_count);
#endif
            auto map_stop = std::chrono::steady_clock::now();
            double map_elapsed =
                std::chrono::duration<double>(map_stop - map_start).count();
            print_phase_result("map_local", map_elapsed, total_native_unique,
                               collect_phase_stats() - map_stats_before);

            auto merge_stats_before = collect_phase_stats();
            auto merge_start = std::chrono::steady_clock::now();
            std::atomic_size_t merged_unique{0};
            if (keep_local_far_maps) {
                size_t merge_worker_count =
                    env_size("FARLIB_WORDCOUNT_MERGE_WORKERS",
                             std::min<size_t>(threads, map_task_count));
                merge_worker_count = std::max<size_t>(
                    1, std::min<size_t>(merge_worker_count, map_task_count));
                FarLib::uthread::parallel_for_with_scope<1, true>(
                    merge_worker_count, map_task_count,
                    [&](size_t tid, auto &scope) {
                        local_maps[tid]->for_each_snapshot(
                            [&](const FixedWord &word, const CountValue &value,
                                auto &inner_scope) {
                                size_t lock_idx =
                                    FixedWordHash{}(word) &
                                    (kGlobalLockCount - 1);
                                global_locks[lock_idx].lock(inner_scope);
                                add_to_map(global_map, word, value.count,
                                           inner_scope);
                                global_locks[lock_idx].unlock();
                                ++merged_unique;
                            },
                            scope);
                        local_maps[tid].reset();
                    });
            } else {
                merged_unique = global_map.size.load();
            }
            auto merge_stop = std::chrono::steady_clock::now();
            double merge_elapsed =
                std::chrono::duration<double>(merge_stop - merge_start).count();
            print_phase_result("merge_global", merge_elapsed, merged_unique.load(),
                               collect_phase_stats() - merge_stats_before);

            auto reduce_stats_before = collect_phase_stats();
            auto reduce_start = std::chrono::steady_clock::now();
            std::unordered_map<std::string, uint64_t> native_counts;
            native_counts.reserve(global_map.size.load());
            const size_t reduce_readers =
                env_size("FARLIB_WORDCOUNT_REDUCE_READERS", 1);
            if (reduce_readers > 1) {
                global_map.for_each_readonly_parallel(
                    [&](const FixedWord &word, const CountValue &value) {
                        native_counts.emplace(std::string(word.view()), value.count);
                    }, reduce_readers);
            } else {
                FarLib::RootDereferenceScope scope;
                global_map.for_each_locked(
                    [&](const FixedWord &word, const CountValue &value,
                        auto &) {
                        native_counts.emplace(std::string(word.view()), value.count);
                    },
                    scope);
            }
            std::vector<std::pair<std::string, uint64_t>> sorted;
            sorted.reserve(native_counts.size());
            for (auto &entry : native_counts) {
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
            auto reduce_stop = std::chrono::steady_clock::now();
            double reduce_elapsed =
                std::chrono::duration<double>(reduce_stop - reduce_start).count();
            print_phase_result("reduce_checksum", reduce_elapsed, sorted.size(),
                               collect_phase_stats() - reduce_stats_before);

            std::cout
                << "wordcount_far_work_phase_end phase=reduce_checksum\n";
            FarLib::profile::end_work();

#if FARLIB_WORDCOUNT_RECOVERY_TEST
            if (recipe_mode) {
                const auto runtime_recipe_stats =
                    FarLib::Cache::get_default()->recomputable_stats();
                std::cout << "wordcount_recipe_summary restored_objects="
                          << recipe_restored_objects.load()
                          << " partition_rebuilds="
                          << recipe_partition_rebuilds.load()
                          << " failed=" << recipe_failures.load()
                          << " runtime_registered="
                          << runtime_recipe_stats.registered
                          << " runtime_restored="
                          << runtime_recipe_stats.restored
                          << " runtime_failures="
                          << runtime_recipe_stats.failures
                          << " virtual_evictions="
                          << runtime_recipe_stats.virtual_evictions
                          << " failed_writes="
                          << runtime_recipe_stats.failed_writes << "\n";
            }
#endif

            std::cout << "wordcount_far_result"
                      << " file=" << argv[2]
                      << " bytes=" << size
                      << " threads=" << chunks.size()
                      << " total_words=" << total_words
                      << " unique_words=" << sorted.size()
                      << " hashmap_size=" << global_map.size.load()
                      << " checksum=0x" << std::hex << checksum << std::dec
                      << " too_long_words=" << too_long_words
                      << " max_word_len=" << max_word_len
                      << " remote_used_bytes="
                      << FarLib::allocator::remote::remote_global_heap.get_used_bytes()
                      << "\n";
#if FARLIB_WORDCOUNT_RECOVERY_TEST
            if (recomputable_mode) {
                const auto recomputable_stats =
                    FarLib::Cache::get_default()->recomputable_stats();
                std::cout << "wordcount_far_recomputable_stats"
                          << " marked=" << recomputable_stats.marked
                          << " flat_writebacks="
                          << recomputable_stats.flat_writebacks
                          << " flat_write_bytes="
                          << recomputable_stats.flat_write_bytes
                          << " flat_read_posts="
                          << recomputable_stats.flat_read_posts << "\n";
            }
#endif
            for (size_t i = 0; i < std::min(top_k, top.size()); ++i) {
                std::cout << "top" << (i + 1)
                          << " word=" << top[i].first
                          << " count=" << top[i].second
                          << "\n";
            }

            FarLib::runtime_quiesce_cache();
        }
        FarLib::runtime_destroy();
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
