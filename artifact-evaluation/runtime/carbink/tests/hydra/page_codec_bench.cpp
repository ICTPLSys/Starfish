// Optional CPU diagnostic, deliberately not a CTest/performance acceptance test.
// Reports encoded input bytes, not network payload or end-to-end throughput.
#include "cache/alloc/small_object_stripe_codec.hpp"
#include "hydra/page_codec.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr size_t shard_bytes = 2048;
struct Page {
    std::array<std::array<uint8_t, shard_bytes>, 6> shards;
};
using Encoder = bool (*)(const void *const *, void *const *, size_t);

__attribute__((noinline)) bool lut(const void *const *d, void *const *p,
                                   size_t n) {
    return FarLib::cache::small_object_stripe_encode_shards(d, p, n);
}

__attribute__((noinline)) bool isa_l(const void *const *d, void *const *p,
                                     size_t n) {
    return FarLib::hydra::page_codec_encode_4plus2(d, p, n);
}

bool one(Page &page, Encoder encode) {
    const void *d[] = {page.shards[0].data(), page.shards[1].data(),
                       page.shards[2].data(), page.shards[3].data()};
    void *p[] = {page.shards[4].data(), page.shards[5].data()};
    return encode(d, p, shard_bytes);
}

void measure(std::vector<Page> &pages, Encoder encode, const char *name,
             size_t count, unsigned repeat) {
    for (auto &p : pages) {
        if (!one(p, encode)) std::abort();
    }
    uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; ++i) {
        auto &page = pages[i % pages.size()];
        if (!one(page, encode)) std::abort();
        checksum += page.shards[4][i % shard_bytes];
        checksum += page.shards[5][i % shard_bytes];
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    std::printf(
        "codec=%s isa_l=%d repeat=%u pages=%zu seconds=%.9f "
        "input_gbps=%.6f checksum=%llu working_pages=%zu working_bytes=%zu\n",
        name, name[0] == 'i' ? 1 : 0, repeat, count, seconds,
        count * 8192.0 * 8 / seconds / 1e9,
        static_cast<unsigned long long>(checksum), pages.size(),
        pages.size() * sizeof(Page));
}
}  // namespace

int main(int argc, char **argv) {
    const size_t count =
        argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 131072;
    if (count == 0 || count > 10000000) return 2;
    const size_t working_pages =
        argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 64;
    if (working_pages == 0 || working_pages > 65536) return 2;
    // Default remains the old 512KiB-input/768KiB-total ring. A larger ring
    // distinguishes cache-resident codec speed from a streaming working set.
    std::vector<Page> pages(working_pages);
    uint32_t random = 0x12345678;
    for (auto &page : pages) {
        for (auto &shard : page.shards) {
            for (auto &byte : shard) {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                byte = static_cast<uint8_t>(random);
            }
        }
        auto reference = page;
        if (!one(reference, lut) || !one(page, isa_l) ||
            reference.shards != page.shards) {
            return 1;
        }
    }
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        if (repeat % 2 == 0) {
            measure(pages, lut, "lut", count, repeat);
            measure(pages, isa_l, "isa_l", count, repeat);
        } else {
            measure(pages, isa_l, "isa_l", count, repeat);
            measure(pages, lut, "lut", count, repeat);
        }
    }
}
