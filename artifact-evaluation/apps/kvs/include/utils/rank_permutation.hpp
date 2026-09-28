#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace FarLib::benchmark {

// Immutable, seeded mapping from a Zipf rank to a logical object id.
class RankPermutation {
 public:
  struct Audit {
    std::size_t unique_count = 0;
    std::size_t duplicate_count = 0;
    std::size_t missing_count = 0;
    std::size_t out_of_range_count = 0;
    std::uint64_t checksum = 0;

    [[nodiscard]] bool valid() const noexcept {
      return duplicate_count == 0 && missing_count == 0 &&
             out_of_range_count == 0;
    }
  };

  RankPermutation(std::size_t count, std::uint64_t seed)
      : count_(validated_count(count)), table_(count_) {
    for (std::size_t i = 0; i < count_; ++i) {
      table_[i] = static_cast<std::uint32_t>(i);
    }

    // This is deliberately not std::shuffle: its distribution implementation
    // is permitted to vary between standard-library implementations.
    std::mt19937_64 engine(seed);
    for (std::size_t i = count_; i > 1; --i) {
      const std::size_t j = static_cast<std::size_t>(
          bounded_draw(engine, static_cast<std::uint64_t>(i)));
      const std::size_t last = i - 1;
      const std::uint32_t value = table_[last];
      table_[last] = table_[j];
      table_[j] = value;
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return count_; }

  // rank is expected to be in [0, size()). The table never changes after
  // construction, so concurrent lookup needs neither locks nor TLS.
  [[nodiscard]] std::uint32_t operator[](std::size_t rank) const noexcept {
    return table_[rank];
  }

  [[nodiscard]] Audit audit() const {
    Audit result;
    std::vector<std::uint8_t> seen(count_, 0);
    constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
    constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);
    result.checksum = kFnvOffset;

    for (const std::uint32_t value : table_) {
      // Define the checksum byte order explicitly, independent of host endian.
      for (unsigned shift = 0; shift < 32; shift += 8) {
        result.checksum ^= static_cast<std::uint8_t>(value >> shift);
        result.checksum *= kFnvPrime;
      }

      if (value >= count_) {
        ++result.out_of_range_count;
      } else if (seen[value] != 0) {
        ++result.duplicate_count;
      } else {
        seen[value] = 1;
        ++result.unique_count;
      }
    }
    result.missing_count = count_ - result.unique_count;
    return result;
  }

 private:
  static std::size_t validated_count(std::size_t count) {
    if (count == 0) {
      throw std::invalid_argument("RankPermutation count must be non-zero");
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("RankPermutation count exceeds uint32_t domain");
    }
    return count;
  }

  static std::uint64_t bounded_draw(std::mt19937_64& engine,
                                    std::uint64_t bound) noexcept {
    // Accept an exact multiple of bound outcomes, then reduce modulo bound.
    // mt19937_64 supplies every uint64_t result with equal probability.
    const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t limit = max - (max % bound);
    std::uint64_t value;
    do {
      value = engine();
    } while (value >= limit);
    return value % bound;
  }

  std::size_t count_;
  std::vector<std::uint32_t> table_;
};

}  // namespace FarLib::benchmark
