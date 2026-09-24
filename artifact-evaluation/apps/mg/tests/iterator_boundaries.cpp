#include <array>
#include <cstdint>
#include <iostream>

#include "data_structure/far_vector.hpp"

template <size_t GroupSize>
bool check_negative_offsets() {
    using Vector = FarLib::FarVector<double, GroupSize>;
    using Group = std::array<double, GroupSize>;
    using GroupPtr = FarLib::UniqueFarPtr<Group>;

    constexpr size_t group_count = 8;
    std::array<GroupPtr, group_count> groups{};
    constexpr size_t origin_group = 5;
    using Iterator = typename Vector::const_lite_iterator;
    const Iterator origin(groups.data() + origin_group, 0, groups.data(),
                          groups.data() + groups.size());

    const std::array<int64_t, 10> offsets = {
        -1,
        -static_cast<int64_t>(GroupSize),
        -static_cast<int64_t>(GroupSize) - 1,
        -2 * static_cast<int64_t>(GroupSize),
        -2 * static_cast<int64_t>(GroupSize) - 1,
        -3 * static_cast<int64_t>(GroupSize) + 1,
        0,
        1,
        static_cast<int64_t>(GroupSize),
        static_cast<int64_t>(GroupSize) + 1,
    };
    bool ok = true;
    for (const int64_t offset : offsets) {
        const Iterator shifted(origin, offset);
        if (!shifted.valid() || shifted - origin != offset) {
            std::cerr << "GroupSize=" << GroupSize << " offset=" << offset
                      << " valid=" << shifted.valid()
                      << " distance=" << (shifted - origin) << '\n';
            ok = false;
        }
    }
    return ok;
}

bool check_mg_psinv_halo() {
    using Vector = FarLib::FarVector<double, 509>;
    using Group = std::array<double, 509>;
    using GroupPtr = FarLib::UniqueFarPtr<Group>;
    using Iterator = typename Vector::const_lite_iterator;

    // MG level k=6, n=66, ir=1235212832, fibre=6, i3=19.
    constexpr int64_t k6_start = 1235295663;
    static_assert(k6_start % 509 == 0);

    std::array<GroupPtr, 8> groups{};
    constexpr size_t origin_group = 5;
    const Iterator narrow_origin(groups.data() + origin_group, 0,
                                  groups.data() + origin_group,
                                  groups.data() + groups.size());
    const Iterator narrow_left(narrow_origin, -1);
    const Iterator narrow_right(narrow_origin, 1);

    bool ok = true;
    if (narrow_left.valid()) {
        std::cerr << "narrow left unexpectedly valid\n";
        ok = false;
    }
    if (!narrow_right.valid() || narrow_right - narrow_origin != 1) {
        std::cerr << "narrow right halo mismatch\n";
        ok = false;
    }

    const Iterator widened_origin(groups.data() + origin_group, 0,
                                  groups.data() + origin_group - 1,
                                  groups.data() + groups.size());
    const Iterator widened_left(widened_origin, -1);
    if (!widened_left.valid() || widened_left - widened_origin != -1) {
        std::cerr << "widened left halo mismatch\n";
        ok = false;
    }
    return ok;
}

int main() {
    bool ok = true;
    ok = check_negative_offsets<509>() && ok;
    ok = check_negative_offsets<510>() && ok;
    ok = check_negative_offsets<512>() && ok;
    ok = check_mg_psinv_halo() && ok;
    return ok ? 0 : 1;
}
