// Compile with -Iruntime/nonft/include (no RDMA device is required).
#include "rdma/config.hpp"
#include <cassert>
#include <sstream>

int main() {
    for (unsigned int expected : {0u, 1u, 8u, 16u, 255u}) {
        std::istringstream input(std::to_string(expected));
        uint8_t value = 99;
        assert(FarLib::rdma::detail::read_config_value(input, value));
        assert(value == expected);
    }
    for (const char *invalid : {"-1", "+1", "256", "1x", "999999999999999999999"}) {
        std::istringstream input(invalid);
        uint8_t value = 99;
        assert(!FarLib::rdma::detail::read_config_value(input, value));
        assert(value == 99);
    }
}
