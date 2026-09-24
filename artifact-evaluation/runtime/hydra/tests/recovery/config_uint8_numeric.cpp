// CPU-only regression test for numeric uint8_t configuration parsing.
//
// The RDMA configuration header is intentionally tested directly: parsing a
// configuration file does not create an RDMA context or contact a device.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "rdma/config.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace {

namespace fs = std::filesystem;
using FarLib::rdma::Configure;
using FarLib::rdma::detail::read_config_value;

class ConfigFiles {
  public:
    ConfigFiles() {
        static unsigned long long sequence = 0;
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        root_ = fs::temp_directory_path() /
                ("farlib-config-uint8-" + std::to_string(stamp) + "-" +
                 std::to_string(sequence++));
        std::error_code error;
        assert(fs::create_directory(root_, error));
        assert(!error);
    }

    ~ConfigFiles() {
        std::error_code error;
        fs::remove_all(root_, error);
    }

    fs::path write(const std::string &name, const std::string &contents) {
        const fs::path path = root_ / name;
        std::ofstream output(path);
        assert(output.is_open());
        output << contents;
        assert(output.good());
        return path;
    }

  private:
    fs::path root_;
};

void expect_byte_token(const std::string &token, uint8_t expected) {
    std::istringstream input(token);
    uint8_t value = 0xff;
    assert(read_config_value(input, value));
    assert(value == expected);
}

void expect_invalid_byte_token(const std::string &token) {
    std::istringstream input(token);
    uint8_t value = 0x5a;
    assert(!read_config_value(input, value));
    assert(value == 0x5a);
    assert(input.fail());
}

void test_byte_token_validation() {
    for (const auto &[token, expected] :
         std::array<std::pair<const char *, uint8_t>, 6>{
             std::pair{"0", uint8_t{0}},
             std::pair{"1", uint8_t{1}},
             std::pair{"3", uint8_t{3}},
             std::pair{"7", uint8_t{7}},
             std::pair{"16", uint8_t{16}},
             std::pair{"255", uint8_t{255}}}) {
        expect_byte_token(token, expected);
    }

    // The entire whitespace-delimited token must be decimal digits in range;
    // a prefix parse ("1x", "0x10") or a negative/overflow value is invalid.
    for (const char *token : {"", " ", "256", "999999999999999999999999",
                              "-1", "+1", "1x", "x1", "0x10"}) {
        expect_invalid_byte_token(token);
    }
}

void test_defaults_and_numeric_fields() {
    ConfigFiles files;
    const fs::path empty = files.write("empty.cfg", "");

    Configure defaults;
    defaults.from_file(empty.string().c_str());
    assert(defaults.qp_retry_cnt == uint8_t{7});
    assert(defaults.qp_timeout == uint8_t{8});

    for (const uint8_t expected : {uint8_t{0}, uint8_t{1}, uint8_t{3},
                                   uint8_t{7}}) {
        const fs::path path = files.write(
            "retry.cfg", "qp_retry_cnt " + std::to_string(expected) + "\n");
        Configure config;
        config.from_file(path.string().c_str());
        assert(config.qp_retry_cnt == expected);
        assert(config.qp_timeout == uint8_t{8});
    }

    for (const uint8_t expected : {uint8_t{8}, uint8_t{16}, uint8_t{31}}) {
        const fs::path path = files.write(
            "timeout.cfg", "qp_timeout " + std::to_string(expected) + "\n");
        Configure config;
        config.from_file(path.string().c_str());
        assert(config.qp_timeout == expected);
        assert(config.qp_retry_cnt == uint8_t{7});
    }

    const fs::path all_bytes = files.write(
        "all-bytes.cfg",
        "qp_max_rd_atomic 255\n"
        "qp_min_rnr_timer 255\n"
        "qp_rnr_retry 255\n"
        "ib_port 255\n");
    Configure config;
    config.from_file(all_bytes.string().c_str());
    assert(config.qp_max_rd_atomic == uint8_t{255});
    assert(config.qp_min_rnr_timer == uint8_t{255});
    assert(config.qp_rnr_retry == uint8_t{255});
    assert(config.ib_port == uint8_t{255});
}

void test_include_and_non_byte_fields() {
    ConfigFiles files;
    const fs::path included = files.write(
        "included.cfg",
        "qp_retry_cnt 1\n"
        "qp_timeout 31\n"
        "qp_max_rd_atomic 255\n"
        "cq_entries 17\n"
        "server_addr 10.0.0.7\n"
        "exclusive_cache 0\n");
    const fs::path main = files.write(
        "main.cfg", "qp_retry_cnt 0\nqp_timeout 8\ninclude " +
                           included.string() + "\n");

    Configure config;
    config.from_file(main.string().c_str());

    // Values from the included file override the values read before it.
    assert(config.qp_retry_cnt == uint8_t{1});
    assert(config.qp_timeout == uint8_t{31});
    assert(config.qp_max_rd_atomic == uint8_t{255});

    // Non-byte fields still use their ordinary stream extraction semantics.
    assert(config.cq_entries == 17);
    assert(config.server_addr == "10.0.0.7");
    assert(config.exclusive_cache == false);
}

}  // namespace

int main() {
    test_byte_token_validation();
    test_defaults_and_numeric_fields();
    test_include_and_non_byte_fields();
    std::cout << "CONFIG_UINT8_NUMERIC_PASS\n";
    return 0;
}
