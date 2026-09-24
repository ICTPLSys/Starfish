#include "rdma/config.hpp"
#include <cassert>
#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>
using FarLib::rdma::Configure;
Configure valid() {
    Configure c;
    c.ft_method = "hydra";
    c.server_count = 7;
    c.server_addrs = "192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4,192.0.2.5,192.0.2.6,192.0.2.7";
    c.server_ports = "23610,23611,23612,23613,23614,23615,23616";
    c.local_resident_budget_bytes = 4194304;
    c.post_process_config();
    return c;
}
int main() {
    auto c = valid();
    assert(c.is_hydra_mode() && c.is_ec_batch_mode());
    c.self_check();
    // Fixed regions are allowed without enabling a behavior-group planner.
    c.enable_region_resident_placement = true;
    c.self_check();
    for (int option = 0; option < 6; ++option) {
        const auto pid = fork();
        assert(pid >= 0);
        if (!pid) {
            auto bad = valid();
            if (option == 0) bad.enable_selective_backup = true;
            if (option == 1) bad.enable_logical_object_profile = true;
            if (option == 2) bad.enable_resident_profile_planner = true;
            if (option == 3) bad.region_placement_bind_groups = true;
            if (option == 4) bad.enable_region_hotness_placement = true;
            if (option == 5) bad.enable_region_fetch_hotness_placement = true;
            bad.self_check();
            _exit(0);
        }
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 2);
    }
    std::puts("HYDRA_CONFIGURATION_PASS backup-off groups-off fixed-resident-allowed");
}
