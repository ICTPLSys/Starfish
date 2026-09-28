"""Offline CPU-pool validation checks for Figure 9 launch placement."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
import topology


def records(*, app_node=0, background_node=0, smt=None):
    """Return one non-SMT record per selected CPU plus optional SMT override."""
    result = []
    for cpu in range(24):
        result.append({"cpu": cpu, "core": cpu, "socket": 0, "node": app_node})
    for cpu in range(40, 46):
        result.append({"cpu": cpu, "core": cpu, "socket": 0, "node": background_node})
    if smt:
        for cpu, physical in smt.items():
            for record in result:
                if record["cpu"] == cpu:
                    record["socket"], record["core"] = physical
    return result


def site(**overrides):
    result = {
        "numa_node": 0,
        "fibre_workers": 24,
        "fibre_cpu_set": "0-23",
        "background_cpu_base": 40,
    }
    result.update(overrides)
    return result


class CpuPlacement(unittest.TestCase):
    def test_valid_explicit_24_plus_6_pool(self):
        placement = topology.check_cpu_placement(
            site(), cpu_records=records(), allowed_cpus=set(range(64)))
        self.assertEqual(placement["app_cpus"], tuple(range(24)))
        self.assertEqual(placement["background_cpus"], tuple(range(40, 46)))
        self.assertEqual(placement["background_mark_workers"], 2)
        self.assertEqual(placement["background_evict_workers"], 4)

    def test_missing_cpu_fields_fail_without_defaults(self):
        for missing in ("fibre_cpu_set", "background_cpu_base"):
            candidate = site()
            del candidate[missing]
            with self.assertRaisesRegex(ValueError, "explicit site field"):
                topology.check_cpu_placement(candidate)

    def test_cpu_list_rejects_negative_duplicate_malformed_and_wrong_count(self):
        for value in ("-1,0-22", "0-22,22", "0--23", "0-22"):
            with self.assertRaises(ValueError):
                topology.check_cpu_placement(
                    site(fibre_cpu_set=value), cpu_records=records(),
                    allowed_cpus=set(range(64)))

    def test_cpu_must_be_online_and_in_process_affinity(self):
        with self.assertRaisesRegex(ValueError, "not online"):
            topology.check_cpu_placement(
                site(), cpu_records=records()[:-1], allowed_cpus=set(range(64)))
        with self.assertRaisesRegex(ValueError, "outside sched_getaffinity"):
            topology.check_cpu_placement(
                site(), cpu_records=records(), allowed_cpus=set(range(45)))

    def test_smt_is_rejected_within_or_across_pools(self):
        same_app_core = records(smt={1: (0, 0)})
        with self.assertRaisesRegex(ValueError, "shares physical core"):
            topology.check_cpu_placement(
                site(), cpu_records=same_app_core, allowed_cpus=set(range(64)))

        across_pools = records(smt={40: (0, 0)})
        with self.assertRaisesRegex(ValueError, "shares physical core"):
            topology.check_cpu_placement(
                site(), cpu_records=across_pools, allowed_cpus=set(range(64)))

    def test_background_pool_stays_contiguous_and_on_site_numa(self):
        with self.assertRaisesRegex(ValueError, "background CPU 40"):
            topology.check_cpu_placement(
                site(), cpu_records=records(background_node=1),
                allowed_cpus=set(range(64)))

    def test_app_pool_requires_site_numa_unless_explicit_exception(self):
        with self.assertRaisesRegex(ValueError, "expected site NUMA"):
            topology.check_cpu_placement(
                site(), cpu_records=records(app_node=1),
                allowed_cpus=set(range(64)))
        placement = topology.check_cpu_placement(
            site(allow_cross_numa_app=True), cpu_records=records(app_node=1),
            allowed_cpus=set(range(64)))
        self.assertTrue(placement["allow_cross_numa_app"])

    def test_lscpu_parser_skips_comments_and_rejects_malformed_records(self):
        output = chr(10).join(("# CPU,Core,Socket,Node", "0,0,0,0", "1,1,0,0"))
        self.assertEqual(topology.parse_lscpu_records(output)[1]["cpu"], 1)
        with self.assertRaises(ValueError):
            topology.parse_lscpu_records(output + chr(10) + "bad")

    def test_default_probe_uses_lscpu_and_sched_getaffinity(self):
        output = chr(10).join(
            f"{item['cpu']},{item['core']},{item['socket']},{item['node']}"
            for item in records())
        with patch.object(topology.subprocess, "check_output",
                          return_value=output) as probe:
            with patch.object(topology.os, "sched_getaffinity",
                              return_value=set(range(64))) as affinity:
                topology.check_cpu_placement(site())
        probe.assert_called_once_with(
            ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
        affinity.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
