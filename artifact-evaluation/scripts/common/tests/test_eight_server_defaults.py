"""Offline coverage of the eight-physical-server default and CPU selections."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fast_check
import prepare_data
import run_case
import site_defaults
import topology

AE = run_case.AE_ROOT
DEFAULT = AE / "scripts/common/site.eight-server.example.json"
APPS = ("bfs", "llama", "mg", "wordcount", "kv-b", "kv-a", "kv-s", "nq")
HOSTS = ("22", "54", "58", "74", "76", "82", "84")


class EightServerDefaults(unittest.TestCase):
    def raw(self):
        return json.loads(DEFAULT.read_text())

    def plan(self, system, app="llama", raw=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site.json"
            site.write_text(json.dumps(self.raw() if raw is None else raw))
            args = argparse.Namespace(app=app, system=system, ratio=25, site=site,
                                      out=root / "case", timeout=1800, dry_run=True)
            with patch.object(run_case, "ssh") as ssh, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(run_case.run(args), 0)
            ssh.assert_not_called()
            self.assertFalse(args.out.exists())
            return json.loads(output.getvalue())

    def test_new_site_registration_uses_eight_physical_servers(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site.json"
            prepare_data.register_inputs(site, {"bfs": "/fixture/graph"})
            actual = json.loads(site.read_text())
        self.assertEqual(actual["compute_ip"], "10.208.130.56")
        self.assertEqual([r["ip"] for r in actual["memory_endpoints"]],
                         ["10.208.130." + suffix for suffix in HOSTS])
        self.assertEqual(actual["inputs"]["bfs"], "/fixture/graph")
        self.assertNotIn("memory_server_count", actual)

    def test_registration_preserves_an_existing_explicit_site(self):
        original = {"memory_ip": "192.0.2.9", "fibre_cpu_set": "custom"}
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site.json"
            site.write_text(json.dumps(original))
            prepare_data.register_inputs(site, {"bfs": "/fixture/graph"})
            actual = json.loads(site.read_text())
        actual.pop("inputs")
        self.assertEqual(actual, original)

    def test_every_recipe_keeps_24_workers_and_disjoint_physical_cores(self):
        for system in site_defaults.SYSTEMS:
            for app in APPS:
                with self.subTest(system=system, app=app):
                    recipe = run_case.load_recipe(AE / "configs" / app / (system + ".config"))
                    site = site_defaults.resolve(self.raw(), AE, system=system, app=app)
                    profile = topology.check_cpu_profile(site, recipe)
                    background = (13 if (app, system) == ("bfs", "carbink") else
                                  11 if system == "carbink" else
                                  9 if (app, system) == ("nq", "starfish") else
                                  8 if system == "hydra" and app in ("bfs", "llama") else 6)
                    self.assertEqual(profile["app_workers"], 24)
                    self.assertEqual(profile["background_workers"], background)
                    self.assertEqual(profile["background_cpus"],
                                     tuple(range(24 - background, 24)))
                    self.assertEqual(profile["app_cpus"],
                                     tuple(range(24 - background)) +
                                     tuple(range(24, 24 + background)))
                    # Verified 56 topology: CPU0..47 are distinct physical cores;
                    # CPU0..23 belong to the NIC/local-data NUMA0.
                    self.assertTrue(all(c < 24 for c in profile["background_cpus"]))
                    self.assertTrue(all(c < 48 for c in profile["app_cpus"]))
                    self.assertFalse(set(profile["app_cpus"]) &
                                     set(profile["background_cpus"]))

    def test_explicit_site_settings_override_machine_case_defaults(self):
        raw = self.raw()
        raw.update(fibre_cpu_set="0-11,24-35", background_cpu_base=12)
        resolved = site_defaults.resolve(raw, AE, system="starfish", app="nq")
        self.assertEqual(resolved["fibre_cpu_set"], raw["fibre_cpu_set"])
        self.assertEqual(resolved["background_cpu_base"], 12)
        # Existing site specificity remains root < system < app < exact case.
        raw["placement_by_system"] = {"starfish": {"background_cpu_base": 11}}
        raw["placement_by_app"] = {"nq": {"background_cpu_base": 10}}
        raw["placement_by_case"] = {"nq/starfish": {"background_cpu_base": 9}}
        resolved = site_defaults.resolve(raw, AE, system="starfish", app="nq")
        self.assertEqual(resolved["background_cpu_base"], 9)

    def test_bad_case_selector_is_rejected(self):
        raw = self.raw()
        raw["placement_by_case"] = {"starfish/nq": {"background_cpu_base": 15}}
        with self.assertRaisesRegex(ValueError, "unknown placement selector"):
            site_defaults.resolve(raw, AE, system="starfish", app="nq")

    def test_full_matrix_plans_do_not_launch_and_record_reserved_node(self):
        for system in site_defaults.SYSTEMS:
            for app in APPS:
                with self.subTest(system=system, app=app):
                    plan = self.plan(system, app)
                    self.assertEqual(len(plan["memory_endpoint_inventory"]), 7)
                    self.assertEqual(len(plan["memory_endpoints"]),
                                     6 if system == "carbink" else 7)
                    self.assertEqual(sum(e["selected"] for e in
                                         plan["memory_endpoint_inventory"]),
                                     6 if system == "carbink" else 7)
                    if system == "carbink":
                        spare = plan["memory_endpoint_inventory"][-1]
                        self.assertEqual(spare["memory_addr"], "10.208.130.84")
                        self.assertFalse(spare["selected"])
                        self.assertIn("server_count 6", plan["effective_config"])
                    elif system in ("starfish", "hydra"):
                        self.assertIn("ft_standby_endpoint 6", plan["effective_config"])

    def test_selection_does_not_mutate_inventory_and_renumbers_active_list(self):
        raw = self.raw()
        raw["memory_endpoint_indices_by_system"]["carbink"] = [5, 4, 3, 2, 1, 0]
        original = copy.deepcopy(raw)
        plan = self.plan("carbink", raw=raw)
        self.assertEqual(raw, original)
        self.assertEqual([e["index"] for e in plan["memory_endpoints"]], list(range(6)))
        self.assertEqual([e["inventory_index"] for e in plan["memory_endpoints"]],
                         [5, 4, 3, 2, 1, 0])

    def test_invalid_selected_endpoint_indices_are_rejected(self):
        for indices in ([], [0, 0], [True], [-1], [7], ["0"], "all"):
            with self.subTest(indices=indices):
                raw = self.raw()
                raw["memory_endpoint_indices_by_system"]["carbink"] = indices
                with self.assertRaises(ValueError):
                    self.plan("carbink", raw=raw)

    def test_fast_check_derived_sites_can_be_resolved_again(self):
        derived = fast_check.case_sites(DEFAULT, dry_run=True)
        for system, expected in (("nonft", 1), ("starfish", 7)):
            plan = self.plan(system, raw=derived[system])
            self.assertEqual(len(plan["memory_endpoints"]), expected)


if __name__ == "__main__":
    unittest.main()
