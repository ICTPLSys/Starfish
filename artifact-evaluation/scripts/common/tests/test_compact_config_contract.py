"""Contract tests for the compact AE recipe boundary.

These tests deliberately exercise only planning/rendering. They do not start
an application or a memory service. Deployment identity belongs to the site
and machine layers; a recipe is an application/runtime policy overlay.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest


COMMON = Path(__file__).resolve().parents[1]
AE_ROOT = COMMON.parents[1]
sys.path.insert(0, str(COMMON))

from render_config import load_recipe, render  # noqa: E402
from workloads import FOOTPRINT_BYTES  # noqa: E402


SYSTEMS = {"nonft", "starfish", "hydra", "carbink"}
DEPLOYMENT_KEYS = {
    "server_count", "server_addr", "server_port", "server_addrs", "server_ports",
    "ib_device_name", "ib_port",
}


def _values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s+(\S+)", line)
        if not match or line.lstrip().startswith("#"):
            continue
        key, value = match.groups()
        if key in result:
            raise AssertionError(f"duplicate rendered key: {key}")
        result[key] = value
    return result


def _runtime_keys(system: str) -> set[str]:
    path = AE_ROOT / "runtime" / system / "include" / "rdma" / "config.def"
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*CONFIG\([^,]+,\s*([A-Za-z_][A-Za-z_0-9]*)\s*,", line)
        if match:
            keys.add(match.group(1))
    return keys


def _system(recipe: Path) -> str:
    if recipe.name == "recomputable_ec.config":
        return "starfish"
    return recipe.stem


def _recipes() -> list[Path]:
    return sorted(AE_ROOT.joinpath("configs").glob("*/*.config"))


def _endpoints(count: int) -> list[dict[str, object]]:
    return [{"server_addr": f"198.51.100.{index + 1}",
             "server_port": 1900 + index} for index in range(count)]


def _render(recipe: Path, *, ratio: int = 25, count: int = 1,
            backup: bool | None = None, resident: bool | None = None) -> dict[str, str]:
    app = recipe.parent.name
    system = _system(recipe)
    text = render(recipe, system=system, ratio=ratio,
                  footprint_bytes=FOOTPRINT_BYTES[app],
                  server_endpoints=_endpoints(count),
                  ib_device="mlx5_contract", ib_port=1,
                  backup_enabled=backup, resident_enabled=resident)
    return _values(text)


class CompactConfigContract(unittest.TestCase):
    def test_all_recipes_are_runtime_fields_without_deployment_identity(self):
        recipes = _recipes()
        self.assertEqual(len(recipes), 34)
        for recipe in recipes:
            system = _system(recipe)
            direct = _values(recipe.read_text(encoding="utf-8"))
            self.assertTrue(set(direct) <= _runtime_keys(system), recipe)
            self.assertFalse(set(direct) & DEPLOYMENT_KEYS, recipe)

    def test_external_endpoint_count_and_ib_identity_win_for_every_recipe(self):
        for recipe in _recipes():
            for count in (1, 6, 7):
                values = _render(recipe, count=count)
                self.assertEqual(values["server_count"], str(count), recipe)
                self.assertEqual(values["server_addr"], "198.51.100.1", recipe)
                self.assertEqual(values["server_port"], "1900", recipe)
                self.assertEqual(values["server_addrs"], ",".join(
                    f"198.51.100.{i}" for i in range(1, count + 1)), recipe)
                self.assertEqual(values["server_ports"], ",".join(
                    str(1899 + i) for i in range(1, count + 1)), recipe)
                self.assertEqual(values["ib_device_name"], "mlx5_contract", recipe)
                self.assertEqual(values["ib_port"], "1", recipe)

    def test_loader_values_are_known_runtime_fields_and_modes_remain_explicit(self):
        expected = {"nonft": "none", "starfish": "ec_batch",
                    "hydra": "hydra", "carbink": "carbink"}
        for recipe in _recipes():
            system = _system(recipe)
            loaded = _values(load_recipe(recipe))
            self.assertTrue(set(loaded) <= _runtime_keys(system), recipe)
            rendered = _render(recipe)
            self.assertEqual(rendered.get("ft_method"), expected[system], recipe)

    def test_feature_overrides_zero_only_the_requested_features(self):
        for recipe in _recipes():
            values = _render(recipe, backup=False, resident=False)
            self.assertEqual(values["enable_selective_backup"], "0", recipe)
            self.assertEqual(values["remote_backup_budget_bytes"], "0", recipe)
            self.assertEqual(values["remote_backup_budget_pct"], "0", recipe)
            self.assertEqual(values["local_resident_budget_bytes"], "0", recipe)
            for key in ("enable_region_resident_placement",
                        "enable_resident_profile_planner",
                        "resident_profile_apply_plan",
                        "enable_region_hotness_placement",
                        "enable_region_fetch_hotness_placement",
                        "region_placement_bind_groups"):
                self.assertEqual(values[key], "0", f"{recipe}: {key}")


if __name__ == "__main__":
    unittest.main()
