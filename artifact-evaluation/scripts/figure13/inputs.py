"""Validate measured Figure13 batch manifests without relabelling repeats."""
import json
from pathlib import Path

def load_manifest(path, repeat=1):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema") != "figure13-native-runs-v1" or repeat < 1:
        raise ValueError("expected figure13-native-runs-v1 and positive repeat")
    entries, seen = [], set()
    indexed = "batch_plan" in payload
    batch_cases = {}
    if indexed:
        index_path = Path(payload["batch_plan"])
        if not index_path.is_absolute():
            index_path = path.parent / index_path
        batch = json.loads(index_path.read_text())
        if batch.get("schema") not in (
                "figure13-batch-v1", "figure13-site-batch-v1"):
            raise ValueError("invalid batch-plan schema")
        cases = batch.get("cases", batch.get("runs"))
        if not isinstance(cases, list):
            raise ValueError("batch plan lacks cases/runs")
        for case in cases:
            if not isinstance(case, dict) or "result" not in case:
                raise ValueError("batch plan case lacks result")
            source = Path(case["result"])
            if not source.is_absolute():
                source = index_path.parent / source
            batch_cases[str(source.resolve())] = case
        payload["_batch_schema"] = batch.get("schema")
    elif repeat != 1:
        raise ValueError("historical native files are unindexed; --repeat cannot relabel them")
    for raw in payload["runs"]:
        entry = dict(raw)
        if indexed:
            actual_repeat = entry.get("repeat")
            if isinstance(actual_repeat, bool) or not isinstance(actual_repeat, int):
                raise ValueError("indexed run repeat must be an integer")
            if actual_repeat < 1:
                raise ValueError("indexed run repeat must be positive")
            if actual_repeat != repeat:
                continue
        source = Path(entry["result"])
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        key = (entry["scenario"], entry["system"])
        if key in seen:
            raise ValueError("duplicate selected condition: " + str(key))
        if key[0] not in ("1-node", "2-node") or key[1] not in (
                "hydra", "carbink", "starfish"):
            raise ValueError("unknown selected condition: " + str(key))
        if indexed:
            case = batch_cases.get(str(source))
            if case is None or any(case.get(k) != entry.get(k)
                                   for k in ("system", "scenario", "repeat")):
                raise ValueError("run entry disagrees with batch-plan")
            if source.name == "figure13-result.json":
                result = json.loads(source.read_text())
                if result.get("schema") != "figure13-site-result-v1":
                    raise ValueError("site result has invalid schema")
                for field in ("system", "scenario", "repeat", "run_id"):
                    if field in entry and result.get(field) != entry[field]:
                        raise ValueError("site result disagrees with runs.json: " + field)
                if result.get("manifest") != "manifest.json":
                    raise ValueError("site result must reference manifest.json")
                manifest_path = source.parent / result["manifest"]
                if not manifest_path.is_file():
                    raise ValueError("site result manifest is missing")
                site_manifest = json.loads(manifest_path.read_text())
                if site_manifest.get("run_id") not in (None, entry.get("run_id")):
                    raise ValueError("site manifest run_id disagrees with runs.json")
            else:
                plan_path = source.parent / "plan.json"
                if not plan_path.is_file():
                    raise ValueError("indexed legacy native result lacks plan.json")
                plan = json.loads(plan_path.read_text())
                own = plan.get("figure13_batch_case", {})
                if any(own.get(k) != entry[k]
                       for k in ("system", "scenario", "repeat")):
                    raise ValueError("native plan does not confirm indexed repetition")
        seen.add(key)
        entry["result"] = str(source)
        entries.append(entry)
    if not entries:
        raise ValueError("no measured native runs in selected repetition")
    return payload, entries, indexed
