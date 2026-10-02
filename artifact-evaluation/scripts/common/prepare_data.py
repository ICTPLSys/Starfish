#!/usr/bin/env python3
"""Download and prepare the fixed LLaMA/BFS inputs used by the AE example."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = json.loads((HERE / "data_sources.json").read_text())


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def fingerprint(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     prefix="." + path.name, delete=False) as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        temporary = Path(f.name)
    temporary.replace(path)


def cached_download(path, url, expected):
    receipt = path.with_name(path.name + ".download.json")
    if not path.is_file() or not receipt.is_file():
        return False
    saved = json.loads(receipt.read_text())
    return (saved.get("url") == url and saved.get("expected") == expected
            and saved.get("file") == fingerprint(path))


def download(url, path, expected, token=None):
    """Resume .part downloads; hash new files once, reuse unchanged receipts."""
    if cached_download(path, url, expected):
        return path
    retrieval_url = url
    source_cache = os.environ.get("AE_SOURCE_CACHE")
    if source_cache:
        source = Path(source_cache).expanduser().resolve() / path.name
        if source.is_file():
            if source.stat().st_size != expected["size"]:
                raise ValueError(f"cached source has the wrong size: {source}")
            retrieval_url = source.as_uri()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"unverified existing input: {path}; choose a fresh data directory")
    partial = path.with_name(path.name + ".part")
    size = partial.stat().st_size if partial.exists() else 0
    if size > expected["size"]:
        raise ValueError(f"oversized partial download: {partial}")
    if size < expected["size"]:
        print(f"Downloading {path.name} ({expected['size']:,} bytes)", flush=True)
        command = ["curl", "--fail", "--location", "--continue-at", "-",
                   "--retry", "3", "--connect-timeout", "15",
                   "--speed-time", "120", "--speed-limit", "1024",
                   "--output", partial]
        # Do not put credentials in argv, URLs, logs, or manifests. curl strips
        # Authorization on cross-host redirects (do not use --location-trusted).
        if token:
            command += ["--header", "@-"]
        command += ["--url", retrieval_url]
        run(command, input=f"Authorization: Bearer {token}\n" if token else None,
            text=True)
    if partial.stat().st_size != expected["size"]:
        raise ValueError(f"incomplete download: {partial}")
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {expected['size']}\0".encode())
    with partial.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
            blob.update(chunk)
    actual = digest.hexdigest()
    if expected.get("sha256") and actual != expected["sha256"]:
        raise ValueError(f"SHA256 mismatch: {partial}; refusing to publish")
    if expected.get("git_blob") and blob.hexdigest() != expected["git_blob"]:
        raise ValueError(f"Git blob checksum mismatch: {partial}; refusing to publish")
    partial.replace(path)
    write_json(path.with_name(path.name + ".download.json"),
               {"url": url, "retrieved_from": retrieval_url,
                "expected": expected, "sha256": actual,
                "file": fingerprint(path)})
    return path


def model_url(name):
    source = SOURCES["llama"]
    return (f"https://huggingface.co/{source['repository']}/resolve/"
            f"{source['revision']}/{name}")


def ensure_tools(deps):
    source = SOURCES["llama"]
    checkout = deps / "llama2.c"
    if not checkout.exists():
        checkout.mkdir(parents=True)
        run(["git", "init", "--quiet", checkout])
    if not (checkout / ".git").is_dir():
        raise ValueError(f"not a managed exporter checkout: {checkout}")
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          text=True, capture_output=True)
    if head.returncode:
        run(["git", "-C", checkout, "fetch", "--depth", "1",
             source["exporter_repository"], source["exporter_revision"]])
        run(["git", "-C", checkout, "checkout", "--detach", "FETCH_HEAD"])
    elif head.stdout.strip() != source["exporter_revision"]:
        raise ValueError(f"exporter revision mismatch: {checkout}")
    dirty = run(["git", "-C", checkout, "status", "--porcelain",
                 "--untracked-files=no"], capture_output=True, text=True).stdout
    if dirty:
        raise ValueError(f"modified exporter source: {checkout}")

    venv = deps / "python"
    python = venv / "bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", venv])
    requirements = HERE / "requirements-data.txt"
    marker = venv / "prepared-requirements.txt"
    wanted = requirements.read_text()
    if not marker.exists() or marker.read_text() != wanted:
        run([python, "-m", "pip", "install", "-r", requirements])
        marker.write_text(wanted)
    run([python, "-c", "import torch, numpy, sentencepiece"])
    return checkout, python


def register_inputs(site, inputs):
    """Fill only empty/example paths. Never replace a user's configured input."""
    # BFS and NQ use the same prepared Friendster shards unless NQ was
    # explicitly configured. Record the alias for non-default data directories.
    inputs = dict(inputs)
    template = HERE / "site.eight-server.example.json"
    document = json.loads(site.read_text() if site.exists() else template.read_text())
    current = document.setdefault("inputs", {})
    if "bfs" in inputs:
        existing_bfs = current.get("bfs")
        graph = (existing_bfs if existing_bfs and not existing_bfs.startswith("/path/to/")
                 else inputs["bfs"])
        inputs.setdefault("nq", graph)
    for key, value in inputs.items():
        if current.get(key) and not current[key].startswith("/path/to/"):
            if current[key] != value:
                print(f"Keeping configured inputs.{key}: {current[key]}", flush=True)
            continue
        current[key] = value
    site.parent.mkdir(parents=True, exist_ok=True)
    write_json(site, document)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="llama,bfs",
                        help="comma-separated inputs (default: llama,bfs)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("AE_DATA_DIR", ROOT / "data/inputs")))
    parser.add_argument("--site", type=Path, default=ROOT / "data/site.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    selected = args.datasets.split(",")
    if not selected or len(set(selected)) != len(selected) or any(
            name not in ("llama", "bfs") for name in selected):
        parser.error("--datasets must select llama and/or bfs without duplicates")
    data = args.data_dir.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"datasets": selected, "data_dir": str(data),
                          "site": str(args.site.resolve()),
                          "sources": {name: SOURCES[name] for name in selected}},
                         indent=2))
        print("dry-run: no files written, no downloads, no conversion")
        return 0
    for tool in ("curl", "git", "gzip", "split", "wc"):
        if not shutil.which(tool):
            raise ValueError(f"missing {tool}; install the data preparation prerequisites")
    if sys.byteorder != "little":
        raise ValueError("the legacy LLaMA input format requires a little-endian host")
    token = os.environ.get("HF_TOKEN")
    if token and ("\n" in token or "\r" in token):
        raise ValueError("invalid HF_TOKEN")
    raw = data / "downloads"
    source_cache = Path(os.environ["AE_SOURCE_CACHE"]).expanduser() if os.environ.get("AE_SOURCE_CACHE") else None
    needs_model_download = "llama" in selected and any(
        not cached_download(raw / "llama" / name, model_url(name), expected)
        and not (source_cache and (source_cache / name).is_file())
        for name, expected in SOURCES["llama"]["files"].items())
    if needs_model_download:
        if not token:
            raise ValueError("Llama 2 access is gated. Obtain access from Meta/Hugging Face "
                             "and set HF_TOKEN for your authorized account, then rerun. "
                             "No model or graph has been downloaded.")
        # Check authorization on a small metadata file before any large transfer.
        run(["curl", "--fail", "--silent", "--show-error", "--location",
             "--head", "--connect-timeout", "15", "--max-time", "30",
             "--header", "@-", "--output", os.devnull, model_url("params.json")],
            input=f"Authorization: Bearer {token}\n", text=True)

    data.mkdir(parents=True, exist_ok=True)
    with (data / ".prepare.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another preparation is using {data}")
        # Peak workspace allowance includes compressed/raw data, output, and
        # graph splitting scratch. Never download into an already full disk.
        verified = {}
        for name in selected:
            output = data / name
            if not output.exists():
                continue
            if name == "llama":
                from prepare_llama import prepare
                if any(not cached_download(raw / "llama" / filename, model_url(filename), spec)
                       for filename, spec in SOURCES["llama"]["files"].items()):
                    raise ValueError("existing LLaMA output has missing or changed source records")
                verified[name] = prepare(raw / "llama", ROOT / "deps/data-tools/llama2.c",
                                         output, str(ROOT / "deps/data-tools/python/bin/python"))
            else:
                from prepare_friendster import prepare
                source = SOURCES[name]
                archive = raw / "com-friendster.ungraph.txt.gz"
                if not cached_download(archive, source["url"],
                                       {"size": source["size"], "sha256": source["sha256"]}):
                    raise ValueError("existing Friendster output has missing or changed source records")
                verified[name] = prepare(archive, output, expected_lines=source["lines"],
                                        expected_uncompressed_bytes=source["uncompressed_size"])
        needed = 0.0
        for name in selected:
            if name in verified:
                continue
            paths = ([raw / "llama" / item for item in SOURCES["llama"]["files"]]
                     if name == "llama" else [raw / "com-friendster.ungraph.txt.gz"])
            present = sum(candidate.stat().st_size for path in paths
                          for candidate in (path, path.with_name(path.name + ".part"))
                          if candidate.is_file())
            needed += max(0, {"llama": 48, "bfs": 76}[name] - present / (1 << 30))
        free = shutil.disk_usage(data).free / (1 << 30)
        if free < needed:
            raise ValueError(f"{data}: {free:.1f} GiB free, need about {needed:.1f} GiB "
                             "for selected inputs; set AE_DATA_DIR to larger storage")
        inputs = {}
        records = {}
        for name in selected:
            if name in verified:
                records[name] = verified[name]
                if name == "llama":
                    inputs.update(llama=str(data / "llama/llama2_7b_chat.bin"),
                                  llama_tokenizer=str(data / "llama/tokenizer.bin"))
                else:
                    inputs["bfs"] = records[name]["prefix"]
                continue
            if name == "llama":
                from prepare_llama import prepare
                meta = raw / "llama"
                for filename, expected in SOURCES["llama"]["files"].items():
                    download(model_url(filename), meta / filename, expected, token)
                checkout, python = ensure_tools(ROOT / "deps/data-tools")
                records[name] = prepare(meta, checkout, data / "llama", str(python))
                inputs.update(llama=str(data / "llama/llama2_7b_chat.bin"),
                              llama_tokenizer=str(data / "llama/tokenizer.bin"))
            else:
                from prepare_friendster import prepare
                source = SOURCES["bfs"]
                archive = download(source["url"], raw / "com-friendster.ungraph.txt.gz",
                                   {"size": source["size"], "sha256": source["sha256"]})
                records[name] = prepare(
                    archive, data / "bfs", expected_lines=source["lines"],
                    expected_uncompressed_bytes=source["uncompressed_size"])
                inputs["bfs"] = records[name]["prefix"]
        write_json(data / "inputs.json", {"inputs": inputs,
                   "sources": {name: SOURCES[name] for name in selected},
                   "preparation": records})
        register_inputs(args.site.resolve(), inputs)
    print(f"Prepared inputs: {data / 'inputs.json'}")
    print(f"Site configuration: {args.site.resolve()}")
    print("Set the site's SSH/RDMA addresses before starting experiments.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"data preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
