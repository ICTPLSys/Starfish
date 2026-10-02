#!/usr/bin/env python3
"""Build Figure13 KV/server targets from the three in-repository runtimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def record(path):
    path = path.resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            digest.update(chunk)
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", default="starfish,hydra,carbink")
    p.add_argument("--build-root", type=Path, default=ROOT / "build")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    systems = args.systems.split(",")
    if args.jobs < 1 or len(set(systems)) != len(systems) or not set(systems) <= {"starfish","hydra","carbink"}:
        p.error("select distinct recovery systems and positive --jobs")
    for system in systems:
        directory = args.build_root.resolve() / system
        command = ["bash", str(ROOT / "scripts/common/build.sh"), "--system", system,
                   "--build-dir", str(directory), "--targets", "server,kvs_throughput",
                   "--jobs", str(args.jobs)]
        if args.dry_run:
            print(json.dumps(command)); continue
        subprocess.run(command, check=True)
        client = directory / "benchmark/kvs/kvs_throughput"
        server = directory / ("legacy/carbink_server/carbink_server" if system == "carbink" else "server")
        libraries = {}
        for binary in (client,server):
            result = subprocess.run(["ldd",str(binary)],text=True,capture_output=True,check=True)
            if "not found" in result.stdout:
                raise RuntimeError("unresolved build dependencies: "+str(binary))
            for name,path in re.findall(r"^\s*(\S+)\s+=>\s+(/\S+)",result.stdout,re.M):
                if re.match(r"lib(c|m|dl|rt|pthread|gcc_s|stdc\+\+)\.so",name): continue
                libraries[name] = dict(name=name,path=path)
        # A Zenodo/source archive need not contain a .git directory.
        commit, dirty = "unavailable-source-archive", None
        try:
            revision = subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],
                                      text=True,capture_output=True)
            if revision.returncode == 0:
                commit = revision.stdout.strip()
                dirty = bool(subprocess.check_output(
                    ["git","-C",str(ROOT),"status","--porcelain"],text=True).strip())
        except FileNotFoundError:
            pass
        manifest = dict(schema="figure13-source-build-v1",system=system,
                        source_commit=commit,source_dirty=dirty,
                        runtime_source=str(ROOT / "runtime" / system),
                        app_source=str(ROOT / "apps/kvs/throughput.cpp"),
                        artifacts=dict(client=record(client),server=record(server)),
                        libraries=list(libraries.values()))
        (directory / "figure13-build.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
