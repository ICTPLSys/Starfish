"""Shared CSV input, figure style and PNG/PDF export. No experiment execution."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

SYSTEM_STYLES = {
    "hydra": dict(label="Hydra", facecolor="#e7c66a", hatch=""),
    "carbink": dict(label="Carbink", facecolor="#b86b43", hatch="\\"),
    "starfish": dict(label="Starfish", facecolor="#4c78a8", hatch="|"),
    "nonft": dict(label="Non-FT", facecolor="#7aa37a", hatch=""),
}


def read_csv(path: Path, required: set[str]):
    """Read once so the content hash identifies the data actually plotted."""
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), strict=True)
    header = reader.fieldnames or []
    if len(set(header)) != len(header):
        raise ValueError("duplicate CSV column names")
    if not required.issubset(header):
        raise ValueError("missing CSV columns: " + ", ".join(sorted(required - set(header))))
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV line {reader.line_num}: wrong number of fields")
        rows.append((reader.line_num, {key: value.strip() for key, value in row.items()}))
    return rows, hashlib.sha256(payload).hexdigest()


def get_pyplot():
    """Headless rendering for SSH sessions; no display or LaTeX required."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; install scripts/common/requirements-plot.txt"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 25,
        "axes.labelsize": 21,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    return plt


def export_figure(fig, output_dir: Path, stem: str, metadata: dict):
    """Render first; publish metadata last, with hashes of both images."""
    import matplotlib
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [stem + ".png", stem + ".pdf", stem + ".json"]
    # Serialize writers. The JSON is the completion marker, not a promise
    # that three independent filenames can be replaced atomically.
    with (output_dir / ("." + stem + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another plot is writing {output_dir / stem}") from exc
        return _export_locked(fig, output_dir, stem, metadata, names, matplotlib.__version__)


def _export_locked(fig, output_dir, stem, metadata, names, version):
    with tempfile.TemporaryDirectory(prefix=".plot-", dir=output_dir) as temporary:
        stage = Path(temporary)
        for suffix in ("png", "pdf"):
            fig.savefig(stage / (stem + "." + suffix), dpi=220, bbox_inches="tight")
        manifest = dict(metadata, matplotlib_version=version)
        manifest["outputs"] = {
            name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
            for name in names[:2]
        }
        (stage / names[2]).write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # If publication is interrupted, no stale completion marker remains.
        (output_dir / names[2]).unlink(missing_ok=True)
        for name in names:
            os.replace(stage / name, output_dir / name)
    return [output_dir / name for name in names]
