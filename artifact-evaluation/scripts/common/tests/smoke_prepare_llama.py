"""Exercise the real pinned export/tokenizer CLI on a small generated checkpoint.

Run using deps/data-tools/python/bin/python after preparing model tools.
This is an interface test, not a paper-scale model conversion or benchmark.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from unittest.mock import patch

import torch

COMMON = Path(__file__).resolve().parents[1]
ROOT = COMMON.parent.parent
TOOLS = ROOT / "deps/data-tools/llama2.c"
sys.path[:0] = [str(COMMON), str(TOOLS)]
import prepare_llama
from model import ModelArgs, Transformer

with tempfile.TemporaryDirectory(prefix="ae-real-export-test-") as directory:
    root = Path(directory)
    meta = root / "meta"
    meta.mkdir()
    config = ModelArgs(dim=16, n_layers=2, n_heads=2, n_kv_heads=2,
                       vocab_size=32000, multiple_of=8)
    model = Transformer(config)
    state = model.state_dict()
    state["output.weight"] = state["output.weight"].clone() + 1
    torch.save(state, meta / "consolidated.00.pth")
    (meta / "params.json").write_text(json.dumps(dict(
        dim=16, n_layers=2, n_heads=2, multiple_of=8, norm_eps=1e-5)))
    tokenizer = TOOLS / "tokenizer.model"
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == (
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347")
    shutil.copyfile(tokenizer, meta / "tokenizer.model")
    # Legacy v0 appends cos/sin RoPE tables to all parameter tensors.
    expected_size = 28 + 4 * (sum(value.numel() for value in state.values()) + 2048 * 8)
    with patch.object(prepare_llama, "MODEL_HEADER", (16, 48, 2, 2, 2, -32000, 2048)), \
         patch.object(prepare_llama, "MODEL_SIZE", expected_size):
        first = prepare_llama.prepare(meta, TOOLS, root / "prepared", sys.executable)
        second = prepare_llama.prepare(meta, TOOLS, root / "prepared", sys.executable)
    assert first["model"]["size"] == expected_size
    assert second["reused"]
    generated = (root / "prepared/tokenizer.bin").read_bytes()
    assert hashlib.sha256(generated).hexdigest() == (
        "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361")
    assert (meta / "tokenizer.model").read_bytes() == tokenizer.read_bytes()
    print("PASS: actual export.py/tokenizer.py interfaces, atomic publication, reuse")
    print("PASS: tokenizer matches the historical experiment tokenizer")
    print("Small generated checkpoint only; no full 7B model conversion or RDMA run.")
