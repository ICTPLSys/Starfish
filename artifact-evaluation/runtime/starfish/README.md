# Starfish runtime

The runtime builds the shared LLaMA and BFS applications in `../../apps/`.
Selective remote backup is organized under `include/design1/` and
`src/design1/`. Cache, allocation, RDMA, and runtime utilities remain under
their corresponding `include/` and `src/` directories.

From `artifact-evaluation/` on a Linux server:

```bash
bash scripts/common/build.sh --system starfish --jobs 4
```

The build uses the dependency setup described in
[the scripts guide](../../scripts/README.md) and writes binaries under
`build/starfish/`.
