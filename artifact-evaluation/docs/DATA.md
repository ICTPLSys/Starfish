# Data and models

There are two ways to obtain inputs. Both use the same directory layout.

## Use the prepared data

On the provided servers, the prepared data is available read-only under
`/data/starfish-ae/`:

~~~text
/data/starfish-ae/
  llama/llama2_7b_chat.bin
  llama/tokenizer.bin
  bfs/graph.0 ... graph.31
  wordcount/enwiki-small-8g.txt
~~~

From `artifact-evaluation/`, check and register these inputs with:

~~~bash
bash scripts/common/use_prepared_data.sh
~~~

This checks that the files are readable and have the expected application
formats and sizes, then fills `data/site.json`. It does not download, convert,
or modify the shared data. For a different installation directory, set
`AE_PREPARED_DATA_DIR` before running the same command.

## Download and prepare from source

To download and process the original LLaMA and Friendster inputs yourself:

~~~bash
bash scripts/common/prepare_data.sh
~~~

The script downloads the source files, checks them, prepares the application
formats, and fills the input paths in `data/site.json`. Set the server IPs
in that file before running experiments. Existing custom input paths
are preserved.

## Before downloading

Llama 2 requires an account with approved access to
[Meta's Llama 2 7B Chat checkpoint](https://huggingface.co/meta-llama/Llama-2-7b-chat).
After obtaining access, make its Hugging Face token available as `HF_TOKEN`.
For example, read it without putting the token in shell history:

~~~bash
read -rsp "Hugging Face token: " HF_TOKEN; echo
export HF_TOKEN
bash scripts/common/prepare_data.sh
unset HF_TOKEN
~~~

The script does not accept license terms on your behalf. Tokens are not
written to configuration files, download manifests, or command arguments.

Inputs default to `data/inputs/`. Allow about 125 GiB of free disk space for
the initial preparation, including downloaded sources and temporary graph
files. The LLaMA conversion runs on CPU; use the reference 256-GiB host.
For another storage location, set `AE_DATA_DIR` before running the same command.

Downloads resume from `.part` files. Completed inputs are published only after
validation; subsequent runs reuse unchanged files. Source revisions, download
checksums, preparation details, and output paths are recorded in
`data/inputs/inputs.json` and per-input manifests. With `AE_DATA_DIR`, these
records are under that directory instead.

An existing archive of the original source files can be reused by setting
`AE_SOURCE_CACHE` to its directory. Cached source files still undergo the
same pinned checksum checks; the download record distinguishes the upstream
URL from the local cache used to retrieve the bytes.

The helper uses Python 3, curl, Git, gzip, and GNU coreutils. It creates an
isolated Python environment under `deps/data-tools/` for the pinned model
conversion dependencies; it does not change system Python.

## LLaMA

The source is the original Meta checkpoint, pinned to revision
`617522e5466cec3391aac2de97f4f9d5672ed05b`. The script verifies the downloaded
checkpoint and tokenizer checksums and uses
[llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb)
at revision `350e04fe35433e6d2941dce5a1f53308f87058eb` to export version 0
(FP32) and its binary tokenizer.

The prepared model is `llama/llama2_7b_chat.bin` (26,954,711,068 bytes).
The script validates the seven-field model header and all 32,000 tokenizer
records. GGUF and quantized checkpoints are not interchangeable with this
format. The chat input is supplied in `apps/llama/llama_user_chat.txt`.

## Friendster

The script downloads the complete
[SNAP Friendster undirected edge list](https://snap.stanford.edu/data/com-Friendster.html).
It checks the compressed file size and SHA256 against the archived original
download, gzip integrity, uncompressed size
(32,364,651,776 bytes), and line count (1,806,067,135 edges plus four header
lines).

It then reproduces the original preparation procedure: divide the total
line count by 32, round up, and split at those line boundaries. The output is
`bfs/graph.0` through `bfs/graph.31`, preserving the original text and order.
There is no sorting, filtering, or vertex renumbering in preparation. The
application itself remaps vertex IDs and adds the reverse edge; preparation
must not duplicate the edge directions.

Set `inputs.bfs` to the common prefix ending in `bfs/graph`, not to a
particular shard. The download helper sets this automatically.

## Other workloads

KV records and requests, NQ queries, and MG grids are generated at runtime
using the workload parameters specified in the paper; NQ uses the prepared
Friendster graph.

WordCount uses the prepared `enwiki-small-8g.txt` input derived from the
2019-02-01 English Wikipedia XML snapshot. Its expected size is 8,053,063,696
bytes and its recorded SHA256 is
`f7dda4098b30d03891cf0b287f03e804248126eed85e3534c4d5adac5e6e998b`.
Use this prepared input with the [WordCount application](../apps/wordcount/README.md).
