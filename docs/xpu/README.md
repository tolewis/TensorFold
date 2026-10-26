# Intel Arc Pro B70 (XPU) backend for the TensorFold Zig engine

**Unofficial** port: an `xpu` backend for the TensorFold Zig engine that drives an Intel Xe2 card (Arc Pro B70,
Battlemage G31, 32 GB) directly through **Level Zero**, with no PyTorch, no SYCL and no Python at run time. It serves
**Qwen3.8-27B** behind the OpenAI API of the `tensorfold-native` server, with the DFlash2 tree drafter.

## What it serves

| | |
|---|---|
| Model | Qwen3.8-27B (`model_type` `qwen3_5`: 48 GDN layers + 16 attention layers) |
| Formats | MLX affine 4-bit, group size 64 (`mlx-q4g64`); GPTQ symmetric 4-bit, group size 128 (`gptq-int4-g128`) |
| Measured checkpoints | GPTQ: [`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16) (revision `9d189a60`), with its tokenizer as published; full recipe: [RECIPE-qwen38-27b-b70.md](RECIPE-qwen38-27b-b70.md) |
| Drafts | DFlash2 (`--drafter`), best-first tree of 15 nodes, greedy and sampled |
| Serving | several streams (`--parallel N`, 8 by default, the engine's "lanes" host): shared rounds, a lone stream runs as a DFlash2 tree (also with `--parallel 1`) |

In the GPTQ format, the layers are int4 g128 as stored. The GDN `in_proj_a/b` gates stay in dense fp16, as in the
checkpoint, and the embedding is fp16. The output head is requantized at load time to affine 4/64 with MLX's
algorithm: its accuracy is that of the checkpoint's MLX 4-bit head (same token as the fp16 head in 97.4 % of cases,
KL 3.9e-3; `tools/xpu/head_study.py`).

## Building

Prerequisites: Zig 0.17, the `xe` driver and the Intel compute runtime (Level Zero), `ocloc` (Intel's offline
compiler), a C++17 compiler and the oneDNN nGEN headers.

```sh
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=<path to ngen>   # zig-out/xpu/bin/tensorfold-native
zig build xpu-qwen27 -Doptimize=ReleaseFast -Dngen-include=<path to ngen>   # xpu-qwen27 test executable
```

Device tests: `zig build xpu-qwen27-tests -Doptimize=ReleaseFast -Dngen-include=<path to ngen>`, then `zig-out/bin/tf-xpu-qwen27-means-test` (group means against FP64, no model), `zig-out/bin/tf-xpu-qwen27-seam-test MODEL` (or `TF_XPU_QWEN27=MODEL`: a prompt cut at 64, 1024 or whole gives the same logits) and `zig-out/bin/tf-xpu-qwen27-close-test MODEL` (three opens and closes give the card back); each prints `skipped:` and exits 0 without a Level Zero device or a model. Nemotron's op tests stay `zig build -Dxpu xpu-tests`.

The OpenCL C kernels are compiled by `ocloc` for `-Dxpu-device` (default `bmg`). The prompt GEMMs are generated as
ISA by nGEN at build time (`zig/tools/xpu/w4_ngen.cpp`); everything is embedded in the executable.

## Running

```sh
tensorfold-native serve MODEL --drafter DFLASH2 --parallel 8 --context 16384 --host 127.0.0.1 --port 8081
```

With `--parallel N` (N > 1), each stream has its own slot on the card: GDN state (2 × 151 MB, the state and its
backup for verification) and K/V cache (64 KB per token of `--context`). The number of slots is bounded by the memory
left after loading, as the device reports it (`slotsThatFit`); a stream that does not fit is refused at startup.
A lone stream that asks for drafts runs as a DFlash2 tree. As soon as another
stream arrives, the core takes it over in shared rounds: one forward for the windows of all streams (the pending
token, then either copies of the context as a chain or the stream's DFlash2 drafts: a tree from 5 drafts up, a chain
below), 16 rows at most. Each slot has its own DFlash2 context (the drafter's K/V in a ring of 2,080 positions,
~42 MB) which absorbs all of its kept rows; the drafts of all streams are proposed in a single pass of the drafter.
When the other streams finish, the one that remains goes back to a DFlash2 tree (the core's `Lone.resumes`).

Prefix cache (`--parallel N`): the core keeps the state of a slot at planned points of the prompt (multiples of 64,
since the prompt kernels compute in blocks of 64 rows from the start of a call), and a request whose prompt begins
the same way restarts from there: GDN and convolution states, K/V cache up to that point, and the stream's DFlash2
context (~0.8 GB at 9.7 k tokens). A resume gives exactly the state of a full pass cut at the same points: the same
answer, bit for bit. Budget: `--prompt-cache-gib` (2 GiB by default, reserved before the streams: 6 streams instead
of 7 in GPTQ at 16 k; `0` disables it). The core keeps nothing below 4,096 tokens. Measured on a 9.7 k-token
document: the same request replayed 6.1 → 1.2 s, the next turn of a conversation 6.1 → 1.1 s.
Prefix shared between different requests: put the common base in the **system** message (or in the tools); the
server places points at the end of the system prefix (and 512 / 2,048 tokens before it), and any request that shares
that system message restarts from there, whatever follows (measured: two other questions on the same 9.7 k-token
document in the system message, 6.1 → 1.2 s each, same answers as with an empty cache). A base placed in the user
message is reused only by the same request or by the following turns of the conversation.

Structured outputs (`response_format` JSON / `json_schema`, `guided_*` fields) with `--parallel N`: each row is
drawn among the tokens the grammar allows (argmax, or the keyed sampler over the allowed tokens). `libtfgrammar.so`
must be next to the server (`zig build grammar -Dxgrammar=DIR`, xgrammar v0.2.8, copied into `zig-out/xpu/lib/`). A
structured request does not go through the lone stream's DFlash2 tree: ~86 tok/s against ~145 on a long JSON answer.

Options through environment variables:

| Variable | Effect |
|---|---|
| `XPU_PREFILL_INT4=1` | **Local option**, with `XPU_PREFILL_INT8=1` and a rotated checkpoint (`zig/tools/xpu/rotate_w4.py`, marker `hadamard128`): `down` input in int4 (DPAS u4 × s4), +8 % prefill at equal quality to GPTQ W4A8; `XPU_PREFILL_INT4_SITES=10` adds the mixer outputs (+12 %, perplexity +1 %). |
| `XPU_KV_INT8=1` | **Option** (the default stays bf16): K/V cache in int8 (scale per position and K/V head): half the memory, 12 streams instead of 7 at 16 k tokens, 7 instead of 3 at 32 k; perplexity unchanged (±0.04 %), DFlash2 acceptance unchanged. Answers are not bit-identical to those of the bf16 cache (15 of 24 greedy answers identical over 256 tokens, the others diverge on near-ties); the drafted == serial and multi-stream == serial contracts hold in this mode. Prompt attention `attn_prompt8`: Q·Kᵀ in int8 DPAS, P·V in fp16. |
| `XPU_PREFILL_INT8=1` | **Local option**: prompt GEMMs in W4A8, int8 activations per (row, group of 128), GPTQ only. Decoding stays in fp16. |
| `XPU_TREE=N` | DFlash2 tree nodes per round (15 by default; 0: a chain) |
| `XPU_BLOCK=N` | Rows of the drafter block (8) |
| `XPU_FP64_REF=1` | `xpu-qwen27` runner only: on the prompt's real activations, each int4 GEMM of layers 0, 1, 31, 63 and the head (decode on 1 and 8 rows, prompt) against FP64 references of the same int4 weights (`core/quant` MLX affine; GPTQ as affine with bias −8·scale), worst column and error over the outputs' rms (`zig/src/xpu/fp64_ref.zig`) |
| `XPU_FIDELITY=1` | `xpu-qwen27` runner only: teacher-forced top tokens of the prompt and decode paths at every position of the IDS file against the same forward with every GEMM in FP64, all differences and margins (`XPU_FIDELITY_CACHE=FILE` keeps the FP64 forward's tokens) |

## K/V pool (Sage-A1.5, `qwen3_5_moe`)

The XPU Sage engine keeps the K/V of all its streams in one pool of pages. A page holds 64 positions. There is one
allocation for each of the 10 attention layers and one for the MTP head. A page id means the same 64 positions in
all 11 allocations, so one page table for each stream serves every layer. The GDN and convolution states stay one
fixed state for each slot.

```sh
tensorfold-native serve SAGE_MLX --vision --drafter SAGE_AUTOROUND --parallel 4 --context 262144 --kv-pool-tokens 700000
```

| Flag | Effect |
|---|---|
| `--context N` | The most positions one stream holds: its prompt and its reply. 262,144 by default. |
| `--kv-pool-tokens N` | The pool's size in tokens, rounded up to whole pages. Without it, the pool takes what the card leaves after the weights, the vision tower, the activations and the GDN state of every slot. It is never larger than every slot's whole context. |

A token takes 11,440 bytes of the pool with `XPU_KV_INT8=1` (11 allocations, K and V, 520 bytes each), and 22,528
bytes in bf16.

How a request gets its pages:

1. A request waits in the queue until the pool can promise pages for its prompt, its `max_tokens` and one verify
   window of 16 rows, at most `--context`. Requests behind it that fit can go first.
2. The prompt pass maps every page that the stream can need into its page table. A running stream takes no new page,
   so it never waits on the pool.
3. When the stream ends, its pages go back to the pool.

A request without `max_tokens` promises the server's `--max-tokens` (4,096 by default). A request that needs more
pages than the whole pool holds fails at once, and the message names `--kv-pool-tokens`. If a stream ever holds
fewer pages than its request asks for, its reply stops where the pages end, with `finish_reason: length` and a log
line. The kernels never write past the pages that a stream holds.

The load counts the pool beside the weights. The pool is zeroed at load, so a pool that the card cannot hold fails at
startup and not under a request. A refusal gives the most tokens the pool can hold. The startup line names the pool's
pages, tokens and GB. `/metrics` shows `tensorfold:kv_pool_pages`, `kv_pool_page_tokens`, `kv_pool_pages_used`,
`kv_pool_pages_promised` and `kv_pool_pages_free`.

Measured on one Arc Pro B70 (power cap 200 W), `XPU_KV_INT8=1`, MTP head, `--vision`, `--parallel 4`, greedy,
median of 3 (`benchmark_openai_chat.py`) unless the row says otherwise:

| | A fixed cache for each slot (4 x 220,000, the most that fits) | One pool (4 x 262,144, default pool) |
|---|---|---|
| K/V | 880,000 tokens, 10.07 GB | 994,176 tokens, 11.37 GB |
| 1k prompt, 1 stream, decode | 242.0 tok/s | 241.0 tok/s |
| 1k prompt, 4 streams, aggregate | 252.6 tok/s | 250.7 tok/s |
| 20k prompt, 1 stream, first token | 5.87 s | 5.90 s |
| 240k prompt alone (1 run) | refused (more than 220,000) | first token 181.9 s, 40.9 tok/s |
| 4 x 200k prompts at once | not run | 4 of 4 answer, the four hold 801,024 tokens of pages at once |
| 4 x 200k, `--kv-pool-tokens 700000` | not run | 3 run, the fourth waits until one ends, 4 of 4 answer |

Greedy replies over 256 tokens (4 prompts of 1k, 4 of 20k, each alone) are the same text with the pool as with the
fixed cache, and the greedy eval set scores the same (39/41). At 240k, one stream alone decodes 4.6 % slower than
with a fixed cache at `--parallel 1` (40.9 against 42.9 tok/s): the decode reads its K/V through the page table.

## Numbers (B70 at 230 W, server `--parallel 8 --context 16384`, validation of 2026-10-13)

| | MLX g64 | GPTQ g128 | GPTQ g128 + W4A8 | Rotated RTN + W4A4 |
|---|---|---|---|---|
| Prefill 1 k / 4 k / 8 k tokens (tok/s) | 1,783 / 1,772 / 1,742 | 1,975 / 1,965 / 1,930 | 2,617 / 2,585 / 2,540 | 2,830 / 2,811 / 2,754 |
| Single stream, DFlash2 tree, code / chat, greedy | 129 / 98 | 133 / 90 | 133 / 97 | 126 / 99 |
| 2 / 4 / 8 simultaneous streams, code, greedy (aggregate) | 211 / 294 / 326 | 235 / 310 / 270 | 235 / 310 / 270 | 222 / 288 / 260 |
| 2 / 4 / 8 simultaneous streams, chat, greedy (aggregate) | 151 / 239 / 300 | 148 / 215 / 208 | 162 / 230 / 224 | 162 / 249 / 227 |
| `bench_openai` decode, 1 / 2 / 4 / 8 requests (aggregate) | 173 / 243 / 343 / 350 | 165 / 300 / 337 / 439 | 165 / 336 / 367 / 456 | 179 / 325 / 364 / 463 |

Before the per-stream DFlash2 drafts (copies only in the shared rounds), 2 / 4 / 8 code streams ran at
64 / 120 / 182 tok/s in GPTQ. Validation: `zig build test`, oracle, drafted == serial, multi-stream == serial (chains
and trees), and `bench_concurrent --alone --serial` on 1, 2, 4 and 8 streams, greedy and sampled, for the four
configurations: 64 cells all equal, no errors in the server logs.

Benches: `bench_openai.py` (prefill, decode) and `bench_concurrent.py --alone --serial` (drafts), in `tools/` on the
`main` branch of this repository. Drafted throughput depends on the text produced; the two formats do not write the
same text.

Prefill is **power-bound**: at 230 W, the card runs at ~2,180 MHz during a prompt (PL2), not at the 2,600 of pure
DPAS. The gate|up GEMM alone sustains 146 TFLOPS fp16 and ~209 TOPS in int8.

## Correctness: what is verified

- **Oracle**: the greedy tokens equal those of the Python port's fp32 torch reference (same prompt, 32 tokens), in
  MLX and in GPTQ, with and without W4A8.
- **Drafted == serial**: bit for bit, with perfect, wrong and real drafts, as chain and as tree, greedy and sampled
  (`xpu-qwen27` with `XPU_VERIFY_TEST`, and `bench_concurrent --alone --serial` on the server).
- **Several streams == one**: bit for bit, up to 16 slots, with windows of 1 and 4 rows and partial keeps
  (`XPU_MULTI_TEST`), and each concurrent answer equal to the same request run alone
  (`bench_concurrent --alone`).
- **W4A8**: perplexity on three raw texts (English, code, French) +0.04 to +0.13 %, KL ≈ 3e-3 against fp16;
  for comparison, MLX and GPTQ differ from each other by 0.06 to 0.11 in KL
  (`tools/xpu/a8_study.py`).
- **Measure on raw text**: in a chat user turn, the model (the fp32 reference included) predicts the next token
  poorly, because it is trained only on the answers. Perplexities there are meaningless.

## Code layout

| File | Role |
|---|---|
| `zig/src/xpu/ze.zig` | Level Zero access: memory, modules, kernels, command lists |
| `zig/src/xpu/qwen27.zig` | The model: loading (MLX, GPTQ), layer-by-layer forward, `Runner` (`prompt`, `step`, `verify`, `verifyTree`, `commit`, `commitPath`; slots `select`, `verifyMulti`, `keepMulti`) |
| `zig/src/xpu/drafter.zig` | DFlash2: tap context, proposed block, selector, best-first tree, `treeRound` |
| `zig/src/native/xpu.zig` | The hook into the server: the "lanes" host for every `--parallel` |
| `zig/src/native/xpu_lanes.zig` | The `lanes` backend: slots, shared rounds (`verifyMulti`, `keepMulti`), lone stream as a tree (`lone`) |
| `zig/src/xpu/qwen27_main.zig` | Test executable: oracle, contracts, profiles, probes |
| `zig/kernels/xpu/qwen27/*.cl` | OpenCL C kernels: decode int4 GEMM, chunked and serial GDN, decode attention, norms, SwiGLU, int8 quantization, drafter |
| `zig/kernels/xpu/qwen27/decode.cl` (`attn_prompt`, `attn_prompt8`) | Prompt attention: FlashAttention-2 on DPAS in oneDNN's orientation (S = K·Qᵀ, one query per lane, softmax within the lane); bf16 or int8 cache |
| `zig/tools/xpu/w4_ngen.cpp`, `w4_prompt.hpp`, `w4a8_prompt.hpp` | nGEN generator of the prompt GEMMs: int4 fp16 (`w4g_dpb`, `w4g_s4dpb`, SwiGLU epilogue), W4A8 (`w4a8`) and W4A4 (`w4a4`) |
| `zig/build/xpu.zig` | Build steps |

No core file is modified: the backend uses the core's lane host and its lone driver (`Lone.resumes`) as they are;
`build.zig` only adds the XPU steps.

## Test executable

```sh
xpu-qwen27 MODEL ids.txt 32          # prompt (ids separated by spaces), 32 greedy tokens, timings
xpu-qwen27 MODEL --bench=4096 2      # synthetic prompt of 4,096 tokens (caution: does not replace real tokens)
```

| Variable | Effect |
|---|---|
| `XPU_DRAFTER=DIR`, `XPU_TREE`, `XPU_BLOCK` | drafted rounds, compared with serial |
| `XPU_VERIFY_TEST=1` | drafted == serial contract with perfect and wrong drafts |
| `XPU_MULTI_TEST=N` | multi-stream contract: N slots, shared rounds (one row, windows of 4 rows, then trees of 7 rows) == serial, and aggregate throughput |
| `XPU_ROWS_PROBE=1` | cost of a forward of 1 to 16 rows |
| `XPU_PROFILE=1` | GPU time per kernel |
| `XPU_DUMP_H=FILE` | final hidden states of each row (accuracy studies) |
| `XPU_A8=128` or `token`, `XPU_A8_SITES` | W4A8 study: simulated quantization of the prompt GEMM inputs |
| `XPU_CHUNK`, `XPU_CTX` | rows per prompt chunk (1,024), context (4,096) |
| `XPU_OVERLAP_PROBE`, `XPU_GEMM_ZEBIN`, `XPU_A8_ZEBIN` (+ `XPU_A8_PROJ`) | probes: GEMM/GDN overlap, fp16 or W4A8 GEMM in a loop |

## Limitations

- A single model: the 27B dimensions are constants (`qwen27.zig`) and some shapes are written into the kernels (GDN
  of 128, attention of 256, head ratio 6).
- On very repetitive text, where copies are enough, the drafter pass of the shared rounds costs a few percent at 4-8
  streams. No response resume yet (`decode_chunk`).
- 4-bit formats only (MLX g64, symmetric GPTQ g128). W4A8 requires GPTQ g128.
