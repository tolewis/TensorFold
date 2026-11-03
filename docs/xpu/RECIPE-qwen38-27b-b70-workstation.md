# Recipe: Qwen3.8-27B on one Arc Pro B70 that also drives the desktop

This recipe serves Qwen3.8-27B from one Intel Arc Pro B70 in a workstation. The same card drives the desktop
display. The server loads the model on the first request, and it gives the VRAM back to the desktop after 300 s
idle. Production since 2026-11-03: one stream of 217,088 tokens with K1 decode attention and DFlash2 chains.

Measured on jubiku on 2026-11-02 and 2026-11-03. For the two-card server recipe, see
[RECIPE-qwen38-27b-b70.md](RECIPE-qwen38-27b-b70.md).

## Production benchmark (2026-11-03, p39 K1)

Build `lane/p39-k1cover`, commit `138de4f4b3c6b6adc240bc8a39b31bf3b692941e`, with `XPU_K1=1 XPU_TREE=0`. One stream of
217,088 tokens in an int8 pool of 245,760 tokens (8.18 GB). VRAM 26.1 GiB loaded.

| Context | K1 chain (production) | K1 tree (`XPU_K1_TREE=1`) | p36, scalar tree |
|---|---|---|---|
| 30 tokens, 300 out | 142 tok/s | 136 tok/s | 153 tok/s |
| 33 tokens, 64 out (code) | 197 tok/s | 165 tok/s | 191 tok/s |
| 39,464 | 122 tok/s | 109 to 123 tok/s | 44 tok/s |
| 120,173 | 91.5 tok/s | 99 to 102 tok/s | 16 to 18 tok/s |

Correctness, greedy:

- K1 plain at 120k gives sha `55532ae52f6c`, the same as K1 off. On p36, K1 drifted (`b85485f84991`).
- K1 chain + DFlash2 at 120k gives `55532ae52f6c`.
- K1 tree at 120k gives `07fc38779555` on both runs. The text is sane, but it is not the reference sha. This is why
  production uses chains.
- K1 tree at 39k: `70278af8e8b3` on both runs, the same as the chain. Acceptance 257/720.
- K1 off, tree, 30-token prompt: `d317b4792cff` on both runs. On p36 the repeats were not the same.
- No `xe` hang, reset, or fault in the kernel log during the tests.

Prefill is about the same as p36: 39k in 26.7 s, 120k in 115 to 121 s.

### Agent harness settings

pi always sends its model `maxTokens` (32,768) as `max_tokens`. The stock server refuses a request when prompt +
`max_tokens` is more than the window. Production runs `lane/p42-replyclamp` (commit `7d08c424b`, p39 K1 plus a
reply fit) with `TF_REPLY_FLOOR=1`. The server cuts `max_tokens` to the room left, and the engine keeps 17 positions
for the drafter, so a reply that fills the window is cut 17 tokens short and never fails. Tested on 2026-11-03:
`max_tokens` 217,088, 217,060 and 300,000 with a 30-token prompt all return 200. The earlier p39 clamp
(`TF_CLAMP_MAX_TOKENS=1`) gave a 500 `PromptTooLong` when `max_tokens` filled the window. Do not use it.

pi compaction rule: `reserveTokens` = one full reply (32,768), not a percentage of the window. pi compacts at
217,088 − 32,768 = 184,320 tokens and keeps the last 32,768 tokens after a compaction. Fleet policy:
agent-loadout `pi/compaction-policy.json`.

## Two-stream benchmark (2026-11-02, p36)

| Item | Value |
|---|---|
| Streams | 2 × 114,688 tokens of context |
| Total K/V cache | One int8 pool of 245,760 tokens: 3,840 pages, 8.18 GB |
| Prompt cache | Kept prompt states share the pool (`XPU_KEEP_SHARED=1`), up to 7.5 GiB |
| VRAM, loaded | 25.2 GiB after open, 25.9 GiB with a 99k prompt kept. The card has 30.3 GiB. The desktop uses about 2 GiB. |
| VRAM, idle | 2.7 MB after `--idle-unload 300`. The next request reloads the model in about 10 to 28 s. |

Prompt processing (prefill), 1 stream:

| Prompt | Time to first token | Rate |
|---|---|---|
| 30 tokens | 0.10 s | n/a |
| 39,464 tokens | 26.4 s | 1,490 tok/s |
| 98,758 to 99,011 tokens | 86 to 94 s | 1,050 to 1,150 tok/s |
| 179,931 tokens (1-stream 217k config) | 205 s | 877 tok/s |
| Next turn, same prefix, 99k to 180k | 0.3 to 0.6 s | resumes from the prompt cache |

Decode, greedy, DFlash2 drafter:

| Load | Context | tok/s each | tok/s total |
|---|---|---|---|
| 1 stream, code, 64 tokens | 33 | 191 to 194 | 191 to 194 |
| 1 stream, 300 tokens | 30 | 153 | 153 |
| 2 streams, 300 tokens each | 30 | 120 to 130 | about 250 |
| 1 stream | 39,464 | 44 | 44 |
| 1 stream | 120,173 | 16 to 18 | 16 to 18 |
| 2 streams, same cached prompt | 98,758 | 13.7 and 13.9 | 27.6 |

A lone stream drafts with a DFlash2 tree. Two streams use chains in shared rounds. The two 99k streams gave the
same sha (`75dc6c44ecfd`) and the same text.

## Hardware and software

| Item | Value |
|---|---|
| Host | Dell Precision 7820, 2 × Xeon Platinum 8168 (Skylake-SP), 157 GB RAM |
| GPU | Intel Arc Pro B70, 32 GB, `xe` driver, stock 275 W. Do not set a power cap. |
| Runtime | Level Zero 1.32.0, compute-runtime 26.35.39758.10 |
| Model | `Qwen3.8-27B-Uncensored-MLX-4bit` (MLX affine 4-bit, group 64) |
| Drafter | `Qwen3.8-27B-DFlash2` |
| Engine | TensorFold `lane/p36-mlx-tp2`, commit `a6bcd2a3ffa83996fdec8699fbeadb9c9d372e87`. This lane is not on `main` yet. |

MLX 4/64 is faster than EXL3 on Xe. The verify window stays at 32 to 37 ms from 1 to 16 rows. EXL3 went from 49 ms
to 82 ms. MLX decode was about 2× the EXL3 decode on the same card.

## Build

Build on a host that has the toolchain. Set `-Dcpu` to the target CPU. A build for a newer Xeon fails with SIGILL
on Skylake-SP. The build does not open the GPU.

```sh
git archive a6bcd2a3f | tar -x -C src-p36
cd src-p36
zig build xpu-native -Dtp=1 -Doptimize=ReleaseFast -Dcpu=skylake_avx512 \
  -Dngen-include=<oneDNN>/third_party/ngen -p <out>
```

Copy `<out>/xpu/bin/tensorfold-native` to the workstation.

## Configuration

`qwen27.env`:

```sh
QWEN_BIN=$HOME/b70/qwen27/bin/tensorfold-native-a6bcd2a3f-p36-tp1-skx
QWEN_MODEL=$HOME/models/qwen38-27b/Qwen3.8-27B-Uncensored-MLX-4bit
QWEN_DRAFTER=$HOME/models/qwen38-27b/Qwen3.8-27B-DFlash2
QWEN_PARALLEL=2
QWEN_CONTEXT=114688
QWEN_POOL_TOKENS=245760
QWEN_IDLE_UNLOAD=300
XPU_MTP=0
XPU_KV_POOL=1
XPU_EMBED_HOST=1
XPU_KEEP_SHARED=1
XPU_CHUNK=896
```

The serve script turns these into `serve` flags:

```sh
export XPU_KV_INT8=1
tensorfold-native serve "$QWEN_MODEL" --drafter "$QWEN_DRAFTER" \
  --parallel 2 --context 114688 --kv-pool-tokens 245760 --idle-unload 300 \
  --host 127.0.0.1 --port 18197 --no-update-check
```

What each setting does:

- `XPU_KV_POOL=1` with `--kv-pool-tokens` gives all streams one paged int8 K/V pool. A stream takes pages as its
  context grows.
- `XPU_EMBED_HOST=1` keeps the embedding table in host memory. This saves 0.72 GB of VRAM, and that space goes to
  the pool.
- `XPU_KEEP_SHARED=1` stores kept prompt states in free pool pages. A later turn resumes at any length.
- `--idle-unload 300` closes the engine after 300 s idle. The process stays up. Kept states are lost at an unload.

## Run it

A systemd user unit runs the server at `Nice=5`, `CPUWeight=50`, `MemoryHigh=24G`, and `MemoryMax=32G`. Then the
desktop stays responsive. The unit is not enabled at boot. A wrapper starts and stops it:

```sh
qwen-local up       # start, wait for /health
qwen-local status   # up, loaded: 25.2 GiB of B70 VRAM | up, unloaded (idle) | down
qwen-local down     # stop, VRAM back to the desktop
```

The server listens on 127.0.0.1 only. `tailscale serve --bg --tcp 18097 tcp://127.0.0.1:18197` publishes it on the
tailnet. Nothing listens on the LAN.

## Limits

- Two different 100k prompts cannot both stay in the prompt cache. One 99k prompt keeps two states of about
  3.4 GiB, and the cache holds about 7.5 GiB. A second long prompt drops the first prompt's states.
- Three streams of 114,688 do not fit with the desktop on the card. They need about 3.3 GB more pool.
- Long-context decode is slow: 16 to 18 tok/s at 120k and about 8 tok/s at 180k. Decode attention is the
  limit.
- A DFlash2 tree at temperature 0 can give different text on repeat runs (`d317b4792cff`, `7b62cdadaf9b`).

## Phase two (done 2026-11-03)

The p39 tests passed, and p39 with `XPU_K1=1 XPU_TREE=0` is in production. See the production benchmark above.
Open items: the K1 tree is not exact against the reference, and two streams of K1 at 99k are not measured.

The plan as written on 2026-11-02:

1. `XPU_K1=1` (K1 decode attention) doubles plain decode at 120k, from 12.2–12.7 to 23.4–24.4 tok/s. The
   text matches for the first 300 characters, then it drifts from the K1-off text (greedy drift). K1 does not run
   on DFlash2 trees.
2. `XPU_K1_TREE=1` on the p36 build gives wrong text at 40k and 120k. Acceptance falls to 0.2–0.4 %.
3. `lane/p39-k1cover` (commit `138de4f4b3c6b6adc240bc8a39b31bf3b692941e`) has the fixes for both: a per-layer host table for
   the K1 tree race, `attn_merge_span2` with exp2, and a K1 chain for `XPU_TREE=0`. The jubiku build exists
   (`tensorfold-native-138de4f4-p39-tp1-skx`). Its four acceptance tests have not run: K1 tree at 39k (2 reps),
   K1 plain at 120k against sha `55532ae52f6c`, a K1 chain with DFlash2 at 120k, and a 30-token tree repeat at
   temperature 0.
4. If p39 passes, use it in production with `XPU_K1=1`, and measure again with 2 streams at 99k.
