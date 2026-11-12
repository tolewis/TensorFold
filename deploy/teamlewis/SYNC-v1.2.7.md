# Sync B70 line to upstream v1.2.7

## State

- Time: 2026-11-12 21:20:00Z
- Head: `577b16405` (`v1.2.7` merge `62b77f8b0` + `origin/lane/popeye-cuda`, clean)
- Step: 5 — compile on B70 host; step 6 popeye merge **done**

## Done

Merge conflict paths resolved and included in `62b77f8b0` (26 hand-merged paths in the dual-changed set; 0 `<<<<<<<` remain):

| Area | Files |
|------|-------|
| Build | `build.zig`, `zig/build/cuda.zig`, `zig/build/xpu.zig` |
| Core | `engine_api.zig`, `engine_status.zig`, `lane_host.zig`, `lane_host_fill.zig`, `lane_host_status.zig`, `prompt_cache.zig`, `prompt_cache_policy.zig`, `prompt_cache_status.zig`, `lanes/open.zig` |
| Server | `chat.zig`, `cli.zig`, `idle_unload.zig`, `main.zig`, `openai.zig`, `responses_translate.zig`, `serve.zig` |
| Native / XPU | `native/xpu.zig`, `native/xpu_lanes.zig`, `native/qwen27_lanes.zig`, `xpu/root.zig`, `xpu/dev.zig`, `xpu/kernels.zig`, `xpu/qwen27.zig` |
| Docs | `docs/api.md` |
| Kernels (fork + `sync-patches/`) | `decode.cl`, `gemm.cl`, `glue.cl`, `glue_attn.cl`, `glue_quant.cl`, `shape.h` |

Also: lean baseline on `62b77f8b0` (1481 findings). Popeye CUDA: 5 files merged in `577b16405` (`cuda_kernels.zig`, `ck_lane.cu`, `ck.zig`, `qlinear.zig`, `cuda_modelopt.zig`).

## Open

1. Compile `tensorfold-native` / XPU steps on jubiku (no zig on agent host).
2. Push `sync/v1.2.7` (then build on jubiku from pushed head).
4. Host gates (rollout §4).

**Conflict markers:** none — CL checkpoint “22 of 26 staged, 4 open” superseded by completed merge commit.

Exact next command (compile on jubiku):

```bash
ssh jubiku-linux 'cd /path/to/sync-v127 && nice -n 19 zig build xpu-native -Dtp=1'
```

(Adjust clone path on jubiku to this worktree or a pushed `sync/v1.2.7`.)

## Needs the kernel lane

| File | Note |
|------|------|
| `zig/kernels/xpu/qwen27/decode.cl` | Fork kept; upstream in `deploy/teamlewis/sync-patches/zig_kernels_xpu_qwen27_decode.cl.upstream` |
| `zig/kernels/xpu/qwen27/gemm.cl` | same pattern |
| `zig/kernels/xpu/qwen27/glue.cl` | same |
| `zig/kernels/xpu/qwen27/glue_attn.cl` | same |
| `zig/kernels/xpu/qwen27/glue_quant.cl` | same |
| `zig/kernels/xpu/qwen27/shape.h` | same |

## Compile and test results

(not run yet)

## Lean checker counts

HEAD `62b77f8b0`: 6249 files, 1481 findings (record only).

## popeye-cuda

Merged at `577b16405` (no conflicts).
