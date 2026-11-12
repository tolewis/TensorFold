# Sync B70 line to upstream v1.2.7

## State

- Time: 2026-11-12 19:56:42Z
- Head: `5beb52d87da1376f63e8254854267cd145d1acfa` (status commit; merge in progress, not committed)
- Step: 3 (merge `v1.2.7` started; conflict resolution in progress)

## Done

1. Fetched `origin` and `upstream --tags`.
2. Checked out `sync/v1.2.7` from `origin/lane/fleet-unified`. Start sha `73efe744175514172091fad5b6913344a6c3ccea`.
3. Wrote dual-changed list (51 files). Committed status file `5beb52d87`.
4. Ran `git merge v1.2.7` (32 unmerged paths).
5. Resolved qwen27 kernel `.cl` / `shape.h` on fork side; upstream copies under `deploy/teamlewis/sync-patches/` (`*_qwen27_*.upstream`).
6. Resolved: `build.zig`, `zig/build/cuda.zig`, `engine_api.zig`, `engine_status.zig`, `lane_host.zig`, `lane_host_fill.zig`, `lane_host_status.zig`, `prompt_cache_policy.zig`, `prompt_cache_status.zig` (`noteFamilyPages`).

## Open

1. Finish conflict markers in remaining paths (see counts below). Next: `zig/build/xpu.zig`, then `zig/src/core/prompt_cache.zig`, `lanes/open.zig`, server, native, xpu sources, `docs/api.md`.
2. `git commit` the merge when no `<<<<<<<` remain.
3. Compile and CPU test steps (step 5 in rollout).
4. Steps 6 to 8 (popeye-cuda, lean_check, push).

Exact next command:

```bash
cd /home/tlewis/projects/TensorFold/.worktrees/sync-v127 && rg -l '^<<<<<<<' --glob '*.zig' --glob '*.md'
```

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

(not run yet)

## popeye-cuda

`origin/lane/popeye-cuda` not present at last fetch (step 6 pending).

## Conflict progress

Rough marker count in worktree: falling (re-check with `rg -c '^<<<<<<<'`).
