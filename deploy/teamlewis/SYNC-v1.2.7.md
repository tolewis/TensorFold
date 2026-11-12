# Sync B70 line to upstream v1.2.7

State: done

## State

- Time: 2026-11-12 21:45:00Z
- Head: `0a08fd1b8` on `origin/sync/v1.2.7` (v1.2.7 merge + fleet-unified deploy + popeye-cuda + 6 compile-fix commits)
- Step: compile gate on server done; host rollout §4 still open on B70 fleet

## Done

- Pushed `sync/v1.2.7` (no force).
- Merged `origin/lane/fleet-unified` (`61d1f48a3`) and `origin/lane/popeye-cuda` (`577b16405`).
- Server compile (Zig 0.17.0, capacity lease, `nice -n 19`):
  - **TP2** (`inference1` README): `zig build native -Dtp=2 -Dcpu=skylake -Dversion=1.0.0 -p /tmp/sync-v127-build/tp2` → **ok** (`native/bin/tensorfold-native`).
  - **TP1** (`jubiku-linux` README): `zig build xpu-native … -p /tmp/sync-v127-build/tp1` → **not on server** (no `ocloc`, no nGEN headers under `oneDNN/ngen`).
  - **Popeye CUDA** (`deploy/teamlewis/popeye/README.md` `zig build native`): **not on server** (no CUDA toolkit/GPU here; not attempted).
- Compile fixes pushed (one commit each): duplicate CUDA test steps, `decode_share` / `isActiveRewind` / `before()` / `short(h,j)`.
- Lean: `python3 tools/zig/lean_check.py` on HEAD and on tag `v1.2.7` worktree (removed after run).

## Open

- Host gates (rollout §4): deploy only after jubiku/inference1 builds with ocloc+ngen on B70 hosts.
- Named XPU CPU tests that link `qwen27` (`test-mode-word`, `test-draft-win`, `test-k1-*`): need B70 toolchain or `-Dxpu-images=…` zebins (same blocker as TP1).

## Needs the kernel lane

| File | Note |
|------|------|
| `zig/kernels/xpu/qwen27/decode.cl` | Fork kept; upstream in `deploy/teamlewis/sync-patches/` |
| `zig/kernels/xpu/qwen27/gemm.cl` | same |
| `zig/kernels/xpu/qwen27/glue.cl` | same |
| `zig/kernels/xpu/qwen27/glue_attn.cl` | same |
| `zig/kernels/xpu/qwen27/glue_quant.cl` | same |
| `zig/kernels/xpu/qwen27/shape.h` | same |

## Compile and test results (server 2026-11-12)

| Target | Result |
|--------|--------|
| TP2 `native` | **pass** |
| TP1 `xpu-native` | **blocked** (no ocloc/ngen) |
| Popeye CUDA `native` | **N/A** (no CUDA on server) |
| `test-mlx-tp-slice` | **pass** |
| `test-mode-word`, `test-draft-win`, `test-k1-mt`, `test-k1-cover`, `test-k1-merge` | **blocked** (qwen27 module needs ocloc/ngen embed) |

## Lean checker counts

| Rev | Files | Problems |
|-----|-------|----------|
| HEAD `0a08fd1b8` | 6249 | 1496 |
| tag `v1.2.7` (`93a2e93ec`) | 6211 | 386 |

## popeye-cuda

Merged at `577b16405` (no conflicts). CUDA binary build not verified on this host.
