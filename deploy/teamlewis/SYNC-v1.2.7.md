# Sync B70 line to upstream v1.2.7

## State

- Time: 2026-11-12 22:00:00Z
- Head: `bfc8f297e` on `origin/sync/v1.2.7`
- Step: TP1 `xpu-native` on jubiku **stopped** (Tim's workstation offline ~21:50Z, no lease); server TP2 done
- TP1 build resumes in a daytime window under a capacity lease of at most **16 cores** and **24 GB** (no overnight jubiku work).

## Jubiku TP1 (2026-11-12)

- Archive: `git archive --prefix=src-sync-<sha>/ <sha>` → `~/b70/build` (ok through `bfc8f297e` push; last full tree `src-sync-1fe736e2f` on disk before SSH loss).
- Build cmd: `nice -n 19 ~/b70/build/zig-x86_64-linux-0.17.0/zig build xpu-native -Dtp=1 -Doptimize=ReleaseFast -Dcpu=skylake_avx512 -Dngen-include=$HOME/b70/build/oneDNN/third_party/ngen -p ~/b70/build/out-sync-<sha>-tp1`
- `tools capacity lease` on jubiku: broker unavailable (`toolsd.sock`); used `nice -n 19` only.
- Progress: ocloc/ngen kernels built (83/86 steps); link failed on merge residues, fixed in commits `b3f863d3a` … `bfc8f297e` (single `core` module, kv `joinReclaim`, DFlash2 `chain_*`/`absorb`, `prefill_step` `max_rows`).
- **Not re-run** after `bfc8f297e`: SSH/Tailscale to jubiku-linux 100% loss from server ~22:50Z.
- CPU tests (CL list): **not run** (no successful binary; host down).

## Server compile (2026-11-12)

| Target | Result |
|--------|--------|
| TP2 `native` (`0a08fd1b8`) | **pass** `/tmp/sync-v127-build/tp2` |
| TP1 `xpu-native` | jubiku only |
| Popeye CUDA `native` | **N/A** on server |
| `test-mlx-tp-slice` (server) | **pass** |

## Lean checker counts

| Rev | Files | Problems |
|-----|-------|----------|
| HEAD `0a08fd1b8` | 6249 | 1496 |
| tag `v1.2.7` (`93a2e93ec`) | 6211 | 386 |

## popeye-cuda

Merged at `577b16405`. CUDA binary build not verified on server.

## Open

- Re-run TP1 build + named CPU tests on jubiku when host is back: `test-draft-win`, `test-mode-word`, `test-kv-pool`, `test-prompt-cache-longturn`, `test-idle-unload-host`.
- Rollout §4 host gates.
