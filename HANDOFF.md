# Lane p22-k1tree handoff

K1 for the single-stream tree verify. The tree verify runs the prefix keys through the split-K `attn_k1`
path, the draft keys through the new `attn_k1_draft` kernel, and joins them in one `attn_merge_span` call.
The path is off unless `XPU_K1=1` and `XPU_K1_TREE=1`.

## GPU runs on 2026-11-02

Every process I started on inference1 that opened a card. All of them opened Level Zero device index 0
(the first B70; `sycl-ls` at 09:18:30Z shows two B70s). None set `XPU_DEVICE`, `ZE_AFFINITY_MASK`, or any
`TF_TP_*` variable, so every run used the default card and, for the TP 2 runs, the default group file.
Start times are UTC from the session log. End times are the upper bound from the next command in the log.

### Bench runs (card only, no shm)

| Start (UTC) | Command (remote part) | End (UTC) |
|---|---|---|
| 08:06:27 | `./bin/tf-xpu-qwen27-k1-bench` from `~/b70/qwen27/zig-out-p22-k1tree` | before 08:10:35 |
| 08:10:35 | `./bin/tf-xpu-qwen27-k1-bench` | before 08:18:09 |
| 08:18:09 | `./bin/tf-xpu-qwen27-k1-bench > /tmp/k1bench.log` | before 08:19:06 |
| 08:20:45 | `gdb -batch -ex run -ex bt ./bin/tf-xpu-qwen27-k1-bench` | before 08:24:28 |
| 08:24:28 | build, then `./bin/tf-xpu-qwen27-k1-bench` | before 08:25:56 |
| 08:25:56 | `~/b70/qwen27/zig-out-p22-k1tree-base/bin/tf-xpu-qwen27-k1-bench` (base build) | before 08:29:15 |
| 08:29:15 | build Debug, then `~/b70/qwen27/zig-out-p22-k1tree-db/bin/tf-xpu-qwen27-k1-bench` | before 08:31:30 |
| 08:31:30 | build base Debug, then `~/b70/qwen27/zig-out-p22-k1tree-dbbase/bin/tf-xpu-qwen27-k1-bench` | before 08:33:44 |
| 08:38:08 | build, then `./bin/tf-xpu-qwen27-k1-bench` (all 10 cases pass, tree 19.9x / 31.8x) | before 08:39:29 |

### TP 2 tree-test runs (card + /dev/shm/tf-tp-qwen27)

Binary `tf-xpu-qwen27-tree-test` from `~/b70/qwen27/zig-out-p22-k1tree` (built `-Dtp=2`), arg
`~/models/qwen38-27b/Qwen3.8-27B-Uncensored-MLX-4bit`. Rank 0 only. Each open created and truncated
`/dev/shm/tf-tp-qwen27` (the file production and the gate pair use). Each run loaded the 64 layers, then
exited on the first all-reduce with `PeerLost`.

| Start (UTC) | Command | End (UTC) |
|---|---|---|
| 09:36:12 | `time ./bin/tf-xpu-qwen27-tree-test <model>` (background job) | job record gone at the 09:49:28 poll |
| 09:50:38 | same (background job) | job record gone at the 10:01:18 poll |
| 10:02:31 | `timeout 600 ./bin/tf-xpu-qwen27-tree-test <model>` | before 10:10:24 (PeerLost) |

### TP 1 tree-test runs (card only, no group at TP 1)

Binary from `~/b70/qwen27/zig-out-p22-k1tree-tp1` (built `-Dtp=1`).

| Start (UTC) | Command | End (UTC) | Result |
|---|---|---|---|
| 10:11:40 | `timeout 600 ./bin/tf-xpu-qwen27-tree-test <model>` | before 10:14:10 | `skipped: no page table (kv pool off)` |
| 10:14:10 | `timeout 600 env XPU_KV_POOL=1 ./bin/tf-xpu-qwen27-tree-test <model>` | before 10:21:12 | `did not drain (Level Zero 0x70000001)` |
| 10:21:12 | build, then the same `XPU_KV_POOL=1` run (card open ~10:23) | before 10:23:46 | `did not drain (Level Zero 0x70000001)` |
| 10:23:46 | build, then the same `XPU_KV_POOL=1` run (card open ~10:25) | before 10:32:01 | `did not drain (Level Zero 0x70000001)` |

Notes:

- `0x70000001` is `ZE_RESULT_ERROR_DEVICE_LOST`: the device hung, reset, or was removed while the run held it.
  Production restarted at 10:31 (ps start time), so the 10:14-10:28Z window overlapped the cutover.
- The card events you logged at 10:12Z and 10:21Z match the 10:11:40 run and the 10:21:12 command.
- `sycl-ls` also opened the card at 09:18:30Z (a driver query, seconds long).
- The `zig build xpu-qwen27-tests` runs did not open a card: their host test binary runs pure-CPU tests and
  a fake Level Zero (the `tp.test.an` failure is pre-existing on base).
- At 10:32:01Z a ps check showed no leftover processes of mine; only the production serve.

## Windowed GPU test (run in a gate window, TP 1, card 0)

```
cd ~/b70/qwen27/zig-out-p22-k1tree-tp1
timeout 600 env TF_TP_SHM=/dev/shm/tf-tp-k1tree-22 XPU_KV_POOL=1 ./bin/tf-xpu-qwen27-tree-test ~/models/qwen38-27b/Qwen3.8-27B-Uncensored-MLX-4bit
```

Expect `rows 16: K1 tree and scalar tree agree`, `rows 8: ...`, then `ok`. The binary and the worktree
`~/b70/integrate/TensorFold/.worktrees/p22-k1tree` are current (last sync 10:23:46Z, no file changed since).
`TF_TP_SHM` is set to a lane-own name; at TP 1 the group does not open at all, so the name is belt and
braces. If `did not drain` returns in a clean window, run the same command from a base build first to
separate the lane from the host: `git stash -q`, rebuild `-Dtp=1`, run, `git stash pop -q`.
