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

## Item 15 C1 (2026-11-02, lane B70.p22-k1tree, CL took over from the pi agent)

Source review (CPU only). Launch argument counts against the kernel signatures, all match:
`attn_k1_decode` 15 args (launch 15), `attn_k1_draft` 11 (launch 11), `attn_merge_span` 8 (launch 8).
Every tree-path buffer is sized for its use: `pm`/`pl`/`po` are sized for `limits.splits(MAX_CTX)` slices,
the tree table `a.tree` 16 x 17 ints, `grp` 16 x 2 ints, `kmt` 16 x 8 u64, `kpos`/`kposm` 16 ints. The tree
table (`qwen27.zig` verifyTree) keeps depth in `row[0] & 255`, ancestors in `row[1 ..]`, and ORs the 256/512
flags after the table is built.

Defect found and fixed (commit `885e35d35`): the partials of a tree verify had stride `NS` per (row, head), but
`attn_k1_draft` wrote its slice `NS` at `(row, head) * NS + NS`, which is slot 0 of the next head. The draft
overwrote the next head's prefix slice 0, and the merge read it. Wrong output for every tree row, not a
device fault. Fix: the prefix launch and `attn_merge_span` take `NS + 1` as the stride (`forward.zig`,
`k1_bench.zig` tree branch); the draft keeps `NS` as its slice index and writes at stride `NS + 1` (`k1.cl`).
The K1 tree branch also needs an int8 pool now (`q.kv_i8`), since the draft reads int8 only.

Not established: the cause of `DEVICE_LOST` (0x70000001) in the TP 1 runs. I traced every read and write of the
tree path in source and found no out-of-bounds index, so the stride fix may not remove it. The next split, if
it returns: the test runs the K1 tree round first, so a run that skips it (scalar tree only) separates the
K1 tree kernels from the rest of the verify.

Build (inference1, `-Dtp=1`, from a clean export of `885e35d35`, not from the dirty worktree there):

```
~/b70/port/zig/zig build xpu-native -Dtp=1 -Doptimize=ReleaseFast -Dngen-include=$HOME/b70/stack/oneDNN/third_party/ngen -p ~/b70/qwen27/zig-out-p22-k1tree-885e35d-tp1
```

Result: exit 0, empty log. `~/b70/qwen27/zig-out-p22-k1tree-885e35d-tp1/xpu/bin/tensorfold-native`,
sha256 `8adbd1cb7cb758e5181bf8c2ef178b15267752323eb3d88f383df9b91ec46aed`. The tree test is not in
`xpu-native`: build it with `xpu-qwen27-tests` (it does not open a card) before the window run.

Remote worktree: `~/b70/integrate/TensorFold/.worktrees/p22-k1tree` (inference1) has uncommitted edits from the
pi agent at base `c5acd064` (`HANDOFF.md`, `tree_test.zig`, `zig/build/xpu.zig`, `dev.zig`, `k1.zig`,
`qwen27.zig`, `k1_bench.zig`). `git push` to it is refused for that reason. The lane commit `885e35d35`
contains all of it except the older variants of the same changes, so the lane branch is the source. Nobody
has reset or stashed the remote worktree.

Windowed GPU test for CL (TP 1, card 0). CL builds the test binary first, at `-Dtp=1`, from the export:

```
cd ~/b70/qwen27/src-p22-k1tree-885e35d
~/b70/port/zig/zig build xpu-qwen27-tests -Dtp=1 -Doptimize=ReleaseFast -Dngen-include=$HOME/b70/stack/oneDNN/third_party/ngen -p ~/b70/qwen27/zig-out-p22-k1tree-885e35d-tp1
TREE=$(find ~/b70/qwen27/zig-out-p22-k1tree-885e35d-tp1 -name tf-xpu-qwen27-tree-test -type f | head -1) && echo "$TREE"
timeout 600 env TF_TP_SHM=/dev/shm/tf-tp-k1tree-22 XPU_KV_POOL=1 "$TREE" ~/models/qwen38-27b/Qwen3.8-27B-Uncensored-MLX-4bit
```

Expect `rows 16: K1 tree and scalar tree agree`, `rows 8: ...`, then `ok`. The test sets `TF_TP_SHM` itself
(a lane-own name; the group does not open at TP 1). The `find` prints the binary path before the run: check it
is under `zig-out-p22-k1tree-885e35d-tp1` and not the older `zig-out-p22-k1tree-tp1`.

## Superseded: windowed GPU test of the 10:23Z build (stale, pre-fix)

```
cd ~/b70/qwen27/zig-out-p22-k1tree-tp1
timeout 600 env TF_TP_SHM=/dev/shm/tf-tp-k1tree-22 XPU_KV_POOL=1 ./bin/tf-xpu-qwen27-tree-test ~/models/qwen38-27b/Qwen3.8-27B-Uncensored-MLX-4bit
```

Expect `rows 16: K1 tree and scalar tree agree`, `rows 8: ...`, then `ok`. The binary and the worktree
`~/b70/integrate/TensorFold/.worktrees/p22-k1tree` are current (last sync 10:23:46Z, no file changed since).
`TF_TP_SHM` is set to a lane-own name; at TP 1 the group does not open at all, so the name is belt and
braces. If `did not drain` returns in a clean window, run the same command from a base build first to
separate the lane from the host: `git stash -q`, rebuild `-Dtp=1`, run, `git stash pop -q`.
