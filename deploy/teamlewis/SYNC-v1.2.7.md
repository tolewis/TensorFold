# Sync B70 line to upstream v1.2.7

## State

- Time: 2026-11-12 19:53:05Z
- Head: `73efe744175514172091fad5b6913344a6c3ccea` (start of `origin/lane/fleet-unified`)
- Step: 2 (dual-changed list written, merge not started)

## Done

1. Fetched `origin` and `upstream --tags`.
2. Checked out branch `sync/v1.2.7` from `origin/lane/fleet-unified`. Start sha `73efe744175514172091fad5b6913344a6c3ccea`.
3. Listed 51 dual-changed files (merge-base `84e0d560303d85c50becb730720c04a56e38abe5`).

## Open

1. Run `git merge v1.2.7` and resolve conflicts in order: build, core, core/lanes, server, native, xpu, kernels.

## Needs the kernel lane

(none yet)

## Compile and test results

(not run yet)

## Lean checker counts

(not run yet)

## Dual-changed files (pre-merge)

| File | State |
|------|-------|
| build.zig | pending |
| docs/api.md | pending |
| zig/build/cuda.zig | pending |
| zig/build/lane_halt_host.zig | pending |
| zig/build/xpu.zig | pending |
| zig/kernels/xpu/qwen27/decode.cl | pending |
| zig/kernels/xpu/qwen27/gemm.cl | pending |
| zig/kernels/xpu/qwen27/glue_attn.cl | pending |
| zig/kernels/xpu/qwen27/glue.cl | pending |
| zig/kernels/xpu/qwen27/glue_quant.cl | pending |
| zig/kernels/xpu/qwen27/shape.h | pending |
| zig/src/core/engine_api.zig | pending |
| zig/src/core/engine_status.zig | pending |
| zig/src/core/lane_host_fill.zig | pending |
| zig/src/core/lane_host_status.zig | pending |
| zig/src/core/lane_host.zig | pending |
| zig/src/core/lanes/backend.zig | pending |
| zig/src/core/lanes/config.zig | pending |
| zig/src/core/lanes/engine_test.zig | pending |
| zig/src/core/lanes/fake.zig | pending |
| zig/src/core/lanes/handover.zig | pending |
| zig/src/core/lanes/open.zig | pending |
| zig/src/core/lanes/sampling.zig | pending |
| zig/src/core/lanes/windows.zig | pending |
| zig/src/core/prompt_cache_fake.zig | pending |
| zig/src/core/prompt_cache_policy.zig | pending |
| zig/src/core/prompt_cache.zig | pending |
| zig/src/native/qwen27_lanes.zig | pending |
| zig/src/native/xpu_lanes.zig | pending |
| zig/src/native/xpu_lane.zig | pending |
| zig/src/native/xpu.zig | pending |
| zig/src/server/chat.zig | pending |
| zig/src/server/cli_test.zig | pending |
| zig/src/server/cli.zig | pending |
| zig/src/server/fields.zig | pending |
| zig/src/server/idle_unload_review_test.zig | pending |
| zig/src/server/idle_unload.zig | pending |
| zig/src/server/main.zig | pending |
| zig/src/server/metrics.zig | pending |
| zig/src/server/openai.zig | pending |
| zig/src/server/responses_translate.zig | pending |
| zig/src/server/server.zig | pending |
| zig/src/server/serve.zig | pending |
| zig/src/server/stream_preflight_test.zig | pending |
| zig/src/xpu/dev.zig | pending |
| zig/src/xpu/kernels.zig | pending |
| zig/src/xpu/load.zig | pending |
| zig/src/xpu/module.zig | pending |
| zig/src/xpu/qwen27.zig | pending |
| zig/src/xpu/root.zig | pending |
| zig/src/xpu/ze.zig | pending |

## Conflict resolutions

(fill during merge)
