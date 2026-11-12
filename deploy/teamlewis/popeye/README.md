# popeye

NVIDIA RTX 5090. Qwen3.8-27B NVFP4 on TensorFold CUDA.

## Live state (2026-11-12T03:50Z)

Another agent moved inference off vLLM.
Process `~/tf/bin/tensorfold-native` serves port 18081.
Health reports backend cuda and version 1.2.7.
Build tree `~/tf/src` is upstream `ashhart/TensorFold` at `93a2e93` (Release 1.2.7).
`qwen36-vllm.service` is inactive.

## Target

Binary from `tolewis/TensorFold` branch `lane/fleet-unified` after CUDA merge work lands.
Swap only in the 07:00Z to 11:45Z window with 0 running requests.

## Build

From `lane/fleet-unified` on popeye or a CUDA builder:

```
zig build native -Dversion=1.0.0
```

Copy `zig-out/native/bin/tensorfold-native` to `~/tf/bin/`.

## Flags

fp8 K/V cache, context 102400, parallel 1, prompt cache 8 GiB.
Thinking on with reasoning effort low. DFlash2 drafter.

## Rollback

Re enable `qwen36-vllm.service` and stop the TensorFold process.
Override still points at `launch_qwen38_v0271.sh`.
