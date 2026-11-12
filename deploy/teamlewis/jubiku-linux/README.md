# jubiku-linux

One Intel Arc Pro B70. TensorFold XPU TP 1. Qwen3.8-27B MLX 4-bit.

## Build

From `lane/fleet-unified` on a Xeon 654 host or cross compile:

```
zig build native -Dtp=1 -Dcpu=skylake_avx512 -Dversion=1.0.0
```

On jubiku use `nice -n 19` before the compile. Install the binary as `tensorfold-native-<sha>-fleet-unified-tp1-skx`.

## Serve

User unit `qwen27-jubiku.service` runs `qwen27-serve.sh` with `qwen27.env`.
Tailscale publishes `127.0.0.1:18197` on tailnet port 18097.

## Why this host differs

- `-Dcpu=skylake_avx512` is required. The inference1 binary uses Xeon 654 ISA and hits SIGILL here.
- `NEOReadDebugKeys` and `EnableRecoverablePageFaults=0` stop desktop GPU resets.
- `QWEN_IDLE_UNLOAD=300` returns VRAM to Hyprland on an idle desktop.
- One stream only. The card also drives the display.
