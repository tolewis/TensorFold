# jubiku-linux

One Intel Arc Pro B70. TensorFold XPU TP 1. Qwen3.8-27B MLX 4-bit.

## Build

From `lane/fleet-unified` on jubiku itself (Xeon 654 host or cross):

```
zig build xpu-native -Dtp=1 -Doptimize=ReleaseFast -Dcpu=skylake_avx512 \
  -Dngen-include=$HOME/b70/build/oneDNN/third_party/ngen -p <out>
```

On jubiku use `nice -n 19` before the compile. Install the binary as `tensorfold-native-<sha>-fleet-tp1-skx`.

Live: `tensorfold-native-ab047651-jubiku-p4-tp1-skx` (git `ab047651`, lane/jubiku-p4).
The 2026-11-12 unify window built `c664084e3` as `tensorfold-native-c664084e-fleet-tp1-skx`
(tag `prod/jubiku/2026-11-12`); it ran neutral to -2.5% against the live set, so the p4 binary stays.

## Serve

User unit `qwen27-jubiku.service` runs `qwen27-serve.sh` with `qwen27.env`.
Tailscale publishes `127.0.0.1:18197` on tailnet port 18097.

## Why this host differs

- `-Dcpu=skylake_avx512` is required. The inference1 binary uses Xeon 654 ISA and hits SIGILL here.
- `NEOReadDebugKeys` and `EnableRecoverablePageFaults=0` stop desktop GPU resets.
- `QWEN_IDLE_UNLOAD=300` returns VRAM to Hyprland on an idle desktop.
- One stream only. The card also drives the display.
