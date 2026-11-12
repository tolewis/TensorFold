# inference1

Two Intel Arc Pro B70 cards. TensorFold XPU TP 2. Qwen3.8-27B MLX 4-bit.

## Build

From `lane/fleet-unified` on inference1:

```
zig build native -Dtp=2 -Dcpu=skylake -Dversion=1.0.0
```

Name the output like `tensorfold-native-tp2-<sha>-W<tag>-k1c3-rf2-<label>`.

## Serve

Rank 1 unit `qwen27-tp2-rank1.service` starts first.
Rank 0 unit `qwen27-tensorfold@b.service` runs `qwen27-serve.sh`.
HAProxy unit `sage-proxy.service` listens on 0.0.0.0:18095 and sends to 127.0.0.1:18196.

Card env files set `ZE_AFFINITY_MASK` and `SAGE_PORT`.
Model env files `qwen27-b.env` and `qwen27-tp2.env` must match on all XPU flags.

## Swap gate

Do not restart production before 09:00Z while the kernel lane holds the cards.
After 09:00Z use `window-p4-ab.sh` with a new rollback script.
Check `pgrep -af window-p4` is empty before a window.
