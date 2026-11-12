# Fleet unify status

Updated by tf-unify (Cursor). Tim reads this in the morning.

## Steps

| Step | Time (Z) | SHA / note | Result |
|------|----------|------------|--------|
| 1 Inventory four hosts | 2026-11-12T03:48Z | | done |
| 2 Branch `lane/fleet-unified` | 2026-11-12T03:52Z | `097e87cc6` | merged p4-ship7 + fleet-serve + jubiku-p4 |
| 3 Config tree `deploy/teamlewis/` | 2026-11-12T03:55Z | same | done |
| 4 Per-host builds | 2026-11-12T04:02Z | `c2aa037a` tp2 sha `ee72db142833`, tp1-skx `ec7ab9f77527` | done on inference1 (no card) |
| 5 Production swaps | | | not started (earliest 07:00Z; inference1 09:00Z) |
| 6 inference1 gate | | | blocked until 09:00Z and kernel lane idle |
| 7 Catalog update | | | pending after swaps |
| 8 Push branches | 2026-11-12T03:57Z | `62635d1e5` | done |

## Hosts

| Host | Inventory (Z) | Live binary / engine | Swap |
|------|---------------|----------------------|------|
| inference1 | 03:48Z | `tensorfold-native-tp2-c5acd064-W5-k1c3-rf2-p4ship7`, fork `cbd2babae` | pending |
| jubiku-linux | 03:48Z | `ab047651-jubiku-p4-tp1-skx` | 2026-11-12T21:00Z unify window (gate/unify1): fleet c664084e judge FAIL (100k -0.1%, 150k -2.5%, pooled +0.0%), no switch, tag on c664084e3 |
| popeye | 03:50Z | upstream build 1.2.7 cuda, health 200 | pending (do not fight live agent) |
| ascent01 | 03:48Z | Mia kit `31457d0`, container up | pending |

## Blockers

- Production swap window opens at 07:00Z.
- inference1 cards belong to `p4-kernel` until 09:00Z.
- `jubiku-p4` lane still working at 03:48Z herdr list.
- Popeye binary is not from `tolewis/TensorFold` yet. Fleet-unified needs a CUDA promote path.

## Next after 07:00Z

1. Build `lane/fleet-unified` on each host with `tools capacity lease reserve`.
2. Rollback script per host, 0 running requests, health and chat probe.
3. inference1 only through `window-p4-ab.sh` after 09:00Z.
