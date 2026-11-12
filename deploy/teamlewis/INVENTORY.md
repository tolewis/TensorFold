# Inventory snapshot (before swaps)

Captured 2026-11-12T03:48Z UTC on server.

## inference1

- Binary: `~/b70/qwen27/bin/tensorfold-native-tp2-c5acd064-W5-k1c3-rf2-p4ship7`
- Fork: `lane/p4-ship7` = `cbd2babae`, tag `b70-qwen27-prod-2026-11-11f`
- Binary content sha (nightshift): `86079afa13a0`
- Model: `~/models/qwen38-27b/Qwen3.8-27B-Uncensored-MLX-4bit`
- Drafter: `Qwen3.8-27B-DFlash2`
- Units: `qwen27-tensorfold@b`, `qwen27-tp2-rank1` (user); haproxy on 18095
- Rank 0 port 18196. Context 131072. Parallel 6.
- systemd reported inactive labels while processes were up (manual or linger gap).
- Catalog lane: `b70-qwen27`, served id `Qwen3.8-27B-Ablit-Infa1`

## jubiku-linux

- Binary: `tensorfold-native-690e93f6-fleetserve-tp1-skx`
- Fork: `lane/fleet-serve` `690e93f61`
- Built candidate: `tensorfold-native-ab047651-jubiku-p4-tp1-skx` in bin dir
- Model: same MLX 4-bit tree as inference1
- Unit: `qwen27-jubiku.service` (user), tailscale 18097
- Context 217088, pool 245760 tokens, parallel 1
- Catalog lane: `jubiku-b70`, endpoint drift at probe time (connection reset)
- Lane `jubiku-p4` working; do not restart before it idles

## popeye

- Binary: `~/tf/bin/tensorfold-native` built 2026-11-11
- Source tree: `~/tf/src` at `93a2e93` upstream ashhart, not TeamLewis fork
- vLLM unit inactive. Manual nohup serve on 18081
- Model NVFP4 under `/home/katya/models/qwen36/`
- Flags: fp8 kv, context 102400, parallel 1, prompt-cache 8 GiB
- Health ok at 03:50Z with backend cuda
- Catalog still lists `popeye-vllm` (drift: endpoint refused at catalog probe)

## ascent01

- Container `dsv41-1017-exl3-tf` up 4 days
- Kit `~/dsv41-recipe` at `31457d0`
- Image `tensorfold-dsv41:exl3-tp2-03d5b88f05ef`
- Engine default `TF_REF=fbbc971` on bertholomus fork
- Port 8888, served name `DS-4.1-Flash`
- Catalog lane `ds41`, health ok
