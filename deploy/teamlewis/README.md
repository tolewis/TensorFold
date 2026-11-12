# TeamLewis TensorFold fleet configs

This tree holds one copy of each host serve layout for fork branch `lane/fleet-unified`.

## Host table

| Host | Card | Model | Format | Fork branch | Key flags | Port |
|------|------|-------|--------|-------------|-----------|------|
| inference1 | 2x Arc Pro B70 | Qwen3.8-27B Uncensored | MLX 4-bit | `lane/p4-ship7` `cbd2babae` live binary tag `p4ship7` | TP 2, `XPU_GDN=103`, `TF_REPLY_FLOOR=8192`, phase 4 XPU set | 18095 (proxy) |
| jubiku-linux | 1x Arc Pro B70 | Qwen3.8-27B Uncensored | MLX 4-bit | `lane/fleet-serve` `690e93f6` live; target `lane/jubiku-p4` | TP 1, `XPU_K1_TREE=1`, driver NEO keys, idle unload 300 s | 18097 (tailscale) |
| popeye | RTX 5090 | Qwen3.8-27B Uncensored | NVFP4 | upstream `93a2e93` live; target `lane/fleet-unified` | CUDA fp8 KV, context 102400, DFlash2 | 18081 |
| ascent01+02 | 2x GB10 | DS 4.1 Flash | EXL3 2.9 bpw | Mia kit; engine `fbbc971` | TP 2, pool 1M, parallel 4 | 8888 |

## Unified values

- DFlash2 drafter path pattern under each host `models/` tree.
- `XPU_CHUNK=1024` on all B70 Qwen lanes that run phase 4 flags.
- `TF_REPLY_FLOOR` on B70 hosts clamps `max_tokens` to free context (8192 on inference1, 1 on jubiku).
- Thinking on with low effort on Qwen lanes where the serve script passes it.

## Per host differences

| Topic | inference1 | jubiku-linux |
|-------|------------|--------------|
| TP | 2 ranks and HAProxy | 1 rank |
| CPU target | Xeon (`skylake`) | `skylake_avx512` |
| Parallel | 6 streams shared pool | 1 stream |
| Desktop | headless server | NEO debug keys required |

## Catalog lanes

| Lane id | Host |
|---------|------|
| `b70-qwen27` | inference1 |
| `jubiku-b70` | jubiku-linux |
| `popeye-vllm` | popeye (catalog still names vLLM; live is TensorFold CUDA) |
| `ds41` | ascent01 |

## Build branch

`lane/fleet-unified` merges `lane/p4-ship7`, `lane/fleet-serve`, and `lane/jubiku-p4`.
Popeye CUDA and ascent DS kit pins are documented per host until their swaps run.
