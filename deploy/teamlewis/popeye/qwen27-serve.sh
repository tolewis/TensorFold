#!/usr/bin/env bash
# Popeye TensorFold CUDA lane (manual start 2026-11-12 while vLLM unit stays disabled).
# Model: Qwen3.8-27B-Uncensored-NVFP4. Port 18081 replaces vLLM on the same port.
set -euo pipefail
M=/home/katya/models/qwen36
exec /home/katya/tf/bin/tensorfold-native serve \
  "$M/Qwen3.8-27B-Uncensored-NVFP4" \
  --drafter "$M/Qwen3.8-27B-DFlash2" \
  --kv-cache fp8 \
  --context 102400 \
  --parallel 1 \
  --prompt-cache-gib 8 \
  --name latest \
  --alias qwen-3.8-27B --alias qwen36 \
  --thinking --reasoning-effort low \
  --host 0.0.0.0 --port 18081
