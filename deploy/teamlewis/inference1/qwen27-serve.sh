#!/usr/bin/env bash
# Qwen3.8-27B (abliterated, MLX 4/64) on one Arc Pro B70, TensorFold XPU 27B path.
# Env from ~/b70/release/card-<a|b>.env: ZE_AFFINITY_MASK, SAGE_PORT (18195 card a, 18196 card b).
# QWEN_CONTEXT 131072 gives 3 slots a card at int8 K/V (4.67 GB a slot). 262144 gives 1 slot.
set -euo pipefail
M=$HOME/models/qwen38-27b
export XPU_KV_INT8=1
DRAFTER=()
[ "${QWEN_DRAFT:-1}" = 1 ] && DRAFTER=(--drafter "${QWEN_DRAFTER:-$M/Qwen3.8-27B-DFlash2}")
CACHE=()
[ -n "${QWEN_CACHE_GIB:-}" ] && CACHE=(--prompt-cache-gib "$QWEN_CACHE_GIB")
exec "${QWEN_BIN:-$HOME/b70/release/tensorfold-native}" serve "${QWEN_MODEL:-$M/Qwen3.8-27B-Uncensored-MLX-4bit}" \
  --name "${QWEN_NAME:-Qwen3.8-27B-Ablit-Infa1}" \
  --parallel "${QWEN_PARALLEL:-3}" --context "${QWEN_CONTEXT:-131072}" "${DRAFTER[@]}" "${CACHE[@]}" \
  --host "${QWEN_HOST:-127.0.0.1}" --port "${SAGE_PORT:-18195}"
