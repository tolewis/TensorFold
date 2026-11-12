#!/usr/bin/env bash
# Qwen3.8-27B (MLX 4/64 Uncensored) on jubiku's Arc Pro B70, TensorFold XPU, TP 1.
# QWEN_PARALLEL streams share one int8 K/V pool. Drafts: DFlash2 when QWEN_DRAFTER is set, else the MTP head (XPU_MTP=1, K=3).
# The B70 also drives the desktop: keep VRAM headroom. Recipe: TensorFold docs/xpu/RECIPE-qwen38-27b-b70-workstation.md.
# Env from ~/b70/qwen27/qwen27.env (unit) or the caller.
set -euo pipefail
M=$HOME/models/qwen38-27b
export XPU_KV_INT8=1 XPU_MTP=${XPU_MTP:-1} XPU_MTP_K=${XPU_MTP_K:-3}
export LD_LIBRARY_PATH=$HOME/b70/qwen27/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ARGS=()
[ -n "${QWEN_CACHE_GIB:-}" ] && ARGS+=(--prompt-cache-gib "$QWEN_CACHE_GIB")
[ -n "${QWEN_POOL_TOKENS:-}" ] && ARGS+=(--kv-pool-tokens "$QWEN_POOL_TOKENS")
[ -n "${QWEN_DRAFTER:-}" ] && ARGS+=(--drafter "$QWEN_DRAFTER")
[ -n "${QWEN_IDLE_UNLOAD:-}" ] && ARGS+=(--idle-unload "$QWEN_IDLE_UNLOAD")
[ "${QWEN_CACHE_OVER_CAP:-0}" = 1 ] && ARGS+=(--prompt-cache-over-cap)
[ "${QWEN_LOOP_GUARD:-0}" = 1 ] && ARGS+=(--loop-guard)
[ -n "${QWEN_API_KEY_FILE:-}" ] && ARGS+=(--api-key-file "$QWEN_API_KEY_FILE")
exec "${QWEN_BIN:-$HOME/b70/qwen27/bin/tensorfold-native}" serve \
  "${QWEN_MODEL:-$M/Huihui-Qwen3.8-27B-abliterated-exl3-4.00bpw}" \
  --name "${QWEN_NAME:-Qwen3.8-27B-Ablit-Jubiku}" \
  --parallel "${QWEN_PARALLEL:-1}" --context "${QWEN_CONTEXT:-262144}" "${ARGS[@]}" \
  --allowed-host jubiku-linux.tail81923e.ts.net --allowed-host jubiku-linux --allowed-host 100.95.49.5 \
  --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-18197}" --no-update-check
