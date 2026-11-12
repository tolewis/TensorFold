#!/usr/bin/env bash
# TP 2 rank 1 of the 27B: holds half the weights on its card and replays rank 0's steps through /dev/shm/tf-tp-qwen27.
# Exits when rank 0 closes. The unit restarts it, so a new rank 0 finds its peer within RestartSec.
set -euo pipefail
export XPU_KV_INT8=1 TF_TP_RANK=1
# Rank 0 reads QWEN_PARALLEL from qwen27-b.env (qwen27-serve.sh passes it). At TP 2 the same var on qwen27-tp2.env
# drives rank 1, so --parallel lands on both ranks when the pool runs them (context.sh b pool sets it).
PAR="${QWEN_PARALLEL:+--parallel ${QWEN_PARALLEL}}"
# XPU_DRAFT_TP=1 (phase 4 package K10, the drafter split): rank 1 holds half the drafter, so it loads it too.
DRAFTER=()
[ "${XPU_DRAFT_TP:-0}" = 1 ] && DRAFTER=(--drafter "${QWEN_DRAFTER:-$HOME/models/qwen38-27b/Qwen3.8-27B-DFlash2}")
exec "${QWEN_BIN:?}" serve "${QWEN_MODEL:?}" --tp 2 --rank 1 --context "${QWEN_CONTEXT:-131072}" $PAR "${DRAFTER[@]}"
