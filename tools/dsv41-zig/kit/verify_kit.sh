#!/bin/bash
# verify_kit.sh [--rope EXL3_MODEL_DIR TP2_SRC] [--bias ORIGINAL_CHECKPOINT_DIR] [--rows N [--strict]]: check this kit against
# MANIFEST, and (--rows) that its Triton set has a variant for every row-dependent launch of rounds of 1..N rows.
#
# Run from anywhere; it checks the kit folder it lives in (tools/..). Every file MANIFEST lists must be present with
# its sha256. Two kinds of file are generated rather than stored in git, and this script can make them first:
#   --rope EXL3_MODEL_DIR TP2_SRC   the four RoPE tables (rope-*.f32, 136 MB each), recomputed on the CPU by
#                                   tools/make_rope.py with the deepseek-v41-tp2 Python family (TP2_SRC = its src/
#                                   folder) and torch, e.g. inside nvcr.io/nvidia/pytorch:26.07-py3;
#   --bias ORIGINAL_CHECKPOINT_DIR  vision/gate_bias_vl.safetensors, copied by tools/make_bias_vl.py from your own
#                                   copy of deepseek-ai/DeepSeek-V4.1-Flash (standard library only).
# The Hugging Face mirror of this kit ships both, so a download needs neither option.
#   --rows N                        also audit aot/aot.json for rounds of up to N rows (TF_DS_ROUND_ROWS) with
#                                   kit_variants.py (beside this script; standard library only), as cuda/aot.zig would
#                                   resolve each launch; a launch with no variant is a MissingTritonVariant at serve time.
#                                   As the engine serves it: a MoE gate past 16 rows with no single-launch variant
#                                   counts as covered ('split') when its 16-row pieces and leftover all resolve
#                                   (tri_norm.rowmmGate runs it that way).
#   --strict                        with --rows: every gate as one launch (the kit build_kit.sh EXTRA_ROWS makes).
# Exit 0: every file matches (and every audited launch has a variant). Exit 1: a file is missing or differs (each one
# is printed), or a launch has no variant.
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$KIT"
ROWS=
SPLIT=--engine-split
while [ $# -gt 0 ]; do
  case "$1" in
    --rows) ROWS=$2; shift 2 ;;
    --strict) SPLIT=; shift ;;
    --rope) PYTHONPATH="$3${PYTHONPATH:+:$PYTHONPATH}" python3 tools/make_rope.py "$2" "$KIT" || exit 1; shift 3 ;;
    --bias) python3 tools/make_bias_vl.py "$2" "$KIT/vision/gate_bias_vl.safetensors" || exit 1; shift 2 ;;
    *) echo "usage: $0 [--rope EXL3_MODEL_DIR TP2_SRC] [--bias ORIGINAL_CHECKPOINT_DIR] [--rows N [--strict]]"; exit 2 ;;
  esac
done
bad=0; n=0
while read -r sum path; do
  n=$((n + 1))
  if [ ! -f "$path" ]; then echo "MISSING  $path"; bad=$((bad + 1)); continue; fi
  got=$(sha256sum "$path" | cut -d' ' -f1)
  [ "$got" = "$sum" ] || { echo "DIFFERS  $path"; bad=$((bad + 1)); }
done < MANIFEST
if [ -n "$ROWS" ]; then
  audit=$(python3 "$HERE/kit_variants.py" audit --kit "$KIT" --rows "$ROWS" $SPLIT); rc=$?
  printf '%s\n' "$audit" | grep -E '^(MISSING|fallback|split)|launches'
  if [ $rc != 0 ]; then echo "kit NOT OK: rounds of up to $ROWS rows would miss a Triton variant"; [ $bad = 0 ] && bad=-1; fi
fi
how=
if [ -n "$ROWS" ]; then
  how="; every launch of rounds up to $ROWS rows has a variant"
  [ -n "$SPLIT" ] && how="$how (or is served by the engine 16-row split)"
fi
if [ $bad = 0 ]; then echo "kit OK: $n files match MANIFEST$how"; exit 0; fi
[ $bad -lt 0 ] || echo "kit NOT OK: $bad of $n files missing or different"; exit 1
