#!/bin/bash
# build_kit.sh [--dry-run] OUT: the dsv41 family's kernel kit (the folder TF_DS_KIT and tf-dsv41-lanes read) built from
# source, optionally with the Triton variants a wider round needs compiled in (step 6, below).
#
# Inputs, as environment variables:
#   IMAGE       the deepseek-v41-tp2 engine's container image (github.com/bertholomus/TensorFold, branch
#               deepseek-v41-tp2, installed in nvcr.io/nvidia/pytorch:26.07-py3), run on a GB10 (aarch64, sm_121)
#   EXT_DIR     that engine's torch-extension cache after its first start (TORCH_EXTENSIONS_DIR, holding
#               tensorfold_exl3_linear_v7/ and tensorfold_exl3_experts_v19/); tools/dsv41-zig/build_inputs.sh makes it
#   MODEL       the EXL3 checkpoint (Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
#   ORIGINAL    the original checkpoint (deepseek-ai/DeepSeek-V4.1-Flash), for the image routing bias
#   TOKEN_MAP   the Engram token map the engine wrote on its first start (build_inputs.sh)
#   RECORDINGS  space-separated recording dirs (each with rank0/ and rank1/: launches.json, triton/), made by running
#               the engine with tools/dsv41-zig/rec on PYTHONPATH and TF_ZREC_DIR=<dir>/rank<R> (see zrec.py); every
#               request shape you serve should be in one of them (a launch with no captured variant runs the
#               closest more general one, cuda/aot.zig)
#   MANIFEST    optional: a kit MANIFEST (sha256sum format) to check the result against
# Optional, the extra variants (step 6; docs/port/kit.md). Unset, the kit is byte for byte what steps 1-5 make:
#   EXTRA_ROWS  a round-row bound (17 .. 48, e.g. 32): every Triton variant rounds of 1..EXTRA_ROWS rows need that the
#               kit lacks (kit/kit_variants.py plan: today _rowmm_parts RB 17..EXTRA_ROWS), compiled from the image's own
#               kernel source with kit/aot_extra.py (Triton's compiler and ptxas, no GPU) and added to aot/
#   EXTRA_HEADS64=1  also the 64-head attention variants of a single-launch owner decode (0118; not the planned path)
#   EXTRA_GPU=1 pass --gpus all to step 6 (only if this image's Triton will not compile without a driver)
#   BASE_KIT    a kit folder to start from instead of steps 1-5 (e.g. the published kit/ at the recipe's KIT_REF):
#               copied to OUT, then step 6; needs only IMAGE and EXTRA_ROWS. Its MANIFEST is updated for the added
#               files (the original kept as MANIFEST.base), so tools/verify_kit.sh checks the result.
# --dry-run: prints the inputs, each step's command and (given a kit to plan from: BASE_KIT, or OUT holding a kit) the
# variants step 6 would compile; runs no container and writes nothing.
# Steps (no GPU except the recordings): 1 the EXL3 extension cubins and 2 PyTorch's attention cubin, extracted with
# cuobjdump; 3 Engram's constants and the RoPE tables (zrec_fixtures.py, CPU); 4 the image routing bias
# (make_bias_vl.py); 5 the Triton kernel set packed from the recordings (triton_aot_manifest.py + aot_pack.py).
# Byte-identical to a published kit: steps 2-4 always; step 1's linear cubins and step 5 given the same .so files and
# recordings. An extension rebuilt from source is not byte-identical (nvcc, build to build): its experts cubins
# then differ from the published ones in a few bytes (18 and 1,246 in our rebuild); the published kit stays the record.
set -eu
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; shift; fi
[ $# = 1 ] || { echo "usage: $0 [--dry-run] OUT" >&2; exit 2; }
T=$(cd "$(dirname "$0")" && pwd)            # tools/dsv41-zig
ROOT=$(cd "$T/../.." && pwd)
if [ $DRY = 1 ]; then
  OUT=$1
  echo "dry run: nothing is built, no container runs"
  echo "OUT=$OUT IMAGE=${IMAGE:-<unset>} BASE_KIT=${BASE_KIT:-<unset>} EXTRA_ROWS=${EXTRA_ROWS:-<unset>} EXTRA_HEADS64=${EXTRA_HEADS64:-0}"
  if [ -z "${BASE_KIT:-}" ]; then
    echo "steps 1-5 (docker run ... \$IMAGE): EXL3 cubins from EXT_DIR=${EXT_DIR:-<unset>}; PyTorch's attention cubin;"
    echo "  Engram constants and RoPE tables from MODEL=${MODEL:-<unset>} TOKEN_MAP=${TOKEN_MAP:-<unset>}; image bias from"
    echo "  ORIGINAL=${ORIGINAL:-<unset>}; Triton set from RECORDINGS=${RECORDINGS:-<unset>}"
  else
    echo "steps 1-5 skipped: OUT starts as a copy of BASE_KIT"
  fi
  if [ -z "${EXTRA_ROWS:-}" ]; then echo "step 6 off (EXTRA_ROWS unset)"; exit 0; fi
  SRC=${BASE_KIT:-$OUT}
  if [ ! -f "$SRC/aot/aot.json" ]; then
    echo "step 6: no kit at $SRC yet: the plan below is against the engine's recorded launches (kit_variants.py --kit"
    echo "  fixtures: past 16 rows only, synthetic templates); the real step plans against the built kit"
    SRC=fixtures
  fi
  echo "step 6: python3 $T/kit/kit_variants.py plan --kit $SRC --rows $EXTRA_ROWS${EXTRA_HEADS64:+ --heads64}, then in the image:"
  echo "  python3 /tf/tools/dsv41-zig/kit/aot_extra.py /out spec.json   (adds to OUT/aot and OUT/MANIFEST)"
  SPEC=$(mktemp); trap 'rm -f "$SPEC"' EXIT
  h64=(); [ "${EXTRA_HEADS64:-0}" = 1 ] && h64=(--heads64)
  python3 "$T/kit/kit_variants.py" plan --kit "$SRC" --rows "$EXTRA_ROWS" ${h64[@]+"${h64[@]}"} --out "$SPEC" >/dev/null
  python3 - "$SPEC" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
for v in spec["variants"]:
    cs = " ".join(f"{n}={x}" for n, x in sorted(v["consts"].items()) if n in ("RB", "BN", "N", "H", "PDL", "POSTED", "WORLD"))
    print(f"  would compile {v['fn']:<26} {v['why']:<22} {cs} ints {v['ints']}")
print(f"  {len(spec['variants'])} variants")
PY
  exit 0
fi
: "${IMAGE:?}"
if [ -n "${BASE_KIT:-}" ]; then
  # never into the served kit: copying a folder onto itself (or into itself) corrupts it, and step 6 then rewrites
  # its aot.json and MANIFEST in place. Checked on the resolved paths before OUT is created.
  B=$(cd "$BASE_KIT" 2>/dev/null && pwd -P) || { echo "BASE_KIT: no folder $BASE_KIT" >&2; exit 2; }
  O=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1")
  case "$O/" in "$B"/*) echo "OUT ($O) is BASE_KIT or inside it ($B): pick a new folder" >&2; exit 2 ;; esac
  case "$B/" in "$O"/*) echo "BASE_KIT ($B) is inside OUT ($O): pick a new folder" >&2; exit 2 ;; esac
fi
if [ -n "${EXTRA_ROWS:-}" ] && [ -z "${BASE_KIT:-}" ] && [ -z "${MANIFEST:-}" ]; then
  echo "EXTRA_ROWS without BASE_KIT needs MANIFEST (the published kit's), so the kit ships one verify_kit.sh can check" >&2; exit 2
fi
OUT=$(mkdir -p "$1" && cd "$1" && pwd -P)
if [ -n "${EXTRA_ROWS:-}" ] && ! { [ "$EXTRA_ROWS" -ge 1 ] && [ "$EXTRA_ROWS" -le 48 ]; } 2>/dev/null; then
  echo "EXTRA_ROWS: 1 .. 48" >&2; exit 2
fi
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# step 6: the variants a round of up to EXTRA_ROWS rows needs that OUT's Triton set lacks, compiled into OUT/aot
extra() {
  echo "6/6 extra Triton variants for rounds of up to $EXTRA_ROWS rows"
  local h64=(); [ "${EXTRA_HEADS64:-0}" = 1 ] && h64=(--heads64)
  python3 "$T/kit/kit_variants.py" plan --kit "$OUT" --rows "$EXTRA_ROWS" ${h64[@]+"${h64[@]}"} --out "$W/spec.json"
  local gpus=(); [ "${EXTRA_GPU:-0}" = 1 ] && gpus=(--gpus all)
  docker run --rm --network none ${gpus[@]+"${gpus[@]}"} --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$W":/w -v "$OUT":/out \
    -v "$ROOT":/tf:ro -w /w "$IMAGE" python3 /tf/tools/dsv41-zig/kit/aot_extra.py /out /w/spec.json
  # the kit's own checker learns --rows: this verify_kit.sh and kit_variants.py replace the kit's, MANIFEST follows
  mkdir -p "$OUT/tools"; cp "$T/kit/kit_variants.py" "$T/kit/verify_kit.sh" "$OUT/tools/"
  python3 -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import aot_extra
aot_extra.update_manifest(Path(sys.argv[2]), ["tools/kit_variants.py", "tools/verify_kit.sh"])' "$T/kit" "$OUT"
  # strict: every gate of up to EXTRA_ROWS rows as one launch, which is what step 6 is for; a miss fails the build
  local audit rc=0
  audit=$(python3 "$T/kit/kit_variants.py" audit --kit "$OUT" --rows "$EXTRA_ROWS") || rc=$?
  printf '%s\n' "$audit" | grep -E '^MISSING' || true
  printf '%s\n' "$audit" | tail -1
  if [ $rc != 0 ]; then echo "step 6: the kit in $OUT still misses variants for rounds of up to $EXTRA_ROWS rows" >&2; exit 1; fi
}

if [ -n "${BASE_KIT:-}" ]; then
  : "${EXTRA_ROWS:?BASE_KIT is for adding variants: set EXTRA_ROWS}"
  [ -f "$BASE_KIT/aot/aot.json" ] || { echo "BASE_KIT: no aot/aot.json in $BASE_KIT" >&2; exit 2; }
  echo "1-5/6 copying $BASE_KIT"
  (cd "$BASE_KIT" && tar --exclude=.git -cf - .) | (cd "$OUT" && tar -xf -)
  extra
  echo "kit in $OUT"
  exit 0
fi
: "${EXT_DIR:?}" "${MODEL:?}" "${ORIGINAL:?}" "${TOKEN_MAP:?}" "${RECORDINGS:?}"
run() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$W":/w -v "$OUT":/out \
  -v "$EXT_DIR":/ext:ro -v "$MODEL":/model:ro -v "$TOKEN_MAP":/token_map.json:ro -v "$ROOT":/tf:ro -w /w "$IMAGE" "$@"; }
mkdir -p "$OUT/cubins" "$OUT/vision" "$OUT/aot"

echo "1/5 EXL3 extension cubins"
run bash -c 'set -e; mkdir -p l e; cd l; cuobjdump -xelf all /ext/tensorfold_exl3_linear_v7/tensorfold_exl3_linear_v7.so >/dev/null
  cd ../e; cuobjdump -xelf all /ext/tensorfold_exl3_experts_v19/tensorfold_exl3_experts_v19.so >/dev/null; cd ..
  for f in l/*.sm_121.cubin; do
    if cuobjdump -symbols "$f" | grep -q glinear; then cp "$f" /out/cubins/linear_grouped.cubin; else cp "$f" /out/cubins/linear.cubin; fi
  done
  cp e/experts.sm_121.cubin /out/cubins/experts.cubin
  cp e/experts_cb2.sm_121.cubin /out/cubins/experts_cb.cubin'

echo "2/5 PyTorch's attention cubin (sm_120)"
run bash -c 'set -e; mkdir -p t; cd t; cuobjdump -xelf all "$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\", \"libtorch_cuda.so\"))")" >/dev/null
  K=_ZN39fmha_cutlassF_f32_aligned_64x64_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIfN7cutlass4arch4Sm80ELb1ELi64ELi64ELi64ELb1ELb1EE6ParamsE
  f=$(grep -l "$K" *.sm_120.cubin | head -1); [ -n "$f" ]; cp "$f" /out/vision/torch_fmha_sm120.cubin'

echo "3/5 Engram constants and RoPE tables"
run bash -c 'set -e; python3 /tf/tools/dsv41-zig/rec/zrec_fixtures.py /model /token_map.json /w/fx >/dev/null
  cp /w/fx/engram.json /w/fx/rope.json /w/fx/rope-*.f32 /out/; cp /w/fx/jit.json /w/jit.json'

echo "4/5 image routing bias"
python3 "$T/kit/make_bias_vl.py" "$ORIGINAL" "$OUT/vision/gate_bias_vl.safetensors"

echo "5/5 Triton kernel set from the recordings"
args=()
for d in $RECORDINGS; do
  for r in 0 1; do
    python3 "$ROOT/tools/zig/triton_aot_manifest.py" --cache "$d/rank$r/triton" --launches "$d/rank$r/launches.json" \
      --mount "$d/rank$r/triton" --out "$W/manifest-$(basename "$d")-$r.json"
    args+=(--manifest "$W/manifest-$(basename "$d")-$r.json" --cache "$d/rank$r/triton")
  done
done
python3 "$ROOT/zig/tests/cuda/nemotron/aot_pack.py" "${args[@]}" --jit "$W/jit.json" --out "$OUT/aot"

if [ -n "${MANIFEST:-}" ]; then
  echo "checking against $MANIFEST"
  (cd "$OUT" && grep -E '  (aot/|cubins/|engram.json|rope|vision/)' "$MANIFEST" | sha256sum -c --quiet) && echo "kit equals MANIFEST"
fi
if [ -n "${EXTRA_ROWS:-}" ]; then
  [ -n "${MANIFEST:-}" ] && cp "$MANIFEST" "$OUT/MANIFEST"   # step 6 updates it for the files it adds
  extra
fi
echo "kit in $OUT"
