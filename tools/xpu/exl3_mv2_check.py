"""Rows-1 decode through the 2-operand image (exl3_mv2, lane p29-exl3bw) with tf-xpu-exl3-test.

A dense EXL3 checkpoint (Qwen3.8-27B, GDN linear attention, no experts): gate_proj is a rows-1 matvec, the
shape the mv2 image takes over. Each run forces one (NS, KU) image with XPU_EXL3_MV2 and sets XPU_EXL3_MV2_VERIFY,
which makes the host print a marker when the launch took an mv2 image (the fallback is exl3_matvec). The marker
plus the numpy error are the check: the mv2 accumulation order differs from the PyOpenCL host, so the bits do
not match, but both stay within 1e-3 of the numpy reference.

Usage: exl3_mv2_check.py <dense EXL3 checkpoint> <tf-xpu-exl3-test> <work dir>   (one GPU job at a time)
Env: GROUP (default model.language_model.layers.0.mlp.gate_proj), SELS (default 2_1,4_2,8_4)."""
import os, struct, subprocess, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3_ref as ref
import exl3_check as ck

MAGIC = 0x334C5845
GROUP = "model.language_model.layers.0.mlp.gate_proj"
SELS = "2_1,4_2,8_4"


def run_capture(prog, work, mats, rows, x):
    """One mode 0 launch; returns (y fp32, stderr). stderr carries the XPU_EXL3_MV2_VERIFY marker."""
    tr0 = mats[0][0]
    k, n, halves = tr0.shape[0] * 16, tr0.shape[1] * 16, tr0.shape[2] // 8
    parts = [struct.pack("<10I", MAGIC, 0, len(mats), k, n, halves, rows, 0, 0, 1)]
    for tr, suh, svh, mul1 in mats:
        ck.refuse(halves, mul1)
        parts += [np.ascontiguousarray(a).tobytes() for a in (tr, suh, svh)]
    parts.append(np.ascontiguousarray(x).tobytes())
    case, out = os.path.join(work, "case.bin"), os.path.join(work, "y.bin")
    with open(case, "wb") as f:
        f.write(b"".join(parts))
    r = subprocess.run([prog, case, out], check=True, capture_output=True, text=True)
    y = np.fromfile(out, np.float32)
    return y, r.stderr


def main():
    root, prog, work = sys.argv[1:4]
    os.makedirs(work, exist_ok=True)
    st = ref.SafeTensors(root)
    group = os.environ.get("GROUP", GROUP)
    sels = os.environ.get("SELS", SELS).split(",")
    # EXL3_KS=1: a deterministic K split so (k_tiles_gcd / ks) % ku is always 0 and the XPU_EXL3_MV2 image is taken
    # (the marker fires). This is a correctness check of the mv2 image, not of the split rule.
    os.environ["EXL3_KS"] = "1"
    tr, suh, svh, mul1 = st.group(group)
    tr = np.ascontiguousarray(tr)
    kt, nt, w = tr.shape
    halves = w // 8
    W = ref.dequantize(halves, 2, tr, suh, svh)
    rng = np.random.default_rng(9)
    x = ck.to_bf16(rng.standard_normal((1, kt * 16)).astype(np.float32))
    yr = ck.from_bf16(x).astype(np.float64) @ W
    worst = 0.0
    for sel in sels:
        os.environ["XPU_EXL3_MV2"] = sel
        os.environ["XPU_EXL3_MV2_VERIFY"] = "1"
        y, err = run_capture(prog, work, [(tr, suh, svh, mul1)], 1, x)
        used = "exl3_mv2 image" in err
        e = float(np.abs(y.reshape(1, nt * 16) - yr).max() / np.abs(yr).max())
        worst = max(worst, e)
        print(f"mv2 {sel} {group}: mv2 image used: {used}, max|y-ref|/max|ref| = {e:.2e}")
        assert used, f"XPU_EXL3_MV2={sel} set but the launch did not take an mv2 image (stderr: {err[-200:]})"
        assert e <= 1e-3, e
        del os.environ["XPU_EXL3_MV2"], os.environ["XPU_EXL3_MV2_VERIFY"]
    print(f"exl3_mv2 rows 1: {len(sels)} images OK, worst max|y-ref|/max|ref| = {worst:.2e}")


if __name__ == "__main__":
    main()
