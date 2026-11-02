"""The prompt GEMM of exl3_gemm2 (XPU_PROMPT_GEMM2=1) against exl3_gemm on the Qwen3.8-27B dense shapes at TP 2.
Runs tf-xpu-exl3-test in mode 3 (prompt GEMM A/B): one case, both kernels timed, the max abs and relative
difference of the flag output against the live kernel's, and the flag output written. Cases come from the
checkpoint (real trellis words). One GPU job at a time.
Usage: exl3_27b_gemm2_bench.py <EXL3 checkpoint> <tf-xpu-exl3-test> <work dir> [reps]
Env: ROWS (896,1152), CASES (comma list), TAG (name in the printed lines).
"""
import json, os, struct, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3_ref as ref

MAGIC = 0x334C5845
MODE = 3  # prompt GEMM A/B: exl3_gemm against exl3_gemm2
L = "model.language_model.layers"


def to_bf16(a):
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def write_case(path, mats, rows, x):
    tr0 = mats[0][0]
    k, n, halves = tr0.shape[0] * 16, tr0.shape[1] * 16, tr0.shape[2] // 8
    parts = [struct.pack("<10I", MAGIC, MODE, len(mats), k, n, halves, rows, 0, 0, 1)]
    for tr, suh, svh, _ in mats:
        parts += [np.ascontiguousarray(a).tobytes() for a in (tr, suh, svh)]
    parts.append(np.ascontiguousarray(x).tobytes())
    with open(path, "wb") as f:
        f.write(b"".join(parts))
    return k, n


FREQ = os.environ.get("ACT_FREQ", "/sys/devices/pci0000:61/0000:61:02.0/0000:62:00.0/0000:63:01.0/0000:64:00.0/tile0/gt0/freq0/act_freq")


def act_freq():
    try:
        return int(open(FREQ).read())
    except OSError:
        return 0


def run(prog, case, out, reps):
    r = subprocess.run([prog, case, out, str(reps)], capture_output=True, text=True)
    return r


def parse(stderr):
    """The two lines of mode 3: the path check and the A/B result."""
    path, ab = None, None
    for line in stderr.splitlines():
        if line.startswith("gemm2 path "):
            path = line.strip()
        if line.startswith("gemm2 ab:"):
            f = line.replace(",", " ").split()
            # gemm2 ab: base <us> us gemm2 <us> us speedup <x>x max abs <e> rel <e> <d> of <d> values differ
            ab = {"base_us": float(f[2]), "gemm2_us": float(f[5]), "speedup": float(f[7].rstrip("x")),
                  "max_abs": float(f[10]), "rel": float(f[12]), "diff": int(f[13]), "values": int(f[16])}
    return path, ab


def main():
    ckpt, prog, work = sys.argv[1:4]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    os.makedirs(work, exist_ok=True)
    st = ref.SafeTensors(ckpt)
    rows_list = [int(r) for r in os.environ.get("ROWS", "896,1152").split(",")]
    g = st.group
    cases = {
        "gu": [g(f"{L}.0.mlp.gate_proj"), g(f"{L}.0.mlp.up_proj")],
        "down": [g(f"{L}.0.mlp.down_proj")],
        "qkv": [g(f"{L}.0.linear_attn.in_proj_qkv")],
        "z": [g(f"{L}.0.linear_attn.in_proj_z")],
        "out": [g(f"{L}.0.linear_attn.out_proj")],
        "q": [g(f"{L}.3.self_attn.q_proj")],
        "head": [g("lm_head")],
    }
    want = os.environ.get("CASES")
    if want:
        cases = {c: cases[c] for c in want.split(",")}
    rng = np.random.default_rng(27)
    results = []
    print(f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} reps {reps} prog {prog}", flush=True)
    print(f"{'case':6} {'k':>6} {'n':>6} {'rows':>5} {'base us':>10} {'gemm2 us':>10} {'x':>5} {'max abs':>10} {'rel':>10} {'MHz':>5}  note", flush=True)
    for name, mats in cases.items():
        for rows in rows_list:
            k = mats[0][0].shape[0] * 16
            x = to_bf16(rng.standard_normal((len(mats) * rows, k)).astype(np.float32))
            case = os.path.join(work, f"{name}-r{rows}-gemm2.bin")
            k, n = write_case(case, mats, rows, x)
            out = os.path.join(work, f"{name}-r{rows}-gemm2-y.bin")
            r = run(prog, case, out, reps)
            mhz = act_freq()
            if r.returncode != 0:
                err = (r.stderr.strip().splitlines() or ["?"])[-1][:140]
                print(f"{name:6} {k:6} {n:6} {rows:5} {'fail':>10} {'':>10} {'':>5} {'':>10} {'':>10} {mhz:5}  {err}", flush=True)
                results.append({"case": name, "k": k, "n": n, "rows": rows, "error": err})
                continue
            path, ab = parse(r.stderr)
            note = path or "?"
            if ab is None:
                print(f"{name:6} {k:6} {n:6} {rows:5} {'?':>10} {'':>10} {'':>5} {'':>10} {'':>10} {mhz:5}  {note}", flush=True)
                results.append({"case": name, "k": k, "n": n, "rows": rows, "note": note})
                continue
            print(f"{name:6} {k:6} {n:6} {rows:5} {ab['base_us']:10.1f} {ab['gemm2_us']:10.1f} {ab['speedup']:5.2f} {ab['max_abs']:10.3e} {ab['rel']:10.2e} {mhz:5}  {note}", flush=True)
            results.append({"case": name, "k": k, "n": n, "rows": rows, "note": note, "mhz": mhz, **ab})
    json.dump(results, open(os.path.join(work, "gemm2-results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
