"""exl3_matvec on the Qwen3.8-27B dense shapes through tf-xpu-exl3-test (mode 0, groups of one width).
Cases from the checkpoint (real trellis words: constant fills read faster than DRAM on Xe2): gate + up as one
launch of two groups, down, the GDN in_proj_qkv and in_proj_z, out_proj, the attention q_proj, and lm_head (6 bits).
Each case runs at every row count in ROWS, first under the engine's rule (pick and split), then under every
(EXL3_NS, EXL3_KS) in the sweep. One line a run: case, rows, ns, ks, us a call, GB/s.
Usage: exl3_27b_bench.py <EXL3 checkpoint> <tf-xpu-exl3-test> <work dir> [reps]   (one GPU job at a time)
Env: ROWS (1,4,8,16), NS (rule,1,2,4), KS (rule,1,2,4,8), CASES (comma list), SAVE_Y=<tag> keeps y as <case>-r<rows>-<tag>.bin,
     CHECK_Y=<tag> compares each run's y with that saved file (max abs and rel error)."""
import json, os, struct, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3_ref as ref

MAGIC = 0x334C5845
L = "model.language_model.layers"


def to_bf16(a):
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def write_case(path, mats, rows, x):
    tr0 = mats[0][0]
    k, n, halves = tr0.shape[0] * 16, tr0.shape[1] * 16, tr0.shape[2] // 8
    parts = [struct.pack("<10I", MAGIC, 0, len(mats), k, n, halves, rows, 0, 0, 1)]
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


def split_bound(n, ks_max=16, target=4096):
    """exl3.zig splitBound: the sums scratch holds this many K splits for a group of width n."""
    strips = n // 16
    b = 1
    while b < -(-target // strips):
        b *= 2
    return min(ks_max, b)


def run(prog, case, out, reps, env):
    e = dict(os.environ)
    e.pop("EXL3_NS", None)
    e.pop("EXL3_KS", None)
    e.update(env)
    r = subprocess.run([prog, case, out, str(reps)], capture_output=True, text=True, env=e)
    mhz = act_freq()
    if r.returncode != 0:
        return None, (r.stderr.strip().splitlines() or ["?"])[-1][:120], mhz
    line = r.stderr.strip().splitlines()[-1]  # call 123.4 us, weights 89.13 MB, 400.1 GB/s
    f = line.replace(",", "").split()
    return (float(f[1]), float(f[4]), float(f[6])), line, mhz


def main():
    ckpt, prog, work = sys.argv[1:4]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    os.makedirs(work, exist_ok=True)
    st = ref.SafeTensors(ckpt)
    rows_list = [int(r) for r in os.environ.get("ROWS", "1,2,4,8,16").split(",")]
    ns_list = os.environ.get("NS", "rule,1,2,4").split(",")
    ks_list = os.environ.get("KS", "rule,1,2,4,8").split(",")
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
    save, check = os.environ.get("SAVE_Y"), os.environ.get("CHECK_Y")
    results = []
    print(f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} reps {reps} prog {prog}", flush=True)
    print(f"{'case':6} {'k':>6} {'n':>6} {'rows':>4} {'ns':>4} {'ks':>4} {'us':>9} {'GB/s':>7}  note", flush=True)
    for name, mats in cases.items():
        for rows in rows_list:
            k = mats[0][0].shape[0] * 16
            x = to_bf16(rng.standard_normal((len(mats) * rows, k)).astype(np.float32))
            case = os.path.join(work, f"{name}-r{rows}.bin")
            k, n = write_case(case, mats, rows, x)
            out = os.path.join(work, "y.bin")
            seen = set()
            for ns in ns_list:
                for ks in ks_list:
                    env = {}
                    if ns != "rule":
                        env["EXL3_NS"] = ns
                    if ks != "rule":
                        env["EXL3_KS"] = ks
                    key = (ns, ks)
                    if key in seen:
                        continue
                    seen.add(key)
                    if ks != "rule" and int(ks) > split_bound(n):
                        continue  # the sums scratch holds splitBound(n) splits: more fails at the launch
                    res, line, mhz = run(prog, case, out, reps, env)
                    note = f"{mhz} MHz"
                    if res is None:
                        print(f"{name:6} {k:6} {n:6} {rows:4} {ns:>4} {ks:>4} {'fail':>9} {'':>7}  {line}", flush=True)
                        results.append({"case": name, "k": k, "n": n, "rows": rows, "ns": ns, "ks": ks, "error": line})
                        continue
                    y = np.fromfile(out, np.float32)
                    if save and ns == "rule" and ks == "rule":
                        y.tofile(os.path.join(work, f"{name}-r{rows}-{save}.bin"))
                    if check:
                        ref_y = np.fromfile(os.path.join(work, f"{name}-r{rows}-{check}.bin"), np.float32)
                        err = np.abs(y - ref_y)
                        note += f" max abs {err.max():.3e} rel {err.max() / (np.abs(ref_y).max() + 1e-30):.2e}"
                    us, mb, gbs = res
                    print(f"{name:6} {k:6} {n:6} {rows:4} {ns:>4} {ks:>4} {us:9.1f} {gbs:7.1f}  {note}", flush=True)
                    results.append({"case": name, "k": k, "n": n, "rows": rows, "ns": ns, "ks": ks, "us": us, "mb": mb, "gbs": gbs, "note": note})
    json.dump(results, open(os.path.join(work, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
