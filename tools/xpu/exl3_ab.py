"""Interleaved A/B of exl3.cl variants: each round runs every variant on every case in turn; median over rounds.
Usage: exl3_ab.py <sage dir> <rounds> name=repo:NS [name=repo:NS ...]   (under flock ~/b70/gpu.lock)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref, exl3_check as ck

sage, rounds = sys.argv[1], int(sys.argv[2])
ROWS = tuple(int(r) for r in os.environ.get("AB_ROWS", "1,8").split(","))
variants = []
for spec in sys.argv[3:]:
    name, rest = spec.split("=")
    repo, ns = rest.split(":")
    e = ck.Exl3(repo); e.ns_fixed = int(ns)
    variants.append((name, e))
st = ref.SafeTensors(sage)
rng = np.random.default_rng(6)
cases = []
for proj in ("gate_proj", "down_proj"):
    cases.append((f"experts {proj} 8x40", [st.group(f"model.language_model.layers.{l}.mlp.experts.{x}.{proj}") for l in range(40) for x in range(8)]))
cases.append(("lm_head", [st.group("lm_head")]))
res = {}
for rnd in range(rounds):
    for label, groups in cases:
        for rows in ROWS:
            xs = [ck.to_bf16(rng.standard_normal((rows, g[0].shape[0] * 16)).astype(np.float32)) for g in groups]
            for name, e in variants:
                _, t = ck.project(e, groups, xs, rows, reps=20)
                res.setdefault((label, rows, name), []).append(t[1])
for label, groups in cases:
    tb = sum(g[0].nbytes for g in groups)
    for rows in ROWS:
        line = f"{label:24s} rows={rows:2d}"
        for name, _ in variants:
            ts = res[(label, rows, name)]
            line += f" | {name} {tb/np.median(ts)/1e9:6.1f} GB/s (range {tb/max(ts)/1e9:.0f}-{tb/min(ts)/1e9:.0f})"
        print(line)
