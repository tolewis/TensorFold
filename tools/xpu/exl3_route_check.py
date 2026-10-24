"""P3.2 gate: routed experts by device-built descriptors against the fp64 host reference, per (token, pick).

Usage: exl3_route_check.py <repo> <sage EXL3 dir>   (run under flock ~/b70/gpu.lock)
"""
import os, sys
import numpy as np
import pyopencl as cl

sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref, exl3_check as ck


def main():
    repo, sage = sys.argv[1], sys.argv[2]
    e = ck.Exl3(repo)
    st = ref.SafeTensors(sage)
    rng = np.random.default_rng(4)
    E, TOPK = 256, 8
    for proj in ("gate_proj", "down_proj"):
        groups = [st.group(f"model.language_model.layers.0.mlp.experts.{x}.{proj}") for x in range(E)]
        kt, nt, w = groups[0][0].shape
        k, n = kt * 16, nt * 16
        for tokens in (1, 4, 16):
            ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(tokens)]).astype(np.uint32)
            if tokens > 1:
                ids[1, 0] = ids[0, 0]  # one expert picked by two tokens
            per = proj == "down_proj"
            x = ck.to_bf16(rng.standard_normal((tokens * TOPK if per else tokens, k)).astype(np.float32))
            ar = ck.Arena(e.ctx)
            table = np.zeros(E, ck.DESC)
            for i, (tr, suh, svh, mul1) in enumerate(groups):
                ck.refuse(w // 8, mul1)
                table[i]["trellis"], table[i]["suh"], table[i]["svh"] = ar.add(tr), ar.add(suh), ar.add(svh)
                table[i]["k"], table[i]["n"], table[i]["halves"] = k, n, w // 8
            slots = tokens * TOPK
            xo, ro, so, oo = ar.add(x), ar.add(nbytes=slots * k * 4), ar.add(nbytes=slots * n * 4), ar.add(nbytes=slots * n * 4)
            base = ar.build(e.q)
            tb, ib = e.buf(table), e.buf(ids.ravel())
            db = cl.Buffer(e.ctx, cl.mem_flags.READ_WRITE, slots * ck.DESC.itemsize)
            e.p.exl3_route(e.q, (slots,), None, tb, ib, db, np.uint32(TOPK), np.uint64(xo), np.uint32(k), np.uint32(per),
                           np.uint64(ro), np.uint64(so), np.uint64(oo), np.uint32(n), np.uint32(1))
            mv = e.matvec(w // 16, 1)
            e.p.exl3_rotate_in(e.q, (k // 128 * 16, 1, slots), (16, 1, 1), base, db)
            mv(e.q, ((n // 16 * 16 + 127) // 128 * 128, slots), (128, 1), base, db)
            e.p.exl3_rotate_out(e.q, (n // 128 * 16, 1, slots), (16, 1, 1), base, db)
            e.q.finish()
            y = ar.read(e.q, oo, (tokens, TOPK, n), np.float32)
            worst, cache = 0.0, {}
            for t in range(tokens):
                for j in range(TOPK):
                    i = int(ids[t, j])
                    if i not in cache:
                        cache[i] = ref.dequantize(w // 8, 2, *groups[i][:3])
                    yr = ck.from_bf16(x[t * TOPK + j] if per else x[t]).astype(np.float64) @ cache[i]
                    worst = max(worst, np.abs(y[t, j] - yr).max() / np.abs(yr).max())
            print(f"route {proj} x per {'slot' if per else 'token'} tokens={tokens:2d} slots={slots:3d}: worst max|y-ref|/max|ref| = {worst:.2e}")
            assert worst <= 1e-3


if __name__ == "__main__":
    main()
