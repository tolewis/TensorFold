#!/usr/bin/env python3
"""aot_extra.py KIT SPEC: compile the Triton variants a plan names (kit_variants.py plan) from the served engine's own
kernel source and add them to KIT's aot/ (aot.json + cubins/), as aot_pack.py packs a recording's.

UNTESTED: written without the engine image at hand; run it on a Spark inside the deepseek-v41-tp2 engine image (the
same Triton and kernel source the kit was recorded with), e.g. through build_kit.sh's EXTRA step:

  docker run --rm -v "$KIT":/kit -v "$TF":/tf:ro "$IMAGE" python3 /tf/tools/dsv41-zig/kit/aot_extra.py /kit SPEC

It needs no GPU by design (triton.compile with an explicit GPUTarget only runs Triton's passes and its bundled ptxas);
should this Triton insist on a driver anyway, add `--gpus all` (it still launches nothing).

How a variant is made: the plan names a template, a variant of the same function already in the kit. Its runtime
argument types and 16-byte assumptions are kept; the plan's constexprs replace the template's, and each `ints` value
re-specializes that argument as Triton's JIT would (1: a constexpr; divisible by 16: tt.divisibility; else plain). The
template's own launch options (num_warps, num_ctas, launch_pdl, num_stages) are recovered first by recompiling the
template itself until its hash equals the recorded one (--no-check skips that, at the risk of a variant that differs
from what the engine would have compiled). The hash is Triton's cache key, so a variant built here equals the
engine's own JIT build of that specialization in this image, and aot.zig takes it by its constexprs exactly as a
recorded one.

  --dry-run   print what would be compiled (no Triton import)
  --arch N    the SASS target (121: GB10)
Exit 0: every variant added (or already present); 2: bad input; 3: a template does not reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import struct
import sys
from pathlib import Path


def fail(code: int, msg: str):
    print(f"aot_extra: {msg}", file=sys.stderr)
    raise SystemExit(code)


def value(v):
    """A plan's constexpr as Python passes it: bool, int, or a float from its fp64 bits."""
    if isinstance(v, dict) and "fp64_bits" in v:
        return struct.unpack("<d", int(v["fp64_bits"], 16).to_bytes(8, "little"))[0]
    if isinstance(v, dict) and "f32" in v:
        return struct.unpack("<f", int(v["f32"]).to_bytes(4, "little"))[0]
    return v


def kit_const(v):
    """aot_pack.py const(): ints and bools as integers, floats by their fp32 bits."""
    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"f32": struct.unpack("<I", struct.pack("<f", v))[0]}
    fail(2, f"constexpr {v!r} has no Zig form")


def jit_functions() -> dict:
    """Every Triton JITFunction of the engine's modules by function name (zrec_fixtures.jit_table's import walk)."""
    from triton.runtime.jit import JITFunction

    for base in ("tensorfold.families.deepseek_v41", "tensorfold.cuda"):
        for p in list(importlib.import_module(base).__path__):
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not d.startswith("."))
                for fn in sorted(filenames):
                    if fn.endswith(".py") and not fn.startswith("_"):
                        name = base + "." + os.path.relpath(os.path.join(dirpath, fn), p)[:-3].replace(os.sep, ".")
                        try:
                            importlib.import_module(name)
                        except Exception as e:  # a module the kit's kernels never came from may need a GPU to import
                            print(f"aot_extra: skipped {name} ({e!r})", file=sys.stderr)
    out: dict = {}
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("tensorfold."):
            for v in vars(mod).values():
                if isinstance(v, JITFunction) and v not in out.setdefault(v.fn.__name__, []):
                    out[v.fn.__name__].append(v)
    return out


def pick_fn(cands: list, tmpl: dict):
    """The JITFunction whose arguments are exactly the template's runtime parameters plus its constexprs."""
    want = {p["name"] for p in tmpl["params"]} | set(tmpl["consts"])
    hits = [f for f in cands if set(f.arg_names) == want]
    if len(hits) != 1:
        fail(2, f"{tmpl['fn']}: {len(hits)} JIT functions take exactly the template's arguments")
    return hits[0]


def nospec_of(fn) -> set:
    names = list(fn.arg_names)
    return {names[i] if isinstance(i, int) else i for i in (fn.do_not_specialize or [])}


def source(fn, tmpl: dict, consts: dict, ints: dict):
    """The ASTSource of a specialization (the template's argument types and assumptions, these constexprs and ints),
    with its signature, the arguments it assumes 16-divisible and its constexprs by name."""
    from triton.compiler import ASTSource

    params = {p["name"]: p for p in tmpl["params"]}
    nospec = nospec_of(fn)
    sig, cx, attrs, d16s = {}, {}, {}, set()
    for i, n in enumerate(fn.arg_names):
        if n in consts:
            sig[n], cx[n] = "constexpr", value(consts[n])
            continue
        if n not in params:
            fail(2, f"{tmpl['fn']}: argument {n} is neither a constexpr of the plan nor a template parameter")
        p = params[n]
        if n in ints and p["type"] == "i32" and ints[n] == 1 and n not in nospec:
            sig[n], cx[n] = "constexpr", 1  # Triton folds an int argument of 1
            continue
        sig[n] = p["type"]
        d16 = (n not in nospec and ints[n] % 16 == 0) if (n in ints and p["type"] == "i32") else p["div16"]
        if d16:
            attrs[(i,)] = [["tt.divisibility", 16]]
            d16s.add(n)
    return ASTSource(fn=fn, signature=sig, constexprs=cx, attrs=attrs), sig, d16s, cx


def options_of(tmpl: dict, stages) -> dict:
    from triton.backends.nvidia.compiler import CUDAOptions

    fields = set(CUDAOptions.__dataclass_fields__)
    o = {"num_warps": tmpl["num_warps"], "num_ctas": tmpl.get("num_ctas", 1)}
    if "launch_pdl" in fields:
        o["launch_pdl"] = bool(tmpl.get("pdl", False))
    if stages is not None:
        o["num_stages"] = stages
    return o


def compile_one(fn, tmpl: dict, consts: dict, ints: dict, opts: dict, arch: int):
    """(compiled kernel, signature, 16-divisible arguments, constexprs)."""
    import triton
    from triton.backends.compiler import GPUTarget

    src, sig, d16s, cx = source(fn, tmpl, consts, ints)
    return triton.compile(src, target=GPUTarget("cuda", arch, 32), options=opts), sig, d16s, cx


def template_consts(tmpl: dict) -> dict:
    """The template's own constexprs typed as the engine passed them (bools and floats from the recorded fixtures)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kit_variants import typed_consts

    raw = {n: (c["int"] if c.get("int") is not None else {"f32": c["f32"]}) for n, c in tmpl["consts"].items()}
    return typed_consts(tmpl["fn"], raw)


def reproduce(fn, tmpl: dict, arch: int, check: bool) -> dict:
    """The launch options that rebuild the template's own hash (num_stages tried: Triton's default, then 1..4)."""
    consts = template_consts(tmpl)
    tried = []
    for stages in (None, 1, 2, 3, 4):
        opts = options_of(tmpl, stages)
        ck = compile_one(fn, tmpl, consts, {}, opts, arch)[0]
        if ck.hash == tmpl["hash"] or not check:
            return opts
        tried.append((stages, ck.hash[:12]))
    fail(3, f"{tmpl['fn']}: no launch options rebuild template {tmpl['hash'][:12]} (num_stages, hash: {tried}); "
            "this image's Triton or kernel source is not the kit's (or a constexpr's Python type differs): "
            "record the variants instead (build_kit.sh RECORDINGS with TF_DS_ROUND_ROWS) or pass --no-check")


def entry(fn, built) -> dict:
    """aot_pack.py entry() of a compiled kernel."""
    ck, sig, d16s, cx = built
    md = ck.metadata._asdict() if hasattr(ck.metadata, "_asdict") else dict(vars(ck.metadata))
    runtime = [n for n in fn.arg_names if sig[n] != "constexpr"]
    m = re.search(r"\.entry\s+\S+\s*\((.*?)\)", ck.asm["ptx"], re.S)
    nptx = len([x for x in m.group(1).split(",") if x.strip()]) if m else -1
    if nptx != len(runtime) + 2:
        fail(2, f"{fn.fn.__name__} {ck.hash}: PTX takes {nptx} parameters, not the {len(runtime)} runtime arguments + "
                "global and profile scratch (aot_pack.py's ABI)")
    if md.get("global_scratch_size", 0) or md.get("profile_scratch_size", 0):
        fail(2, f"{fn.fn.__name__} {ck.hash}: needs scratch memory, which cuda/aot.zig refuses")
    nospec = nospec_of(fn)
    return {
        "fn": fn.fn.__name__, "hash": ck.hash, "name": md["name"], "num_warps": md["num_warps"],
        "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
        "global_scratch": md.get("global_scratch_size", 0), "global_align": md.get("global_scratch_align", 1),
        "profile_scratch": md.get("profile_scratch_size", 0), "pdl": bool(md.get("launch_pdl", False)),
        "params": [{"name": n, "type": sig[n], "div16": n in d16s, "nospec": n in nospec} for n in runtime],
        "consts": {n: kit_const(v) for n, v in cx.items()},
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(kit: Path, files: list[str]) -> None:
    """MANIFEST (sha256sum format) with aot/aot.json's new sum and the added cubins; the original kept as MANIFEST.base.
    A kit with no MANIFEST gets a fresh one over every file it holds, so its verify_kit.sh has something to check."""
    mf = kit / "MANIFEST"
    if not mf.exists():
        files = sorted(str(f.relative_to(kit)) for f in kit.rglob("*")
                       if f.is_file() and f.name not in ("MANIFEST", "MANIFEST.base") and ".git" not in f.parts)
        mf.write_text("".join(f"{sha256_file(kit / f)}  {f}\n" for f in files))
        return
    base = kit / "MANIFEST.base"
    if not base.exists():
        base.write_text(mf.read_text())
    lines = {}
    for ln in mf.read_text().splitlines():
        if ln.strip():
            s, p = ln.split(None, 1)
            lines[p.strip()] = s
    for f in files:
        lines[f] = sha256_file(kit / f)
    mf.write_text("".join(f"{s}  {p}\n" for p, s in sorted(lines.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kit", help="the kit folder (holding aot/aot.json)")
    ap.add_argument("spec", help="kit_variants.py plan's output")
    ap.add_argument("--arch", type=int, default=121)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    kit = Path(a.kit)
    aot = kit / "aot" / "aot.json"
    meta = json.loads(aot.read_text())
    by_hash = {k["hash"]: k for k in meta["kernels"]}
    spec = json.loads(Path(a.spec).read_text())
    if spec.get("pseudo_kit"):
        fail(2, "this plan was made against the fixtures pseudo-kit (synthetic template hashes): plan against the kit itself")
    todo = spec.get("variants", [])
    for v in todo:
        if v["template"] not in by_hash:
            fail(2, f"{v['fn']}: template {v['template'][:12]} is not in {aot}")
    if a.dry_run:
        for v in todo:
            cs = " ".join(f"{n}={value(x)}" for n, x in sorted(v["consts"].items()))
            print(f"would compile {v['fn']} [{v.get('why', '')}] {cs} ints {v['ints']} from {v['template'][:12]}")
        print(f"{len(todo)} variants; nothing compiled (--dry-run)")
        return 0
    fns = jit_functions()
    opts_of: dict = {}
    added, files = [], []
    for v in todo:
        tmpl = by_hash[v["template"]]
        fn = pick_fn(fns.get(v["fn"], []), tmpl)
        if tmpl["hash"] not in opts_of:
            opts_of[tmpl["hash"]] = reproduce(fn, tmpl, a.arch, not a.no_check)
        built = compile_one(fn, tmpl, v["consts"], v["ints"], opts_of[tmpl["hash"]], a.arch)
        ck = built[0]
        if ck.hash in by_hash:
            print(f"have {v['fn']} {ck.hash[:12]} [{v.get('why', '')}]")
            continue
        e = entry(fn, built)
        (kit / "aot" / "cubins").mkdir(parents=True, exist_ok=True)
        (kit / "aot" / "cubins" / f"{ck.hash}.cubin").write_bytes(ck.asm["cubin"])
        files.append(f"aot/cubins/{ck.hash}.cubin")
        by_hash[ck.hash] = e
        added.append(e)
        print(f"added {v['fn']} {ck.hash[:12]} [{v.get('why', '')}]")
    if added:
        meta["kernels"] = sorted(by_hash.values(), key=lambda x: (x["fn"], x["hash"]))
        meta["extra"] = sorted(set(meta.get("extra", [])) | {e["hash"] for e in added})
        meta["extra_generator"] = "tools/dsv41-zig/kit/aot_extra.py"
        aot.write_text(json.dumps(meta, indent=1) + "\n")
        update_manifest(kit, files + ["aot/aot.json"])
    print(json.dumps({"added": len(added), "planned": len(todo), "kit": str(kit)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
