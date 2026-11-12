#!/usr/bin/env python3
"""kit_variants.py: which captured Triton variants a dsv41 kit has, which a round-row bound needs, and the plan of the
missing ones (standard library only; runs anywhere, no GPU, no Triton).

  kit_variants.py list   --kit KIT [--fn NAME]              variants by function (constexprs, runtime specialization)
  kit_variants.py audit  --kit KIT --rows N [--json]        every launch a round of 1..N rows makes whose variant
                         [--engine-split]                   depends on its rows, resolved as cuda/aot.zig resolves it;
                                                            exit 1 when one has no variant (MissingTritonVariant).
                                                            --engine-split: as the engine serves it, a _rowmm_parts
                                                            gate past 16 rows with no variant is covered ('split')
                                                            when its 16-row pieces and leftover all resolve
                                                            (tri_norm.rowmmPartsPiece); without it, strict: every
                                                            gate as one launch (what plan builds)
  kit_variants.py plan   --kit KIT --rows N [--out SPEC]    the missing variants as aot_extra.py's spec (dry run: print)
                         [--heads64]                        ... plus a single-launch 64-head owner decode's variants
  kit_variants.py selftest                                  the audit on a pseudo-kit made of the recorded launches

KIT is a kit folder (its aot/aot.json), an aot.json, or `fixtures`: the engine's recorded launches
(zig/src/families/dsv41/fixtures/launches.json) as a pseudo-kit, each launch a variant of exactly its specialization,
for planning without a kit at hand (a real kit can only resolve more launches, by fallback, never fewer exactly). The launches modelled are the engine's own wrappers (Zig):
tri_norm.rowmmGate (_rowmm_parts: RB = rows, BN by rpTile), tri_hc.mixPart (_hc_mix_part: RB = hcRb(rows), the
`rows` argument) and tri_attn.sparseAttn's merges (_sparse_attn_merge_rot2 / _rot: the `rows` argument). Every other
Triton launch of a round depends on its rows only through its grid, which needs no variant (docs/port/kit.md has the
audit behind that). The draft's MoE gate (N = draft experts) runs only up to 16 rows whatever the round bound
(draft.zig moe: tri.decode_rows is 16 outside round.forward), so its families are checked to 16.

Matching follows zig/src/cuda/aot.zig exactly: the same constexprs, then the variant compiled for this specialization
(an int is 1, divisible by 16, or neither; a pointer 16-aligned or not), else the fitting one that assumes less.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROUND_BOUND = 16           # the qualified round-row bound (TF_DS_ROUND_ROWS unset): families in use at 16 rows
DRAFT_BOUND = 16           # the draft passes' row-invariant bound (tri.decode_rows outside round.forward)
EXPERTS = 384              # config.experts: the round's MoE gate N
HIDDEN = 5120
FIXTURES = Path(__file__).resolve().parents[3] / "zig/src/families/dsv41/fixtures/launches.json"


# -- the engine's row-dependent launch rules (mirrors of the Zig wrappers) ---------------------------------------------

def rp_tile_bn(rows: int) -> int:
    """tri_norm.rpTile(rows).bn."""
    return 16 if rows <= 6 else 8


HC_RB = [1, 2, 3, 2, 3, 2, 4, 4]


def hc_rb(rows: int) -> int:
    """tri_hc.hcRb(rows)."""
    if 1 <= rows <= len(HC_RB):
        return HC_RB[rows - 1]
    return 4 if rows <= 12 else 8


# fn -> (constexprs a row count sets, the int arguments it sets, keys that are not part of a family, round-bound filter)
RULES = {
    "_rowmm_parts": {
        "consts": lambda r: {"RB": r, "BN": rp_tile_bn(r)},
        "ints": lambda r: {"rows": r},
        "row_keys": {"RB", "BN", "rows"},
        # the round's gate (N = experts) follows the round bound; the draft's (any other N) stops at DRAFT_BOUND
        "bound": lambda fam, rows: rows if fam.get("N") == EXPERTS else min(rows, DRAFT_BOUND),
    },
    "_hc_mix_part": {
        "consts": lambda r: {"RB": hc_rb(r)},
        "ints": lambda r: {"rows": r},
        "row_keys": {"RB", "rows"},
        "bound": lambda fam, rows: rows,
    },
    "_sparse_attn_merge_rot2": {
        "consts": lambda r: {},
        "ints": lambda r: {"rows": r},
        "row_keys": {"rows"},
        "bound": lambda fam, rows: rows,
    },
    "_sparse_attn_merge_rot": {
        "consts": lambda r: {},
        "ints": lambda r: {"rows": r},
        "row_keys": {"rows"},
        "bound": lambda fam, rows: rows,
    },
}

# a single-launch owner decode (0118 stage 4, not the recommended path): these functions at H = 64
HEADS64 = ("_sparse_attn_part", "_sparse_attn_merge_rot2", "_sparse_attn_merge")


# -- kits and variants -------------------------------------------------------------------------------------------------

def load_kit(path: str) -> list[dict]:
    if path == "fixtures":  # the engine's recorded launches as a pseudo-kit (no kit at hand: a dry run's plan)
        return pseudo_kit()
    p = Path(path)
    if p.is_dir():
        p = p / "aot" / "aot.json" if (p / "aot" / "aot.json").exists() else p / "aot.json"
    return json.loads(p.read_text())["kernels"]


def cval(c: dict):
    """An aot.json constexpr as compared: ints (bools as 0/1), floats by their fp32 bits."""
    return ("f32", c["f32"]) if c.get("f32") is not None else ("int", c.get("int"))


def spec_class(x: int, nospec: bool) -> str:
    """How Triton specializes an int argument: 'one' (folded to a constexpr), 'd16' or 'any'."""
    if nospec:
        return "any"
    if x == 1:
        return "one"
    return "d16" if x % 16 == 0 else "any"


class Launch:
    """A launch as aot.zig sees it: constexprs (name -> cval) and runtime args (name -> ('ptr', type, aligned) |
    ('i32', value) | ('fp32',))."""

    def __init__(self, fn: str, consts: dict, args: dict):
        self.fn, self.consts, self.args = fn, consts, args

    def key(self):
        return (self.fn, tuple(sorted(self.consts.items())), tuple(sorted((k, self._cls(k)) for k in self.args)))

    def _cls(self, name):
        a = self.args[name]
        if a[0] == "i32":
            return ("i32", "one" if a[1] == 1 else ("d16" if a[1] % 16 == 0 else "any"))
        if a[0] == "ptr":
            return ("ptr", a[1], a[2])
        return a


def matches(k: dict, ln: Launch) -> bool:
    """aot.zig matches()."""
    kc = k["consts"]
    for n, v in ln.consts.items():
        if n not in kc or cval(kc[n]) != v:
            return False
    params = {p["name"]: p for p in k["params"]}
    runtime = 0
    for n, a in ln.args.items():
        p = params.get(n)
        if a[0] == "i32":
            if p is None:
                if a[1] != 1 or n not in kc or cval(kc[n]) != ("int", 1):
                    return False
                continue
            if p["type"] != "i32" or (not p["nospec"] and a[1] == 1):
                return False
            if p["div16"] != (not p["nospec"] and a[1] % 16 == 0):
                return False
        elif a[0] == "ptr":
            if p is None or p["type"] != a[1] or p["div16"] != a[2]:
                return False
        else:
            if p is None or p["type"] != "fp32":
                return False
        runtime += 1
    return runtime == len(k["params"])


def fits(k: dict, ln: Launch) -> int | None:
    """aot.zig fits(): how many of the launch's specializations a narrower-or-equal variant shares, or None."""
    kc = k["consts"]
    for n, v in ln.consts.items():
        if n not in kc or cval(kc[n]) != v:
            return None
    params = {p["name"]: p for p in k["params"]}
    held = runtime = 0
    for n, a in ln.args.items():
        p = params.get(n)
        if a[0] == "i32":
            if p is None:
                if a[1] != 1 or n not in kc or cval(kc[n]) != ("int", 1):
                    return None
                held += 1
                continue
            if p["type"] != "i32":
                return None
            d16 = not p["nospec"] and a[1] != 1 and a[1] % 16 == 0
            if p["div16"] and not d16:
                return None
            if p["div16"] == d16 and (p["nospec"] or a[1] != 1):
                held += 1
        elif a[0] == "ptr":
            if p is None or p["type"] != a[1]:
                return None
            if p["div16"] and not a[2]:
                return None
            if p["div16"] == a[2]:
                held += 1
        else:
            if p is None or p["type"] != "fp32":
                return None
            held += 1
        runtime += 1
    return held if runtime == len(k["params"]) else None


def find(kit: list[dict], ln: Launch):
    """aot.zig Set.find: (variant, 'exact' | 'fallback') or (None, 'missing')."""
    for k in kit:
        if k["fn"] == ln.fn and matches(k, ln):
            return k, "exact"
    best, held = None, -1
    for k in kit:
        if k["fn"] != ln.fn:
            continue
        h = fits(k, ln)
        if h is not None and h > held:
            best, held = k, h
    return (best, "fallback") if best is not None else (None, "missing")


def template_args(k: dict) -> dict:
    """The runtime arguments of a launch like the one variant k was compiled for (ints by a representative value)."""
    args = {}
    for p in k["params"]:
        if p["type"] == "i32":
            args[p["name"]] = ("i32", 16 if p["div16"] else 3)
        elif p["type"] == "fp32":
            args[p["name"]] = ("fp32",)
        else:
            args[p["name"]] = ("ptr", p["type"], p["div16"])
    for n, c in k["consts"].items():
        # an int argument Triton folded because it was 1 (not a declared constexpr of the wrapper): `rows` here
        if n in ("rows",) and cval(c) == ("int", 1):
            args[n] = ("i32", 1)
    return args


def families(kit: list[dict], fn: str, row_keys: set) -> dict:
    """fn's variants grouped by their constexprs other than the row-dependent ones."""
    fams: dict = {}
    for k in kit:
        if k["fn"] != fn:
            continue
        fam = tuple(sorted((n, cval(c)) for n, c in k["consts"].items() if n not in row_keys))
        fams.setdefault(fam, []).append(k)
    return fams


def launch_for(fn: str, rule: dict, fam: tuple, tmpl: dict, rows: int) -> Launch:
    consts = dict(fam)
    consts.update({n: ("int", v) for n, v in rule["consts"](rows).items()})
    args = {n: a for n, a in template_args(tmpl).items() if n not in rule["ints"](rows)}
    args.update({n: ("i32", v) for n, v in rule["ints"](rows).items()})
    return Launch(fn, consts, args)


SPLIT_PIECE = 16          # tri_norm.rowmm_parts_piece: the rows of each piece of a gate the kit cannot run in one launch


def split_pieces(rows: int) -> list[int]:
    """tri_norm.rowmmGate's pieces of a gate past SPLIT_PIECE rows: 16-row pieces, then the leftover."""
    return [SPLIT_PIECE] * (rows // SPLIT_PIECE) + ([rows % SPLIT_PIECE] if rows % SPLIT_PIECE else [])


def audit(kit: list[dict], rows: int, engine_split: bool = False) -> list[dict]:
    """Every distinct row-dependent launch of rounds of 1..rows rows of each family in use at ROUND_BOUND rows; a launch
    is listed once, at the fewest rows that make it (rows 32 specializes as 16 does, so a family's launch at 32 rows
    is its 16-row one). engine_split: a _rowmm_parts launch past SPLIT_PIECE rows with no variant is 'split' when
    every piece the engine would launch instead resolves (the pieces keep X's and S's alignment, as in the engine)."""
    out, seen = [], set()
    for fn, rule in RULES.items():
        for fam, ks in sorted(families(kit, fn, rule["row_keys"]).items(), key=lambda kv: str(kv[0])):
            # a template: prefer a variant whose `rows` is a runtime argument (not folded to 1)
            tmpl = sorted(ks, key=lambda k: (any(p["name"] == "rows" for p in k["params"]) is False, k["hash"]))[0]
            famd = dict((n, v[1]) for n, v in fam)
            if find(kit, launch_for(fn, rule, fam, tmpl, min(ROUND_BOUND, rule["bound"](famd, ROUND_BOUND))))[1] == "missing":
                continue  # a family the qualified 16-row config never launches at 16 rows (or a pinned shape)
            for r in range(1, rule["bound"](famd, rows) + 1):
                ln = launch_for(fn, rule, fam, tmpl, r)
                if ln.key() in seen:
                    continue
                seen.add(ln.key())
                k, how = find(kit, ln)
                pieces = None
                if engine_split and how == "missing" and fn == "_rowmm_parts" and r > SPLIT_PIECE:
                    got = [find(kit, launch_for(fn, rule, fam, tmpl, pr))[0] for pr in split_pieces(r)]
                    if all(g is not None for g in got):
                        how, pieces = "split", "+".join(str(pr) for pr in split_pieces(r))
                out.append({"fn": fn, "rows": r, "family": famd, "consts": {n: v[1] for n, v in ln.consts.items()},
                            "ints": {n: a[1] for n, a in ln.args.items() if a[0] == "i32"},
                            "how": how, "variant": k["hash"][:12] if k else pieces, "template": tmpl["hash"]})
    return out


def heads64(kit: list[dict]) -> list[dict]:
    """A single-launch 64-head owner decode: every family of HEADS64 at H = 32, wanted again at H = 64."""
    out, seen = [], set()
    for fn in HEADS64:
        for k in kit:
            if k["fn"] != fn or cval(k["consts"].get("H", {"int": 0})) != ("int", 32):
                continue
            consts = {n: cval(c) for n, c in k["consts"].items()}
            consts["H"] = ("int", 64)
            ln = Launch(fn, consts, template_args(k))
            if ln.key() in seen:
                continue
            seen.add(ln.key())
            v, how = find(kit, ln)
            out.append({"fn": fn, "rows": None, "family": {}, "consts": {n: x[1] for n, x in consts.items()},
                        "ints": {n: a[1] for n, a in ln.args.items() if a[0] == "i32"}, "how": how,
                        "variant": v["hash"][:12] if v else None, "template": k["hash"]})
    return out


# -- typed constexprs for the compiler (aot.json keeps bools as ints and floats as fp32 bits) --------------------------

def typed_consts(fn: str, consts: dict) -> dict:
    """The constexprs as Python passed them, from the recorded launches (fixtures/launches.json): a bool stays a bool,
    a float keeps its fp64 bits. Unknown names stay ints."""
    kinds: dict = {}
    try:
        fx = json.loads(FIXTURES.read_text()).get(fn, [])
    except OSError:
        fx = []
    for e in fx:
        for n, v in e["consts"].items():
            if isinstance(v, bool):
                kinds[n] = "bool"
            elif isinstance(v, dict) and "fp64_bits" in v:
                kinds.setdefault(n, ("float", v["fp64_bits"]))
    out = {}
    for n, v in consts.items():
        k = kinds.get(n)
        if k == "bool":
            out[n] = bool(v)
        elif isinstance(k, tuple):
            out[n] = {"fp64_bits": k[1]}
        else:
            out[n] = v
    return out


def plan(kit: list[dict], rows: int, with_heads64: bool) -> dict:
    need = [a for a in audit(kit, rows) if a["how"] == "missing"]
    if with_heads64:
        need += [a for a in heads64(kit) if a["how"] == "missing"]
    variants = []
    for a in need:
        # the row-set ints only; every other argument keeps its template's specialization
        rule = RULES.get(a["fn"])
        ints = {n: v for n, v in a["ints"].items() if rule and n in rule["ints"](1)}
        variants.append({"fn": a["fn"], "consts": typed_consts(a["fn"], a["consts"]), "ints": ints,
                         "template": a["template"], "why": f"rows {a['rows']}" if a["rows"] else "64-head owner decode"})
    return {"generator": "tools/dsv41-zig/kit/kit_variants.py", "rows": rows, "heads64": with_heads64,
            "variants": variants}


# -- the commands ------------------------------------------------------------------------------------------------------

def cmd_list(a) -> int:
    kit = load_kit(a.kit)
    by: dict = {}
    for k in kit:
        by.setdefault(k["fn"], []).append(k)
    for fn in sorted(by):
        if a.fn and fn != a.fn:
            continue
        print(f"{fn}: {len(by[fn])} variants")
        if a.fn:
            for k in sorted(by[fn], key=lambda k: json.dumps(k["consts"], sort_keys=True)):
                cs = " ".join(f"{n}={cval(c)[1]}" for n, c in sorted(k["consts"].items()))
                ps = " ".join(f"{p['name']}:{p['type']}{'/16' if p['div16'] else ''}" for p in k["params"])
                print(f"  {k['hash'][:12]} w{k['num_warps']} {cs} | {ps}")
    return 0


def beyond_published(a, res: list[dict]) -> list[dict]:
    """With the fixtures pseudo-kit, only the launches past the qualified bound: the fixtures record a sample of the
    launches, while the published kit serves every round of 1..16 rows (the deployed config), so a pseudo-kit miss at
    16 rows or fewer is a sampling gap, not a kit gap."""
    if a.kit != "fixtures":
        return res
    return [r for r in res if r["rows"] is None or r["rows"] > ROUND_BOUND]


def cmd_audit(a) -> int:
    kit = load_kit(a.kit)
    res = beyond_published(a, audit(kit, a.rows, a.engine_split))
    miss = [r for r in res if r["how"] == "missing"]
    split = [r for r in res if r["how"] == "split"]
    if a.json:
        print(json.dumps({"rows": a.rows, "engine_split": a.engine_split, "launches": res, "missing": len(miss),
                          "split": len(split)}, indent=1))
    else:
        for r in res:
            fam = " ".join(f"{n}={v}" for n, v in sorted(r["family"].items()))
            mark = {"exact": "ok      ", "fallback": "fallback", "missing": "MISSING ", "split": "split   "}[r["how"]]
            print(f"{mark} {r['fn']:<26} rows {r['rows']:>2}  {fam}  -> {r['variant'] or '-'}")
        how = f", {len(split)} served by the engine's 16-row split" if a.engine_split else " (strict: one launch a gate)"
        print(f"{len(res)} distinct row-dependent launches for rounds of 1..{a.rows} rows; {len(miss)} missing{how}")
    return 1 if miss else 0


def cmd_plan(a) -> int:
    spec = plan(load_kit(a.kit), a.rows, a.heads64)
    if a.kit == "fixtures":  # a dry run's table: its templates are synthetic, aot_extra.py refuses it
        spec["pseudo_kit"] = True
        spec["variants"] = [v for v in spec["variants"] if not v["why"].startswith("rows ") or int(v["why"][5:]) > ROUND_BOUND]
    text = json.dumps(spec, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
        print(f"{len(spec['variants'])} variants to build -> {a.out}")
    else:
        sys.stdout.write(text)
    return 0


def pseudo_kit() -> list[dict]:
    """The recorded launches (fixtures/launches.json) as aot.json variants: each distinct launch a variant compiled
    for exactly its specialization (no do_not_specialize known)."""
    ty = {"bfloat16": "*bf16", "float32": "*fp32", "float16": "*fp16", "int64": "*i64", "int32": "*i32",
          "uint8": "*u8", "bool": "*u8"}
    out, seen = [], set()
    for fn, launches in json.loads(FIXTURES.read_text()).items():
        for e in launches:
            params = []
            for n, t in e["tensors"].items():
                params.append({"name": n, "type": ty[t[0]], "div16": bool(t[3]), "nospec": False})
            for n, v in e.get("scalars", {}).items():
                if isinstance(v, dict):
                    params.append({"name": n, "type": "fp32", "div16": False, "nospec": False})
                else:
                    params.append({"name": n, "type": "i32", "div16": v % 16 == 0, "nospec": False})
            consts = {}
            for n, v in e["consts"].items():
                if isinstance(v, dict):
                    consts[n] = {"f32": int(v["fp32_bits"], 16)}
                else:
                    consts[n] = {"int": int(v)}
            h = json.dumps([fn, consts, params], sort_keys=True)
            if h in seen:
                continue
            seen.add(h)
            out.append({"fn": fn, "hash": hashlib.sha256(h.encode()).hexdigest()[:32], "num_warps": e.get("num_warps", 4),
                        "params": params, "consts": consts})
    return out


def cmd_selftest(_a) -> int:
    kit = pseudo_kit()
    ok = True

    def check(what, cond):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok = ok and cond

    r16 = audit(kit, 16)
    r32 = audit(kit, 32)
    rb32 = sorted(a["consts"]["RB"] for a in r32 if a["fn"] == "_rowmm_parts" and a["how"] == "missing"
                  and a["family"].get("N") == EXPERTS)
    check("_rowmm_parts N=384 RB 17..32 missing at 32 rows", set(range(17, 33)) <= set(rb32))
    check("no draft (N=128) family checked past 16 rows",
          all(a["rows"] <= 16 for a in r32 if a["fn"] == "_rowmm_parts" and a["family"].get("N") != EXPERTS))
    check("_rowmm_parts RB=16 resolves exactly", any(a["fn"] == "_rowmm_parts" and a["rows"] == 16 and a["how"] == "exact"
                                                     for a in r16))
    check("_hc_mix_part needs no launch shape past 16 rows (RB 8 from 13 rows; rows 32 specializes as 16 does)",
          not any(a["fn"] == "_hc_mix_part" and a["rows"] > 16 for a in r32))
    check("_sparse_attn_merge_rot2 needs no launch shape past 16 rows",
          not any(a["fn"] == "_sparse_attn_merge_rot2" and a["rows"] > 16 for a in r32))
    pl = plan(kit, 32, False)
    check("the plan builds RB 17..32 of _rowmm_parts",
          {v["consts"]["RB"] for v in pl["variants"] if v["fn"] == "_rowmm_parts" and v["consts"]["N"] == EXPERTS}
          >= set(range(17, 33)))
    check("the plan types PDL as a bool (as Triton hashed it)",
          all(v["consts"].get("PDL") is True for v in pl["variants"] if v["fn"] == "_rowmm_parts"))
    rb = [v for v in pl["variants"] if v["fn"] == "_rowmm_parts" and v["consts"]["RB"] == 32]
    check("RB=32 is planned with rows=32 (divisible by 16, as the launch is)", bool(rb) and rb[0]["ints"]["rows"] == 32)
    s32 = [a for a in audit(kit, 32, engine_split=True) if a["fn"] == "_rowmm_parts" and a["family"].get("N") == EXPERTS
           and a["rows"] == 32]
    check("engine split: a 32-row gate on a kit without RB 32 is covered as 16+16", bool(s32) and s32[0]["how"] == "split"
          and s32[0]["variant"] == "16+16")
    check("split pieces: 33 rows are 16+16+1, 16 rows one piece", split_pieces(33) == [16, 16, 1] and split_pieces(16) == [16])
    check("strict audit never reports a split", not any(a["how"] == "split" for a in r32))
    h = heads64(kit)
    check("heads64 lists the attention functions at H=64", {a["fn"] for a in h} >= {"_sparse_attn_part", "_sparse_attn_merge"}
          and all(a["consts"]["H"] == 64 for a in h))
    # matching edge cases straight from aot.zig: a d16 variant cannot run an odd launch; an 'any' one can run d16
    k_any = {"fn": "f", "hash": "a" * 32, "params": [{"name": "n", "type": "i32", "div16": False, "nospec": False}], "consts": {}}
    k_d16 = {"fn": "f", "hash": "b" * 32, "params": [{"name": "n", "type": "i32", "div16": True, "nospec": False}], "consts": {}}
    check("aot.zig fallback: div16 variant refuses n=17", fits(k_d16, Launch("f", {}, {"n": ("i32", 17)})) is None)
    check("aot.zig fallback: plain variant takes n=32", find([k_any], Launch("f", {}, {"n": ("i32", 32)}))[1] == "fallback")
    check("aot.zig fallback: n=1 runs a plain variant", find([k_any, k_d16], Launch("f", {}, {"n": ("i32", 1)}))[1] == "fallback")
    check("aot.zig: n=1 has no variant among div16 ones", find([k_d16], Launch("f", {}, {"n": ("i32", 1)}))[1] == "missing")
    print("selftest " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--kit", required=True)
    p.add_argument("--fn")
    p.set_defaults(run=cmd_list)
    p = sub.add_parser("audit")
    p.add_argument("--kit", required=True)
    p.add_argument("--rows", type=int, default=ROUND_BOUND)
    p.add_argument("--json", action="store_true")
    p.add_argument("--engine-split", action="store_true")
    p.set_defaults(run=cmd_audit)
    p = sub.add_parser("plan")
    p.add_argument("--kit", required=True)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--heads64", action="store_true")
    p.add_argument("--out")
    p.set_defaults(run=cmd_plan)
    p = sub.add_parser("selftest")
    p.set_defaults(run=cmd_selftest)
    a = ap.parse_args()
    if getattr(a, "rows", 1) is not None and not 1 <= getattr(a, "rows", 1) <= 48:
        ap.error("--rows: 1 .. 48 (round.max_rows)")
    return a.run(a)


if __name__ == "__main__":
    raise SystemExit(main())
