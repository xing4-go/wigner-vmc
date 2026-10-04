"""Blocker B, stages B1+B2: rebuild the r_s = 75 crystal INITIALIZATION layer by
layer and locate the FIRST divergence, in the order the authorization fixes:

    c0 -> geometry inputs -> O_overlap -> v_seed -> v0 -> theta0 -> _init_digest

Why this file exists
--------------------
``tools/recon_rs75_init.py`` writes ``initial_conditions.json``, which
``scripts/bench_rs75.py --init`` reads as the Stage 2E starting point.  Its
``_init_digest(v0, c0)`` does not match the frozen campaign's own digest for any
of the five r_s = 75 converged seeds:

    s0  clean 4380bceb  frozen cadd5d6a      s3  clean b9d3cf7e  frozen 7f284e09
    s1  clean 5c12cfbe  frozen 46d9bf1e      s4  clean 186d86ff  frozen (absent)
    s2  clean b0e13b26  frozen eba63d0f

Guessing from a digest is exactly what B2 forbids, so this script measures the
chain instead.  Two branches are rebuilt side by side for the same seed:

  * LEGACY -- the generator production contract, ``_diag/part1_scan.py``'s crystal
    path, which produced the ``p1{ tier }{ seed }`` tags (line 741:
    ``label=f"p1{job['tier']}{job['seed']}"``; the label is in the cache NAME).
    It passes ``job["kappa"] = round(rs/sqrt(2), 4)`` = 53.033.
  * CLEAN  -- ``recon_rs75_init.build_init``, which passes
    ``KAPPA = 75/sqrt(2)`` = 53.03300858899106.

The two differ in exactly one input, and it is the *same* clean-vs-legacy kappa
split Blocker A closed, now entering through the Jastrow warm start instead of
through the Hamiltonian.

Two facts are already provable by reading, and are therefore CHECKED here rather
than assumed:

  1. with ``L0`` given explicitly, ``rs`` never reaches the overlap --
     ``gaussian_overlap_seed`` uses it only in ``drummond_width(rs)`` when
     ``L0 is None`` -- so ``O_overlap`` and ``v0`` must be bit-identical across the
     two branches and cannot be the divergence;
  2. ``_L0S`` is a function of the ``--seeds`` argument
     (``np.linspace(0, len(L0_GRID_J)-1, seeds)``), and ``_L0S[0]`` is invariant
     while every later entry moves.  That is why the frozen ``p1fast0`` tag shares
     seed 0's digest with ``p1conv0`` while ``p1fast1`` does not: the fast tier ran
     with a different ``--seeds``.  Digests are therefore only comparable within a
     run that used the same ``--seeds``.

READ-ONLY, AND ENFORCED.  The notebook's cells call ``cached()`` functions that
WRITE on a cache MISS, so ``CKPT`` is pointed at a working COPY and the frozen
tree is re-fingerprinted on exit (the fix for the 2026-10-02 incident).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frozen_guard import FROZEN_CKPT, readonly_frozen            # noqa: E402
from recon_rs75_init import CLEAN, namespace_cells               # noqa: E402

RS = 75.0
K_LEGACY = round(RS / np.sqrt(2.0), 4)          # 53.033     -- campaign convention
K_CLEAN = RS / np.sqrt(2.0)                     # 53.03300858899106 -- physical
FIVE = 5                                        # the conv tier ran with --seeds 5
NK = 36


def frozen_digest(name):
    """The ``i<digest>`` field of a frozen checkpoint tag, or (None, why not)."""
    path = os.path.join(FROZEN_CKPT, name + ".meta")
    if not os.path.exists(path):
        return None, "ABSENT (no .meta in the frozen store)"
    tag = open(path, encoding="utf-8").read().strip()
    last = tag.split("|")[-1]
    if not last.startswith("i"):
        return None, f"last field {last!r} is not an i<digest> field: {tag}"
    return last[1:], tag


def _A(x):
    return np.asarray(x, float)


def geometry_layer(E):
    """The lattice inputs, named by the ROLE Blocker A had to separate."""
    out = {}
    L1, L2 = _A(E["L1"]), _A(E["L2"])
    A1, A2 = L1 / 6.0, L2 / 6.0                 # primitive cell, notebook cell 10
    out["|L1| (simulation supercell)"] = float(np.linalg.norm(L1))
    out["|A1| (primitive)"] = float(np.linalg.norm(A1))
    out["CToCart == column_stack([A1, A2])"] = bool(
        np.array_equal(_A(E["CToCart"]), np.column_stack([A1, A2])))
    llb8 = E["_llb8B"]
    try:
        out["_llb8B class attrs"] = sorted(k for k in vars(llb8) if not k.startswith("__"))
    except TypeError:                       # __slots__
        out["_llb8B class attrs"] = "no __dict__ (slots)"
    for attr in ("a_ints", "a_cart", "nk", "n_max", "nband"):
        if hasattr(llb8, attr):
            v = getattr(llb8, attr)
            out[f"_llb8B.{attr}"] = list(np.shape(v)) if hasattr(v, "shape") else v
    out["_lints8 #vectors (Gaussian-overlap lattice)"] = int(_A(E["_lints8"]).shape[0])
    out["LUMAX"] = float(E["LUMAX"])
    out["LUMAX_LL"] = float(E.get("LUMAX_LL", float("nan")))
    out["NMAX_STAGE_B"] = int(E["NMAX_STAGE_B"])
    return out


def branch(E, si, kappa, L0, k_near, work, want_cache_note=False):
    """One complete initialization chain for one seed at one kappa."""
    rec = {"si": si, "L0": float(L0), "kappa": kappa}
    jpath = os.path.join(work, "jastrow_crystal_k53.033")
    before = os.stat(jpath + ".meta").st_mtime_ns if os.path.exists(jpath + ".meta") else None
    warm = _A(E["J_OPT"][("crystal", k_near)]) * (kappa / k_near)
    rec["warm[0]"] = float(warm[0])
    c0 = _A(E["jastrow_opt"]("crystal", kappa, L0=L0, warm=warm)["c"])
    after = os.stat(jpath + ".meta").st_mtime_ns if os.path.exists(jpath + ".meta") else None
    if want_cache_note:
        rec["cache"] = ("HIT -- the frozen bytes were returned"
                        if before is not None and before == after
                        else "MISS -- recomputed; the store's tag did not match")

    ov = np.asarray(E["gaussian_overlap_seed"](
        E["_llb8B"], int(E["NMAX_STAGE_B"]), E["L1"], E["L2"],
        E["_lints8"], E["_lcart8"], rs=RS, numx=121, L0=L0), complex)
    v0 = np.asarray(E["v_from_overlap"](ov), complex)
    rec.update(c0=c0, ov=ov, v0=v0,
               theta0=np.asarray(E["make_ll_theta"](c0, v0), float),
               digest=E["_init_digest"](v0, c0))
    return rec


def report_si(si, leg, cln, geo):
    print(f"\n{'='*78}\n  seed s{si}   L0 = {leg['L0']:g}\n{'='*78}")
    if geo:
        for k, v in geo.items():
            print(f"  [geometry] {k:46s} {v}")
    fdig, note = frozen_digest(f"llcryst_nb2_k53.033_p1conv{si}")
    print(f"  [c0]     legacy kappa {K_LEGACY:<8} c0[0] = {leg['c0'][0]:+.12e}"
          + (f"   {leg['cache']}" if "cache" in leg else ""))
    print(f"           clean  kappa {K_CLEAN:.12f}  c0[0] = {cln['c0'][0]:+.12e}")
    dc = float(np.abs(leg["c0"] - cln["c0"]).max())
    scale = max(float(np.abs(leg["c0"]).max()), 1e-300)
    print(f"           max|c0_legacy - c0_clean| = {dc:.3e}"
          f"   (c0 scale {scale:.6f} -> relative {dc / scale:.3e})"
          f"   bit-identical? {np.array_equal(leg['c0'], cln['c0'])}")
    same_ov = bool(np.array_equal(leg["ov"], cln["ov"]))
    same_v0 = bool(np.array_equal(leg["v0"], cln["v0"]))
    print(f"  [O_ov]   shape {list(leg['ov'].shape)}   bit-identical across kappa? {same_ov}")
    print(f"  [v0]     max|dv0| = {float(np.abs(leg['v0'] - cln['v0']).max()):.3e}"
          f"   bit-identical across kappa? {same_v0}")
    print(f"  [th0]    len {leg['theta0'].size} = 5 + 2*{NK}*{NB_BANDS - 1}"
          f"   bit-identical? {np.array_equal(leg['theta0'], cln['theta0'])}")
    print(f"  [digest] legacy kappa -> {leg['digest']}")
    print(f"           clean  kappa -> {cln['digest']}")
    if fdig is None:
        print(f"           frozen       -> {note}")
    else:
        print(f"           frozen       -> {fdig}   [{note}]")
        print(f"           LEGACY matches frozen: {leg['digest'] == fdig}"
              f"     CLEAN matches frozen: {cln['digest'] == fdig}")
    return {"si": si, "L0": leg["L0"], "digest_legacy": leg["digest"],
            "digest_clean": cln["digest"], "digest_frozen": fdig,
            "c0_max_abs_diff": dc, "c0_rel_diff": dc / scale,
            "c0_bit_identical": bool(np.array_equal(leg["c0"], cln["c0"])),
            "ov_bit_identical": same_ov, "v0_bit_identical": same_v0,
            "legacy_matches_frozen": (None if fdig is None else leg["digest"] == fdig),
            "clean_matches_frozen": (None if fdig is None else cln["digest"] == fdig)}


E_nb = 2            # NMAX_STAGE_B: the r_s = 75 crystal records are all nb = 2
NB_BANDS = E_nb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--si", type=int, action="append", default=None,
                    help="seed index to probe; repeatable (default 2,0,1,3,4 -- "
                         "s2 first, so its cache tag is tested before any clobber)")
    ap.add_argument("--out", default=os.path.join(CLEAN, "logs", "diag_init_layers.json"))
    args = ap.parse_args(argv)
    # s2 (L0 = 0.5) goes first because the frozen `jastrow_crystal_k53.033` store
    # currently holds an L0 = 0.5 write; every earlier call with another L0
    # overwrites the same filename, so only the FIRST call can report a HIT.
    sis = args.si if args.si is not None else [2, 0, 1, 3, 4]

    print(f"frozen store : {FROZEN_CKPT}")
    print(f"RS = {RS}   kappa legacy {K_LEGACY}   kappa clean {K_CLEAN:.12f}")
    print("building the notebook namespace against a WORKING COPY\n", flush=True)

    with readonly_frozen(FROZEN_CKPT) as work:
        E = namespace_cells(work)
        KAPPAS = _A(E["KAPPAS"])
        L0_GRID = _A(E["L0_GRID_J"])
        L0S = [float(L0_GRID[i]) for i in
               np.linspace(0, len(L0_GRID) - 1, FIVE).round().astype(int)]
        k_near = float(min(KAPPAS, key=lambda kk: abs(kk - K_LEGACY)))
        print(f"\nKAPPAS = {KAPPAS}\nL0_GRID_J = {L0_GRID}"
              f"\n_L0S (--seeds {FIVE}) = {L0S}   nearest kappa to {K_LEGACY} = {k_near}\n",
              flush=True)

        geo = geometry_layer(E)
        legs, clns = {}, {}
        for n, si in enumerate(sis):                 # every legacy call first: the
            legs[si] = branch(E, si, K_LEGACY, L0S[si], k_near, work,  # store is
                              want_cache_note=(n == 0))                # shared, so
        for si in sis:                                                 # only the
            clns[si] = branch(E, si, K_CLEAN, L0S[si], k_near, work)   # first can HIT
        rows = [report_si(si, legs[si], clns[si], geo if si == sis[0] else None)
                for si in sis]

    blob = {"rs": RS, "kappa_legacy": K_LEGACY, "kappa_clean": K_CLEAN, "_L0S": L0S,
            "nearest_kappa": k_near, "seeds_probed": sis, "geometry": geo,
            "seeds": rows,
            "frozen_tree": "unchanged (readonly_frozen re-fingerprinted on exit)"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print(f"\nwrote {args.out}")
    print("frozen tree: UNCHANGED -- the guard verified it on exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
