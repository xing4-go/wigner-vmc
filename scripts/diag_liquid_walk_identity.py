"""A8c -- the decisive test: the REAL legacy sampler against the REAL clean one.

`diag_liquid_a8.py` PART 2 and `diag_liquid_traj_full.py` both drove the walk with
their own inline replica of ``sample``.  That replica is faithful to the *code* of
``sample`` (same sweep loop, same 1000-attempt window, same reset_period), but it
has one coverage hole that matters: it generates the proposal with the CLEAN
``wrap_to_supercell`` for BOTH engines.  So it can prove clean's Markov kernel
matches legacy's, and it CANNOT see a divergence in the periodic wrap itself --
the single remaining link in the chain.

This script closes it the only way that counts: it calls

    legacy.sample(...)        the frozen engine's own function
    clean.sample(...)         the port's own function

with the identical protocol, and compares their returned sigma, acceptance rate
and every snapshot elementwise.

The frozen protocol, read off ``_diag/part1_scan.py:603-604``:

    R0 = (np.random.default_rng(3).random((ne, 2)) - .5) @ sc_to_cart.T
    walk_and_decompose(wf, R0, 0.4, 11, key, kappa)
      -> sample(wf, R0, nsweep=1500, sigma=0.4, rng=default_rng(11),
                snapshot_every=1, equil=400, target_acc=0.4)

That is ``Setup.liquid_R0()`` and SIGMA/SEED/SWEEPS/EQUIL, so nothing is
re-derived here -- only the entry point changes.

Also checked, cheaply and first: the two periodic-wrap helpers, on random points.

    python -u scripts/diag_liquid_walk_identity.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
PROJ = os.path.dirname(CLEAN)
for _p in (os.path.join(CLEAN, "src"), PROJ, HERE, os.path.join(PROJ, "_diag")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qhvmc_engine as legacy                                  # noqa: E402
from wigner_vmc.physics.geometry import wrap_to_supercell      # noqa: E402
from wigner_vmc.vmc.sampler import sample as clean_sample      # noqa: E402
import bench_rs75 as B                                         # noqa: E402
from diag_liquid_a8 import build_both, SIGMA, SEED, SWEEPS, EQUIL   # noqa: E402

TARGET_ACC = 0.4


def check_wrap(S, wfL):
    """`send_to_first_supercell` vs `wrap_to_supercell` -- the untested link."""
    print("=" * 78)
    print("the periodic wrap: legacy send_to_first_supercell vs clean wrap_to_supercell")
    print("=" * 78)
    rng = np.random.default_rng(2026)
    P = (rng.random((20000, 2)) - 0.5) * 40.0      # well outside one supercell
    a = legacy.send_to_first_supercell(P.copy(), wfL.sc_to_cart, wfL.cart_to_sc)[0]
    b = wrap_to_supercell(P.copy(), S.C, S.Ci)[0]
    d = np.abs(a - b).max()
    print(f"  20000 random points spanning +-20 (supercell side ~{np.linalg.norm(S.L1):.3f})")
    print(f"  max |legacy - clean| = {d:.3e}")
    print(f"  frames equal: sc_to_cart {np.abs(wfL.sc_to_cart - S.C).max():.3e}"
          f"   cart_to_sc {np.abs(wfL.cart_to_sc - S.Ci).max():.3e}")
    print(f"  wrap AGREES" if d < 1e-12 else "  wrap DIFFERS  <-- first divergence")
    print()
    return d


def main():
    S, wfL, wfC = build_both()
    dwrap = check_wrap(S, wfL)

    R0 = S.liquid_R0()
    fz = (np.random.default_rng(3).random((S.ne, 2)) - 0.5) @ S.C.T
    print(f"R0 == the frozen formula: {np.abs(R0 - fz).max():.3e}")
    print()

    print("=" * 78)
    print(f"the REAL samplers, frozen protocol: {SWEEPS} sweeps, equil {EQUIL}, "
          f"sigma {SIGMA}, seed {SEED}, target_acc {TARGET_ACC}")
    print("=" * 78)
    t0 = time.time()
    snL, sigL, accL = legacy.sample(wfL, R0, nsweep=SWEEPS, sigma=SIGMA,
                                    rng=np.random.default_rng(SEED),
                                    snapshot_every=1, equil=EQUIL,
                                    target_acc=TARGET_ACC)
    snC, sigC, accC = clean_sample(wfC, R0, nsweep=SWEEPS, sigma=SIGMA,
                                   rng=np.random.default_rng(SEED),
                                   snapshot_every=1, equil=EQUIL,
                                   target_acc=TARGET_ACC)
    print(f"  ({time.time()-t0:.0f}s)\n")
    print(f"  {'':<10}{'legacy':>16}{'clean':>16}{'delta':>14}")
    print("  " + "-" * 56)
    for nm, a, b in (("n", len(snL), len(snC)), ("sigma", sigL, sigC),
                     ("acc", accL, accC)):
        print(f"  {nm:<10}{a:>16.9f}{b:>16.9f}{b - a:>+14.3e}")

    A = np.asarray(snL)
    Bc = np.asarray(snC)
    print(f"\n  snapshot arrays  legacy {A.shape}  clean {Bc.shape}")
    if A.shape == Bc.shape:
        dd = np.abs(A - Bc)
        per = dd.reshape(len(A), -1).max(axis=1)
        ndiff = int((per > 0).sum())
        print(f"  snapshots differing at all : {ndiff} / {len(A)}")
        print(f"  max |dR| over all snapshots: {dd.max():.3e}")
        if ndiff:
            k = int(np.argmax(per > 0))
            print(f"  FIRST differing snapshot   : {k} "
                  f"(= sweep {k + EQUIL} of {SWEEPS})")
            print(f"    max |dR| there           : {per[k]:.3e}")

    print()
    ref = None
    try:
        import json
        ref = json.load(open(os.path.join(PROJ, "_diag", "part1", "liq_rs75.json"),
                             encoding="utf-8"))
    except Exception as e:                                       # pragma: no cover
        print(f"  (frozen record not read: {e})")
    if ref is not None:
        print("  against the frozen record liq_rs75.json:")
        print(f"    {'':<10}{'run':>16}{'frozen':>16}{'delta':>14}")
        print("    " + "-" * 56)
        for nm, a in (("n", len(snL)), ("sigma", sigL), ("acc", accL)):
            f = ref[nm]
            print(f"    {nm:<10}{a:>16.9f}{f:>16.9f}{a - f:>+14.3e}")
    return 0 if dwrap < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
