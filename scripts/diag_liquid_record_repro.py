"""Blocker A: does the NOTEBOOK-wired walk reproduce the frozen record exactly?

`diag_liquid_basis_frame.py` showed that handing `LandauLevelBasis` the notebook's
frame pair and sampling reproduces the frozen `sigma` (0.650729310) and `acc`
(0.414000000) with delta 0.000e+00, while the in-tree production wiring gives
0.616647362 / 0.412944444.

Matching `sigma` and `acc` over 54,000 accept/reject decisions cannot be a
coincidence, but a trajectory match is only half of A8.  This completes it: the
same walk, pushed through `walk_and_decompose`'s estimator exactly
(`_diag/part1_scan.py:287-320`), against the frozen record's T, V, E and errors.

Two wirings are walked back to back so the table is like-for-like:

    notebook   LandauLevelBasis(..., CToCart, cartToC)   CToCart = [A1, A2]  PRIMITIVE
    in-tree    LandauLevelBasis(..., Setup.C, Setup.Ci)  Setup.C = [L1, L2]  SUPERCELL

Nothing is written back to the frozen tree; this is read-only.

    python -u scripts/diag_liquid_record_repro.py
"""
from __future__ import annotations

import json
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
import qhvmc_engine_llrot as legacy_ll                         # noqa: E402
import bench_rs75 as B                                         # noqa: E402
from diag_liquid_a8 import build_both, SIGMA, SEED, SWEEPS, EQUIL   # noqa: E402
from mech import kinetic_parts, tau_int as legacy_tau_int      # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
REF = os.path.join(PROJ, "_diag", "part1", "liq_rs75.json")


def walk_and_decompose(wf, R0, label):
    """`_diag/part1_scan.py:287`, verbatim in the parts that touch the numbers."""
    snaps, sig, acc = legacy.sample(wf, R0, nsweep=SWEEPS, sigma=SIGMA,
                                    rng=np.random.default_rng(SEED),
                                    snapshot_every=1, equil=EQUIL,
                                    target_acc=0.4)
    n = len(snaps)
    T = np.empty(n)
    V = np.empty(n)
    for s, S in enumerate(snaps):
        a, b, c = kinetic_parts(wf, wf.build(S))
        T[s] = a + b + c
        V[s] = wf.ham.energy(S).real
    E = T + wf.kappa * V
    out = dict(label=label, n=n, sigma=sig, acc=acc)
    for nm, arr in (("T", T), ("V", V), ("E", E)):
        ti = legacy_tau_int(arr)
        out[nm] = float(arr.mean())
        out[nm + "_err"] = float(arr.std(ddof=1) / np.sqrt(n) * np.sqrt(ti))
        out[nm + "_err_naive"] = float(arr.std(ddof=1) / np.sqrt(n))
        out[nm + "_tau"] = float(ti)
    return out


def main():
    raw = json.load(open(INIT, encoding="utf-8"))
    ref = json.load(open(REF, encoding="utf-8"))
    S, _wfL, _wfC = build_both()
    c = np.asarray(raw["c_liquid"], float)
    v0 = np.zeros((S.nk, S.n_band_B - 1), complex)
    ai, ac = legacy.circular_lattice(float(raw["LUMAX_LL"]), S.A1, S.A2)
    gam = np.linalg.norm(S.L1) / np.sqrt(2) / np.pi * S.kappa / 3
    HAM = legacy.CoulombEwald(S.ne, S.L1, S.L2, S.G1, S.G2)
    R0 = S.liquid_R0()
    ne = S.ne
    kappa = S.kappa

    def build(lat_to_cart, cart_to_lat):
        bl = legacy.LandauLevelBasis(S.mesh, 2, ai, ac, lat_to_cart, cart_to_lat)
        orb = legacy_ll.LLRotatedOrbitals(bl, 2, v0)
        J = legacy.SinSplineJastrow(c, S.G1, S.G2, gam)
        return legacy_ll.LLRotationWavefunction(orb, J, ne, S.C,
                                                kappa=kappa, ham=HAM)

    CToCart = np.column_stack([S.A1, S.A2])
    runs = {}
    for tag, l2c, c2l in (("notebook", CToCart, np.linalg.inv(CToCart)),
                          ("in-tree ", S.C, S.Ci)):
        print(f"walking the {tag} wiring ...", flush=True)
        t0 = time.time()
        runs[tag] = walk_and_decompose(build(l2c, c2l), R0, tag)
        print(f"  done in {time.time()-t0:.0f}s  sigma={runs[tag]['sigma']:.9f}\n",
              flush=True)

    print("=" * 92)
    print(f"r_s=75 liquid, frozen protocol "
          f"({SWEEPS} sweeps, equil {EQUIL}, sigma {SIGMA}, seed {SEED}), PER PARTICLE")
    print("=" * 92)
    hdr = (f"  {'quantity':<9}{'frozen record':>16}{'notebook wiring':>18}"
           f"{'in-tree wiring':>17}{'delta(nb)':>13}{'delta(tree)':>13}")
    print(hdr)
    print("  " + "-" * 86)
    for nm in ("T", "V", "E"):
        f = ref[nm] / ne
        a = runs["notebook"][nm] / ne
        b = runs["in-tree "][nm] / ne
        print(f"  {nm + '/N':<9}{f:>16.9f}{a:>18.9f}{b:>17.9f}"
              f"{a - f:>+13.2e}{b - f:>+13.2e}")

    print()
    print(f"  {'E_err/N':<9}{ref['E_err']/ne:>16.9f}"
          f"{runs['notebook']['E_err']/ne:>18.9f}"
          f"{runs['in-tree ']['E_err']/ne:>17.9f}")
    print(f"  {'T_err/N':<9}{ref['T_err']/ne:>16.9f}"
          f"{runs['notebook']['T_err']/ne:>18.9f}"
          f"{runs['in-tree ']['T_err']/ne:>17.9f}")
    print(f"  {'V_err/N':<9}{ref['V_err']/ne:>16.9f}"
          f"{runs['notebook']['V_err']/ne:>18.9f}"
          f"{runs['in-tree ']['V_err']/ne:>17.9f}")
    print()
    print(f"  {'tau_E':<9}{ref['E_tau']:>16.9f}"
          f"{runs['notebook']['E_tau']:>18.9f}"
          f"{runs['in-tree ']['E_tau']:>17.9f}")
    print(f"  {'n':<9}{ref['n']:>16d}{runs['notebook']['n']:>18d}"
          f"{runs['in-tree ']['n']:>17d}")
    print(f"  {'sigma':<9}{ref['sigma']:>16.9f}"
          f"{runs['notebook']['sigma']:>18.9f}"
          f"{runs['in-tree ']['sigma']:>17.9f}")

    print()
    print("  z = delta / frozen error, where meaningful")
    for nm in ("T", "V", "E"):
        f = ref[nm] / ne
        e = ref[nm + "_err"] / ne
        a = runs["notebook"][nm] / ne
        b = runs["in-tree "][nm] / ne
        print(f"    {nm + '/N':<7} notebook {((a-f)/e):>+8.2f}   "
              f"in-tree {((b-f)/e):>+8.2f}")

    d = max(abs(runs["notebook"][nm] - ref[nm]) for nm in ("T", "V", "E"))
    print()
    print(f"  worst |notebook - frozen| over T, V, E = {d:.3e}")
    print("  VERDICT: the notebook wiring REPRODUCES the frozen record"
          if d < 1e-9 else
          "  VERDICT: the notebook wiring still differs -- look further")
    return 0 if d < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
