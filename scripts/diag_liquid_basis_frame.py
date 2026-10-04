"""Blocker A, second divergence: which matrix does `LandauLevelBasis` want?

`diag_liquid_walk_identity.py` shows the clean sampler is BIT-IDENTICAL to the
frozen legacy sampler over the whole 1500-sweep walk -- and that neither of them
reproduces the frozen record's `sigma` (0.616647 vs 0.650729).  Since R0, seed,
sigma0, sweeps, equil and target_acc all match, and the Jastrow `c` matches, the
only remaining input is the WAVEFUNCTION.

`Setup.basis()` passes `self.C = column_stack([L1, L2])`, the SUPERCELL matrix.
The notebook passes `CToCart`:

    cell 10 of the LL-rotation notebook
        CToCart = np.column_stack([A1, A2]);  cartToC = np.linalg.inv(CToCart)

`A1, A2` are the PRIMITIVE vectors (`A1 = L1/N1`).  `LandauLevelBasis` uses that
pair for exactly one thing -- `send_to_first_cell` folds `r - k x zhat` into the
first cell -- so the two matrices give different orbitals, and the notebook even
generates its own test points as `uniform(-0.5, 0.5) @ CToCart`, i.e. inside the
PRIMITIVE cell, which is only consistent with the primitive matrix.

Note the wavefunction's own frame is unaffected: the notebook passes
`sc_to_cart` (supercell) to `LLRotationWavefunction`, and so does `Setup`.

Three checks, cheapest first:

    1. the orbitals at fixed r, both matrices
    2. the local energy at fixed R, both wavefunctions
    3. the walk -- does the notebook's matrix reproduce the frozen sigma?

    python -u scripts/diag_liquid_basis_frame.py
"""
from __future__ import annotations

import json
import os
import sys

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

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
REF = os.path.join(PROJ, "_diag", "part1", "liq_rs75.json")
NJ = 5


def main():
    raw = json.load(open(INIT, encoding="utf-8"))
    ref = json.load(open(REF, encoding="utf-8"))
    S, wfL, wfC = build_both()
    c = np.asarray(raw["c_liquid"], float)
    v0 = np.zeros((S.nk, S.n_band_B - 1), complex)

    ai, ac = legacy.circular_lattice(float(raw["LUMAX_LL"]), S.A1, S.A2)
    # THE notebook's frame pair, verbatim: cell 10
    CToCart = np.column_stack([S.A1, S.A2])
    cartToC = np.linalg.inv(CToCart)

    print("=" * 78)
    print("the two matrices handed to LandauLevelBasis")
    print("=" * 78)
    print(f"  notebook CToCart = column_stack([A1, A2])   |A1|={np.linalg.norm(S.A1):.6f}"
          f"  |A2|={np.linalg.norm(S.A2):.6f}   PRIMITIVE")
    print(f"  Setup.C          = column_stack([L1, L2])   |L1|={np.linalg.norm(S.L1):.6f}"
          f"  |L2|={np.linalg.norm(S.L2):.6f}   SUPERCELL")
    print(f"  ratio L/A = {np.linalg.norm(S.L1)/np.linalg.norm(S.A1):.4f}")
    print()

    b_right = legacy.LandauLevelBasis(S.mesh, 2, ai, ac, CToCart, cartToC)
    b_wrong = legacy.LandauLevelBasis(S.mesh, 2, ai, ac, S.C, S.Ci)

    print("-- 1. orbitals psi_{k,n}(r), notebook matrix vs Setup's matrix --")
    rng = np.random.default_rng(2026)
    worst = 0.0
    for t in range(5):
        r = (rng.uniform(-0.5, 0.5, 2) @ CToCart) + t * S.L1
        a = b_right.orbitals(r)
        b = b_wrong.orbitals(r)
        d = np.abs(a - b).max()
        scale = max(np.abs(a).max(), 1e-300)
        worst = max(worst, d / scale)
        print(f"    r={np.round(r, 4)}  |dn|={d:.4e}  rel={d/scale:.4e}")
    print(f"  worst relative orbital difference = {worst:.4e}")
    print()

    # -- 2. the two wavefunctions, at the frozen R0 ------------------------
    gam = np.linalg.norm(S.L1) / np.sqrt(2) / np.pi * S.kappa / 3
    HAM = legacy.CoulombEwald(S.ne, S.L1, S.L2, S.G1, S.G2)

    def wf_with(lat_to_cart, cart_to_lat):
        bl = legacy.LandauLevelBasis(S.mesh, 2, ai, ac, lat_to_cart, cart_to_lat)
        orb = legacy_ll.LLRotatedOrbitals(bl, 2, v0)
        J = legacy.SinSplineJastrow(c, S.G1, S.G2, gam)
        return legacy_ll.LLRotationWavefunction(orb, J, S.ne, S.C,
                                                kappa=S.kappa, ham=HAM)

    wf_notebook = wf_with(CToCart, cartToC)          # the notebook's wiring
    wf_setup = wf_with(S.C, S.Ci)                    # Setup.basis()'s wiring

    print("-- 2. local energy at the frozen R0 (TOTALS) --")
    R0 = S.liquid_R0()
    for tag, w in (("notebook matrix", wf_notebook), ("Setup.C", wf_setup)):
        st = w.build(R0)
        T, V, E = w.local_energy(st)
        print(f"    {tag:<16} T = {T.real:>16.6f}   V = {V.real:>12.6f}")
    print()

    # -- 3. the walk: does the notebook wiring reproduce the frozen sigma? --
    print("=" * 78)
    print(f"-- 3. walk with the NOTEBOOK wiring, frozen protocol "
          f"({SWEEPS} sweeps, equil {EQUIL}, sigma {SIGMA}, seed {SEED}) --")
    print("=" * 78)
    sn, sig, acc = legacy.sample(wf_notebook, R0, nsweep=SWEEPS, sigma=SIGMA,
                                 rng=np.random.default_rng(SEED),
                                 snapshot_every=1, equil=EQUIL, target_acc=0.4)
    print(f"  {'':<12}{'this run':>16}{'frozen':>16}{'delta':>14}")
    print("  " + "-" * 60)
    for nm, a in (("n", len(sn)), ("sigma", sig), ("acc", acc)):
        f = ref[nm]
        print(f"  {nm:<12}{a:>16.9f}{f:>16.9f}{a - f:>+14.3e}")
    ok = abs(sig - ref["sigma"]) < 1e-12 and len(sn) == ref["n"]
    print()
    print("  VERDICT: the notebook's matrix pair reproduces the frozen sigma"
          if ok else
          "  VERDICT: still not the frozen sigma -- look further")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
