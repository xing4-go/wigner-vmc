"""Stage 2E / Blocker A: same-configuration differential test, legacy vs clean.

NO SAMPLING.  One fixed configuration R, the r_s = 75 liquid wavefunction
(``make_ll_crystal(kappa, 2)`` at ``v = 0``, ``c = c_liquid``), compared layer by
layer in the order the diagnosis was specified:

    orbitals / D , D_inv , pi_columns , T_complex , Re(T) , Im(T) , V , E_loc

Three clean variants are built, to separate a *kernel* error from a *wiring*
error and to keep the pre-fix evidence reproducible:

  production   -- ``bench_rs75.Setup.maker(2)`` -- the driver's own path
  pre-fix      -- the same class, wired to the Gaussian-overlap set (the bug)
  rewired      -- the same class, wired to the notebook's LL set (the target)

The legacy side is built from the FROZEN modules (``qhvmc_engine`` /
``qhvmc_engine_llrot``) with the notebook's own lattice-vector sets AND the
notebook's own folding cell (``CToCart = column_stack([A1, A2])``, cell 10);
nothing is imported from the notebook and nothing is written back to the frozen
tree.

Both of those choices are load-bearing.  An earlier revision gave the legacy
oracle ``Setup.C`` -- the simulation supercell -- which is what the driver was
wrongly passing at the time, so expected and actual shared the disputed choice
and this whole file agreed with the bug (Blocker A, second divergence).  The
legacy oracle is now built from the notebook contract and from nothing else; the
``clean_wf`` variants take the frame explicitly for the same reason.

    python scripts/diag_liquid_diff.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
PROJ = os.path.dirname(CLEAN)

for _p in (os.path.join(CLEAN, "src"), PROJ, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qhvmc_engine as legacy            # noqa: E402
import qhvmc_engine_llrot as legacy_ll   # noqa: E402
from wigner_vmc.physics import landau_levels as llb     # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr  # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw      # noqa: E402
import bench_rs75 as B                                  # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")

NAMES = ["Ops (padded orbitals)", "D  = psi_k(r_b)", "D_inv", "pi_x columns",
         "pi^2 columns", "T  (complex)", "Re(T)", "Im(T)", "V",
         "E_loc = T + kappa V"]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a - b).max()
    s = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    return d, d / s


def main():
    raw = json.load(open(INIT, encoding="utf-8"))
    S = B.Setup(raw)
    L1, L2, G1, G2 = S.L1, S.L2, S.G1, S.G2
    mesh, sc, C, Ci = S.mesh, S.C, S.C, S.Ci
    kappa, ne, nk = S.kappa, S.ne, S.nk
    c = np.asarray(raw["c_liquid"], float)
    v0 = np.zeros((nk, S.n_band_B - 1), complex)

    ai_leg, ac_leg = legacy.circular_lattice(float(raw["LUMAX_LL"]), S.A1, S.A2)
    ai_ov, ac_ov = S.ov_ai, S.ov_ac

    print("lattice-vector sets")
    print(f"  LL  (notebook _aiB)  n={len(ai_leg):>4}  "
          f"max|a|={np.linalg.norm(ac_leg,axis=1).max():.6f}"
          "   circular_lattice(LUMAX_LL, A1, A2)  PRIMITIVE")
    print(f"  overlap (_lints8)    n={len(ai_ov):>4}  "
          f"max|a|={np.linalg.norm(ac_ov,axis=1).max():.6f}"
          "   circular_lattice(LUMAX, L1, L2)     SUPERCELL")
    print(f"  Setup.ll_ac used by production: n={len(S.ll_ac)}  "
          f"n={len(S.ov_ac)} for the overlap set")
    print()

    # ---- fixed configuration set (A2) ------------------------------------
    rng = np.random.default_rng(3)
    R0 = S.liquid_R0()
    pert = np.random.default_rng(20_261_002)
    CONFIGS = [("R0", R0)]
    for i in range(4):
        CONFIGS.append((f"R0+pert{i}",
                        (R0 + 0.3 * pert.standard_normal((ne, 2))) @ Ci.T @ C.T))
    R = R0

    # ---- the four wavefunctions ------------------------------------------
    HAM = legacy.CoulombEwald(ne, L1, L2, G1, G2)
    gam = np.linalg.norm(L1) / np.sqrt(2) / np.pi * kappa / 3
    # the notebook's folding cell -- cell 10 -- NOT the supercell `sc`
    ToCart = np.column_stack([S.A1, S.A2])
    cartToC = np.linalg.inv(ToCart)
    llbL = legacy.LandauLevelBasis(mesh, 2, ai_leg, ac_leg, ToCart, cartToC)
    wfL = legacy_ll.LLRotationWavefunction(
        legacy_ll.LLRotatedOrbitals(llbL, 2, v0),
        legacy.SinSplineJastrow(c, G1, G2, gam), ne, sc, kappa=kappa, ham=HAM)

    wfP = S.maker(S.n_band_B)(S.theta(c, v0))          # the driver's own path

    def clean_wf(ai, ac):                              # explicit wiring
        b = llb.LandauLevelBasis(mesh, 1, ai, ac, ToCart, cartToC)
        o = lr.LLRotatedOrbitals(b, 2, v0)
        j = jw.SinSplineJastrow(c, G1, G2, jw.cusp_gamma(kappa, L1))
        return lr.LLRotationWavefunction(o, j, ne, C, kappa=kappa, ham=HAM)

    wfO = clean_wf(ai_ov, ac_ov)                       # pre-fix wiring (the bug)
    wfX = clean_wf(ai_leg, ac_leg)                     # notebook LL set

    def layers(wf, R):
        st = wf.build(R)
        T, V, E = wf.local_energy(st)
        px, _, p2 = wf.pi_columns(st)
        return [st["Ops"], st["D"], st["D_inv"], px, p2, T, T.real, T.imag, V, E]

    # ---- A2: first divergence on each fixed configuration ----------------
    print(f"A2 -- first divergent layer vs legacy, {len(CONFIGS)} FIXED "
          "configurations (no sampling):")
    print(f"{'config':<12} {'production':<26} {'pre-fix wiring':<26} "
          f"{'rewired (max)':>14}")
    print("-" * 80)
    for label, RR in CONFIGS:
        L = layers(wfL, RR)
        first = {}
        for tag, wf in (("production", wfP), ("pre-fix wiring", wfO),
                        ("rewired", wfX)):
            got = layers(wf, RR)
            first[tag] = next((n for n, a, b in zip(NAMES, L, got)
                               if rel(a, b)[1] > 1e-10), None)
            if tag == "rewired":
                worst = max(rel(a, b)[0] for a, b in zip(L, got))
        print(f"{label:<12} {str(first['production']):<26} "
              f"{str(first['pre-fix wiring']):<26} {worst:>14.3e}")

    # ---- the layer table on R0 -------------------------------------------
    L = layers(wfL, R0)
    P = layers(wfP, R0)
    O = layers(wfO, R0)
    X = layers(wfX, R0)
    print()
    print(f"{'layer (R0)':<24} {'|legacy-production|':>20} {'rel':>9}   "
          f"{'|legacy-prefix|':>16} {'rel':>9}")
    print("-" * 84)
    for name, a, p, o in zip(NAMES, L, P, O):
        d1, r1 = rel(a, p)
        d2, r2 = rel(a, o)
        print(f"{name:<24} {d1:>20.6e} {r1:>9.2e}   {d2:>16.6e} {r2:>9.2e}")

    print()
    print("scalars on R0 (TOTALS, not per particle)")
    for tag, w in (("legacy", L), ("production", P), ("pre-fix", O),
                   ("rewired", X)):
        # w[5] is the complex T, w[7] is ALREADY T.imag -- taking .imag of a real
        # array silently returns zeros, which is exactly the kind of restated
        # number this diagnostic exists to avoid.
        print(f"  {tag:<11} T = {w[5]:>16.6f}   V = {w[8]:>10.6f}"
              f"   Im(T) = {w[7].max():>13.6e}")

    worst_p = max(rel(a, b)[0] for a, b in zip(L, P))
    worst_x = max(rel(a, b)[0] for a, b in zip(L, X))
    print()
    print(f"worst |legacy - production| over all layers = {worst_p:.3e}")
    print(f"worst |legacy - rewired|    over all layers = {worst_x:.3e}")
    verdict = "MATCHES" if worst_p < 1e-9 else "MISMATCHES"
    print(f"VERDICT: the production path {verdict} the frozen legacy engine at "
          "fixed configuration")
    return 0 if worst_p < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
