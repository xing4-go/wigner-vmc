"""A8 -- Blocker A's last gate: the r_s=75 liquid, trajectory and end to end.

Three parts, in increasing cost, each one a gate for the next:

  1. ESTIMATOR IDENTITY on one fixed configuration.  The frozen record's T comes
     from ``kinetic_parts`` (a recomputation of calc_kinetic_energy's three terms),
     NOT from ``local_energy``.  Those two must agree before any walk is compared,
     or a walk-level difference could be an artefact of the two conventions.

  2. TRAJECTORY IDENTITY, short.  Both samplers driven with the same seed, the same
     sigma and the same R0, for the first N attempts; proposals, acceptance ratios,
     accept/reject decisions, the configuration after every attempt and the final
     RNG state are compared.  This is the deterministic half of the claim.

  3. PRODUCTION WALK, the legacy protocol exactly: 1500 sweeps, equil 400,
     snapshot_every 1, sigma 0.4, seed 11 -- the same call the frozen
     ``_diag/part1/liq_rs75.json`` came from.  Compared to that record on
     T/N, V/N, E/N, E_err, and on the complex-T statistics the pause flagged.

Nothing is tuned.  If part 3 disagrees, that is a new first divergence to chase,
not a reason to add sweeps.

    python scripts/diag_liquid_a8.py
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

import qhvmc_engine as legacy            # noqa: E402
import qhvmc_engine_llrot as legacy_ll   # noqa: E402
from wigner_vmc.physics.landau_levels import LandauLevelBasis  # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr         # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw             # noqa: E402
from wigner_vmc.physics.geometry import wrap_to_supercell      # noqa: E402
from wigner_vmc.vmc.sampler import sample as clean_sample      # noqa: E402
import bench_rs75 as B                                         # noqa: E402
from mech import kinetic_parts, tau_int as legacy_tau_int      # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
REF = os.path.join(PROJ, "_diag", "part1", "liq_rs75.json")

SIGMA, SEED, SWEEPS, EQUIL = 0.4, 11, 1500, 400
NATT = 200          # attempts for the trajectory check


def build_both():
    raw = json.load(open(INIT, encoding="utf-8"))
    S = B.Setup(raw)
    c = np.asarray(raw["c_liquid"], float)
    v0 = np.zeros((S.nk, S.n_band_B - 1), complex)

    ai, ac = legacy.circular_lattice(float(raw["LUMAX_LL"]), S.A1, S.A2)
    HAM = legacy.CoulombEwald(S.ne, S.L1, S.L2, S.G1, S.G2)
    gam = np.linalg.norm(S.L1) / np.sqrt(2) / np.pi * S.kappa / 3
    # the notebook's folding cell (cell 10), not the simulation supercell: the
    # legacy counterpart must be built from the notebook contract, or the two
    # sides of every comparison below share the choice under test.
    ToCart = np.column_stack([S.A1, S.A2])
    llbL = legacy.LandauLevelBasis(S.mesh, 2, ai, ac, ToCart, np.linalg.inv(ToCart))
    wfL = legacy_ll.LLRotationWavefunction(
        legacy_ll.LLRotatedOrbitals(llbL, 2, v0),
        legacy.SinSplineJastrow(c, S.G1, S.G2, gam), S.ne, S.C,
        kappa=S.kappa, ham=HAM)
    wfC = S.maker(S.n_band_B)(S.theta(c, v0))
    return S, wfL, wfC


def part1(S, wfL, wfC):
    print("=" * 78)
    print("PART 1 -- estimator identity on one fixed configuration")
    print("=" * 78)
    R0 = S.liquid_R0()
    rows = []
    for tag, wf in (("legacy", wfL), ("clean", wfC)):
        st = wf.build(R0)
        T, V, E = wf.local_energy(st)
        td, tm, tj = kinetic_parts(wf, st)
        rows.append((tag, T, V, td, tm, tj))
        print(f"  {tag:<7} local_energy T = {T:>16.6f}  (Im {T.imag:+.6e})")
        print(f"  {'':<7} kinetic_parts  T_det={td:+.6f}  T_mix={tm:+.6f}  "
              f"T_J={tj:+.6f}   sum={td+tm+tj:+.6f}")
    Tl, Tc = rows[0][1], rows[1][1]
    print(f"\n  |T_legacy - T_clean|          = {abs(Tl-Tc):.3e}")
    print(f"  |local_energy T - parts sum|  legacy {abs(rows[0][1]-sum(rows[0][3:])):.3e}"
          f"   clean {abs(rows[1][1]-sum(rows[1][3:])):.3e}")
    print(f"  T_det (LLL, pi^2 = 1 band 0): legacy {rows[0][3]:.6f}"
          f"   clean {rows[1][3]:.6f}   (ne/2 = {S.ne/2})")
    print()


def rollout(wf, S, seed, natt):
    """The first ``natt`` attempts of ``sample``, verbatim, with no adaptation.

    With natt < 1000 the sigma-adaptation block never fires, and
    reset_period = 100*ne = 3600 > natt, so no rebuild happens either: these ARE
    the sampler's first natt attempts, not a re-implementation of them.
    """
    rng = np.random.default_rng(seed)
    st = wf.build(S.liquid_R0())
    rnew = np.empty((natt, 2))
    accr = np.empty(natt, complex)
    ok = np.empty(natt, bool)
    Rtr = np.empty((natt, S.ne, 2))
    sigma = SIGMA
    for t in range(natt):
        i = t % S.ne
        r_new = wrap_to_supercell(
            st["R"][i] + sigma * rng.standard_normal(2), wf.sc_to_cart,
            wf.cart_to_sc)[0]
        acc, info = wf.move_ratio(st, i, r_new)
        u = rng.random()
        good = u < acc
        if good:
            wf.accept_move(st, i, r_new, info)
        rnew[t] = r_new
        accr[t] = acc
        ok[t] = good
        Rtr[t] = st["R"]
    return rnew, accr, ok, Rtr, rng.bit_generator.state


def part2(S, wfL, wfC):
    print("=" * 78)
    print(f"PART 2 -- trajectory identity, first {NATT} attempts, seed {SEED}")
    print("=" * 78)
    rl, al, ol, Rl, sl = rollout(wfL, S, SEED, NATT)
    rc, ac, oc, Rc, sc = rollout(wfC, S, SEED, NATT)
    nmis = int((ol != oc).sum())
    print(f"  decision mismatches      : {nmis} / {NATT}")
    print(f"  max |r_new_legacy - clean| = {np.abs(rl-rc).max():.3e}")
    print(f"  max |acc_legacy - clean|   = {np.abs(al-ac).max():.3e}")
    print(f"  max |R_legacy - clean|     = {np.abs(Rl-Rc).max():.3e}")
    print(f"  RNG state identical        : {sl == sc}")
    print(f"  min acc seen {np.abs(al).min():.6e}   max {np.abs(al).max():.6f}")
    print()
    return nmis


def part3(S, wfC):
    print("=" * 78)
    print(f"PART 3 -- production walk, legacy protocol "
          f"({SWEEPS} sweeps, equil {EQUIL}, sigma {SIGMA}, seed {SEED})")
    print("=" * 78)
    ne = S.ne
    ref = json.load(open(REF, encoding="utf-8"))

    # (a) the production path itself -- the authoritative numbers
    rec = B.production_walk(wfC, S.liquid_R0(), SWEEPS, EQUIL, SIGMA, SEED,
                            "liq_rs75")

    # (b) the same walk again, keeping the per-snapshot series.  Same seed, so the
    #     trajectory is identical and the means must reproduce (a) exactly; that
    #     agreement is the internal check that this inline estimator copies
    #     production_walk rather than paraphrasing it.
    snaps, sig, acc = clean_sample(wfC, S.liquid_R0(), nsweep=SWEEPS,
                                   sigma=SIGMA, rng=np.random.default_rng(SEED),
                                   snapshot_every=1, equil=EQUIL, target_acc=0.4)
    T = np.empty(len(snaps))
    V = np.empty(len(snaps))
    ImT = np.empty(len(snaps))
    for k, Sn in enumerate(snaps):
        t, v, _ = wfC.local_energy(wfC.build(Sn))
        T[k], V[k], ImT[k] = t.real, v.real, t.imag
    E = T + wfC.kappa * V

    print(f"  clean   n={rec['n']:<5} acc={rec['acc']:.4f} sigma={rec['sigma']:.6f}")
    print(f"  legacy  n={ref['n']:<5} acc={ref['acc']:.4f} sigma={ref['sigma']:.6f}")
    for k in ("T", "V", "E"):
        d = abs(rec[k] - (T if k == "T" else V if k == "V" else E).mean() / ne)
        assert d < 1e-12, f"inline {k} disagrees with production_walk by {d:.3e}"
    print("  (inline series reproduces production_walk exactly on T, V, E)")

    def leg(x):
        ti = legacy_tau_int(x)
        return x.std(ddof=1) / np.sqrt(len(x)) * np.sqrt(ti) / ne, ti

    print()
    print(f"  {'quantity':<10} {'clean':>14} {'legacy':>14} {'delta':>12} {'z':>8}")
    print("  " + "-" * 62)
    for name, cv, lv, lerr in (
            ("T/N", T.mean() / ne, ref["T"] / ne, ref["T_err"] / ne),
            ("V/N", V.mean() / ne, ref["V"] / ne, ref["V_err"] / ne),
            ("E/N", E.mean() / ne, ref["E"] / ne, ref["E_err"] / ne)):
        d = cv - lv
        z = d / lerr if lerr else float("nan")
        print(f"  {name:<10} {cv:>14.6f} {lv:>14.6f} {d:>+12.6f} {z:>8.2f}")

    ce, cti = leg(E)
    cte, ctti = leg(T)
    print()
    print(f"  E_err  clean {ce:.6f}  (production {rec['E_err']:.6f})  "
          f"legacy {ref['E_err']:.6f}")
    print(f"  T_err  clean {cte:.6f}   legacy {ref['T_err']:.6f}")
    print(f"  tau_E  clean {cti:.4f}  legacy {ref['E_tau']:.4f}   "
          f"tau_T clean {ctti:.4f}  legacy {ref['T_tau']:.4f}")
    print(f"  sigma final  clean {rec['sigma']:.6f}  legacy {ref['sigma']:.6f}")

    print()
    print("  the pause's two anomalies, over this walk:")
    print(f"    (1) T/N sign     clean {T.mean()/ne:+.6f}   legacy "
          f"{ref['T']/ne:+.6f}   -> {'SAME SIGN' if np.sign(T.mean())==np.sign(ref['T']) else 'SIGN DIFFERS'}")
    print(f"    (2) complex T    mean Im(T) = {ImT.mean():>12.6f}   rms Im(T) = "
          f"{np.sqrt((ImT**2).mean()):>12.6f}")
    print("        (legacy stores T already .real, from kinetic_parts, so the record"
          " has no Im(T);")
    print("         PART 1's fixed-configuration comparison is the legacy-side check.)")
    return rec, (T, V, E, ImT), ref


def main():
    S, wfL, wfC = build_both()
    part1(S, wfL, wfC)
    nmis = part2(S, wfL, wfC)
    rec, series, ref = part3(S, wfC)
    print()
    print("=" * 78)
    print(f"trajectory decision mismatches in the first {NATT} attempts: {nmis}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
