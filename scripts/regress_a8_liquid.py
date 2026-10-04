"""A8, re-run after the folding-frame fix: the formal r_s=75 liquid regression.

Blocker A's first A8 run FAILED, and the second divergence it found was the cell
matrix handed to `LandauLevelBasis`: `Setup.basis()` folded into the SIMULATION
SUPERCELL where the notebook folds into the PRIMITIVE cell.  `bench_rs75.py` now
carries both frames under their own names, and this is the end-to-end re-run of
the SAME gate, on the PRODUCTION path -- `Setup.maker`, not a hand-built
wavefunction -- with nothing else changed.

The protocol is the frozen one, verbatim (`_diag/part1_scan.py`):
1500 sweeps, equil 400, snapshot_every 1, sigma 0.4, seed 11, target_acc 0.4,
R0 = rng(3) uniform in the supercell, and the campaign's rounded kappa --
`Setup.legacy_regression` supplies 53.033, which is what the published record was
built with.  The clean PHYSICS convention is 53.03300858899106 and is reported
alongside; the two are different objects and the difference is 2.3e-4 in E.

Five parts, each a gate:

    1. trajectory identity -- the two SAMPLERS, elementwise over all snapshots
    2. raw E sequence identity -- the two ESTIMATORS, per snapshot
    3. the production path against the frozen record
    4. the error bar, which is a separate question from parts 1-3
    5. verdict

Nothing is written to the frozen tree.  No sweeps are added.

    python -u scripts/regress_a8_liquid.py
"""
from __future__ import annotations

import argparse
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
from wigner_vmc.analysis import statistics as st               # noqa: E402
from wigner_vmc.vmc.sampler import sample as clean_sample      # noqa: E402
import bench_rs75 as B                                         # noqa: E402
from mech import kinetic_parts, tau_int as legacy_tau_int      # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
REF = os.path.join(PROJ, "_diag", "part1", "liq_rs75.json")
# Parts 1-2 (two walks + two per-snapshot decompositions) are ~11 of the ~19
# minutes.  The cache is keyed to this exact protocol, so it is only valid while
# SIGMA/SEED/SWEEPS/EQUIL/TARGET below are unchanged.
SERIES = os.path.join(CLEAN, "logs", "regress_a8_series.npz")

SIGMA, SEED, SWEEPS, EQUIL, TARGET = 0.4, 11, 1500, 400, 0.4
KS = (2.0, 3.0, 5.0, 10.0)
BLOCKS = (5, 10, 20, 40, 80)


def build_pair(raw):
    """The production clean wavefunction and its legacy counterpart.

    Both now use the NOTEBOOK's frame pair for the basis -- `column_stack([A1,
    A2])` -- and the SUPERCELL for the wavefunction itself.  The legacy side is
    built from the raw initial conditions by hand, so it cannot inherit anything
    from `Setup`; that was the coverage hole the first fix left open.
    """
    S = B.Setup.legacy_regression(raw)          # the campaign's kappa = 53.033
    c = np.asarray(raw["c_liquid"], float)
    v0 = np.zeros((S.nk, S.n_band_B - 1), complex)

    ToCart = np.column_stack([S.A1, S.A2])      # cell 10
    ai, ac = legacy.circular_lattice(float(raw["LUMAX_LL"]), S.A1, S.A2)
    gam = np.linalg.norm(S.L1) / np.sqrt(2.0) / np.pi * S.kappa / 3.0
    HAM = legacy.CoulombEwald(S.ne, S.L1, S.L2, S.G1, S.G2)
    wfL = legacy_ll.LLRotationWavefunction(
        legacy_ll.LLRotatedOrbitals(
            legacy.LandauLevelBasis(S.mesh, 2, ai, ac, ToCart,
                                    np.linalg.inv(ToCart)), 2, v0),
        legacy.SinSplineJastrow(c, S.G1, S.G2, gam), S.ne, S.C,
        kappa=S.kappa, ham=HAM)

    wfC = S.maker(S.n_band_B)(S.theta(c, v0))   # the PRODUCTION path
    return S, wfL, wfC


def series_legacy(wf, snaps):
    """`_diag/part1_scan.py:287-320`'s decomposition, term for term."""
    n = len(snaps)
    T = np.empty(n)
    V = np.empty(n)
    for s, S in enumerate(snaps):
        a, b, c = kinetic_parts(wf, wf.build(S))
        T[s] = a + b + c
        V[s] = wf.ham.energy(S).real
    return T, V


def series_clean(wf, snaps):
    """The clean package's own estimator, per snapshot."""
    n = len(snaps)
    T = np.empty(n)
    V = np.empty(n)
    ImT = np.empty(n)
    for s, S in enumerate(snaps):
        t, v, _ = wf.local_energy(wf.build(S))
        T[s], V[s], ImT[s] = t.real, v.real, t.imag
    return T, V, ImT


def err_legacy(x):
    ti = legacy_tau_int(x)
    return float(x.std(ddof=1) / np.sqrt(len(x)) * np.sqrt(ti)), float(ti)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="A8 r_s=75 liquid regression, after the folding-frame fix")
    ap.add_argument("--reuse-series", action="store_true",
                    help="re-print parts 1-2 from the cached walk in "
                         f"{os.path.relpath(SERIES, CLEAN)} instead of walking "
                         "again (only valid for the frozen protocol above)")
    return ap.parse_args(argv)


def main(args):
    raw = json.load(open(INIT, encoding="utf-8"))
    ref = json.load(open(REF, encoding="utf-8"))
    S, wfL, wfC = build_pair(raw)
    R0 = S.liquid_R0()
    ne = S.ne
    kappa = S.kappa

    print("=" * 84)
    print("A8 -- r_s = 75 liquid, frozen protocol, PRODUCTION path, after the")
    print("folding-frame fix")
    print("=" * 84)
    print(f"  kappa  legacy-regression {kappa!r}   physical "
          f"{B.physical_kappa(S.rs)!r}")
    print(f"  basis frame  |A1| = {np.linalg.norm(S.prim_C[:, 0]):.6f}  (primitive)"
          f"   |L1| = {np.linalg.norm(S.C[:, 0]):.6f}  (supercell)")
    print(f"  protocol  {SWEEPS} sweeps, equil {EQUIL}, snapshot_every 1, "
          f"sigma {SIGMA}, seed {SEED}, target_acc {TARGET}")

    # -- 1 + 2. the two samplers, then the two estimators -------------------
    # The walks and per-snapshot decompositions dominate the runtime, so they are
    # cached.  `--reuse-series` re-prints the parts 1-2 table from the cache
    # instead of re-walking: useful when only the reporting below changed, which
    # is exactly how the mislabelled error-bar line was caught.
    print()
    print("-- 1. trajectory: legacy sample vs clean sample --")
    if args.reuse_series and os.path.exists(SERIES):
        z = np.load(SERIES)
        (TL, VL, TC, VC, ImT) = (z["TL"], z["VL"], z["TC"], z["VC"], z["ImT"])
        sigL, accL, sigC, accC = (float(z["sigL"]), float(z["accL"]),
                                  float(z["sigC"]), float(z["accC"]))
        nL, nC, n_diff, dRmax = (int(z["nL"]), int(z["nC"]),
                                 int(z["n_diff"]), float(z["dRmax"]))
        print(f"  [reused the cached series: {SERIES}]")
    else:
        t0 = time.time()
        snapL, sigL, accL = legacy.sample(wfL, R0, nsweep=SWEEPS, sigma=SIGMA,
                                          rng=np.random.default_rng(SEED),
                                          snapshot_every=1, equil=EQUIL,
                                          target_acc=TARGET)
        print(f"  legacy walk done in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        snapC, sigC, accC = clean_sample(wfC, R0, nsweep=SWEEPS, sigma=SIGMA,
                                         rng=np.random.default_rng(SEED),
                                         snapshot_every=1, equil=EQUIL,
                                         target_acc=TARGET)
        print(f"  clean  walk done in {time.time() - t0:.0f}s", flush=True)

        dR = (np.abs(np.asarray(snapL) - np.asarray(snapC))
              .reshape(len(snapL), -1).max(axis=1))
        nL, nC, n_diff, dRmax = len(snapL), len(snapC), int((dR > 0).sum()), dR.max()
        print()
        print("-- 2. raw energy sequences, per snapshot, same trajectory --")
        t0 = time.time()
        TL, VL = series_legacy(wfL, snapL)
        print(f"  legacy decomposition in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        TC, VC, ImT = series_clean(wfC, snapC)
        print(f"  clean  decomposition in {time.time() - t0:.0f}s", flush=True)
        np.savez(SERIES, TL=TL, VL=VL, TC=TC, VC=VC, ImT=ImT, sigL=sigL,
                 accL=accL, sigC=sigC, accC=accC, nL=nL, nC=nC,
                 n_diff=n_diff, dRmax=dRmax)
        print(f"  series cached to {SERIES}")

    print(f"  n      legacy {nL}  clean {nC}  record {ref['n']}")
    print(f"  sigma  legacy {sigL:.9f}  clean {sigC:.9f}  record {ref['sigma']:.9f}"
          f"   delta {sigC - ref['sigma']:+.3e}")
    print(f"  acc    legacy {accL:.9f}  clean {accC:.9f}  record {ref['acc']:.9f}"
          f"   delta {accC - ref['acc']:+.3e}")
    print(f"  snapshots that differ: {n_diff} / {nL}   max |dR| = {dRmax:.3e}")
    walk_ok = (n_diff == 0 and nC == ref["n"]
               and sigC == ref["sigma"] and accC == ref["acc"])

    EL = TL + kappa * VL
    EC = TC + kappa * VC

    print(f"  max |T_legacy - T_clean| = {np.abs(TL - TC).max():.3e}   (TOTALS,"
          f" scale {abs(TL).max():.1f})")
    print(f"  max |V_legacy - V_clean| = {np.abs(VL - VC).max():.3e}   (scale"
          f" {abs(VL).max():.1f})")
    print(f"  max |E_legacy - E_clean| = {np.abs(EL - EC).max():.3e}   (scale"
          f" {abs(EL).max():.1f})  relative "
          f"{np.abs(EL - EC).max() / np.abs(EL).max():.3e}")
    raw_e_identical = bool(np.abs(EL - EC).max() / np.abs(EL).max() < 1e-12)
    print(f"  RAW E SEQUENCE IDENTICAL: {raw_e_identical}")
    print(f"  mean Im(T) = {ImT.mean():+.3e}   rms Im(T) = "
          f"{np.sqrt((ImT ** 2).mean()):.3e}   (record stores T already .real)")

    # -- 3. the production path against the record -------------------------
    print()
    print("-- 3. the production path against the frozen record --")
    t0 = time.time()
    rec = B.production_walk(wfC, S.liquid_R0(), SWEEPS, EQUIL, SIGMA, SEED,
                            "liq_rs75")
    print(f"  production_walk in {time.time() - t0:.0f}s", flush=True)
    assert rec["kappa"] == ref["kappa"] and rec["kappa_mode"] == "legacy-regression", \
        f"provenance is wrong: {rec['kappa']!r} / {rec['kappa_mode']!r}"

    print(f"  {'quantity':<10}{'production':>16}{'record':>16}{'delta':>13}{'z':>8}")
    print("  " + "-" * 63)
    for nm, key, rerr in (("T/N", "T", "T_err"), ("V/N", "V", "V_err"),
                          ("E/N", "E", "E_err")):
        a = rec[key]
        b = ref[key] / ne
        e = ref[rerr] / ne
        print(f"  {nm:<10}{a:>16.9f}{b:>16.9f}{a - b:>+13.2e}{(a - b) / e:>8.2f}")
    d = max(abs(rec[k] - ref[k] / ne) for k in ("T", "V", "E"))
    print(f"  worst |production - record| over T/N, V/N, E/N = {d:.3e}")
    prod_ok = d < 1e-9

    # -- 4. the error bar --------------------------------------------------
    print()
    print("-- 4. the error bar: a separate question (Blocker A-errorbar) --")
    naive = float(EC.std(ddof=1) / np.sqrt(len(EC)))
    print(f"  E_err_naive   this walk {naive:.12f}   record {ref['E_err_naive']:.12f}"
          f"   delta {naive - ref['E_err_naive']:+.3e}")
    # `err_legacy` returns (error, tau) in that order.  Taking them the other way
    # round prints the error as a tau and then multiplies the naive error by its
    # square root, which looks plausible and is nonsense -- the same
    # read-the-wrong-variable family as the sign bugs.
    err_leg, ti_leg = err_legacy(EC)
    print(f"  tau_int       legacy rule (first rho<=0) {ti_leg:.6f}   record "
          f"{ref['E_tau']:.6f}   delta {ti_leg - ref['E_tau']:+.3e}")
    print(f"  E_err         legacy rule {err_leg:.12f}   record "
          f"{ref['E_err']:.12f}   delta {err_leg - ref['E_err']:+.3e}")
    ti_cl, win = st.tau_int(EC)
    print(f"  tau_int       clean  rule (Sokal c=5)   {ti_cl:.6f}  window {win}"
          f"   -> E_err {naive * np.sqrt(ti_cl):.12f}")
    print(f"  the two rules differ by {ti_cl / ti_leg:.3f}x on ONE sequence")
    print()
    print(f"  {'c':>6}{'tau':>12}{'E_err':>14}   {'block':>7}{'E_err':>14}")
    for c, blk in zip(KS, BLOCKS):
        tc, _ = st.tau_int(EC, c=float(c))
        print(f"  {c:>6.1f}{tc:>12.4f}{naive * np.sqrt(tc):>14.9f}   "
              f"{blk:>7d}{st.blocked(EC, blk):>14.9f}")
    print("  window truncation, sum_{t=1..W} rho_t with the legacy "
          "normalisation:")
    x = EC - EC.mean()
    var = np.dot(x, x) / len(x)
    row = []
    for W in (5, 10, 20, 40, 80, 200):
        s = sum(np.dot(x[:-t], x[t:]) / (len(x) * var) for t in range(1, W + 1))
        row.append(f"W={W}:{1 + 2 * s:.3f}")
    print("    " + "  ".join(row))
    print(f"  per particle: campaign rule {err_leg / ne:.9f}   record "
          f"{ref['E_err'] / ne:.9f}   ratio {err_leg / ref['E_err']:.6f}")
    # Both sides of THIS ratio must be per particle: `rec['E_err']` is already
    # per electron and `ref['E_err']` is a TOTAL, so dividing them directly
    # prints 0.0267 -- a number 36x too small, from the total-vs-per-particle
    # family (see the `hist[i]["E"]` trap in the analysis notes).
    print(f"                production  {rec['E_err']:.9f}   (production uses the "
          f"CLEAN rule; ratio to record "
          f"{rec['E_err'] / (ref['E_err'] / ne):.4f})")
    print("  The block curve has no plateau below 40 samples and the window sum "
          "turns over at W=40,")
    print("  so at n=1100 no estimator here is entitled to more than 2 decimals; "
          "the exactly-reproduced")
    print("  number is E_err_naive, which depends on the sequence alone and not "
          "on any window rule.")

    # -- 5. verdict --------------------------------------------------------
    print()
    print("=" * 84)
    print(f"  trajectory identical .................. {walk_ok}")
    print(f"  raw E sequence identical .............. {raw_e_identical}")
    print(f"  production matches the record (T,V,E) . {prod_ok}")
    print(f"  E_err_naive matches the record ........ "
          f"{abs(naive - ref['E_err_naive']) < 1e-15}")
    print(f"  E_err matches under the CAMPAIGN rule . "
          f"{abs(err_leg - ref['E_err']) < 1e-12}")
    ok = walk_ok and raw_e_identical and prod_ok
    print("  A8: PASS" if ok else "  A8: FAIL -- STOP, do not add sweeps")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
