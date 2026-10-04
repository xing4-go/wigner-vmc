"""A8, corrected wiring: does the CLEAN package now reproduce the frozen record?

`diag_liquid_basis_frame.py` shows the notebook's frame pair reproduces the
frozen sigma/acc exactly, and `diag_liquid_record_repro.py` shows T/N and V/N
follow (1.4e-10 and 0).  But both of those built the wavefunction from the LEGACY
engine.  This one builds it from the clean package -- the same objects `Setup`
would build if `basis()` passed the primitive matrix -- and drives it with the
clean sampler.  That is the end-to-end claim, not the kernel claim.

The frame matrix is the ONLY thing changed; `Setup` itself is untouched.

On E: the frozen record's `kappa` is `round(rs/sqrt(2), 4)` = 53.033
(`_diag/part1_scan.py:414`), while the clean tree carries the exact
53.03300858899106.  That alone moves E by 2.3e-04, so both are reported.

    python -u scripts/diag_liquid_clean_repro.py
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

from wigner_vmc.physics import landau_levels as llb            # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr         # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw             # noqa: E402
from wigner_vmc.vmc.sampler import sample as clean_sample      # noqa: E402
from wigner_vmc.physics.coulomb import CoulombEwald            # noqa: E402
import bench_rs75 as B                                         # noqa: E402
from mech import tau_int as legacy_tau_int                     # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
REF = os.path.join(PROJ, "_diag", "part1", "liq_rs75.json")
SIGMA, SEED, SWEEPS, EQUIL = 0.4, 11, 1500, 400


def main():
    raw = json.load(open(INIT, encoding="utf-8"))
    ref = json.load(open(REF, encoding="utf-8"))
    S = B.Setup(raw)
    c = np.asarray(raw["c_liquid"], float)
    v0 = np.zeros((S.nk, S.n_band_B - 1), complex)
    Cp = np.column_stack([S.A1, S.A2])            # the notebook's CToCart
    Cpi = np.linalg.inv(Cp)

    # the clean package, wired the notebook's way -- the one line under test
    b = llb.LandauLevelBasis(S.mesh, S.n_band_B - 1, S.ll_ai, S.ll_ac, Cp, Cpi)
    orb = lr.LLRotatedOrbitals(b, S.n_band_B, v0)
    wf = lr.LLRotationWavefunction(orb, S.jastrow(c), S.ne, S.C,
                                   kappa=S.kappa,
                                   ham=CoulombEwald(S.ne, S.L1, S.L2, S.G1, S.G2))
    R0 = S.liquid_R0()
    ne = S.ne

    t0 = time.time()
    snaps, sig, acc = clean_sample(wf, R0, nsweep=SWEEPS, sigma=SIGMA,
                                   rng=np.random.default_rng(SEED),
                                   snapshot_every=1, equil=EQUIL, target_acc=0.4)
    T = np.empty(len(snaps))
    V = np.empty(len(snaps))
    for k, Sn in enumerate(snaps):
        t, v, _ = wf.local_energy(wf.build(Sn))
        T[k], V[k] = t.real, v.real
    print(f"clean walk done in {time.time()-t0:.0f}s\n")

    print("=" * 84)
    print(f"r_s=75 liquid, clean package, notebook frame pair, frozen protocol")
    print("=" * 84)
    print(f"  n      clean {len(snaps):<6}  frozen {ref['n']:<6}  delta "
          f"{len(snaps) - ref['n']:+d}")
    print(f"  sigma  clean {sig:.9f}  frozen {ref['sigma']:.9f}  delta "
          f"{sig - ref['sigma']:+.3e}")
    print(f"  acc    clean {acc:.9f}  frozen {ref['acc']:.9f}  delta "
          f"{acc - ref['acc']:+.3e}")
    walk_ok = (len(snaps) == ref["n"] and sig == ref["sigma"] and acc == ref["acc"])
    print(f"  trajectory reproduced bit-for-bit: {walk_ok}")

    print()
    print(f"  {'quantity':<9}{'clean':>18}{'frozen':>18}{'delta':>14}{'z':>9}")
    print("  " + "-" * 68)
    Emine = T + S.kappa * V                       # clean kappa, exact
    Erec = T + ref["kappa"] * V                   # the record's rounded kappa
    rows = (("T/N", T.mean(), ref["T"], ref["T_err"]),
            ("V/N", V.mean(), ref["V"], ref["V_err"]),
            ("E/N", Emine.mean(), ref["E"], ref["E_err"]),
            ("E/N (k rec)", Erec.mean(), ref["E"], ref["E_err"]))
    for nm, a, f, fe in rows:
        a, f, fe = a / ne, f / ne, fe / ne
        print(f"  {nm:<9}{a:>18.9f}{f:>18.9f}{a - f:>+14.2e}{(a - f)/fe:>9.2f}")

    # The error bar must be built from ONE sequence.  An earlier revision of this
    # file wrote `T.std(ddof=1)/sqrt(n) * sqrt(tau_int(T + kappa*V))` -- T's naive
    # error on the mean times the SQUARE ROOT OF E's tau -- which mixes two series
    # and is not any estimator of anything.  It produced 0.022792021 and is the
    # reason the log from before 2026-10-02 must not be quoted.
    naive = float(Emine.std(ddof=1) / np.sqrt(len(Emine)))
    ti = float(legacy_tau_int(Emine))
    print()
    print(f"  E_err_naive/N  clean {naive / ne:.9f}   frozen "
          f"{ref['E_err_naive'] / ne:.9f}   delta "
          f"{naive - ref['E_err_naive']:+.3e}  (TOTALS)")
    print(f"  tau_int  E     clean {ti:.6f}   frozen {ref['E_tau']:.6f}")
    print(f"  E_err/N        clean {naive * np.sqrt(ti) / ne:.9f}   frozen "
          f"{ref['E_err'] / ne:.9f}   (campaign tau rule, one sequence)")
    print()
    d = max(abs(T.mean() - ref["T"]), abs(V.mean() - ref["V"]),
            abs(Erec.mean() - ref["E"]))
    print(f"  worst |clean - frozen| over T, V, E(record kappa) = {d:.3e}  (TOTALS)")
    print("  VERDICT: the clean package reproduces the frozen record"
          if d < 1e-6 else "  VERDICT: still differs -- look further")
    return 0 if d < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
