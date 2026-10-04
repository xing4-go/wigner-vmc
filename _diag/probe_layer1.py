"""Layer 1 at the real operating point and at the test-sized one.  Read-only."""
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
PATH = os.path.join(CLEAN, "examples", "phase_competition_2",
                    "crystal_c6_trajectory.py")
spec = importlib.util.spec_from_file_location("c6traj", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from wigner_vmc import VMC, load_budget                     # noqa: E402

for N, nsp, tag in ((36, 120, "cheap test nsp"), (36, 240, "two-decade nsp")):
    t0 = time.time()
    bud = load_budget("quick")
    vmc = VMC(N=N, rs=90.0, phase="crystal", nmax=1)
    torus = m.analysis_torus(vmc.N)
    ctx = m.c6_context(vmc, torus, nsp=nsp)
    v0, rec = m.seed_controls(vmc, torus, ctx, float(bud.width_for(0)),
                              nsp2=4 * nsp, seed2=m.C6_SEED + 1)
    rec.update(rs=float(vmc.rs), nmax=int(vmc.nmax), N=int(vmc.N))
    sv = m.seed_verdict(rec)
    print(f"\n== N={N}, nsp={nsp} ({tag})  [{time.time() - t0:.1f}s]")
    for k in ("nbar_seed", "nbar_defect", "c6_correct", "c6_correct_ctx2",
              "c6_defect", "c6_defect_ctx2", "c6_gauge_shift"):
        print(f"   {k:20s} {rec[k]:.6e}")
    print(f"   verdict ok={sv['ok']}  failed={sv['controls_that_failed']}")
    print(f"   {sv['reason'][:160]}")

    # what the *run's* own nsp would give at N=36, for the metadata default
    if N == 36:
        res = m.c6_residuals(v0, ctx)
        print("   five-vector at nsp=500: "
              + ", ".join(f"{th:.0f}:{r:.3e}" for th, r in res.items()))
        print(f"   liquid reference (v=0): "
              f"{max(m.c6_residuals(np.zeros((vmc.lat.nk, vmc.n_bands - 1), complex), ctx).values()):.3e}")
