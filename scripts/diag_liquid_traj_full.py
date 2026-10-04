"""A8b -- where do the legacy and clean trajectories separate over the full walk?

PART 2 of `diag_liquid_a8.py` shows the first 200 attempts are BIT-IDENTICAL
(0/200 decision mismatches, max |dR| = 0, RNG state equal).  Yet the two full
walks end with different adapted sigma (0.6166 vs 0.6507), so they separate
somewhere later.  This finds the exact attempt.

That matters for the acceptance argument: a divergence at attempt 201 would be a
defect; a divergence at the first `rebuild` (attempt 100*ne = 3600) would be the
expected consequence of two mathematically equal but separately coded floating
point paths, after which the two chains are simply two independent samples of the
same distribution and only the STATISTICS can be compared.

Same protocol as the frozen record: sigma 0.4, seed 11, 1500 sweeps, target 0.4.

    python scripts/diag_liquid_traj_full.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
PROJ = os.path.dirname(CLEAN)
for _p in (os.path.join(CLEAN, "src"), PROJ, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wigner_vmc.physics.geometry import wrap_to_supercell   # noqa: E402
import bench_rs75 as B                                      # noqa: E402
from diag_liquid_a8 import build_both, SIGMA, SEED, SWEEPS   # noqa: E402

NE = 36
REBUILD_AT = 100 * NE        # sample.jl reset_period


def trajectory(wf, R0, nsweep, trace_to=None):
    """`sample`, verbatim, optionally tracing every attempt up to `trace_to`."""
    import numpy as _np
    rng = _np.random.default_rng(SEED)
    st = wf.build(R0)
    ne = wf.ne
    sigma = SIGMA
    hist = _np.zeros(1000, dtype=_np.int8)
    nacc = ntot = 0
    per_sweep = []
    tr = {"t": [], "R": [], "ok": [], "acc": []}
    for sweep in range(nsweep):
        for i in range(ne):
            r_new = wrap_to_supercell(
                st["R"][i] + sigma * rng.standard_normal(2), wf.sc_to_cart,
                wf.cart_to_sc)[0]
            acc, info = wf.move_ratio(st, i, r_new)
            ok = rng.random() < acc
            if ok:
                wf.accept_move(st, i, r_new, info)
            if trace_to is not None and ntot < trace_to:
                tr["t"].append(ntot)
                tr["R"].append(st["R"].copy())
                tr["ok"].append(bool(ok))
                tr["acc"].append(acc)
            hist[ntot % 1000] = ok
            nacc += ok
            ntot += 1
            if ntot % 1000 == 0 and ntot >= 1000:
                rate = hist.mean()
                if rate > 0.4:
                    sigma *= 1.05
                elif rate < 0.4:
                    sigma *= 0.95
            if ntot % REBUILD_AT == 0:
                wf.rebuild(st)
        per_sweep.append(st["R"].copy())
    return _np.array(per_sweep), sigma, nacc / max(ntot, 1), tr


def main():
    S, wfL, wfC = build_both()
    R0 = S.liquid_R0()

    t0 = time.time()
    swL, sigL, accL, _ = trajectory(wfL, R0, SWEEPS)
    swC, sigC, accC, _ = trajectory(wfC, R0, SWEEPS)
    print(f"full sweep trajectory: 2 x {SWEEPS} sweeps in {time.time()-t0:.0f}s")
    print(f"  legacy  sigma={sigL:.6f} acc={accL:.4f}")
    print(f"  clean   sigma={sigC:.6f} acc={accC:.4f}")

    d = np.abs(swL - swC).reshape(len(swL), -1).max(axis=1)
    # `np.argmax(d > 0)` returns 0 when NOTHING differs -- it would report "the
    # first differing sweep is 0" for a pair that never separates.  Test for the
    # absence explicitly first; that is the whole point of this script.
    differs = np.flatnonzero(d > 0)
    print(f"\n  max |dR| over ALL {len(swL)} sweeps        : {d.max():.3e}")
    if len(differs) == 0:
        print("  NO sweep's configuration differs -- the two chains are the same "
              "trajectory end to end")
        return 0
    first = int(differs[0])
    print(f"  first sweep whose configuration differs : {first}  "
          f"(of {len(swL)})")
    print(f"  max |dR| over sweeps before it          : {d[:first].max():.3e}")
    print(f"  REBUILD_AT (reset_period) = {REBUILD_AT} attempts = sweep "
          f"{REBUILD_AT//NE}")

    # attempt-level trace up to one sweep past the divergence
    upto = (first + 1) * NE
    _, _, _, trL = trajectory(wfL, R0, first + 1, trace_to=upto)
    _, _, _, trC = trajectory(wfC, R0, first + 1, trace_to=upto)
    okl = np.array(trL["ok"])
    okc = np.array(trC["ok"])
    mis = np.flatnonzero(okl != okc)
    print(f"\n  attempt-level trace to attempt {upto}:")
    if len(mis):
        k = int(mis[0])
        print(f"    FIRST DIFFERENT DECISION at attempt {k}  "
              f"(sweep {k//NE}, electron {k%NE})")
        print(f"      acc legacy {trL['acc'][k]!r}")
        print(f"      acc clean  {trC['acc'][k]!r}")
        print(f"      |dacc| = {abs(trL['acc'][k]-trC['acc'][k]):.3e}")
        print(f"      decision legacy={okl[k]} clean={okc[k]}")
    else:
        print(f"    no decision differs in the first {upto} attempts "
              "(the configurations differ anyway -- investigate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
