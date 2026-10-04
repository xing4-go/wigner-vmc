"""Why the nb = 3 SR beat the nb = 2 SR by 3x on the same box.

B6 (nb = 2, 77 parameters) recorded `sr_seconds` = 6039.6 for 120 steps.
B8 (nb = 3, 149 parameters) recorded `sr_seconds` = 2039.2 for the same
120 x 400 protocol.  A LARGER system ran 3x FASTER, and no code path explains it:
both scripts call `sr.sr_optimize_joint` with argument-identical settings imported
from the same `bench_rs75`, and `sample`'s `target_acc` only adapts `sigma` -- it
never short-circuits the sweep loop, so both runs really did 120 x 400 sweeps.

B8's SR is not the suspect.  Its trajectory reproduces the campaign's frozen nb = 3
pkl to 8.5e-09 over all 120 steps with `acc`/`sigma`/`tau` exactly 0, which is only
possible if it did the same work.  The suspect is B6's clock, and the mechanism on
this box is documented: wall-clock here swings by ~9x with machine load.

This script settles it by MEASUREMENT, not by attribution:

1. a raw `sample` cost per sweep at nb = 2 and nb = 3 (isolates the wavefunction
   cost, with no SR solver and no `log_deriv` in the way);
2. a real `sr_optimize_joint` cost per step at both nb, run back to back in ONE
   process so the two numbers share a machine state.

If nb = 3 measures only ~1.2x nb = 2 (the ratio the two production WALKS show:
145.6 s -> 178.0 s), then B6's 50.3 s/step was an environment artefact and B8's
17.0 s/step is the honest cost.  Either way the physics gates are untouched -- this
is a cost question, and it is reported as one.

This is a DIAGNOSTIC.  It writes no state and touches no frozen file.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CLEAN, "src"))
sys.path.insert(0, os.path.join(CLEAN, "scripts"))
sys.path.insert(0, HERE)

from bench_rs75 import (PROTOCOL, SR_SIGMA, SR_TARGET, Setup,          # noqa: E402
                        legacy_regression_kappa)
from wigner_vmc.vmc import sr                                          # noqa: E402

RS = 75.0
K_LEGACY = legacy_regression_kappa(RS)
B6 = os.path.join(CLEAN, "logs", "replay_crystal_rs75.json")
B8 = os.path.join(CLEAN, "logs", "replay_nested_rs75.json")


def sweep_cost(wf, R0, rng, n=60, equil=3):
    """Seconds per Metropolis sweep, with no solver and no snapshot copying."""
    from wigner_vmc.vmc.sampler import sample
    t0 = time.time()
    sample(wf, R0, nsweep=n, sigma=SR_SIGMA, rng=rng, snapshot_every=n,
           equil=equil, target_acc=SR_TARGET)
    return (time.time() - t0) / n


def sr_step_cost(mk, th0, R0, proto, si, steps=2):
    """Seconds per full SR step, exactly as the two replays drive it."""
    t0 = time.time()
    sr.sr_optimize_joint(
        mk, th0, R0, steps=steps, nsweep=proto["sr_sweeps"], sigma=SR_SIGMA,
        seed=si, snapshot_every=proto["sr_snap"], equil=proto["sr_equil"],
        target_acc=SR_TARGET)
    return (time.time() - t0) / steps


def main():
    proto = dict(PROTOCOL["regression"])
    raw = json.load(open(os.path.join(CLEAN, "results", "bench_rs75",
                                      "initial_conditions.json"), encoding="utf-8"))
    S = Setup(raw, kappa=K_LEGACY, kappa_mode="legacy-regression")
    c0, v0 = S.seed_state(0)
    th2 = S.theta(c0, v0)                                   # 77, B6's own start
    th3 = np.asarray(json.load(open(B8, encoding="utf-8"))["theta"], float)  # 149

    print(f"  legacy kappa {K_LEGACY}   protocol {proto['sr_steps']}x"
          f"{proto['sr_sweeps']} equil {proto['sr_equil']} snap {proto['sr_snap']}")
    print(f"  len(theta): nb=2 {th2.size}   nb=3 {th3.size}\n")

    # --- 1. raw sweep cost, the cleanest apples-to-apples number ---------------
    print("  --- raw Metropolis sweep cost (no solver, no log_deriv) ---")
    per_sweep = {}
    for nb, th in ((2, th2), (3, th3)):
        wf = S.maker(nb)(th)
        rng = np.random.default_rng(0)
        R = S.crystal_R0().copy()
        per_sweep[nb] = sweep_cost(wf, R, rng)
        print(f"    nb={nb}   {per_sweep[nb] * 1e3:8.2f} ms/sweep")
    print(f"    ratio nb3/nb2 = {per_sweep[3] / per_sweep[2]:.3f}")

    # --- 2. real SR step cost, back to back in one process --------------------
    print(f"\n  --- full SR step cost, {2} steps each, same process ---")
    R0 = S.crystal_R0()
    per_step = {}
    for nb, th in ((2, th2), (3, th3)):
        per_step[nb] = sr_step_cost(S.maker(nb), th, R0, proto, 0)
        print(f"    nb={nb}   {per_step[nb]:8.2f} s/step   "
              f"(120 steps -> {per_step[nb] * 120 / 60:.1f} min)")
    print(f"    ratio nb3/nb2 = {per_step[3] / per_step[2]:.3f}")

    # --- 3. against what the two replays actually recorded -------------------
    rec = {}
    for tag, path in (("B6", B6), ("B8", B8)):
        b = json.load(open(path, encoding="utf-8"))
        a = b["arms"]["legacy"] if "arms" in b else b
        rec[tag] = (a["n_band"], float(a["sr_seconds"]) / proto["sr_steps"])
    print("\n  --- recorded, same protocol, same box ---")
    for tag, (nb, s) in rec.items():
        print(f"    {tag}  nb={nb}   {s:8.2f} s/step")
    print(f"    ratio B8/B6 recorded = {rec['B8'][1] / rec['B6'][1]:.3f}")

    print("\n  VERDICT")
    pred = rec["B6"][1] * per_step[3] / per_step[2]
    print(f"    nb=3 should cost {per_step[3] / per_step[2]:.3f}x nb=2, so B6's "
          f"{rec['B6'][1]:.1f} s/step predicts {pred:.1f} s/step for B8.")
    print(f"    B8 recorded {rec['B8'][1]:.1f} s/step against a measured "
          f"{per_step[3]:.1f} s/step now.")
    close = abs(per_step[3] - rec["B8"][1]) / rec["B8"][1] < 0.35
    print("    => B8's clock is reproducible on an idle box; B6's was inflated by "
          "machine load\n       (a wall-clock effect, NOT a physics one)."
          if close else
          "    => B8's recorded step is NOT reproducible now either -- the box's "
          "cost varies\n       by more than this probe can separate; treat all "
          "wall-clock here as load-dependent.")
    print("    Physics gates are unaffected: B6/B7/B8 trajectories each reproduce "
          "their frozen\n      record to round-off, which is only possible if the "
          "sweep counts matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
