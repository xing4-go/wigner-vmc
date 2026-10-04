"""B7's one open number: why clean and legacy report different error bars.

The walk reproduces the legacy record's MOMENTS to round-off -- `E`, `T`, `V`,
`acc`, `sigma`, `cov_TV`, `n` all agree, and `E_err_naive` is identical -- so the
only field that differs is `E_tau`: clean 27.1475, legacy 32.5464.  Both sides
apply `E_err = E_err_naive * sqrt(E_tau)`, so the whole disagreement is in how
`tau` is estimated, not in what was sampled.

This script settles it by MEASUREMENT rather than by reading code:

1. rebuild the B6 endpoint, re-drive the production walk, and check the recovered
   `E` sequence still hashes to the `sequence_sha1` B7 recorded -- i.e. this is
   the SAME series, not a re-sample;
2. apply the two estimators to that one series:
   * the clean package's, `statistics.tau_int(x, c=5.0)`;
   * the LEGACY's own, imported from the frozen `_diag/mech.py`, so the legacy
     number is produced by the legacy's code and not by my transcription of it.

If the legacy estimator returns the legacy record's 32.5464 on a series the clean
estimator reads as 27.1475, the difference is a convention -- which is what the
Blocker B brief says must be reported separately and must NOT be scored as a
physics failure.

This is a DIAGNOSTIC.  It writes no state and touches no frozen file.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CLEAN, "src"))
sys.path.insert(0, os.path.join(CLEAN, "scripts"))
sys.path.insert(0, HERE)

from bench_rs75 import PROTOCOL, Setup                                  # noqa: E402
from replay_crystal_rs75 import (K_LEGACY, hashlib_sha1,                # noqa: E402
                                 walk_and_sequence)
from wigner_vmc.analysis.statistics import tau_int                      # noqa: E402

PARENT = os.path.join(CLEAN, "logs", "replay_crystal_rs75.json")
LEGACY_MECH = os.path.join(os.path.dirname(CLEAN), "_diag", "mech.py")


def legacy_mech():
    """The frozen campaign's own ``tau_int``, loaded from ``_diag/mech.py``.

    Imported rather than re-implemented: a transcription would only show that I
    can copy a loop, and the claim being tested is about what the LEGACY code
    computes.
    """
    spec = importlib.util.spec_from_file_location("legacy_mech", LEGACY_MECH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_mech"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    blob = json.load(open(PARENT, encoding="utf-8"))
    arm = blob["arms"]["legacy"]
    raw = json.load(open(os.path.join(CLEAN, "results", "bench_rs75",
                                      "initial_conditions.json"), encoding="utf-8"))
    proto = dict(PROTOCOL["regression"])

    S = Setup(raw, kappa=K_LEGACY, kappa_mode="legacy-regression")
    wf = S.maker(2)(np.asarray(arm["theta"], float))

    # The JSON keeps only the sequence's hash, so the series has to be recovered
    # by re-driving the walk.  Cache it: this is the expensive step (147 s) and
    # the estimators below are what the script is for.
    cache = os.path.join(CLEAN, "logs", "diag_tau_series.npy")
    if os.path.exists(cache):
        E = np.load(cache)
        wrec = json.load(open(os.path.join(CLEAN, "logs",
                                           "diag_tau_walkrecord.json"),
                              encoding="utf-8"))
        secs = float("nan")
        print(f"  using the cached E series ({E.size} points)", flush=True)
        wseq = {"E": E.tolist()}
    else:
        print("re-driving the production walk to recover the E series ...",
              flush=True)
        wrec, wseq, secs = walk_and_sequence(S, wf, proto, "tau-probe")
        E = np.asarray(wseq["E"], float)
        np.save(cache, E)
        with open(os.path.join(CLEAN, "logs", "diag_tau_walkrecord.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(wrec, fh, indent=1, sort_keys=True)

    got = hashlib_sha1(wseq["E"])
    want = arm["walk"]["sequence_sha1"]
    print(f"\n  sequence sha1  {got}   B7 recorded {want}   "
          f"{'SAME SERIES' if got == want else '*** DIFFERENT SERIES ***'}")
    print(f"  n = {E.size}   ({secs:.0f}s)")

    mech = legacy_mech()
    tau_clean, win = tau_int(E)
    tau_legacy = float(mech.tau_int(E))

    # `E` is a per-snapshot TOTAL (T + kappa*V), and the record's `E_err_naive`
    # is PER PARTICLE, so the comparison has to cross that factor of ne once.
    naive_tot = float(E.std(ddof=1) / np.sqrt(E.size))
    naive_pp = naive_tot / S.ne
    print(f"\n  on ONE series, two estimators:")
    print(f"    clean  statistics.tau_int(E, c=5.0)  tau = {tau_clean:9.4f}  "
          f"window {win}")
    print(f"    legacy _diag/mech.tau_int(E)         tau = {tau_legacy:9.4f}")
    print(f"    ratio legacy/clean {tau_legacy / tau_clean:.4f}   "
          f"error-bar ratio {np.sqrt(tau_legacy / tau_clean):.4f}")

    print(f"\n  the clean walk's own record:")
    print(f"    E_tau                {wrec['E_tau']:.6f}")
    print(f"    E_err_naive (per/el) {wrec['E_err_naive']:.9f}   "
          f"recomputed {naive_pp:.9f}")
    print(f"    E_err       (per/el) {wrec['E_err']:.9f}")
    print(f"    clean  predicts E_err/el {naive_pp * np.sqrt(tau_clean):.9f}")
    print(f"    legacy predicts E_err/el {naive_pp * np.sqrt(tau_legacy):.9f}")

    print(f"\n  the frozen campaign reporter's row, the number B7 compared against:")
    rec = json.load(open(os.path.join(os.path.dirname(CLEAN), "_diag", "part1",
                                      "cry_conv_rs75_nb2_s0.json"), encoding="utf-8"))
    # UNIT ASYMMETRY, and it is a known trap between the two JSON families: the
    # legacy record's E_err / E_err_naive are TOTALS (sum over electrons) while
    # the clean record's are PER PARTICLE.  They differ by exactly ne = 36, which
    # is why the ratio E_err/E_err_naive -- the only thing this script uses --
    # is comparable across them and the raw values are not.
    print(f"    legacy record E_tau            {rec['E_tau']:.6f}")
    print(f"    legacy record E_err    (TOTAL) {rec['E_err']:.9f}")
    print(f"    legacy record E_err_naive(TOT) {rec['E_err_naive']:.9f}")
    print(f"      ratio = {rec['E_err'] / rec['E_err_naive']:.5f} = "
          f"sqrt({rec['E_tau']:.6f}) -> {np.sqrt(rec['E_tau']):.5f}")
    print(f"      (legacy totals / ne = {rec['E_err_naive'] / S.ne:.9f} per electron, "
          f"the clean convention)")
    ok = abs(tau_legacy - float(rec["E_tau"])) < 1e-4
    print(f"\n  VERDICT: the legacy estimator reproduces the LEGACY record's E_tau "
          f"({rec['E_tau']:.6f}) on the clean series: {ok}")
    print("  => the two error bars differ by ESTIMATOR CONVENTION on an identical "
          "series,\n     not by any difference in the sampled configurations."
          if ok else
          "  => the estimators do NOT explain the difference; something else does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
