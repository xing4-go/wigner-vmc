"""Blocker B / B6 + B7 -- replay the frozen crystal at r_s = 75.

B6, the SR trajectory (always run)
----------------------------------
After the initialization and the digest are aligned (B4/B5), run the stochastic
reconfiguration and compare the TRAJECTORY, not just the endpoint:

    same theta0, parameter ordering, RNG, SR steps/sweeps, regularization,
    update rule, kappa mode;  compare E, S, f, delta_theta, theta per step.

B7, the production walk (``--walk``)
------------------------------------
The walk is a separate protocol with its own RNG seed, so it is a separate
statement about the clean sampler.  It is run in the SAME process because it
needs the just-optimised wavefunction: `bench_rs75.run_crystal` does SR-then-walk
for that reason, and re-running the 120 x 400 SR to obtain the same theta again
would spend half an hour re-deriving a number already in hand.

The frozen campaign left both in the store, so this is a comparison and not a
reconstruction:

    _diag/ckpt_rebuild/llcryst_nb2_k53.033_p1conv{0..4}.pkl
        {"theta": (77,), "hist": [120 x {step, E, E_err, acc, sigma, theta,
                                         force, tau, cond, eig_min, eig_max}],
         "n_band": 2, "kappa": 53.033}
    %TEMP%/qhvmc_checkpoints/llcryst_nb2_k53.033_p1conv4.pkl   (seed 4)

``llcryst_nb2_k53.033_p1conv{0..3}`` are in the frozen store and are covered by
the pre-incident manifest; ``p1conv4`` lives in ``%TEMP%`` and was never touched
by the 2026-10-02 incident.  The stored ``kappa`` field is 53.033 -- the rounded
campaign value -- which is the same reading that ``_init_digest`` gave
independently, from the frozen Jastrow warm starts, in ``diag_c0_kappa.py``.

TWO INDEPENDENT ORACLES, DELIBERATELY
-------------------------------------
The pkl holds the per-step trajectory.  The campaign's own reporter also wrote
scalars into ``_diag/part1/cry_conv_rs75_nb2_s{si}.json``:

    E_start, E_final, E_min, force_first, force_final, cond_final, theta (77)

Those were produced by a different code path from the pkl, so agreement with
both is stronger than agreement with either.  Neither is an input to the run:
the clean SR is driven entirely by the clean package and the ``c0``/``v0`` from
``initial_conditions.json``.

THE KAPPA ARM
-------------
The campaign optimised at ``kappa = round(rs/sqrt(2), 4) = 53.033``.  The clean
physics value is ``53.03300858899106``.  A trajectory is a nonlinear iteration,
so this script runs BOTH and reports each against the frozen record.  That is
the error-bar statement the Blocker B brief asks for, in the only place where it
is quantitative: the legacy arm says whether the clean SR *is* the frozen SR;
the physical arm says how far the physics convention moves it.

NOT a pass/fail gate on the endpoint
------------------------------------
Step 0 is the decisive step.  It starts from the same theta0 with the same
sampler, so if the wiring, the parameter ordering and the RNG stream agree it
must match to floating-point noise.  Later steps can then separate by chaos
amplification even when every input is identical, so the per-step profile is
reported rather than reduced to one number, and a late-step divergence is not
by itself a defect.

    python tools/replay_crystal_rs75.py --seed 0                # B6, both arms
    python tools/replay_crystal_rs75.py --seed 0 --steps 10 --kappa-mode legacy
    python tools/replay_crystal_rs75.py --seed 0 --walk         # B6 + B7
    python tools/replay_crystal_rs75.py --budget smoke          # plumbing, ~1 min

THE WALK'S ORACLE IS MOMENTS, NOT A SEQUENCE
--------------------------------------------
``_diag/part1/cry_conv_rs75_nb2_s{si}.json`` stores the walk's MOMENTS --
E/N, E_err, T, V, acc, sigma, E_tau, V_tau, cov_TV, n -- and not the per-snapshot
E sequence, so a sequence-to-sequence comparison has no counterpart on the legacy
side and is not claimed here.  What is claimed is stronger than a mean: ten
moments of the same sequence, reproduced together.  A single differing snapshot
would move several of them, so agreement across all ten at 1e-12 is a fingerprint
of the sequence, and this script says so rather than calling it an identity.

The clean side DOES record its raw sequence (``--walk`` writes it), because a
sequence is the thing whose moments were compared and it should be inspectable.
It is recovered by re-driving ``sample`` with the identical arguments and is
accepted only if it reproduces ``production_walk``'s own moments exactly -- a
reconstruction that has not been checked against the production path is not
evidence about the production path.
"""
from __future__ import annotations

import argparse
import hashlib as _h
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SCRIPTS = os.path.join(CLEAN, "scripts")
SRC = os.path.join(CLEAN, "src")
for p in (SCRIPTS, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_rs75 import (CRY_MEAS_SEED, CRY_MEAS_SIGMA, PROTOCOL,       # noqa: E402
                        SR_SIGMA, SR_TARGET, Setup, legacy_regression_kappa,
                        physical_kappa, production_walk)
from wigner_vmc.vmc import sr as sr                                     # noqa: E402
from wigner_vmc.vmc.sampler import sample                               # noqa: E402

PROJ = os.path.dirname(CLEAN)
FROZEN_CKPT = os.path.join(PROJ, "_diag", "ckpt_rebuild")
TEMP_CKPT = os.path.join(os.environ.get("TEMP", "/tmp"), "qhvmc_checkpoints")
LEGACY_DIR = os.path.join(PROJ, "_diag", "part1")
INIT_JSON = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
OUT = os.path.join(CLEAN, "logs", "replay_crystal_rs75.json")

RS = 75.0
K_LEGACY = legacy_regression_kappa(RS)      # 53.033
K_PHYS = physical_kappa(RS)                 # 53.03300858899106

#: seed index -> (L0, the digest the frozen tag carries, which store holds the pkl).
#: The index IS the seed: the tuple's own leading seed number was redundant with it,
#: and `main` unpacks the tuple into three names, so a fourth element made every run
#: die with "too many values to unpack" before it started.
#: The digests were read off the frozen tags by `diag_c0_kappa.py`, and
#: `test_init_provenance.py` re-reads them from the tags on every run.
SEEDS = (
    (0.30, "cadd5d6a", "frozen"),
    (0.45, "46d9bf1e", "frozen"),
    (0.50, "eba63d0f", "frozen"),
    (0.60, "7f284e09", "frozen"),
    (0.80, "6d38078f", "temp"),
)

#: The fields the frozen hist and the clean hist both carry.  They are the same
#: eleven names -- that identity is itself part of what B6 checks.
FIELDS = ("E", "E_err", "acc", "sigma", "force", "tau", "cond",
          "eig_min", "eig_max")


def _init_digest(v0, c0, nd=12):
    """VERBATIM from the notebook's cell 80.  Copied, not re-derived."""
    _v0, _c0 = np.asarray(v0, dtype=complex), np.asarray(c0, dtype=float)
    _w = np.concatenate([_v0.ravel().view(float), _c0.ravel(),
                         np.array(_v0.shape + _c0.shape, dtype=float)])
    _mag = np.maximum(np.abs(_w), 1e-300)
    _exp = np.clip((nd - 1) - np.floor(np.log10(_mag)), -300.0, 290.0)
    _r = np.rint(_w * np.power(10.0, _exp)).astype(np.int64)
    return _h.blake2b(np.ascontiguousarray(_r).tobytes(), digest_size=4).hexdigest()


def frozen_state(si):
    """The frozen SR record for seed ``si``: (blob, path) or (None, path)."""
    where = SEEDS[si][2]
    base = TEMP_CKPT if where == "temp" else FROZEN_CKPT
    path = os.path.join(base, f"llcryst_nb2_k53.033_p1conv{si}.pkl")
    if not os.path.exists(path):
        return None, path
    with open(path, "rb") as fh:
        return pickle.load(fh), path


def legacy_json(si):
    path = os.path.join(LEGACY_DIR, f"cry_conv_rs75_nb2_s{si}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dev(a, b):
    """max |a-b| and the index at which it occurs."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    i = int(np.argmax(d))
    return float(d.reshape(-1)[i] if d.ndim == 1 else d.reshape(-1).max()), i


PROFILE_STEPS = (0, 1, 2, 3, 5, 10, 20, 40, 60, 80, 100, -1)


def _profile(d, n):
    """The deviation at the requested checkpoints, restricted to steps that exist.

    ``PROFILE_STEPS`` is written for a 120-step run.  Comparing fewer steps -- the
    ``--steps 10`` escalation probe compares exactly one -- leaves ``d`` shorter
    than the table, and indexing it unguarded raised IndexError AFTER the SR had
    already been paid for, throwing away the run.  Only the available checkpoints
    are reported; ``n_compared`` beside them says how many steps there were.
    """
    return {str(k): float(d[k]) for k in PROFILE_STEPS if -n <= k < n}


def compare_hist(clean, frozen, n_cmp):
    """Field-by-field comparison of the two trajectories over ``n_cmp`` steps."""
    ch, fh_ = clean, frozen
    n = min(n_cmp, len(ch), len(fh_))
    out = {"n_clean": len(ch), "n_frozen": len(fh_), "n_compared": n}
    if n <= 0:
        return out
    out["fields"] = {}

    # the field-name identity: the same eleven keys, so a field cannot be
    # compared against a differently-named quantity on the other side.
    keys_c, keys_f = sorted(ch[0]), sorted(fh_[0])
    out["clean_keys"] = keys_c
    out["frozen_keys"] = keys_f
    out["keys_identical"] = bool(keys_c == keys_f)

    for f in FIELDS:
        c = np.array([h[f] for h in ch[:n]], float)
        z = np.array([h[f] for h in fh_[:n]], float)
        d = np.abs(c - z)
        out["fields"][f] = {
            "max_abs_dev": float(d.max()),
            "step_of_max": int(np.argmax(d)),
            "dev_step0": float(d[0]),
            "frozen_step0": float(z[0]),
            "clean_step0": float(c[0]),
            "frozen_last": float(z[-1]),
            "clean_last": float(c[-1]),
            "profile": _profile(d, n),
        }

    # theta: a vector per step, so the same treatment with a norm.
    dt = np.array([np.max(np.abs(np.asarray(ch[i]["theta"], float)
                                 - np.asarray(fh_[i]["theta"], float)))
                   for i in range(n)])
    out["theta"] = {
        "shape_clean": list(np.shape(ch[0]["theta"])),
        "shape_frozen": list(np.shape(fh_[0]["theta"])),
        "max_abs_dev": float(dt.max()),
        "step_of_max": int(np.argmax(dt)),
        "dev_step0": float(dt[0]),
        "profile": _profile(dt, n),
    }
    # final theta versus the legacy reporter's own final theta (oracle 2)
    out["theta_final_max_abs_dev"] = float(np.max(np.abs(
        np.asarray(ch[-1]["theta"], float) - np.asarray(fh_[-1]["theta"], float))))
    return out


#: The campaign reporter's scalars, and the clean quantity each one is the
#: counterpart of.  Both are read off the same SR hist; the names differ because
#: the two reporters were written years apart, and the mapping is stated here
#: rather than assumed at the comparison site.
JSON_ORACLE = (
    ("E_start", "hist[0].E / ne"),
    ("E_final", "hist[-1].E / ne"),
    ("E_min", "min(hist.E) / ne"),
    ("force_first", "hist[0].force"),
    ("force_final", "hist[-1].force"),
    ("cond_final", "hist[-1].cond"),
)


def compare_json(clean_hist, rec, ne, full):
    """The campaign reporter's scalars -- an independent second oracle.

    ``rec`` is ``_diag/part1/cry_conv_rs75_nb2_s{si}.json``, written by the
    campaign's own reporter rather than by the SR loop that wrote the pkl.  The
    two oracles therefore fail independently.

    ``full`` says whether the clean run has the legacy length.  ``E_start`` and
    ``force_first`` are properties of step 0 and are comparable either way; the
    end-of-run and min-over-run scalars are properties of all 120 steps, so on a
    short escalation run they are reported as NOT COMPARABLE rather than as a
    mismatch.  Calling a 10-step run's endpoint "different from the legacy
    endpoint" would be a category error, and it is exactly the kind of stale
    comparison this firewall exists to prevent.
    """
    if rec is None:
        return None
    eh = np.array([h["E"] for h in clean_hist], float) / ne
    clean = {
        "E_start": float(eh[0]),
        "E_final": float(eh[-1]),
        "E_min": float(eh.min()),
        "force_first": float(clean_hist[0]["force"]),
        "force_final": float(clean_hist[-1]["force"]),
        "cond_final": float(clean_hist[-1]["cond"]),
    }
    step0 = ("E_start", "force_first")
    rows = {}
    for k, _how in JSON_ORACLE:
        w = rec.get(k)
        comparable = full or k in step0
        rows[k] = {"clean": clean[k], "legacy": w, "comparable": bool(comparable),
                   "dev": (None if (w is None or not comparable)
                           else float(abs(clean[k] - w)))}
    th_c = np.asarray(clean_hist[-1]["theta"], float)
    th_l = rec.get("theta")
    rows["theta_final"] = {
        "clean": None, "legacy": None, "comparable": bool(full),
        "clean_len": int(th_c.size),
        "legacy_len": (None if th_l is None else len(th_l)),
        "dev": (None if (th_l is None or not full)
                else float(np.max(np.abs(th_c - np.asarray(th_l, float))))),
    }
    return rows


#: The legacy reporter's walk moments, and the clean key each one is the
#: counterpart of.  The legacy record's bare ``T``/``V``/``E`` are TOTALS -- its
#: own ``T + kappa * V == E`` closes to <1e-9 -- while the per-electron values
#: are the separate ``*_perpart`` fields.  The clean walk reports both, so the
#: mapping is stated here rather than guessed at the comparison site.
WALK_ORACLE = (
    ("E_perpart", "E", "per electron"),
    ("E_perpart_err", "E_err", "per electron"),
    ("T", "T_total", "total"),
    ("V", "V_total", "total"),
    ("acc", "acc", "walker"),
    ("sigma", "sigma", "walker"),
    ("E_tau", "E_tau", "autocorrelation"),
    ("V_tau", "V_tau", "autocorrelation"),
    ("cov_TV", "cov_TV", "per snapshot"),
)


def walk_and_sequence(S, wf, proto, label, verbose=True):
    """``bench_rs75.production_walk`` PLUS the raw sequence behind its moments.

    The production function is what is called for the numbers -- B7 is a claim
    about the production path, and computing the moments by hand would test this
    script instead.  The sequence is then recovered by re-driving ``sample`` with
    the identical arguments, and is accepted only if it reproduces the production
    moments EXACTLY.  That check is the whole reason the sequence is trustworthy:
    an unchecked reconstruction would be evidence about nothing.
    """
    t0 = time.time()
    rec = production_walk(wf, S.crystal_R0(), proto["meas_sweeps"],
                          proto["meas_equil"], CRY_MEAS_SIGMA, CRY_MEAS_SEED, label)
    snaps, sig, acc = sample(wf, S.crystal_R0(), nsweep=proto["meas_sweeps"],
                             sigma=CRY_MEAS_SIGMA,
                             rng=np.random.default_rng(CRY_MEAS_SEED),
                             snapshot_every=1, equil=proto["meas_equil"],
                             target_acc=0.4)
    T = np.empty(len(snaps))
    V = np.empty(len(snaps))
    for i, Sn in enumerate(snaps):
        t, v, _ = wf.local_energy(wf.build(Sn))
        T[i], V[i] = t.real, v.real
    E = T + wf.kappa * V
    seq = {"n": int(len(snaps)), "E": E.tolist(), "T": T.tolist(), "V": V.tolist()}
    checks = {
        "n": len(snaps) == rec["n"],
        "acc": float(acc) == float(rec["acc"]),
        "sigma": float(sig) == float(rec["sigma"]),
        "T_mean": float(T.mean()) == float(rec["T_total"]),
        "V_mean": float(V.mean()) == float(rec["V_total"]),
        "E_mean": float(E.mean()) == float(rec["E_total"]),
    }
    seq["reproduces_production_walk_exactly"] = all(checks.values())
    seq["checks"] = checks
    secs = time.time() - t0
    if verbose:
        print(f"    [{label}] walk {rec['n']} snapshots in {secs:.0f}s   "
              f"E/ne {rec['E']:+.6f} +- {rec['E_err']:.6f}   acc {rec['acc']:.4f}"
              f"   sequence reproduces production_walk exactly: "
              f"{seq['reproduces_production_walk_exactly']}", flush=True)
        if not seq["reproduces_production_walk_exactly"]:
            print(f"      !! the reconstruction did NOT reproduce the production "
                  f"moments: {[k for k, v in checks.items() if not v]}")
    return rec, seq, secs


def compare_walk(rec, legacy, ne):
    """The clean walk's moments against the campaign reporter's, field by field."""
    if legacy is None:
        return None
    rows = {}
    for lk, ck, what in WALK_ORACLE:
        w = legacy.get(lk)
        c = rec.get(ck)
        rows[lk] = {"legacy": w, "clean": c, "unit": what,
                    "dev": (None if (w is None or c is None) else float(abs(c - w)))}
    # n, and the two error conventions side by side.  `E_err` on the legacy side
    # is the TOTAL's error; the clean `E_err` is per electron.
    rows["n"] = {"legacy": legacy.get("n"), "clean": rec.get("n"), "unit": "count",
                 "dev": (None if legacy.get("n") is None
                         else float(abs(rec["n"] - legacy["n"])))}
    rows["E_err_total"] = {
        "legacy": legacy.get("E_err"), "clean": rec.get("E_err", 0.0) * ne,
        "unit": "total",
        "dev": (None if legacy.get("E_err") is None
                else float(abs(rec["E_err"] * ne - legacy["E_err"])))}
    rows["E_err_naive_total"] = {
        "legacy": legacy.get("E_err_naive"),
        "clean": rec.get("E_err_naive", 0.0) * ne, "unit": "total",
        "dev": (None if legacy.get("E_err_naive") is None
                else float(abs(rec["E_err_naive"] * ne - legacy["E_err_naive"])))}
    return rows


def run_arm(S, si, proto, label):
    """One SR replay at the Setup's own kappa.  Returns (hist, theta, seconds, ...)."""
    c0, v0 = S.seed_state(si)
    dig = _init_digest(v0, c0)
    want = SEEDS[si][1]
    if dig != want:
        raise SystemExit(
            f"seed {si}: the clean start digests to {dig} but the frozen tag "
            f"says {want}.  The initialization is NOT the legacy one -- run "
            f"tools/recon_rs75_init.py first (Blocker B / B4).  Refusing to "
            f"compare SR trajectories between two different starting points.")
    th0 = S.theta(c0, v0)
    R0 = S.crystal_R0()
    mk = S.maker(2)
    t0 = time.time()
    th, hist = sr.sr_optimize_joint(
        mk, th0, R0, steps=proto["sr_steps"], nsweep=proto["sr_sweeps"],
        sigma=SR_SIGMA, seed=si, snapshot_every=proto["sr_snap"],
        equil=proto["sr_equil"], target_acc=SR_TARGET)
    secs = time.time() - t0
    print(f"    [{label}] SR {proto['sr_steps']}x{proto['sr_sweeps']} in {secs:.0f}s"
          f"   E_start {hist[0]['E'] / S.ne:+.6f} -> "
          f"E_final {hist[-1]['E'] / S.ne:+.6f}", flush=True)
    return hist, th, secs, dig, R0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0,
                    help="conv-tier seed (0..4); default 0 = the most complete "
                         "provenance (frozen pkl in the manifest-covered store)")
    ap.add_argument("--kappa-mode", choices=("legacy", "physical", "both"),
                    default="both")
    ap.add_argument("--budget", choices=("regression", "smoke"), default="regression",
                    help="`smoke` proves the plumbing in ~1 min and is NOT a "
                         "comparison -- its trajectory is far shorter")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the SR step count.  A SHORT run is the "
                         "escalation probe: step 0 is comparable to the frozen "
                         "step 0 (tau_0 = tau in every schedule), so a 10-step "
                         "run answers 'does the SR kernel agree?' in minutes "
                         "instead of half an hour")
    ap.add_argument("--sweeps", type=int, default=None,
                    help="override the SR sweeps per step.  Changing this does "
                         "NOT change step 0's tau, but it does change the sample "
                         "step 0 is estimated from, so step 0 is only comparable "
                         "at the legacy 400")
    ap.add_argument("--walk", action="store_true",
                    help="also run the B7 production walk from the optimised "
                         "wavefunction and compare its moments against the "
                         "campaign reporter's.  Only meaningful at the "
                         "regression budget")
    ap.add_argument("--init", default=INIT_JSON)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    raw = json.load(open(args.init, encoding="utf-8"))
    if "kappa_init" not in raw:
        raise SystemExit(
            "initial_conditions.json has no `kappa_init`: it predates the Blocker B "
            "fix.  Run tools/recon_rs75_init.py first.")

    si = args.seed
    if not 0 <= si < len(SEEDS):
        raise SystemExit(f"--seed must be 0..{len(SEEDS) - 1}")
    L0, want_dig, where = SEEDS[si]
    proto = dict(PROTOCOL[args.budget])
    if args.steps is not None:
        proto["sr_steps"] = int(args.steps)
    if args.sweeps is not None:
        proto["sr_sweeps"] = int(args.sweeps)

    # How many steps may be compared index-by-index.  Two separate conditions,
    # and both must hold:
    #
    #   the SAMPLE: step i is estimated from `sweeps` sweeps with `equil`
    #   equilibration and a `snap` stride.  Change any of the three and even
    #   step 0 is a different estimator, so nothing is comparable.
    #
    #   the SCHEDULE: the update magnitude is `tau_i = tau/(1 + xi*i/steps)`, so
    #   `steps` enters every step AFTER the first.  A shorter run is a different
    #   algorithm from step 1 on.  Step 0 has tau_0 = tau in both, which is what
    #   makes a short run a meaningful escalation probe rather than merely cheap.
    SAME_SAMPLE = (proto["sr_sweeps"], proto["sr_snap"], proto["sr_equil"]) == (400, 4, 150)
    if not SAME_SAMPLE:
        n_cmp, why = 0, (
            f"NOTHING -- this run samples {proto['sr_sweeps']} sweeps, snap "
            f"{proto['sr_snap']}, equil {proto['sr_equil']}; the legacy record is "
            f"400 / 4 / 150.  Every step would be a different estimator")
    elif proto["sr_steps"] == 120:
        n_cmp, why = 120, "the full legacy protocol (120 x 400)"
    else:
        n_cmp, why = 1, (
            f"STEP 0 ONLY -- this run is {proto['sr_steps']} x {proto['sr_sweeps']}, "
            f"not 120 x 400, and tau_i = tau/(1+xi*i/steps) makes every later step "
            f"a different schedule.  Step 0 has tau_0 = tau in both")

    print("=" * 78)
    print("  Blocker B / B6 -- crystal SR replay at r_s = 75")
    print("=" * 78)
    print(f"  seed {si}  L0 = {L0}  frozen digest {want_dig}  "
          f"frozen state in: {where}")
    print(f"  init  {args.init}")
    print(f"  kappa artifact {raw['kappa']!r}  kappa_init {raw['kappa_init']!r}")
    print(f"  kappa legacy   {K_LEGACY}   kappa physical {K_PHYS:.12f}")
    print(f"  budget {args.budget}: SR {proto['sr_steps']}x{proto['sr_sweeps']} "
          f"equil {proto['sr_equil']} snap {proto['sr_snap']} "
          f"sigma {SR_SIGMA} target_acc {SR_TARGET}")
    print(f"  comparable steps: {n_cmp}  ({why})\n", flush=True)

    fz, fz_path = frozen_state(si)
    print(f"  frozen pkl: {fz_path}")
    if fz is None:
        print("  MISSING -- the trajectory cannot be compared.  Stopping.")
        return 2
    print(f"    hist {len(fz['hist'])} steps   theta {np.shape(fz['theta'])}   "
          f"stored kappa {fz['kappa']!r}   n_band {fz['n_band']}")
    rec = legacy_json(si)
    print(f"    campaign reporter: {'present' if rec else 'MISSING'}"
          f"{'' if rec is None else '  E_perpart %.6f' % rec['E_perpart']}\n")

    modes = ("legacy", "physical") if args.kappa_mode == "both" else (args.kappa_mode,)
    blob = {"rs": RS, "seed": si, "L0": L0, "budget": args.budget,
            "protocol": proto, "n_steps_compared": n_cmp,
            "comparable_why": why, "frozen_pkl": fz_path,
            "frozen_kappa": float(fz["kappa"]),
            "kappa_legacy": K_LEGACY, "kappa_physical": K_PHYS,
            "init_kappa": raw["kappa"], "init_kappa_init": raw["kappa_init"],
            "arms": {}}

    for mode in modes:
        if mode == "legacy":
            S = Setup(raw, kappa=K_LEGACY, kappa_mode="legacy-regression")
        else:
            S = Setup(raw, kappa=K_PHYS, kappa_mode="physical")
        print(f"  --- arm '{mode}'  kappa = {S.kappa!r} ---", flush=True)
        hist, th, secs, dig, R0 = run_arm(S, si, proto, mode)
        arm = {"kappa": float(S.kappa), "kappa_mode": S.kappa_mode,
               "init_digest": dig, "R0_sha1": hashlib_sha1(R0),
               "sr_seconds": float(secs),
               # the endpoint, kept so B8 can nest from it WITHOUT re-running
               # the 120 x 400 SR.  It is the same object `wf = mk(th)` builds,
               # so nesting from it is nesting from exactly what was measured.
               "theta": th.tolist(), "n_band": 2,
               "hist": compare_hist(hist, fz["hist"], n_cmp),
               "vs_legacy_json": compare_json(hist, rec, S.ne, n_cmp == 120)}
        blob["arms"][mode] = arm

        vj = arm["vs_legacy_json"]
        if vj:
            print("      vs the campaign reporter (independent oracle):")
            for k in ("E_start", "E_final", "E_min", "force_first",
                      "force_final", "cond_final", "theta_final"):
                r = vj[k]
                if not r["comparable"]:
                    print(f"        {k:12s} not comparable at this run length")
                elif r.get("dev") is None:
                    print(f"        {k:12s} legacy MISSING")
                elif k == "theta_final":
                    print(f"        {k:12s} max|dev| {r['dev']:.6e}   "
                          f"(len {r['clean_len']} vs {r['legacy_len']})")
                else:
                    print(f"        {k:12s} dev {r['dev']:.6e}   "
                          f"clean {r['clean']:+.9f}   legacy {r['legacy']:+.9f}")
        h = arm["hist"]
        if h.get("n_compared"):
            print(f"      vs the frozen pkl trajectory "
                  f"({h['n_compared']} step(s) compared, keys identical: "
                  f"{h['keys_identical']}):")
            print(f"        {'field':9s} {'dev@step0':>12s} {'max dev':>12s} "
                  f"{'@step':>6s}")
            for f in FIELDS:
                d = h["fields"][f]
                print(f"        {f:9s} {d['dev_step0']:12.3e} "
                      f"{d['max_abs_dev']:12.3e} {d['step_of_max']:6d}")
            t = h["theta"]
            print(f"        {'theta':9s} {t['dev_step0']:12.3e} "
                  f"{t['max_abs_dev']:12.3e} {t['step_of_max']:6d}"
                  f"   (shape {t['shape_clean']} vs {t['shape_frozen']})")
        else:
            print("      vs the frozen pkl trajectory: NOT COMPARABLE -- " + why)

        if args.walk:
            print("\n      --- B7: the production walk from this theta ---")
            if args.budget != "regression":
                print(f"        SKIPPED: budget {args.budget!r} is not the legacy "
                      f"walk protocol ({proto['meas_sweeps']} x "
                      f"{proto['meas_equil']} vs 1500 x 400)")
                arm["walk"] = {"skipped": "budget is not the legacy walk protocol"}
            else:
                wf = S.maker(2)(th)
                wrec, wseq, wsecs = walk_and_sequence(S, wf, proto, mode)
                arm["walk"] = {"record": wrec, "sequence_sha1": hashlib_sha1(wseq["E"]),
                               "n": wseq["n"],
                               "sequence_reproduces_production_walk":
                                   wseq["reproduces_production_walk_exactly"],
                               "seconds": float(wsecs),
                               "vs_legacy": compare_walk(wrec, rec, S.ne)}
                wj = arm["walk"]["vs_legacy"]
                if wj:
                    print(f"        {'moment':18s} {'dev':>12s}   "
                          f"{'clean':>16s} {'legacy':>16s}  unit")
                    for k, _ck, what in WALK_ORACLE:
                        r = wj[k]
                        if r["dev"] is None:
                            print(f"        {k:18s} {'legacy MISSING':>12s}")
                        else:
                            print(f"        {k:18s} {r['dev']:12.3e}   "
                                  f"{r['clean']:16.9f} {r['legacy']:16.9f}  {what}")
                    for k in ("n", "E_err_total", "E_err_naive_total"):
                        r = wj[k]
                        if r["dev"] is None:
                            print(f"        {k:18s} {'legacy MISSING':>12s}")
                        else:
                            print(f"        {k:18s} {r['dev']:12.3e}   "
                                  f"{r['clean']:16.9f} {r['legacy']:16.9f}  {r['unit']}")
                print("        the legacy record stores MOMENTS of the walk, not the "
                      "sequence; the clean sequence was\n"
                      "        recovered by re-driving sample() and is accepted only "
                      "because it reproduces these\n"
                      "        moments exactly.")
        print(flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print(f"  wrote {args.out}")
    return 0


def hashlib_sha1(a):
    return _h.sha1(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
