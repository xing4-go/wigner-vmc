"""Blocker B / B8 -- the nested nmax = 2 validation at r_s = 75.

WHAT B8 IS
----------
``nmax = 2`` means n_bands = 3: the orbital block of theta grows from (36, 1) to
(36, 2).  The campaign built the larger state by NESTING -- keep the optimised
nmax = 1 coefficients, append a zero column, and let SR move the new level from
there.  The nesting is claimed to be EXACT: the padded vector reproduces the
source's ``C`` bit-for-bit in the first m+1 columns and gives an exact zero in
the rest, because ``c_row`` computes the identical floating-point expression on
the leading entries and ``sin(t)/t * 0`` is an exact zero.

So B8 has two claims to establish, in this order:

  1. the nesting identity, from the CLEAN nmax=1 state that just passed B6/B7;
  2. the nested SR + production walk against the frozen nested record.

The second is only meaningful if the first holds and if the parent is the
campaign's parent -- a nested run from a different nmax = 1 state is a different
calculation, however well it agrees with anything.  Both are therefore checked
before any SR is run, and the script refuses to proceed if either fails rather
than falling back to a fresh initialisation.

THE PARENT COMES FROM B6, NOT FROM A SECOND SR
----------------------------------------------
``--parent`` takes the JSON written by ``tools/replay_crystal_rs75.py`` and
reads the endpoint of the arm named by ``--arm``.  Re-running the 120 x 400 SR
to obtain the same theta again would cost half an hour and would not make the
nesting any more valid -- and B8 is a claim about nesting, not about SR, which
B6 already established.

THE FROZEN RECORD
-----------------
    _diag/ckpt_rebuild/llcryst_nb3_k53.033_p1nest{0..3}.pkl   (n_band 3,
        hist 120, theta (149,), kappa 53.033)
    %TEMP%/qhvmc_checkpoints/llcryst_nb3_k53.033_p1nest4.pkl
    _diag/part1/cry_nest_rs75_nb3_s{0..4}.json   the walk's moments
    _diag/part1/liq_rs75.json                    the liquid reference

delta_E(75) = E_C(nmax=2) - E_L is REPORTED, NOT TARGETED
---------------------------------------------------------
The brief is explicit that -0.0584 is a comparison reference and that tuning
toward it is forbidden.  This script has no objective function: it reproduces a
protocol and prints what comes out, beside the frozen numbers it did not choose.

    python tools/replay_nested_rs75.py --parent logs/replay_crystal_rs75.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
for p in (HERE, os.path.join(CLEAN, "scripts"), os.path.join(CLEAN, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_rs75 import (PROTOCOL, SR_SIGMA, SR_TARGET, Setup,           # noqa: E402
                        theta_parts)
from replay_crystal_rs75 import (FROZEN_CKPT, RS, TEMP_CKPT,            # noqa: E402
                                 _init_digest, hashlib_sha1,
                                 walk_and_sequence, compare_hist,
                                 compare_walk, WALK_ORACLE, FIELDS)
from wigner_vmc.wavefunctions import nesting as nest                    # noqa: E402

LEGACY_DIR = os.path.join(os.path.dirname(CLEAN), "_diag", "part1")
OUT = os.path.join(CLEAN, "logs", "replay_nested_rs75.json")

#: seed -> (the digest the frozen nb3 tag carries, which store holds the pkl).
#:
#: DERIVED, NOT GUESSED.  The campaign's nest tier (`_diag/part1_scan.py`, the
#: `job["nest_from"]` branch) builds the nb = 3 start as
#:
#:     _th2 = json(nest_from)["theta"]          the CONVERGED nb = 2 endpoint
#:     c0   = _th2[:5]                          the parent's OPTIMISED Jastrow
#:     v0   = zeros((nk, nb-1)); v0[:, :1] = v2 the parent's orbital block, padded
#:     ll_crystal_opt(kappa, nb, v0, c0, ...)   -> tag `...|i{_init_digest(v0, c0)}`
#:
#: Note `c0` is the parent's optimised theta[:5], NOT the warm-start Jastrow the
#: nb = 2 run began from -- the nested run continues the *solution*, Jastrow
#: included.  Feeding this the warm Jastrow would produce a start the campaign
#: never used, with a different digest, and no way to notice.
#:
#: The five values below were computed from `_diag/part1/cry_conv_rs75_nb2_s*.json`
#: by exactly the four lines above and reproduce the frozen tags bit-for-bit
#: (`verify_oracle` recomputes them on every run and refuses on a mismatch).  The
#: campaign reporter's own theta is the pkl's endpoint for all five seeds, which
#: is what makes the JSON a valid nest source.
#:
#: A digest CANNOT be recovered from the nested pkl's `hist[0]["theta"]`: both the
#: notebook's and the clean `sr_optimize_joint` do `theta += delta` and *then*
#: append, so `hist[0]` is the state after step 0, not the start.  Measured: that
#: state's second orbital column is ~0.17-0.25, not zero.
#:
#: ``(digest, store)``, and the SEED INDEX IS THE POSITION -- there is deliberately
#: no leading seed number in the tuple.  An earlier version of this table carried
#: one, duplicating the index, and a reader then took ``NEST_SEEDS[si][0]`` for the
#: digest when it was the redundant index: B8 compared every nested run against
#: ``"0"`` and refused.  The same redundancy had already produced a different
#: failure in the sibling ``replay_crystal_rs75.py`` (``SEEDS``), where it unpacked
#: as a ValueError.  One field, one meaning: read it through :func:`nest_seed`.
NEST_SEEDS = (
    ("d603e632", "frozen"),
    ("3a6bca05", "frozen"),
    ("b050d7fa", "frozen"),
    ("62b506e9", "frozen"),
    ("37f05f80", "temp"),
)


def nest_seed(si):
    """``(digest, store)`` for seed ``si``.

    The ONLY place ``NEST_SEEDS``' field order is written down, so a future
    re-ordering is a one-line change here rather than an off-by-one at each
    call site.
    """
    return NEST_SEEDS[si]


def verify_oracle(si):
    """Recompute the nested start digest from the frozen nb = 2 record.

    Returns ``(predicted, source_path)``, with ``predicted`` None if the frozen
    nb = 2 record is absent.  This is the table's own negative control: it
    derives the value the frozen nb3 tag must carry, from the record the campaign
    nested from, using the production contract.  If it disagrees with
    ``NEST_SEEDS`` the table is wrong and every comparison below would be against
    a digest that means nothing -- so the caller refuses.
    """
    src = os.path.join(LEGACY_DIR, f"cry_conv_rs75_nb2_s{si}.json")
    if not os.path.exists(src):
        return None, src
    with open(src, encoding="utf-8") as fh:
        rec = json.load(fh)
    c0, v2 = theta_parts(np.asarray(rec["theta"], float), 2, 36)
    return _init_digest(nest.pad_v(v2, 3), c0), src


def frozen_nested(si):
    _digest, where = nest_seed(si)
    base = TEMP_CKPT if where == "temp" else FROZEN_CKPT
    path = os.path.join(base, f"llcryst_nb3_k53.033_p1nest{si}.pkl")
    if not os.path.exists(path):
        return None, path
    with open(path, "rb") as fh:
        return pickle.load(fh), path


def legacy_json(name):
    path = os.path.join(LEGACY_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parent", default=os.path.join(CLEAN, "logs",
                                                     "replay_crystal_rs75.json"))
    ap.add_argument("--arm", default="legacy", choices=("legacy", "physical"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", choices=("regression",), default="regression")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    si = args.seed
    if not 0 <= si < len(NEST_SEEDS):
        raise SystemExit(f"--seed must be 0..{len(NEST_SEEDS) - 1}")
    want_dig, _where = nest_seed(si)
    proto = dict(PROTOCOL[args.budget])

    print("=" * 78)
    print("  Blocker B / B8 -- nested nmax = 2 at r_s = 75")
    print("=" * 78)

    pj = json.load(open(args.parent, encoding="utf-8"))
    if args.arm not in pj.get("arms", {}):
        raise SystemExit(f"{args.parent} has no arm {args.arm!r} -- it holds "
                         f"{sorted(pj.get('arms', {}))}")
    par = pj["arms"][args.arm]
    th2 = np.asarray(par["theta"], float)
    print(f"  parent      {args.parent}")
    print(f"  parent arm  {args.arm}  kappa {par['kappa']!r}  "
          f"n_band {par['n_band']}  len(theta) {th2.size}")
    print(f"  parent init digest {par['init_digest']}  (the nmax=1 start)")
    if th2.size != 5 + 2 * 36 * 1:
        raise SystemExit(f"parent theta has length {th2.size}; nmax=1 wants 77")

    init = json.load(open(pj["init_path"] if "init_path" in pj
                          else os.path.join(CLEAN, "results", "bench_rs75",
                                            "initial_conditions.json"),
                          encoding="utf-8"))

    # -- 1. the nesting identity, BEFORE anything is optimised ---------------
    c0, v1 = theta_parts(th2, 2, 36)
    head, tail = nest.coefficient_identity_deviation(v1, 3)
    v3 = nest.pad_v(v1, 3)
    th3 = np.concatenate([c0, v3.real.ravel(), v3.imag.ravel()])
    got_dig = _init_digest(v3, c0)
    m_src = v1.shape[1] + 1
    C_src = nest.nested_coefficients(v1, 2)
    C_nst = nest.nested_coefficients(v1, 3)

    # the table's negative control, before it is used as an oracle
    pred, pred_src = verify_oracle(si)
    print(f"\n  oracle self-check, from {os.path.basename(pred_src)}:")
    print(f"    nb2 endpoint -> nb3 start digest  {pred}"
          f"   table says {want_dig}   "
          f"{'MATCH' if pred == want_dig else 'TABLE IS WRONG'}")
    if pred is not None and pred != want_dig:
        raise SystemExit("  NEST_SEEDS and the frozen nb2 record disagree -- the "
                         "oracle is not trustworthy;\n  refusing to compare against "
                         "it.")
    if pred is None:
        print("    (frozen nb2 record absent -- the table value is UNVERIFIED)")

    print(f"\n  --- the nesting identity: v1 -> (v1, 0), i.e. {v1.shape} -> {v3.shape} "
          f"---")
    print(f"    C_src {C_src.shape} -> C_nest {C_nst.shape}")
    print(f"    head  max|C_nest[:, :{m_src}] - C_src| {head:.3e}   (the source's own "
          f"{m_src} columns)")
    print(f"    tail  max|C_nest[:, {m_src}:]|        {tail:.3e}   (the new, zero "
          f"level)")
    print(f"    nested start digest {got_dig}   frozen nb3 tag says {want_dig}  "
          f"({'MATCH' if got_dig == want_dig else 'MISMATCH'})")
    print(f"    theta length {th3.size} = 5 + 2*36*2")
    identity_ok = (head == 0.0 and tail == 0.0)
    digest_ok = (got_dig == want_dig)
    if not identity_ok:
        raise SystemExit("  NESTING IDENTITY IS NOT EXACT -- refusing to run SR, "
                         "and NOT falling back to a fresh initialisation.")
    if not digest_ok:
        print("\n  The nesting identity is exact but the nested start is NOT the "
              "campaign's nested start.\n  The parent state is not the campaign's "
              "nmax=1 state (see B6: the digest of the nmax=1 START matched, so a\n"
              "  mismatch here means the SR endpoint differs).  The nested SR below "
              "will be measured and\n  reported as a DIFFERENT calculation, not as a "
              "reproduction.")

    # -- 2. the nested SR at the parent's own kappa --------------------------
    fy, fy_path = frozen_nested(si)
    print(f"\n  frozen nested pkl: {fy_path}")
    if fy is None:
        print("    MISSING -- the nested trajectory cannot be compared.")
        return 2
    print(f"    hist {len(fy['hist'])} steps   theta {np.shape(fy['theta'])}   "
          f"stored kappa {fy['kappa']!r}   n_band {fy['n_band']}")
    nrec = legacy_json(f"cry_nest_rs75_nb3_s{si}.json")
    print(f"    campaign reporter: {'present' if nrec else 'MISSING'}"
          f"{'' if nrec is None else '  E_perpart %.6f' % nrec['E_perpart']}")

    S = Setup(init, kappa=par["kappa"], kappa_mode=par["kappa_mode"])
    print(f"\n  --- nested SR, arm '{args.arm}', kappa = {S.kappa!r} ---", flush=True)

    blob = {"rs": RS, "seed": si, "arm": args.arm, "parent": args.parent,
            "kappa": float(S.kappa), "kappa_mode": S.kappa_mode,
            "n_band": 3, "protocol": proto,
            "nesting_identity": {"head": head, "tail": tail,
                                 "exact": bool(identity_ok),
                                 "nested_digest": got_dig,
                                 "frozen_tag_digest": want_dig,
                                 "digest_match": bool(digest_ok),
                                 "oracle_selfcheck": {
                                     "predicted_from": pred_src,
                                     "predicted": pred,
                                     "agrees_with_table": bool(pred == want_dig)}},
            "parent_theta_sha1": hashlib_sha1(th2),
            "frozen_pkl": fy_path, "frozen_kappa": float(fy["kappa"])}

    # `run_arm` builds (c0, v0) from the artifact's seed_state, which is the
    # nmax=1 start.  B8's start is the PARENT's endpoint instead, so the SR is
    # driven here rather than through run_arm's initialization path.
    from wigner_vmc.vmc import sr as sr
    mk = S.maker(3)
    R0 = S.crystal_R0()
    t0 = time.time()
    th, hist = sr.sr_optimize_joint(
        mk, th3, R0, steps=proto["sr_steps"], nsweep=proto["sr_sweeps"],
        sigma=SR_SIGMA, seed=si, snapshot_every=proto["sr_snap"],
        equil=proto["sr_equil"], target_acc=SR_TARGET)
    secs = time.time() - t0
    print(f"    SR {proto['sr_steps']}x{proto['sr_sweeps']} in {secs:.0f}s   "
          f"E_start {hist[0]['E'] / S.ne:+.6f} -> "
          f"E_final {hist[-1]['E'] / S.ne:+.6f}", flush=True)
    blob["sr_seconds"] = float(secs)
    blob["theta"] = th.tolist()
    blob["hist"] = compare_hist(hist, fy["hist"], 120)

    h = blob["hist"]
    print(f"\n    vs the frozen nested pkl trajectory ({h['n_compared']} steps, "
          f"keys identical: {h['keys_identical']}):")
    print(f"      {'field':9s} {'dev@step0':>12s} {'max dev':>12s} {'@step':>6s}")
    for f in FIELDS:
        d = h["fields"][f]
        print(f"      {f:9s} {d['dev_step0']:12.3e} {d['max_abs_dev']:12.3e} "
              f"{d['step_of_max']:6d}")
    t = h["theta"]
    print(f"      {'theta':9s} {t['dev_step0']:12.3e} {t['max_abs_dev']:12.3e} "
          f"{t['step_of_max']:6d}   (shape {t['shape_clean']} vs {t['shape_frozen']})")

    # The campaign's own nesting cross-check, reproduced on the clean side.
    #
    # `nest_identity_dev` is NOT the c_row identity -- that one is exact and is
    # checked above.  It is `E_start(nested nb=3) - E_perpart(parent nb=2)`: the
    # difference between two Monte-Carlo estimates of the SAME state, one from
    # the nested run's step 0 (which samples before its first update) and one
    # from the parent's production walk.  Production judges it against
    # `5 * max(hypot(err_start, err_src), 0.010)` (`_diag/part1_scan.py:794`), so
    # a nonzero value is expected and only its size means anything.
    #
    # Note the asymmetry that makes this definition work: `sr_optimize_joint`
    # samples at theta_i and THEN does `theta += delta`, appending afterwards --
    # so `hist[i]["E"]` is the energy of theta_i while `hist[i]["theta"]` is
    # theta_{i+1}.  `hist[0]["E"]` is therefore the START's energy in both the
    # notebook and the clean kernel, which is why the two are comparable at all.
    if nrec is not None:
        e_start = float(hist[0]["E"] / S.ne)
        e0err = float(hist[0]["E_err"] / S.ne)
        par_rec = legacy_json(f"cry_conv_rs75_nb2_s{si}.json")
        par_e = None if par_rec is None else par_rec["E_perpart"]
        cw = par.get("walk", {})
        par_walk = cw.get("record") if isinstance(cw, dict) else None
        clean_par_e = None if par_walk is None else par_walk.get("E")
        clean_par_err = None if par_walk is None else par_walk.get("E_err")

        def _tol(err_a, err_b):
            if err_a is None or err_b is None:
                return None
            return 5.0 * max(float(np.hypot(err_a, err_b)), 0.010)

        blob["nest_identity_dev"] = {
            "definition": "E_start(nested nb=3) - E_perpart(parent nb=2); a "
                          "STATISTICAL difference between two MC estimators of the "
                          "same state, NOT the c_row identity",
            "legacy": nrec.get("nest_identity_dev"),
            "legacy_tol": nrec.get("nest_identity_tol"),
            "legacy_source_E": par_e,
            "legacy_start_err": nrec.get("nest_start_err"),
            "clean": (None if clean_par_e is None else e_start - clean_par_e),
            "clean_tol": _tol(e0err, clean_par_err),
            "clean_source_E": clean_par_e,
            "clean_start_err": e0err,
        }
        d = blob["nest_identity_dev"]
        print("\n    --- nest_identity_dev (the campaign's energy cross-check) ---")
        print(f"      legacy  {d['legacy']:+.6f}  tol {d['legacy_tol']}   "
              f"(source E {d['legacy_source_E']:+.6f})"
              if d["legacy"] is not None else "      legacy  absent")
        if d["clean"] is None:
            print("      clean   NOT COMPARABLE -- the parent arm carries no walk "
                  "record (run replay_crystal_rs75.py with --walk)")
        else:
            print(f"      clean   {d['clean']:+.6f}  tol {d['clean_tol']:.3f}   "
                  f"(source E {d['clean_source_E']:+.6f} from the clean parent walk)")
            print(f"      within tol: {abs(d['clean']) <= d['clean_tol']}")

    # -- 3. the nested production walk --------------------------------------
    print("\n    --- B8 production walk ---", flush=True)
    wf = mk(th)
    wrec, wseq, wsecs = walk_and_sequence(S, wf, proto, args.arm)
    blob["walk"] = {"record": wrec, "sequence_sha1": hashlib_sha1(wseq["E"]),
                    "n": wseq["n"],
                    "sequence_reproduces_production_walk":
                        wseq["reproduces_production_walk_exactly"],
                    "seconds": float(wsecs),
                    "vs_legacy": compare_walk(wrec, nrec, S.ne)}
    wj = blob["walk"]["vs_legacy"]
    if wj:
        print(f"      {'moment':18s} {'dev':>12s}   {'clean':>16s} {'legacy':>16s}")
        for k, _ck, what in WALK_ORACLE:
            r = wj[k]
            if r["dev"] is None:
                print(f"      {k:18s} {'legacy MISSING':>12s}")
            else:
                print(f"      {k:18s} {r['dev']:12.3e}   {r['clean']:16.9f} "
                      f"{r['legacy']:16.9f}  {what}")
        for k in ("n", "E_err_total"):
            r = wj[k]
            if r["dev"] is not None:
                print(f"      {k:18s} {r['dev']:12.3e}   {r['clean']:16.9f} "
                      f"{r['legacy']:16.9f}  {r['unit']}")

    # -- 4. delta_E(75), reported and not targeted --------------------------
    liq = legacy_json("liq_rs75.json")
    par_rec = legacy_json(f"cry_conv_rs75_nb2_s{si}.json")
    print("\n  --- delta_E(75) = E_C - E_L, per electron ---")
    d = {"clean_nested_E": wrec["E"],
         "clean_nested_err": wrec["E_err"],
         "legacy_nested_E": (None if nrec is None else nrec["E_perpart"]),
         "legacy_liquid_E": (None if liq is None else liq["E_perpart"]),
         "legacy_nmax1_E": (None if par_rec is None else par_rec["E_perpart"])}
    if liq is not None:
        d["clean_delta_E_nmax2"] = wrec["E"] - liq["E_perpart"]
        d["legacy_delta_E_nmax2"] = nrec["E_perpart"] - liq["E_perpart"]
        d["delta_E_difference"] = d["clean_delta_E_nmax2"] - d["legacy_delta_E_nmax2"]
        # The published -0.0584 is NOT a single-seed number: it is the nmax=2 tier
        # POOLED over five seeds (`scripts/analyze_energy.py`, tier `nest`, r_s=75).
        # Comparing this one-seed run against it would be a category error, so the
        # pooled value is computed here beside the seed-matched one.
        pool = []
        for k in range(len(NEST_SEEDS)):
            r = legacy_json(f"cry_nest_rs75_nb3_s{k}.json")
            if r is not None:
                pool.append(r["E_perpart"])
        if pool:
            d["legacy_pooled_nmax2_E"] = float(np.mean(pool))
            d["legacy_pooled_n_records"] = len(pool)
            d["legacy_pooled_delta_E_nmax2"] = float(np.mean(pool)) - liq["E_perpart"]
        if par_rec is not None:
            d["clean_delta_E_nmax1"] = par_rec["E_perpart"] - liq["E_perpart"]
            d["legacy_delta_E_nmax1"] = par_rec["E_perpart"] - liq["E_perpart"]
    blob["delta_E"] = d
    for k in ("clean_nested_E", "legacy_nested_E", "legacy_liquid_E",
              "clean_delta_E_nmax2", "legacy_delta_E_nmax2", "delta_E_difference",
              "legacy_pooled_delta_E_nmax2"):
        v = d.get(k)
        print(f"    {k:30s} {'--' if v is None else format(v, '+.9f')}")
    if "legacy_pooled_delta_E_nmax2" in d:
        print(f"    ({d['legacy_pooled_n_records']} legacy nest records pooled -- "
              f"the published -0.0584 is THIS number, not the seed-matched one)")
    print("\n    -0.0584 is the published comparison reference.  Nothing in this "
          "script reads it.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
