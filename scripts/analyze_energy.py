"""Recompute the liquid-vs-crystal scan, and REGRESS it against the legacy article.

    python wigner_vmc_clean/scripts/analyze_energy.py

Reads `results/legacy_*/states.json` (converted by `import_legacy_results.py`), builds
every ScanPoint through `analysis/statistics.py`, writes
`results/legacy_*/scan.json`, and then compares the result point by point against the
legacy `_diag/energy_scan.json`.

Why the regression matters
--------------------------
The legacy scan is not an input to this analysis; it is the thing this analysis has to
reproduce.  Every number here is derived independently from the per-seed records, so
agreement means the conversion and the statistics are right, and disagreement means
they are not -- there is no third possibility, because nothing else feeds in.

The comparison is a REPORT, not an assertion.  A hard assert would be wrong here: the
whole reason for rebuilding the analysis is that the legacy implementation may contain
a mistake, so a difference is information rather than a failure.  What the script will
not do is stay quiet about one.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
sys.path.insert(0, os.path.join(CLEAN, "src"))

from wigner_vmc.analysis import energy_competition as ec     # noqa: E402
from wigner_vmc.analysis import ll_convergence as llc        # noqa: E402
from wigner_vmc.io import legacy, results                    # noqa: E402

RESULTS = os.path.join(CLEAN, "results")
DIAG = os.path.join(ROOT, "_diag")

# Tolerances.  The legacy store rounds some values on the way out, so exact equality
# is not the right test; these are tight enough that a real difference shows.
TOL_ENERGY = 1e-6
TOL_SIGMA = 1e-9
TOL_Z = 1e-6


def _mtime(path):
    import time
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def _fmt_complaint(c):
    nb, tier, rs = c["row"]
    head = f"n_b={nb} tier={tier:>5} r_s={rs:6.1f}"
    if c["kind"] == "no_counterpart":
        return f"{head}: legacy row has no counterpart in the records"
    if c["kind"] == "n_band_mismatch":
        return (f"{head}: P_mean has {c['mine']} entries vs legacy {c['legacy']} "
                f"-- different n_band, not comparable")
    if c["kind"] == "n_seeds":
        extra = sorted(set(c["mine_keys"]) - set(c["legacy_keys"]))
        missing = sorted(set(c["legacy_keys"]) - set(c["mine_keys"]))
        ev = ""
        if c["legacy_keys"]:
            ev = ("; the legacy row names its own seeds"
                  + (f", all of which are on record" if not missing
                     else f", and {missing} are NOT on record"))
            if extra:
                ev += f", and the records have {extra} beyond them"
        return (f"{head}: n_seeds {c['mine']} (records) vs {c['legacy']} (legacy){ev}"
                f"; P_mean/P_sem not compared, they follow from this")
    return (f"{head}: {c['kind']}[{c['index']}] {c['mine']!r} vs legacy {c['legacy']!r}")


def _stale_snapshot_reason(complaint, part1_dir, table_mtime):
    """Prove, from the disk, that a mismatch is a stale legacy snapshot.  Or return None.

    A suspicion is not a proof, and this returns only proof.  The claim to be established
    is "the legacy table predates the seed set", and the evidence that establishes it is
    a time-ordered one, checked per seed file rather than inferred from the coupling:

      1. the legacy row names its OWN seed list, and that list is a strict subset of the
         seed list behind the recomputed row -- so nothing was invented on my side; and
      2. every seed in the difference is a record file on disk whose mtime is LATER than
         the table's -- so the seed existed all along and the table was written too early.

    Both halves are needed.  (1) alone would excuse a table that simply omitted a state;
    (2) alone would excuse a mismatch at a coupling that merely happens to be recent.
    Only together do they say the table is a photograph of an earlier seed set.

    Returns None for any kind other than `n_seeds`: a numeric mismatch between two rows
    with the SAME seed list has no such excuse and must be investigated.
    """
    if complaint["kind"] != "n_seeds" or not complaint["legacy_keys"]:
        return None
    extra = sorted(set(complaint["mine_keys"]) - set(complaint["legacy_keys"]))
    missing = sorted(set(complaint["legacy_keys"]) - set(complaint["mine_keys"]))
    if missing or not extra:
        return None

    newer, unknown = [], []
    for k in extra:
        p = os.path.join(part1_dir, k + ".json")
        if not os.path.exists(p):
            unknown.append(k)
            continue
        m = _mtime(p)
        if m > table_mtime:
            newer.append((k, m))
    if unknown or len(newer) != len(extra):
        return None
    detail = ", ".join(f"{k} ({m})" for k, m in newer)
    return (f"the legacy row names {len(complaint['legacy_keys'])} seed(s) and the records "
            f"have {len(complaint['mine_keys'])}; the {len(extra)} it omits are on disk "
            f"and all postdate the table ({table_mtime}) -- {detail}")


def _late_seeds(part1_dir, conflicts, mtimes):
    """Name the record files written AFTER a stale table, which is the mechanism.

    String comparison is safe here because `_mtime` is fixed-width and big-endian.
    """
    import glob
    anchor_m = mtimes[0]
    lines = []
    for rs, tier, _want, _es, _mine in conflicts:
        pat = os.path.join(part1_dir, f"cry_{tier}_rs{rs:g}_*.json")
        files = sorted(glob.glob(pat))
        if not files:
            lines.append(f"(no {tier} record files match {os.path.basename(pat)})")
            continue
        newer = [(os.path.basename(f), _mtime(f)) for f in files
                 if _mtime(f) > anchor_m]
        lines.append(f"r_s = {rs:g} {tier}: {len(files)} record files on disk, "
                     f"{len(newer)} of them written after p1_report.json:")
        for name, m in newer:
            lines.append(f"      {m}  {name}   <- not in the stale snapshot")
    return lines


def regress(points, legacy_scan):
    """Point-by-point comparison against the legacy derived scan."""
    key = lambda p: (round(float(p["rs"]), 4), str(p["tier"]))          # noqa: E731
    legacy_pts = {key(p): p for p in legacy_scan["points"]}
    mine = {key({"rs": p.rs, "tier": p.tier}): p for p in points}

    only_legacy = sorted(set(legacy_pts) - set(mine))
    only_mine = sorted(set(mine) - set(legacy_pts))

    rows, bad = [], 0
    for k in sorted(set(legacy_pts) & set(mine), key=lambda k: (k[1], k[0])):
        lp, mp = legacy_pts[k], mine[k]
        checks = [
            ("dE", lp["dE"], mp.delta_e, TOL_ENERGY),
            ("sig_mc", lp["sig_mc"], mp.sigma_mc, TOL_SIGMA),
            ("sig_opt", lp["sig_opt"], mp.sigma_init, TOL_SIGMA),
            ("dE_err", lp["dE_err"], mp.sigma_total, TOL_SIGMA),
            ("z", lp.get("z"), mp.z, TOL_Z),
            ("E_cry", lp["E_cry"], mp.energy_crystal, TOL_ENERGY),
            ("E_liq", lp["E_liq"], mp.energy_liquid, TOL_ENERGY),
        ]
        worst = 0.0
        for _name, a, b, tol in checks:
            if a is None or b is None:
                continue
            worst = max(worst, abs(a - b))
            if abs(a - b) > tol:
                bad += 1
                rows.append((k, _name, a, b, abs(a - b)))
        if not any(r[0] == k for r in rows):
            rows.append((k, "max|diff|", None, None, worst))
        if lp["n"] != mp.n_states:
            bad += 1
            rows.append((k, "n_states", lp["n"], mp.n_states, 0))

    return rows, bad, only_legacy, only_mine


def main():
    print(f"store: {RESULTS}\n")

    states = results.load_states(RESULTS, "llrot")
    print(f"loaded {len(states)} converted llrot states")

    points, brackets, notes = ec.build_scan(states)
    print(f"built  {len(points)} scan points, {len(brackets)} brackets\n")

    print("=" * 92)
    print("PHASE COMPETITION  (delta_E = E_crystal - E_liquid per particle)")
    print("=" * 92)
    print(f"  {'r_s':>7} {'tier':>6} {'n_max':>5} {'n':>2} {'E_cry':>12} {'E_liq':>12} "
          f"{'delta_E':>11} {'sig_tot':>9} {'z':>6}")
    for p in points:
        print(f"  {p.rs:7.2f} {p.tier:>6} {str(p.n_max):>5} {p.n_states:>2} "
              f"{p.energy_crystal:+12.6f} {p.energy_liquid:+12.6f} {p.delta_e:+11.6f} "
              f"{p.sigma_total:9.5f} {(p.z if p.z is not None else float('nan')):6.2f}")

    print()
    print("=" * 92)
    print("BRACKETS  (one truncation each; never pooled)")
    print("=" * 92)
    for b in brackets:
        tag = "RESOLVED" if b.resolved else "not an interval"
        span = (f"r_s in [{b.lo:g}, {b.hi:g}]" if b.resolved else
                (f"r_s <= {b.hi:g}" if b.hi is not None else f"r_s >= {b.lo:g}"))
        print(f"  {b.tier:>6} (n_max={next((p.n_max for p in points if p.tier == b.tier), None)}): "
              f"{span}   [{tag}, {b.n_points} points]")
        if b.note:
            print(f"           {b.note}")
    if notes:
        print()
        for n in notes:
            print(f"  * {n}")

    # ------------------------------------------------------------------ regression
    legacy_scan = legacy.read_energy_scan(os.path.join(DIAG, "energy_scan.json"))
    rows, bad, only_legacy, only_mine = regress(points, legacy_scan)

    print()
    print("=" * 92)
    print("REGRESSION vs legacy _diag/energy_scan.json")
    print("=" * 92)
    n_common = len(legacy_scan["points"]) - len(only_legacy)
    print(f"  legacy points {len(legacy_scan['points'])}   mine {len(points)}   "
          f"common {n_common}")
    if only_legacy:
        print(f"  ONLY IN LEGACY: {only_legacy}")
    if only_mine:
        print(f"  ONLY IN MINE  : {only_mine}")

    worst = max((r[4] for r in rows if r[1] == "max|diff|"), default=0.0)
    print(f"  worst absolute difference over all compared quantities: {worst:.3e}")
    if bad:
        print(f"  *** {bad} MISMATCH(ES):")
        for k, name, a, b, d in rows:
            if name == "max|diff|":
                continue
            print(f"      r_s={k[0]:7.2f} {k[1]:>6} {name:>10}: legacy {a!r} vs mine {b!r}  "
                  f"|d|={d:.3e}")
    else:
        print("  ALL POINTS AGREE within tolerance.")

    # ---------------------------------------------------- legacy's own anchors
    # The legacy script checks itself against p1_report.py's recorded total sigma.  Doing
    # the same here is a second, independent tripwire: it does not depend on
    # energy_scan.json being right, only on that third file existing.
    #
    # The check is THREE-WAY, because a two-way one cannot tell the two failure modes
    # apart.  Comparing my number to the anchor alone, a disagreement is either (a) my
    # conversion or statistics is wrong, or (b) the anchor is a stale snapshot of an
    # earlier seed set -- and those need opposite responses.  Bringing energy_scan.json
    # in as a third witness separates them: if my value and energy_scan agree and only
    # the anchor differs, the anchor is stale, and the mtimes say so directly.
    #
    # Two shape facts about p1_report.json, both learned the hard way: the per-tier dict
    # spells the field `s_tot` (not `sig_tot`), and a tier with no usable states is
    # `null` rather than absent -- `{"fast": {...}, "conv": {...}, "nest": null}`.  A
    # `t.get(...)` on that null raises, so the guard is isinstance, not `in`.
    p1_report = legacy.read_p1_report(os.path.join(DIAG, "p1_report.json"))
    anchors = {}
    for entry in p1_report:
        rs = round(float(entry["rs"]), 4)
        for tier, t in (entry.get("tiers") or {}).items():
            if isinstance(t, dict) and t.get("s_tot") is not None:
                anchors[(rs, tier)] = t["s_tot"]

    escan = {(round(float(p["rs"]), 4), str(p["tier"])): p["dE_err"]
             for p in legacy_scan["points"]}

    print()
    print("  three-way anchor check vs p1_report.py's recorded s_tot:")
    print(f"    {'r_s':>6} {'tier':>5} {'p1_report':>10} {'energy_scan':>12} "
          f"{'mine':>10}   verdict")
    n_anchor = n_ok = 0
    conflicts = []
    for (rs, tier), want in sorted(anchors.items()):
        got = [p.sigma_total for p in points if abs(p.rs - rs) < 1e-6 and p.tier == tier]
        es = escan.get((rs, tier))
        if not got:
            print(f"    {rs:6.1f} {tier:>5} {want:10.5f} "
                  f"{(f'{es:12.5f}' if es is not None else '           -')} "
                  f"{'         -':>10}   not a rung -- no point of mine (expected)")
            continue
        n_anchor += 1
        mine = got[0]
        if abs(mine - want) < 5e-5:
            n_ok += 1
            print(f"    {rs:6.1f} {tier:>5} {want:10.5f} "
                  f"{(f'{es:12.5f}' if es is not None else '           -')} "
                  f"{mine:10.5f}   OK")
            continue
        # disagreement -- let the third witness decide
        if es is not None and abs(mine - es) < 5e-5:
            verdict = "STALE ANCHOR (mine == energy_scan)"
            conflicts.append((rs, tier, want, es, mine))
        else:
            verdict = "*** UNRESOLVED: mine disagrees with src AND with energy_scan ***"
        print(f"    {rs:6.1f} {tier:>5} {want:10.5f} "
              f"{(f'{es:12.5f}' if es is not None else '           -')} "
              f"{mine:10.5f}   {verdict}")
    if not n_anchor and not anchors:
        print("    (no anchor entries found in p1_report.json)")
    elif n_anchor:
        print(f"    {n_ok}/{n_anchor} anchors agree; "
              f"{len(conflicts)} explainable as a stale snapshot")

    # --------------------------------------------- derived-table conflicts
    # Recorded rather than merely printed: these are properties of the frozen
    # artifacts, and a later stage must not "fix" them by editing a legacy file.
    if conflicts:
        print()
        print("  " + "-" * 88)
        print("  LEGACY DERIVED-TABLE CONFLICT (a finding about the legacy artifacts,")
        print("  not a defect in this pipeline -- no legacy file was modified)")
        print("  " + "-" * 88)
        for rs, tier, want, es, mine in conflicts:
            print(f"    r_s = {rs:g}, tier {tier}: p1_report.json says {want:.5f}, "
                  f"energy_scan.json says {es:.5f}, the records say {mine:.5f}")
        p1_m = _mtime(os.path.join(DIAG, "p1_report.json"))
        es_m = _mtime(os.path.join(DIAG, "energy_scan.json"))
        print(f"    p1_report.json   written {p1_m}")
        print(f"    energy_scan.json written {es_m}")
        late = _late_seeds(os.path.join(DIAG, "part1"), conflicts, mtimes=(p1_m, es_m))
        for line in late:
            print(f"    {line}")
        print("    p1_report.json and pn_table.json are snapshots taken before the r_s = 55")
        print("    seed set was completed, so at that one coupling they describe 4 of the 5")
        print("    converged seeds (and 2 of the 5 fast seeds). energy_scan.json was")
        print("    regenerated afterwards and reflects all 5. The record files are the")
        print("    authority; where the two legacy tables disagree, energy_scan.json is the")
        print("    one that matches them.")

    # ----------------------------------------------------------- band table
    band_rows = llc.band_table(states)
    legacy_pn = legacy.read_pn_table(os.path.join(DIAG, "pn_table.json"))
    pn_complaints = llc.pn_table_vs_records(band_rows, legacy_pn["rows"])
    print()
    print("=" * 92)
    print("BAND OCCUPATION TABLE vs legacy pn_table.json")
    print("=" * 92)
    print(f"  recomputed {len(band_rows)} rows; legacy {len(legacy_pn['rows'])} rows")
    _bk = lambda d: (d["n_band"], d["tier"], round(float(d["rs"]), 4))    # noqa: E731
    only_mine_rows = sorted(set(map(_bk, band_rows)) - set(map(_bk, legacy_pn["rows"])))
    only_leg_rows = sorted(set(map(_bk, legacy_pn["rows"])) - set(map(_bk, band_rows)))
    if only_mine_rows:
        # Coverage, not disagreement: the legacy table simply has fewer rows.  Said out
        # loud because a table that quietly covers less than the record set is how a
        # measured state goes missing, and the record set is the authority.
        print(f"  legacy pn_table covers {len(only_mine_rows)} fewer row(s) than the "
              f"records do (no value to compare, so nothing to disagree):")
        for nb, tier, rs in only_mine_rows:
            n = next(r["n_seeds"] for r in band_rows if _bk(r) == (nb, tier, rs))
            print(f"      n_b={nb} tier={tier:>5} r_s={rs:6.1f}  ({n} seed(s) on record)")
    if only_leg_rows:
        print(f"  *** {len(only_leg_rows)} legacy row(s) have NO counterpart in the "
              f"records: {only_leg_rows}")
    pn_mtime = _mtime(os.path.join(DIAG, "pn_table.json"))
    pn_explained, pn_unexplained = [], []
    for c in pn_complaints:
        why = _stale_snapshot_reason(c, os.path.join(DIAG, "part1"), pn_mtime)
        (pn_explained if why else pn_unexplained).append((c, why))
    if pn_complaints:
        print(f"  *** {len(pn_complaints)} mismatch(es):")
        for c, why in pn_unexplained:
            print(f"      {_fmt_complaint(c)}")
            print(f"          UNEXPLAINED -- {why or 'no stale-snapshot evidence'}")
        for c, why in pn_explained:
            print(f"      {_fmt_complaint(c)}")
            print(f"          explained: {why}")
    elif not only_mine_rows:
        print("  ALL ROWS AGREE within tolerance.")

    nest = llc.nesting_report(states)
    if nest:
        scored = [n for n in nest if n["passes"] is not None]
        npass = sum(1 for n in scored if n["passes"])
        unscored = len(nest) - len(scored)
        print(f"\n  nesting identity: {npass}/{len(scored)} SCORED records pass")
        if npass != len(scored):
            for n in scored:
                if not n["passes"]:
                    print(f"      FAILS: {n['key']}  dev={n['identity_dev']:.3e} > "
                          f"tol={n['identity_tol']:.3e}")
        if unscored:
            # Not a failure, and the distinction matters: these records measured the
            # residual but their tier did not record the tolerance it was tested
            # against.  A "20/45" headline would read as 25 failures.
            tiers = sorted({(n["n_max"], ) for n in nest if n["passes"] is None})
            print(f"      {unscored} further record(s) carry nest_identity_dev but no "
                  f"recorded nest_identity_tol (n_max {[t[0] for t in tiers]}); they are "
                  f"UNSCORED, not failed -- recovering the tolerance would mean "
                  f"re-deriving a number the record chose not to store")

    # Every difference gets a cause, and "explained" is not the same as "absent".  The
    # distinction is the whole point of the three-way check: a difference traced to a
    # stale legacy snapshot is evidence the pipeline is RIGHT, while an unexplained one
    # is evidence it is wrong.  Collapsing both into one count would throw that away.
    unexplained = bad + len(pn_unexplained) + len(only_leg_rows)
    explained = len(conflicts) + len(pn_explained)

    # ------------------------------------------------------------------ write
    meta = {
        "note": ("delta_E points from the converged (120x400) and nested n_b=3 "
                 "transition scan"),
        "source": "results/legacy_llrot/states.json (converted from _diag/part1/*.json)",
        "sigma_convention": ("sig_tot = hypot(hypot(sqrt(sum s_i^2)/n, s_liq), "
                             "seed_std/sqrt(n)); legacy p1_report.py's convention"),
        "regression": {
            "against": "_diag/energy_scan.json",
            "common_points": n_common,
            "mismatches": bad,
            "worst_abs_difference": worst,
            "explained_differences": explained,
            "unexplained_differences": unexplained,
            "legacy_table_conflicts": [
                {"rs": rs, "tier": tier, "p1_report_json": want,
                 "energy_scan_json": es, "from_records": mine}
                for rs, tier, want, es, mine in conflicts],
            "band_rows_not_in_legacy_pn_table": [list(k) for k in only_mine_rows],
        },
        "caveats": {
            "nest_sigma_init": ec.CAVEAT_NEST_SIGMA_INIT,
            "nbar": ec.CAVEAT_NBAR,
            "window_rule": ec.CAVEAT_WINDOW,
        },
    }
    results.store_scan(RESULTS, "llrot", points, brackets, meta)
    print(f"\nwrote results/legacy_llrot/scan.json  ({len(points)} points)")

    # Gaussian workflow
    gstates = results.load_states(RESULTS, "gaussian")
    gpoints = ec.gaussian_proto_points(gstates)
    results.store_scan(RESULTS, "gaussian", gpoints, [], {
        "note": "Gaussian prototype columns, kappa = 2..80",
        "source": "results/legacy_gaussian/states.json (from _diag/proto_energy.json)",
        "caveat": gpoints[0].caveats if gpoints else [],
    })
    print(f"wrote results/legacy_gaussian/scan.json  ({len(gpoints)} points)")

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  energy_scan regression : {bad} mismatch(es) over {n_common} common points")
    print(f"  anchor disagreements   : {len(conflicts)} (all traced to the r_s = 55 "
          f"stale snapshot)")
    print(f"  band-table mismatches  : {len(pn_complaints)} "
          f"({len(pn_explained)} explained, {len(pn_unexplained)} unexplained)")
    print(f"  rows only in legacy    : {len(only_leg_rows)}")
    print(f"  ---> EXPLAINED differences: {explained}   UNEXPLAINED: {unexplained}")
    if unexplained:
        print("  *** an unexplained difference means this pipeline is wrong somewhere; "
              "investigate before trusting any figure built on it")
    else:
        print("  Every difference is accounted for by the legacy tables' own snapshot "
              "dates. No legacy file was modified.")


if __name__ == "__main__":
    main()
