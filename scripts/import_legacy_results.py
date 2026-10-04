"""Convert the frozen legacy results into the canonical store.  READS ONLY.

    python wigner_vmc_clean/scripts/import_legacy_results.py

This script does not measure anything.  It opens files the old campaign wrote, reads
the fields that are already in them, reshapes them into `io.schema.StateRecord`, and
writes `results/`.  It never opens a legacy file for writing, never touches the
figure directories, and never runs a walk.

The point of insisting on "convert, never recompute" is that it makes the Stage 1
regression check meaningful: `analyze_energy.py` will recompute every delta_E from
these converted states and compare against the legacy `energy_scan.json`.  If those
two disagree, the cause has to be in the conversion or the analysis -- there is no
third place for it to hide.

What lands where
----------------
    results/legacy_llrot/states.json        the transition scan (part1: crystal + liquid)
    results/legacy_llrot/walk_states.json   the walk analysis (structure only, no energy)
    results/legacy_llrot/chains.json        the warm-started ladders, kept separate
    results/legacy_gaussian/states.json     the Gaussian prototypes' energy columns
    results/manifest.json                   every source file, by sha256

Tables that are not state records (pn_table, p1_report, mech_probe, reweight,
llrot_partV_summary) are NOT copied.  They are recorded in the manifest by path and
hash and read in place, so this store never becomes a second copy that can drift.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
sys.path.insert(0, os.path.join(CLEAN, "src"))

from wigner_vmc.io import legacy, results                       # noqa: E402
from wigner_vmc.io.legacy import sha256_of                      # noqa: E402

RESULTS = os.path.join(CLEAN, "results")
DIAG = os.path.join(ROOT, "_diag")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def src_entry(path, role, note=""):
    return {"path": rel(path), "sha256": sha256_of(path), "size": os.path.getsize(path),
            "role": role, "note": note}


def src_entry_dir(path, pattern, role, note=""):
    """One entry for a directory of records, with a combined digest.

    Hashes the sorted (name, sha256) list into a single digest, so the store records
    the exact set of files behind it without 140 separate entries.  Adding or removing
    a record -- or editing one -- changes the digest, which is the point.
    """
    import glob
    import hashlib
    files = sorted(glob.glob(os.path.join(path, pattern)))
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.basename(f).encode())
        h.update(sha256_of(f).encode())
    return {"path": rel(path), "pattern": pattern, "n_files": len(files),
            "combined_sha256": h.hexdigest(),
            "size": sum(os.path.getsize(f) for f in files),
            "role": role, "note": note}


def main():
    print(f"project root : {ROOT}")
    print(f"legacy _diag : {DIAG}")
    print(f"store        : {RESULTS}\n")

    if not os.path.isdir(DIAG):
        sys.exit(f"refusing to run: {DIAG} is not a directory")

    problems = []

    # ---------------------------------------------------------------- llrot
    recs, complaints = legacy.read_part1(DIAG)
    problems += complaints
    cry = [r for r in recs if r.phase == "crystal"]
    liq = [r for r in recs if r.phase == "liquid"]
    print(f"part1        : {len(recs)} records ({len(cry)} crystal, {len(liq)} liquid)")
    if complaints:
        print(f"               {len(complaints)} complaint(s):")
        for c in complaints[:10]:
            print(f"                 - {c}")

    paths = {
        "part1": os.path.join(DIAG, "part1"),
        "v_analysis": os.path.join(DIAG, "v_analysis.json"),
        "energy_scan": os.path.join(DIAG, "energy_scan.json"),
        "p1_report": os.path.join(DIAG, "p1_report.json"),
        "pn_table": os.path.join(DIAG, "pn_table.json"),
        "mech_probe": os.path.join(DIAG, "mech_probe.json"),
        "reweight": os.path.join(DIAG, "reweight.json"),
        "partv": os.path.join(DIAG, "llrot_diag", "llrot_partV_summary.json"),
        "proto_energy": os.path.join(DIAG, "proto_energy.json"),
        "proto_sq": os.path.join(DIAG, "proto_sq.json"),
    }

    sources = [src_entry_dir(paths["part1"], "*.json", "part1_scan_records",
                             "the transition scan; PRIMARY energy source")]
    results.store_states(
        RESULTS, "llrot", recs,
        sources=sources,
        notes=("Converted from _diag/part1/*.json. production_energy_per_particle <- "
               "E_perpart (per particle, independent production measurement). E, E_min "
               "and E_final are quarantined under `optimization`."))
    print(f"  wrote results/legacy_llrot/states.json")

    walks = legacy.read_v_analysis(paths["v_analysis"])
    results.store_states(
        RESULTS, "llrot", walks, filename="walk_states.json",
        sources=[src_entry(paths["v_analysis"], "walk_analysis_records",
                           "T/V only -- NO energy and no covariance")],
        notes=("Converted from _diag/v_analysis.json. Deliberately carries NO production "
               "energy: these records have T and V but no E and no cov(T,V), so an energy "
               "derived from them would have an invented error bar. Structure "
               "observables only (R_B, S(Q_WC), nbar, P)."))
    print(f"  wrote results/legacy_llrot/walk_states.json  ({len(walks)} records)")

    chains = legacy.read_chains(DIAG)
    results.store_states(
        RESULTS, "llrot", chains, filename="chains.json",
        sources=[src_entry(os.path.join(DIAG, "part1", "chain_up.json"), "chain"),
                 src_entry(os.path.join(DIAG, "part1", "chain_down.json"), "chain")],
        notes=("Warm-started ladder rungs. Kept in their own store because the rungs "
               "descend from each other, so they are NOT independent optimisations and "
               "must never enter a seed spread."))
    print(f"  wrote results/legacy_llrot/chains.json  ({len(chains)} rungs)")

    # ------------------------------------------------------------- gaussian
    gstates = legacy.read_proto_energy(paths["proto_energy"])
    results.store_states(
        RESULTS, "gaussian", gstates,
        sources=[src_entry(paths["proto_energy"], "prototype_energy_columns",
                           "already-reduced columns; no per-seed record exists")],
        notes=("Converted from _diag/proto_energy.json. These are DERIVED prototype "
               "columns, so no per-seed spread can be formed and sigma_init is 0 by "
               "construction rather than by measurement -- the figure carries that "
               "caveat."))
    print(f"  wrote results/legacy_gaussian/states.json  ({len(gstates)} records)")

    # --------------------------------------------------------------- tables
    tables = {}
    for key, role in (("energy_scan", "legacy derived scan; REGRESSION TARGET"),
                      ("p1_report", "legacy per-coupling report"),
                      ("pn_table", "legacy band-occupation table"),
                      ("mech_probe", "legacy mechanism probe"),
                      ("reweight", "legacy correlated-vs-independent errors"),
                      ("partv", "legacy Part V summary"),
                      ("proto_sq", "Gaussian S(q) shell data")):
        p = paths[key]
        if os.path.exists(p):
            tables[key] = src_entry(p, role)
    results.update_store_manifest(RESULTS, {
        "artifact": "legacy_llrot+legacy_gaussian states",
        "written_utc": None,
        "sources": sources,
        "reference_tables": tables,
        "conversion_problems": problems,
    })
    print(f"  wrote results/manifest.json  ({len(tables)} reference tables)")

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    have_energy = [r for r in recs if r.has_production_energy]
    print(f"states with a production energy : {len(have_energy)} / {len(recs)}")
    tiers = {}
    for r in recs:
        if r.phase == "crystal":
            tiers[(r.raw.get("tier"), r.raw.get("nb"))] = \
                tiers.get((r.raw.get("tier"), r.raw.get("nb")), 0) + 1
    print("crystal tiers present           :")
    for k in sorted(tiers, key=lambda k: (str(k[0]), k[1] or 0)):
        print(f"    tier={k[0]:6s} n_b={k[1]}  {tiers[k]:3d} records")

    if problems:
        print(f"\n{len(problems)} conversion complaint(s) -- see results/manifest.json")
    print("\nNEXT: python wigner_vmc_clean/scripts/analyze_energy.py")


if __name__ == "__main__":
    main()
