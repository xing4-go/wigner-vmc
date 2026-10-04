"""Blocker B: at which kappa were the frozen ``jastrow_*_k53.033`` pickles
actually written?  A byte-level test against three independently recorded hashes.

``incident_20261002/README.md`` records a puzzle it could not solve:

    "the frozen ``k53.033`` entry does **not** reproduce from the recorded inputs
     (``warm = J_OPT[crystal,48.0] . 53.033/48.0``), and none of twelve other
     plausible warm sources reproduces it either."

Thirteen attempts, all varying ``warm``.  But ``warm`` is not the only input the
tag cannot see.  The tag's kappa field is formatted ``{kappa:g}`` -- SIX
significant digits -- and

    f"{53.033:g}"              -> '53.033'
    f"{53.03300858899106:g}"   -> '53.033'

so ``k53.033`` is written identically by BOTH the campaign's rounded kappa and the
physical one.  All thirteen reconstructions pinned kappa to the literal 53.033, so
if the file was written at the physical kappa, every one of them was guaranteed to
fail no matter how ``warm`` varied.  Worse, kappa does not only scale the warm
start: it is also an argument to ``jastrow_opt`` itself (through ``cusp_gamma`` and
the wavefunction), so it moves the SR trajectory, not just its starting point.

This script varies the input those thirteen held fixed, and compares the pickles
BYTE-FOR-BYTE against three hashes recorded independently of any reconstruction:

  A. ``_diag/ckpt_rebuild``, pre-incident manifest entry.  L0 = 0.6.
     pkl 680d52e6...   mtime 2026-09-28T20:21:24
  B. ``%TEMP%/qhvmc_checkpoints`` -- NEVER touched by the incident (the pre-incident
     manifest records it ``status: frozen``, and its sha256 is unchanged on disk).
     L0 = 0.8.   pkl 951d1440...   mtime 2026-09-28T20:17:22
  C. the LIQUID entry.  Its call passes ``L0 = None``, so its tag carries ``L--``
     and is width-free -- part1_scan.py:109-110 makes the same point about why
     liquids are cache-safe.  Kappa is then the ONLY input the tag cannot see, and
     there is no warm ambiguity to rule out, so this case has no confounder at all.
     L0 = None.  pkl 973574421e14ac33...

The method is known to be exact: the README reports that all ten chain entries
``jastrow_crystal_k{2,4,8,...,80}`` reproduce byte-for-byte.  A byte match is
therefore decisive; a byte mismatch is informative.

WHY THERE IS NO SECONDARY ``warm`` SWEEP
----------------------------------------
The README searched thirteen warm sources because it did not know which was used.
We do know -- ``part1_scan.py:284`` is the whole rule -- so the job is to apply
it, not to search.  And the alternatives are excluded by the tag itself, by
arithmetic that needs no computation:

    a cold start        c_guess(kappa)[0] = -kappa/6 = -8.8388  -> tag reads w-8.84
    an unscaled warm    J_OPT[(kind, 48.0)][0] ~ -6.78         -> tag reads w-6.78

The frozen tags read ``w-7.49`` (both crystal cases) and ``w-12.38`` (liquid).
``w{...:+.2f}`` discriminates at 0.01, and these differ by >0.7, so both
alternatives are inconsistent with the recorded tag.  The README called that
field too coarse to recover ``warm`` from -- true, it cannot recover the last
four components -- but it is amply sufficient to EXCLUDE these sources.  That
leaves kappa as the only input not already tested, which is exactly what the
primary sweep varies.

READ-ONLY.  The working copy is a scratch tree; nothing is written back, and the
frozen store is re-fingerprinted on exit.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frozen_guard import FROZEN_CKPT, readonly_frozen            # noqa: E402
from recon_rs75_init import CLEAN, namespace_cells               # noqa: E402

RS = 75.0
K_LEGACY = round(RS / np.sqrt(2.0), 4)      # 53.033            -- campaign convention
K_CLEAN = RS / np.sqrt(2.0)                 # 53.03300858899106 -- physical
K_NEAR = 48.0                               # nearest KAPPAS grid point to either

CASES = (
    {"label": "A. curated pre-incident      (L0=0.6)",
     "short": "A_ckpt_rebuild_L0.6", "kind": "crystal", "L0": 0.60,
     "sha256": "680d52e685a6cc7e883282699da6c2bb00c780ba7b96ab8d8996f44470a24d6c",
     "meta_sha256": "56480d7fd45b037ecc0f931cd1c20dbe2648c908248be63e47e2f235bbf059a2",
     "meta_tag": "v1|ne36|LU30|J5|crystal|k53.033|L0.6|st8|n150|sig0.3|w-7.49|sr1",
     "src": "incident_20261002/LEGACY_MANIFEST.pre_incident.json",
     "sweep": "kappa_only"},
    {"label": "B. %TEMP%, incident-UNTOUCHED (L0=0.8)",
     "short": "B_temp_L0.8", "kind": "crystal", "L0": 0.80,
     "sha256": "951d1440a9b87178f7d87757695161d63a041d3186d22a47e79a032642708540",
     "meta_sha256": "2062be9ff013d6cb5a803c663577b3d397ab0627dc5a80c842cb24bbbe329841",
     "meta_tag": "v1|ne36|LU30|J5|crystal|k53.033|L0.8|st8|n150|sig0.3|w-7.49|sr1",
     "src": "%TEMP%/qhvmc_checkpoints (sha unchanged since the pre-incident manifest)",
     "sweep": "kappa_only"},
    {"label": "C. curated LIQUID, L0=None -- kappa is the ONLY hidden input",
     "short": "C_ckpt_rebuild_liquid", "kind": "liquid", "L0": None,
     "sha256": "973574421e14ac3331d8659e17bf3924a1765d6639431eb582fd21575298ac9e",
     "meta_sha256": "38ff3e6e209dcb88d600a71c5f5abeea7cd888a2cf0c1ada938cd0faa79a5c17",
     "meta_tag": "v1|ne36|LU30|J5|liquid|k53.033|L--|st8|n150|sig0.4|w-12.38|sr1",
     "src": "incident_20261002/LEGACY_MANIFEST.pre_incident.json (sha unchanged on disk)",
     "sweep": "kappa_only"},
)


def sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def purge(work, kind):
    """Delete the shared cache entry so the next call must recompute."""
    base = os.path.join(work, f"jastrow_{kind}_k53.033")
    for ext in (".pkl", ".meta"):
        try:
            os.remove(base + ext)
        except FileNotFoundError:
            pass
    return base


def attempt(E, case, kappa, warm, work, note):
    """Rebuild one entry; return its hashes and whether they match the record."""
    kind = case["kind"]
    base = purge(work, kind)
    E["jastrow_opt"](kind, kappa, L0=case["L0"], warm=warm)
    pkl, meta = base + ".pkl", base + ".meta"
    tag = open(meta, encoding="utf-8").read().strip() if os.path.exists(meta) else None
    s, sm = sha256(pkl), sha256(meta)
    pkl_hit, meta_hit = (s == case["sha256"]), (sm == case["meta_sha256"])
    mark = ""
    if pkl_hit and meta_hit:
        mark = "   <<<< BYTE-EXACT (pkl AND meta)"
    elif pkl_hit:
        mark = "   <<<< pkl BYTE-EXACT"
    elif meta_hit:
        mark = "   <<<  meta byte-exact, pkl differs"
    print(f"      k {kappa:<19.12f} {note:32s} pkl {str(s)[:16]}  "
          f"meta {str(sm)[:16]}{mark}")
    return {"kappa": kappa, "L0": case["L0"], "warm_note": note,
            "pkl_sha256": s, "meta_sha256": sm, "meta_tag": tag,
            "pkl_byte_exact": bool(pkl_hit), "meta_byte_exact": bool(meta_hit)}


def run(E, work):
    """The byte test, given an already-built namespace.  Returns (blob, rc).

    Exposed as a function so a combined driver can build the namespace ONCE and
    run this beside the c0/kappa probe -- a namespace build costs ~17 minutes.
    """
    out_rows = []
    for case in CASES:
        kind = case["kind"]
        base_warm = np.asarray(E["J_OPT"][(kind, K_NEAR)], float)
        print(f"  {case['label']}")
        print(f"      target pkl  sha256 {case['sha256']}")
        print(f"      target meta sha256 {case['meta_sha256']}")
        print(f"      target meta tag    {case['meta_tag']}")
        print(f"      source: {case['src']}")
        print(f"      warm base J_OPT[({kind!r}, {K_NEAR:g})] = "
              f"{np.round(base_warm, 6).tolist()}")
        # The ONLY sweep, and the only input the thirteen earlier reconstructions
        # held fixed.  The warm rule is known from part1_scan.py:284, and every
        # alternative warm source is excluded by the tag's own w field (see the
        # module docstring), so there is nothing else to vary.
        rows = []
        for kappa, note in ((K_LEGACY, "campaign rounded kappa"),
                            (K_CLEAN, "physical kappa (tag-invisible)")):
            rows.append(attempt(E, case, kappa, base_warm * (kappa / K_NEAR),
                                work, note))
        exact = [r for r in rows if r["pkl_byte_exact"]]
        if exact:
            hits = "; ".join("k=%.12f (%s)" % (r["kappa"], r["warm_note"])
                             for r in exact)
        else:
            hits = "NOTHING"
        print(f"      -> byte-exact at: {hits}\n", flush=True)
        out_rows.append({"case": case["short"], "label": case["label"],
                         "kind": kind, "L0": case["L0"],
                         "target_pkl_sha256": case["sha256"],
                         "target_meta_sha256": case["meta_sha256"],
                         "target_meta_tag": case["meta_tag"],
                         "attempts": rows,
                         "byte_exact_kappas": [r["kappa"] for r in exact]})

    print("=" * 78)
    for row in out_rows:
        if row["byte_exact_kappas"]:
            for k in row["byte_exact_kappas"]:
                which = ("LEGACY (53.033)" if abs(k - K_LEGACY) < 1e-9
                         else "CLEAN (physical)")
                print(f"  RESOLVED  {row['label']}")
                print(f"            was written at {which} kappa = {k:.12f}")
        else:
            print(f"  UNRESOLVED  {row['label']} -- no kappa x warm candidate matches")
    print("=" * 78, flush=True)

    blob = {"rs": RS, "kappa_legacy": K_LEGACY, "kappa_clean": K_CLEAN,
            "cases": out_rows}
    return blob, 0


def main(argv=None):
    print("=" * 78)
    print("  At which kappa were the frozen jastrow_*_k53.033 pickles written?")
    print("=" * 78)
    print(f"  kappa legacy {K_LEGACY}   kappa clean {K_CLEAN:.12f}")
    print(f"  tag kappa field is {{kappa:g}} -> both render '53.033' "
          f"({f'{K_LEGACY:g}'!r} vs {f'{K_CLEAN:g}'!r})")
    print("  building the notebook namespace against a WORKING COPY\n", flush=True)

    with readonly_frozen(FROZEN_CKPT) as work:
        E = namespace_cells(work)
        blob, rc = run(E, work)

    out = os.path.join(CLEAN, "logs", "diag_incident_pkl_sha.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print(f"  wrote {out}")
    print("  frozen tree: UNCHANGED -- the guard verified it on exit")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
