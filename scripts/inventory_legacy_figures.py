"""Fingerprint every existing figure, in place.  READS ONLY, WRITES NOTHING BACK.

    python wigner_vmc_clean/scripts/inventory_legacy_figures.py

Requirement 1E: before any new plotting pipeline is trusted, record what the current
official figures ARE -- filename, sha256, byte size, pixel dimensions, mtime -- so that
"did this pipeline reproduce that figure?" is answerable by comparison rather than by
memory.  Nothing here opens a legacy directory for writing.

Why the dimensions and not just the hash
----------------------------------------
The hash answers "same bytes".  But the acceptance rule for Stage 1 is explicitly NOT
pixel-identical: (1) the plotted numbers must agree, (2) the error bars must agree,
(3) the labels must agree, (4) the axes and units must agree, and only then (5) the
visual appearance should be broadly similar.  A re-derived figure that plots the same
quantities with the same error bars but is styled differently will have a different
hash and the SAME data, and that is a pass.  Recording the size in pixels as well means
a figure that came back the right shape but the wrong content is visible as a mismatch
in the data columns, not hidden behind "the hash differs, so it must be different".

Where the figures live, and which count
---------------------------------------
Only the two delivered directories are inventoried as OFFICIAL.  The scratch and
diagnostic directories are inventoried separately and marked as such, because an image
sitting in `_diag/v_figs/` is not an official figure and must never be promoted into the
figure set merely by being present.  Two LL-rotation files exist in both delivered
directories; where they are byte-identical that is recorded, and where they are not, the
sha256 pair records the drift rather than resolving it.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)

OUT = os.path.join(CLEAN, "results", "legacy_figures.json")

# (path relative to ROOT, workflow, tier)
#   tier: "official"  -- a delivered figure directory; frozen, must not be overwritten
#         "scratch"   -- diagnostic / preview output; never auto-promoted
SOURCES = [
    ("figs",                                       "gaussian", "official"),
    ("figs_LLRotation",                            "llrot",    "official"),
    ("reproduction/figs_LLRotation",               "llrot",    "official"),
    ("_diag/v_figs",                               "llrot",    "scratch"),
    ("_diag/iv_figs",                              "gaussian", "scratch"),
    ("_diag/figs_proto",                           "llrot",    "scratch"),
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(path: str):
    """(width, height) from the IHDR chunk, or (None, None) if this is not a PNG.

    Read straight from the header rather than via an imaging library, so this works the
    same whether or not Pillow happens to be installed -- an inventory that silently
    reports nulls on a machine without PIL is worse than no inventory.
    """
    with open(path, "rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None, None
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def main():
    report = {"schema": "wigner_vmc_clean/legacy_figures",
              "note": ("Read-only fingerprint of every figure present when Stage 1 "
                       "began. Nothing was moved, renamed, re-rendered or overwritten."),
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "directories": []}

    by_hash = {}
    for rel, workflow, tier in SOURCES:
        d = os.path.join(ROOT, rel)
        entry = {"path": rel, "workflow": workflow, "tier": tier,
                 "exists": os.path.isdir(d), "files": []}
        if not entry["exists"]:
            report["directories"].append(entry)
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.png"))):
            w, h = png_size(p)
            sha = sha256_of(p)
            name = os.path.basename(p)
            entry["files"].append({
                "filename": name,
                "sha256": sha,
                "size_bytes": os.path.getsize(p),
                "width": w, "height": h,
                "mtime": time.strftime("%Y-%m-%d %H:%M",
                                       time.localtime(os.path.getmtime(p))),
            })
            by_hash.setdefault(sha, []).append(f"{rel}/{name}")
        report["directories"].append(entry)

    # Cross-directory identity: which files are the same bytes, and which share a name
    # but not a hash.  The second list is the one that matters -- it is the drift.
    names = {}
    for entry in report["directories"]:
        if entry["tier"] != "official":
            continue
        for f in entry["files"]:
            names.setdefault(f["filename"], []).append(
                {"where": entry["path"], "sha256": f["sha256"]})
    identical, drifted = [], []
    for name, copies in sorted(names.items()):
        if len(copies) < 2:
            continue
        shas = {c["sha256"] for c in copies}
        (identical if len(shas) == 1 else drifted).append(
            {"filename": name, "copies": copies})
    report["official_cross_directory"] = {
        "byte_identical_in_both": identical,
        "same_name_different_bytes": drifted,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    os.replace(OUT + ".tmp", OUT)

    print(f"wrote {os.path.relpath(OUT, ROOT)}\n")
    for entry in report["directories"]:
        tag = entry["tier"].upper()
        if not entry["exists"]:
            print(f"  [{tag:8s}] {entry['path']}  -- NOT PRESENT")
            continue
        print(f"  [{tag:8s}] {entry['path']}  ({len(entry['files'])} png)")
        for f in entry["files"]:
            dim = f"{f['width']}x{f['height']}" if f["width"] else "not a png"
            print(f"      {f['filename']:44s} {f['sha256'][:12]}  "
                  f"{f['size_bytes']:>8d} B  {dim:>10s}  {f['mtime']}")

    print()
    print("  official figures present in BOTH delivered directories:")
    for e in identical:
        print(f"      identical : {e['filename']}")
    for e in drifted:
        print(f"      *** DRIFT : {e['filename']}")
        for c in e["copies"]:
            print(f"                  {c['where']:32s} {c['sha256'][:16]}")

    n_off = sum(len(e["files"]) for e in report["directories"] if e["tier"] == "official")
    n_scr = sum(len(e["files"]) for e in report["directories"] if e["tier"] == "scratch")
    print(f"\n  {n_off} official png(s) across the delivered directories; "
          f"{n_scr} scratch png(s) (never auto-promoted).")


if __name__ == "__main__":
    main()
