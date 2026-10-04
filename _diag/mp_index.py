"""Read-only index of every stored structure_slice source, from metadata."""
import glob
import json
import os
import time

import numpy as np

CLEAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAM = os.path.join(CLEAN, "results", "figure_construction", "structure_slice")
FIX = time.mktime(time.strptime("2026-10-03 16:29:56", "%Y-%m-%d %H:%M:%S"))

dirs = sorted({d for d in glob.glob(FAM) + glob.glob(FAM + "_*") if os.path.isdir(d)})
print("FIX mtime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(FIX)))
print()
print("%-58s %-8s %-12s %-6s %-13s %-6s %-8s %s" %
      ("directory", "phase", "rs", "nmax", "budget", "post", "schema", "npz mtime"))
rows = []
for d in dirs:
    mp = os.path.join(d, "run_metadata.json")
    npz = os.path.join(d, "structure_factor.npz")
    base = os.path.basename(d).replace("structure_slice_", "ss_")
    if not (os.path.isfile(mp) and os.path.isfile(npz)):
        print("%-58s MISSING metadata or npz" % base)
        continue
    meta = json.load(open(mp, encoding="utf-8"))
    mt = os.path.getmtime(npz)
    for ph in ("liquid", "crystal"):
        p = meta.get(ph) or {}
        if p.get("rs") is None:
            continue
        z = np.load(npz)
        n = len(z["qn"])
        rows.append((ph, float(p["rs"]), p.get("nmax"), str(meta.get("budget", "")),
                     mt >= FIX, d, n, str(meta.get("run_metadata_schema", ""))))
        print("%-58s %-8s %-12.9f %-6s %-13s %-6s %-8s %s  nq=%d" %
              (base, ph, float(p["rs"]), p.get("nmax"), meta.get("budget", ""),
               "YES" if mt >= FIX else "no",
               str(meta.get("run_metadata_schema", ""))[:22],
               time.strftime("%m-%d %H:%M", time.localtime(mt)), n))

print()
print("=== available couplings by phase/budget/nmax (post-fix only) ===")
for ph in ("liquid", "crystal"):
    for bud in ("reproduction", "quick"):
        sel = sorted({(round(r[1], 6), r[2]) for r in rows
                      if r[0] == ph and r[3] == bud and r[4]})
        print("  %-8s %-13s post-fix: %s" % (ph, bud, sel if sel else "NONE"))
print()
print("=== paper points: rs = 0, 5, 30, 60 ===")
for rs in (5.0, 30.0, 60.0):
    ph = "liquid" if rs < 47.0 else "crystal"
    hits = [(r[1], r[2], r[3], r[4]) for r in rows if r[0] == ph and abs(r[1] - rs) < 1e-9]
    print("  rs=%-5g -> %-8s : %s" % (rs, ph, hits if hits else "MISSING"))
