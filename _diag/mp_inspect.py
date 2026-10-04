"""Read-only inspection of the stored structure_slice S(q-vector) arrays."""
import json
import os
import sys
import time

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=200)

CLEAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(CLEAN, "results", "figure_construction", "structure_slice")

z = np.load(os.path.join(D, "structure_factor.npz"))
print("KEYS:", list(z.keys()))
for k in z.keys():
    print("  %-24s shape=%-12s dtype=%s" % (k, z[k].shape, z[k].dtype))

q = np.asarray(z["q"], float)
qn = np.asarray(z["qn"], float)
print()
print("q[:6]  =", q[:6].tolist())
print("qn[:6] =", qn[:6].tolist())
print("qn == |q| exactly (atol 1e-12):", bool(np.allclose(qn, np.linalg.norm(q, axis=1), rtol=0, atol=1e-12)))
print("max | qn - |q| | =", float(np.abs(qn - np.linalg.norm(q, axis=1)).max()))
print("n momenta =", len(qn))

print()
print("S_liquid ", z["S_liquid"].shape, "S_crystal", z["S_crystal"].shape)
print("S_filled_LLL_exact", z["S_filled_LLL_exact"].shape)
print("g1_mag", float(z["g1_mag"]), "G1_mag", float(z["G1_mag"]))

meta = json.load(open(os.path.join(D, "run_metadata.json"), encoding="utf-8"))
print()
print("meta budget", meta["budget"], "schema", meta["run_metadata_schema"])
for ph in ("liquid", "crystal"):
    p = meta[ph]
    print("  %-8s rs=%.9f kappa=%.9f nmax=%s phase=%s" % (ph, p["rs"], p["kappa"], p["nmax"], p["phase"]))
print("torus_L1", meta["torus_L1"], "torus_L2", meta["torus_L2"], "area", meta["torus_area"])
print("observables.q_grid:", meta["observables"]["q_grid"])
print("npz mtime", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(D, "structure_factor.npz")))))

# shells of |q|
order = np.argsort(qn, kind="stable")
qs = qn[order]
new = np.ones(qs.size, bool)
new[1:] = ~np.isclose(qs[1:], qs[:-1], rtol=1e-9, atol=0.0)
starts = np.flatnonzero(new)
bounds = list(starts) + [qs.size]
print()
print("N shells =", len(starts))
print("%-4s %-10s %-5s %-10s %-10s" % ("i", "q_shell", "n", "S_liq_mean", "S_cry_mean"))
Sl = np.asarray(z["S_liquid"], float)
Sc = np.asarray(z["S_crystal"], float)
for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
    idx = order[a:b]
    if i < 40:
        print("%-4d %-10.6f %-5d %-10.5f %-10.5f" % (i, qs[a], b - a, Sl[idx].mean(), Sc[idx].mean()))
print("...")
print("total momenta", qs.size, "shell sizes histogram:",
      np.bincount([b - a for a, b in zip(bounds[:-1], bounds[1:])]))

# directional structure: list all vectors in shell 0
idx0 = order[bounds[0]:bounds[1]]
print()
print("shell 0 |q| =", qs[0])
print("vectors:", q[idx0].tolist())
print("S_liquid:", Sl[idx0].tolist())
print("S_crystal:", Sc[idx0].tolist())
