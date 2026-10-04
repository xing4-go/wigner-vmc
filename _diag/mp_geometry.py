"""Read-only: torus geometry, collinear sequences, shell structure."""
import json
import math
import os
import sys

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=200)

CLEAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CLEAN, "src"))
from wigner_vmc.analysis import structure as st   # noqa: E402

D = os.path.join(CLEAN, "results", "figure_construction", "structure_slice")
z = np.load(os.path.join(D, "structure_factor.npz"))
q = np.asarray(z["q"], float)
qn = np.asarray(z["qn"], float)

torus = st.Torus(2.0 * math.pi * 36, n_electrons=36, n_cells_per_side=6)
print("A1", torus.A1, "A2", torus.A2, "a =", torus.A_WC)
print("L1", torus.L1, "L2", torus.L2)
print("G1", torus.G1, "|G1|", np.linalg.norm(torus.G1))
print("G2", torus.G2, "|G2|", np.linalg.norm(torus.G2))
print("g1", torus.g1, "|g1|", np.linalg.norm(torus.g1))
print("g2", torus.g2, "|g2|", np.linalg.norm(torus.g2))
print("angle(G1) deg", math.degrees(math.atan2(*torus.G1[::-1])))
print("angle(G2) deg", math.degrees(math.atan2(*torus.G2[::-1])))
print("angle(g1) deg", math.degrees(math.atan2(*torus.g1[::-1])))
print("angle(g2) deg", math.degrees(math.atan2(*torus.g2[::-1])))
print("n_side", torus.n_side, "area", torus.area)

q_re, qn_re = torus.allowed_momenta(4.0)
print()
print("reconstructed q matches stored:", q_re.shape == q.shape, bool(np.allclose(q_re, q)))
print("stored q_max =", qn.max(), " (Q_MAX = 4.0; note |q|<q_max strict)")

# ---- collinear sequences -------------------------------------------------
# a direction dhat is "collinear-accessible" if several allowed momenta lie on it
def collinear(dhat, tol_deg=0.5):
    dhat = np.asarray(dhat, float)
    dhat = dhat / np.linalg.norm(dhat)
    perp = np.abs(q @ np.array([-dhat[1], dhat[0]]))
    par = q @ dhat
    sel = perp < 1e-9
    out = []
    for p in np.unique(np.round(np.abs(par[sel]), 12)):
        if p <= 0:
            continue
        idx = np.flatnonzero(sel & np.isclose(np.abs(par), p, rtol=0, atol=1e-9))
        out.append((float(p), idx))
    out.sort()
    return out

cands = {
    "x = (1,0)  [A1, real-space NN bond]": (1.0, 0.0),
    "-30 deg   [G1/g1, WC Bragg]": (math.cos(math.radians(-30)), math.sin(math.radians(-30))),
    "+90 deg   [G2/g2, WC Bragg]": (0.0, 1.0),
    "0 deg again": (1.0, 0.0),
    "+30 deg   [G1+G2, WC Bragg]": (math.cos(math.radians(30)), math.sin(math.radians(30))),
    "60 deg    [A2, real-space NN bond]": (math.cos(math.radians(60)), math.sin(math.radians(60))),
    "120 deg   [A2-A1]": (math.cos(math.radians(120)), math.sin(math.radians(120))),
}
print()
for name, d in cands.items():
    seq = collinear(d)
    print("%-40s n_q=%d  |q| = %s" % (name, len(seq), [round(s[0], 5) for s in seq]))

print()
print("Q_WC =", np.linalg.norm(torus.g1))
print("shells of |q|:")
order = np.argsort(qn, kind="stable")
qs = qn[order]
new = np.ones(qs.size, bool)
new[1:] = ~np.isclose(qs[1:], qs[:-1], rtol=1e-9, atol=0.0)
starts = np.flatnonzero(new)
bounds = list(starts) + [qs.size]
for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
    idx = order[a:b]
    angs = sorted(round(math.degrees(math.atan2(q[j, 1], q[j, 0])), 3) for j in idx)
    print("  shell %2d |q|=%.6f n=%2d angles=%s" % (i, qs[a], b - a, angs))
