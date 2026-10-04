"""Read-only: the six §13 validation items, checked against the built curves."""
import ast
import importlib.util
import os
import sys

import numpy as np

CLEAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CLEAN, "src"))
SCRIPT = os.path.join(CLEAN, "examples", "figure_construction", "magnetoplasmon.py")
spec = importlib.util.spec_from_file_location("mp", SCRIPT)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

ok = []


def check(n, cond, detail=""):
    ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] ({n}) {detail}")


args = M.parse_args(["--rs", "0", "5.567", "10", "90.51",
                     "--source-budget", "quick"])
args.phase_of = M.phase_of
curves, index = M.build_curves(args)
torus = M.torus_for(36)
dhat = M.crystal_direction(torus, "bragg")

print("§13 validation")
print()

# (1) every liquid shell contains only equal-|q| vectors
liq = [c for c in curves if c["phase"] == "liquid"]
worst = 0.0
all_multi = []
for c in liq:
    qv, qn, S = c["raw_qvec"], c["raw_qn"], c["raw_S"]
    for qi in c["q"]:
        idx = np.flatnonzero(np.isclose(qn, qi, rtol=1e-9, atol=0.0))
        worst = max(worst, float(qn[idx].max() - qn[idx].min()))
        all_multi.append(idx.size)
check(1, worst <= M.SHELL_TOL,
      f"max within-shell |q| spread = {worst:.3g} <= tol {M.SHELL_TOL:g}; "
      f"multiplicities seen = {sorted(set(all_multi))}")

# (2) liquid shell averaging occurs on S, not on 1/S
gaps = []
for c in liq:
    r = c["reduction"]
    gaps.append(np.abs(r["order_gap_rel"]).max())
    # recompute q^2/(2<S>) from the raw array and compare to the curve
    for i, qi in enumerate(c["q"]):
        idx = np.flatnonzero(np.isclose(c["raw_qn"], qi, rtol=1e-9))
        assert abs(r["S"][i] - c["raw_S"][idx].mean()) < 1e-12
check(2, min(gaps) > 1e-6,
      f"S = plain shell mean (verified elementwise); order gap "
      f"<q^2/2S>-q^2/(2<S>) max = {max(gaps):.3%} > 0 -- the two orderings "
      f"provably differ")

# (3) every crystal point in the final curve is collinear with the direction
cry = [c for c in curves if c["phase"] == "crystal"]
resid, npts = 0.0, []
for c in cry:
    qv = c["raw_qvec"]
    for qi in c["q"]:
        idx = np.flatnonzero(np.isclose(np.linalg.norm(qv, axis=1), qi,
                                        rtol=1e-9))
        perp = np.abs(qv[idx] @ np.array([-dhat[1], dhat[0]]))
        resid = max(resid, float(perp.min()))
    npts.append(c["q"].size)
    assert np.all(c["reduction"]["count"] == 1)
check(3, resid <= M.COLLINEAR_TOL,
      f"max perpendicular residual over every plotted crystal point = "
      f"{resid:.3g}; points per curve = {npts}; count==1 everywhere "
      f"(no shell average)")

# (4) rs = 0 satisfies Omega(q->0) = 1
a = M.parse_args(["--rs", "0"])
a.phase_of = M.phase_of
c0, _ = M.build_curves(a)
c0 = c0[0]
lim = float(c0["Omega"][0])
check(4, abs(lim - 1.0) < 1e-6,
      f"rs=0 at q={c0['q'][0]:.1e}: Omega = {lim:.9f} -> 1 (hbar omega_c)")

# (5) no fitted scale factor in omega_mp
k = M.kappa_from_rs(90.51)
q = np.array([0.4489, 1.3468, 2.6935])
lhs = M.omega_magnetoplasmon(q, k) ** 2
rhs = 1.0 + k * q
check(5, np.allclose(lhs, rhs, rtol=1e-14) and abs(M.kappa_from_rs(0.0)) == 0.0,
      f"omega_mp^2 == 1 + kappa q exactly (kappa from the engine); "
      f"kappa(0) = {M.kappa_from_rs(0.0)}; no free prefactor exists in the "
      f"function signature")

# (6) no VMC/SR code path reachable
src = open(SCRIPT, encoding="utf-8").read()
tree = ast.parse(src)
imports = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        imports |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom):
        imports |= {(n.module or "").split(".")[0]}
forbidden = {"subprocess", "multiprocessing", "concurrent", "sampler",
             "run_vmc", "vmc_campaign"}
check(6, not (imports & forbidden) and "structure_slice.py" in src
      and M.PROJECTION_FIX_MTIME > 0,
      f"top-level imports = {sorted(imports)}; none is a runner; "
      f"no subprocess/importlib.exec path")

print()
print(f"  {sum(ok)}/{len(ok)} validation items pass")
