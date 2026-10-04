"""Reproduce every number in ANISOTROPY_DIAGNOSIS_rs75_nmax2.md.

READ-ONLY.  This script opens the stored clean-VMC snapshots and the frozen
metadata, rebuilds geometry and one deterministic one-body density from the
recorded initialization parameters, and prints the diagnostic tables.  It starts
no VMC and no SR, writes nothing into results/ or figures/, and mutates no state --

    python examples/figure_construction/anisotropy_diagnosis_rs75_nmax2.py

Run from the repository root (the package must be importable).  Sections are
numbered to match the report.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

LONG = "--long" in sys.argv

CLEAN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "structure.py")
spec = importlib.util.spec_from_file_location("pssp", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
st = mod.st

from wigner_vmc.api import VMC                                        # noqa: E402
import wigner_vmc.vmc.sr as sr                                        # noqa: E402
from wigner_vmc.physics import landau_levels as llb                   # noqa: E402
from wigner_vmc.physics.landau_levels import magnetic_translation_phase  # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr                # noqa: E402

SLUG = "liquid_rs75__crystal_rs75__llrot_nmax2__reproduction"
RES = os.path.join(CLEAN, "results", "figure_construction", "structure", SLUG)
NE = 36


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def head(title):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


tor = mod.torus_for()
g1, g2 = np.asarray(tor.g1), np.asarray(tor.g2)
GBAS = np.array([g1, g2])
area = float(tor.area)
n_mean = NE / area
vmc = VMC(N=NE, rs=75.0, phase="crystal", nmax=2, kappa_mode="physical",
          ansatz="ll_rotation")
lat = vmc.lat


def pairs6():
    raw = np.asarray(tor.wc_shell_vectors())
    out, used = [], set()
    for i in range(len(raw)):
        if i in used:
            continue
        for j in range(i + 1, len(raw)):
            if j not in used and np.allclose(raw[i], -raw[j], atol=1e-9):
                out.append((i, j))
                used.update((i, j))
                break
    return raw, out


SHELL, PAIRS = pairs6()
FAMILY = ("G1 = g1+g2", "G2 = g1", "G3 = g2")
G3 = SHELL[np.array([a for a, _ in PAIRS])]

# ---------------------------------------------------------------- 1. raw table
head("1.  raw S at the six first-shell WC reciprocal vectors  (no plot, no grid)")
q_all, _ = tor.allowed_momenta(q_max=5.2)
with open(os.path.join(RES, "run_metadata.json"), encoding="utf-8") as fh:
    meta = json.load(fh)
print(f"  metadata: budget={meta['budget']!r} liquid_rs={meta['liquid_rs']} "
      f"crystal_rs={meta['crystal_rs']} crystal_nmax={meta['crystal_nmax']}")
print(f"  {meta['crystal']['ansatz']}, init_L0={meta['crystal']['init_L0']}, "
      f"kappa={meta['crystal']['kappa']!r}")
print()
print(f"  {'family':<12} {'G = (Gx, Gy)':<26} {'nearest allowed q':<26} "
      f"{'|q-G|':>9} {'S_liquid':>10} {'S_crystal':>10}")
S = {}
for phase in ("liquid", "crystal"):
    with np.load(os.path.join(RES, phase, "run.npz")) as z:
        S[phase] = st.structure_factor_snapshots(
            np.asarray(z["snaps"], float), q_all).mean(axis=0)
    print(f"  [{phase}: {np.load(os.path.join(RES, phase, 'run.npz'))['snaps'].shape[0]}"
          f" configurations]")
si = {p: [] for p in S}
for k, (i, _) in enumerate(PAIRS):
    for sgn in (1.0, -1.0):
        G = sgn * SHELL[i]
        m = int(np.argmin(np.linalg.norm(q_all - G, axis=1)))
        d = float(np.linalg.norm(q_all[m] - G))
        vals = [S[p][m] for p in ("liquid", "crystal")]
        if sgn > 0:
            for p in si:
                si[p].append(vals[("liquid", "crystal").index(p)])
        print(f"  {FAMILY[k] if sgn > 0 else '  (-G)':<12} ({G[0]:+9.6f},{G[1]:+9.6f})   "
              f"({q_all[m][0]:+9.6f},{q_all[m][1]:+9.6f})   {d:9.3e} "
              f"{vals[0]:10.5f} {vals[1]:10.5f}")
print()
for p in ("liquid", "crystal"):
    v = np.array(si[p])
    print(f"  {p:<8} S1,S2,S3 = {np.round(v, 5).tolist()}   "
          f"A = {(v.max()-v.min())/v.mean():.6f}   max/min = {v.max()/v.min():.6f}")

head("1b. the q -> -q identity over a large random q sample")
rngq = np.random.default_rng(101)
qr = rngq.uniform(-4, 4, (400, 2))
with np.load(os.path.join(RES, "crystal", "run.npz")) as z:
    snaps_c = np.asarray(z["snaps"], float)
with np.load(os.path.join(RES, "liquid", "run.npz")) as z:
    snaps_l = np.asarray(z["snaps"], float)
for nm, sn in (("crystal", snaps_c), ("liquid", snaps_l)):
    a = st.structure_factor(sn, qr, NE)
    b = st.structure_factor(sn, -qr, NE)
    print(f"  {nm:<8} max |S(q) - S(-q)| over 400 random q = "
          f"{np.abs(a - b).max():.3e}")

# ------------------------------------------------------- 3. commensurability
head("3.  allowed-momentum mismatch (built from the cell, not from the plot)")
print(f"  |g1|/|G1| = {np.linalg.norm(g1)/np.linalg.norm(np.asarray(tor.G1)):.12f}")
worst = max(float(np.linalg.norm(q_all - SHELL[i], axis=1).min()) for i, _ in PAIRS)
print(f"  worst min_q |q - G| over the six targets = {worst:.3e}   "
      f"(allowed_momenta tolerance 1e-6)")
print(f"  all six are exact allowed momenta: {worst < 1e-6}")

head("3b. the allowed-momentum lattice is DENSE around the shell (not a 'sole q' effect)")
print(f"  allowed momenta with |q| <= 5.2: {len(q_all)};  "
      f"the six targets are 6 of them.")

# ----------------------------------------------------- 5. reciprocal oracle
head("5.  independent reciprocal basis oracle,  b = 2 pi inv(A)^T")
A = np.column_stack([np.asarray(tor.A1), np.asarray(tor.A2)])
b = 2 * np.pi * np.linalg.inv(A).T
print(f"  max | b_i . A_j - 2 pi delta_ij | = "
      f"{np.abs(b.T @ A - 2*np.pi*np.eye(2)).max():.3e}")
print(f"  max | b - (g1,g2) |               = "
      f"{np.abs(b - GBAS.T).max():.3e}")
print(f"  max | G1 - b1/6 |                 = "
      f"{np.abs(np.asarray(tor.G1) - b[:, 0] / 6).max():.3e}")
cand = sorted([(m, n) for m in range(-3, 4) for n in range(-3, 4) if (m, n) != (0, 0)],
              key=lambda mn: np.linalg.norm(mn[0] * g1 + mn[1] * g2))[:6]
oracle = np.array([m * g1 + n * g2 for m, n in cand])
key = lambda X: {tuple(np.round(x, 9)) for x in X}
print(f"  oracle's six shortest == Torus.wc_shell_vectors() as a set: "
      f"{key(oracle) == key(SHELL)}")

# ------------------------------------------------- 6. real-space symmetry
head("6.  real-space symmetry of the SAME snapshots:  g(x, y) at r = a")
with np.load(os.path.join(RES, "pair_correlation_xy.npz")) as z:
    gx, gy = z["x"], z["y"]                    # both are (nr, nr) MESHGRIDS
    gl, gc = z["g_liquid"], z["g_crystal"]
    rmax, nr, smooth = float(z["rmax"]), int(z["nr"]), float(z["smooth"])
a_nn = float(np.linalg.norm(np.asarray(tor.A1)))
print(f"  g grid {gc.shape} on the meshgrid, x in [{gx.min():.3f},{gx.max():.3f}], "
      f"rmax={rmax:g}, nr={nr}, smooth={smooth:g}")
print(f"  nearest-neighbour distance a = |A1| = {a_nn:.6f}")
def g_at_six(gfield):
    """g(a, theta) for the six first-shell directions, on the field's own grid."""
    gx, gy, g = gfield["x"], gfield["y"], gfield["g"]
    out = []
    for th in np.arange(0, 360, 60):
        p = a_nn * np.array([np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))])
        i, j = np.unravel_index(int(np.argmin((gx - p[0]) ** 2 + (gy - p[1]) ** 2)),
                                g.shape)
        out.append(g[i, j])
    return np.array(out)


for nm, g in (("crystal", gc), ("liquid", gl)):
    v = g_at_six({"x": gx, "y": gy, "g": g})
    print(f"  {nm:<8} g(a,theta) for theta = 0,60,...,300: {np.round(v, 3).tolist()}"
          f"   spread {v.max()-v.min():.3f}")
print(f"  (nearest grid index to each exact displacement; grid step "
      f"{float(2*rmax/nr):.4f} l_B)")

head("6c. CONTROL -- the same estimator on a sample set that is C6 BY CONSTRUCTION")
# Rotating every snapshot by 60*k and pooling the six copies makes the pair
# histogram exactly sixfold in displacement space, so ANY residual spread in
# g(a, theta) is the estimator's own C6 bias, not physics.
sub = snaps_c[:200]
sym = np.concatenate([sub @ rot(60.0 * k).T for k in range(6)])
vs = g_at_six(mod.pair_correlation_field(sym, tor))
print(f"  D6-symmetrised ({sym.shape[0]} configs = 200 crystal snapshots x 6 "
      f"rotations):")
print(f"    g(a,theta) = {np.round(vs, 4).tolist()}   spread {vs.max()-vs.min():.4f}"
      f"   <-- the estimator's OWN C6 bias")
print(f"  (for scale, the same 200 raw snapshots give "
      f"[{', '.join(f'{x:.4f}' for x in g_at_six(mod.pair_correlation_field(sub, tor)))}])")
print("  -> the delivered 1000-snapshot crystal spread must be read against this floor,")
print("     not against zero.")

head("6b. the same six directions in n(x, y)  (a Fourier measure: offset-proof)")
# A constant index offset only changes a PHASE of the Fourier amplitude, never its
# modulus, so |F|^2 at a family needs no calibration of the bin-centre convention.
# `x` in the npz is the ((nb+1)^2,) array of bin EDGES while the density array is
# the (nb^2,) set of bin CENTRES -- a naive (x - x0)/step map is wrong here.
with np.load(os.path.join(RES, "density_xy.npz")) as z:
    dc, dl = z["density_crystal"], z["density_liquid"]
    zL1, zL2 = z["L1"], z["L2"]
nb, mg = 72, 6
cell_dc, cell_dl = dc[mg:nb + mg, mg:nb + mg], dl[mg:nb + mg, mg:nb + mg]
sc = np.column_stack([zL1, zL2])
Brec = 2 * np.pi * np.linalg.inv(A).T          # columns b1, b2
fq = (np.arange(nb) + 0.5) / nb
Fr1, Fr2 = np.meshgrid(fq, fq, indexing="ij")
rcart = np.stack([Fr1.ravel(), Fr2.ravel()], axis=-1) @ sc.T
FAM = (Brec[:, 0], Brec[:, 1], Brec[:, 0] + Brec[:, 1])


def fam_amp(arr):
    c = arr / arr.mean() - 1.0
    return (np.array([abs(np.mean(c.ravel() * np.exp(-1j * (rcart @ q)))) ** 2
                      for q in FAM]), float(np.sqrt((c ** 2).mean())))


print(f"  padded density {dc.shape} = cell {nb}x{nb} + {mg} bins of periodic margin")
print(f"  crystal cell range (units of the mean) {cell_dc.min():.3f} .. {cell_dc.max():.3f}")
print()
print(f"  {'':<26} {'|F|^2(g1)':>10} {'|F|^2(g2)':>10} {'|F|^2(g1+g2)':>13} {'rms':>7}")
for nm, arr in (("crystal (delivered)", cell_dc),
                ("liquid (delivered)", cell_dl),
                ("D6-symmetrised control", mod.density_field(sym, tor)["rho"][mg:nb + mg,
                                                                              mg:nb + mg])):
    a3, r = fam_amp(arr)
    print(f"  {nm:<26} {a3[0]:10.6f} {a3[1]:10.6f} {a3[2]:13.6f} {r:7.4f}")
print()
print("  The control row is the DENSITY estimator's own C6 bias, measured on a set that")
print("  is sixfold by construction.  Read the crystal row against it.")

# ------------------------------------------------------ 7. seed anisotropy
head("7.  seed symmetry: the DETERMINISTIC one-body density of crystal_v0(L0=0.4)")
nb = 120
f = (np.arange(nb) + 0.5) / nb
F1, F2 = np.meshgrid(f, f, indexing="ij")
carts = np.stack([F1.ravel(), F2.ravel()], axis=-1) @ np.asarray(lat.C).T
w = area / len(carts)


def seed_density(v):
    orb = lr.LLRotatedOrbitals(vmc.basis(vmc.n_bands), vmc.n_bands, v)
    rho = np.array([np.sum(np.abs(orb.orbitals(p)) ** 2) for p in carts])
    return rho / rho.mean()


def ft(rho):
    return np.array([abs(np.sum(rho * np.exp(1j * (carts @ q))) * w * n_mean) ** 2 / NE ** 2
                     for q in G3])


v0 = np.asarray(vmc.crystal_v0(0.4), complex)
rho0 = seed_density(v0)
print(f"  crystal_v0(0.4): shape {v0.shape}, |v| min {np.abs(v0).min():.4f} "
      f"max {np.abs(v0).max():.4f}")
print(f"  |rho_g|^2/N^2 at g1+g2, g1, g2 = {np.round(ft(rho0), 6).tolist()}")
print(f"  ratio max/min = {ft(rho0).max()/ft(rho0).min():.2f}   "
      f"(a perfect crystal = 1:1:1; a filled LLL = 0:0:0)")
print(f"  density range (units of the mean) {rho0.min():.4f} .. {rho0.max():.4f}")

head("7b. the same, as a function of the Gaussian width L0 -- the axis FLIPS")
for L0 in (0.3, 0.4, 0.5, 0.8, 1.5):
    pw = ft(seed_density(np.asarray(vmc.crystal_v0(L0), complex)))
    print(f"  L0 = {L0:4.2f}: |rho_g|^2/N^2 = {np.round(pw, 6).tolist()!s:<40}"
          f"  ratio {pw.max()/pw.min():9.2f}")

head("7c. quadrature independence -- different numx must agree exactly")
# numx = 201 costs ~8 minutes on its own; pass --long to include it.  It was run
# and gives the same three numbers to all printed digits.
NUMS = (41, 101, 201) if LONG else (41, 101)
print(f"  (numx = {NUMS};  201 is behind --long)")
for numx in NUMS:
    ov = sr.gaussian_overlap_seed(vmc.basis(vmc.n_bands), vmc.n_bands,
                                  lat.L1, lat.L2, lat.ov_ai, lat.ov_ac,
                                  rs=75.0, numx=numx, L0=0.4)
    pw = ft(seed_density(np.asarray(sr.v_from_overlap(ov), complex)))
    print(f"  numx = {numx:4d} (grid step {np.linalg.norm(np.asarray(lat.L1))/numx:.4f}): "
          f"{np.round(pw, 6).tolist()}")

# ------------------------------------------- 7d. is the LL manifold C6-covariant?
head("7d. is the Landau-level Bloch manifold C6-covariant?  T_k(r) vs T_k(R^-1 r)")
ng = 30
fg = (np.arange(ng) + 0.5) / ng
F1g, F2g = np.meshgrid(fg, fg, indexing="ij")
pts = np.stack([F1g.ravel(), F2g.ravel()], axis=-1) @ np.asarray(lat.C).T


def basis_at(kv, upto=2):
    return llb.LandauLevelBasis(np.asarray(kv, float).reshape(1, 2), upto,
                                lat.ll_ai, lat.ll_ac, lat.prim_C, lat.prim_Ci)


def trace(b, rp):
    return np.array([np.sum(np.abs(b.orbitals(p)[0]) ** 2) for p in rp])


for i in (0, 1, 7, 13, 20, 31):
    k = np.asarray(lat.mesh[i], float)
    b = basis_at(k)
    T = trace(b, pts)
    row = []
    for deg in (0.0, 60.0, 120.0, 180.0):
        Rm = rot(deg)
        Tr = trace(b, (Rm.T @ pts.T).T)
        rel = np.abs(Tr - T).max() / T.mean()
        row.append(f"{deg:3.0f}:{rel:9.2e}{'' if rel < 1e-9 else '*'}")
    print(f"  k=({k[0]:+8.4f},{k[1]:+8.4f})  " + "  ".join(row))
print("  (* = NOT invariant;  theta = 0 is the identity and must read 0)")
b0 = basis_at(np.zeros(2), upto=0)
T0 = trace(b0, pts)
for deg in (0.0, 60.0, 180.0):
    Tr = trace(b0, (rot(deg).T @ pts.T).T)
    print(f"  LLL alone, k = 0, theta = {deg:3.0f}: max rel. change "
          f"{np.abs(Tr-T0).max()/T0.mean():.3e}")

head("7e. the proximate object:  magnetic_translation_phase = exp(i pi a1 a2)")
ai = np.asarray(lat.ll_ai)
ac = np.asarray(lat.ll_ac)
ph = magnetic_translation_phase(ai, 1.0)
print(f"  ll_ai {ai.shape}, phases take the values {np.unique(np.round(ph, 9)).tolist()}")
keys = {tuple(np.round(v, 9)): i for i, v in enumerate(ac)}
for deg in (60.0, 120.0, 180.0):
    Rm = rot(deg)
    miss = bad = 0
    worstp = 0.0
    for i, vec in enumerate(ac):
        j = keys.get(tuple(np.round(Rm @ vec, 9)))
        if j is None:
            miss += 1
            continue
        d = abs(ph[j] - ph[i])
        bad += d > 1e-12
        worstp = max(worstp, d)
    print(f"  R({deg:5.1f}): missing {miss}, phase mismatch {bad} of {len(ac)}, "
          f"worst |dphase| = {worstp:.3e}")
print("  the (A1,A2) frame's point group is {E, R(180), sigma_x, sigma_y} = D2 --")
print("  exactly the group seen in S(q): pm g1 and pm(g1+g2) exchanged by sigma_x")
print("  and exactly equal; pm g2 invariant under both and enhanced.")

# ------------------------------------------------ 9. estimator controlled input
head("9.  the estimator on controlled inputs")
# `structure_factor(snaps, qvecs, ne)` takes a SNAPSHOT BATCH: snaps is
# (n_snap, n_particles, 2) and S(q) = <|sum_j e^{i q.r_j}|^2> / n_particles.
# A bare (36, 2) lattice would be read as 36 snapshots of ONE particle.
sites = np.asarray(st.lattice_sites(tor) @ tor.sc.T, float)
print(f"  perfect triangular lattice, no jitter: "
      f"{np.round(st.structure_factor(sites[None], G3, NE), 6).tolist()}")
rng = np.random.default_rng(5)
jit = sites + 0.15 * rng.normal(size=sites.shape)
v = st.structure_factor(jit[None], G3, NE)
print(f"  same lattice + 0.15 isotropic jitter : {np.round(v, 6).tolist()}"
      f"   spread {v.max()-v.min():.4f}")
j6 = np.concatenate([jit @ rot(60.0 * k).T for k in range(6)])
v = st.structure_factor(j6[None], G3, NE)
print(f"  that jittered set D6-SYMMETRISED      : {np.round(v, 6).tolist()}"
      f"   spread {v.max()-v.min():.1e}   <-- exact")
v = st.structure_factor(np.concatenate([snaps_c @ rot(60.0 * k).T for k in range(6)]),
                        G3, NE)
print(f"  the 1000 crystal snapshots ROTATED and pooled: {np.round(v, 6).tolist()}"
      f"   spread {v.max()-v.min():.1e}   <-- exact")
v = st.structure_factor(snaps_c, G3, NE)
print(f"  the SAME 1000 snapshots as delivered : {np.round(v, 6).tolist()}"
      f"   A = {(v.max()-v.min())/v.mean():.6f}")
print("  -> the estimator gives sixfold for sixfold input and reproduces the delivered")
print("     inequality on the delivered samples.  Estimator excluded.")

# -------------------------------------------------------- 10. significance
head("10. error bars on S3 - S1 and S2 - S1 (existing snapshots only)")
per = st.structure_factor_snapshots(snaps_c, G3)
mean = per.mean(axis=0)
sem = st.snapshot_sem(per, 1)
nblk = 20
blk = np.array_split(per, nblk)
bsem = np.array([b.mean(axis=0) for b in blk]).std(axis=0, ddof=1) / np.sqrt(nblk)
print(f"  per-snapshot SEM (ddof=1): {np.round(sem, 4).tolist()}")
print(f"  20-block SEM            : {np.round(bsem, 4).tolist()}")
for lab, x, y in (("S3 - S1", 2, 0), ("S2 - S1", 1, 0), ("S1 - S2", 0, 1)):
    d = mean[x] - mean[y]
    for nm, e in (("snapshot", np.hypot(sem[x], sem[y])),
                  ("20-block", np.hypot(bsem[x], bsem[y]))):
        print(f"  crystal {lab} = {d:+.4f} +- {e:.4f}  "
              f"({abs(d)/e:.2f} sigma, {nm})")

print()
print("=" * 100)
print("  DONE -- read-only; nothing under results/ or figures/ was written.")
print("=" * 100)
