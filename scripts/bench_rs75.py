"""Stage 2E: the r_s = 75 benchmark, run end to end by the CLEAN package.

What this is for
----------------
Stages 2A-2C ported the engine and verified it function by function (B1-B7).  This
script is the first time the clean package runs the whole production protocol on
its own, so that the frozen legacy point can be reproduced as a *number* and not
only as a set of agreeing functions:

    liquid          LL ansatz with v = 0  (the filled-Landau-level reference)
    crystal nmax=1  n_bands = 2
    nested nmax=2   n_bands = 3, started from the clean nmax=1 state

and the comparison  delta_E = E_crystal - E_liquid,  per electron.

What is legacy and what is clean
--------------------------------
The clean package does ALL of the work: the sampler, the local energy, the Ewald
sum, the LL orbitals, the joint stochastic reconfiguration and the production
walk.  The frozen notebook supplies only the STARTING POINT -- the warm-started
Jastrow c0 and the projected-Gaussian v0 -- which `--init` reads from a JSON
dump.  That is deliberate: "the same initialization semantics" is a statement
about where the run begins, and reading the beginning as data keeps the physics
under test entirely on the clean side.

The legacy reference values are NEVER hard-coded here.  `--legacy` reads them
from the frozen `_diag/part1/` records if they are present, and the comparison
prints a z-score; it does not tune anything toward them.

Budgets
-------
`regression` is the legacy budget: SR 120 steps x 400 sweeps (equil 150, sigma
0.3, target_acc 0.4, snapshot every 4) and a production walk of 1500 sweeps
(equil 400, snapshot every 1).  `smoke` and `quick` exist to prove the pipeline
runs; they are NOT converged and must not be compared against the legacy number.

    python scripts/bench_rs75.py --budget smoke
    python scripts/bench_rs75.py --budget regression --phase liquid
    python scripts/bench_rs75.py --budget regression --phase crystal
    python scripts/bench_rs75.py --budget regression --phase nested
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wigner_vmc.analysis.statistics import tau_int                  # noqa: E402
from wigner_vmc.physics import coulomb as cb                        # noqa: E402
from wigner_vmc.physics import geometry as ge                       # noqa: E402
from wigner_vmc.physics import landau_levels as llb                 # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw                  # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr              # noqa: E402
from wigner_vmc.wavefunctions import nesting as nest                # noqa: E402
from wigner_vmc.vmc import sr as sr                                 # noqa: E402
from wigner_vmc.vmc.sampler import sample                           # noqa: E402

DEFAULT_INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
DEFAULT_OUT = os.path.join(CLEAN, "results", "bench_rs75")
LEGACY_DIR = os.path.join(os.path.dirname(CLEAN), "_diag", "part1")

#: The legacy protocol, per budget.  `regression` is the notebook's own
#: `_CONV_STEPS/_CONV_SWEEPS = 120, 400` plus the 1500/400 production walk.
PROTOCOL = {
    "smoke": dict(sr_steps=2, sr_sweeps=20, sr_equil=6, sr_snap=1,
                  meas_sweeps=200, meas_equil=60),
    "quick": dict(sr_steps=10, sr_sweeps=40, sr_equil=13, sr_snap=2,
                  meas_sweeps=400, meas_equil=150),
    "regression": dict(sr_steps=120, sr_sweeps=400, sr_equil=150, sr_snap=4,
                       meas_sweeps=1500, meas_equil=400),
}
SR_SIGMA, SR_TARGET = 0.3, 0.4
CRY_MEAS_SEED, CRY_MEAS_SIGMA = 17, 0.3
LIQ_MEAS_SEED, LIQ_MEAS_SIGMA = 11, 0.4
NJ = 5


# -- kappa: a physics convention and a campaign protocol are not the same thing
def physical_kappa(rs):
    """The clean physics convention: ``r_s = sqrt(2) * kappa``, full precision.

    Kept as a DIVISION by ``sqrt(2)`` rather than delegated to
    ``physics.hamiltonian.kappa_from_rs`` (which multiplies by ``sqrt(1/2)``).
    The two agree to 1 ULP, but they do not agree exactly -- 53.03300858899106
    against ...107 at r_s = 75 -- and this is the clean tree's own convention,
    the one `initial_conditions.json` already carries.  Delegating would silently
    move every clean calculation by one bit, which is precisely the kind of
    unannounced drift these two functions exist to prevent.
    """
    return float(rs) / math.sqrt(2.0)


def legacy_regression_kappa(rs):
    """The frozen campaign's kappa -- for exact-reproduction runs ONLY.

    ``_diag/part1_scan.py:414`` builds every job with
    ``kappa = round(rs / np.sqrt(2.0), 4)``, so at r_s = 75 the published record
    used 53.033 where the physics gives 53.03300858899106.  That is 8.6e-6 in
    kappa and 2.3e-4 in E: small, but it is the WHOLE of the residual once the
    claim is that a walk reproduces the record exactly, so it has to be a named
    mode rather than a fudge.  It is provenance, not a definition -- the clean
    Hamiltonian never rounds.
    """
    return float(round(float(rs) / np.sqrt(2.0), 4))


# --------------------------------------------------------------------------
class Setup:
    """The frozen geometry + the clean objects built on it."""

    def __init__(self, init, kappa=None, kappa_mode="physical"):
        self.raw = init
        self.rs = float(init["rs"])
        # `kappa` is a PROTOCOL knob.  Default is the physics; the campaign's
        # rounded value is opt-in and is carried in `kappa_mode` so every result
        # can record which one produced it.
        self.kappa = float(init["kappa"]) if kappa is None else float(kappa)
        self.kappa_mode = kappa_mode
        self.ne = int(init["ne"])
        self.n_band_B = int(init["NMAX_STAGE_B"])
        g = lambda k: np.asarray(init[k], float)                     # noqa: E731
        self.L1, self.L2 = g("L1"), g("L2")
        self.G1, self.G2 = g("G1"), g("G2")
        self.mesh = g("mesh")
        # the SIMULATION SUPERCELL frame -- the notebook's `sc_to_cart`.  The
        # primitive cell the basis needs is a different matrix; see below.
        self.C = np.column_stack([self.L1, self.L2])
        self.Ci = np.linalg.inv(self.C)
        self.sites = g("sites")
        self.nk = self.mesh.shape[0]

        # TWO lattice-vector sets, for two different roles.  They are NOT
        # interchangeable and the notebook keeps them strictly apart; an earlier
        # revision of this class carried a single `self.ai/self.ac` filled from
        # the overlap set, which changed phi_{k,n} by 73% relative and flipped
        # the sign of T on the r_s=75 liquid (BLOCKER_A_FINDING.md).
        #
        #   ll_ai/ll_ac  the LANDAU-LEVEL Bloch sum: circular_lattice(LUMAX_LL,
        #                A1, A2) on the PRIMITIVE cell.  The notebook builds every
        #                LandauLevelBasis with it, and chose LUMAX_LL from the
        #                convergence scan against a cutoff-60 reference.
        #   ov_ai/ov_ac  the GAUSSIAN-OVERLAP image sum: circular_lattice(LUMAX,
        #                L1, L2) on the SUPERCELL -- the notebook's `_lints8`.
        #
        # They are named for their roles rather than for the notebook's variable
        # names, so that re-copying one into the other's place is visibly wrong.
        self.A1, self.A2 = self.L1 / 6.0, self.L2 / 6.0
        self.ll_ai, self.ll_ac = ge.circular_lattice(
            float(init["LUMAX_LL"]), self.A1, self.A2)
        self.ov_ai, self.ov_ac = g("_lints8"), g("_lcart8")

        # TWO cell matrices, for two different roles -- the same split as the two
        # lattice-vector sets above, and the second Blocker A defect.
        #
        #   prim_C/prim_Ci  the PRIMITIVE cell [A1, A2].  `LandauLevelBasis`
        #                   uses its matrix for exactly one thing: folding
        #                   `r - k x zhat` into the first cell before the Bloch
        #                   sum.  That sum runs over `a_cart`, the LL lattice
        #                   vectors -- a disc of radius ~15 -- so the folding cell
        #                   must be the SAME lattice, or an electron at large |r|
        #                   is summed where the disc is truncated.  The notebook's
        #                   `CToCart = np.column_stack([A1, A2])` (cell 10), used
        #                   at every one of its 19 LandauLevelBasis call sites.
        #   C/Ci            the SIMULATION SUPERCELL [L1, L2] -- the many-electron
        #                   wavefunction's own frame and the sampler's wrap cell,
        #                   the notebook's `sc_to_cart`.
        #
        # Using the supercell for the basis moves phi_{k,n}(r) by up to 2.0
        # RELATIVE (a sign flip) on electrons that fold outside the primitive
        # cell, and leaves sigma, acc, T/N and E/N all off
        # (BLOCKER_A_FINDING.md, second divergence).
        self.prim_C = np.column_stack([self.A1, self.A2])
        self.prim_Ci = np.linalg.inv(self.prim_C)
        # (C/Ci, the simulation supercell, were set above from L1/L2.)

        # the clean geometry must agree with the notebook's own frame, or the
        # comparison is between two different tori.
        sc = g("sc_to_cart")
        assert np.allclose(sc, self.C, rtol=0, atol=1e-12), \
            "clean frame != notebook sc_to_cart"

        self.HAM = cb.CoulombEwald(self.ne, self.L1, self.L2, self.G1, self.G2)

    @classmethod
    def legacy_regression(cls, init):
        """A ``Setup`` at the frozen campaign's rounded kappa, for exact
        reproduction only.  ``kappa_mode`` travels with it so the result can say
        which convention produced it -- provenance, not a physics choice."""
        return cls(init, kappa=legacy_regression_kappa(init["rs"]),
                   kappa_mode="legacy-regression")

    def jastrow(self, c):
        return jw.SinSplineJastrow(np.asarray(c, float), self.G1, self.G2,
                                   jw.cusp_gamma(self.kappa, self.L1))

    def basis(self, n_band):
        """The LL Bloch basis -- the notebook's `_aiB/_acB` and `CToCart`.

        Both arguments are PRIMITIVE-cell quantities and neither may be swapped
        for its supercell counterpart:

          * ``ll_ai/ll_ac``  the lattice vectors the Bloch sum runs over;
          * ``prim_C/prim_Ci`` the cell that sum folds into.

        ``self.C/self.Ci`` (the simulation supercell) is the wavefunction's own
        frame, set in ``maker``, and is wrong here.
        """
        return llb.LandauLevelBasis(self.mesh, n_band - 1,
                                    self.ll_ai, self.ll_ac,
                                    self.prim_C, self.prim_Ci)

    def maker(self, n_band):
        """``make_wf(theta)`` in the legacy theta ordering."""
        m = (n_band - 1) * self.nk

        def _mk(theta):
            theta = np.asarray(theta, float)
            v = (theta[NJ:NJ + m].reshape(self.nk, n_band - 1)
                 + 1j * theta[NJ + m:].reshape(self.nk, n_band - 1))
            orb = lr.LLRotatedOrbitals(self.basis(n_band), n_band, v)
            wf = lr.LLRotationWavefunction(orb, self.jastrow(theta[:NJ]),
                                           self.ne, self.C,
                                           kappa=self.kappa, ham=self.HAM)
            # provenance, not physics: which kappa convention built this wf.
            # The wavefunction itself has no opinion; the result must be able to
            # say whether it was the physics or the campaign's rounded value.
            wf.kappa_mode = self.kappa_mode
            return wf
        return _mk

    # -- the legacy starting points, read as data ---------------------------
    def liquid_R0(self):
        return (np.random.default_rng(3).random((self.ne, 2)) - 0.5) @ self.C.T

    def crystal_R0(self):
        """``crystal_sites_seed(kappa, 0.5, 100 + int(kappa))`` -- deterministic."""
        rng = np.random.default_rng(100 + int(self.kappa))
        return np.array(self.sites) + 0.25 * rng.standard_normal((self.ne, 2))

    def seed_state(self, si, n_band=None):
        """(c0, v0) of conv-tier seed ``si``, zero-padded to ``n_band``."""
        nb = self.n_band_B if n_band is None else n_band
        d = self.raw["crystal"][f"s{si}"]
        c0 = np.asarray(d["c0"], float)
        v0 = np.asarray(d["v0_real"], float) + 1j * np.asarray(d["v0_imag"], float)
        if v0.shape[1] != nb - 1:
            v0 = nest.pad_v(v0, nb)
        return c0, v0

    def theta(self, c, v):
        v = np.asarray(v, complex)
        return np.concatenate([np.asarray(c, float).ravel(),
                               v.real.ravel(), v.imag.ravel()])


# --------------------------------------------------------------------------
def theta_parts(theta, n_band, nk):
    m = (n_band - 1) * nk
    v = (theta[NJ:NJ + m].reshape(nk, n_band - 1)
         + 1j * theta[NJ + m:].reshape(nk, n_band - 1))
    return np.asarray(theta[:NJ], float), v


def production_walk(wf, R0, sweeps, equil, sigma, seed, label):
    """One walk with every snapshot decomposed -- the clean counterpart of the
    legacy ``walk_and_decompose``.  E is accumulated PER SNAPSHOT as T + kappa*V,
    so the error bar is the error of the quantity of interest rather than the
    quadrature sum of two strongly anti-correlated halves.

    T and V come back from ``local_energy`` as TOTALS for the whole configuration
    (they are the sum over electrons), so every mean is divided by ``ne`` exactly
    once, at the point of reporting.  ``_total`` keys carry the undivided values so
    the two conventions can never be confused in the output file.

    The autocorrelation correction is ``sigma_naive * sqrt(tau)``.  That is the
    form the frozen campaign used -- verified against the legacy record, whose
    ``E_err / E_err_naive`` equals ``sqrt(E_tau)`` to 4 decimals -- and it is the
    form notebook Part V concludes is right (the ``sqrt(2*tau)`` variant
    double-counts by exactly sqrt(2)).  ``E_tau`` is stored so this is checkable.
    """
    snaps, sig, acc = sample(wf, R0, nsweep=sweeps, sigma=sigma,
                             rng=np.random.default_rng(seed), snapshot_every=1,
                             equil=equil, target_acc=0.4)
    n = len(snaps)
    T = np.empty(n)
    V = np.empty(n)
    for s, S in enumerate(snaps):
        t, v, _ = wf.local_energy(wf.build(S))
        T[s], V[s] = t.real, v.real
    E = T + wf.kappa * V
    ne = wf.ne
    tau_E, win_E = tau_int(E)
    tau_V, _ = tau_int(V)
    return {
        "label": label, "n": int(n), "acc": float(acc), "sigma": float(sig),
        # the kappa that ACTUALLY produced these numbers.  Recording only r_s
        # would not distinguish the physical convention from the campaign's
        # rounded one, and that difference alone moves E by 2.3e-4.
        "kappa": float(wf.kappa),
        "kappa_mode": getattr(wf, "kappa_mode", None),
        # per electron -- what the legacy records call E_perpart
        "E": float(E.mean() / ne),
        "E_err": float(E.std(ddof=1) / np.sqrt(n) * np.sqrt(tau_E) / ne),
        "E_err_naive": float(E.std(ddof=1) / np.sqrt(n) / ne),
        "E_tau": float(tau_E),
        "E_tau_window": int(win_E),
        "V": float(V.mean() / ne),
        "V_err": float(V.std(ddof=1) / np.sqrt(n) * np.sqrt(tau_V) / ne),
        "T": float(T.mean() / ne),
        # totals -- the legacy records' own convention for the bare names
        "E_total": float(E.mean()),
        "T_total": float(T.mean()),
        "V_total": float(V.mean()),
        "cov_TV": float(np.cov(T, V, ddof=1)[0, 1]),
    }


def run_crystal(S, n_band, c0, v0, seed, proto, label, verbose=True):
    mk = S.maker(n_band)
    th0 = S.theta(c0, v0)
    R0 = S.crystal_R0()
    t0 = time.time()
    th, hist = sr.sr_optimize_joint(
        mk, th0, R0, steps=proto["sr_steps"], nsweep=proto["sr_sweeps"],
        sigma=SR_SIGMA, seed=seed, snapshot_every=proto["sr_snap"],
        equil=proto["sr_equil"], target_acc=SR_TARGET)
    sr_s = time.time() - t0
    if not np.all(np.isfinite(th)):
        return {"label": label, "failed": "theta non-finite"}
    wf = mk(th)
    rec = production_walk(wf, S.crystal_R0(), proto["meas_sweeps"],
                          proto["meas_equil"], CRY_MEAS_SIGMA, CRY_MEAS_SEED, label)
    c, v = theta_parts(th, n_band, S.nk)
    C = lr.c_row(v)
    P = (np.abs(C) ** 2).mean(axis=0)
    eh = np.array([h["E"] for h in hist]) / S.ne
    rec.update(
        n_band=n_band, kappa=S.kappa, rs=S.rs, seed=seed, theta=th.tolist(),
        c=c.tolist(), v_real=v.real.tolist(), v_imag=v.imag.tolist(),
        E_start=float(eh[0]), E_final=float(eh[-1]),
        P=P.tolist(), nbar=float((np.arange(n_band) * P).sum()),
        sr_seconds=float(sr_s), sr_steps=proto["sr_steps"],
    )
    if verbose:
        print(f"    SR {proto['sr_steps']}x{proto['sr_sweeps']} in {sr_s:.0f}s   "
              f"E_start {eh[0]:+.6f} -> E_final {eh[-1]:+.6f}   "
              f"walk E {rec['E']:+.6f} +- {rec['E_err']:.6f}  n={rec['n']} "
              f"acc={rec['acc']:.3f}", flush=True)
    return rec


def run_liquid(S, proto, label, verbose=True):
    c = np.asarray(S.raw["c_liquid"], float)
    v = np.zeros((S.nk, S.n_band_B - 1), dtype=complex)
    wf = S.maker(S.n_band_B)(S.theta(c, v))
    rec = production_walk(wf, S.liquid_R0(), proto["meas_sweeps"],
                          proto["meas_equil"], LIQ_MEAS_SIGMA, LIQ_MEAS_SEED, label)
    rec.update(n_band=S.n_band_B, kappa=S.kappa, rs=S.rs, c=c.tolist(),
               v_real=v.real.tolist(), v_imag=v.imag.tolist())
    if verbose:
        print(f"    walk E {rec['E']:+.6f} +- {rec['E_err']:.6f}  n={rec['n']} "
              f"acc={rec['acc']:.3f}  sigma={rec['sigma']:.4f}", flush=True)
    return rec


# --------------------------------------------------------------------------
def legacy_ref(name):
    p = os.path.join(LEGACY_DIR, name)
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return {"E": d.get("E_perpart"), "E_err": d.get("E_perpart_err"),
            "n": d.get("n")}


def compare(rec, ref, what, budget):
    if budget != "regression":
        print(f"    {what}: budget {budget!r} is not converged -- not compared "
              f"against the legacy r_s=75 record")
        return None
    if ref is None or ref.get("E") is None:
        print(f"    {what}: no legacy record to compare against")
        return None
    d = rec["E"] - ref["E"]
    s = float(np.hypot(rec["E_err"], ref.get("E_err") or 0.0))
    z = abs(d) / s if s > 0 else float("inf")
    print(f"    {what}: clean {rec['E']:+.6f} +- {rec['E_err']:.6f}   "
          f"legacy {ref['E']:+.6f} +- {ref['E_err']:.6f}   "
          f"diff {d:+.6f}  z={z:.2f}")
    return {"clean": rec["E"], "legacy": ref["E"], "diff": d, "z": float(z)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", choices=sorted(PROTOCOL), default="smoke")
    ap.add_argument("--phase", default="all",
                    choices=["all", "liquid", "crystal", "nested"])
    ap.add_argument("--seed", type=int, default=0,
                    help="which conv-tier starting point (0..4); L0 = _L0S[seed]")
    ap.add_argument("--init", default=DEFAULT_INIT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--parent", default=None,
                    help="nested only: the clean nmax=1 result JSON to nest from")
    args = ap.parse_args(argv)

    proto = PROTOCOL[args.budget]
    S = Setup(json.load(open(args.init, encoding="utf-8")))
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.budget}_s{args.seed}"

    print("resolved configuration")
    for k, v in (("N", S.ne), ("phase", args.phase), ("rs", S.rs),
                 ("kappa", S.kappa), ("nmax", S.n_band_B - 1),
                 ("n_bands", S.n_band_B), ("init seed", args.seed),
                 ("L0", S.raw["_L0S"][args.seed]), ("budget", args.budget),
                 ("SR steps", proto["sr_steps"]), ("SR sweeps", proto["sr_sweeps"]),
                 ("SR equil", proto["sr_equil"]), ("SR sigma", SR_SIGMA),
                 ("SR target_acc", SR_TARGET),
                 ("measurement sweeps", proto["meas_sweeps"]),
                 ("measurement equil", proto["meas_equil"]),
                 ("crystal walk seed", CRY_MEAS_SEED),
                 ("liquid walk seed", LIQ_MEAS_SEED),
                 ("out", os.path.abspath(args.out))):
        print(f"  {k:22s} {v}")
    print(flush=True)

    summary = {"budget": args.budget, "phase": args.phase, "seed": args.seed,
               "protocol": proto, "rs": S.rs, "kappa": S.kappa,
               "n_bands_B": S.n_band_B}

    if args.phase in ("all", "liquid"):
        print("[liquid]")
        rec = run_liquid(S, proto, f"liq_{tag}")
        json.dump(rec, open(os.path.join(args.out, f"liquid_{tag}.json"), "w"),
                  indent=1, sort_keys=True)
        summary["liquid_E"] = rec["E"]
        summary["liquid_err"] = rec["E_err"]
        summary["liquid"] = compare(rec, legacy_ref("liq_rs75.json"), "liquid",
                                    args.budget)

    cry = None
    if args.phase in ("all", "crystal", "nested"):
        print("[crystal nmax=1]")
        c0, v0 = S.seed_state(args.seed)
        cry = run_crystal(S, S.n_band_B, c0, v0, args.seed, proto, f"cry_{tag}")
        json.dump(cry, open(os.path.join(args.out, f"crystal_{tag}.json"), "w"),
                  indent=1, sort_keys=True)
        summary["crystal_E"] = cry.get("E")
        summary["crystal_err"] = cry.get("E_err")
        summary["crystal"] = compare(
            cry, legacy_ref(f"cry_conv_rs75_nb2_s{args.seed}.json"), "crystal nb=2",
            args.budget)

    if args.phase in ("all", "nested"):
        print("[nested crystal nmax=2]")
        parent = args.parent or os.path.join(args.out, f"crystal_{tag}.json")
        if not os.path.exists(parent):
            print(f"  MISSING PARENT {parent} -- not nesting, and NOT falling back "
                  f"to a fresh start.  Run --phase crystal first.")
            summary["nested"] = "MISSING PARENT"
        else:
            pj = json.load(open(parent, encoding="utf-8"))
            c0 = np.asarray(pj["c"], float)
            v0 = (np.asarray(pj["v_real"], float)
                  + 1j * np.asarray(pj["v_imag"], float))
            nb = pj["n_band"] + 1
            head, tail = nest.coefficient_identity_deviation(v0, nb)
            print(f"  nesting identity: max|C_nest[:, :m+1] - C_src| = {head:.3e}"
                  f"   max|C_nest[:, m+1:]| = {tail:.3e}")
            if head != 0.0 or tail != 0.0:
                print("  NESTING IDENTITY IS NOT EXACT -- aborting rather than "
                      "falling back to a fresh initialisation.")
                summary["nested"] = "identity failed"
            else:
                v0 = nest.pad_v(v0, nb)
                nrec = run_crystal(S, nb, c0, v0, args.seed, proto, f"nest_{tag}")
                json.dump(nrec, open(os.path.join(args.out, f"nested_{tag}.json"),
                                     "w"), indent=1, sort_keys=True)
                summary["nested_E"] = nrec.get("E")
                summary["nested_err"] = nrec.get("E_err")
                summary["nested"] = compare(
                    nrec, legacy_ref(f"cry_nest_rs75_nb3_s{args.seed}.json"),
                    "crystal nb=3", args.budget)

    eL = summary.get("liquid_E")
    if eL is not None:
        for key, label in (("nested_E", "nmax=2"), ("crystal_E", "nmax=1")):
            if summary.get(key) is not None:
                dE = summary[key] - eL
                print(f"\ndelta_E = E_crystal({label}) - E_liquid = {dE:+.6f}")
                summary["delta_E_" + key[:-2]] = dE

    json.dump(summary, open(os.path.join(args.out, f"summary_{tag}.json"), "w"),
              indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
