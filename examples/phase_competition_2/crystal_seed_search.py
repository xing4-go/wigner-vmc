"""Stage 1A -- a Haar-random crystal seed at large ``r_s``.

    python examples/phase_competition_2/crystal_seed_search.py --rs 90

What this is, and what it is not
--------------------------------
The paper's crystal seed is a **randomly initialized determinant**.  Reddy & Fu,
PRB 113, L161403 (2026) (arXiv:2508.21000), section "Variational Monte Carlo
methods", verbatim::

    "For the Wigner crystal phase, we generate crystal 'seed' orbitals by
     optimizing a randomly initialized determinant with a Jastrow factor at
     large r_s."

Every crystal point this project has delivered instead starts from
``VMC.crystal_v0`` -> ``sr.gaussian_overlap_seed`` -> ``sr.v_from_overlap``, a
site-centred Gaussian projected onto the kept Landau levels.  The paper uses
that construction only as an independent **benchmark** (SM III D), never as the
seed.  Those two names are therefore deliberately *absent from this file's code*
and named here so the ban is not vacuous; the Gaussian construction survives
below only as a **reference run** (``VMC.run``), which is a control and never
feeds a random start.

This recipe is **Stage 1A: a pipeline validation**, not a result.  ``quick`` is
SR 10x40, and a fully random determinant asks the optimiser to find
translational-symmetry breaking, triangular order, the right Landau-level mixing
and the right Jastrow correlation at the same time.  If none of the starts forms
a Wigner crystal, that is **not** evidence the basin is unreachable -- it is
equally consistent with the optimiser not having had time.  No scientific claim
is made in either direction.  Convergence of independent random starts is Stage
1B, at ``reproduction``, and is a separate decision.

The random determinant, and why Haar
------------------------------------
At ``N = 36`` there are ``nk = 36`` momenta; at ``n_max = 1`` the orbital matrix
``C = lr.c_row(v)`` gives ``C[k] = (cos t_k, i sin t_k e^{i phi_k})`` with
``t_k = |v_k|``.  So per momentum the manifold is **CP1 in ``(t_k, phi_k)``**,
and the excited-band weight of orbital ``k`` is exactly ``sin^2 t_k``.  ``t = 0``
is the pure lowest Landau level -- which is why a uniform draw is not a
disguised Gaussian.

The Fubini-Study measure on CP1 is ``sin(2t) dt dphi``, whose CDF is
``sin^2 t ~ U[0,1]``.  Haar is therefore the **least arbitrary** reading of
"a random determinant", and it carries no scale knob::

    t_k = arcsin(sqrt(u_k)),  u_k ~ U[0,1]        E[sin^2 t] = 1/2
    phi_k ~ U[0, 2*pi)                    i.i.d. over the momenta

There are deliberately no ``t_max`` arms.  Whether the *measure* changes the WC
basin is a different scientific question; here the only random variable is the
random determinant realization.

``n_max = 1`` is required, not defaulted: the CP1 statement above is what makes
the draw canonical, and it does not generalise to a product of independent
per-column draws at higher ``n_max`` without saying so.

The WC criterion, fixed before any run
--------------------------------------
``bragg_ratio`` is an absolute number whose scale depends on budget and state, so
the threshold is calibrated against the **liquid at the same ``r_s``, measured in
the same run** -- the repo's own established move (``structure.py``, on why the
liquid columns are the control).  A trial counts as a triangular WC iff all
three hold:

  1. ``bragg_ratio >= BRAGG_GATE * bragg_ratio_liquid``
  2. ``c6_residual <= C6_GATE``
  3. ``site_midpoint_contrast >= CONTRAST_GATE``, on the density smoothed at the
     historical 0.40 ``l_B`` -- on the RAW histogram the nearest-neighbour
     midpoints of a sharp lattice fall in empty bins and the statistic is
     ``inf``, which would pass this gate for free

Gate 2 is the primary symmetry gate and it is a different kind of object from
the anisotropy ``A`` that is *reported* alongside it.  ``c6_residual`` is the
normalized least-squares residual of the rotated occupied space against the
unrotated one (``test_projection_convention.py``'s ``_space_residual``): bounded
in ``[0, 1]``, invariant under a change of orbital gauge, and measured at
3.0e-14 for a C6-covariant seed against 7.9e-01 for a C6-broken one.  ``A`` is a
ratio of two noisy ``S`` values whose denominator is small in the liquid, so it
is unfit to be a gate and is never thresholded here.

**Escape hatch.**  If the Gaussian-seed reference itself fails gate 2, the run
reports ``NO VERDICT`` -- the budget cannot resolve the gate -- rather than
silently returning ``0/4``.  That is the one outcome that must not be read as
physics.

Root selection, fixed before any run
------------------------------------
Among trials with ``wc_formed`` true, the root is the one with the lowest
``E_final``.  If none pass, the run reports ``NO ROOT`` and does not relax the
criterion and does not fall back to the best-energy trial.

Where things live
-----------------
    examples/phase_competition_2/                                    this recipe
    results/phase_competition_2/crystal_seed_search/<slug>/          the numbers
    figures/phase_competition_2/crystal_seed_search/<slug>/          the QC picture

The table is the deliverable; the figure is quality control.  Both are stamped
unless ``--budget reproduction``, so a smoke test cannot be mistaken for a
result.

What this recipe does not read
------------------------------
It recomputes everything and reads no stored result of any kind: not the
defective LL-rotation crystal records (``legacy_llrot``), not the analysis store
derived from them (``energy_scan``), not the scan driver (``_diag/part1``), and
it does not treat the frozen ``make_notebook.py`` as a specification.  Each of
those names appears here, in the prose, precisely because it must not appear in
the code -- the contract test checks the second while this paragraph keeps the
first from being vacuous.

BLAS threading
--------------
The same crystal point evaluates to ``-37.793807191316347`` with the BLAS thread
counters pinned to 1 and ``-37.793807191311664`` without -- 4.7e-12 from
reduction order alone.  That is far below anything physical, but it means a
stored baseline is only a bit-level baseline under the threading it was made
with.  The banner and the metadata record the configuration.
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

from wigner_vmc import VMC, __version__, load_budget, resolve, theta, theta_parts
from wigner_vmc.analysis import structure as st
from wigner_vmc.vmc import sr
from wigner_vmc.vmc.measure import measure_decomposed
from wigner_vmc.wavefunctions import ll_rotation as lr

#: ``NJ = 5``, the Jastrow depth.  It lives in ``api`` and is deliberately not
#: re-exported; importing it rather than writing a second ``5`` here is what
#: keeps ``len(theta)`` and ``sr.jastrow_vector`` from drifting apart.
from wigner_vmc.api import NJ

# ==========================================================================
# where things live
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))          # .../wigner_vmc_clean

RESULTS = os.path.join(CLEAN, "results", "phase_competition_2", "crystal_seed_search")
FIGDIR = os.path.join(CLEAN, "figures", "phase_competition_2", "crystal_seed_search")

# ==========================================================================
# the run
# ==========================================================================
N_ELECTRONS = 36
DEFAULT_RS = 90.0
DEFAULT_NMAX = 1
DEFAULT_TRIALS = 4
DEFAULT_SEED_BASE = 20261004
DEFAULT_BUDGET = "quick"

#: The paper specifies no NUMERIC seeding coupling -- only "at large r_s" (the
#: r_s = 60 and 80 in it are data points, not the seeding coupling).  So this is
#: a choice, and it is recorded as one.
DEFAULT_RS_NOTE = (
    "the paper gives no numeric seeding coupling, only 'at large r_s'; "
    "r_s = 90 is this recipe's choice")

# ==========================================================================
# the criterion constants -- the only arbitrary numbers here
# ==========================================================================
#: "Sufficiently C6-symmetric": the occupied subspace is within 10% of invariant.
#: On a bounded [0, 1] order parameter that is a real statement, and the run
#: prints it next to both references so it can be judged rather than trusted.
C6_GATE = 0.1

#: The trial's Bragg ratio must clearly exceed the disordered reference's.
BRAGG_GATE = 3.0

#: Real-space corroboration, from an estimator that is exactly 1 for anything
#: uniform and does not reuse `S(q)`.
CONTRAST_GATE = 2.0

#: The five rotations a C6 state must survive.  180 degrees alone is not enough:
#: it is exactly the one that survived the reversed-conjugation defect.
C6_ROTS = (60.0, 120.0, 180.0, 240.0, 300.0)
C6_NSP = 500
C6_SEED = 20261003

#: Real-space density grid, in bins per supercell side, and the Gaussian width it
#: is smoothed at in units of the magnetic length.  Both are the historical
#: values (`examples/figure_construction/structure.py:135-136`, pinned by
#: `tests/test_phase_structure_contract.py`), NOT tuned here.
#:
#: The smoothing is not cosmetic and it is not optional.  `density_grid` is a
#: hard histogram, so at 72 bins the nearest-neighbour midpoints of a sharp
#: lattice land in EMPTY bins and `site_midpoint_contrast` returns ``inf`` --
#: which would pass gate 3 for free and make it no gate at all.  `structure.py`
#: computes its contrast on the blurred `rho_cell` for the same reason.
DENSITY_BINS = 72
DENSITY_KERNEL_L_B = 0.40

QUALITY_MARK = "QUICK / not publication quality"
QUALITY_STAMP = "QUICK\nnot publication quality"

#: Recorded, and checked, because they belong to no source file and yet they move
#: a number at the 12th digit.
_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ==========================================================================
# the random determinant
# ==========================================================================
def haar_random_v(nk, n_bands, rng):
    """A Haar-random orbital matrix, as ``v`` of shape ``(nk, n_bands-1)``.

    ``sin^2 t ~ U[0,1]`` is the CP1 Fubini-Study CDF, so this is the measure-free
    "random determinant" -- there is no width, no scale and no projection here.
    ``t = 0`` (the pure lowest Landau level) is a measure-zero endpoint, which is
    what separates this family from a Gaussian seed.
    """
    if n_bands != 2:
        raise ValueError(
            f"the Haar draw here is the CP1 one, which is the two-level case: "
            f"n_bands must be 2 (n_max = 1), got {n_bands}.  A product of "
            f"independent per-column draws at higher n_max is a different "
            f"measure and would need to say so rather than inherit this name.")
    u = rng.random((int(nk), 1))
    t = np.arcsin(np.sqrt(u))
    phi = rng.random((int(nk), 1)) * 2.0 * math.pi
    return t * np.exp(1j * phi)


def seed_occupation(v):
    """``nbar`` of the START: mean ``sin^2 t_k``, i.e. the excited-band weight.

    Reported per trial so "was the start artificially mixing-heavy?" is a number
    rather than an assumption.
    """
    C = lr.c_row(np.asarray(v, complex))
    return float((np.abs(C[:, 1:]) ** 2).sum(axis=1).mean())


# ==========================================================================
# geometry-derived measurement sets
# ==========================================================================
def analysis_torus(n_electrons):
    """The torus ``RunState.structure_factor`` builds, so the two cannot differ."""
    return st.Torus(2.0 * math.pi * float(n_electrons),
                    n_electrons=int(n_electrons),
                    n_cells_per_side=int(round(math.sqrt(int(n_electrons)))))


def bragg_families(torus):
    """The three first-shell pairs ``{+-g1}``, ``{+-g2}``, ``{+-(g1+g2)}``.

    All six have the same ``|q|`` -- that is what makes them one shell -- so a C6
    state has equal ``S`` on all three, and their spread is evidence for the C6
    gate rather than a gate of its own.
    """
    g1 = np.asarray(torus.g1, float)
    g2 = np.asarray(torus.g2, float)
    return [(g1, -g1), (g2, -g2), (g1 + g2, -(g1 + g2))]


def bragg_vectors(torus):
    """The six first-shell vectors, from the library's own search."""
    return np.asarray(torus.wc_shell_vectors(), float)


def background_vectors(torus):
    """Supercell momenta in the ``0.75-1.25 |g1|`` annulus, minus the peaks.

    The transcription ``tests/test_measure_against_legacy.py`` pins: 120 momenta
    at N = 36, which is what makes ``bragg_ratio``'s median comparable with the
    baseline's number.
    """
    G1 = np.asarray(torus.G1, float)
    G2 = np.asarray(torus.G2, float)
    mag_G1 = float(np.linalg.norm(G1))
    mag_g1 = float(np.linalg.norm(torus.g1))
    peaks = bragg_vectors(torus)
    mb = int(np.ceil(1.3 * mag_g1 / mag_G1))
    allq = np.array([m * G1 + n * G2
                     for m in range(-mb, mb + 1) for n in range(-mb, mb + 1)])
    n = np.linalg.norm(allq, axis=1)
    sel = (n > 0.75 * mag_g1) & (n < 1.25 * mag_g1)
    sel &= np.array([np.min(np.linalg.norm(peaks - q, axis=1)) > 1e-6
                     for q in allq])
    return allq[sel]


# ==========================================================================
# the C6 gate
# ==========================================================================
def rot(deg):
    """A plane rotation, as the C6 test applies it (``pts @ rot(deg)``)."""
    t = math.radians(float(deg))
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def c6_context(vmc, torus, nsp=C6_NSP, seed=C6_SEED):
    """Sample points and basis orbitals for the C6 residual -- INDEPENDENT of ``v``.

    The rotated and unrotated samples are both basis functions, so they are built
    once per ``(N, r_s, n_max)`` and reused by every trial; only ``C = c_row(v)``
    changes between trials.
    """
    n_band = vmc.n_bands
    basis = vmc.basis(n_band)
    rng = np.random.default_rng(int(seed))
    pts = rng.random((int(nsp), 2)) @ np.asarray(torus.sc, float).T
    V = np.array([basis.orbitals(p)[:, :n_band] for p in pts])
    W = {th: np.array([basis.orbitals(p)[:, :n_band]
                       for p in (pts @ rot(th))]) for th in C6_ROTS}
    return {"V": V, "W": W, "w": float(torus.area) / float(nsp), "nsp": int(nsp)}


def space_residual(Psi, PsiR, w):
    """How far the rotated occupied space is from the unrotated one.

    The determinant is invariant under the rotation iff ``PsiR = Psi @ cf`` for
    some matrix ``cf``; this is the least-squares residual of that fit, which is
    invariant under a change of orbital gauge on either side -- the property the
    withdrawn ``T_k(R^-1 r) == T_k(r)`` test lacked.

    Transcribed from ``tests/test_projection_convention.py``, where the corner
    values are pinned: 3.0e-14 for the C6-covariant seed, 7.9e-01 for the
    defective one.
    """
    S = w * (Psi.conj().T @ Psi)
    cf = np.linalg.solve(S, w * (Psi.conj().T @ PsiR))
    return float(np.linalg.norm(PsiR - Psi @ cf) / np.linalg.norm(PsiR))


def c6_residuals(v, ctx):
    """The C6 residual at each rotation, keyed by degree."""
    C = lr.c_row(np.asarray(v, complex))
    Psi = np.einsum("ikn,kn->ik", ctx["V"], C)
    return {th: space_residual(Psi, np.einsum("ikn,kn->ik", ctx["W"][th], C),
                               ctx["w"])
            for th in C6_ROTS}


# ==========================================================================
# diagnostics on a state
# ==========================================================================
def density_field(snaps, torus, nbins=DENSITY_BINS, kernel_l_b=DENSITY_KERNEL_L_B):
    """The smoothed real-space density, in units of the cell mean.

    ``density_grid`` -> divide by the cell mean -> periodic Gaussian blur, which
    is ``examples/figure_construction/structure.py``'s ``density_field`` with the
    drawing-only parts removed.  The blur width is expressed in magnetic lengths
    and converted to bins here, so the smoothing is a stated physical width and
    not an artefact of the grid resolution.
    """
    n_mean = float(torus.ne) / float(torus.area)
    cell = st.density_grid(snaps, torus, nbins=int(nbins)) / n_mean
    bin_l_b = float(np.linalg.norm(torus.L1)) / float(nbins)
    return st.periodic_gaussian_blur(cell, float(kernel_l_b) / bin_l_b)


def diagnose(state, torus, bragg_q, bg_q, families, ctx, ne):
    """Every published number for one state, as a flat dict.

    ``S(+G) == S(-G)`` is an exact identity (``rho_{-q} = conj(rho_q)``), so the
    ``+-G`` residual is a code-correctness control that must read ``0.0`` and is
    not a measurement.
    """
    snaps = state.snaps
    if snaps is None:
        raise RuntimeError("the state carries no snapshots; nothing to diagnose")
    S = np.real(st.structure_factor(snaps, bragg_q, ne))
    peak, bg, ratio = st.bragg_ratio(snaps, bragg_q, bg_q, int(ne))

    # One pass over the three families carries both the reported spread and the
    # +-G control.  The control is exact by construction (rho_{-q} = conj(rho_q)),
    # so a non-zero value is a defect in the estimator, not physics -- it is
    # measured rather than asserted so a broken estimator cannot pass untested.
    fam, pm = [], 0.0
    for pair in families:
        S_pair = np.real(st.structure_factor(snaps, np.array(pair), int(ne)))
        fam.append(float(S_pair.mean()))
        pm = max(pm, abs(float(S_pair[0]) - float(S_pair[1])))
    fam_mean = float(np.mean(fam))
    anisotropy = (float(max(fam) - min(fam)) / fam_mean) if fam_mean > 1e-12 else 0.0

    H = density_field(snaps, torus)
    contrast = st.site_midpoint_contrast(H, torus, DENSITY_BINS)

    res = c6_residuals(state.v, ctx)
    occ = state.ll_occupation()
    return {
        "bragg_peak": float(peak), "bragg_bg": float(bg), "bragg_ratio": float(ratio),
        "S_G1": fam[0], "S_G2": fam[1], "S_G3": fam[2],
        "anisotropy_A": float(anisotropy),
        "c6_residual": float(max(res.values())),
        "c6_by_rotation": {f"{th:.0f}": float(r) for th, r in res.items()},
        "site_midpoint_contrast": float(contrast),
        "pm_G_residual": float(pm),
        "ll_P": [float(x) for x in occ["P"]],
        "ll_nbar": float(occ["nbar"]),
        "_S_bragg": [float(x) for x in S],
    }


# ==========================================================================
# the criterion
# ==========================================================================
def wc_verdict(rec, liquid, c6_gate=C6_GATE, bragg_gate=BRAGG_GATE,
               contrast_gate=CONTRAST_GATE):
    """The three gates, pre-registered.  Returns the gate table and the verdict.

    A ratio rather than an absolute Bragg level, because the absolute scale is a
    property of the budget; the ``liquid`` is measured in the same run at the
    same ``r_s`` for exactly that reason.  Every component is returned whether or
    not it passes, so the constants can be audited rather than trusted.
    """
    gates = {
        "bragg": (float(rec["bragg_ratio"])
                  >= bragg_gate * float(liquid["bragg_ratio"])),
        "c6": float(rec["c6_residual"]) <= float(c6_gate),
        "contrast": float(rec["site_midpoint_contrast"]) >= float(contrast_gate),
    }
    return {"gates": gates, "wc_formed": bool(all(gates.values()))}


def overall_verdict(trials, reference, budget="quick", c6_gate=C6_GATE):
    """``PASSED`` / ``FAILED`` / ``NO VERDICT`` for the whole run.

    ``NO VERDICT`` is the escape hatch and it is checked FIRST: if the
    Gaussian-seed reference -- a point known to be a Wigner crystal -- cannot
    clear the C6 gate at this budget, then the gate is not resolvable here and
    ``0/4`` would be an artefact of the budget rather than a statement about
    random starts.

    ``budget`` is quoted in the ``FAILED`` reason, because "the optimiser may not
    have had time" is only meaningful against the protocol that was actually run:
    a smoke-budget failure is a much weaker statement than a reproduction-budget
    one, and the message must not blur them.
    """
    if reference is not None and float(reference["c6_residual"]) > float(c6_gate):
        return ("NO VERDICT",
                f"the Gaussian-seed reference reads c6_residual "
                f"{reference['c6_residual']:.3e} > C6_GATE {c6_gate:g}: this "
                f"budget cannot resolve the C6 gate, so nothing is concluded")
    n = sum(1 for t in trials if t["wc_formed"])
    if n:
        return "PASSED", f"{n}/{len(trials)} random starts formed a triangular WC"
    return ("FAILED",
            f"0/{len(trials)} random starts formed a triangular WC at budget "
            f"{budget}.  This is NOT evidence the basin is unreachable -- it is "
            f"equally consistent with the optimiser not having had time at "
            f"{budget}; see the module docstring.")


def choose_root(trials):
    """Lowest ``E_final`` among the trials that formed a WC, or ``None``.

    Deliberately does NOT fall back to the best-energy trial when nothing passes:
    an energetic winner that is not a WC is a different object, and returning it
    under the name "root" is how a criterion gets quietly relaxed.
    """
    passing = [t for t in trials if t.get("wc_formed")]
    if not passing:
        return None
    return min(passing, key=lambda t: float(t["E_final"]))["trial"]


# ==========================================================================
# running
# ==========================================================================
def measure_now(vmc, th, bud, label):
    """The production measurement, on the walk ``_run_from`` will also use.

    Same start, same seed, same estimator as the final measurement, so
    ``E_init -> E_final`` isolates the optimization rather than a changed walk.
    """
    proto = bud.protocol
    rec = measure_decomposed(vmc.maker()(th), vmc.crystal_R0(),
                             proto["meas_sweeps"], proto["meas_equil"],
                             bud.crystal_measure["sigma"],
                             bud.crystal_measure["seed"], label=label)
    rec.pop("snaps", None)
    return rec


def run_trial(vmc, bud, seed, torus, bragg_q, bg_q, families, ctx, trial, verbose=True):
    """One random start: draw, optimise, diagnose."""
    t0 = time.time()
    rng = np.random.default_rng(int(seed))
    v0 = haar_random_v(vmc.lat.nk, vmc.n_bands, rng)
    c0 = sr.jastrow_vector(vmc.kappa, NJ)
    th0 = theta(c0, v0)
    expect = NJ + 2 * (vmc.n_bands - 1) * vmc.lat.nk
    if len(th0) != expect:
        raise RuntimeError(f"theta has {len(th0)} entries, expected {expect}")

    init = measure_now(vmc, th0, bud, f"trial{trial}-init")
    cfg = resolve(phase="crystal", rs=vmc.rs, N=vmc.N, nmax=vmc.nmax,
                  init_id=0, budget=bud.name, rng_seed=int(seed))
    # The warm-start primitive `VMC.nest` itself uses.  R0 is the SAME
    # `crystal_R0()` for every trial: the experimental variable is `v` alone.
    res = vmc._run_from(cfg, th0, vmc.crystal_R0(), verbose=False)

    rec = {"kind": "trial", "trial": int(trial), "seed": int(seed),
           "rs": float(vmc.rs), "nmax": int(vmc.nmax), "n_bands": int(vmc.n_bands),
           "nbar_init": seed_occupation(v0),
           "v_abs_mean": float(np.abs(v0).mean()),
           "v_abs_max": float(np.abs(v0).max()),
           "E_init": float(init["E"]), "E_init_err": float(init["E_err"]),
           "E_final": float(res.energy_per_particle),
           "E_final_err": float(res.error),
           "dE": float(res.energy_per_particle - init["E"]),
           "acceptance": float(res.acceptance),
           "theta_digest": state_digest(res.state.c, res.state.v),
           "seconds": float(time.time() - t0)}
    z_den = math.hypot(rec["E_final_err"], rec["E_init_err"])
    rec["z"] = float(rec["dE"] / z_den) if z_den > 0 else 0.0
    rec.update(diagnose(res.state, torus, bragg_q, bg_q, families, ctx, vmc.N))
    rec["_state"] = res.state
    if verbose:
        print("  trial %d  seed %-10d  nbar0=%.3f  E %+.6f -> %+.6f  "
              "bragg=%7.3f  c6=%.2e  ctr=%5.2f  [%.0fs]"
              % (trial, seed, rec["nbar_init"], rec["E_init"], rec["E_final"],
                 rec["bragg_ratio"], rec["c6_residual"],
                 rec["site_midpoint_contrast"], rec["seconds"]), flush=True)
    return rec


def run_references(vmc_liquid, vmc_crystal, bud, torus, bragg_q, bg_q, families,
                   ctx, verbose=True):
    """The two in-run references: the disordered null, and a known WC."""
    liq = vmc_liquid.run(init_id=0, budget=bud.name, verbose=False)
    liq_rec = {"kind": "liquid", "trial": -1, "seed": None,
               "rs": float(vmc_liquid.rs), "nmax": int(vmc_liquid.nmax),
               "n_bands": int(vmc_liquid.n_bands),
               "E_final": float(liq.energy_per_particle),
               "E_final_err": float(liq.error),
               "acceptance": float(liq.acceptance),
               "theta_digest": state_digest(liq.state.c, liq.state.v),
               "seconds": None}
    liq_rec.update(diagnose(liq.state, torus, bragg_q, bg_q, families, ctx,
                            vmc_liquid.N))

    cry = vmc_crystal.run(init_id=0, budget=bud.name, verbose=False)
    ref_rec = {"kind": "reference", "trial": -2, "seed": None,
               "rs": float(vmc_crystal.rs), "nmax": int(vmc_crystal.nmax),
               "n_bands": int(vmc_crystal.n_bands),
               "E_final": float(cry.energy_per_particle),
               "E_final_err": float(cry.error),
               "acceptance": float(cry.acceptance),
               "theta_digest": state_digest(cry.state.c, cry.state.v),
               "seconds": None}
    ref_rec.update(diagnose(cry.state, torus, bragg_q, bg_q, families, ctx,
                            vmc_crystal.N))
    if verbose:
        print("  liquid      E %+.6f  bragg=%7.3f  c6=%.2e  ctr=%5.2f"
              % (liq_rec["E_final"], liq_rec["bragg_ratio"],
                 liq_rec["c6_residual"], liq_rec["site_midpoint_contrast"]),
              flush=True)
        print("  reference   E %+.6f  bragg=%7.3f  c6=%.2e  ctr=%5.2f"
              % (ref_rec["E_final"], ref_rec["bragg_ratio"],
                 ref_rec["c6_residual"], ref_rec["site_midpoint_contrast"]),
              flush=True)
    return liq_rec, ref_rec


# ==========================================================================
# bookkeeping
# ==========================================================================
def state_digest(c, v):
    """sha256 over the raw parameter bytes.  Exact, never rounded: a change too
    small to matter physically must still move this."""
    h = hashlib.sha256()
    for arr in (np.asarray(c, float), np.asarray(v, complex)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def request_slug(N, ansatz, budget, rs, nmax):
    """The directory name for a request: everything that changes the numbers."""
    rs_s = f"{float(rs):.0f}" if abs(float(rs) - round(float(rs))) < 1e-9 \
        else f"{float(rs):.6g}"
    return f"N{int(N)}__{ansatz}__{budget}__rs{rs_s}__nmax{int(nmax)}"


def blas_env():
    """The BLAS thread configuration, as the RUN will see it."""
    env = {v: os.environ.get(v) for v in _BLAS_VARS}
    env["numpy"] = np.__version__
    return env


def _threading_is_pinned(env=None):
    """True when every BLAS thread variable is explicitly set.

    ``OMP_NUM_THREADS`` alone is not enough: OpenBLAS and MKL read their own
    variables first and ignore OMP when they are present.
    """
    env = blas_env() if env is None else env
    return all(env.get(v) for v in _BLAS_VARS[:3])


def _rel(path):
    """Relative to the repo when it can be, absolute when the drive differs.

    ``os.path.relpath`` raises across Windows drives, so a recipe whose output
    is relocatable would otherwise die of where its input happened to live.
    """
    try:
        return os.path.relpath(path, CLEAN)
    except ValueError:
        return path


def figure_stamp(budget):
    """The mark a non-production figure carries, or None for the real one.

    ONE definition, read by both the drawing and the metadata.  The comparison is
    against ``reproduction`` rather than for ``quick`` so an unrecognised budget
    stamps: over-marking a smoke test is untidy, under-marking one is the failure
    this exists to prevent.
    """
    return None if str(budget) == "reproduction" else QUALITY_STAMP


def banner(args, bud, vmc, torus, n_trials):
    pin = _threading_is_pinned()
    lines = [
        "=" * 78,
        f"crystal_seed_search -- Haar-random crystal seed at large r_s "
        f"(wigner_vmc {__version__})",
        "=" * 78,
        f"  budget           {bud.name}   "
        f"({'clean-package (NOT converged, NOT comparable to published numbers)' if bud.name == 'quick' else bud.provenance})",
        f"  N                {vmc.N}   nk = {vmc.lat.nk}",
        f"  r_s              {vmc.rs:g}   kappa = {vmc.kappa:.8f}",
        f"  n_max            {vmc.nmax}   n_bands = {vmc.n_bands}",
        f"  len(theta)       {NJ + 2 * (vmc.n_bands - 1) * vmc.lat.nk}",
        f"  random measure   Haar on the CP1 orbital manifold (no scale knob)",
        f"  trials           {n_trials}",
        f"  seed base        {args.seed_base}",
        f"  references       liquid (disordered null) + Gaussian seed (known WC)",
        f"  BLAS threads     {'pinned' if pin else 'NOT pinned'}: "
        + ", ".join(f"{v}={os.environ.get(v)}" for v in _BLAS_VARS[:3])
        + f", numpy={np.__version__}",
        f"  gate             bragg >= {BRAGG_GATE:g}x liquid;  "
        f"c6_residual <= {C6_GATE:g};  contrast >= {CONTRAST_GATE:g}",
        f"  results          {_rel(RESULTS)}",
        f"  figures          {_rel(FIGDIR)}",
        "=" * 78,
        f"  root r_s note    {DEFAULT_RS_NOTE}",
        "=" * 78,
    ]
    print("\n".join(lines), flush=True)
    if not pin:
        print("  NOTE: BLAS threading is not pinned; numbers are reproducible "
              "only to ~1e-11 between runs.", flush=True)


# ==========================================================================
# the QC figure
# ==========================================================================
def write_figure(fig_dir, trials, liquid, reference, args):
    """``E_final``, ``bragg_ratio`` and ``c6_residual`` per trial, with the two
    reference levels drawn as nulls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "crystal_seed_search.png")
    stamp = figure_stamp(args.budget)

    xs = [t["trial"] for t in trials]
    panels = [("E_final", r"$E/N$", None),
              ("bragg_ratio", "Bragg ratio", liquid["bragg_ratio"]),
              ("c6_residual", r"$C_6$ residual", C6_GATE)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (key, label, level) in zip(axes, panels):
        ys = [t[key] for t in trials]
        colors = ["tab:green" if t.get("wc_formed") else "tab:red" for t in trials]
        ax.scatter(xs, ys, c=colors, s=48, zorder=3)
        for x, y, t in zip(xs, ys, trials):
            ax.annotate(f"{t['trial']}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
        if level is not None:
            ax.axhline(level, color="0.35", ls="--", lw=1.1, zorder=1)
        if key == "c6_residual":
            ax.axhline(reference["c6_residual"], color="tab:blue", ls=":", lw=1.3,
                       zorder=1, label="Gaussian-seed reference")
            ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("trial")
        ax.set_ylabel(label)
        ax.set_title({"E_final": "energy after SR",
                      "bragg_ratio": "Bragg ratio (dashed = liquid)",
                      "c6_residual": "C6 residual (dashed = gate, dotted = ref)"}[key],
                     fontsize=10)
        ax.set_xticks(xs)
    fig.suptitle(f"Haar-random crystal seeds at r_s = {trials[0]['rs']:g}, "
                 f"n_max = {trials[0]['nmax']}, budget {args.budget}", fontsize=11)
    if stamp:
        fig.text(0.5, 0.015, stamp.replace("\n", " -- "), ha="center",
                 fontsize=11, color="crimson", alpha=0.85)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ==========================================================================
# driver
# ==========================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--budget", default=DEFAULT_BUDGET,
                   help="protocol name from configs/ (default: %(default)s)")
    p.add_argument("--rs", type=float, default=DEFAULT_RS,
                   help="the large-r_s seeding coupling (default: %(default)s)")
    p.add_argument("--nmax", type=int, default=DEFAULT_NMAX,
                   help="Landau-level cutoff; the Haar CP1 draw needs 1 "
                        "(default: %(default)s)")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                   help="number of independent random starts (default: %(default)s)")
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                   help="first trial seed; trial i uses seed_base + i")
    p.add_argument("--n-electrons", type=int, default=N_ELECTRONS,
                   help="electron count (default: %(default)s)")
    p.add_argument("--nsp", type=int, default=C6_NSP,
                   help="sample points for the C6 residual (default: %(default)s)")
    p.add_argument("--no-figure", action="store_true",
                   help="write the table and metadata only")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if int(args.nmax) != 1:
        raise SystemExit(
            f"--nmax {args.nmax} is refused: the Haar draw in this recipe is the "
            f"CP1 one, which is the two-level (n_max = 1) case.  A product of "
            f"independent per-column draws at higher n_max is a different measure "
            f"and must be justified as one rather than inherit this name.")
    if int(args.trials) < 1:
        raise SystemExit(f"--trials must be >= 1, got {args.trials}")

    bud = load_budget(args.budget)
    vmc = VMC(N=int(args.n_electrons), rs=float(args.rs), phase="crystal",
              nmax=int(args.nmax))
    vmc_liq = VMC(N=int(args.n_electrons), rs=float(args.rs), phase="liquid",
                  nmax=int(args.nmax))
    torus = analysis_torus(vmc.N)
    bragg_q = bragg_vectors(torus)
    bg_q = background_vectors(torus)
    families = bragg_families(torus)
    ctx = c6_context(vmc, torus, nsp=int(args.nsp))

    banner(args, bud, vmc, torus, int(args.trials))
    print(f"  bragg set        {len(bragg_q)} first-shell vectors")
    print(f"  background set   {len(bg_q)} momenta\n", flush=True)

    slug = request_slug(vmc.N, vmc.ansatz, bud.name, vmc.rs, vmc.nmax)
    res_dir = os.path.join(RESULTS, slug)
    fig_dir = os.path.join(FIGDIR, slug)
    os.makedirs(res_dir, exist_ok=True)

    t0 = time.time()
    liquid, reference = run_references(vmc_liq, vmc, bud, torus, bragg_q, bg_q,
                                       families, ctx)
    trials = []
    for i in range(int(args.trials)):
        rec = run_trial(vmc, bud, int(args.seed_base) + i, torus, bragg_q, bg_q,
                        families, ctx, i)
        rec.update(wc_verdict(rec, liquid))
        trials.append(rec)

    verdict, reason = overall_verdict(trials, reference, bud.name)
    root = choose_root(trials)

    # --- the table the brief asks for -------------------------------------
    print("\n" + "=" * 78)
    print("PER-TRIAL")
    print("=" * 78)
    hdr = ("  trial seed       nbar0   E_init     E_final    dE      z    "
           "bragg    c6      ctr   WC")
    print(hdr)
    for t in trials:
        print("  %5d %-10d %.3f  %+.6f %+.6f %+.6f %+5.2f %8.3f %.2e %5.2f  %s"
              % (t["trial"], t["seed"], t["nbar_init"], t["E_init"], t["E_final"],
                 t["dE"], t["z"], t["bragg_ratio"], t["c6_residual"],
                 t["site_midpoint_contrast"], "YES" if t["wc_formed"] else "no"))
    print("\n  references")
    for r in (liquid, reference):
        print("  %-10s %-10s %s  %+.6f %s %8.3f %.2e %5.2f"
              % (r["kind"], "", "", r["E_final"], " " * 21, r["bragg_ratio"],
                 r["c6_residual"], r["site_midpoint_contrast"]))

    print("\n  three first-shell families and their spread (REPORTED, not gated)")
    for t in trials:
        print("    trial %d  S(G1)=%.3f S(G2)=%.3f S(G3)=%.3f  A=%.4f  "
              "+-G residual=%.3e"
              % (t["trial"], t["S_G1"], t["S_G2"], t["S_G3"],
                 t["anisotropy_A"], t["pm_G_residual"]))

    print("\n" + "=" * 78)
    print(f"VERDICT  {verdict}")
    print(f"  {reason}")
    if root is None:
        print("  NO ROOT -- no trial formed a WC, and the best-energy trial is "
              "NOT substituted.")
    else:
        win = next(t for t in trials if t["trial"] == root)
        print(f"  root     trial {root} (seed {win['seed']}), "
              f"E/N = {win['E_final']:+.9f} +- {win['E_final_err']:.9f}, "
              f"nbar0 = {win['nbar_init']:.3f} -> {win['ll_nbar']:.3f}")
        states = os.path.join(res_dir, "states")
        os.makedirs(states, exist_ok=True)
        import pickle
        st_path = os.path.join(states, f"root__trial{root}.pkl")
        win["_state"].snaps = None
        with open(st_path, "wb") as fh:
            pickle.dump(win["_state"], fh)
        print(f"  state    {_rel(st_path)}")
    print(f"  elapsed  {time.time() - t0:.0f}s")
    print("=" * 78, flush=True)

    # --- persist ----------------------------------------------------------
    for t in trials:
        t.pop("_state", None)
        t.pop("_S_bragg", None)
    liquid.pop("_S_bragg", None)
    reference.pop("_S_bragg", None)
    payload = {"trials": trials, "liquid": liquid, "reference": reference,
               "verdict": verdict, "verdict_reason": reason, "root_trial": root}
    with open(os.path.join(res_dir, "runs.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    meta = {
        "run_metadata_schema": "phase_competition_2/crystal_seed_search/1",
        "recipe": "examples/phase_competition_2/crystal_seed_search.py",
        "wigner_vmc_version": __version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": "1A -- pipeline validation only; no scientific claim if no WC "
                 "forms, because a random determinant may simply not have had "
                 f"time at {bud.name} (SR {bud.protocol['sr_steps']}x"
                 f"{bud.protocol['sr_sweeps']})",
        "budget": bud.name,
        "quality": QUALITY_MARK if bud.name != "reproduction" else bud.name,
        "figure_stamp": figure_stamp(bud.name),
        "blas_env": blas_env(),
        "blas_threading_pinned": bool(_threading_is_pinned()),
        "N": vmc.N, "rs": float(vmc.rs), "kappa": float(vmc.kappa),
        "nmax": int(vmc.nmax), "n_bands": int(vmc.n_bands),
        "len_theta": NJ + 2 * (vmc.n_bands - 1) * vmc.lat.nk,
        "ansatz": str(vmc.ansatz),
        "root_rs_note": DEFAULT_RS_NOTE,
        "random_measure": {
            "name": "haar_cp1",
            "statement": "sin^2(t) ~ U[0,1] (the CP1 Fubini-Study CDF), "
                         "phi ~ U[0, 2pi), i.i.d. over the nk momenta; "
                         "E[sin^2 t] = 1/2",
            "no_scale_knob": True,
            "construction": "c_row(v), the same map the optimiser uses",
        },
        "seed_base": int(args.seed_base), "trials": int(args.trials),
        "criterion": {
            "bragg_gate": BRAGG_GATE, "c6_gate": C6_GATE,
            "contrast_gate": CONTRAST_GATE,
            "bragg_definition": "bragg_ratio(trial) >= bragg_gate * "
                                "bragg_ratio(liquid measured in the SAME run)",
            "c6_definition": "max over 60/120/180/240/300 deg of the normalized "
                             "least-squares residual of the rotated occupied "
                             "space (test_projection_convention.py); bounded in "
                             "[0,1]; 3.0e-14 for a C6-covariant seed, 7.9e-01 "
                             "for a C6-broken one",
            "anisotropy_is_reported_not_gated": (
                "A = (max-min)/mean over the three first-shell families is a "
                "ratio of noisy S values with a small denominator in the liquid, "
                "so it is reported and never thresholded"),
            "contrast_definition": (
                f"site density over nearest-neighbour-midpoint density on the "
                f"density smoothed at {DENSITY_KERNEL_L_B} l_B over "
                f"{DENSITY_BINS} bins; 1 for anything uniform, and inf on the "
                f"raw histogram"),
            "escape_hatch": "if the Gaussian-seed reference fails the C6 gate, "
                            "the run reports NO VERDICT rather than 0/N",
        },
        "root_selection": "lowest E_final among trials with wc_formed true; "
                          "None (and NOT the best-energy trial) when none pass",
        "verdict": verdict, "verdict_reason": reason, "root_trial": root,
        "bragg_set_size": int(len(bragg_q)),
        "background_set_size": int(len(bg_q)),
        "result_dir": _rel(res_dir),
        "not_in_scope": ["r_s continuation", "n_max nesting",
                         "phase competition", "reproduction/full"],
    }
    with open(os.path.join(res_dir, "run_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    print(f"\n  wrote {_rel(os.path.join(res_dir, 'runs.json'))}")
    print(f"  wrote {_rel(os.path.join(res_dir, 'run_metadata.json'))}", flush=True)

    if not args.no_figure:
        path = write_figure(fig_dir, trials, liquid, reference, args)
        print(f"  wrote {_rel(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
