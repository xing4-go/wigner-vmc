#!/usr/bin/env python
"""Why does a C6-correct Gaussian seed leave the C6 sector under joint SR?

The question, and why it is the one to ask
------------------------------------------
``crystal_seed_search.py`` (Stage 1A) reported ``NO VERDICT`` at ``quick``
because its Gaussian-seed reference read ``c6_residual = 0.209`` against
``C6_GATE = 0.1``.  The same construction reads **0.0866** at ``smoke``.  So the
seed starts C6-covariant -- the pinned corner of
``tests/test_projection_convention.py`` is ~3e-14 -- and more SR moved it
*further* from C6-invariance.  Every read of the random-start campaign goes
through this gate, so the gate's drift has to be explained first.

Three layers, and the plan is built on them
-------------------------------------------
1. **Is the seed strictly correct at THIS operating point?**  The pinned corner
   is ``r_s = 75, n_max = 2``; nothing pins ``r_s = 90, n_max = 1``, where the
   orbital manifold has one excited band instead of two.  Measured here, in-run,
   with a positive control (the corrected overlap), a second independent context
   (different sample points and seed), a gauge-invariance check, and a negative
   control (the conjugated overlap).  If the seed is not C6 here, "SR broke it"
   is the wrong framing and the verdict says so.

2. **Does the first update break it, and what does the trajectory look like?**
   Per step: ``E`` in BOTH units, the C6 residual with its full five-rotation
   pattern, ``S1/S2/S3`` and their split, ``|f|``, ``||dv||``, the registry
   phases, the density contrast, the Bragg ratio and the conditioning record.
   ``S``, the registry and the contrast need their own Metropolis walk, which
   uses its OWN rng so it cannot perturb the SR stream.

3. **Is the breaking component the algorithm or the MC noise?**  The decisive
   layer, and it has a free first answer: ``c6_residual`` is the SINE OF THE
   PRINCIPAL ANGLE between the occupied space and its rotation
   (``tests/test_projection_convention.py:277``), so it is *linear* in the
   breaking amplitude and ``R_C6(1)`` **is** the relative first-order
   symmetry-breaking amplitude of the first SR update -- directly, with no
   square root.  (An earlier draft of the plan called it quadratic and drew a
   square root from it; corrected against the source before any code was
   written.)  The sub-experiments then separate the candidates:
   ``--rng-seeds`` (3a: same everything but the SR sampling stream -- do the
   runs break the same way?), ``--nsweep-ladder`` (3b: does ``R_C6(1)`` fall as
   ``n^-1/2``?), ``--pilot-scale`` (3c: is the orbital update regulariser-set,
   i.e. a damped gradient step?) and ``--pinned-control`` (3d: does breaking C6
   actually buy energy?).

Why the first step is the fair step
-----------------------------------
``tau_i = tau / (1 + xi*i/steps)`` is the ONLY place ``steps`` enters
(``src/wigner_vmc/vmc/sr.py:311``), and at ``i = 0`` it is ``tau`` for every
``steps``.  So ``R_C6(1)`` is schedule-free and comparable across budgets and
step counts, while ``R_C6(t > 1)`` is not.  That is what makes the cheap Layer-3
experiments legitimate rather than approximate.

What it does NOT claim
----------------------
``c6_residual`` is **not an order parameter**: ``v = 0`` (the filled lowest
Landau level) is invariant under every rotation, so the liquid reads ~3e-14 and
a *low* C6 is also the liquid.  The verdict is therefore two-staged -- first
"is it still a crystal?" (Bragg ratio and real-space contrast on the closing
row), and only then the shape of the C6 trajectory.  And ``E`` is never the
decisive observable: at the cheap density there are
``len(range(equil, nsweep, snapshot_every))`` snapshots per step, and
consecutive rows share the walker (``sr.py:293``), so ``E_err`` is a LOWER bound
and resolving ``dE ~ 0.01`` per particle would need ``nsweep ~ 1000``.

What this recipe does not read
------------------------------
It recomputes everything and reads no stored result of any kind: not the
defective LL-rotation crystal records (``legacy_llrot``), not the analysis store
derived from them (``energy_scan``), not the scan driver (``_diag/part1``), and
it does not treat the frozen ``make_notebook.py`` as a specification.

It also does not build the seed the paper does not use.  This is the sibling's
ban and it applies here with the same force: the Haar-random determinant
(``haar_random_v``) is Stage 1A's subject, and this diagnostic starts from the
Gaussian overlap on purpose -- the question is what SR does to *that* seed.
Each of these names appears here, in the prose, precisely because it must not
appear in the code -- the contract test checks the second while this paragraph
keeps the first from being vacuous.

The sibling recipe ``crystal_seed_search.py`` is imported by path for its C6
context, its geometry sets and its stamp, so the two cannot drift.  Importing
``wigner_vmc.vmc.sr`` directly is deliberate and must not be "cleaned up": the
public API reduces the SR trace to ``Optimization(E_total, acc)``
(``src/wigner_vmc/api.py:544-556``, ``_run_from`` at 886-891) and the entire
content of this diagnostic IS the per-step trace.  ``examples/phase_competition_2``
sits outside ``tests/test_api_contract.py``'s ``EXAMPLE_SCRIPTS`` -- a fixed
three-name tuple at line 64, not a directory walk -- so the layering rule does
not apply here.
"""
import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np

from wigner_vmc import VMC, __version__, load_budget, resolve, theta, theta_parts
from wigner_vmc.analysis import structure as st
from wigner_vmc.vmc import sr
from wigner_vmc.vmc.measure import measure_decomposed

#: ``NJ = 5``.  Imported rather than re-typed for the same reason the sibling
#: imports it: ``len(theta)`` and ``sr.jastrow_vector`` must not drift apart.
from wigner_vmc.api import NJ, OVERLAP_NUMX

# ==========================================================================
# the sibling recipe, loaded by path
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))              # .../wigner_vmc_clean


def _load_sibling():
    """``crystal_seed_search`` by path -- and deliberately without ``sys.path``.

    The sibling is a script, not a package module, so it is loaded the way
    ``tests/test_crystal_seed_search_contract.py`` loads it.  A ``sys.path``
    insert here would be picked up by every later import in the process, which
    is why ``sys`` is not imported into this module at all.
    """
    path = os.path.join(HERE, "crystal_seed_search.py")
    spec = importlib.util.spec_from_file_location("crystal_seed_search_shared", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


css = _load_sibling()

# --- geometry, gates and stamps: imported, never re-typed ------------------
N_ELECTRONS = css.N_ELECTRONS
DEFAULT_RS = css.DEFAULT_RS
DEFAULT_NMAX = css.DEFAULT_NMAX
DEFAULT_BUDGET = css.DEFAULT_BUDGET
C6_GATE = css.C6_GATE
BRAGG_GATE = css.BRAGG_GATE
CONTRAST_GATE = css.CONTRAST_GATE
C6_ROTS = css.C6_ROTS
C6_NSP = css.C6_NSP
C6_SEED = css.C6_SEED
DENSITY_BINS = css.DENSITY_BINS
QUALITY_MARK = css.QUALITY_MARK
analysis_torus = css.analysis_torus
bragg_families = css.bragg_families
bragg_vectors = css.bragg_vectors
background_vectors = css.background_vectors
c6_context = css.c6_context
c6_residuals = css.c6_residuals
density_field = css.density_field
seed_occupation = css.seed_occupation
state_digest = css.state_digest
blas_env = css.blas_env
figure_stamp = css.figure_stamp
request_slug = css.request_slug
_rel = css._rel

RESULTS = os.path.join(CLEAN, "results", "phase_competition_2", "c6_trajectory")
FIGDIR = os.path.join(CLEAN, "figures", "phase_competition_2", "c6_trajectory")

DEFAULT_NSWEEP_LADDER = (40, 150, 400)
DEFAULT_RNG_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_STEPS_3A = 3

#: The closing state's class.  ``NOT_A_CRYSTAL`` means C6 is uninformative about
#: symmetry, NOT that the state is disordered -- see the liquid in the docstring.
NOT_A_CRYSTAL = "NOT_A_CRYSTAL"
CRYSTAL = "CRYSTAL"
CLASS_UNKNOWN = "CLASS_UNKNOWN"

#: Labels that name the NEXT experiment rather than an answer about the five
#: candidate explanations.
NOT_A_RESULT = ("STILL_RISING", "PLATEAU_BROKEN", "REGISTRY_DRIFT")

#: The keys the S walk produces.  A row the stride skips gets all of them as
#: ``None`` -- empty, never filled with a plausible value.
#:
#: ``s_E_*`` are the walk's OWN energy, from the production record
#: (``measure_decomposed``), so it is the corrected error bar and it is per
#: particle.  It is stored beside the optimiser's ``E_*`` because the two differ
#: in kind: ``E_t`` is the SR step's own sampling, ``s_E_*`` is an independent
#: walk at the same state.  Neither is the other.
#:
#: ``pm_G_residual`` is ``max |S(+G) - S(-G)|`` over the three families.  That
#: difference is an EXACT identity (``rho_{-q} = conj(rho_q)``), so a non-zero
#: value is a defect in the estimator and not physics -- which is why it is
#: measured rather than asserted, exactly as ``diagnose`` does.
S_KEYS = ("S_G1", "S_G2", "S_G3", "S_split", "S_mean", "S_anisotropy_A",
          "bragg_peak", "bragg_bg", "bragg_ratio", "site_midpoint_contrast",
          "pm_G_residual",
          "s_E_per_particle", "s_E_err_per_particle",
          "s_E_err_naive_per_particle", "s_E_tau", "s_E_tau_window",
          "s_acc", "s_sigma", "s_n", "s_seconds",
          "reg_g1", "reg_g2", "reg_g1g2", "reg_identity",
          "reg_g1_coh", "reg_g2_coh", "reg_g1g2_coh",
          "reg_g1_abs", "reg_g2_abs", "reg_g1g2_abs",
          "reg_g1_phase_err", "reg_g2_phase_err", "reg_g1g2_phase_err")

hist_KEYS = ("E", "E_err", "acc", "sigma", "force", "tau", "cond", "eig_min",
             "eig_max")


# ==========================================================================
# small shared helpers
# ==========================================================================
def wrap_pi(x):
    """``x`` folded into ``(-pi, pi]``."""
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def snapshot_count(nsweep, equil, snapshot_every):
    """How many snapshots a walk takes, from ``sampler.sample``'s own rule.

    ``sampler.py:98`` takes a snapshot when ``sweep >= equil`` and every
    ``snapshot_every`` thereafter, so the count is ``len(range(equil, nsweep,
    snap))``.  The tempting ``(nsweep - equil) // snap + 1`` over-counts by one
    whenever the division is exact -- it reports **15** at ``smoke``'s 6/20/1,
    where the walk takes 14 -- and every error bar computed from it would
    inherit the flattery.
    """
    return len(range(int(equil), int(nsweep), int(snapshot_every)))


def int_list(s):
    """``"0, 1 ,2"`` -> ``[0, 1, 2]``."""
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def monotone(xs):
    """True when the sequence is non-decreasing or non-increasing."""
    d = np.diff(np.asarray(xs, float))
    return bool(np.all(d >= 0.0) or np.all(d <= 0.0))


def _opt(h, key):
    """A ``hist`` key that not every optimiser records.

    ``sr_optimize_jastrow`` has neither ``cond`` nor the eigenvalues, and
    inventing them for the pinned arm would make two different records look
    alike.  They are reported as ``None`` instead.
    """
    return float(h[key]) if key in h else None


# ==========================================================================
# Layer 1 -- the seed, at THIS operating point
# ==========================================================================
def seed_controls(vmc, torus, ctx, L0, nsp2, seed2):
    """The corrected seed, a second context, gauge invariance, and the defect.

    Returns the corrected ``v0`` -- what every arm starts from -- plus the four
    controls that decide whether the rest of the run is interpretable.

    The negative control is not decoration.  ``space_residual`` reads ~1e-14 for
    anything C6-covariant AND ~1e-14 for the liquid (``v = 0`` is invariant under
    every rotation), so a context that returns small numbers for the wrong reason
    would pass a gate it does not actually test.  ``ov.conj()`` is the
    historical defect: the same input with the projection's conjugation
    reversed, which ``tests/test_projection_convention.py`` pins at ~7.9e-01.
    """
    ov = sr.gaussian_overlap_seed(vmc.basis(), vmc.n_bands, vmc.lat.L1,
                                  vmc.lat.L2, vmc.lat.ov_ai, vmc.lat.ov_ac,
                                  rs=vmc.rs, numx=OVERLAP_NUMX, L0=L0)
    v_correct = sr.v_from_overlap(ov)
    v_defect = sr.v_from_overlap(ov.conj())

    def max_res(v, context):
        return float(max(c6_residuals(v, context).values()))

    base = c6_residuals(v_correct, ctx)
    rec = {"L0": float(L0), "nsp": int(ctx["nsp"]), "c6_seed": int(C6_SEED),
           "c6_correct": max(base.values()),
           "c6_defect": max_res(v_defect, ctx),
           "nbar_seed": seed_occupation(v_correct),
           "nbar_defect": seed_occupation(v_defect)}
    # A second context on different sample points and a different rng: a bug in
    # `c6_context` itself would otherwise pass vacuously in both controls.
    ctx2 = c6_context(vmc, torus, nsp=int(nsp2), seed=int(seed2))
    rec["nsp2"] = int(nsp2)
    rec["c6_seed2"] = int(seed2)
    rec["c6_correct_ctx2"] = max_res(v_correct, ctx2)
    rec["c6_defect_ctx2"] = max_res(v_defect, ctx2)
    # Gauge invariance, per rotation: rescaling the orbital columns changes
    # neither the occupied space nor the residual.  Compared ROTATION BY
    # ROTATION rather than as a difference of maxima, so a gauge error that
    # moved one rotation up and another down would still be caught.
    C = css.lr.c_row(np.asarray(v_correct, complex)) * (1.0 + 0.5j)
    Psi = np.einsum("ikn,kn->ik", ctx["V"], C)
    shifted = {th: css.space_residual(
        Psi, np.einsum("ikn,kn->ik", ctx["W"][th], C), ctx["w"]) for th in C6_ROTS}
    rec["c6_gauge_shift"] = float(max(abs(shifted[th] - base[th])
                                      for th in C6_ROTS))
    rec["c6_gauge_value"] = float(max(shifted.values()))
    return v_correct, rec


def seed_verdict(rec, gate=C6_GATE, min_separation=1e6):
    """``ok`` only when every control that can fail has been made to pass.

    Four checks, and the third is the one an earlier draft got wrong.  The
    negative control has to read **large**; ``space_residual`` is bounded in
    ``[0, 1]``, so an absolute threshold like ``10 * gate`` demands something
    impossible and would have failed a context with a 13-decade separation.
    What "the context resolves the gate" actually means is a *separation*: the
    broken seed far above the correct one, and both of them away from zero.  So
    the test is a ratio, with an absolute floor under the broken value so a
    degenerate context where both read 1e-30 cannot pass on the ratio alone.
    """
    reasons = []
    sep = (float(rec["c6_defect"]) / float(rec["c6_correct"])
           if float(rec["c6_correct"]) > 0.0 else float("inf"))
    if rec["c6_correct"] > gate:
        reasons.append(
            f"the corrected Gaussian seed reads c6 = {rec['c6_correct']:.3e} > "
            f"{gate:g} at THIS operating point, so the seed is not C6-covariant "
            f"here and SR cannot be blamed for leaving the sector")
    if rec["c6_correct_ctx2"] > gate:
        reasons.append(
            f"a second independent context (nsp={rec['nsp2']}, seed="
            f"{rec['c6_seed2']}) reads {rec['c6_correct_ctx2']:.3e} where the "
            f"first reads {rec['c6_correct']:.3e}")
    if not (sep > min_separation and float(rec["c6_defect"]) > gate / 10.0):
        reasons.append(
            f"the conjugated-overlap control reads {rec['c6_defect']:.3e} "
            f"against {rec['c6_correct']:.3e}: a separation of {sep:.3e}, which "
            f"does not establish that this context resolves the gate (needs "
            f">{min_separation:.0e} and the broken value above {gate / 10.0:g})")
    if rec["c6_gauge_shift"] > 1e-9:
        reasons.append(
            f"the residual moved under a column rescaling "
            f"({rec['c6_gauge_shift']:.3e}), so it is not gauge invariant")
    ok = not reasons
    facts = {"separation": float(sep), "min_separation": float(min_separation),
             "gauge_shift": float(rec["c6_gauge_shift"])}
    if ok:
        why = (f"the seed IS C6-covariant here: {rec['c6_correct']:.3e} at "
               f"r_s = {rec['rs']:g}, n_max = {rec['nmax']} (the pinned corner is "
               f"r_s = 75, n_max = 2, so this is measured rather than "
               f"inherited); the conjugated-overlap control reads "
               f"{rec['c6_defect']:.3e} -- a separation of {sep:.1e}; a second "
               f"context agrees to the same order; the residual is invariant "
               f"under a column rescaling to {rec['c6_gauge_shift']:.1e}")
    else:
        why = "; ".join(reasons)
    return {"ok": bool(ok), "reason": why, "facts": facts,
            "controls_that_failed": len(reasons)}


# ==========================================================================
# the per-step measurements
# ==========================================================================
def registry(snaps, torus):
    """The phase of the ordering peak at ``g1``, ``g2`` and ``g1+g2``.

    ``rho(q) = sum_j exp(i q.r_j)``, and translating the lattice by ``d``
    multiplies it by ``exp(i q.d)``.  So the three phases move by ``g1.d``,
    ``g2.d`` and ``(g1+g2).d`` -- and the third is the SUM of the first two, mod
    ``2*pi``.  That identity is what separates a lattice that has SLID from one
    that has been STRAINED, which is the whole reason this is a measurement and
    not a restatement of the C6 residual: a slide leaves a good crystal that is
    no longer invariant about the torus origin, and the residual is measured
    about the torus origin (``c6_context`` rotates the sample points).

    The prediction a slide makes is sharp, so this is a test and not a rubber
    stamp: ``R_theta V_d = V_{R_theta^{-1} d}``, so for small ``d`` the
    five-rotation residual should follow ``2|d| sin(theta/2)`` -- largest at
    180 degrees, smallest at +-60.  The stored smoke pattern is nearly flat with
    180 in the MIDDLE, which is not that shape; a slide large enough to saturate
    would flatten it.

    ``coherence = |<rho>| / <|rho|>`` lies in ``[0, 1]`` and says whether the
    angle means anything at all: a phase read where there is no peak is noise,
    and reporting it without this would be reporting an arbitrary number.

    ``phase_err`` turns that into a number the verdict can compare against, so
    "the phase moved" is a claim with a floor rather than a claim about a
    difference of two floats.  For ``n`` unit phasors about a common direction,
    ``Re m`` fluctuates with ``sqrt((1 - c^2) / 2n)`` and the angle is that over
    ``c``: ``sqrt(1 - c^2) / (c sqrt(2n))``.  At ``c = 1`` it is exactly zero,
    which is the right statement about a noiseless lattice.
    """
    R = np.asarray(snaps, float)
    g1 = np.asarray(torus.g1, float)
    g2 = np.asarray(torus.g2, float)
    n = max(int(R.shape[0]), 1)
    out = {}
    for name, q in (("g1", g1), ("g2", g2), ("g1g2", g1 + g2)):
        z = np.exp(1j * (R @ np.asarray(q, float))).sum(axis=1)
        m = z.mean()
        coh = float(abs(m) / max(float(np.abs(z).mean()), 1e-300))
        out[f"reg_{name}"] = float(np.angle(m))
        out[f"reg_{name}_coh"] = coh
        out[f"reg_{name}_abs"] = float(np.abs(z).mean())
        out[f"reg_{name}_phase_err"] = float(
            np.sqrt(max(1.0 - coh * coh, 0.0)) / (max(coh, 1e-12) * np.sqrt(2.0 * n)))
    out["reg_identity"] = wrap_pi(out["reg_g1g2"] - out["reg_g1"] - out["reg_g2"])
    return out


def s_measurements(vmc, th, torus, bragg_q, bg_q, families, sweeps, equil,
                   sigma, seed):
    """One independent walk at this state: S split, Bragg, contrast, registry.

    It goes through ``measure_decomposed`` -- the same production walk the
    sibling recipe and ``_run_from`` use -- so its ``E`` is the campaign's own
    convention (per electron, with the ``sqrt(tau)`` correction) and the
    ``snaps`` are available for the structural estimators.  Hand-rolling
    ``sample`` here would produce a second, silently different ``E``.

    Its rng is its OWN (``default_rng(seed)``, not the optimiser's) and it always
    starts from ``crystal_R0()``, so this is a pure function of ``th``.  That is
    what lets ``--s-every`` be a cost knob rather than a change to the
    experiment: ``sr_optimize_joint`` draws its walk AND its update noise from
    one ``default_rng(seed)``, so a single extra draw from that stream would move
    the trajectory and break the anchor of the contract suite.  ``crystal_R0()``
    is likewise deterministic (``default_rng(100 + int(kappa))``) and is NOT
    carried between steps: ``sr_optimize_joint`` restarts its walker from the
    previous configuration, and mirroring that here would be a different walk.
    """
    rec = measure_decomposed(vmc.maker()(np.asarray(th, float)), vmc.crystal_R0(),
                             int(sweeps), int(equil), float(sigma), int(seed),
                             label="c6traj")
    snaps = rec.pop("snaps")

    S, pm = [], 0.0
    for pair in families:
        S_pair = np.real(st.structure_factor(snaps, np.array(pair), int(vmc.N)))
        S.append(float(S_pair.mean()))
        pm = max(pm, abs(float(S_pair[0]) - float(S_pair[1])))
    peak, bg, ratio = st.bragg_ratio(snaps, bragg_q, bg_q, int(vmc.N))
    fam_mean = float(np.mean(S))
    out = {"S_G1": S[0], "S_G2": S[1], "S_G3": S[2],
           "S_split": float(max(S) - min(S)), "S_mean": fam_mean,
           "S_anisotropy_A": (float(max(S) - min(S)) / fam_mean
                              if fam_mean > 1e-12 else 0.0),
           "bragg_peak": float(peak), "bragg_bg": float(bg),
           "bragg_ratio": float(ratio),
           "site_midpoint_contrast": float(st.site_midpoint_contrast(
               density_field(snaps, torus), torus, DENSITY_BINS)),
           "pm_G_residual": float(pm),
           "s_E_per_particle": float(rec["E"]),
           "s_E_err_per_particle": float(rec["E_err"]),
           "s_E_err_naive_per_particle": float(rec["E_err_naive"]),
           "s_E_tau": float(rec["E_tau"]), "s_E_tau_window": int(rec["E_tau_window"]),
           "s_acc": float(rec["acc"]), "s_sigma": float(rec["sigma"]),
           "s_n": int(rec["n"])}
    out.update(registry(snaps, torus))
    return out


def make_meas(vmc, ctx, s_of, every=1):
    """The bundle ``build_rows`` needs, with the S stride applied.

    The CLOSING row is always measured, whatever the stride: Stage A reads it,
    and a stride must not be able to leave the class undecidable.  Row 0 is
    measured too, so the seed's own S split is present even at a long stride --
    the user's Layer-2 list wants the split from where it starts.
    """
    null = {k: None for k in S_KEYS}

    def s_of_t(t, theta_t, closing):
        if closing or int(t) == 0 or int(t) % int(every) == 0:
            return s_of(theta_t)
        return dict(null)

    return {"N": int(vmc.N), "parts": lambda th: theta_parts(
        np.asarray(th, float), vmc.n_bands, vmc.lat.nk),
        "ctx": ctx, "s_of_t": s_of_t}


def build_rows(theta0, final_theta, hist, state_at, meas):
    """Pair every ``hist`` row with the state it actually MEASURED.

    ``sr_optimize_joint`` samples at ``theta_i`` and *then* updates, appending
    afterwards -- so ``hist[i]["E"]`` is the energy of ``theta_i`` while
    ``hist[i]["theta"]`` is ``theta_{i+1}``.  Row ``t`` is therefore
    ``theta_t = theta0 if t == 0 else state_at(t - 1)``, with ``E_t =
    hist[t]["E"]``.  Pairing ``hist[t]["E"]`` with ``hist[t]["theta"]`` lags the
    state by one step and produces a trajectory that reads *exactly* like "the
    energy fell while C6 rose" -- an artefact that would be the headline if it
    were not caught here.

    The returned state has NO ``hist`` row, so it gets an explicit closing row
    (``closing=True``) carrying its own measurements.  Row ``len(hist)-1`` has an
    energy and no closing measurement; the closing row has measurements and no
    optimiser record.  Both absences are stated in the row rather than filled
    with a plausible number.
    """
    rows = [_row(t, np.array(theta0, float) if t == 0
                 else np.asarray(state_at(t - 1), float), hist[t], meas, False)
            for t in range(len(hist))]
    rows.append(_row(len(hist), np.asarray(final_theta, float), None, meas, True))
    # The block-resolved step, free from the stored thetas, and the honest
    # "is the orbital sector still moving?" -- `|f|` cannot answer it, because
    # d ln Psi/dc for the Jastrow is O(10)-O(40) while d ln Psi/dv is O(1e-3)
    # (sr_pilot_scale's docstring), so |f| -> 0 means the JASTROW converged.
    prev_c = prev_v = None
    for row in rows:
        c, v = meas["parts"](np.asarray(row["theta"], float))
        row["dc_norm"] = None if prev_c is None else float(np.linalg.norm(c - prev_c))
        row["dv_norm"] = None if prev_v is None else float(np.linalg.norm(v - prev_v))
        prev_c, prev_v = c, v
    return rows


def _row(t, theta_t, h, meas, closing):
    """One row: the state, its symmetry, and the optimiser's record for it."""
    c, v = meas["parts"](np.asarray(theta_t, float))
    res = c6_residuals(v, meas["ctx"])
    row = {"t": int(t), "closing": bool(closing),
           "theta": [float(x) for x in theta_t],
           "theta_digest": state_digest(c, v),
           "nbar": seed_occupation(v),
           "c6_residual": float(max(res.values())),
           "c6_by_rotation": {f"{th:.0f}": float(r) for th, r in res.items()},
           "c6_five_vector": [float(res[th]) for th in C6_ROTS],
           "v_abs_mean": float(np.abs(v).mean()),
           "v_abs_max": float(np.abs(v).max())}
    row.update(meas["s_of_t"](t, theta_t, closing))
    if h is None:
        row.update({k: None for k in ("E_total", "E_per_particle", "E_err_total",
                                      "E_err_per_particle", "acc", "sigma",
                                      "force", "tau", "cond", "eig_min",
                                      "eig_max")})
        return row
    row.update(E_total=float(h["E"]),
               E_per_particle=float(h["E"]) / meas["N"],
               E_err_total=float(h["E_err"]),
               E_err_per_particle=float(h["E_err"]) / meas["N"],
               acc=float(h["acc"]), sigma=float(h["sigma"]),
               force=float(h["force"]), tau=float(h["tau"]),
               cond=_opt(h, "cond"), eig_min=_opt(h, "eig_min"),
               eig_max=_opt(h, "eig_max"))
    return row


# ==========================================================================
# the arms
# ==========================================================================
def sr_kwargs(bud, seed, steps, nsweep, equil):
    """The production call's kwargs, so an arm IS the project's optimiser."""
    return dict(steps=int(steps), nsweep=int(nsweep), sigma=float(bud.sr["sigma"]),
                seed=int(seed), snapshot_every=int(bud.protocol["sr_snap"]),
                equil=int(equil), target_acc=float(bud.sr["target_acc"]))


def run_joint_arm(vmc, theta0, bud, seed, steps, nsweep, equil, scale=None):
    """``VMC._run_from``'s exact call (``api.py:886-891``), plus the trace."""
    kw = sr_kwargs(bud, seed, steps, nsweep, equil)
    if scale is not None:
        kw["scale"] = np.asarray(scale, float)
    th, hist = sr.sr_optimize_joint(vmc.maker(), np.asarray(theta0, float),
                                    vmc.crystal_R0(), **kw)
    return th, hist, (lambda i: hist[i]["theta"])


def run_pinned_arm(vmc, v0, c0, bud, seed, steps, nsweep, equil):
    """The orbitals FROZEN at the seed: ``ansatz='ll_rotation_pinned``'s scope.

    ``VMC._run``'s own pinned branch (``api.py:762-769``).  Its C6 residual
    cannot move -- it is ``C6_0`` at every step -- which is the point: the
    comparison that answers "does breaking C6 buy energy?" is this arm's
    endpoint against the joint arm's.
    """
    kw = sr_kwargs(bud, seed, steps, nsweep, equil)
    c_opt, hist = sr.sr_optimize_jastrow(vmc.maker(v=v0), np.asarray(c0, float),
                                         vmc.crystal_R0(), **kw)
    return theta(c_opt, v0), hist, (lambda i: theta(hist[i]["c"], v0))


def pilot_scale_for(vmc, theta0, bud, seed, nsweep, equil):
    """3c's counterfactual scale, from a pilot walk at this budget's shape."""
    return sr.sr_pilot_scale(vmc.maker(), np.asarray(theta0, float),
                             vmc.crystal_R0(), nsweep=int(nsweep),
                             seed=int(seed), sigma=float(bud.sr["sigma"]),
                             equil=int(equil),
                             snapshot_every=int(bud.protocol["sr_snap"]),
                             target_acc=float(bud.sr["target_acc"]))


# ==========================================================================
# the verdict
# ==========================================================================
def stage_a(closing, liquid, bragg_gate=BRAGG_GATE, contrast_gate=CONTRAST_GATE):
    """Is the closing state still a crystal?  Without this, C6 names nothing.

    The liquid is measured in the run for the same reason Stage 1A measures it
    in the run: ``bragg_ratio`` is an absolute number whose scale is a property
    of the budget, so a fixed threshold would be a statement about the protocol
    rather than about the state.
    """
    if liquid is None or closing["bragg_ratio"] is None:
        return (CLASS_UNKNOWN,
                "no liquid reference in this run (--skip-liquid) or no closing "
                "measurement, so the Bragg gate cannot be evaluated")
    gates = {"bragg": float(closing["bragg_ratio"])
             >= bragg_gate * float(liquid["bragg_ratio"]),
             "contrast": float(closing["site_midpoint_contrast"]) >= contrast_gate}
    if all(gates.values()):
        return CRYSTAL, (f"bragg {closing['bragg_ratio']:.3f} >= {bragg_gate:g}x "
                         f"{liquid['bragg_ratio']:.3f} and contrast "
                         f"{closing['site_midpoint_contrast']:.2f} >= "
                         f"{contrast_gate:g}: the crystal survived, so C6 is "
                         f"measuring symmetry")
    return (NOT_A_CRYSTAL,
            f"gates {gates}: the state degraded (bragg "
            f"{closing['bragg_ratio']:.3f} vs liquid {liquid['bragg_ratio']:.3f}, "
            f"contrast {closing['site_midpoint_contrast']:.2f}), so its C6 "
            f"residual says nothing about symmetry")


def stage_b(rows, liquid, gate=C6_GATE):
    """The shape of the C6 trajectory.  A label about the EVIDENCE, not physics."""
    c6 = [float(r["c6_residual"]) for r in rows]
    inner = c6[1:]
    fam = {"c6_0": c6[0], "c6_T": c6[-1],
           "c6_peak": max(inner) if inner else c6[0],
           "t_peak": (int(np.argmax(inner)) + 1) if inner else 0,
           "c6_max_all": max(c6),
           "turned_over": bool(inner) and c6[-1] < (max(inner) if inner else c6[0]),
           "n_steps": len(rows) - 1}
    if c6[0] > gate:
        return "SEED_NOT_C6", fam
    if max(c6) <= gate:
        return "STABLE_C6", fam
    if c6[-1] <= gate:
        return "UNDERCONVERGED", fam
    if not fam["turned_over"]:
        return "STILL_RISING", fam
    # Registry drift needs a monotone phase walk, a coherent peak to read it
    # from, an intact crystal, and the translation identity to hold (a STRAINED
    # lattice fails phi(g1+g2) = phi(g1) + phi(g2)).
    #
    # It also needs the phase to have actually MOVED.  `monotone` alone does not
    # say that: a constant series is monotone, so without the swing test a
    # lattice that never slid reads as the strongest possible slide.  The floor
    # is the measurement's own resolution -- `registry` returns the phase error
    # its coherence implies -- so this is a claim with an error bar rather than a
    # comparison of two floats, and it is a floor that a noiseless lattice drives
    # to exactly zero.
    ph = [r["reg_g1"] for r in rows if r["reg_g1"] is not None]
    coh = [r["reg_g1_coh"] for r in rows if r["reg_g1_coh"] is not None]
    perr = [r["reg_g1_phase_err"] for r in rows
            if r.get("reg_g1_phase_err") is not None]
    ident = [abs(float(r["reg_identity"])) for r in rows
             if r["reg_identity"] is not None]
    swing = (max(ph) - min(ph)) if ph else 0.0
    floor = 3.0 * max(perr) if perr else None
    fam["reg_swing"] = float(swing)
    fam["reg_phase_floor"] = None if floor is None else float(floor)
    fam["reg_moved"] = bool(floor is not None and swing > floor)
    fam["reg_monotone"] = bool(ph) and monotone(ph)
    fam["reg_min_coherence"] = min(coh) if coh else None
    fam["reg_max_identity"] = max(ident) if ident else None
    bragg_held = (liquid is None or all(
        float(r["bragg_ratio"]) >= BRAGG_GATE * float(liquid["bragg_ratio"])
        for r in rows if r["bragg_ratio"] is not None))
    fam["bragg_held"] = bool(bragg_held)
    # `is not None`, NOT `or <default>`: the identity residual is EXACTLY 0.0 for
    # a slid lattice, and `0.0 or 9.9` is 9.9 -- which would have made the one
    # case this branch exists for the one case it could not detect.
    coherence_ok = (fam["reg_min_coherence"] is not None
                    and fam["reg_min_coherence"] > 0.5)
    identity_ok = (fam["reg_max_identity"] is not None
                   and fam["reg_max_identity"] < 1.0)
    fam["coherence_ok"] = bool(coherence_ok)
    fam["identity_ok"] = bool(identity_ok)
    if (fam["reg_monotone"] and fam["reg_moved"] and coherence_ok
            and identity_ok and bragg_held):
        return "REGISTRY_DRIFT", fam
    if c6[-1] >= 0.9 * fam["c6_peak"]:
        return "DRIFT_SETTLED", fam
    return "PLATEAU_BROKEN", fam


def direction_table(arm_records):
    """3a's read-out: WHICH way each RNG stream broke, not only how far.

    A scalar ``c6`` cannot separate "the algorithm has a fixed direction bias"
    from "the noise picked a direction".  The five-rotation pattern and the
    ``S1/S2/S3`` ordering can: if every seed enhances the same family the bias is
    geometric or in the parameterisation; if different seeds enhance different
    families at near-degenerate energies, MC noise is selecting among
    near-degenerate directions.
    """
    out = []
    for a in arm_records:
        r1 = a["rows"][1] if len(a["rows"]) > 1 else a["rows"][-1]
        five = np.asarray(r1["c6_five_vector"], float)
        S = np.array([r1["S_G1"], r1["S_G2"], r1["S_G3"]], float) \
            if r1["S_G1"] is not None else np.zeros(3)
        out.append({"rng_seed": int(a["rng_seed"]),
                    "c6_0": float(a["rows"][0]["c6_residual"]),
                    "c6_1": float(r1["c6_residual"]),
                    "five_vector": [float(x) for x in five],
                    "argmax_rotation": float(C6_ROTS[int(np.argmax(five))]),
                    "fam_argmax": (int(np.argmax(S)) + 1) if S.any() else None,
                    "fam_argmin": (int(np.argmin(S)) + 1) if S.any() else None,
                    "S": [float(x) for x in S],
                    "E_per_particle_1": r1["E_per_particle"],
                    "E_err_per_particle_1": r1["E_err_per_particle"]})
    return out


def direction_agreement(tab):
    """Do the seeds agree?  Reported; never folded into a per-seed label."""
    if len(tab) < 2:
        return {"n": len(tab), "note": "fewer than two seeds: no direction claim"}
    rot = [t["argmax_rotation"] for t in tab]
    fam = [t["fam_argmax"] for t in tab if t["fam_argmax"] is not None]
    return {"n": len(tab), "rotation_argmax": rot,
            "all_same_rotation": len(set(rot)) == 1,
            "family_argmax": fam,
            "all_same_family": bool(fam) and len(set(fam)) == 1,
            "c6_1_spread": float(max(t["c6_1"] for t in tab)
                                 - min(t["c6_1"] for t in tab)),
            "note": "reported separately: finite-size spontaneous breaking "
                    "predicts seed-dependent direction selection too, so this "
                    "alone is not evidence against a real broken basin"}


def scale_scaling(rungs):
    """3b's read-out: fit ``R_C6(1)`` against ``n`` on a log-log line.

    A slope of ``-1/2`` is what pure MC noise gives, and it is the operational
    stand-in for the ideal no-noise limit, which cannot be reached directly.  A
    slope near 0 is an algorithmic anisotropy that more sampling will not fix.
    """
    xs = np.array([math.log(max(float(r["n"]), 1.0)) for r in rungs], float)
    ys = np.array([math.log(max(float(r["c6_1"]), 1e-300)) for r in rungs], float)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(rungs) >= 2 else None
    per_decade = (None if slope is None
                  else float(rungs[0]["c6_1"] * (rungs[-1]["n"]
                                                 / rungs[0]["n"]) ** slope))
    return {"n": [int(r["n"]) for r in rungs],
            "nsweep": [int(r["nsweep"]) for r in rungs],
            "equil": [int(r["equil"]) for r in rungs],
            "c6_1": [float(r["c6_1"]) for r in rungs],
            "slope_log_c6_vs_log_n": slope,
            "c6_1_last_rung_extrapolated_from_first": per_decade,
            "consistent_with_noise": (None if slope is None
                                      else bool(abs(slope + 0.5) <= 0.25)),
            "note": "R_C6(1) is linear in the breaking amplitude, so a "
                    "noise-limited first update gives slope -0.5.  The first "
                    "rung is the budget's own walk, so it doubles as an internal "
                    "consistency check against the main trajectory."}


def seed_floor(arm_records, key, arms=("joint",)):
    """The seed-to-seed scatter of ``key`` at ``t = 0`` -- the noise floor.

    Every joint arm starts from the SAME ``theta0`` (``crystal_v0`` and
    ``crystal_R0`` are deterministic and identical for every seed; only the SR
    sampling stream differs).  So the arms' ``t = 0`` values are independent
    estimates of *the same* number, and their scatter **is** the measurement
    floor.

    This is what stops a "the three S families split apart" claim that is
    smaller than the S estimator's own noise.  It matters here: in the pilot the
    seed's own ``S(G1), S(G2), S(G3)`` read 6.135 / 5.975 / 6.412 -- a spread of
    0.44, or 7% -- *before any SR step*, which is the liquid's-own-shell-spread
    trap this project has already paid for once.  One arm cannot support a
    floor, so a single arm gets ``None`` and the report says so, rather than
    0.0, which would make every later value look significant.
    """
    vals = [float(a["rows"][0][key]) for a in arm_records
            if a["arm"] in arms and a["rows"] and a["rows"][0].get(key) is not None]
    if len(vals) < 2:
        return {"n": len(vals), "mean": None, "sd": None, "values": vals,
                "note": f"fewer than two arms: no floor for {key} is available "
                        f"and no claim is made about it"}
    return {"n": len(vals), "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)), "values": vals,
            "note": f"seed-to-seed scatter of {key} at t = 0, where every arm "
                    f"is the SAME state"}


def force_floor(arm_records):
    """The ``|f|`` noise floor, derived rather than assumed -- or ``None``.

    Every joint arm starts from the SAME ``theta0``, so their ``t = 0`` values
    are independent estimates of the same gradient and their scatter *is* the
    floor.  A single seed cannot support an ``|f|`` claim at all, so a single
    seed gets ``None`` and the report says no floor is available -- rather than
    a floor of 0.0, which would make every later ``|f|`` look significant.
    """
    f0 = [a["rows"][0]["force"] for a in arm_records
          if a["arm"] == "joint" and a["rows"][0]["force"] is not None]
    if len(f0) < 2:
        return {"n": len(f0), "floor": None,
                "values": [float(x) for x in f0],
                "note": "fewer than two joint arms: no |f| floor is available "
                        "and no |f| claim is made"}
    return {"n": len(f0), "floor": float(np.std(f0, ddof=1)),
            "values": [float(x) for x in f0],
            "note": "the seed-to-seed scatter of |f| at t = 0, where every arm "
                    "starts from the same theta0"}


def energy_buys_breaking(joint_row, pinned_row):
    """3d's read-out: does the broken joint endpoint beat the pinned optimum?"""
    if pinned_row is None or joint_row is None:
        return None
    dj = float(joint_row["E_per_particle"])
    dp = float(pinned_row["E_per_particle"])
    se = math.hypot(float(joint_row["E_err_per_particle"] or 0.0),
                    float(pinned_row["E_err_per_particle"] or 0.0))
    z = (dj - dp) / se if se > 0 else 0.0
    return {"E_joint_per_particle": dj, "E_pinned_per_particle": dp,
            "delta": dj - dp, "z": float(z),
            "reads": ("joint is LOWER: breaking buys energy" if dj - dp < -2.0 * se
                      else "joint is HIGHER: breaking costs energy"
                      if dj - dp > 2.0 * se
                      else "indistinguishable at this budget"),
            "c6_joint": float(joint_row["c6_residual"]),
            "c6_pinned": float(pinned_row["c6_residual"]),
            "note": "E is the least-resolved observable here; read z, not delta"}


# ==========================================================================
# reporting
# ==========================================================================
def print_rows(rows, label):
    """The table, one line per step.

    Every column that can be missing goes through ``_f``, which renders a
    skipped measurement as blanks.  That is the point: an empty cell says the
    stride skipped it, and a filled one would say something was measured.
    """
    print(f"\n    {label}", flush=True)
    print("       t   C6        nbar   E/ne       +-        |f|      |dv|    "
          "S(G1)  S(G2)  S(G3)  bragg    ctr   reg(g1)     id     E_walk+-"
          "   tau   n", flush=True)
    for r in rows:
        e, ee = r["E_per_particle"], r["E_err_per_particle"]
        print("     {t:3d} {c6:9.2e} {nb:6.3f} {e:>9s} {ee:>9s} {f:>8s} "
              "{dv:>6s} {s1:>6s} {s2:>6s} {s3:>6s} {br:>7s} {ct:>6s} "
              "{rg:>9s} {id:>6s} {se:>10s} {tau:>6s} {n:>4s}  {mark}".format(
                  t=r["t"], c6=r["c6_residual"], nb=r["nbar"],
                  e="--" if e is None else f"{e:+.6f}",
                  ee="--" if ee is None else f"{ee:.6f}",
                  f=_f(r["force"], 8),
                  dv=_f(r["dv_norm"], 6),
                  s1=_f(r["S_G1"], 6), s2=_f(r["S_G2"], 6), s3=_f(r["S_G3"], 6),
                  br=_f(r["bragg_ratio"], 7),
                  ct=_f(r["site_midpoint_contrast"], 6),
                  rg=_f(r["reg_g1"], 9, signed=True),
                  id=_f(r["reg_identity"], 6),
                  se=("--" if r["s_E_per_particle"] is None
                      else f"{r['s_E_per_particle']:+.4f}"),
                  tau=_f(r["s_E_tau"], 6), n=_f(r["s_n"], 4, integer=True),
                  mark="closing" if r["closing"] else ""), flush=True)


def _f(x, w=6, signed=False, integer=False):
    """A fixed-width cell, or blanks when the measurement was skipped."""
    if x is None:
        return " " * w
    if integer:
        return f"{int(x):d}".rjust(w)
    return (f"{x:+.{max(w - 3, 1)}f}" if signed
            else f"{x:.{max(w - 3, 1)}f}").rjust(w)


def banner(args, bud, vmc, steps, nsweep, equil, s_nsweeps, s_equil, raw, mode):
    pin = css._threading_is_pinned()
    n_per_step = snapshot_count(nsweep, equil, bud.protocol["sr_snap"])
    lines = [
        "=" * 78,
        f"crystal_c6_trajectory -- why a C6-correct seed leaves the C6 sector "
        f"(wigner_vmc {__version__})",
        "=" * 78,
        f"  mode             {mode}",
        f"  budget           {bud.name}   ({bud.provenance})",
        f"  N                {vmc.N}   nk = {vmc.lat.nk}",
        f"  r_s              {vmc.rs:g}   kappa = {vmc.kappa:.8f}",
        f"  n_max            {vmc.nmax}   n_bands = {vmc.n_bands}",
        f"  len(theta)       {NJ + 2 * (vmc.n_bands - 1) * vmc.lat.nk}",
        f"  SR               {steps} x {nsweep} sweeps   (equil {equil}, "
        f"snap {bud.protocol['sr_snap']})",
        f"  E resolution     n = {n_per_step} snapshots/step -> E_err is a LOWER "
        f"bound (autocorrelation ignored; consecutive rows share the walker)",
        f"  S walk           every {args.s_every} step(s), {s_nsweeps} sweeps "
        f"(equil {s_equil}), its OWN rng",
        f"  raw rng seeds    {raw}   (index 0 = the budget's own rng_seed)",
        f"  gate             c6_residual <= {C6_GATE:g};  bragg >= "
        f"{BRAGG_GATE:g}x liquid;  contrast >= {CONTRAST_GATE:g}",
        f"  BLAS threads     {'pinned' if pin else 'NOT pinned'}: "
        + ", ".join(f"{v}={os.environ.get(v)}" for v in css._BLAS_VARS[:3]),
        f"  results          {_rel(RESULTS)}",
        "=" * 78,
        "  C6 is NOT an order parameter: v = 0 (the filled LLL) is invariant under",
        "  every rotation, so the liquid reads ~3e-14.  Stage A first asks whether",
        "  the state is still a CRYSTAL; only then does C6 mean symmetry.",
        "=" * 78,
    ]
    print("\n".join(lines), flush=True)
    if not pin:
        print("  NOTE: BLAS threading is not pinned; the anchor comparison is "
              "reproducible only to ~1e-11.", flush=True)


def plottable(rows, key):
    """The ``(t, value)`` pairs a panel can actually draw.

    A row is allowed to have no value for a key -- the closing row has no
    optimiser record (so no ``E_total``, no ``force``), the first row has no
    difference, and a stride leaves gaps in the S keys.  Those are *absent
    measurements*, not zeros, so a panel draws the subset that exists and the
    figure never invents a point.  Returning a shorter series is the honest
    outcome; returning zeros, or dropping the whole panel because one row is
    missing, are both wrong.
    """
    return [(int(r["t"]), float(r[key])) for r in rows if r.get(key) is not None]


def write_figure(fig_dir, arms, liquid, ffloor, budget, header):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "c6_trajectory.png")
    stamp = figure_stamp(budget)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.6))
    joints = [a for a in arms if a["arm"] == "joint"]

    def draw(ax, key, label, logy=False, level=None, level_label=None):
        for a in arms:
            pts = plottable(a["rows"], key)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=3.0, lw=0.9,
                    alpha=0.55, ls="-" if a["arm"] == "joint" else "--",
                    label=f"{a['arm']} seed {a['idx']}")
        if level is not None:
            ax.axhline(level, color="0.3", ls=":", lw=1.2)
            if level_label:
                ax.annotate(level_label, (0.02, level),
                            xycoords=("axes fraction", "data"), fontsize=7.5,
                            color="0.3", va="bottom")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("SR step $t$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.5)

    draw(axes[0, 0], "c6_residual", r"$C_6$ residual", logy=True,
         level=C6_GATE, level_label=f"gate {C6_GATE:g}")
    axes[0, 0].set_title(r"$C_6$ is linear in the breaking amplitude",
                         fontsize=10)

    ax = axes[0, 1]
    for a in joints:
        pts = plottable(a["rows"], "E_per_particle")
        if not pts:
            continue
        xs, ys = zip(*pts)
        es = [float(r.get("E_err_per_particle") or 0.0)
              for r in a["rows"] if r.get("E_per_particle") is not None]
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.0, lw=0.9, capsize=2,
                    alpha=0.6, label=f"seed {a['idx']}")
    ax.set_xlabel("SR step $t$")
    ax.set_ylabel(r"$E/N$ (per particle)")
    ax.set_title("energy -- descriptive only, bars are a LOWER bound;\n"
                 "the closing row has no optimiser record and is not plotted",
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    draw(axes[0, 2], "dv_norm", r"$\|\Delta v_t\|$ (orbital block)")
    axes[0, 2].set_title("the honest “has it settled?”", fontsize=10)

    ax = axes[1, 0]
    draw(ax, "force", r"$\|f_t\|$", logy=True,
         level=ffloor["floor"],
         level_label=("t = 0 seed-to-seed floor (+-1sd band shown)"
                      if ffloor["floor"] is not None else None))
    if ffloor["floor"] is not None:
        ax.axhspan(max(ffloor["floor"] - np.std(ffloor["values"], ddof=1), 1e-300),
                   ffloor["floor"] + np.std(ffloor["values"], ddof=1),
                   color="0.3", alpha=0.12, lw=0)
    ax.set_title(r"$\|f\|\to 0$ is the JASTROW converging", fontsize=10)

    ax = axes[1, 1]
    for a in joints:
        for fam, ls in zip(("S_G1", "S_G2", "S_G3"), ("-", "--", ":")):
            pts = plottable(a["rows"], fam)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=2.5, ls=ls, lw=0.9, alpha=0.5,
                    label=f"$S_{{{fam[-1]}}}$ seed {a['idx']}")
    if liquid is not None:
        ax.axhline(float(liquid["bragg_peak"]), color="0.3", ls=":", lw=1.0)
    ax.set_xlabel("SR step $t$")
    ax.set_ylabel("$S$ per first-shell family")
    ax.set_title(r"the split: $S_1 \approx S_2 \approx S_3$ separating",
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1, 2]
    for a in joints:
        for key, ls, lab in (("reg_g1", "-", r"$\phi(g_1)$"),
                             ("site_midpoint_contrast", "--", "contrast")):
            pts = plottable(a["rows"], key)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=2.5, ls=ls, lw=0.9, alpha=0.5,
                    label=f"{lab} seed {a['idx']}")
    ax.set_xlabel("SR step $t$")
    ax.set_ylabel(r"$\phi(g_1)$  /  contrast")
    ax.set_title("registry phase and real-space contrast", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    for ax in axes.ravel():
        ax.legend(fontsize=5.6, ncol=2, loc="best", framealpha=0.7)
    fig.suptitle(f"C6 SR trajectory at r_s = {header['rs']:g}, N = {header['N']}, "
                 f"n_max = {header['nmax']}, budget {budget}", fontsize=11)
    if stamp:
        fig.text(0.5, 0.011, stamp.replace("\n", " -- "), ha="center",
                 fontsize=11, color="crimson", alpha=0.85)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
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
                   help="the seeding coupling (default: %(default)s)")
    p.add_argument("--nmax", type=int, default=DEFAULT_NMAX,
                   help="Landau-level cutoff (default: %(default)s)")
    p.add_argument("--n-electrons", type=int, default=N_ELECTRONS,
                   help="electron count (default: %(default)s)")
    p.add_argument("--steps", type=int, default=None,
                   help="SR steps; default is 3x the budget's sr_steps")
    p.add_argument("--nsweep", type=int, default=None,
                   help="SR sweeps per step (default: the budget's sr_sweeps)")
    p.add_argument("--equil", type=int, default=None,
                   help="SR equilibrium sweeps (default: the budget's sr_equil)")
    p.add_argument("--init-id", type=int, default=0,
                   help="which inits[] width to seed from (default: %(default)s)")
    p.add_argument("--seeds", default="0,1,2",
                   help="seed INDICES, offset from the budget's configured "
                        "rng_seed: raw = base + index, so index 0 IS the "
                        "production reference (default: %(default)s)")
    p.add_argument("--s-every", type=int, default=1,
                   help="measure S/registry/contrast every N steps; rows 0 and "
                        "the closing row are always measured (default: "
                        "%(default)s -- the S split is the Layer-2 signature and "
                        "a stride can step over it)")
    p.add_argument("--s-nsweeps", type=int, default=None,
                   help="sweeps in the S walk (default: the budget's meas_sweeps)")
    p.add_argument("--s-equil", type=int, default=None,
                   help="equilibration in the S walk (default: meas_equil)")
    p.add_argument("--s-sigma", type=float, default=None,
                   help="proposal sigma for the S walk (default: crystal_measure)")
    p.add_argument("--rng-seeds", nargs="?", const="", default=None,
                   help="3a: RAW rng_seed values (comma list) for short "
                        "trajectories that differ ONLY in the SR sampling stream; "
                        "a bare flag uses " + ",".join(str(s) for s in DEFAULT_RNG_SEEDS))
    p.add_argument("--nsweep-ladder", nargs="?", const="", default=None,
                   help="3b: comma list of SR sweep counts (bare flag uses "
                        + ",".join(str(s) for s in DEFAULT_NSWEEP_LADDER)
                        + "); first 3 steps, equil scaled with nsweep")
    p.add_argument("--pilot-scale", action="store_true",
                   help="3c: add an arm with scale = sr_pilot_scale(...)")
    p.add_argument("--pinned-control", action="store_true",
                   help="3d: add an arm with the orbitals frozen at the seed")
    p.add_argument("--skip-liquid", action="store_true",
                   help="skip the liquid null (Stage A then reports UNKNOWN)")
    p.add_argument("--nsp", type=int, default=C6_NSP,
                   help="sample points for the C6 residual (default: %(default)s)")
    p.add_argument("--no-figure", action="store_true",
                   help="write the table and metadata only")
    return p


def plan_jobs(args, bud, steps, nsweep, equil):
    """The jobs to run: ``(idx, rng_seed, steps, nsweep, equil, arms, tag)``.

    Three modes, and they are three DIFFERENT questions rather than three sizes
    of one.  ``--seeds`` indexes off the budget's own rng_seed, so index 0 is
    the production reference; ``--rng-seeds`` and ``--nsweep-ladder`` are the
    Layer-3 axes and carry no comparison with it.
    """
    proto = bud.protocol
    if args.rng_seeds is not None:
        raw = int_list(args.rng_seeds) if args.rng_seeds else list(DEFAULT_RNG_SEEDS)
        s = int(steps) if args.steps is not None else min(int(steps), DEFAULT_STEPS_3A)
        return "3a -- RNG direction test", "__stage3a", [
            dict(idx=i, rng_seed=int(r), steps=s, nsweep=int(nsweep),
                 equil=int(equil), arms=["joint"], tag="3a")
            for i, r in enumerate(raw)]
    if args.nsweep_ladder is not None:
        ladder = int_list(args.nsweep_ladder) if args.nsweep_ladder \
            else list(DEFAULT_NSWEEP_LADDER)
        frac = float(proto["sr_equil"]) / float(proto["sr_sweeps"])
        s = int(steps) if args.steps is not None else min(int(steps), DEFAULT_STEPS_3A)
        jobs = []
        for sw in ladder:
            eq = max(1, int(round(frac * sw)))
            jobs.append(dict(idx=int(sw), rng_seed=None, steps=s, nsweep=int(sw),
                             equil=eq, arms=["joint"], tag="3b"))
        return ("3b -- sample-size scaling",
                "__stage3b-" + "-".join(str(x) for x in ladder), jobs)
    seeds = int_list(args.seeds)
    arms = ["joint"]
    if args.pilot_scale:
        arms.append("joint_pilot")
    if args.pinned_control:
        arms.append("pinned")
    extra = ("__pilot" if args.pilot_scale else "") + \
            ("__pinned" if args.pinned_control else "")
    return ("layers 1-2 -- the trajectory", extra, [
        dict(idx=int(s), seed_index=int(s), rng_seed=None, steps=int(steps),
             nsweep=int(nsweep), equil=int(equil), arms=arms, tag="traj")
        for s in seeds])


def main(argv=None):
    args = build_parser().parse_args(argv)

    if int(args.nmax) != 1:
        # Not a limitation of the C6 context -- that is n_max-agnostic -- but the
        # pinned corner, the reference point and the observed drift are all at
        # n_max = 1, and a mismatch would silently compare different manifolds.
        raise SystemExit(
            f"--nmax {args.nmax} is refused: this diagnostic is built around the "
            f"n_max = 1 point at r_s = 90 where the drift was observed.")
    if int(args.s_every) < 1:
        raise SystemExit(f"--s-every must be >= 1, got {args.s_every}")
    if args.rng_seeds is not None and args.nsweep_ladder is not None:
        raise SystemExit("--rng-seeds (3a) and --nsweep-ladder (3b) are separate "
                         "experiments; run them one at a time.")

    bud = load_budget(args.budget)
    proto = bud.protocol
    steps = int(args.steps) if args.steps is not None else 3 * int(proto["sr_steps"])
    nsweep = int(args.nsweep) if args.nsweep is not None else int(proto["sr_sweeps"])
    equil = int(args.equil) if args.equil is not None else int(proto["sr_equil"])
    s_nsweeps = int(args.s_nsweeps) if args.s_nsweeps is not None \
        else int(proto["meas_sweeps"])
    s_equil = int(args.s_equil) if args.s_equil is not None \
        else int(proto["meas_equil"])
    s_sigma = float(args.s_sigma) if args.s_sigma is not None \
        else float(bud.crystal_measure["sigma"])
    s_seed = int(bud.crystal_measure["seed"])
    target_acc = float(bud.sr["target_acc"])

    mode, slug_extra, jobs = plan_jobs(args, bud, steps, nsweep, equil)

    vmc = VMC(N=int(args.n_electrons), rs=float(args.rs), phase="crystal",
              nmax=int(args.nmax))
    torus = analysis_torus(vmc.N)
    bragg_q = bragg_vectors(torus)
    bg_q = background_vectors(torus)
    families = bragg_families(torus)
    ctx = c6_context(vmc, torus, nsp=int(args.nsp))

    # The rng_seed the BUDGET configures for this point.  For `--seeds`, index i
    # is `base + i`, so index 0 is bit-for-bit the production reference -- which
    # is what makes the pre-registered anchor a real prediction rather than a
    # coincidence.  `v0`, `c0` and `R0` are identical for EVERY seed; only the
    # optimiser's sampling stream varies.
    base_cfg = resolve(phase="crystal", rs=vmc.rs, N=vmc.N, nmax=vmc.nmax,
                       init_id=int(args.init_id), budget=bud.name)
    base_seed = int(base_cfg.rng_seed)
    L0 = float(bud.width_for(int(args.init_id)))
    v0, controls = seed_controls(vmc, torus, ctx, L0, nsp2=4 * int(args.nsp),
                                 seed2=C6_SEED + 1)
    controls.update(rs=float(vmc.rs), nmax=int(vmc.nmax), N=int(vmc.N),
                    L0=float(L0), L0_source=f"inits[{int(args.init_id)}].L0 of "
                                            f"{bud.name}",
                    base_rng_seed=base_seed,
                    pinned_corner="r_s = 75, n_max = 2 "
                                  "(tests/test_projection_convention.py)")
    c0 = sr.jastrow_vector(vmc.kappa, NJ)
    theta0 = theta(c0, v0)
    expect_len = NJ + 2 * (vmc.n_bands - 1) * vmc.lat.nk
    if len(theta0) != expect_len:
        raise SystemExit(f"theta has {len(theta0)} entries, expected {expect_len}")
    # `base + index`, NOT `base` for every job.  Only the trajectory mode carries
    # a `seed_index`; the 3b ladder deliberately does not, because its rungs
    # differ in `nsweep` and must share one sampling stream or the scaling would
    # be measuring the seed as well as the sample size.  (3a sets `rng_seed`
    # outright and never reaches here.)  Assigning the bare base to every job was
    # the bug this comment exists for: it made `--seeds 0,1,2` three bit-identical
    # trajectories, so the per-seed spread, the direction pattern and the `|f|`
    # floor would all have read zero -- and index 0 alone cannot see it.
    for j in jobs:
        if j["rng_seed"] is None:
            j["rng_seed"] = base_seed + int(j.get("seed_index", 0))
    sv = seed_verdict(controls)

    banner(args, bud, vmc, jobs[0]["steps"], jobs[0]["nsweep"], jobs[0]["equil"],
           s_nsweeps, s_equil, [j["rng_seed"] for j in jobs], mode)
    print(f"  bragg set        {len(bragg_q)} first-shell vectors")
    print(f"  background set   {len(bg_q)} momenta")
    print(f"\n  LAYER 1 -- is the seed C6-correct at THIS operating point?")
    print(f"    corrected Gaussian seed     c6 = {controls['c6_correct']:.3e}"
          f"   nbar = {controls['nbar_seed']:.4f}")
    print(f"    second context (nsp={controls['nsp2']}, seed="
          f"{controls['c6_seed2']})   c6 = {controls['c6_correct_ctx2']:.3e}")
    print(f"    conjugating the overlap     c6 = {controls['c6_defect']:.3e}"
          f"   nbar = {controls['nbar_defect']:.4f}   <- the negative control "
          f"(separation {sv['facts']['separation']:.2e})")
    print(f"    columns x(1+.5i), per-rotation max |d c6| = "
          f"{controls['c6_gauge_shift']:.3e}   <- the gauge-invariance control")
    print(f"    -> {'OK' if sv['ok'] else 'FAILED'}: {sv['reason']}\n", flush=True)

    if not sv["ok"]:
        # The seed is not C6 here, so anything downstream measures the seed and
        # not SR.  The run stops rather than producing a trajectory that would
        # be read as "SR broke the symmetry".
        print("  No trajectory is run: the question 'does SR break C6?' is not "
              "well posed until the seed is C6-covariant at this operating "
              "point.", flush=True)
        _write(args, bud, vmc, mode, slug_extra, jobs, s_nsweeps, s_equil,
               s_sigma, target_acc, controls, sv, None, {}, {}, None)
        return 2

    # --- the liquid null, measured in the run ------------------------------
    liquid = None
    if not args.skip_liquid:
        t_liq = time.time()
        liq = VMC(N=vmc.N, rs=vmc.rs, phase="liquid", nmax=vmc.nmax).run(
            init_id=int(args.init_id), budget=bud.name, verbose=False)
        liq_peak, liq_bg, liq_ratio = st.bragg_ratio(liq.state.snaps, bragg_q,
                                                     bg_q, vmc.N)
        liquid = {"E_per_particle": float(liq.energy_per_particle),
                  "E_err_per_particle": float(liq.error),
                  "bragg_peak": float(liq_peak), "bragg_bg": float(liq_bg),
                  "bragg_ratio": float(liq_ratio),
                  "site_midpoint_contrast": float(st.site_midpoint_contrast(
                      density_field(liq.state.snaps, torus), torus, DENSITY_BINS)),
                  "c6_residual": float(max(c6_residuals(
                      np.zeros((vmc.lat.nk, vmc.n_bands - 1), complex),
                      ctx).values())),
                  "seconds": float(time.time() - t_liq)}
        print(f"  liquid null      bragg = {liq_ratio:.4f}   contrast = "
              f"{liquid['site_midpoint_contrast']:.4f}   c6 = "
              f"{liquid['c6_residual']:.2e}   [{liquid['seconds']:.0f}s]\n",
              flush=True)

    # --- the S walk: cached per STATE, so identical states share one walk ----
    s_cache = {}

    def s_of_for(tag):
        def s_of(th):
            key = state_digest(*theta_parts(np.asarray(th, float), vmc.n_bands,
                                            vmc.lat.nk))
            if key not in s_cache:
                t0 = time.time()
                rec = s_measurements(vmc, th, torus, bragg_q, bg_q, families,
                                     s_nsweeps, s_equil, s_sigma, s_seed)
                rec["s_seconds"] = float(time.time() - t0)
                s_cache[key] = rec
                print(f"      S walk [{tag}] {rec['s_seconds']:6.1f}s   bragg "
                      f"{rec['bragg_ratio']:6.3f}   ctr "
                      f"{rec['site_midpoint_contrast']:5.2f}   c6 "
                      f"{rec['s_E_per_particle']:+.6f}+-"
                      f"{rec['s_E_err_per_particle']:.6f}", flush=True)
            return dict(s_cache[key])
        return s_of

    # --- run ---------------------------------------------------------------
    arms = []
    t_start = time.time()
    for j in jobs:
        s_of = s_of_for(f"{j['tag']}:{j['idx']}")
        for arm_name in j["arms"]:
            t0 = time.time()
            if arm_name == "joint":
                th, hist, state_at = run_joint_arm(vmc, theta0, bud, j["rng_seed"],
                                                  j["steps"], j["nsweep"], j["equil"])
            elif arm_name == "joint_pilot":
                tp = time.time()
                scale = pilot_scale_for(vmc, theta0, bud, j["rng_seed"],
                                        j["nsweep"], j["equil"])
                print(f"    pilot scale [{time.time() - tp:.0f}s]  Jastrow block "
                      f"{scale[:NJ].min():.3g}..{scale[:NJ].max():.3g}   orbital "
                      f"block {scale[NJ:].min():.3g}..{scale[NJ:].max():.3g}   "
                      f"ratio {scale[:NJ].mean() / scale[NJ:].mean():.1f}x",
                      flush=True)
                th, hist, state_at = run_joint_arm(vmc, theta0, bud, j["rng_seed"],
                                                   j["steps"], j["nsweep"],
                                                   j["equil"], scale=scale)
            else:
                th, hist, state_at = run_pinned_arm(vmc, v0, c0, bud, j["rng_seed"],
                                                    j["steps"], j["nsweep"],
                                                    j["equil"])
            if not np.all(np.isfinite(th)):
                # `_run_from` refuses to measure non-finite parameters
                # (api.py:791) and calls it a convergence failure rather than a
                # result.  Same refusal here: a trajectory that diverged is not
                # evidence about C6.
                raise SystemExit(
                    f"the optimiser returned non-finite parameters on arm "
                    f"{arm_name} (tag {j['tag']}, idx {j['idx']}); this is a "
                    f"convergence failure, not a result.")
            meas = make_meas(vmc, ctx, s_of, int(args.s_every))
            rows = build_rows(theta0, th, hist, state_at, meas)
            arms.append({"arm": arm_name, "tag": j["tag"], "idx": int(j["idx"]),
                         "rng_seed": int(j["rng_seed"]), "steps": int(j["steps"]),
                         "nsweep": int(j["nsweep"]), "equil": int(j["equil"]),
                         "rows": rows, "seconds": float(time.time() - t0)})
            r1 = rows[1] if len(rows) > 1 else rows[-1]
            print(f"\n  arm {arm_name}  tag {j['tag']}  idx {j['idx']}  rng_seed "
                  f"{j['rng_seed']}  [{arms[-1]['seconds']:.0f}s]", flush=True)
            print(f"    c6: {rows[0]['c6_residual']:.3e} -> "
                  f"{r1['c6_residual']:.3e} (first update) -> "
                  f"{rows[-1]['c6_residual']:.3e} (closing)")
            print_rows(rows, f"rows ({arm_name}, {j['tag']}, idx {j['idx']})")
            if j is jobs[0] and arm_name == j["arms"][0]:
                el = time.time() - t_start
                print(f"\n  measured {el / max(int(j['steps']), 1):.1f} s/SR-step "
                      f"(first job, {j['steps']} steps); "
                      f"{len(jobs) * len(j['arms']) - 1} arm(s) to go\n", flush=True)

    # --- the |f| noise floor, free from the shared theta0 -------------------
    joints = [a for a in arms if a["arm"] == "joint"]
    floor = force_floor(joints)
    if floor["floor"] is None:
        print(f"\n  |f| floor: {floor['note']}", flush=True)
    else:
        print("\n  |f| at t = 0: "
              + ", ".join(f"{x:.3e}" for x in floor["values"])
              + f"   -> seed-to-seed floor {floor['floor']:.3e}  (every seed "
              f"starts from the SAME theta0, so this scatter IS the floor)",
              flush=True)

    # The same argument applies to every quantity measured on the seed, and it
    # has to: the S split is the Layer-2 signature, and the estimator's own
    # scatter is the same size as the split being looked for.  Without this
    # floor a "split" that is pure estimator noise reads as a finding.
    t0 = {k: seed_floor(joints, k)
          for k in ("S_G1", "S_G2", "S_G3", "bragg_ratio",
                    "site_midpoint_contrast", "c6_residual")}
    print("\n  t = 0 floors (every joint arm starts from the SAME state):",
          flush=True)
    for k, f in t0.items():
        if f["sd"] is None:
            print(f"    {k:24s} n = {f['n']}: {f['note']}", flush=True)
            continue
        # The relative size is the readable number for S, bragg and contrast;
        # for `c6_residual` the mean is at the 1e-14 floor, so a percentage of
        # it would be meaningless and is not printed.
        rel = (f"   ({f['sd'] / abs(f['mean']):.2%} of the mean)"
               if abs(f["mean"]) > 1e-6 else "")
        print(f"    {k:24s} mean {f['mean']:.6g}   sd {f['sd']:.3e}{rel}",
              flush=True)
    if t0["S_G1"]["sd"] is not None:
        sp = joints[0]["rows"][0]["S_split"]
        if sp is not None:
            f = max(t0["S_G1"]["sd"], t0["S_G2"]["sd"], t0["S_G3"]["sd"])
            print(f"    -> the seed's OWN S split is {sp:.3f} against a "
                  f"single-family floor of {f:.3f}: a later split must clear "
                  f"{f:.3f} before it is a finding.", flush=True)

    # --- the verdicts ------------------------------------------------------
    for a in arms:
        cls, why = stage_a(a["rows"][-1], liquid)
        lab, fam = stage_b(a["rows"], liquid)
        a["stage_a"] = {"class": cls, "reason": why}
        a["stage_b"] = {"label": lab, "facts": fam,
                        "is_a_result": lab not in NOT_A_RESULT}
        print(f"\n  VERDICT arm={a['arm']} tag={a['tag']} idx={a['idx']}: "
              f"Stage A = {cls}   Stage B = {lab}")
        print(f"    Stage A: {why}")
        print(f"    Stage B: c6_0 = {fam['c6_0']:.3e}, peak = "
              f"{fam['c6_peak']:.3e} at t = {fam['t_peak']}, closing = "
              f"{fam['c6_T']:.3e}, turned over = {fam['turned_over']}")
        if lab in NOT_A_RESULT:
            print(f"    '{lab}' names the NEXT experiment, NOT an answer about "
                  f"(a)-(d).", flush=True)

    cross = {"n_joint_arms": len(joints),
             "n_closing_below_gate": sum(
                 1 for a in joints if a["stage_b"]["facts"]["c6_T"] <= C6_GATE),
             "labels": [a["stage_b"]["label"] for a in joints],
             "note": "reported separately and never folded into a per-seed label"}
    if len(joints) >= 2:
        cross["per_t_spread"] = [
            float(np.ptp([a["rows"][t]["c6_residual"] for a in joints
                          if t < len(a["rows"])]))
            for t in range(max(len(a["rows"]) for a in joints))]

    # --- the Layer-3 sub-experiments ---------------------------------------
    extra = {}
    if any(j["tag"] == "3a" for j in jobs):
        tab = direction_table([a for a in arms if a["tag"] == "3a"])
        ag = direction_agreement(tab)
        extra["stage3a"] = {"table": tab, "agreement": ag,
                            "protocol": {"steps": jobs[0]["steps"],
                                         "nsweep": jobs[0]["nsweep"],
                                         "equil": jobs[0]["equil"],
                                         "raw_rng_seeds": [j["rng_seed"]
                                                           for j in jobs]}}
        print("\n" + "=" * 78)
        print("STAGE 3a -- same theta0, same R0, same everything but the SR rng")
        print("=" * 78)
        for t in tab:
            print("  rng %-10d c6_1 = %.3e   argmax rotation %5.0f   "
                  "S argmax G%s argmin G%s   E/ne %+.6f"
                  % (t["rng_seed"], t["c6_1"], t["argmax_rotation"],
                     t["fam_argmax"], t["fam_argmin"], t["E_per_particle_1"]))
        print(f"  {ag}")
        print("  SAME direction every seed  -> a fixed bias in the geometry, the "
              "parameterisation, or the SR implementation;")
        print("  DIFFERENT directions at near-degenerate energies -> MC noise "
              "selecting among near-degenerate directions.", flush=True)

    if any(j["tag"] == "3b" for j in jobs):
        rungs = [dict(nsweep=a["nsweep"], equil=a["equil"],
                      n=snapshot_count(a["nsweep"], a["equil"],
                                       proto["sr_snap"]),
                      c6_1=a["rows"][1]["c6_residual"],
                      c6_0=a["rows"][0]["c6_residual"],
                      E_per_particle_1=a["rows"][1]["E_per_particle"])
                 for a in arms if a["tag"] == "3b"]
        sc = scale_scaling(rungs)
        extra["stage3b"] = {"rungs": rungs, "scaling": sc,
                            "equil_fraction": float(proto["sr_equil"])
                            / float(proto["sr_sweeps"])}
        print("\n" + "=" * 78)
        print("STAGE 3b -- does R_C6(1) fall like n^-1/2?")
        print("=" * 78)
        for r in rungs:
            print(f"  nsweep {r['nsweep']:5d}  equil {r['equil']:4d}  n "
                  f"{r['n']:4d}  c6_1 = {r['c6_1']:.3e}")
        print(f"  log-log slope vs n: {sc['slope_log_c6_vs_log_n']}")
        print(f"  consistent with pure MC noise (slope -0.5 +- 0.25): "
              f"{sc['consistent_with_noise']}")
        print(f"  {sc['note']}", flush=True)

    if args.pilot_scale and not any(j["tag"] == "3a" for j in jobs):
        extra["stage3c"] = {
            "statement": "scale_i = 1/sqrt(S_ii) from a pilot walk, so eps is "
                         "applied on the same footing to both blocks instead of "
                         "against raw units, where the Jastrow's O(10)-O(40) "
                         "derivatives swamp the orbital O(1e-3) by ~5 orders",
            "read": "compare the joint_pilot arm's c6_1 and dv_norm against the "
                    "joint arm's at the SAME rng_seed"}
    if args.pinned_control and not any(j["tag"] == "3a" for j in jobs):
        comp = []
        for j in jobs:
            ja = next((a for a in arms if a["arm"] == "joint"
                       and a["rng_seed"] == j["rng_seed"]), None)
            pa = next((a for a in arms if a["arm"] == "pinned"
                       and a["rng_seed"] == j["rng_seed"]), None)
            if ja is None or pa is None:
                continue
            r = energy_buys_breaking(ja["rows"][-1], pa["rows"][-1])
            comp.append({"idx": int(j["idx"]), "rng_seed": int(j["rng_seed"]), **r})
            print(f"  idx {j['idx']}: joint E/ne {r['E_joint_per_particle']:+.6f} "
                  f"vs pinned {r['E_pinned_per_particle']:+.6f}   delta "
                  f"{r['delta']:+.6f}   z {r['z']:+.2f}   -> {r['reads']}",
                  flush=True)
        extra["stage3d"] = {
            "comparison": comp,
            "statement": "the pinned arm freezes v at the seed, so its C6 cannot "
                         "move; this asks whether the broken joint endpoint is "
                         "energetically better than the symmetric optimum"}

    # --- persist -----------------------------------------------------------
    res_dir, fig_dir = _write(args, bud, vmc, mode, slug_extra, jobs, s_nsweeps,
                              s_equil, s_sigma, target_acc, controls, sv, liquid,
                              cross, extra, arms, floor=floor, t0=t0)
    print(f"\n  wrote {_rel(os.path.join(res_dir, 'trajectory.json'))}")
    print(f"  wrote {_rel(os.path.join(res_dir, 'run_metadata.json'))}", flush=True)
    if not args.no_figure:
        p = write_figure(fig_dir, arms, liquid, floor, bud.name,
                         {"N": int(vmc.N), "rs": float(vmc.rs),
                          "nmax": int(vmc.nmax)})
        print(f"  wrote {_rel(p)}", flush=True)
    return 0


def _write(args, bud, vmc, mode, slug_extra, jobs, s_nsweeps, s_equil, s_sigma,
           target_acc, controls, sv, liquid, cross, extra, arms, floor=None,
           t0=None):
    """Persist the trajectory and the metadata; return ``(res_dir, fig_dir)``.

    The full per-step ``theta`` is kept.  The ``quick`` numbers this diagnostic
    exists to explain are stored nowhere, so if C6 later turns out to be the
    wrong instrument the whole trajectory has to be re-derivable without paying
    for SR a second time.
    """
    steps = int(jobs[0]["steps"])
    nsweep = int(jobs[0]["nsweep"])
    slug = (request_slug(vmc.N, vmc.ansatz, bud.name, vmc.rs, vmc.nmax)
            + f"__steps{steps}__nsweep{nsweep}" + slug_extra)
    res_dir = os.path.join(RESULTS, slug)
    fig_dir = os.path.join(FIGDIR, slug)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "trajectory.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"mode": mode, "arms": arms, "liquid": liquid,
                   "cross_seed": cross, "layer1_seed_controls": controls,
                   "layer1_verdict": sv, **extra}, fh, indent=2,
                  sort_keys=True, default=_jd)
    meta = {
        "run_metadata_schema": "phase_competition_2/c6_trajectory/1",
        "recipe": "examples/phase_competition_2/crystal_c6_trajectory.py",
        "wigner_vmc_version": __version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "question": "why does a C6-correct Gaussian seed leave the C6 sector "
                    "under unconstrained joint SR?",
        "budget": bud.name,
        # ONE definition, read from the sibling: `figure_stamp` returns None for
        # the production budget and the QUICK stamp otherwise.  Re-typing the
        # comparison here would let the mark and the metadata disagree.
        "quality": QUALITY_MARK,
        "figure_stamp": figure_stamp(bud.name),
        "blas_env": blas_env(),
        "blas_threading_pinned": bool(css._threading_is_pinned()),
        "N": int(vmc.N), "rs": float(vmc.rs), "kappa": float(vmc.kappa),
        "nmax": int(vmc.nmax), "n_bands": int(vmc.n_bands),
        "ansatz": str(vmc.ansatz),
        "len_theta": NJ + 2 * (vmc.n_bands - 1) * vmc.lat.nk,
        "jobs": [{k: v for k, v in j.items()} for j in jobs],
        "protocol": {
            "steps": steps, "nsweep": nsweep, "equil": int(jobs[0]["equil"]),
            "snapshot_every": int(bud.protocol["sr_snap"]),
            "snapshots_per_step": snapshot_count(nsweep, jobs[0]["equil"],
                                                 bud.protocol["sr_snap"]),
            "sr_sigma": float(bud.sr["sigma"]), "target_acc": target_acc,
            "s_every": int(args.s_every), "s_nsweeps": int(s_nsweeps),
            "s_equil": int(s_equil), "s_sigma": float(s_sigma),
            "s_seed": int(bud.crystal_measure["seed"]),
            "base_rng_seed": int(controls["base_rng_seed"]),
            "seed_index_rule": "raw = base_rng_seed + index, so index 0 IS the "
                               "production reference for this budget",
            "tau_schedule": "tau_i = tau / (1 + xi i / steps); at i = 0 this is "
                            "tau for EVERY steps, which is why R_C6(1) is "
                            "comparable across budgets and step counts while "
                            "R_C6(t>1) is not",
        },
        "criterion": {
            "c6_gate": C6_GATE, "bragg_gate": BRAGG_GATE,
            "contrast_gate": CONTRAST_GATE,
            "stage_a": "is the CLOSING state still a crystal (bragg ratio against "
                       "the liquid measured in the same run, and real-space "
                       "contrast)?  If not, its C6 residual names nothing.",
            "stage_b": "the shape of the C6 trajectory.  SEED_NOT_C6 is checked "
                       "first; STILL_RISING / PLATEAU_BROKEN / REGISTRY_DRIFT "
                       "name the next experiment rather than an answer about "
                       "(a)-(d).",
            "c6_is_linear": "space_residual is the sine of the largest principal "
                            "angle (tests/test_projection_convention.py:277), so "
                            "R_C6(1) IS the relative first-order symmetry-"
                            "breaking amplitude of the first SR update -- there "
                            "is no square root",
            "c6_is_not_an_order_parameter": "v = 0 (the filled LLL) is invariant "
                                            "under every rotation, so the liquid "
                                            "reads ~3e-14: a low C6 is also the "
                                            "liquid",
            "energy_resolution": f"{snapshot_count(nsweep, jobs[0]['equil'], bud.protocol['sr_snap'])} "
                                 f"snapshots/step, naive SEM, autocorrelation "
                                 f"ignored and consecutive rows sharing the "
                                 f"walker -> E_err is a LOWER bound; E is "
                                 f"descriptive here and never decisive",
            "force_is_jastrow_dominated": "d ln Psi/dc is O(10)-O(40) while "
                                          "d ln Psi/dv is O(1e-3) "
                                          "(sr_pilot_scale's docstring), so "
                                          "|f| -> 0 means the JASTROW converged; "
                                          "||dv|| is the orbital-block answer",
        },
        "layer1_seed_controls": controls, "layer1_verdict": sv,
        "force_floor": floor, "t0_floors": t0 if t0 is not None else {},
        "extra_blocks": sorted(extra),
        "not_in_scope": ["r_s continuation", "n_max nesting", "phase "
                         "competition", "reproduction/full", "random-start "
                         "ensemble"],
        "result_dir": _rel(res_dir),
    }
    with open(os.path.join(res_dir, "run_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=_jd)
    return res_dir, fig_dir


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    raise SystemExit(main())
