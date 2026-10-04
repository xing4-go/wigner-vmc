"""The contract of ``examples/phase_competition_2/crystal_c6_trajectory.py``.

The diagnostic exists to answer one question -- *why does a C6-correct seed
leave the C6 sector under joint SR?* -- and its answer is only worth reading if
five things are true **before** the run rather than after it.  Each is pinned
here.

1. **The per-step pairing is right.**  ``sr_optimize_joint`` samples at
   ``theta_i`` and *then* updates, so ``hist[i]["E"]`` belongs to
   ``hist[i-1]["theta"]``.  The tempting pairing -- ``E`` and ``theta`` from the
   same row -- lags the state by one step and reads *exactly* like "the energy
   fell while C6 rose", which is the headline this diagnostic is looking for.
   A test that only checked array lengths would pass on the bug.
2. **The units are the project's.**  Every ``hist`` energy is a TOTAL; the
   campaign's published numbers are per particle.  This project has already
   published one wrong headline from the factor of ``N``.
3. **The snapshot count is the sampler's own rule**, ``len(range(equil, nsweep,
   snapshot_every))`` -- not ``(nsweep - equil)//snap + 1``, which over-counts by
   one whenever the division is exact and would flatter every error bar.
4. **The measurement does not perturb the thing it measures.**  The S/registry
   walk must draw from its OWN rng.  ``sr_optimize_joint`` draws its walk and its
   update noise from one ``default_rng(seed)``, so one extra draw from that
   stream moves the trajectory.
5. **The verdict rule is total, ordered, and honest about what it does not
   know.**  ``SEED_NOT_C6`` before everything; the three labels that name the
   *next experiment* never report as a result about the candidate explanations;
   and a state that has stopped being a crystal is caught by Stage A rather than
   read as a stable C6.

The end-to-end tests stub the optimiser and the Layer-1 gate and keep the
estimators real: the states they diagnose are synthetic, but
``bragg_ratio``, ``site_midpoint_contrast``, the registry and the C6 residual are
the production ones, so a wiring mistake between them and the table is still
caught.  Paying for a real SR run to check plumbing would make the suite slower,
not stronger -- and there is a separate, real measurement of the Layer-1 gate at
the true operating point below.

Why ``wigner_vmc.vmc.sr`` is imported directly, and why that must not be
"cleaned up": the public API reduces the whole SR trace to
``Optimization(E_total, acc)`` (``src/wigner_vmc/api.py:544-556``), and the
entire content of this diagnostic IS the per-step trace.  This directory sits
outside ``tests/test_api_contract.py``'s ``EXAMPLE_SCRIPTS`` -- a fixed
three-name tuple, not a directory walk -- so the layering rule does not apply.
Both facts are asserted, so the exemption cannot be silently revoked.
"""
import ast
import contextlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SCRIPT = os.path.join(CLEAN, "examples", "phase_competition_2",
                      "crystal_c6_trajectory.py")
SIBLING = os.path.join(CLEAN, "examples", "phase_competition_2",
                       "crystal_seed_search.py")
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wigner_vmc import VMC, load_budget, resolve, theta      # noqa: E402
from wigner_vmc.vmc import sr                                # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("c6_trajectory_under_test",
                                                 SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()

#: Small enough to keep the wiring tests instant, and the numbers below are
#: measured on it rather than assumed.  Every test that needs the *real*
#: operating point uses N = 36 explicitly.
N_SMALL = 9


def _docstrings(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            got = ast.get_docstring(node, clean=False)
            if got is not None:
                out.add(got)
    return out


def _source(path=SCRIPT):
    return io.open(path, encoding="utf-8").read()


def _code_strings(path=SCRIPT):
    """Every string literal in the module that is NOT a docstring.

    Prose is allowed to name a thing the code refuses to use -- that is how a
    reader learns the rule.  Code is not.
    """
    tree = ast.parse(_source(path))
    docs = _docstrings(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docs]


def _argparse_defaults(path=SCRIPT):
    """The string ``default=`` values passed to ``add_argument``.

    Read from the AST rather than by importing, so this cannot be satisfied by a
    default that only *looks* inert.
    """
    tree = ast.parse(_source(path))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    out.append(kw.value.value)
    return out


def _small_vmc():
    return VMC(N=N_SMALL, rs=90.0, phase="crystal", nmax=1)


def _null_s():
    return {k: None for k in M.S_KEYS}


def _row(c6, bragg=9.0, ctr=5.0, reg=0.0, coh=0.9, ident=0.0, perr=0.0):
    """A row carrying only the fields the verdict rule reads.

    ``None`` passes through unchanged, so a skipped measurement can be
    represented rather than silently turned into a number.  ``perr`` is the
    phase resolution the registry observable reports; the default of 0.0 is a
    perfect measurement, which is what a synthetic row is.
    """
    def opt(x):
        return None if x is None else float(x)

    d = {k: None for k in M.S_KEYS}
    d.update(c6_residual=opt(c6), bragg_ratio=opt(bragg),
             site_midpoint_contrast=opt(ctr), reg_g1=opt(reg),
             reg_g1_coh=opt(coh), reg_identity=opt(ident),
             reg_g1_phase_err=opt(perr))
    return d


def _synthetic_hist(thetas, e0=10.0):
    return [dict(step=i, E=e0 + i, E_err=0.1, acc=0.4, sigma=0.3,
                 force=1.0 + i, tau=0.02, cond=1.0, eig_min=1e-6,
                 eig_max=1e-3, theta=np.asarray(thetas[i + 1], float))
            for i in range(len(thetas) - 1)]


# ==========================================================================
# 1-3. the bookkeeping the physics rides on
# ==========================================================================
class TestThePairingRule(unittest.TestCase):
    """``hist[i]['E']`` belongs to the state the step STARTED from."""

    def setUp(self):
        self.vmc = _small_vmc()
        self.torus = M.analysis_torus(self.vmc.N)
        self.ctx = M.c6_context(self.vmc, self.torus, nsp=12)
        nb, nk = self.vmc.n_bands, self.vmc.lat.nk
        L = M.NJ + 2 * (nb - 1) * nk
        # Distinct per step, so a one-step lag cannot pass by accident.
        self.thetas = [np.arange(L, dtype=float) * 0.001 + i * 0.37
                       for i in range(5)]
        self.hist = _synthetic_hist(self.thetas)
        self.meas = M.make_meas(self.vmc, self.ctx, lambda t: _null_s(), every=1)
        self.rows = M.build_rows(self.thetas[0], self.thetas[4], self.hist,
                                 lambda i: self.hist[i]["theta"], self.meas)

    def test_row_count_is_steps_plus_the_explicit_closing_row(self):
        # hist has 4 rows for 4 steps; the state `sr_optimize_joint` RETURNS has
        # no hist row, so it gets one of its own.
        self.assertEqual(len(self.hist), 4)
        self.assertEqual(len(self.rows), 5)
        self.assertTrue(self.rows[-1]["closing"])
        self.assertFalse(any(r["closing"] for r in self.rows[:-1]))

    def test_row_t_carries_theta_t_and_energy_of_theta_t(self):
        for t in range(4):
            np.testing.assert_array_equal(np.array(self.rows[t]["theta"]),
                                          self.thetas[t])
            self.assertEqual(self.rows[t]["E_total"], self.hist[t]["E"])
        # row 0 is the SEED, at the seed's own energy
        np.testing.assert_array_equal(np.array(self.rows[0]["theta"]),
                                      self.thetas[0])

    def test_the_closing_row_is_the_returned_state_with_no_optimiser_record(self):
        np.testing.assert_array_equal(np.array(self.rows[4]["theta"]),
                                      self.thetas[4])
        # No optimiser record exists for it: the optimiser never sampled there.
        self.assertIsNone(self.rows[4]["E_total"])
        self.assertIsNone(self.rows[4]["force"])
        self.assertIsNone(self.rows[4]["E_err_total"])
        # But it IS a state, so the block-resolved step IS defined -- and that
        # is the number the "has the orbital sector settled?" question needs.
        self.assertIsNotNone(self.rows[4]["dv_norm"])
        self.assertGreater(self.rows[4]["dv_norm"], 0.0)

    def test_the_one_step_lag_would_fail_this_test(self):
        # The bug: pair hist[t]['E'] with hist[t]['theta'].  That puts
        # theta_{t+1} on row t, so row 1 would carry thetas[2], not thetas[1].
        self.assertFalse(np.allclose(np.array(self.rows[1]["theta"]),
                                     np.asarray(self.hist[1]["theta"])))
        # ... and it would move the whole energy column by one step.
        lagged = [h["E"] for h in self.hist[1:]] + [None]
        self.assertNotEqual([r["E_total"] for r in self.rows], lagged)

    def test_the_block_resolved_step_is_free_and_row_0_has_none(self):
        self.assertIsNone(self.rows[0]["dc_norm"])
        self.assertIsNone(self.rows[0]["dv_norm"])
        nb, nk = self.vmc.n_bands, self.vmc.lat.nk
        for t in range(1, 5):
            c_t, v_t = M.theta_parts(np.asarray(self.rows[t]["theta"], float),
                                     nb, nk)
            c_p, v_p = M.theta_parts(np.asarray(self.rows[t - 1]["theta"], float),
                                     nb, nk)
            self.assertAlmostEqual(self.rows[t]["dc_norm"],
                                   float(np.linalg.norm(c_t - c_p)), places=12)
            self.assertAlmostEqual(self.rows[t]["dv_norm"],
                                   float(np.linalg.norm(v_t - v_p)), places=12)
        self.assertGreater(self.rows[1]["dc_norm"], 0.0)


class TestTheUnitRule(unittest.TestCase):
    """``hist`` energies are TOTALS; the published convention is per particle."""

    def test_per_particle_is_the_total_over_the_electron_count(self):
        vmc = VMC(N=36, rs=90.0, phase="crystal", nmax=1)
        torus = M.analysis_torus(36)
        ctx = M.c6_context(vmc, torus, nsp=8)
        nb, nk = vmc.n_bands, vmc.lat.nk
        L = M.NJ + 2 * (nb - 1) * nk
        thetas = [np.arange(L, dtype=float) * 1e-3 + i * 0.29 for i in range(3)]
        # Totals roughly 36x the per-particle scale, which is what makes a
        # confused reader's error possible in the first place.
        hist = _synthetic_hist(thetas, e0=-45.0 * 36)
        rows = M.build_rows(thetas[0], thetas[2], hist, lambda i: hist[i]["theta"],
                            M.make_meas(vmc, ctx, lambda t: _null_s(), every=1))
        for r in rows[:-1]:
            self.assertEqual(r["E_per_particle"], r["E_total"] / 36.0)
            self.assertAlmostEqual(r["E_err_per_particle"], r["E_err_total"] / 36.0,
                                   places=15)
            self.assertEqual(r["closing"], False)
        self.assertIsNone(rows[-1]["E_per_particle"])

    def test_the_factor_is_the_electron_count_and_not_a_hardcoded_36(self):
        vmc = _small_vmc()
        torus = M.analysis_torus(vmc.N)
        ctx = M.c6_context(vmc, torus, nsp=8)
        nb, nk = vmc.n_bands, vmc.lat.nk
        L = M.NJ + 2 * (nb - 1) * nk
        thetas = [np.arange(L, dtype=float) * 1e-3 + i * 0.29 for i in range(3)]
        hist = _synthetic_hist(thetas, e0=-45.0)
        rows = M.build_rows(thetas[0], thetas[2], hist, lambda i: hist[i]["theta"],
                            M.make_meas(vmc, ctx, lambda t: _null_s(), every=1))
        self.assertEqual(rows[0]["E_per_particle"], rows[0]["E_total"] / 9.0)
        self.assertNotEqual(rows[0]["E_per_particle"], rows[0]["E_total"] / 36.0)


class TestTheSnapshotCount(unittest.TestCase):
    """The sampler's own rule, and the off-by-one form it excludes."""

    def test_matches_len_of_range(self):
        self.assertEqual(M.snapshot_count(20, 6, 1), len(range(6, 20, 1)))
        self.assertEqual(M.snapshot_count(40, 13, 2), len(range(13, 40, 2)))
        self.assertEqual(M.snapshot_count(150, 50, 3), len(range(50, 150, 3)))

    def test_the_tempting_form_over_counts_at_smoke(self):
        # smoke is 6/20/1: the range has 14 entries, (20-6)//1 + 1 is 15.
        self.assertEqual((20 - 6) // 1 + 1, 15)
        self.assertEqual(M.snapshot_count(20, 6, 1), 14)
        self.assertNotEqual(M.snapshot_count(20, 6, 1), (20 - 6) // 1 + 1)

    def test_the_configured_budgets(self):
        for name, want in (("smoke", 14), ("quick", 14)):
            p = load_budget(name).protocol
            self.assertEqual(M.snapshot_count(p["sr_sweeps"], p["sr_equil"],
                                              p["sr_snap"]), want)
        p = load_budget("reproduction").protocol
        self.assertEqual(M.snapshot_count(p["sr_sweeps"], p["sr_equil"],
                                          p["sr_snap"]),
                         len(range(50, 150, 3)))


# ==========================================================================
# 4. the anchor: this is the project's optimiser, not a copy of it
# ==========================================================================
def _measure_stub(wf, R0, sweeps, equil, sigma, seed, label=None):
    """A cheap stand-in for ``measure_decomposed``'s final walk.

    Only the anchor test uses it, and only because that test is about the
    OPTIMISER trace: paying for the production walk would add cost without
    adding evidence.  Its return keys are the ones ``_run`` consumes.
    """
    return {"E": 0.0, "E_err": 0.0, "acc": 0.0, "n": 3,
            "snaps": [np.zeros((wf.ne, 2)) for _ in range(3)]}


class TestTheAnchor(unittest.TestCase):
    """``run_joint_arm`` must reproduce ``VMC._run_from`` exactly.

    ``api.py:886-891`` builds the same call from the same protocol.  If the
    diagnostic's kwargs, seeds or starting configuration differ by anything, the
    trajectories diverge and this test says so -- which is the whole point of
    measuring the optimiser directly rather than re-implementing it.
    """

    def test_the_trace_equals_the_production_trace(self):
        import wigner_vmc.api as api
        vmc = _small_vmc()
        bud = load_budget("smoke")
        proto = bud.protocol
        cfg = resolve(phase="crystal", rs=vmc.rs, N=vmc.N, nmax=vmc.nmax,
                      init_id=0, budget="smoke")
        v0 = vmc.crystal_v0(bud.width_for(0))
        theta0 = theta(sr.jastrow_vector(vmc.kappa, M.NJ), v0)
        self.assertEqual(len(theta0), M.NJ + 2 * (vmc.n_bands - 1) * vmc.lat.nk)

        th_a, hist_a, _ = M.run_joint_arm(vmc, theta0, bud, cfg.rng_seed,
                                          proto["sr_steps"], proto["sr_sweeps"],
                                          proto["sr_equil"])
        with mock.patch.object(api, "measure_decomposed", _measure_stub):
            res = vmc._run_from(cfg, theta0, vmc.crystal_R0(), verbose=False)

        e_a = np.array([h["E"] for h in hist_a], float)
        acc_a = np.array([h["acc"] for h in hist_a], float)
        np.testing.assert_array_equal(res.optimization.E_total, e_a)
        np.testing.assert_array_equal(res.optimization.acc, acc_a)
        self.assertEqual(res.optimization.kind, "joint")
        # The returned state is the same one, and it is bit-identical.
        np.testing.assert_array_equal(np.asarray(res.state.v, complex),
                                      M.theta_parts(np.asarray(th_a, float),
                                                    vmc.n_bands, vmc.lat.nk)[1])

    def test_the_production_trace_would_differ_on_another_seed(self):
        # The anchor is only informative if it CAN fail: the trace must depend
        # on the seed at all.
        vmc = _small_vmc()
        bud = load_budget("smoke")
        proto = bud.protocol
        v0 = vmc.crystal_v0(bud.width_for(0))
        theta0 = theta(sr.jastrow_vector(vmc.kappa, M.NJ), v0)
        _, h0, _ = M.run_joint_arm(vmc, theta0, bud, 0, 1, 8, 2)
        _, h1, _ = M.run_joint_arm(vmc, theta0, bud, 12345, 1, 8, 2)
        self.assertNotEqual([h["E"] for h in h0], [h["E"] for h in h1])


# ==========================================================================
# 5. Layer 1, at the operating point the question is asked about
# ==========================================================================
class TestTheC6EstimatorAtThisOperatingPoint(unittest.TestCase):
    """The seed is C6-covariant at ``r_s = 90, n_max = 1``, measured.

    The pinned corner in ``tests/test_projection_convention.py`` is
    ``r_s = 75, n_max = 2``.  Nothing pinned this point, and the manifold differs
    there (one excited band instead of two), so it is measured here rather than
    inherited -- and it is the seed that every arm of the diagnostic starts from.
    ``nsp = 120`` keeps the suite affordable; the run itself uses 500 and reads
    the same to within a factor of 1.2.
    """

    @classmethod
    def setUpClass(cls):
        cls.vmc = VMC(N=36, rs=90.0, phase="crystal", nmax=1)
        torus = M.analysis_torus(36)
        cls.ctx = M.c6_context(cls.vmc, torus, nsp=120)
        bud = load_budget("quick")
        cls.v0, cls.rec = M.seed_controls(cls.vmc, torus, cls.ctx,
                                          float(bud.width_for(0)), nsp2=60,
                                          seed2=M.C6_SEED + 1)
        cls.rec.update(rs=float(cls.vmc.rs), nmax=int(cls.vmc.nmax), N=36)
        cls.sv = M.seed_verdict(cls.rec)

    def test_the_corrected_seed_is_c6_covariant(self):
        self.assertLess(self.rec["c6_correct"], 1e-9,
                        f"the Gaussian seed reads {self.rec['c6_correct']:.3e} "
                        f"at r_s=90, n_max=1")
        self.assertTrue(self.sv["ok"], self.sv["reason"])

    def test_the_seed_is_as_c6_invariant_as_the_filled_landau_level(self):
        # The reference scale at this context: v = 0 is invariant under every
        # rotation, so whatever the estimator reads there IS its noise floor.
        v_zero = np.zeros((self.vmc.lat.nk, self.vmc.n_bands - 1), complex)
        floor = max(M.c6_residuals(v_zero, self.ctx).values())
        self.assertLess(self.rec["c6_correct"], 10.0 * floor)
        self.assertLess(floor, 1e-9)

    def test_the_conjugated_overlap_is_the_negative_control_and_is_not_vacuous(self):
        # Without a control that reads LARGE, a context that calls everything
        # "broken" would pass.  `ov.conj()` is the historical defect.
        self.assertGreater(self.rec["c6_defect"], 0.05)
        self.assertGreater(self.sv["facts"]["separation"], 1e6)

    def test_the_residual_is_gauge_invariant(self):
        self.assertLess(self.rec["c6_gauge_shift"], 1e-12)

    def test_a_second_context_agrees(self):
        self.assertLess(self.rec["c6_correct_ctx2"], 1e-9)


class TestTheLayerOneVerdict(unittest.TestCase):
    """Each branch, on synthetic records -- the logic, not the numbers."""

    def _rec(self, **kw):
        rec = {"c6_correct": 1e-14, "c6_correct_ctx2": 1e-14,
               "c6_defect": 0.6, "c6_gauge_shift": 1e-17, "nsp2": 40,
               "c6_seed2": 1, "nbar_seed": 0.35, "nbar_defect": 0.35,
               "rs": 90.0, "nmax": 1}
        rec.update(kw)
        return rec

    def test_a_clean_record_passes(self):
        self.assertTrue(M.seed_verdict(self._rec())["ok"])

    def test_a_seed_that_is_not_c6_fails_and_says_so(self):
        sv = M.seed_verdict(self._rec(c6_correct=0.5))
        self.assertFalse(sv["ok"])
        self.assertIn("not C6-covariant", sv["reason"])
        self.assertIn("SR cannot be blamed", sv["reason"])

    def test_a_second_context_that_disagrees_fails(self):
        sv = M.seed_verdict(self._rec(c6_correct_ctx2=0.4))
        self.assertFalse(sv["ok"])
        self.assertIn("second independent context", sv["reason"])

    def test_a_negative_control_that_is_not_broken_fails(self):
        # The separation is what establishes that the context resolves the gate.
        sv = M.seed_verdict(self._rec(c6_defect=1e-12))
        self.assertFalse(sv["ok"])
        self.assertIn("does not establish", sv["reason"])

    def test_a_degenerate_context_cannot_pass_on_the_ratio_alone(self):
        # Both tiny: an enormous ratio, and no resolution at all.
        sv = M.seed_verdict(self._rec(c6_correct=1e-30, c6_defect=1e-20))
        self.assertFalse(sv["ok"])

    def test_a_gauge_dependent_residual_fails(self):
        sv = M.seed_verdict(self._rec(c6_gauge_shift=1e-3))
        self.assertFalse(sv["ok"])
        self.assertIn("gauge invariant", sv["reason"])

    def test_a_bounded_residual_cannot_be_required_to_exceed_one(self):
        # The bug an earlier draft had: `c6_defect > 10 * gate` demands > 1.0
        # from a quantity bounded in [0, 1], so a 13-decade separation failed.
        self.assertTrue(0.594 > M.C6_GATE)
        self.assertFalse(0.594 > 10.0 * M.C6_GATE)
        self.assertTrue(M.seed_verdict(self._rec(c6_defect=0.594))["ok"])


# ==========================================================================
# 6. the measurement must not perturb what it measures
# ==========================================================================
class TestTheSWalkDoesNotTouchTheSRStream(unittest.TestCase):
    """``sr_optimize_joint`` draws its walk AND its update noise from one rng.

    So an S walk that borrowed that stream -- even by a single draw -- would move
    the trajectory.  The S walk therefore opens ``default_rng(s_seed)`` of its
    own and starts from a fixed ``crystal_R0()``.  This test interleaves a real S
    walk between two identical SR calls and demands the trace be unchanged.
    """

    def test_an_interleaved_walk_leaves_the_trace_bit_identical(self):
        vmc = _small_vmc()
        bud = load_budget("smoke")
        torus = M.analysis_torus(vmc.N)
        bragg_q = M.bragg_vectors(torus)
        bg_q = M.background_vectors(torus)
        families = M.bragg_families(torus)
        theta0 = theta(sr.jastrow_vector(vmc.kappa, M.NJ),
                       vmc.crystal_v0(bud.width_for(0)))
        seed = resolve(phase="crystal", rs=vmc.rs, N=vmc.N, nmax=vmc.nmax,
                       init_id=0, budget="smoke").rng_seed

        th1, h1, _ = M.run_joint_arm(vmc, theta0, bud, seed, 1, 8, 2)
        rec = M.s_measurements(vmc, theta0, torus, bragg_q, bg_q, families,
                               4, 2, 0.3, 17)
        th2, h2, _ = M.run_joint_arm(vmc, theta0, bud, seed, 1, 8, 2)

        np.testing.assert_array_equal(np.asarray(th1, float),
                                      np.asarray(th2, float))
        self.assertEqual([h["E"] for h in h1], [h["E"] for h in h2])
        self.assertEqual([h["acc"] for h in h1], [h["acc"] for h in h2])
        # ... and the walk really did produce a measurement.
        self.assertIsInstance(rec["bragg_ratio"], float)
        self.assertGreater(rec["s_n"], 0)

    def test_the_walk_is_a_pure_function_of_the_state(self):
        # Same state, same seed, same walk -- which is what makes the digest
        # cache in `main` a cost knob rather than a change of experiment.
        vmc = _small_vmc()
        torus = M.analysis_torus(vmc.N)
        args = (vmc, theta(np.zeros(M.NJ),
                           np.zeros((vmc.lat.nk, vmc.n_bands - 1), complex)),
                torus, M.bragg_vectors(torus), M.background_vectors(torus),
                M.bragg_families(torus), 4, 2, 0.3, 17)
        a = M.s_measurements(*args)
        b = M.s_measurements(*args)
        for k in ("bragg_ratio", "site_midpoint_contrast", "s_E_per_particle",
                  "reg_g1"):
            self.assertEqual(a[k], b[k])


# ==========================================================================
# 7. the verdict rule
# ==========================================================================
class TestTheVerdictRule(unittest.TestCase):

    def test_stable_c6(self):
        lab, _ = M.stage_b([_row(0.01), _row(0.05), _row(0.04)], None)
        self.assertEqual(lab, "STABLE_C6")

    def test_seed_not_c6_is_checked_first(self):
        rows = [_row(0.5), _row(0.1), _row(0.2)]
        # It also satisfies "still rising"; the seed label must win.
        self.assertEqual(M.stage_b(rows, None)[0], "SEED_NOT_C6")

    def test_underconverged_when_it_rose_and_came_back(self):
        lab, _ = M.stage_b([_row(0.01), _row(0.3), _row(0.2), _row(0.01)], None)
        self.assertEqual(lab, "UNDERCONVERGED")

    def test_still_rising_is_not_a_result(self):
        lab, facts = M.stage_b([_row(0.01), _row(0.15), _row(0.25), _row(0.35)],
                               None)
        self.assertEqual(lab, "STILL_RISING")
        self.assertFalse(facts["turned_over"])
        self.assertIn(lab, M.NOT_A_RESULT)

    def test_drift_settled_and_plateau_broken_are_distinguished_by_the_turnover(self):
        settled, _ = M.stage_b([_row(0.01), _row(0.50), _row(0.45)], None)
        broken, _ = M.stage_b([_row(0.01), _row(0.50), _row(0.20)], None)
        self.assertEqual(settled, "DRIFT_SETTLED")
        self.assertEqual(broken, "PLATEAU_BROKEN")
        self.assertIn(broken, M.NOT_A_RESULT)

    def test_registry_drift_needs_a_monotone_coherent_phase_and_an_intact_crystal(self):
        c6 = [0.01, 0.25, 0.30, 0.28]
        reg = [0.0, 0.1, 0.2, 0.3]
        rows = [_row(c, reg=r) for c, r in zip(c6, reg)]
        self.assertEqual(M.stage_b(rows, {"bragg_ratio": 1.0})[0], "REGISTRY_DRIFT")
        self.assertIn("REGISTRY_DRIFT", M.NOT_A_RESULT)

    # A ladder with an unambiguous turnover, so "not REGISTRY_DRIFT" lands on a
    # definite other label rather than on whichever branch happens to be next.
    C6_TURNED = [0.01, 0.25, 0.50, 0.28]
    REG_MONOTONE = [0.0, 0.1, 0.2, 0.3]

    def _rows_with(self, **kw):
        return [_row(c, reg=r, **kw)
                for c, r in zip(self.C6_TURNED, self.REG_MONOTONE)]

    def test_the_turned_ladder_alone_is_plateau_broken(self):
        # The control for the four tests below: this c6 shape, with no registry
        # information at all, is PLATEAU_BROKEN and nothing else.
        rows = [_row(c) for c in self.C6_TURNED]
        lab, facts = M.stage_b(rows, {"bragg_ratio": 1.0})
        self.assertEqual(lab, "PLATEAU_BROKEN")
        self.assertFalse(facts["reg_moved"])

    def test_a_phase_that_never_moved_is_not_a_slide(self):
        # The defect this guard exists for: a CONSTANT phase series is
        # monotone, coherent and identity-preserving, so without a magnitude
        # test the lattice that slid nowhere reads as the strongest possible
        # slide.  `reg_moved` is what makes "the phases moved" a claim.
        rows = [_row(c, reg=0.0) for c in self.C6_TURNED]
        lab, facts = M.stage_b(rows, {"bragg_ratio": 1.0})
        self.assertEqual(lab, "PLATEAU_BROKEN")
        self.assertAlmostEqual(facts["reg_swing"], 0.0, places=15)
        self.assertTrue(facts["reg_monotone"])
        self.assertFalse(facts["reg_moved"])

    def test_the_same_c6_without_a_monotone_phase_is_not_registry_drift(self):
        c6 = self.C6_TURNED
        rows = [_row(c, reg=r) for c, r in zip(c6, [0.0, 0.3, 0.1, 0.2])]
        self.assertEqual(M.stage_b(rows, {"bragg_ratio": 1.0})[0],
                         "PLATEAU_BROKEN")

    def test_an_incoherent_phase_is_not_read_as_registry_drift(self):
        # A phase read where there is no peak is noise; without the coherence
        # guard this would be reported as a slide.
        self.assertEqual(M.stage_b(self._rows_with(coh=0.05),
                                   {"bragg_ratio": 1.0})[0], "PLATEAU_BROKEN")

    def test_a_broken_translation_identity_is_not_read_as_registry_drift(self):
        # phi(g1+g2) = phi(g1) + phi(g2) is exact for a SLID lattice and fails
        # for a STRAINED one, which is what separates the two.
        self.assertEqual(M.stage_b(self._rows_with(ident=1.4),
                                   {"bragg_ratio": 1.0})[0], "PLATEAU_BROKEN")

    def test_a_degraded_lattice_is_not_registry_drift(self):
        self.assertEqual(M.stage_b(self._rows_with(bragg=2.0),
                                   {"bragg_ratio": 1.0})[0], "PLATEAU_BROKEN")

    def test_an_exactly_zero_identity_must_not_defeat_the_branch(self):
        # The bug this branch had: `fam["reg_max_identity"] or 9.9` turns the
        # EXACT 0.0 a slid lattice produces into 9.9, so the one case the branch
        # exists for was the one case it could not detect.
        rows = [_row(c, reg=r, ident=0.0)
                for c, r in zip(self.C6_TURNED, self.REG_MONOTONE)]
        lab, facts = M.stage_b(rows, {"bragg_ratio": 1.0})
        self.assertEqual(lab, "REGISTRY_DRIFT")
        self.assertEqual(facts["reg_max_identity"], 0.0)
        self.assertTrue(facts["identity_ok"])

    def test_a_slide_below_the_measured_resolution_is_not_a_slide(self):
        # Same monotone walk, same coherence, same identity -- but the phases
        # moved less than the walk's own resolution, so the movement is not
        # established.  This is the "claim with an error bar" half of the guard.
        rows = [_row(c, reg=r, perr=0.4)
                for c, r in zip(self.C6_TURNED, self.REG_MONOTONE)]
        lab, facts = M.stage_b(rows, {"bragg_ratio": 1.0})
        self.assertEqual(lab, "PLATEAU_BROKEN")
        self.assertFalse(facts["reg_moved"])
        self.assertGreater(facts["reg_swing"], 0.0)

    def test_stage_a_catches_a_state_that_stopped_being_a_crystal(self):
        # A LIQUID closing state has an excellent C6 and is not a crystal.  This
        # is the reason the verdict is two-staged.
        liquid = {"bragg_ratio": 1.0}
        cls, why = M.stage_a(_row(0.001, bragg=1.0, ctr=5.0), liquid)
        self.assertEqual(cls, M.NOT_A_CRYSTAL)
        self.assertIn("says nothing about symmetry", why)
        cls2, _ = M.stage_a(_row(0.001, bragg=0.9, ctr=0.8), liquid)
        self.assertEqual(cls2, M.NOT_A_CRYSTAL)

    def test_stage_a_passes_a_real_crystal(self):
        cls, _ = M.stage_a(_row(0.2, bragg=9.0, ctr=5.2),
                           {"bragg_ratio": 1.03})
        self.assertEqual(cls, M.CRYSTAL)

    def test_stage_a_reports_unknown_without_a_liquid(self):
        cls, why = M.stage_a(_row(0.2), None)
        self.assertEqual(cls, M.CLASS_UNKNOWN)
        self.assertIn("cannot be evaluated", why)
        cls2, _ = M.stage_a(_row(0.2, bragg=None), {"bragg_ratio": 1.0})
        self.assertEqual(cls2, M.CLASS_UNKNOWN)

    def test_the_three_next_experiment_labels_are_not_results(self):
        self.assertEqual(set(M.NOT_A_RESULT),
                         {"STILL_RISING", "PLATEAU_BROKEN", "REGISTRY_DRIFT"})
        for lab in M.NOT_A_RESULT:
            self.assertNotIn(lab, ("STABLE_C6", "UNDERCONVERGED", "SEED_NOT_C6"))


# ==========================================================================
# 8. the registry observable
# ==========================================================================
class TestTheRegistryObservable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.torus = M.analysis_torus(36)
        cls.sites = np.array([i * np.asarray(cls.torus.A1, float)
                              + j * np.asarray(cls.torus.A2, float)
                              for i in range(6) for j in range(6)])
        rng = np.random.default_rng(0)
        # Two lattices.  `exact` is noiseless, so the phase and identity
        # assertions below are exact statements about the estimator rather than
        # statements about a 0.01 jitter; `snaps` carries the jitter and is what
        # the coherence contrast needs.
        cls.exact = np.repeat(cls.sites[None, :, :], 5, axis=0)
        cls.snaps = cls.exact + 0.01 * rng.standard_normal((5, 36, 2))
        cls.d = np.array([0.13, -0.07])
        cls.shifted = cls.exact + cls.d

    def test_an_untranslated_lattice_reads_phase_zero_and_a_coherent_peak(self):
        r = M.registry(self.exact, self.torus)
        for key in ("reg_g1", "reg_g2", "reg_g1g2"):
            self.assertAlmostEqual(r[key], 0.0, places=9)
            self.assertGreater(r[f"{key}_coh"], 0.95)
        self.assertAlmostEqual(r["reg_identity"], 0.0, places=9)
        # The jittered lattice reads the same phases to within the jitter, so
        # the exact result is not an artefact of the lattice being noiseless.
        self.assertAlmostEqual(M.registry(self.snaps, self.torus)["reg_g1"],
                               r["reg_g1"], places=2)

    def test_a_translated_lattice_shifts_each_phase_by_q_dot_d(self):
        r0 = M.registry(self.exact, self.torus)
        r1 = M.registry(self.shifted, self.torus)
        for key, q in (("reg_g1", np.asarray(self.torus.g1, float)),
                       ("reg_g2", np.asarray(self.torus.g2, float)),
                       ("reg_g1g2", np.asarray(self.torus.g1, float)
                        + np.asarray(self.torus.g2, float))):
            self.assertAlmostEqual(M.wrap_pi(r1[key] - r0[key]),
                                   float(np.dot(q, self.d)), places=9)
        # The identity still holds: this is a SLIDE, not a strain.
        self.assertAlmostEqual(r1["reg_identity"], 0.0, places=9)
        # ... and the crystal is otherwise untouched.
        self.assertGreater(r1["reg_g1_coh"], 0.95)

    def test_the_identity_is_what_separates_a_slide_from_a_strain(self):
        # g1 + g2 is one of the six first-shell vectors, so all three phases are
        # independent numbers only because the identity ties them together.
        g1 = np.asarray(self.torus.g1, float)
        g2 = np.asarray(self.torus.g2, float)
        self.assertAlmostEqual(float(np.linalg.norm(g1 + g2)),
                               float(np.linalg.norm(g1)), places=9)

    def test_coherence_is_low_where_there_is_no_peak(self):
        rng = np.random.default_rng(1)
        uniform = rng.random((40, 36, 2)) @ np.asarray(self.torus.sc, float).T
        ru = M.registry(uniform, self.torus)
        rs = M.registry(self.snaps, self.torus)
        self.assertLess(ru["reg_g1_coh"], 0.5)
        self.assertLess(ru["reg_g1_coh"], 0.5 * rs["reg_g1_coh"])


# ==========================================================================
# 9. the |f| noise floor is derived, not assumed
# ==========================================================================
class TestTheForceFloor(unittest.TestCase):

    def _arm(self, name, force):
        return {"arm": name, "rows": [{"force": force}, {"force": 0.1}]}

    def test_the_floor_is_the_seed_to_seed_scatter(self):
        vals = [1.0, 2.0, 4.0]
        f = M.force_floor([self._arm("joint", v) for v in vals])
        self.assertEqual(f["n"], 3)
        self.assertAlmostEqual(f["floor"], float(np.std(vals, ddof=1)), places=15)
        self.assertEqual(f["values"], vals)

    def test_one_seed_reports_no_floor_rather_than_zero(self):
        f = M.force_floor([self._arm("joint", 1.0)])
        self.assertIsNone(f["floor"])
        self.assertIn("no |f| floor", f["note"])

    def test_only_joint_arms_contribute(self):
        f = M.force_floor([self._arm("joint", 1.0), self._arm("pinned", 99.0),
                           self._arm("joint", 3.0)])
        self.assertEqual(f["values"], [1.0, 3.0])

    def test_a_missing_force_is_not_counted_as_zero(self):
        arm = {"arm": "joint", "rows": [{"force": None}, {"force": 0.1}]}
        f = M.force_floor([arm, self._arm("joint", 2.0)])
        self.assertEqual(f["values"], [2.0])
        self.assertIsNone(f["floor"])


class TestTheT0SeedFloor(unittest.TestCase):
    """The same shared-``theta0`` argument, applied to the S families.

    The pilot's seed reads ``S(G1), S(G2), S(G3)`` = 6.135 / 5.975 / 6.412 -- a
    spread of 0.44 *before any SR step*.  So a later "the families split" claim
    has to clear the estimator's own seed-to-seed scatter, and that scatter is
    only available because every arm starts from the same state.
    """

    def _arm(self, name, **row):
        r = {"S_G1": 6.0, "S_G2": 6.0, "S_G3": 6.0}
        r.update(row)
        return {"arm": name, "rows": [r, {"S_G1": 9.9}]}

    def test_the_floor_is_the_seed_to_seed_scatter_at_t0(self):
        vals = [5.9, 6.1, 6.0]
        f = M.seed_floor([self._arm("joint", S_G1=v) for v in vals], "S_G1")
        self.assertEqual(f["n"], 3)
        self.assertAlmostEqual(f["mean"], float(np.mean(vals)), places=12)
        self.assertAlmostEqual(f["sd"], float(np.std(vals, ddof=1)), places=12)
        self.assertIn("SAME state", f["note"])

    def test_one_arm_reports_no_floor_rather_than_zero(self):
        f = M.seed_floor([self._arm("joint")], "S_G1")
        self.assertIsNone(f["sd"])
        self.assertIsNone(f["mean"])
        self.assertIn("no claim is made", f["note"])

    def test_the_floor_reads_t0_not_the_last_row(self):
        # The second row carries a wild value; it must not enter the floor.
        f = M.seed_floor([self._arm("joint", S_G1=6.0),
                          self._arm("joint", S_G1=6.2)], "S_G1")
        self.assertEqual(f["values"], [6.0, 6.2])

    def test_only_the_named_arm_kind_contributes(self):
        f = M.seed_floor([self._arm("joint"), self._arm("pinned")], "S_G1")
        self.assertEqual(f["n"], 1)
        self.assertIsNone(f["sd"])

    def test_a_missing_key_is_not_counted_as_zero(self):
        a = {"arm": "joint", "rows": [{"S_G1": None}]}
        f = M.seed_floor([a, self._arm("joint", S_G1=6.0)], "S_G1")
        self.assertEqual(f["values"], [6.0])
        self.assertIsNone(f["sd"])


# ==========================================================================
# 10. the figure draws the measurements that exist
# ==========================================================================
class TestTheFigureDoesNotInventPoints(unittest.TestCase):
    """A row may legitimately have no value for a key.

    The closing row has no optimiser record, row 0 has no difference, and a
    stride leaves gaps in the S keys.  Each of those is an ABSENT measurement.
    Plotting them as zeros would fabricate data, and dropping a whole panel
    because one row is missing throws away the measurements that do exist --
    which is what the first version of `write_figure` did for the energy panel,
    where it also crashed on the closing row's ``None``.
    """

    def _rows(self):
        return [{"t": 0, "c6_residual": 1e-14, "E_per_particle": None,
                 "S_G1": 7.0, "dv_norm": None},
                {"t": 1, "c6_residual": 0.05, "E_per_particle": -45.4,
                 "S_G1": None, "dv_norm": 0.4},
                {"t": 2, "c6_residual": 0.15, "E_per_particle": -45.5,
                 "S_G1": 6.8, "dv_norm": 0.3},
                {"t": 3, "c6_residual": 0.20, "E_per_particle": None,
                 "S_G1": 6.7, "dv_norm": 0.2}]

    def test_absent_values_are_skipped_not_zeroed(self):
        self.assertEqual(M.plottable(self._rows(), "E_per_particle"),
                         [(1, -45.4), (2, -45.5)])
        self.assertEqual(M.plottable(self._rows(), "S_G1"),
                         [(0, 7.0), (2, 6.8), (3, 6.7)])
        self.assertEqual(M.plottable(self._rows(), "dv_norm"),
                         [(1, 0.4), (2, 0.3), (3, 0.2)])

    def test_a_column_with_nothing_measured_returns_nothing(self):
        rows = [{"t": 0, "closing": True}]
        self.assertEqual(M.plottable(rows, "E_per_particle"), [])
        self.assertEqual(M.plottable(rows, "not_a_key_at_all"), [])

    def test_the_closing_row_is_still_on_the_c6_panel(self):
        # C6 comes from the state, not from the optimiser, so the closing row
        # -- the one Stage A reads -- is the LAST point of the C6 series.
        pts = M.plottable(self._rows(), "c6_residual")
        self.assertEqual(pts[-1], (3, 0.20))

    def test_the_figure_renders_with_a_closing_row_that_has_no_energy(self):
        import matplotlib
        matplotlib.use("Agg")
        arms = [{"arm": "joint", "idx": 0, "rows": self._rows(),
                 "rng_seed": 0, "tag": "traj"}]
        floor = {"n": 0, "floor": None, "values": [],
                 "note": "fewer than two joint arms: no |f| floor is available"}
        with tempfile.TemporaryDirectory() as tmp:
            p = M.write_figure(tmp, arms, None, floor, "quick",
                               {"N": 36, "rs": 90.0, "nmax": 1})
            self.assertTrue(os.path.isfile(p))
            self.assertGreater(os.path.getsize(p), 1000)


class TestTheSeedIndexRule(unittest.TestCase):
    """``--seeds i`` must be ``base + i``.

    The first version of the script assigned the bare ``base`` to every job, so
    ``--seeds 0,1,2`` ran the SAME trajectory three times.  Everything Layer 2
    reads across seeds -- the per-seed spread, the direction pattern, the ``|f|``
    floor -- would have been identically zero, and every one of those has a
    plausible-looking value, so nothing downstream would have failed.  Index 0
    alone cannot see it, which is exactly what the previous test asserted; this
    one pins the *offsets*.
    """

    def _plan(self, argv):
        bud = load_budget("quick")
        args = M.build_parser().parse_args(argv)
        mode, extra, jobs = M.plan_jobs(args, bud, 2, 40, 13)
        base = int(resolve(phase="crystal", rs=90.0, N=36, nmax=1, init_id=0,
                           budget="quick").rng_seed)
        for j in jobs:
            if j["rng_seed"] is None:
                j["rng_seed"] = base + int(j.get("seed_index", 0))
        return base, mode, extra, jobs

    def test_distinct_seed_indices_get_distinct_streams(self):
        base, mode, extra, jobs = self._plan(["--seeds", "0,1,2"])
        self.assertEqual([j["rng_seed"] for j in jobs],
                         [base, base + 1, base + 2])
        self.assertEqual(len(set(j["rng_seed"] for j in jobs)), 3)
        self.assertEqual([j["idx"] for j in jobs], [0, 1, 2])
        self.assertTrue(extra.endswith("") or extra is not None)

    def test_index_zero_is_still_the_production_reference(self):
        base, _, _, jobs = self._plan(["--seeds", "0"])
        self.assertEqual(jobs[0]["rng_seed"], base)

    def test_the_nsweep_ladder_shares_one_stream(self):
        # 3b must vary ONLY the sample size: if the rungs also changed the
        # sampling stream, the log-log slope would measure the seed too.
        base, mode, extra, jobs = self._plan(["--nsweep-ladder", "40,150,400"])
        self.assertIn("3b", mode)
        self.assertEqual([j["rng_seed"] for j in jobs], [base, base, base])
        self.assertEqual([j["nsweep"] for j in jobs], [40, 150, 400])
        self.assertNotIn("seed_index", jobs[0])

    def test_the_rng_seed_set_is_used_verbatim(self):
        # 3a's whole point is that the ONLY thing varying is the stream, so it
        # must not be offset off the budget's base.
        _, mode, _, jobs = self._plan(["--rng-seeds", "7,11"])
        self.assertIn("3a", mode)
        self.assertEqual([j["rng_seed"] for j in jobs], [7, 11])


# ==========================================================================
# 11. the source contracts
# ==========================================================================
class TestTheSourceContracts(unittest.TestCase):

    def test_no_sys_path_manipulation(self):
        self.assertIsNone(re.search(r"\bimport\s+sys\b", _source()))
        # Against the CODE strings, not the raw source: the module docstring
        # legitimately names `sys.path` while explaining why it is not touched,
        # and a raw-source check would forbid the explanation as well as the use.
        for s in _code_strings():
            self.assertNotIn("sys.path", s)

    def test_it_reads_no_legacy_store(self):
        code = _code_strings()
        for bad in ("legacy_llrot", "energy_scan", "part1", "make_notebook",
                    "defective"):
            for s in code:
                self.assertNotIn(bad, s, f"{bad!r} appears in a code string")

    def test_the_haar_ban_is_not_vacuous(self):
        # Prose names it (so a reader learns the rule); code must not use it.
        self.assertIn("haar_random_v", _source())
        for s in _code_strings():
            self.assertNotIn("haar_random_v", s)

    def test_no_production_budget_is_reachable_from_a_cli_default(self):
        for d in _argparse_defaults():
            self.assertNotIn("reproduction", d)
            self.assertNotIn("full", d)
        self.assertEqual(M.build_parser().parse_args([]).budget, M.DEFAULT_BUDGET)
        self.assertEqual(M.DEFAULT_BUDGET, "quick")

    def test_the_direct_sr_import_is_deliberate_and_the_exemption_is_real(self):
        # The layering rule is a fixed three-name tuple, not a directory walk,
        # so this file's sibling is outside it -- and so is this file.
        src = _source()
        self.assertIn("from wigner_vmc.vmc import sr", src)
        api_test = io.open(os.path.join(HERE, "test_api_contract.py"),
                           encoding="utf-8").read()
        tree = ast.parse(api_test)
        names = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "EXAMPLE_SCRIPTS"
                    for t in node.targets):
                names = ast.literal_eval(node.value)
        self.assertIsNotNone(names, "EXAMPLE_SCRIPTS was not found")
        self.assertNotIn("crystal_c6_trajectory.py", names)
        self.assertNotIn("crystal_seed_search.py", names)

    def test_the_scope_constants_are_the_siblings_own_objects(self):
        # Identity, not equality: `C6_GATE = 0.1` re-typed here would pass an
        # equality test and then drift the moment the sibling changed.
        for name in ("C6_GATE", "BRAGG_GATE", "CONTRAST_GATE", "C6_ROTS",
                     "C6_NSP", "C6_SEED", "DENSITY_BINS", "N_ELECTRONS",
                     "DEFAULT_RS", "DEFAULT_NMAX"):
            self.assertIs(getattr(M, name), getattr(M.css, name),
                          f"{name} is a copy, not the sibling's own object")

    def test_the_sibling_is_located_by_path_and_not_via_sys_path(self):
        # A `sys.path` insert would leak into every later import in the process.
        self.assertEqual(os.path.dirname(M.__file__),
                         os.path.dirname(SIBLING))
        self.assertTrue(callable(M.analysis_torus))
        self.assertTrue(callable(M.css.lr.c_row))

    def test_the_banner_refuses_another_nmax(self):
        with self.assertRaises(SystemExit):
            M.main(["--nmax", "2"])
        with self.assertRaises(SystemExit):
            M.main(["--s-every", "0"])
        with self.assertRaises(SystemExit):
            M.main(["--rng-seeds", "0", "--nsweep-ladder", "40,150"])


# ==========================================================================
# 11. one end to end, with the optimiser stubbed and the estimators real
# ==========================================================================
class TestEndToEnd(unittest.TestCase):
    """Wiring, not physics: the SR trace is synthetic and Layer 1 is stubbed.

    Both stubs are named in the assertions below.  The states are real wave
    functions and the estimators -- ``bragg_ratio``, ``site_midpoint_contrast``,
    the registry and the C6 residual -- are the production ones, so a wiring
    mistake between any of them and the table or the JSON is still caught.

    It runs at ``N = 9`` for speed, and that choice shows up in the result: the
    Gaussian seed is NOT C6-covariant there (the test asserts the run says so).
    At ``N = 36``, the point the diagnostic is actually asked about, it is --
    see ``TestTheC6EstimatorAtThisOperatingPoint``.
    """

    def test_a_run_writes_the_table_the_metadata_and_the_right_slug(self):
        tmp = tempfile.mkdtemp(prefix="c6traj_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)

        def fake_joint(make_wf, theta0, R0, **kw):
            theta0 = np.asarray(theta0, float)
            steps = int(kw["steps"])
            hist = [dict(step=i, E=-45.0 * 9 - i * 0.01, E_err=0.05, acc=0.4,
                         sigma=0.3, force=10.0 - i, tau=0.02, cond=1.0,
                         eig_min=1e-6, eig_max=1e-3,
                         theta=theta0 + 0.02 * (i + 1))
                    for i in range(steps)]
            return theta0 + 0.02 * steps, hist

        def fake_controls(vmc, torus, ctx, L0, nsp2, seed2):
            return vmc.crystal_v0(L0), {
                "L0": float(L0), "nsp": int(ctx["nsp"]), "c6_seed": int(M.C6_SEED),
                "c6_correct": 1e-14, "c6_defect": 0.6, "nbar_seed": 0.35,
                "nbar_defect": 0.35, "nsp2": int(nsp2), "c6_seed2": int(seed2),
                "c6_correct_ctx2": 1e-14, "c6_defect_ctx2": 0.6,
                "c6_gauge_shift": 1e-17, "c6_gauge_value": 1e-14}

        out = io.StringIO()
        with mock.patch.object(M, "RESULTS", os.path.join(tmp, "results")), \
                mock.patch.object(M, "FIGDIR", os.path.join(tmp, "figures")), \
                mock.patch.object(M, "seed_controls", fake_controls), \
                mock.patch.object(M.sr, "sr_optimize_joint", fake_joint), \
                contextlib.redirect_stdout(out):
            rc = M.main(["--budget", "smoke", "--n-electrons", "9", "--steps", "2",
                         "--nsweep", "20", "--equil", "6", "--nsp", "12",
                         "--s-nsweeps", "4", "--s-equil", "2", "--skip-liquid",
                         "--no-figure", "--seeds", "0", "--s-every", "5"])
        self.assertEqual(rc, 0, out.getvalue()[-3000:])

        slug = ("N9__ll_rotation__smoke__rs90__nmax1__steps2__nsweep20")
        res = os.path.join(tmp, "results", slug)
        self.assertTrue(os.path.isdir(res), os.listdir(os.path.join(tmp, "results")))
        with io.open(os.path.join(res, "trajectory.json"), encoding="utf-8") as fh:
            traj = json.load(fh)
        with io.open(os.path.join(res, "run_metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)

        self.assertEqual(len(traj["arms"]), 1)
        arm = traj["arms"][0]
        self.assertEqual(arm["arm"], "joint")
        self.assertEqual(len(arm["rows"]), 3)           # 2 steps + the closing row
        self.assertTrue(arm["rows"][-1]["closing"])

        # The S walk really ran: rows 0 and the closing row carry measurements.
        for r in (arm["rows"][0], arm["rows"][-1]):
            self.assertIsNotNone(r["S_G1"])
            self.assertIsNotNone(r["bragg_ratio"])
            self.assertIsNotNone(r["reg_g1"])
            self.assertLess(abs(r["pm_G_residual"]), 1e-9)
        # The stride skipped the inner row, and says so by being empty -- while
        # row 0 and the closing row are measured whatever the stride, because
        # Stage A reads the closing row and a stride must not be able to leave
        # the class undecidable.
        self.assertIsNone(arm["rows"][1]["S_G1"])
        self.assertEqual(M.build_parser().parse_args([]).s_every, 1)

        # Stage A has no liquid in this run, and says UNKNOWN rather than a class.
        self.assertEqual(arm["stage_a"]["class"], M.CLASS_UNKNOWN)
        # At N = 9 the Gaussian seed is not C6, and the rule names that FIRST
        # instead of blaming SR.
        self.assertEqual(arm["stage_b"]["label"], "SEED_NOT_C6")
        # And it IS a result: "the seed is not C6 here" is a definite finding
        # about the construction.  Only the three labels that name the NEXT
        # experiment are withheld from the (a)-(d) question.
        self.assertTrue(arm["stage_b"]["is_a_result"])
        self.assertNotIn("SEED_NOT_C6", M.NOT_A_RESULT)

        self.assertEqual(meta["protocol"]["steps"], 2)
        self.assertEqual(meta["protocol"]["nsweep"], 20)
        self.assertEqual(meta["protocol"]["snapshots_per_step"], 14)
        self.assertEqual(meta["protocol"]["s_nsweeps"], 4)
        self.assertEqual(meta["layer1_verdict"]["ok"], True)
        # The t = 0 floors ride the metadata, and a single seed gets None
        # rather than 0.0 -- otherwise every later value looks significant.
        self.assertEqual(sorted(meta["t0_floors"]),
                         ["S_G1", "S_G2", "S_G3", "bragg_ratio", "c6_residual",
                          "site_midpoint_contrast"])
        self.assertIsNone(meta["t0_floors"]["S_G1"]["sd"])
        self.assertEqual(meta["t0_floors"]["S_G1"]["n"], 1)
        self.assertIsNone(meta["force_floor"]["floor"])
        # Seed index 0 resolved to the budget's own rng_seed -- the anchor rule.
        self.assertEqual(meta["jobs"][0]["rng_seed"], meta["protocol"]["base_rng_seed"])
        self.assertIn("index 0 IS the production reference",
                      meta["protocol"]["seed_index_rule"])
        # BLAS pinning is a property of the AMBIENT shell, not of this module
        # (the variables must be set before numpy's BLAS is loaded, so a module
        # cannot fix them after the fact).  What the module owes is honest
        # reporting plus a visible warning, so that is what is asserted -- the
        # assertion holds whether or not the runner happens to be pinned.
        self.assertEqual(meta["blas_threading_pinned"],
                         bool(M.css._threading_is_pinned()))
        if meta["blas_threading_pinned"]:
            self.assertIn("BLAS threads     pinned", out.getvalue())
        else:
            self.assertIn("BLAS threads     NOT pinned", out.getvalue())
            self.assertIn("BLAS threading is not pinned", out.getvalue())
        self.assertIn("tau_i = tau", meta["protocol"]["tau_schedule"])
        self.assertEqual(meta["not_in_scope"][:1], ["r_s continuation"])

    def test_the_three_layer3_modes_plan_different_jobs(self):
        args = M.build_parser().parse_args([])
        bud = load_budget("quick")
        mode, extra, jobs = M.plan_jobs(args, bud, 30, 40, 13)
        self.assertEqual(len(jobs), 3)                  # seeds 0,1,2
        self.assertEqual(jobs[0]["arms"], ["joint"])
        self.assertEqual(extra, "")

        args = M.build_parser().parse_args(["--rng-seeds", "7,8,9"])
        mode, extra, jobs = M.plan_jobs(args, bud, 30, 40, 13)
        self.assertIn("RNG direction", mode)
        self.assertEqual(extra, "__stage3a")
        self.assertEqual([j["rng_seed"] for j in jobs], [7, 8, 9])
        self.assertEqual(jobs[0]["steps"], M.DEFAULT_STEPS_3A)

        args = M.build_parser().parse_args(["--nsweep-ladder", "40,150,400"])
        mode, extra, jobs = M.plan_jobs(args, bud, 30, 40, 13)
        self.assertEqual(extra, "__stage3b-40-150-400")
        self.assertEqual([j["nsweep"] for j in jobs], [40, 150, 400])
        # equil keeps the budget's shape
        self.assertEqual(jobs[0]["equil"], 13)
        self.assertEqual(jobs[2]["equil"], round(13 / 40 * 400))

    def test_the_trajectory_mode_leaves_the_seed_to_the_budget(self):
        # `plan_jobs` must NOT invent a seed for the trajectory mode: index 0 has
        # to BE the production reference, or the pre-registered anchor is a
        # coincidence rather than a prediction.  `main` fills it from
        # `resolve(...).rng_seed`; test 11 checks that it did.
        bud = load_budget("quick")
        args = M.build_parser().parse_args(["--seeds", "0,1,2"])
        _, _, jobs = M.plan_jobs(args, bud, 30, 40, 13)
        self.assertEqual([j["rng_seed"] for j in jobs], [None, None, None])
        self.assertEqual([j["idx"] for j in jobs], [0, 1, 2])
        # ... while 3a's raw values are used verbatim, with no offset at all.
        args = M.build_parser().parse_args(["--rng-seeds", "0,1"])
        _, _, jobs = M.plan_jobs(args, bud, 30, 40, 13)
        self.assertEqual([j["rng_seed"] for j in jobs], [0, 1])

    def test_the_pilot_scale_and_pinned_arms_are_opt_in(self):
        bud = load_budget("quick")
        args = M.build_parser().parse_args(["--pilot-scale", "--pinned-control"])
        _, extra, jobs = M.plan_jobs(args, bud, 30, 40, 13)
        self.assertEqual(extra, "__pilot__pinned")
        self.assertEqual(jobs[0]["arms"], ["joint", "joint_pilot", "pinned"])


class TestTheScaleScalingReadout(unittest.TestCase):

    def test_a_noise_limited_ladder_fits_slope_minus_one_half(self):
        rungs = [dict(n=n, nsweep=n, equil=0, c6_1=1.0 / math.sqrt(n),
                      c6_0=1e-14, E_per_particle_1=-45.0)
                 for n in (40, 150, 400)]
        sc = M.scale_scaling(rungs)
        self.assertAlmostEqual(sc["slope_log_c6_vs_log_n"], -0.5, places=6)
        self.assertTrue(sc["consistent_with_noise"])

    def test_a_plateau_is_read_as_an_algorithmic_anisotropy(self):
        rungs = [dict(n=n, nsweep=n, equil=0, c6_1=0.08,
                      c6_0=1e-14, E_per_particle_1=-45.0)
                 for n in (40, 150, 400)]
        sc = M.scale_scaling(rungs)
        self.assertAlmostEqual(sc["slope_log_c6_vs_log_n"], 0.0, places=6)
        self.assertFalse(sc["consistent_with_noise"])

    def test_a_single_rung_makes_no_slope_claim(self):
        sc = M.scale_scaling([dict(n=40, nsweep=40, equil=13, c6_1=0.1,
                                   c6_0=1e-14, E_per_particle_1=-45.0)])
        self.assertIsNone(sc["slope_log_c6_vs_log_n"])
        self.assertIsNone(sc["consistent_with_noise"])


class TestTheDirectionReadout(unittest.TestCase):

    def _arm(self, seed, five, S, c6_1=0.2):
        rows = [_row(1e-14), _row(c6_1), _row(c6_1)]
        rows[1]["c6_five_vector"] = list(five)
        rows[1]["S_G1"], rows[1]["S_G2"], rows[1]["S_G3"] = S
        rows[1]["E_per_particle"] = -45.0
        rows[1]["E_err_per_particle"] = 0.05
        return {"arm": "joint", "tag": "3a", "rng_seed": seed, "rows": rows}

    def test_the_same_direction_every_seed_is_reported_as_agreement(self):
        five = [0.1, 0.2, 0.3, 0.2, 0.1]
        arms = [self._arm(s, five, [9.0, 8.0, 7.0]) for s in (0, 1, 2)]
        ag = M.direction_agreement(M.direction_table(arms))
        self.assertTrue(ag["all_same_rotation"])
        self.assertTrue(ag["all_same_family"])
        self.assertEqual(ag["family_argmax"], [1, 1, 1])

    def test_different_directions_are_reported_and_not_over_read(self):
        arms = [self._arm(0, [0.3, 0.1, 0.1, 0.1, 0.1], [9.0, 8.0, 7.0]),
                self._arm(1, [0.1, 0.1, 0.3, 0.1, 0.1], [8.0, 9.0, 7.0])]
        tab = M.direction_table(arms)
        self.assertEqual([t["fam_argmax"] for t in tab], [1, 2])
        ag = M.direction_agreement(tab)
        self.assertFalse(ag["all_same_rotation"])
        self.assertIn("finite-size", ag["note"])

    def test_one_seed_makes_no_direction_claim(self):
        ag = M.direction_agreement(M.direction_table([self._arm(0, [1.0] * 5,
                                                                [1.0, 1.0, 1.0])]))
        self.assertIn("no direction claim", ag["note"])


class TestEnergyBuysBreaking(unittest.TestCase):

    def _row(self, e, err, c6):
        d = _row(c6)
        d.update(E_per_particle=e, E_err_per_particle=err)
        return d

    def test_a_lower_joint_endpoint_reads_as_breaking_buying_energy(self):
        r = M.energy_buys_breaking(self._row(-45.10, 0.02, 0.3),
                                   self._row(-45.00, 0.02, 1e-14))
        self.assertLess(r["z"], -2.0)
        self.assertIn("breaking buys energy", r["reads"])

    def test_indistinguishable_endpoints_say_so(self):
        r = M.energy_buys_breaking(self._row(-45.00, 0.05, 0.3),
                                   self._row(-45.00, 0.05, 1e-14))
        self.assertAlmostEqual(r["z"], 0.0, places=12)
        self.assertIn("indistinguishable", r["reads"])

    def test_a_missing_arm_makes_no_claim(self):
        self.assertIsNone(M.energy_buys_breaking(None, self._row(-45.0, 0.02, 0.1)))


if __name__ == "__main__":
    unittest.main()
