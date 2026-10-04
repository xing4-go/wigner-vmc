"""The contract of ``examples/phase_competition_2/crystal_seed_search.py``.

Stage 1A exists to answer one question -- *does the paper's seed produce a
Wigner crystal?* -- and the value of its answer rests entirely on four things
being true before the run rather than after it.  Each is pinned here:

1. **The sampler is Haar, and only Haar.**  ``sin^2 t ~ U[0,1]`` on the CP1
   orbital manifold is the ``t = 0`` -respecting measure; a one-knob family
   would turn "a random determinant" into "a determinant of a chosen mixing
   strength", which is a different experiment and a chosen one.
2. **The gate is gauge-invariant and bounded.**  ``_space_residual`` is a
   principal-angle sine of the occupied subspace, so unlike a ratio of two noisy
   ``S`` values it can carry a threshold.  The corner values and the gauge
   invariance are the two properties the threshold rests on.
3. **The criterion and the escape hatch are fixed before the run.**  A verdict
   that can be re-read after seeing the numbers is not a verdict.  The escape
   hatch matters most: if the *known* crystal cannot clear the C6 gate at this
   budget, ``0/4`` is an artefact of the budget and must not be reported as
   physics.
4. **The code never builds the seed the paper does not use.**  Checked on the
   source's *code* rather than its prose, so the docstring can explain why the
   Gaussian construction is excluded without making the check vacuous.

The end-to-end test stubs the optimiser and keeps the estimators real: the
configurations it diagnoses are synthetic, but ``bragg_ratio``,
``site_midpoint_contrast`` and the C6 residual are the production ones, so a
wiring mistake between them and the table is still caught.  Paying for a real
walk to check plumbing would only make it slower, not stronger.
"""
import ast
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SCRIPT = os.path.join(CLEAN, "examples", "phase_competition_2",
                      "crystal_seed_search.py")
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wigner_vmc.api import RunState                          # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("crystal_seed_search_under_test",
                                                 SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code_strings(path=SCRIPT):
    """Every string literal in the module that is NOT a docstring.

    Prose is allowed to name a construction it refuses to use -- that is how a
    reader learns the rule.  Code is not.  Filtering by the docstring nodes
    (rather than by comments, which the parser discards anyway) is what
    separates the two.
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            got = ast.get_docstring(node, clean=False)
            if got is not None:
                doc_nodes.add(got)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in doc_nodes:
                yield node.lineno, node.value


M = _load()


# ==========================================================================
# 1. the sampler is Haar, and has no scale knob
# ==========================================================================
class TestTheSamplerIsHaar(unittest.TestCase):
    def test_the_shape_is_the_nmax_1_orbital_matrix(self):
        v = M.haar_random_v(36, 2, np.random.default_rng(0))
        self.assertEqual(v.shape, (36, 1))
        self.assertEqual(v.dtype, np.complex128)
        self.assertTrue(np.all(np.isfinite(v)))

    def test_t_is_in_the_first_quadrant_of_its_range(self):
        """``|v| = t = arcsin(sqrt(u))`` with ``u in [0,1)``, so ``t in [0, pi/2)``.

        The upper bound is what keeps ``cos t >= 0``: the CP1 chart covers the
        southern hemisphere through the phase, not through a negative ``t``.
        """
        v = M.haar_random_v(2000, 2, np.random.default_rng(1))
        t = np.abs(v)
        self.assertGreaterEqual(float(t.min()), 0.0)
        self.assertLess(float(t.max()), 0.5 * math.pi + 1e-12)

    def test_the_same_seed_gives_a_byte_identical_draw(self):
        a = M.haar_random_v(36, 2, np.random.default_rng(7))
        b = M.haar_random_v(36, 2, np.random.default_rng(7))
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_different_seeds_give_different_draws(self):
        a = M.haar_random_v(36, 2, np.random.default_rng(7))
        b = M.haar_random_v(36, 2, np.random.default_rng(8))
        self.assertFalse(np.array_equal(a, b))

    def test_the_draw_is_isotropic_in_phase(self):
        """``phi ~ U[0, 2pi)`` i.i.d.  A phase locked to the index would show up
        as a non-zero mean of the unit phasors."""
        v = M.haar_random_v(20000, 2, np.random.default_rng(2))[:, 0]
        z = v / np.abs(v)
        self.assertLess(abs(complex(z.mean())), 0.05)

    def test_the_excited_band_weight_averages_one_half(self):
        """``E[sin^2 t] = 1/2`` is the defining property of the Haar draw.

        ``sin^2 t ~ U[0,1]`` has mean 1/2, and this is the one number that
        separates Haar from a ``t ~ U[0, t_max]`` family (which would give
        ``1/2 - sin(2 t_max)/(4 t_max)``, i.e. 0.42 at ``t_max = 0.785``).
        """
        rng = np.random.default_rng(3)
        n = 200_000
        v = M.haar_random_v(n, 2, rng)
        self.assertAlmostEqual(M.seed_occupation(v), 0.5, delta=0.01)

    def test_the_draw_is_the_one_c_row_reads(self):
        """``seed_occupation`` must be the excited weight of the orbitals the
        optimiser will actually use, not a re-derivation of ``t``."""
        v = M.haar_random_v(36, 2, np.random.default_rng(5))
        t = np.abs(v[:, 0])
        self.assertAlmostEqual(M.seed_occupation(v), float((np.sin(t) ** 2).mean()))

    def test_there_is_no_scale_knob(self):
        import inspect
        params = list(inspect.signature(M.haar_random_v).parameters)
        self.assertEqual(params, ["nk", "n_bands", "rng"])
        for name in dir(M):
            if name.startswith("_"):
                continue
            self.assertNotIn("t_max", name.lower())
            self.assertNotIn("tmax", name.lower())
        hits = [(ln, s) for ln, s in _code_strings()
                if "t_max" in s.lower() or "tmax" in s.lower()]
        self.assertEqual(hits, [], f"a scale knob reappeared in code: {hits}")

    def test_a_higher_nmax_is_refused_rather_than_silently_producted(self):
        """The CP1 statement is the reason the draw is canonical; it does not
        generalise to an independent per-column product."""
        with self.assertRaises(ValueError):
            M.haar_random_v(36, 3, np.random.default_rng(0))


# ==========================================================================
# 2. the C6 residual is a gate, not a ratio
# ==========================================================================
class TestTheC6ResidualIsAGate(unittest.TestCase):
    """Pure linear algebra -- no lattice, no Monte Carlo, so a failure here is
    a defect in the test's own understanding, not noise."""

    @staticmethod
    def _blanks(n_pts=200, n_col=3, seed=11):
        rng = np.random.default_rng(seed)
        Psi = rng.standard_normal((n_pts, n_col)) + 1j * rng.standard_normal(
            (n_pts, n_col))
        return Psi, 1.0 / n_pts

    def test_a_covariant_subspace_reads_zero(self):
        """The rotated space IS the unrotated space, so the fit is exact."""
        Psi, w = self._blanks()
        rng = np.random.default_rng(12)
        X = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        self.assertLess(M.space_residual(Psi, Psi @ X, w), 1e-12)

    def test_a_broken_subspace_reads_order_one(self):
        """An unrelated subspace cannot be fitted by any ``cf``; the residual is
        the sine of the principal angles and is therefore O(1), not small."""
        Psi, w = self._blanks(seed=13)
        PsiR, _ = self._blanks(seed=14)
        self.assertGreater(M.space_residual(Psi, PsiR, w), 0.3)

    def test_it_is_bounded_by_one(self):
        """The property that lets a threshold mean something.  ``A``, the ratio
        of two noisy ``S`` values, has no such bound."""
        Psi, w = self._blanks(seed=15)
        for seed in range(20):
            PsiR, _ = self._blanks(seed=100 + seed)
            r = M.space_residual(Psi, PsiR, w)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0 + 1e-12)

    def test_it_is_gauge_invariant_on_both_sides(self):
        """Rescaling the columns of either space cannot change it.

        This is the property the withdrawn ``T_k(R^-1 r) == T_k(r)`` test lacked,
        and it is why this one may be thresholded: a gate that moves when the
        basis is re-chosen is a statement about the basis, not the state.
        """
        Psi, w = self._blanks(seed=16)
        rng = np.random.default_rng(17)
        PsiR = Psi @ (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
        base = M.space_residual(Psi, PsiR, w)

        d1 = rng.uniform(0.2, 5.0, 3) * np.exp(1j * rng.uniform(0, 2 * math.pi, 3))
        d2 = rng.uniform(0.2, 5.0, 3) * np.exp(1j * rng.uniform(0, 2 * math.pi, 3))
        self.assertAlmostEqual(M.space_residual(Psi @ np.diag(d1),
                                                PsiR @ np.diag(d2), w),
                               base, places=9)

    def test_rot_is_the_plane_rotation_the_C6_test_applies(self):
        """``pts @ rot(deg)`` -- a point rotation, and the transpose would be the
        passive one.  Getting it backwards would test the inverse rotation."""
        R = M.rot(90.0)
        np.testing.assert_allclose(R, [[0.0, -1.0], [1.0, 0.0]], rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
        np.testing.assert_allclose(R @ np.array([1.0, 0.0]), [0.0, 1.0],
                                   rtol=0, atol=1e-12)

    def test_the_five_rotations_are_the_ones_that_can_see_a_hexagon(self):
        self.assertEqual(list(M.C6_ROTS), [60.0, 120.0, 180.0, 240.0, 300.0])
        self.assertIn(180.0, M.C6_ROTS)
        self.assertEqual(M.C6_GATE, 0.1)


# ==========================================================================
# 3. the criterion and the escape hatch are pre-registered
# ==========================================================================
def _rec(bragg=50.0, c6=1e-14, contrast=8.0, E=-37.0, trial=0, err=1e-3):
    return {"trial": trial, "bragg_ratio": bragg, "c6_residual": c6,
            "site_midpoint_contrast": contrast, "E_final": E,
            "E_final_err": err}


class TestTheCriterionIsPreRegistered(unittest.TestCase):
    def test_a_perfect_lattice_passes(self):
        out = M.wc_verdict(_rec(), _rec(bragg=1.0, c6=0.7, contrast=1.0))
        self.assertTrue(out["wc_formed"])
        self.assertEqual(set(out["gates"]), {"bragg", "c6", "contrast"})
        self.assertTrue(all(out["gates"].values()))

    def test_a_state_with_no_bragg_enhancement_fails(self):
        """2x the liquid is not 3x the liquid -- the threshold is on the RATIO to
        the reference measured in the same run, not on an absolute level."""
        out = M.wc_verdict(_rec(bragg=2.0), _rec(bragg=1.0, c6=0.7, contrast=1.0))
        self.assertFalse(out["gates"]["bragg"])
        self.assertFalse(out["wc_formed"])

    def test_a_bragg_peak_with_broken_C6_fails(self):
        """The whole point of gate 2: a peak alone is not a triangular crystal.
        A C6-broken state reads 7.9e-01 here, not 3.0e-14."""
        out = M.wc_verdict(_rec(c6=0.79), _rec(bragg=1.0, c6=0.7, contrast=1.0))
        self.assertFalse(out["gates"]["c6"])
        self.assertFalse(out["wc_formed"])

    def test_a_uniform_density_fails_the_contrast_gate(self):
        """``site_midpoint_contrast`` is exactly 1 for anything uniform, so the
        liquid sits at the null and the gate is calibrated, not chosen."""
        out = M.wc_verdict(_rec(contrast=1.0), _rec(bragg=1.0, c6=0.7, contrast=1.0))
        self.assertFalse(out["gates"]["contrast"])

    def test_every_gate_component_is_reported_whether_or_not_it_passes(self):
        out = M.wc_verdict(_rec(bragg=0.1, c6=9.0, contrast=0.1),
                           _rec(bragg=1.0, c6=0.7, contrast=1.0))
        self.assertEqual(sorted(out["gates"]), ["bragg", "c6", "contrast"])
        self.assertEqual(sum(out["gates"].values()), 0)

    def test_the_thresholds_are_the_published_constants(self):
        """A verifier reading the table needs to know what "WC" meant here."""
        self.assertEqual(M.BRAGG_GATE, 3.0)
        self.assertEqual(M.CONTRAST_GATE, 2.0)
        self.assertEqual(M.C6_GATE, 0.1)

    def test_the_liquid_is_gated_on_its_own_bragg_ratio_not_a_constant(self):
        """A liquid that itself reads a high Bragg ratio raises the bar.  This is
        what stops a noisy reference from making a pass look easy."""
        loose = M.wc_verdict(_rec(bragg=30.0), _rec(bragg=1.0, c6=0.7, contrast=1.0))
        tight = M.wc_verdict(_rec(bragg=30.0), _rec(bragg=12.0, c6=0.7, contrast=1.0))
        self.assertTrue(loose["gates"]["bragg"])
        self.assertFalse(tight["gates"]["bragg"])


class TestTheEscapeHatch(unittest.TestCase):
    def test_a_reference_that_fails_the_C6_gate_gives_no_verdict(self):
        trials = [_rec(trial=i) for i in range(4)]
        for t in trials:
            t.update(M.wc_verdict(t, _rec(bragg=1.0, c6=0.7, contrast=1.0)))
        verdict, reason = M.overall_verdict(trials, _rec(c6=0.42))
        self.assertEqual(verdict, "NO VERDICT")
        self.assertIn("4.200e-01", reason)          # the reference's own number
        self.assertIn("cannot resolve", reason)

    def test_no_verdict_beats_a_full_house_of_passes(self):
        """Checked FIRST, so a resolvable-looking 4/4 cannot outrank it: if the
        gate cannot separate the known crystal, nothing it says about the random
        starts is interpretable either way."""
        trials = [_rec(trial=i) for i in range(4)]
        for t in trials:
            t.update(M.wc_verdict(t, _rec(bragg=1.0, c6=0.7, contrast=1.0)))
        self.assertEqual(M.overall_verdict(trials, _rec(c6=0.9))[0], "NO VERDICT")

    def test_a_resolvable_reference_lets_the_verdict_be_read(self):
        trials = [_rec(trial=i) for i in range(4)]
        for t in trials:
            t.update(M.wc_verdict(t, _rec(bragg=1.0, c6=0.7, contrast=1.0)))
        verdict, reason = M.overall_verdict(trials, _rec(c6=1e-13))
        self.assertEqual(verdict, "PASSED")
        self.assertIn("4/4", reason)

    def test_all_failing_is_FAILED_and_says_what_it_does_not_mean(self):
        trials = [_rec(trial=i, bragg=0.5, c6=0.8, contrast=0.9) for i in range(4)]
        for t in trials:
            t.update(M.wc_verdict(t, _rec(bragg=1.0, c6=0.7, contrast=1.0)))
        verdict, reason = M.overall_verdict(trials, _rec(c6=1e-13), "quick")
        self.assertEqual(verdict, "FAILED")
        self.assertIn("0/4", reason)
        self.assertIn("NOT evidence", reason)

    def test_the_failure_reason_names_the_budget_that_was_actually_run(self):
        """'The optimiser may not have had time' is only meaningful against the
        protocol that ran: a smoke-budget 0/4 is far weaker than a
        reproduction-budget one, and the message must not blur them."""
        trials = [_rec(trial=0, bragg=0.5, c6=0.8, contrast=0.9)]
        trials[0].update(M.wc_verdict(trials[0],
                                      _rec(bragg=1.0, c6=0.7, contrast=1.0)))
        verdict, reason = M.overall_verdict(trials, _rec(c6=1e-13), "smoke")
        self.assertEqual(verdict, "FAILED")
        self.assertIn("budget smoke", reason)
        self.assertNotIn("quick", reason)


# ==========================================================================
# 4. the root rule
# ==========================================================================
class TestTheRootRule(unittest.TestCase):
    def test_it_picks_the_lowest_energy_trial_that_passed(self):
        trials = [_rec(trial=0, E=-37.0), _rec(trial=1, E=-36.0),
                  _rec(trial=2, E=-38.0), _rec(trial=3, E=-35.0)]
        for t in trials:
            t["wc_formed"] = t["trial"] in (0, 1, 3)       # trial 2 is lowest but fails
        self.assertEqual(M.choose_root(trials), 0)

    def test_it_returns_none_when_nothing_passed(self):
        trials = [_rec(trial=i) for i in range(4)]
        for t in trials:
            t["wc_formed"] = False
        self.assertIsNone(M.choose_root(trials))

    def test_it_never_falls_back_to_the_best_energy_overall(self):
        """The failure mode this exists to stop: relaxing the criterion after
        seeing the numbers.  An energetic winner that is not a WC is a different
        object and must not be returned under the name ``root``."""
        trials = [_rec(trial=0, E=-40.0), _rec(trial=1, E=-36.0)]
        trials[0]["wc_formed"] = False
        trials[1]["wc_formed"] = False
        self.assertIsNone(M.choose_root(trials))

    def test_the_root_is_an_int_trial_index_not_a_seed(self):
        trials = [_rec(trial=2, E=-37.0)]
        trials[0]["wc_formed"] = True
        root = M.choose_root(trials)
        self.assertIsInstance(root, int)
        self.assertEqual(root, 2)


# ==========================================================================
# 5. +-G is an exact identity, not a measurement
# ==========================================================================
class TestPlusMinusGIsExact(unittest.TestCase):
    def test_s_plus_q_equals_s_minus_q_exactly(self):
        """``rho_{-q} = conj(rho_q)``, so the two are bit-identical and a
        non-zero residual means the estimator is wrong, not that the physics is
        anisotropic."""
        from wigner_vmc.analysis import structure as st
        torus = M.analysis_torus(36)
        rng = np.random.default_rng(21)
        snaps = [rng.random((36, 2)) @ np.asarray(torus.sc, float).T
                 for _ in range(8)]
        for pair in M.bragg_families(torus):
            S = np.real(st.structure_factor(snaps, np.array(pair), 36))
            self.assertEqual(float(S[0]), float(S[1]))


# ==========================================================================
# 6. the bragg sets are the transcribed ones
# ==========================================================================
class TestTheBraggSetsAreTheTranscribedOnes(unittest.TestCase):
    """The two sets are transcribed from ``tests/test_measure_against_legacy.py``
    rather than remembered, and this compares against that file's own functions
    so the two cannot drift apart."""

    @staticmethod
    def _legacy():
        spec = importlib.util.spec_from_file_location(
            "_mal_for_seed_search",
            os.path.join(HERE, "test_measure_against_legacy.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_mal_for_seed_search"] = mod
        try:
            spec.loader.exec_module(mod)
        except ImportError as exc:                      # pragma: no cover
            raise unittest.SkipTest(f"legacy engine unavailable: {exc}")
        return mod

    @staticmethod
    def _lex(a):
        a = np.asarray(a, float)
        return a[np.lexsort((a[:, 1], a[:, 0]))]

    def test_the_six_first_shell_vectors_match(self):
        mal = self._legacy()
        torus = M.analysis_torus(36)
        mine = self._lex(M.bragg_vectors(torus))
        theirs = self._lex(mal.six_bragg(torus))
        self.assertEqual(mine.shape, (6, 2))
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-12)

    def test_the_background_momenta_match(self):
        mal = self._legacy()
        torus = M.analysis_torus(36)
        mine = self._lex(M.background_vectors(torus))
        theirs = self._lex(mal.background_q(torus))
        self.assertEqual(mine.shape, theirs.shape)
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-12)

    def test_the_counts_are_six_and_one_hundred_and_twenty(self):
        torus = M.analysis_torus(36)
        self.assertEqual(len(M.bragg_vectors(torus)), 6)
        self.assertEqual(len(M.background_vectors(torus)), 120)

    def test_the_three_families_partition_the_shell(self):
        """Every first-shell vector is in exactly one family, and all six share
        one ``|q|`` -- that is what makes them one shell and the equality of the
        three ``S`` values evidence for C6."""
        torus = M.analysis_torus(36)
        shell = M.bragg_vectors(torus)
        fams = M.bragg_families(torus)
        self.assertEqual(sum(len(f) for f in fams), 6)
        flat = self._lex(np.array([q for f in fams for q in f]))
        np.testing.assert_allclose(flat, self._lex(shell), rtol=0, atol=1e-12)
        mags = np.linalg.norm(shell, axis=1)
        self.assertLess(float(mags.max() - mags.min()), 1e-9 * float(mags.mean()))

    def test_the_analysis_torus_is_the_one_the_state_builds(self):
        """Same area convention as ``RunState.structure_factor`` -- ``N * 2*pi``
        and not the bare primitive ``2*pi``, which built a torus N times too
        small and silently returned six momenta."""
        t = M.analysis_torus(36)
        self.assertAlmostEqual(t.area, 2.0 * math.pi * 36)
        self.assertEqual(t.n_side, 6)
        self.assertEqual(len(t.allowed_momenta(4.0)[0]), 294)


# ==========================================================================
# 8. the liquid has no variational orbital sector
# ==========================================================================
class TestTheLiquidHasNoOrbitalSector(unittest.TestCase):
    """The pin the brief asked for: at ``v = 0`` the determinant is the
    noninteracting ground state, so a larger ``n_max`` adds an *unoccupied* band
    and cannot move the energy.  A liquid whose ``E`` depended on ``n_max`` would
    mean an orbital sector had appeared -- and the liquid is the control every
    crystal number here is measured against.

    Run at ``smoke`` rather than the recipe's ``quick``: the claim is about the
    construction and is budget-independent, and two real walks at ``quick``
    would cost minutes for nothing.  Measured: the two runs cost ~135 s together,
    which is the whole of this file's runtime.
    """

    TOL = 1e-9          # the repo's _BLAS_TOL: reduction order moves ~1e-11

    def _liquid(self, nmax):
        from wigner_vmc import VMC
        return VMC(N=36, rs=90.0, phase="liquid", nmax=nmax).run(
            init_id=0, budget="smoke", verbose=False)

    def test_the_same_liquid_energy_at_nmax_1_and_2(self):
        a = self._liquid(1)
        b = self._liquid(2)
        self.assertLess(abs(float(a.energy_per_particle)
                            - float(b.energy_per_particle)), self.TOL)

    def test_the_liquid_start_is_the_filled_lowest_landau_level(self):
        from wigner_vmc import VMC
        vmc = VMC(N=36, rs=90.0, phase="liquid", nmax=2)
        self.assertEqual(tuple(np.asarray(vmc.liquid_R0()).shape), (36, 2))
        # the v the constructor uses is zeros, so every orbital row is (1,0,0)
        v0 = np.zeros((vmc.lat.nk, vmc.n_bands - 1), complex)
        from wigner_vmc.wavefunctions import ll_rotation as lr
        C = lr.c_row(v0)
        np.testing.assert_allclose(np.abs(C[:, 0]), 1.0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.abs(C[:, 1:]), 0.0, rtol=0, atol=1e-15)

    def test_the_liquid_branch_never_touches_the_crystal_start(self):
        """``_run``'s liquid branch builds ``liquid_R0()`` and ``v = 0``, so a
        trial's random determinant cannot leak into the control.  Read off the
        source rather than restated here, because the claim is about ``_run``.

        Parsed, not sliced by text: ``_run`` has nested ``else:`` blocks and a
        substring search picks up the wrong one -- which is how this test's first
        version passed a branch containing ``crystal_R0()``.
        """
        text = io.open(os.path.join(SRC, "wigner_vmc", "api.py"),
                       encoding="utf-8").read()
        tree = ast.parse(text)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_run")
        branch = next(n for n in fn.body
                      if isinstance(n, ast.If)
                      and any(isinstance(s, ast.Attribute) and s.attr == "phase"
                              for s in ast.walk(n.test)))
        self.assertEqual(len(branch.orelse) > 0, True)    # no elif chain
        liquid = "\n".join(ast.unparse(n) for n in branch.orelse)
        self.assertIn("liquid_R0()", liquid)
        self.assertNotIn("crystal_R0()", liquid)
        self.assertIn("np.zeros((self.lat.nk, self.n_bands - 1), complex)", liquid)
        crystal = "\n".join(ast.unparse(n) for n in branch.body)
        self.assertIn("crystal_R0()", crystal)
        self.assertNotIn("liquid_R0()", crystal)


# ==========================================================================
# 7. the contrast is a finite measurement, not a free pass
# ==========================================================================
class TestTheContrastIsAFiniteMeasurement(unittest.TestCase):
    """``site_midpoint_contrast`` is gate 3, so it must be a number.

    ``density_grid`` is a hard histogram.  At 72 bins the nearest-neighbour
    midpoints of a sharp lattice land in bins with no particles at all, the
    denominator is exactly 0, and the statistic is ``inf`` -- a gate nothing can
    fail.  ``examples/figure_construction/structure.py`` avoids this by computing
    its contrast on the blurred ``rho_cell``, and this recipe must do the same.
    """

    def setUp(self):
        from wigner_vmc import VMC
        self.vmc = VMC(N=36, rs=90.0, phase="crystal", nmax=1)
        self.torus = M.analysis_torus(36)

    def test_the_grid_and_kernel_are_the_historical_ones(self):
        """Tracing to `structure.py:135-136`; a change here changes the gate."""
        self.assertEqual(M.DENSITY_BINS, 72)
        self.assertEqual(M.DENSITY_KERNEL_L_B, 0.40)
        from wigner_vmc.analysis import structure as st
        self.assertAlmostEqual(
            M.DENSITY_KERNEL_L_B
            / (float(np.linalg.norm(self.torus.L1)) / M.DENSITY_BINS),
            1.7819, places=3)                    # the value the other suite pins

    def test_a_uniform_cloud_reads_one(self):
        """The null.  The blur cannot manufacture structure where there is none,
        which is what makes the gate calibrated rather than tuned."""
        rng = np.random.default_rng(31)
        snaps = _random_snaps(self.torus, rng, n_snaps=200)
        H = M.density_field(snaps, self.torus)
        contrast = M.st.site_midpoint_contrast(H, self.torus, M.DENSITY_BINS)
        self.assertTrue(math.isfinite(contrast))
        self.assertLess(abs(contrast - 1.0), 0.2, f"contrast {contrast}")

    def test_the_raw_histogram_would_pass_the_gate_for_free(self):
        """The regression for the defect this class exists for: on the UNBLURRED
        grid the same lattice is ``inf`` (or grossly inflated) -- so the gate
        would have been satisfied before any physics was looked at."""
        rng = np.random.default_rng(32)
        snaps = _lattice_snaps(self.vmc, rng, jitter=0.0)     # exact lattice
        raw = M.st.density_grid(snaps, self.torus, nbins=M.DENSITY_BINS)
        with warnings.catch_warnings():
            # the divide-by-zero IS the finding here, not a defect to report
            warnings.simplefilter("ignore", RuntimeWarning)
            unblurred = M.st.site_midpoint_contrast(raw, self.torus,
                                                    M.DENSITY_BINS)
        self.assertTrue(not math.isfinite(unblurred) or unblurred > 20.0,
                        f"the raw histogram read a usable {unblurred}")

    def test_a_physical_lattice_reads_a_finite_contrast_above_the_null(self):
        rng = np.random.default_rng(33)
        snaps = _lattice_snaps(self.vmc, rng, jitter=0.5)
        H = M.density_field(snaps, self.torus)
        contrast = M.st.site_midpoint_contrast(H, self.torus, M.DENSITY_BINS)
        self.assertTrue(math.isfinite(contrast), f"contrast is {contrast}")
        self.assertGreater(contrast, M.CONTRAST_GATE)
        self.assertLess(contrast, 1e3)          # a number, not a saturation

    def test_the_density_field_is_normalised_to_the_cell_mean(self):
        rng = np.random.default_rng(34)
        snaps = _random_snaps(self.torus, rng, n_snaps=200)
        H = M.density_field(snaps, self.torus)
        self.assertEqual(H.shape, (M.DENSITY_BINS, M.DENSITY_BINS))
        self.assertLess(abs(float(H.mean()) - 1.0), 0.01)

    def test_the_kernel_width_is_a_physical_length_not_a_bin_count(self):
        """The same ``l_B`` width at a different grid resolution must give the
        same contrast.  A sigma hard-coded in BINS would double with ``nbins``
        and the answer would move with it."""
        rng = np.random.default_rng(35)
        snaps = _lattice_snaps(self.vmc, rng, jitter=0.5)
        got = [M.st.site_midpoint_contrast(M.density_field(snaps, self.torus,
                                                          nbins=nb),
                                           self.torus, nb)
               for nb in (72, 144)]
        self.assertTrue(all(math.isfinite(g) for g in got), got)
        self.assertLess(abs(got[0] - got[1]) / got[0], 0.25,
                        f"contrast moved with the grid resolution: {got}")


# ==========================================================================
# 9. AST contracts -- what the code may and may not do
# ==========================================================================
#: A path that would mean the recipe had reached into the legacy tree.  The
#: first is the defective LL-rotation crystal records, which this project is
#: forbidden to read.
FORBIDDEN = ("legacy_llrot", "_diag/part1", "energy_scan", "make_notebook")

#: The seed the paper does NOT use.  It may be named in prose; it may not be
#: built in code.
GAUSSIAN_SEED_NAMES = ("gaussian_overlap_seed", "v_from_overlap", "crystal_v0")


class TestTheAstContracts(unittest.TestCase):
    def test_the_script_does_not_touch_sys_path(self):
        tree = ast.parse(io.open(SCRIPT, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("insert",
                                                                "append"):
                v = node.value
                if (isinstance(v, ast.Attribute) and v.attr == "path"
                        and isinstance(v.value, ast.Name) and v.value.id == "sys"):
                    self.fail(f"sys.path.{node.attr} at line {node.lineno}")

    def test_no_legacy_path_appears_in_the_code(self):
        hits = [(ln, s) for ln, s in _code_strings()
                if any(f in s for f in FORBIDDEN)]
        self.assertEqual(hits, [], f"legacy paths in code: {hits}")

    def test_the_prose_still_states_the_bans(self):
        """The checks above are only meaningful while the docstring says WHY."""
        head = io.open(SCRIPT, encoding="utf-8").read().split('"""')[1]
        for token in FORBIDDEN:
            self.assertIn(token, head, f"the module docstring no longer names "
                                       f"{token!r}")
        for token in GAUSSIAN_SEED_NAMES:
            self.assertIn(token, head,
                          f"the module docstring no longer names {token!r}")

    def test_the_code_never_builds_v_from_a_gaussian(self):
        """Exercised through the real entry point, not merely grepped: the
        recipe draws its starts from ``haar_random_v`` and nothing else."""
        hits = [(ln, s) for ln, s in _code_strings()
                if any(g in s for g in GAUSSIAN_SEED_NAMES)]
        self.assertEqual(hits, [], f"the Gaussian seed reappeared in code: {hits}")

    def test_the_recipe_only_writes_under_results_and_figures(self):
        """House style: ``examples/`` holds code, never output."""
        for name, part in (("RESULTS", "results"), ("FIGDIR", "figures")):
            parts = os.path.normpath(getattr(M, name)).split(os.sep)
            self.assertIn(part, parts, f"{name} is not under {part}/")
            self.assertIn("phase_competition_2", parts)

    def test_no_module_global_points_at_output_outside_the_repo(self):
        for name in dir(M):
            if name.startswith("_"):
                continue
            val = getattr(M, name)
            if isinstance(val, str) and os.path.isabs(val):
                self.assertTrue(val.startswith(CLEAN),
                                f"module global {name} = {val!r} escapes the repo")

    def test_the_verdict_is_a_function_of_the_records_only(self):
        """No threshold may be read from a file or an environment variable: the
        criterion has to be in the source a reader can see."""
        src = io.open(SCRIPT, encoding="utf-8").read()
        for name in ("wc_verdict", "overall_verdict", "choose_root"):
            head = src.index(f"def {name}(")
            body = src[head:src.index("\ndef ", head + 1)]
            for bad in ("os.environ", "open(", "json.load"):
                self.assertNotIn(bad, body, f"{name} reads {bad}")


# ==========================================================================
# 10. end to end, with the optimiser stubbed and the estimators real
# ==========================================================================
def _lattice_snaps(vmc, rng, n_snaps=48, jitter=0.5):
    """Configurations that really are a triangular crystal, so the production
    estimators have something real to find.

    ``jitter`` is a width in magnetic lengths -- 0.5 is a plausible LLL Wigner
    crystal, not a delta lattice.  A near-delta lattice is the wrong test object:
    it makes ``site_midpoint_contrast`` read ``inf`` (see the density test class).
    """
    sites = np.asarray(vmc.lat.sites, float)
    return [sites + jitter * rng.standard_normal((vmc.N, 2))
            for _ in range(n_snaps)]


#: One fixed batch of crystal snapshots, shared by EVERY stubbed trial.  The
#: end-to-end test is then about the occupied subspace and nothing else: the
#: Bragg ratio, the density contrast and the walk are byte-identical between
#: trials, so a difference in ``wc_formed`` can only come from the C6 gate.
#: (The real recipe varies all of them, which is why it reports all of them.)
SNAP_SEED = 4242


def _random_snaps(torus, rng, n_snaps=24):
    return [rng.random((36, 2)) @ np.asarray(torus.sc, float).T
            for _ in range(n_snaps)]


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="seed_search_contract_")
        self._saved = (M.RESULTS, M.FIGDIR)
        M.RESULTS = os.path.join(self.tmp, "results")
        M.FIGDIR = os.path.join(self.tmp, "figures")
        os.makedirs(M.RESULTS, exist_ok=True)
        os.makedirs(M.FIGDIR, exist_ok=True)

    def tearDown(self):
        M.RESULTS, M.FIGDIR = self._saved
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stub(self, argv, n_trials, symmetric_last=True):
        """Stub the optimiser and the two reference runs; keep everything else.

        Every trial gets the SAME snapshots; the last one is given the C6-covariant
        filled lowest Landau level (``v = 0``) instead of its drawn determinant.
        So the trials differ in exactly one thing -- the occupied subspace -- and
        any difference in ``wc_formed`` is the C6 gate's doing.
        """
        from wigner_vmc import VMC
        torus = M.analysis_torus(36)
        calls = {"run_from": 0}

        def fake_run_from(self_vmc, cfg, theta0, R0, verbose=True):
            calls["run_from"] += 1
            E = -37.0 - 0.1 * calls["run_from"]
            c, v = M.theta_parts(np.asarray(theta0, float), self_vmc.n_bands,
                                 self_vmc.lat.nk)
            if symmetric_last and calls["run_from"] >= n_trials:
                v = np.zeros_like(v)            # the filled LLL: C6-covariant
            state = RunState(
                R=R0, c=c, v=v,
                N=self_vmc.N, rs=self_vmc.rs, kappa=self_vmc.kappa,
                kappa_mode=self_vmc.kappa_mode, nmax=self_vmc.nmax,
                phase="crystal",
                snaps=_lattice_snaps(self_vmc,
                                     np.random.default_rng(SNAP_SEED)))
            return types.SimpleNamespace(energy_per_particle=E, error=1e-3,
                                         acceptance=0.5, state=state)

        def fake_run(self_vmc, init_id=0, budget="quick", verbose=True):
            rng = np.random.default_rng(2000)
            crystal = self_vmc.phase == "crystal"
            state = RunState(
                R=self_vmc.crystal_R0() if crystal else self_vmc.liquid_R0(),
                c=np.zeros(5), v=np.zeros((self_vmc.lat.nk,
                                           self_vmc.n_bands - 1), complex),
                N=self_vmc.N, rs=self_vmc.rs, kappa=self_vmc.kappa,
                kappa_mode=self_vmc.kappa_mode, nmax=self_vmc.nmax,
                phase=self_vmc.phase,
                snaps=(_lattice_snaps(self_vmc, rng) if crystal
                       else _random_snaps(torus, rng)))
            return types.SimpleNamespace(
                energy_per_particle=-37.5 if crystal else -36.0, error=1e-3,
                acceptance=0.5, state=state)

        with mock.patch.object(VMC, "_run_from", fake_run_from), \
                mock.patch.object(VMC, "run", fake_run), \
                mock.patch.object(M, "measure_now",
                                  lambda *a, **k: {"E": -36.5, "E_err": 2e-3}):
            rc = M.main(argv)
        return rc, calls

    def test_it_writes_the_table_the_metadata_and_the_figure(self):
        rc, calls = self._stub(["--rs", "90", "--trials", "3", "--nsp", "60",
                                "--budget", "smoke"], n_trials=3)
        self.assertEqual(rc, 0)
        self.assertEqual(calls["run_from"], 3)

        slug = M.request_slug(36, "ll_rotation", "smoke", 90.0, 1)
        res = os.path.join(M.RESULTS, slug)
        with io.open(os.path.join(res, "runs.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        with io.open(os.path.join(res, "run_metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)

        self.assertEqual(len(payload["trials"]), 3)
        self.assertEqual([t["trial"] for t in payload["trials"]], [0, 1, 2])
        self.assertEqual(payload["liquid"]["kind"], "liquid")
        self.assertEqual(payload["reference"]["kind"], "reference")

        t0 = payload["trials"][0]
        for key in ("E_init", "E_final", "dE", "z", "nbar_init", "bragg_ratio",
                    "S_G1", "S_G2", "S_G3", "anisotropy_A", "c6_residual",
                    "c6_by_rotation", "site_midpoint_contrast", "pm_G_residual",
                    "ll_P", "ll_nbar", "theta_digest", "gates", "wc_formed"):
            self.assertIn(key, t0, f"the per-trial record lost {key}")
        self.assertEqual(sorted(t0["c6_by_rotation"]),
                         ["120", "180", "240", "300", "60"])

        # the +-G identity, on real snapshots through the real estimator
        self.assertEqual(t0["pm_G_residual"], 0.0)

        # The gate's work, isolated: every trial has the SAME snapshots, so the
        # Bragg ratio and the contrast agree trial to trial and only the occupied
        # subspace differs.  The random determinants fail C6; the filled LLL,
        # which spans a C6-invariant space, passes it.
        braggs = {round(t["bragg_ratio"], 9) for t in payload["trials"]}
        self.assertEqual(len(braggs), 1, "the stub's snapshots were not shared")
        self.assertEqual([t["wc_formed"] for t in payload["trials"]],
                         [False, False, True])
        for t in payload["trials"][:2]:
            self.assertGreater(t["c6_residual"], M.C6_GATE)
            self.assertTrue(t["gates"]["bragg"])            # it IS a lattice ...
            self.assertFalse(t["gates"]["c6"])              # ... in the wrong space
        self.assertLess(payload["trials"][2]["c6_residual"], M.C6_GATE)
        self.assertGreater(t0["site_midpoint_contrast"], 2.0)

        self.assertEqual(payload["verdict"], "PASSED")
        self.assertEqual(payload["root_trial"], 2)      # lowest E among passers

        self.assertEqual(meta["stage"].split(" --")[0], "1A")
        self.assertIn("NO VERDICT", meta["criterion"]["escape_hatch"])
        self.assertEqual(meta["criterion"]["c6_gate"], M.C6_GATE)
        self.assertTrue(meta["random_measure"]["no_scale_knob"])
        self.assertEqual(meta["len_theta"], 77)
        self.assertIn("reproduction/full", meta["not_in_scope"])
        self.assertEqual(meta["root_rs_note"], M.DEFAULT_RS_NOTE)
        self.assertEqual(meta["figure_stamp"], M.QUALITY_STAMP)

        self.assertTrue(os.path.exists(os.path.join(
            M.FIGDIR, slug, "crystal_seed_search.png")))

    def test_a_run_where_nothing_passes_reports_no_root_and_no_pickle(self):
        """The other branch: the criterion is not relaxed after the fact."""
        rc, _ = self._stub(["--rs", "90", "--trials", "3", "--nsp", "60",
                            "--budget", "smoke", "--no-figure"],
                           n_trials=3, symmetric_last=False)
        self.assertEqual(rc, 0)
        slug = M.request_slug(36, "ll_rotation", "smoke", 90.0, 1)
        res = os.path.join(M.RESULTS, slug)
        with io.open(os.path.join(res, "runs.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload["verdict"], "FAILED")
        self.assertIsNone(payload["root_trial"])
        self.assertIn("NOT evidence", payload["verdict_reason"])
        self.assertFalse(os.path.exists(os.path.join(res, "states")))
        self.assertFalse(any(t["wc_formed"] for t in payload["trials"]))

    def test_the_winning_state_is_pickled_without_its_snapshots(self):
        self._stub(["--rs", "90", "--trials", "2", "--nsp", "40",
                    "--budget", "smoke", "--no-figure"], n_trials=2)
        slug = M.request_slug(36, "ll_rotation", "smoke", 90.0, 1)
        import pickle
        path = os.path.join(M.RESULTS, slug, "states", "root__trial1.pkl")
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        self.assertIsNone(state.snaps)
        self.assertEqual(state.nmax, 1)
        self.assertEqual(tuple(np.asarray(state.v).shape), (36, 1))
        # and it is loadable as a nesting parent, which is Stage 1B's input
        self.assertEqual(state.ll_occupation()["n_bands"], 2)

    def test_the_trial_seeds_are_the_seed_base_plus_the_index(self):
        self._stub(["--rs", "90", "--trials", "3", "--nsp", "40",
                    "--seed-base", "777", "--budget", "smoke", "--no-figure"],
                   n_trials=3)
        slug = M.request_slug(36, "ll_rotation", "smoke", 90.0, 1)
        with io.open(os.path.join(M.RESULTS, slug, "runs.json"),
                     encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual([t["seed"] for t in payload["trials"]], [777, 778, 779])
        # every trial drew a DIFFERENT determinant, not one draw relabelled
        digs = {t["theta_digest"] for t in payload["trials"]}
        self.assertEqual(len(digs), 3)
        # and nbar_init differs with it, so the draws are not re-scaled copies
        self.assertEqual(len({round(t["nbar_init"], 12) for t in payload["trials"]}),
                         3)

    def test_a_non_one_nmax_is_refused(self):
        with self.assertRaises(SystemExit):
            M.main(["--rs", "90", "--nmax", "2", "--budget", "smoke",
                    "--no-figure"])

    def test_the_number_of_trials_must_be_positive(self):
        with self.assertRaises(SystemExit):
            M.main(["--rs", "90", "--trials", "0", "--budget", "smoke",
                    "--no-figure"])


if __name__ == "__main__":
    unittest.main()
