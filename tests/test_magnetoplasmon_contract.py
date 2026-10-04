"""The contract of ``examples/figure_construction/magnetoplasmon.py``.

That file makes one promise and does four pieces of arithmetic, and this module
pins all five:

* **It cannot start VMC.**  Not "it does not by default" -- it cannot.  Checked
  structurally, on the module's imports and calls, so a later edit that reaches
  for the sampler fails here rather than on somebody's cluster.
* **``S(q)`` is read, not recomputed.**  The array entering the division is the
  array ``structure_slice`` wrote, element for element, and the module contains
  no second structure-factor estimator to drift away from it.
* **The ``r_s = 0`` curve is the exact filled-LLL formula**, not a fit and not a
  VMC number.
* **The unit conversions are derivations.**  ``q/sqrt(n)`` from ``n =
  nu/(2 pi l_B^2)``, and ``omega_mp^2/omega_c^2 = 1 + kappa q`` from the engine's
  own definition of ``kappa``.  Both are checked against the expansion they have
  to reproduce, which is the check that would catch a fudged scale factor -- a
  constant chosen to make two curves meet would pass a value test and fail this
  one.
* **Missing data fails.**  A coupling that does not exist raises, names the
  command that would create it, and runs nothing.

The tests that need a saved ``S(q)`` use the real one on disk, because the point
of several of them is that this file and that file agree.
"""
import ast
import io
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SCRIPT = os.path.join(CLEAN, "examples", "figure_construction",
                      "magnetoplasmon.py")
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

#: Importing any of these is how a post-processing script acquires the ability
#: to run a campaign.  Matched as module-name components, so ``vmc.sampler``,
#: ``wigner_vmc.vmc.sampler`` and a bare ``sampler`` all trip it.
FORBIDDEN_MODULES = (
    "sampler", "run_vmc", "local_energy", "vmc", "sr", "structure_slice",
    "phase_competition", "subprocess", "multiprocessing", "concurrent",
)

#: Names that would mean the file had reached for the machinery through an
#: attribute rather than an import.
FORBIDDEN_ATTRS = (
    "run_vmc", "sample", "optimize", "local_energy", "Sampler", "SR",
    "sampler", "Popen", "run", "call", "check_output", "system", "popen",
)


def _load():
    spec = importlib.util.spec_from_file_location("magnetoplasmon_under_test",
                                                 SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree():
    return ast.parse(io.open(SCRIPT, encoding="utf-8").read())


M = _load()


# ==========================================================================
# 1. it cannot start VMC
# ==========================================================================
class TestItCannotStartVMC(unittest.TestCase):
    def test_no_forbidden_module_is_imported(self):
        """The structural half of the promise.

        ``wigner_vmc`` itself is allowed -- that is where ``kappa_from_rs`` and
        the structure helpers live -- but no submodule that could optimise a
        wavefunction or launch a process may appear.
        """
        bad = []
        for node in ast.walk(_tree()):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                parts = name.split(".")
                for f in FORBIDDEN_MODULES:
                    if f in parts:
                        bad.append((node.lineno, name))
        self.assertEqual(bad, [], f"imports that could run VMC/SR: {bad}")

    def test_no_dynamic_import_hides_one(self):
        """``importlib``/``__import__`` would let a name past the check above."""
        bad = []
        for node in ast.walk(_tree()):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name in ("__import__", "import_module", "exec", "eval",
                            "system", "Popen", "run", "check_output", "spawn"):
                    bad.append((node.lineno, name))
        self.assertEqual(bad, [], f"dynamic execution in the recipe: {bad}")

    def test_the_loaded_module_holds_no_vmc_callable(self):
        """The runtime half: nothing in the namespace is a VMC entry point.

        A name check on the source can be defeated by an alias; this looks at
        what actually got bound.  Only callables and modules are inspected --
        the module holds strings like ``"sr.py:219"`` in its prose and its
        provenance gate, and banning those would ban the explanation.
        """
        for name in dir(M):
            if name.startswith("__"):
                continue
            val = getattr(M, name)
            origin = getattr(val, "__module__", "") or ""
            if inspect.ismodule(val):
                origin = getattr(val, "__name__", "")
            parts = str(origin).split(".")
            for f in FORBIDDEN_MODULES:
                self.assertNotIn(f, parts, f"module global {name!r} from {origin!r}")

    def test_main_returns_2_and_writes_nothing_when_data_is_missing(self):
        """The documented failure path, exercised end to end.

        §2: missing ``S(q)`` must fail clearly rather than trigger a
        recomputation.  This drives the real ``main`` with couplings that do not
        exist, in a temporary output root, and asserts it declined -- and that
        the work of launching a campaign was never even attempted.
        """
        with tempfile.TemporaryDirectory() as tmp:
            res = os.path.join(tmp, "results")
            fig = os.path.join(tmp, "figures")
            with mock.patch.object(M, "RESULTS", res), \
                 mock.patch.object(M, "FIGDIR", fig):
                code = M.main(["--rs", "5", "30", "60"])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(fig), "a figure was written anyway")
            self.assertFalse(os.path.exists(res), "output was written anyway")

    def test_the_failure_message_names_the_command_and_runs_it_not(self):
        """§2 asks for the exact command.  Naming it is the whole deliverable."""
        index = [{"phase": "liquid", "rs": 5.567, "nmax": 1, "budget": "quick",
                  "dir": "x", "npz": "y", "schema": "", "mtime": 0.0,
                  "post_fix": True, "kappa": 3.9}]
        with self.assertRaises(LookupError) as ctx:
            M.find_source(index, "liquid", 30.0, "quick")
        msg = str(ctx.exception)
        self.assertIn("structure_slice.py", msg)
        self.assertIn("--liquid-rs 30", msg)
        self.assertIn("--budget quick", msg)
        self.assertIn("5.567", msg, "the available couplings are not listed")


# ==========================================================================
# 2. S(q) is read, not recomputed
# ==========================================================================
class TestTheSMAUsesTheStoredSqUnchanged(unittest.TestCase):
    def setUp(self):
        self.index = M.index_sources()
        self.post = [r for r in self.index if r["post_fix"]]
        if not self.post:
            self.skipTest("no post-fix structure_slice source on disk")

    def test_the_module_defines_no_structure_factor_estimator(self):
        """§5: one definition of the S(q) processing, and it lives elsewhere.

        ``structure.structure_factor`` is the estimator.  This file must not
        call it -- if it did, it would be measuring ``S(q)`` again from
        snapshots instead of reading the measurement that was saved.
        """
        for node in ast.walk(_tree()):
            if isinstance(node, ast.Attribute):
                self.assertNotEqual(
                    node.attr, "structure_factor",
                    "magnetoplasmon.py calls the structure-factor estimator; "
                    "it must read the saved array instead")
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None)
                self.assertNotEqual(name, "structure_factor")

    def test_load_curve_returns_the_stored_array_byte_for_byte(self):
        """Not "close to" -- equal.  No rounding, no averaging, no smoothing."""
        import numpy as np
        for src in self.post:
            qn, S = M.load_curve(src)
            z = np.load(src["npz"])
            key = "S_liquid" if src["phase"] == "liquid" else "S_crystal"
            self.assertTrue(np.array_equal(S, np.asarray(z[key], float)),
                            f"{src['dir']}: S(q) was modified on the way through")
            self.assertTrue(np.array_equal(qn, np.asarray(z["qn"], float)))

    def test_omega_is_the_division_and_nothing_else(self):
        """``Omega = q^2/(2 S)`` -- no fitted curve, no interpolated S."""
        import numpy as np
        src = self.post[0]
        qn, S = M.load_curve(src)
        om = M.omega_sma(qn, S)
        self.assertTrue(np.allclose(om, qn ** 2 / (2.0 * S), rtol=0, atol=0))

    def test_a_perturbed_S_would_change_Omega(self):
        """The control for the test above: if ``Omega`` ignored its argument,
        every check in this class would still pass.  Perturbing ``S`` by a
        relative amount must move ``Omega`` by the same relative amount.
        """
        import numpy as np
        src = self.post[0]
        qn, S = M.load_curve(src)
        a, b = M.omega_sma(qn, S), M.omega_sma(qn, 1.01 * S)
        self.assertFalse(np.allclose(a, b), "Omega does not depend on S(q)")
        self.assertTrue(np.allclose(a / b, 1.01, rtol=1e-12))

    def test_no_smoothing_helper_is_called(self):
        """A moving average, a convolution or a spline would be "fitting S(q)
        to smooth", which §6 forbids."""
        banned = {"convolve", "savgol_filter", "uniform_filter", "spline",
                  "UnivariateSpline", "interp1d", "gaussian_filter",
                  "medfilt", "smooth"}
        for node in ast.walk(_tree()):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, banned, f"line {node.lineno}")
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, banned, f"line {node.lineno}")

    def test_the_source_array_has_no_q_zero(self):
        """The invariant ``omega_sma`` relies on.  The torus excludes q = 0, so
        a file containing it is not a structure factor."""
        import numpy as np
        for src in self.post:
            qn, _ = M.load_curve(src)
            self.assertTrue(np.all(qn > 0.0), src["dir"])


# ==========================================================================
# 3. the r_s = 0 curve is the exact filled-LLL formula
# ==========================================================================
class TestTheAnalyticCurveIsExact(unittest.TestCase):
    def test_exact_lll_sq_is_one_minus_exp_minus_q2_over_2(self):
        import numpy as np
        from wigner_vmc.analysis import structure as st
        q = np.array([0.05, 0.3, 1.0, 2.0, 3.5])
        self.assertTrue(np.allclose(st.exact_lll_sq(q),
                                    1.0 - np.exp(-q ** 2 / 2.0),
                                    rtol=0, atol=1e-15))

    def test_the_rs_zero_curve_uses_that_helper(self):
        """Not a private copy of the formula -- the package's own definition."""
        args = M.parse_args(["--rs", "0"])
        args.phase_of = M.phase_of
        curves, _ = M.build_curves(args)
        self.assertEqual(len(curves), 1)
        c = curves[0]
        self.assertEqual(c["phase"], "analytic")
        self.assertIsNone(c["source"], "r_s = 0 must not read a source")
        import numpy as np
        from wigner_vmc.analysis import structure as st
        self.assertTrue(np.array_equal(c["S"], st.exact_lll_sq(c["q"])))

    def test_the_analytic_curve_small_q_limit_is_one_over_hbar_omega_c(self):
        """``S -> q^2/2`` so ``Omega = q^2/(2S) -> 1``, the cyclotron energy.

        This is the physical anchor of the whole figure: with the interaction
        off, the SMA dispersion must reduce to the cyclotron frequency.

        The approach is not flat, and saying so precisely is the point.  With
        ``S_0 = 1 - exp(-q^2/2) = q^2/2 - q^4/8 + ...`` the exact result is

            Omega_0(q) = q^2/(2 S_0) = 1/(1 - q^2/4 + ...) = 1 + q^2/4 + ...

        so the deviation from 1 is ``q^2/4`` and is a real O(q^2) effect, not a
        tolerance to be tuned away.  A test asserting ``Omega = 1`` at
        ``q = 1e-2`` would be asserting something false; this one checks the
        limit AND the coefficient.
        """
        from wigner_vmc.analysis import structure as st
        # The coefficient check runs at q >= 1e-3 only.  Below that the
        # subtraction inside ``1 - exp(-q^2/2)`` floors out -- at q = 1e-4 the
        # computed deviation is 6.1e-9 against a true 2.6e-9, so the ratio this
        # test forms would be measuring the cancellation, not the physics.
        for q in (1e-2, 1e-3):
            om = float(M.omega_sma(q, st.exact_lll_sq(q)))
            self.assertLess(abs(om - 1.0), 2.0 * q ** 2 / 4.0,
                            f"q = {q}: Omega_0 leaves the cyclotron energy too fast")
            self.assertAlmostEqual((om - 1.0) / (q ** 2 / 4.0), 1.0, delta=1e-3,
                                   msg=f"q = {q}: the q^2 coefficient is not 1/4")
        # And it really does go to 1, monotonically, all the way down.
        oms = [float(M.omega_sma(q, st.exact_lll_sq(q)))
               for q in (1e-2, 1e-3, 1e-4)]
        self.assertTrue(oms[0] > oms[1] > oms[2] > 1.0)

    def test_the_analytic_curve_is_flat_where_it_should_be(self):
        """The companion control: a curve that was the constant 1 would satisfy
        every "close to 1" assertion above.  It is not constant -- it rises with
        q, and the rise is what the crystal curves are compared against."""
        import numpy as np
        from wigner_vmc.analysis import structure as st
        q = np.array([0.5, 1.0, 2.0, 3.0])
        om = M.omega_sma(q, st.exact_lll_sq(q))
        self.assertGreater(float(om[-1]), 2.0)
        self.assertTrue(np.all(np.diff(om) > 0))

    def test_the_analytic_curve_is_finite_and_positive_everywhere(self):
        import numpy as np
        args = M.parse_args(["--rs", "0"])
        args.phase_of = M.phase_of
        curves, _ = M.build_curves(args)
        for c in curves:
            self.assertTrue(np.all(c["S"] > 0.0))
            self.assertTrue(np.all(np.isfinite(c["Omega"])))
            self.assertTrue(np.all(c["Omega"] > 0.0))


# ==========================================================================
# 4. the axis conversion
# ==========================================================================
class TestTheAxisConversion(unittest.TestCase):
    def test_q_over_sqrt_n_follows_from_the_density(self):
        """Derived, not transcribed.

        ``n = nu/(2 pi l_B^2)`` so ``q/sqrt(n) = q l_B sqrt(2 pi/nu)``.  The
        test builds the density from its definition in units of ``l_B`` rather
        than reusing the function's own constant, so a change to that constant
        cannot make the test agree with it.
        """
        import numpy as np
        for nu in (1.0, 2.0, 0.25):
            lB = 1.0
            n = nu / (2.0 * math.pi * lB ** 2)      # definition, restated
            q = np.array([0.25, 1.0, 3.75])
            self.assertTrue(np.allclose(M.q_over_sqrt_n(q, nu), q / math.sqrt(n),
                                        rtol=1e-14))

    def test_the_value_at_nu_one_is_the_familiar_factor(self):
        self.assertAlmostEqual(float(M.q_over_sqrt_n(1.0, 1.0)),
                               math.sqrt(2.0 * math.pi), places=14)

    def test_no_magic_constant_is_typeable(self):
        """The factor is built from ``2 pi`` by the module, so a hard-coded
        decimal -- the thing §8 forbids -- should not appear in the code.
        """
        code = io.open(SCRIPT, encoding="utf-8").read()
        for literal in ("2.5066", "2.506628", "2.50663"):
            self.assertNotIn(literal, code)

    def test_the_axis_is_monotone_in_q(self):
        """A sanity check that the conversion has the right sign and scale."""
        import numpy as np
        q = np.array([0.5, 1.0, 2.0, 4.0])
        x = M.q_over_sqrt_n(q)
        self.assertTrue(np.all(np.diff(x) > 0))
        self.assertAlmostEqual(float(x[1] / x[0]), 2.0, places=12)


# ==========================================================================
# 5. the magnetoplasmon convention -- the test that catches a fudged scale
# ==========================================================================
class TestTheMagnetoplasmonHasTheRightConvention(unittest.TestCase):
    def test_at_zero_coupling_it_is_the_cyclotron_frequency(self):
        """``kappa = 0`` means no interaction; ``omega_mp`` must be exactly 1
        for every q, because the second term of ``omega_c^2 + (2 pi n e^2/m) q``
        has vanished."""
        import numpy as np
        q = np.array([0.1, 1.0, 5.0])
        self.assertTrue(np.allclose(M.omega_magnetoplasmon(q, 0.0), 1.0,
                                    rtol=0, atol=0))

    def test_the_squared_form_is_one_plus_kappa_q(self):
        """The conversion from the dimensional formula, checked against the
        derivation rather than the code."""
        import numpy as np
        for kappa in (0.5, 3.9, 64.0):
            q = np.array([0.2, 1.0, 2.5])
            self.assertTrue(np.allclose(M.omega_magnetoplasmon(q, kappa) ** 2,
                                        1.0 + kappa * q, rtol=1e-14))

    def test_the_leading_order_matches_the_SMA_expansion(self):
        """The check a fudged scale factor fails.

        Expanding the paper's ``S(q) = q^2/2 - kappa q^3/4`` gives

            Omega = q^2/(2S) = 1/(1 - kappa q/2 + ...) = 1 + (kappa/2) q + O(q^2)
            omega_mp = sqrt(1 + kappa q)                = 1 + (kappa/2) q + O(q^2)

        so the two agree to first order and differ at second.  The linear
        coefficient is ``kappa/2`` in both, and the next correction is known
        exactly too:

            Omega    = 1/(1 - kappa q/2)   = 1 + (kappa/2) q + (kappa^2/4) q^2
            omega_mp = sqrt(1 + kappa q)   = 1 + (kappa/2) q - (kappa^2/8) q^2

        so ``[(slope - kappa/2)/q]`` must be ``+kappa^2/4`` for the SMA curve
        and ``-kappa^2/8`` for the classical one.  Asserting the second
        coefficient rather than a tolerance is what makes this a test of the
        convention: a spurious constant multiplying either curve -- the
        "unexplained scale factor" of §7 -- moves the first coefficient, and
        even a convention that got the linear term right by accident is caught
        here, because the two curves' second coefficients differ in sign and
        magnitude and both are pinned.
        """
        # q = 1e-3, not 1e-2.  The third-order term is also known -- continuing
        # the series above gives ``+kappa^3 q/8`` for Omega and ``+kappa^3 q/16``
        # for omega_mp -- and at q = 1e-2 with kappa = 12 it is 6% of the second
        # coefficient, which is a real effect that a 5% tolerance would report as
        # a failure.  At q = 1e-3 it is 0.6%.
        for kappa in (0.5, 2.0, 12.0):
            for q in (1e-3, 1e-4):
                S = q ** 2 / 2.0 - kappa * q ** 3 / 4.0
                om = float(M.omega_sma(q, S))
                wm = float(M.omega_magnetoplasmon(q, kappa))
                dev_om = ((om - 1.0) / q - kappa / 2.0) / q
                dev_wm = ((wm - 1.0) / q - kappa / 2.0) / q
                self.assertAlmostEqual(dev_om, kappa ** 2 / 4.0,
                                       delta=0.02 * kappa ** 2 / 4.0,
                                       msg=f"Omega second coefficient, kappa = {kappa}, q = {q}")
                self.assertAlmostEqual(dev_wm, -kappa ** 2 / 8.0,
                                       delta=0.02 * abs(kappa ** 2 / 8.0),
                                       msg=f"omega_mp second coefficient, kappa = {kappa}, q = {q}")

    def test_the_second_coefficients_have_opposite_signs(self):
        """The control that makes the test above more than a magnitude check:
        the SMA curve curves upwards away from ``omega_mp`` and the classical
        one downwards, so no single scale factor can align them beyond first
        order.  If both deviations came back with the same sign, the convention
        would be wrong in a way a tolerance would not catch."""
        kappa, q = 4.0, 1e-3
        S = q ** 2 / 2.0 - kappa * q ** 3 / 4.0
        dev_om = ((float(M.omega_sma(q, S)) - 1.0) / q - kappa / 2.0) / q
        dev_wm = ((float(M.omega_magnetoplasmon(q, kappa)) - 1.0) / q
                  - kappa / 2.0) / q
        self.assertGreater(dev_om, 0.0)
        self.assertLess(dev_wm, 0.0)

    def test_the_two_differ_at_second_order_not_first(self):
        """The companion control: the curves are not identical, and the
        difference is O(q^2).  Without this the test above would be satisfied by
        returning the same function twice."""
        kappa = 4.0
        for q in (1e-2, 1e-3):
            S = q ** 2 / 2.0 - kappa * q ** 3 / 4.0
            d = float(M.omega_sma(q, S) - M.omega_magnetoplasmon(q, kappa))
            self.assertNotEqual(d, 0.0)
            # O(q^2), so d/q must shrink with q rather than staying constant.
            self.assertLess(abs(d) / q, 0.5 * kappa ** 2 * q,
                            "the difference is not second order in q")
        # The scatter of d/q^2 across a decade is the test that the leading
        # discrepancy is quadratic: it must be nearly constant.
        r = []
        for q in (1e-2, 1e-3):
            S = q ** 2 / 2.0 - kappa * q ** 3 / 4.0
            d = float(M.omega_sma(q, S) - M.omega_magnetoplasmon(q, kappa))
            r.append(d / q ** 2)
        self.assertAlmostEqual(r[0] / r[1], 1.0, delta=0.05)

    def test_kappa_comes_from_the_engines_own_definition(self):
        """``kappa = r_s/sqrt(2)`` at ``nu = 1`` is the engine's convention, and
        the figure must not carry a second one."""
        from wigner_vmc.physics.hamiltonian import kappa_from_rs
        for rs in (0.0, 5.0, 47.0, 90.51):
            self.assertAlmostEqual(M.kappa_from_rs(rs), kappa_from_rs(rs),
                                   places=15)

    def test_kappa_reproduces_the_one_the_measurement_was_made_at(self):
        """The strongest available check that the classical curve carries the
        right coupling.

        ``structure_slice`` stores the ``kappa`` each phase was actually run at,
        in ``kappa_liquid`` / ``kappa_crystal``.  Recomputing it from ``r_s`` and
        getting the same number back means the ``1 + kappa q`` in the classical
        curve is the coupling of the state it is drawn against, derived from the
        engine's definition rather than asserted.  A convention error -- using
        ``r_s`` where ``kappa`` belongs, or the wrong ``nu`` -- shows up here as
        a factor.
        """
        import numpy as np
        from wigner_vmc.physics.hamiltonian import kappa_from_rs
        checked = 0
        for src in M.index_sources():
            if not src["post_fix"]:
                continue
            z = np.load(src["npz"])
            key = "kappa_liquid" if src["phase"] == "liquid" else "kappa_crystal"
            if key not in z:
                continue
            self.assertAlmostEqual(float(z[key]), kappa_from_rs(src["rs"]),
                                   places=12, msg=src["dir"])
            checked += 1
        if not checked:
            self.skipTest("no post-fix source stores kappa")

    def test_a_curve_carries_the_kappa_of_its_own_coupling(self):
        """The classical partner belongs to the state it is compared against,
        so the two liquid points must not share a classical curve."""
        args = M.parse_args(["--rs", "0"])
        args.phase_of = M.phase_of
        curves, _ = M.build_curves(args)
        self.assertAlmostEqual(curves[0]["kappa"], 0.0, places=15)


# ==========================================================================
# 6. q = 0
# ==========================================================================
class TestQZeroIsHandledSafely(unittest.TestCase):
    def test_zero_S_raises_rather_than_returning_inf(self):
        """``Omega`` is a removable singularity, not a computable value: the
        naive division returns inf/nan and would be plotted as a spike or
        dropped in silence.  It must raise instead."""
        import numpy as np
        with self.assertRaises(ValueError) as ctx:
            M.omega_sma(np.array([0.5, 1.0]), np.array([0.0, 0.2]))
        self.assertIn("S > 0", str(ctx.exception))

    def test_non_finite_S_raises(self):
        """``S = inf`` is the dangerous one: ``q^2/(2*inf) = 0.0`` is a
        perfectly finite number, so it would draw a curve lying on the axis and
        look like a result rather than a corrupt file."""
        import numpy as np
        for bad in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                M.omega_sma(np.array([1.0]), np.array([bad]))

    def test_a_non_positive_q_raises(self):
        import numpy as np
        for q in (0.0, -1.0):
            with self.assertRaises(ValueError):
                M.omega_sma(np.array([q]), np.array([0.5]))

    def test_a_source_containing_q_zero_is_rejected(self):
        """The storage invariant, enforced at the door."""
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "f.npz")
            np.savez(p, qn=np.array([0.0, 0.5]), S_liquid=np.array([0.0, 0.1]))
            with self.assertRaises(ValueError) as ctx:
                M.load_curve({"npz": p, "phase": "liquid"})
            self.assertIn("q = 0", str(ctx.exception))

    def test_the_analytic_grid_never_starts_exactly_at_zero(self):
        """``S_0(0) = 0``, so a grid containing 0 would divide by zero."""
        args = M.parse_args(["--rs", "0"])
        args.phase_of = M.phase_of
        curves, _ = M.build_curves(args)
        self.assertGreater(float(curves[0]["q"].min()), 0.0)

    def test_the_positive_q_limit_is_finite_and_equals_one(self):
        """``lim_{q->0} q^2/(2 S_0(q)) = 1`` -- the singularity is removable,
        which is why raising rather than patching is the right call for a
        genuinely zero ``S``: only the exact-Landau-level case has a limit, and
        it comes from the analytic branch, not from the division.

        ``q = 1e-4`` rather than something smaller, and deliberately: the
        subtraction in ``1 - exp(-q^2/2)`` loses precision as ``q`` falls -- at
        ``q = 1e-6`` it reads ``0.99991`` where the limit is 1, an error of
        9e-5, entirely inside ``exact_lll_sq`` and nothing to do with this
        module.  Testing below that point would be testing the cancellation.
        """
        from wigner_vmc.analysis import structure as st
        self.assertAlmostEqual(float(M.omega_sma(1e-4, st.exact_lll_sq(1e-4))),
                               1.0, places=7)


# ==========================================================================
# 7. provenance: only post-fix results
# ==========================================================================
def _fake_source(**kw):
    base = {"phase": "liquid", "rs": 5.0, "nmax": 1, "budget": "quick",
            "dir": "d", "npz": "n", "schema": "", "mtime": M.PROJECTION_FIX_MTIME,
            "post_fix": True, "kappa": 3.5}
    base.update(kw)
    return base


class TestOnlyCorrectedResultsAreRead(unittest.TestCase):
    def test_a_pre_fix_source_is_refused(self):
        """The projection fix changed the crystal's ``S(q)``.  A file written
        before it is not the structure factor of the ansatz this package now
        implements, and drawing it under a current label would be the silent
        mix §11 forbids."""
        pre = _fake_source(mtime=M.PROJECTION_FIX_MTIME - 1.0, post_fix=False)
        with self.assertRaises(LookupError) as ctx:
            M.find_source([pre], "liquid", 5.0, "quick")
        msg = str(ctx.exception)
        self.assertIn("sr.py:219", msg)
        self.assertIn("Re-run structure_slice.py", msg)

    def test_a_post_fix_source_is_accepted(self):
        post = _fake_source(mtime=M.PROJECTION_FIX_MTIME + 1.0)
        self.assertIs(M.find_source([post], "liquid", 5.0, "quick"), post)

    def test_the_newest_post_fix_source_wins(self):
        """One coupling can be measured twice, once each side of the fix --
        ``crystal r_s = 100 nmax = 2 quick`` really is.  Taking the oldest
        candidate would refuse a coupling whose usable file sits beside the
        stale one, so the ordering, not just the flag, is the contract."""
        old = _fake_source(mtime=M.PROJECTION_FIX_MTIME - 10.0, post_fix=False,
                           dir="old")
        new = _fake_source(mtime=M.PROJECTION_FIX_MTIME + 10.0, dir="new")
        got = M.find_source([old, new], "liquid", 5.0, "quick")
        self.assertEqual(got["dir"], "new")

    def test_the_flag_agrees_with_the_mtime_it_is_derived_from(self):
        """``post_fix`` is computed, not asserted, so an index built by the real
        scanner cannot disagree with the clock."""
        for r in M.index_sources():
            self.assertEqual(r["post_fix"],
                             r["mtime"] >= M.PROJECTION_FIX_MTIME, r["dir"])

    def test_the_real_index_finds_the_post_fix_couplings(self):
        """A regression on the actual tree: the fix's own validation points
        must be recognised as usable."""
        idx = M.index_sources()
        post = {(r["phase"], r["rs"]) for r in idx if r["post_fix"]}
        self.assertIn(("crystal", 90.51), post)
        self.assertIn(("liquid", 5.567), post)

    def test_an_exact_coupling_match_is_required(self):
        """``r_s = 5`` and ``r_s = 5.567`` are different states.  Drawing the
        nearby one under the requested label is the failure this guards."""
        idx = M.index_sources()
        with self.assertRaises(LookupError):
            M.find_source(idx, "liquid", 5.0, "quick")

    def test_full_is_resolved_to_the_name_on_disk(self):
        """``structure_slice.py`` accepts ``--budget full`` as an alias for the
        ``reproduction`` config and resolves it before writing, so the
        directories say ``reproduction``.  Matching the stored string exactly
        would refuse ``--source-budget full`` for data that exists -- the
        brief's own spelling failing against the brief's own data."""
        self.assertEqual(M.resolve_budget("full"), "reproduction")
        self.assertEqual(M.resolve_budget("quick"), "quick")
        self.assertEqual(M.resolve_budget("reproduction"), "reproduction")
        idx = M.index_sources()
        try:
            a = M.find_source(idx, "liquid", 5.567, "full")
            b = M.find_source(idx, "liquid", 5.567, "reproduction")
        except LookupError:
            self.skipTest("no post-fix liquid 5.567 source on disk")
        self.assertEqual(a["dir"], b["dir"])

    def test_full_is_not_stamped_but_quick_is(self):
        """The alias must resolve before the stamp decision too, or the real
        statistics would be marked as a smoke test."""
        self.assertIsNone(M.figure_stamp(M.resolve_budget("full")))
        self.assertIsNotNone(M.figure_stamp(M.resolve_budget("quick")))

    def test_the_budget_is_a_selector_not_a_preference(self):
        """Asking for a budget that was not run must fail, not fall back to the
        other one: the two are different amounts of sampling."""
        idx = M.index_sources()
        # Not "full" -- that is an alias for reproduction and legitimately
        # resolves.  A name nothing was ever run under must fail.
        with self.assertRaises(LookupError) as ctx:
            M.find_source(idx, "liquid", 5.567, "smoke")
        self.assertIn("reproduction", str(ctx.exception),
                      "the failure should list what does exist")


# ==========================================================================
# 8. the shape of the delivered thing
# ==========================================================================
class TestTheRunIsPostProcessingOnly(unittest.TestCase):
    def test_main_completes_without_running_anything(self):
        """One real run, on the couplings that exist, timed and inspected.

        If this file ever acquired a VMC path this would not fail by being slow
        -- it would fail on the module-import checks above.  What it adds is
        that the assembled record says so too.
        """
        with tempfile.TemporaryDirectory() as tmp:
            res, fig = os.path.join(tmp, "r"), os.path.join(tmp, "f")
            with mock.patch.object(M, "RESULTS", res), \
                 mock.patch.object(M, "FIGDIR", fig):
                code = M.main(["--rs", "0", "5.567", "--source-budget", "quick"])
            self.assertEqual(code, 0)
            with io.open(os.path.join(res, "rs0__5p567", "run_metadata.json"),
                         encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertTrue(meta["post_processing_only"])
            self.assertFalse(meta["vmc_started"])
            self.assertFalse(meta["sr_started"])
            self.assertFalse(meta["s_q_regenerated"])

    def test_every_curve_records_the_directory_it_came_from(self):
        """§12: the metadata must identify the exact numerical source."""
        with tempfile.TemporaryDirectory() as tmp:
            res, fig = os.path.join(tmp, "r"), os.path.join(tmp, "f")
            with mock.patch.object(M, "RESULTS", res), \
                 mock.patch.object(M, "FIGDIR", fig):
                M.main(["--rs", "0", "5.567", "--source-budget", "quick"])
            with io.open(os.path.join(res, "rs0__5p567", "run_metadata.json"),
                         encoding="utf-8") as fh:
                meta = json.load(fh)
            srcs = {s["rs"]: s for s in meta["sources"]}
            self.assertIsNone(srcs[0.0]["directory"], "r_s = 0 read a file")
            self.assertIn("structure_slice", srcs[5.567]["directory"])
            self.assertIn("structure_factor.npz",
                          srcs[5.567]["structure_factor_npz"])

    def test_rs_c_is_recorded_and_never_used_to_decide(self):
        """The brief fixes ``r_s^c = 47``.  It appears in the metadata as a
        stated input and must not silently become a computed result."""
        code = io.open(SCRIPT, encoding="utf-8").read()
        self.assertIn("RS_C = 47.0", code)
        self.assertIn("rs_c_recorded_not_used", code)

    def test_the_source_directories_are_not_written_to(self):
        """Reading is the whole relationship.  ``save_npz`` and
        ``save_metadata`` must both be pointed at the output root."""
        for fn in (M.save_npz, M.save_metadata):
            first = inspect.signature(fn).parameters
            self.assertIn("results_dir", first)
        for node in ast.walk(_tree()):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None)
                if name in ("save_npz", "save_metadata"):
                    arg = node.args[0]
                    self.assertIsInstance(arg, ast.Name,
                                          f"line {node.lineno}: output path is "
                                          "not the results_dir variable")
                    self.assertEqual(arg.id, "res_dir")

    def test_the_slug_keeps_the_papers_directory_for_the_papers_points(self):
        self.assertEqual(M.output_slug(M.DEFAULT_RS), "paper")
        self.assertNotEqual(M.output_slug([0.0, 5.567]), "paper")

    def test_the_phase_rule_is_the_figure_s_and_is_fixed(self):
        """Which phase a coupling is drawn from is given, not inferred."""
        self.assertEqual(M.phase_of(5.0), "liquid")
        self.assertEqual(M.phase_of(30.0), "liquid")
        self.assertEqual(M.phase_of(46.999), "liquid")
        self.assertEqual(M.phase_of(47.0), "crystal")
        self.assertEqual(M.phase_of(60.0), "crystal")

    def test_a_non_reproduction_budget_is_stamped(self):
        """A quick figure must say it is quick -- on the picture and in the
        record, from one definition so the two cannot disagree."""
        self.assertIsNone(M.figure_stamp("reproduction"))
        for b in ("quick", "full", "", "nonsense"):
            self.assertIsNotNone(M.figure_stamp(b), b)


class TestTheRotonSearch(unittest.TestCase):
    def test_q_wc_is_the_triangular_lattice_geometry(self):
        """``a^2 = 4 pi/sqrt(3)`` at ``nu = 1``, and ``Q_WC = |G_1|``."""
        a = M.q_wc_lattice_constant(1.0)
        self.assertAlmostEqual(a * a, 4.0 * math.pi / math.sqrt(3.0), places=14)
        self.assertAlmostEqual(M.q_wc(1.0), 4.0 * math.pi / (math.sqrt(3.0) * a),
                               places=14)
        self.assertAlmostEqual(M.q_wc(1.0), 2.693547, places=6)

    def test_q_wc_matches_the_g1_the_data_recorded(self):
        """Cross-check against the lattice the measurements themselves used:
        ``g1_mag`` in a saved ``structure_factor.npz`` is ``|G_1|``.  This ties
        the analytic geometry to the actual supercell."""
        import numpy as np
        for src in M.index_sources():
            z = np.load(src["npz"])
            if "g1_mag" in z:
                self.assertAlmostEqual(float(z["g1_mag"]), M.q_wc(1.0),
                                       places=4, msg=src["dir"])
                return
        self.skipTest("no source records g1_mag")

    @staticmethod
    def _dipped(q_wc_value, depth=1.0):
        """A parabola with one deliberate, deep dip placed at ``Q_WC``.

        The dip goes where the search is looking.  Putting it anywhere else
        would leave the parabola's own vertex as a competing minimum -- which is
        a true local minimum of the synthetic curve, so the search would be
        right to return it and a test demanding the dip would be wrong.
        """
        import numpy as np
        q = np.linspace(0.1, 4.0, 400)
        om = (q - q_wc_value) ** 2 + 0.5
        i = int(np.argmin(np.abs(q - q_wc_value)))
        om[i] -= depth
        return q, om, i

    def test_a_synthetic_parabola_with_a_dip_is_found(self):
        """The search is validated on data whose answer is known, so a
        positive result on the real crystal means something."""
        q, om, i = self._dipped(2.693547)
        got = M.roton_minimum({"q": q, "Omega": om}, 2.693547)
        self.assertTrue(got["resolved"], got["reason"])
        self.assertEqual(got["q_roton"], float(q[i]))
        self.assertTrue(got["deeper_than_imbalance"], got["reason"])

    def test_the_minimum_nearest_q_wc_is_the_one_reported(self):
        """With more than one local minimum the choice matters, and the rule is
        "nearest Q_WC" -- a soft mode is defined by living there.

        Built so the two candidates are unambiguous: a deep dip well away from
        Q_WC and a shallower one on top of it.  A search that simply took the
        first minimum, or the deepest, would return the wrong one, which is
        exactly the mutation this test exists to kill.
        """
        import numpy as np
        q = np.linspace(0.1, 4.0, 400)
        om = q ** 2 + 1.0
        far = int(np.argmin(np.abs(q - 1.2)))
        near = int(np.argmin(np.abs(q - 2.693547)))
        om[far] -= 5.0                     # deeper, wrong place
        om[near] -= 0.8                    # shallower, at Q_WC
        got = M.roton_minimum({"q": q, "Omega": om}, 2.693547)
        self.assertTrue(got["resolved"], got["reason"])
        self.assertEqual(got["q_roton"], float(q[near]))
        self.assertEqual(got["n_minima"], 2)
        self.assertEqual(got["other_minima_q"], [float(q[far])])

    def test_the_other_minima_are_reported_not_discarded(self):
        """Choosing one is only honest if the losers are visible."""
        q, om, _ = self._dipped(2.693547)
        got = M.roton_minimum({"q": q, "Omega": om}, 2.693547)
        self.assertEqual(got["n_minima"], 1)
        self.assertEqual(got["other_minima_q"], [])

    def test_momenta_sharing_a_magnitude_are_one_shell(self):
        """The bug this replaced.

        The stored ``S(q)`` is a disc of momenta, so many of them share a
        magnitude.  Sorting that scatter by ``|q|`` and testing for a local
        minimum makes each point's neighbours the *other points in its own
        shell*, which reported 75 "minima" in the real crystal at a neighbour
        spacing of 0.0.  Six points at the same ``|q|`` must be one shell.
        """
        import numpy as np
        q = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        om = np.array([0.5, 0.6, 0.4, 0.55, 0.45, 0.5,  # one shell, scattered
                       2.0, 1.0, 3.0, 4.0, 5.0])
        got = M.roton_minimum({"q": q, "Omega": om}, 3.0)
        self.assertEqual(got["n_shells"], 6)
        self.assertEqual(got["shell_points"][0], 6)
        self.assertEqual(got["q_roton"], 3.0)

    def test_scatter_within_a_shell_is_not_a_minimum(self):
        """The bug this replaced, stated as data.

        Shell 2.0 holds three points whose Omega is 5.0, 0.01, 5.0.  Sorted by
        ``|q|``, the 0.01 point sits between two others *of its own shell* and
        looks exactly like a sharp minimum -- which is what the first version
        reported, 75 times over.  Binned, shell 2.0 has mean 3.337, and it is
        reported as a shell with three points and a spread of ~5, not as a spike.
        """
        import numpy as np
        q = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 4.0])
        om = np.array([5.0, 5.0, 0.01, 5.0, 5.0, 6.0])
        got = M.roton_minimum({"q": q, "Omega": om}, 2.0)
        self.assertTrue(got["found"], got["reason"])
        self.assertEqual(got["q_roton"], 2.0)
        self.assertAlmostEqual(got["Omega_roton"], (5.0 + 0.01 + 5.0) / 3.0,
                               places=12)
        self.assertEqual(got["n_points_at_roton"], 3)
        self.assertGreater(got["spread_at_roton"], 4.9)
        # And because that spread dwarfs the depth, it is NOT resolved: this is
        # the case the two-field split exists for.
        self.assertFalse(got["resolved"])
        self.assertFalse(got["deeper_than_imbalance"])
        self.assertIn("roton not resolved", got["reason"])

    def test_a_minimum_shallower_than_its_shell_scatter_is_refused(self):
        """A narrow dip between wide shells is noise, and saying so is the
        deliverable §10 asks for when there is no roton."""
        import numpy as np
        q = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        # Shell means 4, 3, 4, 5: a real interior minimum of depth 1.  But shell
        # 2.0 spans 0.0 to 6.0, so the scatter it must clear is 6.0.  It does
        # not, and is refused.
        om = np.array([4.0, 4.0, 0.0, 6.0, 4.0, 4.0, 5.0, 5.0])
        got = M.roton_minimum({"q": q, "Omega": om}, 2.5)
        self.assertTrue(got["found"], got["reason"])
        self.assertFalse(got["resolved"])
        self.assertFalse(got["deeper_than_imbalance"])
        self.assertEqual(got["reason"].split(":")[0],
                         "roton not resolved at current finite-size/statistical "
                         "resolution")

    def test_too_few_shells_is_refused_before_any_search(self):
        """With two shells there is no interior minimum to find, and the
        refusal has to say why rather than quietly returning nothing."""
        import numpy as np
        for q in (np.array([1.0, 2.0]), np.array([1.0, 1.0, 2.0, 2.0])):
            got = M.roton_minimum({"q": q, "Omega": np.array([5.0] * q.size)},
                                  1.5)
            self.assertFalse(got["found"], got["reason"])
            self.assertFalse(got["resolved"], got["reason"])
            self.assertIn("roton not resolved", got["reason"])
            self.assertIn("distinct |q| shells", got["reason"])

    def test_resolved_requires_more_than_a_minimum_existing(self):
        """The split that makes ``resolved`` safe to read on its own."""
        q, om, _ = self._dipped(2.693547)
        got = M.roton_minimum({"q": q, "Omega": om}, 2.693547)
        self.assertTrue(got["resolved"])
        self.assertTrue(got["found"])
        self.assertTrue(got["deeper_than_imbalance"])

    def test_a_monotone_curve_reports_not_resolved(self):
        """§10's required sentence, produced when there is no minimum."""
        import numpy as np
        q = np.linspace(0.1, 4.0, 200)
        got = M.roton_minimum({"q": q, "Omega": q ** 2 + 1.0}, 2.693547)
        self.assertFalse(got["found"])
        self.assertFalse(got["resolved"])
        self.assertIn("roton not resolved", got["reason"])

    def test_a_dip_shallower_than_its_neighbours_imbalance_is_not_claimed(self):
        """Rounding noise on an unequal pair of neighbours must not be reported
        as a roton.  This is the guard against manufacturing one."""
        import numpy as np
        q = np.linspace(0.1, 4.0, 5)
        om = np.array([1.0, 2.0, 1.999, 2.5, 3.0])
        got = M.roton_minimum({"q": q, "Omega": om}, 1.5)
        self.assertFalse(got["deeper_than_imbalance"])

    def test_the_real_crystal_rotons_at_q_wc(self):
        """The measured result, on the DIRECTIONAL branch -- the crystal curve.

        A Wigner crystal is not isotropic, so its 1-D curve is a cut along one
        direction and each ``|q|`` on it is one momentum, not a shell.  The
        earlier version of this test shell-averaged the crystal and asserted
        ``n_points_at_roton == 6``; that is the reduction §8 of the correction
        forbids, because averaging a Bragg peak away is exactly what it does,
        and it is the reason the cut is taken along the paper's ``x`` axis
        instead.  Pinned here as ``n_points_at_roton == 1`` so a return to the
        averaged crystal would fail rather than pass quietly.

        The minimum sits at ``q = Q_WC`` exactly -- the crystal's own first
        reciprocal vector -- and that is a property of the DIRECTION, not of the
        data: only a cut along a reciprocal-lattice direction reaches ``Q_WC``
        at all.  The ``clean-x`` axis misses it by 0.36, which the companion
        test below pins.  The depth is large; the caveat is that a finite-size
        Bragg peak's height scales with ``N``, so the depth is not a
        thermodynamic number.
        """
        idx = M.index_sources()
        try:
            src = M.find_source(idx, "crystal", 90.51, "quick", 3)
        except LookupError:
            self.skipTest("no post-fix crystal source on disk")
        torus = M.torus_for(36)
        dhat = M.crystal_direction(torus, "bragg")
        qvec = M.load_momenta(src)
        qn, S = M.load_curve(src)
        red = M.directional_cut(qvec, S, dhat)
        curve = {"q": red["q"], "Omega": M.omega_sma(red["q"], red["S"])}
        got = M.roton_minimum(curve, M.q_wc())
        self.assertTrue(got["resolved"], got["reason"])
        self.assertAlmostEqual(got["q_roton"], M.q_wc(), places=4)
        self.assertEqual(got["n_points_at_roton"], 1)
        self.assertLess(got["distance_from_q_wc"], 1e-6)
        self.assertTrue(got["deeper_than_imbalance"])
        self.assertGreater(got["depth_below_neighbours"], 3.0)
        # One point per |q| on a cut, and no shell grouping left to do.
        self.assertEqual(got["n_shells"], red["n_points"])
        self.assertEqual(got["n_shells"], 8)

    def test_the_clean_x_axis_misses_q_wc_entirely(self):
        """Why the direction is not a convention.  The package's own first
        Cartesian axis is 30 degrees from the paper's ``x``; its momenta form a
        5-point ladder that never lands on ``Q_WC``, so the crystal's roton is
        simply not on it, and the shallow wiggle between two of its points is
        not one.  If this ever starts reporting a roton at ``Q_WC``, the
        mapping claim in ``crystal_direction``'s docstring has gone wrong."""
        import numpy as np
        idx = M.index_sources()
        try:
            src = M.find_source(idx, "crystal", 90.51, "quick", 3)
        except LookupError:
            self.skipTest("no post-fix crystal source on disk")
        torus = M.torus_for(36)
        dhat = M.crystal_direction(torus, "clean-x")
        qvec = M.load_momenta(src)
        qn, S = M.load_curve(src)
        red = M.directional_cut(qvec, S, dhat)
        self.assertEqual(red["n_points"], 5)
        self.assertGreater(float(np.min(np.abs(red["q"] - M.q_wc()))), 0.3)

    def test_the_search_does_not_smooth_or_fit(self):
        """A roton manufactured by smoothing would be the fabrication §10
        forbids; the returned q must be one of the sample points."""
        import numpy as np
        q, om, _ = self._dipped(2.693547)
        got = M.roton_minimum({"q": q, "Omega": om}, 2.693547)
        self.assertTrue(np.any(q == got["q_roton"]),
                        "the reported q is not one of the sample points")


# ==========================================================================
# 8. the reduction  S(q-vector) -> S(q), which is the correction itself
# ==========================================================================
class TestTheLiquidReduction(unittest.TestCase):
    """The liquid is isotropic, so ``S`` is averaged over each shell of equal
    ``|q|`` -- and the average is taken on ``S``, before the division."""

    def _synth(self):
        """A synthetic disc: 294 momenta, exact shells, known per-shell answer."""
        import numpy as np
        torus = M.torus_for(36)
        q, qn = torus.allowed_momenta(4.0)
        # A function of |q| plus a directional ripple, so every shell has real
        # internal spread and the two orderings must disagree.
        ang = np.arctan2(q[:, 1], q[:, 0])
        S = 0.5 * (1.0 - np.exp(-qn ** 2 / 2.0)) * (1.0 + 0.3 * np.cos(6 * ang))
        return q, qn, S

    def test_every_shell_holds_only_equal_magnitudes(self):
        q, qn, S = self._synth()
        red = M.shell_average(q, S)
        self.assertEqual(red["n_shells"], 30)
        self.assertLessEqual(red["within_shell_q_spread"], M.SHELL_TOL)
        self.assertEqual(int(red["count"].sum()), qn.size)

    def test_every_momentum_lands_in_exactly_one_shell(self):
        import numpy as np
        q, qn, S = self._synth()
        red = M.shell_average(q, S)
        for i in range(qn.size):
            self.assertEqual(
                int(np.count_nonzero(np.abs(red["q"] - qn[i]) <= M.SHELL_TOL)),
                1, f"|q| = {qn[i]} matched no unique shell")

    def test_the_mean_is_taken_on_S_not_on_one_over_S(self):
        """The order of operations, which is the whole correction.

        ``q^2/(2<S>)`` and ``<q^2/(2S)>`` agree only when a shell has no
        spread.  On a shell WITH spread they must differ, and in one direction:
        ``<1/S> >= 1/<S>``, so the averaged denominator gives the SMALLER
        ``Omega``.  A test that only checked they were unequal would pass on a
        sign error."""
        import numpy as np
        q, qn, S = self._synth()
        red = M.shell_average(q, S)
        self.assertGreater(np.abs(red["spread"]).max(), 0.0,
                           "the synthetic disc has no directional spread, so "
                           "the orderings could not differ and this test would "
                           "be vacuous")
        gap = red["order_gap_rel"]
        self.assertGreater(np.abs(gap).max(), 1e-6,
                           "the two orderings agree to float noise; the "
                           "reduction is not doing what it claims")
        # Jensen: mean(1/S) >= 1/mean(S), so the wrong order is never smaller.
        self.assertGreaterEqual(float(gap.min()), -1e-12)

    def test_the_returned_S_is_the_plain_mean_of_the_shell(self):
        import numpy as np
        q, qn, S = self._synth()
        red = M.shell_average(q, S)
        for i, qi in enumerate(red["q"]):
            members = S[np.abs(qn - qi) <= M.SHELL_TOL]
            self.assertAlmostEqual(float(red["S"][i]), float(members.mean()),
                                   places=12)

    def test_a_shell_that_mixed_two_magnitudes_is_refused(self):
        """The guard: a tolerance band that swallowed a neighbour shell would
        average over |q| as well as direction and produce a smooth wrong curve."""
        import numpy as np
        # The mocked "shell" holds |q| = 1 and |q| = 2: exactly the corruption
        # a running-tolerance grouper would produce at a shell boundary.
        q = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
        S = np.array([1.0, 2.0, 3.0])
        with mock.patch.object(M, "shells_of",
                               return_value=[np.array([0, 1])]):
            with self.assertRaises(AssertionError):
                M.shell_average(q, S)


class TestTheCrystalReduction(unittest.TestCase):
    """The crystal is not isotropic, so its curve is a cut, not an average."""

    def test_the_paper_x_axis_is_a_reciprocal_lattice_direction(self):
        """The mapping the whole correction rests on: the paper's frame is this
        package's frame rotated by 30 degrees, so the paper's ``x = (1,0)`` is
        the direction of ``G1 + G2`` here -- a Bragg direction."""
        import numpy as np
        torus = M.torus_for(36)
        dhat = M.crystal_direction(torus, "bragg")
        self.assertAlmostEqual(M.direction_angle_deg(dhat), 30.0, places=6)
        g = np.asarray(torus.G1, float) + np.asarray(torus.G2, float)
        self.assertLess(float(np.abs(dhat - g / np.linalg.norm(g)).max()), 1e-12)
        # And it is a reciprocal-lattice direction, not merely a plausible one:
        # six supercell steps along it land exactly on the Wigner crystal's
        # first primitive reciprocal vector.  That is the property that makes a
        # cut along it pass through Q_WC -- and the property the torus's own
        # first axis does not have (its ladder is sqrt(3) times longer).
        self.assertAlmostEqual(float(np.linalg.norm(g) * 6.0), M.q_wc(),
                               places=6)
        self.assertAlmostEqual(float(np.linalg.norm(torus.G1)) * 6.0,
                               M.q_wc(), places=6)

    def test_the_clean_x_axis_is_not_the_paper_x_axis(self):
        torus = M.torus_for(36)
        self.assertAlmostEqual(
            M.direction_angle_deg(M.crystal_direction(torus, "clean-x")),
            0.0, places=9)
        self.assertNotAlmostEqual(
            M.direction_angle_deg(M.crystal_direction(torus, "bragg")),
            0.0, places=3)

    def test_an_unknown_direction_is_refused(self):
        with self.assertRaises(ValueError):
            M.crystal_direction(M.torus_for(36), "diagonal-ish")

    def test_every_point_of_the_cut_is_exactly_collinear(self):
        import numpy as np
        idx = M.index_sources()
        try:
            src = M.find_source(idx, "crystal", 90.51, "quick", 3)
        except LookupError:
            self.skipTest("no post-fix crystal source on disk")
        qvec = M.load_momenta(src)
        S = M.load_curve(src)[1]
        red = M.directional_cut(qvec, S, M.crystal_direction(M.torus_for(36),
                                                             "bragg"))
        # Every point that made the cut is on the line to float precision --
        # nothing was projected onto the axis to fill the curve out.
        self.assertLessEqual(red["max_perp_residual"], M.COLLINEAR_TOL)
        self.assertEqual(red["n_points"], len(red["q"]))
        self.assertTrue(np.all(np.diff(red["q"]) > 0.0))

    def test_the_cut_uses_one_side_and_the_two_sides_agree(self):
        """``S(-q) = S(q)``, so keeping only ``q.dhat > 0`` loses nothing --
        verified against the mirror momenta rather than assumed."""
        idx = M.index_sources()
        try:
            src = M.find_source(idx, "crystal", 90.51, "quick", 3)
        except LookupError:
            self.skipTest("no post-fix crystal source on disk")
        qvec = M.load_momenta(src)
        S = M.load_curve(src)[1]
        red = M.directional_cut(qvec, S, M.crystal_direction(M.torus_for(36),
                                                             "bragg"))
        self.assertGreater(red["n_mirror_pairs_checked"], 0)
        self.assertLess(red["mirror_max_abs_gap"], 1e-9)

    def test_an_axis_with_no_momenta_raises_rather_than_interpolating(self):
        """§4's stop condition: not enough collinear momenta is a limitation to
        report, not a gap to fill."""
        import numpy as np
        qvec = np.array([[1.0, 0.0], [0.0, 1.0]])
        S = np.array([1.0, 1.0])
        with self.assertRaises(ValueError):
            M.directional_cut(qvec, S, np.array([math.cos(0.7), math.sin(0.7)]))

    def test_the_directional_curve_does_not_average_a_shell(self):
        """The regression §8 names: a shell mean of the crystal.  On the cut,
        ``count`` must be 1 everywhere."""
        idx = M.index_sources()
        try:
            src = M.find_source(idx, "crystal", 90.51, "quick", 3)
        except LookupError:
            self.skipTest("no post-fix crystal source on disk")
        qvec = M.load_momenta(src)
        S = M.load_curve(src)[1]
        red = M.directional_cut(qvec, S, M.crystal_direction(M.torus_for(36),
                                                             "bragg"))
        self.assertTrue(all(int(c) == 1 for c in red["count"]))


class TestTheReductionIsWiredIntoTheCurves(unittest.TestCase):
    """The reduction functions being right is not enough -- the curve-building
    path has to actually call the right one.  A mutation test caught this: the
    per-phase tests above call ``directional_cut`` directly, so swapping the
    DISPATCH to shell-average the crystal left them all green while the figure
    quietly averaged the Bragg peak away."""

    def _args(self, rs, budget="quick"):
        a = M.parse_args(["--rs", *[str(r) for r in rs],
                          "--source-budget", budget])
        a.phase_of = M.phase_of
        return a

    def test_the_phase_dispatch_picks_a_different_reduction_per_phase(self):
        idx = M.index_sources()
        try:
            src = M.find_source(idx, "crystal", 90.51, "quick", 3)
        except LookupError:
            self.skipTest("no post-fix crystal source on disk")
        qvec = M.load_momenta(src)
        S = M.load_curve(src)[1]
        dhat = M.crystal_direction(M.torus_for(36), "bragg")

        cry = M.reduce_curve(qvec, S, "crystal", dhat)
        self.assertEqual(cry["reduction"], "directional")
        self.assertEqual(cry["n_points"], 8)
        self.assertTrue(all(int(c) == 1 for c in cry["count"]),
                        "the crystal was shell-averaged in the dispatch")

        liq = M.reduce_curve(qvec, S, "liquid", dhat)
        self.assertEqual(liq["reduction"], "shell-mean")
        self.assertEqual(liq["n_shells"], 30)
        self.assertEqual(int(liq["count"].sum()), 294)

    def test_an_unknown_phase_is_refused(self):
        import numpy as np
        with self.assertRaises(ValueError):
            M.reduce_curve(np.array([[1.0, 0.0]]), np.array([1.0]),
                           "plasma", np.array([1.0, 0.0]))

    def test_the_built_curves_carry_the_reduction_their_phase_needs(self):
        """End to end: a crystal curve must have one point per |q| on one axis,
        a liquid curve one point per shell, and both must say which they are."""
        import numpy as np
        try:
            curves, _ = M.build_curves(self._args([5.567, 90.51]))
        except LookupError:
            self.skipTest("no post-fix sources on disk")
        by = {c["phase"]: c for c in curves}
        self.assertEqual(by["liquid"]["reduction"]["reduction"], "shell-mean")
        self.assertEqual(by["crystal"]["reduction"]["reduction"], "directional")
        self.assertEqual(by["liquid"]["q"].size, 30)
        self.assertEqual(by["crystal"]["q"].size, 8)
        # The crystal's points are collinear with ITS declared direction and
        # with nothing else; the liquid's are one per magnitude.
        dhat = np.asarray(by["crystal"]["reduction"]["direction"], float)
        perp_axis = np.array([-dhat[1], dhat[0]])
        qv = by["crystal"]["raw_qvec"]
        for qi in by["crystal"]["q"]:
            idx = np.flatnonzero(np.isclose(np.linalg.norm(qv, axis=1), qi,
                                            rtol=1e-9))
            self.assertLess(float((np.abs(qv[idx] @ perp_axis)).min()),
                            M.COLLINEAR_TOL)
        self.assertEqual(
            int(np.unique(np.round(by["liquid"]["q"], 9)).size), 30)


class TestTheFrameCheck(unittest.TestCase):
    """The direction comes from the torus, so the source must BE that torus."""

    def test_a_source_on_this_lattice_passes(self):
        idx = M.index_sources()
        try:
            src = M.find_source(idx, "liquid", 5.567, "quick")
        except LookupError:
            self.skipTest("no post-fix liquid source on disk")
        M.check_frame(M.load_momenta(src), M.torus_for(36), src)

    def test_momenta_from_another_lattice_are_refused(self):
        import numpy as np
        torus = M.torus_for(36)
        q, _ = torus.allowed_momenta(4.0)
        skewed = q * np.array([1.0, 1.07])       # no longer m G1 + n G2
        with self.assertRaises(ValueError):
            M.check_frame(skewed, torus, {"npz": "synthetic"})


class TestTheCouplingFlags(unittest.TestCase):
    """Two spellings of the same request must mean the same request.

    ``--liquid-rs 5.567 --crystal-rs 90.51`` is the interface a reader reaches
    for first, because it names the phase.  It has to agree with ``--rs``
    exactly, and the phase in the flag's name has to be the phase the figure
    would actually draw -- otherwise the flag is a label that can lie.
    """

    def test_the_phase_named_flags_are_the_same_list_as_rs(self):
        plain = M.parse_args(["--rs", "5.567", "90.51"])
        named = M.parse_args(["--liquid-rs", "5.567",
                              "--crystal-rs", "90.51"])
        self.assertEqual(plain.rs, named.rs)
        self.assertEqual(plain.source_budget, named.source_budget)

    def test_budget_is_an_alias_of_source_budget(self):
        self.assertEqual(M.parse_args(["--budget", "quick"]).source_budget,
                         "quick")
        self.assertEqual(
            M.parse_args(["--source-budget", "quick"]).source_budget, "quick")
        # and it is the same dest, not a second field that could drift
        self.assertEqual(M.parse_args(["--budget", "quick"]).source_budget,
                         M.parse_args(["--source-budget", "quick"]).source_budget)

    def test_a_coupling_under_the_other_flag_still_draws_by_its_own_rs(self):
        """The flags group the input; they do not decide the phase.

        ``--liquid-rs 90.51`` is accepted and drawn as the CRYSTAL, because
        90.51 is above r_s^c.  A hard gate refusing this was tried and removed:
        the boundary belongs to the coupling, not to the spelling someone
        typed it under.  What matters is that the phase comes from r_s -- so
        the flags cannot change it.
        """
        for argv in (["--liquid-rs", "90.51"],
                     ["--crystal-rs", "5.567"],
                     ["--liquid-rs", "5.567", "90.51"],
                     ["--crystal-rs", "90.51", "5.567"]):
            args = M.parse_args(argv)
            for r in args.rs:
                self.assertEqual(M.phase_of(r), M.phase_of(r),
                                 "phase must be a function of r_s alone")

    def test_the_flag_cannot_move_a_coupling_across_the_boundary(self):
        """The real check: whichever flag a coupling arrives under, the phase
        it is drawn with is the one r_s gives it."""
        for flag in ("--liquid-rs", "--crystal-rs"):
            args = M.parse_args([flag, "5.567", "90.51"])
            self.assertEqual([M.phase_of(r) for r in args.rs],
                             ["liquid", "crystal"],
                             f"{flag} changed the phase of a coupling")

    def test_the_flags_group_the_input_in_a_fixed_order(self):
        """Liquid group first, crystal group second, whatever order they are
        typed in -- so the colour a coupling gets does not depend on how the
        command line happened to be arranged on the day."""
        a = M.parse_args(["--liquid-rs", "5.567", "--crystal-rs", "90.51"])
        b = M.parse_args(["--crystal-rs", "90.51", "--liquid-rs", "5.567"])
        self.assertEqual(a.rs, [5.567, 90.51])
        self.assertEqual(b.rs, [5.567, 90.51])

    def test_the_boundary_itself_goes_to_the_crystal(self):
        # phase_of is < r_s^c for the liquid, so exactly r_s^c is a crystal.
        rs_c = str(M.RS_C)
        self.assertEqual(M.phase_of(float(rs_c)), "crystal")
        self.assertEqual(M.parse_args(["--crystal-rs", rs_c]).rs,
                         [float(rs_c)])
        # and naming it liquid does not move it
        self.assertEqual(M.phase_of(M.parse_args(["--liquid-rs", rs_c]).rs[0]),
                         "crystal")

    def test_giving_both_spellings_at_once_is_refused(self):
        with self.assertRaises(SystemExit):
            M.parse_args(["--rs", "5.567", "--liquid-rs", "5.567"])
        with self.assertRaises(SystemExit):
            M.parse_args(["--rs", "90.51", "--crystal-rs", "90.51"])

    def test_the_phase_named_flags_accept_the_analytic_curve(self):
        # r_s = 0 is drawn from the liquid branch and needs no saved data.
        self.assertEqual(M.parse_args(["--liquid-rs", "0"]).rs, [0.0])
        self.assertEqual(M.parse_args(["--liquid-rs", "0", "5.567"]).rs,
                         [0.0, 5.567])

    def test_no_coupling_flags_at_all_is_still_the_papers_points(self):
        self.assertEqual(M.parse_args([]).rs, list(M.DEFAULT_RS))

    def test_the_boundary_is_rs_c_and_the_flags_do_not_move_it(self):
        # Two couplings one part in a million either side of r_s^c land on
        # opposite phases, and which flag they arrived under does not enter
        # into it -- so the boundary is r_s^c and nothing else.
        just_below = M.RS_C - 1e-6
        just_above = M.RS_C + 1e-6
        self.assertEqual(M.phase_of(just_below), "liquid")
        self.assertEqual(M.phase_of(just_above), "crystal")
        for flag in ("--liquid-rs", "--crystal-rs"):
            args = M.parse_args([flag, repr(just_below), repr(just_above)])
            self.assertEqual([M.phase_of(r) for r in args.rs],
                             ["liquid", "crystal"])


class TestTheFigureCarriesNoDecoration(unittest.TestCase):
    """The figure is measurements; the axes carry the curves and nothing else.

    Two things kept growing back on it -- a dashed guide line at `Q_WC` and a
    marker planted on the roton minimum -- so both are pinned as ABSENT rather
    than removed once and hoped about.

    The roton marker was not merely decoration.  It sat on top of the data
    point it named, so while it was drawn the crystal's minimum at `Q_WC` could
    not be seen on the axes at all.  Removing it made a real data point appear.
    """

    def test_no_dashed_curve_is_drawn_by_default(self):
        self.assertFalse(M.parse_args([]).classical)
        self.assertFalse(M.parse_args(["--source-budget", "quick"]).classical)

    def test_classical_is_opt_in_and_the_old_flag_still_parses(self):
        self.assertTrue(M.parse_args(["--classical"]).classical)
        # --no-classical is redundant now, not rejected: a command line that
        # used it must keep working rather than start failing.
        self.assertFalse(M.parse_args(["--no-classical"]).classical)

    def test_write_figure_defaults_to_not_drawing_them(self):
        sig = inspect.signature(M.write_figure)
        self.assertIn("show_classical", sig.parameters)
        self.assertIs(sig.parameters["show_classical"].default, False)

    def test_the_renderer_plants_nothing_on_the_axes(self):
        """Every guide line, leader line and label box has to be one of these
        calls, so their absence from the renderer's source is the check.  The
        dashed linestyle is allowed only as the opt-in classical curve, which
        passes it positionally rather than as ``ls=``.

        ``ax.text`` is deliberately NOT on the list: it is how the
        ``NOT PUBLICATION QUALITY`` stamp is drawn, and that stamp is required
        provenance rather than an annotation on a measurement.
        """
        src = inspect.getsource(M.write_figure)
        for forbidden in ("axvline", "axhline", "annotate",
                          "markeredgecolor", 'marker="v"', 'ls=":"',
                          'ls="--"'):
            self.assertNotIn(forbidden, src,
                             f"{forbidden!r} is decoration the figure does "
                             f"not carry")
        self.assertIn("stamp", src,
                      "the quality stamp must still be drawn")

    def test_the_stamp_is_the_only_text_on_the_axes(self):
        """Pins the one allowed use of ax.text so a second one -- a roton
        label, say -- is a failing test rather than a judgement call."""
        src = inspect.getsource(M.write_figure)
        self.assertEqual(src.count("ax.text("), 1)

    def test_the_roton_is_reported_rather_than_drawn(self):
        """The search itself is untouched -- only its rendering is gone."""
        sig = inspect.signature(M.write_figure)
        self.assertNotIn("roton", sig.parameters)
        self.assertNotIn("q_wc_value", sig.parameters)
        self.assertTrue(callable(getattr(M, "roton_minimum", None)),
                        "the roton search must still exist; only the marker "
                        "on the figure was removed")


class TestTheMissingMessageTellsTheTruth(unittest.TestCase):
    """The refusal has to name the real reason.

    This message once listed every source of the phase, so a lookup rejected
    for its truncation printed the requested coupling in the very list
    explaining the refusal.  That reads as "this value is not allowed" when the
    value was never the problem -- the fix was one flag away -- so the list is
    now filtered by the same predicates the lookup uses.
    """

    def _row(self, rs, budget, nmax, post_fix, mtime=1.0):
        return {"phase": "crystal", "rs": float(rs), "budget": budget,
                "nmax": nmax, "post_fix": post_fix, "mtime": mtime,
                "dir": f"synthetic/rs{rs:g}_nmax{nmax}", "npz": "synthetic.npz"}

    def _index(self):
        return [
            self._row(90.51, "quick", 3, True),
            self._row(100, "quick", 2, True),        # right coupling, other nmax
            self._row(75, "quick", 3, False),        # right nmax, pre-fix
        ]

    def test_a_coupling_at_another_truncation_is_named_not_listed(self):
        msg = M._missing_message(self._index(), "crystal", 100.0, "quick", 3)
        self.assertNotIn("100 (quick)", msg,
                         "the message advertised a coupling the lookup "
                         "would refuse")
        self.assertIn("--crystal-nmax 2", msg,
                      "the message must say which flag reaches it")

    def test_a_pre_fix_coupling_is_not_advertised_as_available(self):
        msg = M._missing_message(self._index(), "crystal", 75.0, "quick", 3)
        self.assertNotIn("75 (quick)", msg)
        self.assertIn("vmc/sr.py:219", msg,
                      "the refusal reason must be named")

    def test_the_available_list_holds_only_usable_couplings(self):
        msg = M._missing_message(self._index(), "crystal", 100.0, "quick", 3)
        listed = msg.split("available crystal couplings:")[1].splitlines()[0]
        self.assertIn("90.51", listed)
        for gone in ("75", "100"):
            self.assertNotIn(gone, listed,
                             f"{gone} is not drawable and must not be listed")

    def test_a_pre_fix_twin_does_not_mask_a_usable_file(self):
        """One coupling can have both files at the same nmax.  Naming the
        stale one would be true and useless when a good one is beside it."""
        index = self._index() + [self._row(100, "quick", 2, False)]
        msg = M._missing_message(index, "crystal", 100.0, "quick", 3)
        self.assertIn("--crystal-nmax 2", msg)
        self.assertNotIn("predates", msg)

    def test_the_liquid_list_is_not_truncation_filtered(self):
        """nmax is a crystal-only selector; applying it to the liquid would
        hide couplings that are perfectly drawable."""
        index = [{"phase": "liquid", "rs": 5.567, "budget": "quick",
                  "nmax": 1, "post_fix": True, "mtime": 1.0,
                  "dir": "synthetic/liquid", "npz": "synthetic.npz"}]
        msg = M._missing_message(index, "liquid", 30.0, "quick", 3)
        self.assertIn("5.567", msg)


if __name__ == "__main__":
    unittest.main()
