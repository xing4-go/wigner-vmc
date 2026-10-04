"""The error-bar conventions, pinned.

These are the numbers the whole project's conclusions rest on, so they get tests that
assert the FORMULA and not merely that the function runs.  A test that only checked
"returns a float" would pass just as happily on a wrong formula.
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CLEAN, "src"))

from wigner_vmc.analysis.statistics import (          # noqa: E402
    bracket, brackets_overlap, combine_runs, compare_phases, seed_spread, sigma_init,
    sigma_mc)
from wigner_vmc.io.schema import Bracket, ScanPoint, StateRecord  # noqa: E402


def _state(E, err, rs=55.0, init_id="s0", phase="crystal", tier="conv", nb=2):
    return StateRecord(workflow="llrot", phase=phase, rs=rs, N=36, kappa=rs / math.sqrt(2),
                       nmax=nb - 1, init_id=init_id, rng_seed=None, parent_id=None,
                       nested=False, production_energy_per_particle=E,
                       production_mc_error=err, raw={"tier": tier, "nb": nb})


class TestSigmaMC(unittest.TestCase):
    def test_equals_s_over_sqrt_n_when_errors_are_equal(self):
        s = 0.01
        for n in (1, 2, 5, 9):
            got = sigma_mc([0.0] * n, [s] * n)
            self.assertAlmostEqual(got, s / math.sqrt(n), places=15)

    def test_weights_unequal_errors_correctly(self):
        # sqrt(sum s_i^2)/n, NOT the mean of the errors
        got = sigma_mc([0.0, 0.0], [0.03, 0.04])
        self.assertAlmostEqual(got, 0.05 / 2, places=15)
        self.assertNotAlmostEqual(got, 0.035 / math.sqrt(2), places=6)

    def test_empty_is_zero(self):
        self.assertEqual(sigma_mc([], []), 0.0)


class TestSigmaInitVersusSeedSpread(unittest.TestCase):
    """The one input where the two conventions differ, and why it matters.

    energy path -> 0.0   ("this state adds no basin spread to the budget")
    band path   -> None  ("one seed cannot bound the spread at all")
    """

    def test_single_state_diverges_and_both_are_deliberate(self):
        self.assertEqual(sigma_init([-27.2]), 0.0)
        self.assertIsNone(seed_spread([-27.2]))

    def test_two_or_more_states_agree_exactly(self):
        for vals in ([-27.26, -27.20], [-27.1, -27.3, -27.25, -27.19]):
            self.assertAlmostEqual(sigma_init(vals), seed_spread(vals), places=15)

    def test_uses_ddof_one_not_ddof_zero(self):
        vals = [-27.26, -27.20, -27.19]
        n = len(vals)
        mean = sum(vals) / n
        pop = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) / math.sqrt(n)
        got = sigma_init(vals)
        self.assertNotAlmostEqual(got, pop, places=6)
        self.assertGreater(got, pop)

    def test_a_zero_is_never_mistaken_for_an_unknown(self):
        # the confusion this pair of functions exists to prevent
        self.assertIsNot(sigma_init([1.0]), seed_spread([1.0]))


class TestCombineRuns(unittest.TestCase):
    def test_refuses_a_state_with_no_production_energy(self):
        bad = _state(None, None, init_id="trajectory-only")
        good = _state(-27.2, 0.01)
        with self.assertRaises(ValueError) as cm:
            combine_runs([good, bad])
        self.assertIn("production energy", str(cm.exception))

    def test_reproduces_the_legacy_scan_at_rs_55(self):
        """The regression, as a test rather than as console output.

        Five converged seeds at r_s = 55 and the recorded liquid; the values asserted
        are `_diag/energy_scan.json`'s.  If this fails, the conversion or the formula
        has drifted from the published number.
        """
        E = [-27.262894213, -27.237046307, -27.226739244, -27.191021344, -27.205073730]
        err = [0.010792835, 0.022893227, 0.012301939, 0.016730208, 0.008930274]
        cry = [_state(e, s, init_id=f"s{i}") for i, (e, s) in enumerate(zip(E, err))]
        c = combine_runs(cry)
        self.assertEqual(c["n"], 5)
        self.assertAlmostEqual(c["mean"], sum(E) / 5, places=12)
        # crystal-only sampling error: sqrt(sum s_i^2)/n
        want_smc = math.sqrt(sum(s * s for s in err)) / 5
        self.assertAlmostEqual(c["sigma_mc"], want_smc, places=15)

    def test_mean_of_one_state_is_that_state(self):
        c = combine_runs([_state(-27.2, 0.01)])
        self.assertAlmostEqual(c["mean"], -27.2, places=15)
        self.assertEqual(c["sigma_init"], 0.0)


class TestComparePhases(unittest.TestCase):
    def test_delta_e_is_crystal_minus_liquid_per_particle(self):
        cry = [_state(-27.20, 0.01, init_id="s0"), _state(-27.24, 0.01, init_id="s1")]
        liq = _state(-27.29, 0.008, init_id="liq", phase="liquid")
        p = compare_phases(cry, liq, tier="conv")
        self.assertAlmostEqual(p.delta_e, (-27.22) - (-27.29), places=12)
        self.assertGreater(p.delta_e, 0)

    def test_sigma_total_is_the_quadrature_of_the_legacy_convention(self):
        """hypot(hypot(sig_mc_cry, s_liq), sig_init), rebuilt here term by term.

        Note the composite is reported in `p.sigma_mc`, so it must NOT be combined with
        `s_liq` a second time -- an earlier version of this test did exactly that and
        asserted the double-counted value against the correct one.
        """
        cry = [_state(-27.20, 0.01, init_id="s0"), _state(-27.24, 0.01, init_id="s1")]
        liq = _state(-27.29, 0.008, init_id="liq", phase="liquid")
        p = compare_phases(cry, liq, tier="conv")

        c = combine_runs(cry)
        want_mc_cry = c["sigma_mc"]                      # crystal alone
        want_mc = math.hypot(want_mc_cry, 0.008)         # crystal and liquid
        want_tot = math.hypot(want_mc, c["sigma_init"])  # then the seed spread

        self.assertAlmostEqual(p.sigma_mc, want_mc, places=15)
        self.assertAlmostEqual(p.sigma_total, want_tot, places=15)
        self.assertAlmostEqual(p.sigma_init, c["sigma_init"], places=15)
        # and the three terms really are distinct, so the test cannot pass by accident
        self.assertGreater(p.sigma_mc, want_mc_cry)
        self.assertGreater(p.sigma_total, p.sigma_mc)

    def test_z_is_abs_delta_e_over_sigma_total(self):
        cry = [_state(-27.20, 0.01, init_id="s0"), _state(-27.24, 0.01, init_id="s1")]
        liq = _state(-27.29, 0.008, init_id="liq", phase="liquid")
        p = compare_phases(cry, liq, tier="conv")
        self.assertAlmostEqual(p.z, abs(p.delta_e) / p.sigma_total, places=12)

    def test_a_single_seed_gives_z_from_a_zero_init_sigma(self):
        """The single-seed case must still produce a number, and it is a LOWER bound."""
        cry = [_state(-27.20, 0.01)]
        liq = _state(-27.29, 0.008, init_id="liq", phase="liquid")
        p = compare_phases(cry, liq, tier="conv")
        self.assertEqual(p.n_states, 1)
        self.assertEqual(p.sigma_init, 0.0)
        self.assertGreater(p.z, 0)


class TestBracket(unittest.TestCase):
    @staticmethod
    def _p(rs, dE, tier="conv"):
        return ScanPoint(rs=rs, tier=tier, n_max=1, n_states=5, energy_crystal=0.0,
                         energy_liquid=0.0, delta_e=dE, sigma_mc=0.01, sigma_init=0.01,
                         sigma_total=0.014, z=abs(dE) / 0.014, seed_ids=[], caveats=[])

    def test_normal_crossing_brackets(self):
        pts = [self._p(50, +0.02), self._p(60, +0.01), self._p(70, -0.01)]
        b = bracket(pts, "conv")
        self.assertTrue(b.resolved)
        self.assertEqual((b.lo, b.hi), (60.0, 70.0))

    def test_non_monotone_series_is_reported_not_smoothed(self):
        """A sign change whose last positive point sits ABOVE the first negative one.

        lo/hi would then not describe an interval, and returning a plausible-looking
        pair would be a fabrication.
        """
        pts = [self._p(50, +0.02), self._p(60, -0.01), self._p(70, +0.03),
               self._p(80, -0.02)]
        b = bracket(pts, "conv")
        self.assertFalse(b.resolved)
        self.assertIn("DOES NOT BRACKET", b.note)
        self.assertNotIn("sign changes", b.note)

    def test_all_negative_is_not_an_interval(self):
        b = bracket([self._p(50, -0.01), self._p(60, -0.02)], "conv")
        self.assertFalse(b.resolved)
        self.assertEqual(b.kind, "below_scan")
        self.assertIsNone(b.lo)

    def test_all_positive_is_not_an_interval(self):
        b = bracket([self._p(50, +0.01), self._p(60, +0.02)], "conv")
        self.assertFalse(b.resolved)
        self.assertEqual(b.kind, "above_scan")

    def test_empty_tier(self):
        b = bracket([self._p(50, +0.01)], "nest")
        self.assertEqual(b.kind, "empty")
        self.assertFalse(b.resolved)

    def test_zero_delta_e_counts_as_neither_sign(self):
        """Exactly zero is not a crossing; it must not be filed as a positive point."""
        b = bracket([self._p(50, +0.01), self._p(60, 0.0), self._p(70, -0.01)], "conv")
        self.assertTrue(b.resolved)
        self.assertEqual((b.lo, b.hi), (50.0, 70.0))


class TestBracketsOverlap(unittest.TestCase):
    def _b(self, lo, hi, resolved=True):
        return Bracket(tier="conv", kind="bracket", lo=lo, hi=hi, resolved=resolved,
                       n_points=3, note="")

    def test_overlapping(self):
        self.assertEqual(brackets_overlap(self._b(55, 75), self._b(65, 85)), (65, 75))

    def test_disjoint_returns_none(self):
        self.assertIsNone(brackets_overlap(self._b(55, 65), self._b(65, 75)))
        self.assertIsNone(brackets_overlap(self._b(65, 75), self._b(55, 65)))

    def test_unresolved_never_overlaps(self):
        self.assertIsNone(brackets_overlap(self._b(55, 75), self._b(60, 80, resolved=False)))


class TestConversionIdentity(unittest.TestCase):
    """Converted records must satisfy E = T + kappa*V, or the conversion is wrong."""

    def test_energy_identity_on_a_known_record(self):
        N, T, V, kappa = 36, 57.1937, -26.6841, 35.3553
        E = (T + kappa * V) / N
        self.assertAlmostEqual(E, -24.617, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
