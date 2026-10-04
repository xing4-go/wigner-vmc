"""`physics/landau_levels.py` against the frozen engine, and against the mathematics.

This file carries the regression that the whole `nmax` rule rests on.  The frozen
engine's `LandauLevelBasis` takes a parameter it *calls* ``n_max`` which is in fact the
BAND COUNT: ``(mesh, n_max=k)`` returns ``k+1`` columns -- bands ``0..k-1`` plus one
padding band at index ``k``.  The clean class takes ``nmax``, the highest Landau index,
and derives ``n_bands = nmax + 1``.  `test_nmax_is_not_the_legacy_band_count` measures
the legacy behaviour rather than trusting the docstring, so that the mapping these
tests use cannot drift from the object it maps.

No Monte Carlo, no optimisation: every input is a fixed array and every position is a
fixed 2-vector.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import qhvmc_engine as legacy                                    # noqa: E402
from wigner_vmc.physics import geometry as g                     # noqa: E402
from wigner_vmc.physics import landau_levels as ll               # noqa: E402

PRIM_AREA = 2.0 * np.pi
A_WC = np.sqrt(4 * np.pi / np.sqrt(3))
RTOL = 1e-13
ATOL = 1e-14

# fixed evaluation points; deliberately not symmetric, and not near a cell centre
POINTS = (np.array([0.31, -0.77]), np.array([1.10, 0.40]), np.array([-2.00, 1.30]),
          np.array([0.0, 0.0]), np.array([0.5 * A_WC, 0.25 * A_WC]),
          np.array([-1.7, -2.9]))


def setup(nmax_clean):
    """(clean basis, legacy basis, geometry) for one truncation."""
    A1, A2 = g.triangular_cell(PRIM_AREA)
    Ge = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
    ai, ac = g.circular_lattice(8.5 * np.linalg.norm(A1), A1, A2)
    C = np.column_stack([A1, A2])
    Ci = np.linalg.inv(C)
    new = ll.LandauLevelBasis(Ge.mesh, nmax_clean, ai, ac, C, Ci)
    old = legacy.LandauLevelBasis(Ge.mesh, nmax_clean + 1, ai, ac, C, Ci)
    return new, old, Ge


class TestTheLegacyConventionItself(unittest.TestCase):
    """Measure the legacy object, so the mapping in the module docstring is a fact."""

    def test_nmax_is_not_the_legacy_band_count(self):
        A1, A2 = g.triangular_cell(PRIM_AREA)
        Ge = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
        ai, ac = g.circular_lattice(8.5 * np.linalg.norm(A1), A1, A2)
        C = np.column_stack([A1, A2])
        Ci = np.linalg.inv(C)
        for k in (1, 2, 3, 4):
            old = legacy.LandauLevelBasis(Ge.mesh, k, ai, ac, C, Ci)
            self.assertEqual(old.orbitals(POINTS[0]).shape, (36, k + 1))
            self.assertEqual(old.pi_orbitals_ladder(POINTS[0])[0].shape, (36, k))

    def test_nmax_from_n_bands_is_the_only_adapter(self):
        self.assertEqual(ll.nmax_from_n_bands(1), 0)     # one band  -> only n = 0
        self.assertEqual(ll.nmax_from_n_bands(2), 1)     # conv      -> n = 0, 1
        self.assertEqual(ll.nmax_from_n_bands(3), 2)     # nest      -> n = 0, 1, 2
        with self.assertRaises(ValueError):
            ll.nmax_from_n_bands(0)
        with self.assertRaises(ValueError):
            ll.nmax_from_n_bands(-1)


class TestOrbitalsAgainstLegacy(unittest.TestCase):
    def test_physical_levels_match(self):
        for nmax in (0, 1, 2, 3):
            new, old, _ = setup(nmax)
            self.assertEqual(new.orbitals(POINTS[0]).shape, (36, nmax + 1))
            for r in POINTS:
                np.testing.assert_allclose(
                    new.orbitals(r), old.orbitals(r)[:, :nmax + 1],
                    rtol=RTOL, atol=ATOL,
                    err_msg=f"nmax={nmax} r={r}")

    def test_padded_orbitals_match_legacy_exactly(self):
        # clean nmax + 1 == legacy n_max, so the padded array IS the legacy array
        for nmax in (0, 1, 2, 3):
            new, old, _ = setup(nmax)
            self.assertEqual(new.padded_orbitals(POINTS[0]).shape, (36, nmax + 2))
            for r in POINTS:
                np.testing.assert_allclose(new.padded_orbitals(r), old.orbitals(r),
                                           rtol=RTOL, atol=ATOL)

    def test_the_padding_band_is_not_reachable_from_orbitals(self):
        new, old, _ = setup(2)
        r = POINTS[0]
        self.assertEqual(new.orbitals(r).shape[1], new.n_bands)
        # a^dag|nmax> is exactly the column `orbitals` refuses to hand out
        np.testing.assert_allclose(new.padded_orbitals(r)[:, -1],
                                   old.orbitals(r)[:, -1], rtol=RTOL, atol=ATOL)
        self.assertFalse(np.allclose(new.orbitals(r)[:, -1],
                                     new.padded_orbitals(r)[:, -1]))


class TestLadderAgainstLegacy(unittest.TestCase):
    def test_matches_legacy(self):
        for nmax in (0, 1, 2, 3):
            new, old, _ = setup(nmax)
            for r in POINTS:
                mine = new.ladder(r)
                theirs = old.pi_orbitals_ladder(r)
                self.assertEqual(mine[0].shape, (36, nmax + 1))
                for a, b in zip(mine, theirs):
                    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_pi_squared_is_exactly_diagonal(self):
        # a property of THIS basis and of no other in the project
        for nmax in (0, 1, 2, 3):
            new, _, _ = setup(nmax)
            for r in POINTS:
                _, _, pi2 = new.ladder(r)
                expected = new.orbitals(r) * (2 * np.arange(nmax + 1) + 1)[None, :]
                np.testing.assert_allclose(pi2, expected, rtol=0, atol=1e-15)


class TestThePhysicsTheBasisMustSatisfy(unittest.TestCase):
    """Checks that do not mention legacy at all.  A port reproduces a bug in both
    copies; only these notice."""

    def test_pi_psi_is_the_covariant_derivative_in_the_stated_gauge(self):
        # pi_x = -i d_x - A_x and pi_y = -i d_y - A_y with A = (1/2)(-y, x).
        # This is the test that pins the SIGN OF B: the opposite gauge gives the same
        # pi^2 (hence the same kinetic energy) and the opposite pi psi.
        new, _, _ = setup(2)
        h = 1e-5
        for r in POINTS:
            px, py, _ = new.ladder(r)
            dx = (new.orbitals(r + [h, 0]) - new.orbitals(r - [h, 0])) / (2 * h)
            dy = (new.orbitals(r + [0, h]) - new.orbitals(r - [0, h])) / (2 * h)
            PX = -1j * dx + (r[1] / 2.0) * new.orbitals(r)
            PY = -1j * dy - (r[0] / 2.0) * new.orbitals(r)
            self.assertLess(float(np.abs(PX - px).max()), 5e-9, f"pi_x at r={r}")
            self.assertLess(float(np.abs(PY - py).max()), 5e-9, f"pi_y at r={r}")

    def test_the_wrong_gauge_sign_is_detectable(self):
        # Guard against the finite-difference test above going vacuous: with the
        # opposite sign of A the residual is orders of magnitude larger.
        new, _, _ = setup(2)
        h = 1e-5
        r = POINTS[0]
        px, py, _ = new.ladder(r)
        dx = (new.orbitals(r + [h, 0]) - new.orbitals(r - [h, 0])) / (2 * h)
        dy = (new.orbitals(r + [0, h]) - new.orbitals(r - [0, h])) / (2 * h)
        wrong = -1j * dx - (r[1] / 2.0) * new.orbitals(r)
        self.assertGreater(float(np.abs(wrong - px).max()), 1e-3)

    def test_a_lowering_then_raising_round_trip(self):
        # a|n> = sqrt(n)|n-1> and a^dag|n> = sqrt(n+1)|n+1>: reconstruct the ladder from
        # pi_x and pi_y and check the action on the band index.
        new, _, _ = setup(3)
        r = POINTS[0]
        px, py, _ = new.ladder(r)
        a = (px + 1j * py) / np.sqrt(2.0)
        O = new.orbitals(r)
        # a|0> must vanish
        self.assertLess(float(np.abs(a[:, 0]).max()), 1e-12)
        # a|n> = sqrt(n) |n-1> for n >= 1
        for n in range(1, new.n_bands):
            np.testing.assert_allclose(a[:, n], np.sqrt(n) * O[:, n - 1],
                                       rtol=1e-12, atol=1e-12)

    def test_basis_is_a_bloch_sum_that_is_actually_periodic_in_the_gauge(self):
        # |psi_k(r + a)| must equal |psi_k(r)| for every primitive lattice vector a:
        # the orbital is a probability amplitude on the torus.  This holds only if the
        # magnetic translation phase and the fold agree.
        new, _, Ge = setup(2)
        r = POINTS[1]
        for a in (Ge.A1, Ge.A2, -2 * Ge.A1 + 3 * Ge.A2, 6 * Ge.A1):
            np.testing.assert_allclose(np.abs(new.orbitals(r + a)),
                                       np.abs(new.orbitals(r)),
                                       rtol=1e-12, atol=1e-12,
                                       err_msg=f"shift {a}")

    def test_a_magnitude_that_shifts_without_changing_norm_is_not_a_symmetry(self):
        # Guard against a tempting-but-wrong generalisation of the test above: r -> r +
        # t*L1 for fractional t is NOT a lattice translation, and |psi| is not invariant
        # under it.  Recorded so that the passing translation test is not read as
        # "shifting r does not matter".
        new, _, Ge = setup(0)
        r = POINTS[1]
        self.assertGreater(float(np.abs(np.abs(new.orbitals(r + 0.25 * Ge.L1))
                                        - np.abs(new.orbitals(r))).max()), 1e-3)

    def test_flux_quanta_per_cell_is_one(self):
        new, _, _ = setup(2)
        self.assertAlmostEqual(new.flux_quanta_per_cell, 1.0, places=12)


class TestValidation(unittest.TestCase):
    def test_rejects_a_negative_nmax(self):
        A1, A2 = g.triangular_cell(PRIM_AREA)
        Ge = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
        ai, ac = g.circular_lattice(8.5 * np.linalg.norm(A1), A1, A2)
        C = np.column_stack([A1, A2])
        with self.assertRaises(ValueError):
            ll.LandauLevelBasis(Ge.mesh, -1, ai, ac, C, np.linalg.inv(C))

    def test_rejects_a_bad_mesh_or_point(self):
        A1, A2 = g.triangular_cell(PRIM_AREA)
        Ge = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
        ai, ac = g.circular_lattice(8.5 * np.linalg.norm(A1), A1, A2)
        C = np.column_stack([A1, A2])
        Ci = np.linalg.inv(C)
        with self.assertRaises(ValueError):
            ll.LandauLevelBasis(np.zeros(36), 1, ai, ac, C, Ci)
        b = ll.LandauLevelBasis(Ge.mesh, 1, ai, ac, C, Ci)
        with self.assertRaises(ValueError):
            b.orbitals(np.zeros(3))

    def test_n_bands_is_derived_not_settable(self):
        new, _, _ = setup(3)
        self.assertEqual(new.n_bands, 4)
        with self.assertRaises(AttributeError):
            new.n_bands = 5


if __name__ == "__main__":
    unittest.main(verbosity=2)
