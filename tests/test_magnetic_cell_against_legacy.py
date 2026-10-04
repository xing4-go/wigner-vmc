"""`physics/magnetic_cell.py` against the frozen engine.

The three magnetic-cell conventions (gauge, flux quanta, the fold) are checked against
legacy where legacy implements them, and against their own defining property where it
does not.

The flux-quanta factor is the interesting one.  Legacy writes it twice, in two
different shapes, and the shapes differ because the *cell* differs -- it is `1` for the
Landau-level Bloch sum (primitive cell, one flux quantum) and `36` for the Gaussian
Bloch sum (supercell, 36 flux quanta).  Both are reproduced here from the single rule
`magnetic_translation_phase`, which is the point of having the rule in one place.
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
from wigner_vmc.physics import magnetic_cell as mc               # noqa: E402

PRIM_AREA = 2.0 * np.pi
AREA_PROD = 36.0 * PRIM_AREA
A_WC = np.sqrt(4 * np.pi / np.sqrt(3))


def geom(magnetic=True):
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, 4.0, magnetic=magnetic)


class TestFluxQuanta(unittest.TestCase):
    def test_production_values(self):
        self.assertAlmostEqual(mc.flux_quanta(AREA_PROD), 36.0, places=12)
        self.assertAlmostEqual(mc.flux_quanta(PRIM_AREA), 1.0, places=12)

    def test_matches_the_engine_n_phi(self):
        # the Gaussian basis computes n_phi = area/(2 pi l^2) on the supercell
        Ge = geom()
        sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
        lints, lcart = legacy.circular_lattice(30.0, Ge.L1, Ge.L2)
        gb = legacy.GaussianBasis(sites, lints, lcart, Ge.L1, Ge.L2, l=1.0)
        self.assertAlmostEqual(mc.flux_quanta(Ge.area, 1.0), float(gb.n_phi), places=12)

    def test_rejects_nothing_but_is_a_pure_function_of_area(self):
        # scaling the area scales the flux: this is what makes it a *density* statement
        self.assertAlmostEqual(mc.flux_quanta(2 * AREA_PROD), 72.0, places=12)


class TestMagneticTranslationPhase(unittest.TestCase):
    def test_primitive_cell_matches_the_landau_level_basis(self):
        # legacy LandauLevelBasis.phase_a = exp(i pi m n), no explicit flux factor,
        # because the primitive cell holds exactly one flux quantum
        A1, A2 = g.triangular_cell(PRIM_AREA)
        ai, _ac = legacy.circular_lattice(8.5 * np.linalg.norm(A1), A1, A2)
        C = np.column_stack([A1, A2])
        from wigner_vmc.physics import landau_levels as ll
        lb = legacy.LandauLevelBasis(geom().mesh, 1, ai, _ac, C, np.linalg.inv(C))
        # the EXACT 1.0 the basis uses, so the comparison is bit-for-bit.  (The
        # determinant of the computed cell is 1 + 2e-16 flux quanta, not exactly 1 --
        # which is why the constant is a named integer rather than a computed ratio.)
        self.assertAlmostEqual(mc.flux_quanta(abs(np.linalg.det(C))), 1.0, places=15)
        self.assertEqual(ll.FLUX_QUANTA_PER_PRIMITIVE_CELL, 1.0)
        mine = mc.magnetic_translation_phase(ai, ll.FLUX_QUANTA_PER_PRIMITIVE_CELL)
        np.testing.assert_allclose(mine, lb.phase_a, rtol=0, atol=0)

    def test_supercell_matches_the_gaussian_phase_term(self):
        # the SAME rule with n_phi = 36 reproduces the Gaussian basis' phase term
        #     l_ints[:,0] * l_ints[:,1] * n_phi * pi
        Ge = geom()
        lints, _ = legacy.circular_lattice(30.0, Ge.L1, Ge.L2)
        n_phi = mc.flux_quanta(Ge.area)
        legacy_phase = np.exp(1j * lints[:, 0] * lints[:, 1] * n_phi * np.pi)
        np.testing.assert_allclose(mc.magnetic_translation_phase(lints, n_phi),
                                   legacy_phase, rtol=0, atol=1e-15)

    def test_an_even_flux_makes_the_translation_phase_trivial(self):
        # 36 flux quanta is EVEN, so n_phi*m*n is always even and the phase is 1 for
        # every lattice vector (to roundoff: exp(i pi k) is evaluated in floating
        # point, so "1" here means 1 + O(1e-14)).  This is why the Gaussian basis' explicit
        # `l_ints[:,0]*l_ints[:,1]*n_phi*pi` term is a constant in this system, and it
        # is a real statement about nu = 1 on a 6x6 torus, not a coding accident.
        Ge = geom()
        lints, _ = legacy.circular_lattice(30.0, Ge.L1, Ge.L2)
        np.testing.assert_allclose(mc.magnetic_translation_phase(lints, 36.0),
                                   np.ones(len(lints)), rtol=0, atol=1e-13)

    def test_an_odd_flux_is_not_trivial(self):
        # with ONE flux quantum the phase is (-1)^(m*n): a genuine sign that flips on
        # half the lattice vectors.  This is the factor the Landau-level basis carries,
        # and dropping it there is not a no-op.
        ai, _ = legacy.circular_lattice(8.5 * A_WC, *g.triangular_cell(PRIM_AREA))
        phase = mc.magnetic_translation_phase(ai, 1.0)
        np.testing.assert_allclose(phase, (-1.0) ** (ai[:, 0] * ai[:, 1]),
                                   rtol=0, atol=1e-13)
        self.assertGreater(float(np.mean(phase.real < 0)), 0.25)

    def test_the_two_flux_counts_give_different_phases_only_for_odd_products(self):
        # the precise relation between the two cells' phases, recorded so that the
        # "36 vs 1" difference is not mistaken for a factor of 36
        ai, _ = legacy.circular_lattice(8.5 * A_WC, *g.triangular_cell(PRIM_AREA))
        a = mc.magnetic_translation_phase(ai, 1.0)
        b = mc.magnetic_translation_phase(ai, 36.0)
        odd = ((ai[:, 0] * ai[:, 1]) % 2 != 0)
        np.testing.assert_allclose(a[odd], -1.0, rtol=0, atol=1e-13)
        np.testing.assert_allclose(a[~odd], 1.0, rtol=0, atol=1e-13)
        # roundoff grows with the argument: m*n*36 reaches ~3600 here, so exp(i pi k)
        # lands within 5e-13 of +1 rather than 1e-14.  Both this module and the frozen
        # engine evaluate the same formula, so the two agree bit-for-bit -- but neither
        # is exactly +-1, and the tolerance has to be the argument's, not a blanket 1e-15.
        np.testing.assert_allclose(b, 1.0, rtol=0, atol=1e-11)

    def test_rejects_a_bad_shape(self):
        with self.assertRaises(ValueError):
            mc.magnetic_translation_phase(np.zeros(3), 1.0)


class TestPrimitiveFrame(unittest.TestCase):
    def test_columns_are_the_vectors_and_the_inverse_is_exact(self):
        A1, A2 = g.triangular_cell(PRIM_AREA)
        C, Ci = mc.primitive_frame(A1, A2)
        np.testing.assert_allclose(C[:, 0], A1, rtol=0, atol=0)
        np.testing.assert_allclose(C[:, 1], A2, rtol=0, atol=0)
        np.testing.assert_allclose(C @ Ci, np.eye(2), rtol=0, atol=1e-15)
        np.testing.assert_allclose(Ci @ C, np.eye(2), rtol=0, atol=1e-15)


class TestFold(unittest.TestCase):
    def setUp(self):
        self.Ge = geom()
        self.C, self.Ci = mc.primitive_frame(self.Ge.A1, self.Ge.A2)
        self.rng = np.random.default_rng(1234)
        self.pts = self.rng.uniform(-6.0, 6.0, size=(400, 2))

    def test_matches_legacy_on_the_primitive_frame(self):
        mine = mc.fold_to_primitive_cell(self.pts, self.C, self.Ci)
        theirs = legacy.send_to_first_cell(self.pts, self.C, self.Ci)
        for a, b in zip(mine, theirs):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)

    def test_matches_legacy_on_the_supercell_frame(self):
        mine = mc.fold_to_primitive_cell(self.pts, self.Ge.sc_to_cart,
                                         self.Ge.cart_to_sc)
        theirs = legacy.send_to_first_cell(self.pts, self.Ge.sc_to_cart,
                                           self.Ge.cart_to_sc)
        for a, b in zip(mine, theirs):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)

    def test_the_four_returned_objects_are_consistent(self):
        r_lat, r_cart, shift_lat, shift_cart = mc.fold_to_primitive_cell(
            self.pts, self.C, self.Ci)
        # shift_cart really is shift_lat in Cartesian
        np.testing.assert_allclose(shift_lat @ self.C.T, shift_cart, rtol=0, atol=1e-14)
        # shift_lat is integral
        np.testing.assert_allclose(shift_lat, np.round(shift_lat), rtol=0, atol=1e-12)
        # and the fold is the round trip
        np.testing.assert_allclose(r_lat @ self.C.T, r_cart, rtol=0, atol=1e-14)
        np.testing.assert_allclose(r_cart + shift_cart, self.pts, rtol=0, atol=1e-14)

    def test_folded_point_is_inside_the_cell(self):
        r_lat, *_ = mc.fold_to_primitive_cell(self.pts, self.C, self.Ci)
        self.assertLessEqual(float(np.abs(r_lat).max()), 0.5 + 1e-12)

    def test_half_cell_edges_pin_the_rounding_convention(self):
        # np.round is half-to-even.  Both +0.5 and -0.5 are fixed points of t - round(t)
        # (0.5 - 0 = 0.5, -0.5 + 0 = -0.5), so the folded range is the CLOSED
        # [-0.5, 0.5] and the half-cell points are the two ends of it.  Pinned here
        # because the orbital phases depend on which representative each k is given.
        #
        # Done on a diagonal frame (2*I), where Cartesian -> fractional is an exact
        # division by two, so the test measures the rounding rule and not the round trip
        # through an oblique basis.
        C = 2.0 * np.eye(2)
        Ci = np.linalg.inv(C)
        halves = np.array([[0.5, 0.0], [-0.5, 0.0], [0.5, 0.5], [-0.5, -0.5],
                           [1.5, 2.5], [2.5, 1.5]])
        r_lat, _, shift_lat, _ = mc.fold_to_primitive_cell(halves @ C.T, C, Ci)
        np.testing.assert_allclose(r_lat[0], [0.5, 0.0], rtol=0, atol=0)
        np.testing.assert_allclose(r_lat[1], [-0.5, 0.0], rtol=0, atol=0)
        np.testing.assert_allclose(r_lat[2], [0.5, 0.5], rtol=0, atol=0)
        np.testing.assert_allclose(r_lat[3], [-0.5, -0.5], rtol=0, atol=0)
        # 1.5 -> round(1.5) = 2 -> -0.5 ;  2.5 -> round(2.5) = 2 -> +0.5
        np.testing.assert_allclose(r_lat[4], [-0.5, 0.5], rtol=0, atol=0)
        np.testing.assert_allclose(r_lat[5], [0.5, -0.5], rtol=0, atol=0)
        np.testing.assert_allclose(shift_lat[4], [2.0, 2.0], rtol=0, atol=0)
        np.testing.assert_allclose(shift_lat[5], [2.0, 2.0], rtol=0, atol=0)

    def test_offsets_land_strictly_inside(self):
        C, Ci = 2.0 * np.eye(2), 0.5 * np.eye(2)
        off = np.array([[0.5 + 1e-9, 0.0], [0.5, -1e-9], [-0.5 - 1e-9, 0.0]])
        r_lat, *_ = mc.fold_to_primitive_cell(off @ C.T, C, Ci)
        np.testing.assert_allclose(r_lat[0], [-0.5 + 1e-9, 0.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(r_lat[1], [0.5, -1e-9], rtol=0, atol=1e-15)
        np.testing.assert_allclose(r_lat[2], [0.5 - 1e-9, 0.0], rtol=0, atol=1e-15)

    def test_is_idempotent_and_shape_preserving(self):
        a = mc.fold_to_primitive_cell(self.pts, self.C, self.Ci)[1]
        b = mc.fold_to_primitive_cell(a, self.C, self.Ci)[1]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-13)
        self.assertEqual(mc.fold_to_primitive_cell(np.zeros(2), self.C, self.Ci)[1].shape,
                         (1, 2))

    def test_rejects_a_bad_shape(self):
        with self.assertRaises(ValueError):
            mc.fold_to_primitive_cell(np.zeros((4, 3)), self.C, self.Ci)


class TestGaugeString(unittest.TestCase):
    def test_the_gauge_is_the_one_the_orbital_code_assumes(self):
        # Not decoration: the string is what a future reader greps for, and it must
        # name the same sign the Landau basis' finite-difference test pins.
        self.assertIn("curl A = +B z_hat", mc.GAUGE)
        self.assertIn("B = 1", mc.GAUGE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
