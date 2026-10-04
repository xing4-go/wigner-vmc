"""`physics/geometry.py` against the frozen engine, on fixed inputs.

Two kinds of check, and the second is the one that matters:

* **against legacy** -- element-wise on the same inputs, at machine precision;
* **against the mathematics** -- the property the function exists for, checked without
  reference to legacy at all.  A port that reproduces a bug reproduces it in both
  copies and only the second kind of test notices.

No Monte Carlo, no RNG without a seed, no optimisation: every input here is a fixed
array.
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
    sys.path.insert(0, ROOT)          # the legacy engine is a top-level module

import qhvmc_engine as legacy                                    # noqa: E402
from wigner_vmc.physics import geometry as g                     # noqa: E402

RTOL = 1e-13
ATOL = 1e-14

PRIM_AREA = 2.0 * np.pi          # one primitive cell: one flux quantum, one electron
AREA_PROD = 36.0 * PRIM_AREA     # the production supercell = 72 pi
A_WC = np.sqrt(4 * np.pi / np.sqrt(3))


def prod_clean(magnetic=True, fermi=False, rl_cut=4.0):
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, rl_cut, magnetic=magnetic,
                                fermi_surface=fermi)


def prod_legacy(rl_cut=4.0, B_field=True, fermi_surface=False):
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return legacy.geometry_setup(A1, A2, 6, 6, rl_cut, B_field, fermi_surface), A1, A2


class TestCrossZHat(unittest.TestCase):
    def test_matches_legacy(self):
        for v in ([1.0, 0.0], [0.0, 1.0], [A_WC, A_WC / 2], [-2.5, 3.25],
                  [0.0, 0.0], [1e-8, -1e-8]):
            np.testing.assert_allclose(g.cross_z_hat(v), legacy.cross_z_hat(v),
                                       rtol=0, atol=0)

    def test_is_a_quarter_turn_with_the_right_handedness(self):
        # z_hat x V = (V_y, -V_x).  Rotating (1,0) must give (0,-1), not (0,+1):
        # the sign of every reciprocal vector in the package follows from this.
        np.testing.assert_allclose(g.cross_z_hat([1.0, 0.0]), [0.0, -1.0])
        # and it is orthogonal with the same norm
        v = np.array([0.7, -1.9])
        w = g.cross_z_hat(v)
        self.assertAlmostEqual(float(v @ w), 0.0, places=14)
        self.assertAlmostEqual(float(np.linalg.norm(w)), float(np.linalg.norm(v)),
                               places=14)


class TestCircularLattice(unittest.TestCase):
    def test_matches_legacy_on_the_bloch_sum_basis(self):
        # the LL basis' Bloch sum: primitive vectors, cutoff 8.5 |A1|
        A1, A2 = g.triangular_cell(PRIM_AREA)
        cut = 8.5 * np.linalg.norm(A1)
        ci, cc = g.circular_lattice(cut, A1, A2)
        li, lc = legacy.circular_lattice(cut, A1, A2)
        np.testing.assert_array_equal(ci, li)
        np.testing.assert_allclose(cc, lc, rtol=0, atol=0)

    def test_matches_legacy_on_the_supercell_basis(self):
        geom = prod_clean()
        ci, cc = g.circular_lattice(30.0, geom.L1, geom.L2)
        li, lc = legacy.circular_lattice(30.0, geom.L1, geom.L2)
        np.testing.assert_array_equal(ci, li)
        np.testing.assert_allclose(cc, lc, rtol=0, atol=0)

    def test_int_and_cart_are_index_aligned(self):
        A1, A2 = g.triangular_cell(PRIM_AREA)
        ints, carts = g.circular_lattice(3.0 * A_WC, A1, A2)
        rebuilt = ints[:, 0:1] * A1[None, :] + ints[:, 1:2] * A2[None, :]
        np.testing.assert_allclose(carts, rebuilt, rtol=0, atol=1e-15)

    def test_the_set_does_not_depend_on_the_enumerating_basis(self):
        # The 60-degree cell has a second, equally valid basis: A1' = A1, A2' = A2 - A1.
        # The SET of lattice vectors within the cutoff must be identical even though the
        # enumeration order is not.
        A1, A2 = g.triangular_cell(PRIM_AREA)
        cut = 5.0
        _, c1 = g.circular_lattice(cut, A1, A2)
        _, c2 = g.circular_lattice(cut, A1, A2 - A1)
        s1 = np.array(sorted(map(tuple, np.round(c1, 10))))
        s2 = np.array(sorted(map(tuple, np.round(c2, 10))))
        np.testing.assert_allclose(s1, s2, rtol=0, atol=1e-9)

    def test_output_is_not_sorted_by_norm(self):
        # A documented property, asserted so that nobody "tidies" it into a sort and
        # silently changes the order of every Gauss-Legendre-free lattice sum.
        A1, A2 = g.triangular_cell(PRIM_AREA)
        _, carts = g.circular_lattice(4.0 * A_WC, A1, A2)
        norms = np.linalg.norm(carts, axis=1)
        self.assertFalse(bool(np.all(np.diff(norms) >= -1e-15)),
                         "circular_lattice came out sorted; the contract says it is not")


class TestGeometrySetup(unittest.TestCase):
    def _compare(self, geom, tup):
        area, L1, L2, G1, G2, mesh, g1, g2, RL = tup
        self.assertAlmostEqual(geom.area, area, places=13)
        for name, mine, theirs in (("L1", geom.L1, L1), ("L2", geom.L2, L2),
                                   ("G1", geom.G1, G1), ("G2", geom.G2, G2),
                                   ("g1", geom.g1, g1), ("g2", geom.g2, g2)):
            np.testing.assert_allclose(mine, theirs, rtol=RTOL, atol=ATOL,
                                       err_msg=f"{name} differs")
        # `mesh` order IS part of the contract: row order, i outer
        np.testing.assert_allclose(geom.mesh, mesh, rtol=0, atol=1e-15)
        np.testing.assert_allclose(geom.RL, RL, rtol=RTOL, atol=ATOL)

    def test_production_cell(self):
        self._compare(prod_clean(), prod_legacy()[0])

    def test_smaller_cell(self):
        A1, A2 = g.triangular_cell(16.0 * PRIM_AREA)
        geom = g.Geometry.from_cell(A1, A2, 4, 4, 4.0)
        self._compare(geom, legacy.geometry_setup(A1, A2, 4, 4, 4.0, True))

    def test_oblique_non_production_cell(self):
        # not 60 degrees, and N1 != N2 -- the reciprocal construction must not be a
        # triangular special case
        A1, A2 = np.array([1.0, 0.0]), np.array([0.3, 1.1])
        geom = g.Geometry.from_cell(A1, A2, 4, 3, 3.0)
        self._compare(geom, legacy.geometry_setup(A1, A2, 4, 3, 3.0, True))

    def test_non_magnetic_branches(self):
        # Dead code for both production generators, migrated verbatim and pinned here
        # so the inventory is complete.
        A1, A2 = g.triangular_cell(PRIM_AREA)
        for fermi in (False, True):
            geom = g.Geometry.from_cell(A1, A2, 6, 6, 4.0, magnetic=False,
                                        fermi_surface=fermi)
            self._compare(geom, legacy.geometry_setup(A1, A2, 6, 6, 4.0, False, fermi))

    def test_derived_constants(self):
        geom = prod_clean()
        self.assertAlmostEqual(geom.area, AREA_PROD, places=12)
        self.assertEqual(geom.n_cells, 36)
        self.assertAlmostEqual(geom.cell_area, 2 * np.pi, places=13)
        self.assertAlmostEqual(geom.a_wc, A_WC, places=13)
        self.assertAlmostEqual(abs(np.linalg.norm(geom.g1) / np.linalg.norm(geom.G1)),
                               6.0, places=12)
        # sqrt_n is a DENSITY to the 1/2, sqrt(ne/area) = 1/sqrt(2 pi) = 0.398942..., and
        # NOT 1/a_wc = 0.371235... -- the two are close enough to swap by accident, so
        # both the value and the gap are asserted
        self.assertAlmostEqual(geom.sqrt_n, 1.0 / np.sqrt(2 * np.pi), places=13)
        self.assertAlmostEqual(geom.sqrt_n, np.sqrt(36.0 / AREA_PROD), places=13)
        self.assertAlmostEqual(1.0 / geom.a_wc, 0.37125762, places=6)
        self.assertGreater(abs(geom.sqrt_n - 1.0 / geom.a_wc), 0.02)

    def test_sc_and_cart_to_sc_are_inverses(self):
        geom = prod_clean()
        np.testing.assert_allclose(geom.sc_to_cart @ geom.cart_to_sc, np.eye(2),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(geom.sc_to_cart[:, 0], geom.L1, rtol=0, atol=0)
        np.testing.assert_allclose(geom.sc_to_cart[:, 1], geom.L2, rtol=0, atol=0)


class TestWrapping(unittest.TestCase):
    def setUp(self):
        self.geom = prod_clean()
        self.rng = np.random.default_rng(20260930)
        self.pts = self.rng.uniform(-8.0, 8.0, size=(400, 2))

    def test_matches_legacy(self):
        mine = g.wrap_to_supercell(self.pts, self.geom.sc_to_cart, self.geom.cart_to_sc)
        theirs = legacy.send_to_first_supercell(self.pts, self.geom.sc_to_cart,
                                                self.geom.cart_to_sc)
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-15)

    def test_result_is_inside_the_cell(self):
        out = g.wrap_to_supercell(self.pts, self.geom.sc_to_cart, self.geom.cart_to_sc)
        frac = out @ self.geom.cart_to_sc.T
        self.assertLess(float(np.abs(frac).max()), 0.5000001)

    def test_is_idempotent(self):
        once = g.wrap_to_supercell(self.pts, self.geom.sc_to_cart, self.geom.cart_to_sc)
        twice = g.wrap_to_supercell(once, self.geom.sc_to_cart, self.geom.cart_to_sc)
        np.testing.assert_allclose(once, twice, rtol=0, atol=1e-14)


class TestMinimumImage(unittest.TestCase):
    def setUp(self):
        self.geom = prod_clean()
        self.rng = np.random.default_rng(7)

    def test_matches_legacy(self):
        d = self.rng.uniform(-3.0, 3.0, size=(2000, 2))
        mine = g.minimum_image(d, self.geom.sc_to_cart, self.geom.cart_to_sc)
        theirs = legacy.minimum_image_displacement(d, self.geom.sc_to_cart,
                                                   self.geom.cart_to_sc)
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-14)

    def test_is_the_actual_minimum_against_brute_force(self):
        # The property the function exists for, checked WITHOUT legacy: no image
        # n1*L1 + n2*L2 with n in [-4,4]^2 may be shorter.
        d = self.rng.uniform(-3.0, 3.0, size=(600, 2))
        best = g.minimum_image(d, self.geom.sc_to_cart, self.geom.cart_to_sc)
        n = np.array([(i, j) for i in range(-4, 5) for j in range(-4, 5)], dtype=float)
        images = (d[:, None, :] + n[None, :, :] @ self.geom.sc_to_cart.T)
        norms = np.linalg.norm(images, axis=-1)
        self.assertLessEqual(float(np.abs(np.linalg.norm(best, axis=1)
                                          - norms.min(axis=1)).max()), 1e-12)

    def test_the_documented_counterexample(self):
        # L1 = (1,0), L2 = (1/2, sqrt3/2); fractional displacement (0.49, 0.49).
        L1 = np.array([1.0, 0.0])
        L2 = np.array([0.5, np.sqrt(3.0) / 2.0])
        sc = np.column_stack([L1, L2])
        d = 0.49 * L1 + 0.49 * L2
        self.assertAlmostEqual(float(np.linalg.norm(d)), 0.8487049, places=6)
        out = g.minimum_image(d, sc, np.linalg.inv(sc))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.5002999, places=6)
        # the fold, by contrast, returns the point itself -- which is exactly why the
        # two functions must not be interchanged
        folded = g.wrap_to_supercell(d, sc, np.linalg.inv(sc))
        np.testing.assert_allclose(folded, d[None, :], rtol=0, atol=1e-13)
        self.assertLess(float(np.linalg.norm(out)), float(np.linalg.norm(folded)))

    def test_shape_is_preserved(self):
        sc, ci = self.geom.sc_to_cart, self.geom.cart_to_sc
        self.assertEqual(g.minimum_image(np.zeros(2), sc, ci).shape, (2,))
        self.assertEqual(g.minimum_image(np.zeros((5, 2)), sc, ci).shape, (5, 2))
        self.assertEqual(g.minimum_image(np.zeros((5, 3, 2)), sc, ci).shape, (5, 3, 2))


class TestWignerCrystalSites(unittest.TestCase):
    def test_matches_legacy(self):
        A1, A2 = g.triangular_cell(PRIM_AREA)
        mine = g.wigner_crystal_sites(6, 6, A1, A2)
        theirs = legacy.gaussian_sites(6, 6, A1, A2)
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-15)

    def test_matches_legacy_off_the_production_point(self):
        A1, A2 = g.triangular_cell(16.0 * PRIM_AREA)
        np.testing.assert_allclose(g.wigner_crystal_sites(4, 4, A1, A2),
                                   legacy.gaussian_sites(4, 4, A1, A2),
                                   rtol=0, atol=1e-15)

    def test_sites_are_distinct_and_the_lattice_constant_comes_out(self):
        A1, A2 = g.triangular_cell(PRIM_AREA)
        sites = g.wigner_crystal_sites(6, 6, A1, A2)
        self.assertEqual(sites.shape, (36, 2))
        geom = prod_clean()
        d = sites[:, None, :] - sites[None, :, :]
        d = g.minimum_image(d, geom.sc_to_cart, geom.cart_to_sc)
        norms = np.linalg.norm(d, axis=-1) + np.eye(36) * 1e9
        self.assertAlmostEqual(float(norms.min()), A_WC, places=12)
        # each site has exactly six neighbours at a_WC
        self.assertTrue(bool(np.all((norms < A_WC * 1.0001).sum(axis=1) == 6)))

    def test_the_set_is_centred_on_the_origin_not_on_a_corner(self):
        # Documented: the fold centres the set, so it does not tile [0,L1)x[0,L2).
        # This is what put 20 of 36 site markers off the axes in the Part IV figure,
        # where the cell outline ran from 0 to L1 and the sites did not.
        A1, A2 = g.triangular_cell(PRIM_AREA)
        sites = g.wigner_crystal_sites(6, 6, A1, A2)
        geom = prod_clean()
        for ax in (0, 1):
            self.assertLess(float(sites[:, ax].min()), 0.0)
            self.assertGreater(float(sites[:, ax].max()), 0.0)
        # every site lies within one supercell radius of the origin
        self.assertLess(float(np.linalg.norm(sites, axis=1).max()),
                        float(np.linalg.norm(geom.L1)))
        # and NOT all of them are inside [0,L1)x[0,L2)
        frac = sites @ geom.cart_to_sc.T
        self.assertGreater(int(np.any((frac < 0) | (frac >= 1), axis=1).sum()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
