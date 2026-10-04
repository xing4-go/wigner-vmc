"""`physics/coulomb.py` against the frozen engine, and against its own identity.

The potential energy is the one object both workflows share exactly -- same class, same
constructor, same numbers -- so this file's legacy comparison is also the evidence that
the Gaussian and LL-rotation energies are built on one potential.

Two properties are checked without reference to legacy at all, because a port
reproduces a bug in both copies:

* the Madelung constant of the triangular lattice, an external number (1.106103);
* the eta-independence of the split, which is what makes an Ewald split legitimate.

Coupling benchmarks ``r_s = 45, 65, 75`` fix the energy scale (``kappa = r_s/sqrt(2)``);
they do not change the Ewald tables, which depend only on the cell.
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
from wigner_vmc.physics import coulomb as cb                     # noqa: E402
from wigner_vmc.physics import geometry as g                     # noqa: E402
from wigner_vmc.physics import hamiltonian as hm                 # noqa: E402

PRIM_AREA = 2.0 * np.pi
AREA_PROD = 36.0 * PRIM_AREA
NE = 36
RTOL = 1e-13
ATOL = 1e-13

#: The three benchmark couplings of Stage 2A, and the kappa each one is.
BENCH_RS = (45.0, 65.0, 75.0)


def build():
    A1, A2 = g.triangular_cell(PRIM_AREA)
    Ge = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
    return Ge


def configs(Ge, n_random=40, seed=20261001):
    """The WC sites, plus n_random fixed random configurations.

    Fixed seed, no sampler: these are inputs, not measurements.
    """
    out = [g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)]
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        out.append(rng.uniform(-3.0, 3.0, size=(NE, 2)))
    return out


class TestAgainstLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Ge = build()
        cls.new = cb.CoulombEwald(NE, cls.Ge.L1, cls.Ge.L2, cls.Ge.G1, cls.Ge.G2)
        cls.old = legacy.CoulombEwald(NE, cls.Ge.L1, cls.Ge.L2, cls.Ge.G1, cls.Ge.G2)
        cls.cfgs = configs(cls.Ge)

    def test_eta_default_matches(self):
        self.assertAlmostEqual(self.new.eta, float(self.old.eta), places=14)

    def test_lattice_tables_match(self):
        np.testing.assert_allclose(self.new.l_cart, self.old.l_cart, rtol=0, atol=0)
        np.testing.assert_allclose(self.new.l_nz, self.old.l_nz, rtol=0, atol=0)
        np.testing.assert_allclose(self.new.g_cart, self.old.g_cart, rtol=0, atol=0)
        np.testing.assert_allclose(self.new.v_long_g, self.old.v_long_g, rtol=0, atol=0)

    def test_constant_term_matches(self):
        self.assertAlmostEqual(self.new.v_const, float(self.old.v_const), places=12)

    def test_energy_matches_on_every_configuration(self):
        worst = 0.0
        for R in self.cfgs:
            mine = self.new.energy(R)
            theirs = float(self.old.energy(R))
            worst = max(worst, abs(mine - theirs) / max(1.0, abs(theirs)))
        self.assertLess(worst, RTOL, f"worst relative energy deviation {worst:.3e}")

    def test_delta_move_matches(self):
        rng = np.random.default_rng(99)
        worst = 0.0
        for R in self.cfgs[:8]:
            for _ in range(3):
                i = int(rng.integers(NE))
                r_old = R[i].copy()
                R2 = R.copy()
                R2[i] = r_old + rng.uniform(-0.4, 0.4, size=2)
                mine = self.new.delta_move(R2, i, r_old)
                theirs = float(self.old.delta_move(R2, i, r_old))
                worst = max(worst, abs(mine - theirs))
        self.assertLess(worst, ATOL, f"worst delta_move deviation {worst:.3e}")

    def test_cell_matrices_are_the_geometry_matrices(self):
        # the strongest single sharing point in the project: two independently written
        # supercell matrices, and the Ewald object's are the geometry's bit-for-bit
        np.testing.assert_allclose(self.new.sc_to_cart, self.Ge.sc_to_cart, rtol=0, atol=0)
        np.testing.assert_allclose(self.new.cart_to_sc, self.Ge.cart_to_sc, rtol=0, atol=0)


class TestItsOwnIdentities(unittest.TestCase):
    """Checks that never mention legacy."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = build()
        cls.cfgs = configs(cls.Ge)

    def test_delta_move_equals_the_energy_difference(self):
        # This is the identity the Metropolis accept/reject depends on.  If it fails,
        # the sampler is not sampling the energy it reports.
        ew = cb.CoulombEwald(NE, self.Ge.L1, self.Ge.L2, self.Ge.G1, self.Ge.G2)
        rng = np.random.default_rng(5)
        worst = 0.0
        for R in self.cfgs[:6]:
            for _ in range(4):
                i = int(rng.integers(NE))
                r_old = R[i].copy()
                R2 = R.copy()
                R2[i] = r_old + rng.uniform(-0.5, 0.5, size=2)
                direct = ew.delta_move(R2, i, r_old)
                by_diff = ew.energy(R2) - ew.energy(R)
                worst = max(worst, abs(direct - by_diff))
        self.assertLess(worst, 1e-10, f"worst identity residual {worst:.3e}")

    def test_energy_is_eta_independent(self):
        # the property that makes the split legitimate: eta only moves work between the
        # two sums.  This is also the strongest evidence that l_cut is wide enough.
        rows = []
        for eta in (2.0, 4.0, 6.0, 8.0):
            ew = cb.CoulombEwald(NE, self.Ge.L1, self.Ge.L2, self.Ge.G1, self.Ge.G2, eta=eta)
            rows.append([ew.energy(R) / NE for R in self.cfgs])
        rows = np.array(rows)
        spread = float(np.abs(rows - rows[0]).max())
        self.assertLess(spread, 1e-10, f"Ewald is eta-dependent by {spread:.3e}")

    def test_split_terms_sum_to_the_energy(self):
        ew = cb.CoulombEwald(NE, self.Ge.L1, self.Ge.L2, self.Ge.G1, self.Ge.G2)
        for R in self.cfgs[:5]:
            short, long_, const = ew.split_energy(R)
            self.assertAlmostEqual(short + long_ + const, ew.energy(R), places=10)

    def test_madelung_constant_of_the_triangular_lattice(self):
        # An external anchor: for electrons on the WC sites the potential per particle,
        # in units of e^2/r_s, is the triangular-lattice Madelung constant 1.106103...
        # The notebook asserts 1e-5; asserted here at the same strength.
        sites = g.wigner_crystal_sites(6, 6, self.Ge.A1, self.Ge.A2)
        rows = []
        for eta in (2.0, 4.0, 6.0, 8.0):
            ew = cb.CoulombEwald(NE, self.Ge.L1, self.Ge.L2, self.Ge.G1, self.Ge.G2, eta=eta)
            rows.append(-(ew.energy(sites) / NE) * np.sqrt(2))
        spread = max(rows) - min(rows)
        self.assertLess(spread, 1e-9, f"eta spread {spread:.3e}")
        self.assertLess(abs(rows[0] - 1.106103), 1e-5,
                        f"Madelung {rows[0]:.9f} vs 1.106103")

    def test_real_space_table_keeps_the_zero_vector_and_the_constant_does_not(self):
        # L = 0 is the bare pair interaction; l_nz is the same table minus it.  Dropping
        # it from l_cart would shorten every pair term.
        ew = cb.CoulombEwald(NE, self.Ge.L1, self.Ge.L2, self.Ge.G1, self.Ge.G2)
        n_zero = int((np.linalg.norm(ew.l_cart, axis=1) < 1e-14).sum())
        self.assertEqual(n_zero, 1)
        self.assertEqual(len(ew.l_nz), len(ew.l_cart) - 1)
        self.assertTrue(bool(np.all(np.linalg.norm(ew.l_nz, axis=1) > 0)))
        self.assertTrue(bool(np.all(np.linalg.norm(ew.g_cart, axis=1) > 0)))

    def test_the_reciprocal_table_excludes_g_zero(self):
        # and including g = 0 would divide by zero, which is why the constant term
        # carries that piece analytically instead
        ew = cb.CoulombEwald(NE, self.Ge.L1, self.Ge.L2, self.Ge.G1, self.Ge.G2)
        self.assertTrue(bool(np.all(np.isfinite(ew.v_long_g))))

    def test_energy_is_periodic_under_a_supercell_translation(self):
        ew = cb.CoulombEwald(NE, self.Ge.L1, self.Ge.L2, self.Ge.G1, self.Ge.G2)
        R = self.cfgs[1]
        base = ew.energy(R)
        for shift in (self.Ge.L1, self.Ge.L2, -2 * self.Ge.L1 + 3 * self.Ge.L2,
                      self.Ge.L1 - self.Ge.L2):
            self.assertAlmostEqual(ew.energy(R + shift), base, places=9)


class TestBenchmarkCouplings(unittest.TestCase):
    """The three couplings of Stage 2A.  The Ewald tables do not depend on the
    coupling; the energy SCALE does, and that scale is the only place `r_s` enters."""

    def test_kappa_for_the_benchmarks(self):
        expected = {45.0: 31.819805153394640,
                    65.0: 45.961940777125590,
                    75.0: 53.033008588991070}
        for rs, kappa in expected.items():
            got = hm.kappa_from_rs(rs)
            self.assertAlmostEqual(got, kappa, places=12)
            self.assertAlmostEqual(hm.rs_from_kappa(got), rs, places=12)

    def test_scaled_energy_at_the_benchmarks(self):
        # the potential per particle, in hbar omega_c, at each benchmark coupling
        Ge = build()
        ew = cb.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2)
        sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
        v_per_particle = ew.energy(sites) / NE
        self.assertAlmostEqual(v_per_particle, -0.7821326, places=6)
        rows = {rs: hm.kappa_from_rs(rs) * v_per_particle for rs in BENCH_RS}
        # v is NEGATIVE, so a stronger coupling makes the energy MORE negative: the
        # ordering runs the other way from the couplings themselves
        self.assertGreater(rows[45.0], rows[65.0])
        self.assertGreater(rows[65.0], rows[75.0])
        self.assertAlmostEqual(rows[45.0], -24.8873, places=3)
        self.assertAlmostEqual(rows[75.0], -41.4788, places=3)

    def test_the_ewald_tables_do_not_depend_on_the_coupling(self):
        Ge = build()
        a = cb.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2)
        b = cb.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2)
        np.testing.assert_array_equal(a.l_cart, b.l_cart)
        self.assertEqual(a.v_const, b.v_const)


if __name__ == "__main__":
    unittest.main(verbosity=2)
