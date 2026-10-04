"""The conventions themselves: units, ``r_s`` vs ``kappa``, and the ``nmax`` rule.

These are the tests that would fail if somebody "tidied" a convention into a more
standard-looking one.  They are deliberately about numbers that are recorded elsewhere
in the project -- the frozen coupling grid, the rung ladder of the analysis layer, the
production geometry -- so that a change here shows up as a disagreement with the frozen
tree rather than as a quietly different plot.

Nothing here evaluates an energy.
"""
import inspect
import math
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
from wigner_vmc.analysis import energy_competition as ec         # noqa: E402
from wigner_vmc.analysis import structure as st                  # noqa: E402
from wigner_vmc.physics import geometry as g                     # noqa: E402
from wigner_vmc.physics import hamiltonian as hm                 # noqa: E402
from wigner_vmc.physics import landau_levels as ll               # noqa: E402

PRIM_AREA = 2.0 * np.pi
AREA_PROD = 36.0 * PRIM_AREA
A_WC = np.sqrt(4 * np.pi / np.sqrt(3))
MAP = os.path.join(CLEAN, "PHYSICS_MAP.md")


class TestTheLockedRelation(unittest.TestCase):
    """``r_s = sqrt(2) kappa`` at nu = 1.  Locked: every stored number depends on it."""

    def test_benchmark_values(self):
        self.assertAlmostEqual(hm.rs_from_kappa(2.0), 2.8284271247461903, places=13)
        self.assertAlmostEqual(hm.rs_from_kappa(32.0), 45.254833995939045, places=12)
        self.assertAlmostEqual(hm.rs_from_kappa(80.0), 113.13708498984761, places=12)

    def test_round_trip(self):
        for kappa in hm.PRODUCTION_KAPPAS:
            self.assertAlmostEqual(hm.kappa_from_rs(hm.rs_from_kappa(kappa)), kappa,
                                   places=12)

    def test_matches_the_analysis_layer(self):
        # KappaScan.rs is the frozen conversion the figures were drawn with.  The clean
        # one writes `kappa / sqrt(nu/2)` and the analysis layer writes
        # `kappas * math.sqrt(2.0)`: algebraically the same expression at nu = 1, and
        # 1 ULP apart in floating point (measured: 1.42e-14 absolute, 2.09e-16
        # relative, over the ten production couplings).  The tolerance is stated as
        # that, rather than either implementation being rewritten to chase bit
        # identity -- the frozen layer is part of the figure contract and the clean
        # formula is the one the parameter file documents.
        scan = ec.KappaScan(list(hm.PRODUCTION_KAPPAS), [0.0] * 10, [0.0] * 10,
                            [0.0] * 10, [0.0] * 10, [0.0] * 10)
        np.testing.assert_allclose(scan.rs,
                                   [hm.rs_from_kappa(k) for k in hm.PRODUCTION_KAPPAS],
                                   rtol=1e-15, atol=0)

    def test_general_nu(self):
        # kappa = r_s sqrt(nu/2); nu = 1 is the only value the campaign used, but the
        # formula must not be hard-coded to it
        self.assertAlmostEqual(hm.kappa_from_rs(10.0, nu=2.0), 10.0, places=13)
        self.assertAlmostEqual(hm.rs_from_kappa(10.0, nu=2.0), 10.0, places=13)
        self.assertAlmostEqual(hm.kappa_from_rs(10.0, nu=0.5), 5.0, places=13)

    def test_validation_lives_in_the_model_not_in_the_conversion(self):
        # `kappa_from_rs` and `rs_from_kappa` are pure conversions.  A negative r_s
        # describes no physical model but is still a number, and the conversion returns
        # it; the contract that refuses an unphysical model is ModelParams'.  Pinned in
        # this direction so that a later "defensive" guard added to the conversion
        # function -- which would make every caller inherit a validation side effect --
        # fails here.
        self.assertAlmostEqual(hm.kappa_from_rs(-1.0), -1.0 / math.sqrt(2.0), places=15)
        self.assertEqual(hm.rs_from_kappa(0.0), 0.0)
        with self.assertRaises(ValueError):
            hm.ModelParams(kappa=0.0)
        with self.assertRaises(ValueError):
            hm.ModelParams(kappa=-3.0)


class TestConstants(unittest.TestCase):
    def test_units_are_the_documented_ones(self):
        self.assertEqual(hm.HBAR_OMEGA_C, 1.0)
        self.assertEqual(hm.MAGNETIC_LENGTH, 1.0)
        self.assertIn("curl A = +B z_hat", hm.UNITS["gauge"])
        self.assertIn("kappa", hm.UNITS["charge"])

    def test_production_numbers(self):
        self.assertEqual(hm.PRODUCTION_N_ELECTRONS, 36)
        self.assertEqual(hm.PRODUCTION_N_CELLS_PER_SIDE, 6)
        Ge = hm.production_geometry()
        self.assertAlmostEqual(Ge.area, AREA_PROD, places=12)
        self.assertAlmostEqual(Ge.a_wc, A_WC, places=13)
        self.assertEqual(Ge.mesh.shape, (36, 2))

    def test_the_frozen_coupling_grid_is_the_frozen_one(self):
        self.assertEqual(tuple(hm.PRODUCTION_KAPPAS), tuple(ec.KAPPAS_PROTO))
        self.assertEqual(len(hm.PRODUCTION_KAPPAS), 10)
        self.assertEqual(hm.PRODUCTION_KAPPAS[0], 2.0)
        self.assertEqual(hm.PRODUCTION_KAPPAS[-1], 80.0)

    def test_the_two_geometry_module_constants_are_not_interchangeable(self):
        # A_WC is a distance, sqrt_n is a density to the 1/2
        Ge = hm.production_geometry()
        self.assertAlmostEqual(Ge.sqrt_n, math.sqrt(36.0 / AREA_PROD), places=13)
        self.assertGreater(abs(Ge.a_wc - 1.0 / Ge.sqrt_n), 0.18)


class TestTheCleanGeometryIsTheFrozenGeometry(unittest.TestCase):
    """`analysis.structure.Torus` was written independently of `geometry.Geometry`.
    They must describe one object -- that is the strongest evidence that the clean
    layer's geometry is the campaign's geometry and not a lookalike.

    "One object" means bit-for-bit where the two compute the same thing by the same
    route, and algebraically-equivalent-within-an-ULP where they do not.  Measured
    agreement (production cell, 6x6):

    * ``A_WC``, ``A1``, ``A2``, ``L1``, ``L2``, ``sc``, ``c2sc`` -- bit-identical;
    * ``area`` -- 5.68e-14 apart (2 ULP), because `Torus` stores the ``72*pi`` it was
      handed while `Geometry` computes ``|L1 x L2|``;
    * ``G1``, ``G2`` -- 1.11e-16 (1 ULP); ``g1``, ``g2`` -- 8.88e-16; ``sqrt_n`` -- 1
      ULP.  These are reciprocal/density quantities, i.e. all of them divide by that
      area, so the area's 2 ULP is where the difference enters.

    The physical content is identical (the relative disagreement is <= 3e-16, twenty
    orders of magnitude below any quantity either workflow measures), and the two
    conventions are pinned separately: the clean one against `geometry_setup` in
    `test_geometry_against_legacy.py`, and `Torus` against the frozen figures it
    drew.  Neither is rewritten to match the other.
    """

    ULP_RTOL = 1e-15

    @classmethod
    def setUpClass(cls):
        cls.T = st.Torus(AREA_PROD, n_electrons=36, n_cells_per_side=6)
        cls.G = hm.production_geometry()

    def test_scalar_constants(self):
        # A_WC is bit-exact; the other two go through that area
        self.assertEqual(self.T.A_WC, self.G.a_wc)
        np.testing.assert_allclose(self.T.area, self.G.area,
                                   rtol=self.ULP_RTOL, atol=0)
        np.testing.assert_allclose(self.T.sqrt_n, self.G.sqrt_n,
                                   rtol=self.ULP_RTOL, atol=0)

    def test_direct_space_vectors_are_bit_identical(self):
        # the four vectors that are not divided by the area
        for name in ("A1", "A2", "L1", "L2"):
            np.testing.assert_allclose(getattr(self.T, name), getattr(self.G, name),
                                       rtol=0, atol=0, err_msg=name)

    def test_reciprocal_vectors_agree_to_one_ulp(self):
        # every one of these divides by the area, so they inherit its 2 ULP
        for name in ("G1", "G2", "g1", "g2"):
            np.testing.assert_allclose(getattr(self.T, name), getattr(self.G, name),
                                       rtol=self.ULP_RTOL, atol=0, err_msg=name)

    def test_cell_matrices(self):
        np.testing.assert_allclose(self.T.sc, self.G.sc_to_cart, rtol=0, atol=0)
        np.testing.assert_allclose(self.T.c2sc, self.G.cart_to_sc, rtol=0, atol=0)

    def test_against_the_legacy_engine(self):
        tup = legacy.geometry_setup(self.G.A1, self.G.A2, 6, 6, 4.0, True)
        area, L1, L2, G1, G2, mesh, g1, g2, RL = tup
        self.assertAlmostEqual(self.G.area, area, places=13)
        np.testing.assert_allclose(self.G.L1, L1, rtol=0, atol=1e-15)
        np.testing.assert_allclose(self.G.G1, G1, rtol=0, atol=1e-15)
        np.testing.assert_allclose(self.G.g1, g1, rtol=0, atol=1e-15)
        np.testing.assert_allclose(self.G.mesh, mesh, rtol=0, atol=0)

    def test_the_two_momentum_sets_are_different_objects(self):
        # 36 BZ momenta in row order vs 294 structure-factor momenta in lexsort order.
        # Recorded because aligning one against the other by index is the mistake the
        # structure-factor figures are always one step away from.
        #
        # `Torus.allowed_momenta` returns a two-tuple `(q, |q|)`, not a bare array: the
        # norm comes back with the vector and is already sorted ascending, which is the
        # order the structure-factor loop reads it in.
        q, qn = self.T.allowed_momenta(q_max=4.0)
        self.assertEqual(self.G.mesh.shape, (36, 2))
        self.assertEqual(q.shape, (294, 2))
        self.assertEqual(qn.shape, (294,))
        np.testing.assert_allclose(qn, np.linalg.norm(q, axis=1), rtol=0, atol=1e-12)
        self.assertTrue(bool(np.all(np.diff(qn) >= -1e-12)))
        self.assertGreater(abs(q.shape[0] - self.G.mesh.shape[0]), 200)
        # and the BZ mesh's own norms are NOT sorted -- it is in row order
        mesh_norm = np.linalg.norm(self.G.mesh, axis=1)
        self.assertFalse(bool(np.all(np.diff(mesh_norm) >= -1e-12)))


class TestTheNmaxRule(unittest.TestCase):
    """§2A.5: the physics core takes ``nmax`` only; ``n_bands`` is derived."""

    def test_nmax_maps_to_the_levels_it_names(self):
        # the rule, stated as the thing it means: which Landau levels are kept
        self.assertEqual(ll.nmax_from_n_bands(1), 0)   # nmax = 0 -> n = 0
        self.assertEqual(ll.nmax_from_n_bands(2), 1)   # nmax = 1 -> n = 0, 1
        self.assertEqual(ll.nmax_from_n_bands(3), 2)   # nmax = 2 -> n = 0, 1, 2

    def test_the_constructor_does_not_accept_a_band_count(self):
        # a signature check, so that a later refactor cannot reintroduce the second
        # spelling without failing here
        params = set(inspect.signature(ll.LandauLevelBasis.__init__).parameters)
        self.assertIn("nmax", params)
        self.assertNotIn("nb", params)
        self.assertNotIn("n_bands", params)
        self.assertNotIn("n_max", params)

    def test_n_bands_is_derived_and_read_only(self):
        p = hm.ModelParams.from_n_bands(2, kappa=32.0)
        self.assertEqual(p.nmax, 1)
        self.assertEqual(p.n_bands, 2)
        with self.assertRaises(AttributeError):
            p.n_bands = 3

    def test_the_basis_returns_nmax_plus_one_levels(self):
        Ge = hm.production_geometry()
        ai, ac = g.circular_lattice(8.5 * Ge.a_wc, Ge.A1, Ge.A2)
        C = np.column_stack([Ge.A1, Ge.A2])
        Ci = np.linalg.inv(C)
        for nmax in (0, 1, 2, 3):
            b = ll.LandauLevelBasis(Ge.mesh, nmax, ai, ac, C, Ci)
            self.assertEqual(b.n_bands, nmax + 1)
            self.assertEqual(b.orbitals(np.array([0.3, -0.7])).shape, (36, nmax + 1))
            self.assertEqual(b.padded_orbitals(np.array([0.3, -0.7])).shape,
                             (36, nmax + 2))

    def test_the_analysis_layers_rung_ladder_uses_the_same_rule(self):
        # RUNG is keyed by (tier, n_band); the physics core must agree that
        # n_band = nmax + 1, or the ladder names two different truncations
        ladder = [(tier, ll.nmax_from_n_bands(nb)) for tier, nb in ec.RUNG]
        self.assertEqual(ladder, [("conv", 1), ("nest", 2), ("nest4", 3), ("nmax", 2)])

    def test_model_params_from_n_bands_is_the_legacy_door(self):
        p = hm.ModelParams.from_n_bands(3, kappa=32.0)
        self.assertEqual(p.nmax, 2)
        self.assertEqual(p.to_legacy_n_bands(), 3)
        # and the legacy basis built with that count is the one the clean nmax names
        self.assertEqual(p.n_bands, 3)
        with self.assertRaises(ValueError):
            hm.ModelParams.from_n_bands(0, kappa=32.0)


class TestModelParams(unittest.TestCase):
    def test_from_rs(self):
        p = hm.ModelParams.from_rs(45.0)
        self.assertAlmostEqual(p.kappa, 31.819805153394640, places=12)
        self.assertAlmostEqual(p.r_s, 45.0, places=12)
        self.assertEqual(p.n_electrons, 36)
        self.assertEqual(p.n_flux_quanta, 36.0)
        self.assertAlmostEqual(p.area, 72 * math.pi, places=9)
        self.assertEqual(p.n_bands, 1)          # default nmax = 0

    def test_rejects_bad_values(self):
        for kw in ({"kappa": 32.0, "nu": 0.0}, {"kappa": 32.0, "nu": -1.0},
                   {"kappa": 32.0, "nmax": -1}, {"kappa": 32.0, "n_electrons": 0}):
            with self.assertRaises(ValueError):
                hm.ModelParams(**kw)

    def test_is_hashable_and_frozen(self):
        p = hm.ModelParams.from_rs(45.0)
        with self.assertRaises(AttributeError):
            p.kappa = 1.0
        self.assertEqual(len({p, hm.ModelParams.from_rs(45.0)}), 1)

    def test_the_benchmark_couplings_are_inside_the_production_range(self):
        lo, hi = hm.PRODUCTION_KAPPAS[0], hm.PRODUCTION_KAPPAS[-1]
        for rs in (45.0, 65.0, 75.0):
            k = hm.kappa_from_rs(rs)
            self.assertGreater(k, lo)
            self.assertLess(k, hi)


class TestTheMapIsPresent(unittest.TestCase):
    """A light check that the Stage 2A inventory exists and still states the two
    conventions that are easiest to lose."""

    def test_map_exists(self):
        self.assertTrue(os.path.exists(MAP), f"missing {MAP}")

    def test_map_locks_the_relation_and_the_nmax_rule(self):
        with open(MAP, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("r_s = √2 κ", text)
        self.assertIn("nmax", text)
        self.assertIn("n_bands = nmax + 1", text)
        # the legacy trap it was written to record
        self.assertIn("band count", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
