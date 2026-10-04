"""B5: local energy and kinetic term against the frozen engine.

Both ansaetze, on fixed configurations, no Monte Carlo.  `V`, `T` and
`E_loc = T + kappa*V` are compared **separately and reported separately**:
comparing only `E_loc` would let an error in `T` cancel an error in `V`, and the
whole reason this subsystem is the highest-risk one in Stage 2B is that the
kinetic term is assembled from three pieces (`T_det`, `T_mix`, `T_jastrow`)
whose signs are opposite.

What is compared:

    Gaussian : pi_columns(R), local_energy(st) -> (T, V, E_loc)
    LL       : pi_columns(st), local_energy(st) -> (T, V, E_loc)
    both     : calc_kinetic_energy on frozen inputs, and its three pieces

The two ansaetze are exercised through the SAME class hierarchy and the SAME
`calc_kinetic_energy`, but `pi_columns` takes a configuration in one and a state
dict in the other -- the legacy interface, preserved on purpose
(BUG_CANDIDATE.md #1).  The tests below call each with what its own class wants;
a test that forced one calling convention would be testing a redesign.

Tolerances are conditioning-aware where a determinant is inverted.  `D_inv` for
the LL ansatz at the test configuration has `cond(D) = 2.4e6`, so a 1-ULP
disagreement in `D` becomes ~1e-10 in `D_inv` and reaches `T` through
`einsum("li,il->i", pix_D, D_inv)`; that is measured and bounded, not assumed.
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
import qhvmc_engine_llrot as legacy_ll                           # noqa: E402
from wigner_vmc.physics import coulomb as cb                     # noqa: E402
from wigner_vmc.physics import geometry as g                     # noqa: E402
from wigner_vmc.physics import landau_levels as ll               # noqa: E402
from wigner_vmc.wavefunctions import gaussian as gauss           # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw               # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr           # noqa: E402
from wigner_vmc.wavefunctions import slater as sl                # noqa: E402

PRIM_AREA = 2.0 * np.pi
NE = 36
LUMAX = 30.0
LUMAX_LL = 8.5 * np.sqrt(4 * np.pi / np.sqrt(3))
KAPPA = 32.0
L0 = 0.6
C5 = np.array([0.30, -0.55, 0.42, -0.18, 0.07])
N_BANDS = 2

#: `V`, `T` and `E_loc` are O(100), O(10) and O(100) here; the energies are a
#: difference of large terms, so a rel tolerance on each of them has to be read
#: against the size of THAT quantity, not against E_loc.
RTOL = 1e-12
ATOL = 1e-12


def geom():
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, 4.0)


def configs(Ge, seed=20261001):
    """Configurations strictly inside the supercell: the only regime the sampler
    produces, because it folds every proposal onto the torus."""
    sc = np.column_stack([Ge.L1, Ge.L2])
    rng = np.random.default_rng(seed)
    return {tag: rng.uniform(0.05, 0.95, size=(NE, 2)) @ sc.T
            for tag in ("interior", "jittered")}


def jastrow_pair(Ge):
    gam = jw.cusp_gamma(KAPPA, Ge.L1)
    return (jw.SinSplineJastrow(C5, Ge.G1, Ge.G2, gam),
            legacy.SinSplineJastrow(C5, Ge.G1, Ge.G2, gam))


def coulomb_pair(Ge):
    return (cb.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2),
            legacy.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2))


def gaussian_pair(Ge):
    sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
    li, lc = g.circular_lattice(LUMAX, Ge.L1, Ge.L2)
    new_b = gauss.GaussianBasis(sites, li, lc, Ge.L1, Ge.L2, l=1.0)
    old_b = legacy.GaussianBasis(sites, li, lc, Ge.L1, Ge.L2, l=1.0)
    Jn, Jo = jastrow_pair(Ge)
    Hn, Ho = coulomb_pair(Ge)
    sc = np.column_stack([Ge.L1, Ge.L2])
    wn = sl.Wavefunction(lambda r: new_b.orbitals(r, L0),
                         lambda r: new_b.pi_orbitals(r, L0),
                         lambda r: new_b.pi_square_orbitals(r, L0),
                         Jn, NE, sc, kappa=KAPPA, ham=Hn)
    wo = legacy.Wavefunction(lambda r: old_b.orbitals(r, L0),
                             lambda r: old_b.pi_orbitals(r, L0),
                             lambda r: old_b.pi_square_orbitals(r, L0),
                             Jo, NE, sc, kappa=KAPPA, ham=Ho)
    return wn, wo


def ll_pair(Ge, n_band=N_BANDS, seed=7):
    ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
    C = np.column_stack([Ge.A1, Ge.A2])
    Ci = np.linalg.inv(C)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(len(Ge.mesh), n_band - 1)) \
        + 1j * rng.normal(size=(len(Ge.mesh), n_band - 1))
    Jn, Jo = jastrow_pair(Ge)
    Hn, Ho = coulomb_pair(Ge)
    sc = np.column_stack([Ge.L1, Ge.L2])
    new_b = ll.LandauLevelBasis(Ge.mesh, n_band - 1, ai, ac, C, Ci)
    old_b = legacy.LandauLevelBasis(Ge.mesh, n_band, ai, ac, C, Ci)
    wn = lr.LLRotationWavefunction(lr.LLRotatedOrbitals(new_b, n_band, v),
                                   Jn, NE, sc, kappa=KAPPA, ham=Hn)
    wo = legacy_ll.LLRotationWavefunction(
        legacy_ll.LLRotatedOrbitals(old_b, n_band, v), Jo, NE, sc,
        kappa=KAPPA, ham=Ho)
    return wn, wo, v


class TestCalcKineticEnergy(unittest.TestCase):
    """The free function, on frozen inputs -- no wavefunction involved.

    This is the sharpest test of the three pieces: `T_det`, `T_mix` and
    `T_jastrow` are checked separately, so a sign error in the Jastrow term
    cannot hide behind the determinant term.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4242)
        n = 36
        cls.D_inv = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        cls.pix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        cls.piy = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        cls.pi2 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        # the Jastrow pair matrices are real, antisymmetric (grad) / symmetric (lap)
        cls.gx = rng.normal(size=(n, n))
        cls.gy = rng.normal(size=(n, n))
        cls.lap = rng.normal(size=(n, n))

    def _both(self):
        args = (self.pix, self.piy, self.pi2, self.D_inv,
                self.gx, self.gy, self.lap)
        return (sl.calc_kinetic_energy(*args), legacy.calc_kinetic_energy(*args))

    def test_the_total_agrees(self):
        a, b = self._both()
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    @staticmethod
    def _pieces(pix, piy, pi2, D_inv, gx_m, gy_m, lap_m):
        """The decomposition, transcribed independently of the function under test."""
        D_inv = np.asarray(D_inv)
        px = np.einsum("li,il->i", pix, D_inv)
        py = np.einsum("li,il->i", piy, D_inv)
        q = np.einsum("li,il->i", pi2, D_inv)
        gx = np.sum(gx_m, axis=1)
        gy = np.sum(gy_m, axis=1)
        T_j = -0.5 * (np.sum(gx * gx) + np.sum(gy * gy) + np.sum(lap_m))
        return 0.5 * np.sum(q), -1j * (np.dot(gx, px) + np.dot(gy, py)), T_j

    def test_the_three_pieces_sum_to_the_total(self):
        # the decomposition is re-derived here and must reconstruct BOTH
        # implementations' total: a dropped or doubled piece cannot do that
        args = (self.pix, self.piy, self.pi2, self.D_inv,
                self.gx, self.gy, self.lap)
        d, m, j = self._pieces(*args)
        for f, nm in ((sl.calc_kinetic_energy, "clean"),
                      (legacy.calc_kinetic_energy, "legacy")):
            np.testing.assert_allclose(d + m + j, f(*args), rtol=RTOL, atol=ATOL,
                                       err_msg=f"{nm} total != its own pieces")

    def test_the_cross_term_carries_the_factor_of_i(self):
        # On REAL inputs T_det and T_jastrow are real and T_mix is pure
        # imaginary, so the total's imaginary part IS the cross term.  Dropping
        # the -1j (or conjugating the wrong factor) makes that part real and
        # this fails; nothing else in the suite would notice, because in
        # production both implementations would change in the same way.
        rng = np.random.default_rng(99)
        n = 36
        r = lambda: rng.normal(size=(n, n))                      # noqa: E731
        args = (r(), r(), r(), r(), r(), r(), r())
        T = sl.calc_kinetic_energy(*args)
        T_det, T_mix, T_jastrow = self._pieces(*args)
        self.assertAlmostEqual(float(T.real), float((T_det + T_jastrow).real),
                               places=9)
        self.assertAlmostEqual(float(T.imag), float(T_mix.imag), places=9)
        self.assertGreater(abs(float(T.imag)), 1e-6,
                           "the cross term must not vanish on these inputs")


class TestGaussianLocalEnergy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.new, cls.old = gaussian_pair(cls.Ge)
        cls.Rs = configs(cls.Ge)

    def _compare(self, R):
        sn, so = self.new.build(R), self.old.build(R)
        Tn, Vn, En = self.new.local_energy(sn)
        To, Vo, Eo = self.old.local_energy(so)
        return (Tn, Vn, En), (To, Vo, Eo)

    def test_pi_columns_match(self):
        for tag, R in self.Rs.items():
            pn = self.new.pi_columns(R)
            po = self.old.pi_columns(R)
            for a, b, nm in zip(pn, po, ("pix", "piy", "pi2")):
                np.testing.assert_allclose(a, b, rtol=0, atol=0,
                                           err_msg=f"{nm} {tag}")
            # the base class really does take a configuration, not a state
            self.assertEqual(pn[0].shape, (NE, NE))

    def test_T_V_and_E_loc_each_agree(self):
        for tag, R in self.Rs.items():
            (Tn, Vn, En), (To, Vo, Eo) = self._compare(R)
            np.testing.assert_allclose(Tn, To, rtol=RTOL, atol=ATOL,
                                       err_msg=f"T {tag}")
            np.testing.assert_allclose(Vn, Vo, rtol=RTOL, atol=ATOL,
                                       err_msg=f"V {tag}")
            np.testing.assert_allclose(En, Eo, rtol=RTOL, atol=ATOL,
                                       err_msg=f"E_loc {tag}")

    def test_the_coupling_is_applied_to_V_and_not_to_T(self):
        # E_loc = T + kappa*V, with V the BARE sum 1/r.  Swapping that -- folding
        # kappa into V, or into T -- leaves E_loc unchanged and is invisible to
        # a comparison of E_loc alone, which is exactly why V and T are separate.
        for tag, R in self.Rs.items():
            (Tn, Vn, En), _ = self._compare(R)
            np.testing.assert_allclose(En, Tn + KAPPA * Vn, rtol=RTOL, atol=ATOL,
                                       err_msg=f"E_loc = T + kappa V {tag}")
            self.assertNotAlmostEqual(float(np.real(Vn)), float(np.real(En)),
                                      places=3)

    def test_E_loc_is_not_the_sum_of_its_endpoints(self):
        # a guard against the trivial port that returns (T, V, T+V)
        (Tn, Vn, En), _ = self._compare(self.Rs["interior"])
        self.assertNotAlmostEqual(float(np.real(En)),
                                  float(np.real(Tn) + np.real(Vn)), places=3)

    def test_the_imaginary_part_matches_and_is_what_production_discards(self):
        """`T` is complex and its imaginary part is O(1), not round-off.

        The trial function is complex, so the local energy of a Hermitian
        operator has a nonzero imaginary part whose expectation vanishes; the
        legacy measurement path discards it explicitly with `.real`
        (`qhvmc_engine.py:1008`, `qhvmc_engine_llrot.py:505`).  Both
        implementations must agree on that imaginary part too -- a port that
        "cleaned up" by forcing T real would pass every other test here.
        """
        for tag, R in self.Rs.items():
            (Tn, _, En), (To, _, Eo) = self._compare(R)
            np.testing.assert_allclose(complex(Tn).imag, complex(To).imag,
                                       rtol=RTOL, atol=ATOL, err_msg=f"T {tag}")
            np.testing.assert_allclose(complex(En).imag, complex(Eo).imag,
                                       rtol=RTOL, atol=ATOL, err_msg=f"E {tag}")
            self.assertGreater(abs(complex(Tn).imag), 1e-3,
                               "this test is vacuous if the imaginary part is round-off")


class TestLLRotationLocalEnergy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.new, cls.old, cls.v = ll_pair(cls.Ge)
        cls.Rs = configs(cls.Ge)

    def test_pi_columns_matches_and_takes_a_state(self):
        """The two `pi_columns` bodies are character-identical (checked by AST),
        so any difference has to be inherited from the Bloch sum that feeds them.
        That is measured here and used as the bound, instead of a flat tolerance
        that would be either vacuous or an unexamined guess.

        Unlike the Gaussian case, bit identity is NOT available: the clean and
        legacy Landau-level bases sum their lattice vectors in a different order
        (~4e-16 relative on `Ops`, ~3e-16 on `_kin`), and that is the documented
        floating-point reassociation of B3 -- not a defect introduced in B5.
        """
        eps = float(np.finfo(float).eps)
        for tag, R in self.Rs.items():
            sn, so = self.new.build(R), self.old.build(R)
            pn = self.new.pi_columns(sn)
            po = self.old.pi_columns(so)

            self.assertEqual(pn[0].shape, (NE, NE))
            for a, b, nm in zip(pn, po, ("pix", "piy", "pi2")):
                scale = float(np.abs(b).max())
                d_ops = float(np.abs(sn["Ops"] - so["Ops"]).max()
                              / np.abs(so["Ops"]).max())
                self.assertLess(d_ops, 1e-15,
                                f"the Bloch sums must agree before this bound "
                                f"means anything ({tag}: {d_ops:.3e})")
                dev = float(np.abs(a - b).max())
                bound = 100.0 * (d_ops + eps) * scale
                self.assertLess(dev, bound,
                                f"{nm} deviation {dev:.3e} exceeds the bound "
                                f"{bound:.3e} implied by the basis ({tag})")
            # the override wants a dict; giving it an array must fail loudly
            # rather than silently returning something plausible
            with self.assertRaises((TypeError, IndexError)):
                self.new.pi_columns(R)

    def test_T_V_and_E_loc_each_agree(self):
        for tag, R in self.Rs.items():
            sn, so = self.new.build(R), self.old.build(R)
            Tn, Vn, En = self.new.local_energy(sn)
            To, Vo, Eo = self.old.local_energy(so)
            np.testing.assert_allclose(Tn, To, rtol=RTOL, atol=ATOL,
                                       err_msg=f"T {tag}")
            np.testing.assert_allclose(Vn, Vo, rtol=RTOL, atol=ATOL,
                                       err_msg=f"V {tag}")
            np.testing.assert_allclose(En, Eo, rtol=RTOL, atol=ATOL,
                                       err_msg=f"E_loc {tag}")

    def test_the_coupling_is_applied_to_V_and_not_to_T(self):
        for tag, R in self.Rs.items():
            sn = self.new.build(R)
            Tn, Vn, En = self.new.local_energy(sn)
            np.testing.assert_allclose(En, Tn + KAPPA * Vn, rtol=RTOL, atol=ATOL,
                                       err_msg=f"E_loc = T + kappa V {tag}")
            self.assertNotAlmostEqual(float(np.real(Vn)), float(np.real(En)),
                                      places=3)

    def test_the_imaginary_part_matches_and_is_what_production_discards(self):
        for tag, R in self.Rs.items():
            sn, so = self.new.build(R), self.old.build(R)
            Tn, _, En = self.new.local_energy(sn)
            To, _, Eo = self.old.local_energy(so)
            np.testing.assert_allclose(complex(Tn).imag, complex(To).imag,
                                       rtol=1e-11, atol=1e-11, err_msg=f"T {tag}")
            np.testing.assert_allclose(complex(En).imag, complex(Eo).imag,
                                       rtol=1e-11, atol=1e-11, err_msg=f"E {tag}")
            self.assertGreater(abs(complex(Tn).imag), 1e-3)

    def test_the_kinetic_agreement_is_bounded_by_the_conditioning(self):
        # `D_inv` is the only quantity in the T path that is not assembled the
        # same way on both sides; cond(D) is 2.4e6 here, so a 1-ULP difference in
        # D is amplified to ~1e-10 in D_inv.  Bound T's deviation by that,
        # rather than by a flat tolerance that would either be vacuous or wrong.
        for tag, R in self.Rs.items():
            sn, so = self.new.build(R), self.old.build(R)
            cond = float(np.linalg.cond(so["D"]))
            d_rel = float(np.abs(sn["D"] - so["D"]).max() / np.abs(so["D"]).max())
            Tn = self.new.local_energy(sn)[0]
            To = self.old.local_energy(so)[0]
            bound = 100.0 * cond * d_rel * max(abs(To), 1.0)
            self.assertLess(abs(Tn - To), bound,
                            f"T deviation exceeds the conditioning bound ({tag}): "
                            f"cond={cond:.3e} d_rel={d_rel:.3e} bound={bound:.3e}")
            self.assertLess(cond * d_rel, 1e-6,
                            "the two determinants must agree before this bound means anything")

    def test_the_jastrow_half_is_bit_identical(self):
        # T_jastrow is fed by the pair matrices; if those agree bit-for-bit then
        # the whole T difference is attributable to the determinant half
        for tag, R in self.Rs.items():
            sn, so = self.new.build(R), self.old.build(R)
            for k in ("u", "gx", "gy", "lap"):
                np.testing.assert_array_equal(sn[k], so[k], err_msg=f"{k} {tag}")


class TestTheTwoAnsaetzeDoNotShareAPath(unittest.TestCase):
    """The Gaussian and LL local energies are different numbers -- if they agreed,
    something would be returning a stale or shared buffer."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.gn, cls.go = gaussian_pair(cls.Ge)
        cls.ln, cls.lo, cls.v = ll_pair(cls.Ge)

    def test_the_energies_are_different(self):
        R = configs(self.Ge)["interior"]
        Tn_g, Vn_g, _ = self.gn.local_energy(self.gn.build(R))
        Tn_l, Vn_l, _ = self.ln.local_energy(self.ln.build(R))
        self.assertNotAlmostEqual(float(np.real(Tn_g)), float(np.real(Tn_l)),
                                  places=3)
        # V is a property of the configuration alone, so it MUST agree
        np.testing.assert_allclose(Vn_g, Vn_l, rtol=1e-12, atol=1e-12)

    def test_pi_columns_are_not_interchangeable(self):
        R = configs(self.Ge)["interior"]
        gauss_pix = self.gn.pi_columns(R)[0]
        ll_pix = self.ln.pi_columns(self.ln.build(R))[0]
        self.assertEqual(gauss_pix.shape, ll_pix.shape)
        self.assertGreater(float(np.abs(gauss_pix - ll_pix).max()), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
