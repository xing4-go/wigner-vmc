"""B1: `wavefunctions/jastrow.py` against the frozen engine.

Fixed configuration, fixed parameters, no Monte Carlo.  What is compared:

    J(r) = u(r),  dJ,  lap J   -- pointwise, on a fixed set of separations
    grad_params                -- the parameter derivative the SR loop consumes
    u_pairs(R)                 -- the pair matrices the sampler and the kinetic
                                  estimator actually read

The Jastrow is the one object whose *value* is shared by both workflows but whose
*arguments* are not: the Gaussian workflow passes the supercell reciprocal
vectors and the LL-rotation workflow the primitive-cell ones.  Both are tested,
so that "same function" is not confused with "same numbers".
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
from wigner_vmc.wavefunctions import jastrow as jw               # noqa: E402

PRIM_AREA = 2.0 * np.pi
NE = 36
RTOL = 1e-13
ATOL = 1e-14

# fixed separation vectors: generic points, the origin (where f = 0 and the
# 1/f terms in grad_f / laplacian_f are the interesting branch), and a point on
# a reciprocal-lattice direction where the spline argument is an integer.
SEPS = np.array([[0.0, 0.0], [0.37, -1.21], [1.0, 0.5], [-2.3, 1.7],
                 [0.0, 1.9], [3.3, -0.4], [0.5, 0.5]])

# two fixed production-like Jastrow parameter vectors
C5 = np.array([0.30, -0.55, 0.42, -0.18, 0.07])
C2 = np.array([0.11, -0.23])


def geom():
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, 4.0)


def configs(Ge, seed=20261001):
    rng = np.random.default_rng(seed)
    return rng.uniform(-3.0, 3.0, size=(NE, 2))


class TestAgainstLegacy(unittest.TestCase):
    """Both argument conventions, since the two workflows use different ones."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        # the production argument set: SUPERCELL reciprocal vectors, with the
        # Kato gamma from |L1|.  Both workflows use this one -- including
        # LL-rotation, whose name invites the wrong guess (make_notebook_llrot.py
        # :541 takes G1, G2 from geometry_setup, i.e. the supercell pair).
        cls.GS = (cls.Ge.G1, cls.Ge.G2)
        # a second, out-of-production argument set, to check the function is
        # generic in its arguments rather than tuned to one call site
        cls.GP = (cls.Ge.g1 / 6.0, cls.Ge.g2 / 6.0)
        cls.gamma_s = float(np.linalg.norm(cls.Ge.L1) / np.sqrt(2) / np.pi * 32.0 / 3)
        cls.gamma_p = float(np.linalg.norm(cls.Ge.A1) / np.sqrt(2) / np.pi * 32.0 / 3)
        cls.new_s = jw.SinSplineJastrow(C5, *cls.GS, cls.gamma_s)
        cls.old_s = legacy.SinSplineJastrow(C5, *cls.GS, cls.gamma_s)
        cls.new_p = jw.SinSplineJastrow(C5, *cls.GP, cls.gamma_p)
        cls.old_p = legacy.SinSplineJastrow(C5, *cls.GP, cls.gamma_p)

    def test_spline_functions_match(self):
        x = np.linspace(-2.5, 1.5, 401)
        np.testing.assert_allclose(jw.b3(x), legacy.b3(x), rtol=0, atol=0)
        np.testing.assert_allclose(jw.dx_b3(x), legacy.dx_b3(x), rtol=0, atol=0)
        np.testing.assert_allclose(jw.d2x_b3(x), legacy.d2x_b3(x), rtol=0, atol=0)

    def test_the_parameter_list_is_the_same_object(self):
        for new, old in ((self.new_s, self.old_s), (self.new_p, self.old_p)):
            np.testing.assert_allclose(new.pm, old.pm, rtol=0, atol=0)
            self.assertEqual(new.M, old.M)
            self.assertEqual(new.gamma, float(old.gamma))

    def test_value_gradient_laplacian(self):
        # SEPS[0] is the origin, where grad and lap are nan in BOTH
        # implementations; that point is covered by its own test below, and it
        # is excluded here so this comparison cannot be satisfied by nan == nan
        # (which assert_allclose accepts as agreement).
        off_origin = SEPS[1:]
        for new, old, tag in ((self.new_s, self.old_s, "supercell"),
                              (self.new_p, self.old_p, "primitive")):
            np.testing.assert_allclose(new.u(SEPS), old.u(SEPS), rtol=RTOL,
                                       atol=ATOL, err_msg=f"u {tag}")
            for f in ("u", "grad", "laplacian"):
                a = getattr(new, f)(off_origin)
                b = getattr(old, f)(off_origin)
                self.assertTrue(np.all(np.isfinite(a)), f"{f} is not finite ({tag})")
                np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL,
                                           err_msg=f"{f} {tag}")

    def test_the_origin_is_a_singular_point_of_grad_and_lap_only(self):
        """u(0) is finite; grad(0) and lap(0) are nan, in both implementations.

        This is recorded, not asserted away.  It is reachable only by calling
        grad/laplacian at exactly r = 0: `u_pairs` computes them on the diagonal
        and then masks it to zero, and `accept_move` only ever asks for
        r_new - R_j with r_new a fresh proposal, so no production path reaches
        the singularity.  A test that merely compared the arrays would report
        "agreement" here, because nan == nan is accepted -- which is how a nan
        could sit in this file unnoticed.
        """
        z = np.array([0.0, 0.0])
        for new, old, tag in ((self.new_s, self.old_s, "supercell"),
                              (self.new_p, self.old_p, "primitive")):
            self.assertTrue(np.isfinite(new.u(z)), f"u(0) must be finite ({tag})")
            self.assertEqual(float(new.u(z)), float(old.u(z)))
            for f in ("grad", "laplacian"):
                self.assertTrue(np.all(np.isnan(getattr(new, f)(z))), f"{f}(0) {tag}")
                self.assertTrue(np.all(np.isnan(getattr(old, f)(z))), f"{f}(0) {tag}")
        # approaching the origin the values are finite on both sides and agree
        # exactly, so the nan is a 0/0 at one point and not a wrong branch
        near = np.array([1e-9, 0.0])
        np.testing.assert_array_equal(self.new_s.grad(near), self.old_s.grad(near))
        np.testing.assert_array_equal(self.new_s.laplacian(near),
                                      self.old_s.laplacian(near))
        self.assertGreater(float(self.new_s.laplacian(near)), 1e9)

    def test_the_scalar_and_batch_paths_agree(self):
        # u() takes (2,) or (...,2); the single-point branch must be the same
        # number as the batch branch, or the sampler and the pair matrices
        # disagree about the same separation
        for new in (self.new_s, self.new_p):
            np.testing.assert_allclose(new.u(SEPS[1]), new.u(SEPS)[1],
                                       rtol=0, atol=0)
            np.testing.assert_allclose(new.grad(SEPS[1]), new.grad(SEPS)[1],
                                       rtol=0, atol=0)
            np.testing.assert_allclose(new.laplacian(SEPS[1]), new.laplacian(SEPS)[1],
                                       rtol=0, atol=0)

    def test_grad_params(self):
        # the production parameter derivative: this is what SR consumes
        for new, old in ((self.new_s, self.old_s), (self.new_p, self.old_p)):
            np.testing.assert_allclose(new.grad_params(SEPS), old.grad_params(SEPS),
                                       rtol=RTOL, atol=ATOL)

    def test_u_pairs(self):
        R = configs(self.Ge)
        for new, old, tag in ((self.new_s, self.old_s, "supercell"),
                              (self.new_p, self.old_p, "primitive")):
            mine = new.u_pairs(R)
            theirs = old.u_pairs(R)
            self.assertEqual(len(mine), len(theirs))
            for a, b in zip(mine, theirs):
                np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=tag)


class TestTheJastrowsOwnProperties(unittest.TestCase):
    """Checks that never mention legacy.  A port reproduces a bug in both copies."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.J = jw.SinSplineJastrow(C5, cls.Ge.G1, cls.Ge.G2, 0.0)

    def test_u_pairs_is_symmetric_with_a_zero_diagonal(self):
        R = configs(self.Ge)
        u, gx, gy, lap = self.J.u_pairs(R)
        np.testing.assert_allclose(u, u.T, rtol=0, atol=1e-14)
        # the gradient is ODD and the laplacian EVEN under r -> -r, so the pair
        # matrices are antisymmetric and symmetric respectively -- this is the
        # property `accept_move` relies on when it fills the transposed entries
        np.testing.assert_allclose(gx, -gx.T, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gy, -gy.T, rtol=0, atol=1e-12)
        np.testing.assert_allclose(lap, lap.T, rtol=0, atol=1e-12)
        for m in (u, gx, gy, lap):
            np.testing.assert_allclose(np.diag(m), 0.0, rtol=0, atol=0)

    def test_the_gradient_is_the_gradient_of_u(self):
        # central differences on u itself, the only external check that the
        # analytic gradient is the derivative of the function it claims to be
        h = 1e-5
        for r in (np.array([0.37, -1.21]), np.array([1.0, 0.5])):
            num = np.array([(self.J.u(r + [h, 0]) - self.J.u(r - [h, 0])) / (2 * h),
                            (self.J.u(r + [0, h]) - self.J.u(r - [0, h])) / (2 * h)])
            self.assertLess(float(np.abs(num - self.J.grad(r)).max()), 1e-7,
                            f"gradient at r={r}")

    def test_the_laplacian_is_the_divergence_of_the_gradient(self):
        h = 1e-5
        r = np.array([0.37, -1.21])
        num = ((self.J.grad(r + [h, 0])[0] - self.J.grad(r - [h, 0])[0])
               + (self.J.grad(r + [0, h])[1] - self.J.grad(r - [0, h])[1])) / (2 * h)
        self.assertLess(abs(float(num) - float(self.J.laplacian(r))), 1e-6)

    def test_the_cusp_constraint_is_what_sets_pm0(self):
        # pm[0] = c[1] - 3 gamma/M.  With gamma = 0 the extra term vanishes and
        # pm[0] is c[1]; the whole point of the term is that gamma carries the
        # Kato cusp, so a Jastrow with the wrong gamma has the wrong short-range
        # slope and a slightly wrong energy -- not a visibly wrong wavefunction.
        J0 = jw.SinSplineJastrow(C5, self.Ge.G1, self.Ge.G2, 0.0)
        self.assertAlmostEqual(J0.pm[0], C5[1], places=15)
        Jg = jw.SinSplineJastrow(C5, self.Ge.G1, self.Ge.G2, 0.25)
        self.assertNotAlmostEqual(Jg.pm[0], C5[1], places=6)
        # and the shift is exactly -3 gamma/M
        self.assertAlmostEqual(Jg.pm[0] - C5[1], -3 * 0.25 / len(C5), places=15)


class TestCuspGamma(unittest.TestCase):
    def test_matches_the_generator_formula(self):
        Ge = geom()
        for kappa in (2.0, 32.0, 80.0):
            expected = np.linalg.norm(Ge.L1) / np.sqrt(2) / np.pi * kappa / 3
            self.assertAlmostEqual(jw.cusp_gamma(kappa, Ge.L1), float(expected),
                                   places=13)

    def test_is_linear_in_kappa(self):
        Ge = geom()
        a = jw.cusp_gamma(1.0, Ge.L1)
        self.assertAlmostEqual(jw.cusp_gamma(48.0, Ge.L1), 48.0 * a, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
