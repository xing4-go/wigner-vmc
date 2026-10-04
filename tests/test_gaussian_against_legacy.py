"""B2: `wavefunctions/gaussian.py` against the frozen engine.

The Gaussian orbital is the one ansatz the LL-rotation workflow does not use, so
this file is the only place its numbers are anchored.  It is compared

* bit-for-bit against the legacy `GaussianBasis` on fixed configurations, and
* against its own defining properties, which no shared bug can satisfy:

    - `pi psi` really is ``-i grad psi - A psi`` in the stated gauge;
    - `pi^2 psi / psi -> 1` at ``L0 = l_B``, where every Gaussian Bloch orbital is
      an exact lowest-Landau-level state;
    - `dpsi_dL0` is the derivative of `orbitals` with respect to ``L0``.

The third of those is the external anchor.  The LLL value is exact and known in
closed form, so a sign error in the gauge or a mis-assembled ladder action fails
here even if it were copied faithfully into both implementations.
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
from wigner_vmc.wavefunctions import gaussian as gb              # noqa: E402

PRIM_AREA = 2.0 * np.pi
NE = 36
LUMAX = 30.0
RTOL = 1e-13
ATOL = 1e-13

# fixed widths, including the LLL value L0 = l_B = 1 and a deliberately bad one
L0S = (0.35, 0.6, 1.0, 1.4, 2.2)
POINTS = (np.array([0.31, -0.77]), np.array([1.10, 0.40]), np.array([-2.00, 1.30]),
          np.array([0.0, 0.0]), np.array([1.9, -3.4]))


def setup():
    A1, A2 = g.triangular_cell(PRIM_AREA)
    Ge = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
    sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
    l_ints, l_cart = g.circular_lattice(LUMAX, Ge.L1, Ge.L2)
    new = gb.GaussianBasis(sites, l_ints, l_cart, Ge.L1, Ge.L2, l=1.0)
    old = legacy.GaussianBasis(sites, l_ints, l_cart, Ge.L1, Ge.L2, l=1.0)
    return new, old, Ge, sites


class TestAgainstLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.new, cls.old, cls.Ge, cls.sites = setup()

    def test_lattice_tables_and_flux_match(self):
        self.assertEqual(self.new.ns, self.old.ns)
        self.assertEqual(self.new.nL, self.old.nL)
        np.testing.assert_allclose(self.new.sites, self.old.sites, rtol=0, atol=0)
        np.testing.assert_allclose(self.new.l_cart, self.old.l_cart, rtol=0, atol=0)
        self.assertEqual(self.new.n_phi, float(self.old.n_phi))
        self.assertAlmostEqual(self.new.n_phi, 36.0, places=12)

    def test_orbitals(self):
        for L0 in L0S:
            for r in POINTS:
                np.testing.assert_allclose(self.new.orbitals(r, L0),
                                           self.old.orbitals(r, L0),
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f"L0={L0} r={r}")

    def test_pi_and_pi_square(self):
        for L0 in L0S:
            for r in POINTS:
                np.testing.assert_allclose(self.new.pi_orbitals(r, L0),
                                           self.old.pi_orbitals(r, L0),
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f"pi L0={L0} r={r}")
                np.testing.assert_allclose(self.new.pi_square_orbitals(r, L0),
                                           self.old.pi_square_orbitals(r, L0),
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f"pi2 L0={L0} r={r}")

    def test_dpsi_dL0(self):
        for L0 in L0S:
            for r in POINTS:
                np.testing.assert_allclose(self.new.dpsi_dL0(r, L0),
                                           self.old.dpsi_dL0(r, L0),
                                           rtol=RTOL, atol=ATOL,
                                           err_msg=f"L0={L0} r={r}")

    def test_shapes(self):
        r = POINTS[0]
        self.assertEqual(self.new.orbitals(r, 1.0).shape, (self.new.ns,))
        self.assertEqual(self.new.pi_orbitals(r, 1.0).shape, (2, self.new.ns))
        self.assertEqual(self.new.pi_square_orbitals(r, 1.0).shape, (self.new.ns,))
        self.assertEqual(self.new.dpsi_dL0(r, 1.0).shape, (self.new.ns,))


class TestItsOwnPhysics(unittest.TestCase):
    """Checks that never mention legacy."""

    @classmethod
    def setUpClass(cls):
        cls.B, cls.old, cls.Ge, cls.sites = setup()

    def test_pi_psi_is_the_covariant_derivative(self):
        # pi_x = -i d_x - A_x, pi_y = -i d_y - A_y with A = (1/2)(-y, x).
        # Pins the sign of B: the opposite gauge gives the same pi^2 (hence the
        # same kinetic energy) and the opposite pi psi.
        h = 1e-5
        L0 = 1.0
        for r in POINTS:
            p = self.B.pi_orbitals(r, L0)
            dx = (self.B.orbitals(r + [h, 0], L0) - self.B.orbitals(r - [h, 0], L0)) / (2 * h)
            dy = (self.B.orbitals(r + [0, h], L0) - self.B.orbitals(r - [0, h], L0)) / (2 * h)
            PX = -1j * dx + (r[1] / 2.0) * self.B.orbitals(r, L0)
            PY = -1j * dy - (r[0] / 2.0) * self.B.orbitals(r, L0)
            self.assertLess(float(np.abs(PX - p[0]).max()), 5e-8, f"pi_x r={r}")
            self.assertLess(float(np.abs(PY - p[1]).max()), 5e-8, f"pi_y r={r}")

    def test_pi_square_is_the_square_of_pi(self):
        # pi^2 = pi_x^2 + pi_y^2, checked by applying pi_x / pi_y a second time to
        # the ALREADY analytic pi_x psi / pi_y psi:
        #     pi_x^2 psi = (-i d_x + y/2)(pi_x psi),  A_x = -y/2
        #     pi_y^2 psi = (-i d_y - x/2)(pi_y psi),  A_y = +x/2
        # Taking the derivative of an analytic quantity instead of differencing
        # psi twice keeps the finite-difference error at O(h^2) on a first
        # derivative, so the tolerance can be tight enough to catch a dropped or
        # double-counted term -- which is what the hand-assembled
        # `2 pi^dag pi + 1/l^2` form in the source is at risk of.
        h = 1e-5
        L0 = 1.4
        for r in POINTS:
            def ux(q):
                return self.B.pi_orbitals(q, L0)[0]

            def uy(q):
                return self.B.pi_orbitals(q, L0)[1]

            dx_ux = (ux(r + [h, 0]) - ux(r - [h, 0])) / (2 * h)
            dy_uy = (uy(r + [0, h]) - uy(r - [0, h])) / (2 * h)
            pix2 = -1j * dx_ux + (r[1] / 2.0) * ux(r)
            piy2 = -1j * dy_uy - (r[0] / 2.0) * uy(r)
            got = self.B.pi_square_orbitals(r, L0)
            scale = max(float(np.abs(got).max()), 1e-30)
            self.assertLess(float(np.abs(got - (pix2 + piy2)).max()) / scale, 1e-8,
                            f"pi2 r={r}")

    def test_every_orbital_is_an_exact_lll_state_at_L0_equals_lB(self):
        # At L0 = l_B each Gaussian Bloch orbital IS a lowest-Landau-level state,
        # so pi^2 psi / psi = 1 exactly.  An external anchor: nobody's port can
        # make this true by being faithful to a bug.
        for r in POINTS:
            psi = self.B.orbitals(r, 1.0)
            pi2 = self.B.pi_square_orbitals(r, 1.0)
            ratio = pi2 / psi
            self.assertLess(float(np.abs(ratio - 1.0).max()), 1e-10,
                            f"pi2/psi at L0=l_B, r={r}")

    def test_the_lll_property_fails_away_from_lB(self):
        # Guard against the anchor above going vacuous.
        r = POINTS[0]
        ratio = self.B.pi_square_orbitals(r, 1.4) / self.B.orbitals(r, 1.4)
        self.assertGreater(float(np.abs(ratio - 1.0).max()), 1e-3)

    def test_dpsi_dL0_is_the_derivative_of_the_orbital(self):
        h = 1e-6
        for L0 in (0.6, 1.0, 1.4):
            for r in POINTS:
                num = (self.B.orbitals(r, L0 + h) - self.B.orbitals(r, L0 - h)) / (2 * h)
                got = self.B.dpsi_dL0(r, L0)
                self.assertLess(float(np.abs(num - got).max()), 1e-8,
                                f"d/dLO at L0={L0} r={r}")

    def test_the_orbital_is_periodic_when_the_bloch_sum_is_not_truncated(self):
        # a probability amplitude on the torus: |psi(r + L)| = |psi(r)| for every
        # supercell vector L.  This is what the Bloch sum is for.
        #
        # It is exact only once the sum is not cut off.  At the PRODUCTION cutoff
        # LUMAX = 30 there are just 13 lattice vectors, and shifting r by a
        # supercell vector needs images the cutoff excluded -- see the companion
        # test below, and the module docstring.  Here the cutoff is widened until
        # the shifts are inside it, which is the regime where the statement is a
        # statement about the ansatz rather than about the cutoff.
        L0 = 1.0
        r = POINTS[1]
        li, lc = g.circular_lattice(60.0, self.Ge.L1, self.Ge.L2)
        B = gb.GaussianBasis(self.sites, li, lc, self.Ge.L1, self.Ge.L2, 1.0)
        self.assertGreater(len(lc), 40)
        for L in (self.Ge.L1, self.Ge.L2, -2 * self.Ge.L1 + 3 * self.Ge.L2):
            np.testing.assert_allclose(np.abs(B.orbitals(r + L, L0)),
                                       np.abs(B.orbitals(r, L0)),
                                       rtol=0, atol=1e-13, err_msg=f"L={L}")

    def test_at_the_production_cutoff_periodicity_holds_only_to_the_tail(self):
        # The honest form of the statement above at the cutoff actually used.
        # A shift by L1 needs lattice vectors out to ~2*|L1|; with LUMAX = 30 the
        # sum stops well short, so |psi(r + L1)| differs from |psi(r)| by the
        # amplitude of the terms that were dropped -- 1.96e-6 here, on orbitals
        # whose own scale is 1e-8..3e-1.
        #
        # Recorded as a property of the ANSATZ, and checked to be the frozen
        # engine's number rather than this port's: both give the identical
        # deviation, at every cutoff, so nothing was lost in the migration.  What
        # this test stops is somebody later reading the widened-cutoff test above
        # as "the production orbitals are periodic" and building on it.
        L0 = 1.0
        r = POINTS[1]
        d_new = float(np.abs(np.abs(self.B.orbitals(r + self.Ge.L1, L0))
                             - np.abs(self.B.orbitals(r, L0))).max())
        d_old = float(np.abs(np.abs(self.old.orbitals(r + self.Ge.L1, L0))
                             - np.abs(self.old.orbitals(r, L0))).max())
        self.assertEqual(d_new, d_old)
        self.assertGreater(d_new, 1e-9)       # it is NOT periodic at this cutoff
        self.assertLess(d_new, 1e-4)          # but the tail is small

    def test_widening_the_cutoff_removes_the_tail(self):
        # the tail test above is about the cutoff and not about a bug: raising
        # LUMAX drives the same deviation to machine precision
        L0 = 1.0
        r = POINTS[1]
        d = []
        for LU in (30.0, 45.0, 60.0):
            li, lc = g.circular_lattice(LU, self.Ge.L1, self.Ge.L2)
            B = gb.GaussianBasis(self.sites, li, lc, self.Ge.L1, self.Ge.L2, 1.0)
            d.append(float(np.abs(np.abs(B.orbitals(r + self.Ge.L1, L0))
                                  - np.abs(B.orbitals(r, L0))).max()))
        self.assertGreater(d[0], d[1])
        self.assertLess(d[2], 1e-13)

    def test_the_truncation_reaches_the_production_determinant_only_at_1e10(self):
        # The test above measures the tail on ONE orbital, relative to nothing --
        # it is an absolute deviation, and picking an orbital that happens to be
        # tiny at the probe point inflates the ratio between it and |psi| into
        # anything you like.  The generator's own diagnostic does exactly that
        # and prints 1.0e+00 at make_notebook.py:403.
        #
        # What production actually consumes is the determinant, built at 36
        # electrons INSIDE the supercell (the sampler folds onto the torus, so
        # that is the only regime that occurs).  That is the number that decides
        # whether the truncation is worth caring about, so it is the one pinned.
        rng = np.random.default_rng(7)
        sc = np.column_stack([self.Ge.L1, self.Ge.L2])
        R = rng.uniform(0.05, 0.95, size=(NE, 2)) @ sc.T
        L0 = 0.6

        def det_at(cut):
            li, lc = g.circular_lattice(cut, self.Ge.L1, self.Ge.L2)
            B = gb.GaussianBasis(self.sites, li, lc, self.Ge.L1, self.Ge.L2, 1.0)
            D = np.column_stack([B.orbitals(R[j], L0) for j in range(NE)])
            return D

        D30 = det_at(30.0)
        D90 = det_at(90.0)                      # converged: 121 vectors
        amp = float(np.abs(np.abs(D30) - np.abs(D90)).max() / np.abs(D90).max())
        dlog = abs(float(np.log(np.abs(np.linalg.det(D30)))
                         - np.log(np.abs(np.linalg.det(D90)))))
        self.assertLess(amp, 1e-8, "truncation must not reach the production amplitude")
        self.assertLess(dlog, 1e-9, "truncation must not reach the production determinant")
        # and the same cutoff in the frozen engine lands on the same determinant,
        # so this is the ansatz's truncation and not this port's
        li, lc = g.circular_lattice(30.0, self.Ge.L1, self.Ge.L2)
        Bold = legacy.GaussianBasis(self.sites, li, lc, self.Ge.L1, self.Ge.L2, 1.0)
        Dold = np.column_stack([Bold.orbitals(R[j], L0) for j in range(NE)])
        self.assertEqual(float(np.abs(Dold).max()), float(np.abs(D30).max()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
