"""The Gaussian -> LL projection convention, and the seed's C6 covariance.

Both classes guard ONE defect: `vmc/sr.py:219` had the complex conjugation on the
wrong operand, so the seed was projected with ``sum_i G* phi = <G|phi>`` instead of
``<phi|G> = sum_i phi* G``.  The code's own docstring (``sr.py:159``) and the
authors' Julia oracle (``slaterdet_gaussian.jl:123``,
``sum(conj(mag_bloch) .* mag_gaussian)``) both already specified the right one.

Why the existing suite could not see it
--------------------------------------
The Gaussian amplitude G is REAL (tested below), so the two conventions differ by
exactly one conjugation of the overlap array.  ``v_from_overlap`` phase-fixes and
``c_row`` is its exact inverse, so **every** modulus test, every normalisation test
and the ``c_row(v_from_overlap(ov)) == ov`` round trip are invariant under
``ov -> conj(ov)``.  The defect survived a full suite because the only thing that
distinguishes the two conventions is the DIRECTION of the inner product.

What is tested instead
----------------------
1. **Convention.**  ``<phi_{k,n}|G>`` is pinned against an oracle that shares no
   code with the production path: G is written out from the definition in the
   authors' ``gaussian_orbitals.jl`` and the integral is a Gauss-Legendre tensor
   rule, not the production midpoint rule.  The oracle is checked to be converged
   in its own node count, so a quadrature artefact cannot explain agreement.

2. **C6 covariance, gauge-covariantly.**  A Slater determinant is invariant under R
   iff its **occupied space** is -- NOT iff each orbital maps to another orbital.
   A magnetic rotation acts on the orbitals by a gauge transformation (an
   r-dependent phase), so the literal ``|psi(R^-1 r)|^2 == |psi(r)|^2`` test is
   invalid: it fails on a *correct* manifold too, which is how the earlier
   diagnosis came to blame the Bloch manifold.  The test here measures the
   residual of the rotated occupied space after a least-squares fit, which is
   gauge-invariant.

Measured on the seed ``VMC(N=36, rs=75, phase="crystal", nmax=2)``, ``L0 = 0.4``.
Every tolerance in this file is one of these numbers with headroom, never a guess:

    oracle convergence, max|oracle - production|
        GL nodes/dim    24        40        60         80
        <phi|G>      2.525e-02  2.184e-06  8.955e-14  2.422e-15
        <G|phi>      1.663e+00  1.660e+00  1.660e+00  1.660e+00
    the production midpoint rule itself, numx 101 vs 201   4.386e-15
    max|Im G| / max|G|   (the amplitude is real)           4.128e-34

    corrected  occupied-space residual, R = 60..300 deg    3.0e-14
    defective  (the conjugated overlap), same rotations    7.9e-01

The oracle table is the load-bearing one.  Its residual falls eleven orders from
24 to 60 nodes while the conjugated oracle stays pinned at 1.66e0 -- so the
agreement is quadrature truncation vanishing, not a coincidence, and at 60 nodes
the two conventions are separated by thirteen orders of magnitude.  ``C6_TOL``
likewise sits thirteen orders below the signal it guards, and the negative
controls below assert that the defective seed really does fail each gate.
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wigner_vmc.api import VMC                                       # noqa: E402
from wigner_vmc.physics import geometry as g                         # noqa: E402
from wigner_vmc.physics import landau_levels as ll                    # noqa: E402
from wigner_vmc.vmc import sr as sr                                   # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr                # noqa: E402

#: The point the anisotropy was observed at, so these guards guard THAT state.
RS = 75.0
L0 = 0.4
N_BAND = 2
NE = 36
PRIM_AREA = 2.0 * np.pi
LUMAX = 30.0
LUMAX_LL = 8.5 * np.sqrt(4.0 * np.pi / np.sqrt(3.0))

#: The production quadrature, and the oracle's independent node count.  The
#: oracle's residual is TRUNCATION and falls fast with nodes; both numbers below
#: are the measured ones (the whole convergence table is in the module docstring).
NUMX = 101
GL_NODES = 60
GL_COARSE = 40

#: MEASURED: |oracle(60) - production| = 8.955e-14.  Three orders of headroom.
ORACLE_TOL = 1e-10
#: MEASURED: |oracle(40) - production| = 2.184e-06, i.e. the coarse oracle is NOT
#: converged.  The bounds bracket that value from both sides, so this asserts a
#: stated fact about quadrature convergence rather than a fitted constant.
COARSE_MIN = 1e-8
COARSE_MAX = 1e-3
#: The occupied-space residual that counts as "invariant".  12 orders below the
#: defective seed's 7.9e-01.
C6_TOL = 1e-9
#: The negative control must be THIS far from invariant: a guard that cannot
#: fail is not a guard.
DEFECT_MIN = 1e-2
#: G is real to this relative precision -- the fact that makes the defect
#: invisible to every modulus test.
REAL_TOL = 1e-12

_CACHE = {}


def rot(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def setup():
    """The rs=75, nmax=2 crystal seed's ingredients.  Pure, so it is cached."""
    if "s" in _CACHE:
        return _CACHE["s"]
    A1, A2 = g.triangular_cell(PRIM_AREA)
    Ge = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
    ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
    C = np.column_stack([Ge.A1, Ge.A2])
    basis = ll.LandauLevelBasis(Ge.mesh, 8, ai, ac, C, np.linalg.inv(C))
    l_vals, l_cart = g.circular_lattice(LUMAX, Ge.L1, Ge.L2)
    L1, L2 = np.asarray(Ge.L1, float), np.asarray(Ge.L2, float)
    _CACHE["s"] = {
        "Ge": Ge, "basis": basis, "l_vals": l_vals, "l_cart": l_cart,
        "L1": L1, "L2": L2, "area": abs(L1[0] * L2[1] - L1[1] * L2[0]),
    }
    return _CACHE["s"]


# ==========================================================================
# the independent oracle.  Nothing here calls `gaussian_overlap_seed`.
# ==========================================================================
def _gaussian_amplitude(pts, s):
    """G(r): the site-centred magnetic-Bloch Gaussian of the Wigner crystal.

    Transcribed from the author's implementation, ``gaussian_orbitals.jl:75-93``
    (``gaussian_mbc``), which is the object ``gaussian_overlap_seed`` builds
    internally: one Gaussian of width L0 per lattice vector, each carrying the
    magnetic-translation phase ``(r x l)/2 + l_x l_y n_phi pi``.  Written out
    here rather than called, so that the oracle does not inherit a change in the
    production path.
    """
    l_vals, l_cart = np.asarray(s["l_vals"], float), np.asarray(s["l_cart"], float)
    n_phi = s["area"] / (2.0 * np.pi)
    big = pts[:, None, :] + l_cart[None, :, :]
    theta = ((pts[:, 0:1] * l_cart[None, :, 1] - pts[:, 1:2] * l_cart[None, :, 0]) / 2.0
             + l_vals[None, :, 0] * l_vals[None, :, 1] * n_phi * np.pi)
    gg = np.exp(-np.sum(big * big, axis=-1) / L0**2 / 4.0) / np.sqrt(2 * np.pi) / L0
    return np.sum(gg * np.exp(1j * theta), axis=1)


def _gl_nodes(ndeg, s):
    """A Gauss-Legendre tensor rule on the supercell -- NOT the midpoint rule."""
    x, w = np.polynomial.legendre.leggauss(ndeg)
    f = (x + 1.0) / 2.0
    L1, L2 = s["L1"], s["L2"]
    pts = (f[:, None, None] * L1[None, None, :]
           + f[None, :, None] * L2[None, None, :]).reshape(-1, 2) - (L1 + L2) / 2.0
    return pts, np.outer(w, w).ravel() * s["area"] / 4.0


def _oracle_overlap(ndeg, direction="phi_bra_gaussian", s=None):
    """``<phi_{k,n}|G>`` by direct quadrature, per momentum, normalised over n."""
    s = s or setup()
    pts, wts = _gl_nodes(ndeg, s)
    phi = np.empty((len(pts), s["basis"].nk, N_BAND), dtype=complex)
    for i in range(len(pts)):
        phi[i] = s["basis"].orbitals(pts[i])[:, :N_BAND]
    G = _gaussian_amplitude(pts, s)
    if direction == "phi_bra_gaussian":
        integrand = G[:, None, None] * phi.conj()
    elif direction == "gaussian_bra_phi":
        integrand = G.conj()[:, None, None] * phi
    else:
        raise ValueError(direction)
    ov = np.einsum("i,ikn->kn", wts, integrand)
    return ov / np.linalg.norm(ov, axis=1)[:, None]


def _production_overlap(numx=NUMX, s=None):
    """The seed through the production function.  Cached: it is the slow step
    (numx^2 orbital evaluations) and three test classes want the same array."""
    key = ("prod", numx)
    if key not in _CACHE:
        s = s or setup()
        _CACHE[key] = sr.gaussian_overlap_seed(
            s["basis"], N_BAND, s["L1"], s["L2"], s["l_vals"], s["l_cart"], RS,
            numx=numx, L0=L0)
    return _CACHE[key]


# ==========================================================================
# 1. the convention
# ==========================================================================
class TestProjectionConvention(unittest.TestCase):
    """`overlap[k, n]` is `<phi_{k,n} | G>`, pinned against an outside oracle."""

    @classmethod
    def setUpClass(cls):
        cls.s = setup()
        cls.prod = _production_overlap()
        cls.oracle = _oracle_overlap(GL_NODES)

    def test_the_overlap_is_phi_bra_gaussian_ket(self):
        """The number the production path returns IS the integral of conj(phi) G."""
        np.testing.assert_allclose(
            self.prod, self.oracle, rtol=ORACLE_TOL, atol=ORACLE_TOL,
            err_msg="gaussian_overlap_seed does not reproduce an independent "
                    "<phi|G> quadrature")

    def test_the_conjugate_convention_is_excluded(self):
        """The negative control: the reversed inner product is NOT what is returned.

        Without this, `test_the_overlap_is_phi_bra_gaussian_ket` would pass on any
        array that happened to be close to the oracle for an unrelated reason.
        Measured separation at 60 nodes: 1.660e+00 against a tolerance of 1e-10.
        """
        gap = np.abs(self.prod - self.oracle.conj()).max()
        self.assertGreater(
            gap, 1.0,
            "the conjugated oracle is within 1.0 of the production overlap -- the "
            "test above cannot discriminate the two conventions")

    def test_the_agreement_is_convergence_and_not_a_coincidence(self):
        """The residual is TRUNCATION: it must shrink by orders with the node count.

        A coarse oracle is genuinely far from production (measured 2.184e-06 at 40
        nodes) and a fine one is not (8.955e-14 at 60).  If the two conventions
        were being confused, or if the oracle shared code with the production
        path, this bracketing would collapse or the coarse point would already be
        at machine precision.
        """
        coarse = np.abs(_oracle_overlap(GL_COARSE, s=self.s) - self.prod).max()
        self.assertGreater(coarse, COARSE_MIN, f"coarse oracle already converged: {coarse:.3e}")
        self.assertLess(coarse, COARSE_MAX, f"coarse oracle is not small: {coarse:.3e}")
        self.assertLess(np.abs(self.oracle - self.prod).max(), ORACLE_TOL)

    def test_the_gaussian_amplitude_is_real(self):
        """Why the defect was invisible to every modulus test in the suite.

        G real => `ov_defect == conj(ov_correct)` exactly.  A conjugation is a
        pure phase change per row, and `v_from_overlap`/`c_row` are a phase-fixing
        pair, so the round trip and the row norms are identical under both
        conventions.  The direction of the inner product is the only observable.
        """
        pts, _ = _gl_nodes(20, self.s)
        G = _gaussian_amplitude(pts, self.s)
        self.assertLess(np.abs(G.imag).max() / np.abs(G).max(), REAL_TOL)

    def test_the_row_round_trip_cannot_see_the_convention(self):
        """The blindness, asserted: both conventions satisfy the existing guards.

        This is the reason a NEW guard was needed rather than a tightened
        tolerance on an old one.  If this ever fails, the old suite became
        sensitive and this file can be simplified.
        """
        for ov in (self.prod, self.prod.conj()):
            v = sr.v_from_overlap(ov)
            psi = ov / np.linalg.norm(ov, axis=1)[:, None]
            psi = psi * np.exp(-1j * np.angle(psi[:, 0]))[:, None]
            np.testing.assert_allclose(lr.c_row(v), psi, rtol=1e-12, atol=1e-12)


# ==========================================================================
# 2. C6 covariance of the seed, gauge-covariantly
# ==========================================================================
def _sample_orbitals(pts, s):
    out = np.empty((len(pts), s["basis"].nk, N_BAND), dtype=complex)
    for i in range(len(pts)):
        out[i] = s["basis"].orbitals(pts[i])[:, :N_BAND]
    return out


def _space_residual(Psi, PsiR, w):
    """How far the rotated occupied space is from the unrotated one.

    The determinant is invariant under R iff `PsiR = Psi @ cf` for some
    `(n_band, n_band)` matrix `cf`; this is the least-squares residual of that
    fit, which is invariant under a change of orbital gauge on either side --
    exactly the property the earlier `T_k(R^-1 r) == T_k(r)` test lacked.
    """
    S = w * (Psi.conj().T @ Psi)
    cf = np.linalg.solve(S, w * (Psi.conj().T @ PsiR))
    return np.linalg.norm(PsiR - Psi @ cf) / np.linalg.norm(PsiR)


class TestSeedC6Covariance(unittest.TestCase):

    NSP = 500
    ROTS = (60.0, 120.0, 180.0, 240.0, 300.0)

    @classmethod
    def setUpClass(cls):
        cls.s = setup()
        rng = np.random.default_rng(20261003)
        pts = rng.random((cls.NSP, 2)) @ np.column_stack([cls.s["L1"], cls.s["L2"]]).T
        w = cls.s["area"] / cls.NSP
        cls.w = w
        cls.V = _sample_orbitals(pts, cls.s)
        cls.W = {}
        for th in cls.ROTS:
            cls.W[th] = _sample_orbitals(pts @ rot(th), cls.s)
        # the two seeds, from the two conventions, straight through the
        # production map -- no hand-built parameter vectors.
        ov = _production_overlap()
        cls.v_correct = sr.v_from_overlap(ov)
        cls.v_defect = sr.v_from_overlap(ov.conj())

    def _residuals(self, v):
        C = lr.c_row(v)
        Psi = np.einsum("ikn,kn->ik", self.V, C)
        return {th: _space_residual(Psi, np.einsum("ikn,kn->ik", self.W[th], C),
                                    self.w)
                for th in self.ROTS}

    def test_the_corrected_seed_space_is_c6_invariant(self):
        """The gate.  All five rotations, not just the surviving 180 degrees."""
        res = self._residuals(self.v_correct)
        worst = max(res.values())
        self.assertLess(
            worst, C6_TOL,
            "the corrected seed's occupied space is not C6-invariant: "
            + ", ".join(f"R{th:.0f}={r:.3e}" for th, r in res.items()))

    def test_the_defective_seed_space_is_not(self):
        """The negative control, at the exact defect the suite must catch.

        This is what a mutation to `G.conj(), bloch` reproduces, so a passing
        run here means the guard above is measuring something real.
        """
        res = self._residuals(self.v_defect)
        for th in (60.0, 120.0, 240.0, 300.0):
            self.assertGreater(
                res[th], DEFECT_MIN,
                f"the conjugated seed passes the C6 gate at R{th:.0f} "
                f"({res[th]:.3e}) -- the gate is blind to the defect it exists for")

    def test_the_one_rotation_that_survives_both_conventions_is_180(self):
        """Why this defect hid for as long as it did.

        The conjugation flips the band phase `n theta -> -n theta`, and
        `e^{-i n theta} == e^{+i n theta}` only at theta = 180 deg.  So C2 is
        invariant under BOTH conventions and only C6 separates them -- the exact
        D2 pattern the delivered S(q) showed.
        """
        correct = self._residuals(self.v_correct)[180.0]
        defect = self._residuals(self.v_defect)[180.0]
        self.assertLess(correct, C6_TOL)
        self.assertLess(defect, C6_TOL)


# ==========================================================================
# 3. the production initializer really goes through the corrected line
# ==========================================================================
class TestProductionSeedWiring(unittest.TestCase):
    """`crystal_v0` must equal the corrected projection, not just be near it.

    The defect was in a helper; this pins the wiring from the public initializer
    down to it, so a future caller that re-derives the seed on its own path --
    or a revert of the one-line fix -- is caught here as well as in section 1.
    """

    @classmethod
    def setUpClass(cls):
        cls.vmc = VMC(N=NE, rs=RS, phase="crystal", nmax=N_BAND)
        # The production path's own quadrature (`api.OVERLAP_NUMX`), not the
        # test's, so this pins what actually runs.
        lat = cls.vmc.lat
        ov = sr.gaussian_overlap_seed(
            cls.vmc.basis(), cls.vmc.n_bands,
            np.asarray(lat.L1, float), np.asarray(lat.L2, float),
            np.asarray(lat.ov_ai, float), np.asarray(lat.ov_ac, float),
            RS, numx=121, L0=L0)
        cls.correct = np.asarray(sr.v_from_overlap(ov), complex)
        cls.crossed = np.asarray(sr.v_from_overlap(ov.conj()), complex)
        cls.actual = np.asarray(cls.vmc.crystal_v0(L0), complex)

    def test_crystal_v0_is_the_corrected_projection_of_the_gaussian(self):
        np.testing.assert_allclose(self.actual, self.correct,
                                   rtol=0.0, atol=0.0)

    def test_the_initializer_is_not_the_conjugated_projection(self):
        """The negative control on the wiring itself."""
        self.assertGreater(np.abs(self.actual - self.crossed).max(), DEFECT_MIN)


if __name__ == "__main__":
    unittest.main()
