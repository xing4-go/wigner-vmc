"""B7: stochastic reconfiguration against the frozen engine.

The optimiser is the last legacy subsystem and the one whose failures are
loudest: a difference here does not show up as a wrong number, it shows up as a
different state, and every energy downstream is measured on that state.  So the
comparison is not only "do the pieces agree" but "does the whole trajectory
agree, step for step, from one seed".

What is compared:

    free functions : jastrow_log_deriv, jastrow_vector, drummond_width,
                     v_from_overlap, gaussian_overlap_seed
    LL methods     : orbital_log_deriv, log_deriv, v_from_theta
    drivers        : sr_optimize_jastrow, sr_optimize_joint, sr_pilot_scale

The brief asks for ``S_old = S_new``, ``F_old = F_new``, ``Delta theta_old =
Delta theta_new``.  ``S`` and ``F`` are locals inside the two drivers and are not
returned, so they are pinned three ways rather than assumed:

1. their INPUTS are compared directly -- ``D = log_deriv`` and ``E`` from each
   implementation, on the same configurations (``TestLogDeriv``,
   ``TestOrbitalLogDeriv``);
2. ``|F|`` IS returned, as ``hist[i]["force"]``, and is compared step by step;
3. ``Delta theta`` is compared step by step as the recorded trajectory, and for
   the first step it is rebuilt here from the shared inputs by the transcribed
   update rule: ``S`` and ``F`` formed from ``D`` and ``E`` must reproduce the
   step the driver recorded.  A wrong ``S`` that produced the same ``Delta
   theta`` on every step of a 3-step run is not a realistic failure mode.

The drivers run real Metropolis walks.  They are the smallest walks that still
exercise the code path -- ``nsweep`` in the tens, not the notebooks' hundreds --
because the point is the update algebra, not the statistics.  Because ``sample``
is bit-identical to legacy (B4) and the RNG stream is consumed in the same order,
the two trajectories are identical move for move; one flipped Metropolis
acceptance would decorrelate them completely, which is why the trajectory test is
worth running at all.

Tolerances: the Gaussian path is bit-identical.  The LL path inherits the B5
deviation chain (clean ``padded_orbitals`` vs legacy ``orbitals`` at ~4e-16
relative), so ``D`` carries ~1e-15 and the tolerance is set from that
measurement, not guessed.
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
from wigner_vmc.vmc import sr                                    # noqa: E402
from wigner_vmc.wavefunctions import gaussian as gauss           # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw               # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr           # noqa: E402
from wigner_vmc.wavefunctions import slater as sl                # noqa: E402

PRIM_AREA = 2.0 * np.pi
NE = 36
LUMAX = 30.0
LUMAX_LL = 8.5 * np.sqrt(4 * np.pi / np.sqrt(3))
KAPPA = 32.0
NJ = 5
C5 = np.array([0.30, -0.55, 0.42, -0.18, 0.07])
N_BAND = 2

#: The Gaussian path is bit-identical (no determinant-basis deviation).
EXACT_RTOL = 0.0
EXACT_ATOL = 0.0
#: The LL path inherits B5's chain: clean `padded_orbitals` vs legacy `orbitals`
#: disagree at ~4e-16 relative, which reaches `D` (measured 9.2e-16 abs /
#: 4.1e-16 rel) and the metric built from it.  A tolerance two orders above the
#: measured input deviation would still fail on a real change.
LL_RTOL = 1e-12
LL_ATOL = 1e-12

#: The joint SR step solves an ill-conditioned system.  For the state used here
#: eig_min(S) ~ -1e-12 while eig_max ~ 20, so sigma_min(S_reg) is set by eps and
#: cond(S_reg) runs 2e4-4e4 (and ~2e6 in the eps-rescaling test, where eps is
#: deliberately made smaller).  A relative input deviation eps_in is therefore
#: amplified by roughly cond * eps_in in theta.  The tolerances below are built
#: from the MEASURED input deviation (`_LLSetup.input_rel`) times the cond the
#: run reports, times this slack -- never guessed.  A real defect in this code
#: moves theta by O(1), so even the loosest bound used here is five orders of
#: magnitude tighter than the signal it is guarding.
COND_SAFETY = 100.0

#: From step 1 on, the two joint-SR runs no longer evaluate the same function
#: at the same argument: their theta's differ by ~1e-10 (see
#: `TestSrOptimizeJoint`), and every theta-dependent scalar inherits that gap.
#: The honest comparison is therefore `gap * (that scalar's own rate of change
#: along the trajectory)`, where the rate is MEASURED from the runs
#: (`_traj_lip`) rather than assumed.  The gap is a near-random direction in the
#: 77-dimensional parameter space while the measured rate is along the direction
#: the trajectory actually took, so a decade of headroom is allowed.  At the
#: state used here the observed gap is 40-60x SMALLER than that extrapolation,
#: i.e. this is a floor with margin, not a fitted constant.
TRAJ_SAFETY = 10.0

#: `cond` and the `eig_*` pair come out of `eigvalsh`, where a 1e-15 relative
#: perturbation of the matrix entries can rotate near-degenerate eigenvectors.
#: Measured: |dcond|/cond ~ 5e-12.  Nothing about the implementation is being
#: tested at the 1e-12 level in those three numbers, so they use this instead.
EIG_RTOL = 1e-6

_CACHE = {}


def geom():
    """The shared geometry.  Pure, so caching it across test classes is free."""
    if "geom" not in _CACHE:
        A1, A2 = g.triangular_cell(PRIM_AREA)
        _CACHE["geom"] = g.Geometry.from_cell(A1, A2, 6, 6, 4.0)
    return _CACHE["geom"]


def configs(Ge, seed=20261002):
    """Configurations inside the supercell -- the regime the sampler produces."""
    key = ("cfgs", seed)
    if key not in _CACHE:
        sc = np.column_stack([Ge.L1, Ge.L2])
        rng = np.random.default_rng(seed)
        _CACHE[key] = {tag: rng.uniform(0.05, 0.95, size=(NE, 2)) @ sc.T
                       for tag in ("interior", "jittered")}
    return _CACHE[key]


def jastrow_pair(Ge, c=C5, kappa=KAPPA):
    gam = jw.cusp_gamma(kappa, Ge.L1)
    return (jw.SinSplineJastrow(c, Ge.G1, Ge.G2, gam),
            legacy.SinSplineJastrow(c, Ge.G1, Ge.G2, gam))


def coulomb_pair(Ge):
    """Both Ewald objects.  `energy(R)` does not mutate them, so sharing is safe."""
    if "ewald" not in _CACHE:
        _CACHE["ewald"] = (cb.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2),
                           legacy.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2))
    return _CACHE["ewald"]


def input_deviation(S, R):
    """Relative disagreement of the two implementations on the SR solve's inputs.

    The inputs are ``D = log_deriv`` and ``E``.  Whatever the metric then does to
    them is the reason the driver tolerances below are conditioning-aware, so
    this is measured rather than assumed.
    """
    wn, wo = S.mk_new(S.theta0), S.mk_old(S.theta0)
    sn, so = wn.build(R), wo.build(R)
    dn, do = wn.log_deriv(sn), wo.log_deriv(so)
    en, eo = wn.local_energy(sn)[2], wo.local_energy(so)[2]
    return (float(np.abs(dn - do).max() / np.abs(do).max()),
            float(abs(en - eo) / abs(eo)))


class _LLSetup:
    """One geometry, both implementations' wavefunction factories.

    The factories take the CONCATENATED theta so they can be handed straight to
    ``sr_optimize_joint``/``sr_pilot_scale``; the ordering is the legacy one,
    ``[c (NJ), Re v (nk*(NB-1)), Im v (nk*(NB-1))]``.

    The two bases are built with the values each class demands --
    ``nmax = n_band - 1`` for the clean one, ``n_max = n_band`` for the legacy
    one -- which is the §2A.5 mapping, not a fudge.
    """

    def __init__(self, Ge, n_band=N_BAND, kappa=KAPPA, v_seed=7):
        self.Ge = Ge
        self.n_band = n_band
        self.kappa = kappa
        ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
        C = np.column_stack([Ge.A1, Ge.A2])
        Ci = np.linalg.inv(C)
        self.new_basis = ll.LandauLevelBasis(Ge.mesh, n_band - 1, ai, ac, C, Ci)
        self.old_basis = legacy.LandauLevelBasis(Ge.mesh, n_band, ai, ac, C, Ci)
        self.nk = self.new_basis.nk
        rng = np.random.default_rng(v_seed)
        v0 = (rng.normal(size=(self.nk, n_band - 1))
              + 1j * rng.normal(size=(self.nk, n_band - 1)))
        self.v0 = v0
        self.theta0 = np.concatenate([C5, v0.real.ravel(), v0.imag.ravel()])
        self.Hn, self.Ho = coulomb_pair(Ge)
        self.sc = np.column_stack([Ge.L1, Ge.L2])
        self.input_rel = None       # filled by `measure_input_deviation(R)`

    def measure_input_deviation(self, R):
        """Latch the measured input disagreement so the tolerances can use it."""
        self.input_rel = max(input_deviation(self, R))
        return self.input_rel

    def step_tol(self, cond, scale):
        """Absolute tolerance on a theta step from a solve at conditioning `cond`."""
        assert self.input_rel is not None, "call measure_input_deviation first"
        return COND_SAFETY * cond * self.input_rel * scale

    def split(self, theta):
        m = (self.n_band - 1) * self.nk
        v = (theta[NJ:NJ + m].reshape(self.nk, self.n_band - 1)
             + 1j * theta[NJ + m:NJ + 2 * m].reshape(self.nk, self.n_band - 1))
        return np.asarray(theta[:NJ], float), v

    def mk_new(self, theta):
        c, v = self.split(theta)
        J, _ = jastrow_pair(self.Ge, c, self.kappa)
        orb = lr.LLRotatedOrbitals(self.new_basis, self.n_band, v)
        return lr.LLRotationWavefunction(orb, J, NE, self.sc,
                                         kappa=self.kappa, ham=self.Hn)

    def mk_old(self, theta):
        c, v = self.split(theta)
        _, J = jastrow_pair(self.Ge, c, self.kappa)
        orb = legacy_ll.LLRotatedOrbitals(self.old_basis, self.n_band, v)
        return legacy_ll.LLRotationWavefunction(orb, J, NE, self.sc,
                                                kappa=self.kappa, ham=self.Ho)


# ----------------------------------------------------------------------------
# The free functions
# ----------------------------------------------------------------------------


class TestJastrowLogDeriv(unittest.TestCase):
    """d ln|Psi|/dc -- the metric's input on the Gaussian path."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.cfgs = configs(cls.Ge)

    def test_it_agrees_bit_for_bit(self):
        for tag, R in self.cfgs.items():
            Jn, Jo = jastrow_pair(self.Ge)
            with self.subTest(tag):
                np.testing.assert_allclose(sr.jastrow_log_deriv(Jn, R),
                                           legacy.jastrow_log_deriv(Jo, R),
                                           rtol=EXACT_RTOL, atol=EXACT_ATOL)

    def test_it_is_half_the_masked_sum_and_not_the_whole(self):
        """The 1/2 is the reachable error: `_set_u` halves U exactly the same way."""
        R = self.cfgs["interior"]
        Jn, _ = jastrow_pair(self.Ge)
        ne = len(R)
        gp = Jn.grad_params((R[:, None, :] - R[None, :, :]).reshape(-1, 2))
        gp = gp.reshape(ne, ne, -1)
        mask = ~np.eye(ne, dtype=bool)
        got = sr.jastrow_log_deriv(Jn, R)
        self.assertEqual(got.shape, (NJ,))
        np.testing.assert_allclose(got, 0.5 * gp[mask].sum(axis=0),
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)
        self.assertFalse(np.allclose(got, gp[mask].sum(axis=0)))

    def test_the_diagonal_is_excluded_and_it_matters(self):
        """i == j is not a pair.  Including it would add a different constant to
        every Delta_k, so the test shows the two sums differ rather than equal."""
        R = self.cfgs["jittered"]
        Jn, _ = jastrow_pair(self.Ge)
        ne = len(R)
        gp = Jn.grad_params((R[:, None, :] - R[None, :, :]).reshape(-1, 2))
        gp = gp.reshape(ne, ne, -1)
        mask = ~np.eye(ne, dtype=bool)
        excl = 0.5 * gp[mask].sum(axis=0)
        incl = 0.5 * gp.sum(axis=0).sum(axis=0)
        self.assertFalse(np.allclose(excl, incl))
        np.testing.assert_allclose(sr.jastrow_log_deriv(Jn, R), excl,
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)


class TestJastrowVector(unittest.TestCase):
    """The reference's starting point -- a constant, so this one is exact."""

    def test_it_is_the_legacy_vector(self):
        for kappa in (0.0, 2.0, 32.0, 45.0):
            with self.subTest(kappa=kappa):
                np.testing.assert_array_equal(sr.jastrow_vector(kappa),
                                              legacy_ll.jastrow_vector(kappa))

    def test_its_shape_is_the_reference_five(self):
        want = -32.0 / 6 * np.array([4.0, 3.0, 2.0, 1.0, 0.0]) / 4.0
        np.testing.assert_array_equal(sr.jastrow_vector(32.0), want)
        self.assertEqual(sr.jastrow_vector(32.0, depth=3).shape, (3,))


class TestDrummondWidth(unittest.TestCase):

    def test_it_is_the_legacy_width(self):
        for rs in (1.0, 10.0, 45.0, 90.0):
            with self.subTest(rs=rs):
                np.testing.assert_allclose(sr.drummond_width(rs),
                                           legacy_ll.drummond_width(rs),
                                           rtol=EXACT_RTOL, atol=EXACT_ATOL)

    def test_it_falls_as_rs_to_the_quarter(self):
        """L0 = 0.5/sqrt(lam rs^0.5 nu/2) ~ rs^-1/4.  A pure old/new comparison
        would still pass if both sides had the same exponent slip, so the
        exponent is pinned independently."""
        a, b = sr.drummond_width(16.0), sr.drummond_width(256.0)
        self.assertAlmostEqual(b / a, 16.0 ** -0.25, places=12)


class TestVFromOverlap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(99)
        cls.ov = rng.normal(size=(12, 4)) + 1j * rng.normal(size=(12, 4))

    def test_it_is_the_legacy_map(self):
        np.testing.assert_allclose(sr.v_from_overlap(self.ov),
                                   legacy_ll.v_from_overlap(self.ov),
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)

    def test_it_is_the_inverse_of_c_row_up_to_the_gauge(self):
        """The seed is exact, not approximate -- the claim the notebooks print."""
        v = sr.v_from_overlap(self.ov)
        psi = self.ov / np.linalg.norm(self.ov, axis=1)[:, None]
        psi = psi * np.exp(-1j * np.angle(psi[:, 0]))[:, None]
        np.testing.assert_allclose(lr.c_row(v), psi, rtol=1e-12, atol=1e-12)

    def test_a_pure_lowest_level_overlap_gives_v_equals_zero(self):
        """b_norm < 1e-12 takes the `continue` branch -- the branch is load-bearing."""
        ov = np.zeros((3, 3), dtype=complex)
        ov[:, 0] = np.array([1.0, -1.0, 1j])
        np.testing.assert_array_equal(sr.v_from_overlap(ov),
                                      np.zeros((3, 2), dtype=complex))


class TestGaussianOverlapSeed(unittest.TestCase):
    """The LL seed: a Wigner crystal written in the Landau-level basis."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        ai, ac = g.circular_lattice(LUMAX_LL, cls.Ge.A1, cls.Ge.A2)
        C = np.column_stack([cls.Ge.A1, cls.Ge.A2])
        Ci = np.linalg.inv(C)
        # Both engines' `orbitals(r)` return n + 1 columns for constructor value
        # n, so this pair is built with the SAME number on both sides -- and it
        # is the notebook's own pair: LandauLevelBasis(mesh, 8, ...) seeded with
        # n_band = 2 (reproduction/make_notebook_llrot.py:1164,940).
        cls.new_b = ll.LandauLevelBasis(cls.Ge.mesh, 8, ai, ac, C, Ci)
        cls.old_b = legacy.LandauLevelBasis(cls.Ge.mesh, 8, ai, ac, C, Ci)
        cls.l_ints, cls.l_cart = g.circular_lattice(LUMAX, cls.Ge.L1, cls.Ge.L2)

    def _both(self, n_band, **kw):
        a = sr.gaussian_overlap_seed(self.new_b, n_band, self.Ge.L1, self.Ge.L2,
                                     self.l_ints, self.l_cart, **kw)
        b = legacy_ll.gaussian_overlap_seed(self.old_b, n_band, self.Ge.L1,
                                            self.Ge.L2, self.l_ints, self.l_cart,
                                            **kw)
        return a, b

    def test_the_seed_agrees(self):
        a, b = self._both(2, rs=45.0, numx=15)
        self.assertEqual(a.shape, (self.new_b.nk, 2))
        np.testing.assert_allclose(a, b, rtol=LL_RTOL, atol=LL_ATOL)

    def test_the_slice_from_a_wider_basis_is_what_runs(self):
        """`[:, :n_band]` on a 9-column basis: the notebook seeds an
        NMAX_STAGE_B=2 state from an LL_NMAX_BASIS=8 basis, so this is the path
        that actually runs, not the exact-width one.

        The returned array is normalised over n, so a 2-band and a 3-band seed
        are NOT nested column-wise -- they differ by that one normalisation and
        by nothing else.  Removing it must leave the first two columns equal,
        which is what pins the slice.
        """
        a, b = self._both(2, rs=45.0, numx=15)
        wide, wide_old = self._both(3, rs=45.0, numx=15)
        self.assertEqual(wide.shape, (self.new_b.nk, 3))
        w2 = wide[:, :2] / np.linalg.norm(wide[:, :2], axis=1)[:, None]
        np.testing.assert_allclose(a, w2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(wide, wide_old, rtol=LL_RTOL, atol=LL_ATOL)

    def test_the_default_width_is_the_drummond_width(self):
        """L0=None must reach `drummond_width`, not some other default."""
        a, _ = self._both(2, rs=45.0, numx=15)
        c, _ = self._both(2, rs=45.0, numx=15, L0=sr.drummond_width(45.0))
        np.testing.assert_allclose(a, c, rtol=EXACT_RTOL, atol=EXACT_ATOL)

    def test_it_is_normalised_per_momentum(self):
        a, _ = self._both(2, rs=45.0, numx=15)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1),
                                   np.ones(a.shape[0]), rtol=1e-13, atol=1e-13)

    def test_the_overlap_is_not_flat_in_n(self):
        """Guards a seed that returns something normalised but meaningless."""
        a, _ = self._both(2, rs=45.0, numx=15)
        self.assertGreater(float(np.abs(a[:, 1]).max()), 1e-3)


# ----------------------------------------------------------------------------
# The LL derivative methods
# ----------------------------------------------------------------------------


class TestVFromTheta(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.S = _LLSetup(cls.Ge)

    def test_it_is_the_inverse_of_the_concatenation(self):
        wf = self.S.mk_new(self.S.theta0)
        np.testing.assert_allclose(wf.v_from_theta(self.S.theta0), self.S.v0,
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)

    def test_it_agrees_with_legacy(self):
        wn, wo = self.S.mk_new(self.S.theta0), self.S.mk_old(self.S.theta0)
        np.testing.assert_allclose(wn.v_from_theta(self.S.theta0),
                                   wo.v_from_theta(self.S.theta0),
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)


class TestOrbitalLogDeriv(unittest.TestCase):
    """d ln det D / dv -- the complex half of the metric."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.S = _LLSetup(cls.Ge)
        cls.cfgs = configs(cls.Ge)

    def _both(self, R):
        wn, wo = self.S.mk_new(self.S.theta0), self.S.mk_old(self.S.theta0)
        return (wn.orbital_log_deriv(wn.build(R)),
                wo.orbital_log_deriv(wo.build(R)))

    def test_it_agrees(self):
        for tag, R in self.cfgs.items():
            an, bn = self._both(R)
            with self.subTest(tag):
                self.assertEqual(an[0].shape, (self.S.nk, N_BAND - 1))
                self.assertEqual(an[1].shape, (self.S.nk, N_BAND - 1))
                for a, b in zip(an, bn):
                    np.testing.assert_allclose(a, b, rtol=LL_RTOL, atol=LL_ATOL)

    def test_the_real_part_is_checked_against_finite_differences(self):
        """Independent of legacy: the analytic identity itself.

        d ln|det D| / d theta = Re Tr[D^-1 dD/dtheta], differenced on the orbital
        parameters.  The notebook prints this check (TEST 3); running it here pins
        the clean method even if the legacy one were wrong.
        """
        S = self.S
        R = self.cfgs["interior"]
        wf = S.mk_new(S.theta0)
        da, db = wf.orbital_log_deriv(wf.build(R))
        h = 1e-6
        m = N_BAND - 1
        for arr, block in ((da, 0), (db, 1)):
            k, j = 3, 0
            idx = NJ + block * m * S.nk + k * m + j
            fd = []
            for sgn in (+1, -1):
                t = S.theta0.copy()
                t[idx] += sgn * h
                fd.append(np.log(abs(np.linalg.det(S.mk_new(t).build(R)["D"]))))
            with self.subTest(part=("da", "db")[block]):
                self.assertAlmostEqual(arr[k, j].real,
                                       (fd[0] - fd[1]) / (2 * h), places=5)
        self.assertTrue(np.any(da.imag != 0.0))   # and it is genuinely complex


class TestLogDeriv(unittest.TestCase):
    """The full DR vector D -- the metric's input on the LL path."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.S = _LLSetup(cls.Ge)
        cls.cfgs = configs(cls.Ge)
        cls.npar = NJ + 2 * (N_BAND - 1) * cls.S.nk

    def _both(self, R):
        wn, wo = self.S.mk_new(self.S.theta0), self.S.mk_old(self.S.theta0)
        return wn.log_deriv(wn.build(R)), wo.log_deriv(wo.build(R))

    def test_it_agrees(self):
        for tag, R in self.cfgs.items():
            a, b = self._both(R)
            with self.subTest(tag):
                self.assertEqual(a.shape, (self.npar,))
                np.testing.assert_allclose(a, b, rtol=LL_RTOL, atol=LL_ATOL)

    def test_the_three_blocks_are_the_documented_ordering(self):
        """[c, Re v, Im v] -- the ordering sr_optimize_joint and the notebooks share."""
        R = self.cfgs["interior"]
        wf = self.S.mk_new(self.S.theta0)
        st = wf.build(R)
        d = wf.log_deriv(st)
        da, db = wf.orbital_log_deriv(st)
        np.testing.assert_allclose(d[:NJ], sr.jastrow_log_deriv(wf.jastrow, R),
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)
        np.testing.assert_allclose(d[NJ:NJ + da.size], da.ravel(),
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)
        np.testing.assert_allclose(d[NJ + da.size:], db.ravel(),
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)

    def test_the_jastrow_block_is_real_and_the_orbital_block_is_not(self):
        R = self.cfgs["interior"]
        wf = self.S.mk_new(self.S.theta0)
        d = wf.log_deriv(wf.build(R))
        self.assertEqual(float(np.abs(d[:NJ].imag).max()), 0.0)
        self.assertGreater(float(np.abs(d[NJ:].imag).max()),
                           1e-3 * float(np.abs(d[NJ:]).max()))

    def test_a_state_without_a_jastrow_drops_the_first_block(self):
        S = self.S
        wf = lr.LLRotationWavefunction(
            lr.LLRotatedOrbitals(S.new_basis, N_BAND, S.v0), None, NE, S.sc,
            kappa=0.0, ham=S.Hn)
        old = legacy_ll.LLRotationWavefunction(
            legacy_ll.LLRotatedOrbitals(S.old_basis, N_BAND, S.v0), None, NE,
            S.sc, kappa=0.0, ham=S.Ho)
        R = self.cfgs["interior"]
        a, b = wf.log_deriv(wf.build(R)), old.log_deriv(old.build(R))
        self.assertEqual(a.shape, (2 * (N_BAND - 1) * S.nk,))
        np.testing.assert_allclose(a, b, rtol=LL_RTOL, atol=LL_ATOL)


# ----------------------------------------------------------------------------
# The drivers
# ----------------------------------------------------------------------------


def gaussian_factories(Ge):
    """``make_wf(c)`` for both Gaussian implementations, sharing the Ewald objects.

    ``sr_optimize_jastrow`` calls ``make_wf`` once per step, so the bases and the
    Ewald sums are built here and closed over rather than rebuilt.
    """
    Hn, Ho = coulomb_pair(Ge)
    sc = np.column_stack([Ge.L1, Ge.L2])
    sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
    li, lc = g.circular_lattice(LUMAX, Ge.L1, Ge.L2)
    bn = gauss.GaussianBasis(sites, li, lc, Ge.L1, Ge.L2, l=1.0)
    bo = legacy.GaussianBasis(sites, li, lc, Ge.L1, Ge.L2, l=1.0)
    L0 = 0.6

    def mk_new(c):
        J, _ = jastrow_pair(Ge, c)
        return sl.Wavefunction(lambda r: bn.orbitals(r, L0),
                               lambda r: bn.pi_orbitals(r, L0),
                               lambda r: bn.pi_square_orbitals(r, L0),
                               J, NE, sc, kappa=KAPPA, ham=Hn)

    def mk_old(c):
        _, J = jastrow_pair(Ge, c)
        return legacy.Wavefunction(lambda r: bo.orbitals(r, L0),
                                   lambda r: bo.pi_orbitals(r, L0),
                                   lambda r: bo.pi_square_orbitals(r, L0),
                                   J, NE, sc, kappa=KAPPA, ham=Ho)
    return mk_new, mk_old


class TestSrOptimizeJastrow(unittest.TestCase):
    """The Gaussian driver: SR over the Jastrow alone (real throughout)."""

    STEPS, NSWEEP, EQUIL, SNAP = 3, 25, 8, 2
    SEED, SIGMA = 11, 0.4

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.R0 = configs(cls.Ge)["interior"]
        # staticmethod: a plain function stored as a class attribute becomes a
        # descriptor, so `self.mk_new(c)` would bind `self` as the first argument
        # (the same trap as in test_measure_against_legacy.py).
        mk_new, mk_old = gaussian_factories(cls.Ge)
        cls.mk_new = staticmethod(mk_new)
        cls.mk_old = staticmethod(mk_old)

    def _run(self, mk, c0=None, **over):
        kw = dict(steps=self.STEPS, nsweep=self.NSWEEP, sigma=self.SIGMA,
                  seed=self.SEED, tau=2e-2, xi=2.0, eps=1e-3, equil=self.EQUIL,
                  snapshot_every=self.SNAP, verbose=False)
        kw.update(over)
        return sr.sr_optimize_jastrow(mk, C5 if c0 is None else c0, self.R0, **kw)

    def test_the_trajectory_is_identical(self):
        cn, hn = self._run(self.mk_new)
        co, ho = self._run(self.mk_old)
        self.assertEqual(len(hn), self.STEPS)
        for i, (a, b) in enumerate(zip(hn, ho)):
            with self.subTest(step=i):
                self.assertEqual(sorted(a), sorted(b))
                np.testing.assert_array_equal(a["c"], b["c"])
                self.assertEqual(a["acc"], b["acc"])
                self.assertEqual(a["sigma"], b["sigma"])
                self.assertEqual(a["step"], b["step"])
                for key in ("E", "E_err", "force", "tau"):
                    np.testing.assert_allclose(a[key], b[key],
                                               rtol=EXACT_RTOL, atol=EXACT_ATOL)
        np.testing.assert_array_equal(cn, co)

    def test_the_recorded_step_is_what_the_legacy_update_rule_predicts(self):
        """S, F and Delta theta on the real path, rebuilt from scratch.

        The walk is replayed from the same seed so the snapshots are the ones the
        driver saw, D and E come from the clean implementation, and S and F are
        formed by the transcribed rule.  The step the driver recorded must be the
        step that metric produces.
        """
        cn, hn = self._run(self.mk_new)
        eps, tau = 1e-3, 2e-2
        wf = self.mk_new(C5)
        snaps, _, _ = sr.sample(wf, self.R0, nsweep=self.NSWEEP, sigma=self.SIGMA,
                                rng=np.random.default_rng(self.SEED),
                                snapshot_every=self.SNAP, equil=self.EQUIL,
                                target_acc=0.5)
        E = np.array([wf.local_energy(wf.build(S))[2].real for S in snaps])
        D = np.array([sr.jastrow_log_deriv(wf.jastrow, S) for S in snaps])
        n = len(snaps)
        ebar, dbar = E.mean(), D.mean(axis=0)
        S_mat = (D.T @ D) / n - np.outer(dbar, dbar) + eps * np.eye(NJ)
        f = (D.T @ E) / n - dbar * ebar
        delta = -(tau / (1.0 + 2.0 * 0 / self.STEPS)) * np.linalg.solve(S_mat, f)
        # C5 + delta, NOT hn[0]["c"] - C5: the driver stores c = C5 + delta, so
        # subtracting C5 back out cancels ~1 digit and would compare a rounded
        # difference against an exact one.
        np.testing.assert_allclose(C5 + delta, hn[0]["c"],
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)
        np.testing.assert_allclose(np.linalg.norm(f), hn[0]["force"],
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)
        np.testing.assert_allclose(E.mean(), hn[0]["E"],
                                   rtol=EXACT_RTOL, atol=EXACT_ATOL)

    def test_the_recorded_E_is_a_total_and_not_per_electron(self):
        """The factor-of-ne trap: hist[i]["E"] is ebar, and measure()["E"] is
        ebar/ne.  A later refactor that 'fixed' this would move every quoted
        optimisation energy by 36."""
        _, hn = self._run(self.mk_new)
        self.assertGreater(abs(hn[0]["E"]), 10.0)

    def test_a_different_starting_point_gives_a_different_optimum(self):
        """Sanity: the comparison above is not passing for any c0."""
        a, _ = self._run(self.mk_new)
        b, _ = self._run(self.mk_new, c0=C5 * 1.5)
        self.assertFalse(np.allclose(a, b))


class TestSrOptimizeJoint(unittest.TestCase):
    """The LL driver: SR over [c, a, b] with a complex metric."""

    STEPS, NSWEEP, EQUIL, SNAP = 3, 25, 8, 3
    SEED, SIGMA = 5, 0.4

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.S = _LLSetup(cls.Ge)
        cls.R0 = configs(cls.Ge)["interior"]
        cls.dD, cls.dE = input_deviation(cls.S, cls.R0)
        cls.S.measure_input_deviation(cls.R0)

    def _run(self, mk, **over):
        kw = dict(steps=self.STEPS, nsweep=self.NSWEEP, sigma=self.SIGMA,
                  seed=self.SEED, tau=2e-2, xi=2.0, eps=1e-3, equil=self.EQUIL,
                  snapshot_every=self.SNAP, verbose=False)
        kw.update(over)
        return sr.sr_optimize_joint(mk, self.S.theta0, self.R0, **kw)

    def _step_bound(self, cond, scale):
        return self.S.step_tol(cond, scale)

    @staticmethod
    def _traj_lip(h, key):
        """The rate at which a recorded scalar moves along a trajectory.

        ``max_i |X_{i+1} - X_i| / |theta_{i+1} - theta_i|`` over the run -- the
        only thing that can turn a theta gap into a gap in ``X``.  Measured from
        the run itself, so it needs no model of how ``X`` depends on theta.
        """
        r = 0.0
        for i in range(len(h) - 1):
            dth = float(np.abs(h[i + 1]["theta"] - h[i]["theta"]).max())
            r = max(r, abs(float(h[i + 1][key]) - float(h[i][key])) / dth)
        return r

    @staticmethod
    def _floor(key, mag):
        """Rounding floor for one recorded scalar, at ONE theta.

        ``mag`` is the magnitude to scale the floor by -- the value itself for
        every scalar, except that the eigenvalue-derived numbers are all scaled
        by ``eig_max``, because ``eig_min`` sits on top of zero and its own
        magnitude says nothing about how far it can move.
        """
        return LL_ATOL + (EIG_RTOL if key in ("cond", "eig_min", "eig_max")
                          else LL_RTOL) * abs(float(mag))

    def test_the_inputs_agree_to_the_measured_deviation(self):
        """D and E on one configuration.  This is the number the driver
        tolerances are built from, so it is asserted, not assumed."""
        self.assertLess(self.dD, 1e-13)
        self.assertLess(self.dE, 1e-13)

    def test_the_first_step_agrees_field_by_field(self):
        """One step from identical inputs: every recorded field to rounding.

        This is the sharp regression on the joint path, and it has to be ONE
        step.  Both runs start at the same theta and consume the same walk, so
        S, f, Delta theta and every diagnostic the driver records
        (E, E_err, force, tau, cond, the spectrum) are the same function of the
        same inputs, and a changed implementation is visible at 1e-12.

        From step 1 that is no longer true -- the metric is ill-conditioned
        enough that the 1e-15 input disagreement grows to ~1e-10 in theta, and
        the two runs then evaluate at different arguments.  See
        `test_the_trajectory_agrees` for what is asserted past step 0.
        """
        tn, hn = self._run(self.S.mk_new, steps=1)
        to, ho = self._run(self.S.mk_old, steps=1)
        self.assertEqual(len(hn), 1)
        a, b = hn[0], ho[0]
        self.assertEqual(sorted(a), sorted(b))
        scale = float(np.abs(self.S.theta0).max())

        # theta is the SOLVED quantity, so it carries the solve's conditioning:
        # a 1e-15 relative disagreement in S and f lands at ~2e-12 in theta here
        # (cond ~ 2e4).  This is asserted against the same measured-conditioning
        # bound the trajectory test uses -- tight enough to be a real check
        # (the observed margin is ~2e4) but not pretending the solve is exact.
        dev = float(np.abs(a["theta"] - b["theta"]).max())
        bound = self._step_bound(a["cond"], scale)
        self.assertLess(dev, bound,
                        f"step 0: |dtheta| {dev:.3e} vs bound {bound:.3e}")

        # everything the driver records FROM those inputs is not amplified
        for key in ("E", "E_err", "force", "tau"):
            np.testing.assert_allclose(a[key], b[key], rtol=LL_RTOL,
                                       atol=LL_ATOL, err_msg=key)
        for key in ("cond", "eig_max"):
            np.testing.assert_allclose(a[key], b[key], rtol=EIG_RTOL,
                                       atol=self._floor(key, a[key]), err_msg=key)
        np.testing.assert_allclose(a["eig_min"], b["eig_min"], rtol=EIG_RTOL,
                                   atol=self._floor("eig_min", a["eig_max"]),
                                   err_msg="eig_min")
        self.assertEqual(a["acc"], b["acc"])
        self.assertEqual(a["sigma"], b["sigma"])

        # the step is not trivially zero -- otherwise everything above passes
        # for free
        self.assertGreater(float(np.abs(a["theta"] - self.S.theta0).max()), 0.0)

    def test_the_trajectory_agrees(self):
        """Three steps: the two runs stay on the same path.

        Only step 0 has the same input on both sides, so only step 0 can be
        compared at rounding tolerance -- that is what
        `test_the_first_step_agrees_field_by_field` does, on the same quantities
        and more of them.  Here the theta trajectory itself is still asserted
        tightly (against the conditioning bound), the theta-INDEPENDENT fields
        are asserted exactly, and each theta-dependent scalar is asserted
        against theta's own gap propagated through that scalar's measured rate
        of change along this trajectory -- so the check is on the trajectory,
        not on a tolerance standing in for one.
        """
        tn, hn = self._run(self.S.mk_new)
        to, ho = self._run(self.S.mk_old)
        self.assertEqual(len(hn), self.STEPS)
        scale = float(np.abs(self.S.theta0).max())
        keys = ("E", "E_err", "force", "cond")
        lip = {k: max(self._traj_lip(hn, k), self._traj_lip(ho, k)) for k in keys}

        prev_dev = 0.0
        for i, (a, b) in enumerate(zip(hn, ho)):
            with self.subTest(step=i):
                self.assertEqual(sorted(a), sorted(b))
                dev = float(np.abs(a["theta"] - b["theta"]).max())
                bound = self._step_bound(a["cond"], scale)
                self.assertLess(dev, bound,
                                f"step {i}: |dtheta| {dev:.3e} vs bound {bound:.3e}")

                # nothing to do with theta: exact, at every step
                self.assertEqual(a["acc"], b["acc"])
                self.assertEqual(a["sigma"], b["sigma"])
                self.assertEqual(a["tau"], b["tau"])

                # theta-dependent scalars, propagated.  At step 0 `prev_dev` is
                # 0, so the bound collapses to the rounding floor and this is
                # still a sharp check.
                for k in keys:
                    d = abs(float(a[k]) - float(b[k]))
                    lim = (TRAJ_SAFETY * lip[k] * prev_dev + self._floor(k, a[k]))
                    self.assertLess(d, lim, f"step {i} {k}: {d:.3e} vs {lim:.3e}")
            prev_dev = dev

        self.assertLess(float(np.abs(tn - to).max()),
                        max(self._step_bound(h["cond"], scale) for h in hn))

    def test_the_recorded_step_is_what_the_legacy_update_rule_predicts(self):
        """S, F and Delta theta on the complex (orbital) path.

        D is complex here and E real; the legacy rule takes `.real` of both the
        metric and the gradient, and this reproduces exactly that.  A dropped or
        misplaced `.real` changes the step and the test fails.
        """
        tn, hn = self._run(self.S.mk_new)
        eps, tau = 1e-3, 2e-2
        wf = self.S.mk_new(self.S.theta0)
        snaps, _, _ = sr.sample(wf, self.R0, nsweep=self.NSWEEP,
                                sigma=self.SIGMA,
                                rng=np.random.default_rng(self.SEED),
                                snapshot_every=self.SNAP, equil=self.EQUIL,
                                target_acc=0.5)
        n = len(snaps)
        np_ = len(self.S.theta0)
        E = np.empty(n)
        D = np.empty((n, np_), dtype=complex)
        for s, cfg in enumerate(snaps):
            st = wf.build(cfg)
            E[s] = wf.local_energy(st)[2].real
            D[s] = wf.log_deriv(st)
        ebar, dbar = E.mean(), D.mean(axis=0)
        S_full = (D.conj().T @ D) / n - np.outer(dbar.conj(), dbar)
        f_full = (D.conj().T @ E) / n - dbar.conj() * ebar
        delta = -(tau / (1.0 + 2.0 * 0 / self.STEPS)) * np.linalg.solve(
            S_full.real + eps * np.eye(np_), f_full.real)
        np.testing.assert_allclose(self.S.theta0 + delta, hn[0]["theta"],
                                   rtol=LL_RTOL, atol=LL_ATOL)
        np.testing.assert_allclose(np.linalg.norm(f_full.real), hn[0]["force"],
                                   rtol=LL_RTOL, atol=LL_ATOL)
        # and the part that is discarded is not negligible -- otherwise taking
        # `.real` would be untestable and its removal would go unnoticed
        self.assertGreater(float(np.abs(S_full.imag).max()),
                           1e-6 * float(np.abs(S_full.real).max()))

    def test_a_nonuniform_scale_agrees_with_legacy(self):
        sc = np.ones(len(self.S.theta0))
        sc[:NJ] = 0.05
        a, ha = self._run(self.S.mk_new, scale=sc)
        b, hb = self._run(self.S.mk_old, scale=sc)
        t, _ = self._run(self.S.mk_new)
        scale = float(np.abs(self.S.theta0).max())
        cond = max(h["cond"] for h in ha)
        self.assertLess(float(np.abs(a - b).max()), self._step_bound(cond, scale))
        for x, y in zip(ha, hb):
            self.assertLess(float(np.abs(x["theta"] - y["theta"]).max()),
                            self._step_bound(x["cond"], scale))
        self.assertFalse(np.allclose(a, t))   # the scale really did something

    def test_a_uniform_scale_is_exactly_a_rescaled_regulariser(self):
        """theta = s*x makes S_x = s^2 S and f_x = s f, so the theta-step is the
        unscaled step with eps -> eps/s^2.  This is the precise content of
        'eps is applied AFTER the rescaling'; moving the `+=` would break it.

        The second run is much better conditioned than the first (its
        sigma_min is eps/s^2, not eps), so the tolerance is taken from the worse
        of the two.
        """
        s = 10.0
        a, ha = self._run(self.S.mk_new, scale=np.full(len(self.S.theta0), s),
                          eps=1e-3)
        b, hb = self._run(self.S.mk_new, scale=None, eps=1e-3 / s**2)
        c, _ = self._run(self.S.mk_new, scale=None, eps=1e-3)
        scale = float(np.abs(self.S.theta0).max())
        cond = max(max(h["cond"] for h in ha), max(h["cond"] for h in hb))
        self.assertLess(float(np.abs(a - b).max()), self._step_bound(cond, scale))
        self.assertFalse(np.allclose(a, c))

    def test_the_history_reports_the_metric_conditioning(self):
        """cond/eig_min/eig_max exist because 'SR is unstable' is a statement
        about them (II.12b).  They are the spectrum of S_x BEFORE the
        regulariser (eig_*) and of the matrix actually solved (cond), so the
        three are tied together by one identity -- which is what pins their
        meaning rather than just their presence."""
        _, hn = self._run(self.S.mk_new)
        row = hn[0]
        eps = 1e-3
        self.assertLessEqual(row["eig_min"], row["eig_max"])
        # S is numerically singular -- the raw metric's smallest eigenvalue is
        # negative round-off -- and eps is what makes the solved matrix positive
        # definite.  That is the whole reason the regulariser is there.
        self.assertLess(row["eig_min"], 1e-9)
        self.assertGreater(row["eig_min"] + eps, 0.0)
        self.assertAlmostEqual(
            row["cond"], (row["eig_max"] + eps) / (row["eig_min"] + eps),
            delta=1e-6 * row["cond"])


class TestSrPilotScale(unittest.TestCase):
    """``scale_i = 1/sqrt(var(D_i))`` -- the diagnosed fix for the eps mismatch."""

    NSWEEP, EQUIL, SEED = 20, 6, 3

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.S = _LLSetup(cls.Ge)
        cls.R0 = configs(cls.Ge)["interior"]

    def _run(self, mk, **over):
        kw = dict(nsweep=self.NSWEEP, seed=self.SEED, sigma=0.3, equil=self.EQUIL,
                  snapshot_every=1, target_acc=0.4)
        kw.update(over)
        return sr.sr_pilot_scale(mk, self.S.theta0, self.R0, **kw)

    def test_it_agrees(self):
        a = self._run(self.S.mk_new)
        b = self._run(self.S.mk_old)
        self.assertEqual(a.shape, (len(self.S.theta0),))
        np.testing.assert_allclose(a, b, rtol=LL_RTOL, atol=LL_ATOL)

    def test_it_normalises_the_derivative_variance_to_one(self):
        """The definition, recomputed independently: scale_i^2 * var(D_i) = 1."""
        S = self.S
        wf = S.mk_new(S.theta0)
        snaps, _, _ = sr.sample(wf, self.R0, nsweep=self.NSWEEP, sigma=0.3,
                                rng=np.random.default_rng(self.SEED),
                                snapshot_every=1, equil=self.EQUIL,
                                target_acc=0.4)
        D = np.array([wf.log_deriv(wf.build(cfg)) for cfg in snaps])
        Sii = (np.abs(D) ** 2).mean(axis=0) - np.abs(D.mean(axis=0)) ** 2
        scale = self._run(S.mk_new)
        live = Sii > 1e-14
        np.testing.assert_allclose(scale[live] ** 2 * Sii[live],
                                   np.ones(live.sum()), rtol=1e-10)
        self.assertGreater(int(live.sum()), 0)

    def test_the_scales_are_not_all_one_size(self):
        """The reason it exists: `eps` is one absolute number added to every
        diagonal entry, so it is only the right regulariser if the parameters
        are on the same footing -- and they are not.

        What is asserted is the SPREAD, which is the property the mechanism
        needs, not the direction of the block split, which the legacy
        docstring's own numbers (Jastrow O(10-40), orbital O(1e-3)) describe
        for a production state and which does NOT survive at this one: measured
        here, the block maxima of `var(D)` differ by only ~1.5x and the scales
        interleave.  Measured at this state the 77 scales span 0.57 to 11.

        The derivative magnitudes themselves are a different matter and are
        checked in `test_the_derivative_magnitudes_differ_between_blocks`.
        """
        a = self._run(self.S.mk_new)
        self.assertGreater(a.max() / a.min(), 10.0)

    def test_the_derivative_magnitudes_differ_between_blocks(self):
        """The factual content behind "not on the same footing", measured.

        The legacy docstring's exponents are for a production state; here the
        Jastrow block still carries derivatives two orders larger than the
        orbital block.  That is the disparity an absolute `eps` mishandles, and
        it is measured rather than taken from the docstring.
        """
        S = self.S
        wf = S.mk_new(S.theta0)
        snaps, _, _ = sr.sample(wf, self.R0, nsweep=self.NSWEEP, sigma=0.3,
                                rng=np.random.default_rng(self.SEED),
                                snapshot_every=1, equil=self.EQUIL,
                                target_acc=0.4)
        D = np.array([wf.log_deriv(wf.build(cfg)) for cfg in snaps])
        self.assertGreater(float(np.abs(D[:, :NJ]).max()),
                           10.0 * float(np.abs(D[:, NJ:]).max()))

    def test_an_unstable_parameter_is_held_at_the_floor(self):
        """`np.maximum(Sii, floor)` -- reached when a parameter does not move.

        A pilot walk where every derivative is exactly constant gives Sii = 0 for
        every parameter, which is the degenerate case the floor exists for.
        """
        npar = len(self.S.theta0)
        frozen = np.arange(npar, dtype=float)

        class _Frozen:
            """Minimal stand-in: `sample` needs build/move_ratio/accept_move."""
            ne = NE
            sc_to_cart = np.eye(2)
            cart_to_sc = np.eye(2)

            def build(self, R):
                return {"R": np.asarray(R, float)}

            def log_deriv(self, st):
                return frozen

            def move_ratio(self, st, i, r_new):
                return 0.0, None

            def accept_move(self, st, i, r_new, info):
                pass

        wf = _Frozen()
        scale = sr.sr_pilot_scale(lambda theta: wf, self.S.theta0, self.R0,
                                  nsweep=4, seed=self.SEED, sigma=0.3, equil=1,
                                  snapshot_every=1, target_acc=0.4)
        np.testing.assert_allclose(scale, np.full(npar, 1.0 / np.sqrt(1e-14)),
                                   rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
