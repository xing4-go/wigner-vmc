"""B3: `wavefunctions/ll_rotation.py` + `nesting.py` against the frozen engine.

Two separate claims are checked here.

**Against legacy.**  `c_row`, `dc_row_dparams`, `LLRotatedOrbitals.orbitals`,
`.pi`, `.pi2`, `.batch`, and the wavefunction's `build`/`rebuild` determinant,
on fixed ``v``.  The legacy basis is built with ``n_max = n_band`` and the clean
one with ``nmax = n_band - 1``; that is the §2A.5 convention, and the test
constructs both so the mapping cannot drift.

**Nesting.**  ``(c_0, c_1) -> (c_0, c_1, 0)``.  The whole ``nmax`` ladder's
one-sidedness rests on this identity, so it is measured rather than asserted:
the padded coefficients must equal the source's columns verbatim, with an exact
zero in the new column.

The gauge sign is pinned by finite differences, as in the LL-basis file, because
``pi^2`` is blind to it and every energy is built from ``pi^2``.
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
from wigner_vmc.physics import geometry as g                     # noqa: E402
from wigner_vmc.physics import landau_levels as ll               # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw               # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr           # noqa: E402
from wigner_vmc.wavefunctions import nesting as nest             # noqa: E402

PRIM_AREA = 2.0 * np.pi
LUMAX_LL = 8.5 * np.sqrt(4 * np.pi / np.sqrt(3))
NE = 36
RTOL = 1e-12
ATOL = 1e-13

POINTS = (np.array([0.31, -0.77]), np.array([1.10, 0.40]), np.array([-2.00, 1.30]),
          np.array([0.0, 0.0]), np.array([0.5, 0.25]))


def geom():
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, 4.0)


def fixed_v(nk, m, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(nk, m)) + 1j * rng.normal(size=(nk, m))


def setup(n_band, seed=7):
    """(clean orb, legacy orb, geometry) for one band count."""
    Ge = geom()
    ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
    C = np.column_stack([Ge.A1, Ge.A2])
    Ci = np.linalg.inv(C)
    v = fixed_v(len(Ge.mesh), n_band - 1, seed)
    new_basis = ll.LandauLevelBasis(Ge.mesh, n_band - 1, ai, ac, C, Ci)
    old_basis = legacy.LandauLevelBasis(Ge.mesh, n_band, ai, ac, C, Ci)
    new = lr.LLRotatedOrbitals(new_basis, n_band, v)
    old = legacy_ll.LLRotatedOrbitals(old_basis, n_band, v)
    return new, old, Ge, v


class TestTheBandCountMapping(unittest.TestCase):
    def test_the_clean_basis_needs_nmax_equals_n_band_minus_one(self):
        Ge = geom()
        ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
        C = np.column_stack([Ge.A1, Ge.A2])
        Ci = np.linalg.inv(C)
        for n_band in (1, 2, 3, 4):
            b = ll.LandauLevelBasis(Ge.mesh, n_band - 1, ai, ac, C, Ci)
            self.assertEqual(b.n_bands, n_band)
            orb = lr.LLRotatedOrbitals(b, n_band, fixed_v(36, n_band - 1))
            self.assertEqual(orb.nb, n_band)
        # and the mismatched pairing is refused rather than silently producing
        # orbitals with the wrong number of bands
        b2 = ll.LandauLevelBasis(Ge.mesh, 1, ai, ac, C, Ci)
        with self.assertRaises(ValueError):
            lr.LLRotatedOrbitals(b2, 3, fixed_v(36, 2))


class TestAgainstLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.new, cls.old, cls.Ge, cls.v = setup(3)

    def test_wtaylor_limits(self):
        # the t -> 0 branch is the liquid, so it is not a corner case
        t = np.array([0.0, 1e-12, 1e-8, 1e-6, 1e-5, 0.5, 3.0])
        for k, f in (("w1", lambda x: lr._wtaylor(x)[0]),
                     ("w4", lambda x: lr._wtaylor(x)[1])):
            mine = f(t)
            theirs = legacy_ll._wtaylor(t)[0 if k == "w1" else 1]
            np.testing.assert_allclose(mine, theirs, rtol=0, atol=0, err_msg=k)
        self.assertAlmostEqual(float(lr._wtaylor(np.array([0.0]))[0][0]), 1.0, places=15)
        self.assertAlmostEqual(float(lr._wtaylor(np.array([0.0]))[1][0]), -1.0 / 3.0,
                               places=15)

    def test_c_row(self):
        for m in (1, 2, 3):
            v = fixed_v(36, m)
            np.testing.assert_allclose(lr.c_row(v), legacy_ll.c_row(v), rtol=0, atol=0)
        # v = 0 is the liquid: C = (1, 0, ...)
        C0 = lr.c_row(np.zeros((36, 2), dtype=complex))
        np.testing.assert_allclose(C0[:, 0], 1.0, rtol=0, atol=0)
        np.testing.assert_allclose(C0[:, 1:], 0.0, rtol=0, atol=0)

    def test_c_row_is_normalised_and_row_zero_is_real(self):
        # sum_n |C_kn|^2 = 1 -- the invariant the whole rotation rests on
        for m in (1, 2, 3):
            C = lr.c_row(fixed_v(36, m))
            np.testing.assert_allclose(np.sum(np.abs(C) ** 2, axis=1), 1.0,
                                       rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(C[:, 0].imag, 0.0, rtol=0, atol=0)

    def test_dc_row_dparams(self):
        for m in (1, 2, 3):
            v = fixed_v(36, m)
            a, b = lr.dc_row_dparams(v)
            c, d = legacy_ll.dc_row_dparams(v)
            np.testing.assert_allclose(a, c, rtol=0, atol=0)
            np.testing.assert_allclose(b, d, rtol=0, atol=0)

    def test_dc_row_dparams_is_the_derivative_of_c_row(self):
        # the SR gradient is analytic; this is the only external check that it
        # differentiates the function it claims to.  Only ROW k moves: t = |v_k|
        # depends on v_k alone, so a perturbation of v[k] must leave every other
        # momentum's row untouched -- asserted, since a formula that mixed rows
        # would still pass a slice-wise comparison.
        h = 1e-5
        v = fixed_v(36, 2, seed=11)
        dC_da, dC_db = lr.dc_row_dparams(v)
        for k in (0, 5, 17):
            for j in (0, 1):
                for shift, dC in ((h, dC_da), (1j * h, dC_db)):
                    vp = v.copy()
                    vp[k, j] += shift
                    vm = v.copy()
                    vm[k, j] -= shift
                    num = (lr.c_row(vp) - lr.c_row(vm)) / (2 * h)
                    self.assertLess(float(np.abs(num[k] - dC[k, j]).max()), 1e-9,
                                    f"row {k} param {j}")
                    self.assertEqual(float(np.abs(np.delete(num, k, axis=0)).max()),
                                     0.0, f"row {k} param {j} leaked into another row")

    def test_orbitals_pi_pi2(self):
        for r in POINTS:
            np.testing.assert_allclose(self.new.orbitals(r), self.old.orbitals(r),
                                       rtol=RTOL, atol=ATOL, err_msg=f"orb {r}")
            np.testing.assert_allclose(self.new.pi(r), self.old.pi(r),
                                       rtol=RTOL, atol=ATOL, err_msg=f"pi {r}")
            np.testing.assert_allclose(self.new.pi2(r), self.old.pi2(r),
                                       rtol=RTOL, atol=ATOL, err_msg=f"pi2 {r}")

    def test_basis_at_and_batch(self):
        R = np.random.default_rng(3).uniform(-3.0, 3.0, size=(NE, 2))
        np.testing.assert_allclose(self.new.batch(R), self.old.batch(R),
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(self.new.basis_at(POINTS[0]),
                                   self.old.basis_at(POINTS[0]), rtol=RTOL, atol=ATOL)

    def test_shapes(self):
        nk, nb = self.new.nk, self.new.nb
        self.assertEqual(self.new.C.shape, (nk, nb))
        self.assertEqual(self.new.orbitals(POINTS[0]).shape, (nk,))
        self.assertEqual(self.new.pi(POINTS[0]).shape, (2, nk))
        self.assertEqual(self.new.pi2(POINTS[0]).shape, (nk,))


class TestTheWavefunctionState(unittest.TestCase):
    """B4's machinery, exercised on the LL ansatz since that is where the
    override lives."""

    @classmethod
    def setUpClass(cls):
        cls.new, cls.old, cls.Ge, cls.v = setup(2)
        cls.C = np.column_stack([cls.Ge.A1, cls.Ge.A2])
        cls.Ci = np.linalg.inv(cls.C)
        cls.R = np.random.default_rng(5).uniform(-3.0, 3.0, size=(NE, 2))

    def _pair(self, seed_v=7, with_jastrow=True):
        orb_new, _, _, _ = setup(2, seed_v)
        Ge = self.Ge
        ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
        old_basis = legacy.LandauLevelBasis(Ge.mesh, 2, ai, ac, self.C, self.Ci)
        v = fixed_v(len(Ge.mesh), 1, seed_v)
        old_orb = legacy_ll.LLRotatedOrbitals(old_basis, 2, v)
        sc = np.column_stack([Ge.L1, Ge.L2])
        if with_jastrow:
            # the LL-rotation generator builds its Jastrow from the SUPERCELL
            # reciprocal vectors and gamma from |L1| (make_notebook_llrot.py:541),
            # i.e. the same convention as the Gaussian workflow -- not the
            # primitive-cell one, which is easy to assume from "LL" in the name
            c = np.array([0.2, -0.35, 0.15])
            gam = float(np.linalg.norm(Ge.L1) / np.sqrt(2) / np.pi * 32.0 / 3)
            Jn = jw.SinSplineJastrow(c, Ge.G1, Ge.G2, gam)
            Jo = legacy.SinSplineJastrow(c, Ge.G1, Ge.G2, gam)
        else:
            Jn = Jo = None
        wn = lr.LLRotationWavefunction(orb_new, Jn, NE, sc, kappa=32.0)
        wo = legacy_ll.LLRotationWavefunction(old_orb, Jo, NE, sc, kappa=32.0)
        return wn, wo

    def test_the_jastrow_half_of_the_state_is_bit_identical(self):
        wn, wo = self._pair()
        sn, so = wn.build(self.R), wo.build(self.R)
        for key in ("u", "gx", "gy", "lap"):
            self.assertTrue(np.array_equal(sn[key], so[key]), key)
        self.assertEqual(sn["U"], so["U"])

    def test_build_matches(self):
        """D to one ULP, and D_inv to what that ULP can buy.

        ``D`` is a 36x36 complex matrix (one row per momentum k, one column per
        electron) with condition number ~2.4e6 here, so a 1-ULP disagreement in
        ``D`` becomes ~1e-10 in ``D_inv`` no matter how the inverse is computed.
        ``n_band`` counts the LANDAU LEVELS mixed at each k, not the rows -- it
        is deliberately not the determinant size.  Asserting a flat tolerance on
        ``D_inv`` would
        either fail on this (harmless) amplification or have to be loosened so
        far that a genuine port error -- which shows up at O(1) -- would pass.
        So the assertion is the amplification bound itself, with the measured
        condition number, and it fails if the agreement is worse than the
        conditioning can explain.
        """
        wn, wo = self._pair()
        sn, so = wn.build(self.R), wo.build(self.R)
        self.assertEqual(sn["D"].shape, (wn.nk, NE))

        d_rel = float(np.abs(sn["D"] - so["D"]).max() / np.abs(so["D"]).max())
        self.assertLess(d_rel, 4 * np.finfo(float).eps * 100,
                        f"D agrees only to {d_rel:.3e} relative")

        cond = float(np.linalg.cond(so["D"]))
        bound = 10.0 * cond * d_rel
        inv_rel = float(np.abs(sn["D_inv"] - so["D_inv"]).max()
                        / np.abs(so["D_inv"]).max())
        self.assertLess(inv_rel, bound,
                        f"D_inv deviates {inv_rel:.3e}, beyond the {bound:.3e} "
                        f"that cond(D)={cond:.2e} times the D agreement allows")

    def test_build_without_jastrow_matches(self):
        wn, wo = self._pair(with_jastrow=False)
        sn, so = wn.build(self.R), wo.build(self.R)
        np.testing.assert_allclose(sn["D"], so["D"], rtol=RTOL, atol=ATOL)
        self.assertEqual(sn["U"], 0.0)
        np.testing.assert_allclose(sn["u"], 0.0, rtol=0, atol=0)

    def test_the_move_ratio_agrees_at_a_fresh_state(self):
        wn, wo = self._pair()
        sn, so = wn.build(self.R), wo.build(self.R)
        rng = np.random.default_rng(21)
        worst_acc = worst_ratio = 0.0
        for _ in range(12):
            i = int(rng.integers(NE))
            r_new = self.R[i] + rng.uniform(-0.5, 0.5, size=2)
            an, infon = wn.move_ratio(sn, i, r_new)
            ao, infoo = wo.move_ratio(so, i, r_new)
            worst_acc = max(worst_acc, abs(float(an) - float(ao)) / abs(float(ao)))
            sn_ratio = np.asarray(infon[0])
            so_ratio = np.asarray(infoo[0])
            worst_ratio = max(worst_ratio,
                              float(np.abs(sn_ratio - so_ratio).max()
                                    / np.abs(so_ratio).max()))
        self.assertLess(worst_acc, 1e-9, f"worst |Psi'/Psi|^2 dev {worst_acc:.3e}")
        self.assertLess(worst_ratio, 1e-9, f"worst Slater ratio dev {worst_ratio:.3e}")

    def test_the_accept_decisions_agree_along_a_chain(self):
        """The sampling claim, not the arithmetic one.

        Both walkers are driven with the SAME proposal stream; what has to match
        for the two to sample the same distribution is every accept/reject
        DECISION.  A ratio that is right to 1e-10 but lands the other side of
        the threshold would show up here and not in the test above.
        """
        wn, wo = self._pair()
        sn, so = wn.build(self.R), wo.build(self.R)
        rng = np.random.default_rng(88)
        accepted = 0
        for _ in range(60):
            i = int(rng.integers(NE))
            r_new = sn["R"][i] + rng.uniform(-0.6, 0.6, size=2)
            an, infon = wn.move_ratio(sn, i, r_new)
            ao, infoo = wo.move_ratio(so, i, r_new)
            if (float(an) > 1.0) != (float(ao) > 1.0):
                self.fail(f"acceptance decision differs at step {_}: "
                          f"{float(an):.17g} vs {float(ao):.17g}")
            if an > 1.0:
                accepted += 1
                wn.accept_move(sn, i, r_new, infon)
                wo.accept_move(so, i, r_new, infoo)
        self.assertGreater(accepted, 5, "chain accepted almost nothing; not a test")
        np.testing.assert_allclose(sn["R"], so["R"], rtol=0, atol=0)
        np.testing.assert_allclose(sn["u"], so["u"], rtol=0, atol=0)
        # D_inv has accumulated 60 Sherman-Morrison updates on top of a 1e-10
        # initial difference; it is the CONDITIONING that bounds it, as above.
        self.assertLess(float(np.abs(sn["D_inv"] - so["D_inv"]).max()
                              / np.abs(so["D_inv"]).max()), 1e-6)

    def test_rebuild_restores_the_inverse(self):
        wn, _ = self._pair()
        st = wn.build(self.R)
        rng = np.random.default_rng(4)
        for _ in range(30):
            i = int(rng.integers(NE))
            r_new = st["R"][i] + rng.uniform(-0.4, 0.4, size=2)
            acc, info = wn.move_ratio(st, i, r_new)
            if acc > 1e-12:
                wn.accept_move(st, i, r_new, info)
        wn.rebuild(st)
        np.testing.assert_allclose(st["D_inv"], np.linalg.inv(st["D"]),
                                   rtol=1e-11, atol=1e-11)
        # and the rebuild recomputed the batch from R, so it is the batch of R
        self.assertTrue(np.array_equal(st["Ops"], wn.orb.batch(st["R"])))


class TestNesting(unittest.TestCase):
    """``(c_0, c_1) -> (c_0, c_1, 0)``, to machine precision."""

    def test_the_identity_is_exact(self):
        # the statement the nmax ladder depends on.  Not "< 1e-12" by accident:
        # c_row computes the identical floating-point expression for the head
        # (same norm, same leading entries) and an exact zero for the tail.
        for m in (1, 2, 3):
            v = fixed_v(36, m, seed=100 + m)
            head, tail = nest.coefficient_identity_deviation(v, m + 1)
            self.assertEqual(head, 0.0, f"head deviation {head:.3e} at m={m}")
            self.assertEqual(tail, 0.0, f"tail deviation {tail:.3e} at m={m}")

    def test_padding_is_on_the_right(self):
        v = fixed_v(36, 2, seed=9)
        p = nest.pad_v(v, 4)
        self.assertEqual(p.shape, (36, 3))
        np.testing.assert_allclose(p[:, :2], v, rtol=0, atol=0)
        np.testing.assert_allclose(p[:, 2], 0.0, rtol=0, atol=0)

    def test_padding_is_a_no_op_when_already_the_right_size(self):
        v = fixed_v(36, 2, seed=9)
        np.testing.assert_allclose(nest.pad_v(v, 3), v, rtol=0, atol=0)

    def test_the_nested_state_is_the_same_wavefunction(self):
        # the coefficient identity is the algebra; this is the physics it means.
        # Same v, one extra band, and the ORBITALS must be identical -- which is
        # what makes an energy rise on the ladder a statement about the optimiser.
        Ge = geom()
        ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
        C = np.column_stack([Ge.A1, Ge.A2])
        Ci = np.linalg.inv(C)
        v2 = fixed_v(len(Ge.mesh), 1, seed=13)
        v3 = nest.pad_v(v2, 3)
        b2 = ll.LandauLevelBasis(Ge.mesh, 1, ai, ac, C, Ci)
        b3 = ll.LandauLevelBasis(Ge.mesh, 2, ai, ac, C, Ci)
        o2 = lr.LLRotatedOrbitals(b2, 2, v2)
        o3 = lr.LLRotatedOrbitals(b3, 3, v3)
        for r in POINTS:
            np.testing.assert_allclose(o3.orbitals(r), o2.orbitals(r),
                                       rtol=0, atol=0, err_msg=f"psi {r}")
            np.testing.assert_allclose(o3.pi(r), o2.pi(r), rtol=0, atol=0)
            np.testing.assert_allclose(o3.pi2(r), o2.pi2(r), rtol=0, atol=0)

    def test_nesting_rejects_an_oversized_source(self):
        with self.assertRaises(ValueError):
            nest.pad_v(fixed_v(36, 3), 3)

    def test_v_zero_is_the_liquid(self):
        # U = I is the filled LLL band 0, in both the clean and the legacy spelling
        Ge = geom()
        ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
        C = np.column_stack([Ge.A1, Ge.A2])
        Ci = np.linalg.inv(C)
        b = ll.LandauLevelBasis(Ge.mesh, 0, ai, ac, C, Ci)
        orb = lr.LLRotatedOrbitals(b, 1, np.zeros((36, 0), dtype=complex))
        for r in POINTS:
            np.testing.assert_allclose(orb.orbitals(r), b.orbitals(r)[:, 0],
                                       rtol=0, atol=0)


class TestGaugeSign(unittest.TestCase):
    def test_pi_psi_is_the_covariant_derivative(self):
        orb, _, _, _ = setup(2)
        Ge = geom()
        h = 1e-5
        for r in POINTS:
            p = orb.pi(r)
            dx = (orb.orbitals(r + [h, 0]) - orb.orbitals(r - [h, 0])) / (2 * h)
            dy = (orb.orbitals(r + [0, h]) - orb.orbitals(r - [0, h])) / (2 * h)
            PX = -1j * dx + (r[1] / 2.0) * orb.orbitals(r)
            PY = -1j * dy - (r[0] / 2.0) * orb.orbitals(r)
            self.assertLess(float(np.abs(PX - p[0]).max()), 5e-8, f"pi_x r={r}")
            self.assertLess(float(np.abs(PY - p[1]).max()), 5e-8, f"pi_y r={r}")
        del Ge

    def test_pi_squared_is_diagonal_weighted_by_C(self):
        # pi^2 psi_k = sum_n C_kn (2n+1) phi_kn -- the property that makes the
        # rotation exact in the kinetic energy.  Checked without legacy.
        orb, _, _, _ = setup(3)
        for r in POINTS:
            expected = np.einsum("kn,kn->k", orb.C,
                                 orb.basis_at(r) * (2 * np.arange(orb.nb) + 1))
            np.testing.assert_allclose(orb.pi2(r), expected, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
