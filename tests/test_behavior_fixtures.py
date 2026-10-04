"""Stage 2C: the golden fixtures, rebuilt from their own inputs.

Where the B1-B7 regressions ask "does clean agree with legacy on this input?",
these fixtures ask the complementary question: "does clean still agree with
ITSELF?".  Five small JSON files under ``fixtures/behavior/`` carry a frozen
input -- a configuration R, band coefficients v, and the model parameters -- and
the observables the accepted code returned for it.  The test rebuilds the ansatz
from the stored input and re-derives those observables.

That is what makes the fixture a *behavior lock* rather than a copy of an answer:
the input is the payload, and a fixture that has drifted is re-derived, not
compared against a remembered number.  ``fixtures/behavior/_build.py``
regenerates them, and is deliberately not run by the suite.

What this test is, and is not
-----------------------------
It is a **snapshot**: the expected values were produced by the clean code itself.
So on its own it cannot show that clean matches legacy -- that is the job of the
B1-B7 regressions, which rebuild the same recipes against the frozen engine.  Its
job is the other half: the moment a Stage 2D refactor changes Gaussian or LL
behaviour, this fails in a second, with the legacy tree absent if necessary.  The
two together pin legacy behaviour without this file ever importing the engine.

No Monte Carlo and no RNG: every input is a fixed array on disk.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SRC = os.path.join(CLEAN, "src")
FIX = os.path.join(HERE, "fixtures", "behavior")
for p in (SRC, FIX):
    if p not in sys.path:
        sys.path.insert(0, p)

import recipe as R                                                # noqa: E402
from wigner_vmc.wavefunctions import nesting as nest              # noqa: E402

#: The rebuild is bit-identical in practice -- measured 0.0 on every observable
#: of every fixture, across separate processes, after the JSON round trip
#: (Python's float repr is round-trip exact, and the arithmetic is deterministic).
#: The tolerance is NOT zero because the determinant inversion goes through
#: LAPACK: a different BLAS or thread count may reassociate `D_inv` and move the
#: last bits.  Nothing here is being tested at the 1e-12 level; the bound is
#: there so a legitimate library difference is not reported as a physics change.
RTOL = 1e-12
ATOL = 1e-13

NAMES = ("gaussian_regular", "gaussian_boundary",
         "ll_nmax1", "ll_nmax2", "nested_nmax1_to_2")

#: nx2 complex observables, stored as [re, im] pairs.
COMPLEX_KEYS = ("logdet_D", "T", "V", "E_local")


def rebuild(fx, Ge):
    """Reconstruct the ansatz named by a fixture and its configuration."""
    Rcfg = R.as_R(fx["inputs"]["R"])
    if fx["ansatz"] == "gaussian":
        wf = R.gaussian_wf(Ge)
    elif fx["ansatz"] == "ll_rotation":
        wf = R.ll_wf(Ge, R.as_c(fx["inputs"]["v"]))
    else:
        raise AssertionError(f"unknown ansatz {fx['ansatz']!r}")
    return wf, Rcfg, R.observe(wf, Rcfg)


class TestGoldenFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Ge = R.geometry()
        cls.fx = {n: R.load(n) for n in NAMES}

    def test_every_fixture_rebuilds(self):
        for name in NAMES:
            with self.subTest(fixture=name):
                fx = self.fx[name]
                _, _, got = rebuild(fx, self.Ge)
                exp = fx["expected"]
                self.assertEqual(sorted(got), sorted(exp),
                                 "observable list drifted from the fixture")
                for k in COMPLEX_KEYS:
                    np.testing.assert_allclose(R.as_c(got[k]), R.as_c(exp[k]),
                                               rtol=RTOL, atol=ATOL, err_msg=k)
                np.testing.assert_allclose(got["U"], exp["U"],
                                           rtol=RTOL, atol=ATOL, err_msg="U")

    def test_the_rebuild_reproduces_a_nonzero_determinant(self):
        # a guard on the guard: if `observe` ever returned a constant, every
        # comparison above would pass against a fixture of that same constant.
        for name in NAMES:
            with self.subTest(fixture=name):
                ld = R.as_c(self.fx[name]["expected"]["logdet_D"])
                self.assertGreater(abs(ld), 1.0)
                self.assertGreater(abs(R.as_c(
                    self.fx[name]["expected"]["E_local"])), 1.0)

    def test_the_five_fixtures_are_not_the_same_question(self):
        # distinctness, so that a fixture accidentally built from the wrong
        # recipe is caught here rather than at the next refactor.
        e = {n: R.as_c(self.fx[n]["expected"]["E_local"]) for n in NAMES}
        pairs = [("gaussian_regular", "gaussian_boundary"),
                 ("gaussian_boundary", "ll_nmax1"),
                 ("ll_nmax1", "ll_nmax2")]
        for a, b in pairs:
            self.assertGreater(abs(e[a] - e[b]), 1e-3, f"{a} vs {b}")


class TestNestingInvariant(unittest.TestCase):
    """``(c_0, c_1) -> (c_0, c_1, 0)``, carried through to the local energy.

    This is the firewall's statement of the invariant the whole ``n_max`` ladder
    rests on: adding an empty band must not change the wavefunction.  Layer C's
    coefficient-identity test proves the algebra exactly; this proves the same
    thing about a real ansatz, on a real configuration, through the energies.
    """

    @classmethod
    def setUpClass(cls):
        cls.Ge = R.geometry()
        cls.fx = {n: R.load(n) for n in NAMES}

    def _rebuilt(self, name):
        fx = self.fx[name]
        return rebuild(fx, self.Ge)[2]

    def test_the_padding_is_a_zero_column(self):
        v1 = R.as_c(self.fx["ll_nmax1"]["inputs"]["v"])
        vn = R.as_c(self.fx["nested_nmax1_to_2"]["inputs"]["v"])
        self.assertEqual(vn.shape, (v1.shape[0], v1.shape[1] + 1))
        np.testing.assert_array_equal(vn[:, :v1.shape[1]], v1)
        np.testing.assert_array_equal(vn[:, -1], 0.0)

    def test_the_nested_ansatz_has_the_same_observables(self):
        a = self._rebuilt("ll_nmax1")
        b = self._rebuilt("nested_nmax1_to_2")
        for k in COMPLEX_KEYS:
            np.testing.assert_allclose(R.as_c(a[k]), R.as_c(b[k]),
                                       rtol=RTOL, atol=ATOL, err_msg=k)
        np.testing.assert_allclose(a["U"], b["U"], rtol=RTOL, atol=ATOL)

    def test_the_band_counts_differ_so_the_match_is_not_trivial(self):
        # the two fixtures really are different ansatz sizes -- otherwise the
        # identity above would hold for the boring reason.
        self.assertEqual(self.fx["ll_nmax1"]["model"]["n_bands"], 2)
        self.assertEqual(self.fx["nested_nmax1_to_2"]["model"]["n_bands"], 3)


class TestNestingLadder(unittest.TestCase):
    """The same invariant on all three rungs the ladder is actually used on.

    ``TestNesting.test_the_identity_is_exact`` proves the coefficient identity on
    1->2, 2->3 and 3->4; this asks the question of the ASSEMBLED orbitals and of a
    real configuration's energies, at each rung.  The orbital comparison is
    exact rather than tolerant because ``c_row`` evaluates the identical
    floating-point expression for the head and an exact zero for the tail -- the
    same reason layer C's coefficient test asserts ``== 0.0``.
    """

    POINTS = (np.array([0.31, -0.77]), np.array([1.10, 0.40]),
              np.array([0.0, 0.0]))

    @classmethod
    def setUpClass(cls):
        cls.Ge = R.geometry()
        cls.nk = len(cls.Ge.mesh)
        cls.cfg = R.cell_points(cls.Ge, R.NE, 0.05, 0.95, R.SEED_GAUSSIAN)

    def _rung(self, m):
        v = R.ll_v(self.nk, m, R.SEED_LL + m)
        return v, nest.pad_v(v, m + 2)

    def test_each_rung_leaves_the_orbitals_alone(self):
        for m in (1, 2, 3):
            with self.subTest(rung=f"{m}->{m + 1}"):
                v, vp = self._rung(m)
                o, op = R.ll_orb(self.Ge, v), R.ll_orb(self.Ge, vp)
                self.assertEqual(op.nb, o.nb + 1)
                for r in self.POINTS:
                    np.testing.assert_array_equal(op.orbitals(r), o.orbitals(r))
                    np.testing.assert_array_equal(op.pi(r), o.pi(r))
                    np.testing.assert_array_equal(op.pi2(r), o.pi2(r))

    def test_each_rung_leaves_the_observables_alone(self):
        for m in (1, 2, 3):
            with self.subTest(rung=f"{m}->{m + 1}"):
                v, vp = self._rung(m)
                a = R.observe(R.ll_wf(self.Ge, v), self.cfg)
                b = R.observe(R.ll_wf(self.Ge, vp), self.cfg)
                for k in COMPLEX_KEYS + ("U",):
                    np.testing.assert_allclose(a[k], b[k], rtol=RTOL, atol=ATOL,
                                               err_msg=k)


class TestTheEnergyConvention(unittest.TestCase):
    """``E_local = T + kappa * V``, on the same fixtures.

    The Gaussian and LL local energies are separate implementations, and this is
    the one relation both must satisfy.  `V` is the BARE ``1/r`` sum: with kappa
    folded into V or into T, ``E_local`` is unchanged, which is why the pieces are
    checked rather than only their sum.
    """

    @classmethod
    def setUpClass(cls):
        cls.Ge = R.geometry()
        cls.fx = {n: R.load(n) for n in NAMES}

    def test_the_relation_holds_on_every_fixture(self):
        for name in NAMES:
            with self.subTest(fixture=name):
                fx = self.fx[name]
                _, _, got = rebuild(fx, self.Ge)
                kap = float(fx["model"]["kappa"])
                T, V, E = (R.as_c(got[k]) for k in ("T", "V", "E_local"))
                np.testing.assert_allclose(E, T + kap * V,
                                           rtol=RTOL, atol=ATOL)
                # and V is not E: the coupling is large, so the two differ.
                self.assertGreater(abs(E - V), 1.0)

    def test_the_fixtures_record_the_locked_coupling_relation(self):
        # kappa = r_s / sqrt(2), carried into the fixture rather than assumed,
        # so a fixture built at the wrong coupling cannot silently agree.
        for name in NAMES:
            with self.subTest(fixture=name):
                m = self.fx[name]["model"]
                self.assertAlmostEqual(float(m["kappa"]),
                                       float(m["rs"]) / np.sqrt(2.0), places=12)


class TestTheFixtureSetItself(unittest.TestCase):
    """Guards on the fixtures as data, not on the physics they record."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = R.geometry()
        cls.fx = {n: R.load(n) for n in NAMES}

    def test_every_fixture_records_inputs_not_just_outputs(self):
        for name in NAMES:
            with self.subTest(fixture=name):
                fx = self.fx[name]
                self.assertEqual(R.as_R(fx["inputs"]["R"]).shape, (R.NE, 2))
                if fx["ansatz"] == "ll_rotation":
                    v = R.as_c(fx["inputs"]["v"])
                    self.assertEqual(v.shape[0], len(self.Ge.mesh))
                    self.assertEqual(v.shape[1], fx["model"]["n_bands"] - 1)
                else:
                    self.assertIsNone(fx["inputs"]["v"])
                for key in ("name", "layer", "legacy_source", "model",
                            "inputs", "expected"):
                    self.assertIn(key, fx)

    def test_the_boundary_fixture_really_leaves_the_cell(self):
        # `gaussian_boundary` exists to exercise the wrapping path; if its
        # points all sat inside the supercell it would be a duplicate of
        # `gaussian_regular` wearing a different name.
        sc = np.column_stack([self.Ge.L1, self.Ge.L2])
        inv = np.linalg.inv(sc)
        frac = R.as_R(self.fx["gaussian_boundary"]["inputs"]["R"]) @ inv.T
        self.assertTrue(np.any((frac < 0.0) | (frac >= 1.0)),
                        "no boundary point: the fixture is not testing wrapping")
        frac_in = R.as_R(self.fx["gaussian_regular"]["inputs"]["R"]) @ inv.T
        self.assertTrue(np.all((frac_in >= 0.0) & (frac_in < 1.0)),
                        "interior fixture has a point outside the cell")

    def test_the_fixture_names_are_the_documented_ones(self):
        on_disk = {f[:-5] for f in os.listdir(FIX) if f.endswith(".json")}
        self.assertEqual(on_disk, set(NAMES))


if __name__ == "__main__":
    unittest.main()
