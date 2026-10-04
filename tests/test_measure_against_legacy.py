"""B6: the production measurement path against the frozen legacy source.

The legacy implementations of everything tested here are **emitted cell text**,
not engine symbols: `make_notebook.py` has exactly four module-level definitions
(`_src`, `_push`, `md`, `code`) and `measure`/`sq_error_bars`/`tau_int`/`blocked`
are not among them.  There is therefore nothing to import, and the only way to
test against the real legacy implementation rather than against my transcription
of it is to read the text out of the frozen generator.  `_extract` below does
that; if the frozen file is absent the tests skip rather than silently pass.

The three energy semantics this file exists to keep apart
---------------------------------------------------------
``E_optimization_min`` and ``E_optimization_final`` come from the SR history and
are **totals**; ``E_production`` comes from `measure` and is **per electron**.
The two differ by a factor of `ne`.  A phase comparison may use only the third,
and only from a measurement that is independent of the optimisation.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
SRC = os.path.join(CLEAN, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import qhvmc_engine as legacy                                    # noqa: E402
import qhvmc_engine_llrot as legacy_ll                           # noqa: E402
from wigner_vmc.analysis import statistics as stats               # noqa: E402
from wigner_vmc.analysis import structure as st                   # noqa: E402
from wigner_vmc.physics import coulomb as cb                      # noqa: E402
from wigner_vmc.physics import geometry as g                      # noqa: E402
from wigner_vmc.physics import landau_levels as ll                # noqa: E402
from wigner_vmc.vmc import measure as measure_mod                 # noqa: E402
from wigner_vmc.wavefunctions import gaussian as gauss            # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw                # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr            # noqa: E402
from wigner_vmc.wavefunctions import slater as sl                 # noqa: E402

GEN = os.path.join(ROOT, "make_notebook.py")
GEN_LL = os.path.join(ROOT, "reproduction", "make_notebook_llrot.py")

PRIM_AREA = 2.0 * np.pi
NE = 36
LUMAX = 30.0
LUMAX_LL = 8.5 * np.sqrt(4 * np.pi / np.sqrt(3))
KAPPA = 32.0
L0 = 0.6
C5 = np.array([0.30, -0.55, 0.42, -0.18, 0.07])
N_BANDS = 2

NSWEEP, EQUIL, SNAP_EVERY, SEED = 40, 15, 4, 424242


def _extract(path, name):
    """The text of a top-level ``def name(...)`` out of a generator's cell source.

    The generators hold their cell source as string literals, so this is a text
    scan, not an import.  Consume the def line and every following line that is
    blank or indented, stopping at the first non-blank line at column 0.
    """
    if not os.path.exists(path):
        raise unittest.SkipTest(f"frozen generator absent: {path}")
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith(f"def {name}(")), None)
    if start is None:
        raise unittest.SkipTest(f"def {name}( not found in {path}")
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0].isspace()):
        end += 1
    return "\n".join(lines[start:end])


def legacy_measure():
    """The frozen `measure`, exec'd with the FROZEN engine's `sample`."""
    ns = {"np": np, "ne": NE, "sample": legacy.sample}
    exec(_extract(GEN, "measure"), ns)
    return ns["measure"]


def legacy_sq_error_bars():
    ns = {"np": np}
    exec(_extract(GEN, "sq_error_bars"), ns)
    return ns["sq_error_bars"]


def legacy_bragg_ratio():
    """`_bragg_ratio` with the engine's `structure_factor` and its own globals."""
    ns = {"np": np}
    exec(_extract(GEN_LL, "_bragg_ratio"), ns)
    # _bragg and _bgq are notebook globals; supply them from the same source the
    # clean side gets them from, so the comparison is of the ESTIMATOR only.
    T = g.Geometry.from_cell(*g.triangular_cell(PRIM_AREA), 6, 6, 4.0)
    ns["_bragg"] = six_bragg(T)
    ns["_bgq"] = background_q(T)
    ns["structure_factor"] = legacy.structure_factor
    return ns["_bragg_ratio"]


def six_bragg(T):
    """The six first-shell (WC) reciprocal vectors.

    Transcribed from ``reproduction/make_notebook_llrot.py:1397-1399``: build
    {m g1 + n g2} over a 7x7 window and keep the vectors whose magnitude IS |g1|,
    then lexsort.  The magnitude test is exact rather than a tolerance because
    the peaks are members of the allowed set by construction.
    """
    g1, g2 = np.asarray(T.g1, float), np.asarray(T.g2, float)
    mag = float(np.linalg.norm(g1))
    allq = np.array([m * g1 + n * g2 for m in range(-3, 4) for n in range(-3, 4)])
    keep = np.abs(np.linalg.norm(allq, axis=1) - mag) < 1e-9 * mag
    allq = allq[keep]
    return allq[np.lexsort((allq[:, 1], allq[:, 0]))]


def background_q(T):
    """The legacy `_bgq`: supercell momenta with 0.75|g1| < |q| < 1.25|g1|, minus the peaks.

    Transcribed from ``reproduction/make_notebook_llrot.py:1401-1407``.
    """
    G1, G2 = np.asarray(T.G1, float), np.asarray(T.G2, float)
    mag_G1 = float(np.linalg.norm(G1))
    mag_g1 = float(np.linalg.norm(T.g1))
    bragg = six_bragg(T)
    MB = int(np.ceil(1.3 * mag_g1 / mag_G1))
    bg_all = np.array([m * G1 + n * G2
                       for m in range(-MB, MB + 1) for n in range(-MB, MB + 1)])
    n = np.linalg.norm(bg_all, axis=1)
    sel = (n > 0.75 * mag_g1) & (n < 1.25 * mag_g1)
    sel &= np.array([np.min(np.linalg.norm(bragg - q, axis=1)) > 1e-6
                     for q in bg_all])
    return bg_all[sel]


def geom():
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, 4.0)


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
    v = (rng.normal(size=(len(Ge.mesh), n_band - 1))
         + 1j * rng.normal(size=(len(Ge.mesh), n_band - 1)))
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
    return wn, wo


def start_config(Ge, seed=3):
    return (np.random.default_rng(seed).random((NE, 2)) - 0.5) @ np.column_stack(
        [Ge.L1, Ge.L2]).T


class TestTheExtractorItself(unittest.TestCase):
    """A text scan that silently returned the wrong thing would make every
    comparison below vacuous, so it is checked before it is trusted."""

    def test_it_returns_the_whole_function_and_it_compiles(self):
        for path, name in ((GEN, "measure"), (GEN, "sq_error_bars"),
                           (GEN, "tau_int"), (GEN, "blocked"),
                           (GEN_LL, "_bragg_ratio")):
            text = _extract(path, name)
            self.assertTrue(text.startswith(f"def {name}("), name)
            compile(text, f"<{name}>", "exec")
            self.assertIn("return", text, f"{name} has no return; body truncated")

    def test_measure_is_the_same_object_for_both_workflows(self):
        # the LL notebook has no `def measure` of its own -- it calls the
        # Gaussian notebook's.  If a second definition ever appears, the two
        # workflows would have silently diverged and this file would only be
        # testing one of them.
        with open(GEN_LL, encoding="utf-8") as fh:
            lines = fh.readlines()
        self.assertIsNone(
            next((i for i, ln in enumerate(lines)
                  if ln.startswith("def measure(")), None),
            "the LL generator now defines its own measure(); the shared-path "
            "assumption behind this test file no longer holds")


class TestMeasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        # staticmethod, NOT the bare function: a plain function stored on the
        # class is a DESCRIPTOR, so `self.old_measure(wf, R0, ...)` binds self
        # as the first argument and the call fails with "multiple values for
        # argument 'nsweep'" rather than anything that points at the real cause.
        cls.old_measure = staticmethod(legacy_measure())
        cls.R0 = start_config(cls.Ge)

    def _both(self, new_wf, old_wf):
        a = measure_mod.measure(new_wf, self.R0, nsweep=NSWEEP, sigma=0.3,
                                seed=SEED, snapshot_every=SNAP_EVERY,
                                equil=EQUIL)
        b = self.old_measure(old_wf, self.R0, nsweep=NSWEEP, sigma=0.3,
                             seed=SEED, snapshot_every=SNAP_EVERY, equil=EQUIL)
        return a, b

    def _assert_identical(self, a, b, tag, energy_rtol=0.0):
        """`energy_rtol=0` means bit identity; a nonzero value is a bound the
        caller has to justify from a measurement made elsewhere."""
        self.assertEqual(a["n"], b["n"], tag)
        self.assertEqual(a["n"], 7, "the fixed walk must yield 7 snapshots")
        # The walk itself must be the SAME walk, not two runs that happen to
        # agree on a summary -- the sampler is bit-identical (B4) in both
        # workflows, so the snapshots are compared exactly either way.
        for k in ("acc", "sigma"):
            self.assertEqual(float(a[k]), float(b[k]), f"{k} {tag}")
        np.testing.assert_array_equal(np.asarray(a["snaps"]), np.asarray(b["snaps"]),
                                      err_msg=f"snaps {tag}")
        self.assertEqual(set(a), set(b), f"record keys {tag}")
        # compared without coercion: `V`/`V_err` are complex (see the test
        # below), and casting them to float would both warn and hide the fact.
        for k in ("T", "T_err", "V", "V_err", "E", "E_err"):
            if energy_rtol == 0.0:
                self.assertEqual(a[k], b[k], f"{k} {tag}: {a[k]!r} != {b[k]!r}")
            else:
                np.testing.assert_allclose(a[k], b[k], rtol=energy_rtol, atol=0,
                                           err_msg=f"{k} {tag}")

    def test_gaussian_workflow(self):
        new_wf, old_wf = gaussian_pair(self.Ge)
        a, b = self._both(new_wf, old_wf)
        self._assert_identical(a, b, "gaussian")

    def test_ll_rotation_workflow(self):
        new_wf, old_wf = ll_pair(self.Ge)
        a, b = self._both(new_wf, old_wf)
        # NOT bit-identical, and the reason is not in this subsystem: B5
        # measured the LL local energy itself differing by 1.0e-14 relative
        # (max abs 2.2e-12), inherited from the two Landau-level bases summing
        # their lattice vectors in a different order.  `measure` only averages
        # that number, so it inherits it.  The bound is set two orders above
        # that measured effect and six orders below anything a real migration
        # error would produce -- and the record fields that do NOT come from
        # local_energy are still compared exactly, above.
        self._assert_identical(a, b, "ll", energy_rtol=1e-12)

    def test_the_record_fields_downstream_consumers_read(self):
        new_wf, _ = gaussian_pair(self.Ge)
        out = measure_mod.measure(new_wf, self.R0, nsweep=NSWEEP, sigma=0.3,
                                  seed=SEED, snapshot_every=SNAP_EVERY, equil=EQUIL)
        self.assertEqual(sorted(out), ["E", "E_err", "T", "T_err", "V", "V_err",
                                       "acc", "n", "sigma", "snaps"])
        self.assertEqual(np.shape(out["snaps"]), (7, NE, 2))

    def test_every_energy_field_is_per_electron(self):
        """The factor of `ne` between this record and the SR history is the most
        reachable error in the whole path, so the conversion is asserted against
        the total it came from rather than taken on trust."""
        new_wf, _ = gaussian_pair(self.Ge)
        out = measure_mod.measure(new_wf, self.R0, nsweep=NSWEEP, sigma=0.3,
                                  seed=SEED, snapshot_every=SNAP_EVERY, equil=EQUIL)
        res = np.array([new_wf.local_energy(new_wf.build(R)) for R in out["snaps"]])
        n = len(out["snaps"])
        np.testing.assert_allclose(out["E"] * NE, res[:, 2].real.mean(), rtol=1e-15)
        np.testing.assert_allclose(out["T"] * NE, res[:, 0].real.mean(), rtol=1e-15)
        np.testing.assert_allclose(out["V"] * NE, res[:, 1].mean(), rtol=1e-15)
        np.testing.assert_allclose(out["E_err"] * NE,
                                   res[:, 2].real.std() / np.sqrt(n), rtol=1e-15)
        # and the total is roughly ne times the per-electron value
        self.assertGreater(abs(float(res[:, 2].real.mean() / out["E"])), 30.0)

    def test_V_comes_back_complex_and_V_err_does_not(self):
        """Legacy behaviour, reproduced rather than tidied up (BUG_CANDIDATE #4).

        `res` is built from `local_energy`, which returns a complex `T`, so the
        whole (n, 3) array is complex; `measure` then takes `res[:, 1].mean()`
        WITHOUT `.real` -- unlike `T` and `E`, which have it.  `V` therefore
        comes back complex, and it is *printed* that way: cell 18 of the
        delivered `WignerCrystal_to_HallLiquid.ipynb` shows `-0.7559+0.0000j`
        in the V column of its scan table.

        `V_err` does NOT, and the asymmetry is numpy's rather than the
        notebook's: `ndarray.std()` on a complex array returns a real, because
        it reduces |x - mean|^2 rather than (x - mean)^2.  So exactly one of the
        two V fields is complex, and a reader who assumed they matched would be
        wrong in a way no test here would otherwise catch.
        """
        new_wf, _ = gaussian_pair(self.Ge)
        out = measure_mod.measure(new_wf, self.R0, nsweep=NSWEEP, sigma=0.3,
                                  seed=SEED, snapshot_every=SNAP_EVERY, equil=EQUIL)
        self.assertIsInstance(out["V"], complex)
        self.assertEqual(complex(out["V"]).imag, 0.0)
        self.assertNotIsInstance(out["V_err"], complex)
        for k in ("T", "T_err", "E", "E_err"):
            self.assertNotIsInstance(out[k], complex, k)
        # the formatted form is the one that appears in the published output, so
        # round-trip through that exact string rather than reconstructing it
        s = f"{out['V']:9.4f}"
        self.assertTrue(s.endswith("+0.0000j"), s)
        self.assertEqual(complex(s).imag, 0.0)
        self.assertAlmostEqual(complex(s).real, complex(out["V"]).real, places=4)

    def test_E_err_is_the_plain_spread_not_the_autocorrelation_corrected_one(self):
        """`measure` uses std/sqrt(n) and nothing else.

        The corrected estimates live in `analysis.statistics` and are NOT applied
        here.  The two must not be conflated -- on this walk they differ by a
        measurable factor, and the naive one is the smaller of the two.
        """
        new_wf, _ = gaussian_pair(self.Ge)
        out = measure_mod.measure(new_wf, self.R0, nsweep=200, sigma=0.3,
                                  seed=SEED, snapshot_every=1, equil=40)
        res = np.array([new_wf.local_energy(new_wf.build(R)) for R in out["snaps"]])
        tot = res[:, 2].real
        naive_total = tot.std() / np.sqrt(tot.size)
        blocked_total = max(stats.blocked(tot, B) for B in (2, 4, 8, 16)
                            if tot.size // B >= 4)
        tau, _win = stats.tau_int(tot)
        self.assertEqual(tot.size, out["n"])
        self.assertGreater(tau, 1.0, "no autocorrelation was measured; the test is vacuous")
        self.assertNotAlmostEqual(naive_total, blocked_total, places=12)
        self.assertLess(naive_total, blocked_total,
                        "the uncorrected error must be the optimistic one")
        # measure's own total-unit error IS the naive one, exactly
        np.testing.assert_allclose(out["E_err"] * NE, naive_total, rtol=1e-15)


class TestSqErrorBars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.old = staticmethod(legacy_sq_error_bars())
        cls.new = staticmethod(st.sq_error_bars)
        Ge = geom()
        sc = np.column_stack([Ge.L1, Ge.L2])
        rng = np.random.default_rng(11)
        cls.snaps = [rng.uniform(0, 1, size=(NE, 2)) @ sc.T for _ in range(9)]
        cls.q = six_bragg(Ge)[0]

    def test_matches_the_frozen_text(self):
        q = self.q
        for snaps in (self.snaps, self.snaps[:1]):
            a = self.new(snaps, q, NE)
            b = self.old(snaps, q, NE)
            self.assertEqual(float(a[0]), float(b[0]))
            self.assertEqual(float(a[1]), float(b[1]))

    def test_it_is_snapshot_sem_with_ddof_zero(self):
        # snapshot_sem reduces over axis 0, so the per-snapshot values have to
        # be shaped (n_snaps, 1) for a single q
        per = st.structure_factor_snapshots(self.snaps, [self.q])
        a = self.new(self.snaps, self.q, NE)
        np.testing.assert_allclose(a[0], per[:, 0].mean(), rtol=1e-15)
        np.testing.assert_allclose(a[1], st.snapshot_sem(per, ddof=0)[0], rtol=1e-15)
        self.assertNotAlmostEqual(float(st.snapshot_sem(per, ddof=1)[0]),
                                  float(a[1]), places=15)

    def test_a_perfect_lattice_puts_the_peak_at_ne(self):
        # the estimator's own control: sites ON the reciprocal vectors give
        # |rho_q|^2/ne = ne exactly, so a broken estimator cannot pass this
        Ge = geom()
        sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
        for q in six_bragg(Ge):
            mean, sem = self.new([sites], q, NE)
            np.testing.assert_allclose(mean, NE, rtol=1e-12)
            self.assertEqual(sem, 0.0)


class TestBraggRatio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.old = staticmethod(legacy_bragg_ratio())
        cls.bragg, cls.bg = six_bragg(cls.Ge), background_q(cls.Ge)
        sites = g.wigner_crystal_sites(6, 6, cls.Ge.A1, cls.Ge.A2)
        sc = np.column_stack([cls.Ge.L1, cls.Ge.L2])
        uniform = (np.random.default_rng(0).random((NE, 2)) - 0.5) @ sc.T
        cls.controls = {"exact triangular lattice": [sites],
                        "uniform random": [uniform]}

    def test_it_sees_the_legacy_globals_the_notebook_had(self):
        # 6 first-shell peaks and 120 background momenta is what the legacy
        # comment states; if these differ the comparison below is of two
        # different measurements
        self.assertEqual(len(self.bragg), 6)
        self.assertEqual(len(self.bg), 120)

    def test_matches_the_frozen_text_on_the_controls(self):
        for name, snaps in self.controls.items():
            a = st.bragg_ratio(snaps, self.bragg, self.bg, NE)
            b = self.old(snaps, NE)
            for x, y, lab in zip(a, b, ("peak", "background", "ratio")):
                self.assertEqual(float(x), float(y), f"{lab} {name}")

    def test_the_lattice_control_is_a_sharp_peak_and_the_random_one_is_not(self):
        lat = st.bragg_ratio(self.controls["exact triangular lattice"],
                             self.bragg, self.bg, NE)
        rnd = st.bragg_ratio(self.controls["uniform random"],
                             self.bragg, self.bg, NE)
        np.testing.assert_allclose(lat[0], NE, rtol=1e-12)
        self.assertGreater(lat[2] / rnd[2], 100.0,
                           "the estimator must separate order from disorder")

    def test_the_background_is_a_median_not_a_mean(self):
        snaps = self.controls["uniform random"]
        S_bg = np.real(st.structure_factor(snaps, self.bg, NE))
        _pk, bk, _r = st.bragg_ratio(snaps, self.bragg, self.bg, NE)
        self.assertEqual(bk, float(np.median(S_bg)))
        self.assertNotAlmostEqual(bk, float(S_bg.mean()), places=6)


class TestDensityGrid(unittest.TestCase):
    """`density_grid` was migrated in Stage 1 but never compared to the engine."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        sc = np.column_stack([cls.Ge.L1, cls.Ge.L2])
        rng = np.random.default_rng(21)
        cls.snaps = [rng.uniform(0, 1, size=(NE, 2)) @ sc.T for _ in range(4)]
        cls.torus = st.Torus(abs(cls.Ge.L1[0] * cls.Ge.L2[1]
                                 - cls.Ge.L1[1] * cls.Ge.L2[0]))

    @staticmethod
    def _binidx(c2sc, nbins, snaps):
        """The integer bin each electron lands in -- the part a real error moves."""
        out = []
        for R in snaps:
            f = (np.asarray(R, float) @ c2sc.T) % 1.0
            out.append(np.floor(f * nbins).astype(int) % nbins)
        return np.array(out)

    def test_the_torus_uses_the_campaign_vectors(self):
        # Not bit-identical, and the difference is a recorded fact rather than a
        # tolerance of convenience: `Torus` gets A_WC from the area through a
        # sqrt, `Geometry` reaches the same lattice by another route, and they
        # agree to 3.6e-15.  That is the size of the input difference the
        # comparison below has to absorb.
        np.testing.assert_allclose(self.torus.L1, self.Ge.L1, rtol=1e-14, atol=0)
        np.testing.assert_allclose(self.torus.L2, self.Ge.L2, rtol=1e-14, atol=0)

    def test_matches_the_engine_on_a_fixed_snapshot_set(self):
        c2sc_old = np.linalg.inv(np.column_stack([self.Ge.L1, self.Ge.L2]))
        for nbins in (60, 72, 13):
            a = st.density_grid(self.snaps, self.torus, nbins=nbins)
            b = legacy.density_grid(self.snaps, self.Ge.L1, self.Ge.L2, NE,
                                    nbins=nbins)
            # the BINNING must be exactly the same integers: a boundary error
            # would move a whole particle between cells, which is a large,
            # visible change, and no tolerance on the output should be allowed
            # to hide it.
            np.testing.assert_array_equal(
                self._binidx(self.torus.c2sc, nbins, self.snaps),
                self._binidx(c2sc_old, nbins, self.snaps),
                err_msg=f"binning nbins={nbins}")
            # the values then differ only by the last bit of the normalisation
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=0,
                                       err_msg=f"nbins={nbins}")

    def test_it_is_normalised_to_the_density_the_cell_holds(self):
        a = st.density_grid(self.snaps, self.torus, nbins=60)
        np.testing.assert_allclose(a.mean(), NE / self.torus.area, rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
