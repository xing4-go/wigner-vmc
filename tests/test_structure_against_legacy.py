"""Does the clean structure layer reproduce the legacy numbers?  Asserted, not remembered.

This is the verification that was originally done by hand in a throwaway script, moved
here so it runs on every change instead of on the day someone thinks to re-run it.  The
hand-run version had already drifted once: `structure.py` was rewritten and the S(q)
check silently went from 0.0 to 1.141e+01 because the momenta had been permuted, which
is exactly the class of failure a remembered check does not catch.

Two kinds of test live here and they are deliberately separated.

  * Against the FROZEN TREE -- skipped when `../_diag` or the legacy notebook is absent,
    because the clean package is supposed to stand alone.  Skipping is honest: it says
    "not checked here", not "checked and fine".

  * Against CONFIGURATIONS WHOSE ANSWER IS KNOWN -- never skipped.  A uniform liquid must
    give g = 1, a perfect triangular lattice must give six equal nearest-neighbour peaks,
    and the filled LLL's SMA must approach Kohn's hbar*omega_c.  These catch a
    normalisation error with no legacy tree in sight, which is the only kind of control
    that still works after the frozen tree is eventually archived.
"""
import ast
import json
import math
import os
import sys
import unittest
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# The legacy engine is a top-level module of the project root; the frozen-tree classes
# import it directly so that they compare against the REAL implementation rather than
# against a copy of it.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wigner_vmc.analysis import structure as st                       # noqa: E402
from wigner_vmc.io.checkpoints import (read_ensemble,                 # noqa: E402
                                       read_structure_reference)

CKPT = os.path.join(ROOT, "_diag", "ckpt_rebuild")
NOTEBOOK = os.path.join(ROOT, "WignerCrystal_to_HallLiquid.ipynb")
ENGINE = os.path.join(ROOT, "qhvmc_engine.py")

HAVE_TREE = os.path.isdir(CKPT) and os.path.exists(ENGINE)

# The frozen geometry: 6x6 cells, 36 electrons, nu = 1.
AREA = 226.194671058465
N_E = 36
N_SIDE = 6

# The notebook's 2-D map binning.  These are the legacy render parameters, not choices
# of the clean pipeline -- they are here so the comparison is against the same grid.
SQRT_N = math.sqrt(N_E / AREA)
R_HALF = 4.0 / SQRT_N
Q_HALF = 8.0 * SQRT_N
NR, NQ, G_SMOOTH = 81, 160, 1.0

# Tolerance.  Double-precision accumulation over 1500 snapshots cannot be exact, but a
# real bug in this layer is not subtle -- the regression that motivated this file was
# off by 11.4, and a permuted column is off by ~10% of the max, not by 1e-15.
TOL = 1e-9


def _kohn_series(u: float, order: int = 12) -> float:
    """The small-q expansion of q^2/(2(1-e^{-q^2/2})), in u = q^2/2, by exact inversion.

        u / (1 - e^{-u}) = 1 / (u/2 - u^2/6 + u^3/24 - u^4/120 + ...)

    i.e. the reciprocal of the exponential's own Taylor series.  The inversion runs in
    `fractions.Fraction`, so the coefficients are exact ratios of integers and the
    reference carries no floating point of its own until the final evaluation.

    The coefficients are 1, 1/2, 1/12, 0, -1/720, 0, 1/30240, ...  The zero at u^3 is
    real, which makes a two-term truncation look unusually good here and makes it easy
    to get the u^4 coefficient wrong by writing q^4/24 instead of q^4/48.  The first
    version of this file did exactly that: it was wrong by 8.5e-4 at the smallest
    allowed q, which is LARGER than the whole Kohn-mode deviation the test was checking,
    so the test failed on a correct implementation.  Deriving the series instead of
    quoting it is what makes the reference trustworthy, and it is also the only reason
    the mistake was visible.
    """
    d = [Fraction((-1) ** k, math.factorial(k + 1)) for k in range(order + 1)]
    w = [Fraction(0)] * (order + 1)
    w[0] = Fraction(1) / d[0]
    for n in range(1, order + 1):
        w[n] = -sum(d[k] * w[n - k] for k in range(1, n + 1)) / d[0]
    return sum(float(c) * u ** k for k, c in enumerate(w))


@unittest.skipUnless(HAVE_TREE, "frozen legacy tree not present")
class TestAgainstFrozenTree(unittest.TestCase):
    """Reproduce the legacy numbers from the legacy files.  Read-only."""

    @classmethod
    def setUpClass(cls):
        import qhvmc_engine as eng                                     # noqa: E402
        cls.eng = eng
        A_WC = math.sqrt(4 * math.pi / math.sqrt(3))
        A1 = np.array([A_WC, 0.0])
        A2 = np.array([A_WC / 2, A_WC * math.sqrt(3) / 2])
        (cls.area, cls.L1, cls.L2, cls.G1, cls.G2,
         cls.mesh, cls.g1, cls.g2, cls.RL) = eng.geometry_setup(A1, A2, 6, 6, 4.0, True)
        cls.tor = st.Torus(cls.area, N_E, N_SIDE)
        cls.ref = read_structure_reference(CKPT)

    # ---------------------------------------------------------------- geometry
    def test_geometry_matches_the_engine(self):
        t = self.tor
        for name, mine, theirs in (
            ("area", t.area, self.area),
            ("|L1|", np.linalg.norm(t.L1), np.linalg.norm(self.L1)),
            ("|L2|", np.linalg.norm(t.L2), np.linalg.norm(self.L2)),
            ("G1", t.G1, self.G1),
            ("G2", t.G2, self.G2),
            ("g1", t.g1, self.g1),
        ):
            with self.subTest(quantity=name):
                self.assertLess(float(np.max(np.abs(np.asarray(mine)
                                                    - np.asarray(theirs)))), 1e-12,
                                f"{name} differs from the engine's geometry_setup")

    def test_the_bragg_vector_is_six_supercell_reciprocal_vectors(self):
        """|g1| = 6|G1| exactly, which is why the mask radius is |G1| and not |g1|."""
        self.assertAlmostEqual(float(np.linalg.norm(self.tor.g1)
                                     / np.linalg.norm(self.tor.G1)), 6.0, places=12)

    # ---------------------------------------------------------------- the q grid
    def test_allowed_momenta_cover_the_engines_grid(self):
        """Same SET of |q|, and every star has the same membership.

        Compared per star rather than per column because all 294 momenta sit in
        degenerate stars of 6, 12 or 18 -- there is no singleton to fix the column
        order -- so the column ORDER is a convention and only the star contents are
        physics.  A one-ulp difference in A_WC permutes every column while leaving
        every star identical, which is exactly what happened.
        """
        eng = self.eng
        mx = 2 * int(np.floor(4.0 / np.linalg.norm(self.G1)))
        q_all = np.array([m * self.G1 + n * self.G2
                          for m in range(-mx, mx + 1) for n in range(-mx, mx + 1)])
        qn_e = np.linalg.norm(q_all, axis=1)
        keep = (qn_e < 4.0) & (qn_e > 1e-9)
        qn_e = qn_e[keep]

        q, qn = self.tor.allowed_momenta(4.0)
        self.assertEqual(len(q), len(qn_e))
        np.testing.assert_allclose(np.sort(qn), np.sort(qn_e), atol=1e-12)

        mine = st.shells_by_magnitude(qn)
        theirs = st.shells_by_magnitude(qn_e)
        self.assertEqual(len(mine), len(theirs))
        for (_, im), (_, ie) in zip(mine, theirs):
            self.assertEqual(len(im), len(ie))

    def test_stars_have_no_singletons(self):
        """The premise of the test above: if a singleton appeared, order would matter."""
        _, qn = self.tor.allowed_momenta(4.0)
        sizes = sorted(len(ix) for _, ix in st.shells_by_magnitude(qn))
        self.assertGreater(len(sizes), 0)
        self.assertGreaterEqual(min(sizes), 6,
                                f"a star of size {min(sizes)} appeared; the column order "
                                f"is now load-bearing and the q grid needs a second look")

    # ---------------------------------------------------------------- stored arrays
    def _stored_grid(self):
        """The engine's own q grid, in the order the stored array's columns are in.

        The stored `Sq_all` was written with `np.argsort(qn)`, while this module's
        canonical order is `np.lexsort` on the rounded (|q|, qx, qy).  Both are the same
        294 momenta; the columns are permuted.
        """
        mx = 2 * int(np.floor(4.0 / np.linalg.norm(self.G1)))
        q_all = np.array([m * self.G1 + n * self.G2
                          for m in range(-mx, mx + 1) for n in range(-mx, mx + 1)])
        qn_e = np.linalg.norm(q_all, axis=1)
        keep = (qn_e < 4.0) & (qn_e > 1e-9)
        q_e, qn_e = q_all[keep], qn_e[keep]
        o = np.argsort(qn_e)
        return q_e[o], qn_e[o]

    def test_Sq_matches_the_stored_array_column_by_column(self):
        """The sharpest form of the check: same number in the same place.

        Columns are matched by the q VECTOR rather than by position, because the two
        column orders differ by convention and not by physics.  Comparing them without
        matching gives a maximum difference of 11.4 -- which reads like a broken
        structure factor and is a bookkeeping error, and which is exactly what an
        earlier version of this file silently did (it indexed the stored array, which is
        in `argsort` order, with indices from the RAW order).  So the alignment is
        asserted here rather than assumed: each match must be a distinct column.
        """
        snaps = self.ref.payload["snaps"]
        Sq_stored = np.asarray(self.ref.payload["Sq_all"])
        q, _ = self.tor.allowed_momenta(4.0)
        Sq = st.structure_factor_snapshots(snaps, q)
        self.assertEqual(Sq.shape, Sq_stored.shape)

        q_e, _ = self._stored_grid()
        match = np.array([int(np.argmin(np.linalg.norm(q - qv, axis=1))) for qv in q_e])
        miss = max(float(np.linalg.norm(q[i] - qv)) for i, qv in zip(match, q_e))
        self.assertLess(miss, 1e-12,
                        f"a stored momentum has no counterpart on this torus "
                        f"(closest is {miss:.3e} away)")
        self.assertEqual(len(set(match.tolist())), len(q_e),
                         "two stored columns matched the same column here; the two grids "
                         "are not the same set of momenta after all")

        worst = float(np.max(np.abs(Sq[:, match] - Sq_stored)))
        self.assertLess(worst, TOL, f"per-column S(q) differs by {worst:.3e}")

    def test_Sq_matches_the_stored_array_per_star(self):
        """The order-independent form, so a permutation alone can never fail this.

        Both sides are reduced to one number per degenerate star by their OWN grouping,
        which is the comparison that stays meaningful if either column order is ever
        changed again.
        """
        snaps = self.ref.payload["snaps"]
        Sq_stored = np.asarray(self.ref.payload["Sq_all"])
        q, qn = self.tor.allowed_momenta(4.0)
        Sq = st.structure_factor_snapshots(snaps, q)
        _, qn_e = self._stored_grid()

        def star_means(S, mags):
            return np.array([S[:, ix].mean() for _, ix in st.shells_by_magnitude(mags)])

        mine, theirs = star_means(Sq, qn), star_means(Sq_stored, qn_e)
        self.assertEqual(len(mine), len(theirs))
        worst = float(np.max(np.abs(mine - theirs)))
        self.assertLess(worst, TOL, f"per-star S(q) differs by {worst:.3e}")

    def test_gr_matches_the_stored_array(self):
        snaps = self.ref.payload["snaps"]
        r_st = np.asarray(self.ref.payload["r"])
        g_st = np.asarray(self.ref.payload["g"])
        r, g, _ = st.pair_correlation(snaps, self.tor, rmax=6.0, nbins=40)
        self.assertEqual(g.shape, g_st.shape)
        self.assertLess(float(np.max(np.abs(g - g_st))), TOL)

    def test_gr_plateau_is_the_exact_finite_N_value_within_its_own_bar(self):
        """N/(N-1) is the CONTINUUM plateau; over a finite window it is a sampled mean.

        Asserting it to 9 decimal places -- which is what this file did first -- asserts
        that the sampler has no noise, and fails on a correct implementation by 6.1e-4
        (measured 1.02918 against an exact 1.02857).  The tolerance below is the one the
        legacy cell itself accepted this quantity under, so this is the campaign's
        criterion rather than one invented to fit the number.
        """
        snaps = self.ref.payload["snaps"]
        r, g, _ = st.pair_correlation(snaps, self.tor, rmax=6.0, nbins=40)
        FIN = N_E / (N_E - 1.0)
        plateau = float(g[r > 4].mean())
        bar = st.poisson_pair_error(g, r, 6.0, 40, self.tor, len(snaps))
        self.assertLess(abs(plateau - FIN), 0.02,
                        f"plateau {plateau:.5f} against the exact {FIN:.5f}: the sampler "
                        f"is biased or the walk is unequilibrated")
        # and the whole curve stays inside 5 sigma of the exact finite-N curve, which is
        # the legacy cell's second acceptance test
        dev = float(np.max(np.abs(g - st.exact_lll_g(r, N_E))))
        self.assertLess(dev, 5.0 * float(np.max(bar)),
                        f"g(r) departs from the exact curve by {dev:.4f}, more than the "
                        f"5 x {float(np.max(bar)):.4f} this sample supports")

    # ---------------------------------------------------------------- engine functions
    def test_gr_matches_the_engine_function(self):
        snaps = self.ref.payload["snaps"]
        _, g_m, _ = st.pair_correlation(snaps, self.tor, rmax=6.0, nbins=40)
        _, g_e = self.eng.pair_correlation(snaps, self.L1, self.L2, N_E,
                                           rmax=6.0, nbins=40)
        self.assertLess(float(np.max(np.abs(g_m - g_e))), TOL)

    def test_minimum_image_matches_the_engine_function(self):
        d = np.array([[3.0, 1.0], [0.5, 0.5], [12.0, -3.0], [-0.1, 7.7]])
        sc = np.column_stack([self.L1, self.L2])
        c2sc = np.linalg.inv(sc)
        d_e = self.eng.minimum_image_displacement(d, sc, c2sc)
        d_m = st.minimum_image_displacement(d, self.tor.sc, self.tor.c2sc)
        self.assertLess(float(np.max(np.abs(d_m - d_e))), 1e-12)

    def test_Sq_at_zero_is_the_box_form_factor(self):
        """S(0) = N for any configuration -- the reason the S panels mask 0 < |q| < |G1|.

        This is not a physics result, it is the statement that the gridded estimator
        returns the electron count whenever it is asked for q = 0.  The figure masks
        that disc so the panel is not read as "the liquid has a huge S(0)".
        """
        snaps = self.ref.payload["snaps"]
        S0 = st.structure_factor(snaps, np.zeros((1, 2)), N_E)
        self.assertAlmostEqual(float(S0[0]), float(N_E), places=6)


@unittest.skipUnless(HAVE_TREE and os.path.exists(NOTEBOOK),
                     "frozen legacy tree or notebook not present")
class TestAgainstTheNotebookSource(unittest.TestCase):
    """The 2-D map functions live in the generator/notebook, not the engine.

    They cannot be imported, so the four function nodes are pulled out of the notebook
    JSON and compiled one at a time.  The notebook's module body is NEVER executed --
    only the def nodes -- which is why this is safe to run and why it does not need a
    sampler.  No 2-D map is stored anywhere in the frozen tree, so this is the only way
    to check them at all.
    """

    WANT = ("_gauss_blur", "structure_factor_2d", "pair_correlation_2d", "wc_shell_vectors")

    @classmethod
    def setUpClass(cls):
        with open(NOTEBOOK, encoding="utf-8") as fh:
            nb = json.load(fh)
        src = None
        for c in nb["cells"]:
            s = "".join(c["source"])
            if "def pair_correlation_2d" in s:
                src = s
                break
        if src is None:
            raise unittest.SkipTest("no cell defines pair_correlation_2d")
        tree = ast.parse(src, filename=NOTEBOOK)
        nodes = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in cls.WANT]
        if sorted(n.name for n in nodes) != sorted(cls.WANT):
            raise unittest.SkipTest(
                f"extraction incomplete: got {sorted(n.name for n in nodes)}")

        import qhvmc_engine as eng                                     # noqa: E402
        A_WC = math.sqrt(4 * math.pi / math.sqrt(3))
        A1 = np.array([A_WC, 0.0])
        A2 = np.array([A_WC / 2, A_WC * math.sqrt(3) / 2])
        area, L1, L2, G1, G2, mesh, g1, g2, RL = eng.geometry_setup(A1, A2, 6, 6, 4.0, True)
        cls.L1, cls.L2 = L1, L2
        cls.tor = st.Torus(area, N_E, N_SIDE)
        # G_SMOOTH is a DEFAULT ARGUMENT in the legacy signature, so it is evaluated at
        # def time and must exist in the namespace BEFORE the compile step, not after.
        ns = {"np": np, "g1": g1, "g2": g2, "G_SMOOTH": G_SMOOTH}
        for n in nodes:
            exec(compile(ast.Module(body=[n], type_ignores=[]), NOTEBOOK, "exec"), ns)
        cls.ns = ns

    def _snaps(self, name):
        return read_ensemble(CKPT, name).payload["snaps"]

    def test_pair_correlation_2d_matches_on_three_states(self):
        for name in ("struct_liquid_k4", "struct_crystal_k64", "ladder_crystal_k64"):
            with self.subTest(state=name):
                snaps = self._snaps(name)
                _, g_e = self.ns["pair_correlation_2d"](snaps, self.L1, self.L2, N_E,
                                                        R_HALF, NR)
                _, g_m = st.pair_correlation_2d(snaps, self.tor, R_HALF, NR, G_SMOOTH)
                self.assertEqual(g_m.shape, g_e.shape)
                self.assertEqual(float(np.max(np.abs(g_m - g_e))), 0.0)

    def test_structure_factor_2d_matches_on_three_states(self):
        qg = np.linspace(-Q_HALF, Q_HALF, NQ)
        QX, QY = np.meshgrid(qg, qg, indexing="ij")
        for name in ("struct_liquid_k4", "struct_crystal_k64", "ladder_crystal_k64"):
            with self.subTest(state=name):
                snaps = self._snaps(name)
                S_e = self.ns["structure_factor_2d"](snaps, QX, QY, N_E)
                S_m = st.structure_factor_2d(snaps, QX, QY, N_E)
                self.assertEqual(S_m.shape, S_e.shape)
                self.assertEqual(float(np.max(np.abs(S_m - S_e))), 0.0)

    def test_wc_shell_vectors_match(self):
        v_e = np.asarray(self.ns["wc_shell_vectors"]())
        v_m = self.tor.wc_shell_vectors()
        self.assertEqual(len(v_e), len(v_m))
        a = np.sort(v_e, axis=0)
        b = np.sort(v_m, axis=0)
        self.assertLess(float(np.max(np.abs(a - b))), TOL)


class TestControlsWithNoLegacyTree(unittest.TestCase):
    """Configurations whose answer is known.  These never skip.

    Everything above depends on the frozen tree being present, so none of it survives
    the day `_diag/` is archived.  These three do.
    """

    def setUp(self):
        self.tor = st.Torus(AREA, N_E, N_SIDE)

    def test_uniform_liquid_gives_g_equals_one(self):
        rng = np.random.default_rng(7)
        snaps = [(rng.random((N_E, 2)) - 0.5) @ self.tor.sc.T for _ in range(60)]
        r, g, _ = st.pair_correlation(snaps, self.tor, rmax=6.0, nbins=40)
        self.assertAlmostEqual(float(g[r > 4].mean()), 1.0, delta=0.05)

        _, g2d = st.pair_correlation_2d(snaps, self.tor, R_HALF, NR, G_SMOOTH)
        self.assertAlmostEqual(float(g2d.mean()), 1.0, delta=0.05)

    def test_perfect_triangular_lattice_gives_six_equal_neighbour_peaks(self):
        """Six-fold symmetry of the crystal's own g(x,y), with no fit and no reference."""
        sites = np.array([i * self.tor.A1 + j * self.tor.A2
                          for i in range(N_SIDE) for j in range(N_SIDE)])
        sites = sites - np.floor(sites @ self.tor.c2sc.T + 0.5) @ self.tor.sc.T
        _, g = st.pair_correlation_2d([sites], self.tor, R_HALF, NR, 0.0)

        edges = np.linspace(-R_HALF, R_HALF, NR + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        Xg, Yg = np.meshgrid(centres, centres, indexing="ij")
        peaks = []
        for k in range(6):
            th = k * math.pi / 3.0
            m = ((Xg - self.tor.A_WC * math.cos(th)) ** 2
                 + (Yg - self.tor.A_WC * math.sin(th)) ** 2) <= (0.32 * self.tor.A_WC) ** 2
            self.assertTrue(m.any(), f"nearest-neighbour disc {k} landed off the grid")
            peaks.append(float(g[m].max()))
        peaks = np.asarray(peaks)
        self.assertLess(float(peaks.max() / peaks.min() - 1.0), 1e-6,
                        f"the six NN peaks are not equal: {peaks}")

    def test_filled_lll_sma_approaches_kohns_mode_from_above(self):
        """The one limit of the SMA plot that is known independently: omega -> hbar*omega_c.

        With the EXACT filled-LLL S(q) = 1 - e^{-q^2/2} the estimator gives q^2/(2S), and
        the reference is the small-q expansion of that same expression -- reached here by
        inverting the exponential's Taylor series in exact rational arithmetic
        (`_kohn_series`), so it shares no arithmetic with `sma_dispersion`, which calls
        `np.exp` and divides.  The limit is reached from ABOVE and only at q = 0, which
        no allowed momentum on a 6x6 torus is.  (The figure's curve that dips BELOW 1 at
        the smallest q is the VMC-measured S, whose noise is comparable to q^2/4 there.
        Asserting a bracket here would be asserting the noise.)
        """
        q, qn = self.tor.allowed_momenta(4.0)
        S = st.exact_lll_sq(qn)
        w = st.sma_dispersion(qn, S)
        lo = qn < 0.8
        self.assertGreaterEqual(int(lo.sum()), 2, "no small-q momenta to test")
        w_lo = w[lo][:2]
        self.assertGreater(float(w_lo.min()), 1.0,
                           f"exact filled-LLL SMA dipped below Kohn's mode: {w_lo}")
        for qi, wi in zip(qn[lo][:2], w_lo):
            self.assertAlmostEqual(float(wi), _kohn_series(qi ** 2 / 2.0), places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
