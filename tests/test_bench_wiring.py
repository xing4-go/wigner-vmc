"""Blocker A: the production driver's two LL frame/set choices.

Why this file exists
--------------------
The B1-B7 regressions all passed while `scripts/bench_rs75.py` got the r_s = 75
liquid wrong -- per particle T = -1.53 against the legacy +1.75, wrong sign.  Two
independent wiring defects sat behind that, and both were invisible to a
fixture-parameterised test for the same structural reason.

    defect 1  the LATTICE-VECTOR SET.  `Setup` carried a single `self.ai/self.ac`
              filled from `_lints8/_lcart8`, the notebook's **Gaussian-overlap**
              image set, and handed it to the **Landau-level** Bloch basis.
    defect 2  the FOLDING CELL.  `Setup.basis()` passed `self.C =
              column_stack([L1, L2])`, the **simulation supercell**, where the
              notebook passes `CToCart = column_stack([A1, A2])`, the **primitive**
              cell.

Why the earlier gates could not see either
------------------------------------------
A fixture that takes a set or a matrix as an INPUT and gives it to both engines
only ever verifies the kernel -- "same inputs, same output?" -- which is true to
1e-16 and always will be.  The driver's choice is never exercised.  Worse, the
first version of THIS file built the legacy oracle with `self.S.C` as well, so
expected and actual shared the disputed frame selection and the test agreed with
the bug.

The rule this file now follows, and the reason it is written this way:

    the legacy oracle is built from the NOTEBOOK CONTRACT, verbatim, from the raw
    initial conditions -- never from `Setup`.  Expected and actual must not share
    the choice under test.

The notebook contract, quoted from the two cells that define it:

    cell 4   sc_to_cart = np.column_stack([L1, L2])          SUPERCELL
    cell 10  CToCart    = np.column_stack([A1, A2])          PRIMITIVE
             cartToC    = np.linalg.inv(CToCart)
    cell 62  _aiB, _acB = circular_lattice(LUMAX_LL, A1, A2) PRIMITIVE
             with A1 = L1/N1, A2 = L2/N2, N1 = N2 = 6

`LandauLevelBasis` uses its matrix pair for exactly one thing --
`send_to_first_cell(r - k x zhat, lat_to_cart, cart_to_lat)` folds into the first
cell before the Bloch sum, and that sum runs over `a_cart`, a DISC of radius ~15.
The folding cell must therefore be the same lattice as the sum: the primitive
cell.  An electron folded into the supercell instead can land at |r| ~ 16, at the
very edge of the disc, where the sum is truncated.

    python -m pytest tests/test_bench_wiring.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
for _p in (os.path.join(CLEAN, "src"), ROOT, os.path.join(CLEAN, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qhvmc_engine as legacy                                    # noqa: E402
from wigner_vmc.physics.geometry import wrap_to_supercell        # noqa: E402
import bench_rs75 as B                                           # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
RTOL = 1e-12
NJ = 5
N1 = 6          # cell 4: N1 = N2 = 6, i.e. the 6x6 mesh


# -- the notebook contract, rebuilt from the raw initial conditions ---------
# Deliberately NOT via `Setup`: this is the independent side of every
# comparison below, and it must not inherit the driver's frame selection.
def notebook_frames(raw):
    """Cells 4 and 10: the supercell pair and the primitive pair.

    The frame convention matters and is easy to get backwards: these matrices are
    used as ``cart = frac @ M.T``, so their COLUMNS are the lattice vectors (that
    is what ``fold_to_primitive_cell`` does -- ``r_lat = r @ cart_to_lat.T``,
    ``r_cart = r_lat @ lat_to_cart.T``).  ``column_stack([A1, A2])`` therefore
    really is the basis ``{A1, A2}``, which is why the fold lattice and the LL
    disc's lattice -- also built from ``{A1, A2}`` -- coincide.
    """
    L1 = np.asarray(raw["L1"], float)
    L2 = np.asarray(raw["L2"], float)
    sc = np.column_stack([L1, L2])                     # cell 4
    prim = np.column_stack([L1 / N1, L2 / N1])         # cells 4 + 10
    return sc, prim, np.linalg.inv(prim)


def fractional(r, M):
    """Cell coordinates of ``r`` -- the inverse of ``cart = frac @ M.T``."""
    return np.atleast_2d(r) @ np.linalg.inv(M).T


def notebook_ll_set(raw):
    """Cell 62: `_aiB, _acB = circular_lattice(LUMAX_LL, A1, A2)`."""
    A1 = np.asarray(raw["L1"], float) / N1
    A2 = np.asarray(raw["L2"], float) / N1
    return legacy.circular_lattice(float(raw["LUMAX_LL"]), A1, A2)


class TestTheDriverWiresTheLLBasisCorrectly(unittest.TestCase):
    """`Setup.basis()` against the notebook contract, on the production inputs."""

    def setUp(self):
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        raw = json.load(open(INIT, encoding="utf-8"))
        self.raw = raw
        self.S = B.Setup(raw)

        self.sc, self.prim, self.prim_i = notebook_frames(raw)
        self.ai, self.ac = notebook_ll_set(raw)

        # the real starting configuration, by the notebook's own formula
        # (`_diag/part1_scan.py`: rng(3), uniform in the supercell).
        self.R0 = (np.random.default_rng(3).random((int(raw["ne"]), 2)) - 0.5) @ self.sc.T
        # ... and which of those electrons sit OUTSIDE the home primitive cell.
        # That is the whole point of the comparison: inside the home cell both
        # foldings agree by construction, so a test made of in-cell points
        # measures nothing (the first version of this file did exactly that and
        # reported 2e-14 while the kinetic energy was off by 8.4).
        frac = fractional(self.R0, self.prim)
        self.outside = np.abs(frac).max(axis=1) > 0.5

    def oracle(self, n_band, lat_to_cart, cart_to_lat):
        """A legacy basis, built the notebook's way and nothing else."""
        return legacy.LandauLevelBasis(np.asarray(self.raw["mesh"], float), n_band,
                                       self.ai, self.ac, lat_to_cart, cart_to_lat)

    # -- B1: the positive claim --------------------------------------------
    def test_B1_the_corrected_basis_agrees_with_the_notebook_oracle(self):
        """`Setup.basis(2)` == the notebook basis, on electrons outside the cell.

        This is the test the first Blocker A fix was missing: the oracle is the
        notebook's `LandauLevelBasis(mesh, n_band, _aiB, _acB, CToCart, cartToC)`
        and it is built from the raw initial conditions, so it cannot inherit
        whatever frame `Setup` happens to pass.
        """
        old = self.oracle(2, self.prim, self.prim_i)
        new = self.S.basis(2)
        self.assertEqual(new.padded_orbitals(self.R0[0]).shape, (self.S.nk, 3))

        checked = 0
        for r, out in zip(self.R0, self.outside):
            if not out:
                continue
            np.testing.assert_allclose(
                new.padded_orbitals(r), old.orbitals(r), rtol=RTOL, atol=RTOL,
                err_msg=(f"Setup.basis() disagrees with the notebook oracle at "
                         f"r={np.round(r, 6)}, |r|={np.linalg.norm(r):.4f} -- an "
                         f"electron outside the primitive cell"))
            checked += 1
        self.assertGreaterEqual(
            checked, 30,
            "almost every R0 electron must lie outside the home primitive cell "
            "or this test cannot see a folding error (measured: 36 of 36)")

    def test_the_folding_cell_is_the_primitive_one(self):
        """The basis must fold into `[A1, A2]`, not `[L1, L2]` -- the cheap tripwire."""
        np.testing.assert_allclose(self.S.basis(2).lat_to_cart, self.prim,
                                   rtol=0, atol=0)
        self.assertFalse(np.allclose(self.S.basis(2).lat_to_cart, self.sc),
                         "the basis is folding into the SIMULATION SUPERCELL "
                         "-- Blocker A, second divergence, is back")

    # -- B2: the negative claim --------------------------------------------
    def test_B2_the_supercell_frame_would_be_wrong(self):
        """Hand the basis the simulation supercell on purpose; it must hurt.

        The assertion is that the mistake is LARGE, not that it equals any
        particular number: the point is that this test can catch the original
        bug, so the bar is the measured order of magnitude with room to spare,
        never the measured value itself.
        """
        right = self.S.basis(2)
        # the simulation supercell, deliberately -- [L1, L2], six times larger
        wrong = self.oracle(2, self.sc, np.linalg.inv(self.sc))

        self.assertFalse(np.allclose(self.prim, self.sc),
                         "the two frames are the same matrix; the roles have "
                         "been merged again")
        self.assertAlmostEqual(abs(np.linalg.det(self.sc)) / abs(np.linalg.det(self.prim)),
                               N1 * N1, places=9)

        rel, spread = [], []
        for r, out in zip(self.R0, self.outside):
            a = right.padded_orbitals(r)
            b = wrong.orbitals(r)
            scale = max(np.abs(a).max(), 1e-300)
            rel.append(np.abs(a - b).max() / scale)
            if out:
                # a pure rescaling or a global phase would be a different story:
                # under those, b/a is the same number at every point.
                with np.errstate(all="ignore"):
                    q = b / np.where(np.abs(a) > 1e-9 * scale, a, np.nan)
                m = np.nanmedian(q)
                spread.append(np.nanmax(np.abs(q - m)) if np.isfinite(m) else 0.0)
        rel = np.asarray(rel)
        self.assertGreater(
            rel.max(), 1e-2,
            f"the supercell folding reproduces the primitive one to "
            f"{rel.max():.3e} relative -- either the two frames have become "
            f"equivalent or the fix was reverted")
        self.assertGreater(
            np.nanmax(spread), 1e-2,
            "the supercell orbitals differ from the primitive ones by a single "
            "overall factor; that would make this a convention, not a defect")

    def test_the_gaussian_overlap_set_would_be_wrong(self):
        """Wiring the overlap set into the LL basis must be visibly wrong.

        Frame held fixed at the notebook's primitive pair on BOTH sides, so this
        test isolates defect 1 (the set) and nothing else.
        """
        ov_ai = np.asarray(self.raw["_lints8"], float)
        ov_ac = np.asarray(self.raw["_lcart8"], float)
        # the two sets must differ in size -- that is the cheap tripwire
        self.assertNotEqual(len(self.ai), len(ov_ai),
                            "the LL and overlap sets are the same size; the "
                            "roles may have been merged again")
        wrong = legacy.LandauLevelBasis(np.asarray(self.raw["mesh"], float), 2,
                                        ov_ai, ov_ac, self.prim, self.prim_i)
        right = self.S.basis(2)
        d = np.abs(wrong.orbitals(self.R0[0]) - right.padded_orbitals(self.R0[0])).max()
        self.assertGreater(d, 1e-2,
                           "the overlap set reproduces the LL basis; either the "
                           "sets have become equivalent or the fix was reverted")

    # -- B3: the frame-role contract ---------------------------------------
    def test_B3_the_two_frames_keep_their_two_roles(self):
        """`LandauLevelBasis` -> primitive; wavefunction and sampler -> supercell.

        The regression is against merging the two matrices back into one
        ambiguous `C`: whichever way it were merged, one of these four
        assertions would fail.
        """
        # 1. the basis is bound to the primitive frame
        np.testing.assert_allclose(self.S.basis(2).lat_to_cart, self.prim,
                                   rtol=0, atol=0)
        self.assertFalse(np.allclose(self.S.basis(2).lat_to_cart, self.sc))

        # 2. the wavefunction is bound to the supercell frame
        theta = np.zeros(NJ + 2 * self.S.nk * (self.S.n_band_B - 1))
        wf = self.S.maker(self.S.n_band_B)(theta)
        np.testing.assert_allclose(wf.sc_to_cart, self.sc, rtol=0, atol=1e-12,
                                   err_msg="the wavefunction's frame is not the "
                                           "simulation supercell")
        self.assertFalse(np.allclose(wf.sc_to_cart, self.prim))
        # 3. ... and so is the basis the wavefunction actually carries
        np.testing.assert_allclose(wf.orb.ll.lat_to_cart, self.prim,
                                   rtol=0, atol=1e-12)

        # 4. the sampler folds by SUPERCELL vectors and not by primitive ones:
        #    a supercell translation is folded away, a primitive one is not.
        #    Points are chosen down the middle of the cell, away from the
        #    boundary, where the fold is unambiguous.
        for f in (np.array([0.2, -0.3]), np.array([-0.35, 0.1]),
                  fractional(self.R0[0], self.sc)[0]):
            q = f @ self.sc.T
            self.assertLess(abs(f).max(), 0.45,
                            "pick an interior point or the fold is ambiguous")
            back = wrap_to_supercell(q + 3 * self.sc[:, 0] - 2 * self.sc[:, 1],
                                     wf.sc_to_cart, wf.cart_to_sc)[0]
            np.testing.assert_allclose(back, q, rtol=0, atol=1e-12,
                                       err_msg="the sampler does not fold by "
                                               "supercell vectors")
            kept = wrap_to_supercell(q + self.prim[:, 0],
                                     wf.sc_to_cart, wf.cart_to_sc)[0]
            np.testing.assert_allclose(kept, q + self.prim[:, 0], rtol=0, atol=1e-12,
                                       err_msg="a PRIMITIVE translation was folded "
                                               "away; the wrap cell is wrong")

    def test_both_sets_are_kept_and_distinct(self):
        """Production must keep BOTH, under role names, not one overloaded field."""
        self.assertFalse(hasattr(self.S, "ai"),
                         "Setup.ai is back -- one field for two roles is the bug")
        np.testing.assert_allclose(self.S.ov_ac,
                                   np.asarray(self.raw["_lcart8"], float))
        self.assertNotEqual(self.S.ll_ac.shape, self.S.ov_ac.shape)

    def test_both_frames_are_kept_and_distinct(self):
        """... and the same for the two cell matrices."""
        np.testing.assert_allclose(self.S.prim_C, self.prim, rtol=0, atol=0)
        np.testing.assert_allclose(self.S.C, self.sc, rtol=0, atol=0)
        np.testing.assert_allclose(self.S.prim_Ci, self.prim_i, rtol=0, atol=1e-15)
        self.assertAlmostEqual(abs(np.linalg.det(self.S.C))
                               / abs(np.linalg.det(self.S.prim_C)),
                               float(N1 * N1), places=9)


if __name__ == "__main__":
    unittest.main()
