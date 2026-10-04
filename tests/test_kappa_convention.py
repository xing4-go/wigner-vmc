"""The coupling `kappa`: a physics convention and a campaign protocol differ.

Why this file exists
--------------------
Reproducing the frozen r_s = 75 record exactly left a residual of 2.303e-04 in E
that no amount of care with the wavefunction could remove.  It was neither
statistics nor round-off: the campaign builds every job with

    _diag/part1_scan.py:414        kappa = round(rs / np.sqrt(2.0), 4)

so its published record used ``53.033`` where the physics gives
``53.03300858899106`` -- 8.6e-6 in kappa.  The record is internally consistent
with the rounded value (`E == T + 53.033 * V` to 4e-13), which is how it was
identified.

The resolution, and what these tests pin down:

    the clean physics convention is ``kappa = r_s / sqrt(2)`` at FULL floating
    point precision, and it lives in the physics core, which never rounds;
    ``round(..., 4)`` is the FROZEN CAMPAIGN'S protocol -- provenance, not a
    definition -- and it is opt-in at the driver, under a name that says so.

`round(rs/sqrt(2), 4)` must never migrate into `physics/hamiltonian.py`; if it
did, every clean calculation would silently inherit a historical campaign
convention and the two modes would become indistinguishable.

    python -m pytest tests/test_kappa_convention.py
"""
import json
import math
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

from wigner_vmc.physics import hamiltonian as ham               # noqa: E402
import bench_rs75 as B                                          # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
RS75 = 75.0
PHYSICAL_75 = 53.03300858899106
LEGACY_75 = 53.033


class TestTheTwoKappaConventions(unittest.TestCase):

    def test_the_two_conventions_are_named_and_distinct(self):
        """At r_s = 75 the two modes must be different objects, not one fudge."""
        phys = B.physical_kappa(RS75)
        leg = B.legacy_regression_kappa(RS75)
        self.assertNotEqual(phys, leg,
                            "the physics and the campaign protocol have been "
                            "merged into a single convention")
        self.assertEqual(phys, PHYSICAL_75)
        self.assertEqual(leg, LEGACY_75)
        # the campaign's value is exactly what the frozen scan computes
        self.assertEqual(leg, round(RS75 / np.sqrt(2.0), 4))
        # small, but it is the whole of the A8 residual: 8.6e-6 in kappa
        self.assertAlmostEqual(phys - leg, 8.58899106e-06, places=13)

    def test_the_clean_convention_is_full_precision_everywhere(self):
        """No r_s in the frozen grid may come out rounded in physics mode."""
        for rs in (46.5, 55.0, 65.0, 75.0, 90.0, 100.0):
            self.assertEqual(B.physical_kappa(rs), rs / math.sqrt(2.0))
            self.assertNotEqual(B.physical_kappa(rs),
                                round(rs / math.sqrt(2.0), 4),
                                f"physics mode rounded kappa at r_s={rs}")

    def test_the_hamiltonian_never_rounds_kappa(self):
        """The rounding must not migrate into the physics core.

        `kappa_from_rs` is the core's own definition and it is exact; a source
        guard is the only way to catch `round(..., 4)` being added there later,
        since a rounded kappa is a perfectly valid-looking float.
        """
        src = open(ham.__file__, encoding="utf-8").read()
        self.assertNotIn("round(", src,
                         "physics/hamiltonian.py rounds something -- kappa is "
                         "the likely candidate and it must stay exact")
        self.assertEqual(ham.kappa_from_rs(RS75), RS75 * math.sqrt(0.5))

    def test_the_driver_defaults_to_physics_and_opts_into_the_protocol(self):
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        raw = json.load(open(INIT, encoding="utf-8"))

        clean = B.Setup(raw)
        self.assertEqual(clean.kappa, PHYSICAL_75)
        self.assertEqual(clean.kappa, float(raw["kappa"]))
        self.assertEqual(clean.kappa_mode, "physical")

        frozen = B.Setup.legacy_regression(raw)
        self.assertEqual(frozen.kappa, LEGACY_75)
        self.assertEqual(frozen.kappa_mode, "legacy-regression")

        # everything except kappa is the same object
        np.testing.assert_allclose(frozen.C, clean.C)
        np.testing.assert_allclose(frozen.prim_C, clean.prim_C)

    def test_the_wavefunction_carries_the_convention_that_built_it(self):
        """A result must be able to say which kappa produced it, not just r_s."""
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        raw = json.load(open(INIT, encoding="utf-8"))
        theta = np.zeros(5 + 2 * int(raw["nk"]) * (int(raw["NMAX_STAGE_B"]) - 1))

        for S, kappa, mode in ((B.Setup(raw), PHYSICAL_75, "physical"),
                               (B.Setup.legacy_regression(raw), LEGACY_75,
                                "legacy-regression")):
            wf = S.maker(S.n_band_B)(theta)
            self.assertEqual(wf.kappa, kappa)
            self.assertEqual(wf.kappa_mode, mode)


if __name__ == "__main__":
    unittest.main()
