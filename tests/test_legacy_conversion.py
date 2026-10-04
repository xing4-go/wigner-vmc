"""The conversion layer and the phase comparison, checked against the legacy numbers.

Two things this file exists to prevent, both of which have already happened once in this
campaign:

  * A phase comparison reading the WRONG energy field.  The legacy records carry `E`
    (a total), `E_perpart` (the production measurement), `E_min` (the best point of the
    SR trajectory) and `E_final` (its last).  Mixing them is wrong by a factor of 36 or
    by a systematic the size of the optimiser's own drift, and the result still looks
    like a number.  So the first tests build a record whose four fields DISAGREE and
    assert which one survives.

  * A re-derived scan drifting from the stored one.  `_diag/energy_scan.json` is the old
    campaign's own reduction.  The conversion here re-derives the same points from
    `_diag/part1/*.json`; if the two disagree, one of them is wrong and the difference
    is a conversion bug, because no measurement happens in either direction.  Tests that
    need the frozen tree skip when it is absent; the synthetic ones never skip.
"""
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wigner_vmc.analysis import energy_competition as ec                # noqa: E402
from wigner_vmc.analysis import statistics as stats                     # noqa: E402
from wigner_vmc.io import legacy as L                                   # noqa: E402
from wigner_vmc.io import results as R                                  # noqa: E402
from wigner_vmc.io.schema import (CRYSTAL, GAUSSIAN, LIQUID, LLROT,     # noqa: E402
                                  N_ELECTRONS, StateRecord)

DIAG = os.path.join(ROOT, "_diag")
PART1 = os.path.join(DIAG, "part1")
HAVE_PART1 = os.path.isdir(PART1)
HAVE_SCAN = os.path.exists(os.path.join(DIAG, "energy_scan.json"))


# `part1_state` fingerprints its source file for provenance, so a synthetic record has
# to have a real file behind it.  One directory for the module, written once.
_TMP = tempfile.mkdtemp(prefix="wigner_vmc_test_")
_SYNTH = os.path.join(_TMP, "synthetic.json")


def _synth_path(name="synthetic.json"):
    p = os.path.join(_TMP, name)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"synthetic": True}, fh)
    return p


def _legacy_record(**over):
    """A minimal legacy part1 record whose four energy fields all DISAGREE.

    The numbers are chosen so that each of the four would give a visibly different
    answer: E/N = -0.5, E_perpart = -0.61, E_min = -0.75, E_final = -0.55.  A test that
    used a record where they happened to coincide could not tell a correct reader from
    a wrong one.
    """
    r = {
        "key": "cry_conv_rs55_nb2_s0", "kind": "crystal", "tier": "conv", "nb": 2,
        "rs": 55.0, "kappa": 55.0 / math.sqrt(2.0), "L0": 0.5, "seed": 3,
        "T": 0.4 * N_ELECTRONS, "V": -1.0 * N_ELECTRONS, "kappa_v": None,
        "E": -0.5 * N_ELECTRONS,
        "E_perpart": -0.61, "E_perpart_err": 0.02,
        "E_start": -0.40, "E_min": -0.75, "E_final": -0.55,
        "nbar": 0.1, "P": [0.5, 0.5], "R_B": 3.0, "acc": 0.5,
    }
    r["kappa"] = 1.0            # so that T + kappa*V = 0.4N - 1.0N = -0.6N -> /N = -0.6
    r.update(over)
    return r


class TestEnergyFieldDiscipline(unittest.TestCase):
    def test_conversion_reads_E_perpart_and_nothing_else(self):
        rec = L.part1_state(_legacy_record(), _synth_path())
        self.assertEqual(rec.production_energy_per_particle, -0.61)
        self.assertEqual(rec.production_mc_error, 0.02)
        # The other three survive, in the quarantined bag, where a comparison cannot
        # reach them.
        self.assertEqual(rec.optimization.min_energy_per_particle, -0.75)
        self.assertEqual(rec.optimization.final_energy_per_particle, -0.55)
        self.assertEqual(rec.total_energy, -0.5 * N_ELECTRONS)
        for bad in (-0.5, -0.75, -0.55, -0.5 * N_ELECTRONS):
            self.assertNotEqual(rec.production_energy_per_particle, bad)

    def test_positional_argument_order_is_checked_by_construction(self):
        """`part1_state` must never accept the per-particle field as the total.

        A total of -0.5*36 = -18 and a per-particle -0.61 are two orders of magnitude
        apart, so a swap would be impossible to miss -- which is exactly the point of
        asserting the magnitudes rather than just the values.
        """
        rec = L.part1_state(_legacy_record(), _synth_path())
        self.assertLess(abs(rec.production_energy_per_particle), 5.0)
        self.assertGreater(abs(rec.total_energy or 0.0), 5.0)
        self.assertAlmostEqual(rec.total_energy / N_ELECTRONS, -0.5)

    def test_the_identity_is_cross_checked(self):
        """E_perpart = (T + kappa*V)/N.  A record that fails it is reported, not used.

        The complaint is a RETURN value, not an exception: the offending record is still
        converted (its other fields are fine), but `read_part1` surfaces the complaint so
        a caller cannot be unaware of it.
        """
        good = _legacy_record()
        good["E_perpart"] = (good["T"] + good["kappa"] * good["V"]) / N_ELECTRONS
        self.assertIsNone(L._energy_cross_check(good))

        bad = _legacy_record()          # E_perpart = -0.61 vs identity's -0.60
        msg = L._energy_cross_check(bad)
        self.assertIsNotNone(msg)
        self.assertIn("energy identity", msg)

    def test_a_record_with_no_production_energy_is_not_combineable(self):
        rec = L.part1_state(_legacy_record(E_perpart=None), _synth_path())
        self.assertFalse(rec.has_production_energy)
        with self.assertRaises(ValueError) as cm:
            stats.combine_runs([rec])
        self.assertIn("no production energy", str(cm.exception))

    def test_an_optimiser_trajectory_value_cannot_be_substituted(self):
        """The failure the whole schema exists to prevent, asserted directly.

        A state that carries E_min but no E_perpart must NOT be treated as a state whose
        energy is E_min.  `combine_runs` raises; it does not fall back.
        """
        rec = L.part1_state(_legacy_record(E_perpart=None, E_min=-0.75), _synth_path())
        self.assertEqual(rec.optimization.min_energy_per_particle, -0.75)
        with self.assertRaises(ValueError):
            stats.combine_runs([rec])


class TestPhaseComparisonArithmetic(unittest.TestCase):
    def _liq(self, e, err):
        return StateRecord(workflow=LLROT, phase=LIQUID, rs=55.0, init_id="liq",
                           production_energy_per_particle=e, production_mc_error=err)

    def _cry(self, values, errors, base="c"):
        return [StateRecord(workflow=LLROT, phase=CRYSTAL, rs=55.0, nmax=1,
                            init_id=f"{base}{i}",
                            production_energy_per_particle=v,
                            production_mc_error=e)
                for i, (v, e) in enumerate(zip(values, errors))]

    def test_sigma_total_is_the_legacy_quadrature(self):
        cry = self._cry([-0.60, -0.62, -0.61], [0.02, 0.03, 0.01])
        liq = self._liq(-0.65, 0.015)
        p = stats.compare_phases(cry, liq, "conv")

        # sigma_mc = sqrt(sum s_i^2)/n over the crystal, then hypot with the liquid's.
        s_cry = math.sqrt(0.02 ** 2 + 0.03 ** 2 + 0.01 ** 2) / 3
        s_mc = math.hypot(s_cry, 0.015)
        vals = [-0.60, -0.62, -0.61]
        mean = sum(vals) / 3
        s_init = math.sqrt(sum((v - mean) ** 2 for v in vals) / 2) / math.sqrt(3)

        self.assertAlmostEqual(p.energy_crystal, mean, places=12)
        self.assertAlmostEqual(p.delta_e, mean - (-0.65), places=12)
        self.assertAlmostEqual(p.sigma_mc, s_mc, places=12)
        self.assertAlmostEqual(p.sigma_init, s_init, places=12)
        self.assertAlmostEqual(p.sigma_total, math.hypot(s_mc, s_init), places=12)
        self.assertAlmostEqual(p.z, abs(p.delta_e) / p.sigma_total, places=12)

    def test_delta_E_is_crystal_minus_liquid_per_particle(self):
        """The sign convention: negative means the CRYSTAL is lower, i.e. it wins."""
        cry = self._cry([-0.70], [0.01])
        liq = self._liq(-0.65, 0.01)
        self.assertLess(stats.compare_phases(cry, liq, "conv").delta_e, 0.0)

    def test_a_single_seed_carries_zero_sigma_init_not_a_fabricated_one(self):
        p = stats.compare_phases(self._cry([-0.70], [0.01]), self._liq(-0.65, 0.01), "fast")
        self.assertEqual(p.n_states, 1)
        self.assertEqual(p.sigma_init, 0.0)
        self.assertAlmostEqual(p.sigma_total, math.hypot(0.01, 0.01), places=12)

    def test_unequal_crystal_errors_are_weighted_by_the_quadrature_not_averaged(self):
        """sqrt(sum s^2)/n, not mean(s)/sqrt(n): with unequal s_i the two differ."""
        cry = self._cry([-0.6, -0.6], [0.01, 0.05])
        s = math.hypot(math.sqrt(0.01 ** 2 + 0.05 ** 2) / 2, 0.0)
        p = stats.compare_phases(cry, self._liq(-0.6, 0.0), "conv")
        self.assertAlmostEqual(p.sigma_mc, s, places=12)
        self.assertNotAlmostEqual(p.sigma_mc, (0.01 + 0.05) / 2 / math.sqrt(2), places=6)


class TestCoverageGuard(unittest.TestCase):
    def test_an_unlisted_tier_raises_rather_than_being_dropped(self):
        """The guard that caught a measured nmax nb=4 state being silently omitted.

        The original guard was keyed on the tier NAME alone, so a `nmax` state at a
        different n_band slipped through as "already seen".  A synthetic instance is the
        only way to test that: the real store no longer contains the state that exposed
        it, which is precisely why the guard has to be tested on a constructed one.
        """
        rec = StateRecord(workflow=LLROT, phase=CRYSTAL, rs=80.0,
                          production_energy_per_particle=-1.0, init_id="x")
        rec.raw = {"tier": "not_a_rung", "nb": 2}
        with self.assertRaises(ec.CoverageError):
            ec.check_coverage([rec])

    def test_a_known_rung_passes(self):
        rec = StateRecord(workflow=LLROT, phase=CRYSTAL, rs=80.0,
                          production_energy_per_particle=-1.0, init_id="x")
        rec.raw = {"tier": "nest", "nb": 3}
        self.assertIn(("nest", 3), ec.check_coverage([rec]))


class TestResultsRoundTrip(unittest.TestCase):
    def test_states_survive_a_write_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            recs = [L.part1_state(_legacy_record(), _synth_path())]
            R.store_states(d, LLROT, recs, sources=[{"path": "synthetic.json", "role": "primary"}])
            back = R.load_states(d, LLROT)
            self.assertEqual(len(back), 1)
            self.assertEqual(back[0].init_id, recs[0].init_id)
            self.assertEqual(back[0].production_energy_per_particle,
                             recs[0].production_energy_per_particle)
            self.assertEqual(back[0].workflow, LLROT)

    def test_the_store_manifest_records_its_sources(self):
        with tempfile.TemporaryDirectory() as d:
            R.store_states(d, GAUSSIAN,
                           [StateRecord(workflow=GAUSSIAN, phase=CRYSTAL, rs=1.0,
                                        production_energy_per_particle=-1.0)],
                           sources=[{"path": "a.json", "role": "primary"}])
            payload = R.read_json(os.path.join(d, "legacy_gaussian", "states.json"))
            self.assertEqual(payload["sources"], [{"path": "a.json", "role": "primary"}])
            self.assertEqual(payload["n_records"], 1)


@unittest.skipUnless(HAVE_PART1 and HAVE_SCAN, "frozen part1 store not present")
class TestScanRegressionAgainstTheStoredTable(unittest.TestCase):
    """Re-derive the scan from the raw records and compare with the stored reduction.

    This is Stage 1's report item 8 expressed as a test.  Nothing is measured in either
    direction -- both sides are arithmetic on the same frozen records -- so any
    difference is a conversion or formula bug, and the tolerance can be correspondingly
    tight.
    """

    TOL_ENERGY = 1e-6
    TOL_SIGMA = 1e-9

    @classmethod
    def setUpClass(cls):
        cls.records, cls.complaints = L.read_part1(DIAG)
        cls.points, cls.brackets, cls.notes = ec.build_scan(cls.records)
        with open(os.path.join(DIAG, "energy_scan.json"), encoding="utf-8") as fh:
            cls.stored = json.load(fh)

    def test_nothing_in_the_stored_scan_is_missing_from_ours(self):
        """The direction that matters.  An extra point on our side is a question; a
        point on theirs that we cannot reproduce is a failure."""
        mine = {(round(p.rs, 4), p.tier) for p in self.points}
        theirs = {(round(float(q["rs"]), 4), q["tier"]) for q in self.stored["points"]}
        missing = sorted(theirs - mine)
        self.assertEqual(missing, [], f"the stored scan has points we do not reproduce: "
                                      f"{missing}")

    def test_every_common_point_agrees(self):
        mine = {(round(p.rs, 4), p.tier): p for p in self.points}
        worst = {"dE": 0.0, "sig": 0.0, "z": 0.0}
        n = 0
        for q in self.stored["points"]:
            key = (round(float(q["rs"]), 4), q["tier"])
            p = mine.get(key)
            if p is None:
                continue
            n += 1
            worst["dE"] = max(worst["dE"], abs(p.delta_e - float(q["dE"])))
            worst["sig"] = max(worst["sig"], abs(p.sigma_total - float(q["dE_err"])))
            if q.get("z") is not None and p.z is not None:
                worst["z"] = max(worst["z"], abs(p.z - float(q["z"])))
        self.assertGreater(n, 0, "no common points -- the two are not the same scan")
        self.assertLess(worst["dE"], self.TOL_ENERGY, f"worst dE difference {worst['dE']:.3e}")
        self.assertLess(worst["sig"], self.TOL_SIGMA, f"worst sigma difference {worst['sig']:.3e}")
        self.assertLess(worst["z"], 1e-6, f"worst z difference {worst['z']:.3e}")

    def test_the_coverage_guard_admits_the_real_store(self):
        """If this raises, a measured rung exists that the tables do not account for."""
        ec.check_coverage(self.records)

    def test_the_energy_identity_holds_on_every_legacy_record(self):
        """A non-empty complaint list here means a record was read through a wrong field."""
        identity = [c for c in self.complaints if "energy identity" in c]
        self.assertEqual(identity, [], "\n  " + "\n  ".join(identity))

    def test_the_stored_windows_match_the_brackets_we_form(self):
        """The bracket is the campaign's headline, so it is checked, not just printed."""
        mine = {b.tier: b for b in self.brackets}
        for tier, w in self.stored["windows"].items():
            b = mine.get(tier)
            if b is None:
                continue
            self.assertEqual(b.kind, w["kind"], f"{tier}: kind")
            if "resolved" in w:
                self.assertEqual(b.resolved, w["resolved"], f"{tier}: resolved")
            if w["lo"] is not None and b.lo is not None:
                self.assertAlmostEqual(b.lo, float(w["lo"]), places=6, msg=f"{tier}: lo")
            if w["hi"] is not None and b.hi is not None:
                self.assertAlmostEqual(b.hi, float(w["hi"]), places=6, msg=f"{tier}: hi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
