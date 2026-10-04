"""The nested-replay oracle table is checked against the frozen records.

Not a physics claim -- a PROVENANCE one, so it carries no firewall marker and
runs in the full suite (``python -m pytest tests``), like ``test_frozen_guard``.

What it pins is a bug that reached B8 and was caught by the tool's own
self-check rather than by a wrong number:

``NEST_SEEDS`` used to be ``(seed, digest, store)`` triples, with the leading
seed number merely duplicating the tuple's position.  ``main`` read
``NEST_SEEDS[si][0]`` for the digest, got the redundant index, and compared
every nested run against ``"0"``.  The table was right and the reader was wrong,
and the failure was silent in the worst way: the comparison would have been
against a digest that means nothing.

The same redundancy had already produced a *different* failure in the sibling
``replay_crystal_rs75.py``, where a four-tuple met a three-name unpack and the
run died at startup.  One field, one meaning is the fix; these tests keep it.

The last test is the point of the file: the oracle is only worth anything if
``main`` actually consults it, and consults it BEFORE paying for the SR.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CLEAN, "src"))
sys.path.insert(0, os.path.join(CLEAN, "tools"))
sys.path.insert(0, os.path.join(CLEAN, "scripts"))

import replay_nested_rs75 as R                                  # noqa: E402

NK = 36
NJ = 5


def _parent_blob(theta):
    """A minimal parent JSON: only what ``main`` touches before the SR."""
    return {"arms": {"legacy": {"theta": list(theta), "kappa": 53.033,
                                "kappa_mode": "legacy-regression", "n_band": 2,
                                "init_digest": "cadd5d6a"}}}


class TestTheTableHasOneFieldPerMeaning(unittest.TestCase):
    """The bug class, not the one site: a redundant field invites an off-by-one."""

    def test_every_entry_is_a_digest_store_pair(self):
        for si, entry in enumerate(R.NEST_SEEDS):
            with self.subTest(seed=si):
                self.assertEqual(len(entry), 2, "no redundant leading seed index")
                digest, store = entry
                self.assertIn(store, ("frozen", "temp"))

    def test_the_digest_field_is_a_hex_digest_and_not_the_seed_index(self):
        """The exact confusion: ``[0]`` returned an int and was compared to a str."""
        for si, (digest, _store) in enumerate(R.NEST_SEEDS):
            with self.subTest(seed=si):
                self.assertIsInstance(digest, str)
                self.assertEqual(len(digest), 8)
                int(digest, 16)                     # raises unless it is hex
                self.assertNotEqual(digest, str(si),
                                    "the digest must not be the seed's position")
                self.assertEqual(digest, digest.lower())

    def test_nest_seed_is_the_only_accessor_and_agrees_with_the_table(self):
        for si in range(len(R.NEST_SEEDS)):
            with self.subTest(seed=si):
                self.assertEqual(R.nest_seed(si), R.NEST_SEEDS[si])
                R.nest_seed(si)                      # (digest, store) unpacks


class TestTheTableMatchesTheFrozenRecords(unittest.TestCase):
    """The check that actually caught the bug, kept as a test."""

    def test_every_seed_is_reproduced_from_its_frozen_nb2_record(self):
        checked = 0
        for si in range(len(R.NEST_SEEDS)):
            pred, src = R.verify_oracle(si)
            if pred is None:                         # record absent: nothing to check
                continue
            checked += 1
            with self.subTest(seed=si):
                self.assertEqual(pred, R.nest_seed(si)[0],
                                 f"the frozen nb2 record {os.path.basename(src)} "
                                 f"does not reproduce the table's digest")
        self.assertGreater(checked, 0, "no frozen nb2 record was available to check")


class TestMainConsultsTheOracleBeforePayingForTheSR(unittest.TestCase):
    """A self-check that is never consulted is decoration."""

    def _run_main_with(self, oracle):
        """Drive ``main`` with a synthetic parent and a canned oracle.

        The parent is synthetic so this does not depend on a finished replay run,
        and the SR is never reached: the oracle check comes first.  If that
        ordering ever changes, this test starts costing two hours, which is
        itself the signal.
        """
        c0 = np.zeros(NJ)
        v = np.zeros((NK, 1), complex)
        theta = np.concatenate([c0, v.real.ravel(), v.imag.ravel()])
        self.assertEqual(theta.size, 77, "the synthetic parent must be nmax = 1")

        with tempfile.TemporaryDirectory() as tmp:
            pj = _parent_blob(theta)
            pj["init_path"] = os.path.join(tmp, "init.json")
            with open(pj["init_path"], "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            parent = os.path.join(tmp, "parent.json")
            with open(parent, "w", encoding="utf-8") as fh:
                json.dump(pj, fh)

            saved = R.verify_oracle
            R.verify_oracle = oracle
            try:
                R.main(["--parent", parent, "--arm", "legacy",
                        "--out", os.path.join(tmp, "out.json")])
            finally:
                R.verify_oracle = saved

    def test_a_disagreeing_oracle_stops_the_run(self):
        """Negative control: a wrong oracle must abort, not warn and continue."""
        with self.assertRaises(SystemExit) as cm:
            self._run_main_with(lambda si: ("deadbeef", "synthetic"))
        self.assertIn("oracle", str(cm.exception).lower())

    def test_the_real_table_passes_its_own_self_check_on_a_real_parent(self):
        """The gate B8 died on, driven end to end with the REAL oracle.

        The parent is the campaign's own nmax = 1 endpoint, so the oracle gate,
        the nesting identity and the nested-start digest must all agree.  This is
        the test that catches ``main`` binding ``want_dig`` to the wrong field:
        with the digest replaced by the seed index, the self-check reads
        ``TABLE IS WRONG`` and the run stops.
        """
        source = R.verify_oracle(0)[1]
        if not os.path.exists(source):
            self.skipTest("the frozen nb2 record is not available")
        with open(source, encoding="utf-8") as fh:
            theta = json.load(fh)["theta"]
        self.assertEqual(len(theta), 77)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                self._run_main_with_real_oracle(theta)
        except BaseException:                        # past the gate, by design
            pass

        out = buf.getvalue()
        self.assertIn(R.nest_seed(0)[0], out, "the table's digest must be printed")
        self.assertNotIn("TABLE IS WRONG", out,
                         "the real table must not disagree with the real record")
        self.assertNotIn("oracle is not trustworthy", out)
        self.assertIn("MATCH", out, "the oracle self-check must report a match")

    def _run_main_with_real_oracle(self, theta):
        """Drive ``main`` with a real parent and no oracle override."""
        with tempfile.TemporaryDirectory() as tmp:
            pj = _parent_blob(theta)
            pj["init_path"] = os.path.join(tmp, "init.json")
            with open(pj["init_path"], "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            parent = os.path.join(tmp, "parent.json")
            with open(parent, "w", encoding="utf-8") as fh:
                json.dump(pj, fh)
            R.main(["--parent", parent, "--arm", "legacy",
                    "--out", os.path.join(tmp, "out.json")])

    def test_an_absent_oracle_record_is_not_treated_as_a_mismatch(self):
        """A missing frozen record warns; it must not be confused with a mismatch.

        "cannot verify" and "verifiably wrong" call for different responses, and
        collapsing them would either hide a real mismatch or block a legitimate
        run.  The assertion is on what the oracle gate PRINTED, because the
        synthetic parent cannot survive past the gate (``Setup`` needs a real
        init file) -- so a test that only checked "no exception" would be
        asserting about the wrong failure.
        """
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                self._run_main_with(lambda si: (None, "synthetic-absent"))
        except BaseException:                        # past the gate, by design
            pass
        out = buf.getvalue()
        self.assertIn("UNVERIFIED", out,
                      "an absent record must be reported as unverified")
        self.assertNotIn("oracle is not trustworthy", out,
                         "an absent record is not a disagreement")


if __name__ == "__main__":
    unittest.main()
