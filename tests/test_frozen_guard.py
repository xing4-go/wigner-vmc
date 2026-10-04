"""The frozen legacy tree is read-only to reconnaissance -- as a machine check.

This is a PROCESS regression, not a physics one, so it carries no firewall
marker: it says nothing about whether the VMC physics changed, and the
``-m behavior`` firewall would be a worse place for it.  It runs in the full
suite (``python -m pytest tests``).

What it pins is the 2026-10-02 incident, as a property rather than a story:

* a reconnaissance step is HANDED a working copy, and cannot be handed the frozen
  tree -- ``assert_not_frozen`` refuses before any work happens;
* if a write reaches the frozen tree anyway, ``readonly_frozen`` says so on exit
  instead of leaving a silent, self-consistent corruption;
* the real probe (``tools/recon_rs75_init.py``) refuses the frozen store on the
  exact code path that caused the incident -- ``namespace_cells`` -- and refuses
  it before exec'ing a single notebook cell.

The last one is the point of the file.  The first two test the mechanism; the
last tests that the mechanism is actually wired into the program that broke.
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CLEAN, "src"))
sys.path.insert(0, os.path.join(CLEAN, "tools"))

import frozen_guard                                            # noqa: E402
from frozen_guard import (FrozenTreeChanged, FrozenWriteRefused,   # noqa: E402
                          assert_not_frozen, diff_fingerprints, fingerprint,
                          is_inside, readonly_frozen, working_copy)


class _SyntheticFrozen:
    """A stand-in frozen store: the incident's shape, in miniature.

    It has the two properties that mattered -- more than one file, and a cache
    entry whose FILENAME does not carry its parameters -- so a test can reproduce
    the collision without touching the real tree.
    """

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._write("jastrow_crystal_k53.033.pkl", b"frozen-pickle-L0.6")
        self._write("jastrow_crystal_k53.033.meta", b"v1|L0.6|w-7.49|sr1")
        self._write("llcryst_nb2_k53.033_p1conv0.pkl", b"frozen-theta")
        os.makedirs(os.path.join(root, "nested"), exist_ok=True)
        with open(os.path.join(root, "nested", "deep.meta"), "wb") as fh:
            fh.write(b"deep")

    def _write(self, rel, data):
        with open(os.path.join(self.root, rel), "wb") as fh:
            fh.write(data)

    def path(self, rel):
        return os.path.join(self.root, rel)


def _cached_like_notebook(ckpt, name, tag):
    """The notebook's ``cached()``, reduced to the part that caused the incident.

    Keyed by FILENAME, writing on a MISS, exactly as cell 6 of the frozen
    notebook does.
    """
    pkl = os.path.join(ckpt, name + ".pkl")
    meta = os.path.join(ckpt, name + ".meta")
    if os.path.exists(pkl) and os.path.exists(meta):
        if open(meta).read().strip() == tag:
            return "HIT"
    with open(pkl, "wb") as fh:
        fh.write(b"recomputed")
    with open(meta, "w") as fh:
        fh.write(tag)
    return "MISS"


class TestContainment(unittest.TestCase):
    def test_is_inside_covers_root_children_and_siblings(self):
        self.assertTrue(is_inside(r"C:\a\b", r"C:\a\b"))
        self.assertTrue(is_inside(r"C:\a\b\c.txt", r"C:\a\b"))
        self.assertFalse(is_inside(r"C:\a\bc", r"C:\a\b"))       # prefix, not a child
        self.assertFalse(is_inside(r"C:\a", r"C:\a\b"))

    def test_assert_not_frozen_refuses_the_root_and_its_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            for target in (f.root, f.path("jastrow_crystal_k53.033.pkl")):
                with self.assertRaises(FrozenWriteRefused):
                    assert_not_frozen(target, f.root)

    def test_assert_not_frozen_allows_a_sibling_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            work = os.path.join(tmp, "work")
            self.assertEqual(assert_not_frozen(work, f.root), work)


class TestWorkingCopy(unittest.TestCase):
    def test_copy_is_outside_the_frozen_root_and_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            work = working_copy(f.root, dest=os.path.join(tmp, "work"))
            self.assertFalse(is_inside(work, f.root))
            self.assertEqual(fingerprint(work), fingerprint(f.root))

    def test_a_copy_destination_inside_the_frozen_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            with self.assertRaises(FrozenWriteRefused):
                working_copy(f.root, dest=os.path.join(f.root, "work"))

    def test_writing_the_copy_leaves_the_frozen_tree_untouched(self):
        """The incident, with the guard in place: writes land on the copy."""
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            before = fingerprint(f.root)
            work = working_copy(f.root, dest=os.path.join(tmp, "work"))

            # the five-seeds-one-filename collision that did the damage
            for L0 in (0.3, 0.45, 0.5, 0.6, 0.8):
                _cached_like_notebook(work, "jastrow_crystal_k53.033",
                                      f"v1|L0{L0:g}|sr1")

            self.assertEqual(fingerprint(f.root), before)

    def test_the_same_writes_without_the_guard_do_damage(self):
        """Negative control.  Without the copy this is the actual incident, so a
        test that could not fail if the guard were removed would be worthless."""
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            before = fingerprint(f.root)
            for L0 in (0.3, 0.45, 0.5, 0.6, 0.8):
                _cached_like_notebook(f.root, "jastrow_crystal_k53.033",
                                      f"v1|L0{L0:g}|sr1")
            self.assertNotEqual(fingerprint(f.root), before)


class TestReadonlyFrozen(unittest.TestCase):
    def test_untouched_body_yields_the_copy_and_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            before = fingerprint(f.root)
            with readonly_frozen(f.root, dest=os.path.join(tmp, "work")) as work:
                self.assertFalse(is_inside(work, f.root))
                # the collision that did the damage, contained
                for L0 in (0.3, 0.45, 0.5, 0.6, 0.8):
                    _cached_like_notebook(work, "jastrow_crystal_k53.033",
                                          f"v1|L0{L0:g}|sr1")
            # exits without raising, and the frozen tree is byte-identical
            self.assertEqual(fingerprint(f.root), before)

    def test_a_write_that_reaches_the_frozen_tree_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            with self.assertRaises(FrozenTreeChanged):
                with readonly_frozen(f.root, dest=os.path.join(tmp, "work")) as work:
                    self.assertFalse(is_inside(work, f.root))
                    # a hard-coded absolute path no CKPT redirection would stop
                    f._write("jastrow_crystal_k53.033.meta", b"v1|L0.5|sr1")

    def test_an_untracked_file_appearing_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            with self.assertRaises(FrozenTreeChanged):
                with readonly_frozen(f.root, dest=os.path.join(tmp, "work")):
                    f._write("surprise.pkl", b"new")

    def test_a_deletion_inside_the_block_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            with self.assertRaises(FrozenTreeChanged):
                with readonly_frozen(f.root, dest=os.path.join(tmp, "work")):
                    os.remove(f.path("llcryst_nb2_k53.033_p1conv0.pkl"))


class TestTheGuardCoversNestedFiles(unittest.TestCase):
    def test_fingerprint_sees_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            fp = fingerprint(f.root)
            self.assertIn("nested/deep.meta", fp)

    def test_a_change_below_the_top_level_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            with self.assertRaises(FrozenTreeChanged):
                with readonly_frozen(f.root, dest=os.path.join(tmp, "work")):
                    f._write(os.path.join("nested", "deep.meta"), b"changed")


class TestFingerprintComparisonAcrossRuns(unittest.TestCase):
    """A guard that cries wolf is worse than no guard.

    ``readonly_frozen`` compares two LIVE fingerprints, so it was never wrong.
    The persisted workflow was: write a fingerprint before a phase, reload it
    after, diff.  JSON has no tuples, so the reloaded records are lists, and
    ``(a, b, c) != [a, b, c]`` -- every path in the tree reads as changed.  The
    real drift that matters would have been one line inside a 812-line report.
    """

    def _round_trip(self, fp):
        """Exactly what keeping a fingerprint in a JSON log does to it."""
        return json.loads(json.dumps({k: list(v) for k, v in fp.items()}))

    def test_a_json_round_trip_of_the_same_tree_is_not_a_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            before = fingerprint(f.root)
            # the assertion the bug broke: a reloaded fingerprint of an
            # untouched tree must diff clean
            self.assertEqual(diff_fingerprints(before, self._round_trip(before)), [])

    def test_the_same_tree_in_either_direction_diffs_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            before = fingerprint(f.root)
            after = fingerprint(f.root)
            self.assertEqual(diff_fingerprints(self._round_trip(before), after), [])
            self.assertEqual(diff_fingerprints(before, self._round_trip(after)), [])

    def test_a_real_change_survives_the_round_trip(self):
        """Negative control for the fix: it must not simply return [] always."""
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            before = fingerprint(f.root)
            f._write("jastrow_crystal_k53.033.meta", b"v1|L0.5|sr1")
            self.assertEqual(diff_fingerprints(before, fingerprint(f.root)),
                             ["jastrow_crystal_k53.033.meta"])
            # and still, with the reloaded form on one side
            self.assertEqual(
                diff_fingerprints(self._round_trip(before), fingerprint(f.root)),
                ["jastrow_crystal_k53.033.meta"])

    def test_an_added_and_a_removed_file_still_report(self):
        """The set-difference half, i.e. a key present on only one side."""
        with tempfile.TemporaryDirectory() as tmp:
            f = _SyntheticFrozen(os.path.join(tmp, "frozen"))
            before = self._round_trip(fingerprint(f.root))
            os.remove(f.path("llcryst_nb2_k53.033_p1conv0.pkl"))
            f._write("surprise.pkl", b"new")
            self.assertEqual(diff_fingerprints(before, fingerprint(f.root)),
                             ["llcryst_nb2_k53.033_p1conv0.pkl", "surprise.pkl"])


class TestTheProbeIsWiredToTheGuard(unittest.TestCase):
    """The mechanism is only worth anything if the program that broke uses it."""

    def test_namespace_cells_refuses_the_frozen_store_before_any_work(self):
        import recon_rs75_init as recon

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FrozenWriteRefused):
                # If the refusal did not come first, this would exec 59 notebook
                # cells and take minutes; the test completing quickly IS the
                # assertion that the guard runs before the build.
                recon.namespace_cells(recon.FROZEN_CKPT, figdir=tmp)

    def test_the_probe_modules_agree_on_which_tree_is_frozen(self):
        import recon_rs75_init as recon

        self.assertEqual(os.path.abspath(recon.FROZEN_CKPT),
                         os.path.abspath(frozen_guard.FROZEN_CKPT))
        self.assertTrue(is_inside(recon.OUT_JSON, CLEAN))
        self.assertFalse(is_inside(recon.FIGDIR, frozen_guard.PROJ),
                         "the notebook's savefig must not land in the project")

    def test_the_real_frozen_store_is_refused_as_a_ckpt_target(self):
        with self.assertRaises(FrozenWriteRefused):
            assert_not_frozen(frozen_guard.FROZEN_CKPT)


if __name__ == "__main__":
    unittest.main()
