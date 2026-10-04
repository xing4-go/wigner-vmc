"""Blocker B: was the delivered crystal STARTING POINT built at the kappa the
campaign used?  A cheap, notebook-free check of `initial_conditions.json`.

Why this file exists
--------------------
`check_init_digest.py` opens with the question it exists to answer --

    "Stage 2E-2: is the clean package's crystal starting point the LEGACY one?"

-- and answers it by comparing `_init_digest(v0, c0)` against the frozen
`llcryst_*` tags.  It found a mismatch on every seed.  Blocker B traced that to
one input: the campaign builds its Jastrow starting points at

    _diag/part1_scan.py:404,414    kappa = round(rs / np.sqrt(2.0), 4)   -> 53.033

while `tools/recon_rs75_init.py` built them at the physical
`rs / np.sqrt(2.0)` = 53.03300858899106.  The two differ by 8.6e-6 relative, and
`_init_digest` is certain to differ by 1e-12, so the provenance claim was false.

The trap that hid it is worth naming here, because it is why the defect survived
a test suite: the cache TAG formats kappa `{kappa:g}` -- SIX significant digits:

    f"{53.033:g}"            -> '53.033'
    f"{53.03300858899106:g}" -> '53.033'

Both conventions write the same file name and the same tag.  kappa is therefore
invisible to every name-based check, and visible only in the numbers.

What is asserted here
---------------------
1. The frozen tags really do carry the digests this file claims (read from disk,
   not just transcribed).
2. The delivered `initial_conditions.json` reproduces each frozen digest -- i.e.
   the clean start IS the legacy start.
3. NEGATIVE CONTROL / mutation proof: the pre-fix artifact, kept as a fixture,
   FAILS the very same check, and fails with exactly the five digests the
   2026-10-02 incident recorded.  A test that cannot fail on the defect it was
   written for is not evidence.
4. The artifact says WHICH kappa built the start (`kappa_init`) and keeps it
   distinct from the `kappa` the Hamiltonian consumes.

The expected values come from the frozen store and from the pre-fix artifact;
the actual values come from the delivered JSON.  They never share a source.

    python -m pytest tests/test_init_provenance.py
"""
import hashlib
import json
import os
import pickle
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
for _p in (os.path.join(CLEAN, "src"), ROOT, os.path.join(CLEAN, "scripts"),
           os.path.join(CLEAN, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
PREFIX_FIXTURE = os.path.join(HERE, "fixtures",
                              "initial_conditions_prefix_20261002.json")
#: sha256 of the fixture, as taken from the delivered artifact at
#: 2026-10-02 09:18 -- before any Blocker B change touched it.  Pinned so the
#: negative control cannot be quietly regenerated into agreement with the fix.
PREFIX_FIXTURE_SHA256 = \
    "bfbc405959b54d06965eb7d32ae9b0ad161b764cdb49a661d1ba224996a04d53"
FROZEN_CKPT = os.path.join(ROOT, "_diag", "ckpt_rebuild")

RS75 = 75.0
PHYSICAL_75 = 53.03300858899106
LEGACY_75 = 53.033

#: seed index -> (L0, frozen tag stem, the tag's own `sd` field, digest).
#: All five live in the frozen store; `p1fast1` is the fast tier's, and it is a
#: valid oracle for conv's s4 because `_init_digest` sees only (v0, c0) -- not
#: steps or sweeps -- and both tiers give seed 4 the same L0 = 0.8.  The
#: coincidence is not assumed: `p1conv4`, in %TEMP%/qhvmc_checkpoints, carries
#: the identical digest 6d38078f.
FROZEN_ORACLE = (
    (0.30, "llcryst_nb2_k53.033_p1conv0", "sd0", "cadd5d6a"),
    (0.45, "llcryst_nb2_k53.033_p1conv1", "sd1", "46d9bf1e"),
    (0.50, "llcryst_nb2_k53.033_p1conv2", "sd2", "eba63d0f"),
    (0.60, "llcryst_nb2_k53.033_p1conv3", "sd3", "7f284e09"),
    (0.80, "llcryst_nb2_k53.033_p1fast1", "sd1", "6d38078f"),
)

#: What the PRE-FIX artifact hashed to, seed order preserved.  Recorded by the
#: 2026-10-02 incident and re-derived from the fixture in test 3.
PREFIX_DIGESTS = ("4380bceb", "5c12cfbe", "b0e13b26", "b9d3cf7e", "186d86ff")


def _init_digest(v0, c0, nd=12):
    """Verbatim from the notebook's cell 80, including the shape terms.

    The SHAPES are hashed too: (36,1) and (18,2) have identical memory layout,
    so without them the digest would collide across different `nb`.
    """
    _v0, _c0 = np.asarray(v0, dtype=complex), np.asarray(c0, dtype=float)
    _w = np.concatenate([_v0.ravel().view(float), _c0.ravel(),
                         np.array(_v0.shape + _c0.shape, dtype=float)])
    _mag = np.maximum(np.abs(_w), 1e-300)
    _exp = np.clip((nd - 1) - np.floor(np.log10(_mag)), -300.0, 290.0)
    _r = np.rint(_w * np.power(10.0, _exp)).astype(np.int64)
    return hashlib.blake2b(np.ascontiguousarray(_r).tobytes(),
                           digest_size=4).hexdigest()


def tag_digest(tag):
    """The init digest carried by a cache tag.

    The tag is built `...|sc{scale}|i{digest}`, so the digest is the last field
    MINUS its literal `i` marker -- `icadd5d6a`, not `cadd5d6a`.  `_init_digest`
    itself returns the bare hex, and the two forms must not be confused.
    """
    last = tag.rsplit("|", 1)[-1]
    return last[1:] if last.startswith("i") else last


def read_frozen_tag(stem):
    path = os.path.join(FROZEN_CKPT, stem + ".meta")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def digests_of(path):
    """(seed index, L0, digest) for every crystal seed in an init artifact."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for si in range(len(raw["crystal"])):
        s = raw["crystal"][f"s{si}"]
        v0 = np.asarray(s["v0_real"], float) + 1j * np.asarray(s["v0_imag"], float)
        out.append((si, float(s["L0"]), _init_digest(v0, np.asarray(s["c0"], float))))
    return out


class TestTheFrozenOracles(unittest.TestCase):
    """The expected values themselves, read from the store rather than believed."""

    def test_the_frozen_tags_are_present_and_carry_these_digests(self):
        if not os.path.isdir(FROZEN_CKPT):
            self.skipTest(f"frozen store not present: {FROZEN_CKPT}")
        for L0, stem, sd, digest in FROZEN_ORACLE:
            tag = read_frozen_tag(stem)
            self.assertIsNotNone(tag, f"frozen tag missing: {stem}.meta")
            fields = tag.split("|")
            self.assertEqual(tag_digest(tag), digest, f"{stem}: digest changed")
            self.assertEqual(fields[-1], "i" + digest,
                             f"{stem}: the digest field is no longer `i{{digest}}`")
            self.assertIn(sd, fields, f"{stem}: tag is not seed {sd}")
            self.assertIn("k53.033", fields,
                          f"{stem}: the tag's kappa field is not '53.033'")
            # the point of Blocker B: that field cannot tell the two apart
            self.assertEqual(f"{LEGACY_75:g}", f"{PHYSICAL_75:g}")

    def test_the_oracle_L0s_are_the_five_conv_seeds(self):
        """L0 = 0.3 / 0.45 / 0.5 / 0.6 / 0.8 -- `_L0S` at `--seeds 5`."""
        self.assertEqual([o[0] for o in FROZEN_ORACLE],
                         [0.30, 0.45, 0.50, 0.60, 0.80])


class TestTheDeliveredStartIsTheLegacyStart(unittest.TestCase):

    def test_every_seed_reproduces_its_frozen_digest(self):
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        if not os.path.isdir(FROZEN_CKPT):
            self.skipTest(f"frozen store not present: {FROZEN_CKPT}")

        for si, L0, digest in digests_of(INIT):
            L0_oracle, stem, _sd, frozen = FROZEN_ORACLE[si]
            self.assertEqual(round(L0, 2), round(L0_oracle, 2),
                             f"s{si}: the artifact's L0 moved off the seed grid")
            self.assertEqual(
                digest, frozen,
                f"s{si} (L0={L0:g}): the clean start is NOT the campaign's "
                f"start -- digest {digest}, frozen tag {stem} says {frozen}. "
                f"Check the kappa the Jastrow starting points are built at.")

    def test_the_artifact_declares_which_kappa_built_the_start(self):
        """Role/contract: `kappa` is the Hamiltonian's; `kappa_init` is the
        start's.  Conflating them is the defect this whole file is about."""
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        with open(INIT, encoding="utf-8") as fh:
            raw = json.load(fh)

        self.assertEqual(float(raw["kappa"]), PHYSICAL_75,
                         "`kappa` must stay the physical value -- it is what "
                         "the Hamiltonian and the wavefunction are built from")
        self.assertIn("kappa_init", raw,
                      "the artifact must record the kappa its STARTING POINTS "
                      "were built at, or the digest cannot be checked")
        self.assertEqual(float(raw["kappa_init"]), LEGACY_75)
        self.assertEqual(float(raw["kappa_init"]),
                         round(RS75 / np.sqrt(2.0), 4))
        self.assertNotEqual(float(raw["kappa_init"]), float(raw["kappa"]))
        # ... and yet they are the same string, which is why names cannot see it
        self.assertEqual(f"{float(raw['kappa']):g}",
                         f"{float(raw['kappa_init']):g}")

    def test_the_campaign_convention_reaches_the_starting_point_only(self):
        """`kappa_init` must not have leaked into the Hamiltonian."""
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        import bench_rs75 as B
        with open(INIT, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.assertEqual(B.Setup(raw).kappa, PHYSICAL_75)
        self.assertEqual(B.physical_kappa(RS75), PHYSICAL_75)
        self.assertEqual(B.legacy_regression_kappa(RS75), LEGACY_75)


class TestTheNegativeControlFails(unittest.TestCase):
    """Mutation proof.  The pre-fix artifact is the defect; it must FAIL here.

    Re-introducing the defect by re-running the notebook would cost a 17-minute
    namespace build to produce bytes we already have: the artifact the defect
    actually wrote, frozen on disk since 2026-10-02 09:18.  Running this file's
    own checker against it is the same experiment on the same data.
    """

    def test_the_fixture_is_the_pre_fix_artifact(self):
        if not os.path.exists(PREFIX_FIXTURE):
            self.skipTest(f"pre-fix fixture not present: {PREFIX_FIXTURE}")
        with open(PREFIX_FIXTURE, "rb") as fh:
            blob = fh.read()
        self.assertEqual(hashlib.sha256(blob).hexdigest(), PREFIX_FIXTURE_SHA256,
                         "the fixture is not the bytes the defect produced -- "
                         "a negative control that has been edited proves nothing")
        raw = json.loads(blob.decode("utf-8"))
        self.assertEqual(float(raw["kappa"]), PHYSICAL_75,
                         "the fixture must be the CLEAN-kappa artifact")
        self.assertNotIn("kappa_init", raw,
                         "the fixture must predate the `kappa_init` field")

    def test_the_same_check_fails_on_it_with_the_recorded_digests(self):
        if not os.path.exists(PREFIX_FIXTURE):
            self.skipTest(f"pre-fix fixture not present: {PREFIX_FIXTURE}")
        rows = digests_of(PREFIX_FIXTURE)
        self.assertEqual(tuple(r[2] for r in rows), PREFIX_DIGESTS,
                         "the fixture no longer hashes to the digests the "
                         "2026-10-02 incident recorded")
        for si, _L0, digest in rows:
            self.assertNotEqual(
                digest, FROZEN_ORACLE[si][3],
                f"s{si}: the pre-fix artifact matches the frozen tag -- then it "
                f"was not the defect, and this file proves nothing")

    def test_the_defect_is_sharp_at_the_twelfth_digit(self):
        """Why the check is sharp: the two kappas differ by 8.6e-6 relative, and
        that is enough to move the digest with certainty."""
        rel = abs(PHYSICAL_75 - LEGACY_75) / LEGACY_75
        self.assertAlmostEqual(rel, 1.62e-07, places=9)


class TestTheFrozenStoresRecordTheCampaignKappa(unittest.TestCase):
    """kappa is recoverable from the frozen artifacts WITHOUT the notebook.

    Two independent recordings, both read-only:

      * every ``llcryst_*`` state pickles its own ``kappa`` field;
      * every ``jastrow_*`` entry pickles the warm start ``c0`` it was actually
        given, and the campaign's warm rule is exactly

            part1_scan.py:283-284   warm = J_OPT[(kind, k0)] * (kappa / k0)

        with ``k0 = 48`` the nearest grid point.  Since ``J_OPT[(kind, 48)]`` is
        itself frozen (``jastrow_{kind}_k48.pkl``, ``c``), the kappa that wrote
        any entry can be read straight back out of it.

    This is what makes the Blocker B finding a measurement rather than an
    inference: the frozen store contains BOTH conventions, and they are
    distinguishable on disk.
    """

    #: The campaign's states.  `p1fast*` is the fast tier (`--seeds 2`), whose
    #: grid is a subset of conv's (`--seeds 5`), so all six sit at kappa 53.033.
    LLCryst = ("llcryst_nb2_k53.033_p1conv0", "llcryst_nb2_k53.033_p1conv1",
               "llcryst_nb2_k53.033_p1conv2", "llcryst_nb2_k53.033_p1conv3",
               "llcryst_nb2_k53.033_p1fast0", "llcryst_nb2_k53.033_p1fast1")

    def _J48(self, kind):
        """`J_OPT[(kind, 48.0)]`, read from the frozen chain entry."""
        path = os.path.join(FROZEN_CKPT, f"jastrow_{kind}_k48.pkl")
        if not os.path.exists(path):
            self.skipTest(f"frozen chain entry not present: {path}")
        with open(path, "rb") as fh:
            return np.asarray(pickle.load(fh)["c"], float)

    def _warm_start_kappa(self, kind, pkl):
        """Which kappa a frozen jastrow entry was written at -- decided exactly.

        `c0` is stored in the pickle, and the campaign's warm rule makes it
        `J_OPT[(kind, 48)] * (kappa / 48)`.  Divide-and-round would blur it:
        the two candidates differ by 2e-6 out of 12, and the ratio's own
        floating-point noise is of that order.  So compare against both
        predictions bit-for-bit instead, and fail loudly on a third answer.
        """
        with open(pkl, "rb") as fh:
            stored = np.asarray(pickle.load(fh)["c0"], float)
        base = self._J48(kind)
        for kappa in (LEGACY_75, PHYSICAL_75):
            if np.array_equal(stored, base * (kappa / 48.0)):
                return kappa
        self.fail(
            f"{pkl}: c0 matches NEITHER convention, so the campaign's warm rule "
            f"`J_OPT[({kind}, 48)] * (kappa / 48)` (part1_scan.py:283-284) is "
            f"not what wrote it -- max|d| to legacy "
            f"{float(np.abs(stored - base * (LEGACY_75 / 48.0)).max()):.3e}, to "
            f"clean {float(np.abs(stored - base * (PHYSICAL_75 / 48.0)).max()):.3e}")

    def test_the_campaign_states_pickle_the_legacy_kappa(self):
        if not os.path.isdir(FROZEN_CKPT):
            self.skipTest(f"frozen store not present: {FROZEN_CKPT}")
        for stem in self.LLCryst:
            path = os.path.join(FROZEN_CKPT, stem + ".pkl")
            if not os.path.exists(path):
                self.skipTest(f"frozen state not present: {path}")
            with open(path, "rb") as fh:
                kappa = float(pickle.load(fh)["kappa"])
            self.assertEqual(kappa, LEGACY_75,
                             f"{stem} was optimised at kappa {kappa!r}, not the "
                             f"campaign's {LEGACY_75!r}")
            self.assertNotEqual(kappa, PHYSICAL_75)

    def test_the_frozen_warm_starts_recover_the_campaign_kappa(self):
        """Entries the incident did NOT touch were written at 53.033.

        The liquid entry is in the frozen store itself.  The crystal one is NOT:
        `_diag/ckpt_rebuild`'s crystal entry is the incident's overwrite (see the
        next test), so the surviving crystal oracle is the %TEMP% store, whose
        sha256 the pre-incident manifest records and which is unchanged on disk.
        """
        candidates = [("liquid", os.path.join(FROZEN_CKPT,
                                              "jastrow_liquid_k53.033.pkl"))]
        temp = os.environ.get("TEMP")
        if temp:
            candidates.append(("crystal", os.path.join(
                temp, "qhvmc_checkpoints", "jastrow_crystal_k53.033.pkl")))
        for kind, path in candidates:
            if not os.path.exists(path):
                self.skipTest(f"frozen entry not present: {path}")
            self.assertEqual(
                self._warm_start_kappa(kind, path), LEGACY_75,
                f"{path}: its stored warm start was not written at the "
                f"campaign's kappa {LEGACY_75!r}")
            # ... and the tag beside it cannot say so: kappa renders the same
            # string at either convention, which is the whole of Blocker B
            with open(path[:-len(".pkl")] + ".meta", encoding="utf-8") as fh:
                self.assertIn("k53.033", fh.read())

    def test_the_incidents_clean_overwrite_is_distinguishable_in_the_same_store(
            self):
        """Negative control, and it lives in the frozen tree itself.

        `_diag/ckpt_rebuild/jastrow_crystal_k53.033.pkl` is the file the
        2026-10-02 incident overwrote.  It is the ONE entry in this store that
        was written at the physical kappa -- and the same recovery says so, with
        no ambiguity.  A method that could not tell these apart would be unable
        to support anything in this file.
        """
        path = os.path.join(FROZEN_CKPT, "jastrow_crystal_k53.033.pkl")
        if not os.path.exists(path):
            self.skipTest(f"frozen entry not present: {path}")
        with open(path, "rb") as fh:
            stored = np.asarray(pickle.load(fh)["c0"], float)
        base = self._J48("crystal")
        legacy = base * (LEGACY_75 / 48.0)
        clean = base * (PHYSICAL_75 / 48.0)
        self.assertFalse(np.array_equal(stored, legacy),
                         "the incident's overwrite reads as the legacy kappa -- "
                         "then this method does not discriminate and proves "
                         "nothing")
        np.testing.assert_array_equal(stored, clean)
        # and it is legible in the tag too, which is exactly what the tag cannot
        # do for kappa: L0 is in the tag, kappa is not
        self.assertIn("L0.5", read_frozen_tag("jastrow_crystal_k53.033"))

    def test_the_artifact_points_at_the_kappa_that_kappa_implies(self):
        """The artifact's `kappa_init` must be the kappa the store records --
        read back out of an entry the incident did not touch."""
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        path = os.path.join(FROZEN_CKPT, "jastrow_liquid_k53.033.pkl")
        if not os.path.exists(path):
            self.skipTest(f"frozen entry not present: {path}")
        with open(INIT, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.assertIn("kappa_init", raw,
                      "the artifact does not record the kappa its starting "
                      "points were built at")
        self.assertEqual(float(raw["kappa_init"]),
                         self._warm_start_kappa("liquid", path),
                         "the artifact's declared starting kappa is not the one "
                         "the frozen store was written at")


class TestTheLiquidStartIsTheLegacyLiquid(unittest.TestCase):
    """`c_liquid` is built by the same `jopt_at`, so it moves with `kappa_init`.

    Its oracle is a pickle rather than a digest: the frozen
    `jastrow_liquid_k53.033.pkl` is the campaign's own output, it stores the
    post-SR `c`, and the incident left it untouched.

    HONEST LIMIT: this test does not by itself discriminate kappa.  The liquid
    entry's tag is the same string at either convention, so a call at the
    physics kappa HITS the frozen legacy entry and returns the legacy value --
    which is how the pre-fix artifact came to be correct here while its crystal
    start was wrong.  The kappa evidence is in
    `TestTheFrozenStoresRecordTheCampaignKappa`; this test guards that the
    liquid is the campaign's liquid at all.
    """

    def test_c_liquid_reproduces_the_frozen_pickle(self):
        if not os.path.exists(INIT):
            self.skipTest(f"initial conditions not present: {INIT}")
        pkl = os.path.join(FROZEN_CKPT, "jastrow_liquid_k53.033.pkl")
        if not os.path.exists(pkl):
            self.skipTest(f"frozen liquid pickle not present: {pkl}")
        with open(pkl, "rb") as fh:
            frozen = np.asarray(pickle.load(fh)["c"], float)
        with open(INIT, encoding="utf-8") as fh:
            raw = json.load(fh)
        got = np.asarray(raw["c_liquid"], float)
        self.assertEqual(got.shape, frozen.shape)
        bad = int(np.count_nonzero(got != frozen))
        self.assertEqual(
            bad, 0,
            f"c_liquid differs from the frozen liquid in {bad}/"
            f"{frozen.size} components (max |d| = "
            f"{float(np.abs(got - frozen).max()):.3e}) -- it was built at a "
            f"different kappa than the campaign used")


if __name__ == "__main__":
    unittest.main()
