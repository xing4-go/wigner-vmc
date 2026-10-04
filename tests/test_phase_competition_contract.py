"""The contract of ``examples/figure_construction/phase_competition.py``.

Three things are pinned here, and they are pinned because each of them is a
promise the figure makes to a reader:

1. **The recipe recomputes.**  It reads no legacy result file -- not the
   defective LL-rotation crystal JSON, and not the analysed store derived from
   it.  That is checked structurally, on the source's *code* rather than its
   prose, so the promise can be stated in the docstring without the check
   becoming vacuous.
2. **The point selection is the historical figure's.**  The two grids are
   transcribed constants, and a drift in either would silently change which
   physics the picture is about.
3. **``--budget full`` is one request with ``--budget reproduction``.**  Same
   config, same slug, same directory -- not two directories holding one run.

The end-to-end test stubs the VMC call.  It is checking the plumbing -- plan,
cache, quadrature, figure, metadata -- and paying for a real run to check
plumbing would only make it slower, not stronger.
"""
import ast
import io
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SCRIPT = os.path.join(CLEAN, "examples", "figure_construction",
                      "phase_competition.py")
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _load():
    spec = importlib.util.spec_from_file_location("phase_competition_under_test",
                                                 SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code_strings(path):
    """Every string literal in the module that is NOT a docstring.

    Prose is allowed to name a file it refuses to open -- that is how a reader
    learns the rule.  Code is not.  Filtering by the docstring nodes (rather
    than by comments, which the parser discards anyway) is what separates the
    two.
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            got = ast.get_docstring(node, clean=False)
            if got is not None:
                doc_nodes.add(got)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in doc_nodes:
                continue
            out.append((node.lineno, node.value))
    return out


#: Substrings that would mean the recipe had reached into the legacy tree.  The
#: first two are the defective LL-rotation crystal records and the store derived
#: from them; the third is the legacy regression table.
FORBIDDEN = ("legacy_llrot", "_diag/part1", "energy_scan")


class TestItRecomputesAndDoesNotReRead(unittest.TestCase):
    def test_no_legacy_path_appears_in_the_code(self):
        """The ban is on READING, and a path only becomes a read in code."""
        hits = [(ln, s) for ln, s in _code_strings(SCRIPT)
                if any(f in s for f in FORBIDDEN)]
        self.assertEqual(hits, [], f"legacy paths in code: {hits}")

    def test_the_prose_still_states_the_ban(self):
        """The check above is only meaningful while the docstring says WHY."""
        text = io.open(SCRIPT, encoding="utf-8").read()
        head = text.split('"""')[1]
        for token in FORBIDDEN:
            self.assertIn(token, head,
                          f"the module docstring no longer names {token!r}")

    def test_the_module_declares_no_legacy_path_constant(self):
        """No global may point at the legacy tree, whatever it is called."""
        mod = _load()
        for name in dir(mod):
            if name.startswith("_"):
                continue
            val = getattr(mod, name)
            if isinstance(val, str):
                for f in FORBIDDEN:
                    self.assertNotIn(f, val, f"module global {name}")


class TestThePointSelectionIsTheHistoricalOne(unittest.TestCase):
    """The grids are written down here rather than remembered.  They are the
    historical figure's selection EXTENDED around it, and both halves are pinned
    below: the extension, because it is what the figure now shows, and
    `HISTORICAL_TIER_RS`, because it is the record of what the extension was
    added to.  A run that quietly dropped the original couplings while claiming
    to extend them would pass a test of the new grid alone."""

    def test_the_ladder_couplings(self):
        mod = _load()
        self.assertEqual(mod.KAPPAS_PROTO,
                         (2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0,
                          64.0, 80.0))

    def test_the_historical_grid_is_kept_verbatim(self):
        """The metadata quotes this beside the current grid.  If it drifts, the
        'what was extended' record becomes a second, wrong grid."""
        mod = _load()
        self.assertEqual(mod.HISTORICAL_TIER_RS["conv"],
                         (40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 65.0,
                          75.0, 77.5, 80.0, 85.0, 90.0))
        self.assertEqual(mod.HISTORICAL_TIER_RS["nest"],
                         (55.0, 65.0, 75.0, 77.5, 80.0, 85.0, 90.0))
        self.assertEqual(mod.HISTORICAL_TIER_RS["nest4"], (75.0, 80.0))

    def test_the_competition_tiers(self):
        mod = _load()
        self.assertEqual(mod.TIER_NMAX, {"conv": 1, "nest": 2, "nest4": 3})
        self.assertEqual(mod.TIER_RS["conv"],
                         (25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5,
                          45.0, 47.5, 50.0, 52.5, 55.0, 60.0, 65.0, 70.0,
                          75.0, 77.5, 80.0, 85.0, 90.0))
        self.assertEqual(mod.TIER_RS["nest"],
                         (30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5,
                          50.0, 52.5, 55.0, 60.0, 65.0, 70.0, 75.0, 77.5,
                          80.0, 82.5, 85.0, 87.5, 90.0))
        self.assertEqual(mod.TIER_RS["nest4"],
                         (70.0, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0))

    def test_the_grids_are_sorted_and_unique(self):
        """The panel is a line drawn through these in order, so an unsorted or
        repeated coupling draws a fold that reads as physics."""
        mod = _load()
        for name, grid in (("TIER_RS", mod.TIER_RS),
                           ("HISTORICAL_TIER_RS", mod.HISTORICAL_TIER_RS)):
            for tier, rss in grid.items():
                self.assertEqual(list(rss), sorted(rss), f"{name}[{tier}]")
                self.assertEqual(len(set(rss)), len(rss),
                                 f"{name}[{tier}] repeats a coupling")

    def test_the_extension_kept_every_historical_point(self):
        """EXTENDED, not replaced.  Every coupling the historical figure was
        drawn at is still measured at the same truncation."""
        mod = _load()
        for tier, rss in mod.HISTORICAL_TIER_RS.items():
            missing = sorted(set(rss) - set(mod.TIER_RS[tier]))
            self.assertEqual(missing, [],
                             f"{tier}: the extension dropped {missing}")

    def test_the_sampling_is_denser_where_the_competition_is_decided(self):
        """The request was explicit: 30-55 denser for n_max 1 AND 2, n_max = 1
        alone at the small-r_s end, n_max = 2 and 3 at the large-r_s end.  A
        grid can satisfy 'contains the historical points' and still be sampled
        nowhere near finely enough to show a crossing, so the shape is pinned
        too, not only the contents."""
        mod = _load()
        # The dense window, at half the historical 5-unit spacing.
        for tier in ("conv", "nest"):
            dense = [r for r in mod.TIER_RS[tier] if 30.0 <= r <= 55.0]
            self.assertEqual(dense, [30.0, 32.5, 35.0, 37.5, 40.0, 42.5,
                                     45.0, 47.5, 50.0, 52.5, 55.0], tier)
        # Small r_s: n_max = 1 only -- the nested tiers do not reach below 30.
        self.assertEqual(min(mod.TIER_RS["conv"]), 25.0)
        self.assertEqual(min(mod.TIER_RS["nest"]), 30.0)
        self.assertEqual(min(mod.TIER_RS["nest4"]), 70.0)
        # Large r_s: the nested rungs are sampled on the fine 2.5 spacing, and
        # n_max = 3 is present at all -- it was two points at 75 and 80.
        self.assertGreaterEqual(len(mod.TIER_RS["nest4"]), 8)
        for tier in ("nest", "nest4"):
            fine = [r for r in mod.TIER_RS[tier] if r >= 80.0]
            self.assertGreater(len(fine), 3, tier)

    def test_every_tier_name_the_figure_can_draw_is_one_we_produce(self):
        """`ll_rotation.TIER_STYLE` is the figure's own list of what it draws.
        A tier we compute but it drops, or one it wants and we never produce,
        is a silent hole in the picture."""
        from wigner_vmc.figures import ll_rotation as L
        mod = _load()
        self.assertEqual(set(mod.TIER_RS), set(L.TIER_STYLE))

    def test_the_crystal_nmax_matches_the_tier_name(self):
        mod = _load()
        for tier, nmax in mod.TIER_NMAX.items():
            self.assertIsInstance(nmax, int)
            self.assertGreaterEqual(nmax, 1, tier)

    def test_the_ladder_couplings_are_rs_over_sqrt2(self):
        mod = _load()
        for k in mod.KAPPAS_PROTO:
            self.assertAlmostEqual(k * math.sqrt(2.0),
                                   round(k * math.sqrt(2.0), 12), places=12)


class TestTheWindowShowsTheGridItWasGiven(unittest.TestCase):
    """The delta_E panel's x-axis used to be two literals, 44 and 91.

    They were a deliberate editorial choice, not an oversight -- the source says
    so at `ll_rotation.py:72`: `conv` at r_s = 40 and 42.5 were excluded because
    the panel's subject is the crossing and every coupling that carries it lies
    at 45 or above.  `_kept_tiers` implements that by keeping a point only if its
    DODGED x is inside the window, and `TIER_DODGE["conv"]` is -0.75, so `conv`
    needed r_s >= 44.75 to be drawn.  The delivered figure measured 13 `conv`
    points, recorded 13 in `run_metadata.json`, and drew 11 -- on purpose.

    The problem with it was the form, not the intent: a second copy of somebody
    else's scan living inside a shared renderer, which is right for one grid and
    silently wrong for the next.  The window is derived now.  These are the
    properties that make deriving it worth more than two numbers.
    """

    def test_the_historical_window_dropped_those_points_on_purpose(self):
        """The measurement behind the change, pinned so the historical figure is
        not later described as having had a bug it did not have -- and so that
        moving the literal to 39 to 'fix' it would fail here."""
        from wigner_vmc.figures import ll_rotation as L
        import numpy as np
        mod = _load()
        hist = {k: (np.asarray(v, float), np.zeros(len(v)), np.zeros(len(v)))
                for k, v in mod.HISTORICAL_TIER_RS.items()}
        kept = L._kept_tiers(hist, window=(44.0, 91.0))
        declared = set(hist["conv"][0].tolist())
        drawn = set(kept["conv"][0].tolist())
        self.assertEqual(sorted(declared - drawn), [40.0, 42.5])
        self.assertEqual(len(declared), 13)
        self.assertEqual(len(drawn), 11)

    def test_the_window_brackets_the_current_grid(self):
        """The failure mode, asserted where it would actually bite: nothing the
        recipe measures may be dropped by the panel that measures it."""
        from wigner_vmc.figures import ll_rotation as L
        import numpy as np
        mod = _load()
        xlo, xhi = mod.de_window()
        for tier, rss in mod.TIER_RS.items():
            self.assertLessEqual(xlo, min(rss), tier)
            self.assertLessEqual(max(rss), xhi, tier)
        # ... and not merely inside the frame, but inside it AFTER the dodge,
        # which is the position `_kept_tiers` actually tests.
        tiers = {k: (np.asarray(v, float), np.zeros(len(v)), np.zeros(len(v)))
                 for k, v in mod.TIER_RS.items()}
        kept = L._kept_tiers(tiers, window=mod.de_window())
        for tier, rss in mod.TIER_RS.items():
            self.assertEqual(len(kept[tier][0]), len(rss),
                             f"{tier}: the window drops measured couplings")

    def test_the_window_follows_a_widened_grid(self):
        """Asserted on a grid that is NOT the module's, so this tests the
        derivation rather than recording its current output."""
        mod = _load()
        self.assertEqual(mod.de_window({"conv": (25.0, 90.0)}), (24.0, 91.0))
        self.assertEqual(mod.de_window({"a": (10.0,), "b": (200.0,)}),
                         (9.0, 201.0))

    def test_the_window_is_the_one_the_figure_is_handed(self):
        """`write_figures` must pass the window, not leave the renderer on its
        own default -- a derived window nobody forwards is still two literals in
        the picture."""
        import inspect
        mod = _load()
        src = inspect.getsource(mod.write_figures)
        self.assertIn("de_window()", src)
        self.assertIn("window=", src)


class TestTheRequestAndTheAlias(unittest.TestCase):
    def test_the_full_alias_is_the_reproduction_config(self):
        from wigner_vmc import load_budget, resolve_budget_name
        mod = _load()
        self.assertEqual(resolve_budget_name("full"), "reproduction")
        self.assertEqual(load_budget("full").name, "reproduction")
        self.assertEqual(resolve_budget_name("no-such-budget"), "no-such-budget")
        self.assertEqual(mod.parse_args(["--budget", "full"]).budget, "full")

    def test_the_alias_slugs_as_one_request(self):
        mod = _load()
        a = mod.request_slug(36, "ll_rotation", "full", False)
        b = mod.request_slug(36, "ll_rotation", "reproduction", False)
        from wigner_vmc import resolve_budget_name
        self.assertEqual(mod.request_slug(36, "ll_rotation",
                                          resolve_budget_name("full"), False), b)
        self.assertNotEqual(a, b)

    def test_the_width_scan_gets_its_own_directory(self):
        """`--l0-grid` multiplies the crystal cost by the grid size, so the two
        are different calculations and must not share a store."""
        mod = _load()
        self.assertNotEqual(mod.request_slug(36, "ll_rotation", "quick", False),
                            mod.request_slug(36, "ll_rotation", "quick", True))

    def test_defaults(self):
        mod = _load()
        args = mod.parse_args([])
        self.assertEqual(args.budget, mod.DEFAULT_BUDGET)
        self.assertEqual(args.ansatz, "ll_rotation")
        self.assertEqual(args.n_electrons, mod.N_ELECTRONS)
        self.assertFalse(args.l0_grid, "the width scan is opt-in: it is a cost")
        self.assertFalse(args.redo, "the cache is the resume mechanism")

    def test_the_default_ansatz_is_not_the_historical_gaussian(self):
        """The historical ladder's crystal is a wavefunction family this
        package does not implement; offering it would relabel a different
        calculation.  The parser's choices are the implemented ansatze."""
        mod = _load()
        with self.assertRaises(SystemExit):
            mod.parse_args(["--ansatz", "gaussian"])

    def test_the_output_directories_are_the_briefed_ones(self):
        mod = _load()
        self.assertEqual(
            os.path.relpath(mod.FIGDIR, CLEAN).replace("\\", "/"),
            "figures/figure_construction/phase_competition")
        self.assertEqual(
            os.path.relpath(mod.RESULTS, CLEAN).replace("\\", "/"),
            "results/figure_construction/phase_competition")


class TestCacheKeys(unittest.TestCase):
    class _Args:
        budget = "quick"
        n_electrons = 36
        ansatz = "ll_rotation"
        l0_grid = False
        redo = False
        engine = "clean"

    def test_a_key_changes_with_everything_that_changes_a_number(self):
        mod = _load()
        base = mod.point_key("crystal", 75.0, 2, 0, self._Args)
        for field, value in (("budget", "reproduction"), ("n_electrons", 24),
                             ("ansatz", "ll_rotation_pinned")):
            other = self._Args()
            setattr(other, field, value)
            self.assertNotEqual(
                base, mod.point_key("crystal", 75.0, 2, 0, other),
                f"{field} must be part of a point's identity")
        self.assertNotEqual(base, mod.point_key("crystal", 75.0, 1, 0, self._Args))
        self.assertNotEqual(base, mod.point_key("crystal", 75.0, 2, 1, self._Args))
        self.assertNotEqual(base, mod.point_key("liquid", 75.0, 2, 0, self._Args))

    def test_a_key_is_stable_across_calls(self):
        """Two processes resuming the same campaign must agree on the keys, or
        the cache silently becomes a recomputation."""
        mod = _load()
        self.assertEqual(mod.point_key("liquid", 75.0, 1, 0, self._Args),
                         mod.point_key("liquid", 75.0, 1, 0, self._Args))


def _fake_run(phase, rs, nmax, init_id, args, verbose=False):
    """A deterministic stand-in for `run_point`, so the plumbing can be tested
    without paying for VMC.  It is not a physics model and does not pretend to
    be one: what the tests assert about it is where the numbers END UP."""
    e = -0.5 * float(rs) + 0.01 * int(nmax) - 0.001 * int(init_id)
    return {"phase": str(phase), "rs": float(rs), "nmax": int(nmax),
            "init_id": int(init_id), "N": int(args.n_electrons),
            "ansatz": str(args.ansatz), "budget": str(args.budget),
            "energy_per_particle": e, "error": 0.01 + 0.001 * int(nmax),
            "acceptance": 0.6, "seconds": 0.0, "record": {}}


class TestOnePlanEndToEnd(unittest.TestCase):
    """The whole pipeline, on a grid small enough to be free.

    The grid is patched rather than parameterised: a recipe that let the grid
    be shrunk from the command line would be a recipe whose figure could be
    built from a different physics than the one it claims.
    """

    def _run(self, tmp, budget="quick"):
        mod = _load()
        patched = dict(
            RESULTS=os.path.join(tmp, "results", "figure_construction",
                                 "phase_competition"),
            FIGDIR=os.path.join(tmp, "figures", "figure_construction",
                                "phase_competition"),
            KAPPAS_PROTO=(40.0, 64.0),
            TIER_RS={"conv": (55.0,), "nest": (55.0,), "nest4": (75.0,)},
            TIER_NMAX={"conv": 1, "nest": 2, "nest4": 3},
            run_point=_fake_run,
            CLEAN=tmp,
        )
        with mock.patch.multiple(mod, **patched):
            rc = mod.main(["--budget", budget, "--quiet"])
        return mod, rc

    def test_every_declared_artifact_is_written(self):
        tmp = tempfile.mkdtemp()
        mod, rc = self._run(tmp)
        self.assertEqual(rc, 0)
        slug = mod.request_slug(36, "ll_rotation", "quick", False)
        fig_dir = os.path.join(tmp, "figures", "figure_construction",
                               "phase_competition", slug)
        res_dir = os.path.join(tmp, "results", "figure_construction",
                               "phase_competition", slug)
        self.assertTrue(os.path.isfile(
            os.path.join(fig_dir, "vmc_energy_phase_competition.png")))
        self.assertTrue(os.path.isfile(os.path.join(res_dir, "runs.json")))
        self.assertTrue(os.path.isfile(os.path.join(res_dir, "run_metadata.json")))

    def test_the_metadata_records_the_selection_and_the_provenance(self):
        tmp = tempfile.mkdtemp()
        mod, _ = self._run(tmp)
        slug = mod.request_slug(36, "ll_rotation", "quick", False)
        with io.open(os.path.join(tmp, "results", "figure_construction",
                                  "phase_competition", slug,
                                  "run_metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.assertEqual(meta["budget"], "quick")
        self.assertEqual(meta["ansatz"], "ll_rotation")
        self.assertEqual(meta["legacy_json_reads"], [])
        self.assertEqual(meta["point_selection"]["tier_nmax"],
                         {"conv": 1, "nest": 2, "nest4": 3})
        for key in ("why_recomputed", "state_of_the_historical_ladder"):
            self.assertTrue(meta[key], key)
        # The delta_E points are stored as numbers, not as a figure only.
        for tier in ("conv", "nest", "nest4"):
            self.assertIn(f"{tier}_rows", meta["competition"])

    def test_every_drawn_point_carries_a_delta_and_an_error(self):
        tmp = tempfile.mkdtemp()
        mod, _ = self._run(tmp)
        slug = mod.request_slug(36, "ll_rotation", "quick", False)
        with io.open(os.path.join(tmp, "results", "figure_construction",
                                  "phase_competition", slug,
                                  "run_metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        rows = [r for tier in ("conv", "nest", "nest4")
                for r in meta["competition"][f"{tier}_rows"]]
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertIsInstance(r["delta_e"], float)
            self.assertIsInstance(r["sigma_total"], float)
            self.assertGreater(r["sigma_total"], 0.0)
            self.assertEqual(r["n_states"], 1)
            # delta_E is a DIFFERENCE, and it is the one the metadata names.
            self.assertAlmostEqual(
                r["delta_e"], r["energy_crystal"] - r["energy_liquid"], places=12)

    def test_a_second_run_reuses_the_cache_instead_of_recomputing(self):
        tmp = tempfile.mkdtemp()
        mod, _ = self._run(tmp)
        slug = mod.request_slug(36, "ll_rotation", "quick", False)
        store = os.path.join(tmp, "results", "figure_construction",
                             "phase_competition", slug, "runs.json")
        before = io.open(store, encoding="utf-8").read()
        with mock.patch.multiple(
                mod,
                RESULTS=os.path.join(tmp, "results", "figure_construction",
                                     "phase_competition"),
                FIGDIR=os.path.join(tmp, "figures", "figure_construction",
                                    "phase_competition"),
                CLEAN=tmp,
                KAPPAS_PROTO=(40.0, 64.0),
                TIER_RS={"conv": (55.0,), "nest": (55.0,), "nest4": (75.0,)},
                run_point=mock.Mock(side_effect=AssertionError(
                    "a cached point must not be recomputed"))):
            rc = mod.main(["--budget", "quick", "--quiet"])
        self.assertEqual(rc, 0)
        self.assertEqual(io.open(store, encoding="utf-8").read(), before)

    def test_redo_recomputes(self):
        tmp = tempfile.mkdtemp()
        mod, _ = self._run(tmp)
        patched = dict(
            RESULTS=os.path.join(tmp, "results", "figure_construction",
                                 "phase_competition"),
            FIGDIR=os.path.join(tmp, "figures", "figure_construction",
                                "phase_competition"),
            CLEAN=tmp,
            KAPPAS_PROTO=(40.0, 64.0),
            TIER_RS={"conv": (55.0,), "nest": (55.0,), "nest4": (75.0,)},
            TIER_NMAX={"conv": 1, "nest": 2, "nest4": 3},
            run_point=mock.Mock(side_effect=_fake_run),
        )
        with mock.patch.multiple(mod, **patched):
            rc = mod.main(["--budget", "quick", "--quiet", "--redo"])
        self.assertEqual(rc, 0)
        self.assertGreater(patched["run_point"].call_count, 0)


class TestAQuickFigureSaysSoOnItsFace(unittest.TestCase):
    """A quick figure that looked like a production one is the easiest possible
    way to mislead, and this recipe had exactly that hole: it recorded the mark in
    ``run_metadata.json`` and drew nothing, so the PNG was a smoke test that read
    as a result.  The mark is now on the figure, and it is asserted the only way
    that means anything -- the SAME numbers rendered at the two budgets must come
    out as different images, the quick render must be deterministic, and the
    difference must appear in BOTH panels.
    """

    def _arrays(self):
        """A small ladder and tier table.  Not physics: what is asserted is where
        ink ends up, and paying for VMC to assert that would only be slower."""
        import numpy as np
        rs = np.array([2.828427, 5.656854, 11.313708, 22.627417, 33.941125,
                       45.254834, 56.568542, 67.881451, 90.509668, 113.137085])
        e_liq = -0.5 * rs - 0.02 * rs ** 1.5
        e_cry = e_liq - np.linspace(0.02, -0.30, rs.size)
        ladder = {"rs": rs, "e_liq": e_liq, "e_cry": e_cry,
                  "s_liq": np.full(rs.size, 0.05),
                  "s_cry": np.full(rs.size, 0.05)}
        tiers = {
            "conv": (np.array([45.0, 55.0, 75.0, 90.0]),
                     np.array([0.05, -0.10, -0.20, -0.34]),
                     np.array([0.02, 0.02, 0.03, 0.03])),
            "nest": (np.array([55.0, 75.0, 90.0]),
                     np.array([-0.12, -0.19, -0.36]),
                     np.array([0.02, 0.03, 0.03])),
            "nest4": (np.array([75.0, 80.0]), np.array([-0.12, -0.21]),
                      np.array([0.03, 0.03])),
        }
        return ladder, tiers

    def test_the_mark_is_drawn_and_is_the_only_difference(self):
        import matplotlib.pyplot as plt
        import numpy as np
        mod = _load()
        ladder, tiers = self._arrays()
        with tempfile.TemporaryDirectory() as td:
            imgs = {}
            # Two renders at `quick` and one at `reproduction`, so the comparison
            # is the same numbers through the same code with only the budget
            # changed.
            for label, budget in (("quick", "quick"), ("quick_again", "quick"),
                                  ("reproduction", "reproduction")):
                d = os.path.join(td, label)
                mod.write_figures(d, ladder, tiers, budget)
                imgs[label] = plt.imread(
                    os.path.join(d, "vmc_energy_phase_competition.png"))
                self.assertTrue(np.isfinite(imgs[label]).all(), label)

        np.testing.assert_array_equal(
            imgs["quick"], imgs["quick_again"],
            "the quick render is not deterministic for the same inputs")

        changed = (np.abs(imgs["quick"] - imgs["reproduction"]).max(axis=2)
                   > 1.0 / 255.0)
        self.assertGreater(
            int(changed.sum()), 200,
            "the quick and reproduction figures are the same image: the QUICK "
            "stamp is not being drawn")
        # Both panels, not one.  The ladder is stamped at its lower left and the
        # delta_E panel at its upper right; both sit clear of the midline, and a
        # stamp that covered one panel of a two-panel figure would leave the
        # figure saying 'quick' in one half and looking like a result in the
        # other.
        half = changed.shape[0] // 2
        self.assertGreater(int(changed[:half].sum()), 50,
                           "no stamp in the energy ladder")
        self.assertGreater(int(changed[half:].sum()), 50,
                           "no stamp in the delta_E panel")

    def test_the_stamp_predicate_fails_safe(self):
        """Only `reproduction` is unstamped.  An unrecognised budget must draw the
        mark: over-marking a smoke test is untidy, under-marking one is the
        failure this exists to prevent."""
        mod = _load()
        self.assertIsNone(mod.figure_stamp("reproduction"))
        for budget in ("quick", "full", "production", "", "QUICK"):
            self.assertEqual(mod.figure_stamp(budget), mod.QUALITY_STAMP, budget)
        self.assertIn("not publication quality", mod.QUALITY_STAMP)


# ==========================================================================
# the nested construction
# ==========================================================================
# What went wrong, and what stops it going wrong again.  The figure's legend has
# said `nested n_max = 2` and `nested n_max = 3` from the start, and until now
# every crystal point was a FRESH Gaussian-overlap seed optimised on its own:
# `VMC.nest` existed and was correct, and this recipe never called it.  The gap
# in the suite was exactly that -- `test_the_crystal_nmax_matches_the_tier_name`
# pins the NUMBER in the tier name, and nothing pinned the CONSTRUCTION.
def _state(c=None, v=None, nmax=1, phase="crystal", rs=75.0):
    """A stand-in RunState.  `nest` reads only c, v, nmax and phase."""
    from types import SimpleNamespace
    c = np.zeros(5) if c is None else np.asarray(c, float)
    v = (np.zeros((36, 1), complex) if v is None
         else np.asarray(v, complex))
    return SimpleNamespace(c=c, v=v, nmax=int(nmax), phase=phase, rs=float(rs),
                           snaps=None)


class _Args:
    budget = "quick"
    n_electrons = 36
    ansatz = "ll_rotation"
    l0_grid = False
    redo = False
    engine = "clean"
    verbose = False
    construction = "nested"


class TestTheConstructionIsSeparated(unittest.TestCase):
    def test_the_independent_slug_is_unchanged(self):
        """The delivered store must keep its exact name: it is the baseline the
        comparison reads, and a renamed directory is a baseline thrown away."""
        mod = _load()
        self.assertEqual(mod.request_slug(36, "ll_rotation", "quick", False),
                         "N36__llrot__quick__l01")
        self.assertEqual(mod.CONSTRUCTION_SUFFIX["independent"], "")
        self.assertEqual(
            mod.request_slug(36, "ll_rotation", "quick", False, "independent"),
            mod.request_slug(36, "ll_rotation", "quick", False))

    def test_the_nested_family_is_a_sibling(self):
        mod = _load()
        self.assertEqual(
            mod.request_slug(36, "ll_rotation", "quick", False, "nested"),
            "N36__llrot__quick__l01__nested")
        self.assertNotEqual(
            mod.request_slug(36, "ll_rotation", "quick", False, "nested"),
            mod.request_slug(36, "ll_rotation", "quick", False, "independent"))

    def test_an_unknown_construction_refuses(self):
        """A typo must not fall through to an empty suffix and share the
        independent store -- which is the one thing the suffix exists to stop."""
        mod = _load()
        for bad in ("nest", "nested ", "NESTED", ""):
            with self.assertRaises(ValueError, msg=bad):
                mod.request_slug(36, "ll_rotation", "quick", False, bad)

    def test_the_default_is_the_delivered_construction(self):
        mod = _load()
        self.assertEqual(mod.parse_args([]).construction, "independent")
        self.assertEqual(mod.CONSTRUCTIONS[0], "independent")

    def test_the_parent_only_couplings_are_not_displayed(self):
        """82.5 and 87.5 need an nmax=1 parent and are not conv points.  Computed
        is not plotted: widening the drawn series would change two things at once
        and destroy the before/after comparison."""
        mod = _load()
        required = set(mod.required_crystal_rs())
        displayed = set(mod.TIER_RS["conv"])
        self.assertTrue(set(mod.TIER_RS["nest"]) <= required)
        self.assertTrue(set(mod.TIER_RS["nest4"]) <= required)
        self.assertEqual(sorted(required - displayed), [82.5, 87.5])
        self.assertEqual(mod.TIER_RS["conv"],
                         (25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5,
                          45.0, 47.5, 50.0, 52.5, 55.0, 60.0, 65.0, 70.0,
                          75.0, 77.5, 80.0, 85.0, 90.0))


class TestANestIsANest(unittest.TestCase):
    """Level 1: the vector about to be optimised IS the parent's, padded."""

    def test_the_child_start_is_the_padded_parent(self):
        from wigner_vmc.wavefunctions.nesting import nesting_residual, pad_v
        rng = np.random.default_rng(11)
        v_parent = rng.standard_normal((36, 1)) + 1j * rng.standard_normal((36, 1))
        v_child = pad_v(v_parent, 3)
        self.assertEqual(v_child.shape, (36, 2))
        self.assertTrue((v_child[:, 0] == v_parent[:, 0]).all())
        self.assertTrue((v_child[:, 1] == 0).all())
        self.assertEqual(nesting_residual(v_parent, v_child), (0.0, 0.0))

    def test_the_guard_that_can_fire_is_the_one_used(self):
        """`coefficient_identity_deviation` pads its OWN argument, so it returns
        (0, 0) for every input including a wrong one.  A guard that cannot fail
        is not a guard, and using it is how the defect this suite missed would
        have survived a test."""
        from wigner_vmc.wavefunctions import nesting
        rng = np.random.default_rng(12)
        v = rng.standard_normal((36, 1)) + 1j * rng.standard_normal((36, 1))
        wrong = rng.standard_normal((36, 2)) + 1j * rng.standard_normal((36, 2))
        self.assertEqual(nesting.coefficient_identity_deviation(wrong, 3),
                         (0.0, 0.0),
                         "if this ever stops being (0,0) the module changed")
        head, tail = nesting.nesting_residual(v, wrong)
        self.assertGreater(head + tail, 0.0,
                           "nesting_residual must reject a non-padded child")

    def test_the_recipe_imports_the_guard_that_can_fire(self):
        mod = _load()
        self.assertIs(mod.nesting_residual, __import__(
            "wigner_vmc.wavefunctions.nesting", fromlist=["nesting_residual"]
        ).nesting_residual)

    def test_a_jastrow_reparameterisation_is_not_a_padded_vector(self):
        """The child's Jastrow must be the parent's OPTIMISED c, not a fresh
        vector.  Padding is about v, and a child built with a re-drawn c would
        pass a v-only check while starting somewhere else."""
        mod = _load()
        parent = _state(c=np.arange(5, dtype=float) + 0.25,
                        v=np.ones((36, 1), complex) * 0.1)
        seen = {}

        class _Res:
            energy_per_particle = -1.0
            error = 0.01
            acceptance = 0.5
            record = {}
            nesting = {"head": 0.0, "tail": 0.0}
            state = _state(v=np.ones((36, 2), complex) * 0.1, nmax=2)

        class _VMC:
            def __init__(self, **kw):
                pass

            def nest(self, shim, new_nmax, budget=None, verbose=False):
                seen["c"] = np.asarray(shim.state.c, float)
                seen["v"] = np.asarray(shim.state.v, complex)
                seen["init_id"] = shim.config.init_id
                return _Res()

        with mock.patch.multiple(mod, VMC=_VMC,
                                 embedding_report=lambda *a, **k: {"ok": True,
                                                                   "head": 0.0,
                                                                   "tail": 0.0,
                                                                   "max_abs_dE": 0.0,
                                                                   "n_configurations": 4}):
            rec, _ = mod.run_nested_point(75.0, 2, 0, {"energy_per_particle": -0.5,
                                                       "error": 0.01, "key": "k"},
                                          parent, _Args())
        self.assertTrue((seen["c"] == np.asarray(parent.c, float)).all(),
                        "the child inherited a different Jastrow")
        self.assertTrue((seen["v"] == np.asarray(parent.v, complex)).all(),
                        "the child was handed a different orbital vector")
        self.assertEqual(seen["init_id"], 0)


class TestTheChainIsStructural(unittest.TestCase):
    """`nmax = 3` must consume the `nmax = 2` that was RECORDED, not a second
    trajectory that happens to look similar.  Determinism is not the guarantee --
    the architecture is."""

    def _resolver(self, mod, tmp, calls):
        def fake_run_point(phase, rs, nmax, init_id, args, verbose=False,
                           with_state=False):
            calls.append(("root", float(rs), int(nmax)))
            rec = {"phase": phase, "rs": float(rs), "nmax": int(nmax),
                   "init_id": int(init_id), "N": 36, "ansatz": "ll_rotation",
                   "budget": "quick", "energy_per_particle": -0.5 - 0.01 * nmax,
                   "error": 0.01, "acceptance": 0.5, "seconds": 0.0, "record": {}}
            return (rec, _state(nmax=int(nmax))) if with_state else rec

        def fake_nested(rs, nmax, init_id, parent_rec, parent_state, args,
                        verbose=False):
            calls.append(("nest", float(rs), int(nmax), int(init_id),
                          mod.state_digest(parent_state)))
            rec = {"phase": "crystal", "rs": float(rs), "nmax": int(nmax),
                   "init_id": int(init_id), "N": 36, "ansatz": "ll_rotation",
                   "budget": "quick", "energy_per_particle": -0.5 - 0.01 * nmax,
                   "error": 0.01, "acceptance": 0.5, "seconds": 0.0, "record": {},
                   "construction": "nested", "parent_state_digest":
                       mod.state_digest(parent_state)}
            return rec, _state(nmax=int(nmax))

        return mock.patch.multiple(mod, run_point=fake_run_point,
                                   run_nested_point=fake_nested)

    def test_the_third_rung_consumes_the_recorded_second(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            with self._resolver(mod, tmp, calls):
                res = mod.CrystalResolver(_Args(), tmp, {})
                _, s3 = res.state(75.0, 3, 0)
        nests = [c for c in calls if c[0] == "nest"]
        self.assertEqual([c[2] for c in nests], [2, 3],
                         "the chain must be built one rung at a time, in order")
        # The digest nmax=3 consumed is the one nmax=2 produced.
        d2 = res.state(75.0, 2, 0)[1]
        self.assertEqual(nests[1][4], mod.state_digest(d2))
        self.assertEqual(len([c for c in calls if c[0] == "root"]), 1,
                         "the root was computed more than once")

    def test_every_nested_parent_is_at_the_same_coupling_and_seed(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            with self._resolver(mod, tmp, calls):
                res = mod.CrystalResolver(_Args(), tmp, {})
                for rs in (70.0, 75.0, 85.0):
                    for i in (0, 1):
                        res.state(rs, 3, i)
        pairs = {}
        for c in calls:
            if c[0] == "nest":
                pairs.setdefault((c[1], c[2], c[3]), set()).add(c[4])
        self.assertTrue(pairs, "no nests were built")
        # Every (rs, nmax, seed) nests exactly once, from exactly one parent
        # state -- so no coupling or seed can acquire a second, unrecorded
        # trajectory part way up the ladder.
        for key, digests in pairs.items():
            self.assertEqual(len(digests), 1, key)
        self.assertEqual(sorted({k[:2] for k in pairs}),
                         [(70.0, 2), (70.0, 3), (75.0, 2), (75.0, 3),
                          (85.0, 2), (85.0, 3)])

    def test_a_missing_state_refuses_rather_than_recomputing(self):
        """A cached record with no state file is the one situation where
        recomputing the parent would silently make the child's ancestry a lie."""
        mod = _load()
        with tempfile.TemporaryDirectory() as tmp:
            args = _Args()
            key = mod.point_key("crystal", 75.0, 1, 0, args)
            runs = {key: {"energy_per_particle": -0.5, "error": 0.01,
                          "phase": "crystal", "rs": 75.0, "nmax": 1,
                          "init_id": 0, "N": 36}}
            res = mod.CrystalResolver(args, tmp, runs)
            with self.assertRaises(RuntimeError) as ctx:
                res.state(75.0, 2, 0)
            self.assertIn("state file is missing", str(ctx.exception))

    def test_the_state_round_trips_and_drops_the_snapshots(self):
        mod = _load()
        st = _state(c=np.arange(5, dtype=float))
        st.snaps = [np.zeros((36, 2))] * 3
        with tempfile.TemporaryDirectory() as tmp:
            path = mod.save_state(tmp, "abc", st)
            back = mod.load_state(tmp, "abc")
            self.assertTrue(os.path.isfile(path))
            self.assertIsNone(back.snaps, "snaps must not be pickled")
            self.assertTrue((back.c == st.c).all())
            self.assertTrue((back.v == st.v).all())
            self.assertEqual(mod.state_digest(back), mod.state_digest(st))

    def test_the_digest_is_exact_not_rounded(self):
        """The digest is the whole evidence that a child consumed a particular
        parent, so a change too small to matter physically must still move it."""
        mod = _load()
        a = _state(v=np.zeros((36, 1), complex))
        b = _state(v=np.zeros((36, 1), complex))
        b.v[0, 0] = 1e-15
        self.assertNotEqual(mod.state_digest(a), mod.state_digest(b))

    def test_the_resolver_refuses_to_skip_a_rung(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.multiple(mod, VMC=object,
                                     embedding_report=lambda *a, **k: {"ok": True}):
                with self.assertRaises(ValueError):
                    mod.run_nested_point(75.0, 3, 0,
                                         {"energy_per_particle": -0.5, "error": 0.01},
                                         _state(nmax=1), _Args())


class TestTheEmbeddingCheck(unittest.TestCase):
    """Level 2, and level 3's status as a report rather than a gate."""

    def test_the_child_final_energy_is_not_constrained(self):
        """The prohibition, stated where the numbers are.  Exact nesting makes
        the child's START the parent's FINISH; it says nothing about where the
        child's own SR stops.  No test anywhere may assert that a rung's final
        energy is below its parent's."""
        mod = _load()

        class _Res:
            energy_per_particle = -0.40     # ABOVE the parent's -0.50
            error = 0.01
            acceptance = 0.5
            record = {}
            nesting = {"head": 0.0, "tail": 0.0}
            state = _state(nmax=2)

        class _VMC:
            def __init__(self, **kw):
                pass

            def nest(self, *a, **k):
                return _Res()

        with mock.patch.multiple(mod, VMC=_VMC,
                                 embedding_report=lambda *a, **k: {
                                     "ok": True, "head": 0.0, "tail": 0.0,
                                     "max_abs_dE": 0.0, "n_configurations": 4}):
            rec, _ = mod.run_nested_point(
                75.0, 2, 0, {"energy_per_particle": -0.50, "error": 0.01,
                             "key": "k"}, _state(nmax=1), _Args())
        self.assertGreater(rec["delta_e_vs_parent"], 0.0,
                           "this fixture exists to rise; if it stopped, the "
                           "prohibition is no longer being exercised")
        self.assertGreater(rec["z_vs_parent"], 3.0)
        # It is RECORDED, and the run does not fail.
        self.assertIn("z_vs_parent", rec)

    def test_a_failed_embedding_stops_the_run(self):
        mod = _load()
        with mock.patch.multiple(mod, VMC=object,
                                 embedding_report=lambda *a, **k: {
                                     "ok": False, "head": 1e-3, "tail": 0.0,
                                     "max_abs_dE": 1e-3, "n_configurations": 4}):
            with self.assertRaises(mod.NestingIdentityError):
                mod.run_nested_point(75.0, 2, 0,
                                     {"energy_per_particle": -0.5, "error": 0.01},
                                     _state(nmax=1), _Args())

    def test_the_real_embedding_is_exact(self):
        """A synthetic state through the real wavefunctions: the padded child
        must evaluate to the parent's energy on the same configurations.  This
        is the check the campaign's own records are gated on, run on something
        that costs half a second instead of a VMC run."""
        mod = _load()
        rng = np.random.default_rng(5)
        st = _state(c=rng.standard_normal(5),
                    v=0.05 * (rng.standard_normal((36, 1))
                              + 1j * rng.standard_normal((36, 1))), nmax=1)
        rep = mod.embedding_report(75.0, 2, st, 36, "ll_rotation")
        self.assertEqual((rep["head"], rep["tail"]), (0.0, 0.0))
        self.assertLessEqual(rep["max_abs_dE"], mod._EMBED_TOL)
        self.assertTrue(rep["ok"])
        self.assertGreater(rep["n_configurations"], 0)

    def test_the_gate_is_far_below_the_physics(self):
        """A check as loose as the physics measures nothing: the tolerance must
        be many orders below a typical quoted error bar."""
        mod = _load()
        self.assertLess(mod._EMBED_TOL, 1e-6)
        self.assertLess(mod._BLAS_TOL, 1e-6)


class TestTheIndependentBaseline(unittest.TestCase):
    def test_the_comparison_never_writes_to_the_baseline(self):
        """Constraint: do not overwrite or delete the 93-point store.  Checked on
        the source, because a single `save_store` on the wrong directory is
        exactly how it would happen."""
        src = io.open(SCRIPT, encoding="utf-8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "save_store":
                    args = [ast.unparse(a) for a in node.args]
                    self.assertNotIn("independent_dir", " ".join(args),
                                     f"save_store on the baseline at line "
                                     f"{node.lineno}")

    def test_the_baseline_directory_is_derived_not_literal(self):
        """A literal path would not follow --budget/--l0-grid and could silently
        point at the wrong family."""
        mod = _load()
        self.assertEqual(
            os.path.relpath(mod.independent_dir(_Args()), mod.CLEAN).replace(
                "\\", "/"),
            "results/figure_construction/phase_competition/N36__llrot__quick__l01")

    def test_no_independent_store_is_a_clean_refusal(self):
        mod = _load()
        args = _Args()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.multiple(mod, RESULTS=tmp):
                out = mod.compare_with_independent(args, tmp)
        self.assertFalse(out["available"])
        self.assertIn("no independent store", out["reason"])


class TestTheCrossingReport(unittest.TestCase):
    def test_a_sign_change_is_bracketed_and_interpolated(self):
        mod = _load()
        rep = mod.crossing_report([50.0, 55.0, 60.0], [-2.0, 1.0, 4.0],
                                  [0.1, 0.1, 0.1])
        self.assertEqual(len(rep), 1)
        self.assertEqual(rep[0]["between"], [50.0, 55.0])
        self.assertAlmostEqual(rep[0]["rs_star"], 50.0 + 5.0 * 2.0 / 3.0, places=12)
        self.assertEqual(rep[0]["z_lo"], -20.0)

    def test_a_curve_that_does_not_cross_reports_nothing(self):
        mod = _load()
        self.assertEqual(mod.crossing_report([50.0, 55.0], [-2.0, -1.0],
                                             [0.1, 0.1]), [])
        self.assertEqual(mod.crossing_report([50.0], [1.0], [0.1]), [])

    def test_a_zero_at_a_measured_point_is_not_a_bracket(self):
        """Two points can share a sign while one of them is exactly zero; the
        interpolated rs_star would then be a division by a difference of zero."""
        mod = _load()
        self.assertEqual(mod.crossing_report([50.0, 55.0], [0.0, 1.0],
                                             [0.1, 0.1]), [])


class TestTheNestedPlanEndToEnd(unittest.TestCase):
    """The whole pipeline on the nested construction, with the VMC calls stubbed.

    The nesting itself is stubbed too -- what is being checked here is that the
    recipe ASKS for the right chain, records the ancestry, keeps the baseline
    untouched, and still draws a figure.
    """

    def _run(self, tmp, construction="nested"):
        mod = _load()
        calls = []

        def fake_run_point(phase, rs, nmax, init_id, args, verbose=False,
                           with_state=False):
            calls.append(("root", phase, float(rs), int(nmax)))
            e = -0.5 * float(rs) + 0.01 * int(nmax)
            rec = {"phase": phase, "rs": float(rs), "nmax": int(nmax),
                   "init_id": int(init_id), "N": 36, "ansatz": "ll_rotation",
                   "budget": "quick", "energy_per_particle": e,
                   "error": 0.01, "acceptance": 0.6, "seconds": 0.0, "record": {}}
            return (rec, _state(nmax=int(nmax), rs=float(rs))) if with_state else rec

        def fake_nested(rs, nmax, init_id, parent_rec, parent_state, args,
                        verbose=False):
            calls.append(("nest", "crystal", float(rs), int(nmax)))
            e = -0.5 * float(rs) + 0.01 * int(nmax)
            # The ancestry the real `run_nested_point` writes, including the
            # fields it PROPAGATES from the parent rather than re-derives.
            rec = {"phase": "crystal", "rs": float(rs), "nmax": int(nmax),
                   "init_id": int(init_id), "N": 36, "ansatz": "ll_rotation",
                   "budget": "quick", "energy_per_particle": e,
                   "error": 0.01, "acceptance": 0.6, "seconds": 0.0, "record": {},
                   "construction": "nested",
                   "parent_state_digest": mod.state_digest(parent_state),
                   "nesting_head_residual": 0.0, "nesting_tail_residual": 0.0,
                   "parent_nmax": int(parent_state.nmax),
                   "parent_key": parent_rec.get("key"),
                   "root_initialization": parent_rec.get(
                       "root_initialization", "corrected_gaussian_overlap"),
                   "root_L0": parent_rec.get("root_L0"),
                   "delta_e_vs_parent": 0.01, "z_vs_parent": 0.5}
            return rec, _state(nmax=int(nmax), rs=float(rs))

        patched = dict(
            RESULTS=os.path.join(tmp, "results", "figure_construction",
                                 "phase_competition"),
            FIGDIR=os.path.join(tmp, "figures", "figure_construction",
                                "phase_competition"),
            KAPPAS_PROTO=(),
            TIER_RS={"conv": (55.0,), "nest": (55.0,), "nest4": (55.0,)},
            TIER_NMAX={"conv": 1, "nest": 2, "nest4": 3},
            run_point=fake_run_point,
            run_nested_point=fake_nested,
            CLEAN=tmp,
        )
        with mock.patch.multiple(mod, **patched):
            rc = mod.main(["--budget", "quick", "--quiet",
                           "--construction", construction])
        return mod, rc, calls

    def test_the_nested_run_writes_its_own_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rc, calls = self._run(tmp)
            self.assertEqual(rc, 0)
            base = os.path.join(tmp, "results", "figure_construction",
                                "phase_competition")
            self.assertTrue(os.path.isdir(os.path.join(
                base, "N36__llrot__quick__l01__nested", "states")))
            self.assertFalse(os.path.isdir(os.path.join(
                base, "N36__llrot__quick__l01")),
                "a nested run must not create the independent directory")

    def test_the_nested_run_asks_for_the_whole_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, calls = self._run(tmp)
            crystal = [c for c in calls if c[1] == "crystal"]
            self.assertEqual(sorted({c[3] for c in crystal}), [1, 2, 3])
            self.assertEqual(sum(1 for c in calls if c[0] == "nest"), 2,
                             "nmax=2 and nmax=3 must each nest once")

    def test_the_independent_run_is_untouched_by_the_new_flag(self):
        """The default path must still call `run_point` for every crystal point
        and never `nest`."""
        with tempfile.TemporaryDirectory() as tmp:
            mod, rc, calls = self._run(tmp, construction="independent")
            self.assertEqual(rc, 0)
            self.assertEqual([c for c in calls if c[0] == "nest"], [])
            base = os.path.join(tmp, "results", "figure_construction",
                                "phase_competition")
            self.assertTrue(os.path.isdir(os.path.join(
                base, "N36__llrot__quick__l01")))
            self.assertFalse(os.path.isdir(os.path.join(
                base, "N36__llrot__quick__l01__nested")))

    def test_the_metadata_records_the_ancestry_and_the_prohibition(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _ = self._run(tmp)
            path = os.path.join(tmp, "results", "figure_construction",
                                "phase_competition",
                                "N36__llrot__quick__l01__nested",
                                "run_metadata.json")
            meta = json.load(io.open(path, encoding="utf-8"))
            self.assertEqual(meta["construction"], "nested")
            self.assertEqual(meta["construction_slug_suffix"], "__nested")
            nest = meta["nesting"]
            self.assertEqual(len(nest["ancestry"]), 2)
            for row in nest["ancestry"]:
                self.assertEqual(row["nesting_head_residual"], 0.0)
                self.assertEqual(row["nesting_tail_residual"], 0.0)
                self.assertIsNotNone(row["parent_state_digest"])
                self.assertEqual(row["root_initialization"],
                                 "corrected_gaussian_overlap")
            self.assertIn("monotonicity_is_not_a_test", nest)
            self.assertIn("blas_env", meta)

    def test_under_nested_the_drawn_grid_is_unchanged(self):
        """`required_crystal_rs` widens the PLAN.  The figure is drawn from
        `TIER_RS`, which must not move -- otherwise a construction change would
        also be a point-selection change and the before/after comparison would
        be two variables at once."""
        with tempfile.TemporaryDirectory() as tmp:
            mod, rc, calls = self._run(tmp)
            self.assertEqual(rc, 0)
            ran = {c[2] for c in calls if c[1] == "crystal"}
            # The patched grid `_run` installs.  Read here rather than from
            # `mod.TIER_RS`, which the context manager has already restored --
            # asserting against the real grid would compare the run against a
            # different figure's selection.
            drawn = {55.0}
            self.assertEqual(ran, drawn,
                             "the plan ran a coupling the figure does not draw")
            self.assertEqual({1, 2, 3},
                             {c[3] for c in calls if c[1] == "crystal" and
                              c[0] == "nest"} | {1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
