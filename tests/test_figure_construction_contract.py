"""The figure-construction recipe's contract: architecture, provenance, wiring.

``examples/figure_construction/structure_slice.py`` is a user workflow, not a
figure module and not physics.  What has to be true of it is therefore neither
"it draws the right picture" nor "it computes the right energy" -- both of those
are checked where they belong -- but the four claims that make it a recipe
rather than a second engine:

1. **It goes through the public API.**  The recipe may import ``wigner_vmc`` and
   ``wigner_vmc.analysis``; it may not import the engine's own layers.  A recipe
   that can reach ``wigner_vmc.vmc.sampler`` can grow its own Metropolis loop,
   and then there are two engines.

2. **The clean path does not read the historical tree.**  ``_diag`` may be named
   in exactly one place, the ``--compare`` report, and that report runs after
   every clean number is already saved.  The check is structural: the module
   constant that names the frozen store must be referenced from
   ``compare_with_legacy`` and nowhere else, so a future edit cannot quietly
   route a figure through it.

3. **The coupling is ``kappa = r_s/sqrt(2)``**, so ``4*sqrt(2)/sqrt(2) == 4``
   exactly.  This is the one identity that distinguishes the physical coupling
   from the Stage-2E rounded-regression value, and it is asserted as equality
   rather than as a tolerance because that is how the geometry defines it.

4. **The API's own ``S(q)`` is on the right torus.**  ``RunState.structure_factor``
   builds a supercell whose area is ``N * 2*pi``; the primitive cell's ``2*pi``
   is a different object that produces six momenta instead of 294 and an ``S(q)``
   array nobody would notice was six points long.  That regression is pinned
   here, against a hand-built state, so no VMC run is needed to catch it.

Deliberately NOT here: any test that runs a full production VMC.  The end-to-end
check uses the cheapest budget that exercises every stage -- clean VMC, samples,
observables, numbers, figures -- and the fast tests use synthetic snapshots.
"""
import ast
import importlib.util
import math
import os
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SCRIPT = os.path.join(CLEAN, "examples", "figure_construction", "structure_slice.py")

#: The engine layers a recipe must not reach.  `wigner_vmc` and
#: `wigner_vmc.analysis` are the public surface and are NOT in this list.
FORBIDDEN_IMPORTS = (
    "wigner_vmc.vmc", "wigner_vmc.physics", "wigner_vmc.wavefunctions",
    "wigner_vmc.optimization",
)


def _load():
    """Import the recipe by path.  Module level only -- it defines constants and
    imports the API; it runs no VMC until a function is called."""
    spec = importlib.util.spec_from_file_location("structure_slice_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree():
    with open(SCRIPT, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=SCRIPT)


def _enclosing_functions(tree):
    """[(func_node, [names loaded anywhere inside it])], innermost only.

    Built by walking each function's own body without descending into nested
    ``def``s, so ``LEGACY_STORE`` used inside a nested closure still counts as
    used by the outer function -- which is the honest attribution.
    """
    out = []

    def walk(node, fname):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((child, fname))
                walk(child, child.name)
            else:
                walk(child, fname)

    out.append((tree, "<module>"))
    walk(tree, "<module>")

    result = []
    for node, _ in out:
        names = []
        stack = list(ast.iter_child_nodes(node))
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue                      # attributed to the nested function
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.append(n.id)
            if isinstance(n, ast.Attribute):
                names.append(n.attr)
            stack.extend(ast.iter_child_nodes(n))
        result.append((node, names))
    return result


class TestArchitecture(unittest.TestCase):
    def test_recipe_exists(self):
        self.assertTrue(os.path.exists(SCRIPT), f"missing {SCRIPT}")

    def test_imports_only_the_public_surface(self):
        bad = []
        for node in ast.walk(_tree()):
            if isinstance(node, ast.ImportFrom) and node.module:
                name = node.module
            elif isinstance(node, ast.Import):
                name = ",".join(a.name for a in node.names)
            else:
                continue
            for forbidden in FORBIDDEN_IMPORTS:
                if name == forbidden or name.startswith(forbidden + "."):
                    bad.append(f"line {node.lineno}: imports {name!r}")
        self.assertEqual(bad, [], "the recipe must use the public API only:\n  "
                         + "\n  ".join(bad))

    def test_no_engine_symbol_is_reimplemented(self):
        """A recipe must not carry its own physics.  Named after the functions it
        would have to copy if it did."""
        own = {f.name for f, _ in _enclosing_functions(_tree())
               if isinstance(f, ast.FunctionDef)}
        banned = {"sample", "local_energy", "metropolis", "sr_optimize_jastrow",
                  "sr_optimize_joint", "coulomb_energy", "structure_factor",
                  "pair_correlation", "gaussian_sites", "allowed_momenta"}
        self.assertEqual(own & banned, set(),
                         f"the recipe defines engine-shaped functions: "
                         f"{sorted(own & banned)}")

    def test_frozen_store_is_named_only_in_the_compare_path(self):
        """``LEGACY_STORE`` -- and therefore ``_diag`` -- is reachable from one
        function, and it is the one that runs after the clean numbers exist."""
        users = []
        for node, names in _enclosing_functions(_tree()):
            if "LEGACY_STORE" in names:
                users.append(getattr(node, "name", "<module>"))
        self.assertEqual(sorted(set(users)), ["compare_with_legacy"],
                         f"LEGACY_STORE is referenced from {sorted(set(users))}; it must "
                         f"be referenced only from compare_with_legacy")

    def test_the_clean_path_does_no_file_io(self):
        """``run_point`` and ``observables`` compute; they do not read.  A recipe
        that loaded a cached array inside ``observables`` would look identical
        from the outside and would silently break the provenance rule."""
        offenders = []
        for node, names in _enclosing_functions(_tree()):
            fname = getattr(node, "name", "<module>")
            if fname not in ("run_point", "observables"):
                continue
            for bad in ("open", "load", "loadtxt", "pickle", "loads", "loadz"):
                if bad in names:
                    offenders.append(f"{fname} calls {bad}()")
        self.assertEqual(offenders, [], "\n  ".join(offenders))

    def test_results_and_figures_leave_the_example_directory(self):
        mod = _load()
        examples_dir = os.path.normcase(os.path.dirname(SCRIPT))
        for label, path, parent in (("RESULTS", mod.RESULTS, "results"),
                                    ("FIGDIR", mod.FIGDIR, "figures")):
            self.assertNotEqual(
                os.path.normcase(os.path.commonpath([path, examples_dir])), examples_dir,
                f"{label} must NOT be inside examples/ -- generated numbers and PNGs "
                f"belong under {parent}/, but it is {path}")
            self.assertTrue(path.startswith(os.path.join(CLEAN, parent)),
                            f"{label} must be under {parent}/, got {path}")
            self.assertIn(os.path.join("figure_construction", "structure_slice"), path)


class TestCoupling(unittest.TestCase):
    """The physical coupling, asserted as exact equality -- that IS the point."""

    def test_kappa_to_rs_identity(self):
        mod = _load()
        self.assertEqual(mod.KAPPA_LIQUID, 4.0)
        self.assertEqual(mod.KAPPA_CRYSTAL, 64.0)
        self.assertEqual(mod.RS_LIQUID / math.sqrt(2.0), 4.0)
        self.assertEqual(mod.RS_CRYSTAL / math.sqrt(2.0), 64.0)
        self.assertEqual(mod.RS_LIQUID, 4.0 * math.sqrt(2.0))
        self.assertEqual(mod.RS_CRYSTAL, 64.0 * math.sqrt(2.0))

    def test_default_budget_is_the_cheap_one(self):
        """A bare invocation must not start a production run by accident."""
        mod = _load()
        args = mod.parse_args([])
        self.assertEqual(args.budget, "quick")
        self.assertFalse(args.compare, "--compare reads the historical tree; it is opt-in")
        self.assertFalse(args.resume)


class TestReproductionBudget(unittest.TestCase):
    """The budget is a transcription of the notebook, so its numbers are pinned
    here rather than remembered.  A drift in one of them would silently move the
    protocol the figure claims to reproduce."""

    def test_protocol_matches_the_notebook(self):
        from wigner_vmc import load_budget
        bud = load_budget("reproduction")
        self.assertEqual(bud.protocol["sr_steps"], 8)
        self.assertEqual(bud.protocol["sr_sweeps"], 150)
        self.assertEqual(bud.protocol["sr_equil"], 50)
        self.assertEqual(bud.protocol["sr_snap"], 3)
        self.assertEqual(bud.protocol["meas_sweeps"], 1500)
        self.assertEqual(bud.protocol["meas_equil"], 500)
        self.assertEqual(bud.crystal_measure["seed"], 13)
        self.assertEqual(bud.crystal_measure["sigma"], 0.2)
        self.assertEqual(bud.liquid_measure["seed"], 11)
        self.assertEqual(bud.liquid_measure["sigma"], 0.4)

    def test_init_id_zero_is_the_width_fig06_used(self):
        """``L0_best = L0_optJ[KAPPAS.index(64.0)]`` -> 0.40, the width the frozen
        ``crystalJ_k64_L0.4`` checkpoint records."""
        from wigner_vmc import load_budget
        mod = _load()
        self.assertEqual(mod.CRYSTAL_INIT_ID, 0)
        self.assertEqual(load_budget("reproduction").width_for(mod.CRYSTAL_INIT_ID), 0.40)

    def test_the_full_alias_resolves_to_this_budget(self):
        """``--budget full`` is a SPELLING of ``reproduction``, not a budget.

        It is what a reader reaches for when they mean "the real statistics",
        and it has to name the same config -- otherwise the alias would mint a
        second directory holding the identical run, and ``is_historical_request``
        (which decides who may write to the canonical ``structure_slice/``)
        would stop recognising the historical request whenever it was spelled
        ``full``.

        Both halves are asserted, because they are why ``main`` resolves the
        alias before anything reads ``args.budget``: the resolution, and the
        fact that the unresolved spelling is a DIFFERENT request.  Resolving
        late would fork the output directory *and* lose the canonical namespace.
        """
        from wigner_vmc import load_budget, resolve_budget_name
        mod = _load()
        self.assertEqual(resolve_budget_name("full"), "reproduction")
        self.assertEqual(load_budget("full").name, "reproduction")
        # An alias is not a config: a budget that does not exist must still be
        # an error rather than silently passing through as itself.
        self.assertEqual(resolve_budget_name("no-such-budget"), "no-such-budget")
        hist = (mod.RS_LIQUID, mod.RS_CRYSTAL, mod.DEFAULT_ANSATZ,
                mod.DEFAULT_CRYSTAL_NMAX, mod.HISTORICAL_BUDGET)
        resolved = hist[:4] + (resolve_budget_name("full"),)
        self.assertTrue(mod.is_historical_request(*resolved))
        self.assertEqual(mod.namespace("b", *resolved), "b")
        self.assertEqual(mod.namespace("b", *resolved), mod.namespace("b", *hist))
        # ... while the raw spelling is not that request, so a late resolution
        # would both fork the directory and lose the canonical namespace.
        self.assertNotEqual(
            mod.namespace("b", mod.RS_LIQUID, mod.RS_CRYSTAL, mod.DEFAULT_ANSATZ,
                          mod.DEFAULT_CRYSTAL_NMAX, "full"),
            "b")


class TestStructureFactorTorusRegression(unittest.TestCase):
    """``RunState.structure_factor`` must measure on the SUPERCELL.

    The defect this pins: building the torus with the primitive cell's area
    (``2*pi``) instead of ``N * 2*pi`` gave ``|G1| = 2.6935`` and
    ``allowed_momenta(4.0)`` returned SIX momenta, so ``S(q)`` was a six-point
    array that still satisfied every shape assertion anyone had written.
    """

    def _state(self, n_snaps=2, seed=0):
        from wigner_vmc.api import RunState
        rng = np.random.default_rng(seed)
        snaps = [rng.random((36, 2)) * 8.0 for _ in range(n_snaps)]
        return RunState(R=snaps[-1], c=np.zeros(5), v=np.zeros((36, 1), complex),
                        N=36, rs=75.0, kappa=75.0 / math.sqrt(2.0),
                        kappa_mode="physical", nmax=1, phase="liquid", snaps=snaps)

    def test_supercell_momenta_not_primitive_cell_momenta(self):
        out = self._state().structure_factor()
        qn = out["qn"]
        self.assertEqual(len(qn), 294,
                         "the supercell at N=36, q_max=4 has 294 allowed momenta; a "
                         "primitive-cell torus returns 6")
        self.assertEqual(len(out["S"]), len(qn))
        self.assertAlmostEqual(float(qn.min()), 0.44892456236286615, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(out["torus"].G1)),
                               0.44892456236286615, places=12)
        self.assertAlmostEqual(out["torus"].area, 72.0 * math.pi, places=9)

    def test_the_recipe_measures_on_the_same_torus(self):
        """The recipe builds its own Torus; it must be the API's, bit for bit."""
        mod = _load()
        a = mod.torus_for()
        b = self._state().structure_factor()["torus"]
        self.assertEqual(a.area, b.area)
        np.testing.assert_array_equal(a.G1, b.G1)
        np.testing.assert_array_equal(a.allowed_momenta(mod.Q_MAX)[1],
                                      b.allowed_momenta(mod.Q_MAX)[1])


class TestArtifactsAreWrittenAndRead(unittest.TestCase):
    """The wiring, on synthetic numbers: the expected files appear, in the
    expected directories, and the figures are drawn from the arrays that were
    just written rather than from anything else."""

    def _points(self):
        rng = np.random.default_rng(7)
        out = {}
        for phase, rs in (("liquid", 4.0 * math.sqrt(2.0)), ("crystal", 64.0 * math.sqrt(2.0))):
            snaps = [rng.random((36, 2)) * 8.0 for _ in range(6)]
            out[phase] = {
                "phase": phase, "rs": rs, "kappa": rs / math.sqrt(2.0),
                "kappa_mode": "physical", "nmax": 1, "init_id": 0, "init_L0": 0.4,
                "ansatz": "ll_rotation", "optimized": "orbitals+jastrow",
                "budget": "smoke", "rng_seed": 0, "energy_per_particle": -0.5,
                "error": 0.01, "acceptance": 0.4, "n_snapshots": len(snaps),
                "sr_steps": 2, "sr_seconds": 1.0, "seconds": 2.0,
                "snaps": snaps, "R": snaps[-1],
            }
        return out

    def test_expected_files_appear_in_the_expected_directories(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as td:
            res, fig = os.path.join(td, "results"), os.path.join(td, "figures")
            points = self._points()
            torus = mod.torus_for()
            obs = {k: mod.observables(points[k]["snaps"], torus) for k in points}

            written = mod.write_figures(fig, points, obs, "quick")
            names = {os.path.basename(p) for p in written}
            self.assertEqual(names, {"structure_factor.png", "pair_correlation.png",
                                     "combined.png"})
            for p in written:
                self.assertTrue(os.path.getsize(p) > 0, p)

            for phase in ("liquid", "crystal"):
                mod.save_point(res, phase, points[phase],
                               mod._config_key(phase, points[phase]["rs"], 0, "quick"))
            mod.save_merged(res, points, obs, {"budget": "quick"})
            for f in ("structure_factor.npz", "pair_correlation.npz",
                      "real_space_snapshot.npz", "run_metadata.json"):
                self.assertTrue(os.path.exists(os.path.join(res, f)), f)
            for phase in ("liquid", "crystal"):
                self.assertTrue(os.path.exists(os.path.join(res, phase, "run.npz")))
                self.assertTrue(os.path.exists(os.path.join(res, phase, "run.json")))

            with np.load(os.path.join(res, "structure_factor.npz")) as z:
                self.assertEqual(z["S_liquid"].shape, z["qn"].shape)
                self.assertEqual(z["S_crystal"].shape, z["qn"].shape)
            with np.load(os.path.join(res, "real_space_snapshot.npz")) as z:
                self.assertEqual(z["R"].shape, (36, 2))
                self.assertEqual(z["lattice_sites"].shape, (36, 2))

    def test_quick_figures_are_marked_as_such(self):
        """A quick figure that looked like a production one would be the easiest
        possible way to mislead -- so the mark is required, not a nicety.

        Asserted the only way that means anything: the SAME numbers rendered at
        the two budgets must come out as different images, and the quick render
        must be deterministic.  "The file exists and has finite pixels" would
        pass just as well with no stamp at all.
        """
        import matplotlib.pyplot as plt
        mod = _load()
        points = self._points()
        obs = {k: mod.observables(points[k]["snaps"], mod.torus_for())
               for k in points}
        with tempfile.TemporaryDirectory() as td:
            imgs = {}
            for budget in ("quick", "quick_again", "reproduction"):
                d = os.path.join(td, budget)
                mod.write_figures(d, points, obs, budget)
                imgs[budget] = {}
                for name in ("structure_factor.png", "pair_correlation.png",
                             "combined.png"):
                    a = plt.imread(os.path.join(d, name))
                    self.assertTrue(np.isfinite(a).all(), name)
                    imgs[budget][name] = a
        for name in imgs["quick"]:
            np.testing.assert_array_equal(
                imgs["quick"][name], imgs["quick_again"][name],
                f"{name} is not deterministic for the same inputs")
            self.assertFalse(
                np.array_equal(imgs["quick"][name], imgs["reproduction"][name]),
                f"{name} is identical at --budget quick and --budget reproduction: "
                f"the QUICK stamp is not being drawn")
            self.assertGreater(imgs["quick"][name].shape[0], 100, name)

    def test_resume_refuses_a_mismatched_configuration(self):
        """Reuse is gated on the config digest, so a resume cannot put one
        state's label on another state's data."""
        mod = _load()
        points = self._points()
        key = mod._config_key("liquid", mod.RS_LIQUID, 0, "quick")
        with tempfile.TemporaryDirectory() as td:
            mod.save_point(td, "liquid", points["liquid"], key)
            self.assertIsNotNone(mod.load_point(td, "liquid", key))
            for other in (mod._config_key("liquid", mod.RS_LIQUID, 0, "reproduction"),
                          mod._config_key("liquid", mod.RS_CRYSTAL, 0, "quick"),
                          mod._config_key("crystal", mod.RS_LIQUID, 0, "quick"),
                          mod._config_key("liquid", mod.RS_LIQUID, 1, "quick")):
                self.assertIsNone(mod.load_point(td, "liquid", other),
                                  f"resumed a run that does not match {other}")


class TestSourceIdentifier(unittest.TestCase):
    """The metadata's source identifier must exist, and must not be invented.

    The task asks for a "source/git identifier if available".  This checkout has
    no git, so the honest answer is to say so.  An empty field is
    indistinguishable from a commit id that got dropped, and a fabricated one is
    worse than either, so the absence itself is asserted.
    """

    def test_the_source_identifier_is_recorded_and_is_not_invented(self):
        mod = _load()
        sid = mod._source_id()
        self.assertIsInstance(sid, str)
        self.assertTrue(sid.strip(), "run_metadata.json would record an empty source")
        self.assertIn(mod.__version__, sid)
        if os.path.isdir(os.path.join(mod.CLEAN, ".git")):
            self.assertTrue(sid.startswith("git:"), sid)
        else:
            self.assertIn("not a git repository", sid)


class TestEndToEnd(unittest.TestCase):
    """One real run through every stage, on the cheapest budget that has one.

    This is not a physics check -- the numbers are unconverged and nothing is
    asserted about them.  What is asserted is that the chain is CONNECTED: a
    cold start from ``(N, r_s)``, a walk, snapshots, observables, numbers on
    disk, and PNGs drawn from those numbers.

    It uses ``smoke`` rather than ``quick`` purely for cost: ``--budget quick``
    is the same code path and takes ~90 s, which is too much to add to the
    default suite.  ``RESULTS``/``FIGDIR`` are redirected into a temp directory,
    so a test run cannot overwrite the recipe's real outputs.
    """

    def test_smoke_budget_produces_every_declared_artifact(self):
        import json
        mod = _load()
        with tempfile.TemporaryDirectory() as td:
            mod.RESULTS = os.path.join(td, "results")
            mod.FIGDIR = os.path.join(td, "figures")
            self.assertEqual(mod.main(["--budget", "smoke", "--quiet"]), 0)

            # `smoke` is a different protocol from the delivered `reproduction`,
            # so this run must have gone to its own suffixed directory rather
            # than over the canonical structure_slice/.
            slug = mod.request_slug(mod.RS_LIQUID, mod.RS_CRYSTAL, "ll_rotation",
                                    mod.DEFAULT_CRYSTAL_NMAX, "smoke")
            res = mod.namespace(mod.RESULTS, mod.RS_LIQUID, mod.RS_CRYSTAL,
                                "ll_rotation", mod.DEFAULT_CRYSTAL_NMAX, "smoke")
            self.assertTrue(os.path.isdir(res), f"expected the namespaced dir {res}")
            self.assertFalse(os.path.exists(mod.RESULTS),
                             "a smoke run wrote straight into the canonical "
                             "structure_slice/")

            for f in ("structure_factor.npz", "pair_correlation.npz",
                      "real_space_snapshot.npz", "run_metadata.json"):
                self.assertTrue(os.path.exists(os.path.join(res, f)), f)
            for f in ("structure_factor.png", "pair_correlation.png",
                      "combined.png"):
                self.assertTrue(os.path.exists(
                    os.path.join(mod.FIGDIR + "_" + slug, f)), f)
            for phase in ("liquid", "crystal"):
                for f in ("run.npz", "run.json"):
                    self.assertTrue(os.path.exists(os.path.join(res, phase, f)),
                                    f"{phase}/{f}")

            with open(os.path.join(res, "run_metadata.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["budget"], "smoke")
            self.assertEqual(meta["quality"], "QUICK / not publication quality")
            self.assertEqual(meta["N"], 36)

            # The resolved REQUEST, recorded at full precision.
            self.assertEqual(meta["request_slug"], slug)
            self.assertFalse(meta["request_is_the_historical_fig06"],
                             "a smoke run is not the delivered reproduction")
            self.assertEqual(meta["liquid_rs"], mod.RS_LIQUID)
            self.assertEqual(meta["crystal_rs"], mod.RS_CRYSTAL)
            self.assertEqual(meta["crystal_nmax"], 1)
            self.assertEqual(meta["crystal_n_bands"], 2)
            self.assertEqual(meta["crystal_ll_indices"], [0, 1])
            self.assertIn("recipe default", meta["liquid_rs_source"])
            self.assertIn("recipe default", meta["crystal_rs_source"])
            self.assertIn("recipe default", meta["crystal_nmax_source"])
            self.assertIn("not a git repository", meta["source_id"])
            self.assertEqual(meta["optimization_protocol"]["sr_steps"],
                             meta["budget_protocol"]["sr_steps"])
            self.assertEqual(meta["production_protocol"]["sweeps"],
                             meta["budget_protocol"]["meas_sweeps"])

            # Per phase, the parameters are named in that phase's own terms.
            liq, cry = meta["liquid"], meta["crystal"]
            self.assertEqual(liq["orbital_ansatz"], "filled_lll_fixed")
            self.assertFalse(liq["orbital_sr"])
            self.assertTrue(liq["jastrow_sr"])
            self.assertEqual(liq["ll_indices"], [0])
            self.assertFalse(liq["nmax_is_a_physical_parameter"])
            self.assertEqual(meta["liquid_nmax_is_a_physical_parameter"], False)

            self.assertEqual(cry["orbital_ansatz"], "ll_rotation")
            self.assertTrue(cry["orbital_sr"])
            self.assertTrue(cry["jastrow_sr"])
            self.assertEqual(cry["ll_indices"], [0, 1])
            self.assertEqual(cry["n_bands"], 2)
            self.assertTrue(cry["nmax_is_a_physical_parameter"])

            for phase in ("liquid", "crystal"):
                self.assertGreater(meta[phase]["n_snapshots"], 0)
                self.assertEqual(meta[phase]["nmax"], 1)
                self.assertEqual(meta[phase]["kappa_mode"], "physical")
                self.assertAlmostEqual(meta[phase]["rs"] / math.sqrt(2.0),
                                       meta[phase]["kappa"], places=12)

            # the plotted arrays are the ones on disk, not objects from the run
            with np.load(os.path.join(res, "structure_factor.npz")) as z:
                self.assertEqual(set(z.files),
                                 {"q", "qn", "S_liquid", "S_crystal",
                                  "S_filled_LLL_exact", "g1_mag", "G1_mag",
                                  "kappa_liquid", "kappa_crystal"})
                self.assertEqual(len(z["qn"]), 294)
            with np.load(os.path.join(res, "pair_correlation.npz")) as z:
                self.assertEqual(z["g_liquid"].shape, (40,))
                self.assertEqual(z["r"].shape, (40,))

    def test_a_failed_budget_is_an_error_not_a_fallback(self):
        """The one behaviour that would quietly turn this recipe into a redraw:
        a clean run that fails and is replaced by historical numbers."""
        mod = _load()
        with tempfile.TemporaryDirectory() as td:
            mod.RESULTS, mod.FIGDIR = os.path.join(td, "r"), os.path.join(td, "f")
            with self.assertRaises(FileNotFoundError):
                mod.main(["--budget", "definitely_not_a_budget", "--quiet"])
            self.assertFalse(os.path.exists(mod.RESULTS))


class TestAnsatzSelection(unittest.TestCase):
    """The crystal optimisation scope is a named, recorded choice.

    Both choices build the same wavefunction class; they differ in which
    parameters the optimiser may move.  The guard that matters is the exact one:
    under ``ll_rotation_pinned`` the returned orbitals must be bit-identical to
    the pinned seed, because "pinned" is a claim about the STATE, and a
    Jastrow-only optimisation that quietly drifted the orbitals would still
    return a plausible energy and a plausible figure.

    Deliberately NOT claimed here: that either choice reproduces the historical
    fig06 crystal.  Neither does -- that state's ansatz is not implemented in
    this package, and the recipe must not pretend otherwise.
    """

    def test_the_default_ansatz_is_the_packages_own(self):
        """No per-budget ansatz table: there is no ansatz this recipe could
        select to reproduce the historical crystal, so it must not imply one."""
        mod = _load()
        self.assertEqual(mod.DEFAULT_ANSATZ, "ll_rotation")

    def test_ansatz_is_unset_unless_asked_for(self):
        """`None` and "the default value" are different answers, and the
        pre-flight needs to know which one it got: a user who typed the
        historical coupling has still made a choice."""
        mod = _load()
        args = mod.parse_args([])
        self.assertIsNone(args.ansatz)
        self.assertIsNone(args.liquid_rs)
        self.assertIsNone(args.crystal_rs)
        self.assertIsNone(args.crystal_nmax)
        self.assertFalse(hasattr(args, "nmax"),
                         "`--nmax` was replaced by `--crystal-nmax`: the old bare "
                         "name invited reading the crystal's truncation as a "
                         "liquid knob, and there is deliberately no liquid one")
        self.assertEqual(mod.parse_args(["--ansatz", "ll_rotation_pinned"]).ansatz,
                         "ll_rotation_pinned")

    def test_the_historical_ansatz_is_not_offered_as_a_choice(self):
        """A `--ansatz gaussian` would be a promise this package cannot keep."""
        mod = _load()
        with self.assertRaises(SystemExit):
            mod.parse_args(["--ansatz", "gaussian"])

    def test_a_variant_writes_to_its_own_directory(self):
        """Only the historical request may land on the canonical `structure_slice/`.

        Each of the five inputs changes the numbers, so each must be able to
        send a run somewhere else -- otherwise the second request of the day
        silently erases the first one's artifacts.
        """
        mod = _load()
        base = os.path.join("b", "structure_slice")
        hist = (mod.RS_LIQUID, mod.RS_CRYSTAL, "ll_rotation", 1, "reproduction")
        self.assertEqual(mod.namespace(base, *hist), base,
                         "the historical request must keep the canonical "
                         "`structure_slice/`")
        self.assertTrue(mod.is_historical_request(*hist))

        # Every one-input-away request is a different calculation.
        variants = [
            (75.0, mod.RS_CRYSTAL, "ll_rotation", 1, "reproduction"),
            (mod.RS_LIQUID, 75.0, "ll_rotation", 1, "reproduction"),
            (mod.RS_LIQUID, mod.RS_CRYSTAL, "ll_rotation_pinned", 1, "reproduction"),
            (mod.RS_LIQUID, mod.RS_CRYSTAL, "ll_rotation", 2, "reproduction"),
            (mod.RS_LIQUID, mod.RS_CRYSTAL, "ll_rotation", 1, "quick"),
        ]
        seen = {base}
        for v in variants:
            path = mod.namespace(base, *v)
            self.assertNotEqual(path, base, f"{v} must not overwrite structure_slice/")
            self.assertNotIn(path, seen, f"{v} collides with another request")
            self.assertTrue(path.startswith(base + "_"), path)
            seen.add(path)
            self.assertFalse(mod.is_historical_request(*v))

        # The slug names the physics, and carries the couplings the user chose.
        slug = mod.request_slug(*variants[0])
        self.assertIn("liquid_rs75", slug)
        self.assertIn("crystal_rs", slug)
        self.assertIn("llrot_nmax1", slug)
        self.assertIn("reproduction", slug)
        # ... and never a fragile full-precision float.
        self.assertNotIn("5.656854249", slug)

    def test_resume_digest_covers_the_ansatz_and_nmax(self):
        """Two requests differing only in ansatz must not share a saved run."""
        mod = _load()
        rs = 64.0 * math.sqrt(2.0)
        joint = mod._config_key("crystal", rs, 0, "reproduction", ansatz="ll_rotation")
        pinned = mod._config_key("crystal", rs, 0, "reproduction",
                                 ansatz="ll_rotation_pinned")
        deeper = mod._config_key("crystal", rs, 0, "reproduction",
                                 ansatz="ll_rotation", nmax=2)
        self.assertNotEqual(joint, pinned)
        self.assertNotEqual(joint, deeper)

    def test_the_api_rejects_an_unknown_or_mislocated_ansatz(self):
        from wigner_vmc import VMC, api
        self.assertEqual(api.resolve(phase="crystal").ansatz, "ll_rotation")
        self.assertEqual(
            api.resolve(phase="crystal", ansatz="ll_rotation_pinned").ansatz,
            "ll_rotation_pinned")
        with self.assertRaises(ValueError):
            api.resolve(phase="liquid", ansatz="ll_rotation_pinned")
        with self.assertRaises(ValueError):
            api.resolve(phase="crystal", ansatz="not_an_ansatz")
        with self.assertRaises(ValueError):
            VMC(N=36, rs=64.0 * math.sqrt(2.0), phase="liquid",
                ansatz="ll_rotation_pinned")

    def test_nesting_refuses_a_pinned_orbital_state(self):
        """The guard is a receiver precondition, so it must fire before the
        parent is inspected -- hence the deliberately unusable parent."""
        from wigner_vmc import VMC
        with self.assertRaises(ValueError) as ctx:
            VMC(N=36, rs=64.0 * math.sqrt(2.0), phase="crystal",
                ansatz="ll_rotation_pinned").nest(parent=None, new_nmax=2)
        self.assertIn("cannot nest", str(ctx.exception))

    def test_ll_rotation_pinned_leaves_the_orbitals_bit_identically(self):
        """The decisive check, on the state rather than on the energy."""
        from wigner_vmc import VMC, load_budget
        bud = load_budget("smoke")
        rs = 64.0 * math.sqrt(2.0)
        vmc = VMC(N=36, rs=rs, phase="crystal", nmax=1, ansatz="ll_rotation_pinned")
        res = vmc.run(init_id=0, budget="smoke", verbose=False)
        self.assertEqual(res.state.ansatz, "ll_rotation_pinned")
        self.assertEqual(res.optimization.kind, "jastrow")
        self.assertEqual(res.config.ansatz, "ll_rotation_pinned")
        pinned = np.asarray(vmc.crystal_v0(bud.width_for(0)), complex)
        np.testing.assert_array_equal(np.asarray(res.state.v, complex), pinned)


class TestUserChosenParameters(unittest.TestCase):
    """The three physical parameters are the user's, and they reach the physics.

    The failure this class exists to prevent is the quiet one: a flag that only
    relabels a plot.  So each flag is checked at the point where it enters the
    CALCULATION -- the resolved coupling, the basis the wavefunction is built
    from, and the call the recipe makes -- not at the legend.
    """

    def test_the_defaults_are_the_historical_recipe(self):
        mod = _load()
        req = mod.resolve_request(mod.parse_args([]))
        self.assertEqual(req["liquid_rs"], 4.0 * math.sqrt(2.0))
        self.assertEqual(req["crystal_rs"], 64.0 * math.sqrt(2.0))
        self.assertEqual(req["crystal_nmax"], 1)
        self.assertEqual(req["ansatz"], "ll_rotation")
        # Not-given is a distinct answer from given-the-default, and the
        # pre-flight and the metadata both report which one it was.
        self.assertFalse(req["liquid_rs_given"])
        self.assertFalse(req["crystal_rs_given"])
        self.assertFalse(req["crystal_nmax_given"])

    def test_a_given_default_is_recorded_as_given(self):
        """`--liquid-rs 5.656854249492381` is still a user choice."""
        mod = _load()
        req = mod.resolve_request(
            mod.parse_args(["--liquid-rs", repr(4.0 * math.sqrt(2.0))]))
        self.assertEqual(req["liquid_rs"], mod.RS_LIQUID)
        self.assertTrue(req["liquid_rs_given"])

    def test_liquid_rs_changes_the_resolved_liquid_coupling(self):
        mod = _load()
        req = mod.resolve_request(mod.parse_args(["--liquid-rs", "75"]))
        self.assertEqual(req["liquid_rs"], 75.0)
        self.assertEqual(mod._phase_kappa(req["liquid_rs"]), 75.0 / math.sqrt(2.0))
        # ... and only the liquid's: the crystal is untouched.
        self.assertEqual(req["crystal_rs"], mod.RS_CRYSTAL)

    def test_crystal_rs_changes_the_resolved_crystal_coupling(self):
        mod = _load()
        req = mod.resolve_request(mod.parse_args(["--crystal-rs", "75"]))
        self.assertEqual(req["crystal_rs"], 75.0)
        self.assertEqual(mod._phase_kappa(req["crystal_rs"]), 75.0 / math.sqrt(2.0))
        self.assertEqual(req["liquid_rs"], mod.RS_LIQUID)

    def test_crystal_nmax_is_the_highest_landau_index(self):
        """`--crystal-nmax N` -> LL basis n = 0..N, n_bands = N + 1.

        Asserted on the BASIS the wavefunction is actually built from, because
        the historical naming ambiguity (`n_max` meaning a band count) is exactly
        what this would silently reintroduce.
        """
        from wigner_vmc import VMC, api
        mod = _load()
        for n_max in (1, 2, 3):
            req = mod.resolve_request(mod.parse_args(["--crystal-nmax", str(n_max)]))
            self.assertEqual(req["crystal_nmax"], n_max)
            vmc = VMC(N=36, rs=mod.RS_CRYSTAL, phase="crystal", nmax=n_max)
            self.assertEqual(vmc.n_bands, n_max + 1,
                             f"n_max={n_max} must keep {n_max + 1} bands")
            basis = vmc.basis()
            self.assertEqual(basis.nmax, n_max, "basis keeps levels 0 .. n_max")
            self.assertEqual(basis.orbitals(np.zeros((1, 2))).shape, (36, n_max + 1))
            # the shared parameter vector grows with the truncation, because the
            # rotation amplitudes v are per (momentum, higher band)
            v = np.zeros((basis.nk, n_max), complex)
            self.assertEqual(len(api.theta(np.zeros(api.NJ), v)),
                             api.NJ + 2 * n_max * basis.nk)
            self.assertEqual(basis.nk, 36)

    def test_the_liquid_never_receives_the_crystals_truncation(self):
        """`--crystal-nmax` is a crystal knob.  Checked at the CALL, not on the
        label: a liquid handed `nmax=3` would still be the filled LLL here, but
        the forcing is what makes that true by construction rather than by luck.
        """
        mod = _load()
        recorded = {}

        class _Boom(Exception):
            pass

        class _Recorder:
            def __init__(self, **kw):
                recorded.update(kw)
                raise _Boom

        real = mod.VMC
        mod.VMC = _Recorder
        try:
            with self.assertRaises(_Boom):
                mod.run_point("liquid", mod.RS_LIQUID, 0, "smoke", nmax=3)
        finally:
            mod.VMC = real

        self.assertEqual(recorded["phase"], "liquid")
        self.assertEqual(recorded["nmax"], mod.LIQUID_NMAX)
        with self.assertRaises(SystemExit):
            mod.parse_args(["--liquid-nmax", "2"])

    def test_the_crystal_receives_the_users_coupling_and_truncation(self):
        """The other half of the same check -- the flags are not swallowed."""
        mod = _load()
        recorded = {}

        class _Boom(Exception):
            pass

        class _Recorder:
            def __init__(self, **kw):
                recorded.update(kw)
                raise _Boom

        real = mod.VMC
        mod.VMC = _Recorder
        try:
            with self.assertRaises(_Boom):
                mod.run_point("crystal", 75.0, 0, "smoke", nmax=2)
        finally:
            mod.VMC = real

        self.assertEqual(recorded["phase"], "crystal")
        self.assertEqual(recorded["rs"], 75.0)
        self.assertEqual(recorded["nmax"], 2)

    def test_the_default_calculation_key_is_unchanged(self):
        """A hermetic guard for "existing default numerical behaviour is
        unchanged": the exact record that decides what the engine is asked to
        compute.  If any resolved input drifts, this fails before a run starts.
        """
        mod = _load()
        self.assertEqual(mod._phase_kappa(mod.RS_LIQUID), 4.0)
        self.assertEqual(mod._phase_kappa(mod.RS_CRYSTAL), 64.0)
        self.assertEqual(
            mod._config_key("liquid", mod.RS_LIQUID, 0, "reproduction"),
            {"phase": "liquid", "rs": repr(4.0 * math.sqrt(2.0)), "N": 36,
             "nmax": 1, "ansatz": "ll_rotation", "init_id": 0,
             "budget": "reproduction", "point_schema": 2})
        self.assertEqual(
            mod._config_key("crystal", mod.RS_CRYSTAL, mod.CRYSTAL_INIT_ID,
                            "reproduction"),
            {"phase": "crystal", "rs": repr(64.0 * math.sqrt(2.0)), "N": 36,
             "nmax": 1, "ansatz": "ll_rotation",
             "init_id": mod.CRYSTAL_INIT_ID,
             "budget": "reproduction", "point_schema": 2})


class TestThePreflightBlock(unittest.TestCase):
    """The resolved calculation is printed, in each phase's own terms, before
    anything expensive runs.  It is the only place a user can catch a mistyped
    coupling before paying for the SR ladder, so its content is a contract."""

    def _text(self, liquid_rs, crystal_rs, nmax, ansatz="ll_rotation", budget="quick"):
        import contextlib
        import io
        mod = _load()
        bud = mod.load_budget(budget)
        res = mod.namespace(mod.RESULTS, liquid_rs, crystal_rs, ansatz, nmax, budget)
        fig = mod.namespace(mod.FIGDIR, liquid_rs, crystal_rs, ansatz, nmax, budget)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.print_preflight(liquid_rs, crystal_rs, nmax, ansatz, bud, res, fig)
        return mod, buf.getvalue()

    def test_the_liquid_block_claims_no_orbital_sector(self):
        _, text = self._text(75.0, 90.50966799187809, 1)
        self.assertIn("RESOLVED CALCULATION", text)
        self.assertIn("filled LLL (fixed)", text)
        self.assertIn("LL rotation      none", text)
        self.assertIn("orbital SR       disabled", text)
        self.assertIn("Jastrow SR       enabled", text)
        liquid_block = text.split("CRYSTAL")[0]
        self.assertNotIn("n_max", liquid_block,
                         "the liquid block must not present an n_max")
        self.assertNotIn("n_bands", liquid_block)

    def test_the_crystal_block_names_the_basis_it_builds(self):
        _, text = self._text(75.0, 75.0, 2)
        self.assertIn("n_max            2", text)
        self.assertIn("LL basis         n = 0, 1, 2", text)
        self.assertIn("n_bands          3   (= n_max + 1)", text)
        self.assertIn("joint orbital + Jastrow SR", text)

    def test_the_preflight_prints_the_physical_kappa_for_the_users_rs(self):
        _, text = self._text(75.0, 50.0, 1)
        self.assertIn(f"{75.0 / math.sqrt(2.0):.9f}", text)
        self.assertIn(f"{50.0 / math.sqrt(2.0):.9f}", text)
        self.assertNotIn("legacy", text.lower())

    def test_a_nonhistorical_request_says_why_its_directory_differs(self):
        _, text = self._text(75.0, 75.0, 2)
        self.assertIn("This is NOT the request `structure_slice/` holds", text)
        self.assertIn("liquid r_s 75", text)
        self.assertIn("crystal r_s 75", text)
        self.assertIn("crystal n_max 2", text)
        self.assertIn("budget quick", text)

    def test_the_historical_request_is_not_warned_about(self):
        mod = _load()
        _, text = self._text(mod.RS_LIQUID, mod.RS_CRYSTAL, 1,
                             budget="reproduction")
        self.assertNotIn("NOT the request", text)
        self.assertIn("results\\figure_construction\\structure_slice", text)
        self.assertNotIn("structure_slice_", text)


class TestPlotLabels(unittest.TestCase):
    """The legend names the user's parameter, not the internal conversion.

    `kappa` is `r_s/sqrt(2)` and is the same information; worse, two couplings
    a user would type (`50` and `50.0000001`) can share a rounded kappa, so a
    kappa-only label can silently merge two different runs.
    """

    def _points(self, liquid_rs, crystal_rs, nmax=1, ansatz="ll_rotation"):
        return {
            "liquid": {"rs": liquid_rs, "nmax": 1, "ansatz": "ll_rotation"},
            "crystal": {"rs": crystal_rs, "nmax": nmax, "ansatz": ansatz},
        }

    def test_the_labels_name_rs_and_nmax(self):
        mod = _load()
        lab_l, lab_c = mod._labels(self._points(75.0, 75.0, nmax=2))
        self.assertIn("liquid", lab_l)
        self.assertIn("75", lab_l)
        self.assertNotIn("kappa", lab_l)
        self.assertIn("crystal", lab_c)
        self.assertIn("75", lab_c)
        self.assertIn("n_{\\rm max}", lab_c)
        self.assertIn("2", lab_c)
        self.assertNotIn("kappa", lab_c)

    def test_two_truncations_at_one_coupling_labelled_differently(self):
        """The whole point of the nmax sweep: same r_s, different curve, and the
        legend must say which."""
        mod = _load()
        _, one = mod._labels(self._points(75.0, 75.0, nmax=1))
        _, two = mod._labels(self._points(75.0, 75.0, nmax=2))
        self.assertNotEqual(one, two)

    def test_two_couplings_labelled_differently(self):
        mod = _load()
        l50, _ = mod._labels(self._points(50.0, 100.0))
        l75, _ = mod._labels(self._points(75.0, 75.0))
        self.assertNotEqual(l50, l75)

    def test_the_ansatz_appears_only_when_it_is_not_the_default(self):
        mod = _load()
        _, default = mod._labels(self._points(75.0, 75.0))
        _, pinned = mod._labels(self._points(75.0, 75.0, ansatz="ll_rotation_pinned"))
        self.assertNotIn("pinned", default)
        self.assertIn("ll_rotation_pinned", pinned)


class TestInputValidation(unittest.TestCase):
    """An invalid request must die before it costs anything, and say why."""

    def test_the_validator_refuses_what_the_engine_cannot_build(self):
        mod = _load()
        bad = [(-1.0, mod.RS_CRYSTAL, 1), (0.0, mod.RS_CRYSTAL, 1),
               (float("nan"), mod.RS_CRYSTAL, 1), (float("inf"), mod.RS_CRYSTAL, 1),
               (mod.RS_LIQUID, -5.0, 1), (mod.RS_LIQUID, 0.0, 1),
               (mod.RS_LIQUID, mod.RS_CRYSTAL, 0), (mod.RS_LIQUID, mod.RS_CRYSTAL, -1)]
        for liquid_rs, crystal_rs, nmax in bad:
            with self.assertRaises(SystemExit, msg=f"accepted {(liquid_rs, crystal_rs, nmax)}"):
                mod.validate_inputs(liquid_rs, crystal_rs, nmax)
        # ... and accepts the legitimate ones, including the historical pair.
        for good in [(mod.RS_LIQUID, mod.RS_CRYSTAL, 1), (75.0, 75.0, 1),
                     (50.0, 100.0, 2), (1.0, 1.0, 3)]:
            mod.validate_inputs(*good)

    def test_nmax_zero_is_refused_with_the_engines_own_reason(self):
        """The task sketched `>= 0`; the engine's own bound is `>= 1` (its nmax
        is the highest index, not a band count), and the message says so rather
        than failing opaquely."""
        mod = _load()
        with self.assertRaises(SystemExit) as ctx:
            mod.validate_inputs(mod.RS_LIQUID, mod.RS_CRYSTAL, 0)
        self.assertIn("nmax", str(ctx.exception))
        self.assertIn(">= 1", str(ctx.exception))

    def test_an_invalid_request_fails_before_any_calculation(self):
        """Nothing is constructed, nothing is written -- proved by making any
        construction an immediate, unmistakable failure."""
        mod = _load()

        class _Never:
            def __init__(self, *a, **kw):
                raise AssertionError("a VMC was constructed for an invalid request")

        real = mod.VMC
        mod.VMC = _Never
        try:
            with tempfile.TemporaryDirectory() as td:
                mod.RESULTS, mod.FIGDIR = os.path.join(td, "r"), os.path.join(td, "f")
                for argv in (["--liquid-rs", "-1"], ["--crystal-rs", "0"],
                             ["--crystal-nmax", "0"], ["--liquid-rs", "nan"]):
                    with self.assertRaises(SystemExit, msg=f"accepted {argv}"):
                        mod.main(argv + ["--budget", "smoke", "--quiet"])
                self.assertFalse(os.path.exists(mod.RESULTS),
                                 "an invalid request created its output directory")
        finally:
            mod.VMC = real


class TestNamespaceCollisionWarning(unittest.TestCase):
    """Two different requests must not share a directory in silence."""

    def test_a_reused_directory_names_the_request_it_holds(self):
        import json
        mod = _load()
        with tempfile.TemporaryDirectory() as td:
            other = mod.request_slug(mod.RS_LIQUID, mod.RS_CRYSTAL, "ll_rotation", 2,
                                     "reproduction")
            with open(os.path.join(td, "run_metadata.json"), "w", encoding="utf-8") as fh:
                json.dump({"request_slug": other}, fh)
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mod.check_namespace_reuse(td, "something_else")
            text = buf.getvalue()
            self.assertIn("WARNING", text)
            self.assertIn(other, text)

            # The same request re-run says nothing at all.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mod.check_namespace_reuse(td, other)
            self.assertEqual(buf.getvalue(), "")

    def test_a_fresh_directory_says_nothing(self):
        mod = _load()
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mod.check_namespace_reuse(td, "anything")
            self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
