"""The single-point structural diagnostic's contract.

``examples/figure_construction/structure.py`` measures three
structural observables at one ``(r_s^L, r_s^C, n_max^C)`` point.  What has to be
true of it is neither "it draws the right picture" nor "it computes the right
energy" -- both of those are checked where they belong -- but the claims that
make it a diagnostic rather than a second engine:

1. **The CLI flags resolve to the physics they name.**  ``--liquid-rs`` moves
   the liquid's coupling, ``--crystal-rs`` the crystal's, ``--crystal-nmax``
   the crystal's Landau-level truncation and nothing else.  Checked at the
   ``VMC(...)`` CALL, not on a label, so a flag that is parsed and then dropped
   fails here.

2. **The liquid has no variational orbital sector and no truncation.**  Its
   ``nmax`` is padding -- ``c_row(0)`` is exactly the first unit vector, so every
   higher band enters multiplied by zero -- and it must never receive the
   crystal's truncation.  There is deliberately no ``--liquid-nmax``.

3. **The crystal's truncation is the engine's.**  ``n_max = N`` keeps
   ``n = 0 ... N``, so ``n_bands = N + 1``: two bands at ``nmax = 1``, three at
   ``nmax = 2``.  Asserted against the basis the engine actually builds.

4. **The observable arrays are measured, not remembered.**  The module may not
   read ANY input file: no checkpoint, no frozen snapshot, no stored observable
   array, no ``_diag`` product and no old PNG.  Checked structurally (no path in
   the source names the historical tree) and behaviourally (the observables and
   the figures still work with the file-read entry points disabled).

5. **The observable conventions are the documented ones, and they are the
   historical ones.**  Density is periodic and normalised to a mean of 1;
   g(x,y) is 1 for a uniform liquid; S(qx,qy) keeps the raw array and records
   the q -> 0 disc as a mask.

6. **Two points never share a directory.**  Every input that changes the
   numbers is in the slug.

Deliberately NOT here: any test that runs a production VMC.  One test runs the
cheapest budget that exercises every stage of the pipeline; everything else uses
synthetic snapshots and finishes in seconds.
"""
import ast
import builtins
import importlib.util
import json
import math
import os
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SCRIPT = os.path.join(CLEAN, "examples", "figure_construction",
                      "structure.py")

#: The engine layers a recipe must not reach into.  ``wigner_vmc`` and
#: ``wigner_vmc.analysis`` are the public surface and are NOT in this list.
CORE_LAYERS = ("wigner_vmc.vmc", "wigner_vmc.physics", "wigner_vmc.wavefunctions",
               "wigner_vmc.optimization")

#: Every module the recipe is allowed to import.  A whitelist rather than a
#: blacklist, because the point of §6 is that the clean path has NO route to the
#: stored results -- and the cheapest way to grow one is to import something.
ALLOWED_IMPORTS = frozenset((
    "__future__", "argparse", "hashlib", "json", "math", "os", "time",
    "datetime", "textwrap", "subprocess", "numpy", "matplotlib",
    "matplotlib.colors", "matplotlib.figure", "matplotlib.pyplot", "wigner_vmc",
    "wigner_vmc.analysis", "wigner_vmc.analysis.structure",
))

#: Words that would mean a path in this module points at the historical tree.
#: They occur in the PROSE of this module (which explains that it touches
#: nothing of the sort) and that is fine; what is checked is that no PATH and no
#: module-level constant contains one.
LEGACY_TOKENS = ("_diag", "ckpt", "legacy", "frozen")

#: The only functions allowed to read a file at all.
#:
#: ``load_point`` and ``check_namespace_reuse`` read the recipe's OWN
#: ``run.json``/``run.npz`` out of this point's results directory -- the first to
#: reuse a run under ``--resume``, the second to warn before overwriting; neither
#: number reaches an observable.  ``_file_sha`` reads SOURCE FILES, to put their
#: content hash in the metadata: it is a source identifier, not a data input, and
#: ``test_source_hashing_reads_only_the_source`` pins its only caller.
READERS = frozenset(("load_point", "check_namespace_reuse", "_file_sha"))

EXPECTED_ARTIFACTS = ("density_xy.npz", "pair_correlation_xy.npz",
                      "structure_factor_xy.npz", "run_metadata.json")
EXPECTED_FIGURES = ("density_xy.png", "pair_correlation_xy.png",
                    "structure_factor_xy.png", "combined_structure_summary.png")


# ==========================================================================
# helpers
# ==========================================================================
def _load():
    """Import the recipe by path.  Module level only -- it defines constants and
    imports the API; no VMC runs until a function is called."""
    spec = importlib.util.spec_from_file_location("phase_structure_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree():
    with open(SCRIPT, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=SCRIPT)


def _calls_with_owner(tree):
    """``(owner_name, call_node)`` for every call in the module.

    A call inside a nested ``def`` is attributed to THAT def, not to its parent,
    so a read hidden in a closure is reported under its own name and cannot
    inherit an allowance.
    """
    out = []

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, child.name)
            else:
                if isinstance(child, ast.Call):
                    out.append((owner, child))
                visit(child, owner)

    visit(tree, "<module>")
    return out


def _uniform_snaps(torus, n=40, seed=0):
    """Positions of a structureless gas, in the Cartesian frame the estimators use."""
    rng = np.random.default_rng(seed)
    return [(rng.random((36, 2)) - 0.5) @ torus.sc.T for _ in range(n)]


def _lattice_snaps(torus, n=20, seed=0, jitter=0.30):
    """Positions of a jittered triangular lattice: the answer is known, so the
    estimators can be checked against it instead of against themselves."""
    rng = np.random.default_rng(seed)
    sites = _load().st.lattice_sites(torus) @ torus.sc.T
    return [sites + jitter * rng.standard_normal((36, 2)) for _ in range(n)]


def _ctx(mod, liquid_rs=55.0, crystal_rs=55.0, nmax=2, budget="quick"):
    return {"liquid_rs": float(liquid_rs), "crystal_rs": float(crystal_rs),
            "crystal_nmax": int(nmax),
            "kappa_liquid": mod._phase_kappa(liquid_rs),
            "kappa_crystal": mod._phase_kappa(crystal_rs),
            "budget": str(budget),
            "comparison_kind": mod.comparison_kind(liquid_rs, crystal_rs)}


def _phases(mod, liquid_snaps=None, crystal_snaps=None):
    torus = mod.torus_for()
    if liquid_snaps is None:
        liquid_snaps = _uniform_snaps(torus, n=5)
    if crystal_snaps is None:
        crystal_snaps = _lattice_snaps(torus, n=5)
    phases = {"liquid": {"point": {"phase": "liquid"}, "g_rn": 1.01},
              "crystal": {"point": {"phase": "crystal"}, "g_rn": 2.90}}
    for name, snaps in (("liquid", liquid_snaps), ("crystal", crystal_snaps)):
        phases[name].update(mod.fields_for(snaps, torus))
    return torus, phases


# ==========================================================================
# architecture: the public surface, and no route to the historical tree
# ==========================================================================
class TestArchitecture(unittest.TestCase):
    def test_recipe_exists(self):
        self.assertTrue(os.path.exists(SCRIPT), f"missing {SCRIPT}")

    def test_imports_only_the_public_surface(self):
        seen = set()
        for node in ast.walk(_tree()):
            if isinstance(node, ast.ImportFrom):
                seen.add(node.module or "")
            elif isinstance(node, ast.Import):
                seen.update(a.name for a in node.names)
        bad = sorted(n for n in seen if n not in ALLOWED_IMPORTS)
        self.assertEqual(bad, [], f"imports outside the whitelist: {bad}")
        for layer in CORE_LAYERS:
            with self.subTest(core_layer=layer):
                self.assertNotIn(layer, seen)

    def test_no_engine_symbol_is_reimplemented(self):
        """No Metropolis loop, no SR, no local energy, no Coulomb, no
        wavefunction formula, no histogram or kernel of its own.  A recipe that
        grows one of these is a second engine with a figure attached."""
        src = open(SCRIPT, encoding="utf-8").read()
        for banned, why in (
                ("local_energy", "the local energy estimator"),
                ("metropolis", "the sampler's Markov kernel"),
                ("sr_optimize", "stochastic reconfiguration"),
                ("coulomb", "the Hamiltonian"),
                ("jastrow_vector", "the wavefunction's parameters"),
                ("np.exp(-0.5 * (x / sigma", "a Gaussian kernel of its own"),
                ("np.histogram", "a histogram estimator of its own")):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, src,
                                 f"the recipe reimplements {why} ({banned!r})")

    def test_no_path_in_the_source_names_the_historical_tree(self):
        """Structural: a stored result cannot be loaded if no path points at one.

        Two candidates are checked, which together cover every path this module
        constructs or names: the module-level constants, and the string literals
        handed to ``os.path.join``.  The LEGACY_TOKENS do occur in the module's
        prose -- it is the prose that explains the firewall -- so the check is
        deliberately about PATH construction and not about the text.
        """
        tree = _tree()
        named = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        named.append(sub.value)
        for _, call in _calls_with_owner(tree):
            fn = call.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "join"
                    and isinstance(fn.value, ast.Attribute) and fn.value.attr == "path"):
                continue
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    named.append(arg.value)
        for value in named:
            for token in LEGACY_TOKENS:
                with self.subTest(path_component=value, token=token):
                    self.assertNotIn(token, value, f"{value!r} names the historical tree")

    def test_only_the_recipe_s_own_saved_runs_are_ever_read(self):
        """Every ``open``/``np.load`` in the module must be a WRITE, or a read
        attributed to one of ``READERS``.  A read added anywhere else fails,
        wherever it is put -- including inside a nested def."""
        reads = []
        for owner, call in _calls_with_owner(_tree()):
            fn = call.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name not in ("open", "load", "loadtxt", "imread", "load_npz"):
                continue
            if isinstance(fn, ast.Attribute) and name == "load" \
                    and getattr(fn.value, "id", None) not in ("np", "numpy"):
                continue                      # json.load(fh) on an already-open handle
            if name != "open":
                reads.append((owner, name))
                continue
            mode = None
            if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
                mode = call.args[1].value
            for kw in call.keywords:
                if kw.arg == "mode":
                    mode = getattr(kw.value, "value", None)
            if mode is None or "r" in str(mode):
                reads.append((owner, f"open(mode={mode!r})"))
        bad = [r for r in reads if r[0] not in READERS]
        self.assertEqual(bad, [], f"file reads outside {sorted(READERS)}: {bad}")

    def test_source_hashing_reads_only_the_source(self):
        """The one read allowance that is not a saved run is a SOURCE hash.

        Checked as a pin rather than a promise: ``_file_sha`` has exactly one
        caller, that caller is ``_source_hashes``, and what it hashes is the
        recipe and the two installed modules it calls -- never a path under this
        recipe's own results or figures directories, and never a stored number.
        """
        mod = _load()
        callers = {owner for owner, call in _calls_with_owner(_tree())
                   if (getattr(call.func, "id", None)
                       or getattr(call.func, "attr", None)) == "_file_sha"}
        self.assertEqual(callers, {"_source_hashes"}, callers)

        keys = set(mod._source_hashes("quick"))
        self.assertEqual(keys, {"recipe", "wigner_vmc/api.py",
                                "wigner_vmc/analysis/structure.py",
                                "configs/quick.yaml"})
        # and the hashes are of the files those names point at, not of anything
        # this recipe produced
        self.assertEqual(mod._source_hashes("quick")["recipe"],
                         mod._file_sha(os.path.join(
                             CLEAN, "examples", "figure_construction",
                             "structure.py")))
        # a file that is not there is a recorded gap, not a crash
        self.assertIsNone(mod._file_sha(os.path.join(CLEAN, "configs",
                                                     "no-such-budget.yaml")))

    def test_results_and_figures_leave_the_example_directory(self):
        mod = _load()
        for base in (mod.RESULTS, mod.FIGDIR):
            with self.subTest(base=base):
                self.assertTrue(base.startswith(CLEAN), base)
                self.assertNotIn(os.path.join("examples", "figure_construction"), base)
                self.assertIn(os.path.join("figure_construction", "structure"), base)


# ==========================================================================
# the CLI: what the flags resolve to
# ==========================================================================
class TestTheRequestResolves(unittest.TestCase):
    def test_the_defaults_are_this_recipe_s_own(self):
        mod = _load()
        req = mod.resolve_request(mod.parse_args(["--budget", "quick"]))
        self.assertEqual(req["liquid_rs"], mod.DEFAULT_LIQUID_RS)
        self.assertEqual(req["crystal_rs"], mod.DEFAULT_CRYSTAL_RS)
        self.assertEqual(req["crystal_nmax"], mod.DEFAULT_CRYSTAL_NMAX)
        self.assertEqual(req["ansatz"], "ll_rotation")
        self.assertEqual(req["budget"], "quick")
        for key in ("liquid_rs_given", "crystal_rs_given", "crystal_nmax_given"):
            self.assertFalse(req[key], f"{key} should be False when nothing was typed")

    def test_a_given_default_is_still_recorded_as_given(self):
        """``55`` typed is a choice even though it equals the default, and the
        metadata has to be able to say which it was."""
        mod = _load()
        req = mod.resolve_request(
            mod.parse_args(["--liquid-rs", str(mod.DEFAULT_LIQUID_RS),
                            "--crystal-rs", str(mod.DEFAULT_CRYSTAL_RS),
                            "--crystal-nmax", str(mod.DEFAULT_CRYSTAL_NMAX)]))
        self.assertTrue(req["liquid_rs_given"])
        self.assertTrue(req["crystal_rs_given"])
        self.assertTrue(req["crystal_nmax_given"])

    def test_each_flag_moves_its_own_parameter_and_only_its_own(self):
        mod = _load()
        req = mod.resolve_request(mod.parse_args(
            ["--liquid-rs", "45", "--crystal-rs", "65", "--crystal-nmax", "3"]))
        self.assertEqual(req["liquid_rs"], 45.0)
        self.assertEqual(req["crystal_rs"], 65.0)
        self.assertEqual(req["crystal_nmax"], 3)
        self.assertEqual(req["budget"], mod.DEFAULT_BUDGET)
        self.assertEqual(req["ansatz"], mod.DEFAULT_ANSATZ)

    def test_there_is_no_liquid_nmax(self):
        mod = _load()
        with self.assertRaises(SystemExit):
            mod.parse_args(["--liquid-nmax", "2"])

    def test_the_full_alias_resolves_to_the_reproduction_config(self):
        """``--budget full`` is a SPELLING of ``reproduction``, not a budget.

        ``full`` is what a reader reaches for when they mean "the real
        statistics".  It must name the same config, so the alias cannot mint a
        second directory holding the identical run -- and it must be resolved
        BEFORE the slug is built, because the raw spelling slugs differently and
        would fork the output tree.
        """
        from wigner_vmc import load_budget, resolve_budget_name
        mod = _load()
        self.assertEqual(resolve_budget_name("full"), "reproduction")
        self.assertEqual(load_budget("full").name, "reproduction")
        # An alias is not a config: an unknown budget still passes through as
        # itself so that ``load_budget`` fails on it with the list of what
        # exists, rather than the alias table guessing.
        self.assertEqual(resolve_budget_name("no-such-budget"), "no-such-budget")
        self.assertEqual(mod.parse_args(["--budget", "full"]).budget, "full")
        two = mod.request_slug(55.0, 55.0, "ll_rotation", 2,
                               resolve_budget_name("full"))
        self.assertEqual(two,
                         mod.request_slug(55.0, 55.0, "ll_rotation", 2,
                                          "reproduction"))
        self.assertTrue(two.endswith("__reproduction"))
        self.assertNotEqual(two,
                            mod.request_slug(55.0, 55.0, "ll_rotation", 2, "full"))

    def test_the_coupling_is_the_physical_rs_over_sqrt2(self):
        mod = _load()
        self.assertAlmostEqual(mod._phase_kappa(4.0 * math.sqrt(2.0)), 4.0, places=12)
        self.assertAlmostEqual(mod._phase_kappa(64.0 * math.sqrt(2.0)), 64.0, places=12)
        # and NOT the Stage-2E rounded regression
        self.assertNotEqual(mod._phase_kappa(55.0), round(55.0 / math.sqrt(2.0), 4))


class TestTheLiquidStaysAFilledLLL(unittest.TestCase):
    """Checked at the ``VMC`` call, because the forcing in ``run_point`` is what
    makes it true by construction rather than by luck."""

    class _Boom(Exception):
        pass

    def _record(self, phase, nmax, rs=55.0, ansatz="ll_rotation"):
        mod = _load()
        seen = {}
        boom = self._Boom

        class _Recorder:
            def __init__(self, **kw):
                seen.update(kw)
                raise boom

        real = mod.VMC
        mod.VMC = _Recorder
        try:
            with self.assertRaises(boom):
                mod.run_point(phase, rs, 0, "smoke", ansatz=ansatz, nmax=nmax)
        finally:
            mod.VMC = real
        return mod, seen

    def test_the_liquid_never_receives_the_crystals_truncation(self):
        mod, seen = self._record("liquid", nmax=3)
        self.assertEqual(seen["phase"], "liquid")
        self.assertEqual(seen["nmax"], mod.LIQUID_NMAX)
        self.assertEqual(seen["ansatz"], "ll_rotation")

    def test_the_crystal_receives_the_users_coupling_and_truncation(self):
        mod, seen = self._record("crystal", nmax=2, rs=75.0)
        self.assertEqual(seen["phase"], "crystal")
        self.assertEqual(seen["rs"], 75.0)
        self.assertEqual(seen["nmax"], 2)

    def test_the_module_says_the_liquid_has_no_truncation_and_no_flag(self):
        """The claim is not only in the code path but in what the module tells a
        reader: the liquid's ``nmax`` is named as padding and the flag that would
        invite reading it as physics does not exist."""
        mod = _load()
        src = open(SCRIPT, encoding="utf-8").read()
        self.assertIn('"filled_lll_fixed"', src)
        self.assertIn("There is deliberately no ``--liquid-nmax``", src)
        self.assertEqual(mod.LIQUID_NMAX, 1)
        with self.assertRaises(SystemExit):
            mod.parse_args(["--liquid-nmax", "1"])


class TestTheCrystalTruncationIsReal(unittest.TestCase):
    def test_the_basis_truncation_tracks_the_flag(self):
        from wigner_vmc import VMC
        for nmax in (1, 2, 3):
            with self.subTest(nmax=nmax):
                vmc = VMC(N=36, rs=55.0, phase="crystal", nmax=nmax)
                self.assertEqual(vmc.n_bands, nmax + 1)
                basis = vmc.basis()
                self.assertEqual(basis.nmax, nmax)
                self.assertEqual(basis.n_bands, nmax + 1)
                self.assertEqual(basis.orbitals(np.zeros(2)).shape, (36, nmax + 1))

    def test_nmax_one_gives_two_bands_and_nmax_two_gives_three(self):
        from wigner_vmc import VMC
        for nmax, n_bands in ((1, 2), (2, 3)):
            with self.subTest(nmax=nmax):
                vmc = VMC(N=36, rs=55.0, phase="crystal", nmax=nmax)
                self.assertEqual(vmc.n_bands, n_bands)
                self.assertEqual(vmc.nmax, n_bands - 1)

    def test_the_shared_parameters_grow_with_the_band_count(self):
        """``theta`` is ``[c (NJ), Re v, Im v]``, so its length is a direct
        read-out of the truncation: 77 at nmax=1, 149 at nmax=2."""
        from wigner_vmc import VMC, theta
        from wigner_vmc.api import NJ
        lengths = {}
        for nmax in (1, 2):
            with self.subTest(nmax=nmax):
                vmc = VMC(N=36, rs=55.0, phase="crystal", nmax=nmax)
                nb = vmc.n_bands
                th = theta(np.zeros(NJ), np.zeros((36, nb - 1), complex))
                self.assertEqual(len(th), NJ + 2 * (nb - 1) * 36)
                lengths[nmax] = len(th)
        self.assertEqual(lengths, {1: 77, 2: 149})

    def test_the_truncation_changes_the_basis_not_just_a_label(self):
        """The extra bands are genuinely extra columns, and the columns the two
        bases share are the same numbers -- so ``nmax = 2`` is ``nmax = 1`` plus
        one more Landau level, not a differently-normalised basis."""
        from wigner_vmc import VMC
        probe = np.array([0.37, -1.24])
        a = VMC(N=36, rs=55.0, phase="crystal", nmax=1).basis().orbitals(probe)
        b = VMC(N=36, rs=55.0, phase="crystal", nmax=2).basis().orbitals(probe)
        self.assertEqual((a.shape[1], b.shape[1]), (2, 3))
        np.testing.assert_allclose(a, b[:, :2], rtol=1e-12, atol=1e-12)

    def test_the_engine_refuses_nmax_below_one(self):
        """So ``>= 0`` in the CLI would be a lie about the engine, and the
        validator's bound is the engine's own."""
        from wigner_vmc import VMC
        with self.assertRaises(Exception):
            VMC(N=36, rs=55.0, phase="crystal", nmax=0)


class TestInputValidation(unittest.TestCase):
    def test_the_validator_refuses_what_the_engine_cannot_build(self):
        mod = _load()
        for bad in ((-1.0, 55.0, 2), (0.0, 55.0, 2), (float("nan"), 55.0, 2),
                    (55.0, -1.0, 2), (55.0, float("inf"), 2),
                    (55.0, 55.0, 0), (55.0, 55.0, -1)):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    mod.validate_inputs(*bad)

    def test_nmax_below_one_is_refused_with_the_engines_own_reason(self):
        mod = _load()
        with self.assertRaises(SystemExit) as cm:
            mod.validate_inputs(55.0, 55.0, 0)
        self.assertIn("highest Landau index", str(cm.exception))

    def test_a_valid_request_passes(self):
        mod = _load()
        mod.validate_inputs(55.0, 55.0, 2)
        mod.validate_inputs(45.0, 65.0, 1)

    def test_an_invalid_request_fails_before_any_run(self):
        """The failure has to land before the SR ladder is paid for, so this
        asserts that ``VMC`` is never even constructed."""
        mod = _load()
        constructed = []

        class _Recorder:
            def __init__(self, **kw):
                constructed.append(kw)
                raise AssertionError("VMC was constructed for an invalid request")

        real = mod.VMC
        mod.VMC = _Recorder
        try:
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(SystemExit):
                    mod.main(["--liquid-rs", "-1", "--budget", "quick", "--quiet"])
        finally:
            mod.VMC = real
        self.assertEqual(constructed, [])


class TestNamespaces(unittest.TestCase):
    def test_every_input_that_changes_the_numbers_is_in_the_slug(self):
        mod = _load()
        ref = mod.request_slug(55.0, 55.0, "ll_rotation", 2, "quick")
        variants = {
            "liquid_rs": mod.request_slug(45.0, 55.0, "ll_rotation", 2, "quick"),
            "crystal_rs": mod.request_slug(55.0, 65.0, "ll_rotation", 2, "quick"),
            "crystal_nmax": mod.request_slug(55.0, 55.0, "ll_rotation", 1, "quick"),
            "ansatz": mod.request_slug(55.0, 55.0, "ll_rotation_pinned", 2, "quick"),
            "budget": mod.request_slug(55.0, 55.0, "ll_rotation", 2, "reproduction"),
        }
        for name, slug in variants.items():
            with self.subTest(one_input_away=name):
                self.assertNotEqual(slug, ref, f"{name} does not change the name")
        self.assertEqual(len({ref, *variants.values()}), 1 + len(variants))

    def test_the_namespace_is_a_subdirectory_per_point(self):
        mod = _load()
        slug = mod.request_slug(55.0, 55.0, "ll_rotation", 2, "quick")
        self.assertEqual(slug, "liquid_rs55__crystal_rs55__llrot_nmax2__quick")
        self.assertEqual(
            mod.namespace("/base", 55.0, 55.0, "ll_rotation", 2, "quick"),
            os.path.join("/base", slug))

    def test_the_compact_slug_is_not_the_key_but_the_digest_is(self):
        """Two couplings that round to the same six digits share a directory, so
        the full-precision digest -- not the slug -- has to be what gates reuse.
        """
        mod = _load()
        a = mod._config_key("crystal", 55.0000001, 0, "quick", nmax=2)
        b = mod._config_key("crystal", 55.0000002, 0, "quick", nmax=2)
        self.assertEqual(mod.request_slug(55.0000001, 55.0, "ll_rotation", 2, "quick"),
                         mod.request_slug(55.0000002, 55.0, "ll_rotation", 2, "quick"))
        self.assertNotEqual(mod._sha(a), mod._sha(b))


class TestComparisonLabel(unittest.TestCase):
    def test_equal_couplings_are_called_the_same_coupling(self):
        mod = _load()
        self.assertEqual(mod.comparison_kind(55.0, 55.0),
                         "same-coupling comparison")
        self.assertEqual(mod.comparison_kind(45.0, 65.0),
                         "different-coupling comparison")

    def test_nearly_equal_is_not_equal(self):
        mod = _load()
        self.assertEqual(mod.comparison_kind(55.0, 55.0000001),
                         "different-coupling comparison")

    def test_the_label_is_actually_drawn_on_the_figure(self):
        """§17 asks for the label on the figure, so the check is on the title the
        figure was given -- not on the fact that some function exists."""
        import matplotlib
        import matplotlib.pyplot as plt
        mod = _load()
        _, phases = _phases(mod)
        recorded = []
        real = matplotlib.figure.Figure.suptitle
        matplotlib.figure.Figure.suptitle = (
            lambda self, t, *a, **k: (recorded.append(str(t)), real(self, t, *a, **k))[1])
        try:
            with tempfile.TemporaryDirectory() as td:
                mod.write_figures(os.path.join(td, "same"), phases, _ctx(mod))
                mod.write_figures(os.path.join(td, "diff"), phases,
                                  _ctx(mod, crystal_rs=65.0))
        finally:
            matplotlib.figure.Figure.suptitle = real
            plt.close("all")
        heads = [t for t in recorded if "comparison" in t]
        self.assertTrue(any("same-coupling comparison" in t for t in heads), heads)
        self.assertTrue(any("different-coupling comparison" in t for t in heads), heads)


# ==========================================================================
# the observables: conventions, and where the numbers come from
# ==========================================================================
class TestTheObservablesCannotReadAnything(unittest.TestCase):
    """The behavioural half of the provenance firewall: with every file-read
    entry point disabled, the observables and the figures still work.

    A structural check on the source can be satisfied by a read spelled
    differently; this one cannot.  Library internals are exempt, because
    matplotlib reads its own font index -- that is not data, and the point of
    the check is that the CLEAN PATH reads no data file.
    """

    def test_observables_and_figures_work_with_reads_disabled(self):
        import matplotlib
        import matplotlib.font_manager as fm
        mod = _load()
        mod._mpl()
        fm.fontManager                                # warm: load the font index NOW
        torus, phases = _phases(mod)
        lib = os.path.dirname(os.path.abspath(matplotlib.__file__))
        cache = ""
        try:
            cache = os.path.abspath(matplotlib.get_cachedir())
        except Exception:
            pass
        real_open, real_load = builtins.open, np.load

        def _allowed(path):
            p = os.path.abspath(str(path))
            return p.startswith(lib) or (cache and p.startswith(cache))

        def _no_open(*a, **k):
            mode = k.get("mode", a[1] if len(a) > 1 else "r")
            if "r" in str(mode) and "+" not in str(mode) and not _allowed(a[0]):
                raise AssertionError(f"the clean path tried to read {a[0]!r}")
            return real_open(*a, **k)

        def _no_load(*a, **k):
            raise AssertionError("the clean path called np.load")

        builtins.open = _no_open
        np.load = _no_load
        try:
            fields = mod.fields_for(_uniform_snaps(torus, n=4), torus)
            self.assertEqual(set(fields), {"density", "gxy", "sq", "g_rn"})
            with tempfile.TemporaryDirectory() as td:
                written = mod.write_figures(td, phases, _ctx(mod))
                self.assertEqual(len(written), 4)
                for path in written:
                    self.assertGreater(os.path.getsize(path), 1000, path)
                mod.save_observables(os.path.join(td, "res"), phases,
                                     {"budget": "quick"})
                for name in EXPECTED_ARTIFACTS:
                    self.assertTrue(os.path.exists(os.path.join(td, "res", name)), name)
        finally:
            builtins.open, np.load = real_open, real_load


class TestTheDensityConvention(unittest.TestCase):
    def test_the_grid_mean_is_the_cell_density(self):
        """``density_grid`` is normalised so its mean is ``ne/area``; the recipe
        then divides by exactly that, so the convention is "in units of the mean
        density" and a uniform state reads 1."""
        mod = _load()
        torus = mod.torus_for()
        f = mod.density_field(_uniform_snaps(torus, n=40), torus)
        self.assertAlmostEqual(float(f["rho_cell"].mean()), 1.0, places=9)
        self.assertGreaterEqual(float(f["rho_cell"].min()), 0.0)
        self.assertAlmostEqual(f["n_mean"], 36.0 / torus.area, places=12)
        self.assertAlmostEqual(f["n_mean"], float(torus.ne) / torus.area, places=15)

    def test_the_smoothing_is_periodic_not_edge_held(self):
        """The blur is a CYCLIC convolution, which is the only correct boundary
        on a torus.  Asserted as equivariance under a roll -- which also proves
        the kernel reaches across the boundary, since the spike sits 2 bins from
        it -- and as conservation of the total."""
        mod = _load()
        H = np.zeros((36, 36))
        H[0, 0] = 1.0
        B = mod.st.periodic_gaussian_blur(H, 1.8)
        self.assertAlmostEqual(float(B.sum()), 1.0, places=12)
        rolled = mod.st.periodic_gaussian_blur(np.roll(H, (34, 34), (0, 1)), 1.8)
        np.testing.assert_allclose(rolled, np.roll(B, (34, 34), (0, 1)),
                                   rtol=1e-12, atol=1e-14)
        self.assertGreater(float(B[-2, 0]), 0.0, "the kernel did not wrap")

    def test_the_drawn_array_is_the_wrapped_pad_of_the_cell(self):
        mod = _load()
        torus = mod.torus_for()
        f = mod.density_field(_uniform_snaps(torus, n=12), torus)
        m, n = f["margin_bins"], f["nbins"]
        self.assertEqual(f["rho"].shape, (n + 2 * m, n + 2 * m))
        np.testing.assert_array_equal(f["rho"][m:-m, m:-m], f["rho_cell"])
        np.testing.assert_array_equal(f["rho"][:m, m:-m], f["rho_cell"][-m:])
        np.testing.assert_array_equal(f["rho"][m:-m, :m], f["rho_cell"][:, -m:])
        np.testing.assert_array_equal(f["rho"][:m, :m], f["rho_cell"][-m:, -m:])
        np.testing.assert_array_equal(f["rho"][m:-m, -m:], f["rho_cell"][:, :m])
        np.testing.assert_array_equal(f["rho"][-m:, m:-m], f["rho_cell"][:m])
        # the drawn coordinates are the cell edges, one cell plus the margin
        self.assertAlmostEqual(f["frac"][0], -m / n, places=12)
        self.assertAlmostEqual(f["frac"][-1], 1.0 + m / n, places=12)
        self.assertEqual(f["frac"].size, n + 2 * m + 1)
        self.assertEqual(f["outline"].shape, (5, 2))
        np.testing.assert_allclose(f["outline"][0], f["outline"][-1])

    def test_the_grid_is_the_historical_one(self):
        mod = _load()
        self.assertEqual(mod.DENSITY_NBINS, 72)
        self.assertEqual(mod.DENSITY_KERNEL_L_B, 0.40)
        torus = mod.torus_for()
        f = mod.density_field(_uniform_snaps(torus, n=4), torus)
        self.assertEqual(f["rho_cell"].shape, (72, 72))
        self.assertAlmostEqual(f["bin_l_b"], float(np.linalg.norm(torus.L1)) / 72,
                               places=12)
        self.assertAlmostEqual(f["kernel_l_b"] / f["bin_l_b"], 1.7819, places=3)

    def test_a_uniform_liquid_reads_near_one_everywhere(self):
        mod = _load()
        torus = mod.torus_for()
        f = mod.density_field(_uniform_snaps(torus, n=200), torus)
        self.assertLess(abs(float(f["rho_cell"].mean()) - 1.0), 0.01)
        # the site/midpoint contrast is the null-calibrated control: exactly 1
        # for anything uniform, plus the estimator's own noise at this sample count
        contrast = mod.st.site_midpoint_contrast(f["rho_cell"], torus,
                                                 mod.DENSITY_NBINS)
        self.assertLess(abs(contrast - 1.0), 0.15, f"contrast {contrast}")


class TestThePairCorrelationConvention(unittest.TestCase):
    def test_a_uniform_liquid_averages_to_one(self):
        """``g = 1`` for a uniform liquid is the normalisation's whole content."""
        mod = _load()
        torus = mod.torus_for()
        f = mod.pair_correlation_field(_uniform_snaps(torus, n=200), torus)
        self.assertLess(abs(float(f["g"].mean()) - 1.0), 0.02)
        self.assertLess(float(f["g"].min()), 1.0)
        self.assertGreater(float(f["g"].max()), 1.0)

    def test_a_lattice_state_peaks_at_the_lattice_vectors(self):
        """The estimator is checked against a state built ON the lattice, where
        the answer is known: the first shell has to be a tall peak, and the map
        six-fold rather than a featureless ring."""
        mod = _load()
        torus = mod.torus_for()
        snaps = _lattice_snaps(torus, n=20)
        f = mod.pair_correlation_field(snaps, torus)
        self.assertGreater(float(f["g"].max()), 3.0)
        self.assertGreater(mod.nearest_neighbour_height(snaps, torus), 1.5)

    def test_the_window_and_grid_are_the_historical_ones(self):
        mod = _load()
        self.assertEqual(mod.GXY_NR, 81)
        self.assertEqual(mod.GXY_SMOOTH, 1.0)
        self.assertEqual(mod.GXY_RMAX_NUM, 4.0)
        torus = mod.torus_for()
        f = mod.pair_correlation_field(_uniform_snaps(torus, n=4), torus)
        self.assertEqual(f["nr"], 81)
        self.assertEqual(f["g"].shape, (81, 81))
        self.assertAlmostEqual(f["rmax"], 4.0 / torus.sqrt_n, places=12)
        self.assertAlmostEqual(f["rmax"], 10.0265, places=3)
        self.assertAlmostEqual(f["centres"][f["nr"] // 2], 0.0, places=12)
        # the window must span several lattice constants, which is why this map
        # uses periodic images rather than the minimum image
        self.assertGreater(f["rmax"], 3.0 * float(np.linalg.norm(torus.g1)))


class TestTheStructureFactorConvention(unittest.TestCase):
    def test_the_kinematic_identity_at_small_q(self):
        """``S(q -> 0) = ne`` for EVERY state: a sum of ne unit phases over ne,
        which the state cannot enter.  With a finite torus the neighbourhood of
        q = 0 inherits it -- which is the whole reason the disc is masked."""
        mod = _load()
        torus = mod.torus_for()
        qs = np.array([[0.0, 0.0], [1e-7, 0.0], [0.02, 0.0], [0.05, 0.0],
                       [0.0, 0.05]])
        for snaps in (_uniform_snaps(torus, n=40), _lattice_snaps(torus, n=40)):
            with self.subTest(first_particle_x=float(snaps[0][0, 0])):
                S = mod.st.structure_factor_2d(snaps, qs[:, 0], qs[:, 1], 36)
                self.assertAlmostEqual(float(S[0]), 36.0, places=6)
                self.assertAlmostEqual(float(S[1]), 36.0, places=6)
                for value in S[2:]:
                    self.assertGreater(float(value), 20.0, f"S = {value}")

    def test_q_equals_zero_is_not_on_the_grid_and_the_disc_is_a_mask(self):
        mod = _load()
        torus = mod.torus_for()
        f = mod.structure_factor_field(_uniform_snaps(torus, n=12), torus)
        qx, qy, S = f["qx"], f["qy"], f["S"]
        self.assertEqual(S.shape, (mod.SQ_NQ, mod.SQ_NQ))
        np.testing.assert_array_equal(
            f["disc"], np.hypot(qx, qy) < float(np.linalg.norm(torus.G1)))
        # the raw array is kept, finite everywhere, and never NaN-ed or zeroed
        self.assertTrue(np.isfinite(S).all())
        self.assertGreater(float(S[f["disc"]].min()), 0.0)
        # q = 0 itself is not among the grid points -- documented, not assumed
        self.assertFalse(bool((qx == 0.0).any()))
        self.assertFalse(bool((qy == 0.0).any()))
        # the grid is linspace(-q_half, q_half, nq) with q_half a multiple of
        # sqrt(n), so the two central points straddle zero at +/- half a step
        self.assertAlmostEqual(float(np.abs(qx).min()),
                               f["q_half"] / (mod.SQ_NQ - 1), places=12)
        self.assertAlmostEqual(float(np.abs(qx).min()), 0.0200725676, places=9)
        # the innermost points still read ~ ne, and the disc is NOT uniformly ne
        self.assertGreater(float(S[f["inner"]].mean()), 15.0)
        self.assertLess(float(S[f["inner"]].mean()), 36.0)

    def test_the_disc_contains_no_allowed_momentum(self):
        """The mask can only be called a display convention if nothing MEASURED
        lives inside it: the allowed torus momenta start at |G1|, which is the
        disc's own radius."""
        mod = _load()
        torus = mod.torus_for()
        _, qn = torus.allowed_momenta(q_max=1.0)
        self.assertGreater(float(qn.min()), 0.0)
        self.assertAlmostEqual(float(qn.min()), float(np.linalg.norm(torus.G1)),
                               places=12)
        self.assertEqual(int((qn < float(np.linalg.norm(torus.G1))).sum()), 0)
        disc = mod.structure_factor_field(_uniform_snaps(torus, n=2), torus)["disc"]
        self.assertGreater(int(disc.sum()), 100)          # the disc is not empty

    def test_the_markers_are_the_six_first_shell_vectors(self):
        mod = _load()
        torus = mod.torus_for()
        f = mod.structure_factor_field(_uniform_snaps(torus, n=4), torus)
        bragg = f["bragg"]
        self.assertEqual(bragg.shape, (6, 2))
        g1 = float(np.linalg.norm(torus.g1))
        np.testing.assert_allclose(np.linalg.norm(bragg, axis=1), g1 * np.ones(6),
                                   rtol=1e-8)
        self.assertEqual(len({tuple(np.round(v, 9)) for v in bragg}), 6)

    def test_a_lattice_state_peaks_on_those_markers(self):
        """The strongest available check that the reciprocal-space grid is
        registered to the same lattice the real-space maps use."""
        mod = _load()
        torus = mod.torus_for()
        f = mod.structure_factor_field(_lattice_snaps(torus, n=20, jitter=0.16),
                                       torus)
        qx, qy = f["qx"], f["qy"]
        at_markers = []
        for v in f["bragg"]:
            i = int(np.argmin(np.abs(qx[:, 0] - v[0])))
            j = int(np.argmin(np.abs(qy[0, :] - v[1])))
            at_markers.append(float(f["S"][i, j]))
        off = float(np.median(f["S"][f["annulus"]]))
        self.assertGreater(float(np.mean(at_markers)), 5.0 * max(off, 1e-6),
                           f"peaks {at_markers} vs annulus median {off}")


# ==========================================================================
# end to end
# ==========================================================================
class TestEndToEnd(unittest.TestCase):
    """One real run through every stage, on the ``quick`` budget.

    Not a physics check: the numbers are unconverged and nothing is asserted
    about them.  What is asserted is that the chain is CONNECTED -- a cold start
    from ``(N, r_s)``, a walk, snapshots, observables, numbers on disk and PNGs
    drawn from those numbers -- and that the artifact set is the declared one.
    ``quick`` and not ``smoke`` because §19 asks about ``quick`` specifically,
    and the difference is a minute.
    """

    def test_the_quick_budget_produces_every_declared_artifact(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as td:
            mod.RESULTS = os.path.join(td, "results")
            mod.FIGDIR = os.path.join(td, "figures")
            self.assertEqual(mod.main(["--budget", "quick", "--quiet"]), 0)

            slug = mod.request_slug(mod.DEFAULT_LIQUID_RS, mod.DEFAULT_CRYSTAL_RS,
                                    "ll_rotation", mod.DEFAULT_CRYSTAL_NMAX, "quick")
            res = os.path.join(mod.RESULTS, slug)
            fig = os.path.join(mod.FIGDIR, slug)
            # one point, one directory -- and nothing written beside it
            self.assertEqual(sorted(os.listdir(mod.RESULTS)), [slug])
            self.assertEqual(sorted(os.listdir(mod.FIGDIR)), [slug])
            self.assertTrue(os.path.isdir(res), f"expected {res}")

            for name in EXPECTED_ARTIFACTS:
                self.assertTrue(os.path.exists(os.path.join(res, name)), name)
            for name in EXPECTED_FIGURES:
                path = os.path.join(fig, name)
                self.assertTrue(os.path.exists(path), name)
                self.assertGreater(os.path.getsize(path), 1000, path)
            for phase in ("liquid", "crystal"):
                for name in ("run.npz", "run.json"):
                    self.assertTrue(os.path.exists(os.path.join(res, phase, name)),
                                    f"{phase}/{name}")

            with open(os.path.join(res, "run_metadata.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["budget"], "quick")
            self.assertEqual(meta["quality"], mod.QUALITY_MARK)
            self.assertEqual(meta["N"], 36)
            self.assertEqual(meta["request_slug"], slug)
            self.assertEqual(meta["comparison_kind"], "same-coupling comparison")
            self.assertEqual(meta["crystal_nmax"], mod.DEFAULT_CRYSTAL_NMAX)
            self.assertEqual(meta["crystal_n_bands"], mod.DEFAULT_CRYSTAL_NMAX + 1)
            self.assertEqual(meta["crystal_ll_indices"],
                             list(range(mod.DEFAULT_CRYSTAL_NMAX + 1)))
            self.assertEqual(meta["liquid_ansatz"], "filled_lll_fixed")
            self.assertIn("recipe default", meta["liquid_rs_source"])
            self.assertIn("recipe default", meta["crystal_rs_source"])
            self.assertIn("recipe default", meta["crystal_nmax_source"])
            self.assertIn("physical", meta["kappa_mode"])
            self.assertGreater(len(meta["timestamp"]), 10)

            # the source identifier is checkable, not just a label: the recipe
            # hashes itself and the modules it does not own, so a reader can
            # recompute them and know which code produced these numbers
            sh = meta["source_hashes"]
            self.assertEqual(sh["recipe"], mod._file_sha(SCRIPT))
            for key in ("wigner_vmc/api.py", "wigner_vmc/analysis/structure.py",
                        "configs/quick.yaml"):
                with self.subTest(hashed=key):
                    self.assertIsNotNone(sh[key], f"{key} not hashed")
                    self.assertEqual(len(sh[key]), 64)

            liq, cry = meta["liquid"], meta["crystal"]
            self.assertEqual(liq["orbital_ansatz"], "filled_lll_fixed")
            self.assertFalse(liq["orbital_sr"])
            self.assertTrue(liq["jastrow_sr"])
            self.assertEqual(liq["ll_indices"], [0])
            self.assertEqual(liq["nmax"], mod.LIQUID_NMAX)
            self.assertFalse(liq["nmax_is_a_physical_parameter"])
            self.assertGreater(liq["n_snapshots"], 0)
            self.assertEqual(liq["kappa_mode"], "physical")
            self.assertGreaterEqual(liq["walk_seed"], 0)
            self.assertGreaterEqual(liq["rng_seed"], 0)
            self.assertAlmostEqual(liq["rs"] / math.sqrt(2.0), liq["kappa"], places=12)
            self.assertIsInstance(liq["energy_per_particle"], float)
            self.assertGreater(liq["acceptance"], 0.0)

            self.assertEqual(cry["orbital_ansatz"], "ll_rotation")
            self.assertTrue(cry["orbital_sr"])
            self.assertTrue(cry["jastrow_sr"])
            self.assertEqual(cry["nmax"], mod.DEFAULT_CRYSTAL_NMAX)
            self.assertEqual(cry["n_bands"], mod.DEFAULT_CRYSTAL_NMAX + 1)
            self.assertTrue(cry["nmax_is_a_physical_parameter"])
            self.assertGreater(cry["n_snapshots"], 0)
            self.assertAlmostEqual(cry["rs"] / math.sqrt(2.0), cry["kappa"], places=12)

            for block in ("density", "gxy", "sq"):
                self.assertIn(block, meta)
            self.assertIn("grid", meta["density"])
            self.assertIn("smoothing_kernel", meta["density"])
            self.assertIn("normalization", meta["density"])
            self.assertIn("spatial_window", meta["gxy"])
            self.assertIn("normalization", meta["gxy"])
            self.assertIn("q_grid", meta["sq"])
            self.assertIn("q0_treatment", meta["sq"])

            with np.load(os.path.join(res, "density_xy.npz")) as z:
                self.assertAlmostEqual(float(z["density_liquid_cell"].mean()), 1.0,
                                       places=8)
                self.assertAlmostEqual(float(z["density_crystal_cell"].mean()), 1.0,
                                       places=8)
                self.assertEqual(z["lattice_sites_frac"].shape, (36, 2))
                self.assertEqual(z["cell_outline"].shape, (5, 2))
                self.assertEqual(z["density_liquid"].shape, z["density_crystal"].shape)
                for key in ("A1", "A2", "L1", "L2", "nbins", "kernel_l_b",
                            "n_mean", "n_electrons"):
                    self.assertIn(key, z.files)
            with np.load(os.path.join(res, "pair_correlation_xy.npz")) as z:
                self.assertEqual(z["g_liquid"].shape, (81, 81))
                self.assertEqual(z["g_crystal"].shape, (81, 81))
                self.assertIn("rmax", z.files)
            with np.load(os.path.join(res, "structure_factor_xy.npz")) as z:
                self.assertEqual(z["S_liquid"].shape, (160, 160))
                self.assertEqual(z["S_crystal"].shape, (160, 160))
                self.assertEqual(z["bragg_vectors"].shape, (6, 2))
                self.assertTrue(np.isfinite(z["S_liquid"]).all())
                self.assertTrue(np.isfinite(z["S_crystal"]).all())
                self.assertTrue(z["mask_q0_disc"].any())
                self.assertIn("mask_q0_inner", z.files)

    def test_a_failed_budget_is_an_error_not_a_fallback(self):
        """The one behaviour that would quietly turn this into a redraw: a clean
        run that fails and is replaced by historical numbers."""
        mod = _load()
        with tempfile.TemporaryDirectory() as td:
            mod.RESULTS = os.path.join(td, "results")
            mod.FIGDIR = os.path.join(td, "figures")
            with self.assertRaises(Exception):
                mod.main(["--budget", "no-such-budget", "--quiet"])

    def test_the_declared_artifact_set_is_the_quick_one(self):
        """The artifact set does not depend on the budget, so it is checked at
        ``quick`` -- the documented non-production budget -- without paying for a
        second walk."""
        mod = _load()
        _, phases = _phases(mod)
        with tempfile.TemporaryDirectory() as td:
            res, fig = os.path.join(td, "res"), os.path.join(td, "fig")
            mod.save_observables(res, phases,
                                 {"budget": "quick", "quality": mod.QUALITY_MARK})
            mod.write_figures(fig, phases, _ctx(mod, budget="quick"))
            for name in EXPECTED_ARTIFACTS:
                self.assertTrue(os.path.exists(os.path.join(res, name)), name)
            for name in EXPECTED_FIGURES:
                self.assertTrue(os.path.exists(os.path.join(fig, name)), name)
            with open(os.path.join(res, "run_metadata.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["liquid"]["phase"], "liquid")
            self.assertEqual(meta["crystal"]["phase"], "crystal")
            self.assertEqual(meta["run_metadata_schema"],
                             "figure_construction/structure/1")

    def test_quick_figures_are_marked_and_reproduction_figures_are_not(self):
        """A quick figure that looked like a production one is the easiest
        possible way to mislead.  Asserted the way that means something: the
        SAME numbers rendered at the two budgets must produce different images,
        and the quick render must be reproducible."""
        import matplotlib.pyplot as plt
        mod = _load()
        _, phases = _phases(mod)
        with tempfile.TemporaryDirectory() as td:
            imgs = {}
            for tag, budget in (("quick_0", "quick"), ("quick_1", "quick"),
                                ("reproduction_2", "reproduction")):
                d = os.path.join(td, tag)
                mod.write_figures(d, phases, _ctx(mod, budget=budget))
                imgs[tag] = {n: plt.imread(os.path.join(d, n))
                             for n in EXPECTED_FIGURES}
            plt.close("all")
        for name in EXPECTED_FIGURES:
            with self.subTest(figure=name):
                np.testing.assert_array_equal(imgs["quick_0"][name],
                                              imgs["quick_1"][name])
                self.assertFalse(np.array_equal(imgs["quick_0"][name],
                                                imgs["reproduction_2"][name]),
                                 f"{name} is identical at the two budgets: the "
                                 f"QUICK stamp is not being drawn")

    def test_resume_refuses_a_mismatched_configuration(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as td:
            key = mod._config_key("crystal", 55.0, 0, "quick", nmax=2)
            point = {"phase": "crystal", "rs": 55.0, "nmax": 2, "snaps": [],
                     "R": np.zeros((36, 2))}
            mod.save_point(td, "crystal", point, key)
            back = mod.load_point(td, "crystal", key)
            self.assertIsNotNone(back)
            self.assertEqual(back["rs"], 55.0)
            for other in (mod._config_key("crystal", 56.0, 0, "quick", nmax=2),
                          mod._config_key("crystal", 55.0, 0, "quick", nmax=1),
                          mod._config_key("crystal", 55.0, 0, "reproduction", nmax=2),
                          mod._config_key("crystal", 55.0, 1, "quick", nmax=2),
                          mod._config_key("crystal", 55.0, 0, "quick", nmax=2,
                                          ansatz="ll_rotation_pinned"),
                          mod._config_key("liquid", 55.0, 0, "quick", nmax=2)):
                with self.subTest(other=other):
                    self.assertIsNone(mod.load_point(td, "crystal", other))
            self.assertTrue(os.path.exists(os.path.join(td, "crystal", "run.npz")))


if __name__ == "__main__":
    unittest.main()
