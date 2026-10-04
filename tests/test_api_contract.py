"""Stage 3: the API / CLI / packaging contract.

Stage 2E proved the clean engine reproduces the frozen records.  This file is
about a different claim: that there is exactly ONE workflow, that a user can
reach it without setting a path, and that the words on the surface mean what
they say.  None of it is physics -- so it carries the ``contract`` marker and is
deliberately NOT part of the ``-m behavior`` firewall, which must stay a
statement about VMC behaviour.

What is pinned here, and why each one is worth a test
-----------------------------------------------------
1. ``import wigner_vmc`` works in a CLEAN interpreter.  Run in a subprocess with
   PYTHONPATH cleared and the cwd in a temp directory, so it can only pass
   through the installed package.  Inside the suite the other tests prepend
   ``src/`` to ``sys.path``; that is fine for them and useless as evidence that a
   user can import the package.
2. ``result.energy_total == N * result.energy_per_particle``, and there is **no**
   ``result.energy``.  The bare name is the one that does not say which of the
   two it is, and every reader would read it as the one they expected.
3. The API's crystal start is built here, so the module may not reach the frozen
   store.  Checked structurally on ``api.py``'s own AST: its imports, its one
   ``open``, and its string literals.
4. The CLI and the ``examples/`` scripts are shells.  A script that can import
   ``sampler`` can run a sampler, and the day it does the two entry points stop
   being the same workflow.
5. ``nest`` aborts rather than falling back.  ``coefficient_identity_deviation``
   cannot serve as that guard -- it pads its own argument and returns exactly
   ``(0.0, 0.0)`` for every input -- so the guard is
   ``nesting_residual(parent, child)`` and it is tested against a genuinely
   perturbed child, with ``pad_v`` monkeypatched to produce one.
6. The geometry the API builds from ``(N, r_s)`` alone IS the verified one, and
   each lattice-vector set belongs to its own cell (the primitive-cell Landau
   sum vs the supercell overlap sum -- Blocker A's second defect).
7. ``measure_decomposed`` has not drifted from
   ``scripts/bench_rs75.py:production_walk``.  The duplication is deliberate --
   the benchmark is the Stage 2E path and is deliberately not refactored -- so
   it needs a test rather than a promise.
"""
import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SRC = os.path.join(CLEAN, "src")
SCRIPTS = os.path.join(CLEAN, "scripts")
EXAMPLES = os.path.join(CLEAN, "examples")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wigner_vmc import api                                        # noqa: E402
from wigner_vmc.wavefunctions import nesting as nest              # noqa: E402

USER_SCRIPTS = ("run_vmc.py", "run_nested.py", "run_scan.py")
EXAMPLE_SCRIPTS = ("run_crystal.py", "run_liquid.py", "playground.py")

#: The layers a script may not reach.  A CLI that can import these can implement
#: physics, which is exactly the drift the layering exists to prevent.
FORBIDDEN_SUBMODULES = ("sampler", "measure", "local_energy", "sr", "coulomb",
                        "jastrow", "ll_rotation", "nesting", "landau_levels",
                        "wavefunctions", "physics", "vmc.", "observables",
                        "optimization")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _imported_modules(tree):
    """``(module, [(name, lineno), ...])`` for every import in the module."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.append((None, [(a.name, node.lineno) for a in node.names]))
        elif isinstance(node, ast.ImportFrom):
            out.append((node.module or "", [(a.name, node.lineno)
                                            for a in node.names]))
    return out


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# 1. packaging
# ===========================================================================
class TestPackaging(unittest.TestCase):
    def test_a_clean_interpreter_can_import_the_package(self):
        """No PYTHONPATH, no cwd in the repo, no ``sys.path`` help.

        This is the only evidence that ``pip install -e .`` produced a working
        package rather than the suite's own path juggling doing the work.
        """
        code = ("import wigner_vmc, wigner_vmc.api\n"
                "from wigner_vmc import VMC, run_vmc\n"
                "print(wigner_vmc.__version__)\n")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=180)
        self.assertEqual(
            proc.returncode, 0,
            "a clean interpreter could not import wigner_vmc.  Install the "
            "package (`pip install -e .`) or fix the package layout; do not "
            "add a sys.path insert to work around it.\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
        self.assertEqual(proc.stdout.strip(), "0.1.0")

    def test_the_public_surface_is_the_documented_one(self):
        import wigner_vmc
        for name in ("VMC", "run_vmc", "RunResult", "RunState", "Optimization",
                     "RunConfig", "Budget", "Lattice", "NestingIdentityError",
                     "build_lattice", "load_budget", "resolve", "theta",
                     "theta_parts", "__version__"):
            self.assertIn(name, wigner_vmc.__all__, f"{name} is not exported")


# ===========================================================================
# 2. the energy names
# ===========================================================================
class TestTheEnergyNamesAreUnambiguous(unittest.TestCase):
    """Decision 3: the bare ``energy`` is deleted, and ``energy_total`` is
    derived rather than stored."""

    def test_there_is_no_bare_energy_attribute(self):
        self.assertFalse(
            hasattr(api.RunResult, "energy"),
            "RunResult.energy exists again.  It is the one name that does not "
            "say whether it is per particle or a total; use "
            "energy_per_particle or energy_total.")
        self.assertNotIn("energy", api.RunResult.__annotations__)

    def test_energy_total_is_exactly_N_times_the_per_particle(self):
        res = _result(energy_per_particle=-37.823241349, N=36)
        self.assertEqual(res.energy_total, res.N * res.energy_per_particle)
        self.assertEqual(res.error_total, res.N * res.error)

    def test_the_factor_of_N_appears_once_in_the_source(self):
        """``energy_total`` is a property, not a stored field -- so the factor of
        N cannot be written a second time somewhere and drift."""
        src = _read(os.path.join(SRC, "wigner_vmc", "api.py"))
        tree = ast.parse(src)
        for cls in [n for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == "RunResult"]:
            names = [n.target.id for n in cls.body
                     if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
            self.assertNotIn("energy_total", names,
                             "energy_total became a stored field; it must stay a "
                             "property so N is applied exactly once")


def _result(energy_per_particle, N, error=0.01, phase="crystal", nmax=1):
    """A RunResult built by hand -- no run, so these tests are free."""
    state = api.RunState(R=np.zeros((N, 2)), c=np.zeros(5),
                         v=np.zeros((int(N), nmax), complex), N=N, rs=75.0,
                         kappa=53.03300858899106, kappa_mode="physical",
                         nmax=nmax, phase=phase, snaps=None)
    opt = api.Optimization(kind="joint", steps=3, seconds=1.0,
                           E_total=np.array([-1.0, -2.0, -2.5]), acc=np.zeros(3),
                           N=N)
    return api.RunResult(config=api.resolve(phase=phase, N=N, nmax=nmax),
                         energy_per_particle=energy_per_particle, error=error,
                         acceptance=0.4, state=state, optimization=opt,
                         record={"n": 1, "sigma": 0.3, "E_tau": 1.0,
                                 "E_total": N * energy_per_particle},
                         seconds=1.0, budget=api.load_budget("smoke"))


# ===========================================================================
# 3. the starting point is built here
# ===========================================================================
class TestTheApiDoesNotReachTheFrozenStore(unittest.TestCase):
    """The structural fix for the 2026-10-02 incident: a run with no reason to
    open the frozen tree cannot mutate it."""

    #: ``io`` is matched as a WHOLE segment -- it is a substring of
    #: ``annotations``, ``jastrow``, ``ll_rotation`` and ``nesting``, all of
    #: which are legitimate.  The rest are long enough to match as substrings,
    #: so ``ckpt_rebuild`` and friends are caught whatever they are called.
    EXACT_SEGMENT = ("io",)
    FORBIDDEN_IMPORTS = ("ckpt", "checkpoint", "cache", "legacy", "recon",
                         "frozen", "notebook", "pickle", "shelve", "runpy")

    def setUp(self):
        self.path = os.path.join(SRC, "wigner_vmc", "api.py")
        self.tree = ast.parse(_read(self.path), filename=self.path)

    def test_it_imports_no_store_layer(self):
        """Segment-wise, so ``os`` does not match ``io`` and ``functools`` does
        not match anything -- only a whole dotted segment counts."""
        bad = []
        for module, names in _imported_modules(self.tree):
            for name, lineno in names:
                full = f"{module}.{name}" if module else name
                segs = full.lower().lstrip(".").split(".")
                if any(s in self.EXACT_SEGMENT for s in segs):
                    bad.append(f"line {lineno}: imports {full!r} -- the results "
                               f"conversion layer is store-side")
                for tok in self.FORBIDDEN_IMPORTS:
                    if any(tok in s for s in segs):
                        bad.append(f"line {lineno}: imports {full!r} (matches "
                                   f"{tok!r})")
        self.assertEqual(bad, [], "\n  " + "\n  ".join(bad))

    def test_the_only_file_it_opens_is_a_config(self):
        """One ``open``, inside ``load_budget``.  Named explicitly rather than
        counted, so a second one anywhere fails with its own name."""
        owners = []
        for fn in [n for n in ast.walk(self.tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "open"):
                    owners.append((fn.name, node.lineno))
        # module level counts too
        for node in self.tree.body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "open"):
                owners.append(("<module>", node.lineno))
        self.assertEqual([o[0] for o in owners], ["load_budget"],
                         f"unexpected open() sites: {owners}")

    def test_it_names_no_store_path_in_code(self):
        """String literals OUTSIDE docstrings.  Prose may explain the incident;
        code may not contain a path into the frozen tree."""
        docstrings = set()
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        bad = []
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings):
                # path-shaped fragments only.  ``legacy-regression`` is NOT one
                # of them: it is the name of a kappa convention the API
                # legitimately supports, not a place on disk.
                for frag in ("ckpt", "_diag", ".pkl", "figs_", "results/",
                             "initial_conditions", "jastrow_cache"):
                    if frag in node.value:
                        bad.append(f"line {node.lineno}: {frag!r} in "
                                   f"{node.value[:60]!r}")
        self.assertEqual(bad, [], "\n  " + "\n  ".join(bad))


# ===========================================================================
# 4. the CLI and the examples are shells
# ===========================================================================
class TestTheCliIsAThinShell(unittest.TestCase):
    def test_the_user_facing_scripts_exist(self):
        for name in USER_SCRIPTS:
            self.assertTrue(os.path.exists(os.path.join(SCRIPTS, name)),
                            f"scripts/{name} is missing")
        for name in EXAMPLE_SCRIPTS:
            self.assertTrue(os.path.exists(os.path.join(EXAMPLES, name)),
                            f"examples/{name} is missing")

    def test_scripts_reach_nothing_below_the_api(self):
        problems = []
        for name in USER_SCRIPTS:
            path = os.path.join(SCRIPTS, name)
            tree = ast.parse(_read(path), filename=path)
            for module, names in _imported_modules(tree):
                for imported, lineno in names:
                    if module is None:                       # import x.y
                        if imported.startswith("wigner_vmc"):
                            problems.append(f"{name}:{lineno}: import {imported}")
                        continue
                    if module == "wigner_vmc" and imported != "api":
                        problems.append(
                            f"{name}:{lineno}: from wigner_vmc import {imported} "
                            f"-- a CLI script goes through wigner_vmc.api only")
                    elif module.startswith("wigner_vmc."):
                        problems.append(
                            f"{name}:{lineno}: from {module} import {imported} "
                            f"-- the CLI must not import a physics layer")
                    for tok in FORBIDDEN_SUBMODULES:
                        if module == tok or module.endswith("." + tok):
                            problems.append(f"{name}:{lineno}: imports {module}")
            for token in ("sys.path", "sys.path.insert"):
                if token in _read(path):
                    problems.append(f"{name}: touches {token} -- the package is "
                                    f"installed, not path-hacked")
        self.assertEqual(problems, [], "\n  " + "\n  ".join(problems))

    def test_examples_use_only_the_public_surface(self):
        import wigner_vmc
        public = set(wigner_vmc.__all__)
        problems = []
        for name in EXAMPLE_SCRIPTS:
            path = os.path.join(EXAMPLES, name)
            tree = ast.parse(_read(path), filename=path)
            for module, names in _imported_modules(tree):
                if module is None:
                    continue
                if module == "wigner_vmc":
                    for imported, lineno in names:
                        if imported not in public:
                            problems.append(
                                f"{name}:{lineno}: {imported!r} is not part of "
                                f"the public surface")
                elif module.startswith("wigner_vmc."):
                    problems.append(
                        f"{name}: imports the submodule {module!r}; an example "
                        f"uses `from wigner_vmc import ...`")
            for token in ("sys.path",):
                if token in _read(path):
                    problems.append(f"{name}: touches {token}")
        self.assertEqual(problems, [], "\n  " + "\n  ".join(problems))

    def test_the_cli_hands_its_arguments_to_the_api(self):
        """``parse_args -> resolve -> run_vmc``, and exactly one call."""
        mod = _load(os.path.join(SCRIPTS, "run_vmc.py"), "cli_run_vmc_under_test")
        calls = []
        real = api.run_vmc

        def fake(config, verbose=True):
            calls.append((config, verbose))
            return "SENTINEL"

        api.run_vmc = fake
        try:
            rc = mod.main(["--phase", "crystal", "--rs", "75", "--nmax", "2",
                           "--budget", "smoke", "--init-id", "3",
                           "--rng-seed", "5"])
        finally:
            api.run_vmc = real
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1, "the CLI must call api.run_vmc exactly once")
        cfg, _ = calls[0]
        self.assertIsInstance(cfg, api.RunConfig)
        self.assertEqual((cfg.phase, cfg.rs, cfg.N, cfg.nmax, cfg.init_id,
                          cfg.budget, cfg.rng_seed),
                         ("crystal", 75.0, 36, 2, 3, "smoke", 5))

    def test_the_scan_drives_the_api_and_refuses_a_missing_parent(self):
        mod = _load(os.path.join(SCRIPTS, "run_scan.py"), "cli_run_scan_under_test")
        seen = []

        class FakeVMC:
            def __init__(self, **kw):
                self.kw = kw

            def run(self, init_id=0, budget="quick", verbose=True):
                seen.append(("run", init_id))
                if init_id == 2:
                    raise RuntimeError("synthetic failure")
                return SimpleNamespace(energy_per_particle=-37.0 - init_id,
                                       error=0.01,
                                       config=SimpleNamespace(init_id=init_id))

            def nest(self, parent, new_nmax, budget="quick", verbose=True):
                seen.append(("nest", parent.config.init_id, new_nmax))
                return SimpleNamespace(energy_per_particle=-37.5, error=0.02)

        real = api.VMC
        api.VMC = FakeVMC
        try:
            rc = mod.main(["--budget", "smoke", "--rs", "75", "--init-ids",
                           "0", "1", "2"])
        finally:
            api.VMC = real
        self.assertEqual(rc, 1, "an incomplete scan must not exit 0")
        self.assertIn(("nest", 0, 2), seen)
        self.assertIn(("nest", 1, 2), seen)
        self.assertNotIn(("nest", 2, 2), seen,
                         "the scan nested a parent that failed")


# ===========================================================================
# 5. nesting is checked, and the check can fail
# ===========================================================================
class TestTheNestingGuard(unittest.TestCase):
    def test_the_exact_pad_has_zero_residual(self):
        """The mathematical claim, on two genuinely different objects: the
        parent ``(nk, 1)`` and the child ``(nk, 2)``."""
        rng = np.random.default_rng(0)
        v = (rng.standard_normal((36, 1)) + 1j * rng.standard_normal((36, 1)))
        child = nest.pad_v(v, 3)
        self.assertEqual(child.shape, (36, 2))
        head, tail = nest.nesting_residual(v, child)
        self.assertEqual((head, tail), (0.0, 0.0))

    def test_a_perturbed_child_is_caught(self):
        rng = np.random.default_rng(1)
        v = (rng.standard_normal((36, 1)) + 1j * rng.standard_normal((36, 1)))
        child = nest.pad_v(v, 3)
        child[7, 0] += 1e-6
        head, tail = nest.nesting_residual(v, child)
        self.assertGreater(head, 0.0)
        self.assertEqual(tail, 0.0)
        child2 = nest.pad_v(v, 3)
        child2[3, 1] += 1e-9
        head2, tail2 = nest.nesting_residual(v, child2)
        self.assertEqual(head2, 0.0)
        self.assertGreater(tail2, 0.0)

    def test_the_old_guard_could_not_have_caught_anything(self):
        """``coefficient_identity_deviation`` pads its own argument, so it answers
        a question that is true by construction.  Recorded here so nobody
        reintroduces it as the guard."""
        rng = np.random.default_rng(2)
        v = rng.standard_normal((36, 1)) + 1j * rng.standard_normal((36, 1))
        self.assertEqual(nest.coefficient_identity_deviation(v, 3), (0.0, 0.0))

    def test_nest_aborts_instead_of_falling_back(self):
        """Monkeypatch ``pad_v`` to produce a child that is NOT the parent's
        wavefunction.  ``nest`` must raise, not re-initialise."""
        v = np.zeros((36, 1), complex)
        v[0, 0] = 0.7
        parent = SimpleNamespace(state=SimpleNamespace(nmax=1, phase="crystal",
                                                       v=v),
                                 config=SimpleNamespace(init_id=0))
        vmc = api.VMC(N=36, rs=75, phase="crystal", nmax=2)
        real = nest.pad_v

        def bad_pad(v_par, n_bands):
            out = np.array(real(v_par, n_bands), copy=True)
            out[5, 0] += 1e-3
            return out

        nest.pad_v = bad_pad
        try:
            with self.assertRaises(api.NestingIdentityError) as cm:
                vmc.nest(parent, new_nmax=2, budget="smoke", verbose=False)
        finally:
            nest.pad_v = real
        self.assertIn("head deviation", str(cm.exception))

    def test_nest_refuses_a_liquid_parent_and_a_shrinking_nmax(self):
        vmc = api.VMC(N=36, rs=75, phase="crystal", nmax=2)
        v = np.zeros((36, 1), complex)
        liquid = SimpleNamespace(state=SimpleNamespace(nmax=1, phase="liquid", v=v),
                                 config=SimpleNamespace(init_id=0))
        with self.assertRaises(ValueError):
            vmc.nest(liquid, new_nmax=2, budget="smoke", verbose=False)
        crystal = SimpleNamespace(state=SimpleNamespace(nmax=2, phase="crystal", v=v),
                                  config=SimpleNamespace(init_id=0))
        with self.assertRaises(ValueError):
            vmc.nest(crystal, new_nmax=2, budget="smoke", verbose=False)


# ===========================================================================
# 6. the geometry
# ===========================================================================
class TestTheLatticeIsTheVerifiedOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lat = api.build_lattice(36, 75.0)
        with open(os.path.join(CLEAN, "results", "bench_rs75",
                               "initial_conditions.json"), encoding="utf-8") as fh:
            cls.rec = json.load(fh)

    def test_it_reproduces_the_frozen_record_exactly(self):
        """``L1/L2/G1/G2`` and the overlap lattice vectors, which is what the
        frozen ``initial_conditions.json`` actually carries.  Note the record's
        ``_lints8``/``_lcart8`` are the SUPERCELL overlap set -- the name says
        "lattice" but they are passed to ``gaussian_overlap_seed``."""
        for key, built in (("L1", self.lat.L1), ("L2", self.lat.L2),
                           ("G1", self.lat.G1), ("G2", self.lat.G2),
                           ("_lints8", self.lat.ov_ai),
                           ("_lcart8", self.lat.ov_ac)):
            np.testing.assert_array_equal(
                np.asarray(built, float), np.asarray(self.rec[key], float),
                err_msg=f"{key} does not reproduce the frozen record exactly")
        self.assertEqual(api.LUMAX, float(self.rec["LUMAX"]))
        self.assertEqual(api.LUMAX_LL, float(self.rec["LUMAX_LL"]))

    def test_each_vector_set_belongs_to_its_own_cell(self):
        """Blocker A's second defect was a basis folded into the wrong cell.  The
        primitive Landau sum is integer in the PRIMITIVE basis and the overlap
        sum is integer in the SUPERCELL basis; neither is integer in the other."""
        lat = self.lat
        np.testing.assert_allclose(lat.ll_ac, lat.ll_ai @ lat.prim_C.T,
                                   rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(lat.ov_ac, lat.ov_ai @ lat.C.T,
                                   rtol=1e-15, atol=0.0)
        # and the two are not interchangeable
        self.assertGreater(float(np.abs(lat.ll_ac
                                        - lat.ll_ai @ lat.C.T).max()), 1.0)
        self.assertGreater(float(np.abs(lat.ov_ac
                                        - lat.ov_ai @ lat.prim_C.T).max()), 1.0)
        # the cutoffs apply to the cell they name
        self.assertLessEqual(float(np.linalg.norm(lat.ll_ac, axis=1).max()),
                             api.LUMAX_LL)
        self.assertLessEqual(float(np.linalg.norm(lat.ov_ac, axis=1).max()),
                             api.LUMAX)
        self.assertEqual(lat.ll_ai.shape, (121, 2))

    def test_it_agrees_with_the_production_geometry(self):
        from wigner_vmc.physics import hamiltonian as ham
        G = ham.production_geometry()
        for a, b, name in ((self.lat.L1, G.L1, "L1"), (self.lat.L2, G.L2, "L2"),
                           (self.lat.G1, G.G1, "G1"), (self.lat.G2, G.G2, "G2"),
                           (self.lat.mesh, G.mesh, "mesh")):
            np.testing.assert_array_equal(np.asarray(a, float),
                                          np.asarray(b, float),
                                          err_msg=f"{name} differs")
        np.testing.assert_allclose(self.lat.A1, np.asarray(self.lat.L1) / 6.0,
                                   atol=0.0)


# ===========================================================================
# 7. the deliberate duplication has not drifted
# ===========================================================================
class TestMeasureDecomposedMatchesTheBenchmark(unittest.TestCase):
    """``measure_decomposed`` is a twin of ``bench_rs75.production_walk``.

    The duplication is deliberate -- ``bench_rs75.py`` is the Stage 2E path and
    must keep injecting the legacy start as data, so it is not refactored onto
    the API.  A deliberate duplication needs a test, or it becomes a divergence.
    """

    def test_the_two_record_builders_agree_bit_for_bit(self):
        bench = _load(os.path.join(SCRIPTS, "bench_rs75.py"),
                      "bench_rs75_under_test")
        from wigner_vmc.vmc import sr as sr_mod

        vmc = api.VMC(N=36, rs=75, phase="crystal", nmax=1)
        c0 = sr_mod.jastrow_vector(vmc.kappa, api.NJ)
        v0 = np.zeros((vmc.lat.nk, vmc.n_bands - 1), complex)
        v0[:, 0] = 0.10 + 0.05j
        th = api.theta(c0, v0)

        a_wf, b_wf = vmc.maker()(th), vmc.maker()(th)     # independent objects
        kw = dict(sweeps=80, equil=20, sigma=0.3, seed=17)
        a = bench.production_walk(a_wf, vmc.crystal_R0(), label="bench", **kw)
        b = api.measure_decomposed(b_wf, vmc.crystal_R0(), kw["sweeps"],
                                   kw["equil"], kw["sigma"], kw["seed"],
                                   label="api")
        # the API's copy carries the snapshots (the state needs them) and the
        # benchmark's does not; every OTHER key must match.
        self.assertEqual(sorted(set(b) - {"snaps"}), sorted(a),
                         "the two records no longer have the same keys")
        for key in sorted(a):
            if key == "label":
                continue
            self.assertEqual(a[key], b[key],
                             f"{key} drifted: benchmark {a[key]!r} vs api {b[key]!r}")
        # sample() returns the POST-equilibration snapshots, so n = sweeps - equil
        self.assertEqual(a["n"], kw["sweeps"] - kw["equil"])
        self.assertGreater(a["acc"], 0.0)


# ===========================================================================
# 8. one end-to-end run, on the smallest budget
# ===========================================================================
class TestASmokeRunSatisfiesTheContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = api.VMC(N=36, rs=75, phase="crystal", nmax=1).run(
            init_id=0, budget="smoke", verbose=False)

    def test_the_result_fields_are_all_populated(self):
        r = self.res
        for value in (r.energy_per_particle, r.error, r.acceptance):
            self.assertTrue(math.isfinite(value))
        self.assertGreater(r.acceptance, 0.0)
        self.assertLess(r.error, 1.0)

    def test_total_and_per_particle_are_one_number(self):
        r = self.res
        self.assertEqual(r.energy_total, r.N * r.energy_per_particle)
        self.assertEqual(r.N, 36)
        # the walk measured a total independently; the property must agree with
        # it to float rounding, which is what pins the convention.
        self.assertLessEqual(abs(r.energy_total - r.record["E_total"]),
                             8 * np.finfo(float).eps * abs(r.record["E_total"]))

    def test_the_record_is_the_surface(self):
        r = self.res
        self.assertEqual(r.energy_per_particle, r.record["E"])
        self.assertEqual(r.error, r.record["E_err"])
        self.assertEqual(r.acceptance, r.record["acc"])
        self.assertEqual(len(r.optimization.E_total), r.optimization.steps)

    def test_the_state_is_usable(self):
        occ = self.res.state.ll_occupation()
        self.assertEqual(occ["n_bands"], 2)
        self.assertAlmostEqual(float(occ["P"].sum()), 1.0, places=12)
        sq = self.res.state.structure_factor()
        self.assertEqual(len(sq["S"]), len(sq["q"]))
        self.assertTrue(np.all(np.isfinite(sq["S"])))
        self.assertEqual(self.res.state.R.shape, (36, 2))


class TestTheDescriptionIsPhaseAware(unittest.TestCase):
    """A user-facing print must describe the calculation, not the parameter
    vector.

    The defect this pins: both phases share one flat ``theta``, so the liquid
    carries an ``nmax`` too -- and printing that as ``n_max 1`` invites a reader
    to think the liquid is a Landau-level truncation they could widen.  It is
    not: the liquid's orbitals are the filled ``n = 0`` level held at ``v = 0``,
    and `c_row(0)` multiplies every higher band by zero.  Nothing about the
    liquid is a function of that number.
    """

    def setUp(self):
        self.bud = api.load_budget("smoke")

    def test_the_liquid_description_claims_no_truncation(self):
        cfg = api.resolve(phase="liquid", rs=50.0, N=36, budget="smoke")
        text = cfg.describe(self.bud)
        self.assertIn("phase            liquid", text)
        self.assertIn("filled LLL (fixed)", text)
        self.assertIn("orbital SR       disabled", text)
        self.assertIn("Jastrow SR       enabled", text)
        self.assertIn("LL rotation      none", text)
        # The only mention of nmax is on the line that says it is technical.
        self.assertEqual(text.count("nmax"), 1, text)
        self.assertIn("technical", text)
        self.assertIn("not a liquid parameter", text)
        # ... and no `n_max`/`n_bands` presented as physics.
        self.assertNotIn("n_max            ", text)
        self.assertNotIn("LL basis", text)
        # The crystal's Gaussian seed is not a liquid parameter either.
        self.assertNotIn("init L0", text)

    def test_the_crystal_description_names_the_basis_it_builds(self):
        cfg = api.resolve(phase="crystal", rs=75.0, N=36, nmax=2, budget="smoke")
        text = cfg.describe(self.bud)
        self.assertIn("phase            crystal", text)
        self.assertIn("orbital ansatz   LL rotation", text)
        self.assertIn("n_max            2", text)
        self.assertIn("LL basis         n = 0, 1, 2", text)
        self.assertIn("n_bands          3   (= n_max + 1)", text)
        self.assertIn("orbital SR       enabled", text)
        self.assertIn("optimisation     joint orbital + Jastrow SR", text)
        self.assertIn("init L0", text)

    def test_the_pinned_crystal_says_what_stays_put(self):
        cfg = api.resolve(phase="crystal", rs=75.0, N=36, nmax=1,
                          ansatz="ll_rotation_pinned", budget="smoke")
        text = cfg.describe(self.bud)
        self.assertIn("(pinned)", text)
        self.assertIn("disabled (orbitals pinned at the seed)", text)
        self.assertIn("Jastrow only", text)

    def test_the_run_summary_carries_the_same_description(self):
        """The CLI prints ``result.summary()``, so the wording has to be in the
        summary, not only in a method nobody calls."""
        res = api.VMC(N=36, rs=50.0, phase="liquid").run(
            init_id=0, budget="smoke", verbose=False)
        self.assertIn("filled LLL (fixed)", res.summary())
        self.assertIn("LL rotation      none", res.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
