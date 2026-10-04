"""The figure data-flow contract, enforced mechanically.

The Stage 1 rule is a shape, not a style preference:

    frozen result -> io/results -> analysis -> figure -> PNG

and NEVER

    notebook cell -> run VMC -> analyse -> savefig

A rule like that cannot be kept by remembering it, because the day it is broken is the
day someone is in a hurry and the notebook is right there.  So it is checked here, by
parsing the figure modules rather than by importing them: an import would execute the
code, and the code we are testing is precisely the code we do not want to execute by
accident.

Four things are forbidden inside a figure module, each for its own reason.

1. Importing the physics / VMC / optimisation layers.  A figure that can reach the
   sampler can run the sampler.  The whole point of the split is that the figure cannot.
2. `exec`, `eval`, `runpy`, `importlib`, `subprocess`.  These are how a notebook gets
   executed from inside something that was supposed to be a plot function -- which is
   literally how the legacy tree ended up with figures that only exist as a side effect
   of a cell.
3. Naming a legacy directory.  Figures receive paths and arrays; where the legacy tree
   lives is the composition root's business.  If a figure module knows the string
   `figs_LLRotation`, then a figure can be pointed at the frozen official PNG directory,
   and the first thing that happens next is a savefig over an official figure.
4. Writing outside the clean tree -- i.e. any savefig target that is not built from a
   directory the caller passed in.

The AST is parsed as text, so this test keeps working on a module with a syntax error
(which it reports as a failure) and never imports matplotlib or numpy.
"""
import ast
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
SRC = os.path.join(CLEAN, "src")
FIGDIR = os.path.join(SRC, "wigner_vmc", "figures")
MAKE_FIGURES = os.path.join(CLEAN, "scripts", "make_figures.py")

# Layers a figure may not reach.  A figure is allowed io/ and analysis/ and nothing else
# from this package.
FORBIDDEN_PACKAGE_ROOTS = (
    "wigner_vmc.vmc", "wigner_vmc.optimization", "wigner_vmc.physics",
    "wigner_vmc.wavefunctions", "wigner_vmc.observables",
)
# Names that mean "execute something else".
FORBIDDEN_NAMES = {"exec", "eval", "compile", "__import__"}
FORBIDDEN_MODULES = {"runpy", "importlib", "subprocess", "pickle", "shelve"}
# Any of these appearing as a string literal in a figure module means it knows where the
# frozen legacy tree lives, which it must not.
FORBIDDEN_PATH_FRAGMENTS = (
    "figs_LLRotation", "_diag", "ckpt_rebuild", "reproduction",
    "WignerCrystal_to_HallLiquid",
)


def _modules():
    out = []
    if os.path.isdir(FIGDIR):
        for f in sorted(os.listdir(FIGDIR)):
            if f.endswith(".py") and not f.startswith("_"):
                out.append(os.path.join(FIGDIR, f))
    if os.path.exists(MAKE_FIGURES):
        out.append(MAKE_FIGURES)
    return out


def _is_composition_root(path):
    return os.path.abspath(path) == os.path.abspath(MAKE_FIGURES)


class _Visitor(ast.NodeVisitor):
    """Checks one module.  `check_paths` is False for the composition root.

    `make_figures.py` is the ONE place that is supposed to know where the frozen tree
    lives -- it is the wiring, and wiring names destinations.  Every other rule still
    applies to it: it may not sample, may not exec, may not unpickle.
    """

    def __init__(self, path, check_paths=True):
        self.path = path
        self.check_paths = check_paths
        self.violations = []

    # --- imports
    def visit_Import(self, node):
        for a in node.names:
            self._check_module(a.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self._check_module(node.module, node.lineno)
        self.generic_visit(node)

    def _check_module(self, name, lineno):
        root = name.split(".")
        joined = ".".join(root[:2])
        if joined in FORBIDDEN_PACKAGE_ROOTS or name in FORBIDDEN_PACKAGE_ROOTS:
            self.violations.append(
                (lineno, f"imports {name!r} -- a figure must not be able to reach the "
                         f"sampler or the optimiser"))
        top = root[0]
        if top in FORBIDDEN_MODULES:
            self.violations.append(
                (lineno, f"imports {top!r} -- executing or unpickling from a figure is "
                         f"how a plot turns into a run"))

    # --- calls
    def visit_Call(self, node):
        f = node.func
        nm = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else None)
        if nm in FORBIDDEN_NAMES:
            self.violations.append((node.lineno, f"calls {nm}()"))
        # a call whose name looks like sampling
        if nm and any(t in nm.lower() for t in ("sample", "walk", "sweep", "optimiz")):
            self.violations.append(
                (node.lineno, f"calls {nm}() -- a name that suggests sampling inside a "
                              f"figure"))
        self.generic_visit(node)

    # --- string literals that name the frozen tree
    def visit_Constant(self, node):
        if isinstance(node.value, str) and self.check_paths:
            for frag in FORBIDDEN_PATH_FRAGMENTS:
                if frag in node.value:
                    self.violations.append(
                        (node.lineno, f"names {frag!r} -- figures receive paths from the "
                                      f"caller; only make_figures.py may know where the "
                                      f"legacy tree is"))
        self.generic_visit(node)


class TestFigureDataFlowContract(unittest.TestCase):
    def test_figure_modules_exist(self):
        mods = _modules()
        self.assertTrue(
            mods,
            f"no figure modules found under {FIGDIR} or {MAKE_FIGURES}; the contract "
            f"below cannot be vacuously satisfied by having nothing to check")

    def test_no_forbidden_imports_calls_or_paths(self):
        problems = []
        for path in _modules():
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            try:
                tree = ast.parse(src, filename=path)
            except SyntaxError as e:
                problems.append(f"{os.path.basename(path)}:{e.lineno}: syntax error {e.msg}")
                continue
            v = _Visitor(path, check_paths=not _is_composition_root(path))
            v.visit(tree)
            rel = os.path.relpath(path, CLEAN)
            for lineno, msg in v.violations:
                problems.append(f"{rel}:{lineno}: {msg}")
        self.assertEqual(problems, [], "\n  " + "\n  ".join(problems))


class TestFiguresNeverImportTheLegacyNotebook(unittest.TestCase):
    """The notebook is the source of the engine; a figure must never touch it."""

    def test_no_notebook_reference(self):
        bad = []
        for path in _modules():
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            for tok in (".ipynb", "runpy", "nbformat"):
                if tok in src:
                    bad.append(f"{os.path.relpath(path, CLEAN)} mentions {tok!r}")
        self.assertEqual(bad, [], "\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
