"""Regenerate the Stage 2C golden fixtures.

    python tests/fixtures/behavior/_build.py

Run from anywhere.  Deterministic: every random input comes from a seeded
generator, and the seeds are recorded in ``recipe.py``.  What lands on disk is
the INPUT (the configuration R, the band coefficients v, the model parameters)
together with the observables the current clean code returns for them -- so the
fixture can be re-derived from its own inputs, which is the property the test
relies on.

This file is named with a leading underscore on purpose: pytest collects
``test_*.py``, so the builder is not a test and never runs during the suite.  It
is invoked by hand, and only when a fixture is meant to move.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
CLEAN = os.path.dirname(TESTS)
SRC = os.path.join(CLEAN, "src")
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import recipe as R                                                # noqa: E402


def payload(name, layer, ansatz, legacy_source, inputs, wf, R_cfg, model):
    exp = R.observe(wf, R_cfg)
    return {"name": name, "layer": layer, "ansatz": ansatz,
            "legacy_source": legacy_source,
            "model": model, "inputs": inputs, "expected": exp}


def main():
    Ge = R.geometry()
    nk = len(Ge.mesh)

    interior = R.cell_points(Ge, R.NE, 0.05, 0.95, R.SEED_GAUSSIAN)
    straddling = R.cell_points(Ge, R.NE, -0.05, 1.05, R.SEED_GAUSSIAN + 1)

    gw = R.gaussian_wf(Ge)

    v1 = R.ll_v(nk, 1, R.SEED_LL)
    v2 = R.ll_v(nk, 2, R.SEED_LL)
    v_nested = np.concatenate([v1, np.zeros((nk, 1), dtype=complex)], axis=1)

    model_g = {"rs": R.RS, "kappa": R.KAPPA, "l0": R.L0, "c": list(R.C5),
               "ne": R.NE, "lattice": "supercell (LUMAX=30.0)"}
    model_ll = {"rs": R.RS, "kappa": R.KAPPA, "c": list(R.C5), "ne": R.NE,
                "lattice": "primitive (LUMAX_LL = 8.5*sqrt(4pi/sqrt3))"}

    todo = [
        ("gaussian_regular", "B", "gaussian", "qhvmc_engine.py:300-432",
         {"R": interior, "v": None}, gw, interior,
         dict(model_g, nmax=None, n_bands=None)),
        ("gaussian_boundary", "B", "gaussian", "qhvmc_engine.py:300-432",
         {"R": straddling, "v": None}, gw, straddling,
         dict(model_g, nmax=None, n_bands=None)),
        ("ll_nmax1", "B", "ll_rotation", "qhvmc_engine_llrot.py:59-243",
         {"R": interior, "v": v1}, R.ll_wf(Ge, v1), interior,
         dict(model_ll, nmax=1, n_bands=2, legacy_n_max=2)),
        ("ll_nmax2", "B", "ll_rotation", "qhvmc_engine_llrot.py:59-243",
         {"R": interior, "v": v2}, R.ll_wf(Ge, v2), interior,
         dict(model_ll, nmax=2, n_bands=3, legacy_n_max=3)),
        ("nested_nmax1_to_2", "C", "ll_rotation",
         "reproduction/make_notebook_llrot.py:1748-1750",
         {"R": interior, "v": v_nested}, R.ll_wf(Ge, v_nested), interior,
         dict(model_ll, nmax=2, n_bands=3, legacy_n_max=3,
              nested_from="ll_nmax1")),
    ]

    for name, layer, ansatz, src, inputs, wf, cfg, model in todo:
        pl = payload(name, layer, ansatz, src, inputs, wf, cfg, model)
        R.dump(name, pl)
        e = pl["expected"]
        print(f"{name:22s} layer {layer}  E_local = "
              f"{e['E_local'][0]:+.12e} {e['E_local'][1]:+.3e}j   "
              f"U = {e['U']:+.12e}   |v| cells = "
              f"{np.shape(inputs['v'])}")


if __name__ == "__main__":
    main()
