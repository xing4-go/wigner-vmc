"""Stage 2C: the physics behavior firewall's classification layer.

The B1-B7 regressions already exist and already pass; what was missing was a
single command that answers *"did the VMC physics change?"*.  This file supplies
it by classifying the existing tests, not by duplicating them:

    python -m pytest -m behavior      the physics firewall (layers A-F)
    python -m pytest -m layer_D       one layer at a time
    python -m pytest -m contract      the figure data-flow contract
    python -m pytest tests            the whole suite

Nothing here changes a test body, and no production module is touched.  The
classification lives in one place -- the table below -- so that a test moved
between layers is a one-line edit rather than a marker sprinkled through files.
Markers are applied in ``pytest_collection_modifyitems``, which pytest runs
before its own ``-m`` deselection, so ``-m behavior`` selects on markers that
were never written into the test modules themselves.  A consequence worth
knowing: ``python -m unittest`` still runs every test, it simply has no notion of
these markers.

What is in the firewall, and what is not
----------------------------------------
In: the six layers of physics behaviour, A through F -- the conventions and the
Hamiltonian, the two wavefunctions, the nesting invariant, the sampler's Markov
kernel and RNG stream, the local energy and the production estimators, and SR.

Out: the figure data-flow contract (an architecture rule, marked ``contract``),
and the results-conversion arithmetic other than its energy-field discipline --
both are still run by the full suite; they are not statements about VMC physics.
"""
from pathlib import Path

import pytest

LAYER_TITLES = {
    "A": "geometry / Hamiltonian conventions",
    "B": "wavefunctions (Gaussian and LL rotation)",
    "C": "LL nesting",
    "D": "sampler",
    "E": "energy / measurement",
    "F": "stochastic reconfiguration",
}

#: (layer, file name, class name or None for the whole file, classes to skip).
#: Applied in order and the first match wins, so a class rule must precede the
#: file-wide rule that would otherwise swallow it.
RULES = (
    ("C", "test_ll_rotation_against_legacy.py", "TestNesting", ()),
    ("B", "test_ll_rotation_against_legacy.py", None, ("TestNesting",)),
    ("C", "test_behavior_fixtures.py", "TestNestingInvariant", ()),
    ("C", "test_behavior_fixtures.py", "TestNestingLadder", ()),
    ("E", "test_behavior_fixtures.py", "TestTheEnergyConvention", ()),
    ("B", "test_behavior_fixtures.py", None, ("TestNestingInvariant",
                                              "TestNestingLadder",
                                              "TestTheEnergyConvention")),
    ("E", "test_legacy_conversion.py", "TestEnergyFieldDiscipline", ()),
    ("A", "test_units_and_conventions.py", None, ()),
    ("A", "test_geometry_against_legacy.py", None, ()),
    ("A", "test_magnetic_cell_against_legacy.py", None, ()),
    ("A", "test_coulomb_against_legacy.py", None, ()),
    ("B", "test_jastrow_against_legacy.py", None, ()),
    ("B", "test_gaussian_against_legacy.py", None, ()),
    ("B", "test_landau_basis_against_legacy.py", None, ()),
    # The Gaussian -> LL projection convention and the seed's C6 covariance.
    # B-layer: it is a claim about the wavefunction's own basis and the state the
    # crystal starts from.  It is registered here rather than left unmarked
    # because the defect it guards (`sr.py:219`, the conjugation on the wrong
    # operand) survived a full green suite -- every modulus test, every
    # normalisation test and the `c_row(v_from_overlap(ov)) == ov` round trip are
    # invariant under `ov -> conj(ov)`.  `-m behavior` is the run that is supposed
    # to answer "did the VMC physics change?", so the guard has to be in it.
    ("B", "test_projection_convention.py", None, ()),
    # Blocker A: the driver's own LL-basis wiring.  B-layer because it is a claim
    # about the wavefunction's basis, but it is a *wiring* test -- it asserts the
    # set production passes, which the fixture-parameterised B-tests cannot see.
    ("B", "test_bench_wiring.py", None, ()),
    # Blocker B: the kappa the driver's crystal STARTING POINT was built at.
    # B-layer for the same reason as test_bench_wiring -- it is a wiring claim
    # about the state production starts from, which the fixture-parameterised
    # tests cannot see (they are handed a (v0, c0) pair and never ask where it
    # came from).  It is checked against the frozen tags, not against itself.
    ("B", "test_init_provenance.py", None, ()),
    # Blocker A, section D: the coupling convention.  A-layer because it is a
    # claim about the Hamiltonian's own definition of kappa, not about a state.
    ("A", "test_kappa_convention.py", None, ()),
    ("D", "test_sampler_against_legacy.py", None, ()),
    ("E", "test_local_energy_against_legacy.py", None, ()),
    ("E", "test_measure_against_legacy.py", None, ()),
    ("E", "test_statistics.py", None, ()),
    ("E", "test_structure_against_legacy.py", None, ()),
    ("F", "test_sr_against_legacy.py", None, ()),
)

#: Architecture, not physics: the figure data-flow rule of Stage 1, and the
#: Stage 3 API / CLI / packaging contract.  Both are claims about the shape of
#: the code rather than about VMC behaviour, so neither belongs in the firewall.
CONTRACT = ("test_figure_contract.py", "test_api_contract.py",
            "test_figure_construction_contract.py",
            "test_phase_structure_contract.py")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "behavior: physics behavior firewall (Stage 2C, layers A-F)")
    config.addinivalue_line(
        "markers", "contract: figure / IO data-flow contract, not physics")
    for layer, title in sorted(LAYER_TITLES.items()):
        config.addinivalue_line("markers", f"layer_{layer}: {title}")


def _layer_of(file_name, class_name):
    for layer, fname, cls, skip in RULES:
        if fname != file_name:
            continue
        if cls is not None and cls != class_name:
            continue
        if class_name in skip:
            continue
        return layer
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        name = Path(str(item.fspath)).name
        cls = item.cls.__name__ if item.cls is not None else None
        layer = _layer_of(name, cls)
        if layer is not None:
            item.add_marker("behavior")
            item.add_marker(f"layer_{layer}")
        elif name in CONTRACT:
            item.add_marker("contract")


def pytest_report_header(config):
    """Make the firewall self-describing in the run header."""
    lines = ["behavior firewall layers:"]
    for layer, title in sorted(LAYER_TITLES.items()):
        files = sorted({f for lay, f, _, _ in RULES if lay == layer})
        lines.append(f"  {layer}  {title:42s} {', '.join(files)}")
    return lines
