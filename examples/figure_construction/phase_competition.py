"""Rebuild ``vmc_energy_phase_competition.png`` from clean VMC runs only.

    python examples/figure_construction/phase_competition.py --budget quick
    python examples/figure_construction/phase_competition.py --budget full

What this is
------------
A research recipe.  The historical figure has two panels and answers two
questions:

  * the **ladder** (upper) -- absolute ``E/N`` of the liquid and of the crystal
    over the coupling grid, i.e. how the two phases descend;
  * the **competition** (lower) -- ``delta_E = E_crystal - E_liquid`` at the
    couplings where the two are close enough for the sign to be in question.

This recipe rebuilds BOTH panels from VMC runs it starts itself, at the
historical figure's point selection extended around it (see below), and it
**recomputes every energy difference**.  It does not read the historical
figure's numbers, from any file, in any format.

Why a rebuild and not a redraw
------------------------------
The historical ``delta_E`` points came from legacy Landau-level-rotation crystal
records produced with the reversed complex conjugation at ``vmc/sr.py:219``.
That defect makes the crystal's first shell **uniaxial** where it must be
hexagonal, and it moves the crystal energy -- measured at two independent
couplings: -0.545978 +- 0.055683 (z = -9.81) at r_s = 75, n_max = 2, and
-0.3958 (z ~ -6.9) at the fig06 coupling.  A phase competition read off those
states is a competition between the liquid and a state that is not the one the
ansatz describes.  So the points are recomputed rather than re-read, and this
file contains **no path to** ``results/legacy_llrot/scan.json``,
``results/legacy_llrot/states.json``, ``_diag/part1/*.json`` or the legacy
regression table ``_diag/energy_scan.json``.  That is a
property a test can check, and one does:
``tests/test_phase_competition_contract.py`` asserts none of those strings
appears in this source.

The same point selection, extended, on a different state
--------------------------------------------------------
The ``(r_s, tier)`` grid STARTS from the historical one and is extended around
it -- ``HISTORICAL_TIER_RS`` is the original selection, kept in the metadata
beside the current one so the difference is a recorded fact rather than
something a reader has to reconstruct.  The historical window ran from
r_s = 40 to 90 and could not show where the competition starts; the extended
grid samples 30 to 55 every 2.5 for n_max = 1 and 2, carries n_max = 1 alone
below that, and adds n_max = 2 and 3 rungs above 70.  The ladder couplings
``KAPPAS_PROTO`` are the original's ten, unchanged.

"Same point selection" was never a claim that the historical and the rebuilt
ladders show the same state: the historical ladder is a scan of the
**Gaussian** crystal (a site-centred magnetic Gaussian determinant), which this
package does not implement -- see ``DIVERGENCE_fig06_crystal.md``.  The rebuilt
ladder is the implemented ``ll_rotation`` ansatz on the same couplings, and its
metadata says so.  The two are different wavefunction families; the comparison
that matters for phase competition is liquid-versus-crystal WITHIN one recipe,
which is what the lower panel is.

Two constructions, and why there are now two of them
----------------------------------------------------
The legend draws ``nested n_max = 2`` and ``nested n_max = 3``.  For the first
delivered version of this figure that wording was **false**: every crystal point
went through ``VMC.run``, which builds a fresh Gaussian-overlap seed at its own
``nmax``, so the three truncations were three independent cold starts.  ``VMC.nest``
-- the real embedding ``theta_opt^(m) -> pad_v -> theta_0^(m+1)`` -- existed and
was correct, but nothing under ``examples/`` or ``figures/`` called it.

That matters because ``wavefunctions/nesting.py``'s one-sidedness argument, the
reason an ``nmax`` ladder is interpretable at all, rests on there being no
initialisation difference between rungs.  A fresh seed per ``nmax`` puts one
back, so an energy change with ``nmax`` could no longer be attributed to the
optimiser.

``--construction`` therefore selects between two families, and they write to
**different directories** so neither can overwrite the other:

  * ``independent`` (default) -- the delivered construction, kept as it was.
  * ``nested`` -- the ladder the legend always claimed:
    ``nmax=1 -> pad -> nmax=2 -> pad -> nmax=3``, each rung consuming the
    *completed* previous rung, with the parent state stored so ``nmax=3`` cannot
    silently rebuild a second ``nmax=2``.

Both are legitimate VMC results.  Keeping both is the point: the difference
between them,

    delta_E_init(rs) = E_(nested) - E_(independent)

is what says whether a corrected Gaussian seed reaches the same basin as a
grown state.  If it does, initialisation is not the systematic behind the
``nmax = 2`` crystal sitting below the published curve; if it does not, finite-SR
basin dependence is.

Where things live
-----------------
    examples/figure_construction/                        this recipe
    results/figure_construction/phase_competition/<slug>/   the numbers
    figures/figure_construction/phase_competition/<slug>/   the picture

``<slug>`` gains the suffix ``__nested`` under ``--construction nested`` and is
unchanged under ``independent``.  So the two families are siblings::

    .../N36__llrot__quick__l01/            the delivered independent numbers
    .../N36__llrot__quick__l01__nested/    the nested ones, plus states/

The delivered store is never archived, rewritten or extended by a nested run --
it is read, as the baseline.  Each nested run also writes its final states under
``states/``, because a child has to grow from the parent's OWN finished run
rather than from a recomputation of it that would be a second trajectory.

BLAS threading, which is not in any source file
-----------------------------------------------
Under ``--construction nested`` the ``n_max = 1`` roots are the same calculation
as the independent family's, so they are compared against them as a control.  On
this machine that comparison is **bit-exact only under pinned BLAS threading**:
the identical point evaluates to ``-37.793807191316347`` with
``OMP/OPENBLAS/MKL_NUM_THREADS=1`` and ``-37.793807191311664`` without, a 4.7e-12
difference from reduction order alone -- and single-threaded it is also the
faster of the two here.  The banner and the metadata therefore record the thread
configuration, the control is a tolerance rather than ``== 0``, and the run is
meant to be launched with those variables pinned.

Each point is cached separately, so a run that is interrupted resumes where it
stopped instead of starting over.  ``--resume`` is on by default; ``--redo``
forces a recomputation.

A figure says which budget made it
----------------------------------
``--budget quick`` is a smoke test, and the figure it writes says so: the words
``QUICK / not publication quality`` are drawn inside both panels, and the same
fact is recorded in ``run_metadata.json`` as ``figure_stamp``.  On the picture,
not only beside it -- a smoke test whose render is visually identical to the
real figure is the one outcome this recipe must not produce.  Only
``reproduction`` is unstamped, so a budget nobody recognises marks the figure
rather than passing as a result.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import sys
import time

import numpy as np

from wigner_vmc import (VMC, NestingIdentityError, __version__, load_budget,
                        resolve_budget_name, theta)
from wigner_vmc.analysis import statistics as st
# The nesting primitives are imported by NAME rather than reached through the
# package, because this module is the one place that has to prove the embedding
# it claims.  `nesting_residual` is the guard that can actually fail;
# `coefficient_identity_deviation` pads its own argument and returns (0, 0) for
# every input, so it is not a check and is not used here.
from wigner_vmc.wavefunctions.nesting import nesting_residual, pad_v
from wigner_vmc.vmc.sampler import sample

# ==========================================================================
# where things live
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))          # .../wigner_vmc_clean

RESULTS = os.path.join(CLEAN, "results", "figure_construction", "phase_competition")
FIGDIR = os.path.join(CLEAN, "figures", "figure_construction", "phase_competition")

# ==========================================================================
# the point selection -- transcribed from the historical figure
# ==========================================================================
#: The ladder's couplings, as kappa = r_s/sqrt(2).  Exactly the historical grid.
KAPPAS_PROTO = (2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 64.0, 80.0)

#: The competition's couplings, and which truncation is measured at each.  The
#: tier names are the figure's own: `conv` is the converged n_max = 1 crystal,
#: `nest`/`nest4` are the crystals grown from it by nesting.
TIER_NMAX = {"conv": 1, "nest": 2, "nest4": 3}

#: THE GRID IS NOW THE HISTORICAL ONE *EXTENDED*, and `HISTORICAL_TIER_RS`
#: below is what it was extended FROM -- the original figure's selection, kept
#: here because the metadata quotes it and a reader is owed the difference.
#:
#: The extension answers one question the historical grid could not: where the
#: crossing starts, which is below r_s = 40 and was not measured at all there.
#:
#:   * 30 to 55, every 2.5, for n_max = 1 AND n_max = 2 -- the window the
#:     competition is decided in, sampled at half the historical spacing;
#:   * below 30, n_max = 1 only: the small-r_s end is where the converged
#:     truncation is the one that runs, and the two nested tiers are still
#:     converging there;
#:   * above 70, n_max = 2 and 3 additionally: that is where the nested rungs
#:     separate and where the crossing's far side is.
TIER_RS = {
    "conv": (25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5,
             50.0, 52.5, 55.0, 60.0, 65.0, 70.0, 75.0, 77.5, 80.0, 85.0, 90.0),
    "nest": (30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5,
             55.0, 60.0, 65.0, 70.0, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0),
    "nest4": (70.0, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0),
}

#: The selection the historical figure was drawn at, verbatim, so the metadata
#: can state what changed rather than only what is.
HISTORICAL_TIER_RS = {
    "conv": (40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 65.0, 75.0, 77.5,
             80.0, 85.0, 90.0),
    "nest": (55.0, 65.0, 75.0, 77.5, 80.0, 85.0, 90.0),
    "nest4": (75.0, 80.0),
}

N_ELECTRONS = 36
DEFAULT_ANSATZ = "ll_rotation"
DEFAULT_BUDGET = "quick"
QUALITY_MARK = "QUICK / not publication quality"

#: The same fact, drawn ON the figure.  Two lines because it lives inside an
#: axes and a single long line would reach across the data; the single-line
#: ``QUALITY_MARK`` above is what the metadata records.
QUALITY_STAMP = "QUICK\nnot publication quality"

#: The liquid's nmax is a TECHNICAL value, not a physical one -- its determinant
#: rows are ``sum_n C[k,n] phi_{k,n}`` with ``v = 0``, and ``c_row(0)`` is the
#: first unit vector, so every higher band enters multiplied by zero.  There is
#: deliberately no ``--liquid-nmax``.
LIQUID_NMAX = 1

#: The historical ladder took, per coupling, the MINIMUM over a grid of Gaussian
#: widths L0 -- a property of that scan's protocol, not of the ansatz.  The
#: clean budget carries its own width grid (`budget.inits`), and ``--l0-grid``
#: selects over it.  Off by default: one seed per point is what a phase
#: competition needs, and the width scan multiplies the cost by the grid size.
DEFAULT_L0_GRID = False

#: The two constructions.  ``independent`` is the delivered one -- a fresh
#: Gaussian-overlap seed optimised on its own at each truncation.  ``nested``
#: grows each rung from the completed rung below it by exact embedding.  They are
#: different calculations with different numbers, so they never share a store.
CONSTRUCTIONS = ("independent", "nested")

#: The directory suffix per construction.  EMPTY for ``independent``, so the
#: delivered slugs are unchanged to the byte and the delivered store stays where
#: it is -- read as the comparison baseline rather than archived away.
CONSTRUCTION_SUFFIX = {"independent": "", "nested": "__nested"}

#: Where the nested family keeps final states.  A child has to nest from the
#: parent's OWN finished run; without this, recomputing the parent would be the
#: only option and a second trajectory could enter the chain unnoticed.
STATE_DIRNAME = "states"

#: The level-2 probe: a short walk whose configurations are handed to BOTH the
#: parent and the embedded child.  It verifies an identity, so it needs to be
#: exact rather than well averaged -- a handful of sweeps is enough and keeps the
#: check off the critical path.
_PROBE_SWEEPS = 6
_PROBE_EQUIL = 2
_PROBE_SEED = 20260104
_PROBE_SIGMA = 0.6

#: The level-2 gate.  ``Psi_child^(0) = Psi_parent^opt (+) 0`` is an identity, so
#: the tolerance is floating-point association order and not a physical one --
#: deliberately many orders below every error bar in the figure, because a check
#: as loose as the physics measures nothing.
_EMBED_TOL = 1e-9

#: The gate for re-running an ``n_max = 1`` root that the independent store
#: already holds.  Same reasoning, but the scale is set by MEASUREMENT rather
#: than by argument: the identical point run single-threaded reproduces the
#: delivered value to the last bit, and run multi-threaded differs by 4.7e-12
#: because BLAS re-associates its reductions.  So the tolerance has to be above
#: the threading scale and still ten orders below the physics (~4e-2 here).
#:
#: Why this is worth a named constant rather than a bare ``== 0``: a control
#: written as bit-identity is a control that fires on a thread count, which is
#: the kind of false alarm that gets a real leak dismissed as noise later.
_BLAS_TOL = 1e-9

#: The environment variables that decide BLAS thread count.  Recorded, and
#: checked, because they are not part of any source file and yet they move a
#: number at the 12th digit.
_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ==========================================================================
# the request
# ==========================================================================
def _fmt_rs(rs):
    rs = float(rs)
    return f"{rs:.0f}" if abs(rs - round(rs)) < 1e-9 else f"{rs:.3f}"


def _slug_rs(rs):
    rs = float(rs)
    return f"{rs:.0f}" if abs(rs - round(rs)) < 1e-9 else f"{rs:.6g}"


def request_slug(N, ansatz, budget, l0_grid, construction="independent"):
    """The directory name for a request: everything that changes the numbers.

    The couplings and the tiers are FIXED by the figure, so they are not in the
    slug -- they are in the metadata, where the grid is written out.  What is
    here is what a user can change, plus the choices that change the cost by
    orders of magnitude or change the construction, and so must never share a
    directory.

    ``construction`` is appended and is EMPTY for ``independent``, so every
    directory the figure has already written keeps its exact name.  That is not
    tidiness: the independent store is the comparison baseline, and the whole
    exercise is to read it, not to replace it.
    """
    tag = {"ll_rotation": "llrot", "ll_rotation_pinned": "llrot_pinned"}.get(
        str(ansatz), str(ansatz))
    l0 = "l0grid" if l0_grid else "l01"
    suffix = CONSTRUCTION_SUFFIX[_construction(construction)]
    return f"N{N}__{tag}__{budget}__{l0}{suffix}"


def _construction(value):
    """Validate a construction name, rather than letting an unknown one pass.

    An unrecognised token would otherwise get an EMPTY suffix and quietly share
    the independent store -- the one failure mode this separation exists to
    prevent.
    """
    name = str(value)
    if name not in CONSTRUCTION_SUFFIX:
        raise ValueError(f"unknown construction {value!r}; "
                         f"expected one of {CONSTRUCTIONS}")
    return name


def config_key(cfg):
    """A digest of everything that would change a point's number."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def point_key(phase, rs, nmax, init_id, args):
    """The cache key of ONE VMC run."""
    return config_key({
        "phase": phase, "rs": float(rs), "nmax": int(nmax),
        "init_id": int(init_id), "N": int(args.n_electrons),
        "ansatz": args.ansatz, "budget": args.budget,
        "wigner_vmc": __version__, "engine": args.engine,
    })


# ==========================================================================
# running one point
# ==========================================================================
def _record_from_result(phase, rs, nmax, init_id, args, res, seconds):
    """The cache record for one finished run -- one definition, so the
    independent and nested paths cannot report the same run two ways."""
    return {
        "phase": str(phase), "rs": float(rs), "nmax": int(nmax),
        "init_id": int(init_id), "N": int(args.n_electrons),
        "ansatz": str(args.ansatz), "budget": str(args.budget),
        "energy_per_particle": float(res.energy_per_particle),
        "error": float(res.error),
        "acceptance": float(res.acceptance),
        "seconds": float(seconds),
        # The engine's own record, kept raw so a reader can see the run rather
        # than a summary of it.
        "record": {k: v for k, v in dict(res.record).items()
                   if isinstance(v, (int, float, str, bool, type(None)))},
    }


def run_point(phase, rs, nmax, init_id, args, verbose=False, with_state=False):
    """One clean VMC run through the public API.  Returns the record to cache.

    The numbers come from ``RunResult`` -- ``energy_per_particle`` and its
    autocorrelation-corrected ``error`` -- and nothing here recomputes either.

    This is the INDEPENDENT construction: a fresh Gaussian-overlap seed at this
    ``nmax``, optimised on its own.  It is kept exactly as the delivered figure
    left it, so the store it produced stays readable and its numbers stay
    reproducible.

    ``with_state`` returns ``(record, RunState)`` instead.  The nested ladder
    needs the parent's final parameters, and they must be the parameters of the
    run that produced the record rather than of a second run of the same point
    -- so the state is handed back by the run itself rather than rebuilt later.
    """
    t0 = time.time()
    vmc = VMC(N=int(args.n_electrons), rs=float(rs), phase=str(phase),
              nmax=int(nmax), ansatz=str(args.ansatz))
    res = vmc.run(init_id=int(init_id), budget=str(args.budget),
                  verbose=bool(verbose))
    rec = _record_from_result(phase, rs, nmax, init_id, args, res,
                              time.time() - t0)
    return (rec, res.state) if with_state else rec


# ==========================================================================
# the nested construction
# ==========================================================================
# Everything below exists to make one sentence true of the numbers rather than
# of the legend: `n_max = 3` GROWS FROM the `n_max = 2` that was measured, which
# in turn grew from the `n_max = 1`.  The embedding is exact and its residual is
# checked on the vector actually about to be optimised; a final energy that rises
# with `n_max` is an OPTIMISER signal and never a nesting failure.
def state_digest(state):
    """A digest of a state's variational parameters, for the ancestry record.

    Over the raw bytes of ``c`` and ``v``, not a rounded rendering.  The one
    property this is used for is byte identity between the parent a child
    consumed and the parent that was recorded, so rounding it -- the way an
    energy is rounded for display -- would destroy exactly the thing it is for.
    """
    h = hashlib.sha256()
    for name, dtype in (("c", np.float64), ("v", np.complex128)):
        arr = np.ascontiguousarray(getattr(state, name), dtype=dtype)
        h.update(f"{name}{arr.shape}".encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _state_dir(res_dir):
    return os.path.join(res_dir, STATE_DIRNAME)


def state_path(res_dir, key):
    return os.path.join(_state_dir(res_dir), f"{key}.pkl")


def save_state(res_dir, key, state):
    """Persist a finished state so a child nests from the parent's OWN run.

    ``snaps`` is dropped: it is about a megabyte per state, belongs to the
    measurement rather than to the parameters, and ``nest`` reads only ``c`` and
    ``v`` (``USER_GUIDE.md``, the manual-pickling recipe).  Written to a temp name
    and moved, so a killed run cannot leave a half-written pickle for a later
    child to nest from.
    """
    os.makedirs(_state_dir(res_dir), exist_ok=True)
    slim = copy.copy(state)
    slim.snaps = None
    path = state_path(res_dir, key)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(slim, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_state(res_dir, key):
    path = state_path(res_dir, key)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


class _NestParent:
    """The shim ``VMC.nest`` expects: a finished state and an ``init_id``.

    ``nest`` reads exactly ``parent.state.{nmax, phase, v, c}`` and
    ``parent.config.init_id``, and nothing else.  Written out as named attributes
    rather than assembled as a bare namespace so that if ``nest`` ever starts
    reading something more, the missing attribute raises here instead of
    silently arriving as ``None``.
    """

    def __init__(self, state, init_id):
        self.state = state
        self.config = _ConfigRef(int(init_id))


class _ConfigRef:
    def __init__(self, init_id):
        self.init_id = init_id


def embedding_report(rs, nmax, parent_state, n_electrons, ansatz):
    """Levels 1 and 2: the child's START is the parent's FINISH.

    Level 1 needs no walk.  ``pad_v`` is the embedding, and ``nesting_residual``
    says whether the vector about to be optimised really is the parent's
    wavefunction with one level appended as an exact zero.

    Level 2 is the half that could be wrong while level 1 still reads zero.  A
    residual of zero says the ORBITALS were padded; it does not by itself say
    that the child's determinant equals the parent's, because the two are built
    from parameter vectors of different lengths.  So both wavefunctions are
    evaluated on the SAME configurations from the SAME walk with the SAME
    estimator -- which is what ``Psi_child^(0) = Psi_parent^opt (+) 0`` actually
    asserts.

    Returns the measurements rather than a verdict, so the caller records them
    and a test can pin them; ``ok`` is the conjunction, and the caller refuses to
    optimise when it is false.  This is a check on the EMBEDDING.  It says
    nothing about whether the child's optimisation then converged -- that is
    level 3, and it is reported, never asserted.
    """
    nmax = int(nmax)
    v_parent = np.asarray(parent_state.v, complex)
    parent_vmc = VMC(N=int(n_electrons), rs=float(rs), phase="crystal",
                     nmax=nmax - 1, ansatz=str(ansatz))
    child_vmc = VMC(N=int(n_electrons), rs=float(rs), phase="crystal",
                    nmax=nmax, ansatz=str(ansatz))
    v_child = pad_v(v_parent, child_vmc.n_bands)
    head, tail = nesting_residual(v_parent, v_child)

    c = np.asarray(parent_state.c, float)
    wf_parent = parent_vmc.maker()(theta(c, v_parent))
    wf_child = child_vmc.maker()(theta(c, v_child))

    snaps, _, _ = sample(wf_parent, parent_vmc.crystal_R0(),
                         nsweep=_PROBE_SWEEPS, sigma=_PROBE_SIGMA,
                         rng=np.random.default_rng(_PROBE_SEED),
                         snapshot_every=1, equil=_PROBE_EQUIL, target_acc=0.4)
    d_t = d_v = 0.0
    for S in snaps:
        # `local_energy` returns complex for a complex determinant; only the real
        # part is the energy, and taking it here rather than inside `float()`
        # keeps a non-zero imaginary part from being discarded in silence.
        t_p, v_p, _ = wf_parent.local_energy(wf_parent.build(S))
        t_c, v_c, _ = wf_child.local_energy(wf_child.build(S))
        d_t = max(d_t, abs(float(np.real(t_p)) - float(np.real(t_c))))
        d_v = max(d_v, abs(float(np.real(v_p)) - float(np.real(v_c))))
    kappa = float(parent_vmc.kappa)
    return {
        "head": float(head), "tail": float(tail),
        "n_configurations": int(len(snaps)),
        "max_abs_dT": float(d_t), "max_abs_dV": float(d_v),
        # |dE| <= dT + |kappa| dV; kappa is positive here but the bound should
        # not quietly depend on that.
        "max_abs_dE": float(d_t + abs(kappa) * d_v),
        "kappa": kappa,
        "tolerance": float(_EMBED_TOL),
        "ok": bool(head == 0.0 and tail == 0.0
                   and d_t <= _EMBED_TOL and d_v <= _EMBED_TOL),
    }


def run_nested_point(rs, nmax, init_id, parent_rec, parent_state, args,
                     verbose=False):
    """One crystal point grown from its parent by exact embedding.

    The child is optimised from ``theta(parent.c, pad_v(parent.v, nb))`` and from
    nothing else -- no fresh Gaussian width, no re-drawn Jastrow.  That is the
    whole content of the nesting claim, and it is why an energy difference
    between rungs is attributable to the enlarged orbital space.

    Returns ``(record, state)``, the record carrying the full ancestry.
    """
    t0 = time.time()
    nmax = int(nmax)
    if int(parent_state.nmax) != nmax - 1:
        raise ValueError(
            f"the nested chain grows one level at a time: asked for nmax={nmax} "
            f"from a parent at nmax={int(parent_state.nmax)}")

    report = embedding_report(rs, nmax, parent_state, args.n_electrons,
                             args.ansatz)
    if not report["ok"]:
        raise NestingIdentityError(
            f"the nmax={nmax} start at r_s={rs} is not the parent's "
            f"wavefunction: residual ({report['head']:.3e}, {report['tail']:.3e}) "
            f"and max |dE| {report['max_abs_dE']:.3e} over "
            f"{report['n_configurations']} shared configurations (tolerance "
            f"{_EMBED_TOL:g}).  Refusing to optimise: a re-initialised child "
            f"would make any n_max trend unattributable.")

    vmc = VMC(N=int(args.n_electrons), rs=float(rs), phase="crystal",
              nmax=nmax, ansatz=str(args.ansatz))
    res = vmc.nest(_NestParent(parent_state, init_id), new_nmax=nmax,
                   budget=str(args.budget), verbose=bool(verbose))
    rec = _record_from_result("crystal", rs, nmax, init_id, args, res,
                              time.time() - t0)
    nest_info = dict(getattr(res, "nesting", None) or {})

    d_e = float(res.energy_per_particle) - float(parent_rec["energy_per_particle"])
    s_c, s_p = float(res.error), float(parent_rec["error"])
    denom = math.sqrt(s_c ** 2 + s_p ** 2)
    # Level 3.  Variational minimality gives E_(n+1)^min <= E_n^min, but finite
    # stochastic SR gives noisy, non-monotone ESTIMATES, so a small positive dE
    # is expected and is not a failure.  A large one is worth a sentence: it says
    # the child did not exploit the space it was handed.
    z = (d_e / denom) if denom > 0 else None

    rec.update({
        "construction": "nested",
        "parent_nmax": int(parent_state.nmax),
        "parent_rs": float(rs),
        "parent_init_id": int(init_id),
        "parent_key": parent_rec.get("key"),
        "parent_state_digest": state_digest(parent_state),
        "parent_energy": float(parent_rec["energy_per_particle"]),
        # Propagated, not re-derived: nmax = 2 and 3 no longer start from a
        # Gaussian, so the Gaussian origin has to travel with the record.
        "root_initialization": parent_rec.get("root_initialization",
                                              "corrected_gaussian_overlap"),
        "root_L0": parent_rec.get("root_L0"),
        "nesting_head_residual": float(nest_info.get("head", report["head"])),
        "nesting_tail_residual": float(nest_info.get("tail", report["tail"])),
        "embedding_check": report,
        "delta_e_vs_parent": float(d_e),
        "z_vs_parent": None if z is None else float(z),
    })
    if z is not None and z > 3.0:
        print(f"          WARNING nested child optimization failed to exploit "
              f"the enlarged space: dE={d_e:+.9f} ({z:+.2f} sigma)")
    return rec, res.state


class CrystalResolver:
    """The nested ladder's single source of states.

    The failure this has to design out is a child nesting from a DIFFERENT
    parent than the one that was recorded for display::

        root -> nmax2^A              # computed, plotted
        root -> nmax2^B -> nmax3     # a second, independent SR trajectory

    Both are deterministic and both are legitimate runs; they agree only if the
    seed happens to make them.  So there is exactly ONE function producing a
    state at ``(rs, nmax, init_id)``, it memoises, and ``nmax = 3`` is
    structurally unable to see anything but the object ``nmax = 2`` returned.
    ``parent_state_digest`` then pins that in the record, so a reader can check
    the claim instead of trusting the architecture.

    Ordering falls out of the same design: asking for ``nmax = 3`` computes its
    ``nmax = 2`` parent first, so the plan never has to be sorted by rung.
    """

    def __init__(self, args, res_dir, runs):
        self.args = args
        self.res_dir = res_dir
        self.runs = runs
        self._memo = {}
        self._root_L0 = {}

    def state(self, rs, nmax, init_id):
        """``(record, RunState)`` for one crystal point.  Cached if possible."""
        sig = (round(float(rs), 12), int(nmax), int(init_id))
        if sig in self._memo:
            return self._memo[sig]

        key = point_key("crystal", rs, nmax, init_id, self.args)
        rec = self.runs.get(key)
        st = load_state(self.res_dir, key) if rec is not None else None
        if rec is not None and st is not None and not self.args.redo:
            print(f"  cached  construction=nested crystal r_s={_fmt_rs(rs):>7s} "
                  f"nmax={nmax} init={init_id}  "
                  f"E/N={rec['energy_per_particle']:.9f}  "
                  f"[state {state_digest(st)}]")
            self._memo[sig] = (rec, st)
            return self._memo[sig]
        if rec is not None and st is None and not self.args.redo:
            # A record whose state is gone.  Recomputing the parent would give a
            # different trajectory to nest from while the displayed point kept
            # the old number -- the exact discrepancy this class exists to
            # prevent, so it refuses instead of quietly repairing.
            raise RuntimeError(
                f"r_s={rs} nmax={nmax} init={init_id} is cached but its state "
                f"file is missing ({state_path(self.res_dir, key)}).  A child "
                f"cannot nest from a recomputed parent; re-run with --redo.")

        self._memo[sig] = self._compute(rs, nmax, init_id, key, sig)
        return self._memo[sig]

    def _compute(self, rs, nmax, init_id, key, sig):
        if int(nmax) == 1:
            t0 = time.time()
            print(f"  ROOT    crystal r_s={_fmt_rs(rs):>7s} nmax=1 "
                  f"init={init_id} ...", flush=True)
            rec, state = run_point("crystal", rs, 1, init_id, self.args,
                                   verbose=getattr(self.args, "verbose", False),
                                   with_state=True)
            rec["construction"] = "independent"
            rec["root_initialization"] = "corrected_gaussian_overlap"
            rec["root_L0"] = float(self.root_L0(init_id))
            print(f"          done in {time.time() - t0:.1f}s  "
                  f"E/N={rec['energy_per_particle']:.9f} +- {rec['error']:.9f}")
        else:
            parent_rec, parent_state = self.state(rs, int(nmax) - 1, init_id)
            print(f"  NEST    crystal r_s={_fmt_rs(rs):>7s} nmax={nmax} "
                  f"<- nmax={int(nmax) - 1} "
                  f"[parent {state_digest(parent_state)}] ...", flush=True)
            rec, state = run_nested_point(
                rs, nmax, init_id, parent_rec, parent_state, self.args,
                verbose=getattr(self.args, "verbose", False))
        rec["key"] = key
        self.runs[key] = rec
        save_state(self.res_dir, key, state)
        save_store(self.res_dir, self.runs)
        return rec, state

    def root_L0(self, init_id):
        """The Gaussian width the chain's root started from.

        Read from the budget rather than recorded from the run, because it is a
        property of ``init_id`` and the budget and not of the trajectory.
        """
        if init_id not in self._root_L0:
            bud = load_budget(self.args.budget)
            self._root_L0[init_id] = float(bud.width_for(int(init_id)))
        return self._root_L0[init_id]


def _store_path(res_dir):
    return os.path.join(res_dir, "runs.json")


def load_store(res_dir):
    path = _store_path(res_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("runs", {})


def save_store(res_dir, runs):
    os.makedirs(res_dir, exist_ok=True)
    path = _store_path(res_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"schema": "figure_construction/phase_competition/runs/1",
                   "runs": runs}, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)          # atomic: a killed run never leaves a half file


def ensure(runs, phase, rs, nmax, init_id, args, res_dir):
    """The cached record for a point, computing it if it is not there yet.

    ``--redo`` is the only way to recompute a cached point, so a resumed run is
    guaranteed to be a continuation of the same calculation rather than a mix of
    two.

    ``res_dir`` is passed in rather than derived from the arguments, because the
    construction decides the directory and a second derivation here would be a
    second place for the two to disagree.
    """
    key = point_key(phase, rs, nmax, init_id, args)
    if key in runs and not args.redo:
        rec = runs[key]
        print(f"  cached  {phase:7s} r_s={_fmt_rs(rs):>7s} nmax={nmax} "
              f"init={init_id}  E/N={rec['energy_per_particle']:.9f}")
        return rec
    t0 = time.time()
    print(f"  RUN     {phase:7s} r_s={_fmt_rs(rs):>7s} nmax={nmax} "
          f"init={init_id} ...", flush=True)
    rec = run_point(phase, rs, nmax, init_id, args, verbose=getattr(args, 'verbose', False))
    rec["key"] = key
    runs[key] = rec
    save_store(res_dir, runs)
    print(f"          done in {time.time() - t0:.1f}s  "
          f"E/N={rec['energy_per_particle']:.9f} +- {rec['error']:.9f}")
    return rec


def _init_ids(args):
    """Which Gaussian widths to start from at each crystal point.

    One seed by default.  ``--l0-grid`` selects over the budget's own width
    grid, which is the historical ladder's selection rule.
    """
    if not args.l0_grid:
        return (0,)
    bud = load_budget(args.budget)
    return tuple(int(i["init_id"]) for i in bud.inits)


# ==========================================================================
# assembling the two panels
# ==========================================================================
def liquid_record(rec):
    return st.StateRecord(
        workflow="phase_competition", phase="liquid", rs=rec["rs"],
        N=rec["N"], nmax=rec["nmax"], init_id=f"liq{rec['init_id']}",
        production_energy_per_particle=rec["energy_per_particle"],
        production_mc_error=rec["error"], raw=dict(rec))


def crystal_records(recs):
    """The records the comparison consumes, carrying their provenance.

    ``parent_id`` and ``nested`` are declared on ``StateRecord`` and were never
    populated on this path -- which is how a fresh Gaussian seed could be
    labelled nested for as long as it was.  They are set from the record's own
    construction now, so the label a reader sees and the run that produced it
    come from one source.  Safe to set: ``build_scan``/``check_coverage`` are the
    only consumers of these fields and neither is called by this recipe.
    """
    return [st.StateRecord(
        workflow="phase_competition", phase="crystal", rs=r["rs"], N=r["N"],
        nmax=r["nmax"], init_id=f"cry{r['init_id']}",
        parent_id=r.get("parent_key"),
        nested=bool(r.get("construction") == "nested"),
        production_energy_per_particle=r["energy_per_particle"],
        production_mc_error=r["error"], raw=dict(r)) for r in recs]


def required_crystal_rs():
    """Couplings that must exist as a crystal at EVERY truncation in the plan.

    ``TIER_RS["nest"]`` measures 82.5 and 87.5 with no ``n_max = 1`` crystal
    below them, and a nested child needs a parent at the SAME coupling.  The
    legacy scan solved this the same way, by widening its converged bracket
    rather than by reaching across couplings.

    A computational parent is not a scientific point.  Required and displayed are
    different sets: the plotted ``conv`` series stays exactly ``TIER_RS["conv"]``,
    so the extra couplings are run and then not drawn.
    """
    return sorted(set(TIER_RS["conv"]) | set(TIER_RS["nest"])
                  | set(TIER_RS["nest4"]))


def build(args, runs, res_dir):
    """Every point the two panels need.  Returns (ladder, competition, plan)."""
    inits = _init_ids(args)
    nested = _construction(args.construction) == "nested"
    liquid_rs = sorted({round(k * math.sqrt(2.0), 12) for k in KAPPAS_PROTO}
                       | set(TIER_RS["conv"]) | set(TIER_RS["nest"])
                       | set(TIER_RS["nest4"]))
    # Under `independent` this is the displayed grid and nothing more: there is
    # no child to serve, and running 82.5/87.5 at n_max = 1 would add records to
    # the delivered store for nobody.
    crystal_root_rs = required_crystal_rs() if nested else list(TIER_RS["conv"])
    plan = [("liquid", rs, LIQUID_NMAX, 0) for rs in liquid_rs]
    for k in KAPPAS_PROTO:
        rs = round(k * math.sqrt(2.0), 12)
        plan += [("crystal", rs, 1, i) for i in inits]
    for rs in crystal_root_rs:
        plan += [("crystal", float(rs), 1, i) for i in inits]
    for tier, rss in TIER_RS.items():
        for rs in rss:
            plan += [("crystal", float(rs), TIER_NMAX[tier], i) for i in inits]

    # Deduplicate, keeping the order the plan was built in.
    seen, ordered = set(), []
    for phase, rs, nmax, init_id in plan:
        sig = (phase, float(rs), int(nmax), int(init_id))
        if sig not in seen:
            seen.add(sig)
            ordered.append(sig)

    parent_only = sorted(set(crystal_root_rs) - set(TIER_RS["conv"]))
    print(f"  plan: {len(ordered)} VMC runs "
          f"({sum(1 for p in ordered if p[0] == 'liquid')} liquid, "
          f"{sum(1 for p in ordered if p[0] == 'crystal')} crystal)")
    print(f"        construction: {_construction(args.construction)}")
    print(f"        crystal widths per point: {list(inits)}")
    if parent_only:
        print(f"        run as parents but NOT drawn: r_s = "
              f"{', '.join(_fmt_rs(r) for r in parent_only)} (n_max = 1)")
    print()

    resolver = CrystalResolver(args, res_dir, runs) if nested else None
    for phase, rs, nmax, init_id in ordered:
        if resolver is not None and phase == "crystal":
            resolver.state(rs, nmax, init_id)
        else:
            ensure(runs, phase, rs, nmax, init_id, args, res_dir)

    def get(phase, rs, nmax, init_id):
        return runs[point_key(phase, rs, nmax, init_id, args)]

    # ---- the ladder -----------------------------------------------------
    ladder = {"rs": [], "e_liq": [], "s_liq": [], "e_cry": [], "s_cry": [],
              "l0": []}
    for k in KAPPAS_PROTO:
        rs = round(k * math.sqrt(2.0), 12)
        liq = get("liquid", rs, LIQUID_NMAX, 0)
        crys = [get("crystal", rs, 1, i) for i in inits]
        # The historical ladder's rule: the width that minimised the energy.
        best = min(crys, key=lambda r: r["energy_per_particle"])
        ladder["rs"].append(rs)
        ladder["e_liq"].append(liq["energy_per_particle"])
        ladder["s_liq"].append(liq["error"])
        ladder["e_cry"].append(best["energy_per_particle"])
        ladder["s_cry"].append(best["error"])
        ladder["l0"].append(best["init_id"])

    # ---- the competition ------------------------------------------------
    tiers = {}
    for tier, rss in TIER_RS.items():
        rs_col, de_col, s_col, rows = [], [], [], []
        for rs in rss:
            liq = get("liquid", float(rs), LIQUID_NMAX, 0)
            crys = [get("crystal", float(rs), TIER_NMAX[tier], i)
                    for i in inits]
            pt = st.compare_phases(crystal_records(crys), liquid_record(liq),
                                   tier=tier)
            rs_col.append(float(rs))
            de_col.append(float(pt.delta_e))
            s_col.append(float(pt.sigma_total))
            rows.append({"rs": float(rs), "tier": tier, "n_max": pt.n_max,
                         "n_states": pt.n_states, "energy_crystal": pt.energy_crystal,
                         "energy_liquid": pt.energy_liquid, "delta_e": pt.delta_e,
                         "sigma_mc": pt.sigma_mc, "sigma_init": pt.sigma_init,
                         "sigma_total": pt.sigma_total, "z": pt.z,
                         "seed_ids": pt.seed_ids})
            print(f"    {tier:6s} r_s={_fmt_rs(rs):>7s}  "
                  f"dE={pt.delta_e:+.9f} +- {pt.sigma_total:.9f}  "
                  f"z={pt.z if pt.z is None else round(pt.z, 2)}")
        tiers[tier] = (np.asarray(rs_col), np.asarray(de_col), np.asarray(s_col))
        tiers[tier + "_rows"] = rows
    return ladder, tiers


# ==========================================================================
# nested versus independent -- the comparison this construction exists for
# ==========================================================================
def _rel(path):
    """``os.path.relpath`` that survives a different drive.

    On Windows a path on another mount has no relative form and ``relpath``
    raises rather than returning something odd.  A recipe whose output can be
    relocated must not die of where its input happens to live, so the absolute
    path is the fallback -- less pretty, still true.
    """
    try:
        return os.path.relpath(path, CLEAN)
    except ValueError:
        return path


def independent_dir(args):
    """The delivered store's directory, DERIVED and never written to.

    Derived from the same ``request_slug`` the running construction uses, so the
    two families cannot drift apart into different names; and this module only
    ever reads it.
    """
    slug = request_slug(args.n_electrons, args.ansatz, args.budget,
                        args.l0_grid, "independent")
    return os.path.join(RESULTS, slug)


def blas_env():
    """The BLAS thread configuration, as the RUN will see it.

    Not a curiosity.  On this machine the same crystal point evaluates to
    ``-37.793807191316347`` single-threaded and ``-37.793807191311664``
    multi-threaded -- a 4.7e-12 difference from reduction order alone, which is
    small against the physics and enormous against a bit-identity check.  A
    baseline store is therefore only a bit-level baseline against the threading
    it was made with, and this is what says whether that holds.
    """
    env = {v: os.environ.get(v) for v in _BLAS_VARS}
    env["numpy"] = np.__version__
    return env


def _threading_is_pinned(env=None):
    """True when every BLAS thread variable is explicitly set.

    ``OMP_NUM_THREADS`` alone is not enough: OpenBLAS and MKL read their own
    variables first and ignore OMP when they are present.
    """
    env = blas_env() if env is None else env
    return all(env.get(v) for v in _BLAS_VARS[:3])


def _delta_e_column(store, tier, args, inits):
    """``(rs, delta_E, sigma)`` for one tier, rebuilt from a store.

    The same estimator the figure uses, so the two columns are comparable by
    construction rather than by hope.  A coupling missing from the store is
    skipped rather than guessed.
    """
    nmax = TIER_NMAX[tier]
    rs_col, de_col, s_col = [], [], []
    for rs in TIER_RS[tier]:
        liq = store.get(point_key("liquid", float(rs), LIQUID_NMAX, 0, args))
        crys = [store.get(point_key("crystal", float(rs), nmax, i, args))
                for i in inits]
        if liq is None or any(c is None for c in crys):
            continue
        pt = st.compare_phases(crystal_records(crys), liquid_record(liq),
                               tier=tier)
        rs_col.append(float(rs))
        de_col.append(float(pt.delta_e))
        s_col.append(float(pt.sigma_total))
    return np.asarray(rs_col, float), np.asarray(de_col, float), np.asarray(s_col, float)


def crossing_report(rs, de, sigma):
    """Where a delta_E curve changes sign, with each side's resolution.

    A bracket, not a critical coupling: the curve is measured at discrete
    couplings and the zero is between two of them, so the interpolated value is
    a reading of the line the figure draws and not a measurement of the physics.
    Both endpoint z-scores travel with it, because a crossing bracketed by two
    unresolvable points is a crossing of noise.
    """
    rs = np.asarray(rs, float)
    de = np.asarray(de, float)
    sigma = np.asarray(sigma, float)
    out = []
    for j in range(max(0, len(rs) - 1)):
        lo, hi = float(de[j]), float(de[j + 1])
        if lo == 0.0 or hi == 0.0 or (lo > 0) == (hi > 0):
            continue
        out.append({
            "between": [float(rs[j]), float(rs[j + 1])],
            "rs_star": float(rs[j] + (rs[j + 1] - rs[j]) * (-lo) / (hi - lo)),
            "delta_at_lo": lo, "delta_at_hi": hi,
            "z_lo": float(lo / sigma[j]) if sigma[j] > 0 else None,
            "z_hi": float(hi / sigma[j + 1]) if sigma[j + 1] > 0 else None,
        })
    return out


def compare_with_independent(args, res_dir):
    """``delta_E_init(rs) = E_(nested) - E_(independent)``, and the crossings.

    Three things, in increasing order of interest:

    * a CONTROL.  The ``n_max = 1`` roots are the same calculation in both
      families -- same config, same seed, same budget -- so their energies must
      agree to the last bit.  Anything else means the nested run changed
      something it does not own, and the rest of the table would be unreadable.
    * the measurement.  Per coupling and per rung, the energy of the grown state
      minus the energy of the independently seeded one, with the sigma of the
      difference.  Small means a corrected Gaussian reaches the same basin as a
      grown state; large means it does not and finite-SR basin dependence is in
      the figure.
    * the crossings.  Each family's own sign change, so the two can be compared
      where the figure makes its claim.
    """
    nested = load_store(res_dir)
    indep = load_store(independent_dir(args))
    inits = _init_ids(args)
    out = {"independent_dir": _rel(independent_dir(args)),
           "nested_dir": _rel(res_dir),
           "available": bool(indep)}
    if not indep:
        out["reason"] = ("no independent store to compare against at "
                         + out["independent_dir"])
        return out

    ctrl_n, ctrl_i = [], []
    for rs in TIER_RS["conv"]:
        for i in inits:
            key = point_key("crystal", float(rs), 1, i, args)
            if key in nested and key in indep:
                ctrl_n.append(float(nested[key]["energy_per_particle"]))
                ctrl_i.append(float(indep[key]["energy_per_particle"]))
    ctrl = np.abs(np.asarray(ctrl_n, float) - np.asarray(ctrl_i, float))
    worst = float(ctrl.max()) if len(ctrl) else None
    out["blas_env"] = blas_env()
    out["control_nmax1"] = {
        "tier": "conv", "n": int(len(ctrl)),
        "max_abs_delta": worst,
        "bit_identical": bool(len(ctrl) and ctrl.max() == 0.0),
        "within_tolerance": bool(len(ctrl) and ctrl.max() <= _BLAS_TOL),
        "tolerance": float(_BLAS_TOL),
        "threading_pinned": bool(_threading_is_pinned()),
        "note": ("the nmax=1 roots are the same calculation in both families, so "
                 "under pinned BLAS threading they must agree to the last bit.  "
                 "A difference at ~1e-12 with unpinned threading is reduction "
                 "order, not a leaked parameter; anything larger is a leak."),
    }

    out["delta_E_init"] = {}
    for tier in ("nest", "nest4"):
        nmax = TIER_NMAX[tier]
        rows = []
        for rs in TIER_RS[tier]:
            dn, di = [], []
            for i in inits:
                key = point_key("crystal", float(rs), nmax, i, args)
                if key in nested and key in indep:
                    dn.append(nested[key])
                    di.append(indep[key])
            if not dn:
                continue
            e_n = float(np.mean([r["energy_per_particle"] for r in dn]))
            e_i = float(np.mean([r["energy_per_particle"] for r in di]))
            s_n = float(np.mean([r["error"] for r in dn]))
            s_i = float(np.mean([r["error"] for r in di]))
            d = e_n - e_i
            denom = math.sqrt(s_n ** 2 + s_i ** 2)
            rows.append({"rs": float(rs), "n_max": nmax, "n": int(len(dn)),
                         "energy_nested": e_n, "energy_independent": e_i,
                         "delta": d,
                         "sigma_nested": s_n, "sigma_independent": s_i,
                         "sigma": denom,
                         "z": (d / denom) if denom > 0 else None})
        out["delta_E_init"][tier] = rows

    out["crossings"] = {}
    for tier in ("conv", "nest", "nest4"):
        rs_n, de_n, s_n = _delta_e_column(nested, tier, args, inits)
        rs_i, de_i, s_i = _delta_e_column(indep, tier, args, inits)
        out["crossings"][tier] = {
            "nested": crossing_report(rs_n, de_n, s_n),
            "independent": crossing_report(rs_i, de_i, s_i),
            "n_points": [int(len(rs_n)), int(len(rs_i))],
        }
    return out


def print_comparison(cmp_):
    """The table a reader needs to decide whether to run the reproduction."""
    print()
    print("=" * 78)
    print("nested vs independent   delta_E_init = E_nested - E_independent")
    print("=" * 78)
    if not cmp_.get("available"):
        print(f"  no independent store at {cmp_['independent_dir']}")
        print("  -- nothing to compare against; the nested figure stands alone.")
        return
    c = cmp_["control_nmax1"]
    if c["bit_identical"]:
        verdict = "OK (bit-identical)"
    elif c["within_tolerance"] and not c["threading_pinned"]:
        verdict = ("OK, but not bit-identical -- UNPINNED BLAS THREADING.  Set "
                   "OMP/OPENBLAS/MKL_NUM_THREADS to reproduce the baseline exactly.")
    elif c["within_tolerance"]:
        verdict = "*** within tolerance but threading IS pinned: investigate ***"
    else:
        verdict = "*** LEAK ***"
    print(f"  control  n_max=1 roots: {c['n']} points, max |delta| = "
          f"{c['max_abs_delta']:.3e}   {verdict}")
    print(f"           BLAS env: "
          f"{', '.join(f'{k}={v}' for k, v in cmp_['blas_env'].items())}")
    for tier in ("nest", "nest4"):
        rows = cmp_["delta_E_init"].get(tier, [])
        if not rows:
            continue
        print(f"\n  {tier}  (n_max = {rows[0]['n_max']}, {rows[0]['n']} seed(s))")
        print(f"    {'r_s':>7s}  {'nested':>16s}  {'independent':>16s}  "
              f"{'delta':>13s}  {'sigma':>11s}  {'z':>7s}")
        for r in rows:
            print(f"    {_fmt_rs(r['rs']):>7s}  {r['energy_nested']:>16.9f}  "
                  f"{r['energy_independent']:>16.9f}  {r['delta']:>+13.9f}  "
                  f"{r['sigma']:>11.9f}  "
                  f"{(r['z'] if r['z'] is None else round(r['z'], 2)):>7}")
    print("\n  crossings of delta_E (crystal - liquid)")
    for tier in ("conv", "nest", "nest4"):
        blk = cmp_["crossings"][tier]
        for name in ("nested", "independent"):
            if not blk[name]:
                print(f"    {tier:6s} {name:12s} none in the measured range")
                continue
            for x in blk[name]:
                print(f"    {tier:6s} {name:12s} "
                      f"r_s* = {x['rs_star']:.3f} between "
                      f"{_fmt_rs(x['between'][0])} and {_fmt_rs(x['between'][1])}"
                      f"   z = {x['z_lo']:+.2f} / {x['z_hi']:+.2f}")
    print("=" * 78)


# ==========================================================================
# the picture
# ==========================================================================
def de_window(tier_rs=None):
    """The delta_E panel's ``(xlo, xhi)``, derived from the grid it has to show.

    One unit wider than the measured couplings on each side, so every point is
    inside the frame with a margin rather than on the spine.

    Derived, and not two numbers, because the two numbers the renderer used to
    carry -- 44 and 91 -- were chosen for the historical grid and are a second
    copy of somebody else's scan living inside a shared figure module.  That
    copy was not careless: the comment at ``ll_rotation.py:72`` records that it
    deliberately excludes ``conv`` at r_s = 40 and 42.5, because the panel's
    subject is the crossing and every coupling that carries it lies at 45 or
    above -- so the delivered figure measured 13 ``conv`` points, recorded 13 in
    its metadata, and drew 11, intentionally.

    Intentional is not the same as safe.  A literal that is right for one grid
    is silently wrong for the next, and the failure mode is not a crash: the
    extra couplings fall outside the frame while the axes still look complete.
    Under the extended grid those same literals would have dropped the whole
    small-r_s end that the extension exists to show.  Hence the derivation, and
    hence the recipe, not the module, deciding how wide its own panel must be.

    ``tier_rs`` defaults to this recipe's grid.  It is a parameter so the
    derivation can be tested against a grid that is not the current one -- a
    function tested only on its present output is a function whose output is all
    anyone knows about it.
    """
    grid = TIER_RS if tier_rs is None else tier_rs
    lo = min(min(v) for v in grid.values())
    hi = max(max(v) for v in grid.values())
    return (float(lo) - 1.0, float(hi) + 1.0)


def figure_stamp(budget):
    """The mark a non-production figure carries, or None for the real one.

    ONE definition, read by both the drawing and the metadata, so the picture and
    the record of it cannot disagree about whether the picture is a result.  The
    comparison is against ``reproduction`` rather than for ``quick`` so an
    unrecognised budget stamps: over-marking a smoke test is untidy, and
    under-marking one is the failure this exists to prevent.
    """
    return None if str(budget) == "reproduction" else QUALITY_STAMP


def write_figures(fig_dir, ladder, tiers, budget=DEFAULT_BUDGET):
    from wigner_vmc.figures import ll_rotation as L

    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "vmc_energy_phase_competition.png")
    L.energy_phase_competition(
        np.asarray(ladder["rs"]), np.asarray(ladder["e_liq"]),
        np.asarray(ladder["s_liq"]), np.asarray(ladder["e_cry"]),
        np.asarray(ladder["s_cry"]),
        {k: v for k, v in tiers.items() if not k.endswith("_rows")},
        ll_rot=None, path=path, dpi=170, stamp=figure_stamp(budget),
        window=de_window())
    return [path]


def nesting_provenance(runs):
    """The ancestry of every nested point in the store, as a table.

    One entry per grown state, carrying the parent it consumed (coupling, rung,
    seed, cache key and a byte digest of the parent's parameters), the exact
    nesting residuals, and the energy change against that parent.  The digest is
    what lets a reader check that the ``n_max = 3`` in a figure consumed the
    ``n_max = 2`` that was recorded rather than a second trajectory that happened
    to look similar.
    """
    fields = ("rs", "nmax", "init_id", "parent_nmax", "parent_rs",
              "parent_init_id", "parent_key", "parent_state_digest",
              "parent_energy", "root_initialization", "root_L0",
              "nesting_head_residual", "nesting_tail_residual",
              "delta_e_vs_parent", "z_vs_parent")
    out = []
    for r in runs.values():
        if r.get("construction") != "nested":
            continue
        out.append({k: r.get(k) for k in fields})
    out.sort(key=lambda d: (float(d.get("rs") or 0.0), int(d.get("nmax") or 0)))
    return out


def _embedding_summary(provenance):
    """The worst level-1 residual and level-2 deviation over all nested points.

    Two numbers, because the claim has two halves: the orbitals were padded
    exactly (residual), and the padded child really is the same wavefunction on
    the configurations it will be optimised on (deviation).
    """
    if not provenance:
        return None
    heads = [abs(float(p["nesting_head_residual"] or 0.0)) for p in provenance]
    tails = [abs(float(p["nesting_tail_residual"] or 0.0)) for p in provenance]
    return {"nested_points": len(provenance),
            "max_abs_head_residual": max(heads),
            "max_abs_tail_residual": max(tails),
            "tolerance": float(_EMBED_TOL),
            "exact": bool(max(heads) == 0.0 and max(tails) == 0.0)}


def write_metadata(results_dir, written, ladder, tiers, args, runs=None,
                   comparison=None):
    construction = _construction(args.construction)
    meta = {
        "run_metadata_schema": "figure_construction/phase_competition/1",
        "recipe": "examples/figure_construction/phase_competition.py",
        "wigner_vmc_version": __version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "budget": args.budget,
        "quality": QUALITY_MARK if str(args.budget).startswith("quick") else args.budget,
        "figure_stamp": figure_stamp(args.budget),
        "N": int(args.n_electrons),
        "ansatz": str(args.ansatz),
        "l0_grid": bool(args.l0_grid),
        "crystal_widths": list(_init_ids(args)),
        "construction": construction,
        "construction_slug_suffix": CONSTRUCTION_SUFFIX[construction],
        "construction_contract": (
            "independent: every crystal point is a fresh corrected "
            "Gaussian-overlap seed optimised on its own at that n_max.  "
            "nested: n_max grows one level at a time from the COMPLETED rung "
            "below it, theta_child_0 = theta(parent.c, pad_v(parent.v, nb)), "
            "with the embedding checked on the vector about to be optimised "
            "(nesting residual exactly zero) and on shared configurations "
            "(E_child_0 == E_parent_final).  An energy that RISES with n_max is "
            "an optimiser/convergence signal and is recorded as z_vs_parent, "
            "never as a nesting failure and never as a test."),
        "construction_scope": (
            "the two families write to different directories and share no "
            "numbers; the independent store this compares against is READ, "
            "never rewritten by a nested run"),
        "blas_env": blas_env(),
        "blas_threading_pinned": bool(_threading_is_pinned()),
        "point_selection": {
            "ladder_kappa": list(KAPPAS_PROTO),
            "competition_tiers": {k: list(v) for k, v in TIER_RS.items()},
            "tier_nmax": dict(TIER_NMAX),
            "delta_e_window": list(de_window()),
            # The original figure's grid, kept here because the README used to
            # promise "the same point selection as the original" and that promise
            # is now "the original's grid, extended".  A reader can see exactly
            # what was added instead of having to diff two versions of a source
            # file to find out.
            "historical_tiers": {k: list(v) for k, v in HISTORICAL_TIER_RS.items()},
            "source": ("the historical figure's own scan grids, EXTENDED -- see "
                       "historical_tiers for the original selection and the module "
                       "comment on TIER_RS for what was added and why.  The ladder "
                       "couplings are the original's, unchanged."),
        },
        "state_of_the_historical_ladder": (
            "the historical ladder is the GAUSSIAN crystal; this one is the "
            "implemented ll_rotation ansatz on the same couplings.  Different "
            "wavefunction families -- see DIVERGENCE_fig06_crystal.md.  The "
            "lower panel is liquid-versus-crystal WITHIN this recipe, which is "
            "the comparison phase competition needs."),
        "why_recomputed": (
            "the historical delta_E points came from LL-rotation crystal records "
            "produced with the reversed complex conjugation at vmc/sr.py:219, "
            "which makes the first shell uniaxial.  Every energy here is a VMC "
            "run this recipe started; no legacy JSON is read."),
        "legacy_json_reads": [],
        "ladder": {k: [float(x) for x in v] for k, v in ladder.items()},
        "competition": {k: v for k, v in tiers.items() if k.endswith("_rows")},
        "figures": [_rel(p) for p in written],
    }
    if runs is not None:
        provenance = nesting_provenance(runs)
        meta["nesting"] = {
            "embedding": _embedding_summary(provenance),
            "ancestry": provenance,
            # The prohibition, recorded beside the numbers it governs: the
            # energy a nested rung ends at is not constrained to fall, so no
            # reader and no test may treat a rise as an error.
            "monotonicity_is_not_a_test": (
                "exact nesting makes the child's START equal to the parent's "
                "FINISH; it says nothing about where the child's own SR stops. "
                "A positive delta_e_vs_parent inside a few sigma is expected "
                "and is not a failure."),
        }
    if comparison is not None:
        meta["nested_vs_independent"] = comparison
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path


# ==========================================================================
# the command line
# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Rebuild the phase-competition figure -- the absolute ladder "
                    "and the liquid/crystal delta_E -- from clean VMC runs at the "
                    "historical figure's point selection, extended around it.")
    p.add_argument("--budget", default=DEFAULT_BUDGET,
                   help="quick (smoke test, minutes) | full (the real statistics). "
                        "'full' is an alias for the 'reproduction' config and that "
                        "spelling is accepted too.")
    p.add_argument("--n-electrons", type=int, default=N_ELECTRONS, metavar="N",
                   help=f"electrons.  Default {N_ELECTRONS}, the size every scan "
                        f"this figure summarises was run at.")
    p.add_argument("--ansatz", default=DEFAULT_ANSATZ,
                   choices=("ll_rotation", "ll_rotation_pinned"),
                   help="crystal orbital ansatz.  Default ll_rotation; the historical "
                        "ladder's Gaussian crystal is not implemented by this package "
                        "and is not offered.")
    p.add_argument("--construction", default=CONSTRUCTIONS[0],
                   choices=CONSTRUCTIONS,
                   help="how each crystal point is started.  'independent' "
                        "(default) gives every point its own corrected "
                        "Gaussian-overlap seed at its own n_max -- the delivered "
                        "construction, and the baseline.  'nested' grows each "
                        "n_max from the completed rung below it by exact "
                        "embedding, which is what the figure's 'nested' labels "
                        "have always claimed.  The two never share a directory "
                        "or a number.")
    p.add_argument("--l0-grid", action="store_true",
                   help="select the starting Gaussian width over the budget's whole "
                        "width grid at each crystal point (the historical ladder's "
                        "rule) instead of one seed.  Multiplies the crystal cost by "
                        "the grid size.")
    p.add_argument("--redo", action="store_true",
                   help="recompute every point instead of reusing the cache.  Off by "
                        "default so an interrupted campaign resumes rather than "
                        "restarts.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Resolve the user-facing budget alias ONCE, before anything reads it, so
    # that `--budget full` and `--budget reproduction` are one calculation with
    # one slug and one output directory.
    args.budget = resolve_budget_name(args.budget)
    args.engine = "clean"
    args.verbose = not args.quiet
    construction = _construction(args.construction)
    t0 = time.time()

    # Refused here rather than inside VMC.nest, so the reason arrives before an
    # hour of liquid runs rather than after them.
    if construction == "nested" and str(args.ansatz) != "ll_rotation":
        print(f"--construction nested grows the ORBITAL space, so it cannot use "
              f"the {args.ansatz!r} ansatz: those orbitals are pinned at the "
              f"Gaussian seed and are not variational.  Use ll_rotation.")
        return 2

    bud = load_budget(args.budget)
    slug = request_slug(args.n_electrons, args.ansatz, args.budget, args.l0_grid,
                        construction)
    res_dir = os.path.join(RESULTS, slug)
    fig_dir = os.path.join(FIGDIR, slug)

    print("phase_competition -- clean VMC, from scratch   "
          f"(wigner_vmc {__version__})")
    print("=" * 78)
    print(f"  budget           {bud.name}   ({bud.provenance})")
    print(f"  N                {args.n_electrons}")
    print(f"  crystal ansatz   {args.ansatz}")
    print(f"  construction     {construction}"
          + ("   (each n_max grown from the completed rung below it)"
             if construction == "nested" else
             "   (every point its own Gaussian-overlap seed)"))
    pinned = _threading_is_pinned()
    print(f"  BLAS threads     " + ("pinned: " if pinned else "NOT pinned: ")
          + ", ".join(f"{k}={v}" for k, v in blas_env().items()))
    if not pinned:
        print("                   the same point evaluates 4.7e-12 apart between "
              "thread settings,\n"
              "                   so an existing store is only a bit-level "
              "baseline under its own\n"
              "                   threading.  For an exact nmax=1 control set "
              "OMP/OPENBLAS/MKL_NUM_THREADS.")
    print(f"  ladder couplings {len(KAPPAS_PROTO)} kappa values")
    print(f"  competition      {sum(len(v) for v in TIER_RS.values())} (r_s, tier) points")
    print(f"  delta_E window   r_s = {de_window()[0]:g} .. {de_window()[1]:g}")
    print(f"  results          {_rel(res_dir)}")
    print(f"  figures          {_rel(fig_dir)}")
    print("=" * 78)
    print()

    runs = load_store(res_dir)
    ladder, tiers = build(args, runs, res_dir)

    # The comparison is a READ of the other family's store, so it runs after the
    # numbers exist and before the picture is drawn -- and it never writes to
    # that store.
    comparison = compare_with_independent(args, res_dir) if construction == "nested" else None
    if comparison is not None:
        print_comparison(comparison)

    written = write_figures(fig_dir, ladder, tiers, args.budget)
    meta = write_metadata(res_dir, written, ladder, tiers, args, runs=runs,
                          comparison=comparison)

    print()
    print("=" * 78)
    print(f"  {len(runs)} runs cached in {_rel(_store_path(res_dir))}")
    for path in written:
        print(f"  figure  {_rel(path)}")
    print(f"  metadata {_rel(meta)}")
    print(f"  total {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
