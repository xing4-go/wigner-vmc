"""The one VMC workflow, and the only place a run is assembled.

Everything above this file -- the CLI, the `examples/` scripts, the VS Code
launch configurations, a Jupyter cell -- calls down into here and implements no
part of the workflow itself.  That layering is a testable constraint, not a
slogan: ``tests/test_api_contract.py`` asserts the scripts import none of
``sampler``/``measure``/``sr``/``coulomb``/``local_energy``.

    vmc = VMC(N=36, rs=75, phase="crystal", nmax=1)
    result = vmc.run(init_id=0, budget="quick")
    result.energy_per_particle        # the production number, per electron
    result.energy_total               # == N * energy_per_particle, exactly
    result.error                      # its MC error, AUTOCORRELATION-CORRECTED
    result.acceptance
    result.state.ll_occupation()
    result.state.structure_factor()

    child = vmc.nest(result, new_nmax=2)

------------------------------------------------------------------------------
THE STARTING POINT IS BUILT HERE, AND THE LEGACY BOUNDARY IS NOT CROSSED
------------------------------------------------------------------------------
``run`` computes its own start from the clean package -- ``v0`` from
``sr.gaussian_overlap_seed`` + ``sr.v_from_overlap``, ``c0`` from the analytic
cusp vector ``sr.jastrow_vector`` -- and reads no checkpoint store and no legacy
record.  It therefore cannot write to one either.

That is deliberate and it is also the structural fix for the 2026-10-02 incident,
in which a probe exec'd a notebook and, through ``cached()``, wrote into the
frozen store.  **A run that has no reason to open the frozen tree cannot mutate
it.**  The dependency direction is::

    verified clean core  ->  this API  ->  CLI / VS Code / notebooks

and nothing points back.

**Consequence, stated plainly rather than discovered later:** the API's crystal
starting point is NOT byte-identical to the frozen campaign's, which used a warm
start chain out of the checkpoint store.  These are two different requirements:

* ``scripts/bench_rs75.py`` reproduces the legacy records, so it *must* inject the
  legacy start as explicit data.  That is the Stage 2E path.
* this API gives a user a runnable run from nothing, so it computes its own.

Do not conflate them, and do not score this API against a legacy digest.

------------------------------------------------------------------------------
ENERGY SEMANTICS -- three names, deliberately
------------------------------------------------------------------------------
``result.energy_per_particle``
    The production number, per electron.  This is the only energy a physics
    claim may be made from.
``result.energy_total``
    ``N * energy_per_particle``.  A read-only property, so the factor of ``N``
    appears exactly once in the whole API and cannot drift.
``result.optimization.E_total``
    The SR trace, which is a **total** per step and is NOT the production
    energy: the optimiser is free to end worse than its own minimum.  It lives
    in a separate namespace precisely so it cannot be mistaken for the answer.

There is intentionally **no** ``result.energy``.  It is the one name that does
not say which of the two it is, and every reader would read it as the one they
expected.
"""
from __future__ import annotations

import functools
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from .physics import coulomb as cb
from .physics import geometry as ge
from .physics import landau_levels as llb
from .wavefunctions import jastrow as jw
from .wavefunctions import ll_rotation as lr
from .wavefunctions import nesting as nest
from .vmc import sr
from .vmc.measure import measure_decomposed

__all__ = [
    "VMC", "run_vmc", "RunResult", "RunState", "Optimization", "RunConfig",
    "Budget", "Lattice", "NestingIdentityError", "build_lattice", "resolve",
    "load_budget", "theta", "theta_parts", "NJ", "LUMAX", "LUMAX_LL",
]

#: Jastrow parameter count.  Five, as the frozen campaign used.
NJ = 5

#: Gaussian-overlap image-sum cutoff, on the SUPERCELL.  The campaign's value
#: (``initial_conditions.json``); it is a converged cutoff, not a tuned knob.
LUMAX = 30.0

#: Landau-level Bloch-sum cutoff, on the PRIMITIVE cell.  Chosen by the
#: campaign's own convergence scan against a cutoff-60 reference.  The two
#: cutoffs are different numbers because they are different sums -- see
#: ``bench_rs75.Setup`` for why they must not be interchanged.
LUMAX_LL = 16.0

#: Quadrature points for the overlap seed.  The campaign's value; the default in
#: ``sr.gaussian_overlap_seed`` is 101.
OVERLAP_NUMX = 121

_PKG = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(_PKG)), "configs")


class NestingIdentityError(RuntimeError):
    """Raised when a nested start does not contain its parent exactly.

    Deliberately fatal.  There is no fallback to a fresh initialisation: the
    whole point of the ``nmax`` ladder is that the truncations are nested, and a
    silently re-initialised child would make an energy that rises with ``nmax``
    unattributable -- the reader could not tell an optimiser failing to use the
    larger space from a different starting point.
    """


# ===========================================================================
# geometry
# ===========================================================================
@dataclass(frozen=True, eq=False)
class Lattice:
    """The torus and everything derived from it, built from (N, rs) alone.

    ``eq=False`` because the fields are arrays; equality would be ambiguous and
    identity is the only meaningful comparison.
    """

    N: int
    rs: float
    n_side: int
    A1: np.ndarray
    A2: np.ndarray
    L1: np.ndarray
    L2: np.ndarray
    G1: np.ndarray
    G2: np.ndarray
    mesh: np.ndarray
    sites: np.ndarray
    ll_ai: np.ndarray
    ll_ac: np.ndarray
    ov_ai: np.ndarray
    ov_ac: np.ndarray
    prim_C: np.ndarray
    prim_Ci: np.ndarray
    C: np.ndarray
    Ci: np.ndarray
    HAM: Any

    @property
    def nk(self):
        return int(self.mesh.shape[0])


def build_lattice(N=36, rs=75.0):
    """The r_s/N torus from scratch: primitive cell, supercell, mesh, sites.

    One electron per primitive cell and ``nu = 1``, so the primitive cell area is
    ``2*pi`` in units of ``l_B = 1`` and the supercell is ``sqrt(N) x sqrt(N)``
    cells of it.  This is the same recipe as
    ``physics.hamiltonian.production_geometry`` generalised to any square ``N``;
    ``tests/test_api_contract.py`` asserts the two agree at N = 36.

    The three lattice-vector sets are named for their ROLE because two of them
    are easy to swap and doing so is Blockers A's second defect:
    ``ll_ai/ll_ac`` (primitive, cutoff ``LUMAX_LL``) feed the Landau-level Bloch
    sum; ``ov_ai/ov_ac`` (supercell, cutoff ``LUMAX``) feed the Gaussian overlap
    image sum.  ``prim_C`` and ``C`` are the matching cell matrices.
    """
    N = int(N)
    n_side = int(round(math.sqrt(N)))
    if n_side * n_side != N:
        raise ValueError(
            f"N must be a perfect square -- the crystal places one electron per "
            f"primitive cell on a square mesh of cells -- got {N}")
    A1, A2 = ge.triangular_cell(2.0 * math.pi)
    G = ge.Geometry.from_cell(A1, A2, n_side, n_side, 4.0, magnetic=True)
    prim_C = np.column_stack([A1, A2])
    C = np.column_stack([G.L1, G.L2])
    ll_ai, ll_ac = ge.circular_lattice(LUMAX_LL, A1, A2)
    ov_ai, ov_ac = ge.circular_lattice(LUMAX, G.L1, G.L2)
    return Lattice(
        N=N, rs=float(rs), n_side=n_side, A1=A1, A2=A2,
        L1=G.L1, L2=G.L2, G1=G.G1, G2=G.G2, mesh=G.mesh,
        sites=ge.wigner_crystal_sites(n_side, n_side, A1, A2),
        ll_ai=ll_ai, ll_ac=ll_ac, ov_ai=ov_ai, ov_ac=ov_ac,
        prim_C=prim_C, prim_Ci=np.linalg.inv(prim_C), C=C, Ci=np.linalg.inv(C),
        HAM=cb.CoulombEwald(N, G.L1, G.L2, G.G1, G.G2),
    )


@functools.lru_cache(maxsize=8)
def _lattice_cached(N, rs):
    """``build_lattice`` memoised.  The Lattice holds an Ewald sum, which is the
    one expensive part; callers must treat the result as READ-ONLY."""
    return build_lattice(N, rs)


def theta(c, v):
    """Pack ``(c, v)`` into the flat real parameter vector.

    Layout, which is the frozen engine's and is not negotiable:
    ``[c (NJ), Re v (nk*(n_bands-1)), Im v (nk*(n_bands-1))]`` -- length
    ``NJ + 2*(n_bands-1)*nk``, i.e. **77 at nmax=1 and 149 at nmax=2**.
    """
    v = np.asarray(v, complex)
    return np.concatenate([np.asarray(c, float).ravel(), v.real.ravel(),
                           v.imag.ravel()])


def theta_parts(theta, n_bands, nk):
    """Inverse of :func:`theta` -- ``(c, v)`` with ``v`` complex ``(nk, nb-1)``."""
    m = (n_bands - 1) * nk
    v = (np.asarray(theta, float)[NJ:NJ + m].reshape(nk, n_bands - 1)
         + 1j * np.asarray(theta, float)[NJ + m:].reshape(nk, n_bands - 1))
    return np.asarray(theta, float)[:NJ], v


# ===========================================================================
# budgets
# ===========================================================================
@dataclass(frozen=True)
class Budget:
    """A named, recorded protocol.  Nothing here is defaulted at call time."""

    name: str
    provenance: str
    protocol: dict
    sr: dict
    crystal_measure: dict
    liquid_measure: dict
    inits: tuple

    def width_for(self, init_id):
        """``L0`` of ``init_id``, from the named list.

        ``init_id`` is an INDEX into a list of starting points -- here the
        campaign's five fixed Jastrow widths -- and is NOT a random seed.  It is
        validated rather than clipped so a typo cannot silently select a
        different width.
        """
        init_id = int(init_id)
        for row in self.inits:
            if int(row["init_id"]) == init_id:
                return float(row["L0"])
        raise KeyError(f"init_id {init_id} is not in budget {self.name!r}; "
                       f"available: {[r['init_id'] for r in self.inits]}")

#: User-facing aliases for a budget name.  ``full`` is what people reach for
#: when they mean "the real statistics, not the smoke test", and the config it
#: names is ``reproduction`` -- the protocol transcribed from the historical
#: figure's own notebook cells (see the header of ``configs/reproduction.yaml``).
#: The table exists so that ``--budget full`` and ``--budget reproduction`` are
#: ONE request rather than two directories holding the same calculation.
BUDGET_ALIASES = {"full": "reproduction"}


def resolve_budget_name(name):
    """Map a user-facing budget alias onto the config it names.

    Returns the input unchanged when it is not an alias, so this is safe to
    apply unconditionally: an unknown name stays unknown and still fails in
    ``load_budget`` with the list of what exists, rather than being guessed at.
    """
    return BUDGET_ALIASES.get(str(name), str(name))


def load_budget(name):
    """Read ``configs/<name>.yaml``.  There is no built-in fallback table: an
    unknown budget is an error, not a guess.  User-facing aliases
    (``BUDGET_ALIASES``, e.g. ``full``) are resolved first."""
    name = resolve_budget_name(name)
    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        have = sorted(f[:-5] for f in os.listdir(CONFIG_DIR)
                      if f.endswith(".yaml")) if os.path.isdir(CONFIG_DIR) else []
        raise FileNotFoundError(
            f"budget {name!r} has no config at {path}.  Available: {have}.  "
            f"Add configs/{name}.yaml rather than inventing protocol numbers.")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Budget(
        name=str(raw.get("name", name)),
        provenance=str(raw.get("provenance", "")),
        protocol=dict(raw["protocol"]),
        sr=dict(raw["sr"]),
        crystal_measure=dict(raw["crystal_measure"]),
        liquid_measure=dict(raw["liquid_measure"]),
        inits=tuple(raw["inits"]),
    )


# ===========================================================================
# config
# ===========================================================================
_PHASES = ("crystal", "liquid")
_KAPPA_MODES = ("physical", "legacy-regression")

#: The two crystal optimisation protocols this engine can express.  Both build
#: the SAME wavefunction class (``LLRotationWavefunction``); they differ in WHICH
#: parameters the optimiser is allowed to move, which is worth naming and
#: recording rather than defaulting.
#:
#: ``ll_rotation``         orbitals and Jastrow optimised jointly (the default).
#: ``ll_rotation_pinned``  orbitals held at the Gaussian-overlap seed
#:                         ``crystal_v0(L0)``; only the 5 Jastrow parameters move.
#:
#: NEITHER of these is the historical fig06 crystal.  That state is a
#: determinant of site-centred magnetic Gaussians -- notebook cell 18's
#: ``crystal_wavefunction``, built straight from ``GaussianBasis.orbitals`` --
#: which is a different wavefunction FAMILY and is not implemented here.  See
#: ``examples/figure_construction/DIVERGENCE_fig06_crystal.md``.  Pinning the
#: seed inside the LL-rotated basis is not the same thing: measured, it moves
#: ``S(|g1|)`` from 1.94 to 2.22, where the historical ensemble reads 11.92.
_ANSATZE = ("ll_rotation", "ll_rotation_pinned")


@dataclass(frozen=True)
class RunConfig:
    """A fully resolved run request.  Immutable, so a run's parameters cannot be
    edited after the fact and reported as something else."""

    phase: str
    rs: float
    N: int
    nmax: int
    init_id: int
    budget: str
    kappa_mode: str = "physical"
    rng_seed: int = 0
    ansatz: str = "ll_rotation"

    @property
    def n_bands(self):
        return int(self.nmax) + 1

    def kappa(self):
        """The coupling this run will use.

        ``physical`` is ``rs/sqrt(2)`` at full precision and is what every normal
        run uses.  ``legacy-regression`` is ``round(rs/sqrt(2), 4)`` and exists
        ONLY to reproduce the frozen campaign, whose jobs were all built at the
        rounded value (Blocker B).  The difference is 1.6e-07 relative and moves
        a per-particle energy by ~6e-06 -- it is provenance, never a physics
        choice.
        """
        if self.kappa_mode == "legacy-regression":
            return float(round(float(self.rs) / math.sqrt(2.0), 4))
        return float(self.rs) / math.sqrt(2.0)

    def describe(self, budget=None):
        """The resolved-config print.  ``init_id``, its ``L0`` and ``rng_seed``
        are on SEPARATE lines because they are three different things and the
        campaign conflated two of them.

        The two phases do NOT have the same parameters, so they do not get the
        same print.  The liquid's orbitals are the filled n=0 Landau level held
        at ``v = 0``: its ``nmax`` is only the length of a padding block that
        `c_row` multiplies by zero, so printing ``n_max``/``n_bands`` as though
        they were physics would invite a reader to think the liquid has a
        Landau-level truncation to choose.  It does not.
        """
        if self.phase == "crystal":
            lines = [
                "  phase            crystal",
                f"  r_s              {self.rs:g}",
                f"  orbital ansatz   LL rotation"
                f"{'' if self.ansatz == 'll_rotation' else ' (pinned)'}",
                f"  n_max            {self.nmax}",
                f"  LL basis         n = {', '.join(str(n) for n in range(self.n_bands))}",
                f"  n_bands          {self.n_bands}   (= n_max + 1)",
                f"  orbital SR       "
                f"{'enabled' if self.ansatz == 'll_rotation' else 'disabled (orbitals pinned at the seed)'}",
                f"  Jastrow SR       enabled",
                f"  optimisation     "
                f"{'joint orbital + Jastrow SR' if self.ansatz == 'll_rotation' else 'Jastrow only'}",
                f"  len(theta)       {NJ + 2 * (self.n_bands - 1) * self.N}",
                f"  technical        ansatz={self.ansatz}",
            ]
        else:
            lines = [
                "  phase            liquid",
                f"  r_s              {self.rs:g}",
                "  orbital ansatz   filled LLL (fixed)",
                "  orbital SR       disabled",
                "  Jastrow SR       enabled",
                "  LL rotation      none",
                # Technical, and labelled as such: the liquid carries an nmax
                # only because the shared parameter vector reserves room for it.
                f"  technical        nmax={self.nmax} n_bands={self.n_bands} "
                f"len(theta)={NJ + 2 * (self.n_bands - 1) * self.N} "
                f"(padding block, held at v = 0 -- not a liquid parameter)",
            ]
        lines += [
            f"  N                {self.N}",
            f"  kappa            {self.kappa():.8f}   [{self.kappa_mode}]",
            f"  init_id          {self.init_id}",
            f"  rng_seed         {self.rng_seed}",
        ]
        if budget is not None:
            # `init_id` selects a Gaussian WIDTH, which only the crystal's
            # starting point uses (`crystal_v0`).  Printing it for the liquid
            # would present the crystal's seed -- and the crystal's ansatz --
            # as a liquid parameter, which is the confusion this whole
            # phase-aware description exists to remove.
            if self.phase == "crystal":
                lines += [f"  init L0      {budget.width_for(self.init_id):g}   "
                          f"(from budget {budget.name!r})"]
            else:
                lines += ["  start        R0 built from (N, rs) alone; `init_id` "
                          "selects nothing for the liquid"]
            lines += [
                f"  budget       {budget.name}   source: {budget.provenance}",
                f"  protocol     SR {budget.protocol['sr_steps']}x"
                f"{budget.protocol['sr_sweeps']}, walk "
                f"{budget.protocol['meas_sweeps']}/{budget.protocol['meas_equil']}",
            ]
        return "\n".join(lines)


def resolve(args=None, **overrides):
    """Turn parsed CLI arguments -- or plain keywords -- into a ``RunConfig``.

    Accepts an ``argparse.Namespace``, a mapping, or keywords, so the CLI's
    ``resolve(args)`` and a library caller's ``resolve(phase=...)`` are the same
    code path.  Validation lives here rather than in ``argparse`` so that a
    library caller gets the same errors a CLI user does.
    """
    src = {}
    if args is not None:
        if isinstance(args, dict):
            src.update(args)
        else:
            src.update(vars(args))
    src.update(overrides)
    src = {k: v for k, v in src.items() if v is not None}

    phase = str(src.get("phase", "crystal")).lower()
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES}, got {phase!r}")
    mode = str(src.get("kappa_mode", "physical"))
    if mode not in _KAPPA_MODES:
        raise ValueError(f"kappa_mode must be one of {_KAPPA_MODES}, got {mode!r}")
    nmax = int(src.get("nmax", 1))
    if nmax < 1:
        raise ValueError(f"nmax must be >= 1, got {nmax}")
    N = int(src.get("N", 36))
    ansatz = str(src.get("ansatz", "ll_rotation"))
    if ansatz not in _ANSATZE:
        raise ValueError(f"ansatz must be one of {_ANSATZE}, got {ansatz!r}")
    if ansatz != "ll_rotation" and phase != "crystal":
        raise ValueError(
            f"ansatz {ansatz!r} is a crystal choice: the liquid's orbitals are "
            f"fixed at v = 0 by definition, so there is nothing to pin.")
    return RunConfig(
        phase=phase, rs=float(src.get("rs", 75.0)), N=N, nmax=nmax,
        init_id=int(src.get("init_id", 0)), budget=str(src.get("budget", "quick")),
        kappa_mode=mode, rng_seed=int(src.get("rng_seed", 0)), ansatz=ansatz,
    )


# ===========================================================================
# results
# ===========================================================================
@dataclass
class RunState:
    """The end state of a run -- small, picklable, and enough to nest or analyse.

    ``snaps`` is the production walk's snapshots; it is what ``structure_factor``
    needs and it is the reason this object is ~a megabyte rather than a few
    kilobytes.  Drop it (``state.snaps = None``) before pickling a large state
    somewhere tight -- ``nest`` and ``ll_occupation`` do not need it.
    """

    R: np.ndarray
    c: np.ndarray
    v: np.ndarray
    N: int
    rs: float
    kappa: float
    kappa_mode: str
    nmax: int
    phase: str
    snaps: Any = field(default=None, repr=False)
    #: Which crystal ansatz protocol produced this state.  Recorded because the
    #: orbitals are stored either way and the stored ``v`` alone does not say
    #: whether it was optimised or pinned at the Gaussian seed.
    ansatz: str = "ll_rotation"

    @property
    def n_bands(self):
        return int(self.nmax) + 1

    def ll_occupation(self):
        """Band weights ``P_n = mean_k |C[k, n]|^2`` and ``nbar = sum n P_n``.

        Both from ``c_row`` -- the same call the campaign used -- so there is one
        definition of the occupation and not two.

        CAVEAT, and it is a real one: ``P_n`` and ``nbar`` are **not comparable
        across ``n_bands``**.  A larger basis redistributes weight among more
        bands, so ``nbar`` at nmax=2 is not a refinement of ``nbar`` at nmax=1.
        """
        C = lr.c_row(self.v)
        P = (np.abs(C) ** 2).mean(axis=0)
        nbar = float((np.arange(self.n_bands) * P).sum())
        return {"P": P, "nbar": nbar, "n_bands": self.n_bands,
                "nmax": int(self.nmax)}

    def structure_factor(self, q_max=4.0):
        """``S(q)`` on the production snapshots, via ``analysis.structure``.

        Imported lazily: ``analysis`` is the post-processing layer and pulling it
        in at package import would drag plotting into every ``import wigner_vmc``.
        """
        from .analysis import structure as st
        if self.snaps is None:
            raise RuntimeError(
                "this state carries no production snapshots, so S(q) cannot be "
                "computed from it.  Re-run with snapshots kept, or use a state "
                "straight from run().")
        # The AREA, not the primitive cell's area.  At nu = 1 one electron carries
        # one flux quantum, so the primitive cell is 2*pi*l_B^2 and this supercell
        # is N of them -- N * 2*pi, i.e. 72*pi at N = 36.  Passing the bare 2*pi
        # built a torus N times too small: its |G1| came out N times too large, so
        # `allowed_momenta(4.0)` returned SIX momenta (all at |q| = 2.69) instead of
        # the 294 every caller expects, and S(q) was silently a six-point array.
        # The closed form N * 2*pi is the same quantity `make_figures.py` builds as
        # `Torus(72*pi, 36, 6)`, so the two agree by construction and not by luck.
        torus = st.Torus(2.0 * math.pi * float(self.N), n_electrons=int(self.N),
                         n_cells_per_side=int(round(math.sqrt(self.N))))
        q, qn = torus.allowed_momenta(q_max)
        return {"q": q, "qn": qn,
                "S": st.structure_factor(self.snaps, q, int(self.N)),
                "torus": torus}


@dataclass
class Optimization:
    """The SR trace.  **Every energy in here is a TOTAL, not per electron.**

    Kept in its own namespace so the optimisation minimum cannot masquerade as
    the production energy -- the two differ by a factor of ``N`` and are answers
    to different questions.
    """

    kind: str
    steps: int
    seconds: float
    E_total: np.ndarray
    acc: np.ndarray
    N: int

    @property
    def E_total_min(self):
        return float(self.E_total.min())

    @property
    def E_total_final(self):
        return float(self.E_total[-1])

    @property
    def E_per_particle_min(self):
        return self.E_total_min / self.N

    def summary(self):
        return (f"    SR ({self.kind}) {self.steps} steps in {self.seconds:.0f}s   "
                f"E_total {self.E_total[0]:+.4f} -> {self.E_total[-1]:+.4f}   "
                f"min {self.E_total_min:+.4f}  "
                f"(per particle min {self.E_per_particle_min:+.6f}, NOT the answer)")


@dataclass
class RunResult:
    """What a run returns.  See the module docstring for the energy semantics."""

    config: RunConfig
    energy_per_particle: float
    error: float
    acceptance: float
    state: RunState
    optimization: Optimization
    record: dict
    seconds: float
    budget: Budget

    @property
    def N(self):
        return int(self.state.N)

    @property
    def energy_total(self):
        """``N * energy_per_particle`` -- a property, so the factor of ``N``
        exists exactly once in this API and cannot drift away from it.

        Note ``record["E_total"]`` is the mean total the walk measured directly;
        it can differ from this in the last bit or two, because one is a division
        followed by a multiplication and the other is neither.  When the two
        disagree it is this property that is authoritative, because it is the one
        the contract pins."""
        return self.N * self.energy_per_particle

    @property
    def error_total(self):
        return self.N * self.error

    def summary(self):
        st = self.state
        return "\n".join([
            self.config.describe(self.budget),
            f"  start        R0 built from (N, rs) alone; no checkpoint read",
            self.optimization.summary(),
            f"    production walk  n={self.record['n']}  "
            f"acc={self.acceptance:.4f}  sigma={self.record['sigma']:.4f}  "
            f"tau_E={self.record['E_tau']:.4f}",
            f"  E/N          {self.energy_per_particle:+.9f} +- {self.error:.9f}  "
            f"(per electron)",
            f"  E_total      {self.energy_total:+.9f} +- {self.error_total:.9f}  "
            f"(= N * E/N, N={self.N})",
            f"  elapsed      {self.seconds:.0f}s",
        ])

    def __str__(self):
        return self.summary()


# ===========================================================================
# the workflow
# ===========================================================================
class VMC:
    """A system at one ``(N, rs, phase, nmax)``.  Holds no run state.

    The geometry and the coupling are fixed at construction and verified there,
    so ``run`` cannot be called with a mismatched pair.
    """

    def __init__(self, N=36, rs=75.0, phase="crystal", nmax=1,
                 kappa_mode="physical", ansatz="ll_rotation"):
        if phase not in _PHASES:
            raise ValueError(f"phase must be one of {_PHASES}, got {phase!r}")
        if kappa_mode not in _KAPPA_MODES:
            raise ValueError(f"kappa_mode must be one of {_KAPPA_MODES}, "
                             f"got {kappa_mode!r}")
        if ansatz not in _ANSATZE:
            raise ValueError(f"ansatz must be one of {_ANSATZE}, got {ansatz!r}")
        if ansatz != "ll_rotation" and phase != "crystal":
            raise ValueError(
                f"ansatz {ansatz!r} is a crystal choice: the liquid's orbitals "
                f"are fixed at v = 0 by definition.")
        if int(nmax) < 1:
            raise ValueError(f"nmax must be >= 1, got {nmax}")
        self.N = int(N)
        self.rs = float(rs)
        self.phase = str(phase)
        self.nmax = int(nmax)
        self.kappa_mode = str(kappa_mode)
        self.ansatz = str(ansatz)
        self.lat = _lattice_cached(self.N, self.rs)

    # -- derived ----------------------------------------------------------
    @property
    def n_bands(self):
        return self.nmax + 1

    @property
    def kappa(self):
        return resolve(phase=self.phase, rs=self.rs, N=self.N, nmax=self.nmax,
                       kappa_mode=self.kappa_mode).kappa()

    def basis(self, n_band=None):
        """The Landau-level Bloch basis at ``n_band`` (PRIMITIVE-cell quantities)."""
        nb = self.n_bands if n_band is None else int(n_band)
        return llb.LandauLevelBasis(self.lat.mesh, nb - 1, self.lat.ll_ai,
                                    self.lat.ll_ac, self.lat.prim_C,
                                    self.lat.prim_Ci)

    def maker(self, n_band=None, v=None):
        """``make_wf(theta)``, or ``make_wf(c)`` when ``v`` is supplied.

        The two-argument form is what the liquid needs: its orbitals are fixed at
        ``v = 0`` by definition, so only the five Jastrow parameters are free and
        the optimiser must see a five-parameter function.
        """
        nb = self.n_bands if n_band is None else int(n_band)
        basis = self.basis(nb)
        kappa = self.kappa
        kappa_mode = self.kappa_mode
        C, HAM, N = self.lat.C, self.lat.HAM, self.N

        def _wf(th):
            th = np.asarray(th, float)
            c, vv = theta_parts(th, nb, self.lat.nk)
            orb = lr.LLRotatedOrbitals(basis, nb, vv)
            jast = jw.SinSplineJastrow(c, self.lat.G1, self.lat.G2,
                                       jw.cusp_gamma(kappa, self.lat.L1))
            wf = lr.LLRotationWavefunction(orb, jast, N, C, kappa=kappa, ham=HAM)
            wf.kappa_mode = kappa_mode
            return wf

        if v is None:
            return _wf
        v_fixed = np.asarray(v, complex)

        def _wf_c(c):
            return _wf(theta(c, v_fixed))
        return _wf_c

    # -- deterministic starting configurations -----------------------------
    def crystal_R0(self):
        """``sites`` plus a quarter-width jitter, seeded off ``int(kappa)``.

        The campaign's own recipe (``crystal_sites_seed``); deterministic, so two
        runs at the same coupling start from the same places.
        """
        rng = np.random.default_rng(100 + int(self.kappa))
        return np.asarray(self.lat.sites, float) + 0.25 * rng.standard_normal(
            (self.N, 2))

    def liquid_R0(self):
        """A uniform random start on the torus, the campaign's seed 3."""
        return (np.random.default_rng(3).random((self.N, 2)) - 0.5) @ self.lat.C.T

    def crystal_v0(self, L0):
        """The Gaussian-overlap orbital seed at width ``L0``.

        ``L0`` is the Gaussian width, which is what ``init_id`` selects.  It is a
        WIDTH, not a random seed, and the campaign's five "seeds" are five of
        these -- which is why they are not independent draws.
        """
        ov = sr.gaussian_overlap_seed(self.basis(), self.n_bands, self.lat.L1,
                                      self.lat.L2, self.lat.ov_ai, self.lat.ov_ac,
                                      rs=self.rs, numx=OVERLAP_NUMX, L0=L0)
        return np.asarray(sr.v_from_overlap(ov), complex)

    # -- the run -----------------------------------------------------------
    def run(self, init_id=0, budget="quick", verbose=True):
        cfg = resolve(phase=self.phase, rs=self.rs, N=self.N, nmax=self.nmax,
                      init_id=init_id, budget=budget, kappa_mode=self.kappa_mode,
                      ansatz=self.ansatz)
        return self._run(cfg, verbose=verbose)

    def _run(self, cfg, verbose=True):
        t0 = time.time()
        bud = load_budget(cfg.budget)
        proto = bud.protocol
        mk = self.maker()
        rng_seed = cfg.rng_seed

        if cfg.phase == "crystal":
            v0 = self.crystal_v0(bud.width_for(cfg.init_id))
            c0 = sr.jastrow_vector(self.kappa, NJ)
            kwargs = dict(steps=proto["sr_steps"], nsweep=proto["sr_sweeps"],
                          sigma=bud.sr["sigma"], seed=rng_seed,
                          snapshot_every=proto["sr_snap"], equil=proto["sr_equil"],
                          target_acc=bud.sr["target_acc"])
            if cfg.ansatz == "ll_rotation_pinned":
                # Orbitals held at the Gaussian-overlap seed: the optimiser sees
                # a 5-parameter function.  `maker(v=v0)` is the same two-argument
                # form the liquid already uses.  This is the historical
                # OPTIMISATION SCOPE, not the historical ansatz -- see _ANSATZE.
                c_opt, hist = sr.sr_optimize_jastrow(self.maker(v=v0), c0,
                                                     self.crystal_R0(), **kwargs)
                th, kind = theta(c_opt, v0), "jastrow"
            else:
                th, hist = sr.sr_optimize_joint(mk, theta(c0, v0),
                                                self.crystal_R0(), **kwargs)
                kind = "joint"
            R0_walk, mseed, msig = (self.crystal_R0(),
                                    bud.crystal_measure["seed"],
                                    bud.crystal_measure["sigma"])
        else:
            v0 = np.zeros((self.lat.nk, self.n_bands - 1), complex)
            c0 = sr.jastrow_vector(self.kappa, NJ)
            kwargs = dict(steps=proto["sr_steps"], nsweep=proto["sr_sweeps"],
                          sigma=bud.sr["sigma"], seed=rng_seed,
                          snapshot_every=proto["sr_snap"], equil=proto["sr_equil"],
                          target_acc=bud.sr["target_acc"])
            c_opt, hist = sr.sr_optimize_jastrow(self.maker(v=v0), c0,
                                                 self.liquid_R0(), **kwargs)
            th, kind = theta(c_opt, v0), "jastrow"
            R0_walk, mseed, msig = (self.liquid_R0(),
                                    bud.liquid_measure["seed"],
                                    bud.liquid_measure["sigma"])

        if not np.all(np.isfinite(th)):
            raise RuntimeError(
                "the optimiser returned non-finite parameters; refusing to "
                "measure.  This is a convergence failure, not a result.")

        wf = mk(th)
        rec = measure_decomposed(wf, R0_walk, proto["meas_sweeps"],
                                 proto["meas_equil"], msig, mseed,
                                 label=f"{cfg.phase}-nmax{cfg.nmax}")
        c, v = theta_parts(th, self.n_bands, self.lat.nk)
        # Snapshots belong to the STATE (structure_factor needs them) and not to
        # the record, which is meant to stay small and serialisable.  R is the
        # walk's last configuration -- the state's "where the walker ended up".
        snaps = rec.pop("snaps")
        state = RunState(R=np.asarray(snaps[-1] if snaps else R0_walk, float),
                         c=c, v=v, N=self.N, rs=self.rs, kappa=self.kappa,
                         kappa_mode=self.kappa_mode, nmax=self.nmax,
                         phase=cfg.phase, snaps=snaps, ansatz=cfg.ansatz)
        opt = Optimization(
            kind=kind, steps=int(len(hist)), seconds=float(time.time() - t0),
            E_total=np.array([h["E"] for h in hist], float),
            acc=np.array([h["acc"] for h in hist], float), N=self.N)
        result = RunResult(config=cfg, energy_per_particle=float(rec["E"]),
                           error=float(rec["E_err"]), acceptance=float(rec["acc"]),
                           state=state, optimization=opt, record=rec,
                           seconds=float(time.time() - t0), budget=bud)
        if verbose:
            print(result.summary(), flush=True)
        return result

    # -- nesting -----------------------------------------------------------
    def nest(self, parent, new_nmax, budget="quick", verbose=True):
        """Grow a parent's ``nmax`` by exact embedding, then optimise.

        The embedding is ``v_child = pad_v(v_parent, n_bands)``, which adds the
        new level as an exact zero.  That is not an approximation: because
        ``C = (cos|v|, i sin|v|/|v| v)`` and ``|(v, 0)| = |v|``, the child's
        coefficients are the parent's with a zero column appended, so the child
        ansatz CONTAINS the parent.  That containment is what makes an energy
        that rises with ``nmax`` attributable to the optimiser rather than to a
        changed starting point.

        Checked with :func:`nesting.nesting_residual` on the vector actually
        about to be optimised -- and it aborts on failure rather than falling
        back to a fresh initialisation, so the containment claim stays true of
        every result this API produces.
        """
        new_nmax = int(new_nmax)
        # A precondition on the receiver, checked before the parent is inspected:
        # nesting grows the ORBITAL space, and a pinned-orbital ansatz has no
        # variational orbitals to grow.
        if self.ansatz != "ll_rotation":
            raise ValueError(
                f"nesting grows the ORBITAL space, so a {self.ansatz!r} state "
                f"cannot nest: its orbitals are pinned at the Gaussian seed and "
                f"are not variational.  Nest an 'll_rotation' state instead.")
        if new_nmax <= parent.state.nmax:
            raise ValueError(
                f"nesting grows nmax: parent is nmax={parent.state.nmax}, "
                f"asked for nmax={new_nmax}")
        if parent.state.phase != "crystal":
            raise ValueError("only a crystal nests; the liquid has no orbitals "
                             "to grow")
        child = VMC(N=self.N, rs=self.rs, phase="crystal", nmax=new_nmax,
                    kappa_mode=self.kappa_mode, ansatz=self.ansatz)
        v_parent = np.asarray(parent.state.v, complex)
        v_child = nest.pad_v(v_parent, child.n_bands)

        head, tail = nest.nesting_residual(v_parent, v_child)
        if head != 0.0 or tail != 0.0:
            raise NestingIdentityError(
                f"the nested start is not the parent's wavefunction: "
                f"head deviation {head:.3e}, tail {tail:.3e} (both must be "
                f"exactly 0.0).  Refusing to optimise -- a re-initialised child "
                f"would make any nmax trend unattributable.")

        cfg = resolve(phase="crystal", rs=self.rs, N=self.N, nmax=new_nmax,
                      init_id=parent.config.init_id, budget=budget,
                      kappa_mode=self.kappa_mode)
        result = child._run_from(cfg, theta(parent.state.c, v_child),
                                 R0=child.crystal_R0(), verbose=verbose)
        result.nesting = {"from_nmax": int(parent.state.nmax),
                          "to_nmax": new_nmax, "head": head, "tail": tail}
        return result

    def _run_from(self, cfg, theta0, R0, verbose=True):
        """``_run`` with an explicitly supplied start -- the nesting path.

        Separate from ``_run`` because the two have different starting-point
        contracts: ``_run`` builds its own, this one is handed the parent's.
        """
        t0 = time.time()
        bud = load_budget(cfg.budget)
        proto = bud.protocol
        mk = self.maker()
        th, hist = sr.sr_optimize_joint(
            mk, np.asarray(theta0, float), R0,
            steps=proto["sr_steps"], nsweep=proto["sr_sweeps"],
            sigma=bud.sr["sigma"], seed=cfg.rng_seed,
            snapshot_every=proto["sr_snap"], equil=proto["sr_equil"],
            target_acc=bud.sr["target_acc"])
        if not np.all(np.isfinite(th)):
            raise RuntimeError("the optimiser returned non-finite parameters; "
                               "refusing to measure.")
        wf = mk(th)
        rec = measure_decomposed(wf, self.crystal_R0(), proto["meas_sweeps"],
                                 proto["meas_equil"],
                                 bud.crystal_measure["sigma"],
                                 bud.crystal_measure["seed"],
                                 label=f"crystal-nmax{cfg.nmax}")
        c, v = theta_parts(th, self.n_bands, self.lat.nk)
        snaps = rec.pop("snaps")
        state = RunState(R=np.asarray(snaps[-1] if snaps else R0, float), c=c, v=v,
                         N=self.N, rs=self.rs, kappa=self.kappa,
                         kappa_mode=self.kappa_mode, nmax=self.nmax,
                         phase="crystal", snaps=snaps)
        opt = Optimization(kind="joint", steps=int(len(hist)),
                           seconds=float(time.time() - t0),
                           E_total=np.array([h["E"] for h in hist], float),
                           acc=np.array([h["acc"] for h in hist], float),
                           N=self.N)
        result = RunResult(config=cfg, energy_per_particle=float(rec["E"]),
                           error=float(rec["E_err"]), acceptance=float(rec["acc"]),
                           state=state, optimization=opt, record=rec,
                           seconds=float(time.time() - t0), budget=bud)
        if verbose:
            print(result.summary(), flush=True)
        return result


def run_vmc(config=None, verbose=True, **overrides):
    """One-shot entry point: ``run_vmc(config)`` or ``run_vmc(phase=..., ...)``.

    The CLI's whole body.  It resolves, constructs the system, and calls the same
    private workflow ``VMC.run`` calls -- there is no second implementation for
    the command line to drift away from.
    """
    cfg = config if isinstance(config, RunConfig) else resolve(config, **overrides)
    vmc = VMC(N=cfg.N, rs=cfg.rs, phase=cfg.phase, nmax=cfg.nmax,
              kappa_mode=cfg.kappa_mode, ansatz=cfg.ansatz)
    return vmc._run(cfg, verbose=verbose)
