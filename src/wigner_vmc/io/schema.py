"""The canonical record a legacy measurement is converted into.

There is exactly one energy field that a phase comparison may read, and it is
`production_energy_per_particle`.  That name is long on purpose.

The legacy records carry four different energies that all look like "the energy":

    E            total over the cell, not per particle
    E_perpart    the independent production measurement, per particle  <-- the one
    E_min        minimum along the SR trajectory, per particle
    E_final      last value of the SR trajectory, per particle

A comparison that mixes them is wrong by a factor of N (36) or by a systematic the
size of the optimiser's own drift -- and both of those have already happened in this
campaign.  Rather than trust a reader to pick the right one, the conversion in
`legacy.py` reads `E_perpart` and nothing else, and the other three are preserved
under `optimization` where a phase comparison cannot reach them.

`ll_convergence.py` and `energy_competition.py` accept only `StateRecord`, which has
no `E_min`; the optimiser trajectory is not reachable from the analysis layer at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# The cell used by every campaign in this repository: 6x6 torus, nu = 1.
N_ELECTRONS = 36

# Workflow tags.
GAUSSIAN = "gaussian"
LLROT = "llrot"

# Phase tags.
LIQUID = "liquid"
CRYSTAL = "crystal"


@dataclass
class Provenance:
    """Where a record came from, so a number can always be traced back to a file."""
    source_path: str                 # project-root-relative
    source_sha256: str
    legacy_energy_field: str         # the field actually read for the energy
    legacy_error_field: str
    legacy_label: str                # the record's own key/tag
    notes: str = ""

    def as_dict(self):
        return asdict(self)


@dataclass
class OptimizationMeta:
    """The SR trajectory.  PRESERVED, AND FORBIDDEN IN PHASE COMPARISONS.

    None of these describe the state; they describe the path the optimiser took to
    it.  `min_energy_per_particle` in particular is systematically below the
    production measurement, because it is the best point ever seen rather than the
    converged one, so quoting it would make every state look better than it is.
    """
    start_energy_per_particle: Optional[float] = None
    final_energy_per_particle: Optional[float] = None
    min_energy_per_particle: Optional[float] = None
    steps: Optional[int] = None
    sweeps: Optional[int] = None
    force_first: Optional[float] = None
    force_final: Optional[float] = None
    cond_final: Optional[float] = None
    c_norm_dev: Optional[float] = None

    def as_dict(self):
        return asdict(self)


@dataclass
class StateRecord:
    """One variational state's production measurement, in canonical form.

    Energy fields are per particle and in units of hbar*omega_c.  `kappa` is the
    Coulomb coupling; r_s = sqrt(2) * kappa.
    """

    # ---- identity --------------------------------------------------------
    workflow: str                    # gaussian | llrot
    phase: str                       # liquid | crystal
    rs: float
    N: int = N_ELECTRONS
    kappa: Optional[float] = None

    # ---- the variational family -----------------------------------------
    nmax: Optional[int] = None       # n_band - 1; None for the liquid
    L0: Optional[float] = None       # Gaussian site width; None for the liquid
    init_id: str = ""                # the seed/start identifier

    # ---- provenance of the configuration ---------------------------------
    rng_seed: Optional[int] = None
    parent_id: Optional[str] = None  # the state this one was warmed from
    nested: bool = False             # part of the n_max ladder

    # ---- THE production measurement --------------------------------------
    production_energy_per_particle: Optional[float] = None
    production_mc_error: Optional[float] = None

    # ---- decomposition and order parameters ------------------------------
    total_energy: Optional[float] = None
    kinetic_total: Optional[float] = None
    potential_total: Optional[float] = None
    nbar: Optional[float] = None
    band_occupations: Optional[list] = None
    R_B: Optional[float] = None      # Bragg ratio, the crystal order parameter
    S_QWC: Optional[float] = None
    S_bg: Optional[float] = None
    acceptance: Optional[float] = None

    # ---- never compared --------------------------------------------------
    optimization: OptimizationMeta = field(default_factory=OptimizationMeta)
    provenance: Optional[Provenance] = None

    # ---- the untouched legacy record -------------------------------------
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def as_dict(self, include_raw=False):
        d = asdict(self)
        if not include_raw:
            d.pop("raw", None)
        return d

    @property
    def has_production_energy(self) -> bool:
        return self.production_energy_per_particle is not None

    def __repr__(self):
        e = ("None" if self.production_energy_per_particle is None
             else f"{self.production_energy_per_particle:+.6f}")
        return (f"<{self.workflow}/{self.phase} r_s={self.rs:g} "
                f"nb={self.nmax} E/N={e} id={self.init_id}>")


@dataclass
class ScanPoint:
    """One (r_s, tier) point of a phase comparison.

    This is what the analysis layer produces and what a figure consumes.  It is
    deliberately flat: a figure may not reach past it into a StateRecord, because
    doing so is how a figure ends up recomputing a statistic.
    """
    rs: float
    tier: str
    n_max: Optional[int]
    n_states: int
    energy_crystal: float
    energy_liquid: float
    delta_e: float                       # crystal - liquid, per particle

    sigma_mc: float                      # sampling only
    sigma_init: float                    # across independent optimisations only
    sigma_total: float                   # hypot of the two, plus the liquid's MC
    z: Optional[float]

    seed_ids: list = field(default_factory=list)
    caveats: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class Bracket:
    """The r_s interval one truncation supports, and whether it is an interval at all."""
    tier: str
    kind: str                # bracket | above_scan | below_scan | empty
    lo: Optional[float]
    hi: Optional[float]
    resolved: bool
    n_points: int
    note: str = ""

    def as_dict(self):
        return asdict(self)
