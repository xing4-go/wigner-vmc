"""Parameter conventions and the model container.  Stage 2A: definitions only.

**This module evaluates no physics.**  It holds the units, the one conversion between
the coupling and the density, and a validated parameter container.  The local-energy
evaluation is NOT migrated in Stage 2A and stays in the frozen engines.

The locked relation
-------------------
The engine's Hamiltonian, in units ``l_B = hbar = m = 1`` with energy in ``hbar
omega_c``, is

    H/(hbar omega_c) = 1/2 sum_i [ -i l_B grad_i + (1/2 l_B) r_i x z_hat ]^2
                       + kappa sum_{i<j} l_B / |r_i - r_j|

and the coupling ``kappa`` is the dimensionless Coulomb strength:

    kappa = r_s * sqrt(nu / 2)          so        nu = 1  =>  kappa = r_s / sqrt(2)
                                                  and       r_s   = sqrt(2) * kappa

``kappa`` is what ``e^2`` becomes in these units.  The relation ``r_s = sqrt(2) kappa``
is LOCKED: it is what every stored result, every checkpoint tag and every published
number in this project was produced under.  It is implemented here, once, in
`ModelParams`, and nowhere else in the clean package.  Two symbols this close
numerically (``kappa = 32`` is ``r_s = 45.25``) do not produce an absurd-looking plot
when they are swapped, so a second conversion site is a real hazard rather than a
tidiness question.

What is NOT rescaled
--------------------
Nothing.  In particular ``A_WC`` (a distance) is not ``1/sqrt_n`` (a density to the
1/2); the two are 2.6935 and 2.5066 for the production cell and both appear in the
figure code.  See `PHYSICS_MAP.md` section 1.4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .landau_levels import nmax_from_n_bands

__all__ = [
    "UNITS", "HBAR_OMEGA_C", "MAGNETIC_LENGTH", "PRODUCTION_KAPPAS",
    "PRODUCTION_N_ELECTRONS", "PRODUCTION_N_CELLS_PER_SIDE", "production_geometry",
    "rs_from_kappa", "kappa_from_rs", "ModelParams",
]

#: The unit system, as data, so that a reader can find it without reading prose.
UNITS = {
    "length": "l_B (magnetic length) = 1",
    "action": "hbar = 1",
    "mass": "m = 1",
    "charge": "absorbed into kappa: kappa = e^2 / (4 pi eps0 l_B hbar omega_c)",
    "energy": "hbar omega_c",
    "gauge": "A(r) = (1/2)(-y, x), curl A = +B z_hat, B = 1",
}

#: Energy unit and magnetic length, as numbers.  They are 1 by construction; they are
#: named so that a conversion written later has something to multiply by.
HBAR_OMEGA_C = 1.0
MAGNETIC_LENGTH = 1.0

#: The production system: 6x6 cells, one electron per cell, nu = 1.
PRODUCTION_N_CELLS_PER_SIDE = 6
PRODUCTION_N_ELECTRONS = PRODUCTION_N_CELLS_PER_SIDE ** 2

#: The production coupling grid.  The SAME tuple is the frozen
#: `analysis.energy_competition.KAPPAS_PROTO`; the two are asserted equal by
#: `tests/test_units_and_conventions.py` rather than one importing the other, because
#: the analysis layer's copy is part of the frozen figure contract and must not move.
PRODUCTION_KAPPAS: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0,
                                        64.0, 80.0)


def rs_from_kappa(kappa: float, nu: float = 1.0) -> float:
    """``r_s = kappa / sqrt(nu/2)``.  At ``nu = 1`` this is ``sqrt(2) kappa``."""
    return float(kappa) / math.sqrt(nu / 2.0)


def kappa_from_rs(r_s: float, nu: float = 1.0) -> float:
    """``kappa = r_s sqrt(nu/2)``.  At ``nu = 1`` this is ``r_s/sqrt(2)``."""
    return float(r_s) * math.sqrt(nu / 2.0)


def production_geometry(n_cells_per_side: int = PRODUCTION_N_CELLS_PER_SIDE):
    """The production cell as a `geometry.Geometry`, with the production cutoff.

    ``rl_cut = 4.0`` is the value both generators pass.  The primitive cell area is
    ``2*pi`` (one flux quantum, one electron at nu = 1), so the supercell is
    ``n^2 * 2*pi``; `triangular_cell` takes the PRIMITIVE area.  Imported lazily so
    that this parameters module stays a leaf.
    """
    from .geometry import Geometry, triangular_cell
    n = int(n_cells_per_side)
    A1, A2 = triangular_cell(2.0 * math.pi)
    return Geometry.from_cell(A1, A2, n, n, 4.0, magnetic=True)


@dataclass(frozen=True)
class ModelParams:
    """The model parameters, with the coupling stored once and the rest derived.

    ``kappa`` is the stored field and ``r_s`` is a property, so there is exactly one
    conversion in the object and no way to set the two inconsistently.

    ``nmax`` is the highest Landau index kept; ``n_bands = nmax + 1`` is derived.
    A legacy band count goes through `nmax_from_n_bands` on the way in -- there is no
    constructor that takes both.
    """

    kappa: float
    nu: float = 1.0
    n_electrons: int = PRODUCTION_N_ELECTRONS
    nmax: int = 0

    def __post_init__(self):
        if self.kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {self.kappa}")
        if self.nu <= 0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.nmax < 0:
            raise ValueError(f"nmax must be >= 0, got {self.nmax}")
        if self.n_electrons < 1:
            raise ValueError(f"n_electrons must be >= 1, got {self.n_electrons}")

    # -- conventions ---------------------------------------------------------
    @property
    def r_s(self) -> float:
        """Wigner-Seitz radius in units of ``l_B``.  ``r_s = sqrt(2) kappa`` at nu = 1."""
        return rs_from_kappa(self.kappa, self.nu)

    @property
    def n_bands(self) -> int:
        """``nmax + 1``.  Derived."""
        return self.nmax + 1

    @property
    def n_flux_quanta(self) -> float:
        """``n_electrons / nu``.  Equals the electron count at ``nu = 1``."""
        return self.n_electrons / self.nu

    @property
    def filling(self) -> float:
        """``nu`` again, named for the direction it is usually read in."""
        return self.nu

    @property
    def area(self) -> float:
        """Torus area in units of ``l_B^2``: ``2 pi n_flux``."""
        return 2.0 * math.pi * self.n_flux_quanta

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_rs(cls, r_s: float, nu: float = 1.0, **kw) -> "ModelParams":
        """Build from the density.  The only place ``r_s`` becomes ``kappa``."""
        return cls(kappa=kappa_from_rs(r_s, nu), nu=nu, **kw)

    @classmethod
    def from_n_bands(cls, n_bands: int, kappa: float, nu: float = 1.0,
                     **kw) -> "ModelParams":
        """Legacy adapter: accepts a BAND COUNT and converts it to ``nmax``.

        The parameter is called ``n_bands``, not ``nmax``, because it is one.
        """
        return cls(kappa=kappa, nu=nu, nmax=nmax_from_n_bands(n_bands), **kw)

    def to_legacy_n_bands(self) -> int:
        """The band count the frozen engine's ``LandauLevelBasis`` expects."""
        return self.n_bands
