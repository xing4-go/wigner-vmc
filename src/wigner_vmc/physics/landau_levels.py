"""Landau-level magnetic Bloch orbitals on the torus.

Stage 2A.  The one implementation of the Landau-level basis for both workflows: the
Gaussian notebook uses it to calibrate, and the LL-rotation ansatz is built on top of
it.

The `nmax` rule
---------------
This module takes **``nmax`` only** -- the highest Landau index kept -- and derives
``n_bands = nmax + 1``.  That is a deliberate departure from the frozen engine, whose
`LandauLevelBasis` takes a parameter it *calls* ``n_max`` but which is in fact the
BAND COUNT: ``LandauLevelBasis(mesh, n_max=k)`` returns ``k + 1`` columns, being bands
``0 .. k-1`` plus one padding band at index ``k`` that supplies ``a^dag |k-1>``.  Its
own docstring says so ("Band n_max is the padding band"), and
``LLRotatedOrbitals.__init__`` enforces ``ll_basis.n_max == n_band``, which is only
consistent under that reading.

So the mapping, used by every regression test, is

    clean LandauLevelBasis(mesh, nmax)  ==  legacy(mesh, nmax + 1)[:, :nmax + 1]
    clean .padded_orbitals()            ==  legacy(mesh, nmax + 1).orbitals()
    clean .ladder()                     ==  legacy(mesh, nmax + 1).pi_orbitals_ladder()

and `nmax_from_n_bands` is the *only* place in the physics core where a band count is
understood.  It exists for the legacy loaders; nothing else in this package accepts
one.

Conventions
-----------
Gauge ``A = (1/2)(-y, x)``, ``l_B = 1``.  The disk-basis state is

    phi_n(z) = i^n / sqrt(2^n n!) * z_bar^n * exp(-|z|^2 / 4)

which is ``(a^dag)^n |0> / sqrt(n!)``.  ``pi^2 phi_n = (2n + 1) phi_n``; `ladder`
returns exactly ``(2n+1)`` on the diagonal, and the finite-difference test in
`tests/test_landau_basis_against_legacy.py` is what pins the sign of B (the opposite
gauge gives the same ``pi^2`` and the opposite ``pi psi``, so a sign error does not
show up in an energy).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .magnetic_cell import (flux_quanta, fold_to_primitive_cell,
                            magnetic_translation_phase)

__all__ = ["disk_state", "LandauLevelBasis", "nmax_from_n_bands",
           "FLUX_QUANTA_PER_PRIMITIVE_CELL"]

#: The Landau-level Bloch sum runs over PRIMITIVE-cell translations, and the primitive
#: cell of the 60-degree lattice holds exactly one flux quantum.  This is why the
#: legacy LL basis carries no explicit flux factor while the Gaussian basis (whose sum
#: runs over the 36-flux-quantum supercell) carries ``n_phi``.  Written here so the
#: rule is visible in one place instead of being absent in one and implicit in another.
FLUX_QUANTA_PER_PRIMITIVE_CELL = 1.0


def disk_state(z_bar, r_square, n: int):
    """The n-th Landau level in the symmetric gauge: ``i^n z_bar^n e^{-r^2/4}/sqrt(2^n n!)``.

    ``z_bar = x - i y`` and ``r_square = x^2 + y^2`` are passed separately because the
    Bloch sum evaluates both on the same arrays.
    """
    pref = (1j ** n) * math.exp(-0.5 * (n * math.log(2.0) + math.lgamma(n + 1)))
    return (z_bar ** n) * np.exp(-r_square / 4.0) * pref


def nmax_from_n_bands(n_bands: int) -> int:
    """``n_bands - 1``.  The legacy adapter -- the only place a band count is accepted.

    Callers outside a legacy loader must pass ``nmax`` directly.  Raises on a
    non-positive band count rather than returning ``-1``, because ``n_bands = 0``
    would otherwise propagate into an empty orbital array and produce a determinant
    of a 0x0 matrix.
    """
    n = int(n_bands)
    if n < 1:
        raise ValueError(f"n_bands must be >= 1, got {n}")
    return n - 1


class LandauLevelBasis:
    """Magnetic Bloch orbitals ``psi_{k,n}(r)`` on the torus.

    Parameters
    ----------
    mesh : (nk, 2)
        the allowed momenta of the magnetic Brillouin zone.
    nmax : int
        the highest Landau index kept.  ``n_bands = nmax + 1`` is derived.
    a_ints : (na, 2) int
        integer coefficients of the Bloch-sum lattice vectors.
    a_cart : (na, 2)
        the same vectors in Cartesian coordinates.
    lat_to_cart, cart_to_lat : (2, 2)
        the frame whose columns are the primitive vectors, and its inverse.

    Shapes
    ------
    ``orbitals(r)``         -> (nk, nmax + 1)   the physical levels 0 .. nmax
    ``padded_orbitals(r)``  -> (nk, nmax + 2)   the above plus the padding level nmax+1
    ``ladder(r)``           -> three (nk, nmax + 1) arrays: ``pi_x psi, pi_y psi, pi^2 psi``

    Only `ladder` and any future ansatz code need the padding level, and it is not
    reachable through `orbitals`.  A caller that wants ``a^dag |nmax>`` must ask for it
    by name.
    """

    def __init__(self, mesh, nmax: int, a_ints, a_cart, lat_to_cart, cart_to_lat):
        self.mesh = np.asarray(mesh, dtype=float)
        if self.mesh.ndim != 2 or self.mesh.shape[1] != 2:
            raise ValueError(f"mesh must be (nk, 2), got {self.mesh.shape}")
        self.nk = len(self.mesh)
        self.nmax = int(nmax)
        if self.nmax < 0:
            raise ValueError(f"nmax must be >= 0, got {self.nmax}")
        self.a_cart = np.asarray(a_cart, dtype=float)
        self.lat_to_cart = np.asarray(lat_to_cart, dtype=float)
        self.cart_to_lat = np.asarray(cart_to_lat, dtype=float)
        A = np.asarray(a_ints, dtype=float)
        if A.ndim != 2 or A.shape[1] != 2:
            raise ValueError(f"a_ints must be (na, 2), got {A.shape}")
        self.phase_a = magnetic_translation_phase(A, FLUX_QUANTA_PER_PRIMITIVE_CELL)
        self.na = len(A)

    @property
    def n_bands(self) -> int:
        """``nmax + 1``.  Derived, never an input -- see the module docstring."""
        return self.nmax + 1

    @property
    def flux_quanta_per_cell(self) -> float:
        """``area / (2 pi l_B^2)`` of the cell the Bloch sum runs over (here, 1)."""
        area = abs(np.linalg.det(self.lat_to_cart))
        return flux_quanta(area)

    # -- the Bloch sum -------------------------------------------------------
    def _in_cell(self, r_in_cell, upto: int) -> np.ndarray:
        """``psi_n(r)`` for r inside the primitive cell, levels ``0 .. upto``."""
        r_in_cell = np.atleast_2d(r_in_cell)
        npts = r_in_cell.shape[0]
        big = r_in_cell[:, None, :] + self.a_cart[None, :, :]        # (npts, na, 2)
        x = big[:, :, 0]
        y = big[:, :, 1]
        z_bar = x - 1j * y
        r_sq = x * x + y * y
        # exp(i (r x a)_z / 2), built from the FOLDED point r_in_cell, not from big.
        phase_r = np.exp(1j * (r_in_cell[:, 0:1] * self.a_cart[None, :, 1]
                               - r_in_cell[:, 1:2] * self.a_cart[None, :, 0]) / 2.0)
        w = self.phase_a[None, :] * phase_r                          # (npts, na)
        out = np.zeros((npts, upto + 1), dtype=complex)
        for n in range(upto + 1):
            out[:, n] = np.sum(disk_state(z_bar, r_sq, n) * w, axis=1)
        return out

    def padded_orbitals(self, r) -> np.ndarray:
        """``psi_{k,n}`` for levels ``0 .. nmax+1``.  Returns (nk, nmax + 2)."""
        return self._bloch(r, self.nmax + 1)

    def orbitals(self, r) -> np.ndarray:
        """``psi_{k,n}`` for the physical levels ``0 .. nmax``.  Returns (nk, nmax + 1)."""
        return self._bloch(r, self.nmax)

    def _bloch(self, r, upto: int) -> np.ndarray:
        r = np.asarray(r, dtype=float).ravel()
        if r.shape[0] != 2:
            raise ValueError(f"r must be a 2-vector, got shape {r.shape}")
        k = self.mesh
        # rk = k x z_hat = (k_y, -k_x); the orbital is evaluated at r - z_hat x k.
        rk = np.stack([k[:, 1], -k[:, 0]], axis=1)
        rk_pt = r[None, :] - rk                                      # (nk, 2)
        _, rk_in_cell, shift_lat, shift_cart = fold_to_primitive_cell(
            rk_pt, self.lat_to_cart, self.cart_to_lat)
        shift_fac = np.exp(-1j * (
            (rk_in_cell[:, 0] * shift_cart[:, 1] - rk_in_cell[:, 1] * shift_cart[:, 0]) / 2.0
            + np.pi * shift_lat[:, 0] * shift_lat[:, 1]))
        psi = self._in_cell(rk_in_cell, upto)
        # the gauge-completion factor uses the UNFOLDED r
        exp_fac = np.exp(1j * (r[0] * k[:, 0] + r[1] * k[:, 1]) / 2.0)
        return psi * (shift_fac * exp_fac)[:, None]

    def ladder(self, r) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pi_x psi, pi_y psi, pi^2 psi)`` for the physical bands.

        The magnetic ladder operators act on the **band index**, so no coordinate
        derivative is taken::

            a |n> = sqrt(n) |n-1>    a^dag |n> = sqrt(n+1) |n+1>
            pi_x = (a + a^dag)/sqrt(2)    pi_y = (a - a^dag)/(i sqrt(2))
            pi^2 |n> = (2n + 1) |n>

        ``pi^2`` is exactly diagonal, which is a property of this basis and of no other
        one in this project -- a Gaussian basis has no such property, and the kinetic
        estimator for it must take derivatives instead.  That is why the two kinetic
        paths are not merged.

        Returns three ``(nk, nmax + 1)`` complex arrays.  The padding level ``nmax + 1``
        is consumed here (as ``a^dag |nmax>``) and does not appear in the output.
        """
        Op = self.padded_orbitals(r)                                 # (nk, nmax + 2)
        nb = self.n_bands
        O = Op[:, :nb]
        a = np.zeros((self.nk, nb), dtype=complex)
        ad = np.zeros((self.nk, nb), dtype=complex)
        ad[:, 0] += Op[:, 1]
        for n in range(1, nb):
            ad[:, n] += np.sqrt(n + 1) * Op[:, n + 1]
            a[:, n] += np.sqrt(n) * Op[:, n - 1]
        pix = (a + ad) / np.sqrt(2)
        piy = (a - ad) / (1j * np.sqrt(2))
        pi2 = O * (2 * np.arange(nb) + 1)[None, :].astype(float)
        return pix, piy, pi2
