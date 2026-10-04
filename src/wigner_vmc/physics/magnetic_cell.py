"""Magnetic-cell conventions: gauge, flux quanta, and the fold a Bloch orbital needs.

Stage 2A.  `geometry.py` owns the Euclidean lattice; this module owns everything that
only exists because there is a magnetic field.

The three conventions that are easy to get wrong, and what they are here
-----------------------------------------------------------------------
1. **Gauge.**  ``A(r) = (1/2) * (-y, x)``, so ``curl A = +B z_hat`` with ``B = 1``.
   With ``pi = -i grad - A`` and ``a = (pi_x + i pi_y)/sqrt(2)`` this gives
   ``pi^2 = 2 a^dag a + 1`` and the n-th Landau level at ``pi^2 = 2n + 1``.  The
   opposite sign of ``B`` swaps ``a`` and ``a^dag``, which is invisible in ``pi^2``
   (so it does not show up in an energy) but flips the sign of every ``pi psi`` --
   and therefore shows up in the kinetic estimator through ``psi^dag pi^2 psi`` only
   at second order.  `landau_levels` pins the sign with a finite-difference test.

2. **Flux quanta, and the factor that is invisible in one place and 36 in the other.**
   A magnetic Bloch sum over a lattice with ``n_phi`` flux quanta per cell carries the
   phase ``exp(i * pi * m * n * n_phi)`` for the lattice vector ``m*a1 + n*a2``.
   In the Landau-level basis the lattice is the *primitive* cell, whose area is
   ``2*pi = 2*pi*l_B^2`` -- exactly one flux quantum -- so the factor is ``1`` and the
   legacy code does not write it.  In the Gaussian basis the sum runs over *supercell*
   translations, ``36`` flux quanta per cell, and the factor is written out
   (``l_ints[:,0]*l_ints[:,1]*n_phi*pi``).  The two lines of code look different and
   are the same rule; taking the Gaussian form and copying it into the LL basis, or
   dropping the factor from the Gaussian form because "the LL basis does not have it",
   are the same mistake in two directions.  `magnetic_translation_phase` is the rule,
   written once, with the flux count as an argument.

3. **The fold.**  A Bloch orbital needs one representative per image class, and the
   right one is the point inside the primitive parallelogram -- obtained by rounding
   fractional coordinates -- *not* the nearest image.  See `geometry.minimum_image`
   for why those differ, and `fold_to_primitive_cell` for the phases the fold drags
   along with it.

Units: ``l_B = 1`` throughout, as in the frozen engine.  The magnetic length is not a
parameter here; `flux_quanta` takes it as an argument only so that its appearance in
the formula is visible, and every caller passes the default.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "GAUGE", "flux_quanta", "magnetic_translation_phase", "primitive_frame",
    "fold_to_primitive_cell",
]

#: The vector potential, written down so that it is greppable.  ``B = curl A = +1 z_hat``.
GAUGE = "A(r) = (1/2) * (-y, x),  curl A = +B z_hat,  B = 1,  l_B = 1"


def flux_quanta(area: float, l_B: float = 1.0) -> float:
    """Flux through ``area`` in units of the flux quantum: ``area / (2*pi*l_B^2)``.

    For the production supercell (``area = 72*pi``) this is ``36`` -- equal to the
    electron count, which is what ``nu = 1`` means.  For the primitive cell
    (``area = 2*pi``) it is ``1``.

    The name is the engine's ``n_phi``.
    """
    return float(area / (2.0 * np.pi * l_B ** 2))


def magnetic_translation_phase(a_ints, n_phi) -> np.ndarray:
    """``exp(i * pi * m * n * n_phi)`` for each integer pair ``(m, n)``.

    ``n_phi`` must be the flux quanta of the cell that ``a_ints`` are coefficients
    of.  Passing the supercell count for primitive-cell coefficients (or the reverse)
    multiplies every orbital by a wrong phase: the orbitals stay unit-modulus and the
    determinant stays non-singular, so nothing raises and only the energies move.
    """
    A = np.asarray(a_ints, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError(f"a_ints must be (n, 2), got {A.shape}")
    return np.exp(1j * np.pi * A[:, 0] * A[:, 1] * float(n_phi))


def primitive_frame(A1, A2) -> Tuple[np.ndarray, np.ndarray]:
    """``(lat_to_cart, cart_to_lat)`` for a primitive cell, columns = the vectors.

    ``lat_to_cart`` is the matrix whose columns are ``A1, A2``, so that
    ``r_cart = lat_to_cart @ r_frac``; the second return value is its exact inverse.
    """
    C = np.column_stack([np.asarray(A1, float), np.asarray(A2, float)])
    return C, np.linalg.inv(C)


def fold_to_primitive_cell(r, lat_to_cart, cart_to_lat):
    """Fold a point into the first cell of an arbitrary lattice frame.

    Returns ``(r_lat, r_cart, shift_lat, shift_cart)``:

    * ``r_lat``   -- fractional coordinates of the folded point, each in the CLOSED
      ``[-0.5, 0.5]``.  Both ends occur, but only for inputs that are exactly half a
      cell: ``np.round`` is half-to-even, so ``t - round(t)`` fixes ``+0.5`` (rounds to
      0) and ``-0.5`` (rounds to -0) and sends everything else strictly inside;
    * ``r_cart``  -- the folded point in Cartesian coordinates;
    * ``shift_lat``  -- the integer lattice vector that was subtracted, in fractional
      coordinates (so ``shift_cart = shift_lat @ lat_to_cart.T``);
    * ``shift_cart`` -- the same vector in Cartesian coordinates.

    ``shift_cart`` carries the FULL lattice vector, not a fractional remainder.  The
    Landau-level orbitals use it in a phase together with ``shift_lat``; a version
    that returned ``r - r_cart`` in some other sense, or reduced ``shift_lat`` modulo
    anything, would produce orbitals that are still smooth and still periodic-looking
    and are not the magnetic Bloch orbitals.

    Rounding is ``np.round``, i.e. half-to-even.  A point at exactly half a cell is a
    fixed point of the fold and stays where it is; a point a hair inside comes back a
    hair inside the other edge.  That is a convention rather than a consequence of
    anything, and the regression test pins both halves of it.
    """
    r = np.atleast_2d(np.asarray(r, dtype=float))
    if r.shape[1] != 2:
        raise ValueError(f"r must be (n, 2), got {r.shape}")
    r_lat = r @ cart_to_lat.T
    r_lat = r_lat - np.round(r_lat)
    shift_lat = r @ cart_to_lat.T - r_lat
    r_cart = r_lat @ lat_to_cart.T
    shift_cart = r - r_cart
    return r_lat, r_cart, shift_lat, shift_cart
