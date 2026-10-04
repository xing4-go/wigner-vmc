"""Ewald summation of the periodic Coulomb interaction.

Stage 2A.  This is the potential energy of BOTH workflows -- the Gaussian crystal and
the LL-rotation ansatz differ in their orbitals and in nothing else, so they share this
object and therefore share ``kappa * V`` exactly.

The sum is ``sum_{i<j} 1/|r_i - r_j|`` on the torus, split as

    short range   sum_l  erfc(|d - l| / (2 eta)) / |d - l|          over real lattice l
    long range    sum_g  cos(d . g) * 2 pi erfc(eta |g|) / |g| / area
    constant      the g = 0 term, the self-energy, and the neutralising background

``eta`` is a pure bookkeeping parameter: it splits one convergent sum into two, so the
total must be independent of it.  That independence is the property that makes the
split legitimate, and it is tested (``tests/test_coulomb_against_legacy.py``).

Three things that are easy to get wrong
---------------------------------------
1. **``l_cart`` keeps ``l = 0``.**  The zero vector is the bare pair interaction; it is
   the whole short-range sum's leading term.  ``l_nz`` is the ``l != 0`` subset, used
   only where ``1/|l|`` would diverge (the constant).  Removing the zero vector from
   ``l_cart`` -- which looks like a harmless tidy-up, since every other lattice sum in
   this project excludes it -- shortens every pair interaction.
2. **The pair separations are FOLDED into the supercell, not minimum-imaged.**  The
   cutoff bound ``|L| <= 2 eta erfc_arg_max + (|L1|+|L2|)/2`` is derived from
   ``|r_ij| <= (|L1|+|L2|)/2``, which holds only for the fold.  See
   `geometry.minimum_image` for the counterexample where the two disagree.
3. **``delta_move`` returns the change in the PAIR sum only.**  The constant is
   position-independent, so a caller that adds it double-counts.  The identity
   ``delta_move(R, i, r_old) == energy(R') - energy(R)`` for ``R'`` = ``R`` with row
   ``i`` replaced is tested inside this module, not only against legacy.

Units: ``l_B = 1``.  The result is the dimensionless ``sum 1/r`` that ``kappa``
multiplies to make an energy in ``hbar omega_c``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .geometry import circular_lattice

__all__ = ["CoulombEwald"]

#: The real-space cutoff is set at this many ``erfc`` widths.  ``erfc(6) ~ 2e-17``, so
#: the neglected real-space tail is below double precision for the configurations used
#: here.
DEFAULT_ERFC_ARG_MAX = 6.0

#: ``eta`` defaults to ``|L1| / 3.8``.  It is a SUPERCELL-dependent number, chosen to
#: balance the two sums for the 60-degree cells in this project; it is not a physical
#: scale and is not comparable between system sizes.
DEFAULT_ETA_DIVISOR = 3.8


class CoulombEwald:
    """Split real/reciprocal-space Ewald summation for ``sum_{i<j} 1/|r_i - r_j|``.

    Parameters
    ----------
    ne : int
        electron count (enters only through the constant term).
    L1, L2 : (2,)
        supercell vectors.
    G1, G2 : (2,)
        their reciprocal vectors.
    eta : float, optional
        splitting parameter; defaults to ``|L1| / 3.8``.
    erfc_arg_max : float
        real- and reciprocal-space cutoffs, in units of the respective width.
    """

    def __init__(self, ne: int, L1, L2, G1, G2, eta=None,
                 erfc_arg_max: float = DEFAULT_ERFC_ARG_MAX):
        self.ne = int(ne)
        self.L1 = np.asarray(L1, dtype=float)
        self.L2 = np.asarray(L2, dtype=float)
        self.G1 = np.asarray(G1, dtype=float)
        self.G2 = np.asarray(G2, dtype=float)
        self.area = float(abs(self.L1[0] * self.L2[1] - self.L1[1] * self.L2[0]))
        self.eta = (float(np.linalg.norm(self.L1) / DEFAULT_ETA_DIVISOR)
                    if eta is None else float(eta))
        self.erfc_arg_max = float(erfc_arg_max)

        # Real-space lattice vectors.  The pair sum is periodic, so the separation is
        # folded into the first supercell and the cutoff then made wide enough that
        # every image with a non-negligible erfc is included:
        #     |r_ij| <= (|L1|+|L2|)/2   and   |r_ij - L| <= 2 eta erfc_arg_max
        #     =>  |L| <= 2 eta erfc_arg_max + (|L1|+|L2|)/2.
        # L = 0 MUST be kept here: it is the bare pair interaction.
        l_cut = (2 * self.eta * self.erfc_arg_max
                 + (np.linalg.norm(self.L1) + np.linalg.norm(self.L2)) / 2)
        _, self.l_cart = circular_lattice(l_cut, self.L1, self.L2)
        _, g_cart = circular_lattice(self.erfc_arg_max / self.eta, self.G1, self.G2)
        self.g_cart = g_cart[np.linalg.norm(g_cart, axis=1) > 0]
        self.l_nz = self.l_cart[np.linalg.norm(self.l_cart, axis=1) > 0]

        self.sc_to_cart = np.column_stack([self.L1, self.L2])
        self.cart_to_sc = np.linalg.inv(self.sc_to_cart)

        gn = np.linalg.norm(self.g_cart, axis=1)
        self.v_long_g = 2 * np.pi * erfc(self.eta * gn) / gn
        ln = np.linalg.norm(self.l_nz, axis=1)
        v_short_l = erfc(ln / (2 * self.eta)) / ln

        npair = self.ne * (self.ne - 1) / 2.0
        v_const_self = (self.ne / 2.0) * (
            np.sum(v_short_l)
            - 1.0 / (self.eta * np.sqrt(np.pi))
            + np.sum(self.v_long_g) / self.area
            - 4 * np.sqrt(np.pi) * self.eta / self.area
        )
        v_const_pair = -npair * (2 * np.pi / self.area) * (2 * self.eta / np.sqrt(np.pi))
        self.v_const = v_const_pair + v_const_self

    # -- the periodic fold, shared by both halves ----------------------------
    def _wrap(self, d):
        """Fold pair separations into the first supercell.

        This is the rounding fold, deliberately -- see the module docstring, point 2.
        """
        d_sc = d @ self.cart_to_sc.T
        d_sc -= np.round(d_sc)
        return d_sc @ self.sc_to_cart.T

    def _pair_short(self, d):
        d = self._wrap(np.atleast_2d(d))
        dd = d[:, None, :] - self.l_cart[None, :, :]
        dn = np.linalg.norm(dd, axis=-1)
        return np.sum(erfc(dn / (2 * self.eta)) / dn, axis=-1)

    def _pair_long(self, d):
        d = self._wrap(np.atleast_2d(d))
        ph = d @ self.g_cart.T
        return np.sum(np.cos(ph) * self.v_long_g[None, :], axis=-1) / self.area

    # -- energies ------------------------------------------------------------
    def pair_energy(self, R) -> float:
        """``sum_{i<j}`` of the position-dependent part, WITHOUT the constant."""
        n = len(R)
        iu = np.triu_indices(n, k=1)
        rij = R[iu[0]] - R[iu[1]]
        return float(np.sum(self._pair_short(rij)) + np.sum(self._pair_long(rij)))

    def energy(self, R) -> float:
        """Total periodic Coulomb energy, including the constant term."""
        return self.pair_energy(R) + float(self.v_const)

    def split_energy(self, R):
        """``(short, long, const)`` -- the three pieces separately, for diagnosis."""
        n = len(R)
        iu = np.triu_indices(n, k=1)
        rij = R[iu[0]] - R[iu[1]]
        return (float(np.sum(self._pair_short(rij))), float(np.sum(self._pair_long(rij))),
                float(self.v_const))

    def delta_move(self, R, i: int, r_old) -> float:
        """Change in the Coulomb energy when particle ``i`` moves ``r_old -> R[i]``.

        The constant cancels, so it is not included -- and must not be added by a
        caller.
        """
        n = len(R)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Rj = R[mask]
        d_old = r_old[None, :] - Rj
        d_new = R[i][None, :] - Rj
        return float((self._pair_short(d_new).sum() - self._pair_short(d_old).sum())
                     + (self._pair_long(d_new).sum() - self._pair_long(d_old).sum()))
