"""The Slater-Jastrow determinant core: state, and the Metropolis move.

Physical role:
    ``Psi(R) = det[ psi_a(r_b) ] * exp( sum_{a<b} u(r_a - r_b) )``.  This module
    owns the determinant and its rank-1 updates, the Jastrow pair matrices
    attached to a configuration, and the single-electron move that the Metropolis
    sampler proposes.  It is orbital-agnostic: it is handed an ``orbital_fn`` (and
    later a ``pi_fn``/``pi2_fn``) and never asks which ansatz produced them.

Main mathematical object:
    ``D[a, b] = psi_a(r_b)`` -- rows are orbitals, COLUMNS are electrons, so
    moving electron ``b`` changes column ``b``.  The ratio
    ``det(D')/det(D) = D_new_col_i . D_inv[i, :]`` and the Sherman-Morrison update
    of ``D_inv``.  The Jastrow pair matrices ``u``, ``gx``, ``gy``, ``lap`` are
    (ne, ne) with a zero diagonal and ``U = sum(u)/2``.

Units/convention:
    ``l_B = 1``; energies in ``hbar omega_c``.  ``kappa`` is stored but NOT
    applied here -- ``V`` is the bare ``sum 1/r`` and the coupling enters only in
    ``E = T + kappa*V``.

Legacy source:
    ``qhvmc_engine.py:719-900`` -- ``calc_D_ratio``, ``slater_ratio``,
    ``update_D_inv``, ``calc_kinetic_energy``, ``Wavefunction`` in full:
    ``__init__``, ``build``, ``rebuild``, ``_set_u``, ``move_ratio``,
    ``accept_move``, ``pi_columns``, ``local_energy``.

Stage 2B / B4 (state and move) and B5 (``calc_kinetic_energy``, ``pi_columns``,
``local_energy``).  RNG, proposal, acceptance and energy semantics are the
frozen engine's.  The class is split across B4/B5 along the seam the staging
draws (sampler state vs local-energy evaluation), not along an architectural
idea of my own.

B5 note -- ``pi_columns`` really does take a configuration here and a state dict
in the LL-rotation subclass.  That is the legacy interface, the two call sites
inside the engine are each self-consistent with their own class, and the
difference is deliberately preserved rather than unified: unifying it is an API
redesign and belongs to Stage 2D.  See ``BUG_CANDIDATE.md`` #1.
"""
from __future__ import annotations

import numpy as np

__all__ = ["calc_D_ratio", "slater_ratio", "update_D_inv",
           "calc_kinetic_energy", "Wavefunction"]


def calc_D_ratio(D_new_coli, D_inv_rowi):
    """det(D')/det(D) when only COLUMN i of D changes to D_new_coli.

    Mirrors calc_D_ratio() in fast_determinant_updates.jl.  Note the Julia
    source writes dot(conj.(D_inv_rowi), D_new_coli) -- but Julia's dot()
    already conjugates its first argument, so the two conjugations cancel and
    this is a plain (unconjugated) contraction.

    No conjugation here, and that is not an oversight: conjugating either
    argument turns the Metropolis ratio into a different (wrong) number while
    leaving it real and in [0, 1], so it would sample a different distribution
    without ever raising.
    """
    return np.dot(D_new_coli, D_inv_rowi)


# kept as a descriptive alias
slater_ratio = calc_D_ratio


def update_D_inv(i, delta_D_new_coli, D_inv, denom=None):
    """Sherman-Morrison rank-1 update of the inverse Slater matrix.

    Mirrors update_D_inv() in fast_determinant_updates.jl:
        D_inv <- D_inv - outer(D_inv @ delta, D_inv[i,:]) / (1 + D_inv[i,:] . delta)

    `denom` may be supplied (it equals calc_D_ratio(D[:,i] + delta, D_inv[i,:])).
    """
    if denom is None:
        denom = 1.0 + np.dot(delta_D_new_coli, D_inv[i, :])
    return D_inv - np.outer(D_inv @ delta_D_new_coli, D_inv[i, :]) / denom


def calc_kinetic_energy(pix_D, piy_D, pi_square_D, D_inv,
                        gradx_u_pair_mat, grady_u_pair_mat, laplacian_u_pair_mat):
    """Mirrors calc_kinetic_energy() in kinetic.jl.

    T = T_det + T_mix + T_jastrow with
      T_det     = (1/2) sum_i  <i| pi^2 |i>_D
      T_mix     = -i sum_i [ gx_i <i|pi_x|i>_D + gy_i <i|pi_y|i>_D ]
      T_jastrow = -(1/2) sum_i [ gx_i^2 + gy_i^2 ] - (1/2) sum_{i,j} lap_ij
    where gx_i = sum_j d_x u(r_i - r_j) (row sums of the pair matrices) and
    <i|O|i>_D = calc_D_ratio(O[:, i], D_inv[i, :]).
    """
    D_inv = np.asarray(D_inv)
    px = np.einsum("li,il->i", pix_D, D_inv)
    py = np.einsum("li,il->i", piy_D, D_inv)
    q = np.einsum("li,il->i", pi_square_D, D_inv)

    gx = np.sum(gradx_u_pair_mat, axis=1)
    gy = np.sum(grady_u_pair_mat, axis=1)

    T_jastrow = -0.5 * (np.sum(gx * gx) + np.sum(gy * gy) + np.sum(laplacian_u_pair_mat))
    T_det = 0.5 * np.sum(q)
    T_mix = -1j * (np.dot(gx, px) + np.dot(gy, py))
    return T_det + T_mix + T_jastrow


class Wavefunction:
    """Psi(R) = det[ psi_a(r_b) ] * exp( sum_{a<b} u(r_a - r_b) ).

    `orbital_fn(r)` returns the (ne,) column of orbitals at r; `pi_fn(r)`
    returns (2, ne) = (pi_x psi, pi_y psi) and `pi2_fn(r)` returns (ne,) = pi^2 psi.
    `jastrow` is a SinSplineJastrow or None.

    The determinant convention matches the reference: D[a,b] = psi_a(r_b),
    so a move of electron b changes COLUMN b.
    """

    def __init__(self, orbital_fn, pi_fn, pi2_fn, jastrow, ne,
                 sc_to_cart, kappa=0.0, ham=None):
        self.orbital_fn = orbital_fn
        self.pi_fn = pi_fn
        self.pi2_fn = pi2_fn
        self.jastrow = jastrow
        self.ne = int(ne)
        self.sc_to_cart = np.asarray(sc_to_cart, dtype=float)
        self.cart_to_sc = np.linalg.inv(self.sc_to_cart)
        self.kappa = float(kappa)
        self.ham = ham

    # -- state ---------------------------------------------------------------
    def build(self, R):
        ne = self.ne
        R = np.asarray(R, dtype=float)
        D = np.column_stack([self.orbital_fn(R[j]) for j in range(ne)])
        D_inv = np.linalg.inv(D)
        st = {"R": R.copy(), "D": D, "D_inv": D_inv,
              "u": None, "gx": None, "gy": None, "lap": None, "U": 0.0}
        self._set_u(st)
        return st

    def rebuild(self, st):
        """Recompute D, D_inv and the Jastrow pair matrices from st["R"].

        Rank-1 Sherman-Morrison updates accumulate round-off; measured drift is
        exponential (~1e-13 after 3.6e3 updates but ~1.0 after 4e4), and an
        inaccurate D_inv corrupts every Metropolis ratio.  The reference calls
        this every reset_period steps -- the same reason.
        """
        R = st["R"]
        st["D"] = np.column_stack([self.orbital_fn(R[j]) for j in range(self.ne)])
        st["D_inv"] = np.linalg.inv(st["D"])
        self._set_u(st)

    def _set_u(self, st):
        if self.jastrow is None:
            ne = self.ne
            st["u"] = np.zeros((ne, ne))
            st["gx"] = np.zeros((ne, ne))
            st["gy"] = np.zeros((ne, ne))
            st["lap"] = np.zeros((ne, ne))
            st["U"] = 0.0
        else:
            u, gx, gy, lap = self.jastrow.u_pairs(st["R"])
            st["u"], st["gx"], st["gy"], st["lap"] = u, gx, gy, lap
            st["U"] = u.sum() / 2.0

    # -- Metropolis move -----------------------------------------------------
    def move_ratio(self, st, i, r_new):
        """|Psi'/Psi|^2 for moving electron i to r_new, plus accept bookkeeping."""
        col_new = self.orbital_fn(r_new)
        slater_ratio = calc_D_ratio(col_new, st["D_inv"][i, :])
        delta_col = col_new - st["D"][:, i]

        if self.jastrow is None:
            return abs(slater_ratio) ** 2, (slater_ratio, delta_col, None)

        idx = np.arange(self.ne) != i
        Rj = st["R"][idx]                            # the other ne-1 electrons
        u_row = self.jastrow.u(r_new - Rj)
        u_col = self.jastrow.u(Rj - r_new)
        # U = (1/2) sum_{a,b} u_mat[a,b]; only row i and column i change
        dU = 0.5 * ((u_row.sum() + u_col.sum())
                    - (st["u"][i, :].sum() + st["u"][:, i].sum()))
        acc = abs(slater_ratio) ** 2 * np.exp(2.0 * dU)
        return acc, (slater_ratio, delta_col, (idx, Rj, u_row, u_col))

    def accept_move(self, st, i, r_new, info):
        slater_ratio, delta_col, jinfo = info
        st["R"][i] = r_new
        st["D"][:, i] += delta_col
        st["D_inv"] = update_D_inv(i, delta_col, st["D_inv"], slater_ratio)

        if jinfo is not None:
            idx, Rj, u_row, u_col = jinfo
            # d stays antisymmetric: grad(r_j - r_i) = -grad(r_i - r_j),
            # laplacian(r_j - r_i) = laplacian(r_i - r_j) (even in r).
            gr = self.jastrow.grad(r_new - Rj)
            lp = self.jastrow.laplacian(r_new - Rj)
            st["u"][i, idx] = u_row
            st["u"][idx, i] = u_col
            st["gx"][i, idx] = gr[:, 0]
            st["gx"][idx, i] = -gr[:, 0]
            st["gy"][i, idx] = gr[:, 1]
            st["gy"][idx, i] = -gr[:, 1]
            st["lap"][i, idx] = lp
            st["lap"][idx, i] = lp
            st["U"] = st["u"].sum() / 2.0

    # -- local energy --------------------------------------------------------
    def pi_columns(self, R):
        ne = self.ne
        pix = np.empty((ne, ne), dtype=complex)
        piy = np.empty((ne, ne), dtype=complex)
        pi2 = np.empty((ne, ne), dtype=complex)
        for j in range(ne):
            p = self.pi_fn(R[j])
            pix[:, j] = p[0]
            piy[:, j] = p[1]
            pi2[:, j] = self.pi2_fn(R[j])
        return pix, piy, pi2

    def _pi_columns_for(self, st):
        """The state's (pix, piy, pi2), in THIS ansatz's addressing.

        The base class addresses ``pi_columns`` by CONFIGURATION; the rotated
        ansatz addresses it by STATE, because the batch ``build`` already paid
        for is worth reusing.  That split is the legacy interface and is kept on
        purpose (``BUG_CANDIDATE.md`` #1), so it is bridged here rather than by
        unifying the two public signatures.  ``local_energy`` is consequently one
        formula, not one per ansatz.
        """
        return self.pi_columns(st["R"])

    def local_energy(self, st):
        """(T, V, E_loc).  V is the bare sum 1/r; the coupling kappa is applied
        to V inside E_loc = T + kappa*V (mirroring kinetic.jl + coulomb.jl)."""
        R = st["R"]
        pix, piy, pi2 = self._pi_columns_for(st)
        T = calc_kinetic_energy(pix, piy, pi2, st["D_inv"],
                                st["gx"], st["gy"], st["lap"])
        V = self.ham.energy(R) if self.ham is not None else 0.0
        return T, V, T + self.kappa * V
