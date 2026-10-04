"""Landau-level-rotation orbitals: ``U = exp(iK)`` acting on the band index.

Physical role:
    The crystal orbital basis of the *LL-rotation* workflow.  Instead of mixing
    Gaussians, it takes the magnetic Bloch states of the Landau-level basis and
    mixes the LANDAU LEVELS at each momentum.  ``v = 0`` is the identity rotation
    and therefore IS the filled-LLL liquid; nonzero ``v`` rotates weight into
    higher levels and is what lets one family interpolate between the liquid and
    the crystal.

Main mathematical object:
    ``psi_k(r) = sum_n C[k,n] phi_{k,n}(r)`` with ``C[k]`` the first row of
    ``exp(i K^k)``, written in closed form as
    ``C[k] = (cos|v_k|, i (sin|v_k|/|v_k|) v_k)``.  Row 0 is real by
    construction -- that is the gauge fixing.  The kinetic operators act on the
    BAND index through the magnetic ladder, so ``pi^2`` is diagonal in ``n`` and
    no coordinate derivative is ever taken.

Units/convention:
    ``l_B = 1``.  ``v`` is complex with shape ``(nk, n_bands - 1)``; ``C`` has
    shape ``(nk, n_bands)``.  ``n_band`` here is the physics-core ``nmax + 1``;
    the legacy engine's ``LandauLevelBasis.n_max`` is the BAND COUNT, so a caller
    holding a band count converts it once via
    ``physics.landau_levels.nmax_from_n_bands`` rather than by hand.

Legacy source:
    ``qhvmc_engine_llrot.py:59-243`` -- ``_wtaylor``, ``c_row``,
    ``dc_row_dparams``, ``LLRotatedOrbitals``, verbatim.  The wavefunction
    subclass is split across B3/B4/B5/B7 along the seams the staging draws:
    ``build``/``rebuild``/``_ops_of``/``_D_from_ops`` (here, needed by the
    sampler), ``pi_columns``/``local_energy`` (B5), ``orbital_log_deriv``/
    ``log_deriv``/``v_from_theta`` (B7).

Stage 2B / B3 -- mechanical migration.  Gaussian and LL-rotation orbital
architecture are deliberately NOT unified; see the note in `magnetic_cell`.
"""
from __future__ import annotations

import numpy as np

from .jastrow import jastrow_log_deriv
from .slater import Wavefunction

__all__ = ["_wtaylor", "c_row", "dc_row_dparams", "LLRotatedOrbitals",
           "LLRotationWavefunction"]


def _wtaylor(t):
    """(sin t)/t and (t cos t - sin t)/t^3, with their t -> 0 limits.

    Both are entire functions of t^2.  The t -> 0 branch is not a formality:
    v = 0 IS the liquid (U = I), the starting point of every liquid calculation
    and of the crystal's continuation ladder.
    """
    small = t < 1e-6
    ts = np.where(small, 1.0, t)
    w1 = np.where(small, 1.0 - t * t / 6.0, np.sin(ts) / ts)
    w4 = np.where(small, -1.0 / 3.0 + t * t / 30.0,
                  (ts * np.cos(ts) - np.sin(ts)) / ts**3)
    return w1, w4


def c_row(v):
    """v : (nk, NB-1) complex  ->  C : (nk, NB) complex.

    C[k] = (cos|v_k|, i (sin|v_k|/|v_k|) v_k) -- row 0 of exp(i K^k).
    ``C[:, 0]`` is real by construction; that is the gauge fixing.
    """
    v = np.atleast_2d(np.asarray(v, dtype=complex))
    nk, m = v.shape
    t = np.linalg.norm(v, axis=1)
    w1, _ = _wtaylor(t)
    C = np.empty((nk, m + 1), dtype=complex)
    C[:, 0] = np.cos(t)
    C[:, 1:] = 1j * w1[:, None] * v
    return C


def dc_row_dparams(v):
    """Analytic dC[k]/da[k,j] and dC[k]/db[k,j] for v = a + i b.

    Returns (dC_da, dC_db), each (nk, NB-1, NB); index [k, j, n] is the
    derivative of component n of row k with respect to a[k,j] (resp. b[k,j]).

    Derived by writing C = C(v, conj v) and using, for a real parameter,
    d/da = d/dv + d/dconj(v) and d/db = i (d/dv - d/dconj(v)).  At v = 0 the
    expressions reduce to dC[1+m]/da_j = i delta_mj and dC[1+m]/db_j = -delta_mj,
    i.e. the identity-rotation directions, which is what they must be.
    Checked against central finite differences in the notebook (test 3).
    """
    v = np.atleast_2d(np.asarray(v, dtype=complex))
    nk, m = v.shape
    t = np.linalg.norm(v, axis=1)
    w1, w4 = _wtaylor(t)
    a, b = v.real, v.imag

    dC_da = np.zeros((nk, m, m + 1), dtype=complex)
    dC_db = np.zeros((nk, m, m + 1), dtype=complex)

    dC_da[:, :, 0] = -(w1[:, None] * a)
    dC_db[:, :, 0] = -(w1[:, None] * b)

    # index [k, j, m]: derivative of component 1+m with respect to param j
    outer_a = v[:, None, :] * a[:, :, None]     # (nk, m, m): v_j * Re(v_m)
    outer_b = v[:, None, :] * b[:, :, None]
    dC_da[:, :, 1:] = 1j * w4[:, None, None] * outer_a
    dC_db[:, :, 1:] = 1j * w4[:, None, None] * outer_b
    dC_da[:, :, 1:] += 1j * w1[:, None, None] * np.eye(m, dtype=complex)[None]
    dC_db[:, :, 1:] -= w1[:, None, None] * np.eye(m, dtype=complex)[None]
    return dC_da, dC_db


class LLRotatedOrbitals:
    """psi_k, pi psi_k, pi^2 psi_k for the rotated Landau-level orbitals.

    Wraps a ``LandauLevelBasis`` built with ``nmax = NB - 1`` (so the basis
    returns NB bands plus the one padding band the ladder operators need)
    together with a parameter vector ``v`` of shape (nk, NB-1).

    The legacy class required ``ll_basis.n_max == n_band`` because its
    ``n_max`` WAS the band count; here the basis is built by the clean class,
    whose ``nmax`` is the highest Landau index, so the check is
    ``ll_basis.n_bands == n_band``.  The object being validated is the same.

    The underlying Bloch sum is evaluated once per distinct position and reused:
    the kinetic-energy estimator asks for pi and pi^2 at the same point back to
    back, and pi^2 at a point already visited by pi.  The cache holds the last
    position only, because during a Metropolis move the next call is always a new
    point -- a larger cache would cost memory and buy nothing.
    """

    def __init__(self, ll_basis, n_band, v):
        self.ll = ll_basis
        self.nb = int(n_band)
        if ll_basis.n_bands != self.nb:
            raise ValueError(f"LandauLevelBasis n_bands={ll_basis.n_bands} but "
                             f"n_band={self.nb}; the basis must be built with "
                             "nmax = n_band - 1")
        self.nk = ll_basis.nk
        self.v = np.asarray(v, dtype=complex).reshape(self.nk, self.nb - 1)
        self.C = c_row(self.v)
        self._rx = None
        self._rO = None
        self._bR = None
        self._bOps = None

    # -- cached raw basis evaluation ----------------------------------------
    def _Op(self, r):
        r = np.asarray(r, dtype=float)
        if self._rx is not None and np.array_equal(r, self._rx):
            return self._rO
        Op = self.ll.padded_orbitals(r)           # (nk, nb+1)
        self._rx = r.copy()
        self._rO = Op
        return Op

    def basis_at(self, r):
        """(nk, nb) : the unrotated phi_{k,n}(r), for the SR derivative."""
        return self._Op(r)[:, :self.nb]

    def batch(self, R):
        """(ne, nk, nb+1) : the padded basis at every electron, in one call.

        This exists purely because the cost of this ansatz IS the Bloch sum: one
        evaluation is ~0.6 ms at N=36 and nb=2, so a single local energy spends
        ~22 ms on the 36 electrons.  ``build``, ``pi_columns`` and ``log_deriv``
        all need exactly the same 36 evaluations, so they share them through the
        per-configuration state instead of each paying for their own.
        """
        Ops = np.empty((len(R), self.nk, self.nb + 1), dtype=complex)
        for j in range(len(R)):
            Ops[j] = self.ll.padded_orbitals(R[j])
        self._bR = np.array(R, dtype=float, copy=True)
        self._bOps = Ops
        return Ops

    def batch_for(self, R):
        """``batch`` with a one-configuration memo, so repeat calls are free."""
        if (self._bR is not None and self._bR.shape == np.shape(R)
                and np.array_equal(self._bR, R)):
            return self._bOps
        return self.batch(R)

    @staticmethod
    def _kin(Op, nb):
        """(pi_x, pi_y, pi^2) applied to the nb bands, from the padded block.

        Transcribed from calc_O_kin_b() in slaterdet_orbitalrotation.jl: the
        magnetic ladder operators act on the BAND index, so no coordinate
        derivatives are taken --

            a|n> = sqrt(n)|n-1>,  a^dag|n> = sqrt(n+1)|n+1>,  pi^2|n> = (2n+1)|n>.

        This is the whole reason the Landau-level basis is the right one to mix
        in.  pi^2 is DIAGONAL in n, so a rotated orbital is an exact eigenvector
        relation term by term: no numerical derivative of the orbital is ever
        taken, and mixing levels costs no kinetic-energy accuracy at all -- the
        kinetic energy of the mixture is the mixture of the exact kinetic
        energies, weighted by |C|^2.  A Gaussian basis has no such property; its
        pi^2 has to be differenced out of the orbital, which is where a
        Gaussian-specific formula stops being valid for a general determinant.

        Leading axes broadcast, so this accepts (nk, nb+1) for one point or
        (ne, nk, nb+1) for a whole configuration.
        """
        O = Op[..., :nb]
        a = np.zeros(O.shape, dtype=complex)
        ad = np.zeros(O.shape, dtype=complex)
        ad[..., 0] += Op[..., 1]
        for n in range(1, nb):
            ad[..., n] += np.sqrt(n + 1) * Op[..., n + 1]
            a[..., n] += np.sqrt(n) * Op[..., n - 1]
        pix = (a + ad) / np.sqrt(2)
        piy = (a - ad) / (1j * np.sqrt(2))
        pi2 = O * (2 * np.arange(nb) + 1)
        return pix, piy, pi2

    # -- public API ----------------------------------------------------------
    def orbitals(self, r):
        """(nk,) : psi_k(r)."""
        return np.einsum("kn,kn->k", self.C, self._Op(r)[:, :self.nb])

    def pi(self, r):
        """(2, nk) : (pi_x psi_k, pi_y psi_k)."""
        pix, piy, _ = self._kin(self._Op(r), self.nb)
        return np.stack([np.einsum("kn,kn->k", self.C, pix),
                         np.einsum("kn,kn->k", self.C, piy)], axis=0)

    def pi2(self, r):
        """(nk,) : pi^2 psi_k."""
        _, _, pi2 = self._kin(self._Op(r), self.nb)
        return np.einsum("kn,kn->k", self.C, pi2)


class LLRotationWavefunction(Wavefunction):
    """Slater-Jastrow wavefunction whose determinant rows are rotated LL orbitals.

    B3 carried the state machinery -- ``build``, ``rebuild`` and the shared batch
    of Bloch sums -- because the sampler needs it.  ``pi_columns`` and
    ``local_energy`` arrived in B5, and ``orbital_log_deriv``/``log_deriv``/
    ``v_from_theta`` in B7.  The legacy class is one object, and it is split here
    only along the seams the staging already draws; with B7 done the split is
    complete and the class is symbol-for-symbol the legacy one.
    """

    def __init__(self, orb, jastrow, ne, sc_to_cart, kappa=0.0, ham=None):
        super().__init__(orb.orbitals, orb.pi, orb.pi2, jastrow, ne, sc_to_cart,
                         kappa=kappa, ham=ham)
        self.orb = orb
        self.n_band = orb.nb
        self.nk = orb.nk
        self.n_jastrow = 0 if jastrow is None else len(jastrow.c)

    # -- state, sharing one batch of Bloch sums ------------------------------
    def _D_from_ops(self, Ops):
        """D[k, b] = psi_k(r_b) = sum_n C[k,n] phi_{k,n}(r_b)."""
        return np.einsum("kn,bkn->kb", self.orb.C, Ops[:, :, :self.n_band])

    def build(self, R):
        R = np.asarray(R, dtype=float)
        Ops = self.orb.batch(R)
        st = {"R": R.copy(), "Ops": Ops, "D": self._D_from_ops(Ops),
              "u": None, "gx": None, "gy": None, "lap": None, "U": 0.0}
        st["D_inv"] = np.linalg.inv(st["D"])
        self._set_u(st)
        return st

    def rebuild(self, st):
        Ops = self.orb.batch(st["R"])
        st["Ops"] = Ops
        st["D"] = self._D_from_ops(Ops)
        st["D_inv"] = np.linalg.inv(st["D"])
        self._set_u(st)

    def _ops_of(self, st):
        """The batch belonging to st["R"], recomputed only if the state moved."""
        R = st["R"]
        Ops = st.get("Ops")
        if Ops is None or self.orb._bR is None or not np.array_equal(R, self.orb._bR):
            Ops = self.orb.batch_for(R)
            st["Ops"] = Ops
        return Ops

    # -- local energy --------------------------------------------------------
    def pi_columns(self, st):
        """(pix, piy, pi2), each (ne, ne): [a, j] is pi^2 psi_a at r_j.

        Reuses the batch ``build`` already paid for.  This is also where a
        Gaussian-specific formula would quietly stop being valid: the rotated
        orbital is a SUM over bands, so the kinetic operator has to act on the
        sum, and the cross terms between bands are carried by C rather than
        dropped.  Applying pi^2 band-by-band first and contracting with C
        afterwards is the same thing, and is what makes the operation cheap.

        The argument is the STATE, not a configuration -- unlike the base
        ``Wavefunction.pi_columns(R)``.  That difference is the legacy
        interface and is preserved on purpose (BUG_CANDIDATE.md #1); the batch
        is what makes it worth taking the state, and recomputing it here would
        throw away the work ``build`` already did.
        """
        Ops = self._ops_of(st)
        pixb, piyb, pi2b = LLRotatedOrbitals._kin(Ops, self.n_band)
        C = self.orb.C
        return (np.einsum("kn,bkn->bk", C, pixb).T,
                np.einsum("kn,bkn->bk", C, piyb).T,
                np.einsum("kn,bkn->bk", C, pi2b).T)

    def _pi_columns_for(self, st):
        """The base's hook, answered with the state-addressed ``pi_columns``.

        The rotated orbital is a SUM over bands, so applying the kinetic operator
        needs the batch ``build`` already paid for -- hence the state rather than
        the configuration (``BUG_CANDIDATE.md`` #1).  ``local_energy`` itself is
        the base class's, so ``T + kappa*V`` is written once for both ansatz.
        """
        return self.pi_columns(st)

    # -- logarithmic derivative ---------------------------------------------
    def orbital_log_deriv(self, st):
        """(da, db) : d ln det D / d a[k,j], d b[k,j], each (nk, NB-1).

        Uses the analytic identity d ln det M / d theta = Tr[ M^-1 dM/dtheta ].
        For the row of D that carries momentum k,

            d D[k, :] / d theta = ( dC^k / d theta ) . Phi_k ,

        so the trace collapses onto

            d ln det D / d theta = sum_k ( dC^k / d theta ) . W_k ,
            W_k = sum_j D_inv[j, k] Phi_k(r_j)          (a vector of length NB).

        The permutation (row k <-> momentum k) is exact at nu = 1, where the
        number of occupied orbitals equals the number of k points.

        Legacy source: ``qhvmc_engine_llrot.py:321-343``, body unchanged.  Note
        that ``Phi`` is the UNROTATED basis -- ``dC/dtheta`` carries the whole
        parameter dependence -- which is why this is ``basis_at``'s batch and not
        the rotated orbitals.
        """
        D_inv = st["D_inv"]
        Phi = self._ops_of(st)[:, :, :self.n_band]      # (ne, nk, nb)
        W = np.einsum("jk,jkn->kn", D_inv, Phi)
        dC_da, dC_db = dc_row_dparams(self.orb.v)
        da = np.einsum("kjn,kn->kj", dC_da, W)
        db = np.einsum("kjn,kn->kj", dC_db, W)
        return da, db

    def log_deriv(self, st):
        """d ln Psi / d theta, concatenated to match ``v_from_theta``.

        Order is [ Jastrow(c) , Re(v).ravel() , Im(v).ravel() ] -- the same order
        ``sr_optimize_joint`` and the notebooks' ``make_ll_theta`` assemble, and
        the reverse of ``v_from_theta``.

        Legacy source: ``qhvmc_engine_llrot.py:345-353``, body unchanged.  The
        legacy docstring says "``theta_from``/``v_from_theta``"; ``theta_from``
        does not exist in the frozen engine (nor anywhere else) -- it is a stale
        reference inherited from the original, noted in BUG_CANDIDATE.md under
        "Checked and NOT a bug".  Only the name has been dropped here.

        The result is complex (the orbital block is), while the Jastrow block is
        real; ``sr_optimize_joint`` takes ``.real`` of S and f, which is what
        makes the two commensurate.
        """
        parts = []
        if self.jastrow is not None:
            parts.append(jastrow_log_deriv(self.jastrow, st["R"]))
        da, db = self.orbital_log_deriv(st)
        parts.append(da.ravel())
        parts.append(db.ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    # -- parameter-vector plumbing ------------------------------------------
    def v_from_theta(self, theta):
        """The inverse of ``log_deriv``'s ordering for the orbital block.

        Legacy source: ``qhvmc_engine_llrot.py:356-361``, body unchanged.
        """
        nj = self.n_jastrow
        m = (self.n_band - 1) * self.nk
        a = np.asarray(theta[nj:nj + m], float).reshape(self.nk, self.n_band - 1)
        b = np.asarray(theta[nj + m:nj + 2 * m], float).reshape(self.nk, self.n_band - 1)
        return a + 1j * b
