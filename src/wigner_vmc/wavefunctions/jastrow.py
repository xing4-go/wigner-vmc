"""The sin-spline Jastrow factor.

Physical role:
    The two-body correlating factor ``exp(sum_{a<b} u(r_a - r_b))`` of the
    Slater-Jastrow wavefunction.  It is what makes the pair distribution respect
    the Kato cusp and what lowers the variational energy below the determinant's.

Main mathematical object:
    ``u(r) = sum_m pm[m] * b3(f(r)*M - m)`` with ``b3`` the cubic B-spline,
    ``f(r) = (2/3) sqrt(sum over the three shortest reciprocal vectors of
    sin^2(G.r/2))`` the triangular-lattice "distance", and the cusp constraint
    ``pm[0] = c[1] - 3*gamma/M`` folded into the parameter list.

Units/convention:
    ``l_B = 1``.  ``G1, G2`` are the SUPERCELL reciprocal vectors for the
    Gaussian workflow and the primitive-cell ones for the LL-rotation workflow;
    the caller decides which, exactly as the frozen engine does.  ``gamma`` is
    the Kato cusp constant, ``|L1|/sqrt(2)/pi * kappa/3``.

Legacy source:
    ``qhvmc_engine.py:469-619`` -- ``b3``, ``dx_b3``, ``d2x_b3``, ``_calc_f``,
    ``_calc_grad_f``, ``_calc_laplacian_f``, ``SinSplineJastrow``.
    ``cusp_gamma`` is at ``make_notebook.py:930`` (a generator-level helper, not
    in the engine); it is included here because it defines ``gamma`` and every
    production Jastrow was built through it.

Stage 2B / B1 -- mechanical migration.  The formulas, the parameter order, the
defaults and the numerical semantics are the frozen engine's, unchanged.  The
only signature change is ``cusp_gamma(kappa) -> cusp_gamma(kappa, L1)``: the
legacy version read a module global ``L1``, which a library module cannot have.

Stage 2B / B7 adds ``jastrow_log_deriv``.  The legacy engine keeps it in the
*optimization* section of ``qhvmc_engine.py`` (line 959), three lines above
``sr_optimize_jastrow``; it lives here instead because it is a property of the
Jastrow alone -- two lines around ``grad_params`` -- and because the LL-rotation
wavefunction needs it for ``log_deriv`` while the SR module needs it for the
metric.  Putting it in the SR module would make ``wavefunctions`` import from
``vmc``, which already imports ``wavefunctions``.  ``vmc.sr`` re-exports it, so
every legacy call site keeps its name.
"""
from __future__ import annotations

import numpy as np

__all__ = ["b3", "dx_b3", "d2x_b3", "cusp_gamma", "SinSplineJastrow",
           "jastrow_log_deriv"]


def b3(x):
    """Cubic B-spline, mirroring calc_b3()."""
    ax = np.abs(x + 1.0)
    out = np.zeros_like(ax)
    m1 = ax < 1.0
    m2 = (ax >= 1.0) & (ax < 2.0)
    out[m1] = (4 - 6 * ax[m1] ** 2 + 3 * ax[m1] ** 3) / 6.0
    out[m2] = (2 - ax[m2]) ** 3 / 6.0
    return out


def dx_b3(x):
    ax = np.abs(x + 1.0)
    sx = np.sign(x + 1.0)
    out = np.zeros_like(ax)
    m1 = (ax > 0) & (ax < 1.0)
    m2 = (ax >= 1.0) & (ax < 2.0)
    out[m1] = sx[m1] * (-12 * ax[m1] + 9 * ax[m1] ** 2) / 6.0
    out[m2] = -0.5 * sx[m2] * (2 - ax[m2]) ** 2
    return out


def d2x_b3(x):
    ax = np.abs(x + 1.0)
    out = np.zeros_like(ax)
    m0 = ax == 0
    m1 = (ax > 0) & (ax < 1.0)
    m2 = (ax >= 1.0) & (ax < 2.0)
    out[m0] = -2.0
    out[m1] = (18 * ax[m1] - 12) / 6.0
    out[m2] = 2 - ax[m2]
    return out


def cusp_gamma(kappa, L1):
    """Kato cusp constant, ``gamma = |L1|/(sqrt(2) pi) * kappa/3``.

    Legacy ``make_notebook.py:930`` read ``L1`` as a module global; here it is an
    argument.  The formula is unchanged.
    """
    return np.linalg.norm(L1) / np.sqrt(2) / np.pi * kappa / 3


def _calc_f(r, G1, G2):
    """f(r) = (2/3) sqrt( sin^2(r.G1/2) + sin^2(r.G2/2) + sin^2(r.(G1+G2)/2) )"""
    r = np.asarray(r, dtype=float)
    t1 = np.sin((r[..., 0] * G1[0] + r[..., 1] * G1[1]) / 2.0) ** 2
    t2 = np.sin((r[..., 0] * G2[0] + r[..., 1] * G2[1]) / 2.0) ** 2
    t3 = np.sin((r[..., 0] * (G1[0] + G2[0]) + r[..., 1] * (G1[1] + G2[1])) / 2.0) ** 2
    return (2.0 / 3.0) * np.sqrt(t1 + t2 + t3)


def _calc_grad_f(r, G1, G2, f):
    num = (
        np.sin(r[..., 0] * G1[0] + r[..., 1] * G1[1])[..., None] * G1[None, :]
        + np.sin(r[..., 0] * G2[0] + r[..., 1] * G2[1])[..., None] * G2[None, :]
        + np.sin(
            (r[..., 0] * (G1[0] + G2[0]) + r[..., 1] * (G1[1] + G2[1]))
        )[..., None]
        * (G1 + G2)[None, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return (2.0 / 3.0) ** 2 * num / (4.0 * f[..., None])


def _calc_laplacian_f(r, G1, G2, f, grad_f):
    d1 = np.cos(r[..., 0] * G1[0] + r[..., 1] * G1[1])
    d2 = np.cos(r[..., 0] * G2[0] + r[..., 1] * G2[1])
    d3 = np.cos(r[..., 0] * (G1[0] + G2[0]) + r[..., 1] * (G1[1] + G2[1]))
    num = (
        d1 * (G1 @ G1)
        + d2 * (G2 @ G2)
        + d3 * ((G1 + G2) @ (G1 + G2))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((2.0 / 3.0) ** 2 * num / 4.0 - np.sum(grad_f * grad_f, axis=-1)) / f


class SinSplineJastrow:
    """u(r) = sum_m pm[m] b3(f(r)*M - m),  pm[0] = c[1] - 3 gamma/M.

    Mirrors calc_u_sin_spl, calc_grad_u_sin_spl, calc_laplacian_u_sin_spl and
    calc_grad_params_u_sin_spl.
    """

    def __init__(self, c, G1, G2, gamma):
        self.c = np.asarray(c, dtype=float)
        self.M = len(self.c)
        self.G1 = np.asarray(G1, dtype=float)
        self.G2 = np.asarray(G2, dtype=float)
        self.gamma = float(gamma)
        self.pm = np.concatenate([[self.c[1] - 3 * self.gamma / self.M], self.c])

    def _f(self, r):
        return _calc_f(r, self.G1, self.G2)

    @staticmethod
    def _prep(r):
        """Accept (2,) or (...,2); report whether the input was a single point."""
        r = np.asarray(r, dtype=float)
        single = r.ndim == 1
        return np.atleast_2d(r), single

    def u(self, r):
        r, single = self._prep(r)
        f = self._f(r)
        x = f[..., None] * self.M - np.arange(self.M + 1)[None, :]
        out = np.sum(self.pm[None, :] * b3(x), axis=-1)
        return out[0] if single else out

    def grad(self, r):
        r, single = self._prep(r)
        f = self._f(r)
        gf = _calc_grad_f(r, self.G1, self.G2, f)
        x = f[..., None] * self.M - np.arange(self.M + 1)[None, :]
        ds = self.M * np.sum(self.pm[None, :] * dx_b3(x), axis=-1)
        out = gf * ds[..., None]
        return out[0] if single else out

    def laplacian(self, r):
        r, single = self._prep(r)
        f = self._f(r)
        gf = _calc_grad_f(r, self.G1, self.G2, f)
        lf = _calc_laplacian_f(r, self.G1, self.G2, f, gf)
        x = f[..., None] * self.M - np.arange(self.M + 1)[None, :]
        d1 = self.M * np.sum(self.pm[None, :] * dx_b3(x), axis=-1)
        d2 = self.M**2 * np.sum(self.pm[None, :] * d2x_b3(x), axis=-1)
        out = d2 * np.sum(gf * gf, axis=-1) + d1 * lf
        return out[0] if single else out

    def grad_params(self, r):
        """d u / d c_k, shape (..., M).  Entries b3(fM-1), ..., b3(fM-M),
        with entry 1 (m=2) picking up the extra b3(fM) from the cusp constraint
        pm[0] = c[1] - 3 gamma/M."""
        r, single = self._prep(r)
        f = self._f(r)
        m = np.arange(1, self.M + 1)
        x = f[..., None] * self.M - m[None, :]
        g = b3(x)
        g[..., 1] += b3(f * self.M)
        return g[0] if single else g

    def u_pairs(self, R):
        """All pairwise u(r_i - r_j) in one vectorized call, zero diagonal.

        Returns (u_mat, gx_mat, gy_mat, lap_mat), each (ne, ne), with the
        diagonal zero and row i holding u/grad/lap of r_i - r_j.
        """
        R = np.atleast_2d(np.asarray(R, dtype=float))
        ne = len(R)
        d = R[:, None, :] - R[None, :, :]              # d[i,j] = r_i - r_j
        flat = d.reshape(-1, 2)
        u = self.u(flat).reshape(ne, ne)
        g = self.grad(flat).reshape(ne, ne, 2)
        lap = self.laplacian(flat).reshape(ne, ne)
        mask = ~np.eye(ne, dtype=bool)
        return (np.where(mask, u, 0.0),
                np.where(mask, g[:, :, 0], 0.0),
                np.where(mask, g[:, :, 1], 0.0),
                np.where(mask, lap, 0.0))


def jastrow_log_deriv(jastrow, R):
    """Delta_k(R) = d ln|Psi| / d c_k  for the SinSplineJastrow parameters.

    ln|Psi| contains sum_{a<b} u(r_ab) and the determinant does not depend on c,
    so Delta_k = sum_{a<b} du(r_ab)/dc_k = (1/2) sum_{a != b} du(r_ab)/dc_k --
    the same factor of 1/2 that `_set_u` uses for U.

    Legacy source: ``qhvmc_engine.py:959-971``, body unchanged.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    ne = len(R)
    d = (R[:, None, :] - R[None, :, :]).reshape(-1, 2)
    gp = jastrow.grad_params(d).reshape(ne, ne, -1)
    mask = ~np.eye(ne, dtype=bool)
    return 0.5 * gp[mask].sum(axis=0)
