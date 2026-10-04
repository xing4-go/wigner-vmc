"""Magnetic-Bloch Gaussian orbitals on Wigner-crystal sites.

Physical role:
    The crystal orbital basis of the *Gaussian* workflow: one Gaussian orbital
    per Wigner-crystal site, made periodic by summing over supercell
    translations with the magnetic Bloch phase.  Feeding the resulting ``(ns,)``
    column into the Slater determinant gives the Gaussian crystal trial state.

Main mathematical object:
    ``psi_i(r) = sum_L g(r + R_i + L; L0) * exp(i theta_iL) * exp(i (r x R_i)_z/2)``
    with ``g`` a normalised Gaussian of width ``L0`` and ``theta`` the
    magnetic-Bloch phase of the supercell translation ``L`` (``n_phi = 36`` flux
    quanta, so that phase is identically 1 here -- but it is written, not
    dropped).  ``pi psi`` and ``pi^2 psi`` are assembled from the analytic
    single-Gaussian ladder actions, not by differentiating the orbital numerically.

Units/convention:
    ``l_B = 1``; gauge ``A = (-y, x)/2``, ``curl A = +B z``.  Care needed:
    ``pi_orbitals`` labels its returns ``pix = (a + a_dag)/sqrt2`` and
    ``piy = i (a - a_dag)/sqrt2``, whereas the Landau-level module uses
    ``piy = (a - a_dag)/(i sqrt2)``.  Those are the same expression; the two
    modules just spell the ``i`` on opposite sides of the fraction.

Legacy source:
    ``qhvmc_engine.py:300-432`` -- ``GaussianBasis``, verbatim: ``_g``,
    ``orbitals``, ``_pi_bare``, ``pi_orbitals``, ``pi_square_orbitals``,
    ``dpsi_dL0``.  The site generator ``gaussian_sites`` (line 435) is NOT
    repeated here; Stage 2A migrated it as
    ``physics.geometry.wigner_crystal_sites``.

Stage 2B / B2 -- mechanical migration.  No algorithm change, no API redesign,
no performance work.  The one thing worth stating because it is easy to "tidy"
into a bug: ``orbitals`` shifts by ``+r`` (``rr = sites + r``) where the
Landau-level basis shifts by ``-rk``.  That sign is the gauge, and it is pinned
by finite differences in the test file, not by the name of the variable.

Periodicity is only as good as the cutoff (measured during B2)
--------------------------------------------------------------
The Bloch sum runs over ``l_cart``, which `physics.geometry.circular_lattice`
fills out to ``LUMAX``.  At the production ``LUMAX = 30`` that is 13 vectors,
and the sum is truncated: a supercell translation is a relabelling ``L -> L+L1``
of the summation set, and ``{|L| <= 30}`` is not closed under it.

Two different numbers describe the residual, and they are easy to confuse:

* *Per orbital, relative to that same orbital.*  ``max_i | |psi_i(r+L1)|/|psi_i(r)| - 1 |``
  is ``1.0e+00`` -- the figure the generator prints at ``make_notebook.py:403``.
  It is dominated by sites whose orbital is exponentially small at the probe
  point, where the ratio is a quotient of two ~1e-10 amplitudes and carries no
  information.  **This is not the error in the wavefunction.**
* *In the determinant production actually builds* -- 36 electrons inside the
  supercell at ``L0 = 0.6`` -- ``max |psi|`` differs from the converged
  (``LUMAX = 90``) value by ``1.8e-10``, and ``log|det|`` by ``1.8e-11``.

So the production state is affected at the ``1e-10`` level, far below every
quoted energy error, while the generator's own diagnostic prints a number that
looks catastrophic.  Convergence to machine precision needs ``LUMAX >= 40`` for
the ``L1``/``L2`` shifts and ``>= 60`` for longer ones, so ``LUMAX > |L1|`` --
the condition the guard at ``make_notebook.py:400`` actually tests -- does not
express "untruncated".  See ``BUG_CANDIDATE.md`` #2.

This is the frozen engine's behaviour term for term (the two implementations
give the identical deviation at every cutoff), so it is an ansatz property, not
a migration defect.
"""
from __future__ import annotations

import numpy as np

__all__ = ["GaussianBasis"]


class GaussianBasis:
    """Gaussian orbitals on Wigner-crystal sites with magnetic BC.

    Mirrors gaussian_mbc(), calc_pi_gaussian_mbc(),
    calc_pi_square_gaussian_mbc(), calc_dl0_gaussian_mbc().

    Gauge: like the Landau-level module this uses A = (-y,x)/2 (curl A = +B z).
    Verified by finite difference in test_engine.py -- an earlier note claiming
    the two modules used opposite signs was wrong.
    """

    def __init__(self, sites, l_ints, l_cart, L1, L2, l=1.0):
        self.sites = np.asarray(sites, dtype=float)
        self.ns = len(self.sites)
        self.l_ints = np.asarray(l_ints, dtype=float)
        self.l_cart = np.asarray(l_cart, dtype=float)
        self.L1 = np.asarray(L1, dtype=float)
        self.L2 = np.asarray(L2, dtype=float)
        self.l = float(l)
        area = abs(self.L1[0] * self.L2[1] - self.L1[1] * self.L2[0])
        self.n_phi = area / (2 * np.pi * self.l**2)
        self.nL = len(self.l_cart)

    # -- bare Gaussian -------------------------------------------------------
    def _g(self, r, L0):
        return np.exp(-np.sum(r * r, axis=-1) / L0**2 / 4.0) / np.sqrt(2 * np.pi) / L0

    # -- assembled orbital ---------------------------------------------------
    def orbitals(self, r, L0):
        """psi_i(r) for all sites i. Returns (ns,) complex."""
        r = np.asarray(r, dtype=float).ravel()
        rr = self.sites + r[None, :]  # (ns,2)
        # r + R0 + L
        big = rr[:, None, :] + self.l_cart[None, :, :]  # (ns, nL, 2)
        theta = (
            (rr[:, 0:1] * self.l_cart[None, :, 1] - rr[:, 1:2] * self.l_cart[None, :, 0])
            / self.l**2
            / 2.0
            + self.l_ints[None, :, 0]
            * self.l_ints[None, :, 1]
            * self.n_phi
            * np.pi
        )
        psi = np.sum(self._g(big, L0) * np.exp(1j * theta), axis=1)
        # site-dependent phase
        psi = psi * np.exp(1j * (r[0] * self.sites[:, 1] - r[1] * self.sites[:, 0]) / 2.0)
        return psi

    # -- kinetic operators ---------------------------------------------------
    def _pi_bare(self, rr, L0):
        """a^dag g and a g evaluated on bare shifted Gaussians.

        Reference names these calc_pi_gaussian_m / calc_pi_dagger_gaussian_m;
        verified by finite differences in tests to satisfy
            calc_pi_gaussian_m       = a^dag g
            calc_pi_dagger_gaussian_m = a g
        with a = (pi_x + i pi_y)/sqrt(2), A = (y,-x)/2.
        """
        z = rr[:, 0] + 1j * rr[:, 1]
        zs = rr[:, 0] - 1j * rr[:, 1]
        g = self._g(rr, L0)
        ratio = (self.l / L0) ** 2
        a_dag_g = -1j * z * g * (1 - ratio) / 2.0 / np.sqrt(2) / self.l**2
        a_g = 1j * zs * g * (1 + ratio) / 2.0 / np.sqrt(2) / self.l**2
        return a_dag_g, a_g

    def pi_orbitals(self, r, L0):
        """(pi_x psi, pi_y psi) for all sites: returns (2, ns)."""
        r = np.asarray(r, dtype=float).ravel()
        rr = self.sites + r[None, :]
        big = rr[:, None, :] + self.l_cart[None, :, :]  # (ns,nL,2)
        theta = (
            (rr[:, 0:1] * self.l_cart[None, :, 1] - rr[:, 1:2] * self.l_cart[None, :, 0])
            / self.l**2
            / 2.0
            + self.l_ints[None, :, 0] * self.l_ints[None, :, 1] * self.n_phi * np.pi
        )
        ph = np.exp(1j * theta)
        flat = big.reshape(-1, 2)
        a_dag_g, a_g = self._pi_bare(flat, L0)
        a_dag_g = a_dag_g.reshape(self.ns, self.nL)
        a_g = a_g.reshape(self.ns, self.nL)
        # pi_x = (a + a^dag)/sqrt(2),  pi_y = i (a - a^dag)/sqrt(2)
        pix = np.sum((a_g + a_dag_g) * ph, axis=1) / np.sqrt(2)
        piy = np.sum(1j * (a_g - a_dag_g) * ph, axis=1) / np.sqrt(2)
        site_ph = np.exp(1j * (r[0] * self.sites[:, 1] - r[1] * self.sites[:, 0]) / 2.0)
        return np.stack([pix * site_ph, piy * site_ph], axis=0)

    def pi_square_orbitals(self, r, L0):
        """pi^2 psi for all sites. Returns (ns,)."""
        r = np.asarray(r, dtype=float).ravel()
        rr = self.sites + r[None, :]
        big = rr[:, None, :] + self.l_cart[None, :, :]
        theta = (
            (rr[:, 0:1] * self.l_cart[None, :, 1] - rr[:, 1:2] * self.l_cart[None, :, 0])
            / self.l**2
            / 2.0
            + self.l_ints[None, :, 0] * self.l_ints[None, :, 1] * self.n_phi * np.pi
        )
        ph = np.exp(1j * theta)
        flat = big.reshape(-1, 2)
        z = flat[:, 0] + 1j * flat[:, 1]
        zs = flat[:, 0] - 1j * flat[:, 1]
        g = self._g(flat, L0)
        # pi^dag pi g  (reference calc_pi_dagger_pi_gaussian_m)
        pdp_g = g * (z * zs / self.l**2 * (1 + (self.l / L0) ** 2) - 4) * (
            1 - (self.l / L0) ** 2
        ) / 8 / self.l**2
        val = (
            2 * pdp_g + g / self.l**2
        )  # pi^2 = 2 pi^dag pi + 1/l^2   (this module's sign of B)
        pi2 = np.sum(val.reshape(self.ns, self.nL) * ph, axis=1)
        site_ph = np.exp(1j * (r[0] * self.sites[:, 1] - r[1] * self.sites[:, 0]) / 2.0)
        return pi2 * site_ph

    def dpsi_dL0(self, r, L0):
        """d psi / d L0 for all sites. Returns (ns,)."""
        r = np.asarray(r, dtype=float).ravel()
        rr = self.sites + r[None, :]
        big = rr[:, None, :] + self.l_cart[None, :, :]
        theta = (
            (rr[:, 0:1] * self.l_cart[None, :, 1] - rr[:, 1:2] * self.l_cart[None, :, 0])
            / self.l**2
            / 2.0
            + self.l_ints[None, :, 0] * self.l_ints[None, :, 1] * self.n_phi * np.pi
        )
        ph = np.exp(1j * theta)
        rsq = np.sum(big * big, axis=-1)
        g = self._g(big, L0)
        dg = (rsq / L0**3 / 2.0 - 1.0 / L0) * g
        val = np.sum(dg * ph, axis=1)
        site_ph = np.exp(1j * (r[0] * self.sites[:, 1] - r[1] * self.sites[:, 0]) / 2.0)
        return val * site_ph
