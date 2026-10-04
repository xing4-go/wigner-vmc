"""Stochastic reconfiguration: the optimiser, and the seed it starts from.

Physical role:
    SR minimises ``<E>`` by stepping the variational parameters along the
    direction that the *wavefunction* moves least for a given change in the
    energy -- the natural gradient, ``delta = -tau S^-1 f``, with ``S`` the
    quantum geometric tensor (Fubini-Study metric) of the logarithmic
    derivatives and ``f`` the energy gradient.  It is what turns a guess into
    the optimised state every quoted energy is measured on.

    Two drivers, because the two workflows parametrise differently:

    ``sr_optimize_jastrow`` -- Gaussian workflow.  Parameters are the ``NJ``
        Jastrow coefficients ``c``.  Real ``S``, real ``f``.
    ``sr_optimize_joint``   -- LL-rotation workflow.  Parameters are
        ``theta = [c, a, b]`` with ``v = a + i b`` the orbital rotation.  The
        orbital block of the derivatives is COMPLEX, so ``S`` and ``f`` are
        built complex and their real parts are taken -- that is the reference's
        convention, and what makes the two blocks commensurate.

Main mathematical object:
    ``S_ij = <D_i* D_j> - <D_i*><D_j>``, ``f_i = <D_i* H> - <D_i*><H>``, with
    ``D = d ln Psi / d theta`` from ``Wavefunction.log_deriv`` (LL) or
    ``jastrow_log_deriv`` (Gaussian).  Regularised ``S += eps I``, learned at a
    decaying rate ``tau_i = tau / (1 + xi i / steps)``, with a momentum term
    ``eps mu / tau_i`` times the previous step.

Units/convention:
    ``tau``, ``xi``, ``eps``, ``mu`` are dimensionless here and their legacy
    defaults are kept exactly (``2e-2``, ``2.0``, ``1e-3``, ``0.0``).  ``E`` in
    the history is a **total**, not per electron -- see ``vmc.measure`` for why
    that matters.

Legacy source:
    ``qhvmc_engine.py:959-1025`` (``jastrow_log_deriv``, ``sr_optimize_jastrow``)
    and ``qhvmc_engine_llrot.py:369-562`` (``drummond_width``,
    ``gaussian_overlap_seed``, ``v_from_overlap``, ``sr_optimize_joint``,
    ``sr_pilot_scale``, ``jastrow_vector``).  All bodies are unchanged; the
    adaptations are recorded on each function.

Stage 2B / B7 -- mechanical migration.

WHAT "MECHANICAL" HAS TO MEAN HERE
----------------------------------
The update rule is four lines and every one of them is load-bearing, so none of
them was rewritten, reordered or algebraically simplified.  In particular:

* the parameter ordering ``[c, Re v, Im v]`` is the legacy ordering, and the LL
  notebook's ``make_ll_theta`` produces the same; it is not reorganised because a
  cleaner layout would be nicer.
* ``S`` and ``f`` are assembled with the same ``(D.conj().T @ D)/n - outer(...)``
  form as legacy, not with a numerically nicer equivalent.
* ``eps`` is added to the diagonal of the possibly-RESCALED metric, in the same
  place in the expression, because where it is added is exactly the difference
  between ``sr_optimize_joint`` and ``sr_pilot_scale``.
* the defaults, the ``verbose`` printing, the history keys and the RNG
  consumption order (one walk per step, restarted from the last configuration)
  are the legacy ones.
* the observed basin sensitivity of SR is NOT fixed here.  That is a physics
  question (II.12b) and an optimiser question, and B7 is neither.

``vmc.sr`` imports the legacy engine nowhere, and the frozen engine imports
nothing from here.
"""
from __future__ import annotations

import numpy as np

from ..wavefunctions.jastrow import jastrow_log_deriv
from .sampler import sample

__all__ = ["jastrow_log_deriv", "jastrow_vector", "sr_optimize_jastrow",
           "drummond_width", "gaussian_overlap_seed", "v_from_overlap",
           "sr_optimize_joint", "sr_pilot_scale"]


# ----------------------------------------------------------------------------
# The Gaussian-workflow driver: SR over the Jastrow alone
# ----------------------------------------------------------------------------


def sr_optimize_jastrow(make_wf, c0, R0, steps=30, nsweep=300, sigma=0.4, seed=0,
                        tau=2e-2, xi=2.0, eps=1e-3, mu=0.0, equil=None,
                        snapshot_every=2, target_acc=0.5, verbose=True):
    """Minimise <E> by stochastic reconfiguration over the Jastrow parameters.

    Mirrors stochasticreconfig!() in optimization.jl:

        S_ij = <D_i* D_j> - <D_i*><D_j>          (quantum geometric tensor, Re)
        f_i  = <D_i* H>  - <D_i*><H>
        S   += eps * I                            (regularisation)
        tau_i = tau / (1 + xi*i/steps)            (decaying learning rate)
        delta = -tau_i * S^-1 (f - (eps*mu/tau_i) * delta_prev)
        c    += delta

    ``make_wf(c)`` must return a Wavefunction whose Jastrow is built from c, so the
    same routine optimises either state.  The walk is restarted from the last
    configuration each step (the reference keeps one walker and does the same).

    Returns (c, history) with history a list of dicts per step.

    Legacy source: ``qhvmc_engine.py:974-1025``, body unchanged.  ``jastrow_log_deriv``
    now comes from ``wavefunctions.jastrow`` (re-exported above) rather than from
    this module, which is a move and not an edit.
    """
    rng = np.random.default_rng(seed)
    c = np.array(c0, dtype=float)
    R = np.array(R0, dtype=float)
    delta = np.zeros_like(c)
    hist = []
    for i in range(steps):
        wf = make_wf(c)
        snaps, sigma, acc = sample(wf, R, nsweep=nsweep, sigma=sigma, rng=rng,
                                   snapshot_every=snapshot_every, equil=equil,
                                   target_acc=target_acc)
        R = snaps[-1].copy()
        E = np.empty(len(snaps))
        D = np.empty((len(snaps), len(c)))
        for k, S in enumerate(snaps):
            E[k] = wf.local_energy(wf.build(S))[2].real
            D[k] = jastrow_log_deriv(wf.jastrow, S)
        ebar = E.mean()
        dbar = D.mean(axis=0)
        S_mat = (D.T @ D) / len(E) - np.outer(dbar, dbar)     # <DD> - <D><D>
        f = (D.T @ E) / len(E) - dbar * ebar                  # <DE> - <D><E>
        S_mat = S_mat + eps * np.eye(len(c))
        tau_i = tau / (1.0 + xi * i / steps)
        delta = -tau_i * np.linalg.solve(S_mat, f - (eps * mu / tau_i) * delta)
        c = c + delta
        hist.append(dict(step=i, E=ebar, E_err=E.std() / np.sqrt(len(E)),
                         acc=acc, sigma=sigma, c=c.copy(),
                         force=np.linalg.norm(f), tau=tau_i))
        if verbose:
            print(f"    SR {i:3d}  E/ne = {ebar / wf.ne:10.5f} +- "
                  f"{E.std() / np.sqrt(len(E)) / wf.ne:.5f}   acc {acc:.3f}   "
                  f"|f| {np.linalg.norm(f):.3e}   tau {tau_i:.2e}")
    return c, hist


# ----------------------------------------------------------------------------
# The LL-rotation-workflow driver: SR over (Jastrow, orbital rotation)
# ----------------------------------------------------------------------------


def drummond_width(rs, nu=1.0, lam=0.15):
    """The Drummond-2009 starting width used by the reference seed.

    C = lam * rs^(1/2) * nu / 2,  L0 = 0.5 / sqrt(C).

    Legacy source: ``qhvmc_engine_llrot.py:369-374``, body unchanged.
    """
    return 0.5 / np.sqrt(lam * rs**0.5 * nu / 2.0)


def gaussian_overlap_seed(ll_basis, n_band, L1, L2, l_vals, l_cart, rs, nu=1.0,
                          numx=101, L0=None):
    """Project a site-centred Gaussian onto the kept Landau levels, momentum by momentum.

    Returns ``overlap`` of shape (nk, n_band), ``overlap[k, n] = <phi_{k,n} | G>``
    up to a common normalisation (only the direction over n matters).

    Why this is the right seed, and why it is not a device
    -----------------------------------------------------
    The momentum label enters the magnetic Bloch orbital only through the shift
    r -> r - k x zhat.  Projecting the SAME site-centred Gaussian (at the origin)
    onto the levels kept at momentum k therefore gives a packet centred at
    k x zhat -- one localised orbital per lattice site, indexed by k.  So the seed
    is a Wigner crystal written in the Landau-level basis, and the optimisation
    that follows is free to leave it.  That is the point of the ansatz: the old
    Gaussian crystal is not a special case of it at any finite NB, it is a state
    this basis has to *approximate*, and this seed is the best approximation
    available at the start.

    Legacy source: ``qhvmc_engine_llrot.py:377-425``, body and signature
    unchanged -- including ``ll_basis.orbitals(...)``.  The only thing to know is
    what basis to hand it, because the two ansatz classes disagree about that:

    * HERE the two engines' ``LandauLevelBasis.orbitals(r)`` are the SAME
      accessor -- both return ``n + 1`` columns for constructor value ``n`` -- so
      the clean basis is built with the number the legacy caller passed as
      ``n_max``.
    * ``LLRotatedOrbitals`` is the one that differs: it needs a padding band, so
      its clean basis must be built ONE LOWER (``nmax = n_band - 1``, i.e.
      through ``nmax_from_n_bands``) to match the legacy ``n_max = n_band``.
      That is the §2A.5 convention trap, and it is local to that class.

    The ``[:, :n_band]`` slice keeps working when a caller passes a basis wider
    than it asks for -- which the LL notebook does, seeding an ``NMAX_STAGE_B``
    state from an ``LL_NMAX_BASIS`` basis.
    """
    L1 = np.asarray(L1, float)
    L2 = np.asarray(L2, float)
    nk = ll_basis.nk

    f = (np.arange(numx) + 0.5) / numx
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    pts = (F1.ravel()[:, None] * L1[None, :] + F2.ravel()[:, None] * L2[None, :]
           - (L1 + L2) / 2.0)
    npts = len(pts)

    # -- the site-centred magnetic-Bloch Gaussian on the same grid ----------
    if L0 is None:
        L0 = drummond_width(rs, nu)
    l_vals = np.asarray(l_vals, float)
    l_cart = np.asarray(l_cart, float)
    area = abs(L1[0] * L2[1] - L1[1] * L2[0])
    n_phi = area / (2 * np.pi)
    big = pts[:, None, :] + l_cart[None, :, :]                   # (npts, nL, 2)
    theta = ((pts[:, 0:1] * l_cart[None, :, 1] - pts[:, 1:2] * l_cart[None, :, 0]) / 2.0
             + l_vals[None, :, 0] * l_vals[None, :, 1] * n_phi * np.pi)
    g = np.exp(-np.sum(big * big, axis=-1) / L0**2 / 4.0) / np.sqrt(2 * np.pi) / L0
    G = np.sum(g * np.exp(1j * theta), axis=1)                   # (npts,)

    # -- the kept Landau levels, all momenta, all grid points ---------------
    bloch = np.empty((npts, nk, n_band), dtype=complex)
    for i in range(npts):
        bloch[i] = ll_basis.orbitals(pts[i])[:, :n_band]
    bloch = bloch / np.sqrt((np.abs(bloch) ** 2).sum(axis=0))[None, :, :]
    ov = np.einsum("i,ikn->kn", G, bloch.conj())
    return ov / np.linalg.norm(ov, axis=1)[:, None]


def v_from_overlap(overlap):
    """Turn the projected states into the orbital-rotation parameter vector.

    Mirrors calc_D_params_from_overlap_k_nu1() in slaterdet_gaussian.jl: for each
    k, phase-fix so that the lowest-Landau-level amplitude is real and
    non-negative, write it as cos t, and put the remaining amplitude into v with
    |v| = t.  That is the exact inverse of ``c_row``, so the seed reproduces the
    projected state itself and not merely something near it.

    Legacy source: ``qhvmc_engine_llrot.py:428-449``, body unchanged.
    """
    ov = np.asarray(overlap, dtype=complex)
    nk, nb = ov.shape
    v = np.zeros((nk, nb - 1), dtype=complex)
    for k in range(nk):
        psi = ov[k] / np.linalg.norm(ov[k])
        psi = psi * np.exp(-1j * np.angle(psi[0]))
        a = psi[0].real
        b = psi[1:]
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-12:
            continue
        v[k] = (-1j * (np.arctan2(b_norm, a) / b_norm)) * b
    return v


def sr_optimize_joint(make_wf, theta0, R0, steps=8, nsweep=150, sigma=0.4,
                      seed=0, tau=2e-2, xi=2.0, eps=1e-3, mu=0.0, equil=None,
                      snapshot_every=3, target_acc=0.5, verbose=False,
                      scale=None):
    """SR over the concatenated parameter vector theta = [c (Jastrow), a, b].

    Same update as ``sr_optimize_jastrow`` in the baseline engine and as
    ``stochasticreconfig!`` in optimization.jl:

        S_ij = Re[ <D_i* D_j> - <D_i*><D_j> ]        (Fubini-Study metric)
        f_i  = Re[ <D_i* H>  - <D_i*><H>  ]
        S   += eps I
        tau_i = tau / (1 + xi i / steps)
        delta = -tau_i S^-1 ( f - (eps mu / tau_i) delta_prev )
        theta += delta

    The orbital part of D is COMPLEX while the Jastrow part is real; taking the
    real part of S and f is what makes the two commensurate, and is the
    convention the reference uses (stochasticreconfig! takes real.(S), real.(f)).

    ``scale`` optionally gives a per-parameter natural size.  The update is then
    solved in the rescaled variables x = theta/scale, which leaves the SR
    direction untouched but changes how the absolute regulariser eps is applied:
    eps is measured against each parameter's own scale instead of against raw
    units.  Its eigenvalues and condition number are reported per step, because
    "SR is unstable" is almost always a statement about those two numbers.

    Legacy source: ``qhvmc_engine_llrot.py:457-528``, body unchanged.
    """
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=float)
    R = np.array(R0, dtype=float)
    npar = len(theta)
    if scale is None:
        scale = np.ones(npar)
    scale = np.asarray(scale, dtype=float)
    delta_x = np.zeros(npar)
    hist = []

    for i in range(steps):
        wf = make_wf(theta)
        snaps, sig, acc = sample(wf, R, nsweep=nsweep, sigma=sigma, rng=rng,
                                 snapshot_every=snapshot_every, equil=equil,
                                 target_acc=target_acc)
        R = snaps[-1].copy()
        n = len(snaps)
        E = np.empty(n)
        D = np.empty((n, npar), dtype=complex)
        for s, S in enumerate(snaps):
            st = wf.build(S)
            E[s] = wf.local_energy(st)[2].real
            D[s] = wf.log_deriv(st)
        ebar = E.mean()
        dbar = D.mean(axis=0)
        S_theta = (D.conj().T @ D) / n - np.outer(dbar.conj(), dbar)
        f_theta = (D.conj().T @ E) / n - dbar.conj() * ebar

        # to the rescaled variables x (theta = scale * x): S_x = L S L, f_x = L f
        S_x = (scale[:, None] * scale[None, :]) * S_theta.real
        f_x = scale * f_theta.real
        eig = np.linalg.eigvalsh(S_x)
        S_reg = S_x + eps * np.eye(npar)
        tau_i = tau / (1.0 + xi * i / steps)
        delta_x = -tau_i * np.linalg.solve(S_reg, f_x - (eps * mu / tau_i) * delta_x)
        theta = theta + scale * delta_x
        hist.append(dict(step=i, E=ebar, E_err=E.std() / np.sqrt(n), acc=acc,
                         sigma=sig, theta=theta.copy(), force=np.linalg.norm(f_x),
                         tau=tau_i, cond=float(np.linalg.cond(S_reg)),
                         eig_min=float(eig.min()), eig_max=float(eig.max())))
        if verbose:
            print(f"    SR {i:3d}  E/ne = {ebar / wf.ne:10.5f} +- "
                  f"{E.std() / np.sqrt(n) / wf.ne:.5f}   acc {acc:.3f}   "
                  f"|f| {np.linalg.norm(f_x):.3e}  cond {np.linalg.cond(S_reg):.2e}")
    return theta, hist


def sr_pilot_scale(make_wf, theta0, R0, nsweep=150, seed=3, sigma=0.3, equil=75,
                   snapshot_every=1, target_acc=0.4, floor=1e-14):
    """Per-parameter natural scale for SR, from a pilot sample: scale_i = 1/sqrt(S_ii).

    ``S_ii`` is the variance of ``d ln Psi / d theta_i``.  Solving in ``theta = scale*x``
    gives ``S_x,ii = scale_i^2 S_ii = 1`` for every ``i``, so the absolute regulariser
    ``eps`` is applied on the same footing to the Jastrow block (derivatives O(10)) and
    to the orbital block (O(1e-3)).

    Why this exists
    ---------------
    ``sr_optimize_joint``'s ``eps`` is an absolute number added to the metric's diagonal.
    ``d ln Psi/d c`` for the Jastrow is O(10)-O(40) while ``d ln Psi/d v`` for the orbital
    block is O(1e-3), so their squared variances differ by ~5 orders of magnitude.  With
    ``scale = 1`` and ``eps = 1e-3`` the regulariser can exceed the entire orbital signal,
    which turns the orbital update into a heavily damped gradient step.  This is a
    *hypothesis about why the SR wanders*, so it is tested by measurement rather than
    assumed -- see II.12b.

    Legacy source: ``qhvmc_engine_llrot.py:531-557``, body unchanged.
    """
    wf = make_wf(theta0)
    snaps, sig, acc = sample(wf, np.array(R0, dtype=float), nsweep=nsweep, sigma=sigma,
                             rng=np.random.default_rng(seed),
                             snapshot_every=snapshot_every, equil=equil,
                             target_acc=target_acc)
    D = np.array([wf.log_deriv(wf.build(S)) for S in snaps])
    Sii = (np.abs(D) ** 2).mean(axis=0) - np.abs(D.mean(axis=0)) ** 2
    return 1.0 / np.sqrt(np.maximum(Sii, floor))


def jastrow_vector(kappa, depth=5):
    """The reference's Jastrow starting point: -kappa/6 * [(NJ-i)/(NJ-1)].

    Legacy source: ``qhvmc_engine_llrot.py:560-562``, body unchanged.
    """
    return -kappa / 6 * np.array([(depth - i) / (depth - 1) for i in range(1, depth + 1)])
