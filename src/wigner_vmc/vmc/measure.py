"""Production measurement: the energy record a VMC run is quoted from.

Physical role:
    One Metropolis walk in, one energy record out -- the mean local energy per
    electron, its standard error, and the same two for the kinetic and potential
    halves separately.  This is the ``E_production`` of the project's three
    energy semantics (see below); every number the notebooks quote as "the
    energy at this coupling" comes from here and nowhere else.

Main mathematical object:
    ``E_loc = T + kappa*V`` per snapshot, averaged over snapshots.  ``T`` and
    ``V`` are kept as separate record fields rather than only their sum,
    because they are individually large and opposite in sign -- quoting the sum
    alone is how an error in one gets cancelled by an error in the other.

Units/convention:
    Every energy field is **per electron** (divided by ``ne``); the errors are
    divided by ``ne`` too.  ``V`` is the bare ``sum 1/r``, so its per-electron
    value is the thing that gets multiplied by ``kappa`` in ``E``.

Legacy source:
    ``make_notebook.py:707-719`` (``measure``) and ``make_notebook.py:1435-1443``
    (``sq_error_bars``), with ``sq_error_bars`` living in
    ``analysis/structure.py`` next to the rest of the ``S(q)`` family.  Both are
    **emitted cell text**, not engine symbols: the generator has exactly four
    module-level definitions (``_src``, ``_push``, ``md``, ``code``) and these
    are not among them.  So the source of truth for this migration is the
    notebook source, which is the only place the legacy implementation exists.
    ``measure`` is defined once and used by BOTH workflows -- the Gaussian
    notebook calls it directly and the LL-rotation notebook calls it through
    ``reproduction/make_notebook_llrot.py:1269`` -- so this one function is the
    shared production path, not a per-workflow copy.

Stage 2B / B6.

-------------------------------------------------------------------------------
THE THREE ENERGY SEMANTICS ARE NOT INTERCHANGEABLE
-------------------------------------------------------------------------------
This project quotes energy in three different senses, and two of them differ by
a factor of ``ne``.  They are recorded here once so that no later reader has to
rediscover it:

``E_optimization_min`` / ``E_optimization_final`` -- from the SR history,
    ``hist[i]["E"]``.  The engine appends ``E=ebar`` where ``ebar = E.mean()``
    over the snapshot energies (``qhvmc_engine.py:1016``,
    ``qhvmc_engine_llrot.py:520``).  That is a **total** energy, NOT per
    electron.  ``min`` and ``[-1]`` of the history are two different numbers and
    neither is the production energy: the first is the best point the
    optimiser passed through, the second is where it stopped, and the
    optimiser is free to end worse than its own minimum.

``E_production`` -- ``measure(...)["E"]`` from THIS module.  A **per-electron**
    energy measured on an independent walk.  The factor of ``ne`` between it and
    the SR history is the single most reachable error in this file.

A phase comparison is only allowed to use ``E_production``: comparing an
optimisation-time energy against a production-time one, or a total against a
per-electron value, produces a crossing that is an artefact of which number was
read, not of the physics.

``E_err`` here is the plain snapshot spread, ``std/sqrt(n)``.  It is NOT
autocorrelation-corrected and it is therefore a LOWER BOUND -- the snapshots are
MCMC-correlated.  The corrected estimates live in ``analysis.statistics``
(``tau_int``, ``blocked``), and ``blocked`` must be applied to the **total**
energy and compared against a naive **total** error: dividing a total-unit
blocked error by a per-electron naive error returns exactly ``ne`` times the
true correction.
"""
from __future__ import annotations

import numpy as np

from .sampler import sample

__all__ = ["measure", "measure_decomposed"]


def measure(wf, R0, nsweep, sigma=0.3, seed=0, snapshot_every=5, equil=None,
            ne=None, **kw):
    """Run a walk and return the mean local energy (per electron) and observables.

    Legacy source: ``make_notebook.py:707-719``, body unchanged.  The one
    adaptation is ``ne``: the cell closed over a module-level global ``ne = 36``,
    which a library module cannot have.  It now defaults to ``wf.ne``, which is
    the same object every call site in the notebooks was using -- asserted
    bit-for-bit in ``tests/test_measure_against_legacy.py`` rather than assumed.

    Keys: ``n``, ``acc``, ``sigma``, ``snaps``, and ``T``/``T_err``, ``V``/
    ``V_err``, ``E``/``E_err``.  Every energy is per electron.  ``E_err`` is the
    plain snapshot spread -- a lower bound, not the autocorrelation-corrected
    error (see the module docstring).

    ``**kw`` is forwarded to ``sample`` unchanged, which is how
    ``target_acc``/``reset_period`` reach the walk.
    """
    ne = wf.ne if ne is None else ne
    rng = np.random.default_rng(seed)
    snaps, sig, acc = sample(wf, R0, nsweep=nsweep, sigma=sigma, rng=rng,
                             snapshot_every=snapshot_every, equil=equil, **kw)
    res = np.array([wf.local_energy(wf.build(R)) for R in snaps])
    n = len(snaps)
    out = dict(n=n, acc=acc, sigma=sig, snaps=snaps,
               T=res[:, 0].real.mean() / ne, T_err=res[:, 0].real.std() / np.sqrt(n) / ne,
               V=res[:, 1].mean() / ne,    V_err=res[:, 1].std() / np.sqrt(n) / ne,
               E=res[:, 2].real.mean() / ne, E_err=res[:, 2].real.std() / np.sqrt(n) / ne)
    return out


def measure_decomposed(wf, R0, sweeps, equil, sigma, seed, label=None):
    """The production record the campaign quotes, with the correlation correction.

    Why this exists next to ``measure`` rather than inside it: ``measure`` returns
    ``E_err`` as the plain snapshot spread and says so -- it is a LOWER BOUND,
    because consecutive snapshots are MCMC-correlated.  The frozen campaign's
    published error bars are not that number; they are
    ``sigma_naive * sqrt(tau)``.  A caller that wants a legacy-comparable error
    bar must therefore keep the per-snapshot series, which ``measure`` reduces
    away.  This function keeps it.

    TWIN.  ``scripts/bench_rs75.py:production_walk`` computes the same record for
    the Stage 2E reproduction harness.  That harness produced the B6/B7/B8 PASS
    evidence and is deliberately NOT refactored to call this function -- the
    duplication is the price of not disturbing a verified path, and
    ``tests/test_api_contract.py`` asserts the two agree bit-for-bit on shared
    inputs so they cannot drift apart silently.

    Conventions, each of which has a trap behind it:

    * ``E = T + kappa*V`` is accumulated **per snapshot**, so the error bar is the
      error of the quantity of interest rather than the quadrature sum of two
      strongly anti-correlated halves.
    * ``local_energy`` returns ``T`` and ``V`` as TOTALS over electrons; every
      reported mean is divided by ``ne`` exactly once.  The ``*_total`` keys keep
      the undivided values so the two conventions cannot be confused downstream
      (the legacy JSON family stores TOTALS under the bare names -- see the tau
      probe in ``logs/diag_tau_convention.log``).
    * The correction is ``sqrt(tau)``, not ``sqrt(2*tau)``: the latter
      double-counts by exactly ``sqrt(2)``.
    """
    from ..analysis.statistics import tau_int

    snaps, sig, acc = sample(wf, R0, nsweep=sweeps, sigma=sigma,
                             rng=np.random.default_rng(seed), snapshot_every=1,
                             equil=equil, target_acc=0.4)
    n = len(snaps)
    T = np.empty(n)
    V = np.empty(n)
    for s, S in enumerate(snaps):
        t, v, _ = wf.local_energy(wf.build(S))
        T[s], V[s] = t.real, v.real
    E = T + wf.kappa * V
    ne = wf.ne
    tau_E, win_E = tau_int(E)
    tau_V, _ = tau_int(V)
    return {
        "label": label, "n": int(n), "acc": float(acc), "sigma": float(sig),
        "snaps": snaps,
        "kappa": float(wf.kappa),
        "kappa_mode": getattr(wf, "kappa_mode", None),
        # per electron -- what the legacy records call E_perpart
        "E": float(E.mean() / ne),
        "E_err": float(E.std(ddof=1) / np.sqrt(n) * np.sqrt(tau_E) / ne),
        "E_err_naive": float(E.std(ddof=1) / np.sqrt(n) / ne),
        "E_tau": float(tau_E),
        "E_tau_window": int(win_E),
        "V": float(V.mean() / ne),
        "V_err": float(V.std(ddof=1) / np.sqrt(n) * np.sqrt(tau_V) / ne),
        "T": float(T.mean() / ne),
        # totals -- the legacy records' own convention for the bare names
        "E_total": float(E.mean()),
        "T_total": float(T.mean()),
        "V_total": float(V.mean()),
        "cov_TV": float(np.cov(T, V, ddof=1)[0, 1]),
    }
