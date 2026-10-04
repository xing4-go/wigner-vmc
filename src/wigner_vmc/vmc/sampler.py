"""The Metropolis walk: proposals, the acceptance rule and the sigma adaptation.

Physical role:
    One sweep attempts one move per electron.  The proposal is a Gaussian step of
    width ``sigma`` around the electron's current position, folded back into the
    first supercell; the move is accepted with probability ``|Psi'/Psi|^2``.
    Nothing here knows which ansatz it is walking -- it is handed a wavefunction
    with ``build``/``move_ratio``/``accept_move`` and calls them.

Main mathematical object:
    The Markov chain on configurations ``R`` with transition kernel
    ``T(R -> R') = sigma-step * min(1, |Psi(R')/Psi(R)|^2)``.  ``sigma`` is
    adapted from the acceptance rate of the LAST 1000 ATTEMPTS -- a windowed
    rate.  That window is not a detail: the cumulative rate converges to
    ``target_acc`` and then freezes ``sigma``, so a chain that equilibrates
    slowly never gets a larger step.  ``sigma`` is a property of the chain, not
    of the state, and is returned rather than mutated in place.

Units/convention:
    ``l_B = 1``.  Positions are Cartesian in the supercell; ``sc_to_cart`` /
    ``cart_to_sc`` are the fold matrices, whose columns are ``L1, L2``.

Legacy source:
    ``qhvmc_engine.py:903-951`` -- ``sample``, verbatim.  The RNG contract is
    preserved exactly: the default is ``np.random.default_rng(0)`` (NOT the
    global NumPy stream), each attempt draws one ``standard_normal(2)`` and then
    one ``random()``, and the decisions are consumed in the same order.  A test
    drives the frozen engine and this one from two generators with the same seed
    and requires the accept/reject decisions to agree move for move.

Stage 2B / B4.  RNG semantics, proposal semantics, acceptance rule, boundary
handling and defaults are unchanged.  The RNG was not swapped for something
easier to make deterministic.
"""
from __future__ import annotations

import numpy as np

from ..physics.geometry import wrap_to_supercell

__all__ = ["sample"]


def sample(wf, R0, nsweep, sigma=0.5, rng=None, snapshot_every=5,
           equil=None, reset_period=None, target_acc=0.5, callback=None):
    """Metropolis walk over whole sweeps (one move attempt per electron).

    Returns ``(snapshots, sigma, acc_rate)``.  Snapshots are taken every
    ``snapshot_every`` sweeps once ``sweep >= equil`` (default
    ``equil = nsweep // 2``), and are copies -- the walk keeps mutating its own
    array, and handing out a view is how a "snapshot" silently becomes a
    sequence of identical late-time configurations.

    ``rng`` defaults to ``default_rng(0)``.  Pass an explicit generator to make a
    run reproducible independently of anything else that has drawn from the
    default stream.

    ``reset_period`` (default ``100 * ne``) is the number of ATTEMPTS between
    exact recomputations of ``D_inv``; the rank-1 updates drift, so this is a
    correctness parameter, not a tuning knob.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ne = wf.ne
    if equil is None:
        equil = nsweep // 2
    if reset_period is None:
        reset_period = 100 * ne          # sample.jl: reset_period
    st = wf.build(R0)

    hist = np.zeros(1000, dtype=np.int8)
    nacc = 0
    ntot = 0
    snaps = []
    for sweep in range(nsweep):
        for i in range(ne):
            r_new = wrap_to_supercell(
                st["R"][i] + sigma * rng.standard_normal(2), wf.sc_to_cart,
                wf.cart_to_sc
            )[0]
            acc, info = wf.move_ratio(st, i, r_new)
            ok = rng.random() < acc
            if ok:
                wf.accept_move(st, i, r_new, info)
            hist[ntot % 1000] = ok
            nacc += ok
            ntot += 1
            if ntot % 1000 == 0 and ntot >= 1000:
                rate = hist.mean()
                if rate > target_acc:
                    sigma *= 1.05
                elif rate < target_acc:
                    sigma *= 0.95
            if reset_period and ntot % reset_period == 0:
                wf.rebuild(st)
        if callback is not None:
            callback(sweep, st)
        if sweep >= equil and (sweep - equil) % snapshot_every == 0:
            snaps.append(st["R"].copy())
    return snaps, sigma, nacc / max(ntot, 1)
