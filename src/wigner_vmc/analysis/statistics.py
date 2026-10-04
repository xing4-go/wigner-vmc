"""The one and only implementation of every error bar in this project.

Nothing else -- not a figure function, not a script, not a report -- may compute a
sigma, a z or a delta_E.  The campaign has already lost time twice to two cells
carrying their own copy of this formula and drifting apart; the fix is structural
rather than disciplinary, so this module is the only place the arithmetic exists.

The convention is `_diag/p1_report.py`'s, which `_diag/energy_scan.py` copies and
self-checks against.  It is reproduced here exactly:

    sigma_mc    = sqrt(sum_i s_i^2) / n      each walk independent; = s/sqrt(n) if equal
    sigma_init  = seed_std / sqrt(n)         spread across INDEPENDENT OPTIMISATIONS
    sigma_total = hypot(hypot(sigma_mc_cry, s_liq), sigma_init)

    delta_E = mean_i(E_cry,i) - E_liq        per particle
    z       = |delta_E| / sigma_total

What `sigma_init` MEANS, and what it does not
---------------------------------------------
It is the standard error of the mean over the states in the sample, and it is called
`sigma_init` rather than `sigma_seed` because what varies between those states is
their INITIALISATION and the basin the optimiser fell into from it -- not a draw from
a Gaussian noise model.  With five starts at five fixed Jastrow widths the "sample" is
the seed set, so `sigma_init` describes basin sensitivity of the optimiser, and it is
NOT an estimate of an underlying population spread.  It must not be quoted as if the
five states were five random samples; the LL-rotation campaign's `nest` tier is the
sharp case, where all five states descend from the five `conv` states of one seed.

One known asymmetry, carried over deliberately rather than quietly corrected: the
liquid's own start-to-start spread never enters, because the liquid is a single
deterministic fill of the lowest Landau level with no optimiser and therefore no basin
to fall into.  Only the crystal contributes a `sigma_init`.  Changing that would change
every published number, so it is documented here and left alone.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from ..io.schema import Bracket, ScanPoint, StateRecord


def sigma_mc(values: Sequence[float], errors: Sequence[float]) -> float:
    """Sampling uncertainty of the MEAN of independently-walked states.

    sqrt(sum_i s_i^2)/n rather than the population form, so that unequal per-walk
    errors are weighted correctly.  Equals s/sqrt(n) when every s_i is equal.
    """
    n = len(values)
    if n == 0:
        return 0.0
    return math.sqrt(sum(float(s) ** 2 for s in errors)) / n


def sigma_init(values: Sequence[float]) -> float:
    """Standard error of the mean across independent optimisations.

    Zero for a single state -- NOT an estimate from one sample, which would be
    meaningless.  The caller is expected to record `n=1` alongside.

    Zero rather than None is the ENERGY path's legacy convention, and the Stage 1
    regression pins it: `_diag/energy_scan.py` sets `sig_opt = 0.0` when n == 1, so
    reproducing its points requires the same.  A single-seed point therefore carries an
    error bar that omits basin spread entirely -- it is a lower bound, and the point's
    `n_states` is the field that says so.  For the case where "unknown" must be spelled
    as unknown, use `seed_spread`.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)     # ddof=1
    return math.sqrt(var) / math.sqrt(n)


def seed_spread(values: Sequence[float]) -> Optional[float]:
    """Standard error of the mean, or None when it cannot be estimated.

    The BAND-OCCUPATION path's legacy convention, and the opposite choice from
    `sigma_init`: `_diag/pn_table.json` writes null, not 0, for its single-seed rows.

    Both conventions are defensible and they are not interchangeable.  0 says "this
    state contributes no basin spread to the budget"; null says "one sample cannot
    bound the spread at all".  The legacy campaign used the first for energies and the
    second for occupations; that divergence is preserved here rather than smoothed
    over, because changing either one would change numbers that have already been
    published.  What must not happen is a caller reading a 0 as if it were a null.
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def combine_runs(states: Iterable[StateRecord]) -> dict:
    """Reduce a set of states of ONE phase at ONE coupling to (mean, sigma_mc, sigma_init, n).

    Refuses to combine states that do not carry a production energy: an OPTIMISER
    trajectory value is not a measurement, and silently substituting one is exactly
    the mistake this module exists to prevent.
    """
    states = list(states)
    missing = [s for s in states if not s.has_production_energy]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(states)} states carry no production energy "
            f"(e.g. {missing[0].init_id!r}).  An optimiser-trajectory energy is not a "
            f"measurement; refusing to combine.")

    values = [s.production_energy_per_particle for s in states]
    errors = [s.production_mc_error if s.production_mc_error is not None else 0.0
              for s in states]
    n = len(values)
    mean = sum(values) / n
    return {
        "mean": mean,
        "sigma_mc": sigma_mc(values, errors),
        "sigma_init": sigma_init(values),
        "n": n,
        "seed_std": (math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
                     if n > 1 else 0.0),
        "errors": errors,
        "values": values,
    }


def compare_phases(crystal_states: Sequence[StateRecord],
                   liquid: StateRecord,
                   tier: str,
                   caveats: Optional[Sequence[str]] = None) -> ScanPoint:
    """The phase comparison at one coupling: delta_E and its error budget.

    delta_E = (E_crystal - E_liquid) per particle, and sigma_total is the legacy
    quadrature -- the crystal's sampling error and the liquid's combined first, then
    the crystal's initialisation spread.
    """
    if not liquid.has_production_energy:
        raise ValueError(f"liquid {liquid.init_id!r} carries no production energy")

    c = combine_runs(crystal_states)
    e_liq = liquid.production_energy_per_particle
    s_liq = liquid.production_mc_error if liquid.production_mc_error is not None else 0.0

    s_mc = math.hypot(c["sigma_mc"], s_liq)
    s_init = c["sigma_init"]
    s_tot = math.hypot(s_mc, s_init)

    dE = c["mean"] - e_liq

    return ScanPoint(
        rs=round(float(liquid.rs), 6),
        tier=tier,
        n_max=(crystal_states[0].nmax if crystal_states else None),
        n_states=c["n"],
        energy_crystal=c["mean"],
        energy_liquid=e_liq,
        delta_e=dE,
        sigma_mc=s_mc,
        sigma_init=s_init,
        sigma_total=s_tot,
        z=(abs(dE) / s_tot) if s_tot > 0 else None,
        seed_ids=[s.init_id for s in crystal_states],
        caveats=list(caveats or []),
    )


def bracket(points: Sequence[ScanPoint], tier: str,
            caveats: Optional[Sequence[str]] = None) -> Bracket:
    """The r_s interval ONE truncation supports, from its own points only.

    Never pools tiers.  n_band = n_max + 1, so two tiers are two different variational
    families; a bracket formed across them describes neither.

    `resolved` is load-bearing, not decoration.  A non-monotone series can have its
    last positive point ABOVE its first negative one, and then lo/hi do not describe
    an interval at all.  That case is reported rather than turned into a plausible
    looking [lo, hi].
    """
    pts = sorted([p for p in points if p.tier == tier], key=lambda p: p.rs)
    if not pts:
        return Bracket(tier=tier, kind="empty", lo=None, hi=None, resolved=False,
                       n_points=0, note="no points in this tier")

    pos = [p for p in pts if p.delta_e > 0]
    neg = [p for p in pts if p.delta_e < 0]

    if pos and neg:
        lo = max(p.rs for p in pos)
        hi = min(p.rs for p in neg)
        if lo < hi:
            note = (f"sign changes between r_s = {lo:g} and {hi:g}")
        else:
            note = (f"DOES NOT BRACKET: last positive point (r_s = {lo:g}) sits above the "
                    f"first negative one (r_s = {hi:g}) -- the series is not monotone and "
                    f"no interval describes it")
        return Bracket(tier=tier, kind="bracket", lo=lo, hi=hi, resolved=bool(lo < hi),
                       n_points=len(pts), note=note)

    if neg:
        return Bracket(tier=tier, kind="below_scan", lo=None,
                       hi=min(p.rs for p in neg), resolved=False, n_points=len(pts),
                       note=f"every point is negative; the crossing lies below r_s = "
                            f"{min(p.rs for p in neg):g}")
    return Bracket(tier=tier, kind="above_scan", lo=max(p.rs for p in pos), hi=None,
                   resolved=False, n_points=len(pts),
                   note=f"every point is positive; no crossing inside the scan")


def brackets_overlap(a: Bracket, b: Bracket) -> Optional[tuple]:
    """The intersection of two RESOLVED brackets, or None.

    Returned rather than a bool because the intersection, not either bracket alone,
    is the strongest statement a two-truncation scan supports.
    """
    if not (a.resolved and b.resolved):
        return None
    lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
    return (lo, hi) if lo < hi else None


# ==========================================================================
# autocorrelation correction to a walk's error bar
# ==========================================================================
def tau_int(x, c: float = 5.0):
    """Integrated autocorrelation time in units of the sampling interval.

        tau = 1 + 2 sum_{t>=1} rho_t,   truncated by the automatic window:
        stop at the first t with t >= c*tau(t).

    Without the window the noisy tail of rho_t adds a random walk of its own variance,
    and for a series of a few hundred points that is the difference between an estimate
    and a number.

    Returns (tau, window).  `tau` is the ONE-EXPONENTIAL model's answer: it assumes
    rho_t decays as a single exponential, which is a stronger assumption than the data
    supports at this sample length.  `blocked()` makes no such assumption, so where the
    two disagree it is `blocked` that is likelier to be right -- and the disagreement
    itself, not either number, is what the caller is entitled to quote.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    ac = np.correlate(x, x, mode="full")[n - 1:] / (n * np.arange(n, 0, -1))
    ac = ac / ac[0]
    tau, t = 1.0, 1
    while t < n:
        tau += 2.0 * ac[t]
        if t >= c * tau:
            break
        t += 1
    return float(tau), int(t)


def blocked(x, block: int) -> float:
    """Error on the mean from the scatter of consecutive block means.

    Model-free, in the sense that it assumes nothing about the shape of rho_t -- only
    that blocks of `block` samples are close to independent.  The caller must apply it
    to the TOTAL energy and compare it against a naive TOTAL error; dividing a
    total-unit blocked() by a per-electron naive error returns exactly N times the true
    correction, which at N = 36 turned an honest factor of 1.9 into a printed 51 --
    the difference between "resolved" and "degenerate everywhere".
    """
    x = np.asarray(x, float)
    nb = x.size // block
    b = x[:nb * block].reshape(nb, block).mean(axis=1)
    return float(b.std(ddof=1) / np.sqrt(nb))


def autocorrelation_correction(series, blocks=(1, 2, 4, 8, 16, 32, 64, 100),
                               block_for_corr: int = 8):
    """Both correction factors for one walk's energy series.

    `corr` (tau_int) and `block_corr` (block averaging at `block_for_corr`) are returned
    as a PAIR and never collapsed into one number.  On the sample lengths this campaign
    ran, they disagree; the campaign carries the tau_int value and prints the block
    value beside it, and that is a choice about which to prefer, not a licence to
    pretend the other does not exist.

    The naive error used as the denominator is in TOTAL units, to match `blocked()`.
    """
    E = np.asarray(series, float)
    naive_tot = float(E.std(ddof=0) / math.sqrt(E.size))
    tau, window = tau_int(E)
    b8 = blocked(E, block_for_corr)
    return {
        "tau": tau, "window": window, "n": int(E.size),
        "naive_total": naive_tot,
        "tau_corr": float(math.sqrt(2.0 * tau)),
        "block_corr": float(b8 / naive_tot) if naive_tot > 0 else float("nan"),
        "corr": float(math.sqrt(2.0 * tau)),
        "blocks": {int(b): blocked(E, b) for b in blocks if b <= E.size},
    }
