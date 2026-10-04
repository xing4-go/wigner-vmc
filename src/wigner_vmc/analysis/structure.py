"""Structure observables: S(q), g(r), g(x,y), S(qx,qy), and the SMA dispersion.

These are ANALYSIS, not plotting, and they are also not physics: nothing here touches
the Hamiltonian, the wavefunction or the sampler.  A snapshot is an (N,2) array of
positions; everything below is a function of snapshots and geometry.

Why this module exists
----------------------
The legacy notebook computed every one of these inside the cell that drew them, so the
quantity and its picture were the same unit of code -- which is exactly why the figures
could not be re-derived without running the notebook.  Here the figure receives numbers.

VERIFIED AGAINST THE LEGACY ARRAYS, NOT AGAINST MY READING OF THE FORMULA
------------------------------------------------------------------------
`_diag/ckpt_rebuild/valid_lll_structure.pkl` stores the filled-LLL snapshots AND the
S(q) and g(r) the legacy pipeline computed from them.  Running this module on those
snapshots reproduces both stored arrays, WITH THE COLUMNS MATCHED BY q VECTOR:

    S(q)   (334, 294)   max |difference| = 3.3e-14
    g(r)   (40,)        max |difference| = 0.0

so the conventions below are the legacy ones rather than my reconstruction of them.
(That equality is asserted, not remembered: see `tests/test_structure_against_legacy.py`,
which skips when the legacy tree is absent.)

The 3.3e-14 rather than 0.0 is the honest number, and the alignment clause is not
decoration.  The stored array is in the engine's `np.argsort(qn)` order; this module's
canonical order is the rounded `np.lexsort`.  Same 294 momenta, permuted columns, and
the two orders are not interchangeable: comparing them without matching gives a maximum
difference of 11.4.  That number is large enough to look like a broken structure factor
and is entirely a bookkeeping artefact -- and an earlier version of the test indexed the
stored array with indices from the wrong order and so measured that 11.4 without
anything else noticing.  The per-column match is therefore asserted in the test (each
stored momentum must find a distinct partner on this torus), and a per-star comparison
is kept alongside it so that a future change of column order cannot fail the check for
a reason that is not physics.

The three conventions that are easy to get wrong
------------------------------------------------
1. **S(q) is normalised by N and averaged over snapshots; the average is of |rho_q|^2
   and not of rho_q.**  The two differ by <|rho_q|^2> - |<rho_q>|^2, which is the WHOLE
   of S(q) for a liquid: a filled Landau level has uniform density, so |<rho_q>|^2
   vanishes at every q != 0 and averaging rho_q first would draw an empty panel.  This
   is not a subtlety that can be left to the reader.

2. **g(r) uses the MINIMUM periodic image; g(x,y) uses periodic IMAGES.**  Different
   routines, opposite choices, and both are deliberate:
     - g(r) bins |d|, and |d| must be the true shortest separation between the two
       electrons, so the minimum image is the correct one.
     - g(x,y) keeps the DIRECTION, over a window (half-width 10 l_B) larger than the
       Wigner-Seitz cell (inradius 8.1 l_B).  Minimum-imaging there would fold the
       crystal's lattice peaks back at the cell boundary; images let them reach the edge.
   Each docstring states which it does, because the two look interchangeable and are not.

3. **Every pair appears in BOTH orientations in g(x,y) but only once in g(r).**  g(r)
   = g(-r) identically, so binning |d| is already mirror-symmetric and the triangle
   i<j is safe.  The two-dimensional map keeps the direction and is NOT mirror-safe:
   summing over i<j puts each pair's weight at its own d and never at -d, so the map
   would carry the anisotropy of the index ordering instead of that of the
   configuration.  It is invisible in the liquid (broad peaks overlap their own
   mirrors) and shows up for a crystal as spurious Bragg-peak height modulation, up to
   a factor of five between peaks the lattice symmetry requires to be equal.  The
   normalisation doubles with the count: ne*(ne-1) displacements, not ne*(ne-1)/2.

The error bar on a snapshot average is taken ACROSS SNAPSHOTS, and the ddof is a
required argument
--------------------------------------------------------------------------------
There is no per-snapshot sigma to combine here -- the snapshots are successive
configurations of one walk, so the sample is the snapshot set.  The legacy pipeline
used two different ddof values on this path (`Sq` uses numpy's default ddof=0, the
band-occupation table uses ddof=1), and `snapshot_sem` therefore has NO DEFAULT: every
call site has to state which one it means.  A default would silently pick one of two
legacy conventions and there would be nothing in the code to notice it by.

None of these error bars accounts for autocorrelation between successive snapshots.
They are snapshot scatters and are a LOWER bound on the true integrated-autocorrelation
error; the docstrings of the callers say so where it matters.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


# ==========================================================================
# geometry
# ==========================================================================
class Torus:
    """The triangular supercell the campaign ran on, derived from its density.

    A small object rather than a set of module constants, because a constant is
    reusable by accident: a function that reaches for `G1` keeps working when someone
    changes the system size, and returns numbers for the wrong lattice.  Everything
    here follows from (area, n_electrons, n_cells), so a different run cannot silently
    inherit this one's vectors.
    """

    def __init__(self, area: float, n_electrons: int = 36, n_cells_per_side: int = 6):
        self.area = float(area)
        self.ne = int(n_electrons)
        self.n_side = int(n_cells_per_side)
        n_cells = self.n_side ** 2
        if n_cells != self.ne:
            raise ValueError(f"{n_cells} cells but {self.ne} electrons; the campaign is "
                             f"one electron per cell")
        # triangular lattice constant at nu = 1: (sqrt3/2) a^2 = area/N_cells
        self.A_WC = math.sqrt(2.0 * self.area / (math.sqrt(3.0) * n_cells))
        self.A1 = np.array([self.A_WC, 0.0])
        self.A2 = np.array([0.5 * self.A_WC, 0.5 * math.sqrt(3.0) * self.A_WC])
        self.L1 = self.n_side * self.A1
        self.L2 = self.n_side * self.A2
        self.sc = np.column_stack([self.L1, self.L2])
        self.c2sc = np.linalg.inv(self.sc)
        # supercell reciprocal vectors, engine convention: G1 = 2pi zhat x L2 / area
        self.G1 = 2 * np.pi * np.array([self.L2[1], -self.L2[0]]) / self.area
        self.G2 = -2 * np.pi * np.array([self.L1[1], -self.L1[0]]) / self.area
        # underlying-lattice reciprocal vectors (the WC Bragg vectors)
        cell_area = self.area / n_cells
        self.g1 = 2 * np.pi * np.array([self.A2[1], -self.A2[0]]) / cell_area
        self.g2 = -2 * np.pi * np.array([self.A1[1], -self.A1[0]]) / cell_area
        self.sqrt_n = math.sqrt(self.ne / self.area)

    def __repr__(self):
        return (f"Torus(area={self.area:.6f}, ne={self.ne}, A_WC={self.A_WC:.6f}, "
                f"|G1|={np.linalg.norm(self.G1):.6f}, |g1|={np.linalg.norm(self.g1):.6f})")

    def allowed_momenta(self, q_max: float = 4.0, tol: float = 1e-6):
        """Every q = m*G1 + n*G2 with 0 < |q| < q_max, in a canonical order.

        The m,n window is the legacy one: mx = 2*floor(q_max/|G1|), so the search
        square is twice the radius needed.  Kept identical rather than tightened,
        because the surviving SET is what the stored arrays index; a "cleaner" window
        that dropped a boundary momentum would silently change the observable.

        q = 0 is excluded: it is the conserved total charge, not an observable, and
        including it would put an S(q=0) = N entry in a structure-factor plot.

        WHY THE ORDER IS ROUNDED, AND WHY THAT IS NOT A HACK
        ---------------------------------------------------
        On this torus NOT ONE momentum is alone in its star: all 294 fall into 30
        shells of 6, 12 or 18 degenerate |q|.  So a plain `argsort(qn)` puts the whole
        column order at the mercy of the last bit of `qn` -- and the last bit is not a
        physical quantity.  It depends on how the lattice constant was computed:
        `sqrt(4*pi/sqrt(3))` (the legacy's closed form) and `sqrt(2*area/(sqrt(3)*N))`
        (this module's, which follows from the frozen area and so stays correct at any
        system size) agree to 1e-16 but NOT bit-for-bit, and that 1-ulp difference
        permutes every star.  Measured: same snapshots, same physics, and the two
        orderings disagree column-by-column with a maximum difference of 11.4 -- while
        the sorted values agree to 3e-14 and the star structure is identical.

        Rounding the sort key to `tol` collapses the float noise and nothing else: the
        closest distinct shells are ~0.33 apart, five orders of magnitude above the
        noise, and the closest momenta within a shell are ~0.1 apart.  The result is an
        order that is reproducible across constructions instead of an accident of one.
        """
        lo = float(np.linalg.norm(self.G1))
        mx = 2 * int(np.floor(q_max / lo))
        q = np.array([m * self.G1 + n * self.G2
                      for m in range(-mx, mx + 1) for n in range(-mx, mx + 1)])
        qn = np.linalg.norm(q, axis=1)
        keep = (qn < q_max) & (qn > 1e-9)
        q, qn = q[keep], qn[keep]
        key = np.round(np.column_stack([qn, q[:, 0], q[:, 1]]), int(round(-math.log10(tol))))
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        return q[order], qn[order]

    def wc_shell_vectors(self):
        """The six shortest reciprocal vectors of the triangular Wigner lattice."""
        v = np.array([m * self.g1 + n * self.g2
                      for m in (-2, -1, 0, 1, 2) for n in (-2, -1, 0, 1, 2)])
        v = v[np.abs(np.linalg.norm(v, axis=1) - np.linalg.norm(self.g1))
              < 1e-9 * np.linalg.norm(self.g1)]
        return v[np.lexsort((v[:, 1], v[:, 0]))]

    def lll_reference_momenta(self, q_max: float = 4.0):
        """The quietest m,n window, for the filled-LLL calibration the legacy used.

        Not the same call as `allowed_momenta`: same set, but the legacy `qall` in the
        calibration cell is built with the same doubling.  Kept as a named alias so a
        reader looking for the calibration's grid finds it.
        """
        return self.allowed_momenta(q_max)


# ==========================================================================
# S(q): one and two dimensional
# ==========================================================================
def structure_factor_snapshots(snaps, qvecs) -> np.ndarray:
    """|rho_q|^2 / N per snapshot.  Shape (n_snaps, n_q).

    The average over the first axis is <|rho_q|^2>/N, NOT |<rho_q>|^2/N.  See the
    module docstring: for a liquid the second is zero at every q != 0.
    """
    q = np.asarray(qvecs, float)
    out = np.empty((len(snaps), len(q)), float)
    for i, R in enumerate(snaps):
        R = np.asarray(R, float)
        phase = np.exp(1j * (R @ q.T))
        out[i] = np.abs(phase.sum(axis=0)) ** 2 / R.shape[0]
    return out


def structure_factor(snaps, qvecs, ne: int) -> np.ndarray:
    """<|rho_q|^2>/ne, the snapshot average -- the 1-D observable itself."""
    per = structure_factor_snapshots(snaps, qvecs)
    return per.mean(axis=0)


def structure_factor_2d(snaps, qx, qy, ne: int) -> np.ndarray:
    """S(q_x,q_y) on a grid; returns an array shaped like `qx`."""
    qs = np.column_stack([np.ravel(qx), np.ravel(qy)])
    S = np.zeros(len(qs))
    for R in snaps:
        S += np.abs(np.exp(1j * (np.asarray(R, float) @ qs.T)).sum(axis=0)) ** 2
    return (S / (ne * len(snaps))).reshape(np.shape(qx))


def snapshot_sem(per_snapshot, ddof: int) -> np.ndarray:
    """Standard error of the mean over snapshots.  `ddof` is REQUIRED -- see module doc.

    ddof=0 is numpy's default and what the legacy S(q) error used; ddof=1 is the
    sample convention and what the band-occupation table used.  Both are taken
    somewhere in this project, so neither is allowed to be implicit.
    """
    a = np.asarray(per_snapshot, float)
    n = a.shape[0]
    if n < 2:
        return np.full(a.shape[1:], np.nan)
    return a.std(axis=0, ddof=ddof) / math.sqrt(n)


# ==========================================================================
# g(r): one and two dimensional
# ==========================================================================
def minimum_image_displacement(d, sc, c2sc):
    """Shortest periodic image of a 2-D displacement, for any supercell shape.

    The nine images around the parallelogram representative are searched, which is
    enough for the 60-degree rhombi used here (checked against a +/-4 brute force on a
    dense scan of the cell).  A strongly skewed basis would want a wider window.

    This is NOT the same as rounding the fractional coordinates into the cell:
    rounding picks one fixed representative per image class, which is all a Bloch
    orbital needs but is not always the nearest one.  For this cell the fractional
    displacement (0.49, 0.49) is already inside the parallelogram at |d| = 0.849, while
    its image one lattice vector away sits at |d| = 0.500; only the second is the
    distance between those two electrons.
    """
    d = np.asarray(d, float)
    dsc = d @ c2sc.T
    dsc = dsc - np.round(dsc)
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
    images = (dsc[..., None, :] + shifts) @ sc.T
    best = np.argmin(np.linalg.norm(images, axis=-1), axis=-1)
    return (dsc + shifts[best]) @ sc.T


def pair_correlation(snaps, torus: Torus, rmax: float = 6.0, nbins: int = 40):
    """Radial g(r) with the minimum-image convention, and the per-snapshot scatter.

    Normalised so that a uniform liquid gives g = 1.  `triu` counts each pair ONCE, so
    the count that normalises the histogram is ne*(ne-1)/2, not ne*(ne-1).

    The filled-LLL calibration is the reason to trust it: on the snapshots the legacy
    stored, this reproduces the stored g(r) with a maximum difference of exactly zero.

    Returns (r, g, g_sem), where g_sem is the ddof=1 scatter over snapshots.  `rmax`
    must stay below half the shortest supercell vector (8.08 l_B here) or the
    minimum-image convention starts DROPPING pairs -- at rmax = 8.08 it loses 10.4% of
    the cell, which bends g(r) down over the top of the range.  That is checked, not
    trusted.
    """
    L1, L2 = torus.L1, torus.L2
    ne = torus.ne
    half_min = 0.5 * min(np.linalg.norm(L1), np.linalg.norm(L2))
    if rmax > half_min:
        raise ValueError(
            f"rmax={rmax} exceeds half the shortest supercell vector ({half_min:.4f}); "
            f"past that the minimum-image convention drops pairs and g(r) bends down "
            f"over the top of its range")
    dr = rmax / nbins
    edges = np.arange(nbins + 1) * dr
    nrm = (ne * (ne - 1) / 2.0)
    per_snap = np.empty((len(snaps), nbins), float)
    for i, R in enumerate(snaps):
        R = np.asarray(R, float)
        d = R[:, None, :] - R[None, :, :]
        dmin = np.linalg.norm(minimum_image_displacement(d, torus.sc, torus.c2sc), axis=-1)
        rr = dmin[np.triu_indices(ne, 1)]
        per_snap[i] = np.histogram(rr, bins=edges)[0]
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 2 * np.pi * r * dr
    g = per_snap.mean(axis=0) * torus.area / (nrm * shell)
    g_sem = (per_snap * torus.area / (nrm * shell)).std(axis=0, ddof=1) / math.sqrt(len(snaps)) \
        if len(snaps) > 1 else np.full(nbins, np.nan)
    return r, g, g_sem


def poisson_pair_error(g, r, rmax: float, nbins: int, torus: Torus, n_snaps: int):
    """The legacy Poisson error on g(r): g / sqrt(number of pairs in the shell).

    Not the snapshot scatter -- a count-based estimate, used by the filled-LLL
    calibration cell.  Valid there because the shell counts run to the thousands;
    the caller must not reuse it where they do not.
    """
    dr = rmax / nbins
    npr = (torus.ne * (torus.ne - 1) / 2.0) * n_snaps * (2 * np.pi * r * dr) / torus.area
    return g / np.sqrt(npr)


def gauss_blur(A, sigma: float):
    """Separable Gaussian blur of a histogram; EDGES ARE HELD, not wrapped.

    The plotted window is not a period of the supercell, so wrapping would fold one
    edge of the picture onto the other.  The kernel sums to 1, so the normalisation --
    and the g = 1 of a uniform liquid -- is unchanged.
    """
    if sigma <= 0:
        return A
    rad = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    out = A
    for axis in (0, 1):
        pad = np.pad(out, [(rad, rad) if a == axis else (0, 0) for a in (0, 1)],
                     mode="edge")
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, pad)
    return out


def pair_correlation_2d(snaps, torus: Torus, rmax: float, nbins: int, smooth: float):
    """g(x,y) on a square grid: two-dimensional, NO angular average.

    Periodic images rather than the minimum image -- see the module docstring for why
    the two routines differ.  Every pair contributes at d + L AND at -d + L for each
    lattice vector L that can reach the window, and the count in a bin of area dA is
    compared with the uniform value (ne-1)*dA/area per snapshot, so a uniform liquid
    gives g = 1.  Both orientations mean ne*(ne-1) displacements and a normalisation
    that doubles with them.

    Returns (centres, g) where `centres` is the 1-D bin-centre coordinate and g[i,j]
    is the map at (x_i, y_j).
    """
    L1, L2 = torus.L1, torus.L2
    ne = torus.ne
    dmax = 0.5 * (np.linalg.norm(L1) + np.linalg.norm(L2))
    imgs = np.array([m * L1 + n * L2 for m in range(-3, 4) for n in range(-3, 4)
                     if np.linalg.norm(m * L1 + n * L2) <= rmax + dmax])
    edges = np.linspace(-rmax, rmax, nbins + 1)
    delta_A = (edges[1] - edges[0]) ** 2
    iu = np.triu_indices(ne, 1)
    H = np.zeros((nbins, nbins))
    for R in snaps:
        R = np.asarray(R, float)
        d = R[:, None, :] - R[None, :, :]
        dsc = d @ torus.c2sc.T
        dsc -= np.round(dsc)
        dd = (dsc @ torus.sc.T)[iu]
        dd = np.concatenate([dd, -dd])
        u = (dd[:, None, :] + imgs[None, :, :]).reshape(-1, 2)
        u = u[np.max(np.abs(u), axis=1) <= rmax]
        H += np.histogram2d(u[:, 0], u[:, 1], bins=(edges, edges))[0]
    g = gauss_blur(H, smooth) * torus.area / (len(snaps) * ne * (ne - 1.0) * delta_A)
    return 0.5 * (edges[1:] + edges[:-1]), g


# ==========================================================================
# closed-form references
# ==========================================================================
def exact_lll_sq(q):
    """The filled LLL structure factor, 1 - exp(-q^2/2).  Exact at any N."""
    return 1.0 - np.exp(-np.asarray(q, float) ** 2 / 2.0)


def exact_lll_g(r, n_electrons: int = 36):
    """The filled-LLL g(r), (N/(N-1)) (1 - exp(-r^2/2)).

    The N/(N-1) is the finite-N plateau, NOT a normalisation applied to the measured
    g: the engine's g already tends to 1 for a uniform liquid, and on the stored
    calibration snapshots the measured plateau is 1.0292, i.e. N/(N-1) = 1.0286 to
    within the r>4 cut.  It is the exact curve, not a correction to the data.
    """
    r = np.asarray(r, float)
    return (n_electrons / (n_electrons - 1.0)) * (1.0 - np.exp(-r ** 2 / 2.0))


# ==========================================================================
# the SMA dispersion
# ==========================================================================
def sma_dispersion(q, S):
    """omega_SMA / omega_c = q^2 / (2 S(q)).

    Kohn's theorem forces this to 1 as q -> 0 for the filled LLL, which is the one
    limit where the answer is known -- so the smallest allowed q is reported and the
    limit itself is never extrapolated to q = 0, which is off this torus entirely.
    """
    q = np.asarray(q, float)
    S = np.asarray(S, float)
    return q ** 2 / (2 * np.maximum(S, 1e-12))


def sma_error(q, S, S_err):
    """Propagated error on the SMA from the error on S: |domega/dS| * sigma_S."""
    q = np.asarray(q, float)
    S = np.asarray(S, float)
    return (q ** 2 / (2 * np.maximum(S, 1e-12) ** 2)) * np.asarray(S_err, float)


# ==========================================================================
# shell bookkeeping
# ==========================================================================
def shells_by_magnitude(qn, tol: float = 1e-6):
    """[(|q|, indices), ...] grouping degenerate momenta into stars.

    The multiplicity travels with the number on purpose: a shell of six equivalent
    momenta is not one measurement, and a reader who takes the multiplicity for
    decoration will misjudge the error bar.
    """
    qn = np.asarray(qn, float)
    order = np.argsort(qn)
    srt = qn[order]
    cuts = np.where(np.diff(srt) > tol)[0] + 1
    return [(float(qn[ix].mean()), ix) for ix in np.split(order, cuts)]


def shell_average(snaps, qvecs, ne: int, shells=None):
    """Reduce S(q) to one row per degenerate star.

    Returns an (n_shells, 4) array of [|q|_shell, S_mean, S_sem, multiplicity].

    The reduction happens on the PER-SNAPSHOT values, not on the already-averaged
    S(q): `S` is the mean over snapshots of |rho_q|^2/ne, and the error on that mean is
    the snapshot scatter divided by sqrt(n_snaps) with ddof=1.  Averaging the stars
    first and then taking the scatter would treat six equivalent momenta as six
    independent measurements of the same number, which they are not -- they are six
    measurements of six different q that the lattice symmetry says must agree, so their
    spread measures the symmetry breaking, not the sampling error.  For a liquid the
    first is the honest bar and it is the one the legacy `_shell_S` used.

    `ddof=1` here and `ddof` is required in `snapshot_sem` for the same reason: the
    legacy S(q) error path used ddof=0 and the shell table used ddof=1, and this
    function reproduces the SHELL TABLE.  Making it implicit is how the two conventions
    got confused in the first place.
    """
    q = np.asarray(qvecs, float)
    if shells is None:
        shells = shells_by_magnitude(np.linalg.norm(q, axis=1))
    per = structure_factor_snapshots(snaps, q)          # (n_snaps, n_q)
    ns = per.shape[0]
    out = np.empty((len(shells), 4), float)
    for i, (q_shell, ix) in enumerate(shells):
        v = per[:, ix].mean(axis=1)                     # per snapshot, star-averaged
        sem = (v.std(ddof=1) / math.sqrt(ns)) if ns > 1 else float("nan")
        out[i] = (q_shell, float(v.mean()), float(sem), float(len(ix)))
    return out


# ==========================================================================
# the production estimators: the two numbers an order parameter is quoted from
#
# Both are emitted CELL text in the generators rather than engine symbols -- the
# Gaussian generator has exactly four module-level definitions and these are not
# among them -- so the notebook source is the source of truth for them.
# ==========================================================================
def sq_error_bars(snaps, q, ne):
    """Mean and standard error of S(q) at ONE wavevector, from the snapshot spread.

    Legacy source: ``make_notebook.py:1435-1443``, body unchanged.

    The snapshots are MCMC-correlated, so this is an underestimate of the true
    (integrated-autocorrelation) error -- treat it as a lower bound.

    This is ``snapshot_sem(per, ddof=0)`` in different words; the two are kept
    side by side because the legacy call sites used this one and the shell table
    used ``ddof=1``, and `snapshot_sem` requires the choice to be explicit.
    """
    per = np.array([np.abs(np.exp(1j * (R @ q)).sum()) ** 2 / ne for R in snaps])
    return per.mean(), per.std() / np.sqrt(len(per))


def bragg_ratio(snaps, bragg_q, bg_q, ne):
    """(mean peak S(q_Bragg), median background, ratio) -- the order parameter.

    Legacy source: ``reproduction/make_notebook_llrot.py:1427-1440``
    (``_bragg_ratio``), algebra unchanged.  The cell closed over two module-level
    arrays, ``_bragg`` (the six first-shell vectors) and ``_bgq`` (the background
    momenta); a library function cannot, so they are parameters here.  That is
    the same shape the engine's own ``structure_factor(snaps, qs, ne)`` already
    had, so no calling convention is being invented.

    The ratio is a MEAN PEAK over a MEDIAN background, and the two halves are
    not interchangeable: the baseline's estimator is the median, so a ratio
    computed against a mean background is not comparable with any number quoted
    from it.  The numerator is a peak, not a ratio -- the two are easy to
    conflate and the legacy comment records that an earlier draft did, everywhere
    it quoted the baseline.
    """
    S_pk = np.real(structure_factor(snaps, bragg_q, ne))
    S_bg = np.real(structure_factor(snaps, bg_q, ne))
    pk, bk = float(S_pk.mean()), float(np.median(S_bg))
    return pk, bk, (pk / bk if bk > 1e-12 else np.inf)


# ==========================================================================
# real-space density
# ==========================================================================
def lattice_sites(torus) -> np.ndarray:
    """The 36 ansatz sites of the triangular lattice, folded onto the outlined cell.

    `fractional in [0, 1)` is the useful form for the figures: the density panels
    outline one cell with a periodic margin, and these are the sites that appear inside
    it.  The engine's `gaussian_sites` instead folds into the Wigner-Seitz cell centred
    on the ORIGIN, which puts most of them at negative coordinates -- correct for the
    wavefunction, and the reason a naive `sites` scatter lands off the axes.
    """
    n = torus.n_side
    s = np.array([i * torus.A1 + j * torus.A2 for i in range(n) for j in range(n)])
    return (s @ torus.c2sc.T) % 1.0


def density_grid(snaps, torus, nbins: int = 72) -> np.ndarray:
    """Real-space density histogram over the supercell, normalised to <rho> = ... .

    Counts particles into `nbins` x `nbins` cells in FRACTIONAL coordinates and scales
    by nbins^2/area, so the mean of the grid is ne/area = the density the cell actually
    holds.  A hard histogram, not a kernel estimate -- the smoothing is a separate,
    explicit step (`periodic_gaussian_blur`) so that the width stays a visible choice
    rather than being buried in the estimator.
    """
    sc = torus.sc
    c2sc = torus.c2sc
    H = np.zeros((nbins, nbins), float)
    for R in snaps:
        f = (np.asarray(R, float) @ c2sc.T) % 1.0
        idx = np.floor(f * nbins).astype(int) % nbins
        np.add.at(H, (idx[:, 0], idx[:, 1]), 1.0)
    H /= len(snaps)
    return H * (nbins * nbins) / torus.area


def periodic_gaussian_blur(H, sigma_bins: float, truncate: float = 4.0) -> np.ndarray:
    """Separable Gaussian blur with WRAP boundary -- i.e. on the torus.

    Wrap and not the held-edge form `gauss_blur` uses: this grid IS periodic, and a
    held edge would make the cell boundary a place where the density appears to change
    for a reason that is not physical.  The kernel is truncated at `truncate` sigma and
    renormalised, which is the same convention scipy's `gaussian_filter(mode="wrap")`
    uses, so the two agree to machine precision rather than merely closely -- and the
    dependency stays out of this package, which has no business pulling in scipy for a
    display choice.
    """
    H = np.asarray(H, float)
    if sigma_bins <= 0:
        return H.copy()
    r = int(truncate * sigma_bins + 0.5)
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()

    def blur_axis(A, axis):
        n = A.shape[axis]
        idx = (np.arange(n)[:, None] + np.arange(-r, r + 1)[None, :]) % n
        B = np.take(A, idx, axis=axis)
        # np.take inserts the index dimensions at `axis`, so for a 1-D index of length
        # 2r+1 the contracted axis sits at axis+1 for a 2-D array.
        return np.tensordot(B, k, axes=([axis + 1], [0]))

    return blur_axis(blur_axis(H, 0), 1)


def site_midpoint_contrast(H, torus, nbins: int, period: int = 6) -> float:
    """Mean density at the 36 sites over the mean at their 216 nearest-neighbour midpoints.

    ONE number, no threshold and no peak counting, and exactly 1 for anything uniform --
    which is what makes it a null-calibrated statistic rather than a tuned one.  The
    liquid columns of the density figure are the control: whatever they read is the
    estimator's own bias at this sample count, not structure.
    """
    sites = lattice_sites(torus)                     # fractional, in [0,1)
    n = torus.n_side
    frac = np.array([i * torus.A1 + j * torus.A2 for i in range(n) for j in range(n)])
    d = (frac[:, None, :] - frac[None, :, :]) @ torus.c2sc.T
    d -= np.round(d)
    dist = np.linalg.norm(d @ torus.sc.T, axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :6]
    mids = np.array([0.5 * (frac[i] + frac[j]) for i in range(len(frac)) for j in nn[i]])
    mids = (mids @ torus.c2sc.T) % 1.0

    def sample(pts):
        g = pts * nbins - 0.5
        i0 = np.floor(g).astype(int)
        t = g - i0
        i0 %= nbins
        i1 = (i0 + 1) % nbins
        return (H[i0[:, 0], i0[:, 1]] * (1 - t[:, 0]) * (1 - t[:, 1])
                + H[i0[:, 0], i1[:, 1]] * (1 - t[:, 0]) * t[:, 1]
                + H[i1[:, 0], i0[:, 1]] * t[:, 0] * (1 - t[:, 1])
                + H[i1[:, 0], i1[:, 1]] * t[:, 0] * t[:, 1])

    return float(sample(sites).mean() / sample(mids).mean())
