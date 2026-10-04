"""A single-point structural diagnostic, from an independent clean VMC run.

    (r_s^L, r_s^C, n_max^C)  ->  clean VMC  ->  { n(x,y), g(x,y), S(qx,qy) }

    python examples/figure_construction/structure.py --budget quick
    python examples/figure_construction/structure.py \\
        --liquid-rs 5.567 --crystal-rs 90.510 --crystal-nmax 3 --budget quick
    python examples/figure_construction/structure.py \\
        --liquid-rs 5.567 --crystal-rs 90.510 --crystal-nmax 3 --budget full

``--budget full`` is an alias for the ``reproduction`` config -- the real
statistics, not the smoke test.  ``quick`` stamps every figure "QUICK / not
publication quality"; ``full`` does not.

What this is
------------
One invocation is ONE point: one liquid coupling, one crystal coupling, one
crystal Landau-level truncation.  It runs two independent clean VMC
calculations -- a **filled-LLL liquid at r_s^L** and an **LL-rotation crystal at
r_s^C with n = 0 ... n_max^C** -- and builds three structural observables from
the configurations those two walks just produced:

    1. ``n(x,y)``        smoothed real-space density, periodic Gaussian kernel;
    2. ``g(x,y)``        displacement-resolved pair correlation (not radial);
    3. ``S(qx,qy)``      static structure factor on a reciprocal-space grid.

This is deliberately NOT the historical coupling ladder.  There is no sweep, no
column of kappa, no automatic coupling sequence: the point is the input.

Nothing here is a redraw, and the recipe reads NO input file at all
-----------------------------------------------------------------
Every number comes from a VMC run this script starts itself.  The API builds its
own starting point from ``(N, r_s)`` alone.  This module contains no path to a
checkpoint, a frozen snapshot, a stored observable array, a ``_diag`` product or
an old PNG -- there is no legacy constant in it to reference, which is a
stronger statement than "the legacy path is off by default".

The historical figures and their notebook cells were read while this was
written, for exactly one purpose: to recover the observable DEFINITIONS -- grid,
kernel, window, binning, normalisation, periodic wrapping and the q -> 0
convention.  Those recovered numbers are named as module constants below, each
with the source line it came from.  None of them is data.

Where things live
-----------------
    results/figure_construction/structure/<slug>/   the numbers it measures
    figures/figure_construction/structure/<slug>/   the pictures it draws

``<slug>`` names every input that changes the numbers, so two points can be
compared side by side instead of the second erasing the first.

Units: ``l_B = hbar = m = 1``, energy in ``hbar*omega_c``.  ``r_s = sqrt(2) *
kappa``, and kappa is the PHYSICAL ``r_s/sqrt(2)`` at full precision -- never
the Stage-2E rounded regression value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime

import numpy as np

import wigner_vmc
from wigner_vmc import VMC, __version__, load_budget, resolve_budget_name
from wigner_vmc.analysis import structure as st

# ==========================================================================
# paths.  Derived from this file's location, so the recipe works from any cwd.
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))          # .../wigner_vmc_clean
ROOT = os.path.dirname(CLEAN)                           # the notebook tree
RESULTS = os.path.join(CLEAN, "results", "figure_construction", "structure")
FIGDIR = os.path.join(CLEAN, "figures", "figure_construction", "structure")

# ==========================================================================
# the system.  One flux quantum per electron at nu = 1, so the supercell area
# is N * 2*pi l_B^2 -- the same torus the API's own structure factors use.
# ==========================================================================
N_ELECTRONS = 36
CRYSTAL_INIT_ID = 0

#: The crystal of this diagnostic IS the LL-rotation crystal.  There is no
#: ``--ansatz`` flag: the historical fig06 crystal is a determinant of
#: site-centred magnetic Gaussians, a different wavefunction family this package
#: does not implement, and offering it as a choice here would invite reading a
#: picture of the wrong object.  See ``DIVERGENCE_fig06_crystal.md``.
DEFAULT_ANSATZ = "ll_rotation"

#: The liquid's shared parameter vector reserves room for an orbital sector it
#: does not have: the filled LLL is held at ``v = 0``, and ``c_row(0)`` is
#: exactly the first unit vector, so every higher band enters multiplied by
#: zero.  Measured, not argued -- a liquid walk at padding 1 and at padding 3
#: returns bit-identical energies.  There is deliberately no ``--liquid-nmax``.
LIQUID_NMAX = 1

#: The saved-point FORMAT.  ``load_point`` gates reuse on a digest that includes
#: it, so bumping this simply retires every older point instead of silently
#: reading a file whose metadata lacks fields this version writes.
POINT_SCHEMA = 1

# ==========================================================================
# this recipe's own defaults.  They are NOT a historical recipe: the point this
# script measures is the point the user passes, and the defaults exist only so
# that a bare invocation does something legible and says what it did.
# ==========================================================================
DEFAULT_LIQUID_RS = 55.0
DEFAULT_CRYSTAL_RS = 55.0
DEFAULT_CRYSTAL_NMAX = 2

# ==========================================================================
# the observable conventions.  EVERY constant here was read out of the
# historical source and the file:line is given, because the brief's numbers
# ("grid 72 x 72, kernel 0.40 l_B") are a description of a figure and had to be
# verified before being treated as exact.
#
#   density   `make_notebook.py:1577-1579` (NBINS_IV, W_KERNEL_IV, PAD_IV),
#             `make_notebook.py:5932-5934` (the periodic margin)
#   g(x,y)    `make_notebook.py:1569-1571`  (R_HALF = 4/sqrt(n)) and the
#             `pair_correlation_2d` call at `make_notebook_llrot.py:3848`
#   S(qx,qy)  `make_notebook.py:1570,1578`  (Q_HALF = 8*sqrt(n); NR, NQ = 81,160)
#             and the q -> 0 disc / peak-annulus scale at
#             `make_notebook_llrot.py:3880-3900`
#
# The two grids are genuinely different sizes and that is the historical choice,
# kept: the g map's cells collect ~85 pair counts each (11% Poisson) while S(q)
# averages hundreds of configurations, so the real-space map needs the finer
# mesh and the reciprocal one does not.
# ==========================================================================
DENSITY_NBINS = 72                 # bins per supercell vector
DENSITY_KERNEL_L_B = 0.40          # Gaussian width, l_B -- the width the figure is DRAWN at
DENSITY_MARGIN_BINS = 6            # bins of periodic margin drawn around the outlined cell

GXY_NR = 81                        # bins per axis in the g(x,y) map
GXY_SMOOTH = 1.0                   # Gaussian kernel, in g-map cells (0 = raw histogram)
GXY_RMAX_NUM = 4.0                 # R_HALF = 4/sqrt(n) ~ 10.03 l_B on this torus

SQ_NQ = 160                        # grid points per axis in the S(qx,qy) map
SQ_QHALF_NUM = 8.0                 # Q_HALF = 8*sqrt(n) ~ 3.19 l_B^-1 on this torus

#: The colour scale of the S(qx,qy) panels is set from this annulus alone, which
#: is where the physics is: the q -> 0 disc is a constant of size ne, and letting
#: it set the scale would crush every real peak into the dark end.
SQ_ANNULUS = (0.75, 1.25)          # in units of |g1|, the first WC Bragg vector

GR_RMAX = 4.0                      # the 1-D radial g(r) that supplies g(r_nn)
GR_NBINS = 80

DEFAULT_BUDGET = "quick"
QUALITY_MARK = "QUICK / not publication quality"


# ==========================================================================
# the request: slug, namespace, resolution
# ==========================================================================
def _fmt_rs(rs):
    """A compact r_s for a LABEL: ``55`` or ``5.657``."""
    rs = float(rs)
    if abs(rs - round(rs)) < 1e-9:
        return f"{rs:.0f}"
    return f"{rs:.3f}"


def _slug_rs(rs):
    """A compact, deterministic, filename-safe r_s: ``55``, ``5.65685``.

    Six significant digits, so a path stays readable while two couplings a user
    would plausibly type stay apart.  It is not a perfect key and does not have
    to be: the slug only NAMES the directory, ``--resume`` checks the
    full-precision digest before reusing anything, and ``main`` refuses to
    overwrite a directory whose recorded request differs from this one.
    """
    rs = float(rs)
    if abs(rs - round(rs)) < 1e-9:
        return f"{rs:.0f}"
    return f"{rs:.6g}"


def request_slug(liquid_rs, crystal_rs, ansatz, nmax, budget):
    """The directory name for a point: every input that changes the numbers.

    The two couplings, the crystal ansatz and the crystal truncation are
    physical; the budget is the protocol, and it changes the numbers just as
    surely, because ``quick`` and ``reproduction`` walk the same state for
    different lengths and start the crystal from different Gaussian widths.
    Leaving the budget out is how a smoke run comes to overwrite a delivered
    reproduction -- which is exactly what happened once on the sibling recipe.
    """
    tag = {"ll_rotation": "llrot", "ll_rotation_pinned": "llrot_pinned"}.get(
        str(ansatz), str(ansatz))
    return (f"liquid_rs{_slug_rs(liquid_rs)}__crystal_rs{_slug_rs(crystal_rs)}"
            f"__{tag}_nmax{int(nmax)}__{budget}")


def namespace(base, liquid_rs, crystal_rs, ansatz, nmax, budget):
    """``<base>/<slug>``.  A subdirectory, not a suffix: this recipe's whole
    output set belongs to one point, and grouping them keeps
    ``.../structure/`` a list of the points that have been measured."""
    return os.path.join(base, request_slug(liquid_rs, crystal_rs, ansatz,
                                           nmax, budget))


def comparison_kind(liquid_rs, crystal_rs):
    """§17: say which comparison this is, on the figure and in the metadata.

    Exact float equality, deliberately.  ``55`` and ``55.000000001`` are two
    different calculations and a reader told they are "the same coupling" would
    be reading a comparison of two couplings.
    """
    if float(liquid_rs) == float(crystal_rs):
        return "same-coupling comparison"
    return "different-coupling comparison"


def _phase_kappa(rs):
    """kappa = r_s / sqrt(2), the PHYSICAL mode, at full precision."""
    return float(rs) / math.sqrt(2.0)


def resolve_request(args):
    """The four resolved inputs, from parsed command-line arguments.

    Split out of ``main`` so that what the flags RESOLVE to is testable without
    paying for a VMC run -- and so that there is exactly one place that decides
    whether a value was the user's choice or the recipe's.  ``None`` means "not
    given", which is kept distinct from "given the default value": the
    pre-flight reports which parameters the user actually chose, and a user who
    typed a coupling that happens to equal the default has still made a choice.
    """
    return {
        "liquid_rs": (DEFAULT_LIQUID_RS if args.liquid_rs is None
                      else float(args.liquid_rs)),
        "crystal_rs": (DEFAULT_CRYSTAL_RS if args.crystal_rs is None
                       else float(args.crystal_rs)),
        "crystal_nmax": (DEFAULT_CRYSTAL_NMAX if args.crystal_nmax is None
                         else int(args.crystal_nmax)),
        "ansatz": DEFAULT_ANSATZ,
        "budget": str(args.budget),
        "liquid_rs_given": args.liquid_rs is not None,
        "crystal_rs_given": args.crystal_rs is not None,
        "crystal_nmax_given": args.crystal_nmax is not None,
    }


def validate_inputs(liquid_rs, crystal_rs, crystal_nmax):
    """Refuse a request the engine cannot honour -- BEFORE anything is computed.

    The bounds are the ENGINE'S, not invented here.  ``api.resolve`` and
    ``VMC.__init__`` both require ``nmax >= 1``, because this package's ``nmax``
    is the highest Landau *index* (``LandauLevelBasis`` is built with ``nmax =
    n_bands - 1``), so ``nmax = 0`` would ask for a single band and the ladder
    operators need the padding band above it.  There is no upper bound in the
    engine, so none is imposed: the cost of a large ``nmax`` is real (the Bloch
    sum runs over every band) and it is the user's to spend.

    Raising ``SystemExit`` rather than returning a flag is deliberate: this is a
    CLI contract, and the failure has to land before the SR ladder, not after it.
    """
    problems = []
    for flag, value in (("--liquid-rs", liquid_rs), ("--crystal-rs", crystal_rs)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            problems.append(f"{flag} must be finite and > 0, got {value!r}")
    if int(crystal_nmax) < 1:
        problems.append(
            f"--crystal-nmax must be >= 1, got {crystal_nmax!r}.  The engine's "
            f"nmax is the highest Landau index (n = 0 ... nmax), so 0 would mean "
            f"a single band, which the Landau-level basis cannot build.")
    if problems:
        raise SystemExit("structure: invalid request\n  "
                         + "\n  ".join(problems))


# ==========================================================================
# the calculation
# ==========================================================================
def run_point(phase, rs, init_id, budget, ansatz=DEFAULT_ANSATZ, nmax=1,
              verbose=True):
    """One cold VMC run through the public API.  Nothing is loaded.

    ``ansatz`` and ``nmax`` are crystal choices; the liquid has no variational
    orbital sector, so both are forced to their liquid values here rather than
    passed on and rejected by the API.  That forcing is the point: the two
    phases do not share a parameter space, and a caller cannot accidentally hand
    the crystal's truncation to the liquid.
    """
    if phase != "crystal":
        ansatz, nmax = DEFAULT_ANSATZ, LIQUID_NMAX
    vmc = VMC(N=N_ELECTRONS, rs=rs, phase=phase, nmax=int(nmax), ansatz=ansatz)
    t0 = time.time()
    result = vmc.run(init_id=init_id, budget=budget, verbose=verbose)
    snaps = [np.asarray(R, float) for R in result.state.snaps]

    if phase == "crystal":
        orbital_ansatz = str(result.state.ansatz)
        orbital_sr = (result.optimization.kind == "joint")
        ll_indices = list(range(int(result.state.nmax) + 1))
    else:
        orbital_ansatz = "filled_lll_fixed"
        orbital_sr = False
        ll_indices = [0]

    return {
        "phase": phase,
        "rs": float(rs),
        "kappa": float(result.state.kappa),
        "kappa_mode": str(result.state.kappa_mode),
        "orbital_ansatz": orbital_ansatz,
        "orbital_sr": bool(orbital_sr),
        "jastrow_sr": True,
        "nmax": int(result.state.nmax),
        "nmax_is_a_physical_parameter": phase == "crystal",
        "n_bands": int(result.state.n_bands),
        "ll_indices": ll_indices,
        "ansatz": str(result.state.ansatz),
        "optimized": ("orbitals+jastrow" if result.optimization.kind == "joint"
                      else "jastrow only (orbitals pinned)"),
        "init_id": int(init_id),
        "init_L0": (float(result.budget.width_for(init_id))
                    if phase == "crystal" else None),
        "budget": str(budget),
        # Two different things and both recorded: the SR/immediate seed the
        # engine's run config carries, and the production walk's seed, which
        # comes from the budget's measure block for this phase.
        "rng_seed": int(result.config.rng_seed),
        "walk_seed": int(result.budget.liquid_measure["seed"] if phase != "crystal"
                         else result.budget.crystal_measure["seed"]),
        "energy_per_particle": float(result.energy_per_particle),
        "error": float(result.error),
        "acceptance": float(result.acceptance),
        "n_snapshots": int(len(snaps)),
        "sr_steps": int(result.optimization.steps),
        "sr_seconds": float(result.optimization.seconds),
        "seconds": float(time.time() - t0),
        "snaps": snaps,
        "R": np.asarray(result.state.R, float),
    }


def torus_for(n_electrons=N_ELECTRONS):
    """The supercell Torus: one flux quantum per electron at nu = 1, so the area
    is ``N * 2*pi`` l_B^2.  The same object the API's own ``structure_factor``
    builds, derived from the same closed form."""
    return st.Torus(2.0 * math.pi * float(n_electrons), n_electrons=int(n_electrons),
                    n_cells_per_side=int(round(math.sqrt(n_electrons))))


# ==========================================================================
# the three observables.  Every one of them is a CALL into the validated
# analysis layer with chosen arguments -- this file computes no physics of its
# own, and duplicates no sampler, no structure factor and no density estimator.
# ==========================================================================
def density_field(snaps, torus, nbins=DENSITY_NBINS,
                  kernel_l_b=DENSITY_KERNEL_L_B,
                  margin_bins=DENSITY_MARGIN_BINS):
    """``n(x,y)``: the real-space density, smoothed, on the periodic torus.

    Grid ``nbins x nbins`` over the supercell, a Gaussian of ``kernel_l_b``
    applied with PERIODIC wrapping -- the only correct boundary on a torus, and
    a held-edge kernel would make the cell boundary a place where the density
    appears to change for a reason that is not physical.

    Returned in units of the cell mean ``n = ne/area``, so a uniform liquid
    reads 1.0 and the two panels are directly comparable; the density itself is
    ``rho * n_mean``.  ``rho`` is the DRAWN array: the cell plus ``margin_bins``
    of periodic margin, so a peak sitting on the cell edge is drawn whole rather
    than as a half-blob clipped by the boundary.
    """
    bin_l_b = float(np.linalg.norm(torus.L1)) / nbins
    n_mean = float(torus.ne) / float(torus.area)
    cell = st.density_grid(snaps, torus, nbins=nbins) / n_mean
    cell = st.periodic_gaussian_blur(cell, kernel_l_b / bin_l_b)
    rho = np.pad(cell, margin_bins, mode="wrap")

    f = np.arange(-margin_bins, nbins + margin_bins + 1) / nbins      # cell edges
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    cart = np.stack([F1, F2], axis=-1) @ torus.sc.T
    outline = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                        [0.0, 0.0]]) @ torus.sc.T
    return {"x": cart[..., 0], "y": cart[..., 1], "rho": rho,
            "rho_cell": cell, "frac": f, "outline": outline,
            "cell_vectors": {"A1": torus.A1, "A2": torus.A2,
                             "L1": torus.L1, "L2": torus.L2},
            "sites_frac": st.lattice_sites(torus),
            "sites_cart": st.lattice_sites(torus) @ torus.sc.T,
            "nbins": nbins, "bin_l_b": bin_l_b, "kernel_l_b": kernel_l_b,
            "margin_bins": margin_bins, "n_mean": n_mean}


def pair_correlation_field(snaps, torus, nr=GXY_NR, smooth=GXY_SMOOTH,
                           rmax=None):
    """``g(x,y)``: the displacement-resolved pair correlation.

    Two-dimensional and NOT angularly averaged -- the angular average at |r|
    folds the triangular lattice's six first-shell peaks into one number and
    throws away exactly the anisotropy this diagnostic is for.

    The window is ``R_HALF = 4/sqrt(n)`` (10.03 l_B here), wider than the
    Wigner-Seitz inradius (8.1 l_B), so the map uses periodic IMAGES and not the
    minimum image: minimum-imaging inside a window this size would fold the
    crystal's lattice peaks back at the cell boundary.  Every pair enters in
    both orientations, and the normalisation doubles with them, so a uniform
    liquid gives g = 1.
    """
    rmax = (GXY_RMAX_NUM / torus.sqrt_n) if rmax is None else float(rmax)
    centres, g = st.pair_correlation_2d(snaps, torus, rmax, nr, smooth)
    gx, gy = np.meshgrid(centres, centres, indexing="ij")
    return {"x": gx, "y": gy, "g": g, "centres": centres, "nr": nr,
            "rmax": rmax, "smooth": smooth, "bin_l_b": 2.0 * rmax / nr}


def nearest_neighbour_height(snaps, torus, rmax=GR_RMAX, nbins=GR_NBINS):
    """``g(r_nn)``, the first-shell height, for the panel annotation.

    Taken from the EXISTING radial estimator with this project's own shell rule
    (the maximum of g(r) within half a lattice constant of ``a_WC = |g1|``), so
    the annotation is a number the analysis layer already defines rather than a
    new order parameter invented to imitate an old figure.
    """
    r, g, _ = st.pair_correlation(snaps, torus, rmax, nbins)
    g1 = float(np.linalg.norm(torus.g1))
    m = np.abs(r - g1) < 0.5 * g1
    return float(g[m].max()) if m.any() else float("nan")


def structure_factor_field(snaps, torus, nq=SQ_NQ, q_half=None):
    """``S(qx,qy)`` on a reciprocal-space grid, and the masks the figure needs.

    ``S(q) = <|rho_q|^2>/ne`` averaged over configurations -- the average of
    ``|rho_q|^2`` and NOT of ``rho_q``: a filled Landau level has uniform
    density, so ``|<rho_q>|^2`` vanishes at every q != 0 and averaging the
    amplitude first would draw an empty panel.

    The returned ``S`` is the RAW map.  Two regions are recorded rather than
    baked in, because they are display conventions and a reader deserves to be
    able to undo them:

    * ``disc``  -- ``|q| < |G1|``.  The estimator returns exactly ``ne`` at q = 0
      for EVERY state (a sum of ne unit phases over ne, which the state cannot
      enter), so that disc is kinematics and not structure.  No allowed torus
      momentum other than q = 0 lies inside it, so masking it discards nothing
      physical; leaving it in would let a constant of size 36 set the colour
      scale against peaks of order 9.  It is masked in the FIGURE only.
    * ``annulus`` -- 0.75 to 1.25 ``|g1|``, the first-shell annulus the figure's
      common colour scale is set from.
    * ``inner`` -- ``|q| < 1/|L1|``, the sliver where ``q`` is small enough that
      the finite torus still reads as a coherent sum: there the estimator is
      already at ``ne`` and has not begun to fall towards the ``O(1)`` it has
      reached by the edge of the disc.  Recorded so the metadata can say what
      the masked disc actually holds instead of implying it is all ``ne``.
    """
    q_half = (SQ_QHALF_NUM * torus.sqrt_n) if q_half is None else float(q_half)
    ql = np.linspace(-q_half, q_half, nq)
    qx, qy = np.meshgrid(ql, ql, indexing="ij")
    S = st.structure_factor_2d(snaps, qx, qy, int(torus.ne))
    qm = np.hypot(qx, qy)
    G1 = float(np.linalg.norm(torus.G1))
    g1 = float(np.linalg.norm(torus.g1))
    return {"qx": qx, "qy": qy, "S": S, "q_half": q_half, "nq": nq,
            "disc": qm < G1, "annulus": (qm > SQ_ANNULUS[0] * g1)
            & (qm < SQ_ANNULUS[1] * g1),
            "inner": qm < (1.0 / float(np.linalg.norm(torus.L1))),
            "bragg": torus.wc_shell_vectors(), "G1_mag": G1, "g1_mag": g1}


def fields_for(snaps, torus):
    """All three observables for one phase, from that phase's own snapshots."""
    return {"density": density_field(snaps, torus),
            "gxy": pair_correlation_field(snaps, torus),
            "sq": structure_factor_field(snaps, torus),
            "g_rn": nearest_neighbour_height(snaps, torus)}


# ==========================================================================
# pre-flight.  The resolved calculation, in full, BEFORE anything expensive.
# ==========================================================================
def print_preflight(liquid_rs, crystal_rs, crystal_nmax, bud, results_dir,
                    figdir):
    """The resolved structure calculation, printed before the first SR step.

    Two phase blocks because the two phases do not have the same parameters: an
    ``nmax`` printed for the liquid as though it were liquid physics is the
    confusion this block exists to prevent.  The OBSERVABLES block names each
    observable's grid, kernel and normalisation, because those are the choices
    that make two runs comparable or not.
    """
    w = 74
    torus = torus_for()
    print("=" * w)
    print("RESOLVED STRUCTURE CALCULATION")
    print("=" * w)
    print()
    print("LIQUID")
    print(f"  rs               {float(liquid_rs):.9f}")
    print(f"  kappa            {_phase_kappa(liquid_rs):.9f}   [physical: rs/sqrt(2)]")
    print(f"  orbital ansatz   filled LLL (fixed)")
    print(f"  LL rotation      none")
    print(f"  orbital SR       disabled")
    print(f"  Jastrow SR       enabled")
    print(f"  walk seed        {bud.liquid_measure['seed']}")
    print()
    print("CRYSTAL")
    print(f"  rs               {float(crystal_rs):.9f}")
    print(f"  kappa            {_phase_kappa(crystal_rs):.9f}   [physical: rs/sqrt(2)]")
    print(f"  orbital ansatz   LL rotation")
    print(f"  n_max            {crystal_nmax}")
    print(f"  LL basis         n = {', '.join(str(n) for n in range(int(crystal_nmax) + 1))}")
    print(f"  n_bands          {int(crystal_nmax) + 1}   (= n_max + 1)")
    print(f"  orbital SR       enabled")
    print(f"  Jastrow SR       enabled")
    print(f"  optimisation     joint orbital + Jastrow SR")
    print(f"  init L0          {bud.width_for(CRYSTAL_INIT_ID):g}   "
          f"(init_id {CRYSTAL_INIT_ID}, budget {bud.name!r})")
    print(f"  walk seed        {bud.crystal_measure['seed']}")
    print()
    print("OBSERVABLES")
    bin_l_b = float(np.linalg.norm(torus.L1)) / DENSITY_NBINS
    print(f"  density n(x,y)   grid {DENSITY_NBINS}x{DENSITY_NBINS} on the supercell, "
          f"bin {bin_l_b:.4f} l_B")
    print(f"                   Gaussian kernel {DENSITY_KERNEL_L_B:.2f} l_B, "
          f"PERIODIC wrap, in units of the mean n")
    print(f"                   drawn with {DENSITY_MARGIN_BINS} bins of periodic "
          f"margin, one cell outlined")
    r_half = GXY_RMAX_NUM / torus.sqrt_n
    print(f"  pair corr g(x,y) grid {GXY_NR}x{GXY_NR} over +/-{r_half:.2f} l_B, "
          f"bin {2 * r_half / GXY_NR:.4f} l_B")
    print(f"                   periodic IMAGES (window > the WS inradius), "
          f"g = 1 for a uniform liquid")
    q_half = SQ_QHALF_NUM * torus.sqrt_n
    print(f"  structure S(q)   grid {SQ_NQ}x{SQ_NQ} over +/-{q_half:.4f} l_B^-1")
    print(f"                   S = <|rho_q|^2>/ne; the disc |q| < |G1| = "
          f"{float(np.linalg.norm(torus.G1)):.4f} is masked in the figure only")
    print(f"                   ({float(np.linalg.norm(torus.g1)):.4f} = |g1|, "
          f"the six first-shell Q_WC, marked")
    print(f"                   with open circles)")
    print()
    print("BUDGET")
    print(f"  {bud.name}   source: {bud.provenance}")
    print(f"  SR               {bud.protocol['sr_steps']}x{bud.protocol['sr_sweeps']} "
          f"(equil {bud.protocol['sr_equil']}, snapshot_every {bud.protocol['sr_snap']})")
    print(f"  production walk  {bud.protocol['meas_sweeps']}/{bud.protocol['meas_equil']} "
          f"(sweeps/equil)")
    print()
    print("OUTPUT")
    print(f"  results          {_rel(results_dir)}")
    print(f"  figures          {_rel(figdir)}")
    print(f"  comparison       {comparison_kind(liquid_rs, crystal_rs)}")
    print("=" * w)
    print()


def check_namespace_reuse(results_dir, slug):
    """Warn if this directory already holds a DIFFERENT request.

    The slug is short enough to read and therefore short enough, in principle, to
    collide: two couplings that round to the same six significant digits land on
    the same path.  ``--resume`` is safe regardless (it checks the full-precision
    digest), but a plain run would overwrite in silence, so this says so first.
    """
    path = os.path.join(results_dir, "run_metadata.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            prior = json.load(fh).get("request_slug")
    except (OSError, ValueError):
        return
    if prior is not None and prior != slug:
        print(f"  WARNING: {_rel(results_dir)} already holds a different request")
        print(f"           saved as {prior!r}; this run is {slug!r} and will")
        print(f"           overwrite it.  Use --resume to be told instead.")
        print()


# ==========================================================================
# persistence.  Numbers on disk, not only pixels.
# ==========================================================================
def _sha(payload):
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _rel(path, start=CLEAN):
    """``os.path.relpath`` that survives a different drive.

    On Windows ``relpath`` raises ``ValueError`` when the two paths are on
    different mounts, which would turn a purely cosmetic line of the banner into
    a crash for anyone who points the outputs at another drive.
    """
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def _source_id():
    """A source identifier for the tree that produced this run, or an honest gap."""
    if os.path.isdir(os.path.join(CLEAN, ".git")):
        import subprocess
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLEAN,
                                 capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                return "git:" + out.stdout.strip()
        except OSError:
            pass
        return f"git repository present, HEAD unreadable; wigner_vmc {__version__}"
    return (f"not a git repository (no .git at the checkout root); "
            f"wigner_vmc {__version__}")


def _file_sha(path):
    """sha256 of a file's BYTES, or ``None`` if it is not there.

    Bytes and not text: this has to be able to tell two states of the source
    apart, and a line-ending or encoding change is a different file to anything
    that reads it.
    """
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _source_hashes(budget):
    """Content hashes of the code that produced this run.

    ``_source_id`` names the tree; this identifies its CONTENT, which matters
    because this checkout is not a git repository -- ``wigner_vmc 0.1.0`` alone
    cannot tell two states of the package apart.  A reader can recompute these
    and know whether the numbers below came from the recipe they are holding.

    The API and the structure module are hashed because they are where the
    physics and the estimator conventions actually live; the recipe calls them
    and owns neither.  The budget file is hashed because the protocol is part of
    what a run means.
    """
    pkg = os.path.dirname(os.path.abspath(wigner_vmc.__file__))
    return {
        "recipe": _file_sha(os.path.join(HERE, "structure.py")),
        "wigner_vmc/api.py": _file_sha(os.path.join(pkg, "api.py")),
        "wigner_vmc/analysis/structure.py":
            _file_sha(os.path.join(pkg, "analysis", "structure.py")),
        f"configs/{budget}.yaml":
            _file_sha(os.path.join(CLEAN, "configs", f"{budget}.yaml")),
    }


def _config_key(phase, rs, init_id, budget, ansatz=DEFAULT_ANSATZ, nmax=1):
    return {"phase": phase, "rs": repr(float(rs)), "N": N_ELECTRONS,
            "nmax": int(nmax), "ansatz": str(ansatz), "init_id": int(init_id),
            "budget": str(budget), "point_schema": POINT_SCHEMA}


def save_point(outdir, name, point, key):
    """``<outdir>/<name>/run.npz`` + ``run.json``.  The json carries the config
    digest ``--resume`` checks before it will reuse anything."""
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "run.npz"),
                        snaps=np.asarray(point["snaps"], float),
                        R=np.asarray(point["R"], float))
    meta = {k: v for k, v in point.items() if k not in ("snaps", "R")}
    meta["config_key"] = key
    meta["config_sha"] = _sha(key)
    meta["wigner_vmc_version"] = __version__
    with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return d


def load_point(outdir, name, key):
    """Reload a saved point, but ONLY if it was produced by this exact request.

    A resume that silently reused a run from a different coupling, budget or
    starting point would be worse than no resume: the figure would carry one
    state's label on another state's data.  A digest mismatch is therefore
    treated as "not resumable" rather than as an error.
    """
    d = os.path.join(outdir, name)
    jf, nf = os.path.join(d, "run.json"), os.path.join(d, "run.npz")
    if not (os.path.exists(jf) and os.path.exists(nf)):
        return None
    with open(jf, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("config_sha") != _sha(key):
        return None
    with np.load(nf) as z:
        point = dict(meta)
        point["snaps"] = [np.asarray(R, float) for R in z["snaps"]]
        point["R"] = np.asarray(z["R"], float)
    return point


def save_observables(results_dir, phases, extra):
    """The three numerical products plus ``run_metadata.json``.

    PNGs are not a scientific record: a reader has to be able to re-plot,
    re-normalise and re-analyse without re-running the walk, so every array the
    figures draw is written here beside the metadata that says what it is.
    """
    os.makedirs(results_dir, exist_ok=True)
    den_l, den_c = phases["liquid"]["density"], phases["crystal"]["density"]
    gxy_l, gxy_c = phases["liquid"]["gxy"], phases["crystal"]["gxy"]
    sq_l, sq_c = phases["liquid"]["sq"], phases["crystal"]["sq"]

    np.savez_compressed(
        os.path.join(results_dir, "density_xy.npz"),
        x=den_l["x"], y=den_l["y"], frac=den_l["frac"],
        density_liquid=den_l["rho"], density_crystal=den_c["rho"],
        density_liquid_cell=den_l["rho_cell"],
        density_crystal_cell=den_c["rho_cell"],
        A1=den_l["cell_vectors"]["A1"], A2=den_l["cell_vectors"]["A2"],
        L1=den_l["cell_vectors"]["L1"], L2=den_l["cell_vectors"]["L2"],
        cell_outline=den_l["outline"],
        lattice_sites_frac=den_l["sites_frac"],
        lattice_sites_cart=den_l["sites_cart"],
        nbins=np.asarray(den_l["nbins"]), bin_l_b=np.asarray(den_l["bin_l_b"]),
        kernel_l_b=np.asarray(den_l["kernel_l_b"]),
        margin_bins=np.asarray(den_l["margin_bins"]),
        n_mean=np.asarray(den_l["n_mean"]),
        n_electrons=np.asarray(N_ELECTRONS))

    np.savez_compressed(
        os.path.join(results_dir, "pair_correlation_xy.npz"),
        x=gxy_l["x"], y=gxy_l["y"], centres=gxy_l["centres"],
        g_liquid=gxy_l["g"], g_crystal=gxy_c["g"],
        g_rn_liquid=np.asarray(phases["liquid"]["g_rn"]),
        g_rn_crystal=np.asarray(phases["crystal"]["g_rn"]),
        rmax=np.asarray(gxy_l["rmax"]), nr=np.asarray(gxy_l["nr"]),
        smooth=np.asarray(gxy_l["smooth"]), bin_l_b=np.asarray(gxy_l["bin_l_b"]))

    np.savez_compressed(
        os.path.join(results_dir, "structure_factor_xy.npz"),
        qx=sq_l["qx"], qy=sq_l["qy"],
        S_liquid=sq_l["S"], S_crystal=sq_c["S"],
        mask_q0_disc=sq_l["disc"], mask_peak_annulus=sq_l["annulus"],
        mask_q0_inner=sq_l["inner"],
        bragg_vectors=sq_l["bragg"],
        G1_mag=np.asarray(sq_l["G1_mag"]), g1_mag=np.asarray(sq_l["g1_mag"]),
        q_half=np.asarray(sq_l["q_half"]), nq=np.asarray(sq_l["nq"]))

    meta = dict(extra)
    meta["run_metadata_schema"] = "figure_construction/structure/1"
    for name in ("liquid", "crystal"):
        p = dict(phases[name]["point"])
        p.pop("snaps", None)
        p.pop("R", None)
        meta[name] = p
    with open(os.path.join(results_dir, "run_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return meta


# ==========================================================================
# figures
# ==========================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _quick_banner(ax, budget):
    """Stamp a non-production figure, INSIDE the axes.

    Above the axes it collides with the panel title, which is how a quick run
    and a production run come to look the same -- the one outcome that would
    make this recipe misleading.  Inside, on an opaque box, it is legible over
    whatever the data does.
    """
    if budget == "reproduction":
        return
    ax.text(0.985, 0.975, "QUICK\nnot publication quality", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color="crimson", linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor="crimson", alpha=0.85,
                      boxstyle="round,pad=0.25", linewidth=0.6))


def _panel_title(phase, rs, nmax):
    """A panel is named by the parameters the user chose, ``r_s`` first."""
    if phase == "crystal":
        return f"LL crystal\n$r_s$={_fmt_rs(rs)}   $n_{{\\rm max}}$={int(nmax)}"
    return f"liquid\n$r_s$={_fmt_rs(rs)}"


def _footer(fig, ctx, extra="", bottom=0.03):
    """The head line, the observable's own conventions, and the layout they need.

    A single unwrapped ``suptitle`` was tried first and the observable line ran
    off the right edge of the canvas -- the reader lost exactly the sentence that
    says what the panel is a picture OF, which is the sentence most worth
    keeping.  So the text is wrapped and the axes are given a rectangle that
    RESERVES the rows it occupies, rather than laid over the plot and hoped for.
    """
    import textwrap
    head = (f"{ctx['comparison_kind']}:  liquid $r_s$={_fmt_rs(ctx['liquid_rs'])} "
            f"($\\kappa$={ctx['kappa_liquid']:.3f})   |   LL crystal "
            f"$r_s$={_fmt_rs(ctx['crystal_rs'])} "
            f"($\\kappa$={ctx['kappa_crystal']:.3f}), "
            f"$n_{{\\rm max}}$={int(ctx['crystal_nmax'])}")
    head_lines = textwrap.wrap(head, 112)
    fig.suptitle("\n".join(head_lines), fontsize=10.5, y=0.985, va="top")
    y = 0.985 - 0.045 * len(head_lines)

    if ctx["budget"] != "reproduction":
        extra = (extra + "   " if extra else "") + QUALITY_MARK
    if extra:
        note = textwrap.wrap(extra, 138)
        fig.text(0.5, y, "\n".join(note), ha="center", va="top", fontsize=8.5,
                 color="0.25")
        y -= 0.030 * len(note)

    fig.tight_layout(rect=(0, bottom, 1, y - 0.01))


def draw_density_xy(figdir, phases, ctx):
    """``density_xy.png``: two panels, ``liquid | crystal``.

    One colour scale PER PANEL, which is the historical density figure's own
    convention (one scale per row, set from that row's strongest panel) and is
    recorded in the metadata.  A common scale is the tempting alternative and it
    is the wrong one here: the liquid's modulation is a few per cent of the mean
    while the crystal's can be many times it, so one shared scale renders the
    liquid as a uniform square and destroys the very control the liquid panel
    exists to be.  The quantitative comparison the colour cannot carry --
    site/midpoint contrast -- is printed and stored instead.
    """
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.8))
    vmaxes = {}
    for ax, phase in zip(axs, ("liquid", "crystal")):
        f = phases[phase]["density"]
        vmax = float(np.percentile(f["rho"], 99.5))
        vmaxes[phase] = vmax
        m = ax.pcolormesh(f["x"], f["y"], f["rho"], cmap="viridis", vmin=0.0,
                          vmax=vmax, shading="flat", rasterized=True)
        # One cell outlined, with a periodic margin around it.  No site markers:
        # the peaks ARE the lattice sites, so a dot on each would restate the
        # definition on top of its own output and sit exactly on the maxima the
        # panel exists to show.
        ax.plot(f["outline"][:, 0], f["outline"][:, 1], "-", lw=1.2, color="w",
                alpha=0.9)
        ax.set_title(_panel_title(phase, ctx["%s_rs" % phase], ctx["crystal_nmax"]),
                     fontsize=10)
        ax.set_xlabel("$x\\,\\,[l_B]$", fontsize=9.5)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        _quick_banner(ax, ctx["budget"])
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("$n(x,y)\\,/\\,\\bar n$", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    axs[0].set_ylabel("$y\\,\\,[l_B]$", fontsize=9.5)
    _footer(fig, ctx, f"$n(x,y)$: grid {DENSITY_NBINS}$\\times${DENSITY_NBINS}, "
                      f"Gaussian {DENSITY_KERNEL_L_B:.2f} $l_B$ (periodic wrap); "
                      f"one cell outlined, periodic margin drawn; the two panels "
                      f"carry separate colour scales (see run_metadata.json)")
    path = os.path.join(figdir, "density_xy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, vmaxes


def draw_pair_correlation_xy(figdir, phases, ctx):
    """``pair_correlation_xy.png``: two panels, ``liquid | LL crystal``.

    ONE colour scale across both panels -- unlike the density figure, and for a
    reason: g(x,y) is normalised so a uniform liquid sits at exactly 1, so the
    two panels are already on the same scale and a shared bar lets the reader
    see that the crystal's first shell is a peak above 1 while the liquid's is a
    correlation hole below it.  Equal aspect, identical window and binning.
    """
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.8))
    vmax = max(float(np.nanmax(phases[p]["gxy"]["g"])) for p in ("liquid", "crystal"))
    for ax, phase in zip(axs, ("liquid", "crystal")):
        f = phases[phase]["gxy"]
        m = ax.pcolormesh(f["x"], f["y"], f["g"], cmap="viridis", vmin=0.0,
                          vmax=vmax, shading="auto", rasterized=True)
        ax.set_title(f"{_panel_title(phase, ctx['%s_rs' % phase], ctx['crystal_nmax'])}\n"
                     f"$g(r_{{\\rm nn}})$={phases[phase]['g_rn']:.2f}", fontsize=10)
        ax.set_xlabel("$x\\,\\,[l_B]$", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        ax.tick_params(labelsize=8)
        _quick_banner(ax, ctx["budget"])
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("$g(x,y)$", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    axs[0].set_ylabel("$y\\,\\,[l_B]$", fontsize=9.5)
    _footer(fig, ctx, "$g(x,y)$: common colour scale 0 to %.2f; window "
                      "$\\pm$%.2f $l_B$, grid %d$\\times$%d, Gaussian %.1f cell, "
                      "periodic images (g = 1 for a uniform liquid)"
            % (vmax, phases["liquid"]["gxy"]["rmax"], GXY_NR, GXY_NR, GXY_SMOOTH))
    path = os.path.join(figdir, "pair_correlation_xy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, vmax


def draw_structure_factor_xy(figdir, phases, ctx):
    """``structure_factor_xy.png``: two panels, ``liquid | LL crystal``.

    Identical q range and display convention in both panels, and one common
    colour scale set from the 0.75-1.25 ``|g1|`` annulus alone -- the historical
    convention.  The ``|q| < |G1|`` disc is drawn as a grey masked region rather
    than left in: it is the kinematic ``S(0) = ne`` and would spend most of the
    scale on a constant that does not depend on the state.  Every non-zero q is
    kept.
    """
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.0))
    ann_l = phases["liquid"]["sq"]["annulus"]
    vmax = max(float(np.nanmax(phases[p]["sq"]["S"][ann_l]))
               for p in ("liquid", "crystal"))
    for ax, phase in zip(axs, ("liquid", "crystal")):
        f = phases[phase]["sq"]
        S = np.where(f["disc"], np.nan, f["S"])
        ax.set_facecolor("0.88")
        m = ax.pcolormesh(f["qx"], f["qy"], S, cmap="inferno", vmin=0.0,
                          vmax=vmax, shading="auto", rasterized=True)
        ax.plot(f["bragg"][:, 0], f["bragg"][:, 1], "o", mfc="none", mec="cyan",
                ms=6.5, mew=1.1)
        ax.set_title(_panel_title(phase, ctx["%s_rs" % phase], ctx["crystal_nmax"]),
                     fontsize=10)
        ax.set_xlabel("$q_x\\,\\,[l_B^{-1}]$", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelsize=8)
        _quick_banner(ax, ctx["budget"])
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("$S(q_x,q_y)$", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    axs[0].set_ylabel("$q_y\\,\\,[l_B^{-1}]$", fontsize=9.5)
    _footer(fig, ctx, "$S(q_x,q_y)$: common scale 0 to %.2f, set from the "
                      "%.2f-%.2f $|g_1|$ annulus; grey disc is $|q|<|G_1|$, "
                      "where $S(q\\to0)=n_e$ is kinematic; open circles mark the "
                      "six $Q_{\\rm WC}$" % (vmax, SQ_ANNULUS[0], SQ_ANNULUS[1]))
    path = os.path.join(figdir, "structure_factor_xy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, vmax


def draw_combined_summary(figdir, phases, ctx):
    """``combined_structure_summary.png``: row 1 liquid, row 2 crystal; column 1
    ``n(x,y)``, column 2 ``g(x,y)``, column 3 ``S(qx,qy)``.  ONE parameter pair.

    A convenience view; the three standalone figures remain the primary outputs.
    Each column carries the scale its own standalone figure carries, so the two
    cannot disagree.
    """
    plt = _mpl()
    from matplotlib.colors import Normalize
    fig, axs = plt.subplots(2, 3, figsize=(17.0, 10.4))
    vmax_den = {p: float(np.percentile(phases[p]["density"]["rho"], 99.5))
                for p in ("liquid", "crystal")}
    vmax_g = max(float(np.nanmax(phases[p]["gxy"]["g"])) for p in ("liquid", "crystal"))
    ann = phases["liquid"]["sq"]["annulus"]
    vmax_s = max(float(np.nanmax(phases[p]["sq"]["S"][ann]))
                 for p in ("liquid", "crystal"))
    for r, phase in enumerate(("liquid", "crystal")):
        f = phases[phase]["density"]
        m0 = axs[r, 0].pcolormesh(f["x"], f["y"], f["rho"], cmap="viridis", vmin=0.0,
                                  vmax=vmax_den[phase], shading="flat", rasterized=True)
        axs[r, 0].plot(f["outline"][:, 0], f["outline"][:, 1], "-", lw=0.8,
                       color="w", alpha=0.55)

        g = phases[phase]["gxy"]
        m1 = axs[r, 1].pcolormesh(g["x"], g["y"], g["g"], cmap="viridis", vmin=0.0,
                                  vmax=vmax_g, shading="auto", rasterized=True)

        s = phases[phase]["sq"]
        m2 = axs[r, 2].pcolormesh(s["qx"], s["qy"], np.where(s["disc"], np.nan, s["S"]),
                                  cmap="inferno", vmin=0.0, vmax=vmax_s,
                                  shading="auto", rasterized=True)
        axs[r, 2].set_facecolor("0.88")
        axs[r, 2].plot(s["bragg"][:, 0], s["bragg"][:, 1], "o", mfc="none",
                       mec="cyan", ms=5.5, mew=1.0)

        for c, m in enumerate((m0, m1, m2)):
            axs[r, c].set_aspect("equal")
            axs[r, c].tick_params(labelsize=7.5)
            _quick_banner(axs[r, c], ctx["budget"])
            fig.colorbar(m, ax=axs[r, c], fraction=0.046, pad=0.03).ax.tick_params(
                labelsize=7)
        axs[r, 0].set_ylabel(f"{_panel_title(phase, ctx['%s_rs' % phase], ctx['crystal_nmax'])}\n"
                             "$y\\,\\,[l_B]$", fontsize=9)
        axs[r, 0].set_xlabel("$x\\,\\,[l_B]$", fontsize=8.5)
        axs[r, 1].set_xlabel("$x\\,\\,[l_B]$", fontsize=8.5)
        axs[r, 2].set_xlabel("$q_x\\,\\,[l_B^{-1}]$", fontsize=8.5)
    axs[0, 0].set_title("$n(x,y)$", fontsize=11)
    axs[0, 1].set_title("$g(x,y)$", fontsize=11)
    axs[0, 2].set_title("$S(q_x,q_y)$", fontsize=11)
    _footer(fig, ctx, "one parameter pair; the three standalone figures are the "
                      "primary outputs, and each column carries that figure's own "
                      "scale")
    path = os.path.join(figdir, "combined_structure_summary.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_figures(figdir, phases, ctx):
    """All four figures.  Returns the paths written."""
    os.makedirs(figdir, exist_ok=True)
    written = []
    p, vmax_den = draw_density_xy(figdir, phases, ctx)
    written.append(p)
    ctx["density_vmax"] = vmax_den
    p, vmax_g = draw_pair_correlation_xy(figdir, phases, ctx)
    written.append(p)
    ctx["gxy_vmax"] = vmax_g
    p, vmax_s = draw_structure_factor_xy(figdir, phases, ctx)
    written.append(p)
    ctx["sq_vmax"] = vmax_s
    written.append(draw_combined_summary(figdir, phases, ctx))
    return written


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="One structural point: (liquid r_s, crystal r_s, crystal n_max) -> "
                    "clean VMC -> {n(x,y), g(x,y), S(qx,qy)}.  One invocation is one "
                    "point; there is no coupling sweep.")
    p.add_argument("--liquid-rs", type=float, default=None, metavar="RS",
                   help=f"Wigner-Seitz radius of the LIQUID, in units of l_B.  "
                        f"Default {DEFAULT_LIQUID_RS:g} (this recipe's own default; it "
                        f"is not a historical coupling).")
    p.add_argument("--crystal-rs", type=float, default=None, metavar="RS",
                   help=f"Wigner-Seitz radius of the CRYSTAL, in units of l_B.  "
                        f"Default {DEFAULT_CRYSTAL_RS:g}.  Equal to --liquid-rs gives a "
                        f"same-coupling comparison, which the figure says so.")
    p.add_argument("--crystal-nmax", type=int, default=None, metavar="N",
                   help=f"Highest Landau level the CRYSTAL basis keeps: n = 0 ... N, so "
                        f"n_bands = N + 1.  Default {DEFAULT_CRYSTAL_NMAX}.  Must be "
                        f">= 1 (the engine's own bound -- its nmax is the highest index, "
                        f"not a band count).  There is deliberately no --liquid-nmax: "
                        f"the liquid is the filled LLL, fixed, and has no truncation.")
    p.add_argument("--budget", default=DEFAULT_BUDGET,
                   help="quick (smoke test / visual sanity check) | full "
                        "(higher-quality clean sampling -- an alias for the "
                        "'reproduction' config, which is also accepted by name).  "
                        "Any other name in configs/ is accepted; those two are the "
                        "documented ones.")
    p.add_argument("--resume", action="store_true",
                   help="reuse a saved clean run in this point's own results directory "
                        "when its recorded configuration -- including both couplings, "
                        "the ansatz, the truncation and the budget -- matches exactly.")
    p.add_argument("--quiet", action="store_true", help="suppress the per-run summaries.")
    return p.parse_args(argv)


# ==========================================================================
# main
# ==========================================================================
def main(argv=None):
    args = parse_args(argv)
    # Resolve the user-facing budget alias ONCE, before anything reads it, so
    # that `--budget full` and `--budget reproduction` are one calculation with
    # one slug and one output directory rather than two directories holding the
    # same run.
    args.budget = resolve_budget_name(args.budget)
    t0 = time.time()

    req = resolve_request(args)
    liquid_rs = req["liquid_rs"]
    crystal_rs = req["crystal_rs"]
    crystal_nmax = req["crystal_nmax"]
    ansatz = req["ansatz"]

    # Both of these are cheap and both must fail BEFORE the first SR step: an
    # invalid request that dies after a 10x40 ladder has already been paid for is
    # the same error, reported 200 seconds too late.
    validate_inputs(liquid_rs, crystal_rs, crystal_nmax)
    bud = load_budget(args.budget)          # no fallback table; names what exists

    slug = request_slug(liquid_rs, crystal_rs, ansatz, crystal_nmax, args.budget)
    results_dir = namespace(RESULTS, liquid_rs, crystal_rs, ansatz,
                            crystal_nmax, args.budget)
    figdir = namespace(FIGDIR, liquid_rs, crystal_rs, ansatz,
                       crystal_nmax, args.budget)

    print("structure -- clean VMC, from scratch   "
          f"(wigner_vmc {__version__})")
    print()
    print_preflight(liquid_rs, crystal_rs, crystal_nmax, bud, results_dir, figdir)
    check_namespace_reuse(results_dir, slug)

    phases = {}
    requests = (("liquid", "liquid", liquid_rs, 0),
                ("crystal", "crystal", crystal_rs, CRYSTAL_INIT_ID))
    for name, phase, rs, init_id in requests:
        # The liquid carries no crystal truncation: `run_point` forces it, and
        # the key is built from what the run will ACTUALLY use, so a saved point
        # can never be reused under a label it was not produced with.
        nmax = crystal_nmax if phase == "crystal" else LIQUID_NMAX
        key = _config_key(phase, rs, init_id, args.budget, ansatz=ansatz, nmax=nmax)
        print("-" * 74)
        print(f"{name}: {phase} at r_s = {rs:.9f} (kappa = {_phase_kappa(rs):.9f})")
        print("-" * 74)
        point = load_point(results_dir, name, key) if args.resume else None
        if point is not None:
            print(f"  resumed from {_rel(results_dir)}/{name}/ (config digest matches)")
        else:
            if args.resume:
                print("  no matching saved run; computing from scratch")
            point = run_point(phase, rs, init_id, args.budget, ansatz=ansatz,
                              nmax=nmax, verbose=not args.quiet)
        save_point(results_dir, name, point, key)
        phases[name] = {"point": point}
        print(f"  E/N {point['energy_per_particle']:+.9f} +- {point['error']:.9f}"
              f"   acc {point['acceptance']:.4f}   {point['n_snapshots']} snapshots"
              f"   {point['seconds']:.0f}s")

    # --- observables, from these snapshots and nothing else -----------------
    torus = torus_for()
    for name in ("liquid", "crystal"):
        phases[name].update(fields_for(phases[name]["point"]["snaps"], torus))

    ctx = {
        "liquid_rs": float(liquid_rs),
        "crystal_rs": float(crystal_rs),
        "crystal_nmax": int(crystal_nmax),
        "kappa_liquid": _phase_kappa(liquid_rs),
        "kappa_crystal": _phase_kappa(crystal_rs),
        "budget": str(args.budget),
        "comparison_kind": comparison_kind(liquid_rs, crystal_rs),
    }

    # --- the quantitative read-out the colour scales cannot carry -----------
    print("-" * 74)
    print("OBSERVABLES")
    print("-" * 74)
    for name in ("liquid", "crystal"):
        ph = phases[name]
        H = ph["density"]["rho_cell"]
        ph["contrast"] = st.site_midpoint_contrast(H, torus, DENSITY_NBINS)
        print(f"  {name:>7}: n(x,y) mean {float(H.mean()):.4f} (cell mean is 1), "
              f"range {float(H.min()):.3f} to {float(H.max()):.3f}")
        print(f"           site/midpoint contrast {ph['contrast']:.3f}"
              f"   (exactly 1 for anything uniform)")
        print(f"           g(r_nn) {ph['g_rn']:.3f}"
              f"   g(x,y) range {float(ph['gxy']['g'].min()):.3f} to "
              f"{float(ph['gxy']['g'].max()):.3f}")
        ann = ph["sq"]["annulus"]
        print(f"           S(qx,qy) range outside the q->0 disc "
              f"{float(ph['sq']['S'][~ph['sq']['disc']].min()):.3f} to "
              f"{float(ph['sq']['S'][~ph['sq']['disc']].max()):.3f}"
              f"   peak-annulus max {float(ph['sq']['S'][ann].max()):.3f}")
    print(f"  comparison: {ctx['comparison_kind']}")

    # --- draw first, so the metadata can record the scales actually used ----
    written = write_figures(figdir, phases, ctx)

    # --- persist the numbers the figures were drawn from --------------------
    extra = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "recipe": "examples/figure_construction/structure.py",
        "source_id": _source_id(),
        "source_hashes": _source_hashes(str(args.budget)),
        "wigner_vmc_version": __version__,
        "budget": str(args.budget),
        "budget_provenance": bud.provenance,
        "budget_protocol": {k: v for k, v in bud.protocol.items()},
        "quality": (QUALITY_MARK if args.budget != "reproduction" else "reproduction"),
        "N": N_ELECTRONS,
        "request_slug": slug,
        "comparison_kind": ctx["comparison_kind"],
        "comparison_rule": ("liquid_rs == crystal_rs, exact float equality"
                            if ctx["comparison_kind"].startswith("same")
                            else "liquid_rs != crystal_rs, exact float equality"),
        "liquid_rs": float(liquid_rs),
        "crystal_rs": float(crystal_rs),
        "liquid_rs_source": ("--liquid-rs" if req["liquid_rs_given"]
                             else f"recipe default ({DEFAULT_LIQUID_RS:g})"),
        "crystal_rs_source": ("--crystal-rs" if req["crystal_rs_given"]
                              else f"recipe default ({DEFAULT_CRYSTAL_RS:g})"),
        "crystal_nmax": int(crystal_nmax),
        "crystal_nmax_source": ("--crystal-nmax" if req["crystal_nmax_given"]
                                else f"recipe default ({DEFAULT_CRYSTAL_NMAX})"),
        "crystal_n_bands": int(crystal_nmax) + 1,
        "crystal_ll_indices": list(range(int(crystal_nmax) + 1)),
        "crystal_ansatz": ansatz,
        "liquid_ansatz": "filled_lll_fixed",
        "liquid_nmax_technical": LIQUID_NMAX,
        "liquid_nmax_is_a_physical_parameter": False,
        "liquid_ll_indices": [0],
        "kappa_mode": ("physical (rs/sqrt(2)); the Stage-2E rounded regression mode "
                       "is not used by this recipe"),
        "kappa_liquid": _phase_kappa(liquid_rs),
        "kappa_crystal": _phase_kappa(crystal_rs),
        "torus_area": float(torus.area),
        "torus_L1": torus.L1.tolist(),
        "torus_L2": torus.L2.tolist(),
        "torus_A_WC": float(torus.A_WC),
        "torus_G1_mag": float(np.linalg.norm(torus.G1)),
        "torus_g1_mag": float(np.linalg.norm(torus.g1)),
        "density": {
            "grid": f"{DENSITY_NBINS} x {DENSITY_NBINS} over the supercell vectors",
            "bin_l_b": phases["liquid"]["density"]["bin_l_b"],
            "smoothing_kernel": (f"Gaussian, sigma = {DENSITY_KERNEL_L_B} l_B "
                                 f"= {DENSITY_KERNEL_L_B / phases['liquid']['density']['bin_l_b']:.4f} "
                                 f"bins, applied with PERIODIC wrapping"),
            "normalization": ("particles per area, divided by the cell mean "
                              "n = ne/area; a uniform liquid reads 1.0"),
            "drawn_window": (f"one cell plus {DENSITY_MARGIN_BINS} bins of periodic "
                             f"margin; one cell outlined"),
            "scale_choice": ("one colour scale per panel, vmin 0, vmax = 99.5th "
                             "percentile of that panel -- the historical density "
                             "figure's per-row convention; a shared scale would "
                             "flatten the liquid control"),
            "vmax_liquid": ctx.get("density_vmax", {}).get("liquid"),
            "vmax_crystal": ctx.get("density_vmax", {}).get("crystal"),
            "site_midpoint_contrast_liquid": phases["liquid"]["contrast"],
            "site_midpoint_contrast_crystal": phases["crystal"]["contrast"],
        },
        "gxy": {
            "grid": f"{GXY_NR} x {GXY_NR} bins",
            "bin_l_b": phases["liquid"]["gxy"]["bin_l_b"],
            "spatial_window": (f"+/- {phases['liquid']['gxy']['rmax']:.6f} l_B "
                               f"(R_HALF = 4/sqrt(n)); wider than the Wigner-Seitz "
                               f"inradius, so the map keeps periodic IMAGES rather "
                               f"than the minimum image"),
            "normalization": ("uniform value (ne-1)*dA/area per snapshot, every pair "
                              "counted in BOTH orientations; g = 1 for a uniform "
                              "liquid"),
            "smoothing": f"Gaussian, sigma = {GXY_SMOOTH} cells (edges held; the "
                         f"window is not a period of the supercell)",
            "scale_choice": "common colour scale across both panels, vmin 0",
            "vmax": ctx.get("gxy_vmax"),
            "g_r_nn_definition": ("maximum of the radial g(r) within half a lattice "
                                  "constant of a_WC = |g1| (the existing estimator's "
                                  "own shell rule)"),
            "g_r_nn_liquid": phases["liquid"]["g_rn"],
            "g_r_nn_crystal": phases["crystal"]["g_rn"],
        },
        "sq": {
            "q_grid": f"{SQ_NQ} x {SQ_NQ}, uniform in qx and qy",
            "q_range": (f"+/- {phases['liquid']['sq']['q_half']:.6f} l_B^-1 "
                        f"(Q_HALF = 8*sqrt(n)); identical in both panels"),
            "normalization": "S = <|rho_q|^2>/ne, averaged over configurations",
            "q0_treatment": (
                f"the disc |q| < |G1| = {float(np.linalg.norm(torus.G1)):.6f} is "
                f"MASKED in the figure.  S(q=0) = ne exactly for every state -- a "
                f"sum of ne unit phases over ne, which the state cannot enter -- so "
                f"it is kinematics, not structure.  The mask covers no MEASURED "
                f"momentum: the allowed torus momenta start at |G1| itself, so every "
                f"grid point inside the disc is a display sample of the q -> 0 "
                f"continuum.  What that continuum holds: the "
                f"{int(phases['liquid']['sq']['inner'].sum())} points inside "
                f"|q| < 1/|L1| = "
                f"{1.0 / float(np.linalg.norm(torus.L1)):.6f} still read "
                f"{float(phases['liquid']['sq']['S'][phases['liquid']['sq']['inner']].mean()):.1f} "
                f"~ ne, and the rest of the disc has already decayed to O(1).  NOTE "
                f"q=0 itself is not ON the grid: {SQ_NQ} points over "
                f"[-Q_HALF, Q_HALF] is an even count, so the smallest |qx| is "
                f"{float(np.abs(phases['liquid']['sq']['qx']).min()):.6f}.  The mask "
                f"is a display convention: the saved S array is raw and "
                f"mask_q0_disc is stored beside it, so nothing is deleted."),
            "scale_choice": ("common colour scale across both panels, vmin 0, vmax "
                             "set from the 0.75-1.25 |g1| peak annulus alone"),
            "vmax": ctx.get("sq_vmax"),
            "bragg_markers": ("the six first-shell triangular reciprocal-lattice "
                              "vectors, recomputed from the clean geometry "
                              "(Torus.wc_shell_vectors), marked with open circles"),
            "bragg_vector_count": int(len(phases["liquid"]["sq"]["bragg"])),
        },
        "results_dir": _rel(results_dir),
        "figures_dir": _rel(figdir),
        "provenance_firewall": (
            "Every number in these products comes from the two VMC runs this script "
            "started.  This module reads no input file: no checkpoint, no frozen "
            "snapshot, no stored observable array, no _diag product and no old PNG is "
            "reachable from it."),
    }

    meta = save_observables(results_dir, phases, extra)

    print()
    print("=" * 74)
    print("OUTPUTS")
    print("=" * 74)
    for name, vals in (
            ("density_xy.npz", "x, y, density_liquid, density_crystal, cell vectors, lattice sites"),
            ("pair_correlation_xy.npz", "x, y, g_liquid, g_crystal, g_rn_*"),
            ("structure_factor_xy.npz",
             "qx, qy, S_liquid, S_crystal, mask_q0_disc, bragg_vectors")):
        print(f"  {_rel(os.path.join(results_dir, name))}")
        print(f"      {vals}")
    print(f"  {_rel(os.path.join(results_dir, 'run_metadata.json'))}")
    print(f"  {_rel(results_dir)}/{{liquid,crystal}}/{{run.npz,run.json}}")
    for path in written:
        print(f"  {_rel(path)}")

    print()
    if args.budget != "reproduction":
        print(f"  NOTE: budget {args.budget!r} is NOT publication quality -- "
              f"the figures say so.")
    print(f"  total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
