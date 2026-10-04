"""Rebuild every reproducible figure from the frozen results.  One command:

    python wigner_vmc_clean/scripts/make_figures.py            # all of them
    python wigner_vmc_clean/scripts/make_figures.py --list     # names, no work
    python wigner_vmc_clean/scripts/make_figures.py --only fig02_validation_lll

This file is the composition root, and it is the ONLY place in the tree that knows where
the frozen legacy data lives.  Everything under `src/wigner_vmc/figures/` receives arrays
and a destination path; nothing there can load a file, and nothing there can sample.
`tests/test_figure_contract.py` enforces that split by parsing the figure modules rather
than importing them.

The shape of the data flow is the deliverable
---------------------------------------------
    frozen result  ->  io/  ->  analysis/  ->  figures/  ->  PNG

and never

    notebook cell  ->  run VMC  ->  analyse  ->  savefig

which is what the legacy tree does and why four of its figures cannot be rebuilt at all.
So the steps here are literally: read a checkpoint or a results JSON, hand it to an
analysis function, hand THAT to a figure function, save.  No figure computes an error
bar, and no figure function is called with a raw checkpoint.

What "one command" means, and what it does not
---------------------------------------------
Eleven of the fifteen frozen official PNGs are rebuilt here.  The other four --
`structure_factor_phase_map`, `correlation_function_phase_map`,
`shell_averaged_companions` and `energy_order_summary` -- plotted arrays that were
reduced inside a notebook cell and never written to disk, so no amount of wiring can
recover them; they are listed in FIGURE_MANIFEST.md as `legacy-only` with their original
PNGs kept exactly as they are.  Rebuilding those would mean re-running VMC, which figure
code is forbidden to do.

Nothing here writes to a legacy directory.  The output directory is
`wigner_vmc_clean/figures/` and it is created if absent; the frozen PNGs are read-only
by this pipeline and are never a destination.

Runtime
-------
The 2-D structure factors dominate: ten panels of 250 snapshots x 36 electrons x 25600
q-points each.  `--only` exists so a single figure can be rebuilt without paying for the
rest, and every figure prints its own wall time.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
sys.path.insert(0, os.path.join(CLEAN, "src"))

import numpy as np                                                     # noqa: E402

from wigner_vmc.analysis import energy_competition as ec               # noqa: E402
from wigner_vmc.analysis import statistics as stats                    # noqa: E402
from wigner_vmc.analysis import structure as st                        # noqa: E402
from wigner_vmc.figures import gaussian as G                           # noqa: E402
from wigner_vmc.figures import ll_rotation as L                        # noqa: E402
from wigner_vmc.io import checkpoints as ck                            # noqa: E402
from wigner_vmc.io import results as store                             # noqa: E402

RESULTS = os.path.join(CLEAN, "results")
FIGDIR = os.path.join(CLEAN, "figures")
CKPT = os.path.join(ROOT, "_diag", "ckpt_rebuild")

# ==========================================================================
# the campaign's fixed numbers, each derived rather than transcribed
# ==========================================================================
# The area is exact: A_WC = sqrt(4*pi/sqrt(3)) for nu = 1 and the supercell is 6 x 6
# cells, so A = 36 * (sqrt(3)/2) * A_WC^2 = 36 * 2*pi = 72*pi.  Written as the closed
# form rather than as 226.194671 the Torus then RE-DERIVES A_WC, |G1| and |g1| from it,
# so a transcription slip in one vector cannot survive: the four quantities would stop
# being mutually consistent and the ratio |g1|/|G1| = 6 would fail.
AREA = 72.0 * math.pi
N_ELECTRONS = 36
TORUS = st.Torus(AREA, n_electrons=N_ELECTRONS, n_cells_per_side=6)

SQRT_N = TORUS.sqrt_n
G1_MAG = float(np.linalg.norm(TORUS.G1))        # shortest supercell reciprocal vector
G1_WC = float(np.linalg.norm(TORUS.g1))         # first triangular-lattice Bragg vector
BRAGG = TORUS.wc_shell_vectors()                # the six of them

# The Gaussian workflow's render settings.  Both are the legacy cells' own: `dpi=110`
# comes from the notebook's rcParams and the tight bounding box from `savefig`'s default
# in those cells, while the LL-rotation cells set `dpi=170` and no bounding box.  The
# two workflows genuinely differ here; copying one's settings onto the other would change
# the output size for no reason a reader asked for, so they are kept apart.
DPI_G = 110
DPI_LL = 170

# --- geometry of the 1-D and 2-D structure observables ---------------------
QMAX_1D = 4.0                    # the legacy calibration grid's q cutoff, l_B^-1
R_HALF = 4.0 / SQRT_N            # real-space half-width, l_B   (10.0 on this torus)
Q_HALF = 8.0 * SQRT_N            # reciprocal half-width, 1/l_B (3.19 on this torus)
NR, NQ = 81, 160                 # grid points per axis in the g and S maps
G_SMOOTH = 1.0                   # Gaussian kernel, in g-map cells (0 = raw histogram)

# --- the couplings ---------------------------------------------------------
K_LIQUID, K_CRYSTAL = 4.0, 64.0          # the two contrast states of the maps
LADDER = (4.0, 16.0, 32.0, 48.0, 64.0)   # the shared ladder's five rungs

# --- the density maps of the LL-rotation set -------------------------------
NBINS_IV = 72
W_KERNEL_IV = 0.40               # l_B; the width the figure is DRAWN at
PAD_IV = 6                       # bins of periodic margin around the outlined cell

# ==========================================================================
# loaders, memoised for the length of one run
# ==========================================================================
# Ten of the eleven figures share two configuration ensembles and five more share the ten
# ladder checkpoints.  Reading a pickle is cheap and sampling is impossible, so the memo
# changes nothing about the contract -- it only stops the same file being opened five
# times and the same arrays being held under five names.
_CACHE: dict = {}


def _ens(name: str) -> ck.Checkpoint:
    """A configuration ensemble from the frozen store.  Carries no energy, on purpose."""
    if name not in _CACHE:
        _CACHE[name] = ck.read_ensemble(CKPT, name)
    return _CACHE[name]


def _snaps(name: str):
    return [np.asarray(R, float) for R in _ens(name).snaps]


def _grid_2d():
    """The (QX, QY, |q|) mesh every 2-D structure map is evaluated on."""
    qg = np.linspace(-Q_HALF, Q_HALF, NQ)
    qx, qy = np.meshgrid(qg, qg, indexing="ij")
    return qx, qy, np.hypot(qx, qy)


def _mask_below_g1(S):
    """Blank the region 0 < |q| < |G1|, which holds no allowed nonzero momentum.

    Masked to NaN rather than zeroed, and the figures' colour maps already carry
    `set_bad("0.4")`.  Below the shortest supercell reciprocal vector the box is narrower
    than one wavelength, so every electron adds in phase whatever the state is: the
    estimator returns the finite-box form factor, and it returns it for a liquid and a
    crystal alike.  Drawn, it is a bright disc at the centre of both S panels that is a
    statement about the size of the simulation box.  Nothing is hidden by removing it --
    the crystal's first reciprocal shell is at |g1| = 6|G1|, far outside.
    """
    return np.where(_grid_2d()[2] >= G1_MAG, S, np.nan)


def _sites_on_axes():
    """The 36 ansatz sites, in the frame the configuration panels are drawn in.

    `structure.lattice_sites` returns FRACTIONAL coordinates in [0, 1), which is the form
    the density panels want.  A scatter over a configuration needs Cartesian, and it needs
    the SAME frame the snapshots are in: the legacy walks start from
    `(uniform(0,1) - 0.5) @ sc.T`, i.e. a cell centred on the origin.  So the half-cell
    shift is applied here, once, rather than left to each caller.

    This is deliberately not the engine's own `gaussian_sites`, which folds into the
    Wigner-Seitz cell centred on the origin: that puts a site exactly on each corner of
    the drawn cell and pushes most of the rest to negative coordinates, so an overlay
    built from it lands off the axes it is drawn on.
    """
    return (st.lattice_sites(TORUS) - 0.5) @ TORUS.sc.T


def _proto_scan() -> ec.KappaScan:
    """The Gaussian workflow's Jastrow scan -- the frozen `crystalJ_*` / `liquidJ_*` store.

    A DIFFERENT dataset from the LL-rotation transition scan, and the two are never
    pooled: this one varies the coupling on a coarse grid with the Jastrow optimised at
    each point; that one varies the band-occupation truncation at fixed couplings.
    """
    if "scan" in _CACHE:
        return _CACHE["scan"]
    crystal = {}
    for k, _L0, name in ck.crystal_scan_names(CKPT, ec.KAPPAS_PROTO, ec.L0_GRID_PROTO):
        crystal.setdefault(k, []).append(ck.read_energy(CKPT, name))
    # The liquid column is the `liquidJ_*` family, NOT `ladder_liquid_*`.  Both exist and
    # both are called "liquid", but only the former stores an energy: the ladder family is
    # a configuration ensemble (`acc`, `sigma`, `snaps`) because it was walked to be drawn,
    # not measured.  Reading the ladder family here would raise KeyError('E') -- which is
    # the good outcome; the bad one would be a liquid whose energy came from a run that
    # never measured one.
    liquid = {}
    for k in ec.KAPPAS_PROTO:
        name = f"liquidJ_k{k:g}"
        if os.path.exists(os.path.join(CKPT, name + ".pkl")):
            liquid[float(k)] = ck.read_energy(CKPT, name)
    _CACHE["scan"] = ec.kappa_scan(crystal, liquid, ec.KAPPAS_PROTO)
    return _CACHE["scan"]


def _proto_boundary() -> ec.PhaseBoundary:
    """The autocorrelation-corrected bars and the raw crossing, from the tau checkpoints.

    `tau_crystal` / `tau_liquid` are the fine-decimated walks the notebook recorded for
    exactly this purpose; `read_tau` verifies the tuple's positional layout against the
    coupling in its own tag before returning a series, because E and V are both float
    series of the same length and a swap would produce plausible numbers silently.

    CAVEAT, recorded here because the figure cannot state it: the factor is measured at
    ONE coupling, kappa = 40 -- the grid point nearest the crossing -- and applied across
    the whole scan.  It is a correction borrowed from one point, not a per-point
    measurement.
    """
    if "pb" in _CACHE:
        return _CACHE["pb"]
    corr = {}
    for kind, name in (("crystal", "tau_crystal"), ("liquid", "tau_liquid")):
        series = ck.read_tau(CKPT, name)
        corr[kind] = stats.autocorrelation_correction(series["E"])
    _CACHE["pb"] = ec.phase_boundary(_proto_scan(), corr)
    return _CACHE["pb"]


def _tiers() -> dict:
    """`{tier: (rs, delta_E, sigma)}` for the LL-rotation truncation ladder.

    The three truncations are kept APART and never pooled, because they share couplings:
    at r_s = 75, 77.5 and 80 a converged n_max = 1 point and a nested n_max = 2 point sit
    at the same x, and at two of those a third joins them.  n_b = n_max + 1, so these are
    different truncations, and a crossing read off a mixture of them is a crossing of
    neither.  `nmax` is excluded for the same reason rather than by omission: those rungs
    are single-seed fresh-width starts, a different experiment from either ladder.

    Everything here comes from `results/legacy_llrot/scan.json`, i.e. from the records
    via `analysis/statistics.py` -- not from the legacy `energy_scan.json`, which is what
    the analysis is regression-tested AGAINST.
    """
    if "tiers" in _CACHE:
        return _CACHE["tiers"]
    pts, _brackets, _meta = store.load_scan(RESULTS, "llrot")
    acc: dict = {}
    for p in pts:
        if p.tier not in L.TIER_STYLE:
            continue
        acc.setdefault(p.tier, []).append((p.rs, p.delta_e, p.sigma_total))
    out = {}
    for tier, rows in acc.items():
        rows.sort(key=lambda r: r[0])
        out[tier] = (np.array([r[0] for r in rows], float),
                     np.array([r[1] for r in rows], float),
                     np.array([r[2] for r in rows], float))
    _CACHE["tiers"] = out
    return out


def _ladder_maps():
    """The (g, S) panel sets of fig09/fig10: five couplings x two states, one recipe.

    One memo for both figures, because they draw the SAME ensembles -- a second walk here
    would let the two pages disagree about the states they are supposed to share.
    """
    if "lmaps" in _CACHE:
        return _CACHE["lmaps"]
    qx, qy, rad = _grid_2d()
    S_maps, g_maps = {}, {}
    for k in LADDER:
        for kind in ("liquid", "crystal"):
            sn = _snaps(f"ladder_{kind}_k{k:g}")
            _c, g_maps[(kind, k)] = st.pair_correlation_2d(sn, TORUS, R_HALF, NR, G_SMOOTH)
            S = st.structure_factor_2d(sn, qx, qy, N_ELECTRONS)
            S_maps[(kind, k)] = np.where(rad >= G1_MAG, S, np.nan)
    _CACHE["lmaps"] = (S_maps, g_maps)
    return _CACHE["lmaps"]


# ==========================================================================
# the Gaussian workflow -- seven figures
# ==========================================================================
def fig01_phase_boundary(path, dpi):
    """delta_E(r_s) with both bar sets and |delta_E|/sigma.

    `scan` and `pb` come from the frozen `crystalJ_*` / `liquidJ_*` checkpoints and the
    two `tau_*` walk records; the figure receives a KappaScan and a PhaseBoundary and
    computes nothing itself, which is what keeps the crossing and the near-degeneracy
    window from being re-derived -- differently -- on the page.
    """
    return G.phase_boundary(_proto_scan(), _proto_boundary(), path=path, dpi=dpi)


def fig02_validation_lll(path, dpi):
    """VMC against the closed-form filled-LLL S(q) and g(r), with the residual.

    The frozen result is the SNAPSHOT ENSEMBLE, not the stored S(q) array: S and g are
    recomputed here from `valid_lll_structure`'s configurations through the analysis
    layer, which is what makes this figure a test of the pipeline rather than a redraw of
    it.  Measured agreement with the stored arrays is 3.3e-14 for S(q) and exact for
    g(r) -- and the S(q) comparison only agrees once the columns are matched by q VECTOR,
    because the stored array is in the engine's `argsort` order while
    `allowed_momenta` returns a rounded-lexsort order.  Same 294 momenta, permuted;
    compared column-by-column without that match the difference is 11.4.
    """
    ref = ck.read_structure_reference(CKPT, "valid_lll_structure")
    snaps = [np.asarray(R, float) for R in ref.snaps]
    q, qn = TORUS.allowed_momenta(QMAX_1D)

    per = st.structure_factor_snapshots(snaps, q)
    Sq = per.mean(axis=0)
    # ddof = 0 here, and it is REQUIRED rather than defaulted: the legacy calibration
    # cell took `.std(axis=0) / sqrt(n)`, while the band-occupation table in the other
    # workflow took ddof=1.  Making the convention explicit at every call site is the
    # only thing that has kept the two from being confused for each other.
    Sq_err = st.snapshot_sem(per, ddof=0)

    r, g, _g_sem = st.pair_correlation(snaps, TORUS, rmax=6.0, nbins=40)
    # The POISSON pair-count bar, not the snapshot scatter: it is the bar the legacy cell
    # drew, and it is valid here only because the shell counts run to the thousands.
    g_err = st.poisson_pair_error(g, r, 6.0, 40, TORUS, len(snaps))
    return G.lll_validation(qn, Sq, Sq_err, r, g, g_err, path=path, dpi=dpi)


def fig06_fingerprints_1d(path, dpi):
    """S(q) at the WC reciprocal vectors, g(r), and one configuration.

    The two states are `struct_liquid_k4` and `struct_crystal_k64`: the real variational
    states, each with the Jastrow the campaign optimised for it, at the two couplings the
    contrast figure uses.
    """
    q, qn = TORUS.allowed_momenta(QMAX_1D)
    i_b = int(np.argmin(np.abs(qn - G1_WC)))
    r_l, g_l, _ = st.pair_correlation(_snaps("struct_liquid_k4"), TORUS, 6.0, 40)
    r_c, g_c, _ = st.pair_correlation(_snaps("struct_crystal_k64"), TORUS, 6.0, 40)
    return G.fingerprints_1d(
        qn,
        st.structure_factor(_snaps("struct_liquid_k4"), q, N_ELECTRONS),
        st.structure_factor(_snaps("struct_crystal_k64"), q, N_ELECTRONS),
        st.exact_lll_sq(qn),
        r_l, g_l, r_c, g_c,
        _snaps("struct_liquid_k4")[-1], _sites_on_axes(), i_b, G1_WC,
        f"kappa={K_LIQUID:g}", f"kappa={K_CRYSTAL:g}",
        path=path, dpi=dpi)


def fig07_structure_maps(path, dpi):
    """g(x,y) and S(qx,qy) for both states, with the mask and the guides."""
    liq, cry = _snaps("struct_liquid_k4"), _snaps("struct_crystal_k64")
    qx, qy, _rad = _grid_2d()
    _c, g_l2 = st.pair_correlation_2d(liq, TORUS, R_HALF, NR, G_SMOOTH)
    _c, g_c2 = st.pair_correlation_2d(cry, TORUS, R_HALF, NR, G_SMOOTH)
    S_l2 = _mask_below_g1(st.structure_factor_2d(liq, qx, qy, N_ELECTRONS))
    S_c2 = _mask_below_g1(st.structure_factor_2d(cry, qx, qy, N_ELECTRONS))
    return G.structure_maps(g_l2, g_c2, S_l2, S_c2, TORUS,
                            K_LIQUID * math.sqrt(2.0), K_CRYSTAL * math.sqrt(2.0),
                            R_HALF, Q_HALF, SQRT_N, TORUS.A_WC, BRAGG,
                            path=path, dpi=dpi)


def fig08_sma(path, dpi):
    """omega_SMA(q) for both states against the filled-LLL reference and Kohn's mode."""
    q, qn = TORUS.allowed_momenta(QMAX_1D)
    return G.sma(qn,
                 st.structure_factor(_snaps("struct_liquid_k4"), q, N_ELECTRONS),
                 st.structure_factor(_snaps("struct_crystal_k64"), q, N_ELECTRONS),
                 st.exact_lll_sq(qn),
                 f"kappa={K_LIQUID:g}", f"kappa={K_CRYSTAL:g}",
                 path=path, dpi=dpi)


def fig09_ladder_Sq(path, dpi):
    """The transition in reciprocal space, one column per coupling."""
    S_maps, _g = _ladder_maps()
    return G.ladder_Sq(S_maps, LADDER, Q_HALF, SQRT_N, BRAGG, path=path, dpi=dpi)


def fig10_ladder_g(path, dpi):
    """The same ladder in real space."""
    _S, g_maps = _ladder_maps()
    return G.ladder_g(g_maps, LADDER, R_HALF, SQRT_N, TORUS.A_WC, path=path, dpi=dpi)


# ==========================================================================
# the LL-rotation set -- the four whose arrays survive
# ==========================================================================
def fig_llrot_energy_phase_competition(path, dpi):
    """The absolute E/N ladder, and delta_E on its own linear axis.

    The LL-rotated comparison series is NOT drawn.  Its checkpoints (`job1_conv_*`) were
    held in temporary sandboxes that have since been emptied, so the series cannot be
    rebuilt from anything on disk -- and the notebook itself already declines to draw it
    rather than approximating it.  The frozen PNG may carry it; if it does, that one
    series is legacy-only inside an otherwise reproducible figure.  Recorded in
    FIGURE_MANIFEST.md.
    """
    scan = _proto_scan()
    return L.energy_phase_competition(scan.rs, scan.E_liq, scan.E_liq_err,
                                      scan.E_cry, scan.E_cry_err, _tiers(),
                                      ll_rot=None, path=path, dpi=dpi)


def fig_llrot_crossing_linear(path, dpi):
    """The same 45-90 window on a page of its own, with |z| beside every point.

    Nothing is recomputed: it is the same tier table the competition figure draws, from
    the same binding, so the two pages cannot disagree about a value.
    """
    return L.energy_crossing_linear(_tiers(), path=path, dpi=dpi)


def fig_llrot_structure_factor_sma(path, dpi):
    """S(q) on the allowed torus momenta and the SMA mode it implies.

    The ladder rungs are the only states in the campaign walked with ONE recipe -- same
    sweeps, same equilibrating steps, same stride, same seed -- so they are the only
    states whose curves may be compared across r_s.  The requested r_s ~ 30 and ~ 60 have
    no matched-recipe state anywhere, and none is fabricated.
    """
    q, _qn = TORUS.allowed_momenta(QMAX_1D)
    shells = st.shells_by_magnitude(np.linalg.norm(q, axis=1))
    sqa = {}
    for k in LADDER:
        for kind in ("liquid", "crystal"):
            sqa[(kind, k)] = st.shell_average(_snaps(f"ladder_{kind}_k{k:g}"),
                                              q, N_ELECTRONS, shells)
    ref = ck.read_structure_reference(CKPT, "valid_lll_structure")
    lll = st.shell_average([np.asarray(R, float) for R in ref.snaps],
                           q, N_ELECTRONS, shells)
    qg = np.array([m for m, _ix in shells], float)
    qn = qg / SQRT_N
    i_g1 = int(np.argmin(np.abs(qg - G1_WC)))
    return L.structure_factor_sma(qn, qg, sqa, lll, i_g1, LADDER, G1_WC / SQRT_N,
                                  path=path, dpi=dpi)


def fig_llrot_real_space_density(path, dpi):
    """The two ansaetze as a density in the plane, along the full coupling ladder.

    Ten couplings, not the five of the ladder figure: this page's ladder is the campaign's
    full coupling grid, and the five rungs it shares with the reciprocal-space page are
    the SAME checkpoints -- which is what makes a difference between two columns a
    difference of coupling and of nothing else.
    """
    ladder = tuple(ec.KAPPAS_PROTO)
    bin_width = float(np.linalg.norm(TORUS.L1)) / NBINS_IV
    grids = {}
    for k in ladder:
        for kind in ("liquid", "crystal"):
            grids[(kind, k)] = st.density_grid(_snaps(f"ladder_{kind}_k{k:g}"),
                                               TORUS, nbins=NBINS_IV)
    return L.real_space_density(grids, TORUS, ladder, NBINS_IV, bin_width,
                                W_KERNEL_IV, PAD_IV, path=path, dpi=dpi)


# ==========================================================================
# the registry
# ==========================================================================
# (name, workflow, physical question, builder)
FIGURES = (
    ("fig01_phase_boundary", "Gaussian", "where does E_cry - E_liq change sign?",
     fig01_phase_boundary, DPI_G),
    ("fig02_validation_lll", "Gaussian", "does the sampler reproduce the exact filled LLL?",
     fig02_validation_lll, DPI_G),
    ("fig06_fingerprints_1d", "Gaussian", "what does the structure difference look like?",
     fig06_fingerprints_1d, DPI_G),
    ("fig07_structure_maps", "Gaussian", "is S(q) a ring or six Bragg peaks?",
     fig07_structure_maps, DPI_G),
    ("fig08_sma", "Gaussian", "what collective mode does S(q) imply?",
     fig08_sma, DPI_G),
    ("fig09_ladder_Sq", "Gaussian", "when does the reciprocal-space order appear?",
     fig09_ladder_Sq, DPI_G),
    ("fig10_ladder_g", "Gaussian", "when does the real-space order appear?",
     fig10_ladder_g, DPI_G),
    ("vmc_energy_phase_competition", "LL-rotation",
     "the absolute ladder, and the competition magnified",
     fig_llrot_energy_phase_competition, DPI_LL),
    ("vmc_energy_crossing_linear", "LL-rotation",
     "the crossing, on its own page, with |z| per point",
     fig_llrot_crossing_linear, DPI_LL),
    ("vmc_structure_factor_sma", "LL-rotation",
     "S(q) on the torus shells, and the mode it implies",
     fig_llrot_structure_factor_sma, DPI_LL),
    ("vmc_real_space_density", "LL-rotation",
     "what the two ansaetze look like as a density",
     fig_llrot_real_space_density, DPI_LL),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", action="append", default=None, metavar="NAME",
                    help="rebuild just this figure; repeatable")
    ap.add_argument("--list", action="store_true", help="list the figures and exit")
    ap.add_argument("--out", default=FIGDIR, help=f"output directory (default {FIGDIR})")
    args = ap.parse_args(argv)

    if args.list:
        for name, wf, q, _fn, dpi in FIGURES:
            print(f"  {name:32s} {wf:12s} dpi={dpi:<4d} {q}")
        print("\n  not rebuildable (arrays never stored; see FIGURE_MANIFEST.md):")
        for fn, why in L.LEGACY_ONLY:
            print(f"  {fn:32s} legacy-only  {why[:58]}")
        return 0

    wanted = set(args.only) if args.only else None
    if wanted:
        known = {f[0] for f in FIGURES}
        missing = sorted(wanted - known)
        if missing:
            print(f"unknown figure(s): {missing}\nknown: {sorted(known)}", file=sys.stderr)
            return 2

    os.makedirs(args.out, exist_ok=True)
    print(f"frozen results : {RESULTS}")
    print(f"checkpoints    : {CKPT}")
    print(f"output         : {args.out}\n")

    rows, failures = [], []
    for name, wf, _q, fn, dpi in FIGURES:
        if wanted and name not in wanted:
            continue
        path = os.path.join(args.out, name + ".png")
        t0 = time.perf_counter()
        try:
            fn(path, dpi)
        except Exception as exc:                                       # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name:32s} {type(exc).__name__}: {exc}")
            continue
        secs = time.perf_counter() - t0
        size = os.path.getsize(path) if os.path.exists(path) else 0
        rows.append((name, wf, secs, size))
        print(f"  ok    {name:32s} {secs:7.1f} s  {size:>9,d} B")

    print()
    if rows:
        total = sum(r[2] for r in rows)
        print(f"{len(rows)} figure(s) written to {args.out} in {total:.1f} s")
    if failures:
        print(f"*** {len(failures)} figure(s) FAILED:")
        for name, why in failures:
            print(f"      {name}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
