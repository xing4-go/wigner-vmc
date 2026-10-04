"""A two-panel structural SLICE: ``S(q)`` and ``g(r)`` for a liquid and a crystal.

    python examples/figure_construction/structure_slice.py --budget quick
    python examples/figure_construction/structure_slice.py --budget full

What this is
------------
A research recipe.  It drives the validated public ``wigner_vmc`` API from a cold
start at the two couplings the historical figure compares -- a **filled-LLL
liquid at kappa = 4** and a **Wigner crystal at kappa = 64** -- and builds two
observables from the configurations it just produced:

    1. ``S(q)`` on the supercell's allowed momenta, against the exact filled-LLL
       curve and the triangular lattice's first Bragg vector ``|g1|``;
    2. ``g(r)``.

There is deliberately NO real-space panel.  The historical figure had a third
panel drawing one configuration with the ansatz sites overlaid; that picture is
a different question (it is answered with more room, and with the density and
pair-correlation FIELDS, by ``structure.py``), and mixing it into a
reciprocal-space slice made both harder to read.  ``real_space_snapshot.npz`` is
still written -- the raw configuration is data, and the numbers stay available
even though no figure draws them.

The crystal projection convention is the CORRECTED one.  The seed comes from
``sr.gaussian_overlap_seed``, whose overlap is ``<phi|G>``; the reversed
``G.conj(), bloch`` (``<G|phi>``) that produced the earlier uniaxial crystals
was fixed on 2026-10-03 and this recipe has no path back to it.  See
``PROJECTION_CONJUGATION_FIX_VALIDATION.md``.

Budgets: ``quick`` (minutes, a smoke test -- the figures are stamped) and
``full`` (the real statistics).  ``full`` is an alias that resolves to the
``reproduction`` config; ``--budget reproduction`` is accepted and identical.

Nothing here is a redraw.  Every number comes from a VMC run this script starts
itself: the API builds its own starting point from ``(N, r_s)`` alone, and no
checkpoint, cached observable array or old figure is read on this path.  The
historical data is touched in exactly one place -- ``--compare``, off by default
-- and only *after* the clean numbers exist, to be printed next to them.

Where things live
-----------------
    examples/figure_construction/                    this recipe
    results/figure_construction/structure_slice/     the numbers it measures
    figures/figure_construction/structure_slice/     the pictures it draws

No generated file is written back into ``examples/``.  The historical
``figures/fig06_fingerprints_1d.png`` is not a destination and is never opened
for writing.

Units: ``l_B = hbar = m = 1``, energy in ``hbar*omega_c``.  ``r_s = sqrt(2) *
kappa``, so kappa = 4 is ``r_s = 4*sqrt(2)`` and kappa = 64 is ``r_s =
64*sqrt(2)``; both resolve to the exact coupling in floating point, which the
contract test pins.
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

from wigner_vmc import VMC, __version__, load_budget, resolve_budget_name
from wigner_vmc.analysis import structure as st

# ==========================================================================
# paths.  Derived from this file's location, so the recipe works from any cwd.
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))          # .../wigner_vmc_clean
ROOT = os.path.dirname(CLEAN)                           # the notebook tree
RESULTS = os.path.join(CLEAN, "results", "figure_construction", "structure_slice")
FIGDIR = os.path.join(CLEAN, "figures", "figure_construction", "structure_slice")

# The historical tree, read ONLY by `--compare`, and never an input to a figure.
LEGACY_STORE = os.path.join(ROOT, "_diag", "ckpt_rebuild")

# ==========================================================================
# the physical points, and the observable settings -- all traced, none tuned
# ==========================================================================
N_ELECTRONS = 36

# The two couplings of the historical contrast.  Written as kappa and DERIVED
# through r_s = sqrt(2) kappa, so the identity the contract test pins is
# structural rather than a coincidence of two transcriptions that agree.
KAPPA_LIQUID, KAPPA_CRYSTAL = 4.0, 64.0
RS_LIQUID = KAPPA_LIQUID * math.sqrt(2.0)
RS_CRYSTAL = KAPPA_CRYSTAL * math.sqrt(2.0)

# fig06's own observable settings: the notebook's cell 28 q window, and the
# rmax/nbins its `pair_correlation` calls pass.
Q_MAX = 4.0
R_MAX = 6.0
N_BINS = 40

# Which starting point the crystal uses, per budget.  `init_id` is an INDEX into
# the budget's own list, not a seed; in `reproduction` its entry 0 is L0 = 0.40,
# the width fig06's crystal was built at (see configs/reproduction.yaml).
CRYSTAL_INIT_ID = 0

# Which crystal ansatz the recipe runs by default: the package's own, joint SR
# over orbitals and Jastrow.  There is deliberately NO per-budget ansatz table.
#
# The historical fig06 crystal is a THIRD ansatz this package does not
# implement -- a determinant of site-centred magnetic Gaussians (notebook cell
# 18's `crystal_wavefunction`).  So there is no ansatz this recipe could select
# to "reproduce" that state exactly, and pretending otherwise would just relabel
# a different calculation.  The two families are nonetheless not far apart: the
# reciprocal-space weight converges with the truncation, and at `--crystal-nmax
# 3` it matches the historical ensemble within the protocol spread.  The
# measurements are in DIVERGENCE_fig06_crystal.md next to this file.
DEFAULT_ANSATZ = "ll_rotation"

# Which budget the canonical `structure_slice/` artifacts are produced with.
# Together with the two couplings, the ansatz and n_max, this defines the ONE
# request that owns the canonical `structure_slice/` namespace.
HISTORICAL_BUDGET = "reproduction"
DEFAULT_CRYSTAL_NMAX = 1

# The liquid's nmax is a TECHNICAL value, not a physical one.  Its determinant
# rows are `sum_n C[k,n] phi_{k,n}` with `v = 0`, and `c_row(0)` is exactly the
# first unit vector -- so `C` selects `phi_{k,0}` and every higher band enters
# multiplied by zero.  The liquid is therefore the filled n=0 Landau level
# whatever this number is; it exists only because the shared parameter vector
# reserves room for a rotation that the liquid never uses.  Naming it here, once,
# is what keeps it from being mistaken for a user-facing knob: there is no
# `--liquid-nmax`, and the pre-flight prints it as a `technical` field.
LIQUID_NMAX = 1

# The legacy snapshot-selection rule for the real-space panel: cell 28 plots
# `snaps[-1]`, the last configuration the walk kept.  Same rule here.
def _last_snapshot(snaps):
    return np.asarray(snaps[-1], float)


def _fmt_rs(rs):
    """A compact r_s for a LABEL: ``75`` or ``5.657``.

    Three decimals separate every coupling this recipe is used at without
    turning a legend into a float dump.  Full precision is in the metadata --
    and the metadata is where a reader should go, because two r_s that render
    the same here are still two different calculations.
    """
    rs = float(rs)
    if abs(rs - round(rs)) < 1e-9:
        return f"{rs:.0f}"
    return f"{rs:.3f}"


def _slug_rs(rs):
    """A compact, deterministic, filename-safe r_s: ``75``, ``5.65685``.

    Six significant digits, so a path stays readable while two couplings a user
    would plausibly type stay apart.  It cannot be a perfect key -- and it does
    not have to be, because the directory only *groups* a run: `--resume` checks
    the full-precision config digest before reusing anything, and `main` refuses
    to overwrite a directory whose recorded request differs from this one.
    """
    rs = float(rs)
    if abs(rs - round(rs)) < 1e-9:
        return f"{rs:.0f}"
    return f"{rs:.6g}"


#: Short, stable ansatz tokens.  `ll_rotation` is the default and so rarely
#: appears; the pinned scope is spelled out because it changes what the
#: optimiser may move.
_ANSATZ_TAG = {"ll_rotation": "llrot", "ll_rotation_pinned": "llrot_pinned"}


def request_slug(liquid_rs, crystal_rs, ansatz, nmax, budget):
    """The directory name for a request: every input that changes the numbers.

    Four of the five are physical (the two couplings, the ansatz, the
    truncation); the budget is the protocol, which changes the numbers just as
    surely -- `quick` and `reproduction` walk the same state for different
    lengths.  Leaving any of them out would let one calculation overwrite
    another, which is the failure this exists to prevent.
    """
    tag = _ANSATZ_TAG.get(str(ansatz), str(ansatz))
    return (f"liquid_rs{_slug_rs(liquid_rs)}__crystal_rs{_slug_rs(crystal_rs)}"
            f"__{tag}_nmax{int(nmax)}__{budget}")


def is_historical_request(liquid_rs, crystal_rs, ansatz, nmax, budget):
    """Is this exactly the request the canonical ``structure_slice/`` holds?

    Exact float equality on the couplings, deliberately.  ``4*sqrt(2)`` is not
    ``5.656854249``; they differ in the tenth digit and give different energies.
    Treating them as the same request is how a near-miss gets filed as a
    reproduction.
    """
    return (float(liquid_rs) == RS_LIQUID and float(crystal_rs) == RS_CRYSTAL
            and str(ansatz) == DEFAULT_ANSATZ
            and int(nmax) == DEFAULT_CRYSTAL_NMAX
            and str(budget) == HISTORICAL_BUDGET)


def namespace(base, liquid_rs, crystal_rs, ansatz, nmax, budget):
    """``base`` for the historical request, ``<base>_<slug>`` for anything else.

    ``base`` is fixed by the layout (``.../structure_slice``), so the historical
    request keeps that name -- the canonical artifacts stay where they are and
    `--resume` still finds them.  Every other request is a
    different calculation and gets a name that says which one, so two couplings
    can be compared side by side instead of the second erasing the first.
    """
    if is_historical_request(liquid_rs, crystal_rs, ansatz, nmax, budget):
        return base
    return base + "_" + request_slug(liquid_rs, crystal_rs, ansatz, nmax, budget)


def _fmt_kappa(rs):
    """kappa = r_s / sqrt(2) at the precision the engine resolves it to."""
    return f"{float(rs) / math.sqrt(2.0):.9f}"


# ==========================================================================
# the calculation
# ==========================================================================
def run_point(phase, rs, init_id, budget, ansatz="ll_rotation", nmax=1,
              verbose=True):
    """One cold VMC run through the public API.  Nothing is loaded.

    ``ansatz`` and ``nmax`` are crystal choices; the liquid has no variational
    orbital sector, so both are forced to their liquid values here rather than
    passed on and rejected by the API.  That forcing is the point: the two
    phases do not share a parameter space, and a caller cannot accidentally hand
    the crystal's truncation to the liquid.

    Returns a plain dict, so the caller can persist it without knowing anything
    about the API's own objects.
    """
    if phase != "crystal":
        ansatz, nmax = DEFAULT_ANSATZ, LIQUID_NMAX
    vmc = VMC(N=N_ELECTRONS, rs=rs, phase=phase, nmax=int(nmax), ansatz=ansatz)
    t0 = time.time()
    result = vmc.run(init_id=init_id, budget=budget, verbose=verbose)
    snaps = [np.asarray(R, float) for R in result.state.snaps]

    # The two phases are described in their own terms.  The liquid's `nmax` is
    # padding (see LIQUID_NMAX) and is recorded as such, so nothing downstream
    # can read it as a liquid truncation.
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
        # "orbitals pinned" describes the liquid too: it has no variational
        # orbital sector (v = 0 by definition), and `phase` is recorded beside
        # this field, so the phrase cannot be read as a claim about the crystal.
        "optimized": ("orbitals+jastrow" if result.optimization.kind == "joint"
                      else "jastrow only (orbitals pinned)"),
        "init_id": int(init_id),
        "init_L0": float(result.budget.width_for(init_id)) if phase == "crystal" else None,
        "budget": str(budget),
        "rng_seed": int(result.config.rng_seed),
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


def observables(snaps, torus, q_max=Q_MAX, r_max=R_MAX, n_bins=N_BINS):
    """``S(q)`` and ``g(r)`` from the configurations, through the analysis layer.

    Both estimators are the validated ones -- this function computes no physics
    of its own, it only chooses the arguments.  The q set is the torus's allowed
    momenta ``q = m G1 + n G2``, which is what the legacy cell measured on; the
    exact filled-LLL curve is the closed form ``1 - exp(-q^2/2)``.
    """
    snaps = [np.asarray(R, float) for R in snaps]
    q, qn = torus.allowed_momenta(q_max)
    r, g, g_sem = st.pair_correlation(snaps, torus, r_max, n_bins)
    return {
        "q": q,
        "qn": qn,
        "S": st.structure_factor(snaps, q, int(torus.ne)),
        "S_lll": st.exact_lll_sq(qn),
        "r": r,
        "g": g,
        "g_sem": g_sem,
        "g_lll": st.exact_lll_g(r, int(torus.ne)),
        "g1_mag": float(np.linalg.norm(torus.g1)),
        "G1_mag": float(np.linalg.norm(torus.G1)),
        "sites": (st.lattice_sites(torus) - 0.5) @ torus.sc.T,
        "torus": torus,
    }


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
    """A source identifier for the tree that produced this run, or an honest gap.

    The brief asks for a "source/git identifier if available".  This checkout has
    no ``.git`` at its root, so there is no commit to record.  Recording the bare
    absence is better than an empty field, which a later reader could mistake for
    a commit that was lost.
    """
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
    return f"not a git repository (no .git at the checkout root); wigner_vmc {__version__}"


#: The saved-point FORMAT.  Bumping it retires every point written by an older
#: format: `load_point` compares the digest, so an old file simply stops being
#: resumable and the run recomputes.  That is the wanted behaviour -- a point
#: saved before the per-phase metadata existed cannot supply the fields
#: `run_metadata.json` now has to carry, and the alternative (backfilling them
#: from the new code) would record descriptions the original run never made.
POINT_SCHEMA = 2


def _config_key(phase, rs, init_id, budget, ansatz="ll_rotation", nmax=1):
    return {"phase": phase, "rs": repr(float(rs)), "N": N_ELECTRONS,
            "nmax": int(nmax), "ansatz": str(ansatz),
            "init_id": int(init_id), "budget": str(budget),
            "point_schema": POINT_SCHEMA}


def save_point(outdir, name, point, key):
    """``<outdir>/<name>/run.npz`` + ``run.json``.  The json carries the config
    digest `--resume` checks before it will reuse anything."""
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
    state's label on another state's data.  So the digest is checked, and a
    mismatch is treated as "not resumable" rather than as an error.
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


def save_merged(outdir, points, obs, extra):
    """The three merged numerical products plus the run metadata."""
    os.makedirs(outdir, exist_ok=True)
    liq, cry = points["liquid"], points["crystal"]

    np.savez_compressed(
        os.path.join(outdir, "structure_factor.npz"),
        q=obs["liquid"]["q"], qn=obs["liquid"]["qn"],
        S_liquid=obs["liquid"]["S"], S_crystal=obs["crystal"]["S"],
        S_filled_LLL_exact=obs["liquid"]["S_lll"],
        g1_mag=obs["liquid"]["g1_mag"], G1_mag=obs["liquid"]["G1_mag"],
        kappa_liquid=liq["kappa"], kappa_crystal=cry["kappa"])

    np.savez_compressed(
        os.path.join(outdir, "pair_correlation.npz"),
        r=obs["liquid"]["r"], g_liquid=obs["liquid"]["g"],
        g_crystal=obs["crystal"]["g"], g_liquid_sem=obs["liquid"]["g_sem"],
        g_crystal_sem=obs["crystal"]["g_sem"],
        g_filled_LLL_exact=obs["liquid"]["g_lll"])

    torus = obs["liquid"]["torus"]
    snap = _last_snapshot(liq["snaps"])
    np.savez_compressed(
        os.path.join(outdir, "real_space_snapshot.npz"),
        R=snap, lattice_sites=obs["liquid"]["sites"],
        L1=torus.L1, L2=torus.L2, A1=torus.A1, A2=torus.A2,
        area=np.asarray(torus.area), n_electrons=np.asarray(torus.ne),
        n_cells_per_side=np.asarray(torus.n_side),
        selection_rule=np.asarray("last production snapshot of the liquid walk"))

    meta = dict(extra)
    meta["run_metadata_schema"] = "figure_construction/structure_slice/1"
    for name in ("liquid", "crystal"):
        p = dict(points[name])
        p.pop("snaps", None)
        p.pop("R", None)
        meta[name] = p
    meta["observables"] = {
        "q_grid": ("allowed_momenta of the supercell: q = m*G1 + n*G2, "
                   "0 < |q| < q_max, rounded-lexsort order"),
        "q_max": Q_MAX,
        "S_q_definition": "<|rho_q|^2>/ne, averaged over snapshots (rho_q = sum_j e^{i q.r_j})",
        "S_LLL_reference": "1 - exp(-q^2/2), exact at any N",
        "r_max": R_MAX,
        "n_bins": N_BINS,
        "g_r_definition": ("minimum-image radial pair histogram, normalised so a uniform "
                           "liquid gives g = 1; ne*(ne-1)/2 counts per snapshot"),
        "g_LLL_reference": "(N/(N-1)) (1 - exp(-r^2/2))",
        "sites": ("analysis.structure.lattice_sites, fractional in [0,1), shifted by half "
                  "a cell into the frame the snapshots are drawn in"),
        "snapshot_rule": "snaps[-1] of the liquid walk (the legacy cell 28 rule)",
    }
    with open(os.path.join(outdir, "run_metadata.json"), "w", encoding="utf-8") as fh:
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
            ha="right", va="top", fontsize=6.5, color="crimson", linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor="crimson", alpha=0.85,
                      boxstyle="round,pad=0.25", linewidth=0.6))


def _labels(points):
    """Legend labels, named by the parameters the user actually chose.

    ``r_s`` first, because it is the control parameter; ``n_max`` on the crystal,
    because at the SAME ``r_s`` two truncations give two different curves and a
    label that named only the coupling would not say which one the reader is
    looking at.  ``kappa`` is deliberately not in the label -- it is the same
    information as ``r_s``, and run_metadata.json records both.

    The ansatz appears only when it is not the default, so the ordinary case
    stays short.
    """
    liq, cry = points["liquid"], points["crystal"]
    lab_c = (f"crystal, $r_s$ = {_fmt_rs(cry['rs'])}, "
             f"$n_{{\\rm max}}$ = {int(cry['nmax'])}")
    if cry["ansatz"] != DEFAULT_ANSATZ:
        lab_c += f" ({cry['ansatz']})"
    return f"liquid, $r_s$ = {_fmt_rs(liq['rs'])}", lab_c


def draw_structure_factor(ax, points, obs, budget):
    lab_l, lab_c = _labels(points)
    ax.plot(obs["liquid"]["qn"], obs["liquid"]["S"], "o", ms=3, label=lab_l)
    ax.plot(obs["crystal"]["qn"], obs["crystal"]["S"], "s", ms=3, label=lab_c)
    ax.plot(obs["liquid"]["qn"], obs["liquid"]["S_lll"], "-", lw=1, color="0.6",
            label="filled LLL (exact)")
    ax.axvline(obs["liquid"]["g1_mag"], color="r", ls=":", lw=1)
    ax.set(xlabel=r"$q\,\ell_B$", ylabel=r"$S(q)$",
           title=r"Bragg peak at $|g_1|$")
    ax.legend(fontsize=7)
    _quick_banner(ax, budget)


def draw_pair_correlation(ax, points, obs, budget):
    lab_l, lab_c = _labels(points)
    ax.plot(obs["liquid"]["r"], obs["liquid"]["g"], "o-", ms=3, label=lab_l)
    ax.plot(obs["crystal"]["r"], obs["crystal"]["g"], "s-", ms=3, label=lab_c)
    ax.axhline(1, color="k", lw=.8)
    ax.set(xlabel=r"$r/\ell_B$", ylabel=r"$g(r)$", title="real-space order")
    ax.legend(fontsize=7)
    _quick_banner(ax, budget)


def write_figures(figdir, points, obs, budget, dpi=110):
    """Two standalone panels, plus the same two drawn side by side.

    There is deliberately NO real-space panel here -- see the module docstring.
    ``combined.png`` is not a third observable; it is the same two draws on one
    row, so the two files cannot disagree about the physics.
    """
    plt = _mpl()
    os.makedirs(figdir, exist_ok=True)
    written = []

    def save(fig, name):
        path = os.path.join(figdir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    for name, draw in (("structure_factor.png", draw_structure_factor),
                       ("pair_correlation.png", draw_pair_correlation)):
        fig, ax = plt.subplots(figsize=(5.0, 3.7))
        draw(ax, points, obs, budget)
        fig.tight_layout()
        save(fig, name)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7))
    draw_structure_factor(ax[0], points, obs, budget)
    draw_pair_correlation(ax[1], points, obs, budget)
    fig.tight_layout()
    save(fig, "combined.png")
    return written


# ==========================================================================
# the comparison -- AFTER the clean numbers exist, never before
# ==========================================================================
def _summarise(o, n_snapshots):
    """Reduce an ``observables()`` dict to the handful of scalars a comparison
    quotes.  One function, so the clean and the historical side of the
    comparison are reduced by the same code and cannot be measured differently.

    The first ``g(r)`` maximum is taken over ``r > 1``: the r -> 0 correlation
    hole is the larger feature but it is a minimum, and ``argmax`` over the whole
    range would pick up the small-r rise out of the hole rather than the first
    real shell.
    """
    qn, S, r, g = o["qn"], o["S"], o["r"], o["g"]
    i_b = int(np.argmin(np.abs(qn - o["g1_mag"])))
    off = int(np.sum(r <= 1.0))
    i_pk = off + int(np.argmax(g[off:])) if off < len(r) else int(np.argmax(g))
    return {"n_snapshots": int(n_snapshots), "qn": qn, "S": S, "r": r, "g": g,
            "g1_mag": o["g1_mag"], "bragg_q": float(qn[i_b]),
            "bragg_S": float(S[i_b]), "g_max_r": float(r[i_pk]),
            "g_max": float(g[i_pk])}


def compare_with_legacy():
    """The historical ensembles, reduced by the CLEAN estimators.

    This is the only place the historical tree is touched: read-only, and called
    only after every clean number has been computed and saved.  The legacy
    ``S(q)`` and ``g(r)`` arrays are not stored anywhere -- the historical cell
    reduced them in memory -- so what is compared is the frozen
    *configurations*, run through the same ``observables()`` as the clean ones.
    That makes a difference a difference of samples and protocol rather than of
    two estimators.

    Returns ``{phase: summary}``, or ``None`` if the frozen store is absent.
    """
    import pickle
    torus = torus_for()
    out = {}
    for stem, key in (("struct_liquid_k4", "liquid"),
                      ("struct_crystal_k64", "crystal")):
        path = os.path.join(LEGACY_STORE, f"{stem}.pkl")
        if not os.path.exists(path):
            print(f"  [compare] frozen ensemble {stem}.pkl not found; skipping")
            return None
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        snaps = [np.asarray(R, float) for R in payload["snaps"]]
        out[key] = _summarise(observables(snaps, torus), len(snaps))
    return out


def _comparison_record(clean, legacy):
    """The comparison as JSON, so the reproduction claim is auditable later
    rather than only legible in the terminal scrollback it was printed to."""
    out = {}
    for key in ("liquid", "crystal"):
        c, l = clean[key], legacy[key]
        dS = np.interp(l["qn"], c["qn"], c["S"]) - l["S"]
        dg = np.interp(l["r"], c["r"], c["g"]) - l["g"]
        out[key] = {
            "n_snapshots_clean": c["n_snapshots"],
            "n_snapshots_legacy": l["n_snapshots"],
            "g1_mag": c["g1_mag"],
            "S_at_bragg_clean": c["bragg_S"],
            "S_at_bragg_legacy": l["bragg_S"],
            "g_first_max_clean": [c["g_max"], c["g_max_r"]],
            "g_first_max_legacy": [l["g_max"], l["g_max_r"]],
            "S_q_max_abs_dev": float(np.abs(dS).max()),
            "S_q_rms_dev": float(np.sqrt((dS ** 2).mean())),
            "g_r_max_abs_dev": float(np.abs(dg).max()),
            "g_r_rms_dev": float(np.sqrt((dg ** 2).mean())),
            "note": ("the legacy curves are re-derived from the frozen configurations "
                     "through the same clean estimators; the difference is samples and "
                     "protocol, not analysis.  Particle-by-particle equality in real "
                     "space is not claimed."),
        }
    return out


def _report_comparison(clean, legacy):
    print()
    print("=" * 74)
    print("NUMERICAL REPRODUCTION -- clean vs the historical fig06 ensembles")
    print("=" * 74)
    print("  The historical curves are re-derived from the frozen configurations with")
    print("  the SAME clean estimators, so a difference is a difference of samples and")
    print("  protocol, not of analysis.  The historical ensembles hold 250 snapshots")
    print("  each, from an RNG stream this recipe does not share.")
    print()
    for key in ("liquid", "crystal"):
        c, l = clean[key], legacy[key]
        print(f"  {key}:")
        print(f"    snapshots            clean {c['n_snapshots']:>5}"
              f"     legacy {l['n_snapshots']:>5}")
        print(f"    |g1|                 {c['g1_mag']:.6f}"
              f"   (the same torus on both sides)")
        print(f"    S(|g1|)              clean {c['bragg_S']:8.4f}"
              f"   legacy {l['bragg_S']:8.4f}")
        print(f"    g(r) first maximum   clean {c['g_max']:8.4f} at r={c['g_max_r']:.3f}"
              f"   legacy {l['g_max']:8.4f} at r={l['g_max_r']:.3f}")
        dS = np.interp(l["qn"], c["qn"], c["S"]) - l["S"]
        dg = np.interp(l["r"], c["r"], c["g"]) - l["g"]
        print(f"    S(q) max|dev|        {np.abs(dS).max():.4f}"
              f"   (rms {np.sqrt((dS**2).mean()):.4f}, over {len(l['qn'])} momenta)")
        print(f"    g(r) max|dev|        {np.abs(dg).max():.4f}"
              f"   (rms {np.sqrt((dg**2).mean()):.4f}, over {len(l['r'])} shells)")
    print()
    print("  Real-space panel: particle-by-particle equality is NOT required and not")
    print("  claimed -- the RNG streams differ.  What is compared is the cell geometry")
    print("  (identical: the same torus) and the lattice sites (identical set of 36")
    print("  cosets).  See FIGURE_MANIFEST.md for the historical figure's own record.")


# ==========================================================================
# driver
# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build the two structure_slice panels -- S(q) and g(r), no "
                    "real-space panel -- from a clean VMC calculation, at couplings "
                    "the user chooses on the command line.")
    p.add_argument("--liquid-rs", type=float, default=None, metavar="RS",
                   help=f"Wigner-Seitz radius of the LIQUID, in units of l_B.  "
                        f"Default {RS_LIQUID:.9f} (4*sqrt(2), the historical fig06 liquid).")
    p.add_argument("--crystal-rs", type=float, default=None, metavar="RS",
                   help=f"Wigner-Seitz radius of the CRYSTAL, in units of l_B.  "
                        f"Default {RS_CRYSTAL:.9f} (64*sqrt(2), the historical fig06 crystal).")
    p.add_argument("--crystal-nmax", type=int, default=None, metavar="N",
                   help=f"Highest Landau level the CRYSTAL basis keeps: n = 0 ... N, so "
                        f"n_bands = N + 1.  Default {DEFAULT_CRYSTAL_NMAX}.  Must be >= 1 "
                        f"(the engine's own bound -- its nmax is the highest index, not a "
                        f"band count).  There is deliberately no --liquid-nmax: the liquid "
                        f"is the filled LLL, fixed, and has no truncation to choose.")
    p.add_argument("--budget", default="quick",
                   help="quick (smoke test, minutes) | full (the real statistics). "
                        "'full' is an alias for the 'reproduction' config -- the protocol "
                        "transcribed from the historical figure's own notebook cells -- and "
                        "that spelling is accepted too.  Any other name in configs/ is "
                        "accepted; those are the documented ones.")
    p.add_argument("--resume", action="store_true",
                   help="reuse a saved clean run in this request's own results directory "
                        "when its recorded configuration -- including the ansatz, nmax "
                        "and couplings -- matches exactly.")
    p.add_argument("--compare", action="store_true",
                   help="after computing, also print a numerical comparison against the "
                        "historical ensembles (read-only).")
    p.add_argument("--ansatz", choices=("ll_rotation", "ll_rotation_pinned"),
                   default=None,
                   help="crystal optimisation scope.  'll_rotation' (default) optimises "
                        "the orbitals and the Jastrow jointly; 'll_rotation_pinned' "
                        "holds the orbitals at the Gaussian-overlap seed and optimises "
                        "the Jastrow alone.  NEITHER is the historical fig06 crystal, "
                        "whose ansatz this package does not implement; the two families "
                        "converge as --crystal-nmax grows -- see "
                        "DIVERGENCE_fig06_crystal.md.")
    p.add_argument("--quiet", action="store_true", help="suppress the per-run summaries.")
    return p.parse_args(argv)


def validate_inputs(liquid_rs, crystal_rs, crystal_nmax):
    """Refuse a request the engine cannot honour -- before anything is computed.

    The bounds are the ENGINE'S, not invented here.  ``api.resolve`` and
    ``VMC.__init__`` both require ``nmax >= 1``, because this package's ``nmax``
    is the highest Landau *index* (``LandauLevelBasis`` is built with
    ``nmax = n_bands - 1``), so ``nmax = 0`` would ask for a single band and the
    ladder operators need the padding band above it.  There is no upper bound in
    the engine, so none is imposed here; the cost of a large ``nmax`` is real
    (the Bloch sum runs over every band) and it is the user's to spend.

    Raising ``SystemExit`` rather than returning a flag is deliberate: this is a
    CLI contract, and the failure must land before the SR ladder, not after it.
    """
    problems = []
    for flag, value in (("--liquid-rs", liquid_rs), ("--crystal-rs", crystal_rs)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            problems.append(f"{flag} must be finite and > 0, got {value!r}")
    if int(crystal_nmax) < 1:
        problems.append(
            f"--crystal-nmax must be >= 1, got {crystal_nmax!r}.  The engine's nmax "
            f"is the highest Landau index (n = 0 ... nmax), so 0 would mean a single "
            f"band, which the Landau-level basis cannot build.")
    if problems:
        raise SystemExit("structure_slice: invalid request\n  "
                         + "\n  ".join(problems))


def _phase_kappa(rs):
    return float(rs) / math.sqrt(2.0)


def print_preflight(liquid_rs, crystal_rs, crystal_nmax, ansatz, bud,
                    results_dir, figdir):
    """The resolved calculation, in full, BEFORE anything expensive starts.

    Two blocks because the two phases do not have the same parameters.  The
    liquid has no orbital ansatz to vary and no truncation to choose; saying so
    explicitly is the point, because the shared parameter vector happens to
    carry an `nmax` for both and printing it as though it were liquid physics is
    what this block exists to stop.
    """
    w = 74
    print("=" * w)
    print("RESOLVED CALCULATION")
    print("=" * w)
    print()
    print("LIQUID")
    print(f"  rs               {liquid_rs:.9f}")
    print(f"  kappa            {_phase_kappa(liquid_rs):.9f}   [physical: rs/sqrt(2)]")
    print(f"  orbital ansatz   filled LLL (fixed)")
    print(f"  LL rotation      none")
    print(f"  orbital SR       disabled")
    print(f"  Jastrow SR       enabled")
    print(f"  walk seed        {bud.liquid_measure['seed']}")
    print()
    print("CRYSTAL")
    print(f"  rs               {crystal_rs:.9f}")
    print(f"  kappa            {_phase_kappa(crystal_rs):.9f}   [physical: rs/sqrt(2)]")
    print(f"  orbital ansatz   LL rotation"
          f"{'' if ansatz == 'll_rotation' else ' (orbitals pinned at the seed)'}")
    print(f"  n_max            {crystal_nmax}")
    print(f"  LL basis         n = {', '.join(str(n) for n in range(crystal_nmax + 1))}")
    print(f"  n_bands          {crystal_nmax + 1}   (= n_max + 1)")
    print(f"  orbital SR       {'enabled' if ansatz == 'll_rotation' else 'disabled'}")
    print(f"  Jastrow SR       enabled")
    print(f"  optimisation     "
          f"{'joint orbital + Jastrow SR' if ansatz == 'll_rotation' else 'Jastrow only'}")
    init_id = CRYSTAL_INIT_ID
    print(f"  init L0          {bud.width_for(init_id):g}   "
          f"(init_id {init_id}, budget {bud.name!r})")
    print(f"  walk seed        {bud.crystal_measure['seed']}")
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
    print("=" * w)

    # Why this directory and not the bare `structure_slice/`.  A user who typed
    # couplings that differ from the historical ones in the tenth digit has still
    # asked for a different calculation, and that is worth saying out loud rather
    # than leaving them to wonder why the path grew a suffix.
    if os.path.basename(results_dir) != os.path.basename(RESULTS):
        diffs = []
        if liquid_rs != RS_LIQUID:
            diffs.append(f"liquid r_s {liquid_rs:g} (historical {RS_LIQUID:.9f})")
        if crystal_rs != RS_CRYSTAL:
            diffs.append(f"crystal r_s {crystal_rs:g} (historical {RS_CRYSTAL:.9f})")
        if ansatz != DEFAULT_ANSATZ:
            diffs.append(f"ansatz {ansatz}")
        if crystal_nmax != DEFAULT_CRYSTAL_NMAX:
            diffs.append(f"crystal n_max {crystal_nmax}")
        if bud.name != HISTORICAL_BUDGET:
            diffs.append(f"budget {bud.name} (historical {HISTORICAL_BUDGET})")
        print(f"  This is NOT the request `structure_slice/` holds, so it writes to its own")
        print(f"  directory.  Differs in: {'; '.join(diffs)}.")
        print()


def check_namespace_reuse(results_dir, slug):
    """Warn if this directory already holds a DIFFERENT request.

    The slug is short enough to read and therefore short enough, in principle,
    to collide: two couplings that round to the same six significant digits land
    on the same path.  `--resume` is safe regardless (it checks the
    full-precision digest), but a plain run would overwrite in silence, so this
    says so first.
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


def resolve_request(args):
    """The five resolved inputs, from parsed command-line arguments.

    Split out of `main` so that what the flags RESOLVE to is testable without
    paying for a VMC run -- and so that there is exactly one place that decides
    whether a value was the user's choice or the recipe's.

    `None` means "not given", which is deliberately kept distinct from "given
    the default value": the pre-flight reports which parameters the user
    actually chose, and a user who typed the historical coupling has still made
    a choice even though the number is the same.  The engine's own `nmax` is the
    highest Landau index, so `--crystal-nmax N` keeps n = 0 ... N and gives
    `n_bands = N + 1` bands.
    """
    return {
        "liquid_rs": RS_LIQUID if args.liquid_rs is None else float(args.liquid_rs),
        "crystal_rs": RS_CRYSTAL if args.crystal_rs is None else float(args.crystal_rs),
        "crystal_nmax": (DEFAULT_CRYSTAL_NMAX if args.crystal_nmax is None
                         else int(args.crystal_nmax)),
        # The ansatz is a property of THIS request, not of the budget file: the
        # budget says which protocol to walk, `--ansatz` says which wavefunction
        # protocol to run it with, and the two are recorded separately.
        "ansatz": args.ansatz or DEFAULT_ANSATZ,
        "budget": str(args.budget),
        "liquid_rs_given": args.liquid_rs is not None,
        "crystal_rs_given": args.crystal_rs is not None,
        "crystal_nmax_given": args.crystal_nmax is not None,
        "ansatz_given": args.ansatz is not None,
    }


def main(argv=None):
    args = parse_args(argv)
    # Resolve the user-facing budget alias ONCE, here, before anything reads it.
    # `--budget full` and `--budget reproduction` are ONE calculation, so they
    # must produce one slug and therefore one directory; resolving later would
    # let the alias mint a second directory holding the identical run.  It also
    # decides `is_historical_request`, and with it which requests may write to
    # the canonical `structure_slice/`.
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

    results_dir = namespace(RESULTS, liquid_rs, crystal_rs, ansatz,
                            crystal_nmax, args.budget)
    figdir = namespace(FIGDIR, liquid_rs, crystal_rs, ansatz,
                       crystal_nmax, args.budget)
    slug = request_slug(liquid_rs, crystal_rs, ansatz, crystal_nmax, args.budget)

    print("structure_slice -- clean VMC, from scratch   "
          f"(wigner_vmc {__version__})")
    print()
    print_preflight(liquid_rs, crystal_rs, crystal_nmax, ansatz, bud,
                    results_dir, figdir)
    check_namespace_reuse(results_dir, slug)

    points = {}
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
            print(f"  resumed from {_rel(results_dir)}/{name}/ "
                  f"(config digest matches)")
        else:
            if args.resume:
                print("  no matching saved run; computing from scratch")
            point = run_point(phase, rs, init_id, args.budget, ansatz=ansatz,
                              nmax=nmax, verbose=not args.quiet)
        save_point(results_dir, name, point, key)
        points[name] = point
        print(f"  E/N {point['energy_per_particle']:+.9f} +- {point['error']:.9f}"
              f"   acc {point['acceptance']:.4f}   {point['n_snapshots']} snapshots"
              f"   {point['seconds']:.0f}s")

    torus = torus_for()
    obs = {name: observables(points[name]["snaps"], torus) for name in ("liquid", "crystal")}

    extra = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "budget": args.budget,
        "budget_provenance": bud.provenance,
        "budget_protocol": {k: v for k, v in bud.protocol.items()},
        "quality": ("QUICK / not publication quality" if args.budget != "reproduction"
                    else "reproduction"),
        "N": N_ELECTRONS,
        # The resolved REQUEST.  This is what distinguishes one directory from
        # another, and it is recorded at full precision next to the short slug
        # that names the directory, so the slug can stay readable without ever
        # being the only record of what was run.
        "request_slug": slug,
        "request_is_the_historical_fig06": is_historical_request(
            liquid_rs, crystal_rs, ansatz, crystal_nmax, args.budget),
        "liquid_rs": float(liquid_rs),
        "crystal_rs": float(crystal_rs),
        "crystal_rs_source": ("--crystal-rs" if req["crystal_rs_given"]
                              else f"recipe default ({RS_CRYSTAL:.9f} = 64*sqrt(2))"),
        "liquid_rs_source": ("--liquid-rs" if req["liquid_rs_given"]
                             else f"recipe default ({RS_LIQUID:.9f} = 4*sqrt(2))"),
        "crystal_nmax": int(crystal_nmax),
        "crystal_nmax_source": ("--crystal-nmax" if req["crystal_nmax_given"]
                                else f"recipe default ({DEFAULT_CRYSTAL_NMAX})"),
        "crystal_n_bands": int(crystal_nmax) + 1,
        "crystal_ll_indices": list(range(int(crystal_nmax) + 1)),
        # The liquid's nmax is recorded as technical, with the reason, so that a
        # reader of this file cannot mistake it for a liquid truncation.
        "liquid_nmax_technical": LIQUID_NMAX,
        "liquid_nmax_is_a_physical_parameter": False,
        "liquid_ll_indices": [0],
        "phase_parameters": {
            "liquid": "filled_lll_fixed + Jastrow; no orbital sector, no LL rotation, "
                      "no truncation to choose",
            "crystal": "LL rotation + Jastrow; the LL basis is truncated at n_max",
        },
        "kappa_mode": "physical (rs/sqrt(2)); the Stage-2E rounded regression mode "
                      "is not used by this recipe",
        "ansatz": ansatz,
        "ansatz_source": ("--ansatz" if req["ansatz_given"] else "recipe default"),
        "ansatz_implemented": ["ll_rotation", "ll_rotation_pinned"],
        "ansatz_of_the_historical_crystal": (
            "NOT IMPLEMENTED -- the historical fig06 crystal is a determinant of "
            "site-centred magnetic Gaussians (notebook cell 18 crystal_wavefunction), "
            "a different wavefunction family from either implemented ansatz.  The "
            "families converge with the truncation; see DIVERGENCE_fig06_crystal.md."),
        "results_dir": _rel(results_dir),
        "figures_dir": _rel(figdir),
        "source_id": _source_id(),
        "wigner_vmc_version": __version__,
        "kappa_liquid": _phase_kappa(liquid_rs),
        "kappa_crystal": _phase_kappa(crystal_rs),
        # The protocols, spelled out as protocols rather than left inside
        # `budget_protocol` for a reader to decode.
        "optimization_protocol": {
            "sr_steps": bud.protocol["sr_steps"],
            "sr_sweeps": bud.protocol["sr_sweeps"],
            "sr_equil": bud.protocol["sr_equil"],
            "sr_snapshot_every": bud.protocol["sr_snap"],
            "sigma": bud.sr["sigma"],
            "target_acc": bud.sr["target_acc"],
            "rng_seed": 0,
            "liquid": "Jastrow-only SR (orbital sector fixed at v = 0)",
            "crystal": ("joint orbital + Jastrow SR" if ansatz == "ll_rotation"
                        else "Jastrow-only SR, orbitals pinned at the seed"),
        },
        "production_protocol": {
            "sweeps": bud.protocol["meas_sweeps"],
            "equil": bud.protocol["meas_equil"],
            "liquid_walk_seed": bud.liquid_measure["seed"],
            "liquid_walk_sigma": bud.liquid_measure["sigma"],
            "crystal_walk_seed": bud.crystal_measure["seed"],
            "crystal_walk_sigma": bud.crystal_measure["sigma"],
            "snapshot_every": 1,
        },
        "torus_area": float(torus.area),
        "torus_L1": torus.L1.tolist(),
        "torus_L2": torus.L2.tolist(),
        "deviations_from_legacy": [
            "Crystal ansatz: the historical fig06 crystal is a determinant of site-centred "
            "magnetic Gaussians (notebook cell 18); both of this package's crystal ansatze "
            "build LL-rotated orbitals instead.  The two families CONVERGE with the "
            "truncation -- measured at the historical couplings, the crystal's first-shell "
            "S(q) is 7.2817 at n_max=1, 11.5806 at n_max=2 and 12.1661 at n_max=3 (full "
            "budget), against the historical ensemble's 11.9154.  The n_max=1 shortfall "
            "was a TRUNCATION effect; an earlier version of this field called it a located "
            "divergence needing a third wavefunction family, and that claim is withdrawn.  "
            "See DIVERGENCE_fig06_crystal.md",
            "SR seed: the notebook used SR_SEED = 91; the clean driver uses the run's "
            "rng_seed (0). Different RNG stream.",
            "SR step size: the notebook used sigma 0.4 (liquid) / 0.3 (crystal); the clean "
            "budget carries one sr.sigma for both.",
            "snapshot_every: fixed at 1 by measure_decomposed, where the legacy struct_walk "
            "used 4 -- the clean walk stores 1000 snapshots to the legacy 250, from the same "
            "1500/500 walk.",
            "crystal walk start R0: the clean API's crystal_R0() (lat.sites + 0.25*jitter, "
            "seed 100+int(kappa)) rather than the notebook's crystal_sites_seed(kappa, L0, 4).",
            "kappa mode: physical r_s/sqrt(2), not the Stage-2E rounded regression "
            "value.  The two coincide exactly only at the historical couplings "
            "(kappa = 4 and 64), and not at the couplings of a general request.",
            "These deviations from the historical protocol apply to the historical "
            "request.  A run at user-chosen couplings is not a reproduction of "
            "fig06 at all and should not be read as one: its `request_slug` and "
            "`request_is_the_historical_fig06` say which it is.",
        ],
        "clean_only": ("No checkpoint, cached observable array, _diag product or old figure "
                       "was read to produce these numbers; every value comes from a VMC run "
                       "this script started."),
        "comparison_with_legacy_performed": bool(args.compare),
    }

    written = write_figures(figdir, points, obs, args.budget)
    meta = save_merged(results_dir, points, obs, extra)

    print()
    print("=" * 74)
    print("OUTPUTS")
    print("=" * 74)
    for name, vals in (("structure_factor.npz", "q, S_liquid, S_crystal, S_filled_LLL_exact"),
                       ("pair_correlation.npz", "r, g_liquid, g_crystal, g_crystal_sem"),
                       ("real_space_snapshot.npz", "R, lattice_sites, L1, L2")):
        print(f"  {_rel(os.path.join(results_dir, name))}")
        print(f"      {vals}")
    print(f"  {_rel(os.path.join(results_dir, 'run_metadata.json'))}")
    print(f"  {_rel(results_dir)}/{{liquid,crystal}}/{{run.npz,run.json}}")
    for path in written:
        print(f"  {_rel(path)}")

    if args.compare:
        legacy = compare_with_legacy()
        if legacy is not None:
            clean = {k: _summarise(obs[k], points[k]["n_snapshots"])
                     for k in ("liquid", "crystal")}
            _report_comparison(clean, legacy)
            meta["comparison"] = _comparison_record(clean, legacy)
        meta["comparison_with_legacy_performed"] = legacy is not None
        with open(os.path.join(results_dir, "run_metadata.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")

    print()
    if args.budget != "reproduction":
        print(f"  NOTE: budget {args.budget!r} is NOT publication quality -- the figures say so.")
    print(f"  total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
