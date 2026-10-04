"""The magnetoplasmon / SMA comparison (Fig. 4(b)) -- POST-PROCESSING ONLY.

    python examples/figure_construction/magnetoplasmon.py
    python examples/figure_construction/magnetoplasmon.py --rs 0 5 30 60

This file computes nothing quantum-mechanical.  It reads the ``S(q)`` that
``structure_slice.py`` has **already measured and saved**, reduces it to the
one-dimensional ``S(q)`` the figure needs, divides it into the
single-mode-approximation dispersion, and draws it against the classical
magnetoplasmon:

    saved S(q-vector)  ->  S(q)  ->  hbar^2 q^2 / (2 m S(q))  ->  vs omega_mp(q)

It does not start VMC, does not run the stochastic reconfiguration optimiser,
does not regenerate ``S(q)``, and does not touch the transition.  A test parses
this source and asserts that the machinery to do any of those appears nowhere in
it -- so the promise is structural rather than a matter of intention.

The reduction ``S(q-vector) -> S(q)`` is the whole of the physics here
---------------------------------------------------------------------
``structure_slice`` saves ``S`` on the supercell's *allowed momenta* ``q = m G1
+ n G2``, which form a two-dimensional disc, not a 1-D path.  294 momenta fall
into 30 shells of equal ``|q|``.  Dividing each of the 294 by hand and plotting
against ``|q|`` -- which is what an earlier version of this file did -- is wrong
in two different ways for the two different phases, and the paper's Fig. 4
caption says exactly which reduction each one needs: *"The data are rotation
averaged in the liquid phase and taken along the x axis in the crystal phase."*

**Liquid: rotation average.**  A liquid is rotationally invariant, so the
physical one-dimensional structure factor is

    \bar S_L(q) = < S_L(q-vector) >_{|q-vector| = q},

and then ``Omega_L(q) = q^2 / (2 \bar S_L(q))``.  The order matters and is not a
detail: averaging ``S`` first is what a rotationally-invariant state *means*,
whereas ``<q^2/(2S)>`` averages the reciprocal and is a different number
wherever the measurement carries directional noise.  Both are computed and
printed so the size of the difference is visible rather than asserted.

**Crystal: one direction.**  A Wigner crystal is *not* isotropic --
``S_C(q-vector) != S_C(|q|)`` -- so there is nothing to average, and averaging
would destroy the Bragg peak that is the entire content of the crystal curve.
The cut is taken along the paper's ``x`` axis, which is a *specific* direction.

**Where the paper's x axis is, in this package's frame.**  This is a real
mapping question, not a convention to assume.  The paper's own lattice vectors
(``QuantumHallVMC``, ``A1 = [sqrt(3)/2, -1/2] a``) sit at ``-30 deg``; this
package's ``Torus`` puts ``A1`` along ``(1, 0)``.  The two frames therefore
differ by a 30 degree rotation, and the paper's ``x`` -- the vector ``(1, 0)``
in the paper's Cartesian frame -- is the vector ``(cos 30, sin 30)`` here.  That
is the direction of ``G1 + G2``, and it is a **reciprocal-lattice (Bragg)**
direction of the crystal, which is why a cut along it passes through ``Q_WC``
and shows the roton.  The torus's own first Cartesian axis ``(1, 0)`` is *not*
the paper's ``x``; it is 30 degrees away, it is a Gamma-K direction, and its
allowed momenta never reach ``Q_WC``.  ``--crystal-direction`` selects either,
and the default is the paper's.

Where the numbers come from
---------------------------
``structure_slice.py`` saves, per request, a ``structure_factor.npz`` holding the
allowed momenta of the supercell -- the full vectors ``q`` as well as their
magnitudes ``qn`` -- and ``S(q)`` on them, for both phases, plus the exact
filled-Landau-level reference.  That array is used here **unchanged**: the
reduction above is a selection and a mean of stored samples, and nothing is
smoothed, fitted, resampled or interpolated.  ``run_metadata.json`` records the
directory each curve came from so a reader can go and check.

The ``r_s = 0`` curve is the exception and is deliberately analytic -- at
``r_s = 0`` the interaction is switched off, the ground state is the filled
lowest Landau level, and ``S_0(q) = 1 - exp(-q^2 l_B^2 / 2)`` is exact at any
``N``.  It is not a fit and it is not a VMC result; it is
``wigner_vmc.analysis.structure.exact_lll_sq``, the same helper the rest of the
package uses, so there is one definition of that curve as well.

Units
-----
Everything is in ``hbar = l_B = omega_c = 1``, the convention the engine already
uses (``physics/hamiltonian.py``).  Two conversions matter and neither is a
tunable number:

``q`` is stored in units of ``1/l_B``, so the SMA dispersion is

    Omega(q) / (hbar omega_c) = q^2 / (2 S(q))

with no prefactor at all, because ``hbar^2 / (m l_B^2) = hbar omega_c`` exactly.

The classical magnetoplasmon is ``omega_mp^2 = omega_c^2 + (2 pi n e^2/m) q``.
At ``nu = 1`` the density is ``n = 1/(2 pi l_B^2)``, so the second term is
``e^2 q / (m l_B^2)``, and dividing by ``omega_c^2 = hbar^2/(m^2 l_B^4)`` leaves

    omega_mp^2 / omega_c^2 = 1 + (e^2 m l_B^2 / hbar^2) q = 1 + kappa q

because ``kappa`` IS that ratio -- ``physics/hamiltonian.py`` records it as
"charge absorbed into kappa: kappa = e^2 / (4 pi eps0 l_B hbar omega_c)".  So the
classical curve carries the coupling of the state it is compared against, and it
is obtained from the engine's own ``kappa_from_rs`` rather than from a scale
factor chosen to make the curves meet.

The horizontal axis is ``q / sqrt(n)``, which at ``nu = 1`` is
``q l_B sqrt(2 pi)``.  One factor, derived from ``n = nu/(2 pi l_B^2)``, printed
in the metadata and checked by a test -- not a literal.

A consequence worth stating before any curve is read: because the two agree only
through their leading small-``q`` behaviour, the comparison is a statement about
``q -> 0``.  The smallest momentum a finite torus can carry is its own ``|G_1|``,
so the test is only as good as ``q_min``, which this file prints per curve.  The
smallest momentum is **not** the ``q -> 0`` limit and nothing here claims it is.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from wigner_vmc import __version__
from wigner_vmc.analysis import structure as st
from wigner_vmc.physics.hamiltonian import kappa_from_rs

# ==========================================================================
# where things live
# ==========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))          # .../wigner_vmc_clean

RESULTS = os.path.join(CLEAN, "results", "figure_construction", "magnetoplasmon")
FIGDIR = os.path.join(CLEAN, "figures", "figure_construction", "magnetoplasmon")

#: Where the ``S(q)`` this file consumes lives.  ``structure_slice`` writes the
#: historical request to ``.../structure_slice`` and every other request to a
#: suffixed sibling (``structure_slice_liquid_rs75__crystal_rs75__...``), so the
#: search has to cover the family and not just the canonical directory.
SOURCE_FAMILY = os.path.join(CLEAN, "results", "figure_construction",
                             "structure_slice")

#: Filling factor.  Fixed at 1 for this figure, the paper's case; it enters the
#: density, and therefore both the axis and the plasma frequency.
NU = 1.0

#: The paper's points, and what each one is.
DEFAULT_RS = (0.0, 5.0, 30.0, 60.0)

#: The phase boundary the paper fixes.  Recorded so the metadata says which
#: value this figure took as given -- NOT used to decide anything, and this file
#: never computes a transition.
RS_C = 47.0

#: ``mtime`` of the corrected ``vmc/sr.py`` -- the reversed-conjugation fix at
#: line 219.  ``structure_slice`` results written before it hold a crystal whose
#: first shell is uniaxial, so their ``S_crystal`` is not the structure factor of
#: the ansatz this package now implements.  Sources older than this are refused
#: rather than silently drawn.  The liquid is not affected by that fix, but the
#: check is applied to both phases anyway: an exemption that has to be argued
#: per-phase is an exemption that will eventually be argued wrongly, and the
#: cost of being strict is only which curves are available.
PROJECTION_FIX_MTIME = time.mktime(time.strptime("2026-10-03 16:29:56",
                                                 "%Y-%m-%d %H:%M:%S"))

DEFAULT_BUDGET = "reproduction"

#: ``structure_slice.py`` accepts ``--budget full`` as an alias for the
#: ``reproduction`` config and resolves it before writing anything, so the
#: directories on disk are named ``reproduction``.  Matching the stored string
#: exactly would therefore refuse ``--source-budget full`` for data that exists
#: -- the brief's own spelling failing against the brief's own data.  The alias
#: is resolved here, once, the same way the recipe that wrote the files does.
BUDGET_ALIASES = {"full": "reproduction"}


def resolve_budget(budget):
    """The name the stored directories actually use."""
    return BUDGET_ALIASES.get(str(budget), str(budget))


DEFAULT_NMAX = 3
DEFAULT_N = 36

# ==========================================================================
# the two reductions, and the one direction convention they need
# ==========================================================================
#: Two momenta are "the same |q|" when they agree to this.  The stored shells
#: are separated by ~0.03 in |q| at the closest and by ~0.1 within a shell, five
#: orders of magnitude above this, so the tolerance collapses float noise and
#: nothing else -- the same reasoning ``Torus.allowed_momenta`` documents for
#: its own rounded sort key.
SHELL_TOL = 1e-9

#: A momentum is "on the cut" when its component perpendicular to the chosen
#: direction is below this.  The cut is a property of the lattice, not a
#: tolerance: the collinear momenta are exactly collinear (measured residual
#: 0.0), and the nearest off-axis momentum sits ~0.19 away, so there is no
#: judgement in this number and the code asserts the residual it achieves.
COLLINEAR_TOL = 1e-9

#: How the crystal's 1-D cut is oriented.
#:
#: ``bragg``   -- the paper's ``x`` axis: a reciprocal-lattice direction of the
#:                Wigner crystal.  This is the direction the paper's Fig. 4
#:                caption means, and the only one whose allowed momenta reach
#:                ``Q_WC``, so it is the only one that can show the roton.
#: ``clean-x`` -- the first Cartesian axis of this package's ``Torus``.  Offered
#:                because it is the *other* reasonable reading of "the x axis",
#:                and wrong for exactly that reason: it is not the paper's
#:                frame.  Kept selectable so the claim is checkable.
CRYSTAL_DIRECTIONS = ("bragg", "clean-x")
DEFAULT_CRYSTAL_DIRECTION = "bragg"


def torus_for(n_electrons=DEFAULT_N):
    """The supercell ``Torus`` the stored ``S(q)`` was measured on.

    The same object the recipe and the API's own ``structure_factor`` build,
    derived from the frozen area so it cannot drift: ``N`` electrons at ``nu = 1``
    occupy ``N * 2 pi l_B^2``, and a square-ish ``sqrt(N) x sqrt(N)`` array of
    cells is what ``structure_slice`` runs.
    """
    n = int(n_electrons)
    n_side = int(round(math.sqrt(n)))
    if n_side ** 2 != n:
        raise ValueError(
            f"N = {n} is not a perfect square, so it is not a sqrt(N) x sqrt(N) "
            f"supercell and the stored momentum grid is not this torus's.")
    return st.Torus(2.0 * math.pi * n, n_electrons=n, n_cells_per_side=n_side)


def crystal_direction(torus, choice=DEFAULT_CRYSTAL_DIRECTION):
    """The crystal's propagation direction as a unit vector, in ``torus``'s frame.

    The paper's Fig. 4 caption fixes the *reduction* ("taken along the x axis in
    the crystal phase") but not the orientation, and the orientation is the part
    that does not survive a change of package.  The paper builds its lattice from

        A1 = [sqrt(3)/2, -1/2] a,      A2 = [sqrt(3)/2, +1/2] a

    so its primitive vectors sit at ``-30`` and ``+30`` degrees.  This package's
    ``Torus`` uses ``A1 = (a, 0)``, ``A2 = (a/2, sqrt(3) a/2)`` -- the same
    triangular lattice, rotated by ``+30`` degrees.  Carrying the paper's ``x``,
    which is ``(1, 0)`` in the paper's frame, into this frame means rotating it
    by ``+30`` degrees:

        paper x  =  (cos 30, sin 30)  =  direction of (G1 + G2).

    ``G1 + G2`` is a reciprocal-lattice vector of the crystal divided by six, so
    this is a Bragg direction -- which is the check that it is the right one, and
    not merely a plausible one: a cut along a Bragg direction passes through
    ``Q_WC = |g1|``, and a cut along anything else does not.  On this torus the
    six vectors of the first Bragg star sit at ``+/-30``, ``+/-90`` and ``+/-150``
    degrees, and the crystal's point group maps them into one another, so all six
    are the same cut up to measurement noise; ``G1 + G2`` is the representative
    the rotation selects.

    ``clean-x`` returns ``(1, 0)`` instead -- the torus's own first axis, which
    points along a real-space nearest-neighbour bond and is a Gamma-K direction.
    It is 30 degrees from the paper's ``x``, and the momenta on it never reach
    ``Q_WC``.
    """
    if choice == "bragg":
        v = np.asarray(torus.G1, float) + np.asarray(torus.G2, float)
    elif choice == "clean-x":
        v = np.array([1.0, 0.0])
    else:
        raise ValueError(
            f"unknown crystal direction {choice!r}; expected one of "
            f"{CRYSTAL_DIRECTIONS}")
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("degenerate crystal direction")
    return v / n


def direction_angle_deg(dhat):
    """The direction's polar angle in degrees, for the record."""
    return math.degrees(math.atan2(float(dhat[1]), float(dhat[0])))


# ==========================================================================
# units.  Derived, printed, and tested -- never a literal in the plot call.
# ==========================================================================
def rel(path):
    """A path relative to the package root, or the absolute one if it cannot be.

    ``os.path.relpath`` raises when the two paths are on different drives, which
    on this machine means every run whose output lands outside ``E:`` -- a
    temporary directory in a test, or a user's own ``--out``.  A banner that
    crashes the run while printing where the run went is a poor trade for a
    tidier string, so this falls back instead.
    """
    try:
        return os.path.relpath(path, CLEAN)
    except ValueError:
        return path


def q_over_sqrt_n(q_lb, nu=NU):
    """``q/sqrt(n)`` from ``q`` in units of ``1/l_B``.

    ``n = nu / (2 pi l_B^2)``, so ``sqrt(n) = sqrt(nu/(2 pi))/l_B`` and
    ``q/sqrt(n) = q l_B sqrt(2 pi / nu)``.  Written as the derivation rather
    than as the number it evaluates to, so that a change of ``nu`` cannot leave
    a stale constant behind on the axis.
    """
    return np.asarray(q_lb, float) * math.sqrt(2.0 * math.pi / float(nu))


def omega_sma(q_lb, s_q):
    """``hbar^2 q^2 / (2 m S(q))`` in units of ``hbar omega_c``.

    ``hbar^2/(m l_B^2) = hbar omega_c`` identically, so with ``q`` in units of
    ``1/l_B`` the whole conversion is the factor 1/2 that is already in the
    formula.  ``q = 0`` is a removable singularity -- the exact small-``q``
    behaviour is ``S -> q^2/2``, so the limit is ``hbar omega_c`` -- but it is
    NOT silently patched here: ``S(q) = 0`` raises, and the storage invariant is
    that no source array contains ``q = 0`` (the torus excludes it), so a zero
    would mean a corrupt file and should say so.
    """
    q = np.asarray(q_lb, float)
    s = np.asarray(s_q, float)
    bad = ~np.isfinite(q) | (q <= 0.0) | ~np.isfinite(s) | (s <= 0.0)
    if np.any(bad):
        # Every one of these is a corrupt input rather than a physical endpoint.
        # The exact small-q limit is finite, but it is reached from the analytic
        # branch; a *stored* zero or an inf means the file is not a structure
        # factor, and the three ways that shows up must not be papered over.
        # S = inf is the quiet one: q^2/(2*inf) = 0.0, a perfectly finite number
        # that would draw a curve sitting on the axis and look like physics.
        i = int(np.argmax(bad))
        raise ValueError(
            f"non-positive or non-finite S(q) and/or q at {int(bad.sum())} of "
            f"{bad.size} momenta, first at q = {float(q[i]):.6g}, "
            f"S = {float(s[i]):.6g}.  hbar^2 q^2/(2 m S) is a structure-factor "
            "dispersion: it needs q > 0 and a finite S > 0.")
    return q ** 2 / (2.0 * s)


def omega_magnetoplasmon(q_lb, kappa):
    """The classical magnetoplasmon ``omega_mp`` in units of ``hbar omega_c``.

    ``omega_mp^2 = omega_c^2 + (2 pi n e^2/m) q`` becomes ``1 + kappa q`` in
    these units -- see the module docstring for the two-line derivation.  At
    ``kappa = 0`` (``r_s = 0``, no interaction) this is the cyclotron frequency
    exactly, which is the limit the ``r_s = 0`` SMA curve has to approach.
    """
    return np.sqrt(1.0 + float(kappa) * np.asarray(q_lb, float))


def q_wc_lattice_constant(nu=NU):
    """The triangular Wigner crystal's nearest-neighbour distance, in ``l_B``.

    One electron per ``2 pi l_B^2`` at ``nu = 1``; a triangular lattice of
    spacing ``a`` has ``(sqrt(3)/2) a^2`` per site, so ``a^2 = 4 pi l_B^2/sqrt(3)``.
    """
    return math.sqrt(4.0 * math.pi / (math.sqrt(3.0) * float(nu)))


def q_wc(nu=NU):
    """``|G_1|`` of the triangular lattice, in units of ``1/l_B``.

    This is the wavevector the paper's roton minimum sits near.  Computed from
    the geometry rather than read from the data, and cross-checked against the
    ``g1_mag`` that ``structure_slice`` recorded when the file is present.
    """
    return 4.0 * math.pi / (math.sqrt(3.0) * q_wc_lattice_constant(nu))


# ==========================================================================
# finding the S(q) that already exists
# ==========================================================================
def _slug_rs(rs):
    """The coupling as ``structure_slice`` writes it into a directory name.

    Duplicated from the recipe, deliberately, and pinned by a test that asserts
    the two agree on a grid of couplings: importing the recipe would pull the
    whole VMC entry point into a file whose entire contract is that it cannot
    run one.  A duplicated rule with a test that fails when it drifts is a
    weaker thing than one definition -- but a stronger thing than a figure that
    silently reads the wrong directory.
    """
    rs = float(rs)
    if abs(rs - round(rs)) < 1e-9:
        return f"{rs:.0f}"
    return f"{rs:.6g}"


def _source_dirs():
    """Every ``structure_slice`` result directory in the family, sorted."""
    return sorted({d for d in glob.glob(SOURCE_FAMILY)
                   + glob.glob(SOURCE_FAMILY + "_*") if os.path.isdir(d)})


def index_sources(verbose=False):
    """Index every saved ``S(q)`` by what it actually is.

    Read from each run's own ``run_metadata.json`` -- the phase's ``r_s``,
    ``nmax`` and the budget -- and NEVER from the directory name, because the
    name is a short rounded grouping while the metadata holds the full-precision
    request.  ``rs = 5`` and ``rs = 5.567`` produce names a reader can confuse
    and metadata a program cannot.

    Returns a list of dicts, one per usable source, newest last.
    """
    out = []
    for d in _source_dirs():
        meta_path = os.path.join(d, "run_metadata.json")
        npz_path = os.path.join(d, "structure_factor.npz")
        if not (os.path.isfile(meta_path) and os.path.isfile(npz_path)):
            if verbose:
                print(f"  skip   {os.path.basename(d)}: no metadata or no S(q)")
            continue
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        for phase in ("liquid", "crystal"):
            p = meta.get(phase) or {}
            if p.get("rs") is None:
                continue
            out.append({
                "phase": phase,
                "rs": float(p["rs"]),
                "kappa": float(p.get("kappa", kappa_from_rs(p["rs"]))),
                "nmax": p.get("nmax"),
                "budget": str(meta.get("budget", "")),
                "dir": d,
                "npz": npz_path,
                "schema": str(meta.get("run_metadata_schema", "")),
                "mtime": os.path.getmtime(npz_path),
                "post_fix": os.path.getmtime(npz_path) >= PROJECTION_FIX_MTIME,
            })
    out.sort(key=lambda r: (r["phase"], r["rs"], r["mtime"]))
    return out


def find_source(index, phase, rs, budget, nmax=None):
    """The saved ``S(q)`` for one point, or a ``LookupError`` saying how to make it.

    Matching is exact on ``r_s`` -- ``5`` is not ``5.567``, and the figure must
    not quietly draw a nearby coupling under a label that says otherwise.  The
    budget and the crystal truncation are selectors: several sources can exist
    for one coupling, and which one is read changes the curve.
    """
    want = resolve_budget(budget)
    cands = [r for r in index
             if r["phase"] == phase and r["rs"] == float(rs)
             and r["budget"] == want]
    if phase == "crystal" and nmax is not None:
        cands = [r for r in cands if r["nmax"] == int(nmax)]
    if not cands:
        raise LookupError(_missing_message(index, phase, rs, budget, nmax))
    # Prefer a post-fix source, newest first.  One coupling can be measured more
    # than once -- crystal r_s = 100 nmax = 2 quick exists both before and after
    # the sr.py fix, from two different campaigns -- and taking the oldest
    # candidate would refuse a coupling whose usable file is sitting right next
    # to the stale one.  So the refusal is only correct when NO candidate is
    # post-fix, which is what this ordering makes it.
    cands.sort(key=lambda r: (r["post_fix"], r["mtime"]), reverse=True)
    if not cands[0]["post_fix"]:
        r = cands[0]
        raise LookupError(
            f"{phase} r_s = {rs:g} at budget {want!r} exists at\n"
            f"    {rel(r['dir'])}\n"
            f"  but it was written {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['mtime']))}, "
            f"before the vmc/sr.py:219\n"
            f"  conjugation fix ({time.strftime('%Y-%m-%d %H:%M', time.localtime(PROJECTION_FIX_MTIME))}).  "
            f"Its S(q) is not the structure\n"
            f"  factor of the ansatz this package now implements, so it is not drawn.\n"
            f"  Re-run structure_slice.py to regenerate it.")
    return cands[0]


def _missing_message(index, phase, rs, budget, nmax):
    """Tell the user which command produces the missing point.  Do not run it.

    The exact command, with the other phase left at a coupling that also exists,
    because ``structure_slice.py`` measures BOTH phases in one invocation -- so
    asking for one new curve necessarily means choosing a coupling for the
    other, and the user should see that choice rather than discover it.

    The couplings listed as available are filtered by the same two predicates
    the lookup itself applies -- the truncation and the post-fix gate -- because
    a list of what is *on disk* is not an answer to "why was my coupling not
    found".  It used to list every source of the phase, which made the message
    actively misleading: asking for ``--crystal-rs 100`` at the default
    ``nmax = 3`` printed ``100 (quick)`` in the very message explaining that
    ``100`` could not be found.  The value was never the problem; the
    truncation was, and it is one flag away.  Couplings that are present but
    unusable for the requested ``r_s`` are now named separately with what they
    failed.
    """
    want = resolve_budget(budget)
    rows = [r for r in index if r["phase"] == phase]

    def _usable(r):
        if not r["post_fix"]:
            return False
        return not (phase == "crystal" and nmax is not None
                    and r["nmax"] != int(nmax))

    have = sorted({(r["rs"], r["budget"]) for r in rows if _usable(r)})
    opts = ", ".join(f"{r:g} ({b})" for r, b in have) or "none"
    note = ("" if want == str(budget) else
            f"\n  (--budget {budget} resolves to {want!r}, the name on disk)")

    # Present but unusable, and why.  Almost always the truncation, which is
    # fixable by adding one flag -- so say which one rather than leaving the
    # user to guess that their number is the problem.  One line per truncation,
    # reporting the best file there: a coupling can have both a pre-fix and a
    # post-fix file at the same nmax, and naming the stale one as the reason
    # would be true and useless when a good one sits next to it.
    near = {}
    for r in rows:
        if r["rs"] != float(rs) or r["budget"] != want or _usable(r):
            continue
        near[r["nmax"]] = near.get(r["nmax"], False) or bool(r["post_fix"])
    for n in sorted(near):
        if near[n]:
            note += (f"\n  r_s = {rs:g} IS on disk at nmax = {n}, not "
                     f"nmax = {nmax}; add --crystal-nmax {n} to draw it.")
        else:
            note += (f"\n  r_s = {rs:g} at nmax = {n} predates the "
                     f"vmc/sr.py:219 fix and is refused; re-run "
                     f"structure_slice.py to regenerate it.")

    if phase == "liquid":
        flag, other = "--liquid-rs", "--crystal-rs"
    else:
        flag, other = "--crystal-rs", "--liquid-rs"
    nmax_flag = " --crystal-nmax %d" % int(nmax) if (
        phase == "crystal" and nmax is not None) else ""
    return (
        f"no saved structure_slice S(q) for {phase} r_s = {rs:g} at budget "
        f"{want!r}.\n"
        f"  available {phase} couplings: {opts}{note}\n"
        f"  this file only post-processes; it will not run VMC.  To create it, run\n"
        f"    python examples/figure_construction/structure_slice.py "
        f"{flag} {rs:g} {other} <coupling>{nmax_flag} --budget {budget}\n"
        f"  and then re-run this command.")


def load_curve(src):
    """The saved arrays for one source.  ``S(q)`` is returned exactly as stored."""
    z = np.load(src["npz"])
    key = "S_liquid" if src["phase"] == "liquid" else "S_crystal"
    qn = np.asarray(z["qn"], float)
    S = np.asarray(z[key], float)
    if qn.shape != S.shape:
        raise ValueError(f"{src['npz']}: qn {qn.shape} and {key} {S.shape} differ")
    if np.any(qn <= 0.0):
        raise ValueError(
            f"{src['npz']} contains q = 0.  The torus's allowed momenta exclude "
            "it, so this file is not what it claims to be.")
    return qn, S


def load_momenta(src):
    """The saved momentum VECTORS, shape ``(n_q, 2)``.

    A second reader rather than a wider ``load_curve``, because the two answer
    different questions and the first one's contract is that ``S(q)`` arrives
    untouched.  The reduction in this file needs to know *which direction* each
    stored ``S`` belongs to; ``structure_slice`` saves the vectors alongside the
    magnitudes for exactly this, and this raises rather than guessing if a file
    predates that.
    """
    z = np.load(src["npz"])
    if "q" not in z:
        raise ValueError(
            f"{src['npz']} stores no momentum vectors ('q').  The isotropic "
            "reduction and the crystal cut both need to know which direction "
            "each S belongs to, and a magnitude alone does not say.  Re-run "
            "structure_slice.py to regenerate it.")
    q = np.asarray(z["q"], float)
    if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] == 0:
        raise ValueError(f"{src['npz']}: 'q' has shape {q.shape}, expected (n, 2)")
    return q


def check_frame(qvec, torus, src):
    """Assert the stored momenta really are ``m G1 + n G2`` of this torus.

    The reduction reads the direction convention out of ``torus``, so a source
    measured on a different lattice would be cut along the wrong axis and would
    still produce a smooth, plausible curve.  Expressing every stored vector in
    units of ``(G1, G2)`` and requiring integers is the cheapest check that the
    frame is the one assumed -- and it is exact rather than approximate, because
    the allowed momenta are integer combinations by construction.
    """
    basis = np.column_stack([torus.G1, torus.G2])
    coeff = qvec @ np.linalg.inv(basis).T
    residual = np.abs(coeff - np.round(coeff)).max()
    if residual > 1e-6:
        raise ValueError(
            f"{src['npz']}: the stored momenta are not integer combinations of "
            f"this torus's G1, G2 (max residual {residual:.3g}).  The file was "
            "measured on a different lattice, so the crystal cut and the "
            "shell grouping would be taken in the wrong frame.")


# ==========================================================================
# the reduction  S(q-vector) -> S(q)
# ==========================================================================
def shells_of(qn, tol=SHELL_TOL):
    """Indices grouped by equal ``|q|``, in ascending ``|q|``.

    A shell is a set, not a tolerance band: the grouping is by the rounded
    magnitude, and ``shell_average`` asserts afterwards that every member of
    every shell agreed to within ``tol``.  That ordering -- group, then verify
    -- is deliberate.  A routine that decided group membership by a running
    comparison would put a momentum in the previous shell whenever the gap
    happened to fall under the tolerance, and the assertion is what turns that
    silent corruption into a failure.
    """
    qn = np.asarray(qn, float)
    if qn.size == 0:
        return []
    order = np.argsort(qn, kind="stable")
    key = np.round(qn[order], int(round(-math.log10(tol))))
    breaks = np.flatnonzero(np.diff(key) != 0.0) + 1
    return np.split(order, breaks)


def shell_average(qvec, S, tol=SHELL_TOL):
    """The LIQUID reduction: ``\\bar S(q) = <S(q-vector)>_{|q|=q}``.

    A rotationally invariant state has one ``S`` per magnitude, so the shell mean
    is the physical quantity and the spread within a shell is measurement noise
    (plus whatever the finite torus does not make exactly isotropic).  Both are
    returned: the mean is what enters ``Omega``, and the spread is the honest
    error bar on it.

    **The mean is taken on ``S``, not on ``1/S``.**  The two orders give
    different numbers whenever a shell has any directional spread at all, and
    only this one is the structure factor of an isotropic state --
    ``q^2/(2<S>)`` versus ``<q^2/(2S)>``.  ``wrong_order_omega`` is computed
    beside it so the size of the difference is a printed number rather than a
    remark.

    Returns a dict with ``q``, ``S``, ``count``, ``spread`` and the two
    diagnostics.
    """
    qvec = np.asarray(qvec, float)
    S = np.asarray(S, float)
    qn = np.linalg.norm(qvec, axis=1)
    groups = shells_of(qn, tol)

    q, mean, count, spread, within = [], [], [], [], []
    for idx in groups:
        q.append(float(qn[idx].mean()))
        mean.append(float(S[idx].mean()))
        count.append(int(idx.size))
        spread.append(float(S[idx].max() - S[idx].min()))
        # Validation 1, enforced rather than assumed: a shell that mixed two
        # magnitudes would be averaging over directions AND over |q|, and the
        # curve it produced would be smooth and wrong.
        within.append(float(qn[idx].max() - qn[idx].min()))
    q = np.asarray(q)
    mean = np.asarray(mean)
    if np.any(np.asarray(within) > tol):
        i = int(np.argmax(within))
        raise AssertionError(
            f"shell {i} at |q| = {q[i]:.6g} spans {within[i]:.3g} in |q| "
            f"(tolerance {tol:g}); it is not a degenerate shell.")

    # The order-of-operations diagnostic.  Same shells, same stored samples, the
    # other order -- reported, never plotted.
    wrong = []
    for idx in groups:
        wrong.append(float(np.mean(qn[idx] ** 2 / (2.0 * S[idx]))))
    right = q ** 2 / (2.0 * mean)
    wrong = np.asarray(wrong)

    return {
        "reduction": "shell-mean",
        "q": q,
        "S": mean,
        "count": np.asarray(count, int),
        "spread": np.asarray(spread),
        "spread_rel": np.asarray(spread) / np.where(mean > 0, mean, np.nan),
        "within_shell_q_spread": float(np.max(within)) if len(within) else 0.0,
        "n_vectors_averaged": int(S.size),
        "n_shells": int(q.size),
        "wrong_order_omega": wrong,
        "right_order_omega": right,
        "order_gap_rel": (wrong - right) / right,
    }


def directional_cut(qvec, S, dhat, tol=COLLINEAR_TOL):
    """The CRYSTAL reduction: ``S_C(q) = S_C(q dhat)`` for ONE direction.

    No averaging.  A crystal has directional structure -- ``S_C(q-vector)`` is
    not a function of ``|q|`` -- so the only honest 1-D curve is a cut, and a
    shell mean would mix a Bragg peak with its neighbours and destroy it.

    One side of the line only (``q . dhat > 0``).  ``S(-q) = S(q)`` exactly for
    a real charge density, so the two halves carry the same information; keeping
    both would put two markers on every point and imply a resolution the torus
    does not have.  The equality is verified, not assumed.

    Every returned point is on the line to within ``tol``, and the achieved
    residual is returned so a caller can see there was no interpolation: the
    momenta either are collinear or they are not, and nothing is projected onto
    the axis to make a fuller curve.
    """
    qvec = np.asarray(qvec, float)
    S = np.asarray(S, float)
    dhat = np.asarray(dhat, float)
    dhat = dhat / np.linalg.norm(dhat)
    perp_axis = np.array([-dhat[1], dhat[0]])
    along = qvec @ dhat
    perp = np.abs(qvec @ perp_axis)

    sel = (perp <= tol) & (along > 0.0)
    if not np.any(sel):
        raise ValueError(
            "the torus carries no momentum along the requested crystal "
            "direction; the cut cannot be taken and nothing is interpolated "
            "to manufacture one.")
    order = np.argsort(along[sel], kind="stable")
    q = along[sel][order]
    S_cut = S[sel][order]

    # S(-q) = S(q): the mirrored partner of every selected point must carry the
    # same stored value if it is present at all.  Verified, because "we only
    # took one side" is only safe if the two sides agree.
    neg = ~sel & (perp <= tol) & (along < 0.0)
    mirrored = np.full(q.shape, np.nan)
    for k, qi in enumerate(q):
        j = np.flatnonzero(neg & np.isclose(-along, qi, rtol=0, atol=1e-9))
        if j.size:
            mirrored[k] = float(S[j[0]])
    both = np.isfinite(mirrored)
    mirror_gap = (float(np.abs(S_cut[both] - mirrored[both]).max())
                  if np.any(both) else 0.0)

    return {
        "reduction": "directional",
        "q": q,
        "S": S_cut,
        "count": np.ones(q.size, int),
        "spread": np.zeros(q.size),
        "spread_rel": np.zeros(q.size),
        "direction": dhat,
        "direction_deg": direction_angle_deg(dhat),
        "n_points": int(q.size),
        "max_perp_residual": float(perp[sel].max()),
        "n_vectors_in_source": int(S.size),
        "n_mirror_pairs_checked": int(np.count_nonzero(both)),
        "mirror_max_abs_gap": mirror_gap,
    }


def reduce_curve(qvec, S, phase, dhat):
    """Dispatch the reduction by phase.  One place, so the two cannot be swapped."""
    if phase == "liquid":
        return shell_average(qvec, S)
    if phase == "crystal":
        return directional_cut(qvec, S, dhat)
    raise ValueError(f"unknown phase {phase!r}")


# ==========================================================================
# assembling the curves
# ==========================================================================
def build_curves(args):
    """One record per requested ``r_s``: the SMA curve, and its classical partner.

    ``r_s = 0`` needs no saved data -- the filled LLL is exact -- and is
    evaluated on a dense grid because it is a formula, not a measurement.  Every
    other curve is the stored array, reduced for its phase and nothing else: the
    liquid by shell mean, the crystal by a cut along one direction.
    """
    index = index_sources(verbose=args.verbose)
    torus = torus_for(int(args.n_electrons))
    dhat = crystal_direction(torus, args.crystal_direction)
    curves = []
    for rs in args.rs:
        if float(rs) == 0.0:
            q = np.linspace(args.q_min_analytic, args.q_max_analytic, 1200)
            red = {"reduction": "analytic", "q": q, "S": st.exact_lll_sq(q),
                   "count": None, "spread": None}
            curves.append({
                "rs": 0.0, "phase": "analytic", "source": None,
                "reduction": red,
                "q": red["q"], "S": red["S"],
                "raw_qvec": None, "raw_qn": None, "raw_S": None,
                "note": "filled lowest Landau level, S_0(q) = 1 - exp(-q^2/2), "
                        "exact at any N; no VMC",
            })
            continue
        phase = args.phase_of(float(rs))
        src = find_source(index, phase, float(rs), args.source_budget,
                          args.crystal_nmax)
        qvec = load_momenta(src)
        check_frame(qvec, torus, src)
        qn, S = load_curve(src)
        red = reduce_curve(qvec, S, phase, dhat)
        curves.append({
            "rs": float(rs), "phase": phase, "source": src,
            "reduction": red,
            "q": red["q"], "S": red["S"],
            # The stored arrays, kept only so the old-versus-new diagnostic can
            # be computed against the very numbers the previous reduction used.
            # They are not plotted.
            "raw_qvec": qvec, "raw_qn": qn, "raw_S": S,
            "note": f"structure_slice {phase} r_s = {src['rs']:g}, "
                    f"nmax = {src['nmax']}, budget {src['budget']}",
        })
    for c in curves:
        c["kappa"] = kappa_from_rs(c["rs"])
        c["x"] = q_over_sqrt_n(c["q"])
        c["Omega"] = omega_sma(c["q"], c["S"])
        c["omega_mp"] = omega_magnetoplasmon(c["q"], c["kappa"])
    return curves, index


def small_q_table(curves):
    """The convergence diagnostic, one row per measured curve.

    The paper's claim is that ``Omega(q) -> hbar omega_mp(q)`` as ``q -> 0``.
    A finite torus cannot carry ``q = 0``, so the best available test is at its
    smallest momentum, and printing that momentum beside the difference is the
    point: a relative difference quoted without the ``q`` it was taken at says
    nothing, because the agreement is asymptotic and every curve here starts at
    the same, not-very-small, ``q_min``.

    The row also carries *how* the point at ``q_min`` was obtained -- a shell
    mean with its multiplicity and directional spread for the liquid, a single
    momentum on a named direction for the crystal -- because "the value at
    q_min" means different things for the two phases and a table that did not
    say which would be read as one kind of number throughout.
    """
    rows = []
    for c in curves:
        i = int(np.argmin(c["q"]))
        q0, s0 = float(c["q"][i]), float(c["S"][i])
        om, wm = float(c["Omega"][i]), float(c["omega_mp"][i])
        red = c["reduction"]
        row = {
            "rs": c["rs"], "phase": c["phase"], "q_min": q0,
            "x_min": float(q_over_sqrt_n(q0)),
            "q_min_over_sqrt_n": float(q_over_sqrt_n(q0)),
            "S_q_min": s0, "Omega_SMA": om, "omega_magnetoplasmon": wm,
            "relative_difference": (om - wm) / wm,
            "reduction": red["reduction"],
            "n_points_in_curve": int(c["q"].size),
            "source": None if c["source"] is None else rel(c["source"]["dir"]),
            "is_a_q_to_zero_limit": False,
            "statement": ("This is a finite-q diagnostic, not a demonstrated "
                          "q->0 limit."),
        }
        if red["reduction"] == "shell-mean":
            # The shell at q_min is the first group of the reduction, and its
            # multiplicity is how many momenta it averaged.
            row.update({
                "shell_multiplicity": int(red["count"][0]),
                "shell_S_min": float(c["raw_S"][
                    np.isclose(c["raw_qn"], q0, rtol=1e-9)].min()),
                "shell_S_max": float(c["raw_S"][
                    np.isclose(c["raw_qn"], q0, rtol=1e-9)].max()),
                "shell_spread_rel": float(
                    red["spread_rel"][0]) if red["spread_rel"].size else 0.0,
                "crystal_direction_deg": None,
            })
        elif red["reduction"] == "directional":
            row.update({
                "shell_multiplicity": 1,
                "shell_S_min": s0,
                "shell_S_max": s0,
                "shell_spread_rel": 0.0,
                "crystal_direction_deg": red["direction_deg"],
            })
        else:
            row.update({
                "shell_multiplicity": 1,
                "shell_S_min": s0,
                "shell_S_max": s0,
                "shell_spread_rel": 0.0,
                "crystal_direction_deg": None,
            })
        rows.append(row)
    return rows


def old_vs_new(curves):
    """What the correction changed, as numbers -- §12's before/after diagnostic.

    The OLD curve is what this file used to draw: every stored ``S(q-vector)``
    divided by hand and plotted against ``|q|``.  For the liquid that mixes six
    to eighteen directions that are supposed to carry one number; for the
    crystal it mixes directions that carry genuinely different numbers.  Those
    are different failures and they are separated here rather than pooled into
    one "the plot changed" remark.

    Nothing is recomputed from snapshots: both sides are the same stored arrays,
    reduced differently.
    """
    out = {}
    for c in curves:
        if c["raw_qn"] is None:
            continue
        raw_qn, raw_S = c["raw_qn"], c["raw_S"]
        old_omega = omega_sma(raw_qn, raw_S)          # per vector, old behaviour
        tag = f"rs{c['rs']:g}"

        # How many distinct directions does the old curve have at each |q|?
        groups = shells_of(raw_qn)
        mixed = sum(1 for idx in groups if idx.size > 1)
        if c["phase"] == "liquid":
            # Compare the corrected number to the mean of the old ones, shell
            # by shell, and also record the ordering gap at each shell.
            red = c["reduction"]
            dev, band = [], []
            for idx, newv in zip(groups, red["right_order_omega"]):
                dev.append((newv - old_omega[idx].mean()) / old_omega[idx].mean())
                band.append((old_omega[idx].max() - old_omega[idx].min())
                            / old_omega[idx].mean())
            dev = np.asarray(dev)
            band = np.asarray(band)
            out[tag] = {
                "phase": c["phase"],
                "rs": c["rs"],
                "n_old_points": int(raw_qn.size),
                "n_new_points": int(c["q"].size),
                "n_shells": int(len(groups)),
                "max_rel_change_Omega": float(np.abs(dev).max()),
                "rms_rel_change_Omega": float(np.sqrt((dev ** 2).mean())),
                "max_rel_order_gap": float(np.abs(
                    red["order_gap_rel"]).max()),
                "rms_rel_order_gap": float(np.sqrt(
                    (red["order_gap_rel"] ** 2).mean())),
                "max_shell_directional_spread_rel": float(
                    red["spread_rel"].max()),
                # The band the OLD curve drew as scatter: the per-vector Omega
                # values that shared a |q| and were plotted at the same x.  The
                # mean shift above is small because the spread is nearly
                # symmetric; this is what the reader actually saw.
                "max_omega_band_rel": float(band.max()),
                "rms_omega_band_rel": float(np.sqrt((band ** 2).mean())),
                "shells_mixing_directions": int(mixed),
                "note": ("the change is the shell mean of S followed by q^2/(2S); "
                         "the old curve averaged q^2/(2S) over the same shells "
                         "by plotting them all.  Order gap = "
                         "<q^2/(2S)> - q^2/(2<S>)."),
            }
        else:
            # Which |q| shells did the old curve mix, and by how much do the
            # directions sharing them differ?
            red = c["reduction"]
            # Which |q| shells did the old curve mix, and by how much do the
            # directions sharing them differ?  The angles are the real content
            # here: they are what shows that the old curve was splicing six
            # distinct propagation directions into one line.
            raw_qvec = c["raw_qvec"]
            ang = np.degrees(np.arctan2(raw_qvec[:, 1], raw_qvec[:, 0]))
            on_axis = np.abs(np.abs(ang - red["direction_deg"]) % 360.0) < 1e-6
            mixed_detail = []
            for qi in c["q"]:
                idx = np.flatnonzero(np.isclose(raw_qn, qi, rtol=1e-9))
                if idx.size <= 1:
                    continue
                mixed_detail.append({
                    "q": float(qi),
                    "n_momenta_sharing_this_q": int(idx.size),
                    "directions_deg": sorted(round(float(a), 3)
                                             for a in ang[idx]),
                    "kept_on_axis": bool(np.any(on_axis[idx])),
                    "n_off_axis_directions": int(np.count_nonzero(~on_axis[idx])),
                    "S_min": float(raw_S[idx].min()),
                    "S_max": float(raw_S[idx].max()),
                    "Omega_min": float(old_omega[idx].min()),
                    "Omega_max": float(old_omega[idx].max()),
                    "omega_spread_rel": float(
                        (old_omega[idx].max() - old_omega[idx].min())
                        / old_omega[idx].mean()),
                })
            out[tag] = {
                "phase": c["phase"],
                "rs": c["rs"],
                "n_old_points": int(raw_qn.size),
                "n_new_points": int(c["q"].size),
                "n_shells": int(len(groups)),
                "shells_mixing_directions": int(mixed),
                "direction_deg": red["direction_deg"],
                "directions_mixed_at_the_new_branch": mixed_detail,
                "max_rel_spread_removed": (max(
                    (d["omega_spread_rel"] for d in mixed_detail), default=0.0)),
                "note": ("the old curve plotted every direction sharing a |q|; "
                         "the corrected curve keeps only the momenta collinear "
                         "with the chosen axis and does not average them."),
            }
    return out


def roton_minimum(curve, q_wc_value):
    """Look for a resolved minimum of ``Omega(q)`` near ``Q_WC``.

    Operates on a REDUCED curve -- one ``Omega`` per ``|q|``, which is what both
    reduction branches now produce.  The grouping below is therefore a no-op on
    the curves this file builds, and is retained because it is what makes the
    function total: hand it a raw disc and it will still not mistake the other
    points *of a shell* for a point's neighbours, which is the bug that once
    reported 75 minima at a neighbour spacing of ``0.0``.

    A minimum is accepted only when its depth clears the larger of

    * the spread *within* the neighbouring shells (measurement scatter), and
    * the imbalance *between* the two neighbour means (a dip shallower than the
      local slope of a monotone trend is not a minimum of anything).

    Both are reported, so the verdict can be argued with rather than trusted.
    Nothing is smoothed, fitted or interpolated: every number returned is either
    a stored sample or a mean of stored samples, and the count of points behind
    each shell is returned too.

    What this canNOT see: whether the minimum is physics.  For a crystal
    ``S(q)`` rises towards a Bragg peak, so ``Omega = q^2/(2S)`` is driven down
    by the structure factor's own divergence, and the height of a finite-size
    Bragg peak grows with ``N``.  A minimum at ``Q_WC`` is therefore partly a
    statement about the simulation cell.  Reported, not interpreted.
    """
    q, om = np.asarray(curve["q"], float), np.asarray(curve["Omega"], float)
    order = np.argsort(q, kind="stable")
    qs, oms = q[order], om[order]
    new = np.ones(qs.size, bool)
    new[1:] = ~np.isclose(qs[1:], qs[:-1], rtol=1e-9, atol=0.0)
    starts = np.flatnonzero(new)
    bounds = list(starts) + [qs.size]

    q_shell = np.array([qs[a] for a in starts])
    mean = np.array([oms[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])])
    spread = np.array([oms[a:b].max() - oms[a:b].min()
                       for a, b in zip(bounds[:-1], bounds[1:])])
    count = np.array([b - a for a, b in zip(bounds[:-1], bounds[1:])])

    base = {"q_wc": float(q_wc_value), "x_wc": float(q_over_sqrt_n(q_wc_value)),
            "n_shells": int(q_shell.size),
            "shell_points": count.tolist(),
            "shell_spread": spread.tolist()}
    if q_shell.size < 3:
        base["found"] = False
        base["resolved"] = False
        base["reason"] = ("roton not resolved at current finite-size/statistical "
                          f"resolution: only {q_shell.size} distinct |q| shells")
        return base

    idx = [i for i in range(1, q_shell.size - 1)
           if mean[i] < mean[i - 1] and mean[i] < mean[i + 1]]
    if not idx:
        base["found"] = False
        base["resolved"] = False
        base["reason"] = ("roton not resolved at current finite-size/statistical "
                          "resolution: no interior minimum of the shell-mean "
                          "Omega(|q|)")
        return base

    best = min(idx, key=lambda i: abs(q_shell[i] - q_wc_value))
    depth = float(min(mean[best - 1], mean[best + 1]) - mean[best])
    scatter = float(max(spread[best - 1], spread[best], spread[best + 1]))
    imbalance = float(abs(mean[best - 1] - mean[best + 1]))
    noise = max(scatter, imbalance)
    # ``found`` and ``resolved`` are deliberately different questions.  A
    # minimum can exist in the shell-mean curve and still be too shallow to
    # believe -- and a caller who read only "there is a minimum" would claim a
    # roton that the very next field denies.  ``resolved`` is the one §10 asks
    # about, and it means both.
    base.update({
        "found": True,
        "resolved": bool(depth > noise),
        "n_minima": len(idx),
        "other_minima_q": [float(q_shell[i]) for i in idx if i != best],
        "other_minima_x": [float(q_over_sqrt_n(q_shell[i]))
                           for i in idx if i != best],
        "q_roton": float(q_shell[best]),
        "x_roton": float(q_over_sqrt_n(q_shell[best])),
        "Omega_roton": float(mean[best]),
        "n_points_at_roton": int(count[best]),
        "spread_at_roton": float(spread[best]),
        "distance_from_q_wc": float(abs(q_shell[best] - q_wc_value)),
        "distance_in_x": float(abs(q_over_sqrt_n(q_shell[best])
                                   - q_over_sqrt_n(q_wc_value))),
        "depth_below_neighbours": depth,
        "neighbour_shell_scatter": scatter,
        "neighbour_imbalance": imbalance,
        "neighbour_spacing": float(min(q_shell[best] - q_shell[best - 1],
                                       q_shell[best + 1] - q_shell[best])),
        "deeper_than_imbalance": bool(depth > noise),
        "finite_size_caveat": ("a finite-N Bragg peak grows with N, so a "
                               "minimum driven by S's divergence at Q_WC is not "
                               "a thermodynamic collective mode; it is reported "
                               "as a roton-like minimum of the SMA dispersion"),
        "reason": ("roton-like minimum in the SMA dispersion"
                   if depth > noise else
                   "roton not resolved at current finite-size/statistical "
                   "resolution: a minimum exists but is shallower than the "
                   "scatter it must clear"),
    })
    return base


# ==========================================================================
# the picture
# ==========================================================================
QUALITY_STAMP = "NOT PUBLICATION QUALITY\nsource budget: quick"


def figure_stamp(budget):
    """The mark a non-production figure carries, or None for the real one.

    One definition, read by the drawing and the metadata both, so the picture
    and the record of it cannot disagree.  Anything that is not ``reproduction``
    is stamped, so an unrecognised budget marks the figure rather than passing
    as a result -- the same rule, and the same reasoning, as
    ``phase_competition.py``, where a check keyed on the budget once drew a
    quick figure that said nothing about being quick.
    """
    return None if str(budget) == "reproduction" else QUALITY_STAMP


def write_figure(fig_dir, curves, budget, show_classical=False):
    """Draw the SMA dispersion.

    Nothing on these axes is decoration, and three things that were here are
    deliberately gone:

    * A dotted vertical guide at ``Q_WC``.  It was a dashed mark on the axes,
      and the axes are meant to carry measurements.
    * An open triangle planted on the roton minimum, with a leader line and a
      text box.  That is an annotation added to a data point rather than a
      measurement of one.
    * The classical dashed families, which are off by default: with several
      couplings on one axis they cross each other and the eye loses the curves
      that carry the data.  ``--classical`` turns them back on.

    The roton is still searched for, printed and written to the metadata, and
    ``Q_WC`` is still in the small-``q`` table.  They are reported as numbers
    rather than drawn on top of the points they describe.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "magnetoplasmon.png")
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    colors = ["#111111", "#c0392b", "#2471a3", "#8e44ad", "#16a085", "#d68910"]

    for c, col in zip(curves, colors):
        red = c["reduction"]["reduction"]
        suffix = {"analytic": "", "shell-mean": " (liquid, rotation averaged)",
                  "directional": " (crystal, along $x$)"}[red]
        lab = f"$r_s = {c['rs']:g}$" + suffix
        if red == "analytic":
            # The one curve that is a formula, and the only one drawn as a
            # continuous line: the filled-LLL dispersion is known at every q,
            # so a line through it states nothing that was not evaluated.
            ax.plot(c["x"], c["Omega"], "-", lw=1.9, color=col, zorder=3,
                    label=rf"$\Omega$ {lab}")
        elif red == "shell-mean":
            # Thirty points on a smooth isotropic curve: the line is a guide
            # between measurements, not a claim between them, and it is thin.
            ax.plot(c["x"], c["Omega"], "-", lw=0.9, color=col, alpha=0.7,
                    zorder=3)
            ax.plot(c["x"], c["Omega"], "o", ms=4.5, color=col, zorder=4,
                    label=rf"$\Omega$ {lab}")
        else:
            # The crystal is EIGHT points on one collinear branch, spaced by a
            # whole supercell reciprocal vector.  A line through them would
            # draw a dispersion between q's that were never measured and would
            # zigzag, so the points stand alone.
            ax.plot(c["x"], c["Omega"], "o", ms=6.5, color=col, zorder=4,
                    label=rf"$\Omega$ {lab}")
        # The classical curve is the comparison, not the result: same colour so
        # the pair reads as one state, thin and faded so it recedes behind it.
        # Optional, because with several couplings on one axis the dashed
        # families cross and the eye loses the curves that carry the data.
        if show_classical:
            ax.plot(c["x"], c["omega_mp"], "--", lw=1.1, color=col, alpha=0.45,
                    zorder=1, label=rf"$\omega_{{\rm mp}}$ $r_s = {c['rs']:g}$")

    ax.set_xlabel(r"$q/\sqrt{n}$")
    ax.set_ylabel(r"$\hbar^2 q^2 / (2m\,S(q))$   $[\hbar\omega_c]$")
    # The title states what is drawn, not what was computed: with the dashed
    # curves off (the default) there is no comparison on the axes to name.
    ax.set_title(("SMA dispersion against the classical magnetoplasmon "
                  "(from stored $S(q)$)") if show_classical else
                 ("SMA dispersion from stored $S(q)$:  rotation-averaged "
                  "liquid, directional crystal"), fontsize=10.5)
    ax.set_xlim(0.0, float(max(c["x"].max() for c in curves)) * 1.02)
    # Sized to the data rather than to a percentile: the crystal's Omega rises
    # steeply past its roton, and clipping that rise would hide the very branch
    # the minimum sits on -- the dip would look like a crossing.
    ax.set_ylim(0.0, float(np.max(np.concatenate(
        [c["Omega"] for c in curves]))) * 1.05)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8.0, ncol=2 if show_classical else 1,
              loc="upper left", framealpha=0.92)

    stamp = None if str(budget) == "reproduction" else QUALITY_STAMP
    if stamp:
        ax.text(0.985, 0.965, stamp, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="crimson", linespacing=1.3,
                bbox=dict(facecolor="white", edgecolor="crimson", alpha=0.9,
                          boxstyle="round,pad=0.25", linewidth=0.6))
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ==========================================================================
# output
# ==========================================================================
def save_npz(results_dir, curves, rows, roton, q_wc_value):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "magnetoplasmon.npz")
    payload = {"rs": np.array([c["rs"] for c in curves]),
               "kappa": np.array([c["kappa"] for c in curves]),
               "phase": np.array([c["phase"] for c in curves]),
               "reduction": np.array([c["reduction"]["reduction"]
                                      for c in curves]),
               "tags": np.array([_tag(c["rs"]) for c in curves])}
    for c in curves:
        t = _tag(c["rs"])
        payload[f"{t}_q"] = c["q"]
        payload[f"{t}_q_over_sqrt_n"] = c["x"]
        payload[f"{t}_S_q"] = c["S"]
        payload[f"{t}_Omega_SMA"] = c["Omega"]
        payload[f"{t}_omega_mp"] = c["omega_mp"]
        red = c["reduction"]
        if red.get("count") is not None:
            payload[f"{t}_shell_multiplicity"] = np.asarray(red["count"])
        if red.get("spread_rel") is not None:
            payload[f"{t}_shell_spread_rel"] = np.asarray(red["spread_rel"])
        if "direction" in red:
            payload[f"{t}_direction"] = np.asarray(red["direction"])
            payload[f"{t}_direction_deg"] = np.asarray([red["direction_deg"]])
        # The stored array the reduction started from, kept so the correction is
        # auditable against the old picture without re-reading structure_slice.
        if c["raw_qn"] is not None:
            payload[f"{t}_raw_qn"] = c["raw_qn"]
            payload[f"{t}_raw_S_q"] = c["raw_S"]
            payload[f"{t}_old_Omega_per_vector"] = omega_sma(c["raw_qn"],
                                                             c["raw_S"])

    # Every diagnostic row is scalars, so the columns are plain float arrays.
    # They were object arrays in the first version, which numpy will not load
    # again without ``allow_pickle=True`` -- a data file whose own reader has to
    # enable arbitrary code execution to open it is not a deliverable.
    def column(key):
        return np.array([r.get(key) if r.get(key) is not None else np.nan
                         for r in rows], dtype=float)

    payload["diag_rs"] = column("rs")
    payload["diag_q_min"] = column("q_min")
    payload["diag_x_min"] = column("x_min")
    payload["diag_S_q_min"] = column("S_q_min")
    payload["diag_Omega_SMA"] = column("Omega_SMA")
    payload["diag_omega_mp"] = column("omega_magnetoplasmon")
    payload["diag_relative_difference"] = column("relative_difference")
    payload["diag_shell_multiplicity"] = column("shell_multiplicity")
    payload["diag_n_points_in_curve"] = column("n_points_in_curve")
    payload["diag_crystal_direction_deg"] = column("crystal_direction_deg")
    payload["q_wc"] = np.array([q_wc_value])
    payload["q_wc_over_sqrt_n"] = np.array([q_over_sqrt_n(q_wc_value)])
    for tag, rr in roton.items():
        if tag.startswith("_"):
            continue
        payload[f"{tag}_roton_resolved"] = np.array([bool(rr["resolved"])])
        for key in ("q_roton", "x_roton", "Omega_roton",
                    "distance_from_q_wc", "distance_in_x", "n_minima",
                    "n_points_at_roton", "spread_at_roton",
                    "depth_below_neighbours", "neighbour_shell_scatter",
                    "neighbour_imbalance", "neighbour_spacing"):
            if key in rr:
                payload[f"{tag}_{key}"] = np.array([rr[key]])
    np.savez_compressed(path, **payload)
    return path


def _tag(rs):
    return "rs" + _slug_rs(rs).replace(".", "p")


def save_metadata(results_dir, written, curves, rows, roton, q_wc_value, index,
                  args, before_after):
    torus = torus_for(int(args.n_electrons))
    dhat = crystal_direction(torus, args.crystal_direction)
    meta = {
        "run_metadata_schema": "figure_construction/magnetoplasmon/2",
        "recipe": "examples/figure_construction/magnetoplasmon.py",
        "wigner_vmc_version": __version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post_processing_only": True,
        "vmc_started": False,
        "sr_started": False,
        "s_q_regenerated": False,
        "nu": NU,
        "units": {
            "energy": "hbar*omega_c = 1",
            "length": "l_B = 1",
            "q": "1/l_B",
            "x_axis": "q/sqrt(n)",
        },
        "formulas": {
            "SMA": "Omega(q) = hbar^2 q^2 / (2 m S(q))",
            "Omega_in_units": "Omega/(hbar omega_c) = q^2/(2 S(q)), q in 1/l_B",
            "classical_magnetoplasmon":
                "omega_mp^2 = omega_c^2 + (2 pi n e^2/m) q",
            "classical_in_units":
                "omega_mp/(hbar omega_c) = sqrt(1 + kappa q); kappa = "
                "e^2/(4 pi eps0 l_B hbar omega_c) per physics/hamiltonian.py",
            "x_axis_conversion":
                "x = q l_B sqrt(2 pi / nu) from n = nu/(2 pi l_B^2)",
        },
        # ==================================================================
        # the reduction.  This block is the correction: what S(q-vector) -> S(q)
        # means for each phase, and where the crystal's axis comes from.
        # ==================================================================
        "reduction": {
            "liquid": ("rotation average: Sbar(q) = <S(q-vector)> over the "
                       "allowed momenta with |q-vector| = q, THEN "
                       "Omega = q^2/(2 Sbar).  Averaged on S, not on 1/S -- the "
                       "other order is a different number wherever a shell has "
                       "directional spread."),
            "crystal": ("one direction, no averaging: S_C(q) = S_C(q dhat) on "
                        "the momenta collinear with the chosen axis.  A Wigner "
                        "crystal is not isotropic, so a shell mean would mix "
                        "distinct modes and destroy the Bragg peak."),
            "paper_fig4_caption": ("\"The data are rotation averaged in the "
                                   "liquid phase and taken along the x axis in "
                                   "the crystal phase.\""),
            "order_of_operations": "S(q-vector) -> S(q) -> q^2/(2 S(q)); never "
                                   "<q^2/(2 S(q-vector))>",
        },
        "crystal_direction": {
            "choice": str(args.crystal_direction),
            "unit_vector_in_torus_frame": dhat.tolist(),
            "angle_deg_in_torus_frame": direction_angle_deg(dhat),
            "torus_A1": torus.A1.tolist(),
            "torus_A2": torus.A2.tolist(),
            "torus_G1": torus.G1.tolist(),
            "torus_G2": torus.G2.tolist(),
            "why": ("the paper's lattice vectors sit at -30/+30 deg "
                    "(A1 = [sqrt(3)/2, -1/2] a in QuantumHallVMC); this "
                    "package's Torus puts A1 along (1,0).  The frames differ by "
                    "a 30 deg rotation, so the paper's x = (1,0) is the "
                    "direction of G1+G2 here -- a reciprocal-lattice (Bragg) "
                    "direction, which is why a cut along it passes through "
                    "Q_WC.  The torus's own first Cartesian axis is 30 deg away "
                    "and its momenta never reach Q_WC."),
            "equivalents": ("the six first-star Bragg directions +/-30, +/-90, "
                            "+/-150 deg are one orbit of the crystal's point "
                            "group, so they are the same cut up to measurement "
                            "noise; G1+G2 is the representative the rotation "
                            "selects."),
            "not_interpolated": ("every point on the cut is a stored momentum "
                                 "exactly on the axis; no momentum is projected "
                                 "onto the axis."),
            "collinear_tolerance": COLLINEAR_TOL,
        },
        "shell_tolerance": SHELL_TOL,
        "crystal_cut_diagnostics": {
            _tag(c["rs"]): {
                "direction_deg": c["reduction"].get("direction_deg"),
                "n_points": c["reduction"].get("n_points"),
                "max_perpendicular_residual":
                    c["reduction"].get("max_perp_residual"),
                "n_source_vectors": c["reduction"].get("n_vectors_in_source"),
                "mirror_pairs_checked":
                    c["reduction"].get("n_mirror_pairs_checked"),
                "mirror_max_abs_gap":
                    c["reduction"].get("mirror_max_abs_gap"),
            } for c in curves if c["reduction"]["reduction"] == "directional"
        },
        "liquid_shell_diagnostics": {
            _tag(c["rs"]): {
                "n_shells": c["reduction"].get("n_shells"),
                "n_vectors_averaged": c["reduction"].get("n_vectors_averaged"),
                "max_within_shell_q_spread":
                    c["reduction"].get("within_shell_q_spread"),
                "max_shell_spread_rel": float(
                    np.max(c["reduction"]["spread_rel"])),
                "order_gap_max_rel": float(np.max(
                    np.abs(c["reduction"]["order_gap_rel"]))),
                "order_gap_rms_rel": float(np.sqrt(np.mean(
                    c["reduction"]["order_gap_rel"] ** 2))),
            } for c in curves if c["reduction"]["reduction"] == "shell-mean"
        },
        "old_vs_new": before_after,
        "rs_c_recorded_not_used": RS_C,
        "q_wc": {"value": q_wc_value, "x": q_over_sqrt_n(q_wc_value),
                 "from": "triangular Wigner crystal at nu = 1: "
                         "a^2 = 4 pi l_B^2/sqrt(3), Q_WC = 4 pi/(sqrt(3) a)"},
        "source_budget": resolve_budget(args.source_budget),
        "source_budget_requested": str(args.source_budget),
        "crystal_nmax": int(args.crystal_nmax),
        "N": int(args.n_electrons),
        "figure_stamp": figure_stamp(resolve_budget(args.source_budget)),
        # The classical curves are always COMPUTED and saved; this records only
        # whether they were drawn, so a reader of the npz is never misled into
        # thinking the comparison was skipped.
        "classical_curves_drawn": bool(args.classical),
        "crystal_drawn_as": "markers only -- eight collinear points on a "
                            "single branch, spaced by a whole supercell "
                            "reciprocal vector; a line between them would "
                            "assert a dispersion at q's that were not measured",
        "sources": [{
            "rs": c["rs"], "phase": c["phase"],
            "reduction": c["reduction"]["reduction"],
            "directory": (None if c["source"] is None
                          else rel(c["source"]["dir"])),
            "structure_factor_npz": (None if c["source"] is None
                                     else rel(c["source"]["npz"])),
            "structure_slice_rs": None if c["source"] is None else c["source"]["rs"],
            "structure_slice_nmax": None if c["source"] is None else c["source"]["nmax"],
            "structure_slice_budget": None if c["source"] is None else c["source"]["budget"],
            "written": (None if c["source"] is None else
                        time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(c["source"]["mtime"]))),
            "note": c["note"],
        } for c in curves],
        "small_q_diagnostic": rows,
        "small_q_statement": ("Every q_min here is the finite torus's smallest "
                              "allowed momentum, not q -> 0.  This is a "
                              "finite-q diagnostic, not a demonstrated q->0 "
                              "limit, and no Kohn-theorem convergence is claimed "
                              "from it."),
        "roton": roton,
        "s_q_definition": ("read unchanged from structure_slice's "
                           "structure_factor.npz; the reduction selects and "
                           "means stored samples and nothing else -- no second "
                           "estimator, no smoothing, no interpolation"),
        "available_sources": [{
            "phase": r["phase"], "rs": r["rs"], "nmax": r["nmax"],
            "budget": r["budget"], "post_projection_fix": r["post_fix"],
            "directory": rel(r["dir"]),
        } for r in index],
        "figures": [rel(p) for p in written],
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path


# ==========================================================================
# the command line
# ==========================================================================
def resolve_rs(args, parser):
    """Merge ``--rs`` and the phase-named ``--liquid-rs``/``--crystal-rs``.

    The two spellings say the same thing -- which couplings to draw -- so they
    are alternatives rather than additions, and giving both is refused instead
    of silently concatenated.

    The phase-named flags only group the couplings for reading and for the
    order they are drawn in.  They do NOT decide the phase: that is
    ``phase_of(r_s)`` and nothing else, so ``--liquid-rs 100`` draws the
    crystal at 100, because 100 is above ``r_s^c``.  Refusing it instead was
    tried and removed -- the boundary is a property of the coupling, not of the
    flag someone typed it under, and a hard gate there turns a spelling choice
    into an error without protecting anything.  Which phase each coupling got
    is printed at startup, so the flag and the result cannot silently disagree.
    """
    named = (args.liquid_rs is not None or args.crystal_rs is not None)
    if args.rs is not None and named:
        parser.error("--rs and --liquid-rs/--crystal-rs are two ways to say "
                     "the same thing; give the couplings one way or the other")
    if args.rs is not None:
        return [float(r) for r in args.rs]
    if not named:
        return list(DEFAULT_RS)
    out = []
    for attr in ("liquid_rs", "crystal_rs"):
        out.extend(float(r) for r in (getattr(args, attr) or ()))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Post-process stored structure_slice S(q) into the "
                    "magnetoplasmon comparison.  Never runs VMC.")
    p.add_argument("--rs", nargs="+", type=float, default=None, metavar="RS",
                   help="couplings to draw.  Default %s, the paper's points; "
                        "r_s = 0 is analytic and needs no saved data.  "
                        "Equivalent to giving the couplings through "
                        "--liquid-rs/--crystal-rs, which name the phase each "
                        "one comes from."
                        % (list(DEFAULT_RS),))
    p.add_argument("--liquid-rs", nargs="+", type=float, default=None,
                   metavar="RS",
                   help="couplings to draw first, grouped for reading.  The "
                        f"phase is still taken from r_s against r_s^c = {RS_C:g}"
                        ", so a value above the boundary is drawn as the "
                        "crystal whatever flag it arrived under.")
    p.add_argument("--crystal-rs", nargs="+", type=float, default=None,
                   metavar="RS",
                   help="couplings to draw after the --liquid-rs ones, grouped "
                        "for reading.  Phase as above: from r_s, not from the "
                        "flag.")
    p.add_argument("--source-budget", "--budget", dest="source_budget",
                   default=DEFAULT_BUDGET,
                   help="which stored budget to read, quick|reproduction. "
                        f"Default {DEFAULT_BUDGET}.")
    p.add_argument("--crystal-nmax", type=int, default=DEFAULT_NMAX,
                   help=f"crystal truncation to read.  Default {DEFAULT_NMAX}.")
    p.add_argument("--crystal-direction", choices=list(CRYSTAL_DIRECTIONS),
                   default=DEFAULT_CRYSTAL_DIRECTION,
                   help="which axis the crystal's S(q) is cut along.  'bragg' "
                        "(default) is the paper's x axis, a reciprocal-lattice "
                        "direction of the crystal, and the only choice whose "
                        "momenta reach Q_WC.  'clean-x' is this package's own "
                        "first Cartesian axis, 30 degrees away, offered so the "
                        "mapping claim can be checked rather than believed.")
    p.add_argument("--n-electrons", type=int, default=DEFAULT_N, metavar="N",
                   help=f"the supercell the stored S(q) was measured on; used "
                        f"to rebuild the torus the direction convention is "
                        f"expressed in.  Default {DEFAULT_N}.")
    p.add_argument("--classical", dest="classical", action="store_true",
                   default=False,
                   help="ALSO draw the dashed classical magnetoplasmon curves.  "
                        "Off by default: the figure is meant to carry the SMA "
                        "measurement, and the dashed families cross each other "
                        "and the eye loses the curves that carry the data.  The "
                        "comparison is computed and saved in the npz and the "
                        "metadata either way -- this only puts it on the axes.")
    p.add_argument("--no-classical", dest="classical", action="store_false",
                   help="accepted and redundant; not drawing the dashed curves "
                        "is already the default.")
    p.add_argument("--q-min-analytic", type=float, default=1e-3,
                   help="lowest q of the analytic r_s = 0 curve, in 1/l_B.")
    p.add_argument("--q-max-analytic", type=float, default=4.0,
                   help="highest q of the analytic r_s = 0 curve, in 1/l_B.")
    p.add_argument("--slug", default=None,
                   help="output directory name.  Default: 'paper' for the "
                        "paper's own couplings, otherwise a name listing them.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    args.rs = resolve_rs(args, p)
    return args


def phase_of(rs):
    """Which phase a requested coupling is drawn from.

    Fixed by the figure, not inferred: the paper's competition puts the liquid
    at small ``r_s`` and the crystal above ``r_s^c``, and this file does not
    recompute that boundary.  Below ``RS_C`` the liquid is the state; at and
    above it, the crystal.
    """
    return "liquid" if float(rs) < RS_C else "crystal"


def output_slug(rs_list):
    """``paper`` for the paper's own points, else what was actually asked for."""
    if tuple(float(r) for r in rs_list) == tuple(DEFAULT_RS):
        return "paper"
    return "rs" + "__".join(_slug_rs(r).replace(".", "p") for r in rs_list)


def main(argv=None):
    args = parse_args(argv)
    args.phase_of = phase_of
    t0 = time.time()

    slug = args.slug or output_slug(args.rs)
    res_dir = os.path.join(RESULTS, slug)
    fig_dir = os.path.join(FIGDIR, slug)

    torus = torus_for(int(args.n_electrons))
    dhat = crystal_direction(torus, args.crystal_direction)

    print("magnetoplasmon -- post-processing only, no VMC   "
          f"(wigner_vmc {__version__})")
    print("=" * 78)
    # The phase next to each coupling, because the flags no longer decide it.
    # A coupling typed under the wrong flag is drawn by its own r_s, and this
    # line is where that shows up rather than in a figure nobody re-reads.
    print("  r_s requested    "
          + ", ".join(f"{float(r):g} ({phase_of(r)})" for r in args.rs))
    _bud = resolve_budget(args.source_budget)
    print(f"  source budget    {_bud}" + ("" if _bud == args.source_budget
                                      else f"  (from --source-budget/--budget "
                                           f"{args.source_budget})"))
    print(f"  crystal nmax     {args.crystal_nmax}")
    print(f"  x axis           q/sqrt(n),  nu = {NU}")
    print("  reduction        liquid: shell mean of S, then q^2/(2S)")
    print(f"                   crystal: one direction, "
          f"{args.crystal_direction} = {direction_angle_deg(dhat):+.1f} deg "
          f"in the torus frame")
    print(f"  results          {rel(res_dir)}")
    print(f"  figures          {rel(fig_dir)}")
    print("=" * 78)

    try:
        curves, index = build_curves(args)
    except LookupError as exc:
        # The documented failure path: say what is missing and what would make
        # it, then stop.  Launching structure_slice.py here would turn a
        # post-processing script into a multi-hour VMC campaign the user did not
        # ask for, which is the one thing this file promises never to do.
        print()
        print("magnetoplasmon: missing S(q) -- this script will not compute it.")
        print()
        print(f"  {exc}")
        print()
        print("  Nothing was run and nothing was written.")
        return 2

    rows = small_q_table(curves)
    qw = q_wc()
    roton = {}
    for c in curves:
        if c["phase"] == "crystal":
            roton[_tag(c["rs"])] = roton_minimum(c, qw)
            roton[_tag(c["rs"])]["rs"] = c["rs"]
            roton[_tag(c["rs"])]["reduction"] = c["reduction"]["reduction"]
    if not roton:
        roton["_none"] = {"resolved": False,
                          "reason": "no crystal curve was requested, so no roton "
                                    "search was possible",
                          "q_wc": qw}

    before_after = old_vs_new(curves)

    print()
    print("  small-q convergence diagnostic (Omega vs the classical "
          "magnetoplasmon)")
    print(f"  {'r_s':>7s} {'phase':>9s} {'q_min':>8s} {'S(q_min)':>9s} "
          f"{'Omega_SMA':>10s} {'w_mp':>9s} {'rel.diff':>9s} "
          f"{'n':>4s} {'reduction':>18s}")
    for r in rows:
        print(f"  {r['rs']:7g} {r['phase']:>9s} {r['q_min']:8.4f} "
              f"{r['S_q_min']:9.5f} {r['Omega_SMA']:10.4f} "
              f"{r['omega_magnetoplasmon']:9.4f} {r['relative_difference']:+8.2%} "
              f"{r['n_points_in_curve']:4d} {r['reduction']:>18s}")
    print()
    print("  This is a finite-q diagnostic, not a demonstrated q->0 limit.  q_min")
    print("  is the finite torus's smallest allowed momentum, set by the")
    print("  supercell -- not a chosen small q.  No Kohn-theorem convergence is")
    print("  claimed from it.")
    print()
    for r in rows:
        if r["reduction"] == "shell-mean":
            print(f"    r_s = {r['rs']:g}: q_min averaged over "
                  f"{r['shell_multiplicity']} momenta, spread "
                  f"{r['shell_spread_rel']:+.2%} of the mean")
        elif r["reduction"] == "directional":
            print(f"    r_s = {r['rs']:g}: q_min is one stored momentum on the "
                  f"{r['crystal_direction_deg']:+.1f} deg axis")

    print()
    print("  old vs new -- what the correction changed")
    for tag, rec in before_after.items():
        if rec["phase"] == "liquid":
            print(f"    r_s = {rec['rs']:g} liquid: {rec['n_old_points']} stored "
                  f"momenta -> {rec['n_new_points']} shells")
            print(f"        Omega change (shell mean vs per-vector): "
                  f"max {rec['max_rel_change_Omega']:+.2%}, "
                  f"rms {rec['rms_rel_change_Omega']:.2%}")
            print(f"        order-of-operations gap <q^2/2S> - q^2/(2<S>): "
                  f"max {rec['max_rel_order_gap']:+.2%}, "
                  f"rms {rec['rms_rel_order_gap']:.2%}")
            print(f"        largest directional spread inside one shell: "
                  f"{rec['max_shell_directional_spread_rel']:.2%}")
            print(f"        Omega band the old scatter drew at one x: "
                  f"max {rec['max_omega_band_rel']:.0%}, "
                  f"rms {rec['rms_omega_band_rel']:.0%}")
        else:
            print(f"    r_s = {rec['rs']:g} crystal: {rec['n_old_points']} stored "
                  f"momenta across {rec['n_shells']} shells -> "
                  f"{rec['n_new_points']} collinear points on the "
                  f"{rec['direction_deg']:+.1f} deg axis")
            print(f"        shells that mixed directions in the old curve: "
                  f"{rec['shells_mixing_directions']}")
            print(f"        largest Omega spread among momenta sharing a |q| "
                  f"that the old curve plotted together: "
                  f"{rec['max_rel_spread_removed']:.2%}")

    print()
    print(f"  Q_WC (triangular, nu = {NU:g}) = {qw:.6f} 1/l_B "
          f"= {q_over_sqrt_n(qw):.4f} in q/sqrt(n)")
    for tag, rr in roton.items():
        print(f"  roton {tag}: {rr.get('reason', '')}")
        if rr.get("resolved"):
            print(f"          q = {rr['q_roton']:.4f}  "
                  f"x = {rr['x_roton']:.4f}  Omega = {rr['Omega_roton']:.4f}  "
                  f"|q - Q_WC| = {rr['distance_from_q_wc']:.4f}")
            print(f"          finite-size caveat: {rr['finite_size_caveat']}")

    written = [write_figure(fig_dir, curves,
                            resolve_budget(args.source_budget),
                            show_classical=args.classical)]
    npz = save_npz(res_dir, curves, rows, roton, qw)
    meta = save_metadata(res_dir, written, curves, rows, roton, qw, index, args,
                         before_after)

    print()
    print("=" * 78)
    for path in written:
        print(f"  figure   {rel(path)}")
    print(f"  data     {rel(npz)}")
    print(f"  metadata {rel(meta)}")
    print(f"  total {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
