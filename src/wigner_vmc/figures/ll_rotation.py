"""The LL-rotation figures that can be redrawn from numbers, and only those.

Scope, and why it is exactly four
---------------------------------
The frozen LL-rotation set holds eight official PNGs.  Four of them are drawn here:

    vmc_energy_phase_competition.png
    vmc_energy_crossing_linear.png
    vmc_structure_factor_sma.png
    vmc_real_space_density.png

and four are NOT, for one reason: their plotted arrays were never stored anywhere.  The
three phase maps -- `structure_factor_phase_map`, `correlation_function_phase_map`,
`shell_averaged_companions` -- and `energy_order_summary` were reduced inside notebook
cells from snapshots that no longer have a reachable cache, and no intermediate array
was written to disk.  They are marked `legacy-only` in FIGURE_MANIFEST.md and their
original PNGs are kept as they are.  Rebuilding them would mean re-running VMC, which
the Stage 1 rule forbids for figure code outright -- so the honest outcome is a
manifest entry, not a plausible-looking redraw with different numbers behind it.

Every function here takes prepared arrays.  Nothing loads a file and nothing can
sample; `tests/test_figure_contract.py` enforces that by parsing this module rather than
importing it.

The three conventions worth stating once
----------------------------------------
1. `dpi=170` and NO `bbox_inches`, which is what the legacy cells used.  The Gaussian
   workflow's figures used `dpi=110` with `bbox_inches="tight"`; the two workflows
   genuinely differ here and copying one's render settings onto the other would change
   the output size for no reason a reader asked for.
2. The delta_E panels DODGE the three truncations by +-0.75 r_s units.  At r_s = 75,
   77.5 and 80 the tiers share a coupling and span 0.0116 in delta_E -- less than a
   marker's height -- so drawn at their exact x they overprint and the reader sees one
   point where the figure has three.  The offset moves the DRAWING only: no value is
   interpolated and the tick labels still name the real r_s.  It is recorded in the
   axis label so a reader is never left to infer it.
3. The delta_E panels are on a LINEAR axis limited to 44-91, decoupled from the log
   axis above them.  Every point that carries the crossing lies in 45-90; on the upper
   panel's 2.4-150 log range that is the last 18 % of the width, which would cram the
   entire sign change into the right margin.  Two questions, two axes.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# the shared style tables.  Module level, not per-function, because the two delta_E
# figures must agree about what a marker means: a rung read off one has to be the same
# rung in the other.
# ---------------------------------------------------------------------------
TIER_STYLE = {
    "conv":  dict(marker="s", color="#c0392b",
                  label=r"converged $n_{\rm max}=1$ ($n_b=2$)"),
    "nest":  dict(marker="D", color="#8e44ad",
                  label=r"nested $n_{\rm max}=2$ ($n_b=3$)"),
    "nest4": dict(marker="o", color="#16a085",
                  label=r"nested $n_{\rm max}=3$ ($n_b=4$)"),
}
TIER_TITLE = r"LL-rotated scan"

# 0.75, not 1.20.  The couplings the three tiers share are 2.5 apart (75 / 77.5 / 80),
# so an offset of 1.2 would put the converged point of r_s = 77.5 at 76.3 and the nested
# n_max = 3 point of r_s = 75 at 76.2 -- two different couplings drawn as a vertical
# pair, which reads as one coupling with two states and is simply wrong.  At 0.75 the
# closest approach between points of DIFFERENT couplings stays above 1.0.
TIER_DODGE = {"conv": -0.75, "nest": 0.00, "nest4": +0.75}

# The window the delta_E panels carry.  Not 42, which was tried first and drew two
# artefacts rather than two points: the scan also holds converged points at 40 and 42.5,
# and at a limit of 42 the 42.5 one (dodged to 41.75) fell half outside, leaving a bar
# and no marker at the axis edge.  The panel's subject is the crossing and every coupling
# that carries it is at 45 or above.
DE_XLO, DE_XHI = 44.0, 91.0


# Where a non-production stamp goes on THIS figure, one entry per panel, as
# (x, y, ha, va) in axes fraction.  Deliberately not the upper-right corner the
# other figure modules use, and deliberately not the same corner twice: the
# ladder's upper right is occupied by its legend, and the delta_E panel's data
# dives to the lower right with the tier key already at the lower left.  So the
# ladder is stamped at the lower left, which is empty because the curve leaves
# it immediately, and the delta_E panel at the upper right.  A single corner for
# both would put the mark under a legend in one panel and under the data in the
# other -- which is how a stamped figure comes to read as an unstamped one.
# The coordinates are inset from the spines by more than the box's own padding,
# so the mark sits inside the frame rather than straddling it.
_STAMP_CORNERS = ((0.018, 0.045, "left", "bottom"),
                  (0.972, 0.958, "right", "top"))


def _stamp(fig, text):
    """Mark a non-production figure, INSIDE the axes, once per panel.

    ``text`` is None on the publication path, and then nothing at all is drawn --
    not an empty box, not an invisible artist -- so the official render is
    byte-identical to what it was before this parameter existed.

    Every axis is stamped, cycling the corner table if a panel is ever added: a
    stamp that covered one panel of a two-panel figure would be worse than none,
    because the figure would then say "quick" in one half and look like a result
    in the other.
    """
    if not text:
        return
    for i, ax in enumerate(fig.axes):
        x, y, ha, va = _STAMP_CORNERS[i % len(_STAMP_CORNERS)]
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.5,
                color="crimson", linespacing=1.3,
                bbox=dict(facecolor="white", edgecolor="crimson", alpha=0.85,
                          boxstyle="round,pad=0.25", linewidth=0.6))


def _finish(fig, path, dpi):
    if path:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return fig


def _de_window(window):
    """The delta_E window as ``(xlo, xhi)``; ``None`` is the historical one."""
    return (DE_XLO, DE_XHI) if window is None else (float(window[0]),
                                                    float(window[1]))


def _de_ticks(xlo, xhi):
    """Major ticks on the multiples of 5 inside the window, minor on every unit.

    Derived rather than written down, so widening the window cannot leave the
    axis labelled at the couplings of a scan it no longer shows.  At the
    historical window -- 44 to 91 -- this returns exactly the literals it
    replaced: majors 45..90 step 5, minors 43..93.
    """
    major = np.arange(math.ceil(xlo / 5.0) * 5.0, xhi + 1e-9, 5.0)
    minor = np.arange(math.floor(xlo) - 1.0, math.ceil(xhi) + 2.0 + 1e-9, 1.0)
    return major, minor


def _kept_tiers(tiers: dict, window=None) -> dict:
    """Drop the tiers the window excludes, HERE, once, rather than clipping in the axes.

    Two reasons and the second is the one that actually hurt.  `errorbar` is clipped by
    the axes but `annotate` is NOT, so a z label whose point lies outside xlim is still
    drawn -- in the margin between the axes and the tick labels, reading as a number
    with no mark beside it.  And the y-range is set from these arrays, so an out-of-
    window point stretches the axis for a mark nobody can see.

    A point is kept iff its DODGED x is inside the window, which is the position it is
    drawn at; doing it here rather than at the draw sites is what keeps the filter and
    the panels from disagreeing about where a point is.

    ``window`` is ``(xlo, xhi)`` and defaults to the historical one.  It is the CALLER's
    scan that decides how wide the panel has to be, so the caller passes it rather than
    the module keeping a second, silently stale copy of somebody else's grid.
    """
    xlo, xhi = _de_window(window)
    out = {}
    for t, (x, y, e) in tiers.items():
        if t not in TIER_DODGE:
            continue
        m = ((x + TIER_DODGE[t] >= xlo - 1e-9)
             & (x + TIER_DODGE[t] <= xhi + 1e-9))
        if m.any():
            out[t] = (np.asarray(x, float)[m], np.asarray(y, float)[m],
                      np.asarray(e, float)[m])
    return out


# ==========================================================================
# Figure 1a -- the energy competition
# ==========================================================================
def energy_phase_competition(rs, e_liq, s_liq, e_cry, s_cry, tiers, ll_rot=None,
                             path: Optional[str] = None, dpi: int = 170,
                             stamp: Optional[str] = None,
                             window: Optional[Tuple[float, float]] = None):
    """Two panels: the absolute E/N ladder, and delta_E on its own linear axis.

    The two absolute curves are told apart by their MARK and not by their colour -- the
    liquid is bare dots with no joining line, the crystal a dashed line with no markers,
    both black.  Colour was doing no separating work in any case: the two sit within a
    couple of pixels of each other at every coupling (delta_E never exceeds 0.16 against
    a 58-unit axis), and a figure that still reads in greyscale is worth more than one
    that depends on a hue difference nobody can see.

    `tiers` maps a tier name to (rs, delta_E, delta_E_err).  `ll_rot` is an optional
    (rs, E/N) pair for the LL-rotated comparison series, drawn as open triangles; it is
    a DIFFERENT variational family at DIFFERENT couplings, so it is a comparison and not
    more of the same curve.

    `stamp` is drawn inside both panels, and its DEFAULT IS NONE: the publication
    render is untouched by it.  It is the caller's mark -- typically a recipe's
    "this is a smoke test" banner -- passed in rather than composed here, because
    what a run's quality is called is a property of the run and not of the figure.

    `window` is the delta_E panel's ``(xlo, xhi)`` in r_s, defaulting to the
    historical 44-91.  A scan that starts at r_s = 25 and ends at 90 needs a
    wider one; the ticks follow it, so a panel can never be labelled at the
    couplings of a scan it is not showing.  The upper panel is unaffected --
    it is on a log axis over the ladder's own couplings.
    """
    rs = np.asarray(rs, float)
    xlo, xhi = _de_window(window)
    tiers = _kept_tiers(tiers, window)

    fig = plt.figure(figsize=(7.6, 7.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.20, 1.0], hspace=0.32)

    # ---- the absolute energies -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(rs, e_liq, yerr=s_liq, marker="o", ms=5.0, ls="none", lw=1.2,
                elinewidth=1.2, color="k", capsize=2.5,
                label=r"liquid  $E_{\rm liq}/N$")
    # ls="none" kills the joining line but not the error bars: errorbar draws those with
    # elinewidth, set explicitly here so it does not follow lw.
    ax.errorbar(rs, e_cry, yerr=s_cry, marker="", ls="--", lw=1.7, elinewidth=1.2,
                color="k", capsize=2.5, label=r"crystal  $E_{\rm cry}/N$")
    if ll_rot is not None and len(ll_rot[0]):
        ax.plot(ll_rot[0], ll_rot[1], "^:", ms=5, lw=1.3, mfc="none", color="k",
                alpha=0.6, label=r"LL-rotated crystal")
    ax.set_xscale("log")
    ax.set_ylabel(r"$E/N$   $[\hbar\omega_c]$")
    ax.set_xlabel(r"$r_s$", fontsize=9)
    # Upper right: the curve descends left to right, so that is the one region no data
    # reaches.
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.25, which="both", lw=0.5)

    # ---- delta_E on its own linear axis ----------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axhline(0, color="k", lw=0.9)
    for t, st in TIER_STYLE.items():
        if t not in tiers:
            continue
        x, y, e = tiers[t]
        # Labelled only where the series is present: plot() with an empty series still
        # registers a legend entry, which would advertise a rung the data does not carry.
        ax2.errorbar(x + TIER_DODGE[t], y, yerr=e, marker=st["marker"], ms=6.5, lw=1.4,
                     ls="none", mfc="none", mec=st["color"], ecolor=st["color"],
                     color=st["color"], capsize=2.5, label=st["label"])
    major, minor = _de_ticks(xlo, xhi)
    ax2.set_xlim(xlo, xhi)
    ax2.set_xticks(major)
    ax2.set_xticks(minor, minor=True)
    ax2.set_xlabel(r"$r_s = \kappa\sqrt{2}$      "
                   r"(series at one $r_s$ are dodged $\pm0.75$ for legibility)")
    ax2.set_ylabel(r"$\delta E/N$   $[\hbar\omega_c]$")
    # The key is titled with the SERIES and the three entries are then rungs.  Stated
    # once rather than three times: it is a property of the whole series, and the
    # three-line untitled version was the tallest object in a half-height panel.
    ax2.legend(loc="lower left", fontsize=8.5, frameon=False,
               title=TIER_TITLE, title_fontsize=8.5)
    ax2.set_axisbelow(True)
    ax2.grid(alpha=0.25, lw=0.5)
    # No ylim: the range comes from the data.  A hard-coded one would be a second place
    # for the scan's range to go stale, and it would go stale silently.

    # subplots_adjust, not tight_layout: tight_layout does not understand a gridspec
    # figure and warns about it.  The gridspec already fixed hspace, so only the outer
    # margins are set here.
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.095, top=0.975)
    _stamp(fig, stamp)
    return _finish(fig, path, dpi)


# ==========================================================================
# Figure 1b -- the same crossing on a page of its own
# ==========================================================================
def energy_crossing_linear(tiers, path: Optional[str] = None, dpi: int = 170):
    """The same 45-90 window, with |z| printed beside every point.

    Panel (a) above carries this series in its place in the argument, and what that
    placement costs is size: half a figure, the three truncations dodged into a tenth of
    its width, no room for a per-point number.  This page spends a whole page on the
    same window for the reader who wants to check the crossing itself rather than the
    argument around it -- the sign of each point, its size against its own bar, and the
    |z| = |delta_E|/sigma that decides whether a point is a measurement or a fluctuation.

    Nothing is recomputed and no new measurement is made: it is the same tier table
    Figure (a) draws, from one binding, so the two cannot disagree about a value.
    """
    tiers = _kept_tiers(tiers)
    if not tiers:
        raise ValueError("no tier has a point inside the delta_E window; refusing to "
                         "write an empty page")

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.axhline(0.0, color="k", lw=1.0)

    labels = []
    for t, st in TIER_STYLE.items():
        if t not in tiers:
            continue
        x, y, e = tiers[t]
        ax.errorbar(x + TIER_DODGE[t], y, yerr=e, marker=st["marker"], ms=8.0, lw=1.4,
                    ls="none", mfc="none", mec=st["color"], ecolor=st["color"],
                    color=st["color"], capsize=3.2, label=st["label"], zorder=4)
        # ALWAYS above the top of the bar, for a negative point as much as a positive
        # one.  The obvious rule -- above for positive, below for negative, so the label
        # sits on the far side of the bar -- collects all eight labels of the 75-80
        # cluster into a 0.05-tall band and runs the lowest of them off the floor of the
        # panel.  Above the bar top each label lands wherever its own bar ends, which
        # keeps them inside the panel and in the data's own order.
        for xx, yy, ee in zip(x + TIER_DODGE[t], y, e):
            labels.append((float(xx), float(yy + ee),
                           f"{abs(yy) / ee:.2f}" if ee > 0 else "inf", st["color"]))

    ax.set_xlim(DE_XLO, DE_XHI)
    # set_xticks, not MultipleLocator: a locator that is not imported is an ImportError
    # in a delivered notebook rather than a formatting choice.
    ax.set_xticks(np.arange(45.0, 90.1, 5.0))
    ax.set_xticks(np.arange(43.0, 93.1, 1.0), minor=True)
    ax.set_xlabel(r"$r_s = \kappa\sqrt{2}$")
    ax.set_ylabel(r"$\delta E/N = (E_{\rm cry}-E_{\rm liq})/N$   $[\hbar\omega_c]$")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, lw=0.5)          # major only: the minor fence is a grey wash here
    ax.legend(loc="upper right", fontsize=9, frameon=False,
              title=TIER_TITLE, title_fontsize=9)
    # The two reading notes go in the one region that is empty by construction: below
    # y = 0 and left of r_s = 70 no point, no bar and no z label is ever drawn, because
    # every coupling under 70 is positive.
    ax.text(0.015, 0.025,
            "the number beside a point is $|z| = |\\delta E|/\\sigma_{\\rm tot}$\n"
            r"series at one $r_s$ are dodged $\pm0.75$",
            transform=ax.transAxes, fontsize=8.0, color="0.35",
            va="bottom", ha="left", linespacing=1.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # The y-range is DERIVED from the series and its bars, not hard-coded -- a literal
    # here would silently clip the very points the page exists to show.  The pad covers
    # the label offsets, which autoscale cannot: matplotlib measures markers and bars but
    # not annotations.
    ylo = min(float(np.min(y - e)) for _, y, e in tiers.values())
    yhi = max(float(np.max(y + e)) for _, y, e in tiers.values())
    ypad = 0.12 * (yhi - ylo)
    ax.set_ylim(ylo - ypad, yhi + ypad)
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.115, top=0.955)

    # ---- the z labels, placed by MEASUREMENT ------------------------------
    # Two earlier attempts placed these by rule and both failed the same way.  A fixed
    # data-unit offset is tuned to the y-range rather than to the text: at r_s = 80 the
    # converged "1.71" and the nested "1.41" overprinted by 1.4 px -- still one
    # unreadable string, in the exact region the page exists to make legible.  A per-tier
    # band of 9 pt fixes r_s = 80 and breaks r_s = 75, where the converged bar top sits
    # 58 px above the nested one, so the bands swapped the pair that collided.
    #
    # No rule can work: the label height is fixed in POINTS while the gap it must clear
    # is in DATA units and varies by an order of magnitude across the page.  So the
    # offsets are resolved greedily instead -- labels in x order, each raised in 6 pt
    # steps until its rendered box clears every box already placed.  It is deterministic,
    # it depends on the data only through the numbers being drawn, and it cannot silently
    # regress when a record lands at a new coupling and rescales the axis.
    #
    # The fallback is not decoration.  `canvas.get_renderer` is Agg's, and a caller may
    # hold a non-Agg canvas; if it cannot measure, the labels go back to a plain 4 pt
    # offset.  A figure with two labels touching is a worse page than one with a
    # wrong-but-legible offset, and a figure that raises is no page at all.
    labels.sort(key=lambda r: r[0])
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:                                                # noqa: BLE001
        renderer = None
    placed = []
    for lx, lytop, lz, lcol in labels:
        txt = None
        for dypt in (list(np.arange(4.0, 78.0, 6.0)) + [78.0]):
            txt = ax.annotate(lz, (lx, lytop), textcoords="offset points",
                              xytext=(0.0, float(dypt)), ha="center", va="bottom",
                              fontsize=7.5, color=lcol, zorder=5)
            if renderer is None:
                break
            # No redraw per trial: get_window_extent runs update_positions, which
            # resolves the offset transform against the renderer already held, so the box
            # is available immediately.  A draw() here would be ~200 full canvases.
            bb = txt.get_window_extent(renderer=renderer)
            if all(bb.x1 <= o.x0 or bb.x0 >= o.x1 or bb.y1 <= o.y0 or bb.y0 >= o.y1
                   for o in placed):
                break
            txt.remove()
        if renderer is not None and txt is not None:
            placed.append(txt.get_window_extent(renderer=renderer))

    return _finish(fig, path, dpi)


# ==========================================================================
# Figure 2 -- S(q) and the SMA
# ==========================================================================
def structure_factor_sma(qn, qg, sqa, lll, i_g1, ladder, bragg_over_sqrt_n,
                         path: Optional[str] = None, dpi: int = 170):
    """S(q) on the allowed torus momenta, and the SMA mode it implies.

    `sqa[(kind, kappa)]` is a (n_shell, 4) array of [|q|, S, sigma_S, multiplicity];
    `lll` is the same shape for the filled-LLL reference.  The filled LLL is the reason
    this figure is a calibration and not just a plot: its S(q) is known exactly, so a
    normalisation error anywhere upstream shows up as a constant offset here.

    The lower panel is the same data through omega = q^2/2S, and the point of putting
    them together is that they must move in OPPOSITE directions: a peak in S(q) has to
    appear as a dip here.  The crystal's soft mode at the Bragg shell is that dip.

    The classical magnetoplasmon sqrt(1 + kappa q) is drawn on the small-q shells only.
    It is an analytic small-q form, not a fit, and it is drawn where it is valid rather
    than where it would look best.
    """
    qn = np.asarray(qn, float)
    qg = np.asarray(qg, float)
    ladder = list(ladder)
    colsr = {k: plt.get_cmap("viridis")(0.06 + 0.86 * i / max(len(ladder) - 1, 1))
             for i, k in enumerate(ladder)}

    fig = plt.figure(figsize=(11.4, 9.4))
    gs = fig.add_gridspec(2, 1, hspace=0.13)

    axs = fig.add_subplot(gs[0])
    axs.plot(qn, 1 - np.exp(-qg ** 2 / 2), "-", color="0.35", lw=2.6, alpha=0.85,
             zorder=2, label=r"filled LLL:  $1-e^{-q^2/2}$")
    axs.plot(qn, lll[:, 1], "o", ms=3.4, color="0.35", alpha=0.75, zorder=3,
             label="filled LLL, VMC snapshots")
    # Only the liquid curves are labelled: the crystal is the same colour and dashed, and
    # twelve legend rows would cost more clarity than twelve labels buy.
    for k in ladder:
        axs.errorbar(qn, sqa[("liquid", k)][:, 1], yerr=sqa[("liquid", k)][:, 2],
                     color=colsr[k], lw=1.5, marker="o", ms=3.2, capsize=1.6,
                     label=rf"liquid  $r_s={k * math.sqrt(2):.0f}$")
        axs.errorbar(qn, sqa[("crystal", k)][:, 1], yerr=sqa[("crystal", k)][:, 2],
                     color=colsr[k], lw=1.3, ls="--", marker="s", ms=3.0, capsize=1.6,
                     alpha=0.9)
    axs.plot([], [], "--", color="0.55", lw=1.5, label="crystal (same colours)")
    axs.axvline(bragg_over_sqrt_n, color="k", lw=0.9, ls="-.", alpha=0.7)
    axs.text(bragg_over_sqrt_n + 0.12, 0.30, r"$Q_{\rm WC}/\sqrt{n} = 6.7517$",
             fontsize=9, rotation=90, va="bottom")
    axs.set_ylabel(r"$S(q)$")
    axs.set_yscale("log")
    axs.set_ylim(1.5e-2, 4.0e1)
    axs.legend(ncol=2, fontsize=8.0, frameon=False, loc="lower right",
               columnspacing=1.0, handletextpad=0.5)
    axs.set_title(r"static structure factor on the allowed torus momenta, $\nu = 1$",
                  fontsize=11)
    axs.grid(alpha=0.25, lw=0.5, which="both")
    axs.tick_params(labelbottom=False)

    axw = fig.add_subplot(gs[1], sharex=axs)
    axw.axhline(1.0, color="0.35", lw=1.0, ls=":", alpha=0.9)
    axw.text(0.30, 0.06, r"Kohn limit $\omega/\omega_c = 1$", fontsize=8.5, color="0.35",
             transform=axw.transAxes)
    for k in ladder:
        for kind, mk, ls, lw in (("liquid", "o", "-", 1.5), ("crystal", "s", "--", 1.3)):
            q_, S_, eS_ = (sqa[(kind, k)][:, 0], sqa[(kind, k)][:, 1],
                           sqa[(kind, k)][:, 2])
            w = q_ ** 2 / (2 * np.maximum(S_, 1e-12))
            # The error on omega is the error on S scaled by |domega/dS|, which for
            # omega = q^2/2S is just omega/S -- so a relative error carried across.
            ew = w * eS_ / np.maximum(S_, 1e-12)
            axw.errorbar(qn, w, yerr=ew, color=colsr[k], lw=lw, ls=ls, marker=mk,
                         ms=3.2 if kind == "liquid" else 3.0, capsize=1.6,
                         alpha=1.0 if kind == "liquid" else 0.9)

    # the classical magnetoplasmon, on the small-q shells only
    m = qn <= 2.6
    for k in ladder:
        wmp = np.sqrt(1 + k * qg[m])
        axw.plot(qn[m], wmp, ":", color="#7b241c", lw=1.3, alpha=0.55, zorder=1)
        axw.annotate(rf"$\kappa={k:g}$", xy=(qn[m][0], wmp[0]), xytext=(-6, -1),
                     textcoords="offset points", ha="right", fontsize=7.2,
                     color="#7b241c")
    axw.plot([], [], ":", color="#7b241c", lw=1.4, alpha=0.8,
             label=r"classical magnetoplasmon $\sqrt{1+\kappa q}$ (analytic, small $q$ only)")
    axw.plot([], [], "-", color="0.6", lw=1.5, label="liquid")
    axw.plot([], [], "--", color="0.6", lw=1.5, label="crystal")
    axw.axvline(bragg_over_sqrt_n, color="k", lw=0.9, ls="-.", alpha=0.7)
    axw.set_xlabel(r"$q/\sqrt{n}$")
    axw.set_ylabel(r"$\omega_{\rm SMA}/\omega_c = q^2/2S(q)$")
    axw.set_ylim(0, None)
    axw.legend(fontsize=8.2, frameon=False, loc="upper left")
    axw.grid(alpha=0.25, lw=0.5)
    axw.set_title("SMA collective mode; a peak in $S(q)$ must appear as a dip here",
                  fontsize=11)
    k_last = ladder[-1]
    S_last = sqa[("crystal", k_last)][i_g1, 1]
    axw.annotate("crystal soft mode at the Bragg shell",
                 xy=(qn[i_g1], qg[i_g1] ** 2 / (2 * max(S_last, 1e-12))),
                 xytext=(qn[i_g1] + 0.5, 1.35), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.9))

    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.07, top=0.94)
    return _finish(fig, path, dpi)


# ==========================================================================
# Figure 3 -- the two ansaetze as a density in the plane
# ==========================================================================
def real_space_density(grids, lattice, ladder, nbins, bin_width, kernel_width,
                       pad_bins: int = 6, path: Optional[str] = None, dpi: int = 170):
    """Two rows of smoothed density maps, one column per coupling.

    `grids[(kind, kappa)]` is the RAW histogram from the density estimator; the
    smoothing is applied here at `kernel_width`, which is a display choice and is stated
    on the figure as one.  `lattice` supplies the supercell vectors and the 36 sites.

    Three decisions, each of which was wrong in an earlier version.

    * ONE COLOUR SCALE PER ROW, as the rest of this campaign's map figures do.  The
      crystal row spans an order of magnitude more than the liquid row, so a single
      scale would flatten one of them; within a row the heights are comparable, and the
      caption says so.  The scale is the 99.5th percentile of the row's strongest panel,
      not the maximum -- one hot pixel should not set the range for ten panels.

    * THE WINDOW IS ONE CELL PLUS A PERIODIC MARGIN, and the cell is outlined.  Drawing
      exactly one cell misleads at the vertices: the ansatz places a site at fractional
      (0,0), so at each corner that site's Gaussian is cut to the 60-degree wedge inside
      the cell while the remaining 300 degrees land on the other three corners.  The sum
      is exact -- nothing is double-counted -- but a corner peak reads as a half-blob
      clipped by the boundary.  Carrying the margin draws every boundary peak whole and
      makes the periodicity explicit, which is the honest way to show a torus.

    * NO SITE MARKERS, on either row.  The peaks ARE the sites, so a dot on each one
      restates the ansatz's own definition on top of its own output and sits exactly on
      the maxima the panel exists to show.  The outlined cell alone carries the
      registration: every peak outside it is a periodic image of one of its 36 sites.
    """
    ladder = list(ladder)
    from ..analysis.structure import periodic_gaussian_blur

    fig = plt.figure(figsize=(21.0, 6.0))
    # The colour-bar column is sized by width_ratios rather than left as one more panel
    # column: given equal ratios the bar comes out as wide as a map, which puts a
    # 1.4-inch slab of colour next to panels whose whole signal is a colour.
    gs = fig.add_gridspec(2, len(ladder) + 1, wspace=0.14, hspace=0.34,
                          width_ratios=[1.0] * len(ladder) + [0.20],
                          left=0.045, right=0.90, top=0.74, bottom=0.09)

    full = nbins + 2 * pad_bins
    fv = np.arange(-pad_bins, nbins + pad_bins + 1) / nbins
    F1, F2 = np.meshgrid(fv, fv, indexing="ij")
    cart = np.stack([F1, F2], axis=-1).reshape(-1, 2) @ lattice.sc.T
    XC = cart[:, 0].reshape(full + 1, full + 1)
    YC = cart[:, 1].reshape(full + 1, full + 1)
    cell = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]) @ lattice.sc.T

    for r, kind in enumerate(("liquid", "crystal")):
        sm = {k: periodic_gaussian_blur(grids[(kind, k)], kernel_width / bin_width)
              for k in ladder}
        vmax = max(float(np.percentile(sm[k], 99.5)) for k in ladder)
        for c, k in enumerate(ladder):
            ax = fig.add_subplot(gs[r, c])
            m = ax.pcolormesh(XC, YC, np.pad(sm[k], pad_bins, mode="wrap"),
                              vmin=0.0, vmax=vmax, shading="flat", cmap="viridis")
            ax.plot(cell[:, 0], cell[:, 1], "-", lw=0.7, color="w", alpha=0.45)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(rf"$\kappa$={k:g}" + "\n" + rf"$r_s$={k * math.sqrt(2):.1f}",
                         fontsize=8.5)
            if c == 0:
                ax.set_ylabel({"liquid": "liquid ansatz",
                               "crystal": "crystal ansatz"}[kind], fontsize=10)
        cax = fig.add_subplot(gs[r, len(ladder)])
        cb = fig.colorbar(m, cax=cax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(r"$\rho\,/\,n$", fontsize=9)

    fig.suptitle(
        "The two ansätze as a density in the plane, along the coupling ladder\n"
        f"grid {nbins}$\\times${nbins} (bin {bin_width:.3f} $\\ell_B$), "
        f"Gaussian kernel {kernel_width:.2f} $\\ell_B$, {len(ladder)} couplings, "
        "one sampling recipe; one colour scale per row\n"
        "one cell outlined, shown with a periodic margin so that peaks on a boundary "
        "are drawn whole; the four corners of a cell are one site", fontsize=11)
    return _finish(fig, path, dpi)


# The four that CANNOT be redrawn, named here so the manifest and the code agree about
# which they are and why.  Their plotted arrays were never written to disk; the only way
# to rebuild them is to re-run VMC, which figure code is forbidden to do.
LEGACY_ONLY = (
    ("structure_factor_phase_map.png",
     "S(q) maps over the (r_s, B/B_0) plane; the per-point arrays were reduced inside "
     "the cell and never stored"),
    ("correlation_function_phase_map.png",
     "the real-space companion of the above, same missing intermediates"),
    ("shell_averaged_companions.png",
     "shell-averaged S(q) curves across the phase map; the reduction was never written"),
    ("energy_order_summary.png",
     "the energy-order table's figure; its inputs are the un-cached conv-run "
     "checkpoints"),
)
