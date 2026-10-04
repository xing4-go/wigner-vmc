"""The seven Gaussian-workflow figures, redrawn from numbers.

Every function here takes already-reduced arrays and returns a Figure.  Nothing loads a
file, nothing knows where the frozen tree lives, and nothing can sample: the arrays were
produced from the frozen checkpoints by the analysis layer, and this module's only job is
to draw them.  `scripts/make_figures.py` is the composition root that does the loading.

That split is enforced mechanically by `tests/test_figure_contract.py`, which parses this
file and refuses it if it imports the sampler, calls `exec`/`eval`, names a legacy
directory, or calls anything whose name suggests sampling.

Faithfulness
------------
The layouts, axis labels, units and colour-bar caps are the legacy ones, reproduced from
the cells that drew them so the new figures are comparable with the frozen PNGs rather
than merely similar.  What is deliberately NOT reproduced is the pixel-exact render:
Stage 1's acceptance order is (1) the plotted numbers agree, (2) the error bars agree,
(3) the labels and physical meaning agree, (4) the axes and units agree, and only then
(5) the appearance is broadly similar.  So a restyled figure that plots the same
quantities is a pass, and no function below chases a hash.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

# The closed-form filled-LLL references.  Imported from the analysis layer rather than
# written out here, because the same two curves are the reference in the SMA figure, in
# the tests and in the LL-rotation module: two copies of 1 - exp(-q^2/2) is two chances
# for one of them to acquire a factor.
from ..analysis.structure import exact_lll_g, exact_lll_sq


def _finish(fig, path: Optional[str], dpi: int):
    """Save and close.  Closing matters: a figure left open leaks across a long run."""
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


# ==========================================================================
# fig01 -- the phase boundary
# ==========================================================================
def phase_boundary(scan, pb, path: Optional[str] = None, dpi: int = 110):
    """delta_E(r_s) with both sets of bars, and |delta_E|/sigma.

    Both bar sets are drawn and neither is suppressed.  Where the grey band covers zero
    the sign of delta_E is not resolved and the two states are not distinguishable by
    this calculation -- that window IS the result, not a failure to obtain one.  The band
    uses the block-averaged bars; the thinner caps show what the uncorrected bars would
    have claimed, and they are narrower, i.e. more confident, for no good reason.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13.8, 4.8),
                           gridspec_kw=dict(width_ratios=[1.7, 1]))
    rs = scan.rs
    d = scan.delta_e

    a = ax[0]
    a.axhline(0, color="k", lw=1.0)
    a.fill_between(rs, -pb.sigma_cor, pb.sigma_cor, color="0.82", zorder=1,
                   label=r"$\pm\sigma_{\Delta E}$, block-averaged")
    a.errorbar(rs, d, yerr=pb.sigma_cor, fmt="o", ms=5, lw=1.2, capsize=3,
               color="tab:red", zorder=4, label=r"$\Delta E = E_{\rm cry}-E_{\rm liq}$")
    a.errorbar(rs, d, yerr=pb.sigma_rep, fmt="none", lw=1.0, capsize=7,
               color="0.35", zorder=3, label="bars as originally computed")
    if pb.cross is not None:
        a.axvline(pb.cross * math.sqrt(2), color="tab:red", ls=":", lw=1.8, zorder=2,
                  label=f"(1) raw crossing,  $r_s$ = {pb.cross * math.sqrt(2):.1f}")
    if pb.degenerate.any():
        a.axvspan(scan.kappas[pb.degenerate].min() * math.sqrt(2),
                  scan.kappas[pb.degenerate].max() * math.sqrt(2),
                  color="tab:blue", alpha=0.10, zorder=0, label="(2) near-degeneracy window")
    a.axvline(pb.rs_pub, color="k", ls="--", lw=1.4, zorder=2,
              label=rf"(3) published $r_s$ = {pb.rs_pub} (Reddy & Fu)")
    a.set(xlabel=r"$r_s$", ylabel=r"$\Delta E / N\;\;[\hbar\omega_c]$",
          title="crystal minus liquid, both states with their optimised Jastrow")
    a.legend(fontsize=7.5, loc="upper left")
    a.grid(alpha=0.25)

    a = ax[1]
    a.plot(rs, np.abs(d) / pb.sigma_cor, "o-", ms=5, lw=1.2, color="tab:red",
           label="block-averaged bars")
    a.plot(rs, np.abs(d) / pb.sigma_rep, "s--", ms=4, lw=1.0, color="0.35",
           label="bars as originally computed")
    a.axhline(1.0, color="k", lw=1.2, ls=":")
    a.set_yscale("log")
    a.set(xlabel=r"$r_s$", ylabel=r"$|\Delta E|/\sigma_{\Delta E}$",
          title="is the sign of $\\Delta E$ resolved?\n(below 1: not by this calculation)")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return _finish(fig, path, dpi)


# ==========================================================================
# fig02 -- the filled-LLL calibration
# ==========================================================================
def lll_validation(qn, Sq, Sq_err, r, g, g_err,
                   path: Optional[str] = None, dpi: int = 110):
    """VMC against the closed-form filled-LLL S(q) and g(r), with the residual.

    This is the calibration of the whole structure pipeline, not just another data point:
    the filled LLL is the one state whose S(q) = 1 - exp(-q^2/2) and
    g(r) = (N/(N-1))(1 - exp(-r^2/2)) are known exactly, so a normalisation error
    anywhere upstream shows up here as a constant offset rather than as a plausible-
    looking curve.

    `g_err` must be the POISSON PAIR-COUNT bar `g/sqrt(n_pairs in the shell)` and not the
    snapshot scatter: the legacy cell drew that bar, and it is valid here only because
    the shell counts run to the thousands.  It is computed in the analysis layer
    (`structure.poisson_pair_error`), not here -- the figure receives a number, and the
    one place that decides what the bar means stays in one place.
    """
    exact_Sq = exact_lll_sq(qn)
    exact_g = exact_lll_g(r)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].errorbar(qn, Sq, Sq_err, fmt="o", ms=3, lw=.6, label="VMC")
    ax[0].plot(qn, exact_Sq, "-", lw=1, label=r"$1-e^{-q^2/2}$")
    ax[0].set(xlabel=r"$q\,\ell_B$", ylabel=r"$S(q)$", title="structure factor")
    ax[0].legend()

    ax[1].errorbar(r, g, g_err, fmt="o", ms=3, lw=.6, label="VMC")
    ax[1].plot(r, exact_g, "-", lw=1,
               label=r"$\frac{N}{N-1}(1-e^{-r^2/2})$")
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].text(3.0, 1.001, r"$1$", fontsize=9)
    ax[1].set(xlabel=r"$r/\ell_B$", ylabel=r"$g(r)$", title="pair correlation")
    ax[1].legend()

    ax[2].errorbar(r, g - exact_g, g_err, fmt="o-", ms=3, lw=.6)
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set(xlabel=r"$r/\ell_B$", ylabel=r"$g - g_{\rm exact}$", title="g(r) residual")
    fig.tight_layout()
    return _finish(fig, path, dpi)


# ==========================================================================
# fig06 -- 1-D fingerprints of the two states
# ==========================================================================
def fingerprints_1d(qon, S_liq, S_cry, S_free, r_liq, g_liq, r_cry, g_cry,
                    last_snap, sites, i_bragg, bragg_mag, label_liq, label_cry,
                    path: Optional[str] = None, dpi: int = 110):
    """S(q) at the WC reciprocal vectors, g(r), and one configuration.

    The 1-D view of the structure difference: the liquid's S(q) follows the filled-LLL
    curve with no periodic order, the crystal has a Bragg peak at |g1|, and the middle
    panel shows what that looks like in real space.
    """
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    ax[0].plot(qon, S_liq, "o", ms=3, label=f"liquid, {label_liq}")
    ax[0].plot(qon, S_cry, "s", ms=3, label=f"crystal, {label_cry}")
    ax[0].plot(qon, S_free, "-", lw=1, color="0.6", label="filled LLL (exact)")
    ax[0].axvline(bragg_mag, color="r", ls=":", lw=1)
    ax[0].set(xlabel=r"$q\,\ell_B$", ylabel=r"$S(q)$", title=r"Bragg peak at $|g_1|$")
    ax[0].legend(fontsize=8)

    ax[1].plot(r_liq, g_liq, "o-", ms=3, label=f"liquid, {label_liq}")
    ax[1].plot(r_cry, g_cry, "s-", ms=3, label=f"crystal, {label_cry}")
    ax[1].axhline(1, color="k", lw=.8)
    ax[1].set(xlabel=r"$r/\ell_B$", ylabel=r"$g(r)$", title="real-space order")
    ax[1].legend(fontsize=8)

    ax[2].plot(last_snap[:, 0], last_snap[:, 1], "o", ms=4, alpha=.7)
    ax[2].plot(sites[:, 0], sites[:, 1], "k+", ms=8)
    ax[2].set(title="one liquid configuration", xlabel=r"$x/\ell_B$",
              ylabel=r"$y/\ell_B$", aspect="equal")
    fig.tight_layout()
    return _finish(fig, path, dpi)


# ==========================================================================
# fig07 -- the two-dimensional structure maps
# ==========================================================================
def structure_maps(g_liq, g_cry, S_liq, S_cry, torus, rs_liq, rs_cry,
                   r_half, q_half, sqrt_n, awc, bragg,
                   caps=((1.25, 1.25), (2.50, 2.00)),
                   path: Optional[str] = None, dpi: int = 110):
    """g(x,y) and S(qx,qy) for both states, with the mask and the guides.

    The grey disc at the centre of the two S panels is MASKED, not zero.  It is the region
    0 < |q| < |G1|, which holds no allowed nonzero reciprocal momentum: the supercell's
    reciprocal lattice is generated by G1 and its 60-degree partner, so every allowed
    momentum is q = m G1 + n G2 and the shortest nonzero ones sit at |q| = |G1|.  Inside
    that radius the only allowed momentum is q = 0, where rho_q = N identically and
    S(0) = N by construction -- a statement about how many electrons are in the box.
    A gridded estimator asked for S(q) at an unrepresentable momentum returns that box
    form factor for a crystal and a liquid alike.  The mask hides no crystal information:
    the first reciprocal shell is at |g1| = 6|G1|, well outside it.
    """
    qg = np.linspace(-q_half, q_half, S_liq.shape[0])
    QX, QY = np.meshgrid(qg, qg, indexing="ij")

    def scale(M, cap):
        m = float(np.nanmax(M))
        return cap if m > cap else float(np.ceil(20 * m) / 20)

    v_g_liq, v_s_liq = scale(g_liq, caps[0][0]), scale(S_liq, caps[0][1])
    v_g_cry, v_s_cry = scale(g_cry, caps[1][0]), scale(S_cry, caps[1][1])

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.4")
    e_r = [-r_half * sqrt_n, r_half * sqrt_n] * 2
    e_q = [-q_half / sqrt_n, q_half / sqrt_n] * 2
    th = np.linspace(0, 2 * math.pi, 400)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10.4))

    def draw(a, M, ext, title, vmax, xlab, ylab):
        im = a.imshow(M.T, origin="lower", extent=ext, cmap=cmap, vmin=0.0, vmax=vmax,
                      interpolation="nearest", aspect="equal")
        a.grid(False)
        a.set(title=title, xlabel=xlab, ylabel=ylab)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(f"0 – {vmax:.2f}", fontsize=8)   # U+2013 EN DASH, as the legacy

    draw(ax[0, 0], g_liq, e_r, f"(a)  $g(x,y)$  liquid,  $r_s$ = {rs_liq:.1f}",
         v_g_liq, r"$\sqrt{n}\,x$", r"$\sqrt{n}\,y$")
    draw(ax[0, 1], S_liq, e_q, f"(b)  $S(q_x,q_y)$  liquid,  $r_s$ = {rs_liq:.1f}",
         v_s_liq, r"$q_x/\sqrt{n}$", r"$q_y/\sqrt{n}$")
    draw(ax[1, 0], g_cry, e_r, f"(c)  $g(x,y)$  Wigner crystal,  $r_s$ = {rs_cry:.1f}",
         v_g_cry, r"$\sqrt{n}\,x$", r"$\sqrt{n}\,y$")
    draw(ax[1, 1], S_cry, e_q, f"(d)  $S(q_x,q_y)$  Wigner crystal,  $r_s$ = {rs_cry:.1f}",
         v_s_cry, r"$q_x/\sqrt{n}$", r"$q_y/\sqrt{n}$")

    nn = awc * sqrt_n
    for a in (ax[0, 0], ax[1, 0]):
        a.plot(nn * np.cos(th), nn * np.sin(th), ":", color="w", lw=1.0, alpha=.6)
    for a in (ax[0, 1], ax[1, 1]):
        for q in bragg:
            a.plot(q[0] / sqrt_n, q[1] / sqrt_n, "o", mfc="none", mec="w", ms=12,
                   mew=1.0, alpha=.75)
    fig.tight_layout()
    return _finish(fig, path, dpi)


# ==========================================================================
# fig08 -- the SMA dispersion
# ==========================================================================
def sma(qon, S_liq, S_cry, S_free, label_liq, label_cry,
        path: Optional[str] = None, dpi: int = 110):
    """omega_SMA(q) for both states against the filled-LLL reference and Kohn's mode.

    The filled LLL's SMA must tend to hbar*omega_c as q -> 0 (Kohn's theorem), which is
    the one limit of this plot that is known independently.  The torus's smallest allowed
    q is not zero, so the figure shows the trend and never the limit.
    """
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for Sq_, lab, st in ((S_liq, f"liquid, {label_liq}", "o"),
                         (S_cry, f"crystal, {label_cry}", "s")):
        w = qon ** 2 / (2 * np.maximum(Sq_, 1e-6))
        ax.plot(qon, w, st + "-", ms=4, label=lab)
    # No clamp on the reference denominator, unlike the measured curves above: S_free is
    # exact and vanishes only at the q = 0 that `allowed_momenta` excludes, so a clamp
    # would be dead code that quietly disguises a wrong grid.  The measured curves DO
    # carry one, and that clamp is the legacy cell's own.
    ax.plot(qon, qon ** 2 / (2 * S_free), "-", color="0.6", lw=1.5,
            label="filled LLL (exact)")
    ax.axhline(1, color="k", ls="--", lw=1, label=r"Kohn mode $\hbar\omega_c$")
    ax.set(xlabel=r"$q\,\ell_B$", ylabel=r"$\omega_{\rm SMA}(q)\;[\hbar\omega_c]$",
           ylim=(0, 2.5), title="SMA variational estimate of the density-excitation energy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, path, dpi)


# ==========================================================================
# fig09 / fig10 -- the coupling ladder
# ==========================================================================
def _ladder_fig(maps, ladder, figsize_col=2.95, path=None, dpi=110,
                suptitle="", ext=None, ylab="", draw_marks=None):
    """2 rows (liquid, crystal) x len(ladder) columns, one shared scale PER ROW.

    Per row and not per figure: the crystal row spans an order of magnitude more than the
    liquid row, and a single scale would flatten one of them into a flat rectangle.  The
    consequence -- that colours are comparable along a row and not across rows -- is
    printed on every colour bar.
    """
    row = {kind: max(float(np.nanmax(maps[(kind, k)])) for k in ladder)
           for kind in ("liquid", "crystal")}
    fig, ax = plt.subplots(2, len(ladder), figsize=(figsize_col * len(ladder), 6.6))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.4")
    for r, kind in enumerate(("liquid", "crystal")):
        for c, k in enumerate(ladder):
            a = ax[r, c]
            v = row[kind]
            im = a.imshow(maps[(kind, k)].T, origin="lower", extent=ext, cmap=cmap,
                          vmin=0.0, vmax=v, interpolation="nearest", aspect="equal")
            a.grid(False)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{kind},  $r_s$ = {k * math.sqrt(2):.1f}", fontsize=9)
            if c == 0:
                a.set_ylabel(ylab, fontsize=9)
            cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(f"0 - {v:.2f}", fontsize=7)
            if draw_marks is not None:
                draw_marks(a)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return _finish(fig, path, dpi)


def ladder_Sq(maps, ladder, q_half, sqrt_n, bragg,
              path: Optional[str] = None, dpi: int = 110):
    """The transition in reciprocal space: S(qx,qy) along the coupling ladder."""
    def marks(a):
        for q in bragg:
            a.plot(q[0] / sqrt_n, q[1] / sqrt_n, "o", mfc="none", mec="w", ms=11,
                   mew=1.0, alpha=.75)

    return _ladder_fig(maps, ladder, path=path, dpi=dpi,
                       ext=[-q_half / sqrt_n, q_half / sqrt_n] * 2,
                       ylab=r"$q_y/\sqrt{n}$", draw_marks=marks,
                       suptitle="The transition in reciprocal space: $S(q_x,q_y)$ for "
                                "both states along the coupling ladder")


def ladder_g(maps, ladder, r_half, sqrt_n, awc,
             path: Optional[str] = None, dpi: int = 110):
    """The same ladder in real space: g(x,y) for both states."""
    th = np.linspace(0, 2 * math.pi, 400)
    nn = awc * sqrt_n

    def marks(a):
        a.plot(nn * np.cos(th), nn * np.sin(th), ":", color="w", lw=1.0, alpha=.6)

    return _ladder_fig(maps, ladder, path=path, dpi=dpi,
                       ext=[-r_half * sqrt_n, r_half * sqrt_n] * 2,
                       ylab=r"$\sqrt{n}\,y$", draw_marks=marks,
                       suptitle="The same ladder in real space: $g(x,y)$ for both "
                                "states along the coupling ladder")
