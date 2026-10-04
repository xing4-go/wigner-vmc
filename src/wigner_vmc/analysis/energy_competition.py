"""The liquid-versus-crystal comparison, from converted states.

This is the new home of what `_diag/energy_scan.py` did.  It reads StateRecords --
never legacy JSON directly -- and routes every uncertainty through
`analysis/statistics.py`, so there is still exactly one sigma formula in the project.

The tier discipline is carried over verbatim, including the guard, because the guard
caught a real omission rather than a hypothetical one: a measured `nmax` n_b=4 state
sat outside every list while `nmax` itself was listed, and was dropped from the points
table with nothing said.  An exclusion nobody can see is indistinguishable from an
oversight, so unlisted (tier, n_band) pairs are a hard error.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from ..io.schema import CRYSTAL, LIQUID, StateRecord
from .statistics import bracket, compare_phases, brackets_overlap

# n_band = n_max + 1.  These are the tiers that form the n_max ladder.
RUNG = (("conv", 2), ("nest", 3), ("nest4", 4), ("nmax", 3))

# Tiers deliberately not rungs at all.  Each needs its reason written here, because the
# coverage guard below treats an unlisted tier as a bug rather than a design decision.
BY_DESIGN = {
    "fast": ("a low-budget single-seed survey across 40-90 -- not a converged "
             "multi-start point, so it cannot support a bracket or a rung"),
}

# (tier, n_band) PAIRS excluded although their TIER is a rung.  The guard must be keyed
# by the pair: a name-only check cannot see a new n_b arriving under an already-listed
# tier.
BY_DESIGN_PAIRS = {
    ("nmax", 4): ("fresh-L0 single-seed top of the r_s = 45 ladder; it does not nest "
                  "from the n_b = 3 rung (it lands 0.079468 above it), so it cannot be "
                  "drawn as a ladder rung"),
}

# Caveats that travel with the numbers into the figure.
CAVEAT_NEST_SIGMA_INIT = (
    "the nest tier's sigma_init understates the optimiser component: its five states "
    "are seeded from the five conv states of the SAME seed, so they are not five "
    "independent optimisations")
CAVEAT_NBAR = (
    "P_n / nbar_LL must not be compared across n_band: at n_b=2 nbar = P_1, at n_b=3 "
    "nbar = P_1 + 2*P_2, so the band counts are not the same quantity")
CAVEAT_WINDOW = (
    "a bracket is meaningful only WITHIN one truncation: n_b = n_max + 1, so the conv "
    "and nest tiers are different variational families that change sign at different "
    "r_s.  Brackets are keyed by tier and never merge; the nmax tier is excluded "
    "entirely because its states are fresh-L0 single-seed starts, not multi-start "
    "converged points")


class CoverageError(RuntimeError):
    """A measured (tier, n_band) pair that no list accounts for."""


def check_coverage(records: Sequence[StateRecord]) -> List[tuple]:
    """Every crystal (tier, n_band) present must be listed as a rung or excluded.

    Raises rather than warns.  Writing a scan that silently omits a measured rung is
    the failure this exists to prevent.
    """
    seen = sorted({(str(r.raw.get("tier")), int(r.raw.get("nb", 2)))
                   for r in records
                   if r.phase == CRYSTAL and r.raw.get("nb") is not None})
    unknown = [p for p in seen
               if p not in RUNG and p not in BY_DESIGN_PAIRS
               and p[0] not in BY_DESIGN and not p[0].startswith("ll")]
    if unknown:
        raise CoverageError(
            f"crystal (tier, n_band) pairs present but not read: {unknown}.  Their states "
            f"would not appear in the points table or in any bracket, and nothing else "
            f"would say so.  Add each to RUNG if it is part of the n_max ladder, or to "
            f"BY_DESIGN_PAIRS with the reason it is not.")
    return seen


def index_liquids(records: Sequence[StateRecord]) -> Dict[float, StateRecord]:
    """One liquid per coupling.  Keeps the FIRST and reports a duplicate rather than
    averaging two walks into a number that is neither."""
    out: Dict[float, StateRecord] = {}
    for r in records:
        if r.phase != LIQUID:
            continue
        rs = round(float(r.rs), 4)
        out.setdefault(rs, r)
    return out


def build_scan(records: Sequence[StateRecord]) -> tuple:
    """Reduce converted states to (points, brackets, notes).

    Returns points sorted by (r_s, tier) and one Bracket per tier that has points.
    """
    check_coverage(records)
    liquids = index_liquids(records)
    notes: List[str] = []

    points = []
    for tier, nb in RUNG:
        by_rs: Dict[float, List[StateRecord]] = {}
        for r in records:
            if r.phase != CRYSTAL:
                continue
            if r.raw.get("tier") != tier or int(r.raw.get("nb", 2)) != nb:
                continue
            by_rs.setdefault(round(float(r.rs), 4), []).append(r)

        for rs in sorted(by_rs):
            liq = liquids.get(rs)
            if liq is None:
                notes.append(f"{tier} n_b={nb} r_s={rs:g}: no liquid at this coupling "
                             f"-- skipped")
                continue
            cry = sorted(by_rs[rs], key=lambda r: r.init_id)
            caveats = []
            if tier == "nest":
                caveats.append(CAVEAT_NEST_SIGMA_INIT)
            points.append(compare_phases(cry, liq, tier, caveats))

    points.sort(key=lambda p: (p.rs, p.tier))

    brackets = []
    for tier, _nb in (("conv", 2), ("nest", 3), ("nest4", 4)):
        if any(p.tier == tier for p in points):
            brackets.append(bracket(points, tier, [CAVEAT_WINDOW]))

    # Cross-tier statement.  Returned, never asserted: the intersection, not either
    # bracket alone, is the strongest thing a two-truncation scan supports.
    conv = next((b for b in brackets if b.tier == "conv"), None)
    nest = next((b for b in brackets if b.tier == "nest"), None)
    overlap = brackets_overlap(conv, nest) if (conv and nest) else None
    if conv and nest and conv.resolved and nest.resolved:
        if overlap:
            notes.append(f"the two truncations' brackets OVERLAP on r_s = "
                         f"[{overlap[0]:g}, {overlap[1]:g}]; that intersection is the "
                         f"strongest statement the scan supports")
        else:
            notes.append(f"*** the two truncations' brackets do NOT overlap: their union "
                         f"is r_s = [{min(conv.lo, nest.lo):g}, "
                         f"{max(conv.hi, nest.hi):g}], and that is a bracket of neither")

    return points, brackets, notes


def gaussian_proto_points(states: Sequence[StateRecord]):
    """The Gaussian workflow's delta_E points, from `read_proto_energy`.

    Both columns of that store are already reduced, so there is no per-seed record and
    therefore no `sigma_init`.  The returned points carry a zero `sigma_init` and a
    caveat saying so, rather than a fabricated one.
    """
    by_rs = {}
    for s in states:
        by_rs.setdefault(round(s.rs, 6), {})[s.phase] = s

    pts = []
    for rs in sorted(by_rs):
        pair = by_rs[rs]
        liq, cry = pair.get(LIQUID), pair.get(CRYSTAL)
        if not (liq and cry):
            continue
        p = compare_phases([cry], liq, tier="proto")
        p.caveats = ["derived prototype columns: no per-seed record exists, so no "
                     "start-to-start spread can be formed and sigma_init is 0 by "
                     "construction, not by measurement"]
        pts.append(p)
    return pts


# ==========================================================================
# the kappa-prototype scan and its phase boundary
# ==========================================================================
KAPPAS_PROTO = (2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 64.0, 80.0)
L0_GRID_PROTO = (0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8)


class KappaScan:
    """The Jastrow scan over the kappa grid: one minimum per coupling, plus the liquid.

    NOT the same dataset as the part1 converged transition scan, and the two must never be
    pooled.  This one varies kappa on a coarse grid with the Jastrow optimised at each
    point; the LL-rotation transition scan varies the band-occupation truncation at fixed
    couplings.  They share a symbol (kappa) and nothing else.

    `E_cry` is the minimum over the L0 grid, and `E_cry_err` is the error at the
    MINIMISING width -- not the smallest error on the grid, which is a different state and
    would understate it.
    """

    def __init__(self, kappas, E_cry, E_cry_err, L0_opt, E_liq, E_liq_err):
        self.kappas = np.asarray(kappas, float)
        self.E_cry = np.asarray(E_cry, float)
        self.E_cry_err = np.asarray(E_cry_err, float)
        self.L0_opt = np.asarray(L0_opt, float)
        self.E_liq = np.asarray(E_liq, float)
        self.E_liq_err = np.asarray(E_liq_err, float)

    @property
    def rs(self):
        """r_s = sqrt(2) kappa, because r_s = sqrt(2) * e^2 and e^2 = kappa here."""
        return self.kappas * math.sqrt(2.0)

    @property
    def delta_e(self):
        """E_crystal - E_liquid, per particle.  Negative means the crystal wins."""
        return self.E_cry - self.E_liq


def start_width(c) -> float:
    """The walk's STARTING Jastrow width, from a `crystalJ_*` tag.

    Deliberately not `tag_field("L")`.  A real tag reads

        v1|ne36|LU30|J5|crystJ|k2|L0.8|n600|e300|s17|sig0.3|sn8|i182

    and holds BOTH the engine's LU block size (`LU30`) and this walk's starting width
    (`L0.8`), so the prefix `L` has two answers and `tag_field` refuses it rather than
    silently returning `30` -- which is a block size being plotted as a width, with no
    error anywhere.  The width field is the one that is `L` followed by a number.

    RAISES when that is not exactly one field: a tag the reader does not recognise must
    stop the scan, not contribute a guessed width to the figure's L0 annotation.
    """
    cands = [p for p in c.tag_fields() if p.startswith("L") and not p.startswith("LU")]
    if len(cands) != 1:
        raise ValueError(
            f"{c.name}: expected exactly one L-width field in {c.tag!r}, found {cands}")
    try:
        return float(cands[0][1:])
    except ValueError:
        raise ValueError(
            f"{c.name}: L-width field {cands[0]!r} in {c.tag!r} is not a number -- this "
            f"walk's tag does not record a starting width.")


def kappa_scan(crystal: dict, liquid: dict, kappas=KAPPAS_PROTO) -> KappaScan:
    """Assemble the scan from `{kappa: [Checkpoint, ...]}` and `{kappa: Checkpoint}`.

    `crystal[k]` is every L0 on the grid for that coupling; the minimum is taken here,
    once, rather than by each caller -- a second minimum somewhere else is how two figures
    come to disagree about which width was optimal.
    """
    ks, ec, ee, l0, el, elr = [], [], [], [], [], []
    for k in kappas:
        rows = crystal.get(float(k))
        if not rows:
            continue
        best = min(rows, key=lambda c: c.fields["E"])
        if float(k) not in liquid:
            continue
        ks.append(float(k))
        ec.append(best.fields["E"])
        ee.append(best.fields["E_err"])
        l0.append(start_width(best))
        el.append(liquid[float(k)].fields["E"])
        elr.append(liquid[float(k)].fields["E_err"])
    return KappaScan(ks, ec, ee, l0, el, elr)


class PhaseBoundary:
    """The three sigma columns and the crossing, all derived from one scan and one corr.

    sigma_rep  -- the bars as first computed, hypot of the two production errors
    sigma_cor  -- the same after the per-state autocorrelation factors
    sigma_blk  -- the same under the model-free block-8 factors (the sensitivity)
    cross      -- the raw central-value zero of delta_E, linear interpolation, unadjusted
    degenerate -- |delta_E| < sigma_cor: where THIS calculation cannot resolve the sign

    The three are kept separate because they are three different claims.  `cross` is a
    number from the data, `degenerate` is the range the calculation cannot resolve, and
    the published r_s is somebody else's number.  Collapsing them would be the error.
    """

    def __init__(self, sigma_rep, sigma_cor, sigma_blk, cross, degenerate, rs_pub=46.5):
        self.sigma_rep = np.asarray(sigma_rep, float)
        self.sigma_cor = np.asarray(sigma_cor, float)
        self.sigma_blk = np.asarray(sigma_blk, float)
        self.cross = cross
        self.degenerate = np.asarray(degenerate, bool)
        self.rs_pub = float(rs_pub)


def phase_boundary(scan: KappaScan, corr: dict, rs_pub: float = 46.5) -> PhaseBoundary:
    """Propagate the autocorrelation factors into the phase-boundary bars.

    `corr` is `{kind: {"corr": x, "block_corr": y}}` from
    `statistics.autocorrelation_correction`.  The two estimators are BOTH carried: the
    tau_int one sets the window the figure draws, and the block one is drawn as a
    sensitivity, because on a sample this short neither has converged and choosing one is
    a choice rather than a result.
    """
    cC, cL = corr["crystal"], corr["liquid"]
    sig_rep = np.hypot(scan.E_cry_err, scan.E_liq_err)
    sig_cor = np.hypot(scan.E_cry_err * cC["corr"], scan.E_liq_err * cL["corr"])
    sig_blk = np.hypot(scan.E_cry_err * cC["block_corr"],
                       scan.E_liq_err * cL["block_corr"])

    d = scan.delta_e
    cross = None
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            w = d[i] / (d[i] - d[i + 1])
            cross = float(scan.kappas[i] + w * (scan.kappas[i + 1] - scan.kappas[i]))
            break
    return PhaseBoundary(sig_rep, sig_cor, sig_blk, cross, np.abs(d) < sig_cor, rs_pub)


def resolved_fraction(delta_e, sigma) -> int:
    """How many couplings resolve the sign of delta_E at these bars."""
    d = np.asarray(delta_e, float)
    s = np.asarray(sigma, float)
    return int((np.abs(d) / np.maximum(s, 1e-30) > 1).sum())
