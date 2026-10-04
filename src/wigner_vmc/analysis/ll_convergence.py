"""Landau-level convergence: band occupations, and whether the n_max ladder nests.

Two questions, both about the truncation rather than about the phase:

1. **How much weight sits above the lowest band?**  The crystal's orbitals are a
   rotation of the filled LLL by a unitary acting in the LL index.  `P_n` is the
   occupation of band n; `nbar = sum_n n*P_n` collapses it to one number.  `nbar` is
   the honest measure of how truncated the ansatz is, and it must be read WITH its
   band count: at n_b=2 nbar = P_1, at n_b=3 nbar = P_1 + 2*P_2, so the same value of
   nbar means different things at different n_b.  The table therefore never pools
   across n_b, and says so.

2. **Does the ladder nest?**  n_b=3 CONTAINS n_b=2.  So a state optimised at n_b=3,
   with its mixing set to zero, must reproduce the n_b=2 energy exactly -- the nesting
   identity.  `nest_identity_dev` is the measured residual and `nest_identity_tol` the
   tolerance it was tested against; both are stored in the record, so this module
   reports the test rather than re-deriving the tolerance.

The uncertainty on a band occupation is the SEED SCATTER, not the MC error: the
question "how much does P_1 depend on where the optimiser started" is a question about
the seed set, and the walk's own error bar answers a different one.  It is computed by
`statistics.seed_spread`.

Why `seed_spread` and not `sigma_init`, when the two are the same formula
-----------------------------------------------------------------------
They differ on one input, n = 1, and this is the one place in the project where the
legacy campaign answered that case two different ways.  The energy path writes 0.0 (a
single state contributes no basin spread to the budget); `pn_table.json` writes null (a
single seed cannot bound the spread at all).  The band table follows the second, because
that is the artifact it regresses against -- and because a null is the honest answer
here.  The choice is not cosmetic: a single-seed row carries a 0 in one convention and
an unknown in the other, and a reader who mistakes the first for the second would treat
an unbounded quantity as a tightly measured one.  `pn_table_vs_records` therefore
compares null-to-null and 0-to-0 as agreement but null-to-0 as a mismatch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..io.schema import CRYSTAL, StateRecord
from .statistics import seed_spread

# A record may report n_band or only n_max; n_band = n_max + 1 either way.
def _n_band(r: StateRecord):
    if r.raw.get("nb") is not None:
        return int(r.raw["nb"])
    if r.raw.get("n_band") is not None:
        return int(r.raw["n_band"])
    return None if r.nmax is None else r.nmax + 1


def band_table(records: Sequence[StateRecord]) -> List[dict]:
    """Per (n_band, tier, r_s): mean band occupations and their seed scatter.

    Keyed by n_b on purpose -- see the module docstring.
    """
    groups: Dict[tuple, List[StateRecord]] = {}
    for r in records:
        if r.phase != CRYSTAL or not r.band_occupations:
            continue
        nb = _n_band(r)
        if nb is None:
            continue
        groups.setdefault((nb, str(r.raw.get("tier")), round(float(r.rs), 4)), []).append(r)

    rows = []
    for (nb, tier, rs) in sorted(groups):
        states = sorted(groups[(nb, tier, rs)], key=lambda r: r.init_id)
        occ = [list(map(float, s.band_occupations)) for s in states
               if len(s.band_occupations) == nb]
        if not occ:
            continue
        n = len(occ)
        p_mean = [sum(o[i] for o in occ) / n for i in range(nb)]
        p_sem = [seed_spread([o[i] for o in occ]) for i in range(nb)]
        nbar = sum(i * p_mean[i] for i in range(nb))
        caveats = [
            "P_n not comparable across n_band",
            "uncertainty is seed scatter (ddof=1)/sqrt(n), not MC error",
            "nbar_LL = sum_n n*P_n",
        ]
        if n < 2:
            caveats.append(
                f"n_seeds = {n}: P_sem is null, not zero -- one optimisation cannot "
                f"bound the seed spread, so this row's uncertainty is UNKNOWN, not small")
        rows.append({
            "n_band": nb,
            "n_max": nb - 1,
            "tier": tier,
            "rs": rs,
            "n_seeds": n,
            "P_mean": p_mean,
            "P_sem": p_sem,
            "nbar_LL_mean": nbar,
            "nbar_LL_sem": seed_spread([sum(i * o[i] for i in range(nb)) for o in occ]),
            "keys": [s.init_id for s in states],
            "caveats": caveats,
        })
    return rows


def nesting_report(records: Sequence[StateRecord]) -> List[dict]:
    """The nesting-identity test, as the records recorded it.

    Reports only; it does not recompute the tolerance, because the record carries the
    one the test was actually run against and re-deriving it would let the two drift.
    """
    out = []
    for r in sorted(records, key=lambda r: (r.rs, r.init_id)):
        if r.parent_id is None:
            continue
        dev = r.raw.get("nest_identity_dev")
        tol = r.raw.get("nest_identity_tol")
        if dev is None:
            continue
        out.append({
            "key": r.init_id,
            "rs": r.rs,
            "n_max": r.nmax,
            "parent": r.parent_id,
            "identity_dev": dev,
            "identity_tol": tol,
            "passes": (abs(dev) <= tol) if tol is not None else None,
            "start_err": r.raw.get("nest_start_err"),
        })
    return out


def _close(a: Optional[float], b: Optional[float], tol: float) -> bool:
    """Agreement test that treats null as a value, not as a missing number.

    null vs null is agreement (both say "unknown"); null vs a number is a mismatch --
    including null vs 0.0, which is the whole point.  Tolerating that pair would erase
    the distinction between "the spread is zero" and "the spread was never measured",
    and six rows of `pn_table.json` sit on exactly that distinction.
    """
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def pn_table_vs_records(rows_from_records: List[dict], legacy_rows: List[dict],
                        tol: float = 1e-9) -> List[dict]:
    """Compare recomputed band rows against the legacy `pn_table.json`.

    Returns a list of STRUCTURED complaints; empty means agreement.  A regression check
    rather than an assertion, so the caller can report the drift instead of dying on it.

    Structured rather than formatted on purpose.  The caller has to decide whether a
    disagreement means the pipeline is wrong or the legacy table is stale, and it can
    only do that from the seed lists -- which are fields, not prose.  An earlier version
    returned strings and the caller pattern-matched them; that classifier then silently
    missed one of the two rows it was meant to catch, because a string carries no
    structure to be wrong about.  Each complaint therefore carries its own evidence:

        row          (n_band, tier, r_s)
        kind         no_counterpart | n_band_mismatch | n_seeds | P_mean | P_sem
        index        which entry, for the per-entry kinds
        mine, legacy the two values
        mine_keys    the record seed ids behind `mine`
        legacy_keys  the seed ids the legacy row names ([] if it names none)

    Six legacy rows carry `P_sem: [null, null]` -- their tier has a single seed, and
    the legacy table chose to say so rather than to write 0.  Those rows are compared
    presence-to-presence, not numerically.
    """
    key = lambda d: (d["n_band"], d["tier"], round(float(d["rs"]), 4))    # noqa: E731
    have = {key(d): d for d in rows_from_records}
    out = []
    for lr in legacy_rows:
        k = key(lr)
        if k not in have:
            out.append({"row": k, "kind": "no_counterpart", "index": None,
                        "mine": None, "legacy": None, "mine_keys": [],
                        "legacy_keys": list(lr.get("keys", []))})
            continue
        mine = have[k]
        base = {"row": k, "mine_keys": list(mine["keys"]),
                "legacy_keys": list(lr.get("keys", []))}
        if len(mine["P_mean"]) != len(lr["P_mean"]):
            out.append(dict(base, kind="n_band_mismatch", index=None,
                            mine=len(mine["P_mean"]), legacy=len(lr["P_mean"])))
            continue
        # A different seed count is reported ONCE and the row's derived numbers are then
        # skipped: P_mean and P_sem are computed FROM the seed set, so when the sets
        # differ those two are consequences of this one fact, not two more findings.
        # Listing all three would triple-count a single disagreement and make a
        # one-row problem look like a three-row one.
        if mine["n_seeds"] != lr["n_seeds"]:
            out.append(dict(base, kind="n_seeds", index=None,
                            mine=mine["n_seeds"], legacy=lr["n_seeds"]))
            continue
        for i, (a, b) in enumerate(zip(mine["P_mean"], lr["P_mean"])):
            if not _close(a, b, tol):
                out.append(dict(base, kind="P_mean", index=i, mine=a, legacy=b))
        for i, (a, b) in enumerate(zip(mine["P_sem"], lr["P_sem"])):
            if not _close(a, b, tol):
                out.append(dict(base, kind="P_sem", index=i, mine=a, legacy=b))
    return out
