"""Readers for the frozen legacy result files.  CONVERSION, NEVER RECOMPUTATION.

Every function here opens a file written by the old campaign, reads the fields that
already exist in it, and returns canonical records.  Nothing is re-measured, no walk
is re-run, no theta is re-optimised.  That is what makes the Stage 1 regression check
meaningful: if this layer produced different numbers from the legacy JSON, the
difference would have to come from a conversion bug, because there is no other place
it could come from.

The legacy stores, and what each is
-----------------------------------
`_diag/part1/*.json`      the transition scan's measurements.  Crystal records carry
                          the production energy; liquid records are the matching
                          reference walk at the same coupling.  PRIMARY.
`_diag/v_analysis.json`   the walk-analysis records for the LL-rotated crystal.  These
                          carry T and V but NO energy and NO covariance, so they yield
                          structure observables only -- see the note on `read_v_analysis`.
`_diag/energy_scan.json`  a derived product: the delta_E points and windows, already
                          reduced by the old campaign.  Read here as the regression
                          TARGET, never as an input to a new analysis.
`_diag/proto_*.json`      the Gaussian workflow's own prototypes.
others                    p1_report / pn_table / mech_probe / reweight / chains.

Energy field discipline
-----------------------
`E_perpart` is the independent production measurement per particle; it is the only
field converted into `production_energy_per_particle`.  `E` is the same number times
N, `E_min` and `E_final` are optimiser-trajectory quantities.  See `io/schema.py`.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Optional

from .schema import (CRYSTAL, GAUSSIAN, LIQUID, LLROT, N_ELECTRONS,
                     OptimizationMeta, Provenance, StateRecord)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_hash_cache: dict = {}


def sha256_of(path: str) -> str:
    if path not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        _hash_cache[path] = h.hexdigest()
    return _hash_cache[path]


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _opt_meta(r: dict) -> OptimizationMeta:
    """Pull the optimiser trajectory out of a legacy record, into the quarantined bag."""
    def g(k):
        return r.get(k)
    return OptimizationMeta(
        start_energy_per_particle=g("E_start"),
        final_energy_per_particle=g("E_final"),
        min_energy_per_particle=g("E_min"),
        steps=g("steps"),
        sweeps=g("sweeps"),
        force_first=g("force_first"),
        force_final=g("force_final"),
        cond_final=g("cond_final"),
        c_norm_dev=g("c_norm_dev"),
    )


def _energy_cross_check(r: dict) -> Optional[str]:
    """E_perpart should equal (T + kappa*V)/N.  Returns a complaint, or None.

    The legacy records carry T and V as TOTALS with the bare Coulomb sum in V, so the
    identity is E = T + kappa*V.  Checking it here catches a record read through the
    wrong field far more cheaply than a wrong figure would.
    """
    for k in ("E_perpart", "T", "V", "kappa"):
        if r.get(k) is None:
            return None
    want = (r["T"] + r["kappa"] * r["V"]) / N_ELECTRONS
    got = r["E_perpart"]
    if abs(want - got) > 1e-3:
        return (f"{r.get('key')}: E_perpart={got:+.6f} but (T+kappa*V)/N={want:+.6f} "
                f"-- the record does not satisfy the energy identity")
    return None


# --------------------------------------------------------------------------
# part1 -- the primary transition-scan store
# --------------------------------------------------------------------------
def part1_state(r: dict, path: str, workflow: str = LLROT) -> StateRecord:
    """Convert one `_diag/part1/*.json` record."""
    tier = str(r.get("tier", ""))
    nb = r.get("nb")
    nest_from = r.get("nest_from")
    is_liquid = r.get("kind") == "liquid"

    return StateRecord(
        workflow=workflow,
        phase=LIQUID if is_liquid else CRYSTAL,
        rs=float(r["rs"]),
        N=N_ELECTRONS,
        kappa=r.get("kappa"),
        nmax=(None if is_liquid or nb is None else int(nb) - 1),
        L0=r.get("L0"),
        init_id=str(r.get("key") or r.get("label") or ""),
        rng_seed=(r.get("seed_mc") if r.get("seed_mc") is not None else r.get("seed")),
        parent_id=nest_from,
        nested=bool(nest_from) or tier in ("nest", "nest4"),
        production_energy_per_particle=r.get("E_perpart"),
        production_mc_error=r.get("E_perpart_err"),
        total_energy=r.get("E"),
        kinetic_total=r.get("T"),
        potential_total=r.get("V"),
        nbar=r.get("nbar"),
        band_occupations=r.get("P"),
        R_B=r.get("R_B"),
        S_QWC=r.get("S_QWC"),
        S_bg=r.get("S_bg"),
        acceptance=r.get("acc"),
        optimization=_opt_meta(r),
        provenance=Provenance(
            source_path=path, source_sha256=sha256_of(path),
            legacy_energy_field="E_perpart", legacy_error_field="E_perpart_err",
            legacy_label=str(r.get("key") or r.get("label") or ""),
            notes=f"tier={tier}; kappa recorded rounded for some records",
        ),
        raw=r,
    )


def read_part1(diag_dir: str) -> tuple:
    """Read `_diag/part1/*.json`.  Returns (all_records, complaints).

    `chain_*` records are skipped: they hold `rungs` and their rungs are warm-started
    from each other, so they are not independent optimisations and must not enter a
    seed spread.  Failed records are skipped and reported.
    """
    recs, complaints = [], []
    for p in sorted(glob.glob(os.path.join(diag_dir, "part1", "*.json"))):
        b = os.path.basename(p)
        if b.startswith("chain_"):
            continue
        try:
            r = _load(p)
        except Exception as e:                                   # noqa: BLE001
            complaints.append(f"{b}: does not parse ({e})")
            continue
        if r.get("failed"):
            complaints.append(f"{b}: recorded as a failure")
            continue
        recs.append(part1_state(r, os.path.relpath(p).replace("\\", "/")))
        cx = _energy_cross_check(r)
        if cx:
            complaints.append(cx)
    return recs, complaints


def read_chains(diag_dir: str) -> list:
    """`_diag/part1/chain_*.json` -- the warm-started ladders, kept separate on purpose."""
    out = []
    for p in sorted(glob.glob(os.path.join(diag_dir, "part1", "chain_*.json"))):
        c = _load(p)
        for rung in c.get("rungs", []):
            rec = part1_state(rung, os.path.relpath(p).replace("\\", "/"))
            rec.raw["_chain_direction"] = c.get("direction")
            out.append(rec)
    return out


# --------------------------------------------------------------------------
# the walk-analysis store
# --------------------------------------------------------------------------
def read_v_analysis(path: str) -> list:
    """`_diag/v_analysis.json` -- LL-rotated state measurements.

    These records carry T and V but no E and no cov(T,V), so NO production energy is
    set.  `combine_runs` will refuse them, which is the intended behaviour: the
    structure observables (R_B, S(Q_WC), nbar) are trustworthy here and the energy is
    not, and the schema makes that impossible to forget.
    """
    out = []
    rel = os.path.relpath(path).replace("\\", "/")
    for r in _load(path):
        out.append(StateRecord(
            workflow=LLROT,
            phase=(LIQUID if str(r.get("kind")) == "liquid" else CRYSTAL),
            rs=float(r["rs"]),
            N=N_ELECTRONS,
            kappa=r.get("kappa"),
            nmax=(None if r.get("n_band") is None else int(r["n_band"]) - 1),
            L0=None,
            init_id=str(r.get("tag") or r.get("label") or ""),
            parent_id=None,
            nested=False,
            # deliberately NOT derived from T + kappa*V: without cov(T,V) the error
            # would be invented, and these records are used for structure only.
            production_energy_per_particle=None,
            production_mc_error=None,
            total_energy=None,
            kinetic_total=r.get("T"),
            potential_total=r.get("V"),
            nbar=r.get("nbar"),
            band_occupations=r.get("P"),
            R_B=r.get("R_B"),
            S_QWC=r.get("S_QWC"),
            S_bg=r.get("S_bg"),
            acceptance=r.get("acc"),
            optimization=OptimizationMeta(),
            provenance=Provenance(
                source_path=rel, source_sha256=sha256_of(path),
                legacy_energy_field="(none)", legacy_error_field="(none)",
                legacy_label=str(r.get("tag") or r.get("label") or ""),
                notes="walk-analysis record: T/V only, no energy and no covariance; "
                      "structure observables only"),
            raw=r,
        ))
    return out


# --------------------------------------------------------------------------
# the legacy derived products -- regression targets, not inputs
# --------------------------------------------------------------------------
def read_energy_scan(path: str) -> dict:
    """`_diag/energy_scan.json` verbatim.  The REGRESSION TARGET for the new analysis."""
    return _load(path)


def read_p1_report(path: str) -> list:
    return _load(path)


def read_pn_table(path: str) -> dict:
    return _load(path)


def read_mech_probe(path: str) -> list:
    return _load(path)


def read_reweight(path: str) -> list:
    return _load(path)


def read_partv_summary(path: str) -> dict:
    return _load(path)


# --------------------------------------------------------------------------
# the Gaussian workflow's prototypes
# --------------------------------------------------------------------------
def read_proto_energy(path: str) -> list:
    """`_diag/proto_energy.json` -- the baseline kappa = 2..80 scan.

    Keyed by kappa as a string.  This is the Gaussian workflow's whole energy
    evidence, and it is a DERIVED product: R&F-style columns already reduced.  There
    is no per-seed record behind it, so it cannot yield a `sigma_init` -- a point the
    curve's own caption has to carry.
    """
    d = _load(path)
    rel = os.path.relpath(path).replace("\\", "/")
    h = sha256_of(path)
    out = []
    for kappa_s, v in sorted(d.items(), key=lambda kv: float(kv[0])):
        out.append(StateRecord(
            workflow=GAUSSIAN, phase=CRYSTAL, rs=float(v["r_s"]), N=N_ELECTRONS,
            kappa=float(kappa_s), L0=v.get("L0"), init_id=f"proto_cry_k{kappa_s}",
            production_energy_per_particle=v.get("E_cry"),
            production_mc_error=v.get("s_cry"),
            provenance=Provenance(rel, h, "E_cry", "s_cry", f"proto_cry_k{kappa_s}",
                                  "derived prototype column; no per-seed record"),
            raw=v))
        out.append(StateRecord(
            workflow=GAUSSIAN, phase=LIQUID, rs=float(v["r_s"]), N=N_ELECTRONS,
            kappa=float(kappa_s), init_id=f"proto_liq_k{kappa_s}",
            production_energy_per_particle=v.get("E_liq"),
            production_mc_error=v.get("s_liq"),
            provenance=Provenance(rel, h, "E_liq", "s_liq", f"proto_liq_k{kappa_s}",
                                  "derived prototype column; no per-seed record"),
            raw=v))
    return out


def read_proto_sq(path: str) -> dict:
    """`_diag/proto_sq.json` -- shell-resolved S(q) for the Gaussian prototypes.

    Each entry is a list of [q, S(q), sigma, degeneracy] rows.
    """
    return _load(path)
