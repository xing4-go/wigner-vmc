"""The canonical result store: `wigner_vmc_clean/results/`.

Two layers, and the separation is the point.

    results/legacy_*/states.json    the CONVERTED states -- one entry per legacy record,
                                    carrying the production energy as converted.  This
                                    is what `import_legacy_results.py` writes and it is
                                    a pure reshape of what was already on disk.

    results/legacy_*/scan.json      the ANALYSED scan -- ScanPoints and Brackets, i.e.
                                    the output of `analysis/`.  This is what a figure
                                    reads.  It is derived, so it is written by
                                    `analyze_energy.py`, not by the import step.

`results/manifest.json` records which legacy files each layer was built from, by
sha256, so a figure can always be traced back to the exact bytes behind it -- and so
a rerun over a changed legacy tree is visible rather than silent.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .schema import Bracket, ScanPoint, StateRecord

STORE_SCHEMA = "wigner_vmc_clean/results"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    os.replace(path + ".tmp", path)          # atomic; never a half-written store


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# layer 1 -- converted states
# --------------------------------------------------------------------------
def store_states(results_dir: str, workflow: str, records, sources: list,
                 notes: str = "", filename: str = "states.json") -> str:
    path = os.path.join(results_dir, f"legacy_{workflow}", filename)
    payload = {
        "schema": STORE_SCHEMA + "/states",
        "workflow": workflow,
        "written_utc": _now(),
        "notes": notes,
        "n_records": len(records),
        "sources": sources,
        "records": [r.as_dict(include_raw=True) for r in records],
    }
    write_json(path, payload)
    return path


def load_states(results_dir: str, workflow: str,
                filename: str = "states.json") -> list:
    """Rebuild StateRecord objects from the store, for the analysis layer."""
    payload = read_json(os.path.join(results_dir, f"legacy_{workflow}", filename))
    out = []
    for d in payload["records"]:
        d = dict(d)
        opt = d.pop("optimization", {}) or {}
        prov = d.pop("provenance", None)
        from .schema import OptimizationMeta, Provenance
        out.append(StateRecord(
            optimization=OptimizationMeta(**opt),
            provenance=Provenance(**prov) if prov else None,
            **d))
    return out


# --------------------------------------------------------------------------
# layer 2 -- the analysed scan
# --------------------------------------------------------------------------
def store_scan(results_dir: str, workflow: str, points, brackets, meta: dict) -> str:
    path = os.path.join(results_dir, f"legacy_{workflow}", "scan.json")
    write_json(path, {
        "schema": STORE_SCHEMA + "/scan",
        "workflow": workflow,
        "written_utc": _now(),
        "meta": meta,
        "points": [p.as_dict() for p in points],
        "brackets": [b.as_dict() for b in brackets],
    })
    return path


def load_scan(results_dir: str, workflow: str) -> tuple:
    payload = read_json(os.path.join(results_dir, f"legacy_{workflow}", "scan.json"))
    pts = [ScanPoint(**{k: v for k, v in p.items()}) for p in payload["points"]]
    brs = [Bracket(**{k: v for k, v in b.items()}) for b in payload["brackets"]]
    return pts, brs, payload["meta"]


# --------------------------------------------------------------------------
# the store manifest
# --------------------------------------------------------------------------
def load_store_manifest(results_dir: str) -> dict:
    p = os.path.join(results_dir, "manifest.json")
    return read_json(p) if os.path.exists(p) else {"schema": STORE_SCHEMA + "/manifest",
                                                   "entries": []}


def update_store_manifest(results_dir: str, entry: dict) -> dict:
    """Add or replace one entry, keyed by its `artifact` name."""
    m = load_store_manifest(results_dir)
    m["written_utc"] = _now()
    m.setdefault("entries", [])
    m["entries"] = [e for e in m["entries"] if e.get("artifact") != entry["artifact"]]
    m["entries"].append(entry)
    m["entries"].sort(key=lambda e: e["artifact"])
    write_json(os.path.join(results_dir, "manifest.json"), m)
    return m
