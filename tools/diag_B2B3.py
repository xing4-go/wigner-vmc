"""Blocker B, stages B2 and B3 in one pass over ONE namespace build.

``diag_c0_kappa.py`` and ``diag_incident_pkl_sha.py`` are both self-contained
probes that build the notebook namespace themselves.  The build costs ~17
minutes (cell 76 runs seven overlap integrals, cell 80 is Stage B at 228 s), and
the two measurements share every input, so running them separately pays that cost
twice.

This driver builds the namespace once and runs both, unchanged, against it.  Each
probe still purges the shared crystal-Jastrow cache entry before every call it
makes, so neither can be handed a stale pickle by the other -- and each remains
independently runnable, which is what keeps the two results separately
auditable.

Deliberately NOT parallel: the machine has already hit the low-memory reaper
during this campaign, and a namespace build is ~200 MB resident.  Serial it is.

    python tools/diag_B2B3.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_c0_kappa as probe_c0                                   # noqa: E402
import diag_incident_pkl_sha as probe_pkl                          # noqa: E402
from frozen_guard import FROZEN_CKPT, readonly_frozen              # noqa: E402
from recon_rs75_init import CLEAN, namespace_cells                 # noqa: E402

LOGS = os.path.join(CLEAN, "logs")


def _write(blob, name):
    path = os.path.join(LOGS, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print(f"  wrote {path}")


def main():
    print("#" * 78)
    print("#  Blocker B / B2 + B3 -- one namespace build, both measurements")
    print("#" * 78)
    print(f"#  frozen store (read-only) {FROZEN_CKPT}")
    print(f"#  kappa legacy {probe_c0.K_LEGACY}   "
          f"kappa clean {probe_c0.K_CLEAN:.12f}")
    print("#  building the notebook namespace against a WORKING COPY\n", flush=True)

    with readonly_frozen(FROZEN_CKPT) as work:
        E = namespace_cells(work)

        print("\n" + "=" * 78)
        print("  PART 1 / B2-B3 -- is c0 a function of kappa alone?")
        print("=" * 78, flush=True)
        blob2, rc2 = probe_c0.run(E, work)

        print("\n" + "=" * 78)
        print("  PART 2 -- pkl-byte identity: which kappa wrote the frozen files?")
        print("=" * 78, flush=True)
        blob3, rc3 = probe_pkl.run(E, work)

    _write(blob2, "diag_c0_kappa.json")
    _write(blob3, "diag_incident_pkl_sha.json")
    print("\n  frozen tree: UNCHANGED -- the guard verified it on exit")
    print(f"  exit codes: B2-B3 {rc2}   pkl-byte {rc3}")
    return rc2 or rc3


if __name__ == "__main__":
    raise SystemExit(main())
