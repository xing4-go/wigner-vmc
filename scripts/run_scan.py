"""Scan the budget's starting points at one coupling, and nest each.

    python scripts/run_scan.py --rs 75 --budget smoke --new-nmax 2

One row per ``init_id``: the parent energy, the nested energy, and their
difference.  This is the shape of the campaign's own scan -- several starting
points at a fixed ``r_s``, each nested -- but it is a DRIVER, not a result: at
``smoke`` or ``quick`` it is not converged and nothing it prints may be quoted.

A parent that does not produce a finite energy is reported as
``MISSING PARENT`` and the process exits non-zero.  There is deliberately no
silent fallback: skipping a failed parent and averaging the rest would turn a
crash into a physics number.

Implements no physics; see ``scripts/run_vmc.py``.
"""
from __future__ import annotations

import argparse
import sys

from wigner_vmc import api


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rs", type=float, default=75.0)
    p.add_argument("--N", type=int, default=36)
    p.add_argument("--nmax", type=int, default=1)
    p.add_argument("--new-nmax", type=int, default=2)
    p.add_argument("--init-ids", type=int, nargs="+", default=None,
                   help="default: every init_id the budget defines")
    p.add_argument("--budget", default="smoke")
    p.add_argument("--rng-seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = api.resolve(args)
    budget = api.load_budget(config.budget)
    ids = (args.init_ids if args.init_ids is not None
           else [int(r["init_id"]) for r in budget.inits])

    print(f"  scan r_s={config.rs:g}  budget {budget.name}  "
          f"nmax {config.nmax} -> {args.new_nmax}  over {len(ids)} starting points")
    print(f"  source: {budget.provenance}")
    print()
    print(f"  {'init_id':>7}  {'L0':>5}  {'E/N parent':>14}  {'+-':>9}  "
          f"{'E/N nested':>14}  {'+-':>9}  {'deltaE':>12}")

    rows, missing = [], []
    for i in ids:
        vmc = api.VMC(N=config.N, rs=config.rs, phase=config.phase,
                      nmax=config.nmax, kappa_mode=config.kappa_mode)
        try:
            parent = vmc.run(init_id=i, budget=config.budget, verbose=False)
            if parent.energy_per_particle != parent.energy_per_particle:
                raise ValueError("parent energy is NaN")
            child = vmc.nest(parent, new_nmax=args.new_nmax,
                             budget=config.budget, verbose=False)
        except (api.NestingIdentityError, RuntimeError, ValueError) as exc:
            missing.append((i, exc))
            print(f"  {i:>7}  {budget.width_for(i):>5g}  MISSING PARENT   "
                  f"({type(exc).__name__}: {exc})")
            continue
        d = child.energy_per_particle - parent.energy_per_particle
        rows.append((i, parent.energy_per_particle, child.energy_per_particle, d))
        print(f"  {i:>7}  {budget.width_for(i):>5g}  "
              f"{parent.energy_per_particle:>+14.6f}  {parent.error:>9.6f}  "
              f"{child.energy_per_particle:>+14.6f}  {child.error:>9.6f}  "
              f"{d:>+12.6f}")

    print()
    if missing:
        print(f"  {len(missing)} of {len(ids)} parents did not complete: "
              f"{[i for i, _ in missing]}")
        print("  MISSING PARENT -- refusing to summarise over an incomplete set.")
        return 1
    if rows:
        ds = [r[3] for r in rows]
        print(f"  deltaE over {len(ds)} starting points: "
              f"mean {sum(ds) / len(ds):+.6f}   min {min(ds):+.6f}   "
              f"max {max(ds):+.6f}")
        print(f"  spread {max(ds) - min(ds):.6f} -- this is the seed-to-seed "
              f"scatter, and it is why a single point is not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
