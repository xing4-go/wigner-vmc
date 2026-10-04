"""A crystal run, then its exact ``nmax`` nesting.

    python scripts/run_nested.py --rs 75 --new-nmax 2 --budget quick --init-id 0

Runs the parent at ``--nmax`` (default 1), nests to ``--new-nmax``, and prints
the energy gain.  The nesting is checked for exactness and the script REFUSES to
continue if it is not exact -- there is no fallback to a fresh start, because a
re-initialised child would make the energy difference unattributable.

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
    p.add_argument("--nmax", type=int, default=1, help="the PARENT truncation")
    p.add_argument("--new-nmax", type=int, default=2,
                   help="the CHILD truncation; must be > --nmax")
    p.add_argument("--budget", default="quick")
    p.add_argument("--init-id", type=int, default=0)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = api.resolve(args)
    verbose = not args.quiet

    parent = api.run_vmc(config, verbose=verbose)
    vmc = api.VMC(N=config.N, rs=config.rs, phase=config.phase,
                  nmax=config.nmax, kappa_mode=config.kappa_mode)
    child = vmc.nest(parent, new_nmax=args.new_nmax, budget=config.budget,
                     verbose=verbose)

    d = child.energy_per_particle - parent.energy_per_particle
    err = (child.error ** 2 + parent.error ** 2) ** 0.5
    n = child.nesting
    print(f"\n  nesting  nmax {n['from_nmax']} -> {n['to_nmax']}   "
          f"head {n['head']:.3e}  tail {n['tail']:.3e}  (both exactly 0)")
    print(f"  deltaE   {d:+.9f} +- {err:.9f} per electron   "
          f"(E_child - E_parent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
