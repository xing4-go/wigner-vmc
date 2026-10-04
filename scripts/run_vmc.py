"""One VMC run, from the command line.

    python scripts/run_vmc.py --phase crystal --rs 75 --nmax 1 \
        --budget quick --init-id 0

This script implements NO part of the VMC workflow.  It parses arguments,
resolves them, and hands them to ``wigner_vmc.api``; the Python API, the
``examples/`` scripts, VS Code and a notebook cell all reach the same functions
by the same route.  If physics ever appears in this file, it has been put in the
wrong place -- ``tests/test_api_contract.py`` fails if it does.
"""
from __future__ import annotations

import argparse
import sys

from wigner_vmc import api


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phase", choices=("crystal", "liquid"), default="crystal")
    p.add_argument("--rs", type=float, default=75.0,
                   help="Wigner-Seitz radius; kappa = rs/sqrt(2)")
    p.add_argument("--N", type=int, default=36, help="electron count (square)")
    p.add_argument("--nmax", type=int, default=1,
                   help="Landau-level truncation; len(theta) = 5 + 2*(nmax)*N")
    p.add_argument("--ansatz", choices=("ll_rotation", "ll_rotation_pinned"),
                   default="ll_rotation",
                   help="crystal only.  'll_rotation' optimises the orbitals "
                        "jointly with the Jastrow; 'll_rotation_pinned' holds "
                        "them at the Gaussian-overlap seed and optimises the "
                        "Jastrow alone.  Neither is the historical fig06 "
                        "crystal's ansatz, which this package does not "
                        "implement.")
    p.add_argument("--budget", default="quick",
                   help="a recorded protocol from configs/<name>.yaml "
                        "(smoke | quick | production)")
    p.add_argument("--init-id", type=int, default=0,
                   help="INDEX into the budget's starting-point list. "
                        "Not a random seed.")
    p.add_argument("--rng-seed", type=int, default=0,
                   help="the SR random stream.  Separate from --init-id.")
    p.add_argument("--quiet", action="store_true", help="suppress the summary")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = api.resolve(args)
    result = api.run_vmc(config, verbose=not args.quiet)
    if args.quiet:
        print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
