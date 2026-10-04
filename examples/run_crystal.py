"""A Wigner-crystal run, meant to be read and pressed Play on.

Run it in VS Code with the Play button, or:

    python examples/run_crystal.py
    python examples/run_crystal.py --budget production

It calls the same ``VMC`` the CLI calls.  There is nothing to install here and no
path to set up -- if `import wigner_vmc` works, this file works.
"""
from __future__ import annotations

import argparse

from wigner_vmc import VMC


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--budget", default="quick",
                   help="smoke | quick | production (see configs/)")
    p.add_argument("--rs", type=float, default=75.0)
    p.add_argument("--nmax", type=int, default=1)
    p.add_argument("--init-id", type=int, default=0)
    args = p.parse_args()

    # `init_id` picks a starting point from the budget's own list -- here the
    # campaign's five fixed Jastrow widths.  It is NOT a random seed.
    vmc = VMC(N=36, rs=args.rs, phase="crystal", nmax=args.nmax)
    result = vmc.run(init_id=args.init_id, budget=args.budget)

    print()
    print(f"  energy per particle  {result.energy_per_particle:+.6f} "
          f"+- {result.error:.6f}")
    print(f"  energy total         {result.energy_total:+.6f} "
          f"+- {result.error_total:.6f}   ({result.N} electrons)")
    print(f"  acceptance           {result.acceptance:.4f}")

    occ = result.state.ll_occupation()
    print(f"  band weights P_n     {[f'{p:.4f}' for p in occ['P']]}")
    print(f"  <n>                  {occ['nbar']:.4f}")

    if args.budget != "production":
        print()
        print("  NOTE: this budget is NOT converged.  Do not quote this energy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
