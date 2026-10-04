"""A Hall-liquid run -- the phase the crystal is compared against.

    python examples/run_liquid.py
    python examples/run_liquid.py --rs 75 --budget production

The liquid has no orbitals to grow: ``v = 0`` is the uniform state by definition,
so only the five Jastrow parameters are free and the optimiser runs over those
alone.  That is why ``nest`` refuses a liquid parent -- there is nothing to nest.

Same API, same route as the crystal.  See ``examples/run_crystal.py``.
"""
from __future__ import annotations

import argparse

from wigner_vmc import VMC


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--budget", default="quick")
    p.add_argument("--rs", type=float, default=75.0)
    p.add_argument("--init-id", type=int, default=0)
    args = p.parse_args()

    vmc = VMC(N=36, rs=args.rs, phase="liquid", nmax=1)
    result = vmc.run(init_id=args.init_id, budget=args.budget)

    print()
    print(f"  energy per particle  {result.energy_per_particle:+.6f} "
          f"+- {result.error:.6f}")
    print(f"  energy total         {result.energy_total:+.6f} "
          f"+- {result.error_total:.6f}   ({result.N} electrons)")
    print(f"  acceptance           {result.acceptance:.4f}")

    if args.budget != "production":
        print()
        print("  NOTE: this budget is NOT converged.  Do not quote this energy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
