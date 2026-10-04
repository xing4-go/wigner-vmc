"""The clean Wigner-crystal / Hall-liquid VMC package.

Two layers live here, and the boundary between them is the point of the design:

* ``physics/``, ``wavefunctions/``, ``vmc/`` -- the VMC core: geometry and the
  magnetic cell, the Ewald Coulomb Hamiltonian, the Jastrow and the two
  wavefunctions (Gaussian and Landau-level-rotated), the Metropolis sampler, the
  local energy and the production estimators, and stochastic reconfiguration.
  These are ports of the frozen legacy engine, verified against it by the B1-B7
  regressions -- see ``PHYSICS_BEHAVIOR.md`` for the invariant index.
* ``analysis/``, ``io/``, ``figures/`` -- the Stage 1 layer: figures are rebuilt
  from stored results, never by re-running VMC.

The rule this package enforces: **figure code never runs VMC.**

    frozen legacy result
          |  io/legacy.py          conversion only, never recomputation
          v
    results/<workflow>/states.json
          |  analysis/              the ONE implementation of every sigma and delta_E
          v
    results/<workflow>/scan.json
          |  figures/               plotting only
          v
    figures/<name>.png

The public surface is the API in ``api.py``::

    from wigner_vmc import VMC, run_vmc

    vmc = VMC(N=36, rs=75, phase="crystal", nmax=1)
    result = vmc.run(init_id=0, budget="quick")
    child = vmc.nest(result, new_nmax=2)

Everything under ``physics/``, ``wavefunctions/`` and ``vmc/`` is the verified
core that API drives; everything above it (the CLI, ``examples/``, the VS Code
launch configurations) calls down into ``api`` and implements no workflow of its
own.  See ``USER_GUIDE.md``.
"""
from .api import (BUDGET_ALIASES, Budget, Lattice, NestingIdentityError,  # noqa: F401
                  Optimization, RunConfig, RunResult, RunState, VMC,
                  build_lattice, load_budget, resolve, resolve_budget_name,
                  run_vmc, theta, theta_parts)

__version__ = "0.1.0"

__all__ = [
    "VMC", "run_vmc", "RunResult", "RunState", "Optimization", "RunConfig",
    "Budget", "Lattice", "NestingIdentityError",
    "build_lattice", "load_budget", "resolve_budget_name", "BUDGET_ALIASES",
    "resolve", "theta", "theta_parts",
    "__version__",
]
