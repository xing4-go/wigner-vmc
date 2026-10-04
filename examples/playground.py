"""Interactive playground -- one `# %%` cell per step.

Open this in VS Code and use "Run Cell" above each `# %%`, or open it in
Jupyter.  Each cell is independent once the ones above it have run, so you can
change a number and re-run just that cell.

Nothing here is production.  The default budget is `smoke`, which is fast enough
to iterate on and is NOT converged -- never quote an energy from it.
"""

# %% [markdown]
# # VMC playground
#
# The whole API is four objects:
#
# * `VMC(N, rs, phase, nmax)` -- a system, fixed at construction
# * `.run(init_id, budget)`    -- one run, returns a `RunResult`
# * `.nest(result, new_nmax)`  -- grow the truncation exactly
# * `RunResult`                -- `.energy_per_particle`, `.error`,
#                                 `.acceptance`, `.state`, `.optimization`
#
# Two energy names, and they mean different things:
# `.energy_per_particle` is the production number (per electron);
# `.optimization.E_total` is the SR trace and is a TOTAL, not the answer.

# %%
from wigner_vmc import VMC, run_vmc

# %%
# A crystal at r_s = 75, one Landau level, quick budget.
vmc = VMC(N=36, rs=75, phase="crystal", nmax=1)
vmc.kappa                      # rs/sqrt(2) -- the physical coupling

# %%
# One run. `init_id` indexes the budget's starting-point list (the campaign's
# five fixed Jastrow widths); it is NOT a random seed.
result = vmc.run(init_id=0, budget="smoke")
result.energy_per_particle

# %%
# It really is a total: this is the same number, multiplied by N once.
result.energy_total, result.energy_per_particle * result.N

# %%
# The SR trace, and why it is not the answer.
result.optimization.summary()

# %%
# Where the electrons ended up, and how the weight sits across Landau levels.
result.state.R.shape, result.state.ll_occupation()

# %%
# Structure factor. Needs the production snapshots, which the state keeps.
sq = result.state.structure_factor()
sq["qn"][:5], sq["S"][:5]

# %%
# Nest to nmax = 2: the child CONTAINS the parent exactly, and the API aborts
# if it does not.
child = vmc.nest(result, new_nmax=2, budget="smoke")
child.energy_per_particle - result.energy_per_particle   # delta E, per electron
child.nesting

# %%
# The liquid, same API. No orbitals to nest -- v = 0 by definition.
liquid = VMC(N=36, rs=75, phase="liquid", nmax=1).run(budget="smoke")
liquid.energy_per_particle

# %%
# One-shot form, for scripts that do not want to keep a VMC around.
run_vmc(phase="crystal", rs=75, nmax=1, init_id=0, budget="smoke",
        verbose=False).energy_per_particle
