# Figure construction recipes

Executable research recipes that use the validated public `wigner_vmc` API to
perform calculations and construct scientific figures.

This directory is the **user workflow layer**.  It is not part of the physics
engine, and nothing here is imported by `src/`.  The layering is:

| where | what belongs there |
|---|---|
| `src/wigner_vmc/` | the validated physics / VMC implementation |
| `examples/` | user workflows -- executable recipes that *use* the API |
| `results/` | generated numerical data (`.npz`, `.json`) |
| `figures/` | rendered scientific figures (`.png`) |

**No generated `.npz`, `.json`, PNG, checkpoint or large numerical data lives
inside `examples/`.**  A recipe writes its numbers to `results/` and its
pictures to `figures/`, both under the same relative path as the recipe itself.

## Running a recipe

Install the package once, from the repository root:

```powershell
pip install -e .
```

Then, from the repository root, one command performs the whole workflow --
calculation, observables, numbers, figures:

```powershell
python examples/figure_construction/structure_slice.py --budget quick
python examples/figure_construction/structure_slice.py --budget reproduction
```

No `sys.path` manipulation is needed or wanted; the recipes import `wigner_vmc`
as an installed package.

### Choosing the physics

Three physical parameters are yours to set from the command line; nothing needs
to be edited in the source.

| flag | meaning | default |
|---|---|---|
| `--liquid-rs RS` | Wigner-Seitz radius of the **liquid**, in units of `l_B` | `5.656854249` (`4*sqrt(2)`, the historical fig06 liquid) |
| `--crystal-rs RS` | Wigner-Seitz radius of the **crystal** | `90.509667992` (`64*sqrt(2)`, the historical fig06 crystal) |
| `--crystal-nmax N` | highest Landau level the **crystal** basis keeps: `n = 0 ... N`, so `n_bands = N + 1` | `1` |

The coupling is always the physical one, `kappa = r_s / sqrt(2)`, and it is
printed before anything is computed.  The Stage-2E rounded-`kappa` regression is
never used; the two coincide only at the historical couplings (`kappa = 4` and
`64`) and give different energies everywhere else.

There is deliberately **no `--liquid-nmax`**.  The liquid is a filled lowest
Landau level with a Jastrow factor, fixed: its determinant rows are
`sum_n C[k,n] phi_{k,n}` with `v = 0`, and `v = 0` makes `C` select `phi_{k,0}`
alone, so every higher band enters multiplied by zero.  The shared parameter
vector reserves room for a rotation the liquid never uses, and that padding is
reported as a *technical* field, never as a liquid truncation.  Measured: a
liquid run at the padding value `1` and at `3` returns the same energy to the
last bit.

Changing any of these changes the calculation.  `--crystal-nmax 2` grows the LL
basis, the Bloch sum and the rotation-amplitude block
(`len(theta)`: 77 -> 149 -> 221 for `n_max` 1, 2, 3); it is not a relabelling.

Every run prints a `RESOLVED CALCULATION` block first -- both phases, in that
phase's own terms, with the couplings, the LL basis, which optimiser sectors are
enabled, and the budget -- so the request can be checked before the SR ladder
is paid for.  Invalid input (`r_s <= 0`, non-finite, `--crystal-nmax < 1`) is
refused there, with the reason, and exits before any VMC.

### `--budget`

A budget is a named, recorded protocol in `configs/<name>.yaml`.  `load_budget`
has no fallback table, so an unknown name is an error naming the ones that
exist rather than a silent default.

* **`quick`** -- a smoke test.  It proves the whole chain works end to end
  (clean VMC -> samples -> `S(q)`, `g(r)` -> figures) in minutes.
  Its figures are stamped `QUICK -- not publication quality` and its metadata
  says so.  **Do not compare a quick curve against a production curve and call
  that a reproduction.**
* **`reproduction`**, also spelled **`full`** -- the actual scientific
  reproduction, at the protocol the historical figure was produced with.
  `full` is what to type when you mean "the real statistics, not the smoke
  test".  It is a *name*, not a second protocol: `resolve_budget_name` maps it
  onto `reproduction` before anything reads it, so the two spellings are ONE
  calculation with one output directory rather than two directories holding the
  same run.  See `configs/reproduction.yaml`, which traces every number to the
  frozen notebook and lists the deviations.

  **The crystal panel needs the truncation opened.**  At the historical
  couplings the crystal's first-shell `S(q)` reads 7.2817 at `--crystal-nmax 1`,
  11.5806 at `2` and 12.1661 at `3`, against the historical ensemble's 11.9154:
  the gap closes with the Landau-level basis, and the `n_max = 1` shortfall that
  this file used to call a located divergence is a truncation effect.  9% of
  protocol spread remains at `n_max = 3`, and the historical crystal is a
  wavefunction *family* this package still does not implement -- see
  [`DIVERGENCE_fig06_crystal.md`](DIVERGENCE_fig06_crystal.md).  The liquid
  reproduces throughout -- its `S(q)` differs from the historical ensemble by
  0.21 at worst.

Any other file in `configs/` also works; those two are the documented pair.

### `--ansatz`

```powershell
python examples/figure_construction/structure_slice.py --budget reproduction
python examples/figure_construction/structure_slice.py --budget reproduction --ansatz ll_rotation_pinned
python examples/figure_construction/structure_slice.py --budget reproduction --crystal-nmax 2
```

Both crystal ansatz choices build the **same** wavefunction class.  They differ
in which parameters the optimiser is allowed to move.

| `--ansatz` | what moves |
|---|---|
| `ll_rotation` (default) | the orbital parameters **and** the 5 Jastrow parameters, jointly |
| `ll_rotation_pinned` | the 5 Jastrow parameters only; the orbitals stay at the Gaussian-overlap seed `L0` |

**Neither of these is the historical fig06 crystal.**  That state is a
determinant of site-centred magnetic Gaussians (notebook cell 18's
`crystal_wavefunction`, built from `GaussianBasis.orbitals`), a different
wavefunction *family* that this package does not implement.  `ll_rotation_pinned`
pins the seed inside the LL-rotated basis, which is **not** the same object:
measured on the pre-fix tree, it moves `S(|g1|)` from 1.94 to 2.22, where the
historical ensemble reads 11.92.  There is therefore no `--ansatz` value that
*is* the historical crystal, and the recipe offers none -- see
[`DIVERGENCE_fig06_crystal.md`](DIVERGENCE_fig06_crystal.md) for the measurements.
`--crystal-nmax` is the knob that closes the gap between the two families;
`--ansatz` is not.  `run_metadata.json` records the caveat as well.

The liquid has no orbitals to pin, so `--ansatz` is a crystal-only choice and the
API rejects the combination rather than ignoring it.

### Where a run writes

`results/figure_construction/structure_slice/` and
`figures/figure_construction/structure_slice/`
hold exactly one calculation: the historical request, i.e. the two default
couplings, the default ansatz, `--crystal-nmax 1` and `--budget reproduction`.
Anything else -- a different coupling, a different truncation, the pinned
ansatz, the `quick` budget -- is a different calculation and writes to a
suffixed sibling instead:

```
structure_slice_liquid_rs75__crystal_rs75__llrot_nmax2__quick
```

The suffix names every input that changes the numbers, so two requests can be
compared side by side instead of the second erasing the first.  The full
precision values live in `run_metadata.json` (`request_slug`, `liquid_rs`,
`crystal_rs`, `crystal_nmax`, ...); the directory name is only a readable
grouping.  A run that would overwrite a directory holding a *different* request
says so first.

`run_metadata.json` records the resolved request in each phase's own terms:

```jsonc
"liquid":  {"rs": ..., "kappa": ..., "orbital_ansatz": "filled_lll_fixed",
            "orbital_sr": false, "jastrow_sr": true, "ll_indices": [0]},
"crystal": {"rs": ..., "kappa": ..., "orbital_ansatz": "ll_rotation",
            "nmax": 2, "n_bands": 3, "ll_indices": [0, 1, 2],
            "orbital_sr": true, "jastrow_sr": true}
```

plus the budget, the RNG seeds, the optimisation and production protocols, and
a source identifier.

### `--resume`

Reuses a run this recipe already saved under `results/figure_construction/`.
It will only reuse an artifact whose recorded configuration digest matches the
current request exactly (phase, `r_s`, `N`, `nmax`, `ansatz`, `init_id`,
budget, schema).  A mismatch is treated as "nothing to resume" and the run is
redone rather than silently reused -- a figure carrying one state's label on
another state's data would be worse than paying for the run again.

Resume never inspects the frozen legacy store, `_diag/ckpt_rebuild`, or any
production state.  It cannot: it only looks under this recipe's own results
directory.

### `--compare`

Off by default.  After the clean numbers exist and are saved, it also prints a
side-by-side comparison against the historical ensembles, re-derived from the
*frozen configurations* through the same clean estimators.  This is the only
place the historical tree is read, it is read-only, and no historical number is
ever an input to a figure.

### Plot labels

The legends name the coupling the user actually chose -- `liquid, r_s = 75` and
`crystal, r_s = 75, n_max = 2` -- not `kappa` alone, because `kappa` is an
internal conversion and two different requests can share a rounded one.  The
axes stay in units of `l_B` (`q*l_B`, `r/l_B`, `x/l_B`, `y/l_B`).

### The same choices from `scripts/run_vmc.py`

The generic driver takes the same physical parameters, and describes the
calculation in the same terms:

```powershell
python scripts/run_vmc.py --phase crystal --rs 75 --ansatz ll_rotation --nmax 2 --budget quick
python scripts/run_vmc.py --phase liquid  --rs 75 --budget quick
```

For the liquid it prints `orbital ansatz: filled LLL (fixed)`, `orbital SR:
disabled`, `LL rotation: none` -- never a liquid `nmax`.

## Structural diagnostic at a single point

`structure.py` answers a different question from `structure_slice.py`:
not "does this reproduce the historical figure", but "what does the structure look
like at *this* coupling, measured now".  It is independent of the fig06 recipe --
neither imports the other, they share no output directory, and this one never reads
a historical number at all.

```powershell
python examples/figure_construction/structure.py `
    --liquid-rs 55 --crystal-rs 55 --crystal-nmax 2 --budget quick
```

**One invocation is one point.**  The inputs are

| flag | meaning | default |
|---|---|---|
| `--liquid-rs RS` | coupling of the **liquid** | `55` |
| `--crystal-rs RS` | coupling of the **crystal** | `55` |
| `--crystal-nmax N` | highest Landau level the **crystal** keeps: `n = 0 ... N`, so `n_bands = N + 1` | `2` |
| `--budget quick\|full\|reproduction` | named protocol from `configs/`; `full` is an alias for `reproduction` | `quick` |

and there is no way to ask for a sweep.  The two couplings may differ; the liquid
is always a **fixed filled lowest Landau level with a Jastrow factor** (orbital SR
disabled, Jastrow SR enabled), the same object `structure_slice.py` calls its
liquid, and again there is deliberately no `--liquid-nmax` -- its shared parameter
vector reserves padding for a rotation it never uses.  The crystal is the
LL-rotation ansatz at the truncation you choose.

Three observables are measured from a **new clean run** and drawn as two panels,
`liquid | crystal`:

| figure | observable | data |
|---|---|---|
| `density_xy.png` | smoothed real-space density `n(x,y)` | `density_xy.npz` |
| `pair_correlation_xy.png` | displacement-resolved `g(x,y)` | `pair_correlation_xy.npz` |
| `structure_factor_xy.png` | `S(qx,qy) = <|rho_q|^2>/ne` | `structure_factor_xy.npz` |
| `combined_structure_summary.png` | the same six panels in one 2x3 grid | -- |

All four observables are calls into `wigner_vmc.analysis.structure`; the recipe
contains no estimator of its own.  Both panels of each figure share one spatial
extent and one coordinate convention, so a difference between them is a
difference between the states.  `g(x,y)` and `S(qx,qy)` also share a single
colour scale across their two panels; the density figure gives each panel its own
(the historical convention) because the liquid's modulation is a few per cent of
its mean where the crystal's can be many times it, and one shared bar would
render the liquid control as a blank square.  The choice, and the scale actually
used, is in `run_metadata.json` and on the figure itself.  That file also records
the grids, kernels, windows, normalisations and the handling of `q = 0`, and the
`.npz` files hold the raw arrays (the `S(q)` array is stored raw and unmasked --
the `q -> 0` disc is a *display* mask, saved beside it).

### Where a run writes

`results/figure_construction/structure/<slug>/` and
`figures/figure_construction/structure/<slug>/`, where the slug names every input
that changes the numbers, budget included:

```
liquid_rs55__crystal_rs55__llrot_nmax2__quick
```

The budget is part of the name on purpose.  A `quick` run is a smoke test and its
figures are stamped `QUICK / not publication quality`; letting it land on top of a
`reproduction` delivery is exactly the accident the suffix prevents.  The
underlying `liquid/run.npz`, `liquid/run.json`, `crystal/run.npz` and
`crystal/run.json` are saved next to the observables, since a PNG is not a
scientific record.

The title says whether the two couplings are the same -- `same-coupling comparison`
or `different-coupling comparison` -- so a reader cannot mistake a
liquid-at-45-against-crystal-at-65 figure for a phase comparison at one coupling.

### What this recipe will not do

It will not reconstruct the coupling ladder, and it will not draw the Gaussian
crystal.  If the LL-rotation crystal's structural order looks weaker than the
historical Gaussian crystal's, note that the difference is truncation-controlled
and closes as `--crystal-nmax` grows: see
[`DIVERGENCE_fig06_crystal.md`](DIVERGENCE_fig06_crystal.md).

## Phase competition

`phase_competition.py` rebuilds `figures/vmc_energy_phase_competition.png` -- the
absolute `E/N` ladder of the two phases, and the `delta_E = E_crystal - E_liquid`
competition magnified -- from clean VMC runs, at the point selection of the
historical figure extended around it.

```powershell
python examples/figure_construction/phase_competition.py --budget quick
python examples/figure_construction/phase_competition.py --budget full
```

At every budget other than `reproduction` the figure carries a
`QUICK / not publication quality` stamp, drawn inside **both** panels, and
`run_metadata.json` records the same fact under `figure_stamp`.  On the picture
and not only beside it: a smoke test whose render is visually identical to the
real figure is the one outcome this recipe must not produce.  Only
`reproduction` is unstamped, so an unrecognised budget marks rather than
silently passing as a result.

It **recomputes every energy difference**.  The historical `delta_E` points came
from Landau-level-rotation crystal records produced with the reversed complex
conjugation at `vmc/sr.py:219`; the defect makes the crystal's first shell
uniaxial and moves its energy, so a competition read off those states is a
competition against a state the ansatz does not describe.  The recipe therefore
contains no path to the legacy crystal JSON or to the stores derived from it,
and `tests/test_phase_competition_contract.py` checks that structurally -- it
parses the source and asserts those paths appear nowhere in the *code*, so the
docstring can still explain the rule it is obeying.

The `(r_s, tier)` grid is the historical one **extended around it** -- denser
where the competition is decided, wider at both ends:

| tier | `n_max` | couplings |
|---|---|---|
| `conv` | 1 | 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 60, 65, 70, 75, 77.5, 80, 85, 90 |
| `nest` | 2 | 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 60, 65, 70, 75, 77.5, 80, 82.5, 85, 87.5, 90 |
| `nest4` | 3 | 70, 75, 77.5, 80, 82.5, 85, 87.5, 90 |

The historical selection was `conv` 40-90 on the 5-unit ladder, `nest` 55-90 and
`nest4` 75, 80.  Every historical coupling is still measured at the same
truncation -- this is an extension, not a replacement -- and the original grid is
kept verbatim in `run_metadata.json` under `point_selection.historical_tiers`
beside the current one, so the difference is a recorded fact rather than
something a reader has to reconstruct.  What was added, and why:

* **30 to 55, every 2.5, for `n_max` = 1 and 2.**  This is the window the sign
  of `delta_E` is decided in, and the historical grid sampled it at 5-unit
  spacing or not at all below 40.
* **Below 30, `n_max` = 1 only.**  At small `r_s` the converged truncation is
  the one that runs; the nested tiers are not measured where they are still
  converging.
* **Above 70, `n_max` = 2 and 3 additionally.**  That is where the nested rungs
  separate and where the far side of the crossing is.

The `delta_E` panel's x-range is **derived from that grid**, not written down:
`de_window()` returns one unit beyond the widest tier on each side, which gives
`(24, 91)` here.

This is not a bug fix.  The two literals it replaced -- 44 and 91 -- were a
deliberate editorial choice, recorded in the source at `ll_rotation.py:72`:
`conv` at r_s = 40 and 42.5 were excluded on purpose because the panel's subject
is the crossing and every coupling that carries it lies at 45 or above.  The
delivered figure therefore measured 13 `conv` points, recorded all 13 in
`run_metadata.json`, and drew 11 -- intentionally, and correctly for the figure
it was.  What was wrong was the *form*: a second copy of somebody else's scan,
living inside a shared renderer.  A literal that is right for one grid is
silently wrong for the next, and the failure is not a crash -- the extra
couplings simply fall outside the frame while the axes still look complete.
Under the extended grid those literals would have dropped the entire small-r_s
end that the extension exists to show.

**Expect the y-axis to stretch accordingly.**  Excluding 40 and 42.5 was part of
why the historical panel was readable; including 25-42.5 puts points with a much
larger `|delta_E|` on the same linear axis, and the crossing region may compress.
If it does, the fix is a zoomed inset or a split axis -- not putting the
excluded points back outside the window.

`run_metadata.json` records the window under `point_selection.delta_e_window`,
and `tests/test_phase_competition_contract.py` pins both the historical drop and
the property that matters now: after the dodge, no measured coupling is dropped
by the panel that measures it.

The ladder runs over the historical 10 couplings `kappa = 2, 4, 8, 16, 24, 32,
40, 48, 64, 80` -- unchanged.  **The ladder is not the same state as the
historical ladder**: that one scanned the *Gaussian* crystal, which this package
does not implement.  The rebuilt ladder is `ll_rotation` on the same couplings,
and the metadata says so.  The comparison phase competition needs is
liquid-versus-crystal *within* one recipe, which is what the lower panel is --
see [`DIVERGENCE_fig06_crystal.md`](DIVERGENCE_fig06_crystal.md).

Each point is cached in `results/figure_construction/phase_competition/<slug>/runs.json`
as it finishes, so an interrupted campaign resumes rather than restarts
(`--redo` forces recomputation).  The historical ladder took a minimum over a
grid of Gaussian widths; `--l0-grid` reproduces that selection rule over the
budget's own width grid, at a cost multiplied by the grid size.  One seed per
point is the default.

### `--construction`: what the `nested` labels mean

The legend draws `nested n_max = 2` and `nested n_max = 3`.  In the **first**
delivered version of this figure that wording was **false**: every crystal point
went through a fresh Gaussian-overlap seed optimised independently at its own
`n_max`, so the three truncations were three cold starts.  `VMC.nest` -- the real
embedding `theta_opt^(m) -> pad_v -> theta_0^(m+1)` -- existed and was correct,
but nothing under `examples/` or `figures/` called it.  That matters because the
one-sidedness argument in `wavefunctions/nesting.py`, the reason an `n_max`
ladder is interpretable at all, rests on there being no initialisation
difference between rungs.  A gap in the test suite let it through:
`test_the_crystal_nmax_matches_the_tier_name` pins the *number* in the tier name
and nothing pinned the *construction*.

```powershell
# the delivered construction -- the baseline, and the default.  Unchanged slug.
python examples/figure_construction/phase_competition.py --budget quick

# the ladder the legend always claimed: nmax 1 -> pad -> 2 -> pad -> 3
python examples/figure_construction/phase_competition.py --budget quick --construction nested
```

The two write to **different directories** -- `<slug>` and `<slug>__nested` -- so
the nested family is a sibling of the delivered one and the 93-point
independent store is read as the baseline, never archived, overwritten or
extended.  A nested run also keeps each rung's final variational parameters under
`states/`, because a child has to grow from the parent's *own* finished run
rather than from a recomputation of it, which would be a second trajectory.

Each nested point's record carries its full ancestry -- `parent_nmax`,
`parent_rs`, `parent_init_id`, `parent_key`, `parent_state_digest` (an exact
digest of the parent's `c` and `v` bytes), the propagated `root_L0`, and the
exact nesting residuals.  The embedding is verified twice: the residual on the
vector actually about to be optimised must be exactly zero, and the child's
start must equal the parent's finish *on the same configurations*, which is what
`Psi_child^(0) = Psi_parent^opt (+) 0` asserts.  **An energy that rises with
`n_max` is not a nesting failure** -- exact nesting fixes the child's start, not
where its finite stochastic SR stops -- so it is recorded as `z_vs_parent` and
printed as an optimizer warning, and no test anywhere asserts monotonicity.

Under `--construction nested` the recipe ends by printing
`delta_E_init(rs) = E_nested - E_independent` per rung, plus each family's
crossing of `delta_E`, and records them in `run_metadata.json`.  That difference
is the point of the exercise: if it is small, a corrected Gaussian seed reaches
the same basin as a grown state and initialisation is not what separates the
`n_max = 2` crystal from the published curve; if it is not, finite-SR basin
dependence is.

One environment trap, because it is invisible in the source: the same crystal
point evaluates to `-37.793807191316347` with BLAS threads pinned to 1 and
`-37.793807191311664` without, a 4.7e-12 difference from reduction order alone.
The `n_max = 1` roots are the same calculation in both families and are compared
as a control, so an existing store is only a bit-level baseline under the
threading it was made with.  Set `OMP/OPENBLAS/MKL_NUM_THREADS=1` to reproduce it
exactly -- on this machine that is also the faster setting.  The banner and
`run_metadata.json` record which was used, and the control is a tolerance so an
unpinned run reports a threading artefact instead of a false leak.

## Magnetoplasmon comparison

```
python examples/figure_construction/magnetoplasmon.py
```

Reads the `S(q)` that `structure_slice.py` has **already saved** and draws the
single-mode-approximation dispersion `hbar^2 q^2 / (2 m S(q))` against the
classical magnetoplasmon `omega_mp^2 = omega_c^2 + (2 pi n e^2/m) q`, on a
`q/sqrt(n)` axis.

**This recipe post-processes only.**  It does not start VMC, does not run the SR
optimiser, does not regenerate `S(q)`, and does not estimate the transition.  A
test parses its source and asserts that the machinery to do any of that appears
nowhere in it, so the promise is structural.  If a requested `S(q)` does not
exist it says so, names the exact `structure_slice.py` command that would create
it, and stops -- it never launches one on its own.

### The four curves

| `r_s` | state | where the numbers come from |
|---|---|---|
| 0 | filled LLL | analytic: `S_0(q) = 1 - exp(-q^2 l_B^2/2)`, exact at any `N` |
| 5 | liquid | saved `structure_slice` run |
| 30 | liquid | saved `structure_slice` run |
| 60 | crystal | saved `structure_slice` run |

`r_s = 0` is the exception to the reading rule and is deliberate: with the
interaction off the ground state is the filled lowest Landau level and the
structure factor is a closed form, so it comes from
`analysis.structure.exact_lll_sq` -- the package's own helper, so there is still
one definition of that curve.  Every other curve is the stored array, element
for element; nothing is smoothed, fitted, resampled or interpolated.

### `S(q-vector) -> S(q)`: the two reductions

`structure_slice` saves `S` on the supercell's **allowed momenta**
`q = m G1 + n G2`, which form a two-dimensional disc, not a 1-D path: 294
momenta on this torus, falling into 30 shells of 6, 12 or 18 equal-`|q|`
members.  Dividing each of the 294 by hand and plotting against `|q|` is wrong
in a different way for each phase, and the paper's Fig. 4 caption says which
reduction each one needs -- *"The data are rotation averaged in the liquid phase
and taken along the x axis in the crystal phase."*

**Liquid: rotation average.**  A liquid is rotationally invariant, so the
physical 1-D structure factor is `Sbar_L(q) = <S_L(q-vector)>` over the momenta
with `|q-vector| = q`, and *then* `Omega_L(q) = q^2/(2 Sbar_L(q))`.  The order is
not a detail: averaging `S` first is what rotational invariance means, whereas
`<q^2/(2S)>` averages the reciprocal and is a different number wherever a shell
carries directional noise.  It is also a *biased* difference, not noise --
`<1/S> >= 1/<S>`, so the wrong order always gives the larger `Omega`.  Both are
computed and the run prints the gap between them.

**Crystal: one direction, no averaging.**  A Wigner crystal is *not* isotropic,
so the only honest 1-D curve is a cut, and a shell mean would splice distinct
propagation directions together and destroy the Bragg peak that *is* the crystal
curve.

**Where the paper's `x` axis is.**  This is a mapping, not a convention.  The
paper's own lattice vectors (`QuantumHallVMC`, `A1 = [sqrt(3)/2, -1/2] a`) sit
at `-30` degrees; this package's `Torus` puts `A1` along `(1, 0)`.  The frames
differ by a 30 degree rotation, so the paper's `x = (1, 0)` is the direction of
`G1 + G2` here -- a **reciprocal-lattice (Bragg)** direction.  That is the check
that it is right rather than merely plausible: a cut along a Bragg direction
passes through `Q_WC`, and nothing else does.  The torus's own first Cartesian
axis is 30 degrees away and is a Gamma-K direction whose momentum ladder never
reaches `Q_WC`.  `--crystal-direction` selects either; the default is the
paper's.  Both reductions are per-phase dispatch in one place, `reduce_curve`,
so the two cannot be swapped by accident -- a mutation that did swap them is
caught by `TestTheReductionIsWiredIntoTheCurves`.

### Units

Everything is in `hbar = l_B = omega_c = 1`, the engine's convention
(`physics/hamiltonian.py`).  Two conversions matter and neither is a tunable
number:

* **Dispersion.**  `Omega/(hbar omega_c) = q^2/(2 S(q))` with `q` in `1/l_B`,
  because `hbar^2/(m l_B^2) = hbar omega_c` exactly.  No prefactor.
* **Classical curve.**  At `nu = 1`, `n = 1/(2 pi l_B^2)`, so
  `omega_mp^2/omega_c^2 = 1 + kappa q` -- where `kappa` *is*
  `e^2/(4 pi eps0 l_B hbar omega_c)` per the engine's own definition, taken from
  `kappa_from_rs`.  There is no scale factor chosen to make the curves meet.
* **Axis.**  `x = q/sqrt(n) = q l_B sqrt(2 pi/nu)`, built from the density rather
  than typed as `2.5066`.

Because the two agree only through their leading small-`q` behaviour, the
comparison is a statement about `q -> 0`, and the smallest momentum a finite
torus can carry is its own `|G_1|`.  The run prints `q_min` beside every
relative difference for that reason: a discrepancy quoted without the `q` it was
taken at says nothing.

### The small-`q` table, and what its last columns mean

The table reports, per curve, `q_min`, `S(q_min)`, `Omega_SMA(q_min)`,
`omega_mp(q_min)` and the relative difference between them -- together with *how*
the point at `q_min` was obtained, because "the value at `q_min`" means different
things for the two phases.  For the liquid it is a shell mean, so the row carries
the shell multiplicity (6, 12 or 18) and the spread across that shell; for the
crystal it is one stored momentum on a named axis.

That spread is the liquid's own residual anisotropy, and at `r_s = 5.567` on the
`quick` budget the `q_min` shell spans about `+10.8%` of its mean -- comparable
to the `-12.7%` discrepancy it is being used to measure.  It is a caution about
the accuracy of that comparison, not a failure of it.

Every `q_min` here is the finite torus's smallest allowed momentum, set by the
supercell, and **not** `q -> 0`.  The run says so in as many words, and the
metadata records it under `small_q_statement`: this is a finite-`q` diagnostic,
not a demonstrated `q -> 0` limit, and no Kohn-theorem convergence is claimed
from it.

### The roton

`Q_WC` comes from the triangular Wigner crystal's geometry at `nu = 1`
(`a^2 = 4 pi l_B^2/sqrt(3)`, `Q_WC = |G_1| = 4 pi/(sqrt(3) a)`), not from the
data, and the run cross-checks it against the `g1_mag` a saved
`structure_factor.npz` records.

The search runs on the crystal's **directional** curve, so each `|q|` on it is
one momentum on one axis and the reported minimum is a minimum of a single
branch.  The function still groups equal `|q|` before looking for interior
minima -- harmless here, since a cut has one point per magnitude -- and that
generality is retained deliberately: the first version of this code sorted the
raw disc by `|q|` and looked for local minima, which made each point's neighbours
the other points *in its own shell*, and reported 75 "minima" at a neighbour
spacing of `0.0`.

On the default (paper-frame) axis the crystal curve has 8 collinear points and
the minimum lands on the one whose magnitude **is** `Q_WC` -- the crystal's own
first reciprocal vector -- for the `r_s = 90.51` store.  That is a property of
the *direction*: only a reciprocal-lattice direction reaches `Q_WC` at all, and
the `clean-x` axis misses it by `0.36`, where the shallow wiggle between two of
its five points is not a roton.  `TestTheCrystalReduction` pins both.

A minimum is reported as `resolved` only when its depth clears the larger of the
neighbouring shells' internal spread and the imbalance between the two neighbour
means.  `found` and `resolved` are separate fields on purpose: a minimum can
exist and still be too shallow to believe, and a caller reading only `found`
would claim a roton the next field denies.  When there is no roton the reason
string is exactly `roton not resolved at current finite-size/statistical
resolution: ...`, as the brief requires.

**The roton depth is not a thermodynamic number.**  `Omega = q^2/(2S)` is driven
down by `S`'s divergence at a Bragg peak, and a finite-size Bragg peak's height
grows with `N`.  So the minimum is reported as a *roton-like minimum in the SMA
dispersion* -- which is what it is -- and not as a collective mode whose
stability has been established.  Reported, not interpreted -- and reported as a
number in the table and the metadata rather than marked on the figure, where a
triangle over the point would cover the point.

### Output

`results/figure_construction/magnetoplasmon/<slug>/` holds
`magnetoplasmon.npz` (`q`, `q_over_sqrt_n`, `S_q`, `Omega_SMA`, `omega_mp` and
`kappa` per curve, plus the diagnostic table and the roton record) and
`run_metadata.json`, which identifies the exact source directory, `nmax`,
budget and write time of every curve.  `figures/figure_construction/magnetoplasmon/<slug>/`
holds `magnetoplasmon.png`.

The slug is `paper` for the brief's own couplings and a listing of the `r_s`
values otherwise, so a run over different couplings cannot overwrite the
paper's directory.

### Sources are checked for provenance

`r_s` values are matched **exactly**.  `5` is not `5.567`, and the nearby
coupling will not be drawn under a requested label.  Every candidate is also
gated on the `vmc/sr.py:219` conjugation fix of 2026-10-03 16:29:56: a file
written before it holds a crystal whose first shell is uniaxial, so its `S(q)` is
not the structure factor of the ansatz this package now implements, and it is
refused rather than silently drawn.  The liquid is not affected by that fix, but
the gate is applied to both phases anyway -- an exemption that has to be argued
per-phase is one that will eventually be argued wrongly, and being strict only
costs which curves are available.  Where a coupling was measured on both sides
of the fix, the newest post-fix file wins.

Which couplings are *available* therefore changes with what has been run, and the
default `r_s = 5, 30, 60` are the paper's, not necessarily this store's.  Running
it is the quickest way to find out: a coupling that is not there produces the
exact `structure_slice.py` command that would create it, and nothing is run.

That refusal lists only couplings the lookup would actually accept -- the same
truncation and post-fix predicates it applies -- and names a coupling that is
present but unusable separately, with what it failed.  The list is not "what is
on disk": a message that answers "why was my coupling refused" by printing the
coupling in its own list of what is available reads as "that value is not
allowed", when usually the truncation is one flag away from being right.

### `--liquid-rs` / `--crystal-rs`

There are two spellings for "which couplings to draw", and they mean the same
thing:

```
magnetoplasmon.py --rs 0 5.567 90.51 --source-budget quick
magnetoplasmon.py --liquid-rs 0 5.567 --crystal-rs 90.51 --budget quick
```

The second groups the couplings for reading, and puts them in a fixed order --
liquid group first, crystal group second, whatever order they are typed in, so
the colour a coupling gets does not depend on how the command line was arranged
on the day.  `--budget` is an alias of `--source-budget`.  They are
**alternatives, not additions**: giving `--rs` alongside either phase-named flag
is refused rather than concatenated.

**The flags do not decide the phase.**  That is `phase_of(r_s)` against
`r_s^c = 47` and nothing else, exactly as it is for `--rs`, so `--liquid-rs
90.51` is accepted and drawn as the **crystal**.  The startup banner prints the
phase next to each coupling, which is where a flag and its result disagreeing
would show up:

```
r_s requested    100 (crystal), 5.567 (liquid)
```

A hard gate refusing a mis-named coupling was tried and removed.  The boundary
is a property of the coupling, not of the spelling someone typed it under, so
the gate turned a spelling choice into an error without protecting anything --
the phase was never going to come from the flag in the first place.  `r_s = 0`
counts as liquid: it is the analytic curve drawn from the liquid branch, and
needs no saved data.

### `--budget`

`--source-budget quick|reproduction` (default `reproduction`, alias `--budget`)
selects which stored run to read, and is a selector rather than a preference:
asking for a budget that was not run fails rather than falling back.  Anything
that is not `reproduction` stamps the figure `NOT PUBLICATION QUALITY`, on the
picture and in the metadata, from one definition so the two cannot disagree.

### `--classical`

Turns **on** the dashed `omega_mp` classical-magnetoplasmon curves.  They are
off by default, so the plain invocation draws no dashed line at all.  This is a
**drawing** switch, not a physics one: `omega_mp` is computed and validated
against `omega_mp^2 = 1 + kappa q` with the engine's `kappa` on every run, and
written to the `.npz` and the metadata whether or not it is drawn, so the
comparison is reproducible from the saved run either way.
`classical_curves_drawn` in `run_metadata.json` records which way it went.
`--no-classical` is still accepted and is now redundant.

They are off by default because on the published figure the classical curves
and the SMA curves are the same colour per `r_s`, which makes the legend read as
pairs and buries the curves that carry the data: the exact `r_s = 0` line and
the directional crystal points.  This is the same reason the figure carries
**no other annotation** -- no `Q_WC` guide line, no marker planted on the roton
minimum.  Both of those were decoration added on top of the measurement, and
the roton marker in particular sat directly over the data point it described:
with it removed, the crystal's minimum at `Q_WC` is visible on the axes instead
of hidden behind its own label.  The roton is still searched for, printed and
saved; `Q_WC` is still in the small-`q` table.  They are reported as numbers
rather than drawn over the points they describe.

### Crystal density is capped at eight points, and that is intentional

The crystal curve has eight points at every `N` used here, and it cannot be made
denser by asking for more.  Along the paper's Bragg direction the allowed
momenta are `m (G_1 + G_2)`, `m = 1, 2, ...`, spaced by a whole supercell
reciprocal vector; for the `6 x 6` torus `m = 9` already exceeds `q_max = 4.0`,
so only `m = 1..8` exist.  The six C6-equivalent Bragg directions hold the same
probe at the same eight `|q|` values -- folding them in would give error bars at
eight x-positions, not a denser curve.  Interpolating between the eight would
draw a dispersion at `q` the VMC never measured, which is the failure mode this
whole recipe exists to avoid, so the points are drawn as bare markers with no
connecting line.

### Tests

`tests/test_magnetoplasmon_contract.py` covers the claims the recipe makes: that
it cannot start VMC or SR, that `r_s = 0` is the exact formula, that the SMA uses
the stored `S(q)` unchanged, that the `q/sqrt(n)` conversion follows from the
density, that the magnetoplasmon convention is the engine's, that `q = 0` is
handled by refusing rather than by dividing, that only post-fix sources are read,
and that a missing source fails instead of being recomputed.  It also covers the
reduction itself -- equal-`|q|` shells, the mean taken on `S` and not on `1/S`,
the crystal cut's collinearity and one-sidedness, the frame check, and that the
per-phase **dispatch** picks the right reduction for each phase.

Three groups are worth naming because they catch a plausible mistake rather than
a typo.  The convention test pins the **second**-order coefficients of both
curves -- `+kappa^2/4` for the SMA against `-kappa^2/8` for the classical one --
because a scale factor chosen to make the curves meet would leave the values
plausible and move these.  `TestTheCrystalReduction` pins the paper-frame axis
mapping and its corollary, that the package's own `x` axis misses `Q_WC` by
`0.36` and has no roton.  And `TestTheReductionIsWiredIntoTheCurves` exists
because the mutation harness found a real gap: every per-phase test built its
curve by calling the reduction directly, so replacing the *dispatch* to
shell-average the crystal left them all green.  The search tests are validated
on synthetic curves with a known answer before being applied to the real one.

`_diag/mutate_magnetoplasmon.py` mutation-tests the suite: 44 one-line defects,
43 killed.  The survivor is an equivalent mutant, checked rather than assumed --
for a triangular lattice at `nu = 1`, `|G_1| = 4 pi/(sqrt(3) a) = a^2/a = a`, so
the lattice constant and the first reciprocal vector are the same number to one
ulp and no test can tell them apart.

## Adding a recipe

Use the public API -- `VMC`, `run_vmc`, `load_budget`, and the analysis
functions under `wigner_vmc.analysis` -- and let the engine do the physics.  A
recipe that contains its own Metropolis loop, its own local-energy or Coulomb
formula, its own structure factor or pair correlation, is a second engine that
will drift from the validated one.

New figure recipes should normally be added here rather than by modifying the
validated VMC core.
