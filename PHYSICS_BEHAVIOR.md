# Physics behavior firewall (Stage 2C)

The B1--B7 regressions already pin the clean package against the frozen engine.
This document is the index to them, plus the one command that answers the only
question the architecture work will keep asking:

```bash
python -m pytest -m behavior      # the physics firewall, layers A-F (300 tests)
python -m pytest -m layer_D       # one layer at a time
python -m pytest -m contract      # figure / IO data-flow contract
python -m pytest tests            # the whole suite (316 tests)
```

The classification lives entirely in `tests/conftest.py`; no test body and no
production module was changed to create it. `python -m unittest` still runs
everything, it simply has no notion of the markers.

**Green firewall ⇒ the VMC physics has not changed.** Red ⇒ the change that was
just made is the cause. That is the whole contract, and it is only meaningful
because the B1--B7 tests it selects are old/new comparisons against the frozen
engine, not self-consistency checks.

## Layers

| | layer | source modules locked | test files |
|---|---|---|---|
| A | geometry / Hamiltonian conventions | `physics/geometry.py`, `physics/magnetic_cell.py`, `physics/coulomb.py`, `physics/units.py` | `test_units_and_conventions.py`, `test_geometry_against_legacy.py`, `test_magnetic_cell_against_legacy.py`, `test_coulomb_against_legacy.py` |
| B | wavefunctions | `wavefunctions/jastrow.py`, `gaussian.py`, `ll_rotation.py`, `physics/landau_levels.py` | `test_jastrow_against_legacy.py`, `test_gaussian_against_legacy.py`, `test_ll_rotation_against_legacy.py`, `test_landau_basis_against_legacy.py`, `test_behavior_fixtures.py` |
| C | LL nesting | `wavefunctions/nesting.py` | `TestNesting` in `test_ll_rotation_against_legacy.py`, `TestNestingInvariant` / `TestNestingLadder` in `test_behavior_fixtures.py` |
| D | sampler | `vmc/sampler.py` | `test_sampler_against_legacy.py` |
| E | energy / measurement | `wavefunctions/slater.py` (local energy), `vmc/measure.py`, `analysis/statistics.py`, `analysis/structure.py` | `test_local_energy_against_legacy.py`, `test_measure_against_legacy.py`, `test_statistics.py`, `test_structure_against_legacy.py`, `TestEnergyFieldDiscipline` |
| F | stochastic reconfiguration | `vmc/sr.py` | `test_sr_against_legacy.py` |

## Invariants

### A -- geometry / Hamiltonian conventions

| quantity | test | tolerance | why not zero | legacy source |
|---|---|---|---|---|
| `r_s = sqrt(2) * kappa` | `TestTheLockedRelation` | exact | — | `qhvmc_engine.py` module constants |
| `kappa` is stored but NOT applied in the sampler | `TestConstants` | exact | — | `qhvmc_engine.py` |
| coupling grid `{45, 65, 75, ...}` | `TestConstants::test_the_frozen_coupling_grid_is_the_frozen_one` | exact | — | frozen `_diag/energy_scan.json` |
| nmax rule: clean `nmax` = legacy `n_max - 1` | `TestTheNmaxRule` | exact | — | `qhvmc_engine.py` `LandauLevelBasis` |
| cell matrices, direct-space vectors | `test_geometry_against_legacy.py` | `= 0` | — | `qhvmc_engine.py` |
| reciprocal vectors | `test_geometry_against_legacy.py` | 1 ULP | LAPACK inverse | `qhvmc_engine.py` |
| periodic wrapping / minimum image | `TestWrapping`, `TestMinimumImage` | `= 0` | — | `qhvmc_engine.py` |
| gauge `A = (-y, x)/2`, fold, flux quanta (1 primitive / 36 supercell) | `test_magnetic_cell_against_legacy.py` | `= 0` | — | `qhvmc_engine.py` |
| Ewald tables, Madelung constant 1.106103 | `test_coulomb_against_legacy.py` | `= 0` tables; rtol 1e-13 energies | — | `qhvmc_engine.py` `CoulombEwald` |

### B -- wavefunctions

| quantity | test | tolerance | why not zero | legacy source |
|---|---|---|---|---|
| Jastrow `b3`, `dx_b3`, `d2x_b3`, `grad`, `laplacian` | `test_jastrow_against_legacy.py` | `= 0` | — | `qhvmc_engine.py:469-619` |
| Jastrow `u`, `grad_params`, `cusp_gamma` | same | rtol 1e-13, atol 1e-14 | — | `qhvmc_engine.py:469-619`, `make_notebook.py:930` |
| Gaussian `sites`, `l_cart` | `test_gaussian_against_legacy.py` | `= 0` | — | `qhvmc_engine.py:300-432` |
| Gaussian `log_psi`, `pi_columns`, orbitals | same | rtol=atol 1e-13 | — | `qhvmc_engine.py:300-432` |
| LL orbital coefficients, `c_row`, `C0` | `test_ll_rotation_against_legacy.py` | `= 0` | — | `qhvmc_engine_llrot.py:59-243` |
| LL orbitals, `pi`, `pi2` | same | rtol 1e-12, atol 1e-13 | the clean and legacy Landau-level bases sum lattice vectors in a different order (~4e-16 rel on `Ops`, ~3e-16 on `_kin`) | `qhvmc_engine_llrot.py:59-243` |
| `pi^2 psi = sum C_kn (2n+1) phi_kn` | `TestGaugeSign::test_pi_squared_is_diagonal_weighted_by_C` | rtol=atol 1e-13 | — | — (own identity) |
| the two `pi_columns` signatures stay different | `test_landau_basis_against_legacy.py` | structural | — | deliberately NOT unified (Stage 2B/2D boundary) |

### C -- LL nesting

| quantity | test | tolerance | why not zero | legacy source |
|---|---|---|---|---|
| coefficient identity `(c_0,c_1) -> (c_0,c_1,0)`, rungs 1->2, 2->3, 3->4 | `TestNesting::test_the_identity_is_exact` | **exactly 0.0** on head and tail | `c_row` evaluates the identical expression for the head and an exact zero for the tail | `reproduction/make_notebook_llrot.py:1748-1750` |
| assembled orbitals unchanged, all three rungs | `TestNestingLadder::test_each_rung_leaves_the_orbitals_alone` | `= 0` | as above | same |
| `logdet D`, `U`, `T`, `V`, `E_local` unchanged, all three rungs | `TestNestingLadder::test_each_rung_leaves_the_observables_alone` | rtol 1e-12, atol 1e-13 | — | same |
| nested fixture reproduces `ll_nmax1` | `TestNestingInvariant::test_the_nested_ansatz_has_the_same_observables` | rtol 1e-12, atol 1e-13 | — | same |

### D -- sampler

| quantity | test | tolerance | why not zero | legacy source |
|---|---|---|---|---|
| configuration chain | `TestTheChainIsBitIdentical` | `= 0` (`assert_array_equal`) | — | `qhvmc_engine.py:903-951` |
| RNG type, draw order and count | `TestTheRngContract` | `= 0` | — | `qhvmc_engine.py:903-951` |
| 1000-attempt adaptation window, sigma update | `TestTheSigmaAdaptationRule` | `= 0` | — | `qhvmc_engine.py:903-951` |
| reset period, proposal, acceptance rule | `TestTheSamplersOwnBehaviour` | `= 0` | — | `qhvmc_engine.py:903-951` |

The chain test is the mutation-sensitive one: swapping two draws or adapting
sigma from the cumulative rate instead of the window leaves every number looking
plausible and every downstream energy wrong.

### E -- energy / measurement

| quantity | test | tolerance | why not zero | legacy source |
|---|---|---|---|---|
| `T_det`, `T_mix`, `T_jastrow` separately | `TestCalcKineticEnergy` | rtol 1e-12, atol 1e-12 | — | `qhvmc_engine.py:719-900` |
| Gaussian `T`, `V`, `E_local` | `TestGaussianLocalEnergy` | rtol=atol 1e-12 | — | `qhvmc_engine.py:719-900` |
| LL `T`, `V`, `E_local` | `TestLLRotationLocalEnergy` | rtol=atol 1e-12 real, 1e-11 imaginary | `cond(D) = 2.4e6` — see below | `qhvmc_engine.py:719-900` |
| `E_local = T + kappa * V`, both ansatz | `TestTheEnergyConvention` | rtol 1e-12, atol 1e-13 | — | `qhvmc_engine.py` |
| `measure` == engine totals | `test_measure_against_legacy.py` | rtol 1e-15 | — | `make_notebook.py:707-719` |
| `tau_int`, naive vs blocked error band | `TestSqErrorBars` | rtol 1e-15 | — | `make_notebook.py:1435-1443` |
| S(q): `abs(rho_q)^2 / ne = NE` | `TestDensityGrid` | rtol 1e-12 | — | `make_notebook.py` |
| bragg ratio on an ideal lattice | `TestBraggRatio` | `= NE` exactly, background = median | — | `make_notebook.py` |
| the three energy semantics stay apart | `TestEnergyFieldDiscipline` | structural | `E_optimization_*` are TOTALS, `E_production` is PER ELECTRON (factor 36) | legacy records |

### F -- stochastic reconfiguration

| quantity | test | tolerance | why not zero | legacy source |
|---|---|---|---|---|
| `S`, `f` (jastrow and joint) | `test_sr_against_legacy.py` | rtol 1e-12 | — | `qhvmc_engine.py:959-1025`, `qhvmc_engine_llrot.py:369-562` |
| `theta` after one step | `test_the_first_step_agrees_field_by_field` | 1.709e-12 measured, bound 3.157e-8 | `cond(S_reg) ~ 2e4` — see below | same |
| `E`, `E_err`, `force`, `tau` at step 0 | same | exactly 0.0 for `E`/`E_err`, 2.842e-14 for `force` | — | same |
| 3-step trajectory | `test_the_trajectory_agrees` | `theta` vs conditioning bound; `acc`/`sigma`/`tau` exact; theta-dependent scalars vs their own measured rate of change | the two runs' inputs diverge after step 0 | same |
| parameter ordering, regularization, solver, update convention, defaults | `TestSrOptimizeJoint`, `TestSrOptimizeJastrow` | structural | not to be changed in 2C/2D | same |

## The two conditioning tolerances

Both are **measured error propagation, not slack**. They are recorded here so
that a future reader does not see a `1e-10` or `1e-12` deviation and report a
regression.

**`cond(D) = 2.4e6`** (LL ansatz, the B5 test configuration). `D_inv` is the only
quantity in the kinetic-energy path that is not assembled the same way on the two
sides. A 1-ULP disagreement in `D` is amplified to ~1e-10 in `D_inv`, and reaches
`T` through `einsum("li,il->i", pix_D, D_inv)`. `T`'s deviation is therefore
bounded by the measured conditioning rather than by a flat tolerance.

**`cond(S_reg) ~ 2e4`** (B7 joint-SR test state; `eig_min(S_x) ~ -6.7e-13`,
`eig_max ~ 20.6`). A 1e-15 relative disagreement in `S` and `f` lands at ~1.7e-12
in `theta` after one step and ~1.8e-10 by step 2. Everything downstream of
`theta` inherits that gap, so the trajectory test propagates it through each
scalar's own measured rate of change along the run (`_traj_lip`) instead of
assuming a fixed tolerance.

## Golden fixtures

Five files in `tests/fixtures/behavior/`, each storing the **inputs** (a
configuration `R`, band coefficients `v`, the model parameters) and the
observables the accepted code returns for them. `tests/test_behavior_fixtures.py`
rebuilds the ansatz from the stored input and re-derives `logdet(D)`, `U`, `T`,
`V`, `E_local`.

| fixture | layer | ansatz | what it locks |
|---|---|---|---|
| `gaussian_regular` | B | Gaussian | the interior configuration — the sampler's normal regime |
| `gaussian_boundary` | B | Gaussian | points outside the supercell: the wrapping / minimum-image path |
| `ll_nmax1` | B | LL rotation, `n_bands = 2` | the LL orbital pipeline at the first rung |
| `ll_nmax2` | B | LL rotation, `n_bands = 3` | the same at the second |
| `nested_nmax1_to_2` | C | LL rotation, `n_bands = 3`, `v[:, 2] = 0` | the nesting invariant, through the energies |

Regenerate with `python tests/fixtures/behavior/_build.py` (not run by the suite).
The fixtures hold no Monte Carlo state and no RNG: the rebuild needs neither.

Note what a fixture is and is not. It is a **snapshot** -- the expected values
came from the clean code -- so on its own it cannot show that clean matches
legacy. The B1--B7 tests in layers A--F do that. The fixtures are the other half:
they fail in a second, with the legacy tree absent, the moment a refactor moves
Gaussian or LL behaviour. Together the two pin legacy behaviour.

**There is no `log_psi` API** anywhere in the clean package or the frozen engine;
Psi enters the pipeline only as a `|Psi|^2` ratio and as the local energy. So the
fixtures lock `logdet(D)` and `U`, the two first-class ingredients, rather than an
invented `log Psi`. The `|Psi|^2` ratio chain is locked by layer D.

Tolerance: the rebuild is **bit-identical** — measured `0.0` on every observable
of every fixture, across separate processes, after the JSON round trip. The tests
assert rtol 1e-12 / atol 1e-13 anyway, because the determinant inversion goes
through LAPACK and a different BLAS or thread count may reassociate `D_inv`.

## What is outside the firewall

| | why |
|---|---|
| `test_figure_contract.py` (marker `contract`) | a Stage 1 rule about the figure data flow, not VMC physics |
| `test_legacy_conversion.py` other than `TestEnergyFieldDiscipline` | results-conversion and scan arithmetic; in the full suite, not a physics statement |

Both are run by `python -m pytest tests`.

## Not handled in 2C

The three legacy `BUG_CANDIDATE`s (`pi_columns` signature split, the Gaussian
cutoff guard, the coincident-pair Jastrow NaN), the `sr_pilot_scale` docstring's
"orders of magnitude" claim, and the `n_max` naming mismatch are **locked as they
are**. Stage 2C locks behaviour; it does not fix it. Those decisions belong to
Stage 2D.
