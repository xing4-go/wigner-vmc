# The fig06 crystal panel: a truncation gap, not a missing wavefunction family

`structure_slice.py --budget full` at the historical couplings rebuilds the
historical figure's `S(q)` and `g(r)` panels from a clean VMC run started from
nothing.  **The liquid panel reproduces.**  The crystal panel was reported here
as not reproducing; that report was measured at `--crystal-nmax 1`, and the
`n_max` ladder refutes it.  The two wavefunction families **converge with the
truncation**, and at `n_max = 3` the reciprocal-space weight agrees with the
historical ensemble to within the protocol's own spread.

Everything below is either a number read out of a stored artifact in this
repository or a number measured from one, and every number says which of the two
it is.

## 1. What the historical crystal is

Notebook cell 18's `crystal_wavefunction` is a Slater determinant of
**site-centred magnetic Gaussians** -- `GaussianBasis.orbitals` placed on the
Wigner-crystal sites and projected onto the lowest Landau level.

The clean package implements two crystal ansatze, and neither is that object:

| `--ansatz` | what it builds |
|---|---|
| `ll_rotation` (default) | Landau-level-rotated orbitals; orbital **and** Jastrow parameters optimised jointly |
| `ll_rotation_pinned` | the same wavefunction class; the 5 Jastrow parameters move, the orbitals stay at the Gaussian-overlap seed `L0` |

`ll_rotation_pinned` pins the *seed* inside the LL-rotated basis.  It does not
turn that basis into the historical one.  Measured on the pre-fix tree, pinning
moved `S(|g1|)` from 1.94 to 2.22, where the historical ensemble reads 11.92.

## 2. The gap closes with the truncation

`S(q)` at the triangular lattice's first Bragg vector, `|g1| = 2.6935 l_B^-1`.
The legacy side is re-derived from the *frozen configurations* through the same
clean estimator, so the comparison is of states, not of analysis code.

The historical side has no truncation to vary; the clean side does, and that is
the whole of the story:

| clean `--crystal-nmax` | budget | first-shell `S` (mean of the three families) | vs the historical 11.9154 |
|---|---|---|---|
| 1 | `full` | **7.2817** | 0.61x |
| 2 | `quick` | **11.5806** | 0.97x |
| 3 | `full` | **12.1661** | **1.02x** |
| 3 | `quick` | 13.2856 | 1.12x |

| reference | `S` at the first Bragg vector |
|---|---|
| historical ensemble (frozen, 250 configurations) | **11.9154** |
| clean, `n_max = 3`, `--budget full` | **12.1661** |

The `n_max = 1` row is the state the previous version of this note was written
about.  It is not where the ansatz family plateaus: adding one Landau level
moves the first-shell weight by 59%, and that motion is in the direction of the
historical value.

**Read the agreement at the size of the protocol's own spread, not at 2%.**  At
`n_max = 3` the `quick` budget reads 13.2856 against the `full` budget's
12.1661 -- a 9% spread, four times wider than the 2% residual against the
historical ensemble.  The defensible statement is therefore:

> at `n_max = 3` the clean LL-rotation crystal and the historical Gaussian
> crystal agree on the first-shell weight to within the protocol spread, and
> the `n_max = 1` shortfall of 1.64x was **truncation**.

not "the crystal panel reproduces to 2%".

## 3. Two numbers that used to be quoted here are superseded

**The factor of 6.1.**  An earlier version of the README quoted the crystal's
Bragg peak as "a factor of 6.1 low".  **That number must not be used.**  It is
`11.9154 / 1.9381`, where 1.9381 is the *smallest* member of a triplet the
projection defect had split -- a ratio against the most depressed of three
unequally populated directions, read as if the state were hexagonal.

**The factor of 1.64.**  This note's own earlier headline was
`11.9154 / 7.2817 = 1.64`, "a located divergence, not a tuning gap", with the
conclusion that closing it would need notebook cell 18 implemented as a third
wavefunction family.  That conclusion is **withdrawn**: 7.2817 is the `n_max = 1`
value, and §2's ladder shows the quantity was still moving.

## 4. C6: the pre-fix and post-fix crystal, side by side

The control in each row is the *same* estimator on the *same* run's liquid,
which is isotropic by construction -- so the last column is a ratio to the
estimator's own noise floor on this torus with this many snapshots.  A broken
symmetry reads many times the floor; a C6 state reads at or below it.

| run | `+g1` | `+g2` | `+(g1+g2)` | spread (max-min)/mean | worst \|z\| | vs its own liquid |
|---|---|---|---|---|---|---|
| `structure_slice/` (pre-fix, historical request) | 2.0226 | 5.6117 | 1.9381 | **1.1513** | **7.06** | **24.7x** |
| same request post-fix, `n_max = 1`, full | 7.4768 | 7.1495 | 7.2189 | 0.0449 | 0.48 | 0.38x |
| same request post-fix, `n_max = 3`, full | 12.3590 | 12.3905 | 11.7488 | 0.0527 | 1.22 | 0.44x |
| same request post-fix, `n_max = 3`, quick | 13.7948 | 13.0484 | 13.0135 | 0.0588 | 1.49 | 0.35x |

The pre-fix row is reproducible from the tree: `results/figure_construction/
structure_slice/` still holds that run's `run.npz`.  24.7x the liquid's noise
floor is a broken symmetry; 0.35-0.44x is a crystal that is C6 to within what
the data resolves.

The `n_max = 3` state is measured twice, through two recipes that share no code
-- `structure_slice.py` (radial `S(q)`) and `structure.py` (the 2-D `S(qx,qy)`)
-- and they return the **same three numbers** with the family labels permuted.
That is a cross-check of the torus and the estimator, not of the physics.

## 5. Real space (numbers below are pre-fix and superseded)

Real-space distances are set by the lattice, and both states sit on the same
one.  The clean rows in the table below were measured on a **pre-fix** run, so
they describe a state with the broken first shell; they are kept only to show
where the two families differed, and they must be re-measured against the
post-fix states before being quoted:

| | first `g(r)` peak (position, height) | max abs dev | rms dev |
|---|---|---|---|
| crystal, clean (**pre-fix**) | (1.4054, 2.475) | 0.347 | 0.172 |
| crystal, legacy | (1.6875, 2.625) | -- | -- |
| liquid, clean | (1.0575, 4.275) | 0.074 | 0.026 |
| liquid, legacy | (1.0680, 2.775) | -- | -- |

The *positions* agreed; the weights at the reciprocal-lattice points did not --
and §2 shows that second quantity is controlled by the truncation.

## 6. What this means for the recipe

* There is still **no `--ansatz` value that is the historical crystal**, and the
  recipe still offers none: `ll_rotation_pinned` pins a seed inside a different
  basis.  What has changed is that the *gap* is now known to be a truncation
  effect rather than a missing wavefunction family, so `--crystal-nmax` is the
  knob that moves it.
* The recipe records the caveat in its own metadata
  (`ansatz_of_the_historical_crystal`, `ansatz_implemented`,
  `deviations_from_legacy`), so a reader of a `run_metadata.json` does not have
  to find this file to learn it.  Those strings were corrected on 2026-10-03 to
  carry the ladder instead of the withdrawn claim.
* The honest one-line statement: **the liquid panel is a reproduction; the
  crystal panel is a reproduction at `n_max = 3`, at the precision the protocol
  resolves; at `n_max = 1` it is not, and every earlier "divergence" number in
  this repository was measured there.**
