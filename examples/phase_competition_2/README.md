# `phase_competition_2` — rebuilding the phase comparison from the paper

This directory holds a rebuild of the Wigner-crystal / Hall-liquid phase
comparison, reconstructed **from the paper's description of the physics** rather
than from this project's legacy wiring.  The first deliverable is
`crystal_seed_search.py`, which is **Stage 1A**: the paper's crystal seed.

    python examples/phase_competition_2/crystal_seed_search.py --rs 90 --trials 4

**The reconstruction rule.** The frozen `make_notebook.py` and the notebooks it
emits are not a specification for this directory.  They are the object of a
reproduction, and where they and the paper disagree, the paper decides.  Where
the paper is silent, this directory says so and makes a recorded choice.

---

## 1. The question

The paper's crystal is seeded from a **randomly initialized determinant**.
Every crystal point this project has delivered is instead seeded from
`VMC.crystal_v0` → `sr.gaussian_overlap_seed` → `sr.v_from_overlap`: a
site-centred **Gaussian** projected onto the kept Landau levels.  Those are
different initializations, and they are not obviously in the same basin.

`crystal_seed_search.py` asks one question at one coupling: **do independent
random determinants, optimised at large `r_s`, form a triangular Wigner
crystal?**  Nothing else — no continuation, no nesting, no phase comparison.

### What the paper actually says

Reddy & Fu, PRB **113**, L161403 (2026); arXiv:2508.21000.  Methods,
"Variational Monte Carlo methods":

> For the Wigner crystal phase, we generate crystal 'seed' orbitals by
> optimizing a **randomly initialized determinant** with a Jastrow factor at
> large `r_s`.

and for the liquid:

> For the liquid phase, we optimize only `u(r)` and set `U_mn = δ_mn` so that the
> determinant is the noninteracting ground state.

Two consequences are easy to get wrong and are recorded here explicitly because
both were carried into this project's earlier plans:

* **The continuation is a STAR, not a chain.**  The paper's own words: *"For
  other values of `r_s`, we initialize the orbitals to this seed and then
  optimize them together with the Jastrow factor."*  It does **not** walk
  `90 → 85 → 80 → …`.
* **The paper has no `n_max` growth protocol at all.**  The `n_max = 1 → 2 → 3`
  nesting ladder is *this project's* addition — a one-sided refinement, not a
  reproduction of anything the paper does.  It must never be presented as the
  paper's.

The Gaussian-overlap construction does appear in that work, but only as an
independent **benchmark** (SM §III D), never as the seed.  That is exactly the
role it keeps here: a reference run, reported next to the random starts.

---

## 2. The two pipelines

The two phases are built differently, and the difference is physical rather than
a matter of taste.

| | liquid | crystal |
|---|---|---|
| orbitals | `v ≡ 0` — the filled lowest Landau level | a variational `v`, `(nk, n_max)` complex |
| what is optimised | the 5 Jastrow parameters only | Jastrow **and** orbitals (joint SR) |
| `n_max` dependence | none: extra bands are unoccupied, so `E` cannot move | physical — it is the size of the variational orbital space |
| seed | none needed; `v = 0` is the noninteracting ground state | a **Haar-random** determinant at large `r_s` |
| role | the disordered control | the object of study |

The liquid's `n_max`-independence is not an assumption to be trusted: it is a
pinned test (`TestTheLiquidHasNoOrbitalSector`), because the liquid is the
reference against which every crystal number below is read.  A liquid whose
energy moved with `n_max` would mean a variational orbital sector had appeared in
the control.

### The crystal's construction, in order

```
Haar-random determinant  at large r_s      <-- Stage 1A: this directory
        |
        v  optimize (joint SR)
        |
   root state  Psi_seed
        |
        v  STAGE 2 (not implemented): the paper's star
        |
   Psi(r_s) for each target r_s, independently
        |
        v  STAGE 2: exact n_max = 1 -> 2 -> 3 nesting at each target  (ours, not the paper's)
        |
   the n_max ladder at that r_s
```

The star and the ladder are separate: the star is the paper's, the ladder is
ours, and the README says so where a reader will see it.

---

## 3. The random determinant, and why Haar

At `N = 36` there are `nk = 36` momenta.  At `n_max = 1` the orbital matrix
`C = lr.c_row(v)` is `C[k] = (cos t_k, i·sin t_k e^{iφ_k})` with `t_k = |v_k|`, so
**per momentum the manifold is CP¹ in `(t_k, φ_k)`** and the excited-band weight
of orbital `k` is exactly `sin²t_k`.

The Fubini–Study measure on CP¹ is `sin(2t) dt dφ`, whose CDF is `sin²t`.  So:

```
t_k = arcsin(sqrt(u_k)),  u_k ~ U[0,1]        E[sin²t] = 1/2
φ_k ~ U[0, 2π)                    i.i.d. over the 36 momenta
```

Haar is the **least arbitrary** reading of "a random determinant" — it is the
measure the manifold itself carries, and it has **no scale knob**.  There are
deliberately no `t_max` arms: whether the *measure* changes the WC basin is a
different scientific question, and asking it here would replace "does the paper's
method reproduce?" with "which determinant do I like?".  The only random variable
is the random determinant realization; the variation across trials is *seeds*,
not a mixing scale.

`t = 0` is the pure lowest Landau level, so a uniform draw is **not** a disguised
Gaussian seed — it is a genuinely different family, which is the whole point.

`nbar_init = mean_k sin²t_k` is recorded for every trial, so "was the start
artificially mixing-heavy?" is a number rather than an assumption.

---

## 4. What Stage 1A is, and what it is not

**Stage 1A is a pipeline validation.**  `quick` is SR 10×40.  A fully random
determinant asks the optimiser to find translational-symmetry breaking,
triangular order, the right Landau-level mixing *and* the right Jastrow
correlation at the same time.  If none of the starts forms a Wigner crystal:

> **that is not evidence the basin is unreachable.**  It is equally consistent
> with the optimiser not having had time.

Stage 1A therefore makes **no scientific claim in either direction**.  It
validates the pipeline and reports the numbers.  Establishing whether
independent random starts *converge* is Stage 1B, at `reproduction`, and is a
separate decision.

The one outcome that must not be read as physics is spelled out in the code as
an escape hatch: if the **Gaussian-seed reference** — a point known to be a
crystal — cannot clear the C6 gate at this budget, the run reports
**`NO VERDICT`** rather than `0/4`.

---

## 5. The criterion, fixed before the run

A trial counts as a triangular WC iff **all three** hold.  The constants live at
the top of the recipe and are printed in the banner and stored in
`run_metadata.json`, so a reader can audit them rather than trust them:

1. `bragg_ratio ≥ 3 × bragg_ratio_liquid` — the reference is measured **in the
   same run at the same `r_s`**.  `bragg_ratio` is an absolute number whose scale
   is a property of the budget, so a fixed threshold would be a statement about
   the protocol rather than the state.
2. `c6_residual ≤ 0.1` — the **primary structural gate**.  `c6_residual` is the
   normalized least-squares residual of the rotated occupied space against the
   unrotated one, bounded in `[0, 1]`, invariant under a change of orbital gauge,
   and measured at `3.0e-14` for a C6-covariant seed against `7.9e-01` for a
   C6-broken one.  **Bounded and gauge-invariant is exactly what a ratio of two
   noisy `S` values is not**, which is why the anisotropy `A` is *reported* here
   and never thresholded.
3. `site_midpoint_contrast ≥ 2` — real space, from an estimator that is exactly
   1 for anything uniform and does not reuse `S(q)`.  It is computed on the
   density smoothed at the historical `0.40 l_B`: on the **raw** histogram the
   nearest-neighbour midpoints of a sharp lattice fall in empty bins and the
   statistic is `inf`, which would pass this gate for free (pinned by
   `TestTheContrastIsAFiniteMeasurement`).

Reported for every trial, gated on none: `S(G1), S(G2), S(G3)` and their spread
`A = (max−min)/mean`; the per-rotation C6 residuals; the LL occupation of the
start and of the end.  The `±G` residual `S(+G) − S(−G)` is reported too, but it
is an **exact** identity (`ρ_{−q} = conj(ρ_q)`), so a non-zero value is a code
defect rather than physics — and it reads `0.0` exactly in every run so far.

### Root selection

> Among trials with `wc_formed` true, the root is the one with the lowest
> `E_final`.

If none pass, the run reports `NO ROOT` and does **not** relax the criterion and
does **not** fall back to the best-energy trial.  An energetic winner that is not
a Wigner crystal is a different object, and returning it under the name "root" is
how a criterion gets quietly relaxed.

---

## 6. Outputs

```
results/phase_competition_2/crystal_seed_search/<slug>/
    runs.json           per-trial records + both reference records
    run_metadata.json   banner fields, BLAS env, criterion constants, verdict, root
    states/root__*.pkl  the winning (c, v), snaps dropped -- Stage 1B's input
figures/phase_competition_2/crystal_seed_search/<slug>/crystal_seed_search.png
```

`<slug>` is `N36__ll_rotation__<budget>__rs90__nmax1`.  The **table is the
deliverable**; the figure is quality control and is stamped
`QUICK / not publication quality` unless the budget is `reproduction`.

House style: this recipe writes only under `results/` and `figures/`, never into
`examples/`.

---

## 7. Status

* **Stage 1A** — implemented; the `quick` campaign was killed by the harness
  after 2 of 6 points and reports **`NO VERDICT`**.  See §8 for what that does
  and does not mean.
* **The C6 SR-trajectory diagnostic** — §9, and the live work.  It exists
  because §8's `NO VERDICT` turns on a more basic unanswered question.
* **Stage 1B** — a few Haar starts at `reproduction`.  **Not started**; a
  separate decision, and deliberately not triggered automatically.
* **Stage 2** — the star + nesting ladder of §2.  **Not started.**

---

## 8. Stage 1A results — `NO VERDICT`

**The `quick` campaign was killed by the harness for system memory pressure
after 2 of its 6 points** (the liquid and the Gaussian reference).  It was
**not** restarted and it **wrote nothing**: the slug directory
`results/phase_competition_2/crystal_seed_search/N36__ll_rotation__quick__rs90__nmax1/`
exists but is **empty** — it is the `makedirs` of a run that died before its
first write — and no figure was produced.  `_diag/run_1a.log` is the record of
that run and is kept.

### What was measured

The `smoke` reference **is** stored
(`results/phase_competition_2/crystal_seed_search/N36__ll_rotation__smoke__rs90__nmax1/runs.json`).
The killed `quick` reference survives only in `_diag/run_1a.log`, where its
per-step SR trace and its final measurement rows are printed in full.

| budget | SR | `c6_residual` | `E/ne` | `bragg_ratio` | contrast | gates | verdict |
|---|---|---|---|---|---|---|---|
| `smoke` | 2 × 20 | **0.0866** | −45.508413 ± 0.064999 | 8.635 | 4.684 | all three pass | `FAILED` (0/2 trials) |
| `quick` | 10 × 40 | **0.209** | −45.743463 | 9.265 | 5.25 | C6 fails | **`NO VERDICT`** |

The killed `quick` run's liquid reference reads `bragg = 1.109`,
`c6 = 2.56e-14`, `ctr = 0.95`.  Every `E` in these tables is **per particle**.

### The verdict, and its two parts

**As the program verdict**, this is the pre-registered rule doing its job, not a
failure.  The escape hatch of §4 is explicit: the Gaussian reference — a point
known to be a crystal — does not clear the C6 gate at `quick` (0.209 against
`C6_GATE = 0.1`), so `overall_verdict` returns **`NO VERDICT`** rather than a
`0/4`.  The four Haar trials could not have changed that number.

**Physically**, the four Haar trials are **not worthless**.  They still answer a
real question — *does a random determinant tend toward the same broken-`C6`
basin?* — and that question is untouched by this outcome.  They simply do not
need re-running **now**, because a more basic question is open and unresolved,
and it is the one that decides how their result would be read.  That, and not
the `NO VERDICT` label, is the reason the campaign stays killed.

### The finding that redirects the work

Reading the two references side by side changes the picture.  The **same
construction passes the C6 gate at `smoke` and fails it at `quick`**: more SR
moved the state *further* from C6-invariance.  And `bragg_ratio` and the
contrast went **up** while C6 got worse (8.635 → 9.265, 4.684 → 5.25) — the
crystal got *sharper* as its C6 residual grew.

**A sharper crystal with a worse C6 residual is not disorder.**  It points at
the lattice's **registry or orientation on the torus** rather than at a
degrading lattice: the residual is measured about the torus origin, so a crystal
that has *slid* is no longer invariant about it.  That is a hypothesis to test,
not a conclusion to draw from two cross-budget numbers.

The question this leaves is therefore narrower and more basic than Stage 1A's:
**why does a seed that begins as the correct C6 crystal leave the C6 sector
under unconstrained joint SR?**  That is `crystal_c6_trajectory.py`, in §9.

---

## 9. The C6 SR-trajectory diagnostic

    python examples/phase_competition_2/crystal_c6_trajectory.py --steps 30 --seeds 0,1,2

A single-point, per-step diagnostic at `r_s = 90`, `N = 36`, `n_max = 1`,
starting from the **corrected** Gaussian seed.  It answers §8's question in
three layers, because a trajectory alone cannot:

| layer | question | why it must come first |
|---|---|---|
| **1** | is the C6 seed *strictly* correct at this operating point? | without it, "SR broke it" and "the seed was never C6 here" are indistinguishable |
| **2** | does the **first** SR update break it, and what does the trajectory look like? | — |
| **3** | is the breaking component a property of the **algorithm** or of the **MC noise**? | until this is settled, none of the candidate explanations can be separated |

**Layer 1.**  `TestSeedC6Covariance` pins the corrected seed at `r_s = 75,
n_max = 2`.  Nothing pinned `r_s = 90, n_max = 1`, where the manifold differs
(one excited band, `v` of shape `(36, 1)`), so the script re-measures it in-run
at its own context and asserts three things: a **positive control** (the
corrected seed must read `C6_0 ≲ 1e-9`), a **second, independent context**
(different `nsp` *and* different sample seed, so a context bug cannot pass
vacuously), and a **negative control** (`v_from_overlap(ov.conj())` must read
*large*, or a context that calls everything "broken" would pass).  Measured at
the real operating point: `3.031e-14` (and `3.039e-14` in the second context)
against `5.944e-01` for the conjugated control — the seed **is** C6-covariant
there, to the same floor as the filled lowest Landau level (`2.563e-14`).

**Layer 2** records, per step, `E_t` (both units), `C6_t` **with its
five-rotation vector**, `S1/S2/S3` from an independent walk, the density
contrast and `bragg_ratio`, the registry phases, `|f_t|`,
`‖Δv_t‖`/`‖Δc_t‖`, and the optimiser's own `acc, sigma, tau, cond, eig_min,
eig_max`.  Two bookkeeping rules are pinned by tests because both would
manufacture the headline this diagnostic is looking for:

* **Pairing.**  `sr_optimize_joint` samples at `theta_i` and *then* updates, so
  `hist[i]["E"]` belongs to `theta_i` while `hist[i]["theta"]` is
  `theta_{i+1}`.  Pairing `E` and `theta` from the same row lags the state one
  step and reads *exactly* like "the energy fell while C6 rose".  The returned
  state has no `hist` row, so it gets an explicit **closing row**.
* **Units.**  Every `hist` energy is a **TOTAL**; the campaign's published
  numbers are per particle.  Both are stored, under names that say which.

**Layer 3** runs on the **first step**, which is *schedule-free*: at `i = 0`,
`tau_0 = tau / (1 + xi·0/steps) = tau` for every `steps`, so `R_C6(1)` is
comparable across budgets.  `space_residual` is the **sine of the largest
principal angle** — linear in the breaking amplitude, not quadratic — so
`R_C6(1)` *is* the relative first-order symmetry-breaking amplitude of the first
SR update, with no square root.  The experiments: **(3a)** vary only the SR
sampling seed at a fixed `theta0` and read the *direction* each run breaks in,
not just the distance; **(3b)** the `nsweep` ladder, where `R_C6(1) ∝ n^{-1/2}`
means pure MC noise and a plateau means genuine algorithmic anisotropy;
**(3c)** the `scale = pilot` counterfactual; **(3d)** the pinned-orbital control,
which asks whether breaking C6 buys any energy at all.

### The verdict rule

Per seed and **two-staged**, because `C6` alone cannot name a state.

*Stage A* — is the closing state still a crystal?  `NOT_A_CRYSTAL` if
`bragg_ratio < 3 × liquid` or `contrast < 2`; otherwise `CRYSTAL`, and only then
does C6 mean anything about symmetry.

*Stage B* — the C6 trajectory shape: `SEED_NOT_C6` (`C6_0 > C6_GATE`, checked
**first**, so SR is not blamed for a seed that was never C6 here),
`STABLE_C6`, `UNDERCONVERGED`, `REGISTRY_DRIFT` (the lattice slid rather than
broke), `DRIFT_SETTLED`, `PLATEAU_BROKEN`, and `STILL_RISING`.
`STILL_RISING`, `PLATEAU_BROKEN` and `REGISTRY_DRIFT` are **not** verdicts about
the candidate explanations — each names the next experiment.

`C6_GATE` is **imported** from `crystal_seed_search.py`, not re-typed, so §5's
gate and these labels cannot drift apart.

### 9.1 Layer 1, measured at this operating point

`r_s = 90`, `N = 36`, `n_max = 1`, `init_id = 0` (so `L0 = 0.30`, giving
`nbar = 0.3500`):

| control | reading |
|---|---|
| corrected Gaussian seed | **3.032e-14** |
| the same, in a second context (`nsp = 2000`, `c6_seed2 = 20261004`) | 3.039e-14 |
| the conjugated overlap — the negative control | 5.944e-01 |
| columns rescaled by `(1 + 0.5i)`, largest per-rotation change | 3.259e-18 |
| the filled LLL (`v = 0`, the liquid) | 2.563e-14 |

**The seed IS C6-covariant here**, at the same floor as the filled lowest Landau
level, and the context resolves the gate (a separation of ~2e13 against the
broken control).  So the answer to "is the seed itself the problem?" is no, and
Layer 2's question is well posed at this point.

At `N = 9` the same construction reads `4.584e-01` at that run's own coarse
`nsp = 12` (and `5.87e-01` at `nsp = 120`) — the Gaussian seed is **not**
C6-covariant there, and `N = 9` cannot be used for any test that needs the
Layer-1 gate to pass.  The script checks Layer 1 on **every** run and stops with
`rc = 2` *before* any SR if it fails, rather than producing a trajectory that
would be read as "SR broke the symmetry".

### 9.2 The pilot: two steps, one seed

    python examples/phase_competition_2/crystal_c6_trajectory.py --steps 2 --seeds 0

| `t` | C6 | `nbar` | `E/N` | `\|f\|` | `‖Δv‖` | `S(G1)` | `S(G2)` | `S(G3)` | bragg | contrast |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **3.03e-14** | 0.350 | −44.2377 | 37.44 | — | 6.135 | 5.975 | 6.412 | 6.789 | 3.91 |
| 1 | **1.47e-01** | 0.375 | −45.2522 | 12.42 | 0.632 | 6.434 | 5.730 | 5.684 | 6.119 | 3.59 |
| 2 (closing) | **1.55e-01** | 0.369 | — | — | 0.176 | 6.877 | 6.349 | 6.616 | 6.799 | 4.08 |

Stage A = `CRYSTAL` (bragg 6.799 ≥ 3 × 1.109, contrast 4.08 ≥ 2), Stage B =
`STILL_RISING`.  The re-run reproduces this table exactly, row for row.

Two things follow, and the first is the one this diagnostic exists for:

* **The FIRST update does essentially all of it.**  `R_C6(1) = 1.47e-01` against
  `R_C6(0) = 3.03e-14` — the second update moves it by only 5% of its own value.
  So this is *not* a slow drift that accumulates over many steps; it is settled
  within one SR update.  Since `R_C6` is linear in the breaking amplitude, the
  first update carries a **15% symmetry-breaking component**.
* **The S split is not yet readable.**  At `t = 0` — *before any SR step* — the
  three families already read 6.135 / 5.975 / 6.412, a spread of **0.437**.  That
  is the estimator's own noise, not a split, and it is the liquid's-own-shell-
  spread trap this project has paid for once already.  A later "the families
  separated" claim must clear this floor; single-seed runs report no floor at
  all rather than a floor of zero.

  **Where the floor must NOT be read from.**  The script's `t0_floors` block
  tries to derive it from the arms' `t = 0` values and returns `sd = 0.000` for
  every S key — because the S walk is cached **per state**
  (`s_cache`, keyed on `state_digest` of the orbital block), and every joint arm
  starts from the *same* `theta0`.  So the arms do not take independent walks at
  `t = 0`; they share one, and the reported `sd = 0` is a caching identity, not a
  measurement.  A floor of zero would then license any later split whatsoever.
  The floor that *is* measurable is the **within-arm scatter at late `t`**, where
  the state is nearly stationary: over `t = 15..30` it is `sd ≈ 0.5` for each
  `S(Gk)` and `sd ≈ 0.31..0.59` for `S_split`.  That is the number a split claim
  has to clear, and §9.3 uses it.  (The `|f|` floor is unaffected: `force` comes
  from `hist`, which *is* per-arm, so its floor — 20.8 at these three seeds — is
  a real measurement.)

`E/N` is descriptive only: at `quick` a step has 14 snapshots, the SEM is naive
(autocorrelation ignored) and consecutive steps share the walker, so `E_err` is a
**lower bound**.  Resolving `ΔE ≈ 0.01` per particle needs `n ≈ 500`, i.e.
`nsweep ≈ 1000` — 25× the cost.  `E` is never the decisive observable here.

### 9.3 Layer 2: thirty steps, three seeds

    python examples/phase_competition_2/crystal_c6_trajectory.py \
        --steps 30 --seeds 0,1,2

Store: `results/phase_competition_2/c6_trajectory/N36__ll_rotation__quick__rs90__nmax1__steps30__nsweep40/`.
Cost 647 s / 907 s / 2483 s per seed — the box's load, not the code (the same
30-step job ran at 21.6 s/step and at 83 s/step).  `--steps 30` is the same step
*shape* as the reference that drifted, extended 3×.

| seed | `C6(0)` | **`C6(1)`** | `C6(T=30)` | peak (at `t`) | Stage A | Stage B | `‖Δv_1‖` |
|---|---|---|---|---|---|---|---|
| 0 | 3.03e-14 | **1.475e-01** | 3.359e-01 | 3.421e-01 (25) | `CRYSTAL` | `DRIFT_SETTLED` | 0.632 |
| 1 | 3.03e-14 | **1.292e-01** | 3.087e-01 | 3.126e-01 (29) | `CRYSTAL` | `DRIFT_SETTLED` | 0.592 |
| 2 | 3.03e-14 | **1.336e-01** | 2.977e-01 | 2.996e-01 (26) | `CRYSTAL` | `DRIFT_SETTLED` | 0.582 |

`C6(1) = 0.137 ± 0.010` (**7.0%** seed-to-seed spread); `C6(30)/C6(1) = 2.30`.

Stage A is `CRYSTAL` for all three — `bragg` 9.48 / 10.50 / 10.78 against
`3 × 1.109`, contrast 5.30 / 5.72 / 5.65 — and `bragg` *rose* from 6.79. So the
crystal survives and sharpens; `C6` is measuring symmetry, not disorder.

**1. Layer 1 is confirmed inside the run, not inherited.**  The corrected seed
reads `3.032e-14` at this operating point with the conjugated-overlap control at
`5.944e-01` (separation 2.0e13), a second context agreeing, and gauge invariance
to `3.3e-18`.  "SR broke a correct seed" is therefore the right framing.

**2. The first update does it, on every seed.**  `C6` goes from `3e-14` to
`0.137 ± 0.010` in **one** update and then only doubles over the next 29 steps.
`‖Δv_1‖ = 0.63` against the seed's own `|v|_mean = 0.586` — the first orbital
step moves the block by more than its mean amplitude.

**3. The five-vector is flat and `θ → −θ` symmetric, and it is not the slide
shape.**

| | 60° | 120° | 180° | 240° | 300° |
|---|---|---|---|---|---|
| seed 0, `t=1` | 0.1191 | 0.1467 | 0.1294 | 0.1475 | 0.1193 |
| seed 1, `t=1` | 0.1292 | 0.1280 | 0.0924 | 0.1278 | 0.1284 |
| seed 2, `t=1` | 0.1217 | 0.1336 | 0.1146 | 0.1334 | 0.1221 |
| seed 0, `t=30` | 0.2979 | 0.3077 | 0.3359 | 0.3073 | 0.2990 |

The pairs `{60°,300°}` and `{120°,240°}` agree to **2e-4** and **8e-4** at
`t = 1` — the rotational pattern is symmetric to ≤0.5% at every step and every
seed.  **No residual subgroup survives**: a `D2` would drive 180° to zero and a
`C3` would drive `{120°,240°}` to zero; neither happens.  And §10's slide
prediction is `R(180°)/R(60°) = 2`; the measurement is **1.09** at `t = 1` and
**1.13** at `t = 30`.  So the residual is *not* the shape a rigid slide makes —
which matters, because §8's `bragg`-up/`C6`-up pattern was the reason to suspect
one.

**4. The registry phases wander, but they are not a rigid slide.**  All three
seeds have `reg_moved = True` (swing 0.276 / 0.380 / 0.254 rad against a floor of
0.058 / 0.061 / 0.054, i.e. **4.7–6.2×** the measurement's own coherence-derived
resolution) but `reg_monotone = False`.  And the translation identity
`φ(g1+g2) = φ(g1) + φ(g2)` — which a pure slide satisfies exactly — is violated at
`0.095 / 0.071 / 0.102 rad`, against its own noise of
`√3 × 0.020 ≈ 0.034 rad`: **2.0–2.9×**.  So there is a genuine non-translation
(deformation) component, and the phases are neither frozen nor sliding rigidly.
`REGISTRY_DRIFT` therefore does **not** fire, and **(e) is not the mechanism** at
this step count.

  *Honesty note on that check.*  `identity_ok`'s threshold is
  `reg_max_identity < 1.0` rad — far looser than the 0.25–0.38 rad swing it is
  meant to police.  Passing it is a shape bound, not a consistency test.  The
  informative comparison is the one above, against `√3 ×` the phase error.

**5. The `E` floor is measured, and it is 3× the naive error bar.**  At `t = 0`
the three arms are at the *same* `theta0`, and their `hist` energies read
`−44.2377 ± 0.1619`, `−45.1530 ± 0.1178`, `−44.9323 ± 0.2067` — a **seed-to-seed
scatter of 0.478 per particle against a mean naive SEM of 0.162**, i.e. the
naive `E_err` underestimates by **2.95×**.  (The `t = 0` S-walk `E` *is* shared
via `s_cache`, so it shows the same value three times and is not evidence of
anything.)  This is the plan's "`E` is never the decisive observable" made
quantitative: at this density no per-step `E` claim survives.

**6. The closing states are energetically degenerate while their `C6` differs by
13%.**  The 400-sweep closing walks read `−45.8271 / −45.8248 / −45.8074` per
particle — `sd = 0.0107` — while `C6(T)` spans 0.298–0.336.  The energy is
**flat** along the breaking direction to 0.1%.

That last number is the honest limit of Layer 2.  **Flat energy is what (b)
predicts** (near-flat symmetry-breaking directions, so the optimiser drifts along
them at no cost) **and also what (c) predicts** (near-degenerate symmetry-broken
minima).  Layer 2 does not separate them, and it cannot: it never compares the
broken states to a `C6`-**symmetric** state.  Neither can it separate noise from
bias — the 7% seed-to-seed spread in `C6(1)` is consistent with a random draw in
a high-dimensional null space, and equally consistent with a fixed bias of
nearly constant magnitude.  Those are Layers 3a/3b and 3d.

**7. The structural setting that makes the orbital update regulariser-set.**
Each SR step builds the metric from `n = 14` snapshots in a `77`-parameter space
(`sr.py:295-304`), so `S_x` has **rank ≤ 14** — 63 of 77 directions are numerically
null.  Measured: `eig_min ∈ [−4.5e-12, −8.5e-16]` (negative, confirming the rank
deficiency) against `eig_max ∈ [0.58, 5.87]`, with `eps = 1e-3` and
`cond(S_reg) ∈ [583, 5874]`.  Since the orbital block's eigenvalues sit at
`≤1e-6` while `eps = 1e-3`, the orbital update
`Δx = −τ·S_reg⁻¹f_x` is, in those directions, a plain damped step along the
**raw** force — and the raw force carries the finite-sample noise that has no
reason to lie in the `C6`-symmetric tangent sector.  This is mechanism 3c's
premise, measured rather than assumed; it is *not* evidence for it, and 3b/3c are
what would test it.

**One cheap check is still outstanding.**  The pre-registered anchor (plan step 3)
is a run at `--steps 10 --seeds 0 --budget quick`, whose closing `C6` must land on
the killed reference's independently-measured `0.209`.  It has **not** been run.
The `t = 10` rows of the table above are *not* it: `tau_i = tau/(1 + xi·i/steps)`
depends on `steps`, so `--steps 30` and `--steps 10` agree only at `i = 0` and
diverge from `i = 1` on.  The anchor is a genuinely different trajectory and
remains the one unspent validation of the seed, the RNG stream, the protocol and
the estimator in a single number.

---

## 10. Reading a C6 residual — the interpretation key

Three facts that make the difference between a number and a claim.

**C6 is not an order parameter.**  `v = 0` (the filled lowest Landau level) is
invariant under *every* rotation, so the liquid reads `c6 ≈ 2.6e-14`.  **A low
C6 is also the liquid**, which is why Stage A asks "is it still a crystal?"
before Stage B reads the symmetry at all.

**The five-vector carries three independent numbers, not five.**  The
per-rotation pattern is symmetric under `θ → 360° − θ`, so `{60°, 300°}` and
`{120°, 240°}` are paired and only the pairs, plus 180°, are independent.  The
stored `smoke` reference reads 0.0863 / 0.0804 / 0.0843 / 0.0806 / 0.0866 — the
pairs are 0.0865 and 0.0805, straddling 180°.  No residual subgroup survives: a
clean D2 would leave 180° near zero, a C3 would leave `{120°, 240°}` near zero.
**A difference within a pair is not physics.**

**A sharper crystal with a worse C6 residual is a slide, not disorder.**  The
residual is measured about the **torus origin** (`c6_context` rotates the sample
points), so a crystal that has translated is no longer invariant about it.  The
registry observable separates the two, and it rides the Layer-2 S-walk for
nothing: `ρ(q) = Σ_j e^{i q·r_j}` picks up `e^{i q·d}` under a translation `d`,
so the phases at `g1`, `g2` and `g1+g2` move by `g1·d`, `g2·d` and `(g1+g2)·d`.
The third is therefore the **sum** of the first two, mod `2π` — an identity that
holds exactly for a lattice that **slid** and fails for one that was
**strained**.  `reg_identity` is that residual, and `coherence = |⟨ρ⟩| / ⟨|ρ|⟩`
says whether the phase means anything at all: a phase read where there is no
peak is noise.  `stage_b` requires the phase to have **moved** by more than the
resolution its own coherence implies — a constant series is monotone, so
without that test a lattice that never slid reads as the strongest possible
slide.

### What a slide predicts

`R_θ V_d = V_{R_θ^{-1} d}`, so for small `d` the five-rotation residual should
follow `2|d|·sin(θ/2)` — **largest at 180°, smallest at ±60°**.  The stored
`smoke` pattern is nearly flat with 180° *in the middle*, which is **not** that
shape; a slide large enough to saturate the residual would flatten it.  So the
registry-drift hypothesis is established in neither direction by the stored
data, and §9 measures the registry instead of inferring it.

