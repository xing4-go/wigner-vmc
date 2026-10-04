# Gaussian → LL projection fix: one controlled post-fix validation

`rs = 75`, `phase = crystal`, `ansatz = ll_rotation`, `n_max = 2`, `N = 36`,
budget `reproduction` — the same point, the same protocol, as the historical
comparison, with one line of source changed.

---

## STATUS

| | |
|---|---|
| **STATUS** | **PASS** |
| **defect fixed** | `<G\|phi> -> <phi\|G>` (`src/wigner_vmc/vmc/sr.py:219`) |
| **corrected seed C6** | **PASS** (occupied 36-dim space invariant under R(60°…300°) to 3.0e-14) |
| **SR preserves C6** | **PASS at the endpoint; NOT REACHED for intermediate steps** (per-step states are stored by neither run — see §8) |
| **old E_C/N** | `-37.255321159727 +- 0.053365041` |
| **corrected E_C/N** | `-37.801298809724 +- 0.015899913` |
| **Delta E_C** | **`-0.545977650`** (corrected − old) |
| **significance** | sigma_delta `0.055683`, **z = -9.81** — not consistent with zero |
| **liquid affected** | **NO** (Delta E_L = `0.000000000000` exactly; the liquid's stored snapshots are *byte-identical*, see §9) |
| **implication for previous phase results** | **REVALIDATION REQUIRED** (Case B — see §10) |

Two things this table does not say, and §8/§10 say:

* the "PASS" on SR and C6 is an **endpoint** statement. No intermediate SR state
  survives in either run, so a mid-SR excursion that healed by the production
  walk cannot be excluded.
* "REVALIDATION REQUIRED" is scoped to **this test point and this ansatz
  family**. It is not a claim that the whole phase diagram moves.

---

## 1. The diagnosis this supersedes

`ANISOTROPY_DIAGNOSIS_rs75_nmax2.md` has been corrected in place: its
observations stand, its final mechanism is marked `SUPERSEDED / WITHDRAWN`, and
the original text is preserved verbatim under inline `WITHDRAWN` notes rather
than deleted. What it claimed (`GEOMETRY BUG`; "the Landau-level Bloch manifold
itself is **not** six-fold covariant") is refuted. What replaces it is
`IMPLEMENTATION DEFECT: reversed complex conjugation in the Gaussian → LL
projection`.

**The LL Bloch manifold itself IS C₆-covariant.** Measured, not asserted: under
the corrected projection the manifold maps monomial-of-unit-modulus onto itself
under all five rotations with `max ||ratio| - 1| <= 3.95e-13` over all 108
`(k, n)`, quadrature-free. The old §7b `T_k` test reproduces its 56% number and
proves nothing, because `T_k` is gauge-dependent and a magnetic rotation acts on
the state by a gauge transformation. The correct test is invariance of the
occupied **subspace**, which is what §3's regression test now measures.

---

## 2. Source hashes

`src/wigner_vmc/vmc/sr.py`, line 219 — the only line that changed.

| | line 219 | md5 | sha256 | bytes |
|---|---|---|---|---|
| **AFTER** (corrected, on disk, verified live) | `ov = np.einsum("i,ikn->kn", G, bloch.conj())` | `f5462a841d66d9906d0f29ce553426cc` | `18b45dc6314864a22d91a4b559d1aec15b28a3ba96c3d5b10f493ca3e3941266` | 17231 |
| **BEFORE** (defective) | `ov = np.einsum("i,ikn->kn", G.conj(), bloch)` | `73314a94563d26e60616f85275fdec9e` | `e20bd676d79f85733d8c1abde1f910a756804672c8313659ca3ba1af56e3d688` | 17231 |

**The BEFORE digest is reconstructed, not recovered.** No pre-fix copy of the
file was kept. The fix is length-preserving (`|GOOD| == |BAD| == 44`), so
inverting it on the current bytes reproduces the predecessor exactly *if* no
other edit was folded into the same session. That condition is asserted as
provenance and is **not** independently provable from the tree; the record says
so rather than presenting a reconstructed hash as a saved artifact. What *is*
demonstrated: inverting the swap changes exactly 1 line of 361, and that line is
219.

Also recorded, in the corrected run's `run_metadata.json` (`projection_fix`
block): the before/after expressions, the source md5/sha256/bytes, the runner
`argv`, and why the defect breaks C₆ (`exp(-i n theta) == exp(+i n theta)` only
at `theta = 180°`, which is exactly why C₂ survived and C₆ did not).

---

## 3. Regression tests

New file `tests/test_projection_convention.py` — **10 collected, 10 passed**
(396.61 s). The expected values are not produced by the function under test:

* **independent oracle.** The projection is recomputed by direct Gauss–Legendre
  quadrature of `\\int d²r φ*(r) G(r)` at 60 nodes, a different numerical route
  from production's fixed 121-point grid. Agreement with production:
  **8.955e-14**. The conjugated oracle sits at **1.660** — a gap of 1.66, so the
  test can actually discriminate the two conventions.
* **bracketing, so the residual is *shown* to be truncation.** The same oracle
  at 40 nodes is off by `2.184e-06` and at 60 nodes by `8.955e-14`; the test
  asserts the coarse error lies in `(1e-8, 1e-3)` *and* the fine one below
  `1e-10`. A single tight tolerance would not distinguish "converged" from
  "accidentally close". Measured convergence ladder (nodes/dim → max error):
  24 → `2.525e-02`, 40 → `2.184e-06`, 60 → `8.955e-14`, 80 → `2.422e-15` — six
  orders per doubling, which is what a quadrature residual looks like and not
  what a convention error looks like.
* **the negative control is asserted to have teeth.** A separate guard requires
  `|prod - conj(oracle)| > 1.0`, i.e. the test *proves* the conjugated oracle is
  far away rather than assuming it.
* **gauge-covariant C₆ test.** The occupied 36-dimensional subspace is checked
  for R-invariance under R(60°…300°), not the individual orbitals — a Slater
  determinant is R-invariant iff its occupied space is, and demanding literal
  orbital equality is invalid under a magnetic rotation.
* **production wiring**, `atol=0, rtol=0`: the seed the API actually builds is
  the one the oracle predicts.

---

## 4. Mutation proof

The exact defect was restored at line 219 (bytes in, bytes out) and the new file
re-run: **5 of the 10 guards fire** — both convention tests, the oracle test, and
both C₆ tests. Source restored **byte-exactly**; md5/sha256/bytes re-verified
after restoration, and independently after the fact.

The most informative line is the C₆ residual under the restored defect:

```
R60=5.805e-01  R120=5.839e-01  R180=1.089e-14  R240=5.805e-01  R300=5.839e-01
```

That is the C₂-only survival mechanism measured directly: 180° is clean to
1e-14 and the other four rotations are broken at ~0.58, matching the
`theta = 180°` argument exactly. Mutation testing ran with no other pytest or VMC
job in flight.

---

## 5. Test suites, and the classification of every failure

| run | result |
|---|---|
| targeted (`tests/test_projection_convention.py`) | exit 0 — **10 passed**, 0 failing |
| behavior (`-m behavior`) | exit 1 — **2 failed, 333 passed**, 170 deselected (500.70 s) |
| full (`tests`) | exit 1 — **2 failed, 503 passed** (1311.67 s) |

The full-suite failure set is byte-identical to the behavior set (`diff`:
`IDENTICAL`). There is no third failure anywhere in 505 tests and no `ERROR`.

| failure | classification |
|---|---|
| `tests/test_sr_against_legacy.py::TestGaussianOverlapSeed::test_the_seed_agrees` | **EXPECTED — historical defective projection encoded in oracle** |
| `tests/test_sr_against_legacy.py::TestGaussianOverlapSeed::test_the_slice_from_a_wider_basis_is_what_runs` | **EXPECTED — historical defective projection encoded in oracle** |

Both compare the clean seed against `qhvmc_engine_llrot.py:424`, which carries
**the identical reversed conjugation** and is `status: 'frozen'` in
`legacy_frozen/LEGACY_MANIFEST.json` (sha256 `3bd7deac…5677`, 24644 bytes). The
failure is the conjugation and nothing else — the printed arrays are equal in
modulus and conjugated in phase, with `Max relative difference among violations:
2.`, the signature of `z -> z̄`. §11 forbids repairing the frozen file and §5
forbids weakening the assertions, so these two stay red **by design**. No
tolerance was loosened and no expected value was rewritten.

**No UNEXPECTED regression. The §5 gate passes.** Frozen historical data was not
touched (§11, below).

---

## 6. Seed gate (zero-cost, before any SR)

| | value |
|---|---|
| defective seed | `[0.002342, 0.002342, 0.087323]` — one-axis enhancement |
| corrected seed | `[0.087323, 0.087323, 0.087323]`, spread **8.3e-17** |
| corrected seed C₆ residual | **3.0e-14** (all five rotations, occupied subspace) |

The anomalous one-axis enhancement is gone. The seed was C₆ before SR started,
which is what made the SR question (§8) worth asking.

---

## 7. The one corrected run

```
python examples/figure_construction/run_projection_fix_point.py \
    --liquid-rs 75 --crystal-rs 75 --crystal-nmax 2 --budget reproduction
```

Exactly one `rs`; no other `n_max`; no extra sweeps; no other ansatz. The runner
refuses to start unless the source is the known corrected file (md5/sha256/bytes
+ line), and redirects the recipe's `RESULTS`/`FIGDIR` bases so the corrected
point lands in a new namespace.

Protocol verified field-for-field against the historical run before the run
started: budget `reproduction`, `init_id` 0 / `init_L0` 0.4, crystal walk seed
13, liquid walk seed 11, `rng_seed` 0, crystal `n_max` 2 / `n_bands` 3, κ =
53.033008589 = `rs/sqrt(2)`, SR 8×150 (equil 50, snapshot_every 3), production
walk 1500/500. **The projection line is the only difference between the two
runs.**

| | old (defective) | corrected |
|---|---|---|
| crystal E/N | `-37.255321159727 +- 0.053365041` | `-37.801298809724 +- 0.015899913` |
| crystal acceptance | 0.487741 | 0.480648 |
| crystal snapshots | 1000 | 1000 |
| liquid E/N | `-37.599265371149 +- 0.017694411` | `-37.599265371149 +- 0.017694411` |
| liquid acceptance | 0.421556 | 0.421556 |

The corrected crystal's error bar is 3.4× smaller. Wall-clock is **not**
comparable between the two runs (the liquid took 125.4 s historically and 302.5 s
now for a bit-identical result — machine load, not physics), so `seconds` is
excluded from every comparison below.

---

## 8. Symmetry through SR

`S₁, S₂, S₃` are the three first-shell families (`g1+g2`, `g1`, `g2`); at the
first shell all six vectors share one `|q|`, so for an isotropic state the three
family means differ only by estimator noise.

**What §8 asked for cannot be fully delivered, for either run.** `run_point`
returns only `(snaps, R)`; `api.Optimization` holds `E_total`/`acc` per step in
memory, and per-step *states* — which is what `S₁,S₂,S₃` need, since they are
functions of configurations and not of energy — are retained nowhere. So the
possible rows are: the seed, and the production state.

| state | S₁ | S₂ | S₃ | spread (max−min)/mean | max/min |
|---|---|---|---|---|---|
| corrected **seed** | C₆ to 3.0e-14 (§6) | | | | |
| early SR | **NOT AVAILABLE** — no run stores per-step states | | | | |
| middle SR | **NOT AVAILABLE** | | | | |
| corrected **production** | 9.5065 ± 0.3158 | 10.0870 ± 0.2786 | 10.2272 ± 0.3998 | **0.0725** | 1.0758 |
| defective **production** | 2.2765 ± 0.3415 | 2.3405 ± 0.4364 | 8.1405 ± 0.4460 | **1.3790** | 3.5759 |

Errors are block standard errors (10 blocks × 100 snapshots; a 5 × 200 split
gives the same means and is reported in the working log). Pairwise
significances:

| | corrected crystal | defective crystal | liquid (control) |
|---|---|---|---|
| z(S₁−S₂) | −1.38 | −0.12 | +0.90 |
| z(S₁−S₃) | −1.41 | **−10.44** | +0.59 |
| z(S₂−S₃) | −0.29 | **−9.30** | +0.06 |

The **control matters**: on the liquid, where isotropy is known, this same
estimator on the same torus with the same snapshot count reports a 7.18% spread
and all pairwise `|z| <= 0.90`. The corrected crystal's spread is 7.25% with all
pairwise `|z| <= 1.41` — statistically the *same* spread the estimator produces
on a state known to be isotropic. The defective crystal's spread, by contrast,
was 138% with two pairs above 9σ.

**Verdict.** The corrected seed is C₆. The corrected production state shows no
large directional splitting, and its residual first-shell anisotropy is
indistinguishable from the estimator's own noise floor. So SR did **not** produce
the large splitting that would trigger §8's STOP. **Caveat, stated plainly:** with
only the two endpoints available, a transient mid-SR excursion that healed before
the production walk cannot be excluded — that is the sense in which the
intermediate rows are `NOT REACHED` rather than `PASS`. The state was **not**
symmetrised by hand.

Note also that the first-shell *weight* moved a long way (S₁+S₂+S₃: 12.76 → 29.82)
while the global `S(q)` peak barely moved (35.6441 → 35.6407): the defect was
scattering first-shell weight into a uniaxial pattern, not creating the peak.

---

## 9. Old vs corrected

`ΔE_C = E_C^corrected − E_C^old = -0.545977650`, `σ_Δ = 0.055683`, `z = -9.81`.

**`E_L` is unaffected, so `Δ(δE) = ΔE_C` — and this is verified, not assumed.**
Source proof: `gaussian_overlap_seed` has exactly one call site in `src/`
(`api.py:715`) and `v_from_overlap` exactly one (`api.py:718`), both inside
`crystal_v0` (`api.py:708`), whose only caller is `api.py:735`, inside
`if cfg.phase == "crystal":`. The liquid branch sets `v0 = np.zeros(...)`. The
liquid cannot reach the changed line. Measurement agrees, and agrees harder than
"agrees":

* `liquid/run.npz` — **byte-identical** (same md5 old and new).
* In every shared archive (`density_xy.npz`, `pair_correlation_xy.npz`,
  `structure_factor_xy.npz`) every geometry array and every `*_liquid` array is
  **bitwise equal**; only the `*_crystal` arrays differ. That is why those files'
  file-level md5 changed — the file carries both phases.
* `liquid/run.json` differs in **exactly two fields**: `seconds` and
  `sr_seconds`. Every physics field is identical.
* ΔE_L = `0.000000000000`; every liquid observable diff is exactly `0.0`.

Structural observables, crystal (old → corrected):

| observable | old | corrected | change |
|---|---|---|---|
| S₁ (g1+g2) | 2.27648 | 9.50653 | +7.230 |
| S₂ (g1) | 2.34050 | 10.08703 | +7.747 |
| S₃ (g2) | 8.14046 | 10.22721 | +2.087 |
| anisotropy (max−min)/mean | 1.37896 | 0.07250 | −19.0× |
| n(x,y) | — | max abs diff 2.0865, rms 0.6989 | |
| g(x,y) | — | max abs diff 0.9579, rms 0.3745 | |
| g(r) | — | max abs diff 0.2612 | |
| S(q) | — | max abs diff 8.3375, rms 0.6906; peak 35.6441 → 35.6407 | |

Liquid: every observable diff exactly `0.0`.

**Not recoverable from stored products — for either run:** the SR energy
trajectory, the initial seed energy, and any mid-SR `S₁,S₂,S₃`. The two records
have the *same schema* (diffing the crystal `run.json` keys: no key added, no key
removed; only `acceptance`, `energy_per_particle`, `error`, `seconds`,
`sr_seconds` differ), so nothing was captured for the corrected run that the
historical one lacked. The one extra datum that exists is on the corrected side
only, and only in console output: its joint SR spanned `E_total -1328.0169 →
-1361.5071`, with an SR-sample minimum of `-37.819642` per particle — which the
recipe explicitly labels "NOT the answer", because the production walk is the
estimator and the SR minimum is evaluated on the optimisation sample.

---

## 10. Interpretation

**Case B — the energy is affected.** `|ΔE_C| = 0.546` at 9.8σ, with the C₆
verdict clean and no SR re-breaking. Case A (structural only) is excluded by
significance; Case C and D do not arise.

Within this test point the consequence is sharper than "the energy moved":

| | old | corrected |
|---|---|---|
| δE = E_C − E_L | **+0.343944** (liquid lower, z = +6.1) | **−0.202033** (crystal lower, z = −8.5) |

The corrected calculation **reverses the crystal–liquid ordering** at
`rs = 75`, `n_max = 2`, and the reversal is significant on both sides. The
defective state was not merely higher in energy: it was a uniaxial, non-C₆
object (S₁,S₂,S₃ = 2.28/2.34/8.14) sitting *above* the liquid, whereas the
corrected state is an isotropic one sitting *below* it.

So: **previous crystal-energy and phase-competition results require systematic
revalidation.** Per the brief, that revalidation is **NOT** started here — no
phase diagram, no `rs` scan, no other `n_max`, no ansatz change, no finite-size
scaling. This document reports one point.

One honest limit on the interpretation: `ΔE_C` is a **state** difference, not a
pure energy shift of one fixed state. The corrected SR converged somewhere else
(its error bar is 3.4× smaller, its acceptance slightly lower). Separating "the
same state computed correctly" from "a different, better local minimum" needs the
old trajectory, which does not exist.

---

## 11. Provenance

Nothing historical was modified. Re-verified at the end of this work:

* all 8 data files of the historical point under
  `results/figure_construction/structure/liquid_rs75__…__reproduction/` match
  `HISTORICAL_BEFORE.md5` — **OK**
* all 6 historical figures under the matching `figures/…/structure/…` dir match
  the same manifest — **OK** (14/14 across the two directories)
* the frozen legacy tree: `legacy_frozen/verify_manifest.py` → **PASS**, 1607
  assets, 0 missing, 0 added, 0 changed, 0 mtime moved

Corrected results use a new namespace and carry an explicit stamp:

```
projection_convention = "phi_bra_gaussian_ket"
projection_bug_fixed  = True
```

plus a `projection_fix` evidence block (before/after expressions, source hashes,
runner argv, defect statement, the `theta = 180°` reason C₂ survived). Corrected
and legacy-regression results are never silently mixed: the historical tree
remains the record of the old implementation, and the two EXPECTED legacy
failures are the visible seam between them.

---

## 12. Figures

Regenerated into the corrected namespace, **not** over the defective ones:

```
figures/figure_construction/projection_fix/liquid_rs75__crystal_rs75__llrot_nmax2__reproduction/
    density_xy.png
    pair_correlation_xy.png
    structure_factor_xy.png
    combined_structure_summary.png
```

The defective-run figures under `.../projection_fix`'s sibling `.../structure/`
are retained untouched for before/after comparison (md5-verified above).

---

## 13. Unresolved items

1. **Intermediate SR states do not exist, for either run.** §8's per-step table
   cannot be filled in — not for the corrected point and not retrospectively for
   the historical one. Answering "when did the splitting appear / heal" requires
   a run that stores per-step states, which is a change to `run_point`, not a
   measurement.
2. **`ΔE_C` mixes a fix with a different minimum.** See §10.
3. **The legacy engine still carries the defect** at
   `qhvmc_engine_llrot.py:424`. Deliberately not repaired: it is frozen, and §11
   makes it the historical record. Two legacy-comparison tests therefore remain
   red by design, and any result compared against that engine inherits the
   defect.
4. **The reconstruction of the BEFORE source hash** rests on an assertion
   (no other edit in the same session) that the tree cannot prove. See §2.
5. **Cosmetic, flagged not fixed:** the liquid's stored record carries
   `"ansatz": "ll_rotation"` and `"n_bands": 2`, which are schema-uniformity
   carry-throughs rather than liquid physics — the API rejects any other ansatz
   string for a non-crystal phase (`api.py:436`), so it is the only admissible
   value, and the padding is provably inert. The truthful fields in the same file
   are `orbital_ansatz: "filled_lll_fixed"`, `orbital_sr: false`,
   `ll_indices: [0]`, `nmax_is_a_physical_parameter: false`. Changing the schema
   would touch frozen JSON, so it was left alone.

## 14. What was deliberately NOT done

No phase diagram; no `rs` scan; no `n_max` 1/3/4; no ansatz change; no manual
symmetrisation; no modification of historical data; no finite-size scaling. One
point, measured once. Waiting on the decision that follows from the corrected
`E_C`, `ΔE_C` and C₆ results above.
