# Anisotropy diagnosis: `S(q_x, q_y)` at `r_s = 75`, `n_max = 2`

Scope: the one-direction enhancement visible in `structure_factor_xy.png` for the
single-point structural recipe
`liquid_rs75__crystal_rs75__llrot_nmax2__reproduction`.

Every number below comes from **existing artifacts** — the stored clean-VMC
snapshots, the geometry objects, and one deterministic one-body density built from
the stored initialization parameters. **No new VMC or SR run was started** (§14),
and **no production file was modified** (§12). The frozen legacy tree was verified
this session: `legacy_frozen/verify_manifest.py` reports 1607 assets, 0 missing,
0 added, 0 changed, 0 mtime moved — **PASS**.

---

> # ⚠ SUPERSEDED / WITHDRAWN — read this before anything below
>
> **Corrected 2026-10-03.** The *observations* in this report stand; the **final
> mechanism does not**. The text below is preserved **as originally written** —
> nothing has been silently erased — because the reasoning is part of the record
> and because the way it went wrong is instructive. Every place where the original
> mechanism is stated now carries an inline **`WITHDRAWN`** note giving the
> correction.
>
> ## What was withdrawn
>
> | originally claimed | status |
> |---|---|
> | `GEOMETRY BUG` — a geometry/convention defect in the Bloch construction | **WITHDRAWN** |
> | "the Landau-level Bloch manifold itself is **not** six-fold covariant" (§7b, *First cause* 4) | **WITHDRAWN — refuted** |
> | the magnetic-translation cocycle `exp(iπ a₁a₂)` is the proximate object (*First cause* 5) | **WITHDRAWN** |
> | the *delivered object* label `SEED/ANSATZ ANISOTROPY` | **STILL CORRECT** |
>
> ## What replaces it
>
> **`IMPLEMENTATION DEFECT: reversed complex conjugation in the Gaussian → LL
> projection`** — one line, `vmc/sr.py:219`:
>
> ```python
> # was  (defect):  Σᵢ G* φ   = ⟨G|φ_{k,n}⟩,  the CONJUGATE of the projection
> ov = np.einsum("i,ikn->kn", G.conj(), bloch)
> # now  (correct): ⟨φ|G⟩ = Σᵢ φ* G
> ov = np.einsum("i,ikn->kn", G, bloch.conj())
> ```
>
> **The Landau-level Bloch manifold itself IS C₆-covariant.** Measured, not
> asserted: on the corrected projection the seed's occupied 36-dimensional space is
> invariant under all five rotations R(60°…300°) to **3.0e-14**, and the manifold
> maps monomial-of-unit-modulus onto itself under each rotation with
> `max ||ratio| − 1| ≤ 3.95e-13` over all 108 (k, n), quadrature-free. The bare
> torus was never the problem either (§8 said so correctly: integer M ∈ GL(2,ℤ),
> det +1, mesh R-closed to 4.44e-15).
>
> **Why the old mechanism was wrong, and why §7b seemed to prove it.** §7b tested
> the *rotation-invariant trace* `T_k(r) = Σₙ |ψ_{k,n}(r)|²` of the **Bloch basis**
> — the raw `LandauLevelBasis` orbitals — and found it changes by 56% under R(60°).
> That measurement is real and reproduces. What it does **not** show is a defect,
> because `T_k` is computed from the orbitals in the *code's own gauge*, and a
> magnetic rotation is implemented by a gauge transformation: the rotated orbital
> is the original orbital times a **r-dependent phase**, not a copy of it. So
> `|ψ_{k,n}(R⁻¹r)|² ≠ |ψ_{k,n}(r)|²` even for a perfectly covariant manifold, and
> §7b's test was comparing two objects that a legal gauge change is allowed to
> differ between. The cocycle `exp(iπ a₁a₂)` is likewise a **convention, not an
> observable** — §7b's own *Caveat* said so, and the caveat was right; the
> conclusion drawn past it was not. The correct test is the one this report never
> ran: **the invariance of the occupied SUBSPACE**, gauge-covariantly, which is
> what a Slater determinant actually requires (a determinant is R-invariant iff its
> occupied space is, not iff each orbital maps to another orbital). Run that way,
> the manifold passes to 3.0e-14.
>
> **The one thing §7b got right** is that the *seed* was $D_2$ — §7 is exact and
> reproduces to <5e-7. The conjugation flips the band phase `nθ → −nθ`, and
> `e^{−inθ} = e^{+inθ}` **only** at θ = 180°. That is exactly, and only, why C₂
> survived and C₆ did not — the same $D_2$ pattern §7b then (wrongly) attributed to
> the manifold.
>
> ## Consequences, stated honestly
>
> - §7's seed numbers `[0.002342, 0.002342, 0.087323]` (ratio 37.3) are
>   **reproduced exactly** as the *defective* seed — the report's measurement was
>   sound. Corrected, they become `[0.087323, 0.087323, 0.087323]`, spread
>   **8.3e-17**.
> - The §11 label is unchanged and still correct: this was never physics.
> - The candidate fix proposed at the end of this report (rebuild the manifold with
>   a hexagonal cocycle) is **not the fix**, and building it would have changed the
>   ansatz to paper over a one-line bug. The fix that was applied is the one-line
>   conjugation change above.
> - The proposed regression test at the end of this report (`T_k(R⁻¹r) == T_k(r)` for
>   R = 60° at k = 0, "fails today by 5.6e-01") is **withdrawn as a test of this
>   defect**: it is gauge-dependent and would fail even on a correct manifold. The
>   regression test that was actually added tests the occupied subspace.
> - The structural observables of this ansatz at the historical point were computed
>   with the defective seed; see `PROJECTION_CONJUGATION_FIX_VALIDATION.md` for the
>   controlled before/after at `r_s = 75`, `n_max = 2`.

---

## STATUS *(as originally written — see the banner above)*

~~**`SEED/ANSATZ ANISOTROPY`**~~ **`SEED/ANSATZ ANISOTROPY`** — the anisotropy is
real in the raw data, and it does **not** originate in the physics of the state.
The first asymmetric quantity is the
~~**Landau-level Bloch manifold itself**, which is not six-fold covariant; the
deterministic initializer inherits that asymmetry, and SR does not repair it.~~
**deterministic initializer, which inherited a reversed-conjugation projection; SR
does not repair it.**

- NOT a plotting artifact (§4)
- NOT a q-grid commensurability problem (§3)
- NOT a reciprocal-vector construction bug (§5)
- NOT an estimator bug (§9)
- NOT a finite-cell symmetry reduction (§8)
- NOT sampling uncertainty (§10, 13σ)
- NOT a genuine ground-state anisotropy (§11 — see the caveat in *First cause*)

> **`WITHDRAWN`.** The original text read: *"The mechanism is a **geometry/convention
> defect in the Bloch construction**, so `GEOMETRY BUG` is the accurate mechanism
> label; `SEED/ANSATZ ANISOTROPY` is the accurate label for the *delivered object*
> that carries it."* The first half is withdrawn (there is no geometry defect and no
> Bloch-construction defect); the second half is **confirmed**. The accurate
> mechanism label is **`IMPLEMENTATION DEFECT`**, and the accurate label for the
> delivered object is **`SEED/ANSATZ ANISOTROPY`** — the seed really was
> anisotropic, it just was not the geometry's fault.

---

## 1. The observation, frozen numerically

Raw estimator evaluated **exactly** at the six first-shell Wigner-crystal
reciprocal vectors $\pm g_1, \pm g_2, \pm(g_1{+}g_2)$, straight from the stored
snapshots (1000 configurations, `reproduction` budget). No image, no
interpolation, no grid was consulted.

```
family       G = (Gx, Gy)            nearest allowed q          |q-G|    S_liquid  S_crystal
G1 = g1+g2   (-2.332680,-1.346774)   (-2.332680,-1.346774)    2.220e-16   1.34378    2.27648
  (-G)       (+2.332680,+1.346774)   (+2.332680,+1.346774)    2.220e-16   1.34378    2.27648
G2 = g1      (-2.332680,+1.346774)   (-2.332680,+1.346774)    2.220e-16   1.26137    2.34050
  (-G)       (+2.332680,-1.346774)   (+2.332680,-1.346774)    2.220e-16   1.26137    2.34050
G3 = g2      (+0.000000,-2.693547)   (+0.000000,-2.693547)    4.441e-16   1.25151    8.14046
  (-G)       (-0.000000,+2.693547)   (+0.000000,+2.693547)    4.441e-16   1.25151    8.14046
```

$$S_i \equiv \tfrac12[S(+G_i) + S(-G_i)]$$

| | liquid | crystal |
|---|---|---|
| $S_1$ ($\pm(g_1{+}g_2)$) | 1.34378 | **2.27648** |
| $S_2$ ($\pm g_1$) | 1.26137 | **2.34050** |
| $S_3$ ($\pm g_2$) | 1.25151 | **8.14046** |
| $A = (\max S_i - \min S_i)/\overline{S}$ | 0.0718 | **1.3790** |
| $\max/\min$ | 1.074 | **3.576** |

**§2 gate.** $S_1 \ne S_2 \ne S_3$ in the raw data, so the fast "plotting defect"
path does not apply: the enhancement is present in the numbers before any
rendering. $A = 1.378957$ is a diagnostic, not an order parameter.

**±q consistency: exact.** $S(+G) = S(-G)$ to `0.00e+00` for all six targets, and
$\max_q |S(q) - S(-q)| = 0.000e+00$ over 400 random non-lattice q. This is the exact
identity $\rho_{-q} = \rho_q^{*}$ and it is satisfied bit-for-bit.

---

## 3. Allowed-momentum mismatch (commensurability)

Computed independently from the simulation cell, not from the plotted circles.

```
|g1| / |G1| = 6.000000000000          (g basis in the G basis = 6*I)
worst |q - G| over the six targets    = 4.441e-16
allowed_momenta tolerance             = 1e-6
all six are EXACT allowed momenta     = True
```

Geometry: $L_1 = 6A_1$, $L_2 = 6A_2$, hence $G = g/6$, hence **every** WC shell
vector is an allowed torus momentum. $\delta q_i \in \{2.2, 4.4\}\times10^{-16}$ for
all six. There is no commensurability penalty and no nearest-neighbour ambiguity —
498 allowed momenta lie within $|q| \le 5.2$, and the six targets are among them
exactly. *The enhancement is therefore not a "this q is the only available one"
effect: the shell sits inside a dense sea of allowed momenta (see the figure).*

---

## 4. Rendering pipeline

`structure_factor_field` in `phase_structure_single_point.py` evaluates the
estimator **at every grid point** — there is no interpolation, no Gaussian
smoothing, no `griddata`, no binning, and no duplicate q points. The displayed
panel values at the six targets are the raw values in the table above.

A dedicated raw-point figure was produced for §13:

```
figures/figure_construction/structure/liquid_rs75__crystal_rs75__llrot_nmax2__reproduction/
    structure_factor_raw_q_diagnostic.png
    structure_factor_raw_q_diagnostic.txt
```

One marker per allowed torus momentum, area and colour $\propto S(q)$, the six
targets as open circles, the nearest allowed momentum as a white cross (coincident
with the circles to $<5\times10^{-16}$), $q = 0$ omitted as kinematics.

---

## 5. Reciprocal-vector construction

An oracle basis was built directly from the lattice definition, by
$b = 2\pi\,\mathrm{inv}(A)^{\mathsf T}$, with no reference to the plotting helper:

```
b_i . A_j = 2 pi * I                        exact
max| b - torus.g |                          4.441e-16
max| G1 - b1/6 |                            5.551e-17
oracle six shortest vectors == Torus.wc_shell_vectors() as a set   True
```

and the cyan circles in the delivered figure **are** `torus.wc_shell_vectors()`.
Primitive-vs-supercell role confusion, transposes, inverse-transpose, row/column
and qx/qy swap are all excluded for this quantity.

---

## 6. Real-space symmetry of the same clean samples

`g(x,y)` from the same 1000 crystal configurations, at the six symmetry-related
nearest-neighbour displacements ($r = a = 2.693547$), read at the nearest grid
index (grid step 0.2476 $\ell_B$):

```
crystal g(a,theta):  theta =   0   60  120  180  240  300
                            1.893 1.558 1.647 1.893 1.558 1.647   spread 0.334
liquid  g(a,theta):         1.240 1.214 1.281 1.240 1.214 1.281   spread 0.067
```

Six-fold **positions**, unequal **heights**, with the $x$-reflection exact
($g(60) = g(240)$, $g(120) = g(300)$) — i.e. $D_2$ and not a noisy $D_6$.

**But this row cannot be read against zero, and the earlier draft of this section
claimed more than the estimator supports.** Both real-space estimators are
themselves binned on a square grid aligned with $L_1, L_2$, so they have their own
$C_6$ bias. Measured on a sample set that is sixfold **by construction** (200
crystal snapshots $\times$ the six $60^\circ$ rotations, pooled):

```
                          |F|^2(g1)   |F|^2(g2)   |F|^2(g1+g2)     rms
crystal (delivered)       0.008021    0.079459      0.002129      0.4508
liquid  (delivered)       0.000161    0.000035      0.000030      0.1408
D6-symmetrised control    0.003176    0.003176      0.001356      0.1695   <- estimator bias

g(a,theta) on the D6 control   [1.4928, 1.5367, 1.5311, ...]   spread 0.0440
g(a,theta) on the same 200 raw snapshots  [1.5396, 1.4383, 1.5809, ...]  spread 0.143
```

Read against that floor:

- **$n(x,y)$ confirms the $g_2$ axis.** Its $|F|^2$ at $g_2$ is 0.0795 — **25×** the
  density estimator's own $C_6$ bias of 0.0032 — and the crystal's cell range is
  0.229–2.495 in units of the mean. A Fourier amplitude is immune to the bin-centre
  convention (a constant index offset changes only a phase), so this number needs no
  calibration; a direct point-sampling of the density grid does, which is why the
  point-sampled values quoted in the earlier draft are withdrawn.
- **$n(x,y)$ cannot resolve the $D_2$ degeneracy.** Its $g_1$ and $g_1{+}g_2$
  amplitudes, 0.0080 and 0.0021, are only 2.5× and 1.6× the bias — the same order.
  The density grid is not a clean probe of the $g_1 \leftrightarrow g_1{+}g_2$ equality
  that $S(q)$ shows so exactly.
- **$g(x,y)$ corroborates but does not independently establish the pattern.** The two
  rows above are measured on the *same* 200 snapshots, so they are directly
  comparable: signal 0.143 against estimator bias 0.044, a factor 3.2 at that sampling.
  On the full 1000 the delivered spread is 0.334, i.e. 7.6× the bias. But the ordering
  of the $\theta = 0$ and $\theta = 120$ families is **not stable** between the
  1000-snapshot estimate and the 200-snapshot subset — the 1000 reads
  $g(0) > g(120)$, the 200 reads $g(120) > g(0)$. A corroborating signal stable only
  to ~20 %, not a measurement.

So: real space is **consistent with** the reciprocal-space $D_2$ and independently
confirms the enhanced $g_2$ axis, but the $S(q)$ row of §1 — not this one — is what
establishes the symmetry.

---

## 7. Seed symmetry — the first asymmetric quantity **(deterministic, no Monte Carlo)**

`crystal_v0(L0 = 0.4)` — the campaign default, recorded as `init_L0 = 0.4`,
`init_id = 0` in `run.json` — was rebuilt and its **one-body determinant density**

$$\rho(r) = \sum_k |\psi_k(r)|^2$$

evaluated analytically. No walk, no Jastrow, no SR.

```
|rho_g|^2 / N^2 at the three families (±q_y is the enhanced one)
    (-1,-1) and ( 1, 1)   g1+g2   0.002342   amplitude 0.09679
    (-1, 0) and ( 1, 0)   g1      0.002342   amplitude 0.09679
    ( 0,-1) and ( 0, 1)   g2      0.087323   amplitude 0.59101
second shell (|g|/|g1| = sqrt3):  <= 0.000030
everything at |g|/|g1| >= 2.6458: 0.000000
```

**The seed density is already exactly $D_2$-uniaxial, with the same axis, 37.3×
enhancement, before any optimisation.** A 400-configuration walk on the *seed*
wavefunction gives $S = [1.60479, 1.2986, 6.41739]$ — same axis, same pattern.

Stability of the defect:

| variation | result |
|---|---|
| quadrature `numx` = 41 / 101 / 201 | `[0.002342, 0.002342, 0.087323]` — **bit-identical** |
| `n_max` = 1 and 2 | uniaxial, same axis |
| $L_0 \in \{0.3, 0.4, 0.5, 0.8\}$ | uniaxial; at $L_0 = 1.5$ the axis **flips**: `[0.005795, 0.005795, 5.7e-05]` (101.8× on $g_1, g_1{+}g_2$) |
| couplings already run | rs55 nmax1 ratio 18.0, rs75 nmax2 ratio 37.3, rs45/65 nmax2 A = 1.414 |

The independence from `numx` excludes the rectangular-rule quadrature of
`gaussian_overlap_seed` as the cause. The **L0-dependent axis** shows the
anisotropy is a property of the one-body construction, not of a fixed frame.

**SR symmetry evolution: NOT AVAILABLE.** `run.npz` stores only `snaps` (1000, 36, 2)
and `R` (36, 2); the converged `theta` and any SR trace were not saved, so
$S_i^{(t)}$ at intermediate SR steps cannot be recovered. Per §7 this was **not**
recovered by rerunning SR.

### 7b. ~~The deeper cause: the Landau-level Bloch manifold is not C₆-covariant~~

> **`WITHDRAWN` — the heading is false.** The Landau-level Bloch manifold **is**
> C₆-covariant (`max ||ratio| − 1| ≤ 3.95e-13`, all 108 (k, n), all five rotations,
> quadrature-free). Everything in §7b below is preserved as written, and the
> `T_k` measurement it reports **reproduces** — it is simply not a test of
> covariance, because `T_k` is gauge-dependent and a magnetic rotation acts on the
> orbitals by a gauge transformation. Read the `WITHDRAWN` notes inline. The
> mechanism this section reached for is instead the reversed conjugation in
> `vmc/sr.py:219`; see the banner.

The $k$-dependence enters through $C[k] \propto \langle \phi_{k,n} | G \rangle$ — and
**that proportionality is where the defect was**: the code computed the *conjugate*
of that inner product. Two controls bracket it:

- A **k-independent** mixture $C[k] = C$ for every $k$ gives an **exactly uniform**
  density — `[0, 0, 0]` to roundoff — for every mixture angle and every level.
  So the LL basis is $C_6$-clean as a whole, and the $k$-dependence is the carrier.
- A random $k$-dependent $v_k$ gives only $\sim10^{-3}$ amplitudes, whereas the
  seed's enhanced family reaches $8.7\times10^{-2}$: the seed's $v_k$ is a highly
  structured, correlated choice, not noise.

Testing the manifold directly with the rotation-invariant trace
$T_k(r) = \sum_n |\psi_{k,n}(r)|^2$, which needs no assumption about how the
rotation acts on the level index:

```
k = (0.0000, 0.0000)   R( 60): max rel. change 5.643e-01   *** BROKEN ***
k = (0.0000, 0.0000)   R(120): max rel. change 5.643e-01   *** BROKEN ***
k = (0.0000, 0.0000)   R(180): max rel. change 4.312e-15   INVARIANT
LLL alone, k = 0       R( 60): max rel. change 1.339e+00   *** BROKEN ***
LLL alone, k = 0       R(180): max rel. change 5.638e-15   INVARIANT
identity rotation      exact 0.000e+00                     (machinery sound)
```

At $k = 0$ there is **no mesh representative at all** — `basis_at(R@k)` and
`basis_at(k)` are literally the same object — so no permutation ambiguity can
explain this. The subspace test agrees: pulling the $k=0$ manifold back by R(60°)
gives a space whose every principal-angle singular value is `0.0` against the
original (raw sup-norm difference 3.045).

**Proximate object.**

```
magnetic_translation_phase(ll_ai, 1.0)  =  exp(i * pi * a1 * a2),  tied to the (A1, A2) frame
   R(180):  0 of 121 vectors mismatch, worst |dphase| = 0.000e+00
   R( 60): 60 of 121 vectors mismatch, worst |dphase| = 2.000e+00  (sign flips)
   R(120): 60 of 121 vectors mismatch, worst |dphase| = 2.000e+00
```

The lattice-frame point group of `prim_C = [A1, A2]` (with $A_1$ along $x$) is
exactly $\{E, R(180^\circ), \sigma_x, \sigma_y\} = D_2$ — **the same group, with the
same element assignment, as the anisotropy actually observed**: $\pm g_1$ and
$\pm(g_1{+}g_2)$ are exchanged by $\sigma_x$ and are exactly equal, while $\pm g_2$
is invariant under both reflections and is the enhanced family. The group matches,
including which families are degenerate.

*Caveat, stated plainly.* A cocycle is a convention, not an observable, so "the
$k=0$ manifold is R-invariant" is a gauge-relative statement and is **not** by
itself a proof of a wrong number. What is unconditional is that (i) the
deterministic seed's $D_2$ density is what the delivered $S(q)$ shows — and what the
real-space signals of §6 corroborate within their own estimator bias — and (ii) ~~the
reachable set of this ansatz is not $C_6$-covariant, so the ansatz as coded cannot
represent a sixfold crystal.~~ The line-level fix is therefore *proposed*, not applied.

> **`WITHDRAWN`, on both counts — and this paragraph is the report's own honest
> escape hatch.** It says the cocycle statement is *not* a proof of a wrong number.
> That is correct, and it applies to the whole of §7b, including claim (ii) directly
> above it: `T_k` and the cocycle are both gauge-dependent, so neither establishes
> that the delivered number was wrong. The report then made the claim anyway. The
> unconditional half — (i), the deterministic seed's $D_2$ density — is **confirmed
> and reproduced** (`ANCHOR OK`, <5e-7).
>
> What §7b was reaching for is real: the seed *was* $D_2$. Its cause is not the
> cocycle and not the manifold, but the conjugation at `sr.py:219`, which flips the
> band phase and is exactly conjugation-symmetric only at 180°. The test that
> separates "gauge-dependence of my probe" from "the number is wrong" is the
> **occupied-subspace** test this report never ran; it passes at **3.0e-14** on the
> corrected seed and fails at **7.9e-01** on the defective one. The "line-level fix"
> proposed here (rebuild the manifold in a hexagonal cocycle) is **WITHDRAWN**: it
> would have changed the ansatz to mask a one-line bug.

---

## 8. Finite-cell symmetry

R(60°) about the origin, applied to the geometry:

```
R A1 = A2                     exact        R L1 = L2                      exact
R A2 = A2 - A1                exact        R L2 = L2 - L1                 exact
294 allowed momenta with |q| < 4 :  max distance from R q to the nearest allowed q
                                   = 1.256e-15  ->  the set maps to ITSELF
```

The $N = 36$ torus supports the full $D_6$; boundary conditions, flux convention and
magnetic translations do not reduce it. **The reduction to $D_2$ is not geometric.**

---

## 9. Estimator on controlled inputs

`structure_factor` takes a **snapshot batch** `(n_snap, n_particles, 2)`; $S(q) =
\langle|\sum_j e^{iq\cdot r_j}|^2\rangle / N$. (Passing a bare $(36,2)$ lattice
would be read as 36 snapshots of *one* particle — a trap worth naming, because the
earlier draft of this section did exactly that and then quoted a number no
reproducible call produces.)

```
perfect triangular lattice, no jitter      S(G_i) = [36.0, 36.0, 36.0]          spread 0
same lattice, 0.15 isotropic jitter        [30.881955, 31.75002, 31.938456]    spread 1.0565
the same jittered set D6-SYMMETRISED       [188.126517 x3]                      spread 3.1e-13
the 1000 crystal snapshots ROTATED+POOLED  [4.25248 x3]                         spread 4.4e-15
the SAME 1000 snapshots as delivered       [2.276476, 2.340499, 8.140464]       A = 1.378957
liquid  at the same six q                  [1.343778, 1.261371, 1.251513]       A = 0.071771
```

Two of these are exact to machine precision: a set that is sixfold by construction
gives six equal $S(G_i)$ whether it is built from a synthetic jittered lattice or
from the delivered crystal snapshots themselves, and the *same* call on the
*unrotated* delivered snapshots returns the inequality of §1. **Estimator
excluded.**

(The jittered-single-configuration row's spread 1.06 is that one configuration's own
$O(1)$ fluctuation at $N = 36$, not a bias — the two D₆-symmetrised rows are what
settle the question. The "independent Gaussian-lattice oracle" and "equivariance"
rows of the earlier draft are **withdrawn**: they are not produced by the delivered
script, and the two machine-precision rows above serve the same purpose. The
earlier draft's "spread 0.0718 / 1.3790" on the last two rows were its anisotropy
$A$ values mislabelled as spreads.)

---

## 10. Statistical significance

Error bars from the 1000 stored crystal snapshots only, no new sweeps:

```
per-snapshot SEM (ddof=1)   [0.0688, 0.0716, 0.1097]
20-block SEM                [0.2778, 0.3312, 0.3595]

S3 - S1 = +5.8640 +- 0.1295   (45.3 sigma, naive per-snapshot)
S3 - S1 = +5.8640 +- 0.4543   (12.9 sigma, 20-block)
S2 - S1 = +0.0640 +- 0.0993   ( 0.64 sigma, naive)
S2 - S1 = +0.0640 +- 0.4322   ( 0.15 sigma, 20-block)
```

The naive per-snapshot SEM is **3.5× smaller** than the blocked one because the
snapshots are autocorrelated, so the blocked value is the one to quote; the earlier
draft's $\pm 0.146$ was the naive value, i.e. ~3× too optimistic. Either way
$S_1 - S_2$ is consistent with zero, which is what $D_2$ requires, and $S_3 - S_1$
is not.

The anisotropy is not noise. Equally, the $S_1 = S_2$ degeneracy is exact to
measurement precision — exactly the $D_2$ degeneracy pattern, which is strong
independent support for the mechanism in §7b.

---

## 11. Why the state is not called stripe / nematic

Plotting, q-grid mismatch, reciprocal-vector bug, primitive/supercell wiring,
finite-torus symmetry reduction, parameterization and insufficient sampling are
**all excluded**; but **seed anisotropy is not merely unexcluded, it is confirmed**
(deterministic, pre-SR, and traced to a non-$C_6$ Bloch frame). Under §11 that
forbids the stripe/nematic label.

---

## First cause located

1. `S(q)` at the six first-shell vectors is genuinely unequal in the raw data.
2. Not the plot, the q-grid, the reciprocal vectors, the estimator, the finite cell,
   or the sampling.
3. The **deterministic** seed determinant density `crystal_v0(L0=0.4)` is already
   exactly $D_2$, same axis, 37.3× — before SR, independent of quadrature, with an
   $L_0$-dependent axis.
4. ~~Upstream of that, the **Landau-level Bloch manifold** is not $R(60^\circ)$-covariant
   ($T_0$ changes by 56%; R(180°) exact; identity exact; no representative involved
   at $k=0$; the LLL alone is already broken).~~ **`WITHDRAWN` — false.** The manifold
   is C₆-covariant; `T_0`'s 56% is a gauge artefact of the probe (see §7b).
5. ~~The proximate object is the magnetic-translation cocycle
   $\exp(i\pi a_1a_2)$ tied to the $(A_1, A_2)$ frame — invariant under 180°, sign-flipped
   60/121 under 60° — whose frame point group is exactly the observed $D_2$.~~
   **`WITHDRAWN`.** The proximate object is the reversed conjugation at
   `vmc/sr.py:219`; the cocycle is a gauge choice, and its frame point group matching
   the observed $D_2$ was a coincidence of the same 180°-only symmetry.

**The delivered uniaxial $S(q)$ — and the real-space signals that corroborate it —
are therefore a property of the ansatz's basis and its initializer, not evidence
about the physical ground state.** *(This sentence stands, and is the report's
load-bearing conclusion.)* The crystal energies themselves are unaffected
(they are scalars and the ansatz is variational); what is compromised is any
*structural* claim read off this ansatz. Note the asymmetry in how much each
observable carries: the $S(q)$ row is exact at the six targets and degenerate to
0.15σ, while the real-space rows of §6 are read against an estimator bias of the
same order and are corroboration, not evidence.

---

## Fix *(as originally written — both candidates WITHDRAWN)*

**None applied.** §12 forbids changing the ansatz/Hamiltonian/SR to force sixfold
symmetry, the repair would touch the validated, frozen `physics/landau_levels.py`,
and §15 says stop once the cause is located. Recorded here only as the minimal
candidate:

- ~~Build the LL Bloch manifold with a magnetic-translation cocycle compatible with
  the hexagonal frame (or average the truncated Bloch sum over the six equivalent
  frame choices), then re-derive `crystal_v0` from it.~~ **`WITHDRAWN` — not the fix,
  and it would have changed the ansatz to hide a one-line bug.**
- ~~A symmetry-sensitive regression test would assert
  `T_k(R^-1 r) == T_k(r)` to `1e-12` for R = 60° at k = 0, over a grid — the exact
  test in §7b, which **fails today by 5.6e-01**.~~ **`WITHDRAWN` — the test is
  gauge-dependent and would fail on a *correct* manifold too. The regression test
  that was added asserts the invariance of the occupied SUBSPACE; see
  `PROJECTION_CONJUGATION_FIX_VALIDATION.md`.**

> **The fix that was applied (2026-10-03) is one line, `vmc/sr.py:219`:**
> `np.einsum("i,ikn->kn", G.conj(), bloch)` → `np.einsum("i,ikn->kn", G, bloch.conj())`.
> It touches no other physics, changes no file size, and restores the seed's C₆ to
> 3.0e-14. Source hashes, the mutation proof, the regression tests and the single
> controlled post-fix run are recorded in `PROJECTION_CONJUGATION_FIX_VALIDATION.md`.

## Tests

*(2026-10-03: superseded. `tests/test_projection_convention.py` was added with the
fix — an independent `⟨φ|G⟩` quadrature oracle plus gauge-covariant occupied-subspace
C₆ covariance — and is mutation-proved against the exact defect. What follows
describes the original diagnostic-only state.)*

No test was added or changed: no production file was modified, so there is nothing
new to guard. The diagnosis is one standalone read-only script,
`examples/figure_construction/anisotropy_diagnosis_rs75_nmax2.py`, which reproduces
every number above and touches nothing:

```
python examples/figure_construction/anisotropy_diagnosis_rs75_nmax2.py [--long]
```

It starts no VMC and no SR, writes nothing under `results/` or `figures/`, and was
run end-to-end against the frozen tree (exit 0). `--long` adds the `numx = 201`
quadrature point (~8 min on its own); that point was run and returns the same three
numbers as `numx = 41` and `101` to every printed digit.

Three numbers quoted in an earlier draft of this report are **withdrawn** because
the script cannot reproduce them: the §6 $g(a,\theta)$ spreads 0.197/0.054 (the
values themselves reproduce; the spreads were inconsistent with them), the §6
point-sampled $n(x,y)$ values (sampling the density grid needs the bin-centre
convention pinned, and a Fourier amplitude — which is offset-proof — replaces them),
and the §9 Gaussian-lattice-oracle row. §10's $\pm 0.146$ is superseded by the
blocked $\pm 0.4543$. Each is corrected in place above rather than deleted.

## Unresolved *(as originally written — first three RESOLVED 2026-10-03)*

- ~~Whether the cocycle-driven non-covariance is a *defect* to repair or an accepted
  limitation of the torus construction.~~ **`RESOLVED` — there is no cocycle-driven
  non-covariance.** The manifold is C₆-covariant to 3.95e-13; the non-covariance was
  in the seed, and its cause is the reversed conjugation. This bullet's own
  reasoning — that a cocycle is a gauge choice and §7b's failure is not by itself a
  wrong-observable proof — was correct and was the thread that led to the real cause.
- ~~Which single line of `_in_cell`/`_bloch` is at fault.~~ **`RESOLVED` — neither.
  No line of `_in_cell`/`_bloch` is at fault.** The line is `vmc/sr.py:219`, in the
  projection, not in the basis construction. The `exp(i\pi a_1a_2)` hypothesis is
  withdrawn; its 180°-only invariance was a red herring that happened to match.
- ~~The SR trajectory (§7) — the artifacts do not store `theta`.~~ **`RESOLVED` — and
  the resolution is that it stays unavailable, for BOTH runs.** Per-step energies
  exist in memory (`result.optimization.E_total`, `api.py:888`) but the structural
  recipe's `run_point` returns only `(snaps, R)` and persists no trace; per-step
  *states* — which §8 needs, because `S₁,S₂,S₃` are functions of configurations and
  not of energy — are not retained anywhere at all. An earlier revision of this
  bullet claimed the corrected run would store the trace; **that was wrong and is
  withdrawn here**, and §7's single corrected run was deliberately left on the same
  protocol as the historical one rather than given an extra capture the historical
  run never had. See §8 of `PROJECTION_CONJUGATION_FIX_VALIDATION.md`.
- No structural observable from this ansatz should be quoted as physics until the
  basis question is settled. *(Stands, and is now the follow-up question rather than
  a blocker: the basis is settled — what is open is whether the corrected
  structural observables move, and by how much relative to the Monte Carlo error.
  That is measured for one point in `PROJECTION_CONJUGATION_FIX_VALIDATION.md`; the
  phase diagram is explicitly NOT re-run.)*
