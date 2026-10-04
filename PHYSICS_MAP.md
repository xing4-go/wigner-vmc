# PHYSICS_MAP.md — Stage 2A function-level inventory

**Status:** Stage 2A deliverable (§2A.1). Written *before* any clean physics implementation.
**Scope of Stage 2A:** geometry, magnetic geometry / boundary conventions, Landau-level basis,
Coulomb / Ewald, parameter conventions.
**Explicitly out of scope for 2A:** sampler, local energy, kinetic estimator, Jastrow derivatives,
SR, optimisation, production measurement, notebook generators. Those stay in the legacy engines
untouched and are listed in §6 only so the boundary is visible.

Sources read (all read-only; none modified):

| file | lines | role |
|---|---|---|
| `qhvmc_engine.py` | 1103 | the engine both workflows import |
| `qhvmc_engine_llrot.py` | 562 | LL-rotation additions; imports the above |
| `make_notebook.py` | 4173 | generator of the Gaussian notebook (`WignerCrystal_to_HallLiquid.ipynb`) |
| `reproduction/make_notebook_llrot.py` | 10209 | generator of the LL-rotation notebook |

Every entry below records the ten fields §2A.1 asks for. **A function is not judged identical to
its counterpart in the other workflow by its name**; §4 and §5 record where that judgement was
actually made, and on what evidence.

---

## 1. Units and conventions (locked — §2A.4)

Nothing in Stage 2A may "tidy" any of these into a more standard-looking convention. They are the
conventions the frozen results were produced under, and a change here silently invalidates every
number in `results/`, every figure, and every checkpoint.

### 1.1 Base units

**ℓ_B = ℏ = m = 1.** Energy is measured in **ℏω_c**. These are hard-coded, never passed:
`calc_phi_n_alt` writes `exp(-r^2/4)` (i.e. `exp(-r²/4ℓ_B²)` with ℓ_B = 1); `GaussianBasis` carries
`l = 1.0` as a constructor default and the notebook passes `l=1.0` explicitly; `pi^2 |n> = (2n+1)`
in `pi_orbitals_ladder` is `(2n+1)/ℓ_B²`.

### 1.2 The Hamiltonian, as normalised in the engine

```
H/(ℏω_c) = ½ Σ_i [ -i ℓ_B ∇_i + (1/2ℓ_B) r_i × ẑ ]²  +  κ Σ_{i<j} ℓ_B / |r_i - r_j|
```

- `π = -i∇ - A`, gauge `A(r) = ½(-y, x)` so that `curl A = +B ẑ` with B = 1.
- `[π_x, π_y] = i/ℓ_B² = i`. With `a = (π_x + iπ_y)/√2` and `a† = (π_x - iπ_y)/√2`,
  `π² = 2a†a + 1` and the n-th Landau level has `π² = 2n + 1`. **The engine's own
  `pi_orbitals_ladder` returns exactly `pi2 = O*(2*arange(nb)+1)`** — this is the check that
  the sign of B is +1 and not -1.
- The engine's `Wavefunction.local_energy` returns `E = T + kappa * V` with
  `V = CoulombEwald.energy(R) = Σ_{i<j} 1/|r_i - r_j|` (plus the neutralising background, §1.5).
  The tag check in `io/checkpoints.py::read_tau` enforces `E = T + kappa*V` to 2.3e-13.

### 1.3 The coupling: **`r_s = √2 κ` is locked**

```
ν = 1                (N = 36 electrons, N_φ = 36 flux quanta)
κ = r_s √(ν/2) = r_s / √2
⟹  r_s = √2 κ
```

`κ` is `e²/(4πε₀ ℓ_B ℏω_c)` in these units, i.e. **κ plays the role of `e²`**, which is why the
existing reader comment reads "r_s = sqrt(2) * e^2 and e^2 = kappa here"
(`analysis/energy_competition.py::KappaScan.rs`). The 10 couplings used everywhere are
`KAPPAS = (2,4,8,16,24,32,40,48,64,80)` ⇒ `r_s = (2.83, 5.66, 11.31, 22.63, 33.94, 45.25, 56.57,
67.88, 90.51, 113.14)`.

**This is the single most dangerous conversion in the project**, because the two symbols are
numerically close (κ=32 ↔ r_s=45.25) so a mix-up does not look absurd on a plot. Both
`KappaScan.rs` and the figure annotations convert; the clean layer converts in exactly one place
(`physics/hamiltonian.py`) and nowhere else.

### 1.4 Geometry constants (N=6, ν=1)

| symbol | value | definition |
|---|---|---|
| `area` | `72π = 226.19467105846516` | supercell area; exact, `= 36 × 2π` |
| `A_WC` | `sqrt(4π/√3) = 2.693547374177197` | WC lattice constant = **nearest-neighbour distance** |
| `√(area/ne)` | `sqrt(2π) = 2.5066282746310002` | `1/sqrt_n` where `sqrt_n = sqrt(ne/area)` |
| `\|G1\|` | `2π / (6·A_WC) = 0.4489245623628662` | smallest supercell reciprocal vector |
| `\|g1\|` | `2π / A_WC = 2.693547374177197` | unit-cell reciprocal vector |
| `\|g1\|/\|G1\|` | `6` exactly | = `N1` |

**`A_WC` and `sqrt(area/ne)` are different numbers** (2.6935 vs 2.5066) and both appear in the
codebase. `A_WC` is a *distance*; `sqrt_n = sqrt(ne/area)` is a *density^{1/2}*, used only for the
dimensionless axis scaling in the figures. `Torus.A_WC` and `Torus.sqrt_n` are exactly these.

The 60° cell in use: `A1 = (A_WC, 0)`, `A2 = (A_WC/2, A_WC·√3/2)`, `L1 = 6A1`, `L2 = 6A2`.

### 1.5 Other internal rescalings, item by item

| item | convention | where |
|---|---|---|
| flux quanta | `n_phi = area/(2π l²) = 36 = ne` at `l=1` | `GaussianBasis.__init__` |
| Ewald splitting | `eta = \|L1\| / 3.8` when not given | `CoulombEwald.__init__` |
| Ewald real-space cutoff | `l_cut = 2·eta·6.0 + (\|L1\|+\|L2\|)/2` | `CoulombEwald.__init__`, `erfc_arg_max=6.0` |
| Ewald recip. cutoff | `erfc_arg_max / eta` | `CoulombEwald.__init__` |
| Bloch-sum cutoff, Gaussian | `LUMAX = 30.0` | `make_notebook.py:399` |
| Bloch-sum cutoff, LL-rotation | `LL_REF_CUTOFF`, `LUMAX_LL` | `make_notebook_llrot.py:294,334` |
| LL-site lattice cutoff | `8.5 · \|A1\|` | `make_notebook.py:374` |
| Gaussian width seed | `drummond_width(rs, nu) = 0.5/sqrt(C)`, `C = λ rs^{1/2} ν/2`, `λ=0.15` | `qhvmc_engine_llrot.py:369` |
| structure-factor `q` grid | `QMAX_1D = 4.0`, `q_half = 8·sqrt_n` | `make_figures.py` |
| `ddof` | legacy S(q): 0; band-occupation: 1; `snapshot_sem(..., ddof)` has **no default** | `analysis/statistics.py` |

`l` (the magnetic length *parameter*) is written as an explicit argument in `GaussianBasis`
(always 1.0) but is **absorbed into the constants** in `LandauLevelBasis`. That asymmetry is
preserved in the clean layer, not "fixed": `LandauLevelBasis`'s functions are only valid at
ℓ_B = 1 and the clean module says so in its docstring, while `GaussianBasis` keeps `l` as a
parameter because its `1/l²` and `(l/L0)²` ratios are only meaningful together.

### 1.6 The `nmax` / `nb` rule (§2A.5) — and the trap this stage exists to remove

**Legacy `LandauLevelBasis.n_max` is NOT a Landau index. It is the BAND COUNT.**

Measured directly (fixed input, `mesh` from `geometry_setup(A1,A2,6,6,4.0,True)`):

```
legacy LandauLevelBasis(mesh, n_max=1) -> .orbitals() shape (36, 2), ladder shape (36, 1)
legacy LandauLevelBasis(mesh, n_max=3) -> .orbitals() shape (36, 4), ladder shape (36, 3)
```

`.orbitals()` returns `n_max + 1` columns — bands `0 .. n_max-1` plus **one padding band at index
`n_max`** that supplies `a†|n_max-1>`. `pi_orbitals_ladder` returns `n_max` columns. Its own
docstring says so: *"Band n_max is the 'padding' band"*. And
`LLRotatedOrbitals.__init__` enforces `ll_basis.n_max == n_band`, which is only consistent if
`n_max` is the band count.

So the notebook's `LandauLevelBasis(mesh, 1, ...)` for the **Gaussian** path means *one band
(n=0) plus padding*, and `LandauLevelBasis(mesh, 3, ...)` for a `nest` state means *three bands
(n=0,1,2) plus padding*.

Clean layer (`physics/landau_levels.py`):

- the constructor takes **`nmax` only** — the highest Landau index kept;
- `n_bands = nmax + 1` is a **derived property**, never a constructor argument;
- `.orbitals()` returns the `nmax + 1` **physical** columns only;
- the padding band is reached only through `.padded_orbitals()` (which returns `nmax + 2`);
- `nb` is understood in exactly one place, `nmax_from_n_bands(nb)`, documented as the legacy
  adapter. Nothing else in the physics core accepts a band count.

Mapping used by every regression test:

```
clean LandauLevelBasis(mesh, nmax)  ==  legacy LandauLevelBasis(mesh, nmax + 1)[:, :nmax+1]
clean .padded_orbitals()            ==  legacy .orbitals()          (same nmax+1 == legacy n_max)
clean .ladder()                     ==  legacy .pi_orbitals_ladder()
```

The analysis layer already uses the *clean* sense of `n_max` (`RUNG` in
`analysis/energy_competition.py` is keyed `("conv", 2), ("nest", 3)` with `n_band = n_max + 1`),
so the legacy class attribute is the odd one out, not the analysis layer.

---

## 2. Inventory — geometry and the magnetic cell

### 2.1 `cross_z_hat`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:57` |
| mathematical object | `V ↦ (V_y, -V_x)`, i.e. `ẑ × V` |
| units | dimensionless (acts on any 2-vector) |
| input | `V` array-like, shape (2,) |
| output | `np.array` shape (2,), float |
| Gaussian usage | via `geometry_setup`, `CoulombEwald` |
| LL-rotation usage | same, transitively (it imports `geometry_setup`) |
| traps | the sign. `cross_z_hat(L2)` appears with `+2π` and `cross_z_hat(L1)` with `-2π`; flipping either gives a left-handed BZ that still looks plausible. |
| destination | `physics/geometry.py::cross_z_hat` |
| regression | exact equality on 8 fixed vectors including (1,0), (0,1), (±A_WC, ±A_WC/2) |

### 2.2 `circular_lattice`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:63` |
| mathematical object | `{ i·a1 + j·a2 : \|·\| ≤ r_max }` over integers, as `(ints, carts)` |
| units | `r_max`, `a1`, `a2` in the same length unit (ℓ_B = 1) |
| input | `r_max` float; `a1`, `a2` (2,) |
| output | `ints` (n,2) int, `carts` (n,2) float — **the two are index-aligned** |
| Gaussian usage | `lints, lcart = circular_lattice(LUMAX, L1, L2)` (supercell vectors, cutoff 30) |
| LL-rotation usage | `circular_lattice(cut, A1, A2)` (primitive WC vectors, cutoff 8.5·\|A1\|) — **different lattice, different cutoff** |
| traps | (a) the search window is a *rectangle* `n1 = 2⌈r_max/\|a1\|⌉` in `(i,j)`, so the enumeration order depends on the basis even though the *set* does not; (b) callers must not assume the output is sorted by norm — `geometry_setup` sorts explicitly and `CoulombEwald` does not; (c) the same function is called with two different lattice vector pairs in the same notebook. |
| destination | `physics/geometry.py::circular_lattice` |
| regression | identical `(ints, carts)` arrays (element-wise, dtype checked) for `(30, L1, L2)` and `(8.5\|A1\|, A1, A2)`; plus invariance of the *set* under a unimodular basis change, checked by sorting |

### 2.3 `geometry_setup`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:82` |
| mathematical object | from `(A1, A2, N1, N2)`: supercell vectors `L_i = N_i A_i`; area; reciprocal supercell `G1 = 2π ẑ×L2/area`, `G2 = -2π ẑ×L1/area`; reciprocal unit cell `g1 = 2π ẑ×A2/(area/N1N2)`, `g2 = -2π ẑ×A1/(area/N1N2)`; the `N1·N2` **BZ mesh**; and the sorted lattice vectors `RL` within `rl_cut` |
| units | lengths in ℓ_B = 1; area in ℓ_B²; reciprocal in 1/ℓ_B² (they are wavevectors) |
| input | `A1, A2` (2,); `N1, N2` int; `rl_cut` float; `B_field` bool; `fermi_surface` bool |
| output | 9-tuple `(area, L1, L2, G1, G2, mesh, g1, g2, RL)` |
| Gaussian usage | `geometry_setup(A1,A2,6,6,4.0,True)` (`make_notebook.py:173`) |
| LL-rotation usage | `geometry_setup(...)` with `B_field=True` as well (`make_notebook_llrot.py:534`); the LL notebook also re-runs it at other `N` for size tests |
| traps | **`mesh` is in ROW order `i·G1 + j·G2` with `i` outer, 36 entries, NOT sorted, NOT the same object as the 294 S(q) momenta.** `Torus.allowed_momenta` returns 294 vectors in rounded-lexsort order. Two different sets for two different jobs; the accumulated structure factor is stored in the *engine's* order and must be aligned by q **vector**, never by index. (b) The `B_field=False` branch mutates `mesh` — the folding loop and the `fermi_surface` branch — and is dead code for every call in the two generators (both pass `True`); it is migrated verbatim anyway so the map is complete, and marked unused. |
| destination | `physics/geometry.py::Geometry` (a frozen dataclass built by `Geometry.from_cell(...)`) |
| regression | all 9 outputs vs legacy on the production call and on 2 off-production calls (`N=4`, non-60° cell); `mesh` compared element-wise **in order** (order is part of the contract); `RL` compared after norm-sort |

### 2.4 `send_to_first_cell`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:120` |
| mathematical object | fold a point into the first primitive cell by rounding fractional coordinates; return the fold, the lattice shift, and — critically — the *phase-relevant* residual identity `shift_cart = r - r_cart` |
| units | lengths |
| input | `r` (n,2); `lat_to_cart` (2,2) **columns are the lattice vectors**; `cart_to_lat` its inverse |
| output | `(r_lat, r_cart, shift_lat, shift_cart)` |
| Gaussian usage | **not used** (the Gaussian orbitals are not Bloch sums over a lattice with this fold) |
| LL-rotation usage | yes — inside `LandauLevelBasis.orbitals`, on `rk_pt = r - k×ẑ` |
| traps | the returned `shift_cart` is the *original minus folded*, so it carries the full lattice vector, not just the fractional part; `orbitals()` uses it in `shift_fac` with a sign that includes `π·shift_lat_x·shift_lat_y`. Both are silently wrong-looking if the fold convention changes. |
| destination | `physics/magnetic_cell.py::fold_to_primitive_cell` |
| regression | 4-tuple equality on ~200 fixed points including points exactly on cell edges (±0.5 fractional), where `np.round`'s banker's rounding picks the representative |

### 2.5 `send_to_first_supercell`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:134` |
| mathematical object | fold a point into the first supercell, returning only the folded point |
| units | lengths |
| input | `r` (n,2); `sc_to_cart`, `cart_to_sc` |
| output | `(n,2)` |
| Gaussian usage | not directly |
| LL-rotation usage | not directly |
| traps | **answers a different question from `minimum_image_displacement`.** See §2.6. The function name invites the wrong substitution. |
| destination | `physics/geometry.py::wrap_to_supercell` |
| regression | equality on a grid over the cell, including edges |

### 2.6 `minimum_image_displacement`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:142` |
| mathematical object | the image `d + n1 L1 + n2 L2` of minimum Euclidean norm, searched over the 9 shifts around the parallelogram representative |
| units | lengths |
| input | `d` (...,2); `sc_to_cart`, `cart_to_sc` |
| output | same shape as input |
| Gaussian usage | not used — `CoulombEwald._wrap` uses the *supercell* fold instead |
| LL-rotation usage | not used |
| traps | **this is the single most confusable pair in the engine.** The docstring records a concrete counterexample: for the 60° cell with `L1=(1,0)`, `L2=(1/2,√3/2)`, the fractional displacement `(0.49, 0.49)` is inside the parallelogram at `\|d\| = 0.849` but its neighbour image `(-0.51, 0.49)` is at `0.500`. Rounding fractional coordinates gives the first; the physical inter-electron distance is the second. **`CoulombEwald._wrap` deliberately uses the fold, not the minimum image**, because the Ewald sum is over the periodic array and the fold is what makes the cutoff bound `\|r_ij\| ≤ (\|L1\|+\|L2\|)/2` valid. Swapping the two is a silent physics change that still conserves energy. |
| destination | `physics/geometry.py::minimum_image` |
| regression | (a) equality with legacy on 2000 random displacements; (b) an *independent* check that `\|result\| ≤ \|result + n1 L1 + n2 L2\|` for all `n ∈ [-4,4]²` (brute force), which is the property the function exists for; (c) the docstring's counterexample reproduced as a named test |

### 2.7 `gaussian_sites`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:435` |
| mathematical object | the 36 Wigner-crystal sites: `i·A1 + j·A2` for `i,j ∈ [0,6)`, then each folded to the **minimum-norm image under the supercell lattice** (brute force `i,j ∈ [-5,5]`) |
| units | lengths |
| input | `N1, N2, A1, A2, aspect_ratio=1` |
| output | `(N1·N2, 2)` |
| Gaussian usage | `sites = gaussian_sites(N, N, A1, A2)` (`make_notebook.py:323`) — the Gaussian crystal's orbital centres |
| LL-rotation usage | the LL path does **not** use sites for its ansatz; it seeds from `gaussian_overlap_seed`, which is a projection of a site-centred Gaussian onto the LL basis |
| traps | the fold centres the *set* on the origin, so the 36 sites straddle `r=0` and do **not** tile `[0,L1)×[0,L2)`. This is physically fine (the crystal has no origin) but it is why 20 of 36 site markers were clipped off the axes in the Part IV figure — the plot's axes were built from the cell outline, not from the site positions. Recorded because the same mistake is available to any new figure. |
| destination | `physics/geometry.py::wigner_crystal_sites` |
| regression | exact equality with legacy (the `[-5,5]` window is reproduced verbatim, so ties are resolved identically); plus a structural assertion that all 36 are distinct and that the minimum pair distance equals `A_WC` |

### 2.8 `LandauLevelBasis` and the magnetic-cell conventions inside it

Migrated in `physics/landau_levels.py`; the *cell* parts are listed here because they are what
`magnetic_cell.py` owns.

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:200` (`__init__` 209, `_in_cell` 221, `orbitals` 246, `pi_orbitals_ladder` 269) |
| mathematical object | magnetic Bloch orbitals `ψ_{k,n}(r)` built as a Bloch sum of disk-basis states over lattice vectors `a` with magnetic translation phase `exp(iπ A1_int A2_int)` per vector, `exp(i(r×a)_z/2ℓ²)`, and the gauge-completion factor `exp(i r·k/2)`; plus `π_xψ, π_yψ, π²ψ` built **by ladder operators on the band index** rather than by differentiating |
| units | ℓ_B = 1 throughout (`calc_phi_n_alt` has `exp(-r²/4)`) |
| input | `mesh` (nk,2); `n_max` = **band count** (§1.6); `a_ints` (na,2) int; `a_cart` (na,2); `lat_to_cart`, `cart_to_lat` |
| output | `.orbitals(r)` (nk, n_max+1) complex; `.pi_orbitals_ladder(r)` three (nk, n_max) complex arrays |
| Gaussian usage | `LandauLevelBasis(mesh, 1, Avals, AvalsCart, CToCart, cartToC)` — **1 band + padding**, only column 0 used, and it is used only to *calibrate* (`approx` prints, the LLL validation figure), not as the Gaussian crystal's ansatz |
| LL-rotation usage | `LandauLevelBasis(mesh, NB, ...)` — the ansatz itself, via `LLRotatedOrbitals` |
| traps | (a) `n_max` is a band count (§1.6). (b) `phase_a` uses `A[:,0]*A[:,1]` on **integer** coefficients — passing floats that happen to be integers is fine, passing non-integers silently changes the phase. (c) `phase_r` is built from `r_in_cell` (the folded point), **not** from `big = r + a`. (d) `orbitals()` does `rk = (k_y, -k_x)` — i.e. it evaluates at `r - k×ẑ`, with `ẑ×k = (-k_y, k_x)`, so `r - ẑ×k = (r_x + k_y, r_y - k_x)`. Reading `rk` as `ẑ×k` rather than `-ẑ×k` flips the sign of every momentum label and permutes columns in a way that is only visible against the degenerate-star tie-break. (e) `exp_fac = exp(i r·k/2)` uses the **unfolded** `r`. (f) the basis is built on `CToCart = column_stack([A1,A2])` — the *primitive WC* lattice — while the Bloch-sum lattice vectors are a **third** set, `circular_lattice(8.5·\|A1\|, A1, A2)`. Three lattice sets coexist: supercell (`L`), primitive (`A`), and the Bloch-sum cutoff set. |
| destination | `physics/landau_levels.py::LandauLevelBasis`; the fold and the phase conventions in `physics/magnetic_cell.py` |
| regression | `.orbitals()` and `.ladder()` element-wise vs legacy over 12 fixed positions and 3 `nmax` values, `rtol=1e-13`; plus the `π²` diagonal identity `pi2[:,n] = (2n+1)·orbitals[:,n]`; plus a finite-difference check that `pix, piy` really are `πψ` for `A = ½(-y,x)` — this is the check that pins the *sign of B*, and it is run in the clean layer against a clean finite difference, not against legacy |

---

## 3. Inventory — Coulomb / Ewald

### 3.1 `CoulombEwald`

| field | content |
|---|---|
| legacy | `qhvmc_engine.py:627` (`__init__` 630, `_wrap` 673, `_pair_short` 679, `_pair_long` 685, `energy` 690, `delta_move` 698); `_erfc` at 710 |
| mathematical object | Ewald split of `Σ_{i<j} 1/\|r_i-r_j\|` on the torus: short-range `erfc(\|d-l\|/2η)/\|d-l\|` over real lattice vectors `l` (including `l = 0`), long-range `Σ_g cos(d·g)·2π·erfc(η\|g\|)/\|g\| / area`, plus a constant `v_const = v_const_pair + v_const_self` holding the `g = 0` term and the self-energy/neutralising background |
| units | lengths ℓ_B = 1; the result is the dimensionless `Σ 1/r` that `κ` multiplies |
| input | `ne` int; `L1`, `L2` (2,); `G1`, `G2` (2,); `eta=None`; `erfc_arg_max=6.0` |
| output | object with `.energy(R) -> float`, `.delta_move(R, i, r_old) -> float`, and the cached tables |
| Gaussian usage | `CoulombEwald(ne, L1, L2, G1, G2)` — `make_notebook.py:329, 495, 703, 928` |
| LL-rotation usage | identical construction; both workflows share one potential term |
| traps | (a) **`l_cart` must contain `L = 0`** (the bare pair interaction); `l_nz` is the `L ≠ 0` subset used only where `1/\|L\|` would diverge. The reference removes the zero vector only inside `calc_Coulomb_energy_const`. Dropping it from `l_cart` shortens every pair interaction by `1/\|d\|` at `d→0` and is **not** a small change. (b) `eta` defaults to `\|L1\|/3.8` — a *supercell*-dependent number, so `eta` is not comparable between system sizes. (c) `l_cut` is derived, not chosen, from the bound `\|r_ij\| ≤ (\|L1\|+\|L2\|)/2`, valid **because `_wrap` folds rather than taking the minimum image**. (d) `_erfc` is defined at line 710 but called at 659/661/683 — legal (resolved at call time) but a reader grepping upward finds nothing. (e) `V` is stored **complex with zero imaginary part** by the legacy store, which is why `io/checkpoints.py::_energy_fields` casts it. (f) `delta_move` returns the change in the *pair* sum only, **not** the constant — correct, since `v_const` is position-independent, but a caller adding it would double-count. |
| destination | `physics/coulomb.py::CoulombEwald` |
| regression | (a) `energy(R)` vs legacy, `rtol=1e-13`, on 40 fixed configurations at 3 densities; (b) `delta_move(R,i,r_old)` vs `energy(R') - energy(R)` with `R' = R` with row `i` replaced — this is the identity the sampler depends on and it is checked **within the clean module** as well as against legacy; (c) the Madelung check the notebook already does, `madelung_rows[0][2] ≈ 1.106103` to 1e-5, as an external physical anchor that does not depend on legacy at all; (d) an `eta`-sensitivity test: `energy` must be `eta`-independent to ~1e-10 across `eta ∈ {0.4, 0.7, 1.0}·\|L1\|/3.8`, which is the property that makes the split legitimate |

---

## 4. Confirmed fully shared between Gaussian and LL-rotation

These are provable at function level — same function, or bit-identical outputs on fixed inputs.
They may be merged into one clean implementation.

| object | evidence |
|---|---|
| supercell geometry `L1, L2, area, G1, G2, g1, g2` | both generators call the same `geometry_setup`; clean `physics.geometry.Geometry` reproduces it **bit-for-bit** (§4.1) |
| `CoulombEwald` and therefore the entire potential energy `κ·V` | both workflows construct it identically and `Wavefunction.local_energy` is the *same function object* used by both |
| `cross_z_hat`, `circular_lattice`, `send_to_first_cell`, `send_to_first_supercell`, `minimum_image_displacement` | module-level functions of `qhvmc_engine`, imported by `qhvmc_engine_llrot` rather than redefined |
| `calc_phi_n_alt` / the disk basis | the LL-rotation path's orbitals are built *on top of* `LandauLevelBasis`; there is one LL basis implementation |
| the `π² = 2n+1` spectrum | `LandauLevelBasis.pi_orbitals_ladder` returns it and `LLRotatedOrbitals._kin` consumes it |

`CoulombEwald.sc_to_cart == Torus.sc` and `CoulombEwald.cart_to_sc == Torus.c2sc` exactly — the two
independently-written supercell matrices agree bit-for-bit, which is the strongest single piece of
evidence that `analysis/structure.py::Torus` and the legacy `geometry_setup` describe one object.

### 4.1 Floating-point agreement, stated exactly

Three objects claim to be "the geometry": the frozen engine's `geometry_setup`, the clean
`physics.geometry.Geometry`, and the analysis layer's `structure.Torus`. They are not three copies of
one computation — `Torus` is handed an area (`72*pi`) and stores it, while the other two compute the
area as `|L1 x L2|` — so "they agree" needs a tolerance attached. Measured on the production cell
(6x6, `N=36`):

| comparison | result |
|---|---|
| `Geometry` vs legacy `geometry_setup` — `A1 A2 L1 L2 area G1 G2 g1 g2` | **bit-identical, all nine** |
| `Geometry` vs legacy — `A_WC`, `sqrt_n` | bit-identical (`sqrt_n` reproduced as `1/sqrt(area/ne)`) |
| `Geometry.sc_to_cart`/`cart_to_sc` vs `CoulombEwald` | bit-identical (already in the table above) |
| `Torus` vs legacy — `A1 A2 L1 L2` and `A_WC` | bit-identical |
| `Torus` vs legacy — `area` | `5.68e-14` apart (2 ULP: stored `72*pi` vs computed `|L1 x L2|`) |
| `Torus` vs legacy — `G1 G2` | `1.11e-16` (1 ULP); `g1 g2` `8.88e-16` |
| `Torus` vs legacy — `sqrt_n` | 1 ULP |
| `Torus` vs `Geometry` — `G1 G2 g1 g2 sqrt_n area` | same as the two rows above, by transitivity |

**Physical interpretation: unchanged.** Every relative disagreement is `<= 3e-16` — reciprocal and
density quantities divide by the area, so they inherit its 2 ULP, and nothing else does. That is
twenty orders of magnitude below any quantity either workflow measures, and far below the `1e-13`
tolerances the regression tests already carry.

Two consequences, both deliberate:

* The clean `geometry.py` is the strongest claim in Stage 2A — it is the frozen engine's geometry
  **exactly**, not approximately. That is what `test_geometry_against_legacy.py` and the
  `Geometry vs legacy` rows above are for.
* `Torus` is NOT rewritten to close its 2 ULP, and the clean layer is NOT rewritten to reproduce it.
  `Torus` is frozen analysis-layer code inside the figure contract; the clean formula is the one
  that reproduces the engine. `test_units_and_conventions.py` pins both, each against its own
  reference, with tolerances stated as the measured deviations above rather than as `rtol=0`.

An earlier revision of this section said the clean `Torus` and `geometry_setup` "agree exactly".
That was **wrong** — it named `Torus` (analysis layer, 2 ULP) where it meant `Geometry` (clean layer,
bit-exact), and it is corrected above.

## 5. Similar-looking but NOT the same — do not merge

| Gaussian side | LL-rotation side | why they must stay separate |
|---|---|---|
| `GaussianBasis` — site-centred magnetic Gaussians, `πψ` and `π²ψ` obtained **by differentiating/assembling** analytic expressions per site | `LandauLevelBasis` + `LLRotatedOrbitals` — Bloch LL states, `πψ` and `π²ψ` obtained by **ladder operators on the band index** | `_kin`'s docstring is explicit: *"the magnetic ladder operators act on the BAND index, so no coordinate derivatives are taken … π² is DIAGONAL in n … A Gaussian basis has no such property."* The two produce the same operator by completely different mathematics; sharing code between them would mean sharing the wrong one. |
| `n_phi = area/(2π l²)` with `l` a live parameter | `ℓ_B` absorbed as 1 | the Gaussian orbital's width `L0` and the magnetic length `l` enter as a *ratio* `(l/L0)²`; the LL basis has no such ratio |
| ansatz parameterised by **site positions and one width `L0`** | ansatz parameterised by **`v` (nk, n_bands−1) complex rotation angles** | different parameter spaces; §2B will define Ψ for each, and they stay two named objects |
| `dpsi_dL0` | `dc_row_dparams` | different parameters, both analytic derivatives, no shared code |
| seed: `gaussian_sites` + `drummond_width` | seed: `gaussian_overlap_seed` (a projection onto the LL basis) | the LL seed *uses* a site-centred Gaussian, but produces a different object (a `(nk, n_band)` overlap, not site positions) |
| eigenvalues/`π²` per site | `π²` diagonal in band index | see row 1 |

`gaussian_overlap_seed` is the one place that looks like sharing and is not: it is a bridge
*from* the Gaussian picture *into* the LL basis, and it belongs to the LL-rotation workflow.

---

## 6. Not migrated in Stage 2A (boundary of this stage)

Listed so the boundary is explicit and nothing here is accidentally pulled in. All remain in the
legacy engines, unmodified and callable.

| object | legacy | stage |
|---|---|---|
| `calc_D_ratio`, `update_D_inv`, `calc_kinetic_energy` | `qhvmc_engine.py:719, 759, 734` | 2C |
| `Wavefunction` (build/rebuild/move_ratio/accept_move/local_energy) | `qhvmc_engine.py:777` | 2C |
| `sample` | `qhvmc_engine.py:903` | 2C |
| `SinSplineJastrow`, `b3`, `dx_b3`, `d2x_b3`, `_calc_f`, `_calc_grad_f`, `_calc_laplacian_f` | `qhvmc_engine.py:469-625` | 2B/2C |
| `jastrow_log_deriv`, `sr_optimize_jastrow` | `qhvmc_engine.py:959, 974` | 2D |
| `LLRotationWavefunction`, `_ops_of`, `pi_columns`, `local_energy`, `orbital_log_deriv`, `log_deriv`, `v_from_theta` | `qhvmc_engine_llrot.py:250-364` | 2B/2C |
| `sr_optimize_joint`, `sr_pilot_scale`, `jastrow_vector` | `qhvmc_engine_llrot.py:457, 531, 560` | 2D |
| `structure_factor`, `pair_correlation`, `density_grid` | `qhvmc_engine.py:1033-1103` | already migrated (`analysis/structure.py`) |
| `_wtaylor`, `c_row`, `dc_row_dparams` | `qhvmc_engine_llrot.py:59, 74, 90` | 2B (they parameterise Ψ, not the geometry) |
| `drummond_width`, `gaussian_overlap_seed` | `qhvmc_engine_llrot.py:369, 377` | 2B (ansatz seeding); `drummond_width`'s **convention** is recorded in §1.5 but its code is not migrated |

`hamiltonian.py` in Stage 2A is a **parameters/conventions container only** (§2A.2). It parses
`(r_s, nu, nmax)` into the derived quantities of §1 and holds them; it performs **no** local-energy
evaluation and imports nothing from `local_energy`, `sampler` or `optimization`.

---

## 7. Regression strategy, in one place

Every migrated function follows the same four steps, in this order, with no "copy then declare
done":

1. **legacy understanding** — read the legacy source, and where the docstring and the code could
   disagree, *measure* (this is how §1.6 was settled: by running the legacy class, not by reading
   its parameter name).
2. **fixed-input regression** — a deterministic fixture: fixed configurations, fixed cell, fixed
   `r_s`. No Monte Carlo, no RNG without an explicit seed, no optimisation.
3. **clean implementation** — written against the mathematics of §2-§3, not transliterated.
4. **old/new comparison** — element-wise, with a tolerance that is *stated*, not "the plots look
   the same". Numerical functions are required to agree to near machine precision
   (`rtol = 1e-13`, `atol = 1e-14` on quantities of order 1). Where a comparison is deliberately
   looser, the reason is written next to the tolerance.

Benchmarks: three couplings spanning the physics, `r_s = 45, 65, 75`
(⇒ `κ = 31.8198, 45.9619, 53.0330`). These fix `eta` (through `L1`), the Ewald tables, and the
`q`-grid scaling. They are used for the Coulomb and convention tests; the geometry tests do not
depend on the coupling and are run once.

The legacy engines are imported as **top-level modules** in the test process and the clean package
from `src/`, so old and new are callable side by side in one interpreter. Nothing in the clean
package imports legacy, and nothing in legacy is redirected at the clean package.
