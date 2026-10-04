"""Blocker A, second divergence: which folding is CONVERGED?

`LandauLevelBasis.orbitals(r)` does exactly one thing with `lat_to_cart`:

    _, rk_in_cell, shift_lat, shift_cart = send_to_first_cell(
        r - k x zhat, self.lat_to_cart, self.cart_to_lat)

and then evaluates the Bloch sum `_in_cell(rk_in_cell)`, which sums
`calc_phi_n_alt` over `self.a_cart` -- the LL lattice-vector set, a DISC of
radius ~15 at LUMAX_LL=16.

So the folding cell and the sum's own lattice must be the SAME lattice.  The
notebook folds into the PRIMITIVE cell (`CToCart = [A1, A2]`, `A1 = L1/N1`); the
in-tree `bench_rs75.Setup.basis()` folds into the SUPERCELL (`[L1, L2]`, six
times larger).  With the supercell folding an electron can sit at |r| ~ 16, i.e.
at the very edge of the disc, where the sum is truncated and psi is simply wrong.

This measures it the way Blocker A's first divergence was measured: at the frozen
R0, against a CONVERGED reference.  The reference keeps the primitive folding and
raises the cutoff until psi stops moving.

    python -u scripts/diag_liquid_frame_convergence.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
PROJ = os.path.dirname(CLEAN)
for _p in (os.path.join(CLEAN, "src"), PROJ, HERE, os.path.join(PROJ, "_diag")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qhvmc_engine as legacy                                  # noqa: E402
import bench_rs75 as B                                         # noqa: E402

INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")


def basis_at(cut, l2c, c2l, S):
    ai, ac = legacy.circular_lattice(float(cut), S.A1, S.A2)
    return legacy.LandauLevelBasis(S.mesh, 2, ai, ac, l2c, c2l), len(ai)


def main():
    raw = json.load(open(INIT, encoding="utf-8"))
    S = B.Setup(raw)
    Cp = np.column_stack([S.A1, S.A2])          # the notebook's CToCart
    Cpi = np.linalg.inv(Cp)
    R0 = S.liquid_R0()

    print("=" * 78)
    print("psi_{k,n} at the 36 frozen R0 electrons: the supercell folding vs the")
    print("primitive folding, each against a CONVERGED reference")
    print("=" * 78)
    ref, nref = basis_at(40.0, Cp, Cpi, S)
    print(f"  reference: primitive folding, cutoff 40 -> {nref} lattice vectors")
    prim, _ = basis_at(16.0, Cp, Cpi, S)
    sup, _ = basis_at(16.0, S.C, S.Ci, S)
    prim2, _ = basis_at(24.0, Cp, Cpi, S)

    def dev(b, ref=ref):
        d = 0.0
        sc = 0.0
        for r in R0:
            a, e = b.orbitals(r), ref.orbitals(r)
            d = max(d, np.abs(a - e).max())
            sc = max(sc, np.abs(e).max())
        return d, d / sc

    print(f"  max |r| over R0 electrons            : {np.linalg.norm(R0, axis=1).max():.4f}"
          f"   (half the supercell side is {np.linalg.norm(S.L1)/2:.4f})")
    print()
    print(f"  {'folding':<34}{'max |psi-ref|':>16}{'relative':>12}")
    print("  " + "-" * 62)
    for nm, b in (("primitive  cutoff 16 (notebook)", prim),
                  ("primitive  cutoff 24 (self-check)", prim2),
                  ("SUPERCELL  cutoff 16 (in-tree)", sup)):
        d, rel = dev(b)
        print(f"  {nm:<34}{d:>16.3e}{rel:>12.3e}")

    # where does the supercell folding hurt most?
    print()
    print("  the worst electrons, supercell folding, relative deviation:")
    devs = []
    for j, r in enumerate(R0):
        e = ref.orbitals(r)
        d = np.abs(sup.orbitals(r) - e).max() / np.abs(e).max()
        devs.append((d, j, float(np.linalg.norm(r))))
    devs.sort(reverse=True)
    for d, j, rn in devs[:5]:
        print(f"    electron {j:<3} |r|={rn:>8.4f}   rel dev {d:.4e}")
    print(f"    ... median rel dev {np.median([d for d, _, _ in devs]):.3e}")

    dp, _ = dev(prim)
    ok = dp < 1e-9 and dev(sup)[0] > 1e-6
    print()
    print("  VERDICT: the primitive folding is converged and the supercell folding "
          "is not" if ok else "  VERDICT: inconclusive -- look further")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
