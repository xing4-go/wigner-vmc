"""Blocker B, B2/B3: is ``c0`` -- and therefore ``_init_digest`` -- a function of
kappa alone?  The decisive, cache-free measurement.

What round 1 (``tools/diag_init_layers.py``) could and could not settle
-----------------------------------------------------------------------
Round 1 rebuilt the whole chain for both kappa branches, but it had to run the
legacy branch first against whatever the working copy already held, and the
curated store's ``jastrow_crystal_k53.033`` currently carries the tag

    v1|ne36|LU30|J5|crystal|k53.033|L0.5|st8|n150|sig0.3|w-7.49|sr1

-- i.e. seed 2's tag, whose pkl is the probe's OWN clean-kappa write from the
2026-10-02 incident.  A legacy call at L0 = 0.5 therefore cache-HITs a file that
was produced at the other kappa, so ``s2``'s legacy branch in round 1 cannot be
read as a legacy result.  Every other seed's tag (L0 = 0.3 / 0.45 / 0.6 / 0.8)
differs from what the store holds, so those calls genuinely missed and
recomputed -- they are valid.

This script removes the ambiguity by construction: **the cache entry is deleted
before every single call**, so all ten calls are guaranteed recomputations.

Purging is faithful, not a deviation.  ``_diag/part1_scan.py`` ran the five
conv seeds at five DISTINCT L0 values (``_L0S`` with ``--seeds 5`` over a
seven-point grid), and a call whose L0 differs from the stored one is a MISS
that recomputes.  Two calls sharing an L0 would hit, but a HIT returns the same
deterministic output a miss would have computed, because the tag fixes
(kappa, L0, warm, steps, sweeps, seed).  The campaign's cache was therefore
harmless to the campaign; it is only ambiguous to a probe that reads it back
after the fact.

The causal claim under test (B3)
--------------------------------
    correct legacy input -> agreement;  suspected wrong input -> reproduces the failure

For each seed: build ``c0`` twice, once at the campaign's ``round(rs/sqrt(2), 4)``
= 53.033 and once at the physical ``rs/sqrt(2)`` = 53.03300858899106, changing
NOTHING else (same warm-start base ``J_OPT[("crystal", 48.0)]``, same rescaling
rule, same steps/sweeps/seed, same L0, same ``v0``).  Then:

  * legacy kappa must reproduce the frozen campaign digest for that seed;
  * clean  kappa must reproduce the reconstruction's digest (round-1 control);
  * ``v0`` must be bit-identical across the two, because ``rs`` is inert once
    ``L0`` is given.

The frozen digests are independent oracles: all ``llcryst_*`` tags are
byte-identical pre- and post-incident, and the L0 -> digest map is corroborated
by two separate tier stores (``p1fast1`` frozen and ``p1conv4`` in %TEMP%).

One independent value oracle exists as well.  %TEMP%/qhvmc_checkpoints holds an
incident-untouched ``jastrow_crystal_k53.033`` with tag ``...|L0.8|...`` and pkl
sha256 951d1440a9b87178... (identical in the pre- and post-incident manifests).
Its ``c[0] = -9.837301961029251`` must be reproduced BIT-FOR-BIT by the legacy
branch at L0 = 0.8, and is compared against the clean branch at the same L0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frozen_guard import FROZEN_CKPT, readonly_frozen            # noqa: E402
from recon_rs75_init import CLEAN, namespace_cells               # noqa: E402

RS = 75.0
K_LEGACY = round(RS / np.sqrt(2.0), 4)      # 53.033            -- campaign convention
K_CLEAN = RS / np.sqrt(2.0)                 # 53.03300858899106 -- physical
FIVE = 5
NK = 36
NB_BANDS = 2

# L0 -> frozen digest, transcribed from the stores (read, not guessed).
#   _diag/ckpt_rebuild (frozen) : p1conv0..p1conv3, p1fast0, p1fast1
#   %TEMP%/qhvmc_checkpoints    : p1conv4
FROZEN_BY_L0 = {
    0.30: ("cadd5d6a", "llcryst_nb2_k53.033_p1conv0 / p1fast0"),
    0.45: ("46d9bf1e", "llcryst_nb2_k53.033_p1conv1"),
    0.50: ("eba63d0f", "llcryst_nb2_k53.033_p1conv2"),
    0.60: ("7f284e09", "llcryst_nb2_k53.033_p1conv3"),
    0.80: ("6d38078f", "llcryst_nb2_k53.033_p1fast1 / p1conv4 (%TEMP%)"),
}

# The one independent VALUE oracle: %TEMP%'s incident-untouched L0 = 0.8 pkl.
TEMP_C08 = [-9.837301961029251, -4.121304373846473, -1.7029848727246566,
            -0.5870473496402807, -0.3145332908917838]
TEMP_C08_SHA = "951d1440a9b87178"


def _A(x):
    return np.asarray(x, float)


def sha16(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def purge(work):
    """Delete the shared crystal-Jastrow cache entry, forcing the next call to
    recompute.  Returns (sha_of_pkl_before, meta_tag_before)."""
    base = os.path.join(work, "jastrow_crystal_k53.033")
    tag = open(base + ".meta", encoding="utf-8").read().strip() if os.path.exists(base + ".meta") else None
    s = sha16(base + ".pkl")
    for ext in (".pkl", ".meta"):
        try:
            os.remove(base + ext)
        except FileNotFoundError:
            pass
    return s, tag


def one_call(E, kappa, L0, k_near, work):
    """A guaranteed-MISS ``jastrow_opt`` call.  Returns (c, purged_sha, purged_tag)."""
    sha_before, tag_before = purge(work)
    warm = _A(E["J_OPT"][("crystal", k_near)]) * (kappa / k_near)
    c = _A(E["jastrow_opt"]("crystal", kappa, L0=L0, warm=warm)["c"])
    # The call must have written a fresh entry; if it did not, the purge failed
    # and the result may be stale -- refuse to report it.
    base = os.path.join(work, "jastrow_crystal_k53.033")
    if not os.path.exists(base + ".meta"):
        raise RuntimeError("purge did not force a recomputation: no .meta was written")
    return c, sha_before, tag_before


def run(E, work):
    """The measurement, given an already-built namespace.

    Exposed as a function so a combined driver can build the namespace ONCE and
    run this beside the pkl-byte probe -- a namespace build costs ~17 minutes,
    and the two measurements share every input.
    """
    rows, oracle_hit = [], None
    L0_GRID = _A(E["L0_GRID_J"])
    KAPPAS = _A(E["KAPPAS"])
    L0S = [float(L0_GRID[i]) for i in
           np.linspace(0, len(L0_GRID) - 1, FIVE).round().astype(int)]
    k_near = float(min(KAPPAS, key=lambda kk: abs(kk - K_LEGACY)))
    print(f"  L0_GRID_J = {L0_GRID}")
    print(f"  _L0S (--seeds {FIVE}) = {L0S}    nearest kappa = {k_near}\n", flush=True)

    for si, L0 in enumerate(L0S):
            c_leg, sha_b, tag_b = one_call(E, K_LEGACY, L0, k_near, work)
            c_cln, _, _ = one_call(E, K_CLEAN, L0, k_near, work)

            # v0 is built once: it does not see kappa at all when L0 is given.
            ov = np.asarray(E["gaussian_overlap_seed"](
                E["_llb8B"], int(E["NMAX_STAGE_B"]), E["L1"], E["L2"],
                E["_lints8"], E["_lcart8"], rs=RS, numx=121, L0=L0), complex)
            v0 = np.asarray(E["v_from_overlap"](ov), complex)

            d_leg = E["_init_digest"](v0, c_leg)
            d_cln = E["_init_digest"](v0, c_cln)
            fdig, fsrc = FROZEN_BY_L0.get(round(L0, 2), (None, "no oracle at this L0"))
            dc = float(np.abs(c_leg - c_cln).max())
            scale = max(float(np.abs(c_leg).max()), 1e-300)
            bit = bool(np.array_equal(c_leg, c_cln))

            print(f"  s{si}  L0 = {L0:<5g} {'=' * 56}")
            print(f"       purged before the call: tag {tag_b!r}  pkl {sha_b}")
            print(f"       c0 legacy k {K_LEGACY:<8} [{c_leg[0]:+.15e} ...]")
            print(f"       c0 clean  k {K_CLEAN:.12f} [{c_cln[0]:+.15e} ...]")
            print(f"       max|c0_legacy - c0_clean| = {dc:.3e}   relative {dc / scale:.3e}"
                  f"   bit-identical? {bit}")
            print(f"       digest legacy {d_leg}     clean {d_cln}")
            if fdig is None:
                print(f"       frozen        --  ({fsrc})")
            else:
                print(f"       frozen        {fdig}   [{fsrc}]")
                print(f"       LEGACY matches frozen: {d_leg == fdig}"
                      f"     CLEAN matches frozen: {d_cln == fdig}")

            if round(L0, 2) == 0.80:
                ref = np.array(TEMP_C08)
                hit = bool(np.array_equal(c_leg, ref))
                oracle_hit = hit
                print(f"       VALUE ORACLE (%TEMP% pkl sha {TEMP_C08_SHA}, L0=0.8):")
                print(f"         c_legacy bit-identical to the untouched campaign pkl? {hit}"
                      f"   max|d| = {float(np.abs(c_leg - ref).max()):.3e}")
                print(f"         c_clean  vs that same oracle: max|d| ="
                      f" {float(np.abs(c_cln - ref).max()):.3e}")
            print(flush=True)

            rows.append({
                "si": si, "L0": L0,
                "c0_legacy": c_leg.tolist(), "c0_clean": c_cln.tolist(),
                "c0_max_abs_diff": dc, "c0_rel_diff": dc / scale,
                "c0_bit_identical": bit,
                "digest_legacy": d_leg, "digest_clean": d_cln, "digest_frozen": fdig,
                "legacy_matches_frozen": (None if fdig is None else d_leg == fdig),
                "clean_matches_frozen": (None if fdig is None else d_cln == fdig),
                "frozen_source": fsrc,
                "tied_digest_legacy_equals_clean": d_leg == d_cln,
            })

    n_ok = sum(1 for r in rows if r["legacy_matches_frozen"])
    n_or = sum(1 for r in rows if r["legacy_matches_frozen"] is not None)
    print("=" * 78)
    print(f"  legacy kappa reproduces the frozen digest : {n_ok} / {n_or} seeds with an oracle")
    print(f"  clean  kappa reproduces the frozen digest : "
          f"{sum(1 for r in rows if r['clean_matches_frozen'])} / {n_or}")
    print(f"  c0 bit-identical across kappa             : "
          f"{sum(1 for r in rows if r['c0_bit_identical'])} / {len(rows)}"
          "   (False expected -- this is the divergence)")
    if oracle_hit is not None:
        print(f"  L0=0.8 c0 bit-identical to the independent %TEMP% pkl : {oracle_hit}")
    print("=" * 78, flush=True)

    blob = {"rs": RS, "kappa_legacy": K_LEGACY, "kappa_clean": K_CLEAN,
            "kappa_rel_diff": abs(K_CLEAN - K_LEGACY) / K_LEGACY,
            "_L0S": L0S, "nearest_kappa": k_near,
            "cache": "purged before every call; all ten calls are recomputations",
            "frozen_by_L0": {f"{k:.2f}": v[0] for k, v in FROZEN_BY_L0.items()},
            "temp_value_oracle": {"L0": 0.8, "pkl_sha16": TEMP_C08_SHA, "c": TEMP_C08},
            "legacy_matches_frozen_count": n_ok, "with_oracle": n_or,
            "L0_08_legacy_bit_identical_to_temp_oracle": oracle_hit,
            "seeds": rows,
            "frozen_tree": "unchanged (readonly_frozen re-fingerprints on exit)"}
    return blob, (0 if n_ok == n_or else 1)


def write_blob(blob, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print(f"  wrote {out}")
    print("  frozen tree: UNCHANGED -- the guard verified it on exit")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(CLEAN, "logs", "diag_c0_kappa.json"))
    args = ap.parse_args(argv)

    print("=" * 78)
    print("  Blocker B / B2-B3 -- is c0 a function of kappa alone?  (cache purged)")
    print("=" * 78)
    print(f"  frozen store (read-only)   {FROZEN_CKPT}")
    print(f"  kappa legacy {K_LEGACY}    kappa clean {K_CLEAN:.12f}"
          f"    rel diff {abs(K_CLEAN - K_LEGACY) / K_LEGACY:.3e}")
    print("  building the notebook namespace against a WORKING COPY\n", flush=True)

    with readonly_frozen(FROZEN_CKPT) as work:
        E = namespace_cells(work)
        blob, rc = run(E, work)
    write_blob(blob, args.out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
