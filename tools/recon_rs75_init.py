"""Dump the r_s = 75 initial conditions + geometry out of the FROZEN notebook.

Stage 2E runs the clean package end to end; the frozen notebook supplies only the
STARTING POINT -- the warm-started Jastrow ``c0`` and the projected-Gaussian
``v0`` -- and the lattice constants.  Everything numeric is written to JSON and
nothing is optimised here.

READ-ONLY, AND ENFORCED
-----------------------
This script exec's the notebook's own cells, and several of them call functions
wrapped in the notebook's ``cached()`` store, which WRITES on a cache MISS.  It
therefore never points ``CKPT`` at the frozen tree: it is given a working COPY
by ``tools/frozen_guard.py`` and the frozen tree is re-fingerprinted on exit.

    frozen _diag/ckpt_rebuild  --COPY-->  temp working CKPT  -->  exec cells
                                                |
                            FrozenTreeChanged raised on exit if the frozen
                            tree moved at all.

That is the fix for the 2026-10-02 incident, in which exactly this script (an
earlier version) pointed ``CKPT`` at the frozen store and overwrote
``jastrow_crystal_k53.033.{pkl,meta}``.  The account is in
``wigner_vmc_clean/incident_20261002/``.  See ``tools/frozen_guard.py`` for why
there are two independent defences.

Output: ``wigner_vmc_clean/results/bench_rs75/initial_conditions.json``, which
``scripts/bench_rs75.py --init`` reads.

TWO COUPLINGS, AND ONLY ONE OF THEM IS PHYSICS  (Blocker B)
-----------------------------------------------------------
The starting points are the CAMPAIGN'S, and the campaign built them at
``round(rs / sqrt(2.0), 4)``.  That is a different number from the physics value
``rs / sqrt(2.0)`` -- 53.033 against 53.03300858899106 -- and using the physics
value here was wrong for a reason that has nothing to do with accuracy: it makes
the ``_init_digest`` stop matching the frozen tags, so the starting point can no
longer be shown to be the legacy one, which is the only thing this file is for.

So the JSON records both, under names that say which is which:

    "kappa"       53.03300858899106   the Hamiltonian's; full precision
    "kappa_init"  53.033              the campaign's; builds c0 and c_liquid

``kappa`` is unchanged and still what ``Setup`` consumes.  Nothing about the
physics moved; the artifact simply stopped mislabelling its own starting point.
``tests/test_init_provenance.py`` checks the digests against the frozen tags.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")          # BEFORE the exec: a GUI backend blocks on plt.show()
import matplotlib.pyplot as plt  # noqa: E402,F401
import numpy as np

import frozen_guard
from frozen_guard import FROZEN_CKPT, assert_not_frozen, readonly_frozen

PROJ = frozen_guard.PROJ
CLEAN = frozen_guard.CLEAN
NB = os.path.join(PROJ, "WignerCrystal_to_HallLiquid_LLRotation.ipynb")
OUT_JSON = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")

#: `kappa` is the physics: full floating-point precision, no rounding.  It is
#: what the Hamiltonian and both wavefunctions are built from, and it is what
#: `bench_rs75.Setup` reads out of the JSON.  It must never be rounded here --
#: `tests/test_kappa_convention.py` guards that.
#:
#: `KAPPA_INIT` is PROVENANCE, not physics.  The frozen campaign built every
#: Jastrow starting point at `round(rs / sqrt(2.0), 4)` (`_diag/part1_scan.py:404`
#: and `:414`), so that is the coupling the legacy run STARTED from, and the
#: starting points in this file must be the legacy ones or two things break at
#: once: the `_init_digest` no longer matches the frozen tags (nothing can be
#: checked against the record), and the clean run does not start where the
#: legacy run started (which is the whole point of dumping them).
#:
#: The two differ by 8.6e-6 in kappa, and the cache TAG cannot tell them apart:
#: `k{kappa:g}` renders BOTH as `53.033`.  That is Blocker B.  Both values are
#: recorded in the JSON under names that say which is which.
RS = 75.0
KAPPA = RS / np.sqrt(2.0)                       # 53.03300858899106 -- physics

KAPPA_INIT = round(RS / np.sqrt(2.0), 4)        # 53.033 -- the campaign's
KAPPA_INIT_MODE = "legacy-regression"
KAPPA_INIT_SOURCE = ("round(rs/sqrt(2), 4) -- _diag/part1_scan.py:404 (crystal) "
                     "and :414 (liquid)")

#: Where the notebook's `savefig` goes.  Outside the project on purpose: the
#: notebook owns every figs_LLRotation PNG and must not re-render one here.
FIGDIR = os.path.join(os.environ.get("TEMP", "/tmp"), "rs75_recon_figs")

#: The cells that build the namespace this script reads.  The same list the
#: pre-incident script used, so the namespace is the one that was verified to
#: work; several of these cells call `cached()` functions, which is exactly why
#: CKPT has to be a working copy.
CELLS = list(range(53)) + [61, 62, 76, 78, 80, 82]

#: Everything `build_init` reads.  Checked in one pass so a missing binding names
#: itself, instead of surfacing as a KeyError partway through the dump.
REQUIRED = (
    "ne", "mesh", "L1", "L2", "G1", "G2", "sc_to_cart", "sites", "lints", "lcart",
    "_lints8", "_lcart8", "LUMAX", "KAPPAS", "L0_GRID_J", "J_OPT", "jastrow_opt",
    "gaussian_overlap_seed", "v_from_overlap", "_llb8B", "NMAX_STAGE_B",
)


def namespace_cells(ckpt_dir, figdir=FIGDIR, cells=CELLS, verbose=True):
    """Exec the frozen notebook's cells with ``CKPT`` pointed at ``ckpt_dir``.

    ``ckpt_dir`` MUST be clear of the frozen tree.  That is asserted here, as the
    first statement, so a misconfigured caller fails in milliseconds rather than
    after the build -- and so a test can exercise this exact refusal without
    paying for a notebook exec.
    """
    assert_not_frozen(ckpt_dir, FROZEN_CKPT)

    os.makedirs(figdir, exist_ok=True)
    os.environ["LLROT_FIGDIR"] = figdir

    nbj = json.load(open(NB, encoding="utf-8"))
    src_all = [("".join(c["source"]), c["cell_type"]) for c in nbj["cells"]]
    E = {"__name__": "__main__"}
    t0 = time.time()
    for i in cells:
        src, kind = src_all[i]
        if kind != "code" or not src.strip():
            continue
        # the notebook's own Colab path; pointing it at ckpt_dir is the whole point
        src = src.replace('CKPT = "/content/qhvmc_checkpoints"', f'CKPT = r"{ckpt_dir}"')
        tc = time.time()
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings("ignore", message=".*non-interactive.*")
            exec(compile(src, f"cell{i}", "exec"), E)
        if verbose:
            print(f"  cell {i:3d} done in {time.time() - tc:7.1f}s "
                  f"(total {time.time() - t0:7.1f}s)", flush=True)

    assert E["CKPT"] == ckpt_dir, f"CKPT not redirected: {E['CKPT']!r}"
    missing = [k for k in REQUIRED if k not in E]
    if missing:
        raise KeyError(f"the namespace is missing {missing} -- the CELLS list needs "
                       f"the cell that binds them")
    if verbose:
        print(f"namespace built in {time.time() - t0:.1f}s", flush=True)
    return E


def _A(x):
    return np.asarray(x, float).tolist()


def build_init(E, verbose=True):
    ne = int(E["ne"])
    nk = int(E["mesh"].shape[0])
    KAPPAS = np.asarray(E["KAPPAS"], float)
    L0_GRID_J = np.asarray(E["L0_GRID_J"], float)
    _L0S = [float(L0_GRID_J[i])
            for i in np.linspace(0, len(L0_GRID_J) - 1, 5).round().astype(int)]
    # RS, KAPPA and KAPPA_INIT are module-level: one definition, so the JSON's
    # `kappa` and the coupling the starting points are built at cannot drift
    # apart here.
    k0 = float(min(KAPPAS, key=lambda kk: abs(kk - KAPPA_INIT)))
    if verbose:
        print(f"ne={ne} nk={nk} NMAX_STAGE_B={E['NMAX_STAGE_B']}  KAPPAS={KAPPAS}")
        print(f"_L0S={_L0S}   kappa(75)={KAPPA}   kappa_init={KAPPA_INIT}"
              f"   nearest grid {k0}")

    def jopt_at(kind, kappa, L0=None):
        return np.asarray(E["jastrow_opt"](kind, kappa, L0=L0,
                          warm=E["J_OPT"][(kind, k0)] * (kappa / k0))["c"], float)

    out = {
        "_provenance": ("WignerCrystal_to_HallLiquid_LLRotation.ipynb cells "
                        + ",".join(str(c) for c in CELLS)),
        "_written_by": "wigner_vmc_clean/tools/recon_rs75_init.py",
        "rs": RS, "kappa": KAPPA, "ne": ne, "nk": nk,
        "kappa_init": KAPPA_INIT,
        "kappa_init_mode": KAPPA_INIT_MODE,
        "kappa_init_source": KAPPA_INIT_SOURCE,
        "NMAX_STAGE_B": int(E["NMAX_STAGE_B"]),
        "KAPPAS": _A(KAPPAS), "L0_GRID_J": _A(L0_GRID_J), "_L0S": _L0S,
        "nearest_kappa": k0,
        "J_OPT_nearest": {
            "liquid": _A(E["J_OPT"][("liquid", k0)]),
            "crystal": _A(E["J_OPT"][("crystal", k0)]),
        },
        "L1": _A(E["L1"]), "L2": _A(E["L2"]), "G1": _A(E["G1"]), "G2": _A(E["G2"]),
        "mesh": _A(E["mesh"]), "sc_to_cart": _A(E["sc_to_cart"]),
        "sites": _A(np.asarray(E["sites"], float)),
        "lints": _A(np.asarray(E["lints"], float)),
        "lcart": _A(np.asarray(E["lcart"], float)),
        "_lints8": _A(np.asarray(E["_lints8"], float)),
        "_lcart8": _A(np.asarray(E["_lcart8"], float)),
        "LUMAX": float(E["LUMAX"]),
        "LUMAX_LL": float(E.get("LUMAX_LL", float("nan"))),
        "c_liquid": _A(jopt_at("liquid", KAPPA_INIT)),
        "crystal": {},
        "_jastrow_cache": {},
    }

    # The five conv-tier starting points: seed index si -> (L0, c0, v0).
    #
    # Whether a seed HIT or MISSed the notebook's cache is recorded rather than
    # assumed, because it decides how faithful that seed is to the frozen record.
    # `cached()` writes only on a MISS, so an unchanged mtime IS the answer.
    # The crystal entry is PURGED before the loop, and the reason is the same
    # tag blindness that caused Blocker B.  The store's
    # `jastrow_crystal_k53.033` currently holds the 2026-10-02 incident's
    # CLEAN-kappa write at L0 = 0.5; a KAPPA_INIT call at that same L0 renders a
    # byte-identical tag (`k53.033` both ways, `w-7.49` both ways), so it would
    # HIT and hand back the very value this script exists to stop using.  A
    # cache HIT is only sound when the entry was produced by the same
    # computation, and here the tag cannot establish that -- so recompute.
    #
    # This is not a deviation from the campaign: the campaign's five conv seeds
    # ran five DIFFERENT L0 values and so missed on four of them anyway, and a
    # miss returns what a hit would have.  Purging only makes the fifth honest.
    entry = os.path.join(os.environ["_RECON_CKPT"], "jastrow_crystal_k53.033")
    purged = [os.path.basename(entry) + ext for ext in (".pkl", ".meta")
              if os.path.exists(entry + ext)]
    for ext in (".pkl", ".meta"):
        try:
            os.remove(entry + ext)
        except FileNotFoundError:
            pass
    out["_jastrow_cache"]["_purged_before_loop"] = purged
    out["_jastrow_cache"]["_purging_reason"] = (
        "the tag formats kappa `{kappa:g}` and the warm start `{c0[0]:+.2f}`, so "
        "a KAPPA_INIT call at L0=0.5 renders the same tag as the incident's "
        "clean-kappa entry and would hit it")

    for si, L0 in enumerate(_L0S):
        before = os.stat(entry + ".meta").st_mtime_ns if os.path.exists(entry + ".meta") else None
        c0 = jopt_at("crystal", KAPPA_INIT, L0=L0)
        after = os.stat(entry + ".meta").st_mtime_ns if os.path.exists(entry + ".meta") else None
        hit = (before is not None and before == after)
        out["_jastrow_cache"][f"s{si}"] = {"L0": float(L0), "hit": bool(hit)}

        ov = E["gaussian_overlap_seed"](E["_llb8B"], int(E["NMAX_STAGE_B"]),
                                        E["L1"], E["L2"], E["_lints8"], E["_lcart8"],
                                        rs=RS, numx=121, L0=L0)
        v0 = np.asarray(E["v_from_overlap"](ov), complex)
        out["crystal"][f"s{si}"] = {
            "seed": si, "L0": float(L0),
            "c0": _A(c0),
            "v0_real": _A(v0.real), "v0_imag": _A(v0.imag), "v0_shape": list(v0.shape),
        }
        if verbose:
            print(f"  s{si}  L0={L0:<5g}  jastrow cache "
                  f"{'HIT (frozen bytes)' if hit else 'MISS (recomputed)'}"
                  f"  |v0| mean {np.linalg.norm(v0, axis=1).mean():.4f}"
                  f"  c0={np.round(c0, 4)}")
    return out


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    print(f"frozen store : {FROZEN_CKPT}")
    print("building the notebook namespace against a WORKING COPY\n", flush=True)

    with readonly_frozen(FROZEN_CKPT) as work:
        os.environ["_RECON_CKPT"] = work          # read by build_init for hit/miss
        print(f"working copy : {work}\n", flush=True)
        E = namespace_cells(work)
        out = build_init(E)

    out["_frozen_tree"] = "unchanged (verified by tools/frozen_guard.readonly_frozen)"
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"\nwrote {OUT_JSON}")
    print("frozen tree: UNCHANGED -- the guard verified it on exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
