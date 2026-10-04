"""ONE corrected structural point, in its own namespace, with its provenance.

Why this wrapper exists
-----------------------
`structure.py` derives its output directory from its INPUTS
alone (`request_slug` / `namespace`), so re-running the historical point
`liquid_rs75__crystal_rs75__llrot_nmax2__reproduction` after fixing
`vmc/sr.py:219` would write the corrected results **into the directory that holds
the defective ones**.  That is exactly what the brief forbids: the defective run
is the record of the old implementation and must be retained for before/after
comparison, and corrected results must never be silently mixed into it.  The
recipe's own `check_namespace_reuse` only *warns* -- it does not prevent the
overwrite.

Rather than bend the recipe's tested CLI, this wrapper redirects the two base
directories it derives its namespace from, and then stamps the corrected
provenance into the metadata it wrote:

    results/figure_construction/projection_fix/<slug>/
    figures/figure_construction/projection_fix/<slug>/

The recipe's own `structure/` tree is not touched by this program at all.

It also REFUSES to start unless `vmc/sr.py` is byte-identical to the corrected
file, so a corrected run cannot be produced from an unfixed or half-fixed tree --
the failure mode that let the defect survive a full suite in the first place.

    python examples/figure_construction/run_projection_fix_point.py \
        --liquid-rs 75 --crystal-rs 75 --crystal-nmax 2 --budget reproduction
"""
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RECIPE = os.path.join(HERE, "structure.py")
SR_PY = os.path.join(SRC, "wigner_vmc", "vmc", "sr.py")

#: The corrected file's identity.  Recorded in the delivery's report.
SR_PY_MD5 = "f5462a841d66d9906d0f29ce553426cc"
SR_PY_SHA256 = ("18b45dc6314864a22d91a4b559d1aec15b28a3ba96c3d5b10f493ca3e3941266")
SR_PY_BYTES = 17231
SR_PY_LINE = 219

#: What the corrected projection is, in the two keys the brief names.
PROJECTION_CONVENTION = "phi_bra_gaussian_ket"
PROJECTION_BUG_FIXED = True

#: The corrected namespace.  A SIBLING of `structure/`, not a subdirectory of it:
#: `structure/` is the list of points measured with the old projection, and the
#: recipe's own docstring says so.
RESULTS_BASE = os.path.join(CLEAN, "results", "figure_construction",
                            "projection_fix")
FIGDIR_BASE = os.path.join(CLEAN, "figures", "figure_construction",
                           "projection_fix")


def _digests(path):
    blob = open(path, "rb").read()
    return hashlib.md5(blob).hexdigest(), hashlib.sha256(blob).hexdigest(), len(blob)


def guarded_source():
    """Refuse to run on anything but the corrected `sr.py`, and say why."""
    md5, sha256, n = _digests(SR_PY)
    ok = (md5 == SR_PY_MD5 and sha256 == SR_PY_SHA256 and n == SR_PY_BYTES)
    print("PROJECTION FIX -- corrected-point runner")
    print("=" * 78)
    print(f"  vmc/sr.py   md5 {md5}")
    print(f"              sha256 {sha256}")
    print(f"              {n} bytes")
    if not ok:
        print()
        print("  REFUSING TO RUN: this is not the corrected source.")
        print(f"  expected md5 {SR_PY_MD5}, sha256 {SR_PY_SHA256}, {SR_PY_BYTES} bytes")
        print(f"  line {SR_PY_LINE} must read:")
        print('      ov = np.einsum("i,ikn->kn", G, bloch.conj())')
        raise SystemExit(2)
    with open(SR_PY, encoding="utf-8") as fh:
        line = fh.read().splitlines()[SR_PY_LINE - 1].strip()
    if line != 'ov = np.einsum("i,ikn->kn", G, bloch.conj())':
        print(f"  REFUSING TO RUN: line {SR_PY_LINE} reads {line!r}")
        raise SystemExit(2)
    print(f"  line {SR_PY_LINE}  {line}")
    print("  source VERIFIED -- running the corrected projection")
    print("=" * 78)
    print()
    return {"md5": md5, "sha256": sha256, "bytes": n, "line": SR_PY_LINE,
            "line_text": line}


def load_recipe():
    spec = importlib.util.spec_from_file_location("phase_structure_corrected",
                                                  RECIPE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stamp(results_dir, slug, src, argv):
    """Add the two keys the brief names, and the evidence behind them.

    Written AFTER the recipe, so the recipe stays the arbiter of everything it
    knows about; this only records a fact the recipe has no notion of.
    """
    # The directory we are about to stamp must be the one the recipe says it
    # wrote -- if these disagree, the redirection and the slug have drifted
    # apart and stamping would attribute the provenance to the wrong run.
    assert os.path.basename(results_dir) == slug, (results_dir, slug)
    path = os.path.join(results_dir, "run_metadata.json")
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta.get("request_slug") == slug, meta.get("request_slug")
    meta["projection_convention"] = PROJECTION_CONVENTION
    meta["projection_bug_fixed"] = PROJECTION_BUG_FIXED
    meta["projection_fix"] = {
        "defect": ("reversed complex conjugation in the Gaussian -> LL projection: "
                   "sum_i G* phi = <G|phi> instead of <phi|G> = sum_i phi* G"),
        "source": "src/wigner_vmc/vmc/sr.py",
        "line": src["line"],
        "before": 'ov = np.einsum("i,ikn->kn", G.conj(), bloch)',
        "after": src["line_text"],
        "corrected_source_md5": src["md5"],
        "corrected_source_sha256": src["sha256"],
        "corrected_source_bytes": src["bytes"],
        "why_it_broke_c6": ("the conjugation flips the band phase n*theta -> -n*theta; "
                            "exp(-i n theta) == exp(+i n theta) only at theta = 180 "
                            "deg, which is exactly why C2 survived and C6 did not"),
        "namespace": ("written by examples/figure_construction/"
                      "run_projection_fix_point.py, which redirects the recipe's "
                      "RESULTS/FIGDIR bases so the historical defective point under "
                      ".../structure/ is not touched"),
        "historical_point": ("results/figure_construction/structure/"
                             "liquid_rs75__crystal_rs75__llrot_nmax2__reproduction "
                             "-- retained unchanged, the record of the old "
                             "implementation"),
        "runner_argv": list(argv),
    }
    # Round-trip: the file we just wrote must parse, and must carry the keys.
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    with open(path, encoding="utf-8") as fh:
        back = json.load(fh)
    assert back["projection_convention"] == PROJECTION_CONVENTION
    assert back["projection_bug_fixed"] is True
    return path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    src = guarded_source()

    mod = load_recipe()
    # The redirection.  `main` reads these two globals at call time, so this is
    # the whole of the namespace change -- no recipe code is modified.
    assert mod.RESULTS != RESULTS_BASE and mod.FIGDIR != FIGDIR_BASE
    mod.RESULTS = RESULTS_BASE
    mod.FIGDIR = FIGDIR_BASE

    rc = mod.main(argv)
    if rc:
        return rc

    args = mod.parse_args(argv)
    req = mod.resolve_request(args)
    slug = mod.request_slug(req["liquid_rs"], req["crystal_rs"], req["ansatz"],
                            req["crystal_nmax"], args.budget)
    results_dir = mod.namespace(RESULTS_BASE, req["liquid_rs"], req["crystal_rs"],
                                req["ansatz"], req["crystal_nmax"], args.budget)
    path = stamp(results_dir, slug, src, argv)

    print()
    print("=" * 78)
    print("  CORRECTED POINT -- provenance stamped")
    print("=" * 78)
    print(f"  projection_convention = {PROJECTION_CONVENTION}")
    print(f"  projection_bug_fixed  = {PROJECTION_BUG_FIXED}")
    print(f"  metadata              {os.path.relpath(path, CLEAN)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
