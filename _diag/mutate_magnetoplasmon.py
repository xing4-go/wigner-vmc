"""Mutation-test the magnetoplasmon contract suite.

Each mutation is a one-line change that should make the suite fail.  A mutation
the suite survives is a claim the suite does not actually check.

Discipline, because this repo has been burned here before: the file is read and
written as BYTES (no text mode, so no newline translation), the original bytes
are restored in a ``finally``, and the restore is verified by SHA-256 before the
next mutation runs.  A mutation that leaves the tree dirty would silently
invalidate every result after it.
"""
import hashlib
import subprocess
import sys

SCRIPT = "examples/figure_construction/magnetoplasmon.py"

#: (label, old, new).  ``old`` must occur exactly once.
MUTATIONS = [
    ("drop the 1/2 in Omega = q^2/(2S)",
     "return q ** 2 / (2.0 * s)", "return q ** 2 / s"),
    ("square the 2 in Omega's denominator",
     "return q ** 2 / (2.0 * s)", "return q ** 2 / (4.0 * s)"),
    ("drop sqrt(2 pi/nu) from the axis",
     "return np.asarray(q_lb, float) * math.sqrt(2.0 * math.pi / float(nu))",
     "return np.asarray(q_lb, float)"),
    ("hard-code the axis factor",
     "return np.asarray(q_lb, float) * math.sqrt(2.0 * math.pi / float(nu))",
     "return np.asarray(q_lb, float) * 2.5066282746310002"),
    ("drop kappa from the classical curve",
     "return np.sqrt(1.0 + float(kappa) * np.asarray(q_lb, float))",
     "return np.sqrt(1.0 + np.asarray(q_lb, float))"),
    ("scale the classical curve by a fudge factor",
     "return np.sqrt(1.0 + float(kappa) * np.asarray(q_lb, float))",
     "return np.sqrt(1.0 + 1.07 * float(kappa) * np.asarray(q_lb, float))"),
    ("use omega_mp whole instead of its square root",
     "return np.sqrt(1.0 + float(kappa) * np.asarray(q_lb, float))",
     "return 1.0 + float(kappa) * np.asarray(q_lb, float)"),
    ("return inf instead of raising on a zero S",
     "        i = int(np.argmax(bad))\n        raise ValueError(",
     "        i = int(np.argmax(bad))\n        return q ** 2 / (2.0 * np.where(s == 0, 1.0, s))\n    if False:\n        raise ValueError("),
    ("accept a pre-fix source",
     "    if not cands[0][\"post_fix\"]:", "    if False:"),
    ("take the oldest candidate instead of the newest post-fix one",
     "cands.sort(key=lambda r: (r[\"post_fix\"], r[\"mtime\"]), reverse=True)",
     "cands.sort(key=lambda r: r[\"mtime\"])"),
    ("fall back to any coupling of the same phase",
     "    cands = [r for r in index\n             if r[\"phase\"] == phase and r[\"rs\"] == float(rs)\n             and r[\"budget\"] == want]",
     "    cands = [r for r in index\n             if r[\"phase\"] == phase and r[\"budget\"] == want]"),
    ("ignore the budget selector",
     "             and r[\"budget\"] == want]", "             and True]"),
    ("stop resolving the full alias",
     "    return BUDGET_ALIASES.get(str(budget), str(budget))", "    return str(budget)"),
    ("build the analytic curve from a fitted constant instead of the exact formula",
     '"S": st.exact_lll_sq(q),', '"S": q ** 2 / 2.0,'),
    # ----------------------------------------------------------------------
    # The reduction S(q-vector) -> S(q).  This is the correction itself, so
    # these are the mutants that matter most: each one is a way to get a
    # smooth, plausible figure that is not the physics the paper describes.
    # ----------------------------------------------------------------------
    ("average 1/S instead of S in the liquid shell mean",
     "        mean.append(float(S[idx].mean()))",
     "        mean.append(float(1.0 / np.mean(1.0 / S[idx])))"),
    ("shell-average the crystal instead of cutting it",
     "    if phase == \"crystal\":\n        return directional_cut(qvec, S, dhat)",
     "    if phase == \"crystal\":\n        return shell_average(qvec, S)"),
    ("cut the crystal along this package's own x instead of the paper's",
     "        v = np.asarray(torus.G1, float) + np.asarray(torus.G2, float)",
     "        v = np.array([1.0, 0.0])"),
    ("project every momentum onto the cut instead of requiring collinearity",
     "    sel = (perp <= tol) & (along > 0.0)",
     "    sel = (perp <= 1e9) & (along > 0.0)"),
    ("let the cut carry both q and -q",
     "    sel = (perp <= tol) & (along > 0.0)",
     "    sel = (perp <= tol) & (np.abs(along) > 0.0)"),
    ("stop checking the source was measured on this lattice",
     "    if residual > 1e-6:", "    if False:"),
    ("stop checking a shell holds one |q|",
     "    if np.any(np.asarray(within) > tol):", "    if False:"),
    ("stop normalising the crystal direction",
     "    return v / n", "    return v"),
    ("stop stamping a quick figure",
     "    return None if str(budget) == \"reproduction\" else QUALITY_STAMP",
     "    return None"),
    ("alter the stored S(q) on the way in",
     "    S = np.asarray(z[key], float)", "    S = 0.999 * np.asarray(z[key], float)"),
    ("smooth S(q) on the way in",
     "    return qn, S", "    return qn, np.convolve(S, np.ones(3) / 3.0, mode=\"same\")"),
    ("smooth q as well as S",
     "    return qn, S", "    return np.sort(qn), S"),
    ("let phase_of always choose the liquid",
     "    return \"liquid\" if float(rs) < RS_C else \"crystal\"",
     "    return \"liquid\""),
    ("let output_slug always call itself the paper's",
     "    return \"rs\" + \"__\".join(_slug_rs(r).replace(\".\", \"p\") for r in rs_list)",
     "    return \"paper\""),
    ("drop the roton depth guard",
     "        \"deeper_than_imbalance\": bool(depth > noise),",
     "        \"deeper_than_imbalance\": True,"),
    ("search for the roton without reference to Q_WC",
     "    best = min(idx, key=lambda i: abs(q_shell[i] - q_wc_value))",
     "    best = idx[0]"),
    ("stop binning momenta into |q| shells",
     "    new = np.ones(qs.size, bool)\n    new[1:] = ~np.isclose(qs[1:], qs[:-1], rtol=1e-9, atol=0.0)",
     "    new = np.ones(qs.size, bool)"),
    ("report a shallow minimum as resolved",
     "        \"resolved\": bool(depth > noise),", "        \"resolved\": True,"),
    ("ignore the shell scatter in the noise",
     "    noise = max(scatter, imbalance)", "    noise = imbalance"),
    ("report the extreme point instead of the shell mean",
     "        \"Omega_roton\": float(mean[best]),",
     "        \"Omega_roton\": float(spread[best]),"),
    ("let the too-few-shells early return omit the found flag",
     "    if q_shell.size < 3:\n        base[\"found\"] = False",
     "    if q_shell.size < 3:"),
    ("let the no-minimum early return omit the found flag",
     "    if not idx:\n        base[\"found\"] = False",
     "    if not idx:"),
    # ----------------------------------------------------------------------
    # The command-line coupling spelling.  The flags group the input; the
    # phase is phase_of(r_s), and the flags must not be able to change it.
    # A hard gate refusing a mis-named coupling was tried and removed, so
    # there is no longer a check to mutate here -- what is left to guard is
    # that the phase still comes from r_s and nothing else.
    # ----------------------------------------------------------------------
    ("let --rs and the phase-named flags be given together",
     "    if args.rs is not None and named:", "    if False:"),
    ("drop the crystal couplings from the phase-named flags",
     "    for attr in (\"liquid_rs\", \"crystal_rs\"):",
     "    for attr in (\"liquid_rs\",):"),
    ("stop defaulting to the paper's points when no coupling is given",
     "    if not named:\n        return list(DEFAULT_RS)", "    if False:\n        return list(DEFAULT_RS)"),
    # ----------------------------------------------------------------------
    # The figure is measurements, and these are the two ways the decorations
    # grew back on it: turn the dashed families on by default, and mark the
    # roton minimum.  The second is not cosmetic -- the marker sat on the data
    # point it named.
    # ----------------------------------------------------------------------
    ("draw the dashed classical curves by default",
     '    p.add_argument("--classical", dest="classical", action="store_true",\n                   default=False,',
     '    p.add_argument("--classical", dest="classical", action="store_true",\n                   default=True,'),
    ("let the renderer draw the dashed families unless told not to",
     "def write_figure(fig_dir, curves, budget, show_classical=False):",
     "def write_figure(fig_dir, curves, budget, show_classical=True):"),
    # ----------------------------------------------------------------------
    # The refusal message.  It answers "why was my coupling not found", and
    # the wrong answer -- listing a coupling the lookup would refuse -- reads
    # as "that value is not allowed" when the value was never the problem.
    # ----------------------------------------------------------------------
    ("list couplings the lookup would refuse because of the truncation",
     '        return not (phase == "crystal" and nmax is not None\n'
     '                    and r["nmax"] != int(nmax))',
     "        return True"),
    ("advertise a pre-fix coupling as available",
     '        if not r["post_fix"]:\n            return False',
     '        if False:\n            return False'),
    # This one survives, and it is an EQUIVALENT MUTANT rather than a coverage
    # gap.  For a triangular lattice at nu = 1, a^2 = 4 pi/sqrt(3), so
    # |G_1| = 4 pi/(sqrt(3) a) = a^2/a = a -- the nearest-neighbour spacing and
    # the first reciprocal vector are the same number.  Measured: the two
    # expressions differ by 4.4e-16, one ulp.  No test can distinguish them
    # because there is nothing to distinguish; kept here so a later reader does
    # not try.
    ("use the lattice constant where |G_1| is meant (EQUIVALENT, expected to survive)",
     "    return 4.0 * math.pi / (math.sqrt(3.0) * q_wc_lattice_constant(nu))",
     "    return q_wc_lattice_constant(nu)"),
]


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_suite():
    p = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/test_magnetoplasmon_contract.py", "-q",
                        "--no-header", "-x", "--tb=no"],
                       capture_output=True, text=True)
    return p.returncode, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""


def main():
    with open(SCRIPT, "rb") as fh:
        original = fh.read()
    digest = hashlib.sha256(original).hexdigest()
    print(f"baseline sha256 {digest}")
    code, line = run_suite()
    print(f"baseline suite: rc={code}  {line}")
    if code != 0:
        print("BASELINE IS ALREADY RED -- fix that first.")
        return 1
    print()

    survived, killed = [], []
    try:
        for label, old, new in MUTATIONS:
            text = original.decode("utf-8")
            if text.count(old) != 1:
                print(f"  SKIP  {label}: anchor occurs {text.count(old)}x")
                survived.append((label, "anchor"))
                continue
            mutated = text.replace(old, new).encode("utf-8")
            with open(SCRIPT, "wb") as fh:
                fh.write(mutated)
            try:
                code, line = run_suite()
            finally:
                with open(SCRIPT, "wb") as fh:
                    fh.write(original)
            got = sha(SCRIPT)
            if got != digest:
                print(f"  RESTORE FAILED after {label!r}: {got} != {digest}")
                return 2
            if code == 0:
                print(f"  SURVIVED  {label}")
                survived.append((label, "green"))
            else:
                print(f"  killed    {label}")
                killed.append(label)
    finally:
        with open(SCRIPT, "wb") as fh:
            fh.write(original)
    print()
    print(f"restored, sha256 {sha(SCRIPT)} (match: {sha(SCRIPT) == digest})")
    print(f"killed {len(killed)}, survived {len(survived)} of {len(MUTATIONS)}")
    for label, why in survived:
        print(f"  NOT COVERED ({why}): {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
