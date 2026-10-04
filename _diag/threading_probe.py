"""One nmax=1 crystal root, in a scratch dir, printed to full precision.

Used to decide whether the phase_competition `n_max=1` control differs from the
delivered independent store because of BLAS reduction order (thread count) or
because something in the code changed. Run it twice, once with the BLAS thread
counters pinned to 1 and once without. Not a result.

    python _diag/threading_probe.py 35
"""
import importlib.util, os, sys, tempfile, time

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location(
    "pc", "examples/figure_construction/phase_competition.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class A:
    budget = "quick"; n_electrons = 36; ansatz = "ll_rotation"
    l0_grid = False; redo = False; engine = "clean"; verbose = False
    construction = "nested"


rs = float(sys.argv[1])
args = A()
tmp = tempfile.mkdtemp(prefix="pc_thread_probe_")
runs = {}
res = m.CrystalResolver(args, tmp, runs)
t0 = time.time()
rec, _ = res.state(rs, 1, 0)
print("vars   %s" % m.blas_env())
print("rs=%-7g E/N = %.15f  +- %.9f   [%.1fs]"
      % (rs, rec["energy_per_particle"], rec["error"], time.time() - t0))
print("REPR    %r" % (rec["energy_per_particle"],))
