"""One real ROOT + one real NEST, in a scratch directory. Not a result."""
import importlib.util, json, os, shutil, sys, tempfile, time
sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location(
    "pc", "examples/figure_construction/phase_competition.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class A:
    budget = "quick"; n_electrons = 36; ansatz = "ll_rotation"
    l0_grid = False; redo = False; engine = "clean"; verbose = False
    construction = "nested"


args = A()
tmp = tempfile.mkdtemp(prefix="pc_nested_smoke_")
t0 = time.time()
runs = {}
res = m.CrystalResolver(args, tmp, runs)
rec1, st1 = res.state(75.0, 1, 0)
rec2, st2 = res.state(75.0, 2, 0)
print("\n--- root ---")
print({k: rec1[k] for k in ("construction", "root_initialization", "root_L0",
                            "energy_per_particle", "error")})
print("--- nested ---")
print({k: rec2.get(k) for k in (
    "construction", "parent_nmax", "parent_rs", "parent_init_id", "parent_key",
    "parent_state_digest", "root_L0", "nesting_head_residual",
    "nesting_tail_residual", "delta_e_vs_parent", "z_vs_parent")})
print("digest matches root:", rec2["parent_state_digest"] == m.state_digest(st1))
print("state files:", sorted(os.listdir(os.path.join(tmp, "states"))))
print("v shapes:", st1.v.shape, st2.v.shape, "c equal:", (st1.c == st2.c).all())
print("child v first column == parent v:", (st2.v[:, :1] == st1.v).all())
print("child v second column all zero:", (st2.v[:, 1] == 0).all())
print("elapsed %.1fs" % (time.time() - t0))
# resume: the same request must come back from the store, not recompute
runs2 = m.load_store(tmp)
res2 = m.CrystalResolver(args, tmp, runs2)
t1 = time.time()
r2a, _ = res2.state(75.0, 1, 0)
r2b, _ = res2.state(75.0, 2, 0)
print("resume identical:", r2a["energy_per_particle"] == rec1["energy_per_particle"],
      r2b["energy_per_particle"] == rec2["energy_per_particle"],
      "in %.2fs" % (time.time() - t1))
print("TMP", tmp)
