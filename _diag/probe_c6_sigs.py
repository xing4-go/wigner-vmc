"""Read-only signature probe for crystal_c6_trajectory.py.  No runs, no writes."""
import inspect
import numpy as np

from wigner_vmc import VMC, load_budget, resolve, theta, theta_parts
from wigner_vmc.api import NJ, OVERLAP_NUMX
from wigner_vmc.analysis import structure as st
from wigner_vmc.vmc import sr
from wigner_vmc.vmc.sampler import sample
import wigner_vmc.api as api


def sig(fn, name):
    try:
        print(f"  {name}{inspect.signature(fn)}")
    except Exception as e:                                   # builtins etc.
        print(f"  {name}: <{type(e).__name__}: {e}>")


print("== VMC methods ==")
for n in ("maker", "basis", "crystal_v0", "crystal_R0", "run", "nest", "_run_from"):
    sig(getattr(VMC, n, None), f"VMC.{n}")

print("\n== sr ==")
for n in ("sr_optimize_joint", "sr_optimize_jastrow", "sr_pilot_scale",
          "jastrow_vector", "gaussian_overlap_seed", "v_from_overlap", "drummond_width"):
    sig(getattr(sr, n, None), f"sr.{n}")

print("\n== structure ==")
for n in ("structure_factor", "bragg_ratio", "site_midpoint_contrast",
          "sq_error_bars"):
    sig(getattr(st, n, None), f"st.{n}")

print("\n== api ==")
print(f"  NJ = {NJ}   OVERLAP_NUMX = {OVERLAP_NUMX}")

print("\n== budget ==")
for b in ("quick", "smoke", "reproduction"):
    bud = load_budget(b)
    print(f"  [{b}] provenance={getattr(bud, 'provenance', '<none>')!r}")
    print(f"        width_for(0) = {bud.width_for(0)}   width_for(4) = {bud.width_for(4)}")
    print(f"        protocol keys = {sorted(bud.protocol)}")
    print(f"        sr        = {bud.sr}")
    print(f"        crystal_measure = {bud.crystal_measure}")
    print(f"        proto sr_steps={bud.protocol.get('sr_steps')} "
          f"sr_sweeps={bud.protocol.get('sr_sweeps')} "
          f"sr_equil={bud.protocol.get('sr_equil')} "
          f"sr_snap={bud.protocol.get('sr_snap')} "
          f"meas_sweeps={bud.protocol.get('meas_sweeps')} "
          f"meas_equil={bud.protocol.get('meas_equil')}")

print("\n== resolve / VMC attrs / maker ==")
cfg = resolve(phase="crystal", rs=90.0, N=36, nmax=1, init_id=0, budget="quick")
print(f"  resolve(...).rng_seed = {cfg.rng_seed}   type {type(cfg).__name__}")
print(f"  cfg fields = {sorted(vars(cfg)) if hasattr(cfg, '__dict__') else cfg}")

vmc = VMC(N=36, rs=90.0, phase="crystal", nmax=1)
for a in ("N", "rs", "kappa", "nmax", "n_bands", "ansatz", "phase"):
    print(f"  vmc.{a} = {getattr(vmc, a, '<MISSING>')!r}")
print(f"  vmc.lat type = {type(vmc.lat).__name__}")
for a in ("nk", "L1", "L2", "ov_ai", "ov_ac", "C", "A1", "A2", "sc"):
    print(f"  vmc.lat.{a} = {getattr(vmc.lat, a, '<MISSING>')!r}")

mk = vmc.maker
print(f"  vmc.maker is callable: {callable(mk)}")
try:
    print(f"  inspect.signature(vmc.maker) = {inspect.signature(vmc.maker)}")
except Exception as e:
    print(f"  maker sig err {e}")
try:
    f0 = vmc.maker()
    print(f"  vmc.maker() -> {type(f0).__name__}; callable {callable(f0)}")
except Exception as e:
    print(f"  vmc.maker() err: {type(e).__name__}: {e}")
try:
    f1 = vmc.maker(v=np.zeros((vmc.lat.nk, vmc.n_bands - 1), complex))
    print(f"  vmc.maker(v=...) -> {type(f1).__name__}; callable {callable(f1)}")
except Exception as e:
    print(f"  vmc.maker(v=...) err: {type(e).__name__}: {e}")

print("\n== RunResult surface ==")
import wigner_vmc.api as _a
for name, obj in sorted(vars(_a).items()):
    if isinstance(obj, type) and "Result" in name:
        print(f"  {name}: {sorted(getattr(obj, '__dataclass_fields__', {}) or [a for a in dir(obj) if not a.startswith('_')])}")

print("\n== params ==")
print(f"  VMC.__init__{inspect.signature(VMC.__init__)}")
print(f"  sample{inspect.signature(sample)}")
