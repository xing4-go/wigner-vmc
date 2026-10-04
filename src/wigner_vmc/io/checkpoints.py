"""Readers for the legacy Monte-Carlo checkpoints.  READS ONLY.

    _diag/ckpt_rebuild/<name>.pkl    the payload
    _diag/ckpt_rebuild/<name>.meta   the tag string the notebook's `cached()` wrote

Why these are read here rather than in a figure
-----------------------------------------------
The Gaussian workflow's figures do not read a results JSON; their frozen result is a
pickle of Monte-Carlo output.  That is still a frozen result, so it enters the tree the
same way the LL-rotation JSON does -- through `io/` -- and the figure receives numbers,
never a file handle.  The rule is about the SHAPE of the data flow, not about the file
format: `frozen result -> io -> analysis -> figure -> PNG`, and never
`figure -> run VMC -> analyse -> savefig`.

The tag is not decoration
-------------------------
Every `.pkl` sits beside a `.meta` holding the self-describing tag its producer wrote,
e.g.

    v1|ne36|LU30|J5|crystJ|k32|L0.5|n600|e300|s17|sig0.3|sn8|i182

The tag records the parameters the payload was produced under -- 36 electrons, 600
sweeps, seed 17 -- so a payload can never be silently reused under different settings.
This module therefore refuses to load a `.pkl` whose `.meta` is missing: a checkpoint
without its tag is an anonymous array, and an anonymous array is exactly the kind of
thing that gets compared against numbers it does not correspond to.

What is deliberately NOT read
-----------------------------
The `*off` families (`crystalJoff_*`, `liquidJoff_*`) belong to the uncorrelated and
no-Jastrow comparisons whose figures (`fig03`, `fig04`, `fig05`, `fig11`) are excluded
from migration by the Stage 1 scope.  The readers exist; nothing calls them for a
migrated figure, so no code path can accidentally resurrect those plots.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class CheckpointError(RuntimeError):
    pass


@dataclass
class Checkpoint:
    """One payload plus the tag and file it came from."""
    name: str
    tag: str
    path: str
    payload: object
    fields: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.payload[k]

    def get(self, k, default=None):
        return self.payload.get(k, default)

    def __contains__(self, k):
        return k in self.payload

    @property
    def snaps(self) -> List[np.ndarray]:
        return self.payload["snaps"]

    @property
    def n_snaps(self) -> int:
        return len(self.payload["snaps"])

    def tag_fields(self) -> List[str]:
        return self.tag.split("|") if self.tag else []

    def tag_field(self, key: str) -> Optional[str]:
        """One `|`-separated field of the tag, matched on its prefix.

        Tags are positional-free: each field is `<prefix><value>`, so `k32` is found by
        asking for prefix `k`.  Returns None when no field has the prefix.

        RAISES when more than one field has it.  The real tag

            v1|ne36|LU30|J5|crystJ|k32|L0.5|n600|e300|s17|sig0.3|sn8|i182

        contains both `ne36` (electrons) and `n600` (sweeps), so asking for prefix `n`
        has two answers and `startswith` silently returns whichever comes first -- here
        `e36`, i.e. the electron count misread as a sweep count, with no error anywhere.
        A reader that guesses between two plausible fields is worse than one that
        refuses, because the wrong number flows into a plot looking exactly like a right
        one.  Ask for `ne` or `n` unambiguously, or use `tag_fields()` and pick.
        """
        hits = [p for p in self.tag_fields() if p.startswith(key)]
        if not hits:
            return None
        if len(hits) > 1:
            raise CheckpointError(
                f"{self.name}: tag prefix {key!r} is ambiguous -- it matches {hits} in "
                f"{self.tag!r}.  Use a longer prefix, or tag_fields() and choose "
                f"explicitly.")
        return hits[0][len(key):]


def read_checkpoint(ckpt_dir: str, name: str, require_meta: bool = True) -> Checkpoint:
    pkl = os.path.join(ckpt_dir, name + ".pkl")
    if not os.path.exists(pkl):
        raise CheckpointError(f"no checkpoint {name!r} in {ckpt_dir}")
    meta = pkl[:-4] + ".meta"
    tag = ""
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as fh:
            tag = fh.read().strip()
    elif require_meta:
        raise CheckpointError(
            f"{name}.pkl has no .meta.  Refusing to load an untagged payload: the tag is "
            f"what records the parameters it was produced under, and without it there is "
            f"no way to tell which settings these numbers belong to.")
    with open(pkl, "rb") as fh:
        payload = pickle.load(fh)
    return Checkpoint(name=name, tag=tag, path=pkl, payload=payload)


def _energy_fields(payload: dict) -> dict:
    """The E/T/V tuple, with the complex V cast to float.

    `V` is stored complex with a zero imaginary part (the legacy engine carried a
    complex potential accumulator).  Casting here is not cosmetic: a complex dtype
    propagates silently through arithmetic and turns an energy into a complex number,
    which then sorts and compares without error and produces a plot that is wrong in a
    way nothing complains about.
    """
    out = {}
    for k in ("E", "E_err", "T", "T_err", "V", "V_err", "acc", "n", "sigma"):
        if k in payload:
            v = payload[k]
            out[k] = float(np.real(v)) if k in ("V", "T", "E") else v
    if "T_err" in payload:
        out["T_err"] = float(payload["T_err"])
    if "V_err" in payload:
        out["V_err"] = float(payload["V_err"])
    return out


# --------------------------------------------------------------------------
# the three shapes the migrated figures need
# --------------------------------------------------------------------------
def read_energy(ckpt_dir: str, name: str) -> Checkpoint:
    """An energy record: E, E_err, T, V, acc, n, sigma, snaps.

    `E` here is the total over N electrons, as the legacy engine reported it -- NOT a
    per-particle energy.  The LL-rotation store's `E_perpart` is a different quantity
    and the two must never be mixed in one comparison.
    """
    c = read_checkpoint(ckpt_dir, name)
    c.fields = _energy_fields(c.payload)
    return c


def read_ensemble(ckpt_dir: str, name: str) -> Checkpoint:
    """A configuration ensemble: snaps, acc, sigma -- no energy.

    Used for every structure observable.  Carries no energy on purpose, so a caller
    cannot accidentally treat a structure run as an energy measurement.
    """
    c = read_checkpoint(ckpt_dir, name)
    c.fields = {k: c.payload[k] for k in ("acc", "sigma") if k in c.payload}
    return c


def read_structure_reference(ckpt_dir: str, name: str = "valid_lll_structure"
                             ) -> Checkpoint:
    """The filled-LLL calibration: snapshots plus the S(q) and g(r) it produced.

    The one state in this project with a closed-form S(q) = 1 - exp(-q^2/2) and
    g(r) = (N/(N-1))(1 - exp(-r^2/2)), which is why it is the calibration of the whole
    structure-factor pipeline rather than just another data point.
    """
    c = read_checkpoint(ckpt_dir, name)
    c.fields = {"Sq_all": c.payload["Sq_all"], "r": c.payload["r"], "g": c.payload["g"],
                "acc": c.payload.get("acc")}
    return c


def read_tau(ckpt_dir: str, name: str) -> dict:
    """A time-series record from the fine-decimated autocorrelation walk.

    The tuple is POSITIONAL, and its layout is the notebook's `fine_series` return:

        (E, V, T, Jsum_or_None, sigma, acc, n_snapshots)

    **V comes before T.**  That is not the order the names suggest, and it is not a
    detail: positions 1 and 2 are two float series of the same length that both look
    like energy records, so swapping them produces a full set of plausible numbers and
    no error anywhere.  This reader had exactly that swap until the identity below was
    checked against the tag.

    The layout is therefore VERIFIED rather than declared.  `fine_series` builds its
    series from `wf.local_energy`, which returns E = T + kappa*V, and the tag records
    the coupling, so the two together pin the labelling: at kappa = 40 on the stored
    `tau_crystal` the residual of E - T - kappa*V is 2.3e-13, against 3.2e+01 for the
    swapped reading.  A record that fails the check is refused, because a series read
    through the wrong name would silently change a published error bar.
    """
    c = read_checkpoint(ckpt_dir, name)
    t = c.payload
    if not isinstance(t, tuple):
        raise CheckpointError(f"{name}: expected a tuple, got {type(t).__name__}")
    out = {"tag": c.tag, "path": c.path, "n_series": len(t)}
    # E, V, T -- see the docstring; the Jastrow sum is absent when the walk had no
    # Jastrow, which is why the fourth slot is allowed to be None.
    names = ("E", "V", "T", "Jsum")
    for i, nm in enumerate(names):
        if i < len(t) and hasattr(t[i], "__len__"):
            out[nm] = np.asarray(t[i], float)
    out["scalars"] = [float(x) for x in t if np.isscalar(x) or np.ndim(x) == 0]
    out["arrays"] = [np.asarray(x) for x in t if hasattr(x, "__len__")]

    # This family's tag is prose -- "fine-1 sweeps=1000 kappa=40 sigma=0.35 ..." -- not
    # the `|`-separated form `tag_field` handles, so the coupling is read out of the
    # key=value text directly rather than through that helper.
    kappa = None
    for tok in c.tag.split():
        if tok.startswith("kappa="):
            kappa = tok[len("kappa="):]
    if kappa is not None and {"E", "T", "V"} <= set(out):
        resid = float(np.max(np.abs(out["E"] - out["T"] - float(kappa) * out["V"])))
        if resid > 1e-6 * max(1.0, float(np.max(np.abs(out["E"])))):
            raise CheckpointError(
                f"{name}: E != T + kappa*V (kappa={kappa} from the tag, worst residual "
                f"{resid:.3e}).  The tuple's positional layout is not the one this "
                f"reader assumes, and the series would be used under the wrong name.")
        out["kappa"] = float(kappa)
        out["identity_residual"] = resid
    return out


# --------------------------------------------------------------------------
# naming conventions of the legacy store
# --------------------------------------------------------------------------
def crystal_scan_names(ckpt_dir: str, kappas, l0_grid) -> List[tuple]:
    """(kappa, L0, name) for every crystalJ checkpoint present, sorted.

    Enumerated from the DIRECTORY rather than assumed from the grids: a grid cell whose
    checkpoint is missing must show up as a gap the caller can see, not as a KeyError
    three layers down or, worse, as a silently shorter scan.
    """
    out = []
    for k in kappas:
        for L0 in l0_grid:
            nm = f"crystalJ_k{k:g}_L{L0:g}"
            if os.path.exists(os.path.join(ckpt_dir, nm + ".pkl")):
                out.append((float(k), float(L0), nm))
    return out


def ladder_names(ckpt_dir: str, kappas, which: str) -> List[tuple]:
    """(kappa, name) for the ladder_liquid / ladder_crystal family present on disk."""
    if which not in ("liquid", "crystal"):
        raise ValueError("which must be 'liquid' or 'crystal'")
    out = []
    for k in kappas:
        nm = f"ladder_{which}_k{k:g}"
        if os.path.exists(os.path.join(ckpt_dir, nm + ".pkl")):
            out.append((float(k), nm))
    return out
