"""Stage 2E-2: is the clean package's crystal starting point the LEGACY one?

The frozen ``ll_crystal_opt`` cache tags carry ``i{_init_digest(v0, c0)}`` -- a
12-significant-digit digest of the SR starting point.  That makes the start
*verifiable* rather than merely plausible: if the clean package's ``(c0, v0)``
reproduces the digest in the frozen tag, the start is the same one the legacy job
optimised from, element for element.

``_init_digest`` is reproduced here VERBATIM from notebook cell 80 (including the
interleaved ``.view(float)`` on the complex array, which is easy to "tidy" into
something that computes a different hash).  Only the digest function is copied --
nothing is imported from the notebook, and the frozen tree is only READ.

    python scripts/check_init_digest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
PROJ = os.path.dirname(CLEAN)
INIT = os.path.join(CLEAN, "results", "bench_rs75", "initial_conditions.json")
CKPT = os.path.join(PROJ, "_diag", "ckpt_rebuild")

if os.path.join(CLEAN, "src") not in sys.path:
    sys.path.insert(0, os.path.join(CLEAN, "src"))


def _init_digest(v0, c0, nd=12):
    """Verbatim from the frozen notebook, cell 80.  See that cell for the design
    argument; it is not restated here.  Do not reformat: the memory layout of the
    complex array is part of the hash."""
    import hashlib as _h
    _v0, _c0 = np.asarray(v0, dtype=complex), np.asarray(c0, dtype=float)
    _w = np.concatenate([_v0.ravel().view(float), _c0.ravel(),
                         np.array(_v0.shape + _c0.shape, dtype=float)])
    _mag = np.maximum(np.abs(_w), 1e-300)
    _exp = np.clip((nd - 1) - np.floor(np.log10(_mag)), -300.0, 290.0)
    _r = np.rint(_w * np.power(10.0, _exp)).astype(np.int64)
    return _h.blake2b(np.ascontiguousarray(_r).tobytes(), digest_size=4).hexdigest()


def tag_digest(name):
    """The ``i...`` field of a frozen cache tag, or None if the entry is absent.

    The digest is the LAST field.  It is taken by position rather than by matching a
    leading ``i``, because a digest may itself begin with any hex digit -- the
    surviving ``p1conv0`` tag's is ``cadd5d6a``, which a ``startswith("i")`` scan
    combined with any "ic" exclusion would drop.
    """
    meta = os.path.join(CKPT, name + ".meta")
    if not os.path.exists(meta):
        return None
    field = open(meta, encoding="utf-8").read().strip().rsplit("|", 1)[-1]
    if not field.startswith("i"):
        raise ValueError(f"{name}: last tag field {field!r} is not the digest")
    return field[1:]


def main():
    raw = json.load(open(INIT, encoding="utf-8"))
    print(f"init   : {INIT}")
    print(f"frozen : {CKPT}\n")
    print(f"{'seed':>4} {'L0':>5} {'clean digest':>13} {'frozen digest':>14}  verdict")
    print("-" * 62)

    n_match = n_absent = n_diff = 0
    for si in range(5):
        d = raw["crystal"][f"s{si}"]
        c0 = np.asarray(d["c0"], float)
        v0 = (np.asarray(d["v0_real"], float)
              + 1j * np.asarray(d["v0_imag"], float))
        mine = _init_digest(v0, c0)
        ref = tag_digest(f"llcryst_nb2_k53.033_p1conv{si}")
        if ref is None:
            verdict = "no frozen entry (p1conv%d absent)" % si
            n_absent += 1
        elif mine == ref:
            verdict = "MATCH -- same starting point"
            n_match += 1
        else:
            verdict = "DIFFER"
            n_diff += 1
        print(f"{si:>4} {d['L0']:>5g} {mine:>13} {ref or '-':>14}  {verdict}")

    print()
    print(f"match {n_match}   differ {n_diff}   no frozen entry {n_absent}")
    if n_diff:
        print("\nA DIFFERING digest means the clean start is NOT the legacy start for "
              "that seed.  That is exactly what Stage 2E-2 must know before any "
              "energy is compared.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
