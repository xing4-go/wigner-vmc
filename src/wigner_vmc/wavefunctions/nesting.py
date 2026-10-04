"""Nesting of the Landau-level-rotation family in ``nmax``.

Physical role:
    The ``nmax`` truncations are not three unrelated ansatze.  The rotation with
    ``v`` of length ``m`` and the rotation with ``(v, 0)`` of length ``m+1``
    describe the SAME wavefunction: the extra level is rotated into doing
    nothing.  That containment is what makes the ``nmax`` ladder one-sided -- an
    energy that rises when the basis grows is the optimiser failing to use the
    larger space, and there is no initialisation difference to blame it on.

Main mathematical object:
    ``pad_v(v, n_bands)`` embeds ``(nk, m)`` into ``(nk, n_bands - 1)`` by zero
    padding on the right.  Because ``c_row`` is
    ``C = (cos|v|, i (sin|v|/|v|) v)`` and ``|(v, 0)| = |v|``, the padded
    coefficients are ``C_padded = (C, 0)`` -- column for column, exactly.

Units/convention:
    ``v`` is complex, shape ``(nk, n_bands - 1)``; ``n_bands`` is the physics
    core's ``nmax + 1``.  Padding is on the RIGHT, i.e. the new level is the
    highest one; padding on the left would permute the bands and is a different
    (wrong) statement.

Legacy source:
    ``reproduction/make_notebook_llrot.py:1748-1750``, the continuation warm
    start::

        _v0_n = np.zeros((_llb_n.nk, _nb - 1), dtype=complex)
        _v0_n[:, :_v_warm.shape[1]] = _v_warm

    and the identity it relies on, stated at line 7121 of the same file.  There
    is no equivalent in ``qhvmc_engine_llrot.py``; the embedding lives in the
    generator, which is why it is written down here rather than migrated.

Stage 2B / B3.  The identity ``C_padded == (C, 0)`` is verified to machine
precision in `tests/test_ll_rotation_against_legacy.py` -- measured, because the
whole one-sidedness argument for the ladder rests on it.
"""
from __future__ import annotations

import numpy as np

from .ll_rotation import c_row

__all__ = ["pad_v", "nested_coefficients", "coefficient_identity_deviation",
           "nesting_residual"]


def pad_v(v, n_bands):
    """Zero-pad ``v`` to ``(nk, n_bands - 1)``, new levels LAST.

    ``v`` is complex with shape ``(nk, m)``; ``m <= n_bands - 1`` is required.
    A no-op when ``m`` already equals ``n_bands - 1``.
    """
    v = np.atleast_2d(np.asarray(v, dtype=complex))
    if v.ndim != 2:
        raise ValueError(f"v must be (nk, m), got {v.shape}")
    target = int(n_bands) - 1
    if target < 1:
        raise ValueError(f"n_bands must be >= 2 to nest, got {n_bands}")
    if v.shape[1] > target:
        raise ValueError(f"cannot nest {v.shape[1]} parameters into "
                         f"n_bands={n_bands} ({target} available)")
    out = np.zeros((v.shape[0], target), dtype=complex)
    out[:, :v.shape[1]] = v
    return out


def nested_coefficients(v, n_bands):
    """``C`` of the nested parameter vector, shape ``(nk, n_bands)``."""
    return c_row(pad_v(v, n_bands))


def coefficient_identity_deviation(v, n_bands):
    """``max |C_nested[:, :m+1] - C_source|`` and ``max |C_nested[:, m+1:]|``.

    Returns ``(head_dev, tail_dev)``.  Both are zero for an exact nesting: the
    head because the padded vector has the same norm and the same leading
    entries, so ``c_row`` computes the identical floating-point expression; the
    tail because ``sin(t)/t`` times an exact zero is an exact zero.
    """
    v = np.atleast_2d(np.asarray(v, dtype=complex))
    C_src = c_row(v)
    C_nest = nested_coefficients(v, n_bands)
    m = C_src.shape[1]
    head = float(np.abs(C_nest[:, :m] - C_src).max())
    tail = float(np.abs(C_nest[:, m:]).max()) if C_nest.shape[1] > m else 0.0
    return head, tail


def nesting_residual(v_parent, v_child):
    """``(head, tail)`` deviation of an ACTUAL nested vector against its parent.

    This is the guard, and it is NOT the same question as
    ``coefficient_identity_deviation``.  That function pads its argument itself,
    so it compares ``c_row(pad_v(v))`` against ``c_row(v)`` and returns exactly
    ``(0.0, 0.0)`` for *every* input -- it verifies a property of ``pad_v``, not
    the vector the caller is about to optimise.  Used as a guard it can never
    fire, which is worse than no guard, because it reads like one.

    Here the two arguments are independent: ``v_child`` is whatever the caller
    actually built.  A child obtained from ``pad_v(v_parent, n_bands)`` gives
    ``head == tail == 0.0`` exactly (the identity in this module's docstring); a
    child that was re-initialised, truncated, permuted or otherwise constructed
    gives a non-zero head, and the caller must abort rather than optimise from it.

    ``head`` is the deviation on the parent's own ``m+1`` columns, ``tail`` the
    residual weight in the new columns beyond them.
    """
    C_par = c_row(np.atleast_2d(np.asarray(v_parent, dtype=complex)))
    C_chi = c_row(np.atleast_2d(np.asarray(v_child, dtype=complex)))
    m = C_par.shape[1]
    if C_chi.shape[1] < m:
        raise ValueError(f"child has {C_chi.shape[1]} coefficient columns, "
                         f"fewer than the parent's {m}: not a nesting")
    head = float(np.abs(C_chi[:, :m] - C_par).max())
    tail = float(np.abs(C_chi[:, m:]).max()) if C_chi.shape[1] > m else 0.0
    return head, tail
