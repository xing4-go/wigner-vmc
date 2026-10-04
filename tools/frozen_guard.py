"""Read-only reconnaissance against the frozen legacy tree.

Why this module exists
----------------------
On 2026-10-02 a Stage 2E reconnaissance probe exec'd the frozen LL-rotation
notebook's cells with the notebook's own ``cached()`` store still pointing at
``_diag/ckpt_rebuild/``.  ``cached()`` writes on a cache MISS, ``jastrow_opt`` is
wrapped in it, and the store is keyed by FILENAME -- ``jastrow_crystal_k53.033``
carries no ``L0`` -- so five seeds wrote one file five times and two frozen
assets were overwritten.  The full account is in
``wigner_vmc_clean/incident_20261002/`` and ``OVERNIGHT_STOP_REPORT.md``.

The rule this module enforces
-----------------------------
**The frozen tree is read-only to every process in this project.**  A step that
needs the notebook namespace is handed a private COPY to point ``CKPT`` at, and
the frozen tree is fingerprinted before and after, so "nothing was written" is a
measurement rather than an intention:

    from frozen_guard import readonly_frozen, FROZEN_CKPT

    with readonly_frozen(FROZEN_CKPT) as ckpt:
        ...                       # exec the notebook with CKPT = ckpt
    # FrozenTreeChanged is raised here if anything under FROZEN_CKPT moved.

There are two independent defences on purpose, because they fail differently:

* ``assert_not_frozen`` is a *precondition*.  It refuses a write target inside
  the frozen tree before any work happens, so a misconfigured probe fails in
  milliseconds instead of after a 50-minute optimisation.
* ``readonly_frozen`` is a *post-condition*.  It catches a write that reached the
  frozen tree by some path the precondition did not cover -- a hard-coded
  absolute path inside the notebook, say, which no redirection of ``CKPT`` would
  have stopped.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
PROJ = os.path.dirname(CLEAN)

#: The frozen legacy checkpoint store -- the one the 2026-10-02 incident hit.
FROZEN_CKPT = os.path.join(PROJ, "_diag", "ckpt_rebuild")
#: The frozen manifest, whose ``assets`` list is the authority on what is frozen.
MANIFEST = os.path.join(PROJ, "legacy_frozen", "LEGACY_MANIFEST.json")


class FrozenWriteRefused(RuntimeError):
    """A path inside the frozen tree was offered as a write target."""


class FrozenTreeChanged(RuntimeError):
    """The frozen tree was not byte-identical across a reconnaissance block."""


def _norm(path):
    # normcase so Windows' case-insensitive comparison cannot be used to slip a
    # path past the containment test by changing its case.
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def is_inside(path, root):
    """True if ``path`` is ``root`` or lives under it."""
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r + os.sep)


def assert_not_frozen(path, frozen_root=FROZEN_CKPT):
    """Raise ``FrozenWriteRefused`` unless ``path`` is clear of the frozen tree.

    Call this on every directory a reconnaissance step is about to write into --
    in particular on the ``CKPT`` a notebook namespace is given.
    """
    if is_inside(path, frozen_root):
        raise FrozenWriteRefused(
            f"{os.path.abspath(path)!r} is inside the frozen tree "
            f"{os.path.abspath(frozen_root)!r}.\n"
            f"Reconnaissance must write to a working COPY -- see "
            f"working_copy() / readonly_frozen()."
        )
    return path


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(root):
    """``{relative path: (size, mtime_ns, sha256)}`` for every file under ``root``.

    mtime_ns is carried alongside the hash because a rewrite with identical bytes
    is still a write, and knowing about it is cheaper than arguing about whether
    it matters.
    """
    root = os.path.abspath(root)
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            out[os.path.relpath(full, root).replace("\\", "/")] = (
                st.st_size, st.st_mtime_ns, _sha256(full))
    return out


def _comparable(record):
    """A fingerprint record in a comparison-safe form.

    ``fingerprint`` returns tuples.  A fingerprint kept ACROSS runs -- which is
    how the guard is checked at a phase boundary -- has been through JSON, and
    JSON has no tuples, so it comes back with lists.  A tuple never equals a
    list, so comparing the raw records would report *every* path in the tree as
    changed.  That is the worst failure mode available to a guard: a false
    positive that buries a real drift in noise.  Found by hand while checking the
    frozen tree at the B2/B6 boundary, where the call site had to convert with
    ``tuple()`` itself.
    """
    return tuple(record) if isinstance(record, (list, tuple)) else record


def diff_fingerprints(before, after):
    """Sorted paths that were added, removed or changed between two fingerprints.

    Both arguments may be either live (tuples) or JSON round-tripped (lists).
    """
    return [k for k in sorted(set(before) | set(after))
            if _comparable(before.get(k)) != _comparable(after.get(k))]


def working_copy(frozen_root=FROZEN_CKPT, dest=None):
    """A private writable copy of ``frozen_root``, never placed inside it.

    Copying the whole store (not just the entries a run expects to need) is
    deliberate: a cache HIT must still return the frozen value, so the copy has
    to be complete enough for the notebook's own lookups to succeed.
    """
    frozen_root = os.path.abspath(frozen_root)
    if dest is None:
        dest = tempfile.mkdtemp(prefix="ckpt_working_")
    else:
        dest = os.path.abspath(dest)
        assert_not_frozen(dest, frozen_root)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
    shutil.copytree(frozen_root, dest, dirs_exist_ok=True)
    return dest


@contextmanager
def readonly_frozen(frozen_root=FROZEN_CKPT, dest=None):
    """Yield a working COPY of ``frozen_root``; refuse to exit if the tree moved.

    The body runs with the copy; on exit the frozen tree is re-fingerprinted and
    ``FrozenTreeChanged`` is raised if anything under it changed.  The exception
    is raised from ``finally``, so a body that fails for its own reason still
    reports a frozen-tree violation -- the violation is the more important fact.
    """
    frozen_root = os.path.abspath(frozen_root)
    before = fingerprint(frozen_root)
    work = working_copy(frozen_root, dest=dest)
    assert_not_frozen(work, frozen_root)
    try:
        yield work
    finally:
        after = fingerprint(frozen_root)
        changed = diff_fingerprints(before, after)
        if changed:
            shown = "\n  ".join(changed[:20])
            more = f"\n  ... and {len(changed) - 20} more" if len(changed) > 20 else ""
            raise FrozenTreeChanged(
                f"{len(changed)} path(s) under the frozen tree changed during a "
                f"reconnaissance block:\n  {shown}{more}\n"
                f"Nothing was written by this module -- a write reached the frozen "
                f"tree by a path assert_not_frozen() did not cover."
            )
