"""Lattice geometry on the torus: cells, reciprocal vectors, and periodic images.

Stage 2A.  This module is the single source of truth for the *Euclidean* geometry --
cell vectors, reciprocal vectors, enumeration of lattice vectors, wrapping, minimum
images, and the Wigner-crystal site positions.  It knows nothing about magnetic
fields; the magnetic-cell conventions (the fold into the primitive cell that a Bloch
orbital needs, and the phase bookkeeping that goes with it) live in
`magnetic_cell.py`.

Every function here was checked against the frozen engine `qhvmc_engine.py` on fixed
inputs -- see `tests/test_geometry_against_legacy.py`.  Where this module departs from
the legacy *code shape* it is deliberate and said so in the docstring; where it
departs from the legacy *numbers* it is a bug.

Units: lengths in the magnetic length, l_B = 1.  Reciprocals are therefore 1/l_B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "cross_z_hat", "circular_lattice", "Geometry", "wrap_to_supercell",
    "minimum_image", "wigner_crystal_sites", "triangular_cell",
]


def cross_z_hat(V):
    """``[x, y] -> [y, -x]``, i.e. ``z_hat x V``.

    The sign is load-bearing.  ``Geometry`` builds ``G1`` from ``cross_z_hat(L2)``
    with a ``+2*pi`` and ``G2`` from ``cross_z_hat(L1)`` with a ``-2*pi``; flipping
    either one produces a left-handed reciprocal lattice that still looks like a
    perfectly reasonable hexagon of momenta.
    """
    V = np.asarray(V, dtype=float)
    return np.array([V[1], -V[0]])


def circular_lattice(r_max: float, a1, a2) -> Tuple[np.ndarray, np.ndarray]:
    """Every integer combination ``i*a1 + j*a2`` with norm ``<= r_max``.

    Returns ``(ints, carts)``, index-aligned: ``ints`` is ``(n, 2)`` int and
    ``carts`` is ``(n, 2)`` float, with ``carts[k] == ints[k,0]*a1 + ints[k,1]*a2``.

    Two things about the enumeration are part of the contract, because callers depend
    on the arrays staying aligned and on nothing else:

    * the search window is the *rectangle* ``i, j in [-n1, n1] x [-n2, n2]`` with
      ``n_k = 2*ceil(r_max/|a_k|)``.  That is a rectangle in *integer* space, so the
      *order* of the output depends on the basis even though the *set* does not;
    * the output is **not** sorted by norm.  `Geometry` sorts where it needs to;
      `CoulombEwald` does not.
    """
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    n1 = int(2 * np.ceil(r_max / np.linalg.norm(a1)))
    n2 = int(2 * np.ceil(r_max / np.linalg.norm(a2)))
    ii, jj = np.meshgrid(np.arange(-n1, n1 + 1), np.arange(-n2, n2 + 1), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()
    ints = np.stack([ii, jj], axis=1)
    carts = ii[:, None] * a1[None, :] + jj[:, None] * a2[None, :]
    keep = np.linalg.norm(carts, axis=1) <= r_max
    return ints[keep], carts[keep]


def triangular_cell(area: float) -> Tuple[np.ndarray, np.ndarray]:
    """The 60-degree primitive vectors whose cell has the given area.

    ``A1 = (a, 0)``, ``A2 = (a/2, a*sqrt(3)/2)`` with ``a = sqrt(2*area/sqrt(3))``.

    ``area`` is the area of ONE PRIMITIVE CELL.  For the production system that is
    ``2*pi``, which gives ``a = A_WC = 2.693547374177197`` -- also the
    nearest-neighbour distance of the Wigner crystal, the two being the same number
    because the crystal has one electron per primitive cell.  Passing the supercell
    area (``72*pi``) instead gives a cell six times too wide and a system 36 times too
    large; the mistake is easy to make and produces a perfectly plausible lattice, so
    the argument is named for what it is and the test pins the production value.
    """
    a = float(np.sqrt(2.0 * area / np.sqrt(3.0)))
    return np.array([a, 0.0]), np.array([a / 2.0, a * np.sqrt(3.0) / 2.0])


@dataclass(frozen=True)
class Geometry:
    """A periodic cell and everything derived from it.

    Built by `from_cell`.  Frozen, because a geometry that is edited after the
    reciprocal vectors have been cached is a geometry whose two halves disagree.

    Attributes
    ----------
    A1, A2 : (2,)
        primitive (unit-cell) vectors.
    N1, N2 : int
        how many primitive cells along each direction.
    L1, L2 : (2,)
        supercell vectors, ``L_i = N_i * A_i``.
    area : float
        supercell area.
    G1, G2 : (2,)
        reciprocal vectors of the *supercell* (the small ones).
    g1, g2 : (2,)
        reciprocal vectors of the *primitive* cell (the large ones).
    mesh : (N1*N2, 2)
        the momenta ``i*G1 + j*G2`` for ``i in range(N1), j in range(N2)``, in
        **row order**.  This is a specific object: it is the set of allowed momenta
        of the magnetic Brillouin zone, ordered as the engine orders it.  It is *not*
        the structure-factor grid -- that is `analysis.structure.Torus.allowed_momenta`
        and it has a different size (294) and a different order (rounded lexsort).
        Anything that reads a stored spectrum must align by q **vector**; aligning by
        index between these two is the mistake this note exists to prevent.
    RL : (n, 2)
        primitive lattice vectors within the construction cutoff, sorted by norm.
    sc_to_cart, cart_to_sc : (2, 2)
        2x2 matrices whose *columns* are ``L1, L2`` and their inverse.  Named for what
        they do: ``r_cart = sc_to_cart @ r_frac``.
    """

    A1: np.ndarray
    A2: np.ndarray
    N1: int
    N2: int
    L1: np.ndarray
    L2: np.ndarray
    area: float
    G1: np.ndarray
    G2: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    mesh: np.ndarray
    RL: np.ndarray
    sc_to_cart: np.ndarray
    cart_to_sc: np.ndarray

    # -- derived -------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.N1 * self.N2

    @property
    def cell_area(self) -> float:
        """Area of one primitive cell = ``area / n_cells``."""
        return self.area / self.n_cells

    @property
    def a_wc(self) -> float:
        """Primitive-cell lattice constant (the WC nearest-neighbour distance)."""
        return float(np.linalg.norm(self.A1))

    @property
    def sqrt_n(self) -> float:
        """``sqrt(n_electrons / area)``, the density to the 1/2.

        NOT ``1/a_wc``.  For the production cell this is ``sqrt(2*pi) = 2.506628...``
        against ``a_wc = 2.693547...``; the two are close enough to be swapped by
        accident.  This one scales the dimensionless axes in the structure figures.
        """
        return float(np.sqrt(self.n_cells / self.area))

    @classmethod
    def from_cell(cls, A1, A2, N1: int, N2: int, rl_cut: float,
                  magnetic: bool = True, fermi_surface: bool = False) -> "Geometry":
        """Build from primitive vectors, cell counts and a lattice-vector cutoff.

        ``magnetic``/``fermi_surface`` control the momentum mesh:

        * ``magnetic=True`` (every call in both production generators): ``mesh`` is
          the row-ordered ``i*G1 + j*G2`` grid.
        * ``magnetic=False``: the mesh is either folded into the first magnetic
          Brillouin zone, or -- with ``fermi_surface=True`` -- replaced by the
          ``N1*N2`` shortest vectors of the extended grid.  **No production call
          takes this branch.**  It is carried because the inventory has to be
          complete, and it is exercised only by the regression test.
        """
        A1 = np.asarray(A1, dtype=float)
        A2 = np.asarray(A2, dtype=float)
        L1 = N1 * A1
        L2 = N2 * A2
        area = float(abs(L1[0] * L2[1] - L1[1] * L2[0]))
        G1 = 2 * np.pi * cross_z_hat(L2) / area
        G2 = -2 * np.pi * cross_z_hat(L1) / area
        g1 = 2 * np.pi * cross_z_hat(A2) / (area / (N1 * N2))
        g2 = -2 * np.pi * cross_z_hat(A1) / (area / (N1 * N2))

        mesh = np.array([i * G1 + j * G2 for i in range(N1) for j in range(N2)])

        if not magnetic:
            big_i, big_j = np.meshgrid(np.arange(-N1, N1 + 1), np.arange(-N2, N2 + 1),
                                       indexing="ij")
            big = np.stack([big_i.ravel(), big_j.ravel()], axis=1)
            raw = big[:, 0:1] * G1[None, :] + big[:, 1:2] * G2[None, :]
            if fermi_surface:
                mesh = raw[np.argsort(np.linalg.norm(raw, axis=1))[: N1 * N2]]
            else:
                folded = mesh.copy()
                for i in range(-3, 4):
                    for j in range(-3, 4):
                        cand = mesh + i * g1 + j * g2
                        better = (np.linalg.norm(cand, axis=1)
                                  < np.linalg.norm(folded, axis=1))
                        folded[better] = cand[better]
                mesh = folded

        _, RL = circular_lattice(rl_cut, g1, g2)
        RL = RL[np.argsort(np.linalg.norm(RL, axis=1))]

        sc = np.column_stack([L1, L2])
        return cls(A1=A1, A2=A2, N1=int(N1), N2=int(N2), L1=L1, L2=L2, area=area,
                   G1=G1, G2=G2, g1=g1, g2=g2, mesh=mesh, RL=RL,
                   sc_to_cart=sc, cart_to_sc=np.linalg.inv(sc))


def wrap_to_supercell(r, sc_to_cart, cart_to_sc):
    """Fold a point into the first supercell.  Returns the folded point only.

    This is NOT the minimum-image displacement.  Rounding the fractional coordinates
    picks one fixed representative of each image class -- the one inside the
    parallelogram -- and for an oblique cell that is not always the nearest one.
    See `minimum_image` for the counterexample.

    The Ewald pair sum uses *this* fold, deliberately: the fold is what makes the
    cutoff bound ``|r_ij| <= (|L1|+|L2|)/2`` valid.  Substituting the minimum image
    there is a silent physics change.
    """
    r = np.atleast_2d(np.asarray(r, dtype=float))
    rsc = r @ cart_to_sc.T
    rsc = rsc - np.round(rsc)
    return rsc @ sc_to_cart.T


def minimum_image(d, sc_to_cart, cart_to_sc):
    """Shortest periodic image of a 2D displacement, for any supercell shape.

    ``d`` is ``(..., 2)``; the result is the image ``d + n1*L1 + n2*L2`` of smallest
    Euclidean norm, found by searching the nine shifts around the parallelogram
    representative.

    Why nine shifts and not a rounding
    ----------------------------------
    ``|d|^2 = s1^2|L1|^2 + s2^2|L2|^2 + 2 s1 s2 L1.L2`` has a cross term, so for an
    oblique cell the two fractional coordinates cannot be minimised one at a time.
    Concretely, for the 60-degree cell ``L1 = (1,0)``, ``L2 = (1/2, sqrt3/2)`` the
    fractional displacement ``(0.49, 0.49)`` is already inside the parallelogram at
    ``|d| = 0.849``, but its image ``(-0.51, 0.49)`` one lattice vector away is at
    ``|d| = 0.500``.  Only the second is the distance between those two electrons.

    The window is enough for the cells used here -- a 60-degree rhombus with
    ``|L1| = |L2|``, as reduced as a 2D basis gets -- and was checked against a +/-4
    brute force on a dense scan of the cell.  A basis past roughly aspect ratio 2, or
    at an angle near 0, would want a wider window.
    """
    d = np.asarray(d, dtype=float)
    dsc = d @ cart_to_sc.T
    dsc = dsc - np.round(dsc)
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
    images = (dsc[..., None, :] + shifts) @ sc_to_cart.T          # (..., 9, 2)
    best = np.argmin(np.linalg.norm(images, axis=-1), axis=-1)
    # Add the winning shift back rather than gathering the image: the two are the
    # same vector, and this needs no index gymnastics for any input rank.
    return (dsc + shifts[best]) @ sc_to_cart.T


def wigner_crystal_sites(N1: int, N2: int, A1, A2, aspect_ratio: int = 1):
    """Wigner-crystal site positions, ``(N1*N2, 2)``.

    The sites are ``i*A1/aspect_ratio + j*A2`` for ``i in range(N1*aspect_ratio)``,
    ``j in range(N2)``, each then folded to its minimum-norm image under the
    *supercell* lattice by a brute-force search over ``i, j in [-5, 5]``.

    The fold centres the SET on the origin, so the sites straddle ``r = 0`` and do not
    tile ``[0, L1) x [0, L2)``.  That is the right thing for a crystal -- it has no
    origin -- but it means the site positions cannot be used as plotting coordinates
    against a cell outline drawn from ``0`` to ``L1``.  (They were, once: 20 of 36
    markers landed off the axes.)  Use `analysis.structure.lattice_sites`, which
    returns fractional coordinates in ``[0, 1)``, for that job.

    The brute-force window is reproduced exactly as the engine has it, so that ties
    between equidistant images resolve to the same site.  A tie is broken towards the
    first candidate found in the ``i``-then-``j`` order, and that is part of the
    contract this function is checked against.
    """
    A1 = np.asarray(A1, dtype=float)
    A2 = np.asarray(A2, dtype=float)
    sites = np.array([i * A1 / aspect_ratio + j * A2
                      for i in range(N1 * aspect_ratio)
                      for j in range(N2)])
    L1 = N1 * A1
    L2 = N2 * A2
    out = []
    for s in sites:
        best = s.copy()
        bestd = np.linalg.norm(best)
        for i in range(-5, 6):
            for j in range(-5, 6):
                c = s - i * L1 - j * L2
                d = np.linalg.norm(c)
                if d < bestd:
                    bestd = d
                    best = c
        out.append(best)
    return np.array(out)
