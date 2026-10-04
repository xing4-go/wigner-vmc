"""The reconstruction recipes the Stage 2C golden fixtures are built from.

Both the builder (``_build.py``) and the test that reads the fixtures
(``tests/test_behavior_fixtures.py``) import this module, so the two cannot drift:
whatever the builder froze is what the test rebuilds, with the same geometry, the
same ansatz and the same observable list.

The observables are the ones the package actually defines
--------------------------------------------------------
There is no ``log_psi`` method anywhere in the clean package or in the frozen
engine -- Psi enters the pipeline only as a |Psi|^2 ratio (in the sampler) and as
the local energy.  So a fixture does not lock an invented "log Psi"; it locks the
two first-class ingredients the state carries:

    logdet(D)   the Slater determinant, assembled from ``orbitals`` over the 36
                electrons -- a whole-ansatz lock on the wavefunction layer
    U           the Jastrow, as ``u.sum()/2``

together with ``T``, ``V`` and ``E_local``.  The |Psi|^2 ratio chain itself is
locked separately, in the layer-D sampler tests.

The recipes are copied from the accepted B1-B7 regressions
(``test_local_energy_against_legacy.py``'s ``gaussian_pair``/``ll_pair``) rather
than invented here: a fixture built from a different recipe would lock a
configuration the old/new tests never compared.
"""
import json
import os

import numpy as np

from wigner_vmc.physics import coulomb as cb
from wigner_vmc.physics import geometry as g
from wigner_vmc.physics import landau_levels as ll
from wigner_vmc.wavefunctions import gaussian as gb
from wigner_vmc.wavefunctions import jastrow as jw
from wigner_vmc.wavefunctions import ll_rotation as lr
from wigner_vmc.wavefunctions import slater as sl

HERE = os.path.dirname(os.path.abspath(__file__))

PRIM_AREA = 2.0 * np.pi
NE = 36
LUMAX = 30.0
LUMAX_LL = 8.5 * np.sqrt(4 * np.pi / np.sqrt(3))

#: A coupling on the production grid, entered as r_s so that the fixture also
#: carries the locked relation kappa = r_s / sqrt(2) (layer A) into its numbers.
RS = 45.0
KAPPA = RS / np.sqrt(2.0)

L0 = 0.6
C5 = (0.30, -0.55, 0.42, -0.18, 0.07)

#: The seeds that produced the frozen configurations and v's.  Recorded here so
#: the inputs can be regenerated and checked, but the fixtures themselves store
#: the ARRAYS -- a rebuild needs no RNG at all.
SEED_GAUSSIAN = 20261001
SEED_LL = 7


def geometry():
    """The production torus: 6x6 mesh, N = 36, primitive cell area 2*pi."""
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, 4.0)


def jastrow(Ge):
    return jw.SinSplineJastrow(list(C5), Ge.G1, Ge.G2,
                               jw.cusp_gamma(KAPPA, Ge.L1))


def coulomb(Ge):
    return cb.CoulombEwald(NE, Ge.L1, Ge.L2, Ge.G1, Ge.G2)


def gaussian_wf(Ge, l0=L0):
    sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
    li, lc = g.circular_lattice(LUMAX, Ge.L1, Ge.L2)
    b = gb.GaussianBasis(sites, li, lc, Ge.L1, Ge.L2, l=1.0)
    sc = np.column_stack([Ge.L1, Ge.L2])
    return sl.Wavefunction(lambda r: b.orbitals(r, l0),
                           lambda r: b.pi_orbitals(r, l0),
                           lambda r: b.pi_square_orbitals(r, l0),
                           jastrow(Ge), NE, sc, kappa=KAPPA, ham=coulomb(Ge))


def ll_v(nk, m, seed):
    """A seeded (nk, m) complex band-coefficient array.

    The same generator the fixtures were built from, kept here so the builder and
    the nesting ladder draw their coefficients the same way.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(nk, m)) + 1j * rng.normal(size=(nk, m))


def ll_orb(Ge, v):
    """The rotated orbitals alone.  ``v`` is (nk, n_bands - 1) complex; the band
    count is read off it, which is also the B3 ``nmax = n_bands - 1`` rule."""
    v = np.asarray(v, dtype=complex)
    n_band = v.shape[1] + 1
    ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
    C = np.column_stack([Ge.A1, Ge.A2])
    Ci = np.linalg.inv(C)
    b = ll.LandauLevelBasis(Ge.mesh, n_band - 1, ai, ac, C, Ci)
    return lr.LLRotatedOrbitals(b, n_band, v)


def ll_wf(Ge, v):
    sc = np.column_stack([Ge.L1, Ge.L2])
    return lr.LLRotationWavefunction(ll_orb(Ge, v), jastrow(Ge), NE, sc,
                                     kappa=KAPPA, ham=coulomb(Ge))


def cell_points(Ge, n, lo, hi, seed):
    """``n`` points in cell coordinates drawn uniformly on [lo, hi) per axis.

    ``lo=-0.05, hi=1.05`` deliberately puts some points OUTSIDE the supercell, so
    the boundary fixture exercises the wrapping / minimum-image path rather than
    only the interior.  The sampler itself always folds onto the torus; this is
    the reconstruction being asked the same question from the other side.
    """
    sc = np.column_stack([Ge.L1, Ge.L2])
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=(n, 2)) @ sc.T


def observe(wf, R):
    """The frozen observable list for one configuration."""
    st = wf.build(R)
    sign, logabs = np.linalg.slogdet(st["D"])
    if sign == 0:
        raise ValueError("singular Slater matrix: pick another configuration")
    logdet = complex(logabs) + np.log(complex(sign))
    T, V, E = wf.local_energy(st)
    return {"logdet_D": [logdet.real, logdet.imag],
            "U": float(st["U"]),
            "T": [complex(T).real, complex(T).imag],
            "V": [complex(V).real, complex(V).imag],
            "E_local": [complex(E).real, complex(E).imag]}


# -- serialisation ----------------------------------------------------------
def _plain(obj):
    """numpy scalars/arrays -> plain lists, so json can write them."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_plain(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    return obj


def path(name):
    return os.path.join(HERE, name + ".json")


def dump(name, payload):
    with open(path(name), "w", encoding="utf-8") as fh:
        json.dump(_plain(payload), fh, indent=1, sort_keys=True)
        fh.write("\n")


def load(name):
    with open(path(name), encoding="utf-8") as fh:
        return json.load(fh)


def as_R(raw):
    return np.array(raw, dtype=float).reshape(NE, 2)


def as_c(raw):
    """A nested list of [re, im] pairs -> complex array of the same shape.

    Used for ``v`` (nk, m) and for every complex observable, so there is one
    decoding path rather than one per quantity.
    """
    a = np.array(raw, dtype=float)
    return a[..., 0] + 1j * a[..., 1]
