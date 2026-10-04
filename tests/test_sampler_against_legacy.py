"""B4: `vmc/sampler.py` against the frozen `sample`, and the RNG contract.

The sampler is the one place where a "harmless" refactor changes the ANSWER rather
than the timing: draw the proposal before the acceptance variate instead of after,
or adapt ``sigma`` from the cumulative rate instead of the last-1000 window, and
every energy the pipeline has ever produced shifts.  So the regression is not a
tolerance on an energy -- it is the whole chain.

Two independent statements are made:

**The chain is bit-identical.**  Legacy and clean walk the same ansatz from two
generators seeded identically.  If a single accept/reject decision differs the
configurations diverge, and every later snapshot comparison fails; the snapshots
are therefore required to be bit-equal, not close.

**The RNG stream is consumed identically.**  The generators' own states are
compared after the walk, which pins the NUMBER and ORDER of draws.  That is a
stronger statement than the configurations agreeing: a walk that drew an extra
number and discarded it would still produce the same snapshots.

The walks here are short -- a few hundred attempts from a fixed configuration.
Nothing is measured and no energy is estimated; this is a regression on the
Markov kernel, not a VMC run.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(CLEAN)
SRC = os.path.join(CLEAN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import qhvmc_engine as legacy                                    # noqa: E402
import qhvmc_engine_llrot as legacy_ll                           # noqa: E402
from wigner_vmc.physics import geometry as g                     # noqa: E402
from wigner_vmc.physics import landau_levels as ll               # noqa: E402
from wigner_vmc.vmc import sampler as sm                         # noqa: E402
from wigner_vmc.wavefunctions import gaussian as gauss           # noqa: E402
from wigner_vmc.wavefunctions import jastrow as jw               # noqa: E402
from wigner_vmc.wavefunctions import ll_rotation as lr           # noqa: E402
from wigner_vmc.wavefunctions import slater as sl                # noqa: E402

PRIM_AREA = 2.0 * np.pi
NE = 36
LUMAX_G = 30.0
LUMAX_LL = 8.5 * np.sqrt(4 * np.pi / np.sqrt(3))
KAPPA = 32.0
L0 = 0.6
C3 = np.array([0.2, -0.35, 0.15])


def geom():
    A1, A2 = g.triangular_cell(PRIM_AREA)
    return g.Geometry.from_cell(A1, A2, 6, 6, 4.0)


def start_config(Ge, jitter=0.12, seed=23):
    """A fixed starting configuration, folded into the cell.

    The fold matters for the containment test below: the walk only ever puts
    wrapped positions into R, so an unwrapped starting point would be the one
    way to see a fractional coordinate outside [-1/2, 1/2].
    """
    sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
    raw = np.asarray(sites) + jitter * np.random.default_rng(seed).standard_normal((NE, 2))
    return g.wrap_to_supercell(raw, Ge.sc_to_cart, Ge.cart_to_sc)


def rng_signature(rng):
    """A comparable fingerprint of a generator's internal state."""
    state = rng.bit_generator.state
    inner = state.get("state", state)
    return (state["bit_generator"], int(inner["state"]), int(inner["inc"]))


# ---------------------------------------------------------------- the two ansatze


def gaussian_pair(Ge):
    """(clean wf, legacy wf) for the Wigner-crystal Gaussian ansatz."""
    sites = g.wigner_crystal_sites(6, 6, Ge.A1, Ge.A2)
    li, lc = g.circular_lattice(LUMAX_G, Ge.L1, Ge.L2)
    new_b = gauss.GaussianBasis(sites, li, lc, Ge.L1, Ge.L2, l=1.0)
    old_b = legacy.GaussianBasis(sites, li, lc, Ge.L1, Ge.L2, l=1.0)
    gam = float(np.linalg.norm(Ge.L1) / np.sqrt(2) / np.pi * KAPPA / 3)
    Jn = jw.SinSplineJastrow(C3, Ge.G1, Ge.G2, gam)
    Jo = legacy.SinSplineJastrow(C3, Ge.G1, Ge.G2, gam)
    sc = np.column_stack([Ge.L1, Ge.L2])
    wn = sl.Wavefunction(lambda r: new_b.orbitals(r, L0),
                         lambda r: new_b.pi_orbitals(r, L0),
                         lambda r: new_b.pi_square_orbitals(r, L0),
                         Jn, NE, sc, kappa=KAPPA)
    wo = legacy.Wavefunction(lambda r: old_b.orbitals(r, L0),
                             lambda r: old_b.pi_orbitals(r, L0),
                             lambda r: old_b.pi_square_orbitals(r, L0),
                             Jo, NE, sc, kappa=KAPPA)
    return wn, wo


def ll_pair(Ge):
    """(clean wf, legacy wf) for the Landau-level-rotation ansatz."""
    ai, ac = g.circular_lattice(LUMAX_LL, Ge.A1, Ge.A2)
    C = np.column_stack([Ge.A1, Ge.A2])
    Ci = np.linalg.inv(C)
    nk = len(Ge.mesh)
    rng = np.random.default_rng(7)
    v = rng.normal(size=(nk, 1)) + 1j * rng.normal(size=(nk, 1))
    new_orb = lr.LLRotatedOrbitals(ll.LandauLevelBasis(Ge.mesh, 1, ai, ac, C, Ci), 2, v)
    old_orb = legacy_ll.LLRotatedOrbitals(
        legacy.LandauLevelBasis(Ge.mesh, 2, ai, ac, C, Ci), 2, v)
    gam = float(np.linalg.norm(Ge.L1) / np.sqrt(2) / np.pi * KAPPA / 3)
    Jn = jw.SinSplineJastrow(C3, Ge.G1, Ge.G2, gam)
    Jo = legacy.SinSplineJastrow(C3, Ge.G1, Ge.G2, gam)
    sc = np.column_stack([Ge.L1, Ge.L2])
    return (lr.LLRotationWavefunction(new_orb, Jn, NE, sc, kappa=KAPPA),
            legacy_ll.LLRotationWavefunction(old_orb, Jo, NE, sc, kappa=KAPPA))


class TestTheChainIsBitIdentical(unittest.TestCase):
    """Two walkers, two identically seeded generators, one answer."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.R0 = start_config(cls.Ge)

    def _compare(self, pair, nsweep, sigma=0.5, seed=7, reset_period=None):
        wn, wo = pair
        gn = np.random.default_rng(seed)
        go = np.random.default_rng(seed)
        kwargs = dict(sigma=sigma, snapshot_every=2, reset_period=reset_period)
        sn, sign, accn = sm.sample(wn, self.R0, nsweep, rng=gn, **kwargs)
        so, sigo, acco = legacy.sample(wo, self.R0, nsweep, rng=go, **kwargs)
        self.assertEqual(len(sn), len(so))
        self.assertGreater(len(sn), 0, "no snapshots -- the comparison is vacuous")
        # a walk that accepted nothing (or everything) would agree with any
        # other such walk for reasons that have nothing to do with the port
        self.assertTrue(0.05 < accn < 0.98, f"acceptance rate {accn} is degenerate")
        self.assertFalse(any(np.array_equal(sn[0], s) for s in sn[1:]),
                         "every snapshot identical -- the walk never moved")
        for k, (a, b) in enumerate(zip(sn, so)):
            np.testing.assert_array_equal(a, b, err_msg=f"snapshot {k}")
        self.assertEqual(sign, sigo)
        self.assertEqual(accn, acco)
        self.assertEqual(rng_signature(gn), rng_signature(go))

    def test_gaussian_ansatz(self):
        self._compare(gaussian_pair(self.Ge), nsweep=10, reset_period=97)

    def test_ll_rotation_ansatz(self):
        # the rotated ansatz recomputes a Bloch sum per move, so this walk is
        # shorter; agreement here is what licenses the longer Gaussian walk
        self._compare(ll_pair(self.Ge), nsweep=6, reset_period=97)

    def test_sigma_adaptation_matches(self):
        # >1000 attempts, so the last-1000 window is full and sigma is adapted
        # at least once; without this the file would pass on a sampler with the
        # adaptation deleted
        wn, wo = gaussian_pair(self.Ge)
        gn = np.random.default_rng(3)
        go = np.random.default_rng(3)
        sn, sign, accn = sm.sample(wn, self.R0, 29, sigma=0.5, rng=gn,
                                   snapshot_every=5, equil=28)
        so, sigo, acco = legacy.sample(wo, self.R0, 29, sigma=0.5, rng=go,
                                       snapshot_every=5, equil=28)
        self.assertEqual(sign, sigo)
        self.assertEqual(accn, acco)
        self.assertEqual(rng_signature(gn), rng_signature(go))
        for a, b in zip(sn, so):
            np.testing.assert_array_equal(a, b)
        # sigma must actually have moved, otherwise this test would also pass on
        # a sampler that never adapts (the two would still agree)
        self.assertNotEqual(sign, 0.5, "sigma did not adapt; the window never filled")


class TestTheRngContract(unittest.TestCase):
    """The generator is part of the interface, not an implementation detail."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.R0 = start_config(cls.Ge, seed=31)
        cls.wn, _ = gaussian_pair(cls.Ge)

    def test_the_default_generator_is_default_rng_of_zero(self):
        # NOT the global np.random stream: if it were, the chain would depend on
        # every other draw in the process
        a, siga, acca = sm.sample(self.wn, self.R0, 3, snapshot_every=1)
        b, sigb, accb = sm.sample(self.wn, self.R0, 3, snapshot_every=1)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(siga, sigb)
        self.assertEqual(acca, accb)

        np.random.seed(12345)
        before = np.random.get_state()[1][:4].copy()
        sm.sample(self.wn, self.R0, 2, snapshot_every=1)
        np.testing.assert_array_equal(np.random.get_state()[1][:4], before)

    def test_a_given_generator_is_used_and_advanced(self):
        gen = np.random.default_rng(4)
        start = rng_signature(gen)
        sm.sample(self.wn, self.R0, 2, rng=gen, snapshot_every=1)
        self.assertNotEqual(rng_signature(gen), start)

    def test_the_stream_is_consumed_in_the_documented_order(self):
        # one attempt = one standard_normal(2) THEN one random().  PCG64's state
        # depends on how the draws interleave, not only on how many there are,
        # so replaying the interleaving pins the order as well as the count.
        gen = np.random.default_rng(11)
        sm.sample(self.wn, self.R0, 1, rng=gen, snapshot_every=1, equil=99)
        ref = np.random.default_rng(11)
        for _ in range(NE):
            ref.standard_normal(2)
            ref.random()
        self.assertEqual(rng_signature(gen), rng_signature(ref))


class _StubWavefunction:
    """A wavefunction with a preprogrammed acceptance probability.

    Used to pin the sigma adaptation by construction.  On a real ansatz the
    windowed and cumulative rates track each other to within a percent, so no
    practical run length distinguishes the two rules -- measured, over 3 windows
    and from an unequilibrated start, and the final sigma was bit-identical.  A
    schedule that accepts everything in one window and nothing in the next does
    distinguish them, so the rule is pinned here rather than argued for.
    """

    def __init__(self, Ge, ne, schedule):
        self.ne = int(ne)
        self.sc_to_cart = Ge.sc_to_cart
        self.cart_to_sc = Ge.cart_to_sc
        self._schedule = schedule        # attempt index -> acceptance probability
        self.attempt = 0

    def build(self, R):
        self.attempt = 0
        return {"R": np.array(R, dtype=float, copy=True), "D": None, "D_inv": None,
                "u": None, "gx": None, "gy": None, "lap": None, "U": 0.0}

    def rebuild(self, st):
        pass

    def move_ratio(self, st, i, r_new):
        p = float(self._schedule(self.attempt))
        self.attempt += 1
        return p, (i, r_new)

    def accept_move(self, st, i, r_new, info):
        st["R"][i] = r_new


class TestTheSigmaAdaptationRule(unittest.TestCase):
    """The window is part of the algorithm, so it is tested as one."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.R0 = start_config(cls.Ge, seed=51)

    def test_sigma_adapts_only_at_a_multiple_of_a_thousand_attempts(self):
        # 1000 attempts is 27.8 sweeps of 36, so windows land at 28 and 56
        for sweeps, moves in ((27, 0), (28, 1), (55, 1), (56, 2)):
            wf = _StubWavefunction(self.Ge, NE, lambda n: 1.0)
            _, sig, _ = sm.sample(wf, self.R0, sweeps, sigma=0.5,
                                  rng=np.random.default_rng(1), equil=10 ** 9)
            self.assertEqual(wf.attempt, sweeps * NE)
            self.assertAlmostEqual(sig, 0.5 * 1.05 ** moves, places=15,
                                   msg=f"{sweeps} sweeps ({moves} windows)")

    def test_the_rate_is_the_last_thousand_attempts_not_the_cumulative_one(self):
        # accepts everything for attempts 0..999, nothing afterwards.
        #   windowed: at 1000 rate=1.0 -> x1.05; at 2000 rate=0.0 -> x0.95
        #   cumulative: at 1000 rate=1.0 -> x1.05; at 2000 rate=0.5 -> NO branch
        # The two rules therefore predict different sigmas, and this asserts the
        # one the frozen engine implements.
        wf = _StubWavefunction(self.Ge, NE, lambda n: 1.0 if n < 1000 else 0.0)
        _, sig, acc = sm.sample(wf, self.R0, 56, sigma=0.5,
                                rng=np.random.default_rng(1), equil=10 ** 9)
        self.assertEqual(wf.attempt, 2016)
        self.assertAlmostEqual(sig, 0.5 * 1.05 * 0.95, places=15)
        self.assertAlmostEqual(acc, 1000 / 2016, places=12)
        # and the cumulative rule would have left it here instead
        self.assertNotAlmostEqual(sig, 0.5 * 1.05, places=6)

    def test_target_acc_zero_or_one_edge(self):
        # rate > target -> grow, rate < target -> shrink; equal -> neither.  A
        # chain that accepts everything with target 1.0 must NOT grow silently.
        wf = _StubWavefunction(self.Ge, NE, lambda n: 1.0)
        _, sig, _ = sm.sample(wf, self.R0, 28, sigma=0.5, target_acc=1.0,
                              rng=np.random.default_rng(1), equil=10 ** 9)
        self.assertAlmostEqual(sig, 0.5, places=15)


class TestTheSamplersOwnBehaviour(unittest.TestCase):
    """Properties that hold regardless of what the frozen engine does."""

    @classmethod
    def setUpClass(cls):
        cls.Ge = geom()
        cls.R0 = start_config(cls.Ge, seed=41)
        cls.wn, _ = gaussian_pair(cls.Ge)

    def test_positions_stay_in_the_supercell(self):
        snaps, _, _ = sm.sample(self.wn, self.R0, 8, rng=np.random.default_rng(2),
                                snapshot_every=1, equil=0)
        for R in snaps:
            frac = R @ self.wn.cart_to_sc.T
            self.assertLess(float(np.abs(frac).max()), 0.5 + 1e-12)

    def test_snapshots_are_copies_not_views(self):
        # a view would be overwritten by the next sweep, so a "sequence of
        # configurations" would silently become one configuration repeated
        snaps, _, _ = sm.sample(self.wn, self.R0, 8, rng=np.random.default_rng(2),
                                snapshot_every=1, equil=0)
        self.assertGreater(len(snaps), 2)
        self.assertFalse(any(np.array_equal(snaps[0], s) for s in snaps[1:]),
                         "snapshots are identical -- views, or the walk is frozen")
        before = snaps[1].copy()
        snaps[0][:] = -99.0
        np.testing.assert_array_equal(snaps[1], before)

    def test_equil_and_snapshot_every_select_the_right_sweeps(self):
        for nsweep, equil, every in ((12, None, 5), (12, 4, 3), (7, 0, 1), (10, 9, 2)):
            snaps, _, _ = sm.sample(self.wn, self.R0, nsweep,
                                    rng=np.random.default_rng(1), equil=equil,
                                    snapshot_every=every)
            e = nsweep // 2 if equil is None else equil
            expected = len([s for s in range(nsweep) if s >= e and (s - e) % every == 0])
            self.assertEqual(len(snaps), expected,
                             f"nsweep={nsweep} equil={equil} every={every}")

    def test_the_callback_runs_once_per_sweep(self):
        seen = []
        sm.sample(self.wn, self.R0, 5, rng=np.random.default_rng(1),
                  callback=lambda sweep, st: seen.append(sweep))
        self.assertEqual(seen, [0, 1, 2, 3, 4])

    def test_the_acceptance_rate_is_a_real_rate(self):
        n = 8
        snaps, _, acc = sm.sample(self.wn, self.R0, n,
                                  rng=np.random.default_rng(1), equil=n)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertEqual(len(snaps), 0)     # equil = nsweep takes no snapshots

    def test_sigma_moves_the_right_way(self):
        # a step far too small is accepted almost always, so the adaptation must
        # GROW it; a step far too large must be shrunk.  Both need >1000 attempts
        # because the window is 1000 long.
        _, up, _ = sm.sample(self.wn, self.R0, 29, sigma=0.02,
                             rng=np.random.default_rng(1), equil=99)
        self.assertAlmostEqual(up, 0.02 * 1.05, places=15)
        _, down, _ = sm.sample(self.wn, self.R0, 29, sigma=20.0,
                               rng=np.random.default_rng(1), equil=99)
        self.assertAlmostEqual(down, 20.0 * 0.95, places=12)

    def test_reset_period_keeps_the_inverse_honest(self):
        # D_inv is carried by rank-1 updates; reset_period exists so the
        # accumulated error never reaches the Metropolis ratio.  Measured at
        # every sweep end through the callback.
        drift = []

        def watch(sweep, st):
            drift.append(float(np.abs(st["D_inv"] @ st["D"] - np.eye(NE)).max()))

        sm.sample(self.wn, self.R0, 20, rng=np.random.default_rng(5),
                  equil=99, reset_period=50, callback=watch)
        self.assertEqual(len(drift), 20)
        self.assertLess(max(drift), 1e-8, f"worst |D_inv D - I| = {max(drift):.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
