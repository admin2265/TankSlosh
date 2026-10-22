import unittest

import numpy as np

from tankslosh import CUP_3MM_DIAPHRAGM
from tankslosh.diaphragm_static import DiaphragmStatics, ShellStaticSolver
from tankslosh.shell import AxisymmetricShell

MU = 1.8e6


class TestShellEnergy(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        ds = DiaphragmStatics(CUP_3MM_DIAPHRAGM, n_segments=40)
        sh = ds.shell
        rng = np.random.default_rng(1)
        r = sh.r0 + 2e-3 * rng.standard_normal(sh.N + 1)
        z = sh.z0 + 2e-3 * rng.standard_normal(sh.N + 1)
        r[0], z[0], r[-1] = sh.r0[0], sh.z0[0], 0.0
        U, gr, gz = sh.energy(r, z, grad=True)
        _, vr, vz = sh.enclosed_volume(r, z, grad=True)
        h = 1e-7
        for i in range(1, sh.N + 1):
            for arr, g, vg in ((r, gr, vr), (z, gz, vz)):
                if arr is r and i == sh.N:
                    continue
                old = arr[i]
                arr[i] = old + h
                Up, Vp = sh.energy(r, z), sh.enclosed_volume(r, z)
                arr[i] = old - h
                Um, Vm = sh.energy(r, z), sh.enclosed_volume(r, z)
                arr[i] = old
                self.assertAlmostEqual((Up - Um) / (2 * h), g[i], delta=1e-6 * np.abs(g).max())
                self.assertAlmostEqual((Vp - Vm) / (2 * h), vg[i], delta=1e-6 * np.abs(vg).max())

    def test_inflated_sphere_pressure(self):
        # uniformly inflated neo-Hookean sphere: dp = (2 mu t / R0)(1/l - 1/l^7)
        R0, t, N = 0.2, 1e-5, 200
        th = np.linspace(0, np.pi / 2, N + 1)
        r0, z0 = R0 * np.cos(th), -R0 * np.sin(th)
        r0[-1] = 0.0
        sh = AxisymmetricShell(r0, z0, MU, t)
        lam, h = 1.3, 1e-6
        dU = sh.energy((lam + h) * r0, (lam + h) * z0) - sh.energy((lam - h) * r0, (lam - h) * z0)
        dV = sh.enclosed_volume((lam + h) * r0, (lam + h) * z0) - \
            sh.enclosed_volume((lam - h) * r0, (lam - h) * z0)
        exact = 2 * MU * t / R0 * (1 / lam - lam ** -7)
        self.assertAlmostEqual((dU / dV) / exact, 1.0, delta=1e-4)


class TestStaticSolver(unittest.TestCase):
    def test_clamped_plate_small_deflection(self):
        # Kirchhoff clamped circular plate: w0 = p a^4 / (64 D), V = pi a^2 w0 / 3
        a, t, N = 0.2, 0.02, 80
        sh = AxisymmetricShell(np.linspace(a, 0, N + 1), np.zeros(N + 1), MU, t)
        solver = ShellStaticSolver(sh, contact=None)
        solver.U_scale = 1e-3
        w0 = 0.05 * t
        s = solver.solve(np.pi * a ** 2 * w0 / 3)
        self.assertTrue(s.converged)
        self.assertAlmostEqual(s.dp / (-64 * sh.D * w0 / a ** 4), 1.0, delta=5e-3)
        self.assertAlmostEqual(-s.z[-1] / w0, 1.0, delta=5e-3)


class TestBaselineDiaphragm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = DiaphragmStatics(CUP_3MM_DIAPHRAGM, n_segments=50)
        cls.sols = {round(s.fill, 3): s for s in cls.ds.sweep([0.1, 0.29, 0.31, 0.5, 0.7])}

    def test_all_converged(self):
        self.assertEqual(sorted(self.sols), [0.1, 0.29, 0.31, 0.5, 0.7])
        for s in self.sols.values():
            self.assertTrue(s.converged, s.message)
            self.assertEqual(s.self_intersections, [])

    def test_stress_free_at_half_fill(self):
        s = self.sols[0.5]
        self.assertLess(s.energy, 1e-9)
        np.testing.assert_allclose(s.r, self.ds.shell.r0, atol=1e-9)

    def test_volume_and_contact_constraints(self):
        for f, s in self.sols.items():
            V_liq = self.ds.V_ref - self.ds.shell.enclosed_volume(s.r, s.z) + \
                self.ds.d.stress_free_fill * self.ds.d.tank.volume
            self.assertAlmostEqual(V_liq / self.ds.d.tank.volume, f, places=7)
            self.assertGreater(self.ds.solver.contact.g(s.r, s.z).min(), -1e-8)

    def test_frictionless_contact_uniform_tension(self):
        # on the straight wall r = R with no friction, meridional tension is constant
        s = self.sols[0.1]
        T1, _ = self.ds.shell.tensions(s.r, s.z)
        c = set(s.contact_nodes.tolist())
        seg = [j for j in range(len(T1)) if j in c and j + 1 in c
               and s.z[j + 1] > self.ds.d.tank.straight_bottom]
        self.assertGreater(len(seg), 5)
        self.assertLess(np.ptp(T1[seg]) / T1[seg].mean(), 1e-5)

    def test_pressure_is_energy_derivative(self):
        # dp = dU/dV_liquid (frictionless contact does no work)
        s1, s2 = self.sols[0.29], self.sols[0.31]
        dUdV = (s2.energy - s1.energy) / (0.02 * self.ds.d.tank.volume)
        self.assertAlmostEqual(dUdV / (0.5 * (s1.dp + s2.dp)), 1.0, delta=2e-3)

    def test_regression_values(self):
        # mesh-converged to < 0.1 % below 50 % fill (see docs/diaphragm_statics.md)
        self.assertAlmostEqual(self.sols[0.1].dp / -52.35e3, 1.0, delta=2e-3)
        self.assertAlmostEqual(self.sols[0.1].max_stretch, 2.18, delta=0.02)
        self.assertGreater(self.sols[0.7].dp, 0.0)
        self.assertLess(self.sols[0.7].dp, 1e3)


if __name__ == "__main__":
    unittest.main()
