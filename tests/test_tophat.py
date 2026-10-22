"""Tests for the ribbed top-hat baseline: smeared radial ribs, top-hat geometry,
self-contact of folded layers."""
import unittest

import numpy as np
from scipy.linalg import eigh, null_space

import tankslosh as ts
from tankslosh.diaphragm_static import DiaphragmStatics
from tankslosh.harmonic_shell import HarmonicShell
from tankslosh.shell import AxisymmetricShell

MU = 1.8e6


def _cap(n, R=0.3, zc=0.3):
    th = np.linspace(0, np.pi / 2, n + 1)
    r0, z0 = R * np.cos(th), zc + R * np.sin(th)
    r0[-1] = 0.0
    return r0, z0


class TestRibs(unittest.TestCase):
    def test_gradients_with_ribs(self):
        r0, z0 = _cap(40)
        rng = np.random.default_rng(5)
        for mat in ("neo-hookean", "linear"):
            sh = AxisymmetricShell(r0, z0, MU, 2e-3, nu=0.3 if mat == "linear" else 0.5,
                                   material=mat, youngs_modulus=6e6, ribs=ts.BASELINE_RIBS)
            sh.set_gravity(9.81, 1000.0, 1150.0)
            r = r0 + 1e-3 * rng.standard_normal(41)
            z = z0 + 1e-3 * rng.standard_normal(41)
            r[0], z[0], r[-1] = r0[0], z0[0], 0.0
            _, gr, gz = sh.energy(r, z, grad=True)
            scale = np.abs(np.r_[gr, gz]).max()
            h = 1e-7
            for i in range(1, 41):
                for arr, g in ((r, gr), (z, gz)):
                    if arr is r and i == 40:
                        continue
                    o = arr[i]
                    arr[i] = o + h
                    up = sh.energy(r, z)
                    arr[i] = o - h
                    um = sh.energy(r, z)
                    arr[i] = o
                    self.assertAlmostEqual((up - um) / (2 * h), g[i], delta=1e-6 * scale)

    def test_lateral_model_matches_axisymmetric_with_ribs(self):
        r0, z0 = _cap(40)
        sh = AxisymmetricShell(r0, z0, MU, 2e-3, ribs=ts.BASELINE_RIBS)
        r = 0.98 * r0
        z = z0 - 0.02 * np.sin(np.linspace(0, np.pi, 41))
        r[0], z[0] = r0[0], z0[0]
        self.assertAlmostEqual(HarmonicShell(sh, r, z).energy() / sh.energy(r, z), 1.0, delta=1e-12)

    def test_rib_section_limits(self):
        t, h = 2e-3, 2e-3
        r0, z0 = _cap(20)
        sh = AxisymmetricShell(r0, z0, MU, t, ribs=ts.RibSpec(count=36, height=h, width=3e-3))
        # where the ribs merge (apex) the section is solid t + h
        self.assertAlmostEqual(sh.Ds[-1] / sh.D, ((t + h) / t) ** 3, places=10)
        self.assertAlmostEqual(sh.t_area_node[-1], t + h, places=12)
        # coverage 36 x 3 mm on the 0.3 m rim, T-section second moment by hand
        phi = 36 * 3e-3 / (2 * np.pi * 0.3)
        A = t + phi * h
        yb = phi * h * (t + h) / 2 / A
        I = t ** 3 / 12 + t * yb ** 2 + phi * (h ** 3 / 12 + h * ((t + h) / 2 - yb) ** 2)
        self.assertAlmostEqual(sh.Ds[0] / sh.D, 12 * I / t ** 3, places=10)

    def test_fully_ribbed_plate_equals_solid_plate(self):
        # ribs covering everything: clamped plate behaves as a solid plate t + h
        # (stiffness x8, mass x2 -> frequencies x2), one nodal diameter, lambda^2 = 21.26
        a, t, h, rho, N = 0.2, 0.01, 0.01, 1000.0, 40
        r0, z0 = np.linspace(a, 0, N + 1), np.zeros(N + 1)
        ribs = ts.RibSpec(count=100000, height=h, width=0.01)
        sh = AxisymmetricShell(r0, z0, MU, t, ribs=ribs)
        hs = HarmonicShell(sh, r0, z0)
        HU, _ = hs.hessians()
        _, Ms, _, _ = hs.rubber_mass(rho)
        n = 3 * (N + 1)
        rows = [np.eye(n)[d] for d in range(3)]
        e = np.zeros(n)
        e[3 * N] = e[3 * N + 1] = 1
        rows += [e, np.eye(n)[3 * N + 2]]
        Z = null_space(np.array(rows))
        w2, phi = eigh(Z.T @ HU @ Z, Z.T @ Ms @ Z)
        U = Z @ phi
        md = np.diag(Ms)[:, None]
        fz = np.sum(U[2::3] ** 2 * md[2::3], 0) / np.sum(U ** 2 * md, 0)
        f = np.sqrt(np.abs(w2[fz > 0.9][0])) / (2 * np.pi)
        D_solid = MU * (t + h) ** 3 / 3
        exact = 21.26 / a ** 2 * np.sqrt(D_solid / (rho * (t + h))) / (2 * np.pi)
        self.assertAlmostEqual(f / exact, 1.0, delta=5e-3)


class TestTopHat(unittest.TestCase):
    def test_geometry(self):
        d, T = ts.BASELINE_DIAPHRAGM, ts.BASELINE_TANK
        gas = (T.volume - T.volume_below(d.clamp_z)) + AxisymmetricShell.enclosed_volume(d.r0, d.z0)
        self.assertAlmostEqual(gas / T.volume, 0.05, places=9)
        self.assertAlmostEqual(d.clamp_z, 0.2)
        self.assertAlmostEqual(d.r0[0], T.radius)
        self.assertTrue(0.005 < d.gap < 0.008)
        self.assertEqual(d.stress_free_fill, 0.95)

    def test_stress_free_at_95_percent(self):
        ds = DiaphragmStatics(ts.BASELINE_DIAPHRAGM)
        s = ds.solve(0.95)
        self.assertLess(s.energy, 1e-9)
        np.testing.assert_allclose(s.r, ts.BASELINE_DIAPHRAGM.r0, atol=1e-9)

    def test_self_contact_keeps_layers_apart(self):
        # coarse top hat swept down into the range where the rolling fold closes
        d = ts.top_hat_diaphragm(ts.BASELINE_TANK, n_segments=60, thickness=2e-3,
                                 density=1150.0, shear_modulus=1.8e6, ribs=ts.BASELINE_RIBS)
        ds = DiaphragmStatics(d)
        s = ds.sweep([0.45], max_step=0.02, min_step=0.002)[-1]
        self.assertTrue(s.converged)
        self.assertAlmostEqual(s.fill, 0.45)
        self.assertEqual(s.self_intersections, [])
        self.assertGreater(len(s.self_contact), 0)          # the fold has closed
        sv = ds.solver
        pi = np.array([p[0] for p in s.self_contact])
        pj = np.array([p[1] for p in s.self_contact])
        dist = np.abs(sv._pair_geometry(s.r, s.z, pi, pj)[0])
        self.assertGreater(dist.min(), d.thickness - 1e-6)


if __name__ == "__main__":
    unittest.main()
