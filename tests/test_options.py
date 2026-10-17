"""Tests for the optional model features: gravity, linear-elastic membrane,
bending switch and diaphragms given by their molded meridian."""
import unittest

import numpy as np
from scipy.linalg import eigh

from tankslosh import EllipticalHeadTank, MeridianDiaphragm, hemisphere_diaphragm
from tankslosh.diaphragm_static import DiaphragmStatics
from tankslosh.fluid import Fluid
from tankslosh.harmonic_shell import HarmonicShell
from tankslosh.liquid_fe import HarmonicPotential, build_mesh, surface_normals
from tankslosh.modal import lateral_modes
from tankslosh.shell import AxisymmetricShell

G = 9.8


def _hemisphere(n, R=0.3, zc=0.3):
    th = np.linspace(0, np.pi / 2, n + 1)
    r0, z0 = R * np.cos(th), zc + R * np.sin(th)
    r0[-1] = 0.0
    return r0, z0


class TestLinearMaterialAndGravity(unittest.TestCase):
    def test_linear_inflated_sphere(self):
        # equibiaxial linear membrane: dU/dV = 2 E t e / ((1 - nu) lam^2 R0), e = lam - 1
        E, t, nu, R0, N = 6.2e6, 1.2e-3, 0.3, 0.365, 200
        th = np.linspace(0, np.pi / 2, N + 1)
        r0, z0 = R0 * np.cos(th), -R0 * np.sin(th)
        r0[-1] = 0.0
        sh = AxisymmetricShell(r0, z0, None, t, nu=nu, material="linear", youngs_modulus=E,
                               bending=False)
        lam, h = 1.05, 1e-6
        dU = sh.energy((lam + h) * r0, (lam + h) * z0) - sh.energy((lam - h) * r0, (lam - h) * z0)
        dV = sh.enclosed_volume((lam + h) * r0, (lam + h) * z0) - \
            sh.enclosed_volume((lam - h) * r0, (lam - h) * z0)
        exact = 2 * E * t / (1 - nu) * (lam - 1) / (lam ** 2 * R0)
        self.assertAlmostEqual((dU / dV) / exact, 1.0, delta=1e-4)

    def test_gradients_with_gravity(self):
        r0, z0 = _hemisphere(40)
        rng = np.random.default_rng(0)
        for mat in ("neo-hookean", "linear"):
            sh = AxisymmetricShell(r0, z0, 1e6, 1e-3, nu=0.3 if mat == "linear" else 0.5,
                                   material=mat, youngs_modulus=6e6)
            sh.set_gravity(G, 1444.0, 1040.0)
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

    def test_first_moment_exact(self):
        # int z dV of the hemispherical cap above the clamp plane z = zc (signed negative)
        R, zc = 0.3, 0.3
        exact = -(zc * 2 / 3 * np.pi * R ** 3 + np.pi * R ** 4 / 4)
        r0, z0 = _hemisphere(400, R, zc)
        self.assertAlmostEqual(AxisymmetricShell.enclosed_zmoment(r0, z0) / exact, 1.0, delta=1e-5)

    def test_harmonic_shell_matches_axisymmetric_with_gravity(self):
        r0, z0 = _hemisphere(40)
        sh = AxisymmetricShell(r0, z0, None, 1.2e-3, nu=0.3, material="linear", youngs_modulus=6.2e6)
        sh.set_gravity(G, 1444.0, 1040.0)
        r, z = 0.98 * r0, z0 - 0.02 * np.sin(np.linspace(0, np.pi, 41))
        r[0], z[0] = r0[0], z0[0]
        hs = HarmonicShell(sh, r, z)
        self.assertAlmostEqual(hs.energy() / sh.energy(r, z), 1.0, delta=1e-9)
        self.assertAlmostEqual(hs.enclosed_zmoment(), sh.enclosed_zmoment(r, z), delta=1e-12)

    def test_bending_switch(self):
        r0, z0 = _hemisphere(20)
        self.assertEqual(AxisymmetricShell(r0, z0, 1e6, 1e-3, bending=False).D, 0.0)
        self.assertGreater(AxisymmetricShell(r0, z0, 1e6, 1e-3).D, 0.0)


def _pinned_free_surface(b, surf, h, rho):
    """omega^2 b / g of a free surface pinned at the wall (reference for a slack
    diaphragm under gravity)."""
    th = np.linspace(np.pi, np.pi / 2, int(np.ceil(b * np.pi / 2 / h)) + 1)
    wall = np.column_stack([b * np.sin(th), b + b * np.cos(th)])
    wall[0], wall[-1] = (0, 0), (b, b)
    hp = HarmonicPotential(build_mesh(wall, surf, h), 1)
    Gm = hp.surface_load_matrix(surface_normals(surf), 3)
    Ma = rho * np.pi * Gm.T @ hp.solve(Gm)
    ns = len(surf)
    Kg = np.zeros((3 * ns, 3 * ns))
    for j in range(ns - 1):
        a, c = surf[j], surf[j + 1]
        L = np.linalg.norm(c - a)
        for q in (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)):
            rr = (1 - q) * a[0] + q * c[0]
            Nq = np.array([1 - q, q])
            idx = [3 * j + 2, 3 * (j + 1) + 2]
            Kg[np.ix_(idx, idx)] += rho * G * np.pi * 0.5 * L * rr * np.outer(Nq, Nq)
    keep = np.arange(2, 3 * ns, 3)[1:-1]
    return eigh(Kg[np.ix_(keep, keep)], Ma[np.ix_(keep, keep)], eigvals_only=True)[0] * b / G


class TestSlackDiaphragmUnderGravity(unittest.TestCase):
    def test_reduces_to_pinned_free_surface_on_any_mesh(self):
        # a stress-free, very thin flat membrane over a half-full sphere at 1 g:
        # only gravity restores, so the slosh mode must equal free-surface sloshing
        # with the contact line pinned (also checks the wall-contact rule is mesh-independent)
        b, rho, h = 0.2, 1000.0, 0.004
        tank = EllipticalHeadTank(radius=b, height=2 * b, head_ratio=1.0)
        fluid = Fluid(rho, 1e-6, 0.07, 2.2e9)
        for N in (60, 120):
            r0, z0 = np.linspace(b, 0, N + 1), np.full(N + 1, b)
            d = MeridianDiaphragm(tank=tank, r0=r0, z0=z0, stress_free_fill=0.5, thickness=1e-4,
                                  density=10.0, youngs_modulus=6e6, poisson=0.3,
                                  material="linear", bending=False)
            ds = DiaphragmStatics(d, gravity=(G, rho))
            s = ds.solve(0.5)
            lm = lateral_modes(ds, s, fluid=fluid, h_mesh=h)
            k = lm.dominant(1)[0]
            w2 = (2 * np.pi * lm.freq_hz[k]) ** 2 * b / G
            ref = _pinned_free_surface(b, np.column_stack([r0, z0]), h, rho)
            self.assertAlmostEqual(w2 / ref, 1.0, delta=5e-4)
            self.assertEqual(lm.detach_node, 0)

    def test_fluid_density_must_match_gravity(self):
        b = 0.2
        tank = EllipticalHeadTank(radius=b, height=2 * b, head_ratio=1.0)
        d = MeridianDiaphragm(tank=tank, r0=np.linspace(b, 0, 21), z0=np.full(21, b),
                              stress_free_fill=0.5, thickness=1e-4, density=10.0,
                              youngs_modulus=6e6, poisson=0.3, material="linear")
        ds = DiaphragmStatics(d, gravity=(G, 1444.0))
        with self.assertRaises(ValueError):
            lateral_modes(ds, ds.solve(0.5), fluid=Fluid(1000.0, 1e-6, 0.07, 2.2e9))


class TestMeridianDiaphragm(unittest.TestCase):
    def test_hemisphere_diaphragm_statics(self):
        tank = EllipticalHeadTank(radius=0.365, height=0.73, head_ratio=1.0)
        d = hemisphere_diaphragm(tank, 40, thickness=1.2e-3, density=1040.0, youngs_modulus=6.2e6,
                                 poisson=0.3, material="linear")
        self.assertAlmostEqual(d.clamp_z, 0.365)
        self.assertAlmostEqual(d.z0[-1], 0.73)
        full = DiaphragmStatics(d).solve(1.0)                      # no gravity
        np.testing.assert_allclose(full.r, d.r0, atol=1e-9)        # stress-free when full
        self.assertLess(full.energy, 1e-9)
        ds = DiaphragmStatics(d, gravity=(G, 1444.0))
        s = ds.solve(0.9, full.r, full.z)
        self.assertTrue(s.converged, s.message)
        # liquid volume = 90 % of the tank, wall contact respected
        self.assertAlmostEqual(ds.enclosed_volume_for_fill(0.9),
                               ds.shell.enclosed_volume(s.r, s.z), delta=1e-9)
        self.assertGreater(ds.solver.contact.g(s.r, s.z).min(), -1e-8)


if __name__ == "__main__":
    unittest.main()
