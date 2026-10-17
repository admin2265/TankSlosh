import unittest

import numpy as np
from scipy.linalg import eigh, null_space
from scipy.special import jn_zeros

from tankslosh import cylinder_free_surface
from tankslosh.diaphragm_static import DiaphragmStatics
from tankslosh.harmonic_shell import HarmonicShell
from tankslosh.liquid_fe import (HarmonicPotential, build_mesh, surface_normals,
                                 translation_coupling)
from tankslosh.modal import lateral_modes
from tankslosh.shell import AxisymmetricShell

MU = 1.8e6


def _vacuum_modes(sh, r, z, rho):
    """m = 1 modes of a clamped shell without liquid; returns freqs and u_z mass fraction."""
    hs = HarmonicShell(sh, r, z)
    HU, _ = hs.hessians()
    _, Ms, _, _ = hs.rubber_mass(rho)
    N, n = sh.N, 3 * (sh.N + 1)
    rows = [np.eye(n)[d] for d in range(3)]
    e = np.zeros(n)
    e[3 * N] = e[3 * N + 1] = 1
    rows += [e, np.eye(n)[3 * N + 2]]
    Z = null_space(np.array(rows))
    w2, phi = eigh(Z.T @ HU @ Z, Z.T @ Ms @ Z)
    U = Z @ phi
    md = np.diag(Ms)[:, None]
    fz = np.sum(U[2::3] ** 2 * md[2::3], 0) / np.sum(U ** 2 * md, 0)
    return np.sqrt(np.abs(w2)) / (2 * np.pi), fz


class TestHarmonicShell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = DiaphragmStatics(n_segments=50)
        cls.sols = cls.ds.sweep([0.2, 0.8])

    def test_unperturbed_energy_and_volume_match_axisymmetric(self):
        for s in self.sols:
            hs = HarmonicShell(self.ds.shell, s.r, s.z)
            U2 = self.ds.shell.energy(s.r, s.z)
            self.assertAlmostEqual(hs.energy() / U2, 1.0, delta=1e-8)
            self.assertAlmostEqual(hs.enclosed_volume(), self.ds.shell.enclosed_volume(s.r, s.z),
                                   delta=1e-15)

    def test_rigid_translation_is_free(self):
        for s in self.sols:
            hs = HarmonicShell(self.ds.shell, s.r, s.z)
            u = np.zeros(3 * (self.ds.shell.N + 1))
            u[0::3], u[1::3] = 1e-3, -1e-3
            self.assertAlmostEqual(hs.energy(u), hs.energy(), delta=1e-9)
            HU, HV = hs.hessians()
            self.assertLess(abs(u @ HU @ u), 1e-6 * np.abs(HU).max() * 1e-6)
            np.testing.assert_allclose(HU, HU.T, atol=1e-9 * np.abs(HU).max())

    def test_prestressed_membrane_frequencies(self):
        # flat membrane under equibiaxial tension: f = j_{1,n} / (2 pi a) sqrt(T / rho_A)
        a0, t0, lam, rho, N = 0.15, 1e-4, 1.3, 1000.0, 40
        r0 = np.linspace(a0, 0, N + 1)
        sh = AxisymmetricShell(r0, np.zeros(N + 1), MU, t0)
        f, fz = _vacuum_modes(sh, lam * r0, np.zeros(N + 1), rho)
        a, T, rA = lam * a0, MU * t0 * (1 - lam ** -6), rho * t0 / lam ** 2
        exact = jn_zeros(1, 3) / a * np.sqrt(T / rA) / (2 * np.pi)
        np.testing.assert_allclose(f[fz > 0.9][:3], exact, rtol=3e-3)

    def test_clamped_plate_frequencies(self):
        # clamped plate, one nodal diameter: lambda^2 = 21.26, 60.82
        a, t, rho, N = 0.2, 0.01, 1000.0, 40
        sh = AxisymmetricShell(np.linspace(a, 0, N + 1), np.zeros(N + 1), MU, t)
        f, fz = _vacuum_modes(sh, sh.r0, sh.z0, rho)
        exact = np.array([21.26, 60.82]) / a ** 2 * np.sqrt(sh.D / (rho * t)) / (2 * np.pi)
        np.testing.assert_allclose(f[fz > 0.9][:2], exact, rtol=5e-3)


class TestLiquidFE(unittest.TestCase):
    def test_free_surface_slosh_matches_theory(self):
        R, h, g, rho, hm = 0.2, 0.3, 9.80665, 998.2, 0.006
        nb, nh = int(round(R / hm)), int(round(h / hm))
        wall = np.vstack([np.column_stack([np.linspace(0, R, nb + 1), np.zeros(nb + 1)]),
                          np.column_stack([np.full(nh, R), np.linspace(0, h, nh + 1)[1:]])])
        surf = np.column_stack([np.linspace(R, 0, nb + 1), np.full(nb + 1, h)])
        mesh = build_mesh(wall, surf, hm)
        V, zc = mesh.volume_and_cg()
        self.assertAlmostEqual(V / (np.pi * R ** 2 * h), 1.0, places=10)
        self.assertAlmostEqual(zc, h / 2, places=10)
        hp = HarmonicPotential(mesh, 1)
        nrm = surface_normals(surf)
        G = hp.surface_load_matrix(nrm, 3)
        X = hp.solve(G)
        Ma = rho * np.pi * G.T @ X
        Gam = rho * np.pi * translation_coupling(surf, nrm, 3)
        H = rho * np.pi * (hp.c @ X)
        ns = len(surf)
        Kg = np.zeros((3 * ns, 3 * ns))
        for j in range(ns - 1):
            a, b = surf[j], surf[j + 1]
            L = np.linalg.norm(b - a)
            for q in (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)):
                rr = (1 - q) * a[0] + q * b[0]
                Nq = np.array([1 - q, q])
                idx = [3 * j + 2, 3 * (j + 1) + 2]
                Kg[np.ix_(idx, idx)] += rho * g * np.pi * 0.5 * L * rr * np.outer(Nq, Nq)
        keep = np.arange(2, 3 * ns, 3)[:-1]
        w2, phi = eigh(Kg[np.ix_(keep, keep)], Ma[np.ix_(keep, keep)])
        Gn = phi.T @ Gam[keep]
        ref = cylinder_free_surface(R, h, g, n_modes=2, zeta=0.0)
        m = rho * np.pi * R ** 2 * h
        for k in range(2):
            md = ref.lateral_modes[k]
            self.assertAlmostEqual(np.sqrt(w2[k]) / (2 * np.pi) / md.freq_hz, 1.0, delta=3e-3)
            self.assertAlmostEqual(Gn[k] ** 2 / md.mass, 1.0, delta=1e-2)
            self.assertAlmostEqual((phi[:, k] @ H[keep]) / Gn[k] / md.height, 1.0, delta=1e-3)
        self.assertLess(np.sum(Gn ** 2), m)


class TestBaselineModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = DiaphragmStatics(n_segments=50)
        cls.res = {round(s.fill, 2): lateral_modes(cls.ds, s) for s in cls.ds.sweep([0.1, 0.55])}

    def test_sanity(self):
        for f, lm in self.res.items():
            self.assertTrue(lm.stable)
            self.assertLess(abs(lm.mesh_volume_error), 1e-2)
            self.assertLess(lm.slosh_mass.sum(), lm.contents_mass)
            p = lm.emm()
            self.assertAlmostEqual(p.cg_height, lm.contents_cg_z, places=9)
            self.assertGreater(p.fixed_mass, 0)

    def test_regression(self):
        # 50-segment values; mesh-converged to ~2 % (docs/slosh_modes.md)
        k = self.res[0.1].dominant(1)[0]
        self.assertAlmostEqual(self.res[0.1].freq_hz[k], 6.08, delta=0.15)
        self.assertAlmostEqual(self.res[0.1].slosh_mass[k] / self.res[0.1].contents_mass, 0.495, delta=0.01)
        k = self.res[0.55].dominant(1)[0]
        self.assertAlmostEqual(self.res[0.55].freq_hz[k], 1.10, delta=0.05)


if __name__ == "__main__":
    unittest.main()
