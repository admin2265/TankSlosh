import unittest
import warnings

import numpy as np

from tankslosh import WATER_20C, bond_number, cylinder_free_surface, cylinder_free_surface_table
from tankslosh.freesurface import j1_prime_roots

G = 9.80665


class TestBesselIdentities(unittest.TestCase):
    def test_first_root(self):
        self.assertAlmostEqual(j1_prime_roots(1)[0], 1.841184, places=5)

    def test_shallow_limit_mass_sum(self):
        # sum 2/(xi^2 - 1) = 1: all liquid sloshes in the shallow limit
        n = 400
        xi = j1_prime_roots(n)
        tail = 2.0 / (np.pi ** 2 * n)
        self.assertAlmostEqual(np.sum(2.0 / (xi ** 2 - 1)) + tail, 1.0, delta=2e-4)

    def test_static_cg_identity(self):
        # sum 1/(xi^2 (xi^2 - 1)) = 1/8 gives the exact tilted-surface CG shift
        xi = j1_prime_roots(60)
        self.assertAlmostEqual(np.sum(1.0 / (xi ** 2 * (xi ** 2 - 1))), 0.125, places=7)


class TestCylinderParams(unittest.TestCase):
    def setUp(self):
        self.R, self.h = 0.2, 0.3
        self.p = cylinder_free_surface(self.R, self.h, G, n_modes=5, zeta=0.01, z_bottom=0.1)

    def test_mass_and_cg_bookkeeping(self):
        p = self.p
        m = WATER_20C.density * np.pi * self.R ** 2 * self.h
        self.assertAlmostEqual(p.liquid_mass, m, places=9)
        self.assertAlmostEqual(p.fixed_mass + sum(md.mass for md in p.lateral_modes), m, places=9)
        self.assertAlmostEqual(p.cg_height, 0.1 + self.h / 2, places=12)

    def test_first_mode_frequency(self):
        xi = 1.841184
        w = np.sqrt(G * xi / self.R * np.tanh(xi * self.h / self.R))
        self.assertAlmostEqual(self.p.lateral_modes[0].freq_hz, w / (2 * np.pi), places=5)

    def test_deep_tank_heights(self):
        R, h = 0.1, 1.0
        p = cylinder_free_surface(R, h, G, n_modes=2, zeta=0.0)
        xi = j1_prime_roots(2)
        for md, x in zip(p.lateral_modes, xi):
            self.assertAlmostEqual(md.height, h - 2 * R / x, places=7)  # tanh(xi h/2R) ~ 1
        self.assertLess(p.lateral_modes[0].mass / p.liquid_mass, 0.05)

    def test_low_bond_warning(self):
        self.assertLess(bond_number(1e-4, 0.2), 100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cylinder_free_surface(0.2, 0.3, 1e-4)
        self.assertTrue(any("Bond number" in str(x.message) for x in w))

    def test_table(self):
        tab = cylinder_free_surface_table(0.2, 0.6, [0.1, 0.5, 0.9], G, zeta=0.01)
        p = tab.at(0.3)
        self.assertAlmostEqual(p.fixed_mass + sum(md.mass for md in p.lateral_modes),
                               p.liquid_mass, places=9)


if __name__ == "__main__":
    unittest.main()
