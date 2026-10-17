import unittest

import numpy as np
from scipy.integrate import quad

from tankslosh import BASELINE_DIAPHRAGM, BASELINE_MOUNT_POINT, BASELINE_TANK


class TestTank(unittest.TestCase):
    def test_baseline_volumes(self):
        t = BASELINE_TANK
        self.assertAlmostEqual(t.head_depth, 0.1)
        self.assertAlmostEqual(t.volume * 1e3, 67.0206, places=3)
        self.assertAlmostEqual(t.liquid_mass(0.5), 33.450, places=2)

    def test_volume_below_matches_integration(self):
        t = BASELINE_TANK
        for z in (0.05, 0.1, 0.3, 0.55, 0.6):
            ref = quad(lambda s: np.pi * t.radius_at(s) ** 2, 0.0, z, points=[0.1, 0.5], limit=200)[0]
            self.assertAlmostEqual(t.volume_below(z), ref, places=9)
        self.assertAlmostEqual(t.volume_below(0.6), t.volume, places=12)

    def test_mount_point(self):
        self.assertEqual(BASELINE_MOUNT_POINT, (0.2, 0.0, 0.3))


class TestCup(unittest.TestCase):
    def test_molded_shape(self):
        d = BASELINE_DIAPHRAGM
        self.assertAlmostEqual(d.dome_equator_z, 0.36667, places=5)
        self.assertAlmostEqual(d.apex_z, 0.26667, places=5)
        self.assertAlmostEqual(d.meridian_length, 0.37554, places=5)
        # liquid below the cup = lower head + cylinder to equator - dome = 50 %
        t = d.tank
        v = t.volume_below(d.dome_equator_z) - t.head_volume
        self.assertAlmostEqual(v / t.volume, 0.5, places=12)

    def test_mass(self):
        d = BASELINE_DIAPHRAGM
        a, b = d.tank.radius, d.tank.head_depth
        # dome area by surface-of-revolution integral
        area = quad(lambda th: 2 * np.pi * a * np.sin(th) * np.hypot(a * np.cos(th), b * np.sin(th)),
                    0, np.pi / 2)[0]
        expected = d.density * d.thickness * (area + 2 * np.pi * a * d.sidewall_length)
        self.assertAlmostEqual(d.mass, expected, places=9)


if __name__ == "__main__":
    unittest.main()
