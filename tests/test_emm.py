import os
import tempfile
import unittest

from tankslosh import EMMParams, EMMTable, SloshMode


def make(fill, m=10.0, m1=3.0, f1=1.0):
    return EMMParams(liquid_mass=m, fixed_mass=m - m1, fixed_height=0.2,
                     lateral_modes=[SloshMode(m1, f1, 0.02, 0.3)],
                     axial_modes=[SloshMode(1.0, 2.0 * f1, 0.05)], fill_fraction=fill)


class TestEMM(unittest.TestCase):
    def test_mass_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            EMMParams(liquid_mass=10, fixed_mass=5, fixed_height=0,
                      lateral_modes=[SloshMode(4, 1, 0.01)])

    def test_json_round_trip(self):
        p = make(0.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            p.save_json(path)
            q = EMMParams.load_json(path)
        self.assertEqual(p.to_dict(), q.to_dict())

    def test_table_interpolation(self):
        tab = EMMTable([make(0.9, m=18, m1=2, f1=2.0), make(0.1, m=2, m1=1, f1=1.0)])
        self.assertEqual(list(tab.fills), [0.1, 0.9])
        mid = tab.at(0.5)
        self.assertAlmostEqual(mid.liquid_mass, 10.0)
        self.assertAlmostEqual(mid.lateral_modes[0].freq_hz, 1.5)
        self.assertAlmostEqual(mid.fixed_mass + mid.lateral_modes[0].mass, mid.liquid_mass)
        self.assertEqual(tab.at(0.9).to_dict()["liquid_mass"], 18)
        with self.assertRaises(ValueError):
            tab.at(0.95)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            tab.save_json(path)
            tab2 = EMMTable.load_json(path)
        self.assertAlmostEqual(tab2.at(0.5).liquid_mass, 10.0)


if __name__ == "__main__":
    unittest.main()
