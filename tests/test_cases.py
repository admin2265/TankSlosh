"""Case folders: case.toml parsing and an end-to-end run of a tiny case."""
import csv
import os
import tempfile
import textwrap
import unittest

import numpy as np

import tankslosh as ts
from tankslosh.cases import load_case
from tankslosh.emm import EMMTable
from tankslosh.run_case import rebuild_report, run_case

ROOT = os.path.join(os.path.dirname(__file__), "..")

TINY = textwrap.dedent("""
    name = "tiny"
    estimated = ["everything"]

    [tank]
    radius = 0.2
    height = 0.6

    [diaphragm]
    type = "cup"
    clamp_z = 0.5
    stress_free_fill = 0.5
    thickness = 3e-3
    shear_modulus = 1.8e6
    density = 1150.0
    segments = 40

    [environment]
    orbit = true
    ground_test = true

    [analysis]
    fills = [0.45, 0.5, 0.55]
    liquid_mesh = 0.01
    n_keep_lateral = 2
    n_keep_axial = 1
    """)


class TestLoadCase(unittest.TestCase):
    def test_committed_cases_match_library_baselines(self):
        c = load_case(os.path.join(ROOT, "cases", "top-hat-95pct-2mm"))
        b = ts.BASELINE_DIAPHRAGM
        np.testing.assert_array_equal(c.diaphragm.r0, b.r0)
        np.testing.assert_array_equal(c.diaphragm.z0, b.z0)
        self.assertEqual(c.diaphragm.ribs, b.ribs)
        self.assertEqual((c.diaphragm.thickness, c.diaphragm.shear_modulus, c.diaphragm.density),
                         (b.thickness, b.shear_modulus, b.density))
        self.assertEqual(c.tank, ts.BASELINE_TANK)
        self.assertEqual(c.mount_point, ts.BASELINE_MOUNT_POINT)
        np.testing.assert_array_equal(c.fills, np.round(np.arange(0.05, 0.951, 0.05), 3))
        c = load_case(os.path.join(ROOT, "cases", "cup-50pct-3mm"))
        self.assertEqual(c.diaphragm, ts.CUP_3MM_DIAPHRAGM)
        self.assertEqual(c.n_segments, 100)

    def test_meridian_points_file(self):
        th = np.linspace(0, np.pi / 2, 31)
        r0, z0 = 0.2 * np.cos(th), 0.5 + 0.1 * np.sin(th)
        r0[-1] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            np.savetxt(os.path.join(tmp, "m.csv"), np.column_stack([r0, z0]), delimiter=",",
                       header="r,z")
            with open(os.path.join(tmp, "case.toml"), "w") as fh:
                fh.write(TINY.replace('type = "cup"\nclamp_z = 0.5',
                                      'type = "meridian"\npoints_file = "m.csv"\n'
                                      'material = "linear"\nyoungs_modulus = 6e6\npoisson = 0.45'))
            c = load_case(tmp)
        self.assertEqual(c.n_segments, 30)
        self.assertEqual(c.diaphragm.material, "linear")
        np.testing.assert_allclose(c.diaphragm.z0, z0)


class TestRunTinyCase(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "case.toml"), "w") as fh:
                fh.write(TINY)
            results = run_case(tmp, verbose=False)
            out = os.path.join(tmp, "output")
            for f in ("statics.csv", "statics.png", "shapes.npz", "orbit_emm.json", "orbit_modes.csv",
                      "orbit_lateral.png", "orbit_axial.png", "orbit_pulse_response.png",
                      "ground_statics.csv", "ground_emm.json", "ground_modes.csv", "ground_modes.png",
                      "results.md"):
                self.assertTrue(os.path.exists(os.path.join(out, f)), f)
            table = EMMTable.load_json(os.path.join(out, "orbit_emm.json"))
            self.assertEqual(len(table.entries), 3)
            self.assertEqual(len(table.entries[0].lateral_modes), 2)
            with open(os.path.join(out, "orbit_modes.csv")) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([float(r["fill"]) for r in rows], [0.45, 0.5, 0.55])
            self.assertEqual(len(results["ground"]), 3)
            # stress-free fill: no pressure difference
            dp = {r["fill"]: r["dp_kPa"] for r in results["statics"]}
            self.assertLess(abs(dp[0.5]), 1e-6)
            with open(os.path.join(out, "results.md"), encoding="utf-8") as fh:
                md = fh.read()
            self.assertIn("Inputs that are estimates", md)
            self.assertIn("1 g, gas on top", md)
            # report rebuilt from the CSVs keeps the run line and the same tables
            rebuild_report(tmp)
            with open(os.path.join(out, "results.md"), encoding="utf-8") as fh:
                md2 = fh.read()
            self.assertIn("- Run:", md2)
            self.assertIn("Report rebuilt", md2)
            strip = lambda t: [ln for ln in t.splitlines() if ln.startswith("| ")]  # noqa: E731
            self.assertEqual(strip(md), strip(md2))


if __name__ == "__main__":
    unittest.main()
