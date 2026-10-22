"""Pre-test predictions for a 1-g shaker test of the baseline tank (tank axis
vertical, gas on top): lateral and axial slosh modes at each test fill, with
gravity on. Compare these with the measured frequency response functions; the
orbit predictions come from examples/slosh_modes.py.

Run from the project root:  py -3.12 examples/ground_test_predictions.py   (about 10 min)
Writes examples/output/ground_test_modes.csv and ground_test_emm.json.
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tankslosh as ts  # noqa: E402
from tankslosh.diaphragm_static import DiaphragmStatics  # noqa: E402
from tankslosh.modal import emm_table  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
G = 9.81


def main():
    fills = np.round(np.arange(0.05, 0.951, 0.05), 3)
    t0 = time.perf_counter()
    ds = DiaphragmStatics(ts.BASELINE_DIAPHRAGM, gravity=(G, ts.WATER_20C.density))
    table, lat, ax, _ = emm_table(ds, fills, n_keep_lat=3, n_keep_ax=2, zeta=0.02,
                                  sweep_kwargs=dict(max_step=0.01, min_step=0.001))
    print(f"1-g statics + modes for {len(lat)} fills in {time.perf_counter() - t0:.0f} s")
    table.notes += "; 1 g, gas on top (ground test). Level A runs need specific force z = +9.81"
    table.save_json(os.path.join(OUT, "ground_test_emm.json"))
    rows = []
    for lm, am in zip(lat, ax):
        row = dict(fill=lm.fill, stable=bool(lm.stable and am.stable))
        for j, i in enumerate(lm.dominant(3), 1):
            row[f"lat{j}_hz"], row[f"lat{j}_frac"] = lm.freq_hz[i], lm.slosh_mass[i] / lm.contents_mass
        for j, i in enumerate(am.dominant(2), 1):
            row[f"ax{j}_hz"], row[f"ax{j}_frac"] = am.freq_hz[i], am.slosh_mass[i] / am.contents_mass
        rows.append(row)
        print(f" {lm.fill:4.0%} stable={row['stable']!s:5s} lateral " + ", ".join(
            f"{lm.freq_hz[i]:.2f} Hz ({100 * lm.slosh_mass[i] / lm.contents_mass:.1f}%)" for i in lm.dominant(3))
            + " | axial " + ", ".join(
            f"{am.freq_hz[i]:.2f} Hz ({100 * am.slosh_mass[i] / am.contents_mass:.1f}%)" for i in am.dominant(2)))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, "ground_test_modes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
