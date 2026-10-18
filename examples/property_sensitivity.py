"""Sensitivity of the Level B slosh results to the assumed diaphragm properties:
shear modulus mu, thickness t and rubber density.

For each variant the full chain is recomputed (static shapes over fill, then
lateral and axial modes). Results are cached in
examples/output/property_sensitivity.json so plots can be redrawn quickly.

Run from the project root:  py -3.12 examples/property_sensitivity.py   (about 7 min)
                            py -3.12 examples/property_sensitivity.py --plot   (redraw from cache)
Writes examples/output/property_sensitivity.{json,png,csv}.
"""
import csv
import dataclasses
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tankslosh import BASELINE_DIAPHRAGM as B  # noqa: E402
from tankslosh.diaphragm_static import DiaphragmStatics  # noqa: E402
from tankslosh.modal import axial_modes, lateral_modes  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
CACHE = os.path.join(OUT, "property_sensitivity.json")
FILLS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95]
REPORT_FILLS = [0.1, 0.3, 0.55, 0.65, 0.8, 0.95]

VARIANTS = {
    "baseline": {},
    "mu -30%": {"shear_modulus": 0.7 * B.shear_modulus},
    "mu +30%": {"shear_modulus": 1.3 * B.shear_modulus},
    "t = 2.5 mm": {"thickness": 2.5e-3},
    "t = 3.5 mm": {"thickness": 3.5e-3},
    "rho = 1000": {"density": 1000.0},
    "rho = 1300": {"density": 1300.0},
}
GROUPS = [("shear modulus μ (±30 %)", ["mu -30%", "baseline", "mu +30%"]),
          ("thickness", ["t = 2.5 mm", "baseline", "t = 3.5 mm"]),
          ("rubber density", ["rho = 1000", "baseline", "rho = 1300"])]


def run_variant(changes):
    d = dataclasses.replace(B, **changes)
    ds = DiaphragmStatics(d, n_segments=100)
    out = []
    for s in ds.sweep(FILLS):
        lm, am = lateral_modes(ds, s), axial_modes(ds, s)
        il, ia = lm.dominant(1)[0], am.dominant(1)[0]
        out.append(dict(fill=s.fill, dp_kPa=s.dp / 1e3, max_stretch=s.max_stretch,
                        lat_f=float(lm.freq_hz[il]), lat_m=float(lm.slosh_mass[il] / lm.contents_mass),
                        lat_h=float(lm.height[il]), ax_f=float(am.freq_hz[ia]),
                        ax_m=float(am.slosh_mass[ia] / am.contents_mass),
                        stable=bool(lm.stable and am.stable)))
    return out


def compute():
    res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for name, ch in VARIANTS.items():
        if name in res:
            continue
        t0 = time.perf_counter()
        res[name] = run_variant(ch)
        print(f"{name}: {time.perf_counter() - t0:.0f} s", flush=True)
        json.dump(res, open(CACHE, "w"), indent=1)
    return res


def plot_and_table(res):
    qty = [("lat_f", "lateral mode 1\nfrequency [Hz]"), ("lat_m", "lateral mode 1\nslosh mass [%]"),
           ("ax_f", "axial mode 1\nfrequency [Hz]"), ("dp_kPa", "Δp across\ndiaphragm [kPa]")]
    fig, axs = plt.subplots(len(qty), 3, figsize=(14, 11), sharex=True)
    styles = {0: ("--", "tab:blue"), 1: ("-", "k"), 2: (":", "tab:red")}
    for c, (title, names) in enumerate(GROUPS):
        for k, nm in enumerate(names):
            f = [r["fill"] for r in res[nm]]
            for rr, (key, lab) in enumerate(qty):
                y = [100 * r[key] if key == "lat_m" else r[key] for r in res[nm]]
                ls, col = styles[k]
                axs[rr, c].plot(f, y, ls, color=col, marker="o", ms=3, label=nm)
                axs[rr, 0].set_ylabel(lab)
        axs[0, c].set_title(title)
    for a in axs.ravel():
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    for a in axs[-1]:
        a.set_xlabel("fill fraction")
    fig.suptitle("Sensitivity of slosh results to the assumed diaphragm properties")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "property_sensitivity.png"), dpi=110)
    plt.close(fig)

    base = {round(r["fill"], 3): r for r in res["baseline"]}
    rows = []
    print("% change vs baseline:  lateral f1 / lateral mass1 / axial f1")
    for nm in VARIANTS:
        if nm == "baseline":
            continue
        cur = {round(r["fill"], 3): r for r in res[nm]}
        cells = []
        for f in REPORT_FILLS:
            b, v = base[f], cur[f]
            d = {k: 100 * (v[k] / b[k] - 1) for k in ("lat_f", "lat_m", "ax_f")}
            rows.append(dict(variant=nm, fill=f, **{f"d_{k}_pct": d[k] for k in d}))
            cells.append(f"{f:.0%}: {d['lat_f']:+5.1f}/{d['lat_m']:+5.1f}/{d['ax_f']:+5.1f}")
        print(f"{nm:11s} " + " | ".join(cells))
    with open(os.path.join(OUT, "property_sensitivity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(CACHE)) if "--plot" in sys.argv else compute()
    plot_and_table(res)
