"""Static diaphragm shape, pressure difference and stretch over the 5-95 % fill range
for the baseline tank (PROBLEM_STATEMENT.md).

Run from the project root:  py -3.12 examples/diaphragm_statics.py
Writes examples/output/diaphragm_statics.{png,csv} and diaphragm_shapes.npz.
"""
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tankslosh import BASELINE_DIAPHRAGM, BASELINE_TANK  # noqa: E402
from tankslosh.diaphragm_static import DiaphragmStatics  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def contact_length(ds, s):
    c = set(s.contact_nodes.tolist())
    L = np.hypot(np.diff(s.r), np.diff(s.z))
    return float(sum(L[j] for j in range(len(L)) if j in c and j + 1 in c))


def main():
    fills = np.round(np.arange(0.05, 0.951, 0.05), 3)
    ds = DiaphragmStatics(BASELINE_DIAPHRAGM, n_segments=100)
    t0 = time.perf_counter()
    sols = ds.sweep(fills)
    print(f"solved {len(sols)} fill levels in {time.perf_counter() - t0:.0f} s")

    rows = []
    print(" fill   dp [kPa]  energy [J]  max stretch  min stretch  min t [mm]  contact [m]")
    for s in sols:
        row = dict(fill=s.fill, dp_kPa=s.dp / 1e3, energy_J=s.energy,
                   max_stretch=s.max_stretch, min_stretch=s.min_stretch,
                   min_thickness_mm=1e3 * BASELINE_DIAPHRAGM.thickness * s.thickness_ratio.min(),
                   contact_length_m=contact_length(ds, s), apex_z_m=s.z[-1],
                   converged=s.converged, equilibrium_residual=s.equilibrium_residual)
        rows.append(row)
        print(f" {s.fill:4.0%}  {row['dp_kPa']:8.2f}  {s.energy:10.2f}  {s.max_stretch:11.2f}"
              f"  {s.min_stretch:11.2f}  {row['min_thickness_mm']:10.2f}  {row['contact_length_m']:10.3f}"
              + ("" if s.converged else "  NOT CONVERGED"))

    with open(os.path.join(OUT, "diaphragm_statics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(os.path.join(OUT, "diaphragm_shapes.npz"), fills=[s.fill for s in sols],
             r=np.array([s.r for s in sols]), z=np.array([s.z for s in sols]),
             r0=ds.shell.r0, z0=ds.shell.z0)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3)
    zz = np.linspace(0, BASELINE_TANK.height, 601)
    rr = np.array([BASELINE_TANK.radius_at(z) for z in zz])
    for k, (sel, title) in enumerate(((lambda f: f <= 0.5, "fill 5–50 %"),
                                      (lambda f: f >= 0.5, "fill 50–95 %"))):
        ax = fig.add_subplot(gs[:, k])
        ax.plot(rr, zz, "k", lw=1)
        ax.plot(-rr, zz, "k", lw=1)
        cmap = plt.get_cmap("viridis")
        for s in sols:
            if sel(s.fill):
                c = cmap(s.fill)
                ax.plot(s.r, s.z, color=c, lw=1.2, label=f"{s.fill:.0%}")
                ax.plot(-s.r, s.z, color=c, lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.legend(fontsize=7, ncol=2, loc="upper center" if k == 0 else "lower center")
    f = [r["fill"] for r in rows]
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(f, [r["dp_kPa"] for r in rows], "o-")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("p_liquid − p_gas [kPa]")
    ax.set_title("pressure difference across diaphragm")
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(f, [r["max_stretch"] for r in rows], "o-", label="max stretch")
    ax.plot(f, [r["min_stretch"] for r in rows], "s-", label="min stretch")
    ax.set_xlabel("fill fraction")
    ax.set_ylabel("principal stretch")
    ax.legend()
    fig.suptitle("Baseline diaphragm statics: 3 mm Shore 70A molded cup (μ = 1.8 MPa, assumed)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diaphragm_statics.png"), dpi=120)
    print(f"wrote {os.path.relpath(OUT)}/diaphragm_statics.png, .csv, diaphragm_shapes.npz")


if __name__ == "__main__":
    main()
