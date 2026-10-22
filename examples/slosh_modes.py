"""Lateral and axial slosh modes of the baseline diaphragm tank over 5-95 % fill,
the resulting Level A parameter table, and a Level A run with it.

Run from the project root:  py -3.12 examples/slosh_modes.py   (about 2 min)
Writes to examples/output/:
  baseline_emm.json           EMMTable (lateral + axial) for tankslosh.simulate; zeta is a placeholder
  slosh_modes.csv             dominant modes per fill
  slosh_modes.png             lateral: frequency / slosh mass / height vs fill
  axial_modes.png             axial: frequency / slosh mass vs fill
  slosh_pulse_response.png    Level A response to lateral and axial pulses at three fills
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
import tankslosh as ts  # noqa: E402
from tankslosh.diaphragm_static import DiaphragmStatics  # noqa: E402
from tankslosh.modal import emm_table  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
N_KEEP, N_KEEP_AX, ZETA = 3, 2, 0.02


def main():
    fills = np.round(np.arange(0.05, 0.951, 0.05), 3)
    t0 = time.perf_counter()
    ds = DiaphragmStatics(ts.BASELINE_DIAPHRAGM, n_segments=100)   # top hat uses its own 150-segment mesh
    table, results, axial, _ = emm_table(ds, fills, n_keep_lat=N_KEEP, n_keep_ax=N_KEEP_AX, zeta=ZETA,
                                         sweep_kwargs=dict(max_step=0.01, min_step=0.001))
    print(f"statics + modes for {len(results)} fills in {time.perf_counter() - t0:.0f} s")
    table.notes += f"; {N_KEEP} lateral + {N_KEEP_AX} axial modes kept, zeta = {ZETA} placeholder (fit to CFD)"
    table.save_json(os.path.join(OUT, "baseline_emm.json"))

    rows = []
    print(" fill  contents[kg]  stable   f [Hz] (slosh mass %, height m) of dominant modes")
    for lm, am in zip(results, axial):
        sel = lm.dominant(N_KEEP)
        row = dict(fill=lm.fill, contents_kg=lm.contents_mass, cg_z_m=lm.contents_cg_z,
                   stable=lm.stable, mesh_volume_error=lm.mesh_volume_error)
        for j, i in enumerate(sel, 1):
            row[f"f{j}_hz"] = lm.freq_hz[i]
            row[f"m{j}_frac"] = lm.slosh_mass[i] / lm.contents_mass
            row[f"h{j}_m"] = lm.height[i]
        for j, i in enumerate(am.dominant(N_KEEP_AX), 1):
            row[f"ax{j}_hz"] = am.freq_hz[i]
            row[f"ax{j}_frac"] = am.slosh_mass[i] / am.contents_mass
        row["axial_stable"] = am.stable
        rows.append(row)
        print(f" {lm.fill:4.0%}  {lm.contents_mass:10.2f}  {str(lm.stable):6s}  " + "  ".join(
            f"{lm.freq_hz[i]:6.2f} ({100 * lm.slosh_mass[i] / lm.contents_mass:4.1f}%, {lm.height[i]:.3f})"
            for i in sel) + "   axial: " + "  ".join(
            f"{am.freq_hz[i]:6.2f} ({100 * am.slosh_mass[i] / am.contents_mass:4.1f}%)"
            for i in am.dominant(N_KEEP_AX)))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(OUT, "slosh_modes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    f = np.array([r["fill"] for r in rows])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for j in range(1, N_KEEP + 1):
        y = [r.get(f"f{j}_hz", np.nan) for r in rows]
        axs[0].plot(f, y, "o-", label=f"mode {j}")
        axs[1].plot(f, [100 * r.get(f"m{j}_frac", np.nan) for r in rows], "o-", label=f"mode {j}")
        axs[2].plot(f, [r.get(f"h{j}_m", np.nan) for r in rows], "o-", label=f"mode {j}")
    axs[1].plot(f, [100 * (1 - p.fixed_mass / p.liquid_mass) for p in table.entries], "k--",
                label="sum of kept modes")
    axs[0].set_ylabel("frequency [Hz]")
    axs[1].set_ylabel("slosh mass [% of contents]")
    axs[2].plot(f, [r["cg_z_m"] for r in rows], "k:", label="contents CG")
    axs[2].set_ylabel("height z [m]")
    axs[2].set_xlabel("fill fraction")
    for a in axs:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("Baseline diaphragm tank: lateral slosh modes (μ = 1.8 MPa assumed)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slosh_modes.png"), dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for j in range(1, N_KEEP_AX + 1):
        axs[0].plot(f, [r.get(f"ax{j}_hz", np.nan) for r in rows], "o-", label=f"axial mode {j}")
        axs[1].plot(f, [100 * r.get(f"ax{j}_frac", np.nan) for r in rows], "o-", label=f"axial mode {j}")
    axs[0].set_ylabel("frequency [Hz]")
    axs[1].set_ylabel("slosh mass [% of contents]")
    axs[1].set_xlabel("fill fraction")
    for a in axs:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("Baseline diaphragm tank: axial (breathing) modes (μ = 1.8 MPa assumed)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "axial_modes.png"), dpi=120)
    plt.close(fig)

    # Level A response at three fills: 1 mm/s^2 lateral pulse at 0.2 s, axial pulse at 2.2 s
    t = ts.time_grid(4.5, 5e-4)
    ax = ts.trapezoid_pulse(t, 0.2, 0.5, 1e-3, rise=0.01)
    az = ts.trapezoid_pulse(t, 2.2, 0.5, 1e-3, rise=0.01)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for fill in (0.25, 0.55, 0.75):
        p = table.at(fill)
        r = ts.simulate_prescribed(p, t, ts.vector_input(t, x=ax, z=az), ref_point=ts.BASELINE_MOUNT_POINT)
        axs[0].plot(t, 1e3 * r.interface_force[:, 0], label=f"{fill:.0%} fill")
        axs[1].plot(t, 1e6 * r.liquid_cg[:, 0], label=f"{fill:.0%} fill")
        axs[2].plot(t, 1e3 * r.interface_force[:, 2], label=f"{fill:.0%} fill")
    axs[0].set_ylabel("liquid force F_x [mN]")
    axs[1].set_ylabel("contents CG x [µm]")
    axs[2].set_ylabel("liquid force F_z [mN]")
    axs[2].set_xlabel("time [s]")
    for a in axs:
        a.legend(fontsize=8)
    fig.suptitle(f"Level A with Level B parameters: 1 mm/s² lateral then axial pulse (ζ = {ZETA} placeholder)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slosh_pulse_response.png"), dpi=120)
    print(f"wrote {os.path.relpath(OUT)}: baseline_emm.json, slosh_modes.csv/.png, "
          "axial_modes.png, slosh_pulse_response.png")


if __name__ == "__main__":
    main()
