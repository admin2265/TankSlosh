"""Demonstration of the Level A slosh model.

Case 1  1-g free-surface reference: flat-bottom cylinder on the ground, lateral
        acceleration pulse. Classical theory gives the parameters, so this is
        also a ready-made check case for a StarCCM+ VOF set-up.
Case 2  Spacecraft in orbit with a diaphragm tank, using the PLACEHOLDER parameter
        file diaphragm_emm_template.json: a thruster force pulse on the dry
        structure, giving the accelerometer signal and liquid CG motion.

Run from the project root:  py -3.12 examples/demo.py
Plots and CSVs are written to examples/output/.
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tankslosh as ts  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
G = 9.80665


def case1_ground_reference():
    R, H = 0.2, 0.5
    print("Case 1: 1-g free-surface cylinder, R = 0.2 m, H = 0.5 m (reference only)")
    print("  fill   m1/m    f1 [Hz]   h1 [m]   zeta1")
    table = ts.cylinder_free_surface_table(R, H, np.arange(0.1, 0.91, 0.1), G, n_modes=3)
    for e in table.entries:
        m1 = e.lateral_modes[0]
        print(f"  {e.fill_fraction:4.0%}  {m1.mass / e.liquid_mass:6.3f}  {m1.freq_hz:8.4f}"
              f"  {m1.height:7.4f}  {m1.zeta:.2e}")

    p = table.at(0.5)
    t = ts.time_grid(10.0, 0.002)
    ax = ts.trapezoid_pulse(t, 0.5, 0.5, 0.5, rise=0.05)        # 0.5 m/s^2 lateral pulse
    r = ts.simulate_prescribed(p, t, ts.vector_input(t, x=ax, z=G))  # +g: tank on the ground
    r.to_csv(os.path.join(OUT, "case1_ground.csv"))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    axs[0].plot(t, ax)
    axs[0].set_ylabel("tank accel x [m/s²]")
    axs[1].plot(t, r.interface_force[:, 0], label="model")
    axs[1].plot(t, -p.liquid_mass * ax, "--", label="frozen liquid (−m·a)")
    axs[1].set_ylabel("liquid force on tank, x [N]")
    axs[1].legend()
    axs[2].plot(t, 1e3 * r.liquid_cg[:, 0])
    axs[2].set_ylabel("liquid CG x [mm]")
    axs[2].set_xlabel("time [s]")
    fig.suptitle("Case 1: 1-g free-surface cylinder, 50 % fill (reference)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_ground.png"), dpi=120)
    plt.close(fig)


def case2_orbit_coupled():
    p = ts.EMMParams.load_json(os.path.join(HERE, "diaphragm_emm_template.json"))
    print("\nCase 2: orbit, diaphragm tank with PLACEHOLDER parameters")
    print(p.summary())
    dry_mass = 50.0
    t = ts.time_grid(20.0, 0.001)
    Fx = ts.trapezoid_pulse(t, 1.0, 2.0, 5.0, rise=0.05)   # 5 N lateral thruster for 2 s
    Fz = ts.trapezoid_pulse(t, 8.0, 2.0, 10.0, rise=0.05)  # 10 N axial thruster for 2 s

    t0 = time.perf_counter()
    r = ts.simulate_coupled(p, t, ts.vector_input(t, x=Fx, z=Fz), dry_mass=dry_mass)
    dt_run = time.perf_counter() - t0
    print(f"  simulated {t[-1]:.0f} s at {1 / (t[1] - t[0]):.0f} Hz in {dt_run * 1e3:.1f} ms")
    r.to_csv(os.path.join(OUT, "case2_orbit.csv"))

    rigid = (Fx / (dry_mass + p.liquid_mass), Fz / (dry_mass + p.liquid_mass))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    axs[0].plot(t, Fx, label="F_x")
    axs[0].plot(t, Fz, label="F_z")
    axs[0].set_ylabel("thruster force [N]")
    axs[0].legend()
    axs[1].plot(t, 1e3 * r.specific_force[:, 0], label="accel x")
    axs[1].plot(t, 1e3 * rigid[0], "--", label="accel x, frozen liquid")
    axs[1].plot(t, 1e3 * r.specific_force[:, 2], label="accel z")
    axs[1].set_ylabel("accelerometer [mm/s²]")
    axs[1].legend()
    axs[2].plot(t, 1e3 * r.liquid_cg[:, 0], label="CG x")
    axs[2].plot(t, 1e3 * (r.liquid_cg[:, 2] - p.cg_height), label="CG z − rest")
    axs[2].set_ylabel("liquid CG shift [mm]")
    axs[2].set_xlabel("time [s]")
    axs[2].legend()
    fig.suptitle("Case 2: orbit, coupled spacecraft (PLACEHOLDER diaphragm parameters)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case2_orbit.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    case1_ground_reference()
    case2_orbit_coupled()
    print(f"\nPlots and CSVs written to {os.path.relpath(OUT)}")
