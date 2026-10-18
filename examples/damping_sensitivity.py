"""Sensitivity of the Level A results to the damping ratio zeta.

Loads examples/output/baseline_emm.json (Level B parameters, written by
slosh_modes.py), overrides zeta for every mode, and compares responses at
three fills: pulse responses (time domain) and steady-shaking force
transmissibility (frequency domain).

Run from the project root:  py -3.12 examples/damping_sensitivity.py [zeta ...]
Default zetas: 0.02 (current placeholder) and 0.04.
Writes examples/output/damping_sensitivity_{pulse,frf}.png and .csv.
"""
import copy
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tankslosh as ts  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
FILLS = (0.25, 0.55, 0.75)
ACC = 1e-3          # pulse amplitude, m/s^2
PULSE = (0.2, 0.5)  # start, duration (s)


def with_zeta(p: ts.EMMParams, zeta: float) -> ts.EMMParams:
    q = copy.deepcopy(p)
    for m in q.lateral_modes + q.axial_modes:
        m.zeta = zeta
    return q


def envelope_decay_time(t, y, t0, frac=0.1):
    """Time after t0 until |y| stays below frac * (max |y| after t0)."""
    sel = t >= t0
    yy, tt = np.abs(y[sel]), t[sel]
    peak = yy.max()
    above = np.where(yy > frac * peak)[0]
    return float(tt[above[-1]] - t0) if len(above) else 0.0


def transmissibility(p: ts.EMMParams, f, axial=False):
    """|F| / (m_contents |a|) for steady sinusoidal tank acceleration."""
    w = 2 * np.pi * np.asarray(f)
    modes = p.axial_modes if axial else p.lateral_modes
    H = np.ones_like(w, dtype=complex)
    for md in modes:
        wn = md.omega
        H += (md.mass / p.liquid_mass) * w ** 2 / (wn ** 2 - w ** 2 + 2j * md.zeta * wn * w)
    return np.abs(H)


def main(zetas):
    table = ts.EMMTable.load_json(os.path.join(OUT, "baseline_emm.json"))
    t = ts.time_grid(25.0, 5e-4)
    a_x = ts.trapezoid_pulse(t, *PULSE, ACC, rise=0.01)
    a_z = ts.trapezoid_pulse(t, *PULSE, ACC, rise=0.01)
    t_end = PULSE[0] + PULSE[1]
    styles = ["-", "--", ":", "-."]

    rows = []
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for c, fill in enumerate(FILLS):
        base = table.at(fill)
        for k, z in enumerate(zetas):
            p = with_zeta(base, z)
            rl = ts.simulate_prescribed(p, t, ts.vector_input(t, x=a_x), ref_point=ts.BASELINE_MOUNT_POINT)
            ra = ts.simulate_prescribed(p, t, ts.vector_input(t, z=a_z), ref_point=ts.BASELINE_MOUNT_POINT)
            Fx, cgx, My = rl.interface_force[:, 0], rl.liquid_cg[:, 0], rl.interface_moment[:, 1]
            Fz = ra.interface_force[:, 2]
            after = t >= t_end
            rigid_x = -p.liquid_mass * a_x
            rows.append(dict(
                fill=fill, zeta=z, f1_hz=p.lateral_modes[0].freq_hz,
                peak_Fx_mN=1e3 * np.abs(Fx).max(),
                peak_Fx_over_rigid=np.abs(Fx).max() / (p.liquid_mass * ACC),
                peak_cgx_um=1e6 * np.abs(cgx).max(),
                residual_Fx_rms_mN=1e3 * np.sqrt(np.mean((Fx[after] - rigid_x[after]) ** 2)),
                ringing_10pct_s=envelope_decay_time(t, cgx, t_end),
                peak_My_mNm=1e3 * np.abs(My).max(),
                peak_Fz_mN=1e3 * np.abs(Fz).max(),
                peak_resonance_gain=float(transmissibility(p, np.linspace(0.2, 80, 8000)).max()),
            ))
            lab = f"ζ = {z:g}"
            axs[0, c].plot(t, 1e3 * Fx, styles[k], label=lab)
            axs[1, c].plot(t, 1e6 * cgx, styles[k], label=lab)
            axs[2, c].plot(t, 1e3 * Fz, styles[k], label=lab)
        axs[0, c].set_title(f"{fill:.0%} fill  (f₁ = {base.lateral_modes[0].freq_hz:.2f} Hz)")
    axs[0, 0].set_ylabel("liquid force F_x [mN]\n(lateral pulse)")
    axs[1, 0].set_ylabel("contents CG x [µm]\n(lateral pulse)")
    axs[2, 0].set_ylabel("liquid force F_z [mN]\n(axial pulse)")
    for a in axs[2]:
        a.set_xlabel("time [s]")
    for a in axs.ravel():
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"Damping sensitivity: {1e3 * ACC:g} mm/s² pulse, {PULSE[1]} s, orbit")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "damping_sensitivity_pulse.png"), dpi=110)
    plt.close(fig)

    f = np.linspace(0.2, 30, 3000)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for c, fill in enumerate(FILLS):
        base = table.at(fill)
        for k, z in enumerate(zetas):
            p = with_zeta(base, z)
            axs[0, c].semilogy(f, transmissibility(p, f), styles[k], label=f"ζ = {z:g}")
            axs[1, c].semilogy(f, transmissibility(p, f, axial=True), styles[k], label=f"ζ = {z:g}")
        axs[0, c].set_title(f"{fill:.0%} fill")
    axs[0, 0].set_ylabel("lateral |F| / (m |a|)")
    axs[1, 0].set_ylabel("axial |F| / (m |a|)")
    for a in axs[1]:
        a.set_xlabel("shaking frequency [Hz]")
    for a in axs.ravel():
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle("Force transmissibility under steady shaking (1 = contents move as a rigid mass)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "damping_sensitivity_frf.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT, "damping_sensitivity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = ("fill", "zeta", "peak Fx/rigid", "peak CGx um", "resid Fx rms mN", "ring to 10% s",
           "peak My mNm", "peak Fz mN", "resonance gain")
    print(" | ".join(hdr))
    for r in rows:
        print(f"{r['fill']:.0%} | {r['zeta']:g} | {r['peak_Fx_over_rigid']:.3f} | {r['peak_cgx_um']:.2f} | "
              f"{r['residual_Fx_rms_mN']:.2f} | {r['ringing_10pct_s']:.2f} | {r['peak_My_mNm']:.2f} | "
              f"{r['peak_Fz_mN']:.2f} | {r['peak_resonance_gain']:.1f}")


if __name__ == "__main__":
    zs = [float(a) for a in sys.argv[1:]] or [0.02, 0.04]
    main(zs)
