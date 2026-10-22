"""Run one customer case: statics, orbit slosh modes and (if requested) 1-g modes.

    py -3.12 -m tankslosh.run_case cases/<name>
    py -3.12 -m tankslosh.run_case cases/<name> --fills 0.3,0.6 --out scratch/   (quick look)

Reads cases/<name>/case.toml (see tankslosh/cases.py) and writes to
cases/<name>/output/:
  statics.csv, statics.png, shapes.npz   static shape, pressure difference, stretch (orbit)
  orbit_emm.json, orbit_modes.csv        Level A table and dominant modes per fill (orbit)
  orbit_lateral.png, orbit_axial.png     frequency / slosh mass / height vs fill
  orbit_pulse_response.png               Level A response to lateral then axial pulses
  ground_statics.csv, ground_emm.json, ground_modes.csv, ground_modes.png   (1 g, gas on top)
  results.md                             short summary of inputs and results
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .cases import Case, load_case  # noqa: E402
from .diaphragm_static import DiaphragmStatics  # noqa: E402
from .inputs import time_grid, trapezoid_pulse, vector_input  # noqa: E402
from .modal import emm_table  # noqa: E402
from .simulate import simulate_prescribed  # noqa: E402


def contact_length(s) -> float:
    """Meridian length of the diaphragm lying on the tank wall, m."""
    c = set(s.contact_nodes.tolist())
    L = np.hypot(np.diff(s.r), np.diff(s.z))
    return float(sum(L[j] for j in range(len(L)) if j in c and j + 1 in c))


def _write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def statics_rows(case: Case, sols):
    return [dict(fill=s.fill, dp_kPa=s.dp / 1e3, energy_J=s.energy,
                 max_stretch=s.max_stretch, min_stretch=s.min_stretch,
                 min_thickness_mm=1e3 * case.diaphragm.thickness * s.thickness_ratio.min(),
                 contact_length_m=contact_length(s), apex_z_m=s.z[-1],
                 converged=s.converged, equilibrium_residual=s.equilibrium_residual)
            for s in sols]


def mode_rows(case: Case, lat, ax):
    rows = []
    for lm, am in zip(lat, ax):
        row = dict(fill=lm.fill, contents_kg=lm.contents_mass, cg_z_m=lm.contents_cg_z,
                   stable=lm.stable, mesh_volume_error=lm.mesh_volume_error)
        for j, i in enumerate(lm.dominant(case.n_keep_lateral), 1):
            row[f"f{j}_hz"] = lm.freq_hz[i]
            row[f"m{j}_frac"] = lm.slosh_mass[i] / lm.contents_mass
            row[f"h{j}_m"] = lm.height[i]
        for j, i in enumerate(am.dominant(case.n_keep_axial), 1):
            row[f"ax{j}_hz"] = am.freq_hz[i]
            row[f"ax{j}_frac"] = am.slosh_mass[i] / am.contents_mass
        row["axial_stable"] = am.stable
        rows.append(row)
    return rows


def plot_statics(case: Case, sols, rows, path):
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3)
    tank = case.tank
    zz = np.linspace(0, tank.height, 601)
    rr = np.array([tank.radius_at(z) for z in zz])
    f_mid = float(np.median([s.fill for s in sols]))
    for k, (sel, title) in enumerate(((lambda f: f <= f_mid, f"fill ≤ {f_mid:.0%}"),
                                      (lambda f: f >= f_mid, f"fill ≥ {f_mid:.0%}"))):
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
    fig.suptitle(f"{case.name}: diaphragm statics (orbit)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_lateral(case: Case, rows, table, title, path):
    f = np.array([r["fill"] for r in rows])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for j in range(1, case.n_keep_lateral + 1):
        axs[0].plot(f, [r.get(f"f{j}_hz", np.nan) for r in rows], "o-", label=f"mode {j}")
        axs[1].plot(f, [100 * r.get(f"m{j}_frac", np.nan) for r in rows], "o-", label=f"mode {j}")
        axs[2].plot(f, [r.get(f"h{j}_m", np.nan) for r in rows], "o-", label=f"mode {j}")
    tf = table.fills
    axs[1].plot(tf, [100 * (1 - p.fixed_mass / p.liquid_mass) for p in table.entries], "k--",
                label="sum of kept modes")
    axs[0].set_ylabel("frequency [Hz]")
    axs[1].set_ylabel("slosh mass [% of contents]")
    axs[2].plot(f, [r["cg_z_m"] for r in rows], "k:", label="contents CG")
    axs[2].set_ylabel("height z [m]")
    axs[2].set_xlabel("fill fraction")
    unstable = [r["fill"] for r in rows if not (r["stable"] and r["axial_stable"])]
    for a in axs:
        for fu in unstable:
            a.axvline(fu, color="r", alpha=0.15, lw=6)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(title + ("  (red: unstable fills)" if unstable else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_axial(case: Case, rows, title, path):
    f = np.array([r["fill"] for r in rows])
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for j in range(1, case.n_keep_axial + 1):
        axs[0].plot(f, [r.get(f"ax{j}_hz", np.nan) for r in rows], "o-", label=f"axial mode {j}")
        axs[1].plot(f, [100 * r.get(f"ax{j}_frac", np.nan) for r in rows], "o-", label=f"axial mode {j}")
    axs[0].set_ylabel("frequency [Hz]")
    axs[1].set_ylabel("slosh mass [% of contents]")
    axs[1].set_xlabel("fill fraction")
    for a in axs:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pulse_fills(case: Case, table):
    tf = table.fills
    want = case.pulse_fills or [0.25, 0.55, 0.75]
    out = sorted({float(min(max(f, tf[0]), tf[-1])) for f in want})
    return out


def plot_pulse(case: Case, table, path):
    """Level A response to a 1 mm/s² lateral pulse at 0.2 s and an axial pulse at 2.2 s."""
    t = time_grid(4.5, 5e-4)
    ax = trapezoid_pulse(t, 0.2, 0.5, 1e-3, rise=0.01)
    az = trapezoid_pulse(t, 2.2, 0.5, 1e-3, rise=0.01)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for fill in pulse_fills(case, table):
        r = simulate_prescribed(table.at(fill), t, vector_input(t, x=ax, z=az), ref_point=case.mount_point)
        axs[0].plot(t, 1e3 * r.interface_force[:, 0], label=f"{fill:.0%} fill")
        axs[1].plot(t, 1e6 * r.liquid_cg[:, 0], label=f"{fill:.0%} fill")
        axs[2].plot(t, 1e3 * r.interface_force[:, 2], label=f"{fill:.0%} fill")
    axs[0].set_ylabel("liquid force F_x [mN]")
    axs[1].set_ylabel("contents CG x [µm]")
    axs[2].set_ylabel("liquid force F_z [mN]")
    axs[2].set_xlabel("time [s]")
    for a in axs:
        a.legend(fontsize=8)
    fig.suptitle(f"{case.name}: Level A, 1 mm/s² lateral then axial pulse (ζ = {case.zeta} placeholder)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _modes_text(rows, prefix_f, prefix_m, n):
    parts = []
    for j in range(1, n + 1):
        if f"{prefix_f}{j}_hz" in rows:
            parts.append(f"{rows[f'{prefix_f}{j}_hz']:.2f} Hz ({100 * rows[f'{prefix_m}{j}_frac']:.0f} %)")
    return ", ".join(parts) or "–"


def _modes_table(case: Case, rows):
    lines = ["| Fill | Lateral modes (slosh mass) | Axial modes (slosh mass) | Stable |",
             "|---|---|---|---|"]
    for r in rows:
        stable = r["stable"] and r["axial_stable"]
        lines.append(f"| {r['fill']:.0%} | {_modes_text(r, 'f', 'm', case.n_keep_lateral)} | "
                     f"{_modes_text(r, 'ax', 'ax', case.n_keep_axial)} | "
                     f"{'yes' if stable else '**no**'} |")
    unstable = [f"{r['fill']:.0%}" for r in rows if not (r["stable"] and r["axial_stable"])]
    lines.append("")
    lines.append("Unstable fills (left out of the parameter table): " + (", ".join(unstable) or "none") + ".")
    return lines


def _git_version(folder):
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=folder,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _input_lines(case: Case):
    d = case.raw["diaphragm"]
    t = case.tank
    lines = ["| Input | Value |", "|---|---|",
             f"| Tank | R = {t.radius} m, H = {t.height} m, heads 1:{t.head_ratio:g} "
             f"(volume {1e3 * t.volume:.2f} L) |",
             f"| Diaphragm | {case.diaphragm_type}, mass {case.diaphragm.mass:.3f} kg, "
             f"{case.n_segments} segments |"]
    for k, v in d.items():
        if k not in ("type", "segments"):
            lines.append(f"| diaphragm.{k} | {v} |")
    gap = getattr(case.diaphragm, "gap", None)
    if gap is not None:
        lines.append(f"| top-hat gap to wall (derived) | {1e3 * gap:.2f} mm |")
    lines += [f"| Liquid | {case.liquid_name}, ρ = {case.fluid.density} kg/m³ |",
              f"| Environments | {'orbit' if case.orbit else ''}"
              f"{', ' if case.orbit and case.ground_test else ''}"
              f"{f'1 g = {case.g} m/s² (gas on top)' if case.ground_test else ''} |",
              f"| Fills | {', '.join(f'{f:.0%}' for f in case.fills)} |",
              f"| Statics sweep step | max {case.sweep_max_step}, min {case.sweep_min_step} |",
              f"| Liquid mesh | {case.liquid_mesh} m |",
              f"| Self-contact of folded layers | {'on' if case.self_contact else 'off'} |",
              f"| Modes kept | {case.n_keep_lateral} lateral, {case.n_keep_axial} axial |",
              f"| Damping ζ | {case.zeta} (placeholder until fitted to CFD or test) |",
              f"| Mount point | {case.mount_point} m |"]
    return lines


def write_results_md(case: Case, out, results, elapsed):
    L = [f"# {case.name}: results", "",
         "_Generated by `python -m tankslosh.run_case`; do not edit by hand — edit `case.toml` "
         "and rerun._", "",
         (f"- Run: {datetime.datetime.now():%Y-%m-%d %H:%M}, code version `{_git_version(case.folder)}`, "
          f"{elapsed / 60:.1f} min" if elapsed is not None else
          f"- Report rebuilt {datetime.datetime.now():%Y-%m-%d %H:%M} from the existing output files "
          f"(code version `{_git_version(case.folder)}`)"), ""]
    if case.description:
        L += [case.description.strip(), ""]
    L += ["## Inputs", ""] + _input_lines(case) + [""]
    if case.estimated:
        L += ["**Inputs that are estimates:** " + "; ".join(case.estimated) + ".", ""]
    if case.notes:
        L += ["**Notes:** " + case.notes.strip(), ""]
    if "statics" in results:
        rows = results["statics"]
        dp = [r["dp_kPa"] for r in rows]
        L += ["## Statics (orbit)", "",
              f"- Pressure difference across the diaphragm: {min(dp):.2f} to {max(dp):.2f} kPa.",
              f"- Principal stretch: {min(r['min_stretch'] for r in rows):.2f} to "
              f"{max(r['max_stretch'] for r in rows):.2f}.",
              f"- Converged: {sum(r['converged'] for r in rows)} of {len(rows)} fills "
              f"(of {len(case.fills)} requested).", "",
              "| Fill | Δp [kPa] | Max stretch | Min stretch | Wall contact [m] | Apex z [m] |",
              "|---|---|---|---|---|---|"]
        L += [f"| {r['fill']:.0%} | {r['dp_kPa']:.3f} | {r['max_stretch']:.3f} | {r['min_stretch']:.3f} | "
              f"{r['contact_length_m']:.3f} | {r['apex_z_m']:.3f} |" for r in rows]
        L.append("")
    for key, title in (("orbit", "Slosh modes: orbit (`orbit_emm.json`)"),
                       ("ground", f"Slosh modes: 1 g, gas on top (`ground_emm.json`)")):
        if key in results:
            L += [f"## {title}", ""] + _modes_table(case, results[key]) + [""]
    if "ground" in results:
        L += ["Level A runs with `ground_emm.json` need specific force z = +g.", ""]
    L += ["## Files", ""]
    L += [f"- `{f}`" for f in sorted(os.listdir(out)) if f != "results.md"]
    L.append("")
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _read_rows(path):
    def conv(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v
    with open(path, newline="") as fh:
        return [{k: conv(v) for k, v in r.items() if v != ""} for r in csv.DictReader(fh)]


def rebuild_report(folder: str, out: str = None):
    """Rewrite results.md from the CSVs already in the output folder (after editing
    the description, estimates or notes in case.toml), without rerunning."""
    case = load_case(folder)
    out = out or case.output_dir
    results = {key: _read_rows(os.path.join(out, f)) for key, f in
               (("statics", "statics.csv"), ("orbit", "orbit_modes.csv"), ("ground", "ground_modes.csv"))
               if os.path.exists(os.path.join(out, f))}
    write_results_md(case, out, results, elapsed=None)
    return results


def run_case(folder: str, out: str = None, fills=None, orbit=None, ground=None, verbose=True):
    """Run the case in `folder`; returns a dict of result rows per analysis."""
    case = load_case(folder)
    if fills is not None:
        case.fills = np.round(np.asarray(fills, float), 6)
    if orbit is not None:
        case.orbit = orbit
    if ground is not None:
        case.ground_test = ground
    out = out or case.output_dir
    os.makedirs(out, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    sweep = dict(max_step=case.sweep_max_step, min_step=case.sweep_min_step)
    common = dict(n_keep_lat=case.n_keep_lateral, n_keep_ax=case.n_keep_axial, zeta=case.zeta,
                  h_mesh=case.liquid_mesh, fluid=case.fluid)
    results = {}
    t_start = time.perf_counter()

    if case.orbit:
        t0 = time.perf_counter()
        ds = DiaphragmStatics(case.diaphragm, n_segments=case.n_segments, self_contact=case.self_contact)
        sols = ds.sweep(case.fills, **sweep)
        say(f"[{case.name}] orbit statics: {len(sols)} fills in {time.perf_counter() - t0:.0f} s")
        rows = statics_rows(case, sols)
        _write_csv(os.path.join(out, "statics.csv"), rows)
        np.savez(os.path.join(out, "shapes.npz"), fills=[s.fill for s in sols],
                 r=np.array([s.r for s in sols]), z=np.array([s.z for s in sols]),
                 r0=ds.shell.r0, z0=ds.shell.z0)
        plot_statics(case, sols, rows, os.path.join(out, "statics.png"))
        results["statics"] = rows

        t0 = time.perf_counter()
        table, lat, ax, _ = emm_table(ds, case.fills, sols=sols, notes=f"Level B lateral + axial slosh modes, "
                                      f"case {case.name}, orbit (no gravity)", **common)
        say(f"[{case.name}] orbit modes in {time.perf_counter() - t0:.0f} s")
        table.notes += (f"; {case.n_keep_lateral} lateral + {case.n_keep_axial} axial modes kept, "
                        f"zeta = {case.zeta} placeholder (fit to CFD)")
        table.save_json(os.path.join(out, "orbit_emm.json"))
        rows = mode_rows(case, lat, ax)
        _write_csv(os.path.join(out, "orbit_modes.csv"), rows)
        plot_lateral(case, rows, table, f"{case.name}: lateral slosh modes, orbit",
                     os.path.join(out, "orbit_lateral.png"))
        plot_axial(case, rows, f"{case.name}: axial (breathing) modes, orbit",
                   os.path.join(out, "orbit_axial.png"))
        plot_pulse(case, table, os.path.join(out, "orbit_pulse_response.png"))
        results["orbit"] = rows

    if case.ground_test:
        t0 = time.perf_counter()
        ds = DiaphragmStatics(case.diaphragm, n_segments=case.n_segments, self_contact=case.self_contact,
                              gravity=(case.g, case.fluid.density))
        sols = ds.sweep(case.fills, **sweep)
        _write_csv(os.path.join(out, "ground_statics.csv"), statics_rows(case, sols))
        table, lat, ax, _ = emm_table(ds, case.fills, sols=sols, notes=f"Level B lateral + axial slosh modes, "
                                      f"case {case.name}", **common)
        say(f"[{case.name}] 1-g statics + modes in {time.perf_counter() - t0:.0f} s")
        table.notes += (f"; {case.n_keep_lateral} lateral + {case.n_keep_axial} axial modes kept, "
                        f"zeta = {case.zeta} placeholder; {case.g:g} m/s^2, gas on top (ground test). "
                        f"Level A runs need specific force z = +{case.g:g}")
        table.save_json(os.path.join(out, "ground_emm.json"))
        rows = mode_rows(case, lat, ax)
        _write_csv(os.path.join(out, "ground_modes.csv"), rows)
        plot_lateral(case, rows, table, f"{case.name}: lateral slosh modes, {case.g:g} m/s² (gas on top)",
                     os.path.join(out, "ground_modes.png"))
        results["ground"] = rows

    elapsed = time.perf_counter() - t_start
    write_results_md(case, out, results, elapsed)
    say(f"[{case.name}] done in {elapsed / 60:.1f} min; wrote {out}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m tankslosh.run_case", description=__doc__.split("\n\n")[0])
    p.add_argument("case", help="case folder holding case.toml, e.g. cases/top-hat-95pct-2mm")
    p.add_argument("--out", help="output folder (default: <case>/output)")
    p.add_argument("--fills", help="comma-separated fills overriding the case, e.g. 0.3,0.6")
    p.add_argument("--no-orbit", action="store_true", help="skip statics and orbit modes")
    p.add_argument("--no-ground", action="store_true", help="skip the 1-g modes even if the case asks")
    p.add_argument("--report-only", action="store_true",
                   help="only rewrite results.md from the existing output files")
    a = p.parse_args(argv)
    if a.report_only:
        rebuild_report(a.case, out=a.out)
        return
    fills = [float(f) for f in a.fills.split(",")] if a.fills else None
    run_case(a.case, out=a.out, fills=fills, orbit=False if a.no_orbit else None,
             ground=False if a.no_ground else None)


if __name__ == "__main__":
    sys.exit(main())
