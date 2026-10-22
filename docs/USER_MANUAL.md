# tankslosh: user manual

This manual explains how to run the slosh models from your own Python scripts. No other tools are needed. It assumes basic Python (running a script, editing numbers in it). The physics behind each step is described in the documents linked at the end.

---

## 1. What the package does

`tankslosh` predicts how liquid sloshes in a tank whose liquid is held under a rubber diaphragm, when the tank is given small translational accelerations (no rotation). It works at two levels.

| Level | What it is | Speed | Use it to |
|---|---|---|---|
| **A: spring-mass model** (`tankslosh.simulate`) | The liquid is a fixed mass plus a few spring-mass oscillators, with parameters that depend on fill level | milliseconds per run | Get the force and moment on the tank, the liquid CG motion and the accelerometer signal for any acceleration or thrust history |
| **B: physics model** (`tankslosh.diaphragm_static`, `tankslosh.modal`) | The diaphragm is a large-deformation rubber shell. The liquid is potential flow. Static shapes and vibration modes are computed and turned into Level A parameters | about 1.5 min (plain cup) to 17 min (ribbed top hat) for a full 5–95 % fill table | Produce or update the Level A parameters when the tank, diaphragm or material changes |

Normal workflow:
1. Level B makes a parameter table for each tank design (a *case*, section 7), for example `cases/top-hat-95pct-2mm/output/orbit_emm.json`.
2. Level A runs any number of load cases with it.

A ready-made table for the baseline tank is included, so most users only need Level A.

**The baseline tank** (see `PROBLEM_STATEMENT.md`):
- **Tank:** cylinder, 0.4 m diameter, 0.6 m tall overall, with 2:1 elliptical heads. Volume 67 L, water.
- **Diaphragm:** a molded rubber "top hat" with radial ribs. The web is 2 mm thick, with about 36 ribs on the gas side, and the material is EPDM-class rubber (μ ≈ 1.8 MPa, assumed). Its brim is clamped to the wall at z = 0.2 m. It lines the upper tank with a 6.4 mm gas gap and is stress-free at 95 % fill. As the tank empties it rolls into a U-shaped fold. The previous 3 mm cup baseline is still available as `ts.CUP_3MM_DIAPHRAGM`.
- **Fill range:** 5–95 %. The environment is orbit (no gravity).
- **Reported loads:** at the side mount at mid-height, point (0.2, 0, 0.3) m.

---

## 2. Installation

1. Install Python 3.10 or newer. The package was tested with Python 3.12, numpy 2.4, scipy 1.17 and matplotlib 3.11.
2. Get the code:
   ```bash
   git clone https://github.com/admin2265/TankSlosh.git
   cd TankSlosh
   ```
   To use the gravity, linear-material and custom-shape options, check out the branch that has them until it is merged:
   ```bash
   git checkout utsumi-validation
   ```
3. Install the three dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Check the installation by running the test suite. It takes about 40 s and should end with `OK`:
   ```bash
   python -m unittest discover -s tests -t .
   ```

Run everything from the top-level `TankSlosh` folder. Python must also be able to find the `tankslosh` package. Use one of these:
- Save your scripts in the `TankSlosh` folder itself.
- Set the environment variable `PYTHONPATH` to that folder.
- Start each script with:
```python
import sys; sys.path.insert(0, r"C:\path\to\TankSlosh")
import tankslosh as ts
```
The examples below use paths such as `examples/output/baseline_emm.json`, which are relative to the `TankSlosh` folder, so run them from there.
On Windows, `py -3.12` can replace `python` in the commands.

---

## 3. Ready-made scripts

Each script writes its plots and CSV files to `examples/output/`.

| Command | What it does | Run time |
|---|---|---|
| `python examples/demo.py` | Level A demo: a 1-g free-surface reference case, plus an orbit case with placeholder parameters | seconds |
| `python examples/diaphragm_statics.py` | Diaphragm shape, pressure difference and stretch at 5–95 % fill | about 17 min for the top hat (1.5 min for the old cup) |
| `python examples/ground_test_predictions.py` | 1-g slosh modes for a shaker test of the baseline tank; writes `ground_test_emm.json` and `ground_test_modes.csv` | about 18 min |
| `python examples/slosh_modes.py` | Lateral and axial slosh modes over fill. **Rewrites `baseline_emm.json`**, then runs Level A pulses with it | about 17 min for the top hat |
| `python examples/damping_sensitivity.py 0.01 0.02 0.04` | Effect of the damping ratio on pulse response and resonance | seconds |
| `python examples/property_sensitivity.py` | Effect of rubber stiffness, thickness and density on the results. `--plot` redraws from the cache | about 7 min |

To analyse a changed tank or diaphragm, set up a case (section 7): copy a folder under `cases/`, edit its `case.toml` and run `python -m tankslosh.run_case cases/<name>`. The scripts above are kept as references for the baseline; section 6 shows how to build the objects yourself for one-off studies.

---

## 4. Units and conventions (read once)

- **Units:** SI throughout (m, kg, s, N, Pa). Convert psi with 1 psi = 6894.76 Pa.
- **Tank frame:**
  - z runs along the tank axis, with z = 0 at the inside bottom of the lower head and gas at the top.
  - x and y are sideways.
- **Fill fraction:** liquid volume ÷ total tank volume, from 0 to 1.
- **Specific force `f`:** the input to Level A is what an accelerometer fixed to the tank reads, i.e. tank acceleration minus gravity.
  - In orbit it equals the tank acceleration.
  - For a tank resting on the ground it is (0, 0, +9.81).
  - All input arrays have shape (N, 3) for x, y, z. Build them with `ts.vector_input(t, x=..., y=..., z=...)`.
- **Time grid:** must be uniformly spaced. `ts.time_grid(duration, dt)` makes one. Inputs are treated as varying linearly between samples, and the integration is exact, so `dt` only has to resolve your input and the outputs you want.
- **Outputs (`SimResult`):**
  - **`interface_force` (N×3, N):** the force the liquid (plus diaphragm) applies to the tank.
  - **`interface_moment` (N×3, N·m):** the moment of that force about `ref_point`.
  - **`liquid_cg` (N×3, m):** the contents' CG in the tank frame.
  - **`specific_force`:** the accelerometer reading.
  - **`lateral_disp`, `axial_disp`:** the oscillator displacements.
- **Damping ratio ζ:** a per-mode input. The supplied table uses a placeholder 0.02; see section 5.3.

---

## 5. Level A: running load cases

### 5.1 Prescribed tank motion (for example a measured or CFD acceleration)
```python
import numpy as np
import tankslosh as ts

table = ts.EMMTable.load_json("examples/output/baseline_emm.json")
p = table.at(0.30)                      # parameters at 30 % fill (interpolated)
print(p.summary())

t = ts.time_grid(10.0, 1e-3)            # 10 s at 1 kHz
ax = ts.trapezoid_pulse(t, start=0.5, duration=1.0, amplitude=2e-3, rise=0.05)   # 2 mm/s^2
res = ts.simulate_prescribed(p, t, ts.vector_input(t, x=ax),
                             ref_point=ts.BASELINE_MOUNT_POINT)

print("peak Fx  [N]  :", np.abs(res.interface_force[:, 0]).max())
print("peak My  [N m]:", np.abs(res.interface_moment[:, 1]).max())
print("peak CG x [m] :", np.abs(res.liquid_cg[:, 0]).max())
res.to_csv("my_case.csv")               # columns: t, f_x..z, F_x..z, M_x..z, cg_x..z
```

**Using your own acceleration history** (for example from a CSV with columns t, ax, ay, az): resample it to a uniform grid first.
```python
import numpy as np
import tankslosh as ts
data = np.column_stack([np.linspace(0, 5, 501), 1e-3 * np.sin(np.linspace(0, 30, 501)),
                        np.zeros(501), np.zeros(501)])          # stand-in for np.loadtxt("accel.csv", ...)
t = ts.time_grid(data[-1, 0], 1e-3)
f = np.column_stack([np.interp(t, data[:, 0], data[:, k]) for k in (1, 2, 3)])
p = ts.EMMTable.load_json("examples/output/baseline_emm.json").at(0.55)
res = ts.simulate_prescribed(p, t, f, ref_point=ts.BASELINE_MOUNT_POINT)
```

### 5.2 Thrust on a spacecraft (the accelerometer signal is computed)
```python
import tankslosh as ts
p = ts.EMMTable.load_json("examples/output/baseline_emm.json").at(0.75)
t = ts.time_grid(20.0, 1e-3)
Fx = ts.trapezoid_pulse(t, 1.0, 2.0, 5.0, rise=0.05)      # 5 N for 2 s
res = ts.simulate_coupled(p, t, ts.vector_input(t, x=Fx), dry_mass=50.0,
                          ref_point=ts.BASELINE_MOUNT_POINT)
accel_x = res.specific_force[:, 0]                          # accelerometer reading, m/s^2
```
`dry_mass` is everything except the liquid and diaphragm. The structure is treated as rigid.

### 5.3 Changing the damping
```python
import copy
import tankslosh as ts
p = ts.EMMTable.load_json("examples/output/baseline_emm.json").at(0.55)
q = copy.deepcopy(p)
for m in q.lateral_modes + q.axial_modes:
    m.zeta = 0.04
```
**What damping affects:**
- Damping hardly changes peak loads from a single pulse.
- It controls how long the slosh rings afterwards, and how strongly it amplifies steady shaking near a slosh frequency.
- See `examples/damping_sensitivity.py`.

### 5.4 Looking at the parameters
- `p.lateral_modes` and `p.axial_modes` are lists of `SloshMode(mass, freq_hz, zeta, height)`.
- `p.fixed_mass` and `p.fixed_height` describe the rigid part.
- `p.liquid_mass` is the total contents mass.
- Tables save and load as JSON with `save_json` / `load_json`.

---

## 6. Level B: making parameters for a changed design

### 6.1 Same tank, different diaphragm properties
```python
import dataclasses
import tankslosh as ts
from tankslosh.diaphragm_static import DiaphragmStatics
from tankslosh.modal import emm_table

d = dataclasses.replace(ts.BASELINE_DIAPHRAGM,
                        shear_modulus=2.2e6,     # Pa (neo-Hookean mu = 2*C10)
                        thickness=3.5e-3,        # m
                        density=1200.0)          # kg/m^3
ds = DiaphragmStatics(d, n_segments=100)
table, lateral, axial, statics = emm_table(ds, fills=[0.3, 0.55, 0.8], zeta=0.02,
                                           sweep_kwargs=dict(max_step=0.01))
table.save_json("my_table.json")
for lm in lateral:
    i = lm.dominant(1)[0]
    print(f"{lm.fill:.0%}: f1 = {lm.freq_hz[i]:.2f} Hz, slosh mass = {lm.slosh_mass[i]:.2f} kg")
```
- **Fill list:** use `fills=np.round(np.arange(0.05, 0.951, 0.05), 3)` for a full table. That takes about 1.5 min for a plain cup and about 17 min for the ribbed top hat. `EMMTable.at` interpolates between the fills you computed.
- **Stiffness only:** if just μ changes, every frequency scales exactly with √μ. Masses and heights do not change.
- **Accuracy at high fill:** for final numbers at 80–90 % fill, use `n_segments=200`. It is about 4× slower, and the 100-segment results read up to 8 % low there.

### 6.2 Top hat with different dimensions
```python
import tankslosh as ts
hat = ts.top_hat_diaphragm(ts.BASELINE_TANK, stress_free_fill=0.95, brim_z=0.2, n_segments=150,
                           thickness=2e-3, density=1150.0, shear_modulus=1.8e6,
                           ribs=ts.RibSpec(count=36, height=2e-3, width=3e-3))
print(f"gap to wall {hat.gap*1e3:.2f} mm, mass {hat.mass:.3f} kg")
```
- **Gap:** chosen automatically so the gas layer is `1 − stress_free_fill` of the tank when the hat is undeformed.
- **Ribs:** `ribs=None` gives a plain sheet.
- **Folding:** folded layers are kept at least one thickness apart (self-contact, on by default in `DiaphragmStatics`).
- **Sweep steps:** use `ds.sweep(fills, max_step=0.01)` for rolling diaphragms. Smaller continuation steps let the fold develop smoothly.

### 6.2b Different tank size or cup
```python
import tankslosh as ts
tank = ts.EllipticalHeadTank(radius=0.25, height=0.8, head_ratio=2.0)     # head depth = radius / head_ratio
cup = ts.MoldedCupDiaphragm(tank=tank, clamp_z=0.65, stress_free_fill=0.5,
                            thickness=3e-3, shear_modulus=1.8e6, density=1150.0)
print(cup.dome_equator_z, cup.apex_z, cup.mass)
```
**What these define:**
- **The cup:** a straight sidewall on the wall, below the clamp, followed by an elliptical dome with the same shape as the heads. Its depth is set so that it holds `stress_free_fill`.
- **The clamp:** must sit on the straight section.
- **Next step:** pass `cup` to `DiaphragmStatics` as in 6.1, and pass `ref_point=(R, 0, z_mount)` to the Level A runs.

### 6.3 Any axisymmetric diaphragm shape
Give the molded (stress-free) mid-surface as points from the clamp, on the tank wall, to the axis (r = 0). About 100 evenly spaced points work well.
```python
import numpy as np
import tankslosh as ts
from tankslosh.diaphragm_static import DiaphragmStatics
from tankslosh.shell import AxisymmetricShell

tank = ts.BASELINE_TANK
s = np.linspace(0.0, 1.0, 101)
r0 = 0.2 * (1 - s)                       # example: a shallow cone from the clamp to the axis
z0 = 0.30 - 0.05 * s
r0[-1] = 0.0
# fill at which the liquid exactly fills the space below the molded shape
V_liq = tank.volume_below(z0[0]) - AxisymmetricShell.enclosed_volume(r0, z0)
fill0 = V_liq / tank.volume
d = ts.MeridianDiaphragm(tank=tank, r0=r0, z0=z0, stress_free_fill=fill0, thickness=3e-3,
                         density=1150.0, shear_modulus=1.8e6)
ds = DiaphragmStatics(d)
print(f"stress-free at {fill0:.1%}; clamp at z = {d.clamp_z} m")
sols = ds.sweep([0.3, fill0 + 0.1])
```
- **`stress_free_fill`:** must be the fill at which the liquid exactly fills the space below the molded shape, computed as above. Otherwise the solver starts from a stretched state.
- **Spherical tank:** use `EllipticalHeadTank(radius=b, height=2*b, head_ratio=1.0)`.
- **Hemispherical diaphragm:** `ts.hemisphere_diaphragm(tank, 100, thickness=..., density=..., ...)` builds one molded to the upper head, stress-free when full.

### 6.4 Material model and bending
- **Default: neo-Hookean rubber** (`material="neo-hookean"`, needs `shear_modulus`). It is valid at large stretch.
- **Linear-elastic, small-strain** (`material="linear"`, `youngs_modulus=E`, `poisson=nu`). Use it only to match references that use it.
- **`bending=False`:** treats the diaphragm as a pure membrane. Avoid it for real designs: a slack membrane can then fold at no cost and the results stop being unique.

### 6.5 Gravity (1-g ground tests, 1-g CFD)
```python
import tankslosh as ts
from tankslosh.diaphragm_static import DiaphragmStatics
from tankslosh.modal import lateral_modes
ds = DiaphragmStatics(ts.BASELINE_DIAPHRAGM, gravity=(9.81, ts.WATER_20C.density))
sol = ds.solve(0.5)
lm = lateral_modes(ds, sol, fluid=ts.WATER_20C)
```
- The modal functions check that `fluid.density` matches the density given for gravity.
- In Level A runs of a 1-g case, include `z=9.81` in the specific force.

### 6.6 Looking inside a single fill
```python
import tankslosh as ts
from tankslosh.diaphragm_static import DiaphragmStatics
from tankslosh.modal import lateral_modes, axial_modes
ds = DiaphragmStatics(ts.BASELINE_DIAPHRAGM)
sol = ds.sweep([0.85], max_step=0.01)[0]   # sweep() walks there from the stress-free fill; robust for any fill
print("dp = p_liquid - p_gas [Pa]:", sol.dp, " max stretch:", sol.max_stretch)
lm, am = lateral_modes(ds, sol), axial_modes(ds, sol)
print("lateral f [Hz]:", lm.freq_hz[:4], " slosh mass [kg]:", lm.slosh_mass[:4])
print("axial   f [Hz]:", am.freq_hz[:3], " stable:", lm.stable and am.stable)
```
**What the static solution contains:**
- `sol.r` and `sol.z` give the diaphragm shape.
- `sol.contact_nodes` lists the nodes touching the wall.
- `sol.stretch_meridional` and `sol.stretch_hoop` give the stretches.

**Solving single fills:**
- Use `ds.solve(fill)` only close to the stress-free fill.
- For any other fill use `ds.sweep([...])`. It steps there gradually, and the diaphragm's large deformations need that.

### 6.7 Free-surface reference (no diaphragm, under gravity)
```python
import tankslosh as ts
p = ts.cylinder_free_surface(radius=0.2, fill_height=0.3, accel=9.81, n_modes=3)
```
This gives the classical flat-bottom cylinder values. Use it to check a CFD set-up on a 1-g sloshing case.

---

## 7. Setting up a new customer case

Each customer tank and diaphragm is a **case**: a folder `cases/<name>/` holding one input file, `case.toml`, and the results in `cases/<name>/output/`. All cases are run by the same code, so a fix or improvement to the model reaches every case when you rerun it. **Do not make a git branch per case**; add a case folder on the main line instead.

Cases supplied:

| Case | Diaphragm | Run time |
|---|---|---|
| `cases/top-hat-95pct-2mm` | Current baseline: ribbed top hat, brim at z = 0.2 m, 2 mm web, stress-free at 95 %; orbit and 1 g | about 25 min |
| `cases/cup-50pct-3mm` | Previous baseline: plain 3 mm cup, stress-free at 50 %; orbit | about 2 min |

### 7.1 Steps
1. **Copy the closest case** and give the folder a short descriptive name, for example `cases/top-hat-90pct-2p5mm`:
   ```bash
   cp -r cases/top-hat-95pct-2mm cases/top-hat-90pct-2p5mm
   ```
   Delete the copied `output/` folder.
2. **Edit `case.toml`.** Change `name`, `description`, the dimensions and the material. List every input that is a guess under `estimated`; the list is printed in the results.
3. **Try a few fills first** to check the set-up (writes to a scratch folder, not the case's `output/`):
   ```bash
   python -m tankslosh.run_case cases/top-hat-90pct-2p5mm --fills 0.3,0.6,0.9 --out scratch/check
   ```
4. **Run the full case:**
   ```bash
   python -m tankslosh.run_case cases/top-hat-90pct-2p5mm
   ```
   Long runs can be left in a separate terminal. `--no-ground` skips the 1-g modes.
5. **Read `output/results.md`**, look at the plots, and commit the case folder (input and output together).
6. After changing only `description`, `estimated` or `notes`, refresh the report without rerunning:
   ```bash
   python -m tankslosh.run_case cases/top-hat-90pct-2p5mm --report-only
   ```

### 7.2 What the runner writes (`cases/<name>/output/`)

| File | Contents |
|---|---|
| `results.md` | Inputs, estimated inputs, statics summary, mode tables per fill, unstable fills, code version |
| `statics.csv`, `statics.png`, `shapes.npz` | Orbit static shapes, pressure difference, stretch, wall contact |
| `orbit_emm.json` | Level A parameter table for orbit; load with `ts.EMMTable.load_json` |
| `orbit_modes.csv` | Dominant lateral (f, slosh-mass fraction, height) and axial modes per fill, stability flags |
| `orbit_lateral.png`, `orbit_axial.png`, `orbit_pulse_response.png` | Modes against fill; Level A response to 1 mm/s² lateral and axial pulses |
| `ground_statics.csv`, `ground_emm.json`, `ground_modes.csv`, `ground_modes.png` | The same at 1 g, gas on top (only when `ground_test = true`). Level A runs with this table need specific force z = +g |

### 7.3 `case.toml` reference
Lengths in m, pressures in Pa, densities in kg/m³. Keys with a default may be left out.

| Key | Meaning | Default |
|---|---|---|
| `name`, `description` | Case name and a few lines of description for `results.md` | folder name, empty |
| `estimated` | List of inputs that are estimates (free text) | empty |
| `notes` | Anything else to print in `results.md` | empty |
| **`[tank]`** `radius`, `height` | Inner radius; inner height apex to apex | required |
| `head_ratio` | Radius / head depth: 2 for 2:1 elliptical heads, 1 for hemispherical | 2.0 |
| `mount_point` | Reference point for moments in the Level A pulse runs, `[x, y, z]` | `[R, 0, H/2]` |
| **`[diaphragm]`** `type` | `top_hat`, `cup`, `meridian` (points file) or `hemisphere` (molded to the upper head, stress-free full) | required |
| `stress_free_fill` | Fill at which the molded shape is unstressed | required (not for `hemisphere`) |
| `brim_z`, `lip` | `top_hat`: clamp height on the wall; height of the flare from clamp to sidewall. The gap to the wall is solved from `stress_free_fill` | –, 0.01 |
| `clamp_z` | `cup`: clamp height on the wall (straight section) | – |
| `points_file` | `meridian`: CSV of `r, z` (m) of the molded mid-surface, clamp first, last point on the axis (r = 0); `#` lines are comments. See 6.3 for computing `stress_free_fill` | – |
| `thickness`, `density` | Web thickness and rubber density | required |
| `material` | `neo-hookean` (needs `shear_modulus`) or `linear` (needs `youngs_modulus`, optional `poisson`) | `neo-hookean` |
| `bending` | Include bending stiffness (leave on for real designs) | true |
| `segments` | Meridian mesh; ignored for `meridian` (the file's points are used) | 150 top hat, 100 others |
| **`[diaphragm.ribs]`** `count`, `height`, `width` | Radial ribs, smeared (omit the table for a plain sheet) | none |
| **`[liquid]`** `preset` | Liquid property set; any field (`density`, `kinematic_viscosity`, `surface_tension`, `bulk_modulus`) can be overridden | `water_20c` |
| **`[environment]`** `orbit` | Statics and modes without gravity | true |
| `ground_test`, `g` | Also compute modes at `g` with the tank axis vertical and gas on top | false, 9.81 |
| **`[analysis]`** `fills` | List, or `{ start, stop, step }` | 5–95 % in 5 % steps |
| `sweep_max_step`, `sweep_min_step` | Continuation steps in fill for the statics; rolling diaphragms need 0.01 / 0.001 | 0.025, 0.002 |
| `self_contact` | Keep folded layers one thickness apart | true |
| `liquid_mesh` | Potential-flow triangle size | 0.005 |
| `n_keep_lateral`, `n_keep_axial` | Modes kept per fill in the parameter table | 3, 2 |
| `pulse_fills` | Fills shown in `orbit_pulse_response.png` | 25, 55, 75 % |
| **`[damping]`** `zeta` | Damping ratio for every mode (a placeholder until fitted) | 0.02 |

### 7.4 Good practice
- **Change one thing per case** when comparing designs, and say what changed in `description`.
- **Check convergence once per new geometry:** rerun two or three fills with more `segments` (for example 150 → 200) using `--fills` and `--out`, and note the result in `notes`.
- **Keep model code out of cases.** If a new diaphragm needs a new feature (a new shape type, a new material), add it to `tankslosh/` with a test, so every case can use it. `tankslosh/cases.py` turns `case.toml` into library objects and holds no case numbers.
- **Unstable fills** (sideways buckling) are listed in `results.md` and left out of the parameter table.

---

## 8. How far to trust the results

- **Verification:** each step is checked against exact solutions in `tests/`, including inflated spheres, plates, membranes and free-surface sloshing. The documents in `docs/` list the checks and their accuracy.
- **Mesh accuracy:**
  - About ±2 % on frequencies below 60 % fill.
  - Up to 8 % low at 80–90 % fill with 100 segments; use 200 segments there.
  - Slosh masses are within about 1 %.
- **Assumed inputs:**
  - The rubber modulus (1.8 MPa, estimated from Shore 70A, ±30 %) shifts every frequency by its square root.
  - Density (1150 kg/m³) has a negligible effect.
  - Damping (0.02) is a placeholder.
- **Model limits:**
  - Small motions only (linear), and translation only.
  - Axisymmetric diaphragm shapes, so real non-axisymmetric wrinkling is not captured. This matters most in the 55–65 % "soft band" and the folded high-fill states.
  - Frictionless wall contact.
  - Incompressible liquid. Gas pressure does not affect slosh.
- **Ribbed top-hat baseline:**
  - The diaphragm is slack and rolls, so modes are about 0.5–7 Hz.
  - Several fills are flagged unstable to sideways buckling, and the model cannot follow that buckling.
  - See `docs/top_hat_baseline.md` for confidence by fill. The parameter table leaves out unstable modes, and each entry's `notes` says when its fill is unstable.

---

## 9. Troubleshooting

| Symptom | What to do |
|---|---|
| `ValueError: t must be uniformly spaced` | Resample your input onto `ts.time_grid(...)`. |
| `input must have shape (len(t), 3)` | Build inputs with `ts.vector_input(t, x=..., y=..., z=...)`. |
| `fill ... outside table range` | The table covers 5–95 %; regenerate it with the fills you need. |
| A static solution has `converged == False`, or fills are missing from `sweep` output | Reduce `max_step` in `sweep` (for example 0.01) or use more segments. A diaphragm that cannot reach a fill without crossing itself is reported rather than forced. |
| `liquid mesh lost boundary points` | Very thin liquid regions. Use a smaller `h_mesh` in `lateral_modes` / `axial_modes` (for example 0.003). |
| Very slow Level B runs | Run time grows quickly with `n_segments`: 100 for exploring, 200 only for final high-fill numbers. |
| `fluid density for gravity differs` | Pass the same liquid to `DiaphragmStatics(gravity=...)` and to `lateral_modes(fluid=...)`. |

---

## 10. Further reading

- `PROBLEM_STATEMENT.md`: the baseline problem and its assumptions.
- `docs/diaphragm_statics.md`: diaphragm statics, method, checks and results.
- `docs/slosh_modes.md`: lateral and axial modes, convergence and results.
- `docs/model_options.md`: gravity, linear material, custom shapes, and a comparison with a published paper.
- `cases/*/output/results.md`: results of each customer case.
- `README.md`: file layout.
