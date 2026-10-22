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
1. Level B makes a parameter table, `baseline_emm.json`.
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

To change the tank or the diaphragm, edit the numbers in `tankslosh/geometry.py` (`BASELINE_TANK`, `BASELINE_DIAPHRAGM`), or build your own objects as in section 6. Then rerun `examples/slosh_modes.py`.

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

## 7. How far to trust the results

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

## 8. Troubleshooting

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

## 9. Further reading

- `PROBLEM_STATEMENT.md`: the baseline problem and its assumptions.
- `docs/diaphragm_statics.md`: diaphragm statics, method, checks and results.
- `docs/slosh_modes.md`: lateral and axial modes, convergence and results.
- `docs/model_options.md`: gravity, linear material, custom shapes, and a comparison with a published paper.
- `README.md`: file layout.
