# Model options: gravity, linear-elastic membrane, any diaphragm shape

The baseline problem (orbit, neo-Hookean 3 mm cup) is unchanged by these options. They are off unless you ask for them.

## Gravity (1-g ground tests, 1-g CFD)
```python
ds = DiaphragmStatics(diaphragm, gravity=(9.81, fluid.density))
```
- Adds the hydrostatic liquid below the diaphragm and the diaphragm's own weight, with gravity acting along −z.
- In the statics this is the potential ρ_L·g·∫z dV of the liquid plus Σ m·g·z of the rubber.
- In both mode types it adds the matching hydrostatic stiffness. In the lateral modes that is the second variation of the liquid's first moment, integrated exactly around the circumference.
- `lateral_modes` / `axial_modes` refuse to run if the `fluid` passed to them has a different density from the one given for gravity.
- `StaticSolution.dp` is still p_liquid(z = 0) − p_gas. With gravity, the pressure difference at height z is `dp − ρ g z`.
- The spring-mass parameters keep the Level A convention. Level A's moment includes the weight-shift term m_n·a_z·x_n, so pass a_z = +g as the specific force for a 1-g test (see `simulate.py`).

## Linear-elastic membrane, bending switch
```python
AxisymmetricShell(r0, z0, None, t, nu=0.3, material="linear", youngs_modulus=E, bending=False)
```
- **`material="linear"`:** small-strain membrane energy E t /(2(1−ν²)) · (e₁² + e₂² + 2ν e₁e₂) per molded area, with e = stretch − 1. It is valid only for small strains, so prefer neo-Hookean beyond a few percent.
- **`bending=False`:** a pure membrane. Use it only to match membrane-only references. Without bending, a slack membrane can fold at no cost, so the statics lose uniqueness.

## Diaphragm given by its molded shape
```python
d = MeridianDiaphragm(tank=tank, r0=r_points, z0=z_points, stress_free_fill=..., thickness=...,
                      density=..., shear_modulus=... | youngs_modulus=..., poisson=..., material=..., bending=...)
d = hemisphere_diaphragm(tank, n_segments, thickness=..., ...)   # molded to the upper head, stress-free when full
```
- **Points:** run from the clamp (on the wall) to the axis (r = 0). The points are used as the mesh, so space them evenly and finely, about 100 segments.
- **Spherical tank:** `EllipticalHeadTank(radius=b, height=2*b, head_ratio=1.0)`.

## Verification (tests/test_options.py)
| Check | Result |
|---|---|
| Linear membrane, uniformly inflated sphere vs exact | within 1e-5 |
| Energy gradients with gravity, both materials, vs finite differences | within 1e-6 |
| Liquid first moment of a hemispherical cap vs exact | within 1e-5 |
| Lateral-mode (3D) energy and first moment vs axisymmetric, with gravity | equal to roundoff |
| Slack flat membrane over a half-full sphere at 1 g vs free-surface slosh pinned at the wall | within 0.003 % at 60 and 120 segments |
| Hemispherical diaphragm: stress-free at 100 %, volume and contact at 90 % | exact |

The slack-membrane check is the important one. With no rubber stiffness, the diaphragm model must reduce exactly to gravity sloshing with the contact line pinned at the clamp, and it does.

## Fixes found while building these checks (affect all runs)
1. **Wall-contact rule in the slosh modes.**
   - The old rule held to the wall every node within 2 mm of it, in the run from the clamp, even where the diaphragm leaves the wall steeply. Next to the clamp that pinned one or two extra nodes, depending on the mesh.
   - A node now counts as resting on the wall only if it touches the wall, or if it is within 2 mm and the diaphragm runs along the wall there (within 20°).
   - **Effect on the baseline tank:** only 90 % fill changed. Lateral f₁ went from 6.10 to 5.44 Hz, m₁ from 17.5 % to 16.8 %, and axial f₁ from 7.20 to 7.02 Hz. Other fills changed by less than 0.1 %.
2. **Bending normal at the clamp node.** The lateral-mode model used a slightly different normal there from the axisymmetric model. It now matches exactly. There is no measurable effect on results (< 0.0001 %).
3. **Element stiffness by finite differences** is now Richardson-extrapolated over two step sizes. This is a safeguard for slack membranes; it causes no measurable change in the baseline.

## Comparison with Utsumi (2022), *Vibration analysis of liquid in an axisymmetric tank covered by a diaphragm*
The case is a spherical tank, b = 0.365 m, liquid density 1444 kg/m³, at 1 g. The diaphragm is molded to the upper hemisphere, 1.2 mm thick, E = 6.2 MPa, ν = 0.3.

- **Free surface, no diaphragm (paper Fig. 6, matching NASA SP-8009):**
  - Frequency, slosh mass and heights agree within 0–2 % up to 65 % fill.
  - They agree within 2–5 % at 79–90 %, where the published curves are steep and hard to read.
  - The fixed-mass height comes out at 0.99 b; the exact value is b.
- **Diaphragm:** the results differ because of the modelling assumptions, not because of an identified error.
  - The paper assumes a smooth membrane that cannot wrinkle.
  - Our model finds that this very thin, soft diaphragm lies almost flat on the liquid, with its excess material in axisymmetric ripples. Its pressure difference is about 0, against the paper's ±4.5–6.5 kPa.
  - When restricted to smooth shapes, our model reproduces the paper's diaphragm shapes: the centre height is within about 0.01–0.03 m.
  - The pressure difference still differs. With the paper's stated E·h, its pressures would need about 8 % stretch, which its shapes do not show.
- **Lesson for slack diaphragms:** results depend on how wrinkling is treated, and an axisymmetric model can only represent it as rings. Real wrinkles run around the circumference. For the baseline tank, this applies to the soft band (55–65 % fill) and the folded high-fill states.
