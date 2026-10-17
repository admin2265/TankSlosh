# Diaphragm statics (Level B, step 1)

Static shape of the baseline molded cup diaphragm over 5–95 % fill. This is the pre-stressed state that the slosh modes (step 2) are linearized about.

Code:
- `tankslosh/shell.py`: shell energy.
- `tankslosh/diaphragm_static.py`: solver and continuation.
- `examples/diaphragm_statics.py`: sweep, plots, CSV.

Run it with:
```bash
py -3.12 examples/diaphragm_statics.py
```
The sweep takes about 1.5 min and writes `examples/output/diaphragm_statics.{png,csv}` and `diaphragm_shapes.npz`.

## Model
- **Geometry:** an axisymmetric meridian from the clamp (z = 0.5 m, r = R) to the axis, with 100 segments. The molded shape is the stress-free reference (`PROBLEM_STATEMENT.md`).
- **Membrane:** incompressible neo-Hookean. The energy per reference area is (μt/2)(λ₁² + λ₂² + λ₁⁻²λ₂⁻² − 3), with μ = 1.8 MPa and t = 3 mm.
- **Bending:** Kirchhoff, D = μt³/3 (ν = ½). It acts on changes of meridional and hoop curvature from the molded shape and uses the molded thickness.
- **Boundary conditions:**
  - At the clamp, the diaphragm is fixed in both position and rotation.
  - At the axis, the meridian meets the axis with a horizontal tangent, by symmetry.
- **Liquid volume:** held exactly. The liquid and the gas each have a uniform pressure (no gravity in orbit). The pressure difference Δp = p_liquid − p_gas = dU/dV_liquid is the constraint multiplier.
- **Wall contact:** frictionless, against the cylinder and both elliptical heads.
- **Solver:** SLSQP with exact (hand-derived) gradients, continuing outward from 50 % fill with adaptive steps. A solution is accepted only if the volume error, contact penetration and equilibrium residual are all small, and the meridian does not cross itself.

## Verification (tests/test_shell_static.py)
| Check | Result |
|---|---|
| Energy and volume gradients vs finite differences | agree to 1e-6 relative |
| Uniformly inflated neo-Hookean sphere, Δp = (2μt/R₀)(λ⁻¹ − λ⁻⁷) | within 1e-5 |
| Clamped circular plate, small deflection, w₀ = pa⁴/64D | within 0.05 % (80 segments) |
| Stress-free at 50 % fill | energy < 1e-9 J, shape unchanged |
| Frictionless contact on the straight wall: uniform meridional tension | spread < 1e-8 |
| Δp equals dU/dV from neighbouring solutions | within 0.05 % |
| Mesh convergence (50 / 100 / 200 segments) | Δp converged to < 0.1 % for fill ≤ 50 %, ~1 % at 70 %; at 90 % Δp ≈ 0 ± 0.05 kPa |

## Results (μ = 1.8 MPa assumed)
| Fill | Δp [kPa] | Max stretch | Min thickness [mm] | Shape |
|---|---|---|---|---|
| 5 % | −53.9 | 2.43 | 0.51 | Dome rests on the lower head; sidewall lies along the wall |
| 10 % | −52.4 | 2.18 | 0.63 | |
| 20 % | −48.9 | 1.70 | 1.04 | |
| 30 % | −41.1 | 1.37 | 1.60 | |
| 40 % | −25.8 | 1.15 | 2.25 | |
| 50 % | 0 | 1.00 | 3.00 | Molded shape |
| 60 % | +0.18 | 1.02 | 2.98 | Dome inverting, with a dimple near the wall |
| 75 % | +0.59 | 1.04 | 2.98 | Inverted dome plus a rolling fold at the wall |
| 90 % | ≈ 0 | 1.04 | 3.00 | Inverted dome nests under the upper head |

The full table is in `examples/output/diaphragm_statics.csv`.

## Findings
1. **The two halves of the fill range behave very differently.**
   - Below 50 % the gas must push the diaphragm down by stretching it. Δp reaches about −54 kPa, the stored energy is up to about 1.1 kJ, and the diaphragm is taut.
   - Above 50 % the dome turns inside out, and the sidewall rolls into a fold beside the wall. The rubber barely stretches (≤ 4 %), so |Δp| < 1 kPa and the stored energy is a few joules. Only bending holds the shape.
   - For slosh this means the diaphragm should be much stiffer below 50 % fill than above it. Above 50 % the slosh frequency will be low and governed by bending.
2. **The peak strain is at the dome apex, not the sidewall.** The centre of the dome balloons equally in both directions, by 1.7× at 20 % fill and 2.4× at 5 %. The sidewall slides along the wall and stretches only about 1.35×. At 5 % the rubber at the apex thins to about 0.5 mm.
3. **The gas has to be at least about 54 kPa above the liquid** to reach 5 % fill. This sets a lower limit on the N₂ pressure, if the liquid outlet pressure must stay positive.

## Limitations and risks
- **Material model at large stretch.** Neo-Hookean rubber softens under equal two-way stretch (the balloon limit point) and does not stiffen as real rubber does near its limit. That makes it exaggerate the ballooning at the apex below about 15 % fill. A Gent or Mooney-Rivlin model fitted to test data (ideally an equibiaxial test) would reduce it. The 5–15 % results should be treated as indicative.
- **Friction.** The wall contact is frictionless. Real rubber–metal friction would hold the sidewall back and spread the strain differently.
- **Symmetry.** The model is axisymmetric. Above 50 % the hoop direction is compressed by up to 7 %, while the dome inverts and the fold forms. A real 3 mm sheet may wrinkle or fold unevenly around the circumference instead. The step-2 modal analysis will check this: a negative m = 1 stiffness would signal an asymmetric buckle.
- **Self-contact.** The fold is not modelled as contacting itself; the solver only rejects shapes where the meridian crosses itself. No crossings were found at 100 segments.
- **Bending thickness.** Bending uses the molded thickness. This matters only where bending matters, near and above 50 % fill, where the rubber thins by less than 1 %.

## Next: step 2, slosh modes
Linearize about each static state for m = 1 (lateral) and m = 0 (axial) perturbations of the shell, coupled to potential flow in the liquid (added-mass matrix). Then:
- solve the eigenproblem,
- convert the results into `EMMTable` entries over 5–95 % fill,
- check the m = 1 stiffness for asymmetric buckling above 50 %.
