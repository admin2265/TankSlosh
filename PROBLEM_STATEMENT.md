# Problem statement (baseline)

Status: locked baseline, 2026-10-17. Values marked *assumed* are placeholders until confirmed.
The same numbers are in code in `tankslosh/geometry.py` (`BASELINE_TANK`, `BASELINE_DIAPHRAGM`).

## Goal
Predict the slosh of water in a diaphragm tank under small translational accelerations in orbit:
- the motion of the liquid's CG,
- the reaction force and moment the liquid applies to the tank at the side mount where the accelerometer is located.

The model must run far faster than CFD and serve as a sanity check on StarCCM+ results.

## Tank
| Item | Value |
|---|---|
| Shape | Vertical cylinder with 2:1 elliptical heads, top and bottom |
| Inner diameter D | 0.4 m (R = 0.2 m) |
| Overall inner height H | 0.6 m |
| Head depth | D/4 = 0.1 m each |
| Straight section | z = 0.1 to 0.5 m (0.4 m long) |
| Total volume | 67.02 L (lower head 8.38 L, cylinder 50.27 L, upper head 8.38 L) |
| Wall | Rigid |

**Frame:** z runs along the tank axis, with z = 0 at the inside apex of the lower head. The gas side is at the top. x and y are lateral.

## Liquid
- Water at 20 °C: ρ = 998.2 kg/m³, ν = 1.004e-6 m²/s.
- Treated as incompressible for slosh. The acoustic modes are around 1 kHz and are handled separately if they are ever needed.
- Fill range: 5–95 % of the total volume, i.e. 3.35–63.67 kg. At 50 % the liquid is 33.45 kg.

## Diaphragm
| Item | Value |
|---|---|
| Type | Molded elastomer cup, clamped all round at the upper head seam (z = 0.5 m, r = R) |
| Molded (stress-free) shape | Straight sidewall at r = R from z = 0.5 down to 0.3667 m, then a 2:1 elliptical dome (same shape as the heads) down to the apex at z = 0.2667 m |
| Stress-free fill | 50 %. Liquid below the cup = 33.51 L |
| Meridian length | 0.1333 m sidewall + 0.2422 m dome = 0.3755 m |
| Thickness | 3 mm (molded) |
| Material | Shore 70A rubber, neo-Hookean, shear modulus μ ≈ 1.8 MPa (*assumed*, from Gent's Shore-to-modulus conversion; replace with datasheet value), nearly incompressible |
| Density | 1150 kg/m³ (*assumed*) |
| Wall contact | Frictionless sliding |
| Gas side | N₂, quasi-static. Its absolute pressure does not affect slosh, because the gas volume is fixed by the liquid volume; only the pressure difference across the diaphragm enters, and the model computes it. |

### What the diaphragm has to do across the fill range
- **5 % fill:** the dome nests about 27 mm above the lower head (dome equator at z ≈ 0.127 m). The meridian has to grow from 0.376 m to about 0.616 m. The average stretch is about 1.64. If only the sidewall stretches, the stretch is about 2.8. Shore 70A typically breaks at 250–400 % elongation, so this is within its limits.
- **95 % fill:** the dome has to **invert** into the upper head. The 0.133 m sidewall then has nowhere to go and forms a **rolling fold** (convolution) next to the wall. Expect snap-through of the dome and possibly non-axisymmetric folding. The diaphragm model must check for this (an m = 1 stability check).
- **Near 50 % fill:** the diaphragm is close to stress-free, so its stiffness comes from **bending** (the 3 mm shell) more than from membrane tension. The diaphragm model must therefore be a shell, not just a membrane.

## Loading
- Translational acceleration only, no tank rotation, small amplitude (linear slosh).
- Inputs: the specific force history f(t) (what the accelerometer reads), or thruster force on a rigid dry structure (coupled case).
- Residual (background) acceleration level and typical pulse profiles: *to be provided*.

## Outputs
| Output | Definition |
|---|---|
| Liquid CG | Position in the tank frame vs. time |
| Mount reaction (liquid only) | Force and moment the **liquid** applies to the tank, taken about the side mount at mid-height, P = (R, 0, 0.3) m by default. Excludes the tank shell's own inertia. |
| Accelerometer | Specific force at P. The tank is rigid and does not rotate, so this equals the tank's specific force. |

## Modelling approach
1. **Level A (done):** the runtime model, a fixed mass plus spring-mass oscillators, read from a table against fill fraction.
2. **Level B (next):** the diaphragm model.
   - Find the diaphragm's static shape at each fill level: an axisymmetric neo-Hookean shell with a volume constraint, frictionless wall contact and dome inversion.
   - Compute its small m = 0 and m = 1 vibration modes coupled to potential flow in the liquid.
   - Convert those modes into Level A oscillator parameters over the 5–95 % fill range.
3. **CFD comparison:** fit only the damping ratio ζ from StarCCM+ runs.

## Open items
- [ ] Datasheet modulus (or test curve) for the Shore 70A compound. Also decide whether a Mooney-Rivlin or Gent model is needed at stretches above 2.
- [ ] Rubber density.
- [ ] Residual acceleration level and design acceleration pulses.
- [ ] Azimuth of the side mount (x axis assumed).
- [ ] StarCCM+ case set-up details for the comparison runs.
