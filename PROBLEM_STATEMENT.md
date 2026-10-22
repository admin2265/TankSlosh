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

## Diaphragm (baseline since 2026-10-21: ribbed top hat)
Identified from a photograph of the actual part. Rib dimensions, clamp height and material constants are estimates.

| Item | Value |
|---|---|
| Type | Molded elastomer "top hat", off-white, with radial ribs. Clamped all round at its brim on the wall at **z = 0.2 m** (*estimated*: makes the hat about as tall as it is wide, as in the photo) |
| Molded (stress-free) shape | Short flare (10 mm) in from the clamp, then a straight sidewall at r = R − gap up to the head seam (z = 0.5 m), then a dome following the upper head offset inward by the gap, to the apex at z = 0.594 m |
| Gap to the wall | 6.37 mm, chosen so the gas between the hat and the upper wall is exactly 5 % of the tank volume when the hat is undeformed |
| Stress-free fill | **95 %**. Liquid inside the hat; N₂ in the thin layer between the hat and the top of the tank |
| Meridian length | 0.534 m |
| Web thickness | **2 mm** (measured) |
| Ribs | About 36 radial ribs on the outer (gas) side, assumed 2 mm high × 3 mm wide, running from the brim over the dome to the apex (*estimated from the photo*). Modelled as smeared meridional stiffening (membrane fibres plus a T-section in bending) that becomes a solid 4 mm section where the ribs merge near the apex |
| Material | Silica-filled EPDM class (off-white, tensile strength 1650 psi = 11.4 MPa). Neo-Hookean, shear modulus μ ≈ 1.8 MPa (*assumed*, about Shore 70A), nearly incompressible |
| Density | 1150 kg/m³ (*assumed*). Mass with ribs is about 1.34 kg |
| Wall contact | Frictionless sliding. Folded layers of the diaphragm are kept at least one wall thickness (2 mm) apart (self-contact) |
| Gas side | N₂, quasi-static. Its absolute pressure does not affect slosh, because the gas volume is fixed by the liquid volume; only the pressure difference across the diaphragm enters, and the model computes it. |

### What the diaphragm does across the fill range
- **95 % fill:** the stress-free shape.
- **Below 95 %:**
  - The crown descends, staying almost flat.
  - The top of the sidewall rolls inward into a U-shaped fold, which travels down the wall as the tank empties. This is a classic rolling diaphragm.
  - Stretches stay within a few percent, so the pressure difference across the diaphragm is small. Bending (web plus ribs) and the fold geometry control the stiffness.
- **Below about 52 % fill:** the two layers of the fold close onto each other. The liquid sliver between them is squeezed out, and the layers are held one thickness apart by self-contact.

### Previous baseline (kept for reference and as verification cases)
`CUP_3MM_DIAPHRAGM`: a plain 3 mm cup clamped at the upper head seam (z = 0.5 m), stress-free at 50 % fill. Its results are documented in `docs/diaphragm_statics.md` and `docs/slosh_modes.md`, and the original verification tests use it.

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
2. **Level B:** the diaphragm model. Static shapes ([docs/diaphragm_statics.md](docs/diaphragm_statics.md)) and lateral and axial slosh modes ([docs/slosh_modes.md](docs/slosh_modes.md)) are done; the parameter table is `examples/output/baseline_emm.json`.
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
