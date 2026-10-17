# Lateral slosh modes (Level B, step 2)

These are the small sideways oscillations of the diaphragm and liquid about each static shape from [step 1](diaphragm_statics.md), converted into the spring-mass parameters (Level A) used by `tankslosh.simulate`.

Code:
- `tankslosh/harmonic_shell.py`: shell stiffness and mass for m = 1.
- `tankslosh/liquid_fe.py`: liquid potential flow.
- `tankslosh/modal.py`: coupled modes and the conversion to Level A parameters.
- `examples/slosh_modes.py`: sweep and plots.

Run it with:
```bash
py -3.12 examples/slosh_modes.py
```
It takes about 1.5 min and writes `examples/output/baseline_lateral_emm.json`, which can be loaded directly with `EMMTable.load_json`.

## Model
- **Displacement pattern.** Sideways tank acceleration excites only the cos θ (m = 1) pattern. Each meridian node carries (u_r cos θ, u_θ sin θ, u_z cos θ).
- **Diaphragm stiffness.** The 3D neo-Hookean membrane and bending energy of the perturbed surface is integrated exactly around the circumference. It reproduces the step-1 axisymmetric energy to machine precision. The stiffness is K = ∂²U − λ∂²V. The second term is the pressure acting on the deformed surface, with λ = dU/dV from the static solution; it is exact because the liquid is incompressible. Element stiffnesses come from central finite differences of the local energies.
- **Liquid.** Potential flow on a triangle mesh of the liquid region (about 4000 nodes). The region is bounded by the wall, the wetted diaphragm and the axis. Solving it gives the added-mass matrix ρπ GᵀA⁻¹G. The first-moment vector gives the CG shift and the force; the angular-momentum vector gives the moment.
- **Diaphragm inertia.** The rubber's own inertia (1.2 kg) is included. In the parameter table, the "liquid" mass is the contents: water plus diaphragm.
- **Boundary conditions:**
  - The clamp is fixed.
  - At the axis, the displacement must be a single translation.
  - Wall contact is frictionless and stays closed: nodes may slide along the wall but not lift off.
  - Sidewall parts within 2 mm of the wall, in the run from the clamp, are also held to the wall. A sealed sliver that thin cannot hold liquid, and an open one acts as a squeeze film. The same rule applies to a dome centre resting on the bottom (5 % fill).
- **Converting modes to spring-mass parameters.** Each mode n has frequency ω_n, slosh mass m_n = (φ_nᵀΓ)² and height h_n = φ_nᵀH / φ_nᵀΓ. The fixed mass is m₀ = m − Σm_n, at a height h₀ that keeps the CG of the contents correct. This mapping onto `simulate.py` is exact. The three lowest-frequency modes that each carry at least 0.2 % of the contents are kept.

## Verification (tests/test_modal.py)
| Check | Result |
|---|---|
| 3D energy and volume of the unperturbed state vs axisymmetric | equal to 1e-8 / 1e-15 |
| Rigid sideways translation of the whole diaphragm | zero energy, zero stiffness |
| Pre-tensioned flat membrane, f = j₁ₙ√(T/ρ_A)/(2πa), first 3 modes | within 0.07 % (80 segments) |
| Clamped circular plate, one nodal diameter (λ² = 21.26, 60.82) | within 0.07 % (80 segments) |
| Free-surface slosh in a flat-bottom cylinder vs classical theory: f₁, m₁, h₁ | within 0.007 %, 0.01 %, 0.01 % (4 mm mesh) |
| Contents CG preserved by the spring-mass parameters | exact |

The free-surface check also independently confirms the spring-mass height convention used in Level A.

**Mesh convergence, first mode** (diaphragm segments 50 / 100 / 150):

| Fill | 50 segments | 100 segments | 150 segments |
|---|---|---|---|
| 10 % | 6.08 Hz | 6.08 Hz | 6.18 Hz |
| 55 % | 1.10 Hz | 1.12 Hz | 1.12 Hz |
| 90 % | 4.61 Hz | 6.10 Hz | 5.91 Hz |

Slosh mass agrees within 1 % in every case. The liquid mesh size changes results by less than 0.05 %. Expect about ±2 % on frequency below 60 % fill and about ±5 % at high fill, where the tight rolling fold needs fine resolution.

## Results (μ = 1.8 MPa assumed; ζ is a placeholder 0.02)
| Fill | f₁ [Hz] | m₁ [% contents] | h₁ [m] | Next significant modes |
|---|---|---|---|---|
| 5 % | 4.34 | 34 | 0.05 | 21.3 Hz (16 %) |
| 10 % | 6.08 | 50 | 0.02 | 26.3 Hz (4 %) |
| 20 % | 8.78 | 47 | 0.01 | |
| 30 % | 9.88 | 38 | 0.02 | 44.6 Hz (2 %) |
| 40 % | 9.35 | 30 | 0.04 | |
| 50 % | 6.53 | 23 | 0.08 | 9.5 Hz (1 %) |
| 55 % | **1.12** | 20 | 0.14 | 13.0 Hz (5 %) |
| 60 % | **1.73** | 25 | 0.17 | |
| 65 % | **2.99** | 26 | 0.20 | |
| 70 % | 5.71 | 26 | 0.23 | 11.5 Hz (3 %) |
| 80 % | 5.33 | 22 | 0.28 | 10.4 Hz (5 %) |
| 90 % | 6.10 | 18 | 0.33 | 9.1 Hz (10 %) |
| 95 % | 6.09 | 18 | 0.36 | 8.7 Hz (12 %) |

The full table is in `examples/output/slosh_modes.csv`.

## Findings
1. **Below 50 % fill the diaphragm is taut.** The first mode is 4–10 Hz and carries 25–50 % of the contents. It is a near-bottom mode: the slosh height is close to the tank bottom.
2. **There is a soft band just above the stress-free fill (55–65 %).** The dome is passing through flat as it inverts, and only bending holds it, so f₁ drops to 1.1–3 Hz. A lateral pulse there moves the CG about 30 times more than at other fills (`slosh_pulse_response.png`). It is also where the response is most likely to be nonlinear (snap-through), so the linear model is least reliable there.
3. **Above 70 % fill** the inverted dome and fold give about 5–6 Hz, with a growing second mode near 9–11 Hz.
4. **No buckling was predicted.** Every static state is stable to m = 1 perturbations: all eigenvalues are positive. This is within the axisymmetric-statics assumption.
5. **All slosh frequencies (1–80 Hz) are far below the liquid's acoustic modes** (about 1 kHz), so treating the liquid as incompressible is consistent.

## Limitations
- **Material, contact and symmetry.** Everything in [step 1](diaphragm_statics.md#limitations-and-risks) carries over: the neo-Hookean model at large stretch, frictionless contact, and axisymmetric static shapes.
- **Linear only.** Large-amplitude slosh, contact opening and closing, and snap-through in the soft band are not captured.
- **Placeholder damping.** ζ is a placeholder, 0.02 for all modes. Fit it to StarCCM+ or test data.
- **Lateral only.** Axial (m = 0) modes are not yet computed, so the axial force is that of a rigid contents mass.
- **No residual acceleration.** There is no hydrostatic stiffness from residual acceleration. This is negligible at about 1e-3 m/s² or less, but it is not included.
- **Minor-mode heights.** Modes with very small slosh mass can have heights outside the tank, which is normal for spring-mass models. The heights of the minor modes (2 and 3) jump between fills because mode identities swap.

## Next
- Axial (m = 0) modes, using the step-1 axisymmetric energy with the volume constraint.
- A StarCCM+ comparison case at 2–3 fills. Suggested fills: 30 % (taut), 55 % (soft band) and 80 % (inverted). Fit ζ there.
