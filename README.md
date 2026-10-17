# tankslosh: fast slosh model for diaphragm tanks

A spring-mass slosh model of the liquid in a tank under translational acceleration (Level A). It is a fast companion and sanity check for full CFD (StarCCM+).

The liquid (mass *m*) is split into:
- a **fixed mass** m0 at height h0 that moves rigidly with the tank,
- **lateral oscillators** (m_n, f_n, ζ_n, h_n), the same set used for x and y,
- optional **axial oscillators** (m_n, f_n, ζ_n) along the tank axis z.

The bookkeeping requires `m0 + Σ m_n(lateral) = m`.

## Layout
| File | Contents |
|---|---|
| `tankslosh/emm.py` | `SloshMode`, `EMMParams` (one fill level), `EMMTable` (values vs. fill fraction, interpolated, saved and loaded as JSON) |
| `tankslosh/freesurface.py` | Analytic free-surface cylinder values (Abramson SP-106 / Dodge 2000), Bond number, viscous damping estimate |
| `tankslosh/simulate.py` | `simulate_prescribed` (tank motion given) and `simulate_coupled` (thruster force on the dry structure) |
| `tankslosh/inputs.py` | Time grid, trapezoid pulse, stacking per-axis inputs into (N, 3) |
| `examples/demo.py` | 1-g reference case and orbit case; writes plots and CSVs to `examples/output/` |
| `examples/diaphragm_emm_template.json` | **Placeholder** diaphragm-tank values (format example only) |
| `tests/` | Verification tests (stdlib `unittest`) |

## Running
```bash
py -3.12 -m unittest discover -s tests -t .
```
```bash
py -3.12 examples/demo.py
```

## Conventions
- Tank frame: x, y lateral, z along the tank axis. SI units.
- **Specific force** `f` is the tank acceleration minus gravity, i.e. what a tank-mounted accelerometer reads. In orbit it equals the tank acceleration. On the ground at rest it is (0, 0, +g).
- `interface_force` is the force the liquid exerts on the tank. `interface_moment` is taken about (0, 0, `z_ref`).
- `liquid_cg` is the liquid CG in the tank frame, accurate to first order in slosh amplitude.
- Oscillator equation: `x_n'' + 2ζ_n ω_n x_n' + ω_n² x_n = −f`.
- Coupled run: `f = (F_ext + Σ(k_n x_n + c_n x_n')) / (M_dry + m0)`.
- Integration is exact for inputs that vary linearly between samples (first-order hold). The time step only needs to resolve the input and the outputs you want. 20 s at 1 kHz takes about 0.1 s.
- `initial="static"` starts the oscillators in equilibrium under the first input sample. `"rest"` starts them undeflected.

## Validity
- The free-surface values hold only for **gravity-dominated** sloshing (Bond number ≳ 100). A warning is issued below that. They are a check case and a 1-g reference for a StarCCM+ VOF set-up, **not** diaphragm-tank values.
- Values for the diaphragm tank must come from the Level B model (diaphragm + potential flow) or be identified from CFD or test. Store them as an `EMMTable` JSON over fill fraction.
- The model is linear (small amplitude) and translation only. No tank rotation.

## Verified by the tests
- First root of J1′. Bessel-series identities for shallow-tank mass and static CG.
- Mass and CG bookkeeping. Frequency formula. Deep-tank slosh heights.
- First-order-hold result equals the analytic step response to 1e-9. Static deflection and force. Free-decay frequency.
- Interface force equals the liquid's momentum rate (−m(f + ẍ_cg)).
- The exact tilted-surface static CG shift and base moment are recovered: `x_cg = −aR²/(4gh)`, `M = −ma(h/2 + R²/4h)`.
- Momentum conservation in the coupled spacecraft run (lateral and axial).

## Next steps
1. Level B: compute the diaphragm's static shape (neo-Hookean, volume-constrained) and its coupled m=0/m=1 modes with potential flow, giving `EMMTable` values over 10–90 % fill.
2. CFD comparison tool: import StarCCM+ force and CG histories, filter out acoustic content, fit f_n and ζ_n.
