# Baseline: ribbed top-hat diaphragm (results)

This is the current baseline (see `PROBLEM_STATEMENT.md`): a 2 mm EPDM-class top hat with about 36 radial ribs. The brim is clamped at z = 0.2 m, the hat lines the upper tank with a 6.4 mm gas gap, and it is stress-free at 95 % fill.

The shape and rib layout come from a photograph. **The clamp height, rib size and rubber modulus are estimates.**

How to reproduce the results:

| Command | Output | Run time |
|---|---|---|
| `py -3.12 -m tankslosh.run_case cases/top-hat-95pct-2mm` | **All of the below in one run** (case folder, see the [manual](USER_MANUAL.md#7-setting-up-a-new-customer-case)): `cases/top-hat-95pct-2mm/output/` | about 35 min |
| `py -3.12 examples/diaphragm_statics.py` | Shapes, pressure difference, stretch | about 17 min |
| `py -3.12 examples/slosh_modes.py` | Orbit parameter table: `examples/output/baseline_emm.json` | about 17 min |
| `py -3.12 examples/ground_test_predictions.py` | 1-g parameter table: `examples/output/ground_test_emm.json` and `ground_test_modes.csv` | about 18 min |

The previous baseline, the 3 mm cup stress-free at 50 % (`CUP_3MM_DIAPHRAGM`), and its results are kept in `docs/diaphragm_statics.md` and `docs/slosh_modes.md`.

## What was added to the model for this diaphragm
- **Ribs (`RibSpec`):**
  - Smeared meridional stiffening. Rib fibres add stretching stiffness along the meridian, and a web + rib T-section adds meridional bending stiffness. Ribs add mass.
  - Hoop stiffness stays the web's, rising to the solid section where the ribs merge near the apex.
  - Along the sidewall, meridional bending is 2.0× the plain web; at the apex it is 8×.
  - Checks: a fully ribbed plate reproduces the solid-plate frequency within 0.5 %. Gradients match finite differences, and the lateral and axisymmetric models give identical energies.
- **Top-hat geometry (`top_hat_diaphragm`):** the gap to the wall is solved so the gas layer is exactly 5 % of the tank volume at the stress-free shape. This is checked to 1e-9.
- **Self-contact (statics):** folded layers are kept at least one wall thickness (2 mm) apart. Without it, the rolling fold passes through itself below about 52 % fill.
- **Self-contact (slosh modes):** layers in contact move together normal to the contact, and the liquid sliver squeezed out between them is removed from the liquid region.
- **Unstable shapes:** modes with negative stiffness (shape unstable to sideways buckling) are reported and left out of the parameter table.

## Statics (orbit)
- **95 → about 55 % fill:** the crown descends almost flat, and the top of the sidewall rolls inward into a U-shaped fold. This is a classic rolling diaphragm.
- **Below about 55 % fill:**
  - The fold stays at z ≈ 0.3–0.42 m, and its two layers close onto each other (self-contact).
  - The crown keeps descending inside, past the brim, to z ≈ 0.014 m at 5 % fill.
  - The last liquid sits under the crown and in a roughly 1 cm annulus between the hat's sidewall and the tank wall.
- **Pressure difference across the diaphragm:** |Δp| < 0.3 kPa (0.04 psi) at every fill. The diaphragm hardly stretches (−5 % to +3 %), so expulsion takes almost no pressure difference, unlike the stretched 3 mm cup (−54 kPa at 5 % fill).
- **Strain energy:** a few joules at every fill. The stiffness comes from bending (web plus ribs) and the geometry of the fold.

## Slosh modes: orbit (`baseline_emm.json`)
| Fill | Main sideways modes (slosh mass) | Main axial mode | Stable to sideways buckling |
|---|---|---|---|
| 5 % | 0.47 Hz (29 %), 2.84 Hz (4 %) | 2.2 Hz (18 %) | yes |
| 10 % | 1.06 Hz (27 %), 4.20 Hz (17 %) | 3.4 Hz (16 %) | yes |
| 20 % | 4.33 Hz (41 %) | 5.6 Hz (7 %) | yes |
| 30 % | 4.1–4.5 Hz (about 38 % in two close modes) | 5.7 Hz (5 %) | yes |
| 40 % | 1.23 Hz (10 %), 4.21 Hz (25 %) | 5.3 Hz (3 %) | yes |
| 50 % | 4.47 Hz (23 %), 9.80 Hz (7 %) | 5.3 Hz (2 %) | **no** |
| 60 % | 2.59 Hz (26 %), 7.11 Hz (20 %) | 4.0 Hz (4 %) | yes |
| 70 % | 1.99 Hz (22 %), 6.09 Hz (37 %) | 5.4 Hz (22 %) | yes |
| 80 % | 1.27 Hz (19 %), 4.03 Hz (40 %) | 4.8 Hz (40 %) | yes |
| 90 % | 0.75 Hz (11 %), 3.15 Hz (50 %) | 3.9 Hz (42 %) | yes |
| 95 % | 2.62 Hz (62 %) | 4.4 Hz (37 %) | yes |

Fills flagged unstable: 45, 50, 55, 65 and 75 %. The full table is in `examples/output/slosh_modes.csv`.

## Slosh modes: 1 g, gas on top (shaker test, `ground_test_modes.csv`)
| Fill | Main sideways modes (slosh mass) | Main axial mode | Stable |
|---|---|---|---|
| 10 % | 1.00 Hz (31 %), 1.63 Hz (16 %) | 2.5 Hz (6 %) | yes |
| 20 % | 1.77 Hz (33 %), 10.5 Hz (10 %) | 7.9 Hz (3 %) | yes |
| 30 % | 1.86 Hz (22 %), 6.7 Hz (21 %) | 11.3 Hz (2 %) | yes |
| 40 % | 1.64 Hz (14 %), 4.70 Hz (18 %) | 8.3 Hz (1 %) | yes |
| 50 % | 1.91 Hz (14 %), 4.69 Hz (15 %), 10.3 Hz (15 %) | 8.6 Hz (2 %) | **no** |
| 60 % | 1.87 Hz (14 %), 5.14 Hz (28 %), 9.1 Hz (17 %) | 8.7 Hz (18 %) | yes |
| 70 % | 2.30 Hz (25 %), 5.99 Hz (34 %) | 6.0 Hz (23 %) | yes |
| 80 % | 1.86 Hz (24 %), 4.56 Hz (31 %) | 5.9 Hz (24 %) | yes |
| 90 % | 1.58 Hz (16 %), 4.25 Hz (32 %) | 6.0 Hz (23 %) | yes |
| 95 % | 1.65 Hz (9 %), 4.03 Hz (36 %) | 5.4 Hz (17 %) | yes |

Fills flagged unstable: 5, 15, 45, 50 and 75 %.

**What the 1-g modes mean:**
- At almost every fill there is a mode at about 1.6–1.9 Hz with 14–33 % of the contents. That is close to free-surface sloshing in this tank with the edge pinned. Under gravity the slack hat barely resists the liquid's own sloshing.
- A second, diaphragm-controlled mode sits at about 4–7 Hz with 15–36 %.
- At 1 g and in orbit the results differ a lot, so a ground test validates the 1-g model, and that model then predicts orbit.

## Accuracy and confidence
| Aspect | Assessment |
|---|---|
| Mesh (100 / 150 / 200 segments) | At 85 % fill, 100 and 200 segments agree to 0.3 % (for example 1.00 / 3.58 Hz in orbit, 1.70 / 4.38 Hz at 1 g). At 30 % the dominant mass agrees within a few percent, but how it splits between closely spaced modes, and the minor modes, vary with the mesh. |
| Unstable fills | At these fills the real diaphragm will fold or tilt to one side. An axisymmetric model cannot follow that, so treat those fills (and modes near 0.5–1 Hz) as low confidence. |
| Inputs that are estimates | Clamp height, rib size and count, and μ. The ribs and the web thickness control bending, so frequencies scale roughly with √(bending stiffness). |
| Strongest outputs | The pressure difference (near zero), the overall frequency band (about 0.5–7 Hz) and the large axial participation at high fill. |
| Weakest outputs | Individual mode frequencies at low fill and at the unstable fills. |

## Implications for the shaker test (supersedes the advice given for the 3 mm cup)
- **Frequency range:** sweep 0.3–30 Hz sideways and axially. The important content is about 1–7 Hz at 1 g.
- **Settling time:** stepped-sine dwells need about 40 s at 1.7 Hz (≈ 5/(ζω), ζ ≈ 0.02).
- **Axial testing is now important at high fill.** At 60–95 % fill about 20–25 % of the contents moves axially near 6 Hz at 1 g.
- **Look for asymmetric folding.** Video of the fold and crown, and a run along the second horizontal axis, are essential: the model predicts sideways buckling at several fills.
- **Low input levels with linearity checks:** for example 0.002–0.01 g. The diaphragm is slack throughout.
- **Pressure transducers:** they should show almost no pressure difference across the diaphragm (< 0.05 psi). A larger value would mean friction against the wall or something else not modelled.
