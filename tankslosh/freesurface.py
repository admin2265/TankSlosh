"""Analytic spring-mass slosh parameters for a flat-bottom upright cylinder
with a free liquid surface (no diaphragm), in a steady axial acceleration.

Classical linear potential-flow results (Abramson, NASA SP-106, 1966;
Dodge, "The New Dynamic Behavior of Liquids in Moving Containers", SwRI 2000):

    xi_n      roots of J1'(xi) = 0  (1.8412, 5.3314, 8.5363, ...)
    omega_n^2 = (g xi_n / R) tanh(xi_n h / R)
    m_n / m   = 2 tanh(xi_n h / R) / (xi_n (xi_n^2 - 1) h / R)
    h_n       = h - (2 R / xi_n) tanh(xi_n h / (2 R))     (above the bottom)
    m0 h0     = m h / 2 - sum(m_n h_n)

The heights assume the spring-mass convention used in simulate.py, where
each slosh mass also transmits its share of the axial load at its displaced
position (the "weight shift" moment m_n * a_z * x_n).

These results are the *gravity-dominated* limit. They are used here
  * to check the simulator and the later diaphragm (Level B) model, and
  * as a 1-g reference case that a StarCCM+ VOF set-up can be checked against.
They do NOT apply to the diaphragm tank in orbit, and they do not apply to any
free surface at low Bond number (surface tension dominates).
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
from scipy.special import jnp_zeros

from .emm import EMMParams, EMMTable, SloshMode
from .fluid import WATER_20C, Fluid

BOND_NUMBER_LIMIT = 100.0  # below this, surface tension is not negligible


def j1_prime_roots(n: int) -> np.ndarray:
    return jnp_zeros(1, n)


def bond_number(accel: float, radius: float, fluid: Fluid = WATER_20C) -> float:
    return fluid.density * accel * radius ** 2 / fluid.surface_tension


def viscous_damping_ratio(radius: float, fill_height: float, accel: float,
                          fluid: Fluid = WATER_20C) -> float:
    """First-mode damping ratio from wall/bottom boundary layers, clean free surface
    (Mikishev & Dorozhkin empirical fit, as quoted by Abramson and Dodge)."""
    hr = fill_height / radius
    x = 1.84 * hr
    return (0.79 * np.sqrt(fluid.kinematic_viscosity) / (radius ** 0.75 * accel ** 0.25)
            * (1.0 + 0.318 / np.sinh(x) * (1.0 + (1.0 - hr) / np.cosh(x))))


def cylinder_free_surface(radius: float, fill_height: float, accel: float = 9.80665,
                          fluid: Fluid = WATER_20C, n_modes: int = 3,
                          zeta: Union[None, float, Sequence[float]] = None,
                          z_bottom: float = 0.0, fill_fraction: Optional[float] = None,
                          warn_low_bond: bool = True) -> EMMParams:
    """Spring-mass parameters for a flat-bottom cylinder.

    radius, fill_height : m
    accel     : steady axial acceleration settling the liquid (m/s^2)
    zeta      : damping ratio(s); None -> viscous estimate for mode 1, used for all modes
    z_bottom  : z of the tank bottom in the tank frame (heights are offset by it)
    """
    if radius <= 0 or fill_height <= 0 or accel <= 0:
        raise ValueError("radius, fill_height and accel must be positive")
    bo = bond_number(accel, radius, fluid)
    if warn_low_bond and bo < BOND_NUMBER_LIMIT:
        warnings.warn(f"Bond number {bo:.3g} < {BOND_NUMBER_LIMIT:g}: surface tension is "
                      "significant and the gravity-dominated free-surface model is not valid",
                      stacklevel=2)

    m = fluid.density * np.pi * radius ** 2 * fill_height
    xi = j1_prime_roots(n_modes)
    hr = fill_height / radius
    th = np.tanh(xi * hr)
    omega = np.sqrt(accel * xi / radius * th)
    masses = m * 2.0 * th / (xi * (xi ** 2 - 1.0) * hr)
    heights = fill_height - (2.0 * radius / xi) * np.tanh(xi * hr / 2.0)

    if zeta is None:
        zetas = np.full(n_modes, viscous_damping_ratio(radius, fill_height, accel, fluid))
    else:
        zetas = np.broadcast_to(np.asarray(zeta, dtype=float), (n_modes,))

    m0 = m - masses.sum()
    m0h0 = m * fill_height / 2.0 - np.sum(masses * heights)
    h0 = m0h0 / m0 if m0 > 1e-12 * m else 0.0

    modes = [SloshMode(mass=float(mi), freq_hz=float(wi / (2 * np.pi)), zeta=float(zi),
                       height=float(hi + z_bottom))
             for mi, wi, zi, hi in zip(masses, omega, zetas, heights)]
    return EMMParams(liquid_mass=float(m), fixed_mass=float(m0), fixed_height=float(h0 + z_bottom),
                     lateral_modes=modes, axial_modes=[], fill_fraction=fill_fraction,
                     notes=(f"free-surface cylinder R={radius} m, h={fill_height} m, "
                            f"a={accel} m/s^2, Bo={bo:.3g}"))


def cylinder_free_surface_table(radius: float, tank_height: float, fills: Sequence[float],
                                accel: float = 9.80665, **kwargs) -> EMMTable:
    entries = [cylinder_free_surface(radius, f * tank_height, accel, fill_fraction=float(f),
                                     **kwargs)
               for f in fills]
    return EMMTable(entries=entries,
                    notes=f"free-surface cylinder R={radius} m, H={tank_height} m, a={accel} m/s^2")
