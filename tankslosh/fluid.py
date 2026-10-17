"""Liquid property sets (SI units)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fluid:
    density: float               # kg/m^3
    kinematic_viscosity: float   # m^2/s
    surface_tension: float       # N/m (against air/N2)
    bulk_modulus: float          # Pa (for acoustic checks only)

    @property
    def sound_speed(self) -> float:
        return (self.bulk_modulus / self.density) ** 0.5


WATER_20C = Fluid(density=998.2, kinematic_viscosity=1.004e-6,
                  surface_tension=0.0728, bulk_modulus=2.2e9)
