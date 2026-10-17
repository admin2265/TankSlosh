"""Tank and diaphragm geometry for the baseline problem (see PROBLEM_STATEMENT.md).

Frame: z along the tank axis, z = 0 at the inside apex of the lower head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .fluid import WATER_20C, Fluid


@dataclass(frozen=True)
class EllipticalHeadTank:
    """Vertical cylinder closed by two elliptical heads of depth radius / head_ratio."""
    radius: float
    height: float          # overall inner height, apex to apex
    head_ratio: float = 2.0

    def __post_init__(self):
        if self.height < 2 * self.head_depth:
            raise ValueError("height is less than the two head depths")

    @property
    def head_depth(self) -> float:
        return self.radius / self.head_ratio

    @property
    def straight_bottom(self) -> float:
        return self.head_depth

    @property
    def straight_top(self) -> float:
        return self.height - self.head_depth

    @property
    def head_volume(self) -> float:
        return 2.0 / 3.0 * np.pi * self.radius ** 2 * self.head_depth

    @property
    def volume(self) -> float:
        return 2 * self.head_volume + np.pi * self.radius ** 2 * (self.straight_top - self.straight_bottom)

    def radius_at(self, z: float) -> float:
        a, b = self.radius, self.head_depth
        if z <= 0 or z >= self.height:
            return 0.0
        if z < b:
            return a * np.sqrt(1 - ((b - z) / b) ** 2)
        if z > self.straight_top:
            return a * np.sqrt(1 - ((z - self.straight_top) / b) ** 2)
        return a

    def volume_below(self, z: float) -> float:
        """Volume below the horizontal plane at height z."""
        a, b = self.radius, self.head_depth
        z = float(np.clip(z, 0.0, self.height))

        def cap(s):  # volume of a head bowl from its apex up to depth s
            return np.pi * a ** 2 * (s ** 2 / b - s ** 3 / (3 * b ** 2))

        if z <= b:
            return cap(z)
        if z <= self.straight_top:
            return self.head_volume + np.pi * a ** 2 * (z - b)
        return self.volume - cap(self.height - z)

    def liquid_volume(self, fill: float) -> float:
        return fill * self.volume

    def liquid_mass(self, fill: float, fluid: Fluid = WATER_20C) -> float:
        return fluid.density * self.liquid_volume(fill)

    @property
    def head_meridian_length(self) -> float:
        a, b = self.radius, self.head_depth
        return quad(lambda t: np.hypot(a * np.sin(t), b * np.cos(t)), 0.0, np.pi / 2)[0]


@dataclass(frozen=True)
class MoldedCupDiaphragm:
    """Elastomer cup clamped around the wall at clamp_z, molded as a straight
    sidewall at r = R with an elliptical dome bottom matching the tank heads.
    Stress-free when the tank holds stress_free_fill."""
    tank: EllipticalHeadTank
    clamp_z: float
    stress_free_fill: float
    thickness: float
    shear_modulus: float     # neo-Hookean mu = 2 C10
    density: float

    def __post_init__(self):
        if not self.tank.straight_bottom <= self.dome_equator_z <= self.clamp_z <= self.tank.straight_top:
            raise ValueError("molded cup does not fit in the straight section")

    @property
    def dome_equator_z(self) -> float:
        # liquid below the cup = lower head + cylinder up to equator - dome (same shape as head)
        t = self.tank
        return t.straight_bottom + t.liquid_volume(self.stress_free_fill) / (np.pi * t.radius ** 2)

    @property
    def apex_z(self) -> float:
        return self.dome_equator_z - self.tank.head_depth

    @property
    def sidewall_length(self) -> float:
        return self.clamp_z - self.dome_equator_z

    @property
    def meridian_length(self) -> float:
        return self.sidewall_length + self.tank.head_meridian_length

    @property
    def mass(self) -> float:
        a = self.tank.radius
        b = self.tank.head_depth
        e2 = 1 - (b / a) ** 2  # oblate spheroid half-surface area
        e = np.sqrt(e2)
        dome_area = np.pi * a ** 2 * (1 + (1 - e2) / e * np.arctanh(e))
        side_area = 2 * np.pi * a * self.sidewall_length
        return self.density * self.thickness * (dome_area + side_area)


BASELINE_TANK = EllipticalHeadTank(radius=0.2, height=0.6, head_ratio=2.0)
BASELINE_DIAPHRAGM = MoldedCupDiaphragm(tank=BASELINE_TANK, clamp_z=0.5, stress_free_fill=0.5,
                                        thickness=3e-3, shear_modulus=1.8e6, density=1150.0)
BASELINE_FILL_RANGE = (0.05, 0.95)
BASELINE_MOUNT_POINT = (BASELINE_TANK.radius, 0.0, BASELINE_TANK.height / 2)
