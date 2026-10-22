"""Tank and diaphragm geometry for the baseline problem (see PROBLEM_STATEMENT.md).

Frame: z along the tank axis, z = 0 at the inside apex of the lower head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class RibSpec:
    """Radial (meridional) ribs on one face of the diaphragm."""
    count: int          # number of ribs around the circumference
    height: float       # m, standing above the web
    width: float        # m


@dataclass
class MeridianDiaphragm:
    """Diaphragm given by its molded (stress-free) mid-surface meridian, as points
    from the clamp (on the tank wall) to the axis (r = 0)."""
    tank: EllipticalHeadTank
    r0: np.ndarray
    z0: np.ndarray
    stress_free_fill: float
    thickness: float
    density: float
    shear_modulus: float = None       # neo-Hookean
    youngs_modulus: float = None      # linear
    poisson: float = 0.5
    material: str = "neo-hookean"
    bending: bool = True
    ribs: Optional[RibSpec] = None

    @property
    def clamp_z(self) -> float:
        return float(self.z0[0])

    @property
    def mass(self) -> float:
        """Web + ribs, kg (ribs counted until they merge near the axis)."""
        r, z = np.asarray(self.r0, float), np.asarray(self.z0, float)
        L = np.hypot(np.diff(r), np.diff(z))
        rm = 0.5 * (r[:-1] + r[1:])
        t_eff = np.full_like(rm, self.thickness)
        if self.ribs is not None:
            phi = np.minimum(1.0, self.ribs.count * self.ribs.width / (2 * np.pi * np.maximum(rm, 1e-12)))
            t_eff = t_eff + phi * self.ribs.height
        return float(self.density * np.sum(2 * np.pi * rm * L * t_eff))


def hemisphere_diaphragm(tank: EllipticalHeadTank, n_segments: int = 100, **kw) -> MeridianDiaphragm:
    """Diaphragm molded to the upper head of the tank, clamped at the top of the
    straight section; stress-free when the tank is full (Utsumi 2022 layout)."""
    a, b, zt = tank.radius, tank.head_depth, tank.straight_top
    t = np.linspace(0, np.pi / 2, 4001)
    ds = np.hypot(a * np.sin(t), b * np.cos(t))
    s = np.concatenate([[0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(t))])
    ti = np.interp(np.linspace(0, s[-1], n_segments + 1), s, t)
    r0, z0 = a * np.cos(ti), zt + b * np.sin(ti)
    r0[-1] = 0.0
    return MeridianDiaphragm(tank=tank, r0=r0, z0=z0, stress_free_fill=1.0, **kw)


def top_hat_meridian(tank: EllipticalHeadTank, brim_z: float, gap: float, lip: float = 0.01,
                     n_segments: int = 150):
    """Top-hat meridian: clamped on the wall at brim_z, a short flare (lip) in to a
    straight sidewall at r = R - gap up to the head seam, then a dome following the
    upper head offset inward by gap, to the axis. Points uniform in arc length."""
    R, b, zt = tank.radius, tank.head_depth, tank.straight_top
    th = np.linspace(0, np.pi / 2, 600)[1:]
    dome = np.column_stack([(R - gap) * np.cos(th), zt + (b - gap) * np.sin(th)])
    P = np.vstack([[(R, brim_z), (R - gap, brim_z + lip), (R - gap, zt)], dome])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    si = np.linspace(0, s[-1], n_segments + 1)
    r0, z0 = np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])
    r0[-1] = 0.0
    return r0, z0


def top_hat_diaphragm(tank: EllipticalHeadTank, stress_free_fill: float = 0.95,
                      brim_z: float = 0.2, lip: float = 0.01, n_segments: int = 150,
                      **kw) -> MeridianDiaphragm:
    """Molded top-hat diaphragm lining the upper part of the tank, liquid inside.
    The gap to the wall is chosen so the gas between hat and wall is
    (1 - stress_free_fill) of the tank volume when the hat is undeformed."""
    from scipy.optimize import brentq
    from .shell import AxisymmetricShell

    gas_target = (1 - stress_free_fill) * tank.volume
    above = tank.volume - tank.volume_below(brim_z)

    def excess(gap):
        r0, z0 = top_hat_meridian(tank, brim_z, gap, lip, n_segments)
        return above + AxisymmetricShell.enclosed_volume(r0, z0) - gas_target

    gap = brentq(excess, 1e-5, 0.5 * tank.head_depth, xtol=1e-12)
    r0, z0 = top_hat_meridian(tank, brim_z, gap, lip, n_segments)
    d = MeridianDiaphragm(tank=tank, r0=r0, z0=z0, stress_free_fill=stress_free_fill, **kw)
    d.gap = gap
    return d


BASELINE_TANK = EllipticalHeadTank(radius=0.2, height=0.6, head_ratio=2.0)
# previous baseline (Level B verification cases): 3 mm plain cup, stress-free at 50 %
CUP_3MM_DIAPHRAGM = MoldedCupDiaphragm(tank=BASELINE_TANK, clamp_z=0.5, stress_free_fill=0.5,
                                       thickness=3e-3, shear_modulus=1.8e6, density=1150.0)
# current baseline: ribbed top hat (2 mm web, ~36 radial ribs on the gas side, estimated
# from a photo), brim at z = 0.2 m, stress-free at 95 % fill; EPDM-class rubber (assumed)
BASELINE_RIBS = RibSpec(count=36, height=2e-3, width=3e-3)
BASELINE_DIAPHRAGM = top_hat_diaphragm(BASELINE_TANK, stress_free_fill=0.95, brim_z=0.2,
                                       thickness=2e-3, density=1150.0, shear_modulus=1.8e6,
                                       ribs=BASELINE_RIBS)
BASELINE_FILL_RANGE = (0.05, 0.95)
BASELINE_MOUNT_POINT = (BASELINE_TANK.radius, 0.0, BASELINE_TANK.height / 2)
