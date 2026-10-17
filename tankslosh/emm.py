"""Equivalent mechanical model (EMM) parameters for liquid slosh.

The liquid mass m is represented by
  * a fixed mass m0 rigidly attached to the tank at height h0,
  * N spring-mass oscillators per lateral axis (mass m_n, natural frequency
    f_n, damping ratio zeta_n, spring attachment height h_n); the tank is
    axisymmetric, so the same set is used for the x and y axes,
  * optionally N_ax axial oscillators along the tank axis (z). The axial
    mass not carried by axial oscillators moves rigidly with the tank.

Lateral bookkeeping: m0 + sum(m_n) = m.  Heights are z coordinates in the
tank frame; the frame origin is the reference point used for moments.
SI units throughout.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SloshMode:
    mass: float          # kg
    freq_hz: float       # undamped natural frequency, Hz
    zeta: float          # damping ratio (-)
    height: float = 0.0  # m, spring attachment height (lateral modes only)

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq_hz

    @property
    def stiffness(self) -> float:
        return self.mass * self.omega ** 2

    @property
    def damping(self) -> float:
        return 2.0 * self.zeta * self.mass * self.omega


@dataclass
class EMMParams:
    liquid_mass: float
    fixed_mass: float
    fixed_height: float
    lateral_modes: List[SloshMode] = field(default_factory=list)
    axial_modes: List[SloshMode] = field(default_factory=list)
    fill_fraction: Optional[float] = None
    notes: str = ""

    def __post_init__(self):
        self.lateral_modes = [m if isinstance(m, SloshMode) else SloshMode(**m)
                              for m in self.lateral_modes]
        self.axial_modes = [m if isinstance(m, SloshMode) else SloshMode(**m)
                            for m in self.axial_modes]
        self.validate()

    @property
    def axial_rigid_mass(self) -> float:
        return self.liquid_mass - sum(m.mass for m in self.axial_modes)

    @property
    def cg_height(self) -> float:
        """Liquid CG height at rest (z, tank frame)."""
        first_moment = self.fixed_mass * self.fixed_height + sum(
            m.mass * m.height for m in self.lateral_modes)
        return first_moment / self.liquid_mass

    def validate(self, rtol: float = 1e-6) -> None:
        if self.liquid_mass <= 0:
            raise ValueError("liquid_mass must be positive")
        if self.fixed_mass < -rtol * self.liquid_mass:
            raise ValueError(f"fixed_mass is negative ({self.fixed_mass})")
        for kind, modes in (("lateral", self.lateral_modes), ("axial", self.axial_modes)):
            for i, m in enumerate(modes):
                if m.mass <= 0 or m.freq_hz <= 0 or m.zeta < 0:
                    raise ValueError(f"{kind} mode {i}: need mass>0, freq_hz>0, zeta>=0 ({m})")
        total = self.fixed_mass + sum(m.mass for m in self.lateral_modes)
        if abs(total - self.liquid_mass) > rtol * self.liquid_mass:
            raise ValueError(f"fixed_mass + sum(lateral masses) = {total:.6g} "
                             f"!= liquid_mass = {self.liquid_mass:.6g}")
        if self.axial_rigid_mass < -rtol * self.liquid_mass:
            raise ValueError("sum of axial mode masses exceeds liquid_mass")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EMMParams":
        return cls(**d)

    def save_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path) -> "EMMParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        lines = [f"liquid mass {self.liquid_mass:.4g} kg"
                 + (f", fill {self.fill_fraction:.0%}" if self.fill_fraction is not None else ""),
                 f"  fixed mass {self.fixed_mass:.4g} kg at z = {self.fixed_height:.4g} m",
                 f"  CG height  {self.cg_height:.4g} m"]
        for i, m in enumerate(self.lateral_modes, 1):
            lines.append(f"  lateral {i}: m = {m.mass:.4g} kg ({m.mass / self.liquid_mass:.1%}), "
                         f"f = {m.freq_hz:.4g} Hz, zeta = {m.zeta:.3g}, z = {m.height:.4g} m")
        for i, m in enumerate(self.axial_modes, 1):
            lines.append(f"  axial   {i}: m = {m.mass:.4g} kg ({m.mass / self.liquid_mass:.1%}), "
                         f"f = {m.freq_hz:.4g} Hz, zeta = {m.zeta:.3g}")
        return "\n".join(lines)


@dataclass
class EMMTable:
    """EMM parameters tabulated against fill fraction, linearly interpolated.

    All entries must have the same number of lateral and axial modes.
    Masses, frequencies, damping ratios and first moments (m*h) are
    interpolated, so mass bookkeeping and the CG height stay consistent.
    """
    entries: List[EMMParams]
    notes: str = ""

    def __post_init__(self):
        self.entries = [e if isinstance(e, EMMParams) else EMMParams.from_dict(e)
                        for e in self.entries]
        if not self.entries:
            raise ValueError("EMMTable needs at least one entry")
        if any(e.fill_fraction is None for e in self.entries):
            raise ValueError("every entry needs fill_fraction")
        self.entries.sort(key=lambda e: e.fill_fraction)
        n_lat = {len(e.lateral_modes) for e in self.entries}
        n_ax = {len(e.axial_modes) for e in self.entries}
        if len(n_lat) > 1 or len(n_ax) > 1:
            raise ValueError("all entries must have the same number of modes")

    @property
    def fills(self) -> np.ndarray:
        return np.array([e.fill_fraction for e in self.entries])

    def at(self, fill: float) -> EMMParams:
        fills = self.fills
        if not fills[0] - 1e-12 <= fill <= fills[-1] + 1e-12:
            raise ValueError(f"fill {fill} outside table range [{fills[0]}, {fills[-1]}]")
        if len(fills) == 1:
            return EMMParams.from_dict(self.entries[0].to_dict())
        j = int(np.clip(np.searchsorted(fills, fill), 1, len(fills) - 1))
        a, b = self.entries[j - 1], self.entries[j]
        w = (fill - a.fill_fraction) / (b.fill_fraction - a.fill_fraction)

        def lerp(x, y):
            return (1 - w) * x + w * y

        def lerp_modes(ma, mb, lateral):
            out = []
            for p, q in zip(ma, mb):
                mass = lerp(p.mass, q.mass)
                height = lerp(p.mass * p.height, q.mass * q.height) / mass if lateral else 0.0
                out.append(SloshMode(mass=mass, freq_hz=lerp(p.freq_hz, q.freq_hz),
                                     zeta=lerp(p.zeta, q.zeta), height=height))
            return out

        m0 = lerp(a.fixed_mass, b.fixed_mass)
        m0h0 = lerp(a.fixed_mass * a.fixed_height, b.fixed_mass * b.fixed_height)
        return EMMParams(liquid_mass=lerp(a.liquid_mass, b.liquid_mass),
                         fixed_mass=m0,
                         fixed_height=m0h0 / m0 if m0 > 0 else 0.0,
                         lateral_modes=lerp_modes(a.lateral_modes, b.lateral_modes, True),
                         axial_modes=lerp_modes(a.axial_modes, b.axial_modes, False),
                         fill_fraction=fill,
                         notes=f"interpolated between fills {a.fill_fraction} and {b.fill_fraction}")

    def save_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump({"notes": self.notes, "entries": [e.to_dict() for e in self.entries]},
                      f, indent=2)

    @classmethod
    def load_json(cls, path) -> "EMMTable":
        with open(path) as f:
            d = json.load(f)
        return cls(entries=d["entries"], notes=d.get("notes", ""))
