"""Customer cases: one small TOML file per tank/diaphragm problem.

A case lives in its own folder, cases/<name>/case.toml, and its results go to
cases/<name>/output/ (written by ``python -m tankslosh.run_case cases/<name>``).
This module reads the file and builds the library objects; it holds no
case-specific numbers. See docs/USER_MANUAL.md, "Setting up a new customer case",
for every key.
"""
from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .fluid import WATER_20C, Fluid
from .geometry import (EllipticalHeadTank, MeridianDiaphragm, MoldedCupDiaphragm, RibSpec,
                       hemisphere_diaphragm, top_hat_diaphragm)

FLUIDS = {"water_20c": WATER_20C}
DIAPHRAGM_TYPES = ("top_hat", "cup", "meridian", "hemisphere")
G_EARTH = 9.81


@dataclass
class Case:
    """A case as read from case.toml, with the objects the runner needs."""
    name: str
    folder: str
    description: str
    tank: EllipticalHeadTank
    diaphragm: object                  # MoldedCupDiaphragm or MeridianDiaphragm
    diaphragm_type: str
    n_segments: int                    # diaphragm mesh (meridian segments)
    fluid: Fluid
    liquid_name: str
    fills: np.ndarray
    sweep_max_step: float
    sweep_min_step: float
    liquid_mesh: float
    self_contact: bool
    n_keep_lateral: int
    n_keep_axial: int
    zeta: float
    orbit: bool
    ground_test: bool
    g: float
    mount_point: Tuple[float, float, float]
    pulse_fills: List[float]
    estimated: List[str] = field(default_factory=list)
    notes: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return os.path.join(self.folder, "output")


def _fills(spec) -> np.ndarray:
    if isinstance(spec, dict):
        start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
        f = np.arange(start, stop + 0.5 * step, step)
    else:
        f = np.asarray(spec, float)
    f = np.round(f, 6)
    if f.size == 0 or f.min() <= 0 or f.max() > 1:
        raise ValueError("fills must lie in (0, 1]")
    return f


def _fluid(spec: dict) -> Tuple[Fluid, str]:
    spec = dict(spec)
    preset = spec.pop("preset", "water_20c")
    name = spec.pop("name", preset)
    if preset not in FLUIDS:
        raise ValueError(f"unknown liquid preset {preset!r}; known: {sorted(FLUIDS)}")
    return dataclasses.replace(FLUIDS[preset], **spec), name


def _material(d: dict) -> dict:
    """Material keywords shared by the meridian-type diaphragms."""
    material = d.get("material", "neo-hookean")
    kw = dict(thickness=float(d["thickness"]), density=float(d["density"]), material=material,
              bending=bool(d.get("bending", True)))
    if material == "neo-hookean":
        kw["shear_modulus"] = float(d["shear_modulus"])
    elif material == "linear":
        kw["youngs_modulus"] = float(d["youngs_modulus"])
        kw["poisson"] = float(d.get("poisson", 0.5))
        kw["shear_modulus"] = kw["youngs_modulus"] / (2 * (1 + kw["poisson"]))
    else:
        raise ValueError(f"material must be 'neo-hookean' or 'linear', not {material!r}")
    if "ribs" in d:
        r = d["ribs"]
        kw["ribs"] = RibSpec(count=int(r["count"]), height=float(r["height"]), width=float(r["width"]))
    return kw


def build_diaphragm(tank: EllipticalHeadTank, d: dict, folder: str = "."):
    """Diaphragm object from the [diaphragm] table. Returns (diaphragm, n_segments)."""
    kind = d.get("type")
    if kind not in DIAPHRAGM_TYPES:
        raise ValueError(f"diaphragm.type must be one of {DIAPHRAGM_TYPES}, not {kind!r}")
    n = int(d.get("segments", 150 if kind == "top_hat" else 100))
    sf = float(d["stress_free_fill"]) if "stress_free_fill" in d else None
    if sf is None and kind != "hemisphere":
        raise ValueError(f"diaphragm.stress_free_fill is needed for a {kind} diaphragm")
    if kind == "cup":
        plain = (d.get("material", "neo-hookean") == "neo-hookean" and "ribs" not in d
                 and d.get("bending", True))
        kw = _material(d)
        cup = MoldedCupDiaphragm(tank=tank, clamp_z=float(d["clamp_z"]), stress_free_fill=sf,
                                 thickness=kw["thickness"], shear_modulus=kw["shear_modulus"],
                                 density=kw["density"])
        if plain:
            return cup, n
        # ribs / linear material / no bending: same molded shape as a meridian diaphragm
        from .diaphragm_static import cup_reference_meridian
        r0, z0 = cup_reference_meridian(cup, n)
        return MeridianDiaphragm(tank=tank, r0=r0, z0=z0, stress_free_fill=sf, **kw), n
    kw = _material(d)
    if kind == "top_hat":
        dia = top_hat_diaphragm(tank, stress_free_fill=sf, brim_z=float(d["brim_z"]),
                                lip=float(d.get("lip", 0.01)), n_segments=n, **kw)
    elif kind == "hemisphere":
        dia = hemisphere_diaphragm(tank, n_segments=n, **kw)
    else:  # meridian: points file (r, z columns, clamp first, axis last), relative to the case
        path = os.path.join(folder, d["points_file"])
        pts = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
        if pts.shape[1] < 2:
            raise ValueError(f"{path}: need two columns r, z")
        r0, z0 = pts[:, 0].copy(), pts[:, 1].copy()
        if abs(r0[-1]) > 1e-9 * tank.radius:
            raise ValueError(f"{path}: the last point must be on the axis (r = 0)")
        dia = MeridianDiaphragm(tank=tank, r0=r0, z0=z0, stress_free_fill=sf, **kw)
        n = len(r0) - 1
    return dia, n


def load_case(folder: str) -> Case:
    """Read <folder>/case.toml."""
    folder = os.path.abspath(folder)
    path = os.path.join(folder, "case.toml")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    t = raw["tank"]
    tank = EllipticalHeadTank(radius=float(t["radius"]), height=float(t["height"]),
                              head_ratio=float(t.get("head_ratio", 2.0)))
    diaphragm, n_seg = build_diaphragm(tank, raw["diaphragm"], folder)
    fluid, liquid_name = _fluid(raw.get("liquid", {}))
    env = raw.get("environment", {})
    a = raw.get("analysis", {})
    damp = raw.get("damping", {})
    mount = tuple(float(x) for x in t.get("mount_point", (tank.radius, 0.0, tank.height / 2)))
    fills = _fills(a.get("fills", dict(start=0.05, stop=0.95, step=0.05)))
    return Case(
        name=raw.get("name", os.path.basename(folder)), folder=folder,
        description=raw.get("description", ""), tank=tank, diaphragm=diaphragm,
        diaphragm_type=raw["diaphragm"]["type"], n_segments=n_seg, fluid=fluid,
        liquid_name=liquid_name, fills=fills,
        sweep_max_step=float(a.get("sweep_max_step", 0.025)),
        sweep_min_step=float(a.get("sweep_min_step", 0.002)),
        liquid_mesh=float(a.get("liquid_mesh", 0.005)),
        self_contact=bool(a.get("self_contact", True)),
        n_keep_lateral=int(a.get("n_keep_lateral", 3)), n_keep_axial=int(a.get("n_keep_axial", 2)),
        zeta=float(damp.get("zeta", 0.02)),
        orbit=bool(env.get("orbit", True)), ground_test=bool(env.get("ground_test", False)),
        g=float(env.get("g", G_EARTH)), mount_point=mount,
        pulse_fills=[float(f) for f in a.get("pulse_fills", [])],
        estimated=list(raw.get("estimated", [])), notes=raw.get("notes", ""), raw=raw)
