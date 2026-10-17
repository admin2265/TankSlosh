"""Static shape of the diaphragm against fill level (Level B, step 1).

The shell energy (tankslosh.shell) is minimized with the enclosed volume held
fixed and frictionless contact with the tank wall. The liquid and gas pressures
are each uniform (no gravity in orbit), so the pressure difference across the
diaphragm is the Lagrange multiplier of the volume constraint:

    dp = p_liquid - p_gas = dU / dV_liquid.

Fill levels are solved by continuation, starting from the stress-free fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .geometry import BASELINE_DIAPHRAGM, EllipticalHeadTank, MoldedCupDiaphragm
from .shell import AxisymmetricShell, self_intersections


def _uniform_ellipse_arc(a: float, b: float, n: int):
    """n+1 points on the quarter ellipse r = a cos(t), z = -b sin(t), t in [0, pi/2],
    spaced uniformly in arc length (equator -> apex)."""
    t = np.linspace(0, np.pi / 2, 4001)
    ds = np.hypot(a * np.sin(t), b * np.cos(t))
    s = np.concatenate([[0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(t))])
    ti = np.interp(np.linspace(0, s[-1], n + 1), s, t)
    return a * np.cos(ti), -b * np.sin(ti)


def cup_reference_meridian(d: MoldedCupDiaphragm, n_segments: int = 100):
    """Molded meridian from the clamp to the axis: straight sidewall, then dome."""
    tank = d.tank
    n_side = max(2, int(round(n_segments * d.sidewall_length / d.meridian_length)))
    n_dome = n_segments - n_side
    zs = np.linspace(d.clamp_z, d.dome_equator_z, n_side + 1)
    rs = np.full_like(zs, tank.radius)
    rd, zd = _uniform_ellipse_arc(tank.radius, tank.head_depth, n_dome)
    r = np.concatenate([rs, rd[1:]])
    z = np.concatenate([zs, d.dome_equator_z + zd[1:]])
    r[-1] = 0.0
    return r, z


class WallContact:
    """Frictionless contact with the inside of an elliptical-head tank:
    g = 1 - (r/a)^2 - (below/b)^2 - (above/b)^2 >= 0."""

    def __init__(self, tank: EllipticalHeadTank):
        self.a, self.b = tank.radius, tank.head_depth
        self.zb, self.zt = tank.straight_bottom, tank.straight_top

    def g(self, r, z):
        lo = np.maximum(0.0, self.zb - z)
        hi = np.maximum(0.0, z - self.zt)
        return 1.0 - (r / self.a) ** 2 - (lo / self.b) ** 2 - (hi / self.b) ** 2

    def dg(self, r, z):
        lo = np.maximum(0.0, self.zb - z)
        hi = np.maximum(0.0, z - self.zt)
        return -2 * r / self.a ** 2, 2 * lo / self.b ** 2 - 2 * hi / self.b ** 2


@dataclass
class StaticSolution:
    target_volume: float          # enclosed volume V_d, m^3
    r: np.ndarray
    z: np.ndarray
    energy: float                 # J
    dp: float                     # p_liquid - p_gas, Pa
    equilibrium_residual: float   # |grad U - dp grad V| / |grad U| on free DOFs
    converged: bool
    message: str
    contact_nodes: np.ndarray     # indices of nodes touching the wall
    stretch_meridional: np.ndarray
    stretch_hoop: np.ndarray
    self_intersections: list = field(default_factory=list)
    fill: Optional[float] = None

    @property
    def max_stretch(self) -> float:
        return float(max(self.stretch_meridional.max(), self.stretch_hoop.max()))

    @property
    def min_stretch(self) -> float:
        return float(min(self.stretch_meridional.min(), self.stretch_hoop.min()))

    @property
    def thickness_ratio(self) -> np.ndarray:
        """Current / molded thickness per segment (incompressible)."""
        return 1.0 / (self.stretch_meridional * self.stretch_hoop)


class ShellStaticSolver:
    """Minimize shell energy with fixed enclosed volume and optional wall contact."""

    def __init__(self, shell: AxisymmetricShell, contact: Optional[WallContact] = None):
        self.shell = shell
        self.contact = contact
        self.N = shell.N
        self.U_scale = shell.mu * shell.t * shell.area0
        self.V_scale = shell.area0 * float(np.sum(shell.L0)) * 0.1

    def _unpack(self, x):
        N = self.N
        r = np.empty(N + 1)
        z = np.empty(N + 1)
        r[0], z[0] = self.shell.r0[0], self.shell.z0[0]
        r[1:N] = x[:N - 1]
        r[N] = 0.0
        z[1:] = x[N - 1:]
        return r, z

    def _pack_grad(self, gr, gz):
        return np.concatenate([gr[1:self.N], gz[1:]])

    def solve(self, target_volume: float, r_init=None, z_init=None,
              maxiter: int = 2000) -> StaticSolution:
        sh, N = self.shell, self.N
        r_init = sh.r0 if r_init is None else r_init
        z_init = sh.z0 if z_init is None else z_init
        x0 = np.concatenate([r_init[1:N], z_init[1:]])

        def fun(x):
            r, z = self._unpack(x)
            U, gr, gz = sh.energy(r, z, grad=True)
            return U / self.U_scale, self._pack_grad(gr, gz) / self.U_scale

        def vol(x):
            r, z = self._unpack(x)
            return (sh.enclosed_volume(r, z) - target_volume) / self.V_scale

        def vol_jac(x):
            r, z = self._unpack(x)
            _, gr, gz = sh.enclosed_volume(r, z, grad=True)
            return self._pack_grad(gr, gz) / self.V_scale

        cons = [{"type": "eq", "fun": vol, "jac": vol_jac}]
        if self.contact is not None:
            c = self.contact

            def con_g(x):
                r, z = self._unpack(x)
                return c.g(r[1:], z[1:])

            def con_jac(x):
                r, z = self._unpack(x)
                dgr, dgz = c.dg(r[1:], z[1:])
                J = np.zeros((N, 2 * N - 1))
                idx = np.arange(N - 1)
                J[idx, idx] = dgr[:-1]                 # r of nodes 1..N-1
                J[np.arange(N), N - 1 + np.arange(N)] = dgz
                return J

            cons.append({"type": "ineq", "fun": con_g, "jac": con_jac})

        res = minimize(fun, x0, jac=True, method="SLSQP", constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-13})
        r, z = self._unpack(res.x)
        return self._finish(target_volume, r, z, res)

    def _finish(self, target_volume, r, z, res) -> StaticSolution:
        sh = self.shell
        U, gr, gz = sh.energy(r, z, grad=True)
        _, vr, vz = sh.enclosed_volume(r, z, grad=True)
        gU = self._pack_grad(gr, gz)
        gV = self._pack_grad(vr, vz)
        if self.contact is not None:
            gc = self.contact.g(r[1:], z[1:])
            contact_nodes = np.where(gc < 1e-6)[0] + 1
        else:
            contact_nodes = np.array([], dtype=int)
        free = np.ones(self.N + 1, dtype=bool)
        free[contact_nodes] = False
        mask = np.concatenate([free[1:self.N], free[1:]])
        lam = float(gU[mask] @ gV[mask] / (gV[mask] @ gV[mask]))  # dU/dV_d
        resid = float(np.linalg.norm(gU[mask] - lam * gV[mask]) /
                      max(np.linalg.norm(gU[mask]), 1e-30))
        l1, l2 = sh.stretches(r, z)
        # SLSQP often stops with "positive directional derivative" at machine
        # precision; judge convergence by the constraint and equilibrium errors.
        vol_err = abs(sh.enclosed_volume(r, z) - target_volume) / self.V_scale
        pen = 0.0 if self.contact is None else max(0.0, -float(self.contact.g(r[1:], z[1:]).min()))
        converged = bool(res.success) or (vol_err < 1e-8 and pen < 1e-8 and resid < 1e-3)
        return StaticSolution(target_volume=target_volume, r=r, z=z, energy=float(U),
                              dp=-lam, equilibrium_residual=resid,
                              converged=converged, message=str(res.message),
                              contact_nodes=contact_nodes, stretch_meridional=l1,
                              stretch_hoop=l2, self_intersections=self_intersections(r, z))


class DiaphragmStatics:
    """Static diaphragm shapes for a molded cup diaphragm in an elliptical-head tank."""

    def __init__(self, diaphragm: MoldedCupDiaphragm = BASELINE_DIAPHRAGM, n_segments: int = 100):
        self.d = diaphragm
        r0, z0 = cup_reference_meridian(diaphragm, n_segments)
        self.shell = AxisymmetricShell(r0, z0, diaphragm.shear_modulus, diaphragm.thickness)
        self.solver = ShellStaticSolver(self.shell, WallContact(diaphragm.tank))
        # Measure fill changes from the discretized molded volume, so the
        # stress-free fill is exactly stress-free on this mesh.
        self.V_ref = self.shell.enclosed_volume(r0, z0)

    def enclosed_volume_for_fill(self, fill: float) -> float:
        return self.V_ref - (fill - self.d.stress_free_fill) * self.d.tank.volume

    def solve(self, fill: float, r_init=None, z_init=None) -> StaticSolution:
        sol = self.solver.solve(self.enclosed_volume_for_fill(fill), r_init, z_init)
        sol.fill = fill
        return sol

    def sweep(self, fills: Sequence[float], max_step: float = 0.025,
              min_step: float = 0.002) -> List[StaticSolution]:
        """Solve each fill by continuation from the stress-free fill, with adaptive
        sub-steps (halved on failure)."""
        f0 = self.d.stress_free_fill
        fills = sorted(set(round(float(f), 12) for f in fills))
        out = {}
        ref = self.solve(f0)
        if f0 in fills:
            out[f0] = ref

        def ok(s):
            return s.converged and not s.self_intersections

        for branch in ([f for f in fills if f < f0][::-1], [f for f in fills if f > f0]):
            cur, failed = ref, False
            for f_target in branch:
                while not failed and cur.fill != f_target:
                    step = min(max_step, abs(f_target - cur.fill))
                    direction = np.sign(f_target - cur.fill)
                    while True:
                        f_next = f_target if step == abs(f_target - cur.fill) else \
                            round(cur.fill + direction * step, 12)
                        trial = self.solve(f_next, cur.r, cur.z)
                        if ok(trial) or step / 2 < min_step:
                            break
                        step /= 2
                    if not ok(trial):
                        failed = True   # later fills on this branch are unreachable
                        out[f_target] = trial
                        break
                    cur = trial
                if failed:
                    break
                out[f_target] = cur
        return [out[f] for f in fills if f in out]
