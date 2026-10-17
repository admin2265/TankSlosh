"""Lateral (m = 1) slosh modes of the diaphragm + liquid about a static state,
and their conversion to Level A spring-mass parameters (Level B, step 2).

Generalized coordinates u: harmonic shell displacements (tankslosh.harmonic_shell).
With the tank translating at velocity U along x, the kinetic energy of the
contents (liquid + rubber) is
    T = 1/2 m U^2 + U Gamma^T u_dot + 1/2 u_dot^T M u_dot,
where M = liquid added mass + rubber mass and Gamma^T u = m x_cg (first moment).
The potential energy is 1/2 u^T K u with K = Hess(U_shell) - lambda Hess(V_d),
lambda = dU/dV_d (the follower-pressure term; the liquid is incompressible).
Hence  M u'' + K u = -Gamma a,  and with modes K phi = w^2 M phi, phi^T M phi = 1:
    slosh mass   m_n = (phi_n^T Gamma)^2
    height       h_n = (phi_n^T H) / (phi_n^T Gamma),
H^T u_dot being the relative angular momentum about the y axis. These map
exactly onto the Level A oscillators (tankslosh.emm / simulate).

Linearization assumptions: wall contact stays closed (nodes in contact may
slide along the wall but not lift off), gas pressure stays uniform, no
residual-acceleration (hydrostatic) stiffness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.linalg import eigh, null_space

from .diaphragm_static import DiaphragmStatics, StaticSolution
from .emm import EMMParams, EMMTable, SloshMode
from .fluid import WATER_20C, Fluid
from .geometry import EllipticalHeadTank
from .harmonic_shell import HarmonicShell
from .liquid_fe import (HarmonicPotential, build_mesh, surface_moment, surface_normals,
                        translation_coupling)


def _wall_profile(tank: EllipticalHeadTank):
    a, b = tank.radius, tank.head_depth
    ph = np.linspace(0, np.pi / 2, 1500)
    low = np.column_stack([a * np.sin(ph), b * (1 - np.cos(ph))])
    cyl = np.column_stack([np.full(1500, a), np.linspace(b, tank.straight_top, 1500)])[1:]
    up = np.column_stack([a * np.cos(ph), tank.straight_top + b * np.sin(ph)])[1:]
    prof = np.vstack([low, cyl, up])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    return prof, s


def tank_wall_polyline(tank: EllipticalHeadTank, p_start, p_end, h: float) -> np.ndarray:
    """Points along the tank wall between the wall points nearest p_start and p_end
    (bottom-to-top order). p_start = None starts at the bottom apex (0, 0)."""
    prof, s = _wall_profile(tank)
    k0 = 0 if p_start is None else int(np.argmin(np.linalg.norm(prof - np.asarray(p_start), axis=1)))
    k1 = int(np.argmin(np.linalg.norm(prof - np.asarray(p_end), axis=1)))
    n = max(int(np.ceil((s[k1] - s[k0]) / h)), 1)
    si = np.linspace(s[k0], s[k1], n + 1)
    pts = np.column_stack([np.interp(si, s, prof[:, 0]), np.interp(si, s, prof[:, 1])])
    pts[0], pts[-1] = prof[k0], prof[k1]
    if p_start is None:
        pts[0] = (0.0, 0.0)
    return pts


@dataclass
class LateralModes:
    fill: float
    freq_hz: np.ndarray        # all modes, ascending
    slosh_mass: np.ndarray     # kg
    height: np.ndarray         # m (tank frame)
    liquid_mass: float
    rubber_mass: float
    liquid_cg_z: float
    contents_cg_z: float
    mesh_volume_error: float   # relative
    min_eigenvalue: float      # < 0 means the static state is unstable to m = 1
    n_wet_nodes: int
    detach_node: int

    @property
    def contents_mass(self) -> float:
        return self.liquid_mass + self.rubber_mass

    @property
    def stable(self) -> bool:
        return self.min_eigenvalue > 0

    def dominant(self, n_keep: int = 3, min_fraction: float = 0.002):
        """Indices of the lowest-frequency modes carrying at least min_fraction of the
        contents mass (at most n_keep)."""
        idx = [i for i in range(len(self.freq_hz))
               if self.slosh_mass[i] >= min_fraction * self.contents_mass]
        return idx[:n_keep]

    def emm(self, n_keep: int = 3, zeta: float = 0.02, min_fraction: float = 0.002) -> EMMParams:
        sel = self.dominant(n_keep, min_fraction)
        modes = [SloshMode(mass=float(self.slosh_mass[i]), freq_hz=float(self.freq_hz[i]),
                           zeta=zeta, height=float(self.height[i])) for i in sel]
        m = self.contents_mass
        m0 = m - sum(md.mass for md in modes)
        h0 = (m * self.contents_cg_z - sum(md.mass * md.height for md in modes)) / m0
        return EMMParams(liquid_mass=m, fixed_mass=m0, fixed_height=h0, lateral_modes=modes,
                         axial_modes=[], fill_fraction=self.fill,
                         notes=(f"Level B lateral modes, contents = liquid {self.liquid_mass:.3f} kg"
                                f" + diaphragm {self.rubber_mass:.3f} kg; zeta = {zeta} placeholder"))


@dataclass
class _LiquidRegion:
    contact: set          # nodes held to the wall (bilateral, frictionless)
    k_det: int            # first wetted node (diaphragm leaves the wall)
    k_end: int            # last wetted node (N, or start of a patch resting on the bottom)
    surf: np.ndarray      # wetted surface points k_det..k_end
    normals: np.ndarray   # outward (from liquid) normals of wetted segments
    mesh: object


def _liquid_region(ds: DiaphragmStatics, sol: StaticSolution, h_mesh: float,
                   gap_tol: float) -> _LiquidRegion:
    """Liquid region: tank wall up to where the diaphragm leaves it, then the wetted
    diaphragm. Sidewall nodes within gap_tol of the wall, in the run from the clamp,
    are treated as resting on the wall: a sealed sub-mm sliver cannot hold liquid,
    and an open one acts as a squeeze film that blocks normal motion. A dome centre
    resting on the bottom is treated the same way."""
    tank, N = ds.d.tank, ds.shell.N
    cont = ds.solver.contact
    contact = set(int(i) for i in sol.contact_nodes)
    gap = 0.5 * tank.radius * np.maximum(cont.g(sol.r, sol.z), 0.0)
    k_det = 0
    while k_det + 1 < N and gap[k_det + 1] < gap_tol:
        k_det += 1
    k_end = N
    if gap[N] < gap_tol and N in contact:
        while k_end - 1 > k_det + 1 and gap[k_end - 1] < gap_tol:
            k_end -= 1
    contact |= set(range(1, k_det + 1))
    if k_end < N:
        contact |= set(range(k_end, N + 1))
    surf = np.column_stack([sol.r[k_det:k_end + 1], sol.z[k_det:k_end + 1]])
    p_low = None if k_end == N else surf[-1]
    wall = tank_wall_polyline(tank, p_low, surf[0], h_mesh)
    # end exactly at the detachment node (and start at the resting node), dropping
    # profile points that would double back within half a mesh spacing
    start = np.array([0.0, 0.0]) if p_low is None else surf[-1]
    inner = wall[1:-1]
    far = (np.linalg.norm(inner - surf[0], axis=1) > 0.5 * h_mesh) & \
        (np.linalg.norm(inner - start, axis=1) > 0.5 * h_mesh)
    wall = np.vstack([start, inner[far], surf[0]])
    mesh = build_mesh(wall, surf, h_mesh)
    return _LiquidRegion(contact=contact, k_det=k_det, k_end=k_end, surf=surf,
                         normals=surface_normals(surf), mesh=mesh)


def _wall_rows(ds, sol, contact, n_per_node, ir, iz):
    """Constraint rows: frictionless bilateral wall contact (no motion along the wall normal)."""
    N = ds.shell.N
    n = n_per_node * (N + 1)
    rows = []
    for i in sorted(contact):
        if i == 0:
            continue
        gr, gz = ds.solver.contact.dg(sol.r[i], sol.z[i])
        nn = np.hypot(gr, gz)
        e = np.zeros(n)
        e[n_per_node * i + ir] = gr / nn
        e[n_per_node * i + iz] = gz / nn
        rows.append(e)
    return rows


def lateral_modes(ds: DiaphragmStatics, sol: StaticSolution, fluid: Fluid = WATER_20C,
                  h_mesh: float = 0.005, n_theta: int = 8,
                  gap_tol: float = 2e-3) -> LateralModes:
    sh, tank, N = ds.shell, ds.d.tank, ds.shell.N
    rho = fluid.density
    hs = HarmonicShell(sh, sol.r, sol.z, n_theta)
    HU, HV = hs.hessians()
    K = HU - (-sol.dp) * HV
    m_nodes, Ms, Gs, Hs = hs.rubber_mass(ds.d.density)

    reg = _liquid_region(ds, sol, h_mesh, gap_tol)
    k_det, k_end = reg.k_det, reg.k_end
    V_mesh, zc_L = reg.mesh.volume_and_cg()
    V_liq = tank.liquid_volume(sol.fill)
    hp = HarmonicPotential(reg.mesh, 1)
    n = 3 * (N + 1)
    G = np.zeros((len(hp.free), n))
    G[:, 3 * k_det:3 * (k_end + 1)] = hp.surface_load_matrix(reg.normals, 3)
    X = hp.solve(G)
    Ma = rho * np.pi * (G.T @ X)
    Ma = 0.5 * (Ma + Ma.T)
    GamL = np.zeros(n)
    GamL[3 * k_det:3 * (k_end + 1)] = rho * np.pi * translation_coupling(reg.surf, reg.normals, 3)
    HL = rho * np.pi * (hp.c @ X)

    M = Ma + Ms
    Gam = GamL + Gs
    H = HL + Hs

    # constraints: clamp fixed; axis single-valued; frictionless bilateral wall contact
    rows = [np.eye(n)[d] for d in range(3)]
    e = np.zeros(n)
    e[3 * N] = e[3 * N + 1] = 1.0
    rows += [e, np.eye(n)[3 * N + 2]]
    rows += _wall_rows(ds, sol, reg.contact - {N}, 3, 0, 2)
    Z = null_space(np.array(rows))
    Kr = Z.T @ K @ Z
    Mr = Z.T @ M @ Z
    Kr, Mr = 0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T)
    w2, phi = eigh(Kr, Mr)
    Gn = phi.T @ (Z.T @ Gam)
    Hn = phi.T @ (Z.T @ H)
    with np.errstate(divide="ignore", invalid="ignore"):
        hn = np.where(np.abs(Gn) > 1e-12, Hn / Gn, np.nan)
    m_L = rho * V_liq
    m_d = float(m_nodes.sum())
    zc = (m_L * zc_L + float(m_nodes @ sol.z)) / (m_L + m_d)
    return LateralModes(fill=float(sol.fill), freq_hz=np.sqrt(np.abs(w2)) * np.sign(w2) / (2 * np.pi),
                        slosh_mass=Gn ** 2, height=hn, liquid_mass=m_L, rubber_mass=m_d,
                        liquid_cg_z=zc_L, contents_cg_z=zc,
                        mesh_volume_error=V_mesh / V_liq - 1.0, min_eigenvalue=float(w2[0]),
                        n_wet_nodes=len(reg.surf), detach_node=k_det)


@dataclass
class AxialModes:
    fill: float
    freq_hz: np.ndarray        # all volume-preserving axisymmetric modes, ascending
    slosh_mass: np.ndarray     # kg, participation in axial (z) acceleration
    contents_mass: float
    min_eigenvalue: float

    @property
    def stable(self) -> bool:
        return self.min_eigenvalue > 0

    def dominant(self, n_keep: int = 2, min_fraction: float = 0.002):
        idx = [i for i in range(len(self.freq_hz))
               if self.slosh_mass[i] >= min_fraction * self.contents_mass]
        return idx[:n_keep]

    def modes(self, n_keep: int = 2, zeta: float = 0.02, min_fraction: float = 0.002):
        return [SloshMode(mass=float(self.slosh_mass[i]), freq_hz=float(self.freq_hz[i]),
                          zeta=zeta, height=0.0) for i in self.dominant(n_keep, min_fraction)]


def axisymmetric_hessians(shell, r, z, h: float = 1e-7):
    """Hessians of the axisymmetric shell energy and enclosed volume with respect to
    the nodal (u_r, u_z), by central differences of the analytic gradients."""
    N = shell.N
    n = 2 * (N + 1)
    HU = np.zeros((n, n))
    HV = np.zeros((n, n))

    def grads(rr, zz):
        _, gr, gz = shell.energy(rr, zz, grad=True)
        _, vr, vz = shell.enclosed_volume(rr, zz, grad=True)
        return np.column_stack([gr, gz]).ravel(), np.column_stack([vr, vz]).ravel()

    for k in range(n):
        node, comp = divmod(k, 2)
        rp, zp, rm, zm = r.copy(), z.copy(), r.copy(), z.copy()
        if comp == 0:
            rp[node] += h
            rm[node] -= h
        else:
            zp[node] += h
            zm[node] -= h
        gp, vp = grads(rp, zp)
        gm, vm = grads(rm, zm)
        HU[:, k] = (gp - gm) / (2 * h)
        HV[:, k] = (vp - vm) / (2 * h)
    return 0.5 * (HU + HU.T), 0.5 * (HV + HV.T)


def axial_modes(ds: DiaphragmStatics, sol: StaticSolution, fluid: Fluid = WATER_20C,
                h_mesh: float = 0.005, gap_tol: float = 2e-3) -> AxialModes:
    """Axisymmetric (m = 0) modes excited by axial acceleration. DOFs per node:
    (u_r, u_z). The incompressible liquid makes the motion volume-preserving."""
    sh, tank, N = ds.shell, ds.d.tank, ds.shell.N
    rho = fluid.density
    HU, HV = axisymmetric_hessians(sh, sol.r, sol.z)
    K = HU - (-sol.dp) * HV
    m_nodes = ds.d.density * sh.t * sh.A0_node
    n = 2 * (N + 1)
    Ms = np.diag(np.repeat(m_nodes, 2))
    Gs = np.zeros(n)
    Gs[1::2] = m_nodes

    reg = _liquid_region(ds, sol, h_mesh, gap_tol)
    k_det, k_end = reg.k_det, reg.k_end
    hp = HarmonicPotential(reg.mesh, 0)
    G = np.zeros((len(hp.free), n))
    G[:, 2 * k_det:2 * (k_end + 1)] = hp.surface_load_matrix(reg.normals, 2, normal_dofs=(0, 1))
    X = hp.solve(G)
    Ma = rho * 2 * np.pi * (G.T @ X)
    Ma = 0.5 * (Ma + Ma.T)
    sl = slice(2 * k_det, 2 * (k_end + 1))
    GamL = np.zeros(n)
    GamL[sl] = rho * 2 * np.pi * surface_moment(reg.surf, reg.normals, 2, (0, 1), power_r=1, power_z=1)
    flux = np.zeros(n)
    flux[sl] = surface_moment(reg.surf, reg.normals, 2, (0, 1), power_r=1, power_z=0)

    M = Ma + Ms
    Gam = GamL + Gs
    rows = [np.eye(n)[0], np.eye(n)[1], np.eye(n)[2 * N], flux]
    rows += _wall_rows(ds, sol, reg.contact - {N}, 2, 0, 1)
    if N in reg.contact:
        rows.append(np.eye(n)[2 * N + 1])
    Z = null_space(np.array(rows))
    Kr = Z.T @ K @ Z
    Mr = Z.T @ M @ Z
    Kr, Mr = 0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T)
    w2, phi = eigh(Kr, Mr)
    Gn = phi.T @ (Z.T @ Gam)
    m_c = rho * tank.liquid_volume(sol.fill) + float(m_nodes.sum())
    return AxialModes(fill=float(sol.fill),
                      freq_hz=np.sqrt(np.abs(w2)) * np.sign(w2) / (2 * np.pi),
                      slosh_mass=Gn ** 2, contents_mass=m_c, min_eigenvalue=float(w2[0]))


def combined_emm(lat: LateralModes, ax: Optional[AxialModes], n_keep_lat: int = 3,
                 n_keep_ax: int = 2, zeta: float = 0.02) -> EMMParams:
    p = lat.emm(n_keep_lat, zeta)
    if ax is not None:
        p.axial_modes = ax.modes(n_keep_ax, zeta)
        p.notes += f"; axial modes: {len(p.axial_modes)}"
        p.validate()
    return p


def lateral_emm_table(ds: DiaphragmStatics, fills: Sequence[float], n_keep: int = 3,
                      zeta: float = 0.02, h_mesh: float = 0.005):
    sols = ds.sweep(fills)
    results: List[LateralModes] = [lateral_modes(ds, s, h_mesh=h_mesh) for s in sols]
    table = EMMTable(entries=[r.emm(n_keep, zeta) for r in results],
                     notes="Level B lateral slosh modes, baseline diaphragm tank")
    return table, results, sols


def emm_table(ds: DiaphragmStatics, fills: Sequence[float], n_keep_lat: int = 3,
              n_keep_ax: int = 2, zeta: float = 0.02, h_mesh: float = 0.005):
    """Lateral + axial Level A parameter table over fill. The number of axial modes
    kept is the same at every fill (EMMTable interpolation needs equal counts)."""
    sols = ds.sweep(fills)
    lat = [lateral_modes(ds, s, h_mesh=h_mesh) for s in sols]
    ax = [axial_modes(ds, s, h_mesh=h_mesh) for s in sols]
    n_ax = min(min(len(a.dominant(n_keep_ax)) for a in ax), n_keep_ax)
    table = EMMTable(entries=[combined_emm(lm, am, n_keep_lat, n_ax, zeta) for lm, am in zip(lat, ax)],
                     notes="Level B lateral + axial slosh modes, baseline diaphragm tank")
    return table, lat, ax, sols
