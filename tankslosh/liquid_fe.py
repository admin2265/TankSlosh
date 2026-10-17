"""Potential flow of the liquid for one azimuthal harmonic (finite elements).

The liquid region is the area in the (r, z) half-plane bounded by
  * the tank wall, from the bottom of the axis up to the point where the
    diaphragm leaves the wall,
  * the wetted diaphragm, from that point to the axis,
  * the axis.
It is meshed with linear triangles (Delaunay of boundary + interior points,
triangles outside the region discarded).

For the harmonic m the velocity potential is phi(r, z) cos(m t). Its energy
form, per unit of the azimuth integral, is
    a(phi, psi) = int (phi_r psi_r + phi_z psi_z + m^2 phi psi / r^2) r dr dz.
A normal velocity w_n cos(m t) of the wetted surface enters as the Neumann
load int w_n psi r ds. With phi = 0 on the axis (m >= 1) the problem is
regular. For m = 0 the free constant is fixed at one node and the load must
be volume-preserving (zero net flux). The azimuth integral of cos^2(m t) is
pi for m = 1 and 2 pi for m = 0, so for m = 1
    liquid kinetic energy = 1/2 rho pi  w_dot^T G^T A^-1 G w_dot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay

_TRI3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
_G2 = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _seg_distance(P, A, B):
    """Distance from points P (n,2) to segments A-B (m,2): min over segments."""
    AB = B - A
    L2 = np.maximum(np.sum(AB * AB, 1), 1e-30)
    AP = P[:, None, :] - A[None, :, :]
    t = np.clip(np.sum(AP * AB[None], -1) / L2[None], 0, 1)
    d = AP - t[..., None] * AB[None]
    return np.sqrt(np.min(np.sum(d * d, -1), 1))


@dataclass
class LiquidMesh:
    nodes: np.ndarray        # (n, 2) r, z
    tris: np.ndarray         # (nt, 3)
    surf_nodes: np.ndarray   # FE node index of each wetted-surface point, in surface order
    axis_nodes: np.ndarray   # FE nodes on r = 0
    polygon: np.ndarray      # boundary loop (counter-clockwise)

    @property
    def area(self) -> float:
        p = self.nodes[self.tris]
        return float(0.5 * np.abs(_cross2(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])).sum())

    def volume_and_cg(self):
        p = self.nodes[self.tris]
        A = 0.5 * np.abs(_cross2(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
        q = np.einsum("qa,tad->tqd", _TRI3, p)
        r, z = q[..., 0], q[..., 1]
        V = 2 * np.pi * np.sum(A[:, None] * r / 3)
        Mz = 2 * np.pi * np.sum(A[:, None] * r * z / 3)
        return float(V), float(Mz / V)


def build_mesh(wall: np.ndarray, surface: np.ndarray, h: float) -> LiquidMesh:
    """wall: (nw, 2) points from the bottom of the axis (r = 0) to the attachment
    point; surface: (ns, 2) wetted-surface points from the attachment point
    (= wall[-1]) to the axis (r = 0). h: interior point spacing."""
    wall = np.asarray(wall, float)
    surface = np.asarray(surface, float)
    if np.linalg.norm(wall[-1] - surface[0]) > 1e-12:
        raise ValueError("surface must start at the end of the wall polyline")
    closed = np.linalg.norm(wall[0] - surface[-1]) < 1e-9   # surface ends where the wall starts
    if closed:
        loop = np.vstack([wall, surface[1:-1]])
    else:
        if abs(wall[0, 0]) > 1e-12 or abs(surface[-1, 0]) > 1e-12:
            raise ValueError("wall must start and surface must end on the axis, "
                             "or the surface must end at the start of the wall")
        z_top, z_bot = surface[-1, 1], wall[0, 1]
        n_ax = max(int(np.ceil(abs(z_top - z_bot) / h)), 1)
        axis = np.column_stack([np.zeros(n_ax - 1), np.linspace(z_top, z_bot, n_ax + 1)[1:-1]])
        loop = np.vstack([wall, surface[1:], axis])
    signed = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    if signed <= 0:
        raise ValueError("liquid boundary loop is not counter-clockwise")
    # interior points on a staggered grid, away from the boundary
    rmin, zmin = loop.min(0)
    rmax, zmax = loop.max(0)
    gr = np.arange(rmin + 0.5 * h, rmax, h)
    gz = np.arange(zmin + 0.5 * h, zmax, h * np.sqrt(3) / 2)
    R, Z = np.meshgrid(gr, gz)
    R = R + (np.arange(len(gz))[:, None] % 2) * 0.5 * h
    cand = np.column_stack([R.ravel(), Z.ravel()])
    path = Path(np.vstack([loop, loop[:1]]))
    cand = cand[path.contains_points(cand)]
    if len(cand):
        d = _seg_distance(cand, loop, np.roll(loop, -1, 0))
        cand = cand[d > 0.6 * h]
    pts = np.vstack([loop, cand])
    tri = Delaunay(pts).simplices
    P = pts[tri]
    cen = P.mean(1)
    area = 0.5 * _cross2(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    keep = path.contains_points(cen) & (np.abs(area) > 1e-14)
    tri = tri[keep]
    # orient counter-clockwise
    P = pts[tri]
    area = 0.5 * _cross2(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    tri[area < 0] = tri[area < 0][:, [0, 2, 1]]
    nw = len(wall)
    used = np.zeros(len(pts), dtype=bool)
    used[tri.ravel()] = True
    if not used[: len(loop)].all():
        raise ValueError("liquid mesh lost boundary points (region thinner than the mesh?)")
    if not used.all():   # drop unused interior points
        remap = -np.ones(len(pts), dtype=int)
        remap[used] = np.arange(used.sum())
        pts, tri = pts[used], remap[tri]
    surf_nodes = np.arange(nw - 1, nw - 1 + len(surface))
    if closed:
        surf_nodes[-1] = 0
    axis_nodes = np.where(np.abs(pts[:, 0]) < 1e-12)[0]
    return LiquidMesh(nodes=pts, tris=tri, surf_nodes=surf_nodes, axis_nodes=axis_nodes,
                      polygon=loop)


class HarmonicPotential:
    """Assembled potential-flow operator for harmonic m on a LiquidMesh."""

    def __init__(self, mesh: LiquidMesh, m: int = 1):
        if m < 0:
            raise ValueError("m must be >= 0")
        self.mesh, self.m = mesh, m
        p = mesh.nodes[mesh.tris]                       # (nt, 3, 2)
        x1, x2, x3 = p[:, 0], p[:, 1], p[:, 2]
        det = _cross2(x2 - x1, x3 - x1)
        A = 0.5 * det
        # gradients of the barycentric functions
        g = np.empty((len(p), 3, 2))
        g[:, 0] = np.column_stack([x2[:, 1] - x3[:, 1], x3[:, 0] - x2[:, 0]]) / det[:, None]
        g[:, 1] = np.column_stack([x3[:, 1] - x1[:, 1], x1[:, 0] - x3[:, 0]]) / det[:, None]
        g[:, 2] = np.column_stack([x1[:, 1] - x2[:, 1], x2[:, 0] - x1[:, 0]]) / det[:, None]
        q = np.einsum("qa,tad->tqd", _TRI3, p)          # quadrature points (nt, 3, 2)
        rq, zq = q[..., 0], q[..., 1]
        w = A[:, None] / 3.0
        GG = np.einsum("tad,tbd->tab", g, g)
        Ke = GG * np.sum(w * rq, 1)[:, None, None] + \
            m * m * np.einsum("qa,qb,tq->tab", _TRI3, _TRI3, w / rq)
        n = len(mesh.nodes)
        I = np.repeat(mesh.tris, 3, axis=1).ravel()
        J = np.tile(mesh.tris, (1, 3)).ravel()
        Kfull = coo_matrix((Ke.ravel(), (I, J)), shape=(n, n)).tocsc()
        free = np.ones(n, dtype=bool)
        if m >= 1:
            free[mesh.axis_nodes] = False   # phi = 0 on the axis
        else:
            free[0] = False                  # m = 0: fix the free constant (gauge)
        self.free = np.where(free)[0]
        self.pos = -np.ones(n, dtype=int)
        self.pos[self.free] = np.arange(len(self.free))
        self.A = csc_matrix(Kfull[self.free][:, self.free])
        self.lu = splu(self.A)
        # angular-momentum weights: c_p = int [z (psi_r + psi/r) - r psi_z] r dA  (m = 1)
        cw = np.einsum("tq,tq->t", w, zq * rq)[:, None] * g[..., 0] + \
            np.einsum("qa,tq->ta", _TRI3, w * zq) - \
            np.einsum("tq->t", w * rq * rq)[:, None] * g[..., 1]
        c = np.zeros(n)
        np.add.at(c, mesh.tris.ravel(), cw.ravel())
        self.c = c[self.free]

    def surface_load_matrix(self, normals_seg: np.ndarray, n_dof_per_node: int,
                            normal_dofs=(0, 2)):
        """G: (n_free, n_surf * n_dof_per_node) mapping surface-node displacements to
        Neumann loads. normals_seg: (ns-1, 2) outward (r, z) unit normals of the
        wetted segments; normal_dofs: which nodal DOFs are (u_r, u_z)."""
        mesh = self.mesh
        sp = mesh.nodes[mesh.surf_nodes]
        ns = len(sp)
        rows, cols, vals = [], [], []
        for j in range(ns - 1):
            a, b = sp[j], sp[j + 1]
            L = np.linalg.norm(b - a)
            nr, nz = normals_seg[j]
            for g in _G2:
                Na, Nb = 1 - g, g
                rr = Na * a[0] + Nb * b[0]
                for fe_node, Nf in ((mesh.surf_nodes[j], Na), (mesh.surf_nodes[j + 1], Nb)):
                    pf = self.pos[fe_node]
                    if pf < 0:
                        continue
                    for node, Nu in ((j, Na), (j + 1, Nb)):
                        for comp, nc in zip(normal_dofs, (nr, nz)):
                            rows.append(pf)
                            cols.append(node * n_dof_per_node + comp)
                            vals.append(0.5 * L * Nf * rr * Nu * nc)
        return coo_matrix((vals, (rows, cols)),
                          shape=(len(self.free), ns * n_dof_per_node)).toarray()

    def solve(self, F):
        return self.lu.solve(F)


def surface_normals(surface: np.ndarray) -> np.ndarray:
    """Outward (from the liquid) unit normals of surface segments, for a surface
    traversed in the counter-clockwise boundary direction (attachment -> axis)."""
    d = np.diff(surface, axis=0)
    n = np.column_stack([d[:, 1], -d[:, 0]])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def surface_moment(surface: np.ndarray, normals: np.ndarray, n_dof_per_node: int,
                   normal_dofs=(0, 2), power_r: int = 1, power_z: int = 0) -> np.ndarray:
    """Per surface DOF: int r^power_r z^power_z w_n ds over the wetted surface.
    (power_r=1, power_z=0): volume flux / 2 pi;  (1, 1): axial first moment / 2 pi."""
    ns = len(surface)
    out = np.zeros(ns * n_dof_per_node)
    for j in range(ns - 1):
        a, b = surface[j], surface[j + 1]
        L = np.linalg.norm(b - a)
        for g in _G2:
            rr = (1 - g) * a[0] + g * b[0]
            zz = (1 - g) * a[1] + g * b[1]
            wgt = 0.5 * L * rr ** power_r * zz ** power_z
            for node, Nu in ((j, 1 - g), (j + 1, g)):
                for comp, nc in zip(normal_dofs, normals[j]):
                    out[node * n_dof_per_node + comp] += wgt * Nu * nc
    return out


def translation_coupling(surface: np.ndarray, normals: np.ndarray, n_dof_per_node: int,
                         normal_dofs=(0, 2)) -> np.ndarray:
    """Gamma_L / (rho pi): int r^2 w_n ds per surface DOF (m = 1 first moment)."""
    ns = len(surface)
    out = np.zeros(ns * n_dof_per_node)
    for j in range(ns - 1):
        a, b = surface[j], surface[j + 1]
        L = np.linalg.norm(b - a)
        for g in _G2:
            rr = (1 - g) * a[0] + g * b[0]
            for node, Nu in ((j, 1 - g), (j + 1, g)):
                for comp, nc in zip(normal_dofs, normals[j]):
                    out[node * n_dof_per_node + comp] += 0.5 * L * rr * rr * Nu * nc
    return out
