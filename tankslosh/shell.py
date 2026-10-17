"""Axisymmetric large-deformation shell: incompressible neo-Hookean membrane
energy plus Kirchhoff bending energy, discretized along the meridian.

Meridian nodes i = 0..N run from the clamp (node 0, fixed in position and
rotation) to the axis (node N, r = 0, tangent kept horizontal by symmetry).

Segment j joins nodes j and j+1:
    lambda1 = L / L0                    meridional stretch
    lambda2 = r_mid / r0_mid            hoop stretch
    theta   = atan2(dz, -dr)            tangent angle from the inward radial direction
Membrane energy (per reference area):  (mu t / 2)(l1^2 + l2^2 + 1/(l1 l2)^2 - 3)
Bending energy (per reference area):   (D / 2)(a^2 + b^2 + 2 nu a b),  D = mu t^3 / 3
with a, b the changes of meridional and hoop curvature from the molded state:
    kappa1 = d(theta)/ds0  (at nodes),   kappa2 = -sin(theta) / r.
Bending uses the molded thickness (bending matters only near the stress-free state).

The enclosed volume V_d is the volume between the diaphragm and the clamp plane,
positive when the diaphragm hangs below the clamp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_G2 = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class ShellState:
    r: np.ndarray
    z: np.ndarray


class AxisymmetricShell:
    def __init__(self, r0, z0, shear_modulus: float, thickness: float, nu: float = 0.5,
                 material: str = "neo-hookean", youngs_modulus: float = None,
                 bending: bool = True):
        """material: "neo-hookean" (incompressible, uses shear_modulus) or "linear"
        (small-strain linear-elastic membrane, energy E t/(2(1-nu^2)) (e1^2 + e2^2 +
        2 nu e1 e2) per molded area with e = stretch - 1; uses youngs_modulus and nu).
        bending=False drops the bending energy (pure membrane)."""
        r0 = np.asarray(r0, dtype=float)
        z0 = np.asarray(z0, dtype=float)
        if r0.shape != z0.shape or r0.ndim != 1 or len(r0) < 3:
            raise ValueError("r0, z0 must be 1-D arrays of equal length >= 3")
        if abs(r0[-1]) > 1e-12:
            raise ValueError("last node must lie on the axis (r = 0)")
        self.r0, self.z0 = r0, z0
        if material not in ("neo-hookean", "linear"):
            raise ValueError("material must be 'neo-hookean' or 'linear'")
        self.material = material
        if material == "linear":
            if youngs_modulus is None:
                raise ValueError("linear material needs youngs_modulus")
            self.E = youngs_modulus
            shear_modulus = youngs_modulus / (2 * (1 + nu))
        self.mu, self.t, self.nu = shear_modulus, thickness, nu
        if material == "linear":
            self.D = self.E * thickness ** 3 / (12 * (1 - nu ** 2))
        else:
            self.D = shear_modulus * thickness ** 3 / 3.0 if nu == 0.5 else \
                2 * shear_modulus * (1 + nu) * thickness ** 3 / (12 * (1 - nu ** 2))
        if not bending:
            self.D = 0.0
        self.gravity = None
        self.N = len(r0) - 1

        dr, dz = np.diff(r0), np.diff(z0)
        self.L0 = np.hypot(dr, dz)
        self.rm0 = 0.5 * (r0[:-1] + r0[1:])
        self.A0_seg = 2 * np.pi * self.rm0 * self.L0
        self.c_mem = 0.5 * self.mu * self.t * self.A0_seg
        if material == "linear":
            self.c_lin = 0.5 * self.E * self.t / (1 - nu ** 2) * self.A0_seg
        self.theta_clamp = np.arctan2(dz[0], -dr[0])
        # node lengths and areas for bending
        s0 = np.empty(self.N + 1)
        s0[0] = 0.5 * self.L0[0]
        s0[1:-1] = 0.5 * (self.L0[:-1] + self.L0[1:])
        s0[-1] = self.L0[-1]
        self.s0 = s0
        A0n = 2 * np.pi * r0 * s0
        A0n[-1] = np.pi * (0.5 * self.L0[-1]) ** 2
        self.A0_node = A0n
        self.k1_ref, self.k2_ref = self._curvatures(r0, z0)[:2]
        self.area0 = self.A0_seg.sum()

    # ---------- kinematics ----------
    def _curvatures(self, r, z):
        dr, dz = np.diff(r), np.diff(z)
        th = np.arctan2(dz, -dr)
        N = self.N
        dth = np.empty(N + 1)
        dth[0] = _wrap(th[0] - self.theta_clamp)
        dth[1:-1] = _wrap(th[1:] - th[:-1])
        dth[-1] = _wrap(-2 * th[-1])
        k1 = dth / self.s0
        thbar = np.empty(N + 1)
        thbar[0] = th[0]
        thbar[1:-1] = th[:-1] + 0.5 * dth[1:-1]
        thbar[-1] = 0.0
        k2 = np.empty(N + 1)
        k2[:-1] = -np.sin(thbar[:-1]) / r[:-1]
        k2[-1] = k1[-1]
        return k1, k2, th, thbar, dr, dz

    def stretches(self, r, z):
        L = np.hypot(np.diff(r), np.diff(z))
        l1 = L / self.L0
        l2 = 0.5 * (r[:-1] + r[1:]) / self.rm0
        return l1, l2

    def tensions(self, r, z):
        """Meridional and hoop membrane tensions (N/m, per current length) per segment."""
        l1, l2 = self.stretches(r, z)
        if self.material == "linear":
            c = self.E * self.t / (1 - self.nu ** 2)
            e1, e2 = l1 - 1, l2 - 1
            return c * (e1 + self.nu * e2) / l2, c * (e2 + self.nu * e1) / l1
        T1 = self.mu * self.t * (l1 / l2 - 1.0 / (l1 ** 3 * l2 ** 3))
        T2 = self.mu * self.t * (l2 / l1 - 1.0 / (l1 ** 3 * l2 ** 3))
        return T1, T2

    def set_gravity(self, g: float, rho_liquid: float, rho_shell: float):
        """Include gravity along -z: hydrostatic liquid below the diaphragm (potential
        rho_liquid g int z dV of the liquid) and the diaphragm's own weight."""
        self.gravity = (g, rho_liquid, rho_shell * self.t * self.A0_node)

    def _membrane(self, l1, l2):
        """Per-segment membrane energy and its derivatives w.r.t. the stretches."""
        if self.material == "linear":
            e1, e2 = l1 - 1, l2 - 1
            c = self.c_lin
            W = c * (e1 * e1 + e2 * e2 + 2 * self.nu * e1 * e2)
            return W, 2 * c * (e1 + self.nu * e2), 2 * c * (e2 + self.nu * e1)
        inv = 1.0 / (l1 * l1 * l2 * l2)
        c = self.c_mem
        return (c * (l1 * l1 + l2 * l2 + inv - 3.0), c * (2 * l1 - 2 * inv / l1),
                c * (2 * l2 - 2 * inv / l2))

    # ---------- energy and gradient ----------
    def energy(self, r, z, grad: bool = False):
        k1, k2, th, thbar, dr, dz = self._curvatures(r, z)
        L2 = dr ** 2 + dz ** 2
        L = np.sqrt(L2)
        l1 = L / self.L0
        rm = 0.5 * (r[:-1] + r[1:])
        l2 = rm / self.rm0
        Wm, dW_dl1, dW_dl2 = self._membrane(l1, l2)
        Um = np.sum(Wm)

        a = k1 - self.k1_ref
        b = k2 - self.k2_ref
        nu, D, A = self.nu, self.D, self.A0_node
        ub = 0.5 * D * A * (a * a + b * b + 2 * nu * a * b)
        ub[-1] = 0.5 * D * A[-1] * (2 + 2 * nu) * a[-1] ** 2
        U = Um + ub.sum()
        if self.gravity is not None:
            g, rho_l, m_nodes = self.gravity
            U = U - rho_l * g * self.enclosed_zmoment(r, z) + g * float(m_nodes @ z)
        if not grad:
            return U

        N = self.N
        gr = np.zeros(N + 1)
        gz = np.zeros(N + 1)
        # membrane
        dW_dL = dW_dl1 / self.L0
        g_dr = dW_dL * dr / L
        g_dz = dW_dL * dz / L
        gr[1:] += g_dr
        gr[:-1] -= g_dr
        gz[1:] += g_dz
        gz[:-1] -= g_dz
        g_rm = dW_dl2 / self.rm0 * 0.5
        gr[:-1] += g_rm
        gr[1:] += g_rm

        # bending: derivatives wrt a (via dtheta) and b (via thbar and r)
        dU_da = D * A * (a + nu * b)
        dU_db = D * A * (b + nu * a)
        dU_da[-1] = D * A[-1] * (2 + 2 * nu) * a[-1]
        dU_db[-1] = 0.0
        g_th = np.zeros(N)
        c1 = dU_da / self.s0          # dU/d(dtheta_i)
        g_th[0] += c1[0]
        g_th[1:] += c1[1:-1]
        g_th[:-1] -= c1[1:-1]
        g_th[-1] += -2 * c1[-1]
        # b_i = -sin(thbar_i)/r_i, i < N
        db_dthbar = -np.cos(thbar[:-1]) / r[:-1]
        db_dr = np.sin(thbar[:-1]) / r[:-1] ** 2
        cb = dU_db[:-1] * db_dthbar
        g_th[0] += cb[0]
        g_th[:-1] += 0.5 * cb[1:]
        g_th[1:] += 0.5 * cb[1:]
        gr[:-1] += dU_db[:-1] * db_dr
        # theta wrt segment vector
        dth_ddr = dz / L2
        dth_ddz = -dr / L2
        gr[1:] += g_th * dth_ddr
        gr[:-1] -= g_th * dth_ddr
        gz[1:] += g_th * dth_ddz
        gz[:-1] -= g_th * dth_ddz
        if self.gravity is not None:
            g, rho_l, m_nodes = self.gravity
            _, zr, zz = self.enclosed_zmoment(r, z, grad=True)
            gr -= rho_l * g * zr
            gz += -rho_l * g * zz + g * m_nodes
        return U, gr, gz

    @staticmethod
    def enclosed_zmoment(r, z, grad: bool = False):
        """First moment int z dV of the region between the diaphragm and the clamp
        plane (same sign convention as enclosed_volume)."""
        ra, rb, za, zb = r[:-1], r[1:], z[:-1], z[1:]
        dz = zb - za
        val = 0.0
        if grad:
            gr = np.zeros_like(r)
            gz = np.zeros_like(z)
        for t in _G2:
            rg = ra + t * (rb - ra)
            zg = za + t * dz
            val = val - np.pi * 0.5 * np.sum(rg * rg * zg * dz)
            if grad:
                c = -np.pi * 0.5
                gr[:-1] += c * 2 * rg * zg * dz * (1 - t)
                gr[1:] += c * 2 * rg * zg * dz * t
                gz[:-1] += c * (rg * rg * (1 - t) * dz - rg * rg * zg)
                gz[1:] += c * (rg * rg * t * dz + rg * rg * zg)
        return (val, gr, gz) if grad else val

    # ---------- volume ----------
    @staticmethod
    def enclosed_volume(r, z, grad: bool = False):
        q = r[:-1] ** 2 + r[:-1] * r[1:] + r[1:] ** 2
        h = z[:-1] - z[1:]
        V = np.pi / 3 * np.sum(q * h)
        if not grad:
            return V
        gr = np.zeros_like(r)
        gz = np.zeros_like(z)
        gr[:-1] += np.pi / 3 * (2 * r[:-1] + r[1:]) * h
        gr[1:] += np.pi / 3 * (r[:-1] + 2 * r[1:]) * h
        gz[:-1] += np.pi / 3 * q
        gz[1:] -= np.pi / 3 * q
        return V, gr, gz


def self_intersections(r, z):
    """Pairs of non-adjacent meridian segments that cross (diaphragm passing through itself)."""
    p = np.column_stack([r, z])
    a, b = p[:-1], p[1:]
    hits = []
    n = len(a)
    for i in range(n - 2):
        d1 = b[i] - a[i]
        j = np.arange(i + 2, n)
        d2 = b[j] - a[j]
        den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
        ok = np.abs(den) > 1e-18
        w = a[j] - a[i]
        s = np.where(ok, (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (w[:, 0] * d1[1] - w[:, 1] * d1[0]) / np.where(ok, den, 1), -1)
        m = ok & (s > 0) & (s < 1) & (u > 0) & (u < 1)
        hits += [(i, int(k)) for k in j[m]]
    return hits
