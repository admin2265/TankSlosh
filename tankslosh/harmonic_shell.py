"""Stiffness, mass and translation coupling of the shell for lateral (m = 1)
perturbations about an axisymmetric static state.

Each meridian node i gets a harmonic displacement
    u = u_r,i cos(t) e_r + u_t,i sin(t) e_t + u_z,i cos(t) e_z
which, for a lateral (x-direction) motion, is the only pattern excited.
The 3D shell energy of the perturbed surface is integrated exactly over the
azimuth t with a uniform rule (the second variation contains only low
harmonics), and its Hessian is built element by element with central finite
differences of the local energies. The same membrane and bending measures as
tankslosh.shell are used, so the unperturbed 3D energy equals the
axisymmetric one.

DOF layout: 3 per node, [u_r, u_t, u_z], nodes 0..N (node 0 = clamp).
"""
from __future__ import annotations

import numpy as np

from .shell import AxisymmetricShell

N_THETA = 8
_GAUSS2 = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class HarmonicShell:
    def __init__(self, shell: AxisymmetricShell, r, z, n_theta: int = N_THETA):
        self.sh = shell
        self.r = np.asarray(r, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.N = shell.N
        self.K = n_theta
        th = 2 * np.pi * np.arange(n_theta) / n_theta
        self.c, self.s = np.cos(th), np.sin(th)
        k = np.fft.fftfreq(n_theta, 1.0 / n_theta)
        k[n_theta // 2] = 0.0
        self._ik = 1j * k
        # clamped tangent of the molded first segment, as (r, z) components
        d = np.array([shell.r0[1] - shell.r0[0], shell.z0[1] - shell.z0[0]])
        self.t_clamp = d / np.linalg.norm(d)
        self.kss_ref, self.ktt_ref = self._ref_curvatures()

    # ---------- geometry ----------
    def positions(self, r, z, ur, ut, uz):
        """Cartesian position and its first/second azimuth derivatives, shape (..., K, 3)."""
        c, s = self.c, self.s
        R, Z = np.asarray(r)[..., None], np.asarray(z)[..., None]
        Ur, Ut, Uz = (np.asarray(a)[..., None] for a in (ur, ut, uz))
        w = Ur + Ut
        p = np.stack([R * c + Ur * c * c - Ut * s * s, R * s + w * s * c, Z + Uz * c], -1)
        dp = np.stack([-R * s - 2 * w * s * c, R * c + w * (c * c - s * s), -Uz * s], -1)
        ddp = np.stack([-R * c - 2 * w * (c * c - s * s), -R * s - 4 * w * s * c, -Uz * c], -1)
        return p, dp, ddp

    def _dtheta(self, f):
        return np.real(np.fft.ifft(self._ik[:, None] * np.fft.fft(f, axis=-2), axis=-2))

    def _clamp_tangent(self):
        tr, tz = self.t_clamp
        return np.stack([tr * self.c, tr * self.s, np.full(self.K, tz)], -1)

    def _curvatures(self, r, z, V):
        """Per node: meridional curvature (rotation per molded length), hoop normal
        curvature and twist, each shape (N+1, K). V: (N+1, 9) local vars
        [prev node, node, next node] x [u_r, u_t, u_z]."""
        N = self.N
        i = np.arange(N + 1)
        ip, inx = np.clip(i - 1, 0, N), np.clip(i + 1, 0, N)
        pp = self.positions(r[ip], z[ip], V[:, 0], V[:, 1], V[:, 2])[0]
        pi, dpi, ddpi = self.positions(r, z, V[:, 3], V[:, 4], V[:, 5])
        pn = self.positions(r[inx], z[inx], V[:, 6], V[:, 7], V[:, 8])[0]
        t_prev = np.empty_like(pi)
        t_next = np.empty_like(pi)
        t_prev[1:] = _unit(pi[1:] - pp[1:])
        t_prev[0] = self._clamp_tangent()
        t_next[:-1] = _unit(pn[:-1] - pi[:-1])
        # axis: the meridian continues through the axis along the opposite meridian
        t_next[-1] = -np.roll(t_prev[-1], -self.K // 2, axis=0)
        # hoop direction; on the axis use the hoop direction of the azimuth
        eh = np.empty_like(pi)
        eh[:-1] = _unit(dpi[:-1])
        eh[-1] = np.stack([-self.s, self.c, np.zeros(self.K)], -1)
        cr = np.cross(t_prev, t_next)
        kss = np.arctan2(-np.sum(cr * eh, -1), np.sum(t_prev * t_next, -1)) / self.sh.s0[:, None]
        tav = t_prev + t_next
        # at the clamp the axisymmetric model takes the normal from the first segment
        # alone (not the bisector with the clamped tangent); match it exactly
        tav[0] = t_next[0]
        n = _unit(np.cross(tav, eh))
        dp2 = np.sum(dpi * dpi, -1)
        ktt = np.zeros_like(kss)
        ktt[:-1] = np.sum(ddpi[:-1] * n[:-1], -1) / dp2[:-1]
        T = _unit(tav)
        kst = np.zeros_like(kss)
        kst[:-1] = np.sum(self._dtheta(T)[:-1] * n[:-1], -1) / np.sqrt(dp2[:-1])
        return kss, ktt, kst

    def _ref_curvatures(self):
        V0 = np.zeros((self.N + 1, 9))
        kss, ktt, _ = self._curvatures(self.sh.r0, self.sh.z0, V0)
        return kss, ktt

    # ---------- local energies (vectorized over elements) ----------
    def membrane_local(self, V):
        """V: (N, 6) vars of segment end nodes -> (N,) energies."""
        sh = self.sh
        pj, dpj, _ = self.positions(self.r[:-1], self.z[:-1], V[:, 0], V[:, 1], V[:, 2])
        pk, dpk, _ = self.positions(self.r[1:], self.z[1:], V[:, 3], V[:, 4], V[:, 5])
        a = (pk - pj) / sh.L0[:, None, None]
        b = 0.5 * (dpj + dpk) / sh.rm0[:, None, None]
        aa, bb, ab = np.sum(a * a, -1), np.sum(b * b, -1), np.sum(a * b, -1)
        det = aa * bb - ab * ab
        if sh.material == "linear":
            # principal stretches enter only through S = l1 + l2 and P = l1 l2
            P = np.sqrt(det)
            S = np.sqrt(aa + bb + 2 * P)
            sq = S * S - 2 * P - 2 * S + 2          # e1^2 + e2^2
            pr = P - S + 1                          # e1 e2
            W = 0.5 * sh.E * sh.t / (1 - sh.nu ** 2) * (sq + 2 * sh.nu * pr)
        else:
            W = 0.5 * sh.mu * sh.t * (aa + bb + 1.0 / det - 3.0)
        return W.mean(-1) * sh.A0_seg

    def bending_local(self, V):
        """V: (N+1, 9) -> (N+1,) energies."""
        sh = self.sh
        kss, ktt, kst = self._curvatures(self.r, self.z, V)
        a = kss - self.kss_ref
        b = ktt - self.ktt_ref
        nu = sh.nu
        e = 0.5 * sh.D * (a * a + b * b + 2 * nu * a * b + 2 * (1 - nu) * kst * kst)
        e[-1] = 0.5 * sh.D * (2 + 2 * nu) * a[-1] ** 2
        return e.mean(-1) * sh.A0_node

    def volume_local(self, V):
        """Volume between the strip of segment j and the axis-closing cone, (N,);
        the sum plus the clamp disc term is the enclosed volume V_d."""
        pj, dpj, _ = self.positions(self.r[:-1], self.z[:-1], V[:, 0], V[:, 1], V[:, 2])
        pk, dpk, _ = self.positions(self.r[1:], self.z[1:], V[:, 3], V[:, 4], V[:, 5])
        Xs = pk - pj
        tot = 0.0
        for g in _GAUSS2:
            X = (1 - g) * pj + g * pk
            Xt = (1 - g) * dpj + g * dpk
            tot = tot + 0.5 * np.sum(X * np.cross(Xs, Xt), -1)
        return (2 * np.pi / 3) * tot.mean(-1)

    def zmoment_local(self, V):
        """First moment int z dV of the strip region (divergence form, same closure
        as volume_local), (N,)."""
        pj, dpj, _ = self.positions(self.r[:-1], self.z[:-1], V[:, 0], V[:, 1], V[:, 2])
        pk, dpk, _ = self.positions(self.r[1:], self.z[1:], V[:, 3], V[:, 4], V[:, 5])
        Xs = pk - pj
        tot = 0.0
        for g in _GAUSS2:
            X = (1 - g) * pj + g * pk
            Xt = (1 - g) * dpj + g * dpk
            tot = tot + 0.5 * 0.5 * X[..., 2] ** 2 * np.cross(Xs, Xt)[..., 2]
        return 2 * np.pi * tot.mean(-1)

    def enclosed_zmoment(self, u=None):
        u = np.zeros(3 * (self.N + 1)) if u is None else u
        U = u.reshape(-1, 3)
        V = np.hstack([U[:-1], U[1:]])
        disc = 0.5 * self.z[0] ** 2 * np.pi * self.r[0] ** 2
        return float(self.zmoment_local(V).sum() + disc)

    def enclosed_volume(self, u=None):
        """3D enclosed volume V_d for a full DOF vector u (3(N+1),)."""
        u = np.zeros(3 * (self.N + 1)) if u is None else u
        U = u.reshape(-1, 3)
        V = np.hstack([U[:-1], U[1:]])
        disc = np.pi * self.r[0] ** 2 * self.z[0] / 3.0
        return float(self.volume_local(V).sum() + disc)

    def energy(self, u=None):
        u = np.zeros(3 * (self.N + 1)) if u is None else u
        U = u.reshape(-1, 3)
        Vm = np.hstack([U[:-1], U[1:]])
        N = self.N
        i = np.arange(N + 1)
        Vb = np.hstack([U[np.clip(i - 1, 0, N)], U, U[np.clip(i + 1, 0, N)]])
        Vb[0, :3] = 0.0
        Vb[-1, 6:] = 0.0
        U_el = float(self.membrane_local(Vm).sum() + self.bending_local(Vb).sum())
        if self.sh.gravity is not None:   # rubber weight is linear in u_z cos t: no m = 1 term
            g, rho_l, m_nodes = self.sh.gravity
            U_el += -rho_l * g * self.enclosed_zmoment(u) + g * float(m_nodes @ self.z)
        return U_el

    # ---------- Hessians ----------
    @staticmethod
    def _local_hessian(f, n_el, n_loc, h):
        """Central-difference Hessians of the local energies, Richardson-extrapolated
        (steps h and 2h) to remove the O(h^2) error. That error matters for slack
        membranes, whose stretching energy is quartic in the deflection."""
        H = np.zeros((n_el, n_loc, n_loc))
        base = np.zeros((n_el, n_loc))
        for k in range(n_loc):
            for l in range(k, n_loc):
                def d2(step):
                    def ev(sk, sl):
                        V = base.copy()
                        V[:, k] += sk * step
                        V[:, l] += sl * step
                        return f(V)
                    return (ev(1, 1) - ev(1, -1) - ev(-1, 1) + ev(-1, -1)) / (4 * step * step)
                val = (4 * d2(h) - d2(2 * h)) / 3
                H[:, k, l] = val
                H[:, l, k] = val
        return H

    def hessians(self, h: float = 1e-6):
        """Global Hessians (3(N+1) square) of the potential energy (shell, plus the
        hydrostatic liquid term if gravity is set) and of V_d."""
        N = self.N
        n = 3 * (N + 1)
        HU = np.zeros((n, n))
        HV = np.zeros((n, n))
        Hm = self._local_hessian(self.membrane_local, N, 6, h)
        Hv = self._local_hessian(self.volume_local, N, 6, h)
        if self.sh.gravity is not None:
            g, rho_l, _ = self.sh.gravity
            Hm = Hm - rho_l * g * self._local_hessian(self.zmoment_local, N, 6, h)
        for j in range(N):
            idx = np.arange(3 * j, 3 * j + 6)
            HU[np.ix_(idx, idx)] += Hm[j]
            HV[np.ix_(idx, idx)] += Hv[j]
        Hb = self._local_hessian(self.bending_local, N + 1, 9, h)
        for i in range(N + 1):
            loc = np.arange(9)
            glob = np.concatenate([np.arange(3 * (i - 1), 3 * i), np.arange(3 * i, 3 * i + 3),
                                   np.arange(3 * (i + 1), 3 * i + 6)])
            keep = np.ones(9, dtype=bool)
            if i == 0:
                keep[:3] = False
            if i == N:
                keep[6:] = False
            g, l_ = glob[keep], loc[keep]
            HU[np.ix_(g, g)] += Hb[i][np.ix_(l_, l_)]
        return HU, HV

    # ---------- inertia of the rubber ----------
    def rubber_mass(self, density: float):
        """Lumped ring masses per node (kg); harmonic mass matrix, x-momentum and
        y-angular-momentum coupling vectors for the m = 1 pattern."""
        m = density * self.sh.t * self.sh.A0_node
        M = np.repeat(m / 2.0, 3)
        Gam = np.zeros(3 * (self.N + 1))
        Hy = np.zeros(3 * (self.N + 1))
        Gam[0::3] = m / 2.0
        Gam[1::3] = -m / 2.0
        Hy[0::3] = m / 2.0 * self.z
        Hy[1::3] = -m / 2.0 * self.z
        Hy[2::3] = -m / 2.0 * self.r
        return m, np.diag(M), Gam, Hy
