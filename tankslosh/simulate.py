"""Time-domain simulation of the spring-mass slosh model (translation only).

Conventions (tank frame, axes x, y lateral, z along the tank axis):
  specific_force f(t)  tank acceleration minus gravity, i.e. what an
                       accelerometer rigidly mounted on the tank reads.
                       In orbit this equals the tank acceleration. In a 1-g
                       ground test at rest f = (0, 0, +g).
  x_n, y_n, z_n        oscillator displacement relative to the tank.
  interface_force      force the liquid exerts on the tank (N).
  interface_moment     moment of that force about ref_point (N m), e.g. the
                       tank mount / accelerometer location.
  liquid_cg            liquid CG in the tank frame (m), to first order.

Equation of each oscillator:  x_n'' + 2 zeta_n w_n x_n' + w_n^2 x_n = -f.
Liquid force on the tank:     F = -m0 f + sum(k_n x_n + c_n x_n')
                                = -m f - sum(m_n x_n'').

Two drivers:
  simulate_prescribed : f(t) is given (e.g. a measured or CFD-imposed motion).
  simulate_coupled    : an external force F_ext(t) acts on a rigid dry
                        structure of mass M_dry carrying the tank, and the
                        structure acceleration (accelerometer reading) comes
                        out of the coupled dynamics:
                            f = (F_ext + sum(k_n x_n + c_n x_n')) / (M_dry + m0)

The model is linear and time-invariant, so it is integrated exactly for
inputs that vary linearly between samples (first-order hold).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.linalg import block_diag, expm

from .emm import EMMParams, SloshMode


@dataclass
class SimResult:
    t: np.ndarray                  # (N,)
    specific_force: np.ndarray     # (N, 3) accelerometer reading, m/s^2
    interface_force: np.ndarray    # (N, 3) liquid force on tank, N
    interface_moment: np.ndarray   # (N, 3) about ref_point, N m
    liquid_cg: np.ndarray          # (N, 3) m, tank frame
    lateral_disp: np.ndarray       # (N, n_lat, 2) oscillator x, y displacements, m
    axial_disp: np.ndarray         # (N, n_ax) oscillator z displacements, m
    external_force: Optional[np.ndarray] = None  # (N, 3), coupled runs only
    ref_point: tuple = (0.0, 0.0, 0.0)           # moment reference point, m

    def to_csv(self, path) -> None:
        cols = {"t": self.t}
        for i, ax in enumerate("xyz"):
            cols[f"f_{ax}"] = self.specific_force[:, i]
        for i, ax in enumerate("xyz"):
            cols[f"F_{ax}"] = self.interface_force[:, i]
        for i, ax in enumerate("xyz"):
            cols[f"M_{ax}"] = self.interface_moment[:, i]
        for i, ax in enumerate("xyz"):
            cols[f"cg_{ax}"] = self.liquid_cg[:, i]
        if self.external_force is not None:
            for i, ax in enumerate("xyz"):
                cols[f"Fext_{ax}"] = self.external_force[:, i]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols.keys())
            w.writerows(np.column_stack(list(cols.values())))


def _foh_discretize(A: np.ndarray, B: np.ndarray, dt: float):
    """Exact discretization for inputs linear between samples:
    x[k+1] = Phi x[k] + G0 u[k] + G1 (u[k+1] - u[k])."""
    n, m = B.shape
    big = np.zeros((n + 2 * m, n + 2 * m))
    big[:n, :n] = A * dt
    big[:n, n:n + m] = B * dt
    big[n:n + m, n + m:] = np.eye(m)
    E = expm(big)
    return E[:n, :n], E[:n, n:n + m], E[:n, n + m:]


def _mode_arrays(modes: List[SloshMode]):
    m = np.array([md.mass for md in modes])
    w = np.array([md.omega for md in modes])
    z = np.array([md.zeta for md in modes])
    h = np.array([md.height for md in modes])
    return m, w, z, h


def _axis_system(modes: List[SloshMode], coupled: bool, inertia: float):
    """State x = [x_1, v_1, x_2, v_2, ...] for one axis; input is f (prescribed)
    or F_ext (coupled). Returns A, B and the row r giving sum(k x + c v) = r @ x."""
    n = len(modes)
    m, w, z, _ = _mode_arrays(modes)
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, 1))
    r = np.zeros(2 * n)
    r[0::2] = m * w ** 2
    r[1::2] = 2 * z * m * w
    for i in range(n):
        A[2 * i, 2 * i + 1] = 1.0
        A[2 * i + 1, 2 * i] = -w[i] ** 2
        A[2 * i + 1, 2 * i + 1] = -2 * z[i] * w[i]
    if coupled:
        # v_i' = ... - f,  f = (F_ext + r @ x) / inertia
        A[1::2, :] -= r[None, :] / inertia
        B[1::2, 0] = -1.0 / inertia
    else:
        B[1::2, 0] = -1.0
    return A, B, r


def _as_input(t: np.ndarray, u) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.shape != (len(t), 3):
        raise ValueError(f"input must have shape (len(t), 3) = ({len(t)}, 3); got {u.shape}")
    return u


def _check_time(t) -> tuple:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1-D array with at least 2 samples")
    dt = t[1] - t[0]
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("t must be uniformly spaced and increasing")
    return t, dt


def _run(params: EMMParams, t, u, coupled: bool, dry_mass: float, ref_point,
         initial: str) -> SimResult:
    xr, yr, zr = (float(c) for c in ref_point)
    t, dt = _check_time(t)
    u = _as_input(t, u)
    lat, ax = params.lateral_modes, params.axial_modes
    n_lat, n_ax = len(lat), len(ax)
    m_liq = params.liquid_mass
    inertia_lat = dry_mass + params.fixed_mass
    inertia_ax = dry_mass + params.axial_rigid_mass

    A_lat, B_lat, r_lat = _axis_system(lat, coupled, inertia_lat)
    A_ax, B_ax, r_ax = _axis_system(ax, coupled, inertia_ax)
    A = block_diag(A_lat, A_lat, A_ax)
    B = block_diag(B_lat, B_lat, B_ax)
    sx = slice(0, 2 * n_lat)
    sy = slice(2 * n_lat, 4 * n_lat)
    sz = slice(4 * n_lat, 4 * n_lat + 2 * n_ax)
    ns = A.shape[0]

    # initial state
    x0 = np.zeros(ns)
    if initial == "static":
        f0 = u[0] / (dry_mass + m_liq) if coupled else u[0]
        if n_lat:
            w2 = np.array([md.omega for md in lat]) ** 2
            x0[sx][0::2] = -f0[0] / w2
            x0[sy][0::2] = -f0[1] / w2
        if n_ax:
            x0[sz][0::2] = -f0[2] / np.array([md.omega for md in ax]) ** 2
    elif initial != "rest":
        raise ValueError("initial must be 'static' or 'rest'")

    # integrate
    N = len(t)
    X = np.empty((N, ns))
    X[0] = x0
    if ns:
        Phi, G0, G1 = _foh_discretize(A, B, dt)
        drive = u[:-1] @ G0.T + np.diff(u, axis=0) @ G1.T
        for k in range(N - 1):
            X[k + 1] = Phi @ X[k] + drive[k]

    # outputs
    S = np.column_stack([X[:, sx] @ r_lat, X[:, sy] @ r_lat, X[:, sz] @ r_ax])  # spring forces
    if coupled:
        f = np.empty_like(u)
        f[:, 0] = (u[:, 0] + S[:, 0]) / inertia_lat
        f[:, 1] = (u[:, 1] + S[:, 1]) / inertia_lat
        f[:, 2] = (u[:, 2] + S[:, 2]) / inertia_ax
    else:
        f = u.copy()

    m_lat, _, _, h_lat = _mode_arrays(lat)
    m_ax = np.array([md.mass for md in ax])
    k_lat, c_lat = r_lat[0::2], r_lat[1::2]
    xl, vxl = X[:, sx][:, 0::2], X[:, sx][:, 1::2]
    yl, vyl = X[:, sy][:, 0::2], X[:, sy][:, 1::2]
    zl = X[:, sz][:, 0::2]

    F = np.empty_like(u)
    F[:, 0] = -params.fixed_mass * f[:, 0] + S[:, 0]
    F[:, 1] = -params.fixed_mass * f[:, 1] + S[:, 1]
    F[:, 2] = -params.axial_rigid_mass * f[:, 2] + S[:, 2]

    arm0 = params.fixed_height - zr
    arm = h_lat - zr
    # moment of lateral forces (z * F) plus axial load of each slosh mass at its offset
    mom_xz = -params.fixed_mass * f[:, 0] * arm0 + (xl * k_lat + vxl * c_lat) @ arm
    mom_yz = -params.fixed_mass * f[:, 1] * arm0 + (yl * k_lat + vyl * c_lat) @ arm
    shift_x = xl @ m_lat
    shift_y = yl @ m_lat
    M = np.zeros_like(u)
    M[:, 0] = -mom_yz - f[:, 2] * shift_y      # M_x = y F_z - z F_y
    M[:, 1] = mom_xz + f[:, 2] * shift_x       # M_y = z F_x - x F_z
    # shift from (0, 0, zr) to (xr, yr, zr): M_P = M - (xr, yr, 0) x F
    M[:, 0] -= yr * F[:, 2]
    M[:, 1] += xr * F[:, 2]
    M[:, 2] -= xr * F[:, 1] - yr * F[:, 0]

    cg = np.empty_like(u)
    cg[:, 0] = shift_x / m_liq
    cg[:, 1] = shift_y / m_liq
    cg[:, 2] = params.cg_height + (zl @ m_ax if n_ax else 0.0) / m_liq

    lateral_disp = np.stack([xl, yl], axis=-1)
    return SimResult(t=t, specific_force=f, interface_force=F, interface_moment=M,
                     liquid_cg=cg, lateral_disp=lateral_disp, axial_disp=zl,
                     external_force=u if coupled else None, ref_point=(xr, yr, zr))


def simulate_prescribed(params: EMMParams, t, specific_force, ref_point=(0.0, 0.0, 0.0),
                        initial: str = "static") -> SimResult:
    """Tank motion is prescribed through the specific force f(t) (shape (N, 3)).

    ref_point is the (x, y, z) point the moment is taken about.

    initial = 'static' starts the oscillators in equilibrium under f(t[0]);
    'rest' starts them undeflected.
    """
    return _run(params, t, specific_force, coupled=False, dry_mass=0.0, ref_point=ref_point,
                initial=initial)


def simulate_coupled(params: EMMParams, t, external_force, dry_mass: float,
                     ref_point=(0.0, 0.0, 0.0), initial: str = "static") -> SimResult:
    """External force F_ext(t) (shape (N, 3)) acts on a rigid dry structure of
    mass dry_mass (everything except the liquid) carrying the tank."""
    if dry_mass <= 0:
        raise ValueError("dry_mass must be positive")
    return _run(params, t, external_force, coupled=True, dry_mass=dry_mass, ref_point=ref_point,
                initial=initial)
