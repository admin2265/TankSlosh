"""Helpers for building input time histories."""
from __future__ import annotations

import numpy as np


def time_grid(duration: float, dt: float) -> np.ndarray:
    n = int(round(duration / dt)) + 1
    return np.arange(n) * dt


def trapezoid_pulse(t, start: float, duration: float, amplitude: float,
                    rise: float = 0.0) -> np.ndarray:
    """Pulse of the given amplitude that ramps up over `rise` seconds starting at
    `start`, holds, and ramps down so it is zero again at start + duration."""
    t = np.asarray(t, dtype=float)
    if rise < 0 or 2 * rise > duration:
        raise ValueError("need 0 <= rise <= duration / 2")
    end = start + duration
    if rise == 0:
        return np.where((t >= start) & (t < end), amplitude, 0.0)
    up = np.clip((t - start) / rise, 0.0, 1.0)
    down = np.clip((end - t) / rise, 0.0, 1.0)
    return amplitude * np.minimum(up, down)


def vector_input(t, x=0.0, y=0.0, z=0.0) -> np.ndarray:
    """Stack per-axis histories (arrays or constants) into shape (N, 3)."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([np.broadcast_to(np.asarray(c, dtype=float), t.shape)
                            for c in (x, y, z)])
