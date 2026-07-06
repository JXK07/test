"""Small Bezier helpers used by the BP3333-ellipse model."""

from __future__ import annotations

import numpy as np


def cubic_bezier(control: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate one coordinate of a cubic Bezier curve."""
    control = np.asarray(control, dtype=float)
    u = np.asarray(u, dtype=float)
    if control.shape != (4,):
        raise ValueError("A cubic Bezier curve requires exactly four controls.")
    return (
        control[0] * (1.0 - u) ** 3
        + 3.0 * control[1] * u * (1.0 - u) ** 2
        + 3.0 * control[2] * u**2 * (1.0 - u)
        + control[3] * u**3
    )


def cosine_sine_parameters(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return BP-style cosine/sine Bezier parameter spacing."""
    if n < 4:
        raise ValueError("At least four points are required per segment.")
    i_scaled = np.arange(n, dtype=float) * np.pi / (2.0 * (n - 1))
    return 1.0 - np.cos(i_scaled), np.sin(i_scaled)
