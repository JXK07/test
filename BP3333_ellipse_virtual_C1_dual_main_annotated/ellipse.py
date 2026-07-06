"""Leading-edge ellipse utilities.

The improved thickness law uses an elliptic head before the Bezier curve.  The
ellipse is written as

    ((x - a) / a)^2 + (y / b)^2 = 1

so its centre has y=0 and the curve passes through the thickness nose (0, 0).
The point used to connect to the Bezier segment is controlled by theta:

    x = a * (1 - cos(theta)),  y = b * sin(theta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class EllipseHead:
    """Semi-axes and tangent point for the leading-edge ellipse."""

    a: float
    b: float
    theta: float

    @property
    def x_tangent(self) -> float:
        return float(self.a * (1.0 - np.cos(self.theta)))

    @property
    def y_tangent(self) -> float:
        return float(self.b * np.sin(self.theta))

    @property
    def slope_tangent(self) -> float:
        return float((self.b * np.cos(self.theta)) / (self.a * np.sin(self.theta)))


def fit_ellipse_head(
    x: np.ndarray,
    y: np.ndarray,
    axis_ratio: float = 3.0,
    fit_limit: float = 0.025,
) -> tuple[float, float]:
    """Fit the leading-edge ellipse semi-axes to the first chordwise points.

    Only the size is fitted; the ratio ``a / b`` is prescribed by the user.
    This mirrors the original BP3434 code's local leading-edge fit philosophy,
    but uses the proposed ellipse instead of a circle.
    """
    if axis_ratio <= 0.0:
        raise ValueError("axis_ratio must be positive.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x >= 0.0) & (x <= fit_limit) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 4:
        mask = (x >= 0.0) & (x <= 0.04) & np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = np.maximum(y[mask], 0.0)
    if len(x_fit) < 4:
        # A conservative fallback for already sparse sections.
        b = max(float(np.nanmax(y[: min(len(y), 8)])), 1e-3)
        return float(axis_ratio * b), float(b)

    def y_ellipse(a_value: float) -> np.ndarray:
        b_value = a_value / axis_ratio
        inside = 1.0 - ((x_fit - a_value) / a_value) ** 2
        return b_value * np.sqrt(np.clip(inside, 0.0, None))

    def residual(values: np.ndarray) -> np.ndarray:
        return y_ellipse(float(values[0])) - y_fit

    min_a = max(0.5 * float(np.max(x_fit)) + 1e-6, 5e-4)
    max_a = max(0.20, 4.0 * min_a)
    initial_b = max(float(np.max(y_fit)), 1e-3)
    initial_a = float(np.clip(axis_ratio * initial_b, min_a, max_a))
    result = optimize.least_squares(
        residual,
        x0=np.array([initial_a]),
        bounds=(np.array([min_a]), np.array([max_a])),
        max_nfev=200,
    )
    a = float(result.x[0])
    return a, float(a / axis_ratio)


def make_ellipse_head(a: float, b: float, theta: float, max_x: float) -> EllipseHead:
    """Create a valid ellipse head, clipping theta before maximum thickness."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError("Ellipse semi-axes must be positive.")
    theta = float(np.clip(theta, 1e-4, 0.5 * np.pi - 1e-4))
    max_tangent_x = 0.85 * max(float(max_x), 1e-4)
    if a * (1.0 - np.cos(theta)) >= max_tangent_x:
        theta = float(np.arccos(np.clip(1.0 - max_tangent_x / a, -1.0, 1.0)))
        theta = float(np.clip(theta, 1e-4, 0.5 * np.pi - 1e-4))
    return EllipseHead(a=a, b=b, theta=theta)


def sample_ellipse(head: EllipseHead, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the ellipse from the nose to the tangent point."""
    theta = np.linspace(0.0, head.theta, max(n, 3))
    x = head.a * (1.0 - np.cos(theta))
    y = head.b * np.sin(theta)
    return x.astype(float), y.astype(float)


def fit_trailing_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    fit_start: float = 0.90,
) -> tuple[float, float]:
    """Fit a trailing-edge ellipse passing through the thickness point (1, 0).

    The ellipse is written as

        ((x - 1 + a) / a)^2 + (y / b)^2 = 1

    so the right endpoint is the usual thickness trailing-edge point (1, 0).
    The fit is only used to seed the optimiser; the final trailing ellipse is
    still constrained by the Bezier control-point equations.
    """
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    mask = (x >= fit_start) & (x <= 1.0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 4:
        mask = (x >= 0.85) & (x <= 1.0) & np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 4:
        return 0.06, max(float(np.nanmax(y[-10:])), 1e-3)

    min_a = max(0.5 * (1.0 - float(np.min(x_fit))) + 1e-5, 2e-3)
    max_a = max(0.30, 2.0 * min_a)
    initial_a = float(np.clip(0.75 * (1.0 - float(np.min(x_fit))) + 0.02, min_a, max_a))
    initial_b = float(np.clip(np.nanmax(y_fit), 5e-4, 0.20))

    def model(a_value: float, b_value: float) -> np.ndarray:
        inside = 1.0 - ((x_fit - 1.0 + a_value) / a_value) ** 2
        return b_value * np.sqrt(np.clip(inside, 0.0, None))

    def residual(values: np.ndarray) -> np.ndarray:
        return model(float(values[0]), float(values[1])) - y_fit

    result = optimize.least_squares(
        residual,
        x0=np.array([initial_a, initial_b]),
        bounds=(np.array([min_a, 1e-5]), np.array([max_a, 0.30])),
        max_nfev=250,
    )
    return float(result.x[0]), float(result.x[1])


def trailing_theta_from_beta(a: float, b: float, beta_te: float) -> float:
    """Estimate the tail ellipse tangent parameter from a trailing wedge angle."""
    beta = max(float(beta_te), 1e-4)
    theta = np.arctan2(float(b), float(a) * np.tan(beta))
    return float(np.clip(theta, 0.03, 1.45))
