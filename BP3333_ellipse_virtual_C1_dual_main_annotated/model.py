"""BP3333 virtual-thickness C1 model with elliptic leading and trailing edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import interpolate, optimize

from .bezier import cosine_sine_parameters, cubic_bezier
from .ellipse import EllipseHead, sample_ellipse


@dataclass(frozen=True)
class BP3333EllipseParameters:
    """BP3333 virtual-thickness C1 parameters.

    Angles are radians.  ``theta`` controls the tangent point on the ellipse;
    ``ellipse_a`` and ``ellipse_b`` are the semi-axes of the elliptic head.
    ``k_t`` is kept only as a seed/diagnostic field for compatibility; the C1
    thickness law computes the maximum-thickness curvature from its controls.
    """

    x_t: float
    y_t: float
    k_t: float
    beta_te: float
    gamma_le: float
    x_c: float
    y_c: float
    k_c: float
    alpha_te: float
    dz_te: float
    z_te: float
    ellipse_a: float
    ellipse_b: float
    theta: float
    y0_le: float
    y0_te: float
    te_ellipse_a: float = 0.0
    te_ellipse_b: float = 0.0
    te_theta: float = 0.0
    te_c2: bool = False

    def to_dict(self) -> dict[str, float]:
        return {key: (bool(value) if isinstance(value, bool) else float(value)) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class TailEllipse:
    """Trailing-edge ellipse attached to the aft thickness Bezier segment."""

    a: float
    b: float
    theta: float
    c2_residual: float = np.nan
    c2_success: bool = False

    @property
    def x_tangent(self) -> float:
        return float(1.0 - self.a + self.a * np.cos(self.theta))

    @property
    def y_tangent(self) -> float:
        return float(self.b * np.sin(self.theta))

    @property
    def slope_tangent(self) -> float:
        return float(-(self.b * np.cos(self.theta)) / (self.a * np.sin(self.theta)))

    @property
    def curvature(self) -> float:
        sin_t = np.sin(self.theta)
        cos_t = np.cos(self.theta)
        denom = (self.a**2 * sin_t**2 + self.b**2 * cos_t**2) ** 1.5
        return float(-self.a * self.b / denom)


@dataclass(frozen=True)
class ThicknessControls:
    """Control points and root diagnostics for the thickness law."""

    ellipse: EllipseHead
    p: float
    q: float
    q_residual: float
    root_success: bool
    x_le: np.ndarray
    y_le: np.ndarray
    x_te: np.ndarray
    y_te: np.ndarray
    tail_ellipse: TailEllipse | None = None
    u_s: float = np.nan
    v_r: float = np.nan
    phi_r: float = np.nan
    trailing_residual: float = np.nan
    kappa_t_star: float = np.nan
    leading_curvature_jump: float = np.nan
    trailing_curvature_jump: float = np.nan


def generate_airfoil(params: BP3333EllipseParameters, n_per_segment: int = 220) -> dict[str, np.ndarray]:
    """Generate thickness, camber, and upper/lower surfaces."""
    thickness_x, thickness_y, controls = thickness_distribution(params, n_per_segment)
    camber_x, camber_y = camber_distribution(params, n_per_segment, thickness_x)
    upper_x, upper_y, lower_x, lower_y = thickness_camber_to_surfaces(
        thickness_x,
        thickness_y,
        camber_x,
        camber_y,
    )
    return {
        "thickness_x": thickness_x,
        "thickness_y": thickness_y,
        "camber_x": camber_x,
        "camber_y": camber_y,
        "upper_x": upper_x,
        "upper_y": upper_y,
        "lower_x": lower_x,
        "lower_y": lower_y,
        "controls": controls,
    }


def thickness_distribution(
    params: BP3333EllipseParameters,
    n_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, ThicknessControls]:
    """Build thickness from ellipse head, Bezier interiors, and trailing ellipse."""
    controls = thickness_control_points(params)
    _, u_te = cosine_sine_parameters(n_per_segment)

    x_ellipse, y_ellipse = sample_ellipse(controls.ellipse, max(10, n_per_segment // 2))
    u_le = controls.u_s + (1.0 - controls.u_s) * u_te
    v_te = controls.v_r * u_te
    x_le = cubic_bezier(controls.x_le, u_le[1:])
    y_le = cubic_bezier(controls.y_le, u_le[1:])
    x_te = cubic_bezier(controls.x_te, v_te[1:])
    y_te = cubic_bezier(controls.y_te, v_te[1:])

    pieces_x = [x_ellipse, x_le, x_te]
    pieces_y = [y_ellipse, y_le, y_te]
    if controls.tail_ellipse is not None:
        x_tail, y_tail = sample_trailing_ellipse(controls.tail_ellipse, max(10, n_per_segment // 3))
        pieces_x.append(x_tail[1:])
        pieces_y.append(y_tail[1:])

    x = np.concatenate(pieces_x)
    y = np.concatenate(pieces_y)
    return *_validate_distribution(x, y, "thickness"), controls


def thickness_control_points(params: BP3333EllipseParameters) -> ThicknessControls:
    """Return controls from the C1 virtual-thickness equations."""
    ellipse = EllipseHead(params.ellipse_a, params.ellipse_b, params.theta)
    if not trailing_ellipse_enabled(params):
        raise ValueError("The C1 virtual-thickness model requires a trailing ellipse.")
    tail_ellipse = TailEllipse(
        a=float(params.te_ellipse_a),
        b=float(params.te_ellipse_b),
        theta=float(np.clip(params.te_theta, 0.03, 0.5 * np.pi - 1e-3)),
    )
    solution = solve_virtual_c1_controls(params, ellipse, tail_ellipse)
    x_le = solution["x_le"]
    y_le = solution["y_le"]
    x_te = solution["x_te"]
    y_te = solution["y_te"]
    _validate_control_polygon(x_le, "leading thickness")
    _validate_control_polygon(x_te, "trailing thickness")
    root_success = bool(solution["success"])
    return ThicknessControls(
        ellipse=ellipse,
        p=float(solution["u_s"]),
        q=float(solution["x2"]),
        q_residual=float(solution["residual_norm"]),
        root_success=root_success,
        x_le=x_le,
        y_le=y_le,
        x_te=x_te,
        y_te=y_te,
        tail_ellipse=tail_ellipse,
        u_s=float(solution["u_s"]),
        v_r=float(solution["v_r"]),
        phi_r=float(tail_ellipse.theta),
        trailing_residual=float(solution["residual_norm"]),
        kappa_t_star=float(solution["kappa_t_star"]),
        leading_curvature_jump=float(solution["leading_curvature_jump"]),
        trailing_curvature_jump=float(solution["trailing_curvature_jump"]),
    )


def ellipse_curvature(ellipse: EllipseHead) -> float:
    """Return the signed upper-branch curvature at the ellipse tangent point."""
    sin_t = np.sin(ellipse.theta)
    cos_t = np.cos(ellipse.theta)
    denom = (ellipse.a**2 * sin_t**2 + ellipse.b**2 * cos_t**2) ** 1.5
    return float(-ellipse.a * ellipse.b / denom)


def solve_virtual_c1_controls(
    params: BP3333EllipseParameters,
    ellipse: EllipseHead,
    tail: TailEllipse,
) -> dict[str, object]:
    """Solve the C1-at-tangency, C1+G2-at-max-thickness system.

    ``y0_te`` is a design variable in this version.  Given ``y0_le``,
    ``y0_te``, both ellipse tangent points, and ``x_t/y_t``, the LE C0+C1
    equations express ``x1,y1,x2`` as functions of ``u_s``.  The maximum
    thickness G2 condition gives ``Q2.y = y1``.  The TE x-position equation
    expresses ``Q2.x = xi`` as a function of ``u_s,v_r``.  The two remaining
    equations are TE y-position and TE tangent continuity.
    """

    x_t = float(params.x_t)
    y_t = float(params.y_t)
    y0_le = float(params.y0_le)
    y0_te = float(params.y0_te)
    x_s = ellipse.x_tangent
    y_s = ellipse.y_tangent
    m_s = ellipse.slope_tangent
    x_r = tail.x_tangent
    y_r = tail.y_tangent
    m_r = tail.slope_tangent
    if not (
        0.0 < x_s < x_t < x_r < 1.0
        and 0.0 < y0_le < y_s < y_t
        and 0.0 <= y0_te < y_t
        and 0.0 < y_r < y_t
        and m_s > 0.0
        and m_r < 0.0
    ):
        raise ValueError("Invalid C1 virtual-thickness tangent geometry.")

    y_scale = max(y_t, 1e-3)

    def leading_from_u(u_s: float) -> tuple[float, float, float]:
        u = float(u_s)
        a_s = (1.0 - u) ** 3
        b_s = 3.0 * u * (1.0 - u) ** 2
        c_s = 3.0 * u**2 * (1.0 - u)
        d_s = u**3
        denominator = 3.0 * m_s * u**2 * (1.0 - u)
        if abs(b_s) < 1e-14 or abs(denominator) < 1e-14:
            raise ValueError("Degenerate leading C1 parameter.")
        numerator = (
            m_s * (x_s * (3.0 * u - 1.0) - 2.0 * u**3 * x_t)
            + u**3 * (y0_le - y_t)
            + 3.0 * u**2 * (y_t - y0_le)
            + 3.0 * u * (y0_le - y_s)
            + (y_s - y0_le)
        )
        x2 = numerator / denominator
        x1 = (x_s - c_s * x2 - d_s * x_t) / b_s
        y1 = (y_s - a_s * y0_le - (c_s + d_s) * y_t) / b_s
        return float(x1), float(y1), float(x2)

    def controls_from_uv(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        u_s, v_r = np.asarray(values, dtype=float)
        x1, y1, x2 = leading_from_u(float(u_s))
        a_r = (1.0 - v_r) ** 3
        b_r = 3.0 * v_r * (1.0 - v_r) ** 2
        c_r = 3.0 * v_r**2 * (1.0 - v_r)
        d_r = v_r**3
        if abs(c_r) < 1e-14:
            raise ValueError("Degenerate trailing C1 parameter.")
        xi = (x_r - a_r * x_t - b_r * (2.0 * x_t - x2) - d_r) / c_r
        x_le = np.array([0.0, x1, x2, x_t], dtype=float)
        y_le = np.array([y0_le, y1, y_t, y_t], dtype=float)
        x_te = np.array([x_t, 2.0 * x_t - x2, xi, 1.0], dtype=float)
        y_te = np.array([y_t, y_t, y1, y0_te], dtype=float)
        return x_le, y_le, x_te, y_te, float(u_s), float(v_r)

    def equation_residual(values: np.ndarray) -> np.ndarray:
        x_le, y_le, x_te, y_te, _, v_r = controls_from_uv(values)
        y_te_r = cubic_bezier(y_te, np.array([v_r]))[0]
        dx_te, dy_te = _cubic_derivative(x_te, y_te, v_r)
        return np.array(
            [
                (y_te_r - y_r) / y_scale,
                (dy_te - m_r * dx_te),
            ],
            dtype=float,
        )

    def valid(values: np.ndarray) -> bool:
        try:
            x_le, y_le, x_te, y_te, u_s, v_r = controls_from_uv(values)
            kappa_t = _maximum_thickness_curvature(x_le, y_le)
            active_u = np.linspace(u_s, 1.0, 80)
            active_v = np.linspace(0.0, v_r, 80)
            x_le_active = cubic_bezier(x_le, active_u)
            y_le_active = cubic_bezier(y_le, active_u)
            x_te_active = cubic_bezier(x_te, active_v)
            y_te_active = cubic_bezier(y_te, active_v)
        except Exception:
            return False
        return bool(
            np.all(np.isfinite(np.concatenate([x_le, y_le, x_te, y_te, [u_s, v_r, kappa_t]])))
            and 0.02 < u_s < 0.98
            and 0.02 < v_r < 0.98
            and 0.0 < x_le[1] < x_le[2] < x_t
            and x_t < x_te[1] < x_te[2] < x_r < 1.0
            and y0_le < y_le[1] < y_t
            and kappa_t < 0.0
            and np.all(y_le_active >= -1e-8)
            and np.all(y_te_active >= -1e-8)
            and np.all(np.diff(x_le_active) >= -1e-9)
            and np.all(np.diff(x_te_active) >= -1e-9)
        )

    lower = np.array([0.02, 0.02], dtype=float)
    upper = np.array([0.98, 0.98], dtype=float)
    seed_vectors = _mt_g2_seed_vectors()

    best: tuple[float, np.ndarray, bool] | None = None
    for seed in seed_vectors:
        result = optimize.least_squares(
            equation_residual,
            x0=np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=120,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        for candidate in (result.x, seed):
            if not valid(candidate):
                continue
            residual_norm = float(np.linalg.norm(equation_residual(candidate)))
            success = bool(residual_norm < 1e-7)
            if best is None or residual_norm < best[0]:
                best = (residual_norm, np.asarray(candidate, dtype=float).copy(), success)

    if best is None:
        raise ValueError("No feasible C1+G2 virtual-thickness control point solution.")

    residual_norm, vector, success = best
    x_le, y_le, x_te, y_te, u_s, v_r = controls_from_uv(vector)
    leading_jump = _cubic_curvature(x_le, y_le, u_s) - ellipse_curvature(ellipse)
    trailing_jump = _cubic_curvature(x_te, y_te, v_r) - tail.curvature
    return {
        "x_le": x_le,
        "y_le": y_le,
        "x_te": x_te,
        "y_te": y_te,
        "x2": float(x_le[2]),
        "u_s": float(u_s),
        "v_r": float(v_r),
        "residual_norm": residual_norm,
        "success": success,
        "kappa_t_star": _maximum_thickness_curvature(x_le, y_le),
        "leading_curvature_jump": float(leading_jump),
        "trailing_curvature_jump": float(trailing_jump),
    }


def _mt_g2_seed_vectors() -> list[np.ndarray]:
    """Build robust starting points for the two-variable C1+G2 solve."""
    seeds: list[np.ndarray] = []
    for u_s in (0.12, 0.24, 0.40, 0.58, 0.76):
        for v_r in (0.18, 0.34, 0.52, 0.70, 0.86):
            seeds.append(np.array([u_s, v_r], dtype=float))
    return seeds


def _maximum_thickness_curvature(x_le: np.ndarray, y_le: np.ndarray) -> float:
    """Return the C2-implied curvature at the maximum-thickness point."""
    dx = float(x_le[3] - x_le[2])
    if abs(dx) < 1e-12:
        raise ValueError("Degenerate maximum-thickness curvature.")
    return float(2.0 * (y_le[1] - y_le[3]) / (3.0 * dx**2))


def solve_leading_virtual_controls(
    x_s: float,
    y_s: float,
    tangent_slope: float,
    kappa_s: float,
    x_t: float,
    y_t: float,
    k_t: float,
    y0_le: float,
    min_tail_h: float = 0.0,
) -> dict[str, float | bool]:
    """Solve the leading virtual-thickness controls from ``de_virtual_thickness``.

    The unknowns are reduced to the Bezier internal tangent parameter ``u_s``.
    For each trial ``u_s`` the C0 equations give ``x1`` and ``y1`` after the
    C1 equation gives ``q(u_s)``.  The scalar residual is the C2 curvature
    mismatch at the ellipse-Bezier tangent point.
    """

    m_s = float(tangent_slope)
    if not (0.0 < x_s < x_t and 0.0 < y_s < y_t and m_s > 0.0 and kappa_s < 0.0 and k_t < 0.0):
        raise ValueError("Invalid leading virtual-thickness inputs.")
    y0_le = float(y0_le)
    if not (0.0 < y0_le < y_t):
        raise ValueError("Leading virtual thickness y0_le must be inside (0, y_t).")

    def controls_from_u(u_value: float) -> tuple[float, float, float]:
        u = float(u_value)
        a0 = (1.0 - u) ** 3
        b0 = 3.0 * u * (1.0 - u) ** 2
        c0 = 3.0 * u**2 * (1.0 - u)
        d0 = u**3
        denominator = 3.0 * m_s * u**2 * (1.0 - u)
        if abs(b0) < 1e-14 or abs(denominator) < 1e-14:
            raise ValueError("Degenerate leading virtual parameter.")
        numerator = (
            m_s * (x_s * (3.0 * u - 1.0) - 2.0 * u**3 * x_t)
            + u**3 * (y0_le - y_t)
            + 3.0 * u**2 * (y_t - y0_le)
            + 3.0 * u * (y0_le - y_s)
            + (y_s - y0_le)
        )
        q_value = numerator / denominator
        x1 = (x_s - c0 * q_value - d0 * x_t) / b0
        y1 = (y_s - a0 * y0_le - (c0 + d0) * y_t) / b0
        return float(q_value), float(x1), float(y1)

    def residual(u_value: float) -> float:
        q_value, x1, y1 = controls_from_u(u_value)
        x_control = np.array([0.0, x1, q_value, x_t], dtype=float)
        y_control = np.array([y0_le, y1, y_t, y_t], dtype=float)
        return float(_cubic_curvature(x_control, y_control, u_value) - kappa_s)

    def valid(u_value: float) -> bool:
        try:
            q_value, x1, y1 = controls_from_u(u_value)
            x_control = np.array([0.0, x1, q_value, x_t], dtype=float)
            y_control = np.array([y0_le, y1, y_t, y_t], dtype=float)
            dx, dy = _cubic_derivative(x_control, y_control, u_value)
            kappa = _cubic_curvature(x_control, y_control, u_value)
            h_value = y_t + 1.5 * k_t * (x_t - q_value) ** 2
        except Exception:
            return False
        return bool(
            np.isfinite(q_value + x1 + y1 + dx + dy + kappa + h_value)
            and 0.0 < x1 < q_value < x_t
            and 0.0 < y1 < 1.5 * y_t
            and max(float(min_tail_h), 0.0) < h_value < y_t
            and dx > 0.0
        )

    lower = 0.02
    upper = 0.98
    grid = np.linspace(lower, upper, 220)
    values = np.array([residual(value) if valid(value) else np.nan for value in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left + f_right):
            continue
        if abs(f_left) < 1e-10:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            candidate = float(optimize.brentq(residual, left, right, maxiter=120))
            if valid(candidate):
                roots.append(candidate)
    if roots:
        u_s = float(min(roots, key=lambda value: abs(value - 0.35)))
        q_value, x1, y1 = controls_from_u(u_s)
        return {
            "u_s": u_s,
            "q": q_value,
            "x1": x1,
            "y1": y1,
            "curvature_residual": residual(u_s),
            "success": True,
        }

    valid_grid = np.array([value for value in grid if valid(value)], dtype=float)
    if len(valid_grid) == 0:
        raise ValueError("No feasible leading virtual-thickness control point solution.")
    result = optimize.minimize_scalar(
        lambda value: residual(value) ** 2 if valid(value) else 1e8,
        bounds=(float(valid_grid[0]), float(valid_grid[-1])),
        method="bounded",
    )
    u_s = float(result.x)
    q_value, x1, y1 = controls_from_u(u_s)
    res = residual(u_s)
    return {
        "u_s": u_s,
        "q": q_value,
        "x1": x1,
        "y1": y1,
        "curvature_residual": res,
        "success": bool(result.success and valid(u_s) and abs(res) < 1e-3),
    }


def solve_trailing_virtual_controls(
    params: BP3333EllipseParameters,
    q: float,
    h: float,
) -> dict[str, float | bool]:
    """Solve the recommended trailing virtual-thickness closure.

    The trailing ellipse semi-axes are parameters.  The Bezier internal
    parameter ``v_r`` and ellipse parameter ``phi_r`` are solved from C1 and C2,
    while C0 is enforced analytically through ``r(v_r, phi_r)`` and
    ``y0_te(v_r, phi_r)``.
    """

    a = float(params.te_ellipse_a)
    b = float(params.te_ellipse_b)
    if not (a > 0.0 and b > 0.0):
        raise ValueError("Trailing virtual ellipse semi-axes must be positive.")

    x_t = float(params.x_t)
    y_t = float(params.y_t)
    q = float(q)
    h = float(h)
    x1 = 2.0 * x_t - q
    if not (x_t < x1 < 1.0 and 0.0 < h < y_t):
        raise ValueError("Invalid upstream trailing virtual controls.")

    def ellipse_state(phi_value: float) -> tuple[float, float, float, float]:
        phi = float(phi_value)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)
        if abs(sin_p) < 1e-14:
            raise ValueError("Degenerate trailing ellipse parameter.")
        x_r = 1.0 - a + a * cos_p
        y_r = b * sin_p
        m_r = -(b * cos_p) / (a * sin_p)
        denom = (a**2 * sin_p**2 + b**2 * cos_p**2) ** 1.5
        kappa_r = -a * b / denom
        return float(x_r), float(y_r), float(m_r), float(kappa_r)

    def controls_from_v_phi(v_value: float, phi_value: float) -> tuple[float, float, float, float, float, float]:
        v = float(v_value)
        phi = float(phi_value)
        x_r, y_r, m_r, kappa_r = ellipse_state(phi)
        a0 = (1.0 - v) ** 3
        b0 = 3.0 * v * (1.0 - v) ** 2
        c0 = 3.0 * v**2 * (1.0 - v)
        d0 = v**3
        if abs(c0) < 1e-14 or abs(d0) < 1e-14:
            raise ValueError("Degenerate trailing virtual parameter.")
        r = (x_r - a0 * x_t - b0 * x1 - d0) / c0
        y0_te = (y_r - (a0 + b0) * y_t - c0 * h) / d0
        return float(r), float(y0_te), x_r, y_r, m_r, kappa_r

    def equation_values(v_value: float, phi_value: float) -> tuple[float, float]:
        r, y0_te, _, _, m_r, kappa_r = controls_from_v_phi(v_value, phi_value)
        x_control = np.array([x_t, x1, r, 1.0], dtype=float)
        y_control = np.array([y_t, y_t, h, y0_te], dtype=float)
        dx, dy = _cubic_derivative(x_control, y_control, v_value)
        slope_residual = dy - m_r * dx
        curvature_residual = _cubic_curvature(x_control, y_control, v_value) - kappa_r
        return float(slope_residual), float(curvature_residual)

    def valid(v_value: float, phi_value: float) -> bool:
        try:
            r, y0_te, x_r, y_r, m_r, kappa_r = controls_from_v_phi(v_value, phi_value)
            slope_residual, curvature_residual = equation_values(v_value, phi_value)
        except Exception:
            return False
        return bool(
            np.isfinite(r + y0_te + x_r + y_r + m_r + kappa_r + slope_residual + curvature_residual)
            and 0.0 < v_value < 1.0
            and 0.0 < phi_value < 0.5 * np.pi
            and x1 < r < x_r < 1.0
            and 0.0 < y_r < y_t
            and 0.0 < y0_te < max(0.25, 3.0 * y_t)
            and m_r < 0.0
        )

    def residual_vector(values: np.ndarray) -> np.ndarray:
        v_value, phi_value = values
        if not valid(float(v_value), float(phi_value)):
            return np.array([1e3, 1e3], dtype=float)
        slope_residual, curvature_residual = equation_values(float(v_value), float(phi_value))
        _, _, _, _, _, kappa_r = controls_from_v_phi(float(v_value), float(phi_value))
        curvature_scale = max(1.0, abs(kappa_r))
        return np.array([slope_residual, curvature_residual / curvature_scale], dtype=float)

    phi_seed = float(np.clip(params.te_theta if params.te_theta > 0.0 else 0.55, 0.05, 1.45))
    seed_pairs = [(0.35, phi_seed), (0.55, phi_seed), (0.75, phi_seed)]
    for v_seed in (0.25, 0.45, 0.65, 0.85):
        for phi in np.linspace(0.12, 1.35, 6):
            seed_pairs.append((v_seed, float(phi)))

    best_payload: dict[str, float | bool] | None = None
    best_norm = np.inf
    for seed in seed_pairs:
        result = optimize.least_squares(
            residual_vector,
            x0=np.array(seed, dtype=float),
            bounds=(np.array([0.03, 0.03]), np.array([0.97, 1.52])),
            max_nfev=250,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        v_r = float(result.x[0])
        phi_r = float(result.x[1])
        if not valid(v_r, phi_r):
            continue
        slope_residual, curvature_residual = equation_values(v_r, phi_r)
        residual_norm = float(np.linalg.norm(residual_vector(result.x)))
        if residual_norm < best_norm:
            r, y0_te, _, _, _, _ = controls_from_v_phi(v_r, phi_r)
            best_norm = residual_norm
            best_payload = {
                "v_r": v_r,
                "phi_r": phi_r,
                "r": r,
                "y0_te": y0_te,
                "slope_residual": slope_residual,
                "curvature_residual": curvature_residual,
                "residual_norm": residual_norm,
                "success": bool(result.success and residual_norm < 1e-5),
            }

    if best_payload is None:
        raise ValueError("No feasible trailing virtual-thickness solution.")
    return best_payload


def _cubic_derivative(
    x_control: np.ndarray,
    y_control: np.ndarray,
    u: float,
) -> tuple[float, float]:
    """First derivative of a planar cubic Bezier curve."""
    u = float(u)
    d = (
        3.0 * (1.0 - u) ** 2 * (np.array([x_control[1] - x_control[0], y_control[1] - y_control[0]]))
        + 6.0 * u * (1.0 - u) * (np.array([x_control[2] - x_control[1], y_control[2] - y_control[1]]))
        + 3.0 * u**2 * (np.array([x_control[3] - x_control[2], y_control[3] - y_control[2]]))
    )
    return float(d[0]), float(d[1])


def _cubic_second_derivative(
    x_control: np.ndarray,
    y_control: np.ndarray,
    u: float,
) -> tuple[float, float]:
    """Second derivative of a planar cubic Bezier curve."""
    u = float(u)
    dd = (
        6.0 * (1.0 - u) * np.array(
            [
                x_control[2] - 2.0 * x_control[1] + x_control[0],
                y_control[2] - 2.0 * y_control[1] + y_control[0],
            ]
        )
        + 6.0 * u * np.array(
            [
                x_control[3] - 2.0 * x_control[2] + x_control[1],
                y_control[3] - 2.0 * y_control[2] + y_control[1],
            ]
        )
    )
    return float(dd[0]), float(dd[1])


def _cubic_curvature(x_control: np.ndarray, y_control: np.ndarray, u: float) -> float:
    """Signed curvature of a planar cubic Bezier curve."""
    dx, dy = _cubic_derivative(x_control, y_control, u)
    ddx, ddy = _cubic_second_derivative(x_control, y_control, u)
    denominator = (dx**2 + dy**2) ** 1.5
    if denominator <= 1e-16:
        raise ValueError("Degenerate cubic derivative.")
    return float((dx * ddy - dy * ddx) / denominator)


def trailing_ellipse_enabled(params: BP3333EllipseParameters) -> bool:
    """Return whether the trailing-edge ellipse segment is active."""
    return bool(params.te_ellipse_a > 1e-8 and params.te_ellipse_b > 1e-8)


def make_tail_ellipse(params: BP3333EllipseParameters, q: float, h: float) -> TailEllipse:
    """Create the trailing ellipse, optionally solving theta from C2 continuity."""
    a = float(params.te_ellipse_a)
    b = float(params.te_ellipse_b)
    if not (a > 0.0 and b > 0.0):
        raise ValueError("Trailing ellipse semi-axes must be positive.")
    theta_seed = float(np.clip(params.te_theta, 1e-4, 0.5 * np.pi - 1e-4))
    if params.te_c2:
        theta, residual = solve_tail_theta_c2(
            a=a,
            b=b,
            theta_seed=theta_seed,
            x_t=params.x_t,
            y_t=params.y_t,
            q=q,
            h=h,
        )
        return TailEllipse(a=a, b=b, theta=theta, c2_residual=residual, c2_success=True)

    ellipse = TailEllipse(a=a, b=b, theta=theta_seed)
    residual = tail_c2_residual(ellipse, x_t=params.x_t, y_t=params.y_t, q=q, h=h)
    return TailEllipse(a=a, b=b, theta=theta_seed, c2_residual=residual, c2_success=False)


def solve_tail_theta_c2(
    a: float,
    b: float,
    theta_seed: float,
    x_t: float,
    y_t: float,
    q: float,
    h: float,
) -> tuple[float, float]:
    """Solve deTE's trailing ellipse C2 equation for the tangent parameter."""

    def residual(theta_value: float) -> float:
        ellipse = TailEllipse(a=a, b=b, theta=theta_value)
        return tail_c2_residual(ellipse, x_t=x_t, y_t=y_t, q=q, h=h)

    def valid(theta_value: float) -> bool:
        ellipse = TailEllipse(a=a, b=b, theta=theta_value)
        x_r = ellipse.x_tangent
        y_r = ellipse.y_tangent
        m_r = ellipse.slope_tangent
        if not (np.isfinite(x_r + y_r + m_r) and m_r < 0.0):
            return False
        x1 = 2.0 * x_t - q
        x2 = x_r - (y_r - h) / m_r
        return bool(x_t < x1 < x2 < x_r < 1.0 and 0.0 < y_r < h < y_t)

    lower = 1e-3
    upper = 0.5 * np.pi - 1e-3
    grid = np.linspace(lower, upper, 220)
    values = np.array([residual(value) if valid(value) else np.nan for value in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left + f_right):
            continue
        if abs(f_left) < 1e-10:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            candidate = float(optimize.brentq(residual, left, right, maxiter=100))
            if valid(candidate):
                roots.append(candidate)
    if roots:
        theta = float(min(roots, key=lambda value: abs(value - theta_seed)))
        return theta, residual(theta)

    candidates = np.array([value for value in grid if valid(value)], dtype=float)
    if len(candidates) == 0:
        raise ValueError("No feasible trailing-ellipse theta satisfies control-point ordering.")
    result = optimize.minimize_scalar(
        lambda value: residual(value) ** 2 if valid(value) else 1e6,
        bounds=(float(candidates[0]), float(candidates[-1])),
        method="bounded",
    )
    theta = float(result.x)
    res = residual(theta)
    if not result.success or abs(res) > 1e-5 or not valid(theta):
        raise ValueError("Could not solve trailing-ellipse C2 tangent parameter.")
    return theta, res


def tail_c2_residual(ellipse: TailEllipse, x_t: float, y_t: float, q: float, h: float) -> float:
    """Return the deTE curvature-continuity residual at the tail tangent point."""
    x_r = ellipse.x_tangent
    y_r = ellipse.y_tangent
    m_r = ellipse.slope_tangent
    if abs(m_r) < 1e-12:
        return np.inf
    left = y_t - y_r - m_r * (2.0 * x_t - q - x_r)
    a_r = 1.5 * ellipse.curvature * (1.0 + m_r**2) ** 1.5
    length = (y_r - h) / m_r
    return float(left - a_r * length**2)


def sample_trailing_ellipse(ellipse: TailEllipse, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the trailing ellipse from the Bezier tangent point to (1, 0)."""
    theta = np.linspace(ellipse.theta, 0.0, max(n, 3))
    x = 1.0 - ellipse.a + ellipse.a * np.cos(theta)
    y = ellipse.b * np.sin(theta)
    return x.astype(float), y.astype(float)


def solve_q_from_de_formula(
    x_s: float,
    y_s: float,
    tangent_slope: float,
    kappa_s: float,
    x_t: float,
    y_t: float,
    k_t: float,
) -> tuple[float, float, float, bool]:
    """Solve q using the quartic equation derived in ``de.txt``.

    The de.txt notation is used directly:
    D = y_t - y_s, A_s = 3/2*kappa_s*(1+m_s^2)^(3/2),
    B_t = 3/2*kappa_t, and

        Phi(q) = A_s/m_s^2 * [D + B_t*(x_t-q)^2]^2
                 - D + m_s*(q-x_s) = 0.

    After q is found, p is recovered from
        p = [D + B_t*(x_t-q)^2] / m_s.
    """
    m_s = float(tangent_slope)
    D = float(y_t - y_s)
    if D <= 0.0 or m_s <= 0.0 or k_t >= 0.0 or kappa_s >= 0.0:
        raise ValueError("Invalid de.txt q-root inputs.")

    A_s = 1.5 * kappa_s * (1.0 + m_s**2) ** 1.5
    B_t = 1.5 * k_t
    q_min_candidates = [
        x_s + D / m_s,
        x_t - np.sqrt(max(-2.0 * D / (3.0 * k_t), 0.0)),
    ]
    lower = float(max(q_min_candidates) + 1e-8)
    upper = float(x_t - 1e-8)
    if lower >= upper:
        raise ValueError("Ellipse tangent point must be upstream of maximum thickness.")

    def p_from_q(q_value: float) -> float:
        return float((D + B_t * (x_t - q_value) ** 2) / m_s)

    def residual(q_value: float) -> float:
        return float(A_s / m_s**2 * (D + B_t * (x_t - q_value) ** 2) ** 2 - D + m_s * (q_value - x_s))

    def is_valid(q_value: float) -> bool:
        p_value = p_from_q(q_value)
        h_value = y_s + m_s * p_value
        return (
            np.isfinite(q_value)
            and np.isfinite(p_value)
            and lower < q_value < upper
            and p_value > 0.0
            and x_s < x_s + p_value < q_value < x_t
            and h_value < y_t
        )

    grid = np.linspace(lower, upper, 160)
    values = np.array([residual(value) for value in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left + f_right):
            continue
        if abs(f_left) < 1e-12:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            roots.append(float(optimize.brentq(residual, left, right, maxiter=100)))
    valid_roots = sorted(root for root in roots if is_valid(root))
    if valid_roots:
        q_value = float(valid_roots[0])
        return q_value, p_from_q(q_value), residual(q_value), True

    result = optimize.minimize_scalar(lambda value: residual(value) ** 2, bounds=(lower, upper), method="bounded")
    q_value = float(result.x)
    p_value = p_from_q(q_value)
    if not is_valid(q_value):
        p_value = float(np.clip(p_value, 1e-6, max(q_value - x_s - 1e-6, 1e-6)))
    return q_value, p_value, residual(q_value), False


def camber_distribution(
    params: BP3333EllipseParameters,
    n_per_segment: int,
    fallback_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ordinary BP3333 camber line."""
    if abs(params.y_c) < 1e-7 and abs(params.z_te) < 1e-8:
        return fallback_x.copy(), np.zeros_like(fallback_x)

    try:
        b1 = solve_b1(params.gamma_le, params.y_c, params.k_c, params.z_te, params.alpha_te)
        x_le, y_le, x_te, y_te = camber_control_points(params, b1)
        _validate_control_polygon(x_le, "leading camber")
        _validate_control_polygon(x_te, "trailing camber")
        u_le, u_te = cosine_sine_parameters(n_per_segment)
        x = np.concatenate([cubic_bezier(x_le, u_le), cubic_bezier(x_te, u_te[1:])])
        y = np.concatenate([cubic_bezier(y_le, u_le), cubic_bezier(y_te, u_te[1:])])
        return _validate_distribution(x, y, "camber")
    except Exception:
        slope_le = np.tan(params.gamma_le)
        slope_te = -np.tan(params.alpha_te)
        y = _hermite_camber(fallback_x, params.z_te, slope_le, slope_te)
        return fallback_x.copy(), y


def camber_control_points(
    params: BP3333EllipseParameters,
    b1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the standard BP3333 cubic camber control points."""
    left_root = _positive_sqrt(2.0 * (b1 - params.y_c) / (3.0 * params.k_c))
    x_le = np.array([0.0, b1 / np.tan(params.gamma_le), params.x_c - left_root, params.x_c])
    y_le = np.array([0.0, b1, params.y_c, params.y_c])
    right_root = _positive_sqrt(2.0 * (b1 - params.y_c) / (3.0 * params.k_c))
    x_te = np.array(
        [
            params.x_c,
            params.x_c + right_root,
            1.0 + (params.z_te - b1) / np.tan(params.alpha_te),
            1.0,
        ]
    )
    y_te = np.array([params.y_c, params.y_c, b1, params.z_te])
    return x_le, y_le, x_te, y_te


def solve_b1(gamma_le: float, y_c: float, k_c: float, z_te: float, alpha_te: float) -> float:
    """Solve the ordinary BP3333 camber control equation."""
    if abs(gamma_le) < 1e-7 or abs(alpha_te) < 1e-7 or k_c >= 0.0:
        raise ValueError("Invalid camber parameters for b1 solve.")
    cot_sum = 1.0 / np.tan(gamma_le) + 1.0 / np.tan(alpha_te)
    t1 = 3.0 * k_c * cot_sum**2
    t2 = 16.0 + 3.0 * k_c * cot_sum * (1.0 + z_te / np.tan(alpha_te))
    radicand = 16.0 + 6.0 * k_c * cot_sum * (1.0 - y_c * cot_sum + z_te / np.tan(alpha_te))
    if radicand < 0.0 or abs(t1) < 1e-12:
        raise ValueError("Invalid BP3333 b1 radicand.")
    roots = [(t2 + 4.0 * np.sqrt(radicand)) / t1, (t2 - 4.0 * np.sqrt(radicand)) / t1]
    valid = [root for root in roots if np.isfinite(root) and root < max(y_c, 1.0)]
    if not valid:
        raise ValueError("Could not solve BP3333 b1.")
    below = [root for root in valid if root < y_c]
    return float(below[0] if below else valid[0])


def thickness_camber_to_surfaces(
    thickness_x: np.ndarray,
    thickness_y: np.ndarray,
    camber_x: np.ndarray,
    camber_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Offset the camber line by the local normal using the half-thickness."""
    thickness_spline = interpolate.PchipInterpolator(thickness_x, thickness_y, extrapolate=True)
    thickness_at_camber = np.asarray(thickness_spline(camber_x), dtype=float)
    camber_spline = interpolate.CubicSpline(camber_x, camber_y, bc_type="natural")
    theta = np.arctan(camber_spline.derivative()(camber_x))
    upper_x = camber_x - thickness_at_camber * np.sin(theta)
    upper_y = camber_y + thickness_at_camber * np.cos(theta)
    lower_x = camber_x + thickness_at_camber * np.sin(theta)
    lower_y = camber_y - thickness_at_camber * np.cos(theta)
    return upper_x, upper_y, lower_x, lower_y


def _validate_distribution(x: np.ndarray, y: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    dx = np.diff(x)
    if np.any(dx < -1e-10):
        raise ValueError(f"{label} distribution is not monotonic.")
    keep = np.concatenate([[True], dx > 1e-10])
    x = x[keep]
    y = y[keep]
    if len(x) < 5 or np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{label} distribution is not monotonic.")
    return x, y


def _validate_control_polygon(x_control: np.ndarray, label: str) -> None:
    """Reject Bezier controls that would almost certainly create x-loops."""
    x_control = np.asarray(x_control, dtype=float)
    if np.any(~np.isfinite(x_control)) or np.any(np.diff(x_control) < -1e-10):
        raise ValueError(f"{label} control polygon is not monotonic in x.")


def _positive_sqrt(value: float) -> float:
    if value < 0.0:
        raise ValueError("Negative square-root argument in BP3333 controls.")
    return float(np.sqrt(value))


def _hermite_camber(x: np.ndarray, z_te: float, slope_le: float, slope_te: float) -> np.ndarray:
    h00 = 2.0 * x**3 - 3.0 * x**2 + 1.0
    h10 = x**3 - 2.0 * x**2 + x
    h01 = -2.0 * x**3 + 3.0 * x**2
    h11 = x**3 - x**2
    return h00 * 0.0 + h10 * slope_le + h01 * z_te + h11 * slope_te
