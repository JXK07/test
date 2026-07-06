"""SLSQP/GA fitting for the BP3333 virtual-thickness C1 parameterisation.

The optimiser follows the structure of the original BP3434 implementation:
use a direct parameter estimate as the starting point, minimise the model
surface error with SLSQP, enforce bounds on physical design variables, and
reject control polygons that would make the Bezier distributions non-monotonic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import interpolate, optimize

from .fit import CONNECTION_RESIDUAL_TOL, DirectFitResult
from .model import (
    BP3333EllipseParameters,
    camber_control_points,
    generate_airfoil,
    solve_b1,
    thickness_control_points,
)


VARIABLE_NAMES = (
    "x_t",
    "y_t",
    "beta_te",
    "gamma_le",
    "x_c",
    "y_c",
    "k_c",
    "alpha_te",
    "z_te",
    "dz_te",
    "ellipse_a",
    "axis_ratio",
    "theta",
    "y0_le",
    "y0_te",
    "te_ellipse_a",
    "te_axis_ratio",
    "te_theta",
)


@dataclass(frozen=True)
class OptimizedFitResult:
    """Initial and optimised BP3333 virtual C1 fits."""

    initial: DirectFitResult
    optimized: DirectFitResult
    variable_names: tuple[str, ...]
    initial_vector: np.ndarray
    optimized_vector: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    objective_initial: float
    objective_optimized: float
    method: str
    objective_mode: str
    success: bool
    status: int
    message: str
    iterations: int
    evaluations: int


@dataclass(frozen=True)
class _GAResult:
    """Compact result object returned by the built-in real-coded GA."""

    x: np.ndarray
    fun: float
    generations: int
    evaluations: int
    message: str


def optimize_fit(
    initial: DirectFitResult,
    maxiter: int = 500,
    min_control_spacing: float = 1e-4,
    tail_weight: float = 0.0,
    method: str = "slsqp",
    objective_mode: str = "mae",
    ga_population: int = 96,
    ga_generations: int = 120,
    random_seed: int | None = 7,
) -> OptimizedFitResult:
    """Optimise a direct BP3333 virtual C1 fit with SLSQP.

    The design vector is kept in absolute units, matching the newer BP3434
    SLSQP path.  Bounds are centred on the direct estimate so the optimiser can
    improve the fit without wandering into unrelated blade shapes.
    """
    direct_vector = params_to_vector(initial.params)
    vector0 = direct_vector.copy()
    lower_bounds, upper_bounds = build_bounds(initial)
    vector0 = np.clip(vector0, lower_bounds, upper_bounds)
    objective = build_objective(initial, tail_weight=tail_weight, mode=objective_mode)
    constraints = build_constraints(initial, min_control_spacing=min_control_spacing)

    best_vector = vector0.copy()
    best_objective = objective(vector0)

    def tracked_objective(vector: np.ndarray) -> float:
        nonlocal best_vector, best_objective
        value = objective(vector)
        if np.isfinite(value) and value < best_objective:
            best_objective = float(value)
            best_vector = np.asarray(vector, dtype=float).copy()
        return value

    method_key = method.lower()
    if method_key == "slsqp":
        result = _slsqp_optimize(
            tracked_objective,
            vector0,
            lower_bounds,
            upper_bounds,
            constraints,
            maxiter=maxiter,
        )
        candidate = np.asarray(result.x, dtype=float)
        candidate_objective = objective(candidate)
        success = bool(result.success)
        status = int(result.status)
        message = str(result.message)
        iterations = int(getattr(result, "nit", 0))
        evaluations = int(getattr(result, "nfev", 0))
    elif method_key in {"ga", "genetic"}:
        ga_result = _ga_optimize(
            objective,
            vector0,
            lower_bounds,
            upper_bounds,
            constraints,
            population_size=ga_population,
            generations=ga_generations,
            random_seed=random_seed,
        )
        candidate = ga_result.x
        candidate_objective = ga_result.fun
        success = True
        status = 0
        message = ga_result.message
        iterations = ga_result.generations
        evaluations = ga_result.evaluations
    elif method_key in {"hybrid", "ga+slsqp"}:
        ga_result = _ga_optimize(
            objective,
            vector0,
            lower_bounds,
            upper_bounds,
            constraints,
            population_size=ga_population,
            generations=ga_generations,
            random_seed=random_seed,
        )
        result = _slsqp_optimize(
            tracked_objective,
            ga_result.x,
            lower_bounds,
            upper_bounds,
            constraints,
            maxiter=maxiter,
        )
        candidate = np.asarray(result.x, dtype=float)
        candidate_objective = objective(candidate)
        success = bool(result.success) or np.isfinite(ga_result.fun)
        status = int(result.status)
        message = f"GA seed: {ga_result.message}; SLSQP: {result.message}"
        iterations = int(getattr(result, "nit", 0)) + ga_result.generations
        evaluations = int(getattr(result, "nfev", 0)) + ga_result.evaluations
    else:
        raise ValueError("method must be 'slsqp', 'ga', or 'hybrid'.")

    if not np.isfinite(candidate_objective) or candidate_objective > best_objective:
        candidate = best_vector
        candidate_objective = best_objective

    optimized, candidate, candidate_objective, fallback_message = _make_feasible_optimized_result(
        initial,
        objective,
        [
            (candidate, candidate_objective),
            (best_vector, best_objective),
            (vector0, objective(vector0)),
            (direct_vector, initial.mae),
        ],
    )
    if fallback_message:
        success = False
        message = f"{message}; {fallback_message}"

    return OptimizedFitResult(
        initial=initial,
        optimized=optimized,
        variable_names=VARIABLE_NAMES,
        initial_vector=vector0,
        optimized_vector=candidate,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        objective_initial=float(objective(vector0)),
        objective_optimized=float(candidate_objective),
        method=method_key,
        objective_mode=objective_mode,
        success=success,
        status=status,
        message=message,
        iterations=iterations,
        evaluations=evaluations,
    )


def _make_feasible_optimized_result(
    initial: DirectFitResult,
    objective: Callable[[np.ndarray], float],
    candidates: list[tuple[np.ndarray, float]],
) -> tuple[DirectFitResult, np.ndarray, float, str]:
    """Return the first candidate vector that can actually generate geometry."""
    last_error: Exception | None = None
    for vector, value in candidates:
        vector = np.asarray(vector, dtype=float).copy()
        try:
            result = make_result_from_vector(initial, vector)
            if not _continuous_enough(result):
                raise ValueError(f"C0/C1 join residual too large: {result.q_residual:.3e}")
            objective_value = float(value)
            if not np.isfinite(objective_value):
                objective_value = float(objective(vector))
            return result, vector, objective_value, ""
        except Exception as exc:
            last_error = exc

    return (
        initial,
        params_to_vector(initial.params),
        float(initial.mae),
        f"all optimiser candidates were infeasible; returned direct estimate ({last_error})",
    )


def _slsqp_optimize(
    objective: Callable[[np.ndarray], float],
    start: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    constraints: list[dict[str, object]],
    maxiter: int,
):
    """Run the BP3434-style local SLSQP optimiser."""
    return optimize.minimize(
        objective,
        start,
        method="SLSQP",
        bounds=optimize.Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"maxiter": maxiter, "disp": False},
        jac="3-point",
    )


def _ga_optimize(
    objective: Callable[[np.ndarray], float],
    seed_vector: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    constraints: list[dict[str, object]],
    population_size: int,
    generations: int,
    random_seed: int | None,
) -> _GAResult:
    """Run a small real-coded genetic algorithm without external packages."""
    rng = np.random.default_rng(random_seed)
    n_var = len(seed_vector)
    pop_size = max(12, int(population_size))
    elite_count = max(2, pop_size // 12)
    span = upper_bounds - lower_bounds
    mutation_sigma0 = 0.12 * span

    population = rng.uniform(lower_bounds, upper_bounds, size=(pop_size, n_var))
    population[0] = np.clip(seed_vector, lower_bounds, upper_bounds)
    for idx in range(1, min(pop_size, 10)):
        scale = 0.05 + 0.03 * idx
        population[idx] = np.clip(seed_vector + rng.normal(0.0, scale * span), lower_bounds, upper_bounds)

    evaluations = 0

    def penalized(vector: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        value = objective(vector)
        if not np.isfinite(value):
            value = 1e6
        violation = _constraint_violation(vector, constraints)
        return float(value + 1e5 * violation + 1e3 * violation**2)

    scores = np.array([penalized(individual) for individual in population])
    best_idx = int(np.argmin(scores))
    best_x = population[best_idx].copy()
    best_score = float(scores[best_idx])

    for gen in range(int(generations)):
        order = np.argsort(scores)
        elites = population[order[:elite_count]].copy()
        next_population = [elite for elite in elites]
        mutation_sigma = mutation_sigma0 * max(0.12, 1.0 - gen / max(generations, 1))

        while len(next_population) < pop_size:
            parent_a = population[_tournament(scores, rng)]
            parent_b = population[_tournament(scores, rng)]
            alpha = rng.uniform(-0.15, 1.15, size=n_var)
            child = alpha * parent_a + (1.0 - alpha) * parent_b
            mutation_mask = rng.random(n_var) < 0.22
            child = child + mutation_mask * rng.normal(0.0, mutation_sigma, size=n_var)
            if rng.random() < 0.12:
                reset_mask = rng.random(n_var) < 0.10
                child[reset_mask] = rng.uniform(lower_bounds[reset_mask], upper_bounds[reset_mask])
            next_population.append(np.clip(child, lower_bounds, upper_bounds))

        population = np.asarray(next_population, dtype=float)
        scores = np.array([penalized(individual) for individual in population])
        current_idx = int(np.argmin(scores))
        if scores[current_idx] < best_score:
            best_score = float(scores[current_idx])
            best_x = population[current_idx].copy()

    return _GAResult(
        x=best_x,
        fun=float(objective(best_x)),
        generations=int(generations),
        evaluations=evaluations,
        message=f"Real-coded GA completed: pop={pop_size}, generations={generations}",
    )


def _tournament(scores: np.ndarray, rng: np.random.Generator, size: int = 3) -> int:
    candidates = rng.integers(0, len(scores), size=size)
    return int(candidates[np.argmin(scores[candidates])])


def _constraint_violation(vector: np.ndarray, constraints: list[dict[str, object]]) -> float:
    violation = 0.0
    for constraint in constraints:
        values = np.asarray(constraint["fun"](vector), dtype=float)
        if values.size == 0:
            continue
        bad = values[~np.isfinite(values)]
        if bad.size:
            violation += 10.0 * bad.size
        finite = values[np.isfinite(values)]
        violation += float(np.sum(np.maximum(-finite, 0.0)))
    return violation


def params_to_vector(params: BP3333EllipseParameters) -> np.ndarray:
    """Convert parameters to the SLSQP design vector."""
    te_axis_ratio = params.te_ellipse_a / params.te_ellipse_b if params.te_ellipse_a > 0.0 and params.te_ellipse_b > 0.0 else 1.0
    return np.array(
        [
            params.x_t,
            params.y_t,
            params.beta_te,
            params.gamma_le,
            params.x_c,
            params.y_c,
            params.k_c,
            params.alpha_te,
            params.z_te,
            params.dz_te,
            params.ellipse_a,
            params.ellipse_a / params.ellipse_b,
            params.theta,
            params.y0_le,
            params.y0_te,
            params.te_ellipse_a,
            te_axis_ratio,
            params.te_theta,
        ],
        dtype=float,
    )


def vector_to_params(vector: np.ndarray, seed_k_t: float = -1.0, te_c2: bool = False) -> BP3333EllipseParameters:
    """Convert an SLSQP vector to BP3333 virtual C1 parameters."""
    x = np.asarray(vector, dtype=float)
    axis_ratio = max(float(x[11]), 1e-8)
    ellipse_a = float(x[10])
    y0_le = float(x[13]) if len(x) > 13 else 1e-5
    y0_te = float(x[14]) if len(x) > 14 else 0.0
    te_a = float(x[15]) if len(x) > 15 else 0.0
    te_ratio = max(float(x[16]), 1e-8) if len(x) > 16 else 1.0
    return BP3333EllipseParameters(
        x_t=float(x[0]),
        y_t=float(x[1]),
        k_t=float(seed_k_t),
        beta_te=float(x[2]),
        gamma_le=float(x[3]),
        x_c=float(x[4]),
        y_c=float(x[5]),
        k_c=float(x[6]),
        alpha_te=float(x[7]),
        z_te=float(x[8]),
        dz_te=float(x[9]),
        ellipse_a=ellipse_a,
        ellipse_b=ellipse_a / axis_ratio,
        theta=float(x[12]),
        y0_le=y0_le,
        y0_te=y0_te,
        te_ellipse_a=te_a,
        te_ellipse_b=te_a / te_ratio if te_a > 0.0 else 0.0,
        te_theta=float(x[17]) if len(x) > 17 else 0.0,
        te_c2=te_c2,
    )


def build_bounds(initial: DirectFitResult) -> tuple[np.ndarray, np.ndarray]:
    """Create variable bounds around the direct estimate.

    The multipliers mirror the BP3434 SLSQP philosophy: maximum thickness,
    maximum camber, trailing-edge offset, and trailing-edge half-thickness are
    allowed to move locally around the measured values; angles and auxiliary
    shape variables get wider but still physical ranges.
    """
    p = initial.params
    closed_blunt = _has_auxiliary_closed_trailing_edge(initial)
    symmetric = abs(p.y_c) < 1e-7 and abs(p.z_te) < 1e-8

    x_t_lower = max(0.03, 0.8 * p.x_t)
    x_t_upper = min(0.80, 1.2 * p.x_t)
    y_t_lower = max(1e-6, 0.8 * p.y_t)
    y_t_upper = min(0.35, 1.2 * p.y_t)

    beta_upper = 1.55 if closed_blunt else 0.40
    beta_upper = max(beta_upper, min(1.55, 1.2 * p.beta_te))

    if symmetric:
        x_c_lower, x_c_upper = 0.25, 0.55
        y_c_lower, y_c_upper = 0.0, 1e-6
        z_lower, z_upper = -1e-6, 1e-6
        gamma_lower, gamma_upper = -1e-4, 1e-4
        alpha_lower, alpha_upper = -1e-4, 1e-4
    else:
        x_c_guess = p.x_c if p.x_c > 0.0 else 0.35
        x_c_lower = max(0.0, 0.8 * x_c_guess)
        x_c_upper = min(0.90, 1.2 * x_c_guess)
        y_c_lower = max(0.0, 0.5 * p.y_c)
        y_c_upper = min(max(0.25, 1.5 * p.y_c), 0.80)
        if y_c_upper <= y_c_lower:
            y_c_upper = y_c_lower + max(1e-5, 0.1 * abs(p.y_c))
        if closed_blunt:
            tail_x, tail_thickness = _tail_thickness_points(initial)
            effective_tail = float(np.max(tail_thickness)) if len(tail_thickness) else p.dz_te
            z_delta = max(0.004, 0.5 * effective_tail, 0.2 * abs(p.z_te))
        else:
            z_delta = max(1e-4, 0.2 * abs(p.z_te))
        z_lower, z_upper = p.z_te - z_delta, p.z_te + z_delta
        gamma_delta = max(0.02, 0.2 * abs(p.gamma_le))
        alpha_delta = max(0.02, 0.2 * abs(p.alpha_te))
        gamma_lower, gamma_upper = p.gamma_le - gamma_delta, p.gamma_le + gamma_delta
        alpha_lower, alpha_upper = p.alpha_te - alpha_delta, p.alpha_te + alpha_delta

    k_c_lower, k_c_upper = _negative_curvature_bounds(p.k_c)
    if closed_blunt:
        tail_x, tail_thickness = _tail_thickness_points(initial)
        effective_tail = float(np.max(tail_thickness)) if len(tail_thickness) else p.dz_te
        dz_lower = max(0.0, 0.5 * p.dz_te)
        dz_upper = min(0.08, max(1.1 * p.dz_te, 1.25 * effective_tail, p.dz_te + 0.01))
    else:
        dz_lower = max(0.0, 0.9 * p.dz_te)
        dz_upper = max(1e-6, 1.1 * p.dz_te)
    if dz_upper <= dz_lower:
        dz_upper = dz_lower + 1e-5

    ellipse_a_lower = max(5e-4, 0.5 * p.ellipse_a)
    ellipse_a_upper = min(0.30, max(ellipse_a_lower + 1e-4, 1.5 * p.ellipse_a))
    y0_le_lower = max(1e-7, 0.20 * p.y0_le)
    y0_le_upper = min(0.95 * p.y_t, max(y0_le_lower + 1e-6, 2.80 * p.y0_le))
    y0_te_lower = 0.0
    y0_te_seed = max(p.y0_te, p.dz_te, 1e-7)
    y0_te_upper = min(0.95 * p.y_t, max(y0_te_lower + 1e-6, 3.0 * y0_te_seed))
    if p.te_ellipse_a > 0.0 and p.te_ellipse_b > 0.0:
        te_a_lower = max(2e-3, 0.45 * p.te_ellipse_a)
        te_a_upper = min(0.45, max(te_a_lower + 1e-4, 3.0 * p.te_ellipse_a))
        te_ratio0 = p.te_ellipse_a / p.te_ellipse_b
        te_ratio_lower = max(0.20, 0.35 * te_ratio0)
        te_ratio_upper = min(25.0, max(te_ratio_lower + 1e-4, 2.8 * te_ratio0))
        te_theta_lower = 0.03
        te_theta_upper = 1.45
    else:
        te_a_lower = te_a_upper = 0.0
        te_ratio_lower = te_ratio_upper = 1.0
        te_theta_lower = te_theta_upper = 0.0

    lower = np.array(
        [
            x_t_lower,
            y_t_lower,
            1e-4,
            gamma_lower,
            x_c_lower,
            y_c_lower,
            k_c_lower,
            alpha_lower,
            z_lower,
            dz_lower,
            ellipse_a_lower,
            0.40,
            0.03,
            y0_le_lower,
            y0_te_lower,
            te_a_lower,
            te_ratio_lower,
            te_theta_lower,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            x_t_upper,
            y_t_upper,
            beta_upper,
            gamma_upper,
            x_c_upper,
            y_c_upper,
            k_c_upper,
            alpha_upper,
            z_upper,
            dz_upper,
            ellipse_a_upper,
            10.0,
            1.45,
            y0_le_upper,
            y0_te_upper,
            te_a_upper,
            te_ratio_upper,
            te_theta_upper,
        ],
        dtype=float,
    )
    return lower, upper


def build_constraints(initial: DirectFitResult, min_control_spacing: float) -> list[dict[str, object]]:
    """Return SLSQP inequality constraints in scipy's ``fun(x) >= 0`` form."""
    camber_required = abs(initial.params.y_c) >= 1e-7 or abs(initial.params.z_te) >= 1e-8
    tail_required = initial.params.te_ellipse_a > 0.0 and initial.params.te_ellipse_b > 0.0

    def constraints(vector: np.ndarray) -> np.ndarray:
        params = vector_to_params(vector, seed_k_t=initial.params.k_t, te_c2=initial.params.te_c2)
        values: list[float] = []
        failure_size = 21 if camber_required and tail_required else 15 if tail_required else 16 if camber_required else 10
        try:
            controls = thickness_control_points(params)
            values.extend(np.diff(controls.x_le) - min_control_spacing)
            values.extend(np.diff(controls.x_te) - min_control_spacing)
            values.append(CONNECTION_RESIDUAL_TOL - controls.trailing_residual)
            values.append(params.y_t - params.y0_le - 1e-8)
            values.append(controls.y_te[2] - controls.y_te[-1] - 1e-8)
            values.append(params.y_t - controls.y_te[-1] - 1e-8)
            if tail_required:
                tail = controls.tail_ellipse
                values.append(1.0 if tail is not None else -1.0)
                if tail is not None:
                    values.append(1.0 - tail.x_tangent - min_control_spacing)
                    values.append(tail.x_tangent - controls.x_te[2] - min_control_spacing)
                    values.append(tail.y_tangent - 1e-8)
                    values.append(controls.y_te[2] - tail.y_tangent - 1e-8)
                else:
                    values.extend([-1.0] * 4)
        except Exception:
            return -np.ones(failure_size)

        if camber_required:
            try:
                b1 = solve_b1(params.gamma_le, params.y_c, params.k_c, params.z_te, params.alpha_te)
                x_le, _, x_te, _ = camber_control_points(params, b1)
                values.extend(np.diff(x_le) - min_control_spacing)
                values.extend(np.diff(x_te) - min_control_spacing)
            except Exception:
                values.extend([-1.0] * 6)
        return np.asarray(values, dtype=float)

    return [{"type": "ineq", "fun": constraints}]


def build_objective(initial: DirectFitResult, tail_weight: float, mode: str = "mae") -> Callable[[np.ndarray], float]:
    """Create the surface fitting objective.

    ``mode="bp3434"`` evaluates reference splines at the generated model
    coordinates, matching the original BP3434 script.  ``mode="mae"`` evaluates
    the generated model on the fixed reference grid, matching the error metric
    reported in this package.  The latter is the default because it avoids the
    optimiser improving one metric while worsening the displayed MAE.
    """
    ref = initial.reference
    tail_x, tail_thickness = _tail_thickness_points(initial)
    mode_key = mode.lower()

    def objective(vector: np.ndarray) -> float:
        try:
            params = vector_to_params(vector, seed_k_t=initial.params.k_t, te_c2=initial.params.te_c2)
            geometry = generate_airfoil(params, n_per_segment=max(160, len(ref.x_eval) // 3))
            controls = geometry["controls"]
            if controls.trailing_residual > CONNECTION_RESIDUAL_TOL:
                return 1e4 + 1e4 * float(controls.trailing_residual)
            upper_x = geometry["upper_x"]
            lower_x = geometry["lower_x"]
            if _surface_outside_reference(upper_x, lower_x, ref.contour[:, 0]):
                return 1e6

            if mode_key == "bp3434":
                reference_upper_y = ref.upper_spline(upper_x)
                reference_lower_y = ref.lower_spline(lower_x)
                error = np.linalg.norm(geometry["upper_y"] - reference_upper_y)
                error += np.linalg.norm(geometry["lower_y"] - reference_lower_y)
            elif mode_key in {"mae", "reference_grid"}:
                upper_error, lower_error = _surface_errors_on_reference_grid(geometry, initial)
                all_error = np.concatenate([upper_error, lower_error])
                error = np.mean(np.abs(all_error))
            elif mode_key in {"rms", "reference_l2"}:
                upper_error, lower_error = _surface_errors_on_reference_grid(geometry, initial)
                all_error = np.concatenate([upper_error, lower_error])
                error = np.linalg.norm(all_error)
            else:
                raise ValueError("objective mode must be 'mae', 'rms', or 'bp3434'.")

            if len(tail_x) > 0:
                thickness_spline = interpolate.CubicSpline(geometry["thickness_x"], geometry["thickness_y"])
                tail_delta = thickness_spline(tail_x) - tail_thickness
                tail_error = np.mean(np.abs(tail_delta)) if mode_key in {"mae", "reference_grid"} else np.linalg.norm(tail_delta)
                error += tail_weight * tail_error
            if np.isfinite(controls.leading_curvature_jump):
                error += 1e-6 * min(abs(float(controls.leading_curvature_jump)), 1000.0)
            if np.isfinite(controls.trailing_curvature_jump):
                error += 1e-6 * min(abs(float(controls.trailing_curvature_jump)), 1000.0)
            if not controls.root_success and np.isfinite(controls.q_residual):
                error += 1e-4 * min(abs(float(controls.q_residual)), 100.0)
            return float(error)
        except Exception:
            return 1e6

    return objective


def make_result_from_vector(initial: DirectFitResult, vector: np.ndarray) -> DirectFitResult:
    """Generate a ``DirectFitResult``-compatible object from an optimised vector."""
    params = vector_to_params(vector, seed_k_t=initial.params.k_t, te_c2=initial.params.te_c2)
    geometry = generate_airfoil(params, n_per_segment=max(240, len(initial.geometry["thickness_x"]) // 2))
    upper_error, lower_error = _surface_errors_on_reference_grid(geometry, initial)
    error = np.concatenate([upper_error, lower_error])
    controls = geometry["controls"]
    return DirectFitResult(
        airfoil=initial.airfoil,
        params=params,
        mae=float(np.mean(np.abs(error))),
        max_abs_error=float(np.max(np.abs(error))),
        rms=float(np.sqrt(np.mean(error**2))),
        q_residual=float(controls.q_residual),
        q_root_success=bool(controls.root_success or controls.q_residual <= CONNECTION_RESIDUAL_TOL),
        geometry=geometry,
        reference=initial.reference,
    )


def _surface_errors_on_reference_grid(geometry: dict[str, np.ndarray], initial: DirectFitResult) -> tuple[np.ndarray, np.ndarray]:
    from .fit import surface_errors

    return surface_errors(geometry, initial.reference)


def _continuous_enough(result: DirectFitResult) -> bool:
    """Return whether a generated result satisfies the Bezier/ellipse C0+C1 equations."""
    return bool(result.q_residual <= CONNECTION_RESIDUAL_TOL)


def _negative_curvature_bounds(seed: float) -> tuple[float, float]:
    seed = float(seed if seed < -1e-5 else -1e-4)
    lower = max(-300.0, 3.0 * seed)
    upper = min(-1e-5, 0.2 * seed)
    if lower >= upper:
        lower, upper = -1.0, -1e-5
    return lower, upper


def _has_auxiliary_closed_trailing_edge(result: DirectFitResult) -> bool:
    ref = result.reference
    return bool(
        np.allclose(ref.contour[0], ref.contour[-1], atol=1e-10, rtol=0.0)
        and abs(ref.native_thickness_y[-1]) < 1e-10
        and len(ref.native_thickness_y) >= 2
        and ref.native_thickness_y[-2] > 1e-6
    )


def _tail_thickness_points(result: DirectFitResult) -> tuple[np.ndarray, np.ndarray]:
    if not _has_auxiliary_closed_trailing_edge(result):
        return np.array([], dtype=float), np.array([], dtype=float)
    ref = result.reference
    indices = np.where(
        (ref.native_thickness_x >= 0.9)
        & (np.arange(len(ref.native_thickness_x)) < len(ref.native_thickness_x) - 1)
    )[0]
    return ref.native_thickness_x[indices], ref.native_thickness_y[indices]


def _surface_outside_reference(upper_x: np.ndarray, lower_x: np.ndarray, reference_x: np.ndarray) -> bool:
    x_min = float(np.min(reference_x)) - 0.03
    x_max = float(np.max(reference_x)) + 0.03
    return bool(
        np.min(upper_x) < x_min
        or np.max(upper_x) > x_max
        or np.min(lower_x) < x_min
        or np.max(lower_x) > x_max
    )
