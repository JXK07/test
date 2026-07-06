"""Direct BP3333 virtual-thickness C1 construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate

from .ellipse import fit_ellipse_head, fit_trailing_ellipse, make_ellipse_head, trailing_theta_from_beta
from .geometry import ReferenceAirfoil, extract_seed, read_airfoil
from .model import BP3333EllipseParameters, generate_airfoil


CONNECTION_RESIDUAL_TOL = 3e-3


@dataclass(frozen=True)
class DirectFitResult:
    """Result of one direct BP3333 virtual C1 construction."""

    airfoil: str
    params: BP3333EllipseParameters
    mae: float
    max_abs_error: float
    rms: float
    q_residual: float
    q_root_success: bool
    geometry: dict[str, np.ndarray]
    reference: ReferenceAirfoil


def build_direct_fit(
    path: str | Path,
    n_eval: int = 701,
    n_per_segment: int = 240,
    axis_ratio: float = 3.0,
    theta: float = 0.65,
    y0_le_ratio: float = 0.25,
    y0_te_ratio: float = 0.05,
    ellipse_fit_limit: float = 0.025,
    te_ellipse: bool = True,
    te_c2: bool = False,
    te_fit_start: float = 0.90,
) -> DirectFitResult:
    """Estimate parameters directly and build a BP3333 virtual C1 airfoil."""
    ref = read_airfoil(path, n_eval=n_eval)
    seed = extract_seed(ref)
    ellipse_a, ellipse_b = fit_ellipse_head(
        ref.x_eval,
        ref.thickness_y,
        axis_ratio=axis_ratio,
        fit_limit=ellipse_fit_limit,
    )
    if not te_ellipse:
        raise ValueError("The C1+G2 virtual-thickness model requires a trailing ellipse.")
    te_a, te_b = fit_trailing_ellipse(ref.x_eval, ref.thickness_y, fit_start=te_fit_start)
    te_theta = trailing_theta_from_beta(te_a, te_b, seed["beta_te"])

    theta_candidates = _unique_values([theta, *np.linspace(0.20, 1.20, 9)])
    y0_le_candidates = _unique_values([y0_le_ratio, 0.15, 0.25, 0.35, 0.50])
    y0_te_candidates = _unique_values([y0_te_ratio, 0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.75])
    best: DirectFitResult | None = None
    best_continuous: DirectFitResult | None = None
    last_error: Exception | None = None

    for theta_candidate in theta_candidates:
        try:
            head = make_ellipse_head(ellipse_a, ellipse_b, theta=float(theta_candidate), max_x=seed["x_t"])
        except Exception as exc:
            last_error = exc
            continue
        for y0_le_candidate in y0_le_candidates:
            y0_le = float(np.clip(y0_le_candidate * head.y_tangent, 1e-6, 0.85 * seed["y_t"]))
            for y0_te_candidate in y0_te_candidates:
                y0_te = float(np.clip(max(seed["dz_te"], y0_te_candidate * seed["y_t"]), 0.0, 0.90 * seed["y_t"]))
                params = BP3333EllipseParameters(
                    **seed,
                    ellipse_a=head.a,
                    ellipse_b=head.b,
                    theta=head.theta,
                    y0_le=y0_le,
                    y0_te=y0_te,
                    te_ellipse_a=te_a,
                    te_ellipse_b=te_b,
                    te_theta=te_theta,
                    te_c2=te_c2,
                )
                try:
                    result = _result_from_params(ref, params, n_per_segment)
                except Exception as exc:
                    last_error = exc
                    continue
                if _is_continuous_result(result) and (best_continuous is None or result.mae < best_continuous.mae):
                    best_continuous = result
                if best is None or result.mae < best.mae:
                    best = result
    if best_continuous is not None:
        return best_continuous
    if best is None:
        raise RuntimeError("No valid BP3333 virtual C1+G2 direct estimate.") from last_error
    raise RuntimeError(
        "No C0/C1-continuous BP3333 virtual C1+G2 direct estimate. "
        f"Best residual was {best.q_residual:.3e}; increase the parameter sweep or relax bounds."
    ) from last_error


def _result_from_params(ref: ReferenceAirfoil, params: BP3333EllipseParameters, n_per_segment: int) -> DirectFitResult:
    """Generate geometry and errors for one direct candidate."""
    geometry = generate_airfoil(params, n_per_segment=n_per_segment)
    upper_error, lower_error = surface_errors(geometry, ref)
    error = np.concatenate([upper_error, lower_error])
    controls = geometry["controls"]
    return DirectFitResult(
        airfoil=ref.name,
        params=params,
        mae=float(np.mean(np.abs(error))),
        max_abs_error=float(np.max(np.abs(error))),
        rms=float(np.sqrt(np.mean(error**2))),
        q_residual=float(controls.q_residual),
        q_root_success=bool(controls.root_success or controls.q_residual <= CONNECTION_RESIDUAL_TOL),
        geometry=geometry,
        reference=ref,
    )


def _unique_values(values) -> list[float]:
    """Return sorted unique floating values while preserving practical precision."""
    return sorted({round(float(value), 10) for value in values})


def _is_continuous_result(result: DirectFitResult) -> bool:
    """Return whether the solved Bezier/ellipse joins are numerically C0/C1."""
    return bool(result.q_residual <= CONNECTION_RESIDUAL_TOL)


def sweep_direct_fit(
    path: str | Path,
    theta_values: np.ndarray,
    axis_ratio_values: np.ndarray | None = None,
    n_eval: int = 701,
    n_per_segment: int = 240,
    ellipse_fit_limit: float = 0.025,
    y0_le_ratio_values: np.ndarray | None = None,
    y0_te_ratio_values: np.ndarray | None = None,
    min_control_spacing: float = 1e-3,
    te_ellipse: bool = True,
    te_c2: bool = False,
    te_fit_start: float = 0.90,
) -> DirectFitResult:
    """Try a small user-requested theta/axis-ratio grid and keep lowest MAE.

    This is intentionally separate from ``build_direct_fit`` so the default
    workflow remains the no-optimisation direct estimate requested by the user.
    """
    ratios = np.array([3.0]) if axis_ratio_values is None else np.asarray(axis_ratio_values, dtype=float)
    y0_ratios = np.array([0.15, 0.25, 0.35]) if y0_le_ratio_values is None else np.asarray(y0_le_ratio_values, dtype=float)
    y0_te_ratios = np.array([0.05]) if y0_te_ratio_values is None else np.asarray(y0_te_ratio_values, dtype=float)
    best_preferred: DirectFitResult | None = None
    best_valid: DirectFitResult | None = None
    best_any: DirectFitResult | None = None
    last_error: Exception | None = None
    for ratio in ratios:
        for y0_ratio in y0_ratios:
            for y0_te_ratio in y0_te_ratios:
                for theta in np.asarray(theta_values, dtype=float):
                    try:
                        result = build_direct_fit(
                            path,
                            n_eval=n_eval,
                            n_per_segment=n_per_segment,
                            axis_ratio=float(ratio),
                            theta=float(theta),
                            y0_le_ratio=float(y0_ratio),
                            y0_te_ratio=float(y0_te_ratio),
                            ellipse_fit_limit=ellipse_fit_limit,
                            te_ellipse=te_ellipse,
                            te_c2=te_c2,
                            te_fit_start=te_fit_start,
                        )
                    except Exception as exc:
                        last_error = exc
                        continue
                    if _is_continuous_result(result):
                        controls = result.geometry["controls"]
                        x_le = controls.x_le
                        spacing_ok = bool(
                            (x_le[1] - x_le[0] >= min_control_spacing)
                            and (x_le[2] - x_le[1] >= min_control_spacing)
                        )
                        if spacing_ok and (best_preferred is None or result.mae < best_preferred.mae):
                            best_preferred = result
                        if best_valid is None or result.mae < best_valid.mae:
                            best_valid = result
                    if best_any is None or result.mae < best_any.mae:
                        best_any = result
    if best_preferred is not None:
        return best_preferred
    if best_valid is not None:
        return best_valid
    if best_any is None:
        raise RuntimeError("No valid BP3333 virtual-thickness C1 case in sweep.") from last_error
    return best_any


def surface_errors(geometry: dict[str, np.ndarray], ref: ReferenceAirfoil) -> tuple[np.ndarray, np.ndarray]:
    """Return upper/lower vertical errors on the reference x grid."""
    upper = _safe_surface_spline(geometry["upper_x"], geometry["upper_y"])
    lower = _safe_surface_spline(geometry["lower_x"], geometry["lower_y"])
    upper_error = upper(ref.x_eval) - ref.upper_y
    lower_error = lower(ref.x_eval) - ref.lower_y
    return np.asarray(upper_error, dtype=float), np.asarray(lower_error, dtype=float)


def _safe_surface_spline(x: np.ndarray, y: np.ndarray):
    order = np.argsort(x, kind="mergesort")
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    unique_x: list[float] = []
    unique_y: list[float] = []
    start = 0
    while start < len(x_sorted):
        stop = start + 1
        while stop < len(x_sorted) and abs(x_sorted[stop] - x_sorted[start]) < 1e-10:
            stop += 1
        unique_x.append(float(np.mean(x_sorted[start:stop])))
        unique_y.append(float(np.mean(y_sorted[start:stop])))
        start = stop
    x_unique = np.asarray(unique_x)
    y_unique = np.asarray(unique_y)

    def evaluate(x_new: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(x_new, dtype=float), x_unique, y_unique, left=y_unique[0], right=y_unique[-1])

    return evaluate
