"""Output helpers for BP3333 virtual-thickness runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fit import DirectFitResult
from .optimization import OptimizedFitResult


def save_coordinates(result: DirectFitResult, path: str | Path) -> None:
    """Save reconstructed contour coordinates in upper-then-lower order."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = result.geometry
    upper = np.column_stack([g["upper_x"][::-1], g["upper_y"][::-1]])
    lower = np.column_stack([g["lower_x"], g["lower_y"]])
    contour = np.vstack([upper, lower])
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"BP3333 virtual-thickness reconstructed {result.airfoil}\n")
        for x, y in contour:
            handle.write(f"{x:.12e}  {y:.12e}\n")


def save_parameters(result: DirectFitResult, path: str | Path) -> None:
    """Save direct parameters and error metrics as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    controls = result.geometry["controls"]
    tail = _tail_payload(controls)
    payload = {
        "airfoil": result.airfoil,
        "parameters": result.params.to_dict(),
        "derived": {
            "q": float(controls.q),
            "u_s": float(controls.u_s),
            "v_r": float(controls.v_r),
            "equation_residual_norm": float(result.q_residual),
            "equation_success": bool(result.q_root_success),
            "kappa_t_star": float(controls.kappa_t_star),
            "leading_curvature_jump": float(controls.leading_curvature_jump),
            "trailing_curvature_jump": float(controls.trailing_curvature_jump),
            "x_tangent": float(controls.ellipse.x_tangent),
            "y_tangent": float(controls.ellipse.y_tangent),
            "tangent_slope": float(controls.ellipse.slope_tangent),
            "leading_thickness_x_controls": controls.x_le.tolist(),
            "leading_thickness_y_controls": controls.y_le.tolist(),
            "trailing_thickness_x_controls": controls.x_te.tolist(),
            "trailing_thickness_y_controls": controls.y_te.tolist(),
            **tail,
        },
        "error": {
            "mae": float(result.mae),
            "max_abs_error": float(result.max_abs_error),
            "rms": float(result.rms),
            "q_root_success": bool(result.q_root_success),
        },
        "notes": [
            "Direct parameter estimate only; no global optimisation is applied.",
            "C1+G2-at-maximum-thickness version: ellipse-Bezier tangent curvature is diagnostic only.",
            "Trailing virtual thickness y0_te is an explicit parameter in this version.",
            "Maximum-thickness curvature is computed from the controls as kappa_t_star.",
        ],
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_optimization_parameters(result: OptimizedFitResult, path: str | Path) -> None:
    """Save optimised parameters, bounds, and optimisation diagnostics."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_controls = result.optimized.geometry["controls"]
    tail = _tail_payload(optimized_controls)
    payload = {
        "airfoil": result.optimized.airfoil,
        "parameters": result.optimized.params.to_dict(),
        "optimization": {
            "method": result.method,
            "objective_mode": result.objective_mode,
            "success": bool(result.success),
            "status": int(result.status),
            "message": result.message,
            "iterations": int(result.iterations),
            "evaluations": int(result.evaluations),
            "objective_initial": float(result.objective_initial),
            "objective_optimized": float(result.objective_optimized),
            "variables": {
                name: {
                    "initial": float(result.initial_vector[idx]),
                    "optimized": float(result.optimized_vector[idx]),
                    "lower_bound": float(result.lower_bounds[idx]),
                    "upper_bound": float(result.upper_bounds[idx]),
                }
                for idx, name in enumerate(result.variable_names)
            },
        },
        "derived": {
            "q": float(optimized_controls.q),
            "u_s": float(optimized_controls.u_s),
            "v_r": float(optimized_controls.v_r),
            "equation_residual_norm": float(result.optimized.q_residual),
            "equation_success": bool(result.optimized.q_root_success),
            "kappa_t_star": float(optimized_controls.kappa_t_star),
            "leading_curvature_jump": float(optimized_controls.leading_curvature_jump),
            "trailing_curvature_jump": float(optimized_controls.trailing_curvature_jump),
            "x_tangent": float(optimized_controls.ellipse.x_tangent),
            "y_tangent": float(optimized_controls.ellipse.y_tangent),
            "tangent_slope": float(optimized_controls.ellipse.slope_tangent),
            "leading_thickness_x_controls": optimized_controls.x_le.tolist(),
            "leading_thickness_y_controls": optimized_controls.y_le.tolist(),
            "trailing_thickness_x_controls": optimized_controls.x_te.tolist(),
            "trailing_thickness_y_controls": optimized_controls.y_te.tolist(),
            **tail,
        },
        "error": {
            "direct_mae": float(result.initial.mae),
            "optimized_mae": float(result.optimized.mae),
            "direct_rms": float(result.initial.rms),
            "optimized_rms": float(result.optimized.rms),
            "direct_max_abs_error": float(result.initial.max_abs_error),
            "optimized_max_abs_error": float(result.optimized.max_abs_error),
        },
        "notes": [
            "Optimisation starts from the direct BP3333 virtual-thickness C1+G2 estimate.",
            "P/Q control points are recomputed from de_virtual_C1_mtC1 formulas after every variable update.",
            "y0_te is optimised together with the trailing ellipse parameters.",
            "Ellipse tangent curvature jumps are soft diagnostics, not hard constraints.",
        ],
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _tail_payload(controls) -> dict[str, object]:
    """Return JSON fields for the optional trailing ellipse."""
    tail = getattr(controls, "tail_ellipse", None)
    if tail is None:
        return {"trailing_ellipse": None}
    return {
        "trailing_ellipse": {
            "a": float(tail.a),
            "b": float(tail.b),
            "theta": float(tail.theta),
            "x_tangent": float(tail.x_tangent),
            "y_tangent": float(tail.y_tangent),
            "slope_tangent": float(tail.slope_tangent),
            "curvature": float(tail.curvature),
            "c2_residual": float(tail.c2_residual),
            "c2_success": bool(tail.c2_success),
        }
    }


def save_summary(rows: list[dict[str, float | str]], path: str | Path) -> None:
    """Save a compact CSV summary for all reconstructed test airfoils."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            values = []
            for name in fieldnames:
                value = row[name]
                values.append(f"{value:.12e}" if isinstance(value, float) else str(value))
            handle.write(",".join(values) + "\n")
