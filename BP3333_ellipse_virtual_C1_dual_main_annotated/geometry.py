"""Reference airfoil reading and direct parameter extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate


@dataclass(frozen=True)
class ReferenceAirfoil:
    """Reference contour and derived thickness/camber distributions."""

    name: str
    contour: np.ndarray
    x_eval: np.ndarray
    upper_y: np.ndarray
    lower_y: np.ndarray
    camber_y: np.ndarray
    thickness_y: np.ndarray
    camber_slope: np.ndarray
    thickness_slope: np.ndarray
    upper_spline: interpolate.PchipInterpolator
    lower_spline: interpolate.PchipInterpolator
    native_camber_x: np.ndarray
    native_camber_y: np.ndarray
    native_thickness_x: np.ndarray
    native_thickness_y: np.ndarray


def read_airfoil(path: str | Path, n_eval: int = 701) -> ReferenceAirfoil:
    """Read an airfoil file and sample it using the original BP3434 workflow.

    The surface split and camber/thickness extraction intentionally mirrors
    ``python-par/Parameterisations.py::GetReferenceThicknessCamber``: choose a
    common camber x-grid from the side with fewer points, interpolate the other
    side onto it, then compute thickness with the camber-angle correction.
    """
    airfoil_path = Path(path)
    contour = np.genfromtxt(airfoil_path, dtype=float, skip_header=1)
    contour = contour[np.all(np.isfinite(contour), axis=1)]
    if contour.ndim != 2 or contour.shape[1] < 2:
        raise ValueError(f"No coordinate data found in {airfoil_path}.")

    contour = _normalise_chord(contour[:, :2])
    idx_le = int(np.argmin(np.abs(contour[:, 0])))
    contour[:, 1] -= contour[idx_le, 1]

    upper = contour[: idx_le + 1][::-1]
    lower = contour[idx_le:]
    upper_x, upper_y = _unique_xy(upper[:, 0], upper[:, 1])
    lower_x, lower_y = _unique_xy(lower[:, 0], lower[:, 1])
    upper_spline = interpolate.PchipInterpolator(upper_x, upper_y, extrapolate=True)
    lower_spline = interpolate.PchipInterpolator(lower_x, lower_y, extrapolate=True)
    x_camber_native, camber_native, x_thickness_native, thickness_native = _reference_distributions_python_par(
        contour,
        idx_le,
    )

    beta = np.linspace(0.0, np.pi, n_eval)
    x_eval = 0.5 * (1.0 - np.cos(beta))
    yu = np.asarray(upper_spline(x_eval), dtype=float)
    yl = np.asarray(lower_spline(x_eval), dtype=float)
    camber_spline = _pchip_from_xy(x_camber_native, camber_native)
    thickness_spline = _pchip_from_xy(x_thickness_native, thickness_native)
    camber = np.asarray(camber_spline(x_eval), dtype=float)
    thickness = np.maximum(np.asarray(thickness_spline(x_eval), dtype=float), 0.0)
    camber_slope = _safe_derivative(x_eval, camber)
    thickness_slope = _safe_derivative(x_eval, thickness)

    case_name = airfoil_path.stem if airfoil_path.suffix.lower() == ".dat" else airfoil_path.name.replace(".", "_")
    return ReferenceAirfoil(
        name=case_name,
        contour=contour,
        x_eval=x_eval,
        upper_y=yu,
        lower_y=yl,
        camber_y=camber,
        thickness_y=thickness,
        camber_slope=camber_slope,
        thickness_slope=thickness_slope,
        upper_spline=upper_spline,
        lower_spline=lower_spline,
        native_camber_x=x_camber_native,
        native_camber_y=camber_native,
        native_thickness_x=x_thickness_native,
        native_thickness_y=thickness_native,
    )


def extract_seed(ref: ReferenceAirfoil) -> dict[str, float]:
    """Estimate BP3333-ellipse physical parameters directly from the airfoil."""
    x = ref.x_eval
    thickness = np.maximum(ref.thickness_y, 0.0)
    camber = ref.camber_y

    search = np.where(x > 0.01)[0]
    idx_t = int(search[np.argmax(thickness[search])]) if len(search) else int(np.argmax(thickness))
    idx_c = int(np.argmax(camber))

    x_t = float(np.clip(x[idx_t], 0.03, 0.95))
    y_t = float(max(thickness[idx_t], 1e-5))
    k_t = float(min(_local_second_derivative(x, thickness, idx_t), -1e-4))

    x_c = float(np.clip(x[idx_c], 0.02, 0.98))
    y_c = float(max(camber[idx_c], 0.0))
    k_c = float(min(_local_second_derivative(x, camber, idx_c), -1e-4))
    if abs(y_c) < 1e-6:
        x_c = 0.4
        y_c = 0.0
        k_c = -0.05

    z_te, dz_te = _estimate_trailing_edge_values(ref)
    if abs(z_te) < 1e-8:
        z_te = 0.0

    beta_te = float(max(np.arctan(-_tail_slope(x, thickness)), 1e-4))
    gamma_le = float(np.arctan(_nose_slope(x, camber)))
    alpha_te = float(np.arctan(-_tail_slope(x, camber)))
    if y_c == 0.0 and abs(z_te) < 1e-8:
        gamma_le = 0.0
        alpha_te = 0.0

    return {
        "x_t": x_t,
        "y_t": y_t,
        "k_t": k_t,
        "beta_te": beta_te,
        "gamma_le": gamma_le,
        "x_c": x_c,
        "y_c": y_c,
        "k_c": k_c,
        "alpha_te": alpha_te,
        "dz_te": dz_te,
        "z_te": z_te,
    }


def _estimate_trailing_edge_values(ref: ReferenceAirfoil) -> tuple[float, float]:
    """Estimate TE camber and half-thickness, matching python-par's safeguard."""
    z_te = float(ref.native_camber_y[-1])
    dz_te = float(ref.native_thickness_y[-1])

    if len(ref.native_thickness_y) < 2:
        return z_te, dz_te

    closed_contour = np.allclose(ref.contour[0], ref.contour[-1], atol=1e-10, rtol=0.0)
    has_finite_tail_thickness = ref.native_thickness_y[-2] > 1e-6

    if closed_contour and abs(dz_te) < 1e-10 and has_finite_tail_thickness:
        z_te = float(ref.native_camber_y[-2])
        dz_te = float(ref.native_thickness_y[-2])
    return z_te, dz_te


def _normalise_chord(points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, dtype=float).copy()
    x_min = float(np.min(out[:, 0]))
    x_max = float(np.max(out[:, 0]))
    if -1e-2 <= x_min <= 1e-2 and 0.5 <= x_max <= 1.5:
        # Many supplied airfoil files are already unit chord but contain a tiny
        # negative-x overshoot near the nose.  The original BP3434 script keeps
        # that coordinate and chooses the point nearest x=0 as LE; shifting by
        # x_min would change the upper/lower split and corrupt TE pairing.
        chord = x_max
        x_origin = 0.0
    else:
        chord = x_max - x_min
        x_origin = x_min
    if chord <= 0.0:
        raise ValueError("Airfoil chord length is zero.")
    out[:, 0] = (out[:, 0] - x_origin) / chord
    out[:, 1] = out[:, 1] / chord
    return out


def _unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    unique_x: list[float] = []
    unique_y: list[float] = []
    start = 0
    while start < len(xs):
        stop = start + 1
        while stop < len(xs) and abs(xs[stop] - xs[start]) < 1e-12:
            stop += 1
        unique_x.append(float(np.mean(xs[start:stop])))
        unique_y.append(float(np.mean(ys[start:stop])))
        start = stop
    return np.asarray(unique_x), np.asarray(unique_y)


def _unique_columns(x: np.ndarray, *columns: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sort by x and average duplicate x locations for several aligned arrays."""
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, dtype=float)[order]
    cols = [np.asarray(column, dtype=float)[order] for column in columns]
    unique_x: list[float] = []
    unique_cols: list[list[float]] = [[] for _ in cols]
    start = 0
    while start < len(xs):
        stop = start + 1
        while stop < len(xs) and abs(xs[stop] - xs[start]) < 1e-12:
            stop += 1
        unique_x.append(float(np.mean(xs[start:stop])))
        for out, col in zip(unique_cols, cols):
            out.append(float(np.mean(col[start:stop])))
        start = stop
    return (np.asarray(unique_x), *(np.asarray(col) for col in unique_cols))


def _reference_distributions_python_par(
    contour: np.ndarray,
    idx_le: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract native camber/thickness arrays following the original script."""
    upper = contour[: idx_le + 1]
    lower = contour[idx_le:]
    upper_flip = upper[::-1]
    le_y = upper_flip[0, 1]

    if len(upper) > len(lower):
        x_camber = lower[:, 0]
        lower_coords = lower[:, 1] - le_y
        upper_coords = _fitpack_interp(upper_flip[:, 0], upper_flip[:, 1], x_camber) - le_y
    elif len(upper) < len(lower):
        x_camber = upper_flip[:, 0]
        upper_coords = upper_flip[:, 1] - lower[0, 1]
        lower_coords = _fitpack_interp(lower[:, 0], lower[:, 1], x_camber) - lower[0, 1]
    else:
        x_camber = 0.5 * (upper_flip[:, 0] + lower[:, 0])
        upper_coords = upper_flip[:, 1] - upper_flip[0, 1]
        lower_coords = lower[:, 1] - upper_flip[0, 1]

    order = np.argsort(x_camber, kind="mergesort")
    x_camber = np.asarray(x_camber, dtype=float)[order]
    upper_coords = np.asarray(upper_coords, dtype=float)[order]
    lower_coords = np.asarray(lower_coords, dtype=float)[order]
    x_camber, upper_coords, lower_coords = _unique_columns(x_camber, upper_coords, lower_coords)

    camber = 0.5 * (upper_coords + lower_coords)
    camber_angle = np.arctan(_safe_derivative(x_camber, camber))
    thickness = (upper_coords - lower_coords) / (2.0 * np.cos(camber_angle))
    x_thickness = x_camber - thickness * np.sin(camber_angle)
    return x_camber, camber, x_thickness, np.maximum(thickness, 0.0)


def _fitpack_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Evaluate the BP3434 cubic FITPACK interpolant on a cleaned x sequence.

    The original script uses ``interpolate.make_splrep(..., k=3, s=0)``
    directly.  Some supplied files contain a tiny leading-edge overshoot, so
    the split surface is not strictly increasing in x.  FITPACK requires a
    strict sequence; sorting and averaging repeated x values preserves the
    same interpolation model without changing the BP3434 camber/thickness
    formulas.
    """
    x_fit, y_fit = _unique_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    degree = min(3, len(x_fit) - 1)
    if degree < 1:
        raise ValueError("At least two unique x values are required for interpolation.")
    return np.asarray(interpolate.make_splrep(x=x_fit, y=y_fit, k=degree, s=0)(x_new), dtype=float)


def _pchip_from_xy(x: np.ndarray, y: np.ndarray) -> interpolate.PchipInterpolator:
    x_unique, y_unique = _unique_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return interpolate.PchipInterpolator(x_unique, y_unique, extrapolate=True)


def _safe_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(np.gradient(y, x), dtype=float)


def _local_second_derivative(x: np.ndarray, y: np.ndarray, idx: int, half_window: int = 8) -> float:
    lo = max(0, idx - half_window)
    hi = min(len(x), idx + half_window + 1)
    if hi - lo < 5:
        return -0.1
    coeff = np.polyfit(x[lo:hi] - x[idx], y[lo:hi], deg=2)
    return float(2.0 * coeff[0])


def _tail_slope(x: np.ndarray, y: np.ndarray, start: float = 0.90) -> float:
    mask = x >= start
    if np.count_nonzero(mask) < 6:
        mask = np.arange(len(x)) >= len(x) - 8
    slope, _ = np.polyfit(x[mask], y[mask], deg=1)
    return float(slope)


def _nose_slope(x: np.ndarray, y: np.ndarray, stop: float = 0.04) -> float:
    mask = x <= stop
    if np.count_nonzero(mask) < 6:
        mask = np.arange(len(x)) < 8
    slope, _ = np.polyfit(x[mask], y[mask], deg=1)
    return float(slope)
