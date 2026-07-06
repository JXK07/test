"""Shared BP3333 virtual-thickness C1 running utilities.

This module intentionally contains the reusable workflow only:

* :func:`run_one` reconstructs one airfoil, optionally optimises it, writes
  coordinates/JSON/plots, and returns scalar metrics.
* :func:`run_all` applies the same workflow to every supported file in a test
  directory and writes a CSV summary.

Use ``main_spyder.py`` for interactive Spyder debugging with parameters written
directly in the file.  Use ``main_cli.py`` for command-line execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from BP3333_ellipse_virtual_C1_dual_main_annotated.fit import build_direct_fit, sweep_direct_fit
    from BP3333_ellipse_virtual_C1_dual_main_annotated.io import save_coordinates, save_optimization_parameters, save_parameters, save_summary
    from BP3333_ellipse_virtual_C1_dual_main_annotated.optimization import optimize_fit
    from BP3333_ellipse_virtual_C1_dual_main_annotated.plotting import plot_optimization_result, plot_result
else:
    from .fit import build_direct_fit, sweep_direct_fit
    from .io import save_coordinates, save_optimization_parameters, save_parameters, save_summary
    from .optimization import optimize_fit
    from .plotting import plot_optimization_result, plot_result


def run_one(
    airfoil: str | Path,
    output_dir: str | Path,
    axis_ratio: float = 3.0,
    theta: float = 1.2,
    y0_le_ratio: float = 0.25,
    y0_te_ratio: float = 0.05,
    ellipse_fit_limit: float = 0.025,
    te_ellipse: bool = True,
    te_c2: bool = False,
    te_fit_start: float = 0.90,
    n_per_segment: int = 240,
    theta_sweep: str | None = None,
    axis_ratio_sweep: str | None = None,
    min_control_spacing: float = 1e-3,
    optimize: bool = False,
    optimizer: str = "slsqp",
    objective_mode: str = "mae",
    maxiter: int = 500,
    tail_weight: float = 0.0,
    ga_population: int = 96,
    ga_generations: int = 120,
    random_seed: int | None = 7,
    make_plot: bool = True,
    show_plot: bool = False,
) -> dict[str, float]:
    """Run one direct reconstruction and write outputs."""
    airfoil_path = _resolve_path(airfoil)
    if theta_sweep:
        result = sweep_direct_fit(
            airfoil_path,
            theta_values=_parse_sweep(theta_sweep),
            axis_ratio_values=_parse_sweep(axis_ratio_sweep) if axis_ratio_sweep else np.array([axis_ratio]),
            y0_le_ratio_values=np.array([y0_le_ratio]),
            y0_te_ratio_values=np.array([y0_te_ratio]),
            n_per_segment=n_per_segment,
            ellipse_fit_limit=ellipse_fit_limit,
            min_control_spacing=min_control_spacing,
            te_ellipse=te_ellipse,
            te_c2=te_c2,
            te_fit_start=te_fit_start,
        )
    else:
        result = build_direct_fit(
            airfoil_path,
            n_per_segment=n_per_segment,
            axis_ratio=axis_ratio,
            theta=theta,
            y0_le_ratio=y0_le_ratio,
            y0_te_ratio=y0_te_ratio,
            ellipse_fit_limit=ellipse_fit_limit,
            te_ellipse=te_ellipse,
            te_c2=te_c2,
            te_fit_start=te_fit_start,
        )
    out_dir = Path(output_dir)
    if optimize:
        opt_result = optimize_fit(
            result,
            maxiter=maxiter,
            min_control_spacing=min_control_spacing,
            tail_weight=tail_weight,
            method=optimizer,
            objective_mode=objective_mode,
            ga_population=ga_population,
            ga_generations=ga_generations,
            random_seed=random_seed,
        )
        save_coordinates(result, out_dir / f"{result.airfoil}_bp3333_virtual_mtG2_direct.dat")
        save_parameters(result, out_dir / f"{result.airfoil}_bp3333_virtual_mtG2_direct.json")
        save_coordinates(opt_result.optimized, out_dir / f"{result.airfoil}_bp3333_virtual_mtG2_optimized.dat")
        save_optimization_parameters(opt_result, out_dir / f"{result.airfoil}_bp3333_virtual_mtG2_optimized.json")
        if make_plot:
            plot_optimization_result(
                opt_result,
                save_path=out_dir / f"{result.airfoil}_bp3333_virtual_mtG2_optimization.png",
                show=show_plot,
            )
        print(
            f"{result.airfoil:14s}  direct MAE={result.mae:.6e}  opt MAE={opt_result.optimized.mae:.6e}  "
            f"RMS={opt_result.optimized.rms:.6e}  max={opt_result.optimized.max_abs_error:.6e}  "
            f"obj={opt_result.objective_initial:.4e}->{opt_result.objective_optimized:.4e}  "
            f"{opt_result.method} status={opt_result.status}",
            flush=True,
        )
        return {
            "airfoil": result.airfoil,
            "direct_mae": result.mae,
            "optimized_mae": opt_result.optimized.mae,
            "optimized_rms": opt_result.optimized.rms,
            "optimized_max_abs_error": opt_result.optimized.max_abs_error,
            "objective_initial": opt_result.objective_initial,
            "objective_optimized": opt_result.objective_optimized,
            "theta": float(opt_result.optimized.params.theta),
            "axis_ratio": float(opt_result.optimized.params.ellipse_a / opt_result.optimized.params.ellipse_b),
            "q_root_success": "true" if opt_result.optimized.q_root_success else "false",
            "slsqp_status": float(opt_result.status),
            "method": opt_result.method,
            "objective_mode": opt_result.objective_mode,
        }

    save_coordinates(result, out_dir / f"{result.airfoil}_bp3333_virtual_mtG2.dat")
    save_parameters(result, out_dir / f"{result.airfoil}_bp3333_virtual_mtG2.json")
    if make_plot:
        plot_result(result, save_path=out_dir / f"{result.airfoil}_bp3333_virtual_mtG2.png", show=show_plot)
    print(
        f"{result.airfoil:14s}  MAE={result.mae:.6e}  RMS={result.rms:.6e}  "
        f"max={result.max_abs_error:.6e}  p={result.geometry['controls'].p:.4e}  "
        f"q={result.geometry['controls'].q:.4e}  qres={result.q_residual:.3e}  "
        f"theta={result.params.theta:.4f}  root={'OK' if result.q_root_success else 'FALLBACK'}",
        flush=True,
    )
    return {
        "airfoil": result.airfoil,
        "mae": result.mae,
        "rms": result.rms,
        "max_abs_error": result.max_abs_error,
        "p": float(result.geometry["controls"].p),
        "q": float(result.geometry["controls"].q),
        "theta": float(result.params.theta),
        "axis_ratio": float(result.params.ellipse_a / result.params.ellipse_b),
        "q_root_success": "true" if result.q_root_success else "false",
    }


def run_all(
    test_dir: str | Path,
    output_dir: str | Path,
    axis_ratio: float = 3.0,
    theta: float = 1.2,
    y0_le_ratio: float = 0.25,
    y0_te_ratio: float = 0.05,
    ellipse_fit_limit: float = 0.025,
    te_ellipse: bool = True,
    te_c2: bool = False,
    te_fit_start: float = 0.90,
    n_per_segment: int = 240,
    theta_sweep: str | None = None,
    axis_ratio_sweep: str | None = None,
    min_control_spacing: float = 1e-3,
    optimize: bool = False,
    optimizer: str = "slsqp",
    objective_mode: str = "mae",
    maxiter: int = 500,
    tail_weight: float = 0.0,
    ga_population: int = 96,
    ga_generations: int = 120,
    random_seed: int | None = 7,
    make_plot: bool = True,
    show_plot: bool = False,
) -> None:
    """Run every supported test airfoil in a directory."""
    test_path = _resolve_path(test_dir)
    files = sorted(
        path
        for path in test_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".dat", ".s1", ".s6", ".s11"}
    )
    if not files:
        raise FileNotFoundError(f"No airfoil files found in {test_path}.")
    metrics: list[dict[str, float]] = []
    failures: list[tuple[str, str]] = []
    for path in files:
        try:
            metrics.append(
                run_one(
                    path,
                    output_dir,
                    axis_ratio=axis_ratio,
                    theta=theta,
                    y0_le_ratio=y0_le_ratio,
                    y0_te_ratio=y0_te_ratio,
                    ellipse_fit_limit=ellipse_fit_limit,
                    te_ellipse=te_ellipse,
                    te_c2=te_c2,
                    te_fit_start=te_fit_start,
                    n_per_segment=n_per_segment,
                    theta_sweep=theta_sweep,
                    axis_ratio_sweep=axis_ratio_sweep,
                    min_control_spacing=min_control_spacing,
                    optimize=optimize,
                    optimizer=optimizer,
                    objective_mode=objective_mode,
                    maxiter=maxiter,
                    tail_weight=tail_weight,
                    ga_population=ga_population,
                    ga_generations=ga_generations,
                    random_seed=random_seed,
                    make_plot=make_plot,
                    show_plot=show_plot,
                )
            )
        except Exception as exc:
            failures.append((path.name, str(exc)))
            print(f"{path.stem:14s}  FAILED: {exc}", flush=True)
    if metrics:
        mae_key = "optimized_mae" if optimize else "mae"
        mae = np.array([item[mae_key] for item in metrics], dtype=float)
        save_summary(metrics, Path(output_dir) / "bp3333_virtual_mtG2_summary.csv")
        print("-" * 72)
        print(f"Finished {len(metrics)} of {len(files)} airfoils")
        print(f"Mean MAE:   {float(np.mean(mae)):.6e}")
        print(f"Median MAE: {float(np.median(mae)):.6e}")
        print(f"Worst MAE:  {float(np.max(mae)):.6e}")
    if failures:
        print(f"Failures:   {len(failures)}")
        for name, message in failures:
            print(f"  {name}: {message}")


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="BP3333 virtual-thickness C1 airfoil reconstruction.")
    parser.add_argument("--airfoil", default=str(base / "Test Airfoils" / "uvblade.s1"), help="One airfoil file.")
    parser.add_argument("--test-dir", default=str(base / "Test Airfoils"), help="Directory used with --all.")
    parser.add_argument("--output-dir", default=str(base / "results"), help="Output directory.")
    parser.add_argument("--all", action="store_true", help="Run every test airfoil.")
    parser.add_argument("--axis-ratio", type=float, default=3.0, help="Ellipse a/b ratio.")
    parser.add_argument("--theta", type=float, default=1.2, help="Ellipse tangent parameter in radians.")
    parser.add_argument("--y0-le-ratio", type=float, default=0.25, help="Initial y0_le as a fraction of leading ellipse tangent thickness.")
    parser.add_argument("--y0-te-ratio", type=float, default=0.05, help="Initial y0_te as a fraction of maximum thickness.")
    parser.add_argument(
        "--theta-sweep",
        default=None,
        help="Optional start:stop:count theta grid, for example 0.4:1.45:22.",
    )
    parser.add_argument(
        "--axis-ratio-sweep",
        default=None,
        help="Optional start:stop:count a/b grid used only with --theta-sweep.",
    )
    parser.add_argument("--ellipse-fit-limit", type=float, default=0.025, help="Leading chord fraction used to fit ellipse.")
    parser.add_argument("--te-fit-start", type=float, default=0.90, help="Chord fraction where trailing ellipse seed fit starts.")
    parser.add_argument("--no-te-ellipse", action="store_true", help="Disable the trailing-edge ellipse segment.")
    parser.add_argument("--te-c2", action="store_true", help="Solve the trailing ellipse tangent parameter from C2 continuity.")
    parser.add_argument("--n-per-segment", type=int, default=240, help="Points per Bezier segment.")
    parser.add_argument(
        "--min-control-spacing",
        type=float,
        default=1e-3,
        help="Preferred minimum x spacing between P0/P1/P2 during sweep selection.",
    )
    parser.add_argument("--optimize", action="store_true", help="Run optimization after the direct estimate.")
    parser.add_argument(
        "--optimizer",
        choices=("slsqp", "ga", "hybrid"),
        default="slsqp",
        help="Optimization backend. 'hybrid' runs GA first and SLSQP second.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("mae", "rms", "bp3434"),
        default="mae",
        help="Optimization objective. 'mae' matches the reported grid MAE; 'bp3434' reproduces the original surface L2 style.",
    )
    parser.add_argument("--maxiter", type=int, default=500, help="Maximum SLSQP iterations used with --optimize.")
    parser.add_argument("--tail-weight", type=float, default=0.0, help="Closed blunt-TE tail thickness penalty weight.")
    parser.add_argument("--ga-population", type=int, default=96, help="Population size for --optimizer ga/hybrid.")
    parser.add_argument("--ga-generations", type=int, default=120, help="Number of generations for --optimizer ga/hybrid.")
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for GA initialization.")
    parser.add_argument("--no-plot", action="store_true", help="Do not save comparison plot.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively. Single-airfoil runs show by default.")
    parser.add_argument("--no-show", action="store_true", help="Do not show plots interactively.")
    args = parser.parse_args()

    if args.all:
        run_all(
            args.test_dir,
            args.output_dir,
            axis_ratio=args.axis_ratio,
            theta=args.theta,
            y0_le_ratio=args.y0_le_ratio,
            y0_te_ratio=args.y0_te_ratio,
            ellipse_fit_limit=args.ellipse_fit_limit,
            te_ellipse=not args.no_te_ellipse,
            te_c2=args.te_c2,
            te_fit_start=args.te_fit_start,
            n_per_segment=args.n_per_segment,
            theta_sweep=args.theta_sweep,
            axis_ratio_sweep=args.axis_ratio_sweep,
            min_control_spacing=args.min_control_spacing,
            optimize=args.optimize,
            optimizer=args.optimizer,
            objective_mode=args.objective_mode,
            maxiter=args.maxiter,
            tail_weight=args.tail_weight,
            ga_population=args.ga_population,
            ga_generations=args.ga_generations,
            random_seed=args.random_seed,
            make_plot=not args.no_plot,
            show_plot=args.show and not args.no_show,
        )
    else:
        run_one(
            args.airfoil,
            args.output_dir,
            axis_ratio=args.axis_ratio,
            theta=args.theta,
            y0_le_ratio=args.y0_le_ratio,
            y0_te_ratio=args.y0_te_ratio,
            ellipse_fit_limit=args.ellipse_fit_limit,
            te_ellipse=not args.no_te_ellipse,
            te_c2=args.te_c2,
            te_fit_start=args.te_fit_start,
            n_per_segment=args.n_per_segment,
            theta_sweep=args.theta_sweep,
            axis_ratio_sweep=args.axis_ratio_sweep,
            min_control_spacing=args.min_control_spacing,
            optimize=args.optimize,
            optimizer=args.optimizer,
            objective_mode=args.objective_mode,
            maxiter=args.maxiter,
            tail_weight=args.tail_weight,
            ga_population=args.ga_population,
            ga_generations=args.ga_generations,
            random_seed=args.random_seed,
            make_plot=not args.no_plot,
            show_plot=not args.no_show,
        )


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    local = Path(__file__).resolve().parent / candidate
    if local.exists():
        return local
    return candidate


def _parse_sweep(text: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) == 3:
        start, stop, count = float(parts[0]), float(parts[1]), int(parts[2])
        return np.linspace(start, stop, count)
    return np.array([float(part) for part in text.split(",") if part.strip()], dtype=float)


if __name__ == "__main__":
    main()
