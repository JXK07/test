"""Command-line entry point for BP3333 virtual-thickness C1 reconstruction.

This file is deliberately not Spyder-friendly: all user choices are expressed
as command-line options and parsed by :mod:`argparse`.  It is the right entry
point for terminal runs, shell scripts, and batch reconstruction of the
``Test Airfoils`` directory.

Example
-------
Run one airfoil with optimisation:

```
python -m BP3333_ellipse_virtual_C1_dual_main_annotated.main_cli \
    --airfoil "BP3333_ellipse_virtual_C1_dual_main_annotated/Test Airfoils/uvblade.s1" \
    --optimize --optimizer slsqp
```

Run all bundled test cases:

```
python -m BP3333_ellipse_virtual_C1_dual_main_annotated.main_cli --all --optimize
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from BP3333_ellipse_virtual_C1_dual_main_annotated.main import run_all, run_one
else:
    from .main import run_all, run_one


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without executing any reconstruction work."""

    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="BP3333 virtual-thickness C1 airfoil reconstruction.",
    )
    parser.add_argument("--airfoil", default=str(base / "Test Airfoils" / "uvblade.s1"), help="Single airfoil file.")
    parser.add_argument("--test-dir", default=str(base / "Test Airfoils"), help="Directory used together with --all.")
    parser.add_argument("--output-dir", default=str(base / "results_cli"), help="Output directory.")
    parser.add_argument("--all", action="store_true", help="Run every supported test airfoil.")
    parser.add_argument("--axis-ratio", type=float, default=3.0, help="Initial leading ellipse a/b ratio.")
    parser.add_argument("--theta", type=float, default=1.2, help="Initial leading ellipse tangent parameter in radians.")
    parser.add_argument("--y0-le-ratio", type=float, default=0.25, help="Virtual leading thickness y0 as a fraction of leading tangent thickness.")
    parser.add_argument("--y0-te-ratio", type=float, default=0.05, help="Virtual trailing thickness y0 as a fraction of maximum thickness.")
    parser.add_argument("--theta-sweep", default=None, help="Optional start:stop:count theta grid, for example 0.4:1.45:22.")
    parser.add_argument("--axis-ratio-sweep", default=None, help="Optional start:stop:count a/b grid used only with --theta-sweep.")
    parser.add_argument("--ellipse-fit-limit", type=float, default=0.025, help="Leading chord fraction used to fit the ellipse seed.")
    parser.add_argument("--te-fit-start", type=float, default=0.90, help="Chord fraction where trailing ellipse seed fitting starts.")
    parser.add_argument("--no-te-ellipse", action="store_true", help="Disable the trailing-edge ellipse segment.")
    parser.add_argument("--te-c2", action="store_true", help="Solve trailing ellipse tangent parameter from C2 continuity.")
    parser.add_argument("--n-per-segment", type=int, default=240, help="Sampling points per Bezier segment.")
    parser.add_argument("--min-control-spacing", type=float, default=1e-3, help="Minimum preferred x spacing between control points.")
    parser.add_argument("--optimize", action="store_true", help="Run optimisation after direct parameter extraction.")
    parser.add_argument("--optimizer", choices=("slsqp", "ga", "hybrid"), default="slsqp", help="Optimisation backend.")
    parser.add_argument("--objective-mode", choices=("mae", "rms", "bp3434"), default="mae", help="Optimisation objective.")
    parser.add_argument("--maxiter", type=int, default=500, help="Maximum SLSQP iterations.")
    parser.add_argument("--tail-weight", type=float, default=0.0, help="Closed blunt trailing-edge thickness penalty weight.")
    parser.add_argument("--ga-population", type=int, default=96, help="Population size for GA/hybrid optimisation.")
    parser.add_argument("--ga-generations", type=int, default=120, help="Number of GA generations.")
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for GA initialisation.")
    parser.add_argument("--no-plot", action="store_true", help="Do not save comparison plots.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and dispatch to ``run_one`` or ``run_all``."""

    args = build_parser().parse_args(argv)
    common = dict(
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
        show_plot=args.show,
    )
    if args.all:
        run_all(args.test_dir, args.output_dir, **common)
    else:
        run_one(args.airfoil, args.output_dir, **common)


if __name__ == "__main__":
    main()
