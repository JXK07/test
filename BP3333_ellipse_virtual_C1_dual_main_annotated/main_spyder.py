"""Spyder-friendly entry point for BP3333 virtual-thickness C1 reconstruction.

All run parameters are plain Python variables in the ``USER SETTINGS`` block,
so this file can be opened in Spyder and executed directly with Run File.  No
command-line parsing is used here.  Set ``RUN_ALL`` to ``True`` for batch
processing, or keep it ``False`` to debug a single airfoil.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from BP3333_ellipse_virtual_C1_dual_main_annotated.main import run_all, run_one
else:
    from .main import run_all, run_one


# =========================
# USER SETTINGS FOR SPYDER
# =========================
BASE_DIR = Path(__file__).resolve().parent
RUN_ALL = False

AIRFOIL = BASE_DIR / "Test Airfoils" / "uvblade.s1"
TEST_DIR = BASE_DIR / "Test Airfoils"
OUTPUT_DIR = BASE_DIR / "results_spyder"

AXIS_RATIO = 3.0
THETA = 1.2
Y0_LE_RATIO = 0.25
Y0_TE_RATIO = 0.05
ELLIPSE_FIT_LIMIT = 0.025
TE_ELLIPSE = True
TE_C2 = False
TE_FIT_START = 0.90
N_PER_SEGMENT = 240

THETA_SWEEP = None
AXIS_RATIO_SWEEP = None
MIN_CONTROL_SPACING = 1e-3

OPTIMIZE = True
OPTIMIZER = "slsqp"
OBJECTIVE_MODE = "mae"
MAXITER = 500
TAIL_WEIGHT = 0.0
GA_POPULATION = 96
GA_GENERATIONS = 120
RANDOM_SEED = 7

MAKE_PLOT = True
SHOW_PLOT = True


def main() -> None:
    """Run the configured Spyder case without reading command-line arguments."""

    common = dict(
        axis_ratio=AXIS_RATIO,
        theta=THETA,
        y0_le_ratio=Y0_LE_RATIO,
        y0_te_ratio=Y0_TE_RATIO,
        ellipse_fit_limit=ELLIPSE_FIT_LIMIT,
        te_ellipse=TE_ELLIPSE,
        te_c2=TE_C2,
        te_fit_start=TE_FIT_START,
        n_per_segment=N_PER_SEGMENT,
        theta_sweep=THETA_SWEEP,
        axis_ratio_sweep=AXIS_RATIO_SWEEP,
        min_control_spacing=MIN_CONTROL_SPACING,
        optimize=OPTIMIZE,
        optimizer=OPTIMIZER,
        objective_mode=OBJECTIVE_MODE,
        maxiter=MAXITER,
        tail_weight=TAIL_WEIGHT,
        ga_population=GA_POPULATION,
        ga_generations=GA_GENERATIONS,
        random_seed=RANDOM_SEED,
        make_plot=MAKE_PLOT,
        show_plot=SHOW_PLOT,
    )
    if RUN_ALL:
        run_all(TEST_DIR, OUTPUT_DIR, **common)
    else:
        run_one(AIRFOIL, OUTPUT_DIR, **common)


if __name__ == "__main__":
    main()
