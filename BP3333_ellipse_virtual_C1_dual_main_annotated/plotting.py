"""Plotting utilities for BP3333 virtual-thickness results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .fit import DirectFitResult
from .ellipse import sample_ellipse
from .model import sample_trailing_ellipse
from .optimization import OptimizedFitResult


def plot_result(result: DirectFitResult, save_path: str | Path | None = None, show: bool = False) -> None:
    """Draw thickness/camber and airfoil comparison figures (separately)."""
    cache_dir = Path(tempfile.gettempdir()) / "bp3333_ellipse_matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib.pyplot as plt

    # 全局字体
    plt.rcParams.update({'font.size': 16})

    # ---- 厚度分布图 ----
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    _plot_thickness(ax1, result)
    # 将原 suptitle 作为厚度图的标题
    ax1.set_title(
        f"{result.airfoil} BP3333 virtual direct estimate: "
        f"MAE={result.mae:.2e}, qres={result.q_residual:.1e}, "
        f"root={'OK' if result.q_root_success else 'FALLBACK'}, dz_TE={result.params.dz_te:.3e}",
        fontsize=16,
    )
    fig1.tight_layout(pad=0.5)

    if save_path is not None:
        p = Path(save_path)
        p_th = p.with_stem(p.stem + "_thickness")
        p_th.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(p_th, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # ---- 翼型轮廓图 ----
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    _plot_airfoil(ax2, result)
    fig2.tight_layout(pad=0.5)

    if save_path is not None:
        p = Path(save_path)
        p_af = p.with_stem(p.stem + "_airfoil")
        p_af.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(p_af, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig2)


def plot_optimization_result(
    result: OptimizedFitResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Draw reference, direct, and optimised airfoils (separately)."""
    cache_dir = Path(tempfile.gettempdir()) / "bp3333_ellipse_matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 16})

    # ---- 优化厚度分布图 ----
    fig1, ax1 = plt.subplots(figsize=(15, 5))
    _plot_optimized_thickness(ax1, result)
    ax1.set_title(
        f"{result.optimized.airfoil} BP3333 virtual optimisation: "
        f"MAE {result.initial.mae:.2e} -> {result.optimized.mae:.2e}, "
        f"{result.method} status={result.status}",
        fontsize=16,
    )
    fig1.tight_layout(pad=0.5)

    if save_path is not None:
        p = Path(save_path)
        p_th = p.with_stem(p.stem + "_thickness")
        p_th.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(p_th, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # ---- 优化翼型轮廓图 ----
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    _plot_optimized_airfoil(ax2, result)
    fig2.tight_layout(pad=0.5)

    if save_path is not None:
        p = Path(save_path)
        p_af = p.with_stem(p.stem + "_airfoil")
        p_af.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(p_af, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig2)


# ========== 内部绘图函数（字体大小已统一为16）==========

def _plot_thickness(ax, result: DirectFitResult) -> None:
    ref = result.reference
    g = result.geometry
    controls = g["controls"]
    ex, ey = sample_ellipse(controls.ellipse, 120)
    tx, ty = _tail_ellipse_xy(controls)
    ax.plot(ref.x_eval, ref.thickness_y, color="black", linewidth=1.2, label="Reference thickness")
    ax.plot(g["thickness_x"], g["thickness_y"], color="tab:blue", linestyle="--", linewidth=1.4, label="BP3333 virtual")
    ax.plot(ex, ey, color="tab:cyan", linewidth=2.0, label="Ellipse head")
    if tx is not None:
        ax.plot(tx, ty, color="tab:purple", linewidth=2.0, label="TE ellipse")
    ax.plot(ref.x_eval, ref.camber_y, color="tab:orange", linestyle="-.", linewidth=1.0, label="Reference camber")
    ax.plot(g["camber_x"], g["camber_y"], color="tab:red", linestyle=":", linewidth=1.2, label="BP3333 camber")
    _plot_controls(ax, controls.x_le, controls.y_le, "P", skip_indices=set())
    _plot_controls(ax, controls.x_te, controls.y_te, "Q", skip_indices={0})
    ax.annotate(
        "P3/Q0",
        (controls.x_le[-1], controls.y_le[-1]),
        textcoords="offset points",
        xytext=(10, -22),          # 随字号增大调整
        fontsize=16,
        color="tab:blue",
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    # 不再设置内部标题（由外层控制）
    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.legend(loc="best")                     # 使用全局字号


def _plot_controls(ax, x: np.ndarray, y: np.ndarray, prefix: str, skip_indices: set[int]) -> None:
    ax.plot(x, y, "o", color="tab:blue", markersize=6, label="Thickness controls" if prefix == "P" else None)
    # 加大偏移以适应 16 pt 字体
    offsets = [(10, 8), (10, 20), (10, 20), (10, -22)]
    for idx, (xx, yy) in enumerate(zip(x, y)):
        if idx in skip_indices:
            continue
        dx, dy = offsets[min(idx, len(offsets) - 1)]
        ax.annotate(
            f"{prefix}{idx}",
            (xx, yy),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=16,
            color="tab:blue",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
        )


def _plot_airfoil(ax, result: DirectFitResult) -> None:
    ref = result.reference
    g = result.geometry
    ax.plot(ref.contour[:, 0], ref.contour[:, 1],  linestyle="-.",color="black", linewidth=1.1, label="Reference")
    ax.plot(g["upper_x"], g["upper_y"], color="tab:green", linewidth=1.3, label="BP3333 virtual")
    ax.plot(g["lower_x"], g["lower_y"], color="tab:green", linewidth=1.3)
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title("Airfoil contour comparison")
    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.legend(loc="best")


def _plot_optimized_thickness(ax, result: OptimizedFitResult) -> None:
    ref = result.initial.reference
    initial = result.initial.geometry
    optimized = result.optimized.geometry
    controls = optimized["controls"]
    ex, ey = sample_ellipse(controls.ellipse, 120)
    tx, ty = _tail_ellipse_xy(controls)
    ax.plot(ref.x_eval, ref.thickness_y, color="black", linewidth=1.2, label="Reference thickness")
    ax.plot(initial["thickness_x"], initial["thickness_y"], color="tab:green", linestyle="--", linewidth=1.1, label="Direct thickness")
    ax.plot(optimized["thickness_x"], optimized["thickness_y"], color="tab:blue", linewidth=1.5, label="Optimized thickness")
    ax.plot(ex, ey, color="tab:cyan", linewidth=1.8, label="Optimized ellipse head")
    if tx is not None:
        ax.plot(tx, ty, color="tab:purple", linewidth=1.8, label="Optimized TE ellipse")
    ax.plot(ref.x_eval, ref.camber_y, color="tab:orange", linestyle="-.", linewidth=1.0, label="Reference camber")
    ax.plot(initial["camber_x"], initial["camber_y"], color="tab:olive", linestyle="--", linewidth=1.0, label="Direct camber")
    ax.plot(optimized["camber_x"], optimized["camber_y"], color="tab:red", linestyle=":", linewidth=1.4, label="Optimized camber")
    _plot_controls(ax, controls.x_le, controls.y_le, "P", skip_indices=set())
    _plot_controls(ax, controls.x_te, controls.y_te, "Q", skip_indices={0})
    ax.annotate(
        "P3/Q0",
        (controls.x_le[-1], controls.y_le[-1]),
        textcoords="offset points",
        xytext=(10, -22),
        fontsize=16,
        color="tab:blue",
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    # 外层会覆盖标题
    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.legend(loc="upper left", frameon=False,bbox_to_anchor=(1, 1))


def _plot_optimized_airfoil(ax, result: OptimizedFitResult) -> None:
    ref = result.initial.reference
    initial = result.initial.geometry
    optimized = result.optimized.geometry
    ax.plot(ref.contour[:, 0], ref.contour[:, 1], color="black", linestyle="-.", linewidth=1.1, label="Reference")
    ax.plot(initial["upper_x"], initial["upper_y"], color="tab:green", linestyle="--", linewidth=1.1, label="Direct")
    ax.plot(initial["lower_x"], initial["lower_y"], color="tab:green", linestyle="--", linewidth=1.1)
    ax.plot(optimized["upper_x"], optimized["upper_y"], color="tab:blue", linewidth=1.3, label="Optimized")
    ax.plot(optimized["lower_x"], optimized["lower_y"], color="tab:blue", linewidth=1.3)
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title("Airfoil contour comparison")
    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.legend(frameon=False,loc="best")


def _tail_ellipse_xy(controls) -> tuple[np.ndarray | None, np.ndarray | None]:
    tail = getattr(controls, "tail_ellipse", None)
    if tail is None:
        return None, None
    return sample_trailing_ellipse(tail, 120)