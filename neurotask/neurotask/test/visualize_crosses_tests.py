from __future__ import annotations

"""
Script para visualizar los trails de los tests y marcar los cruces detectados.

Este script extrae los trails de cada test en test_crosses.py, calcula los cruces
usando calculate_crosses_for_trial, y genera visualizaciones mostrando los trails
con los cruces marcados.
"""

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

from neurotask.tmt.crosses.crosses import calculate_crosses_for_trial
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject
from neurotask.tmt.model.tmt_model import Coordinate, TMTTarget

Segment = Tuple[Coordinate, Coordinate]
Cross = Tuple[Segment, Segment, float]  # (seg1, seg2, time_gap_ms)


@dataclass(frozen=True)
class CrossStyle:
    index: int
    color: str


def _trail_xy(cursor_trail: Sequence) -> Tuple[List[float], List[float]]:
    xs = [ci.position.x for ci in cursor_trail]
    ys = [ci.position.y for ci in cursor_trail]
    return xs, ys


def _segment_midpoint(seg: Segment) -> Tuple[float, float]:
    p1, p2 = seg
    return (p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0


def _jitter_for_index(k: int, scale: float = 0.18) -> Tuple[float, float]:
    pattern = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
    ]
    dx, dy = pattern[k % len(pattern)]
    return dx * scale, dy * scale


def plot_crosses_simple(
    cursor_trail: Sequence,
    crosses: Sequence[Cross],
    title: str = "",
    expected_crosses: int | None = None,
    colors: Sequence[str] | None = None,
    linewidth_trail: float = 2.0,
    linewidth_cross: float = 4.0,
    # --- Added from the "old" plotting function ---
    canvas_size: int | None = None,
    invert_y: bool = False,
    targets: Sequence[TMTTarget] | None = None,
    target_radius: float | None = None,
    show_cross_endpoints: bool = True,
    cross_endpoints_size: float = 70.0,
    cross_endpoints_edgecolor: str = "k",
):
    """
    Plot:
      1) The full trail (as background).
      2) Highlight segments participating in each detected cross (using `crosses`).
      3) Visually separate overlapped crosses by jittering labels (not geometry).
      4) Side panel listing cross index -> color (+ time_gap).

    Additions (ported from the old plotter):
      - Optional canvas sizing and inverted Y (typical of js canvas coords).
      - Optional targets drawing as circles + content text.
      - Optional marking of cross segment endpoints (like the old yellow dots).

    expected_crosses:
      If provided, it is shown in the title as "(expected=N)".
    """
    if colors is None:
        colors = (
            "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown",
            "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "gold",
        )

    fig, (ax, ax_side) = plt.subplots(
        1, 2, figsize=(14, 8),
        gridspec_kw={"width_ratios": [3.2, 1.4]},
    )

    # --- 1) Full trail ---
    xs, ys = _trail_xy(cursor_trail)
    ax.plot(xs, ys, alpha=0.35, linewidth=linewidth_trail, label="Trail")
    ax.scatter(xs, ys, s=20, alpha=0.35)

    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6) if xs and ys else 1.0

    # --- 2) Highlight crossing segments + 3) jitter labels ---
    styles: List[CrossStyle] = []

    # Collect endpoints (old behavior): mark the 4 endpoints per cross
    endpoints_x: List[float] = []
    endpoints_y: List[float] = []
    endpoints_colors: List[str] = []

    for i, (seg1, seg2, time_gap_ms) in enumerate(crosses, start=1):
        color = colors[(i - 1) % len(colors)]
        styles.append(CrossStyle(index=i, color=color))

        # highlight segments
        for seg in (seg1, seg2):
            p1, p2 = seg
            ax.plot(
                [p1.x, p2.x], [p1.y, p2.y],
                linewidth=linewidth_cross,
                linestyle="--",
                alpha=0.9,
                color=color,
            )

        # optional: mark endpoints (ported from old plotter)
        if show_cross_endpoints:
            for p in (seg1[0], seg1[1], seg2[0], seg2[1]):
                endpoints_x.append(p.x)
                endpoints_y.append(p.y)

        # label near midpoint of seg1 (jittered)
        mx, my = _segment_midpoint(seg1)
        jx, jy = _jitter_for_index(i, scale=0.08)
        ax.annotate(
            f"#{i}",
            (mx + jx * span, my + jy * span),
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
        )

    if show_cross_endpoints and endpoints_x:
        ax.scatter(
            endpoints_x,
            endpoints_y,
            c="yellow",              # <- fijo como antes
            s=140,                   # <- más grande
            zorder=20,               # <- bien arriba
            linewidths=2.0,          # <- borde marcado
            edgecolor="black",
            alpha=1.0,               # <- sin transparencia
            marker="o",
            label="Cross endpoints", # opcional
        )
    # --- Optional: targets (ported from old plotter) ---
    if targets is not None and target_radius is not None:
        for tgt in targets:
            x, y = tgt.position.x, tgt.position.y
            circle = plt.Circle((x, y), target_radius, color="grey", alpha=0.35, zorder=3)
            ax.add_patch(circle)
            ax.text(
                x, y, str(getattr(tgt, "content", "")),
                color="black", fontsize=10, ha="center", va="center", zorder=4
            )

    # --- Title (includes expected) ---
    detected = len(crosses)
    suffix = f"(detected={detected})" if expected_crosses is None else f"(detected={detected}, expected={expected_crosses})"
    ax.set_title(title.strip() + (" — " if title.strip() else "") + suffix, fontweight="bold")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")

    # --- Optional: canvas coordinate behavior (ported from old plotter) ---
    if canvas_size is not None:
        ax.set_xlim(0, canvas_size)
        if invert_y:
            ax.set_ylim(canvas_size, 0)
        else:
            ax.set_ylim(0, canvas_size)
    elif invert_y:
        # If caller wants inverted Y but no explicit canvas_size, invert current limits.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax, ymin)

    # --- 4) Side panel: list cross index -> color (+ gap) ---
    ax_side.axis("off")
    lines: List[str] = []
    lines.append("Cruces (índice → color)")
    lines.append("----------------------")
    if not crosses:
        lines.append("No se detectaron cruces.")
    else:
        for style, (_, _, gap) in zip(styles, crosses):
            lines.append(f"#{style.index:02d}  {style.color:<9}  gap={gap:.0f} ms")

    ax_side.text(
        0.0, 1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    plt.tight_layout()
    return fig, (ax, ax_side)


def main(output_dir: str = "crosses_visualizations") -> None:
    """
    Generate debug plots for the same trails used in unit tests.

    For each example:
      - builds the trial
      - runs calculate_crosses_for_trial(trial, crosses_time_threshold=500)
      - plots with plot_crosses_simple(cursor_trail, crosses, expected_crosses=...)
      - saves PNG to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    examples = [
        {
            "name": "test_simple_cross_detects_one_cross",
            "expected": 1,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (10.0, 10.0, 100.0),
                (5.0, 5.0, 300.0),
                (0.0, 10.0, 600.0),
                (10.0, 0.0, 700.0),
            ]),
        },
        {
            "name": "test_no_cross_returns_zero",
            "expected": 0,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 100.0),
                (20.0, 0.0, 200.0),
                (30.0, 0.0, 300.0),
            ]),
        },
        {
            "name": "test_multiple_crosses_detects_all",
            "expected": 4,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (10.0, 10.0, 100.0),
                (0.0, 10.0, 600.0),
                (10.0, 0.0, 700.0),
                (5.0, 0.0, 1200.0),
                (5.0, 10.0, 1300.0),
            ]),
        },
        {
            "name": "test_segments_too_close_in_time_are_excluded",
            "expected": 0,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (10.0, 10.0, 100.0),
                (5.0, 5.0, 120.0),
                (0.0, 10.0, 150.0),
                (10.0, 0.0, 200.0),
            ]),
        },
        {
            "name": "test_segments_far_apart_in_time_are_included",
            "expected": 1,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (10.0, 10.0, 100.0),
                (5.0, 5.0, 300.0),
                (0.0, 10.0, 600.0),
                (10.0, 0.0, 700.0),
            ]),
        },
        {
            "name": "test_overlapping_segments_are_excluded",
            "expected": 0,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (10.0, 10.0, 200.0),
                (5.0, 5.0, 150.0),
                (0.0, 10.0, 100.0),
                (10.0, 0.0, 300.0),
            ]),
        },
        {
            "name": "test_segments_that_touch_at_endpoint",
            "expected": 1,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (5.0, 5.0, 100.0),
                (10.0, 0.0, 200.0),
                (5.0, 5.0, 600.0),
                (10.0, 10.0, 700.0),
            ]),
        },
        {
            "name": "test_colinear_segments_do_not_cross",
            "expected": 0,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (5.0, 0.0, 100.0),
                (7.5, 0.0, 300.0),
                (10.0, 0.0, 600.0),
                (15.0, 0.0, 700.0),
            ]),
        },
        # These two are "expected 0" by definition of not enough points/non-adjacent segments
        {
            "name": "test_trail_too_short_returns_zero__3_points",
            "expected": 0,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 100.0),
                (20.0, 0.0, 200.0),
            ]),
        },
        {
            "name": "test_adjacent_segments_do_not_count_as_cross",
            "expected": 0,
            "cursor_trail": build_cursor_trail([
                (0.0, 0.0, 0.0),
                (5.0, 5.0, 100.0),
                (10.0, 0.0, 200.0),
            ]),
        },
    ]

    for ex in examples:
        name = ex["name"]
        expected = ex["expected"]
        cursor_trail = ex["cursor_trail"]

        trial, subject = build_trial_and_subject(cursor_trail)

        detected, cross_segments = calculate_crosses_for_trial(trial, crosses_time_threshold=500)

        fig, _ = plot_crosses_simple(
            cursor_trail=cursor_trail,
            crosses=cross_segments,
            title=name,
            expected_crosses=expected,
        )

        # Save
        out_path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        status = "OK" if detected == expected else "MISMATCH"
        print(f"[{status}] {name}: detected={detected}, expected={expected} -> {out_path}")

    print(f"\nAll plots saved under: {output_dir}/")


if __name__ == "__main__":
    main()
