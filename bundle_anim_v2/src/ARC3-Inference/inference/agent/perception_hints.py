"""Per-turn perception hints injected into the analyzer user prompt.

Computed host-side from the current frame at zero environment cost. Every real
action permanently lowers the level efficiency score, so telling the model
where the interesting pixels are (instead of letting it probe blindly) directly
protects score.
"""
from __future__ import annotations

from typing import Any

from inference.utils.grid_utils import ARC_COLOR_CHARS


def _components(grid: list[list[int]]) -> list[dict[str, Any]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen = [[False] * cols for _ in range(rows)]
    comps: list[dict[str, Any]] = []
    for r0 in range(rows):
        for c0 in range(cols):
            if seen[r0][c0]:
                continue
            color = grid[r0][c0]
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not seen[nr][nc]
                        and grid[nr][nc] == color
                    ):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _medoid(cells: list[tuple[int, int]]) -> tuple[int, int]:
    """A constituent cell nearest the centroid (always ON the object)."""
    size = len(cells)
    cr = sum(r for r, _ in cells) / size
    cc = sum(c for _, c in cells) / size
    return min(cells, key=lambda cell: (cell[0] - cr) ** 2 + (cell[1] - cc) ** 2)


def build_perception_hint_lines(
    current_frame: Any,
    valid_actions: list[str],
    last_action_result: dict[str, Any] | None,  # kept for future budget hints; unused
) -> list[str]:
    lines: list[str] = []

    if current_frame is None or "MOUSE" not in valid_actions:
        return lines
    grid = getattr(current_frame, "grid", None)
    if not grid:
        return lines
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not rows or not cols:
        return lines

    comps = _components(grid)
    if not comps:
        return lines
    counts: dict[int, int] = {}
    for comp in comps:
        counts[comp["color"]] = counts.get(comp["color"], 0) + len(comp["cells"])
    background = max(counts, key=counts.get)

    hud = 0
    candidates: list[tuple[int, int, str, int]] = []
    for comp in comps:
        cells = comp["cells"]
        size = len(cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r_min, r_max = min(rs), max(rs)
        c_min, c_max = min(cs), max(cs)
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        touches_border = r_min == 0 or c_min == 0 or r_max == rows - 1 or c_max == cols - 1
        if touches_border and (height <= 2 or width <= 2) and max(height, width) >= 12:
            hud += 1
            continue
        if comp["color"] == background and size > 400:
            continue
        if size > 800:
            continue
        r, c = _medoid(cells)
        char = ARC_COLOR_CHARS[max(0, min(15, int(comp["color"])))]
        candidates.append((r, c, char, size))

    if not candidates and not hud:
        return lines
    candidates.sort(key=lambda item: -item[3])
    shown = candidates[:8]
    parts = [f"({r},{c} {ch} {px}px)" for r, c, ch, px in shown]
    more = f" +{len(candidates) - len(shown)} more" if len(candidates) > len(shown) else ""
    line = (
        f"Hint (computed for free, not a probe): {len(candidates)} distinct board objects; "
        f"a pixel guaranteed inside each [row,col color size]: " + "; ".join(parts) + more + "."
    )
    if hud:
        line += f" {hud} thin edge strip(s) look like HUD/timer, not clickable pieces."
    lines.append(line)
    return lines
