"""Constraint-Preserving Structured Generation (CPSG) helpers.

Candidate method: model predicts per-column pillar heights on Observation-occupied
cells; program converts heights -> Structure3D. Representation equivalence with the
official hypothesis space was audited in scripts/audit_cpsg_height_equivalence.py.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from run_3d_benchmark import Structure3D  # caller must put official 3d/ on sys.path


def heights_to_structure(heights: np.ndarray, max_height: int) -> Structure3D:
    rows, cols = heights.shape
    H = int(heights.max()) if heights.size else 0
    if H <= 0:
        return Structure3D([np.zeros((rows, cols), dtype=int).tolist()])
    if H > max_height:
        raise ValueError(f"max pillar height {H} > max_height {max_height}")
    layers: List[List[List[int]]] = []
    for z in range(H):
        layer = [[1 if int(heights[r, c]) > z else 0 for c in range(cols)] for r in range(rows)]
        layers.append(layer)
    return Structure3D(layers).normalize()


def structure_to_heights(structure: Structure3D) -> np.ndarray:
    rows, cols = structure.shape
    heights = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            h = 0
            for z in range(len(structure.layers)):
                if int(structure.layers[z][r, c]) == 1:
                    h = z + 1
                else:
                    break
            heights[r, c] = h
    return heights


def _format_height_map(heights: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in heights)


def build_cpsg_prompt(
    observation: np.ndarray,
    max_height: int,
    max_blocks: int,
    avoid_height_maps: Optional[List[np.ndarray]] = None,
) -> str:
    grid = observation.shape[0]
    top_lines = "\n".join(" ".join(str(int(c)) for c in row) for row in observation)
    occupied = [
        f"(r={r}, c={c})"
        for r in range(grid)
        for c in range(grid)
        if int(observation[r, c]) == 1
    ]
    occupied_str = ", ".join(occupied) if occupied else "(none)"
    avoid_block = ""
    if avoid_height_maps:
        parts = []
        for i, h in enumerate(avoid_height_maps[-5:], 1):
            parts.append(f"Prior height map {i} (do NOT repeat):\n{_format_height_map(np.asarray(h))}")
        avoid_block = "\n\n" + "\n\n".join(parts) + "\n\nGenerate a DIFFERENT valid height map from the prior ones above.\n"
    return f"""You reconstruct a 3D voxel structure on a {grid}x{grid} grid.

Official constraints:
- Maximum pillar height: {max_height}
- Top-view occupied cells (Observation=1) count is at most {max_blocks} (already fixed by the Observation below)
- Gravity is automatic if you only choose pillar heights: blocks stack continuously from the ground up
- Coordinates: row 0 is the TOP row of the printed grid; col 0 is the LEFT column
- Layer conversion (done by the program, not by you): a cell with height h has blocks in layers 1..h (Layer 1 = BOTTOM)

CRITICAL Observation (Top view). A 1 means that column MUST have height in 1..{max_height}.
A 0 means that column MUST have height 0 (empty):

Top view:
{top_lines}

Occupied positions that need heights: {occupied_str}
{avoid_block}
Task: output ONLY a {grid}x{grid} height map.
Rules:
1. If Top view cell is 0, height MUST be 0.
2. If Top view cell is 1, height MUST be an integer in 1..{max_height}.
3. Do not output Layer grids. Do not invent extra occupied columns.
4. Output format exactly:

Heights:
<row0: {grid} integers separated by spaces>
<row1: {grid} integers separated by spaces>
...
<row{grid-1}: ...>

Example shape only (numbers are illustrative, not related to this Observation):
Heights:
2 0 1
0 0 0
1 0 0
"""


def build_cpsg_range_retry_prompt(
    observation: np.ndarray,
    max_height: int,
    max_blocks: int,
    bad_heights: Optional[np.ndarray],
    problems: List[str],
    avoid_height_maps: Optional[List[np.ndarray]] = None,
) -> str:
    """One constrained retry after SupportProj: fix range (and remaining) errors.

    Does NOT invent heights programmatically. Model must re-output a full map.
    """
    base = build_cpsg_prompt(observation, max_height, max_blocks, avoid_height_maps=avoid_height_maps)
    bad = _format_height_map(bad_heights) if bad_heights is not None else "(unparseable)"
    probs = "\n".join(f"- {p}" for p in (problems or [])[:20]) or "- height range / format error"
    return (
        base
        + f"""

Your previous height map was INVALID after Observation support projection.
Previous heights:
{bad}

Problems to fix (program will again force empty Observation cells to 0; YOU must fix occupied-cell heights):
{probs}

Hard rules for the retry:
1. Occupied cells (Observation=1): height MUST be an integer in 1..{max_height}. Never use 0 or >{max_height}.
2. Empty cells (Observation=0): height MUST be 0.
3. Do not silently guess random patterns — correct the listed cells.
4. Output ONLY the corrected Heights block in the same format.
"""
    )


def parse_heights_response(
    response: str,
    observation: np.ndarray,
    max_height: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Parse model height map; enforce Observation support and height range."""
    info: Dict[str, Any] = {
        "parse_success": False,
        "format_ok": False,
        "support_ok": False,
        "range_ok": False,
        "error_class": "Parsing failure",
        "heights": None,
    }
    if not isinstance(response, str) or not response.strip():
        return None, info

    grid = observation.shape[0]
    lines = response.strip().splitlines()
    start = -1
    for i, line in enumerate(lines):
        if re.search(r"^\s*Heights\s*:", line, flags=re.IGNORECASE):
            start = i
            break
        if re.search(r"^\s*Height\s+map\s*:", line, flags=re.IGNORECASE):
            start = i
            break

    # Fallback: first block of grid-sized integer rows
    rows: List[List[int]] = []
    scan_from = start + 1 if start >= 0 else 0
    for line in lines[scan_from:]:
        s = line.strip()
        if not s:
            if rows:
                break
            continue
        if s.lower().startswith(("note", "because", "therefore", "explanation", "reasoning")):
            break
        # allow commas
        s = s.replace(",", " ").replace("|", " ")
        s = s.strip("[]()")
        parts = s.split()
        if not parts:
            continue
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            if rows:
                break
            continue
        if len(vals) != grid:
            if rows:
                break
            continue
        rows.append(vals)
        if len(rows) == grid:
            break

    if len(rows) != grid:
        info["error_class"] = "Parsing failure"
        return None, info

    heights = np.array(rows, dtype=int)
    info["format_ok"] = True
    info["parse_success"] = True
    info["heights"] = heights.tolist()

    support_ok = True
    range_ok = True
    problems: List[str] = []
    for r in range(grid):
        for c in range(grid):
            o = int(observation[r, c])
            h = int(heights[r, c])
            if o == 0 and h != 0:
                support_ok = False
                problems.append(f"(r={r},c={c}) obs=0 but height={h}")
            elif o == 1 and not (1 <= h <= max_height):
                range_ok = False
                problems.append(f"(r={r},c={c}) obs=1 but height={h} not in 1..{max_height}")
            elif o == 0 and h == 0:
                pass
            elif h < 0:
                range_ok = False
                problems.append(f"(r={r},c={c}) negative height={h}")

    info["support_ok"] = support_ok
    info["range_ok"] = range_ok
    info["problems"] = problems[:30]

    if not support_ok and not range_ok:
        info["error_class"] = "Support_and_range_violation"
    elif not support_ok:
        info["error_class"] = "Observation_support_violation"
    elif not range_ok:
        info["error_class"] = "Height_range_violation"
    else:
        info["error_class"] = "OK"
    return heights, info


def apply_support_projection(
    heights: np.ndarray,
    observation: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Zero out heights on Observation=0 cells. Does not touch occupied cells.

    This is a program projection onto Observation support. It does not shrink the
    official hypothesis space: occupied-column heights are unchanged.
    """
    out = heights.copy()
    n_cleared = 0
    cleared_cells: List[str] = []
    for r in range(observation.shape[0]):
        for c in range(observation.shape[1]):
            if int(observation[r, c]) == 0 and int(out[r, c]) != 0:
                cleared_cells.append(f"(r={r},c={c}):{int(out[r, c])}->0")
                out[r, c] = 0
                n_cleared += 1
    return out, {"n_cleared": n_cleared, "cleared_cells": cleared_cells[:30]}


def _finalize_heights(
    heights: np.ndarray,
    observation: np.ndarray,
    max_height: int,
    info: Dict[str, Any],
) -> Tuple[Optional[Structure3D], Dict[str, Any]]:
    """Re-check support/range on (possibly projected) heights, then convert."""
    grid = observation.shape[0]
    support_ok = True
    range_ok = True
    problems: List[str] = []
    for r in range(grid):
        for c in range(grid):
            o = int(observation[r, c])
            h = int(heights[r, c])
            if o == 0 and h != 0:
                support_ok = False
                problems.append(f"(r={r},c={c}) obs=0 but height={h}")
            elif o == 1 and not (1 <= h <= max_height):
                range_ok = False
                problems.append(f"(r={r},c={c}) obs=1 but height={h} not in 1..{max_height}")
            elif h < 0:
                range_ok = False
                problems.append(f"(r={r},c={c}) negative height={h}")
    info["support_ok"] = support_ok
    info["range_ok"] = range_ok
    info["problems"] = problems[:30]
    info["heights"] = heights.tolist()
    if not support_ok and not range_ok:
        info["error_class"] = "Support_and_range_violation"
    elif not support_ok:
        info["error_class"] = "Observation_support_violation"
    elif not range_ok:
        info["error_class"] = "Height_range_violation"
    else:
        info["error_class"] = "OK"

    if info["error_class"] != "OK":
        info["structure_hash"] = None
        info["overall_valid_candidate"] = False
        return None, info
    try:
        struct = heights_to_structure(heights, max_height)
    except Exception as e:  # noqa: BLE001
        info["error_class"] = f"Conversion_error:{type(e).__name__}"
        info["structure_hash"] = None
        info["overall_valid_candidate"] = False
        return None, info
    info["structure_hash"] = struct.get_hash()
    info["overall_valid_candidate"] = True
    info["structure"] = struct.to_string()
    return struct, info


def cpsg_decode(
    response: str,
    observation: np.ndarray,
    max_height: int,
    support_projection: bool = False,
) -> Tuple[Optional[Structure3D], Dict[str, Any]]:
    """Parse heights, optionally project empty columns to 0, convert if OK."""
    heights, info = parse_heights_response(response, observation, max_height)
    info["support_projection"] = bool(support_projection)
    info["projection_applied"] = False
    info["n_cleared"] = 0
    if heights is None:
        info["structure_hash"] = None
        info["overall_valid_candidate"] = False
        return None, info

    raw_error = info.get("error_class")
    info["error_class_before_projection"] = raw_error

    if support_projection:
        heights, proj = apply_support_projection(heights, observation)
        info["projection_applied"] = proj["n_cleared"] > 0
        info["n_cleared"] = proj["n_cleared"]
        info["cleared_cells"] = proj["cleared_cells"]
        return _finalize_heights(heights, observation, max_height, info)

    # No projection: keep original gate (reject unless already OK)
    if raw_error != "OK":
        info["structure_hash"] = None
        info["overall_valid_candidate"] = False
        return None, info
    return _finalize_heights(heights, observation, max_height, info)
