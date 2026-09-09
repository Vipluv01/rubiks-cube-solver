"""Sticker-level (facelet) cube model, used only for rendering.

This is deliberately a *separate* implementation from cube.py's cubie-level
(permutation, orientation) model, not a conversion from it: each move's
effect on sticker positions is derived straight from the same rotation
matrices used in cube.py, applied directly to (position, facing-axis) pairs
rather than inverting the compact orientation-code encoding (which bakes in
a chirality-mirroring convention that would be needlessly fragile to
invert correctly). Tracking facelets independently and comparing against
the cubie model's is_solved() on random scrambles (see
tests/test_facelets.py) is a genuine cross-check between two independently
coded models, not a tautology.
"""
from __future__ import annotations

import numpy as np

from .cube import BASE_MOVES, CORNER_POS, EDGE_POS, _MOVE_LAYER, _MOVE_MATRIX

_AXIS_COLOR = {
    (0, 1): "R", (0, -1): "L",
    (1, 1): "U", (1, -1): "D",
    (2, 1): "F", (2, -1): "B",
}


def _sticker_ids():
    ids = []
    for p in CORNER_POS:
        for a in range(3):
            ids.append((p, a))
    for p in EDGE_POS:
        for a in range(3):
            if p[a] != 0:
                ids.append((p, a))
    return ids


_STICKER_IDS = _sticker_ids()


def solved_facelets() -> dict:
    return {(p, a): _AXIS_COLOR[(a, p[a])] for (p, a) in _STICKER_IDS}


def _apply_axis(mat: np.ndarray, a: int) -> int:
    vec = np.zeros(3, dtype=int)
    vec[a] = 1
    newvec = mat @ vec
    return int(np.nonzero(newvec)[0][0])


def _build_facelet_move_table(move: str):
    mat = _MOVE_MATRIX[move]
    axis, sign = _MOVE_LAYER[move]
    mapping = {}
    for p in list(CORNER_POS) + list(EDGE_POS):
        if p[axis] != sign:
            continue
        new_p = tuple(int(v) for v in mat @ np.array(p))
        for a in range(3):
            if p[a] == 0:
                continue
            if (p, a) not in _AXIS_COLOR_KEYS:
                continue
            new_a = _apply_axis(mat, a)
            mapping[(p, a)] = (new_p, new_a)
    return mapping


_AXIS_COLOR_KEYS = set(_STICKER_IDS)

_BASE_FACELET_TABLES = {m: _build_facelet_move_table(m) for m in BASE_MOVES}


def _facelet_tables_all_moves():
    tables = {}
    for m in BASE_MOVES:
        base = _BASE_FACELET_TABLES[m]

        def compose(mapping, base=base):
            out = {}
            for sid in _STICKER_IDS:
                mid = mapping.get(sid, sid)
                out[sid] = base.get(mid, mid)
            return out

        identity = {sid: sid for sid in _STICKER_IDS}
        one = compose(identity)
        tables[m] = one
        two = compose(one)
        tables[m + "2"] = two
        three = compose(two)
        tables[m + "'"] = three
    return tables


FACELET_TABLES = _facelet_tables_all_moves()


def apply_move(facelets: dict, move: str) -> dict:
    table = FACELET_TABLES[move]
    return {new_sid: facelets[old_sid] for old_sid, new_sid in table.items()}


def apply_sequence(facelets: dict, moves) -> dict:
    for m in moves:
        facelets = apply_move(facelets, m)
    return facelets


# Net layout: which (face letter, row, col) each sticker corresponds to,
# for rendering as a 2D cross/net. Face letters index into CORNER_POS /
# EDGE_POS via which axis+sign the sticker's own facing direction is.
_FACE_AXIS = {"U": (1, 1), "D": (1, -1), "F": (2, 1), "B": (2, -1), "R": (0, 1), "L": (0, -1)}

# For each face, the in-plane (row, col) basis vectors (the other two axes),
# chosen so the resulting 3x3 grid reads top-to-bottom, left-to-right the
# way the face looks from outside.
_FACE_BASIS = {
    "U": ((2, -1), (0, 1)),
    "D": ((2, 1), (0, 1)),
    "F": ((1, -1), (0, 1)),
    "B": ((1, -1), (0, -1)),
    "R": ((1, -1), (2, -1)),
    "L": ((1, -1), (2, 1)),
}


def face_grid(facelets: dict, face: str):
    """Return a 3x3 list of colors for `face`, as viewed from outside."""
    axis, sign = _FACE_AXIS[face]
    (row_axis, row_sign), (col_axis, col_sign) = _FACE_BASIS[face]
    grid = [[None] * 3 for _ in range(3)]
    for (p, a), color in facelets.items():
        if a != axis or p[axis] != sign:
            continue
        row = 1 - row_sign * p[row_axis]
        col = 1 + col_sign * p[col_axis]
        grid[row][col] = color
    grid[1][1] = face  # center facelet never moves on a 3x3
    return grid
