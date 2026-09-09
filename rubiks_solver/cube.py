"""Cubie-level Rubik's cube model.

Corners and edges are tracked as (permutation, orientation) arrays, the
standard representation for cube search. Move tables (how each of the 18
face turns permutes positions and shifts orientations) are not hand-derived
by hand-tracing cycles -- they are *simulated*: each of the 6 base face
turns is a genuine 3D rotation matrix acting on cubie positions in {-1,0,1}^3
coordinates, applied once to the solved cube. Orientation is then read off
by tracking, per affected cubie, the accumulated rotation matrix and asking
which axis its home "reference" sticker currently points along.

This sidesteps hand-transcribing cycle notation (an easy place to introduce
a silent sign error) at the cost of a bit more setup code -- see
tests/test_cube.py for the validation this buys: the corner subgroup
reachable from solved is checked to have *exactly* 8! * 3^7 = 88,179,840
elements, the known size of that group. That count only comes out right if
every move's permutation and orientation effect is correct, so it is a
strong end-to-end correctness check on the geometry below.
"""
from __future__ import annotations

import operator as _operator

import numpy as np

# Cubie home positions in (x, y, z) coordinates, x: L(-1)/R(+1),
# y: D(-1)/U(+1), z: B(-1)/F(+1).
CORNER_POS = [
    (1, 1, 1),    # 0 URF
    (-1, 1, 1),   # 1 UFL
    (-1, 1, -1),  # 2 ULB
    (1, 1, -1),   # 3 UBR
    (1, -1, 1),   # 4 DFR
    (-1, -1, 1),  # 5 DLF
    (-1, -1, -1), # 6 DBL
    (1, -1, -1),  # 7 DRB
]

EDGE_POS = [
    (1, 1, 0),    # 0  UR
    (0, 1, 1),    # 1  UF
    (-1, 1, 0),   # 2  UL
    (0, 1, -1),   # 3  UB
    (1, -1, 0),   # 4  DR
    (0, -1, 1),   # 5  DF
    (-1, -1, 0),  # 6  DL
    (0, -1, -1),  # 7  DB
    (1, 0, 1),    # 8  FR
    (-1, 0, 1),   # 9  FL
    (-1, 0, -1),  # 10 BL
    (1, 0, -1),   # 11 BR
]

_CORNER_INDEX = {pos: i for i, pos in enumerate(CORNER_POS)}
_EDGE_INDEX = {pos: i for i, pos in enumerate(EDGE_POS)}

# 90-degree rotation matrices for each face, clockwise viewed from outside
# that face, derived from first-principles compass/rotation reasoning (see
# module docstring / README for the derivation).
BASE_MOVES = ["U", "D", "R", "L", "F", "B"]

_MOVE_MATRIX = {
    "U": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    "D": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    "R": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    "L": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    "F": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    "B": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}

# which coordinate axis (0=x,1=y,2=z) and sign each move's layer occupies
_MOVE_LAYER = {
    "U": (1, 1), "D": (1, -1),
    "R": (0, 1), "L": (0, -1),
    "F": (2, 1), "B": (2, -1),
}


def _corner_chirality(pos: tuple[int, int, int]) -> int:
    """+1/-1 checkerboard class of a corner position (product of its signs).
    Adjacent corners around any face's 4-cycle always alternate class, which
    is why the same physical rotation matrix has to be decoded with mirrored
    sign for the two classes to get a consistent, additive Z/3 orientation
    delta (see test_cube.py's order-4 and sum-invariant checks, which is
    what exposed the need for this)."""
    return pos[0] * pos[1] * pos[2]


def _corner_orientation_code(mat: np.ndarray, source_pos: tuple[int, int, int]) -> int:
    """Which axis the corner's home Y-facing sticker currently points along,
    remapped so identity -> 0, mirrored by source-position chirality."""
    col = mat[:, 1]
    axis = int(np.nonzero(col)[0][0])
    if _corner_chirality(source_pos) > 0:
        return (axis - 1) % 3
    return (1 - axis) % 3


def _edge_orientation_code(mat: np.ndarray, home_axes: tuple[int, int]) -> int:
    """Whether the edge's designated reference sticker still points along
    its home axis. home_axes are the two nonzero coordinate axes for this
    edge; the smaller-indexed one is the reference."""
    ref = home_axes[0]
    col = mat[:, ref]
    axis = int(np.nonzero(col)[0][0])
    return 0 if axis == ref else 1


def _home_axes(pos: tuple[int, int, int]) -> tuple[int, ...]:
    return tuple(i for i, c in enumerate(pos) if c != 0)


def _build_move_table(move: str):
    """Simulate applying `move` once to the solved cube; returns
    (corner_perm, corner_ori_delta, edge_perm, edge_ori_delta) such that,
    for ANY cube state:
        new_cp[i] = old_cp[corner_perm[i]]
        new_co[i] = (old_co[corner_perm[i]] + corner_ori_delta[i]) % 3
    and likewise for edges.
    """
    mat = _MOVE_MATRIX[move]
    axis, sign = _MOVE_LAYER[move]

    corner_perm = list(range(8))
    corner_delta = [0] * 8
    for i, pos in enumerate(CORNER_POS):
        if pos[axis] != sign:
            continue
        new_pos = tuple(int(v) for v in mat @ np.array(pos))
        dest = _CORNER_INDEX[new_pos]
        corner_perm[dest] = i
        corner_delta[dest] = _corner_orientation_code(mat, pos)

    edge_perm = list(range(12))
    edge_delta = [0] * 12
    for i, pos in enumerate(EDGE_POS):
        if pos[axis] != sign:
            continue
        new_pos = tuple(int(v) for v in mat @ np.array(pos))
        dest = _EDGE_INDEX[new_pos]
        edge_perm[dest] = i
        edge_delta[dest] = _edge_orientation_code(mat, _home_axes(pos))

    return (
        np.array(corner_perm), np.array(corner_delta),
        np.array(edge_perm), np.array(edge_delta),
    )


_BASE_TABLES = {m: _build_move_table(m) for m in BASE_MOVES}


def _compose(cp, co, ep, eo, table):
    t_cp, t_co, t_ep, t_eo = table
    new_cp = cp[t_cp]
    new_co = (co[t_cp] + t_co) % 3
    new_ep = ep[t_ep]
    new_eo = (eo[t_ep] + t_eo) % 2
    return new_cp, new_co, new_ep, new_eo


def _all_move_tables():
    """All 18 quarter/half/inverse-quarter turns, by repeated composition
    of the 6 base tables."""
    tables = {}
    for m in BASE_MOVES:
        cp, co, ep, eo = np.arange(8), np.zeros(8, int), np.arange(12), np.zeros(12, int)
        cp, co, ep, eo = _compose(cp, co, ep, eo, _BASE_TABLES[m])
        tables[m] = (cp, co, ep, eo)
        cp2, co2, ep2, eo2 = _compose(cp, co, ep, eo, _BASE_TABLES[m])
        tables[m + "2"] = (cp2, co2, ep2, eo2)
        cp3, co3, ep3, eo3 = _compose(cp2, co2, ep2, eo2, _BASE_TABLES[m])
        tables[m + "'"] = (cp3, co3, ep3, eo3)
    return tables


MOVE_TABLES = _all_move_tables()
MOVES = list(MOVE_TABLES.keys())  # 18 moves

# For search move-pruning: never follow a move with another move of the
# same face (it would either undo progress or be equivalent to one move).
FACE_OF = {m: m[0] for m in MOVES}


class Cube:
    __slots__ = ("cp", "co", "ep", "eo")

    def __init__(self, cp=None, co=None, ep=None, eo=None):
        self.cp = np.arange(8) if cp is None else cp
        self.co = np.zeros(8, dtype=np.int64) if co is None else co
        self.ep = np.arange(12) if ep is None else ep
        self.eo = np.zeros(12, dtype=np.int64) if eo is None else eo

    @staticmethod
    def solved() -> "Cube":
        return Cube()

    def is_solved(self) -> bool:
        return (
            np.array_equal(self.cp, np.arange(8))
            and not self.co.any()
            and np.array_equal(self.ep, np.arange(12))
            and not self.eo.any()
        )

    def apply(self, move: str) -> "Cube":
        cp, co, ep, eo = _compose(self.cp, self.co, self.ep, self.eo, MOVE_TABLES[move])
        return Cube(cp, co, ep, eo)

    def apply_sequence(self, moves) -> "Cube":
        cube = self
        for m in moves:
            cube = cube.apply(m)
        return cube

    def copy(self) -> "Cube":
        return Cube(self.cp.copy(), self.co.copy(), self.ep.copy(), self.eo.copy())

    def __eq__(self, other):
        return (
            np.array_equal(self.cp, other.cp)
            and np.array_equal(self.co, other.co)
            and np.array_equal(self.ep, other.ep)
            and np.array_equal(self.eo, other.eo)
        )


# Plain-tuple move tables for the search hot path: numpy's per-call overhead
# on 8/12-element arrays dominates at the millions-of-nodes-per-second scale
# IDA* needs, so the solver applies moves via these instead of Cube.apply.
MOVE_TABLES_T = {
    m: tuple(tuple(int(x) for x in arr) for arr in tables)
    for m, tables in MOVE_TABLES.items()
}
SOLVED_STATE_T = (tuple(range(8)), (0,) * 8, tuple(range(12)), (0,) * 12)

_ADD3 = [[(a + b) % 3 for b in range(3)] for a in range(3)]
_ADD2 = [[(a + b) % 2 for b in range(2)] for a in range(2)]

# Per-move C-level gather (operator.itemgetter, compiled) plus a precomputed
# per-destination-slot orientation lookup table, so applying a move is just
# two itemgetter calls and two small comprehensions with no modulo op or
# zip() overhead in the loop body -- this runs on every node IDA* visits.
_FAST_MOVE = {}
for _m, (_t_cp, _t_co, _t_ep, _t_eo) in MOVE_TABLES_T.items():
    _cp_get = _operator.itemgetter(*_t_cp)
    _ep_get = _operator.itemgetter(*_t_ep)
    _co_add = tuple(_ADD3[_d] for _d in _t_co)   # co_add[slot][old_value] -> new_value
    _eo_add = tuple(_ADD2[_d] for _d in _t_eo)
    _FAST_MOVE[_m] = (_cp_get, _co_add, _t_cp, _ep_get, _eo_add, _t_ep)


def state_of(cube: "Cube"):
    return (
        tuple(int(x) for x in cube.cp),
        tuple(int(x) for x in cube.co),
        tuple(int(x) for x in cube.ep),
        tuple(int(x) for x in cube.eo),
    )


def fast_apply(state, move: str):
    cp, co, ep, eo = state
    cp_get, co_add, t_cp, ep_get, eo_add, t_ep = _FAST_MOVE[move]
    new_cp = cp_get(cp)
    new_co = tuple(add[co[i]] for add, i in zip(co_add, t_cp))
    new_ep = ep_get(ep)
    new_eo = tuple(add[eo[i]] for add, i in zip(eo_add, t_ep))
    return new_cp, new_co, new_ep, new_eo


INVERSE = {}
for m in MOVES:
    if m.endswith("2"):
        INVERSE[m] = m
    elif m.endswith("'"):
        INVERSE[m] = m[0]
    else:
        INVERSE[m] = m + "'"
