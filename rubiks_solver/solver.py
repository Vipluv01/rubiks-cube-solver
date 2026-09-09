"""IDA* optimal solver.

Two heuristics are provided so the value of the pattern database can be
measured directly rather than just asserted:

- `misplaced_corners_heuristic`: a cheap, valid admissible heuristic (a
  single face turn can fix at most 4 corners, so ceil(misplaced / 4) never
  overestimates) -- a believable "naive but not a strawman" baseline.
- `pdb_heuristic`: the exact distance-to-solved of the corner-only
  subproblem, looked up in the precomputed table. Strictly dominates the
  misplaced-corner heuristic (it's the *exact* answer to a relaxation of
  the real problem, not just a bound), so it should prune far more of the
  search tree -- see scripts/benchmark.py for the measured effect.

Move-pruning follows the standard two rules used by essentially every
from-scratch cube-search implementation: never follow a move with another
move of the same face (redundant), and never follow a move with one on the
opposite face in the "wrong" canonical order (opposite-face moves commute,
so only one of the two orderings needs to be explored).

The search itself works on plain-tuple cube states via cube.fast_apply
rather than on Cube objects -- numpy's per-call overhead on 8/12-element
arrays dominates at the millions-of-nodes-per-second scale IDA* runs at.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .coords import encode_corner
from .cube import Cube, FACE_OF, MOVES, SOLVED_STATE_T, fast_apply, state_of

_AXIS = {"U": 0, "D": 0, "L": 1, "R": 1, "F": 2, "B": 2}
_CANON = {"U": 0, "D": 1, "L": 0, "R": 1, "F": 0, "B": 1}

FOUND = -1


def pdb_heuristic(pdb):
    def h(state) -> int:
        cp, co, _, _ = state
        return int(pdb[encode_corner(cp, co)])
    return h


def misplaced_corners_heuristic():
    def h(state) -> int:
        cp, _, _, _ = state
        misplaced = sum(1 for i, c in enumerate(cp) if c != i)
        return -(-misplaced // 4)  # ceil division
    return h


@dataclass
class SolveResult:
    moves: list
    nodes: int
    elapsed: float
    found: bool


def solve(cube: Cube, heuristic, max_nodes: int | None = None) -> SolveResult:
    t0 = time.time()
    nodes = [0]
    start = state_of(cube)
    threshold = heuristic(start)
    path: list[str] = []

    def dfs(state, g: int, bound: int, last_move: str | None):
        nodes[0] += 1
        if max_nodes is not None and nodes[0] > max_nodes:
            return "abort"
        f = g + heuristic(state)
        if f > bound:
            return f
        if state == SOLVED_STATE_T:
            return FOUND
        min_exceed = math.inf
        for move in MOVES:
            if last_move is not None:
                move_face = FACE_OF[move]
                last_face = FACE_OF[last_move]
                if move_face == last_face:
                    continue
                if (_AXIS[move_face] == _AXIS[last_face]
                        and _CANON[move_face] < _CANON[last_face]):
                    continue
            child = fast_apply(state, move)
            path.append(move)
            result = dfs(child, g + 1, bound, move)
            if result == FOUND:
                return FOUND
            if result == "abort":
                return "abort"
            path.pop()
            if result < min_exceed:
                min_exceed = result
        return min_exceed

    while True:
        result = dfs(start, 0, threshold, None)
        if result == FOUND:
            return SolveResult(list(path), nodes[0], time.time() - t0, True)
        if result == "abort":
            return SolveResult([], nodes[0], time.time() - t0, False)
        if result == math.inf:
            return SolveResult([], nodes[0], time.time() - t0, False)
        threshold = result
