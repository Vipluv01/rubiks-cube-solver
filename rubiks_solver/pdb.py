"""Corner pattern database: an exact-distance table for the 88,179,840-state
corner-only subproblem (permutation x orientation of the 8 corners, ignoring
edges entirely), built by a breadth-first search from the solved state using
all 18 face turns.

Because every corner PDB distance is an admissible heuristic for the full
cube (it can never take *fewer* moves to fix the corners than the corners-
only distance), IDA* can use it directly to prune the full search -- that is
the mechanism this whole project demonstrates.

The BFS is vectorized over the whole frontier with numpy rather than
expanded state-by-state in a Python loop, which is what makes generating an
88-million-state table tractable in minutes rather than hours: each of the
18 moves is applied to the *entire* frontier at once via fancy indexing.
"""
from __future__ import annotations

import time

import numpy as np

from .cube import MOVES, MOVE_TABLES
from .coords import N_CORNER_STATES, N_ORI, _FACT

UNVISITED = 255

_MOVE_TCP = np.array([MOVE_TABLES[m][0] for m in MOVES], dtype=np.uint8)  # (18, 8)
_MOVE_TCO = np.array([MOVE_TABLES[m][1] for m in MOVES], dtype=np.uint8)  # (18, 8)


def _vec_perm_rank(perm_batch: np.ndarray) -> np.ndarray:
    n = perm_batch.shape[0]
    rank = np.zeros(n, dtype=np.int64)
    for i in range(7):
        less = (perm_batch[:, i + 1:] < perm_batch[:, i:i + 1]).sum(axis=1)
        rank += less.astype(np.int64) * _FACT[7 - i]
    return rank


def _vec_ori_rank(ori_batch: np.ndarray) -> np.ndarray:
    n = ori_batch.shape[0]
    r = np.zeros(n, dtype=np.int64)
    for i in range(7):
        r = r * 3 + ori_batch[:, i]
    return r


def _vec_encode(cp_batch: np.ndarray, co_batch: np.ndarray) -> np.ndarray:
    return _vec_perm_rank(cp_batch) * N_ORI + _vec_ori_rank(co_batch)


def build_corner_pdb(verbose: bool = True) -> np.ndarray:
    dist = np.full(N_CORNER_STATES, UNVISITED, dtype=np.uint8)

    frontier_cp = np.arange(8, dtype=np.uint8)[None, :]
    frontier_co = np.zeros((1, 8), dtype=np.uint8)
    start_idx = _vec_encode(frontier_cp, frontier_co)
    dist[start_idx] = 0
    total = 1
    depth = 0
    t0 = time.time()

    while frontier_cp.shape[0] > 0 and total < N_CORNER_STATES:
        depth += 1
        next_cp_parts = []
        next_co_parts = []
        for mi in range(len(MOVES)):
            t_cp = _MOVE_TCP[mi]
            t_co = _MOVE_TCO[mi]
            new_cp = frontier_cp[:, t_cp]
            new_co = (frontier_co[:, t_cp] + t_co) % 3
            idx = _vec_encode(new_cp, new_co)
            unvisited_mask = dist[idx] == UNVISITED
            if not unvisited_mask.any():
                continue
            sel = idx[unvisited_mask]
            dist[sel] = depth
            next_cp_parts.append(new_cp[unvisited_mask])
            next_co_parts.append(new_co[unvisited_mask])

        if next_cp_parts:
            frontier_cp = np.concatenate(next_cp_parts, axis=0)
            frontier_co = np.concatenate(next_co_parts, axis=0)
        else:
            frontier_cp = np.empty((0, 8), dtype=np.uint8)
            frontier_co = np.empty((0, 8), dtype=np.uint8)

        total += frontier_cp.shape[0]
        if verbose:
            print(f"  depth {depth:2d}: +{frontier_cp.shape[0]:>9,} states  "
                  f"total {total:>11,} / {N_CORNER_STATES:,}  "
                  f"({time.time() - t0:6.1f}s elapsed)")

    if total != N_CORNER_STATES:
        raise RuntimeError(
            f"corner PDB BFS closed at {total:,} states, expected "
            f"{N_CORNER_STATES:,} -- move table geometry is wrong."
        )
    return dist


def save_pdb(dist: np.ndarray, path: str) -> None:
    np.save(path, dist)


def load_pdb(path: str) -> np.ndarray:
    return np.load(path)
