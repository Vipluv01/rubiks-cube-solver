import os

import numpy as np
import pytest

from rubiks_solver.cube import Cube, INVERSE
from rubiks_solver.scramble import random_scramble
from rubiks_solver.solver import misplaced_corners_heuristic, solve

PDB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "corner_pdb.npy")
HAVE_PDB = os.path.exists(PDB_PATH)


def _apply(moves):
    return Cube.solved().apply_sequence(moves)


def test_solve_already_solved():
    h = misplaced_corners_heuristic()
    result = solve(Cube.solved(), h)
    assert result.found
    assert result.moves == []


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_solve_short_scrambles_with_weak_heuristic(n):
    h = misplaced_corners_heuristic()
    for seed in range(5):
        cube, scramble_moves = _scramble(n, seed)
        result = solve(cube, h)
        assert result.found
        assert len(result.moves) <= n  # optimal solver: never longer than the scramble itself
        assert cube.apply_sequence(result.moves).is_solved()


def _scramble(n, seed):
    moves = random_scramble(n, seed=seed)
    return _apply(moves), moves


@pytest.mark.skipif(not HAVE_PDB, reason="corner PDB not built (run scripts/build_pdb.py)")
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_solve_short_scrambles_with_pdb(n):
    from rubiks_solver.pdb import load_pdb
    from rubiks_solver.solver import pdb_heuristic

    pdb = load_pdb(PDB_PATH)
    h = pdb_heuristic(pdb)
    for seed in range(5):
        cube, scramble_moves = _scramble(n, seed)
        result = solve(cube, h)
        assert result.found
        assert len(result.moves) <= n
        assert cube.apply_sequence(result.moves).is_solved()


@pytest.mark.skipif(not HAVE_PDB, reason="corner PDB not built (run scripts/build_pdb.py)")
def test_pdb_heuristic_is_admissible_and_at_least_as_tight_as_misplaced():
    # Admissibility check: on many random states, the PDB distance must
    # never exceed the true optimal corner-only distance. We don't have an
    # independent ground truth here, but we *can* check it never exceeds
    # the length of a known solving sequence found via search, and that it
    # dominates (>=) the weaker misplaced-corner bound on every sample --
    # if the PDB heuristic were broken this would very likely be violated.
    from rubiks_solver.pdb import load_pdb
    from rubiks_solver.solver import pdb_heuristic

    pdb = load_pdb(PDB_PATH)
    h_pdb = pdb_heuristic(pdb)
    h_weak = misplaced_corners_heuristic()

    from rubiks_solver.cube import state_of

    cube, moves = _scramble(20, seed=1)
    assert h_pdb(state_of(cube)) <= len(moves)
    for seed in range(30):
        c, _ = _scramble(15, seed=seed + 100)
        assert h_pdb(state_of(c)) >= h_weak(state_of(c))
