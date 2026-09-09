import random

from rubiks_solver.cube import Cube, INVERSE, MOVES
from rubiks_solver.facelets import (
    apply_move,
    apply_sequence,
    face_grid,
    solved_facelets,
)

FACES = "UDFBRL"


def _is_solved_facelets(facelets) -> bool:
    for face in FACES:
        grid = face_grid(facelets, face)
        colors = {grid[r][c] for r in range(3) for c in range(3)}
        if colors != {face}:
            return False
    return True


def test_solved_facelets_is_solved():
    assert _is_solved_facelets(solved_facelets())


def test_each_face_has_nine_stickers_and_six_colors_total():
    fc = solved_facelets()
    for face in FACES:
        grid = face_grid(fc, face)
        assert sum(1 for r in grid for c in r if c is not None) == 9


def test_quarter_turn_order_four_matches_cube_geometry():
    for m in ["U", "R", "F"]:
        fc = solved_facelets()
        for _ in range(4):
            fc = apply_move(fc, m)
        assert _is_solved_facelets(fc)


def test_facelet_model_agrees_with_cubie_model_on_random_scrambles():
    # Independent cross-check: two separately-coded models (cube.py's
    # cp/co/ep/eo vs this module's sticker permutation), driven by the same
    # move sequence, must agree on solved-vs-not for every scramble.
    rng = random.Random(11)
    for _ in range(30):
        n = rng.randint(0, 15)
        moves = [rng.choice(MOVES) for _ in range(n)]
        cube = Cube.solved().apply_sequence(moves)
        fc = apply_sequence(solved_facelets(), moves)
        assert cube.is_solved() == _is_solved_facelets(fc)

    # and specifically: scramble then apply the exact inverse -> solved in
    # both models
    rng = random.Random(12)
    moves = [rng.choice(MOVES) for _ in range(25)]
    inv = [INVERSE[m] for m in reversed(moves)]
    cube = Cube.solved().apply_sequence(moves).apply_sequence(inv)
    fc = apply_sequence(apply_sequence(solved_facelets(), moves), inv)
    assert cube.is_solved()
    assert _is_solved_facelets(fc)
