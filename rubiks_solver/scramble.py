import random

from .cube import Cube, FACE_OF, MOVES


def random_scramble(n: int, seed: int | None = None) -> list:
    rng = random.Random(seed)
    moves = []
    last_face = None
    while len(moves) < n:
        move = rng.choice(MOVES)
        if last_face is not None and FACE_OF[move] == last_face:
            continue
        moves.append(move)
        last_face = FACE_OF[move]
    return moves


def scrambled_cube(n: int, seed: int | None = None):
    moves = random_scramble(n, seed)
    return Cube.solved().apply_sequence(moves), moves
