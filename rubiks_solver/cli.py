import argparse
import os

from .cube import Cube
from .pdb import load_pdb
from .scramble import random_scramble
from .solver import pdb_heuristic, solve

DEFAULT_PDB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "corner_pdb.npy")


def main():
    ap = argparse.ArgumentParser(description="Optimal Rubik's cube solver (IDA* + corner PDB)")
    ap.add_argument("--scramble", type=str, help="space-separated moves, e.g. \"R U R' U'\"")
    ap.add_argument("--random", type=int, metavar="N", help="generate a random N-move scramble instead")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pdb", type=str, default=DEFAULT_PDB_PATH)
    ap.add_argument("--max-nodes", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.pdb):
        raise SystemExit(
            f"pattern database not found at {args.pdb}\n"
            f"run: .venv/bin/python scripts/build_pdb.py"
        )
    pdb = load_pdb(args.pdb)
    h = pdb_heuristic(pdb)

    if args.scramble:
        moves = args.scramble.split()
    elif args.random is not None:
        moves = random_scramble(args.random, seed=args.seed)
    else:
        moves = random_scramble(8, seed=args.seed)

    print(f"scramble ({len(moves)} moves): {' '.join(moves)}")
    cube = Cube.solved().apply_sequence(moves)

    result = solve(cube, h, max_nodes=args.max_nodes)
    if not result.found:
        print(f"no solution found within budget ({result.nodes} nodes, {result.elapsed:.2f}s)")
        return

    print(f"solution ({len(result.moves)} moves): {' '.join(result.moves)}")
    print(f"nodes expanded: {result.nodes:,}   time: {result.elapsed:.3f}s")

    check = cube.apply_sequence(result.moves)
    assert check.is_solved(), "internal error: solution did not solve the cube"


if __name__ == "__main__":
    main()
