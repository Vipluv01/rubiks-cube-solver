"""Build the corner pattern database and save it to results/corner_pdb.npy.

Run once (takes a few minutes); the solver loads the saved file afterwards.
The BFS's own closure count is the strongest correctness check on cube.py's
move-table geometry -- see pdb.py's docstring.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rubiks_solver.pdb import build_corner_pdb, save_pdb

if __name__ == "__main__":
    dist = build_corner_pdb(verbose=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "corner_pdb.npy")
    save_pdb(dist, out)
    print(f"saved {out}  ({dist.nbytes / 1e6:.1f} MB)")
    print(f"max depth in database: {dist.max()}")
