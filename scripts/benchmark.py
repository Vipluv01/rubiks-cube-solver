"""Benchmark: does the corner pattern database actually help?

For a range of scramble depths, solves several random scrambles both with
the plain admissible "misplaced corners / 4" heuristic and with the corner
PDB heuristic, and records nodes expanded + wall time for each. This is the
central result of the project: the PDB is not just "an optimization", it
changes which depths are solvable at all in a reasonable budget.

Writes results/benchmark.json and prints a markdown table (also spliced
into README.md's Results section by scripts/update_readme.py).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rubiks_solver.cube import Cube
from rubiks_solver.pdb import load_pdb
from rubiks_solver.scramble import random_scramble
from rubiks_solver.solver import misplaced_corners_heuristic, pdb_heuristic, solve

PDB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "corner_pdb.npy")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark.json")

DEPTHS = [3, 4, 5, 6, 7, 8, 9, 10]
SEEDS = [0, 1, 2]
WEAK_MAX_DEPTH = 7          # beyond this the weak heuristic reliably blows the budget
WEAK_MAX_NODES = 6_000_000
PDB_MAX_NODES = 30_000_000


def run_one(depth, seed, heuristic, max_nodes):
    moves = random_scramble(depth, seed=seed)
    cube = Cube.solved().apply_sequence(moves)
    result = solve(cube, heuristic, max_nodes=max_nodes)
    assert not result.found or cube.apply_sequence(result.moves).is_solved()
    return {
        "depth": depth,
        "seed": seed,
        "nodes": result.nodes,
        "time_s": round(result.elapsed, 4),
        "found": result.found,
        "solution_len": len(result.moves) if result.found else None,
    }


def summarize(rows):
    found = [r for r in rows if r["found"]]
    return {
        "attempted": len(rows),
        "solved": len(found),
        "avg_nodes": round(sum(r["nodes"] for r in rows) / len(rows)) if rows else None,
        "avg_time_s": round(sum(r["time_s"] for r in rows) / len(rows), 3) if rows else None,
        "avg_solution_len": (
            round(sum(r["solution_len"] for r in found) / len(found), 2) if found else None
        ),
    }


def main():
    pdb = load_pdb(PDB_PATH)
    h_pdb = pdb_heuristic(pdb)
    h_weak = misplaced_corners_heuristic()

    results = {"weak": {}, "pdb": {}}
    t0 = time.time()

    for depth in DEPTHS:
        if depth <= WEAK_MAX_DEPTH:
            rows = [run_one(depth, s, h_weak, WEAK_MAX_NODES) for s in SEEDS]
            results["weak"][depth] = summarize(rows)
            print(f"[weak] depth {depth}: {results['weak'][depth]}")

        rows = [run_one(depth, s, h_pdb, PDB_MAX_NODES) for s in SEEDS]
        results["pdb"][depth] = summarize(rows)
        print(f"[pdb]  depth {depth}: {results['pdb'][depth]}")

    results["_meta"] = {
        "depths": DEPTHS,
        "seeds": SEEDS,
        "weak_max_depth": WEAK_MAX_DEPTH,
        "weak_max_nodes": WEAK_MAX_NODES,
        "pdb_max_nodes": PDB_MAX_NODES,
        "total_wall_time_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")
    print_markdown_table(results)


def print_markdown_table(results):
    print("\n| depth | weak: avg nodes | weak: avg time | PDB: avg nodes | PDB: avg time | speedup (nodes) |")
    print("|---|---|---|---|---|---|")
    for depth in DEPTHS:
        w = results["weak"].get(depth)
        p = results["pdb"][depth]
        if w and w["avg_nodes"]:
            speedup = f"{w['avg_nodes'] / p['avg_nodes']:.0f}x" if p["avg_nodes"] else "-"
            w_nodes = f"{w['avg_nodes']:,}" + ("" if w["solved"] == w["attempted"] else "*")
            w_time = f"{w['avg_time_s']:.2f}s"
        else:
            speedup, w_nodes, w_time = "-", "not run", "-"
        p_nodes = f"{p['avg_nodes']:,}" if p["avg_nodes"] is not None else "-"
        p_time = f"{p['avg_time_s']:.2f}s" if p["avg_time_s"] is not None else "-"
        print(f"| {depth} | {w_nodes} | {w_time} | {p_nodes} | {p_time} | {speedup} |")
    print("\n(* = did not solve all seeds within the node budget; average is over attempted nodes only)")


if __name__ == "__main__":
    main()
