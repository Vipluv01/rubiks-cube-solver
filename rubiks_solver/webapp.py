"""FastAPI backend for the browser demo: runs the real IDA* + pattern-
database solver server-side (it needs the 88MB PDB file and CPU-bound
search, neither of which belong in the browser) and serves the static
frontend that visualizes scrambling/solving as a cube net plus the
weak-vs-PDB benchmark chart.
"""
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .cube import Cube, MOVES
from .facelets import apply_sequence, face_grid, solved_facelets
from .pdb import load_pdb
from .scramble import random_scramble
from .solver import misplaced_corners_heuristic, pdb_heuristic, solve

ROOT = os.path.join(os.path.dirname(__file__), "..")
PDB_PATH = os.path.join(ROOT, "results", "corner_pdb.npy")
BENCHMARK_PATH = os.path.join(ROOT, "results", "benchmark.json")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Rubik's Cube Solver")

_pdb = None


def get_pdb():
    global _pdb
    if _pdb is None:
        if not os.path.exists(PDB_PATH):
            raise HTTPException(
                503,
                "Pattern database not built yet. Run: .venv/bin/python scripts/build_pdb.py",
            )
        _pdb = load_pdb(PDB_PATH)
    return _pdb


class ScrambleRequest(BaseModel):
    n: int = 8
    seed: int | None = None


class SolveRequest(BaseModel):
    moves: list[str]
    heuristic: str = "pdb"  # "pdb" or "weak"


def _net_grids(moves: list[str]):
    fc = apply_sequence(solved_facelets(), moves)
    return {face: face_grid(fc, face) for face in "UDFBRL"}


@app.post("/api/scramble")
def api_scramble(req: ScrambleRequest):
    n = max(0, min(req.n, 40))
    moves = random_scramble(n, seed=req.seed)
    return {"moves": moves, "grids": _net_grids(moves)}


@app.post("/api/net")
def api_net(req: SolveRequest):
    for m in req.moves:
        if m not in MOVES:
            raise HTTPException(400, f"unknown move {m!r}")
    return {"grids": _net_grids(req.moves)}


@app.post("/api/solve")
def api_solve(req: SolveRequest):
    for m in req.moves:
        if m not in MOVES:
            raise HTTPException(400, f"unknown move {m!r}")
    cube = Cube.solved().apply_sequence(req.moves)

    if req.heuristic == "weak":
        h = misplaced_corners_heuristic()
        max_nodes = 3_000_000
    else:
        h = pdb_heuristic(get_pdb())
        max_nodes = 30_000_000

    result = solve(cube, h, max_nodes=max_nodes)
    if not result.found:
        raise HTTPException(
            504,
            f"no solution found within {max_nodes:,} nodes "
            f"({result.nodes:,} explored, {result.elapsed:.1f}s) -- try a shorter scramble",
        )

    check = cube.apply_sequence(result.moves)
    if not check.is_solved():
        raise HTTPException(500, "internal error: solution did not verify")

    return {
        "solution": result.moves,
        "nodes": result.nodes,
        "time_s": result.elapsed,
        "heuristic": req.heuristic,
    }


@app.get("/api/benchmark")
def api_benchmark():
    if not os.path.exists(BENCHMARK_PATH):
        raise HTTPException(404, "no benchmark data yet -- run scripts/benchmark.py")
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


@app.get("/api/pdb_status")
def api_pdb_status():
    return {"available": os.path.exists(PDB_PATH)}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
