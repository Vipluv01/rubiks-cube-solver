const FACE_ORDER = "UDFBRL";

const el = (id) => document.getElementById(id);
const netEl = el("net");
const moveListEl = el("moveList");
const statusEl = el("status");
const moveIndicatorEl = el("moveIndicator");

let scrambleMoves = [];
let solutionMoves = null;
let playIndex = 0;
let playTimer = null;

function buildNetSkeleton() {
  for (const face of FACE_ORDER) {
    const faceEl = document.querySelector(`.face[data-face="${face}"]`);
    faceEl.innerHTML = "";
    for (let i = 0; i < 9; i++) {
      const s = document.createElement("div");
      s.className = "sticker";
      faceEl.appendChild(s);
    }
  }
}

function renderGrids(grids) {
  for (const face of FACE_ORDER) {
    const faceEl = document.querySelector(`.face[data-face="${face}"]`);
    const grid = grids[face];
    const stickers = faceEl.children;
    let i = 0;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        stickers[i].style.background = `var(--c-${grid[r][c]})`;
        i++;
      }
    }
  }
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function setStatus(msg, isError = true) {
  statusEl.textContent = msg || "";
  statusEl.style.color = isError ? "var(--danger)" : "var(--text-dim)";
}

function setBusy(busy) {
  el("btnScramble").disabled = busy;
  el("btnSolve").disabled = busy;
}

function renderScrambleMoveList() {
  moveListEl.textContent = scrambleMoves.length
    ? "scramble: " + scrambleMoves.join(" ")
    : "scramble: (already solved)";
}

function renderSolutionMoveList() {
  if (!solutionMoves) return;
  const parts = solutionMoves.map((m, i) => {
    const cls = i < playIndex ? "done" : i === playIndex ? "cur" : "";
    return cls ? `<span class="${cls}">${m}</span>` : m;
  });
  moveListEl.innerHTML = "solution: " + (parts.join(" ") || "(no moves needed)");
}

async function stepTo(index) {
  playIndex = Math.max(0, Math.min(index, solutionMoves.length));
  const moves = scrambleMoves.concat(solutionMoves.slice(0, playIndex));
  const { grids } = await api("/api/net", { moves });
  renderGrids(grids);
  renderSolutionMoveList();
  moveIndicatorEl.textContent = `step ${playIndex} / ${solutionMoves.length}`;
  el("btnPrev").disabled = playIndex === 0;
  el("btnNext").disabled = playIndex === solutionMoves.length;
  el("btnReset").disabled = playIndex === 0;
  if (playIndex === solutionMoves.length) stopPlaying();
}

function stopPlaying() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    el("btnPlay").textContent = "▶ play";
  }
}

function togglePlay() {
  if (playTimer) {
    stopPlaying();
    return;
  }
  el("btnPlay").textContent = "⏸ pause";
  playTimer = setInterval(() => {
    if (playIndex >= solutionMoves.length) {
      stopPlaying();
      return;
    }
    stepTo(playIndex + 1);
  }, 650);
}

async function doScramble() {
  setStatus("");
  setBusy(true);
  stopPlaying();
  try {
    const n = parseInt(el("scrambleLen").value, 10) || 0;
    const { moves, grids } = await api("/api/scramble", { n });
    scrambleMoves = moves;
    solutionMoves = null;
    playIndex = 0;
    renderGrids(grids);
    renderScrambleMoveList();
    resetStats();
    for (const b of ["btnPrev", "btnPlay", "btnNext", "btnReset"]) el(b).disabled = true;
    moveIndicatorEl.textContent = "";
  } catch (e) {
    setStatus(String(e.message || e));
  } finally {
    setBusy(false);
  }
}

async function doSolve() {
  setStatus("");
  setBusy(true);
  setStatus("solving…", false);
  try {
    const heuristic = el("heuristic").value;
    const { solution, nodes, time_s, heuristic: h } = await api("/api/solve", {
      moves: scrambleMoves,
      heuristic,
    });
    solutionMoves = solution;
    playIndex = 0;
    el("statNodes").textContent = nodes.toLocaleString();
    el("statTime").textContent = `${time_s.toFixed(3)}s`;
    el("statLen").textContent = solution.length;
    el("statHeuristic").textContent = h === "pdb" ? "pattern database" : "misplaced corners";
    renderSolutionMoveList();
    for (const b of ["btnPrev", "btnPlay", "btnNext", "btnReset"]) el(b).disabled = false;
    el("btnPrev").disabled = true;
    await stepTo(0);
    setStatus("");
  } catch (e) {
    setStatus(String(e.message || e));
  } finally {
    setBusy(false);
  }
}

function resetStats() {
  el("statNodes").textContent = "—";
  el("statTime").textContent = "—";
  el("statLen").textContent = "—";
  el("statHeuristic").textContent = "—";
}

function renderChart(data) {
  const depths = data._meta.depths;
  const chartEl = el("chart");
  const w = Math.max(560, depths.length * 70);
  const h = 220;
  const padL = 46, padB = 26, padT = 10, padR = 8;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const points = depths.map((d) => ({
    depth: d,
    weak: data.weak[d] ? data.weak[d].avg_nodes : null,
    pdb: data.pdb[d] ? data.pdb[d].avg_nodes : null,
  }));
  const allVals = points.flatMap((p) => [p.weak, p.pdb]).filter((v) => v != null && v > 0);
  const maxLog = Math.log10(Math.max(...allVals));
  const minLog = 0; // log10(1)

  const groupW = plotW / depths.length;
  const barW = groupW / 3.2;

  const yFor = (v) => {
    const lg = Math.log10(Math.max(v, 1));
    return padT + plotH * (1 - (lg - minLog) / (maxLog - minLog));
  };

  let bars = "";
  points.forEach((p, i) => {
    const cx = padL + i * groupW + groupW / 2;
    if (p.weak != null) {
      const y = yFor(p.weak);
      bars += `<rect class="bar-weak" x="${cx - barW - 2}" y="${y}" width="${barW}" height="${padT + plotH - y}"><title>depth ${p.depth} weak: ${p.weak.toLocaleString()} nodes</title></rect>`;
    }
    if (p.pdb != null) {
      const y = yFor(p.pdb);
      bars += `<rect class="bar-pdb" x="${cx + 2}" y="${y}" width="${barW}" height="${padT + plotH - y}"><title>depth ${p.depth} pdb: ${p.pdb.toLocaleString()} nodes</title></rect>`;
    }
    bars += `<text x="${cx}" y="${h - 8}" text-anchor="middle">${p.depth}</text>`;
  });

  const gridlines = [1, 2, 3, 4, 5, 6, 7]
    .filter((lg) => lg <= maxLog + 0.01)
    .map((lg) => {
      const y = padT + plotH * (1 - lg / maxLog);
      return `<line x1="${padL}" x2="${w - padR}" y1="${y}" y2="${y}" stroke="#2b2f3a" stroke-width="1"/>
              <text x="${padL - 6}" y="${y + 3}" text-anchor="end">10^${lg}</text>`;
    })
    .join("");

  chartEl.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">
      ${gridlines}
      ${bars}
      <text class="axis-label" x="${padL}" y="${padT}" text-anchor="start">nodes (log scale)</text>
      <text class="axis-label" x="${w / 2}" y="${h}" text-anchor="middle">scramble depth</text>
    </svg>
    <div style="display:flex;gap:16px;font-size:0.78rem;color:var(--text-dim);margin-top:4px;">
      <span><span style="display:inline-block;width:10px;height:10px;background:var(--accent);margin-right:5px;border-radius:2px;"></span>pattern database</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:#4b5165;margin-right:5px;border-radius:2px;"></span>misplaced corners (weak)</span>
    </div>`;
}

async function loadBenchmark() {
  try {
    const res = await fetch("/api/benchmark");
    if (!res.ok) return;
    const data = await res.json();
    renderChart(data);
    el("chartCaption").textContent =
      `avg over ${data._meta.seeds.length} scrambles per depth; weak heuristic not run past depth ${data._meta.weak_max_depth} (blows the node budget).`;
  } catch (_) {
    el("chart").textContent = "benchmark data unavailable.";
  }
}

async function init() {
  buildNetSkeleton();
  try {
    const { grids } = await api("/api/net", { moves: [] });
    renderGrids(grids);
  } catch (e) {
    setStatus(String(e.message || e));
  }
  loadBenchmark();

  el("btnScramble").addEventListener("click", doScramble);
  el("btnSolve").addEventListener("click", doSolve);
  el("btnPrev").addEventListener("click", () => stepTo(playIndex - 1));
  el("btnNext").addEventListener("click", () => stepTo(playIndex + 1));
  el("btnReset").addEventListener("click", () => stepTo(0));
  el("btnPlay").addEventListener("click", togglePlay);
}

init();
