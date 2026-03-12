#!/usr/bin/env node
/**
 * ai/dashboard.js — Live training dashboard
 *
 * Serves a progress page on port 8080 while selfplay.js runs alongside it.
 * Reads the stats JSON written by selfplay.js --stats and streams updates via SSE.
 *
 * Usage (started automatically by ec2_train.sh):
 *   node ai/dashboard.js --stats /tmp/acquire-training/stats.json
 */

const http = require('http');
const fs   = require('fs');
const path = require('path');

const args      = process.argv.slice(2);
const statsFile = args[args.indexOf('--stats') + 1] || '/tmp/acquire-training/stats.json';
const PORT      = parseInt(process.env.PORT || '8080', 10);

const HTML = /* html */`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Acquire AI Training</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding: 24px; }
  h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
  .sub { color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 12px; margin-bottom: 28px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
  .card-value { font-size: 1.6rem; font-weight: 700; }
  .card-value.green { color: #3fb950; }
  .card-value.yellow { color: #d29922; }
  .progress-wrap { margin-bottom: 28px; }
  .progress-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 6px; display: flex; justify-content: space-between; }
  .bar-bg { background: #21262d; border-radius: 4px; height: 10px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg,#238636,#3fb950); border-radius: 4px; transition: width .5s; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { text-align: left; padding: 8px 12px; color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; font-size: 0.75rem; text-transform: uppercase; }
  td { padding: 10px 12px; border-bottom: 1px solid #21262d; }
  tr:last-child td { border-bottom: none; }
  .elo-bar-wrap { display: flex; align-items: center; gap: 8px; }
  .elo-bar { height: 6px; background: #3fb950; border-radius: 3px; transition: width .5s; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
  .badge-mcts { background: #1f4788; color: #79c0ff; }
  .badge-hard { background: #3d1f1f; color: #f85149; }
  .badge-medium { background: #2d2d00; color: #e3b341; }
  .status { font-size: 0.75rem; color: #3fb950; margin-top: 20px; }
  .status.stale { color: #8b949e; }
</style>
</head>
<body>
<h1>🤖 Acquire AI Training</h1>
<div class="sub" id="updated">Loading…</div>

<div class="cards">
  <div class="card">
    <div class="card-label">Games Played</div>
    <div class="card-value green" id="games">—</div>
  </div>
  <div class="card">
    <div class="card-label">Training Records</div>
    <div class="card-value" id="records">—</div>
  </div>
  <div class="card">
    <div class="card-label">Elapsed</div>
    <div class="card-value" id="elapsed">—</div>
  </div>
  <div class="card">
    <div class="card-label">Time Remaining</div>
    <div class="card-value yellow" id="remaining">—</div>
  </div>
  <div class="card">
    <div class="card-label">Errors</div>
    <div class="card-value" id="errors">—</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Turns/Game</div>
    <div class="card-value" id="turns">—</div>
  </div>
</div>

<div class="progress-wrap">
  <div class="progress-label"><span>Session Progress</span><span id="pct">0%</span></div>
  <div class="bar-bg"><div class="bar-fill" id="bar" style="width:0%"></div></div>
</div>

<table>
  <thead><tr>
    <th>Bot</th>
    <th>ELO ★</th>
    <th>Win%</th>
    <th>Top 3%</th>
    <th>Avg Cash</th>
    <th>Type</th>
  </tr></thead>
  <tbody id="bots"></tbody>
</table>

<div class="status" id="status">⏳ Waiting for data…</div>

<script>
function fmt(s) {
  if (s === null || s === undefined) return '—';
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  if (h > 0) return h+'h '+String(m).padStart(2,'0')+'m';
  return m+'m '+String(sec).padStart(2,'0')+'s';
}
function fmtCash(n) { return '$'+n.toLocaleString(); }

const DIFF_BADGE = { mcts:'badge-mcts', hard:'badge-hard', medium:'badge-medium' };

function render(d) {
  document.getElementById('games').textContent    = d.gamesPlayed.toLocaleString();
  document.getElementById('records').textContent  = (d.exportedRecords||0).toLocaleString();
  document.getElementById('elapsed').textContent  = fmt(d.elapsedSecs);
  document.getElementById('remaining').textContent= fmt(d.remainingSecs);
  document.getElementById('errors').textContent   = d.errors;
  document.getElementById('turns').textContent    = d.avgTurns;

  const pct = d.timeLimitSecs
    ? Math.min(100, (d.elapsedSecs / d.timeLimitSecs * 100)).toFixed(1)
    : '—';
  document.getElementById('pct').textContent = pct + (d.timeLimitSecs ? '%' : '');
  if (d.timeLimitSecs) document.getElementById('bar').style.width = pct + '%';

  const maxElo = Math.max(...d.bots.map(b=>b.elo), 1600);
  const tbody = document.getElementById('bots');
  tbody.innerHTML = d.bots.map(b => {
    const barW = Math.round((b.elo / maxElo) * 100);
    const badge = DIFF_BADGE[b.difficulty] || 'badge-medium';
    return \`<tr>
      <td><strong>\${b.name}</strong></td>
      <td><div class="elo-bar-wrap">
        <div class="elo-bar" style="width:\${barW}px"></div>
        <span>\${b.elo}</span>
      </div></td>
      <td>\${b.winPct}%</td>
      <td>\${b.top3Pct}%</td>
      <td>\${fmtCash(b.avgCash)}</td>
      <td><span class="badge \${badge}">\${b.difficulty}</span></td>
    </tr>\`;
  }).join('');

  const ago = Math.floor((Date.now() - new Date(d.updatedAt).getTime()) / 1000);
  document.getElementById('updated').textContent = 'Last updated ' + ago + 's ago';
  document.getElementById('status').textContent  = '● Live';
  document.getElementById('status').className    = 'status';
}

// Server-Sent Events for push updates
const es = new EventSource('/events');
es.onmessage = e => { try { render(JSON.parse(e.data)); } catch {} };
es.onerror   = () => {
  document.getElementById('status').textContent = '○ Reconnecting…';
  document.getElementById('status').className   = 'status stale';
};
</script>
</body>
</html>`;

/* ── SSE clients ─────────────────────────────────────────────── */
const clients = new Set();

function readStats() {
  try { return fs.readFileSync(statsFile, 'utf8'); } catch { return null; }
}

function broadcast(data) {
  for (const res of clients) {
    try { res.write(`data: ${data}\n\n`); } catch { clients.delete(res); }
  }
}

// Poll stats file and push to clients every 2 seconds
setInterval(() => {
  const data = readStats();
  if (data) broadcast(data);
}, 2000);

/* ── HTTP server ─────────────────────────────────────────────── */
const server = http.createServer((req, res) => {
  if (req.url === '/events') {
    res.writeHead(200, {
      'Content-Type':  'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection':    'keep-alive',
      'Access-Control-Allow-Origin': '*',
    });
    res.write(': connected\n\n');
    const data = readStats();
    if (data) res.write(`data: ${data}\n\n`);
    clients.add(res);
    req.on('close', () => clients.delete(res));
    return;
  }

  if (req.url === '/stats') {
    const data = readStats() || '{}';
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(data);
    return;
  }

  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end(HTML);
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[dashboard] Live at http://0.0.0.0:${PORT}  (stats: ${statsFile})`);
});
