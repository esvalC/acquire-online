/**
 * ai/masterBot.js — Master Bot inference (pure JavaScript, zero dependencies)
 *
 * Loads the trained weights from ai/models/master_weights.json and runs
 * forward inference on the value network (MLP: 149 → 256 → 128 → 64 → 1).
 *
 * Strategy: one-step lookahead guided by a learned value function.
 * For each legal action, clone the game, apply the action, encode the
 * resulting state, score it with the network, pick the highest.
 *
 * How to get the model:
 *   1. Run selfplay to generate data:
 *        node ai/selfplay.js --time-limit 18000 --export ai/data/games.jsonl
 *   2. Train on Mac (or any machine with Python + PyTorch):
 *        python ai/train.py --data ai/data/games.jsonl --out ai/models/master_weights.json
 *   3. Commit ai/models/master_weights.json (it's ~1MB)
 *   4. Deploy — masterBot.js loads it automatically on first use
 */

const path = require('path');
const fs   = require('fs');
const { exec } = require('child_process');
const engine = require('../gameEngine');

const WEIGHTS_PATH = path.join(__dirname, 'models', 'master_weights.json');
const INPUT_DIM    = 150; // 108 board + 35 chains + 1 myCash + 4 oppCash + 1 bagCount
const BAG_TOTAL    = 102;
const S3_BUCKET    = process.env.S3_BUCKET || 'acquire-training-data';
const S3_KEY       = 'master_weights.json';
const TMP_PATH     = '/tmp/master_weights_s3.json';
const POLL_INTERVAL_MS = 10 * 60 * 1000; // check S3 every 10 minutes

/* ── Weight loading (sync, done once at startup) ─────────────── */
let _weights    = null;
let _loadError  = null;

function loadWeights() {
  if (_weights)   return _weights;
  if (_loadError) return null;
  try {
    const raw = fs.readFileSync(WEIGHTS_PATH, 'utf8');
    _weights = JSON.parse(raw);
    console.log('[masterBot] Weights loaded from disk.');
    return _weights;
  } catch (err) {
    _loadError = `master_weights.json not found — train the model first`;
    return null;
  }
}

/* ── S3 hot-reload — pulls updated weights without restarting ── */
function tryReloadFromS3() {
  exec(`aws s3 cp s3://${S3_BUCKET}/${S3_KEY} "${TMP_PATH}" 2>/dev/null`, (err) => {
    if (err) return; // S3 not accessible (no IAM role, or no weights yet) — silent no-op
    try {
      const raw = fs.readFileSync(TMP_PATH, 'utf8');
      const candidate = JSON.parse(raw);
      if (!candidate.layers || candidate.layers.length === 0) return;
      // Sanity check: input dim must match
      if (candidate.layers[0].W[0].length !== INPUT_DIM) {
        console.warn('[masterBot] S3 weights have wrong input dim — skipping.');
        return;
      }
      _weights   = candidate;
      _loadError = null;
      // Also persist locally so it survives next server restart
      fs.mkdirSync(path.dirname(WEIGHTS_PATH), { recursive: true });
      fs.copyFileSync(TMP_PATH, WEIGHTS_PATH);
      console.log(`[masterBot] Hot-reloaded weights from S3 (${new Date().toISOString()})`);
    } catch {}
  });
}

// Poll S3 every 10 minutes — runs silently in the background
setInterval(tryReloadFromS3, POLL_INTERVAL_MS);
// Also check once shortly after startup (weights may already be in S3)
setTimeout(tryReloadFromS3, 5000);

/* ── Pure-JS MLP forward pass ────────────────────────────────── */
// weights format: { layers: [ {W: [[...]], b: [...]}, ... ] }
// Activations: ReLU on hidden layers, sigmoid on output

function relu(x) { return x > 0 ? x : 0; }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function forward(weights, input) {
  let x = input;
  const layers = weights.layers;
  for (let li = 0; li < layers.length; li++) {
    const { W, b } = layers[li];
    const out = new Float32Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let sum = b[j];
      const Wj = W[j];
      for (let i = 0; i < x.length; i++) sum += Wj[i] * x[i];
      out[j] = li < layers.length - 1 ? relu(sum) : sigmoid(sum);
    }
    x = out;
  }
  return x[0]; // scalar win probability
}

/* ── State encoder (mirrors selfplay.js / train.py exactly) ─── */
function encodeState(game, playerIdx) {
  const player = game.players[playerIdx];
  const CHAIN_IDX = {};
  engine.HOTEL_CHAINS.forEach((c, i) => { CHAIN_IDX[c] = i + 1; });

  const board = [];
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 12; c++) {
      const cell = game.board[r][c];
      board.push(cell && CHAIN_IDX[cell] ? CHAIN_IDX[cell] : (cell ? -1 : 0));
    }
  }

  const chains = engine.HOTEL_CHAINS.map(c => {
    const ch   = game.chains[c];
    const size = ch.tiles.length;
    return [
      ch.active ? 1 : 0,
      size,
      player.stocks[c] || 0,
      game.players.filter((_, i) => i !== playerIdx)
        .reduce((m, p) => Math.max(m, p.stocks[c] || 0), 0),
      engine.stockPrice(c, size) / 1000,
    ];
  });

  return {
    board,
    chains,
    myCash:    player.cash / 6000,
    oppCash:   game.players.filter((_, i) => i !== playerIdx).map(p => p.cash / 6000),
    bagCount:  (game.tileBag ? game.tileBag.length : 0) / BAG_TOTAL,
  };
}

function flattenState(s) {
  const vec = new Float32Array(INPUT_DIM);
  let i = 0;
  for (const v of s.board)   vec[i++] = v / 7.0;
  for (const c of s.chains) {
    vec[i++] = c[0];
    vec[i++] = c[1] / 41.0;
    vec[i++] = c[2] / 25.0;
    vec[i++] = c[3] / 25.0;
    vec[i++] = c[4];
  }
  vec[i++] = s.myCash;
  const opp = (s.oppCash.concat([0,0,0,0,0])).slice(0, 4);
  for (const v of opp) vec[i++] = v;
  vec[i++] = s.bagCount ?? 1.0; // 1.0 = full bag (early game), 0.0 = empty (late game)
  return vec;
}

/* ── Score a state ───────────────────────────────────────────── */
function scoreState(weights, game, playerIdx) {
  try {
    return forward(weights, flattenState(encodeState(game, playerIdx)));
  } catch {
    return 0.5;
  }
}

/* ── Legal action generators ─────────────────────────────────── */
function legalTileActions(game) {
  const { playable } = engine.getPlayableTiles(game);
  return (playable || []).map(t => ({ type: 'placeTile', tile: t }));
}

function legalBuyActions(game, playerIdx) {
  const player  = game.players[playerIdx];
  const actions = [{ type: 'buyStock', purchases: {} }];
  const affordable = engine.HOTEL_CHAINS.filter(c => {
    const ch = game.chains[c];
    if (!ch.active) return false;
    return engine.stockPrice(c, ch.tiles.length) <= player.cash && (ch.stock_available > 0);
  });
  if (affordable.length === 0) return actions;
  for (const c of affordable) {
    const price = engine.stockPrice(c, game.chains[c].tiles.length);
    const maxN  = Math.min(3, Math.floor(player.cash / price), game.chains[c].stock_available || 0);
    for (let n = 1; n <= maxN; n++) actions.push({ type: 'buyStock', purchases: { [c]: n } });
  }
  for (let a = 0; a < affordable.length; a++) {
    for (let b = a + 1; b < affordable.length; b++) {
      const ca = affordable[a], cb = affordable[b];
      const pa = engine.stockPrice(ca, game.chains[ca].tiles.length);
      const pb = engine.stockPrice(cb, game.chains[cb].tiles.length);
      if (pa + pb <= player.cash)     actions.push({ type: 'buyStock', purchases: { [ca]: 1, [cb]: 1 } });
      if (pa + 2*pb <= player.cash)   actions.push({ type: 'buyStock', purchases: { [ca]: 1, [cb]: 2 } });
      if (2*pa + pb <= player.cash)   actions.push({ type: 'buyStock', purchases: { [ca]: 2, [cb]: 1 } });
    }
  }
  return actions;
}

function legalMergerActions(game, playerIdx) {
  const pm      = game.pendingMerger;
  const defunct = pm.defunctChains[pm.currentDefunctIdx];
  const held    = game.players[playerIdx].stocks[defunct] || 0;
  if (held === 0) return [{ type: 'mergerDecision', sell: 0, trade: 0 }];
  const survivorRoom = 25 - game.players.reduce((s, p) => s + (p.stocks[pm.survivor] || 0), 0);
  const maxTrade     = Math.min(Math.floor(held / 2) * 2, survivorRoom * 2);
  const opts = [
    { type: 'mergerDecision', sell: held, trade: 0 },
    { type: 'mergerDecision', sell: 0,    trade: 0 },
  ];
  if (maxTrade > 0) opts.push({ type: 'mergerDecision', sell: held - maxTrade, trade: maxTrade });
  return opts;
}

/* ── Apply action ────────────────────────────────────────────── */
function applyAction(game, playerIdx, action) {
  try {
    if (action.type === 'placeTile')             engine.placeTile(game, playerIdx, action.tile);
    else if (action.type === 'buyStock')         engine.buyStock(game, playerIdx, action.purchases || {});
    else if (action.type === 'mergerDecision')   engine.mergerDecision(game, playerIdx, { sell: action.sell, trade: action.trade });
    else if (action.type === 'chooseChain')      engine.chooseChain(game, playerIdx, action.chain);
    else if (action.type === 'chooseMergerSurvivor') engine.chooseMergerSurvivor(game, playerIdx, action.chain);
  } catch {}
}

/* ── Clone ───────────────────────────────────────────────────── */
function cloneGame(game) { return JSON.parse(JSON.stringify(game)); }

/* ── Main entry (synchronous — no async needed without ONNX) ─── */
/**
 * Returns the best action object, or null if phase not handled
 * (caller should fall through to heuristic).
 */
function decideMasterAction(game, playerIdx) {
  const phase = game.phase;
  if (game.currentPlayerIdx !== playerIdx && phase !== 'mergerDecision') return null;

  const weights = loadWeights();
  if (!weights) return null; // model not trained yet — fall back to heuristic

  let actions;
  if      (phase === 'placeTile')      actions = legalTileActions(game);
  else if (phase === 'buyStock')       actions = legalBuyActions(game, playerIdx);
  else if (phase === 'mergerDecision') actions = legalMergerActions(game, playerIdx);
  else                                 return null;

  if (actions.length === 0) return null;
  if (actions.length === 1) return actions[0];

  let bestAction = actions[0], bestScore = -1;
  const myName = game.players[playerIdx].name;

  for (const action of actions) {
    const sim = cloneGame(game);
    applyAction(sim, playerIdx, action);
    let score;
    if (sim.phase === 'gameOver') {
      score = sim.standings?.[0]?.name === myName ? 1 : 0;
    } else {
      score = scoreState(weights, sim, playerIdx);
    }
    if (score > bestScore) { bestScore = score; bestAction = action; }
  }

  return bestAction;
}

module.exports = { decideMasterAction };
