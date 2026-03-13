/**
 * ai/masterBot.js — Master Bot inference using the trained ONNX value network
 *
 * Loads ai/models/master.onnx and uses it to pick actions.
 * Strategy: for each legal action, clone the game state, apply the action,
 * encode the resulting state, run it through the value network, and pick
 * the action with the highest predicted win probability.
 *
 * This is a one-step lookahead guided by a learned value function — much
 * stronger than random rollouts (MCTS) because the net learned real game
 * outcomes from thousands of training games.
 *
 * Requires: npm install onnxruntime-node
 */

const path   = require('path');
const engine = require('../gameEngine');

const MODEL_PATH = path.join(__dirname, 'models', 'master.onnx');
const INPUT_DIM  = 149;

/* ── ONNX session (lazy-loaded once) ─────────────────────────── */
let _session   = null;
let _ort       = null;
let _loadError = null;

function getSession() {
  if (_session)   return _session;
  if (_loadError) return null;
  try {
    _ort = require('onnxruntime-node');
  } catch {
    _loadError = 'onnxruntime-node not installed — run: npm install onnxruntime-node';
    return null;
  }
  try {
    // Note: ONNX session creation is async but we cache after first await
    // Caller must use getSessionAsync() instead.
    return null; // signal: use async path
  } catch (err) {
    _loadError = err.message;
    return null;
  }
}

let _sessionPromise = null;
async function getSessionAsync() {
  if (_session)   return _session;
  if (_loadError) throw new Error(_loadError);
  if (_sessionPromise) return _sessionPromise;
  _sessionPromise = (async () => {
    try {
      if (!_ort) _ort = require('onnxruntime-node');
    } catch {
      throw new Error('onnxruntime-node not installed — run: npm install onnxruntime-node');
    }
    _session = await _ort.InferenceSession.create(MODEL_PATH);
    return _session;
  })();
  return _sessionPromise;
}

/* ── State encoder (mirrors encodeState in selfplay.js exactly) ── */
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
    myCash:  player.cash / 6000,
    oppCash: game.players.filter((_, i) => i !== playerIdx).map(p => p.cash / 6000),
  };
}

/* ── Flatten encoded state into Float32Array ─────────────────── */
function flattenState(s) {
  const vec = new Float32Array(INPUT_DIM);
  let i = 0;
  for (const v of s.board)          vec[i++] = v / 7.0;
  for (const c of s.chains) {
    vec[i++] = c[0];
    vec[i++] = c[1] / 41.0;
    vec[i++] = c[2] / 25.0;
    vec[i++] = c[3] / 25.0;
    vec[i++] = c[4];
  }
  vec[i++] = s.myCash;
  const opp = (s.oppCash.concat([0,0,0,0,0])).slice(0, 5);
  for (const v of opp) vec[i++] = v;
  return vec;
}

/* ── Run inference on a batch of state vectors ───────────────── */
async function scoreStates(session, stateVecs) {
  const batch = stateVecs.length;
  const flat  = new Float32Array(batch * INPUT_DIM);
  for (let i = 0; i < batch; i++) {
    flat.set(stateVecs[i], i * INPUT_DIM);
  }
  const tensor = new _ort.Tensor('float32', flat, [batch, INPUT_DIM]);
  const output = await session.run({ state: tensor });
  return Array.from(output.win_prob.data); // win probabilities
}

/* ── Clone game state ────────────────────────────────────────── */
function cloneGame(game) {
  return JSON.parse(JSON.stringify(game));
}

/* ── Legal action generators ─────────────────────────────────── */
function legalTileActions(game) {
  const { playable } = engine.getPlayableTiles(game);
  return (playable || []).map(t => ({ type: 'placeTile', tile: t }));
}

function legalBuyActions(game, playerIdx) {
  const player  = game.players[playerIdx];
  const actions = [{ type: 'buyStock', purchases: {} }]; // always include "buy nothing"

  // Generate up to 3 shares across active chains the player can afford
  const affordable = engine.HOTEL_CHAINS.filter(c => {
    const ch = game.chains[c];
    if (!ch.active) return false;
    const price = engine.stockPrice(c, ch.tiles.length);
    return price > 0 && price <= player.cash && (ch.stock_available > 0);
  });

  if (affordable.length === 0) return actions;

  // Enumerate small set of buy combinations (focus on best 1–3 share purchases)
  for (const c of affordable) {
    const price = engine.stockPrice(c, game.chains[c].tiles.length);
    const maxShares = Math.min(3, Math.floor(player.cash / price), game.chains[c].stock_available || 0);
    for (let n = 1; n <= maxShares; n++) {
      actions.push({ type: 'buyStock', purchases: { [c]: n } });
    }
  }
  // Two-chain purchases (most common strong move)
  if (affordable.length >= 2) {
    for (let a = 0; a < affordable.length; a++) {
      for (let b = a + 1; b < affordable.length; b++) {
        const ca = affordable[a], cb = affordable[b];
        const pa = engine.stockPrice(ca, game.chains[ca].tiles.length);
        const pb = engine.stockPrice(cb, game.chains[cb].tiles.length);
        if (pa + pb <= player.cash) {
          actions.push({ type: 'buyStock', purchases: { [ca]: 1, [cb]: 1 } });
        }
        if (pa + pb * 2 <= player.cash) {
          actions.push({ type: 'buyStock', purchases: { [ca]: 1, [cb]: 2 } });
        }
        if (pa * 2 + pb <= player.cash) {
          actions.push({ type: 'buyStock', purchases: { [ca]: 2, [cb]: 1 } });
        }
      }
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
  const maxTrade = Math.min(Math.floor(held / 2) * 2, survivorRoom * 2);
  const options = [
    { type: 'mergerDecision', sell: held, trade: 0 },
    { type: 'mergerDecision', sell: 0,    trade: 0 },
  ];
  if (maxTrade > 0) options.push({ type: 'mergerDecision', sell: held - maxTrade, trade: maxTrade });
  return options;
}

/* ── Apply action to game (same as mcts.js) ──────────────────── */
function applyAction(game, playerIdx, action) {
  try {
    if (action.type === 'placeTile')         engine.placeTile(game, playerIdx, action.tile);
    else if (action.type === 'buyStock')     engine.buyStock(game, playerIdx, action.purchases || {});
    else if (action.type === 'mergerDecision') engine.mergerDecision(game, playerIdx, { sell: action.sell, trade: action.trade });
    else if (action.type === 'chooseChain')  engine.chooseChain(game, playerIdx, action.chain);
    else if (action.type === 'chooseMergerSurvivor') engine.chooseMergerSurvivor(game, playerIdx, action.chain);
  } catch {}
}

/* ── Main entry point ────────────────────────────────────────── */
/**
 * Async. Returns the best action object for the current player,
 * or null if the phase is not handled (caller falls back to heuristic).
 */
async function decideMasterAction(game, playerIdx) {
  const phase = game.phase;
  if (game.currentPlayerIdx !== playerIdx && phase !== 'mergerDecision') return null;

  let actions;
  if      (phase === 'placeTile')      actions = legalTileActions(game);
  else if (phase === 'buyStock')       actions = legalBuyActions(game, playerIdx);
  else if (phase === 'mergerDecision') actions = legalMergerActions(game, playerIdx);
  else                                 return null; // chooseChain / chooseMergerSurvivor handled by heuristic

  if (actions.length === 0) return null;
  if (actions.length === 1) return actions[0];

  let session;
  try {
    session = await getSessionAsync();
  } catch (err) {
    console.error('[masterBot] Model not available:', err.message);
    return null; // fall back to heuristic
  }

  // Score each resulting state
  const stateVecs = [];
  for (const action of actions) {
    const sim = cloneGame(game);
    applyAction(sim, playerIdx, action);
    if (sim.phase === 'gameOver') {
      // Immediate game-over action — check if we won
      const won = sim.standings?.[0]?.name === game.players[playerIdx].name;
      stateVecs.push(new Float32Array(INPUT_DIM).fill(won ? 1 : 0));
    } else {
      try {
        stateVecs.push(flattenState(encodeState(sim, playerIdx)));
      } catch {
        stateVecs.push(new Float32Array(INPUT_DIM).fill(0.5));
      }
    }
  }

  const scores = await scoreStates(session, stateVecs);

  let bestIdx = 0, bestScore = -1;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > bestScore) { bestScore = scores[i]; bestIdx = i; }
  }

  return actions[bestIdx];
}

module.exports = { decideMasterAction };
