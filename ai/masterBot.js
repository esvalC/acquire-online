/**
 * ai/masterBot.js — Master Bot inference
 *
 * Supports weight format v3 (AlphaZero-style):
 *   { version: 3, body: [...], policyHead: [...], valueHead: [...] }
 *
 * Tile placement: use policy head logits directly (fastest — no cloning).
 *   The policy head has learned which tiles are strategically strong.
 *
 * Other phases (buyStock, mergerDecision): value-head guided 1-step lookahead.
 *   Clone game, apply action, evaluate with value head, pick best.
 */

const path   = require('path');
const fs     = require('fs');
const { exec } = require('child_process');
const engine = require('../gameEngine');

const WEIGHTS_PATH     = path.join(__dirname, 'models', 'master_weights.json');
const INPUT_DIM        = 258;  // 108 board + 108 tile hand + 35 chains + 1 myCash + 4 oppCash + 1 bagCount
const POLICY_DIM       = 108; // 9×12 board
const VALUE_DIM        = 5;   // one per player
const BAG_TOTAL        = 102;
const S3_BUCKET        = process.env.S3_BUCKET || 'acquire-training-data';
const S3_KEY           = 'master_weights.json';
const TMP_PATH         = '/tmp/master_weights_s3.json';
const POLL_INTERVAL_MS = 10 * 60 * 1000;

/* ── Weight loading ───────────────────────────────────────────── */
let _weights   = null;
let _loadError = null;

function loadWeights() {
  if (_weights)   return _weights;
  if (_loadError) return null;
  try {
    const raw = fs.readFileSync(WEIGHTS_PATH, 'utf8');
    const w   = JSON.parse(raw);
    if (!w.version || w.version < 3) {
      _loadError = 'Old weight format (pre-v3) — retrain with new AlphaZero architecture';
      console.warn('[masterBot]', _loadError);
      return null;
    }
    _weights = w;
    console.log('[masterBot] v3 weights loaded from disk.');
    return _weights;
  } catch (err) {
    _loadError = 'master_weights.json not found — train first';
    return null;
  }
}

/* ── S3 hot-reload ────────────────────────────────────────────── */
function tryReloadFromS3() {
  exec(`aws s3 cp s3://${S3_BUCKET}/${S3_KEY} "${TMP_PATH}" 2>/dev/null`, (err) => {
    if (err) return;
    try {
      const raw       = fs.readFileSync(TMP_PATH, 'utf8');
      const candidate = JSON.parse(raw);
      if (!candidate.version || candidate.version < 3) return;
      if (!candidate.body || !candidate.policyHead || !candidate.valueHead) return;
      _weights   = candidate;
      _loadError = null;
      fs.mkdirSync(path.dirname(WEIGHTS_PATH), { recursive: true });
      fs.copyFileSync(TMP_PATH, WEIGHTS_PATH);
      console.log(`[masterBot] Hot-reloaded v3 weights from S3 (${new Date().toISOString()})`);
    } catch {}
  });
}

setInterval(tryReloadFromS3, POLL_INTERVAL_MS);
setTimeout(tryReloadFromS3, 5000);

/* ── Activations ──────────────────────────────────────────────── */
function relu(x)    { return x > 0 ? x : 0; }
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x)))); }

/* ── Forward through a list of FC layers ─────────────────────── */
function forwardLayers(layers, input, lastActivation) {
  let x = input;
  for (let li = 0; li < layers.length; li++) {
    const { W, b } = layers[li];
    const isLast   = li === layers.length - 1;
    const out      = new Float32Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let sum = b[j];
      const Wj = W[j];
      for (let i = 0; i < x.length; i++) sum += Wj[i] * x[i];
      if (isLast) {
        out[j] = lastActivation === 'sigmoid' ? sigmoid(sum)
               : lastActivation === 'relu'    ? relu(sum)
               :                                sum; // linear
      } else {
        out[j] = relu(sum);
      }
    }
    x = out;
  }
  return x;
}

/* ── Full forward pass ────────────────────────────────────────── */
function forward(weights, input) {
  const h           = forwardLayers(weights.body,       input, 'relu');
  const policyLogits = forwardLayers(weights.policyHead, h,    'linear');
  const value        = forwardLayers(weights.valueHead,  h,    'sigmoid');
  return { h, policyLogits, value };
}

/* ── State encoder ────────────────────────────────────────────── */
// Input layout (258 total):
//   [0..107]   board state (108): board[r][c] value / 7.0
//   [108..215] tile hand (108):   1 if player holds that tile, else 0
//   [216..250] chain features (35): 7 chains × 5 features each
//   [251]      myCash
//   [252..255] oppCash (4)
//   [256]      bagCount
function encodeState(game, playerIdx) {
  const player    = game.players[playerIdx];
  const CHAIN_IDX = {};
  engine.HOTEL_CHAINS.forEach((c, i) => { CHAIN_IDX[c] = i + 1; });

  // Build a Set of tile strings in the player's hand for fast lookup
  const handSet = new Set(player.tiles || []);

  const vec = new Float32Array(INPUT_DIM);
  let vi = 0;

  // 1. Board state: 108 features
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 12; c++) {
      const cell = game.board[r][c];
      vec[vi++] = (cell && CHAIN_IDX[cell] ? CHAIN_IDX[cell] : (cell ? -1 : 0)) / 7.0;
    }
  }

  // 2. Player's tile hand: 108 features (1 if player holds that tile)
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 12; c++) {
      const tileStr = `${r + 1}${String.fromCharCode(65 + c)}`;
      vec[vi++] = handSet.has(tileStr) ? 1 : 0;
    }
  }

  // 3. Chain features: 7 × 5 = 35
  for (const ch of engine.HOTEL_CHAINS) {
    const info = game.chains[ch];
    const size = info.tiles.length;
    vec[vi++] = info.active ? 1 : 0;
    vec[vi++] = size / 41.0;
    vec[vi++] = (player.stocks[ch] || 0) / 25.0;
    vec[vi++] = game.players.filter((_, i) => i !== playerIdx)
                  .reduce((m, p) => Math.max(m, p.stocks[ch] || 0), 0) / 25.0;
    vec[vi++] = engine.stockPrice(ch, size) / 1000.0;
  }

  // 4. myCash: 1
  vec[vi++] = player.cash / 6000.0;

  // 5. oppCash: 4
  const opps = game.players.filter((_, i) => i !== playerIdx);
  for (let k = 0; k < 4; k++) vec[vi++] = (opps[k] ? opps[k].cash / 6000.0 : 0);

  // 6. bagCount: 1
  vec[vi++] = (game.tileBag ? game.tileBag.length : 0) / BAG_TOTAL;

  return vec;
}

/* ── Tile → policy index ──────────────────────────────────────── */
function tileToIdx(tile) {
  const row = parseInt(tile) - 1;
  const col = tile.charCodeAt(tile.length - 1) - 65;
  return row * 12 + col;
}

/* ── Clone game (strips log for speed) ────────────────────────── */
function cloneGame(game) {
  const savedLog = game.log;
  game.log = [];
  const clone = JSON.parse(JSON.stringify(game));
  game.log = savedLog;
  clone.log = [];
  return clone;
}

/* ── Apply action (for value-guided lookahead) ────────────────── */
function applyAction(game, playerIdx, action) {
  try {
    if      (action.type === 'buyStock')             engine.buyStock(game, playerIdx, action.purchases || {});
    else if (action.type === 'mergerDecision')        engine.mergerDecision(game, playerIdx, { sell: action.sell, trade: action.trade });
    else if (action.type === 'chooseChain')           engine.chooseChain(game, playerIdx, action.chain);
    else if (action.type === 'chooseMergerSurvivor')  engine.chooseMergerSurvivor(game, playerIdx, action.chain);
  } catch {}
}

/* ── Action generators ────────────────────────────────────────── */
function legalBuyActions(game, playerIdx) {
  const player  = game.players[playerIdx];
  const affordable = engine.HOTEL_CHAINS.filter(c => {
    const ch = game.chains[c];
    const issued = game.players.reduce((s, p) => s + (p.stocks[c] || 0), 0);
    return ch.active && engine.stockPrice(c, ch.tiles.length) <= player.cash && issued < 25;
  });
  if (affordable.length === 0) return [{ type: 'buyStock', purchases: {} }];

  // Filter to chains where buying 1 share puts us in 1st or 2nd place.
  // No point fighting for a chain where 2+ opponents are already well ahead —
  // merger bonuses only go to top 2 shareholders.
  function positionAfterBuying(chain) {
    const myNew = (player.stocks[chain] || 0) + 1;
    const aheadCount = game.players.filter((p, i) => i !== playerIdx && (p.stocks[chain] || 0) > myNew).length;
    return aheadCount; // 0 = 1st, 1 = 2nd, 2+ = not in top 2
  }
  const contestable = affordable.filter(c => positionAfterBuying(c) < 2);
  // If everything is too contested, still buy — just into the least contested chain
  // so the bot never sits on cash when it could be building position.
  const targets = contestable.length > 0
    ? contestable
    : affordable.slice().sort((a, b) => positionAfterBuying(a) - positionAfterBuying(b)).slice(0, 1);

  const actions = [];
  for (const c of targets) {
    const price    = engine.stockPrice(c, game.chains[c].tiles.length);
    const issued   = game.players.reduce((s, p) => s + (p.stocks[c] || 0), 0);
    const maxN     = Math.min(3, Math.floor(player.cash / price), 25 - issued);
    for (let n = 1; n <= maxN; n++) actions.push({ type: 'buyStock', purchases: { [c]: n } });
  }
  for (let a = 0; a < targets.length; a++) {
    for (let b = a + 1; b < targets.length; b++) {
      const ca = targets[a], cb = targets[b];
      const pa = engine.stockPrice(ca, game.chains[ca].tiles.length);
      const pb = engine.stockPrice(cb, game.chains[cb].tiles.length);
      if (pa + pb <= player.cash)   actions.push({ type: 'buyStock', purchases: { [ca]: 1, [cb]: 1 } });
      if (pa + 2*pb <= player.cash) actions.push({ type: 'buyStock', purchases: { [ca]: 1, [cb]: 2 } });
      if (2*pa + pb <= player.cash) actions.push({ type: 'buyStock', purchases: { [ca]: 2, [cb]: 1 } });
    }
  }
  // 3-chain combos: buy 1 of each (total 3, only valid combo across 3 chains)
  for (let a = 0; a < targets.length; a++) {
    for (let b = a + 1; b < targets.length; b++) {
      for (let c = b + 1; c < targets.length; c++) {
        const ca = targets[a], cb = targets[b], cc = targets[c];
        const pa = engine.stockPrice(ca, game.chains[ca].tiles.length);
        const pb = engine.stockPrice(cb, game.chains[cb].tiles.length);
        const pc = engine.stockPrice(cc, game.chains[cc].tiles.length);
        if (pa + pb + pc <= player.cash)
          actions.push({ type: 'buyStock', purchases: { [ca]: 1, [cb]: 1, [cc]: 1 } });
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
  const maxTrade     = Math.min(Math.floor(held / 2) * 2, survivorRoom * 2);

  // Sparse representative set — exhaustive combos can be 50–100+ with many shares,
  // which blocks the event loop. ~8 anchor points cover all strategic axes.
  const seen = new Map();
  const add = (sell, trade) => {
    trade = Math.min(trade, maxTrade);
    sell  = Math.max(0, Math.min(sell, held - trade));
    if (sell + trade > held) return;
    const key = `${sell},${trade}`;
    if (!seen.has(key)) seen.set(key, { type: 'mergerDecision', sell, trade });
  };

  add(held, 0);                                       // sell all
  add(0, 0);                                          // hold all
  if (maxTrade > 0) {
    const qTrade = Math.floor(maxTrade / 4) * 2;
    add(held - maxTrade, maxTrade);                   // trade max, sell remainder
    add(0,               maxTrade);                   // trade max, keep remainder
    if (qTrade > 0) {
      add(held - qTrade, qTrade);                     // trade quarter, sell remainder
      add(0,             qTrade);                     // trade quarter, keep remainder
    }
  }
  add(Math.floor(held / 2), 0);                       // sell half, keep half
  add(Math.ceil(held / 2),  0);                       // sell most, keep little

  return [...seen.values()];
}

/* ── Value-guided action selection (for non-tile phases) ──────── */
function bestValueAction(weights, game, playerIdx, actions) {
  if (actions.length === 1) return actions[0];
  const myName = game.players[playerIdx].name;
  let bestAction = actions[0], bestScore = -1;
  for (const action of actions) {
    const sim = cloneGame(game);
    applyAction(sim, playerIdx, action);
    let score;
    if (sim.phase === 'gameOver') {
      score = sim.standings?.[0]?.name === myName ? 1 : 0;
    } else {
      try {
        const { value } = forward(weights, encodeState(sim, playerIdx));
        score = value[playerIdx]; // this player's win probability in sim
      } catch { score = 0.5; }
    }
    if (score > bestScore) { bestScore = score; bestAction = action; }
  }
  return bestAction;
}

/* ── Main entry point ─────────────────────────────────────────── */
/**
 * Returns the best action object, or null to fall through to heuristic bot.
 */
function decideMasterAction(game, playerIdx) {
  const phase = game.phase;
  if (game.currentPlayerIdx !== playerIdx && phase !== 'mergerDecision') return null;

  const weights = loadWeights();
  if (!weights) return null;

  try {
    // ── Tile placement: value-guided lookahead (resolves chooseChain inline) ── */
    if (phase === 'placeTile') {
      const { playable } = engine.getPlayableTiles(game);
      if (!playable || playable.length === 0) return null; // botAI heuristic handles passTile
      if (playable.length === 1) return { type: 'placeTile', tile: playable[0] };

      const myName = game.players[playerIdx].name;
      let best = playable[0], bestScore = -Infinity;
      for (const tile of playable) {
        const sim = cloneGame(game);
        try {
          engine.placeTile(sim, playerIdx, tile);
          // If placing the tile triggers chooseChain, resolve it now so the
          // value head evaluates the fully-settled board state.
          if (sim.phase === 'chooseChain') {
            const avail = engine.HOTEL_CHAINS.filter(c => !sim.chains[c].active);
            if (avail.length === 1) {
              engine.chooseChain(sim, playerIdx, avail[0]);
            } else if (avail.length > 1) {
              const chainBest = bestValueAction(weights, sim, playerIdx,
                avail.map(c => ({ type: 'chooseChain', chain: c })));
              engine.chooseChain(sim, playerIdx, chainBest.chain);
            }
          }
        } catch { continue; }

        let score;
        if (sim.phase === 'gameOver') {
          score = sim.standings?.[0]?.name === myName ? 1 : 0;
        } else {
          try {
            const { value } = forward(weights, encodeState(sim, playerIdx));
            score = value[playerIdx];
          } catch { score = 0.5; }
        }
        if (score > bestScore) { bestScore = score; best = tile; }
      }
      return { type: 'placeTile', tile: best };
    }

    // ── Choose new chain: value-guided lookahead ─────────────── */
    if (phase === 'chooseChain') {
      const available = engine.HOTEL_CHAINS.filter(c => !game.chains[c].active);
      if (available.length === 0) return null;
      if (available.length === 1) return { type: 'chooseChain', chain: available[0] };
      const actions = available.map(c => ({ type: 'chooseChain', chain: c }));
      return bestValueAction(weights, game, playerIdx, actions);
    }

    // ── Choose merger survivor: value-guided lookahead ────────── */
    if (phase === 'chooseMergerSurvivor') {
      const tied = game.pendingMerger?.tiedChains || [];
      if (tied.length === 0) return null;
      if (tied.length === 1) return { type: 'chooseMergerSurvivor', chain: tied[0] };
      const actions = tied.map(c => ({ type: 'chooseMergerSurvivor', chain: c }));
      return bestValueAction(weights, game, playerIdx, actions);
    }

    // ── Buy stock: value-guided lookahead ────────────────────── */
    if (phase === 'buyStock') {
      const actions = legalBuyActions(game, playerIdx);
      return bestValueAction(weights, game, playerIdx, actions);
    }

    // ── Merger decision: value-guided lookahead ───────────────── */
    if (phase === 'mergerDecision') {
      if (game.pendingMerger?.decidingPlayer !== playerIdx) return null;
      const actions = legalMergerActions(game, playerIdx);
      return bestValueAction(weights, game, playerIdx, actions);
    }
  } catch (e) {
    // Fall through to heuristic
  }
  return null;
}

module.exports = { decideMasterAction };
