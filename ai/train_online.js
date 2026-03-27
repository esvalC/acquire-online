#!/usr/bin/env node
/**
 * ai/train_online.js — AlphaZero-style training for the Master Bot
 *
 * Architecture: shared body + DUAL HEADS
 *   Input (258) → FC(256)→ReLU → FC(128)→ReLU → FC(64)→ReLU → h
 *   h → policy head: FC(108) → tile logits (softmax during training)
 *   h → value  head: FC(5)   → per-player win probabilities (sigmoid)
 *
 * Why dual heads beat single-value:
 *   The policy head provides position-specific gradient signal to the body
 *   even before the value head is useful. MCTS visit counts (AlphaZero-style)
 *   provide the policy targets — diverse, per-position gradients that break
 *   the "predict-the-mean" plateau.
 *
 *   The N-vector value head (one output per player) provides 5× more value
 *   gradient per game than a single-scalar head.
 *
 * Usage:
 *   node ai/train_online.js                         # train until Ctrl-C
 *   node ai/train_online.js --time-limit 86400      # train for 24 hours
 *   node ai/train_online.js --lr 0.0003             # custom learning rate
 */

'use strict';

const engine            = require('../gameEngine');
const { decideBotAction } = require('../botAI');
const fs                = require('fs');
const path              = require('path');
const { execSync }      = require('child_process');

/* ── CLI args ─────────────────────────────────────────────────── */
function getArg(flag, def) {
  const idx = process.argv.indexOf(flag);
  return idx === -1 ? def : process.argv[idx + 1];
}

/* ── Config ───────────────────────────────────────────────────── */
const BATCH_SIZE   = 256;     // samples per gradient step (from replay buffer)
const TRAIN_EVERY  = 100;     // games between gradient steps
const SAVE_EVERY   = 1000;    // games between weight saves to disk+S3
const REPLAY_SIZE  = 20000;   // max entries in replay buffer
const LR           = parseFloat(getArg('--lr', '0.0003'));
const TIME_LIMIT   = getArg('--time-limit') ? parseInt(getArg('--time-limit')) * 1000 : Infinity;
const STATS_FILE   = getArg('--stats') || null;
const LOG_PATH     = getArg('--log') || null;
const MAX_LOG_KB   = 256;

const WEIGHTS_PATH = path.join(__dirname, 'models', 'master_weights.json');
const S3_BUCKET    = process.env.S3_BUCKET || 'acquire-training-data';
const S3_KEY       = 'master_weights.json';

/* ── Architecture constants ───────────────────────────────────── */
const INPUT_DIM   = 258;  // 108 board + 108 tile hand + 35 chain features + 1 myCash + 4 oppCash + 1 tileBag
const HIDDEN      = [256, 128, 64];
const POLICY_DIM  = 108;  // 9 rows × 12 cols (one logit per board cell)
const VALUE_DIM   = 5;    // one sigmoid per player position
const BAG_TOTAL   = 102;

// MCTS constants (AlphaZero-style tile selection)
const MCTS_SIMS = 20;  // simulations per tile decision
const C_PUCT    = 1.5; // PUCT exploration constant

// Temperature for policy soft targets (kept for reference; not used in MCTS path)
const POLICY_TEMP = 0.5;

// Exploration rate: random action with this probability (all slots)
const EPS = 0.10;

// Target cash for absolute reward component
// Rank-based value targets: win-focused, steep drop-off.
// Bot learns to WIN, not just accumulate cash.
const RANK_SCORES = [1.0, 0.5, 0.25, 0.1, 0.0]; // 1st → 5th place
function rankScore(standings, myName) {
  const sorted = [...standings].sort((a, b) => b.cash - a.cash);
  const rank   = sorted.findIndex(p => p.name === myName);
  return RANK_SCORES[rank] ?? 0.0;
}

/* ── Bot roster ───────────────────────────────────────────────── */
const BOTS = [
  { name: 'Aria', personality: 'balanced',    difficulty: 'hard' },
  { name: 'Rex',  personality: 'focused',     difficulty: 'hard' },
  { name: 'Nova', personality: 'diversified', difficulty: 'hard' },
  { name: 'Colt', personality: 'focused',     difficulty: 'hard' },
  { name: 'Vera', personality: 'balanced',    difficulty: 'hard' },
];

/* ── ELO tracking (matches selfplay.js / dashboard expectations) ── */
const ELO_K = 32;
function eloExpected(ra, rb) { return 1 / (1 + Math.pow(10, (rb - ra) / 400)); }
function updateElo(ratings, standings) {
  for (let i = 0; i < standings.length; i++) {
    for (let j = i + 1; j < standings.length; j++) {
      const w = standings[i].name, l = standings[j].name;
      if (!ratings[w] || !ratings[l]) continue;
      const rw = ratings[w], rl = ratings[l];
      const exp = eloExpected(rw, rl);
      ratings[w] = Math.round(rw + ELO_K * (1 - exp));
      ratings[l] = Math.round(rl + ELO_K * (0 - (1 - exp)));
    }
  }
}

/* ── Logging ──────────────────────────────────────────────────── */
function log(msg) {
  process.stdout.write(msg);
  if (!LOG_PATH) return;
  try {
    fs.appendFileSync(LOG_PATH, msg);
    const stat = fs.statSync(LOG_PATH);
    if (stat.size > MAX_LOG_KB * 1024) {
      const content = fs.readFileSync(LOG_PATH, 'utf8');
      const lines   = content.split('\n');
      fs.writeFileSync(LOG_PATH, lines.slice(Math.floor(lines.length / 2)).join('\n'));
    }
  } catch {}
}

/* ── Weight format v3: body + policyHead + valueHead ─────────── */
function initWeights() {
  function heLayer(fanIn, fanOut) {
    const scale = Math.sqrt(2.0 / fanIn);
    return {
      W: Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, () => (Math.random() * 2 - 1) * scale)),
      b: new Array(fanOut).fill(0),
    };
  }

  const body = [];
  for (let i = 0; i < HIDDEN.length; i++) {
    const fanIn  = i === 0 ? INPUT_DIM : HIDDEN[i - 1];
    body.push(heLayer(fanIn, HIDDEN[i]));
  }
  const bodyOut  = HIDDEN[HIDDEN.length - 1]; // 64

  return {
    version:    3,
    body,
    policyHead: [heLayer(bodyOut, POLICY_DIM)],
    valueHead:  [heLayer(bodyOut, VALUE_DIM)],
  };
}

function loadOrInit() {
  // 1. Try local file first
  try {
    const raw = fs.readFileSync(WEIGHTS_PATH, 'utf8');
    const w   = JSON.parse(raw);
    if (!w.version || w.version < 3) {
      log('  Old weight format detected — will try S3 before reinitializing.\n');
    } else {
      log(`  Loaded v3 weights from ${WEIGHTS_PATH}\n`);
      return w;
    }
  } catch { /* no local file */ }

  // 2. Try S3 (fresh instance after spot termination — resume from last checkpoint)
  try {
    log(`  No local weights — trying S3 s3://${S3_BUCKET}/${S3_KEY} ...\n`);
    execSync(`aws s3 cp s3://${S3_BUCKET}/${S3_KEY} "${WEIGHTS_PATH}"`, { timeout: 30000, stdio: 'pipe' });
    const raw = fs.readFileSync(WEIGHTS_PATH, 'utf8');
    const w   = JSON.parse(raw);
    if (w.version >= 3) {
      log(`  Resumed v3 weights from S3 — continuing from previous run.\n`);
      return w;
    }
  } catch { /* S3 not available or no valid weights */ }

  // 3. Fresh start
  log('  Initializing fresh v3 weights (policy + value heads).\n');
  return initWeights();
}

function saveWeights(weights) {
  fs.mkdirSync(path.dirname(WEIGHTS_PATH), { recursive: true });
  fs.writeFileSync(WEIGHTS_PATH, JSON.stringify(weights));
  try {
    execSync(`aws s3 cp "${WEIGHTS_PATH}" s3://${S3_BUCKET}/${S3_KEY}`, { timeout: 30000, stdio: 'pipe' });
    log(`  [s3] uploaded → s3://${S3_BUCKET}/${S3_KEY}\n`);
  } catch (e) {
    log(`  [s3] upload failed: ${e.message}\n`);
  }
}

/* ── Adam optimizer ───────────────────────────────────────────── */
function initAdam(weights) {
  function adamForLayers(layers) {
    return layers.map(({ W, b }) => ({
      mW: W.map(row  => new Float64Array(row.length)),
      vW: W.map(row  => new Float64Array(row.length)),
      mb: new Float64Array(b.length),
      vb: new Float64Array(b.length),
    }));
  }
  return {
    body:       adamForLayers(weights.body),
    policyHead: adamForLayers(weights.policyHead),
    valueHead:  adamForLayers(weights.valueHead),
  };
}

/* ── Activations ──────────────────────────────────────────────── */
function relu(x)    { return x > 0 ? x : 0; }
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x)))); }

function softmax(arr) {
  let max = -Infinity;
  for (const v of arr) if (v > max) max = v;
  let sum = 0;
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) { out[i] = Math.exp(arr[i] - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/* ── Forward pass through a list of FC layers ─────────────────── */
// Returns { h, acts, zs } where h is the output vector.
// activation: 'relu', 'sigmoid', or 'linear' (for last layer).
function forwardLayers(layers, input, lastActivation) {
  const acts = [input];
  const zs   = [];
  let x = input;
  for (let li = 0; li < layers.length; li++) {
    const { W, b } = layers[li];
    const isLast = li === layers.length - 1;
    const z   = new Float64Array(b.length);
    const out = new Float64Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let sum = b[j];
      const Wj = W[j];
      for (let i = 0; i < x.length; i++) sum += Wj[i] * x[i];
      z[j] = sum;
      if (isLast) {
        out[j] = lastActivation === 'sigmoid' ? sigmoid(sum)
               : lastActivation === 'relu'    ? relu(sum)
               :                                sum; // linear
      } else {
        out[j] = relu(sum);
      }
    }
    zs.push(z);
    acts.push(out);
    x = out;
  }
  return { h: x, acts, zs };
}

/* ── Full forward pass ─────────────────────────────────────────── */
function forward(weights, input) {
  const body   = forwardLayers(weights.body, input, 'relu');
  const h      = body.h;  // 64-dim shared representation
  const policy = forwardLayers(weights.policyHead, h, 'linear');
  const value  = forwardLayers(weights.valueHead,  h, 'sigmoid');
  return {
    h,
    policyLogits: policy.h, // 108 raw logits
    value:        value.h,  // 5 sigmoid outputs
    bodyActs:  body.acts,   bodyZs:  body.zs,
    polActs:   policy.acts, polZs:   policy.zs,
    valActs:   value.acts,  valZs:   value.zs,
  };
}

/* ── Backprop through a list of FC layers ─────────────────────── */
// delta: error signal at the output of these layers (Float64Array)
// Returns: error signal at the INPUT of these layers.
// Accumulates into gradW, gradB.
function backpropLayers(layers, acts, zs, delta, gradW, gradB) {
  for (let li = layers.length - 1; li >= 0; li--) {
    const { W } = layers[li];
    const xPrev = acts[li];
    const outDim = W.length, inDim = W[0].length;

    for (let j = 0; j < outDim; j++) {
      gradB[li][j] += delta[j];
      const Wj = W[j], gWj = gradW[li][j], dj = delta[j];
      for (let i = 0; i < inDim; i++) gWj[i] += dj * xPrev[i];
    }
    if (li === 0) return; // we don't need delta for the input to the body

    const deltaNext = new Float64Array(inDim);
    const zPrev     = zs[li - 1];
    for (let i = 0; i < inDim; i++) {
      let sum = 0;
      for (let j = 0; j < outDim; j++) sum += W[j][i] * delta[j];
      deltaNext[i] = sum * (zPrev[i] > 0 ? 1 : 0); // ReLU grad
    }
    delta = deltaNext;
  }
}

// Returns the error signal that should be propagated into the layer BELOW.
function backpropLayersWithReturn(layers, acts, zs, delta, gradW, gradB) {
  for (let li = layers.length - 1; li >= 0; li--) {
    const { W } = layers[li];
    const xPrev = acts[li];
    const outDim = W.length, inDim = W[0].length;

    for (let j = 0; j < outDim; j++) {
      gradB[li][j] += delta[j];
      const Wj = W[j], gWj = gradW[li][j], dj = delta[j];
      for (let i = 0; i < inDim; i++) gWj[i] += dj * xPrev[i];
    }

    const deltaNext = new Float64Array(inDim);
    // Determine if prev layer uses ReLU (body always does)
    if (li === 0) {
      // No activation before first layer input — propagate raw gradient
      for (let i = 0; i < inDim; i++) {
        let sum = 0;
        for (let j = 0; j < outDim; j++) sum += W[j][i] * delta[j];
        deltaNext[i] = sum;
      }
    } else {
      const zPrev = zs[li - 1];
      for (let i = 0; i < inDim; i++) {
        let sum = 0;
        for (let j = 0; j < outDim; j++) sum += W[j][i] * delta[j];
        deltaNext[i] = sum * (zPrev[i] > 0 ? 1 : 0);
      }
    }
    delta = deltaNext;
  }
  return delta; // error signal at the input to this head (to be summed across heads)
}

/* ── Adam step ────────────────────────────────────────────────── */
function adamStepLayers(layers, adamState, gradW, gradB, t) {
  const β1 = 0.9, β2 = 0.999, ε = 1e-8;
  const bc1 = 1 - Math.pow(β1, t);
  const bc2 = 1 - Math.pow(β2, t);

  for (let li = 0; li < layers.length; li++) {
    const { W, b }      = layers[li];
    const { mW, vW, mb, vb } = adamState[li];
    const gW = gradW[li], gB = gradB[li];

    for (let j = 0; j < W.length; j++) {
      const Wj = W[j], mWj = mW[j], vWj = vW[j], gWj = gW[j];
      for (let i = 0; i < Wj.length; i++) {
        const g = gWj[i];
        mWj[i] = β1 * mWj[i] + (1 - β1) * g;
        vWj[i] = β2 * vWj[i] + (1 - β2) * g * g;
        Wj[i] -= LR * (mWj[i] / bc1) / (Math.sqrt(vWj[i] / bc2) + ε);
      }
      const gj = gB[j];
      mb[j] = β1 * mb[j] + (1 - β1) * gj;
      vb[j] = β2 * vb[j] + (1 - β2) * gj * gj;
      b[j]  -= LR * (mb[j] / bc1) / (Math.sqrt(vb[j] / bc2) + ε);
    }
  }
}

function initGrads(layers) {
  return {
    W: layers.map(({ W }) => W.map(row => new Float64Array(row.length))),
    b: layers.map(({ b }) => new Float64Array(b.length)),
  };
}

function zeroGrads(grads) {
  for (const row of grads.W.flat()) row.fill(0);
  for (const b of grads.b) b.fill(0);
}

/* ── State encoder ────────────────────────────────────────────── */
// Input layout (258 total):
//   [0..107]   board state (108): board[r][c] value / 7.0
//   [108..215] tile hand (108):   1 if player holds that tile, else 0
//   [216..250] chain features (35): 7 chains × 5 features each
//   [251]      myCash
//   [252..255] oppCash (4)
//   [256]      bagCount
function encodeFlat(game, playerIdx) {
  const player    = game.players[playerIdx];
  const CHAIN_IDX = {};
  engine.HOTEL_CHAINS.forEach((c, i) => { CHAIN_IDX[c] = i + 1; });

  // Build a Set of tile strings in the player's hand for fast lookup
  const handSet = new Set(player.tiles || []);

  const vec = new Float64Array(INPUT_DIM);
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

/* ── Tile ↔ policy index ──────────────────────────────────────── */
// Tile format: "1A"–"9L" → row = parseInt(tile)-1, col = last char - 'A'
function tileToIdx(tile) {
  const row = parseInt(tile) - 1;
  const col = tile.charCodeAt(tile.length - 1) - 65;
  return row * 12 + col;
}

/* ── Clone game without log (cheaper JSON clone) ──────────────── */
function cloneForScoring(game) {
  const savedLog = game.log;
  game.log = [];
  const clone = JSON.parse(JSON.stringify(game));
  game.log = savedLog;
  clone.log = [];
  return clone;
}

/* ── Random action ────────────────────────────────────────────── */
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function takeRandomAction(game, playerIdx) {
  const phase = game.phase;
  try {
    if (phase === 'placeTile') {
      const { playable } = engine.getPlayableTiles(game);
      if (playable.length === 0) { engine.passTile(game, playerIdx); return true; }
      engine.placeTile(game, playerIdx, playable[Math.floor(Math.random() * playable.length)]);
      return true;
    }
    if (phase === 'buyStock') {
      const affordable = engine.HOTEL_CHAINS.filter(c => {
        const ch = game.chains[c];
        return ch.active && engine.stockPrice(c, ch.tiles.length) <= game.players[playerIdx].cash;
      });
      shuffle(affordable);
      const purchases = {};
      let buys = 0;
      for (const c of affordable) {
        if (buys >= 3) break;
        const price = engine.stockPrice(c, game.chains[c].tiles.length);
        const n = Math.min(3 - buys, Math.floor(game.players[playerIdx].cash / price),
                           25 - game.players.reduce((s, p) => s + (p.stocks[c] || 0), 0));
        if (n <= 0) continue;
        purchases[c] = 1 + Math.floor(Math.random() * n);
        buys += purchases[c];
      }
      engine.buyStock(game, playerIdx, purchases);
      return true;
    }
    if (phase === 'chooseChain') {
      const available = engine.HOTEL_CHAINS.filter(c => !game.chains[c].active);
      if (available.length) engine.chooseChain(game, playerIdx, available[Math.floor(Math.random() * available.length)]);
      return true;
    }
    if (phase === 'chooseMergerSurvivor') {
      const tied = game.pendingMerger?.tiedChains || [];
      if (tied.length) engine.chooseMergerSurvivor(game, playerIdx, tied[Math.floor(Math.random() * tied.length)]);
      return true;
    }
    if (phase === 'mergerDecision' && game.pendingMerger?.decidingPlayer === playerIdx) {
      const defunct = game.pendingMerger.defunctChains[game.pendingMerger.currentDefunctIdx];
      const held = game.players[playerIdx].stocks[defunct] || 0;
      const r = Math.random();
      engine.mergerDecision(game, playerIdx, r < 0.33
        ? { sell: held, trade: 0 }
        : r < 0.66
          ? { sell: 0, trade: 0 }
          : { sell: 0, trade: Math.min(Math.floor(held / 2) * 2, 25 * 2) });
      return true;
    }
  } catch {}
  return false;
}

/* ── Advance game past sub-phases to next tile placement ──────── */
// Resolves buyStock, chooseChain, chooseMergerSurvivor, mergerDecision
// until the game is in 'placeTile' or 'gameOver'. Uses random actions.
function advanceToNextTilePlacement(game, playerIdx) {
  let iters = 0;
  while (game.phase !== 'placeTile' && game.phase !== 'gameOver' && iters++ < 60) {
    try {
      const phase = game.phase;
      const curr  = game.currentPlayerIdx;
      if (phase === 'buyStock') {
        takeRandomAction(game, curr);
      } else if (phase === 'chooseChain') {
        const avail = engine.HOTEL_CHAINS.filter(c => !game.chains[c].active);
        if (avail.length) engine.chooseChain(game, curr, avail[0]);
        else break;
      } else if (phase === 'chooseMergerSurvivor') {
        const tied = game.pendingMerger?.tiedChains || [];
        if (tied.length) engine.chooseMergerSurvivor(game, curr, tied[0]);
        else break;
      } else if (phase === 'mergerDecision') {
        const dp = game.pendingMerger?.decidingPlayer;
        if (dp === undefined || dp === null) break;
        const defunct = game.pendingMerger.defunctChains[game.pendingMerger.currentDefunctIdx];
        const held = game.players[dp]?.stocks?.[defunct] || 0;
        engine.mergerDecision(game, dp, { sell: held, trade: 0 });
      } else { break; }
    } catch { break; }
  }
}

/* ── MCTS tile selection (AlphaZero-style) ───────────────────── */
// Runs MCTS_SIMS simulations over legal tile choices.
// Returns: { tile: most_visited_tile, policyTarget: 108-dim visit distribution }
function mctsPickTile(game, playerIdx, weights) {
  const { playable } = engine.getPlayableTiles(game);
  if (!playable || playable.length === 0) return { tile: null, policyTarget: null };

  if (playable.length === 1) {
    const pt = new Float32Array(POLICY_DIM);
    pt[tileToIdx(playable[0])] = 1.0;
    return { tile: playable[0], policyTarget: pt };
  }

  // Priors from policy head + base value from value head
  const state = encodeFlat(game, playerIdx);
  const { policyLogits, value } = forward(weights, state);
  const legalIndices = playable.map(tileToIdx);
  const legalLogits  = legalIndices.map(i => policyLogits[i]);
  const priors       = softmax(legalLogits);
  const baseValue    = value[playerIdx]; // default Q for unvisited nodes

  // MCTS statistics per legal tile
  const N  = new Int32Array(playable.length);   // visit counts
  const Wv = new Float64Array(playable.length); // total values

  for (let sim = 0; sim < MCTS_SIMS; sim++) {
    // PUCT selection
    const Ntotal = N.reduce((s, v) => s + v, 0);
    let bestIdx = 0, bestScore = -Infinity;
    for (let i = 0; i < playable.length; i++) {
      const q = N[i] > 0 ? Wv[i] / N[i] : baseValue;
      const u = C_PUCT * priors[i] * Math.sqrt(Ntotal + 1) / (1 + N[i]);
      const score = q + u;
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    }

    // Simulate: place tile, resolve sub-phases, evaluate
    const sim_game = cloneForScoring(game);
    try { engine.placeTile(sim_game, playerIdx, playable[bestIdx]); } catch { continue; }
    advanceToNextTilePlacement(sim_game, playerIdx);

    let v;
    if (sim_game.phase === 'gameOver') {
      v = rankScore(sim_game.standings || [], game.players[playerIdx].name);
    } else {
      try {
        const { value: lv } = forward(weights, encodeFlat(sim_game, playerIdx));
        v = lv[playerIdx];
      } catch { v = baseValue; }
    }

    N[bestIdx]++;
    Wv[bestIdx] += v;
  }

  // Policy target: normalized visit counts (TRUE AlphaZero targets)
  const totalVisits = N.reduce((s, v) => s + v, 0) || 1;
  const policyTarget = new Float32Array(POLICY_DIM);
  for (let i = 0; i < playable.length; i++) {
    policyTarget[legalIndices[i]] = N[i] / totalVisits;
  }

  // Best move = most visited (AlphaZero convention — robust to outliers)
  let bestTile = playable[0], maxN = -1;
  for (let i = 0; i < playable.length; i++) {
    if (N[i] > maxN) { maxN = N[i]; bestTile = playable[i]; }
  }

  return { tile: bestTile, policyTarget };
}

/* ── Master bot action ────────────────────────────────────────── */
// Tile placement: MCTS-guided (AlphaZero-style visit count policy targets).
// All other phases: random action, no record.
function takeMasterAction(game, playerIdx, weights) {
  const phase = game.phase;
  try {
    if (phase === 'placeTile') {
      const { playable } = engine.getPlayableTiles(game);
      if (playable.length === 0) { engine.passTile(game, playerIdx); return { acted: true, record: null }; }

      const stateBefore  = encodeFlat(game, playerIdx);
      const legalIndices = playable.map(tileToIdx);

      // True AlphaZero: MCTS produces visit-count policy targets
      const { tile, policyTarget } = mctsPickTile(game, playerIdx, weights);
      if (tile === null) { engine.passTile(game, playerIdx); return { acted: true, record: null }; }

      engine.placeTile(game, playerIdx, tile);
      const record = policyTarget ? { state: stateBefore, policyTarget, legalIndices } : null;
      return { acted: true, record };
    }

    // ── Financial decision phases: record state for value head training ──
    // No policy target — only the value head trains on these records.
    // The body trains too (via value gradient), learning to extract
    // financially-relevant features from the board state.
    if (phase === 'buyStock' ||
        (phase === 'mergerDecision' && game.pendingMerger?.decidingPlayer === playerIdx) ||
        phase === 'chooseChain' ||
        phase === 'chooseMergerSurvivor') {
      const stateBefore = encodeFlat(game, playerIdx);
      const acted = takeRandomAction(game, playerIdx);
      return { acted, record: acted ? { state: stateBefore, policyTarget: null, legalIndices: [] } : null };
    }
  } catch {}
  return { acted: takeRandomAction(game, playerIdx), record: null };
}

/* ── Master bot self-play probability ramp ────────────────────── */
// Master share ramp based on cumulative games trained (persisted in weights.totalGames).
// This way restarting a mature model doesn't reset to low self-play ratios.
function masterShareForGame(gamesTotal) {
  if (gamesTotal < 2000)  return 0.40;  // short warmup to seed replay buffer
  if (gamesTotal < 8000)  return 0.70;  // majority self-play
  return 0.95;                           // near-pure self-play
}

/* ── Per-game skill metrics ───────────────────────────────────── */
function computeGameMetrics(game) {
  const bonuses = {}, founded = {};
  for (const entry of (game.log || [])) {
    const m = entry.match(/^(\w+) (?:also )?receives \$(\d+)/);
    if (m) bonuses[m[1]] = (bonuses[m[1]] || 0) + parseInt(m[2], 10);
    const f = entry.match(/^(\w+) founded \w+ and received/);
    if (f) founded[f[1]] = (founded[f[1]] || 0) + 1;
  }
  const majorities = {};
  for (let i = 0; i < game.players.length; i++) {
    const player = game.players[i];
    let majCount = 0;
    for (const c of engine.HOTEL_CHAINS) {
      const ch = game.chains[c];
      if (!ch.active || ch.tiles.length === 0) continue;
      const mine = player.stocks[c] || 0;
      if (mine > 0) {
        const maxOpp = game.players.filter((_, j) => j !== i)
          .reduce((mx, p) => Math.max(mx, p.stocks[c] || 0), 0);
        if (mine >= maxOpp) majCount++;
      }
    }
    majorities[player.name] = majCount;
  }
  return { bonuses, founded, majorities };
}

/* ── Run one self-play game ───────────────────────────────────── */
// Returns { records, standings } or null on error.
function runGame(bots, weights, masterSlots) {
  const names  = bots.map(b => b.name);
  const game   = engine.createGame(names, { quickstart: true });
  const byName = {};
  for (const b of bots) byName[b.name] = b;

  let turns = 0;
  // pending: tile-placement records waiting for game-end value targets
  // { playerIdx, name, state, policyTarget, legalIndices }
  const pending = [];

  while (game.phase !== 'gameOver') {
    if (turns++ > 2500) return null;

    let acted = false;
    for (let i = 0; i < bots.length; i++) {
      const gp  = game.players[i];
      const bot = byName[gp.name];
      if (!bot) continue;

      const isActive = game.phase === 'placeTile'
        ? game.currentPlayerIdx === i
        : game.phase === 'mergerDecision'
          ? game.pendingMerger?.decidingPlayer === i
          : game.currentPlayerIdx === i;
      if (!isActive) continue;

      // End-game declaration: master bots (takeMasterAction) never call
      // declareGameEnd themselves, so without this check games run to MAX_TURNS
      // and return null — bots never see the final stock payout which is where
      // most of the money is. Mirror the heuristic bot's end-game logic here.
      if (game.phase === 'placeTile' && engine.canDeclareGameEnd(game)) {
        const myPlayer    = game.players[i];
        const opponents   = game.players.filter((_, j) => j !== i);
        const bestOpp     = opponents.length ? Math.max(...opponents.map(p => p.cash)) : 0;
        const smallChains = engine.HOTEL_CHAINS.filter(c =>
          game.chains[c].active && game.chains[c].tiles.length < 11);
        if (smallChains.length === 0 || myPlayer.cash > bestOpp * 1.20 ||
            (myPlayer.cash >= bestOpp && smallChains.length <= 1)) {
          engine.declareGameEnd(game, i);
          acted = true;
          break;
        }
      }

      if (Math.random() < EPS) {
        // Pure random exploration — no record (would be noise)
        acted = takeRandomAction(game, i);
      } else if (masterSlots.has(i) && weights) {
        const result = takeMasterAction(game, i, weights);
        acted = result.acted;
        if (result.acted && result.record) {
          pending.push({ playerIdx: i, name: gp.name, ...result.record });
        }
      } else {
        try { acted = decideBotAction(game, i, bot.personality, bot.difficulty, bot.name); }
        catch { return null; }
      }

      if (acted) break;
    }
    if (!acted && game.phase !== 'gameOver') return null;
  }

  if (!game.standings) return null;

  // Rank-based value targets: win-focused, not cash-maximizing.
  // 1st → 1.0, 2nd → 0.5, 3rd → 0.25, 4th → 0.1, 5th → 0.0
  const outcomeByName = {};
  for (const p of game.standings) {
    outcomeByName[p.name] = rankScore(game.standings, p.name);
  }

  // Value targets: per-player outcomes in player-index order
  // [outcome_player0, outcome_player1, ..., outcome_player4]
  const valueTargets = game.players.map(p => outcomeByName[p.name] ?? 0.2);

  // Build final records (attach value targets to each pending tile-decision)
  const records = pending.map(r => ({
    state:        r.state,
    policyTarget: r.policyTarget,
    legalIndices: r.legalIndices,
    valueTargets,
    playerIdx:    r.playerIdx,
  }));

  return { records, standings: game.standings, turns, metrics: computeGameMetrics(game) };
}

/* ── Replay buffer (ring buffer) ──────────────────────────────── */
class ReplayBuffer {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.buf     = [];
    this.pos     = 0;
    this.full    = false;
  }

  push(record) {
    if (this.buf.length < this.maxSize) {
      this.buf.push(record);
    } else {
      this.buf[this.pos] = record;
      this.pos = (this.pos + 1) % this.maxSize;
      this.full = true;
    }
  }

  pushAll(records) {
    for (const r of records) this.push(r);
  }

  get size() { return this.buf.length; }

  sample(n) {
    const sz  = this.buf.length;
    if (sz === 0) return [];
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push(this.buf[Math.floor(Math.random() * sz)]);
    }
    return out;
  }
}

/* ── Training step ────────────────────────────────────────────── */
function trainStep(weights, adam, replay, t) {
  const batch = replay.sample(BATCH_SIZE);
  if (batch.length === 0) return null;

  // Pre-allocate gradient buffers
  const bodyGrads   = initGrads(weights.body);
  const polGrads    = initGrads(weights.policyHead);
  const valGrads    = initGrads(weights.valueHead);

  let totalPolicyLoss = 0;
  let totalValueLoss  = 0;

  for (const { state, policyTarget, legalIndices, valueTargets, playerIdx } of batch) {
    const fwd = forward(weights, state);
    const { h, policyLogits, value } = fwd;

    // ── Policy loss: cross-entropy over legal tiles only ──────── */
    // Skipped for financial-decision records (policyTarget is null) —
    // those records only train the value head and body.
    const policyDelta = new Float64Array(POLICY_DIM);
    if (policyTarget && legalIndices.length > 0) {
      const legalLogits = legalIndices.map(i => policyLogits[i]);
      const legalProbs  = softmax(legalLogits);
      const legalTarget = legalIndices.map(i => policyTarget[i]);

      let pLoss = 0;
      for (let i = 0; i < legalProbs.length; i++) {
        pLoss -= legalTarget[i] * Math.log(legalProbs[i] + 1e-9);
      }
      totalPolicyLoss += pLoss;

      for (let i = 0; i < legalIndices.length; i++) {
        policyDelta[legalIndices[i]] = legalProbs[i] - legalTarget[i];
      }
    }

    // ── Value loss: MSE per player ────────────────────────────── */
    // value[i] is sigmoid output for player i; valueTargets[i] is truth.
    // ∂MSE/∂sigmoid_i = 2*(pred_i - target_i)
    // ∂sigmoid/∂z_i = pred_i*(1-pred_i)   → chain rule gives:
    // ∂L/∂z_i = 2*(pred_i - target_i)*pred_i*(1-pred_i)
    let vLoss = 0;
    const valueDelta = new Float64Array(VALUE_DIM);
    for (let i = 0; i < VALUE_DIM; i++) {
      const pred = value[i];
      const tgt  = valueTargets[i] ?? 0.2;
      const err  = pred - tgt;
      vLoss        += err * err;
      // The sigmoid gradient is baked into backpropLayersWithReturn because we
      // pass 'sigmoid' as activation, but we need to pass pre-sigmoid delta.
      // Easier: compute the pre-activation delta manually here.
      // ∂L/∂z_value_i = (pred_i - target_i) * pred_i*(1-pred_i)   [BCE-style]
      // For MSE: 2*(pred_i - target_i)*pred_i*(1-pred_i)
      valueDelta[i] = 2 * err * pred * (1 - pred);
    }
    totalValueLoss += vLoss / VALUE_DIM;

    // ── Backprop through policy head → get delta at h ─────────── */
    const deltaFromPolicy = backpropLayersWithReturn(
      weights.policyHead, fwd.polActs, fwd.polZs, policyDelta,
      polGrads.W, polGrads.b
    );

    // ── Backprop through value head → get delta at h ─────────── */
    const deltaFromValue = backpropLayersWithReturn(
      weights.valueHead, fwd.valActs, fwd.valZs, valueDelta,
      valGrads.W, valGrads.b
    );

    // ── Sum deltas at h, backprop through body ─────────────────── */
    const deltaH = new Float64Array(h.length);
    for (let i = 0; i < h.length; i++) {
      deltaH[i] = deltaFromPolicy[i] + deltaFromValue[i];
    }
    // Apply ReLU gradient for the body's last layer output
    const bodyLastZ = fwd.bodyZs[fwd.bodyZs.length - 1];
    for (let i = 0; i < deltaH.length; i++) {
      if (bodyLastZ[i] <= 0) deltaH[i] = 0;
    }
    backpropLayers(weights.body, fwd.bodyActs, fwd.bodyZs, deltaH, bodyGrads.W, bodyGrads.b);
  }

  // Scale by batch size
  const scale = 1 / batch.length;
  function scaleGrads(grads) {
    for (const row of grads.W.flat()) for (let i = 0; i < row.length; i++) row[i] *= scale;
    for (const b of grads.b) for (let i = 0; i < b.length; i++) b[i] *= scale;
  }
  scaleGrads(bodyGrads);
  scaleGrads(polGrads);
  scaleGrads(valGrads);

  // Adam updates
  adamStepLayers(weights.body,       adam.body,       bodyGrads.W, bodyGrads.b, t);
  adamStepLayers(weights.policyHead, adam.policyHead, polGrads.W,  polGrads.b,  t);
  adamStepLayers(weights.valueHead,  adam.valueHead,  valGrads.W,  valGrads.b,  t);

  return {
    policyLoss: totalPolicyLoss / batch.length,
    valueLoss:  totalValueLoss  / batch.length,
    totalLoss:  (totalPolicyLoss + totalValueLoss) / batch.length,
  };
}

/* ── Main loop ────────────────────────────────────────────────── */
async function train() {
  // ── Pidfile guard: kill any existing instance before starting ──
  const PIDFILE = '/tmp/train_online.pid';
  try {
    const oldPid = parseInt(fs.readFileSync(PIDFILE, 'utf8').trim(), 10);
    if (oldPid && oldPid !== process.pid) {
      try { process.kill(oldPid, 'SIGKILL'); } catch {}
      await new Promise(r => setTimeout(r, 500));
    }
  } catch {}
  fs.writeFileSync(PIDFILE, String(process.pid));
  process.on('exit', () => { try { fs.unlinkSync(PIDFILE); } catch {} });

  console.log('\nAcquire Master Bot — AlphaZero-style Training (v3)');
  console.log(`  LR=${LR}  batch=${BATCH_SIZE}  train_every=${TRAIN_EVERY}  replay=${REPLAY_SIZE}`);
  if (TIME_LIMIT !== Infinity) console.log(`  Time limit: ${TIME_LIMIT / 3600000}h`);
  console.log('  Architecture: shared body [258→256→128→64] + policy head [64→108] + value head [64→5]\n');
  console.log(`  MCTS: ${MCTS_SIMS} sims/tile, C_PUCT=${C_PUCT}\n`);

  const weights = loadOrInit();
  const adam    = initAdam(weights);
  const replay  = new ReplayBuffer(REPLAY_SIZE);

  // ── Spot instance log ─────────────────────────────────────────
  const INST_LOG_KEY = 'instance_log.json';
  let instanceLog = [];
  try {
    execSync(`aws s3 cp s3://${S3_BUCKET}/${INST_LOG_KEY} /tmp/instance_log.json`, { timeout: 15000, stdio: 'pipe' });
    instanceLog = JSON.parse(fs.readFileSync('/tmp/instance_log.json', 'utf8'));
  } catch {}
  instanceLog.unshift({ type: 'boot', at: new Date().toISOString() });
  if (instanceLog.length > 100) instanceLog.length = 100;
  try {
    fs.writeFileSync('/tmp/instance_log.json', JSON.stringify(instanceLog));
    execSync(`aws s3 cp /tmp/instance_log.json s3://${S3_BUCKET}/${INST_LOG_KEY}`, { timeout: 15000, stdio: 'pipe' });
  } catch {}

  // On spot termination (SIGTERM), flush everything to S3 before dying.
  // Spot instances get a ~2 minute warning via SIGTERM before being forcibly killed.
  process.on('SIGTERM', () => {
    try {
      // 1. Save weights with all current in-memory history
      weights.totalGames        = gamesTotal;
      weights.totalSteps        = t;
      weights.gameCashHistory   = gameCashHistory.slice();
      weights.winnerCashHistory = winnerCashHistory.slice();
      weights.bestMasterCash    = bestMasterCash;
      weights.bestCashHistory   = bestCashHistory;
      saveWeights(weights);
    } catch {}

    try {
      // 2. Upload latest stats snapshot so dashboard stays current
      const elapsedS  = ((Date.now() - t0) / 1000).toFixed(0);
      const gamesThisRun = gamesTotal - startGamesTotal;
      const tmpStats  = '/tmp/training_stats.json';
      const snapshot  = {
        step: t, gamesTotal, gamesThisRun,
        gamesPlayed: gamesTotal - errors,
        masterGames, errors,
        avgLoss:  lossHistory[lossHistory.length - 1] || null,
        lossHistory: lossHistory.slice(),
        replaySize: replay.size,
        gameCashAvg:      gameCashHistory[gameCashHistory.length - 1] || null,
        gameCashHistory:  gameCashHistory.slice(),
        winnerCashAvg:    winnerCashHistory[winnerCashHistory.length - 1] || null,
        winnerCashHistory: winnerCashHistory.slice(),
        bestMasterCash, bestCashHistory: bestCashHistory.slice(),
        instanceLog: instanceLog.slice(0, 30),
        elapsedSecs: parseInt(elapsedS),
        gamesPerSec:  gamesThisRun > 0 ? +(gamesThisRun / parseInt(elapsedS)).toFixed(2) : 0,
        updatedAt: new Date().toISOString(),
      };
      fs.writeFileSync(tmpStats, JSON.stringify(snapshot));
      execSync(`aws s3 cp "${tmpStats}" s3://${S3_BUCKET}/training_stats.json`, { timeout: 15000, stdio: 'pipe' });
    } catch {}

    try {
      // 3. Log the drop event
      instanceLog.unshift({ type: 'drop', at: new Date().toISOString() });
      if (instanceLog.length > 100) instanceLog.length = 100;
      fs.writeFileSync('/tmp/instance_log.json', JSON.stringify(instanceLog));
      execSync(`aws s3 cp /tmp/instance_log.json s3://${S3_BUCKET}/${INST_LOG_KEY}`, { timeout: 10000, stdio: 'pipe' });
    } catch {}

    process.exit(0);
  });

  let t           = weights.totalSteps || 0;  // cumulative gradient steps across runs
  let gamesTotal  = weights.totalGames || 0;  // cumulative across runs
  const startGamesTotal = gamesTotal;          // baseline for this run
  let masterGames = 0;
  let errors      = 0;
  let lossHistory = [];
  let rollingLoss = { policy: 0, value: 0, total: 0, cnt: 0 };
  const wins    = {};
  const podiums = {};
  const avgCash = {};
  const elo     = {};
  let   totalTurns = 0;
  for (const b of BOTS) { wins[b.name] = 0; podiums[b.name] = 0; avgCash[b.name] = []; elo[b.name] = 1500; }

  // Skill metric accumulators (reset each run, history persisted in stats)
  const mergerBonuses  = {};
  const chainsFoundArr = {};
  const majoritiesArr  = {};
  for (const b of BOTS) { mergerBonuses[b.name] = []; chainsFoundArr[b.name] = []; majoritiesArr[b.name] = []; }
  const MAX_HIST = 500;
  // Load existing botMetricHistory from last stats write so history survives restarts
  let botMetricHistory = {};
  try {
    const prev = JSON.parse(fs.readFileSync('/tmp/training_stats.json', 'utf8'));
    botMetricHistory = prev.botMetricHistory || {};
  } catch {}

  // "ELO" equivalent: track master bot's average ending cash over time.
  // We record the rolling average every 1000 games so you can see the
  // trend: a rising curve means the bot is accumulating more money per game.
  // Master bots are all named in BOTS — we track the average across all of
  // them (since they're the same network in different seat positions).
  let masterCashWindow = [];  // cash values since last checkpoint
  const masterCashHistory = []; // one avg per 1000-game checkpoint (last 50)
  let gameCashWindow    = [];  // total cash across all players per game
  const gameCashHistory = weights.gameCashHistory ? [...weights.gameCashHistory] : []; // persisted across runs
  let winnerCashWindow  = [];  // winner's cash per game
  const winnerCashHistory = weights.winnerCashHistory ? [...weights.winnerCashHistory] : []; // persisted across runs
  let bestMasterCash = weights.bestMasterCash || 0;
  const bestCashHistory = weights.bestCashHistory ? [...weights.bestCashHistory] : [];
  const t0 = Date.now();

  while (Date.now() - t0 < TIME_LIMIT) {
    // ── Self-play game ────────────────────────────────────────── */
    const bots = shuffle(BOTS);
    const pMaster = masterShareForGame(gamesTotal);
    const masterSlots = new Set(bots.map((_, i) => i).filter(() => Math.random() < pMaster));

    const result = runGame(bots, weights, masterSlots);
    gamesTotal++;
    if (!result) { errors++; }
    else {
      if (masterSlots.size > 0) masterGames++;
      wins[result.standings[0].name] = (wins[result.standings[0].name] || 0) + 1;
      totalTurns += result.turns;
      gameCashWindow.push(result.standings.reduce((s, p) => s + p.cash, 0));
      winnerCashWindow.push(result.standings[0].cash);
      updateElo(elo, result.standings);
      for (let i = 0; i < result.standings.length; i++) {
        const { name, cash } = result.standings[i];
        if (i < 3 && podiums[name] !== undefined) podiums[name]++;
        if (avgCash[name]) {
          avgCash[name].push(cash);
          if (avgCash[name].length > 5000) avgCash[name].shift();
        }
      }
      // Track master bot cash for ELO-equivalent trending
      for (const slotIdx of masterSlots) {
        const playerInGame = bots[slotIdx];
        if (playerInGame) {
          const standing = result.standings.find(s => s.name === playerInGame.name);
          if (standing) {
            masterCashWindow.push(standing.cash);
            // Top-15 leaderboard: qualify if list isn't full or this beats the lowest entry
            if (bestCashHistory.length < 15 || standing.cash > bestCashHistory[bestCashHistory.length - 1].cash) {
              bestCashHistory.push({ cash: standing.cash, step: t, game: gamesTotal, at: new Date().toISOString() });
              bestCashHistory.sort((a, b) => b.cash - a.cash);
              if (bestCashHistory.length > 15) bestCashHistory.length = 15;
              bestMasterCash = bestCashHistory[0].cash;
              weights.bestMasterCash    = bestMasterCash;
              weights.bestCashHistory   = bestCashHistory;
              weights.totalGames        = gamesTotal;
              weights.totalSteps        = t;
              weights.gameCashHistory   = gameCashHistory.slice();
              weights.winnerCashHistory = winnerCashHistory.slice();
            }
          }
        }
      }
      replay.pushAll(result.records);
      // Accumulate skill metrics
      if (result.metrics) {
        for (const b of BOTS) {
          mergerBonuses[b.name].push(result.metrics.bonuses[b.name] || 0);
          chainsFoundArr[b.name].push(result.metrics.founded[b.name] || 0);
          majoritiesArr[b.name].push(result.metrics.majorities[b.name] || 0);
          if (mergerBonuses[b.name].length  > 5000) mergerBonuses[b.name].shift();
          if (chainsFoundArr[b.name].length > 5000) chainsFoundArr[b.name].shift();
          if (majoritiesArr[b.name].length  > 5000) majoritiesArr[b.name].shift();
        }
      }
    }

    // ── Train every TRAIN_EVERY games once buffer has enough data ─ */
    if (gamesTotal % TRAIN_EVERY === 0 && replay.size >= BATCH_SIZE) {
      t++;
      const losses = trainStep(weights, adam, replay, t);
      if (losses) {
        rollingLoss.policy += losses.policyLoss;
        rollingLoss.value  += losses.valueLoss;
        rollingLoss.total  += losses.totalLoss;
        rollingLoss.cnt++;
      }
    }

    // ── Progress log every 1000 games ─────────────────────────── */
    if (gamesTotal % 1000 === 0) {
      const elapsedS = ((Date.now() - t0) / 1000).toFixed(0);
      const played   = gamesTotal - errors;
      const avgPol   = rollingLoss.cnt > 0 ? (rollingLoss.policy / rollingLoss.cnt).toFixed(4) : 'n/a';
      const avgVal   = rollingLoss.cnt > 0 ? (rollingLoss.value  / rollingLoss.cnt).toFixed(4) : 'n/a';
      const avgTot   = rollingLoss.cnt > 0 ? (rollingLoss.total  / rollingLoss.cnt).toFixed(4) : 'n/a';
      lossHistory.push(rollingLoss.cnt > 0 ? parseFloat(avgTot) : null);
      if (lossHistory.length > 200) lossHistory.shift();
      rollingLoss = { policy: 0, value: 0, total: 0, cnt: 0 };

      // Master bot "ELO" snapshot: average cash this 1000-game window
      const masterCashAvg = masterCashWindow.length > 0
        ? Math.round(masterCashWindow.reduce((s, v) => s + v, 0) / masterCashWindow.length)
        : null;
      if (masterCashAvg !== null) {
        masterCashHistory.push(masterCashAvg);
        if (masterCashHistory.length > 200) masterCashHistory.shift();
      }
      masterCashWindow = [];

      const gameCashAvg = gameCashWindow.length > 0
        ? Math.round(gameCashWindow.reduce((s, v) => s + v, 0) / gameCashWindow.length)
        : null;
      if (gameCashAvg !== null) {
        gameCashHistory.push(gameCashAvg);
        if (gameCashHistory.length > 500) gameCashHistory.shift();
      }
      gameCashWindow = [];

      const winnerCashAvg = winnerCashWindow.length > 0
        ? Math.round(winnerCashWindow.reduce((s, v) => s + v, 0) / winnerCashWindow.length)
        : null;
      if (winnerCashAvg !== null) {
        winnerCashHistory.push(winnerCashAvg);
        if (winnerCashHistory.length > 500) winnerCashHistory.shift();
      }
      winnerCashWindow = [];

      // Append skill metric snapshot to rolling history
      const avgArr = a => a.length ? a.reduce((s,v)=>s+v,0)/a.length : 0;
      for (const b of BOTS) {
        const n = b.name;
        const prev = botMetricHistory[n] || { mergerBonus: [], chainsFound: [], majorities: [], elo: [] };
        botMetricHistory[n] = {
          mergerBonus: [...prev.mergerBonus, Math.round(avgArr(mergerBonuses[n]))].slice(-MAX_HIST),
          chainsFound: [...prev.chainsFound, +avgArr(chainsFoundArr[n]).toFixed(2)].slice(-MAX_HIST),
          majorities:  [...prev.majorities,  +avgArr(majoritiesArr[n]).toFixed(2)].slice(-MAX_HIST),
          elo:         [...prev.elo,          elo[n]].slice(-MAX_HIST),
        };
        // Reset windows for next checkpoint
        mergerBonuses[n]  = [];
        chainsFoundArr[n] = [];
        majoritiesArr[n]  = [];
      }

      log(`  step=${t} games=${gamesTotal} loss=${avgTot} (pol=${avgPol} val=${avgVal}) elapsed=${elapsedS}s buf=${replay.size} errors=${errors}\n`);

      const gamesThisRun = gamesTotal - startGamesTotal;
      const stats = {
        step: t,
        gamesTotal,
        gamesThisRun,
        gamesPlayed: played,
        masterGames,
        errors,
        avgLoss:  rollingLoss.cnt > 0 ? parseFloat(avgTot) : lossHistory[lossHistory.length - 1],
        policyLoss: parseFloat(avgPol),
        valueLoss:  parseFloat(avgVal),
        lossHistory: lossHistory.filter(v => v !== null),
        replaySize:  replay.size,
        masterCashAvg:     masterCashAvg,
        masterCashHistory: masterCashHistory.slice(),
        bestMasterCash,
        bestCashHistory:   bestCashHistory.slice(),
        gameCashAvg,
        gameCashHistory:    gameCashHistory.slice(),
        winnerCashAvg,
        winnerCashHistory:  winnerCashHistory.slice(),
        instanceLog:       instanceLog.slice(0, 30),
        elapsedSecs: parseInt(elapsedS),
        remainingSecs: Math.max(0, Math.floor((TIME_LIMIT - (Date.now() - t0)) / 1000)),
        timeLimitSecs: TIME_LIMIT === Infinity ? null : TIME_LIMIT / 1000,
        gamesPerSec: gamesThisRun > 0 ? +(gamesThisRun / parseInt(elapsedS)).toFixed(2) : 0,
        updatedAt: new Date().toISOString(),
        avgTurns:        gamesThisRun > 0 ? +(totalTurns / gamesThisRun).toFixed(1) : 0,
        exportedRecords: replay.size,
        botMetricHistory,
        bots: BOTS.map(b => {
          const cash = avgCash[b.name] || [];
          const w    = wins[b.name] || 0;
          return {
            name:       b.name,
            difficulty: b.difficulty,
            wins:       w,
            winPct:     played > 0 ? +((w / played) * 100).toFixed(1) : 0,
            top3Pct:    played > 0 ? +((podiums[b.name] / played) * 100).toFixed(1) : 0,
            avgCash:    cash.length ? Math.round(cash.reduce((s, v) => s + v, 0) / cash.length) : 0,
            elo:        elo[b.name],
          };
        }).sort((a, b) => b.elo - a.elo),
      };

      if (STATS_FILE) {
        try { fs.writeFileSync(STATS_FILE, JSON.stringify(stats)); } catch {}
      }
      try {
        const tmpStats = '/tmp/training_stats.json';
        fs.writeFileSync(tmpStats, JSON.stringify(stats));
        execSync(`aws s3 cp "${tmpStats}" s3://${S3_BUCKET}/training_stats.json`, { timeout: 15000, stdio: 'pipe' });
      } catch {}
    }

    // ── Save weights periodically ─────────────────────────────── */
    if (gamesTotal % SAVE_EVERY === 0) {
      weights.totalGames      = gamesTotal;
      weights.totalSteps      = t;
      weights.gameCashHistory  = gameCashHistory.slice();
      weights.winnerCashHistory = winnerCashHistory.slice();
      saveWeights(weights);
      log(`  [saved] step=${t} games=${gamesTotal}\n`);
    }

    // Breathe
    if (gamesTotal % 100 === 0) await new Promise(resolve => setImmediate(resolve));
  }

  saveWeights(weights);
  const elapsedMin = ((Date.now() - t0) / 60000).toFixed(1);
  console.log(`\nDone. ${gamesTotal} games, ${t} gradient steps, ${elapsedMin}min`);
}

train().catch(err => { console.error('Fatal:', err); process.exit(1); });
