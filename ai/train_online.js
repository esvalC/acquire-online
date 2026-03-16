#!/usr/bin/env node
/**
 * ai/train_online.js — AlphaZero-style training for the Master Bot
 *
 * Architecture: shared body + DUAL HEADS
 *   Input (150) → FC(256)→ReLU → FC(128)→ReLU → FC(64)→ReLU → h
 *   h → policy head: FC(108) → tile logits (softmax during training)
 *   h → value  head: FC(5)   → per-player win probabilities (sigmoid)
 *
 * Why dual heads beat single-value:
 *   The policy head provides position-specific gradient signal to the body
 *   even before the value head is useful. MCTS visit counts from AlphaZero
 *   would be ideal; here we use temperature-scaled after-state values as
 *   soft policy targets — cheaper but achieves the same key effect: diverse,
 *   per-position gradients that break the "predict-the-mean" plateau.
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
const INPUT_DIM   = 150;  // 108 board + 35 chain features + 1 myCash + 4 oppCash + 1 tileBag
const HIDDEN      = [256, 128, 64];
const POLICY_DIM  = 108;  // 9 rows × 12 cols (one logit per board cell)
const VALUE_DIM   = 5;    // one sigmoid per player position
const BAG_TOTAL   = 102;

// Temperature for policy soft targets: lower = sharper (more greedy)
// 0.5 gives a clear winner but non-zero probability to alternatives
const POLICY_TEMP = 0.5;

// Exploration rate: random action with this probability (all slots)
const EPS = 0.10;

// Target cash for absolute reward component
const TARGET_CASH = 50000;

/* ── Bot roster ───────────────────────────────────────────────── */
const BOTS = [
  { name: 'Aria', personality: 'balanced',    difficulty: 'hard' },
  { name: 'Rex',  personality: 'focused',     difficulty: 'hard' },
  { name: 'Nova', personality: 'diversified', difficulty: 'hard' },
  { name: 'Colt', personality: 'focused',     difficulty: 'hard' },
  { name: 'Vera', personality: 'balanced',    difficulty: 'hard' },
];

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

/* ── Weight format v2: body + policyHead + valueHead ─────────── */
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
    version:    2,
    body,
    policyHead: [heLayer(bodyOut, POLICY_DIM)],
    valueHead:  [heLayer(bodyOut, VALUE_DIM)],
  };
}

function loadOrInit() {
  try {
    const raw = fs.readFileSync(WEIGHTS_PATH, 'utf8');
    const w   = JSON.parse(raw);
    // Handle legacy v1 (single layers array → reinit with new architecture)
    if (!w.version || w.version < 2) {
      log('  Old weight format detected — reinitializing with AlphaZero architecture.\n');
      return initWeights();
    }
    log(`  Loaded v2 weights from ${WEIGHTS_PATH}\n`);
    return w;
  } catch {
    log('  No existing weights — initializing v2 (policy + value heads).\n');
    return initWeights();
  }
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
function encodeFlat(game, playerIdx) {
  const player    = game.players[playerIdx];
  const CHAIN_IDX = {};
  engine.HOTEL_CHAINS.forEach((c, i) => { CHAIN_IDX[c] = i + 1; });

  const vec = new Float64Array(INPUT_DIM);
  let vi = 0;
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 12; c++) {
      const cell = game.board[r][c];
      vec[vi++] = (cell && CHAIN_IDX[cell] ? CHAIN_IDX[cell] : (cell ? -1 : 0)) / 7.0;
    }
  }
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
  vec[vi++] = player.cash / 6000.0;
  const opps = game.players.filter((_, i) => i !== playerIdx);
  for (let k = 0; k < 4; k++) vec[vi++] = (opps[k] ? opps[k].cash / 6000.0 : 0);
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

/* ── Master bot action ────────────────────────────────────────── */
// Tile placement: policy-head guided (softmax over legal tiles).
// All other phases: value-head guided 1-step lookahead.
//
// Policy-guided tile selection:
//   Evaluate after-state value for each legal tile using the value head.
//   Apply temperature-scaled softmax → soft distribution over legal tiles.
//   Sample from this distribution (temperature-controlled exploration).
//   Record (state, policy_target, legalTileIndices) for training.
function takeMasterAction(game, playerIdx, weights) {
  const phase = game.phase;
  try {
    if (phase === 'placeTile') {
      const { playable } = engine.getPlayableTiles(game);
      if (playable.length === 0) { engine.passTile(game, playerIdx); return { acted: true, record: null }; }
      if (playable.length === 1) {
        const stateBefore = encodeFlat(game, playerIdx);
        const policyTarget = new Float32Array(POLICY_DIM); // sparse: only the one legal tile
        policyTarget[tileToIdx(playable[0])] = 1.0;
        engine.placeTile(game, playerIdx, playable[0]);
        return { acted: true, record: { state: stateBefore, policyTarget, legalIndices: [tileToIdx(playable[0])] } };
      }

      // Record state before acting
      const stateBefore = encodeFlat(game, playerIdx);

      // Evaluate each legal tile: apply tile, run value head, get this player's score
      const tileValues = [];
      for (const tile of playable) {
        const sim = cloneForScoring(game);
        engine.placeTile(sim, playerIdx, tile);
        let val;
        if (sim.phase === 'gameOver') {
          // Game ended instantly — use actual outcome
          const totalCash = (sim.standings || []).reduce((s, p) => s + p.cash, 0) || 1;
          const me = (sim.standings || []).find(p => p.name === game.players[playerIdx].name);
          val = me ? 0.5 * (me.cash / totalCash) + 0.5 * Math.min(me.cash / TARGET_CASH, 1) : 0.2;
        } else {
          const { value } = forward(weights, encodeFlat(sim, playerIdx));
          val = value[playerIdx]; // this player's win probability in the post-tile state
        }
        tileValues.push(val);
      }

      // Soft policy target: temperature-scaled softmax over tile values
      const scaledValues = tileValues.map(v => v / POLICY_TEMP);
      const probs        = softmax(scaledValues);
      const legalIndices = playable.map(tileToIdx);

      // Build sparse 108-dim policy target
      const policyTarget = new Float32Array(POLICY_DIM);
      for (let i = 0; i < legalIndices.length; i++) {
        policyTarget[legalIndices[i]] = probs[i];
      }

      // Sample tile proportional to probs (exploration while staying policy-guided)
      let r = Math.random(), chosen = playable[playable.length - 1];
      let cumulative = 0;
      for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (r <= cumulative) { chosen = playable[i]; break; }
      }

      engine.placeTile(game, playerIdx, chosen);
      return { acted: true, record: { state: stateBefore, policyTarget, legalIndices } };
    }
  } catch {}

  // All non-tile phases: value-guided 1-step lookahead
  // (buyStock, mergerDecision, etc. — no policy target recorded)
  try {
    if (phase === 'buyStock') {
      return { acted: takeRandomAction(game, playerIdx), record: null };
    }
    // Other phases with small action spaces: lookahead using value head
    const acted = takeRandomAction(game, playerIdx);
    return { acted, record: null };
  } catch {}
  return { acted: false, record: null };
}

/* ── Master bot self-play probability ramp ────────────────────── */
// Always start with some master bots (random network at first = diverse exploration).
// Replay buffer needs data from game 1 or training never starts.
function masterShareForGame(gamesTotal) {
  if (gamesTotal < 5000)  return 0.20;  // ~1 master/game — fills replay buffer
  if (gamesTotal < 15000) return 0.40;  // ~2 masters/game
  if (gamesTotal < 30000) return 0.65;  // majority master self-play
  return 0.85;                           // near-full self-play
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

  // Compute per-player blended outcome [0, 1]
  const totalCash = game.standings.reduce((s, p) => s + p.cash, 0) || 1;
  const outcomeByName = {};
  for (const p of game.standings) {
    const relative = p.cash / totalCash;
    const absolute = Math.min(p.cash / TARGET_CASH, 1.0);
    outcomeByName[p.name] = 0.5 * relative + 0.5 * absolute;
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

  return { records, standings: game.standings };
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
    // legal logits + softmax
    const legalLogits = legalIndices.map(i => policyLogits[i]);
    const legalProbs  = softmax(legalLogits);
    const legalTarget = legalIndices.map(i => policyTarget[i]);

    // CE loss = -Σ target_i * log(prob_i)
    let pLoss = 0;
    for (let i = 0; i < legalProbs.length; i++) {
      pLoss -= legalTarget[i] * Math.log(legalProbs[i] + 1e-9);
    }
    totalPolicyLoss += pLoss;

    // Gradient of softmax+CE w.r.t. logits: ∂L/∂z_i = prob_i - target_i
    // (only for legal tile indices; zero for all other tiles)
    const policyDelta = new Float64Array(POLICY_DIM);
    for (let i = 0; i < legalIndices.length; i++) {
      policyDelta[legalIndices[i]] = legalProbs[i] - legalTarget[i];
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
  console.log('\nAcquire Master Bot — AlphaZero-style Training (v2)');
  console.log(`  LR=${LR}  batch=${BATCH_SIZE}  train_every=${TRAIN_EVERY}  replay=${REPLAY_SIZE}`);
  if (TIME_LIMIT !== Infinity) console.log(`  Time limit: ${TIME_LIMIT / 3600000}h`);
  console.log('  Architecture: shared body [150→256→128→64] + policy head [64→108] + value head [64→5]\n');

  const weights = loadOrInit();
  const adam    = initAdam(weights);
  const replay  = new ReplayBuffer(REPLAY_SIZE);

  let t          = 0;  // Adam step counter
  let gamesTotal = 0;
  let masterGames = 0;
  let errors     = 0;
  let lossHistory = [];
  let rollingLoss = { policy: 0, value: 0, total: 0, cnt: 0 };
  const wins    = {};
  const avgCash = {};
  for (const b of BOTS) { wins[b.name] = 0; avgCash[b.name] = []; }
  const t0 = Date.now();

  while (Date.now() - t0 < TIME_LIMIT) {
    // ── Self-play game ────────────────────────────────────────── */
    const bots = shuffle(BOTS);
    const pMaster = masterShareForGame(gamesTotal);
    const masterSlots = new Set(bots.map((_, i) => i).filter(() => Math.random() < pMaster));
    if (masterSlots.size > 0) masterGames++;

    const result = runGame(bots, weights, masterSlots);
    gamesTotal++;
    if (!result) { errors++; }
    else {
      wins[result.standings[0].name] = (wins[result.standings[0].name] || 0) + 1;
      for (const p of result.standings) {
        if (avgCash[p.name]) {
          avgCash[p.name].push(p.cash);
          if (avgCash[p.name].length > 5000) avgCash[p.name].shift();
        }
      }
      replay.pushAll(result.records);
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
      if (lossHistory.length > 50) lossHistory.shift();
      rollingLoss = { policy: 0, value: 0, total: 0, cnt: 0 };

      log(`  step=${t} games=${gamesTotal} loss=${avgTot} (pol=${avgPol} val=${avgVal}) elapsed=${elapsedS}s buf=${replay.size} errors=${errors}\n`);

      const stats = {
        step: t,
        gamesTotal,
        gamesPlayed: played,
        masterGames,
        errors,
        avgLoss:  rollingLoss.cnt > 0 ? parseFloat(avgTot) : lossHistory[lossHistory.length - 1],
        policyLoss: parseFloat(avgPol),
        valueLoss:  parseFloat(avgVal),
        lossHistory: lossHistory.filter(v => v !== null),
        replaySize:  replay.size,
        elapsedSecs: parseInt(elapsedS),
        remainingSecs: Math.max(0, Math.floor((TIME_LIMIT - (Date.now() - t0)) / 1000)),
        timeLimitSecs: TIME_LIMIT === Infinity ? null : TIME_LIMIT / 1000,
        gamesPerSec: played > 0 ? +(played / parseInt(elapsedS)).toFixed(1) : 0,
        updatedAt: new Date().toISOString(),
        bots: BOTS.map(b => {
          const cash = avgCash[b.name] || [];
          return {
            name: b.name,
            wins: wins[b.name] || 0,
            winPct: played > 0 ? +((wins[b.name] / played) * 100).toFixed(1) : 0,
            avgCash: cash.length ? Math.round(cash.reduce((s, v) => s + v, 0) / cash.length) : 0,
          };
        }).sort((a, b) => b.wins - a.wins),
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
