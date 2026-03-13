#!/usr/bin/env node
/**
 * ai/train_online.js — Continuous online training for the Master Bot
 *
 * Combines self-play + training into one loop. After every BATCH_SIZE games,
 * runs a mini-batch gradient update on those games' records and discards them.
 * No giant JSONL file needed, no Python required.
 *
 * Usage:
 *   node ai/train_online.js                         # train until Ctrl-C
 *   node ai/train_online.js --time-limit 36000      # train for 10 hours
 *   node ai/train_online.js --lr 0.0005             # custom learning rate
 *
 * The weights file (ai/models/master_weights.json) is updated every SAVE_EVERY
 * games and can be deployed to the live site at any point.
 */

'use strict';

const engine        = require('../gameEngine');
const { decideBotAction } = require('../botAI');
const fs   = require('fs');
const path = require('path');

/* ── Config ──────────────────────────────────────────────────── */
const BATCH_SIZE  = 200;   // games per gradient step
const SAVE_EVERY  = 1000;  // games between weight saves
const LR          = parseFloat(getArg('--lr', '0.001'));
const TIME_LIMIT  = getArg('--time-limit') ? parseInt(getArg('--time-limit')) * 1000 : Infinity;
const STATS_FILE  = getArg('--stats') || null;

const WEIGHTS_PATH = path.join(__dirname, 'models', 'master_weights.json');
const ARCH = [149, 256, 128, 64, 1];

const BOTS = [
  { name: 'Aria',  personality: 'balanced',    difficulty: 'hard' },
  { name: 'Rex',   personality: 'focused',     difficulty: 'hard' },
  { name: 'Nova',  personality: 'diversified', difficulty: 'hard' },
  { name: 'Colt',  personality: 'focused',     difficulty: 'hard' },
  { name: 'Vera',  personality: 'balanced',    difficulty: 'hard' },
];

function getArg(flag, def) {
  const idx = process.argv.indexOf(flag);
  return idx === -1 ? def : process.argv[idx + 1];
}

/* ── Weight init / load ──────────────────────────────────────── */
function initWeights() {
  const layers = [];
  for (let l = 0; l < ARCH.length - 1; l++) {
    const fanIn  = ARCH[l];
    const fanOut = ARCH[l + 1];
    const scale  = Math.sqrt(2.0 / fanIn); // He init
    const W = Array.from({ length: fanOut }, () =>
      Array.from({ length: fanIn }, () => (Math.random() * 2 - 1) * scale)
    );
    const b = new Array(fanOut).fill(0);
    layers.push({ W, b });
  }
  return { layers };
}

function loadOrInit() {
  try {
    const raw = fs.readFileSync(WEIGHTS_PATH, 'utf8');
    console.log('  Loaded existing weights from', WEIGHTS_PATH);
    return JSON.parse(raw);
  } catch {
    console.log('  No existing weights — initializing randomly.');
    return initWeights();
  }
}

function saveWeights(weights) {
  fs.mkdirSync(path.dirname(WEIGHTS_PATH), { recursive: true });
  fs.writeFileSync(WEIGHTS_PATH, JSON.stringify(weights));
}

/* ── Adam optimizer state ────────────────────────────────────── */
function initAdam(weights) {
  return weights.layers.map(({ W, b }) => ({
    mW: W.map(row => new Float64Array(row.length)),
    vW: W.map(row => new Float64Array(row.length)),
    mb: new Float64Array(b.length),
    vb: new Float64Array(b.length),
  }));
}

/* ── Forward pass (returns activations for backprop) ─────────── */
function relu(x)    { return x > 0 ? x : 0; }
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x)))); }

function forwardFull(weights, input) {
  const acts = [input]; // acts[l] = post-activation at layer l
  const zs   = [];      // zs[l]   = pre-activation at layer l (for backprop)
  const layers = weights.layers;
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
      z[j]   = sum;
      out[j] = isLast ? sigmoid(sum) : relu(sum);
    }
    zs.push(z);
    acts.push(out);
    x = out;
  }
  return { yHat: x[0], acts, zs };
}

/* ── Backprop ─────────────────────────────────────────────────── */
// Accumulates gradients into gradW / gradB arrays (one per layer).
function backprop(weights, acts, zs, yTrue, gradW, gradB) {
  const L = weights.layers.length;
  // BCE + sigmoid output: dL/dz_out = yHat - yTrue
  let delta = new Float64Array(1);
  delta[0] = acts[L][0] - yTrue;

  for (let li = L - 1; li >= 0; li--) {
    const { W } = weights.layers[li];
    const xPrev = acts[li];
    const outDim = W.length;
    const inDim  = W[0].length;

    // Accumulate gradients
    for (let j = 0; j < outDim; j++) {
      gradB[li][j] += delta[j];
      const Wj = W[j];
      const gWj = gradW[li][j];
      const dj = delta[j];
      for (let i = 0; i < inDim; i++) gWj[i] += dj * xPrev[i];
    }

    if (li === 0) break;

    // Propagate delta to previous layer (ReLU grad)
    const deltaNext = new Float64Array(inDim);
    const zPrev = zs[li - 1];
    for (let i = 0; i < inDim; i++) {
      let sum = 0;
      for (let j = 0; j < outDim; j++) sum += W[j][i] * delta[j];
      deltaNext[i] = sum * (zPrev[i] > 0 ? 1 : 0);
    }
    delta = deltaNext;
  }
}

/* ── Adam update step ────────────────────────────────────────── */
function adamStep(weights, adamState, gradW, gradB, t) {
  const β1 = 0.9, β2 = 0.999, ε = 1e-8;
  const bc1 = 1 - Math.pow(β1, t);
  const bc2 = 1 - Math.pow(β2, t);

  for (let li = 0; li < weights.layers.length; li++) {
    const { W, b }  = weights.layers[li];
    const { mW, vW, mb, vb } = adamState[li];
    const gW = gradW[li];
    const gB = gradB[li];

    for (let j = 0; j < W.length; j++) {
      const Wj = W[j];
      const mWj = mW[j], vWj = vW[j], gWj = gW[j];
      for (let i = 0; i < Wj.length; i++) {
        const g = gWj[i];
        mWj[i] = β1 * mWj[i] + (1 - β1) * g;
        vWj[i] = β2 * vWj[i] + (1 - β2) * g * g;
        Wj[i] -= LR * (mWj[i] / bc1) / (Math.sqrt(vWj[i] / bc2) + ε);
      }
      const gj = gB[j];
      mb[j] = β1 * mb[j] + (1 - β1) * gj;
      vb[j] = β2 * vb[j] + (1 - β2) * gj * gj;
      b[j] -= LR * (mb[j] / bc1) / (Math.sqrt(vb[j] / bc2) + ε);
    }
  }
}

/* ── State encoder (matches masterBot.js exactly) ────────────── */
const INPUT_DIM = 149;
function encodeFlat(game, playerIdx) {
  const player = game.players[playerIdx];
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
  return vec;
}

/* ── Self-play ────────────────────────────────────────────────── */
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function runGame(bots) {
  const names = bots.map(b => b.name);
  const game  = engine.createGame(names, { quickstart: true });
  const byName = {};
  for (const b of bots) byName[b.name] = b;

  let turns = 0;
  const pending = []; // { playerIdx, state }

  while (game.phase !== 'gameOver') {
    if (turns++ > 2000) return null; // edge case guard

    let acted = false;
    for (let i = 0; i < bots.length; i++) {
      const gp  = game.players[i];
      const bot = byName[gp.name];
      if (!bot) continue;

      let stateBefore = null;
      try { stateBefore = encodeFlat(game, i); } catch {}

      try { acted = decideBotAction(game, i, bot.personality, bot.difficulty, bot.name); }
      catch { return null; }

      if (acted) {
        if (stateBefore) pending.push({ playerIdx: i, name: gp.name, state: stateBefore });
        if (game.phase !== 'gameOver') engine.declareGameEnd(game, game.currentPlayerIdx);
        break;
      }
    }
    if (!acted && game.phase !== 'gameOver') return null;
  }

  if (!game.standings) return null;
  const winner = game.standings[0].name;

  return pending.map(r => ({
    state:   r.state,
    outcome: r.name === winner ? 1 : 0,
  }));
}

/* ── Main training loop ──────────────────────────────────────── */
async function train() {
  console.log('\nAcquire Master Bot — Online Training');
  console.log(`  LR=${LR}  batch=${BATCH_SIZE}  save_every=${SAVE_EVERY}`);
  if (TIME_LIMIT !== Infinity) console.log(`  Time limit: ${TIME_LIMIT/3600000}h`);
  console.log('');

  const weights  = loadOrInit();
  const adam     = initAdam(weights);

  let t         = 0;   // Adam step counter
  let gamesTotal = 0;
  let errors    = 0;
  let totalLoss = 0;
  let lossCnt   = 0;
  let wins      = {};
  for (const b of BOTS) wins[b.name] = 0;
  const t0 = Date.now();

  // Pre-allocate gradient buffers (reused every batch)
  const gradW = weights.layers.map(({ W }) => W.map(row => new Float64Array(row.length)));
  const gradB = weights.layers.map(({ b }) => new Float64Array(b.length));

  while (Date.now() - t0 < TIME_LIMIT) {
    // Collect one batch of games
    const batch = [];
    for (let g = 0; g < BATCH_SIZE; g++) {
      const bots    = shuffle(BOTS);
      const records = runGame(bots);
      gamesTotal++;
      if (!records) { errors++; continue; }
      const winner = bots[0].name; // already encoded in records outcome
      for (const rec of records) batch.push(rec);
    }

    if (batch.length === 0) continue;

    // Zero gradients
    for (let li = 0; li < gradW.length; li++) {
      for (const row of gradW[li]) row.fill(0);
      gradB[li].fill(0);
    }

    // Accumulate gradients over batch
    let batchLoss = 0;
    for (const { state, outcome } of batch) {
      const { yHat, acts, zs } = forwardFull(weights, state);
      // BCE loss: -[y*log(p) + (1-y)*log(1-p)]
      batchLoss += -(outcome * Math.log(yHat + 1e-9) + (1 - outcome) * Math.log(1 - yHat + 1e-9));
      backprop(weights, acts, zs, outcome, gradW, gradB);
    }

    // Scale gradients by batch size and step
    const scale = 1 / batch.length;
    for (let li = 0; li < gradW.length; li++) {
      for (const row of gradW[li]) for (let i = 0; i < row.length; i++) row[i] *= scale;
      for (let i = 0; i < gradB[li].length; i++) gradB[li][i] *= scale;
    }

    t++;
    adamStep(weights, adam, gradW, gradB, t);

    totalLoss += batchLoss / batch.length;
    lossCnt++;

    // Progress log every 10 batches
    if (t % 10 === 0) {
      const elapsedS = ((Date.now() - t0) / 1000).toFixed(0);
      const avgLoss  = (totalLoss / lossCnt).toFixed(4);
      process.stdout.write(
        `\r  step=${t} games=${gamesTotal} loss=${avgLoss} elapsed=${elapsedS}s errors=${errors}   `
      );
      totalLoss = 0; lossCnt = 0;

      if (STATS_FILE) {
        try {
          fs.writeFileSync(STATS_FILE, JSON.stringify({
            step: t, gamesTotal, errors,
            avgLoss: parseFloat(avgLoss),
            elapsedSecs: parseInt(elapsedS),
            updatedAt: new Date().toISOString(),
          }));
        } catch {}
      }
    }

    // Save weights periodically
    if (gamesTotal % SAVE_EVERY < BATCH_SIZE) {
      saveWeights(weights);
      process.stdout.write(`\n  [saved] step=${t} games=${gamesTotal}\n`);
    }

    // Allow event loop to breathe (drain write streams etc)
    await new Promise(resolve => setImmediate(resolve));
  }

  // Final save
  saveWeights(weights);
  const elapsedMin = ((Date.now() - t0) / 60000).toFixed(1);
  console.log(`\n\nDone. ${gamesTotal} games, ${t} gradient steps, ${elapsedMin}min`);
  console.log(`Weights saved to ${WEIGHTS_PATH}`);
}

train().catch(err => { console.error('Fatal:', err); process.exit(1); });
