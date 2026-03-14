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
const fs            = require('fs');
const path          = require('path');
const { execSync }  = require('child_process');

/* ── Config ──────────────────────────────────────────────────── */
const BATCH_SIZE  = 200;   // games per gradient step
const SAVE_EVERY  = 1000;  // games between weight saves
const LR          = parseFloat(getArg('--lr', '0.001'));
const TIME_LIMIT  = getArg('--time-limit') ? parseInt(getArg('--time-limit')) * 1000 : Infinity;
const STATS_FILE  = getArg('--stats') || null;

const WEIGHTS_PATH = path.join(__dirname, 'models', 'master_weights.json');
const ARCH = [150, 256, 128, 64, 1]; // 150 = 149 base + 1 tile bag count

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

const S3_BUCKET  = process.env.S3_BUCKET || 'acquire-training-data';
const S3_KEY     = 'master_weights.json';
const LOG_PATH   = getArg('--log') || null;
const MAX_LOG_KB = 256; // keep log under 256KB — truncate oldest lines when exceeded

// Write a log line — if a log file is set, append there and rotate if too big.
// Otherwise just write to stdout.
function log(msg) {
  process.stdout.write(msg);
  if (!LOG_PATH) return;
  try {
    fs.appendFileSync(LOG_PATH, msg);
    const stat = fs.statSync(LOG_PATH);
    if (stat.size > MAX_LOG_KB * 1024) {
      // Keep only the last half of the file
      const content = fs.readFileSync(LOG_PATH, 'utf8');
      const lines   = content.split('\n');
      fs.writeFileSync(LOG_PATH, lines.slice(Math.floor(lines.length / 2)).join('\n'));
    }
  } catch {}
}

function saveWeights(weights) {
  fs.mkdirSync(path.dirname(WEIGHTS_PATH), { recursive: true });
  fs.writeFileSync(WEIGHTS_PATH, JSON.stringify(weights)); // always overwrites same file
  // Upload to S3 so weights survive instance termination and the live
  // server can hot-reload them automatically.
  try {
    execSync(`aws s3 cp "${WEIGHTS_PATH}" s3://${S3_BUCKET}/${S3_KEY}`, { timeout: 30000, stdio: 'pipe' });
    log(`  [s3] uploaded → s3://${S3_BUCKET}/${S3_KEY}\n`);
  } catch (e) {
    log(`  [s3] upload failed (non-fatal): ${e.message}\n`);
  }
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
const INPUT_DIM = 150; // 108 board + 35 chains + 1 myCash + 4 oppCash + 1 bagCount
const BAG_TOTAL = 102; // tiles in bag at game start (108 - 6 quickstart)
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
  vec[vi++] = (game.tileBag ? game.tileBag.length : 0) / BAG_TOTAL; // game phase signal
  return vec;
}

/* ── Exploration: random legal action ────────────────────────── */
// Takes a random legal action for the current player in any phase.
// Returns true if an action was taken.
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
      // Pick 0–3 random affordable stocks
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
                           25 - game.players.reduce((s,p)=>s+(p.stocks[c]||0),0));
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
      // Pick a random survivor from the tied chains
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

/* ── Master bot action: one-step lookahead with current weights ─ */
// Scores each legal action by encoding the resulting state and running
// the value network. Falls back to random for non-critical phases.
function takeMasterAction(game, playerIdx, weights) {
  const phase = game.phase;
  try {
    if (phase === 'placeTile') {
      const { playable } = engine.getPlayableTiles(game);
      if (playable.length === 0) { engine.passTile(game, playerIdx); return true; }
      if (playable.length === 1) { engine.placeTile(game, playerIdx, playable[0]); return true; }
      let best = playable[0], bestScore = -1;
      for (const tile of playable) {
        const sim = JSON.parse(JSON.stringify(game));
        engine.placeTile(sim, playerIdx, tile);
        const { yHat } = forwardFull(weights, encodeFlat(sim, playerIdx));
        if (yHat > bestScore) { bestScore = yHat; best = tile; }
      }
      engine.placeTile(game, playerIdx, best);
      return true;
    }
    if (phase === 'buyStock') {
      // Enumerate a subset of buy options and pick best
      const player = game.players[playerIdx];
      const affordable = engine.HOTEL_CHAINS.filter(c => {
        const ch = game.chains[c];
        return ch.active && engine.stockPrice(c, ch.tiles.length) <= player.cash
            && game.players.reduce((s,p)=>s+(p.stocks[c]||0),0) < 25;
      });
      const options = [{}]; // buy nothing
      for (const c of affordable) {
        const price = engine.stockPrice(c, game.chains[c].tiles.length);
        const maxN  = Math.min(3, Math.floor(player.cash / price));
        for (let n = 1; n <= maxN; n++) options.push({ [c]: n });
      }
      if (options.length === 1) { engine.buyStock(game, playerIdx, {}); return true; }
      let best = {}, bestScore = -1;
      for (const purchases of options) {
        const sim = JSON.parse(JSON.stringify(game));
        engine.buyStock(sim, playerIdx, purchases);
        const { yHat } = forwardFull(weights, encodeFlat(sim, playerIdx));
        if (yHat > bestScore) { bestScore = yHat; best = purchases; }
      }
      engine.buyStock(game, playerIdx, best);
      return true;
    }
  } catch {}
  // For other phases fall back to random (they're rare/low-impact)
  return takeRandomAction(game, playerIdx);
}

/* ── Self-play ────────────────────────────────────────────────── */
const EPS = 0.10; // 10% random exploration — always on, every slot

// Each bot slot independently has this probability of using master weights,
// ramping up as training matures so the bot increasingly faces itself:
//   warmup  (<5k)  : 0%   — learn from heuristic opponents first
//   early   (<15k) : 30%  — ~1.5 master bots/game on average
//   mid     (<30k) : 60%  — ~3 master bots/game (majority master vs master)
//   mature  (30k+) : 85%  — ~4.25/game (near-full self-play)
function masterShareForGame(gamesTotal) {
  if (gamesTotal < 5000)  return 0;
  if (gamesTotal < 15000) return 0.30;
  if (gamesTotal < 30000) return 0.60;
  return 0.85;
}

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// masterSlots: Set of bot indices that will use master weights this game
function runGame(bots, weights, masterSlots) {
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

      // Only record state if this player is acting in a relevant phase
      let stateBefore = null;
      const isActive = game.phase === 'placeTile'
        ? game.currentPlayerIdx === i
        : game.phase === 'mergerDecision'
          ? game.pendingMerger?.decidingPlayer === i
          : game.currentPlayerIdx === i;
      if (isActive) try { stateBefore = encodeFlat(game, i); } catch {}

      // ε-greedy: explore with random action
      if (Math.random() < EPS) {
        acted = takeRandomAction(game, i);
      } else if (masterSlots.has(i) && weights) {
        // Master bot: use learned value network
        acted = takeMasterAction(game, i, weights);
      } else {
        try { acted = decideBotAction(game, i, bot.personality, bot.difficulty, bot.name); }
        catch { return null; }
      }

      if (acted) {
        if (stateBefore) pending.push({ playerIdx: i, name: gp.name, state: stateBefore });
        if (game.phase !== 'gameOver') engine.declareGameEnd(game, game.currentPlayerIdx);
        break;
      }
    }
    if (!acted && game.phase !== 'gameOver') return null;
  }

  if (!game.standings) return null;

  // Ranking loss: normalize each player's final cash across all players.
  // Gives continuous signal (0.0–1.0) instead of binary win/lose.
  const totalCash = game.standings.reduce((s, p) => s + p.cash, 0) || 1;
  const outcomeByName = {};
  for (const p of game.standings) outcomeByName[p.name] = p.cash / totalCash;

  const records = pending.map(r => ({
    state:   r.state,
    outcome: outcomeByName[r.name] ?? 0.2,
  }));

  return { records, standings: game.standings };
}

/* ── Main training loop ──────────────────────────────────────── */
async function train() {
  console.log('\nAcquire Master Bot — Online Training');
  console.log(`  LR=${LR}  batch=${BATCH_SIZE}  save_every=${SAVE_EVERY}`);
  if (TIME_LIMIT !== Infinity) console.log(`  Time limit: ${TIME_LIMIT/3600000}h`);
  console.log('');

  const weights  = loadOrInit();
  const adam     = initAdam(weights);

  let t           = 0;   // Adam step counter
  let gamesTotal  = 0;
  let masterGames = 0;
  let errors      = 0;
  let totalLoss   = 0;
  let lossCnt     = 0;
  let lossHistory = []; // rolling window for sparkline
  const wins      = {};
  const avgCash   = {};
  for (const b of BOTS) { wins[b.name] = 0; avgCash[b.name] = []; }
  const t0 = Date.now();

  // Pre-allocate gradient buffers (reused every batch)
  const gradW = weights.layers.map(({ W }) => W.map(row => new Float64Array(row.length)));
  const gradB = weights.layers.map(({ b }) => new Float64Array(b.length));

  while (Date.now() - t0 < TIME_LIMIT) {
    // Collect one batch of games
    const batch = [];
    for (let g = 0; g < BATCH_SIZE; g++) {
      const bots = shuffle(BOTS);
      // Each slot independently gets master weights with probability that ramps over time
      const pMaster = masterShareForGame(gamesTotal);
      const masterSlots = new Set(bots.map((_, i) => i).filter(() => Math.random() < pMaster));
      if (masterSlots.size > 0) masterGames++;
      const result = runGame(bots, weights, masterSlots);
      gamesTotal++;
      if (!result) { errors++; continue; }
      // Track win rates + avg cash per bot
      wins[result.standings[0].name]++;
      for (const p of result.standings) {
        if (avgCash[p.name]) {
          avgCash[p.name].push(p.cash);
          if (avgCash[p.name].length > 5000) avgCash[p.name].shift();
        }
      }
      for (const rec of result.records) batch.push(rec);
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

    // Progress log + stats every 10 batches
    if (t % 10 === 0) {
      const elapsedS  = ((Date.now() - t0) / 1000).toFixed(0);
      const avgLoss   = (totalLoss / lossCnt).toFixed(4);
      const played    = gamesTotal - errors;
      lossHistory.push(parseFloat(avgLoss));
      if (lossHistory.length > 50) lossHistory.shift(); // keep last 50 points
      totalLoss = 0; lossCnt = 0;

      log(`  step=${t} games=${gamesTotal} loss=${avgLoss} elapsed=${elapsedS}s errors=${errors}\n`);

      const stats = {
        step: t,
        gamesTotal,
        gamesPlayed: played,
        masterGames,
        errors,
        avgLoss: parseFloat(avgLoss),
        lossHistory: lossHistory.slice(),
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
            avgCash: cash.length ? Math.round(cash.reduce((s,v)=>s+v,0)/cash.length) : 0,
          };
        }).sort((a, b) => b.wins - a.wins),
      };

      if (STATS_FILE) {
        try { fs.writeFileSync(STATS_FILE, JSON.stringify(stats)); } catch {}
      }
      // Also push stats to S3 so the dashboard can read them
      try {
        const tmpStats = '/tmp/training_stats.json';
        fs.writeFileSync(tmpStats, JSON.stringify(stats));
        execSync(`aws s3 cp "${tmpStats}" s3://${S3_BUCKET}/training_stats.json`, { timeout: 15000, stdio: 'pipe' });
      } catch {}
    }

    // Save weights periodically
    if (gamesTotal % SAVE_EVERY < BATCH_SIZE) {
      saveWeights(weights);
      log(`  [saved] step=${t} games=${gamesTotal}\n`);
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
