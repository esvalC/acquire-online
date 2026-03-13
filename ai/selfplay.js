#!/usr/bin/env node
/**
 * ai/selfplay.js — Acquire headless self-play simulator
 *
 * Runs bot-vs-bot games at full speed using the same gameEngine and botAI
 * that power the live server. Use this to:
 *   - Benchmark bot personalities against each other
 *   - Tune difficulty parameters and measure win rates
 *   - Generate training data for future ML-based bots (Master bot)
 *   - Validate engine correctness at scale (no UI noise)
 *
 * Usage:
 *   node ai/selfplay.js                             # 1000 games, default config
 *   node ai/selfplay.js --games 5000                # run 5000 games
 *   node ai/selfplay.js --time-limit 18000          # run for 5 hours (18000s)
 *   node ai/selfplay.js --export ai/data/games.jsonl  # export training data for ML
 *   node ai/selfplay.js --mcts                      # Rex uses MCTS instead of heuristic
 *   node ai/selfplay.js --quiet                     # suppress progress output
 *
 * Training data format (--export):
 *   One JSON object per line. Each line = one decision point:
 *   { phase, playerIdx, playerName, state: {...}, action: {...}, outcome: 1|0 }
 *   outcome = 1 if this player won the game, 0 otherwise.
 *   This is the raw supervised learning data for the Master bot neural network.
 */

const engine  = require('../gameEngine');
const { decideBotAction } = require('../botAI');
const fs      = require('fs');
const path    = require('path');

/* ── Configuration ───────────────────────────────────────────── */

// Bots that play in every self-play game (order is randomized per game)
// To benchmark MCTS: run with --mcts flag to swap Rex to difficulty:'mcts'
const USE_MCTS = process.argv.includes('--mcts');
const BOTS = [
  { name: 'Aria',  personality: 'balanced',    difficulty: USE_MCTS ? 'hard' : 'hard' },
  { name: 'Rex',   personality: 'focused',     difficulty: USE_MCTS ? 'mcts' : 'hard' },
  { name: 'Nova',  personality: 'diversified', difficulty: 'hard' },
  { name: 'Colt',  personality: 'focused',     difficulty: 'hard' },
  { name: 'Vera',  personality: 'balanced',    difficulty: 'hard' },
];

// Shuffle function (Fisher-Yates)
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* ── State encoder (for ML training data) ────────────────────── */
// Encodes the current game state into a compact representation suitable
// for feeding into a neural network. This is the "input feature vector."
function encodeState(game, playerIdx) {
  const player = game.players[playerIdx];
  const n      = game.players.length;

  // Board: 9×12 = 108 cells, each encoded as chain index (0=empty/lone, 1–7=chain)
  const CHAIN_IDX = {};
  engine.HOTEL_CHAINS.forEach((c, i) => { CHAIN_IDX[c] = i + 1; });
  const board = [];
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 12; c++) {
      const cell = game.board[r][c];
      board.push(cell && CHAIN_IDX[cell] ? CHAIN_IDX[cell] : (cell ? -1 : 0));
    }
  }

  // Chain features: for each of 7 chains: [active, size, my_shares, max_opp_shares, price]
  const chains = engine.HOTEL_CHAINS.map(c => {
    const ch      = game.chains[c];
    const size    = ch.tiles.length;
    const price   = engine.stockPrice(c, size);
    const myS     = player.stocks[c] || 0;
    const maxOppS = game.players.filter((_, i) => i !== playerIdx)
                      .reduce((m, p) => Math.max(m, p.stocks[c] || 0), 0);
    return [ch.active ? 1 : 0, size, myS, maxOppS, price / 1000];
  });

  // Player features: my cash, relative cash vs opponents
  const myCash   = player.cash / 6000;
  const oppCash  = game.players.filter((_, i) => i !== playerIdx)
                     .map(p => p.cash / 6000);

  return { board, chains, myCash, oppCash, phase: game.phase };
}

/* ── Run one game ────────────────────────────────────────────── */

/**
 * Plays one full Acquire game with the given bot lineup.
 * Returns { winner, standings, turns, error, records? }
 * If exportRecords=true, also returns an array of training records.
 */
function runGame(bots, opts = {}) {
  const names = bots.map(b => b.name);
  const game  = engine.createGame(names, { quickstart: true });

  const botsByName = {};
  for (const b of bots) botsByName[b.name] = b;

  let turns = 0;
  const MAX_TURNS = 2000;
  const pendingRecords = []; // collected during game, labeled after

  while (game.phase !== 'gameOver') {
    if (turns++ > MAX_TURNS) {
      return { winner: null, standings: null, turns, error: 'max_turns_exceeded' };
    }

    let acted = false;
    for (let i = 0; i < bots.length; i++) {
      const gp  = game.players[i];
      const bot = botsByName[gp.name];
      if (!bot) continue;

      // Capture state + action for training data (only for hard/mcts bots)
      let stateBefore = null;
      if (opts.exportRecords && (bot.difficulty === 'hard' || bot.difficulty === 'mcts')) {
        try { stateBefore = encodeState(game, i); } catch {}
      }

      try {
        acted = decideBotAction(game, i, bot.personality, bot.difficulty, bot.name);
      } catch (err) {
        return { winner: null, standings: null, turns, error: err.message };
      }

      if (acted) {
        if (stateBefore) {
          pendingRecords.push({ playerIdx: i, playerName: gp.name, state: stateBefore });
        }
        // In selfplay bots don't call declareGameEnd, so we trigger it here
        // when conditions are met. No-op if conditions aren't met.
        if (game.phase !== 'gameOver') {
          engine.declareGameEnd(game, game.currentPlayerIdx);
        }
        break;
      }
    }

    if (!acted && game.phase !== 'gameOver') {
      return { winner: null, standings: null, turns, error: 'stuck: ' + game.phase };
    }
  }

  const standings = game.standings;
  const winner    = standings[0].name;

  // Label records with outcome (1 = winner, 0 = loser)
  let records = null;
  if (opts.exportRecords && pendingRecords.length > 0) {
    records = pendingRecords.map(r => ({
      ...r,
      outcome: r.playerName === winner ? 1 : 0,
    }));
  }

  return { winner, standings, turns, error: null, records };
}

/* ── Aggregate results ───────────────────────────────────────── */

/* ── ELO tracking (chess-comparable, K=32) ───────────────────── */
// Each bot starts at 1500 (chess standard). After each game we do pairwise
// updates for every pair of players based on their finish order.
const K = 32;
function expectedScore(ra, rb) { return 1 / (1 + Math.pow(10, (rb - ra) / 400)); }
function updateElo(ratings, standings) {
  // standings is ordered best→worst: index 0 = 1st place
  for (let i = 0; i < standings.length; i++) {
    for (let j = i + 1; j < standings.length; j++) {
      const winner = standings[i].name;
      const loser  = standings[j].name;
      if (!ratings[winner] || !ratings[loser]) continue;
      const rw = ratings[winner], rl = ratings[loser];
      const exp = expectedScore(rw, rl);
      ratings[winner] = Math.round(rw + K * (1 - exp));
      ratings[loser]  = Math.round(rl + K * (0 - (1 - exp)));
    }
  }
}

/**
 * Run games until `totalGames` is reached OR `timeLimitSecs` expires,
 * and print win-rate statistics. Exports training data if exportFile is set.
 * If opts.statsFile is set, writes live JSON stats every 10 games for the dashboard.
 */
async function runBenchmark(totalGames, opts = {}) {
  const wins    = {};
  const podiums = {};
  const cashes  = {};
  const elo     = {};
  let   errors  = 0;
  let   totalTurns = 0;
  let   exportStream = null;
  let   exportedRecords = 0;

  for (const b of BOTS) {
    wins[b.name]    = 0;
    podiums[b.name] = 0;
    cashes[b.name]  = [];
    elo[b.name]     = 1500; // chess starting ELO
  }

  if (opts.exportFile) {
    const dir = path.dirname(opts.exportFile);
    fs.mkdirSync(dir, { recursive: true });
    exportStream = fs.createWriteStream(opts.exportFile, { flags: 'a' });
    if (!opts.quiet) console.log(`  Exporting training data → ${opts.exportFile}`);
  }

  const t0          = Date.now();
  const timeLimitMs = (opts.timeLimitSecs || Infinity) * 1000;
  let   g = 0;

  // Write live stats JSON for the dashboard
  function writeStats() {
    if (!opts.statsFile) return;
    const elapsedMs  = Date.now() - t0;
    const played     = g - errors;
    const remainMs   = timeLimitMs === Infinity ? null : Math.max(0, timeLimitMs - elapsedMs);
    const stats = {
      gamesPlayed:     played,
      gamesTotal:      g,
      errors,
      exportedRecords,
      elapsedSecs:     Math.floor(elapsedMs / 1000),
      remainingSecs:   remainMs === null ? null : Math.floor(remainMs / 1000),
      timeLimitSecs:   opts.timeLimitSecs || null,
      avgTurns:        played > 0 ? +(totalTurns / played).toFixed(1) : 0,
      updatedAt:       new Date().toISOString(),
      bots: BOTS.map(b => {
        const w   = wins[b.name];
        const arr = cashes[b.name];
        return {
          name:       b.name,
          difficulty: b.difficulty,
          wins:       w,
          winPct:     played > 0 ? +((w / played) * 100).toFixed(1) : 0,
          top3Pct:    played > 0 ? +((podiums[b.name] / played) * 100).toFixed(1) : 0,
          avgCash:    arr.length ? Math.round(arr.reduce((s,v)=>s+v,0)/arr.length) : 0,
          elo:        elo[b.name],
        };
      }).sort((a,b) => b.elo - a.elo),
    };
    try { fs.writeFileSync(opts.statsFile, JSON.stringify(stats, null, 2)); } catch {}
  }

  while (g < totalGames) {
    // Check time limit
    if (Date.now() - t0 > timeLimitMs) {
      if (!opts.quiet) console.log(`\n  Time limit reached after ${g} games.`);
      break;
    }

    const bots   = shuffle(BOTS);
    const result = runGame(bots, { exportRecords: !!exportStream });
    g++;

    if (result.error) {
      errors++;
      if (!opts.quiet) console.error(`[game ${g}] ERROR: ${result.error}`);
      continue;
    }

    totalTurns += result.turns;
    wins[result.winner]++;
    updateElo(elo, result.standings);

    for (let i = 0; i < result.standings.length; i++) {
      const { name, cash } = result.standings[i];
      if (i < 3) podiums[name]++;
      cashes[name].push(cash);
    }

    // Write training records — respect backpressure to avoid OOM
    if (exportStream && result.records) {
      for (const rec of result.records) {
        const ok = exportStream.write(JSON.stringify(rec) + '\n');
        exportedRecords++;
        if (!ok) {
          // Buffer full — wait for drain before writing more
          await new Promise(resolve => exportStream.once('drain', resolve));
        }
      }
    }

    if (g % 10 === 0) writeStats();

    if (!opts.quiet && g % 100 === 0) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
      process.stdout.write(`\r  ${g} games | ${elapsed}s elapsed${exportStream ? ` | ${exportedRecords} records` : ''}…`);
    }
  }

  writeStats(); // final write

  if (!opts.quiet) process.stdout.write('\n');
  if (exportStream) { exportStream.end(); }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const played  = g - errors;

  console.log('\n══════════════════════════════════════════════════');
  console.log(`  Acquire Self-Play Results  (${played} games, ${elapsed}s)`);
  if (exportStream) console.log(`  Training records exported: ${exportedRecords}`);
  console.log('══════════════════════════════════════════════════');
  console.log('  Bot           Win%    Top3%   Avg Cash  Difficulty');
  console.log('  ─────────────────────────────────────────────────');

  const sorted = BOTS.slice().sort((a, b) => wins[b.name] - wins[a.name]);
  for (const bot of sorted) {
    const w   = wins[bot.name];
    const p   = podiums[bot.name];
    const arr = cashes[bot.name];
    const avg = arr.length ? Math.round(arr.reduce((s, v) => s + v, 0) / arr.length) : 0;
    const winPct = played > 0 ? ((w / played) * 100).toFixed(1).padStart(5) : '  0.0';
    const podPct = played > 0 ? ((p / played) * 100).toFixed(1).padStart(5) : '  0.0';
    const avgStr = ('$' + avg.toLocaleString()).padStart(9);
    const diff   = bot.difficulty.padEnd(6);
    console.log(`  ${bot.name.padEnd(14)} ${winPct}%  ${podPct}%  ${avgStr}  ${diff}`);
  }

  console.log('  ─────────────────────────────────────────────────');
  console.log(`  Avg turns/game: ${played > 0 ? (totalTurns / played).toFixed(1) : '—'}   Errors: ${errors}`);
  console.log('══════════════════════════════════════════════════\n');
}

/* ── CLI entry point ─────────────────────────────────────────── */

const args = process.argv.slice(2);

function getArg(flag, defaultVal) {
  const idx = args.indexOf(flag);
  if (idx === -1) return defaultVal;
  return args[idx + 1];
}

const nGames        = parseInt(getArg('--games', '999999999'), 10);
const timeLimitSecs = getArg('--time-limit') ? parseInt(getArg('--time-limit'), 10) : null;
const exportFile    = getArg('--export') || null;
const statsFile     = getArg('--stats') || null;
const quiet         = args.includes('--quiet');

// If neither --games nor --time-limit specified, default to 1000 games
const effectiveGames = args.includes('--games') ? nGames : (timeLimitSecs ? 999999999 : 1000);

if (!quiet) {
  console.log(`\nAcquire Self-Play Simulator`);
  if (timeLimitSecs) console.log(`Time limit: ${(timeLimitSecs / 3600).toFixed(1)}h`);
  else console.log(`Games: ${effectiveGames}`);
  console.log(`Bots: ${BOTS.map(b => `${b.name}(${b.difficulty})`).join(', ')}\n`);
}

runBenchmark(effectiveGames, { quiet, timeLimitSecs, exportFile, statsFile })
  .catch(err => { console.error('Fatal:', err); process.exit(1); });
