#!/usr/bin/env node
/**
 * ai/benchmark.js — measure bot strength using every CPU core you have.
 *
 * Plays 4-player games: one "hero" bot against three opponents, rotating
 * seats. A hero no better than its opponents wins ~25% of the time.
 *
 * Usage (from the repo root):
 *   node ai/benchmark.js                 # default: 40 games per matchup
 *   node ai/benchmark.js --games 100     # more games = tighter numbers
 *   node ai/benchmark.js --quick         # 12 games per matchup, smoke test
 *   node ai/benchmark.js --matchups master:mcts,mcts:hard
 *   node ai/benchmark.js --json out.json # also save raw counts (used by CI shards)
 *   node ai/benchmark.js --merge dir/    # combine shard JSON files into one report
 *
 * Writes ai/benchmark-report.md — paste or attach it back to Claude.
 * Nothing is sent anywhere; it only reads the game code and writes that file.
 */
const os = require('os');
const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// difficulty codes → button names in the UI
const LABEL = { hard: 'Easy', expert: 'Medium', mcts: 'Hard', master: 'Master', easy: 'old-easy', medium: 'old-medium' };

// Default ladder: each rung should beat the one below it.
// Override with --matchups master:mcts,mcts:expert  (hero:opponent codes)
const DEFAULT_MATCHUPS = 'expert:hard,mcts:expert,master:mcts,master:hard';
const _mArg = process.argv.indexOf('--matchups');
const MATCHUPS = (_mArg >= 0 ? process.argv[_mArg + 1] : DEFAULT_MATCHUPS)
  .split(',').map(x => { const [hero, opp] = x.split(':'); return { hero, opp }; });

if (isMainThread) {
  const args = process.argv.slice(2);
  const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const games = args.includes('--quick') ? 12 : +arg('--games', 40);
  const cores = Math.max(1, +arg('--cores', os.cpus().length));

  // Build a queue of single games
  const jobs = [];
  MATCHUPS.forEach((m, mi) => { for (let g = 0; g < games; g++) jobs.push({ mi, seat: g % 4, ...m }); });
  // interleave so partial results cover every matchup
  jobs.sort((a, b) => (a.seat - b.seat) || (a.mi - b.mi));

  const results = MATCHUPS.map(() => ({ games: 0, wins: 0, margin: 0, secs: 0, errors: 0 }));
  const t0 = Date.now();
  let next = 0, done = 0;
  console.log(`Running ${jobs.length} games on ${cores} cores (${os.cpus()[0].model})…`);

  const finish = () => {
    const jsonOut = arg('--json', null);
    if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify({ matchups: MATCHUPS, results, cpu: os.cpus()[0].model, cores: os.cpus().length }));
    writeReport(results, games, (Date.now() - t0) / 60000);
    process.exit(0);
  };
  function writeReport(results, games, minutes) {
    const lines = [];
    lines.push(`# Acquire bot benchmark`);
    lines.push(``);
    lines.push(`- Date: ${new Date().toISOString()}`);
    lines.push(`- Machine: ${os.cpus()[0].model}, ${os.cpus().length} cores, Node ${process.version}`);
    lines.push(`- Games per matchup: ${games}, wall time: ${minutes.toFixed(1)} min`);
    lines.push(``);
    lines.push(`| Hero | vs 3× | Wins | Win % | 95% range | Avg margin vs best opp | Sec/game | Errors |`);
    lines.push(`|---|---|---|---|---|---|---|---|`);
    MATCHUPS.forEach((m, i) => {
      const r = results[i], p = r.wins / Math.max(1, r.games);
      const se = Math.sqrt(p * (1 - p) / Math.max(1, r.games)) * 1.96;
      lines.push(`| ${LABEL[m.hero] || m.hero} | ${LABEL[m.opp] || m.opp} | ${r.wins}/${r.games} | ${(p * 100).toFixed(0)}% | ${Math.max(0, (p - se) * 100).toFixed(0)}–${Math.min(100, (p + se) * 100).toFixed(0)}% | $${Math.round(r.margin / Math.max(1, r.games)).toLocaleString()} | ${(r.secs / Math.max(1, r.games)).toFixed(1)} | ${r.errors} |`);
    });
    lines.push(``);
    lines.push(`25% = no better than its opponents.`);
    const out = path.join(__dirname, 'benchmark-report.md');
    fs.writeFileSync(out, lines.join('\n') + '\n');
    console.log('\n' + lines.join('\n'));
    console.log(`\nReport written to ${out}`);
  }

  const mergeDir = arg('--merge', null);
  if (mergeDir) {
    const files = fs.readdirSync(mergeDir, { recursive: true }).filter(f => String(f).endsWith('.json'));
    const merged = MATCHUPS.map(() => ({ games: 0, wins: 0, margin: 0, secs: 0, errors: 0 }));
    for (const f of files) {
      const d = JSON.parse(fs.readFileSync(path.join(mergeDir, String(f)), 'utf8'));
      d.matchups.forEach((m, i) => {
        const k = MATCHUPS.findIndex(x => x.hero === m.hero && x.opp === m.opp);
        if (k < 0) return;
        for (const f2 of Object.keys(merged[k])) merged[k][f2] += d.results[i][f2];
      });
    }
    console.log(`Merged ${files.length} shard files`);
    writeReport(merged, merged[0].games, 0);
    process.exit(0);
  }


  for (let c = 0; c < Math.min(cores, jobs.length); c++) {
    const w = new Worker(__filename, { workerData: {} });
    const feed = () => {
      if (next >= jobs.length) { w.terminate(); return; }
      w.postMessage(jobs[next++]);
    };
    w.on('message', r => {
      const s = results[r.mi];
      s.games++; s.wins += r.win ? 1 : 0; s.margin += r.margin; s.secs += r.secs; s.errors += r.errors;
      done++;
      process.stdout.write(`\r${done}/${jobs.length} games  (${((Date.now() - t0) / 1000).toFixed(0)}s)   `);
      if (done === jobs.length) finish();
      else feed();
    });
    w.on('error', e => { console.error('\nworker error', e); });
    feed();
  }
} else {
  const engine = require('../gameEngine');
  const { decideBotAction } = require('../botAI');
  const NAMES = ['Aria', 'Rex', 'Nova', 'Colt'];
  const PERS = { Aria: 'balanced', Rex: 'focused', Nova: 'diversified', Colt: 'focused' };

  parentPort.on('message', job => {
    const t = Date.now();
    const order = [...NAMES].sort(() => Math.random() - 0.5);
    const game = engine.createGame(order);
    const hero = order[job.seat];
    let steps = 0, errors = 0;
    while (game.phase !== 'gameOver' && steps++ < 4000) {
      const a = game.phase === 'mergerDecision' && game.pendingMerger ? game.pendingMerger.decidingPlayer : game.currentPlayerIdx;
      const nm = game.players[a].name;
      const before = JSON.stringify([game.phase, game.currentPlayerIdx, game.turnNumber, game.log.length, game.pendingMerger]);
      try { decideBotAction(game, a, PERS[nm], nm === hero ? job.hero : job.opp, nm); } catch { errors++; }
      if (before === JSON.stringify([game.phase, game.currentPlayerIdx, game.turnNumber, game.log.length, game.pendingMerger])) {
        errors++;
        try { decideBotAction(game, a, PERS[nm], 'hard', nm); } catch {}
      }
    }
    const st = game.standings || [];
    const mine = (st.find(s => s.name === hero) || { cash: 0 }).cash;
    const best = Math.max(0, ...st.filter(s => s.name !== hero).map(s => s.cash));
    parentPort.postMessage({ mi: job.mi, win: st[0] && st[0].name === hero, margin: mine - best, secs: (Date.now() - t) / 1000, errors });
  });
}
