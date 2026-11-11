/**
 * ai/searchBot.js — "Expert" bot: determinized Monte Carlo search.
 *
 * For each important decision (tile placement, stock purchase, merger
 * sell/trade/hold) the bot:
 *   1. Lists a handful of sensible candidate moves (always including the
 *      hard heuristic's own choice).
 *   2. Repeatedly: re-deals every tile it CAN'T see (opponents' hands + bag)
 *      at random, applies a candidate, and plays the rest of the game out with
 *      the hard heuristic for every seat.
 *   3. Picks the candidate with the best average result.
 *
 * It never looks at hidden information — the re-deal is what keeps it fair.
 * Search time is bounded by `budgetMs`, so strength scales with CPU.
 *
 * Pure & synchronous: safe to call from a worker thread (see botWorker.js).
 */

const engine = require('../gameEngine');

let _botAI = null;
const botAI = () => (_botAI || (_botAI = require('../botAI')));

const PERSONALITY = { Aria: 'balanced', Rex: 'focused', Nova: 'diversified', Colt: 'focused', Vera: 'balanced' };
const MAX_PLAYOUT_STEPS = 3000;

// Copy a game for simulation. Hand-written for the game's known shape —
// several times faster than JSON round-tripping, which matters because the
// search copies the game for every imagined future. Logs aren't copied.
function clone(game) {
  const c = {};
  for (const k in game) {
    const v = game[k];
    if (k === 'log') c.log = [];
    else if (k === 'board') c.board = v.map(r => r.slice());
    else if (k === 'players') c.players = v.map(p => ({ ...p, stocks: { ...p.stocks }, tiles: p.tiles.slice() }));
    else if (k === 'chains') { const o = {}; for (const ch in v) o[ch] = { ...v[ch], tiles: v[ch].tiles.slice() }; c.chains = o; }
    else if (Array.isArray(v) && (v.length === 0 || typeof v[0] !== 'object')) c[k] = v.slice();
    else if (v && typeof v === 'object') c[k] = JSON.parse(JSON.stringify(v));
    else c[k] = v;
  }
  return c;
}

/* ── Fair re-deal of hidden tiles ──────────────────────────────── */
function determinize(sim, me) {
  const unseen = [...sim.tileBag];
  const handSizes = sim.players.map(p => p.tiles.length);
  sim.players.forEach((p, i) => { if (i !== me) unseen.push(...p.tiles); });
  for (let i = unseen.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [unseen[i], unseen[j]] = [unseen[j], unseen[i]];
  }
  sim.players.forEach((p, i) => { if (i !== me) p.tiles = unseen.splice(0, handSizes[i]); });
  sim.tileBag = unseen;
}

/* ── What each player would have if the game ended right now ─────
 * cash + every share sold at today's price + the majority/minority bonuses
 * each chain on the board would pay (same split rules as the engine).
 * Direct arithmetic, no copying — fast enough to call inside the search. */
function worthNow(sim) { return engine.worthIfEndedNow(sim); }

/* Score a position that isn't finished: a smooth "chance I'm ahead" from my
 * worth-if-it-ended-now versus the best opponent's, on the same 0..1 scale
 * as score() for finished games. */
function evalScore(sim, me, scale = 6000) {
  const w = worthNow(sim);
  let best = -Infinity;
  w.forEach((v, i) => { if (i !== me && v > best) best = v; });
  const d = w[me] - best;
  const p = 1 / (1 + Math.exp(-d / scale));
  return 0.75 * p + 0.25 * (Math.tanh(d / 15000) + 1) / 2;
}

/* "Quick-thinking" tile choice for the next few turns of a future: try each
 * playable tile, let the heuristic finish whatever it triggers (naming a
 * chain, merger choices), and keep the tile that leaves this player best off
 * against the leader. No futures inside futures — just one step ahead. */
function greedyTile(sim, actor) {
  const { decideBotAction } = botAI();
  const tiles = engine.getPlayableTiles(sim).playable;
  if (tiles.length <= 1) return tiles[0] || null;
  let bestTile = null, bestVal = -Infinity;
  for (const t of tiles) {
    const s2 = clone(sim);
    if (engine.placeTile(s2, actor, t).error) continue;
    for (let k = 0; k < 12 && s2.phase !== 'gameOver' && !(s2.phase === 'buyStock' && s2.currentPlayerIdx === actor) && s2.currentPlayerIdx === actor; k++) {
      const a = s2.phase === 'mergerDecision' && s2.pendingMerger ? s2.pendingMerger.decidingPlayer : s2.currentPlayerIdx;
      const nm = s2.players[a].name;
      try { decideBotAction(s2, a, PERSONALITY[nm] || 'balanced', 'hard', nm, { heuristicOnly: true }); } catch { break; }
    }
    const w = worthNow(s2);
    let other = -Infinity;
    w.forEach((v, i) => { if (i !== actor && v > other) other = v; });
    const v = w[actor] - other + Math.random();        // tiny jitter breaks ties
    if (v > bestVal) { bestVal = v; bestTile = t; }
  }
  return bestTile;
}

/* "Quick-thinking" buying for the next few turns of a future: among a few
 * sensible purchases, pick the one that most improves this player's standing
 * if the game ended now (buying itself is cash-neutral, so this is really
 * "which purchase wins me the most bonus position"). */
function greedyBuy(sim, actor) {
  const p = sim.players[actor];
  const issued = ch => sim.players.reduce((s, q) => s + (q.stocks[ch] || 0), 0);
  const opts = [{}];
  for (const ch of engine.HOTEL_CHAINS) {
    if (!sim.chains[ch].active) continue;
    const price = engine.stockPrice(ch, sim.chains[ch].tiles.length), room = 25 - issued(ch);
    for (const k of [1, 2, 3]) if (k <= room && k * price <= p.cash) opts.push({ [ch]: k });
  }
  if (opts.length === 1) return {};
  const base = worthNow(sim);
  let best = {}, bestVal = 0;
  for (const o of opts) {
    const [ch, k] = Object.entries(o)[0] || [];
    if (!ch) continue;
    p.stocks[ch] = (p.stocks[ch] || 0) + k;              // try it in place, then undo
    const w = worthNow(sim);
    p.stocks[ch] -= k;
    let other = -Infinity; w.forEach((v, i) => { if (i !== actor && v > other) other = v; });
    let otherB = -Infinity; base.forEach((v, i) => { if (i !== actor && v > otherB) otherB = v; });
    const cost = k * engine.stockPrice(ch, sim.chains[ch].tiles.length);
    const gain = (w[actor] - cost - other) - (base[actor] - otherB);
    if (gain > bestVal + 1) { bestVal = gain; best = o; }
  }
  return best;
}

/* ── Play the future out ─────────────────────────────────────────
 * Default: to the end of the game with the Easy heuristic for every seat.
 * opts.horizon:  stop after this many turns and score the position instead.
 * opts.greedyTurns: for this many turns, players pick tiles one step ahead
 *                   (greedyTile) instead of by the heuristic's rules.
 * Returns 'end' | 'cut' | false (stuck). */
function playout(sim, policy = 'hard', opts = {}) {
  const { decideBotAction } = botAI();
  const start = sim.turnNumber || 0;
  const horizon = opts.horizon || 0, greedyTurns = opts.greedyTurns || 0;
  for (let steps = 0; sim.phase !== 'gameOver' && steps < MAX_PLAYOUT_STEPS; steps++) {
    if (horizon && sim.turnNumber - start >= horizon) return 'cut';
    const actor = sim.phase === 'mergerDecision' && sim.pendingMerger
      ? sim.pendingMerger.decidingPlayer : sim.currentPlayerIdx;
    const name = sim.players[actor].name;
    const before = sim.log.length + '|' + sim.phase + '|' + sim.currentPlayerIdx + '|' + sim.turnNumber;
    if (greedyTurns && sim.phase === 'placeTile' && sim.turnNumber - start < greedyTurns) {
      const t = greedyTile(sim, actor);
      if (t && !engine.placeTile(sim, actor, t).error) continue;
    }
    if (opts.greedyBuys && greedyTurns && sim.phase === 'buyStock' && sim.turnNumber - start < greedyTurns) {
      const b = greedyBuy(sim, actor);
      if (!engine.buyStock(sim, actor, b).error) continue;
    }
    try { decideBotAction(sim, actor, PERSONALITY[name] || 'balanced', policy, name, { heuristicOnly: true }); } catch { return false; }
    if (before === sim.log.length + '|' + sim.phase + '|' + sim.currentPlayerIdx + '|' + sim.turnNumber
        && sim.phase !== 'mergerDecision') return false;
  }
  return sim.phase === 'gameOver' ? 'end' : false;
}
// Score of a finished or cut-off future
function futureValue(res, sim, me, scoreFn, opts) {
  if (res === 'end') return scoreFn(sim, me);
  if (res === 'cut') return evalScore(sim, me, opts.evalScale);
  return 0.25;
}

/* Seeded RNG so every candidate faces exactly the same simulated future
 * (the heuristic's few random choices included). */
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* Rank-based score: 1 for first, 0 for last, plus a small margin term. */
function rankScore(sim, me) {
  const myName = sim.players[me].name;
  const st = sim.standings || [];
  const idx = st.findIndex(s => s.name === myName);
  if (idx < 0) return 0;
  const n = Math.max(2, st.length);
  const bestOther = Math.max(...st.filter(s => s.name !== myName).map(s => s.cash));
  const margin = Math.tanh((st[idx].cash - bestOther) / 15000);
  return 0.5 * (idx === 0 ? 1 : 0) + 0.3 * (n - 1 - idx) / (n - 1) + 0.2 * (margin + 1) / 2;
}

/* Score in [0,1]: mostly "did I win", plus a margin term so the search
 * prefers bigger wins / smaller losses when win rates are close. */
function score(sim, me) {
  const myName = sim.players[me].name;
  const st = sim.standings || [];
  const mine = st.find(s => s.name === myName);
  if (!mine) return 0;
  const bestOther = Math.max(...st.filter(s => s.name !== myName).map(s => s.cash));
  const margin = Math.tanh((mine.cash - bestOther) / 15000); // -1..1
  return 0.75 * (st[0].name === myName ? 1 : 0) + 0.25 * (margin + 1) / 2;
}

/* ── Candidate generators ──────────────────────────────────────── */
function tileCandidates(game) {
  return engine.getPlayableTiles(game).playable.map(t => ({ type: 'placeTile', tile: t }));
}

function buyCandidates(game, me, heuristicChoice, wide = false) {
  const p = game.players[me];
  const issued = ch => game.players.reduce((s, q) => s + (q.stocks[ch] || 0), 0);
  const chains = engine.HOTEL_CHAINS.filter(ch => game.chains[ch].active).map(ch => ({
    ch, price: engine.stockPrice(ch, game.chains[ch].tiles.length), room: 25 - issued(ch),
  })).filter(c => c.price > 0 && c.room > 0 && c.price <= p.cash);

  const seen = new Map();
  const add = purchases => {
    let cost = 0, n = 0;
    for (const [ch, k] of Object.entries(purchases)) {
      const c = chains.find(x => x.ch === ch);
      if (!c || k > c.room) return;
      cost += c.price * k; n += k;
    }
    if (n > 3 || cost > p.cash) return;
    const key = Object.entries(purchases).filter(([, k]) => k > 0).sort().map(([c, k]) => c + k).join(',');
    if (!seen.has(key)) seen.set(key, { type: 'buyStock', purchases });
  };

  add(heuristicChoice || {});
  add({});
  for (const c of chains) {
    for (const k of [3, 2, 1]) add({ [c.ch]: k });
  }
  // 2+1 splits between the two chains we're most invested in
  const top = [...chains].sort((a, b) => (p.stocks[b.ch] || 0) - (p.stocks[a.ch] || 0)).slice(0, 3);
  for (const a of top) for (const b of top) if (a !== b) add({ [a.ch]: 2, [b.ch]: 1 });
  if (wide) {
    // Shares needed to take (or keep) first place, and to reach second place
    const others = ch => game.players.map((q, i) => i === me ? 0 : (q.stocks[ch] || 0)).sort((x, y) => y - x);
    for (const c of chains) {
      const mine = p.stocks[c.ch] || 0, o = others(c.ch);
      const toFirst = Math.max(0, o[0] + 1 - mine), toSecond = Math.max(0, (o[1] || 0) + 1 - mine);
      for (const need of [toFirst, toSecond]) if (need >= 1 && need <= 3) add({ [c.ch]: need });
    }
    // Spread buys: 1+1 and 1+1+1 across the most relevant chains
    const rel = [...chains].sort((a, b) => ((p.stocks[b.ch] || 0) - (p.stocks[a.ch] || 0)) || (a.price - b.price)).slice(0, 4);
    for (let i = 0; i < rel.length; i++) for (let j = i + 1; j < rel.length; j++) {
      add({ [rel[i].ch]: 1, [rel[j].ch]: 1 });
      for (let k = j + 1; k < rel.length; k++) add({ [rel[i].ch]: 1, [rel[j].ch]: 1, [rel[k].ch]: 1 });
    }
  }
  return [...seen.values()];
}

function mergerCandidates(game, me, heuristicChoice) {
  const pm = game.pendingMerger;
  const defunct = pm.defunctChains[pm.currentDefunctIdx];
  const held = game.players[me].stocks[defunct] || 0;
  const issued = game.players.reduce((s, q) => s + (q.stocks[pm.survivor] || 0), 0);
  const maxTrade = Math.min(Math.floor(held / 2) * 2, (25 - issued) * 2);
  const seen = new Map();
  const add = (sell, trade) => {
    if (sell < 0 || trade < 0 || sell + trade > held || trade % 2) return;
    seen.set(sell + ',' + trade, { type: 'mergerDecision', sell, trade });
  };
  if (heuristicChoice) add(heuristicChoice.sell || 0, heuristicChoice.trade || 0);
  add(held, 0);                       // sell all
  add(0, 0);                          // hold all
  add(held - maxTrade, maxTrade);     // trade max, sell rest
  add(0, maxTrade);                   // trade max, hold rest
  const half = Math.floor(maxTrade / 4) * 2;
  add(held - half, half);
  return [...seen.values()];
}

function apply(game, me, a) {
  switch (a.type) {
    case 'chooseChain':          return engine.chooseChain(game, me, a.chain);
    case 'chooseMergerSurvivor': return engine.chooseMergerSurvivor(game, me, a.chain);
    case 'placeTile':      return engine.placeTile(game, me, a.tile);
    case 'buyStock':       return engine.buyStock(game, me, a.purchases);
    case 'mergerDecision': return engine.mergerDecision(game, me, { sell: a.sell, trade: a.trade });
  }
  return { error: 'unknown action' };
}

/* ── Paired search: every surviving candidate is played out in the SAME
 * simulated future (same re-deal, same tile draws, same random choices).
 * Comparing candidates on identical futures cancels most of the luck, so
 * far fewer playouts are needed to tell a good move from a bad one.
 * Successive halving drops the weaker half at 4, 8, 16, ... futures. ── */
function pairedSearch(game, me, candidates, budgetMs, maxPlayouts, policy, scoreFn, popts = {}) {
  const deadline = Date.now() + budgetMs;
  const totals = candidates.map(() => 0);
  let alive = candidates.map((_, i) => i);
  let worlds = 0, playouts = 0, nextCut = 4;
  const realRandom = Math.random;
  try {
    while (Date.now() < deadline && playouts < maxPlayouts) {
      const world = clone(game);
      determinize(world, me);
      const seed = (realRandom() * 2 ** 31) | 0;
      const results = [];
      for (const i of alive) {
        const sim = clone(world);
        Math.random = mulberry32(seed);
        const r = apply(sim, me, candidates[i]);
        let v;
        if (r && r.error) v = -1000;            // illegal → never pick
        else v = futureValue(playout(sim, policy, popts), sim, me, scoreFn, popts);
        Math.random = realRandom;
        results.push([i, v]);
        playouts++;
      }
      // Only count complete worlds so every survivor has identical samples
      if (results.length < alive.length) break;
      for (const [i, v] of results) totals[i] += v;
      worlds++;
      if (worlds >= nextCut && alive.length > 2) {
        alive.sort((a, b) => totals[b] - totals[a]);
        alive = alive.slice(0, Math.ceil(alive.length / 2));
        nextCut *= 2;
      }
    }
  } finally {
    Math.random = realRandom;
  }
  let pick = alive[0];
  for (const i of alive) if (totals[i] > totals[pick]) pick = i;
  return candidates[pick];
}

/* ── Search: UCB1 over candidates, bounded by time ─────────────── */
function search(game, me, candidates, budgetMs, maxPlayouts, policy, opts = {}) {
  if (candidates.length <= 1) return candidates[0] || null;
  const scoreFn = opts.scoreMode === 'rank' ? rankScore : score;
  if (opts.paired) return pairedSearch(game, me, candidates, budgetMs, maxPlayouts, policy, scoreFn, opts);
  const stats = candidates.map(() => ({ n: 0, total: 0 }));
  const deadline = Date.now() + budgetMs;
  let total = 0;

  // Every candidate gets a couple of samples first
  const runOne = i => {
    const sim = clone(game);
    determinize(sim, me);
    const r = apply(sim, me, candidates[i]);
    if (r && r.error) { stats[i].n += 1000; stats[i].total -= 1000; return; } // illegal → never pick
    const res = playout(sim, policy, opts);
    stats[i].n++; total++;
    stats[i].total += futureValue(res, sim, me, scoreFn, opts);
  };
  for (let round = 0; round < 2; round++)
    for (let i = 0; i < candidates.length && Date.now() < deadline; i++) runOne(i);

  let checkAt = opts.earlyStop ? Math.max(60, candidates.length * 12) : Infinity;
  while (Date.now() < deadline && total < maxPlayouts) {
    if (total >= checkAt) {
      // Stop early once the leader is clearly ahead of the runner-up
      // (scores are 0..1, spread ~0.3, so the std. error of a mean is ~0.3/sqrt(n))
      checkAt = total + 40;
      const order = stats.map((x, i) => i).filter(i => stats[i].n > 0).sort((a, b) => stats[b].total / stats[b].n - stats[a].total / stats[a].n);
      if (order.length >= 2) {
        const a = stats[order[0]], b = stats[order[1]];
        const gap = a.total / a.n - b.total / b.n, se = 0.3 * Math.sqrt(1 / a.n + 1 / b.n);
        if (a.n >= 40 && gap > 3 * se) break;
      }
    }
    let best = 0, bestUcb = -Infinity;
    for (let i = 0; i < stats.length; i++) {
      const s = stats[i];
      const ucb = s.n === 0 ? Infinity : s.total / s.n + 0.5 * Math.sqrt(Math.log(total + 1) / s.n);
      if (ucb > bestUcb) { bestUcb = ucb; best = i; }
    }
    runOne(best);
  }

  // Most-visited = most robust choice
  let pick = 0;
  for (let i = 1; i < stats.length; i++) if (stats[i].n > stats[pick].n) pick = i;
  return candidates[pick];
}

/**
 * Decide an action for `me`, or return null to let the hard heuristic handle
 * this phase (chain naming, tied-merger survivor, passing).
 */
function decideSearchAction(game, me, opts = {}) {
  const budgetMs    = opts.budgetMs    ?? 800;
  const maxPlayouts = opts.maxPlayouts ?? 4000;
  const policy      = opts.policy ?? 'hard';
  const name = game.players[me].name;
  const personality = PERSONALITY[name] || 'balanced';

  if (game.phase === 'placeTile' && game.currentPlayerIdx === me) {
    const c = tileCandidates(game);
    if (c.length === 0) return null; // let heuristic pass
    return search(game, me, c, budgetMs, maxPlayouts, policy, opts);
  }

  if (game.phase === 'buyStock' && game.currentPlayerIdx === me) {
    // Ask the hard heuristic what it would buy so that's always a candidate
    const h = clone(game);
    let heuristic = {};
    try {
      const traits = { mergerSeeking: 0, riskAppetite: 0, chainLoyalty: 0 };
      heuristic = botAI().decideBotBuyStock(h, me, personality, 'hard', traits, 1.0) || {};
    } catch {}
    return search(game, me, buyCandidates(game, me, heuristic, !!opts.wideBuys), budgetMs, maxPlayouts, policy, opts);
  }

  if (game.phase === 'mergerDecision' && game.pendingMerger && game.pendingMerger.decidingPlayer === me) {
    const held = game.players[me].stocks[game.pendingMerger.defunctChains[game.pendingMerger.currentDefunctIdx]] || 0;
    if (held === 0) return { type: 'mergerDecision', sell: 0, trade: 0 };
    return search(game, me, mergerCandidates(game, me, null), budgetMs, maxPlayouts, policy, opts);
  }

  // Full search (Master): also search which chain to found and which chain
  // survives a tied merger instead of leaving them to the heuristic.
  if (opts.full && game.phase === 'chooseChain' && game.currentPlayerIdx === me) {
    // Chains in the same price tier are interchangeable — one per tier is enough.
    const seenTier = new Set(), c = [];
    for (const ch of engine.HOTEL_CHAINS) {
      if (game.chains[ch].active) continue;
      const tier = engine.stockPrice(ch, 2) + '|' + engine.stockPrice(ch, 41);
      if (seenTier.has(tier)) continue;
      seenTier.add(tier); c.push({ type: 'chooseChain', chain: ch });
    }
    return search(game, me, c, budgetMs, maxPlayouts, policy, opts);
  }
  if (opts.full && game.phase === 'chooseMergerSurvivor' && game.currentPlayerIdx === me) {
    const c = (game.pendingMerger.tiedChains || []).map(ch => ({ type: 'chooseMergerSurvivor', chain: ch }));
    return search(game, me, c, budgetMs, maxPlayouts, policy, opts);
  }

  return null;
}

// Thinking budget per difficulty code. Codes are kept from the old bots so
// saved settings and Finance Road levels keep working:
//   'hard'   → Easy button   — plain heuristic, no search (not listed here)
//   'expert' → Medium button — light search
//   'mcts'   → Hard button   — solid search
//   'master' → Master button — as much search as fits in the time budget
//
// Benchmarks (GitHub runners, 4-player games, 25% = no better than opponents):
//   Medium vs 3x Easy   56% (200 games)    Hard vs 3x Medium  56% (200)
//   1.5s Master vs 3x Hard 35% (200)        Master vs 3x Easy  94% (200)
//   3s vs 1.5s Master   31% (160)  → Master uses 3s
//   searching chain naming / tied survivor: 24% vs Master (160) → no gain, off
//   stronger 'expert' heuristic for playouts: 24% vs Hard (240) → no gain, off
//   paired futures + successive halving: 26% vs Master (160) → no gain, off
//   rank-based scoring: 28% vs Master (160) → within noise, off
//   paired + rank: 18% vs Master (160) → worse, off
//   engine speedups (precomputed coords/neighbors): ~55% more playouts per second
//
// Nov 2026 round (see ai/BOT_IMPROVEMENT_NOTES.md): each imagined future now
// uses "quick-thinking" tile choices for its first 16 turns (every player picks
// the tile that best improves their worth-if-the-game-ended-now) and is scored
// directly after 24 turns. Live levels vs the previous versions:
//   new Master vs 3x old Master, 3 s: 38.6% (114 games); at 1 s: 38% (200)
//   new Hard vs 3x old Hard: 35% (200)   new Medium vs 3x old Medium: 34.5% (200)
const QUICK = { horizon: 24, greedyTurns: 16 };
const SEARCH_LEVELS = {
  expert: { budgetMs: 150,  maxPlayouts: 40,    ...QUICK },
  mcts:   { budgetMs: 400,  maxPlayouts: 200,   ...QUICK },
  master: { budgetMs: 3000, maxPlayouts: 16000, ...QUICK },
  // Experiments for ai/benchmark.js only (not selectable in the UI):
  'lab-xp':   { budgetMs: 400,  maxPlayouts: 200, policy: 'expert' },
  'lab-full': { budgetMs: 3000, maxPlayouts: 16000, full: true },
  'lab-pair': { budgetMs: 3000, maxPlayouts: 16000, paired: true },            // paired futures + halving
  'lab-rank': { budgetMs: 3000, maxPlayouts: 16000, scoreMode: 'rank' },       // rank-based scoring
  // Nov 2026 round — compared at 1 s per move (fast to test), winner confirmed at 3 s
  'lab-m1':   { budgetMs: 1000, maxPlayouts: 16000 },                                   // today's Master, 1 s
  'lab-t1':   { budgetMs: 1000, maxPlayouts: 16000, horizon: 20 },                      // futures cut at 20 turns
  'lab-tg1':  { budgetMs: 1000, maxPlayouts: 16000, horizon: 20, greedyTurns: 8 },      // + quick-thinking first 2 rounds
  'lab-tg1b': { budgetMs: 1000, maxPlayouts: 16000, horizon: 20, greedyTurns: 4 },      // + quick-thinking first round
  'lab-w1':   { budgetMs: 1000, maxPlayouts: 16000, horizon: 20, wideBuys: true },      // cut futures + more buy options
  'lab-e1':   { budgetMs: 1000, maxPlayouts: 16000, horizon: 20, earlyStop: true },     // cut futures + stop early when clear
  'lab-h12':  { budgetMs: 1000, maxPlayouts: 16000, horizon: 12 },                      // look 12 turns ahead
  'lab-h32':  { budgetMs: 1000, maxPlayouts: 16000, horizon: 32 },                      // look 32 turns ahead
  'lab-s3':   { budgetMs: 1000, maxPlayouts: 16000, horizon: 20, evalScale: 3000 },     // sharper position scoring
  'lab-g1':   { budgetMs: 1000, maxPlayouts: 16000, greedyTurns: 8 },                   // quick-thinking 2 rounds, full futures
  'lab-g16':  { budgetMs: 1000, maxPlayouts: 16000, greedyTurns: 16 },                  // quick-thinking 4 rounds, full futures
  'lab-tg16': { budgetMs: 1000, maxPlayouts: 16000, horizon: 24, greedyTurns: 16 },     // quick-thinking 4 rounds, cut at 24
  'lab-gb':   { budgetMs: 1000, maxPlayouts: 16000, greedyTurns: 8, greedyBuys: true },  // quick-thinking tiles AND buys, full futures
  'lab-hg32': { budgetMs: 1000, maxPlayouts: 16000, horizon: 32, greedyTurns: 8 },     // look 32 ahead + quick-thinking 2 rounds
  'lab-g12':  { budgetMs: 1000, maxPlayouts: 16000, greedyTurns: 12 },                  // quick-thinking 3 rounds, full futures
  'lab-tgw':  { budgetMs: 1000, maxPlayouts: 16000, horizon: 20, greedyTurns: 8, wideBuys: true }, // + more buy options
};

module.exports = { decideSearchAction, applySearchAction: apply, SEARCH_LEVELS, worthNow, playout, evalScore, clone };
