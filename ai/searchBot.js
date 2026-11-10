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

function clone(game) {
  // Logs aren't needed inside the search and dominate clone cost late-game.
  const log = game.log;
  game.log = [];
  const c = JSON.parse(JSON.stringify(game));
  game.log = log;
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

/* ── Play to the end with the hard heuristic ───────────────────── */
function playout(sim) {
  const { decideBotAction } = botAI();
  for (let steps = 0; sim.phase !== 'gameOver' && steps < MAX_PLAYOUT_STEPS; steps++) {
    const actor = sim.phase === 'mergerDecision' && sim.pendingMerger
      ? sim.pendingMerger.decidingPlayer : sim.currentPlayerIdx;
    const name = sim.players[actor].name;
    const before = sim.log.length + '|' + sim.phase + '|' + sim.currentPlayerIdx + '|' + sim.turnNumber;
    try { decideBotAction(sim, actor, PERSONALITY[name] || 'balanced', 'hard', name); } catch { return false; }
    if (before === sim.log.length + '|' + sim.phase + '|' + sim.currentPlayerIdx + '|' + sim.turnNumber
        && sim.phase !== 'mergerDecision') return false;
  }
  return sim.phase === 'gameOver';
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

function buyCandidates(game, me, heuristicChoice) {
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
    case 'placeTile':      return engine.placeTile(game, me, a.tile);
    case 'buyStock':       return engine.buyStock(game, me, a.purchases);
    case 'mergerDecision': return engine.mergerDecision(game, me, { sell: a.sell, trade: a.trade });
  }
  return { error: 'unknown action' };
}

/* ── Search: UCB1 over candidates, bounded by time ─────────────── */
function search(game, me, candidates, budgetMs, maxPlayouts) {
  if (candidates.length <= 1) return candidates[0] || null;
  const stats = candidates.map(() => ({ n: 0, total: 0 }));
  const deadline = Date.now() + budgetMs;
  let total = 0;

  // Every candidate gets a couple of samples first
  const runOne = i => {
    const sim = clone(game);
    determinize(sim, me);
    const r = apply(sim, me, candidates[i]);
    if (r && r.error) { stats[i].n += 1000; stats[i].total -= 1000; return; } // illegal → never pick
    const ok = playout(sim);
    stats[i].n++; total++;
    stats[i].total += ok ? score(sim, me) : 0.25;
  };
  for (let round = 0; round < 2; round++)
    for (let i = 0; i < candidates.length && Date.now() < deadline; i++) runOne(i);

  while (Date.now() < deadline && total < maxPlayouts) {
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
  const name = game.players[me].name;
  const personality = PERSONALITY[name] || 'balanced';

  if (game.phase === 'placeTile' && game.currentPlayerIdx === me) {
    const c = tileCandidates(game);
    if (c.length === 0) return null; // let heuristic pass
    return search(game, me, c, budgetMs, maxPlayouts);
  }

  if (game.phase === 'buyStock' && game.currentPlayerIdx === me) {
    // Ask the hard heuristic what it would buy so that's always a candidate
    const h = clone(game);
    let heuristic = {};
    try {
      const traits = { mergerSeeking: 0, riskAppetite: 0, chainLoyalty: 0 };
      heuristic = botAI().decideBotBuyStock(h, me, personality, 'hard', traits, 1.0) || {};
    } catch {}
    return search(game, me, buyCandidates(game, me, heuristic), budgetMs, maxPlayouts);
  }

  if (game.phase === 'mergerDecision' && game.pendingMerger && game.pendingMerger.decidingPlayer === me) {
    const held = game.players[me].stocks[game.pendingMerger.defunctChains[game.pendingMerger.currentDefunctIdx]] || 0;
    if (held === 0) return { type: 'mergerDecision', sell: 0, trade: 0 };
    return search(game, me, mergerCandidates(game, me, null), budgetMs, maxPlayouts);
  }

  return null;
}

// Thinking budget per difficulty code. Codes are kept from the old bots so
// saved settings and Finance Road levels keep working:
//   'hard'   → Easy button   — plain heuristic, no search (not listed here)
//   'expert' → Medium button — light search
//   'mcts'   → Hard button   — solid search
//   'master' → Master button — as much search as fits in the time budget
// Measured on a 2.3 GHz Xeon: 150 playouts beat 12 playouts 27/30 games;
// 100 playouts beat the plain heuristic 52/60; 400 beat 40 9/12.
const SEARCH_LEVELS = {
  expert: { budgetMs: 150,  maxPlayouts: 40 },
  mcts:   { budgetMs: 400,  maxPlayouts: 200 },
  master: { budgetMs: 1500, maxPlayouts: 8000 },
};

module.exports = { decideSearchAction, applySearchAction: apply, SEARCH_LEVELS };
