/**
 * ai/mcts.js — Neural-guided search for Acquire
 *
 * For each legal action, apply it to a cloned state and evaluate the result
 * using the Master Bot value network. Pick the action with the highest
 * predicted win probability.
 *
 * If the v3 weights are not loaded (first run or missing file), falls back to
 * flat Monte Carlo with heuristic rollouts so the bot still works.
 *
 * Performance: ~0.5ms per decision (one forward pass per candidate action),
 * versus ~15ms previously for 5 random rollouts to game end.
 */

const engine = require('../gameEngine');

// Lazy loaders to avoid circular dependencies
let _botAI     = null;
let _masterBot = null;
function getBotAI()     { if (!_botAI)     _botAI     = require('../botAI');     return _botAI; }
function getMasterBot() { if (!_masterBot) _masterBot = require('./masterBot');  return _masterBot; }

const MAX_ACTIONS = 7; // cap on candidates considered

/* ── Game state clone ────────────────────────────────────────── */
function cloneGame(game) {
  return JSON.parse(JSON.stringify(game));
}

/* ── Value-network position evaluator ───────────────────────── */
// Returns a score in [0,1] for the given player after the game state
// has already had an action applied. Uses the v3 value head.
// Falls back to a heuristic estimate if weights aren't available.
function evaluatePosition(game, playerIdx, myName) {
  if (game.phase === 'gameOver') {
    return game.standings?.[0]?.name === myName ? 1 : 0;
  }

  const mb = getMasterBot();
  const weights = mb.loadWeights();
  if (weights) {
    try {
      const { value } = mb.forward(weights, mb.encodeState(game, playerIdx));
      return value[playerIdx];
    } catch {}
  }

  // Fallback: heuristic rollout (original behaviour)
  return rolloutFallback(game, myName);
}

/* ── Fallback: heuristic rollout to game end ─────────────────── */
// Only used when v3 weights aren't loaded.
const MAX_ROLLOUT_TURNS = 2000;
function rolloutFallback(game, myName) {
  const { decideBotAction } = getBotAI();
  const KNOWN_PERSONALITY = { Aria:'balanced', Rex:'focused', Nova:'diversified', Colt:'focused', Vera:'balanced' };
  const bots = game.players.map(p => ({
    name: p.name,
    personality: KNOWN_PERSONALITY[p.name] || 'balanced',
  }));

  let t = 0;
  while (game.phase !== 'gameOver' && t++ < MAX_ROLLOUT_TURNS) {
    let acted = false;
    for (let i = 0; i < bots.length; i++) {
      try {
        acted = decideBotAction(game, i, bots[i].personality, 'medium', bots[i].name);
        if (acted) break;
      } catch { break; }
    }
    if (!acted && game.phase !== 'gameOver') break;
  }
  if (game.phase !== 'gameOver') return 0;
  return game.standings?.[0]?.name === myName ? 1 : 0;
}

/* ── Helper ──────────────────────────────────────────────────── */
function issuedShares(game, chain) {
  return game.players.reduce((s, p) => s + (p.stocks[chain] || 0), 0);
}

/* ── Candidate action generators ─────────────────────────────── */

function candidateMergerDecisions(game, playerIdx) {
  const pm      = game.pendingMerger;
  const defunct = pm.defunctChains[pm.currentDefunctIdx];
  const held    = game.players[playerIdx].stocks[defunct] || 0;
  if (held === 0) return [{ type: 'mergerDecision', sell: 0, trade: 0 }];

  const survivorRoom = 25 - issuedShares(game, pm.survivor);
  const maxTrade     = Math.min(Math.floor(held / 2) * 2, survivorRoom * 2);

  const candidates = new Map();
  const add = (sell, trade) => candidates.set(`${sell},${trade}`, { type: 'mergerDecision', sell, trade });
  add(held, 0);  // sell all
  add(0, 0);     // hold all
  if (maxTrade > 0) {
    add(held - maxTrade, maxTrade);
    const half = Math.floor(maxTrade / 4) * 2;
    if (half > 0) add(held - half, half);
  }
  return [...candidates.values()];
}

function legalTileActions(game) {
  const { playable } = engine.getPlayableTiles(game);
  return (playable || []).map(t => ({ type: 'placeTile', tile: t })).slice(0, MAX_ACTIONS);
}

/* ── Apply candidate action ──────────────────────────────────── */
function applyAction(game, playerIdx, action) {
  try {
    if (action.type === 'placeTile')             engine.placeTile(game, playerIdx, action.tile);
    else if (action.type === 'chooseChain')      engine.chooseChain(game, playerIdx, action.chain);
    else if (action.type === 'chooseMergerSurvivor') engine.chooseMergerSurvivor(game, playerIdx, action.chain);
    else if (action.type === 'mergerDecision')   engine.mergerDecision(game, playerIdx, { sell: action.sell, trade: action.trade });
    else if (action.type === 'buyStock')         engine.buyStock(game, playerIdx, action.purchases || {});
  } catch {}
}

/* ── Settle intermediate phases after tile placement ─────────── */
// After placing a tile the game may enter chooseChain (new chain) or
// chooseMergerSurvivor (tied merger). Resolve these with a quick heuristic
// so the value network evaluates a fully-settled board state.
function settlePendingPhases(sim, playerIdx) {
  for (let safety = 0; safety < 10; safety++) {
    if (sim.phase === 'chooseChain') {
      const avail = engine.HOTEL_CHAINS.filter(c => !sim.chains[c].active);
      if (avail.length === 0) break;
      // Pick cheapest to found (gives most merger-survivor flexibility)
      engine.chooseChain(sim, playerIdx, avail[0]);
    } else if (sim.phase === 'chooseMergerSurvivor') {
      const tied = sim.pendingMerger?.tiedChains || [];
      if (tied.length === 0) break;
      engine.chooseMergerSurvivor(sim, playerIdx, tied[0]);
    } else {
      break;
    }
  }
}

/* ── Main entry ──────────────────────────────────────────────── */
/**
 * Decide the best action using value-network guided search.
 * Evaluates each candidate action with a single forward pass.
 * Returns an action object, or null to fall through to heuristic.
 */
function decideMctsAction(game, playerIdx) {
  const phase  = game.phase;
  const myName = game.players[playerIdx].name;

  // ── placeTile: evaluate each tile with the value network ──
  if (phase === 'placeTile' && game.currentPlayerIdx === playerIdx) {
    const actions = legalTileActions(game);
    if (actions.length === 0) return { type: 'buyStock', purchases: {} };
    if (actions.length === 1) return actions[0];

    let best = actions[0], bestScore = -1;
    for (const action of actions) {
      const sim = cloneGame(game);
      applyAction(sim, playerIdx, action);
      settlePendingPhases(sim, playerIdx);
      const score = evaluatePosition(sim, playerIdx, myName);
      if (score > bestScore) { bestScore = score; best = action; }
    }
    return best;
  }

  // ── mergerDecision: evaluate each sell/trade option ──
  if (phase === 'mergerDecision' && game.pendingMerger?.decidingPlayer === playerIdx) {
    const actions = candidateMergerDecisions(game, playerIdx);
    if (actions.length === 1) return actions[0];

    let best = actions[0], bestScore = -1;
    for (const action of actions) {
      const sim = cloneGame(game);
      applyAction(sim, playerIdx, action);
      const score = evaluatePosition(sim, playerIdx, myName);
      if (score > bestScore) { bestScore = score; best = action; }
    }
    return best;
  }

  // All other phases: signal caller to use heuristic
  return null;
}

module.exports = { decideMctsAction, applyAction };
