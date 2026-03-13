/**
 * Acquire – Bot AI (v3.0)
 * Server-side bot decision logic. Called by server.js for solo games.
 *
 * Each bot has hardcoded personality traits (-1 to +1 scale).
 * Difficulty multiplies trait effects: easy=0.3×, medium=1.0×, hard=2.0×
 *
 * Traits:
 *   mergerSeeking  — positive: hunts merger tiles; negative: avoids them
 *   riskAppetite   — positive: prefers expensive chains; negative: prefers cheap/safe
 *   chainLoyalty   — positive: sticks to their chains; negative: switches freely
 */

const engine = require('./gameEngine');

const BOT_ROSTER = [
  { name: 'Aria',  personality: 'balanced'    },
  { name: 'Rex',   personality: 'focused'     },
  { name: 'Nova',  personality: 'diversified' },
  { name: 'Colt',  personality: 'focused'     },
  { name: 'Vera',  personality: 'balanced'    },
];

// Hardcoded per-character traits
// Tuned against 83k selfplay games (2026-03-13):
//   - riskAppetite reduced across the board (high-price bias was a consistent losing strategy)
//   - Aria given a real identity instead of all-zeros
//   - Nova's negative riskAppetite added (buy cheap growing chains, not expensive stagnant ones)
const BOT_TRAITS = {
  Aria: { mergerSeeking:  0.7, riskAppetite: -0.3, chainLoyalty:  0.2 }, // The Broker: patient merger hunter, avoids expensive chains
  Rex:  { mergerSeeking:  0.8, riskAppetite:  0.0, chainLoyalty:  0.2 }, // The Shark: merger-hungry, neutral on price, loosely loyal
  Nova: { mergerSeeking: -0.4, riskAppetite: -0.3, chainLoyalty: -0.7 }, // The Maverick: avoids mergers, buys cheap growing chains, no loyalty
  Colt: { mergerSeeking: -0.4, riskAppetite: -0.8, chainLoyalty:  0.7 }, // The Tycoon: fortress builder, prefers cheap chains, very loyal
  Vera: { mergerSeeking:  0.6, riskAppetite: -0.4, chainLoyalty:  0.4 }, // The Mogul: merger specialist, safe cheap chains, patient
};

// Trait amplification by difficulty
const DIFF_MULT = { easy: 0.3, medium: 1.0, hard: 2.0 };

/* ── Helper: total issued shares for a chain ─────────────────── */
function issuedShares(game, chain) {
  return game.players.reduce((sum, p) => sum + (p.stocks[chain] || 0), 0);
}

/* ── End-game detection ──────────────────────────────────────── */
// The game can end (or will end soon) when 2+ chains are safe (≥11 tiles).
// Hard bots shift to a liquidation strategy when this is true.
function safeChainCount(game) {
  return engine.HOTEL_CHAINS.filter(c =>
    game.chains[c].active && game.chains[c].tiles.length >= 11
  ).length;
}

// Returns the max shares any opponent holds in a chain
function maxOpponentShares(game, chain, myIdx) {
  return game.players
    .filter((_, i) => i !== myIdx)
    .reduce((m, p) => Math.max(m, p.stocks[chain] || 0), 0);
}

/* ── Tile scoring ────────────────────────────────────────────── */
function scoreTile(game, tile, botIdx, difficulty, traits, mult) {
  const analysis = engine.analyzeTilePlacement(game, tile);
  if (!analysis.legal) return -Infinity;

  const player  = game.players[botIdx];
  const endgame = difficulty === 'hard' && safeChainCount(game) >= 2;

  switch (analysis.type) {
    case 'expand': {
      const myShares = player.stocks[analysis.chain] || 0;
      const oppShares = maxOpponentShares(game, analysis.chain, botIdx);
      if (myShares === 0) {
        if (difficulty === 'hard') {
          // Penalize expanding chains where an opponent is well-established
          return oppShares > 3 ? 0.3 : 1.5;
        }
        return 1.5;
      }
      // Expanding own chain is always top-2 priority
      let score = 15 + myShares * 0.2;
      if (difficulty === 'hard') {
        const chainSize = game.chains[analysis.chain].tiles.length;
        // Push chain toward safe size (lock in majority bonus)
        if (chainSize >= 8 && chainSize < 11) score += 2.5;
        // Endgame: extra bonus for growing chains where we lead
        if (endgame && myShares > oppShares) score += 3;
      }
      return score;
    }
    case 'found':
      // Founding a hotel is always the top priority for all bots
      // Endgame: founding is less useful — no time to build it up
      if (endgame) return 8;
      return 20;
    case 'merge': {
      const chains = analysis.chains;
      const sorted = chains
        .map(c => ({ chain: c, size: game.chains[c].tiles.length, myShares: player.stocks[c] || 0 }))
        .sort((a, b) => b.size - a.size);
      const survivorMyShares = sorted[0].myShares;
      const defunctMyShares  = sorted.slice(1).reduce((s, x) => s + x.myShares, 0);
      let score = 2 + survivorMyShares * 0.1 + defunctMyShares * 0.3;
      // mergerSeeking trait: positive = bonus for merge tiles, negative = penalty
      score += traits.mergerSeeking * mult * 2.5;
      // Hard endgame: trigger mergers only if we win the defunct chain payout
      if (endgame) {
        const defunctChain = sorted[1];
        const defunctOpp   = maxOpponentShares(game, defunctChain.chain, botIdx);
        if (defunctChain.myShares > defunctOpp) score += 4; // we get the payout
        else if (defunctChain.myShares === 0)   score -= 2; // waste of a turn
      }
      return score;
    }
    case 'lone':
      return 0.5;
    default:
      return 0;
  }
}

function chooseBotTile(game, botIdx, difficulty, traits, mult) {
  const { playable } = engine.getPlayableTiles(game);
  if (!playable || playable.length === 0) return null;

  if (difficulty === 'easy') {
    // Prefer tiles that found or expand chains, pick randomly among them
    const actionable = playable.filter(t => {
      const a = engine.analyzeTilePlacement(game, t);
      return a.type === 'found' || a.type === 'expand';
    });
    const pool = actionable.length > 0 ? actionable : playable;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  let bestTile = null;
  let bestScore = -Infinity;
  for (const tile of playable) {
    const score = scoreTile(game, tile, botIdx, difficulty, traits, mult);
    if (score > bestScore) { bestScore = score; bestTile = tile; }
  }
  return bestTile;
}

/* ── Chain founding ──────────────────────────────────────────── */
function chooseBotChain(game, botIdx, difficulty) {
  const player = game.players[botIdx];
  const available = engine.HOTEL_CHAINS.filter(c => !game.chains[c].active);
  if (available.length === 0) return null;

  if (difficulty === 'easy') {
    return available[Math.floor(Math.random() * available.length)];
  }

  // 1. Chains we own the most stock in
  // 2. Chains where we already lead (own more than any opponent)
  // 3. Chains with fewer opponent shares (lower competition)
  // 4. Higher-tier chains as final tiebreaker
  const tierOrder = { Continental: 3, Imperial: 3, Festival: 2, Worldwide: 2, American: 2, Tower: 1, Luxor: 1 };
  return available.slice().sort((a, b) => {
    const aOwned = player.stocks[a] || 0;
    const bOwned = player.stocks[b] || 0;
    if (bOwned !== aOwned) return bOwned - aOwned;
    const aMaxOpp = game.players.filter((_, i) => i !== botIdx)
      .reduce((m, p) => Math.max(m, p.stocks[a] || 0), 0);
    const bMaxOpp = game.players.filter((_, i) => i !== botIdx)
      .reduce((m, p) => Math.max(m, p.stocks[b] || 0), 0);
    const aLeads = aOwned > aMaxOpp;
    const bLeads = bOwned > bMaxOpp;
    if (aLeads !== bLeads) return aLeads ? -1 : 1;
    if (aMaxOpp !== bMaxOpp) return aMaxOpp - bMaxOpp;
    return (tierOrder[b] || 1) - (tierOrder[a] || 1);
  })[0];
}

/* ── Merger survivor choice ──────────────────────────────────── */
function chooseBotSurvivor(game, botIdx) {
  const pm = game.pendingMerger;
  const player = game.players[botIdx];
  return pm.tiedChains.slice().sort((a, b) => {
    return (player.stocks[b] || 0) - (player.stocks[a] || 0);
  })[0];
}

/* ── Merger decisions (sell / trade / hold) ──────────────────── */
function decideBotMerger(game, botIdx, difficulty) {
  const pm = game.pendingMerger;
  const defunctChain = pm.defunctChains[pm.currentDefunctIdx];
  const player = game.players[botIdx];
  const defunctShares = player.stocks[defunctChain] || 0;

  if (defunctShares === 0) return { sell: 0, trade: 0 };

  if (difficulty === 'easy') {
    return { sell: defunctShares, trade: 0 };
  }

  const survivorSize  = game.chains[pm.survivor].tiles.length;
  const defunctSize   = game.chains[defunctChain].tiles.length;
  const survivorPrice = engine.stockPrice(pm.survivor, survivorSize);
  const defunctPrice  = engine.stockPrice(defunctChain, defunctSize);
  const survivorRoom  = 25 - issuedShares(game, pm.survivor);

  let trade = 0;
  if (survivorPrice >= defunctPrice && survivorRoom > 0) {
    const maxTradeByDefunct  = Math.floor(defunctShares / 2) * 2;
    const maxTradeBySurvivor = survivorRoom * 2;
    trade = Math.min(maxTradeByDefunct, maxTradeBySurvivor);
  }

  return { sell: defunctShares - trade, trade };
}

/* ── Chain desirability score for stock buying ───────────────── */
function chainDesirability(c, botIdx, game, traits, mult) {
  const maxOpp = game.players.filter((_, i) => i !== botIdx)
    .reduce((m, p) => Math.max(m, p.stocks[c.chain] || 0), 0);
  const isLeading  = c.myShares > maxOpp;
  const couldLead  = c.myShares > 0 && (c.myShares + 3) > maxOpp;

  let score = 0;

  // Base position value
  if (isLeading)      score += 10;
  else if (couldLead) score += 5;
  else if (c.myShares > 0) score += 2;

  // chainLoyalty: amplifies value of chains already invested in
  // high loyalty → really want to stay in own chains
  // low loyalty  → less attached, open to switching
  const posBonus = isLeading ? 4 : couldLead ? 2 : c.myShares > 0 ? 0.5 : 0;
  score += traits.chainLoyalty * mult * posBonus;

  // riskAppetite: positive prefers expensive chains, negative prefers cheap ones
  // prices roughly 200–1200; normalize to ~0–1.5 range
  score += traits.riskAppetite * mult * (c.price / 800);

  // Near-safe-size bonus (always applies)
  if (c.size >= 8 && c.size < 11) score += 1.5;

  return score;
}

/* ── Stock buying ────────────────────────────────────────────── */
function decideBotBuyStock(game, botIdx, personality, difficulty, traits, mult) {
  const player = game.players[botIdx];
  const endgame = difficulty === 'hard' && safeChainCount(game) >= 2;

  let candidates = engine.HOTEL_CHAINS
    .filter(c => game.chains[c].active)
    .map(c => {
      const size     = game.chains[c].tiles.length;
      const price    = engine.stockPrice(c, size);
      const room     = 25 - issuedShares(game, c);
      const myShares = player.stocks[c] || 0;
      return { chain: c, size, price, room, myShares };
    })
    .filter(c => c.price > 0 && c.room > 0 && c.price <= player.cash);

  if (candidates.length === 0) return {};

  // End-game filter (hard bots only): skip chains where we can't realistically lead.
  // In the final stretch, buying into a losing chain is burning cash for no equity.
  if (endgame) {
    const viable = candidates.filter(c => {
      const myShares  = c.myShares;
      const oppShares = maxOpponentShares(game, c.chain, botIdx);
      // Keep if we're already leading, could lead with a few buys, or no opponent holds any
      return myShares >= oppShares || oppShares === 0 || (myShares + 3) >= oppShares;
    });
    if (viable.length > 0) candidates = viable;
    // If no viable chain passes, fall back to all candidates (better to buy something)
  }

  const MAX_SHARES_PER_CHAIN = 13;
  const MAX_BUY = 3;
  const purchases = {};
  let moneyLeft = player.cash;
  let bought = 0;

  const tryBuy = (chain) => {
    const c = candidates.find(x => x.chain === chain);
    if (!c) return false;
    const alreadyOwned = c.myShares + (purchases[chain] || 0);
    if (alreadyOwned >= MAX_SHARES_PER_CHAIN) return false;
    const alreadyBuying = purchases[chain] || 0;
    if (alreadyBuying >= c.room) return false;
    if (c.price > moneyLeft) return false;
    purchases[chain] = alreadyBuying + 1;
    moneyLeft -= c.price;
    bought++;
    return true;
  };

  if (difficulty === 'easy') {
    // Cheapest-first — no strategy
    candidates.sort((a, b) => a.price - b.price);
    for (const c of candidates) {
      if (bought >= MAX_BUY) break;
      tryBuy(c.chain);
    }
    return purchases;
  }

  // Medium + Hard: sort by trait-adjusted desirability score (descending)
  candidates.sort((a, b) =>
    chainDesirability(b, botIdx, game, traits, mult) -
    chainDesirability(a, botIdx, game, traits, mult)
  );

  // Hard endgame: double down on chains we lead — don't spread thin
  if (endgame && personality !== 'diversified') {
    const leading = candidates.filter(c => c.myShares >= maxOpponentShares(game, c.chain, botIdx));
    if (leading.length > 0) {
      const target = leading[0];
      while (bought < MAX_BUY) { if (!tryBuy(target.chain)) break; }
      if (bought < MAX_BUY && leading.length > 1) {
        while (bought < MAX_BUY) { if (!tryBuy(leading[1].chain)) break; }
      }
      return purchases;
    }
  }

  // Personality controls spread; trait-sorted order controls which chains are targeted
  if (personality === 'focused') {
    // All 3 into the top-ranked chain; fall back to second if primary is maxed/unaffordable
    const target = candidates[0];
    while (bought < MAX_BUY) { if (!tryBuy(target.chain)) break; }
    if (bought < MAX_BUY && candidates.length > 1) {
      const fallback = candidates[1];
      while (bought < MAX_BUY) { if (!tryBuy(fallback.chain)) break; }
    }

  } else if (personality === 'balanced') {
    // 2 in top-ranked, 1 in second
    for (let i = 0; i < Math.min(2, candidates.length) && bought < MAX_BUY; i++) {
      const count = i === 0 ? 2 : 1;
      for (let j = 0; j < count && bought < MAX_BUY; j++) {
        tryBuy(candidates[i].chain);
      }
    }

  } else { // diversified
    // 1 in each of top 3
    for (const c of candidates) {
      if (bought >= MAX_BUY) break;
      tryBuy(c.chain);
    }
  }

  return purchases;
}

/* ── Main entry ──────────────────────────────────────────────── */
/**
 * Inspect the current game state and, if it is this bot's turn to act,
 * execute one action directly on the game object.
 * Returns true if an action was taken, false otherwise.
 */
function decideBotAction(game, botIdx, personality, difficulty, botName) {
  difficulty = difficulty || 'medium';
  const traits = BOT_TRAITS[botName] || { mergerSeeking: 0, riskAppetite: 0, chainLoyalty: 0 };
  const mult   = DIFF_MULT[difficulty] || 1.0;

  // Master mode: learned value network (sync, pure-JS, zero deps)
  if (difficulty === 'master') {
    const master = require('./ai/masterBot');
    const action = master.decideMasterAction(game, botIdx);
    if (action) {
      const mcts = require('./ai/mcts');
      mcts.applyAction(game, botIdx, action);
      return true;
    }
    // Model not trained yet or phase not handled — fall through to hard heuristic
    difficulty = 'hard';
  }

  // MCTS mode: use Monte Carlo for tile/merger decisions; heuristic for the rest
  if (difficulty === 'mcts') {
    const mcts   = require('./ai/mcts');
    const action = mcts.decideMctsAction(game, botIdx);
    if (action !== null) {
      // MCTS produced an action — apply and return
      mcts.applyAction(game, botIdx, action);
      return true;
    }
    // null = MCTS defers to heuristic for this phase
    // Fall through to standard heuristic logic below, using 'hard' difficulty
    difficulty = 'hard';
  }

  const phase = game.phase;

  if (phase === 'placeTile' && game.currentPlayerIdx === botIdx) {
    const tile = chooseBotTile(game, botIdx, difficulty, traits, mult);
    if (tile) {
      engine.placeTile(game, botIdx, tile);
    } else {
      const result = engine.passTile(game, botIdx);
      if (result && result.error) return false; // can't advance — don't claim we acted
    }
    return true;
  }

  if (phase === 'chooseChain' && game.currentPlayerIdx === botIdx) {
    const chain = chooseBotChain(game, botIdx, difficulty);
    if (chain) engine.chooseChain(game, botIdx, chain);
    return true;
  }

  if (phase === 'chooseMergerSurvivor' && game.currentPlayerIdx === botIdx) {
    const survivor = chooseBotSurvivor(game, botIdx);
    if (survivor) engine.chooseMergerSurvivor(game, botIdx, survivor);
    return true;
  }

  if (phase === 'mergerDecision' && game.pendingMerger && game.pendingMerger.decidingPlayer === botIdx) {
    const decision = decideBotMerger(game, botIdx, difficulty);
    engine.mergerDecision(game, botIdx, decision);
    return true;
  }

  if (phase === 'buyStock' && game.currentPlayerIdx === botIdx) {
    const purchases = decideBotBuyStock(game, botIdx, personality, difficulty, traits, mult);
    engine.buyStock(game, botIdx, purchases);
    return true;
  }

  return false;
}

module.exports = { decideBotAction, BOT_ROSTER, BOT_TRAITS_EXPORT: BOT_TRAITS, decideBotBuyStock };
