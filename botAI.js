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
const BOT_TRAITS = {
  Aria: { mergerSeeking:  0.0, riskAppetite:  0.0, chainLoyalty:  0.0 }, // The Broker: pure balanced
  Rex:  { mergerSeeking:  1.0, riskAppetite:  1.0, chainLoyalty: -1.0 }, // The Shark: merger-hungry, bold, opportunistic
  Nova: { mergerSeeking: -0.5, riskAppetite:  1.0, chainLoyalty: -0.5 }, // The Maverick: avoids mergers, big bets, low loyalty
  Colt: { mergerSeeking: -0.5, riskAppetite: -1.0, chainLoyalty:  1.0 }, // The Tycoon: anti-merger, conservative, fortress builder
  Vera: { mergerSeeking:  1.0, riskAppetite: -0.5, chainLoyalty:  0.5 }, // The Mogul: patient merger specialist, somewhat loyal
};

// Trait amplification by difficulty
const DIFF_MULT = { easy: 0.3, medium: 1.0, hard: 2.0 };

/* ── Helper: total issued shares for a chain ─────────────────── */
function issuedShares(game, chain) {
  return game.players.reduce((sum, p) => sum + (p.stocks[chain] || 0), 0);
}

/* ── Tile scoring ────────────────────────────────────────────── */
function scoreTile(game, tile, botIdx, difficulty, traits, mult) {
  const analysis = engine.analyzeTilePlacement(game, tile);
  if (!analysis.legal) return -Infinity;

  const player = game.players[botIdx];

  switch (analysis.type) {
    case 'expand': {
      const myShares = player.stocks[analysis.chain] || 0;
      if (myShares === 0) {
        if (difficulty === 'hard') {
          // Penalize expanding chains where an opponent is well-established
          const maxOpp = game.players
            .filter((_, i) => i !== botIdx)
            .reduce((m, p) => Math.max(m, p.stocks[analysis.chain] || 0), 0);
          return maxOpp > 3 ? 0.3 : 1.5;
        }
        return 1.5;
      }
      // Expanding own chain is always top-2 priority — score always beats mergers
      let score = 15 + myShares * 0.2;
      if (difficulty === 'hard') {
        const chainSize = game.chains[analysis.chain].tiles.length;
        if (chainSize >= 8 && chainSize < 11) score += 2.5;
      }
      return score;
    }
    case 'found':
      // Founding a hotel is always the top priority for all bots
      return 20;
    case 'merge': {
      const chains = analysis.chains;
      const sorted = chains
        .map(c => ({ chain: c, size: game.chains[c].tiles.length, myShares: player.stocks[c] || 0 }))
        .sort((a, b) => b.size - a.size);
      const survivorMyShares = sorted[0].myShares;
      const defunctMyShares = sorted.slice(1).reduce((s, x) => s + x.myShares, 0);
      let score = 2 + survivorMyShares * 0.1 + defunctMyShares * 0.3;
      // mergerSeeking trait: positive = bonus for merge tiles, negative = penalty
      score += traits.mergerSeeking * mult * 2.5;
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

  const candidates = engine.HOTEL_CHAINS
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

  // Personality controls spread; trait-sorted order controls which chains are targeted
  if (personality === 'focused') {
    // All 3 into the top-ranked chain
    const target = candidates[0];
    while (bought < MAX_BUY) { if (!tryBuy(target.chain)) break; }

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

  const phase = game.phase;

  if (phase === 'placeTile' && game.currentPlayerIdx === botIdx) {
    const tile = chooseBotTile(game, botIdx, difficulty, traits, mult);
    if (tile) {
      engine.placeTile(game, botIdx, tile);
    } else {
      engine.buyStock(game, botIdx, {});
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

module.exports = { decideBotAction, BOT_ROSTER };
