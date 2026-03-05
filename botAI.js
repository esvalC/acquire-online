/**
 * Acquire – Bot AI (v2.0)
 * Server-side bot decision logic. Called by server.js for solo games.
 * Bots differ by personality (stock diversification) and difficulty.
 *
 * Difficulty levels:
 *   easy   – random tile, sell-all in mergers, buy cheapest stocks
 *   medium – heuristic tile scoring, smart trading, personality-based buying
 *   hard   – enhanced tile scoring (majority/safe-size aware), smarter stock selection
 */

const engine = require('./gameEngine');

const BOT_ROSTER = [
  { name: 'Aria',  personality: 'balanced'    },
  { name: 'Rex',   personality: 'focused'     },
  { name: 'Nova',  personality: 'diversified' },
  { name: 'Colt',  personality: 'focused'     },
  { name: 'Vera',  personality: 'balanced'    },
];

/* ── Helper: total issued shares for a chain ─────────────────── */
function issuedShares(game, chain) {
  return game.players.reduce((sum, p) => sum + (p.stocks[chain] || 0), 0);
}

/* ── Tile scoring (medium + hard) ────────────────────────────── */
function scoreTile(game, tile, botIdx, difficulty) {
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
      let score = 4 + myShares * 0.1;
      if (difficulty === 'hard') {
        // Bonus for pushing a chain we own toward "safe" size (11+)
        const chainSize = game.chains[analysis.chain].tiles.length;
        if (chainSize >= 8 && chainSize < 11) score += 2.5;
      }
      return score;
    }
    case 'found':
      return 3; // founding gives a free share and opens buying
    case 'merge': {
      const chains = analysis.chains;
      const sorted = chains
        .map(c => ({ chain: c, size: game.chains[c].tiles.length, myShares: player.stocks[c] || 0 }))
        .sort((a, b) => b.size - a.size);
      const survivorMyShares = sorted[0].myShares;
      const defunctMyShares = sorted.slice(1).reduce((s, x) => s + x.myShares, 0);
      return 2 + survivorMyShares * 0.1 + defunctMyShares * 0.3;
    }
    case 'lone':
      return 0.5;
    default:
      return 0;
  }
}

function chooseBotTile(game, botIdx, difficulty) {
  const { playable } = engine.getPlayableTiles(game);
  if (!playable || playable.length === 0) return null;

  if (difficulty === 'easy') {
    // Random legal tile — no strategy
    return playable[Math.floor(Math.random() * playable.length)];
  }

  let bestTile = null;
  let bestScore = -Infinity;
  for (const tile of playable) {
    const score = scoreTile(game, tile, botIdx, difficulty);
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

  // Prefer chains we already have stock in, then by tier value
  const tierOrder = { Continental: 3, Imperial: 3, Festival: 2, Worldwide: 2, American: 2, Tower: 1, Luxor: 1 };
  return available.slice().sort((a, b) => {
    const aOwned = player.stocks[a] || 0;
    const bOwned = player.stocks[b] || 0;
    if (bOwned !== aOwned) return bOwned - aOwned;
    return (tierOrder[b] || 1) - (tierOrder[a] || 1);
  })[0];
}

/* ── Merger survivor choice ──────────────────────────────────── */
function chooseBotSurvivor(game, botIdx) {
  const pm = game.pendingMerger;
  const player = game.players[botIdx];
  // Pick the tied chain where we have the most stock
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
    // Easy: just sell everything — no trading calculation
    return { sell: defunctShares, trade: 0 };
  }

  const survivorSize   = game.chains[pm.survivor].tiles.length;
  const defunctSize    = game.chains[defunctChain].tiles.length;
  const survivorPrice  = engine.stockPrice(pm.survivor, survivorSize);
  const defunctPrice   = engine.stockPrice(defunctChain, defunctSize);
  const survivorRoom   = 25 - issuedShares(game, pm.survivor);

  // Trade if survivor shares are worth at least as much as defunct shares
  let trade = 0;
  if (survivorPrice >= defunctPrice && survivorRoom > 0) {
    const maxTradeByDefunct  = Math.floor(defunctShares / 2) * 2;
    const maxTradeBySurvivor = survivorRoom * 2;
    trade = Math.min(maxTradeByDefunct, maxTradeBySurvivor);
  }

  const remaining = defunctShares - trade;
  return { sell: remaining, trade };
}

/* ── Stock buying ────────────────────────────────────────────── */
function decideBotBuyStock(game, botIdx, personality, difficulty) {
  const player = game.players[botIdx];

  const candidates = engine.HOTEL_CHAINS
    .filter(c => game.chains[c].active)
    .map(c => {
      const size  = game.chains[c].tiles.length;
      const price = engine.stockPrice(c, size);
      const room  = 25 - issuedShares(game, c);
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
    // Buy from cheapest chain(s) — no strategic weighting
    candidates.sort((a, b) => a.price - b.price);
    for (const c of candidates) {
      if (bought >= MAX_BUY) break;
      tryBuy(c.chain);
    }
    return purchases;
  }

  if (difficulty === 'hard') {
    // Re-sort: chains where we lead first, near-safe-size bonus, then own share count
    candidates.sort((a, b) => {
      const aMaxOpp = game.players.filter((_, i) => i !== botIdx)
        .reduce((m, p) => Math.max(m, p.stocks[a.chain] || 0), 0);
      const bMaxOpp = game.players.filter((_, i) => i !== botIdx)
        .reduce((m, p) => Math.max(m, p.stocks[b.chain] || 0), 0);
      const aLeading = a.myShares > aMaxOpp;
      const bLeading = b.myShares > bMaxOpp;
      if (aLeading !== bLeading) return aLeading ? -1 : 1;
      const aNearSafe = (a.size >= 8 && a.size < 11) ? 1 : 0;
      const bNearSafe = (b.size >= 8 && b.size < 11) ? 1 : 0;
      if (aNearSafe !== bNearSafe) return bNearSafe - aNearSafe;
      return b.myShares - a.myShares;
    });
  } else {
    // medium: sort by ascending price
    candidates.sort((a, b) => a.price - b.price);
  }

  // Apply personality on the sorted candidates
  if (personality === 'focused') {
    const target = candidates.reduce((best, c) => {
      if (c.myShares > best.myShares) return c;
      if (c.myShares === best.myShares && c.price < best.price) return c;
      return best;
    }, candidates[0]);
    while (bought < MAX_BUY) { if (!tryBuy(target.chain)) break; }

  } else if (personality === 'balanced') {
    for (let i = 0; i < Math.min(2, candidates.length) && bought < MAX_BUY; i++) {
      const count = i === 0 ? 2 : 1;
      for (let j = 0; j < count && bought < MAX_BUY; j++) {
        tryBuy(candidates[i].chain);
      }
    }

  } else { // diversified
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
function decideBotAction(game, botIdx, personality, difficulty) {
  difficulty = difficulty || 'medium';
  const phase = game.phase;

  if (phase === 'placeTile' && game.currentPlayerIdx === botIdx) {
    const tile = chooseBotTile(game, botIdx, difficulty);
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
    const purchases = decideBotBuyStock(game, botIdx, personality, difficulty);
    engine.buyStock(game, botIdx, purchases);
    return true;
  }

  return false;
}

module.exports = { decideBotAction, BOT_ROSTER };
