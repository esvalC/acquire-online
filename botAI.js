/**
 * Acquire – Bot AI (v1.0)
 * Server-side bot decision logic. Called by server.js for solo games.
 * Bots differ by personality, which controls how they buy stock.
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

/* ── Tile scoring ────────────────────────────────────────────── */
function scoreTile(game, tile, botIdx) {
  const analysis = engine.analyzeTilePlacement(game, tile);
  if (!analysis.legal) return -Infinity;

  const player = game.players[botIdx];

  switch (analysis.type) {
    case 'expand': {
      const myShares = player.stocks[analysis.chain] || 0;
      // Expanding a chain we own is very good; expanding one we don't is neutral
      return myShares > 0 ? 4 + myShares * 0.1 : 1.5;
    }
    case 'found':
      return 3; // founding a chain gives a free share and opens buying
    case 'merge': {
      // Score based on stock ownership in all involved chains
      const chains = analysis.chains;
      const sorted = chains
        .map(c => ({ chain: c, size: game.chains[c].tiles.length, myShares: player.stocks[c] || 0 }))
        .sort((a, b) => b.size - a.size);
      // Surviving chain (largest): shares there stay
      const survivorMyShares = sorted[0].myShares;
      // Defunct chains: get bonuses (good!) and must trade/sell
      const defunctMyShares = sorted.slice(1).reduce((s, x) => s + x.myShares, 0);
      return 2 + survivorMyShares * 0.1 + defunctMyShares * 0.3;
    }
    case 'lone':
      return 0.5;
    default:
      return 0;
  }
}

function chooseBotTile(game, botIdx) {
  const { playable } = engine.getPlayableTiles(game);
  if (!playable || playable.length === 0) return null;

  let bestTile = null;
  let bestScore = -Infinity;
  for (const tile of playable) {
    const score = scoreTile(game, tile, botIdx);
    if (score > bestScore) { bestScore = score; bestTile = tile; }
  }
  return bestTile;
}

/* ── Chain founding ──────────────────────────────────────────── */
function chooseBotChain(game, botIdx) {
  const player = game.players[botIdx];
  const available = engine.HOTEL_CHAINS.filter(c => !game.chains[c].active);
  if (available.length === 0) return null;

  // Prefer chains we already have stock in, then by tier value (continental/imperial = expensive, good)
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
function decideBotMerger(game, botIdx) {
  const pm = game.pendingMerger;
  const defunctChain = pm.defunctChains[pm.currentDefunctIdx];
  const player = game.players[botIdx];
  const defunctShares = player.stocks[defunctChain] || 0;

  if (defunctShares === 0) return { sell: 0, trade: 0 };

  const survivorSize   = game.chains[pm.survivor].tiles.length;
  const defunctSize    = game.chains[defunctChain].tiles.length;
  const survivorPrice  = engine.stockPrice(pm.survivor, survivorSize);
  const defunctPrice   = engine.stockPrice(defunctChain, defunctSize);
  const survivorRoom   = 25 - issuedShares(game, pm.survivor);

  // Trade if survivor shares are worth at least as much as defunct shares
  // Trade in pairs (2 defunct → 1 survivor), limited by available survivor stock
  let trade = 0;
  if (survivorPrice >= defunctPrice && survivorRoom > 0) {
    const maxTradeByDefunct  = Math.floor(defunctShares / 2) * 2;
    const maxTradeBySurvivor = survivorRoom * 2;
    trade = Math.min(maxTradeByDefunct, maxTradeBySurvivor);
  }

  const remaining = defunctShares - trade;
  // Sell what's left (hold nothing – it becomes worthless)
  return { sell: remaining, trade };
}

/* ── Stock buying ────────────────────────────────────────────── */
function decideBotBuyStock(game, botIdx, personality) {
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

  // Sort ascending by price (affordable chains first)
  candidates.sort((a, b) => a.price - b.price);

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

  if (personality === 'focused') {
    // All 3 shares into the chain we own the most of (break ties by lowest price)
    const target = candidates.reduce((best, c) => {
      if (c.myShares > best.myShares) return c;
      if (c.myShares === best.myShares && c.price < best.price) return c;
      return best;
    }, candidates[0]);
    while (bought < MAX_BUY) { if (!tryBuy(target.chain)) break; }

  } else if (personality === 'balanced') {
    // 2 shares in the best chain, 1 in the second
    for (let i = 0; i < Math.min(2, candidates.length) && bought < MAX_BUY; i++) {
      const count = i === 0 ? 2 : 1;
      for (let j = 0; j < count && bought < MAX_BUY; j++) {
        tryBuy(candidates[i].chain);
      }
    }

  } else { // diversified
    // 1 share each in up to 3 different chains
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
function decideBotAction(game, botIdx, personality) {
  const phase = game.phase;

  if (phase === 'placeTile' && game.currentPlayerIdx === botIdx) {
    const tile = chooseBotTile(game, botIdx);
    if (tile) {
      engine.placeTile(game, botIdx, tile);
    } else {
      // No playable tile – engine should handle this, but pass through buyStock
      engine.buyStock(game, botIdx, {});
    }
    return true;
  }

  if (phase === 'chooseChain' && game.currentPlayerIdx === botIdx) {
    const chain = chooseBotChain(game, botIdx);
    if (chain) engine.chooseChain(game, botIdx, chain);
    return true;
  }

  if (phase === 'chooseMergerSurvivor' && game.currentPlayerIdx === botIdx) {
    const survivor = chooseBotSurvivor(game, botIdx);
    if (survivor) engine.chooseMergerSurvivor(game, botIdx, survivor);
    return true;
  }

  if (phase === 'mergerDecision' && game.pendingMerger && game.pendingMerger.decidingPlayer === botIdx) {
    const decision = decideBotMerger(game, botIdx);
    engine.mergerDecision(game, botIdx, decision);
    return true;
  }

  if (phase === 'buyStock' && game.currentPlayerIdx === botIdx) {
    const purchases = decideBotBuyStock(game, botIdx, personality);
    engine.buyStock(game, botIdx, purchases);
    return true;
  }

  return false;
}

module.exports = { decideBotAction, BOT_ROSTER };
