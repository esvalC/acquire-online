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
    case 'found': {
      // Founding a hotel is always the top priority for all bots
      // Endgame: founding is less useful — no time to build it up
      if (endgame) return 8;
      // Center preference: a chain founded in the middle of the board has more
      // room to grow in all 4 directions. Tiles on the outer rows/cols are
      // already constrained. Hard bots add up to +2 for a dead-center tile.
      if (difficulty !== 'easy') {
        const num = parseInt(tile);
        const letter = tile.replace(/[0-9]/g, '');
        const row = num - 1; // 0-8
        const col = letter.charCodeAt(0) - 65; // 0-11
        const rowDist = Math.abs(row - 4) / 4;   // 0=center, 1=edge
        const colDist = Math.abs(col - 5.5) / 5.5;
        const edginess = (rowDist + colDist) / 2;
        const centerBonus = difficulty === 'hard' ? (1 - edginess) * 2.0 : (1 - edginess) * 1.0;
        return 20 + centerBonus;
      }
      return 20;
    }
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
      // Tile-timing insight: play isolated lone tiles now to preserve tiles that
      // could expand/found/merge chains for a more strategic moment later.
      // Hard bots actively prefer burning lone tiles over keeping them in hand.
      return difficulty === 'hard' ? 2.0 : 0.5;
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

  const mySurvivorShares  = player.stocks[pm.survivor] || 0;
  const maxOppSurvivor    = maxOpponentShares(game, pm.survivor, botIdx);

  let trade = 0;
  if (survivorRoom > 0) {
    const maxTradeByDefunct  = Math.floor(defunctShares / 2) * 2;
    const maxTradeBySurvivor = survivorRoom * 2;
    const maxPossibleTrade   = Math.min(maxTradeByDefunct, maxTradeBySurvivor);

    if (maxPossibleTrade > 0) {
      // World-champion insight (WBC 2023 final): trade defunct shares into the
      // dominant chain to secure majority. Trading is especially valuable when:
      //   a) trading gives us the lead (or tie) in the survivor chain, OR
      //   b) survivor is more valuable than defunct chain
      const sharesFromFullTrade = Math.floor(maxPossibleTrade / 2);
      const wouldLead  = (mySurvivorShares + sharesFromFullTrade) > maxOppSurvivor;
      const wouldTie   = (mySurvivorShares + sharesFromFullTrade) >= maxOppSurvivor;
      const valueRatio = survivorPrice / Math.max(defunctPrice, 1);

      if (wouldLead || wouldTie) {
        // Trade enough to secure majority/tie — this is the top priority
        trade = maxPossibleTrade;
      } else if (valueRatio >= 1.0) {
        // Survivor is at least as valuable — trade everything we can
        trade = maxPossibleTrade;
      } else if (valueRatio >= 0.7) {
        // Survivor is somewhat cheaper but survivor chain dominance matters
        trade = Math.floor(maxPossibleTrade / 2) * 2; // trade half
      }
      // Otherwise sell everything (defunct stock is more valuable than survivor)
    }
  }

  return { sell: defunctShares - trade, trade };
}

/* ── Early game detection ────────────────────────────────────── */
// tileBag starts at 108 tiles. Early game = most tiles still unplayed.
function isEarlyGame(game) {
  return (game.tileBag?.length ?? 0) > 80;
}

/* ── Edge-position penalty ───────────────────────────────────── */
// Strategy guides warn: "You may not want to invest lots of cash buying stock
// for a corporation that is positioned on the edge of the gameboard and away
// from other corporations." Edge chains have fewer neighbors, grow slower,
// and are harder to merge — so they're usually bad long-term investments.
//
// Returns a 0–1 penalty (1 = maximally edgy). We count what fraction of the
// chain's tiles sit on the outer two rows/cols of the 9×12 board.
function edgePenalty(chain, game) {
  const tiles = game.chains[chain].tiles;
  if (!tiles || tiles.length === 0) return 0;
  const ROWS = 9, COLS = 12;
  let edgeCount = 0;
  for (const tile of tiles) {
    const num = parseInt(tile);
    const letter = tile.replace(/[0-9]/g, '');
    const row = num - 1;
    const col = letter.charCodeAt(0) - 65;
    if (row === 0 || row === ROWS - 1 || col === 0 || col === COLS - 1) edgeCount++;
  }
  return edgeCount / tiles.length;
}

/* ── Chain desirability score for stock buying ───────────────── */
// Incorporates two world-champion insights:
//   1. "I like ties" (Mike Topczewski, 2025 WSBG): deliberately buying to
//      TIE the leader is often better than racing for sole majority.
//      A co-investor wants the merger too — cooperation beats competition.
//   2. "Policing" (WBC 2023 strategy): buying 1 share into an opponent's
//      chain to deny them uncontested cheap majority has real defensive value.
function chainDesirability(c, botIdx, game, traits, mult, difficulty) {
  const maxOpp = game.players.filter((_, i) => i !== botIdx)
    .reduce((m, p) => Math.max(m, p.stocks[c.chain] || 0), 0);
  const isLeading  = c.myShares > maxOpp;
  const isTied     = c.myShares === maxOpp && maxOpp > 0;
  const couldLead  = c.myShares > 0 && (c.myShares + 3) > maxOpp;
  const wouldTie   = (c.myShares + 1) === maxOpp && maxOpp > 0;

  let score = 0;

  // Base position value
  if (isLeading)       score += 10;
  else if (isTied)     score += 8;  // tie is almost as good as leading
  else if (couldLead)  score += 5;
  else if (c.myShares > 0) score += 2;

  // "I like ties" (Mike Topczewski, 2025 WSBG): small bonus for buying into
  // a tie — a co-investor hunts the merge tile with you instead of against you.
  if (wouldTie && difficulty === 'hard') score += 1.0;

  // "Policing" (hard only): small incentive to buy 1 into a chain where an
  // opponent has 2+ uncontested shares — denies free majority.
  // Only applies when we have no shares (pure defensive play).
  if (c.myShares === 0 && maxOpp >= 2 && difficulty === 'hard') score += 0.7;

  // chainLoyalty: amplifies value of chains already invested in
  const posBonus = isLeading ? 4 : (isTied || couldLead) ? 2 : c.myShares > 0 ? 0.5 : 0;
  score += traits.chainLoyalty * mult * posBonus;

  // riskAppetite: positive prefers expensive chains, negative prefers cheap ones
  score += traits.riskAppetite * mult * (c.price / 800);

  // Chain tier preference: premium chains (Continental, Imperial) pay larger bonuses
  // at the same size, so a share is worth more long-term. Slight constant advantage.
  const tierBonus = { Continental: 0.6, Imperial: 0.6, Festival: 0.3, Worldwide: 0.3, American: 0.3, Tower: 0, Luxor: 0 };
  if (difficulty !== 'easy') score += tierBonus[c.chain] || 0;

  // Early-game liquidity preference: small chains are cheap and keep you liquid
  // (Mike: "I generally try to get a cheap hotel on the board early")
  if (isEarlyGame(game) && c.price <= 400) score += 1.0;

  // Safe-size push: urgency ramps up the closer a chain is to locking in bonuses.
  // Size 10 = one tile from safe (critical to hold majority before it locks).
  // Size 8-9 = approaching fast. Hard bots feel this most.
  if (c.size === 10) score += difficulty === 'hard' ? 3.5 : 2.0;
  else if (c.size === 9) score += difficulty === 'hard' ? 2.5 : 1.5;
  else if (c.size === 8) score += difficulty === 'hard' ? 1.5 : 1.0;

  // Edge-position penalty: chains hugging the board edge grow slowly and rarely
  // become merger targets. Scale the penalty by how edgy the chain is.
  // Hard bots fully avoid edge-trapped chains; medium bots are less deterred.
  const edgeFrac = edgePenalty(c.chain, game);
  if (edgeFrac > 0) {
    const penaltyStrength = difficulty === 'hard' ? 2.5 : difficulty === 'medium' ? 1.5 : 0.5;
    score -= edgeFrac * penaltyStrength;
  }

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
    chainDesirability(b, botIdx, game, traits, mult, difficulty) -
    chainDesirability(a, botIdx, game, traits, mult, difficulty)
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

  // End-game cash drain: unspent cash at game-end is dead money.
  // If we still have buys left, spend them on the cheapest chain we can afford
  // rather than passing. Only applies in endgame for medium/hard bots.
  if (endgame && bought < MAX_BUY) {
    const allActive = engine.HOTEL_CHAINS
      .filter(c => game.chains[c].active)
      .map(c => {
        const size  = game.chains[c].tiles.length;
        const price = engine.stockPrice(c, size);
        const room  = 25 - issuedShares(game, c);
        return { chain: c, price, room };
      })
      .filter(c => c.price > 0 && c.room > 0 && c.price <= moneyLeft)
      .sort((a, b) => a.price - b.price); // cheapest first — just spend the cash

    for (const c of allActive) {
      if (bought >= MAX_BUY) break;
      // Re-use tryBuy — it checks room, owned cap, and moneyLeft
      const existing = candidates.find(x => x.chain === c.chain);
      if (!existing) {
        // Add to candidates so tryBuy can find it
        candidates.push({ chain: c.chain, size: 0, price: c.price, room: c.room, myShares: player.stocks[c.chain] || 0 });
      }
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
