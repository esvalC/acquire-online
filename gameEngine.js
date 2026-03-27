/**
 * Acquire – Game Engine (v2.1)
 * Pure game-logic module. No I/O, no sockets.
 *
 * Features:
 *  - Quickstart tile draw (each player draws a tile, closest to 1A goes first, tiles stay on board)
 *  - When all 7 hotel chains are active, tiles that would form a NEW chain are unplayable
 *  - Merger resolution with sequential per-player decisions
 *  - Concede vote system
 *  - Turn timers (cumulative per player)
 *  - Declare game end when conditions met
 */

const ROWS = 9;   // 1-9
const COLS = 12;  // A-L
const HOTEL_CHAINS = [
  'Tower', 'Luxor', 'American', 'Worldwide', 'Festival', 'Imperial', 'Continental'
];
const SAFE_SIZE = 11;
const END_SIZE = 41;
const MAX_STOCK = 25; // per chain

const CHAIN_TIERS = {
  Tower:       'cheap',
  Luxor:       'cheap',
  American:    'mid',
  Worldwide:   'mid',
  Festival:    'mid',
  Imperial:    'expensive',
  Continental: 'expensive',
};

/* ── Price table ───────────────────────────────────────────── */
function stockPrice(chain, size) {
  if (size < 2) return 0;
  const tier = CHAIN_TIERS[chain];
  let base;
  if (size <= 2)       base = 0;
  else if (size <= 3)  base = 1;
  else if (size <= 4)  base = 2;
  else if (size <= 5)  base = 3;
  else if (size <= 10) base = 4;
  else if (size <= 20) base = 5;
  else if (size <= 30) base = 6;
  else if (size <= 40) base = 7;
  else                 base = 8;

  const tierOffset = tier === 'cheap' ? 0 : tier === 'mid' ? 1 : 2;
  return (base + tierOffset) * 100 + 200;
}

function majorityBonus(chain, size) { return stockPrice(chain, size) * 10; }
function minorityBonus(chain, size) { return stockPrice(chain, size) * 5; }

/* ── Coordinate helpers ────────────────────────────────────── */
function tileToCoord(tile) {
  const num = parseInt(tile);
  const letter = tile.replace(/[0-9]/g, '');
  return { row: num - 1, col: letter.charCodeAt(0) - 65 };
}
function coordToTile(row, col) {
  return `${row + 1}${String.fromCharCode(65 + col)}`;
}
function neighbors(row, col) {
  const n = [];
  if (row > 0)        n.push([row - 1, col]);
  if (row < ROWS - 1) n.push([row + 1, col]);
  if (col > 0)        n.push([row, col - 1]);
  if (col < COLS - 1) n.push([row, col + 1]);
  return n;
}

/* ── Create a fresh game state ─────────────────────────────── */
function createGame(playerNames, options = {}) {
  const useQuickstart = options.quickstart !== false;
  const allTiles = [];
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c < COLS; c++)
      allTiles.push(coordToTile(r, c));

  // shuffle
  for (let i = allTiles.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allTiles[i], allTiles[j]] = [allTiles[j], allTiles[i]];
  }

  // ── Always draw tiles to determine turn order ──
  // quickstart=true  → tiles stay on the board
  // quickstart=false → tiles go back in the bag after order is set
  const board = Array.from({ length: ROWS }, () => Array(COLS).fill(null));
  const quickstartLog = [];
  let quickstartTiles = [];

  const startTiles = allTiles.splice(0, playerNames.length);
  const scored = playerNames.map((name, idx) => {
    const { row, col } = tileToCoord(startTiles[idx]);
    return { name, idx, tile: startTiles[idx], row, col, score: row * COLS + col };
  });
  scored.sort((a, b) => a.score - b.score);

  for (const s of scored) {
    quickstartLog.push(`${s.name} drew ${s.tile}`);
  }
  quickstartLog.push(`${scored[0].name} goes first (closest to 1A)`);
  quickstartTiles = scored.map(s => ({ name: s.name, tile: s.tile }));
  const orderedNames = scored.map(s => s.name);

  if (useQuickstart) {
    // Leave tiles on the board as lone tiles
    for (const s of scored) {
      board[s.row][s.col] = 'lone';
    }
  } else {
    // Return tiles to the bag (shuffled back in at random positions)
    for (const tile of startTiles) {
      const pos = Math.floor(Math.random() * (allTiles.length + 1));
      allTiles.splice(pos, 0, tile);
    }
  }

  const players = orderedNames.map((name, idx) => ({
    id: idx,
    name,
    cash: 6000,
    stocks: {},
    tiles: allTiles.splice(0, 6),
  }));

  // Initialize stocks per chain
  HOTEL_CHAINS.forEach(ch => {
    players.forEach(p => { p.stocks[ch] = 0; });
  });

  const chains = {};
  HOTEL_CHAINS.forEach(ch => { chains[ch] = { tiles: [], active: false }; });

  return {
    players,
    board,
    chains,
    tileBag: allTiles,
    currentPlayerIdx: 0,
    phase: 'placeTile',
    pendingMerger: null,
    pendingChainChoice: null,
    log: [...quickstartLog],
    quickstartTiles: quickstartTiles,
    turnNumber: 1,
    turnTimers: orderedNames.map(() => 0),
    turnStartTime: Date.now(),
    concedeVotes: {},
    concedeActive: false,
    concedeInitiator: null,
  };
}

/* ── Tile-playability check ────────────────────────────────── */
function analyzeTilePlacement(state, tile) {
  const { row, col } = tileToCoord(tile);
  const adj = neighbors(row, col);

  const adjacentChains = new Set();
  let adjacentLoneTiles = 0;

  for (const [r, c] of adj) {
    const cell = state.board[r][c];
    if (cell === 'lone') adjacentLoneTiles++;
    else if (cell && cell !== 'lone') adjacentChains.add(cell);
  }

  const activeChains = HOTEL_CHAINS.filter(ch => state.chains[ch].active);
  const allChainsActive = activeChains.length === 7;

  // Would this tile create a new chain?
  const wouldCreateChain = adjacentChains.size === 0 && adjacentLoneTiles > 0;

  // If all 7 chains active and this would create a new chain → unplayable
  if (wouldCreateChain && allChainsActive) {
    return { legal: false, reason: 'allChainsActive' };
  }

  // Check if this merges two+ safe chains
  if (adjacentChains.size >= 2) {
    const safeChains = [...adjacentChains].filter(
      ch => state.chains[ch].tiles.length >= SAFE_SIZE
    );
    if (safeChains.length >= 2) {
      return { legal: false, reason: 'mergeSafeChains' };
    }
  }

  if (adjacentChains.size === 0 && adjacentLoneTiles === 0) {
    return { legal: true, type: 'lone' };
  }
  if (adjacentChains.size === 0 && adjacentLoneTiles > 0) {
    return { legal: true, type: 'found' };
  }
  if (adjacentChains.size === 1) {
    return { legal: true, type: 'expand', chain: [...adjacentChains][0] };
  }
  if (adjacentChains.size >= 2) {
    return { legal: true, type: 'merge', chains: [...adjacentChains] };
  }
  return { legal: true, type: 'lone' };
}

/* ── Get playable tiles for current player ─────────────────── */
function getPlayableTiles(state) {
  const player = state.players[state.currentPlayerIdx];
  const playable = [];
  const unplayable = [];
  for (const tile of player.tiles) {
    const analysis = analyzeTilePlacement(state, tile);
    if (analysis.legal) playable.push(tile);
    else unplayable.push(tile);
  }
  return { playable, unplayable };
}

/* ── Place a tile ──────────────────────────────────────────── */
function placeTile(state, playerIdx, tile) {
  if (state.phase !== 'placeTile') return { error: 'Not in placeTile phase' };
  if (state.currentPlayerIdx !== playerIdx) return { error: 'Not your turn' };

  const player = state.players[playerIdx];
  const tileIdx = player.tiles.indexOf(tile);
  if (tileIdx === -1) return { error: 'You do not have that tile' };

  const analysis = analyzeTilePlacement(state, tile);
  if (!analysis.legal) return { error: `Illegal placement: ${analysis.reason}` };

  // Remove tile from hand
  player.tiles.splice(tileIdx, 1);
  const { row, col } = tileToCoord(tile);

  if (analysis.type === 'lone') {
    state.board[row][col] = 'lone';
    state.log.push(`${player.name} placed ${tile} (lone tile)`);
    state.phase = 'buyStock';
    return { ok: true, action: 'lone' };
  }

  if (analysis.type === 'expand') {
    const chain = analysis.chain;
    state.board[row][col] = chain;
    state.chains[chain].tiles.push(tile);
    absorbLoneTiles(state, row, col, chain);
    state.log.push(`${player.name} placed ${tile}, expanding ${chain}`);
    state.phase = 'buyStock';
    return { ok: true, action: 'expand', chain };
  }

  if (analysis.type === 'found') {
    state.board[row][col] = 'lone';
    state.phase = 'chooseChain';
    state.pendingChainChoice = { tile, row, col };
    state.log.push(`${player.name} placed ${tile}, founding a new chain!`);
    return { ok: true, action: 'found' };
  }

  if (analysis.type === 'merge') {
    state.board[row][col] = 'lone';
    return initiateMerger(state, playerIdx, tile, row, col, analysis.chains);
  }

  return { error: 'Unknown analysis type' };
}

/* ── Absorb lone tiles into a chain ────────────────────────── */
function absorbLoneTiles(state, row, col, chain) {
  const visited = new Set();
  const queue = [[row, col]];
  visited.add(`${row},${col}`);

  while (queue.length > 0) {
    const [r, c] = queue.shift();
    for (const [nr, nc] of neighbors(r, c)) {
      const key = `${nr},${nc}`;
      if (visited.has(key)) continue;
      visited.add(key);
      if (state.board[nr][nc] === 'lone') {
        state.board[nr][nc] = chain;
        state.chains[chain].tiles.push(coordToTile(nr, nc));
        queue.push([nr, nc]);
      }
    }
  }
}

/* ── Choose chain for founding ─────────────────────────────── */
function chooseChain(state, playerIdx, chainName) {
  if (state.phase !== 'chooseChain') return { error: 'Not in chooseChain phase' };
  if (state.currentPlayerIdx !== playerIdx) return { error: 'Not your turn' };
  if (!HOTEL_CHAINS.includes(chainName)) return { error: 'Invalid chain' };
  if (state.chains[chainName].active) return { error: 'Chain already active' };

  const { tile, row, col } = state.pendingChainChoice;
  state.board[row][col] = chainName;
  state.chains[chainName].active = true;
  state.chains[chainName].tiles.push(tile);

  absorbLoneTiles(state, row, col, chainName);

  const player = state.players[playerIdx];
  const totalIssued = state.players.reduce((s, p) => s + (p.stocks[chainName] || 0), 0);
  if (totalIssued < MAX_STOCK) {
    player.stocks[chainName] = (player.stocks[chainName] || 0) + 1;
    state.log.push(`${player.name} founded ${chainName} and received 1 free share`);
  } else {
    state.log.push(`${player.name} founded ${chainName} (no shares available)`);
  }

  state.pendingChainChoice = null;
  state.phase = 'buyStock';
  return { ok: true };
}

/* ── Initiate a merger ─────────────────────────────────────── */
function initiateMerger(state, playerIdx, tile, row, col, involvedChains) {
  const chainSizes = involvedChains.map(ch => ({
    name: ch,
    size: state.chains[ch].tiles.length,
    safe: state.chains[ch].tiles.length >= SAFE_SIZE,
  }));
  chainSizes.sort((a, b) => b.size - a.size);

  const maxSize = chainSizes[0].size;
  const tiedForMax = chainSizes.filter(c => c.size === maxSize);

  if (tiedForMax.length > 1) {
    state.board[row][col] = 'lone';
    state.phase = 'chooseMergerSurvivor';
    state.pendingMerger = {
      tile, row, col,
      involvedChains,
      tiedChains: tiedForMax.map(c => c.name),
      allChainSizes: chainSizes,
    };
    state.log.push(`Merger! ${involvedChains.join(', ')} – ${state.players[playerIdx].name} must choose the surviving chain`);
    return { ok: true, action: 'mergerChooseSurvivor' };
  }

  const survivor = chainSizes[0].name;
  const defunct = chainSizes.slice(1).map(c => c.name);
  return startMergerResolution(state, playerIdx, tile, row, col, survivor, defunct);
}

function chooseMergerSurvivor(state, playerIdx, survivorChain) {
  if (state.phase !== 'chooseMergerSurvivor') return { error: 'Wrong phase' };
  if (state.currentPlayerIdx !== playerIdx) return { error: 'Not your turn' };

  const pm = state.pendingMerger;
  if (!pm.tiedChains.includes(survivorChain)) return { error: 'Not a valid survivor choice' };

  const defunct = pm.involvedChains.filter(c => c !== survivorChain);
  defunct.sort((a, b) => state.chains[b].tiles.length - state.chains[a].tiles.length);

  return startMergerResolution(state, playerIdx, pm.tile, pm.row, pm.col, survivorChain, defunct);
}

function startMergerResolution(state, playerIdx, tile, row, col, survivor, defunctChains) {
  defunctChains.sort((a, b) => state.chains[b].tiles.length - state.chains[a].tiles.length);

  state.pendingMerger = {
    tile, row, col,
    survivor,
    defunctChains: [...defunctChains],
    currentDefunctIdx: 0,
    currentPlayerOffset: 0,
    initiator: playerIdx,
  };

  payMergerBonuses(state, defunctChains[0]);

  state.phase = 'mergerDecision';
  state.pendingMerger.currentPlayerOffset = 0;
  advanceToNextMergerPlayer(state);

  state.log.push(`Merger: ${survivor} absorbs ${defunctChains.join(', ')}`);
  return { ok: true, action: 'merger' };
}

/* ── Pay majority / minority bonuses ───────────────────────── */
function payMergerBonuses(state, defunctChain) {
  const size = state.chains[defunctChain].tiles.length;
  const holdings = state.players
    .map((p, i) => ({ idx: i, shares: p.stocks[defunctChain] || 0 }))
    .filter(h => h.shares > 0)
    .sort((a, b) => b.shares - a.shares);

  if (holdings.length === 0) return;

  const majBonus = majorityBonus(defunctChain, size);
  const minBonus = minorityBonus(defunctChain, size);

  const topShares = holdings[0].shares;
  const majorityHolders = holdings.filter(h => h.shares === topShares);

  if (majorityHolders.length > 1) {
    const totalBonus = majBonus + minBonus;
    const splitBonus = Math.floor(totalBonus / majorityHolders.length / 100) * 100;
    for (const h of majorityHolders) {
      state.players[h.idx].cash += splitBonus;
      state.log.push(`${state.players[h.idx].name} receives $${splitBonus} (split majority/minority bonus for ${defunctChain})`);
    }
  } else {
    state.players[holdings[0].idx].cash += majBonus;
    state.log.push(`${state.players[holdings[0].idx].name} receives $${majBonus} majority bonus for ${defunctChain}`);

    const remainingHolders = holdings.slice(1);
    if (remainingHolders.length > 0) {
      const secondShares = remainingHolders[0].shares;
      const minHolders = remainingHolders.filter(h => h.shares === secondShares);
      const splitMin = Math.floor(minBonus / minHolders.length / 100) * 100;
      for (const h of minHolders) {
        state.players[h.idx].cash += splitMin;
        state.log.push(`${state.players[h.idx].name} receives $${splitMin} minority bonus for ${defunctChain}`);
      }
    } else {
      state.players[holdings[0].idx].cash += minBonus;
      state.log.push(`${state.players[holdings[0].idx].name} also receives $${minBonus} minority bonus (sole holder)`);
    }
  }
}

/* ── Advance to next player needing a merger decision ──────── */
function advanceToNextMergerPlayer(state) {
  const pm = state.pendingMerger;
  const numPlayers = state.players.length;
  const defunctChain = pm.defunctChains[pm.currentDefunctIdx];

  while (pm.currentPlayerOffset < numPlayers) {
    const pIdx = (pm.initiator + pm.currentPlayerOffset) % numPlayers;
    if ((state.players[pIdx].stocks[defunctChain] || 0) > 0) {
      state.pendingMerger.decidingPlayer = pIdx;
      return;
    }
    pm.currentPlayerOffset++;
  }

  finalizeSingleDefunct(state);
}

/* ── Handle a merger decision (sell / trade / hold) ────────── */
function mergerDecision(state, playerIdx, decisions) {
  if (state.phase !== 'mergerDecision') return { error: 'Wrong phase' };
  const pm = state.pendingMerger;
  if (pm.decidingPlayer !== playerIdx) return { error: 'Not your turn to decide' };

  const defunctChain = pm.defunctChains[pm.currentDefunctIdx];
  const player = state.players[playerIdx];
  const held = player.stocks[defunctChain] || 0;

  const { sell = 0, trade = 0 } = decisions;
  const keep = held - sell - trade;

  if (sell < 0 || trade < 0 || keep < 0) return { error: 'Invalid amounts' };
  if (sell + trade > held) return { error: 'Cannot dispose of more shares than you hold' };
  if (trade % 2 !== 0) return { error: 'Must trade in pairs of 2' };

  const defunctSize = state.chains[defunctChain].tiles.length;
  const price = stockPrice(defunctChain, defunctSize);

  if (sell > 0) {
    player.cash += sell * price;
    player.stocks[defunctChain] -= sell;
    state.log.push(`${player.name} sells ${sell} ${defunctChain} shares for $${sell * price}`);
  }

  if (trade > 0) {
    const survivorShares = Math.floor(trade / 2);
    const totalSurvivorIssued = state.players.reduce((s, p) => s + (p.stocks[pm.survivor] || 0), 0);
    const available = MAX_STOCK - totalSurvivorIssued;
    const actualTrade = Math.min(survivorShares, available);
    const actualSpent = actualTrade * 2;

    player.stocks[defunctChain] -= actualSpent;
    player.stocks[pm.survivor] = (player.stocks[pm.survivor] || 0) + actualTrade;

    if (actualTrade < survivorShares) {
      state.log.push(`${player.name} trades ${actualSpent} ${defunctChain} for ${actualTrade} ${pm.survivor} (limited by stock availability)`);
    } else {
      state.log.push(`${player.name} trades ${actualSpent} ${defunctChain} for ${actualTrade} ${pm.survivor}`);
    }
  }

  pm.currentPlayerOffset++;
  advanceToNextMergerPlayer(state);
  return { ok: true };
}

/* ── Finalize one defunct chain in the merger ───────────────── */
function finalizeSingleDefunct(state) {
  const pm = state.pendingMerger;
  const defunctChain = pm.defunctChains[pm.currentDefunctIdx];

  for (const tile of state.chains[defunctChain].tiles) {
    const { row, col } = tileToCoord(tile);
    state.board[row][col] = pm.survivor;
    state.chains[pm.survivor].tiles.push(tile);
  }
  state.chains[defunctChain].tiles = [];
  state.chains[defunctChain].active = false;

  pm.currentDefunctIdx++;
  if (pm.currentDefunctIdx < pm.defunctChains.length) {
    payMergerBonuses(state, pm.defunctChains[pm.currentDefunctIdx]);
    pm.currentPlayerOffset = 0;
    advanceToNextMergerPlayer(state);
    return;
  }

  const { row, col } = tileToCoord(pm.tile);
  state.board[row][col] = pm.survivor;
  state.chains[pm.survivor].tiles.push(pm.tile);
  absorbLoneTiles(state, row, col, pm.survivor);

  state.pendingMerger = null;
  state.phase = 'buyStock';
}

/* ── Pass tile (no playable tiles) ────────────────────────── */
// Called when a player genuinely has no playable tiles. Advances phase to buyStock.
function passTile(state, playerIdx) {
  if (state.phase !== 'placeTile') return { error: 'Wrong phase' };
  if (state.currentPlayerIdx !== playerIdx) return { error: 'Not your turn' };
  const { playable } = getPlayableTiles(state);
  if (playable.length > 0) return { error: 'You have playable tiles' };
  state.phase = 'buyStock';
  return { ok: true };
}

/* ── Buy stock ─────────────────────────────────────────────── */
function buyStock(state, playerIdx, purchases) {
  if (state.phase !== 'buyStock') return { error: 'Wrong phase' };
  if (state.currentPlayerIdx !== playerIdx) return { error: 'Not your turn' };

  let totalBought = 0;
  const player = state.players[playerIdx];

  for (const [chain, count] of Object.entries(purchases)) {
    if (count <= 0) continue;
    if (!HOTEL_CHAINS.includes(chain)) return { error: `Invalid chain: ${chain}` };
    if (!state.chains[chain].active) return { error: `${chain} is not active` };

    totalBought += count;
    if (totalBought > 3) return { error: 'Cannot buy more than 3 shares per turn' };

    const size = state.chains[chain].tiles.length;
    const price = stockPrice(chain, size);
    const cost = price * count;
    if (cost > player.cash) return { error: `Not enough cash for ${chain}` };

    const totalIssued = state.players.reduce((s, p) => s + (p.stocks[chain] || 0), 0);
    if (totalIssued + count > MAX_STOCK) return { error: `Not enough ${chain} shares available` };

    player.cash -= cost;
    player.stocks[chain] = (player.stocks[chain] || 0) + count;
    state.log.push(`${player.name} buys ${count} ${chain} share(s) for $${cost}`);
  }

  if (state.tileBag.length > 0) {
    const newTile = state.tileBag.pop();
    player.tiles.push(newTile);
  }

  replaceDeadTiles(state, playerIdx);

  // End conditions are met — but DON'T auto-end. Let the current player
  // choose when to pull the trigger via declareGameEnd(). canDeclareEnd
  // will show "VOTE END" in the UI so they know the option is available.
  // (Previously auto-ended here, cutting games short without player consent.)

  advanceTurn(state);
  return { ok: true };
}

/* ── Replace tiles that can NEVER be legally played ────────── */
// Only discard PERMANENTLY dead tiles — those that would merge two or more
// safe chains (11+ tiles, which can never be merged). These are discarded
// and replaced immediately at end of turn.
//
// Do NOT discard 'allChainsActive' tiles. Those are TEMPORARILY unplayable:
// after a merger, the defunct chain becomes inactive, freeing a slot, and
// the tile may become a legal "found" or "expand" placement again. The
// player holds those tiles and must play other tiles around them. Only when
// ALL tiles in hand are unplayable may the player pass (via passTile).
function replaceDeadTiles(state, playerIdx) {
  const player = state.players[playerIdx];
  let replaced = true;
  while (replaced) {
    replaced = false;
    for (let i = player.tiles.length - 1; i >= 0; i--) {
      const analysis = analyzeTilePlacement(state, player.tiles[i]);
      if (!analysis.legal && analysis.reason === 'mergeSafeChains') {
        const deadTile = player.tiles.splice(i, 1)[0];
        state.log.push(`${player.name} discards permanently unplayable tile ${deadTile}`);
        if (state.tileBag.length > 0) {
          player.tiles.push(state.tileBag.pop());
          replaced = true;
        }
      }
    }
  }
}

/* ── Advance to next player ────────────────────────────────── */
function advanceTurn(state) {
  const now = Date.now();
  const elapsed = Math.floor((now - state.turnStartTime) / 1000);
  state.turnTimers[state.currentPlayerIdx] += elapsed;
  state.turnStartTime = now;

  state.currentPlayerIdx = (state.currentPlayerIdx + 1) % state.players.length;
  state.turnNumber++;
  state.phase = 'placeTile';

  // Skip players with no playable tiles (iteratively, not recursively)
  let skips = 0;
  while (skips < state.players.length) {
    const { playable } = getPlayableTiles(state);
    if (playable.length > 0) break; // current player has a move

    const cur = state.players[state.currentPlayerIdx];
    if (cur.tiles.length === 0 && state.tileBag.length > 0) {
      // Draw a tile and recheck
      cur.tiles.push(state.tileBag.pop());
      const { playable: p2 } = getPlayableTiles(state);
      if (p2.length > 0) break;
    }

    state.log.push(`${cur.name} has no tiles, skipping`);
    state.currentPlayerIdx = (state.currentPlayerIdx + 1) % state.players.length;
    state.turnNumber++;
    skips++;
  }

  // If every player was skipped (bag empty, all tiles unplayable), force end the game
  if (skips >= state.players.length) {
    state.log.push('No player has a legal move — game ends automatically.');
    endGame(state);
    return;
  }

  // Auto-end when a chain hits 41+ tiles or all active chains are safe
  if (checkGameEnd(state)) {
    state.log.push('End game conditions met — game ends automatically.');
    endGame(state);
  }
}

/* ── End game conditions ───────────────────────────────────── */
function checkGameEnd(state) {
  const activeChains = HOTEL_CHAINS.filter(ch => state.chains[ch].active);
  if (activeChains.length > 0 && activeChains.every(ch => state.chains[ch].tiles.length >= SAFE_SIZE)) return true;
  if (activeChains.some(ch => state.chains[ch].tiles.length >= END_SIZE)) return true;
  return false;
}

function canDeclareGameEnd(state) { return checkGameEnd(state); }

function endGame(state) {
  state.phase = 'gameOver';

  for (const chain of HOTEL_CHAINS) {
    if (state.chains[chain].active) {
      payMergerBonuses(state, chain);
      const size = state.chains[chain].tiles.length;
      const price = stockPrice(chain, size);
      for (const player of state.players) {
        const shares = player.stocks[chain] || 0;
        if (shares > 0) {
          player.cash += shares * price;
          state.log.push(`${player.name} sells ${shares} ${chain} shares for $${shares * price}`);
          player.stocks[chain] = 0;
        }
      }
    }
  }

  const standings = state.players
    .map(p => ({ name: p.name, id: p.id, cash: p.cash }))
    .sort((a, b) => b.cash - a.cash);

  state.log.push(`Game Over! Winner: ${standings[0].name} with $${standings[0].cash}`);
  state.standings = standings;
}

/* ── Concede vote system ───────────────────────────────────── */
function initiateConcedeVote(state, playerIdx) {
  if (state.phase === 'gameOver') return { error: 'Game is already over' };
  if (state.concedeActive) return { error: 'A concede vote is already in progress' };

  state.concedeActive = true;
  state.concedeInitiator = playerIdx;
  state.concedeVotes = {};
  state.concedeVotes[playerIdx] = true;

  state.log.push(`${state.players[playerIdx].name} has called a vote to end the game`);
  return checkConcedeResult(state);
}

function submitConcedeVote(state, playerIdx, vote) {
  if (!state.concedeActive) return { error: 'No active concede vote' };
  if (state.concedeVotes[playerIdx] !== undefined) return { error: 'Already voted' };

  state.concedeVotes[playerIdx] = vote;
  state.log.push(`${state.players[playerIdx].name} votes ${vote ? 'YES' : 'NO'} to end the game`);
  return checkConcedeResult(state);
}

function checkConcedeResult(state) {
  const totalPlayers = state.players.length;
  const votes = Object.values(state.concedeVotes);
  const yesVotes = votes.filter(v => v).length;
  const noVotes = votes.filter(v => !v).length;

  const needed = totalPlayers <= 2 ? totalPlayers : totalPlayers - 1;

  if (yesVotes >= needed) {
    state.concedeActive = false;
    state.log.push('Vote passed! Ending game by concession.');
    endGame(state);
    return { ok: true, result: 'passed', gameOver: true };
  }

  const remaining = totalPlayers - votes.length;
  if (noVotes > totalPlayers - needed) {
    state.concedeActive = false;
    state.log.push('Vote failed. Game continues.');
    return { ok: true, result: 'failed' };
  }

  return { ok: true, result: 'pending', yesVotes, noVotes, needed, totalPlayers };
}

/* ── Declare game end (current player's choice) ────────────── */
function declareGameEnd(state, playerIdx) {
  if (state.currentPlayerIdx !== playerIdx) return { error: 'Not your turn' };
  if (!canDeclareGameEnd(state)) return { error: 'End game conditions not met' };

  state.log.push(`${state.players[playerIdx].name} declares the game over!`);
  endGame(state);
  return { ok: true, gameOver: true };
}

/* ── Build client-safe state (hides other players' tiles) ──── */
function getClientState(state, forPlayerIdx) {
  return {
    board: state.board,
    chains: Object.fromEntries(
      HOTEL_CHAINS.map(ch => [ch, {
        active: state.chains[ch].active,
        size: state.chains[ch].tiles.length,
        price: stockPrice(ch, state.chains[ch].tiles.length),
        safe: state.chains[ch].tiles.length >= SAFE_SIZE,
      }])
    ),
    players: state.players.map((p, i) => ({
      id: p.id,
      name: p.name,
      cash: p.cash,
      stocks: p.stocks,
      tileCount: p.tiles.length,
      tiles: i === forPlayerIdx ? p.tiles : undefined,
    })),
    currentPlayerIdx: state.currentPlayerIdx,
    phase: state.phase,
    pendingMerger: state.pendingMerger ? {
      survivor: state.pendingMerger.survivor,
      defunctChains: state.pendingMerger.defunctChains,
      currentDefunctIdx: state.pendingMerger.currentDefunctIdx,
      decidingPlayer: state.pendingMerger.decidingPlayer,
      tiedChains: state.pendingMerger.tiedChains,
    } : null,
    pendingChainChoice: state.pendingChainChoice ? true : null,
    availableChains: HOTEL_CHAINS.filter(ch => !state.chains[ch].active),
    log: state.log.slice(-30),
    quickstartTiles: state.quickstartTiles,
    turnNumber: state.turnNumber,
    turnTimers: state.turnTimers,
    turnStartTime: state.turnStartTime,
    standings: state.standings,
    concedeActive: state.concedeActive,
    concedeVotes: state.concedeActive ? state.concedeVotes : null,
    concedeInitiator: state.concedeInitiator,
    canDeclareEnd: canDeclareGameEnd(state),
    playableTiles: forPlayerIdx === state.currentPlayerIdx && state.phase === 'placeTile'
      ? getPlayableTiles(state).playable : undefined,
    myIdx: forPlayerIdx,
  };
}

module.exports = {
  createGame,
  placeTile,
  passTile,
  chooseChain,
  chooseMergerSurvivor,
  mergerDecision,
  buyStock,
  declareGameEnd,
  canDeclareGameEnd,
  getClientState,
  getPlayableTiles,
  analyzeTilePlacement,
  initiateConcedeVote,
  submitConcedeVote,
  HOTEL_CHAINS,
  stockPrice,
  majorityBonus,
  minorityBonus,
};
