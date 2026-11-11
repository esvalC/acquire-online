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

const ROWS = 9;   // A-I (rows are lettered)
const COLS = 12;  // 1-12 (columns are numbered)
const HOTEL_CHAINS = [
  'Tower', 'Luxor', 'American', 'Worldwide', 'Festival', 'Imperial', 'Continental'
];
const SAFE_SIZE = 11;
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
// Tiles are named like the real board: column number + row letter ("1A" … "12I").
// All 108 names and neighbor lists are computed once — these lookups run
// millions of times per minute inside the search bots' simulated games.
const _COORD = new Map();
const _NEIGHBORS = [];
for (let r = 0; r < ROWS; r++) {
  _NEIGHBORS.push([]);
  for (let c = 0; c < COLS; c++) {
    _COORD.set(`${c + 1}${String.fromCharCode(65 + r)}`, Object.freeze({ row: r, col: c }));
    const n = [];
    if (r > 0)        n.push([r - 1, c]);
    if (r < ROWS - 1) n.push([r + 1, c]);
    if (c > 0)        n.push([r, c - 1]);
    if (c < COLS - 1) n.push([r, c + 1]);
    _NEIGHBORS[r].push(n);
  }
}
function tileToCoord(tile) {
  const hit = _COORD.get(tile);   // frozen, shared — callers only read it
  if (hit) return hit;
  const num = parseInt(tile);
  const letter = String(tile).replace(/[0-9]/g, '');
  return { row: letter.charCodeAt(0) - 65, col: num - 1 };
}
function coordToTile(row, col) {
  return `${col + 1}${String.fromCharCode(65 + row)}`;
}
function neighbors(row, col) {
  return _NEIGHBORS[row][col];
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
    // Per-seat career stats for this game (chains founded, mergers set off,
    // shares bought). Search-bot copies of the game leave this out, so only
    // real moves are counted.
    pstats: orderedNames.map(() => ({ founded: {}, mergers: 0, bought: {} })),
  };
}

function bumpStat(state, idx, key, chain, n = 1) {
  const ps = state.pstats && state.pstats[idx];
  if (!ps) return;
  if (chain) ps[key][chain] = (ps[key][chain] || 0) + n; else ps[key] += n;
}

/* ── Tile-playability check ────────────────────────────────── */
function analyzeTilePlacement(state, tile) {
  // Hot path (called thousands of times per bot decision): plain arrays
  // instead of Sets, no destructuring. Same results as before.
  const co = tileToCoord(tile);
  const adj = _NEIGHBORS[co.row][co.col];
  const chains = [];
  let lone = 0;
  for (let k = 0; k < adj.length; k++) {
    const cell = state.board[adj[k][0]][adj[k][1]];
    if (cell === 'lone') lone++;
    else if (cell && chains.indexOf(cell) === -1) chains.push(cell);
  }
  if (chains.length === 0) {
    if (lone === 0) return { legal: true, type: 'lone' };
    let active = 0;
    for (let i = 0; i < HOTEL_CHAINS.length; i++) if (state.chains[HOTEL_CHAINS[i]].active) active++;
    if (active === 7) return { legal: false, reason: 'allChainsActive' };
    return { legal: true, type: 'found' };
  }
  if (chains.length === 1) return { legal: true, type: 'expand', chain: chains[0] };
  let safe = 0;
  for (let i = 0; i < chains.length; i++) if (state.chains[chains[i]].tiles.length >= SAFE_SIZE) safe++;
  if (safe >= 2) return { legal: false, reason: 'mergeSafeChains' };
  return { legal: true, type: 'merge', chains };
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
    bumpStat(state, playerIdx, 'bought', chainName);
    state.log.push(`${player.name} founded ${chainName} and received 1 free share`);
  } else {
    state.log.push(`${player.name} founded ${chainName} (no shares available)`);
  }

  bumpStat(state, playerIdx, 'founded', chainName);
  state.pendingChainChoice = null;
  state.phase = 'buyStock';
  return { ok: true };
}

/* ── Initiate a merger ─────────────────────────────────────── */
function initiateMerger(state, playerIdx, tile, row, col, involvedChains) {
  bumpStat(state, playerIdx, 'mergers');
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

  const { sell = 0, trade = 0 } = decisions || {};
  if (!Number.isInteger(sell) || !Number.isInteger(trade)) return { error: 'Share counts must be whole numbers' };
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
  if (!purchases || typeof purchases !== 'object') purchases = {};

  // Validate the WHOLE order first (whole shares only), then apply it — a
  // rejected order must not leave a half-completed purchase behind.
  let totalCost = 0;
  for (const [chain, count] of Object.entries(purchases)) {
    if (count === 0) continue;
    if (!Number.isInteger(count) || count < 0) return { error: 'Share counts must be whole numbers' };
    if (!HOTEL_CHAINS.includes(chain)) return { error: `Invalid chain: ${chain}` };
    if (!state.chains[chain].active) return { error: `${chain} is not active` };
    totalBought += count;
    if (totalBought > 3) return { error: 'Cannot buy more than 3 shares per turn' };
    const issued = state.players.reduce((s, p) => s + (p.stocks[chain] || 0), 0);
    if (issued + count > MAX_STOCK) return { error: `Not enough ${chain} shares available` };
    totalCost += stockPrice(chain, state.chains[chain].tiles.length) * count;
  }
  if (totalCost > player.cash) return { error: 'Not enough cash' };
  totalBought = 0;

  for (const [chain, count] of Object.entries(purchases)) {
    if (count <= 0) continue;

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
    bumpStat(state, playerIdx, 'bought', chain, count);
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
  // The player declared the end earlier this turn and has now finished it.
  if (state.endDeclared) {
    endGame(state, state.endDeclared.reason);
    return;
  }

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
    endGame(state, { type: 'noMoves', text: 'No player had a legal tile left to play.' });
    return;
  }

  // Auto-end when all 7 chains are active and all are safe
  if (checkGameEnd(state)) {
    state.log.push('End game conditions met — game ends automatically.');
    endGame(state, { type: 'allSafe', text: 'All seven chains were on the board and safe (11+ tiles).' });
  }
}

/* ── End game conditions ───────────────────────────────────── */
function checkGameEnd(state) {
  // Game ends only when all 7 chains are active AND all are safe (≥11 tiles).
  // At that point no new chain can be founded (all slots taken) and no merger
  // can happen (safe chains cannot be merged) — nothing meaningful remains.
  return HOTEL_CHAINS.every(ch => state.chains[ch].active && state.chains[ch].tiles.length >= SAFE_SIZE);
}

// Official rule: on their turn, a player MAY end the game when any chain has
// 41+ tiles, or when every chain on the board is safe (11+ tiles).
const GAME_END_SIZE = 41;
function canDeclareGameEnd(state) {
  if (state.phase === 'gameOver') return false;
  const active = HOTEL_CHAINS.filter(ch => state.chains[ch].active);
  if (active.length === 0) return false;
  return active.some(ch => state.chains[ch].tiles.length >= GAME_END_SIZE)
      || active.every(ch => state.chains[ch].tiles.length >= SAFE_SIZE);
}

/* Why the game can be (or was) ended — shown on the final screen. */
function describeEndCondition(state, who) {
  const big = HOTEL_CHAINS.filter(ch => state.chains[ch].active && state.chains[ch].tiles.length >= GAME_END_SIZE)
    .sort((a, b) => state.chains[b].tiles.length - state.chains[a].tiles.length)[0];
  const by = who ? `${who} ended the game: ` : '';
  if (big) return { type: 'chain41', chain: big, text: `${by}${big} reached ${state.chains[big].tiles.length} tiles (41+ ends the game).` };
  return { type: 'allSafe', text: `${by}every chain on the board was safe (11+ tiles).` };
}

function endGame(state, reason) {
  state.phase = 'gameOver';
  state.endReason = reason || state.endReason || null;

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
    // If the game could already be ended for real (a 41+ chain, or every chain
    // safe), agreeing to stop is the same as finishing it — count it as finished.
    if (canDeclareGameEnd(state)) {
      state.log.push('Vote passed! The end conditions were met, so the game is finished.');
      endGame(state, describeEndCondition(state, null));
    } else {
      state.log.push('Vote passed! Ending game by concession.');
      endGame(state, { type: 'vote', text: 'The players voted to end the game early.' });
    }
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
  if (state.phase !== 'placeTile' && state.phase !== 'buyStock') return { error: 'Finish the current action first' };
  if (!canDeclareGameEnd(state)) return { error: 'End game conditions not met' };
  if (state.endDeclared) return { error: 'The end has already been declared' };

  // Official rule: after announcing the end, the player still completes their
  // turn (place a tile, buy stock). The game ends when that turn finishes.
  const name = state.players[playerIdx].name;
  state.endDeclared = { by: playerIdx, reason: describeEndCondition(state, name) };
  state.log.push(`${name} declares the game over — it ends after this turn.`);
  return { ok: true, endsAfterTurn: true };
}

/* Final standings if the game ended right now (no state change). */
/* Each player's money if the game ended now (bonuses paid, all shares sold),
 * by seat — the same totals as standingsIfEndedNow, without copying the game. */
function worthIfEndedNow(state) {
  const w = state.players.map(p => p.cash);
  for (const ch of HOTEL_CHAINS) {
    const c = state.chains[ch];
    if (!c.active) continue;
    const size = c.tiles.length, price = stockPrice(ch, size);
    const maj = majorityBonus(ch, size), min = minorityBonus(ch, size);
    const hold = [];
    state.players.forEach((p, i) => { const n = p.stocks[ch] || 0; if (n > 0) { w[i] += n * price; hold.push([i, n]); } });
    if (!hold.length) continue;
    hold.sort((a, b) => b[1] - a[1]);
    const top = hold.filter(h => h[1] === hold[0][1]);
    if (top.length > 1) {
      const each = Math.floor((maj + min) / top.length / 100) * 100;
      for (const [i] of top) w[i] += each;
    } else {
      w[hold[0][0]] += maj;
      const rest = hold.slice(1);
      if (!rest.length) w[hold[0][0]] += min;
      else {
        const sec = rest.filter(h => h[1] === rest[0][1]);
        const each = Math.floor(min / sec.length / 100) * 100;
        for (const [i] of sec) w[i] += each;
      }
    }
  }
  return w;
}

function standingsIfEndedNow(state) {
  const sim = JSON.parse(JSON.stringify({ ...state, log: [] }));
  endGame(sim, null);
  return sim.standings;
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
    tilesLeft: state.tileBag.length,
    turnTimers: state.turnTimers,
    turnStartTime: state.turnStartTime,
    standings: state.standings,
    concedeActive: state.concedeActive,
    concedeVotes: state.concedeActive ? state.concedeVotes : null,
    concedeInitiator: state.concedeInitiator,
    canDeclareEnd: canDeclareGameEnd(state) && !state.endDeclared,
    endDeclared: state.endDeclared ? { by: state.endDeclared.by } : null,
    endReason: state.endReason || null,
    playableTiles: forPlayerIdx === state.currentPlayerIdx && state.phase === 'placeTile'
      ? getPlayableTiles(state).playable : undefined,
    myIdx: forPlayerIdx,
  };
}

/* Nothing to buy (no chain on the board has shares left that you can afford):
 * skip the buy step instead of waiting for a click. Not when the player could
 * end the game right now — End Game is offered during the buy step. */
function canBuyAny(state, idx) {
  const cash = state.players[idx].cash;
  for (const ch of HOTEL_CHAINS) {
    const c = state.chains[ch];
    if (!c.active) continue;
    if (stockPrice(ch, c.tiles.length) > cash) continue;
    let issued = 0;
    for (const p of state.players) issued += p.stocks[ch] || 0;
    if (issued < MAX_STOCK) return true;
  }
  return false;
}
function autoSkipBuy(state) {
  if (state.phase !== 'buyStock') return;
  const idx = state.currentPlayerIdx;
  if (canBuyAny(state, idx)) return;
  if (!state.endDeclared && canDeclareGameEnd(state)) return;
  if (HOTEL_CHAINS.some(ch => state.chains[ch].active)) state.log.push(`${state.players[idx].name} has nothing to buy`);
  buyStock(state, idx, {});
}
const withAutoSkip = fn => function (state, ...args) {
  const r = fn(state, ...args);
  if (r && !r.error) autoSkipBuy(state);
  return r;
};

module.exports = {
  canBuyAny,
  standingsIfEndedNow, worthIfEndedNow,
  tileToCoord, coordToTile,
  createGame,
  placeTile: withAutoSkip(placeTile),
  passTile: withAutoSkip(passTile),
  chooseChain: withAutoSkip(chooseChain),
  chooseMergerSurvivor: withAutoSkip(chooseMergerSurvivor),
  mergerDecision: withAutoSkip(mergerDecision),
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
