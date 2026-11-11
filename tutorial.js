/* Interactive tutorial: four prepared positions with a fixed script.
 *
 * Every chapter builds the same board, hands, cash, shares and tile draws for
 * everyone, then plays a fixed list of moves. The player's moves must match
 * the script (the server refuses anything else); the bots' moves are played
 * when the tutorial on the client says "go". Tutorial games are never rated
 * or recorded. The words people see live in public/index.html (TUT_CHAPTERS).
 */
const engine = require('./gameEngine');

const CHAINS = ['Tower', 'Luxor', 'American', 'Worldwide', 'Festival', 'Imperial', 'Continental'];
const ALL_TILES = [];
for (let r = 0; r < 9; r++) for (let c = 0; c < 12; c++) ALL_TILES.push(engine.coordToTile(r, c));

/* Build a game from a plain description. */
function build(name, d) {
  const g = engine.createGame([name, 'Aria', 'Rex'], { quickstart: false });
  const names = [name, 'Aria', 'Rex'];
  g.board = Array.from({ length: 9 }, () => Array(12).fill(null));
  for (const ch of CHAINS) g.chains[ch] = { tiles: [], active: false };
  const used = new Set();
  const put = (what, tiles) => tiles.forEach(t => {
    const { row, col } = engine.tileToCoord(t);
    g.board[row][col] = what;
    used.add(t);
    if (what !== 'lone') { g.chains[what].active = true; g.chains[what].tiles.push(t); }
  });
  for (const [what, tiles] of Object.entries(d.board)) put(what, tiles);
  g.players = names.map((n, i) => {
    const stocks = {}; CHAINS.forEach(ch => { stocks[ch] = (d.stocks[ch] && d.stocks[ch][i]) || 0; });
    d.hands[i].forEach(t => used.add(t));
    return { id: i, name: n, cash: d.cash[i], stocks, tiles: [...d.hands[i]] };
  });
  d.draws.forEach(t => used.add(t));
  // Draws come off the end of the bag, in order.
  g.tileBag = [...ALL_TILES.filter(t => !used.has(t)), ...[...d.draws].reverse()];
  g.currentPlayerIdx = 0;
  g.phase = 'placeTile';
  g.turnNumber = d.turn || 1;
  g.turnTimers = [0, 0, 0];
  g.turnStartTime = Date.now();
  g.pendingMerger = null; g.pendingChainChoice = null;
  g.concedeVotes = {}; g.concedeActive = false; g.concedeInitiator = null;
  g.pstats = names.map(() => ({ founded: {}, mergers: 0, bought: {} }));
  g.quickstartTiles = d.draw ? d.draw.map((t, i) => ({ name: names[i], tile: t })) : [];
  g.log = d.draw ? [...d.draw.map((t, i) => `${names[i]} drew ${t}`), `${name} goes first (closest to 1A)`] : [d.logLine || 'A few turns later…'];
  return g;
}

const ME = 0, ARIA = 1, REX = 2;
const place = (p, tile) => ({ p, a: { type: 'placeTile', tile } });
const name_ = (p, chain) => ({ p, a: { type: 'chooseChain', chain } });
const buy = (p, purchases = {}) => ({ p, a: { type: 'buyStock', purchases } });
const decide = (p, sell, trade) => ({ p, a: { type: 'mergerDecision', sell, trade } });
const survivor = (p, chain) => ({ p, a: { type: 'chooseMergerSurvivor', chain } });
const endGame = p => ({ p, a: { type: 'declareGameEnd' } });

const CHAPTERS = [
  { // 1 — first hotel, buying, a quiet turn, 1st/2nd and ties
    title: 'Your first hotel',
    setup: {
      draw: ['5C', '9D', '2G'],
      board: { lone: ['5C', '9D', '2G'] },
      hands: [['6C', '11H', '3I', '8A', '12F', '1E'], ['9E', '7C', '1B', '12C', '4F', '6I'], ['12A', '3A', '11E', '8G', '1H', '10C']],
      draws: ['4H', '2B', '10A', '6F', '5I'],
      cash: [6000, 6000, 6000], stocks: {},
    },
    script: [
      place(ME, '6C'), name_(ME, 'American'), buy(ME, { American: 3 }),
      place(ARIA, '9E'), name_(ARIA, 'Luxor'), buy(ARIA, { Luxor: 2 }),
      place(REX, '12A'), buy(REX, { American: 2 }),
      place(ME, '11H'), buy(ME, { Luxor: 3 }),
      place(ARIA, '7C'), buy(ARIA, { Luxor: 1 }),
    ],
  },
  { // 2 — a merger, selling/trading/keeping, a hotel closing, a tied merger
    title: 'Mergers',
    setup: {
      turn: 14,
      board: {
        American: ['2E', '3E', '4E', '5E', '2F', '3F'], Tower: ['7E', '8E', '9E'],
        Festival: ['3H', '4H', '5H', '6H'], Worldwide: ['8H', '9H', '10H', '11H'], lone: ['12D', '5A'],
      },
      hands: [['6E', '7H', '1C', '12I', '10A', '4B'], ['11B', '2A', '9B', '6B', '12G', '1I'], ['1A', '8C', '3C', '11F', '5I', '9I']],
      draws: ['7B', '10D', '2I', '8A'],
      cash: [3000, 4000, 4500],
      stocks: { Tower: [5, 1, 3], American: [6, 2, 0], Festival: [3, 0, 1], Worldwide: [0, 4, 2] },
    },
    script: [
      place(ME, '6E'), decide(ME, 1, 4), decide(ARIA, 0, 0), decide(REX, 3, 0), buy(ME, { Festival: 2 }),
      place(ARIA, '11B'), buy(ARIA, {}),
      place(REX, '1A'), buy(REX, { Worldwide: 1 }),
      place(ME, '7H'), survivor(ME, 'Festival'), decide(ARIA, 0, 4), decide(REX, 3, 0), buy(ME, {}),
    ],
  },
  { // 3 — safe hotels, a dead tile, a blocked tile
    title: 'Safe hotels & stuck tiles',
    setup: {
      turn: 41,
      board: {
        Continental: ['1A', '2A', '3A', '4A', '5A', '6A', '1B', '2B', '3B', '4B', '5B', '6B'],
        Imperial: ['1D', '2D', '3D', '4D', '5D', '6D', '1E', '2E', '3E', '4E', '5E'],
        Tower: ['11A', '12A'], Luxor: ['11C', '12C'], American: ['9F', '10F'], Worldwide: ['9H', '10H'], Festival: ['12E', '12F'],
        lone: ['6H'],
      },
      hands: [['3C', '7H', '7A', '10B', '8D', '12I'], ['1G', '4I', '8F', '2H', '11I', '7E'], ['3G', '5I', '9C', '1I', '8B', '10D']],
      draws: ['9B', '2I'],
      cash: [4000, 5000, 4500],
      stocks: { Continental: [6, 3, 2], Imperial: [2, 1, 5], Tower: [0, 2, 1], Luxor: [1, 0, 2], American: [2, 1, 0], Worldwide: [0, 2, 1], Festival: [1, 1, 0] },
    },
    script: [place(ME, '7A'), buy(ME, { Continental: 2 })],
  },
  { // 4 — ending the game
    title: 'Ending the game',
    setup: {
      turn: 88,
      board: {
        Continental: [...['A', 'B', 'C'].flatMap(r => Array.from({ length: 12 }, (_, i) => `${i + 1}${r}`)), '1D', '2D', '3D', '4D'],
        Imperial: Array.from({ length: 10 }, (_, i) => `${i + 1}G`),   // 10: not safe, so End Game only unlocks at 41
      },
      hands: [['5D', '8I', '12E', '10F', '2I', '6F'], ['9E', '3I', '11I', '7F', '1E', '12H'], ['4F', '10I', '6I', '1I', '9F', '12F']],
      draws: ['3F', '8E', '5I'],
      cash: [4000, 9000, 8000],
      stocks: { Continental: [12, 6, 4], Imperial: [5, 0, 8] },
    },
    script: [place(ME, '5D'), endGame(ME), buy(ME, {})],
  },
];

/* Apply one scripted move with the real game rules. */
function apply(game, step) {
  const a = step.a, i = step.p;
  switch (a.type) {
    case 'placeTile':            return engine.placeTile(game, i, a.tile);
    case 'chooseChain':          return engine.chooseChain(game, i, a.chain);
    case 'chooseMergerSurvivor': return engine.chooseMergerSurvivor(game, i, a.chain);
    case 'mergerDecision':       return engine.mergerDecision(game, i, { sell: a.sell, trade: a.trade });
    case 'buyStock':             return engine.buyStock(game, i, a.purchases);
    case 'declareGameEnd':       return engine.declareGameEnd(game, i);
  }
  return { error: 'Unknown move' };
}

const samePurchases = (x = {}, y = {}) => {
  const clean = o => Object.entries(o || {}).filter(([, n]) => n > 0).sort().map(([k, n]) => k + n).join();
  return clean(x) === clean(y);
};
/* Does a move the player sent match what the script expects next? */
function matches(step, type, args) {
  if (!step || step.p !== ME || step.a.type !== type) return false;
  const a = step.a;
  if (type === 'placeTile') return args.tile === a.tile;
  if (type === 'chooseChain' || type === 'chooseMergerSurvivor') return args.chain === a.chain;
  if (type === 'mergerDecision') return (args.sell | 0) === a.sell && (args.trade | 0) === a.trade;
  if (type === 'buyStock') return samePurchases(args.purchases, a.purchases);
  return true;
}
function describe(step) {
  const a = step.a;
  if (a.type === 'placeTile') return `play ${a.tile}`;
  if (a.type === 'chooseChain') return `name the hotel ${a.chain}`;
  if (a.type === 'chooseMergerSurvivor') return `keep ${a.chain}`;
  if (a.type === 'mergerDecision') return `sell ${a.sell}, trade ${a.trade}`;
  if (a.type === 'buyStock') {
    const b = Object.entries(a.purchases).filter(([, n]) => n > 0);
    return b.length ? 'buy ' + b.map(([c, n]) => `${n} ${c}`).join(' + ') : 'press Continue without buying';
  }
  if (a.type === 'declareGameEnd') return 'press End Game';
  return 'follow the bubble';
}

module.exports = { CHAPTERS, build, apply, matches, describe, ME };
