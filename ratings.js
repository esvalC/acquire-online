/* Ratings + career stats for members.
 *
 * Rating: each member is compared with every other rated seat in the game
 * (members at their current rating, bots at a fixed rating for their level;
 * guests are skipped). Against each opponent the score is mostly who finished
 * ahead, plus a share for the money margin — a big win earns more than a
 * squeaker, and equal money is a draw.
 *
 *   finished game (end declared / all safe / no moves) → rated normally
 *   ended early by vote → can lower a rating, never raise it (a win scores 0)
 *   quit (or never came back) → a loss against everyone still in the game
 */
const engine = require('./gameEngine');

const K = 32;
const MARGIN_SCALE = 30000;           // a $30k margin is the full margin share
const BOT_RATING = { hard: 900, expert: 1050, mcts: 1200, master: 1400 };
const BOT_LABEL  = { hard: 'Easy', expert: 'Medium', mcts: 'Hard', master: 'Master' };

const expected = (me, opp) => 1 / (1 + Math.pow(10, (opp - me) / 400));
function pairScore(myCash, oppCash) {
  const place = myCash > oppCash ? 1 : myCash < oppCash ? 0 : 0.5;
  const margin = 0.5 + 0.5 * Math.max(-1, Math.min(1, (myCash - oppCash) / MARGIN_SCALE));
  return 0.7 * place + 0.3 * margin;
}

// A game "counts" once every player has had a turn; quitting before that is free.
const underway = room => !!(room && room.game && room.game.turnNumber > room.players.length);
const finishedType = t => t === 'chain41' || t === 'allSafe' || t === 'noMoves';

function seatRating(db, p) {
  if (p.isBot) return BOT_RATING[p.difficulty] || BOT_RATING.expert;
  return db.stats.rating(p.userId);
}
const rated = p => p.isBot || !!p.userId;            // guests aren't rated opponents

/* Record one member's result.
 * kind: 'finished' | 'early' | 'quit'; cashBySeat: final (or current) money by seat. */
function recordMember(db, room, p, kind, cashBySeat, ratingsBefore) {
  const game = room.game;
  const myCash = cashBySeat[p.idx] || 0;
  const opps = room.players.filter(o => o !== p && rated(o) && !o.forfeited);
  const myRating = ratingsBefore.get(p) ?? db.stats.rating(p.userId);

  let delta = 0;
  for (const o of opps) {
    const actual = kind === 'quit' ? 0 : pairScore(myCash, cashBySeat[o.idx] || 0);
    delta += actual - expected(myRating, ratingsBefore.get(o) ?? seatRating(db, o));
  }
  delta = opps.length ? (K * delta) / opps.length : 0;
  if (kind === 'early' && delta > 0) delta = 0;

  const order = room.players.filter(o => !o.forfeited || o === p).map(o => cashBySeat[o.idx] || 0).sort((a, b) => b - a);
  const place = kind === 'quit' ? room.players.length : 1 + order.filter(c => c > myCash).length;
  const others = room.players.filter(o => o !== p).map(o => cashBySeat[o.idx] || 0);
  const won = kind !== 'quit' && others.every(c => myCash > c);
  const fin = kind === 'finished';
  const solo = !!room.isSolo;
  const qm = room.mode === 'Quick Master';
  const ps = (game.pstats && game.pstats[p.idx]) || { founded: {}, bought: {}, mergers: 0 };
  const inc = {
    games: 1, wins: won ? 1 : 0,
    fin_games: fin ? 1 : 0, fin_wins: fin && won ? 1 : 0,
    early_games: kind === 'early' ? 1 : 0, early_wins: kind === 'early' && won ? 1 : 0,
    quits: kind === 'quit' ? 1 : 0,
    multi_games: fin && !solo ? 1 : 0, multi_wins: fin && !solo && won ? 1 : 0,
    solo_games: fin && solo ? 1 : 0, solo_wins: fin && solo && won ? 1 : 0,
    place_sum: fin ? place : 0, cash_sum: fin ? myCash : 0,
    margin_sum: fin && won ? myCash - Math.max(...others) : 0,
    mergers: ps.mergers || 0,
    qm_games: qm && kind !== 'early' ? 1 : 0, qm_wins: qm && fin && won ? 1 : 0,
  };
  const bots = {};
  if (fin) for (const o of room.players) if (o.isBot && !o.forfeited && BOT_LABEL[o.difficulty]) bots[BOT_LABEL[o.difficulty]] = won;
  return db.stats.apply(p.userId, {
    rating: myRating + delta, inc,
    bestCash: fin ? myCash : 0,
    streak: fin && won ? 'win' : (kind === 'quit' || !won) ? 'loss' : null,
    founded: ps.founded, bought: ps.bought, bots,
    frLevel: fin && won && room.frLevel ? room.frLevel : 0,
    won, opponents: room.players.filter(o => o !== p).map(o => ({ name: o.name, cash: cashBySeat[o.idx] || 0 })),
  });
}

/* Game over (finished or voted early): rate every member still seated. */
function recordGameOver(db, room) {
  const game = room.game;
  const type = game.endReason && game.endReason.type;
  const kind = finishedType(type) ? 'finished' : 'early';
  const cash = game.players.map(pl => pl.cash);
  const members = room.players.filter(p => !p.isBot && p.userId && !p.forfeited);
  const before = new Map(room.players.filter(rated).map(p => [p, seatRating(db, p)]));
  const changes = {};
  for (const p of members) {
    try { changes[p.name] = recordMember(db, room, p, kind, cash, before); }
    catch (e) { console.error('[ratings] game over', p.name, e.message); }
  }
  if (kind === 'finished') {
    for (const a of members) for (const b of members) {
      if (a === b || cash[a.idx] === cash[b.idx]) continue;
      try { db.stats.h2h(a.userId, b.userId, cash[a.idx] > cash[b.idx]); } catch {}
    }
  }
  return { kind, changes };
}

/* A member quits (or never comes back to) a game in progress. Returns the
 * rating change, or null if the game hadn't got going / they're a guest. */
function recordQuit(db, room, p) {
  if (!p || p.isBot || !p.userId || p.forfeited || !underway(room) || room.game.phase === 'gameOver') return null;
  const cash = engine.worthIfEndedNow(room.game);
  const before = new Map(room.players.filter(rated).map(o => [o, seatRating(db, o)]));
  let change = null;
  try { change = recordMember(db, room, p, 'quit', cash, before); }
  catch (e) { console.error('[ratings] quit', p.name, e.message); }
  for (const o of room.players) {
    if (o !== p && !o.isBot && o.userId && !o.forfeited) { try { db.stats.h2h(p.userId, o.userId, false); db.stats.h2h(o.userId, p.userId, true); } catch {} }
  }
  p.forfeited = true;
  return change;
}

module.exports = { recordGameOver, recordQuit, underway, finishedType, BOT_RATING, BOT_LABEL, pairScore };
