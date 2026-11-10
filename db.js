/**
 * Acquire – Database (profiles branch)
 * SQLite via better-sqlite3. Single file, no server required.
 * Phone numbers are NEVER stored. Only a salted SHA-256 hash is kept
 * to enforce one account per phone number.
 */

const Database = require('better-sqlite3');
const crypto   = require('crypto');
const path     = require('path');

const DB_PATH = process.env.DB_PATH || path.join(__dirname, 'data', 'acquire.db');

// Ensure data directory exists
const fs = require('fs');
fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

const db = new Database(DB_PATH);

// Enable WAL mode for better concurrent read performance
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

// Base schema
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    UNIQUE NOT NULL COLLATE NOCASE,
    phone_hash      TEXT    UNIQUE NOT NULL,
    phone_encrypted TEXT,
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    elo             INTEGER NOT NULL DEFAULT 1000,
    games_played    INTEGER NOT NULL DEFAULT 0,
    games_won       INTEGER NOT NULL DEFAULT 0,
    is_admin        INTEGER NOT NULL DEFAULT 0,
    public_profile  INTEGER NOT NULL DEFAULT 1
  );

  CREATE TABLE IF NOT EXISTS game_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    played_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    result     TEXT    NOT NULL CHECK(result IN ('win','loss')),
    elo_change INTEGER NOT NULL DEFAULT 0,
    opponents  TEXT    NOT NULL DEFAULT '[]'
  );

  CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT    NOT NULL DEFAULT 'suggestion',
    message      TEXT    NOT NULL,
    contact      TEXT,
    page         TEXT,
    submitted_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    status       TEXT    NOT NULL DEFAULT 'new',
    gh_issue     INTEGER
  );

  CREATE TABLE IF NOT EXISTS game_sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    played_at        INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    is_solo          INTEGER NOT NULL DEFAULT 0,
    player_count     INTEGER NOT NULL DEFAULT 2,
    human_count      INTEGER NOT NULL DEFAULT 1,
    duration_seconds INTEGER,
    turn_count       INTEGER,
    standings        TEXT    NOT NULL DEFAULT '[]',
    replay_data      TEXT
  );

  CREATE TABLE IF NOT EXISTS game_participants (
    session_id INTEGER NOT NULL REFERENCES game_sessions(id),
    user_id    INTEGER NOT NULL REFERENCES users(id),
    rank       INTEGER NOT NULL,
    cash       INTEGER NOT NULL,
    PRIMARY KEY (session_id, user_id)
  );
  CREATE INDEX IF NOT EXISTS idx_gp_user ON game_participants(user_id);
`);

// Migrations — add columns that may be missing in older DBs
const cols = db.prepare(`PRAGMA table_info(users)`).all().map(r => r.name);
if (!cols.includes('is_admin'))        db.exec(`ALTER TABLE users ADD COLUMN is_admin        INTEGER NOT NULL DEFAULT 0`);
if (!cols.includes('public_profile'))  db.exec(`ALTER TABLE users ADD COLUMN public_profile  INTEGER NOT NULL DEFAULT 1`);
if (!cols.includes('phone_encrypted')) db.exec(`ALTER TABLE users ADD COLUMN phone_encrypted TEXT`);

const sessionCols = db.prepare(`PRAGMA table_info(game_sessions)`).all().map(r => r.name);
if (!sessionCols.includes('standings'))    db.exec(`ALTER TABLE game_sessions ADD COLUMN standings TEXT NOT NULL DEFAULT '[]'`);
if (!sessionCols.includes('replay_data')) db.exec(`ALTER TABLE game_sessions ADD COLUMN replay_data TEXT`);
if (!sessionCols.includes('replay_gz'))   db.exec(`ALTER TABLE game_sessions ADD COLUMN replay_gz BLOB`);

// In-progress games, saved as they're played so a restart/deploy doesn't lose them
db.exec(`
  CREATE TABLE IF NOT EXISTS active_rooms (
    code       TEXT PRIMARY KEY,
    data       BLOB    NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );
`);

// News banner: admin posts + replies (guests may reply, only admins post)
db.exec(`
  CREATE TABLE IF NOT EXISTS news_posts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id  INTEGER NOT NULL REFERENCES users(id),
    body       TEXT    NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );
  CREATE TABLE IF NOT EXISTS news_replies (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id    INTEGER NOT NULL REFERENCES news_posts(id) ON DELETE CASCADE,
    user_id    INTEGER REFERENCES users(id),
    guest_name TEXT,
    body       TEXT    NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );
  CREATE INDEX IF NOT EXISTS idx_news_replies_post ON news_replies(post_id);
`);

/* ── Phone hashing ───────────────────────────────────────────── */
// Salt prevents rainbow-table attacks on the hash.
// PHONE_HASH_SALT must be set in .env — a random 32+ char string.
function hashPhone(e164) {
  const salt = process.env.PHONE_HASH_SALT || 'dev-salt-change-in-production';
  return crypto.createHmac('sha256', salt).update(e164).digest('hex');
}

/* ── User queries ────────────────────────────────────────────── */
const stmts = {
  findByUsername:  db.prepare('SELECT * FROM users WHERE username = ? COLLATE NOCASE'),
  findByPhoneHash: db.prepare('SELECT * FROM users WHERE phone_hash = ?'),
  findById:           db.prepare('SELECT id, username, created_at, elo, games_played, games_won, is_admin, public_profile FROM users WHERE id = ?'),
  setAdmin:           db.prepare('UPDATE users SET is_admin = ? WHERE id = ?'),
  setPublicProfile:   db.prepare('UPDATE users SET public_profile = ? WHERE id = ?'),
  createUser:      db.prepare('INSERT INTO users (username, phone_hash, phone_encrypted) VALUES (?, ?, ?)'),
  updateElo:       db.prepare('UPDATE users SET elo = elo + ?, games_played = games_played + 1, games_won = games_won + ? WHERE id = ?'),
  addGameHistory:  db.prepare('INSERT INTO game_history (user_id, result, elo_change, opponents) VALUES (?, ?, ?, ?)'),
  getHistory:      db.prepare('SELECT * FROM game_history WHERE user_id = ? ORDER BY played_at DESC LIMIT 20'),
};

const zlib = require('zlib');

/* ── Replays: gzip-compressed, kept REPLAY_DAYS days ─────────── */
const REPLAY_DAYS = 5;
function replayCutoff() { return Math.floor(Date.now() / 1000) - REPLAY_DAYS * 86400; }
function pruneReplays() {
  const r = db.prepare(`UPDATE game_sessions SET replay_data = NULL, replay_gz = NULL
    WHERE played_at < ? AND (replay_data IS NOT NULL OR replay_gz IS NOT NULL)`).run(replayCutoff());
  // Give freed pages back to the OS (no-op unless auto_vacuum=INCREMENTAL was enabled)
  if (r.changes > 0) { try { db.pragma('incremental_vacuum'); } catch {} }
  return r.changes;
}

/* ── In-progress games ───────────────────────────────────────── */
function saveActiveRoom(code, obj) {
  const gz = zlib.gzipSync(Buffer.from(JSON.stringify(obj)));
  db.prepare(`INSERT INTO active_rooms (code, data, updated_at) VALUES (?, ?, strftime('%s','now'))
    ON CONFLICT(code) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at`).run(code, gz);
}
function deleteActiveRoom(code) { db.prepare('DELETE FROM active_rooms WHERE code = ?').run(code); }
function loadActiveRooms(maxAgeSeconds = 86400) {
  const rows = db.prepare('SELECT code, data, updated_at FROM active_rooms').all();
  const out = [];
  const now = Math.floor(Date.now() / 1000);
  for (const r of rows) {
    if (now - r.updated_at > maxAgeSeconds) { deleteActiveRoom(r.code); continue; }
    try { out.push({ code: r.code, room: JSON.parse(zlib.gunzipSync(r.data).toString()) }); }
    catch { deleteActiveRoom(r.code); }
  }
  return out;
}

/* ── Total games played (with backfill) ──────────────────────────
 * game_sessions records every finished game, but only since it was added.
 * Older games survive only as per-player rows in game_history; rows from the
 * same game share a timestamp, so distinct timestamps before the first
 * session ≈ one game each. Counted once, cached briefly. */
let _totalCache = { at: 0, value: null };
function getTotalGames() {
  const now = Date.now();
  if (_totalCache.value && now - _totalCache.at < 30000) return _totalCache.value;
  const sessions = db.prepare('SELECT COUNT(*) AS n, MIN(played_at) AS first FROM game_sessions').get();
  const cutoff = sessions.first || Number.MAX_SAFE_INTEGER;
  const legacy = db.prepare('SELECT COUNT(DISTINCT played_at) AS n, MIN(played_at) AS first FROM game_history WHERE played_at < ?').get(cutoff);
  const since = Math.min(legacy.first || Infinity, sessions.first || Infinity);
  const value = {
    total: sessions.n + legacy.n,
    recorded: sessions.n,
    backfilled: legacy.n,
    since: Number.isFinite(since) ? since : null,
  };
  _totalCache = { at: now, value };
  return value;
}

/* ── News ────────────────────────────────────────────────────── */
function listNews(limit = 10) {
  const posts = db.prepare(`
    SELECT p.id, p.body, p.created_at, u.username AS author
    FROM news_posts p JOIN users u ON u.id = p.author_id
    ORDER BY p.created_at DESC, p.id DESC LIMIT ?`).all(limit);
  const replyStmt = db.prepare(`
    SELECT r.id, r.body, r.created_at, r.guest_name,
           u.username AS username, COALESCE(u.is_admin, 0) AS is_admin
    FROM news_replies r LEFT JOIN users u ON u.id = r.user_id
    WHERE r.post_id = ? ORDER BY r.created_at ASC, r.id ASC LIMIT 200`);
  return posts.map(p => ({ ...p, replies: replyStmt.all(p.id) }));
}
function createNewsPost(authorId, body) {
  return db.prepare('INSERT INTO news_posts (author_id, body) VALUES (?, ?)').run(authorId, body).lastInsertRowid;
}
function deleteNewsPost(id) {
  db.prepare('DELETE FROM news_replies WHERE post_id = ?').run(id);
  return db.prepare('DELETE FROM news_posts WHERE id = ?').run(id).changes;
}
function newsPostExists(id) {
  return !!db.prepare('SELECT 1 FROM news_posts WHERE id = ?').get(id);
}
function createNewsReply(postId, userId, guestName, body) {
  return db.prepare('INSERT INTO news_replies (post_id, user_id, guest_name, body) VALUES (?, ?, ?, ?)')
    .run(postId, userId || null, userId ? null : guestName, body).lastInsertRowid;
}
function deleteNewsReply(id) {
  return db.prepare('DELETE FROM news_replies WHERE id = ?').run(id).changes;
}

module.exports = {
  REPLAY_DAYS, pruneReplays, saveActiveRoom, deleteActiveRoom, loadActiveRooms,
  getTotalGames, listNews, createNewsPost, deleteNewsPost, newsPostExists, createNewsReply, deleteNewsReply,
  hashPhone,

  findByUsername(username)     { return stmts.findByUsername.get(username); },
  findByPhoneHash(hash)        { return stmts.findByPhoneHash.get(hash); },
  findById(id)                 { return stmts.findById.get(id); },

  createUser(username, e164Phone, encryptedPhone) {
    const hash = hashPhone(e164Phone);
    stmts.createUser.run(username, hash, encryptedPhone || null);
    return stmts.findByUsername.get(username);
  },

  phoneHashExists(e164Phone) {
    const hash = hashPhone(e164Phone);
    return !!stmts.findByPhoneHash.get(hash);
  },

  recordGameResult(userId, result, eloChange, opponents) {
    const won = result === 'win' ? 1 : 0;
    stmts.updateElo.run(eloChange, won, userId);
    stmts.addGameHistory.run(userId, result, eloChange, JSON.stringify(opponents));
  },

  getHistory(userId) { return stmts.getHistory.all(userId); },

  setAdmin(userId, isAdmin) { stmts.setAdmin.run(isAdmin ? 1 : 0, userId); },
  getAllUsers(limit = 200) {
    return db.prepare('SELECT id, username, created_at, elo, games_played, games_won, is_admin FROM users ORDER BY created_at DESC LIMIT ?').all(limit);
  },
  setPublicProfile(userId, isPublic) { stmts.setPublicProfile.run(isPublic ? 1 : 0, userId); },

  // Feedback
  addFeedback(type, message, contact, page) {
    return db.prepare('INSERT INTO feedback (type,message,contact,page) VALUES (?,?,?,?)').run(type, message, contact || null, page || null);
  },
  getFeedback(limit = 200) {
    return db.prepare('SELECT * FROM feedback ORDER BY submitted_at DESC LIMIT ?').all(limit);
  },
  setFeedbackStatus(id, status) {
    db.prepare('UPDATE feedback SET status=? WHERE id=?').run(status, id);
  },
  setFeedbackIssue(id, ghIssue) {
    db.prepare('UPDATE feedback SET gh_issue=? WHERE id=?').run(ghIssue, id);
  },
  recordSession(isSolo, playerCount, humanCount, durationSeconds, turnCount, standings, replaySnapshots) {
    const insertSession = db.prepare(
      'INSERT INTO game_sessions (is_solo, player_count, human_count, duration_seconds, turn_count, standings, replay_gz) VALUES (?,?,?,?,?,?,?)'
    );
    const insertParticipant = db.prepare(
      'INSERT OR IGNORE INTO game_participants (session_id, user_id, rank, cash) VALUES (?,?,?,?)'
    );
    const run = db.transaction(() => {
      const r = insertSession.run(
        isSolo ? 1 : 0, playerCount, humanCount, durationSeconds ?? null, turnCount ?? null,
        JSON.stringify(standings || []),
        replaySnapshots && replaySnapshots.length ? zlib.gzipSync(Buffer.from(JSON.stringify(replaySnapshots))) : null
      );
      const sessionId = r.lastInsertRowid;
      (standings || []).forEach((p, rank) => {
        if (p.userId) insertParticipant.run(sessionId, p.userId, rank, p.cash);
      });
      return sessionId;
    });
    return run();
  },

  getRecentSessions(limit = 15) {
    return db.prepare('SELECT id, played_at, is_solo, player_count, human_count, duration_seconds, turn_count, standings FROM game_sessions ORDER BY played_at DESC LIMIT ?').all(limit);
  },

  getUserSessions(userId, limit = 20) {
    return db.prepare(`
      SELECT gs.id, gs.played_at, gs.is_solo, gs.player_count, gs.human_count,
             gs.duration_seconds, gs.turn_count, gs.standings, gp.rank, gp.cash AS my_cash
      FROM game_participants gp
      JOIN game_sessions gs ON gs.id = gp.session_id
      WHERE gp.user_id = ?
      ORDER BY gs.played_at DESC
      LIMIT ?
    `).all(userId, limit);
  },

  // Returns { id, played_at, expired, snapshots }
  getSessionReplay(id) {
    const row = db.prepare('SELECT id, played_at, replay_data, replay_gz FROM game_sessions WHERE id = ?').get(id);
    if (!row) return null;
    let snapshots = [];
    try {
      if (row.replay_gz) snapshots = JSON.parse(zlib.gunzipSync(row.replay_gz).toString());
      else if (row.replay_data) snapshots = JSON.parse(row.replay_data);
    } catch {}
    const expired = !snapshots.length && row.played_at < replayCutoff();
    return { id: row.id, played_at: row.played_at, expired, snapshots };
  },

  getStats() {
    const dayAgo  = Math.floor(Date.now() / 1000) - 86400;
    const weekAgo = Math.floor(Date.now() / 1000) - 86400 * 7;
    return {
      users:          db.prepare('SELECT COUNT(*) as n FROM users').get().n,
      games:          db.prepare('SELECT COUNT(*) as n FROM game_history').get().n,
      sessions:       db.prepare('SELECT COUNT(*) as n FROM game_sessions').get().n,
      sessionsToday:  db.prepare('SELECT COUNT(*) as n FROM game_sessions WHERE played_at >= ?').get(dayAgo).n,
      sessionsWeek:   db.prepare('SELECT COUNT(*) as n FROM game_sessions WHERE played_at >= ?').get(weekAgo).n,
      soloSessions:   db.prepare("SELECT COUNT(*) as n FROM game_sessions WHERE is_solo=1").get().n,
      feedback:       db.prepare('SELECT COUNT(*) as n FROM feedback').get().n,
      newFeedback:    db.prepare("SELECT COUNT(*) as n FROM feedback WHERE status='new'").get().n,
    };
  },
};
