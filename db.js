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
`);

// Migrations — add columns that may be missing in older DBs
const cols = db.prepare(`PRAGMA table_info(users)`).all().map(r => r.name);
if (!cols.includes('is_admin'))        db.exec(`ALTER TABLE users ADD COLUMN is_admin        INTEGER NOT NULL DEFAULT 0`);
if (!cols.includes('public_profile'))  db.exec(`ALTER TABLE users ADD COLUMN public_profile  INTEGER NOT NULL DEFAULT 1`);
if (!cols.includes('phone_encrypted')) db.exec(`ALTER TABLE users ADD COLUMN phone_encrypted TEXT`);

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

module.exports = {
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
  getStats() {
    return {
      users:       db.prepare('SELECT COUNT(*) as n FROM users').get().n,
      games:       db.prepare('SELECT COUNT(*) as n FROM game_history').get().n,
      feedback:    db.prepare('SELECT COUNT(*) as n FROM feedback').get().n,
      newFeedback: db.prepare("SELECT COUNT(*) as n FROM feedback WHERE status='new'").get().n,
    };
  },
};
