/**
 * Acquire – Server (v3.1 – profiles branch)
 * Express + Socket.io
 * Supports multiplayer (lobby), solo (vs bots), and user profiles.
 */

require('dotenv').config();

const express    = require('express');
const http       = require('http');
const path       = require('path');
const crypto = require('crypto');
const { randomUUID } = crypto;
const { Server } = require('socket.io');
const rateLimit  = require('express-rate-limit');
const engine     = require('./gameEngine');
const { decideBotAction, BOT_ROSTER, maybeDeclareEnd } = require('./botAI');
const { applySearchAction, SEARCH_LEVELS, HINT_LEVEL } = require('./ai/searchBot');
const db         = require('./db');
const { SMS_ENABLED, hashPassword, verifyPassword, passwordProblem, signToken, verifyToken, requireAuth, requireAdmin, sendOtp, verifyOtp, normalisePhone, encryptPhone, decryptPhone, maskPhone } = require('./auth');

const app    = express();
app.set('trust proxy', 1); // nginx sits in front — req.ip = the address nginx saw
app.disable('x-powered-by');

// Refuse to run in production with the built-in development secrets.
if (process.env.TWILIO_ACCOUNT_SID) {
  for (const k of ['JWT_SECRET', 'PHONE_HASH_SALT', 'PHONE_ENCRYPT_KEY']) {
    if (!process.env[k] || process.env[k].length < 16 || /^0+$/.test(process.env[k])) {
      console.error(`FATAL: ${k} is missing or weak in production — refusing to start`);
      process.exit(1);
    }
  }
}

/* ── Real visitor IP ───────────────────────────────────────────
 * Traffic arrives  visitor → Cloudflare → nginx → node, so req.ip is a
 * Cloudflare edge address shared by many visitors. Use Cloudflare's
 * CF-Connecting-IP header — but ONLY when the request really came from a
 * Cloudflare address (otherwise anyone hitting the server directly could
 * fake the header and dodge rate limits). Ranges: cloudflare.com/ips */
const net = require('net');
const CLOUDFLARE = new net.BlockList();
for (const c of ['173.245.48.0/20','103.21.244.0/22','103.22.200.0/22','103.31.4.0/22','141.101.64.0/18',
                 '108.162.192.0/18','190.93.240.0/20','188.114.96.0/20','197.234.240.0/22','198.41.128.0/17',
                 '162.158.0.0/15','104.16.0.0/13','104.24.0.0/14','172.64.0.0/13','131.0.72.0/22']) {
  const [ip, bits] = c.split('/'); CLOUDFLARE.addSubnet(ip, +bits, 'ipv4');
}
for (const c of ['2400:cb00::/32','2606:4700::/32','2803:f800::/32','2405:b500::/32','2405:8100::/32','2a06:98c0::/29','2c0f:f248::/32']) {
  const [ip, bits] = c.split('/'); CLOUDFLARE.addSubnet(ip, +bits, 'ipv6');
}
function clientIp(req) {
  const peer = (req.ip || '').replace(/^::ffff:/, '');
  const cf = req.headers['cf-connecting-ip'];
  const fam = net.isIP(peer) === 6 ? 'ipv6' : 'ipv4';
  if (cf && net.isIP(cf) && net.isIP(peer) && CLOUDFLARE.check(peer, fam)) return cf;
  return peer;
}
const { ipKeyGenerator } = require('express-rate-limit');
const byVisitor = req => ipKeyGenerator(clientIp(req));

// Basic security headers (no inline-script CSP: the single-page app uses inline handlers)
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'same-origin');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  next();
});
const server = http.createServer(app);
const io     = new Server(server, { cors: { origin: ['https://playonlineacquire.com', 'http://localhost:3000'] } });

app.use(express.json({ limit: '50kb' }));
app.use(express.static('public'));

/* ── Rate limiters ───────────────────────────────────────────── */
const smslimit = rateLimit({
  windowMs: 10 * 60 * 1000, max: 5, keyGenerator: byVisitor,
  message: { error: 'Too many SMS requests. Wait 10 minutes.' },
  standardHeaders: true, legacyHeaders: false,
});
// Code checks: Twilio already allows only 5 tries per code; this stops spraying many codes
const verifyLimit = rateLimit({
  windowMs: 10 * 60 * 1000, max: 20, keyGenerator: byVisitor,
  message: { error: 'Too many attempts. Wait 10 minutes.' },
  standardHeaders: true, legacyHeaders: false,
});

/* Texting protection ("SMS pumping"): only US/Canada numbers, and a site-wide
 * cap on texts per hour so a script can't run up the Twilio bill. */
const SMS_HOURLY_CAP = Number(process.env.SMS_HOURLY_CAP || 30);
let smsWindow = { start: Date.now(), count: 0 };
function smsAllowed(e164) {
  if (!process.env.ALLOW_INTL_SMS && !/^\+1[2-9]\d{2}[2-9]\d{6}$/.test(e164)) {
    return 'Only US and Canadian phone numbers are supported.';
  }
  if (Date.now() - smsWindow.start > 3600 * 1000) smsWindow = { start: Date.now(), count: 0 };
  if (smsWindow.count >= SMS_HOURLY_CAP) return 'Sign-ups are busy right now — please try again in an hour.';
  smsWindow.count++;
  return null;
}

/* ── Auth / Profile routes ───────────────────────────────────── */

// Text-message sign-in is retired once Twilio is removed from .env.
function smsOnly(_req, res, next) {
  if (!SMS_ENABLED) return res.status(410).json({ error: 'Text-message sign-in has been retired. Sign in with your username and password.' });
  next();
}
app.get('/api/auth-config', (_req, res) => res.json({ sms: SMS_ENABLED }));

const USERNAME_RE = /^[a-zA-Z0-9_]{2,20}$/;
const publicUser = u => ({ id: u.id, username: u.username, elo: u.elo, has_password: !!u.password_hash });

const signupLimit = rateLimit({
  windowMs: 60 * 60 * 1000, max: 5, keyGenerator: byVisitor,
  message: { error: 'Too many new accounts from this connection. Try again in an hour.' },
  standardHeaders: true, legacyHeaders: false,
});
const loginLimit = rateLimit({
  windowMs: 15 * 60 * 1000, max: 30, keyGenerator: byVisitor,
  message: { error: 'Too many sign-in attempts. Wait 15 minutes.' },
  standardHeaders: true, legacyHeaders: false,
});
// Per-account lockout: 10 wrong passwords in 15 minutes locks that username
// for 15 minutes, from any connection (stops slow guessing spread across IPs).
const failedLogins = new Map(); // username(lower) → { count, until, first }
function accountLocked(name) {
  const f = failedLogins.get(name);
  return f && f.until && f.until > Date.now();
}
function noteFailedLogin(name) {
  const now = Date.now();
  let f = failedLogins.get(name);
  if (!f || now - f.first > 15 * 60 * 1000) f = { count: 0, first: now, until: 0 };
  f.count++;
  if (f.count >= 10) f.until = now + 15 * 60 * 1000;
  failedLogins.set(name, f);
  if (failedLogins.size > 50000) failedLogins.clear();   // memory guard
}

// Create an account: username + password
app.post('/api/signup', signupLimit, async (req, res) => {
  const { username, password, remember } = req.body || {};
  if (typeof username !== 'string' || !USERNAME_RE.test(username)) {
    return res.status(400).json({ error: 'Username must be 2–20 letters, numbers or underscores' });
  }
  const bad = passwordProblem(password);
  if (bad) return res.status(400).json({ error: bad });
  if (password.toLowerCase() === username.toLowerCase()) return res.status(400).json({ error: 'Password can\'t be your username' });
  if (!isClean(username)) return res.status(400).json({ error: 'Please pick a different username' });
  if (db.findByUsername(username)) return res.status(409).json({ error: 'Username already taken' });
  try {
    const user = db.createPasswordUser(username, await hashPassword(password));
    res.json({ ok: true, token: signToken(user.id, { remember: remember !== false }), user: publicUser(user) });
  } catch (err) {
    if (/UNIQUE/.test(err.message)) return res.status(409).json({ error: 'Username already taken' });
    console.error('signup error:', err.message);
    res.status(500).json({ error: 'Could not create account' });
  }
});

// Sign in: username + password
app.post('/api/login-password', loginLimit, async (req, res) => {
  const { username, password, remember } = req.body || {};
  if (typeof username !== 'string' || typeof password !== 'string' || !username || !password) {
    return res.status(400).json({ error: 'Enter your username and password' });
  }
  const key = username.toLowerCase().slice(0, 40);
  if (accountLocked(key)) return res.status(429).json({ error: 'Too many wrong passwords for this account. Try again in 15 minutes.' });
  const user = USERNAME_RE.test(username) ? db.findByUsername(username) : null;
  const ok = await verifyPassword(password, user && user.password_hash);   // same timing either way
  if (!ok) {
    noteFailedLogin(key);
    if (user && !user.password_hash) {
      return res.status(400).json({ error: SMS_ENABLED
        ? 'This account doesn\'t have a password yet — use "Sign in with a text code" below, then set one in your profile.'
        : 'This account doesn\'t have a password yet. Ask the site admin for a temporary password.' });
    }
    return res.status(401).json({ error: 'Wrong username or password' });
  }
  failedLogins.delete(key);
  res.json({ ok: true, token: signToken(user.id, { remember: remember !== false }), user: publicUser(user) });
});

// Swap a remembered token that still has an expiry date (older sign-ins) for a
// permanent one, so "stay signed in" really means staying signed in.
app.post('/api/me/token', requireAuth, (req, res) => {
  res.json({ ok: true, token: signToken(req.userId, { remember: true }) });
});

// Set or change your password (signed in). Signs out every other device.
app.post('/api/me/password', loginLimit, requireAuth, async (req, res) => {
  const { currentPassword, newPassword, remember } = req.body || {};
  const user = db.findByUsername(db.findById(req.userId)?.username || '');
  if (!user) return res.status(404).json({ error: 'User not found' });
  if (user.password_hash && !(await verifyPassword(currentPassword, user.password_hash))) {
    return res.status(401).json({ error: 'Current password is wrong' });
  }
  const bad = passwordProblem(newPassword);
  if (bad) return res.status(400).json({ error: bad });
  if (newPassword.toLowerCase() === user.username.toLowerCase()) return res.status(400).json({ error: 'Password can\'t be your username' });
  db.setPassword(user.id, await hashPassword(newPassword));
  res.json({ ok: true, token: signToken(user.id, { remember: remember !== false }) });
});

// Check if a username is available
app.get('/api/username-available', (req, res) => {
  const { username } = req.query;
  if (!username || username.length < 2 || username.length > 20) {
    return res.json({ available: false, error: 'Username must be 2–20 characters' });
  }
  if (!/^[a-zA-Z0-9_]+$/.test(username)) {
    return res.json({ available: false, error: 'Letters, numbers, and underscores only' });
  }
  const existing = db.findByUsername(username);
  res.json({ available: !existing });
});

// Send OTP — rate limited
app.post('/api/send-otp', smslimit, smsOnly, async (req, res) => {
  const { phone } = req.body;
  if (!phone) return res.status(400).json({ error: 'Phone number required' });
  const e164 = normalisePhone(String(phone));
  const blocked = smsAllowed(e164);
  if (blocked) return res.status(400).json({ error: blocked });
  const result = await sendOtp(e164);
  if (result.error) return res.status(500).json(result);
  res.json({ ok: true, dev: result.dev || false });
});

// Register: verify OTP + create account → return JWT
app.post('/api/register', verifyLimit, smsOnly, async (req, res) => {
  const { username, phone, code } = req.body;
  if (!username || !phone || !code) return res.status(400).json({ error: 'Missing fields' });

  if (username.length < 2 || username.length > 20 || !/^[a-zA-Z0-9_]+$/.test(username)) {
    return res.status(400).json({ error: 'Invalid username' });
  }
  if (!isClean(username)) return res.status(400).json({ error: 'Please pick a different username' });

  const e164 = normalisePhone(phone);
  const check = await verifyOtp(e164, code);
  if (!check.valid) return res.status(400).json({ error: check.error });

  if (db.findByUsername(username)) return res.status(409).json({ error: 'Username already taken' });
  if (db.phoneHashExists(e164))    return res.status(409).json({ error: 'An account already exists for this phone number' });

  try {
    const user  = db.createUser(username, e164, encryptPhone(e164));
    const token = signToken(user.id);
    res.json({ ok: true, token, user: { id: user.id, username: user.username, elo: user.elo } });
  } catch (err) {
    console.error('register error:', err.message);
    res.status(500).json({ error: 'Could not create account' });
  }
});

// Account recovery: look up by phone hash → send OTP (rate limited)
// Used when someone forgets their username.
app.post('/api/login-by-phone', smslimit, smsOnly, async (req, res) => {
  const { phone } = req.body;
  if (!phone) return res.status(400).json({ error: 'Phone number required' });

  const e164 = normalisePhone(String(phone));
  const { hashPhone } = require('./db');
  const user = db.findByPhoneHash(hashPhone(e164));
  if (!user) return res.status(404).json({ error: 'No account found for that phone number' });

  const blocked = smsAllowed(e164);
  if (blocked) return res.status(400).json({ error: blocked });
  const result = await sendOtp(e164);
  if (result.error) return res.status(500).json(result);
  // Don't reveal the username until the code is verified (privacy: anyone
  // could otherwise look up who owns a phone number).
  res.json({ ok: true, dev: result.dev || false });
});

// Verify OTP sent via login-by-phone → return JWT
app.post('/api/login-by-phone/verify', verifyLimit, smsOnly, async (req, res) => {
  const { phone, code } = req.body;
  if (!phone || !code) return res.status(400).json({ error: 'Missing fields' });

  const e164  = normalisePhone(phone);
  const check = await verifyOtp(e164, code);
  if (!check.valid) return res.status(400).json({ error: check.error });

  const { hashPhone } = require('./db');
  const user = db.findByPhoneHash(hashPhone(e164));
  if (!user) return res.status(404).json({ error: 'Account not found' });

  const token = signToken(user.id);
  res.json({ ok: true, token, user: { id: user.id, username: user.username, elo: user.elo } });
});

// Login step 1: look up username → decrypt phone → send OTP (rate limited)
app.post('/api/login-by-username', smslimit, smsOnly, async (req, res) => {
  const { username } = req.body;
  if (!username) return res.status(400).json({ error: 'Username required' });

  const user = db.findByUsername(username);
  if (!user) return res.status(404).json({ error: 'No account found with that username' });
  if (!user.phone_encrypted) return res.status(400).json({ error: 'Account has no phone on file — contact support' });

  try {
    const e164   = decryptPhone(user.phone_encrypted);
    const blocked = smsAllowed(e164);
    if (blocked) return res.status(400).json({ error: blocked });
    const result = await sendOtp(e164);
    if (result.error) return res.status(500).json(result);
    res.json({ ok: true, dev: result.dev || false });
  } catch (err) {
    console.error('login-by-username error:', err.message);
    res.status(500).json({ error: 'Could not send code' });
  }
});

// Login step 2: verify OTP → return JWT
// Accepts { username, code } (username-based login)
app.post('/api/login', verifyLimit, smsOnly, async (req, res) => {
  const { username, code } = req.body;
  if (!username || !code) return res.status(400).json({ error: 'Missing fields' });

  const user = db.findByUsername(username);
  if (!user) return res.status(404).json({ error: 'No account found with that username' });
  if (!user.phone_encrypted) return res.status(400).json({ error: 'Account has no phone on file' });

  try {
    const e164  = decryptPhone(user.phone_encrypted);
    const check = await verifyOtp(e164, code);
    if (!check.valid) return res.status(400).json({ error: check.error });

    const token = signToken(user.id);
    res.json({ ok: true, token, user: { id: user.id, username: user.username, elo: user.elo } });
  } catch (err) {
    console.error('login error:', err.message);
    res.status(500).json({ error: 'Login failed' });
  }
});

// Get current user profile (requires auth)
app.get('/api/me', requireAuth, (req, res) => {
  const raw = db.findById(req.userId);
  if (!raw) return res.status(404).json({ error: 'User not found' });
  const { is_admin, token_version, ...user } = raw;
  user.has_password = !!user.has_password;
  user.role = is_admin ? 'admin' : 'member';
  // Add masked phone for display — never expose raw or encrypted value
  if (user.phone_encrypted) {
    try { user.phone_masked = maskPhone(decryptPhone(user.phone_encrypted)); }
    catch { user.phone_masked = null; }
  }
  const history = db.getHistory(req.userId);
  res.json({ user, history });
});

// Get game sessions the user participated in
app.get('/api/me/sessions', requireAuth, (req, res) => {
  const sessions = db.getUserSessions(req.userId, 20);
  res.json({ sessions });
});

// Toggle profile visibility
app.post('/api/me/visibility', requireAuth, (req, res) => {
  const { isPublic } = req.body;
  if (typeof isPublic !== 'boolean') return res.status(400).json({ error: 'isPublic must be a boolean' });
  db.setPublicProfile(req.userId, isPublic);
  res.json({ ok: true, public_profile: isPublic });
});

// Promote a user to admin — requires the ADMIN_PROMOTE_SECRET env var as a header.
// Usage: curl -X POST /api/admin/promote -H "X-Admin-Secret: <secret>" -H "Content-Type: application/json" -d '{"username":"yourname"}'
// Only run this once against your own account, then remove the secret from .env.
app.post('/api/admin/promote', (req, res) => {
  const secret = process.env.ADMIN_PROMOTE_SECRET;
  if (!secret) return res.status(403).json({ error: 'Admin promotion is disabled' });
  if (req.headers['x-admin-secret'] !== secret) return res.status(403).json({ error: 'Invalid secret' });
  const { username } = req.body;
  const user = db.findByUsername(username);
  if (!user) return res.status(404).json({ error: 'User not found' });
  db.setAdmin(user.id, true);
  res.json({ ok: true, message: `${username} is now an admin` });
});

// Submit user feedback (public — no auth required)
const feedbackLimit = rateLimit({ windowMs: 60 * 60 * 1000, max: 10, keyGenerator: byVisitor, message: { error: 'Too many submissions. Try again later.' } });
app.post('/api/feedback', feedbackLimit, async (req, res) => {
  const { type, message, name, page } = req.body;
  if (!message || message.trim().length < 5) return res.status(400).json({ error: 'Message too short' });
  if (!['bug', 'suggestion'].includes(type)) return res.status(400).json({ error: 'Invalid type' });
  // Optional name: a signed-in user's real username (verified), else what they typed.
  const header = req.headers.authorization || '';
  const payload = header.startsWith('Bearer ') ? verifyToken(header.slice(7)) : null;
  const user = payload ? db.findById(payload.sub) : null;
  const typed = typeof name === 'string' ? name.trim().slice(0, 40) : '';
  const contact = user ? `${user.username} (account)` : (typed ? `${typed} (guest)` : null);
  const result = db.addFeedback(type, message.trim().slice(0, 2000), contact, page?.slice(0, 100));

  // Auto-create GitHub issue for bugs (if GITHUB_TOKEN is set)
  let ghIssue = null;
  if (type === 'bug' && process.env.GITHUB_TOKEN) {
    try {
      const { Octokit } = await import('@octokit/rest').catch(() => null) || {};
      if (Octokit) {
        const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });
        const issue = await octokit.issues.create({
          owner: 'esvalC', repo: 'acquire-online',
          title: `[User Bug] ${message.slice(0, 80)}`,
          body: `**Type:** Bug report\n**Page:** ${page || 'unknown'}\n**Contact:** ${contact || 'none'}\n\n${message}`,
          labels: ['bug', 'user-report'],
        });
        ghIssue = issue.data.number;
        db.setFeedbackIssue(result.lastInsertRowid, ghIssue);
      }
    } catch (err) { console.error('GitHub issue creation failed:', err.message); }
  }

  res.json({ ok: true, id: result.lastInsertRowid, ghIssue });
});

// Recent game sessions (public — no auth required)
app.get('/api/recent-games', (req, res) => {
  const sessions = db.getRecentSessions(15).map(s => ({
    ...s,
    standings: (() => { try { return JSON.parse(s.standings); } catch { return []; } })()
      .map(p => ({ ...p, isAdmin: isAdminUser(p.userId) || undefined })),
  }));
  res.json({ sessions });
});

/* ── Site version: pages reload themselves after an update ─────
 * A fingerprint of the page the server is handing out. Browsers compare it
 * with the one they loaded and refresh at the next safe moment. */
const SITE_VERSION = (() => {
  try { return crypto.createHash('sha1').update(require('fs').readFileSync(path.join(__dirname, 'public', 'index.html'))).digest('hex').slice(0, 12); }
  catch { return String(Date.now()); }
})();
app.get('/api/version', (_req, res) => { res.set('Cache-Control', 'no-store'); res.json({ version: SITE_VERSION }); });

/* ── Leaderboard + your stats ───────────────────────────────── */
const lbName = n => isClean(n) ? n : censor(n);
app.get('/api/leaderboard', (req, res) => {
  const sort = ['rating', 'finished', 'multi', 'solo'].includes(req.query.sort) ? req.query.sort : 'rating';
  const rows = db.stats.leaderboard(sort, 100).map(r => ({ ...r, username: lbName(r.username), is_admin: !!r.is_admin }));
  res.json({ sort, rows });
});
app.get('/api/my-stats', requireAuth, (req, res) => {
  const user = db.findById(req.userId);
  if (!user) return res.status(404).json({ error: 'User not found' });
  const s = db.stats.get(req.userId);
  const { rank, total } = db.stats.rankOf(req.userId);
  const base = { id: user.id, username: user.username, isPublic: !!user.public_profile, rank, total };
  if (!s) return res.json({ ...base, empty: true, rating: 1000 });
  const now = Math.floor(Date.now() / 1000);
  const hist = s.history || [];
  const weekAgo = [...hist].reverse().find(h => h[0] <= now - 7 * 86400);
  const firstThisWeek = hist.find(h => h[0] > now - 7 * 86400);
  const weekStart = weekAgo ? weekAgo[1] : firstThisWeek ? (hist[hist.indexOf(firstThisWeek) - 1] || [0, 1000])[1] : s.rating;
  const favorites = Object.entries(s.chain_counts || {})
    .map(([chain, c]) => ({ chain, founded: c.f || 0, shares: c.b || 0, score: (c.b || 0) + 3 * (c.f || 0) }))
    .filter(c => c.score > 0).sort((a, b) => b.score - a.score).slice(0, 3);
  const friendIds = db.social.friendIds(req.userId);
  res.json({
    ...base, rating: s.rating, peak: s.peak_rating, week: s.rating - weekStart,
    games: s.games, wins: s.wins, finGames: s.fin_games, finWins: s.fin_wins,
    earlyGames: s.early_games, earlyWins: s.early_wins, quits: s.quits,
    multiGames: s.multi_games, multiWins: s.multi_wins, soloGames: s.solo_games, soloWins: s.solo_wins,
    avgPlace: s.fin_games ? Math.round((s.place_sum / s.fin_games) * 10) / 10 : null,
    avgCash: s.fin_games ? Math.round(s.cash_sum / s.fin_games / 100) * 100 : null,
    bestCash: s.best_cash, avgMargin: s.fin_wins ? Math.round(s.margin_sum / s.fin_wins / 100) * 100 : null,
    streak: s.streak, bestStreak: s.best_streak,
    chainsFounded: s.chains_founded, mergers: s.mergers, favorites,
    bots: s.bot_record, qmGames: s.qm_games, qmWins: s.qm_wins, frBest: s.fr_best,
    history: hist.slice(-30).map(h => h[1]),
    friends: db.stats.h2hFor(req.userId, friendIds),
  });
});

// Public profile card (stats + recent games). Respects the "Public Profile" setting.
app.get('/api/users/:id/profile', (req, res) => {
  const id = Number(req.params.id);
  const user = Number.isInteger(id) && id > 0 ? db.findById(id) : null;
  if (!user) return res.status(404).json({ error: 'Player not found' });
  const header = req.headers.authorization || '';
  const viewer = header.startsWith('Bearer ') ? verifyToken(header.slice(7)) : null;
  const isSelf = viewer && viewer.sub === id;
  const relation = viewer ? socialApi.relationFor(viewer.sub, id) : 'guest';
  const base = { id: user.id, username: user.username, isAdmin: !!user.is_admin, joined: user.created_at, relation,
    ...(relation === 'friends' ? { presence: socialApi.statusOf(id) } : {}) };
  if (!user.public_profile && !isSelf && !(viewer && isAdminUser(viewer.sub))) {
    return res.json({ ...base, private: true });
  }
  const cutoff = Math.floor(Date.now() / 1000) - db.REPLAY_DAYS * 86400;
  const recent = db.getUserSessions(id, 10).map(s => {
    let standings = [];
    try { standings = JSON.parse(s.standings || '[]'); } catch {}
    return {
      id: s.id, played_at: s.played_at, is_solo: !!s.is_solo, rank: s.rank, cash: s.my_cash,
      players: s.player_count, turns: s.turn_count,
      opponents: standings.filter(p => p.userId !== id).map(p => ({ name: p.name, isBot: !!p.isBot, userId: p.userId || null })),
      replay: s.played_at >= cutoff,
    };
  });
  res.json({ ...base, elo: user.elo, stats: db.getUserStats(id), recent });
});

// Replay data for a specific session (public)
app.get('/api/sessions/:id/replay', (req, res) => {
  const row = db.getSessionReplay(Number(req.params.id));
  if (!row) return res.status(404).json({ error: 'Session not found' });
  // Works for older replays too: admin names come from the saved standings
  const adminNames = (row.standings || []).filter(p => isAdminUser(p.userId)).map(p => p.name);
  const members = (row.standings || []).filter(p => p.userId && !p.isBot).map(p => ({ name: p.name, userId: p.userId }));
  res.json({ sessionId: row.id, snapshots: row.snapshots, expired: row.expired, replayDays: db.REPLAY_DAYS, adminNames, members });
});

// Plan dashboard data (admin only)
app.get('/api/plan/data', requireAdmin, (req, res) => {
  res.json({
    stats:    db.getStats(),
    feedback: db.getFeedback(500),
  });
});

// Update feedback status (admin only)
app.post('/api/plan/feedback/:id/status', requireAdmin, (req, res) => {
  const { status } = req.body;
  if (!['new', 'triaged', 'done'].includes(status)) return res.status(400).json({ error: 'Invalid status' });
  db.setFeedbackStatus(Number(req.params.id), status);
  res.json({ ok: true });
});

// Promote/demote a user to admin (admin only — no secret env var needed from dashboard)
app.post('/api/plan/users/promote', requireAdmin, (req, res) => {
  const { username, isAdmin } = req.body;
  if (!username) return res.status(400).json({ error: 'username required' });
  const user = db.findByUsername(username);
  if (!user) return res.status(404).json({ error: 'User not found' });
  if (user.id === req.userId) return res.status(400).json({ error: 'Cannot change your own admin status' });
  db.setAdmin(user.id, isAdmin !== false);
  res.json({ ok: true, username: user.username, isAdmin: isAdmin !== false });
});

// Admin: give a user a temporary password (shown once) — for people who
// forgot theirs or had a text-message-only account. Signs them out everywhere.
app.post('/api/plan/users/:id/reset-password', requireAdmin, async (req, res) => {
  const id = Number(req.params.id);
  const target = Number.isInteger(id) ? db.findById(id) : null;
  if (!target) return res.status(404).json({ error: 'User not found' });
  if (target.id === req.userId) return res.status(400).json({ error: 'Change your own password from your profile' });
  const alphabet = 'abcdefghjkmnpqrstuvwxyz23456789';
  const temp = Array.from(crypto.randomBytes(12), b => alphabet[b % alphabet.length]).join('')
    .replace(/(.{4})(?=.)/g, '$1-');
  db.setPassword(target.id, await hashPassword(temp));
  console.log(`[admin] ${req.userId} reset the password of user ${target.id}`);
  res.json({ ok: true, username: target.username, tempPassword: temp });
});

// List users (admin only)
app.get('/api/plan/users', requireAdmin, (req, res) => {
  const users = db.getAllUsers();
  res.json({ users });
});

/* ── Training dashboard stats (proxies S3, cached 60s) ────────── */
const { exec } = require('child_process');
let _trainingStatsCache = null;
let _trainingStatsCacheTime = 0;
app.get('/api/training-stats', (req, res) => {
  const now = Date.now();
  if (_trainingStatsCache && now - _trainingStatsCacheTime < 60000) {
    return res.json(_trainingStatsCache);
  }
  const bucket = process.env.S3_BUCKET || 'acquire-training-data';
  exec(`aws s3 cp s3://${bucket}/training_stats.json /tmp/training_stats_cache.json 2>/dev/null`, (err) => {
    if (err) {
      _trainingStatsCache = null; // clear so next request retries S3 rather than serving stale data
      _trainingStatsCacheTime = 0;
      return res.json({ error: 'Training not running or stats unavailable.' });
    }
    try {
      const data = JSON.parse(require('fs').readFileSync('/tmp/training_stats_cache.json', 'utf8'));
      _trainingStatsCache = data;
      _trainingStatsCacheTime = now;
      res.json(data);
    } catch {
      _trainingStatsCache = null;
      _trainingStatsCacheTime = 0;
      res.json({ error: 'Could not parse stats.' });
    }
  });
});

// Serve index.html for all known routes (client-side routing)
/* ── Total games counter ───────────────────────────────────── */
app.get('/api/stats/total-games', (_req, res) => {
  try { res.json(db.getTotalGames()); }
  catch (e) { console.error('total-games', e.message); res.status(500).json({ error: 'unavailable' }); }
});

/* ── News banner ───────────────────────────────────────────────
 * Admins post updates; anyone (guest or logged in) can reply. */
const newsReplyLimit = rateLimit({ windowMs: 60 * 60 * 1000, max: 20, keyGenerator: byVisitor, message: { error: 'Too many replies — try again later.' } });
const newsPostLimit  = rateLimit({ windowMs: 60 * 60 * 1000, max: 30, keyGenerator: byVisitor, message: { error: 'Too many posts — try again later.' } });
const cleanText = (v, max) => String(v == null ? '' : v).replace(/\r\n/g, '\n').trim().slice(0, max);

app.get('/api/news', (_req, res) => {
  try { res.json({ posts: db.listNews(10) }); }
  catch (e) { console.error('news', e.message); res.status(500).json({ error: 'unavailable' }); }
});

app.post('/api/news', newsPostLimit, requireAdmin, (req, res) => {
  const body = cleanText(req.body && req.body.body, 2000);
  if (!body) return res.status(400).json({ error: 'Post is empty' });
  const id = db.createNewsPost(req.userId, body);
  res.json({ ok: true, id });
});

app.delete('/api/news/:id', requireAdmin, (req, res) => {
  res.json({ ok: db.deleteNewsPost(Number(req.params.id)) > 0 });
});

app.post('/api/news/:id/replies', newsReplyLimit, (req, res) => {
  const postId = Number(req.params.id);
  if (!db.newsPostExists(postId)) return res.status(404).json({ error: 'Post not found' });
  const body = censor(cleanText(req.body && req.body.body, 500));
  if (!body) return res.status(400).json({ error: 'Reply is empty' });

  // Logged-in users reply as themselves; everyone else is a guest.
  const header = req.headers.authorization || '';
  const payload = header.startsWith('Bearer ') ? verifyToken(header.slice(7)) : null;
  const user = payload ? db.findById(payload.sub) : null;
  let guestName = cleanText(req.body && req.body.name, 24).replace(/[^\p{L}\p{N} _.'-]/gu, '') || 'Guest';
  if (!isClean(guestName)) guestName = 'Guest';
  const id = db.createNewsReply(postId, user ? user.id : null, guestName, body);
  res.json({ ok: true, id });
});

app.delete('/api/news/replies/:id', requireAdmin, (req, res) => {
  res.json({ ok: db.deleteNewsReply(Number(req.params.id)) > 0 });
});

const clientRoutes = ['/solo', '/multiplayer', '/multiplayer/:code([0-9]{4})', '/rules', '/leaderboard', '/tutorial', '/feedback', '/finance-road', '/finance-road/:tier/:level', '/quick-master', '/replay/:id', '/game/:code([0-9]{4})'];
app.get('/plan', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'plan.html')));
app.get('/training', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'training.html')));
for (const route of clientRoutes) {
  app.get(route, (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
}
// Legacy: bare 4-digit code still works
app.get('/:code([0-9]{4})', (req, res) => {
  res.redirect(`/multiplayer/${req.params.code}`);
});

// Last-resort error handler: log it, never send stack traces to visitors.
app.use((err, req, res, _next) => {
  if (err && err.type === 'entity.parse.failed') return res.status(400).json({ error: 'Invalid request' });
  if (err && err.type === 'entity.too.large')   return res.status(413).json({ error: 'Request too large' });
  console.error('[http]', req.method, req.path, err && (err.stack || err.message));
  res.status(500).json({ error: 'Something went wrong' });
});

/* ── Room store ────────────────────────────────────────────── */
const rooms = {};

/* Friends, invites, live games, friend chat, reports — see social.js */
const { censor, isClean } = require('./filter');
const ratings = require('./ratings');
const tutorial = require('./tutorial');
const socialApi = require('./social')({
  app, io, db, getRooms: () => rooms, verifyToken, requireAuth, requireAdmin, byVisitor, isAdminUser,
});

function generateCode() {
  let code;
  do { code = String(Math.floor(1000 + Math.random() * 9000)); } while (rooms[code]);
  return code;
}

/* ── State broadcasting ─────────────────────────────────────── */
/* Admin tag: shown next to an admin's name in lobbies, games, chat, replays
 * and recent games. Comes from the signed-in account (never the typed name). */
let _adminIds = { at: 0, set: new Set() };
function isAdminUser(userId) {
  if (!userId) return false;
  if (Date.now() - _adminIds.at > 60000) {
    try { _adminIds = { at: Date.now(), set: new Set(db.getAdminIds()) }; } catch {}
  }
  return _adminIds.set.has(userId);
}

function broadcastState(code) {
  const room = rooms[code];
  if (!room || !room.game) return;

  const playerList = room.players.map(p => ({ name: p.name, isBot: p.isBot || false, difficulty: p.isBot ? p.difficulty : undefined, isAdmin: isAdminUser(p.userId) || undefined }));

  // Capture spectator snapshot for replay before recording game end
  const spectatorState = engine.getClientState(room.game, -1);
  spectatorState.playerList = playerList;
  spectatorState.watching = room.spectators.length;
  spectatorState.watchers = room.spectators.map(s => s.name);
  spectatorState.code = code;
  spectatorState.mode = room.mode || (room.isSolo ? 'Solo' : 'Multiplayer');
  room.replaySnapshots = room.replaySnapshots || [];
  room.replaySnapshots.push(JSON.parse(JSON.stringify(spectatorState)));

  // Record game result BEFORE emitting so sessionId is available in the game-over state
  if (room.game.phase === 'gameOver' && !room.gameRecorded) {
    recordGameEnd(code);
  }

  for (const p of room.players) {
    if (p.socketId && !p.isBot) {
      const clientState = engine.getClientState(room.game, p.idx);
      clientState.playerList = playerList;
      clientState.watching = room.spectators.length;
      clientState.watchers = room.spectators.map(s => s.name);
      clientState.hintsAllowed = hintsAllowed(room);
      if (room.tutorial) clientState.tutorial = { chapter: room.tutorial.chapter, pos: room.tutorial.pos, total: tutorial.CHAPTERS[room.tutorial.chapter].script.length };
      clientState.code = code;
      if (room.sessionId) clientState.sessionId = room.sessionId;
      if (room.ratingResult && p.userId && room.ratingResult.changes[p.name] !== undefined) {
        clientState.myRating = { change: room.ratingResult.changes[p.name], rating: db.stats.rating(p.userId), kind: room.ratingResult.kind };
      }
      io.to(p.socketId).emit('gameState', clientState);
    }
  }

  for (const s of room.spectators) {
    io.to(s.socketId).emit('gameState', spectatorState);
  }

  // Schedule the next bot move (solo rooms, or multiplayer rooms with bots)
  if (roomHasBots(room) && room.game.phase !== 'gameOver') {
    scheduleBotTurn(code);
  }
  persistRoom(code);
}

/* ── Saving in-progress games (survive restarts and deploys) ─────
 * Every game in progress is written to SQLite (gzip, at most once per second
 * per room) whenever its state changes, flushed on shutdown, and reloaded on
 * startup. Players' browsers reconnect and rejoin on their own. */
const SAVE_THROTTLE_MS = 1000;

function serializeRoom(room) {
  const { cleanupTimer, _saveTimer, botScheduled, spectators, ...rest } = room;
  return {
    ...rest,
    spectators: [],
    players: room.players.map(p => ({ ...p, socketId: null })),
  };
}

function saveRoomNow(code) {
  const room = rooms[code];
  if (room) { clearTimeout(room._saveTimer); room._saveTimer = null; }
  try {
    if (room && room.game && room.game.phase !== 'gameOver') db.saveActiveRoom(code, serializeRoom(room));
    else db.deleteActiveRoom(code);
  } catch (e) { console.error(`[persist] save ${code} failed:`, e.message); }
}

function persistRoom(code) {
  const room = rooms[code];
  if (!room) return;
  if (!room.game || room.game.phase === 'gameOver') return saveRoomNow(code);
  if (room._saveTimer) return;                     // a save is already queued
  room._saveTimer = setTimeout(() => saveRoomNow(code), SAVE_THROTTLE_MS);
}

function deleteRoom(code) {
  const room = rooms[code];
  if (room) { clearTimeout(room.cleanupTimer); clearTimeout(room._saveTimer); room.players.forEach(p => clearDropTimer(code, p.name)); }
  delete rooms[code];
  try { db.deleteActiveRoom(code); } catch {}
}

function restoreRooms() {
  let restored = [];
  try { restored = db.loadActiveRooms(24 * 3600); }
  catch (e) { console.error('[persist] restore failed:', e.message); return; }
  for (const { code, room } of restored) {
    if (!room.game || room.game.phase === 'gameOver' || rooms[code]) continue;
    room.players.forEach(p => { if (!p.isBot) p.disconnected = true; });
    room.spectators = [];
    room.botScheduled = false;
    rooms[code] = room;
    scheduleAbandonedRoomCleanup(code);   // deleted if nobody rejoins in time
    room.players.forEach(p => scheduleDropTimer(code, p));
    scheduleBotTurn(code);                // resume if a bot was mid-turn
  }
  if (restored.length) console.log(`[persist] restored ${restored.length} game(s) in progress`);
}

function flushAllRoomsAndExit(signal) {
  console.log(`[persist] ${signal}: saving ${Object.keys(rooms).length} room(s) before exit`);
  for (const code of Object.keys(rooms)) saveRoomNow(code);
  process.exit(0);
}
process.on('SIGINT',  () => flushAllRoomsAndExit('SIGINT'));
process.on('SIGTERM', () => flushAllRoomsAndExit('SIGTERM'));

/* ── Bots in any room (solo, or added to a multiplayer lobby) ── */
function roomHasBots(room) { return !!room && room.players.some(p => p.isBot); }
function roomHost(room) { return room.players.find(p => !p.isBot) || null; }

// Bots go along with ending the game only once every human has voted yes,
// so one person can't end a multiplayer game by out-voting the others with bots.
function botsJoinConcede(room) {
  const g = room.game;
  if (!g || !g.concedeActive) return;
  const humans = room.players.filter(p => !p.isBot);
  if (!humans.every(h => g.concedeVotes[h.idx] === true)) return;
  for (const bp of room.players) {
    if (bp.isBot && g.concedeVotes[bp.idx] === undefined) {
      engine.submitConcedeVote(g, bp.idx, true);
      if (!g.concedeActive) break;
    }
  }
}

function broadcastLobby(code) {
  const room = rooms[code];
  if (!room) return;
  const lobbyData = {
    code,
    players: room.players.map(p => ({ name: p.name, ready: p.ready, isBot: p.isBot || false, difficulty: p.isBot ? p.difficulty : undefined, isAdmin: isAdminUser(p.userId) || undefined })),
    spectators: room.spectators.map(s => ({ name: s.name })),
    maxPlayers: room.config.maxPlayers,
    host: (roomHost(room) || {}).name || null,
    gameInProgress: !!room.game,
  };
  io.to(code).emit('lobbyUpdate', lobbyData);
}

/* ── ELO + game recording ───────────────────────────────────── */
function recordGameEnd(code) {
  const room = rooms[code];
  if (!room || !room.game || room.gameRecorded) return;
  room.gameRecorded = true;
  if (room.tutorial) return;                       // tutorials are never recorded or rated

  const standings = room.game.standings; // [{ name, cash }] sorted desc
  console.log(`[recordGameEnd] code=${code} standings=${JSON.stringify(standings)}`);
  if (!standings || standings.length === 0) return;
  recordSessionRow(room, standings, room.game.endReason && room.game.endReason.type || null);

  // Ratings + career stats (see ratings.js)
  try { room.ratingResult = ratings.recordGameOver(db, room); }
  catch (err) { console.error('[ratings] recordGameOver failed:', err.message); }
}

/* Every game that got going is added to the games counter + Recent Games —
 * finished, ended early by vote, or quit/abandoned partway ('quit'). */
function recordSessionRow(room, standings, endType) {
  const durationSeconds = room.startedAt ? Math.round((Date.now() - room.startedAt) / 1000) : null;
  const humanCount = room.players.filter(p => !p.isBot).length;
  const turnCount  = room.game.turnNumber ?? null;
  const sessionStandings = standings.map(s => {
    const rp = room.players.find(p => p.name === s.name);
    const left = !!(rp && rp.forfeited);
    return { name: s.name, cash: s.cash, isBot: !!(rp?.isBot && !left), userId: rp?.userId || rp?.leftUserId || null, ...(left ? { left: true } : {}) };
  });
  try {
    const sessionId = db.recordSession(room.isSolo, room.players.length, humanCount, durationSeconds, turnCount, sessionStandings, room.replaySnapshots || [],
      room.mode || (room.isSolo ? 'Solo' : 'Multiplayer'), endType);
    console.log(`[session] recorded: solo=${room.isSolo} players=${room.players.length} turns=${turnCount} id=${sessionId} snapshots=${(room.replaySnapshots||[]).length}`);
    room.sessionId = sessionId; // store so broadcastState can include it in gameOver state
  } catch (err) {
    console.error('[session] recordSession failed:', err.message);
  }
}
// A game that never reached its end: everyone quit or walked away.
function recordUnfinishedGame(room) {
  if (!room || room.tutorial || !room.game || room.gameRecorded || room.game.phase === 'gameOver' || !ratings.underway(room)) return;
  room.gameRecorded = true;
  const worth = engine.worthIfEndedNow(room.game);
  const standings = room.game.players.map((p, i) => ({ name: p.name, cash: worth[i] })).sort((a, b) => b.cash - a.cash);
  recordSessionRow(room, standings, 'quit');
}

/* ── Game start ─────────────────────────────────────────────── */
function startGame(code) {
  const room = rooms[code];
  const names = room.players.map(p => p.name);
  room.game = engine.createGame(names, { quickstart: room.config.quickstart });
  room.startedAt = Date.now();
  room.replaySnapshots = [];
  // createGame reorders players by quickstart draw — remap indices
  for (const rp of room.players) {
    rp.idx = room.game.players.findIndex(gp => gp.name === rp.name);
  }
  io.to(code).emit('gameStarted');
  broadcastState(code);
}

/* ── Last-resort legal move for a stuck bot ─────────────────── */
function forceDefaultBotMove(game, idx) {
  const HOTELS = ['Tower','Luxor','American','Worldwide','Festival','Imperial','Continental'];
  switch (game.phase) {
    case 'placeTile': {
      const { playable } = engine.getPlayableTiles(game);
      if (playable.length) engine.placeTile(game, idx, playable[0]);
      else engine.passTile(game, idx);
      break;
    }
    case 'chooseChain': {
      const free = HOTELS.find(ch => !game.chains[ch].active);
      if (free) engine.chooseChain(game, idx, free);
      break;
    }
    case 'chooseMergerSurvivor': {
      const opts = (game.pendingMerger && (game.pendingMerger.survivorOptions || game.pendingMerger.tiedChains)) || [];
      for (const ch of opts) { if (!engine.chooseMergerSurvivor(game, idx, ch).error) break; }
      break;
    }
    case 'mergerDecision':
      engine.mergerDecision(game, idx, { sell: 0, trade: 0 }); // hold everything
      break;
    case 'buyStock':
      engine.buyStock(game, idx, {});
      break;
  }
}

// How long a game survives with no human connected before it's thrown away.
// Solo games wait a full day, so you can walk away (laptop asleep, phone
// locked, tab closed) and pick up where you left off.
const RECONNECT_GRACE_MS = Number(process.env.RECONNECT_GRACE_MS) || 5 * 60 * 1000;   // env override is for tests
const SOLO_GRACE_MS = Number(process.env.SOLO_GRACE_MS) || 24 * 3600 * 1000;
const graceFor = room => room && room.isSolo && !room.tutorial && room.game && room.game.phase !== 'gameOver' ? SOLO_GRACE_MS : RECONNECT_GRACE_MS;

/* A player gives up their seat in a game in progress (Leave, or gone too long).
 * Members take a loss once the game is underway; a Medium bot plays the seat
 * from here so everyone else can finish. */
const dropTimers = new Map();          // `${code}|${name}` → timer
function clearDropTimer(code, name) { const k = `${code}|${name}`; clearTimeout(dropTimers.get(k)); dropTimers.delete(k); }
function forfeitSeat(code, p, why) {
  const room = rooms[code];
  if (!room || !room.game || room.game.phase === 'gameOver' || p.isBot || room.tutorial) return;
  clearDropTimer(code, p.name);
  const change = ratings.recordQuit(db, room, p);
  if (change !== null) console.log(`[ratings] ${p.name} forfeited ${code} (${why}): ${change >= 0 ? '+' : ''}${change}`);
  p.forfeited = true;
  if (room.isSolo) return;
  Object.assign(p, { isBot: true, difficulty: 'expert', personality: 'balanced', socketId: null, disconnected: false, token: null,
    leftUserId: p.userId || null, userId: null });
  room.game.log.push(`${p.name} ${why === 'left' ? 'left the game' : 'didn\'t come back'} — a bot plays their seat`);
  if (!room.players.some(q => !q.isBot)) { recordUnfinishedGame(room); deleteRoom(code); return; }
  broadcastState(code);
}
function scheduleDropTimer(code, p) {
  const room = rooms[code];
  if (!room || room.isSolo || !room.game || p.isBot) return;
  clearDropTimer(code, p.name);
  dropTimers.set(`${code}|${p.name}`, setTimeout(() => {
    dropTimers.delete(`${code}|${p.name}`);
    const r = rooms[code];
    if (r && r.players.includes(p) && p.disconnected && !p.socketId) forfeitSeat(code, p, 'gone');
  }, RECONNECT_GRACE_MS));
}

function scheduleAbandonedRoomCleanup(code) {
  const room = rooms[code];
  if (!room) return;
  clearTimeout(room.cleanupTimer);
  const grace = graceFor(room);
  room.cleanupTimer = setTimeout(() => {
    const r = rooms[code];
    if (!r) return;
    const humansGone = r.players.every(p => p.isBot || p.disconnected || !p.socketId);
    if (humansGone) {
      console.log(`Room ${code} abandoned for ${Math.round(grace / 60000)} min — deleting`);
      if (r.game && r.game.phase !== 'gameOver' && !r.tutorial) {
        for (const p of r.players) if (!p.isBot && !p.forfeited) { try { ratings.recordQuit(db, r, p); } catch {} }
        recordUnfinishedGame(r);
      }
      deleteRoom(code);
    }
  }, grace);
}

/* A player leaves a table that hasn't started (Leave button, closed tab,
 * back button, lost connection). The next person in line becomes host; when
 * the last person leaves, the table is wiped (bots and watchers sent home). */
function leaveLobbySeat(code, player) {
  const room = rooms[code];
  if (!room) return;
  room.players = room.players.filter(p => p !== player);
  room.players.forEach((p, i) => { p.idx = i; });
  if (!roomHost(room)) {
    io.to(code).emit('tableClosed', { reason: 'empty' });
    for (const id of io.sockets.adapter.rooms.get(code) || []) io.sockets.sockets.get(id)?.leave(code);
    deleteRoom(code);
    console.log(`Table ${code} closed — everyone left`);
  } else {
    broadcastLobby(code);
  }
}

/* ── Bot scheduling ─────────────────────────────────────────── */
function scheduleBotTurn(code) {
  const room = rooms[code];
  if (!room || !room.game || !roomHasBots(room) || room.tutorial) return;   // tutorial bots follow the script
  if (room.botScheduled) return;
  if (room.game.phase === 'gameOver') return;

  const game = room.game;
  let actingBotIdx = null;

  if (game.phase === 'mergerDecision' && game.pendingMerger) {
    const dp = game.pendingMerger.decidingPlayer;
    const bp = room.players.find(p => p.idx === dp && p.isBot);
    if (bp) actingBotIdx = dp;
  } else {
    const current = game.currentPlayerIdx;
    const bp = room.players.find(p => p.idx === current && p.isBot);
    if (bp) actingBotIdx = current;
  }

  if (actingBotIdx === null) return;

  room.botScheduled = true;
  const botPlayer = room.players.find(p => p.idx === actingBotIdx);
  const delay = 1800 + Math.random() * 2200;
  const startedAt = Date.now();
  const gameAtStart = game;

  // Search bots (Hard / Master) think in a worker thread while the normal
  // "bot is thinking" delay runs, so they cost the player no extra waiting.
  const level = SEARCH_BUDGETS[botPlayer.difficulty];
  // Declares the end if this bot is leading; its turn then continues normally
  // and the game finishes when the turn does.
  maybeBotDeclaresEnd(game, actingBotIdx);
  const wantsSearch = level
    && (['placeTile', 'buyStock', 'mergerDecision'].includes(game.phase)
        || (level.full && ['chooseChain', 'chooseMergerSurvivor'].includes(game.phase)));
  if (game.phase === 'gameOver') { room.botScheduled = false; broadcastState(code); return; }
  const turnKey = botFingerprint(game); // after any declaration above

  const thinking = wantsSearch
    ? runSearchInWorker(game, actingBotIdx, SEARCH_BUDGETS[botPlayer.difficulty])
    : Promise.resolve(null);

  thinking.then(action => {
    const wait = Math.max(0, delay - (Date.now() - startedAt));
    setTimeout(() => {
      room.botScheduled = false;
      const r = rooms[code];
      // Game restarted/abandoned, or state moved on while we were thinking
      if (!r || r.game !== gameAtStart || botFingerprint(r.game) !== turnKey) {
        if (r && r.game) scheduleBotTurn(code);
        return;
      }
      playBotMove(code, r.game, botPlayer, action);
    }, wait);
  });
}

// Thinking budgets per difficulty live in ai/searchBot.js (SEARCH_LEVELS).
const SEARCH_BUDGETS = SEARCH_LEVELS;
// Button → code: Easy='hard' (heuristic), Medium='expert', Hard='mcts', Master='master'
const PLAYABLE_DIFFICULTIES = ['hard', 'expert', 'mcts', 'master'];

function botFingerprint(game) {
  return JSON.stringify([game.phase, game.currentPlayerIdx, game.turnNumber, game.log.length,
    game.pendingMerger && game.pendingMerger.currentPlayerOffset,
    game.pendingMerger && game.pendingMerger.currentDefunctIdx]);
}

function maybeBotDeclaresEnd(game, idx) {
  try { return maybeDeclareEnd(game, idx); } catch { return false; }
}

/* ── Worker thread for search bots ───────────────────────────── */
const { Worker } = require('worker_threads');
let searchWorker = null;
let searchSeq = 0;
const searchPending = new Map();

function getSearchWorker() {
  if (searchWorker) return searchWorker;
  searchWorker = new Worker(path.join(__dirname, 'ai', 'botWorker.js'));
  searchWorker.on('message', ({ id, action, error, info }) => {
    const done = searchPending.get(id);
    if (!done) return;
    searchPending.delete(id);
    if (error) console.error('Search bot error:', error);
    done(action || null, info || null);
  });
  const reset = (why) => {
    console.error('Search worker stopped:', why);
    searchWorker = null;
    for (const done of searchPending.values()) done(null); // fall back to heuristic
    searchPending.clear();
  };
  searchWorker.on('error', reset);
  searchWorker.on('exit', reset);
  return searchWorker;
}

function runSearchInWorker(game, me, opts) {
  return new Promise(resolve => {
    const id = ++searchSeq;
    // Never wait forever: queue time + budget + slack
    const timer = setTimeout(() => { if (searchPending.delete(id)) resolve(null); }, opts.budgetMs + 15000);
    searchPending.set(id, a => { clearTimeout(timer); resolve(a); });
    try {
      const { log, ...rest } = game; // log isn't needed and can be large
      getSearchWorker().postMessage({ id, game: { ...rest, log: [] }, me, opts });
    } catch (e) {
      searchPending.delete(id); clearTimeout(timer);
      console.error('Could not start search:', e.message);
      resolve(null);
    }
  });
}

// Suggested move for a player: same search, with "how often this won" info
function runHintInWorker(game, me) {
  return new Promise(resolve => {
    const id = ++searchSeq;
    const timer = setTimeout(() => { if (searchPending.delete(id)) resolve(null); }, HINT_LEVEL.budgetMs + 15000);
    searchPending.set(id, (action, info) => { clearTimeout(timer); resolve(action ? { action, info } : null); });
    try {
      const { log, ...rest } = game;
      getSearchWorker().postMessage({ id, game: { ...rest, log: [] }, me, opts: HINT_LEVEL });
    } catch (e) { searchPending.delete(id); clearTimeout(timer); resolve(null); }
  });
}
// Hints exist only in regular solo games — never multiplayer, Quick Master or Finance Road
const hintsAllowed = room => !!(room && room.isSolo && room.mode === 'Solo');

function playBotMove(code, game, botPlayer, searchAction) {
  const idx = botPlayer.idx;
  const before = botFingerprint(game);

  if (searchAction) {
    const res = applySearchAction(game, idx, searchAction);
    if (res && res.error) console.error(`Search move rejected in room ${code}: ${res.error}`);
  }
  if (botFingerprint(game) === before) {
    // Heuristic for phases search doesn't handle (naming a chain, etc.).
    // Search difficulties use the strongest heuristic here.
    try {
      // heuristicOnly: never run a (blocking) search on the main thread
      decideBotAction(game, idx, botPlayer.personality, botPlayer.difficulty, botPlayer.name, { heuristicOnly: true });
    } catch (e) {
      console.error(`Bot error in room ${code} (${botPlayer.difficulty}):`, e.stack || e.message);
    }
  }
  // Safety net: a move that threw or was rejected leaves the state unchanged,
  // and re-scheduling would retry it forever (the "game just stops" bug).
  if (botFingerprint(game) === before) {
    try { decideBotAction(game, idx, botPlayer.personality, 'hard', botPlayer.name); }
    catch (e) { console.error(`Bot fallback error in room ${code}:`, e.stack || e.message); }
  }
  if (botFingerprint(game) === before) {
    forceDefaultBotMove(game, idx);
    console.error(`Bot ${botPlayer.name} stalled in room ${code} (phase ${game.phase}) — forced default move`);
  }
  broadcastState(code);
}

/* ── Socket events ─────────────────────────────────────────── */
/* Player-chosen display names: 1–20 chars, no HTML/quote/control characters. */
function cleanName(raw) {
  const s = String(raw == null ? '' : raw).normalize('NFC')
    .replace(/[\u0000-\u001f\u007f<>"'`\\]/g, '').replace(/\s+/g, ' ').trim().slice(0, 20);
  return s && isClean(s) ? s : 'Player';
}
function uniqueName(room, name) {
  const taken = new Set(room.players.map(p => p.name.toLowerCase()).concat(room.spectators.map(s => s.name.toLowerCase())));
  if (!taken.has(name.toLowerCase())) return name;
  for (let i = 2; i < 100; i++) { const n = `${name.slice(0, 17)} ${i}`; if (!taken.has(n.toLowerCase())) return n; }
  return name;
}

io.on('connection', (socket) => {
  let currentRoom = null;
  let playerName = null;

  // Every handler is wrapped: a malformed or missing payload used to throw an
  // uncaught exception and crash the whole server (all games lost). Now the
  // payload defaults to {} and any error is logged, not fatal.
  const rawOn = socket.on.bind(socket);
  socket.on = (event, handler) => rawOn(event, (...args) => {
    if (args.length === 0 || args[0] == null || typeof args[0] !== 'object') {
      if (typeof args[0] === 'function') args.unshift({}); else args[0] = {};
    }
    const cbIdx = args.findIndex(a => typeof a === 'function');
    if (cbIdx === -1) args.push(() => {});          // handlers may call cb()
    try {
      const r = handler(...args);
      if (r && typeof r.catch === 'function') r.catch(e => console.error(`[socket ${event}]`, e.stack || e.message));
    } catch (e) {
      console.error(`[socket ${event}]`, e.stack || e.message);
    }
  });
  let chatTimes = [];

  // Game actions from a browser the server can't match to a live game used to
  // be ignored silently — the board then looked frozen (the "turn 73" reports).
  // Now the browser is told, so it can rejoin or say the game was lost.
  function actorFor(sock, code) {
    const room = rooms[code];
    const p = room && room.game ? room.players.find(x => x.socketId === sock.id) : null;
    if (!p) sock.emit('sessionLost', { reason: room ? 'not-in-room' : 'no-game' });
    return { room, p };
  }

  /* ── Tutorial ─────────────────────────────────── */
  // The player's move must be the next one in the chapter's script.
  function tutorialMove(room, p, type, args) {
    const t = room.tutorial, script = tutorial.CHAPTERS[t.chapter].script, step = script[t.pos];
    if (!tutorial.matches(step, type, args)) {
      const want = step && step.p === tutorial.ME ? tutorial.describe(step) : 'wait for the bubble';
      return socket.emit('actionError', `Tutorial: ${want}`);
    }
    const result = tutorial.apply(room.game, step);
    if (result.error) return socket.emit('actionError', result.error);
    t.pos++;
    broadcastState(currentRoom);
  }
  socket.on('startTutorial', ({ chapter, name } = {}, cb = () => {}) => {
    const ch = Number.isInteger(chapter) && tutorial.CHAPTERS[chapter] ? chapter : 0;
    name = cleanName(name || 'You');
    if (['aria', 'rex'].includes(name.toLowerCase())) name = 'You';
    detach();
    const code = generateCode();
    enter(code);
    playerName = name;
    const token = randomUUID();
    rooms[code] = {
      players: [
        { name, socketId: socket.id, ready: true, idx: 0, token, isBot: false, userId: null },
        { name: 'Aria', socketId: null, ready: true, idx: 1, token: null, isBot: true, personality: 'balanced', difficulty: 'hard' },
        { name: 'Rex',  socketId: null, ready: true, idx: 2, token: null, isBot: true, personality: 'focused',  difficulty: 'hard' },
      ],
      spectators: [], isSolo: true, mode: 'Tutorial', tutorial: { chapter: ch, pos: 0 },
      config: { maxPlayers: 3, quickstart: true },
      game: tutorial.build(name, tutorial.CHAPTERS[ch].setup),
      startedAt: Date.now(), replaySnapshots: [],
    };
    socket.join(code);
    cb({ ok: true, code, token, chapter: ch, chapters: tutorial.CHAPTERS.length });
    broadcastState(code);
  });
  // The tutorial on screen says "go": play the next scripted bot move.
  socket.on('tutorialBot', (_d, cb = () => {}) => {
    const room = rooms[currentRoom];
    if (!room || !room.tutorial || !room.game) return cb({ error: 'No tutorial' });
    if (!room.players.some(p => p.socketId === socket.id)) return cb({ error: 'Not your tutorial' });
    const t = room.tutorial, step = tutorial.CHAPTERS[t.chapter].script[t.pos];
    if (!step || step.p === tutorial.ME) return cb({ error: 'Not a bot move' });
    const result = tutorial.apply(room.game, step);
    if (result.error) { console.error('[tutorial] scripted move failed', t.chapter, t.pos, result.error); return cb({ error: result.error }); }
    t.pos++;
    broadcastState(currentRoom);
    cb({ ok: true, pos: t.pos });
  });

  /* ── Multiplayer: create room ── */
  socket.on('createRoom', ({ name, maxPlayers, quickstart, authToken }, cb) => {
    name = cleanName(name);
    const code = generateCode();
    enter(code);
    playerName = name;
    const token = randomUUID();
    const payload = authToken ? verifyToken(authToken) : null;
    const userId  = payload ? payload.sub : null;

    rooms[code] = {
      players: [{ name, socketId: socket.id, ready: false, idx: 0, token, userId }],
      spectators: [],
      game: null,
      isSolo: false,
      config: { maxPlayers: Math.min(Math.max(maxPlayers || 4, 2), 6), quickstart: quickstart !== false },
    };

    socket.join(code);
    cb({ ok: true, code, token });
    broadcastLobby(code);
  });

  /* ── Solo: create & start a solo game ── */
  socket.on('createSoloGame', ({ name, botCount, quickstart, botConfigs, mode, authToken, frLevel }, cb) => {
    name = cleanName(name);
    if (BOT_ROSTER.some(b => b.name.toLowerCase() === name.toLowerCase())) name = name.slice(0, 17) + ' (you)';
    if (!Array.isArray(botConfigs)) botConfigs = [];
    const code = generateCode();
    enter(code);
    playerName = name;
    const token = randomUUID();
    const payload = authToken ? verifyToken(authToken) : null;
    const userId  = payload ? payload.sub : null;

    const count = Math.min(Math.max(botCount || 1, 1), 5);
    const bots = BOT_ROSTER.slice(0, count);

    const players = [
      { name, socketId: socket.id, ready: true, idx: 0, token, isBot: false, userId },
      ...bots.map((b, i) => {
        const cfg = (botConfigs && botConfigs[i]) || {};
        // Finance Road and custom modes can override the bot name (must be a valid roster name)
        const botEntry = cfg.botName ? BOT_ROSTER.find(x => x.name === cfg.botName) : null;
        const resolved = botEntry || b;
        return {
          name: resolved.name, socketId: null, ready: true,
          idx: i + 1, token: null, isBot: true,
          personality: cfg.personality || resolved.personality || 'balanced',
          // Only the four real levels; anything else (old codes like 'easy'/'medium',
          // experimental 'lab-*' levels, missing values) becomes Medium.
          difficulty: PLAYABLE_DIFFICULTIES.includes(cfg.difficulty) ? cfg.difficulty
            : PLAYABLE_DIFFICULTIES.includes(resolved.difficulty) ? resolved.difficulty : 'expert',
        };
      }),
    ];

    rooms[code] = {
      players,
      spectators: [],
      game: null,
      isSolo: true,
      mode: typeof mode === 'string' && mode.trim() ? mode.trim().slice(0, 60) : 'Solo',
      frLevel: Number.isInteger(frLevel) && frLevel >= 1 && frLevel <= 15 && /^Finance Road/.test(mode || '') ? frLevel : null,
      config: { maxPlayers: players.length, quickstart: quickstart !== false },
    };

    socket.join(code);
    startGame(code);
    cb({ ok: true, code, token });
  });

  socket.on('getRoomStatus', ({ code }, cb) => {
    const room = rooms[code];
    if (!room) return cb({ exists: false });
    cb({
      exists: true,
      gameInProgress: !!room.game,
      over: !!(room.game && room.game.phase === 'gameOver'),
      playerCount: room.players.length,
      maxPlayers: room.config.maxPlayers,
    });
  });

  socket.on('joinRoom', ({ name, code, spectate, authToken }, cb) => {
    const room = rooms[code];
    if (!room) return cb({ error: 'Room not found' });
    name = uniqueName(room, cleanName(name));

    enter(code);
    playerName = name;
    const payload = authToken ? verifyToken(authToken) : null;
    const userId  = payload ? payload.sub : null;

    if (spectate || room.game || room.players.length >= room.config.maxPlayers) {
      room.spectators.push({ name, socketId: socket.id });
      cb({ ok: true, spectator: true });
      broadcastLobby(code);
      if (room.game) broadcastState(code);
      return;
    }

    const token = randomUUID();
    room.players.push({ name, socketId: socket.id, ready: false, idx: room.players.length, token, userId });
    cb({ ok: true, spectator: false, token });
    broadcastLobby(code);
  });

  socket.on('rejoinRoom', ({ code, token, name }, cb) => {
    const room = rooms[code];
    if (!room) return cb({ error: 'Room not found' });

    // The seat token is a random UUID only this player's browser has — enough on its own.
    const p = typeof token === 'string' && token ? room.players.find(p => p.token === token) : null;
    if (!p) return cb({ error: 'Session not found' });
    name = p.name;

    p.socketId = socket.id;
    p.disconnected = false;
    clearTimeout(room.cleanupTimer);
    clearDropTimer(code, p.name);
    enter(code);
    playerName = name;

    cb({ ok: true, inGame: !!room.game, isSolo: room.isSolo });
    if (room.game) {
      room.game.log.push(`${name} reconnected`);
      broadcastState(code);
    } else {
      broadcastLobby(code);
    }
  });

  /* ── Multiplayer lobby: host adds / removes bots ── */
  socket.on('addBot', ({ difficulty } = {}, cb = () => {}) => {
    const room = rooms[currentRoom];
    if (!room || room.isSolo || room.game) return cb({ error: 'Bots can only be added in the lobby' });
    const me = room.players.find(p => p.socketId === socket.id);
    if (!me || me !== roomHost(room)) return cb({ error: 'Only the host can add bots' });
    if (room.players.length >= room.config.maxPlayers) return cb({ error: 'The table is full' });
    const taken = new Set(room.players.map(p => p.name.toLowerCase()));
    const pick = BOT_ROSTER.find(b => !b.difficulty && !taken.has(b.name.toLowerCase()));
    if (!pick) return cb({ error: 'No more bots available' });
    room.players.push({
      name: pick.name, socketId: null, ready: true, idx: room.players.length, token: null, isBot: true,
      personality: pick.personality || 'balanced',
      difficulty: PLAYABLE_DIFFICULTIES.includes(difficulty) ? difficulty : 'expert',
    });
    broadcastLobby(currentRoom);
    cb({ ok: true });
  });

  socket.on('removeBot', ({ name } = {}) => {
    const room = rooms[currentRoom];
    if (!room || room.isSolo || room.game) return;
    const me = room.players.find(p => p.socketId === socket.id);
    if (!me || me !== roomHost(room)) return;
    room.players = room.players.filter(p => !(p.isBot && p.name === name));
    room.players.forEach((p, i) => { p.idx = i; });
    broadcastLobby(currentRoom);
  });

  socket.on('toggleReady', () => {
    const room = rooms[currentRoom];
    if (!room) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (p) {
      p.ready = !p.ready;
      broadcastLobby(currentRoom);

      if (room.players.length >= 2 && room.players.every(p => p.ready) && !room.game) {
        startGame(currentRoom);
      }
    }
  });

  /* ── Game actions ─────────────────────────────── */

  socket.on('placeTile', ({ tile }) => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    if (room.tutorial) return tutorialMove(room, p, 'placeTile', { tile });
    const result = engine.placeTile(room.game, p.idx, tile);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('chooseChain', ({ chain }) => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    if (room.tutorial) return tutorialMove(room, p, 'chooseChain', { chain });
    const result = engine.chooseChain(room.game, p.idx, chain);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('chooseMergerSurvivor', ({ chain }) => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    if (room.tutorial) return tutorialMove(room, p, 'chooseMergerSurvivor', { chain });
    const result = engine.chooseMergerSurvivor(room.game, p.idx, chain);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('mergerDecision', ({ sell, trade }) => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    if (room.tutorial) return tutorialMove(room, p, 'mergerDecision', { sell, trade });
    const result = engine.mergerDecision(room.game, p.idx, { sell, trade });
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('buyStock', ({ purchases }) => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    if (room.tutorial) return tutorialMove(room, p, 'buyStock', { purchases });
    const result = engine.buyStock(room.game, p.idx, purchases || {});
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('declareGameEnd', () => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    if (room.tutorial) return tutorialMove(room, p, 'declareGameEnd', {});
    const result = engine.declareGameEnd(room.game, p.idx);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  /* ── Concede vote ─────────────────────────────── */

  socket.on('initiateConcedeVote', () => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    if (room.tutorial) return socket.emit('actionError', 'Not in the tutorial — use Exit at the top to leave.');
    const result = engine.initiateConcedeVote(room.game, p.idx);
    if (result.error) return socket.emit('actionError', result.error);
    botsJoinConcede(room);
    broadcastState(currentRoom);
  });

  socket.on('submitConcedeVote', ({ vote }) => {
    const { room, p } = actorFor(socket, currentRoom);
    if (!p) return;
    const result = engine.submitConcedeVote(room.game, p.idx, vote);
    if (result.error) return socket.emit('actionError', result.error);
    botsJoinConcede(room);
    broadcastState(currentRoom);
  });

  /* ── Chat ───────────────────────────────────────── */

  socket.on('chatMessage', ({ message }) => {
    if (!currentRoom || typeof message !== 'string' || !message.trim()) return;
    const now = Date.now();
    chatTimes = chatTimes.filter(t => now - t < 10000);
    if (chatTimes.length >= 8) return;               // max 8 messages per 10 seconds
    chatTimes.push(now);
    const room = rooms[currentRoom];
    if (!room) return;
    const name = playerName || 'Anonymous';
    const msg = censor(message.trim().slice(0, 300));
    const sender = room.players.find(p => p.socketId === socket.id);
    io.to(currentRoom).emit('chatMessage', { code: currentRoom, name, message: msg, time: Date.now(), isAdmin: isAdminUser(sender && sender.userId) || undefined });
  });

  /* ── Play again ───────────────────────────────── */

  socket.on('playAgain', () => {
    const room = rooms[currentRoom];
    if (!room) return;

    if (room.isSolo) {
      // For solo: restart with the same bots
      room.game = null;
      room.players.forEach(p => { p.ready = true; });
      socket.emit('soloRestart');
      return;
    }

    room.game = null;
    room.players.forEach(p => { p.ready = !!p.isBot; }); // bots are always ready

    while (room.spectators.length > 0 && room.players.length < room.config.maxPlayers) {
      const s = room.spectators.shift();
      const token = randomUUID();
      room.players.push({ name: s.name, socketId: s.socketId, ready: false, idx: room.players.length, token });
      io.to(s.socketId).emit('sessionToken', { token });
    }

    io.to(currentRoom).emit('backToLobby');
    broadcastLobby(currentRoom);
  });

  /* ── Solo: play again (restart immediately) ─── */
  socket.on('soloPlayAgain', () => {
    const room = rooms[currentRoom];
    if (!room || !room.isSolo) return;
    room.game = null;
    room.players.forEach(p => { p.ready = true; });
    startGame(currentRoom);
  });

  /* Step away from the current game/table without ending it — what a closed
   * tab does: spectators stop watching, a waiting seat is freed, a player in a
   * game is marked away (they can come back for a few minutes). Also used when
   * this connection moves to a different game, so it never keeps receiving
   * another game's updates or chat. */
  function detach() {
    const code = currentRoom;
    for (const r of socket.rooms) if (r !== socket.id && !String(r).startsWith('user:')) socket.leave(r);
    currentRoom = null;
    if (!code || !rooms[code]) return;
    const currentRoomCode = code;
    const room = rooms[currentRoomCode];
    const watchersBefore = room.spectators.length;
    room.spectators = room.spectators.filter(s => s.socketId !== socket.id);
    if (room.spectators.length !== watchersBefore && room.game) io.to(currentRoomCode).emit('watching', { n: room.spectators.length, names: room.spectators.map(s => s.name) });

    if (room.isSolo) {
      // Don't delete immediately: phones locking, Wi-Fi blips and tab switches
      // all drop the socket. The player can come back via /game/<code> (or the
      // auto-rejoin on reconnect) within RECONNECT_GRACE_MS.
      const human = room.players.find(p => p.socketId === socket.id);
      if (human) { human.socketId = null; human.disconnected = true; }
      scheduleAbandonedRoomCleanup(currentRoomCode);
      return;
    }

    if (room.game) {
      const p = room.players.find(p => p.socketId === socket.id);
      if (p) {
        p.socketId = null;
        p.disconnected = true;
        room.game.log.push(`${p.name} disconnected`);
        broadcastState(currentRoomCode);
        scheduleDropTimer(currentRoomCode, p);
        if (room.players.every(q => q.isBot || q.disconnected)) scheduleAbandonedRoomCleanup(currentRoomCode);
      }
    } else {
      const me = room.players.find(p => p.socketId === socket.id);
      if (me) leaveLobbySeat(currentRoomCode, me);
      else broadcastLobby(currentRoomCode);
    }
  }
  // Join a game/table room, leaving any other first
  function enter(code) {
    if (currentRoom && currentRoom !== code) detach();
    for (const r of socket.rooms) if (r !== socket.id && r !== code && !String(r).startsWith('user:')) socket.leave(r);
    currentRoom = code;
    socket.join(code);
  }

  /* ── Suggested move (solo "Solo" games only) ───── */
  socket.on('getHint', async (_d, cb) => {
    const code = currentRoom, room = code && rooms[code];
    if (!hintsAllowed(room) || !room.game) return cb({ error: 'Hints aren\'t available in this game' });
    const me = room.players.find(p => p.socketId === socket.id && !p.isBot);
    if (!me) return cb({ error: 'Only the player can get hints' });
    const g = room.game;
    const deciding = g.phase === 'mergerDecision' && g.pendingMerger ? g.pendingMerger.decidingPlayer : g.currentPlayerIdx;
    if (deciding !== me.idx || g.phase === 'gameOver') return cb({ error: 'Not your decision right now' });
    const key = [g.turnNumber, g.phase, deciding, g.pendingMerger ? g.pendingMerger.currentDefunctIdx : ''].join('|');
    if (room._hint && room._hint.key === key) return cb(await room._hint.promise);  // one search per decision
    const promise = runHintInWorker(g, me.idx).then(r => r ? { ok: true, key, action: r.action, info: r.info } : { error: 'No suggestion this time' });
    room._hint = { key, promise };
    const out = await promise;
    const g2 = rooms[code] && rooms[code].game;
    const key2 = g2 && [g2.turnNumber, g2.phase, g2.phase === 'mergerDecision' && g2.pendingMerger ? g2.pendingMerger.decidingPlayer : g2.currentPlayerIdx, g2.pendingMerger ? g2.pendingMerger.currentDefunctIdx : ''].join('|');
    cb(key2 === key ? out : { stale: true });
  });

  /* ── Leave room (e.g. solo player goes back to setup) ── */
  // Solo: ends the game. Multiplayer table (before it starts): the host
  // cancels the table for everyone; anyone else just leaves their seat.
  // During a multiplayer game it counts as dropping out (others play on).
  socket.on('leaveRoom', () => {
    const code = currentRoom;
    const room = code && rooms[code];
    currentRoom = null;
    if (!room) return;
    socket.leave(code);
    const me = room.players.find(p => p.socketId === socket.id);

    if (!me) {                                              // spectator (any kind of game, solo too)
      room.spectators = room.spectators.filter(s => s.socketId !== socket.id);
      if (!room.game) broadcastLobby(code); else io.to(code).emit('watching', { n: room.spectators.length, names: room.spectators.map(s => s.name) });
      return;
    }
    if (room.isSolo) {
      if (room.game && room.game.phase !== 'gameOver') { forfeitSeat(code, me, 'left'); recordUnfinishedGame(room); }
      deleteRoom(code); return;
    }
    if (room.game) {
      if (room.game.phase === 'gameOver') { me.socketId = null; me.disconnected = true; return; }
      forfeitSeat(code, me, 'left');
      return;
    }
    leaveLobbySeat(code, me);
  });

  /* ── Disconnect ───────────────────────────────── */

  socket.on('disconnect', () => detach());
});

const PORT = process.env.PORT || 3000;
// HOST=127.0.0.1 in production (only nginx needs to reach it); 0.0.0.0 locally for LAN play
server.listen(PORT, process.env.HOST || '0.0.0.0', () => {
  console.log(`Acquire server running on http://localhost:${PORT}`);
  restoreRooms();
  // Replays are kept REPLAY_DAYS days, then deleted (checked at startup and every 6h)
  const prune = () => { try { const n = db.pruneReplays(); if (n) console.log(`[replays] removed ${n} expired replay(s)`); } catch (e) { console.error('[replays]', e.message); } };
  prune();
  setInterval(prune, 6 * 3600 * 1000).unref();
});
