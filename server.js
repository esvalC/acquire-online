/**
 * Acquire – Server (v3.1 – profiles branch)
 * Express + Socket.io
 * Supports multiplayer (lobby), solo (vs bots), and user profiles.
 */

require('dotenv').config();

const express    = require('express');
const http       = require('http');
const path       = require('path');
const { randomUUID } = require('crypto');
const { Server } = require('socket.io');
const rateLimit  = require('express-rate-limit');
const engine     = require('./gameEngine');
const { decideBotAction, BOT_ROSTER } = require('./botAI');
const db         = require('./db');
const { signToken, verifyToken, requireAuth, requireAdmin, sendOtp, verifyOtp, normalisePhone, encryptPhone, decryptPhone, maskPhone } = require('./auth');

const app    = express();
const server = http.createServer(app);
const io     = new Server(server, { cors: { origin: ['https://playonlineacquire.com', 'http://localhost:3000'] } });

app.use(express.json());
app.use(express.static('public'));

/* ── Rate limiters ───────────────────────────────────────────── */
const smslimit = rateLimit({
  windowMs: 10 * 60 * 1000, max: 5,
  message: { error: 'Too many SMS requests. Wait 10 minutes.' },
  standardHeaders: true, legacyHeaders: false,
});

/* ── Auth / Profile routes ───────────────────────────────────── */

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
app.post('/api/send-otp', smslimit, async (req, res) => {
  const { phone } = req.body;
  if (!phone) return res.status(400).json({ error: 'Phone number required' });
  const e164 = normalisePhone(phone);
  const result = await sendOtp(e164);
  if (result.error) return res.status(500).json(result);
  res.json({ ok: true, dev: result.dev || false });
});

// Register: verify OTP + create account → return JWT
app.post('/api/register', async (req, res) => {
  const { username, phone, code } = req.body;
  if (!username || !phone || !code) return res.status(400).json({ error: 'Missing fields' });

  if (username.length < 2 || username.length > 20 || !/^[a-zA-Z0-9_]+$/.test(username)) {
    return res.status(400).json({ error: 'Invalid username' });
  }

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
app.post('/api/login-by-phone', smslimit, async (req, res) => {
  const { phone } = req.body;
  if (!phone) return res.status(400).json({ error: 'Phone number required' });

  const e164 = normalisePhone(phone);
  const { hashPhone } = require('./db');
  const user = db.findByPhoneHash(hashPhone(e164));
  if (!user) return res.status(404).json({ error: 'No account found for that phone number' });

  const result = await sendOtp(e164);
  if (result.error) return res.status(500).json(result);
  // Return the masked username so the UI can say "Logging in as ___"
  res.json({ ok: true, username: user.username, dev: result.dev || false });
});

// Verify OTP sent via login-by-phone → return JWT
app.post('/api/login-by-phone/verify', async (req, res) => {
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
app.post('/api/login-by-username', smslimit, async (req, res) => {
  const { username } = req.body;
  if (!username) return res.status(400).json({ error: 'Username required' });

  const user = db.findByUsername(username);
  if (!user) return res.status(404).json({ error: 'No account found with that username' });
  if (!user.phone_encrypted) return res.status(400).json({ error: 'Account has no phone on file — contact support' });

  try {
    const e164   = decryptPhone(user.phone_encrypted);
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
app.post('/api/login', async (req, res) => {
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
  const { is_admin, ...user } = raw;
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
const feedbackLimit = rateLimit({ windowMs: 60 * 60 * 1000, max: 10, message: { error: 'Too many submissions. Try again later.' } });
app.post('/api/feedback', feedbackLimit, async (req, res) => {
  const { type, message, contact, page } = req.body;
  if (!message || message.trim().length < 5) return res.status(400).json({ error: 'Message too short' });
  if (!['bug', 'suggestion'].includes(type)) return res.status(400).json({ error: 'Invalid type' });
  const result = db.addFeedback(type, message.trim().slice(0, 2000), contact?.slice(0, 100), page?.slice(0, 100));

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
    standings: (() => { try { return JSON.parse(s.standings); } catch { return []; } })(),
  }));
  res.json({ sessions });
});

// Replay data for a specific session (public)
app.get('/api/sessions/:id/replay', (req, res) => {
  const row = db.getSessionReplay(Number(req.params.id));
  if (!row) return res.status(404).json({ error: 'Session not found' });
  let snapshots = [];
  try { snapshots = JSON.parse(row.replay_data || '[]'); } catch {}
  res.json({ sessionId: row.id, snapshots });
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
    if (err) return res.json({ error: 'Training not running or stats unavailable.' });
    try {
      const data = JSON.parse(require('fs').readFileSync('/tmp/training_stats_cache.json', 'utf8'));
      _trainingStatsCache = data;
      _trainingStatsCacheTime = now;
      res.json(data);
    } catch { res.json({ error: 'Could not parse stats.' }); }
  });
});

// Serve index.html for all known routes (client-side routing)
const clientRoutes = ['/solo', '/multiplayer', '/multiplayer/:code([0-9]{4})', '/rules', '/feedback', '/finance-road', '/finance-road/:tier/:level', '/quick-master', '/replay/:id'];
app.get('/plan', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'plan.html')));
app.get('/training', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'training.html')));
for (const route of clientRoutes) {
  app.get(route, (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
}
// Legacy: bare 4-digit code still works
app.get('/:code([0-9]{4})', (req, res) => {
  res.redirect(`/multiplayer/${req.params.code}`);
});

/* ── Room store ────────────────────────────────────────────── */
const rooms = {};

function generateCode() {
  let code;
  do { code = String(Math.floor(1000 + Math.random() * 9000)); } while (rooms[code]);
  return code;
}

/* ── State broadcasting ─────────────────────────────────────── */
function broadcastState(code) {
  const room = rooms[code];
  if (!room || !room.game) return;

  const playerList = room.players.map(p => ({ name: p.name, isBot: p.isBot || false }));

  // Capture spectator snapshot for replay before recording game end
  const spectatorState = engine.getClientState(room.game, -1);
  spectatorState.playerList = playerList;
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
      if (room.sessionId) clientState.sessionId = room.sessionId;
      io.to(p.socketId).emit('gameState', clientState);
    }
  }

  for (const s of room.spectators) {
    io.to(s.socketId).emit('gameState', spectatorState);
  }

  // Schedule next bot move if this is a solo room
  if (room.isSolo && room.game.phase !== 'gameOver') {
    scheduleBotTurn(code);
  }
}

function broadcastLobby(code) {
  const room = rooms[code];
  if (!room) return;
  const lobbyData = {
    code,
    players: room.players.map(p => ({ name: p.name, ready: p.ready, isBot: p.isBot || false })),
    spectators: room.spectators.map(s => ({ name: s.name })),
    maxPlayers: room.config.maxPlayers,
    gameInProgress: !!room.game,
  };
  io.to(code).emit('lobbyUpdate', lobbyData);
}

/* ── ELO + game recording ───────────────────────────────────── */
const ELO_K = 32;

function expectedScore(myElo, oppElo) {
  return 1 / (1 + Math.pow(10, (oppElo - myElo) / 400));
}

function recordGameEnd(code) {
  const room = rooms[code];
  if (!room || !room.game || room.gameRecorded) return;
  room.gameRecorded = true;

  const standings = room.game.standings; // [{ name, cash }] sorted desc
  console.log(`[recordGameEnd] code=${code} standings=${JSON.stringify(standings)}`);
  if (!standings || standings.length === 0) return;

  // Record session for all games regardless of auth status
  const durationSeconds = room.startedAt ? Math.round((Date.now() - room.startedAt) / 1000) : null;
  const humanCount = room.players.filter(p => !p.isBot).length;
  const turnCount  = room.game.turnNumber ?? null;
  const sessionStandings = standings.map(s => {
    const rp = room.players.find(p => p.name === s.name);
    return { name: s.name, cash: s.cash, isBot: rp?.isBot || false, userId: rp?.userId || null };
  });
  try {
    const sessionId = db.recordSession(room.isSolo, room.players.length, humanCount, durationSeconds, turnCount, sessionStandings, room.replaySnapshots || []);
    console.log(`[session] recorded: solo=${room.isSolo} players=${room.players.length} turns=${turnCount} id=${sessionId} snapshots=${(room.replaySnapshots||[]).length}`);
    room.sessionId = sessionId; // store so broadcastState can include it in gameOver state
  } catch (err) {
    console.error('[session] recordSession failed:', err.message);
  }

  const humanPlayers = room.players.filter(p => !p.isBot && p.userId);
  if (humanPlayers.length === 0) return;

  // Snapshot current ELO for all participants before any updates
  const eloMap = {};
  for (const p of humanPlayers) {
    const user = db.findById(p.userId);
    if (user) eloMap[p.userId] = user.elo;
  }

  for (const p of humanPlayers) {
    const myRank  = standings.findIndex(s => s.name === p.name);
    const myElo   = eloMap[p.userId] || 1000;
    let delta = 0;

    const otherHumans = humanPlayers.filter(o => o.userId !== p.userId);
    if (otherHumans.length > 0) {
      // Pairwise ELO vs other logged-in humans
      for (const opp of otherHumans) {
        const oppRank = standings.findIndex(s => s.name === opp.name);
        const oppElo  = eloMap[opp.userId] || 1000;
        const actual  = myRank < oppRank ? 1 : 0; // lower index = better place
        delta += ELO_K * (actual - expectedScore(myElo, oppElo));
      }
      delta /= otherHumans.length; // normalize
    } else {
      // Solo game — compare vs fixed bot ELO 1000
      const actual = myRank === 0 ? 1 : 0;
      delta = ELO_K * (actual - expectedScore(myElo, 1000));
    }

    const eloChange = Math.round(delta);
    const result    = myRank === 0 ? 'win' : 'loss';
    const opponents = standings
      .filter(s => s.name !== p.name)
      .map(s => ({ name: s.name, cash: s.cash }));

    db.recordGameResult(p.userId, result, eloChange, opponents);
  }
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

/* ── Bot scheduling ─────────────────────────────────────────── */
function scheduleBotTurn(code) {
  const room = rooms[code];
  if (!room || !room.game || !room.isSolo) return;
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
  const delay = 1800 + Math.random() * 2200;

  setTimeout(() => {
    room.botScheduled = false;
    if (!rooms[code] || !rooms[code].game) return;

    const botPlayer = room.players.find(p => p.idx === actingBotIdx);
    if (!botPlayer) return;

    try {
      decideBotAction(rooms[code].game, actingBotIdx, botPlayer.personality, botPlayer.difficulty, botPlayer.name);
    } catch (e) {
      console.error(`Bot error in room ${code}:`, e.message);
    }
    broadcastState(code);
  }, delay);
}

/* ── Socket events ─────────────────────────────────────────── */
io.on('connection', (socket) => {
  let currentRoom = null;
  let playerName = null;

  /* ── Multiplayer: create room ── */
  socket.on('createRoom', ({ name, maxPlayers, quickstart, authToken }, cb) => {
    const code = generateCode();
    currentRoom = code;
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
  socket.on('createSoloGame', ({ name, botCount, quickstart, botConfigs, authToken }, cb) => {
    const code = generateCode();
    currentRoom = code;
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
          difficulty: cfg.difficulty || 'medium',
        };
      }),
    ];

    rooms[code] = {
      players,
      spectators: [],
      game: null,
      isSolo: true,
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
      playerCount: room.players.length,
      maxPlayers: room.config.maxPlayers,
    });
  });

  socket.on('joinRoom', ({ name, code, spectate, authToken }, cb) => {
    const room = rooms[code];
    if (!room) return cb({ error: 'Room not found' });

    currentRoom = code;
    playerName = name;
    socket.join(code);
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

    const p = room.players.find(p => p.token === token && p.name === name);
    if (!p) return cb({ error: 'Session not found' });

    p.socketId = socket.id;
    p.disconnected = false;
    currentRoom = code;
    playerName = name;
    socket.join(code);

    cb({ ok: true, inGame: !!room.game, isSolo: room.isSolo });
    if (room.game) {
      room.game.log.push(`${name} reconnected`);
      broadcastState(code);
    } else {
      broadcastLobby(code);
    }
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
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.placeTile(room.game, p.idx, tile);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('chooseChain', ({ chain }) => {
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.chooseChain(room.game, p.idx, chain);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('chooseMergerSurvivor', ({ chain }) => {
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.chooseMergerSurvivor(room.game, p.idx, chain);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('mergerDecision', ({ sell, trade }) => {
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.mergerDecision(room.game, p.idx, { sell, trade });
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('buyStock', ({ purchases }) => {
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.buyStock(room.game, p.idx, purchases || {});
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  socket.on('declareGameEnd', () => {
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.declareGameEnd(room.game, p.idx);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  /* ── Concede vote ─────────────────────────────── */

  socket.on('initiateConcedeVote', () => {
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.initiateConcedeVote(room.game, p.idx);
    if (result.error) return socket.emit('actionError', result.error);
    // In solo games, bots automatically vote yes
    if (room.isSolo && room.game.concedeActive) {
      for (const bp of room.players) {
        if (bp.isBot && room.game.concedeVotes[bp.idx] === undefined) {
          engine.submitConcedeVote(room.game, bp.idx, true);
          if (!room.game.concedeActive) break; // vote passed, stop
        }
      }
    }
    broadcastState(currentRoom);
  });

  socket.on('submitConcedeVote', ({ vote }) => {
    const room = rooms[currentRoom];
    if (!room || !room.game) return;
    const p = room.players.find(p => p.socketId === socket.id);
    if (!p) return;
    const result = engine.submitConcedeVote(room.game, p.idx, vote);
    if (result.error) return socket.emit('actionError', result.error);
    broadcastState(currentRoom);
  });

  /* ── Chat ───────────────────────────────────────── */

  socket.on('chatMessage', ({ message }) => {
    if (!currentRoom || !message || !message.trim()) return;
    const room = rooms[currentRoom];
    if (!room) return;
    const name = playerName || 'Anonymous';
    const msg = message.trim().slice(0, 300);
    io.to(currentRoom).emit('chatMessage', { name, message: msg, time: Date.now() });
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
    room.players.forEach(p => { p.ready = false; });

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

  /* ── Leave room (e.g. solo player goes back to setup) ── */
  socket.on('leaveRoom', () => {
    if (currentRoom && rooms[currentRoom]) {
      delete rooms[currentRoom];
    }
    currentRoom = null;
  });

  /* ── Disconnect ───────────────────────────────── */

  socket.on('disconnect', () => {
    if (!currentRoom || !rooms[currentRoom]) return;
    const room = rooms[currentRoom];
    room.spectators = room.spectators.filter(s => s.socketId !== socket.id);

    if (room.isSolo) {
      // Clean up solo room when human leaves
      delete rooms[currentRoom];
      return;
    }

    if (room.game) {
      const p = room.players.find(p => p.socketId === socket.id);
      if (p) {
        p.socketId = null;
        p.disconnected = true;
        room.game.log.push(`${p.name} disconnected`);
        broadcastState(currentRoom);
      }
    } else {
      room.players = room.players.filter(p => p.socketId !== socket.id);
      room.players.forEach((p, i) => { p.idx = i; });
      if (room.players.length === 0 && room.spectators.length === 0) {
        delete rooms[currentRoom];
      } else {
        broadcastLobby(currentRoom);
      }
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Acquire server running on http://localhost:${PORT}`);
});
