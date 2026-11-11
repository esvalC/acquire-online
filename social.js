/**
 * Social features: who's online, friends, game invites, live games to watch,
 * friend chat, reports.
 *
 * A signed-in browser's live connection (socket) is tied to its account when it
 * connects (handshake token) or when the player signs in later ('auth' event).
 * The server keeps, per account, the set of open connections; that powers online
 * status and lets a message reach every tab and phone the player has open.
 *
 * Privacy: online status is shown to friends only; requests from blocked players
 * and to players who switched requests off are dropped silently; chat is for
 * signed-in players only, never guests.
 */
const rateLimit = require('express-rate-limit');
const { censor } = require('./filter');

module.exports = function setupSocial({ app, io, db, getRooms, verifyToken, requireAuth, requireAdmin, byVisitor, isAdminUser }) {
  const S = db.social;

  /* ── Online presence ─────────────────────────────────────────── */
  const userSockets = new Map();          // userId → Set<socket>
  function bind(socket, userId) {
    unbind(socket);
    if (!userId) return;
    socket.data.userId = userId;
    if (!userSockets.has(userId)) userSockets.set(userId, new Set());
    userSockets.get(userId).add(socket);
    socket.join(`user:${userId}`);
  }
  function unbind(socket) {
    const uid = socket.data.userId;
    if (!uid) return;
    socket.leave(`user:${uid}`);
    const set = userSockets.get(uid);
    if (set) { set.delete(socket); if (!set.size) { userSockets.delete(uid); try { S.touchLastSeen(uid); } catch {} } }
    socket.data.userId = null;
  }
  const isOnline = uid => userSockets.has(uid);
  const emitToUser = (uid, ev, data) => io.to(`user:${uid}`).emit(ev, data);

  // Where is this player? game > table > online > offline
  function statusOf(uid) {
    if (!isOnline(uid)) return { state: 'offline' };
    const rooms = getRooms();
    for (const [code, room] of Object.entries(rooms)) {
      const p = room.players.find(x => x.userId === uid && x.socketId && !x.isBot);
      if (!p) continue;
      if (room.game && room.game.phase !== 'gameOver') {
        return { state: 'game', code, solo: !!room.isSolo, turn: room.game.turnNumber || 1 };
      }
      if (!room.game && !room.isSolo) {
        return { state: 'table', code, seats: Math.max(0, room.config.maxPlayers - room.players.length) };
      }
    }
    return { state: 'online' };
  }
  const statusKey = st => `${st.state}|${st.code || ''}|${st.seats ?? ''}`;

  // Friends are told when someone's status changes. A light sweep every 3s
  // catches every way status can change (join, leave, game start/end, close tab).
  const lastSent = new Map();             // userId → status key
  function sweepPresence() {
    const ids = new Set([...userSockets.keys(), ...lastSent.keys()]);
    for (const uid of ids) {
      const st = statusOf(uid), key = statusKey(st);
      if (lastSent.get(uid) === key) continue;
      if (st.state === 'offline') lastSent.delete(uid); else lastSent.set(uid, key);
      let friends = [];
      try { friends = S.friendIds(uid); } catch {}
      for (const f of friends) if (isOnline(f)) emitToUser(f, 'presence', { id: uid, ...st });
    }
  }
  setInterval(sweepPresence, 3000).unref();

  /* ── Relationship helpers ────────────────────────────────────── */
  function relationFor(me, other) {
    if (!me || !other) return 'none';
    if (me === other) return 'self';
    const r = S.relation(me, other);
    if (!r) return 'none';
    if (r.status === 'accepted') return 'friends';
    if (r.status === 'blocked') return r.requested_by === me ? 'blocked' : 'none';  // being blocked looks like nothing
    return r.requested_by === me ? 'outgoing' : 'incoming';
  }
  const blockedEitherWay = (a, b) => { const r = S.relation(a, b); return !!(r && r.status === 'blocked'); };
  const changed = (...ids) => ids.forEach(id => emitToUser(id, 'friendsChanged', {}));

  /* ── Sockets ─────────────────────────────────────────────────── */
  const dmTimes = new Map();              // userId → [timestamps]
  const inviteTimes = new Map();
  const invites = new Map();              // inviteId → { from, to, code, expires }
  const allow = (map, uid, n, ms) => {
    const now = Date.now(), list = (map.get(uid) || []).filter(t => now - t < ms);
    if (list.length >= n) { map.set(uid, list); return false; }
    list.push(now); map.set(uid, list); return true;
  };

  io.on('connection', (socket) => {
    const hs = socket.handshake && socket.handshake.auth && socket.handshake.auth.token;
    if (hs) { const p = verifyToken(hs); if (p) bind(socket, p.sub); }
    const on = (ev, fn) => socket.on(ev, (...args) => {
      const cb = typeof args.at(-1) === 'function' ? args.at(-1) : () => {};
      const data = args[0] && typeof args[0] === 'object' ? args[0] : {};
      try { fn(data, cb); } catch (e) { console.error(`[social ${ev}]`, e.message); try { cb({ error: 'Something went wrong' }); } catch {} }
    });

    // Sign in / out after the page loaded (token = null signs the connection out)
    on('auth', ({ token }, cb) => {
      const p = token ? verifyToken(token) : null;
      if (p) bind(socket, p.sub); else unbind(socket);
      cb({ ok: !!p });
    });

    // Invite online friends to the table you're sitting at
    on('sendInvites', ({ to }, cb) => {
      const me = socket.data.userId;
      if (!me) return cb({ error: 'Sign in to invite friends' });
      if (!Array.isArray(to) || !to.length) return cb({ error: 'Pick at least one friend' });
      if (!allow(inviteTimes, me, 20, 60000)) return cb({ error: 'Too many invites — wait a minute' });
      const rooms = getRooms();
      const entry = Object.entries(rooms).find(([, r]) => !r.isSolo && !r.game && r.players.some(p => p.socketId === socket.id));
      if (!entry) return cb({ error: 'Invites work from a table that hasn\'t started yet' });
      const [code] = entry;
      const fromUser = db.findById(me);
      const friends = new Set(S.friendIds(me));
      let sent = 0;
      for (const raw of to.slice(0, 10)) {
        const uid = Number(raw);
        if (!friends.has(uid) || !isOnline(uid)) continue;
        const id = Math.random().toString(36).slice(2, 10);
        const expires = Date.now() + 2 * 60 * 1000;
        invites.set(id, { from: me, to: uid, code, expires });
        setTimeout(() => invites.delete(id), 2 * 60 * 1000 + 5000).unref();
        emitToUser(uid, 'invite', { id, code, from: { id: me, username: fromUser.username }, expires });
        sent++;
      }
      cb(sent ? { ok: true, sent } : { error: 'None of those friends are online right now' });
    });
    on('inviteResponse', ({ id, accept }, cb) => {
      const inv = invites.get(String(id));
      if (!inv || inv.to !== socket.data.userId) return cb({ error: 'That invite has expired' });
      invites.delete(String(id));
      const who = db.findById(inv.to);
      emitToUser(inv.from, 'inviteAnswered', { username: who ? who.username : 'A friend', accept: !!accept });
      const room = getRooms()[inv.code];
      if (accept && !room) return cb({ error: 'That table has closed' });
      cb({ ok: true, code: inv.code, started: !!(room && room.game) });
    });

    // Friend chat (signed-in players only)
    on('dm', ({ to, body }, cb) => {
      const me = socket.data.userId;
      if (!me) return cb({ error: 'Sign in to chat' });
      const other = Number(to);
      const target = Number.isInteger(other) ? db.findById(other) : null;
      if (!target || other === me) return cb({ error: 'Player not found' });
      let text = typeof body === 'string' ? body.replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, '').trim() : '';
      if (!text) return cb({ error: 'Type a message' });
      if (text.length > 500) return cb({ error: 'Messages can be up to 500 characters' });
      if (blockedEitherWay(me, other)) return cb({ error: 'You can\'t message this player' });
      const st = S.settings(other);
      if (st && st.friends_only_dm && relationFor(me, other) !== 'friends') {
        return cb({ error: `${target.username} only accepts messages from friends` });
      }
      if (!allow(dmTimes, me, 10, 10000)) return cb({ error: 'Slow down a little' });
      const msg = S.addMessage(me, other, censor(text));
      const out = { id: msg.id, from: me, to: other, body: msg.body, created_at: msg.created_at };
      emitToUser(other, 'dm', { ...out, fromName: db.findById(me).username });
      emitToUser(me, 'dm', { ...out, fromName: db.findById(me).username, mine: true });
      cb({ ok: true, message: out });
    });
    on('dmRead', ({ from }) => {
      const me = socket.data.userId, other = Number(from);
      if (me && Number.isInteger(other)) S.markRead(me, other);
    });

    socket.on('disconnect', () => unbind(socket));
  });

  /* ── Requests (web) ──────────────────────────────────────────── */
  const friendReqLimit = rateLimit({ windowMs: 60 * 60 * 1000, max: 20, keyGenerator: byVisitor,
    message: { error: 'Too many friend requests — try again later' }, standardHeaders: true, legacyHeaders: false });
  const reportLimit = rateLimit({ windowMs: 60 * 60 * 1000, max: 10, keyGenerator: byVisitor,
    message: { error: 'Too many reports — try again later' }, standardHeaders: true, legacyHeaders: false });
  const idParam = req => { const n = Number(req.params.id); return Number.isInteger(n) && n > 0 ? n : null; };

  app.get('/api/friends', requireAuth, (req, res) => {
    const me = req.userId;
    const rel = S.relationsOf(me);
    const unread = Object.fromEntries(S.unreadCounts(me).map(r => [r.id, r.n]));
    const friends = rel.filter(r => r.status === 'accepted').map(r => ({
      id: r.id, username: r.username, isAdmin: !!r.is_admin, lastSeen: r.last_seen, unread: unread[r.id] || 0, ...statusOf(r.id),
    })).sort((a, b) => (a.state === 'offline') - (b.state === 'offline') || a.username.localeCompare(b.username));
    const incoming = rel.filter(r => r.status === 'pending' && r.requested_by !== me).map(r => ({ id: r.id, username: r.username }));
    const outgoing = rel.filter(r => r.status === 'pending' && r.requested_by === me).map(r => ({ id: r.id, username: r.username }));
    const blocked  = rel.filter(r => r.status === 'blocked' && r.requested_by === me).map(r => ({ id: r.id, username: r.username }));
    const friendSet = new Set(friends.map(f => f.id));
    const chats = S.conversations(me).filter(c => !friendSet.has(c.id) && !blocked.some(b => b.id === c.id))
      .map(c => ({ id: c.id, username: c.username, unread: unread[c.id] || 0 }));
    const st = S.settings(me) || {};
    res.json({ friends, incoming, outgoing, blocked, chats,
      settings: { allowRequests: !!st.allow_requests, friendsOnlyDm: !!st.friends_only_dm } });
  });

  // Same answer whether or not the name exists, so it can't be used to list accounts
  app.post('/api/friends/request', friendReqLimit, requireAuth, (req, res) => {
    const me = req.userId;
    const name = typeof req.body.username === 'string' ? req.body.username.trim() : '';
    const byId = Number(req.body.userId);
    const target = Number.isInteger(byId) && byId > 0 ? db.findById(byId) : (name ? db.findByUsername(name) : null);
    if (!target && !name && !byId) return res.status(400).json({ error: 'Enter a username' });
    const sent = () => res.json({ ok: true, message: 'Friend request sent' });
    if (!target) return sent();
    if (target.id === me) return res.status(400).json({ error: 'That\'s you!' });
    const r = S.relation(me, target.id);
    if (r && r.status === 'accepted') return res.json({ ok: true, message: `You and ${target.username} are already friends` });
    if (r && r.status === 'blocked') return sent();
    if (r && r.status === 'pending' && r.requested_by === target.id) {        // they asked first → accept
      S.setRelation(me, target.id, 'accepted', target.id);
      changed(me, target.id);
      return res.json({ ok: true, message: `You and ${target.username} are now friends` });
    }
    if (r && r.status === 'pending') return sent();
    const st = S.settings(target.id);
    if (st && !st.allow_requests) return sent();
    S.setRelation(me, target.id, 'pending', me);
    changed(me, target.id);
    sent();
  });

  app.post('/api/friends/:id/:action(accept|decline|cancel)', requireAuth, (req, res) => {
    const me = req.userId, other = idParam(req);
    const r = other && S.relation(me, other);
    if (!r || r.status !== 'pending') return res.status(404).json({ error: 'No request to answer' });
    const action = req.params.action;
    if (action === 'cancel' ? r.requested_by !== me : r.requested_by !== other) return res.status(400).json({ error: 'No request to answer' });
    if (action === 'accept') S.setRelation(me, other, 'accepted', other); else S.clearRelation(me, other);
    changed(me, other);
    res.json({ ok: true });
  });

  app.delete('/api/friends/:id', requireAuth, (req, res) => {
    const me = req.userId, other = idParam(req);
    const r = other && S.relation(me, other);
    if (r && r.status === 'accepted') { S.clearRelation(me, other); changed(me, other); }
    res.json({ ok: true });
  });

  app.post('/api/users/:id/block', requireAuth, (req, res) => {
    const me = req.userId, other = idParam(req);
    if (!other || other === me || !db.findById(other)) return res.status(404).json({ error: 'Player not found' });
    const r = S.relation(me, other);
    if (!(r && r.status === 'blocked' && r.requested_by === other)) S.setRelation(me, other, 'blocked', me);
    changed(me, other);
    res.json({ ok: true });
  });
  app.delete('/api/users/:id/block', requireAuth, (req, res) => {
    const me = req.userId, other = idParam(req);
    const r = other && S.relation(me, other);
    if (r && r.status === 'blocked' && r.requested_by === me) { S.clearRelation(me, other); changed(me, other); }
    res.json({ ok: true });
  });

  app.post('/api/me/social-settings', requireAuth, (req, res) => {
    const { allowRequests, friendsOnlyDm } = req.body || {};
    S.setSettings(req.userId, { allowRequests, friendsOnlyDm });
    res.json({ ok: true });
  });

  app.get('/api/messages/:id', requireAuth, (req, res) => {
    const me = req.userId, other = idParam(req);
    const user = other && db.findById(other);
    if (!user) return res.status(404).json({ error: 'Player not found' });
    const before = Number(req.query.before) || null;
    const messages = S.conversation(me, other, before, 50);
    S.markRead(me, other);
    const st = S.settings(other) || {};
    const rel = relationFor(me, other);
    res.json({
      user: { id: user.id, username: user.username, isAdmin: !!user.is_admin },
      relation: rel, messages,
      canMessage: !blockedEitherWay(me, other) && !(st.friends_only_dm && rel !== 'friends'),
    });
  });

  app.post('/api/messages/:id/report', reportLimit, requireAuth, (req, res) => {
    const me = req.userId, other = idParam(req);
    if (!other || !db.findById(other)) return res.status(404).json({ error: 'Player not found' });
    const transcript = S.conversation(me, other, null, 50).map(m => ({ from: m.from_id, body: m.body, at: m.created_at }));
    const reason = typeof req.body.reason === 'string' ? req.body.reason.trim().slice(0, 300) : '';
    S.addReport(me, other, reason, transcript);
    res.json({ ok: true });
  });

  /* ── Live games you can watch: friends' games; every game for admins ── */
  function liveGamesFor(viewerId) {
    const admin = !!(viewerId && (db.findById(viewerId) || {}).is_admin);   // fresh, so a promotion counts at once
    const friends = viewerId ? new Set(S.friendIds(viewerId)) : new Set();
    const out = [];
    for (const [code, room] of Object.entries(getRooms())) {
      if (!room.game || room.game.phase === 'gameOver' || room.tutorial) continue;
      const humans = room.players.filter(p => !p.isBot);
      if (!humans.some(p => p.socketId)) continue;                  // abandoned
      const friendIn = humans.some(p => p.userId && friends.has(p.userId));
      if (!admin && !friendIn) continue;
      out.push({
        code, solo: !!room.isSolo, mode: room.mode || (room.isSolo ? 'Solo' : 'Multiplayer'),
        turn: room.game.turnNumber || 1, watching: room.spectators.length,
        players: room.players.map(p => ({ name: p.name, isBot: !!p.isBot, isFriend: !!(p.userId && friends.has(p.userId)), isAdmin: !!(p.userId && isAdminUser(p.userId)) })),
      });
    }
    return out.sort((a, b) => b.turn - a.turn);
  }
  app.get('/api/live-games', (req, res) => {
    const header = req.headers.authorization || '';
    const p = header.startsWith('Bearer ') ? verifyToken(header.slice(7)) : null;
    res.json({ games: p ? liveGamesFor(p.sub) : [], admin: !!(p && (db.findById(p.sub) || {}).is_admin) });
  });

  /* ── Admin: reports ──────────────────────────────────────────── */
  app.get('/api/plan/reports', requireAdmin, (_req, res) => res.json({ reports: S.listReports(100) }));
  app.post('/api/plan/reports/:id/status', requireAdmin, (req, res) => {
    const id = idParam(req), status = req.body && req.body.status;
    if (!id || !['new', 'reviewed'].includes(status)) return res.status(400).json({ error: 'Bad request' });
    S.setReportStatus(id, status);
    res.json({ ok: true });
  });

  // Messages are kept 90 days
  const prune = () => { try { const n = S.pruneMessages(); if (n) console.log(`[chat] removed ${n} message(s) older than ${S.MESSAGE_DAYS} days`); } catch (e) { console.error('[chat prune]', e.message); } };
  setTimeout(prune, 10000).unref();
  setInterval(prune, 6 * 3600 * 1000).unref();

  return { relationFor, statusOf, isOnline };
};
