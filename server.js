/**
 * Acquire – Server (v3.0)
 * Express + Socket.io
 * Supports multiplayer (lobby) and solo (vs bots) modes.
 */

const express = require('express');
const http = require('http');
const path = require('path');
const { randomUUID } = require('crypto');
const { Server } = require('socket.io');
const engine = require('./gameEngine');
const { decideBotAction, BOT_ROSTER } = require('./botAI');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: ['https://playonlineacquire.com', 'http://localhost:3000'] } });

app.use(express.static('public'));

// Serve index.html for all known routes (client-side routing)
const clientRoutes = ['/solo', '/multiplayer', '/multiplayer/:code([0-9]{4})', '/rules'];
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

  for (const p of room.players) {
    if (p.socketId && !p.isBot) {
      const clientState = engine.getClientState(room.game, p.idx);
      clientState.playerList = playerList;
      io.to(p.socketId).emit('gameState', clientState);
    }
  }
  for (const s of room.spectators) {
    const clientState = engine.getClientState(room.game, -1);
    clientState.playerList = playerList;
    io.to(s.socketId).emit('gameState', clientState);
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

/* ── Game start ─────────────────────────────────────────────── */
function startGame(code) {
  const room = rooms[code];
  const names = room.players.map(p => p.name);
  room.game = engine.createGame(names, { quickstart: room.config.quickstart });
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
  const delay = 700 + Math.random() * 700;

  setTimeout(() => {
    room.botScheduled = false;
    if (!rooms[code] || !rooms[code].game) return;

    const botPlayer = room.players.find(p => p.idx === actingBotIdx);
    if (!botPlayer) return;

    try {
      decideBotAction(rooms[code].game, actingBotIdx, botPlayer.personality);
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
  socket.on('createRoom', ({ name, maxPlayers, quickstart }, cb) => {
    const code = generateCode();
    currentRoom = code;
    playerName = name;
    const token = randomUUID();

    rooms[code] = {
      players: [{ name, socketId: socket.id, ready: false, idx: 0, token }],
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
  socket.on('createSoloGame', ({ name, botCount, quickstart }, cb) => {
    const code = generateCode();
    currentRoom = code;
    playerName = name;
    const token = randomUUID();

    const bots = BOT_ROSTER.slice(0, Math.min(Math.max(botCount || 1, 1), 5));

    const players = [
      { name, socketId: socket.id, ready: true, idx: 0, token, isBot: false },
      ...bots.map((b, i) => ({
        name: b.name, socketId: null, ready: true,
        idx: i + 1, token: null, isBot: true, personality: b.personality,
      })),
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

  socket.on('joinRoom', ({ name, code, spectate }, cb) => {
    const room = rooms[code];
    if (!room) return cb({ error: 'Room not found' });

    currentRoom = code;
    playerName = name;
    socket.join(code);

    if (spectate || room.game || room.players.length >= room.config.maxPlayers) {
      room.spectators.push({ name, socketId: socket.id });
      cb({ ok: true, spectator: true });
      broadcastLobby(code);
      if (room.game) broadcastState(code);
      return;
    }

    const token = randomUUID();
    room.players.push({ name, socketId: socket.id, ready: false, idx: room.players.length, token });
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
