/**
 * Acquire – Server (v2.1)
 * Express + Socket.io
 */

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const engine = require('./gameEngine');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

app.use(express.static('public'));

/* ── Room store ────────────────────────────────────────────── */
const rooms = {};

function generateCode() {
  let code;
  do { code = String(Math.floor(1000 + Math.random() * 9000)); } while (rooms[code]);
  return code;
}

function broadcastState(code) {
  const room = rooms[code];
  if (!room || !room.game) return;
  for (const p of room.players) {
    if (p.socketId) {
      const clientState = engine.getClientState(room.game, p.idx);
      io.to(p.socketId).emit('gameState', clientState);
    }
  }
  for (const s of room.spectators) {
    const clientState = engine.getClientState(room.game, -1);
    io.to(s.socketId).emit('gameState', clientState);
  }
}

function broadcastLobby(code) {
  const room = rooms[code];
  if (!room) return;
  const lobbyData = {
    code,
    players: room.players.map(p => ({ name: p.name, ready: p.ready })),
    spectators: room.spectators.map(s => ({ name: s.name })),
    maxPlayers: room.config.maxPlayers,
    gameInProgress: !!room.game,
  };
  io.to(code).emit('lobbyUpdate', lobbyData);
}

/* ── Socket events ─────────────────────────────────────────── */
io.on('connection', (socket) => {
  let currentRoom = null;
  let playerName = null;

  socket.on('createRoom', ({ name, maxPlayers, quickstart }, cb) => {
    const code = generateCode();
    currentRoom = code;
    playerName = name;

    rooms[code] = {
      players: [{ name, socketId: socket.id, ready: false, idx: 0 }],
      spectators: [],
      game: null,
      config: { maxPlayers: Math.min(Math.max(maxPlayers || 4, 2), 6), quickstart: quickstart !== false },
    };

    socket.join(code);
    cb({ ok: true, code });
    broadcastLobby(code);
  });

  socket.on('joinRoom', ({ name, code }, cb) => {
    const room = rooms[code];
    if (!room) return cb({ error: 'Room not found' });

    currentRoom = code;
    playerName = name;
    socket.join(code);

    if (room.game) {
      room.spectators.push({ name, socketId: socket.id });
      cb({ ok: true, spectator: true });
      broadcastLobby(code);
      broadcastState(code);
      return;
    }

    if (room.players.length >= room.config.maxPlayers) {
      room.spectators.push({ name, socketId: socket.id });
      cb({ ok: true, spectator: true });
    } else {
      room.players.push({ name, socketId: socket.id, ready: false, idx: room.players.length });
      cb({ ok: true, spectator: false });
    }
    broadcastLobby(code);
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
    room.game = null;
    room.players.forEach(p => { p.ready = false; });

    while (room.spectators.length > 0 && room.players.length < room.config.maxPlayers) {
      const s = room.spectators.shift();
      room.players.push({ name: s.name, socketId: s.socketId, ready: false, idx: room.players.length });
    }

    io.to(currentRoom).emit('backToLobby');
    broadcastLobby(currentRoom);
  });

  /* ── Disconnect ───────────────────────────────── */

  socket.on('disconnect', () => {
    if (!currentRoom || !rooms[currentRoom]) return;
    const room = rooms[currentRoom];
    room.spectators = room.spectators.filter(s => s.socketId !== socket.id);

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
