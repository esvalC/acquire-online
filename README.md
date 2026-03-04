# Acquire – Online Multiplayer

A real-time multiplayer implementation of the classic board game **Acquire** (1964, Sid Sackson).

## Quick Start

```bash
npm install
npm start
```

Open `http://localhost:3000`. Share your local IP (e.g. `http://192.168.1.x:3000`) with others on the same WiFi.

## How to Play

1. **Create a Table** – pick your name and max players (2–6), get a 4-digit room code
2. **Join a Table** – enter the code and your name
3. **Ready Up** – game starts when all players are ready
4. **Quickstart Draw** – each player automatically draws a tile; closest to 1A goes first; tiles stay on the board
5. **Play!** – place tiles, found chains, buy stock, trigger mergers, get rich

### In-Game Features
- **Rules on landing page** – full rules summary with chain legend visible before joining a game
- **Board** – color-coded chains, highlighted playable tiles, your hand tiles shown on the grid
- **Stock Market** – live prices, chain sizes, safe indicators
- **Player Chat** – sidebar tab for in-game messaging with unread badge
- **Rules Assistant** – sidebar tab with quick-topic buttons and free-text Q&A
- **Turn Clock** – per-player cumulative timer (not enforced, just for tracking)
- **Vote to End** – any player can call a vote to concede (requires N-1 of N votes)
- **Declare End** – when official end conditions are met, current player sees end game button
- **Play Again** – after game over, sends everyone back to the lobby

## Architecture

```
acquire/
├── gameEngine.js    # Pure game logic (no I/O)
├── server.js        # Express + Socket.io server
├── public/
│   └── index.html   # Single-file frontend
├── package.json
└── README.md
```

- Server owns all game state; clients send actions, server validates and broadcasts
- Socket.io rooms isolate each table
- Tiles are private per-player; other players only see tile count
- Quickstart draw determines turn order and places starting tiles on the board

## Roadmap / Future Plans

### Data & AI
- Store game turn data for every game to eventually train an AI player
- Schema: per-turn snapshots with board state, player holdings, and action taken

### Visual Polish
- Smoother animations – tile placements animate into position, chain expansions ripple, mergers collapse
- Card visuals – show tile backs for other players' hands (count visible, tiles hidden)
- Goal: best-looking Acquire game online

### Resilience & Accounts
- Reconnection – rejoin with same name/room code after disconnect
- User accounts – persistent identity, game history, stats, ELO
- Spectator improvements

### Deployment to AWS
- Phase 1: EC2 + PM2 + Nginx + SSL
- Phase 2: S3/CloudFront for frontend, EC2 for Socket.io
- Phase 3: DynamoDB for persistence
- Phase 4: Serverless with Lambda + API Gateway WebSockets
