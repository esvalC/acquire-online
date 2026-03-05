# Acquire – Online Multiplayer

A real-time multiplayer implementation of the classic board game **Acquire** (1964, Sid Sackson), with solo play against AI bots.

**Live:** https://playonlineacquire.com

---

## Quick Start (Local Dev)

```bash
npm install
npm start
```

Open `http://localhost:3000`. Share your local IP (e.g. `http://192.168.1.x:3000`) with others on the same WiFi.

## How to Play

Visit the landing page to choose your mode:

### Solo (vs. Bots)
1. **Choose Solo** on the landing page
2. **Enter your name** and select 1–5 bot opponents
3. **Start Game** — bots play automatically with a short delay (0.7–1.4s per move)
4. **Play!** — place tiles, found chains, buy stock, trigger mergers, get rich

**Bot Personalities:**
- **Focused** (Rex, Colt): Concentrates all stock purchases in one chain
- **Balanced** (Aria, Vera): Splits purchases across two chains
- **Diversified** (Nova): Buys one share each in multiple chains
- All bots never hold more than 13 of 25 shares in any single chain

### Multiplayer
1. **Choose Multiplayer** on the landing page
2. **Create a Table** — pick your name and max players (2–6), get a 4-digit room code
3. **Join a Table** — enter the code and your name
4. **Ready Up** — game starts when all players are ready
5. **Quickstart Draw** — each player automatically draws a tile; closest to 1A goes first; tiles stay on the board
6. **Play!** — place tiles, found chains, buy stock, trigger mergers, get rich

### In-Game Features
- **Board** — color-coded chains, highlighted playable tiles, your hand tiles shown on the grid
  - Green glow: tile would found a chain
  - Red glow: tile would trigger a merger
  - Gold glow: tile expands an existing chain
- **Stock Market** — live prices, chain sizes, shares remaining, safe indicators
- **Player Chat** — sidebar tab for in-game messaging with unread badge
- **Rules Assistant** — sidebar tab with quick-topic buttons and free-text Q&A
- **Turn Clock** — per-player cumulative timer (not enforced, just for tracking)
- **Vote to End** — any player can call a vote to concede (requires N-1 of N votes)
- **Declare End** — when official end conditions are met, current player sees End Game button
- **Play Again** — after game over, sends everyone back (solo restarts instantly)

---

## Application Architecture

```
acquire-online/
├── gameEngine.js          # Pure game logic (no I/O, no networking)
├── botAI.js               # Bot AI logic and personalities (solo mode)
├── server.js              # Express + Socket.IO server
├── public/
│   └── index.html         # Single-file frontend (HTML + CSS + JS)
├── package.json
└── .github/
    └── workflows/
        └── deploy.yml     # CI/CD pipeline (auto-deploy on push)
```

- **Server owns all game state.** Clients send actions, server validates and broadcasts the new state to everyone in the room.
- **Bot moves run server-side.** `botAI.js` exposes `decideBotAction(game, idx, personality)` — called by a `setTimeout` in `server.js` after each state broadcast for solo rooms. No client changes are needed to support bots.
- **Socket.IO rooms** isolate each table — players in room A never see events from room B.
- **Tiles are private per-player.** Other players only see your tile count, not which tiles you hold.
- **In-memory state.** All game data lives in the server process. If the server restarts, active games are lost.

### URL Structure

| URL | Description |
|---|---|
| `/` | Landing page — choose Solo, Multiplayer, or Rules |
| `/solo` | Solo game setup screen |
| `/multiplayer` | Create or join a multiplayer room |
| `/multiplayer/1234` | Join a specific room (auto-reconnects if session exists) |
| `/rules` | Full standalone rules guide |

---

## Infrastructure & Deployment

### Overview

```
You (push code)
      │
      ▼
 GitHub (private repo)
      │  triggers
      ▼
 GitHub Actions
      │  authenticates via OIDC (no passwords stored)
      ▼
 AWS IAM (temporary credentials, expire after deploy)
      │  sends command via SSM
      ▼
 EC2 Instance (Ubuntu 22.04, t3.micro)
      │
      ├── Nginx (port 80) ← public internet
      │       │ reverse proxy
      └── Node.js / PM2 (port 3000, internal only)
```

### AWS Resources

| Resource | Name / ID | Purpose |
|---|---|---|
| EC2 Instance | `i-0bbc6c13fd3dfe6ab` | Runs the Node.js server |
| Security Group | `acquire-online-sg` | Only ports 80 + 443 open. No SSH. |
| IAM Role (EC2) | `acquire-online-ec2-role` | Lets EC2 use AWS Systems Manager |
| IAM Role (CI/CD) | `acquire-online-github-actions-role` | What GitHub Actions assumes to deploy |
| OIDC Provider | `token.actions.githubusercontent.com` | Trust bridge between GitHub and AWS |

### Why No SSH Port?

Port 22 (SSH) is the most attacked port on the internet. Bots scan for it constantly. We manage the server entirely through **AWS Systems Manager (SSM)**, which connects over HTTPS internally. No SSH port is open, so there's nothing for attackers to find.

### How CI/CD Works (OIDC)

Traditional CI/CD stores an AWS secret key in GitHub. If GitHub gets compromised, your AWS account is compromised too.

With **OIDC (OpenID Connect)**:
1. GitHub Actions requests a short-lived token from AWS, proving it's running from this specific repo
2. AWS verifies the token and issues temporary credentials (expire in ~1 hour)
3. GitHub Actions uses those credentials to send a deploy command via SSM
4. No credentials are ever stored — not in the repo, not in GitHub settings

The GitHub Actions role (`acquire-online-github-actions-role`) has the minimum permissions needed: it can only run shell commands on this one specific EC2 instance. That's it.

### Deploy Flow (What Happens on `git push`)

```
git push origin master
    → GitHub Actions triggers deploy.yml
    → Assumes AWS role via OIDC
    → Sends SSM command to EC2:
          git pull origin master
          npm ci --production
          pm2 restart acquire-online
    → Workflow reports success or failure
```

You can watch it run live at: https://github.com/esvalC/acquire-online/actions

### Manual Deploy (if needed)

```bash
aws ssm send-command \
  --instance-ids i-0bbc6c13fd3dfe6ab \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "cd /home/ubuntu/acquire-online",
    "sudo -u ubuntu GIT_SSH_COMMAND=\"ssh -i /home/ubuntu/.ssh/deploy_key\" git pull origin master",
    "npm ci --production",
    "sudo -u ubuntu pm2 restart acquire-online"
  ]'
```

---

## Security Model

| Concern | How it's handled |
|---|---|
| SSH brute force | Port 22 is closed. Server managed via SSM only. |
| Leaked AWS credentials | No credentials exist. OIDC uses short-lived tokens. |
| Leaked secrets in code | `.gitignore` covers `.env`. No hardcoded secrets in codebase. |
| Overprivileged CI/CD | GitHub Actions role can only SSM into this one instance. |
| CORS | Locked to `playonlineacquire.com` and `localhost:3000` only. |
| Bot cheating | Bots run server-side with the same validated game engine as human players. |

---

## Roadmap / Issues

Open GitHub Issues track future work:
- **#2** – Settings page (tile confirmation toggle, new player mode hints)
- **#3** – New player mode (contextual hints and suggestions)
- **#4** – AI opponent improvements (already partially implemented in v1.3)
- **#5** – Mobile layout improvements
- **#6** – Public lobby listing (see tables looking for players)
- **#7** – No-scroll card layout (all game cards size to content)

### Data & AI
- Store game turn data for every game to eventually train a stronger AI player
- Schema: per-turn snapshots with board state, player holdings, and action taken

### Resilience & Accounts
- User accounts — persistent identity, game history, stats
- Spectator improvements

### Infrastructure (Future)
- Phase 2: S3/CloudFront for static frontend, EC2 for Socket.IO only
- Phase 3: DynamoDB for game persistence (survive server restarts)
- Phase 4: Serverless with Lambda + API Gateway WebSockets
