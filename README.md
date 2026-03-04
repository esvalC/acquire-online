# Acquire – Online Multiplayer

A real-time multiplayer implementation of the classic board game **Acquire** (1964, Sid Sackson).

**Live:** http://3.238.134.173

---

## Quick Start (Local Dev)

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

---

## Application Architecture

```
acquire-online/
├── gameEngine.js          # Pure game logic (no I/O, no networking)
├── server.js              # Express + Socket.IO server
├── public/
│   └── index.html         # Single-file frontend (HTML + CSS + JS)
├── package.json
└── .github/
    └── workflows/
        └── deploy.yml     # CI/CD pipeline (auto-deploy on push)
```

- **Server owns all game state.** Clients send actions, server validates and broadcasts the new state to everyone in the room.
- **Socket.IO rooms** isolate each table — players in room A never see events from room B.
- **Tiles are private per-player.** Other players only see your tile count, not which tiles you hold.
- **In-memory state.** All game data lives in the server process. If the server restarts, active games are lost (persistence is on the roadmap).

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

If you need to deploy without pushing code, use the AWS CLI:

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

### Checking Server Status

```bash
# Check if the app process is running
aws ssm send-command \
  --instance-ids i-0bbc6c13fd3dfe6ab \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo -u ubuntu pm2 list"]'
```

---

## Security Model

| Concern | How it's handled |
|---|---|
| SSH brute force | Port 22 is closed. Server managed via SSM only. |
| Leaked AWS credentials | No credentials exist. OIDC uses short-lived tokens. |
| Leaked secrets in code | `.gitignore` covers `.env`. No hardcoded secrets in codebase. |
| Overprivileged CI/CD | GitHub Actions role can only SSM into this one instance. |
| Public source code | Repo is private until a full security review is done before open-sourcing. |

### Before Going Public (Open Source Checklist)
- [ ] Add SSL/HTTPS via Let's Encrypt (encrypts traffic in transit)
- [ ] Add rate limiting on the server (prevent abuse)
- [ ] Audit `gameEngine.js` for cheating vectors (server already validates all moves)
- [ ] Review all Socket.IO event handlers for injection / unexpected input
- [ ] Enable AWS CloudTrail for audit logging

---

## Roadmap

### Data & AI
- Store game turn data for every game to eventually train an AI player
- Schema: per-turn snapshots with board state, player holdings, and action taken

### Visual Polish
- Smoother animations – tile placements animate into position, chain expansions ripple, mergers collapse
- Card visuals – show tile backs for other players' hands (count visible, tiles hidden)

### Resilience & Accounts
- Reconnection – rejoin with same name/room code after disconnect
- User accounts – persistent identity, game history, stats, ELO
- Spectator improvements

### Infrastructure (Future)
- Phase 2: S3/CloudFront for static frontend, EC2 for Socket.IO only
- Phase 3: DynamoDB for game persistence (survive server restarts)
- Phase 4: Serverless with Lambda + API Gateway WebSockets
