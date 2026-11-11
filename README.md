# Acquire – Online Multiplayer

A real-time multiplayer implementation of the classic board game **Acquire** (1964, Sid Sackson), with solo play against AI bots, user accounts, ELO rankings, and full game replays.

**Live:** https://playonlineacquire.com

---

## Quick Start (Local Dev)

```bash
npm install
cp .env.example .env   # fill in required values (see Environment Variables below)
npm start
```

Open `http://localhost:3000`. Share your local IP (e.g. `http://192.168.1.x:3000`) with others on the same WiFi.

---

## Features

### Game Modes
- **Solo** — play against 1–5 AI bot opponents with distinct personalities
- **Multiplayer** — real-time play with up to 6 players via a 4-digit room code
- **Spectator** — join a game in progress without playing

### In-Game
- Color-coded board with tile placement type indicators (found/expand/merge)
- Live stock market with prices, chain sizes, and remaining shares
- Quickstart draw (closest tile to 1A goes first; tiles stay on the board)
- Vote to End — any player can call a concede vote (requires N−1 of N votes)
- Turn clock — per-player cumulative time tracking
- Player chat and rules assistant in the sidebar
- Reconnect support — rejoin a game in progress if you lose connection

### Accounts & Stats
- Username + password accounts (scrypt-hashed, no email or phone needed); "stay signed in" never expires (ends on sign-out or password change), otherwise 12 hours
- Changing a password signs the account out everywhere; 10 wrong passwords lock an account for 15 minutes
- Admins can issue a one-time temporary password from `/plan` → Users
- Older phone-verified accounts keep working; text-code sign-in switches off automatically once `TWILIO_*` is removed from `.env`
- JWT authentication (stored in localStorage)
- Rating + leaderboard (`/leaderboard`, `ratings.js`) — members only, everyone started at 1000 (Nov 2026 reset). Each game compares you with every other member/bot seat (K=32, averaged): 70% who finished ahead, 30% money margin (capped at $30k). Bots are fixed opponents: Easy 900, Medium 1050, Hard 1200, Master 1400. Finished games (41-tile chain, all safe, no moves) rate normally; games ended early by vote can only lower a rating; quitting (Leave button, back button, or gone 5 min in multiplayer / 24 h in solo) once everyone has had a turn is a loss, and in multiplayer a Medium bot takes the seat
- Career stats kept per member in `user_stats` (added to after every game, independent of game records): finished/multiplayer/solo wins, places, money records, streaks, chains founded, mergers, favorite chains, record vs each bot level, Quick Master and Finance Road, rating history; member-vs-member records in `user_h2h`
- Full game history in your profile with rank, opponents, ELO delta, and date
- Clickable replay button on every game in your history

### Replays
- Every completed game is recorded as a series of snapshots (spectator perspective)
- Browse step-by-step: board state, stock market, player scores, and game log at every move
- Keyboard navigation (← →) and Escape to close

### Admin
- `/plan` dashboard — live stats, recent sessions, feedback inbox, replay viewer
- Feedback system — bug reports and suggestions submitted in-app

### Bot AI
| Difficulty | How it plays | vs 3× the level below (4-player, 25% = even) |
|---|---|---|
| Easy | Hand-tuned heuristic | — |
| Medium | Monte Carlo search, ~40 futures per move | 49% (400 games) |
| Hard | Monte Carlo search, ~200 futures per move | 67.5% (120) |
| Master | Monte Carlo search, 3 s of thinking (thousands of futures) | 46% (120); 97% vs Easy |

Since Nov 2026 each imagined future uses "quick-thinking" tile choices for its first
16 turns (every player picks the tile that most improves their worth if the game ended
now) and is scored directly after 24 turns. New vs previous version, head-to-head:
Master 37.9% (354 games at 3 s), Hard 35%, Medium 34.5%. Details: `ai/BOT_IMPROVEMENT_NOTES.md`.

The search bots (`ai/searchBot.js`) re-deal every tile they can't see before each
simulated game, so they never peek at opponents' hands or the bag. They run in a
worker thread (`ai/botWorker.js`) during the normal bot "thinking" pause.

Benchmark on your own machine: `node ai/benchmark.js` (writes `ai/benchmark-report.md`).
Benchmark on 20 GitHub runners: push any branch named `bench/<name>` (see `.github/workflows/bench.yml`).

All bots run server-side using the same validated game engine as human players.

---

## Application Architecture

```
acquire-online/
├── server.js              # Express + Socket.IO — routes, game rooms
├── ratings.js             # Rating + career stats after each game / quit
├── gameEngine.js          # Pure game logic (stateless, no I/O)
├── botAI.js               # Bot personalities and decision logic
├── auth.js                # JWT sessions, password hashing, legacy Twilio OTP
├── db.js                  # SQLite via better-sqlite3 — all DB access
├── data/
│   └── acquire.db         # SQLite database (gitignored)
├── public/
│   ├── index.html         # Single-file SPA (HTML + CSS + JS)
│   └── plan.html          # Admin dashboard
├── .env                   # Environment variables (gitignored)
├── package.json
└── .github/
    └── workflows/
        ├── deploy.yml     # CI/CD — push to main deploys production
        ├── bench.yml      # bot benchmarks on GitHub runners (bench/** branches)
        └── ops.yml        # run ops/commands.sh on the server (ops/** branches)
```

### Design Principles

- **Server owns all game state.** Clients send actions; server validates and broadcasts new state to the room. Clients cannot cheat.
- **SQLite, single file.** No external database to manage. WAL mode handles concurrent reads from two server processes.
- **Solo games wait for you.** Walk away (tab closed, laptop asleep) and the game is kept for 24 hours; the home page offers a Resume button.
- **Games survive restarts.** In-progress games are saved to SQLite as they're played and reloaded on startup; browsers rejoin automatically.
- **Replays expire.** Replays are stored gzip-compressed and deleted after 5 days.
- **Phone privacy by design.** Raw phone numbers are never written to disk. Only a salted HMAC hash (for uniqueness enforcement) and an AES-256-GCM encrypted value (for masked display) are stored.

### Database Schema

| Table | Purpose |
|---|---|
| `users` | Accounts — username, phone hash, ELO, games played/won |
| `game_history` | Per-user rated game records (logged-in players only) |
| `user_stats` | Rating + career stats per member, updated after every game |
| `user_h2h` | Member-vs-member win/loss records |
| `game_sessions` | Every completed game — standings, metadata, replay snapshots |
| `game_participants` | Junction table `(session_id, user_id, rank, cash)` — indexed on `user_id` for fast per-user lookups |
| `feedback` | In-app bug reports and suggestions |

### URL Structure

| URL | Description |
|---|---|
| `/` | Home — recent games, solo/multiplayer entry |
| `/solo` | Solo game setup |
| `/multiplayer` | Create or join a room |
| `/multiplayer/1234` | Join a specific room (auto-reconnects if session exists) |
| `/rules` | Full standalone rules page |
| `/plan` | Admin dashboard (requires `is_admin` flag) |

---

## Environment Variables

Copy `.env.example` and fill in:

```env
PORT=3000

# JWT signing secret — generate with: openssl rand -hex 32
JWT_SECRET=

# Phone hash salt — generate with: openssl rand -hex 32
PHONE_HASH_SALT=

# AES-256 key for phone encryption (32-byte hex)
PHONE_ENCRYPT_KEY=

# Twilio Verify — LEGACY text-code sign-in. Leave blank to switch it off
# (set ALLOW_DEV_OTP=1 locally to print codes to the console instead).
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_VERIFY_SID=

# SQLite database path (defaults to ./data/acquire.db)
DB_PATH=
```

New accounts use username + password. Text-code sign-in only runs while Twilio is configured (or with `ALLOW_DEV_OTP=1` for local testing).

---

## Infrastructure & Deployment

### Overview

```
git push origin main
      │
      ▼
 GitHub Actions (deploy.yml)
      │  authenticates via OIDC — no credentials stored
      ▼
 AWS IAM (temporary credentials, expire after deploy)
      │  sends command via SSM
      ▼
 EC2 Instance (Ubuntu 22.04, t3.micro)
      │
      ├── Nginx (port 80/443) ← public internet → port 3000
      │
      └── PM2 (user ubuntu): acquire-main (port 3000)
```

The SQLite database lives at `/home/ubuntu/acquire-data/acquire.db`. PM2 runs as the
`ubuntu` user only (`pm2-ubuntu` systemd unit); never run `pm2` as root on the server.

### AWS Resources

| Resource | Name / ID | Purpose |
|---|---|---|
| EC2 Instance | `i-0bbc6c13fd3dfe6ab` | Runs both Node.js servers |
| Security Group | `acquire-online-sg` | Ports 80, 443, 8080. No SSH. |
| S3 Bucket | `acquire-training-data` | Stores AI training data from spot runs |
| IAM Role (EC2) | `acquire-online-ec2-role` | SSM access + S3 write for training runs |
| IAM Role (CI/CD) | `acquire-online-github-actions-role` | What GitHub Actions assumes to deploy |
| OIDC Provider | `token.actions.githubusercontent.com` | Trust bridge between GitHub and AWS |

---

## AI Training

The bot AI can be improved by running self-play games on a cheap EC2 spot instance.

### Launch a training run

```bash
aws ec2 run-instances \
  --image-id ami-0c421724a94bba6d6 \
  --instance-type c5n.xlarge \
  --instance-market-options '{"MarketType":"spot"}' \
  --iam-instance-profile Name=acquire-online-ec2-role \
  --security-group-ids sg-0689315c5c119b6ff \
  --user-data "$(base64 -i ai/ec2_train.sh)" \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=acquire-training}]' \
  --query 'Instances[0].[InstanceId,PublicIpAddress]' \
  --output text
```

- **Cost:** ~$0.10/hr × 5 hours = **~$0.50 per run**
- **Auto-terminates** after 5 hours — no cleanup needed
- Wait ~5 minutes for the instance to boot, then open **`http://<IP>:8080`** to watch the live dashboard

The dashboard shows games played, ELO ratings per bot, training records exported, and time remaining.

### Find a running instance's IP

```bash
aws ec2 describe-instances \
  --filters 'Name=tag:Name,Values=acquire-training' 'Name=instance-state-name,Values=running' \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

### Download results

```bash
aws s3 cp s3://acquire-training-data/latest.jsonl ai/data/games.jsonl
```

### Why No SSH?

Port 22 is the most attacked port on the internet. The server is managed entirely via **AWS Systems Manager (SSM)** over HTTPS. No SSH port is open — there's nothing for scanners to find.

### How OIDC Deployment Works

Instead of storing a long-lived AWS secret key in GitHub:

1. GitHub Actions requests a short-lived token from AWS, proving it's running from this specific repo
2. AWS verifies the token and issues temporary credentials (expire in ~1 hour)
3. GitHub Actions sends a deploy command to the EC2 instance via SSM
4. No credentials are ever stored — not in the repo, not in GitHub settings

The CI/CD role has minimum permissions: it can only run shell commands on this one EC2 instance.

### Manual Deploy

```bash
aws ssm send-command \
  --instance-ids i-0bbc6c13fd3dfe6ab \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "git -C /home/ubuntu/acquire-main -c safe.directory=/home/ubuntu/acquire-main pull origin main",
    "chown -R ubuntu:ubuntu /home/ubuntu/acquire-main",
    "sudo -u ubuntu npm --prefix /home/ubuntu/acquire-main install --omit=dev",
    "sudo -u ubuntu pm2 restart acquire-main"
  ]'
```

---

## Security Model

| Concern | How it's handled |
|---|---|
| SSH brute force | Port 22 closed. Server managed via SSM only. |
| Leaked AWS credentials | No credentials exist. OIDC uses short-lived tokens. |
| Leaked secrets | `.gitignore` covers `.env`. No hardcoded secrets in codebase. |
| Overprivileged CI/CD | GitHub Actions role can only SSM into this one instance. |
| Phone number privacy | Never stored raw. HMAC hash for uniqueness + AES-256-GCM for masked display. |
| Game cheating | All game logic runs server-side. Clients send actions, server validates. |
| Bot cheating | Bots use the same validated engine as humans, no lookahead into hidden state. |
| CORS | Locked to `playonlineacquire.com` and `localhost` only. |

---

## Open Issues

See [GitHub Issues](https://github.com/esvalC/acquire-online/issues) for current work. Key open items:

- **#2** – Settings page (tile confirmation toggle, new player hints)
- **#3** – New player mode (contextual in-game guidance)
- **#5** – Mobile layout improvements
- **#6** – Public lobby listing (browse open tables)
- **#10** – Bot AI improvement via self-play and heuristic tuning

---

## License

This project is a fan implementation for personal and educational use. **Acquire** is a trademark of Hasbro. This project is not affiliated with or endorsed by Hasbro.

## Database backups

`scripts/backup-db.sh` runs nightly from the server's crontab (09:17 UTC) and uploads an
integrity-checked, gzip-compressed copy of the SQLite database to the private S3 bucket:

- `s3://acquire-training-data/backups/daily/acquire-<weekday>.db.gz` — last 7 days (overwritten weekly)
- `s3://acquire-training-data/backups/monthly/acquire-<YYYY-MM>.db.gz` — one per month

Log: `/home/ubuntu/acquire-data/backup.log`. To restore (stop the app first):

```bash
aws s3 cp s3://acquire-training-data/backups/daily/acquire-wed.db.gz - | gunzip > /home/ubuntu/acquire-data/acquire.db
```
