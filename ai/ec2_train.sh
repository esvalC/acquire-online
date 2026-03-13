#!/bin/bash
# ai/ec2_train.sh
#
# Self-play training script designed to run on a Linux machine or EC2 spot instance.
# Runs for exactly 5 hours then exits cleanly, saving all training data to S3.
#
# ── How to run ────────────────────────────────────────────────
#
# Option 1: On your Mac (local training, no AWS needed)
#   bash ai/ec2_train.sh
#
# Option 2: Launch a spot instance and run automatically (see LAUNCH COMMAND below)
#   This costs ~$0.50 for 5 hours on a c7g.2xlarge spot instance.
#
# ── What it does ──────────────────────────────────────────────
#   1. Runs ai/selfplay.js for 5 hours generating game records
#   2. Uploads the JSONL file to S3 (if S3_BUCKET is set)
#   3. Shuts down the EC2 instance automatically (if running on EC2)
#
# ── One-time setup ────────────────────────────────────────────
#   aws s3 mb s3://acquire-training-data   # create S3 bucket once
#
# ── EC2 spot instance launch command ──────────────────────────
# Run this from your Mac to launch a training instance:
#
#   aws ec2 run-instances \
#     --image-id ami-0c55b159cbfafe1f0 \
#     --instance-type c5n.xlarge \
#     --instance-market-options '{"MarketType":"spot"}' \
#     --iam-instance-profile Name=acquire-online-ec2-role \
#     --security-group-ids acquire-online-sg \
#     --user-data "$(base64 -i ai/ec2_train.sh)" \
#     --instance-initiated-shutdown-behavior terminate \
#     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=acquire-training}]'
#
# The instance will:
#   - Boot, install Node, clone repo, run training for 5 hours
#   - Upload results to S3
#   - Terminate itself (you're only billed for what runs)
#
# ── Download results ───────────────────────────────────────────
#   aws s3 cp s3://acquire-training-data/latest.jsonl ai/data/games.jsonl
#
set -e

TIME_LIMIT_HOURS=10
TIME_LIMIT_SECS=$((TIME_LIMIT_HOURS * 3600))
S3_BUCKET="${S3_BUCKET:-acquire-training-data}"
REPO_URL="${REPO_URL:-https://github.com/esvalC/acquire-online.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-/tmp/acquire-training}"
OUTFILE="$WORKDIR/games_$(date +%Y%m%d_%H%M%S).jsonl"
STATSFILE="$WORKDIR/stats.json"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

echo "=============================================="
echo "  Acquire Self-Play Training"
echo "  Time limit: ${TIME_LIMIT_HOURS}h"
echo "  Output: $OUTFILE"
echo "  S3: s3://$S3_BUCKET/"
echo "=============================================="

# ── Install dependencies (EC2 Amazon Linux / Ubuntu) ──────────
if command -v apt-get &>/dev/null; then
  # Ubuntu
  if ! command -v node &>/dev/null; then
    echo "[setup] Installing Node.js (Ubuntu)..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
  fi
  if ! command -v git &>/dev/null; then
    sudo apt-get install -y git
  fi
else
  # Amazon Linux / RHEL
  if ! command -v git &>/dev/null; then
    echo "[setup] Installing git..."
    sudo yum install -y git
  fi
  if ! command -v node &>/dev/null; then
    echo "[setup] Installing Node.js (Amazon Linux)..."
    curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
    sudo yum install -y nodejs
  fi
fi

# ── Clone or update the repo ───────────────────────────────────
mkdir -p "$WORKDIR"
if [ ! -d "$WORKDIR/repo" ]; then
  echo "[setup] Cloning repo..."
  git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR/repo"
else
  echo "[setup] Updating repo..."
  git -C "$WORKDIR/repo" pull origin "$REPO_BRANCH"
fi

cd "$WORKDIR/repo"
npm install --omit=dev

mkdir -p ai/data

# ── Start live dashboard on port 8080 ─────────────────────────
echo "[dashboard] Starting live dashboard on :$DASHBOARD_PORT ..."
node ai/dashboard.js --stats "$STATSFILE" &
DASHBOARD_PID=$!

# ── Run self-play ──────────────────────────────────────────────
echo "[train] Starting self-play for ${TIME_LIMIT_HOURS}h..."
PUBLIC_IP=$(curl -sf --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 || echo "localhost")
echo "  Dashboard: http://$PUBLIC_IP:$DASHBOARD_PORT"
node ai/selfplay.js \
  --time-limit "$TIME_LIMIT_SECS" \
  --export "$OUTFILE" \
  --stats "$STATSFILE" \
  2>&1 | tee "$WORKDIR/train.log"

kill $DASHBOARD_PID 2>/dev/null || true

echo "[train] Done. Records saved to $OUTFILE"
wc -l "$OUTFILE"

# ── Upload to S3 (if bucket is accessible) ────────────────────
if command -v aws &>/dev/null; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  echo "[upload] Uploading to s3://$S3_BUCKET/$TIMESTAMP.jsonl ..."
  aws s3 cp "$OUTFILE" "s3://$S3_BUCKET/$TIMESTAMP.jsonl" && echo "[upload] Done."
  aws s3 cp "$OUTFILE" "s3://$S3_BUCKET/latest.jsonl" && echo "[upload] Updated latest."
else
  echo "[upload] aws CLI not found — skipping S3 upload."
  echo "         File saved at: $OUTFILE"
fi

# ── Shut down EC2 instance if running on EC2 ──────────────────
# The instance-initiated-shutdown-behavior=terminate flag (set at launch)
# means the instance terminates (not just stops) when it shuts down.
if curl -sf --max-time 2 http://169.254.169.254/latest/meta-data/instance-id &>/dev/null; then
  echo "[shutdown] Shutting down EC2 instance in 60s... (ctrl-C to cancel)"
  sleep 60
  sudo shutdown -h now
fi

echo "All done."
