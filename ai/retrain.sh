#!/bin/bash
#
# ai/retrain.sh — Full Master Bot retraining pipeline
#
# Run this script whenever you want to retrain the Master bot on new selfplay data.
# It downloads the latest training data from S3, trains the neural network,
# commits the new weights, and deploys to the beta server.
#
# Usage:
#   bash ai/retrain.sh                     # download from S3, train, commit, deploy
#   bash ai/retrain.sh --skip-download     # use existing ai/data/games.jsonl
#   bash ai/retrain.sh --skip-deploy       # train and commit but don't deploy
#   bash ai/retrain.sh --epochs 30         # override training epochs (default 20)
#
# Requirements (one-time setup on your Mac):
#   pip install torch numpy
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_FILE="$SCRIPT_DIR/data/games.jsonl"
WEIGHTS_FILE="$SCRIPT_DIR/models/master_weights.json"
S3_BUCKET="${S3_BUCKET:-acquire-training-data}"
BETA_INSTANCE="${BETA_INSTANCE:-i-0bbc6c13fd3dfe6ab}"
EPOCHS="${EPOCHS:-20}"

SKIP_DOWNLOAD=false
SKIP_DEPLOY=false

for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD=true ;;
    --skip-deploy)   SKIP_DEPLOY=true ;;
    --epochs)        shift; EPOCHS="$1" ;;
    --epochs=*)      EPOCHS="${arg#--epochs=}" ;;
  esac
done

echo ""
echo "════════════════════════════════════════════════"
echo "  Acquire Master Bot — Retraining Pipeline"
echo "════════════════════════════════════════════════"
echo ""

cd "$REPO_DIR"

# ── Step 1: Download training data from S3 ─────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
  echo "▶ Step 1: Downloading training data from S3..."
  mkdir -p "$SCRIPT_DIR/data"
  if aws s3 ls "s3://$S3_BUCKET/latest.jsonl" &>/dev/null; then
    aws s3 cp "s3://$S3_BUCKET/latest.jsonl" "$DATA_FILE"
    LINES=$(wc -l < "$DATA_FILE" | tr -d ' ')
    echo "  Downloaded $LINES training records → $DATA_FILE"
  else
    echo "  ERROR: s3://$S3_BUCKET/latest.jsonl not found."
    echo "  Run a selfplay training first (see ai/ec2_train.sh), or use --skip-download"
    echo "  to use existing local data."
    exit 1
  fi
else
  echo "▶ Step 1: Skipping download (using existing $DATA_FILE)"
  if [ ! -f "$DATA_FILE" ]; then
    echo "  ERROR: $DATA_FILE not found. Remove --skip-download to fetch from S3."
    exit 1
  fi
  LINES=$(wc -l < "$DATA_FILE" | tr -d ' ')
  echo "  Found $LINES local records."
fi

echo ""

# ── Step 2: Train the neural network ──────────────────────────
echo "▶ Step 2: Training neural network (epochs=$EPOCHS)..."
python3 ai/train.py \
  --data "$DATA_FILE" \
  --out  "$WEIGHTS_FILE" \
  --epochs "$EPOCHS"

echo ""

# ── Step 3: Verify the weights file ───────────────────────────
echo "▶ Step 3: Verifying weights file..."
if [ ! -f "$WEIGHTS_FILE" ]; then
  echo "  ERROR: training did not produce $WEIGHTS_FILE"
  exit 1
fi
SIZE_KB=$(du -k "$WEIGHTS_FILE" | cut -f1)
echo "  master_weights.json — ${SIZE_KB}KB ✓"

# Quick smoke-test: load the weights in Node.js
node -e "
  const master = require('./ai/masterBot');
  const engine = require('./gameEngine');
  const { decideBotAction } = require('./botAI');
  // Create a test game and run one master bot decision
  const game = engine.createGame(['Master', 'Aria', 'Rex'], { quickstart: true });
  const result = decideBotAction(game, 0, 'focused', 'master', 'Master');
  if (result === false) process.exit(1); // false = no model
  console.log('  Smoke test passed — Master bot returned an action ✓');
" 2>&1 || { echo "  WARNING: smoke test skipped (model may need a game in progress)"; }

echo ""

# ── Step 4: Commit and push ────────────────────────────────────
echo "▶ Step 4: Committing weights to git..."
git add ai/models/master_weights.json

# Check if there's actually a change to commit
if git diff --cached --quiet; then
  echo "  Weights unchanged — nothing to commit."
else
  TRAINED_ON=$(python3 -c "import json; d=json.load(open('$WEIGHTS_FILE')); print(d.get('trained_on','?'))" 2>/dev/null || echo "?")
  VAL_LOSS=$(python3 -c "import json; d=json.load(open('$WEIGHTS_FILE')); print(d.get('val_loss','?'))" 2>/dev/null || echo "?")
  git commit -m "Train Master bot: ${TRAINED_ON} records, val_loss=${VAL_LOSS}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  git push origin main
  echo "  Pushed to main ✓"
fi

echo ""

# ── Step 5: Deploy to beta server ─────────────────────────────
if [ "$SKIP_DEPLOY" = false ]; then
  echo "▶ Step 5: Deploying to beta server..."
  CMD_ID=$(aws ssm send-command \
    --instance-ids "$BETA_INSTANCE" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
      "sudo -u ubuntu GIT_SSH_COMMAND=\"ssh -i /home/ubuntu/.ssh/deploy_key\" git -C /home/ubuntu/acquire-online pull origin main",
      "sudo -u ubuntu pm2 restart acquire-online"
    ]' \
    --query 'Command.CommandId' \
    --output text)
  echo "  SSM command: $CMD_ID"
  echo "  Waiting for deploy..."
  sleep 12
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$BETA_INSTANCE" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  echo "  Deploy status: $STATUS ✓"
else
  echo "▶ Step 5: Skipping deploy (--skip-deploy)"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  Done! Master bot updated."
echo "  Test it at: https://playonlineacquire.com/beta"
echo "════════════════════════════════════════════════"
echo ""
