#!/usr/bin/env python3
"""
ai/train.py — Train the Acquire Master Bot value network

Reads JSONL training data from selfplay.js and trains a small MLP
that predicts win probability from a game state.

The output (master.onnx) is loaded by ai/masterBot.js at runtime via
onnxruntime-node. At inference time, every legal action is applied to a
cloned game state, the resulting state is encoded + fed through the network,
and the action with the highest predicted win probability is chosen.

Usage:
  pip install torch numpy
  python ai/train.py --data ai/data/games.jsonl --out ai/models/master_weights.json

  # Optional flags:
  --epochs 20        (default: 20)
  --batch  512       (default: 512)
  --hidden 256       (default: 256)

The output is a plain JSON file with weights + biases that masterBot.js
loads directly — no native dependencies required on the server.

Input feature vector (149 floats):
  board[108]     — board cell chain index, divided by 7 (empty=0, lone=-1/7, chain=1–7/7)
  chains[35]     — 7 chains × 5 features: active, size/41, my_shares/25, max_opp_shares/25, price
  myCash[1]      — already normalised to ~0–1 range by selfplay (divided by 6000)
  oppCash[5]     — up to 5 opponents, padded with 0s (divided by 6000)

Output: sigmoid win probability (scalar)
"""

import json, sys, argparse, os, math, random
import numpy as np

# ── Parse args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',    default='ai/data/games.jsonl')
parser.add_argument('--out',     default='ai/models/master_weights.json')
parser.add_argument('--epochs',  type=int, default=20)
parser.add_argument('--batch',   type=int, default=512)
parser.add_argument('--hidden',  type=int, default=256)
parser.add_argument('--val',     type=float, default=0.1, help='validation split fraction')
args = parser.parse_args()

# ── Load data ───────────────────────────────────────────────────────────────
print(f"Loading {args.data} ...")
records = []
with open(args.data) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass

print(f"  {len(records):,} records loaded")
if not records:
    print("No records found. Run selfplay.js with --export first.")
    sys.exit(1)

# ── Flatten state into a fixed-size feature vector ─────────────────────────
INPUT_DIM = 149  # 108 + 35 + 1 + 5

def flatten(record):
    s = record['state']

    # Board: 108 values normalised by 7 (max chain index)
    board = [x / 7.0 for x in s['board']]

    # Chains: 7 × [active, size/41, my_shares/25, max_opp_shares/25, price(already 0-1)]
    chains = []
    for c in s['chains']:
        chains += [
            float(c[0]),          # active 0/1
            c[1] / 41.0,          # size (max board is 108 but chains rarely exceed 41)
            c[2] / 25.0,          # my shares / max 25
            c[3] / 25.0,          # max opp shares
            float(c[4]),          # price (already divided by 1000 in encodeState)
        ]

    # Cash
    my_cash  = [s['myCash']]
    opp_cash = (s['oppCash'] + [0.0] * 5)[:5]

    vec = board + chains + my_cash + opp_cash
    assert len(vec) == INPUT_DIM, f"Expected {INPUT_DIM}, got {len(vec)}"
    return vec

print("Encoding states ...")
X, y = [], []
for r in records:
    try:
        X.append(flatten(r))
        y.append(float(r['outcome']))
    except Exception as e:
        pass  # skip malformed records

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f"  Dataset: {len(X):,} samples, {X.shape[1]} features, {y.mean()*100:.1f}% wins")

# ── Train/val split ─────────────────────────────────────────────────────────
idx = list(range(len(X)))
random.shuffle(idx)
split = int(len(idx) * (1 - args.val))
train_idx, val_idx = idx[:split], idx[split:]
X_tr, y_tr = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
print(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}")

# ── Model definition ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("\nERROR: PyTorch not installed. Run:  pip install torch onnx numpy")
    sys.exit(1)

class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = ValueNet(INPUT_DIM, args.hidden)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()

X_tr_t  = torch.from_numpy(X_tr)
y_tr_t  = torch.from_numpy(y_tr)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)

# ── Training loop ────────────────────────────────────────────────────────────
print(f"\nTraining for {args.epochs} epochs ...")
best_val_loss = math.inf
best_state = None

for epoch in range(1, args.epochs + 1):
    model.train()
    # Mini-batch SGD
    perm = torch.randperm(len(X_tr_t))
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, len(perm), args.batch):
        batch_idx = perm[start:start + args.batch]
        xb, yb = X_tr_t[batch_idx], y_tr_t[batch_idx]
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
        val_preds = (model(X_val_t) > 0.5).float()
        val_acc   = (val_preds == y_val_t).float().mean().item()

    print(f"  Epoch {epoch:2d}/{args.epochs}  train_loss={total_loss/n_batches:.4f}"
          f"  val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

# ── Export weights as JSON (loaded by masterBot.js, no native deps) ─────────
model.load_state_dict(best_state)
model.eval()

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

# Extract weights from the Sequential net (Linear layers only)
layers = []
with torch.no_grad():
    for module in model.net:
        if hasattr(module, 'weight'):
            layers.append({
                'W': module.weight.numpy().tolist(),   # [out, in]
                'b': module.bias.numpy().tolist(),     # [out]
            })

weights_json = {
    'layers':    layers,
    'input_dim': INPUT_DIM,
    'val_loss':  round(best_val_loss, 6),
    'trained_on': len(X),
}

with open(args.out, 'w') as f:
    import json as _json
    _json.dump(weights_json, f, separators=(',', ':'))

size_kb = os.path.getsize(args.out) / 1024
print(f"\nWeights saved → {args.out}  ({size_kb:.0f} KB,  val_loss={best_val_loss:.4f})")
print(f"Commit this file and redeploy — masterBot.js loads it automatically.")
