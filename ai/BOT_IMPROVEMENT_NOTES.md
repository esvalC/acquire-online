# Bot Improvement Notes
*Written 2026-03-26 — session analysis comparing our system to AlphaZero-General*

---

## Current System Architecture

### Files
- `ai/masterBot.js` — Primary inference engine. Loads v3 weights, does value-guided 1-step lookahead for ALL phases (tile placement, buy stock, merger decisions, chain choices)
- `ai/mcts.js` — Flat Monte Carlo search. Only used when difficulty = `'mcts'`. NOT used by masterBot.
- `ai/selfplay.js` — Headless bot-vs-bot simulator. Tracks win%, ELO, avg cash, top-3%. Also exports JSONL training data.
- `ai/train.py` — Trains a single-value sigmoid network (INPUT_DIM=149). **Mismatch: outputs `{layers}` format, but masterBot expects v3 `{body, policyHead, valueHead}` format.**
- `ai/models/master_weights.json` — Current weights, v3 format (body + policyHead + valueHead), INPUT_DIM=258. **NOT produced by train.py** — produced by a separate training script not in the repo.

### Feature Encoding Mismatch
| | INPUT_DIM | Includes tile hand? |
|--|-----------|---------------------|
| `train.py` | 149 | No |
| `masterBot.js` (v3) | 258 | Yes (108 extra features) |
| `selfplay.js` | 149 | No |

The current v3 weights expect 258 features. If you ran `train.py`, the output would be incompatible with `masterBot.js`. The v3 weights are working but were produced outside this repo.

---

## AlphaZero-General vs Our System

| Aspect | AlphaZero-General | Our System |
|--------|-------------------|------------|
| Network shape | policy head + value head | Same (body + policyHead + valueHead) ✅ |
| Search type | UCT tree search | 1-step value lookahead (masterBot) or flat MC (mcts.js) |
| Search evaluator | Value network + policy priors | masterBot: value network ✅ / mcts.js: random rollouts ❌ |
| Training source | Self-play from neural net | Supervised from hard-bot games |
| Policy head usage | Guides MCTS (action priors) | Not used — code loads it but never uses policy logits |

### The Good News
- masterBot.js already does value-guided lookahead, which is the right approach
- v3 weight format is conceptually identical to AlphaZero dual-head architecture
- Network body is shared across both heads (correct)

### The Main Gap: Search Depth
masterBot does **1-step** lookahead. AlphaZero searches **many moves ahead** using UCT + value network.

mcts.js gap: uses random heuristic rollouts to game end to evaluate positions. These are noisy (a lot can happen in 100+ turns). The value network would give a much cleaner signal and allow 10x more simulations in the same time budget.

---

## Improvements Possible WITHOUT Retraining

### 1. Replace random rollouts with value-network eval in `mcts.js` (HIGH VALUE, ~10 lines)
Currently:
```
For each tile → run 5 random games to end → pick tile with most wins
```
Should be:
```
For each tile → ask value network "how good is this state?" → pick best score
```
Change `rollout()` in mcts.js to call `decideMasterAction`'s value forward pass instead of playing to game end. Would allow SIMS_PER_ACTION to jump from 5 → 50+ with same time budget.

### 2. Use policy head logits as action priors in MCTS (MEDIUM VALUE)
Right now mcts.js tries all tiles equally. The policy head already has opinions on which tiles are good. Could weight tile sampling by softmax(policyLogits) so promising tiles get more simulation budget. This is the core AlphaZero trick.

### 3. Deeper value lookahead in masterBot (MEDIUM VALUE)
Instead of: `place tile → evaluate state → pick best`
Do: `place tile → opponent plays best response → evaluate state → pick best`
2-ply search. More expensive but much stronger signal. Could be gated behind the existing bot delay.

### 4. Increase SIMS_PER_ACTION (easy, after #1)
After replacing rollouts with net eval, can safely go 5 → 50+.

---

## Training Improvements (WOULD require retraining / new data)

- **Policy head training**: Need to record *which move won* as a label, not just *did the player win*. Current selfplay.js doesn't capture the action taken, only the outcome.
- **Self-play from the neural net**: Generate data from masterBot playing itself (not hard-bots). Iteratively improve.
- **Fix feature mismatch**: selfplay.js needs to use the 258-feature encoder (incl. tile hand) to generate data compatible with train.py. Or update train.py to use INPUT_DIM=258.

---

## Better Training Metrics to Add to Dashboard

Current: win%, top-3%, avg cash, ELO

Proposed additions to `selfplay.js`:

| Metric | What it reveals |
|--------|----------------|
| **Avg merger bonus / game** | Core skill signal — are they holding majority/minority when mergers happen? |
| **Chains founded / game** | Aggressive early-chain positioning |
| **Avg majority stock positions at game end** | Chain dominance / investment strategy |
| **Avg portfolio value (stocks × price) at end** | Total wealth, not just cash in hand |
| **Avg cash remaining at end** | Are they spending efficiently or hoarding? |

**Best single metric to add**: avg merger bonus per game. Merger bonuses are the primary way money transfers between players — if a bot wins more mergers, it's demonstrably playing better strategy, regardless of overall win/loss.

---

## Quick Action Items (Ranked)

1. `mcts.js`: swap random rollouts for value-network eval → instant strength boost, no retraining
2. `mcts.js`: use policy head as action prior → more focused search
3. `selfplay.js` + dashboard: add merger bonus and portfolio value metrics
4. Fix the train.py / masterBot.js feature dimension mismatch before running any new training
5. Long-term: generate policy targets in selfplay.js (record the chosen action per turn) to properly train policy head

---

*See also: the conversation transcript at `/Users/calvi/.claude/projects/-Users-calvi-Desktop-Projects-acquire-online-0-3/9c34876e-cf3b-4455-8f90-dcf17abdd4d6.jsonl`*
