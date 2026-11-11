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

---

## Nov 10–11, 2026 — Master search round (overnight)

How the search bots work: for each decision they list candidate moves, then repeatedly
re-deal every unseen tile at random (no peeking), apply a candidate and play the rest of the
game out quickly, and pick the candidate with the best average result (UCB1 over candidates).
Medium/Hard/Master differ only in budget (40 / 200 / up to 16,000 futures; 0.15 / 0.4 / 3 s).

All tests: 4-player games on GitHub runners, one hero vs 3 copies of the baseline, seats
rotated, 200 games unless noted. 25% = equal strength. Ideas compared at 1 s/move (fast),
winner confirmed at 3 s.

### Speed (shipped — identical decisions, verified on 25 seeded full games)
- `botAI.decideBotAction` called `require('./ai/searchBot')` on every simulated move:
  ~36% of search time was module-path lookups. Now cached.
- `maybeDeclareEnd` copied the whole game (JSON) to compute final standings each turn of
  every future. Now `engine.worthIfEndedNow` computes the same totals directly.
- `searchBot.clone` hand-written for the game's shape instead of JSON round-trip.
- `analyzeTilePlacement` (hottest function) without Sets/destructuring.
- Net: ~830 → ~950+ futures/sec on a 4-player mid-game position with quick-thinking, and
  ~940 → ~1,275 for plain full playouts (≈1.35–1.6× more futures in the same 3 s).

### Ideas tested vs today's Master (1 s each side)
| Variant | Win % | Verdict |
|---|---|---|
| Futures cut at 20 turns, scored by worth-if-ended-now | 20% | worse alone |
| Cut at 20 + quick-thinking tiles for 2 rounds (Calvin's idea) | **34%** | big gain |
| Cut at 20 + quick-thinking 1 round | 30.5% | gain |
| Quick-thinking 2 rounds, full-length futures | 35% | gain |
| Quick-thinking 4 rounds, full-length futures | 33% | gain |
| **Quick-thinking 4 rounds, cut at 24** | **38%** | best |
| Quick-thinking 2 rounds + wider buy candidates | 37% | gain (buys ≈ neutral) |

Relative to "cut at 20" (round 2): wider buys 24.5% (neutral), horizon 12 21% (worse),
horizon 32 30.5% (better), sharper eval scale 3000 21% (worse). Early stopping when one move
is clearly ahead saved only ~10% time at 3 s → not used.

"Quick-thinking" = in the first rounds of each imagined future, every player picks the tile
that most improves their worth-if-the-game-ended-now relative to the leader (one step ahead,
no nested search), instead of the Easy heuristic's fixed rules. Nested real Hard bots inside
futures would be ~10,000× too slow; this captures most of the realism cheaply.

### Round 4 and 3-second confirmations
| Variant | Budget | Opponents | Result |
|---|---|---|---|
| Quick-thinking tiles AND buys (full futures) | 1 s | old Master 1 s | 30.5% (buys hurt → dropped) |
| Look 32 + quick-thinking 2 rounds | 1 s | old Master 1 s | 43% (200) |
| Quick-thinking 3 rounds, full futures | 1 s | old Master 1 s | 34.5% |
| Look 40 + quick-thinking 2 rounds | 1 s | old Master 1 s | 38.3% (120) |
| Look 32 + quick-thinking 4 rounds | 1 s | old Master 1 s | 34.2% (120) |
| **Look 24 + quick-thinking 4 rounds** | **3 s** | **live Master 3 s (with speedups)** | **38.6% (114)** |
| Quick-thinking 2 rounds, full futures | 3 s | live Master 3 s | 33.3% (114) |
| Look 32 + quick-thinking 2 rounds | 3 s | live Master 3 s | 35.8% (120) |
| Medium + look 24/quick 4 rounds | 0.15 s | old Medium | 34.5% (200) |
| Hard + look 24/quick 4 rounds | 0.4 s | old Hard | 35% (200) |

**Shipped (Nov 11, 05:19 UTC):** Medium, Hard and Master all use `horizon: 24, greedyTurns: 16`
(`QUICK` in `ai/searchBot.js`). Thinking budgets unchanged (Master 3 s cap).
Look-24 and look-32 were tied within noise at 3 s; look-24 was picked because it was
strong at both 1 s and 3 s and is the same setting that lifted Medium and Hard.

Benchmark tip: `bench/**` branches run on 20 GitHub shards; artifact downloads may be
blocked from some environments — the per-shard job logs contain the same tables.

### After shipping
Ladder with the new levels (120 games each unless noted):
| Matchup | Before | After |
|---|---|---|
| Medium vs 3× Easy | 56% (200) | 49.2% (400; old Medium re-measured 55.5% on 400) |
| Hard vs 3× Medium | 56% | 67.5% |
| Master vs 3× Hard | 35% (1.5 s Master) | 45.8% |
| Master vs 3× Easy | 94% | 96.7% |

New Medium is stronger than old Medium head-to-head (34.5%) but slightly weaker against a
table of Easy bots (49% vs 55.5%): its futures now assume thoughtful opponents. Kept, since
human players aren't Easy bots; revert = drop `...QUICK` from `expert` in SEARCH_LEVELS.

Tried and rejected after shipping (vs shipped config at 1 s, 200 games):
softer quick-thinking (softmax temp 1500 / 4000): 27% / 25.5%; calmer cut-off scoring
(evalScale 10000): 22.5%; quick-thinking for the whole 24-turn future: 21%.

Ideas not yet tried: tune the Easy heuristic's constants for use inside futures only
(previous attempt with the 'expert' heuristic as playout policy showed no gain); a learned
evaluation for cut-off positions instead of worth-if-ended-now.

**Final confirmation (06:28 UTC):** live Master vs 3× previous Master (both 3 s, both with
speedups): 90/240 = 37.5% (95%: 31–44%). With the earlier 44/114 → **134/354 = 37.9%**.
