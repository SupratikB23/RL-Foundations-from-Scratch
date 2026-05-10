# Chapter 01b — Temporal Difference Q-Learning

> Learn from every single step — don't wait for the episode to end.

---

## Core Idea

Temporal Difference (TD) methods update Q-values **after every step**, not at the end of an episode. Instead of using the actual return (like Monte Carlo), TD uses a **bootstrapped estimate** — it uses the current Q-table to estimate what the future is worth.

The update rule (Q-learning / TD-0):

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a')  −  Q(s, a)]
                          └─────── TD Target ──────┘  └─ Current Q ─┘
```

The term in brackets is the **TD error** — how wrong our current estimate was.

---

## Algorithm

```
Initialize Q(s, a) = 0 for all (s, a)

For each episode:
    state ← reset environment
    
    For each step:
        1. Choose action using ε-greedy policy
        2. Take action → observe (next_state, reward, done)
        3. Compute TD target:
           if done:  target = reward
           else:     target = reward + γ · max Q(next_state)
        4. Update: Q(s, a) += α · (target − Q(s, a))
        5. state ← next_state
    
    Decay epsilon
```

---

## State Discretization

Same approach as Monte Carlo — 65,536 discrete states:

```python
NUM_BINS = (8, 8, 8, 8, 8, 8, 2, 2)
```

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `GAMMA` | 0.99 | Discount factor |
| `ALPHA` | 0.1 | Learning rate — how much to shift Q toward new estimate |
| `EPSILON` | 1.0 → 0.05 | Exploration, decays by 0.9995 per episode |
| `NUM_EPISODES` | 50,000 | Total training episodes |

> **Why does TD need more episodes than MC (50K vs 30K)?**
> Each TD update only moves Q a small step (controlled by α). MC averages all returns, so early updates carry more information. TD trades accuracy per update for speed per step.

---

## How to Run

```bash
cd 01_tabular_methods/temporal_difference
python temporal_difference.py
```

**Expected output:**
```
Starting LunarLander Q-Learning training...
State space size: 65,536
Episode  1000/50000 - Avg Reward: -210.3
Episode  5000/50000 - Avg Reward: -120.5
...
=== RESULTS ===
Training Episodes: 50,000
Average Test Reward: XXX
Successful Landings: XX/50 (XX.X%)
```

---

## Monitor with TensorBoard

```bash
tensorboard --logdir runs/lunarlander_td_optimized
```

Metrics logged:
- `Train/EpisodeReward`
- `Train/MovingAverageReward`
- `Train/Epsilon`
- `Test/AverageReward`
- `Test/SuccessRate`

---

## Results

Videos of 5 test episodes are saved to [results/](./results/) after training.

---

## Monte Carlo vs TD — Side-by-Side

| Property | Monte Carlo | TD Q-Learning |
|----------|-------------|---------------|
| Updates | End of episode | Every step |
| Uses | Actual return `G` | Bootstrapped `r + γ·maxQ` |
| Needs learning rate | No | Yes (α = 0.1) |
| Variance | High | Low |
| Bias | Zero | Small (bootstrap bias) |
| Works on continuing tasks | No | Yes |
| Episodes to converge | 30,000 | 50,000 |

---

## Strengths and Weaknesses

**Strengths:**
- Online learning — learns while the episode runs
- Lower variance than Monte Carlo
- Works for non-episodic tasks
- More sample efficient per time-step

**Weaknesses:**
- Biased — the TD target uses an imperfect Q estimate
- Requires careful tuning of learning rate α
- Still limited by the Q-table (can't handle raw images)

---

## Next Step

Both tabular methods break down when the state space explodes (e.g., raw Atari pixels = millions of possible states). The solution is to replace the Q-table with a neural network.

→ [Chapter 02 — Deep Q-Network](../../02_deep_rl/)
